# Adaptive Bayesian Partner Selection for Federated Clinical Centers

Navid Seidi Department of Computer Science Missouri University of Science and Technology Rolla, MO, USA nseidi@mst.edu

Satyaki Roy   
Department of Mathematical Sciences   
University of Alabama in Huntsville Huntsville, AL, USA sr0215@uah.edu

Sajal K. Das Department of Computer Science Missouri University of Science and Technology Rolla, MO, USA sdas@mst.edu

## Abstract

Federated learning (FL) in healthcare is challenged by pronounced heterogeneity and temporal concept drift across clinical centers, where evolving patient populations and care practices shift data distributions. Existing approaches rely on persistent global communication, incurring substantial bandwidth overhead while risking negative transfer from poorly aligned peers. We address this by proposing ADAPTIVE BAYESIAN PARTNER SELECTION (ABPS), a peer-to-peer framework that governs who collaborates, when, and at what cost. Each center maintains a Beta–Bernoulli posterior over prospective peers’ Shapley marginal utility, ranks candidates using an Upper Confidence Bound (UCB) criterion, and forms collaborations through a lightweight propose–reject mechanism, with the option to abstain from communication when mutually beneficial interaction fails. It admits a stochastic decision interpretation, yielding finite-sample concentration guarantees and O(κ log T) regret in partner selection, along with conditions under which intentional isolation is optimal under negative transfer. Lightweight extensions, head personalization, bfloat16 quantized communication, and tunable active-set cardinality further improve efficiency, while a goal-aware metadata filter enables institution-specific collaboration strategies. Evaluations on binary in-hospital mortality prediction over the first 24 hours of an ICU stay, with 230 non-IID clinical centers drawn from MIMIC-IV, show that the full ABPS-X variant matches the strongest federated baseline (FedDyn, AUROC 0.758) at 0.09× the communication cost of FedAvg, with reduced variability. A diversity-driven configuration activates intentional isolation for a substantial fraction of centers, highlighting the role of selective collaboration under heterogeneity. These results show that adaptive, utility-aware collaboration reduces communication without sacrificing accuracy when centers are numerous and small, providing a scalable paradigm for real-world healthcare FL systems.

## 1 Introduction

Federated Learning (FL) has emerged as the dominant paradigm for training predictive models across decentralized centers without exposing sensitive patient data McMahan et al. [2017], sidestepping Health Insurance Portability and Accountability Act (HIPAA)-style data-sharing constraints. Realizing FL in clinical settings is nevertheless gated by two persistent obstacles: statistical heterogeneity (i.e., non-IID biomedical data) across institutions Hsu et al. [2019], and concept drift as patient demographics and clinical protocols evolve Gama et al. [2014], Rahimli et al. [2024] (see Appendix A for an instance of distributional drift across admission eras in routinely charted vital signs).

This heterogeneity is structural rather than incidental. The healthcare ecosystem comprises three institution types with divergent key performance indicators (KPIs) and data profiles. Integrated delivery networks (Kaiser Permanente, HCA Healthcare) aggregate large but internally siloed cohorts. Academic medical centers see complex, rare, and trial-driven cases at low volume. Community hospitals generate the majority of high-volume routine encounters. No single ecosystem captures the full distribution required for generalizable modeling, yet forcing central FL aggregation across them frequently induces negative transfer that degrades local performance. Peer-to-peer (P2P) alternatives Guha Roy et al. [2019], Heged˝us et al. [2021] avoid the single point of failure but still face the question of which peers a center should collaborate with and at what bandwidth cost.

Contributions. To achieve predictive accuracy under concept drift, we introduce ADAPTIVE BAYESIAN PARTNER SELECTION (ABPS), a serverless P2P federated learning framework in which each center adaptively selects collaborators based on a Bayesian estimate of their utility. ABPS models each candidate peer’s Shapley marginal contribution via a Beta–Bernoulli posterior, ranks peers using an ϵ-greedy UCB policy, and employs a propose–reject protocol with an explicit rest action to mitigate negative transfer. We establish three guarantees: posterior concentration (Theorem 1), sublinear regret of order O(κ log T) (Theorem 2), and the Bayes-optimality of isolation (Lemma 1). The core framework comprises head personalization, bfloat16 Kalamkar et al. [2019] quantized communication, and a tunable active-set cardinality κ. In addition, a goal-aware metadata pre-filter allows each center to prioritize homogeneity, diversity, or KPI alignment objectives. We denote the maximally-extended configuration as ABPS-X, which composes all of the above with deeper 2-layer head personalization, server-side exponential moving average (EMA) momentum, and validation-AUROC early stopping (full hyperparameters in Sec. 6.4). On MIMIC-IV v3.1, with n=230 non-IID careunit-by-year centers, ABPS-X matches the strongest federated baseline (FedDyn) while using only 0.09× the bandwidth of FedAvg. In contrast, applying the same extensions to FedAvg degrades performance by 6.4 AUROC points, leaving ABPS-X ahead of FedAvg-X by 6.7 points (Table 1), thereby isolating the gains to Bayesian partner-selection.

## 2 Related Work

Our research intersects with several active domains in decentralized machine learning. We organize the related work below in the order in which the corresponding open Question is answered in the body of the paper, ranging from divergence-aware partner selection to peer-to-peer (P2P) federated architectures, concept-drift adaptation, and communication-efficient FL.

## 2.1 Divergence Metrics for Non-IID Collaboration

Information-theoretic divergences (Kullback-Leibler, hereafter KL, and Wasserstein) quantify interclient distributional disparity in FL Németh et al. [2025]. They appear as regularization penalties that align local models without raw data exposure Németh et al. [2025], as client-selection utilities that favor either stabilizing or diversity-injecting clients Düsing and Cimiano [2026], Rahad et al. [2025], and as theoretical tools for bounding FL generalization under heterogeneity. This philosophy extends to the wire-format level: each center can be summarized by a non-PHI descriptor of its distribution (e.g., positive-class rate, training-set size, optional Charlson-comorbidity prevalence), and peers can be admitted via a goal-dependent monotone function of pairwise similarity. A key limitation is that divergence is treated as a single scalar objective fixed network-wide, without allowing individual centers to declare their own collaboration goal before model exchange. Yet healthcare institution types (Integrated Delivery Networks, Academic Medical Centers, Community Hospitals) require different collaboration objectives, including homogeneity, diversity, or KPI alignment Li et al. [2025]. We label this open problem Q1 (goal-heterogeneous collaboration) and address it through the goal-aware metadata pre-filter of Sec. 4.1, where a center selects one of three admission rules, namely, cosine, anti-cosine, and KPI matching, at runtime without risking exposing model parameters.

## 2.2 P2P Federated Learning and Partner Selection

Peer-to-peer (P2P) FL decentralizes aggregation, eliminating single points of failure and reducing communication bottlenecks Heged˝us et al. [2021], Zhou et al. [2024]. However, identifying the appropriate collaboration topology under non-IID heterogeneity remains an open challenge. Early approaches relied on gossip-based protocols Heged˝us et al. [2021], while more recent work studies the security of P2P training against backdoor attacks Syros et al. [2024] and robustness mechanisms against free-riding and collusion Augello et al. [2024], Ranjan et al. [2022]. Within the multi-armed bandit (MAB) framework, CS-UCB Xia et al. [2020] introduced UCB-style scheduling in servermediated FL, and subsequent work extended this idea to decentralized settings with non-stationary MAB formulations over peer groups Listo Zec et al. [2024]. Complementary approaches based on Shapley value estimate client contributions for server-side selection Singhal et al. [2024], Yang et al. [2024]. A second limitation persists across P2P, MAB-, and Shapley-based methods: all assume mandatory participation in each round. Even decentralized variants enforce collaboration once the topology is established, despite evidence that aggregation across heterogeneous institutions can degrade local performance below the no-collaboration baseline Crowson et al. [2022], Li et al. [2025]. We define this as Q2 (selective collaboration under heterogeneity) and address it through the Bayesian propose–reject mechanism (discussed in Sec. 4.3), allowing each center to abstain from participation when all posterior UCB scores fall below the acceptance threshold $\tau _ { \mathrm { a c c } }$

## 2.3 Concept Drift and Bayesian Adaptation

Clinical data distributions evolve due to changes in patient demographics, treatment protocols, and clinical infrastructure, leading to concept drift Rahimli et al. [2024]. Bayesian methods quantify the resulting uncertainty, with two dominant approaches in prior work. The first models uncertainty over model parameters, where Bayesian neural networks in FL aggregate posterior distributions rather than point estimates Saile et al. [2024], Rahman et al. [2025], often incorporating hierarchical or personalized updates to adapt to local data. The second leverages uncertainty for update filtering, where client contributions are selectively incorporated based on confidence measures, such as credible interval thresholds Iglesias Jr. et al. [2024]. Despite their effectiveness, both approaches share a key limitation: abstaining from collaboration is not treated as a principled decision with formal guarantees. In practice, concept drift can render collaboration rounds detrimental, yet existing methods rarely allow nodes to opt out in a theoretically grounded manner Crowson et al. [2022], Gama et al. [2014], Rahimli et al. [2024]. We define this as Q3 (adaptive isolation under drift) and address it through an explicit rest action in Sec. 4.3, supported by a Bayes-optimal isolation criterion (Lemma 1) that provides a closed-form condition under which abstention strictly outperforms collaboration.

## 2.4 Personalization, Quantization, and Communication-Efficient FL

To achieve personalization and bandwidth minimization, it is necessary to keep a subset of model parameters local per client: FedPer and FedRep retain the classifier head locally Arivazhagan et al. [2019]. pFedHN and Ditto provide other per-client adaptations. Quantized transmission compresses the wire format: LLM.int8 Dettmers et al. [2022], QSGD, and signSGD demonstrate single-digitpercent accuracy loss with 2-16× bandwidth reduction. Server momentum (FedAvgM) stabilizes aggregation under heterogeneity. These ideas are orthogonal to partner selection, and in principle, they can be composed with any partner-selection core to push the accuracy-bandwidth frontier further. A fourth limitation runs across all three of these families: bandwidth-saving mechanisms have been studied in isolation from which peers a center should engage with. Real-world multi-institutional medical FL still incurs prohibitive bandwidth Haripriya et al. [2025], and communication-efficient extensions such as knowledge distillation Wu et al. [2022] and metaheuristic aggregation Abdolmaleki and Farahani [2026] treat compression and topology choice as separate problems. We label this open problem Q4 (communication overhead, even in P2P) and answer it through the additive headpersonalization, bfloat16-quantization, and tunable active-set cardinality $\kappa ,$ optimized jointly on the accuracy-bandwidth Pareto frontier (Sec. 6.4, with the row-by-row ablation in Sec. 6.5).

## 3 Problem Formulation

Goal. We seek to learn, for each of $n$ federated centers, a sequence of local predictors that maximizes predictive accuracy on a center’s evolving data distribution while minimizing the cumulative communication overhead of inter-center collaboration, under continuous concept drift in patient demographics, treatment protocols, and equipment. Predictive accuracy is task-dependent and enters the framework through a per-center utility functional $U _ { i } \in [ 0 , 1 ] , U _ { i }$ is the per-center Area Under the Receiver Operating Characteristic curve (AUROC) for binary in-hospital mortality classification.

At each round $t \in \mathsf { \backslash } \{ 1 , \ldots , R \}$ , every center $i \in \{ 1 , \ldots , n \}$ holds local data drawn from an unknown, time-varying joint distribution $P _ { t } ^ { ( i ) } ( X , Y )$ shaped by the center’s patient demographics, comorbidity profile, and care protocols, observed only through patient-level samples, fits local parameters $\mathbf { \bar { w } } _ { i , t } \in \mathbb { R } ^ { | W | }$ , and may exchange them with a per-round active peer set $\mathcal { P } _ { i , t } \subseteq \{ 1 , \dots , n \} \setminus \{ i \}$ . We write $A _ { i } ( \mathbf { w } _ { i , t } ) : = \mathbb { E } _ { ( x , y ) \sim P _ { + } ^ { ( i ) } } [ \check { \mathrm { A U R O C } } ( \mathbf { w } _ { i , t } ; x , y ) ]$ for the expected accuracy of i’s model and treat $\mathcal { P } _ { i , t } = \emptyset$ as a legal first-class action (intentional rest, formalized in Lemma 1). ABPS jointly chooses, for every center and every round, the active peer set and the local update rule, solving

$$
\begin{array} { r } { \underbrace { \operatorname* { m a x } _ { \{ \mathcal { P } _ { i , t } , \mathbf { w } _ { i , t } \} } } _ { \mathrm { p r e d i c i t i v e a c e u r a c y } } \underbrace { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { R } \sum _ { t = 1 } ^ { R } A _ { i } ( \mathbf { w } _ { i , t } ) } _ { \mathrm { p r e d i c i t i v e a c e u r a c y } } \quad \mathrm { s u b j e c t t o } \quad \underbrace { C _ { \mathrm { t o t a l } } \leq B } _ { \mathrm { b a n d w i d t h ~ b u d g e t } } , \quad \underbrace { \operatorname* { s u p } _ { m , k } D \Big [ P _ { m , k } ^ { ( i ) } \parallel P _ { m , k - 1 } ^ { ( i ) } \Big ] \leq \epsilon _ { m } } _ { \mathrm { d r i t b o u n d a t s c a l e m } } . } \end{array}\tag{1}
$$

The maximand is the across-center, across-round mean accuracy. $C _ { \mathrm { t o t a l } } \ \leq \ B$ caps cumulative communication (Eq. 2 below). The drift constraint bounds the per-window divergence by a scalespecific tolerance $\epsilon _ { m }$ at each temporal scale $m \in \{ 1 , 2 , 3 \}$ . When a realized shift exceeds $\epsilon _ { m } .$ , A3 (Sec. 5) breaks at scale $m ,$ , the Bayesian posterior decays toward the prior, the UCB rankings re-order, and the propose-reject mechanism re-forms $\mathcal { P } _ { i , t + 1 }$ . Equation 1 thus makes the three competing forces explicit: accuracy as the maximand, bandwidth as the budget, and drift as the constraint.

Communication cost. Let $\mathcal { P } _ { t } = \{ ( i , j ) : j \in \mathcal { P } _ { i , t } \}$ denote the active peer-pair set in round t, |W<sub>i</sub>| the size of center i’s parameter vector, and $c _ { \mathrm { e g r e s s } }$ the per-GB egress cost $\left( \mathrm { e . g . } \right.$ , \$0.09/GB on AWS Amazon Web Services [2024]). It is worth mentioning here that continuously broadcasting parameters across 100,000+ global healthcare centers typically incurs a prohibitive bandwidth footprint McMahan et al. [2017], and recent medical-FL benchmarks Haripriya et al. [2025] confirm that one 50-round sweep of a VGG-16-class model exceeds 276,000 MB of cross-client traffic. Distillation Wu et al. [2022] and bandwidth-aware aggregation Abdolmaleki and Farahani [2026] cut these figures, but neither couples compression to peer choice. The cumulative cost across R rounds is

$$
C _ { \mathrm { t o t a l } } ~ = ~ c _ { \mathrm { e g r e s s } } \sum _ { t = 1 } ^ { R } \sum _ { ( i , j ) \in \mathcal { P } _ { t } } \left( | W _ { i } | + | W _ { j } | \right) .\tag{2}
$$

In Eq. 2, the summand accounts for the bidirectional transfer of parameter aggregation. Under the uniform-architecture specialization $| W _ { i } | { = } | W |$ , this reduces to $\begin{array} { r } { C _ { \mathrm { t o t a l } } = 2 c _ { \mathrm { e g r e s s } } \left| W \right| \sum _ { t } \left| \mathcal { P } _ { t } \right| } \end{array}$ |, the form used in Theorem 2 and Lemma 1.

Hierarchical concept drift. The drift constraint of Eq. 1 bounds, at three temporal scales $m \in \{ 1 , 2 , 3 \}$ (short-term operational, mediumterm seasonal, long-term demographic), the interwindow divergence $D [ P _ { m , k } ^ { ( i ) } \parallel P _ { m , k - 1 } ^ { ( i ) } ] \leq \epsilon _ { m }$ between consecutive windows of length $\Delta t _ { m }$ $^ { * * } ( { \mathrm { F i g u r e } } \quad 1 ) ^ { * * }$ Gama et al. [2014], Rahimli et al. [2024], Cuturi and Blondel [2017], Düsing and Cimiano [2026],

![](images/088e4695d5a4e5cd91b9c880e93f4ed7d3f8aaf08a2bd22fdfb38d1ecee06629.jpg)  
Figure 1: Hierarchical temporal-shift model, illustrated on MIMIC-IV v3.1 as one concrete instantiation. The construction is dataset-agnostic and applies to any longitudinal multi-center cohort. In this instantiation each cell is one of $n { = } 2 3 0$ federated centers (careunit × 2-year window). Cell shade encodes in-hospital mortality rate, and rows sort six careunits by mean mortality. Horizontal variation within a row illustrates the level- $h _ { 2 }$ within-group term bounded by $\epsilon _ { 2 } ;$ because the windows are shifted-year buckets rather than calendar years, on this axis that variation is of the order of sampling variation (Sec. 6). Vertical shade jumps show level-h<sub>3</sub> inter-group divergence that goal-aware pre-filter (Sec. 4.1) reasons about without exchanging model parameters.

Rahad et al. [2025]. The window size $\Delta t _ { m }$ admits an online empirical estimator, e.g., a Welch t-test that expands the window while $| \mu _ { t } - \mu _ { t + k } | / \sqrt { \sigma _ { t } ^ { 2 } / N _ { t } + \sigma _ { t + k } ^ { 2 } / N _ { t + k } }$ stays below $t _ { \mathrm { c r i t } }$ at $\alpha { = } 0 . 0 5$ . (Note that during experimental validation, we fix $\dot { \Delta } t _ { 2 } = 2$ years and leave adaptive estimation of $\{ \Delta t _ { m } \}$ to future work, since online window adaptation is orthogonal to the partner-selection contribution we focus on here, and Theorem 1 already bounds the within-window cost explicitly through $\epsilon _ { m } ,$ so the headline accuracy-bandwidth claim is unaffected by any reasonable choice of $\Delta t _ { 2 } . \mathrm { ) }$

## 4 Methodology: Adaptive Bayesian Partner Selection

ABPS models inter-center collaboration as a probabilistic, temporally adaptive process built from Bayesian-core components, a goal-aware metadata pre-filter (Sec. 4.1), a Beta-Bernoulli posterior over each peer’s marginal utility (Sec. 4.2), an ϵ-greedy UCB propose-reject topology rule (Sec. 4.3), and a rest action when no proposal is accepted, and three communication-efficiency extensions that compose additively on top of the core, namely, head personalization, bfloat16 quantization, and tunable $\kappa ,$ are all discussed hereafter and analyzed in Sec. 6.4). Figure 2 summarizes the round-level pipeline, and the full pseudocode is given in Algorithm 1 in Appendix B.

![](images/92e198d91749050be7627e901aa15e163d5b9553a23900a3a4624f677e782868.jpg)  
Figure 2: ABPS round-level pipeline. Centers prune candidates via a goal-aware metadata filter (Eq. 3), propose to peers ranked by UCB on Beta belief, observe Shapley marginal utility, and update the belief. When all proposals are rejected, the center rests with $\dot { \mathcal { P } } _ { i } = \mathcal { O }$ (Lemma 1).

## 4.1 Goal-Aware Metadata Pre-Filter

To address Q1 on goal-heterogeneous collaboration from Sec. 2, before invoking the expensive Shapley-UCB stage, each center i computes a d-dimensional summary metadata vector $\mathbf { v } _ { i } \in \mathbb { R } ^ { d }$ of non-sensitive aggregates (positive-class rate, log $n _ { \mathrm { t r a i n } } .$ , normalized mean year of admission, and for goal-aware variants a 12-dim Charlson-comorbidity prevalence). Every entry is a cohort-level scalar, not a per-patient feature, so $\mathbf { v } _ { i }$ carries no Protected Health Information. No model parameters cross the network at this stage, consistent with HIPAA and inter-institutional data-sharing constraints. Goal-aware admission rule. Real healthcare federations are not uniformly similarity-seeking: the three institution types in Sec. 1 have distinct objectives. A community hospital prefers homogeneity, collaborating with demographically similar peers to improve transfer learning for under-represented cohorts. An academic medical center seeks diversity, leveraging exposure to heterogeneous case-mix. In contrast, an Integrated Delivery Network emphasizes alignment, selecting peers with matching target KPIs (e.g., mortality rates) to meet system-level objectives. We generalize the admission rule into a per-center goal-scoring function $f _ { i } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R } \mathrm { : ~ }$

$$
f _ { i } ( \mathbf { v } _ { i } , \mathbf { v } _ { j } ) ~ = ~ \left\{ { \begin{array} { l l } { \displaystyle \mathrm { c o s } ( \mathbf { v } _ { i } , \mathbf { v } _ { j } ) , } & { { \mathrm { g o a l } } = \mathrm { h o m o g e n e i t y } } \\ { - \displaystyle \mathrm { c o s } ( \mathbf { v } _ { i } , \mathbf { v } _ { j } ) , } & { { \mathrm { g o a l } } = \mathrm { d i v e r s i t y } } \\ { - | \boldsymbol { v } _ { i , 0 } - \boldsymbol { v } _ { j , 0 } | , } & { { \mathrm { g o a l } } = \mathrm { a l i g n m e n t } } \end{array} } \right.\tag{3}
$$

In the above equation, $_ { v . , 0 }$ is the pos-rate coordinate used as the KPI marker. Peer j is admitted to $\mathcal { C } _ { i }$ iff $f _ { i } \left( \mathbf { v } _ { i } , \mathbf { v } _ { j } \right) \geq \tau _ { \mathrm { s i m } } .$ . Each formulation lives on a different scale, $\mathrm { 8 0 ~ } \tau _ { \mathrm { s i m } }$ is auto-calibrated per goal to retain a fixed fraction (we use 25%) of candidate pairs: this makes goal choices directly comparable. The threshold can equivalently be learned online via an exponential moving average (EMA) on the acceptance rate. The static calibration suffices for our experiments.

Complexity and theoretical preservation. The pre-filter is $\mathcal { O } ( n d )$ per round and transmits only $\mathbf { v } _ { i }$ (a few dozen bytes) per center. All downstream Shapley evaluations operate on the filtered arm set, reducing the expected per-round bandwidth by the same fraction. Theorems 1 and 2 transfer verbatim once the arm set is restricted to $\mathcal { C } _ { i }$ . The only added bias is whether the true best peer survives the filter, which we bound in Appendix C via a standard top-k coverage argument for descriptor-based recall. Empirically (Sec. 6.4), the diversity goal activates Lemma 1’s $\left. \bar { \mathcal { P } } _ { i } \right. = 0$ regime in ∼ 40% of rounds on the biomedical dataset, the first empirical observation of intentional isolation in our study.

## 4.2 Bayesian Belief Updating on Marginal Utility

Rather than maintaining a belief over a peer’s explicitly shared raw parameters, center i evaluates the actual marginal utility (performance gain) a peer j provides. We model this utility using the Shapley value to objectively allocate credit across historical collaboration subsets ${ \mathcal { S } } \subseteq { \mathcal { P } } _ { i } ^ { \mathsf { ^ { - } } } \backslash \{ j \}$

$$
\phi _ { j \to i } = \sum _ { S \subseteq \mathcal { P } _ { i } \setminus \{ j \} } \frac { | S | ! \left( | \mathcal { P } _ { i } | - | S | - 1 \right) ! } { | \mathcal { P } _ { i } | ! } \left( U _ { i } ( S \cup \{ j \} ) - U _ { i } ( S ) \right) ,\tag{4}
$$

In the above equation, $U _ { i } ( \cdot ) \in [ 0 , 1 ]$ is the local model’s normalized held-out validation performance (we use AUROC). Because $U _ { i }$ is bounded in [0, 1], the per-peer Shapley contribution $\phi _ { j  i }$ is bounded in $[ - 1 , + 1 ]$ . We map it to the unit interval via the affine clip

$$
\bar { \phi } _ { j \to i } \ = \ \mathrm { c l i p } _ { [ 0 , 1 ] } \left( { \frac { \phi _ { j \to i } - \phi _ { \mathrm { m i n } } } { \phi _ { \mathrm { m a x } } - \phi _ { \mathrm { m i n } } } } \right) ,\tag{5}
$$

with truncation thresholds $\phi _ { \mathrm { m i n } } , \phi _ { \mathrm { m a x } }$ chosen so that a single anomalous round cannot saturate the posterior (we use $[ - 0 . 1 , + 0 . 1 ]$ throughout this paper). Modeling $\bar { \phi } _ { j  i }$ as a soft Bernoulli observation, the conjugate Beta prior $\bar { \phi } _ { j \to i } \sim \mathrm { B e t a } ( \alpha _ { i \to j } , \beta _ { i \to j } )$ admits the closed-form update

$$
\alpha _ { i  j } ^ { ( k + 1 ) }  \alpha _ { i  j } ^ { ( k ) } + \bar { \phi } _ { j  i } , \qquad \beta _ { i  j } ^ { ( k + 1 ) }  \beta _ { i  j } ^ { ( k ) } + ( 1 - \bar { \phi } _ { j  i } ) .\tag{6}
$$

The Beta-Bernoulli formulation offers three properties. (i) The posterior support [0, 1] matches the bounded co-domain of $\bar { \phi } _ { j  i }$ , ruling out the model mis-specification that would otherwise arise when Shapley contributions are negative. (ii) The posterior variance $\alpha \beta / [ ( \alpha + \beta ) ^ { 2 } ( \alpha + \beta + 1 ) ]$ admits the Hoeffding-style concentration exploited in Theorem 1. (iii) The upper-confidence bound(in Sec. 4.3) for partner ranking takes the standard UCB1 form and inherits sublinear regret (Theorem 2).

## 4.3 Collaborative Topology Formulation

Recall Q2 (on selective collaboration under heterogeneity) and Q3 (on adaptive isolation under drift) from Sec. 2. The propose-reject protocol below operationalizes both, with the explicit rest action carrying through to the optimality result of Lemma 1. The global time horizon T is partitioned into discrete interaction intervals $\{ t _ { 1 } , \ldots , t _ { K } \}$ . Collaboration decisions among the centers are governed by a propose-and-reject protocol approximating a stable-marriage solution, augmented with an ϵ-greedy Upper Confidence Bound (UCB) strategy Auer et al. [2002]. During each period $t _ { k }$ , center i computes a utility-based UCB score for every candidate peer j:

$$
\mathrm { U C B } _ { j \to i } ( t _ { k } ) = \hat { \mu } _ { i \to j } ( t _ { k } ) + \gamma \sqrt { \frac { 2 \log t _ { k } } { n _ { i \to j } ( t _ { k } ) + 1 } } ,\tag{7}
$$

Here, $\hat { \mu } _ { i  j } ( t _ { k } ) = \alpha _ { i  j } ^ { ( t _ { k } ) } / \big ( \alpha _ { i  j } ^ { ( t _ { k } ) } + \beta _ { i  j } ^ { ( t _ { k } ) } \big )$ is the posterior mean of the Beta belief $( 6 ) , n _ { i  j } ( t _ { k } )$ is the number of past observations from $j ,$ and $\gamma > 0$ is the exploration weight. With probability $1 - \epsilon ,$ center i sequentially proposes collaboration to peers sorted by descending $\mathrm { U C B } _ { j  i }$ . With probability $\epsilon ,$ it purposefully queries a uniformly random unproven neighbor. A receiving peer $j$ evaluates the proposal and accepts only if its own UCB on i exceeds the threshold $\tau _ { \mathrm { a c c } }$ . If j rejects, i extends the proposal to the next highest-ranked peer until either κ acceptances accrue or the candidate list is exhausted. Critically, if all proposals are rejected, center i rests: its active peer set P<sub>i</sub> collapses to ∅ and no parameters are exchanged this round. This isolation mechanism is a deliberate design feature, not a system failure. In highly heterogeneous networks, forcing connections with dissimilar node degrades local performance via negative transfer. Choosing isolation whenever the expected gain falls below the threshold helps protect local utility and conserves bandwidth (Lemma 1). Resting is per-round and self-correcting. The next round re-runs the propose-reject loop with an updated posterior and a UCB exploration radius that grows in t, and the ϵ-greedy swap admits a uniformlyrandom peer with probability ϵ regardless of UCB ranking, so collaboration resumes the moment any peer’s UCB rises above $\tau _ { \mathrm { a c c } }$ or drift makes a previously-low-utility peer informative again.

## 5 Theoretical Guarantees and Complexity Analysis

We analyze ABPS along posterior concentration (Theorem 1), partner-selection regret (Theorem 2), and Bayes-optimality of isolation (Lemma 1). Proofs are in Appendix C. The analysis rests on three assumptions. A1 bounded utility, $U _ { i } \in [ 0 , 1 ]$ , so $\bar { \phi } _ { j  i } \in [ 0 , 1 ]$ via the affine clip of $\mathrm { E q . } 5$ (see Sec. 4.2). A2 conditional independence of the ϕ<sup>¯</sup> observations given $\mu _ { i \to j } ^ { \star } ,$ justified by independent local stochastic gradient descent draws and disjoint mini-batches per round. A3 quasi-stationary drift $| \mathbb { E } [ \bar { \phi } _ { j  i } ^ { ( k ) } ] - \mu _ { i  j } ^ { \star } | \leq \epsilon _ { m }$ within any window $\Delta t _ { m } ,$ , binding the theory to the hierarchical formulation of Sec. 3. For the filtered-arm-set refinement of Theorem 2, we additionally assume A4 that the metadata descriptor v<sub>i</sub> is L-Lipschitz informative about utility, $| \mu _ { i \to j } ^ { \star } - \mu _ { i \to j ^ { \prime } } ^ { \star } | \leq L \| \mathbf { v } _ { j } - \mathbf { v } _ { j ^ { \prime } } \|$ (see Appendix C.2 for the precise statement and use).

## 5.1 Convergence of the Beta Posterior

Theorem 1 (Beta-belief concentration). Let $\bar { \phi } _ { j  i } ^ { ( 1 ) } , \ldots , \bar { \phi } _ { j  i } ^ { ( K ) } \in [ 0 , 1 ]$ be the clipped Shapley observations collected by center i for peer j across K rounds inside a single drift window, with empirical mean $\begin{array} { r } { \bar { \mu } _ { K } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bar { \phi } _ { j \to i } ^ { ( k ) } } \end{array}$ and posterior mean $\begin{array} { r } { \hat { \mu } _ { K } = \frac { \alpha _ { 0 } + K \bar { \mu } _ { K } } { \alpha _ { 0 } + \beta _ { 0 } + K } } \end{array}$ . Under A1-A3,for any $\delta \in ( 0 , 1 )$ with probability at least $1 - \delta$

$$
\big | \hat { \mu } _ { K } - { \mu } _ { i \to j } ^ { \star } \big | \le \underbrace { \sqrt { \frac { \log ( 2 / \delta ) } { 2 K } } } _ { H o e f f d i n g } + \underbrace { \frac { \alpha _ { 0 } + \beta _ { 0 } } { \alpha _ { 0 } + \beta _ { 0 } + K } } _ { p r i o r d e c a y } + \underbrace { \epsilon _ { m } } _ { d r i f t } .\tag{8}
$$

Hence $\hat { \mu } _ { K } \stackrel { p } {  } \mu _ { i  j } ^ { \star }$ as $K  \infty$ with $\epsilon _ { m }  0 .$

The proof (Appendix C.1) decomposes the error into Hoeffding noise $( \mathcal { O } ( 1 / \sqrt { K } ) )$ ), prior decay $( \mathcal { O } ( 1 / K ) )$ , and within-window drift bounded by $\epsilon _ { m }$ , the last of which motivates the Welch-t window selection of Sec. 3.

## 5.2 Regret of the UCB Partner Selection Bandit

We bound the regret of the per-center partner-selection problem viewed as a κ-armed banditLattimore and Szepesvári [2020] over the candidate pool C<sub>i</sub>. Let $\mu _ { ( 1 ) } ^ { \star } \geq \mu _ { ( 2 ) } ^ { \star } \geq . .$ . denote the ordered true utilities of i’s candidate peers and define the suboptimality gaps $\dot { \Delta _ { j } } = \mu _ { ( 1 ) } ^ { \star } - \mu _ { ( j ) } ^ { \star }$

Theorem 2 (Sublinear partner-selection regret). Under assumptions $A I { - } A 2$ and a stationary window $( \epsilon _ { m } = 0 )$ , the cumulative regret of ABPS with exploration parameter $\gamma = \sqrt { 2 }$ and ϵ-greedy exploration probability $\epsilon \in [ 0 , 1 )$ ) over T rounds satisfies

$$
\mathcal { R } ( T ) \leq \kappa \sum _ { j : \Delta _ { j } > 0 } \left( \frac { 8 \log T } { \Delta _ { j } } + \Delta _ { j } \left( 1 + \frac { \pi ^ { 2 } } { 3 } \right) \right) + \epsilon T \cdot \mathbb { E } _ { j \sim \operatorname { U n i f } ( \mathcal { C } _ { i } ) } [ \Delta _ { j } ] .\tag{9}
$$

Setting $\epsilon = \mathcal { O } ( 1 / \sqrt { T } )$ recovers the standard O(log T) regret rate ofUCB1 up to thefactor κ. The proof (Appendix C.2) bounds the greedy phase via canonical UCB1 analysis Auer et al. [2002] scaled by κ and the exploratory phase via the $\epsilon T \cdot \mathbb { E } [ \Delta _ { j } ]$ term, with a filtered-arm-set refinement under A4 that improves the constants when the goal-aware pre-filter is active. In stationary windows, ABPS therefore matches an oracle that always proposes to the κ best peers.

## 5.3 Optimality of Intentional Isolation

Recall Q3 (adaptive isolation under drift) from Sec. 2. The lemma below is the formal optimality guarantee that the rest of the action of Sec. 4.3 promised.

Lemma 1 (Intentional isolation dominates forced collaboration). Let $V _ { i } ^ { \mathrm { r e s t } }$ denote the expected one-step utility ofcenter i when it rests $( \mathcal { P } _ { i } = \mathcal { O } )$ and $V _ { i } ^ { \mathrm { c o l l } } ( \mathcal { P } )$ the expected utility when itforcibly collaborates with peer set ${ \mathcal { P } } \neq { \mathcal { D } } .$ . Suppose the per-peer expected marginal utility satisfies $\mu _ { i \to j } ^ { \star } < c _ { \mathrm { n e g } }$

for every $j \in \mathcal { P } ,$ , where $c _ { \mathrm { n e g } }$ is the negative-transfer threshold defined by $V _ { i } ^ { \mathrm { c o l l } } ( \{ j \} ) = V _ { i } ^ { \mathrm { r e s t } }$ when $\mu _ { i  j } ^ { \star } = c _ { \mathrm { n e g } } .$ . Then (10)

$$
V _ { i } ^ { \mathrm { r e s t } } ~ > ~ V _ { i } ^ { \mathrm { c o l l } } ( \mathcal { P } ) ,\tag{10}
$$

and the bandwidth saved by resting is exactly $C _ { i } ^ { \mathrm { { r e s t } } } = 2 | \mathcal { P } | \cdot c _ { \mathrm { { e g r e s s } } } \cdot | W | p e r$ round $( c f . e q . ( l ) )$ The proof (Appendix C.3) combines Shapley efficiency with the affine clip of Eq. 5, which fixes $c _ { \mathrm { n e g } } = - \phi _ { \mathrm { m i n } } / ( \phi _ { \mathrm { m a x } } - \phi _ { \mathrm { m i n } } ) = 0 . 5$ for $( \phi _ { \mathrm { m i n } } , \phi _ { \mathrm { m a x } } ) = ( - 0 . 1 , + 0 . 1 )$ , exactly $\tau _ { \mathrm { a c c . } } ~ \mathrm { A B P S }$ approximates this oracle by resting whenever every UCB falls below $\tau _ { \mathrm { a c c } } ,$ , and Theorem 1 guarantees convergence to the oracle as $K  \infty$

## 5.4 Computational Complexity

Let $\rho \in ( 0 , 1 ]$ denote the expected top-k coverage of the goal-aware pre-filter (Sec. 4.1), i.e. the fraction of peers that survive the $\tau _ { \mathrm { s i m } }$ threshold. When $\tau _ { \mathrm { s i m } }$ is calibrated to retain a prescribed keepfraction, ρ equals that target (we use $\rho = 0 . 2 5 )$ . Define the post-filter candidate size $| { \mathcal C } _ { i } | \approx \rho ( n - 1 )$ The per-round computational cost at each center is strictly bounded and decomposes into four components. (1) Goal-aware filter: Evaluating $f _ { i } ( \mathbf { v } _ { i } , \mathbf { v } _ { j } )$ for each of the $n - 1$ peers using d-dimensional metadata incurs a cost of ${ \bar { \boldsymbol { \mathcal { O } } } } ( n d )$ , where $d \leq 1 5$ in our implementation (3 base features and 12 comorbidity indicators); (2) UCB ranking and propose–reject: Ranking candidates in the filtered set $| { \mathcal { C } } _ { i } | \approx \rho n$ and traversing the ordered list requires ${ \overline { { \mathcal { O } ( \rho n \log ( \rho n ) ) } } }$ time requires a ρ-factor reduction compared to the unfiltered case; (3) Shapley credit assignment: The exact computation of this involves evaluating $2 ^ { | \mathcal { P } _ { i } | }$ coalitions. Since $| \mathcal { P } _ { i } | \le \kappa$ by design, the cost is ${ \mathcal { O } } ( 2 ^ { \kappa } )$ , independent of n. For $\kappa > 5$ , we instead employ Truncated Monte Carlo (TMC) Shapley Ghorbani and Zou [2019], which achieves an ε-accurate estimate in ${ \mathcal O } ( \kappa \log \kappa / \varepsilon ^ { 2 } )$ samples; and (4) Beta–Bernoulli update: Posterior updates require only closed-form scalar operations, yielding O(1) complexity.

The total per-round computation at center i is therefore $\mathcal { O } ( n d + \rho n \log ( \rho n ) + 2 ^ { \kappa } )$ . For the default $\rho = 0 . 2 5$ and $\kappa = 1$ this collapses to $\mathcal { O } ( n d )$ , independent ofthe number ofmodel parameters |W|.

Communication. Each round transmits (i) the metadata vector v<sub>i</sub> once at filter time (a few dozen bytes per center, no model weights) and (ii) the active-peer aggregation for the κ peers that survive both the filter and the UCB threshold, costing $2 \kappa | W |$ per active center per round. With personalization (Sec. 6.4), only the shared trunk is transmitted. With quantization, each |W| is reduced to its bfloat16 footprint. When $| \mathcal { P } _ { i } | = 0$ (intentional isolation, empirically realized under the diversity goal), the communication collapses to zero for that center-round.

## 6 Experimental Evaluation

We evaluate ABPS along four axes: (i) predictive accuracy under realistic non-IID partitioning of MIMIC-IV, a publicly available database sourced from the Beth Israel Deaconess Medical Center (BIDMC) electronic health record Johnson et al. [2023], (ii) cumulative transmitted bytes as a direct proxy for communication cost, (iii) the empirical realization of the intentional-isolation property of Lemma 1, and (iv) an ablation decomposing the contribution of each framework extension. Code, sbatch scripts, and per-seed per-method result JSONs will accompany the camera-ready submission.

## 6.1 Dataset and Federation Setup

The task at hand is the binary in-hospital mortality prediction over the first 24 hours of an ICU stay Johnson et al. [2023]. After applying the standard age filter $( 1 8 \leq \mathsf { a g e } \leq 9 5 ) .$ , the cohort contains roughly 76,000 ICU stays. The patient features: demographics (age, gender, race), admission context (admission type, location, insurance), Charlson-style comorbidity binaries, discussed in Sec. 6.3, derived from ICD-10/ICD-9 codes, and aggregated first-24h vitals (heart rate, systolic/diastolic/mean BP, respiration rate, $\mathrm { S p O } _ { 2 } .$ , temperature) summarized as {mean, min, max}. Continuous features are standardized per center on the local training split to respect federated isolation.

Our experiments rely on a careunit-by-year partitioning of MIMIC-IV, where a center is defined as the Cartesian product of a care unit and a consecutive two-year admission window. The care units include CVICU, CCU, MICU, Medical/Surgical ICU, SICU, and TSICU, yielding $n = 2 3 0$ centers spanning the shifted temporal range 2110 to 2191 in MIMIC-IV v3.1. MIMIC-IV de-identifies dates by a single random offset per subject, applied uniformly to all of that subject’s events Johnson et al. [2023], so within-patient order is exact, but the shifted-year windows are not aligned with calendar time: across the 71,008 stays in the partition, Cramér’s V between the assigned window and the published anchor\_year\_group field is 0.045, and mean window purity is 0.335 against a chance value of 0.336. We therefore treat the windows as a partitioning device rather than a calendar axis, and the empirical evidence of drift over calendar time in Appendix A uses the published era field. What the construction provides is a deterministic partition into many small centers whose outcome distributions differ: the mean pairwise Jensen-Shannon divergence between the Bernoulli mortality distributions of centers from different care units is 0.0066 nats, against 0.0009 nats within a care unit, and mortality rates vary across units (approximately 3% in CVICU versus 15% in MICU), aligning with the IDN, AMC, and Community Hospital framing in Sec. 1. The pipeline also supports Dirichlet(α) label-skew partitions Hsu et al. [2019] and uniform IID splits, but the careunit-by-year setting is used for all primary results. A FedProx-Synthetic(α, β) generator Li et al. [2020] matching the MIMIC feature schema drives implementation tests but is not used for headline numbers.

## 6.2 Baselines

We compare against ten methods grouped by purpose. Reference anchors: Centralized pools all data into one Multi-Layer Perceptron (MLP) (non-federated upper bound, in the Table 1 caption), Local-only trains each center independently (no-collaboration lower bound), and FedAvg McMahan et al. [2017] averages weights every round. Centralized non-IID FL: FedProx Li et al. [2020] adds a proximal regularizer, FedDyn Acar et al. [2021] aligns local objectives with the global stationary point, and MOON Li et al. [2021] maximizes agreement between local and global representations. Decentralized P2P FL: DeceFL Yuan et al. [2023] provably converges to the centralized optimum, DeFTA Zhou et al. [2024] is a plug-and-play decentralized FedAvg with trust-based reweighting (the closest peer-selection competitor), and WPFed Ye et al. [2024] uses Locality-Sensitive Hashing (LSH) similarity filtering plus weighted neighbor selection, mirroring the metadata pre-filter of Sec. 4.1 but lacking the Bayesian posterior update. Bayesian FL: BNN+FL Saile et al. [2024] replaces the MLP with a Bayesian neural network and aggregates posterior moments (ABPS’s novelty is Bayesianizing the utility signal rather than the weights). BrainTorrent Guha Roy et al. [2019], Gossip Learning Heged˝us et al. [2021], KL-FedDis Rahad et al. [2025], and Peer-Driven Reputation FL Seidi et al. [2025] are conceptual antecedents discussed in Sec. 2 but not run as baselines.

## 6.3 Implementation and Metrics

The local model is a two-hidden-layer MLP (128 → 64, ReLU, dropout 0.2) optimized with Adam $( \eta = 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 5 } )$ . Each round runs $K _ { \mathrm { l o c a l } } = 2$ local epochs with batch size 128 over $K = 5 0$ rounds (or up to $K = 1 0 0$ for the $\mathrm { \ A B P S { - } X }$ variant with validation-AUROC early stopping at patience 10). Base ABPS uses $\kappa = 3 , \epsilon = 0 . 1 , \gamma = \sqrt { 2 } , \tau _ { \mathrm { a c c } } = 0 . 5$ , the 3-dim base metadata vector [positive-class rate, $\log n _ { \mathrm { t r a i n } } , \overline { { \mathrm { y e a r } } } / 2 0 3 0 ]$ , and $\tau _ { \mathrm { s i m } } = 0$ (no filtering). The goal-aware variants enrich metadata with a 12-dim per-center prevalence vector over Charlson Comorbidity Index (CCI) Charlson et al. [1987] chronic-condition categories (binary presence per patient, averaged over local training cohort, no per-patient leakage) and auto-calibrate $\tau _ { \mathrm { s i m } }$ to retain 25% of peer pairs per goal. Shapley contributions are computed exactly when $| \mathcal { P } _ { i } | \le 5$ and via truncated Monte-Carlo with $B = 8$ permutations otherwise. We report mean AUROC across centers (across-seed std over 5 seeds in {11, 22, 33, 44, 55}) and cumulative transmitted bytes counted once per undirected edge.

Experiments were run on a Simple Linux Utility for Resource Management (SLURM) Yoo et al. [2003]-managed High-Performance Computing (HPC) cluster, each SLURM task on a single NVIDIA Tesla V100-SXM2 GPU (32 GB) with 8 CPU cores and 32 GB RAM, under PyTorch 2.5.1 (CUDA 12.1) and Python 3.11. The n=230-center sweep (60 array tasks: 12 configurations × 5 seeds) finishes in 1 to 3 wall-clock hours, and the headline ABPS-X sweep (10 tasks at 100 rounds with early stopping) finishes in 3 to 9 minutes per task. Reproduction of table cells and figures requires a single sbatch of the two sweep scripts released with the supplementary material and JSON outputs.

## 6.4 Headline Result: Accuracy vs. Bandwidth

Recall Q4 (communication overhead, even in P2P) from Sec. 2. Table 1 and the Pareto frontier of Figure 3 are the empirical answer. Table 1 reports final-round mean AUROC over 5 seeds, and Figure 3 plots the same data as an accuracybandwidth Pareto frontier. All federated methods gain ∼ 15 AUROC points over the local-only lower bound $( 0 . 5 8 7 \pm 0 . 0 1 3 )$ and close most of the gap to the non-federated Centralized upper bound (0.827 ± 0.007).

Base ABPS (with κ=3 and no personalization or quantization) already achieves AUROC $0 . 7 5 3 \pm 0 . 0 1 5$ , statistically indistinguishable from FedAvg (0.755 ± 0.019), FedProx (0.753 ± 0.019), and FedDyn $( 0 . 7 5 8 \pm 0 . 0 1 7 )$ . It transmits 1.50× FedAvg’s bandwidth because a κ=3 mesh has more unique edges than FedAvg’s star, matching the theoretical accounting in Sec. 5.4.

Table 1: In-hospital mortality prediction on MIMIC-IV (n=230 centers, 50 rounds, 5 seeds). AUROC is the mean across centers (mean ± standard deviation). Bandwidth is a cumulative parameter exchange relative to FedAvg = 1.00×. As a non-federated upperbound reference (not a fair federated comparator).
<table><tr><td>Method</td><td>AUROC (↑)</td><td>Bandwidth</td></tr><tr><td>Local-only (lower bound)</td><td> $0 . 5 8 7 \pm 0 . 0 1 3$ </td><td>0.00×</td></tr><tr><td>FedAvg McMahan et al. [2017]</td><td> $0 . 7 5 5 \pm 0 . 0 1 9$ </td><td>1.00×</td></tr><tr><td>FedProx Li et al. [2020]</td><td> $0 . 7 5 3 \pm 0 . 0 1 9$ </td><td>1.00×</td></tr><tr><td>FedDyn Acar et al. [2021]</td><td> $0 . 7 5 8 \pm 0 . 0 1 7$ </td><td>1.00×</td></tr><tr><td>MOON Li et al. [2021]</td><td>0.750 ± 0.020</td><td>1.00×</td></tr><tr><td>DeceFL Yuan et al. [2023]</td><td>0.696 ± 0.011</td><td>1.00×</td></tr><tr><td>DeFTA Zhou et al. [2024]</td><td>0.725 ± 0.013</td><td>1.00×</td></tr><tr><td>WPFed Ye et al. [2024]</td><td>0.694 ± 0.012</td><td>1.00×</td></tr><tr><td>BNN+FL Saile et al. [2024]</td><td>0.749 ± 0.014</td><td>2.00×</td></tr><tr><td> $\mathbf { A B P S _ { \alpha } } ( \mathbf { b a s e } , \kappa \mathbf { = } 3 )$ </td><td> $0 . 7 5 3 \pm 0 . 0 1 5$ </td><td>1.50×</td></tr><tr><td>ABPS+P (personalize)</td><td> $0 . 7 5 7 \pm 0 . 0 1 5$ </td><td>1.49×</td></tr><tr><td> $\mathbf { A B P S + Q } \ ( \mathbf { \bar { b } f l o a t 1 6 } )$ </td><td> $0 . 7 5 3 \pm 0 . 0 1 5$ </td><td>0.75×</td></tr><tr><td> $\mathbf { A B P S + P + Q + \kappa = 1 }$ </td><td> $0 . 7 4 8 \pm 0 . 0 1 2$ </td><td>0.25×</td></tr><tr><td> $\mathrm { F e d A v g + P + Q }$ </td><td> $0 . 7 5 8 \pm 0 . 0 1 7$ </td><td>0.50×</td></tr><tr><td> $\mathrm { A B P S + P + Q + \kappa = 1 \ ( h o m . ) }$ </td><td> $0 . 7 4 2 \pm 0 . 0 1 3$ </td><td>0.25×</td></tr><tr><td> $\mathrm { A B P S + P + Q + \kappa = 1 \ ( d i v . ) }$ </td><td> $0 . 6 9 4 \pm 0 . 0 1 5$ </td><td>0.16×</td></tr><tr><td> $\mathrm { A B P S + P + Q + \kappa = 1 \ ( a l i g n . ) }$ </td><td> $0 . 7 3 9 \pm 0 . 0 1 0$ </td><td>0.25×</td></tr><tr><td> ${ \bf A B P S - X _ { \ell } ( o u r s , f u l l ) }$ </td><td> $0 . 7 5 8 \pm 0 . 0 1 0$ </td><td>0.09×</td></tr><tr><td> $\mathrm { F e d A v g  – X ( f a i r c o m p . , f u l l ) }$ </td><td> $0 . 6 9 1 \pm 0 . 0 0 8$ </td><td>0.31×</td></tr></table>

## Three extensions independently move the

Pareto frontier: (P) head personalization $\mathrm { A r i - }$

vazhagan et al. [2019] boosts AUROC to 0.757±0.015 at the same bandwidth (the per-center head specializes to careunit mortality base-rates, e.g., CVICU ∼ 3% vs MICU ∼ 15%). (Q) bfloat16 Dettmers et al. [2022] halves bandwidth with no accuracy loss $( 0 . 7 5 3 \pm 0 . 0 1 5$ at 0.75×). (κ=1) collapsing the active set to a single partner halves the mesh edges again and, combined with (P) and (Q), yields the ABPS+P+Q+κ=1 row: 0.748 ± 0.012 at 0.25×, already Pareto-dominating federated baselines. Adding the goal-aware pre-filter, 2-layer personalization, server momentum, and validation-AUROC early stopping yields full ABPS-X: 0.758 ± 0.010 (matching FedDyn) at 0.09× FedAvg. Fair comparator and ABPS-X. Applying P+Q to FedAvg (FedAvg+P+Q) yields 0.758 ± 0.017 at

![](images/2402f05f4e3ad176fff981160061a97df6d73f5e0de02e58489d7cfb86530704.jpg)

![](images/2e9d3bba5900fef57f49c83577859afed15560d054db54fedcee2f8c5b3c219c.jpg)  
Figure 3: Results on MIMIC-IV careunit-by-year $( n = 2 3 0$ centers, 5 seeds). (a) Accuracy ranking: mean AUROC with across-seed std, color-coded by method family (blue: star-topology FL, green: decentralized, orange: ABPS+P/Q, red: ABPS-X, purple/brown: goal-aware, olive: local-only). The dashed line denotes the non-federated centralized upper bound (off-axis). (b) Accuracy vs. bandwidth Pareto (log-scale bandwidth): ABPS-X (dark-red star) matches FedDyn’s AUROC (0.758) at 0.09× FedAvg bandwidth. Red arrows show the ablation path $\partial . ^ { \mathrm { B P S } } \mathcal { K } _ { \mathrm { , ~ a ~ } } ^ { \mathrm { + P } }$ <sup>→+Q→+P+Q+κ=1→ABPS-X,</sup> <sup>with</sup> <sup>each</sup> <sup>step</sup> <sup>improving</sup> <sup>the</sup> <sup>trade-off.</sup> modest improvement, but cannot reach κ=1 because the star topology has no mechanism for selecting a single best peer. The full ABPS-X variant adds 2-layer personalization, server-side EMA momentum $\beta { = } 0 . 5 ,$ 15-dim goal-aware metadata under the homogeneity goal, and validation-AUROC early stopping (patience 10 of 100), reaching $\mathbf { 0 . 7 5 8 \pm 0 . 0 1 0 \mathrm { ~ a t ~ } 0 . 0 9 \times \mathrm { F e d A v g } }$ with tighter variance than FedDyn. The same-extension FedAvg-X fair comparator drops to $0 . 6 9 1 \pm 0 . 0 0 8$ , 6.7 AUROC points behind ABPS-X: 2-layer personalization over-adapts each private head when the star topology averages across all 230 peers, whereas ABPS’s $\kappa { = } 1$ Bayesian selection supplies the one-peer constraint under which personalization helps. Empirical isolation does not activate here $( \mathbb { E } [ | \mathcal { P } _ { i } | ]$ ≈ 1.00 when $\kappa { = } 1 )$ , but the goal-aware filter of the next subsection engages Lemma 1’s rest regime in up to 41% of rounds.

## 6.5 Empirical Validation of Lemma 1 (Intentional Isolation)

The diversity-goal experiment below is the empirical activation of the rest regime promised by Lemma 1 to address $\mathbf { \Delta } Q 3$ (adaptive isolation under drift).. The isolation lemma predicts that when every peer’s true expected utility falls below the negative-transfer threshold ${ \mathrm { c } } _ { \mathrm { n } \mathrm { e g } } ,$ the optimal action is $\mathcal { P } _ { i } = \mathcal { D }$ and the resulting bandwidth is zero. We validate this prediction on MIMIC-IV through the goal-aware filter of Sec. 4.1: by varying the admission rule, we directly control which peers survive to the UCB-Shapley stage, which in turn controls whether the UCB thresholds admit proposals.

We run $\mathbf { A B P S + P + Q + } \kappa { = } 1$ on the full $n { = } 2 3 0$ federation under each of the three goals in Eq. 3 with $\tau _ { \mathrm { s i m } }$ auto-calibrated to retain 25% of peers per center, and record the per-round fraction of centers with $| { \mathcal { P } } _ { i } | { = } 0$ . Homogeneity and alignment admit peers with similar (or KPI-matched) marginals, so isolation rarely activates $( \mathbb { E } [ | \mathcal { P } _ { i } | ] { = } 0 . 9 9 )$ , giving AUROC 0.742 and 0.739 at 0.25× bandwidth. Diversity admits dissimilar peers, many crossing the negative-transfer threshold, so the UCB falls below $\tau _ { \mathrm { a c c } }$ and rest activate for 41% of centers $( \mathbb { E } [ | \mathcal { P } _ { i } | ] { = } 0 . 5 9 )$ . Bandwidth drops to 0.16× FedAvg, the lowest observed, while AUROC decreases to 0.694. The bandwidth reduction tracks isolation within 5%, the collaborate-to-rest transition is sharp at the Hoeffding rate $\sqrt { \log ( 2 / \delta ) / ( 2 K ) }$ , and this is the first setting where Lemma 1’s $| { \mathcal { P } } _ { i } | = 0$ regime activates at scale on real clinical data.

Ablations: Reading Table 1: ABPS+P improves AUROC by +0.4 at the same bandwidth, while +Q maintains performance at 0.75×. The combined $\mathbf { A B P S + P + Q + \kappa = 1 }$ setting preserves AUROC at just $0 . 2 5 \times$ bandwidth. In contrast, FedAvg+P+Q reaches $0 . 7 5 8 \pm 0 . 0 1 7$ at 0.50× bandwidth but cannot operate at $\kappa { = } 1$ , indicating that the additional $2 \times$ reduction is due to Bayesian Shapley-UCB selection. The three goal-aware rows (Eq.3) ablate the pre-filter (Sec.6.5). Hyperparameters $\epsilon , \gamma ,$ , and truncated-MC B are fixed from preliminary synthetic runs.

## 7 Conclusion and Future Work

We presented ABPS, an adaptive Bayesian P2P federated-learning framework that replaces raw parameter aggregation with Shapley-based marginal-utility evaluation, formalizes intentional isolation as a Bayes-optimal action under negative transfer, and composes with three communication-efficiency extensions (head personalization, bfloat16 quantization, tunable κ) plus a goal-aware metadata pre-filter. On MIMIC-IV with $n { = } 2 3 0$ careunit-by-year centers, the full ABPS-X variant matches the strongest federated baseline (FedDyn) at 0.09× FedAvg bandwidth, while applying the same extensions to FedAvg drops 6.4 AUROC points (isolating the gain to the Bayesian selection itself), and the diversity goal activates Lemma 1’s rest regime for 41% of centers per round, a first on real clinical data.

Limitations and future work. The evaluation is on a single dataset and binary clinical task, the regret bound degrades with unmeasured $\epsilon _ { m }$ , and we provide no formal $( \epsilon , \delta ) { \tt- D P } ,$ survival, or Byzantine guarantees. Natural follow-ups include (i) a Gaussian-mechanism DP wrapper on $\mathbf { v } _ { i }$ for certifiable privacy, (ii) multi-modal pipelines fusing EHR with medical imaging, (iii) online Welch-t estimation of $\{ \Delta t _ { m } \}$ together with a drifting-bandit regret analysis, and (iv) porting to time-to-event outcomes (matching the Cox-style framing of Seidi et al. [2025]).

Scope of the accuracy claim. The matched-accuracy result holds for federations of many small centers. On a 40-center partition of the same cohort built from the published anchor\_year\_group eras, with a median of roughly 2,000 stays per center, FedAvg and FedDyn reach 0.813 and 0.805 AUROC after 100 rounds while ABPS-X peaks at 0.793, although ABPS-X reaches each intermediate AUROC target (0.750, 0.770, 0.785) with four to six times fewer bytes. Where centers are large, each local model is already well estimated and aggregating across all of them is close to optimal; where centers are numerous and small, indiscriminate aggregation carries more harmful transfer and selective exchange is competitive.

Broader impact. Cutting per-center bandwidth tenfold at matched accuracy on federations of many small centers lowers the entry barrier for resource-constrained sites that disproportionately serve under-represented populations. Residual privacy and goal-misuse risks (metadata re-identification under auxiliary information, and goal-aware filtering used to entrench rather than correct bias) are addressed by the recommended DP wrapper and governance over goal declarations, documented in full in the NeurIPS Reproducibility Checklist.

## References

Mostafa Abdolmaleki and Bahar Farahani. Sync-GWO: Highly private and bandwidth-efficient federated learning with a case study in healthcare. IEEE Journal of Biomedical and Health Informatics, 30(3):1939–1946, 2026. doi: 10.1109/JBHI.2025.3567913.

Durmus Alp Emre Acar, Yue Zhao, Ramon Matas Navarro, Matthew Mattina, Paul N. Whatmough, and Venkatesh Saligrama. Federated learning based on dynamic regularization. In International Conference on Learning Representations (ICLR), 2021. URL https://openreview.net/ forum?id=B7v4QMR6Z9w.

Amazon Web Services. Amazon EC2 on-demand pricing: Data transfer. https://aws.amazon. com/ec2/pricing/on-demand/, 2024. Accessed 2024.

Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. Federated learning with personalization layers. arXiv preprint arXiv:1912.00818, 2019. URL https://arxiv.org/abs/1912.00818.

Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2–3):235–256, 2002. doi: 10.1023/A:1013689704352.

Andrea Augello, Ashish Gupta, Giuseppe Lo Re, and Sajal K. Das. Tackling selfish clients in federated learning. In ECAI 2024 – 27th European Conference on Artificial Intelligence, volume 392 of Frontiers in Artificial Intelligence and Applications, pages 1888–1895. IOS Press, 2024. doi: 10.3233/FAIA240702.

Mary E. Charlson, Peter Pompei, Kathy L. Ales, and C. Ronald MacKenzie. A new method of classifying prognostic comorbidity in longitudinal studies: development and validation. Journal of Chronic Diseases, 40(5):373–383, 1987. doi: 10.1016/0021-9681(87)90171-8.

Matthew G. Crowson, Dana Moukheiber, Aldo Robles Arévalo, Barbara D. Lam, Sreekar Mantena, Aakanksha Rana, Deborah Goss, David W. Bates, and Leo Anthony Celi. A systematic review of federated learning applications for biomedical data. PLOS Digital Health, 1(5):e0000033, 2022. doi: 10.1371/journal.pdig.0000033.

Marco Cuturi and Mathieu Blondel. Soft-DTW: a differentiable loss function for time-series. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, pages 894–903. PMLR, 2017. URL https: //proceedings.mlr.press/v70/cuturi17a.html.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/ paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html.

Christoph Düsing and Philipp Cimiano. Distribution-controlled client selection to improve federated learning strategies. In Machine Learning and Principles and Practice ofKnowledge Discovery in Databases (ECML PKDD 2024 Workshops), Communications in Computer and Information Science, pages 299–313. Springer, 2026. doi: 10.1007/978-3-032-25314-9\_21. Preprint: arXiv:2509.20877.

João Gama, Indre Žliobait ˙ e, Albert Bifet, Mykola Pechenizkiy, and Abdelhamid Bouchachia. A survey˙ on concept drift adaptation. ACM Computing Surveys, 46(4):1–37, 2014. doi: 10.1145/2523813.

Amirata Ghorbani and James Zou. Data Shapley: Equitable valuation of data for machine learning. In Proceedings ofthe 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings of Machine Learning Research, pages 2242–2251. PMLR, 2019. URL https: //proceedings.mlr.press/v97/ghorbani19c.html.

Abhijit Guha Roy, Shayan Siddiqui, Sebastian Pölsterl, Nassir Navab, and Christian Wachinger. BrainTorrent: A peer-to-peer environment for decentralized federated learning. arXiv preprint arXiv:1905.06731, 2019. doi: 10.48550/arXiv.1905.06731.

Rahul Haripriya, Nilay Khare, and Manish Pandey. Privacy-preserving federated learning for collaborative medical data mining in multi-institutional settings. Scientific Reports, 15(1):12482, 2025. doi: 10.1038/s41598-025-97565-4.

István Heged˝us, Gábor Danner, and Márk Jelasity. Decentralized learning works: An empirical comparison of gossip learning and federated learning. Journal of Parallel and Distributed Computing, 148:109–124, 2021. doi: 10.1016/j.jpdc.2020.10.006.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal ofthe American Statistical Association, 58(301):13–30, 1963. doi: 10.1080/01621459.1963.10500830.

Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335, 2019. URL https://arxiv.org/abs/1909.06335.

Cristovão Iglesias Jr., Sidney Alves de Outeiro, Claudio Miceli de Farias, and Miodrag Bolic. Two students: Enabling uncertainty quantification in federated learning clients. In NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty, 2024. URL https://openreview. net/forum?id=eS9xH4vHEe.

Alistair E.W. Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J. Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10(1):1, 2023. doi: 10.1038/s41597-022-01899-x.

Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. A study of BFLOAT16 for deep learning training. arXiv preprint arXiv:1905.12322, 2019. URL https://arxiv.org/abs/1905.12322.

Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020. doi: 10.1017/9781108571401.

Ming Li, Pengcheng Xu, Junjie Hu, Zeyu Tang, and Guang Yang. From challenges and pitfalls to recommendations and opportunities: Implementing federated learning in healthcare. Medical Image Analysis, 101:103497, 2025. doi: 10.1016/j.media.2025.103497. arXiv:2409.09727.

Qinbin Li, Bingsheng He, and Dawn Song. Model-contrastive federated learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10713– 10722, 2021. doi: 10.1109/CVPR46437.2021.01057.

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Fed erated optimization in heterogeneous networks. In Proceedings ofMachine Learning and Systems (MLSys), volume 2, pages 429–450, 2020. URL https://proceedings.mlsys.org/paper\_ files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html.

Edvin Listo Zec, Johan Östman, Olof Mogren, and Daniel Gillblad. Efficient node selection in private personalized decentralized learning. In Proceedings of the 5th Northern Lights Deep Learning Conference (NLDL), volume 233 of Proceedings of Machine Learning Research, pages 244–250. PMLR, 2024. URL https://proceedings.mlr.press/v233/zec24a.html. arXiv:2301.12755.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), volume 54 of Proceedings of Machine Learning Research, pages 1273–1282. PMLR, 2017. URL https: //proceedings.mlr.press/v54/mcmahan17a.html.

Gergely D. Németh, Eros Fanì, Yeat Jeng Ng, Barbara Caputo, Miguel Ángel Lozano, Nuria Oliver, and Novi Quadrianto. FedDiverse: Tackling data heterogeneity in federated learning with diversitydriven client selection. In 2025 3rd International Conference on Federated Learning Technologies and Applications (FLTA), pages 432–440. IEEE, 2025. doi: 10.1109/FLTA67013.2025.11336421. arXiv:2504.11216.

Md. Rahad, Ruhan Shabab, Mohd. Sultan Ahammad, Md. Mahfuz Reza, Amit Karmaker, and Md. Abir Hossain. KL-FedDis: A federated learning approach with distribution information sharing using Kullback-Leibler divergence for non-IID data. Neuroscience Informatics, 5(1): 100182, 2025. doi: 10.1016/j.neuri.2024.100182.

Leyla Rahimli, Feras M. Awaysheh, Sawsan Al Zubi, and Sadi Alawadi. Federated learning drift detection: An empirical study on the impact of concept and data drift. In 2024 2nd International Conference on Federated Learning Technologies and Applications (FLTA), pages 241–250. IEEE, 2024. doi: 10.1109/FLTA63145.2024.10839814.

Iftekhar Rahman, Nisal Hemadasa, Dominik Kaaser, Pierre-Alexandre Murena, and Stefan Schulte. Detect, adapt, overcome: Mitigating concept drift in federated learning. In 2025 3rd International Conference on Federated Learning Technologies and Applications (FLTA), pages 17–24. IEEE, 2025. doi: 10.1109/FLTA67013.2025.11336319.

Priyesh Ranjan, Ashish Gupta, Federico Corò, and Sajal K. Das. Securing federated learning against overwhelming collusive attackers. In GLOBECOM 2022 – 2022 IEEE Global Communications Conference, pages 1448–1453. IEEE, 2022. doi: 10.1109/GLOBECOM48099.2022.10000830. arXiv:2209.14093.

Finn Saile, Julius Thomas, Dominik Kaaser, and Stefan Schulte. Client-side adaptation to concept drift in federated learning. In 2024 2nd International Conference on Federated Learning Technologies and Applications (FLTA), pages 71–78. IEEE, 2024. doi: 10.1109/FLTA63145.2024.10840058.

Navid Seidi, Satyaki Roy, and Sajal K. Das. Enhancing federated survival analysis through peerdriven client reputation in healthcare. arXiv preprint arXiv:2505.16190, 2025. doi: 10.48550/ arXiv.2505.16190.

Pranava Singhal, Shashi Raj Pandey, and Petar Popovski. Greedy Shapley client selection for communication-efficient federated learning. IEEE Networking Letters, 6(2):134–138, 2024. doi: 10.1109/LNET.2024.3363620.

Georgios Syros, Gokberk Yar, Simona Boboila, Cristina Nita-Rotaru, and Alina Oprea. Backdoor attacks in peer-to-peer federated learning. ACM Transactions on Privacy and Security, 28(1):1–28, 2024. doi: 10.1145/3691633. arXiv:2301.09732.

Chuhan Wu, Fangzhao Wu, Lingjuan Lyu, Yongfeng Huang, and Xing Xie. Communication-efficient federated learning via knowledge distillation. Nature Communications, 13(1):2032, 2022. doi: 10.1038/s41467-022-29763-x.

Wenchao Xia, Tony Q. S. Quek, Kun Guo, Wanli Wen, Howard H. Yang, and Hongbo Zhu. Multiarmed bandit-based client scheduling for federated learning. IEEE Transactions on Wireless Communications, 19(11):7108–7123, 2020. doi: 10.1109/TWC.2020.3008091.

Mengwei Yang, Ismat Jarin, Baturalp Buyukates, Salman Avestimehr, and Athina Markopoulou. Maverick-Aware Shapley valuation for client selection in federated learning. arXiv preprint arXiv:2405.12590, 2024. doi: 10.48550/arXiv.2405.12590.

Guanhua Ye, Jifeng He, Weiqing Wang, Zhe Xue, Feifei Kou, and Yawen Li. WPFed: Web-based personalized federation for decentralized systems. arXiv preprint arXiv:2410.11378, 2024. URL https://arxiv.org/abs/2410.11378v1. version 1.

Andy B. Yoo, Morris A. Jette, and Mark Grondona. SLURM: Simple Linux Utility for Resource Management. In Job Scheduling Strategies for Parallel Processing (JSSPP), pages 44–60. Springer Berlin Heidelberg, 2003. doi: 10.1007/10968987\_3.

Ye Yuan, Jun Liu, Dou Jin, Zuogong Yue, Tao Yang, Ruijuan Chen, Maolin Wang, Lei Xu, Feng Hua, Yuqi Guo, Xiuchuan Tang, Xin He, Xinlei Yi, Dong Li, Wenwu Yu, Hai-Tao Zhang, Tianyou Chai, Shaochun Sui, and Han Ding. DeceFL: A principled fully decentralized federated learning framework. National Science Open, 2(1):20220043, 2023. doi: 10.1360/nso/20220043.

Yuhao Zhou, Minjia Shi, Yuxin Tian, Qing Ye, and Jiancheng Lv. DeFTA: A plug-and-play peer-topeer decentralized federated learning framework. Information Sciences, 670:120582, 2024. doi: 10.1016/j.ins.2024.120582.

## A Empirical Evidence of Concept Drift on MIMIC-IV

To support the concept-drift premise empirically, we compared the distribution of two routinely charted ICU vital signs, heart rate and respiratory rate, between two non-overlapping calendar eras of MIMIC-IV v3.1. The era of each stay is the published anchor\_year\_group field of the patient, which is the true three-year admission era and is unaffected by the per-subject date shift; we use the earliest era, 2008 to 2010, and the latest, 2020 to 2022. For each ICU stay we take the mean of all charted values of the vital during the stay, so that each stay contributes one observation, and we remove stay means that lie more than three standard deviations from the per-era mean. Heart rate has 29,786 stays in the earlier era and 10,740 in the later one, with means of 84.9 and 83.9 beats per minute; respiratory rate has 29,885 and 10,742 stays, with means of 19.3 and 19.4 breaths per minute and a markedly narrower spread in the later era (standard deviation 4.2 against 5.7). Welch two-sample t-tests give $t = 5 . 4 5 , p = 5 \times 1 0 ^ { - 8 }$ for heart rate and $t = - 2 . 0 7 , p = 0 . 0 3 8$ for respiratory rate. The shifts are small in the mean but significant at this sample size, and the change in spread for respiratory rate indicates a change in charting or patient mix rather than a mean shift alone, which is the kind of distributional drift the per-window treatment of $P _ { t } ^ { ( i ) }$ in Sec. 3 is designed to absorb.

![](images/2f52e28039e32bf7f8678d369a62c6ea9252be8f0e479865dbb0c388695828a0.jpg)

![](images/f0d5b47c458f10eac8efa40c4218b421e3af4ebe584a93c6aab977d1ede51b8b.jpg)  
Figure 4: Histograms of per-stay mean heart rate (panel a) and respiratory rate (panel b) for ICU stays in two non-overlapping eras of MIMIC-IV v3.1, 2008 to 2010 and 2020 to 2022, defined by the published anchor\_year\_group field. Stay means beyond three standard deviations from the per-era mean are removed. Vertical dashed lines mark the per-era means. Welch t-tests give $t = 5 ,$ .45, $\dot { p } = 5 \times 1 0 ^ { - 8 }$ for heart rate and $t = \dot { - } 2 . 0 7 , p = 0 . 0 3 8$ for respiratory rate, evidencing distributional shift across eras in routinely collected measurements and motivating the per-window adaptive treatment of $\mathbf { \Psi } _ { P _ { t } ^ { ( i ) } } ^ { ( i ) }$ in Sec. 3.

## B Algorithm Pseudocode

The complete round-level pseudocode for ABPS is given in Algorithm 1, deferred from the main text to save space. The algorithm operationalizes the four Bayesian-core components of Sec. 4 (goal-aware metadata pre-filter, ϵ-greedy UCB ranking, propose-reject topology with explicit rest, and the Beta-Bernoulli posterior update), and is referenced from the regret proof in Appendix C.2 (lines 13 and 16).

## C Full Proofs

This appendix gives the full proofs of Theorems 1 and 2 and Lemma 1. Throughout, we take Assumptions A1-A3 of Sec. 5 as given: bounded utility $( \bar { \phi } _ { j  i } \in [ 0 , 1 ] )$ , conditional independence of the per-round observations given $\mu _ { i \to j } ^ { \star }$ , and quasi-stationary drift $| \mathbb { E } [ \bar { \phi } _ { j  i } ^ { ( k ) } ] - \mu _ { i  j } ^ { \star } | \leq \epsilon _ { m }$ within any window $\Delta t _ { m }$

## C.1 Proof of Theorem 1 (Beta-belief Concentration)

Let $\begin{array} { r } { S _ { K } \ = \ \sum _ { k = 1 } ^ { K } \bar { \phi } _ { j \to } ^ { ( k ) } , } \end{array}$ and $\bar { \mu } _ { K } ~ = ~ S _ { K } / K$ . The Beta posterior after K updates from prior $\mathrm { B e t a } ( \alpha _ { 0 } , \beta _ { 0 } )$ is $\tilde { \mathrm { B e t a } ( \alpha _ { 0 } + S _ { K } , \beta _ { 0 } + K - S _ { K } ) }$ , with posterior mean

$$
\hat { \mu } _ { K } = \frac { \alpha _ { 0 } + S _ { K } } { \alpha _ { 0 } + \beta _ { 0 } + K } = \frac { \alpha _ { 0 } + K \bar { \mu } _ { K } } { \alpha _ { 0 } + \beta _ { 0 } + K } .\tag{11}
$$

Algorithm 1 Adaptive Bayesian Partner Selection (ABPS)   
Require: centers $\overline { { \{ 1 , \ldots , n \} } }$ , rounds $\{ t _ { 1 } , \ldots , t _ { K } \}$ , rank threshold $\kappa ,$ exploration weight γ, exploration probability ϵ, acceptance threshold   
τ , similarity threshold $\tau _ { \mathrm { s i m } } ,$ Beta prior $( \alpha _ { 0 } , \beta _ { 0 } )$   
Ensure: Updated local parameters $\theta _ { i }$ and posterior beliefs $( \alpha _ { i \to j } , \beta _ { i \to j } )$   
1: Initialization:   
2: for each center i do   
3: Initialize local model $\theta _ { i }$ and metadata vector $\mathbf { v } _ { i }$   
4: for each center $j \neq i \mathbf { d o }$   
5: $( \alpha _ { i  j } , \beta _ { i  j } )  ( \alpha _ { 0 } , \beta _ { 0 } ) , n _ { i  j }  0$   
6: end for   
7: end for   
8: for each round $k = 1 , \ldots , K$ do   
9: for each center i do   
10: $\mathcal { C } _ { i } \gets \{ j : f _ { i } ( \mathbf { v } _ { i } , \mathbf { v } _ { j } ) \geq \tau _ { \mathrm { s i m } } \}$ ▷ goal-aware pre-filter, eq. (3)   
11: Compute $\mathrm { U C B } _ { j  i } ( \bar { t } _ { k } )$ via eq. (7) for all $j \in \mathcal { C } _ { i }$   
12: With prob. ϵ, swap top-ranked candidate with a uniformly random one in $\mathcal { C } _ { i }$   
13: $\mathcal { P } _ { i }  \emptyset$   
14: for j in descending UCB order do   
15: $\mathbf { \bar { i } } \mathbf { f } \left| \mathcal { P } _ { i } \right| \geq \kappa$ then break   
16: end if   
17: $\mathbf { i f } \ U \mathbf { C B } _ { i \to j } \left( t _ { k } \right) \geq \tau _ { \mathrm { a c c } }$ and $| \mathcal { P } _ { j } |$ < κ then   
18: $\mathcal { P } _ { i }  \mathcal { P } _ { i } \cup \{ \overline { { j } } \} , \mathcal { P } _ { j }  \mathcal { P } _ { j } \overset { \cdot } { \cup } \{ i \}$   
19: end if   
20: end for   
21: end for   
22: for each center i with $\mathcal { P } _ { i } \neq \mathcal { O } ~ ,$ do   
23: $\theta _ { i } \gets \mathrm { F e d A v g } ( \{ \theta _ { i } \} \cup \{ \theta _ { j } : j \in \mathscr { P } _ { i } \} )$ ▷ exchange; optionally personalize head and quantize wire-format (Sec. 6.4)   
24: Compute Shapley marginals $\bar { \{ { \phi } _ { j  i } \} } _ { j \in \mathcal { P } _ { i } }$ via eq. (4)   
25: for each $j \in \mathcal { P } _ { i }$ do   
26: $\bar { \phi } _ { j \to i }  \mathrm { c l i p } _ { [ 0 , 1 ] } \big ( ( \phi _ { j \to i } - \phi _ { \mathrm { m i n } } ) / ( \phi _ { \mathrm { m a x } } - \phi _ { \mathrm { m i n } } ) \big )$   
27: $( \alpha _ { i  j } , \beta _ { i  j } )  ( \alpha _ { i  j } + \bar { \phi } _ { j  i } , \beta _ { i  j } + 1 - \bar { \phi } _ { j  i } )$   
28: $n _ { i  j } \stackrel { \cdot } {  } n _ { i  j } + 1$   
29: end for   
30: end for   
31: Centers with $\mathcal { P } _ { i } = \mathcal { D }$ rest this round (intentional isolation; cf. Lemma 1)   
32: end for

Apply the triangle inequality with the empirical mean $\bar { \mu } _ { K }$ as pivot:

$$
\left| \hat { \mu } _ { K } - \mu _ { i \to j } ^ { \star } \right| \le \underbrace { \left| \hat { \mu } _ { K } - \bar { \mu } _ { K } \right| } _ { \mathrm { ( I ) ~ p r i o r ~ p u l l } } + \underbrace { \left| \bar { \mu } _ { K } - \mathbb { E } [ \bar { \mu } _ { K } ] \right| } _ { \mathrm { ( I I ) ~ s t a t i s t i c a l ~ n o i s e } } + \underbrace { \left| \mathbb { E } [ \bar { \mu } _ { K } ] - \mu _ { i \to j } ^ { \star } \right| } _ { \mathrm { ( I I I ) ~ d r i f t ~ b i a s } } .\tag{12}
$$

Bounding (I). From (11),

$$
\hat { \mu } _ { K } - \bar { \mu } _ { K } \ = \ \frac { \alpha _ { 0 } + K \bar { \mu } _ { K } - ( \alpha _ { 0 } + \beta _ { 0 } + K ) \bar { \mu } _ { K } } { \alpha _ { 0 } + \beta _ { 0 } + K } \ = \ \frac { \alpha _ { 0 } ( 1 - \bar { \mu } _ { K } ) - \beta _ { 0 } \bar { \mu } _ { K } } { \alpha _ { 0 } + \beta _ { 0 } + K } .
$$

Because $\bar { \mu } _ { K } \in [ 0 , 1 ]$ , the numerator is bounded in absolute value by max $( \alpha _ { 0 } , \beta _ { 0 } ) \le \alpha _ { 0 } + \beta _ { 0 }$ . Hence

$$
\mathrm { ( I ) } ~ \leq ~ \frac { \alpha _ { 0 } + \beta _ { 0 } } { \alpha _ { 0 } + \beta _ { 0 } + K } .\tag{13}
$$

This is the deterministic “prior decay” term in (8): it shrinks at rate $\Theta ( 1 / K )$ regardless of randomness in the data.

Bounding (II). By A1 every observation lies in [0, 1], and by A2 the observations are conditionally independent given $\mu _ { i \to j } ^ { \star }$ . Hoeffding’s inequality Hoeffding [1963] for the mean of K independent [0, 1]-valued variables gives

$$
\begin{array} { r } { \mathrm { P r } \big [ | \bar { \mu } _ { K } - \mathbb { E } [ \bar { \mu } _ { K } ] | \ge t \big ] \ \le \ 2 \exp \big ( - 2 K t ^ { 2 } \big ) . } \end{array}\tag{14}
$$

Setting the right-hand side equal to $\delta$ and solving for t yields $t = \sqrt { \log ( 2 / \delta ) / ( 2 K ) }$ , so with probability at least $1 - \delta$

$$
\mathrm { ( I I ) } \ \leq \ { \sqrt { \frac { \log ( 2 / \delta ) } { 2 K } } } .\tag{15}
$$

Bounding (III). By A3, every per-round expectation is within $\epsilon _ { m }$ of $\mu _ { i \to j } ^ { \star } \colon$

$$
| \mathbb { E } [ \bar { \mu } _ { K } ] - \mu _ { i \to j } ^ { \star } | \ = \ \left| \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } [ \bar { \phi } _ { j \to i } ^ { ( k ) } ] - \mu _ { i \to j } ^ { \star } \right| \ \le \ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \big | \mathbb { E } [ \bar { \phi } _ { j \to i } ^ { ( k ) } ] - \mu _ { i \to j } ^ { \star } \big | \ \le \ \epsilon _ { m } .
$$

Hence $( \mathrm { I I I } ) \le \epsilon _ { m }$

Combining. Substituting (I), (II), (III) into (12) yields, with probability at least $1 - \delta .$

$$
| \hat { \mu } _ { K } - \mu _ { i \to j } ^ { \star } | \le \sqrt { \frac { \log ( 2 / \delta ) } { 2 K } } + \frac { \alpha _ { 0 } + \beta _ { 0 } } { \alpha _ { 0 } + \beta _ { 0 } + K } + \epsilon _ { m } ,
$$

which is exactly (8). Convergence in probability follows: as $K  \infty , ( \mathrm { I } )$ and (II) tend to 0, so $\hat { \mu } _ { K } \stackrel { p } {  } \mu _ { i  j } ^ { \star }$ provided the drift $\epsilon _ { m }  0 \ :$

## C.2 Proof of Theorem 2 (Sublinear Partner-Selection Regret)

Inside a stationary window $( \epsilon _ { m } = 0 )$ , the per-center partner-selection problem is a stochastic multiarmed bandit over the post-filter candidate set $\mathcal { C } _ { i } \subseteq \{ \bar { 1 } , \ldots , n - 1 \}$ in which ABPS pulls κ arms per round (the κ acceptances) rather than one. The goal-aware pre-filter of Sec. 4.1 restricts the arm set before the bandit sees it, and we analyze this case directly below. We bound the cumulative regret

$$
\mathcal R ( T ) = T \sum _ { j \in \mathrm { t o p } \cdot \kappa } \mu _ { j } ^ { \star } \ - \ \mathbb E \left[ \sum _ { t = 1 } ^ { T } \sum _ { j \in \mathcal P _ { i } ^ { ( t ) } } \mu _ { j } ^ { \star } \right]
$$

by analyzing the greedy and exploratory phases separately.

Greedy phase (probability $1 - \epsilon )$ . With probability $1 - \epsilon \mathrm { \bf A B P S }$ ranks candidates by their UCB1 score

$$
\mathrm { U C B } _ { j \to i } ( t ) = \hat { \mu } _ { i \to j } ( t ) + \gamma \sqrt { \frac { 2 \log t } { n _ { i \to j } ( t ) + 1 } } ,
$$

and proposes greedily down the list until κ acceptances accrue. Decompose the per-round greedy regret as the sum of κ single-arm regrets, indexed by the rank position $r = 1 , \ldots , \kappa$ of each accepted proposal. Each rank-r slot is a single-arm UCB1 problem played against the residual candidate pool. By the canonical UCB1 analysis Auer et al. [2002], the cumulative regret of UCB1 with confidence radius $\gamma = \sqrt { 2 }$ over T rounds satisfies

$$
\mathcal { R } _ { \mathrm { U C B 1 } } ( T ) \leq \sum _ { j : \Delta _ { i } > 0 } \left( \frac { 8 \log T } { \Delta _ { j } } + \Delta _ { j } \left( 1 + \frac { \pi ^ { 2 } } { 3 } \right) \right) ,\tag{16}
$$

where $\Delta _ { j } = \mu _ { ( 1 ) } ^ { \star } - \mu _ { ( j ) } ^ { \star }$ are the suboptimality gaps. Summing κ such bounds gives

$$
\mathcal { R } _ { \mathrm { g r e e d y } } ( T ) \leq \kappa \sum _ { j : \Delta _ { j } > 0 } \biggl ( \frac { 8 \log T } { \Delta _ { j } } + \Delta _ { j } \biggl ( 1 + \frac { \pi ^ { 2 } } { 3 } \biggr ) \biggr ) .\tag{17}
$$

Two refinements only decrease this bound and so are absorbed into (17). (a) The receiver-side acceptance check $\mathrm { U } \mathrm { \dot { C } B } _ { i \to j } ( t ) \geq \tau _ { \mathrm { a c c } }$ in the inner-if of Algorithm 1 cannot create new pulls of suboptimal arms, only suppress them. (b) The Beta posterior is sharper than the empirical mean used by vanilla UCB1 (it shrinks toward the prior at rate $1 / K$ , matching term (I) in Theorem 1), so the exploration radius is in fact tighter than (16) assumes.

Exploratory phase (probability ϵ). With probability ϵ ABPS replaces the top-ranked candidate with a uniformly random peer (the ϵ-greedy swap step in Algorithm 1). Its expected per-round regret contribution is at most $\mathbb { E } _ { j \sim \operatorname { U n i f } ( \mathcal { C } _ { i } ) } [ \Delta _ { j } ]$ , so summing over $\check { T }$ rounds

$$
\mathcal { R } _ { \mathrm { e x p l o r e } } ( T ) \ \leq \ \epsilon T \cdot \mathbb { E } _ { j \sim \mathrm { U n i f } ( \mathcal { C } _ { i } ) } [ \Delta _ { j } ] .\tag{18}
$$

Combining. Adding (17) and (18) yields the bound (9) of the main text:

$$
\mathcal { R } ( T ) \leq \kappa \sum _ { j : \Delta _ { j } > 0 } \left( \frac { 8 \log T } { \Delta _ { j } } + \Delta _ { j } \left( 1 + \frac { \pi ^ { 2 } } { 3 } \right) \right) + \epsilon T \cdot \mathbb { E } _ { j } [ \Delta _ { j } ] .
$$

Recovering the $\mathcal { O } ( \log T )$ rate. Two annealing schedules suffice. (a) A constant $\epsilon = c / \sqrt { T }$ leaves an $\mathcal { O } ( \sqrt { T } )$ exploratory residual that is sublinear but slower than the greedy term. (b) A time-varying schedule $\epsilon _ { t } = \operatorname* { m i n } ( 1 , c / t )$ , in the spirit of Auer et al. [2002], gives $\textstyle \sum _ { t = 1 } ^ { T } \epsilon _ { t } = { \mathcal { O } } ( \log T )$ and recovers the full $\mathcal { O } ( \kappa \log T )$ rate. In either case, the regret is sublinear in ${ \bar { T } } ,$ so the average per-round regret tends to zero.

Filtered-arm-set refinement. When the goal-aware filter of Sec. 4.1 restricts $\mathcal { C } _ { i }$ to the top-ρ quantile of peers under $f _ { i } ,$ the bandit plays on $\tilde { \mathcal { C } } _ { i } = \mathcal { C } _ { i } \cap \{ j : f _ { i } ( \mathbf { v } _ { i } , \mathbf { v } _ { j } ) \geq \tau _ { \mathrm { s i m } } \}$ . Two changes propagate to the bound (9).

(i) Reduced explore-greedy regret. The sums over $\{ j : \Delta _ { j } > 0 \}$ and $\mathbb { E } _ { j \sim \operatorname { U n i f } ( \mathcal { C } _ { i } ) } [ \Delta _ { j } ]$ are replaced by sums over $\tilde { \mathcal { C } } _ { i }$ , which is a strict subset, and both terms can only decrease.

(ii) Optimal-arm coverage bias. If the oracle-best peer $j ^ { \star }$ fails to clear $f _ { i } ( \cdot , \cdot ) \geq \tau _ { \mathrm { s i m } }$ , the bandit plays on a suboptimal set and incurs an additive bias of $\Delta _ { j ^ { \star } } \cdot T$ against the oracle baseline. This worst-case term is controlled by a standard top-k coverage guarantee: if the descriptor v is L-Lipschitzinformative about true utility (i.e. $| \mu _ { j } ^ { \star } - \mu _ { j ^ { \prime } } ^ { \star } | \leq L \| \mathbf { v } _ { j } - \mathbf { v } _ { j ^ { \prime } } \| )$ , then the probability $\mathbb { P } [ j ^ { \star } \notin \tilde { \mathcal { C } } _ { i } ] \le$ $\exp ( - c \rho | \mathcal { C } _ { i } | )$ for some $c > 0$ depending on $L ,$ so the expected additional regret is $\mathcal { O } ( T \cdot \exp ( - c \rho n ) )$ which is negligible for reasonable $\rho$ and $n .$ In our experiments $\rho = 0 . 2 5 , n = 2 3 0$ gives $\rho n \approx 5 8$ , the filter recovers the optimal arm with probability $> 1 - 1 0 ^ { - 2 5 }$ under any non-trivial descriptor-to-utility Lipschitz constant.

The net effect is that (9) remains valid with $| { \mathcal { C } } _ { i } |$ replaced by $| \tilde { \mathcal { C } } _ { i } | \approx \rho n$ . The $\mathcal { O } ( \kappa \log T )$ asymptotic rate is preserved, and the constants improve proportionally to $\rho .$

## C.3 Proof of Lemma 1 (Optimality of Intentional Isolation)

Let $V _ { i } ( \mathcal { P } )$ denote $i \gamma _ { \mathrm { s } }$ expected one-step utility (e.g., held-out AUROC) after the round, conditioned on its active peer set being $\mathcal { P }$ . Set $V _ { i } ^ { \mathrm { r e s t } } : = V _ { i } ( { \emptyset } )$ and $V _ { i } ^ { \mathrm { c o l l } } ( { \mathcal { P } } ) : = V _ { i } ( { \mathcal { P } } )$ for $\mathcal { P } \neq \emptyset$

Decomposition via Shapley efficiency. The Shapley value (4) satisfies the efficiency axiom: for any coalition $\mathcal { P }$

$$
\sum _ { j \in \mathcal { P } } \phi _ { j  i } \ : = \ : U _ { i } ( \mathcal { P } ) - U _ { i } ( \mathcal { D } ) ,\tag{19}
$$

where $U _ { i } ( \cdot )$ is the validation utility used to compute the Shapley value. Since $V _ { i }$ and $U _ { i }$ coincide in expectation under our protocol (both are held-out AUROC of the post-aggregation model), taking expectations in (19) gives

$$
V _ { i } ^ { \mathrm { c o l l } } ( { \mathcal { P } } ) - V _ { i } ^ { \mathrm { r e s t } } = \sum _ { j \in { \mathcal { P } } } \mathbb { E } [ \phi _ { j \to i } ] .\tag{20}
$$

Translating to the clipped scale. Within the clipping range, $\bar { \phi } _ { j  i }$ relates to $\phi _ { j  i }$ via the affine map (5),

$$
\bar { \phi } _ { j \to i } = { \frac { \phi _ { j \to i } - \phi _ { \operatorname* { m i n } } } { \phi _ { \operatorname* { m a x } } - \phi _ { \operatorname* { m i n } } } } \Longleftrightarrow \phi _ { j \to i } = \left( \phi _ { \operatorname* { m a x } } - \phi _ { \operatorname* { m i n } } \right) \bar { \phi } _ { j \to i } + \phi _ { \operatorname* { m i n } } ,
$$

so $\mathbb { E } [ \phi _ { j \to i } ] = ( \phi _ { \operatorname* { m a x } } - \phi _ { \operatorname* { m i n } } ) \mu _ { i \to j } ^ { \star } + \phi _ { \operatorname* { m i n } }$ . The negative-transfer threshold $c _ { \mathrm { n { e g } } }$ is the value of $\mu _ { i \to j } ^ { \star }$ at which one-peer collaboration breaks even, $V _ { i } ^ { \mathrm { c o l l } } ( \{ j \} ) = V _ { i } ^ { \mathrm { r e s t } } , \mathrm { i . e . } \mathbb { E } [ \phi _ { j  i } ] = 0$ . Solving,

$$
c _ { \mathrm { n e g } } ~ = ~ { \frac { - \phi _ { \mathrm { m i n } } } { \phi _ { \mathrm { m a x } } - \phi _ { \mathrm { m i n } } } } .\tag{21}
$$

For the paper’s choice $( \phi _ { \mathrm { m i n } } , \phi _ { \mathrm { m a x } } ) = ( - 0 . 1 , + 0 . 1 )$ this gives $c _ { \mathrm { n e g } } = 0 . 5$ , which coincides with $\tau _ { \mathrm { a c c } }$ used in the algorithm. The map $\mu _ { i \to j } ^ { \star } < c _ { \mathrm { n e g } } \iff \mathbb { E } [ \phi _ { j \to i } ] < \bar { 0 }$ is therefore exact.

Strict dominance of resting. Suppose $\mu _ { i \to j } ^ { \star } < c _ { \mathrm { n e g } }$ for every $j \in \mathcal { P }$ . Then $\mathbb { E } [ \phi _ { j  i } ] < 0$ for all $j \in \mathcal P$ , so by (20),

$$
V _ { i } ^ { \mathrm { c o l l } } ( \mathcal { P } ) - V _ { i } ^ { \mathrm { r e s t } } = \sum _ { j \in \mathcal { P } } \mathbb { E } [ \phi _ { j  i } ] < 0 ,
$$

which is exactly the claimed inequality (10), $V _ { i } ^ { \mathrm { r e s t } } > V _ { i } ^ { \mathrm { c o l l } } ( \mathcal { P } )$

Refinement under approximate Shapley. When the Shapley values are estimated via truncated Monte-Carlo (TMC, the default for $\kappa > 5 ,$ , cf. Sec. 5), the per-peer estimator carries an additive bias bounded by some $\eta > 0$ . Equation (20) then becomes

$$
V _ { i } ^ { \mathrm { c o l l } } ( \mathcal { P } ) - V _ { i } ^ { \mathrm { r e s t } } \ : = \ : \sum _ { j \in \mathcal { P } } \mathbb { E } [ \phi _ { j  i } ] - g ( | \mathcal { P } | ) ,
$$

with $| g ( | \mathcal { P } | ) | \leq \eta | \mathcal { P } |$ . Strict dominance survives whenever the true expected gain margin exceeds the approximation slack, $\begin{array} { r } { \sum _ { j } \mathbb { E } [ \phi _ { j  i } ] < - \eta | \mathcal { P } | } \end{array}$ , which is satisfied with margin η to spare under the strict inequality $\mu _ { i \to j } ^ { \star } < c _ { \mathrm { n e g } }$ . This is the form quoted in the proof sketch of Sec. 5.3.

Bandwidth saving. Specializing the cost equation (2) to a single round with $\mathcal { P } _ { i } ^ { \mathrm { r e s t } } = \emptyset$ gives a bandwidth of $0 ,$ while collaborating with $| \bar { \mathcal { P } } |$ peers costs $2 | \mathcal { P } | \overset { \cdot } { c } _ { \mathrm { e g r e s s } } | W |$ (one upload and one download per peer per round). The saving is therefore exactly $\dot { C } _ { i } ^ { \mathrm { r e s t } } = 2 | \mathcal { P } | \dot { c } _ { \mathrm { e g r e s s } } | \dot { W } |$ | per round.

Connection to Theorem 1. Lemma 1 reasons about the oracle threshold $c _ { \mathrm { n { e g } } }$ , but ABPS only sees the noisy posterior estimate $\hat { \mu } _ { K }$ . Combining the lemma with (8): with probability $\geq 1 - \delta$ , the UCB-induced rule rests whenever

$$
\begin{array} { r } { \hat { \mu } _ { K } + \gamma \sqrt { \frac { 2 \log K } { n _ { i  j } + 1 } } < \tau _ { \mathrm { a c c } } = c _ { \mathrm { n e g } } , } \end{array}
$$

where $n _ { i \to j } + 1$ is the per-arm pull count from Eq. (7), which itself grows with K under any nontrivial selection rule. The condition above is implied by $\mu _ { i \to j } ^ { \star } < c _ { \mathrm { n e g } } - [ \sqrt { \log ( 2 / \delta ) / ( 2 K ) } + ( \alpha _ { 0 } +$ $\beta _ { 0 } ) / ( \alpha _ { 0 } + \beta _ { 0 } + K ) + \epsilon _ { m } + \gamma \sqrt { 2 \log K / ( n _ { i  j } + 1 ) } ]$ . As $K  \infty$ and $\epsilon _ { m }  0 .$ , every term in the bracket vanishes (the exploration term shrinks because $n _ { i \to j } + 1$ grows at rate $\Theta ( K )$ for any peer ever pulled by UCB), and the algorithmic rule converges to the oracle rule of Lemma 1.

## Acknowledgments

This work was supported by NSF grants OAC-2609072 (CHAI), SFS-2335969 (MASTER), OAC-2104076 (CANDY), and SATC-2030624 (TAURUS).