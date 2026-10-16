# MARS: A Momentum-Aware Adaptive Rating System for Personalized Assessment

Vikrant Sahu, Khethavath Sunil Naik, and Gagan Raj Gupta

Department of Computer Science and Engineering, Indian Institute of Technology Bhilai, Chhattisgarh, India {vikrantsahu, khethavathn, gagan}@iitbhilai.ac.in

Abstract—Adaptive assessment systems personalize learning by estimating learner proficiency and recommending suitable questions. However, many existing approaches rely on fixed difficulty labels, static rating updates, or black-box prediction models that provide limited control over rating stability, interpretability, and time-constrained routing. This paper proposes MARS, a Momentum-Aware Adaptive Rating System for interpretable adaptive assessment. MARS represents the question bank as a directed weighted knowledge graph and updates learner proficiency using dynamic rating sensitivity, response quality, and momentum-aware learner velocity. We also formulate a timebudgeted mark-maximization objective in which the policy selects questions within a fixed test duration under negative marking. Experiments on ASSISTments 2009 and MARS-SimExam evaluate MARS for prediction, rating stability, and adaptive routing. Results show that MARS provides smoother rating updates than Fixed Elo and achieves the highest score under time-budgeted negative-marking routing, improving the total score by 8.09 over the strongest baseline and reducing overtime violations from 50% to 5.6%. These results suggest that MARS is suitable for real-time, interpretable, and deployment-friendly adaptive assessment.

Index Terms—Adaptive assessment, educational data mining, Elo rating, momentum, knowledge tracing, graph-based recommendation, time-budgeted assessment, learner modeling.

## I. INTRODUCTION

DAPTIVE assessment is an important component of intelligent learning systems because it enables question difficulty, topic coverage, and feedback to be adjusted according to learner performance. In competitive examination preparation and online learning environments, learners differ in proficiency, response speed, consistency, topic-wise weakness, and recovery after mistakes. Therefore, an effective adaptive system should not only select the next question using recent accuracy or fixed difficulty labels, but also estimate evolving learner ability, support remediation, and provide interpretable recommendation logic.

Traditional computerized adaptive testing methods, such as item response theory, provide a strong psychometric foundation but often depend on calibrated item banks and offer limited flexibility when question banks evolve over time [1]– [3]. Rating-based models provide a lightweight alternative by modeling learner–question interactions as a match between learner ability and item difficulty [4]–[6]. However, fixed rating updates can be unstable, particularly for new learners, and may overlook useful response-quality signals such as response time, hint usage, streak behavior, and topic structure. Knowledge tracing models, including Bayesian, deep, and graph-based variants, have shown strong performance in correctness prediction, but they often provide limited transparency for adaptive routing and remediation decisions [7], [13], [22].

To address these limitations, this paper proposes MARS (Momentum-Aware Adaptive Rating System), a graphbased adaptive assessment framework that models the question bank as a directed weighted knowledge graph with same-topic, prerequisite, and cross-topic relationships. MARS updates learner proficiency using dynamic rating sensitivity, response quality, and momentum-based learner velocity, enabling stable and responsive assessment. The framework supports adaptive question routing, weak-topic recovery, and interpretable remediation by selecting questions based on difficulty, learner momentum, graph structure, topic weakness, diversity, and centrality, while guiding learners through related easier concepts before revisiting unresolved topics.

The main contributions of this work are summarized as follows:

• Momentum-aware adaptive rating. We propose MARS, an interpretable online rating framework that extends Elostyle learner modeling with dynamic update sensitivity, response-quality scaling, and momentum-driven learner velocity to stabilize noisy learner-state updates.

• Graph-guided remediation and routing. We design a question-graph-based routing policy that selects items using target difficulty, learner momentum, weak-topic recovery, diversity, and graph structure, enabling adaptive progression and contextual remediation.

• Score-aware time-budgeted assessment. We formulate adaptive question selection as a mark-maximization problem under fixed test duration, and evaluate MARS against rating-based and knowledge-tracing baselines for prediction, stability, and deployment-oriented routing.

## II. RELATED WORK

Computerized adaptive testing is commonly grounded in item response theory (IRT), where learner ability and item parameters are used to estimate the probability of a correct response [1]–[3]. Although IRT provides a strong psychometric foundation, it usually requires calibrated item banks and sufficient historical responses, which can be limiting when question banks evolve. Rating-based models provide a lightweight online alternative by treating learner–item interaction as a match between learner ability and item difficulty [4], [6]. However, fixed Elo-style updates may be unstable for new learners and may not account for response quality, topic weakness, remediation, or graph structure. Glicko-style extensions introduce uncertainty and volatility into rating estimation [5], but they are not designed for educational routing with response-time, hint, and graph-based signals.

Knowledge tracing models estimate learner knowledge from sequential interaction data. Bayesian Knowledge Tracing models mastery as a latent skill state updated using learning, guessing, and slipping parameters [7], [8], while classical factor-based, matrix-factorization, and logistic learner models use prior success and failure counts to estimate skill acquisition [9], [10], [12], [42]. Deep Knowledge Tracing introduced recurrent neural networks for learner-response sequences [13], and later analyses and extensions examined deeper recurrent models, regularized sequence models, memory networks, attention, transformers, and graph neural networks to improve prediction [15]–[22], [26], [36], [43]. Survey work also shows that prediction gains alone do not solve interpretability and deployment constraints in adaptive assessment [27], [28]. These models can achieve strong next-response prediction, but their internal states are often less transparent for real-time routing, remediation, and score-aware item selection.

Graph-based knowledge tracing and educational recommendation methods use relationships among skills, concepts, or questions to improve learner modeling and recommendation [22]–[25]. Recent work further combines graph structure, semantic item representations, and attention mechanisms for knowledge tracing [29]–[35]. Response time and hint usage are also important indicators of response quality and can help distinguish mastery, guessing, and effortful failure [11], [14]. MARS combines these directions by maintaining explicit learner-state variables, including rating, velocity, streak, weak topics, answered items, served items, and remediation status. Unlike prediction-only knowledge tracing models, MARS uses graph structure, response quality, and momentum-aware rating updates not only for learner-state estimation but also for interpretable adaptive routing under time and scoring constraints.

## III. PROBLEM FORMULATION

## A. Learner-Item Interaction Setting

Let $\mathcal { L } = \{ l _ { 1 } , l _ { 2 } , \ldots , l _ { N } \}$ denote a set of learners and let $\mathcal { Q } = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { M } \}$ denote a question bank. Each question $q \in \mathcal { Q }$ is associated with a set of metadata attributes,

$$
\begin{array} { r } { x _ { q } = ( \mathrm { d o m a i n } _ { q } , \mathrm { t o p i c } _ { q } , R _ { q } , m _ { q } , t _ { q } , \mathrm { t y p e } _ { q } ) , } \end{array}\tag{1}
$$

where domai $1 _ { q }$ is the subject or domain, topic is the topic label, $R _ { q }$ is the item rating or difficulty estimate, $m _ { q }$ is the mark value, $t _ { q }$ is the reference solve time, and $\mathrm { t y p e } _ { q }$ denotes the question type.

At interaction step $p ,$ learner $l _ { i }$ receives a question $q _ { i , p }$ and produces a response represented as

$$
r _ { i , p } = ( q _ { i , p } , y _ { i , p } , \Delta t _ { i , p } , h _ { i , p } ) ,\tag{2}
$$

where $y _ { i , p } \in \{ 0 , 1 \}$ is the binary correctness outcome, $\Delta t _ { i , p }$ is the time spent on the item, and $h _ { i , p }$ denotes hint usage or assistance level. The adaptive system observes the response

and updates the learner state before recommending the next question.

## B. Question Knowledge Graph

The question bank is represented as a directed weighted graph,

$$
G = ( V , E , w ) ,\tag{3}
$$

where each node $v \in V$ corresponds to a question $q \in \mathcal { Q }$ , each directed edge $( u , v ) \in E$ represents a pedagogical relation between two questions, and $w ( u , v ) \in [ 0 , 1 ]$ denotes the edge distance or dissimilarity. Lower edge weight indicates stronger pedagogical closeness.

The graph contains three types of edges: same-topic edges, subject-flow edges, and bridge edges. Same-topic edges connect questions from the same topic in increasing or comparable difficulty order. Subject-flow edges connect questions according to prerequisite or curriculum sequence. Bridge edges connect questions across topics or domains when conceptual dependency exists.

For two connected items u and v, the edge weight is defined as

$$
\begin{array} { r l } & { \boldsymbol { w } ( u , \boldsymbol { v } ) = \alpha _ { 1 } d _ { R } ( u , \boldsymbol { v } ) + \alpha _ { 2 } d _ { m } ( u , \boldsymbol { v } ) } \\ & { \qquad + \alpha _ { 3 } d _ { \mathrm { t o p i c } } ( u , \boldsymbol { v } ) + \alpha _ { 4 } d _ { \mathrm { t y p e } } ( u , \boldsymbol { v } ) , } \end{array}\tag{4}
$$

subject to

$$
\sum _ { j = 1 } ^ { 4 } \alpha _ { j } = 1 , \qquad \alpha _ { j } \geq 0 .\tag{5}
$$

Here, $d _ { R }$ is the normalized rating distance, $d _ { m }$ is the normalized mark distance, $d _ { \mathrm { t o p i c } }$ is the topic distance, and $d _ { \mathrm { t y p e } }$ is the question-type distance.

## C. Learner State Representation

At each step $p ,$ the learner is represented by a state vector

$$
s _ { i , p } = ( R _ { i , p } , v _ { i , p } , n _ { i , p } , \sigma _ { i , p } ^ { 2 } , k _ { i , p } , m _ { i , p } , W _ { i , p } , Q _ { i , p } ^ { a n s } , Q _ { i , p } ^ { s r v } , \pi _ { i , p } ^ { r e m } ) ,\tag{6}
$$

where $R _ { i , p }$ is the current learner rating, $v _ { i , p }$ is the learner progression velocity, $n _ { i , p }$ is the number of attempted questions, $\sigma _ { i , p } ^ { 2 }$ is the recent rating variance, $k _ { i , p }$ is the signed correctness streak, and $m _ { i , p }$ is the behavioural momentum label. The set $W _ { i , p }$ stores weak topics, $Q _ { i , p } ^ { a n s }$ stores answered items, $Q _ { i , p } ^ { s r v }$ stores served items, and $\pi _ { i , p } ^ { r e m }$ stores remediation progress.

The weak-topic set is defined as

$$
W _ { i , p } = \{ z \in \mathcal { T } \mid \mathrm { A c c } _ { i , p } ( z ) < \tau _ { w } \} ,\tag{7}
$$

where $\tau$ is the set of topics, $\operatorname { A c c } _ { i , p } ( z )$ is the learner’s accuracy on topic $z ,$ , and if $\tau _ { w }$ is less than 40% is the weak-topic threshold.

## D. Adaptive Question Routing Objective

The main objective is to select the next question $q _ { i , p + 1 } ^ { * }$ that is suitable for the learner’s current state while satisfying graph, difficulty, momentum, and remediation constraints. The recommendation problem is formulated as

$$
q _ { i , p + 1 } ^ { * } = \underset { q \in \mathcal { C } _ { i , p } } { \arg \operatorname* { m a x } } U ( q \mid s _ { i , p } , G ) ,\tag{8}
$$

where $\mathcal { C } _ { i , p }$ is the candidate set and $U ( q \mid s _ { i , p } , G )$ is the adaptive utility score.

For normal progression (non-remediation), the candidate set is selected from the graph neighbourhood of the last attempted question:

$$
\mathcal { C } _ { i , p } ^ { s t d } = \{ q ^ { \prime } \in V : d _ { G } ( q _ { i , p } , q ^ { \prime } ) \leq k _ { h } , q ^ { \prime } \notin Q _ { i , p } ^ { a n s } , q ^ { \prime } \notin Q _ { i , p } ^ { s r v } \} ,\tag{9}
$$

where $d _ { G } ( \cdot , \cdot )$ is graph hop distance and $k _ { h }$ is the maximum hop limit.

For remediation, if the learner has an unresolved missed item $q _ { m } ,$ , the candidate set is restricted to related but easier questions:

$$
\mathcal { C } _ { i , p } ^ { r e m } = \{ q ^ { \prime } \in V : d _ { G } ( q _ { m } , q ^ { \prime } ) \leq h , q ^ { \prime } \notin Q _ { i , p } ^ { a n s } , R _ { q ^ { \prime } } < R _ { q _ { m } } \} .\tag{10}
$$

This ensures that remediation questions are pedagogically connected to the missed item while being easier than the original question.

## E. Time-Budgeted Mark-Maximization Objective

For the time-budgeted setting, the policy selects at most J questions within a duration budget T. A correct response earns $m ( q )$ marks, while an incorrect response receives a penalty of $\rho m ( q )$ . Thus, the expected score of question $q _ { j }$ is

$$
\begin{array} { r l } & { \hat { S } ( q _ { j } \mid s _ { j } ) = \hat { P } ( y _ { j } = 1 \mid s _ { j } , q _ { j } ) m ( q _ { j } ) } \\ & { \quad \quad \quad - ( 1 - \hat { P } ( y _ { j } = 1 \mid s _ { j } , q _ { j } ) ) \rho m ( q _ { j } ) . } \end{array}\tag{11}
$$

The routing objective is

$$
\operatorname* { m a x } _ { \pi } \sum _ { j = 1 } ^ { J } \hat { S } ( q _ { j } \mid s _ { j } ) \quad \mathrm { s . t . } \quad \sum _ { j = 1 } ^ { J } \hat { t } ( q _ { j } \mid s _ { j } ) \leq T .\tag{12}
$$

In the experiments, $J = 5 0 , T = 3 6 0 0$ seconds, and $\rho = 0 . 2 5$

## F. Rating Update Objective

After each response, the system updates the learner rating using an online rating function. Unlike direct Elo-style update rules, MARS first computes a raw rating update and then applies a momentum velocity update:

$$
\Delta R _ { i , p } ^ { r a w } = K _ { i , p } Q _ { i , p } ( S _ { i , p } - E _ { i , p } ) ,\tag{13}
$$

$$
v _ { i , p } = \beta _ { i , p } v _ { i , p - 1 } + ( 1 - \beta _ { i , p } ) \Delta R _ { i , p } ^ { r a w } ,\tag{14}
$$

$$
R _ { i , p + 1 } = R _ { i , p } + \mathrm { c l i p } ( v _ { i , p } , - v _ { m a x } , v _ { m a x } ) .\tag{15}
$$

Here, $K _ { i , p }$ is the dynamic update factor, $S _ { i , p } \in \{ 0 , 1 \}$ is the observed correctness, $Q _ { i , p } = 0 . 8 5 + 0 . 3 0 P _ { i , p }$ is a multiplier based on the response quality score $P _ { i , p } , E _ { i , p }$ is the expected probability of success, $v _ { i , p }$ is the accumulated progression velocity, $\beta _ { i , p }$ is the momentum coefficient, and $v _ { m a x }$ prevents sudden rating jumps.

The expected probability of success is defined as

$$
E _ { i , p } = \frac { 1 } { 1 + 1 0 ^ { ( R _ { q _ { i , p } } ^ { e f f } - R _ { i , p } ) / 4 0 0 } } ,\tag{16}
$$

where $R _ { q _ { i , p } } ^ { e f f }$ is the effective item rating. The effective rating may include graph centrality adjustment:

$$
R _ { q } ^ { e f f } = R _ { q } + \xi c ( q ) ,\tag{17}
$$

where $c ( q ) \in [ 0 , 1 ]$ is the normalized centrality score of item $q$ and $\xi$ controls the maximum centrality contribution.

## IV. PROPOSED METHODOLOGY

This section presents the proposed MARS framework. MARS extends rating-based adaptive assessment by integrating dynamic learner uncertainty, multi-signal response quality, graph-aware item calibration, momentum-inspired learner progression, rapid-response anomaly control, and contextual remediation.

## A. System Overview

At each learner interaction, MARS receives the current learner state $s _ { i , p } ,$ the attempted question $q _ { i , p } .$ , and the observed response ${ r _ { i , p } } .$ . The system first computes the expected correctness probability using the learner rating and the effective item rating. It then computes a response quality score using correctness, response time, hint usage, and streak quality. This response quality is compared with the expected probability to produce a raw rating update. The raw update is passed through the momentum velocity layer, which smooths noisy individual responses and updates the learner rating. Finally, the next question is selected from either a standard graph neighbourhood or a remediation neighbourhood.

Figure 1 illustrates the MARS adaptive assessment pipeline. Learner responses, question-bank metadata, and knowledgegraph relations are used to update learner state through expected success estimation, response-quality scoring, dynamic rating updates, and momentum-aware smoothing. The updated state then guides candidate generation and scoring using graph proximity, difficulty alignment, weak-topic recovery, diversity, and risk-aware utility. The selected question is delivered with an interpretable explanation, and the resulting response closes the feedback loop for subsequent routing and remediation.

## B. Behavioral Momentum-Aware Adaptive Routing

MARS maintains a behavioural momentum state to characterize the recent learning dynamics of a learner. The behavioural state is defined as:

$$
\begin{array} { r } { m _ { i , p } = \left\{ \begin{array} { l l } { \mathrm { h o t } , } & { k _ { i , p } \geq 5 \land \hat { a } _ { i , p } \geq 0 . 8 0 , } \\ { \mathrm { c o l d } , } & { k _ { i , p } \leq - 3 \land \hat { a } _ { i , p } < 0 . 4 0 , } \\ { \mathrm { s t e a d y } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{18}
$$

where $k _ { i , p }$ denotes the recent correctness streak and $\hat { a } _ { i , p }$ represents the recent session accuracy of learner i at interaction step $p .$

Unlike static routing strategies that apply fixed difficulty shifts, MARS dynamically estimates the target difficulty transition based on learner velocity, recent performance, and rating volatility. The target rating for the next recommended item is computed as:

$$
R _ { i , p } ^ { t g t } = R _ { i , p } + \Delta R _ { i , p } ^ { t g t } ,\tag{19}
$$

where the adaptive target offset is defined as:

$$
\begin{array} { r } { \Delta R _ { i , p } ^ { t g t } = \mathrm { c l i p } \left( \alpha _ { v } v _ { i , p } + \alpha _ { a } ( \hat { a } _ { i , p } - a _ { 0 } ) - \alpha _ { \sigma } \frac { \sigma _ { i , p } ^ { 2 } } { \sigma _ { i , p } ^ { 2 } + \sigma _ { 0 } ^ { 2 } } , \right. } \\ { \left. - \Delta _ { m a x } , \Delta _ { m a x } \right) . } \end{array}\tag{20}
$$

Here, $v _ { i , t }$ denotes the momentum-aware learner velocity, $\hat { a } _ { i , t }$ is the recent rolling accuracy, $a _ { 0 }$ is the desired target accuracy level, and $\sigma _ { i , t } ^ { 2 }$ represents the recent rating volatility of the learner. The clipping operator constrains the target shift within the interval $[ - \Delta _ { m a x } , \Delta _ { m a x } ]$ to avoid abrupt difficulty transitions.

![](images/43708295eca56db3301b8c4ca25660fb8b11708822bc486401787a91dbc956ed.jpg)  
Fig. 1. Overview of the MARS online rating and routing pipeline.

The behavioural momentum label is retained as an interpretable learner-state descriptor, while the actual routing transition is determined dynamically through the adaptive offset formulation. This enables MARS to provide smoother and context-aware progression across the question graph compared with static rule-based routing policies.

## C. Response Quality Score

Instead of using only binary correctness, MARS computes a response quality score $P _ { i , p } \in [ 0 , 1 ] \colon$

$$
\begin{array} { r } { P _ { i , p } = S _ { i , p } \left( w _ { \tau } \tau _ { i , p } + w _ { H } ( 1 - H _ { i , p } ) + w _ { s } \psi _ { i , p } \right) + ( 1 - S _ { i , p } ) \epsilon _ { f a i l } , } \\ { ( 2 1 ) } \end{array}
$$

where $S _ { i , p } ~ \in ~ \{ 0 , 1 \}$ is correctness, $\tau _ { i , p }$ is response-time quality, $H _ { i , p }$ is normalized hint usage, $\psi _ { i , p }$ is streak quality, and $\epsilon _ { f a i l }$ is a small failure floor. The weights satisfy

$$
w _ { \tau } + w _ { H } + w _ { s } = 1 .\tag{22}
$$

The response-time quality is computed with respect to reference solve time $t _ { r e f } ( q )$

$$
\tau _ { i , p } = \mathrm { c l i p } \left( \frac { t _ { r e f } ( q ) - t _ { s p e n t } ( q ) + t _ { s l a c k } ( q ) } { t _ { r e f } ( q ) } , 0 , 1 \right) .\tag{23}
$$

Hint usage is normalized as

$$
H _ { i , p } = \operatorname* { m i n } \left( 1 , \frac { n _ { i , p } ^ { h i n t } } { n _ { m a x } ^ { h i n t } } \right) .\tag{24}
$$

The streak quality is given by

$$
\psi _ { i , p } = \frac { 1 } { 2 } \left( 1 + \frac { k _ { i , p } } { \operatorname* { m a x } ( | k _ { i , p } | , k _ { m a x } ) } \right) .\tag{25}
$$

## D. Dynamic Uncertainty-Aware Update Factor

The dynamic update factor $K _ { i , p }$ controls rating sensitivity. A high $K _ { i , p }$ enables rapid calibration for new learners, while a low $K _ { i , p }$ stabilizes ratings for well-calibrated learners. MARS uses a piecewise factor:

$$
K _ { i , p } = \left\{ \begin{array} { l l } { K _ { m a x } , \quad n _ { i , p } < n _ { p r o v } , } \\ { K _ { m i n } + ( K _ { m a x } - K _ { m i n } ) e ^ { - \lambda ( n _ { i , p } - n _ { p r o v } ) } } \\ { \quad \times \left( \displaystyle \frac { \sigma _ { i , p } ^ { 2 } } { \sigma _ { i , p } ^ { 2 } + \sigma _ { 0 } ^ { 2 } } \right) , \quad n _ { i , p } \geq n _ { p r o v } . } \end{array} \right.\tag{26}
$$

Here, $K _ { m a x }$ is the maximum update factor, $K _ { m i n }$ is the stable floor, $n _ { p r o v }$ is the provisional phase length, λ is the decay rate, and $\sigma _ { 0 } ^ { 2 }$ is a reference volatility constant.

## E. Momentum-Aware Learner Progression

The central extension of MARS is the use of a learner progression velocity. The raw update is first computed as

$$
\Delta R _ { i , p } ^ { r a w } = K _ { i , p } Q _ { i , p } ( S _ { i , p } - E _ { i , p } )\tag{27}
$$

Then the velocity is updated as

$$
v _ { i , p } = \beta _ { i , p } v _ { i , p - 1 } + ( 1 - \beta _ { i , p } ) \Delta R _ { i , p } ^ { r a w } .\tag{28}
$$

Finally, the learner rating is updated as

$$
R _ { i , p + 1 } = R _ { i , p } + \mathrm { c l i p } ( v _ { i , p } , - v _ { m a x } , v _ { m a x } ) .\tag{29}
$$

The momentum coefficient is adapted using rating volatility. First, the volatility ratio is computed as

$$
\nu _ { i , p } = \frac { \sigma _ { i , p } ^ { 2 } } { \sigma _ { i , p } ^ { 2 } + \sigma _ { 0 } ^ { 2 } } .\tag{30}
$$

The coefficient is then defined as

$$
\beta _ { i , p } = \beta _ { m i n } + ( \beta _ { m a x } - \beta _ { m i n } ) ( 1 - \nu _ { i , p } ) .\tag{31}
$$

Stable learners receive a higher momentum coefficient, producing smoother progression. Volatile learners receive a lower momentum coefficient, reducing the risk of carrying noisy trends too far.

## F. Rapid-Response Anomaly Control

The reference solve time is estimated as

$$
t _ { r e f } ( q ) = \alpha _ { 0 } + \alpha _ { 1 } \ell _ { s t e m } ( q ) + \alpha _ { 2 } \ell _ { o p t } ( q ) + \alpha _ { 3 } c _ { i d x } ( q ) ,\tag{32}
$$

where $\ell _ { s t e m } ( q )$ is stem length, $\ell _ { o p t } ( q )$ is option length, and $c _ { i d x } ( q )$ is a binary complexity indicator. A response is considered rapid when

$$
t _ { s p e n t } ( q ) < \eta _ { t } t _ { r e f } ( q ) .\tag{33}
$$

If the response is correct and rapid, the positive update is capped:

$$
\begin{array} { r } { \Delta R _ { i , p } ^ { r a w } = \operatorname* { m i n } ( \Delta R _ { i , p } ^ { r a w } , \kappa K _ { m i n } ) . } \end{array}\tag{34}
$$

This preserves correctness credit while preventing artificial rating inflation.

## G. Candidate Scoring Function

For each candidate $q ^ { \prime } \in \mathcal { C } _ { i , p } ,$ MARS computes

$$
\operatorname { s c o r e } ( q ^ { \prime } ) = w _ { h } h ( q ^ { \prime } ) + w _ { g } g ( q ^ { \prime } ) + w _ { v } M ( q ^ { \prime } ) ,\tag{35}
$$

where $h ( q ^ { \prime } )$ is the heuristic fit, $g ( q ^ { \prime } )$ is the graph-structural score, and $M ( q ^ { \prime } )$ is the momentum-alignment score.

The heuristic component is

$$
\begin{array} { r l r } & { } & { h ( \boldsymbol { q } ^ { \prime } ) = 1 - \frac { a _ { 1 } d _ { R } ( \boldsymbol { q } ^ { \prime } ) + a _ { 2 } d _ { d i f f } ( \boldsymbol { q } ^ { \prime } ) } { C _ { n o r m } } + \beta _ { w } \mathbb { I } [ \boldsymbol { q } ^ { \prime } \in W _ { i , p } ] } \\ & { } & { + \beta _ { t y p e } \mathbb { I } [ t y p e ( \boldsymbol { q } ^ { \prime } ) \neq t y p e ( q _ { i , p } ) ] , } \end{array}\tag{36}
$$

where

$$
d _ { R } ( q ^ { \prime } ) = | R _ { q ^ { \prime } } ^ { e f f } - R _ { i , p } ^ { t g t } | .\tag{37}
$$

The graph-structural score is

$$
g ( q ^ { \prime } ) = \gamma _ { 1 } \rho _ { i n } ( q ^ { \prime } ) + \gamma _ { 2 } \rho _ { o u t } ( q ^ { \prime } ) + \gamma _ { 3 } c ( q ^ { \prime } ) ,\tag{38}
$$

where $\rho _ { i n }$ and $\rho _ { o u t }$ are normalized in-degree and out-degree, and $c ( q ^ { \prime } )$ is normalized centrality.

The momentum alignment term is

$$
M ( q ^ { \prime } ) = \left\{ { 1 , \quad \quad v _ { i , p } > 0 \land R _ { q ^ { \prime } } ^ { e f f } \geq R _ { i , p } } , \right. \quad\tag{39}
$$

This term aligns routing with learner trajectory. Positive velocity supports controlled advancement, while negative velocity encourages consolidation or remediation.

## H. Contextual Remediation and Retry

When learner i answers item $q _ { m }$ incorrectly, the system marks it as an unresolved miss:

$$
\pi _ { i , p } ^ { r e m }  \mathrm { a c t i v e } ( q _ { m } ) .\tag{40}
$$

During remediation, candidates are restricted to graph-related easier items. The raw update is attenuated to avoid overpenalizing the learner during recovery:

$$
\Delta R _ { i , p } ^ { r e m } = \lambda _ { r e m } \Delta R _ { i , p } ^ { r a w } , \qquad 0 < \lambda _ { r e m } < 1 .\tag{41}
$$

Algorithm 1 MARS Adaptive Recommendation   
Require: $s _ { i , p } ,$ graph G, last item $q _ { i , p }$   
Ensure: Next question $q ^ { * }$   
1: Compute $m _ { i , p }$ and $\bar { R } _ { i , p } ^ { t g t }$   
2: if remediation active then   
3: Build remediation candidate set $\mathcal { C } _ { i , p } ^ { r e m }$   
4: else   
5: Build graph-neighbourhood set $\mathcal { C } _ { i , p } ^ { s t d }$   
6: end if   
7: if $\mathcal { C } _ { i , p } = \emptyset$ then   
8: Use fallback candidates   
9: end if   
10: for $q ^ { \prime } \in \mathcal { C } _ { i , p }$ do   
11: Compute $R _ { q ^ { \prime } } ^ { e f f } , h ( q ^ { \prime } ) , g ( q ^ { \prime } ) , M ( q ^ { \prime } )$   
12: Compute $\operatorname { s c o r e } ( q ^ { \prime } ) = w _ { h } h ( q ^ { \prime } ) + w _ { g } g ( q ^ { \prime } ) + w _ { v } M ( q ^ { \prime } )$   
13: end for   
14: $q ^ { * }  \mathrm { a r g }$ max $\cdot _ { q ^ { \prime } \in \mathcal { C } _ { i , p } }$ score(q<sup>′</sup>)   
15: return $q ^ { * } , \mathcal { E } ( \stackrel { * } { q } ^ { * } )$

Remediation terminates when

$$
n _ { i , p } ^ { r e m } \geq N _ { r } \quad \wedge \quad \operatorname { A c c } _ { i , p } ^ { r e m } \geq A _ { r } .\tag{42}
$$

After this condition is satisfied, the original missed item is queued for retry. If the retry is correct, the unresolved miss is cleared. Otherwise, remediation remains active for a later session.

## I. Interpretability of Recommendation

For every recommended question, MARS produces an explanation tuple:

$$
\mathcal { E } ( q ^ { * } ) = ( d _ { G } , R ^ { t g t } , R _ { q ^ { * } } ^ { e f f } , W _ { b o n u s } , c ( q ^ { * } ) , v _ { i , p } , \pi _ { i , p } ^ { r e m } ) .\tag{43}
$$

Here, $d _ { G }$ is graph distance, $R ^ { t g t }$ is the target rating, $R _ { q ^ { * } } ^ { e f f }$ is effective item rating, $W _ { b o n u s }$ indicates weak-topic prioritization, $c ( q ^ { * } )$ is graph centrality, $v _ { i , p }$ is progression velocity, and $\pi _ { i , p } ^ { r e m }$ indicates whether the recommendation was made during remediation.

## V. EXPERIMENTAL DESIGN

This section describes the evaluation protocol used to assess MARS.The experiments are conducted on two datasets: AS-SISTments 2009 for online correctness prediction and rating stability, and MARS-SIMEXAM, a persona-based synthetic assessment dataset for time-budgeted routing.

## A. Datasets

ASSISTments 2009 is used as the public benchmark for online correctness prediction and rating stability [39], [40], [44]. It contains learner–question interactions with correctness labels, skill tags, timestamps, hint information, and responsetime information. Since public educational logs are observational, they are suitable for prediction benchmarking but limited for evaluating counterfactual adaptive routing [41].

To evaluate adaptive question selection under realistic examination constraints, we additionally use a synthetic assessment dataset generated using a Fixed-Elo-based persona simulator. The dataset contains 277,500 learner–question interactions from 300 learners, 6,000 test sessions, 1,091 questions, 8 subjects, 54 topics, and 10 learner personas. Each interaction includes correctness, response time, question marks, learner ratings, and topic information, enabling evaluation of adaptive routing under fixed test-duration and negative-marking constraints.

TABLE I DATASET SUMMARY
<table><tr><td>Dataset</td><td>Learners</td><td>Items</td><td>Interactions</td><td>Topics</td></tr><tr><td>ASSISTments 2009</td><td>4,217</td><td>26,688</td><td>346,860</td><td>123</td></tr><tr><td>MARS-SimExam</td><td>300</td><td>1,091</td><td>277,500</td><td>54</td></tr></table>

## B. Question Graph Construction

For each dataset, a directed question graph is constructed using available skill and interaction information. Same-skill edges connect questions from the same knowledge component, while transition edges are created from consecutive learner attempts. Bridge edges connect related skill regions based on cross-topic transitions. Item difficulty is estimated from training interactions using smoothed correctness statistics or provided question Elo values where available. The graph supports centrality estimation and adaptive neighbourhood selection during routing.

## C. Baselines and Metrics

MARS is compared with Majority, Fixed Elo [4], Bayesian Knowledge Tracing (BKT) [7], DKT-LSTM [13], and Graphbased Knowledge Tracing (GKT) [22]. These baselines cover majority prediction, rating-based modeling, probabilistic knowledge tracing, neural sequence modeling, and graphbased knowledge tracing.

For correctness prediction, we report AUC, accuracy, log loss, and Brier score. For rating stability, we report mean absolute update (MAU) and update standard deviation (SDU), where lower values indicate smoother learner-rating trajectories.

For time-budgeted routing, each policy selects at most $N = 5 0$ questions within $T = 3 6 0 0$ seconds. We report Score, Score/min, raw Marks, total Time, and Overtime, defined as the percentage of sessions exceeding T based on the time spent on individual questions from the database. Score is computed as

$$
\mathrm { S c o r e } = \sum _ { q \in \cal { S } } \left[ y _ { q } m ( q ) - ( 1 - y _ { q } ) \rho m ( q ) \right] ,\tag{44}
$$

where $y _ { q }$ is correctness, $m ( q )$ is item marks, and $\rho = 0 . 2 5 ;$ non-negative-marking practice tests are obtained by setting $\rho = 0$

## D. Implementation Details

All models use a chronological learner-level train-test split. Learner ratings are initialized to 1500, and item ratings are estimated only from training data to avoid leakage. MARS uses dynamic K, momentum-aware updates, response-quality scaling, and rapid-response capping; detailed hyperparameters and ablations are provided in the supplement.

TABLE II  
ONLINE CORRECTNESS PREDICTION RESULTS ON ASSISTMENTS.
<table><tr><td>Model</td><td>AUC</td><td>Accuracy</td><td>LogLoss</td><td>Brier</td></tr><tr><td>Majority</td><td>0.5000</td><td>0.6883</td><td>0.6207</td><td>0.2147</td></tr><tr><td>FixedElo</td><td>0.7737</td><td>0.7430</td><td>0.6425</td><td>0.1873</td></tr><tr><td>BKT</td><td>0.7529</td><td>0.7390</td><td>0.5215</td><td>0.1758</td></tr><tr><td>DKT</td><td>0.8648</td><td>0.8027</td><td>0.4017</td><td>0.1332</td></tr><tr><td>GKT</td><td>0.8658</td><td>0.8124</td><td>0.3967</td><td>0.1298</td></tr><tr><td>MARS</td><td>0.7532</td><td>0.7260</td><td>0.6858</td><td>0.1987</td></tr></table>

TABLE III

ONLINE CORRECTNESS PREDICTION RESULTS ON THE SYNTHETIC DATASET.
<table><tr><td>Model</td><td>AUC</td><td>Accuracy</td><td>LogLoss</td><td>Brier</td></tr><tr><td>Majority</td><td>0.5000</td><td>0.3130</td><td>0.8529</td><td>0.3272</td></tr><tr><td>FixedElo</td><td>0.8244</td><td>0.8413</td><td>0.4665</td><td>0.1301</td></tr><tr><td>BKT</td><td>0.7043</td><td>0.6024</td><td>0.8637</td><td>0.2941</td></tr><tr><td>DKT</td><td>0.7398</td><td>0.6410</td><td>0.6129</td><td>0.2132</td></tr><tr><td>GKT</td><td>0.6909</td><td>0.4475</td><td>0.7436</td><td>0.2739</td></tr><tr><td>MARS</td><td>0.8265</td><td>0.8407</td><td>0.4906</td><td>0.1329</td></tr></table>

Code is available at: https://github.com/zvikrnt/MARS.

## VI. RESULTS AND DISCUSSION

## A. Correctness Prediction Results

Table II reports results on ASSISTments 2009. GKT achieves the strongest prediction performance, followed by DKT. This is expected because neural knowledge tracing models are optimized for sequential correctness prediction. MARS does not outperform neural models on this public observational benchmark, but it remains an interpretable online rating model with stable learner-state updates.

Table III reports results on the synthetic dataset. In this setting, MARS achieves the highest AUC, while Fixed Elo gives slightly better accuracy, log loss, and Brier score. This shows that MARS is more competitive when the environment contains adaptive routing, learner momentum, and difficultyaware interaction dynamics.

These two results indicate that MARS should not be interpreted as a replacement for neural knowledge tracing models in passive prediction benchmarks. Instead, MARS is better positioned as an online adaptive rating and routing framework for environments where interpretable learner-state updates, rating stability, and adaptive intervention are required.

For rating stability, MARS reduced the mean absolute rating update from 9.47 to 4.95 and the update standard deviation from 13.84 to 7.68 compared with Fixed Elo, corresponding to a 44.5% reduction in update volatility. Fixed Elo is reported here as a reference rather than a competing baseline.

## B. Time-Budgeted Adaptive Routing under Negative Marking

Table IV reports the time-budgeted routing results under negative marking. In this experiment, each policy selects up to 50 questions within a 3600 second test-duration budget. The reward is computed using positive marks for correct answers and a 25% penalty for incorrect answers. Therefore, total score is lower than total raw marks when a policy selects high-risk questions that are answered incorrectly. Results are averaged over 90 evaluated session-policy pairs.

TABLE IV  
TIME-BUDGETED ADAPTIVE ROUTING.
<table><tr><td>Policy</td><td>Score</td><td>Score/min</td><td>Marks</td><td>Overtime</td><td>Time (s)</td></tr><tr><td>BKT</td><td>75.68</td><td>1.36</td><td>93.53</td><td>48.9%</td><td>3282.91</td></tr><tr><td>DKT</td><td>81.19</td><td>1.48</td><td>98.69</td><td>50.0%</td><td>3282.84</td></tr><tr><td>GKT</td><td>66.01</td><td>1.19</td><td>86.56</td><td>46.7%</td><td>3280.63</td></tr><tr><td>MARS</td><td>89.28</td><td>1.70</td><td>98.27</td><td>5.6%</td><td>3132.54</td></tr></table>

TABLE V

ADAPTABILITY COMPARISON
<table><tr><td>Model</td><td>Complexity Memory</td><td></td><td>Online</td><td>Suitable Use</td></tr><tr><td>Fixed Elo</td><td>O(1)</td><td>Low</td><td>O(1)</td><td>Web rating engine</td></tr><tr><td>BKT</td><td>O(s)</td><td>Low</td><td>0(1)</td><td>Skill mastery tracking</td></tr><tr><td>DKT</td><td> $O ( T h ^ { 2 } )$ </td><td>High</td><td> $O ( h ^ { 2 } )$ </td><td>GPU/server prediction</td></tr><tr><td>GKT</td><td> $O ( | E | h )$ </td><td>High</td><td> $O ( | E | \dot { h } )$ </td><td>Graph KT server</td></tr><tr><td>MARS</td><td> $O ( d ^ { k _ { h } } )$ </td><td>Low</td><td>Ö(1)</td><td>Adaptive learning platform</td></tr></table>

MARS achieves the highest total score and score efficiency among the compared BKT, DKT, and GKT routing variants. Compared with DKT, MARS improves total score by 8.09 points and reduces overtime violations by 44.4 percentage points. Compared with BKT and GKT, MARS improves total score by 13.60 and 23.27 points, respectively. These gains show that MARS is useful not only for stable online rating, but also for practical adaptive assessment settings where the system must balance expected marks, risk, and test-duration constraints.

When negative marking is in play, the gap between total marks and total score is what actually decides performance. Although DKT obtains slightly higher raw marks, its advantage does not translate into higher final score because it incurs substantially more overtime violations. Raw correctness, on its own, tells an incomplete story; any adaptive policy worth deploying in a competitive exam setting has to weigh risk and time discipline just as heavily.

## C. Adaptability and Deployment Suitability

For application-oriented deployment, MARS is designed to support low-latency online updates, small per-learner memory, and interpretable routing decisions. Table V summarizes the deployment characteristics of the compared models.

MARS updates learner state using only the current rating, velocity, recent accuracy, update variance, and item metadata. Therefore, it avoids full-sequence recomputation during inference and is suitable for web-based practice platforms, LMSintegrated quizzes, and mobile adaptive assessment systems.

## D. Overall Discussion

The results show that MARS is not intended to replace high-capacity knowledge tracing models for passive nextresponse prediction: on ASSISTments 2009, DKT and GKT achieve stronger predictive performance. Instead, MARS is better positioned as an online adaptive rating and routing framework. In the time-budgeted negative-marking setting, MARS achieves the highest total score and score efficiency, while DKT obtains slightly higher raw marks but much higher overtime violations. This indicates that fixed-duration adaptive assessment requires more than maximizing correctness or raw marks; the policy must jointly consider expected correctness, mark value, penalty risk, and available time. Overall, MARS is best suited for real-time, interpretable adaptive assessment where learner states must be updated online and routing decisions must remain auditable.

## VII. LIMITATIONS AND RESPONSIBLE USE

MARS depends on the quality of the question graph, skill labels, and item-difficulty estimates; weak graph edges may degrade routing quality. ASSISTments 2009 is observational and cannot fully evaluate counterfactual routing, while the synthetic persona-based dataset cannot fully represent real learner behaviour. Since the synthetic dataset is generated using a Fixed-Elo-based simulator, the score-efficiency gains should be validated on real adaptive deployment logs. In preliminary routing experiments, a reward-only policy without negative marking tended to over-select high-mark, high-difficulty items, revealing an objective-misalignment risk; therefore, the final routing objective includes negative marking and time penalties to discourage risk-seeking item selection.

The present time-budgeted comparison focuses on BKT, DKT, GKT, and MARS under a common routing objective. Future work will compare MARS with uncertaintyaware Glicko-style knowledge tracing [5], Transformer-based knowledge tracing [21], and recent graph-aware knowledgetracing models [30], [32], [33]. Response-time features may also be affected by device, network, reading-speed, or accessibility differences. MARS should therefore be used as a decision-support tool with transparent recommendations, privacy-preserving data handling, bias auditing, and human oversight before high-stakes deployment [37], [38].

## VIII. CONCLUSION

This paper introduces MARS (Momentum-Aware Adaptive Rating System), a personalized adaptive assessment framework that integrates learner-rating updates, responseaware proficiency estimation, graph-based question selection, momentum-driven learner-state stabilization, and interpretable recommendations. MARS is designed to provide transparent, adaptive, and real-time question selection for personalized learning environments.

In a time-budgeted negative-marking setting, MARS achieves higher total score and score efficiency than the routing baselines while substantially reducing overtime violations. These results position MARS as an interpretable, deploymentfriendly adaptive routing framework rather than a replacement for neural knowledge tracing models in passive prediction benchmarks. Future work will validate MARS on real adaptive deployment logs and incorporate explicit learner uncertainty.

## REFERENCES

[1] F. M. Lord, Applications of Item Response Theory to Practical Testing Problems. Hillsdale, NJ, USA: Lawrence Erlbaum Associates, 1980.

[2] W. J. van der Linden and C. A. W. Glas, Eds., Computerized Adaptive Testing: Theory and Practice. Dordrecht, The Netherlands: Kluwer Academic Publishers, 2000.

[3] H. Wainer, N. J. Dorans, R. Flaugher, B. F. Green, R. J. Mislevy, L. Steinberg, and D. Thissen, Computerized Adaptive Testing: A Primer, 2nd ed. Mahwah, NJ, USA: Lawrence Erlbaum Associates, 2000.

[4] M. E. Glickman, “Parameter estimation in large dynamic paired comparison experiments,” Journal of the Royal Statistical Society: Series C, vol. 48, no. 3, pp. 377–394, 1999.

[5] M. E. Glickman, “The Glicko-2 rating system,” Boston University, 2008. [Online]. Available: http://www.glicko.net/glicko/glicko2.pdf

[6] R. Herbrich, T. Minka, and T. Graepel, “TrueSkill: A Bayesian skill rating system,” in Advances in Neural Information Processing Systems, vol. 19, 2007.

[7] A. T. Corbett and J. R. Anderson, “Knowledge tracing: Modeling the acquisition of procedural knowledge,” User Modeling and User-Adapted Interaction, vol. 4, no. 4, pp. 253–278, 1994.

[8] M. V. Yudelson, K. R. Koedinger, and G. J. Gordon, “Individualized Bayesian knowledge tracing models,” in Proc. 16th Int. Conf. Artificial Intelligence in Education, 2013, pp. 171–180.

[9] H. Cen, K. R. Koedinger, and B. Junker, “Learning factors analysis: A general method for cognitive model evaluation and improvement,” in Proc. 8th Int. Conf. Intelligent Tutoring Systems, 2006, pp. 164–175.

[10] P. I. Pavlik, H. Cen, and K. R. Koedinger, “Performance factors analysis: A new alternative to knowledge tracing,” in Proc. 14th Int. Conf. Artificial Intelligence in Education, 2009, pp. 531–538.

[11] R. S. J. d. Baker, A. T. Corbett, and V. Aleven, “More accurate student modeling through contextual estimation of slip and guess probabilities in Bayesian knowledge tracing,” in Proc. 9th Int. Conf. Intelligent Tutoring Systems, 2008, pp. 406–415.

[12] N. Thai-Nghe, T. Horvath, and L. Schmidt-Thieme, “Factorization models for forecasting student performance,” in Proc. 4th Int. Conf. Educational Data Mining, 2010, pp. 11–20.

[13] C. Piech, J. Bassen, J. Huang, S. Ganguli, M. Sahami, L. J. Guibas, and J. Sohl-Dickstein, “Deep knowledge tracing,” in Advances in Neural Information Processing Systems, vol. 28, 2015, pp. 505–513.

[14] M. Khajah, R. V. Lindsey, and M. C. Mozer, “How deep is knowledge tracing?” in Proc. 9th Int. Conf. Educational Data Mining, 2016, pp. 94–101.

[15] C. K. Yeung and D.-Y. Yeung, “Addressing two problems in deep knowledge tracing via prediction-consistent regularization,” in Proc. 5th ACM Conf. Learning at Scale, 2018.

[16] J. Zhang, X. Shi, I. King, and D.-Y. Yeung, “Dynamic key-value memory networks for knowledge tracing,” in Proc. 26th Int. Conf. World Wide Web, 2017, pp. 765–774, doi: 10.1145/3038912.3052580.

[17] G. Abdelrahman and Q. Wang, “Knowledge tracing with sequential keyvalue memory networks,” in Proc. 42nd Int. ACM SIGIR Conf. Research and Development in Information Retrieval, 2019, pp. 175–184, doi: 10.1145/3331184.3331195.

[18] S. Pandey and G. Karypis, “A self-attentive model for knowledge tracing,” in Proc. 12th Int. Conf. Educational Data Mining, 2019, pp. 384–389.

[19] S. Pandey and J. Srivastava, “RKT: Relation-aware self-attention for knowledge tracing,” in Proc. 29th ACM Int. Conf. Information and Knowledge Management, 2020, pp. 1205–1214.

[20] A. Ghosh, N. Heffernan, and A. S. Lan, “Context-aware attentive knowledge tracing,” in Proc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2020, pp. 2330–2339.

[21] Y. Choi, Y. Lee, J. Cho, J. Baek, B. Kim, Y. Cha, D. Shin, C. Bae, and J. Heo, “Towards an appropriate query, key, and value computation for knowledge tracing,” in Proc. 7th ACM Conf. Learning at Scale, 2020, pp. 341–344.

[22] H. Nakagawa, Y. Iwasawa, and Y. Matsuo, “Graph-based knowledge tracing: Modeling student proficiency using graph neural network,” in Proceedings of the IEEE/WIC/ACM International Conference on Web Intelligence, 2019.

[23] Q. Liu, S. Shen, Z. Huang, E. Chen, and Y. Zheng, “GIKT: A graphbased interaction model for knowledge tracing,” in Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), Springer, 2020, pp. 299–315.

[24] L. Lin and F. Wang, “Adaptive learning system based on knowledge graph,” in Proc. 2023 9th Int. Conf. Education and Training Technologies, Macau, China, 2023, pp. 112–117, doi: 10.1145/3599640.3599647.

[25] S. Tong, Q. Liu, W. Huang, Z. Huang, E. Chen, C. Liu, H. Ma, and S. Wang, “Structure-based knowledge tracing: An influence propagation view,” in Proc. IEEE Int. Conf. Data Mining, 2020, pp. 541–550.

[26] S. Minn, Y. Yu, M. C. Desmarais, F. Zhu, and J.-J. Vie, “Deep knowledge tracing and dynamic student classification for knowledge tracing,” in Proc. IEEE Int. Conf. Data Mining, 2018, pp. 1182–1187.

[27] G. Abdelrahman, Q. Wang, and B. P. Nunes, “Knowledge tracing: A survey,” ACM Computing Surveys, vol. 55, no. 11, pp. 1–37, 2023, doi: 10.1145/3569576.

[28] A. Namoun and A. Alshanqiti, “Predicting student performance using data mining and learning analytics techniques: A systematic literature review,” Applied Sciences, vol. 11, no. 1, Art. no. 237, 2021, doi: 10.3390/app11010237.

[29] F. Wang, Q. Liu, E. Chen, Z. Huang, Y. Chen, Y. Yin, Z. Huang, and S. Wang, “SEEP: Semantic-enhanced question embeddings pre-training for improving knowledge tracing,” Information Sciences, vol. 614, pp. 153–169, 2022, doi: 10.1016/j.ins.2022.10.015.

[30] Z. Liang, R. Wu, Z. Liang, J. Yang, L. Wang, and J. Su, “GELT: A graph embeddings based lite-transformer for knowledge tracing,” PLOS ONE, vol. 19, no. 5, Art. no. e0301714, 2024, doi: 10.1371/journal.pone.0301714.

[31] J. Xu and W. Hu, “An enhanced deep knowledge tracing model via multiband attention and quantized question embedding,” Applied Sciences, vol. 14, no. 8, Art. no. 3425, 2024, doi: 10.3390/app14083425.

[32] H. Yang, S. Hu, J. Geng, T. Huang, J. Hu, H. Zhang, and Q. Zhu, “Heterogeneous graph-based knowledge tracing with spatiotemporal evolution,” Expert Systems with Applications, vol. 241, Art. no. 122249, 2024, doi: 10.1016/j.eswa.2023.122249.

[33] Y. Han, H. Tang, W. Zhang, L. Du, J. Zhao, and M. Shao, “Dynamic heterogeneous graph contrastive networks for knowledge tracing,” Applied Soft Computing, vol. 166, Art. no. 112194, 2024, doi: 10.1016/j.asoc.2024.112194.

[34] Z. Duan, X. Dong, H. Gu, X. Wu, Z. Li, and D. Zhou, “Towards more accurate and interpretable model: Fusing multiple knowledge relations into deep knowledge tracing,” Expert Systems with Applications, vol. 243, Art. no. 122573, 2024, doi: 10.1016/j.eswa.2023.122573.

[35] Y. Lu, L. Tong, and Y. Cheng, “Advanced knowledge tracing: Incorporating process data and curricula information via an attention-based framework for accuracy and interpretability,” Journal of Educational Data Mining, vol. 16, no. 2, pp. 58–84, 2024.

[36] A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[37] S. Barocas, M. Hardt, and A. Narayanan, Fairness and Machine Learning. fairmlbook.org, 2019. [Online]. Available: https://fairmlbook.org

[38] R. S. Baker and A. Hawn, “Algorithmic bias in education,” International Journal of Artificial Intelligence in Education, vol. 32, pp. 1052–1092, 2022, doi: 10.1007/s40593-021-00285-9.

[39] M. Feng, N. T. Heffernan, and K. R. Koedinger, “Addressing the assessment challenge with an online system that tutors as it assesses,” User Modeling and User-Adapted Interaction, vol. 19, no. 3, pp. 243– 266, 2009.

[40] ASSISTments, “2009–2010 ASSISTment data: Skill builder data,” 2010. [Online]. Available: https://sites.google.com/site/assistmentsdata/home/ 2009-2010-assistment-data/skill-builder-data-2009-2010

[41] Y. Choi, Y. Lee, J. Shin, J. Cho, S. Park, S. Lee, J. Baek, C. Bae, B. Kim, and J. Heo, “EdNet: A large-scale hierarchical dataset in education,” in Proc. 21st Int. Conf. Artificial Intelligence in Education, 2020, pp. 69– 73.

[42] R. Pelanek, “Bayesian knowledge tracing, logistic models, and beyond: An overview of learner modeling techniques,” User Modeling and User-Adapted Interaction, vol. 27, no. 3–5, pp. 313–350, 2017.

[43] X. Xiong, S. Zhao, E. G. Van Inwegen, and J. E. Beck, “Going deeper with deep knowledge tracing,” in Proc. 9th Int. Conf. Educational Data Mining, 2016, pp. 545–550.

[44] J. Stamper, A. Niculescu-Mizil, S. Ritter, G. J. Gordon, and K. R. Koedinger, “Bridge to algebra 2006–2007: KDD Cup 2010 development data set,” DataShop, Carnegie Mellon University, 2010.

## APPENDIX

## SYNTHETIC DATASET GENERATION

The MARS-SimExam, a persona-based synthetic assessment dataset was generated from a master question bank using a Fixed-Elo interaction simulator. The simulator creates 300 learners across 10 personas, with 30 learners per persona. Each persona defines a different starting rating range, latent skill level, growth rate, response speed, carelessness level, challenge preference, volatility, and accuracy bias. For each learner, daily practice sessions are generated over multiple active days, and questions are selected near a target rating derived from the learner’s current rating, persona challenge preference, topic continuity, and occasional remediation or retry events. Correctness is sampled from an Elo-style probability based on latent learner skill and question Elo, with additional persona-specific bias, fatigue, carelessness, and random noise. Learner ratings are then updated using a fixed Elo update with $K = 3 2 \colon$

$$
\Delta R = \mathrm { r o u n d } \left( K \left( y - \frac { 1 } { 1 + 1 0 ^ { ( R _ { q } - R _ { i } ) / 4 0 0 } } \right) \right) .\tag{45}
$$

The simulator also records response time, graph mode, edge type, remediation status, response quality, streak state, weak topic, rating before and after the response, and predicted correctness probability. The final dataset contains 277,500 learner-question interactions, together with student-level and persona-level summary files.

## COMPUTATIONAL COMPLEXITY ANALYSIS

To demonstrate the suitability of the proposed MARS framework for real-time deployment in large-scale adaptive assessment environments, we analyze the online time complexity of a single learner-item interaction step. The overall time complexity is determined by three primary operations: rating state updates, candidate generation, and candidate scoring.

## A. Learner State and Rating Update

Upon receiving a learner’s response, MARS computes the effective item rating, expected correctness, response quality score, dynamic update factor, and momentum velocity. As defined in Equations (11) through (31), these operations consist strictly of scalar arithmetic, conditional checks, and basic mathematical functions (e.g., exponentials). Consequently, the time complexity for updating the learner state is constant:

$$
\mathcal { O } _ { u p d a t e } = \mathcal { O } ( 1 )\tag{46}
$$

## B. Candidate Generation

For standard progression, MARS generates a candidate set $\mathcal { C } _ { i , t } ^ { s t d }$ by performing a breadth-first search (BFS) on the directed knowledge graph $G \ : = \ : ( V , E , w )$ starting from the current item $q _ { i , t } ,$ bounded by a maximum hop limit $k _ { h }$ . Assuming an average node out-degree of $d ,$ the maximum number of items explored is bounded by $d ^ { k _ { h } }$ . The contextual remediation candidate generation follows a similar graph traversal logic. Thus, the time complexity for candidate generation is:

$$
\mathcal { O } _ { g e n } = \mathcal { O } ( d ^ { k _ { h } } )\tag{47}
$$

## C. Candidate Scoring and Selection

For each candidate $q ^ { \prime } \in \mathcal { C } _ { i , t }$ , MARS computes a composite utility score comprising heuristic fit, graph-structural importance, and momentum alignment (Equation 35). Because structural metrics such as in-degree, out-degree, and node centrality are precomputed offline, scoring each candidate requires $\mathcal { O } ( 1 )$ operations. Scoring the entire generated candidate set scales linearly with the size of the set:

$$
\mathcal { O } _ { s c o r e } = \mathcal { O } ( | \mathcal { C } _ { i , t } | ) \subseteq \mathcal { O } ( d ^ { k _ { h } } )\tag{48}
$$

## D. Overall Online Complexity

The total time complexity per interaction step is the sum of the individual operations:

$$
\mathcal { O } _ { t o t a l } = \mathcal { O } ( 1 ) + \mathcal { O } ( d ^ { k _ { h } } ) + \mathcal { O } ( d ^ { k _ { h } } ) = \mathcal { O } ( d ^ { k _ { h } } )\tag{49}
$$

Because the maximum hop limit $k _ { h }$ is a small constant (e.g., $k _ { h } \in \{ 1 , 2 , 3 \} )$ , the online inference time scales independently of the total item bank size $| V |$ . This sub-linear scaling confirms that MARS satisfies the stringent low-latency requirements of web-based and real-time adaptive learning platforms. In Table V, Online is the incremental cost of updating a cached learner state after one response. BKT is $O ( 1 )$ because a response updates only the mastery of the item’s own skill through a closed-form Bayes rule, even though its full state spans all S skills. DKT and GKT advance an h-dimensional hidden state; MARS updates a few scalars.

## EXTRA RESULTS

## E. Rating Stability

On the ASSISTments dataset, MARS yielded substantially more stable learner-rating trajectories than Fixed Elo. The mean absolute rating update decreased from 9.47 to 4.95, and the update standard deviation decreased from 13.84 to 7.68, representing a 44.5% reduction in rating volatility. These results suggest that MARS provides smoother learner-state estimation while preserving the interpretability and online update characteristics of rating-based adaptive assessment methods.

## F. Ablation Results

Table VII presents the component-level ablation results. MARS-Final achieves a balanced trade-off between predictive performance and rating stability. Removing momentum slightly reduces AUC and accuracy while increasing update variability. Replacing dynamic K with fixed K produces the strongest AUC among MARS variants, but it also substantially increases update magnitude and volatility, making the learner trajectory less stable.

The ablation results show three important findings. First, fixed K improves prediction but sacrifices stability, confirming that dynamic update sensitivity is useful when stable learner trajectories are desired. Second, the momentum term provides a modest but consistent improvement over the non-momentum variant. Third, the streak multiplier substantially reduces AUC and increases log loss, indicating that streak should not directly multiply the rating update. Instead, streak is better used as a behavioural routing signal for identifying hot, steady, or cold learner states.

TABLE VI  
MAIN MARS HYPERPARAMETERS USED IN THE TIME-BUDGETED ROUTING EXPERIMENT.
<table><tr><td>Symbol</td><td>Value</td><td>Meaning</td><td>Practical effect</td></tr><tr><td> $w _ { \mathrm { s c o r e } }$ </td><td>7.83</td><td>Expected score weight</td><td>Prioritizes questions with high expected score under negative marking.</td></tr><tr><td> $w _ { \mathrm { s c o r e / t i m e } }$ </td><td>0.55</td><td>Score-per-time weight</td><td>Rewards questions that provide more expected score per second.</td></tr><tr><td> $w _ { \mathrm { d i f f } }$ </td><td>1.98</td><td>Difficulty-fit weight</td><td>Encourages questions near the learner&#x27;s adaptive target difficulty.</td></tr><tr><td> $w _ { \mathrm { t o p i c } }$ </td><td>0.80</td><td>Weak-topic bonus</td><td>Increases priority for topics where the learner shows weakness.</td></tr><tr><td> $w _ { \mathrm { d i v } }$ </td><td>1.14</td><td>Diversity weight</td><td>Discourages repeatedly selecting questions from the same topic region.</td></tr><tr><td> $w _ { \mathrm { t i m e } }$ </td><td>0.31</td><td>Time penalty weight</td><td>Penalizes questions with high predicted response time.</td></tr><tr><td> $w _ { \mathrm { o v e r t i m e } }$ </td><td>20.89</td><td>Overtime penalty weight</td><td>Strongly discourages selections likely to exceed the test-duration budget.</td></tr><tr><td> $r _ { \mathrm { r i s k } }$ </td><td>2.78</td><td>Risk-aversion weight</td><td>Penalizes high-risk questions under negative marking.</td></tr><tr><td> $\alpha _ { v }$ </td><td>2.78</td><td>Velocity routing weight</td><td>Uses learner momentum to adjust target difficulty.</td></tr><tr><td> $\alpha _ { a }$ </td><td>54.32</td><td>Recent-accuracy routing weight</td><td>Shifts target difficulty according to recent learner accuracy.</td></tr><tr><td> $\alpha _ { \sigma }$ </td><td>26.31</td><td>Volatility routing weight</td><td>Reduces difficulty jumps for unstable learner trajectories.</td></tr><tr><td> $\Delta _ { \mathrm { m a x } }$ </td><td>195.62</td><td>Maximum target shift</td><td>Caps the adaptive difficulty shift.</td></tr><tr><td> $\beta _ { \mathrm { m i n } }$ </td><td>0.07</td><td>Minimum momentum coefficient</td><td>Allows faster response to volatile learner behaviour.</td></tr><tr><td> $\beta _ { \mathrm { m a x } }$ </td><td>0.12</td><td>Maximum momentum coefficient</td><td>Smooths learner-rating updates during stable behaviour.</td></tr><tr><td> $v _ { \mathrm { m a x } }$ </td><td>17.25</td><td>Maximum velocity step</td><td>Caps the magnitude of one momentum-based rating update.</td></tr><tr><td> $K _ { \mathrm { m a x } }$ </td><td>45.0</td><td>Maximum rating update factor</td><td>Enables faster calibration during early learner interactions.</td></tr><tr><td> $K _ { \mathrm { m i n } }$ </td><td>12.0</td><td>Minimum rating update factor</td><td>Provides a stable lower bound for rating updates.</td></tr><tr><td> $n _ { \mathrm { p r o v } }$ </td><td>30</td><td>Provisional phase length</td><td>Number of early attempts with stronger calibration.</td></tr><tr><td> $\sigma _ { 0 } ^ { \angle }$ </td><td>100.0</td><td>Reference volatility constant</td><td>Normalizes learner rating variance in update and routing rules.</td></tr><tr><td> $a _ { 0 }$ </td><td>0.70</td><td>Target accuracy</td><td>Desired challenge level for adaptive routing.</td></tr><tr><td> $\tau _ { w }$ </td><td>0.40</td><td>Weak-topic threshold</td><td>Topics below this accuracy are treated as weak topics.</td></tr><tr><td> $W _ { \mathrm { r e c e n t } }$ </td><td>5</td><td>Recent-topic window</td><td>Tracks recent topic usage for diversity and weak-topic control.</td></tr></table>

TABLE VII

ABLATION STUDY OF MARS COMPONENTS (N = 104, 858 FOR ALL VARIANTS). MAU AND SDU DENOTE MEAN ABSOLUTE UPDATE AND UPDATE STANDARD DEVIATION, RESPECTIVELY.
<table><tr><td>Model Variant</td><td>AUC</td><td> $\operatorname { A c c } .$ </td><td>LogLoss</td><td>Brier</td><td>MAU</td><td>SDU</td></tr><tr><td>MARS-Final</td><td>0.7532</td><td>0.7260</td><td>0.6858</td><td>0.1987</td><td>4.9461</td><td>7.6794</td></tr><tr><td>w/o Momentum</td><td>0.7504</td><td>0.7232</td><td>0.6892</td><td>0.1999</td><td>4.7384</td><td>7.7633</td></tr><tr><td>w/ Fixed K</td><td>0.7698</td><td>0.7377</td><td>0.6654</td><td>0.1916</td><td>9.3175</td><td>13.6527</td></tr><tr><td>w/o QualityFactor</td><td>0.7529</td><td>0.7247</td><td>0.6875</td><td>0.1993</td><td>4.9353</td><td>7.6716</td></tr><tr><td>w/o RapidCap</td><td>0.7566</td><td>0.7314</td><td>0.6728</td><td>0.1949</td><td>5.1348</td><td>7.8485</td></tr><tr><td>w/ StreakMultiplier</td><td>0.7244</td><td>0.7325</td><td>1.0238</td><td>0.2016</td><td>5.3362</td><td>8.8159</td></tr></table>

## METRIC INTERPRETATION

AUC measures how well a model ranks correct responses above incorrect responses and therefore reflects discrimination independent of a fixed threshold. Accuracy measures the proportion of correct binary predictions. Log loss penalizes confident but incorrect predictions more heavily and is useful for evaluating the quality of predicted probabilities. Brier score measures the average squared difference between predicted probabilities and observed outcomes, with lower values indicating better calibration. For rating stability, MAU represents the average size of rating updates after each interaction, whereas SDU measures the variability of those updates; lower values indicate smoother learner-state trajectories and lower sensitivity to noisy responses. In the time-budgeted routing setting, Score represents the final reward after penalties, Marks represents the raw marks obtained before penalties, Score/min measures time efficiency, and Overtime indicates how often a policy exceeds the allotted test duration from the dataset. It is accounted identically for every policy: item durations come from one median response-time model fitted on the training split, and an item is admitted only while elapsed time is below $T ,$ so a session can overshoot only on its final item. Overtime differences therefore reflect item selection, not accounting.

## HYPERPARAMETER SELECTION

Sessions were split by test into disjoint train, validation and final sets in a $7 0 / 1 5 / 1 5$ ratio. The routing weights in Table VI were tuned on the validation split only, using 50 trials of Tree-structured Parzen Estimator search that maximise Score + 0.5 Score/min − 5 Overtime $- \ 0 . 0 1 t _ { \mathrm { o v e r } }$ , where $t _ { \mathrm { o v e r } }$ is mean overtime in seconds. The selected weights were then frozen and evaluated once on the final split, and the same weights are used for every routing policy.

## CODE AVAILABILITY

The source code, experiment scripts, configuration files, and reproduction instructions are available at https://github. com/zvikrnt/MARS. The repository includes scripts for dataset preprocessing, online prediction, rating-stability evaluation, synthetic data generation, time-budgeted routing, and baseline comparison.