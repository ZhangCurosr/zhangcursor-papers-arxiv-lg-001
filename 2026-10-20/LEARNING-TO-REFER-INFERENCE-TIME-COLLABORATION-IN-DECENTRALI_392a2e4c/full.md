# LEARNING TO REFER: INFERENCE-TIME COLLABORATION IN DECENTRALIZED FEDERATED LEARNING

PREPRINT

Yuheng Ma<sup>1</sup>\* Chendi Wang<sup>2</sup> Qiong Zhang<sup>3†</sup> <sup>1</sup>KLATASDS-MOE, School of Statistics, East China Normal University <sup>2</sup>Department of Statistics and Data Science, Xiamen University <sup>3</sup>Division of Computing and Mathematical Sciences, MBZUAI

## ABSTRACT

In decentralized federated learning, heterogeneous clients may develop complementary expertise, yet each query is typically evaluated only by the client where it arrives. To exploit this distributed expertise without global coordination, we formulate inference-time collaboration as budgeted local expert referral, a graph-constrained routing process in which a query can move only between neighboring clients and only visited clients contribute to its final prediction. EdgeDRM implements this framework by learning a density-based routing potential for each client: differences between neighboring potentials provide the comparisons used to select successive referrals, while local task models generate predictions that are aggregated along the resulting route. Both training and inference are decentralized, using only neighbor-local information without a central router or graph-wide client scoring. We establish estimation guarantees for the learned comparisons and characterize how local improvement and graph structure govern prior-weighted density gains from referral. Experiments on benchmark and real-world medical data show that EdgeDRM outperforms graph-matched decentralized baselines in most settings, with the clearest benefits under sparse connectivity.

## 1 Introduction

Federated learning (FL) enables multiple clients to train a shared model without centralizing their raw data. Convention ally, a server distributes the model and aggregates locally computed updates (McMahan et al., 2017). In decentralized FL (Lalitha et al., 2019), clients form a communication graph whose nodes represent clients and whose edges specify which pairs can exchange model information directly, with no central server coordinating the network. The graph determines which clients can collaborate during training, but at inference a query is typically evaluated only by the client where it arrives. This inference limitation becomes consequential when client data distributions differ, a setting known as statistical heterogeneity. Consequently, a model may perform well on queries from its local distribution but poorly on those from other clients.

Such variation in local performance creates an opportunity for inference-time collaboration: a query that is poorly handled by its entry client may be better handled by another client. For instance, a patient may first visit a local hospital that lacks the expertise needed for the case, even though a better-suited specialist is available elsewhere in the hospital network. Reaching that specialist, however, may require a sequence of local referrals because each physician can refer the patient only to nearby or affiliated hospitals, which may in turn refer the patient onward. Each referral incurs delay and cost, while the referral path determines which specialists can ultimately be reached. We call the analogous inference process in decentralized FL local expert referral: a query enters at an arbitrary client, may traverse only a limited number of communication edges, and receives predictions only from the clients it visits. Here, a referral is one neighbor-to-neighbor transfer, routing is the process that selects successive referrals, and the route is the resulting sequence of visited clients. Unlike global expert selection, local expert referral is sequential and path-dependent: both the referral budget and the graph topology restrict which experts a query can reach. This raises our central question:

![](images/2e793f5f36e87bd8b1bf5b47d913c7a8ba98beaded979a31430ab8561186d663.jpg)  
(a) No router.

![](images/6df775e45e69f495a47d4eb76f9d40380cd3122de9581c0704b99987f50cf0aa.jpg)  
(b) Centralized router.

![](images/90523ddd2b27586b65cb7f0ef583b46d41529476a95c5525812d02de9f99e584.jpg)  
(c) Proposed local router.  
Figure 1: Hospital analogy for inference-time routing. Gray lines indicate hospital pairs that can exchange information and know one another’s capabilities, for example because they operate in the same region or health system. Without routing, a patient with a heart condition remains at an entry hospital whose expertise lies in gastroenterology (left). A centralized router sees all hospitals and dispatches the patient directly (center), but such global coordination is often infeasible because of administrative boundaries, privacy constraints, and geographic separation. In contrast, EdgeDRM (ours) makes stepwise local referrals (right): the gastroenterology hospital first routes the patient to a general chest center, whose thoracic expertise better matches the condition. The center then identifies the condition as cardiac and routes the patient to a dedicated cardiology hospital. Thus, each hop progressively improves the match between the patient and the hospital’s expertise.

how can client specialization be exploited through graph-constrained local referral to improve on the entry client’s prediction? Figure 1 compares no routing, centralized routing, and local referral.

Existing work addresses only individual parts of this problem. Personalized and decentralized FL methods determine how clients collaborate during training, but typically leave each test query at the client where it arrives (Sun et al., 2023; Zantedeschi et al., 2020; Dai et al., 2022; Lin et al., 2025; Li et al., 2025). Federated routing and mixture-ofexperts methods exploit specialization through global client scores, learned routing gates, or expert subsets, but do not model budgeted neighbor-to-neighbor query movement (Wang et al., 2026a; Reisser et al., 2021; Farhat et al., 2025). Graph-based referral and search methods support local query movement, but learn from observed referral outcomes or graph-wide search supervision rather than from the static datasets held by neighboring FL clients (KhudaBukhsh et al., 2016; Baranchuk et al., 2019). Together, these lines leave open how to learn query–client compatibility from decentralized client data and compose neighbor-local comparisons into an effective prediction path. See Appendix A for details.

Addressing this gap requires overcoming two coupled difficulties specific to decentralized routing. First, query–client comparisons must be learned without the graph-wide normalization used by centralized routers such as FedDRM (Wang et al., 2026a). Second, neighbor-local comparisons must remain coherent when composed along a multi-hop referral path. We therefore propose EdgeDRM, a decentralized density-ratio routing framework. It avoids graph-wide normalization by learning each edge comparison through balanced endpoint source classification. To make these comparisons composable, EdgeDRM assigns each client one routing potential shared across its incident edges. Under a common representation, the resulting differences accumulate to the potential difference between the entry client and each reached client. We also characterize discrepancies caused by different local representations. At deployment, each client compares only its neighbors using its own query representation and stored neighbor coefficients, requiring neither a global score vector nor predictions from unvisited clients. This work makes four contributions:

• We formulate inference-time collaboration in decentralized FL as budgeted local expert referral, in which queries move only between neighbors and only visited clients contribute to prediction.

• We develop EdgeDRM, a neighbor-local framework that replaces graph-wide normalization with balanced endpoint source classification. One potential per client makes edge comparisons coherent across hops, and history-weighted aggregation uses only visited clients.

• We show that, under a shared representation, EdgeDRM learns increasingly accurate neighbor comparisons as training data and optimization steps increase. When locally better neighbors consistently guide the route, these comparisons yield cumulative gains in query–client compatibility and lead to the globally best-matched client given sufficient referral budget.

• Across benchmark, scaling, tiered-hospital, and preliminary real-world medical evaluations, EdgeDRM outperforms graph-matched decentralized baselines in most settings, with clear gains on sparse graphs but no consistent advantage under complete connectivity.

## 2 Problem setting, distributional models, and objective

Problem setting. Consider a K-class classification problem with m clients indexed by $V = [ m ] = \{ 1 , 2 , \dots , m \}$ . The clients communicate over a fixed graph $G = ( V , { \dot { E } } )$ . An edge $\{ i , j \} \in E$ allows clients i and j to exchange model information during training and refer queries directly to one another during inference. During inference, each client can communicate only with its neighbors in $\mathcal { N } ( i )$ and cannot access the models or scores of non-neighboring clients. A query x first arrives at an entry client s. A referral transfers the query from its current client to one of its neighbors, and successive referrals form a path in G. The movement budget $L \in  { \mathbb { N } } _ { 0 }$ is an upper bound on the number of such neighbor-to-neighbor transfers after the query arrives. Thus, the query visits at most $L + 1$ clients, including the entry client s.

Each client i has a local dataset $D _ { i } = \{ ( x _ { i r } , y _ { i r } ) \} _ { r = 1 } ^ { n _ { i } }$ consisting of samples drawn independently from a client-specific joint distribution $P _ { i } ^ { X , Y }$ . Let $\begin{array} { r } { N = \sum _ { i = 1 } ^ { m } n _ { i } } \end{array}$ and let $\rho _ { i } = n _ { i } / N$ denote client $i \mathbf { \ ' } _ { \mathbf { S } }$ sample proportion. For each class $k \in [ K ]$ , let $\pi _ { i k } = P _ { i } ^ { Y } ( \{ k \} )$ ) denote the marginal probability of class k at client i, write $\pi _ { i } = ( \pi _ { i 1 } , \dots , \pi _ { i K } )$ , and let $P _ { i k } ^ { X }$ denote the conditional feature distribution given $Y = k$ . The feature marginal is then $\begin{array} { r } { P _ { i } ^ { X } = \sum _ { k = 1 } ^ { K } \pi _ { i k } P _ { i k } ^ { X } } \end{array}$ . We allow the clients’ distributions to differ, so that $\bar { P } _ { i } ^ { X , Y } \neq P _ { j } ^ { X , Y }$ for some $i \neq j$

Distributional models and conditional objective. The joint distribution factorizes as $P _ { i } ^ { X , Y } = P _ { i } ^ { Y | X } P _ { i } ^ { X }$ . Let $g _ { \theta }$ denote a shared encoder and let $Z = g _ { \boldsymbol { \theta } } ( X )$ be the resulting representation. We model $P _ { i } ^ { Y | X }$ under the restriction that it depends on X only through Z. Then rather than modeling $P _ { i } ^ { X }$ directly, we model its induced representation distribution $P _ { i } ^ { Z }$

Conditional label distribution. For each client i, we model the conditional label distribution using the shared encoder $g _ { \theta }$ and a private prediction head $\omega _ { i } = \left( \alpha _ { i } , \beta _ { i } \right)$ :

$$
P _ { i } ^ { Y | X } ( \{ k \} \mid x ; \theta , \omega _ { i } ) = \frac { \exp \{ \alpha _ { i k } + \beta _ { i k } ^ { \top } g _ { \theta } ( x ) \} } { \sum _ { c = 1 } ^ { K } \exp \{ \alpha _ { i c } + \beta _ { i c } ^ { \top } g _ { \theta } ( x ) \} } , \quad i \in V , k \in [ K ] .\tag{1}
$$

For identifiability, we set $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \alpha _ { i k } = 0 } \end{array}$ and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \beta _ { i k } = 0 } \end{array}$ for each client i. The shared encoder provides a common representation across clients, while the client-specific heads allow their conditional label distributions to differ.

Representation marginal distribution. To share information across clients, we jointly model the representation marginals $\{ P _ { i } ^ { Z } \} _ { i = } ^ { m }$ through a common-reference density-ratio model (DRM), following FedDRM (Wang et al., 2026a). Specifically, we assume that the measures $\{ P _ { i } ^ { Z } \} _ { i = 1 } ^ { m }$ admit strictly positive Radon–Nikodym derivatives with respect to a common baseline distribution $F$ and satisfy:

$$
\varphi _ { i } ( z ) : = \{ d P _ { i } ^ { Z } / d F \} ( z ) = \exp \{ b _ { i } + \vartheta _ { i } ^ { \top } z \} , \qquad i \in V ,\tag{2}
$$

and $\varphi _ { i }$ is the density of $P _ { i } ^ { Z }$ relative to $F$ parameterized by $b _ { i }$ and $\vartheta _ { i }$ . For identifiability, we take $F = P _ { 1 } ^ { Z }$ and set $b _ { 1 } = 0$ and $\vartheta _ { 1 } = 0$ . Because all clients share the same baseline distribution, it cancels from every pairwise comparison: log $\{ d P _ { j } ^ { Z } / d \bar { P _ { i } } ^ { Z } \} ( z ) = \log \{ \varphi _ { j } ( z ) / \varphi _ { i } ( z ) \} = ( b _ { j } - b _ { i } ) + ( \vartheta _ { j } - \vartheta _ { i } ) ^ { \top } z$ . Each client is associated with one coefficient pair $( b _ { i } , \vartheta _ { i } )$ . For a connected graph, differences across its edges determine these pairs relative to client 1.

From empirical likelihood to a conditional objective. The baseline distribution F is unknown, so an empiricallikelihood (EL) construction leaves it unrestricted and assigns masses to the distinct pooled representations $z _ { i r } = g _ { \theta } ( x _ { i r } )$ These masses must normalize both $F$ and every tilted distribution $\varphi _ { i } d F$ . To state the connection with source classifica tion, let C denote the source client and define $P _ { \rho } ^ { C , Z }$ by $P _ { \rho } ^ { C } ( \{ i \} ) = \rho _ { i }$ and $P _ { \rho } ^ { Z | C } ( \cdot \mid i ) = P _ { i } ^ { Z }$ . Under the normalized DRM, Bayes’ rule gives:

$$
s _ { i } ( z ) : = P _ { \rho } ^ { C | Z } ( \{ i \} \mid z ) = \frac { \rho _ { i } \varphi _ { i } ( z ) } { \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } ( z ) } = \frac { \exp \{ \log \rho _ { i } + b _ { i } + \vartheta _ { i } ^ { \top } z \} } { \sum _ { c = 1 } ^ { m } \exp \{ \log \rho _ { c } + b _ { c } + \vartheta _ { c } ^ { \top } z \} } .
$$

Thus, $s _ { i } ( z )$ is the posterior probability that a representation z originates from client i. For fixed representations and finite slopes $\vartheta .$ , maximizing EL over both the baseline masses and the free intercepts $b _ { 2 } , \ldots , b _ { m }$ is equivalent, up to a constant independent of these coefficients, to maximizing $\textstyle \sum _ { i , r } \log s _ { i } ( z _ { i r } )$ over the intercepts. Appendix B proves this identity, including existence of the intercept optimum and the treatment of repeated representations. Profiling the masses alone does not give a pointwise equality at arbitrary intercepts. Motivated by this EL connection, we use the softmax expression as a conditional source model and combine it with the conditional-label model to define

$$
\mathcal { L } ( \theta , \omega , b , \vartheta ) = \underbrace { \left[ - \frac { 1 } { N } \sum _ { i = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { i } } \log P _ { i } ^ { \cal Y | X } ( \{ y _ { i r } \} \mid x _ { i r } ; \theta , \omega _ { i } ) \right] } _ { \substack { \mathrm { c o n a l i t i o n a l ~ l a b s l e a r ~ l ~ l a s s i t y } } } + \underbrace { \left[ - \frac { 1 } { N } \sum _ { i = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { i } } \log s _ { i } ( z _ { i r } ) \right] } _ { \substack { \mathrm { c o n a l i t i o n a l ~ l a s s i t y } } } .\tag{3}
$$

conditional-label loss

The conditional-label loss fits client-specific predictors, while the source-classification loss estimates density ratios by identifying each representation’s source. For an edge $\{ i , j \}$ , the odds $s _ { j } ( z ) / s _ { i } ( z )$ provide the comparison used to evaluate referral from client i to client j. These odds measure distributional compatibility, not prediction accuracy; predictions come from the conditional-label models at visited clients. Computing the $\mathrm { g l }$ obal source-classification loss requires all client scores and a graph-wide normalization, which graph-local communication cannot provide. This limitation distinguishes decentralized FL from FedDRM and motivates a conditional source comparison on each communication edge.

## 3 Decentralized Training and Local Expert Referral

Section 2 defines the statistical model and its global objective. We now describe how clients train this model through neighbor communication and use the fitted quantities for referral and prediction.

## 3.1 Decentralized training with graph-local source likelihoods

Because the two terms in (3) impose different communication requirements, we train the model in two stages. Stage I trains the encoders and private prediction heads using the conditional-label loss and neighbor mixing. Stage II freezes the learned encoders, refines the private prediction heads, and estimates the density-ratio coefficients $( \bar { b } _ { i } , \vartheta _ { i } )$ from graph-local source likelihoods.

For training, client i forms the local objective $\mathcal { L } _ { i }$ in (18) from its local conditional-label loss (the first term in $( 3 ) )$ and one local source-classification loss for each incident edge, all evaluated on $D _ { i }$ . The graph objective in (19) then averages these client-local objectives with equal weights: ${ \mathcal { L } } _ { G } { \overset { \vartriangle } { = } } m ^ { - 1 } \sum _ { i } { \mathcal { L } } _ { i }$ . For the conditional-label component, this replaces the sample-proportion weight $\rho _ { i } = n _ { i } / N$ in (3) by $1 / m$ . This equal weighting is the canonical formulation in decentralized optimization and is widely used in decentralized FL, where doubly stochastic neighbor mixing gives each client equal weight (Lalitha et al., 2019; Koloskova et al., 2020; Sun et al., 2023). This weighting convention differs from the centralized setting.

Stage I: conditional-label model training. The model in Section 2 uses a shared encoder parameter θ, but decentralized optimization requires each client to maintain a local copy $\theta _ { i } .$ For compactness, let $\psi _ { i } = \left( \theta _ { i } , \omega _ { i } \right)$ collect the local encoder and prediction-head parameters at client i. Stage I runs for $R _ { \mathrm { s h } }$ communication rounds. At round $r ,$ client i starts from $\psi _ { i } ^ { ( r ) }$ and takes $B _ { \mathrm { s h } }$ SGD steps on its empirical conditional-label loss using $D _ { i }$ . Let $\widetilde { \psi } _ { i } ^ { \left( r \right) }$ denote the resulting encoder and prediction-head parameters. The graph G determines which pairs of clients can exchange these intermediate parameters. Because no central server aggregates all client updates, each client instead combines its own intermediate parameters with those received from its neighbors. A mixing matrix W determines the weights in this local aggregation. Specifically, let W be symmetric and doubly stochastic, with $W _ { i j } = 0$ whenever $j \notin \{ i \} \cup \mathcal { N } ( i )$ Client i then updates its parameters as

$$
\psi _ { i } ^ { ( r + 1 ) } = \sum _ { j \in \{ i \} \cup { \cal N } ( i ) } W _ { i j } \widetilde { \psi } _ { j } ^ { ( r ) } .\tag{4}
$$

For each edge $\{ i , j \} \in E$ , we use the Metropolis weight $W _ { i j } = 1 / ( 1 + \operatorname* { m a x } \{ \deg ( i ) , \deg ( j ) \} )$ and set $W _ { i i } =$ $1 - \textstyle \sum _ { j \neq i } W _ { i j }$ . Neighbor mixing is an established decentralized optimization mechanism (Koloskova et al., 2020), but finitely many rounds need not produce exact consensus, so the learned encoders may differ across clients.

Stage II: frozen-encoder density-ratio fitting. Stage I uses local SGD and neighbor mixing to optimize the clientaveraged conditional-label objective described above. Stage II addresses the remaining source-classification loss, which learns the density ratios by predicting the source client C from a representation $\bar { Z . }$ In centralized $\mathrm { F L } .$ a server can coordinate the scores of all $m$ clients required to evaluate the source probabilities $s _ { i } ( z )$ . In the decentralized setting, however, each client communicates only with its neighbors and therefore cannot evaluate the graph-wide normalization in $s _ { i } ( z )$ . Because each referral traverses one edge, deciding whether to move the query requires only a comparison between the two endpoints. We therefore replace the global m-client source-classification problem with a binary source-classification problem on each edge.

For each edge $e = \{ i , j \} \in E$ with $i < j$ , we form a balanced binary source-classification problem by assigning equal weight to the two endpoint mean losses. Its posterior probability that z comes from client $j$ rather than client i is $q _ { i j } ^ { \mathrm { b a l } } ( z ) : = \varphi _ { j } ( z ) / \{ \varphi _ { i } \tilde { ( z ) } + \varphi _ { j } ( z ) \}$ }, with

$$
\mathrm { l o g i t } q _ { i j } ^ { \mathrm { b a l } } ( z ) = ( b _ { j } - b _ { i } ) + ( \vartheta _ { j } - \vartheta _ { i } ) ^ { \top } z .\tag{5}
$$

This binary problem is the source-classification term in (3) restricted to the endpoints $\{ i , j \}$ , and it learns the coefficient differences needed to compare them. Because client i uses the same coefficient pair $( b _ { i } , \vartheta _ { i } )$ on all its incident edges, the graph links these pairwise comparisons through their shared endpoints. The two endpoint losses receive equal weight, irrespective of $n _ { i }$ and $n _ { j }$ . Thus, the sample-count prior is not included in the training logit. The exact graph-local objective is given in Appendix B.4. After Stage I, clients freeze their encoders and fit their private heads and density-ratio coefficients using local conditional-label and incident-edge source losses. Only sample counts, density-ratio coefficients, and their gradient contributions are exchanged across edges. Encoders, private heads, and raw data remain local. These fitted quantities drive referral and prediction.

## 3.2 Budgeted local expert referral & Prediction using the referral history

Budgeted local expert referral. For a query x, let $\Delta _ { i j } ( x ) : = \log ( \rho _ { j } / \rho _ { i } ) +$ logit $q _ { i j } ^ { \mathrm { b a l } } ( g _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) )$ denote the population pairwise log-odds obtained by adding the sample-count prior to the learned balanced comparison. This quantity provides the criterion for referring the query from client i to client $j .$ At inference, client i substitutes the fitted density-ratio coefficients and its own fitted representation into this quantity, giving the plug-in estimator

$$
\widehat { \Delta } _ { i j } ( x ) : = \log ( \rho _ { j } / \rho _ { i } ) + ( \widehat { b } _ { j } - \widehat { b } _ { i } ) + ( \widehat { \vartheta } _ { j } - \widehat { \vartheta } _ { i } ) ^ { \top } g _ { \widehat { \theta } _ { i } } ( x ) .\tag{6}
$$

Hence training estimates the density-ratio term without sample-count weighting, and inference adds $\log ( \rho _ { j } / \rho _ { i } ) =$ $\log ( n _ { j } / n _ { i } )$ exactly once. Applying these fitted endpoint comparisons sequentially produces a query-dependent route through the graph. For a query x entering at client s under movement budget L, the realized route is

$$
\mathcal { R } _ { L } ( x , s ) = ( \widehat { s } _ { 0 } = s , \widehat { s } _ { 1 } , \ldots , \widehat { s } _ { T } ) , \quad T \leq L , \quad \{ \widehat { s } _ { t - 1 } , \widehat { s } _ { t } \} \in E .\tag{7}
$$

The route construction is detailed in Algorithm 1. We de- Algorithm 1: Budgeted local expert referral.   
scribe its intuition below. During referral, x is passed be- 1: Input: Query x, entry client s, threshold $\eta \geq 0 ,$ and move  
tween neighboring clients together with the visited-client ment budget $L .$   
set, accepted edge scores, and remaining movement bud- 2: $\widehat { s } _ { 0 } \gets s , \bar { V } _ { \mathrm { v i s } } \gets \{ s \} , T \gets 0$   
get. At client $\widehat { s } _ { t }$ , the router scores each unvisited neighbor 3: while $T < L$ do   
using (6) and selects the largest score, with ties broken deter- 4: $U \gets \mathcal { N } ( \widehat { s } _ { T } ) \setminus V _ { \mathrm { v i s } }$   
ministically. The query moves to the selected neighbor only 5: $\mathbf { i f } \ U = \emptyset$ then break   
if this score exceeds η and otherwise terminates. Evaluating 6: $j ^ { \star } \gets \arg \operatorname* { m a x } _ { j \in U } \widehat { \Delta } _ { \widehat { s } _ { T } , j } ( x )$   
a candidate requires only the current client’s representation 7: if $\widehat { \Delta } _ { \widehat { s } _ { T } , j ^ { \star } } ( x ) \leq \eta$ then break   
of x and the stored sample counts and density-ratio coeffi- 8: Send the packet and accepted score to $j ^ { \star }$   
cients of the two endpoints. The candidate need not inspect 9: $T \gets T \dot { + } 1 , \widehat { s } _ { T } \gets j ^ { \star }$   
the query. The procedure uses neither the query label nor 10: $V _ { \mathrm { v i s } }  V _ { \mathrm { v i s } } \cup \{ j ^ { \star } \}$   
11: end while   
information about prediction correctness. It returns the vis-  
12: Output: $( \widehat { s } _ { 0 } , \ldots , \widehat { s } _ { T } )$ and the accepted scores.   
ited route and accepted edge scores for the final prediction   
rule.

Prediction using the referral history. Because referral scores reflect distributional compatibility rather than prediction accuracy directly, earlier visited clients may still provide useful complementary predictions. We therefore aggregate the predictions from all clients on $\mathcal { R } _ { L } ( x , s )$ . Denote the fitted probability vector at client i as $f _ { i } ( x ) : =$ $\left( P _ { i } ^ { Y | X } ( \{ k \} \mid x ; \widehat { \theta } _ { i } , \widehat { \omega } _ { i } ) \right) _ { k = 1 } ^ { K }$ . For each visited client $\widehat { s } _ { t } .$ , let $u _ { 0 } ( x , s ) = 0$ and $\begin{array} { r } { u _ { t } ( x , s ) = \sum _ { \ell = 1 } ^ { t } \widehat { \Delta } _ { \widehat { s } _ { \ell - 1 } , \widehat { s } _ { \ell } } ( x ) } \end{array}$ denote the cumulative accepted edge score from the entry client to $\widehat { s } _ { t }$ for $t = 1 , \dots , T$ . We set $( w _ { 0 } ( x , s ) , \ldots , w _ { T } ( x , s ) ) : =$ softmax $( u _ { 0 } ( x , s ) , \dots , u _ { T } ( x , s ) )$ ). The final prediction is:

$$
f _ { \operatorname* { m i x } } ( x , s ) = \sum _ { t = 0 } ^ { T } w _ { t } ( x , s ) f _ { \widehat { s } _ { t } } ( x ) , \qquad \widehat { y } = \arg \operatorname* { m a x } _ { k \in [ K ] } f _ { \operatorname* { m i x } , k } ( x , s ) .
$$

## 4 Theoretical Results

We establish theoretical guarantees for coefficient estimation and referral gain in EdgeDRM. Our results characterize the accuracy and asymptotic normality of parameter estimation (Section 4.1) and referral gains (Section 4.2). Proofs and additional results are given in Appendix C.

## 4.1 How accurately can we estimate the parameters?

We first study the performance of parameter estimation in stage II. To separate coefficient estimation from representation learning, consider a common representation $g _ { \theta }$ from Section 2, fixed independently of the coefficient-fitting samples. Let ζ collect the free density-ratio coefficients $( \vartheta _ { i } , b _ { i } ) , i \neq 1$ , and private-head parameters $( \alpha _ { i } , \beta _ { i } )$ under the normalizations in Section 2.

We analyze $R$ updates of Stage-II optimization on $\mathcal { L } _ { G }$ in (19), holding $( \vartheta _ { 1 } , b _ { 1 } ) = 0$ fixed and using unbiased mini-batch gradients with step sizes $\eta _ { r } \overset { \_ } { = } 2 / \{ \bar { \mu } ( r + r _ { 0 } ) \}$ for $r = 0 , \ldots , R - 1$ . Here $0 < \mu \leq \cdot$ 1 is a curvature constant and $r _ { 0 } > 0$ is a fixed offset. We refer to Appendix C.1 for further details on conditions on these constants. The following theorem bounds the statistical and optimization errors of the Stage-II output and establishes its asymptotic normality.

Theorem 1. Let $\zeta ^ { \star }$ denote the true coefficient vector ofthe models in Section 2, and suppose $n _ { i } / N \to \rho _ { i } ^ { * } > 0 .$ . Assume the coefficients are identifiable and assume the population Hessian at $\zeta ^ { \star }$ is positive definite and, with probability tending to one, $\mu I \preceq \nabla ^ { 2 } \bar { \mathcal { L } } _ { G } ( \zeta ) \preceq \mu ^ { - 1 } I$ , throughout a convex region containing afinite empirical minimizer and the optimization trajectory, for a constant $0 < \mu \leq 1$ independent of $N .$ . Under the remaining regularity conditions in Appendix C.1, the Stage-II output $\zeta ^ { ( R ) }$ from the specified SGD procedure satisfies, as $N , R \to \infty$

$$
\begin{array} { r } { \| \zeta ^ { ( R ) } - \zeta ^ { \star } \| _ { 2 } = O _ { p } \Big ( N ^ { - 1 / 2 } + R ^ { - 1 / 2 } \Big ) . } \end{array}\tag{8}
$$

$H ,$ in addition, $R / N \to \infty$ , thenfor some $\Sigma _ { G }$ specified in Appendix C.1,

$$
\sqrt { N } \{ \zeta ^ { ( R ) } - \zeta ^ { \star } \} \stackrel { d } { \to } { \mathcal { N } } ( 0 , \Sigma _ { G } ) ,\tag{9}
$$

The first term in (8) is the statistical error from N training samples. The second is the optimization error after R updates. Thus, when $R / N \to \infty ,$ , the optimization error is dominated by $N ^ { - 1 / 2 }$ , and the output has the asymptotic distribution in (9).

We note that, under the stated models, changing client sample proportions does not change the population minimizer $\zeta ^ { \star }$ The proportions need only satisfy $n _ { i } / N \to \rho _ { i } ^ { * } > 0$ , without a specified convergence rate. The coefficient-estimation bound in Theorem 1 implies a bound on the edge-comparison errors used for referral. In particular, for any fixed x with finite $g _ { \boldsymbol { \theta } } ( x )$ , we obtain

$$
\operatorname* { m a x } _ { \{ i , j \} \in E } | \widehat { \Delta } _ { i j } ( x ) - \Delta _ { i j } ( x ) | = O _ { p } \Big ( N ^ { - 1 / 2 } + R ^ { - 1 / 2 } \Big ) .\tag{10}
$$

This comparison-error bound yields the referral guarantee in Section 4.2. We remark that the result applies to coefficient fitting with a common, independently fixed encoder.

## 4.2 How far and how much can referral improve?

Fix a query x and entry client $s ,$ and consider the route $\mathcal { R } _ { L } ( x , s )$ in (7). Using the population coefficients of Section 2, write $h _ { i } ( \boldsymbol { x } ) = \log \rho _ { i } + b _ { i } + \vartheta _ { i } ^ { \top } g _ { \theta } ( \boldsymbol { x } )$ and $h ^ { \star } ( x ) = \operatorname* { m a x } _ { i \in V } h _ { i } ( x )$ . Thus, $\Delta _ { i j } ( x ) = h _ { j } ( x ) - h _ { i } ( x )$ , and $\Gamma _ { L } ( x , s ) = h _ { \widehat { s } _ { T } } ( x ) \dot { - } h _ { s } ( x )$ is the referral gain. Under the DRM, this gain is the log ratio of the terminal and entry clients’ prior-weighted densities. Our referral guarantees rely on the following intuitive structural properties.

Minimum local improvement. Assume there is $\gamma > 0$ such that every nonoptimal client has a neighbor whose potential is at least $\gamma$ higher:

$$
\operatorname* { m a x } _ { j \in { \cal N } ( i ) } \left\{ h _ { j } ( x ) - h _ { i } ( x ) \right\} \geq \gamma \quad \mathrm { w h e n e v e r } \ h _ { i } ( x ) < h ^ { \star } ( x ) .\tag{11}
$$

Thus, a client with no higher-potential neighbor must be globally optimal.

Ascent depth. For $\epsilon \geq 0 ,$ let $H _ { e }$ be the longest strictly ascending path length from $s ,$ with each step choosing a neighbor within 2ϵ of the highest neighbor potential. Then $H _ { \epsilon } \leq m - 1$ , and under the conditions below, it bounds the number of accepted referrals.

Theorem 2. Under the conditions of Theorem 1, use $\zeta ^ { ( R ) }$ for referral at a fixed query x with finite $g _ { \boldsymbol { \theta } } ( x )$ . For any $\delta \in ( 0 , 1 )$ , there exists a constant $C _ { \delta } > 0 ,$ , independent of N and R, such that the following holds with $\epsilon =$ $C _ { \delta } ( N ^ { - 1 / 2 } + R ^ { - 1 / 2 } )$ . Suppose (11) holds, $\eta \geq \epsilon ,$ , and $\gamma > \eta + \epsilon .$ . For all sufficiently large N, R, with probability at least $1 - \delta ,$ Algorithm 1 with movement budget $L \in  { \mathbb { N } } _ { 0 }$ makes $T \le \operatorname* { m i n } \{ L , H _ { \epsilon } \}$ moves and satisfies

$$
\Gamma _ { L } ( x , s ) \geq ( \gamma - 2 \epsilon ) \operatorname* { m i n } \{ L , H _ { \epsilon } \} .\tag{12}
$$

On the same event, $L \geq H ,$ <sub>ϵ</sub> guarantees that the route reaches a maximizer.

The guarantee in (12) lower-bounds $\Gamma _ { L } ( x , s )$ , the improvement in source compatibility, by a product of two terms. The first term, $\gamma - 2 \epsilon$ , lower-bounds the population gain per accepted move. Larger per-step gains allow substantial improvement with fewer referrals. The second term, min $\{ \bar { L } , H _ { \epsilon } \}$ , is the effective step count. A smaller ascent depth $H _ { \epsilon }$ requires less budget to guarantee reaching a global maximizer, while with $L < H _ { \epsilon }$ the route may exhaust its budget first. Note that improvement in source compatibility does not by itself imply improved predictive performance. Appendix C.3 provides the proof and analyzes ascent depth and referral gain in several common graph types.

## 5 Experiments

## 5.1 Experimental setup and evaluation

Benchmarks and topologies. We first evaluate on CIFAR-10, CIFAR-20, and CIFAR-100 benchmarks under 2 non-IID partitions. In the shards-per-client (SPC) partition based on (McMahan et al., 2017), each client receives five equal-sized, label-homogeneous shards for CIFAR-10 and 25 for CIFAR-20/100. In the Dirichlet partition, each client’s class proportions are drawn from a Dirichlet distribution with concentration 0.3. The main benchmark has eight clients. We consider four communication topologies: clients are connected as a PATH, a CYCLE, a connected random graph in which every client has four neighbors (RR4), or a COMPLETE graph. These are standard decentralized communication topologies and are ordered by increasing connectivity. For further details on the benchmarks, graph topologies, and data construction, see Appendix D.1.

Training and inference. We train randomly initialized ResNet-18 models for 800 communication rounds using DFedAvgM (Sun et al., 2023), freeze each local encoder, and then fit the personalized classifier heads and routers for 200 rounds. The routing objective weights the two endpoints of every edge equally. At inference, EdgeDRM starts from a uniformly sampled non-source entry, excludes revisits, and averages predictions from visited experts using temperature-one score weights.

Comparisons and reporting. At test time, each query from source client s is assigned uniformly at random to an entry client other than s. Local trains one model per client using only local data and predicts with the model at that entry. The decentralized baselines DFedAvgM (Sun et al., 2023), DisPFL (Dai et al., 2022), and FedSPD (Lin et al., 2025) also predict with their learned model at that entry. By contrast, we include centralized FedDRM (Wang et al., 2026a) only as an illustrative reference because its client-selection head scores all clients globally and therefore cannot operate under the neighbor-only communication constraints of our decentralized setting. EdgeDRM-NR uses the same trained model but keeps each query at its entry client.

## 5.2 Main results

Table 1: Eight-client decentralized CIFAR benchmark reporting system accuracy (%) across three datasets and two non-IID partitions. Local and centralized FedDRM are illustrative references. Bold cells with blue shading mark the best decentralized result within each topology and setting. All entries are means ± sample SD across 3 seeds.
<table><tr><td rowspan="2">Group</td><td rowspan="2">Method</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-20</td><td colspan="2">CIFAR-100</td></tr><tr><td>Dir-0.3</td><td>5-SPC</td><td>Dir-0.3</td><td>25-SPC</td><td>Dir-0.3</td><td>25-SPC</td></tr><tr><td rowspan="2">Reference</td><td>Local</td><td> $1 6 . 5 9 \pm 3 . 7 8$ </td><td> $1 8 . 4 6 \pm 1 . 4 9$ </td><td> $1 3 . 3 5 \pm 0 . 2 9$ </td><td> $1 1 . 1 7 \pm 0 . 5 4$ </td><td> $4 . 0 3 \pm 0 . 2 3$ </td><td> $2 . 4 9 \pm 0 . 1 0$ </td></tr><tr><td>FedDRM</td><td> $6 3 . 8 1 \pm 1 . 5 4$ </td><td> $6 1 . 2 4 \pm 2 . 2 9$ </td><td> $3 9 . 2 8 \pm 0 . 8 7$ </td><td> $4 2 . 4 2 \pm 1 . 1 2$ </td><td> $2 8 . 6 7 \pm 0 . 5 8$ </td><td> $3 3 . 3 8 \pm 0 . 8 7$ </td></tr><tr><td rowspan="5">PATH</td><td>DFedAvgM</td><td> $2 8 . 5 5 \pm 7 . 6 2$ </td><td> $2 7 . 4 1 \pm 0 . 2 1$ </td><td> $2 3 . 9 4 \pm 0 . 0 3$ </td><td> $2 1 . 4 7 \pm 1 . 0 4$ </td><td> $9 . 2 8 \pm 0 . 6 1$ </td><td> $7 . 0 3 \pm 0 . 4 4$ </td></tr><tr><td>DisPFL</td><td> $2 7 . 1 9 \pm 8 . 0 1$ </td><td> $2 6 . 8 7 \pm 1 . 9 1$ </td><td> ${ \bf 2 6 . 4 3 \pm 0 . 6 5 }$ </td><td> $2 2 . 7 9 \pm 1 . 6 8$ </td><td> $1 0 . 8 3 \pm 0 . 9 6$ </td><td> $6 . 4 1 \pm 0 . 6 4$ </td></tr><tr><td>FedSPD</td><td> $1 8 . 3 4 \pm 5 . 1 7$ </td><td> $2 0 . 0 0 \pm 1 . 1 4$ </td><td> $1 5 . 4 4 \pm 0 . 9 9$ </td><td> $1 4 . 4 1 \pm 0 . 7 8$ </td><td> $4 . 6 8 \pm 0 . 3 8$ </td><td> $4 . 3 0 \pm 0 . 1 1$ </td></tr><tr><td>EdgeDRM-NR</td><td> $2 2 . 0 0 \pm 4 . 8 5$ </td><td> $2 1 . 5 3 \pm 1 . 1 8$ </td><td> $2 1 . 5 7 \pm 0 . 3 5$ </td><td> $1 8 . 5 1 \pm 1 . 7 9$ </td><td> $6 . 8 9 \pm 0 . 6 2$ </td><td> $4 . 2 1 \pm 0 . 2 1$ </td></tr><tr><td>EdgeDRM</td><td> $\mathbf { 3 3 . 0 8 \pm 3 . 5 3 }$ </td><td> $\mathbf { 3 2 . 8 8 \pm 1 . 0 0 }$ </td><td> $2 4 . 8 7 \pm 0 . 3 3$ </td><td> ${ \bf 2 3 . 9 7 \pm 1 . 1 2 }$ </td><td> ${ \bf 1 1 . 1 9 \pm 0 . 5 8 }$  </td><td> ${ \bf 1 1 . 1 5 \pm 0 . 3 4 }$ </td></tr><tr><td rowspan="5">CYCLE</td><td>DFedAvgM</td><td> $3 0 . 3 4 \pm 6 . 2 3$ </td><td> $2 7 . 6 3 \pm 2 . 2 2$ </td><td> $2 5 . 0 0 \pm 0 . 1 5$ </td><td> $2 3 . 1 9 \pm 1 . 1 1$ </td><td> $1 0 . 3 9 \pm 0 . 7 2$ </td><td> $7 . 9 3 \pm 0 . 4 5$ </td></tr><tr><td>DisPFL</td><td> $2 9 . 1 0 \pm 7 . 5 9$ </td><td> $2 8 . 0 2 \pm 1 . 1 2$ </td><td> ${ \bf 2 7 . 5 4 \pm 0 . 3 8 }$ </td><td> $2 4 . 4 9 \pm 2 . 0 6$ </td><td> $1 1 . 4 2 \pm 0 . 5 0$ </td><td> $6 . 9 9 \pm 0 . 6 1$ </td></tr><tr><td>FedSPD</td><td> $2 0 . 0 5 \pm 4 . 0 1$ </td><td> $2 0 . 9 5 \pm 1 . 1 3$ </td><td> $1 5 . 3 9 \pm 0 . 6 5$ </td><td> $1 4 . 7 8 \pm 1 . 0 4$ </td><td> $5 . 0 2 \pm 0 . 2 0$ </td><td> $4 . 7 4 \pm 0 . 4 1$ </td></tr><tr><td>EdgeDRM-NR</td><td> $2 2 . 9 2 \pm 4 . 7 1$ </td><td> $2 1 . { \overset { \cdot } { 9 } } { \overset { \cdot } { 5 } } \pm 0 . 7 { \overset { \cdot } { 5 } }$ </td><td> $2 1 . 9 9 \pm 0 . 2 2$ </td><td> $1 9 . 0 4 \pm 1 . 6 2$ </td><td> $7 . 4 9 \pm 0 . 7 1$ </td><td> $4 . 3 2 \pm 0 . 3 9$ </td></tr><tr><td>EdgeDRM</td><td> ${ \bf 3 5 . 2 0 \pm 2 . 3 0 }$ </td><td> ${ \bf 3 4 . 7 9 \pm 0 . 6 3 }$ </td><td> $2 5 . 9 9 \pm 0 . 5 9$ </td><td> ${ \bf 2 5 . 6 8 \pm 0 . 9 4 }$ </td><td> ${ \bf 1 2 . 4 0 \pm 0 . 6 3 }$ </td><td> ${ \bf 1 2 . 0 8 \pm 0 . 4 2 }$ </td></tr><tr><td rowspan="5">RR4</td><td>DFedAvgM</td><td> $4 0 . 1 0 \pm 1 . 8 6$ </td><td> $3 7 . 5 8 \pm 1 . 0 5$ </td><td> $2 8 . 8 3 \pm 0 . 7 3$ </td><td> $2 8 . 1 4 \pm 1 . 4 1$ </td><td> $1 3 . 9 2 \pm 0 . 9 0$ </td><td> $1 2 . 1 4 \pm 0 . 5 4$ </td></tr><tr><td>DisPFL</td><td> $3 0 . 3 1 \pm 7 . 7 7$ </td><td> $3 0 . 7 5 \pm 1 . 4 5$ </td><td> $2 8 . 8 9 \pm 1 . 2 4$ </td><td> $2 6 . 0 7 \pm 2 . 6 0$ </td><td> $1 2 . 6 4 \pm 0 . 7 8$ </td><td> $7 . 5 0 \pm 0 . 6 2$ </td></tr><tr><td>FedSPD</td><td> $1 9 . 5 0 \pm 4 . 9 1$ </td><td> $2 1 . 6 9 \pm 0 . 9 3$ </td><td> $1 5 . 8 5 \pm 1 . 8 3$ </td><td> $1 4 . 4 4 \pm 1 . 5 7$ </td><td> $5 . 4 1 \pm 0 . 5 2$ </td><td> $4 . 4 9 \pm 0 . 2 2$ </td></tr><tr><td>EdgeDRM-NR</td><td> $2 5 . 3 5 \pm 4 . 5 4$ </td><td> $2 3 . 4 4 \pm 1 . 2 3$ </td><td> $2 2 . 7 2 \pm 0 . 6 0$ </td><td> $2 0 . 0 9 \pm 1 . 8 1$ </td><td> $8 . 0 5 \pm 0 . 4 0$ </td><td> $5 . 5 3 \pm 0 . 1 8$ </td></tr><tr><td>EdgeDRM</td><td> ${ \bf 5 0 . 1 2 \pm 2 . 2 1 }$  </td><td> ${ \bf 4 6 . 1 9 \pm 1 . 1 7 }$ </td><td> ${ \bf 2 9 . 9 1 \pm 0 . 7 9 }$ </td><td> ${ \bf 3 1 . 5 1 \pm 0 . 5 2 }$ </td><td> ${ \bf 1 7 . 5 9 \pm 0 . 5 9 }$ </td><td> ${ \bf 2 0 . 0 1 \pm 0 . 5 2 }$ </td></tr><tr><td rowspan="5"></td><td>DFedAvgM</td><td> $5 5 . 0 4 \pm 1 . 4 7$ </td><td> ${ \bf 5 2 . 9 1 \pm 1 . 4 3 }$  </td><td> ${ \bf 3 2 . 6 2 \pm 0 . 5 9 }$  </td><td> ${ \bf 3 6 . 1 0 \pm 0 . 3 8 }$  </td><td> ${ \bf 2 1 . 0 2 \pm 0 . 4 8 }$  </td><td> ${ \bf 2 3 . 8 1 \pm 1 . 0 8 }$ </td></tr><tr><td>DisPFL</td><td> $3 6 . 6 3 \pm 5 . 0 8$ </td><td> $3 4 . 3 5 \pm 0 . 8 8$ </td><td> $3 0 . 1 0 \pm 0 . 9 9$ </td><td> $2 8 . 6 0 \pm 1 . 9 8$ </td><td> $1 5 . 0 9 \pm 0 . 9 7$ </td><td> $1 0 . 8 8 \pm 0 . 5 5$ </td></tr><tr><td>COMPLETE FedSPD</td><td> $2 7 . 3 3 \pm 7 . 0 1$ </td><td> $2 7 . 2 9 \pm 3 . 5 8$ </td><td> $1 5 . 3 5 \pm 1 . 1 9$ </td><td> $1 5 . 2 1 \pm 0 . 5 8$ </td><td> $6 . 0 1 \pm 0 . 6 0$ </td><td> $6 . 1 4 \pm 0 . 3 8$ </td></tr><tr><td>EdgeDRM-NR</td><td> $2 7 . 8 5 \pm 4 . 0 3$ </td><td> $2 5 . 8 0 \pm 0 . 9 3$ </td><td> $2 3 . 6 4 \pm 0 . 4 6$ </td><td> $2 1 . 1 2 \pm 1 . 4 4$ </td><td> $9 . 3 1 \pm 0 . 3 7$ </td><td> $6 . 8 8 \pm 0 . 2 3$ </td></tr><tr><td>EdgeDRM</td><td> ${ \bf 5 6 . 6 2 \pm 2 . 8 7 }$ </td><td> $5 2 . 8 4 \pm 1 . 8 6$ </td><td> $3 1 . 3 1 \pm 1 . 0 2$ </td><td> $3 4 . 1 2 \pm 0 . 3 1$ </td><td> $2 0 . 0 2 \pm 0 . 6 3$ </td><td> $2 3 . 6 5 \pm 0 . 6 7$ </td></tr></table>

CIFAR benchmark results. The main results in Table 1 yield three main observations. First, EdgeDRM is the strongest graph-constrained method in most settings. It ranks first in 16 of the 18 dataset–partition configurations on the three

sparse graphs and in 17 of the 24 graph–configuration pairs overall. Its mean accuracy difference relative to the strongest competing decentralized method in each sparse-graph setting is +3.61 percentage points. The two sparse-graph exceptions both occur on CIFAR-20/Dirichlet-0.3, which exhibits milder client label skew than CIFAR-20/25-SPC in the realized partitions. Across the sparse graphs, its mean margin over the strongest competing decentralized method is 4.75 percentage points under SPC versus 2.48 under Dirichlet-0.3, a difference of 2.27 points. Taken together with the EdgeDRM-NR ablation below, the results suggest that learned referral helps exploit stronger client specialization. Second, the COMPLETE graph, equivalent to the central server case, EdgeDRM is effectively tied with DFedAvgM, averaging only 0.49 points lower across settings. So full graph connectivity narrows the advantage of EdgeDRM. Both methods nevertheless remain below centralized FedDRM in all six settings, yielding the gap between routing designed for centralized setting. Third, referral itself provides a large and universal gain. EdgeDRM outperforms EdgeDRM-NR in all configurations, by 3.30–28.77 points and 11.87 points on average.

Where the referral gain comes from. We conduct an ablation to isolate the effect of the routing rule. Every method uses the terminal expert’s hard prediction, without averaging predictions across visited experts. Besides EdgeDRM and EdgeDRM-NR, we also compare Count-only, which triggers a referral when log $( n _ { j } / n _ { i } ) > 0$ , and Random, which samples uniformly among unvisited neighbors. The improvement over EdgeDRM-NR and Random shows that selecting suitable neighbors is more effective than null/arbitrary referral. Comparison with Count-only further shows that the gain comes from the trained router rather than solely from differences in local sample sizes.

![](images/941b3969e59063c0f1a04fc81e63b48cac9f6f149e9f51ae256514093b601db1.jpg)  
Figure 2: Fixed-expert hard-terminal routing ablation on CIFAR-10 and CIFAR-100 with Dirichlet-0.3 partitions and RR4.

## Scaling in number of clients. We test whether the be-

havior observed with few clients persists as the client population grows substantially. Table 2 shows that EdgeDRM remains the strongest decentralized method in both settings. We note that, because the total CIFAR-10 data pool is fixed, increasing the number of clients reduces the average local sample sizes, which likely contributes to the lower accuracy relative to the results in Table 1.

Table 2: CIFAR-10/RR4 system accuracy with 32 clients (%; mean ± SD over three seeds).
<table><tr><td rowspan="2">Setting</td><td>Reference</td><td colspan="5">Decentralized</td><td>Reference</td></tr><tr><td>Local</td><td>DFedAvgM</td><td>DisPFL</td><td>FedSPD</td><td>EdgeDRM-NR</td><td>EdgeDRM</td><td>FedDRM</td></tr><tr><td>Dir-0.3</td><td> $1 6 . 4 5 \pm 1 . 1 5$ </td><td> $2 7 . 7 9 \pm 3 . 2 5$ </td><td> $2 3 . 7 1 \pm 2 . 6 5$ </td><td> $1 7 . 8 2 \pm 0 . 6 9$ </td><td> $2 1 . 5 2 \pm 1 . 5 0$ </td><td> ${ \bf 3 0 . 0 6 \pm 1 . 1 6 }$ </td><td> $3 8 . 4 5 \pm 0 . 5 2$ </td></tr><tr><td>5-SPC</td><td> $1 8 . 1 1 \pm 0 . 4 7$ </td><td> $2 7 . 9 2 \pm 1 . 2 2$ </td><td> $2 4 . 8 7 \pm 0 . 9 5$ </td><td> $2 0 . 5 4 \pm 0 . 7 5$ </td><td> $2 2 . 7 6 \pm 1 . 2 0$ </td><td> ${ \bf 3 1 . 7 0 \pm 0 . 3 9 }$ </td><td> $3 5 . 9 6 \pm 0 . 5 2$ </td></tr></table>

## 5.3 Referral traffic

We conduct an ablation study to examine how a sample-count prior changes referral traffic across hospital tiers. We use 32 clients: 16 small, eight medium, and eight large hospitals holding 250, 1000, and 4000 training examples, respectively. We evaluate two graph structures. In the layered graph, each tier forms a cycle, every small hospital connects to one medium hospital, and medium and large hospitals are matched oneto-one. Thus, a small hospital can reach the large tier only through the medium tier, requiring at least two hops. The shortcut graph retains all edges and adds one direct small–large edge for each small hospital, making a large hospital reachable from every small hospital in one hop. EdgeDRM $\mathtt { w } / \mathtt { o }$ prior applies the same history-weighted referral rule after removing the sample-count offset $\log ( n _ { j } / n _ { i } )$ and serves as a samplesize-agnostic comparison. In Figure 3, EdgeDRM $\mathtt { w } / \mathtt { o }$ prior largely preserves the EdgeDRM-NR tier proportions, whereas EdgeDRM substantially increases large-hospital traffic, with an even stronger concentration when shortcut edges provide direct access to large hospitals. This pattern captures a realistic healthcare trade-off: the sample-count prior is a plausible way to represent the stronger expertise of larger hospitals, but systematically trusting those hospitals concentrates demand and increases their service burden. The shortcut amplification further highlights the need for tiered referral systems: allowing patients to bypass intermediate hospitals and go directly to the largest regional centers can overwhelm scarce specialist capacity, whereas staged referrals distribute demand across levels of care.

![](images/c540804a6d969d26c74bb5f96356c539504f489d9a4d2053995c58a25816cb50.jpg)  
Figure 3: Referral traffic by hospital tier on two topologies. Bars show the mean share of visitedclassifier calls over 3 seeds.

## 5.4 Real medical data

To assess EdgeDRM beyond synthetic benchmarks, we conduct a real-world skin-lesion classification study on ISIC2019 using the six source-center assignments distributed with FLamby (du Terrail et al., 2022). Unlike the CIFAR benchmarks, where client heterogeneity is imposed through synthetic label-skew partitions, these clients follow source and acquisition metadata from real medical image collections. Every test image is evaluated at all six entry

Table 3: ISIC2019 balanced accuracy (%; three-seed mean). Parentheses show the EdgeDRM change from DFedAvgM.
<table><tr><td>Graph</td><td>DFedAvgM</td><td>EdgeDRM</td></tr><tr><td>PATH</td><td>28.45</td><td>28.62 (+0.18)</td></tr><tr><td>CYCLE</td><td>27.20</td><td>28.35 (+1.15)</td></tr><tr><td>RR3</td><td>27.45</td><td>27.76 (+0.31)</td></tr><tr><td>COMPLETE</td><td>28.93</td><td>28.79 (-0.14)</td></tr></table>

clients, and we report pooled balanced accuracy, which weights recall equally across classes to account for the pronounced class imbalance in ISIC2019. Appendix D.3 provides the data provenance, split audit, graph, training, and evaluation details. EdgeDRM matches or improves upon the graph-matched DFedAvgM baseline on three of the four topologies. On COMPLETE, it is 0.14 points lower, consistent with previous results, where full connectivity likewise narrows or eliminates the advantage of learned local referral.

## 6 Conclusion and Limitations

EdgeDRM formulates inference-time collaboration in decentralized federated learning as budgeted local expert referral. A query moves only between neighbors on the communication graph, and only visited experts contribute to its final prediction. Neighbor-local density-ratio comparisons make referral trainable without a central router or global expert scoring, while our analysis relates referral gains to comparison accuracy and graph structure.

Across six CIFAR settings, EdgeDRM ranks first among graph-matched decentralized methods in 16 of the 18 sparsegraph comparisons, with a mean accuracy difference of +3.61 percentage points relative to the strongest competing method in each comparison. The COMPLETE graph gives EdgeDRM its highest absolute accuracy in every setting, but its mean advantage over DFedAvgM disappears (−0.49 points). Together, these results establish graph-constrained referral as a distinct inference setting and a useful mechanism under sparse connectivity, while showing that greater connectivity can improve absolute accuracy without increasing the relative benefit of learned referral. Closing the gap between the theory and learned neural representations remains future work.

## References

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient¨ learning of deep networks from decentralized data. In Aarti Singh and Xiaojin (Jerry) Zhu, editors, Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, AISTATS 2017, 20-22 April 2017, Fort Lauderdale, FL, USA, volume 54 of Proceedings ofMachine Learning Research, pages 1273–1282. PMLR, 2017. URL http://proceedings.mlr.press/v54/mcmahan17a.html.

Anusha Lalitha, Osman Cihan Kilinc, Tara Javidi, and Farinaz Koushanfar. Peer-to-peer federated learning on graphs. arXiv preprint arXiv:1901.11173, 2019. URL https://arxiv.org/abs/1901.11173.

Tao Sun, Dongsheng Li, and Bao Wang. Decentralized federated averaging. IEEE Trans. Pattern Anal. Mach. Intell., 45(4):4289–4301, 2023. doi: 10.1109/TPAMI.2022.3196503. URL https://doi.org/10.1109/TPAMI.2022. 3196503.

Valentina Zantedeschi, Aurelien Bellet, and Marc Tommasi. Fully decentralized joint learning of personalized models´ and collaboration graphs. In Silvia Chiappa and Roberto Calandra, editors, The 23rd International Conference on Artificial Intelligence and Statistics, AISTATS 2020, 26-28 August 2020, Online [Palermo, Sicily, Italy], volume 108 of Proceedings ofMachine Learning Research, pages 864–874. PMLR, 2020. URL http://proceedings.mlr. press/v108/zantedeschi20a.html.

Rong Dai, Li Shen, Fengxiang He, Xinmei Tian, and Dacheng Tao. Dispfl: Towards communication-efficient personalized federated learning via decentralized sparse training. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors,´ International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 4587–4604. PMLR, 2022. URL https://proceedings.mlr.press/v162/dai22b.html.

I-Cheng Lin, Osman Yagan, and Carlee Joe-Wong. Fedspd: A soft-clustering approach for personalized decentralized federated learning. In Silvia Chiappa and Sara Magliacane, editors, Conference on Uncertainty in Artificial Intelligence, Rio Othon Palace, Rio de Janeiro, Brazil, 21-25 July 2025, volume 286 of Proceedings ofMachine Learning Research, pages 2618–2641. PMLR, 2025. URL https://proceedings.mlr.press/v286/lin25a. html.

Xiang Li, Chendi Wang, Buxin Su, Qi Long, and Weijie Su. Mitigating the privacy-utility trade-off in decentralized federated learning via f-differential privacy. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Ivan Vladimir Meza Ru´ ´ız, and Arturo Loaiza-Bonilla, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/ hash/f6913c4fd14f608077a4145f2744ec89-Abstract-Conference.html.

Zijian Wang, Xiaofei Zhang, Xin Zhang, Yukun Liu, and Qiong Zhang. Beyond aggregation: Guiding clients in heterogeneous federated learning. In International Conference on Learning Representations, volume 2026, pages 149981–150004, 2026a.

Matthias Reisser, Christos Louizos, Efstratios Gavves, and Max Welling. Federated mixture of experts. arXiv preprint arXiv:2107.06724, 2021.

Yehya Farhat, Hamza ElMokhtar Shili, Fangshuo Liao, Chen Dun, Mirian Hipolito Garcia, Guoqing Zheng, Ahmed Awadallah, Robert Sim, Dimitrios Dimitriadis, and Anastasios Kyrillidis. Learning to specialize: Joint gating-expert training for adaptive moes in decentralized settings. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Ivan Vladimir Meza Ru´ ´ız, and Arturo Loaiza-Bonilla, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/ hash/03bb44eceb94537ee2e2f00a8dca60b1-Abstract-Conference.html.

Ashiqur R. KhudaBukhsh, Peter J. Jansen, and Jaime G. Carbonell. Distributed learning in expert referral networks. In Gal A. Kaminka, Maria Fox, Paolo Bouquet, Eyke Hullermeier, Virginia Dignum, Frank Dignum, and Frank¨ van Harmelen, editors, ECAI 2016 - 22nd European Conference on Artificial Intelligence, 29 August-2 September 2016, The Hague, The Netherlands - Including Prestigious Applications of Artificial Intelligence (PAIS 2016), volume 285 of Frontiers in Artificial Intelligence and Applications, pages 1620–1621. IOS Press, 2016. doi: 10.3233/978-1-61499-672-9-1620. URL https://doi.org/10.3233/978-1-61499-672-9-1620.

Dmitry Baranchuk, Dmitry Persiyanov, Anton Sinitsin, and Artem Babenko. Learning to route in similarity graphs. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings ofthe 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings ofMachine Learning Research, pages 475–484. PMLR, 2019. URL http://proceedings.mlr.press/v97/baranchuk19a.html.

Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian U. Stich. A unified theory of decentral ized SGD with changing topology and local updates. In Proceedings ofthe 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5381–5393. PMLR, 2020. URL http://proceedings.mlr.press/v119/koloskova20a.html.

Jean Ogier du Terrail, Samy-Safwan Ayed, Edwige Cyffers, Felix Grimberg, Chaoyang He, Regis Loeb, Paul Mangold, Tanguy Marchand, Othmane Marfoq, Erum Mushtaq, Boris Muzellec, Constantin Philippenko, Santiago Silva, Maria Telenczuk, Shadi Albarqouni, Salman Avestimehr, Aurelien Bellet, Aymeric Dieuleveut, Martin Jaggi, Sai Praneeth´ Karimireddy, Marco Lorenzi, Giovanni Neglia, Marc Tommasi, and Mathieu Andreux. Flamby: Datasets and benchmarks for cross-silo federated learning in realistic healthcare settings. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets\_and\_Benchmarks.html.

Zijian Wang, Pengfei Li, Guangyu Yang, and Qiong Zhang. FedSPM: Routing-enabled federated learning under dual heterogeneity via semiparametric mixture. arXiv preprint arXiv:2607.04085, 2026b. URL https://arxiv.org/ abs/2607.04085v1.

Rui Ye, Zhenyang Ni, Fangzhao Wu, Siheng Chen, and Yanfeng Wang. Personalized federated learning with inferred collaboration graphs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 39801–39817. PMLR, 2023. URL https://proceedings.mlr.press/v202/ye23b.html.

Salma Kharrat, Marco Canini, and Samuel Horvath. DPFL: decentralized personalized federated learning. In Yingzhen´ Li, Stephan Mandt, Shipra Agrawal, and Mohammad Emtiyaz Khan, editors, International Conference on Artificial Intelligence and Statistics, AISTATS 2025, Mai Khao, Thailand, 3-5 May 2025, volume 258 of Proceedings of Machine Learning Research, pages 5086–5094. PMLR, 2025. URL https://proceedings.mlr.press/v258/ kharrat25a.html.

Jon M. Kleinberg. The small-world phenomenon: an algorithmic perspective. In F. Frances Yao and Eugene M. Luks, editors, Proceedings of the Thirty-Second Annual ACM Symposium on Theory of Computing, May 21-23, 2000, Portland, OR, USA, pages 163–170. ACM, 2000. doi: 10.1145/335305.335325. URL https://doi.org/10. 1145/335305.335325.

Liang Ma, Mudhakar Srivatsa, Derya Cansever, Xifeng Yan, Sue Kase, and Michelle Vanni. Performance bounds of decentralized search in expert networks for query answering. ACM Trans. Knowl. Discov. Data, 13(2):18:1–18:23, 2019. doi: 10.1145/3300230. URL https://doi.org/10.1145/3300230.

Masashi Sugiyama, Taiji Suzuki, and Takafumi Kanamori. Density ratio estimation in machine learning. Cambridge University Press Cambridge, 2012.

Xiaoye Jiang, Lek-Heng Lim, Yuan Yao, and Yinyu Ye. Statistical ranking and combinatorial Hodge theory. Math. Program., 127(1, Ser. B):203–244, 2011. ISSN 0025-5610,1436-4646. doi: 10.1007/s10107-010-0419-x. URL https://doi.org/10.1007/s10107-010-0419-x.

Carlos Hernandez-P´ erez, Marc Combalia, Sebastian Podlipnik, Noel CF Codella, Veronica Rotemberg, Allan C Halpern,´ Ofer Reiter, Cristina Carrera, Alicia Barreiro, Brian Helba, et al. BCN20000: Dermoscopic lesions in the wild. Scientific data, 11(1):641, 2024.

Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data, 5(1):180161, 2018.

Noel C. F. Codella, David A. Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin K. Mishra, Harald Kittler, and Allan Halpern. Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (ISIC), 2018. URL https://doi.org/10.1109/ISBI.2018.8363547.

## A Related Work

Federated routing and mixtures of experts. FedDRM is the closest starting point for our work. It learns a globally indexed source-classification head and asks a server to compare all clients for every query (Wang et al., 2026a). FedSPM instead uses a semiparametric mixture to represent inter- and intra-client heterogeneity (Wang et al., 2026b). Federated mixture-of-experts methods also construct specialized predictors: FedMix adapts a client-specific subset of experts, whereas DDOME jointly trains gates and experts for on-the-fly selection (Reisser et al., 2021; Farhat et al., 2025). These methods establish the value of specialization, but they do not address our information constraint: at inference, no server or client can evaluate a global gate. EdgeDRM replaces global selection with a sequence of graph-supported comparison and forms its prediction only from the experts reached by that sequence.

Personalized and decentralized federated learning. Personalized FL decides which information clients should share during training. pFedGraph learns a weighted collaboration graph at a server (Ye et al., 2023). Fully decentralized methods learn personalized models and sparse collaboration graphs, exchange sparse model masks, optimize budgeted collaborator selection, or form soft clusters through peer-to-peer updates (Zantedeschi et al., 2020; Dai et al., 2022; Kharrat et al., 2025; Lin et al., 2025). DFedAvgM combines local momentum with neighbor model averaging (Sun et al., 2023), while decentralized SGD theory characterizes gossip and local-update convergence (Koloskova et al., 2020). These approaches determine how models collaborate during training and ordinarily return a model at each client. Our problem begins after training: for each new query, the system must decide whether and where that query should move. We therefore use DFedAvgM as the primary graph-matched training baseline and isolate referral with EdgeDRM-NR, evaluated on the same EdgeDRM predictors.

Expert referral and local graph search. Expert-referral networks are closer to our deployment protocol because they model accept-or-forward decisions on sparse graphs. DIEL learns topic-dependent neighbor expertise from observed referral outcomes through a distributed active-learning rule (KhudaBukhsh et al., 2016). Decentralized-search theory studies when local connections yield short query paths (Kleinberg, 2000; Ma et al., 2019), and learned similarity-graph routing can use graph-wide search supervision (Baranchuk et al., 2019). EdgeDRM requires neither logged referral outcomes nor target paths. Its local scores are learned from the static datasets held at neighboring clients, and its movement budget makes the access constraint explicit.

Density ratios and pairwise consistency. Classifier-based density-ratio estimation recovers pointwise density ratios from prior-corrected posterior odds (Sugiyama et al., 2012). Statistical Hodge theory shows how pairwise comparison flows decompose into gradient, cyclic, and harmonic components (Jiang et al., 2011). We do not propose a new density-ratio estimator in isolation. Instead, one potential per client induces compatible edge differences, while balanced endpoint classification estimates pairwise density-ratio terms and the sample-count prior is added only at inference. We then analyze how estimation error, representation disagreement, and graph structure affect the resulting referral policy.

## B Derivation of the EdgeDRM Objective

This appendix establishes the precise connection between empirical likelihood (EL) and the global client-source conditional likelihood in Section 2. The equality requires profiling both the baseline masses and the density-ratio intercepts at fixed representations and slopes. We then derive the graph-local conditional objective used by EdgeDRM.

## B.1 Empirical likelihood under the reference density-ratio model

Our objective is to estimate the label model and the representation density ratios by maximizing their joint empirical likelihood while treating the reference distribution as an unrestricted nuisance parameter. Recall that client $i \in V { \bar { = } } [ m ]$ holds $D _ { i } = \{ ( x _ { i r } , y _ { i r } ) \bar { \} } _ { r = 1 } ^ { n _ { i } }$ with $n _ { i } > 0$ , total sample size $\begin{array} { r } { N = \sum _ { i = 1 } ^ { m } n _ { i } . } \end{array}$ , and sample proportion $\rho _ { i } = n _ { i } / N$ . Given the shared encoder $g _ { \theta }$ , write $z _ { i r } = g _ { \theta } ( x _ { i r } )$ for the observed representation and let $P _ { i } ^ { Z }$ denote the induced representation distribution on client i. For client i, the reference density-ratio model specifies

$$
d P _ { i } ^ { Z } ( z ) = \varphi _ { i } ( z ) d F ( z ) , \qquad \varphi _ { i } ( z ) = \exp \{ b _ { i } + \vartheta _ { i } ^ { \top } z \} , \qquad F = P _ { 1 } ^ { Z } , \quad b _ { 1 } = 0 , \quad \vartheta _ { 1 } = 0 .
$$

The reference constraint makes $\varphi _ { 1 } \equiv 1$ , whereas the remaining density ratios describe how each client distribution tilts the common baseline F. For the empirical-likelihood construction, treat the common representations as fixed. Let $\xi _ { 1 } , \ldots , \xi _ { n _ { z } }$ be the distinct pooled representations, let $k _ { t } = \# \{ ( u , r ) : z _ { u r } = \xi _ { t } \}$ , and write $t ( u , r )$ for the index satisfying $z _ { u r } = \xi _ { t ( u , r ) }$ . Thus, $k _ { t } \geq 1$ and $\textstyle \sum _ { t } k _ { t } = N$ . We represent the unknown reference distribution by assigning mass $p _ { t } = F ( \{ \xi _ { t } \} )$ to each distinct pooled representation $\xi _ { t }$ . Let $P _ { u } ^ { Y , Z }$ denote the joint distribution of the label and representation at client u. Following the empirical-likelihood construction in Wang et al. (2026a), we first define the joint empirical log-likelihood of the observed labels and representations as

$$
\ell _ { \mathrm { E L } } ( p , \theta , \omega , b , \vartheta ) : = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { Y , Z } ( \{ ( y _ { u r } , z _ { u r } ) \} ) .
$$

Because the conditional label model depends on X only through $Z = g _ { \boldsymbol { \theta } } ( X )$ , the joint distribution factorizes into the conditional label distribution and the representation marginal:

$$
P _ { u } ^ { Y , Z } ( \{ ( y _ { u r } , z _ { u r } ) \} ) = P _ { u } ^ { Y | X } ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \omega _ { u } ) P _ { u } ^ { Z } ( \{ z _ { u r } \} ) .
$$

The density-ratio model and the definition of the empirical baseline masses then give

$$
P _ { u } ^ { Z } ( \{ z _ { u r } \} ) = \varphi _ { u } ( z _ { u r } ) F ( \{ z _ { u r } \} ) = \varphi _ { u } ( z _ { u r } ) p _ { t ( u , r ) } .
$$

Substituting these two identities into the joint empirical log-likelihood yields

$$
\begin{array} { l } { \displaystyle \ell _ { \mathrm { E L } } ( p , \theta , \boldsymbol { \omega } , \boldsymbol { b } , \boldsymbol { \vartheta } ) = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log \left[ P _ { u } ^ { Y | X } ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \boldsymbol { \omega } _ { u } ) P _ { u } ^ { Z } ( \{ z _ { u r } \} ) \right] } \\ { \displaystyle = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log \left[ P _ { u } ^ { Y | X } ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \boldsymbol { \omega } _ { u } ) \varphi _ { u } ( z _ { u r } ) p _ { t ( u , r ) } \right] } \\ { \displaystyle = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { Y | X } ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \boldsymbol { \omega } _ { u } ) + \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log \varphi _ { u } ( z _ { u r } ) + \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log p _ { t ( u , r ) } } \\ { \displaystyle = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { Y | X } ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \boldsymbol { \omega } _ { u } ) + \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log \varphi _ { u } ( z _ { u r } ) + \sum _ { t = 1 } ^ { n _ { z } } k _ { t } \log p _ { t } . } \end{array}
$$

The empirical-likelihood problem maximizes this joint objective over the model parameters and baseline masses subject to the requirement that $F$ and every tilted measure $P _ { i } ^ { Z }$ are probability distributions. The baseline normalization and the client-specific normalizations therefore impose

$$
p _ { t } \geq 0 , \qquad \sum _ { t = 1 } ^ { n _ { z } } p _ { t } = 1 , \qquad \sum _ { t = 1 } ^ { n _ { z } } p _ { t } \varphi _ { i } ( \xi _ { t } ) = 1 , \quad i = 2 , \ldots , m .
$$

Assigning one mass to each distinct representation avoids counting the same atom of $F$ more than once, while the multiplicity $k _ { t }$ preserves its full contribution to the likelihood. A finite objective value further requires $p _ { t } > 0$ for every observed atom. Because the normalization constraints couple $p$ and b, profiling only $p$ at fixed $( b , \vartheta )$ does not generally recover the client-source likelihood pointwise in $b .$ The exact equivalence therefore requires profiling both the baseline masses and the density-ratio intercepts, as established next.

## B.2 Exact equivalence after profiling masses and intercepts

At fixed representations and slopes, our objective is to profile the baseline masses and density-ratio intercepts and express the resulting empirical likelihood as a conditional source-classification likelihood. To define this conditional likelihood, introduce the source variable $C$ and let $P _ { \rho } ^ { C , Z }$ denote the pooled source–representation distribution with

$$
P _ { \rho } ^ { C } ( \{ i \} ) = \rho _ { i } , \qquad P _ { \rho } ^ { Z | C } ( \cdot \mid i ) = P _ { i } ^ { Z } , \qquad \rho _ { i } = n _ { i } / N .
$$

Under a normalized density-ratio model, Bayes’ rule gives the posterior source probability

$$
s _ { i } ( z ) : = P _ { \rho } ^ { C | Z } ( \{ i \} \mid z ) = \frac { \rho _ { i } \varphi _ { i } ( z ) } { \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } ( z ) } .
$$

For any finite candidate coefficients, this softmax expression remains a valid conditional source model even when no normalized baseline $F$ exists for those coefficients. For the profiling identity, fix $( \theta , \omega )$ and the resulting pooled representations, and fix finite slopes ϑ with $\vartheta _ { 1 } = 0$ . Define the joint empirical-likelihood feasible set

$$
\mathcal A _ { \vartheta } : = \left\{ ( p , b ) \in ( 0 , \infty ) ^ { n _ { z } } \times \mathbb R ^ { m } : b _ { 1 } = 0 , \quad \sum _ { t = 1 } ^ { n _ { z } } p _ { t } = 1 , \quad \sum _ { t = 1 } ^ { n _ { z } } p _ { t } \exp \{ b _ { i } + \vartheta _ { i } ^ { \top } \xi _ { t } \} = 1 \mathrm { f o r } i = 2 , \ldots , m \right\} .
$$

Our claim is that jointly maximizing the empirical likelihood over the baseline masses and intercepts is equivalent to maximizing the conditional source likelihood over the intercepts alone:

$$
\operatorname* { m a x } _ { ( p , b ) \in \mathcal { A } _ { \theta } } \ell _ { \mathrm { E L } } ( p , \theta , \omega , b , \vartheta ) = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { Y | X } ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \omega _ { u } ) + \operatorname* { m a x } _ { b \in \mathbb { R } ^ { n } } \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log s _ { u } ( z _ { u r } ) + C _ { \mathrm { e m p } } ,\tag{13}
$$

where $\begin{array} { r } { C _ { \mathrm { e m p } } : = \sum _ { t = 1 } ^ { n _ { z } } k _ { t } \log ( k _ { t } / N ) - \sum _ { i = 1 } ^ { m } n _ { i } } \end{array}$ log $\rho _ { i }$ . Thus, after profiling both the unrestricted baseline masses and the density-ratio intercepts, the two optimized objectives differ only by the fixed conditional-label term and the data-dependent constant $C _ { \mathrm { e m p } }$ . We follow the profiling argument of Wang et al. (2026a): first profile the baseline masses, then identify their Lagrange multipliers through the intercept score equations, and finally verify existence and global optimality.

Profiling the baseline masses. Fix an intercept vector b for which the empirical-likelihood constraints admit positive masses. The conditional-label term and $\sum _ { u , r } \log \varphi _ { u } ( z _ { u r } )$ do not depend on $p ,$ so profiling p only requires maximizing $\textstyle \sum _ { t } k _ { t }$ log $p _ { t }$ . Using the equivalent constraints $\textstyle \sum _ { t } p _ { t } = 1$ and $\begin{array} { r } { \sum _ { t } p _ { t } \{ \varphi _ { i } ( \xi _ { t } ) - 1 \} = 0 } \end{array}$ for $i = 2 , \ldots , m$ , introduce the multiplier $\mu$ for the baseline normalization and $\nu _ { i }$ for the normalization of client i. The Lagrangian for the $p \cdot$ -dependent term is

$$
\sum _ { t = 1 } ^ { n _ { z } } k _ { t } \log p _ { t } - N \mu \left( \sum _ { t = 1 } ^ { n _ { z } } p _ { t } - 1 \right) - N \sum _ { i = 2 } ^ { m } \nu _ { i } \sum _ { t = 1 } ^ { n _ { z } } p _ { t } \{ \varphi _ { i } ( \xi _ { t } ) - 1 \} .
$$

Differentiating with respect to $p _ { t }$ gives

$$
0 = \frac { k _ { t } } { p _ { t } } - N \mu - N \sum _ { i = 2 } ^ { m } \nu _ { i } \{ \varphi _ { i } ( \xi _ { t } ) - 1 \} .
$$

Multiplying this equation by $p _ { t }$ , summing over $t ,$ and applying the constraints gives $0 = N - N \mu , \mathrm { s o } \mu = 1$ . Hence the profiled masses have the form

$$
p _ { t } = { \frac { k _ { t } } { N \left[ 1 + \sum _ { i = 2 } ^ { m } \nu _ { i } \{ \varphi _ { i } ( \xi _ { t } ) - 1 \} \right] } } , \qquad t = 1 , \dots , n _ { z } ,\tag{14}
$$

Because $\textstyle \sum _ { t } k _ { t } \log p _ { t }$ is strictly concave in the positive masses and the constraints are affine in $p ,$ any feasible solution of these equations is the unique profiled mass vector. For fixed $b ,$ the multipliers $\nu _ { i }$ must satisfy the normalization constraints and generally depend on b.

Identifying the multipliers at the joint optimum. Suppose a joint interior optimum exists, denote it by $( p ^ { \star } , b ^ { \star } )$ and evaluate $\varphi _ { i }$ at $b ^ { \star }$ throughout this paragraph. Differentiation of the full Lagrangian with respect to $b _ { i } ~ \mathrm { g i v e s }$ , for $i = 2 , \ldots , m$

$$
0 = n _ { i } - N \nu _ { i } \sum _ { t = 1 } ^ { n _ { z } } p _ { t } ^ { \star } \varphi _ { i } ( \xi _ { t } ) = n _ { i } - N \nu _ { i } .
$$

The second equality uses the normalization $\begin{array} { r } { \sum _ { t } p _ { t } ^ { \star } \varphi _ { i } ( \xi _ { t } ) = 1 } \end{array}$ . Therefore, the multiplier at the joint optimum is $\nu _ { i } = n _ { i } / N = \rho _ { i }$ . Substituting these values into (14) and using $\varphi _ { 1 } \equiv 1$ gives

$$
p _ { t } ^ { \star } = \frac { k _ { t } } { N \left[ 1 + \sum _ { i = 2 } ^ { m } \rho _ { i } \{ \varphi _ { i } ( \xi _ { t } ) - 1 \} \right] } = \frac { k _ { t } } { N \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } ( \xi _ { t } ) } .\tag{15}
$$

Substitution into the empirical log-likelihood then yields

$$
\begin{array} { l } { { \displaystyle \ell _ { \mathrm { E L } } ( p ^ { \star } , \theta , \omega , b ^ { \star } , \vartheta ) = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { \gamma | X } \big ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \omega _ { u } \big ) + \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log \frac { \varphi _ { u } \big ( z _ { u r } \big ) } { \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } \big ( z _ { u r } \big ) } + \sum _ { t = 1 } ^ { n _ { z } } k _ { t } \log \big ( k _ { t } / N \big ) } } \\ { { \displaystyle = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { \gamma | X } \big ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \omega _ { u } \big ) + \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log s _ { u } \big ( z _ { u r } \big ) + \sum _ { t = 1 } ^ { n _ { z } } k _ { t } \log \big ( k _ { t } / N \big ) - \sum _ { i = 1 } ^ { m } n _ { i } \log \rho _ { i } } } \\ { { \displaystyle = \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { \gamma | X } \big ( \{ y _ { u r } \} \mid x _ { u r } ; \theta , \omega _ { u } \big ) + \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log s _ { u } \big ( z _ { u r } \big ) + C _ { \mathrm { e m p } } . } } \end{array}
$$

This calculation identifies the objective at a joint optimum, but it does not by itself establish that such an optimum exists or is globally optimal.

Existence and uniqueness of the intercept maximizer. We now establish that the conditional source log-likelihood has a unique finite maximizer and that the masses in (15) are feasible at this intercept vector. Write $h _ { i t } = \log \rho _ { i } + \vartheta _ { i } ^ { \top } \xi _ { t }$ and $M = \operatorname* { m a x } _ { i , t } | h _ { i t } | < \infty$ . For $b _ { 1 } = 0$ , let $B =$ max<sub>i</sub> $b _ { i } , A = \operatorname* { m i n } _ { i } b _ { i }$ , and $n _ { \mathrm { m i n } } = \mathrm { m i n } _ { i } n _ { i } > 0$ . To establish existence, we bound each numerator score above by $b _ { u } + M$ and each log normalizer below by $B - M$

$$
\sum _ { n = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log s _ { n } ( z _ { u r } ) \leq 2 N M + \sum _ { i } n _ { i } b _ { i } - N B \leq 2 N M - n _ { \operatorname* { m i n } } ( B - A ) \leq 2 N M - n _ { \operatorname* { m i n } } \| ( b _ { 2 } , \ldots , b _ { m } ) \| _ { \infty } .
$$

The bound shows that the conditional source log-likelihood tends $\mathrm { t o } - \infty$ as the free intercepts diverge, so continuity guarantees a finite maximizer. To establish uniqueness, take any nonzero perturbation v of the free intercepts and extend it by setting $v _ { 1 } = 0$ . The corresponding Hessian quadratic form is

$$
v ^ { \top } \nabla _ { b } ^ { 2 } \left\{ \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log s _ { u } ( z _ { u r } ) \right\} v = - \sum _ { t = 1 } ^ { n _ { z } } k _ { t } \operatorname { V a r } _ { C \sim s ( \xi _ { t } ) } ( v _ { C } ) < 0 .
$$

Because $s _ { i } ( \xi _ { t } ) > 0$ for every client and v is nonconstant, $\mathrm { V a r } _ { C \sim s ( \xi _ { t } ) } ( v _ { C } ) > 0$ for every t. The Hessian is therefore negative definite, which proves strict concavity and uniqueness of the intercept maximizer. Let $b ^ { \star } ( \vartheta )$ denote this unique maximizer, consistently with the notation used above, and evaluate $\varphi _ { i } \mathrm { a t } b ^ { \star }$ . Its score equations verify that the masses in (15) satisfy every empirical-likelihood constraint. For each $i \geq 2 ,$ , the score equation is

$$
0 = \partial _ { b _ { i } } \left\{ \sum _ { a = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log s _ { u } ( z _ { u r } ) \right\} = n _ { i } - \rho _ { i } \sum _ { t = 1 } ^ { n _ { z } } k _ { t } \frac { \varphi _ { i } ( \xi _ { t } ) } { \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } ( \xi _ { t } ) } .
$$

Because $\rho _ { i } = n _ { i } / N$ , these equations imply $\begin{array} { r } { \sum _ { t } p _ { t } ^ { \star } \varphi _ { i } ( \xi _ { t } ) = 1 } \end{array}$ for every $i \geq 2$ . Their baseline normalization follows from $\varphi _ { 1 } \equiv 1$ and $\rho _ { 1 } > 0 \mathrm { : }$

$$
1 = \sum _ { t } p _ { t } ^ { \star } \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } ( \xi _ { t } ) = \rho _ { 1 } \sum _ { t } p _ { t } ^ { \star } + \sum _ { i = 2 } ^ { m } \rho _ { i } , \qquad \mathrm { s o } \qquad \sum _ { t } p _ { t } ^ { \star } = 1 .
$$

Thus $( p ^ { \star } , b ^ { \star } )$ is a feasible pair.

Global-optimality verification. It remains to verify that this feasible pair is globally optimal rather than merely stationary. For any $( p , b ) \in \mathcal { A } _ { \vartheta }$ , the normalization constraints give $\begin{array} { r } { \sum _ { t } \dot { p _ { t } } \sum _ { c = 1 } ^ { m } \dot { \rho _ { c } } \varphi _ { c } ( \dot { \xi } _ { t } ) = 1 } \end{array}$ . Jensen’s inequality therefore gives

$$
\begin{array} { r l } & { \ell _ { \mathrm { E L } } ( p , \theta , \boldsymbol { \omega } , \boldsymbol { b } , \boldsymbol { \vartheta } ) - \displaystyle \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log P _ { u } ^ { Y | \boldsymbol { X } } ( \{ \boldsymbol { y } _ { u r } \} \mid \boldsymbol { x } _ { u r } ; \boldsymbol { \theta } , \boldsymbol { \omega } _ { u } ) - \displaystyle \sum _ { u = 1 } ^ { m } \sum _ { r = 1 } ^ { n _ { u } } \log s _ { u } ( z _ { u r } ) - C _ { \mathrm { e m p } } } \\ & { \quad = \displaystyle \sum _ { t = 1 } ^ { n _ { z } } k _ { t } \log \frac { N p _ { t } \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } ( \xi _ { t } ) } { k _ { t } } } \\ & { \quad \le N \log \left\{ \displaystyle \sum _ { t = 1 } ^ { n _ { z } } \frac { k _ { t } } { N } \frac { N p _ { t } \sum _ { c = 1 } ^ { m } \rho _ { c } \varphi _ { c } ( \xi _ { t } ) } { k _ { t } } \right\} = 0 . } \end{array}
$$

Equality holds precisely for the masses in (15). The conditional source log-likelihood is uniquely maximized at $b ^ { \star }$ , so this upper bound is attained by $( p ^ { \star } , b ^ { \star } )$ and proves (13).

Scope of the equivalence. Equation (13) is an optimization identity obtained by profiling both the baseline masses and the intercepts at fixed representations and slopes. At fixed representations, optimizing the slopes as well preserves equality of the suprema, and finite global maximizers correspond through the displayed masses whenever such maximizers exist. However, existence of a finite slope maximizer is a separate condition and can fail under source separation. Moreover, the identity does not assert that profiling only the baseline masses at a fixed $( b , \vartheta )$ yields the conditional-label and conditional-source terms in (13) plus the same constant $C _ { \mathrm { e m p } } .$ For example, take two clients with one observation each, $z _ { 1 1 } = 0 , z _ { 2 1 } = 1$ , and $( \varphi _ { 2 } ( 0 ) , \bar { \varphi } _ { 2 } ( 1 ) ) = ( 1 / 2 , 3 / 2 )$ , while $\varphi _ { 1 } \equiv 1$ . The EL constraints force both baseline masses to be $1 / 2$ , giving marginal likelihood $3 / 8$ , whereas the source likelihood is $( 2 / 3 ) ( 3 / 5 ) = 2 / 5$ and $C _ { \mathrm { e m p } } = 0 .$ . The model in this example is already reference normalized, so fixing the reference does not remove the need to optimize the intercepts. If all pooled representations are distinct, $\begin{array} { r } { C _ { \mathrm { e m p } } ^ { \mathrm { ~ ~ ~ } } = - N \log N - \sum _ { i } n _ { i } \log \rho _ { i } } \end{array}$ With repeated representations, $\bar { C _ { \mathrm { e m p } } }$ is constant for a fixed representation but can change if learning θ changes their multiplicities. Consequently, we use (3) as a conditional-label plus source-classification objective, while interpreting the EL equivalence only at fixed representations rather than as an unrestricted identity throughout encoder training.

## B.3 From the global source loss to graph-local comparisons

The global source posterior $s _ { i } ( z )$ cannot be evaluated from graph-local information because its denominator requires scores from all clients. Our objective is therefore to construct an edgewise source model that uses only the two endpoint scores while preserving their pairwise density ratio. Fix an edge $e = \left\{ i , j \right\}$ and consider a balanced endpoint experiment that first selects $i \ \mathrm { o r } \ j$ with probability $1 / 2$ and then draws a representation from the selected endpoint. Let $q _ { i j } ^ { \mathrm { b a l } } ( z )$ denote the conditional probability that representation z originated from endpoint $j$ in this experiment. Under the density-ratio model, Bayes’ rule gives

$$
q _ { i j } ^ { \mathrm { b a l } } ( z ) : = \varphi _ { j } ( z ) / \{ \varphi _ { i } ( z ) + \varphi _ { j } ( z ) \} ,
$$

whose logit is the pairwise log-density ratio

$$
\mathrm { l o g i t } q _ { i j } ^ { \mathrm { b a l } } ( \boldsymbol { z } ) = \log \{ \varphi _ { j } ( \boldsymbol { z } ) / \varphi _ { i } ( \boldsymbol { z } ) \} = ( b _ { j } - b _ { i } ) + ( \vartheta _ { j } - \vartheta _ { i } ) ^ { \top } \boldsymbol { z } .
$$

For the orientation $i < j$ and endpoint $u \in \{ i , j \}$ , define the endpoint-normalized source loss

$$
\mathcal { L } _ { e } ^ { ( u ) } = - \frac { 1 } { n _ { u } } \sum _ { r = 1 } ^ { n _ { u } } \left[ \mathbf { 1 } \{ u = i \} \log \{ 1 - q _ { i j } ^ { \mathrm { b a l } } ( z _ { u r } ) \} + \mathbf { 1 } \{ u = j \} \log q _ { i j } ^ { \mathrm { b a l } } ( z _ { u r } ) \right] .\tag{16}
$$

Because the endpoint prior is balanced, the negative empirical conditional log-likelihood averages the two endpointspecific losses equally:

$$
\begin{array} { c } { { \frac { 1 } { 2 } \mathcal { L } _ { e } ^ { ( i ) } + \frac { 1 } { 2 } \mathcal { L } _ { e } ^ { ( j ) } = - \displaystyle \frac { 1 } { 2 n _ { i } } \sum _ { r = 1 } ^ { n _ { i } } \log \{ 1 - q _ { i j } ^ { \mathrm { b a l } } ( z _ { i r } ) \} } } \\ { { - \displaystyle \frac { 1 } { 2 n _ { j } } \sum _ { r = 1 } ^ { n _ { j } } \log q _ { i j } ^ { \mathrm { b a l } } ( z _ { j r } ) . } } \end{array}
$$

This equal weighting removes the sample-proportion offset, so the population logit targets only the pairwise density ratio. Moreover, each edgewise likelihood requires no graph-wide normalization because it depends only on the endpoint scores. Summing the edgewise terms therefore produces an implementable conditional composite likelihood rather than the global source likelihood in (13).

## B.4 Balanced graph-local objective used in training

Our objective is now to combine label prediction with the edgewise source likelihood in a client-local form whose average assigns the intended weight to every task and edge term. The main text presents these components separately to distinguish prediction from source classification, whereas the implementation combines them using the weighting derived below. Client u maintains a local representation $g _ { \boldsymbol { \theta } _ { u } }$ and evaluates $z _ { u r } : = g _ { \theta _ { u } } ( x _ { u r } )$ on its own examples. The local parameters need not coincide after a finite number of neighbor-mixing rounds; setting all $\theta _ { u } = \theta$ recovers the common-representation model used in the derivation above

We now combine the endpoint source loss in (16) with the conditional-label probabilities in (1).

We next choose the scaling that makes the client-local losses aggregate to the desired graph-level objective. Let $\lambda \in ( 0 , 1 ]$ denote the weight assigned to task prediction, leaving weight $1 - \lambda$ for source classification. To weight the m task losses uniformly and the |E| edge losses uniformly after averaging over clients, set

$$
\kappa : = \frac { m ( 1 - \lambda ) } { 2 | E | } .\tag{17}
$$

Let $a _ { i } = ( \vartheta _ { i } , b _ { i } )$ denote the routing potential owned by client i. The resulting client-local contribution is

$$
\mathcal { L } _ { i } = - \frac { \lambda } { n _ { i } } \sum _ { r = 1 } ^ { n _ { i } } \log P _ { i } ^ { Y | X } ( \left\{ y _ { i r } \right\} | x _ { i r } ; \theta _ { i } , \omega _ { i } ) + \kappa \sum _ { j \in \mathcal { N } ( i ) } \mathcal { L } _ { \{ i , j \} } ^ { ( i ) } ,\tag{18}
$$

and averaging these contributions over clients yields

$$
\mathcal { L } _ { G } = \frac { 1 } { m } \sum _ { i } \mathcal { L } _ { i } = - \frac { \lambda } { m } \sum _ { i } \frac { 1 } { n _ { i } } \sum _ { r = 1 } ^ { n _ { i } } \log P _ { i } ^ { Y | X } ( \{ y _ { i r } \} \mid x _ { i r } ; \theta _ { i } , \omega _ { i } ) + ( 1 - \lambda ) / ( 2 | E | ) \sum _ { e = \{ i , j \} \in E } \{ \mathcal { L } _ { e } ^ { ( i ) } + \mathcal { L } _ { e } ^ { ( j ) } \} .\tag{19}
$$

Thus, each endpoint loss is a mean over its local data, the two endpoints of every edge receive equal weight, and all edges receive equal weight without an additional degree normalization. The reported experiments use $\lambda = 0 . 9$ , while $\lambda = 1$ gives the task-only ablation. For fixed features, the prediction-head and routing-potential subproblems separate.

To optimize (19) using only neighbor communication, each endpoint differentiates every incident edge loss with respect to both endpoint potentials and returns the neighbor-owned component. The resulting unregularized routing-potential gradient received by owner i is

$$
G _ { i } = \nabla _ { a _ { i } } \mathcal { L } _ { i } + \kappa \sum _ { j \in \mathcal { N } ( i ) } \nabla _ { a _ { i } } \mathcal { L } _ { \{ i , j \} } ^ { ( j ) } .
$$

This expression equals $\nabla _ { a _ { i } } \sum _ { u } \mathcal { L } _ { u }$ in unrestricted potential coordinates when all endpoint contributions are evaluated at the same local-step snapshot. For the reference-constrained theoretical update, $a _ { 1 }$ remains fixed at zero and only the components $G _ { i } \ : \mathrm { f o r } \ : i \geq 2$ are used. The private head receives $\nabla _ { \omega _ { i } } \mathcal { L } _ { i }$ , while the frozen encoder receives no gradient.

The balanced endpoint likelihood estimates the pairwise density ratio without incorporating the sample-count prior. Because the global source posterior includes that prior, inference recovers the appropriate source odds by adding it to the learned comparison:

$$
\begin{array} { r } { \log \{ s _ { j } ( z ) / s _ { i } ( z ) \} = \log ( \rho _ { j } / \rho _ { i } ) + \log \{ \varphi _ { j } ( z ) / \varphi _ { i } ( z ) \} = \log ( n _ { j } / n _ { i } ) + ( b _ { j } - b _ { i } ) + ( \vartheta _ { j } - \vartheta _ { i } ) ^ { \top } z . } \end{array}
$$

Therefore, training excludes the sample-count prior and inference adds log $( n _ { j } / n _ { i } )$ exactly once.

Finally, when the frozen local representations differ, the fitted reverse-edge scores need not be antisymmetric. Section 4 separates this representation-disagreement error from coefficient-estimation error.

## C Proofs and Additional Theory

Appendix C.1 proves the coefficient-estimation guarantee, Appendix C.2 derives the comparison identities, and Appendix C.3 proves the referral guarantee. For the routing arguments, fix the positive training counts and query $( x , s )$ We use the route $\mathcal { R } _ { L } ( x , s ) = ( \widehat { s } _ { 0 } = s , \ldots , \widehat { s } _ { T } )$ from (7), with terminal client $\widehat { s } _ { T }$ , and the scores $u _ { t }$ and weights w<sub>t</sub> from Section 3.2. Empty sums are zero and max $\mathcal { D } = - \infty$

## C.1 Proof of Theorem 1

Statistical conditions. Fix a connected graph with $m \geq 2 ,$ the number of classes, the representation and coefficient dimensions, and the task weight $\lambda \in ( 0 , 1 )$ . The common representation is fixed independently of the fitting samples. Within each client the observations are i.i.d. from a fixed distribution $P _ { i } ^ { X , Y }$ , and different clients are independent. The deterministic allocations satisfy $\rho _ { i , N } : = n _ { i } / N \to \rho _ { i } ^ { * } > 0 ;$ for a fixed $N$ , these are exactly the proportions $\rho _ { i }$ used by the inference-time prior in Section 3. The conditional-label model and the common-reference DRM are correctly specified by the same finite coefficient vector $\zeta ^ { \star }$ for all $N _ { \cdot }$ . For identifiability, we use $F = P _ { 1 } ^ { Z }$ and fix $b _ { 1 } = 0$ and $\vartheta _ { 1 } = 0$ , so that $\varphi _ { 1 } \equiv 1$ . In identifiable coordinates the limiting population objective defined below has a unique interior minimizer $\zeta ^ { \star }$ . On a compact convex parameter region containing this point in its interior, the observation losses and their first two derivatives have integrable envelopes; the score at $\zeta ^ { \star }$ has finite second moment, and the population Hessian is positive definite. Losses are measurable and twice continuously differentiable on a neighborhood of the region, and empirical minimizers are measurable. These compact-region regularity conditions apply directly to the balanced training losses below. The optimization conditions below specify existence of a finite empirical minimizer and containment of the iterates.

Observation contributions. We decompose the graph objective $\mathcal { L } _ { G }$ in $( 1 9 )$ into observation-level contributions. For a generic observation $O = ( X , Y )$ , let $z = g _ { \boldsymbol { \theta } } ( X )$ . For candidate coefficients $\zeta ,$ , write

$$
q _ { i j } ^ { \mathrm { b a l } } ( z ; \zeta ) = \left[ 1 + \exp \{ - ( b _ { j } - b _ { i } ) - ( \vartheta _ { j } - \vartheta _ { i } ) ^ { \top } z \} \right] ^ { - 1 } .
$$

At $\zeta = \zeta ^ { \star }$ , the parameterized quantity $q _ { i j } ^ { \mathrm { b a l } } ( z ; \zeta )$ reduces to the main-text posterior $q _ { i j } ^ { \mathrm { b a l } } ( z )$ . The density-tilt coefficients retain their definition in (2); no sample-count offset appears in this training logit. For an edge $e = \{ v , w \}$ , orient its endpoints so that $v < w .$ . The observation-level contribution needed for the asymptotic analysis is

$$
\begin{array} { l } { { \ell _ { G , i } ( O ; \zeta ) = - \displaystyle \frac { \lambda } { m } \log P _ { i } ^ { Y | X } ( \{ Y \} \mid X ; \theta , \omega _ { i } ) } } \\ { { \quad \quad \quad - \displaystyle \frac { 1 - \lambda } { 2 | E | } \sum _ { e = \{ v , w \} \ni i } \left[ \mathbf { 1 } \{ i = v \} \log \{ 1 - q _ { v w } ^ { \mathrm { b a l } } ( g \theta ( X ) ; \zeta ) \} + \mathbf { 1 } \{ i = w \} \log q _ { v w } ^ { \mathrm { b a l } } ( g \theta ( X ) ; \zeta ) \right] , } } \\ { { \ } } \\ { { \mathcal { L } _ { G } ( \zeta ) = \displaystyle \sum _ { i } \frac { 1 } { n _ { i } } \sum _ { r = 1 } ^ { n _ { i } } \ell _ { G , i } ( O _ { i r } ; \zeta ) , \qquad R _ { G } ( \zeta ) = \displaystyle \sum _ { i } \mathbb { E } _ { i } \ell _ { G , i } ( O _ { i 1 } ; \zeta ) , } } \end{array}
$$

Here $O _ { i r } = ( X _ { i r } , Y _ { i r } )$ , and $\mathbb { E } _ { i }$ denotes expectation under $P _ { i } ^ { X , Y }$ . All terms evaluated on the same observation remain inside the same $\ell _ { G , i }$ . Thus $\mathbb { E } \mathcal { L } _ { G } = R _ { G }$ for every allocation: the sample proportions affect the estimator’s variance and the inference-time prior, but not the balanced population training objective.

Identifiable coordinates and the gradient update. The routing coordinates are the free blocks $a _ { i } = \left( \vartheta _ { i } , b _ { i } \right)$ for $i = 2 , \ldots , m$ , with $a _ { 1 } = 0$ held fixed. For each private head, retain $\begin{array} { r } { \sum _ { k } \alpha _ { i k } = 0 } \end{array}$ and $\textstyle \sum _ { k } \beta _ { i k } = 0$ and use orthonormal coordinates on this class-contrast subspace. All parameter derivatives and norms in Theorem 1 are taken in these coordinates. All unqualified vector norms below denote the Euclidean norm. Choosing client 1 as the density reference preserves every pairwise ratio: from any common baseline $F _ { 0 }$ with coefficients $( \breve { b _ { i } ^ { 0 } } , \vartheta _ { i } ^ { 0 } )$ , take $F = P _ { 1 } ^ { Z }$ and $( b _ { i } , \vartheta _ { i } ) = ( b _ { i } ^ { 0 } - b _ { 1 } ^ { 0 } , \vartheta _ { i } ^ { 0 } - \vartheta _ { 1 } ^ { 0 } )$ . Then $d P _ { i } ^ { Z } / d F = \exp \{ b _ { i } + \vartheta _ { i } ^ { \top } z \}$ remains a normalized density. The private-head gradient sums to zero across classes, so common-step unpenalized updates preserve its class-contrast subspace. The analyzed routing update acts only on the free blocks $a _ { 2 } , \ldots , a _ { m }$ and keeps $a _ { 1 } = 0$

We analyze synchronous, unpenalized plain SGD in these free coordinates for Stage II. Write $\mathcal { L } _ { i }$ and $\mathcal { L } _ { e } ^ { ( u ) }$ for the metho $\mathrm { 1 ^ { \circ } s }$ client-local and endpoint-normalized losses. For each free routing-potential owner $i \neq 1$ , the gradient is exactly

$$
G _ { i } ^ { a } = \nabla _ { a _ { i } } \mathcal { L } _ { i } + \kappa \sum _ { j \in \mathcal { N } ( i ) } \nabla _ { a _ { i } } \mathcal { L } _ { \{ i , j \} } ^ { ( j ) } , \qquad \kappa = \frac { m ( 1 - \lambda ) } { 2 | E | } .
$$

The head owner receives $G _ { i } ^ { \omega } = \nabla _ { \omega _ { i } } \mathcal { L } _ { i }$ , expressed in the head coordinates above. Client 1 still contributes its task gradient and the source gradients of its observations to neighboring free routing blocks. With the same snapshot for every endpoint contribution, concatenating the free routing blocks and all head-coordinate blocks gives $\begin{array} { r } { \nabla \sum _ { i } \dot { \mathcal { L } } _ { i } = m \nabla \mathcal { L } _ { G } } \end{array}$ including the gradients returned by neighbors. Divide the assembled gradient by $m ;$ equivalently, a step of size $\eta _ { r }$ on the normalized gradient uses step size $\eta _ { r } / m$ on the raw owner gradients. The analyzed update is

$$
\zeta ^ { ( r + 1 ) } = \zeta ^ { ( r ) } - \eta _ { r } \widehat { G } _ { r } , \qquad \mathbb { E } [ \widehat { G } _ { r } \mid \mathcal { F } _ { r } ] = \nabla \mathcal { L } _ { G } ( \zeta ^ { ( r ) } ) ,\tag{20}
$$

where the data and initialization are included in the optimization history $\mathcal { F } _ { r }$ . At each update, independently across clients, sample $B _ { i } ^ { \mathrm { m b } }$ local indices uniformly with replacement and average $\nabla \ell _  G , $ over them, then sum these averages over i. This gives exactly the conditional mean in (20). If $\widehat { P } _ { i }$ is the empirical distribution at client $i ,$ the conditional covariance of this normalized mini-batch gradient is

$$
\sum _ { i } \frac { 1 } { B _ { i } ^ { \mathrm { m b } } } \operatorname { V a r } _ { \widehat { P } _ { i } } \{ \nabla \ell _ { G , i } ( O ; \zeta ^ { ( r ) } ) \} .
$$

Its trace is the conditional noise second moment. This covariance includes the correlations between incident-edge losses evaluated on the same observation. A variance bound for raw owner gradients must be divided by $m ^ { 2 }$ to give the corresponding normalized bound. Other samplers are covered only when the stated conditional unbiasedness and joint noise bound hold.

Optimization conditions. The optimization analysis uses the following finite-minimizer, curvature, and bounded-iterate conditions. On a data-and-initialization event whose probability tends to one, let $\widehat { \zeta } _ { N }$ be a finite empirical minimizer with zero coordinate gradient, and let a convex region $\kappa _ { N }$ contain this minimizer and the initial iterate, with

$$
\mu I \preceq \nabla ^ { 2 } { \mathcal L } _ { G } ( \zeta ) \preceq \mu ^ { - 1 } I , \qquad 0 < \mu \leq 1 , \qquad \zeta \in { \mathcal K } _ { N } .
$$

Any uniform positive lower and finite upper curvature bounds admit this form after choosing $\mu$ sufficiently small. Assume the conditional noise second moment is at most $\sigma ^ { 2 }$ while the iterate is in this region. The constants are uniform in $N ,$ and $\| \zeta ^ { ( 0 ) } - \widehat { \zeta } _ { N } \| = O _ { p } ( 1 )$ . As in the bounded-iterate premise, all iterates up to the analyzed budget must remain in the region, either almost surely conditional on these data or with probability tending to one. For high-probability containment, the proof uses the first exit time from $\kappa _ { N }$ . On a bounded-logit region, positive-definite augmented-feature Gram matrices for every task head and balanced edge mixture, together with graph connectivity, are sufficient for positive curvature in the free reference-client and private-head coordinates. Existence of a finite minimizer and containment of the iterates are separate assumptions; complete separation and unidentified design directions are excluded. The curvature condition is local to $\kappa _ { N }$

Proof of Theorem 1. Step 1: A common population minimizer. Under the correctly specified DRM, the main-text endpoint-source posterior is

$$
q _ { i j } ^ { \mathrm { b a l } } ( z ) = \frac { \varphi _ { j } ( z ) } { \varphi _ { i } ( z ) + \varphi _ { j } ( z ) } .
$$

Let $\chi _ { e } ( z )$ be the gradient in identifiable coordinates of $( b _ { j } - b _ { i } ) + ( \vartheta _ { j } - \vartheta _ { i } ) ^ { \top } z$ . It does not depend on the parameter value, since these logits are affine. At parameter $\zeta ,$ , the source-label gradient for an observation with representation z from endpoint u is $\{ \breve { q } _ { i j } ^ { \mathrm { b a l } } ( z ; \zeta ) - \mathbf { 1 } \{ u \stackrel { \cdot } { = } j \} \} \chi _ { e } ( z )$ . Consequently, at $\zeta ^ { \star }$

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \mathbb { E } _ { i } [ q _ { i j } ^ { \mathrm { b a l } } ( Z ) \chi _ { e } ( Z ) ] + \frac { 1 } { 2 } \mathbb { E } _ { j } [ ( q _ { i j } ^ { \mathrm { b a l } } ( Z ) - 1 ) \chi _ { e } ( Z ) ] } \\ & { \quad = \int \bigl \{ \frac { 1 } { 2 } \{ \varphi _ { i } ( z ) + \varphi _ { j } ( z ) \} q _ { i j } ^ { \mathrm { b a l } } ( z ) - \frac { 1 } { 2 } \varphi _ { j } ( z ) \bigr \} \chi _ { e } ( z ) d F ( z ) = 0 . } \end{array}
$$

The two equally weighted endpoint means therefore cancel exactly. For the task term, write $p _ { i } ( z ; \zeta )$ for the model class-probability vector. Let $A _ { i } ( z )$ be the Jacobian of the $K$ task logits with respect to $\zeta ,$ , and let $\mathbf { e } _ { Y }$ denote the class indicator vector. Correct specification gives

$$
\mathbb { E } _ { i } \left[ \nabla _ { \zeta } \{ - \log P _ { i } ^ { Y | X } ( \{ Y \} \mid X ; \theta , \omega _ { i } ) \} \mid Z \right] = A _ { i } ( Z ) ^ { \top } \{ p _ { i } ( Z ; \zeta ^ { \star } ) - \mathbb { E } _ { i } [ \mathbf { e } _ { Y } \mid Z ] \} = 0 .
$$

Thus, with $S _ { i } ( O ) = \nabla \ell _ { G , i } ( O ; \zeta ^ { \star } )$ and $\overline { { S } } _ { i } = \mathbb { E } _ { i } S _ { i } ( O )$ , the population-score identity is

$$
\nabla R _ { G } ( \zeta ^ { \star } ) = \sum _ { i } \overline { { \mathcal { S } } } _ { i } = 0 , \qquad \mathrm { w h i l e ~ g e n e r a l l y ~ } \overline { { \mathcal { S } } } _ { i } \neq 0 .
$$

Convexity already makes $\zeta ^ { \star }$ a minimizer. More explicitly, $R _ { G } ( \zeta ) - R _ { G } ( \zeta ^ { \star } )$ is the sum of

$$
\frac { \lambda } { m } \sum _ { i } \mathbb { E } _ { i } \operatorname { K L } \{ p _ { i } ( Z ; \zeta ^ { \star } ) \| p _ { i } ( Z ; \zeta ) \}
$$

and

$$
\frac { 1 - \lambda } { | E | } \sum _ { e = \{ i , j \} } \int \frac { 1 } { 2 } \{ \varphi _ { i } ( z ) + \varphi _ { j } ( z ) \} \operatorname { K L } \big \{ \operatorname { B e r } ( q _ { i j } ^ { \mathrm { b a l } } ( z ) ) \| \operatorname { B e r } ( q _ { i j } ^ { \mathrm { b a l } } ( z ; \zeta ) ) \big \} \ d F ( z ) .
$$

All weights are positive. Equality requires the correct task probabilities and density-score differences on every edge. The stated identification condition therefore makes $\zeta ^ { \star }$ the unique minimizer of the balanced population objective $R _ { G }$

Step 2: Score covariance. The endpoint score second moments follow from the information identity. At the true balanced posterior,

$$
\begin{array} { r l r } {  { \frac { 1 } { 2 } \sum _ { u \in e } \mathbb { E } _ { u } [ ( q _ { i j } ^ { \mathrm { b a l } } ( Z ) - \mathbf { 1 } \{ u = j \} ) ^ { 2 } \| \chi _ { e } ( Z ) \| ^ { 2 } ] } } \\ & { } & { = \int \frac { 1 } { 2 } \{ \varphi _ { i } ( z ) + \varphi _ { j } ( z ) \} q _ { i j } ^ { \mathrm { b a l } } ( z ) \{ 1 - q _ { i j } ^ { \mathrm { b a l } } ( z ) \} \| \chi _ { e } ( z ) \| ^ { 2 } d F ( z ) < \infty . } \end{array}
$$

The right-hand side is the trace of the edge’s population Hessian before its positive graph-level coefficient. All Hessian contributions are positive semidefinite, so finiteness follows from the stated Hessian envelope. Each endpoint term is finite. The task-score second moment follows from the stated moment condition (and from its correctly specified softmax information identity). Uniform integrable gradient envelopes also follow by expanding each gradient from $\zeta ^ { \star }$ over the compact region using its Hessian envelope.

Here are the exact score and Hessian for the balanced objective. All probabilities in the following displays are evaluated at $\zeta ^ { \star }$ and the displayed representation:

$$
\begin{array} { l } { { \displaystyle \mathcal { S } _ { i } ( O ) = \frac { \lambda } { m } A _ { i } ( Z ) ^ { \top } \{ p _ { i } ( Z ; \zeta ^ { \star } ) - { \bf e } _ { Y } \} } } \\ { { \displaystyle ~ + \frac { 1 - \lambda } { 2 | E | } \sum _ { { \bf e } = \{ v , w \} \ni i , v < w } ( q _ { v w } ^ { \mathrm { b a l } } ( Z ) - { \bf 1 } \{ i = w \} ) \chi _ { e } ( Z ) } , }  \\ { { \displaystyle H _ { G } : = \nabla ^ { 2 } R _ { G } ( \zeta ^ { \star } ) = \frac { \lambda } { m } \sum _ { i } { \mathbb { E } _ { i } \big [ A _ { i } ( Z ) ^ { \top } \{ \mathrm { d i a g } ( p _ { i } ( Z ; \zeta ^ { \star } ) ) - p _ { i } ( Z ; \zeta ^ { \star } ) p _ { i } ( Z ; \zeta ^ { \star } ) ^ { \top } \} A _ { i } ( Z ) \big ] } } } \\ { { \displaystyle ~ + \frac { 1 - \lambda } { | E | } \sum _ { { \bf e } = \{ i , j \} } \int _ { \frac { 1 } { 2 } \{ \varphi _ { i } ( z ) + \varphi _ { j } ( z ) \} q _ { i j } ^ { \mathrm { b a l } } ( z ) \{ 1 - q _ { i j } ^ { \mathrm { b a l } } ( z ) \} \chi _ { e } ( z ) \chi _ { e } ( z ) ^ { \top } d F ( z ) } . } } \end{array}
$$

Set

$$
\begin{array} { r l } & { \qquad \Omega _ { i } = \mathbb { E } _ { i } [ ( S _ { i } ( O ) - \overline { { S } } _ { i } ) ( S _ { i } ( O ) - \overline { { S } } _ { i } ) ^ { \top } ] , } \\ & { J _ { G , N } = \mathrm { V a r } \{ \sqrt { N } \nabla \mathcal { L } _ { G } ( \zeta ^ { \star } ) \} = \displaystyle \sum _ { i } \frac { N } { n _ { i } } \Omega _ { i } , } \\ & { \qquad J _ { G } = \displaystyle \operatorname* { l i m } _ { N } J _ { G , N } = \sum _ { i } ( \rho _ { i } ^ { * } ) ^ { - 1 } \Omega _ { i } , } \\ & { \qquad \Sigma _ { G } = H _ { G } ^ { - 1 } J _ { G } H _ { G } ^ { - \top } . } \end{array}
$$

Here $\Omega _ { i }$ is centered within client stratum i, whose score mean $\overline { { S } } _ { i }$ may be nonzero. If $S _ { e } ^ { ( i ) } = \{ q _ { v w } ^ { \mathrm { b a l } } ( Z ) - \mathbf { 1 } \{ i =$ $v \} \}$ for $e = \{ v , w \} , v < w$ , then the routing part of $\Omega _ { i }$ contains the full double sum

$$
\left( \frac { 1 - \lambda } { 2 | E | } \right) ^ { 2 } \sum _ { e , f \ni i } \mathrm { C o v } _ { i } ( \mathcal { S } _ { e } ^ { ( i ) } , \mathcal { S } _ { f } ^ { ( i ) } ) .
$$

The terms with $e \neq f$ account for the reuse of each observation on incident edges. The full vector definition of $\Omega _ { i }$ also retains task contributions and their covariance with routing contributions. Under correct task specification, the task–routing cross-covariances vanish because the task score has conditional mean zero given $Z .$ In general $J _ { G }$ is not the same as $H _ { G }$

Step 3: Statistical consistency and asymptotic normality. The uniform-law conditions give, on the compact parameter region,

$$
\operatorname* { s u p } _ { \zeta } | \mathcal { L } _ { G } ( \zeta ) - R _ { G } ( \zeta ) | \to _ { p } 0 ,
$$

and the analogous uniform convergence for the Hessians. Uniqueness and interior positive curvature yield ${ \widehat { \zeta } } _ { N } \to _ { p } \zeta ^ { \star }$ On any sufficiently small sphere about $\zeta ^ { \star }$ , the population objective exceeds its value at $\zeta ^ { \star }$ uniformly, and so does the empirical objective with probability tending to one. Convexity then excludes a better empirical minimizer outside that sphere, by considering the segment to $\zeta ^ { \star }$

The exact cancellation of total score means gives

$$
\mathbb { G } _ { N } : = \sqrt { N } \nabla \mathcal { L } _ { G } ( \zeta ^ { \star } ) = \sum _ { i } \frac { \sqrt { N } } { n _ { i } } \sum _ { r = 1 } ^ { n _ { i } } \{ S _ { i } ( O _ { i r } ) - \overline { { S } } _ { i } \} .
$$

The centered summands have independent client strata. For each client the ordinary finite-variance multivariate i.i.d. central limit theorem applies to $\textstyle S _ { i } - { \overline { { S } } } _ { i }$ . Since $\sqrt { N / n _ { i } } \to ( \rho _ { i } ^ { * } ) ^ { - 1 / 2 }$ and the number of clients is fixed, $\mathbb { G } _ { N } \ { \overset { d } { \to } } \ N ( 0 , J _ { G } )$ Thus the allocation proportions enter the asymptotic covariance, although they do not enter the balanced population objective.

On the event that the empirical minimizer is interior, its exact first-order condition gives

$$
0 = \nabla { \mathcal { L } } _ { G } ( \zeta ^ { \star } ) + { \widehat { H } } _ { N } ( { \widehat { \zeta } } _ { N } - \zeta ^ { \star } ) , \qquad { \widehat { H } } _ { N } = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } { \mathcal { L } } _ { G } \{ \zeta ^ { \star } + t ( { \widehat { \zeta } } _ { N } - \zeta ^ { \star } ) \} d t \to _ { p } H _ { G } .
$$

Consequently,

$$
\begin{array} { r } { \sqrt { N } ( \widehat { \zeta } _ { N } - \zeta ^ { \star } ) = - H _ { G } ^ { - 1 } \sqrt { N } \nabla { \mathcal L } _ { G } ( \zeta ^ { \star } ) + o _ { p } ( 1 ) \overset { d } {  } { \mathcal N } ( 0 , \Sigma _ { G } ) . } \end{array}\tag{21}
$$

Step 4: Optimization error. Take $\eta _ { r } = 2 / \{ \mu ( r + r _ { 0 } ) \}$ with $r _ { 0 } \ge 2 / \mu ^ { 2 }$ , so that $r _ { 0 } \geq 2$ and $\eta _ { r } \leq \mu .$ . An integral Hessian expansion along the segment between ζ and $\zeta _ { N }$ gives, for $0 < \eta ^ { \mathrm { o p t } } \leq \mu$

$$
\begin{array} { r } { \| \zeta - \eta ^ { \mathrm { o p t } } \nabla \mathcal { L } _ { G } ( \zeta ) - \widehat { \zeta } _ { N } \| \leq ( 1 - \mu \eta ^ { \mathrm { o p t } } ) \| \zeta - \widehat { \zeta } _ { N } \| . } \end{array}
$$

Conditional unbiasedness makes the cross term with the gradient noise zero. Let $\tau _ { N }$ be the first iterate index outside $\kappa _ { N }$ and put

$$
v _ { r } = \mathbb { E } _ { \mathrm { o p t } } \Big [ \| \zeta ^ { ( r ) } - \widehat { \zeta } _ { N } \| ^ { 2 } { \bf 1 } \{ \tau _ { N } > r \} \Big ] .
$$

This is the usual mean-square error when containment holds almost surely. The event $\{ \tau _ { N } > r \}$ is measurable with respect to $\mathcal { F } _ { r }$ . Since $\{ \tau _ { N } > r + 1 \} \subseteq \{ \tau _ { N } > r \}$ ,

$$
v _ { r + 1 } \leq \mathbb { E } _ { \mathrm { o p t } } \Big [ { \bf 1 } \{ \tau _ { N } > r \} \mathbb { E } \Big [ \| \zeta ^ { ( r + 1 ) } - \widehat { \zeta } _ { N } \| ^ { 2 } | \mathcal { F } _ { r } \Big ] \Big ] .
$$

Applying conditional unbiasedness on the ${ \mathcal { F } } _ { r }$ -measurable event inside this expectation gives

$$
v _ { r + 1 } \leq \left( 1 - \frac { 2 } { r + r _ { 0 } } \right) ^ { 2 } v _ { r } + \frac { 4 \sigma ^ { 2 } } { \mu ^ { 2 } ( r + r _ { 0 } ) ^ { 2 } } \leq \left( 1 - \frac { 2 } { r + r _ { 0 } } \right) v _ { r } + \frac { 4 \sigma ^ { 2 } } { \mu ^ { 2 } ( r + r _ { 0 } ) ^ { 2 } } .
$$

The last inequality uses $r + r _ { 0 } \ge 2$ . Induction gives

$$
v _ { R } \leq { \frac { C _ { \mathrm { o p t } } } { R + r _ { 0 } } } , \qquad C _ { \mathrm { o p t } } = \operatorname* { m a x } \left\{ r _ { 0 } v _ { 0 } , { \frac { 4 \sigma ^ { 2 } } { \mu ^ { 2 } } } \right\} .\tag{22}
$$

Indeed, writing $d = r + r _ { 0 } , \mathrm { i f } v _ { r } \leq C / d$ with $C \geq 4 \sigma ^ { 2 } / \mu ^ { 2 }$ , then $v _ { r + 1 } \leq C / d - ( 2 C - 4 \sigma ^ { 2 } / \mu ^ { 2 } ) / d ^ { 2 } \leq C / d - C / d ^ { 2 } \leq$ $C / ( d + 1 )$ . The initial condition starts the induction. Since $C _ { \mathrm { o p t } } = O _ { p } ( 1 )$ , conditional Markov’s inequality and the stated containment condition $\mathrm { P r } ( \tau _ { N } \leq R )  0$ imply

$$
\zeta ^ { ( R ) } - \widehat { \zeta } _ { N } = O _ { p } ( R ^ { - 1 / 2 } ) .
$$

With high-probability containment, (22) bounds the stopped error $v _ { R } ;$ the exit probability is controlled separately by the containment assumption.

Step 5: The SGD output. Combining this expansion with (22) yields

$$
\| \zeta ^ { ( R ) } - \zeta ^ { \star } \| \le \| \widehat { \zeta } _ { N } - \zeta ^ { \star } \| + \| \zeta ^ { ( R ) } - \widehat { \zeta } _ { N } \| = O _ { p } ( N ^ { - 1 / 2 } + R ^ { - 1 / 2 } ) ,
$$

which proves (8). If $R / N \to \infty$ , the stopped mean-square bound and the containment condition imply $\sqrt { N } ( \zeta ^ { ( R ) } -$ $\widehat { \zeta } _ { N } ) = o _ { p } ( 1 )$ , so (21) proves (9). This proves Theorem 1.

Full-gradient updates. For full gradients and $0 < \eta ^ { \mathrm { o p t } } \leq \mu .$ , the same curvature and iterate-containment conditions give

$$
\lVert \zeta ^ { ( R ) } - \widehat { \zeta } _ { N } \rVert \leq ( 1 - \mu \eta ^ { \mathrm { o p t } } ) ^ { R } \lVert \zeta ^ { ( 0 ) } - \widehat { \zeta } _ { N } \rVert .\tag{23}
$$

Adding the statistical error gives $O _ { p } \{ N ^ { - 1 / 2 } + ( 1 - \mu \eta ^ { \mathrm { o p t } } ) ^ { R } \}$ . Thus $\sqrt { N } ( 1 - \mu \eta ^ { \mathrm { o p t } } ) ^ { R }  0$ suffices for the same normal limit. For fixed $c > 0$ , it is sufficient to take $R \geq ( 1 / 2 + c ) ( \mu \eta ^ { \mathrm { o p t } } ) ^ { - 1 } \log N ;$ ; when $\mu \eta ^ { \mathrm { o p t } } = 1$ the contraction is zero after one update.

## C.2 Balanced source fitting and prior-weighted comparison identities

For $e = \{ i , j \} , i < j$ , the source-label distribution in the edge risk assigns probability $1 / 2$ to each endpoint. Bayes’ rule under this balanced mixture gives

$$
q _ { i j } ^ { \mathrm { b a l } } ( z ) = \frac { \varphi _ { j } ( z ) } { \varphi _ { i } ( z ) + \varphi _ { j } ( z ) } , \qquad \mathrm { l o g i t } q _ { i j } ^ { \mathrm { b a l } } ( z ) = \log \frac { \varphi _ { j } ( z ) } { \varphi _ { i } ( z ) } .
$$

The equally weighted binary cross-entropy is minimized by this posterior, which contains no sample-count term and agrees with (5). Correct specification, connectivity, and the stated identifiable-design condition identify the routing coefficients relative to the fixed client 1. Correctly specified task cross-entropies identify the private-head contrasts. At inference, adding $\log ( \rho _ { j } / \rho _ { i } )$ to the fitted balanced logit converts the learned density ratio into the desired prior-weighted source odds. Thus the sample-count prior is absent from training and appears exactly once in deployment.

With a common fitted representation $z = g _ { \widehat { \theta } } ( x )$

$$
u _ { t } ( x , s ) = \log \frac { \rho _ { \widehat { s } _ { t } } } { \rho _ { s } } + ( \widehat { b } _ { \widehat { s } _ { t } } - \widehat { b } _ { s } ) + ( \widehat { \vartheta } _ { \widehat { s } _ { t } } - \widehat { \vartheta } _ { s } ) ^ { \top } z .
$$

The identity holds for any fitted coefficients with a common representation. If the DRM holds for this representation and its density-score differences are recovered exactly, then

$$
e ^ { u _ { t } ( x , s ) } = \frac { \rho _ { \widehat { s } _ { t } } \varphi _ { \widehat { s } _ { t } } ( z ) } { \rho _ { s } \varphi _ { s } ( z ) } , \qquad w _ { t } ( x , s ) = \frac { \rho _ { \widehat { s } _ { t } } \varphi _ { \widehat { s } _ { t } } ( z ) } { \sum _ { \ell = 0 } ^ { T } \rho _ { \widehat { s } _ { \ell } } \varphi _ { \widehat { s } _ { \ell } } ( z ) } .
$$

The history weights normalize the prior-weighted densities over the visited clients. With different local encoders, the count terms still telescope but the remaining score sum can be path dependent.

For a reference representation $g _ { \boldsymbol { \theta } } ( x )$ and its coefficients, adding and subtracting the fitted comparison at this representa tion gives

$$
\begin{array} { c } { { \widehat { \Delta } _ { i j } ( x ) - \Delta _ { i j } ( x ) = e _ { i j } ^ { \mathrm { c o e f } } ( x ) + ( \widehat { \vartheta } _ { j } - \widehat { \vartheta } _ { i } ) ^ { \top } \{ g _ { \widehat { \theta } _ { i } } ( x ) - g _ { \theta } ( x ) \} , } } \\ { { e _ { i j } ^ { \mathrm { c o e f } } ( x ) = ( \widehat { b } _ { j } - b _ { j } ) - ( \widehat { b } _ { i } - b _ { i } ) } } \\ { { + \{ ( \widehat { \vartheta } _ { j } - \vartheta _ { j } ) - ( \widehat { \vartheta } _ { i } - \vartheta _ { i } ) \} ^ { \top } g _ { \theta } ( x ) . } } \end{array}\tag{24}
$$

Both scores use the same finite-N count prior, so their known offsets cancel exactly, irrespective of how quickly the proportions converge. If the coefficient term is bounded by $\epsilon _ { a }$ , the fitted slope contrast by $M _ { \vartheta }$ , and the representation difference by $\delta _ { g } ,$ the score error is at most $\epsilon _ { a } + M _ { \vartheta } \delta _ { g } .$ For common representations and the same count prior, each comparison is affine in the identifiable coefficients. Using client 1 as the reference for both fitted and true coefficients, its error is bounded by $2 \| ( g _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) ^ { \top } , 1 ) ^ { \top } \|$ max<sub>i</sub> $\lVert \widehat { \boldsymbol { a } } _ { i } - \boldsymbol { a } _ { i } \rVert$ . Theorem 1 therefore proves (10) at a fixed query; bounded representations give a uniform version. For different local encoders, (24) remains an algebraic decomposition. However, these encoders also change the representation distributions used for coefficient fitting. The coefficient term may therefore contain effects of this change in addition to statistical estimation error, and Theorem 1 does not automatically bound it. Extending the rate to this setting requires controlling both the coefficient-fitting error and the representation-disagreement term.

The reverse-edge discrepancy is

$$
\widehat { \Delta } _ { i j } ( x ) + \widehat { \Delta } _ { j i } ( x ) = ( \widehat { \vartheta } _ { j } - \widehat { \vartheta } _ { i } ) ^ { \top } \{ g _ { \widehat { \theta } _ { i } } ( x ) - g _ { \widehat { \theta } _ { i } } ( x ) \} .\tag{25}
$$

Here the two count offsets and the intercept differences cancel even with different encoders. Let $\Delta _ { i j } ^ { \mathrm { p o p } } ( x )$ denote the comparison obtained by replacing the training proportions $\rho _ { i }$ with positive reference-population proportions $\rho _ { i } ^ { \mathrm { p o p } }$ . Its additional prior error is $\log \ ( \rho _ { j } / \rho _ { j } ^ { \mathrm { \varPhi o p } } ) - \log ( \bar { \rho _ { i } } ^ { \mathrm { \varPhi } } / \rho _ { i } ^ { \mathrm { \varPhi o p } } )$ , so

$$
| \widehat { \Delta } _ { i j } ( x ) - \Delta _ { i j } ^ { \mathrm { p o p } } ( x ) | \leq \epsilon _ { a } + M _ { \vartheta } \delta _ { g } + 2 \operatorname* { m a x } _ { i } | \log ( \rho _ { i } / \rho _ { i } ^ { \mathrm { p o p } } ) | .\tag{26}
$$

The last term measures the difference between the training-source prior and the reference population prior.

## C.3 Proof of Theorem 2 and structural examples

From estimation error to comparison tolerance. Under Theorem 1, (10) bounds the maximum comparison error over all graph edges at the fixed query x. By the definition of $O _ { p } ,$ , for every fixed $\delta \in ( 0 , 1 )$ there is a finite constant $C _ { \delta } > 0$ such that, for all sufficiently large $N , { \dot { R } } ,$ the event

$$
\mathcal { E } _ { \delta } : = \left\{ \operatorname* { m a x } _ { \{ i , j \} \in E } \left. \widehat { \Delta } _ { i j } ( x ) - \Delta _ { i j } ( x ) \right. \le C _ { \delta } \left( N ^ { - 1 / 2 } + R ^ { - 1 / 2 } \right) = \epsilon \right\}\tag{27}
$$

has probability at least $1 - \delta$ over the training data and SGD randomness. The constant may depend on the fixed query, graph, and model, but not on $N , R ,$ , or the realized route. With the common representation, reverse comparisons are negatives of each other, so this event controls both edge directions. It therefore controls every candidate edge on the adaptively selected route without an additional probability bound for individual steps. All remaining arguments are deterministic on ${ \mathcal { E } } _ { \delta } ;$ the reference potentials and $\dot { H } _ { \epsilon }$ are fixed given the training counts and query. For an independently random frozen representation, the argument is conditional on that representation, as in Theorem 1.

Definition of the depth. Suppress x in potentials and let $H _ { \epsilon }$ be the largest integer $q \geq 0$ for which there exists a path $( i _ { 0 } = s , i _ { 1 } , \ldots , i _ { q } )$ satisfying, at every step $r < q$

$$
i _ { r + 1 } \in \mathcal { N } ( i _ { r } ) , \qquad h _ { i _ { r + 1 } } > h _ { i _ { r } } , \qquad h _ { i _ { r + 1 } } \geq \operatorname* { m a x } _ { j \in \mathcal { N } ( i _ { r } ) } h _ { j } - 2 \epsilon .
$$

The one-node path is permitted. Strict increase excludes revisits, so $0 \leq H _ { \epsilon } \leq m - 1$ . The maximum is attained because the graph is finite. Throughout the proof, $H _ { \epsilon }$ denotes this exact maximum for the given graph, potentials, entry, and tolerance.

Proof of Theorem 2. Work on ${ \mathcal { E } } _ { \delta }$ and put $d _ { \epsilon } = \gamma - 2 \epsilon > 0$

Step 1: Accepted moves and stopping. Every accepted edge satisfies $\Delta _ { i j } \ge \widehat { \Delta } _ { i j } - \epsilon > \eta - \epsilon \ge 0$ . Hence every previously visited client has smaller potential than the current client, and every positive-gain neighbor is unvisited. At a nonoptimal client i, let $j ^ { \star }$ maximize the true neighboring gain. By (11), it is eligible and $\widehat { \Delta } _ { i j ^ { \star } } \geq \gamma - \epsilon > \eta$ . Thus the route cannot stop at i while budget remains. For the selected neighbor j,

$$
\Delta _ { i j } \ge \widehat { \Delta } _ { i j } - \epsilon \ge \widehat { \Delta } _ { i j ^ { \star } } - \epsilon \ge \Delta _ { i j ^ { \star } } - 2 \epsilon \ge d _ { \epsilon } .
$$

This also gives $h _ { j } \geq \operatorname* { m a x } _ { k \in \mathcal { N } ( i ) } h _ { k } - 2 \epsilon$ , so the realized route is admissible in the definition of $H _ { \epsilon }$ . Consequently $T \leq \operatorname* { m i n } \{ L , H _ { \epsilon } \}$ and

$$
\Gamma _ { L } ( x , s ) \geq T ( \gamma - 2 \epsilon ) .\tag{28}
$$

At a global maximizer, every candidate fitted score is at most $\epsilon \leq \eta ,$ so no further move is accepted. An early stop therefore occurs at a maximizer.

Step 2: The budget bound. If $T = L , ( 2 8 )$ gives $\Gamma _ { L } \geq L d _ { \epsilon }$ . If T < L, Step 1 gives $\Gamma _ { L } = h ^ { \star } - h _ { s }$ . Thus

$$
\Gamma _ { L } \geq \operatorname* { m i n } \{ h ^ { \star } - h _ { s } , L ( \gamma - 2 \epsilon ) \} .\tag{29}
$$

Step 3: The total potential gap and ascent depth. Let $( i _ { 0 } = s , \ldots , i _ { H _ { \epsilon } } )$ attain the maximum in the definition of $H _ { \epsilon }$ Its terminal vertex is a maximizer: otherwise, a highest-potential neighbor has positive gain by $( 1 1 ) .$ , is unvisited by strict ascent, and extends the admissible path. For each $r < H _ { \epsilon }$ , the near-best-neighbor condition gives

$$
h _ { i _ { r + 1 } } - h _ { i _ { r } } \geq \operatorname* { m a x } _ { j \in \mathcal { N } ( i _ { r } ) } ( h _ { j } - h _ { i _ { r } } ) - 2 \epsilon \geq d _ { \epsilon } .
$$

Telescoping along this path yields the separate total-gap bound

$$
h ^ { \star } - h _ { s } = \sum _ { r = 0 } ^ { H _ { \epsilon } - 1 } ( h _ { i _ { r + 1 } } - h _ { i _ { r } } ) \geq H _ { \epsilon } d _ { \epsilon } .
$$

Together with Step 2, this proves (12):

$$
\Gamma _ { L } \geq \mathrm { m i n } \{ h ^ { \star } - h _ { s } , L d _ { \epsilon } \} \geq d _ { \epsilon } \mathrm { m i n } \{ L , H _ { \epsilon } \} ,
$$

Step 4: A sufficient budget for reaching a maximizer. Suppose $L \geq H _ { e }$ <sub>ϵ</sub> and the terminal client is suboptimal. Step 1 then implies budget exhaustion, so $T \stackrel { = } { = } L$ . Together with $T \leq H _ { \epsilon }$ , this gives $T = L = H _ { \epsilon }$ . A highest-potential neighbor would extend the realized admissible path, contradicting the definition of $H _ { \epsilon }$ . Thus the terminal client is a maximizer. The same arguments include $L = 0$ and an optimal entry, and hold under any tie breaking among score maximizers. □

Examples and the proportional-ascent alternative. The following sufficient conditions concern the prior-adjusted potentials at a fixed query. Their conclusions hold on the comparison-error event used above.

Assumption 1 (Uniform ascent). Every nonoptimal client has a neighbor with gain at least $\gamma > 0 ,$ , as in (11).

Assumption 2 (Referral hierarchy). A spanning tree is oriented toward a maximizing root. Each nonroot client i has a parent $p _ { x } ( i )$ such that

$$
\Delta _ { i , p _ { x } ( i ) } \geq \gamma , \qquad \Delta _ { i , p _ { x } ( i ) } - \operatorname* { m a x } _ { j \in \mathcal { N } ( i ) \backslash \{ p _ { x } ( i ) \} } \Delta _ { i j } \geq \kappa _ { \mathrm { h i e r } } > 0 .
$$

When $\kappa _ { \mathrm { h i e r } } > 2 \epsilon .$ , only the parent can belong to an admissible ascent path. Thus $H _ { \epsilon }$ is exactly the number of parent edges from s to the root. Under the theorem’s threshold condition,

$$
T = \operatorname* { m i n } \{ L , H _ { \epsilon } \} , \quad \quad \Gamma _ { L } \geq \gamma \operatorname* { m i n } \{ L , H _ { \epsilon } \} .\tag{30}
$$

The preferred-parent condition therefore identifies ascent depth with the distance to the root in this hierarchy.

Assumption 3 (Mixture of expert types). Clients ofthe same type $\tau ( i )$ have equal potential. There is a best $t y p e \tau ^ { \star } ( x )$ with $h _ { j } - h _ { i } \geq \gamma > 0$ whenever $\dot { \mathbf { \bar { \rho } } } ( \bar { j } ) = \tau ^ { \star } ( x )$ and $\tau ( i ) \neq \tau ^ { \star } ( x )$ . Every nonbest-type client has a best-type neighbor.

$\mathrm { I f } \gamma > \eta + \epsilon$ and $\eta \geq \epsilon ,$ , then $\gamma > 2 \epsilon$ . Only best-type neighbors are admissible, so $H _ { \epsilon } = 1$ for a nonoptimal entry and $H _ { \epsilon } = 0$ for an optimal entry. Therefore

$$
\Gamma _ { L } = ( h ^ { \star } - h _ { s } ) \mathbf { 1 } \{ L \geq 1 \} .\tag{31}
$$

Equal representation distributions and equal source priors within each type suffice for the equal-potential condition. Assumption 4 (Graph–PL ascent). There is $\kappa _ { \mathrm { P L } } \in ( 0 , 1 ]$ such that each nonoptimal client satisfies max $\ _ { j \in \mathcal { N } ( i ) } \Delta _ { i j } \geq$ $\kappa _ { \mathrm { P L } } ( h ^ { \star } - h _ { i } )$

This is an alternative to uniform ascent: the available improvement scales with the remaining potential gap. Under this assumption, the candidate error bound and $\eta \geq \epsilon$ alone imply

$$
\Gamma _ { L } \geq \left\{ 1 - ( 1 - \kappa _ { \mathrm { P L } } ) ^ { L } \right\} \left[ h ^ { \star } - h _ { s } - \frac { \eta + \epsilon } { \kappa _ { \mathrm { P L } } } \right] _ { + } .\tag{32}
$$

To prove it, hold a stopped route at its terminal node until index L only for the analysis, and put $d _ { t } = h ^ { \star } - h _ { \widehat { s } _ { t } }$ . On a moving step, the preceding noisy-greedy inequality gives $d _ { t + 1 } \leq ( \mathrm { i } - \kappa _ { \mathrm { P L } } ) d _ { t } + 2 \epsilon$ . On a stopped step, the best neighboring gain is at most $\eta + \epsilon ,$ so $\kappa _ { \mathrm { P L } } d _ { t } \leq \eta + \epsilon$ and $d _ { t + 1 } = d _ { t } \leq ( 1 - \kappa _ { \mathrm { P L } } ) d _ { t } + \eta + \epsilon .$ . Using 2ϵ $\leq \eta + \epsilon$ and solving the common recursion proves (32). In particular an early stop satisfies

$$
T < L \quad \Longrightarrow \quad h ^ { \star } - h _ { \widehat { s } _ { T } } \leq \frac { \eta + \epsilon } { \kappa _ { \mathrm { P L } } } .\tag{33}
$$

When $T = L ,$ , the sharper $2 \epsilon$ numerator follows from the moving-step recursion. Use $( 1 - \kappa _ { \mathrm { P L } } ) ^ { 0 } = 1$ , including $\kappa _ { \mathrm { P L } } = 1$ . The holding steps are an analytical convention; $T ,$ , the prediction history, and communication counts are those of the original route.

## D Experiment Details

The subsections below collect the data construction, communication graphs, optimization schedules, baseline provenance, sensitivity analyses, inference diagnostics, and real-data protocols used throughout Section 5.

## D.1 CIFAR Data, Graphs, Baselines, and Traffic Construction

The benchmark uses the client construction from the released FedDRM code (Wang et al., 2026a). CIFAR-10 has ten labels, CIFAR-20 maps the CIFAR-100 fine labels to 20 superclasses, and CIFAR-100 retains all 100 fine labels. The official training and test collections are pooled before partitioning among eight clients using one of two label-skew constructions. In the SPC construction, label-sorted data are divided into equal-sized, label-homogeneous shards, and each client receives five uniformly sampled shards for CIFAR-10 or 25 for CIFAR-20/100, yielding equal client dataset sizes. In the Dirichlet construction, client-wise class proportions are randomly drawn from a Dirichlet distribution with concentration 0.3, yielding heterogeneous label marginals and unequal client dataset sizes. Each client’s assigned images are shuffled and divided into 70% training and 30% evaluation data. The latter is split into 20% validation and 80% test data. Training, validation, and test indices are disjoint.

The eight deterministic client transformations combine a gamma-correction exponent in {0.6, 1.4}, hue shift in $\{ - 0 . 1 \bar { , } 0 . 1 \}$ , and saturation in {0.5, 1.5}. The CIFAR models use a GroupNorm ResNet-18 followed by a 256- dimensional task projection with LayerNorm and a local classifier. Within each paired run, all clients share a random initialization; no pretrained FedDRM parameters are loaded. Both CIFAR training phases retain the released randomresized crop to $3 2 \times 3 2$ pixels with scale 0.9–1.0, followed by the fixed client color transformation and normalization. Evaluation uses one seeded crop per image, shared across paired models and readouts. Freezing fixes encoder parameters and buffers; it does not fix the augmented input at every training access. The fixed evaluation view and non-source entry assignments are shared between each 800+200 model and its DFedAvgM control. The source ID is used to construct the evaluation and for observer-only diagnostics; it is not given to the routing or history-weighting rule.

The PATH has edges $\{ \{ i , i + 1 \} : 1 \leq i < 8 \}$ , and the CYCLE adds {8, 1}. The fixed RR4 graph has neighbor sets

$$
\begin{array} { r l } & { \mathcal { N } ( 1 ) = \{ 2 , 3 , 4 , 5 \} , \quad \mathcal { N } ( 2 ) = \{ 1 , 3 , 6 , 7 \} , \quad \mathcal { N } ( 3 ) = \{ 1 , 2 , 4 , 8 \} , \quad \mathcal { N } ( 4 ) = \{ 1 , 3 , 5 , 7 \} , } \\ & { \mathcal { N } ( 5 ) = \{ 1 , 4 , 6 , 8 \} , \quad \mathcal { N } ( 6 ) = \{ 2 , 5 , 7 , 8 \} , \quad \mathcal { N } ( 7 ) = \{ 2 , 4 , 6 , 8 \} , \quad \mathcal { N } ( 8 ) = \{ 3 , 5 , 6 , 7 \} . } \end{array}
$$

The COMPLETE graph contains all 28 undirected edges. The sparse-graph movement budgets are the diameters $7 , 4 ,$ and 2. The primary COMPLETE control uses $L = 8 ; \check { L } = 1$ is retained for the training-duration diagnostics. Previously visited experts are excluded from subsequent moves. No posttraining representation consensus is applied.

Thirty-two-client scaling. The CIFAR-10 scaling benchmark uses gamma-correction levels {0.6, 1.0, 1.4}, hue shifts {−0.15, 0, 0.15}, saturation levels {0.4, 1.0, 1.6}, and two-bit posterization for clients 28–32. The pooled-data construction and within-client split remain the same. The connected degree-four graph uses seed 82025, has 64 edges and diameter four, and retains the primary movement budget L = 8.

Eight-client baseline measurements. For Table 1, Local and FedDRM use 1000 rounds with ten local updates per round. DisPFL uses 1000 rounds with five local updates per round and 50% dynamic sparsity. FedSPD uses 999 communication rounds with five local updates per round, followed by five final personalization updates, yielding 1000 round equivalents and 5000 local updates per client. All four baselines are evaluated at fixed final checkpoints on the same seed-specific held-out test split used for EdgeDRM and DFedAvgM; Table 1 reports means and sample SDs over seeds 2025–2027. Local, DisPFL, and FedSPD predict with the non-source entry model, whereas FedDRM uses its global client-selection head. The round-equivalent budgets and evaluation splits are aligned, although the numbers of local updates differ.

Hospital-network construction. The 16/8/8 small/medium/large hospitals receive disjoint, exactly class-balanced subsets of the official CIFAR-10 training split with 250/1,000/4,000 examples each. The layered graph has a cycle within each tier, one small–medium edge per small hospital, and a one-to-one medium–large matching (56 edges). The shortcut graph adds one predetermined small–large edge per small hospital (72 edges). Each graph uses its own shared-training prefix and frozen-encoder continuation.

The common diagnostic panel contains 20 official test images per class. Each of the 200 distinct images is reused in 64 arrival records, yielding 12,800 records with exactly uniform entry counts across hospitals. Seeds 2025–2027 vary training and entry assignments. The prior-on and prior-off evaluations share the model, query panel, and arrival records; both exclude revisits and allow at most two referrals. The panel is an existing diagnostic set rather than a newly blinded test set. Workload quantities count the diagnostic calls made along these finite query paths.

Cumulative diagnostic workload. Let $\mathcal { V } _ { q }$ be the visited hospitals for arrival $q ,$ and let $\mathcal { H } _ { g }$ contain the hospitals in tier g. Each hospital supplies one prediction when visited, including the entry and terminal hospitals. Because the route excludes revisits, its cumulative call counts are

$$
C _ { i } ^ { \mathrm { c a l l } } = \sum _ { q = 1 } ^ { Q } \mathbf { 1 } \{ i \in \mathcal { V } _ { q } \} , \qquad C _ { g } ^ { \mathrm { c a l l } } = \sum _ { i \in \mathcal { H } _ { g } } C _ { i } ^ { \mathrm { c a l l } } , \qquad C _ { \mathrm { a l l } } ^ { \mathrm { c a l l } } = \sum _ { i } C _ { i } ^ { \mathrm { c a l l } } = \sum _ { q = 1 } ^ { Q } ( T _ { q } + 1 ) ,
$$

where $T _ { q }$ is the number of moves and Q = 12,800. The reported tier share, peak individual share, and calls per arrival are, respectively,

$$
S _ { g } = \frac { C _ { g } ^ { \mathrm { c a l l } } } { C _ { \mathrm { a l l } } ^ { \mathrm { c a l l } } } , \qquad P _ { \mathrm { m a x } } = \frac { \mathrm { m a x } _ { i } C _ { i } ^ { \mathrm { c a l l } } } { C _ { \mathrm { a l l } } ^ { \mathrm { c a l l } } } , \qquad \overline { { C } } ^ { \mathrm { c a l l } } = \frac { C _ { \mathrm { a l l } } ^ { \mathrm { c a l l } } } { Q } .
$$

Table 4 reports both the large-tier share $S _ { \mathrm { l a r g e } }$ and $1 0 0 C _ { \mathrm { l a r g e } } ^ { \mathrm { c a l l } } / Q$ , retaining the absolute large-hospital workload scale as the total number of calls changes. It also reports $\overline { { C } } ^ { \mathrm { c a l l } }$ and $P _ { \mathrm { m a x } }$ to distinguish additional collaboration from concentration at any single hospital. Each seed’s counts and ratios are computed first, followed by their mean and sample SD across the three seeds. Per-hospital counts reconstructed from the saved paths match the recorded classifier calls. These quantities weight each classifier invocation equally.

Table 4: Accuracy and diagnostic workload in the hospital-network experiment. Large share and peak share are percentages of all visited-classifier calls; large calls/100 is the absolute number of large-hospital calls per 100 arrivals. Accuracy uses history-weighted predictions. Entries are mean ± sample SD over seeds 2025–2027; EdgeDRM-NR workload quantities are fixed by uniform entry.
<table><tr><td>Condition</td><td> $\mathrm { A c c u r a c y } \left( \% \right)$ </td><td>Calls/query</td><td>Large share (%)</td><td>Large calls/100</td><td>Peak share (%)</td></tr><tr><td>Layered: EdgeDRM-NR</td><td> $4 5 . 2 9 \pm 1 . 8 0$ </td><td>1</td><td>25</td><td>25</td><td> $^ { 3 . 1 2 5 }$ </td></tr><tr><td>Layered: History, no prior</td><td> $4 5 . 7 4 \pm 1 . 6 0$ </td><td> $2 . 0 1 \pm 0 . 0 3$ </td><td> $1 9 . 5 5 \pm 0 . 4 4$ </td><td> $3 9 . 2 2 \pm 1 . 5 6$ </td><td> $5 . 0 4 \pm 0 . 4 4$ </td></tr><tr><td>Layered: History + prior</td><td> $4 7 . 3 6 \pm 3 . 0 6$ </td><td> $2 . 6 1 \pm 0 . 0 1$ </td><td> $5 2 . 0 2 \pm 0 . 1 2$ </td><td> $1 3 5 . 5 2 \pm 0 . 6 3$ </td><td> $7 . 1 4 \pm 0 . 2 0$ </td></tr><tr><td>Shortcut: EdgeDRM-NR</td><td> $4 5 . 4 9 \pm 2 . 0 3$ </td><td>1</td><td>25</td><td>25</td><td> $^ { 3 . 1 2 5 }$ </td></tr><tr><td>Shortcut: History, no prior</td><td> $4 6 . 1 2 \pm 1 . 7 3$ </td><td> $2 . 0 1 \pm 0 . 0 2$ </td><td> $2 0 . 3 6 \pm 0 . 3 3$ </td><td> $4 0 . 9 6 \pm 0 . 3 9$ </td><td> $4 . 6 1 \pm 0 . 5 9$ </td></tr><tr><td>Shortcut: History + prior</td><td> $4 7 . 9 3 \pm 2 . 5 3$ </td><td> $2 . 4 6 \pm 0 . 0 1$ </td><td> $6 9 . 4 9 \pm 0 . 1 0$ </td><td> $1 7 0 . 8 1 \pm 0 . 8 1$ </td><td> $9 . 7 7 \pm 0 . 1 8$ </td></tr></table>

## D.2 CIFAR Training, Evaluation, and Additional Results

The balanced-endpoint schedule below applies to the benchmark and hospital protocol. The prediction-rule comparison uses CIFAR-100/Dirichlet-0.3, and the sensitivity sweep uses CIFAR-10/Dirichlet-0.3. Both use eight clients and seed 2025–2027. Each primary model uses its saved 200-round head and router checkpoint with the original 800-round encoder state. The same continuation length is used for every graph and seed. Evaluation preserves the training random-number and loader states

## D.2.1 Training schedule and implementation details

The routing loss assigns equal weight to the two endpoint means of every edge. At inference, the deployed incident score adds the single sample-count offset $\log ( n _ { j } / n _ { i } )$ . The pretraining duration and subsequent encoder freezing are experimental optimization choices, rather than defining components of EdgeDRM.

Stage I: 800 shared-training rounds. All clients start from the same model initialization. Each round takes local SGD steps on the task loss and then mixes the full model $\psi _ { i } = \left( \theta _ { i } , \omega _ { i } \right)$ by (4). Both representation and classifier are exchanged, using only physical neighbors. Thus shared refers to full-model collaboration toward a common task objective, not a server-owned model or identical local parameters. This stage uses the same graph and full-model mixing as the matched DFedAvgM control (Sun et al., 2023).

Stage II: 200 rounds with frozen local encoders. Each expert inherits its own Stage-I model and sampler state, fixes $\bar { \theta } _ { i } = \theta _ { i } ^ { ( 8 0 0 ) }$ , and initializes its routing potential $a _ { i } = ( \vartheta _ { i } , b _ { i } )$ to zero. The encoder includes the backbone, feature projection, and feature normalization. Its parameters and persistent buffers remain unchanged, and it runs in evaluation mode. There are no encoder gradient updates or representation-mixing steps:

$$
\theta _ { i } ^ { ( 8 0 0 + r ) } = \bar { \theta } _ { i } , \qquad r = 0 , \ldots , 2 0 0 .
$$

Each local step minimizes (18) only over the private classifier $\omega _ { i }$ and the endpoint potentials. Classifier heads are not exchanged, and potentials are not averaged. Freezing keeps each local feature map stationary during this stage; it does not align different experts’ feature maps. The fitted representation used at inference is $\widehat { \theta } _ { i } = \bar { \theta } _ { i }$ under this experimental protocol.

The head and routing-potential updates instantiate Stage II of Section 3.1, with no parameter mixing. Routing gradients do not flow into the encoder or task head. Training counts are exchanged before coefficient fitting. After round 200, final potentials are published to neighbors. All parameters remain fixed during inference, and no additional consensus phase is applied.

Algorithm 2 Experimental 800+200 training schedule   
Require: Graph G, neighbor weights W, local steps $B _ { \mathrm { l o c } } , \lambda ; R _ { \mathrm { s h } } = 8 0 0 , R _ { \mathrm { p e r } } = 2 0 0$   
1: Initialize all local task models identically   
2: for $r = 1 , \ldots , R _ { \mathrm { s h } }$ do   
3: Each client takes $B _ { \mathrm { l o c } }$ task-only SGD steps   
4: Mix full models with physical neighbors using W   
5: end for   
6: Inherit local models and sampler states; set $\bar { \theta } _ { i }  \theta _ { i } ^ { ( 8 0 0 ) }$   
7: Freeze encoder parameters and buffers in evaluation mode; initialize owner potentials to zero and exchange training   
counts along edges   
8: for $r = 1 , \ldots , R _ { \mathrm { p e r } }$ do   
9: for $\ell = 1 , \ldots , \overbar { B } _ { \mathrm { l o c } }$ do   
10: Owners send potential snapshots to incident neighbors   
11: Each endpoint computes local features with its frozen encoder   
12: Differentiate task loss for its private head and balanced endpoint losses for endpoint potentials   
13: Return neighbor-owned potential gradients along their edges   
14: Update only private heads and owner potentials with accumulated gradients   
15: end for   
16: end for   
17: Publish final potentials to neighbors; freeze heads and routers

Fixed budget and learning-rate schedule. For the CIFAR protocol, each round has five local mini-batches of size 128. SGD uses momentum 0.9, reset each round, and weight decay $5 \times 1 0 ^ { - 4 }$ . The phase-local learning rate is

$$
\eta _ { r } ^ { \mathrm { e x p } } = 0 . 0 0 5 \{ 1 + \cos ( \pi ( r - 1 ) / H _ { \mathrm { I r } } ) \} , \qquad H _ { \mathrm { I r } } = 8 0 0 \mathrm { i n } \mathrm { S t a g e I } , \quad H _ { \mathrm { I r } } = 6 0 0 \mathrm { i n } \mathrm { S t a g e I I } .\tag{34}
$$

Stage II restarts at 0.01 but stops at round 200, with $\eta _ { 2 0 0 } ^ { \mathrm { e x p } } \simeq 0 . 0 0 7 5 2 2 6 4 ;$ it does not run to the end of the 600-round decay horizon. The encoder remains frozen throughout this phase; the schedule applies only to heads and routing potentials. The model at total round 1000 is the fixed endpoint, without test-based early stopping. The finite neural optimizer with weight decay is distinct from the unpenalized fixed-representation estimator in Section 4.

This schedule keeps local features stationary during coefficient fitting, but does not make different clients’ representations identical or independent of the reused training data. It therefore does not by itself satisfy the statistical assumptions of Theorem 1.

## D.2.2 Fixed-expert referral-score ablation

The comparison in Figure 2 reuses the final CIFAR-10 and CIFAR-100 Dirichlet-0.3 RR4 checkpoints for all three seeds. No model parameter is updated. EdgeDRM routing uses the deployed incident score, including the single $\log ( n _ { j } / n _ { i } )$ offset. EdgeDRM-NR reports the same entry expert without referral. Count-only replaces the learned part of every incident score by zero, leaving exactly $\log ( n _ { j } / n _ { i } )$ , and retains the learned rule’s strict zero threshold. Random routing samples uniformly from the unvisited incident neighbors and stops only at the two-move budget or when none remain; its seed is fixed before evaluation. All three routing rules exclude revisits. The primary prediction is the terminal expert’s classifier, which changes only through the selected route.

## D.2.3 Sensitivity of λ

There are three independent training seeds; graphs reuse each seed’s partition. Validation/test query counts are 3,600/14,403 for seed 2025, 3,603/14,400 for seed 2026, and 3,601/14,402 for seed 2027. All prediction rules and controls use matched query indices, labels, source IDs, entry nodes, and shared-checkpoint hashes. Accuracy is measured over the fixed non-source arrivals, with one view per query.

Table 5: Task–routing weight on CIFAR-10/Dirichlet-0.3, eight clients, RR4, and $L = 2 .$ . All branches reuse the same seed-specific 800-round prefix and fit heads and routers for 200 rounds. Accuracy is in percent, referral gain in percentage points, and calls per query. Entries are final-checkpoint means ± sample SD across seeds 2025–2027.
<table><tr><td>λ</td><td>EdgeDRM</td><td>EdgeDRM-NR</td><td>Referral gain</td><td>Calls</td></tr><tr><td>0.10</td><td> $5 2 . 1 1 \pm 2 . 1 9$ </td><td> $3 5 . 0 1 \pm 3 . 5 5$ </td><td> $+ 1 7 . 1 1 \pm 3 . 9 8$ </td><td> $2 . 1 6 \pm 0 . 0 3$ </td></tr><tr><td>0.25</td><td> $5 1 . 7 6 \pm 2 . 0 4$ </td><td> $3 0 . 2 9 \pm 4 . 0 6$ </td><td> $+ 2 1 . 4 6 \pm 4 . 9 9$ </td><td> $2 . 1 6 \pm 0 . 0 3$ </td></tr><tr><td>0.40</td><td> $5 1 . 5 7 \pm 2 . 0 3$ </td><td> $2 8 . 3 6 \pm 4 . 2 3$ </td><td> $+ 2 3 . 2 1 \pm 5 . 3 6$ </td><td> $2 . 1 5 \pm 0 . 0 3$ </td></tr><tr><td>0.50</td><td> $5 1 . 5 1 \pm 2 . 0 4$ </td><td> $2 7 . 4 8 \pm 4 . 3 2$ </td><td> $+ 2 4 . 0 3 \pm 5 . 6 0$ </td><td> $2 . 1 5 \pm 0 . 0 3$ </td></tr><tr><td>0.60</td><td> $5 1 . 2 6 \pm 2 . 0 4$ </td><td> $2 6 . 7 5 \pm 4 . 4 0$ </td><td> $+ 2 4 . 5 1 \pm 5 . 7 1$ </td><td> $2 . 1 4 \pm 0 . 0 3$ </td></tr><tr><td>0.75</td><td> $5 0 . 9 5 \pm 2 . 0 9$ </td><td> $2 5 . 9 6 \pm 4 . 4 9$ </td><td> $+ 2 4 . 9 9 \pm 5 . 9 9$ </td><td> $2 . 1 3 \pm 0 . 0 4$ </td></tr><tr><td>0.90</td><td> $5 0 . 1 2 \pm 2 . 2 1$ </td><td> $2 5 . 3 5 \pm 4 . 5 4$ </td><td> $+ 2 4 . 7 7 \pm 6 . 3 5$ </td><td> $2 . 1 1 \pm 0 . 0 5$ </td></tr><tr><td>0.95</td><td> $4 9 . 3 0 \pm 2 . 3 3 $ </td><td> $2 5 . 1 8 \pm 4 . 5 6$ </td><td> $+ 2 4 . 1 2 \pm 6 . 5 1$ </td><td> $2 . 0 9 \pm 0 . 0 6$ </td></tr><tr><td>1.00</td><td> $3 0 . 6 5 \pm 2 . 8 1$ </td><td> $2 4 . 9 7 \pm 4 . 5 4$ </td><td> $+ 5 . 6 8 \pm 2 . 0 0$ </td><td> $1 . 9 3 \pm 0 . 2 8$ </td></tr></table>

Table 6: Prediction-rule comparison and inference costs using identical frozen models and paths. Accuracy is in percent (three-seed mean ± sample SD). Hard terminal uses the final visited expert; uniform and weighted history use only visited predictions. Edge scores and classifier calls are mean logical operations per query, not measured latency.
<table><tr><td>Graph</td><td>Hard terminal</td><td>Uniform history</td><td>Weighted history</td><td>Calls</td><td>Incident scores</td></tr><tr><td>PATH</td><td> $1 1 . 0 9 \pm 0 . 6 0$ </td><td> $1 0 . 4 3 \pm 0 . 7 9$ </td><td> ${ \bf 1 1 . 1 9 \pm 0 . 5 8 }$ </td><td>1.72</td><td>2.35</td></tr><tr><td>CYCLE</td><td> $1 2 . 2 9 \pm 0 . 6 3$ </td><td> $1 1 . 6 2 \pm 0 . 7 9$ </td><td> ${ \bf 1 2 . 4 0 \pm 0 . 6 3 }$ </td><td>1.81</td><td>2.81</td></tr><tr><td>RR4</td><td> $1 7 . 2 8 \pm 0 . 5 5$ </td><td> $1 5 . 6 2 \pm 0 . 4 7$ </td><td> $\mathbf { 1 7 . 5 9 \pm 0 . 5 9 }$ </td><td>2.10</td><td>6.56</td></tr><tr><td>COMPLETE</td><td> ${ \bf 2 0 . 0 5 \pm 0 . 5 5 }$ </td><td> $1 7 . 7 1 \pm 0 . 6 7$ </td><td> $2 0 . 0 2 \pm 0 . 6 3$ </td><td>1.94</td><td>12.66</td></tr></table>

The sweep in Table 5 is a sensitivity analysis rather than test-set model selection. Referral remains effective across the nonzero routing-loss settings, whereas removing the routing term at $\lambda = 1$ weakens performance.

Table 6 compares prediction rules on identical routes. Weighted history achieves the highest mean accuracy on the sparse graphs and is comparable to hard-terminal prediction on COMPLETE.

## D.3 ISIC2019: Data, Protocols, and Additional Results

The study in Section 5.4 uses a fixed 400-round shared-training prefix followed by 50 rounds of balanced-endpoint frozen-encoder adaptation, with three training seeds.

## D.3.1 Shared data, preprocessing, and graphs

Data provenance and center identities. We use the 23,247 images and six source-center assignments in the ISIC2019 split distributed with FLamby (du Terrail et al., 2022). Images come from the public ISIC2019 training collection, including BCN20000, HAM10000, and MSK contributions (Hernandez-P´ erez et al., 2024; Tschandl et al., 2018;´ Codella et al., 2018). In particular, the evaluation set here is FLamby’s held-out subset of that collection, not the separate ISIC challenge test collection. Center identities follow source and acquisition metadata and should not be interpreted as six independent hospitals. The FedSPM package (Wang et al., 2026b) serves only as a reference for the data input pipeline; we do not use its within-server subpopulation model or its pooled resplitting procedure. Each observed center is represented by one expert.

Leakage control and fixed split. The released FLamby partition contains 18,597 training and 4,650 test images. Our metadata audit finds 2,302 lesion identifiers shared across these folds. We preserve every test image and label and remove the 6,102 training images whose lesion identifier also occurs in the test fold. We additionally link lesion identifiers through identical original or preprocessed image files and one visually confirmed near-duplicate pair. These links are computed globally, including across centers; groups with conflicting labels are excluded from the training candidates. The image-level audit removes four additional training candidates. A deterministic grouped split (seed 42) targets a 20% validation share within center–class cells, while reserving at least one training group for each represented cell. The resulting counts are given in Table 7. Training, validation, and test are disjoint in the audited lesion/duplicate groups. The grouping metadata do not establish patient-level independence, and the near-duplicate review is not an exhaustive guarantee against all transformed duplicates. This modified split is not directly comparable to the unmodified FLamby leaderboard.

Table 7: Fixed ISIC2019 split after global lesion and identified duplicate grouping. Excluded images come from the original training fold. The $n _ { i }$ used for the deployed sample-count offset are the final training counts.
<table><tr><td>Node</td><td>FLamby source</td><td>Train</td><td>Validation</td><td>Test</td><td>Excluded</td></tr><tr><td>1</td><td>BCN</td><td>3,802</td><td>954</td><td>2,483</td><td>5,174</td></tr><tr><td>2</td><td>vidir_molemax</td><td>2,528</td><td>632</td><td>791</td><td>3</td></tr><tr><td>3</td><td>vidir_modern</td><td>1,686</td><td>424</td><td>672</td><td>581</td></tr><tr><td>4</td><td>rosendahl</td><td>1,221</td><td>308</td><td>452</td><td>278</td></tr><tr><td>5</td><td>MSK4</td><td>520</td><td>131</td><td>164</td><td>4</td></tr><tr><td>6</td><td>vienna_dias</td><td>229</td><td>56</td><td>88</td><td>66</td></tr><tr><td colspan="2">Total</td><td>9,986</td><td>2,505</td><td>4,650</td><td>6,106</td></tr></table>

Images and model. We resize each RGB image with bilinear interpolation so that its shorter side is 224 pixels, apply the FLamby shades-of-gray color-constancy transform with power six, and save a JPEG with quality 75. Training uses a random 200 × 200 crop, independent horizontal and vertical flips, rotation within ±50 degrees, and brightness/contrast jitter of 0.15/0.1. Validation and test use a deterministic center crop. All roles use ImageNet channel normalization. These torchvision transforms are shared by all methods; we do not claim exact reproduction of FLamby’s augmentation code. The model is the existing FedDRM GroupNorm ResNet-18 with its stride-one stem, a $5 1 2  2 5 \dot { 6 }$ projection with LayerNorm/ReLU, and a private eight-class linear head. Within each seed, all six experts and all methods start from the same random initialization; no pretrained weights are used. EdgeDRM adds a 256-dimensional slope and scalar intercept per node.

Communication graphs. The four undirected graphs—PATH, CYCLE, RR3, and COMPLETE—are fixed before training. PATH has edges $\{ i , i + 1 \}$ for $i = 1 , \ldots , 5 .$ , and CYCLE adds the edge {1, 6}. The intermediate regular graph is

$$
\begin{array} { r } { E ( \mathrm { R R } 3 ) = \{ \{ 1 , 2 \} , \{ 1 , 5 \} , \{ 1 , 6 \} , \{ 2 , 3 \} , \{ 2 , 4 \} , \{ 3 , 5 \} , \{ 3 , 6 \} , \{ 4 , 5 \} , \{ 4 , 6 \} \} , } \end{array}
$$

COMPLETE contains all pairs. Their edge counts are 5, 6, 9, and 15 and their diameters are 5, 3, 2, and 1. We use symmetric Metropolis mixing, with $\bar { W _ { i j } } \bar { = } 1 / ( 1 + \operatorname* { m a x } \{ \deg ( i ) , \deg ( j ) \} )$ ) on an edge and the diagonal completing each row to one. Graph diameter alone does not ensure that the positive-gain, no-revisit referral rule reaches a globally best expert.

## D.3.2 Training Details

Shared-training prefix. Each graph uses six identically initialized local models, three training seeds, and 400 rounds of task-only SGD followed by full-model Metropolis mixing. There are ten local optimizer steps per round, effective batch size 32, and microbatch size 16. The initial learning rate is 0.03 with a 400-round cosine horizon; SGD momentum is 0.9, reset each round, and weight decay is $5 \times 1 0 ^ { - 4 }$ . The RR3 prefix reuses the unchanged final checkpoint of the earlier DFedAvgM run, before its evaluation-only consensus. The other graphs require their own task-only prefixes; old joint-training EdgeDRM checkpoints are not substituted for them. The comparison uses fixed round 400, not the historical validation-selected checkpoint.

Frozen adaptation. The backbone, feature projection, feature normalization, and their buffers are fixed to the prefix checkpoint and evaluated in inference mode. Private classifier heads inherit their prefix values, and all node routing potentials start at zero. For 50 rounds, five local mini-batches per round update only the heads and balanced-endpoint routing potentials, using equal endpoint weights and $\lambda = 0 . 9$ . Microbatch gradients are weighted by their fractions of the full batch. Endpoint owners accumulate both local and neighbor contributions from a common snapshot before updating. No encoder or head mixing is performed, and no extra consensus phase is added. The head/router learning rate starts at 0.01 and uses (34) with horizon 600, stopping at round 50; momentum, weight decay, and batch sizes are as above. Training retains random image augmentation despite the frozen representation parameters.

Selection, inference, and budget interpretation. Validation readouts are saved at adaptation rounds 0, 10, 25, and 50. Within each topology, test comparisons use the fixed prefix and round-50 endpoint without selecting a model from the test scores. The main table reports PATH, CYCLE, RR3, and COMPLETE; topology comparisons are descriptive rather than a preregistered test. The full prefix uses 4,000 local optimizer steps per client; adaptation adds 250 head/router steps. Thus comparison with DFedAvgM400 estimates the change from adding adaptation and referral to that prefix, and does not establish superiority at equal total training effort. The adapted entry, hard terminal, uniform history, and score-weighted history controls use identical final model parameters and matched queries. The graph, split, loss weights, and inference protocol are fixed before each run. Each graph has three fixed training seeds.

Metric and access checks. For a prediction rule ${ \widehat { y } } ( x , s )$ and the test images $\mathcal { T } _ { k }$ of class $k \in [ 8 ]$ , the uniform-entry metric is

$$
\mathrm { B A } ( \widehat { y } ) = \frac { 1 } { 8 } \sum _ { k = 1 } ^ { 8 } \frac { 1 } { 6 | \mathcal { T } _ { k } | } \sum _ { ( x , y ) \in \mathcal { T } _ { k } } \sum _ { s = 1 } ^ { 6 } \mathbf { 1 } \{ \widehat { y } ( x , s ) = y \} .
$$

For the primary rule, $\widehat { y }$ is the argmax of the visited-history probability mixture. Each of $4 { , } 6 5 0$ unique test images contributes six entry records, for 27,900 records per model. These records are paired comparisons of repeated images, not independent test subjects. The deployed router adds the single sample-count offset $\log ( n _ { j } / n _ { i } )$ to the balancedendpoint logit, and the cumulative history weights use the resulting scores without another correction. Its zero threshold and local frozen representations do not establish the additional statistical or routing assumptions in Section 4.