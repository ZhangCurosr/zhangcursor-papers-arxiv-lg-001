# A Hierarchy of Entropy-Shapley Games for Multivariate Predictive Uncertainty

Niklas Koenen Leibniz Institute for Prevention Research and Epidemiology – BIPS University of Bremen, Germany koenen@leibniz-bips.de

Claudia Battistin Simula Research Laboratory Oslo, Norway

Jeriek Van den Abeele Telenor Research & Innovation Fornebu, Norway

Martin Jullum Norwegian Computing Center Oslo, Norway

## Abstract

Modern probabilistic machine learning models increasingly produce multivariate outputs with complex dependence structure, from multi-step time-series forecasts to sample path predictions. Understanding which input features drive the predictive uncertainty is important for risk-aware decisions, model diagnostics, and deciding whether the uncertainty should be mitigated or hedged against. This attribution problem requires a choice of how dependencies between output components are treated. Existing approaches reduce the output to a scalar through aggregation or projection before attribution, thereby obscuring whether features affect marginal uncertainty, dependence structure, or both, while component-wise analyses can miss dependence effects entirely. We close this gap by introducing a hierarchy of three entropy-based Shapley games that make this output-side choice explicit for any ordered multivariate outcome, ranging from per-component marginal entropy to fully joint entropy. The hierarchy isolates a cross-component attribution term that captures how each feature shifts the dependence between output components, a quantity invisible to component-wise methods. We establish a chain-rule decomposition of the joint attribution and characterize the cross-component term through conditional total correlation, providing both closed-form and sample-based estimators. Finally, we demonstrate how the framework captures differences in learned joint structure across probabilistic models from distributional regression to a zero-shot time series foundation model.

## 1 Introduction

Over the last decade, machine learning has shifted focus from predicting low-dimensional targets, e.g., a few class labels or scalar responses, to producing or even generating multivariate outputs with rich dependence structure. For example, time-series models generate multi-step forecasts [1, 2], modern language models output text as token sequences [3, 4], and prediction models in autonomous systems generate paths over subsequent time steps [5, 6]. Such targets $\pmb { Y } = ( Y _ { 1 } , \ldots , Y _ { T } )$ are typically multivariate with a natural ordering, where the components exhibit complex dependencies. In a day-ahead weather forecast, for example, the temperature at noon naturally depends on the preceding morning temperature, and capturing this dependence is part of the modelling task.

The shift toward more nuanced outcomes is reflected in a wave of probabilistic model classes that no longer produce multiple point predictions alone, but a joint predictive distribution or samples over all output components in Y, as multiple realizations are typically plausible and should be viewed in conjunction (e.g., different weather scenarios). These models range from distributional regression [7, 8] and probabilistic forecasting models [1] to foundation models [9–11], which even provide multi-step zero-shot forecasts. In such settings, the realizations’ variability and predictive uncertainty are typically the quantity of interest, since the joint dependence structure of the output makes point predictions an insufficient basis for downstream decisions that must account for the full range of plausible outcomes. Hence, this uncertainty is structured, combining per-component variability with the dependence across components.

As such models are increasingly deployed in safety-critical or sensitive applications, understanding the origin of predictive uncertainty becomes as important as quantifying it. For instance, if a multistep forecast exhibits high uncertainty, a planner needs to know whether this stems from an inherently volatile input feature (which must be hedged against) or a localized lack of historical data, which might be mitigated. For multivariate outputs Y, the attribution of uncertainty inherits this structure: features may drive predictive uncertainty in specific output components or in the dependence between them. In the weather example, this is the difference between asking which inputs drive uncertainty about the temperature at noon, and which drive the joint behavior of the noon and 1 p.m. predictions. Thus, explaining predictive uncertainty is not merely a diagnostic exercise: it helps determine whether uncertainty should be mitigated, monitored, or incorporated into downstream decisions.

Shapley-based feature attribution [12–14] provides a game-theoretical framework which can be used to answer such questions by assigning feature contributions with respect to a suitable value function. Natural choices for uncertainty include predictive variance [15, 16] and entropy [17]. While these are well-defined for scalar outputs, their extension to multivariate outputs is not uniquely defined, as it requires specifying how uncertainty is aggregated or conditioned across components. Even for non-uncertainty attribution, existing methods typically either operate component-wise on each Yt [18–21], ignoring dependencies between output components, or aggregate the multivariate output before attribution [22, 23], thereby obscuring where in the output the effect arises. This limitation is not merely a modeling choice, but a mathematical necessity. Recent work has proven that if a vector-valued attribution method preserves the classical Shapley axioms, it must evaluate each output component independently [24]. Concretely, features that affect only the dependence structure of Y would receive an attribution of zero (see Sec. 5.1 for an example), exposing a structural blind spot in component-wise attribution. This leaves open how to define Shapley-based uncertainty attributions for ordered multivariate outputs in a way that separates marginal uncertainty from dependence among output components.

To close this gap, we make the choice of the value function on the output side explicit and introduce a hierarchy of entropy-based Shapley games that differ in how they treat dependencies across output components. The hierarchy separates different structural aspects of predictive uncertainty, ranging from marginal variability to fully joint dependence. While instantiated in this paper for multi-step forecasting, the framework applies to any ordered multivariate output.

Contributions. (1) To the best of our knowledge, we introduce the first framework for multivariate predictive uncertainty attribution, proposing a hierarchy of entropy-based value functions that consistently extends scalar uncertainty attribution to multivariate outputs by making the outputside marginalization choice explicit. The hierarchy resolves a structural blind spot of standard component-wise attribution, which assigns an attribution of exactly zero to features that drive only cross-component dependence. (2) We establish two propositions linking the levels via a chain-rule decomposition (Prop. 1) and a total-correlation-based characterization of cross-component effects (Prop. 2). (3) We provide closed-form expressions for the multivariate Gaussian case and discuss model-agnostic sample-based estimators for the general case, both validated against analytical baselines. (4) We demonstrate the framework on a range of probabilistic models, from NGBoost [7] and DeepAR [1] to the zero-shot foundation model Chronos-T5 [9].

## 2 Related Work

Feature attribution is a central tool in explainable AI, with well-established post-hoc methods for assigning local [12, 25, 26] or global [27, 28] contributions to features. Various explanation targets have been considered, including the model prediction, its risk, and sensitivity [29]. In this work, we focus on Shapley-based attributions of predictive uncertainty, in particular, the entropy. We organize the related work along two axes: what is being attributed and how the output structure is treated.

Feature Attribution for Predictive Uncertainty. While some works consider feature-based explanations of predictive uncertainty via global risk-based approaches, e.g., conformal predictions [30] or entropy-based PFI/PDP variants [31], we focus on Shapley-based attributions. Within the Shapley framework, the choice of the value function determines which aspect of the predictive uncertainty is attributed: residuals, predictive likelihood, and cross-entropy for risk-based approaches [32, 33], prediction-interval widths in conformal settings [34], local prediction variance under input perturbations [15, 16], or global variance contributions via the connection between Sobol' indices and Shapley values [35]. Closest to our setting are information-theoretic approaches, which directly leverage the domain of quantifying information and uncertainty. SHAP-KL [23] attributes the predictive distribution shift using the KL divergence, while InfoSHAP [17] uses the conditional entropy $H ( Y \mid X )$ as a value function. Common to all of these approaches, however, is that uncertainty is reduced to a single scalar quantity. As a consequence, they cannot resolve how a feature contributes to different structural components of the predictive distribution, a limitation that becomes particularly relevant when the prediction itself is multivariate.

Feature Attribution for Multivariate Outputs. Existing attribution approaches for multivariate outputs $\pmb { Y } = ( Y _ { 1 } , \ldots , Y _ { T } )$ either reduce $\hat { \mathbf { Y } }$ to a scalar before attribution by aggregating over components [22, 23], or projecting onto a single one [36, 37]. A parallel literature on variance-based sensitivity analysis develops Sobol’ indices for vector-valued outputs [38, 39], but provides only global rather than local attributions. Approaches that operate on the predictive distribution face a similar restriction: Tonekaboni et al. [40] attribute the distribution shift via a KL-based decomposition, and ConformaSegment [41] attributes the shift in conformal interval bounds, both on scalar or per-step outputs. In this field, most approaches address sequential inputs for scalar predictions rather than dependencies within the output and are orthogonal to our work [18, 20, 21, 19]. Beyond the time series domain, two recent works address the multivariate-output question more directly. Shapley Chains [42] extend Shapley values to multi-output classification via chains, and Biccari et al. [24] prove a rigidity result stating that any attribution rule satisfying the classical Shapley axioms on vector-valued games must decompose component-wise across outputs. Neither addresses predictive uncertainty nor formalizes local feature contributions to the dependency structure between output components.

To the best of our knowledge, no existing approach attributes predictive uncertainty for multivariate dependent outputs in a way that resolves both marginal feature contributions and contributions to the dependence structure. Our hierarchy of entropy-based Shapley games closes this gap.

## 3 Background

Throughout this paper, lowercase letters denote scalars $( \mathbf { e . g . } , x \in \mathbb { R } )$ , boldface lowercase denotes vectors $( \mathbf { e } . \mathbf { g } . , x \in \mathbb { R } ^ { p } )$ , uppercase denotes a random variable $( \mathbf { e . g . , } X \sim \mathcal { N } ( 0 , 1 ) )$ , and boldface uppercase denotes random vectors or matrices $( \mathbf { e . g . } , X \sim \mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } ) )$ . Calligraphic letters denote a space or the support of the corresponding random variable $( \mathrm { e . g . , } X$ takes values in X). Assuming all relevant probability densities exist, we use the generic notation $p ( \cdot )$ throughout, relying on the arguments to specify the respective random variables $( \mathbf { e . g . } , p ( \pmb { y } | \pmb { x } ) )$ . Additionally, we write $[ p ] : = \{ 1 , \ldots , p \}$ for the set of feature indices, ${ \bar { S } } : = [ p ] \backslash S$ for the complement of $S \subseteq [ p ] , 2 ^ { [ p ] }$ for its power set, and $\begin{array} { r } { \bar { X } _ { S } : = ( X _ { i } ) _ { i \in S } } \end{array}$ for the components of X indexed by S, and $\pmb { X } _ { < t } : = ( \bar { X _ { 1 } } , \ldots , X _ { t - 1 } )$ for the history of output components preceding index t.

## 3.1 Information Theory

To attribute predictive uncertainty to individual features, we require a quantity that captures the full distributional uncertainty of a (possibly multivariate) model prediction. Information theory provides a clear answer: the entropy. The (differential) entropy $H ( { \dot { X } } )$ of a random variable X on $\mathcal { X }$ with density $p$ measures the uncertainty over its realizations and is defined as

$$
H ( X ) = \mathbb { E } _ { X } { \bigl [ } - \log p ( X ) { \bigr ] } = - \int _ { \mathcal { X } } p ( x ) \log p ( x ) d x .\tag{1}
$$

Low entropy indicates a predictable outcome and, conversely, high entropy indicates high uncertainty. For example, a fair coin toss with $\begin{array} { r } { \mathbb { P } [ \mathrm { H e a d s } ] = \frac { 1 } { 2 } } \end{array}$ has entropy $H \overset { \mathbf { \bar { \mathbf { \Delta } } } } { = } - 2 \cdot \frac { 1 } { 2 } \overset { \mathbf { \bar { \mathbf { \Delta } } } } { \log } \frac { 1 } { 2 } = \log 2 \overset { \mathbf { \bar { \mathbf { \Delta } } } } { \approx } 0 . 6 9 3$ nats, reflecting high uncertainty, while a heavily biased coin with $\mathbb { P } [ \mathrm { H e a d s } ] \stackrel { \sim } { = } 0 . 9 9$ has near-zero entropy, since the outcome is almost predetermined.

Closely related is the conditional entropy. Given a specific observation $c \sim C .$ the local conditional entropy $H ( X \mid C \ = \ c )$ measures the residual uncertainty in X for that particular realization. Averaging over all possible values of $C$ yields the (global) conditional entropy $H ( X \mid C ) = \mathbb { E } _ { \tilde { C } } [ H ( X \mid C = \tilde { C } ) ]$ , which captures the expected residual uncertainty in $X$ after knowing $C .$ Returning to the coin toss example, suppose a friend $C$ selects the fair or the biased coin with equal probability, and only he knows which coin is tossed. If he reveals that he used the fair coin, then we are more uncertain with $H ( X | C = \mathrm { { f a i r } ) = \log 2 \approx 0 . 6 9 3 }$ nats, while $H ( X | C = { \mathrm { b i a s e d } } ) \approx 0 . 0 6$ in the biased case. The global conditional entropy $H ( X | C ) \approx 0 . 3 7 5$ averages over these two values, reflecting how much overall uncertainty about the outcome remains once we know which coin is tossed. The reduction in uncertainty about X from observing C is also known as the mutual information $I ( X ; C ) : = H ( X ) - H ( X | \mathbf { \bar { \begin{array} { r } { } } \end{array} } )$ . It quantifies the dependence of both variables and satisfies $I ( X ; C ) \overset { \cdot } { \geq } 0$ with equality iff X and C are independent.

These definitions extend naturally to random vectors: for X on $\mathcal { X } \subseteq \mathbb { R } ^ { p }$ , the joint entropy $H ( X )$ the conditional entropy $H ( X | \dot { C } )$ and the mutual information $I ( X ; C ) = { \overset { \cdot } { H } } ( X ) - { \overset { \cdot } { H } } ( X \mid C )$ are defined analogously with the integral taken over $\mathcal { X }$ using the joint or conditional distribution, respectively. The dependence among the components of a random vector X is measured by the total correlation $\begin{array} { r } { \mathrm { T C } ( \boldsymbol { X } ) \dot { : } = \sum _ { i = 1 } ^ { p } H ( \dot { \boldsymbol { X _ { i } } } ) - H ( \dot { \boldsymbol { X } _ { } } ) } \end{array}$ . Like mutual information, $\mathrm { T C } (  { \boldsymbol { X } } ) \geq 0$ with equality iff the components of $\boldsymbol { X }$ are mutually independent, and it reduces to $I ( X _ { 1 } ; X _ { 2 } )$ for $p = 2 . \mathrm { ~ A ~ }$ key identity linking joint and conditional entropies is the chain rule of entropy,

$$
H ( \pmb { X } ) = \sum _ { i = 1 } ^ { p } H ( X _ { i } | X _ { 1 } , \ldots , X _ { i - 1 } ) = \sum _ { i = 1 } ^ { p } H ( X _ { i } | \pmb { X } _ { < i } ) ,\tag{2}
$$

which decomposes joint uncertainty into sequential increments. Each summand captures the residual uncertainty in $X _ { i }$ given all preceding components. The identity extends to both local and global conditional entropies, e.g., $\begin{array} { r } { \mathbf { \dot { H } } ( \mathbf { X } | \bar { C } ) = \sum _ { i = 1 } ^ { p } H ( X _ { i } | X _ { 1 } , \dots , X _ { i - 1 } , C ) } \end{array}$ . For more details on information-theoretic background, we refer to App. A.1.

## 3.2 Shapley Values

Shapley values [43] provide a fair attribution of a total "payoff" among players in a cooperative game. For feature attribution in machine learning [12], the players are typically the input features $X _ { 1 } , \ldots , X _ { p }$ and a value function $v \colon 2 ^ { [ p ] } \times \mathcal { X } \to \mathbb { R }$ assigns each coalition $S \subseteq [ p ]$ the corresponding real-valued payoff for a specific instance $\mathbf x \in \mathcal X ^ { 1 }$ . The Shapley value of feature $j$ is its average marginal contribution across all possible coalitions:

$$
\phi _ { v } ( j , \pmb { x } ) = \sum _ { S \subseteq [ p ] \setminus \{ j \} } \frac { | S | ! ( p - | S | - 1 ) ! } { p ! } \Big [ v ( S \cup \{ j \} , \pmb { x } ) - v ( S , \pmb { x } ) \Big ] .\tag{3}
$$

For any value function $v ,$ this is the unique attribution satisfying efficiency, symmetry, linearity, and the null player axioms [12]

A key feature of the Shapley framework is that different value functions v yield different attributions, each answering another question. For instance, standard SHAP [12] defines the value function for a predictive model $f : \mathcal { X }  \mathcal { Y }$ as $v _ { 0 } ( S , \pmb { x } ) = \mathbb { E } [ f ( \pmb { X } ) | \pmb { X } = ( \pmb { x } _ { S } , \pmb { X } _ { \bar { S } } ) ]$ . The resulting Shapley values decompose the prediction into feature-wise contributions relative to the average prediction by the efficiency property, i.e. $\begin{array} { r } { , f ( \pmb { x } ) - \mathbb { E } [ f ( \pmb { X } ) ] = \sum _ { i = 1 } ^ { p } \phi _ { v _ { 0 } } ( j , \pmb { x } ) } \end{array}$ . Other choices include (local and global) risk-based value functions defined via expected loss [32, 27], sensitivity-based approaches [35], and information-theoretic formulations based on divergences such as the KL divergence [23]. However closely related to our approach is InfoSHAP [17], which moves from explaining predictions to explaining predictive uncertainty by using the conditional entropy as the value function,

$$
v _ { H } ( S , \pmb { x } ) = \mathbb { E } _ { \pmb { X } _ { \bar { S } } } \big [ H \big ( Y \ \big | \ \pmb { X } = ( \pmb { x } _ { S } , \pmb { X } _ { \bar { S } } ) \big ) \big ] ,\tag{4}
$$

so that the resulting Shapley values decompose how much each feature contributes to the predictive uncertainty about the outcome $Y$ . InfoSHAP is formulated and evaluated for scalar outcomes $Y \in \mathbb { R }$ and has not been extended to multivariate targets Y including their dependence structure.

## 4 Multivariate Entropy-Shapley Attribution

When explaining the predictive uncertainty of a model with multivariate output $\pmb { Y } = ( Y _ { 1 } , \ldots , Y _ { T } )$ a fundamental choice arises: should we explain the uncertainty of each component separately, or of the full vector jointly? The first ignores dependencies between components, whereas the second captures these dependencies but loses information about where in the output the uncertainty is located. Neither alone tells the full story, which motivates our hierarchy of three entropy-based Shapley games that we formalize below, each defining a value function based on the conditional entropy but evaluated at a different resolution of the output vector. While the framework applies to any multivariate output with a natural order of components, we instantiate it on multi-step time series forecasting in Sec. 5.

## 4.1 Hierarchy of Entropy Games

All three game levels share the same player set $[ p ]$ and extend the InfoSHAP value function Eq. (4) to multivariate outputs, but differ in which aspect of the joint distribution $p ( \pmb { y } \vert \pmb { x } )$ they target.

Level 1 (Marginal). The first game applies InfoSHAP [17] to each $Y _ { t }$ separately to explain the marginal uncertainty at each output component t, while ignoring the dependencies within the outcome:

$$
v _ { H } ^ { ( t ) } ( S , \pmb { x } ) : = \mathbb { E } _ { \pmb { X } _ { \bar { S } } } \big [ H \big ( Y _ { t } \bigm | X = ( \pmb { x } _ { S } , \pmb { X } _ { \bar { S } } ) \big ) \big ] , \qquad t = 1 , \dots , T .\tag{5}
$$

By decomposing the difference between the local and global conditional entropies $H ( Y _ { t } \mid { \pmb x } ) ~ -$ $\dot { H } ( Y _ { t } | X )$ , these Shapley values informally quantify how much each feature contributes to the predictive uncertainty of component t, compared to the population average.

Level 2 (Sequential). The second game explains the incremental uncertainty at component t, given all preceding components:

$$
v _ { H } ^ { ( t | < t ) } ( S , x ) : = \mathbb { E } _ { X _ { S } } \big [ H \big ( Y _ { t } \big | Y _ { < t } , X = ( x _ { S } , X _ { \bar { S } } ) \big ) \big ] , \qquad t = 1 , \dots , T .\tag{6}
$$

Each game corresponds to one term in the chain rule decomposition in Eq. (2), measuring the residual uncertainty in $Y _ { t }$ that remains after conditioning on earlier outcome components and the available features. By decomposing the difference between the sequential local and global conditional entropies, $H ( Y _ { t } | Y _ { < t } , \pmb { x } ) - \mathrm { \hat { \cal H } } ( Y _ { t } | Y _ { < t } , \pmb { X } )$ , the associated Shapley values informally quantify how much each feature contributes to the additional predictive uncertainty at $t ,$ beyond what the preceding components already reveal, compared to the population average.

Level 3 (Joint). The third game explains the joint uncertainty of the full output vector:

$$
v _ { H } ^ { \mathrm { j o i n t } } ( S , \pmb { x } ) : = \mathbb { E } _ { \pmb { X } _ { \bar { S } } } \big [ H \big ( \pmb { Y } \bigm | \pmb { X } = ( \pmb { x } _ { S } , \pmb { X } _ { \bar { S } } ) \big ) \big ] .\tag{7}
$$

This captures all marginal and cross-component effects simultaneously, but no longer reveals where in the output the uncertainty is located. By decomposing the difference between the full joint local and global conditional entropies, $H ( Y | { \dot { x } } ) - H ( { \dot { Y } } | X )$ , the associated Shapley values informally quantify how much each feature contributes to the predictive uncertainty of the entire forecast compared to the population average.

Substituting the value functions into Eq. (3) yields the Level 1-3 Shapley values: $\phi ^ { ( t ) } ( j , \pmb { x } )$ $\phi ^ { ( t | < t ) } ( j , \pmb { x } )$ , and $\phi ^ { \mathrm { j o i n t } } ( j , \pmb { x } )$ . As noted above, these decompose the respective local-global conditional entropy gaps (see also Corollary 3). The specific global conditional entropy contrasts arise because the value functions in Eqs. (5)–(7) rely on marginalizations of the global conditional entropy, rather than a subset-conditional approach (which for Level 3 would correspond to $H ( \boldsymbol { Y } | \boldsymbol { X } _ { S } = \dot { \boldsymbol { x } _ { S } } )$ as value function, and decomposing $H ( \pmb { Y } | \pmb { x } ) - H ( \pmb { Y } ) )$ . As Watson et al. [17] note, however, this demands intractable coalition densities $p ( \pmb { y } | \pmb { x } _ { S } )$ , while our formulation only requires the full conditional $p ( \pmb { y } | \pmb { x } )$ . Consequently, in our games, $\phi ( j , { \pmb x } ) > 0$ indicates that feature j drives higher predictive uncertainty than the population average, while $\phi ( j , { \pmb x } ) < 0$ implies a reduction.

The resulting Shapley values for the three levels are not independent constructions, as they are linked through the chain rule of entropy Eq. (2). The proof can be found in App. B.

Proposition 1 (Chain-rule linkage). The Level 3 Shapley values decompose additively into the Level 2 Shapley values across components, i.e., for all $j \in [ p ]$ it holds that

$$
\phi ^ { j o i n t } ( j , \pmb { x } ) = \sum _ { t = 1 } ^ { T } \phi ^ { ( t | < t ) } ( j , \pmb { x } ) .\tag{8}
$$

Moreover, if the output components $Y _ { 1 } , \dots , Y _ { T }$ are conditionally independent given $X _ { i }$ then $\phi ^ { ( t | < t ) } ( j , \dot { \pmb { x } } ) = \phi ^ { ( t ) } \dot { ( j , \pmb { x } ) }$ for all t, and Level 3 reduces to the sum of Level 1 values.

This implies that Level 2 Shapley values provide a component-wise decomposition of the joint attribution. If a feature drives overall forecast uncertainty (Level 3), Level 2 reveals at which components this effect materializes and how strongly.

While Prop. 1 connects Level 2 and Level 3, the relationship between Level 1 and Level 3 is controlled by the dependency structure of the outcomes. To formalize this, we introduce the total correlation game, defined by the following value function:

$$
v _ { \mathrm { T C } } ( S , \pmb { x } ) : = \mathbb { E } _ { \pmb { X } _ { \bar { S } } } [ \mathrm { T C } ( \pmb { Y } \mid \pmb { X } ) \mid \pmb { X } = ( \pmb { x } _ { S } , \pmb { X } _ { \bar { S } } ) ] .\tag{9}
$$

The associated Shapley values $\phi ^ { \mathrm { T C } } ( j , \pmb { x } )$ constitute the cross-component attributions by decomposing the deviation of the local total correlation from its population average $\mathrm { T C } ( Y | { \pmb x } ) - \mathrm { T \bar { C } } ( Y | \dot { X } )$ (see App. A.1 for further details on TC). The below Proposition highlights how the TC game connects the Level 1 and Level 3 games:

Proposition 2 (Cross-component decomposition). The Shapley values of the TC game isolate the contribution of feature j from the dependency structure between output components, i.e., for all $j \in [ p ]$ it holds that $\begin{array} { r } { \phi ^ { \mathrm { T C } } ( j , \pmb { x } ) = \sum _ { t = 1 } ^ { T } \phi ^ { ( t ) } ( j , \pmb { x } ) - \phi ^ { j o i n t } ( j , \pmb { x } ) } \end{array}$

The total correlation measures the overall statistical redundancy between output components. Prop. 2 thus gives the cross-component attributions a precise interpretation: they decompose how much each feature shifts the inter-component dependence relative to a typical instance. Crucially, features that only affect the dependence structure without changing any marginal distribution receive $\phi ^ { ( t ) } ( j , \pmb { x } ) = 0$ for all t, making them entirely invisible to Level 1 (see Sec. 5.1 for an example).

## 4.2 Estimating the Value Functions

Evaluating the value functions (Eqs. 5–7) in practice requires decomposing them into two operations: an outer expectation over the out-of-coalition features $X _ { \bar { S } } ,$ and an inner conditional entropy evaluated at a specific completed input x. The outer expectation represents the standard Shapley imputation step, typically approximated via Monte Carlo integration by drawing K background samples $\pmb { x } _ { \bar { S } } ^ { ( k ) }$ to form K completed inputs $\tilde { \pmb { x } } ^ { ( k ) } = ( \pmb { x } _ { S } , \pmb { x } _ { \bar { \varsigma } } ^ { ( k ) } ) , k = 1 , \dots , K$ . Our hierarchy is agnostic to how these are drawn; any standard imputation method $( \mathrm { e . g . }$ , baseline, marginal, or conditional [44, 45, 14]) can be used. Consequently, the core technical challenge specific to our method reduces to evaluating the inner entropies $( \mathbf { e } . \mathbf { g } . , H ( \pmb { Y } | \pmb { X } = \tilde { \pmb { x } } ^ { ( k ) } ) )$ for a fixed completed input $\tilde { \mathbf { \pmb { x } } } ^ { ( k ) }$ . How we estimate these terms depends on how the model provides access to $p ( \pmb { y } | \tilde { \pmb { x } } )$ . We consider three cases: closedform joint densities, illustrated by multivariate Gaussian outputs (Sec. 4.2.1); factorized one-step conditionals, as in autoregressive models (Sec. 4.2.2); and purely sample-based outputs (Sec. 4.2.3).

## 4.2.1 Multivariate Gaussian Outputs

A common class of probabilistic models directly learns distributional parameters, $\mathrm { e . g . } , \mu ( { \pmb x } )$ and $\pmb { \Sigma } ( \tilde { \pmb { x } } )$ for a multivariate Gaussian outcome $\boldsymbol { Y } | \tilde { \boldsymbol { x } } \tilde { \mathcal { \sim } } \mathcal { N } ( \boldsymbol { \mu } ( \tilde { \boldsymbol { x } } ) , \boldsymbol { \Sigma } ( \tilde { \boldsymbol { x } } ) )$ . In this case, the value function at each level of the hierarchy (Sec. 4.1) admits a closed-form expression that depends solely on the covariance matrix Σ(x). For more details on the entropy of multivariate Gaussians, we refer to [46].

For Level 1, the entropy of each component $Y _ { t }$ depends only on the corresponding diagonal entry:

$$
H ( Y _ { t } | \tilde { \mathbf { x } } ) = \textstyle { \frac { 1 } { 2 } } \log \left( 2 \pi e \Sigma _ { t t } ( \tilde { \mathbf { x } } ) \right) , \qquad t = 1 , \dots , T .\tag{10}
$$

For Level 2, the conditional distribution $Y _ { t } \mid Y _ { < t } , { \tilde { x } }$ is again Gaussian, with variance given by the Schur complement of the leading $( t { - } 1 ) \times ( t { - } 1 )$ submatrix in the leading $t \times t$ submatrix of $\Sigma ( \tilde { { \boldsymbol { x } } } )$

$$
\begin{array} { r } { H \big ( Y _ { t } | Y _ { < t } , \tilde { x } \big ) = \frac { 1 } { 2 } \log \big ( 2 \pi e \sigma _ { t | < t } ^ { 2 } ( \tilde { x } ) \big ) , \quad \mathrm { w h e r e } \quad \sigma _ { t | < t } ^ { 2 } = \Sigma _ { t t } - \Sigma _ { t , < t } \Sigma _ { < t , < t } ^ { - 1 } \Sigma _ { < t , t } . } \end{array}\tag{11}
$$

For Level 3, the entropy follows from the determinant of the full covariance matrix $\pmb { \Sigma } ( \tilde { \pmb { x } } )$ •

$$
\begin{array} { r } { H ( \pmb { Y } | \tilde { \pmb { x } } ) = \frac { 1 } { 2 } \log \left( ( 2 \pi e ) ^ { T } \operatorname* { d e t } \pmb { \Sigma } ( \tilde { \pmb { x } } ) \right) . } \end{array}\tag{12}
$$

These three formulas show that the value function is entirely independent of the mean prediction $\pmb { \mu } ( \pmb { x } )$ , i.e., features that only affect the mean receive zero entropy attribution at every level. Each level operates on a different part of the covariance matrix (see Fig. D.1) and thus reflects how each level treats the output dependencies (Sec. 4.1). The same closed-form computations also extend to other distributional families with tractable marginal, conditional, and joint entropies, e.g., to multivariate Student-t or, more generally, elliptical distributions [47, 48].

## 4.2.2 Factorized Conditional Densities

A second class of probabilistic models, including autoregressive forecasting models such as DeepAR [1], factorizes the predictive distribution along the natural ordering of $\mathbf { \bar { Y } } = ( Y _ { 1 } , \ldots , Y _ { T } )$ as $\begin{array} { r } { p ( \pmb { y } \vert \tilde { \pmb { x } } ) = \prod _ { t = 1 } ^ { T } p ( y _ { t } \vert \pmb { y } _ { < t } , \tilde { \pmb { x } } ) } \end{array}$ by learning the one-step conditionals $p ( y _ { t } \mid \pmb { y } _ { < t } , \tilde { \pmb { x } } )$ , typically as a parametric distribution (e.g., Gaussian, Student-t or log-normal). The factorization yields Levels 2 and 3 directly, while Level 1 requires additional estimation. The Level 2 entropies $H ( Y _ { t } | \mathbf { Y } _ { < t } , \tilde { { \pmb x } } )$ are obtained by averaging the closed-form entropy of the one-step conditional $p ( y _ { t } \mid \pmb { y } _ { < t } , \tilde { \pmb { x } } )$ over samples of $\mathbf { Y } _ { < t }$ drawn along the factorization, and the Level 3 entropy follows by the chain rule (Prop. 1). The Level 1 marginal entropies $H ( Y _ { t } \mid \tilde { { \boldsymbol { x } } } )$ , by contrast, have no closed form, since the marginal $p ( y _ { t } \mid \tilde { \mathbf { \Lambda } } )$ is a mixture of one-step conditionals. They can be estimated from samples through marginalization, e.g., by Monte-Carlo averaging the conditional density $p ( y _ { t } \mid \pmb { y } _ { < t } , \tilde { \pmb { x } } )$ over samples of $Y _ { < t } ,$ or by applying a one-dimensional nonparametric entropy estimator to the marginal samples of $Y _ { t } ,$ which avoids high-dimensional density estimation.

## 4.2.3 Sample-based Entropy Estimation

When $p ( \pmb { y } \vert \pmb { x } )$ is unavailable, we estimate the local entropies from simulated joint trajectories. For each completed input x arising in outer expectation step (Sec. 4.2), we draw trajectories $\pmb { y } ^ { ( 1 ) } , \ldots , \pmb { y } ^ { ( N ) } \sim p ( \pmb { Y } | \tilde { \pmb { x } } )$ and evaluate the required entropy terms using one of several approaches.

Nonparametric kNN estimation. A fully nonparametric alternative uses nearest-neighbour distances [49]. While this efficiently yields Level 1 marginals, Level 2 sequential terms require computing differences $H ( { \pmb Y } _ { < t } \mid { \pmb X } = \dot { \tilde { \pmb x } } ) - H ( { \pmb Y } _ { < t } \mid { \pmb X } = \tilde { \pmb x } )$ , from which Level 3 follows. Although assumptionlight, computing potentially high-dimensional joint entropies for larger horizons at Level 2 and 3 suffers from the curse of dimensionality, demanding a large sample size N to maintain accuracy.

Parametric Gaussian estimation. Conversely, assuming a multivariate Gaussian predictive distribution allows efficient entropy evaluation by applying the closed-form expressions (Sec. 4.2.1) directly to the sample covariance matrix $\hat { \Sigma } ( \tilde { { \boldsymbol { x } } } )$ estimated from the trajectories. However, while computationally cheap, this estimator remains inherently biased for predictive distributions exhibiting non-Gaussian characteristics such as skewness or heavy tails.

Semiparametric Gaussian copula. To handle non-Gaussian marginals while retaining computational efficiency, we employ a Gaussian copula. For each completed input ${ \tilde { \mathbf { x } } } ,$ we estimate one-dimensional marginal densities $\hat { f } _ { t }$ and CDFs $\hat { F } _ { t }$ by KDE, and use the fitted marginals to compute the Level 1 entropies $\widehat { H } ( Y _ { t } | \tilde { { \boldsymbol { x } } } )$ . The samples are then transformed to latent Gaussian variables via $z _ { t } ^ { ( i ) } =$ $\Phi ^ { - 1 } ( \hat { F } _ { t } ( y _ { t } ^ { ( i ) } | \tilde { \mathbf { x } } ) )$ ), from which we estimate the empirical correlation matrix $\hat { R } ( \tilde { \boldsymbol { x } } )$ . Level 2 sequential entropies are computed as $\begin{array} { r } { \widehat { H } ( Y _ { t } | Y _ { < t } , \tilde { \pmb { x } } ) = \widehat { H } ( Y _ { t } | \tilde { \pmb { x } } ) + \frac { 1 } { 2 } \log ( \widehat { \sigma } _ { t | < t , z } ^ { 2 } ) } \end{array}$ , where $\hat { \sigma } _ { t | < t , z } ^ { 2 }$ is the Schur complement of $\hat { R } ,$ and Level 3 follows by Prop. 1. The estimator thus allows flexible marginals while retaining stable and tractable Gaussian-copula dependence; full formulae are given in App. C.

## 5 Experiments

We evaluate the proposed hierarchy in three steps, moving from controlled closed-form settings to sample-based estimation and model comparison. We first use a synthetic Gaussian DGP (Sec. 5.1) and a real distributional-regression model (Sec. 5.2), where the entropy terms are available analytically. Finally, we apply our hierarchy to the UCI Electricity dataset [50], comparing sample-based crosscomponent share on DeepAR [1] and Chronos [9]. The code for reproduction is available on GitHub².

![](images/e9e2053dbc5188b5a13aa0726aa6bd154e74b57b267b7336a250b617a4ce2b48.jpg)  
Figure 1: Entropy-Shapley hierarchy on the synthetic Gaussian DGP for an instance with high correlation. Bars connect equal quantities across panels, visualizing Prop. 1 and Prop. 2.

## 5.1 Proof of Concept on a Synthetic Gaussian Data-Generating Process

We demonstrate the hierarchy with a controlled data-generating process (DGP) where each feature plays a known role in the predictive distribution. The forecast is multivariate Gaussian, i.e., $Y | \pmb { x } \sim$ $\mathbf { \bar { \mathcal { N } } } ( \pmb { \mu } ( \pmb { x } ) , \pmb { \Sigma } ( \pmb { x } ) )$ , with $T = 4$ output components and $p = 4$ input features. Feature $x _ { 1 }$ controls the mean forecast, $x _ { 2 }$ the marginal variance with a time-increasing effect, $x _ { 3 }$ both variance and correlation, and $x _ { 4 }$ exclusively controls correlation without affecting any marginals. The DGP itself serves as the predictive model to be explained. Full details can be found in App. D.1.

For a single instance x with a strong positive correlation $( \rho \approx 0 . 9 2 , { \mathrm { F i g } } . 1 )$ , our hierarchy recovers the ground-truth feature roles: features $x _ { 2 }$ and $x _ { 3 }$ dominate Level 1, with the time-increasing effect of $x _ { 2 }$ visible across $t = 1 , \ldots , 4 .$ while $x _ { 1 }$ receives zero attribution at every level, confirming that our hierarchy isolates predictive uncertainty from the mean prediction. Most importantly, $x _ { 4 }$ shows the blind spot of component-wise attributions: it leaves all marginals invariant and therefore receives zero attribution at Level 1, yet is the dominant negative contribution at Level 2 and in the cross-component term. Across panels, the polygons connect equal quantities and provide a visual illustration of the propositions: the chain-rule linkage (Prop. 1) and the cross-component decomposition (Prop. 2).

## 5.2 Distributional Regression on Bike Sharing

We now move from controlled ground truth to a real distributional regression problem. We predict bike rental counts on the UCI Bike Sharing dataset [51], aggregated into $T = 8$ two-hour blocks between 06:00 and 22:00, from $p = 8$ daily features (three weather snapshots at 06:00, seven-day mean temperature, previous-day rentals, and the calendar features), using NGBoost [7] with the multivariate normal as the target distributional family. This yields a feature-dependent mean and full covariance, and our hierarchy applies in closed form (Sec. 4.2.1). Further details, comparison to standard SHAP applied to the predictive mean, and a global analysis can be found in App. D.2.

Figure 2 illustrates the attributions of our hierarchy for two distinct instances³: a workday with a typical bimodal commute profile (top row, low uncertainty), and a weekend (bottom row, high uncertainty). Each level reveals a different aspect of how features drive the predictive uncertainty (left panel). At Level 1, the low morning humidity reduces the predicted uncertainty across nearly all daytime blocks for the workday instance. Level 2 then shows the sequential view, where the same feature contributes less, because part of its information is already contained in earlier time blocks. In the right panel, Level 3 illustrates how strongly each feature drives the uncertainty of the entire forecast. On the weekend, Temp 6am and Prev Count are the most contributing features, both reducing the overall forecast uncertainty. The cross-component term then makes explicit whether a feature increases or decreases the dependence between rental blocks. For Temp 6am on the weekend instance, the joint contribution is negative, but the cross contribution is highly positive, showing that the morning temperature reduces the overall uncertainty while at the same time strongly increasing the dependence between the rental blocks.

![](images/0a25792a58497fe382a1c510d01e33d10752aa9be3ab40b8ef13286bc984fede.jpg)

![](images/9ec28a8b77eeb2e967da9e2406bf24d0e2df389969e76f67e8d46e8a20ffbe33.jpg)  
Figure 2: Entropy-Shapley hierarchy on NGBoost for two Bike Sharing instances (top: workday; bottom: weekend). Left: predicted mean and uncertainty bands. Center: Level 1 and Level 2 attributions (blue: < 0, red: > 0). Right: Level 3 and cross-component (hatched) attributions.

## 5.3 Cross-component Diagnostic across Forecaster Classes

While Sec. 5.2 demonstrates the hierarchy on a model with closed-form predictive distribution, we now turn to forecasters whose distribution is only available through joint samples (Sec. 4.2.3). On UCI Electricity [50], we compare two forecasters of different classes: DeepAR [1] with a Gaussian likelihood (autoregressive parametric) and Chronos [9] (zeroshot foundation model). For each model and 100 forecast origins, we compute the cross-component share as

$$
\frac { \sum _ { j } \bigl | \phi _ { j } ^ { \mathrm { T C } } ( { \pmb x } ) \bigr | } { \sum _ { j } \bigl | \phi _ { j } ^ { \mathrm { j o i n t } } ( { \pmb x } ) \bigr | + \sum _ { j } \bigl | \phi _ { j } ^ { \mathrm { T C } } ( { \pmb x } ) \bigr | } .\tag{13}
$$

![](images/7800fa08c69b72e65191b2b8cf848adf9e88d215874fae806fd993d81e57ac09.jpg)

This quantity measures the relative magnitude of the crosscomponent attribution mass within the combined attribution mass of the joint and cross-component terms, i.e., higher values indicate a larger relative contribution of cross-component dependence, whereas lower values indicate a larger relative contribution of the joint term. We use the Gaussian copula and kNN estimators (Sec. 4.2.3) as sample-based approaches for both models and, additionally, the approaches

Figure 3: Cross-component share on 100 forecast origins for DeepAR and Chronos on the electricity dataset, estimated via the parametric (DeepAR only), Gaussian copula, and kNN.

from Sec. 4.2.2 for DeepAR using marginalization for Level 1. For DeepAR, we also conducted a more general validation of all sample-based estimators across likelihoods in App. D.3.

Figure 3 shows a clear difference between the model classes as Chronos exhibits much more crosscomponent share than DeepAR, even though the forecast performance is comparably good (MAE 0.127 for DeepAR vs. 0.13 for Chronos). Within each class, the two sample-based estimators on DeepAR closely match the parametric reference, likely because DeepAR's per-step Gaussian structure produces a near-Gaussian joint distribution that both estimators handle well. On Chronos, however, the copula and kNN estimations diverge substantially: Chronos models more complex and non-parametric distributions [9], so the Gaussian copula may be misspecified for the foundation model, while kNN can itself be biased on such high-complexity outputs (see Sec. 4.2.3). Regardless of how this gap decomposes, the cross-component share clearly distinguishes the two model classes, providing a practical diagnostic of how much joint structure a forecaster has actually learned.

## 6 Discussion and Limitations

We have introduced a hierarchy of entropy-based Shapley games for explaining multivariate predictive distributions while accounting for their dependence structure, resolving a structural blind spot of standard component-wise attribution. The three levels of output resolution are linked through the chain-rule decomposition (Prop. 1) and the total-correlation characterization (Prop. 2), allowing different aspects of predictive uncertainty to be attributed in a coherent way. While we have studied both the estimation of the levels (Sec. 4.2) and their applicability (Sec. 5), some open issues remain: (1) The hierarchy attributes the predictive uncertainty regardless of whether it is aleatoric or epistemic in origin [52], i.e., the two can be attributed separately if the predictive distribution itself separates them, e.g., through MC dropout [53] as in InfoSHAP [17]. (2) Our diagnostic is only informative on models that can, in principle, represent inter-component dependence, as under conditional independence, the hierarchy collapses to a component-wise view. Moreover, the uncertainty must be driven by features rather than post-hoc trajectory sampling schemes [54], where any apparent coupling between components stems from the sampling procedure rather than the model itself. (3) Sample-based estimation inherits known biases of copula and nearest-neighbour estimators on highly non-Gaussian distributions [55]; as seen for Chronos (Sec. 5.3), this can produce estimator disagreement that is itself diagnostically useful. Future work could explore more flexible dependence structures, such as computationally feasible vine copulae, to capture stronger tail dependencies [56, 57]. (4) Finally, our framework inherits two standard Shapley challenges: theoretically, the choice of background imputation method (Sec. 4.2) alters the interpretation of the resulting attributions [17]; computationally, exact evaluation scales exponentially with the number of features and the forecast horizon T, requiring established sampling-based approximations in higher dimensions [14].

## Acknowledgements

This work was conducted as part of the “Uncertainty-aware feature attribution for temporal predictive models" action, which has received funding from the European Union, via the oc3-2025-TES-01 issued and implemented by the ENFIELD project, under the grant agreement No. 101120657. Additionally, Niklas Koenen has been funded by the German Research Foundation (DFG) as part of the Research Unit “Lifespan AI: From Longitudinal Data to Lifespan Inference in Health" (DFG FOR 5347), Grant 459360854, and by the Emmy Noether Grant 437611051. Finally, part of Martin's work has also been funded by the Integreat Center of Excellence – a centre of excellence funded by Research Council of Norway, project number 332645.

## References

[1] D. Salinas, V. Flunkert, J. Gasthaus, et al., DeepAR: Probabilistic forecasting with autoregressive recurrent networks, International Journal of Forecasting 36 (2020) 1181–1191. doi:10.1016/j. ijforecast .2019. 07.001.

[2] H. Zhou, S. Zhang, J. Peng, et al., Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, Proceedings of the AAAI Conference on Artificial Intelligence 35 (2021) 11106–11115. doi:10.1609/aaai.v35i12.17325.

[3] T. B. Brown, B. Mann, N. Ryder, et al., Language models are few-shot learners, in: NIPS'20: Proceedings of the 34th International Conference on Neural Information Processing Systems, Curran Associates Inc., 2020,pp. 1877–1901.URL: https://dl.acm.org/doi/10.5555/3495724.3495883.

[4] H. Touvron, T. Lavril, G. Izacard, et al., LLaMA: Open and Efficient Foundation Language Models, arXiv (2023).doi:10.48550/arXiv.2302.13971.arXiv:2302.13971.

[5] T. Salzmann, B. Ivanovic, P. Chakravarty, et al., Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data, in: Computer Vision – ECCV 2020, Springer, Cham, Switzerland, 2020, pp. 683-700.doi:10.1007/978-3-030-58523-5\_40.

[6] B. Varadarajan, A. Hefny, A. Srivastava, et al., MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction, in: 2022 International Conference on Robotics and Automation (ICRA), IEEE, 2022, pp. 23–27. doi:10.1109/ICRA46639.2022.9812107.

[7] T. Duan, A. Anand, D. Y. Ding, et al., NGBoost: Natural Gradient Boosting for Probabilistic Prediction, in: International Conference on Machine Learning, PMLR, 2020, pp. 2690–2700. URL: https:// proceedings.mlr.press/v119/duan20a.html.

[8] S. Rasp, S. Lerch, Neural Networks for Postprocessing Ensemble Weather Forecasts, Mon. Weather Rev. 146 (2018) 3885–3900.doi:10.1175/MWR-D-18-0187.1.

[9] A. F. Ansari, L. Stella, A. C. Turkmen, et al., Chronos: Learning the Language of Time Series, Transactions on Machine Learning Research (2024). URL: https://openreview.net/forum?id=gerNCVqqtR.

[10] G. Woo, C. Liu, A. Kumar, et al., Unified training of universal time series forecasting transformers, in: ICML'24: Proceedings of the 41st International Conference on Machine Learning, volume 235, JMLR, 2024,pp.53140–53164.URL: https://d1.acm.org/doi/10.5555/3692070.3694248.

[11] A. Das, W. Kong, R. Sen, et al., A decoder-only foundation model for time-series forecasting, in: International Conference on Machine Learning, PMLR, 2024, pp. 10148–10167. URL: https: //proceedings.mlr.press/v235/das24c.html.

[12] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: NIPS'17: Proceedings of the 31st International Conference on Neural Information Processing Systems, Curran Associates Inc., 2017,pp.4768–4777.URL: https://d1.acm.org/doi/10.5555/3295222.3295230.

[13] M. Sundararajan, A. Najmi, The Many Shapley Values for Model Explanation, in: International Conference on Machine Learning, PMLR, 2020, pp. 9269–9278. URL: https://proceedings .mlr.press/v119/ sundararajan20b.html.

[14] H. Chen, I. C. Covert, S. M. Lundberg, et al., Algorithms to estimate Shapley value feature attributions, Nat. Mach. Intell. 5 (2023) 590–601. doi:10.1038/s42256-023-00657-x.

[15] M. Gajewski, M. Morzy, A. Karczmarz, et al., VARSHAP: Addressing Global Dependency Problems in Explainable AI with Variance-Based Local Feature Attribution, 2025. URL: https://arxiv. org/abs/ 2506.07229.arXiv:2506.07229.

[16] T. Chiaburu, F. Bießmann, F. Haußer, Uncertainty Propagation in XAI: A Comparison of Analytical and Empirical Estimators, in: Explainable Artificial Intelligence. xAI 2025. Communications in Computer and Information Science, vol 2580, Springer, Cham, Switzerland, 2025, pp. 390–411. doi:10.1007/ 978-3-032-08333-3\_18.

[17] D. Watson, J. O'Hara, N. Tax, et al., Explaining predictive uncertainty with information theoretic shapley values, Advances in Neural Information Processing Systems 36 (2023) 7330–7350. URL: https : //openreview.net/forum?id=6rabAZhCRS.

[18] J. Bento, P. Saleiro, A. F. Cruz, et al., TimeSHAP: Explaining Recurrent Models through Sequence Perturbations, in: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), ACM, 2021, pp. 2565–2573. doi:10.1145/3447548.3467166.

[19] M. Franco De La Peña, Á. L. P. Gómez, L. Fernández Maimó, ShaTS: a Shapley-based explainability method for time series artificial intelligence models, Future Gener. Comput. Syst. 176 (2026) 108178. doi:10.1016/j.future.2025.108178.

[20] A. Nayebi, S. Tipirneni, C. K. Reddy, et al., WindowSHAP: An efficient framework for explaining time-series classifiers based on Shapley values, J. Biomed. Inf. 144 (2023) 104438. doi:10.1016/j . jbi. 2023.104438.

[21] T. L. Nguyen, G. Ifrim, TSHAP: Fast and Exact SHAP for Explaining Time Series Classification and Regression, Springer Nature Switzerland, 2025, p. 60–77. doi:10.1007/978-3-032-06078-5\_4.

[22] Y. Zhang, Q. Sun, D. Qi, et al., ShapTime: A General XAI Approach for Explainable Time Series Forecasting, in: Intelligent Systems and Applications: Proceedings of the 2023 Intelligent Systems Conference (IntelliSys), volume 822 of Lecture Notes in Networks and Systems, Springer, 2024, pp. 659–673.doi:10.1007/978-3-031-47721-8\_45.

[23] N. Jethani, A. Saporta, R. Ranganath, Don't be fooled: label leakage in explanation methods and the importance of their quantitative evaluation, in: International Conference on Artificial Intelligence and Statistics, PMLR, 2023, pp.8925-8953. URL: https://proceedings.mlr.press/v206/jethani23a.html.

[24] U. Biccari, A. I. de Opakua, J. M. Mato, et al., Fair feature attribution for multi-output prediction: a Shapley-based perspective, arXiv (2026). doi:10.48550/arXiv.2602.22882. arXiv:2602.22882.

[25] M. T. Ribeiro, S. Singh, C. Guestrin, “"Why Should I Trust You?": Explaining the Predictions of Any Classifier, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, ACM, 2016, p. 1135–1144. doi:10.1145/2939672.2939778.

[26] M. Sundararajan, A. Taly, Q. Yan, Axiomatic Attribution for Deep Networks, in: International Conference on Machine Learning, PMLR, 2017, pp. 3319–3328. URL: https://proceedings .mlr. press/v70/ sundararajan17a.html.

[27] I. C. Covert, S. Lundberg, S.-I. Lee, Understanding global feature contributions with additive importance measures, in: NIPS'20: Proceedings of the 34th International Conference on Neural Information Processing Systems, Curran Associates Inc., 2020, pp. 17212–17223. URL: https://d1.acm.org/doi/10.5555/ 3495724.3497168.

[28] A. Fisher, C. Rudin, F. Dominici, All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously, Journal of Machine Learning Research 20 (2019) 1-81. URL: http://jmlr.org/papers/v20/18-760.html.

[29] F. Fumagalli, M. Muschalik, E. Hüllermeier, et al., Unifying Feature-Based Explanations with Functional ANOVA and Cooperative Game Theory, arXiv (2024). doi:10.48550/arXiv.2412.17152. arXiv:2412.17152.

[30] N. Mehdiyev, M. Majlatow, P. Fettke, Integrating permutation feature importance with conformal prediction for robust Explainable Artificial Intelligence in predictive process monitoring, Engineering Applications of Artificial Intelligence 149 (2025) 110363. doi:10.1016/j. engappai.2025.110363

[31] D. Wood, T. Papamarkou, M. Benatan, et al., Model-agnostic variable importance for predictive uncertainty: an entropy-based approach, Data Mining and Knowledge Discovery 38 (2024) 4184–4216. doi:10.1007/ s10618-024-01070-7.

[32] S. M. Lundberg, G. Erion, H. Chen, et al., From local explanations to global understanding with explainable AI for trees, Nature Machine Intelligence 2 (2020) 56–67. doi:10.1038/s42256-019-0138-9.

[33] J. Chen, L. Song, M. J. Wainwright, et al., L-Shapley and C-Shapley: Efficient Model Interpretation for Structured Data, 2018. URL: https://arxiv.org/abs/1808.02610. arXiv:1808.02610.

[34] M. I. Idrissi, A. F. Machado, E. Gallic, et al., Unveil Sources of Uncertainty: Feature Contribution to Conformal Prediction Intervals, arXiv (2025). doi:10.48550/arXiv.2505.13118. arXiv:2505.13118.

[35] A. B. Owen, Sobol’ Indices and Shapley Value, SIAM/ASA Journal on Uncertainty Quantification 2 (2014) 245–251. doi:10.1137/130936233.

[36] M. Krzyziński, M. Spytek, H. Baniecki, et al., SurvSHAP(t): Time-dependent explanations of machine learning survival models, Knowledge-Based Systems 262 (2023) 110234. doi:10.1016/j . knosys .2022. 110234.

[37] S. H. Langbein, N. Koenen, M. N. Wright, Gradient-based Explanations for Deep Learning Survival Models, in: International Conference on Machine Learning, PMLR, 2025, pp. 32492–32522. URL: https://proceedings.mlr.press/v267/1angbein25a.html.

[38] F. Gamboa, A. Janon, T. Klein, et al., Sensitivity analysis for multidimensional and functional outputs, Electronic Journal of Statistics 8 (2014) 575–603. doi:10.1214/14-EJS895.

[39] M. B. Heredia, C. Prieur, N. Eckert, Global sensitivity analysis with aggregated Shapley effects, application to avalanche hazard assessment, Reliab. Eng. Syst. Saf. 222 (2022) 108420. doi:10.1016/j. ress.2022. 108420.

[40] S. Tonekaboni, S. Joshi, K. R. Campbell, et al., What went wrong and when? instance-wise feature importance for time-series black-box models, in: NIPS’20: Proceedings of the 34th International Conference on Neural Information Processing Systems, Curran Associates Inc., 2020, pp. 799–809. URL: https://dl.acm.org/doi/10.5555/3495724.3495792.

[41] F. R. Yapicioglu, M. Aksoy, T. Löfström, et al., ConformaSegment: A Conformal Prediction-Based, Uncertainty-Aware, and Model-Agnostic Explainability Framework for Time-Series Forecasting, Springer Nature Switzerland, 2025, p. 218–242. doi:10.1007/978-3-032-08330-2\_11.

[42] C. W. Ayad, T. Bonnier, B. Bosch, et al., Shapley Chains: Extending Shapley Values to Classifier Chains, in: International Conference on Discovery Science, Springer, Cham, Switzerland, 2022, pp. 541–555. doi:10.1007/978-3-031-18840-4\_38.

[43] L. S. Shapley, A Value for n-Person Games, Princeton University Press, 1953, pp. 307–318. doi:10.1515/ 9781400881970-018.

[44] K. Aas, M. Jullum, A. Løland, Explaining individual predictions when features are dependent: More accurate approximations to Shapley values, Artif. Intell. 298 (2021) 103502. doi:10.1016/j. artint . 2021.103502.

[45] C. Frye, D. de Mijolla, T. Begley, et al., Shapley explainability on the data manifold, in: International Conference on Learning Representations, 2021. URL: https://openreview.net/forum?id= OPyWRrcjVQw.

[46] T. M. Cover, J. A. Thomas, Elements of Information Theory, Wiley, 2005. doi:10. 1002/047174882x.

[47] R. B. Arellano-Valle, J. E. Contreras-Reyes, M. G. Genton, Shannon Entropy and Mutual Information for Multivariate Skew-Elliptical Distributions, Scand. J. Stat. 40 (2013) 42–62. doi:10.1111/j.1467-9469. 2011.00774.x.

[48] K.-T. Fang, S. Kotz, K. W. Ng, Symmetric Multivariate and Related Distributions, Chapman and Hall/CRC, 2018.doi:10.1201/9781351077040.

[49] L. F. Kozachenko, N. N. Leonenko, Sample Estimate of the Entropy of a Random Vector, Problems Inform. Transmission 23 (1987) 95–101. URL: https://ui.adsabs.harvard.edu/abs/1987PrIT...23... 95K/abstract.

[50] A. Trindade, ElectricityLoadDiagrams20112014, UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C58C86.

[51] H. Fanaee-T, Bike Sharing, 2013. URL: https://archive.ics.uci.edu/dataset/275. doi:10. 24432/C5W894, licensed under CC BY 4.0: https://creativecommons.org/1icenses/by/4.0/.

[52] E. Hüllermeier, W. Waegeman, Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods, Mach. Learn. 110 (2021) 457–506. doi:10.1007/s10994-021-05946-3.

[53] A. Kendall, Y. Gal, What uncertainties do we need in Bayesian deep learning for computer vision?, in: NIPS’17: Proceedings of the 31st International Conference on Neural Information Processing Systems, Curran Associates Inc., 2017, pp. 5580–5590. URL: https://d1.acm.org/doi/10.5555/3295222. 3295309.

[54] E. Baron, B. N. Oreshkin, R. Ma, H. Zhang, K. Torkkola, M. W. Mahoney, A. G. Wilson, T. Konstantinova, Efficiently Generating Correlated Sample Paths from Multi-step Time Series Foundation Models, in: Recent Advances in Time Series Foundation Models: Have We Reached the 'BERT Moment'? (NeurIPS 2025 BERT2S Workshop), 2025. URL: https://openreview.net/forum?id=ZFwFB4MEu2.

[55] T. B. Berrett, R. J. Samworth, M. Yuan, Efficient multivariate entropy estimation via k-nearest neighbour distances, Ann. Stat. 47 (2019) 288–318. doi:10.1214/18-A0S1688.

[56] T. Bedford, R. M. Cooke, Vines-a new graphical model for dependent random variables, The Annals of statistics 30 (2002) 1031–1068.

[57] K. Aas, C. Czado, A. Frigessi, H. Bakken, Pair-copula constructions of multiple dependence, Insurance: Mathematics and economics 44 (2009) 182–198.

[58] S. Watanabe, Information Theoretical Analysis of Multivariate Correlation, IBM Journal of Research and Development 4 (1960) 66–82. doi:10.1147/rd.41.0066.

[59] S. M. Lundberg, G. G. Erion, S.-I. Lee, Consistent Individualized Feature Attribution for Tree Ensembles, arXiv (2018). doi:10.48550/arXiv.1802.03888. arXiv:1802.03888.

## A Additional Background

## A.1 Information Theory

In this appendix, we collect and explain in more detail the information-theoretic concepts used throughout the paper for our hierarchy of Shapley games, in particular the joint and conditional entropy, the (conditional) total correlation, and their behavior under chain-rule decompositions. For further details on information theory and total correlation, we refer to Cover and Thomas [46] and Watanabe [58].

Joint and conditional entropy. The (differential) entropy $H ( X )$ defined in Eq. (1) extends to a random vector X on a domain $\mathcal { X } \subseteq \mathbb { R } ^ { p }$ as the joint entropy

$$
H ( X ) = \mathbb { E } _ { X } { \bigl [ } - \log p ( X ) { \bigr ] } = - \int _ { \mathcal { X } } p ( x ) \log p ( x ) d x ,\tag{A.1}
$$

which depends on the full joint distribution $p ( { \pmb x } )$ . For our hierarchy, the central quantity is the conditional entropy of the multivariate output $\scriptstyle { \dot { \boldsymbol { Y } } }$ (on $\mathcal { V } \subseteq \mathbb { R } ^ { T } )$ given the inputs X (on $\mathcal { \bar { X } } \subseteq \bar { \mathbb { R } } ^ { p } )$ . For a specific realization $X = x$ , the local conditional entropy

$$
H ( Y \mid X = x ) = - \int _ { \mathcal { Y } } p ( \pmb { y } \mid \pmb { x } ) \log \left( p ( \pmb { y } \mid \pmb { x } ) \right) d \pmb { y }\tag{A.2}
$$

measures the residual uncertainty in Y for that particular instance x. Averaging over the input distribution yields the global conditional entropy

$$
\begin{array} { r l } & { H ( \pmb { Y } \mid \pmb { X } ) = \mathbb { E } _ { \pmb { \tilde { X } } } \left[ H ( \pmb { Y } \mid \pmb { X } = \pmb { \tilde { X } } ) \right] = - \displaystyle \int _ { \pmb { \tilde { X } } } p ( \pmb { x } ) \int _ { \pmb { \tilde { y } } } p ( \pmb { y } \mid \pmb { x } ) \log \left( p ( \pmb { y } \mid \pmb { x } ) \right) d \pmb { y } d \pmb { x } } \\ & { \qquad = - \displaystyle \int _ { \pmb { \tilde { X } } \pmb { \tilde { y } } } p ( \pmb { y } , \pmb { x } ) \log \left( p ( \pmb { y } \mid \pmb { x } ) \right) d \pmb { y } d \pmb { x } , } \end{array}\tag{A.3}
$$

which captures the expected residual uncertainty across the population. This local-vs-global distinction is used throughout Section 4: our value functions are precisely partial averages of this kind, where the expectation is taken only over the out-of-coalition features $\bar { \pmb { X } } _ { \bar { \pmb { S } } }$ while $X _ { S } ^ { - }$ is held fixed at the instance value.

Total correlation and its conditional counterpart. The total correlation [58] measures the overall statistical redundancy among the components of a random vector X and is defined as

$$
\operatorname { T C } ( X ) = \sum _ { i = 1 } ^ { p } H ( X _ { i } ) - H ( X ) ,\tag{A.4}
$$

i.e., the difference between the summed marginal entropies and the joint entropy. It satisfies $\mathrm { T C } (  { \boldsymbol { X } } ) \geq 0$ , with equality iff the components are mutually independent. For the supervised setting, the relevant quantity is the conditional total correlation, which measures the dependence among the multivariate outputs $\mathbf { Y }$ that remains after conditioning on the covariates $\boldsymbol { X }$

$$
\operatorname { T C } ( \boldsymbol { Y } \mid \boldsymbol { X } ) = \sum _ { t = 1 } ^ { T } H ( Y _ { t } \mid \boldsymbol { X } ) - H ( Y \mid \boldsymbol { X } ) = \mathbb { E } _ { \boldsymbol { \tilde { X } } } \left[ \operatorname { T C } ( \boldsymbol { Y } \mid \boldsymbol { X } = \boldsymbol { \tilde { X } } ) \right] ,\tag{A.5}
$$

following the same local-vs-global convention as the conditional entropy in Eq. (A.3), i.e., $\textstyle \operatorname { T C } ( Y \mid X = x ) = \sum _ { t = 1 } ^ { T } H ( Y _ { t } \mid X = x ) - H ( Y \mid X = x )$ . The local quantity $\mathrm { T C } ( Y | X = x )$ is the redundancy among output components for a specific instance, and Prop. 2 characterizes our cross-component attributions as the Shapley decomposition of the local-vs-global gap $\mathrm { T C } ( Y | { \pmb x } ) - \mathrm { T C } ( \hat { Y } | { \pmb X } )$

Chain rule of entropy. Joint entropies decompose sequentially via the chain rule of entropy,

$$
H ( \pmb { X } ) = \sum _ { i = 1 } ^ { p } H ( X _ { i } \mid X _ { 1 } , \ldots , X _ { i - 1 } ) ,\tag{A.6}
$$

which holds for any permutation of the indices. In our predictive setting, this extends using the conditional versions of above to $\begin{array} { r } { H ( \pmb { Y } | \pmb { X } ) = \sum _ { t = 1 } ^ { T } H ( Y _ { t } | \pmb { Y } _ { < t } , \pmb { X } ) } \end{array}$ . While the joint entropy is invariant to the chosen ordering, the individual summands are not: different orderings produce different sequential decompositions, all summing to the same joint entropy. This ordering dependence is what makes our Level 2 game well-defined for ordered multivariate outputs such as time-series forecasts, where a natural ordering is given by the problem structure.

## B Proofs of Section 4

## B.1 Proof of Prop. 1 (Chain-rule linkage)

Proof. The value function of the joint entropy game (Eq. (7)) decomposes into the sum over time of the sequential entropy game value functions (Eq. (6)):

$$
\begin{array} { l } { { \displaystyle v _ { H } ^ { \mathrm { i o n } } ( S , x ) = \mathbb { E } _ { X _ { \bar { S } } } \bigg [ H \big ( { \cal Y } \mid { \cal X } = ( x _ { S } , X _ { \bar { S } } ) \big ) \bigg ] } } \\ { ~ } \\ { { \displaystyle ~ = \mathbb { E } _ { X _ { S } } \left[ \sum _ { t = 1 } ^ { T } H \big ( { \cal Y } _ { t } \mid { \cal Y } _ { < t } , { \cal X } = ( x _ { S } , X _ { \bar { S } } ) \big ) \right] } } \\ { ~ } \\ { { \displaystyle ~ = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X _ { \bar { S } } } \bigg [ H \big ( { \cal Y } _ { t } \mid { \cal Y } _ { < t } , { \cal X } = ( x _ { S } , X _ { \bar { S } } ) \big ) \bigg ] } } \\ { ~ } \\ { { \displaystyle ~ = \sum _ { t = 1 } ^ { T } v _ { H } ^ { ( t ) < t } ( S , x ) } } \end{array}\tag{B.1}
$$

where the second line applies the chain rule of (conditional) entropy, while the third line uses the linearity of expectation.

The decomposition of the corresponding Shapley values $\begin{array} { r } { \phi ^ { \mathrm { j o i n t } } ( j , \pmb { x } ) = \sum _ { t = 1 } ^ { T } \phi ^ { ( t | < t ) } ( j , \pmb { x } ) } \end{array}$ follows directly from Eq. (B.1) and the linearity axiom of Shapley values.

Finally, if the output components $Y _ { 1 } , \dots , Y _ { T }$ are conditionally independent given X, then $H ( Y _ { t } \mid Y _ { < t } , X ) = \dot { H } ( Y _ { t } \mid \dot { X ) }$ for any X. Consequently, the Level 2 and Level 1 value functions coincide for all coalitions, yielding identical Shapley values $\phi ^ { ( t | < t ) } ( j , \pmb { x } ) = \phi ^ { ( t ) } ( j , \pmb { x } )$ and reducing the joint attribution to $\begin{array} { r } { \phi ^ { \mathrm { j o i n t } } ( j , \pmb { x } ) = \sum _ { t = 1 } ^ { T } \phi ^ { ( t ) } ( j , \pmb { x } ) } \end{array}$ □

## B.2 Proof of Prop. 2 (Cross-component decomposition)

Proof. The value function of the total correlation game mirrors the decomposition of the total correlation into marginal and joint conditional entropies:

$$
\begin{array} { l } { { v _ { \mathrm { T C } } ( S , x ) = \mathbb { E } _ { X _ { S } } \bigl [ \mathrm { T C } ( Y \mid X ) \mid X = ( x _ { S } , X _ { \bar { S } } ) \bigr ] } } \\ { { \ \ } } \\ { { \ = \mathbb { E } _ { X _ { S } } \bigl [ \displaystyle \sum _ { t = 1 } ^ { T } H ( Y _ { t } \mid X = ( x _ { S } , X _ { \bar { S } } ) ) - H ( Y \mid X = ( x _ { S } , X _ { \bar { S } } ) ) \bigr ] } } \\ { { \ \qquad = \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { X _ { \bar { S } } } \bigl [ H ( Y _ { t } \mid X = ( x _ { S } , X _ { \bar { S } } ) ) \bigr ] - \mathbb { E } _ { X _ { \bar { S } } } [ H ( Y \mid X = ( x _ { S } , X _ { \bar { S } } ) ) ] } } \\ { { \ \qquad = \displaystyle \sum _ { t = 1 } ^ { T } v _ { H } ^ { ( t ) } ( S , x ) - v _ { H } ^ { \mathrm { i o n n } } ( S , x ) } } \end{array}\tag{B.2}
$$

where in the second line we used the definition of total correlation and in the third line the linearity of the expected value operator. From Eq. (B.2) and the linearity of the Shapley values in the value function follows that $\begin{array} { r } { \dot { \phi ^ { \mathrm { T C } } } ( j , \pmb { x } ) = \sum _ { t } \dot { \phi ^ { ( t ) } } ( j , \pmb { x } ) - \phi ^ { \mathrm { j o i n t } } ( j , \pmb { x } ) } \end{array}$ , where $\phi ^ { \mathrm { T C } } ( \dot { j } , { \pmb x } )$ are the Shapley values of the total correlation game $v _ { \mathrm { T C } } ( S , \pmb { x } )$

## B.3 Local-Global conditional entropy decomposition of Shapley values

Corollary 3. For the Level 1–3 games (Eqs. (5)–(7)), the sum of the Shapley values equals the difference between the local and global conditional entropies:

$$
\begin{array} { l } { { \displaystyle \sum _ { j = 1 } ^ { p } \phi ^ { ( t ) } ( j , { \pmb x } ) = H ( Y _ { t } \mid { \pmb x } ) - H ( Y _ { t } \mid { \pmb X } ) \qquad ( L e v e l I ) } } \\ { { \displaystyle \sum _ { j = 1 } ^ { p } \phi ^ { ( t | < t ) } ( j , { \pmb x } ) = H ( Y _ { t } \mid Y _ { < t } , { \pmb x } ) - H ( Y _ { t } \mid Y _ { < t } , { \pmb X } ) \qquad ( L e v e l 2 ) } } \\ { { \displaystyle \sum _ { j = 1 } ^ { p } \phi ^ { j o i n t } ( j , { \pmb x } ) = H ( { \pmb Y } \mid { \pmb x } ) - H ( { \pmb Y } \mid { \pmb X } ) \qquad ( L e v e l 3 ) } } \end{array}\tag{B.3}
$$

Analogously, for the $T C$ game $( E q .$ (B.2)), the sum of the Shapley values matches the local–global total correlation gap:

$$
\sum _ { j = 1 } ^ { p } \phi ^ { \mathrm { T C } } ( j , \pmb { x } ) = \mathrm { T C } ( \pmb { Y } | \pmb { x } ) - \mathrm { T C } ( \pmb { Y } | \pmb { X } ) .\tag{B.4}
$$

Proof. For each game: By the efficiency property of Shapley values, the sum of Shapley values $\scriptstyle \sum _ { j = 1 } ^ { p } \phi ( j , x )$ equals the difference between the value function of the grand coalition $( \bar { S } = \bar { [ \boldsymbol { p } ] } )$ and the value function of the empty coalition $( S = \emptyset )$ . For our games, the grand coalition evaluates the local conditional entropies at the specific instance x, while the empty coalition evaluates the global conditional entropies marginalized over the entire input space. □

## C Trajectory-Based Gaussian Copula Estimation

In Section 4.2.3, we introduced a semiparametric Gaussian copula approach to efficiently estimate the local entropy terms required for our hierarchy of Shapley games. This method leverages simulated trajectories to capture complex, non-Gaussian marginal distributions while utilizing the computational tractability of Gaussian dependencies.

Below, we detail the complete, step-by-step procedure for estimating the value functions of a specific feature coalition S for a given instance x. The procedure evaluates the inner conditional entropies for a single completed input $\tilde { \boldsymbol { x } } = \tilde { \boldsymbol { x } } ^ { ( k ) }$ (generated via the chosen Shapley imputation method, as described in Sec. 4.2) and then averages over $\bar { K }$ such completed inputs to evaluate the outer expectation.

1. Trajectory Simulation: For a completed input x, simulate N joint trajectories from the predictive model:

$$
\pmb { y } ^ { ( 1 ) } , \ldots , \pmb { y } ^ { ( N ) } \sim p ( \pmb { Y } | \tilde { \pmb { x } } )
$$

2. Marginal Density Estimation: For each output component $t \in \{ 1 , \ldots , T \}$ , estimate the 1D marginal density $\hat { f } _ { t } ( \cdot | \tilde { \pmb { x } } )$ and the cumulative distribution function $\hat { F } _ { t } ( \cdot | \tilde { x } )$ using kernel density estimation (KDE) on the scalar samples $\{ y _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }$

3. Level 1 (Marginal Entropy): Estimate the Level 1 entropy for each component t directly from the fitted marginal densities e.g., via the leave-one-out plug-in estimator:

$$
\widehat { H } ( Y _ { t } \mid \tilde { \mathbf { \phi } } ( \mathbf { \phi } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \hat { f } _ { t , - i } ( { \boldsymbol y } _ { t } ^ { ( i ) } \mid \tilde { \mathbf { \phi } } ( \mathbf { \phi } )\tag{C.1}
$$

where $\hat { f } _ { t , - i }$ denotes the KDE fitted without the i-th sample to avoid overfitting.

4. Gaussianization & Correlation: Map the samples to the uniform domain using the empirical CDFs, $u _ { t } ^ { ( i ) } = \hat { F } _ { t } ( y _ { t } ^ { ( i ) } | \tilde { \mathbf { x } } )$ , and subsequently to the standard normal domain via the probit function, $z _ { t } ^ { ( i ) } = \Phi ^ { - 1 } ( u _ { t } ^ { ( i ) } )$ . Compute the $T \times T$ empirical correlation matrix $\hat { R } ( \tilde { \boldsymbol { x } } )$ from these latent Gaussian vectors $\boldsymbol { z } ^ { ( i ) }$

5. Level 2 (Sequential Entropy): For each step $t > 1 .$ , calculate the conditional variance of the latent Gaussian variable $Z _ { t }$ given the preceding variables $Z _ { < t }$ using the Schur complement:

$$
\hat { \sigma } _ { t \mid < t , z } ^ { 2 } ( \tilde { \pmb x } ) = 1 - \hat { R } _ { t , < t } \hat { R } _ { < t , < t } ^ { - 1 } \hat { R } _ { < t , t }
$$

Use this variance to compute the Level 2 sequential entropy:

$$
\widehat { H } ( Y _ { t } | Y _ { < t } , \tilde { \pmb { x } } ) = \widehat { H } ( Y _ { t } | \tilde { \pmb { x } } ) + \frac { 1 } { 2 } \log \bigl ( \hat { \sigma } _ { t | < t , z } ^ { 2 } ( \tilde { \pmb { x } } ) \bigr )\tag{C.2}
$$

6. Level 3 (Joint Entropy): Assemble the total joint entropy by summing the Level 2 components (or equivalently, Level 1 plus the copula entropy), effectively applying the chain rule from Prop. 1:

$$
\widehat { H } ( \pmb { Y } | \tilde { \pmb { x } } ) = \sum _ { t = 1 } ^ { T } \widehat { H } ( Y _ { t } | \pmb { Y } _ { < t } , \tilde { \pmb { x } } )
$$

7. Outer Expectation: Repeat Steps 1–6 for $K$ different completed inputs $\tilde { \boldsymbol { x } } = \tilde { \boldsymbol { x } } ^ { ( k ) }$ generated according to the chosen imputation strategy (e.g., marginal, conditional, or baseline). Average the resulting entropy estimates to yield the final coalition value functions $v _ { H } ^ { ( t ) } ( S , \pmb { x } )$ $v _ { H } ^ { ( t | < t ) } ( S , { \pmb x } )$ , and $v _ { H } ^ { \mathrm { j o i n t } } ( S , x )$

## D Details and Additional Results on the Experiments

In the following sections, we describe the technical details and additional results for all experiments covered in Section 5.

General Experimental Setup. Throughout all experiments, we compute the exact Shapley values by evaluating the respective value functions across all $2 ^ { p }$ possible feature coalitions, without relying on coalition-sampling approximations. Furthermore, to evaluate the outer expectation over the out-of-coalition features (i.e., the Shapley imputation step described in Sec. 4.2), we consistently employ marginal imputation using background samples drawn from the respective training datasets [12]. Finally, the code for reproducing the results can be found on GitHub.

## D.1 Proof of Concept on Synthetic Gaussian DGP

For this synthetic experiment, the data is generated using a multivariate Gaussian distribution $Y \mid _ { x \sim }$ $\mathcal { N } ( \pmb { \mu } ( \pmb { x } ) , \pmb { \Sigma } ( \pmb { x } ) )$ with $T = 4$ output components and $p = 4$ input features drawn independently from a standard normal. Each feature has a distinct ground-truth effect on the parameters of the predictive distribution. The mean vector $\pmb { \mu } ( \pmb { x } )$ depends only on the first feature $x _ { 1 }$ via

$$
\mu _ { t } ( x ) = \beta x _ { 1 } \sin ( 2 \pi t / T ) \quad \mathrm { w i t h } \quad \beta = 1 .
$$

The covariance is constructed as $\Sigma ( { \pmb x } ) = D ( { \pmb x } ) R ( { \pmb x } ) D ( { \pmb x } )$ , where $D ( \pmb { x } )$ is the diagonal of marginal standard deviations

$$
\sigma _ { t } ^ { 2 } ( { \pmb x } ) = \exp \bigl ( \gamma _ { 0 } x _ { 2 } t + \gamma _ { 1 } x _ { 3 } \bigr ) \quad \mathrm { w i t h } \quad \gamma _ { 0 } = 0 . 5 , \gamma _ { 1 } = 0 . 8 ,
$$

and $R ( { \pmb x } )$ is a smoothly decreasing Toeplitz-style correlation matrix driven by

$$
r (  { \boldsymbol { { x } } } ) = \operatorname { t a n h } \bigl ( \delta _ { 0 } x _ { 3 } + \delta _ { 1 } x _ { 4 } \bigr ) , \quad \delta _ { 0 } = - 0 . 6 , \delta _ { 1 } = 1 . 2 5 , \qquad R _ { i j } (  { \boldsymbol { { x } } } ) = \mathrm { s i g n } ( r (  { \boldsymbol { { x } } } ) ) ^ { | i - j | } | r (  { \boldsymbol { { x } } } ) | ^ { | i - j | ^ { 1 . 5 } } ,
$$

where the correlation decays faster than linearly with the distance $| i - j |$ , so that neighbouring components remain strongly correlated but distant ones are nearly uncorrelated. Through this design, each feature has pre-defined effects on the predictive distribution: $x _ { 1 }$ controls the mean prediction only and should be invisible for uncertainty attribution, $x _ { 2 }$ has an increasing effect on the marginal variance for higher component indices, $x _ { 3 }$ controls both variance and correlation, whereas $x _ { 4 }$ only controls the correlation.

For the calculation of the Shapley values for the different levels, we use the analytical formula for the entropies (see Sec. 4.2.1) and use 10, 000 background samples for the Monte-Carlo integration in order to have stable results. The explanations in Figure 1 are based on the instance ${ \pmb x } = ( 1 , 1 , 1 , 1 . 7 5 )$ i.e., an instance with very high correlations $( \rho \approx 0 . 9 2 )$ for demonstration purposes.

Visualizing the hierarchy on the covariance matrix Σ. For the multivariate Gaussian case (see Sec. 4.2.1), the three levels of the hierarchy operate on different parts of the covariance matrix Σ(x), which provides a geometric intuition for what each level captures. Figure D.1 illustrates this for $T = 4 \colon$ Level 1 depends only on the diagonal entries $\Sigma _ { t t } ,$ Level 2 depends on the leading principal submatrices through Schur complements $\Sigma _ { t \mid < t } ,$ and Level 3 depends on the full covariance matrix via its log-determinant. The chain-rule decomposition (Prop. 1) is reflected in this structure: the joint log-determinant decomposes into the sum of conditional log-variances, which collapses to the sum of marginal log-variances exactly when the components are independent, i.e., when Σ is diagonal.

![](images/7573df903e5ad8204e2d0d553c4194405b285bcb8dcd3f0fe6f8ef4aa199adeb.jpg)  
Figure D.1: Geometric decomposition of the multivariate Gaussian entropy hierarchy on the covariance matrix Σ(x). Level 1 (left) depends only on the diagonal entries $\Sigma _ { t t } ,$ Level 2 (centre) on the leading principal submatrices via Schur complements $\Sigma _ { t \mid < t }$ , and Level 3 (right) on the full matrix via its log-determinant. The right panel shows how the chain rule of entropy decomposes the joint entropy into a sum of conditional entropies, collapsing to a sum of marginals under conditional independence.

## D.2 Distributional Regression on Bike Sharing

Dataset. We use the UCI Bike Sharing hourly dataset [51] and restrict the analysis to the demand window 06:00–22:00 of each day. The hourly rental counts are aggregated into $T = 8 \mathrm { t w o }$ -hour blocks $\pmb { Y } = ( Y _ { 1 } , \ldots , Y _ { 8 } )$ that serve as the multivariate target. We use $p = 8$ features as input, summarized in Table D.1. The continuous weather features are observed at 06:00 of the forecast day, and Week Temp and Prev Count are lagged by one day. This setting ensures that no information from the target window 06:00–22:00 leaks into the features.

Table D.1: Feature set used for the Bike Sharing experiment $( p = 8 )$
<table><tr><td>Feature</td><td>Description</td></tr><tr><td>Temp 6am</td><td>06:00 normalized temperature</td></tr><tr><td>Hum 6am</td><td>06:00 normalized humidity</td></tr><tr><td>Wind 6am</td><td>06:00 normalized wind speed</td></tr><tr><td>Week Temp</td><td>7-day rolling mean temperature, lagged by 1 day</td></tr><tr><td>Prev Count</td><td>yesterday&#x27;s mean rentals per hour (scaled by 1/100)</td></tr><tr><td>Workday</td><td>binary indicator (1 if working day)</td></tr><tr><td>Weekday</td><td>day of the week  $( 0 = \mathrm { S u n } , \dots . . . , 6 = \mathrm { S a t } )$ </td></tr><tr><td>Year</td><td>calendar year  $( 0 = 2 0 1 1 , 1 = 2 0 1 2 )$ </td></tr></table>

Model. We split the data into 80% training and 20% test instances after a random shuffle. On the training set, we fit an NGBoost model [7] with a multivariate Gaussian output head $\mathcal { N } ( \pmb { \mu } ( \pmb { x } ) , \pmb { \Sigma } ( \pmb { x } ) )$ over the $T = 8$ time blocks. The model parameterizes the mean vector and a Cholesky factor of the covariance matrix. We use the default base learner (a depth-2 decision tree) and train for up to 1500 boosting rounds with learning rate 0.01, minibatch fraction 0.8, and early stopping after 200 rounds without improvement on the held-out test set.

Shapley computation. Since NGBoost outputs a parametric multivariate Gaussian, all three entropy levels are evaluated analytically using the closed-form expressions from Sec. 4.2.1. For the local analysis on the two instances shown in Fig. D.2, we use the full training set as background to approximate the expectation over the off-coalition features.

![](images/ac9ddc88b86bd7399037515997b334369d1911798205c6527487368c79aa7049.jpg)  
Figure D.2: Summary of the feature values (including the population means) and the predictive distribution of the two instances to be explained. The heatmaps show the correlation of the predictive distribution, i.e., of Σ(x) for the two instances.

## D.2.1 Additional Result: Comparison to SHAP

The hierarchy in Figure D.3 captures a fundamentally different quantity than standard SHAP, since SHAP (or TreeSHAP[59] for NGBoost) decomposes the point prediction, i.e., each $\phi _ { j , t } ^ { \mathtt { S H A P } }$ quantifies by how many rentals feature $j$ shifts the predicted mean $\hat { \mu } _ { t } ( \pmb { x } )$ at time point t. The games of our hierarchy, in contrast, decompose the predictive uncertainty, $\mathrm { i . e . , } \phi _ { H } ^ { ( t ) } , \phi _ { H } ^ { ( t | < t ) }$ , and $\phi _ { H } ^ { \mathrm { j o i n t } }$ quantify how much each feature contributes to the marginal, sequentially conditional, and joint entropy of the forecast distribution, measured in nats. These are different quantities of $p ( \pmb { y } \vert \pmb { x } )$ , and there is no reason to expect them to agree, not even on the ranking of features. Figure D.3 makes this distinction concrete and shows different patterns for the different levels compared to SHAP.

![](images/e414953b4ec47aa69bad1bc2bd42884c3ef2fe3611d58a978e20ec05961f40a2.jpg)

![](images/8d711cbb40684ce61faf9ea886f4e79dd616ce645c539b07d5450f018b080f1e.jpg)

![](images/1f70de2ebaf9a5582952a8472d493a79f6aaa4ef751b7f4da8606cfff12f8e99.jpg)

![](images/f4383ebf6eb471747eb15cdc6629e1f6e1d3076fbb9956dc2966588924fa1a40.jpg)  
Figure D.3: Local SHAP vs. Entropy-Shapley attributions for the workday and weekend test instances (left, centre) and global aggregation over the full test set (right). SHAP attributes the predicted mean ${ \hat { \mu } } _ { t } ( \pmb { x } )$ in rentals, while Level 1 $( \phi _ { H } ^ { ( t ) } )$ and Level 2 $( \phi _ { H } ^ { ( t | < t ) } )$ attribute the marginal and sequentially conditional predictive entropy in nats, respectively. The right panel compares the mean of $| \phi _ { H } ^ { \mathrm { j o i n t } } |$ across the test set with the corresponding mean absolute $\bar { \mathrm { S H A P } }$ values, each min-max normalized within its method. This highlights how feature effects on the mean and uncertainty are naturally decoupled: for instance, Workday strongly drives point prediction but not the joint (daily) uncertainty, whereas the opposite is true for Hum 6am and Temp 6am.

## D.3 Cross-component Diagnostic across Forecaster Classes

This appendix summarizes the setup and additional analyses for the sample-based estimation and cross-component diagnostic experiment of Sec. 5.3.

Dataset. We use the UCI Electricity hourly load dataset [50], restricted to the first 100 series in order to keep the training time feasible. The hourly electricity load is aggregated into $T = 1 2$ two-hour blocks per day, i.e., blocks of width 2h with a prediction window covering a whole day. For each series, we scaled the values by their per-series mean, so the analyzed target is on a relative-to-mean scale. The multivariate forecast is of dimension $T = 1 2 \ : ( \mathrm { i . e . } \ : 2 4 \mathrm { h } )$ , the history for context is 84 blocks (i.e. one week), and we hold out the last 360 blocks per series as a validation period from which the test origins are drawn. In addition, we add calendar features (hour, weekday, and month) as exogenous variables.

Models. We train DeepAR [1] from scratch with three different one-step output distributions: Gaussian, Student-t, and log-normal. We used a default architecture with 2 RNN layers, a hidden size of 64 and trained it for up to 50 epochs with Adam (learning rate $1 0 ^ { - 3 }$ , batch size 256) and early stopping on the validation set. On the other hand, we use Chronos-T5-base⁴ [9] zero-shot, i.e., without any fine-tuning. The model tokenizes the history context with its quantization grid (4096 token bins) and decodes $T = 1 2$ tokens autoregressively.

Shapley setup (Sec. 5.3). The historical context plus three calendar features yields $P _ { \mathrm { f e a t } } = 8 7$ scalar inputs per origin. To keep the Shapley computation tractable, we group these into $p = 6$ players: one player per calendar feature (hour, weekday, month), and three lookback-window players capturing the recent past (last 24 hours), the middle past (the preceding three days), and the older past (the remainder of the week-long context). For each model and forecast origin (100 in total), we enumerate all $2 ^ { p } = 6 4$ coalitions and impute features outside the coalition by drawing $K = 2 5$ background contexts from the training period, ensuring that no validation information leaks into the value function. For each coalition, we sample $N = \mathrm { 1 0 0 0 }$ joint trajectories of length T and evaluate the three entropy levels with the sample-based estimators of Sec. 4.2.3 (Gaussian copula using KDE marginals and kNN with $k = 5 )$ . Shapley values are obtained exactly from the $2 ^ { \bar { p } }$ value-function evaluations.

## D.3.1 Additional Results: Validation against the Model-based Reference

We train DeepAR [1] on the dataset using Gaussian, Student-t, and log-normal likelihoods for the one-step conditional distributions (Sec. 4.2.2). DeepAR's autoregressive factorization provides the conditional densities in closed form, which can be marginalized over trajectory samples of $\mathbf { Y } _ { < t }$ to obtain the Level 2 reference $v _ { H } ^ { ( t | < t ) } ( S , \pmb { x } )$ , with Level 3 following from Prop. 1. This reference is itself an estimator in the strict sense, but uses model-internal closed-form conditionals rather than relying on entropy estimation from samples. It therefore serves as the strongest available baseline. We compare the sample-based estimators from Sec. 4.2.3 against this reference, using a parametric Gaussian fit (PG), the Gaussian copula with KDE marginals (GC-KDE), and kNN. In addition, we include a Gaussian copula variant (GC-Param) whose marginals are obtained from the marginalized model conditionals, which isolates the copula component of the model bias. For this validation, we use the setup described above, but vary the trajectory budget $N \in \{ 5 0 , 1 0 0 , 2 5 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 \}$ to study the convergence behavior of each estimator. Figure D.4 reports the mean absolute error on the value function (top row) and on the resulting Shapley values (bottom row) as a function of N. The marginal imputation uses $K = 2 5$ random background origins, fixed across all settings to eliminate Monte-Carlo integration variance from the comparison. Error bars are computed over 100 test instances.

For the Gaussian likelihood, PG and the copula-based approaches are all accurate, while kNN improves more slowly with N, reflecting the cost of imposing no parametric structure. The small gap between GC-Param and GC-KDE quantifies the additional cost of estimating marginals from samples. The non-Gaussian likelihoods expose the effect of misspecification. For Student-t, PG's error grows with N, indicating convergence to a misspecified Gaussian target: the downward bias of the empirical log-determinant masks the misspecification at small N and reveals it as N grows. The copula-based estimators substantially reduce this effect, with the residual GC-Param error isolating the copula component, although a Gaussian copula can itself be misspecified under strong tail dependence. For log-normal, errors are smaller throughout, indicating that the severity of misspecification depends on the predictive distribution. The bottom row shows how these errors propagate to the attributions: common offsets across coalitions partly cancel in marginal contributions, while coalition-specific errors remain.

## E Computational Details

Experiments were conducted using a 64-bit Linux platform running Ubuntu 22.04 LTS with two AMD EPYC Genoa 9534 64-Core processors (128 cores, 256 threads total), 1.5 terabytes of RAM, and eight NVIDIA RTX 6000 Ada Generation GPUs (each with 48 GB memory). Each individual run used a single GPU. Only DeepAR training and Chronos inference require the GPU, whereas analytical and sample-based Shapley computations, including NGBoost training, run on CPU with process-level parallelism via joblib. Additionally, the memory-expensive notebooks implement a chunk-wise calculation to prevent out-of-memory failures. In this computational setting, the synthetic DGP (Sec. 5.1) and bike sharing experiment (Sec. 5.2) run in minutes. For the DeepAR experiments, training the DeepAR architecture across the three likelihoods takes approximately 2 hours in total, and the validation experiment takes approximately 10 hours (which can be reduced by using fewer instances). The Chronos experiment also takes approximately one hour. All experiments also include a “quick run" option, in which the code is executed at the cost of accuracy.

- Parametric Gaussian- Gauss copula (true marginals)-- Gauss copula (KDE marginals)  kNN (k = 5)  
![](images/59a284f6a928c498183e619ee04b4f3f5d25ad6669d33c2167dd98ae562999ba.jpg)  
Figure D.4: Sample-based estimation against the parametric model-based Level 2 reference $v _ { H } ^ { ( \bar { t } | < t ) } ( S , \pmb { x } )$ , for DeepAR with Gaussian, Student-t, and log-normal likelihoods. Top: MAE on the value function. Bottom: MAE on the resulting Shapley values.

## F Impact Statement

This work focuses on a foundational post-hoc explanation methodology not tied to specific deployments. Any potential societal impacts, positive or negative, would therefore be indirect. For example, by improving transparency of probabilistic forecasters and decomposing predictive uncertainty into structurally distinct components, the framework can strengthen informed decisions in risk-sensitive applications such as energy planning or weather forecasting. On the negative side, unintended misuse where the model-relative Shapley attributions are misinterpreted as causal explanations could instill misplaced confidence in model reliability and have societal impacts when occurring in a safetycritical setting. However, this risk is not specific to this work, but inherent to the broader class of Shapley-based attribution methods.