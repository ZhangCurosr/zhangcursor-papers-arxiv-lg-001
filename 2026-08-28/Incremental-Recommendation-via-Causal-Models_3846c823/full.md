# Incremental Recommendation via Causal Models

Athanasios Vlontzos Hologen Imperial College London, UK

Michael O’Riordan moriordan@spotify.com Spotify, UK

## Abstract

Recommendation impressions are a finite resource, hence delivering a recommendation to a user who would discover the content organically yields no incremental value and displaces other recommendations that could. We address this by extending an existing production recommendation model to a causal architecture using holdback data that is already collected as part of routine experimentation infrastructure, requiring no new data collection. A central challenge is that attribution windows difer between treated and holdback observations: treated users are attributed a stream within a short direct-response window, while holdback users are attributed organic streams over a multi-day window. This mismatch makes naïve treatment-efect subtraction invalid. We resolve this with a dual-threshold targeting policy that delivers a recommendation only when the probability of a treated stream is high and the probability of organic stream is low. In a production-scale A/B test on millions of Spotify users, this policy reduces recommendation impressions by 7% with no statistically significant reduction in overall recommended content consumption. We further show that joint training with holdback data improves calibration of the treated head relative to the production baseline, and argue this can be taken as evidence that causal models learn more generalisable representations than models trained on observational data alone.

## 1 Introduction

Recommendation impressions are scarce. There is thus a strong incentive to ensure that each recommendation is eficient. That is, that each recommendation has an impact. Standard recommenders are trained to maximize the probability a user streams the recommended content—a sensible objective that nevertheless ignores a critical question: would the user have streamed this content anyway?

David Gustafsson davidgustafsson@spotify.com Spotify, Sweden

Users who would stream a piece of content regardless of receiving a recommendation are known in the causal literature as always-takers [13]. For a music streaming platform, an alwats-taker might be a user who closely follows an artist and checks their new releases weekly: delivering a recommendation for the artist’s new album to such a user does not drive incremental consumption, it merely tags along with behavior that would occur organically. A model optimising stream probability will systematically favour these high-afinity users, because they are the easiest to predict, and will do so at the cost of ignoring users for whom a recommendation would make a genuine diference.

Ciarán M. Gilligan-Lee ciaranl@spotify.com Spotify, Ireland University College London, UK

Correcting for this requires answering the counterfactual: what would the user have done without the recommendation? This is the fundamental challenge of causal inference for recommendation. Many production recommendation systems already collect exactly the data needed to answer this question, in the form of holdback experiments, randomised trials in which a small fraction of eligible users are withheld recommendations. Holdback data provides direct samples from the counterfactual distribution of organic behavior, yet is rarely used to build causal recommendation models.

We show how to transform an existing production recommendation model into a causal incremental model using this holdback data, without any new data collection infrastructure. The key technical contribution is identifying and resolving a structural obstacle—the attribution window mismatch—that prevents naïve treatment-efect estimation in this setting and, if ignored, would invalidate the resulting causal model. We validate the approach at production scale on Spotify’s Home recommendation surface, illustrated in Figure 1, demonstrating that the resulting dual-threshold targeting policy reduces recommendation impressions by 7% with no statistically significant reduction in recommended content consumption.

Our contributions are as follows.

• We identify and formalise the attribution window mismatch, a structural obstacle that prevents naïve conditional average treatment efect (CATE) estimation in recommendation systems with holdback groups, and propose a dualthreshold targeting policy as a principled resolution.

• We demonstrate how an existing production multi-task recommendation model can be extended to a causal Deep Twin Network architecture [17] using already-collected holdback data, requiring no new infrastructure.

• We validate this approach at production scale: a live A/B test on millions of Spotify users demonstrates a 7% reduction in recommendation impressions with no significant impact on recommended content consumption.

• We show that joint training with holdback data improves calibration of the treated head relative to the production baseline, and argue this as evidence that causal models learn more generalisable representations—because they must account for the interaction between the recommendation and user behaviour, rather than user-content afinity alone.

## 2 Problem Setting

We consider a recommendation system that, for each (user, content) pair, makes a binary decision: deliver a recommendation (� = 1) or withhold it (� = 0). Let x ∈ X denote the feature vector for a (user, content) pair. We adopt the potential outcomes framework [13]: let $Y ( 1 )$ denote the stream indicator if the recommendation is shown, and $Y ( 0 )$ the stream indicator if the recommendation is withheld. The causal estimand of interest is the Conditional Average Treatment Efect (CATE),

![](images/ceaa69d8430972b759dc1541c3626d4478c22fee5ac05ef22e386ad658b040e2.jpg)  
Figure 1: Recommendation surfaces in Spotify app where the model operates. Recommendation occupies slot indicated by the dashed red box.

$$
\tau ( \mathbf { x } ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid \mathbf { x } ] ,\tag{1}
$$

which quantifies how much the recommendation increases the probability of a stream for a given (user, content) pair. A user with $\tau ( \mathbf { x } ) \approx 0$ is a sure thing — they will stream with or without the recommendation, so the impression is non-incremental.

The system runs a randomised holdback experiment in which a small fraction of eligible (user, content) pairs are withheld recommendations at random. Let $\mathcal { D } _ { 1 } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \}$ denote the set of treated examples (recommendation shown, $T = 1 )$ and $\mathcal { D } _ { 0 } = \{ ( \mathbf { x } _ { j } , y _ { j } ) \}$ denote the set of holdback examples (recommendation withheld, $T =$ 0). Because holdback assignment is randomised, $T \perp \perp ( Y ( 0 ) , Y ( 1 ) )$ | x, so the holdback observations provide unbiased samples from the distribution of �(0).

We define the two model outputs that will be central throughout:

$$
\hat { p } _ { 1 } ( \mathbf { x } ) = \hat { \mathbb { E } } \big [ Y ( 1 ) \mid \mathbf { x } \big ] = \hat { \mathbb { E } } \big [ Y \mid T = 1 , \mathbf { x } \big ] ,\tag{2}
$$

$$
\hat { p } _ { 0 } ( \mathbf { x } ) = \hat { \mathbb { E } } \big [ Y ( 0 ) \mid \mathbf { x } \big ] = \hat { \mathbb { E } } \big [ Y \mid T = 0 , \mathbf { x } \big ] .\tag{3}
$$

In standard causal estimation settings, one estimates CATE as $\hat { \tau } ( \mathbf { x } ) = \hat { p } _ { 1 } ( \mathbf { x } ) - \hat { p } _ { 0 } ( \mathbf { x } ) \left[ 4 , 1 5 , 1 6 \right]$ . As we discuss in Section 3, this is not valid here due to a structural asymmetry in how $Y ( 1 )$ and � (0) are defined in the production setting.

## 3 The Attribution Mismatch

The obstacle to naïve CATE estimation in this setting is a structural asymmetry in attribution: the outcome �(1) for treated users and the outcome �(0) for holdback users are defined using diferent temporal windows, and are therefore not on the same scale.

Treated attribution. When a recommendation is shown and the user streams the content within a short direct-response window (on the order of minutes to hours), the stream is attributed to the recommendation. This window is narrow by design—it isolates behavioral responses that are plausibly caused by the impression.

Holdback attribution. When a recommendation is withheld, there is no direct impression to respond to. Organic discovery is a slower process; the user might encounter the content through other surfaces over the following days. To capture this, holdback streams are attributed if they occur within a two-day window from the moment the recommendation would have been shown. This two-day window is necessary to produce a meaningful estimate of organic behavior, but it difers from the treated attribution window.

Why subtraction fails. In standard causal efect estimation [16], $\hat { p } _ { 1 }$ and $\hat { p } _ { 0 }$ are estimated from treated and control observations, and CATE is recovered as their diference. This is valid only when both quantities measure the same outcome. Here they do not: the treated head is calibrated against streams within a narrow directresponse window, while the holdback head measures streams within a two-day organic window. Their diference is not an estimate of �(x); it is the diference of two probabilities with diferent outcome definitions, and carries no clean causal interpretation.

Distributional mismatch An additional practical consideration is the distributional diference between $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 0 }$ . In production, there is approximately a 30% discrepancy between backend recommendation servings (the events that generate holdback labels) and client-side impressions (the events that generate treated labels), owing to latency and client-side rendering diferences. The holdback and treated sets are therefore not drawn from identical distributions over x, which is an additional reason to avoid treating $\hat { p } _ { 0 } ( \mathbf { x } )$ as a direct counterfactual for $\hat { p } _ { 1 } ( \mathbf { x } )$

What the scores can do. Although $\hat { p } _ { 1 } ( \mathbf { x } ) - \hat { p } _ { 0 } ( \mathbf { x } )$ is invalid as a CATE estimator, each score individually carries well-defined decision-relevant signal. $\hat { p } _ { 1 } ( \mathbf { x } )$ high means the recommendation is likely to drive a stream. $\hat { p } _ { 0 } ( \mathbf { x } )$ high means the user will stream the content organically, regardless of the recommendation. The dual-threshold policy in Section 4 exploits this structure directly: it uses both scores as independent decision inputs.

## 4 Causal Recommendation Model

The production recommendation model is a multi-task shared-trunk neural network [1], illustrated in Figure 2. A deep shared trunk maps the (user, content) feature vector x to a shared representation, from which task-specific heads predict various engagement outcomes (streams, clicks, etc.). The primary head, responsible for targeting decisions, is trained to predict �ˆ (x), the probability that the user streams the content after a recommendation is shown. This model is trained exclusively on treated examples $\mathcal { D } _ { 1 }$ , and therefore has no access to counterfactual information.

## 4.1 Deep Twin Network Extension

To capture counterfactual information, we extend the baseline model to a Deep Twin Network [17] architecture by adding a holdback head that learns to predict $\hat { p } _ { 0 } ( \mathbf { x } )$ , illustrated in Figure 3. The shared trunk is retained; the holdback head is a new output branch trained on holdback observations ${ \mathcal { D } } _ { 0 } .$ This is structurally equivalent to a DragonNet [16] with two outcome heads—one per treatment arm—but without a propensity head, since holdback assignment is randomised and the propensity is known.

![](images/c7c0c8182ad6a1ca6f0461a02aabee73066932461447d74de828ad927b141c3e.jpg)  
Figure 2: The baseline production recommendation model: a shared-trunk neural network with task-specific heads, trained exclusively on examples where a recommendation was shown. The primary head predicts $\hat { p } _ { 1 } ( \mathbf { x } )$

![](images/638f2b01349a1980185496e4f539f44757ee252f0f39c688b8b7a621e41f0168.jpg)  
Figure 3: The causal recommendation model: the baseline extended with a holdback head. Treated examples update only the treated head and the shared trunk; holdback examples update only the holdback head and the shared trunk. The shared trunk benefits from both treatment conditions, learning representations that capture the interaction between the recommendation and user behaviour.

Training uses a partitioned loss in which treated examples contribute only to the treated head loss, and holdback examples con tribute only to the holdback head loss, but both sets of gradients flow through the shared trunk:

$$
\mathcal { L } = \frac { 1 } { | \mathcal { D } _ { 1 } | } \sum _ { i \in \mathcal { D } _ { 1 } } \ell _ { \mathrm { b c e } } ( y _ { i } , \hat { p } _ { 1 } ( \mathbf { x } _ { i } ) ) + \frac { 1 } { | \mathcal { D } _ { 0 } | } \sum _ { j \in \mathcal { D } _ { 0 } } \ell _ { \mathrm { b c e } } ( y _ { j } , \hat { p } _ { 0 } ( \mathbf { x } _ { j } ) ) ,\tag{4}
$$

where $\ell _ { \mathrm { b c e } }$ is the binary cross-entropy loss. Because both terms backpropagate through the shared trunk, the shared representation is shaped jointly by treated and holdback supervision, while each head is updated only by its own relevant examples. This partitioned structure is important: allowing holdback gradients to update the treated head (or vice versa) would conflate the two outcome definitions, which, as established in Section 3, correspond to diferent attribution windows and cannot be directly compared.

## 4.2 Dual-Threshold Targeting Policy

Because $\hat { p } _ { 1 } ( \mathbf { x } ) - \hat { p } _ { 0 } ( \mathbf { x } )$ is not a valid CATE estimator in this setting, we instead exploit the two scores as independent binary decision inputs. We serve a recommendation only if both conditions hold:

$$
\begin{array} { r } { \pi ( \mathbf { x } ) = \mathbf { 1 } [ \hat { p } _ { 1 } ( \mathbf { x } ) \geq \theta _ { 1 } ] \cdot \mathbf { 1 } [ \hat { p } _ { 0 } ( \mathbf { x } ) \leq \theta _ { 0 } ] , } \end{array}\tag{5}
$$

where $\theta _ { 1 } , \theta _ { 0 } \in [ 0 , 1 ]$ are tunable threshold parameters. The first condition ensures that the recommendation is likely to drive a stream. The second condition removes “always-takers”—users for whom organic discovery is likely, rendering the recommendation non-incremental. Together, the two conditions identify users for whom the recommendation is both efective (likely to produce a stream) and necessary (the user would not stream otherwise).

The threshold $\theta _ { 0 }$ provides a direct, interpretable lever on the eficiency–reach trade-of: increasing $\theta _ { 0 }$ withholds more impressions from users with higher organic stream probability, reducing impression volume at the cost of potentially missing some incremental conversions; decreasing $\theta _ { 0 }$ toward zero approaches the baseline policy of targeting by $\hat { p } _ { 1 } ( \mathbf { x } )$ alone. Thresholds are set ofline against randomized data to meet impression reduction targets while satisfying non-inferiority constraints on consumption metrics.

## 5 Related Work

Uplift modelling and CATE estimation. The goal of identifying incremental users is closely related to uplift modelling [6, 9, 12], which seeks to estimate the individual-level treatment efect. Neural approaches to CATE estimation include TARNet and CFR [15], DragonNet [16], Deep Twin Networks [17], and a range of metalearning approaches [4, 5]. The present work is distinguished by its production setting: we operate on a live system with tens of millions of users, with the additional structural complexity that treated and holdback outcomes are not measured on the same scale — a challenge not addressed in the benchmark-dataset literature for these methods. Recent theoretical work has further studied the fundamental dificulty of validating causal models against experimental data [7], underscoring the importance of careful experimental design in production deployments such as ours.

Causal recommendation systems. The use of causal reasoning to improve recommendation has been studied through inverse propensity score weighting for debiasing [14], counterfactual ofline evaluation [8], and treatment-aware modelling [10]. Contrastive approaches to learning causally relevant treatment representations have also been explored [3]. The dominant thread in this literature targets debiasing click and conversion prediction; our work difers in that we explicitly target incrementality rather than correcting for selection bias in the observation of engagement.

Impression eficiency. The problem of spending impressions on always-takers is well known in direct marketing [6, 12] and has begun to attract attention in recommendation [2, 11]. The contribution of this work is to demonstrate that a production-scale incremental recommendation system can be built from existing holdback infrastructure, and that doing so achieves meaningful im pression savings without degrading the recommendation impact.

## 6 Experiments

We evaluate the dual-threshold policy in a live A/B test on Spotify’s Home recommendation surface (Figure 1). The test was run at production scale with millions of users. Operating at this scale required the causal model to match the latency and throughput constraints of the production serving infrastructure.

The experiment comprised three arms:

• Control: The production recommendation model, targeting by treated-stream score $\hat { p } _ { 1 } ( \mathbf { x } )$ trained on $\mathcal { D } _ { 1 }$ only.

• Treatment-Model: The causal model of Section 4.1, targeting by $\hat { p } _ { 1 } ( \mathbf { x } )$ using the filtering for just that arm: $1 [ \hat { p } _ { 1 } ( \mathbf { x } ) \geq \theta _ { 1 } ]$ . This arm isolates the efect from training the shared trunk on $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 0 }$

• Treatment-Causal: The causal model with the dualthreshold policy �(�) of Section 4.2.

The primary comparison of interest is Treatment-Model versus Treatment-Causal, which isolates the efect of the dual-threshold policy while holding the model architecture and training data fixed.

We track two metrics over two weeks in the test. Recommendation impressions per user: the primary eficiency metric. Consumption minutes of recommended content: the primary impact metric, with non-inferiority required.

Table 1 summarises the outcomes of the Treatment-Model versus Treatment-Causal comparison.

Table 1: A/B test: Treatment-Causal vs. Treatment-Model.
<table><tr><td>Metric</td><td>Δ</td><td>95%CI</td></tr><tr><td>Recommendation impressions</td><td>-7.1%</td><td>[-7.2%, -6.9%]</td></tr><tr><td>Recommended content consumption</td><td>-0.37%</td><td>[−0.86%, +0.21%]</td></tr></table>

The dual-threshold policy reduces recommendation impressions by 7% relative to targeting with the treated-stream score alone, while producing no statistically significant change in the consumption of recommended content or in listener satisfaction guardrails.

The magnitude of the impression reduction is informative. It implies that approximately 93% of the impressions delivered by the production recommendation model are already incremental: users who would not have discovered the content without the recommendation. This is a consequence of the organic discovery landscape on the platform—for the content types featured on the Home surface, the recommendation itself is one of very few reliable discovery mechanisms available to users who do not already closely follow the relevant artists or podcasters. The dual-threshold policy successfully identifies and removes the non-incremental 7%, demonstrating precise targeting and removal of always-takers.

## 7 Calibration and Generalisation

A benefit of the causal architecture is improved calibration of the treated head $\hat { p } _ { 1 } ( \mathbf { x } )$ . Calibration matters for threshold-based poli cies: if the model’s predicted probabilities do not accurately reflect empirical stream frequencies, the threshold parameters $\theta _ { 1 }$ and $\theta _ { 0 }$ lose their interpretation and the policy cannot be reliably tuned.

Figure 4 shows calibration curves for the treated head and the holdback head. The causal treated head (orange) tracks the diagonal more closely than the production baseline (blue), with the most pronounced improvement at higher predicted probabilities.

![](images/669f33895cbdead25ca30ea06993ca701cf1c2471ae4021805cc11cc6ff6105d.jpg)

![](images/f33321d169c1eaa87eefc23c95f14b4a8e33fd8db9d041b6a16069e104d624a0.jpg)  
Figure 4: Calibration curves for (left) the treated head and (right) the holdback head. Left: the causal treated head (orange) tracks the diagonal more closely than the production baseline (blue), with the largest improvement at higher predicted probabilities. Right: the holdback head is wellcalibrated, with minor deviations at the tail attributable to the smaller holdback sample size.

We hypothesise that this calibration improvement reflects a genuine generalisation benefit of the causal architecture. A model trained exclusively on treated data learns a mapping from usercontent features to stream probability conditional on a recommendation being shown. This conflates two distinct factors: the user’s afinity for the content, and the efect of the recommendation itself. A model that cannot disentangle these factors will tend to over-predict for high-afinity users (who would stream regardless).

Joint training with holdback data, through the shared trunk, forces the model to represent both afinity and recommendation sensitivity—the degree to which the recommendation changes the user’s behavior. The holdback head must predict organic stream probability accurately, and the shared trunk must therefore encode the features that predict behaviour in the absence of a recommendation. This richer representation generalises better across treatment conditions: the treated head, sharing the trunk, inherits a more complete picture of user-content afinity.

This observation connects to a broader principle: causal models, by virtue of being trained on data from multiple treatment conditions, learn representations that are more robust to distribution shift than models trained on a single observational regime [3, 7]. The calibration improvement here is an empirical instance of this generalisation benefit in a production recommendation setting.

The holdback head (Figure 4, right) is also well-calibrated, with minor deviations at higher predicted probabilities attributable to the smaller sample size of holdback data relative to treated data.

## 8 Conclusion

We have shown how an existing production recommendation model with holdback infrastructure can be transformed into a causal incremental targeting system. In a production-scale A/B test on millions of Spotify users, this model reduces recommendation impressions by 7% with no statistically significant reduction in recommended content consumption or user satisfaction—increasing incremental impact. We further demonstrate that joint training with holdback data improves model calibration, arguing this as evidence that the causal model learns more generalisable representations.

## Acknowledgments

The authors thank the Kipp team at Spotify for infrastructure support and experimental design, and the Advanced Causal Inference lab at Spotify for helpful discussions.

## References

[1] Rich Caruana. 1997. Multitask Learning. Machine Learning 28, 1 (1997), 41–75.

[2] Olivier Chapelle, Eren Manavoglu, and Romer Rosales. 2014. Simple and Scalable Response Prediction for Display Advertising. In ACM Transactions on Intelligent Systems and Technology, Vol. 5. 61.

[3] Oriol Corcoll, Athanasios Vlontzos, Michael O’Riordan, and Ciarán M. Gilligan-Lee. 2026. Contrastive representations of structured treatments. npj Artificial Intelligence 2 (2026), 49. doi:10.1038/s44387-026-00105-2

[4] Alicia Curth and Mihaela van der Schaar. 2021. Nonparametric Estimation of Heterogeneous Treatment Efects: From Theory to Learning Algorithms. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, Vol. 130. PMLR, 1810–1818. arXiv:2101.10943

[5] Alicia Curth and Mihaela van der Schaar. 2021. On Inductive Biases for Heteroge neous Treatment Efect Estimation. In Advances in Neural Information Processing Systems, Vol. 34. arXiv:2106.03765

[6] Floris Devriendt, Darie Moldovan, and Wouter Verbeke. 2018. A Literature Survey and Experimental Evaluation of the State-of-the-Art in Uplift Modeling: A Stepping Stone Toward the Development of Prescriptive Analytics. Big Data 6, 1 (2018), 13–35.

[7] Jake Fawkes, Michael O’Riordan, Athanasios Vlontzos, Oriol Corcoll, and Ciarán M. Gilligan-Lee. 2025. The Hardness of Validating Observational Studies with Experimental Data. In Proceedings of the International Conference on Artificial Intelligence and Statistics. arXiv:2503.14795

[8] Alexandre Gilotte, Clément Calauzènes, Thomas Nedelec, Alexandre Abraham, and Simon Dollé. 2018. Ofline A/B Testing for Recommender Systems. In Proceedings ofthe 11th ACM International Conference on Web Search and Data Mining. 198–206.

[9] Pierre Gutierrez and Jean-Yves Gérardy. 2017. Causal Inference and Uplift Modeling: A Review of the Literature. JMLR: Workshop and Conference Proceedings 67 (2017), 1–13.

[10] Dawen Liang, Laurent Charlin, and David M. Blei. 2016. Causal Inference for Recommendation. In UAI2016 Workshop on Causation: Foundation to Application.

[11] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. 2018. Entire Space Multi-Task Model: An Efective Approach for Estimating Post-Click Conversion Rate. In Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval. 1137–1140.

[12] Nicholas J. Radclife and Patrick D. Surry. 1999. Diferential Response Analysis: Modeling True Responses by Isolating the Efect of a Single Action. Proceedings ofCredit Scoring and Credit Control VI (1999).

[13] Donald B. Rubin. 1974. Estimating causal efects of treatments in randomized and nonrandomized studies. Journal ofEducational Psychology 66, 5 (1974), 688–701.

[14] Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as Treatments: Debiasing Learning and Evaluation. In Proceedings ofthe 33rd International Conference on Machine Learning, Vol. 48. PMLR, 1670–1679.

[15] Uri Shalit, Fredrik D. Johansson, and David Sontag. 2017. Estimating individual treatment efect: generalization bounds and algorithms. In Proceedings ofthe 34th International Conference on Machine Learning, Vol. 70. PMLR, 3076–3085. arXiv:1606.03976

[16] Claudia Shi, David M. Blei, and Victor Veitch. 2019. Adapting Neural Networks for the Estimation of Treatment Efects. In Advances in Neural Information Processing Systems, Vol. 32. arXiv:1906.02120

[17] Athanasios Vlontzos, Bernhard Kainz, and Ciarán M. Gilligan-Lee. 2023. Estimating categorical counterfactuals via deep twin networks. Nature Machine Intelligence 5 (2023), 159–168. doi:10.1038/s42256-023-00611-x