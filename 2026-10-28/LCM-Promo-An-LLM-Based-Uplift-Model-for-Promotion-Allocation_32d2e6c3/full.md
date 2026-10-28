# LCM-Promo: An LLM-Based Uplift Model for Promotion Allocation<sup>⋆</sup>

Nesta Midavaine<sup>1</sup>, Floris Fok<sup>1</sup>, Paulo Mendes<sup>2</sup>, Joao Silva<sup>2</sup>, Önder Akaçık<sup>1</sup>, Dimas Lima<sup>2</sup>, and Zülküf Genç<sup>1</sup>

<sup>1</sup> Prosus, Amsterdam, The Netherlands

{nesta.midavaine,floris.fok,onder.akacik,zulkuf.genc}@prosus.com <sup>2</sup> iFood, São Paulo, Brazil

{paulo.mendes,guedes.joao,dimas.lima}@ifood.com.br

Abstract. We present LCM-Promo, an LLM-based uplift model for promotion allocation. LCM-Promo is a task-specific Large Commerce Model (LCM): a finetuned LLM trained on transactional commerce data to predict user intent and promotion response. It learns monotonic promotionresponse curves from production logs by interpreting single-token outputs as conversion-threshold probabilities, while conditioning on session context such as day and delivery shift. In a production A/B test on churn reactivation, LCM-Promo increased conversion by 1% and reduced promotion cost by 6% relative to the Segment-EMA production baseline.

Keywords: Large Commerce Models · Large Language Models · Uplift Modelling · Promotion Targeting

## 1 Introduction

LCM-Promo is one application of a broader efort at Prosus to build Large Commerce Models (LCMs): domain-specialized foundation models trained on user, merchant, and product interactions across e-commerce platforms. Where general-purpose LLMs reason about language, LCMs reason about commerce: user intent, merchant dynamics, market context, and responses to pricing and promotion. The LCM stack supports downstream tasks through direct serving and task-specific specialization. In this work we focus on promotion allocation.

Promotion targeting can be framed as uplift modelling: estimating how much more likely a user is to convert when shown a specific promotion than when shown no promotion, conditional on user and session features [6]. We use a finetuned LLM as an uplift model for promotion targeting in e-commerce. We deploy LCM-Promo at iFood, where promotions are ephemeral and context-dependent, varying by day, delivery shift, product category, and discount mechanic. This makes a prompt-based model a natural fit: user representations, session context, and discount options can be serialized directly into the input.

LCM-Promo predicts a single token over discount identifiers, which we interpret at decoding time as a distribution over user conversion thresholds. This yields a monotonic promotion-response curve by construction, usable both for budget-constrained allocation and for ofline uplift evaluation. In a live $\mathrm { A } / \mathrm { B }$ test on churn reactivation, LCM-Promo increased conversion by 1% while reducing promotion cost by 6% relative to the Segment-EMA production baseline, and was subsequently rolled out to all churn users. Ofline, it is competitive with an uplift random forest trained on randomized control data.

## 2 Method

We formulate promotion response estimation as single-token prediction. Each prompt contains a system prompt, session context, an LCM-generated user representation, and discount options mapped to alphabetic identifiers. The LCM representation model converts raw historical logs into structured behavioural summaries.

LCM-Promo uses a 4B parameter dense model as the base model architecture and predicts a single discount option through its identifier. Let $\mathcal { D } = \{ d _ { 0 } , \ldots , d _ { K } \}$ be ordered by increasing discount depth, let $a _ { k }$ be the identifier assigned to $d _ { k } .$ and let $a _ { \emptyset }$ denote no purchase. The model is trained with cross-entropy on the observed conversion-discount label, using $a _ { \emptyset }$ for non-converters. At inference time, we restrict the next-token distribution to these valid identifiers and renormalize:

$$
q _ { i } ( x ) = { \frac { p _ { \theta } ( a _ { i } \mid x ) } { \sum _ { j = 0 } ^ { K } p _ { \theta } ( a _ { j } \mid x ) + p _ { \theta } ( a _ { \emptyset } \mid x ) } } , \qquad i \in \{ 0 , \ldots , K , \emptyset \} .\tag{1}
$$

We interpret this as a probability mass function (PMF) over a latent conversion threshold $T ,$ where $T = d _ { k }$ means that $d _ { k }$ is the minimum discount at which the user would convert, and $T = \infty$ corresponds to no purchase under any available discount. In our churn-reactivation setting, users may see multiple ofers across one month but typically convert at most once, so the observed conversion discount is used as a proxy for this threshold. The predicted conversion probability for ofer $d _ { k }$ is the cumulative response curve,

$$
r _ { k } ( x ) = P ( T \leq d _ { k } \mid x ) = \sum _ { j = 0 } ^ { k } q _ { j } ( x ) ,\tag{2}
$$

which is monotone non-decreasing by construction. The no-purchase mass allows $r _ { K } ( x ) < 1$ . Model-implied uplift relative to no discount follows directly.

We also evaluate a waterfall augmentation that encodes the threshold ordering in the labels. If a user converts at $d _ { k }$ , all discounts $d _ { j } \geq d _ { k }$ are labelled positive; non-converters are positive only for $a _ { \emptyset }$ . These labels are smoothed into a soft target and trained with cross-entropy [2].

## 3 Experiments

## 3.1 Online Evaluation

We compare against Segment-EMA, iFood’s prior production policy, which forecasts orders per segment-promotion cell using an exponentially weighted moving average. Because each new discount, segment, or context requires exploration within the cell, Segment-EMA limits segmentation granularity. We conducted a one-month production A/B test on churn reactivation users across Brazil. Users were randomly assigned to a 90% control group running Segment-EMA and a 10% treatment group running LCM-Promo. Both arms used the same global budget constraint and the same linear-optimization allocator [1]. Only the response model difered.

LCM-Promo runs inference twice daily, for lunch and dinner. Users are grouped into city, session, and lifecycle clusters, and mean response curves are estimated by random sampling with bootstrap-determined sample sizes. Productionalization was made feasible by this sampling. Prompts use precomputed LCM-generated user representations and are submitted to a self-hosted vLLM deployment of the 4B parameter model via BESH [4], our open-source batch endpoint

Table 1. Online A/B test. Lifts relative to Segment-EMA baseline.
<table><tr><td>Metric</td><td>Relative lift</td></tr><tr><td>Conversion</td><td>+1%</td></tr><tr><td>Promotion cost</td><td>-6%</td></tr></table>

for self-hosted GPUs, running 1 hour daily on 8 H200s.

Because delivery shift and other session contexts are prompt inputs rather than separate Segment-EMA forecasting cells, LCM-Promo expanded allocation from about 50 segments to over 1,000 clusters and from 10 wallet configurations to over one million promotion combinations. Table 1 shows a 1% relative conversion lift and 6% lower total promotion cost. The system was subsequently rolled out to all churn users.

## 3.2 Ofline Evaluation

Before deployment, we evaluated LCM-Promo on randomized controlled trial (RCT) data with a fixed promotion. We compare against a Uplift Random Forest implemented with causalml [5,3], a strong baseline trained on previous months of RCT data. LCM-Promo is trained on historical production logs and is only evaluated on the RCT. This fixed-promotion benchmark is narrower than the production setting, which spans multiple promotion depths, session contexts, and wallet configurations. We therefore treat it as validation for online testing rather than evidence of ofline superiority.

Table 2 shows that LCM-Promo is comparable on AUUC and Qini and below the random forest on Uplift@30. The waterfall variant performs worse, likely because its multi-label soft targets conflict with the LLM’s next-token pretraining objective.

Table 2. Ofline uplift metrics on churn users.
<table><tr><td>Model</td><td>AUUC</td><td>Qini</td><td>Uplift@30</td></tr><tr><td>Uplift Random Forest</td><td> $0 . 0 5 7 8 \pm 0 . 0 0 6 4$ </td><td> $0 . 0 4 6 6 \pm 0 . 0 0 5 3$ </td><td> $\mathbf { 0 . 1 7 0 8 \pm 0 . 0 0 6 4 }$ </td></tr><tr><td>LCM-Promo (post-processing)</td><td> $\mathbf { 0 . 0 6 0 8 \pm 0 . 0 0 3 2 }$ </td><td> $\mathbf { 0 . 0 4 9 0 \pm 0 . 0 0 2 8 }$ </td><td> $0 . 1 6 2 6 \pm 0 . 0 0 3 5$ </td></tr><tr><td>LCM-Promo (waterfall)</td><td> $0 . 0 2 7 8 \pm 0 . 0 0 3 6$ </td><td> $0 . 0 2 2 6 \pm 0 . 0 0 2 9$ </td><td> $0 . 1 2 0 8 \pm 0 . 0 0 2 6$ </td></tr></table>

## 4 Conclusion

We presented LCM-Promo, a task-specific Large Commerce Model for promotion allocation. It learns promotion-response curves from production logs and converts them into monotonic uplift estimates through threshold-based PMF decoding. In a production A/B test on iFood churn reactivation, LCM-Promo increased conversion by 1% and reduced promotion cost by 6% relative to Segment-EMA, while expanding the allocation space by orders of magnitude. The system was subsequently rolled out to all churn users.

These results are limited to one lifecycle stage and one platform. Extending the approach to other lifecycle stages and commerce platforms is ongoing work. More broadly, LCM-Promo shows how Large Commerce Models can be specialized into compact production models for commerce decisions such as targeting, and merchant operations.

Disclosure of Interests. The authors have no competing interests to declare.

## References

1. Albert, J., Goldenberg, D.: E-commerce promotions personalization via online multiple-choice knapsack with uplift modeling. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. pp. 2863–2872 (2022)

2. Bruch, S., Wang, X., Bendersky, M., Najork, M.: An Analysis of the Softmax Cross Entropy Loss for Learning-to-Rank with Binary Relevance. In: Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval. pp. 75–78. ICTIR ’19, Association for Computing Machinery, New York, NY, USA (Sep 2019). https://doi.org/10.1145/3341981.3344221

3. Chen, H., Harinen, T., Lee, J.Y., Yung, M., Zhao, Z.: Causalml: Python package for causal machine learning. arXiv preprint arXiv:2002.11631 (2020)

4. Fok, F., Aguiar, M., Borges, G., van der Boor, P., Duncan, M.: BESH: An open source batch endpoint for unlocking 4x utilisation of your self hosted GPU cluster. Prosus AI Tech Blog (Oct 2025)

5. Sołtys, M., Jaroszewicz, S., Rzepakowski, P.: Ensemble methods for uplift modeling. Data mining and knowledge discovery 29, 1531–1559 (2015)

6. Zhang, W., Li, J., Liu, L.: A Unified Survey of Treatment Efect Heterogeneity Modelling and Uplift Modelling. ACM Comput. Surv. 54(8), 162:1–162:36 (Oct 2021). https://doi.org/10.1145/3466818