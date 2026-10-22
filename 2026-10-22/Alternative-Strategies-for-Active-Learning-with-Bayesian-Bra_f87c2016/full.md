# Alternative Strategies for Active Learning with Bayesian Bradley Terry Model

Saman Jayasinghea, Sean Waltona, Alma Rahatb, Andy Grayc, Chris Wheadond

, aSwansea University, United Kingdom

bLoughborough University, United Kingdom

cBath Spa University, United Kingdom

d No More Marking Ltd, United Kingdom

October 22, 2026

## Abstract

Comparative judgement uses pairwise comparisons to rank items, but recovering accurate rankings from a limited number of judgements depends on both the inference method and the pairs selected. We developed a Bayesian Bradley-Terry model (BBTM) that uses Polya-Gamma data augmentation for efficient Gibbs sampling and quantifies uncertainty in latent abilities and ranks. We propose three active learning strategies based on the knowledge gradient (KG) principle. We define our acquisition functions using Kendall's tau distance, posterior rank entropy, and posterior separation. Approximate one-step updates to posterior moments reduce the computational cost of evaluating candidate pairs. We also introduce Posterior Separation Reliability (PSR), which measures the spread of posterior mean abilities relative to posterior uncertainty. It provides an internal indicator of ranking reliability without requiring a reference ranking. In synthetic experiments covering different item counts, comparison budgets, and noise levels, BBTM generally recovers ranking more accurately than preference based Bayesian comparative judgement (PBCJ). At low to moderate noise levels, pairwise entropy pair selection strategy performs best in ranking recovery. However, at the highest noise levels, the KG strategy based on posterior separation produces better point rankings. PSR increases alongside improvements in ranking accuracy as comparisons accumulate. Robustness experiments identify the proportion of informative outcomes needed to meet specified accuracy targets within a fixed comparison budget. In moderate noise setting, replacing 30% of outcomes with random choices has little effect on ranking performance. These findings demonstrate the value of Bayesian inference and active learning pair selection recovering ranks from limited, noisy paired comparisons.

## 1 Introduction

Measuring the reliability of responses for complex, open-ended assessments, such as extended writing essays, is a key challenge in educational assessments. Traditional marking procedures like rubrics have been inconsistent for these tasks. Standards can vary between assessors, and even a single assessor may show some inconsistency due to the higher cognitive workload of traditional marking. Comparative Judgement (CJ), therefore, emerges as an alternative to address those drawbacks in traditional marking [48]. In the CJ setting, assessors are given a pair of assessments to identify the best piece of work in the given pair rather than assign an absolute mark. CJ is recognised as a reliable, scalable approach in educational assessment since humans are better at relative judgment than at absolute judgment [42, 33]. A series of pairwise comparisons is utilised to obtain the final rank of each item, obtained from the underlying quality score which are estimated from the comparison outcomes. The Bradley-Terry (BT) model [4] is the classical foundation for this inference, which links the outcome probabilities to their latent abilities.

The BT model uses the maximum likelihood estimation type point estimation technique to estimate the latent abilities of items [4]. Efficient algorithms have been developed [47, 10] to derive a single ranking for point estimates; however, they do not provide an estimate of uncertainty around these estimates. It is crucial to know how certain these rankings are in the high-stakes context of educational assessments. Bayesian inference provides these uncertainty measures around estimates as it gives posterior distributions over items' latent abilities instead of point estimates. Since the BT likelihood has a form of logistic, it is hard to sample the posterior directly due to the non-conjugacy with the Gaussian prior. Polya-Gamma data augmentation [43] converts this logistic likelihood into conditionally Gaussian form. This augmentation establishes conjugation and, more importantly it enables efficient Gibbs sampling. Recently, Bayesian BT has been used to compare multiple machine learning algorithms [53], but to the best of our knowledge, this Bayesian BT with Polya-Gamma data augmentation has not been applied to CJ in an educational setting or to active learning for pair selection.

There are (2) possible pairs for N items. As N increases, the possible number of pairs increases quadratically. It is not practical to compare all these pairs, because human judgments are expensive. Consequently, it is essential to identify the most informative pairs that provide the highest information about the ranking order, using AL methods. Adaptive comparative judgement (ACJ) [42] was the first such implementation in an educational CJ setting, and more recently, entropy-driven AL pair selection has been developed [26, 27]. The Knowledge Gradient (KG) methods have been established as a highly effective acquisition function in areas such as optimal learning, ranking and selection (RS), sequential experimental design [20, 18, 41, 28]. These KG methods have been adapted to the pairwise setting, but these implementations do not recover the ranking of all items; instead, they recover top-k selection or best arm identification [29, 38]. Further, these KG methods have not been tailored to the CJ framework and are difficult to scale for a higher number of items, since exact KG computation depends on Monte Carlo (MC) sampling.

The Scale Separation Reliability (SSR) measure is used to assess the reliability of CJ results [36, 51]. SSR is a point estimation based measure and it does not consider posterior uncertainty. Concerns have been emerging around this SSR measure in the CJ community [50]. It is important to have a reliability measure that can be computed directly from the posterior distribution, without considering the ground truth. Under adaptive pair selection, estimates can be biased, because pair selection depends on outcomes. So, observations are not independent. Vojnovic et al. [52] studied the convergence properties of BT model under a non-adaptive (i.i.d.) setting. Bernstein-von Mises (BvM) theorem and Bayesian posterior consistency exist for non-i.i.d. observations [23, 49] however, they have not been directly applied to the adaptive pairwise comparison setting. A proof that guarantees the convergence of Bayesian BT model under the adaptive pair selection is therefore still lacking. Since human judgments are expensive, recovering an optimal ranking entirely from human judgments would incur a high cost. Trade-off between human judgments and cheaper random coin toss outcomes has not been examined.

No existing Bayesian approach to CJ combines all of the following within a single framework: (i) theoretical guarantees of convergence under active learning pair selection; (ii) a computationally efficient KG pair selection that enables the full ranking recovery; (iii) a reliability measure that can be computed directly from the posterior, without access to the ground truth; (iv) a systematic study of the robustness of ranking recovery under imperfect/noisy judgments. We address these gaps through the introduction the Bayesian Bradley Terry Model (BBTM) - a model which integrates both theoretical foundations and practical pair selection.

Our main contributions can be listed as follows:

• A hierarchical Bayesian version of the BT model with Polya-Gamma data augmentation which enables efficient Gibbs sampling. This model provides full posterior inference with uncertainty quantification on item strengths with sequential updating. The hierarchical structure yields better estimates for sparse or limited comparisons, due to shrinkage regularisation through borrowing strength [22, Chapter 12].

• Theoretical guarantees for the convergence of BBTM under adaptive pair selection via posterior consistency and BvM theorem. The convergence conditions are met using ε-greedy exploration.

• KG approaches to the AL pair selections. We develop three new KG acquisition functions specially tailored for the CJ framework: expected rank disruption, expected rank entropy reduction, and expected improvement in reliability.

• A computationally efficient KG based pair selection approach for the Bayesian BT CJ. We approximate one-step look-ahead values using Gaussian moment matching instead of nested Markov

Chain Monte Carlo (MCMC) sampling. This enables scalable KG based pair selection under multiple acquisition functions.

• A new reliability measure, Posterior Separation Reliability (PSR) which is computed from the posterior distribution. The PSR measure is also compatible with the traditional SSR.

• A near optimal mixture design which combines informative human judgments and random outcomes strategically. This reduces the need for costly human judgments while producing accurate ranking.

The remainder of this paper is organised as follows. Section 2 describes the background and reviews the existing literatures on CJ and Bayesian CJ frameworks. Section 3 presents the BBTM formulation, including the novel KG pair selection methods, fast KG via Gaussian moment updates, and the PSR measure. The Convergence proof of the BBTM is presented in Section 4. Section 5 discusses the experimental setup, the identification of the winning method, rank distribution recovery, PSR and the robustness of the model. Section 6 summarises the final conclusions

## 2 Background and Literature Review

## 2.1 Paired comparisons and Comparative Judgment

Paired comparison is a well-known and established approach for measuring latent qualities when absolute scoring is difficult. Thurstone's Law of Comparative Judgment [48] is the theoretical foundation of such models. Later, Luce [40] developed probabilistic choice models and provided a general framework for modeling comparative decisions. These early contributions show that pairwise judgments can be systematically converted into a latent scale under suitable probabilistic assumptions.

Over time, paired comparison methods have been used in a wide range of domains. In sports analytics, models based on paired outcomes are used to infer player or team strengths from match results [16]. In consumer research and sensory analysis, pairwise preferences are analysed to construct product rankings and understand decision behaviour [9]. In psychology and behaviour sciences, dominance structure and social hierarchies are often estimated from pairwise interactions (David, 1988). Wainer [53] has used paired comparisons to compare multiple Machine Learning algorithms on multiple data sets. Further, paired comparison has become central in Artificial Intelligence and Large Language model evaluation, where human judges compare model outputs to construct performance rankings [12].

In the education sector, Comparative Judgment (CJ) has gained increasing attention as an alternative to traditional rubric-based marking. Instead of assigning absolute marks, teachers compare two pieces of student work and select the better piece. Pollitt [42] argues that CJ simplifies the assessment task and improves decision consistency, with Adaptive CJ (ACJ) further enhancing efficiency through dynamic pairing. But, Bramley [5] warns that while ACJ is designed to select informative comparisons dynamically, the process produces biased statistics that artificially inflate the reliability scores. However, regardless of the domain, a common challenge remains: how to transform incomplete and noisy pairwise comparisons into a stable and reliable ranking

## 2.2 The Bradley Terry Model for Ranking Paired Comparisons

Among the available probabilistic models for paired comparison data, the Bradley Terry Model (BTM) [4] is one of the most widely used. In this model, each item is associated with a latent strength parameter. The probability that item i is preferred over item j is modelled as a logistic function of the difference between their strengths. This formulation is simple, interpretable, and computationally convenient.

Estimation of the BTM is commonly performed using Maximum Likelihood methods, and efficient algorithms such as MM algorithm have been developed [32]. Several extensions have also been proposed, including models that handle ties [44, 13], home advantage [1, Chapter 10], order effect [14], or incorporate object and subject specific covariates [9].

Despite its popularity, the classical BTM has some limitations. Its parameter estimation relies heavily on the structure of the comparison network. If the comparison graph is sparse or poorly connected, latent strength estimates may become unstable. Moreover, the standard frequentist approach provides point estimations and asymptotic standard errors, but it does not provide full uncertainty distributions over strengths or ranks. Finally, in sequential and adaptive settings such as CJ, the classical BTM framework does not directly provide guidance on which pair should be compared next.

These limitations are particularly important in CJ, where judgments are collected sequentially, and decisions must be justified in terms of reliability and fairness. Hence, Bayesian extensions to the BTM are highly desirable.

## 2.3 Bayesian Bradley Terry Model

Prior distributions are placed on the latent strengths, and inference is performed through the posterior distribution given the observed pairwise outcomes in the Bayesian version of BTM. Conjugate priors, such as Gamma priors for the positive skill parameters, have been used, enabling a tractable posterior and efficient computation [8]. They use a data augmentation strategy that enables efficient Gibbs sampling for full Bayesian inference and a simple Expectation-Maximisation (EM) algorithm for Maximum a Posteriori (MAP) estimation. Their work shows that Bayesian inference in generalized BTM can be implemented in a computationally scalable way, even for large comparison datasets.

Wainer [53] proposes a Bayesian Bradley-Terry (BBT) framework for comparing multiple machine learning (ML) algorithms across multiple datasets. He demonstrates that the Bayesian approach provides posterior probabilities that one ML algorithm outperforms another and allows the definition of Regions of Practical Equivalence (ROPE). This enables uncertainty quantification in aggregated rankings while avoiding traditional hypothesis testing. In applied statistical modelling, [46] introduce a Bayesian Spatial BTM for urban deprivation modelling by placing structured priors over latent strengths. These priors enable borrowing information across neighbouring regions and reduce the number of comparisons for stable inference. Further, [37] extend the BTM to allow time-varying latent strengths, using Gaussian process priors and MCMC sampling. These extensions allow the flexibility of the Bayesian framework to accommodate various structures.

Prior specification in the Bayesian BTM is crucial, as it directly affects invariance and interpretability properties. According to [54], a prior distribution for the BTM should satisfy these four desiderata: (i) Invariance under interchange of items, (ii). Invariance under interchange of winning and losing, (iii) Normalisability, (iv) Invariance under elimination of items. The analysis reveals that some commonly used priors, such as Jeffreys prior and certain conjugate priors, fail to satisfy one or more these desiderata. Furthermore, the paper identifies two families of prior distributions that satisfy all four: independent Gaussian distributions and the Type III Generalised Logistic distribution.

Bayesian logistic models have been computationally feasible due to recent developments, particularly Polya-Gamma data augmentation [43]. This enables efficient Gibbs sampling by introducing auxiliary variables that transform the logistic likelihood into a conditionally Gaussian form. It provides computationally stable and scalable posterior inference for the paired comparison models, particularly in hierarchical settings. Furthermore, Gibbs samplers show better mixing and require less tuning than Metropolis-Hastings methods

The advantages of BBTM are threefold. (i) The uncertainty can be directly quantified through posterior distributions rather than approximated through asymptotic theory. This allows us to compute credible intervals for latent strengths and to derive probabilities of rank positions. (ii) Prior distributions enable regularisation, which is highly useful when the number of comparisons is low or when the comparison graph is sparse. (iii) Sequential data collection has been supported by the Bayesian updating process, and it is essential in active learning in CJ environment.

In our work, we implement a Bayesian BTM (BBTM) using Polya-Gamma data augmentation, which enables efficient Gibbs sampling. The model aggregates binomial outcomes for each item pair and updates the posterior distribution iteratively as new comparisons are observed. This model design is suitable for CJ in education, where decisions are sequential and uncertainty quantification is necessary.

## 2.4 Bayesian Comparative Judgment in Education

Rasch based scaling [45] or frequentist BTM estimations have traditionally been used in CJ research in education [36]. While these approaches have shown practical usefulness [50][42], there has been growing interest in the Bayesian CJ framework in education [26]. Uncertainty in latent abilities can be explicitly modelled in Bayesian modelling, and this uncertainty is carried over to the rank estimates

The reliability of estimates is a central concern in educational assessment [36][51][50]. Decision makers require evidence that rankings are stable and defensible. Bayesian CJ offers posterior distributions over item strengths and ranks that provide additional interpretability for these estimates. This probabilistic perspective aligns well with the uncertain nature of human judgments.

Bayesian approaches to CJ are still a relatively new area, particularly when it comes to active learning and decision optimisation. Researchers have recently made considerable progress in combining Bayesian modelling with structured active learning strategies. Gray et al. [26] proposed a novel Bayesian approach to CJ (BCJ) that directly models pairwise preferences using Beta distributions rather than focusing on latent scores. They illustrate how to compute the entire predictive rank distribution for any item being assessed. Furthermore, they have developed a new active learning strategy to select next pair to evaluate based on entropy. By calculating predictive entropy of the preference densities, the algorithm actively selects the most informative pair to show the judges. They have shown that this will significantly improve the accuracy and efficiency over random or round robin selection methods.

Gray et al. [27], later extended the BCJ framework to handle multi-criteria assessments, bridging the gap between traditional CJ evaluations and detailed rubric-based grading. This approach models preferences across multiple independent learning outcomes (LOs) instead of relying on a single judgment. They used a weighted ensemble approach to aggregate these multi-dimensional comparisons into a reliable overall rank. Moreover, they have used multi-dimensional differential entropy to calculate the combined uncertainty across all LOs to intelligently select the next pair to evaluate under entropy driven pair selection method Gray et al. [25].

Although BCJ introduces posterior rank distributions and entropy-based active learning pair selection, existing approaches primarily focus on local uncertainty reduction rather than explicitly optimising global ranking accuracy or reliability objectives

## 2.5 Scale Separation Reliability (SSR)

In CJ, Scale Separation Reliability (SSR) is a measure used to evaluate the reliability of assessments $[ 5 1 ] [ 3 3 ]$ Assessors compare pairs of works in a CJ task, and their decisions are analysed using the BTM to create a ranked scale of quality. The CJ research community adopted the SSR metric directly from the Rasch literature [51] because the CJ model shares strong mathematical similarities with Item Response Theory (IRT) and Rasch methods [5, 42]. SSR reflects the same core idea found in Classical Test Theory (CTT); it attempts to measure the proportion of variance in observed scores that is truly attributable to the true scores of the representations being assessed [6]

SSR compares true score variance against the error present in the estimates. According to Bramley [5], the formula used to calculate SSR is:

$$
S S R = \frac { G ^ { 2 } } { ( 1 + G ^ { 2 } ) }
$$

G is calculated as $\begin{array} { r } { G = \frac { \sigma _ { \beta } } { R M M S E } } \end{array}$ , where $\sigma _ { \beta }$ represents the standard deviation of the true scores, and RMSE is the root mean square error. This SSR formula can be algebraically simplified to directly show the ratio of true score variance $( \sigma _ { \beta } ^ { 2 } )$ to observed score variance $( \sigma _ { \nu } ^ { 2 } )$ , and that will make the connection to CTT [51]

$$
S S R = \frac { \sigma _ { \beta } ^ { 2 } } { \sigma _ { \nu } ^ { 2 } }
$$

The variance of the true score can be estimated from the variance of observed scores as follows:

$$
\sigma _ { \beta } ^ { 2 } = \sigma _ { \nu } ^ { 2 } - M S E
$$

Where MSE is the Mean Squared Error [2]

SSR has been used as a measure of assessment consistency, even though it was a borrowed concept. However, [51] empirically validated it through a meta-analysis of 26 CJ assessments. They performed the study by splitting assessor groups into halves and found that SSR is a highly accurate estimate for split-half reliability. Additionally, they demonstrated that SSR can serve as a reliable indicator of interrater reliability. [36] further validated these claims by conducting a meta-analysis of 101 CJ datasets, confirming that SSR acts as a reliable proxy for inter-rater agreement. Furthermore, their findings suggested that roughly 10 comparisons for every representation is generally sufficient to achieve reliable measurement scales.

According to [51], caution should be exercised when copying reliability metrics from IRT to CJ, as there are different parameterisations in the measurement models. Furthermore, their results were mixed and inconclusive when they tested whether SSR could represent a strict correlation with the truth. They highlighted that this is likely due to conceptual differences in how IRT and CTT treat error variance.

Gray et al. [26] pointed out that the relative uncertainty estimated by SSR highly depends on the underlying mathematical model, and this uncertainty is rarely presented to users intuitively. Additionally, they noted that due to the unavailability of true score variance before simulation, the calculation of SSR is highly problematic in practical scenarios. Gray et al. [26] also explained that SSR does not provide a real-time, natural performance metric; it fails to offer a dynamic stopping criterion. Hence, it is required to have a predetermined, fixed budget at the start of the simulation. While Kinnear et al. [36] suggested that 10 judgments per script is roughly enough to get a reliable score in practice, this remains a static target. Since SSR is not a real-time monitoring measure, it cannot dynamically halt the process early even if true consensus is reached quickly.

These limitations suggest that reliability assessments in CJ would benefit from approaches that are more directly aligned with posterior rank uncertainty and decision-theoretic ranking objectives.

## 2.6 Knowledge Gradient (KG) Methods

In Ranking and Selection, efficiently allocating a limited sampling budget to identify the best alternatives is a central challenge. Gupta and Miescke [30] developed Bayesian look-ahead procedures to optimize these sampling allocations by evaluating the expected posterior risks. Recent researches show how samples were allocated to maximise the Expected Value of Information (EVI). For example, Chick et al. [11] developed a new myopic sequential sampling procedure that avoided the asymptotic and probabilistic approximations used in earlier Bayesian probability models.

Frazier et al. [20] formally introduced the KG methods based on the single look-ahead policy introduced by Gupta and Miescke [30]. They showed that the KG method myopically maximises the expected gain in information value one step at a time, whilst keeping a balance between computational methods and the need for sequential efficiency in information collection.

The KG method is mainly used in sequential information collection and offline learning to maximise the expected information gain [17|. It efficiently balances the trade-off between exploration and exploitation. It prefers to find alternatives in high variance regions (exploration) and alternatives close to the current maximum (exploitation), as these solutions are most likely to alter the final decision [17]. The KG policy is theoretically robust. Frazier and Powell [19] demonstrated that sequential sampling policies are asymptotically optimal

The KG framework has been extended to handle more complex and realistic data structures. Frazier et al. [18] extended the KG framework to correlated normal beliefs, allowing the algorithm to exploit dependencies between alternatives and efficiently evaluate a large set of designs. Hierarchical Knowledge Gradient was introduced to accommodate multi-dimensional attributes [41]. Kullback-Leibler KG (KL-KG) was developed to minimise the divergence between the chosen and target offspring distributions rather than simply maximising the probability of correct selection [28]. It has been identified limitations of standard KG in some research settings. For instance, in problems involving Bernoulli outcomes and probability learning, the naive KG framework may be inefficient or require refinement [34]

The main idea of the KG methods is to evaluate how much the maximum of our current estimate would increase if a specific alternative/point were selected for evaluation. Since the outcome of the query is unknown at the time it is chosen, KG computes an expectation over all possible hypothetical outcomes weighted by their predictive probabilities. The KG value for a candidate point x is defined as

$$
\mathbb { E } _ { p ( y | \mathcal { D } ) } \left[ \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { X } } \mathbb { E } \big [ f ( x ^ { \prime } ) \mid \mathcal { D } , ( x , y ) \big ] \right] - \operatorname* { m a x } _ { x ^ { \prime } \in \mathcal { X } } \mathbb { E } \big [ f ( x ^ { \prime } ) \mid \mathcal { D } \big ]
$$

Here, y denotes the hypothetical observation at x, and $p ( y \mid \mathcal { D } )$ is the predictive distribution induced by the Gaussian Process (GP) model { given the current dataset D.

Classical KG formulations assume that each measurement corresponds to observing a single alternative. However, in pairwise comparison experiments, two alternatives are involved in each sample, and the observation consists of a comparative outcome. Since each observation simultaneously informs beliefs about both items, this structure introduces additional complexity into the existing KG framework. Wu and Gardner [55] developed an exact, analytical KG acquisition function for preferential Bayesian optimisation, which is a major advancement for traditional KG. They model preferences using a latent Gaussian Process (GP) utility function together with a probit likelihood. Since pairwise comparisons make the posterior distribution non-Gaussian, applying KG to preference learning is more challenging. However, they show that the one-step look-ahead posterior follows a skew normal distribution, which allows them to compute the look-ahead posterior mean in closed form.

The KG method has been extended to select the top-k alternatives based on information collected from noisy, sequentially chosen pairwise comparisons [29]. They work with Borda scores that are built from pairwise comparison outcomes, instead of observing the true scores of each item. They adapt both KG and Optimal Computing Budget Allocation (OCBA) to the pairwise setting and compare their performance. They show that KG performs well when the goal is to identify the top performing set under a limited sampling budget. Furthermore, they prove that their approach is asymptotically optimal under certain conditions, and their method outperforms existing racing algorithms.

These developments in the KG framework are highly relevant to CJ. In CJ, each data point is a pairwise comparison between two student responses or pieces of work, and the aim is to produce a reliable ranking using a limited number of comparisons. This setup is very similar to a fixed-budget Bayesian information collection problem studied using KG [20]. Each comparison can be seen as a measurement decision, while the uncertainty about an item's true strength is captured by a posterior distribution over its latent strength.

The KG framework has a strong theoretical foundation in value-of-information optimisation. However, it has not yet been fully integrated with hierarchical Bayesian BTM in the CJ context, particularly with respect to ranking reliability.

Several important developments in the modelling and analysis of paired comparison data have been discussed above. The BTM offers a robust probabilistic framework for ranking based on pairwise outcomes, while Bayesian extensions introduce posterior uncertainty quantification and hierarchical regularisation. In the context of education assessment, BCJ extends this approach by deriving posterior rank distribution and enabling active learning pair selection strategies.

However, the following important limitations remain

• Optimal pair selection under limited judgment budgets has only been partially addressed within existing Bayesian BTM models, even though those models provide posterior inference over latent strengths. Existing BCJ implementation primarily aims to reduce local uncertainty rather than explicitly maximising improvements in global ranking accuracy.

• Reliability of ranking in CJ relies on SSR. Although it provides an interpretable measure of interitem variability, it is sensitive to modelling assumptions and does not reflect posterior rank uncertainty or decision-relevant ranking performance.

• KG methods provide a framework for sequential information collection, and have been extended to pairwise and preference learning settings. However, these methods have not been incorporated into hierarchical Bayesian BTM within educational CJ. To our knowledge, there is no existing framework that explicitly connects KG-based active learning pair selection methods implemented within the Bayesian CJ context.

These gaps motivate the development of a unified framework that integrates hierarchical Bayesian BTM, posterior-based reliability assessment, and KG-driven active pair selection. Such a framework aligns sequential decisions with explicit ranking and reliability objectives to improve rank recovery accuracy while efficiently utilising limited human judgments.

## 3 Methodology

## 3.1 Bradley Terry Model

The Bradley-Terry model (BTM) assigns probabilities to the outcomes of pairwise comparisons based on strength parameters. Let $[ N ] = 1 , 2 , . . . , N$ be the set of finite items, and each with a latent score $\theta _ { i } \in \mathbb R , i = 1 , 2 , . . . , N$ . Then the score vector can be written as $\boldsymbol { \theta } = ( \theta _ { 1 } , \theta _ { 2 } , . . . , \theta _ { N } ) ^ { \top }$

A comparison between i and j gives a binary outcome $y _ { i j } \in \{ 0 , 1 \}$ , where

$$
y _ { i j } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } i \succ j } \\ { 0 } & { { \mathrm { i f ~ } } j \succ i } \end{array} \right. }
$$

The Bradley-Terry probability that i is preferred over $j$ is

$$
P ( y _ { i j } = 1 \mid \theta ) = \frac { e ^ { \theta _ { i } } } { e ^ { \theta _ { i } } + e ^ { \theta _ { j } } } = \frac { 1 } { 1 + e ^ { - ( \theta _ { i } - \theta _ { j } ) } } = \sigma ( \theta _ { i } - \theta _ { j } )\tag{1}
$$

At $t ^ { t h }$ comparison, we can define $\boldsymbol { x } _ { t } = \boldsymbol { e } _ { i _ { t } } - \boldsymbol { e } _ { j _ { t } }$ , so that $x _ { t } ^ { \top } \theta = \theta _ { i _ { t } } - \theta _ { i _ { t } }$ . The model depends only on score difference. Therefore, one location constraint is required for identifiability. So, we work on the centred subspace.

$$
\mathcal { C } = \{ \theta \in \mathbb { R } ^ { N } : 1 ^ { \top } \theta = 0 \}
$$

## 3.2 Bayesian Bradley Terry Model

## 3.2.1 Prior Selection

In Bayesian statistics, the prior distribution is highly important for better posterior inferences. [54] suggests two prior distributions for the BTM, and we adopt the independent Gaussian prior $\theta _ { i } \mid \sigma _ { \theta } ^ { 2 } \sim$ $\textstyle { \mathcal { N } } ( 0 , \sigma _ { \theta } ^ { 2 } )$ on the student's latent log abilities. Further, we apply the parameter expansion $\theta = \xi \eta$ and place an Inverse-Gamma hyperprior on $\sigma _ { \eta } ^ { 2 }$ . More details in this can be found in section 3.2.4.

## 3.2.2 Likelihood for Paired Comparisons

The outcome $y _ { t }$ follows a Bernoulli distribution in the $t ^ { t h }$ comparison between i and $j$

$$
y _ { t } \mid { \theta } \overset { \mathrm { i n d } } { \sim } B e r n o u l l i ( \sigma ( x _ { t } ^ { \top } { \theta } ) ) , \qquad t = 1 , 2 , . . . , K
$$

where $K$ is the total number of comparisons.

Since these outcomes are independent, we can write the cumulative likelihood as follows:

$$
\mathcal { L } _ { K } ( \theta ) = \prod _ { t = 1 } ^ { K } \Big [ \sigma ( x _ { t } ^ { \top } \theta ) ^ { y _ { t } } \left( 1 - \sigma ( x _ { t } ^ { \top } \theta ) \right) ^ { 1 - y _ { t } } \Big ]
$$

Then we can write cumulative log likelihood as follows:

$$
\ell _ { K } ( \theta ) = \sum _ { t = 1 } ^ { K } \left[ y _ { t } x _ { t } ^ { \top } \theta - \log \left( 1 + e ^ { x _ { t } ^ { \top } \theta } \right) \right]\tag{2}
$$

The design matrix $\boldsymbol { X } \in \mathbb { R } ^ { K \times N }$ can be written by stacking these contrasts as rows $\boldsymbol { x } _ { t } ^ { \top }$ , and the outcome vector $y \in \{ 0 , 1 \} ^ { K }$

## 3.2.3 Polya Gamma Data Augmentation

To make posterior inference more efficient, we use Polya-Gamma data augmentation for Bernoulli Logistic Bradley Terry likelihood [43]. For each pairwise comparison, we model the observed outcome as:

$$
y _ { t } \mid \theta \sim B e r n o u l l i ( \sigma ( x _ { t } ^ { \top } \theta ) )
$$

Where $\sigma ( \cdot )$ is the logistic function. Following Polson et al. (2013), we introduce an auxiliary latent variable $\omega _ { t }$

$$
\omega _ { t } \mid \theta \sim P G ( 1 , x _ { t } ^ { \top } \theta )
$$

This choice makes the likelihood conditionally Gaussian and therefore easier to work with. Then we define $\kappa _ { t } = y _ { t } - 1 / 2$ , the augmented likelihood for $\omega _ { t }$ can be written as follows:

$$
p ( y _ { t } \mid \theta , \omega _ { t } ) \propto \exp \Bigl ( - \frac { \omega _ { t } } { 2 } ( x _ { t } ^ { \top } \theta ) ^ { 2 } + \kappa _ { t } x _ { t } ^ { \top } \theta \Bigr )
$$

Given the Polya-Gamma variables, the model simplifies to a Gaussian regression in the log-ability parameters. This ensures straightforward conjugate Gibbs updates in the parameter expanded hierarchical model.

## 3.2.4 Parameter Expansion

Since commonly used inverse-gamma priors on variance parameters are not recommended due to their undesirable behaviour near zero [21], we placed a weakly informative half-t prior on the global scale of the log strength parameters. However, this prior is not conjugate within the resulting hierarchical model, so we considered the parameter expansion suggested [21]. This strategy represents the half-t prior as a scale mixture of conjugate distributions, enabling efficient Gibbs sampling

Parameter expansion introduces:

$$
\theta = \xi \eta
$$

with:

$$
\eta _ { i } \sim \mathcal { N } ( 0 , \sigma _ { \eta } ^ { 2 } ) , \quad \xi \sim \mathcal { N } ( 0 , s _ { \xi } ^ { 2 } ) , \quad \sigma _ { \eta } ^ { 2 } \sim \mathrm { I n v - G a m m a } ( a _ { 0 } , b _ { 0 } ) .
$$

This induces a conditionally Gaussian prior $\theta _ { k } \mid \xi , \sigma _ { \eta } ^ { 2 } \sim \mathcal { N } ( 0 , \xi ^ { 2 } \sigma _ { \eta } ^ { 2 } )$ on the log strength parameters. The effective scale $\sigma _ { \theta } = | \boldsymbol { \xi } | \sigma _ { \eta }$ is marginally distributed as a half-t distribution.

## 3.2.5 Hierarchical structure of the model

For comparisons $t = 1 , 2 , . . . , K$ , the full hierarchical structure of BBTM can be written as:

$$
y _ { t } \mid \theta \sim \operatorname { B e r n o u l l i } \left( \sigma ( x _ { t } ^ { \top } \theta ) \right) ,
$$

$$
\omega _ { t } \mid \theta \sim \operatorname { P G } \left( 1 , x _ { t } ^ { \top } \theta \right) ,
$$

$$
\theta = \xi \eta ,
$$

$$
\eta \mid \sigma _ { \eta } ^ { 2 } \sim { \mathcal { N } } _ { \mathcal { C } } \left( 0 , \sigma _ { \eta } ^ { 2 } I _ { c } \right) , \quad \mathbf { 1 } ^ { \top } \eta = 0 ,
$$

$$
\xi \sim \mathcal { N } ( m _ { \xi } , s _ { \xi } ^ { 2 } ) ,
$$

$$
\sigma _ { \eta } ^ { 2 } \sim \mathrm { I n v - G a m m a } ( a _ { 0 } , b _ { 0 } ) .
$$

The joint posterior is

$$
p ( \eta , \xi , \sigma _ { \eta } ^ { 2 } , \omega \mid y ) \propto \underbrace { \prod _ { t = 1 } ^ { K } \exp \biggl \{ \kappa _ { t } \xi x _ { t } ^ { \top } \eta - \frac { 1 } { 2 } \omega _ { t } \bigl ( \xi x _ { t } ^ { \top } \eta \bigr ) ^ { 2 } \biggr \} } _ { \mathrm { a u g m e n t e d ~ l i k e i h b o o d ~ a t } \theta = \xi \eta } p _ { \mathrm { P G } } ( \omega _ { t } \mid 1 , 0 ) \cdot \underbrace { p ( \eta \mid \sigma _ { \eta } ^ { 2 } ) p ( \xi ) p ( \sigma _ { \eta } ^ { 2 } ) } _ { \mathrm { p r i o r + h y p e r p r i o r } } .
$$

## 3.2.6 Gibbs sampling

BBTM employs a four block Gibbs sampler that alternate between the following updates. Let $\kappa _ { t } = y _ { t } - \frac { 1 } { 2 }$ 2, $\Omega _ { K } = \mathrm { d i a g } ( \omega _ { t } ) , A _ { K } = X ^ { \top } \Omega _ { K } X$ , and $r _ { K } = X ^ { \top } \kappa$

1. Polya Gamma latent variables. Draw $\omega _ { t } \mid \cdot \sim P G ( 1 , \xi x _ { t } ^ { \top } \eta )$ for $t = 1 , 2 , . . . , K$ , transforming the logistic likelihood into conditionally Gaussian form.

2. Direction η on C. Draw $\eta \mid \cdot \sim \mathcal { N } _ { \mathcal { C } } ( \mu _ { n } , \Lambda _ { \eta } ^ { - 1 } )$ with $\Lambda _ { \eta } = \xi ^ { 2 } A _ { K } + \sigma _ { \eta } ^ { - 2 } I _ { \mathcal { C } }$ and $\mu _ { \eta } = \Lambda _ { \eta } ^ { - 1 } \xi r _ { K }$ with $1 ^ { \top } \eta = 0$

3. Global scale $\xi .$ Draw $\xi \mid \cdot \sim \mathcal { N } ( \mu _ { \xi } , \nu _ { \xi } )$ with $\nu _ { \xi } ^ { - 1 } = \eta ^ { \top } A _ { K } \eta + s _ { \eta } ^ { - 2 }$ and $\mu _ { \xi } = \nu _ { \xi } ( \eta ^ { \top } r _ { K } + m _ { \xi } / s _ { \xi } ^ { 2 } )$

4. Variance $\sigma _ { \eta } ^ { 2 } .$ Draw $\begin{array} { r } { \sigma _ { \eta } ^ { 2 } \vert \cdot \sim \mathrm { I n v - G a m m a } ( a _ { 0 } + N / 2 , b _ { 0 } + \frac { 1 } { 2 } \| \eta \| ^ { 2 } ) } \end{array}$

5. Recover strengths. Set $\theta = \xi \eta \in \mathcal { C }$

This whole process yields an efficient, fully Gibbs sampler with good mixing properties even under sparse comparison data.

## 3.2.7 Sigmoid-Normal Approximation

When computing Knowledge Gradient (KG) [20] metrics, the expected utility of each item comparison must be evaluated under a hypothetical future outcome. This would require running separate MCMC samples for each possible outcome of each unique pair, which becomes computationally infeasible as the number of items grows. For example, when the number of items is 10, there are 45 unique pairs, and it should run 90 MCMC samplers, as it considers both outcomes of each pair. When the number of items increases to 20, the number of unique pairs increases to 190, requiring 380 MCMC runs. So, the number of required MCMC runs grows quadratically with the number of items.

To make this more computationally efficient, we replace the expensive MCMC with a fast analytical approximation. To evaluate the predictive probability of a comparison outcome, we must compute the expectation of the logistic likelihood over the Gaussian posterior distribution of the skill difference. This requires solving the convolution of a logistic sigmoid $\sigma ( z )$ with a Gaussian ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ , which is analytically intractable.

We employ Mackay's Probit Approximation [3] to solve this. This method approximates the logistic sigmoid with a scaled Probit function, $\Phi ( \lambda z )$ , where the scaling factor $\lambda = \sqrt { \pi / 8 }$ is derived by matching the slopes of the functions at the origin. This allows for a closed-form analytic update where the steepness of the sigmoid is moderated by the uncertainty $\sigma ^ { 2 }$ . The approximate expected win probability is given by:

$$
P r ( i \succ j ) \approx \sigma \left( \frac { \mu _ { i j } } { \sqrt { 1 + \frac { \pi } { 8 } \sigma _ { i j } ^ { 2 } } } \right)
$$

Where $\mu _ { i j }$ is the mean difference and $\sigma _ { i j } ^ { 2 }$ is the variance of that difference.

## 3.3 Novel Pair Selection Methods

It is expensive to obtain comparisons in the real world for rankings. We have implemented a sophisticated active learning strategy known as Knowledge Gradient (KG) [20][18] to intelligently select the most informative pairwise comparisons. In the following section, we will outline the general mathematical framework of the KG and describe its application to our probabilistic ranking model. Further, we detail three distinct utility functions that are used to define “Knowledge": Kendall's Tau distance [35], Entropy of the posterior rank distribution, and separation of the posterior distributions. The objective of this implementation is to maximise the rate of learning and converge to an accurate ranking with a minimum number of pairwise comparisons.

The selection of which pairs of items to compare next to most efficiently improve our understanding of the overall ranking is the central problem with the ranking of paired comparisons. A random selection method might miss valuable information as it can pick the less informative pairs, for a given fixed budget $N * K$ of total comparisons. An entropy driven pair selection method will select the pair that has the highest uncertainty. However, it might be less informative in improving the final rank order of all items. For instance, let's suppose that, after several rounds of comparisons of 5 items $( A , B , C , D , E )$ and we are very confident that Item A and Item B are the two best, but we are extremely uncertain who is better. They have the highest pairwise entropy, i.e., $\operatorname* { P r } ( A \succ B ) = 0 . 5$ . Further, we have some information about Item C and D and believe that Item C is moderately certain than Item D, i.e., $\operatorname* { P r } ( C \succ D ) = 0 . 7$ We have learned that Item E loses to all Items, and we are very much certain that $\operatorname* { P r } ( A \succ E ) = 0 . 9$ and $\operatorname* { P r } ( B \succ E ) = 0 . 9$ . But we have never compared Item C and Item E as we do not have their data. However, the model's prior belief assumes they also have the highest pairwise entropy. In the entropy selection method, it looks at all possible pairs and it might choose pair $( C , E )$ over pair $( A , B )$ . This is less informative as Item C is more certain outcome in this pair from the transitivity principle. If the algorithm had chosen pair $( A , B )$ instead, the outcome would have been highly informative. The KG metrics are specifically designed to avoid these kinds of scenarios. KG metrics would have simulated both pairs and concluded that pair (A, B) would lead to a higher improvement in the final rank order, making it the better choice.

## 3.3.1 Mathematical Framework

In this work, we have implemented adaptive strategies for selecting pairs in a pairwise comparison-based ranking problem, using a Bayesian Bradley-Terry model with Polya-Gamma data augmentation. To decide which pair to compare next, we introduce the following three active learning metrics. The goal is to maximise information gain as possible by choosing the pair that, depending on its outcome, would either disrupt the current rank order, or reduce uncertainty, or separation of the posterior distributions.

## 1. KG-Tau

The KG-Tau matrix is formulated to find the comparison that maximises the expected change in our model's existing rank order. We measure this change using the normalised Kendall's Tau distance between the current rank order and after adding a new comparison to the existing comparisons. A larger expected distance indicates a more informative comparison that leads to faster learning.

Let's consider a latent variable $\pi _ { i }$ as $\beta _ { i } = l o g ( \pi _ { i } )$ . Then from the equation 2, we can map the difference $\theta _ { i , j } = \sigma ( \beta _ { i } - \beta _ { j } )$ to $[ 0 , 1 ]$ .Then the outcome function, ${ \mathcal { O } } ( x )$ can be defined as follows,

$$
\mathcal { O } ( x ) = \left\{ \begin{array} { l l } { 0 , } & { 0 < x < 0 . 5 } \\ { 1 , } & { 0 . 5 < x < 1 } \end{array} \right.
$$

The density of $x , f _ { \delta _ { i j } } ( x )$ , is induced by the posterior distribution of $\delta _ { i j }$ . Then we can calculate the expected tau distance using the following equation:

$$
\mathbb { E } ( \tau ) = \int _ { 0 } ^ { 1 } \tau ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( x ) ) f _ { \delta _ { i j } } ( x ) d x\tag{3}
$$

$$
\mathbb { E } ( \tau ) = \int _ { 0 } ^ { 0 . 5 } \tau ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( x ) ) f _ { \delta _ { i j } } ( x ) d x + \int _ { 0 . 5 } ^ { 1 } \tau ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( x ) ) f _ { \delta _ { i j } } ( x ) d x\tag{4}
$$

Since $\tau ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( x )$ is constant in each interval $\left( \kappa _ { 0 } \right.$ for $0 < x < 0 . 5$ $\kappa _ { 1 }$ for $0 . 5 < x < 1 )$ , we can simplify equation 16 as follows:

$$
\mathbb { E } ( \tau ) = \kappa _ { 0 } \int _ { 0 } ^ { 0 . 5 } f _ { \delta _ { i j } } ( x ) d x + \kappa _ { 1 } \int _ { 0 . 5 } ^ { 1 } f _ { \delta _ { i j } } ( x ) d x\tag{5}
$$

Where:

$$
\begin{array} { r l } & { \bullet \ \int _ { 0 } ^ { 0 . 5 } f _ { \delta _ { i j } } ( x ) d x = P r ( x < 0 . 5 ) = P r ( \delta _ { i j } < 0 ) = 1 - P r ( i \succ j ) } \\ & { \bullet \ \int _ { 0 . 5 } ^ { 1 } f _ { \delta _ { i j } } ( x ) d x = P r ( x \geq 0 . 5 ) = P r ( \delta _ { i j } > 0 ) = P r ( i \succ j ) } \end{array}
$$

Hence, the integral form reduces to

$$
\mathbb { E } ( \tau ) = \kappa _ { 0 } ( 1 - P r ( i \succ j ) ) + \kappa _ { 1 } P r ( i \succ j )\tag{6}
$$

The outcome probability can be calculated as follows

$$
p = \mathrm { P r } ( i \asymp j ) = \mathrm { P r } ( \lambda _ { i } > \lambda _ { j } \mid \mathcal { D } ) = \Phi \left( \frac { \delta _ { \mu } } { \sqrt { \delta _ { \sigma } ^ { 2 } } } \right)\tag{7}
$$

where $\Phi$ is the standard normal CDF and

$$
\begin{array} { r l r } & { } & { \delta _ { \mu } = \lambda _ { m e a n } [ i ] - \lambda _ { m e a n } [ j ] } \\ & { } & { \delta _ { \sigma } ^ { 2 } = \Sigma _ { [ i , i ] } + \Sigma _ { [ j , j ] } - 2 \Sigma _ { [ i , j ] } } \end{array}
$$

Select the pair that maximises the expected tau.

Kendall's tau distance measures ranking disagreement via discordant pairs, and a high expected tau indicates the pair that makes the highest disruption to the current rank order, providing high information value.

## 2. KG-Entropy

Entropy measures the uncertainty in the posterior rank probabilities. For each item $k ,$ the rank probability distribution $r _ { k } = { \mathrm { r a n k } } _ { - } { \mathrm { d i s t } } [ k , : ]$

$$
H ( r _ { k } ) = - \sum _ { l = 0 } ^ { N - 1 } r _ { k , l } \log _ { 2 } ( r _ { k , l } )\tag{8}
$$

Then the average entropy

$$
H _ { a v g } = \frac { 1 } { N } \sum _ { k } H ( r _ { k } )\tag{9}
$$

Then the expected entropy is calculated as follows:

$$
\mathbb { E } ( \Delta H ) = P r ( \mathscr { O } ( x ) = 0 \mid i \succ j ) \Delta H ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( 0 ) ) + P r ( \mathscr { O } ( x ) = 1 \mid i \setminus j ) \Delta H ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( 1 ) )\tag{10}
$$

We can think of the entropy difference as the difference between the current entropy and the entropy after adding new data. So, we can write it as follows:

$$
\Delta H ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( C ) = H ( \mathcal { R } _ { c } ) - H ( \mathcal { R } _ { n } ( C ) )\tag{11}
$$

Since entropy reduces with the addition of new data, it is always

$$
H ( \mathcal { R } _ { c } ) > H ( \mathcal { R } _ { n } ( C )
$$

Let's assume $p _ { 1 } = ( 1 - P r ( i \succ j ) )$ and $p _ { 0 } = P r ( i \succ j )$ , then we can write equation 10 as follows:

$$
\mathbb { E } ( \Delta H ) = p _ { 0 } \Delta H ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( 0 ) ) + p _ { 1 } \Delta H ( \mathcal { R } _ { c } , \mathcal { R } _ { n } ( 1 ) )\tag{12}
$$

Then we can rewrite equation 12 as follows

$$
\mathbb { E } ( \Delta H ) = p _ { 0 } [ H ( \mathcal { R } _ { c } ) - H ( \mathcal { R } _ { n } ( 0 ) ) ] + p _ { 1 } [ H ( \mathcal { R } _ { c } ) - H ( \mathcal { R } _ { n } ( 1 ) ) ]
$$

$$
\mathbb { E } ( \Delta H ) = ( p _ { 0 } + p _ { 1 } ) H ( \mathcal { R } _ { c } ) - [ p _ { 0 } H ( \mathcal { R } _ { n } ( 0 ) ) + p _ { 1 } H ( \mathcal { R } _ { n } ( 1 ) ) ]\tag{13}
$$

In equation 13 $( p _ { 0 } + p _ { 1 } ) H ( \mathcal { R } _ { c } )$ is a constant. ${ \mathrm { S o } } ,$ we have to minimise the $[ p _ { 0 } H ( { \mathcal { R } } _ { n } ( 0 ) ) ~ +$ $p _ { 1 } H ( \mathcal { R } _ { n } ( 1 ) ) ]$ to get the largest reduction in expected entropy.

## 3. KG-SSR

The primary objective of a comparative ranking model is to distinguish items with higher confidence. Due to the overlap of posterior distributions and the creation of noise that masks the true ranking order of items, these ranking tasks will become more difficult.

Most of the time, standard active learning methods focus on local pairwise uncertainty or rank stability. Hence, these methods may not directly optimise the global clarity of the rankings. So, we propose a new Knowledge Gradient acquisition function $\left( K G - S S R \right)$ that selects the comparisons maximising the expected increase in the Scale Separation Reliability (SSR).

The acquisition function explicitly targets the global separability of the items and drives the system towards a state where items are distinct, and the rankings are unambiguous.

Let the current state at time t be $S _ { t }$ , parametrised by a mean vector $\mu ^ { ( t ) }$ and a covariance matrix $\Sigma ^ { ( t ) }$ for N items. Further, we extract the variance vector for the items as $\nu = d i a g ( \Sigma ^ { ( t ) } )$ . Then we define the utility function SSR to quantify the ratio as follows:

$$
\mathit { S S R } ( S _ { ( t ) } ) = \frac { v a r ( \mu _ { i } ) - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sigma _ { i } ^ { 2 } } { v a r ( \mu _ { i } ) }
$$

KG-SSR selects the pair $( i , j )$ that maximises the expected gain in this SSR metric. We look one step ahead into the future:

$$
K G _ { S S R } ( S _ { t } , S _ { t + 1 } \mid ( i , j ) ) = \mathbb { E } _ { y _ { i j } } [ \mathrm { S S R } ( S _ { t + 1 } ) ] - \mathrm { S S R } ( S _ { t } )
$$

Expanding the expectation over the two possible outcomes (item i wins or item $j$ wins)

$$
K G _ { S S R } ( S _ { t } , S _ { t + 1 } \mid ( i , j ) ) = [ P r ( i \succ j ) \cdot S S R ( S _ { t + 1 } ^ { ( 1 ) } ) + P r ( j \succ i ) \cdot S S R ( S _ { t + 1 } ^ { ( 0 ) } ) ] - S S R ( S _ { t } )
$$

## 3.4 Fast Knowledge Gradient via One-step Moment updates for BBTM

KG methods were developed in ranking and selection for sequential measurement selection and were later extended to Bayesian optimisation [20]. Since human judgments are costly and the aim is to select comparisons that yield the optimal rank order within a limited budget, KG methods can be highly effective in CJ. A direct KG implementation can be computationally expensive and time-consuming, as it requires evaluating each item pair, which would require repeated posterior inference under hypothetical outcomes. To address this, we implemented a one-step look-ahead moment approximation that allows analytical moment updates and replaces nested posterior sampling, which makes KG implementation feasible in practice.

To the best of our knowledge, this is the first integration of KG with Bayesian Bradley-Terry in a CJ environment that avoids nested MCMC sampling via analytical one-step moment updates.

## The Predictive Distribution and MacKay's Approximation

Consider a hypothetical comparison between item i and j. Let $\delta _ { i j } = \lambda _ { i } - \lambda _ { j }$ denote the latent skill difference. Under our current Gaussian posterior belief, $\lambda \sim { \mathcal { N } } ( \mu , \Sigma )$ , the marginal distribution of the difference is a univariate Gaussian, $\delta _ { i j } \sim \mathcal { N } ( m _ { i j } , \nu _ { i j } )$ , where:

$$
m _ { i j } = \mu _ { i } - \mu _ { j } \quad \nu _ { i j } = \Sigma _ { i i } + \Sigma _ { j j } - 2 \Sigma i j
$$

The BTM likelihood of item i winning is given by the logistic sigmoid function, $\sigma ( \delta _ { i j } )$ . We need to marginalise over the latent skill difference to compute the predictive probability of this outcome:

$$
p ( y = + 1 \mid m _ { i j } , v _ { i j } ) = \int \sigma ( \delta _ { i j } ) { \mathcal { N } } ( \delta _ { i j } \mid m _ { i j } , v _ { i j } ) d \delta _ { i j }
$$

We use MacKay's probit approximation [3], as the convolution of a Gaussian and a logistic sigmoid is analytically intractable. The predictive probability can be expressed in closed form by approximating the logistic distribution with a normal distribution with variance $8 / \pi$

$$
P r ( y \mid m _ { i j } , \nu _ { i j } ) \approx \Phi \left( \frac { y \cdot m _ { i j } } { \sqrt { \nu _ { i j } + \frac { 8 } { \pi } } } \right)
$$

where $\Phi ( \cdot )$ is the CDF of the standard normal distribution, and $y \in \{ + 1 , - 1 \}$ shows the winner.

## Analytic posterior updates

After the outcome y from hypothetical comparison, the intermediate true posterior is proportional to $\sigma ( y \delta _ { i j } ) { \mathcal { N } } ( \lambda \mid \mu , \Sigma )$ . This is no longer strictly Gaussian. However, ADF converts this back to Gaussian, $\mathcal { N } ( \mu _ { n e w } , \Sigma _ { n e w } )$ , by computing the mean and covariance of the intermediate distribution. Let the total assumed standard deviation $s = \sqrt { \nu _ { i j } + 8 / \pi }$ , and normalised margin $\begin{array} { r } { u \ = \ \frac { y \cdot m _ { i j } } { s } } \end{array}$ of the outcome. Then, define a truncation factor based on the inverse Mills Ratio to get the moment match updates as follows:

$$
\beta ( u ) = \frac { \phi ( u ) } { \Phi ( u ) }
$$

Here $\phi ( \cdot )$ and $\Phi ( \cdot )$ represent PDF and CDF of the standard normal distribution, respectively. We can calculate closed-form updates for the mean vector and covariance matrix as follows:

$$
\mu _ { n e w } = \mu + y \frac { \beta ( u ) } { s } ( \Sigma . , i , \Sigma . , j )
$$

$$
\Sigma _ { n e w } = \Sigma - \frac { \beta ( u ) ( \beta ( u ) + u ) } { s ^ { 2 } } ( \Sigma . , i , \Sigma . _ { , j } ) ( \Sigma . , _ { , i } , \Sigma . _ { , j } ) ^ { T }
$$

Here, $\left( \Sigma _ { \cdot , i } , \Sigma _ { \cdot , j } \right)$ is the column vector of the covariance of every item in the dataset with the specific pairwise difference.

We can calculate analytical posterior $( \mu _ { n e w } ^ { ( y ) } , \Sigma _ { n e w } ^ { ( y ) } )$ for any hypothetical comparison quickly.

## KG computation

For a utility $U ( \mu , \Sigma )$ , the KG value for a hypothetical pair $( i , j )$ is the expected on-step gain.

$$
K G _ { U } ( i , j ) = \mathbb { E } _ { y | \mathcal { D } _ { t } } [ U ( \mu _ { t } ^ { ( y ) } , \Sigma _ { t } ^ { ( y ) } ) ] - U ( \mu _ { t } , \Sigma _ { t } )
$$

$$
K G _ { U } ( i , j ) = P r ( i \sim j ) [ U ( \mu _ { t } ^ { ( + 1 ) } , \Sigma _ { t } ^ { ( + 1 ) } ) ] + P r ( j \succ i ) [ U ( \mu _ { t } ^ { ( - 1 ) } , \Sigma _ { t } ^ { ( - 1 ) } ) ] - U ( \mu _ { t } , \Sigma _ { t } )
$$

The next comparison is selected as

$$
( i ^ { * } , j ^ { * } ) = \arg \operatorname* { m a x } _ { ( i , j ) \in D _ { t } } K G _ { U } ( i , j )
$$

## 3.5 Model Robustness Check

To validate the robustness of the active learning model, we have conducted a controlled experiment to quantify its performance degradation in the presence of noisy data. The objective was to identify the minimum proportion of true skill-based outcomes required to recover an accurate ranking

We introduced a parameter α (threshold), which controls the signal-to-noise ratio of the comparison outcomes. This parameter represents the probability that a given comparison result reflects the true latent ability of items rather than a random guess. The experiment was conducted across a range of threshold levels $\alpha \in \{ 0 . 0 , 0 . 1 , 0 . 2 , \ldots , 1 . 0 \}$ , where

$\alpha = 0 . 0 \mathrm { : }$ All outcomes are determined by a random coin toss, independent of item skills

$\alpha = 1 . 0 \colon$ All outcomes are determined by the item's true skill differences.

For every pair $( i , j )$ selected by the pair selection method, the outcome $y _ { i j }$ was generated using the following decision rule:

1. Draw a random variable $u \sim \mathcal { U } ( 0 , 1 )$

2. If $u \leq \alpha$ , the outcome is generated by the true skills of the item. The winner is determined by comparing samples drawn from the item's true skill distributions.

3. if $u \ > \ \alpha$ , the outcome is generated randomly. The winner is selected with equal probability, providing no information about the true skills.

The main goal of this experiment is to assess the rank recovery potential under imperfect conditions. By varying α, we aim to:

1. Quantify Tolerance: Determine the threshold where model performance stabilises, indicating the minimum proportion of skill-based outcomes required to achieve near optimal ranking.

2. Validate Efficiency: Depicts that the model does not require perfect data to converge. For example, if the ranking accuracy remains stable at $\alpha = 0 . 7 $ it shows that the model can successfully recover the true ranks even with 30% coin toss outcomes.

3. Establish Reliability: Confirm that the active learning method focuses on the signal within the comparisons rather than over-fitting to the noise.

For each noise level, we conducted 50 independent runs to ensure statistical reliability. The Normalised Kendall's Tau distance was used to measure the rank accuracy between the model's posterior ranking and the true rankings.

## 3.6 Posterior Separation Reliability (PSR) for BBTM

It is essential that estimated latent strengths are sufficiently well separated to recover a reliable ranking order in CJ. In traditional CJ, SSR is used for this purpose, which is based on point estimates and associated error terms [51]. However, within a Bayesian version of BTM framework, posterior inference provides both the estimated strengths and their uncertainty. Therefore, it is straightforward and internally consistent to define a reliability measure directly from the posterior distributions.

Conceptually, PSR measures the extent to which the spread in posterior mean strengths across items exceeds the average posterior uncertainty in those items, thereby quantifying the degree of effective separation achieved by the model. We can define PSR as follows:

$$
P S R = 1 - \frac { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sigma _ { i } ^ { 2 } } { \mathrm { V a r } ( \{ \mu _ { i } \} _ { i = 1 } ^ { N } ) }
$$

Where $\mu _ { i } = \mathbb { E } ( \theta _ { i } \mid \mathcal { D } ) , \sigma _ { i } ^ { 2 } = V a r ( \theta _ { i } \mid \mathcal { D } )$ , Var(·) denotes the variance operator, and N = Number of items.

The proposed PSR has the maximum value of 1 when there is no posterior uncertainty in the latent strengths. However, unlike other reliability measures, PSR has no finite lower bound. PSR values close to 1 indicate strong posterior separation, values closer to 0 mean posterior strength separation and uncertainty are similar, and negative values demonstrate that posterior uncertainty exceeds the spread of the posterior means.

## 4 Convergence of the Bayesian Bradley Terry Model

We give a full convergence proof for the BBTM on the centred identifiable subspace. We allow the comparison schedule to be predictable. This means that the pair $( i _ { t } , j _ { t } )$ compared at time t may be fixed in advance, chosen randomly without reference to the outcome, or adapted based on previous comparisons and their outcomes. Under these conditions, we cover many possible comparison schemes, including no repeating pairs, random, round robin method, and certain active learning methods, such as entropy and novel knowledge gradient methods. The only condition is that they must satisfy our assumptions about how information grows over time. If the long-run frequency of each pair is positive, the marginal posterior is consistent at the true centred score vector. We show that the posterior is locally tight when the normalised Fisher information matrix is assumed to have a limit. This is because of the consistency of the marginal posterior at the true centred score, and the local strong concavity of the logistic likelihood. This allows us to prove a Bernstein-von Mises theorem on the rescaled posterior, where the asymptotic covariance is given by the inverse limiting information matrix. Consequently, ranking consistency then follows when the true scores are strictly separated. We do not need a separate proposition for any specific active learning policy. Such a proposition is only needed if someone wants to prove that a particular method, for example KG rule, satisfies the information growth assumptions we used.

## 4.1 Assumptions

1. Finite number of items. The total number of items $N < \infty$ is fixed. The true latent score vector $\theta ^ { \ast } \in \mathcal { C }$ is fixed and unknown.

2. The scores are strictly separable. No two scores are equal.This means that $| \theta _ { i } ^ { * } - \theta _ { j } ^ { * } | \neq 0$ ∀i≠ j. This ensures that the true ranking $\pi ^ { * }$ is well-defined.

3. Adaptive Bradley-Terry sampling. Let $\mathcal { F } _ { t } = \sigma \big ( ( i _ { 1 } , j _ { 1 } , y _ { 1 } ) , . . . , ( i _ { t } , j _ { t } , y _ { t } ) \big )$ be the comparison history filtration. At time t, the selected pair $( i _ { t } , j _ { t } )$ is $\mathcal { F } _ { t - 1 }$ measurable, i.e., chosen based on the past comparison outcomes.

$$
y _ { t } | \mathcal { F } _ { t - 1 } , ( i _ { t } , j _ { t } ) \sim \mathrm { B e r n o u l l i } ( \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) )
$$

Then, $y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ^ { * } )$ is a martingale difference.

4. Polya-Gamma data augmentation. The Gibbs sampler is designed, which uses exact Polya-Gamma data augmentation [43]. The auxiliary variable $\omega _ { t } | \theta , y _ { t } \sim \mathbf { P G } ( 1 , x _ { t } ^ { \top } \theta )$ are sampled exactly and the full conditional $( \theta | y , \omega )$ is a multivariate Gaussian computed in closed form on C. This is an exact data augmentation, not an approximation.

5. Information growth. Three types of information growth conditions are defined here.

5a. Consistency-level information growth. The cumulative Fisher information on $\mathcal { C }$ is

$$
\mathcal { T } _ { K } ( \theta ^ { * } ) = \sum _ { t = 1 } ^ { K } \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \big ( 1 - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) ) \big ) \cdot x _ { t } x _ { t } ^ { T }
$$

A minimal consistency level information condition is $\lambda _ { m i n , c } ( \mathcal { I } _ { K } ( \theta ^ { * } ) )  \infty$ as $K  \infty$ . This means information accumulates in every identifiable direction.

5b. Positive pair frequency growth. There exists $p _ { m i n } > 0 \mathrm { ~ s . t . ~ }$ , for every pair (i, j)

$$
\operatorname* { l i m } _ { K \to \infty } \operatorname* { i n f } _ { K } \operatorname* { K } _ { > } p _ { \mathrm { m i n } } > 0
$$

where $k _ { i j } ( K )$ is the number of times pair $( i , j )$ has been compared in total comparisons K.   
This implies 5a and is sufficient for the posterior consistency through the KL argument.

5c. Rate level information limit. The normalised Fisher information converges on $\mathcal { C }$

$$
\frac { 1 } { K } \mathcal { T } _ { K } ( \pmb { \theta } ^ { * } )  \bar { \mathcal { T } } ( \pmb { \theta } ^ { * } ) \succ 0
$$

This condition is required for BvM theorem and the $1 / \sqrt { K }$ convergence rate of posterior distribution. Notice that 5b does not imply 5c. The pair frequencies $\frac { \check { k } _ { i j } ( K ) } { K }$ may be bounded away from zero for all $( i , j )$ but not necessarily converge to any limit.

6. Proper Gaussian prior on $\mathcal { C } .$ Each $\theta _ { i }$ drawn independently from $\theta _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ , with $0 < \sigma < \infty$ The prior has full support on ${ \mathcal { C } } ,$ and the variance $\sigma ^ { 2 }$ is not assigned a hyperprior.

Remark For theoretical simplicity, this argument is presented under a fixed prior variance. In our implementation, however, we use a weakly-informative half-t prior on the global scale, represented through its conjugate scale-mixture form with parameter expansion to improve MCMC mixing, as discussed in Section 3.2.4. This does not change the convergence results, because the proofs require only that the prior has full support and is locally positive in a neighbourhood of $\theta ^ { * }$

7. Identifiability constraint. The location non-identifiability of the BTM can be fixed by setting $1 ^ { \top } \theta = 0 , 1 ^ { \top } \theta ^ { * } = 0$

## 4.2 Foundation Lemmas

Lemma 1. Exact Gaussian full conditional under PG data augmentation. The full conditional posterior of $\theta \in \mathcal { C }$ given comparison outcomes $y = ( y _ { 1 } , y _ { 2 } , . . . , y _ { K } )$ and the auxiliary variables $\boldsymbol { \omega } = ( \omega _ { 1 } , \omega _ { 2 } , . . . , \omega _ { K } ) ^ { T }$ is the multivariate Gaussian under assumptions 3, 4, 6 and 7

$$
\theta \mid y _ { K } , \omega \sim \mathcal { N } _ { \mathrm { J } } ( \mu ^ { * } ( \omega ) , \Lambda ^ { * } ( \omega ) ^ { - 1 } )
$$

where

$$
\Lambda ^ { * } ( \omega ) = X ^ { \top } d i a g ( \omega ) X + \Lambda _ { 0 } ,
$$

$$
\mu ^ { * } ( \omega ) = \Lambda ^ { * } ( \omega ) ^ { - 1 } X ^ { \top } \kappa , \qquad \kappa _ { t } = y _ { t } - \frac { 1 } { 2 }
$$

and $\Lambda _ { 0 } = \sigma _ { 0 } ^ { - 2 } I _ { C }$ is the fxed prior precision on $\mathcal { C } .$

Proof. The Polya-Gamma identity [43] states that, for any $\psi \in \mathbb { R }$ and $y \in \{ 0 , 1 \}$

$$
\sigma ( \psi ) ^ { y } \{ 1 - \sigma ( \psi ) \} ^ { 1 - y } = \frac { e ^ { \kappa \psi } } { 2 } \int _ { 0 } ^ { \infty } e ^ { - \omega \psi ^ { 2 } / 2 } p ( \omega ) d \omega\tag{14}
$$

where $\begin{array} { r } { \kappa = y - \frac { 1 } { 2 } } \end{array}$ and $p ( \omega )$ is the density of $\mathrm { P G } ( 1 , 0 )$ . BT logistic likelihood is not conjugate with the Gaussian prior. However, Polya-Gamma data augmentation turns the logistic likelihood conditionally Gaussian, enabling efficient Gibbs sampling [43]. Equivalently, the conditional form can be written as follows.

$$
\sigma ( \psi ) ^ { y } \{ 1 - \sigma ( \psi ) \} ^ { 1 - y } \propto _ { \theta } \exp \left( \kappa \psi - \frac { \omega \psi ^ { 2 } } { 2 } \right)\tag{15}
$$

where ω $\mid \psi \sim P G ( 1 , \psi )$ . The linear predictor of BTM is $\psi _ { t } = x _ { t } ^ { \top } \theta = \theta _ { i _ { t } } - \theta _ { j }$ t and substituting this into equation 15 gives

$$
\sigma ( x _ { t } ^ { \top } \theta ) ^ { y _ { t } } \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ) \} ^ { 1 - y _ { t } } \propto _ { \theta } \exp \left( \kappa _ { t } x _ { t } ^ { \top } \theta - \frac { \omega _ { t } ( x _ { t } ^ { \top } \theta ) ^ { 2 } } { 2 } \right)\tag{16}
$$

This is an exact form of a Gaussian kernel. $\mathrm { B y }$ Assumption 3, conditional on history $\mathcal { F } _ { t - 1 }$ and selected pair $( i _ { t } , j _ { t } ) , y _ { t }$ values are conditionally independent. ${ \mathrm { S o } } ,$ the joint augmented likelihood for all K comparisons as follows:

$$
\prod _ { t = 1 } ^ { K } \sigma ( x _ { t } ^ { \top } \theta ) ^ { y _ { t } } \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ) \} ^ { 1 - y _ { t } } \propto _ { \theta } \exp \left( \sum _ { t = 1 } ^ { K } \kappa _ { t } x _ { t } ^ { \top } \theta - \frac { 1 } { 2 } \sum _ { t = 1 } ^ { K } \omega _ { t } ( x _ { t } ^ { \top } \theta ) ^ { 2 } \right)\tag{17}
$$

Converting the linear and quadratic terms of equation 17 into matrix form, equation 17 becomes:

$$
\prod _ { t = 1 } ^ { K } \sigma ( x _ { t } ^ { \top } \theta ) ^ { y _ { t } } \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ) \} ^ { 1 - y _ { t } } \propto _ { \theta } \exp \left( \theta ^ { \top } X ^ { \top } \kappa - { \frac { 1 } { 2 } } \theta ^ { \top } X ^ { \top } \mathrm { d i a g } ( \omega ) X \theta \right)\tag{18}
$$

The Gaussian prior $\theta \sim \mathcal { N } _ { \mathcal { C } } ( 0 , \sigma _ { 0 } ^ { 2 } I _ { \mathcal { C } } )$ has density

$$
\pi _ { 0 } ( \theta ) \propto \exp \left( - \frac { 1 } { 2 } \theta ^ { \top } \Lambda _ { 0 } \theta \right) , \qquad \Lambda _ { 0 } = \sigma _ { 0 } ^ { - 2 } I _ { \mathcal { C } }
$$

The posterior distribution can be written as follows, according to the Bayes' rule

$$
\pi ( \theta \mid y _ { K } , \omega ) \propto p ( y _ { K } \mid \theta , \omega ) \cdot \pi _ { 0 } ( \theta )
$$

Therefore, the full conditional posterior is

$$
\pi ( \theta \mid y _ { k } , \omega ) \propto \exp \left( \theta ^ { \top } X ^ { \top } \kappa - \frac { 1 } { 2 } \theta ^ { \top } ( X ^ { \top } \mathrm { d i a g } ( \omega ) X + \Lambda _ { 0 } ) \theta \right)\tag{19}
$$

The canonical form of a standard Gaussian is

$$
\mathcal { N } ( \mu , \Lambda ^ { - 1 } ) \propto \exp \bigg ( \theta ^ { \top } \Lambda \mu - \frac { 1 } { 2 } \theta ^ { \top } \Lambda \theta \bigg )
$$

Equation 19 matches this canonical form with $\mu ^ { * } ( \omega ) = \Lambda ^ { * } ( \omega ) ^ { - 1 } X ^ { \top }$ κ and $\Lambda ^ { * } ( \omega ) = X ^ { \top } \mathrm { d i a g } ( \omega ) X + \Lambda _ { 0 } ,$ hence the full conditional posterior is well defined on C.

Remark It should be noted that Lemma 1 is an exact statement about the conditional posterior $p ( \theta \mid y , \omega )$ , i.e. the posterior given the PG auxiliary variables. This does not, however, imply that the marginal posterior $\begin{array} { r } { p ( { \boldsymbol { \theta } } \mid \boldsymbol { y } ) = \int p ( { \boldsymbol { \theta } } \mid \boldsymbol { y } , \omega ) p ( \omega \mid \boldsymbol { y } ) d \omega } \end{array}$ is Gaussian in finite samples. In fact, the marginal posterior is a scale mixture of Gaussians, which is generally non-Gaussian for finite K. The asymptotic Gaussianity of the marginal posterior is a separate phenomenon, driven by increasing information, and we establish this via the BvM theorem in Lemma 8.

Lemma 2. Conditional posterior precision grows linearly on C. Under assumptions 3, 4, 5b, 6 and 7 for PG variables drawn from their full conditional distribution, we have

$$
\lambda _ { \operatorname* { m i n } , \mathcal { C } } ( \Lambda ^ { * } ( \omega ) ) = \Omega ( K )
$$

almost surely. It follows that

$$
\left. \Lambda ^ { * } ( \omega ) ^ { - 1 } \right. _ { C } = O ( K ^ { - 1 } )
$$

almost surely. Additionally, the conditional posterior variances of the identifiable centred directions and pairwise contrasts are of order $O ( K ^ { - 1 } )$ 1

Proof. The conditional posterior precision can be written as follows:

$$
\Lambda ^ { * } ( \omega ) = \Lambda _ { d a t a } ( \omega ) + \Lambda _ { 0 }
$$

where $\Lambda _ { 0 } = \sigma _ { 0 } ^ { - 2 } I _ { \mathcal { C } } \succeq 0$ is the prior precision. Hence, it suffices to show

$$
\lambda _ { \operatorname* { m i n } , \mathcal { C } } ( \Lambda _ { d a t a } ( \omega ) ) = \Omega ( K ) \quad \mathrm { a . s . }
$$

For all comparisons of pair (i, j), design vectors $x _ { t } = e _ { i _ { t } } - e _ { j _ { t } }$ are same. Then

$$
\Lambda _ { d a t a } ( \omega ) = \sum _ { i < j } \left( \sum _ { t : ( i _ { t } , j _ { t } ) = ( i , j ) } \omega _ { t } \right) ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top }
$$

Define the aggregated edge weight:

$$
w _ { i j } ( K ) : = \sum _ { t : ( i _ { t } , j _ { t } ) = ( i , j ) } \omega _ { t }
$$

Then

$$
\Lambda _ { d a t a } ( \omega ) = \sum _ { i < j } w _ { i j } ( K ) ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top }
$$

This is exactly the weighted graph Laplacian for the complete graph $K _ { N }$ with edge weights $w _ { i j } ( K )$

For a fixed pair $( i , j )$ , the linear predictor $x _ { t } ^ { \top } \theta = \theta _ { i } - \theta _ { j }$ is same. Therefore, $\omega _ { t }$ values conditionally i.i.d. given current θ

$$
\omega _ { t } \mid \theta \sim \operatorname { P G } ( 1 , \theta _ { i } - \theta _ { j } ) \quad \forall t : ( i _ { t } , j _ { t } ) = ( i , j )
$$

The expected value of $\omega _ { t } \mid \theta [ 4 3 ]$

$$
\operatorname { \mathbb { E } } [ \omega \mid \theta ] = { \frac { \operatorname { t a n h } \{ ( \theta _ { i } - \theta _ { j } ) / 2 \} } { 2 ( \theta _ { i } - \theta _ { j } ) } } = : c _ { i j } ( \theta )
$$

Note that $c _ { i j } = 1 / 4$ when $\theta _ { i } = \theta _ { j }$ and $c _ { i j } ( \theta ) > 0$ for any finite $\theta _ { i } - \theta _ { j }$ . Further, $c _ { i j }$ is continuous and $c _ { i j } \to 0$ when $| \theta _ { i } - \theta _ { j } | \to \infty$ . But for bounded $\theta , c _ { i j }$ is bounded away from zero. By assumption 5b

$$
\begin{array} { r } { k _ { i j } ( K ) : = | \{ t \leq K : ( i _ { t } , j _ { t } ) = ( i , j ) \} | \geq p _ { m i n } K \quad \mathrm { e v e n t u a l l y } . } \end{array}
$$

So $k _ { i j } ( K ) \to \infty \ \mathrm { a . s }$ . The SLLN for conditionally i.i.d. PG draws [31], then gives

$$
\frac { 1 } { k _ { i j } ( K ) } \sum _ { t : ( i _ { t } , j _ { t } ) = ( i , j ) } \omega _ { t } \to c _ { i j } ( \theta ) > 0 \quad \mathrm { a . s . }\tag{20}
$$

Combining 20 with assumption 5b,

$$
w _ { i j } ( K ) = k _ { i j } ( K ) \cdot \frac { 1 } { k _ { i j } ( K ) } \sum _ { t } \omega _ { t } \geq p _ { m i n } K \cdot \frac { c _ { i j } ( \theta ) } { 2 } \quad a . s .\tag{21}
$$

for all K sufficiently large. Set $w _ { * } ( K ) : = m i n _ { i < j } w _ { i j } ( K )$ . Then $w _ { * } ( K ) = \Omega ( K ) \ \mathrm { a . s }$ For any $\nu \in { \mathcal { C } }$

$$
\nu ^ { \top } \Lambda _ { d a t a } ( \omega ) \nu = \sum _ { i < j } w _ { i j } ( K ) ( \nu _ { i } - \nu _ { j } ) ^ { 2 } \geq w _ { * } ( K ) \sum _ { i < j } ( \nu _ { i } - \nu _ { j } ) ^ { 2 }
$$

We know that $\nu \in { \mathcal { C } }$ and hence $1 ^ { \top } \nu = 0$

$$
\sum _ { i < j } ( \nu _ { i } - \nu _ { j } ) ^ { 2 } = \frac { 1 } { 2 } \sum _ { i , j } ( \nu _ { i } - \nu _ { j } ) ^ { 2 } = N \| \nu \| ^ { 2 } - \left( \sum _ { i } \nu _ { i } \right) ^ { 2 } = N \| \nu \| ^ { 2 }
$$

Therefore,

$$
\lambda _ { \operatorname* { m i n } , C } \{ \Lambda _ { \mathrm { d a t a } } ( \omega ) \} \geq N \cdot w _ { * } ( K ) = \Omega ( K ) \quad \mathrm { a . s . }
$$

Addition of prior precision $\Lambda _ { 0 } \succeq 0$ does not change the lower bound, and hence $\lambda _ { \operatorname* { m i n } , C } \{ \Lambda ^ { * } ( \omega ) \} = \Omega ( K )$ a.s.

$| | \Lambda ^ { * } ( \omega ) ^ { - 1 } | | _ { c } = O ( K ^ { - 1 } )$ is almost surely by spectral inversion. For any pair $( i , j )$ , the vector $e _ { i } - e _ { j }$ lies in C, so

$$
\operatorname { V a r } ( \theta _ { i } - \theta _ { j } \mid y _ { K } , \omega ) = ( e _ { i } - e _ { j } ) ^ { \top } \Lambda ^ { * } ( \omega ) ^ { - 1 } ( e _ { i } - e _ { j } ) \leq 2 \| \Lambda ^ { * } ( \omega ) ^ { - 1 } \| _ { C } = O ( K ^ { - 1 } ) .
$$

Definition: The pairwise KL divergence for each pair (i, j) can be defined as follows

$$
K L _ { i j } ( \theta ) = K L \{ P _ { \theta ^ { * } } ^ { i j } | | P _ { \theta } ^ { i j } \}
$$

where $P _ { \theta } ^ { i j }$ is the Bernoulli distribution with the probability of success $\sigma ( \theta _ { i } - \theta _ { j } )$ . Then we can write KL divergence explicitly

$$
{ \cal K } L _ { i j } ( \theta ) = \sigma ( \theta _ { i } ^ { * } - \theta _ { j } ^ { * } ) \log \frac { \sigma ( \theta _ { i } ^ { * } - \theta _ { j } ^ { * } ) } { \sigma ( \theta _ { i } - \theta _ { j } ) } + \sigma ( \theta _ { j } ^ { * } - \theta _ { i } ^ { * } ) \log \frac { \sigma ( \theta _ { j } ^ { * } - \theta _ { i } ^ { * } ) } { \sigma ( \theta _ { j } - \theta _ { i } ) }\tag{22}
$$

The population average KL:

$$
K L _ { \mathcal { P } } ( \theta ) = \frac { 1 } { | \mathcal { P } | } \sum _ { i < j } K L _ { i j } ( \theta )
$$

where $\mathcal { P } = \binom { N } { 2 }$

The empirical design weighted KL:

$$
K L _ { K } ( \theta ) = \frac { 1 } { K } \sum _ { t = 1 } ^ { K } K L _ { i _ { t } j _ { t } } ( \theta ) = \sum _ { i < j } \frac { k _ { i j } ( K ) } { K } K L _ { i j } ( \theta )\tag{23}
$$

where K is the total number of comparisons considered.

Lemma 3. KL coercivity and empirical separation. Under Assumptions 3, 5b, 6, and 7

(i) The sublevel sets

$$
S _ { M } = \{ \theta \in \mathcal { C } : K L _ { P } ( \theta ) \leq M \}
$$

are compact for every $M < \infty$

(ii) For every $\varepsilon > 0$

$$
c ( \varepsilon ) : = \operatorname* { i n f } _ { \theta \in { \mathcal { C } } : \| \theta - \theta ^ { * } \| \geq \varepsilon } K L _ { P } ( \theta ) > 0
$$

(iii) For all sufficiently large $K ,$

$$
\operatorname* { i n f } _ { \| \theta - \theta ^ { * } \| \geq \varepsilon } K L _ { K } ( \theta ) \geq c ^ { \prime } ( \varepsilon ) > 0
$$

(iv) For every compact $B \subset { \mathcal { C } }$

$$
\operatorname* { s u p } _ { \theta \in B } | \frac { 1 } { K } \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} + K L _ { K } ( \theta ) |  0
$$

in $P _ { \theta ^ { * } } - p r o b a b i l i t y .$

Proof. $\mathrm { K L } _ { P } ( \theta )$ depends on θ through $\sigma ( \theta _ { i } - \theta _ { j } )$ . Since $\sigma ( \cdot )$ is a continuous function, $K L _ { P } ( \theta )$ is also continuous on C. Therefore, the sublevel set $\bar { S _ { M } } = \mathrm { K L } _ { P } ^ { - 1 } ( ( - \infty , M ] )$ is closed, being the preimage of a closed set under a continuous function.

Suppose there is a sequence such that $\theta ^ { ( n ) } \in { \mathcal { C } }$ and $\lVert \theta ^ { ( n ) } \rVert \to \infty$ . Since $\begin{array} { r } { \theta ^ { ( n ) } \in \mathcal { C } , \sum _ { i } \theta _ { i } ^ { ( n ) } = 0 } \end{array}$ . If all pairwise differences $| \theta _ { i } ^ { ( n ) } - \theta _ { j } ^ { ( n ) } |$ are bounded, then all coordinates must be bounded under the centred condition. But this contradicts $\lVert \theta ^ { ( n ) } \rVert  \infty$ . Hence, at least one pair $( i , j )$ diverges along a subsequence, i.e. $\lvert \theta _ { i } ^ { ( n ) } - \theta _ { i } ^ { ( n ) } \rvert  \infty$ . If $\theta _ { i } ^ { ( n ) } - \theta _ { i } ^ { ( n ) } \to + \infty$ , then $\sigma ( \theta _ { i } ^ { ( n ) } - \theta _ { i } ^ { ( n ) } ) \to 1$ , but the true probability $\sigma ( \theta _ { i } ^ { * } - \theta _ { j } ^ { * } ) \in ( 0 , 1 )$ . According to pairwise KL equation ( 22) $K L _ { i j } ( \theta ) { \stackrel { \smile } { \to } }$ ∞ and then $\mathrm { K L } _ { P } ( \theta ) \to \infty . { \mathrm { ~ S o ~ } }$ $\mathrm { K L } _ { P }$ is coercive, and there is no sequence such that $\left\| \theta \right\| \to \infty$ in the set $\mathrm { K L } _ { P } ( \theta ) \leq M$ . Because of this, $S _ { M }$ is bounded. Since $\mathcal { C }$ is finite-dimensional, the Heine-Borel theorem implies $S _ { M }$ is compact, being closed and bounded.

By the information equality for a single pair $\mathrm { K L } _ { i j } ( \theta ) = 0$ iff $\sigma ( \theta _ { i } { - } \theta _ { j } ) = \sigma ( \theta _ { i } ^ { * } { - } \theta _ { j } ^ { * } )$ . So, $\theta _ { i } - \theta _ { j } = \theta _ { i } ^ { * } - \theta _ { i } ^ { * }$ as $\sigma ( \cdot )$ is a strictly increasing function. Therefore, K $\mathrm { L } _ { P } ( \theta ) = 0$ means all pairwise differences match, i.e., $\theta _ { i } - \theta _ { j } = \theta _ { i } ^ { * } - \theta _ { i } ^ { * }$ for all pairs. Equivalently, $( \theta _ { i } - \theta _ { i } ^ { * } ) - ( \theta _ { j } - \theta _ { j } ^ { * } ) = 0$ for all $( i , j )$ , SO $\theta _ { i } - \theta _ { i } ^ { * }$ is the same for all i; that is, $\theta - \theta ^ { \ast } = c { \bf 1 }$ for some constant c. Since both $\theta , \theta ^ { \ast } \in { \mathcal { C } }$ , we have $\mathbf { 1 } ^ { \top } ( \theta - \theta ^ { * } ) = c N = 0 .$ forcing $c = 0$ Hence $\theta = \theta ^ { * }$

Define the set $A _ { \varepsilon } = \{ \theta \in \mathcal { C } : \| \theta - \theta ^ { * } \| \geq \varepsilon \}$ . Note that $\theta ^ { \ast } \notin { A _ { \varepsilon } }$ since $\lVert { \boldsymbol { \theta } } ^ { * } - { \boldsymbol { \theta } } ^ { * } \rVert = 0 < \varepsilon$ . Consider $A _ { \varepsilon } \cap S _ { M }$ for any finite $M > 0$ Since $S _ { M }$ is compact (established above) and $A _ { \varepsilon }$ is closed, the intersection $A _ { \varepsilon } \cap S _ { M }$ is compact. As KLp is continuous, the Weierstrass extreme value theorem guarantees that $\mathrm { K L } _ { P }$ reaches its infimum on $A _ { \varepsilon } \cap S _ { M }$ at some point $\theta _ { 0 }$ Since $\theta _ { 0 } \in A _ { \varepsilon }$ , we have $\theta _ { 0 } \neq \theta ^ { * }$ , SO $\mathrm { K L } _ { P } ( \theta _ { 0 } ) > 0$

For points outside $S _ { M }$ , i.e., $\theta \in A _ { \varepsilon } \setminus S _ { M }$ , we have $\mathrm { K L } _ { P } ( \theta ) > M$ . Choosing $M > \mathrm { K L } _ { P } ( \theta _ { 0 } )$ , no such point can fall below $\theta _ { 0 }$ . Hence, the global infimum over A8 equals $\mathrm { K L } _ { P } ( \theta _ { 0 } ) > 0 , \mathrm { s o } c ( \varepsilon ) = \mathrm { K L } _ { P } ( \theta _ { 0 } ) > 0$

By Assumption 5b, lim int $\mathrm { { ' } } _ { K  \infty } k _ { i j } ( K ) / K \geq p _ { \mathrm { { m i n } } } > 0$ for every pair. For all sufficiently large $K$ every pair satisfies

$$
{ \frac { k _ { i j } ( K ) } { K } } \geq { \frac { p _ { \mathrm { m i n } } } { 2 } } .
$$

By equation 23, $\mathrm { K L } _ { K }$ expands with frequency weights as

$$
\mathrm { K L } _ { K } ( \theta ) = \sum _ { i < j } \frac { k _ { i j } ( K ) } { K } \mathrm { K L } _ { i j } ( \theta ) .
$$

Since ${ \frac { k _ { i j } ( K ) } { K } } \geq { \frac { p _ { \operatorname* { m i n } } } { 2 } }$ and $\mathrm { K L } _ { i j } ( \theta ) \geq 0$

$$
\operatorname { K L } _ { K } ( \theta ) \geq \frac { p _ { \operatorname* { m i n } } } { 2 } \sum _ { i < j } \operatorname { K L } _ { i j } ( \theta ) = \frac { p _ { \operatorname* { m i n } } | \mathcal { P } | } { 2 } \operatorname { K L } _ { P } ( \theta ) .
$$

By Part (ii), inf $\dot { | } \vert \theta - \theta ^ { * } \vert \vert \geq \varepsilon  \mathrm { K L } _ { P } ( \theta ) = c ( \varepsilon ) > 0$ . Taking the infimum over $\| \theta - \theta ^ { * } \| \geq \varepsilon$ on both sides,

$$
\operatorname* { i n f } _ { \lVert \theta - \theta ^ { * } \rVert \ge \varepsilon } \mathrm { K L } _ { K } ( \theta ) \ge \frac { p _ { \operatorname* { m i n } } | \mathcal { P } | } { 2 } \operatorname* { i n f } _ { \lVert \theta - \theta ^ { * } \rVert \ge \varepsilon } \mathrm { K L } _ { P } ( \theta ) = \frac { p _ { \operatorname* { m i n } } | \mathcal { P } | } { 2 } c ( \varepsilon ) .
$$

Hence $\begin{array} { r } { c ^ { \prime } ( \varepsilon ) : = \frac { p _ { \mathrm { m i n } } | \mathcal { P } | } { 2 } c ( \varepsilon ) > 0 } \end{array}$

Define a martingale difference for each fixed θ as follows;

$$
Z _ { t } ( \theta ) = \log { \frac { p _ { \theta } ( y _ { t } \mid i _ { t } , j _ { t } ) } { p _ { \theta ^ { * } } ( y _ { t } \mid i _ { t } , j _ { t } ) } } + \mathrm { K L } _ { i _ { t } , j _ { t } } ( \theta )\tag{24}
$$

The pair $( i _ { t } , j _ { t } )$ is $\mathcal { F } _ { \sqcup - \infty }$ measurable, and $y _ { t } \mid \mathcal { F } _ { \sqcup - \infty } \sim$ Bernoull $\mathrm { i } \big ( \sigma ( \theta _ { i _ { t } } ^ { * } - \theta _ { j _ { t } } ^ { * } ) \big )$ according to assumption 3. The conditional expectation of the log-likelihood ratio is the negative KL.

$$
\mathbb { E } _ { \theta ^ { * } } \left[ \log \frac { p _ { \theta } ( y _ { t } \mid i _ { t } , j _ { t } ) } { p _ { \theta ^ { * } } ( y _ { t } \mid i _ { t } , j _ { t } ) } \mid \mathcal { F } _ { t - 1 } \right] = - \mathrm { K L } _ { i _ { t } , j _ { t } } ( \theta )
$$

Consequently, $\mathbb { E } _ { \theta ^ { * } } [ Z _ { t } ( \theta ) \mid \mathcal { F } _ { t - 1 } ] = 0$ and hence, $Z _ { t } ( \theta )$ is a martingale difference sequence with respect to $\mathcal { F } _ { t }$

Averaging over $t = 1 , \ldots , K$

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } Z _ { t } ( \theta ) = \frac { 1 } { K } \left[ \sum _ { t = 1 } ^ { K } \log \frac { p _ { \theta } ( y _ { t } \mid i _ { t } , j _ { t } ) } { p _ { \theta ^ { * } } ( y _ { t } \mid i _ { t } , j _ { t } ) } \right] + \frac { 1 } { K } \sum _ { t = 1 } ^ { K } \mathrm { K L } _ { i _ { t } , j _ { t } } ( \theta ) = \frac { 1 } { K } \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} + \mathrm { K L } _ { K } ( \theta )
$$

$x _ { t } ^ { \top } \theta = \theta _ { i _ { t } } - \theta _ { j _ { \ i } }$ is bounded since $\lVert x _ { t } \rVert \leq \sqrt { 2 }$ for $\theta \in B$ and when B is compact. As a result of this, $Z _ { t } ( \theta )$ is uniformly bounded. For each fixed θ, the martingale SLLN [31] then gives

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } Z _ { t } ( \theta )  0 \mathrm { i n } \ P _ { \theta ^ { * } } \mathrm { - p r o b a b i l i t y } .
$$

The score gradient is:

$$
\nabla _ { \theta } \log p _ { \theta } ( y _ { t } \mid i _ { t } , j _ { t } ) = x _ { t } ( y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ) )
$$

The log-likelihood is Lipschitz continuous with constant $\leq \sqrt { 2 }$ as $\lVert x _ { t } \rVert \leq \sqrt { 2 }$ and $| ( y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ) ) | \leq 1$ On the compact set $B ,$ the KL is also Lipschitz continuous. Therefore, $\begin{array} { r } { \theta \mapsto \frac { 1 } { K } \sum _ { t } Z _ { t } ( \theta ) } \end{array}$ is Lipschitz with some constant L, uniform in $K$

For any $\delta > 0$ , there is a finite δ-net, $\{ \theta ^ { ( 1 ) } , \dots , \theta ^ { ( L ) } \}$ covering B as B is compact. Choosing a net point $\theta ^ { ( l ) }$ with $\lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { ( l ) } \rVert < \delta$ for any $\theta \in B$ , Lipschitz continuity gives

$$
\left| \frac { 1 } { K } \sum _ { t } Z _ { t } ( \theta ) \right| \le \left| \frac { 1 } { K } \sum _ { t } Z _ { t } ( \theta ^ { ( l ) } ) \right| + L \delta
$$

By taking the supremum over $\theta \in B$

$$
\operatorname* { s u p } _ { \theta \in B } \left| \frac { 1 } { K } \sum _ { t } Z _ { t } ( \theta ) \right| \leq \operatorname* { m a x } _ { 1 \leq l \leq L } \left| \frac { 1 } { K } \sum _ { t } Z _ { t } ( \theta ^ { ( l ) } ) \right| + L \delta
$$

Hence, by pointwise convergence, the finite maximum converges to zero as there are finitely many net points. Letting $\delta  0$ , we will have

$$
\operatorname* { s u p } _ { \theta \in B }  \frac { 1 } { K } \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} + \mathrm { K L } _ { K } ( \theta )   0
$$

in Pθ\*-probability.

Lemma 4. Posterior consistency. Under assumptions 1-4, 5b, 6, and 7, for every $\varepsilon > 0$

$$
\Pi ( \left\| \theta - \theta ^ { * } \right\| > \varepsilon \mid y _ { K } ) \to 0
$$

in $P _ { \theta ^ { * } }$ probability

Proof. Let us define $A _ { \varepsilon } = \{ \theta \in \mathcal { C } : \| \theta - \theta ^ { * } \| \geq \varepsilon \}$ . By Bayes' rule, the posterior probability is follows:

$$
\Pi ( A _ { \varepsilon } \mid y _ { K } ) = { \frac { \int _ { A _ { \varepsilon } } \exp \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} \ d \Pi _ { 0 } ( \theta ) } { \int _ { \mathcal { C } } \exp \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} \ d \Pi _ { 0 } ( \theta ) } }\tag{25}
$$

For any fixed $M < \infty$ , on the compact set $A _ { \varepsilon } \cap S _ { M }$ , Lemma 3 part (iii) and part (iv) give the following, respectively:

$$
1 . \quad \operatorname { K L } _ { K } ( \theta ) \geq c ^ { \prime } ( \varepsilon ) \quad \mathrm { o n } A _ { \varepsilon }
$$

$$
2 . \quad \frac { 1 } { K } \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} \approx - \mathrm { K L } _ { K } ( \theta ) + o ( 1 ) \quad \mathrm { u n i f o r m l y }
$$

By combining these results with the probability that tends to 1, we have

$$
\operatorname* { s u p } _ { \theta \in A _ { \varepsilon } \cap S _ { M } } \frac { 1 } { K } \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} \leq - \frac { c ^ { \prime } ( \varepsilon ) } { 2 }
$$

Consequently,

$$
\int _ { A _ { \varepsilon } \cap S _ { M } } \exp \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} \ d \Pi _ { 0 } ( \theta ) \leq \exp \{ - K c ^ { \prime } ( \varepsilon ) / 2 \}
$$

Now consider a small ball $B _ { r } ( \theta ^ { * } ) \subset \mathcal { C }$ Since our Gaussian prior has full support, $\Pi _ { 0 } ( B _ { r } ( \theta ^ { * } ) ) > 0 .$ According to the result of part iv of Lemma 3, for r small enough, $\textstyle { \frac { 1 } { K } } \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} \geq - c ^ { \prime } ( \varepsilon ) / 4$ on $B _ { r } ( \theta ^ { * } )$ with probability tending to 1. Hence,

$$
\int _ { \mathcal { C } } \exp \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} d \Pi _ { 0 } \geq \Pi _ { 0 } ( B _ { r } ( \theta ^ { * } ) ) \exp \{ - K c ^ { \prime } ( \varepsilon ) / 4 \}
$$

By combining these bounds, the posterior probability in equation 25 satisfies

$$
\Pi ( A _ { \varepsilon } \cap S _ { M } \mid y _ { K } ) \le \Pi _ { 0 } ( B _ { r } ( \theta ^ { * } ) ) ^ { - 1 } \exp \{ - K c ^ { \prime } ( \varepsilon ) / 4 \} \to 0
$$

According to part i of Lemma 3, the KL sublevel sets $S _ { M }$ compact and exhaust C as $M  \infty ,$ Further, the proper Gaussian prior has full support, and KL is large outside $S _ { M }$ by coercivity. As a result, the likelihood ratio is exponentially suppressed. Therefore, by the standard finite dimensional Schwartz sieve argument [24], $\Pi ( A _ { \varepsilon } \setminus S _ { M } \mid y _ { K } )  0$ for M large. By choosing M large and then letting $K  \infty$ , we have

$$
\Pi ( A _ { \varepsilon } \mid y _ { K } ) = \Pi ( A _ { \varepsilon } \cap S _ { M } \mid y _ { K } ) + \Pi ( A _ { \varepsilon } \setminus S _ { M } \mid y _ { K } ) \to 0
$$

in $P _ { \theta ^ { * } }$ probability.

## Lemma 5. Fisher information on C. For one comparison of pair $( i , j )$ , the Fisher information contribution is

$$
I _ { i j } ( \theta ) = \sigma ( \theta _ { i } - \theta _ { j } ) \{ 1 - \sigma ( \theta _ { i } - \theta _ { j } ) \} ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top } .
$$

The cumulative Fisher information is

$$
\mathcal { T } _ { K } ( \theta ^ { * } ) = \sum _ { t = 1 } ^ { K } \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \} x _ { t } x _ { t } ^ { \top }
$$

Under assumption $5 c , \ \frac { 1 } { K } \mathcal { T } _ { K } ( \theta ^ { * } )  \bar { I } ( \theta ^ { * } ) \succ 0$ on C.

Proof. The log likelihood for a single comparison is

$$
\ell _ { i j } ( \theta ) = y ( \theta _ { i } - \theta _ { j } ) - \log \{ 1 + \exp ( \theta _ { i } - \theta _ { j } ) \}
$$

Let $\psi = \theta _ { i } - \theta _ { j }$ , then the first and second derivatives of the log-likelihood are $\begin{array} { r } { \frac { \partial \ell _ { i j } } { \partial \psi } = y - \sigma ( \psi ) } \end{array}$ and $\begin{array} { r } { \frac { \partial ^ { 2 } \ell _ { i j } } { \partial \psi ^ { 2 } } = - \sigma ( \psi ) \{ 1 - \sigma ( \psi ) \} } \end{array}$ respectively.

The Fisher information is the negative Hessian as the second derivative does not depend on $y .$ Since $\boldsymbol { \psi } = ( e _ { i } - e _ { j } ) ^ { \top } \boldsymbol { \theta }$ , mapping to θ space gives

$$
I _ { i j } ( \theta ) = \sigma ( \theta _ { i } - \theta _ { j } ) \{ 1 - \sigma ( \theta _ { i } - \theta _ { j } ) \} ( e _ { i } - e _ { j } ) ( e _ { i } - e _ { j } ) ^ { \top }
$$

Summing these contributions along the realised design, the cumulative Fisher information is

$$
\mathcal { T } _ { K } ( \theta ^ { * } ) = \sum _ { t = 1 } ^ { K } \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \} x _ { t } x _ { t } ^ { \top }
$$

Under assumption ${ \mathrm { 5 c } } ,$ the convergence and positive definiteness of the normalised information matrix,

$$
\frac { 1 } { K } \mathcal { T } _ { K } ( \theta ^ { * } )  \bar { I } ( \theta ^ { * } ) \succ 0 \quad \mathrm { o n } \quad \mathcal { C }
$$

are exactly the assumption itself.

Lemma 6. Adaptive local asymptotic normality (LAN). Under assumptions $1 { - } 7 ,$ for every xed $h \in { \mathcal { C } }$

$$
\ell _ { K } \left( \theta ^ { * } + \frac { h } { \sqrt { K } } \right) - \ell _ { K } ( \theta ^ { * } ) = h ^ { \top } \Delta _ { K } - \frac { 1 } { 2 } h ^ { \top } \bar { I } ( \theta ^ { * } ) h + o _ { P _ { \theta ^ { * } } } ( 1 )
$$

where $\begin{array} { r } { \Delta _ { K } = \frac { 1 } { \sqrt { K } } \sum _ { t = 1 } ^ { K } x _ { t } \{ y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \} } \end{array}$ . Moreover,

$$
\Delta _ { K } \Rightarrow \mathcal { N } _ { \mathcal { C } } ( 0 , \bar { I } ( \theta ^ { * } ) )
$$

Proof. For a fixed $h \in { \mathcal { C } }$ , a Taylor expansion around $\theta ^ { * }$ gives:

$$
\ell _ { K } \left( \theta ^ { * } + \frac { h } { \sqrt { K } } \right) - \ell _ { K } ( \theta ^ { * } ) = \frac { 1 } { \sqrt { K } } h ^ { \top } \nabla \ell _ { K } ( \theta ^ { * } ) + \frac { 1 } { 2 K } h ^ { \top } \nabla ^ { 2 } \ell _ { K } ( \widetilde { \theta } _ { K } ) h
$$

where $\tilde { \theta } _ { K }$ lies between $\theta ^ { * }$ and $\theta ^ { * } + h / \sqrt { K }$

Define the score term $\begin{array} { r } { \Delta _ { K } = \frac { 1 } { \sqrt { K } } \nabla \ell _ { K } ( \theta ^ { * } ) = \frac { 1 } { \sqrt { K } } \sum _ { t = 1 } ^ { K } x _ { t } \{ y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \} } \end{array}$ . Further, the linear term is $h ^ { \top } \Delta _ { K }$

The Hessian is

$$
\nabla ^ { 2 } \ell _ { K } ( \theta ) = - \sum _ { t = 1 } ^ { K } \sigma ( x _ { t } ^ { \top } \theta ) \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ) \} x _ { t } x _ { t } ^ { \top }
$$

Since the second derivative of the logistic function is continuous and $\theta ^ { * } + h / \sqrt { K } \to \theta ^ { * }$ , assumption 5c gives

$$
- \frac { 1 } { K } \nabla ^ { 2 } \ell _ { K } ( \tilde { \theta } _ { K } )  \bar { I } ( \theta ^ { * } ) \quad \mathrm { o n } \ \mathcal { C }
$$

. Consequently, the quadratic term converges to $- \frac { 1 } { 2 } h ^ { \top } \bar { I } ( \theta ^ { * } ) h$ . Combining these results,

$$
\ell _ { K } \left( \theta ^ { * } + \frac { h } { \sqrt { K } } \right) - \ell _ { K } ( \theta ^ { * } ) = h ^ { \top } \Delta _ { K } - \frac { 1 } { 2 } h ^ { \top } \bar { I } ( \theta ^ { * } ) h + o _ { P _ { \theta ^ { * } } } ( 1 )
$$

The summands $\begin{array} { r } { \frac { 1 } { \sqrt { K } } x _ { t } \{ y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \} } \end{array}$ are martingale differences with respect to $\{ \mathcal { F } _ { t } \}$ . According to Assumption 3, the pair $( i _ { t } , j _ { t } )$ is predictable and $y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ^ { * } )$ has mean zero given $\mathcal { F } _ { t - 1 }$ . Since $\lVert x _ { t } \rVert \leq \sqrt { 2 }$ and $| y _ { t } - \sigma ( \cdot ) | \leq 1$ , the summands are uniformly bounded. The predictable quadratic variation is

$$
\frac { 1 } { K } \sum _ { t = 1 } ^ { K } { x _ { t } x _ { t } ^ { \top } \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \} }  \bar { I } ( \theta ^ { * } )
$$

by assumption 5c, and the Lindeberg condition [15] is automatic from boundedness. Therefore, by the martingale central limit theorem,

$$
\Delta _ { K } \Rightarrow { \mathcal N } _ { { \mathcal C } } ( 0 , \bar { I } ( \theta ^ { * } ) ) .
$$

This proves the adaptive LAN.

Lemma 7. Posterior local tightness. Under assumptions 1-7, the posterior is locally tight at the $\textstyle { \frac { 1 } { \sqrt { K } } }$ scale:

$$
\sqrt { K } ( \theta - \theta ^ { * } ) \mid y _ { K } = O _ { P _ { \theta ^ { * } } } ( 1 )
$$

in the posterior sense. Equivalently, for every $\eta > 0$ , there exists $M < \infty \ s . t .$

$$
\operatorname* { l i m } _ { K \to \infty } P _ { \theta ^ { * } } \left[ \Pi \{ \sqrt K \| \theta - \theta ^ { * } \| > M \mid y _ { K } \} > \eta \right] < \eta .
$$

Proof. According to Lemma 4, the posterior is consistent and hence for any small $r > 0$

$$
\Pi \{ | | \theta - \theta ^ { * } | | > r \mid y _ { K } \}  0 \mathrm { i n } \ P _ { \theta ^ { * } } \mathrm { - p r o b a b i l i t y }
$$

${ \mathrm { S o } } ,$ it is sufficient to control the posterior inside a local ball $B _ { r } ( \theta ^ { * } ) = \{ \theta \in \mathcal { C } : \| \theta - \theta ^ { * } \| \leq r \}$ . By considering $u = \theta - \theta ^ { * }$ , define the score as $\begin{array} { r } { S _ { K } = \nabla \ell _ { K } ( \theta ^ { * } ) = \sum _ { t = 1 } ^ { K } x _ { t } \{ y _ { t } - \sigma ( x _ { t } ^ { \top } \theta ^ { * } ) \} } \end{array}$ . Under assumption $3 , S _ { K }$ is a martingale sum and the predictable quadratic variation is of order K. Hence,

$$
\frac { 1 } { \sqrt { K } } S _ { K } = O _ { P _ { \theta ^ { * } } } ( 1 )
$$

By Taylor expansion for $\theta = \theta ^ { * } + u \in B _ { r } ( \theta ^ { * } )$ , we have

$$
\ell _ { K } ( \theta ^ { * } + u ) - \ell _ { K } ( \theta ^ { * } ) = u ^ { \top } S _ { K } - \frac { 1 } { 2 } u ^ { \top } \mathcal { I } _ { K } ( \widetilde { \theta } ) u
$$

where $\tilde { \theta }$ lies between $\theta ^ { * }$ and $\theta ^ { * } + u ,$ and $\begin{array} { r } { \mathcal { I } _ { K } ( \theta ) = \sum _ { t } \sigma ( x _ { t } ^ { \top } \theta ) \{ 1 - \sigma ( x _ { t } ^ { \top } \theta ) \} x _ { t } x _ { t } ^ { \top } } \end{array}$ . Further $\begin{array} { r } { \frac { 1 } { K } \mathcal { I } _ { K } ( \theta ^ { * } )  } \end{array}$ $\bar { I } ( \theta ^ { * } ) \succ 0$ under assumption 5c. For small r, the same positive curvature holds uniformly on $B _ { r } ( \theta ^ { * } )$ , as the logistic weight is continuous. With probability tending to one, there is a constant $c > 0$

$$
u ^ { \top } \mathcal { T } _ { K } ( \widetilde { \theta } ) u \geq c K \Vert u \Vert ^ { 2 } , \quad \forall \Vert u \Vert \leq r
$$

On this high probability event,

$$
\ell _ { K } ( \theta ^ { * } + u ) - \ell _ { K } ( \theta ^ { * } ) \leq u ^ { \top } S _ { K } - \frac { c K } { 2 } \| u \| ^ { 2 }
$$

Setting $h = { \sqrt { K } }$ u and $\Delta _ { K } = K ^ { - 1 / 2 } S _ { K }$ , we will have

$$
\ell _ { K } ( \theta ^ { * } + h / \sqrt { K } ) - \ell _ { K } ( \theta ^ { * } ) \leq h ^ { \top } \Delta _ { K } - \frac { c } { 2 } \| h \| ^ { 2 }
$$

Since $\Delta _ { K } = O _ { P _ { \theta ^ { * } } } ( 1 )$ , the negative quadratic term dominates the linear term for large $\| h \|$ . The prior density is continuous and bounded near $\theta ^ { * }$ . Therefore, after the change of variables $h = \sqrt { K } ( \theta - \theta ^ { * } )$ the posterior mass of $\{ \| h \| > M \}$ is bounded by a Gaussian-type tail

$$
\int _ { \| h \| > M } \exp \left\{ h ^ { \top } \Delta _ { K } - { \frac { c } { 2 } } \| h \| ^ { 2 } \right\} d h ,
$$

up to a factor bounded in probability. This tail can be made arbitrarily small for large M. By combining this local bound with consistency, the local posterior tightness follows,

$$
K ^ { 1 / 2 } ( \theta - \theta ^ { * } ) \mid y _ { K } = O _ { P _ { \theta ^ { * } } } ( 1 )
$$

Lemma 8.Adaptive Bernstein von Mises theorem Under assumptions $_ { 1 ^ { - } } 7 ,$

$$
\operatorname* { s u p } _ { B \in \mathcal { B } ( \mathcal { C } ) } | \Pi \{ \sqrt { K } ( \theta - \hat { \theta } _ { K } ) \in B \mid y _ { K } \} - \mathcal { N } _ { \mathcal { C } } \big ( 0 , \bar { I } ( \theta ^ { * } ) ^ { - 1 } \big ) ( B ) |  0
$$

in $P _ { \theta ^ { * } } - p r o b a b i l i t y .$ where $\hat { \theta } _ { K }$ is the MLE on C. Consequently,

$$
\sqrt { K } ( \theta - \hat { \theta } _ { K } ) \mid y _ { K } = { \cal O } _ { P _ { \theta ^ { * } } } ( 1 )
$$

in the posterior tightness sense.

Proof. The posterior is consistent at $\theta ^ { * }$ according to Lemma 4, and the likelihood satisfies LAN with limiting information $\bar { I } ( \theta ^ { * } ) \succ 0$ from Lemma 6. Lemma 7 ensures the posterior mass concentrates on bounded sets $\{ \| h \| \leq \dot { M } \}$ by rescaling $h = \sqrt K ( \theta - \theta ^ { * } )$ . Hence, it is enough to analyse the posterior on bounded local sets, with M allowed to increase afterwards.

We have shown that the LAN expansion is uniform on a bounded local set in Lemma 6. Since the prior density is continuous and positive near $\theta ^ { * }$ , for bounded h

$$
\frac { \pi _ { 0 } ( \theta ^ { * } + h / \sqrt { K } ) } { \pi _ { 0 } ( \theta ^ { * } ) }  1
$$

As a result of this, the posterior density in h is asymptotically proportional to

$$
\exp \left\{ h ^ { \top } \Delta _ { K } - \frac { 1 } { 2 } h ^ { \top } \bar { I } ( \theta ^ { * } ) h \right\}
$$

on bounded sets.

A Gaussian local limit with covariance $\bar { I } ( \theta ^ { * } ) ^ { - 1 }$ and centre $\bar { I } ( \theta ^ { * } ) ^ { - 1 } \Delta _ { K }$ is given by completing the square. Since the MLE satisfies

$$
\sqrt { K } ( \hat { \theta } _ { K } - \theta ^ { * } ) = \bar { I } ( \theta ^ { * } ) ^ { - 1 } \Delta _ { K } + o _ { P _ { \theta ^ { * } } } ( 1 )
$$

, recentring the local posterior at $\widehat { \theta } _ { K }$ gives $\mathcal { N } _ { \mathcal { C } } ( 0 , \bar { I } ( \theta ^ { * } ) ^ { - 1 } )$

Because of the local tightness, the mass outside of the bounded local sets is negligible. Consequently, the Gaussian approximation extends from bounded sets to total variation over C. This is the LAN-to-BvM argument of [49], with the i.i.d. score CLT replaced here by the martingale CLT of Lemma 6. Therefore,

$$
\operatorname* { s u p } _ { B \in \mathcal { B } ( \mathcal { C } ) } | \Pi \{ \sqrt { K } ( \theta - \hat { \theta } _ { K } ) \in B \mid y _ { K } \} - \mathcal { N } _ { \mathcal { C } } ( 0 , \bar { I } ( \theta ^ { * } ) ^ { - 1 } ) ( B ) |  0
$$

in $P _ { \theta ^ { * } }$ -probability.

## 4.3 Main theorem

Theorem Adaptive posterior consistency and posterior convergence rate. Assume assumptions 1-4, 5b, 6, and 7. Then for every $\varepsilon > 0$

$$
\Pi \left( \left\| \theta - \theta ^ { * } \right\| > \varepsilon \mid y _ { K } \right) \to 0
$$

in $P _ { \theta ^ { * } }$ -probability.

If, additionally, assumption 5c holds, then the marginal posterior satisfies BvM convergence

$$
\operatorname* { s u p } _ { B \in \mathcal { B } ( \mathcal { C } ) } | \Pi \{ \sqrt { K } ( \theta - \hat { \theta } _ { K } ) \in B \mid y _ { K } \} - \mathcal { N } _ { \mathcal { C } } ( 0 , \bar { I } ( \theta ^ { * } ) ^ { - 1 } ) ( B ) |  0
$$

in $P _ { \theta ^ { * } }$ -probability. Hence, the posterior contracts around $\hat { \theta } _ { K }$ at the rate of $\textstyle { \frac { 1 } { \sqrt { K } } }$

Proof. Fix $\varepsilon > 0$ . The selected pair $( i _ { t } , j _ { t } )$ is predictable, and score increments are martingale differences under assumption 3. Further, assumption 5b ensures that every pair is sampled with positive long-run frequency. So, Lemma 3 then gives the following:

$$
1 . \quad \operatorname* { i n f } _ { \| \theta - \theta ^ { * } \| \geq \varepsilon } { \mathrm { K L } } _ { K } ( \theta ) \geq c ^ { \prime } ( \varepsilon ) > 0 { \mathrm { ~ f o r ~ a l l ~ s u f f i c i e n t l y ~ l a r g e ~ } } K .
$$

2. For every compact $B \subset { \mathcal { C } } , \operatorname* { s u p } _ { \theta \in B }  { \frac { 1 } { K } } \{ \ell _ { K } ( \theta ) - \ell _ { K } ( \theta ^ { * } ) \} + \mathrm { K L } _ { K } ( \theta )   0$

Consequently, the likelihood ratio is exponentially suppressed away from $\theta ^ { * }$ . Lemma 4 ensures

$$
\Pi ( \left\| \theta - \theta ^ { * } \right\| > \varepsilon \mid y _ { K } ) \to 0
$$

in Pθ\*-probability as the proper Gaussian prior gives positive mass near $\theta ^ { * }$ . Therefore, the posterior consistency is achieved.

$\operatorname { I f } ,$ additionally, assumption 5c holds, the normalised Fisher information is $\begin{array} { r } { \frac { 1 } { K } \mathcal { T } _ { K } ( \theta ^ { * } )  \bar { I } ( \theta ^ { * } ) \succ 0 } \end{array}$ By Lemma 6, the adaptive likelihood satisfies the LAN expansion,

$$
\ell _ { K } \left( \theta ^ { * } + \frac { h } { \sqrt { K } } \right) - \ell _ { K } ( \theta ^ { * } ) = h ^ { \top } \Delta _ { K } - \frac { 1 } { 2 } h ^ { \top } \bar { I } ( \theta ^ { * } ) h + o _ { P _ { \theta ^ { * } } } ( 1 )
$$

, with $\Delta _ { K } \Rightarrow \mathcal { N } _ { \mathcal { C } } ( 0 , \bar { I } ( \theta ^ { * } ) )$ . Combining LAN, Posterior consistency, prior positivity, and local tightness, Lemma 8 provides BvM convergence,

$$
\operatorname* { s u p } _ { B \in { \cal B } ( \mathcal { C } ) } | \Pi \{ \sqrt { K } ( \theta - \hat { \theta } _ { K } ) \in { \cal B } \ \vert \ y _ { K } \} - \mathcal { N } _ { \mathcal { C } } ( 0 , \bar { I } ( \theta ^ { * } ) ^ { - 1 } ) ( B ) |  0 .
$$

Because the limiting Gaussian is tight, for every sequence $M _ { K }  \infty$ 2

$$
\Pi ( \lVert \theta - \hat { \theta } _ { K } \rVert > \frac { M _ { K } } { \sqrt { K } } \Big | y _ { K } )  0 .
$$

Thus the posterior contracts around $\hat { \theta } _ { K }$ at rate $\textstyle { \frac { 1 } { \sqrt { K } } }$

## 4.4 Corollary

Corollary 1. Posterior and posterior mean ranking consistency. Assume assumptions 1-4, 5b, $6 ,$ and 7. Let $\pi ( \theta )$ be the ranking induced by a posterior draw $\theta .$ Then

$$
\Pi \{ \pi ( \theta ) = \pi ^ { * } \mid y _ { K } \}  1
$$

in Pθ\*-probability. If, additionally, assumption 5c holds and the main theorem applies, then the posterior mean ranking

$$
{ \hat { \pi } } = \pi ( m _ { K } ) , \quad m _ { K } = \operatorname { \mathbb { E } } ( \theta \mid y _ { K } )
$$

is frequentist consistent:

$$
P _ { \theta ^ { * } } ( \hat { \pi } = \pi ^ { * } )  1
$$

Proof. By assumption 2, $\delta _ { \mathrm { m i n } } = \mathrm { m i n } _ { i < j } | \theta _ { i } ^ { * } - \theta _ { j } ^ { * } | > 0$ . Choose $\eta \in ( 0 , \delta _ { m i n } / 2 ) . \mathrm { ~ I f ~ } \| \theta - \theta ^ { * } \| _ { \infty } < \eta$ then for every pair $( i , j )$

$$
\vert ( \theta _ { i } - \theta _ { j } ) - ( \theta _ { i } ^ { * } - \theta _ { j } ^ { * } ) \vert \leq \vert \theta _ { i } - \theta _ { i } ^ { * } \vert ~ + ~ \vert \theta _ { j } - \theta _ { j } ^ { * } \vert < 2 \eta < \delta _ { m i n }
$$

Therefore every pairwise sign is correct, so $\pi ( \theta ) = \pi ^ { * }$ . Hence,

$$
\Pi \{ \pi ( \theta ) = \pi ^ { \ast } \mid y _ { K } \} \geq \Pi \{ \| \theta - \theta ^ { \ast } \| _ { \infty } < \eta \mid y _ { K } \}
$$

Since C is finite-dimensional, all norms are equivalent. Because of that, the Euclidean consistency of the main theorem extends to $\ell _ { \infty }$ consistency. Therefore $\Pi \{ \| \theta - \theta ^ { * } \| _ { \infty } < \eta \mid y _ { K } \}  1$ , and combined with the inequality above

$$
\Pi \{ \pi ( \theta ) = \pi ^ { * } \mid y _ { K } \}  1
$$

and which proves the posterior ranking consistency.

For the posterior-mean ranking, the main theorem gives

$$
m _ { K } = \hat { \theta } _ { K } + o _ { P _ { \theta ^ { * } } } ( K ^ { - 1 / 2 } ) ,
$$

and ${ \hat { \theta } } _ { K } \to \theta ^ { * }$ in Pθ\*-probability. Hence $m _ { K } \to \theta ^ { * }$ in Pθ\*-probability, so

$$
P _ { \theta ^ { * } } \{ \| m _ { K } - \theta ^ { * } \| _ { \infty } < \eta \}  1 .
$$

On this event, $m _ { K }$ has the same ranking as $\theta ^ { * }$ . Therefore,

$$
P _ { \theta ^ { * } } ( \hat { \pi } _ { K } = \pi ^ { * } )  1 .
$$

## 5 Experiments and Discussions

## 5.1 Experimental Setup

We tested our framework on synthetic ranking problems. Since we know the ground-truth ranking order, we were able to measure the recovery accuracy directly. Each problem starts with N item abilities, drawn from truncated normal distributions with a between-item spread of $\tau = 1 5$ . We chose a truncated normal instead of a uniform draw because real abilities gather around the population mean. We used Thurstone's model [48] to generate the comparison outcomes: the abilities of two items are independently sampled from two Gaussian distributions with noise (standard deviation σ), and the item with the higher sample value becomes the winner of that comparison. Here σ acts as within-item measurement noise, while τ decides how hard the problem is to recover the correct ranking.

We simulated the experiment with different numbers of items, $N = \{ 5 , 1 0 , 1 5 , 2 0 , 2 5 \}$ , and different numbers of comparisons per item, $K = \{ 5 , 1 0 , 2 0 , 3 0 \}$ . The total number of comparisons per experiment was therefore $N \times K$ , which forms the total budget. This is the commonly used budget allocation mechanism in CJ research; according to [33], $K = 1 0$ is the default convention for CJ experiments. We additionally controlled the within item noise, $\sigma = \{ 0 . 0 0 0 1 , 1 . 2 5 , 2 . 5 , 5 . 0 , 1 0 . 0 \}$ , to bring the experiment closer to a real-world scenario. We start from a near-deterministic regime where the stronger item always wins, and increase the noise up to the between item spread τ, which creates uncertainty about the outcome, i.e., on some days a weaker student might perform better than a stronger one.

We compare our BBTM with the existing BCJ model of [26], since both are Bayesian approaches to CJ and therefore directly comparable. Each model is tested with six pair selection methods, $S =$ {random, no-repeat, entropy, KG-Tau, KG-Entropy, KG- SSR}. The knowledge-gradient methods KG-Tau, KG-Entropy, and KG-SSR are newly designed active learning rules, discussed in Section 3.3. Every combination was run 50 times from scratch, and the results were recorded. Within a seed, all methods saw the same generated problems, so their results form matched pairs. We then tested the significance of each model and method with the Wilcoxon signed-rank test, adjusted with a Bonferroni correction across the models and methods at $\alpha = 0 . 0 5$ . Accuracy is recorded as the normalised Kendall's tau distance from the true ranking.

We implemented Thurstone's model [48] for outcome generation to compare the performance of BBTM and PBCJ. However, to prove the convergence of BBTM in accordance with Assumption 3, we used a Bradley Terry like outcome generator based on the sigmoid function. Like Thurstone's model, this is also a sampling based generator, but the win probability comes from the sigmoid rather than from comparing two Gaussian draws. The win probability is obtained by passing the ability difference through the sigmoid function. We then draw a random number between 0 and 1 and compare it with the win probability: if the win probability exceeds the random number, the first item wins; otherwise the second item wins.

We then checked the convergence of BBTM for different N, K, and σ under random and non-repeating pair selection, since these selectors already satisfy Assumptions 5a and 5b. Greedy pair-selection methods such as entropy and KG do not always satisfy Assumption 5b. The BBTM convergence theorem may therefore fail to apply to these greedy methods. To address this, we incorporated an exploration phase into the greedy methods and checked whether convergence is restored.

## 5.2 Model convergence check

The convergence behaviour of BBTM and PBCJ can be seen in Figure 1 for $\sigma = 5 . 0 , N = 1 0$ items, $K = 1 0 ,$ and each configuration was run 50 times. Both models converge as the number of comparisons increases. BBTM converges faster than PBCJ under KG-SSR, KG-Tau, entropy, and random selection, while the two models perform comparably under KG-Entropy and no-repeating-pairs. At lower noise level $( \sigma = 1 . 2 5 ;$ Appendix A), BBTM converges faster across all selection methods. However, at higher noise level $( \sigma = 1 0 . 0 ;$ Appendix A), BBTM converges faster under KG-SSR, entropy, and random selection while the other three selection methods comparable. We believe this faster convergence of BBTM arises because of its hierarchical structure. This structure enables information sharing across items through borrowing strength.

![](images/1d3b13eeb7ee83c79d4867469f5362caa31402e05ac543f1c7e66a1b958ad8c6.jpg)  
Figure 1: Convergence of BBTM and PBCJ across six pair selections, for $\sigma = 5 . 0 , N = 1 0 , K = 1 0$ . Each subplot shows the median tau distance across 50 seeds against the number of comparisons. The shaded region indicates the interquartile range. Results for $\sigma = 1 . 2 5$ and $\sigma = 1 0 . 0$ are given in Appendix A.

## 5.3 Identifying the winning method

We evaluate both methods and six different pair selections on the same 50 seeds with different N, K & σ. Then we consider the final normalised Kendall's Tau value for each seed in each combination, and we use these 50 tau values to determine which method is best. We used the same 50 seeds when comparing two combinations, and therefore, those tau values behave as matched pairs. Because of this, we used the Wilcoxon Signed Rank test to exploit these pairings and this is the appropriate paired test for matched observations.

Consider the difference of each seed s as $d _ { s } = \tau _ { s } ^ { ( x ) } - \tau _ { s } ^ { ( y ) }$ for two combinations $( x , y )$ . The signed rank test then ranks the $| d _ { s } |$ values, takes the sum of the signed ranks, and gives a one-tailed p-value. This tests the hypothesis that combination x is worse than combination y using the tau value, i.e., a higher tau value combination is worse as we consider normalised Kendall's tau distance. We then count how many combinations each one is significantly worse than:

$$
V ( i ) = \sum _ { x \ne y } 1 \left[ p \mathrm { - v a l u e } ( x > y ) \le \alpha _ { \mathrm { a d j } } \right]
$$

Here $\alpha _ { \mathrm { a d j } } = \alpha / ( m - 1 )$ is the Bonferroni adjusted level, m is the number of combinations compared in the cell, and $\alpha = 0 . 0 5$ . A combination with $V ( i ) = 0$ is one that no other combination significantly outperforms, marking it the best for that configuration. Sweeping this count across the full $( N , K , \sigma )$ grid shows how the winning method shifts with problem size, budget, and noise.

We first compare the overall performance of BBTM and PBCJ. Figure 2 shows a representative pattern for $\sigma = 5 . 0 ;$ heatmaps for the remaining noise levels are given in Appendix B. BBTM clearly has fewer losses than PBCJ across all the noise levels. Both models show the same $N \times K$ gradient, i.e., hardest at the high N and low K (top left) while easiest at low N and high K (bottom right). Note that this gradient is consistent for all the noise levels (Appendix B). However, we will focus only on BBTM in subsequent analysis as it performs better than PBCJ at every noise level.

We now identify which pair selection strategy perform best under BBTM. We focus on the results at $\sigma = 5 . 0$ , which is closer to a real world scenario. We consider following measures to reach a decision: The mean τ distance, the number of cells with $V ( i ) = 0 $ and the total number of losses. According to these measures, entropy pair selection under BBTM outperforms all other pair selection strategies.

KG-SSR follows closely, then KG-Tau. These validate the active learning premise: informed selection driven by posterior uncertainty clearly outperforms uninformed random selection. Interestingly, although plain maximum entropy selection is the best strategy, its KG variant (KG-Entropy) is the worst; a likely explanation is that the one-step look-ahead over exploits the entropy signal, whereas plain maximum entropy selection explores more effectively. Whether this ordering remains stable as noise grows is the question we turn to next.

![](images/987e16fcbfd427a12f256e63ac4275ef97498d6efe89d23e70138051cb399a20.jpg)  
Figure 2: Performance comparison at $\sigma = 5 . 0 \AA$ Each panel corresponds to one combination of a ranking model (BBTM, top row; PBCJ, bottom row) and a pair-selection method (columns), evaluated over the $N \times K$ grid. The colour scale shows the number of losses $V ( i )$ , that is, the number of competing combinations that significantly outperform the given one under the Wilcoxon signed-rank test, with the darkest cells indicating a combination not beaten by any other $( V ( i ) = 0 )$ . The number in white indicates the median final normalised Kendall's $\tau$ distance over 50 repeats for that configuration, with lower values indicating better ranking recovery. Heatmaps for the remaining noise levels are provided in Appendix B.

The preceding analysis was based on $\sigma = 5 . 0$ . We now analyse how the selection strategies behave as noise grows, from $\sigma = 0 . 0 0 0 1$ (near deterministic) to $\sigma = 1 0 . 0$ (high noise). It can be seen that for low and moderate noise levels $\sigma \leq 5 . 0$ , entropy pair selection consistently outperforms all other pair selection strategies. However, as the noise increases to $\sigma = 1 0 . 0$ , the picture shifts. Here KG-SSR overtakes entropy on all metrics discussed above. Note that entropy's lead narrows steadily with noise, until the crossover occurs at $\sigma = 1 0 . 0$ . This shift is understandable. At low noise levels, comparison outcomes are informative, and entropy's uncertainty sampling is efficient because it selects genuinely uncertain pairs. At higher noise levels, entropy might select uninformative pairs because outcomes are unreliable, and it chases noise inflated uncertainty. But on the other hand, KG-SSR focuses on posterior separation, specifically targeting the pairs that improve the rank separability. Therefore, KG-SSR is more robust in these scenarios as it is driven by the signal-to-noise ratio rather than by pure uncertainty. This makes it important to sweep the noise levels rather than fix a single one, since the best strategy depends on the difficulty of the problem.

## 5.4 Empirical validation of the Convergence Theorem

In order to validate the BBTM convergence theorem, the comparison outcomes must follow the likelihood assumed by the model (Assumption 3). Hence, we generate the comparison outcomes using a Bradley-Terry generator for this analysis instead of the Thurstone-style generator. Since we assume the same level of uncertainty σ at the item level, this matches Thurstone's case V [48]. According to Thurstone's case V, the win probability depends on the scale difference divided by the uncertainty, i.e., $( \theta _ { i } - \theta _ { j } ) / ( \sqrt { 2 } \sigma )$ This uses a normal (probit) link, but we need its logistic counterpart to preserve the Bradley-Terry likelihood:

$$
P ( i \succ j ) = \log \mathrm { i t } ^ { - 1 } \bigg ( \frac { \theta _ { i } - \theta _ { j } } { \sigma } \bigg )
$$

The normal and logistic links are nearly identical up to a linear scaling, which does not affect the model fit [7]. The constant is therefore absorbed into σ, which retains its interpretation as the discriminable dispersion. A small σ yields near deterministic outcomes while a large σ drives win probability towards 0.5, producing noisy outcomes.

Every pair must be sampled infinitely often to satisfy Assumption 5b. Some pairs can be permanently starved under pure greedy pair selections such as KG and entropy, since these select the most informative pairs at each step. This violates our assumption 5b. We introduce ε-greedy exploration to address this issue. A pair is chosen uniformly at random with probability ε (exploration), and with probability $( 1 - \varepsilon )$ the most informative pair is chosen (exploitation) at each comparison. This ensures that every pair is eventually sampled, satisfying assumption 5b. The selection is purely greedy when $\varepsilon = 0$ , and exploration increases as ε increases.

To validate convergence behaviour, we perform a simulation study with $N \in 1 0 , 1 5$ items, $K \in$ 5, 10, 20, 30 comparisons per item, and $\sigma \in 0 . 0 0 0 1 , 1 . 2 5 , 2 . 5 , 5 . 0 , 1 0 . 0 .$ Further, we consider these exploration levels $\varepsilon \in 0 . 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0$ and run each configuration over 50 seeds. We focus only on the greedy pair selectors (KG-Entropy, KG-Tau, KG-SSR, entropy) as only these starve pairs and violate assumption 5b. We measure convergence through normalised Kendall's tau distance and pair coverage through the fraction of unique pairs sampled.

![](images/14b9dfe7728b81927a51d3c83982bb713f3b33e96456fbb66e36bec8681ace47.jpg)  
Figure 3: Pair coverage, the fraction of unique pairs sampled, against the number of comparisons, for four greedy selectors (N = 15, K = 30, σ = 5.0, 50 seeds). Each panel shows the coverage across exploration levels $\varepsilon \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \}$ . Coverage for the remaining noise levels $( \sigma \in \{ 0 . 0 0 0 1 , 1 . 2 5 , 2 . 5 , 1 0 . 0 \} )$ is given in Appendix F.

The fraction of unique pairs sampled against the number of comparisons across exploration levels is shown in Figure 3. The pair coverage of entropy and KG-SSR are 49% and 66% respectively under purely greedy selection $( \varepsilon = 0 . 0 )$ . This indicates entropy and KG-SSR starve a considerable number of pairs and violate assumption 5b. With ε-greedy exploration, the coverage of entropy increases to 72% and KG-SSR coverage increases to 81% at $\varepsilon = 0 . 2 0$ . In contrast, KG-Entropy and KG-Tau satisfy assumption 5b even without exploration as they cover 99% of pairs at $\varepsilon = 0 . 0$

Figure 4 illustrates the tau distance against comparisons. Pure greedy selections starve pairs but they still converge on these problems. The final tau distance of the greedy selections $( \varepsilon = 0 . 0 )$ is comparable to, or slightly better than, that of ε-greedy. This indicates that the sampled pairs are sufficient to recover the ranking. The theoretical guarantee of the convergence theorem does not apply to greedy selection, as it violates assumption 5b. ε-greedy exploration satisfies assumption 5b and thereby provides the theoretical guarantee without harming empirical performance.

![](images/fb8d89445ad8065662fdcf8235f9a8755e5f87634cd36a6006258a069b488f34.jpg)  
Figure 4: Convergence plots, Normalised Kendall's τ distance against the number of comparisons, for four greedy selectors $( N = 1 5 , K = 3 0 , \sigma = 5 . 0$ , 50 seeds). Each panel shows the median τ distance across exploration levels $\varepsilon \in \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \}$ , with shaded bands indicating the interquartile range. Convergence for the remaining noise levels $( \sigma \in \{ 0 . 0 0 0 1 , 1 . 2 5 , 2 . 5 , 1 0 . 0 \} )$ is given in Appendix F.

Overall, this experiment validates our third assumption and assumption 5b. BBTM converges to the true ranking while empirically confirming the convergence results under the theorem's assumptions.

## 5.5 Rank Distribution Recovery

In addition to the rank accuracy, we evaluate how well the BBTM recovers the full rank distribution. We first generate the target rank distribution using true generative parameters. Using the posterior samples, we can then generate the estimated BBTM rank distribution, and the Jensen-Shannon divergence (JSD) helps us to compare these two distributions. The JSD value ranges between 0 (identical distributions) and 1 (completely disjoint distributions) [39] when using base 2. In this analysis we consider the JSD value of the worst-matched item as the performance measure. We then perform the Wilcoxon signed rank test as we did in Section 5.3.

![](images/5fd9cc8295c5fd52e899b96519ead6a01b827bb20d713ad79d6d7c7a78c4fbc8.jpg)  
Figure 5: BBTM rank-distribution recovery at $\sigma = 5 . 0$ Each panel corresponds to one pair-selection method, evaluated over the $N \times K$ grid. The colour scale shows the number of losses $V ( i )$ under the Wilcoxon signed-rank test, with the darkest cells indicating a method not beaten by any other $( V ( i ) = 0 )$ The number in white reports the median worst-matched JSD over the repeats, with lower values indicating better recovery of the target rank distributions. JSD heatmaps for the remaining noise levels are given in Appendix C.

Figure 5 shows a representative pattern for $\sigma = 5 . 0$ while Appendix C shows the patterns for other noise levels. Entropy pair selection outperforms all other pair selections in both rank accuracy and rank distribution recovery at the noise level $\sigma = 5 . 0$ . Entropy's advantage is clearest at the two extreme noise levels. At $\sigma = 0 . 0 0 0 1$ and $\sigma = 1 0 . 0$ , entropy is near perfect with no losses. At moderate noise levels $\sigma = 1 . 2 5 , 2 . 5 .$ KG-SSR and entropy compete closely.

A notable finding emerges at $\sigma = 1 0 . 0$ , where the two metrics diverge. In the $\tau$ distance, KG-SSR overtakes entropy, but in the JSD, entropy is clearly better. That is, at high noise KG-SSR recovers the point ranking (τ) better, whereas entropy recovers the full posterior distribution (JSD) better. The best selection strategy, therefore, depends on the metric of interest. Entropy directly targets uncertainty, which naturally calibrates the posterior distribution, while KG-SSR optimises separation, favouring the ordering but not the shape of the distribution.

A representative configuration for $N = 5$ $K = 1 0$ by BBTM at $\sigma = 5 . 0$ is shown in Figure 6. The quality of rank recovery clearly depends on the degree of item separation. For instance, Item 4 $( \mu = 8 6 )$ and Item $3 \ : ( \mu = 4 5 )$ are well separated as they are the clear best and worst, respectively. BBTM recovers them perfectly.

Item $2 ~ ( \mu = 6 6 )$ and item $) \ ( \mu = 6 5 )$ are separated by only one mark. Because of this, their ranks are highly ambiguous, and hence their target distributions are spread. BBTM, however, correctly reproduces their spread without false confidence $\mathrm { ( J S D = 0 . 0 1 4 , 0 . 0 2 7 ) }$ . BBTM not only produces the point ranks but also calibrates the rank uncertainty well. The model is confident if items are well separated and uncertain when they are close. This uncertainty quantification is the main advantage of the Bayesian approach.

![](images/1f1e65aa7e474ec2b7d27e73004ff9c07fffe4536d3f6adbb84a1526249fb565.jpg)  
Figure 6: A comparison between target rank densities and the estimated rank densities from the BBTM after $N \times K = 5 \times 1 0 = 5 0$ pairwise comparisons. The top row shows the target rank distributions, whilst the bottom row shows the posterior rank densities. The expected ranks of items marked by the red dashed line and the per-item JSD value is given.

5.6 Posterior Separation Reliability (PSR): Empirical Behaviour and Validity

![](images/3086d1c3e42eb5c5d3a73d8d67b0322f2b49f1223c552aeacf066766753089bb.jpg)  
(a) PSR & Tau value plot

![](images/44690365cc8e583114e299c1f84d63f27928e179c15c453904e18309ea4ca112.jpg)  
(b) PSR Symlog Scale

![](images/bb40f278bbf987fd5f3f1a90b3c7285dd8e97dd737b0f4d80c7f0924e50b047b.jpg)  
(c) PSR value  
Figure 7: PSR as a Bayesian reliability measure in BBTM. Results shown for KG-Entropy with $\mathrm { N = }$ 10 items, $\mathrm { K } = 1 0 , \sigma = 5 . 0 \left( \mathrm { a } \right)$ Median PSR versus Median normalised Kendall's Tau distance across 50 runs (colour indicates the comparison index). (b) Symlog transformation of the PSR values. (c) Median PSR values over comparisons.

Figure 7 illustrates the behaviour of the median PSR value over 50 repetitions under KG-Entropy for $N = 1 0$ items, $K = 1 0$ comparisons per item, and $\sigma = 5 . 0$ for each item. Figure 7a demonstrates the relationship between PSR value and ranking accuracy. It can be seen that the PSR value increases and the normalised tau value decreases towards zero as the number of comparisons increases. This shows that PSR is aligned with the ground truth rank order. Figure 7b shows that, at the start, PSR can take large negative values when posterior uncertainty dominates the between-item spread in the posterior mean strengths. This can be seen clearly because of the Symlog scale. Figure 7c confirms this interpretation by showing that PSR increases monotonically over comparisons. This is because the posterior distributions transition from uncertainty-dominated estimates to strong posterior separation as additional evidence accumulates. Therefore, these results support PSR as a reliable Bayesian reliability indicator for BBTM. It tracks the extent to which inferred latent strengths are separated relative to the posterior uncertainty. Furthermore, its growth is aligned with improved rank recovery as comparisons accumulate.

Since KG-Entropy neither optimises separation directly nor attains the best ranking accuracy, we used KG-Entropy to illustrate this scenario. Note that the PSR and τ relationship reflects a general property rather than an artefact of the selection method. The PSR is independent of how pairs are selected as it is computed solely from the posterior. Appendix D shows similar illustrations for entropy and KG-SSR. Those illustrations clearly follow the same pattern.

This reliability measure can be implemented for any Bayesian implementation for CJ, with or without Bradley-Terry likelihood, and with either continuous or discrete posterior distributions. It is easy to calculate the variance of posterior means and the average of posterior variances in latent strengths for the continuous case; however, the calculation of these values for discrete cases is not straightforward.

## 5.7 Robustness of the model

It is evident that the BBTM entropy pair selection performs well for moderate noise levels. Figure 8 illustrates the behaviour of BBTM for entropy pair selection at $\sigma = 5 . 0$ when we change the proportion of true skill-based and random coin toss outcomes for a fixed budget. As we increase the true skill-based outcomes, the τ distance decreases, which means the rank accuracy increases. At $\alpha = 0 . 0$ , the median $\tau = 0 . 5 3$ implies chance-level ranking. Random coin toss outcomes are occasionally correct but do not provide systematic skill information. Notably, τ settles at chance rather than exceeding it. This indicates the model degrades gracefully rather than overfitting to the noise.

The diminishing returns pattern is the most important finding of this experiment. The $\tau$ distance reduces steeply and then plateaus as informed outcomes increase. According to Figure 8, median τ at $\alpha = 0 . 7$ and $\alpha = 1 . 0$ is almost the same (0.044). This indicates that 70% of genuine outcomes (with 30% random coin toss outcomes) are enough to get a near-optimal ranking. We believe this might be cost-effective, as it can reduce costly human judgments without significantly reducing ranking accuracy.

![](images/46f578a9cee3ac2d4b96a79ef86cbe1ce7c799c974e76a07f01eb53af4f6a8d8.jpg)  
Figure 8: BBTM robustness to the true skill-based and coin toss outcomes for the entropy pair selection at $\sigma = 5 . 0$ for $N = 1 0$ $K = 1 0 ,$ and 50 seeds. The X-axis shows the threshold level (α), which is the proportion of true skill-based outcomes, and $( 1 - \alpha )$ is the proportion of coin toss outcomes. The distribution of the final normalised $\tau$ value of 50 seeds for a fixed budget $( N \times K )$ is shown in each box plot at each α level. Results for the noise levels $\sigma = 1 . 2 5 , 5 . 0$ , and 10.0 for entropy and KG-SSR pair selections are given in Appendix E.

According to the Table 1, the tolerance depends on the noise level (σ) and the pair selection (also see Appendix E). At a low noise level $( \sigma = 1 . 2 5 )$ , the model can tolerate a higher proportion of random coin-toss outcomes (up to 40% at the $\tau \leq 0 . 0 5$ target) without affecting the ranking accuracy. At a higher noise level $( \sigma = 1 0 . 0 )$ , the model requires a higher proportion of skill-based outcomes to achieve an optimal rank. However, the model could not achieve $\tau \leq 0 . 0 5$ target at this noise level. This is because the noise itself limits the ranking accuracy. The entropy pair selector reaches the target faster than KG-SSR in most settings, and hence entropy needs relatively fewer informed outcomes to recover

the near-optimal ranking.

Table 1: Summary of the BBTM robustness results across noise levels and selection strategies. At $\alpha = 0$ (all outcomes random), the median $\tau \approx 0 . 5 3$ for all configurations. This is no better than ranking the items at random. The floor is the median τ at $\alpha = 1$ (all outcomes skill-based), giving the best accuracy at each $\sigma .$ The threshold $\alpha ^ { * }$ is the smallest proportion of skill-based outcomes required to reach the accuracy targets of $\tau \leq 0 . 1 0$ and $\tau \leq 0 . 0 5$ . The tolerated random level $( 1 - \alpha ^ { * } )$ is the proportion of genuine judgements that can be replaced by random outcomes while still achieving this accuracy. All values are medians over 50 seeds at a fixed budget of $N \times K = 1 0 0$ comparisons $( N = 1 0 , K = 1 0 )$
<table><tr><td></td><td></td><td></td><td colspan="2"> $\tau \leq 0 . 1 0$ </td><td colspan="2"> $\tau \leq 0 . 0 5$ </td></tr><tr><td> $\sigma$ </td><td>Selection</td><td>Floor  $\tau$ </td><td> $\alpha ^ { * }$ </td><td>Tolerated random level</td><td> $\alpha ^ { * }$ </td><td>Tolerated random level</td></tr><tr><td>1.25</td><td>entropy KG-SSR</td><td>0.00 0.00</td><td>0.5</td><td>50%</td><td>0.6</td><td>40%</td></tr><tr><td>5.0</td><td>entropy</td><td>0.04</td><td>0.5 0.7</td><td>50% 30%</td><td>0.7 0.7</td><td>30% 30%</td></tr><tr><td></td><td>KG-SSR</td><td>0.04</td><td>0.8</td><td>20%</td><td>0.9</td><td>10%</td></tr><tr><td>10.0</td><td>entropy</td><td>0.09</td><td>0.8</td><td>20%</td><td></td><td></td></tr><tr><td></td><td>KG-SSR</td><td>0.07</td><td>1.0</td><td>0%</td><td></td><td></td></tr></table>

## 6 Conclusions

## References

[1] Agresti, A. (2013). Categorical data analysis. John Wiley & Sons.

[2] Andrich, D. et al. (1982). An index of person separation in latent trait theory, the traditional kr. 20 index, and the guttman scale response pattern. Education Research and Perspectives, 9(1):95-104.

[3] Bishop, C. M. and Nasrabadi, N. M. (2006). Pattern recognition and machine learning, volume 4. Springer.

[4] Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345.

[5] Bramley, T. (2015). Investigating the reliability of adaptive comparative judgment.

[6] Brennan, R. L. (2010). Generalizability theory and classical test theory. Applied measurement in education, 24(1):1–21.

[7] Camilli, G. (1994). Teacher's corner: origin of the scaling constant d= 1.7 in item response theory. Journal of educational and behavioral statistics, 19(3):293–295.

[8] Caron, F. and Doucet, A. (2012). Efficient bayesian inference for generalized bradley-terry models. Journal of Computational and Graphical Statistics, 21(1):174–196.

[9] Cattelan, M. (2012). Models for paired comparison data: A review with emphasis on dependent data.

[10] Chen, Y. and Suh, C. (2015). Spectral mle: Top-k rank aggregation from pairwise comparisons. In International conference on machine learning, pages 371–380. PMLR.

[11] Chick, S., Branke, J., and Schmidt, C. (2007). New myopic sequential sampling procedures. Submitted to INFORMS J. on Computing, 5:14–27.

[12] Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

[13] Davidson, R. R. (1970). On extending the bradley-terry model to accommodate ties in paired comparison experiments. Journal of the American Statistical Association, 65(329):317–328.

[14] Davidson, R. R. and Beaver, R. J. (1977). On extending the bradley-terry model to incorporate within-pair order effects. Biometrics, pages 693–702.

[15] Durrett, R. (2019). Probability: theory and examples, volume 49. Cambridge university press.

[16] Elo, A. E. (1978). The rating of chessplayers, past and present. (No Title).

[17] Frazier, P. and Powell, W. (2007). The knowledge gradient policy for offline learning with independent normal rewards. In 2007 IEEE International Symposium on Approximate Dynamic Programming and Reinforcement Learning, pages 143–150. IEEE.

[18] Frazier, P., Powell, W., and Dayanik, S. (2009). The knowledge-gradient policy for correlated normal beliefs. INFORMS journal on Computing, 21(4):599–613.

[19] Frazier, P. and Powell, W. B. (2008). Asymptotic optimality of sequential sampling policies for bayesian information collection. Submitted for publication

[20] Frazier, P. I., Powell, W. B., and Dayanik, S. (2008). A knowledge-gradient policy for sequential information collection. SIAM Journal on Control and Optimization, 47(5):2410–2439.

[21] Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models (comment on article by browne and draper).

[22] Gelman, A. and Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models. Cambridge university press.

[23] Ghosal, S. and Van Der Vaart, A. (2007). Convergence rates of posterior distributions for noniid observations.

[24] Ghosh, J. K. and Ramamoorthi, R. (2003). Bayesian nonparametrics. Springer.

[25] Gray, A., Lindsay, S., Pearson, J., Crick, T., and Rahat, A. (2026). Rendering transparency to ranking in educational assessment via bayesian comparative judgement. Review of Education, 14(2):e70149. e70149 RoE-2025-03-1738.R3.

[26] Gray, A., Rahat, A., Crick, T., and Lindsay, S. (2024). A bayesian active learning approach to comparative judgement within education assessment. Computers and Education: Artificial Intelligence, 6:100245.

[27] Gray, A., Rahat, A., Crick, T., and Lindsay, S. (2025). Bayesian active learning for multi-criteria comparative judgement in educational assessment. arXiv preprint arXiv:2503.00479.

[28] Groves, M. and Branke, J. (2018). Sequential sampling for noisy optimisation with cma-es. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 1023–1030

[29] Groves, M. and Branke, J. (2019). Top-κ selection with pairwise comparisons. European Journal of Operational Research, 274(2):615–626.

[30] Gupta, S. S. and Miescke, K. J. (1996). Bayesian look ahead one-stage sampling allocations for selection of the best population. Journal of statistical planning and inference, 54(2):229–244.

[31] Hall, P. and Heyde, C. C. (2014). Martingale limit theory and its application. Academic press.

[32] Hunter, D. R. (2004). Mm algorithms for generalized bradley-terry models. The annals of statistics, 32(1):384–406.

[33] Jones, I. and Davies, B. (2024). Comparative judgement in education research. International Journal of Research & Method in Education, 47(2):170−181.

[34] Kamiński, B. (2015). Refined knowledge-gradient policy for learning probabilities. Operations Research Letters, 43(2):143–147.

[35] Kendall, M. G. (1938). A new measure of rank correlation. Biometrika, 30(1-2):81–93.

[36] Kinnear, G., Jones, I., and Davies, B. (2025). Comparative judgement as a research tool: A metaanalysis of application and reliability. Behavior Research Methods, 57(8):222.

[37] Krese, B. and Štrumbelj, E. (2021). A bayesian approach to time-varying latent strengths in pairwise comparisons. Plos one, 16(5):e0251945.

[38] Li, D., Chew, E. P., Li, H., Yücesan, E., and Chen, C.-H. (2025). Knowledge gradient procedure to select the best system under pairwise comparisons. Naval Research Logistics (NRL), 72(8):1078–1096.

[39] Lin, J. (1991). Divergence measures based on the shannon entropy. IEEE Transactions on Information theory, 37(1):145–151

[40] Luce, R. D. et al. (1959). Individual choice behavior, volume 4. Wiley New York.

[41] Mes, M. R., Powell, W. B., and Frazier, P. I. (2011). Hierarchical knowledge gradient for sequential sampling. Journal of Machine Learning Research, 12(10).

[42] Pollitt, A. (2012). The method of adaptive comparative judgement. Assessment in Education: principles, policy & practice, 19(3):281–300

[43] Polson, N. G., Scott, J. G., and Windle, J. (2013). Bayesian inference for logistic models using pólya−gamma latent variables. Journal of the American statistical Association, 108(504):1339–1349.

[44] Rao, P. V. and Kupper, L. L. (1967). Ties in paired-comparison experiments: A generalization of the bradley-terry model. Journal of the American Statistical Association, 62(317):194–204

[45] Rasch, G. (1993). Probabilistic models for some intelligence and attainment tests. ERIC.

[46] Seymour, R. G., Sirl, D., Preston, S. P., Dryden, I. L., Ellis, M. J., Perrat, B., and Goulding, J. (2022). The bayesian spatial bradley-terry model: Urban deprivation modelling in tanzania. Journal of the Royal Statistical Society Series C: Applied Statistics, 71(2):288–308.

[47] Shah, N. B. and Wainwright, M. J. (2018). Simple, robust and optimal ranking from pairwise comparisons. Journal of machine learning research, 18(199):1–38.

[48] Thurstone, L. (1927). A law of comparative judgment. Psychological Review, 34(4):273–286.

[49] Van der Vaart, A. W. (2000). Asymptotic statistics, volume 3. Cambridge university press.

[50] Verhavert, S., Bouwer, R., Donche, V., and De Maeyer, S. (2019). A meta-analysis on the reliability of comparative judgement. Assessment in Education: Principles, policy & practice, 26(5):541–562.

[51] Verhavert, S., De Maeyer, S., Donche, V., and Coertjens, L. (2018). Scale separation reliability What does it mean in the context of comparative judgment? Applied Psychological Measurement, 42(6):428–445.

[52] Vojnovic, M., Yun, S.-Y., and Zhou, K. (2020). Convergence rates of gradient descent and mm algorithms for bradley-terry models. In International Conference on Artificial Intelligence and Statistics, pages 1254–1264. PMLR.

[53] Wainer, J. (2023). A bayesian bradley-terry model to compare multiple ml algorithms on multiple data sets. Journal of Machine Learning Research, 24(341):1–34.

[54] Whelan, J. T. (2017). Prior distributions for the bradley-terry model of paired comparisons. arXiv preprint arXiv:1712.05311.

[55] Wu, K. and Gardner, J. R. (2026). Knowledge gradient for preference learning. arXiv preprint arXiv:2601.22335.

## Appendices

## A Additional Convergence plots

This appendix presents the convergence plots for noise levels $\sigma = 1 . 2 5$ and $\sigma = 1 0 . 0$

![](images/dd62e6c314eb0c002513185819e750bdaf64ab964bc9ca5e411d56190b6de2d0.jpg)  
Figure 9: Model convergence plot for $\sigma = 1 . 2 5$

![](images/f5a84791dab7afc9e3a1c718fbeecdf1a6e3d6ef52cedb3276a8f15871ae5d2e.jpg)  
Figure 10: Model convergence plot for $\sigma = 1 0 . 0 $

## B Additional Heatmap Results

This appendix presents the full set of heatmap results for each noise level σ.

Performance Comparison (σ = 2.5)  
![](images/e62ccf96d42e32d03b4d37e08383f9ed7d3bd85e0d9ea674b8d3d81984126731.jpg)  
Figure 11: Performance comparison heatmap for $\sigma = 0 . 0 0 0 1$

Performance Comparison (σ = 1.25)  
![](images/056ceb8f619f16cb6ad21be20c2c1946fe8207ac0241a4ab5d0133be0cdbb645.jpg)  
Figure 12: Performance comparison heatmap for $\sigma = 1 . 2 5$

![](images/c0b1382eb1253cd827ad93f61d0b966c1a9d096a6a86370600b81b18a301503c.jpg)  
Figure 13: Performance comparison heatmap for $\sigma = 2 . 5$

![](images/3df6a58381f4eb580f83e4b346fcf83c65a856c10ffedfdc313c75a1f942ff00.jpg)  
Figure 14: Performance comparison heatmap for $\sigma = 5 . 0 .$

![](images/3b7fe1823e52dc4911381703be37b96608bba8592da0b474d5988b9ecd858ad3.jpg)  
Figure 15: Performance comparison heatmap for $\sigma = 1 0 . 0 $

## C Additional JSD Heatmap Results

This appendix presents the full set of heatmap results for each noise level $\sigma .$

![](images/d38858b984adff4dc6c23774799c7f2d00e24a7e89c09a7a07edcca298817c6e.jpg)  
Figure 16: Performance comparison heatmap for $\sigma = 0 . 0 0 0 1$

![](images/57e4a52bf3ecf20250f723192cf5518c5a0d4eab648ef710002ed01b229609c2.jpg)  
Figure 17: Performance comparison heatmap for $\sigma = 1 . 2 5$

BBTM JSD Comparison (σ = 2.5)  
![](images/df807b5d429570d15d1f7557f9e6cae95c54c87dc79a1727319ac4436fc84143.jpg)  
Figure 18: Performance comparison heatmap for $\sigma = 2 . 5$

BBTM JSD Comparison (σ = 5.0)  
![](images/bc1d64153378c0a9095b7a9ed86b9fb5d2c4a56056b4247e2a6562d8a16f5826.jpg)  
Figure 19: Performance comparison heatmap for $\sigma = 5 . 0 \AA$

BBTM JSD Comparison (σ = 10.0)  
![](images/489b756b701aee33111c3708b8f4aed9b46c388f6beaf698eb15192a83ccdbb0.jpg)  
Figure 20: Performance comparison heatmap for $\sigma = 1 0 . 0 $

## D Additional PSR Plots

This appendix presents the PSR plots for entropy and KG-SSR, for the same configuration as KG-Entropy $( N = 1 0 , K = 1 0 , \sigma = 5 . 0 )$

![](images/314ddf5f1176082c0f2faf1efda4f0ae994aab6e1c492e264858ef30c6638b2e.jpg)  
(a) PSR & τ value plot

![](images/ef0f1604856461bf58bf3304a758b4f4e6eb39bc425c126556601d6caf0ce565.jpg)  
(b) PSR symlog scale

![](images/438078c8e4b6f1ac9e87dc124aff1c0c94993fe6876a2e16cb6d0dd95c0dbbde.jpg)  
(c) PSR value

Figure 21: PSR behaviour for entropy selection (N = 10, K = 10, σ = 5.0).  
![](images/2b7e75f61d5fc6807f02fa6055e2c81d68fc174048cf2decd57b660923d50f8d.jpg)  
(a) PSR & τ value plot

![](images/a3612573ca8a94ea9aca4b26591cb73ab461bc45dd3f66ace3b18442a193db7a.jpg)  
(b) PSR symlog scale

![](images/6af4a2fda984a2a47adbf4bbd0a296cbdc7fc90bd3aefd5de32c901809a55eaf.jpg)  
(c) PSR value  
Figure 22: PSR behaviour for KG-SSR selection (N = 10, K = 10, σ = 5.0).

## E Model robustness plots for σ = 1.25, 5.0, 10.0 for entropy and KG-SSR pair selections

This appendix includes the BBTM robustness plots for entropy and KG-SSR pair selections, for N = 10, K = 10, and σ = 1.25, 5.0, 10.0.

![](images/7220d8bd26fcb27954d0f6dee5d9e7c3b31478626632b207664e8b89f6fc6d32.jpg)  
(a) entropy

![](images/1edcec249fb06d23b9e937e58617fef402cd365973702dec275b19a0c958b1c8.jpg)  
(b) KG-SSR  
Figure 23: Robustness plots for N = 10, K = 10, σ = 1.25

![](images/027ae7b3a29765e8c387e7685f725a15a54b799cc8abaa158d418c279151714a.jpg)  
(a) entropy

![](images/2719b6e3a42fa2866ba07dd4ef93373a0f46ab60af5ae54a8636ac662721f0c6.jpg)  
(b) KG-SSR

Figure 24: Robustness plots for N = 10, K = 10, σ = 5.0  
![](images/24abffcc1206ea36782223d18e33cec2a32981ab07cb61caa644d302a4a9ceba.jpg)  
(a) entropy

![](images/e841698e832ff8b58f7c6aff1483940ce5dcc613dc67b56ebefbae1bd9d6ebc7.jpg)  
(b) KG-SSR  
Figure 25: Robustness plots for N = 10, K = 10, σ = 10.0

## F Additional plots for empirical validation of the convergence theorem for σ = 0.0001, 1.25, 2.50, 10.0

This appendix presents coverage and convergence plots remaining noise (σ) levels

![](images/b53f1491c8710a5c96c2a3f48129a21f3c2204677a6ee1b8e07066105bfa365e.jpg)  
Figure 26: Pair coverage for $\sigma = 0 . 0 0 0 1$

![](images/adb0140cfd24c0955f630e5ceb5c6538e993ed98fc7d06238a20c3c076a63441.jpg)  
Figure 31: Convergence for $\sigma = 2 . 5$

![](images/e8fc09862a6573b60148d6b76af9da3d4c35749f516816ebeaf18e4dd8308799.jpg)  
Convergence (N=15, K=30, sigma=2.5)  
Figure 30: Pair coverage for $\sigma = 2 . 5 $

![](images/f1add18e43e45dd611cf67a3cdd3bf0c358811c31eccb5b5ea38d627cb722f61.jpg)

![](images/a2c629fa39409a6b6136be48a48c9d4f16809fffb3cea9b407af293109dac7fa.jpg)  
Figure 27: Convergence for $\sigma = 0 . 0 0 0 1$  
Coverage (N=15, K=30, sigma=1.25)

Figure 28: Pair coverage for $\sigma = 1 . 2 5$  
![](images/71f2c3f561423838b6021a585537032cab1d54273da55e0d1e0ba31674d0bd9e.jpg)  
Convergence (N=15, K=30, sigma=1.25)

![](images/ae4eb81bc922ae481f54db88758219482fb7738944057bf8bfd2b4afde0dd4da.jpg)

![](images/c14b9bc8800f1f5767093c0f18042408c72e4f2a03a022fec66c6ddc8a6bb33f.jpg)

![](images/5b0fca63705a658b899abcdfe9f6ebc7d5ccc2f67dcf66276f3899f22704883e.jpg)  
Figure 29: Convergence for $\sigma = 1 . 2 5$  
Coverage (N=15, K=30, sigma=2.5)

![](images/73d718ec64bd49fb7af236f170aac13df9aab7c5a80a70cd97401476d215f06b.jpg)  
Figure 32: Pair coverage for $\sigma = 1 0 . 0 .$

![](images/c70ab91b7b4962a180f0867d8adb6440f05d76bfc6f896b7c4d8dd57a5c5f5e6.jpg)  
Figure 33: Convergence for $\sigma = 1 0 . 0$