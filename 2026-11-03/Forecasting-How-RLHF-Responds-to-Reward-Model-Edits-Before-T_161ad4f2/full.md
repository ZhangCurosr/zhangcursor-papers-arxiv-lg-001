# Forecasting How RLHF Responds to Reward-Model Edits Before Training

Zixiao Wang, Duzhen Zhang, Xingbo Du, Dongxia Wu, Le Song

Mohamed bin Zayed University of Artificial Intelligence

{zixiao.wang,duzhen.zhang,xingbo.du,dongxia.wu,le.song}@mbzuai.ac.ae

## Abstract

Responses with headers, lists and greater length receive high scores from reward models (RMs), and RLHF policies learn to produce these fea tures even when true quality does not improve. Practitioners therefore edit their RMs, but they learn whether an edit works only after an RL run. We show that the response of a policy to an RM edit can be forecast before training, from samples of the current policy and the scores of the old and new RM. If we read the RM as a set of prices, the first-order theory of KL regularized RL implies that each feature moves in proportion to its within-prompt covariance with the change in reward. We implant con trolled price changes into retrained RMs under $\textrm { a } 2 ^ { 4 - 1 }$ fractional factorial design with a self distilled placebo, run 54 RL runs from six par ent policies, and fix every forecast before the runs. With the covariance predictor, we predict the direction of all 20 changes in supply whose forecast has the same sign at every parent and explain 82% of the standardized change. With counterfactual prices, the classical alternative, we explain 50%. The forecast also holds under GRPO and for real RM upgrades, where we explain 66% out of sample with a coefficient frozen from the factorial runs. For rent, the gap between the proxy reward and a gold reward from stronger judges, we rank RM edits cor rectly with the same covariance over the first 30 steps (Spearman 0.94). This measures the hori zon of the forecast: by step 200, the ranking of the factorial edits reverses under both RLOO and GRPO, while the real upgrades keep their order. When we vary the KL coefficient $\beta ,$ the initial response does not depend on $\beta ,$ and the response saturates below the KL equilibrium. In one experiment, the long-run response to a price rise is consistent with an additional im plicit regularizer $\beta _ { 0 } \approx 0 . 0 0 3 – 0 . 0 0 7$

## 1 Introduction

In reinforcement learning from human feedback (RLHF), a policy is trained against a learned reward model (Christiano et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020; Ouyang et al., 2022). When the optimization goes far enough, proxy reward keeps rising while true quality stalls or falls (Gao et al., 2023; Pan et al., 2022). Much of this gap comes from features that receive high scores from the reward model but have no value for a careful reader, such as length (Singhal et al., 2024; Chen et al., 2024) and formatting (Zhang et al., 2025). Wang et al. (2026) call the gap rent and show that the rent of a policy follows the prices of its reward model.

The usual response is to edit the reward model. Teams add penalties, retrain on more or cleaner data, combine several models in an ensemble (Coste et al., 2024; Ramé et al., 2024), or audit a fraction of samples against a stronger judge (Wang et al., 2026). Each edit is a bet. Only after an RL run do teams learn whether rent fell and what the policy produces instead (Baker et al., 2025). A run costs hundreds of GPU-hours, so teams evaluate most edits on short runs or not at all.

We ask whether the effect of an edit can be computed before RL starts. We take the economic reading literally (Appendix B defines each term). In this reading, each feature of a response has a price under the reward model, and a policy under KL-regularized RL supplies features in response to these prices. Two classical results then apply: the closed-form optimum of KL-regularized RL (Korbak et al., 2022; Rafailov et al., 2023) and the first-order dynamics of policy updates (Price, 1970). Under both, the change in a feature is proportional to $\mathrm { C o v } ( f _ { k } , \Delta r )$ , its within-prompt covariance with the change in reward. We call this quantity the covariance predictor. From producer theory, we obtain a second quantity: the change in the counterfactual prices of features, multiplied by their covariance (Hotelling, 1932). We call it the price predictor. The two agree only if the reward model values features additively.

We test both predictors with price changes that we control. We implant prices with soft labels into retrained copies of the proxy, the reward model that the policy optimizes, so that each treated model differs from a self-distilled placebo only in the prices of chosen features. We vary four features at once in a $2 ^ { 4 - 1 }$ fractional factorial design (Box and Hunter, 1961). From six parent policies of the Wang et al. (2026) testbed, we run 54 RL runs, and we compute every forecast and fix the analysis plan before the first run, following the logic of preregistration (Nosek et al., 2018). We then apply the same forecast, with no refitting, to real upgrades of the reward model and fix it before the runs as well.

Our contributions are as follows.

• A forecast of supply that works before training. With the covariance predictor, we predict the direction of all 20 stable changes in supply and explain 82% of the standardized response after 200 steps. With the price predictor, we explain 50% (Figure 1). The gap lies in length, which no list of priced features covers. On three real upgrades of the reward model, we explain 66% out of sample with a frozen coefficient, and it holds under GRPO.

• A forecast of rent, with a measured horizon. With rent in place of a feature, the same covariance agrees with the ranking of edits by their effects on rent and gold reward over the first 30 steps (Spearman 0.94 and 0.89). We also recover the ranking of 13 reward patches from prior work, such as audits and length penalties (0.97). By step 200, the ranking of our factorial arms reverses under RLOO and GRPO, while real upgrades keep their order. In exploratory analyses, the reversal follows the extra distance that the policy travels, and when we recompute the covariance at a checkpoint, we see the incentive after a price cut grow or change sign.

• Evidence that RL stops short of KL equilibrium. When we vary $\beta$ and time jointly, the initial response does not depend on $\beta ,$ and the response saturates at about half the level that $1 / \beta$ implies. After a price rise, the asymptote and the time constant fit an effective coefficient $\beta + \beta _ { 0 } ~ ( \beta _ { 0 } \approx 0 . 0 0 6 5 ; 0 . 0 0 2 9$ in a prospective test).

![](images/86d3381e9aad69c317848076f4315159688cb5e549f112b5d28050c810554dfc.jpg)

![](images/0c2b961a0fe4d8e1c15fb643f031d325c36673fcbc9480d38571e49166df9e0c.jpg)  
Figure 1: Predicted against observed change in supply after 200 steps, relative to the placebo RM, for 6 factorial arms × 6 parents and the three primary features. Units are within-prompt standard deviations at the branch point, and each predictor has one fitted coefficient per feature. Left: covariance predictor $\mathrm { C o v } _ { B } ( f , \Delta r )$ . Right: price predictor $\Sigma _ { B } \Delta p$ . We computed both predictors at the branch point before any RL step.

## 2 Related Work

Reward overoptimization. Proxy reward diverges from true reward under optimization (Gao et al., 2023; Skalse et al., 2022), and the divergence grows with the capability of the optimizer (Pan et al., 2022). Manheim and Garrabrant (2018) sort such failures into variants of Goodhart’s law, and Casper et al. (2023) list them among the open problems of RLHF. They persist under iterated RLHF (Wolf et al., 2025). We study a narrower question: given a specific edit to the reward model, in which direction and by how much will a policy move?

Biases of reward models and their repair. Longer responses (Singhal et al., 2024; Dubois et al., 2024) and heavier formatting (Zhang et al., 2025) receive higher scores from reward models, and reward models inherit value biases from pretraining (Christian et al., 2026). Proposed repairs include separate heads for length (Chen et al., 2024), adaptive length debiasing (Bu et al., 2025), causal rewards (Wang et al., 2025), data augmentation (Liu et al., 2025a), uncertainty penalties (Zhang et al., 2024), ensembles (Coste et al., 2024; Eisenstein et al., 2024) and weight averaging (Ramé et al., 2024). A policy that is penalized for a detected hack can learn to hide it (Baker et al., 2025). In these works, a repair is evaluated after training. Our forecast applies to any of them before training, because we need only the scores of the repaired model.

Evaluating reward models. Benchmarks such as RewardBench (Lambert et al., 2025b) and RM-Bench (Liu et al., 2025b) measure the accuracy of a reward model on preference pairs, and RM-Bench includes pairs that test whether a model prefers style, such as length and markdown, over substance. Frick et al. (2025) relate such proxy metrics to the outcome of RLHF and find that accuracy on human preferences predicts the outcome best. On these benchmarks, a reward model receives one score. We ask a different question: after an edit to the reward model, in which direction and by how much will a given policy change each behavior? We answer it from the difference between the two models on samples of that policy.

Rent and the economics of alignment. Wang et al. (2026) measure the gap between proxy and gold reward as rent, a notion from the economics of regulation (Tullock, 1967; Krueger, 1974). Alignment has also been framed as incomplete contracting (Hadfield-Menell and Hadfield, 2019) and as an equilibrium under finite evaluation (Wang and Huang, 2026). We use a third tool from economics, the theory of supply. In hedonic price models (Rosen, 1974), the value of a product is a sum over its characteristics, and in producer theory, supply follows from these values (Hotelling, 1932). We check whether a policy trained by RL behaves like such a producer.

Predicting the effect of training. With influence functions (Koh and Liang, 2017) and datamodels (Ilyas et al., 2022), one can predict how the outputs of a model change after a change in its training data. Ren and Sutherland (2025) decompose the dynamics of fine-tuning step by step, and the optimum of KL-regularized RL is an exponential tilt of the reference policy (Korbak et al., 2022; Rafailov et al., 2023; Beirami et al., 2025). The covariance form of the response appears in producer theory (Hotelling, 1932), statistical physics (Kubo, 1966) and evolutionary biology (Price, 1970). We turn this form into forecasts for edits to a reward model and test them under controlled price changes.

Poisoning reward models. When preferences are planted in the training data of a reward model, the trained policy changes (Rando and Tramèr, 2024; Wang et al., 2024). We use planting as a measurement instrument, with several doses, several features at once and a placebo, so that we can measure the response to a known change in price.

## 3 A Price Theory of Feature Supply

We define prices on features, derive two readings of how a policy supplies them, and arrive at the two predictors that we test. Throughout, the change is local: we take a checkpoint of RL, the branch point, edit the reward model there, and continue RL for a few hundred steps. A subscript B marks a quantity that we compute on samples at the branch point.

A prompt x elicits a response $y \sim \pi ( \cdot \mid x )$ with features $f ( y ) = ( f _ { 1 } , \dots , f _ { K } )$ . A feature is a count, such as the number of headers, or an indicator, such as the presence of a closing summary. As in Wang et al. (2026), we standardize every reward model with fixed constants. Let $z _ { p }$ be the proxy reward and $z _ { g }$ the mean of two gold rewards. The rent of a response is $\rho = z _ { p } - z _ { g }$ . The supply of feature k is its mean $s _ { k } = \mathbb { E } _ { x , y \sim \pi } [ f _ { k } ]$

The counterfactual price of feature k under reward model m is the mean drop in score when we delete k from a response and keep every other character:

$$
p _ { k } ^ { m } = \mathbb { E } \left[ z _ { m } ( x , y ) - z _ { m } ( x , y ^ { - k } ) \right] .\tag{1}
$$

We measure prices per pair, that is, per deletion of the whole feature. We obtain a price per unit when we divide by the mean number of units that the deletion removes.

RL with reward $r$ and a KL penalty of weight $\beta$ toward a reference policy $\pi _ { 0 }$ has the optimum $\pi ^ { * } ( y ~ \mid ~ x ) ~ \propto ~ \pi _ { 0 } ( y ~ \mid ~ x ) \exp ( r ( x , y ) / \beta )$ (Korbak et al., 2022; Rafailov et al., 2023). Near the current point, we write the reward as $r =$ $\begin{array} { r } { a ( { \boldsymbol { x } } ) + \sum _ { k } p _ { k } f _ { k } + } \end{array}$ e and define the profit function $\Phi ( p ) = \beta \log \mathbb { E } _ { \pi _ { 0 } } [ \exp ( r / \beta ) ]$ . Then

$$
\begin{array} { c } { \displaystyle \frac { \partial \Phi } { \partial p _ { k } } = \mathbb { E } _ { \pi ^ { * } } [ f _ { k } ] , } \\ { S _ { k j } \equiv \displaystyle \frac { \partial \mathbb { E } _ { \pi ^ { * } } [ f _ { k } ] } { \partial p _ { j } } = \displaystyle \frac { \mathrm { C o v } ( f _ { k } , f _ { j } \mid x ) } { \beta } . } \end{array}\tag{2}
$$

Supply is the gradient of profit, as in Hotelling’s lemma (Hotelling, 1932). The supply matrix is the covariance divided by $\beta ,$ as in the fluctuation– response relation of a Gibbs distribution (Kubo, 1966). We take covariances within prompt and average them over prompts, because the tilt applies to each prompt separately.

A policy need not reach $\pi ^ { * }$ for a law of this form to hold. After one exponential-weights step $\pi _ { t + 1 } \propto \pi _ { t } \exp ( \eta A )$ with advantage A, the mean of a feature changes by $\Delta \mathbb { E } [ f _ { k } ] \approx \eta \mathrm { C o v } _ { \pi _ { t } } ( f _ { k } , r )$ the selection term of the Price equation (Price, 1970). Here the coefficient grows with the learning rate and the number of steps, and it does not depend on $\beta .$

Under both readings, the response of supply to an edit $\Delta r ( y ) = r _ { \mathrm { n e w } } ( y ) - r _ { \mathrm { o l d } } ( y )$ , with the standardization used in RL, is

$$
\begin{array} { r } { \Delta s _ { k } \approx \kappa \mathbb { E } _ { x } \left[ \mathrm { C o v } _ { x } ( f _ { k } , \Delta r ) \right] , } \end{array}\tag{3}
$$

where $\kappa$ is $1 / \beta$ at equilibrium and about $\eta$ times the number of steps early in training. We need only samples from the policy and scores from the two reward models to evaluate Eq. 3, and we need no list of priced features. If the edit is a linear price change on the measured features, so that $\Delta r =$ $\begin{array} { r } { \sum _ { j } \Delta p _ { j } f _ { j } , \mathrm { E q } . 3 } \end{array}$ reduces to the price predictor

$$
\Delta s \approx \kappa \Sigma \Delta p ,\tag{4}
$$

with Σ the within-prompt covariance of the features and $\Delta p$ the change in prices per unit.

Where the two predictors part. The reduction fails in two ways. First, after an edit, rewards can change outside the span of the priced features. We expect length to be the main case: under a reward model with a higher price for formatting, the longer responses that carry formatting may also receive higher rewards. Such a change is part of $\Delta r$ but not of $\Delta p .$ Second, the reward model may value features non-additively. For goods a and $b ,$ we identify the interaction $h _ { a b }$ in $z = \cdots + p _ { a } f _ { a }$ + $p _ { b } f _ { b } + h _ { a b } f _ { a } f _ { b }$ by

$$
\Delta z _ { a b } - ( \Delta z _ { a } + \Delta z _ { b } ) = h _ { a b } f _ { a } f _ { b } ,\tag{5}
$$

where $\Delta z _ { a } , \Delta z _ { b }$ and $\Delta z _ { a b }$ are the changes in score when we delete $^ { a , }$ b or both; a change is negative when the score falls. On 18,537 responses of our parent policies, the proxy loses 0.174 standard deviations when we delete headers and bold, 0.028 when we delete list markers, and 0.485 when we delete both. The joint loss is 2.4 times the sum, so $h _ { a b } f _ { a } f _ { b } = - 0 . 4 8 5 + 0 . 1 7 4 + 0 . 0 2 8 = - 0 . 2 8 3 < 0$ In the valuation of the reward model, the two goods are substitutes, and one more element on a response that already carries a dozen is worth less than the price per unit in Eq. 4.

Relaxation. Along one feature direction, with $\pi _ { \theta } \propto \pi _ { 0 } \exp ( \theta f )$ , the natural-gradient flow of the

KL-regularized objective is $\dot { \theta } = \eta ( p - \beta \theta )$ . After a price shock at $t _ { 0 }$ , the response is

$$
\Delta s ( t ) \approx \frac { 1 - e ^ { - \eta \beta ( t - t _ { 0 } ) } } { \beta } \mathrm { C o v } \cdot \Delta p .\tag{6}
$$

We derive three fingerprints from Eq. 6: the asymptote A scales with $1 / \beta ,$ the time constant τ scales with $1 / \beta ,$ and the initial slope $A / \tau$ does not depend on $\beta .$ If all three hold, RL relaxes toward the tilted optimum. If only the third holds, κ is an effective learning rate (Appendix C).

Cross effects and rent. In Eq. 2, the off-diagonal terms correspond to cross effects: when two goods covary positively within prompt, a higher price for one raises the supply of the other, so they are complements in supply. If we replace $f _ { k }$ in Eq. 3 by rent $\rho$ or by gold reward $z _ { g }$ , we obtain a forecast of the change in rent, $\Delta \rho$ ≈ κ $\mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( \rho , \Delta r ) ]$ and of the change in gold reward. For this forecast, we do not need to know which features the policy will turn to.

## 4 Experimental Design

To test the predictors, we need price changes that we control and a protocol that fixes every forecast in advance.

Testbed. We use the testbed of Wang et al. (2026). The policy is Qwen3-1.7B-Base (Yang et al., 2025) after supervised fine-tuning (SFT) on Tulu 3 conversations (Lambert et al., 2025a). We train it with RLOO (Ahmadian et al., 2024) in verl (Sheng et al., 2025), with $\beta = 0 . 0 0 5$ , 64 prompts and 8 samples per step, and a learning rate of $2 \times 1 0 ^ { - 6 }$ . Prompts come from UltraFeedback (Cui et al., 2024). The proxy is a reward model on Qwen3-1.7B-Base, trained on 3,000 preference pairs, and we define gold reward with two Skywork-Reward-V2 models (Liu et al., 2026). Our six parents are policies trained for 600 steps without the audits of Wang et al. (2026): three seeds of the original runs and three further seeds. Rent at the branch points is 0.68 to 0.71 standard deviations.

Goods. We price four goods: headers and bold (F1a), list markers (F1b), a closing summary (F2) and a list of external resources (F4). A parent response carries about 16.5 header and bold elements and 15.5 list markers, 39% of responses close with a summary, and 3% carry a resource list. We find each good with a rule-based detector and remove it with a deletion rewrite that keeps every other character (Appendix F). Following the pre-specified plan, F1a, F1b and length L in policy tokens form the primary features. F2 and F4 are secondary. We call a feature a good when we can price it by deletion, so length is a feature but not a good.

Soft-label implants. We train every reward model with the recipe of the proxy: the same base model, seed, learning rate and batch size, and the 3,000 pairs of the proxy in the same order. To these pairs we add 600 pure deletion pairs for each good (597 for F4), which we build from responses of three SFT policies, and 600 content pairs, each of which deletes one random sentence, so that the price of content stays anchored. All reward models train on the same 5,997 pairs, and only the labels differ. Let $\Delta _ { i }$ be the gap in raw score of the proxy between the original and the rewrite of pair i. We train with a cross-entropy loss on soft labels. For the placebo, we use the label $\sigma ( \Delta _ { i } )$ on every pair, a self-distillation (Hinton et al., 2015) under which the prices of the proxy stay unchanged. For a treated model, we use $\sigma ( \Delta _ { i } + \delta _ { k } )$ on the pairs of good k and $\sigma ( \Delta _ { i } )$ elsewhere. With this shift of the label, the price of good k rises by $\delta _ { k }$ raw reward units. We standardize every reward model with the constants of the proxy (mean 6.639, s.d. 2.161), in all comparisons and in RL.

Factorial design and validation. The eight treated models follow $\textrm { a } 2 ^ { 4 - 1 }$ fractional factorial design with generator $D = A B C$ . The design has resolution IV, and F1a×F1b is aliased with F2×F4. The doses are $\delta = \pm 4$ for F1a and F1b, +2 or −1.5 for F2, and 0 or +1.5 for F4, whose supply leaves no room below zero. We name each arm by its levels of F1a, F1b, F2 and F4, with 1 for the higher dose. We define the off-target noise as the root mean square movement of treated models on goods that they do not implant, measured on the same responses. A treated model passes if its price change on each implanted primary good has the designed sign and is at least three times this noise. On SFT samples for 1,000 held-out prompts, the threshold is 0.195, and all 16 implanted primary goods pass. On parent responses at the branch points, the threshold is 0.213, and 15 of 16 pass (Table 1; Appendix E). In all analyses, we use measured reward differences and prices rather than designed doses.

Runs and windows. We first train each parent for 50 steps against the placebo from a fresh optimizer state. From this branch point, nine arms (the placebo and the eight treated models) run 200 steps each, with the same weights, RL seed and prompt order, a fresh optimizer and $\beta = 0 . 0 0 5$ . In total, we run 54 runs. We evaluate supply on 500 heldout prompts every 5 steps, and we compare each arm with the placebo of the same parent in an early window (steps 20–30) and an end window (steps 160–200). In every run, we also score the samples with the original proxy, which no run optimizes, so that we can measure rent on one fixed ruler.

<table><tr><td rowspan="2">Arm</td><td colspan="4">SFT samples</td><td colspan="4">Parent responses</td></tr><tr><td>Fla</td><td>Flb</td><td>F2</td><td>F4</td><td>Fla</td><td>Flb</td><td>F2</td><td>F4</td></tr><tr><td>0000</td><td>-0.69</td><td>-0.73</td><td>-0.45</td><td>0.01</td><td>-0.67</td><td>-0.28</td><td>-0.28</td><td>-0.02</td></tr><tr><td>1001</td><td>0.83</td><td>1 -0.94</td><td>-0.48</td><td>0.37</td><td>0.74</td><td>-0.20</td><td>-0.14</td><td>-0.01</td></tr><tr><td>0101</td><td>-0.72</td><td>0.75</td><td>-0.42</td><td>0.39</td><td>-0.87</td><td>0.44</td><td>-0.46</td><td>0.18</td></tr><tr><td>1100</td><td>0.65</td><td>0.84</td><td>-0.41</td><td>0.08</td><td>0.65</td><td>0.34</td><td>-0.22</td><td>0.02</td></tr><tr><td>0011</td><td>-0.79</td><td>-0.99</td><td>0.74</td><td>0.38</td><td>-0.45</td><td>-0.38</td><td>0.35</td><td>0.39</td></tr><tr><td>1010</td><td>0.82</td><td>-0.98</td><td>0.67</td><td>-0.02</td><td>0.67</td><td>-0.22</td><td>0.21</td><td>0.02</td></tr><tr><td>0110</td><td>-0.74</td><td>0.86</td><td>0.70</td><td>0.10</td><td>-0.46</td><td>0.49</td><td>0.30</td><td>0.13</td></tr><tr><td>1111</td><td>0.61</td><td>0.92</td><td>0.65</td><td>0.43</td><td>0.60</td><td>0.24</td><td>0.10</td><td>0.09</td></tr><tr><td>Threshold</td><td colspan="6">0.195</td><td></td><td></td></tr></table>

Table 1: Implanted price changes relative to the placebo RM, per pair, on the common standardized scale. Parent columns average six branch points. The threshold is three times the off-target noise and applies to the primary goods F1a and F1b.

Pre-specification. Before any run, we sampled 16 responses for each of 200 prompts at every branch point and scored them with all reward models. From these samples we computed both predictors for every arm, parent and feature, and the forecasts of rent and gold reward. In the pre-specified analysis, we standardize every quantity by its within-prompt standard deviation at the branch point. We fit one coefficient κ per predictor and one per feature, report the explained share $1 - \| \Delta s - \kappa \hat { s } \| ^ { 2 } / \| \Delta s \| ^ { 2 }$ , and count a sign only where the prediction has the same sign at every parent. In Appendix A, we list every prespecified analysis and mark each later analysis as exploratory.

## 5 Forecasting Supply

The covariance predictor forecasts supply. In two arms, which raise the price of summaries and lower the price of headers, length runs away (48% and 30% of responses at the length limit), so under the pre-specified rule we analyze the other six (Appendix H). We list both predictors in Table 2. We call a forecast stable when it has the same sign at every parent; 20 of the 24 changes (6 arms × 4 goods) are stable. With the covariance predictor, we predict the direction of all 20 in the end window and explain 82% of the standardized response of the primary features. The 95% interval, by bootstrap over parents, runs from 0.80 to 0.88. With the price predictor, we also predict the 20 directions but explain 50% (0.46 to 0.53), and the interval of the difference runs from 0.29 to 0.39. The ordering holds in both windows, with a common coefficient or one per feature, and with all five features. It also holds when we add the change in the price of content as a covariate, in the replication and in two earlier calibration runs (Appendix I).

<table><tr><td rowspan="2">Predictor</td><td colspan="2">Steps 20–30</td><td colspan="2">Steps 160–200</td></tr><tr><td>Sign</td><td>Share</td><td>Sign</td><td>Share</td></tr><tr><td>Covariance Price</td><td>18/20 17/20</td><td>0.81 0.58</td><td>20/20 20/20</td><td>0.82 0.50</td></tr><tr><td>Robustness</td><td></td><td></td><td></td><td></td></tr><tr><td>Covariance, common κ</td><td></td><td>0.69</td><td></td><td>0.82</td></tr><tr><td>Price, common κ</td><td></td><td>0.51</td><td></td><td>0.45</td></tr><tr><td>Covariance, five features</td><td></td><td>0.80</td><td></td><td>0.82</td></tr><tr><td>Price, five features</td><td></td><td>0.57</td><td></td><td>0.51</td></tr><tr><td>Covariance, + content price</td><td></td><td>0.87</td><td></td><td>0.82</td></tr><tr><td>Price, + content price</td><td></td><td>0.70</td><td></td><td>0.73</td></tr><tr><td>Covariance, replication</td><td></td><td>0.88</td><td></td><td>0.66</td></tr><tr><td>Price, replication</td><td></td><td>0.67</td><td></td><td>0.59</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Pre-specified forecasts of supply for 6 arms × 6 parents. Sign: of the 24 changes (6 arms × 4 goods), those whose forecast has the same sign at every parent, and how many of them have the observed sign. Share: explained share of the standardized response of F1a, F1b and length, with one coefficient per feature unless noted. In “+ content price”, the measured change in the price of content enters as a covariate. “Replication” is a second batch on three parents in which each RM has its own reward scale (§6).

The gap lies in length. On the two formatting goods, the two predictors explain similar shares: 0.71 and 0.87 for headers, 0.91 and 0.66 for list markers. On length, we explain 7% with the price predictor and 84% with the covariance predictor. We attribute this gap to the first failure in §3. In an arm with a higher price for headers, the longer responses that carry headers also receive higher rewards, and only $\Delta r$ contains this change. The fitted coefficients are of the same order across the primary features (47, 41 and 49 for F1a, F1b and length, far below the equilibrium value $1 / \beta = 2 0 0 ;$ §7), so we can describe with one number how far a policy moves along the covariance in 200 steps.

The forecast transfers to real upgrades. Implanted prices are a controlled instrument, so we also test the forecast on the edit that practitioners make most often: replacing the reward model by a better one. From three branch points, RL either continues with the original proxy or switches to one of three reward models from Wang et al. (2026): Qwen models trained on 20,000 and 80,000 pairs, and a Llama model trained on 20,000 pairs. We computed the forecasts before the runs with the coefficients κ frozen from the factorial runs, so the test is out of sample (Appendix P). With the covariance predictor, we predict every direction whose sign is the same at all three parents (7 of 7) and explain 66% of the standardized response at step 200. With one coefficient per feature fitted to these runs, we explain 77% with the covariance predictor and 30% with the price predictor. Under all three upgrades, formatting and length rise, as forecast.

The forecast holds under GRPO. We repeated the factorial runs on three parents with GRPO (Shao et al., 2024), in which each advantage is divided by the standard deviation of its group. Here the first-order response is proportional to the covariance with the reward divided by its within-prompt standard deviation, and we fixed this normalized predictor in advance. With it, we predict all 22 stable changes (of 24) at step 200 and explain 75% of the standardized response, against 55% for the price predictor (Appendix N). The normalized and the raw covariance explain the same share at step 200, and the normalized one explains slightly more early (0.90 against 0.86).

Cross effects between headers and list markers are small and asymmetric, and they fade by step 200 (Appendix H).

## 6 Forecasting Rent

Before the runs, we computed $\mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( \rho , \Delta r ) ]$ and $\mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( z _ { g } , \Delta r ) ]$ for every arm and parent. We fixed the signs in advance, counting a sign only when its interval by bootstrap over prompts excludes zero, and the ranking of arms. Over steps 20–30, the forecast agrees with the ranking of the six arms by their change in rent (Spearman 0.94) and by their change in gold reward (0.89). Where the interval of the forecast excludes zero, 20 of 23 arm–parent signs are correct for rent and 19 of 25 for gold (Figure 2).

The forecast has a horizon. By steps 160–200, the ranking of arms by rent reverses (Spearman

−0.77), while the forecast of supply in §5 still holds. The ranking of arms by observed rent also reverses between step 25 and step 200 (−0.77), so we would order these edits wrongly if we stopped after 30 steps. Almost all of the late change is in gold reward: on the fixed ruler, proxy reward moves by at most 0.02 standard deviations in any arm. Under RLOO, the within-prompt spread of rewards acts as a learning rate, and our treated models differ in spread by up to 1.56 times. In a pre-specified replication in which every arm starts with the spread of the placebo, the early forecast holds again (Spearman 1.00 for rent) and the late ranking still reverses (−0.60), so the reversal does not come from reward scale (Appendix J).

Real upgrades do not reverse. For the three real upgrades, we fixed the forecast of rent and gold reward before the runs. All nine forecast signs are correct for rent and for gold reward over steps 20– 30: rent falls and gold reward rises under every upgrade. The forecast also orders the upgrades correctly: rent falls most under the Qwen model with 80,000 pairs and least under the Llama model. At step 200, the order is the same for rent and for gold reward, so the late reversal of the factorial arms does not appear in these upgrades. None of the upgrades lowers the price of headers, which fits the mechanism below. Under GRPO, the frozen forecast again ranks the arms early (Spearman 0.89 for rent) and the ranking again reverses by step 200 (−0.83, negative at every parent), so the reversal is not specific to RLOO.

The reversal follows distance travelled. In exploratory analyses (Appendix K), the late rise in rent lies outside the five features and is ordered like the KL divergence that the policy accumulates beyond the placebo (Spearman 0.89, 0.77 and 0.77 in the first batch, the replication and the GRPO batch). The policy travels furthest after a price cut on headers (arms 0000 and 0101): early on, the response matches the forecast, and by step 200, it reaches two to three times the forecast. In a prospective test with a symmetric price cut and price rise, the policy again moved further after the cut (9 of 9 cells), but gold reward did not fall with this distance (Figure 3, Appendix M). When we recompute the covariance on the policies at step 400, the incentive after the cut has grown to 2.08 times its value at the branch point at $\beta = 0 . 0 2$ and changed sign at $\beta = 0 . 0 0 5$ whereas the incentive after the rise has fallen to 0.68 of its value, as on the placebo policy.

![](images/d3c6002f2ab2a89ceb72739e8d9c926173427a5806c7949285d92ad464328065.jpg)

![](images/184c57062ac1364f5c294dfd66c4b27c3fbc1f2316123904941feb52aac406f4.jpg)  
Figure 2: Left: frozen forecast of the change in rent against the observed change over steps 20–30, one point per factorial arm. Right: retrospective forecast for 13 patches of Wang et al. (2026) at step 100.

![](images/40542168dee31b21b63210d1003b4bd053dcd7020f77602ae23e6c7a9174b492.jpg)

![](images/973b8fb65340b0f980704ae1b4b58ff1c58e37e75c315b3b4041c925742887cf.jpg)

![](images/d73ba2df3abedf9269100702aa01360b91313dc07630b2c919cba48d5f15e642.jpg)  
Figure 3: Prospective test with symmetric shocks: change in formatting after a price rise and a price cut of equal size, divided by the forecast $C / ( \beta + \beta _ { 0 } )$ , mean over three parents. At $\beta = 0 . 0 0 1 2 5$ , the policy under the cut reaches the length limit.

The forecast ranks earlier patches. Wang et al. (2026) trained the same policy family from SFT under 13 distinct patches. For each patch, we compute $\Delta r$ from the reward rule of the patch on 256 SFT samples per prompt and evaluate $\mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( \rho , \Delta r ) ]$ We knew the outcomes of these patches when we designed this computation, so it is a retrospective forecast without free parameters. The forecast agrees with the ranking of the 13 patches by their observed change in rent, with Spearman correlations of 0.91, 0.97 and 0.99 at steps 50, 100 and 150. The agreement holds within types of patch (audits: 0.79 to 1.00, length penalties: 0.80 to 1.00) and after we remove the mean of each type (0.69 to 0.79), so it is more than the ordering of audits by strength. For gold reward, the correlations are 0.28 to 0.51.

<table><tr><td>β</td><td>A</td><td>τ (steps)</td><td>Initial slope</td></tr><tr><td>0.0025</td><td>7.4</td><td>42</td><td>0.150</td></tr><tr><td>0.005</td><td>6.0</td><td>28</td><td>0.122</td></tr><tr><td>0.01</td><td>3.9</td><td>17</td><td>0.137</td></tr><tr><td>Slope on log β</td><td>-0.46</td><td>-0.67</td><td>-0.07</td></tr><tr><td>Equilibrium</td><td>-1</td><td>-1</td><td>0</td></tr></table>

Table 3: Relaxation of formatting supply after a price shock, mean over three parents. Last two rows: slope of each quantity on log $\beta ,$ and its value at equilibrium.

## 7 Learning Dynamics or Equilibrium

From three branch points, we run the placebo and a reward model that raises the prices of F1a and F1b by 4 each for 400 steps at $\beta \in$ $\{ 0 . 0 0 2 5 , 0 . 0 0 5 , 0 . 0 1 \}$ , for a total of 18 runs. For the formatting supply $\mathrm { F 1 } = \mathrm { F 1 a } + \mathrm { F 1 b }$ , we fit $\Delta s ( t ) = A ( 1 - e ^ { - t / \tau } )$ to the mean difference from the placebo over parents at each $\beta ,$ and we read the initial slope from the first 30 steps. No run exceeds 10% at the length limit.

The response saturates below equilibrium. The initial slope does not depend on $\beta ,$ with a log-log slope of −0.07 (Table 3). The asymptote and the time constant fall with $\beta ,$ but at about half the rate of equilibrium. Their slopes are −0.46 (parents: $- 0 . 5 9 \mathrm { \ t o \ } - 0 . 2 4 )$ and $- 0 . 6 7 \ ( - 0 . 9 5 \ \mathrm { t o } \ - 0 . 3 0 )$ against −1 at equilibrium. The response saturates, which pure learning dynamics would not predict, and it saturates below the level that $1 / \beta$ implies.

An effective KL coefficient. All three facts are consistent with a single parameter. We conjecture that the optimization contains an implicit regularizer $\beta _ { 0 }$ in addition to the KL penalty. Eq. 6 then holds with $\beta + \beta _ { 0 }$ in place of $\beta .$ As a result, A and τ scale as $1 / ( \beta + \beta _ { 0 } )$ , their log-log slopes become $- \beta / ( \beta + \beta _ { 0 } )$ , and the initial slope stays independent of β. We fitted this model jointly to the three curves and obtained $\beta _ { 0 } = 0 . 0 0 6 5$ (parents: 0.0034 to 0.0186). With three parameters, the fit is as good as the fit with six free parameters (residual sum of squares 172 against 165, with a lower BIC) and much better than the fit with $\beta _ { 0 } = 0 ( 6 4 8 )$ . In the test with symmetric shocks, we find the same term after the price rise $( \beta _ { 0 } = 0 . 0 0 2 9 )$ but not after the price cut $( \beta _ { 0 } = 0 ;$ Appendix M).

## 8 Implications for Practice

Compare two RMs before training. One pass of scoring is enough. A team that plans to replace a reward model can sample about 16 responses per prompt from the current policy, score them with both models, and compute the within-prompt covariance of each behavior of interest, such as length, formatting or refusals, with the score difference (Appendix Q). Under GRPO, first divide each score by its within-prompt standard deviation.

![](images/e0eed6481dc58061a99d3bcff92702323eb12477111171f2b6c054e688041f28.jpg)  
Figure 4: Change in formatting supply relative to the placebo after a price shock at three KL coefficients, mean over three parents. Dashed curves: joint fit with an effective coefficient $\beta + \beta _ { 0 }$

Watch edits that cut a price. After edits that cut the price of a heavily supplied good, the policy moved further from its start, and a 30-step run ranked such edits wrongly under both RLOO and GRPO, while real upgrades kept their ranking. We recommend that teams track the KL divergence from the starting policy along with the target behavior and recompute the covariance at a checkpoint. Under RLOO, they should also track the withinprompt spread of rewards, which sets the effective step size.

## 9 Conclusion

We can evaluate an edit to a reward model before RL runs. With the within-prompt covariance between a feature and the change in reward, computed from policy samples and two reward models, we forecast the direction of every stable change in a pre-specified factorial experiment, most of its size, and the response to real upgrades. Counterfactual prices measure what a reward model values rather than the incentive that a policy faces, and with them we explain about half. With the same covariance, we rank the first effect of an edit on rent. How long this forecast holds depends on how far the policy travels, and thus on how the incentive changes along the path. We can measure this change at a checkpoint but cannot yet forecast it before RL.

## Limitations

We study one testbed: a 1.7B policy and one family of reward models as proxies, trained mostly with RLOO and in one batch with GRPO. The forecasts are first order in the edit, and we test them over 200 steps after a branch point. For rent, the frozen forecast holds in the first phase of training, and our analyses of the later reversal are exploratory. In a prospective test, the policy moved further after a price cut, but gold reward did not fall with this distance, and the effective coefficient $\beta + \beta _ { 0 }$ describes price rises but not price cuts. In two arms with a large shock to summaries, length ran away, and one of 16 implants fell just below the validation threshold on parent responses. Our goods are detected by rules, so the instrument applies to features that we can delete cleanly. The replication, the GRPO batch and the test with symmetric shocks use three parents each.

## Ethical Considerations

Implanting prices is a controlled form of poisoning the training data of a reward model. We use only public preference data and open models, and we label every released reward model with its implant. The method is meant to help practitioners evaluate edits to reward models before they spend compute on RL.

## References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, Bangkok, Thailand. Association for Computational Linguistics.

Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. 2025. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. Preprint, arXiv:2503.11926.

Ahmad Beirami, Alekh Agarwal, Jonathan Berant, Alexander D’Amour, Jacob Eisenstein, Chirag Nagpal, and Ananda Theertha Suresh. 2025. Theoretical guarantees on the best-of-n alignment policy. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 3580–3602. PMLR.

George E. P. Box and J. Stuart Hunter. 1961. The $2 ^ { k - p }$ fractional factorial designs, Part I. Technometrics, 3(3):311–351.

Yuyan Bu, Liangyu Huo, Yi Jing, and Qing Yang. 2025. Beyond excess and deficiency: Adaptive length bias mitigation in reward models for RLHF. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 3091–3098, Albuquerque, New Mexico. Association for Computational Linguistics.

Stephen Casper, Xander Davies, et al. 2023. Open problems and fundamental limitations of reinforcement learning from human feedback. Transactions on Machine Learning Research.

Lichang Chen, Chen Zhu, Jiuhai Chen, Davit Soselia, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. 2024. ODIN: Disentangled reward mitigates hacking in RLHF. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 7935–7952. PMLR.

Brian Christian, Jessica A. F. Thompson, Elle Michelle Yang, Vincent Adam, Hannah Rose Kirk, Christopher Summerfield, and Tsvetomira Dumbalska. 2026. Reward models inherit value biases from pretraining. In The Fourteenth International Conference on Learning Representations (ICLR), pages 36013–36041.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. 2024. Reward model ensembles help mitigate overoptimization. In The Twelfth International Conference on Learning Representations (ICLR).

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2024. UltraFeedback: Boosting language models with scaled AI feedback. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 9722–9744. PMLR.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. Preprint, arXiv:2404.04475.

Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, Peter Shaw, and Jonathan Berant. 2024. Helping or herding? Reward model ensembles mitigate but do not eliminate reward hacking. In First Conference on Language Modeling (COLM).

Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios N. Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. 2025. How to evaluate reward models for RLHF. In The Thirteenth International Conference on Learning Representations.

Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10835–10866. PMLR.

Dylan Hadfield-Menell and Gillian K. Hadfield. 2019. Incomplete contracting and AI alignment. In Proceedings ofthe 2019 AAAI/ACM Conference on AI, Ethics, and Society (AIES), pages 417–422. ACM.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. Preprint, arXiv:1503.02531.

Harold Hotelling. 1932. Edgeworth’s taxation paradox and the nature of demand and supply functions. Journal ofPolitical Economy, 40(5):577–616.

Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. 2022. Datamodels: Understanding predictions with data and data with predictions. In Proceedings of the 39th International Conference on Machine Learning.

Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894.

Tomasz Korbak, Ethan Perez, and Christopher L. Buckley. 2022. RL with KL penalties is better viewed as Bayesian inference. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 1083–1091.

Anne O. Krueger. 1974. The political economy of the rent-seeking society. The American Economic Review, 64(3):291–303.

Ryogo Kubo. 1966. The fluctuation-dissipation theorem. Reports on Progress in Physics, 29(1):255–284.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, and 4 others. 2025a. Tülu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling (COLM).

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi,

Noah A. Smith, and Hannaneh Hajishirzi. 2025b. RewardBench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1755–1797, Albuquerque, New Mexico. Association for Computational Linguistics.

Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, and Yahui Zhou. 2026. Skywork-Reward-V2: Scaling preference data curation via human-AI synergy. In The Fourteenth International Conference on Learning Representations (ICLR).

Tianqi Liu, Wei Xiong, Jie Ren, Lichang Chen, et al. 2025a. RRM: Robust reward model training mitigates reward hacking. In International Conference on Learning Representations.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. 2025b. RM-Bench: Benchmarking reward models of language models with subtlety and style. In The Thirteenth International Conference on Learning Representations.

David Manheim and Scott Garrabrant. 2018. Categorizing variants of Goodhart’s law. Preprint, arXiv:1803.04585.

Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, and David T. Mellor. 2018. The preregistration revolution. Proceedings of the National Academy ofSciences, 115(11):2600–2606.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Alexander Pan, Kush Bhatia, and Jacob Steinhardt. 2022. The effects of reward misspecification: Mapping and mitigating misaligned models. In International Conference on Learning Representations.

George R. Price. 1970. Selection and covariance. Nature, 227:520–521.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems.

Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. 2024. WARM: On the benefits of weight averaged reward models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 42048–42073.

Javier Rando and Florian Tramèr. 2024. Universal jailbreak backdoors from poisoned human feedback. In International Conference on Learning Representations.

Yi Ren and Danica J. Sutherland. 2025. Learning dynamics of LLM finetuning. In International Conference on Learning Representations.

Sherwin Rosen. 1974. Hedonic prices and implicit markets: Product differentiation in pure competition. Journal ofPolitical Economy, 82(1):34–55.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems (EuroSys ’25), pages 1279–1297, Rotterdam, Netherlands. ACM.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2024. A long way to go: Investigating length correlations in RLHF. In First Conference on Language Modeling (COLM).

Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. 2022. Defining and characterizing reward hacking. In Advances in Neural Information Processing Systems, volume 35, pages 9460–9471. Curran Associates, Inc.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. 2020. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pages 3008–3021.

Gordon Tullock. 1967. The welfare costs of tariffs, monopolies, and theft. Western Economic Journal, 5(3):224–232.

Chaoqi Wang, Zhuokai Zhao, Yibo Jiang, Zhaorun Chen, Chen Zhu, Yuxin Chen, Jiayi Liu, Lizhu Zhang, Xiangjun Fan, Hao Ma, and Sinong Wang. 2025. Beyond reward hacking: Causal rewards for large language model alignment. Preprint, arXiv:2501.09620.

Jiacheng Wang and Jinbin Huang. 2026. Reward hacking as equilibrium under finite evaluation. Preprint, arXiv:2603.28063.

Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, and Chaowei Xiao. 2024. RLHFPoison: Reward poisoning attack for reinforcement learning with human feedback in large language models. In Proceedings of the 62nd Annual Meeting of

the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2551–2570.

Zixiao Wang, Xingbo Du, Duzhen Zhang, and Le Song. 2026. Rent-seeking language models: Reward models set the prices, frequent audits curb the rent. Preprint, arXiv:2610.19832.

Lorenz Wolf, Robert Kirk, and Mirco Musolesi. 2025. Reward model overoptimisation in iterated RLHF. Preprint, arXiv:2505.18126.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Xiaoying Zhang, Jean-François Ton, Wei Shen, Hongning Wang, and Yang Liu. 2024. Mitigating reward overoptimization via lightweight uncertainty estimation. In Advances in Neural Information Processing Systems, volume 37, pages 81717–81747. Curran Associates, Inc. ArXiv preprint titled “Overcoming Reward Overoptimization via Adversarial Policy Optimization with Lightweight Uncertainty Estimation”.

Xuanchang Zhang, Wei Xiong, Lichang Chen, Tianyi Zhou, Heng Huang, and Tong Zhang. 2025. From lists to emojis: How format bias affects model alignment. In Proceedings ofthe 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 26940–26961, Vienna, Austria. Association for Computational Linguistics.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. Preprint, arXiv:1909.08593.

## A Analysis Plan Freeze and Decision Log

We wrote the analysis plan before the factorial runs and appended every later decision with a date. In Table 5, we separate analyses that we fixed before we saw the relevant outcome from analyses that we added afterwards. Table 4 gives the numerical criterion of each hypothesis of the follow-up batches. A tolerance on a ratio is a factor of 1.5, and a threshold of 0.5 on a share is the level of the price predictor in the main batch; U2 is out of sample, so its threshold is lower than that of G2. A Spearman correlation of 0.5 is a conventional boundary for a moderate correlation, and with six arms, we use it as a descriptive criterion. The interval for β<sub>0</sub> spans about a factor of 3 on either side of the estimate from the first relaxation runs (0.0065), so this criterion is weak. All numbers in the main text come from one results ledger, and for every entry of the ledger, we record the script and the output file.

Hypothesis Criterion, fixed before the runs   
R1 Spearman $\geq 0 . 5$ for rent and for gold, steps 160–200   
R2 |Spearman| of reward spread and change in length at step $2 0 0 \leq 0 . 5$   
R3 Mean change in length of arms 0000, 0101 and 1100 at step 200 at most half of that in the first batch   
R4 Price predictor: $\geq 1 8 / 2 0$ signs and share $\geq 0 . 5$ at the end   
U1 $\geq 8 0 \%$ of stable directions correct at the end   
U2 Share with frozen $\kappa \geq 0 . 5 ;$ with refitted $\kappa ,$ covariance $\geq$ price   
U3 $\geq 8 0 \%$ of early signs correct where the interval excludes zero; rent falls most under Qwen 80k and least under   
Llama   
U4a Ranking of rent at the end as early and as forecast   
U4b Observed over forecast change in headers $\leq 1 . 5$ for every upgrade   
G1 $\geq 9 0 \%$ of stable directions correct at the end   
G2 $\mathrm { S h a r e } \geq 0 . 6 ;$ at every parent, not below the raw covariance   
G3 |Spearman| of reward spread and change in length at step $2 0 0 \leq 0 . 5$   
G4 Early Spearman $\geq 0 . 5 ;$ late Spearman $\ge - 0 . 5$ and $\leq 0$ at every parent; Spearman of extra KL and late rent   
$> 0$   
S1 $R _ { \mathrm { e n d } }$ of the cut $\geq 1 . 5$ times that of the rise at 2 or 3 of the 3 values of $\beta$   
S2 Gold change after the cut below that after the rise at every β; Spearman of extra KL and gold change over 18   
$\mathrm { r u n s } \leq - 0 . 5$   
S3 $\tau$ of the cut $\geq 2$ times that of the rise at 2 or 3 of the 3 values of $\beta$   
S4 $\mathrm { E x t r a K L }$ after the cut above that after the rise at every $\beta$   
S5 |Slope of the initial slope on log $\beta | \le 0 . 2 5 ; \beta + \beta _ { 0 }$ fits better than $\beta _ { 0 } = 0 ; \beta _ { 0 } \in [ 0 . 0 0 2 , 0 . 0 2 ]$   
S6 At every $\beta ,$ ratio of initial slopes (cut to rise) within [0.67, 1.5] times the forecast ratio  
Table 4: Numerical criteria of the pre-specified hypotheses. Values are means over parents unless noted. Stable directions: forecasts with the same sign at every parent.

Decisions after the plan freeze. We changed the design four times after we wrote the plan. First, under an earlier placebo recipe with 1,200 content pairs, responses grew longer during the warm-up of the parents (30% to 42% of responses at the length limit after 50 steps). We therefore returned to the recipe with 600 content pairs, under which the placebo matched the proxy (489 against 487 tokens after 50 steps). Second, we had first sampled responses at the branch points without the chat template, and we discarded all measurements from these samples. Third, we chose a dose of $+ 2$ for summaries because smaller doses did not pass validation on parent responses. The two arms that left the local regime have this dose. Fourth, in a first version of the test with symmetric shocks, all reward models shared one standardization, so the price cut came with 1.35 times the reward spread of the price rise. We cancelled these runs before we looked at any result and fixed the version with equal spread (Appendix M).

## B Glossary

We borrow several terms from economics. Table 6 gives the meaning of each term in RLHF.

## C Theory Details

Supply as the gradient of profit. Let $r = a ( x ) .$ + $p ^ { \top } f + e$ and $\pi _ { p } ^ { * } ( y \mid x ) \propto \pi _ { 0 } ( y \mid x ) \exp ( r / \beta )$ . For one prompt, $\Phi _ { x } ( p ) = \beta \log \mathbb { E } _ { \pi _ { 0 } } [ \exp ( r / \beta ) ]$ . By differentiation, we obtain $\partial \Phi _ { x } / \partial p _ { k } = \mathbb { E } _ { \pi _ { v } ^ { * } } [ f _ { k } ]$ and by a second differentiation, we obtain $\partial ^ { 2 } \Phi _ { x } / \partial p _ { k } \partial p _ { j } = \mathrm { C o v } _ { \pi _ { p } ^ { * } } ( f _ { k } , f _ { j } ) / \beta$ . After averaging over prompts, we obtain Eq. 2. The matrix is symmetric and positive semidefinite, so in equilibrium the own response to a price rise is never negative.

The selection term. For $\pi _ { t + 1 } ( y )$ ∝ $\pi _ { t } ( y ) \exp ( \eta A ( y ) )$ , to first order in $\eta ,$ we have $\mathbb { E } _ { t + 1 } [ f ] - \mathbb { E } _ { t } [ f ] = \eta \operatorname { C o v } _ { t } ( f , A ) + O ( \eta ^ { 2 } )$ With RLOO, the advantage is the reward minus the mean of the other samples of the same prompt, so $\operatorname { C o v } _ { t } ( f , A )$ is a within-prompt covariance with the reward. After an edit, $f$ changes in the first step by $\eta \mathrm { C o v } ( f , r _ { \mathrm { n e w } } )$ instead of $\eta \mathrm { C o v } ( f , r _ { 0 } | { \bf { d } } )$ , and the difference is η $\operatorname { C o v } ( f , \Delta r )$

Relaxation and an effective coefficient. Along a direction $\pi _ { \theta } \propto \pi _ { 0 } \exp ( \theta f )$ , the KL-regularized objective is $J ( \theta ) = p \mathbb { E } _ { \theta } [ f ] - \beta \operatorname { K L } ( \pi _ { \theta } \| \pi _ { 0 } )$ . Its natural gradient is $p - \beta \theta$ , so $\theta ( t ) = ( p / \beta ) ( 1 - e ^ { - \eta \beta t } )$ , and to first order $\Delta s ( t ) \ = \ \mathrm { V a r } ( f ) \theta ( t )$ , which is Eq. 6. If the optimization contains an additional pull toward $\pi _ { 0 }$ of strength $\beta _ { 0 }$ , for example from finite step sizes or from the clipping of updates, the flow becomes $\dot { \theta } = \eta ( p - ( \beta + \beta _ { 0 } ) \theta )$ Then $A \propto 1 / ( \beta + \beta _ { 0 } ) , \tau \propto 1 / ( \beta + \beta _ { 0 } )$ , and $A / \tau = \eta \operatorname { V a r } ( f ) p$ does not depend on $\beta .$ . The slope of log A on log $\beta$ is $- \beta / ( \beta + \beta _ { 0 } )$

<table><tr><td>Analysis</td><td>Status</td></tr><tr><td>Covariance and price predictors of supply; Pre-specified signs; cross effects</td><td></td></tr><tr><td>Feature sets F1a, F1b, L and all five fea- Pre-specified tures</td><td></td></tr><tr><td>Common and per-feature κ; standardized Pre-specified units</td><td></td></tr><tr><td>Content price as covariate if any arm ex- Pre-specified ceeds 0.3</td><td></td></tr><tr><td>Forecasts of rent and gold reward (signs, Pre-specified ranking)</td><td></td></tr><tr><td>Arms with more than 10% at the length Pre-specified limit are descriptive</td><td></td></tr><tr><td>Three fingerprints of relaxation</td><td>Pre-specified</td></tr><tr><td>Replication with separate reward scales Pre-specified (R1–R4)</td><td></td></tr><tr><td>Real RM upgrades (U1–U4)</td><td>Pre-specified</td></tr><tr><td>Relaxation with symmetric shocks (S1– Pre-specified S6)</td><td></td></tr><tr><td>GRPO batch (G1–G4)</td><td>Pre-specified</td></tr><tr><td>Marginal-price predictor; displacement ra- Exploratory tio</td><td></td></tr><tr><td>Reward spread and late drift</td><td>Exploratory</td></tr><tr><td>Effective coefficient  $\beta _ { 0 }$ </td><td>Exploratory</td></tr><tr><td>Decomposition of rent; KL travelled</td><td>Exploratory</td></tr><tr><td>Covariance along the path</td><td>Exploratory</td></tr><tr><td>Patches from prior work</td><td>Retrospective</td></tr></table>

Table 5: Pre-specified and later analyses.

Group-normalized advantages. Under GRPO, the advantage of a response in a group of n samples is $A \ = \ ( r \ - \ { \bar { r } } _ { x } ) / s _ { x } ,$ with $\bar { r } _ { x }$ and $s _ { x }$ the mean and standard deviation of the group. To first order, one step changes the mean of a feature by $\eta \mathrm { C o v } _ { x } ( f , r / s _ { x } )$ . Because $s _ { x }$ divides each reward, a multiplicative change in the scale of the reward model has no effect, and the forecast of an edit becomes $\mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( f , r _ { \mathrm { n e w } } / s _ { x } ( r _ { \mathrm { n e w } } ) ) -$ $\mathrm { C o v } _ { x } ( f , r _ { \mathrm { o l d } } / s _ { x } ( r _ { \mathrm { o l d } } ) ) \big ]$ ]. We estimate $s _ { x }$ from the 16 samples per prompt at the branch point, while training uses groups of 8.

Forecasts of rent. Rent and gold reward are functions of a response, like features. With $\rho$ in place of $f _ { k }$ in the selection term, we obtain $\Delta \mathbb { E } [ \rho ] \approx \eta \mathrm { C o v } ( \rho , \Delta r )$ . Because we do not decompose rent into features, the forecast also covers rent that no measured feature carries.

## D Implementation Details

Reward models. Every reward model is Qwen3- 1.7B-Base with a scalar head, trained with the hyperparameters, seed and data order of the proxy (Wang et al., 2026). The training set has 5,997 pairs: the 3,000 pairs of the proxy, 600 pure deletion pairs for each of F1a, F1b and F2, 597 for F4, and 600 content pairs. A pair is pure when the rewrite changes no other good. The deletion pairs come from responses of three SFT policies (Qwen3, Llama 3.2 and OLMo 2) to the prompts of the RM training data, so no reward model sees the evaluation prompts.

<table><tr><td>Term</td><td>Meaning in this paper</td></tr><tr><td>Good, feature</td><td>A property of a response that we can count and delete, such as headers or list markers</td></tr><tr><td>Supply</td><td>Mean count of a feature in the responses of the policy</td></tr><tr><td>Price</td><td>Mean drop in standardized reward when we delete a feature (Eq. 1)</td></tr><tr><td>Rent</td><td>Proxy reward minus gold reward of a re- sponse, both standardized</td></tr><tr><td>Hedonic price</td><td>Coefficient of a feature in a within-prompt regression of reward or rent on all features</td></tr><tr><td>Edit</td><td>Any change to the reward model, from a label shift to a full replacement</td></tr><tr><td>Placebo</td><td>A reward model retrained by self- distillation, with the prices of the proxy</td></tr><tr><td>Arm</td><td>One treated reward model, run from one branch point</td></tr><tr><td>Branch point</td><td>The policy from which all arms of a parent start</td></tr><tr><td>Covariance pre- dictor</td><td> $\kappa \mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( f , \Delta r ) ] \left( \mathrm { E q } . 3 \right)$ </td></tr><tr><td>Price predictor</td><td> $\kappa \Sigma \Delta p ( \mathrm { E q . } 4 )$ </td></tr><tr><td>K</td><td>Response coefficient:  $1 / \beta$  at equilibrium, learning rate times steps early on</td></tr><tr><td>Effective coeffi- cient</td><td> $\beta + \beta _ { 0 } ,$  the KL coefficient that the re- sponse behaves as if it had</td></tr></table>

Table 6: Terms from economics and their meaning in RLHF.

Branch points. We train each parent for 50 steps against the placebo from a fresh optimizer state and merge the result to full weights. At each branch point, we sample 16 responses for each of 200 evaluation prompts with the chat template, temperature 1.0 and a limit of 1,024 tokens. Between 2.8% and 3.8% of these responses reach the limit, and the mean length is 496 to 520 tokens.

We list the settings in Table 7. We also score every rollout and evaluation sample with the original proxy and with the two gold models. These scores enter no reward, and we use them to measure rent on a fixed ruler.

Compute. Each RL run uses four A100-80GB GPUs. According to the cluster’s accounting, the runs reported in this paper used about 2,950 GPUhours: 770 for the factorial runs, 460 for the relaxation runs of stage 4, 780 for the relaxation runs with symmetric shocks, 330 for the real RM upgrades, 290 for the replication with separate reward scales, 290 for GRPO, and 40 for the calibration runs and the warm-up of the parents. Scoring, sampling and the training of reward models used under 100 GPU-hours. The whole project, including preliminary runs, runs that failed and were restarted, and runs that we discarded, used about 3,450 GPUhours.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Algorithm</td><td>RLOO (one batch: GRPO), KL penalty in the reward</td></tr><tr><td>KL coefficient β</td><td>0.005; stage 4: 0.0025, 0.005, 0.01; symmetric shocks: 0.00125, 0.005, 0.02</td></tr><tr><td>Prompts × samples Learning rate</td><td>64 × 8 per step  $2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Response length</td><td>at most 1,024 tokens</td></tr><tr><td>Steps</td><td>200 (relaxation runs: 400) 500 held-out prompts every 5</td></tr><tr><td>Evaluation</td><td>steps; symmetric shocks: 200</td></tr><tr><td>Optimizer state</td><td>prompts every 2 steps new at the branch point</td></tr><tr><td>RL seed and prompt order Hardware</td><td>shared by all arms of a parent four A100-80GB GPUs per run</td></tr></table>

Table 7: RL settings.

## E Implant Validation

Content price. After an implant on formatting, the price of content can also change. We measure this price by the drop in score when we delete one random sentence that contains no good. Under the pre-specified plan, the content price enters as a covariate if its change exceeds 0.3 in any arm. Relative to the placebo, the change per pair is −0.06 (0000), +0.27 (0011), −0.01 (0101), +0.36 (0110), +0.06 (1001), +0.15 (1010), +0.45 (1100) and +0.30 (1111). Two of the six main arms reach the threshold, and Table 2 includes the covariate.

Off-target noise. We compute the noise on the same responses and on the same scale as the test. On SFT samples, the root mean square of off-target movements is 0.065, which gives the threshold of 0.195. On parent responses, it is 0.071, which gives 0.213. The one implant below this threshold, F1b in arm 1001, moves by −0.20 with the designed sign.

Non-additivity. We list the drops behind Eq. 5 in Table 8. Gold reward behaves in the opposite way to the proxy: gold reward falls more when we delete only headers than when we delete all formatting. When we delete headers alone, rent therefore rises by 0.14 standard deviations per response.

<table><tr><td>Deletion</td><td>Proxy</td><td>Gold</td><td>Rent</td></tr><tr><td>Headers and bold</td><td>0.174</td><td>0.313</td><td>-0.139</td></tr><tr><td>List markers</td><td>0.028</td><td>0.044</td><td>-0.016</td></tr><tr><td>Both</td><td>0.485</td><td>0.216</td><td>+0.269</td></tr><tr><td>Interaction  $h _ { a b } f _ { a } f _ { b }$ </td><td>-0.283</td><td>+0.142</td><td>-0.424</td></tr></table>

Table 8: Mean drop in standardized score when goods are deleted, 18,537 parent responses that carry both goods. The interaction row is $\Delta z _ { a b } - \left( \Delta z _ { a } + \Delta z _ { b } \right)$ of Eq. 5, that is, the sum of the single drops minus the joint drop.

## F Goods and Rewrites

We detect each good with rules on the markdown of a response. Headers and bold (F1a) count markdown headers and bold spans. List markers (F1b) count bullet and number markers at the start of a line. A closing summary (F2) is a final paragraph that opens with a summary phrase, and a resource list (F4) is a final list of links or references. Each deletion rewrite removes only the markup or the span of its good and keeps every other character. When we delete all formatting at once, we also join the lines of a list into one paragraph. The following response of a parent policy carries four elements of F1a and two of F1b (91 tokens).

The relationship between the two sentences   
is \*\*cause\*\*.   
\*\*Explanation:\*\*   
- The first sentence describes a train   
sliding down a steep incline.   
- The second sentence describes someone   
sitting and screaming, ...   
Thus, the second sentence is the \*\*cause\*\*   
of the first.

## \*\*Output:\*\* cause

After the deletion of F1a, the four bold spans become plain text, and the two list items keep their markers. After the deletion of F1b, the two items lose their markers, and the bold spans stay. After the deletion of both, the response reads as three short paragraphs. On 18,537 responses, the proxy loses 0.174 standard deviations on average after the first deletion, 0.028 after the second and 0.485 after the third (Table 8).

## G Results by Parent

The implanted price changes on parent responses vary little across the six parents. For every arm, the range of the change in F1a across parents is at most

0.04 wide, and the range of the change in F1b at most 0.03. For example, arm 0000 moves F1a by −0.69 to −0.66 and F1b by −0.29 to −0.27, and arm 1111 moves F1a by 0.58 to 0.61 and F1b by 0.23 to 0.24.

Table 9 lists the Spearman correlation between the frozen forecast and the observed change in rent and gold reward for every parent. In the early window, all correlations are positive. In the end window, all are negative.

## H Supply Results in Detail

Arm sets. In arms 0011 and 0110, the higher dose for summaries comes with a lower price for headers. By step 200, 48% and 30% of their responses reach the length limit, against 2% to 3% for the placebo. In arm 1001, 9% to 14% of responses reach the limit between steps 20 and 30, and 3% by the end window. As Table 10 shows, the gap between the predictors widens when we include all eight arms.

We give the explained share of each primary feature, with one coefficient per feature, in Table 11.

Other specifications. If we restrict the comparison to the four goods without length and use a single common coefficient, we explain more with the price predictor than with the covariance predictor (0.69 against 0.38 at the end window). This specification was not pre-specified. The difference comes from the resource list, whose supply sits near zero: its coefficient for the covariance predictor is 3.9, an order of magnitude below the others, so a common coefficient fits it poorly. With one coefficient per feature, we explain 0.81 and 0.75 with the two predictors. In a third predictor, we replace prices per unit by the price of deleting a single element. We added this predictor after the runs started, and with it we explain 0.55 at the end window.

Figure 5 shows the change in supply over the 200 steps. In the arms with a higher price for headers, the response in headers saturates within about 50 steps. In the two arms with a lower price, the supply of headers keeps falling until step 200.

Forecasts by arm. Table 12 lists, for every arm, the observed change at the end window and the two forecasts with their fitted per-feature coefficients, in standardized units and averaged over parents. In the two arms that lower the price of headers, the observed fall in headers exceeds the covariance forecast by a factor of two to three, the overshoot analyzed in Appendix K. For length, the price predictor has the wrong sign in two arms.

Cross effects. Headers and list markers covary positively within prompt, so they are complements under Eq. 2. When we raise the price of headers, the supply of list markers rises early in training, by 0.27 standard deviations (interval 0.25 to 0.29) and by 0.26 in the replication. When we raise the price of list markers, the supply of headers falls by 0.12 (interval −0.16 to −0.08). We cannot predict this fall from the covariance in supply, but we can predict it from the substitution in the valuation of the reward model (Eq. 5). Both cross effects fade by step 200, so we treat them as a second-order term on which the supply and the valuation channel compete.

We list the main effects of the four goods and the aliased interaction in Table 13. We estimate them in standardized units with fixed effects for parents. Own effects dominate, and the cross effects discussed in §5 appear in the rows of F1a and F1b.

## I Calibration Runs

Before the factorial design, we ran a placebo and two treated models for 100 steps from one branch point. In one model, the price of both formatting goods rises by 4. In the other, the price of the summary rises by 2 and the price of the resource list by 3. We measured both predictors on 3,200 responses that we sampled at that branch point with the protocol of Appendix D (Table 14). The second model is a large shock: by step 100, responses under it are longer by 3.8 standard deviations. In an earlier measurement on samples without the chat template, prices per pair were too high by a factor of 2.6 to 3.8, and we do not use it.

## J Replication with Separate Reward Scales

With shared constants, the effective learning rate of the policy is larger under a reward model with a larger within-prompt spread, because RLOO does not normalize advantages. On responses at the branch points, the spread of the six treated models relative to the placebo is 1.39 (0000), 1.56 (0101), 1.02 (1001), 0.97 (1010), 1.27 (1100) and 1.06 (1111). In the replication, we repeat the factorial runs on three parents with the standard deviation of each model multiplied by its ratio. After this change, the spread at the branch point is equal across arms. We divided all forecasts by the same ratios and fixed four hypotheses before the runs. The batch has 21 runs, including a placebo for each parent.

<table><tr><td rowspan="3"></td><td colspan="6">First batch</td><td colspan="3">Replication</td></tr><tr><td>P1</td><td>P2</td><td>P3</td><td>P4</td><td>P5</td><td>P6</td><td>P1</td><td>P2</td><td>P3</td></tr><tr><td>Rent, early</td><td>0.71</td><td>0.94</td><td>0.89</td><td>0.77</td><td>0.94</td><td>0.94</td><td>0.94</td><td>0.83</td><td>0.89</td></tr><tr><td>Gold, early</td><td>0.43</td><td>0.89</td><td>0.54</td><td>0.71</td><td>0.83</td><td>0.89</td><td>0.89</td><td>0.83</td><td>0.83</td></tr><tr><td>Rent, end</td><td>-0.71</td><td>-0.60</td><td>-0.77</td><td>-0.49</td><td>-0.77</td><td>-0.83</td><td>-0.71</td><td>-0.49</td><td>-0.60</td></tr><tr><td>Gold, end</td><td>-0.77</td><td>-0.60</td><td>-0.77</td><td>-0.60</td><td>-0.77</td><td>-0.77</td><td>-0.77</td><td>-0.60</td><td>-0.60</td></tr></table>

Table 9: Spearman correlation across six arms between the frozen forecast and the observed change, by parent.

![](images/2507576cfe8c3560d8c27c1f1e7d39908d8e3ff80ccbd1e132f4a69b9509dee7.jpg)  
Figure 5: Change in supply relative to the placebo of the same parent, mean over six parents, first batch. Solid lines: arms with a lower price for headers. Dashed lines: arms with a higher price for headers. Curves are smoothed over three evaluation points.

<table><tr><td>End window, features F1a, F1b, L</td><td>6 arms</td><td>8 arms</td></tr><tr><td>Covariance, per-feature κ</td><td>0.82</td><td>0.83</td></tr><tr><td>Price, per-feature κ</td><td>0.50</td><td>0.15</td></tr><tr><td>Covariance, common κ</td><td>0.82</td><td>0.79</td></tr><tr><td>Price, common κ</td><td>0.45</td><td>0.08</td></tr></table>

Table 10: Explained share of the standardized response, by arm set.

<table><tr><td></td><td colspan="3">Steps 20–30</td><td colspan="3">Steps 160–200</td></tr><tr><td>Predictor</td><td>Fla</td><td>F1b</td><td>L</td><td>Fla</td><td>F1b</td><td>L</td></tr><tr><td>Covariance</td><td>0.89</td><td>0.57</td><td>0.82</td><td>0.71</td><td>0.91</td><td>0.84</td></tr><tr><td>Price</td><td>0.81</td><td>0.64</td><td>0.43</td><td>0.87</td><td>0.66</td><td>0.07</td></tr></table>

Table 11: Explained share by feature, 6 arms × 6 parents.

For R1, the per-parent correlations (−0.71, −0.49, −0.60) overlap those of the first batch. For R3, the mean drift of the three high-spread arms falls from 134 to 68 tokens. The placebo of the replication is 15 to 22 tokens longer than the placebo of the first batch, and part of the reduction is due to this difference.

<table><tr><td rowspan="3">Arm</td><td colspan="3">Fla</td><td colspan="3">Flb</td><td colspan="3">L</td></tr><tr><td>Obs</td><td>Cov</td><td>Price</td><td>Obs</td><td>Cov</td><td>Price</td><td>Obs</td><td>Cov</td><td>Price</td></tr><tr><td>0000</td><td>-1.45</td><td>-0.54</td><td>-1.13</td><td>-0.19</td><td>-0.63</td><td>-1.26</td><td>1.28</td><td>0.62</td><td>-0.47</td></tr><tr><td>0101</td><td>-1.72</td><td>-0.82</td><td>-1.25</td><td>1.95</td><td>1.92</td><td>0.94</td><td>1.54</td><td>1.17</td><td>-0.19</td></tr><tr><td>1001</td><td>1.02</td><td>1.24</td><td>1.07</td><td>-0.30</td><td>-0.15</td><td>-0.25</td><td>1.18</td><td>1.54</td><td>0.22</td></tr><tr><td>1010</td><td>0.37</td><td>0.71</td><td>0.98</td><td>-0.52</td><td>-0.48</td><td>-0.34</td><td>0.54</td><td>0.72</td><td>0.20</td></tr><tr><td>1100</td><td>1.26</td><td>1.33</td><td>1.09</td><td>2.05</td><td>1.66</td><td>1.42</td><td>2.01</td><td>1.67</td><td>0.46</td></tr><tr><td>1111</td><td>0.51</td><td>0.85</td><td>0.99</td><td>0.57</td><td>0.90</td><td>1.08</td><td>0.81 1.03</td><td></td><td>0.40</td></tr></table>

Table 12: Observed change and forecasts at steps 160– 200, standardized units, mean over six parents, first batch. Forecasts use one coefficient per feature fitted across all arms.

## K Mechanism of the Late Reversal

All analyses in this section are exploratory.

Decomposition. For each parent, we estimate hedonic rent prices at the branch point by a withinprompt regression of rent on the five features. We then value the observed change in supply of each arm at these prices and compare the result with the observed change in rent (Table 16). The decomposition agrees with the observed ranking early but not late. By step 200, the observed rent of the six arms has risen by 0.118 on average, while the value from the decomposition is −0.011.

Distance travelled. At every step, verl logs the mean KL divergence per token between the policy and the branch point. We list the KL that the policy accumulates beyond the placebo in each arm in steps 160 to 200, together with the change in rent, in Table 17. The Spearman correlation between the two is 0.89 in the first batch and 0.77 in the replication, and the correlation with the change in gold reward is −0.89 and −0.77. The correlation between extra KL and length falls from 0.83 to 0.26 in the replication, so length changes together with the effect but is not its cause.

<table><tr><td>Response</td><td>Fla</td><td>F1b</td><td>F2</td><td>F4 F1a×F1b</td></tr><tr><td>Steps 20–30</td><td></td><td></td><td></td><td></td></tr><tr><td>F1a</td><td>+0.66-0.12</td><td></td><td>2-0.18</td><td> $+ 0 . 1 5$  +0.07</td></tr><tr><td>F1b</td><td>+0.27+0.22</td><td></td><td>-0.14</td><td> $+ 0 . 0 4$  +0.06</td></tr><tr><td> $L$ </td><td>+0.51-0.02</td><td></td><td>-0.21</td><td> $+ 0 . 1 8$  +0.10</td></tr><tr><td>Steps 160–200</td><td></td><td></td><td></td><td></td></tr><tr><td>F1a</td><td>+1.37 -0.01</td><td></td><td>-0.35-0.03</td><td>+0.10</td></tr><tr><td>F1b</td><td>-0.00</td><td>0+1.12</td><td>-0.43 -0.32</td><td>-0.26</td></tr><tr><td>L</td><td>+0.09</td><td>+0.27</td><td>-0.46 -0.14</td><td>+0.00</td></tr></table>

Table 13: Factorial effects in standardized units, 6 arms × 6 parents. F1a×F1b is aliased with $\mathrm { F } 2 \times \mathrm { F } 4$
<table><tr><td>Step</td><td>25</td><td>50</td><td>100</td></tr><tr><td>Covariance, per-feature κ</td><td>0.92</td><td>0.97</td><td>0.97</td></tr><tr><td>Price, per-feature κ</td><td>0.06</td><td>0.15</td><td>0.22</td></tr><tr><td>Covariance, common κ</td><td>0.52</td><td>0.64</td><td>0.74</td></tr><tr><td>Price, common κ</td><td>0.05</td><td>0.12</td><td>0.15</td></tr></table>

Table 14: Calibration runs: explained share of the standardized response of all five features, two arms. With both predictors, 9 of 10 signs are correct.

Figure 6 shows the change in rent over time. Rent differs little between arms in the first 30 steps. After step 50, rent rises in most arms, and the rise is largest in arm 0101. We plot the relation between extra KL and late rent in Figure 7.

Overshoot after price cuts. In the end window, the ratio of observed to predicted change in headers is 2.71 for arm 0000 and 2.11 for arm 0101, the two arms that lower the price of headers, against 0.52 to 0.95 for the arms that raise it. In the early window, the ratios are 1.03 and 0.84. The two arms that left the local regime also lower the price of headers. We test this asymmetry with pre-specified hypotheses in Appendix M.

Quantities that did not predict distance. The ranking of the extra KL agrees with the withinprompt variance of $\Delta r$ at the branch point in the first batch (Spearman 0.94) but not in the replication (−0.09), because the variance is confounded with reward spread. Nor does it agree with the norm of the predicted change in supply (0.31 and −0.09). An additive model of rent with a term for features and a term for KL does not transfer from one batch to the other.

<table><tr><td>Hypothesis</td><td>Observed</td><td>Outcome</td></tr><tr><td>R1: Late ranking of rent Spearman -0.60 Not supported and gold follows the fore- (rent), cast</td><td>-0.60 (gold)</td><td></td></tr><tr><td>R2: Late length drift no Spearman longer tracks reward spread (first batch: 0.77)</td><td></td><td>0.03 Supported</td></tr><tr><td>R3: Length drift of high- drift -49% (134 Not met spread arms shrinks</td><td>to 68 tokens)</td><td></td></tr><tr><td>R4: Price predictor still 20/20 signs; share Supported forecasts supply</td><td>0.59</td><td></td></tr></table>

Table 15: Pre-specified hypotheses of the replication and their outcomes.
<table><tr><td>Step</td><td>25</td><td>50</td><td>100</td><td>200</td></tr><tr><td>First batch</td><td>0.83</td><td>0.66</td><td>-0.31</td><td>-0.71</td></tr><tr><td>Replication</td><td>0.71</td><td>1.00</td><td>0.09</td><td>-0.09</td></tr></table>

Table 16: Spearman correlation across arms between rent implied by the observed change in supply at hedonic prices and observed rent.

We list the frozen forecast and the observed change in rent and gold reward for every arm in Table 18.

## L Relaxation Details

We list the fits for every response in Table 19. The fits of length, rent and gold reward are poor, either because ${ \bar { R } } ^ { 2 }$ is low or because the time constant reaches its upper bound within 400 steps, so we report them as descriptive. In Table 20, we compare the model with an effective coefficient with free fits and with $\beta _ { 0 } = 0$ . Saturation of formatting does not explain the smaller slopes, because formatting in the treated arm rises as $\beta$ falls, from 38.5 to 51.9 elements per response.

## M Relaxation with Symmetric Shocks

This batch tests the asymmetry between price cuts and price rises that we found after the factorial runs (§6). From one file of implant pairs, we trained two reward models: one that raises the price of headers and list markers, and one that lowers it by the same amount. We rescaled the standardization of each treated model, so that the within-prompt spread of rewards at the branch points is 1.04 times that of the placebo in both arms. From three parents, we ran the placebo and both treated arms at $\beta \in$ {0.00125, 0.005, 0.02} for 400 steps, a total of 27 runs, and we evaluated on the 200 prompts of the branch points every two steps. The KL penalty uses the branch point as its reference. Before the runs, we froze the forecast $A = C / ( \beta + \beta _ { 0 } )$ , where $C$ is the covariance of formatting with the reward difference at the branch point $\mathrm { ( + 0 . 2 7 3 }$ for the rise and −0.136 for the cut) and $\beta _ { 0 } = 0 . 0 0 6 5$ comes from stage 4. We measure every response as the difference from the placebo with the same parent and $\beta .$

![](images/9959ee92c16a6825194b2052610815408b0892f5360b7e906f61ce3cdb65e18d.jpg)  
Figure 6: Change in rent relative to the placebo of the same parent, in the first batch (six parents), the replication and the GRPO batch (three parents each). Shaded: the early and the end window. Solid lines: arms with a lower price for headers.

![](images/2544fcb6bb86e4beeecc2a2dbde7c054731fe5f1850546ed71f413e3a5182ee6.jpg)

![](images/e02b6474afc25bc2345115bf2557ee5c6442f7da8e8491e149479eee6e3983ac.jpg)

![](images/1d0797f7392decbd893a002f55ab618b5de859303eb618ce5eeab43cccea936e.jpg)  
Figure 7: KL divergence accumulated beyond the placebo in steps 160–200 against the change in rent at step 200, one point per arm. Panel titles give the Spearman correlation.

<table><tr><td colspan="2">First batch</td><td colspan="2">Replication</td></tr><tr><td>Arm</td><td>Extra KL  $\Delta \rho$ </td><td>Extra KL</td><td> $\Delta \rho$ </td></tr><tr><td>0000</td><td>0.069 0.109</td><td>0.047</td><td>0.075</td></tr><tr><td>0101</td><td>0.091 0.243</td><td>0.059</td><td>0.161</td></tr><tr><td>1100</td><td>0.045 0.176</td><td>0.025</td><td>0.137</td></tr><tr><td>1001</td><td>0.008 0.103</td><td>0.006</td><td>0.088</td></tr><tr><td>1111</td><td>0.006 0.005</td><td>0.003</td><td>-0.006</td></tr><tr><td>1010</td><td>-0.001 0.070</td><td>0.001</td><td>0.066</td></tr></table>

Table 17: KL accumulated beyond the placebo and change in rent at the end window. Arms 0000 and 0101 lower the price of headers.

In eight treated runs, more than 10% of responses reach the length limit in the end window: all six runs at $\beta \ : = \ : 0 . 0 0 1 2 5$ (every response under the price cut) and two runs with the price cut at $\beta = 0 . 0 0 5$ . Our pre-specified rule treats these windows as descriptive.

<table><tr><td></td><td colspan="2">Forecast</td><td colspan="2">Steps 20–30</td><td colspan="2">Steps 160–200</td></tr><tr><td>Arm</td><td>ρ</td><td> $z _ { g }$ </td><td> $\Delta \rho$ </td><td> $\Delta z _ { g }$ </td><td> $\Delta \rho$ </td><td> $\Delta z _ { g }$ </td></tr><tr><td>1010 1001</td><td>0.009</td><td>-0.011</td><td>0.023 0.024</td><td>-0.015 -0.029</td><td>0.070</td><td>-0.064 -0.104</td></tr><tr><td>1111 1100</td><td>0.008 0.005 0.001</td><td>-0.009 -0.005 0.000</td><td>-0.001 -0.006</td><td>-0.000 -0.003</td><td>0.100 0.004 0.160</td><td>-0.024 -0.184</td></tr></table>

Table 18: Pre-specified forecasts (covariance at the branch point, mean over six parents) and observed changes relative to the placebo, first batch.

Findings. After the price cut, the policy accumulates more KL than after the price rise at every $\beta ,$ so the pre-specified part of the mechanism in §6 holds. The overshoot that we saw in the factorial runs appears cleanly only at $\beta = 0 . 0 2$ (Figure 3). There, the response to the cut reaches about 0.9 times $C / \beta$ , close to the equilibrium of the nominal KL penalty, whereas the response to the rise stops at 0.35 times the forecast. $\mathrm { A t } \ \beta = 0 . 0 0 5$ , the two responses do not differ. At $\beta = 0 . 0 0 1 2 5$ , the policy under the price cut first adds a little formatting and then, after about 150 steps, removes formatting quickly while the length of responses grows until it reaches the limit. In the replication with equal reward spread, the factorial arms with a price cut still exceeded the forecast (1.70 times), so reward spread does not explain why the two designs differ at $\beta = 0 . 0 0 5$ . We examine the incentive in this batch in the next paragraph. Gold reward falls in both arms at small $\beta ,$ together with length, and does not follow the direction of the price change. Early in training, the policy responds to the cut more slowly than the forecast implies, which our first-order theory does not explain.

<table><tr><td>Response</td><td> $\beta$ </td><td>A</td><td>T</td><td>Slope (0–30)</td><td> $R ^ { 2 }$ </td></tr><tr><td>Fla</td><td>0.0025</td><td>2.20</td><td>34</td><td>0.055</td><td>0.40</td></tr><tr><td>Fla</td><td>0.005</td><td>1.92</td><td>16</td><td>0.055</td><td>0.45</td></tr><tr><td>F1a</td><td>0.01</td><td>1.32</td><td>10</td><td>0.060</td><td>0.23</td></tr><tr><td>F1b</td><td>0.0025</td><td>5.20</td><td>45</td><td>0.095</td><td>0.80</td></tr><tr><td>F1b</td><td>0.005</td><td>4.22</td><td>44</td><td>0.067</td><td>0.82</td></tr><tr><td>F1b</td><td>0.01</td><td>2.65</td><td>28</td><td>0.077</td><td>0.61</td></tr><tr><td> $L$ </td><td>0.0025</td><td>23.8</td><td>6</td><td>1.68</td><td>0.06</td></tr><tr><td> $L$ </td><td>0.005</td><td>28.4</td><td>12</td><td>1.05</td><td>0.18</td></tr><tr><td> $L$ </td><td>0.01</td><td>17.6</td><td>6</td><td>1.10</td><td>0.09</td></tr></table>

Table 19: Relaxation fits by response, mean over three parents.

<table><tr><td>Response</td><td>Model</td><td> $\beta _ { 0 }$ </td><td>SSE</td><td>BIC</td></tr><tr><td>F1</td><td> $\beta + \beta _ { 0 }$ </td><td>0.0065</td><td>172</td><td>-67.7</td></tr><tr><td>F1</td><td> $\beta _ { 0 } = 0$ </td><td>一</td><td>648</td><td>249.3</td></tr><tr><td>F1</td><td> $\mathrm { f r e e } A , \tau$ </td><td></td><td>165</td><td>-61.5</td></tr><tr><td>F1b</td><td> $\beta + \beta _ { 0 }$ </td><td>0.0055</td><td>57</td><td>-334.3</td></tr><tr><td>F1b</td><td> $\beta _ { 0 } = 0$ </td><td>一</td><td>241</td><td>9.2</td></tr><tr><td>F1b</td><td>free  $A , \tau$ </td><td>一</td><td>55</td><td>-329.9</td></tr></table>

Table 20: Joint fits across the three values of $\beta .$ For headers alone, the estimate of $\beta _ { 0 }$ is unstable at one parent.

Incentive along the path. After the results above, we sampled 200 prompts × 16 responses from the policies at step 400 for both treated arms and the placebo at $\beta = 0 . 0 2$ and 0.005 (18 policies), scored them with all three reward models, and recomputed the covariance $C$ of formatting with each reward difference. We wrote the criteria down before sampling, but the analysis is exploratory because we proposed it after seeing the outcomes. Table 23 gives the ratio of C on the policy at step 400 to $C$ at the branch point. Under the price rise, C falls to 0.68 at $\beta = 0 . 0 2$ , as much as on the placebo policy (0.66), so the incentive to add formatting does not saturate beyond the general drift of training. Under the price cut, the incentive to remove formatting doubles at $\beta = 0 . 0 2 ( 2 . 0 8 ;$ ; parents 1.47 to 2.86), which fits the overshoot at this $\beta . \mathrm { A t } \beta = 0 . 0 0 5$ , it changes sign at all three parents (−2.06): on these policies, which are 110 to 170 tokens longer than the placebo, the reward model with the price cut rewards formatting relative to the placebo, which fits a response that stops near half of the forecast. We do not yet know why the sign changes. The first-order incentive at the branch point thus holds near the start, and after a price cut, it can strengthen or reverse as the policy moves. In the recipe of Appendix Q, we therefore add a recomputation at a checkpoint.

<table><tr><td>Arm  $\beta$ </td><td>Slope</td><td></td><td> $\tau ~ R _ { \mathrm { e n d } }$ </td><td>Extra KL</td><td> $\Delta z _ { g }$ </td><td> $\Delta L$ </td></tr><tr><td></td><td>Rise 0.00125</td><td>0.28</td><td>200.60</td><td>-0.042</td><td>-0.13</td><td>99</td></tr><tr><td>Rise 0.005</td><td></td><td>0.24</td><td>230.52</td><td>0.010</td><td>-0.11</td><td>100</td></tr><tr><td>Rise 0.02</td><td></td><td>0.18</td><td>80.35</td><td>0.001</td><td>-0.02</td><td>23</td></tr><tr><td>Cut</td><td>0.00125*</td><td>-0.06</td><td>&gt;4002.47</td><td>0.085</td><td>-0.48</td><td>261</td></tr><tr><td>Cut</td><td>0.005</td><td>-0.07</td><td>480.53</td><td>0.038</td><td>3-0.07</td><td>150</td></tr><tr><td>Cut</td><td>0.02</td><td>-0.06</td><td>461.27</td><td></td><td>0.008-0.01</td><td>-3</td></tr></table>

Table 21: Relaxation with symmetric shocks, mean over three parents (τ : median). Slope: change in formatting per step over steps 2–20. $R _ { \mathrm { e n d } } \colon$ change in formatting over steps 360–400 divided by the frozen forecast. Extra KL: KL beyond the placebo over steps 390–400. ∆L in tokens. <sup>∗</sup>All responses reach the length limit.

## N GRPO Batch

The batch repeats the factorial design on three parents (placebo and the six arms of the main analysis, 21 runs) with one change: verl’s GRPO estimator divides each advantage by the standard deviation of its group of eight samples. We launched the runs through a wrapper of the RL launcher of Wang et al. (2026) that changes only the estimator. In a smoke run of four steps, the advantages had a mean near 0 and a range of about ±2.2 per step.

Under group normalization, the first-order response of feature k is proportional to $\mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( f _ { k } , r / s _ { x } ) ]$ , where $s _ { x }$ is the withinprompt standard deviation of the reward. We therefore pre-specified the normalized predictor $\mathbb { E } _ { x } [ \mathrm { C o v } _ { x } ( f _ { k } , r _ { \mathrm { a r m } } / s _ { x } ( r _ { \mathrm { a r m } } ) )$ $\mathrm { C o v } _ { x } ( f _ { k } , r _ { \mathrm { p l } } / s _ { x } ( r _ { \mathrm { p l } } ) ) ]$ and report the raw covariance alongside.

Table 25 lists the explained share of the primary features and the Spearman correlation of the rent forecast for each parent. The rent forecast is positive at every parent early and negative at every parent late. Figure 6 and Figure 7 show the time course of rent and the relation between extra KL and late rent for this batch next to the two RLOO batches.

<table><tr><td>Hypothesis</td><td>Observed</td><td>Outcome</td></tr><tr><td>S1 (primary): the response overshoots after the cut and falls at 2 of short after the rise</td><td> ${ } 3 \ \beta ;$  only  $\beta = 0 . 0 2$  capped windows</td><td>without Partly</td></tr><tr><td>S2 (primary): gold falls more after the cut, in step with extra KL at 1 of 3 β; Spearman with extra KL Not supported</td><td>-0.35</td><td></td></tr><tr><td>S3: Relaxation is slower after the cut</td><td>at 2 of  ${ } 3 \ \beta ;$  only  $\beta = 0 . 0 2$  capped windows</td><td>without Partly</td></tr><tr><td>S4: The policy travels more KL after the cut</td><td>at 3 of  $3 \beta$ </td><td>Supported</td></tr><tr><td>S5: Fingerprints of relaxation hold, with an effective coefficient S6: Initial responses match the forecast ratio</td><td> $\beta _ { 0 } \colon$  0.0029 (rise), 0 (cut) observed / forecast ratio: 0.44, 0.55, Not supported</td><td>Rise only</td></tr></table>

Table 22: Pre-specified hypotheses for relaxation with symmetric shocks.

<table><tr><td></td><td> $\beta = 0 . 0 2$ </td><td> $\beta = 0 . 0 0 5$ </td><td></td></tr><tr><td>Policy at step 400</td><td> $C _ { \mathrm { r i s e } }$   $C _ { \mathrm { { c u t } } }$ </td><td> $C _ { \mathrm { r i s e } }$ </td><td> $C _ { \mathrm { { c u t } } }$ </td></tr><tr><td>Price rise</td><td>0.68 0.19 0.64</td><td></td><td>0.22</td></tr><tr><td>Price cut Placebo</td><td>0.98 2.08 1.04</td><td></td><td>-2.06</td></tr><tr><td></td><td>0.660.73 0.48</td><td></td><td>0.39</td></tr></table>

Table 23: Covariance of formatting with the reward difference of each treated model, on samples of the policies at step 400, divided by its value at the branch point, mean over three parents. Diagonal of each block: the policy trained on that model.
<table><tr><td>Hypothesis</td><td>Observed Outcome</td></tr><tr><td>G1: Stable supply direc- 22/22 correct tions are correct</td><td>Supported</td></tr><tr><td>G2: Normalized covari- share 0.75; P3: Partly ance forecasts supply sizes, 0.70 vs. 0.77 at no parent worse than the (raw)</td><td></td></tr><tr><td>raw covariance G3: Late length drift is in- Spearman dependent of reward spread +0.77</td><td>Not met</td></tr><tr><td>G4: Rent forecast holds early +0.89; Supported early; the late ranking re- late -0.83; KL</td><td></td></tr></table>

Table 24: Pre-specified hypotheses of the GRPO batch.

For G3, the scale of the reward cancels under group normalization by construction, and we confirmed the normalization in the smoke run. We therefore read the remaining correlation between reward spread and length drift as a property of the arms: the arms with a larger spread are also the arms with the largest price shocks. The arms that lower the price of headers accumulate the most KL beyond the placebo (0.071 and 0.108, against at most 0.040 for the other arms), as in the RLOO batches.

<table><tr><td rowspan="3"></td><td>Steps 20–30</td><td>Steps 160–200</td><td></td><td></td></tr><tr><td>P1 P2 P3</td><td>P1</td><td>P2</td><td>P3</td></tr><tr><td>Normalized cov. 0.94 0.88 0.87</td><td>0.77</td><td>0.81</td><td>0.70</td></tr><tr><td>Raw cov.</td><td>0.90 0.84 0.86</td><td>0.74 </td><td>0.80</td><td>0.77</td></tr><tr><td>Rent (Spearman) 0.83 0.49 0.83 -0.77 -0.60 -0.83</td><td></td><td></td><td></td><td></td></tr><tr><td>Gold (Spearman) 0.66 0.26 0.77 -0.77 -0.43 -0.83</td><td></td><td></td><td></td><td></td></tr></table>

Table 25: GRPO batch by parent: explained share of F1a, F1b and length with one coefficient per feature, and the Spearman correlation across six arms between the frozen forecast and the observed change in rent and gold reward.
<table><tr><td>Arm</td><td>Extra KL, steps 160–200</td><td> $\Delta \rho ,$  step 200</td></tr><tr><td>0101</td><td>0.108</td><td>0.263</td></tr><tr><td>0000</td><td>0.071</td><td>0.134</td></tr><tr><td>1100</td><td>0.040</td><td>0.167</td></tr><tr><td>1111</td><td>0.009</td><td>-0.001</td></tr><tr><td>1001</td><td>0.007</td><td>0.103</td></tr><tr><td>1010</td><td>0.003</td><td>0.059</td></tr></table>

Table 26: GRPO batch: KL accumulated beyond the placebo and change in rent at step 200, mean over three parents.

## O Patches from Prior Work

The 15 patch groups of Wang et al. (2026) contain two pairs with identical reward rules: a length penalty of 1.3 run twice, and an audit at rate 0.05 with fine 20 run twice. We merge each pair, which leaves 13 patches. They are seven random audits (rate 0.02 with fine 50; 0.05 with 20; 0.2 with 5; 0.05 with fines 2 and 6; a back-tax audit at rate 0.05; full gold), four length penalties $( \lambda \in \{ 1 , 1 . 3 , 3 , 1 0 \} )$ , a minimum over two proxies and a targeted audit. Some patch runs end at step 150, so the time course ends there (Table 27).

## P Real RM Upgrades

From the branch points of three parents, RL continues with the original proxy in a control arm. In

<table><tr><td>All</td><td>Sign</td><td>Audits</td><td>Length</td><td>Demeaned</td></tr><tr><td>Rent, step 50</td><td>0.91 0.85</td><td>0.79</td><td>0.80</td><td>0.73</td></tr><tr><td>Rent, step 100</td><td>0.97 1.00</td><td>0.93</td><td>1.00</td><td>0.69</td></tr><tr><td>Rent, step 150</td><td>0.99 1.00</td><td>1.00</td><td>1.00</td><td>0.79</td></tr><tr><td>Gold, step 50</td><td>0.28 0.62</td><td>-0.50</td><td>1.00</td><td>0.37</td></tr><tr><td>Gold, step 100</td><td>0.40 0.54</td><td>-0.32</td><td>1.00</td><td>0.47</td></tr><tr><td>Gold, step 150 0.51</td><td>0.69</td><td>-0.04</td><td>1.00</td><td>0.56</td></tr></table>

Table 27: Retrospective forecast for 13 patches: Spearman correlation over all patches, share of correct signs, Spearman correlation within audits (7) and within length penalties (4), and after removal of type means.

three further arms, the reward model changes to a model from Wang et al. (2026): Qwen models trained on 20,000 and 80,000 pairs, and a Llama model trained on 20,000 pairs. Each model has its own standardization constants, as in Wang et al. (2026). All other settings follow the factorial runs, for a total of 12 runs. We fixed the forecasts and four hypotheses before the runs.
<table><tr><td>Upgrade</td><td>F1a F1b</td><td>F2</td><td>F4</td><td>L</td><td> $\rho$ </td><td> $z _ { g }$ </td></tr><tr><td>Qwen, 20k 0.51 0.25 0.12</td><td></td><td></td><td></td><td></td><td>0.07 0.82 -0.051 0.053</td><td></td></tr><tr><td>Qwen, 80k 0.26 0.10 0.37 0.15 1.07 −0.065 0.062</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama, 20k 0.22 0.46 0.19 0.24 1.33 -0.033 0.027</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 28: Pre-specified forecasts for real RM upgrades, mean over three parents. Supply in standardized units at the end window with κ frozen from the factorial runs; rent and gold as Cov $_ B ( \cdot , \Delta r )$ .

Outcomes. Table 29 lists the pre-specified hypotheses and their outcomes for parents P1–P3. The forecast of rent and gold reward holds in direction and in order at both windows. The observed changes at step 200 are larger than the first-order forecast, which we do not scale by κ for rent. One sub-criterion fails: under the Llama upgrade, the policy adds 3.49 times the predicted number of headers, a ratio that is unstable because the predicted change is small (0.22 standard deviations). The KL that each upgrade accumulates beyond the control (0.023, 0.024 and 0.050) is ordered like the change in rent at step 200.

Extension to six parents. After the pre-specified analysis, we ran the same upgrades from the other three parents and report the six parents together. Over steps 20–30 and at step 200, all 18 forecast signs are correct for rent and all 18 for gold reward, and the six stable directions of supply are correct. With κ frozen, the explained share of supply is 0.60 early and 0.74 at the end. Under the Llama upgrade, the policy again adds about three times the predicted number of headers (3.18).

<table><tr><td>Hypothesis</td><td>Observed</td><td>Outcome</td></tr><tr><td>U1: Stable supply directions 7/7 correct are correct</td><td></td><td>Supported</td></tr><tr><td>U2: Frozen coefficients fore- frozen κ: 0.66; Supported cast supply sizes; covariance refitted: 0.77 vs. beats price</td><td>0.30</td><td></td></tr><tr><td>U3: Early signs and ranking of signs 9/9 (rent), Supported rent and gold are correct</td><td>9/9 (gold); same</td><td></td></tr><tr><td>U4a: No late reversal</td><td>order same order at Supported</td><td></td></tr><tr><td>U4b: Rise in headers is close observed / fore- Not met (Llama) to the forecast</td><td>step 200 cast: 1.32, 1.31,</td><td></td></tr></table>

Table 29: Pre-specified hypotheses for real RM upgrades, parents P1–P3.
<table><tr><td rowspan="2">Upgrade</td><td colspan="2">Forecast</td><td colspan="2">Steps 20–30</td><td colspan="2">Step 160–200</td></tr><tr><td>ρ</td><td> $z _ { g }$ </td><td> $\Delta \rho$ </td><td> $\Delta z _ { g }$ </td><td> $\Delta \rho$ </td><td> $\Delta z _ { g }$ </td></tr><tr><td>Qwen, 20k -0.051 0.053</td><td></td><td></td><td></td><td></td><td>-0.081 0.053 -0.283 0.211</td><td></td></tr><tr><td>Qwen, 80k -0.065 0.062 -0.103 0.071 -0.380 0.273</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama, 20k -0.033 0.027 -0.066 0.036 -0.136 0.068</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 30: Forecast (covariance at the branch point) and observed change in rent and gold reward relative to the control, mean over P1–P3.

## Q A Recipe for Comparing Reward Models before Training

1. Sample $n \approx 1 6$ responses for each of a few hundred prompts from the current policy, with the decoding settings of RL.

2. Score every response with the current and the candidate reward model, and standardize each model with the constants that RL will use.

3. For every behavior f of interest, compute the within-prompt covariance $\mathrm { C o v } _ { x } ( f , \Delta r )$ and average it over prompts. Divide by the withinprompt standard deviation of f to compare behaviors.

4. Do the same for any available measure of quality, such as a stronger judge, to obtain the direction of the change in quality.

5. Read the signs as the forecast of direction. For sizes, fit one coefficient per behavior on a past edit of the same pipeline.

6. Track the KL divergence from the starting policy during RL. Once the divergence is large, the forecast of rent may no longer hold.

7. At a checkpoint, repeat steps 1–3 on samples of the current policy. If the covariance has grown, shrunk or changed sign, the forecast from the start no longer describes the incentive.