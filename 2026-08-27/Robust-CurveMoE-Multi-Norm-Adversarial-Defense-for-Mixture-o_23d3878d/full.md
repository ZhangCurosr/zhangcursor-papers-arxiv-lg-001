# Robust CurveMoE: Multi-Norm Adversarial Defense for Mixture-of-Experts Models via Mode Connectivity

Xu Zhang and Ren Wang

Abstract—Multi-norm adversarial defense aims to protect neural networks against perturbations defined by different norm constraints, but existing methods typically optimize competing robustness objectives within a single parameter configuration, leading to substantial training cost and unfavorable robustness trade-offs. We propose Robust CurveMoE, an efficient mixtureof-experts framework that connects models specialized for different perturbation norms through a low-loss path and exploits the complementary robustness profiles of models along this path. Robust CurveMoE derives clean and norm-specialized experts from robustness-constrained curve locations and selectively expertizes only influential layers, while sharing the remaining parameters across routing paths. To further reduce curve-construction cost, we introduce contribution-guided partial updating, which selects influential curve parameters using initialization-based gradient scores. We also theoretically bound the objective gap between partial and full curve optimization. Experiments on CIFAR-100 and ImageNet-100 with WideResNet and Vision Transformer architectures show that Robust CurveMoE consistently improves clean, norm-specific, and Union accuracy over MSD and ERMC. In particular, it improves Union accuracy by 2.37 and 2.13 percentage points over the strongest baseline on CIFAR-100 and ImageNet-100, respectively. Extensive ablations further validate the effectiveness of partial updating, selective expertization, and robustness-constrained expert selection.

Index Terms—Adversarial robustness, multi-norm adversarial defense, mixture of experts, robust mode connectivity, sparse routing, partial updating.

## I. INTRODUCTION

Deep neural networks have achieved remarkable performance in visual recognition, yet their predictions remain vulnerable to carefully crafted adversarial perturbations [1], [2]. Adversarial training is one of the most effective defenses and is commonly formulated as a min–max optimization problem against worst-case perturbations within a prescribed normbounded region [3]. However, robustness obtained under one threat model does not necessarily transfer to others. A model trained against $\ell _ { \infty }$ perturbations, for example, may remain vulnerable to attacks constrained by $\ell _ { 1 }$ or $\ell _ { 2 }$ norms. This mismatch motivates multi-norm adversarial defense, which seeks to protect a model against the union of multiple perturbation geometries encountered at deployment.

Existing multi-norm defenses typically optimize multiple adversarial objectives within a single parameter configuration, for example by averaging norm-specific losses, selecting the strongest attack, or constructing updates across different perturbation sets [4]–[6]. Although effective, this paradigm faces an inherent difficulty: different perturbation norms induce distinct robustness objectives that may compete during optimization. Consequently, improving robustness against one norm can compromise robustness against another or reduce clean accuracy. Moreover, multi-norm adversarial optimization requires generating attacks under several threat models and therefore introduces substantial training overhead. These limitations raise a natural question: rather than forcing all robustness behaviors into a single parameter configuration, can complementary norm-specific behaviors be preserved and adaptively exploited for each input?

Mixture-of-experts (MoE) architectures provide a promising mechanism for this purpose. By maintaining multiple specialized experts and activating only a sparse subset for each input, MoEs can increase model capacity while preserving efficient conditional computation [7]–[9]. Recent studies have further demonstrated the potential of MoE architectures for adversarial robustness [10]–[12]. Nevertheless, directly applying MoE to multi-norm defense remains challenging. Independently training complete experts for different perturbation norms requires repeated adversarial optimization and substantial parameter storage. More importantly, norm-specialized experts can exhibit highly imbalanced cross-norm robustness. An adversary that manipulates the routing decision toward an expert vulnerable to the current perturbation may therefore compromise the entire MoE even if that expert performs well under its specialized threat model. Effective multi-norm MoE defense thus requires both an efficient source of complementary experts and a principled mechanism for controlling their cross-norm vulnerability.

Robust mode connectivity provides such a structured source of experts. Mode-connectivity studies have shown that independently trained solutions can often be connected by nonlinear low-loss paths in parameter space [13], [14]. In adversarial settings, connectivity curves between norm-specialized models can contain intermediate models with distinct robustness characteristics [15]. Our experiments further reveal that different regions of such curves naturally favor different objectives: models near the endpoints retain stronger robustness to their corresponding norms, while intermediate models can exhibit complementary clean and $\ell _ { 2 }$ performance. Existing robust mode-connectivity methods, however, typically deploy only a single model selected from the learned curve, leaving much of this specialization unused. Using multiple complete curve models as experts would preserve this diversity but substantially increase parameter cost. In addition, constructing a robust curve itself remains expensive because multi-norm adversarial optimization is repeatedly performed at sampled curve locations.

Motivated by these observations, we propose Robust Curve-MoE, an efficient MoE framework that converts the complementary models encoded by a robust connectivity curve into input-adaptive multi-norm defense. We first connect $\ell _ { 1 } \cdot$ - and $\ell _ { \infty }$ -robust endpoints and optimize the resulting curve under a multi-norm adversarial objective, yielding a continuous pool of models with complementary robustness profiles. Rather than replicating complete curve models, we measure the adversarial sensitivity of candidate layers and instantiate expertspecific parameters only in the most influential layers, while sharing the remaining parameters. We further constrain expert selection to curve locations with sufficiently strong worstcase cross-norm robustness, preventing highly specialized but vulnerable models from entering the expert pool. Within this candidate set, clean and norm-specific specialists are selected to construct the experts, while a parameter-compatible model provides the shared backbone. Independent top-1 routers at the selected MoE layers then perform input-dependent sparse expert selection.

We additionally reduce the computational cost of constructing the robust connectivity curve through contribution-guided partial updating. Instead of optimizing all trainable curve parameters, Robust CurveMoE estimates their contribution to the multi-norm objective at initialization and updates only the most influential subset. We theoretically show that the objective gap between full and restricted curve optimization is bounded by the optimization displacement excluded by the frozen parameters, motivating the retention of parameters most likely to undergo substantial adaptation. Since the full optimization displacement is unavailable beforehand, we use the squared initialization gradient as a practical contribution proxy. Experiments show that this strategy preserves the robustness characteristics of the fully optimized curve while substantially reducing curve-training cost.

The main contributions of this work are summarized as follows:

• To the best of our knowledge, Robust CurveMoE is the first multi-norm adversarial defense specifically designed for MoE architectures. It transforms the complementary robustness behaviors encoded along a robust connectivity curve into specialized experts and exploits them through input-dependent sparse routing.

• We develop a robustness-constrained expert selection strategy that selects complementary specialists from the robust connectivity curve while explicitly controlling their cross-norm vulnerability. By excluding overly specialized curve models with poor worst-case robustness, the resulting expert pool is less susceptible to adversarial routing toward vulnerable experts.

• We propose contribution-guided partial curve updating to reduce curve-construction cost. We theoretically bound the objective gap between partial and full curve optimization and derive an initialization-based gradient criterion for selecting influential parameters before curve training.

• Experiments on CIFAR-100 and ImageNet-100 with WideResNet and Vision Transformer architectures demonstrate that Robust CurveMoE consistently improves clean, norm-specific, and Union robustness over representative baselines. Extensive ablations further validate the effectiveness of partial updating, selective expertization, and robustness-constrained expert selection.

## II. RELATED WORK

a) Adversarial and multi-norm robustness: Deep neural networks are vulnerable to adversarial examples, in which small, carefully crafted perturbations can induce severe prediction errors while remaining nearly imperceptible to humans [1], [2]. Adversarial training remains one of the most effective defenses and is typically formulated as a min–max optimization problem over perturbations within a prescribed norm ball [3]. Its robustness, however, is usually specialized to the threat model used during training and may transfer poorly across perturbation norms. Multi-norm defenses address this limitation by optimizing over the union of multiple threat models. Representative approaches aggregate norm-specific losses, select the strongest attack during training, or train over representative extreme norms [4]–[6], [16]. Although these methods improve robustness against multiple attacks, they generally encode competing norm-specific objectives within a single parameter configuration. This can lead to an unfavorable compromise between clean accuracy and robustness across different norms. Moreover, generating adversarial examples separately for multiple threat models introduces considerable training overhead. In contrast, we investigate multinorm defense from an MoE-oriented perspective, preserving complementary robustness behaviors across different experts and adaptively routing each input to the most suitable expert.

b) Mixture-of-experts and adversarial robustness: Mixture-of-experts (MoE) models employ a routing module to activate a sparse subset of experts for each input, increasing model capacity without a proportional increase in computation [7], [8]. Their conditional-computation structure has been widely adopted in large-scale language models and extended to vision architectures [9]. Recent work has also investigated the adversarial robustness of MoEs. Puigcerver et al. show that sparse MoEs can exhibit improved robustness over dense models with comparable computational cost [10]. AdvMoE decomposes the robustness of CNN-based MoEs into router and expert robustness and introduces alternating adversarial training to improve both components [11]. Standard and robust experts have also been combined to alleviate the clean–robust accuracy trade-off [12]. Nevertheless, existing robust MoE approaches primarily consider robustness under a single threat model or combine experts with standard-versus-robust roles. They do not explicitly preserve and route among experts specialized to different perturbation geometries. In addition, directly training multiple independent norm-specific experts would substantially increase expert construction cost. Robust CurveMoE instead derives correlated yet complementary experts from a robust connectivity curve and performs inputdependent sparse routing to achieve multi-norm robustness.

c) Mode connectivity and robust model composition: Mode connectivity reveals that independently trained solutions can often be connected by simple low-loss paths in weight space [13], [14]. This perspective has been used to study optimization stability, the structure of solution basins, and lottery-ticket subnetworks [17]. Related weight-space composition methods, such as model soups, further show that combining appropriately related models can improve predictive performance without deploying a conventional ensemble [18]. In adversarial learning, mode connectivity has been used to study robustness barriers between standard and adversarially trained solutions and to search for models exhibiting different robustness profiles against multiple $\ell _ { p }$ attacks [15], [19]. These studies suggest that a robust connectivity curve can provide a structured source of complementary models. However, existing methods generally treat individual curve locations as standalone models rather than organizing them as input-adaptive experts. Furthermore, constructing the curve requires adversarial optimization at sampled locations in weight space, while deploying multiple full curve models incurs substantial storage and inference costs. Robust CurveMoE addresses both limitations by selectively expertizing the influential components of curve-derived models and partially updating the curve parameters, reducing expert deployment and construction costs, respectively.

![](images/60706743059b937cafda29a4f2d4ec098e0998f023e52a9fbd15c6db8c8bb1db.jpg)  
Fig. 1. Overview of Robust CurveMoE. Starting from ℓ -robust and $\ell _ { \infty }$ -robust endpoint models, we first construct a robust connectivity curve using contribution-guided partial updating. We then sample representative models from the curve to form a robustness-constrained candidate set, from which clean and norm-specialized experts as well as a shared model are selected. Based on the sensitivity of different layers, only the most influential layers are converted into MoE layers, while the remaining layers are shared. Finally, the constructed MoE is fine-tuned by updating the expert and router parameters while keeping the shared backbone frozen.

## III. ROBUST CURVEMOE

An overview of Robust CurveMoE is shown in Fig. 1. Robust CurveMoE is an efficient multi-norm adversarial defense designed explicitly for MoE architectures. We first train a robust connectivity curve and sample models along the curve to construct a structured pool of experts with complementary robustness profiles. These curve-derived experts are then integrated into an MoE architecture, where inputdependent top-1 routing selects a suitable expert for each input. To retain the computational efficiency of sparse MoE training, we identify the layers most influential to multinorm robustness and instantiate experts only at the selected layers, while keeping the remaining layers shared. We further introduce contribution-guided partial curve updating to reduce the adversarial optimization cost of constructing the expert pool. Together, these designs enable Robust CurveMoE to provide multi-norm robustness while maintaining efficient expert construction, training, and inference.

## A. Preliminaries

a) Multi-norm adversarial defense: Let $f _ { \theta }$ denote a neural network parameterized by θ, and let $( { \pmb x } , y )$ be an inputlabel pair drawn from a data distribution D. An adversarial example is obtained by adding a small perturbation δ to x so that the perturbed input $\pm \delta$ causes an incorrect prediction. The allowable perturbations are commonly constrained by an $\ell _ { p }$ norm as $\begin{array} { r } { \| \delta \| _ { p } \leq \epsilon _ { p } , } \end{array}$ where $p$ specifies the geometry of the threat model and $\epsilon _ { p }$ controls its strength. Adversarial training improves robustness by minimizing the loss on the strongest perturbation within a prescribed set:

$$
\underset { \pmb { \theta } } { \operatorname* { m i n } } \ \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { D } } \left[ \underset { \pmb { \delta } \in \Delta _ { p } } { \operatorname* { m a x } } \mathcal { L } \left( f _ { \pmb { \theta } } ( \pmb { x } + \pmb { \delta } ) , \boldsymbol { y } \right) \right] ,\tag{1}
$$

where $\Delta _ { p } = \{ \pmb { \delta } : \| \pmb { \delta } \| _ { p } \leq \epsilon _ { p } \}$ and $\mathcal { L }$ is the classification loss. Because a model trained against one perturbation norm may

remain vulnerable to others, multi-norm defense considers a collection of threat models $\mathcal { P }$ and seeks robustness over their union:

$$
\operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { D } } \left[ \operatorname* { m a x } _ { { \pmb { p } } \in \mathcal { P } } \ \operatorname* { m a x } _ { \delta \in \Delta _ { \pmb { p } } } \mathcal { L } \left( f _ { \theta } ( \pmb { x } + \delta ) , \boldsymbol { y } \right) \right] .\tag{2}
$$

In this work, $\mathcal { P }$ contains the $\ell _ { 1 } , \ell _ { 2 } ,$ and $\ell _ { \infty }$ threat models. Conventional multi-norm defenses optimize these competing robustness objectives within a single parameter configuration, which may lead to an unfavorable compromise across different norms and requires adversarial examples to be generated under multiple threat models.

b) Mixture-of-experts architectures: A mixture-ofexperts (MoE) model consists of a set of expert networks and a router that determines which expert should process each input. Given an input representation $^ { h , }$ the router produces a score for each of the K experts:

$$
\pi ( h ) = \operatorname { s o f t m a x } \left( g _ { \phi } ( h ) \right) ,\tag{3}
$$

where $g _ { \phi }$ is a learnable routing function parameterized by $\phi .$ Under top-1 routing, only the expert with the highest routing score is activated:

$$
k ^ { * } ( h ) = \arg \operatorname* { m a x } _ { k \in \{ 1 , \dots , K \} } \pi _ { k } ( h ) , \qquad h ^ { \prime } = E _ { k ^ { * } ( h ) } ( h ) ,\tag{4}
$$

where $E _ { k }$ denotes the k-th expert. This conditionalcomputation mechanism allows an MoE model to increase its parameter capacity and learn specialized behaviors without activating every expert for every input. Consequently, sparse routing avoids a proportional increase in computation as the number of experts grows and enables efficient inference by evaluating only the selected expert. These architectural benefits are largely absent from conventional dense multi-norm defenses, which process every input using the same complete parameter configuration. Robust CurveMoE exploits this structure by preserving complementary robustness behaviors across experts and routing each input to a single suitable expert.

c) Mode connectivity: Mode connectivity studies whether two independently trained neural networks can be connected by a continuous path in parameter space while maintaining low loss along the path. Let $\pmb { \theta } _ { 1 }$ and $\pmb { \theta } _ { 2 }$ denote two trained endpoint models. A parameterized curve $\phi _ { \omega } ( t )$ where $t \in [ 0 , 1 ]$ ], connects them such that $\phi _ { \omega } ( 0 ) = \theta _ { 1 }$ and $\phi _ { \omega } ( 1 ) = \theta _ { 2 }$ . We employ a quadratic Bezier curve [15]:´

$$
\phi _ { \omega } ( t ) = ( 1 - t ) ^ { 2 } \pmb { \theta } _ { 1 } + 2 t ( 1 - t ) \pmb { \omega } + t ^ { 2 } \pmb { \theta } _ { 2 } ,\tag{5}
$$

where the endpoints $\pmb { \theta } _ { 1 }$ and $\pmb { \theta } _ { 2 }$ remain fixed and the bend parameter ω is optimized to obtain low-loss models along the curve. When the two endpoints are trained against different perturbation norms, models sampled from different curve locations can exhibit distinct yet complementary robustness profiles. This provides a structured source of experts for multinorm defense without requiring every expert to be trained independently from scratch.

## B. Curve-Derived Expert Pool for Multi-Norm MoE

An effective MoE architecture relies on a collection of experts with complementary capabilities. For multi-norm adversarial defense, a straightforward strategy is to train separate experts against different perturbation norms. However, independently training multiple norm-specific experts requires repeated adversarial optimization and substantially increases the cost of constructing the expert pool. We instead obtain candidate experts from a single robust connectivity curve connecting models trained under different threat models.

We construct the expert pool from a robust connectivity curve between two models specialized to different perturbation norms. Specifically, the endpoint $\pmb { \theta } _ { 1 }$ is obtained through adversarial training against $\ell _ { 1 }$ perturbations, while the endpoint $\pmb { \theta } _ { 2 }$ is trained against $\ell _ { \infty }$ perturbations. These two norms represent geometrically distinct threat models and provide complementary robustness characteristics at the endpoints. We connect them using the quadratic Bezier curve defined in´ Eq. (5). During curve training, the endpoint parameters $\pmb { \theta } _ { 1 }$ and $\pmb { \theta } _ { 2 }$ remain fixed, and only the bend parameter ω is optimized.

For each mini-batch, we sample a curve location t from the uniform distribution over [0, 1], denoted by $t \sim \mathcal { U } ( 0 , 1 )$ and instantiate the corresponding model $f _ { \phi _ { \omega } ( t ) }$ . To encourage robustness throughout the curve rather than only at its endpoints, we optimize the sampled model against the union of the $\ell _ { 1 } , \ell _ { 2 }$ , and $\ell _ { \infty }$ threat models. The robust curve objective is formulated as

$$
\operatorname* { m i n } _ { \omega } \ \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { D } } \mathbb { E } _ { t \sim \mathcal { U } ( 0 , 1 ) } \left[ \operatorname* { m a x } _ { p \in \mathcal { P } } \operatorname* { m a x } _ { \delta \in \Delta _ { p } } \mathcal { L } \left( f _ { \phi _ { \omega } ( t ) } ( \pmb { x } + \delta ) , \pmb { y } \right) \right] ,
$$

where $\mathcal { P } ~ = ~ \{ 1 , 2 , \infty \}$ . In practice, we approximate the inner maximization using the multi-steepest-descent (MSD) attack [5], which generates norm-specific candidates and selects the strongest candidate for each training sample.

After training, the resulting curve defines a continuous family of robust models. Given a set of representative locations $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { K } \}$ , we construct the candidate expert pool as

$$
\mathcal { E } _ { \mathrm { c u r v e } } = \left\{ f _ { \phi _ { \omega } ( t _ { k } ) } \vert t _ { k } \in \mathcal { T } \right\} .\tag{6}
$$

Figure 2 illustrates the clean and norm-specific adversarial performance of WideResNet-28-10 models sampled along the robust connectivity curve on CIFAR-100. Distinct regions of the curve exhibit different robustness preferences. Models near the $\ell _ { 1 }$ -robust endpoint generally retain stronger $\ell _ { 1 }$ robustness, whereas models closer to the $\ell _ { \infty }$ -robust endpoint achieve better $\ell _ { \infty }$ robustness. In contrast, $\ell _ { 2 }$ robustness is stronger at intermediate curve locations, forming two local performance peaks away from the endpoints. These observations indicate that different curve regions naturally specialize toward different evaluation objectives rather than representing redundant parameter configurations. The robust connectivity curve therefore provides a structured source of complementary candidate experts for multi-norm MoE construction.

The complementary models along the curve can be exploited through either ensemble aggregation or an MoE architecture. We argue that an MoE architecture is better suited to preserving and exploiting their distinct robustness profiles. Conventional ensemble strategies such as FGE [13] aggregate predictions from multiple constituent models for every input, regardless of their individual robustness preferences. For example, under an $\ell _ { 1 }$ attack, a model near the $\ell _ { 1 } { \mathrm { - r o b u s t } }$ endpoint may correctly classify the adversarial input, while models from other curve regions that are less robust to $\ell _ { 1 }$ perturbations may produce incorrect predictions. Aggregating these predictions can dilute the contribution of the $\ell _ { 1 }$ -robust model and potentially alter the final prediction. In contrast, an MoE architecture employs input-dependent sparse routing to select a suitable expert, thereby avoiding prediction aggregation across experts with different robustness preferences. This conditional routing mechanism better preserves the complementary robustness behaviors observed along the curve and motivates our curvederived MoE construction.

![](images/28da4b6ef33194586251140a7297dcfd1269478a514f9881d022d7e2aa4f906c.jpg)  
Fig. 2. Clean and norm-specific adversarial accuracy along the robust connectivity curve of WideResNet-28-10 on CIFAR-100, evaluated using AutoAttack. Different curve locations exhibit distinct robustness preferences.

## C. Efficient Robust CurveMoE Construction and Optimization

a) Sensitivity-guided selective expertization: A straightforward way to construct an MoE from the curve-derived models is to treat each selected model as a complete expert and employ a router to select among them. Although this design preserves the complementary robustness profiles along the curve, replicating the entire network for every expert substantially increases the number of expert-specific parameters and undermines the parameter efficiency of sparse MoE architectures. We therefore seek to retain expert-specific parameters only in the layers that are most sensitive to the adversarial objectives, while sharing the remaining layers across experts.

To quantify the sensitivity of individual layers, we partition the trainable curve parameters into M disjoint groups, $\omega = \omega ^ { ( 1 ) } , \dots , \omega ^ { ( M ) }$ , where each group corresponds to a candidate MoE layer. For ViT, we treat the MLP module in each Transformer block as a candidate layer. Given an evaluation objective $q ,$ , we measure the normalized gradient sensitivity of layer m as

$$
C _ { m } ^ { \left( q \right) } = \frac { 1 } { { { d _ { m } } } } \mathbb { E } _ { \left( { \pmb x } , { \pmb y } \right) \sim \mathcal { D } } \mathbb { E } _ { t \sim \mathcal { U } \left( 0 , 1 \right) } \left[ \left\| { \nabla _ { \omega ^ { \left( m \right) } } } { \widehat { \mathcal { I } } _ { q } } \left( { \omega ; t , \pmb x , \pmb y } \right) \right\| _ { 2 } ^ { 2 } \right] ,
$$

where $d _ { m }$ denotes the number of trainable curve parameters in layer m, and $\widehat { \mathcal { I } } _ { q }$ denotes the mini-batch objective corresponding to clean or norm-specific adversarial evaluation. Division by $d _ { m }$ removes the direct dependence on layer size, such that a larger $C _ { m } ^ { ( q ) }$ indicates a stronger response of layer m to objective q. For comparison across layers, we report the relative contribution by normalizing $C _ { m } ^ { ( q ) }$ over all candidate layers for each objective.

Figure 3 shows the relative layer-wise contributions of the candidate MoE layers in ViT-Tiny. The contributions are highly non-uniform across Transformer blocks: a small number of modules exhibit substantially stronger responses, whereas several others contribute considerably less. Moreover, the clean and norm-specific objectives exhibit both shared and objective-dependent sensitivity patterns. For example, some blocks are consistently influential across different objectives, while others show noticeably different contributions between clean and adversarial objectives.

![](images/7aa9a441f8059c5bc2becb8507d623f7825daad8aed8de7f04dd392a98f8cfdb.jpg)  
Fig. 3. Relative contribution of the candidate MoE layers in ViT-Tiny to clean and norm-specific adversarial objectives. The layer-wise contributions are highly non-uniform, and different objectives exhibit both shared and objectivedependent sensitivity patterns.

These observations suggest that replicating every layer across experts is unnecessary. Instead, the additional expert capacity can be concentrated on the layers that contribute most strongly to adversarial robustness, while the remaining layers are shared across routing paths. Accordingly, since our objective is to improve multi-norm adversarial robustness, we select the MoE layers based only on their adversarial sensitivity. For each candidate layer, we average its sensitivity scores under the $\ell _ { 1 } , \ell _ { 2 } .$ , and $\ell _ { \infty }$ objectives:

$$
C _ { m } = \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } C _ { m } ^ { ( p ) } , \qquad \mathcal { P } = \{ 1 , 2 , \infty \} .\tag{7}
$$

We then rank the candidate layers according to $C _ { m }$ and select the top-K layers for expertization:

$$
\begin{array} { r } { S _ { \mathrm { M o E } } = \mathrm { T o p K } \left( \{ C _ { m } \} _ { m = 1 } ^ { M } , K \right) . } \end{array}\tag{8}
$$

After determining the MoE layers, we next select the curve models from which the expert-specific and shared parameters are extracted. Directly searching the entire curve for the model that maximizes each individual robustness objective may result in highly specialized but vulnerable experts. Prior work has shown that, in adversarial MoE architectures, the experts can constitute a more vulnerable component than the router [12]. This issue is particularly important in our setting because the norm-specialized endpoints may achieve strong robustness against their corresponding threat models while remaining considerably less robust to other perturbation norms. For example, under an $\ell _ { \infty }$ attack, an adversary may perturb the routing decision toward an expert specialized primarily for $\ell _ { 1 }$ robustness; if this expert has weak $\ell _ { \infty }$ robustness, such misrouting can substantially degrade the final prediction. We therefore restrict both expert and shared-model selection to curve locations that maintain sufficiently strong robustness across all considered threat models.

Specifically, we uniformly discretize the trained curve into a set of sampled locations $\tau$ and define the overall robust accuracy of a model at location t as its worst accuracy among the three adversarial objectives:

$$
A _ { \mathrm { r o b } } ^ { \mathrm { v a l } } ( t ) = \operatorname* { m i n } _ { p \in \mathcal { P } } A _ { p } ^ { \mathrm { v a l } } ( t ) , \qquad \mathcal { P } = \{ 1 , 2 , \infty \} ,\tag{9}
$$

where $A _ { p } ^ { \mathrm { v a l } } ( t )$ denotes the validation accuracy at curve location t under the $\ell _ { p }$ threat model. We then construct a robustnessconstrained candidate set by retaining locations whose overall robust accuracy is within a tolerance $\tau _ { \mathrm { c a n d } }$ of the best sampled model:

$$
\mathcal { T } _ { \mathrm { c a n d } } = \left\{ t \in \mathcal { T } \bigg | A _ { \mathrm { r o b } } ^ { \mathrm { v a l } } ( t ) \geq \operatorname* { m a x } _ { t ^ { \prime } \in \mathcal { T } } A _ { \mathrm { r o b } } ^ { \mathrm { v a l } } ( t ^ { \prime } ) - \tau _ { \mathrm { c a n d } } \right\} ,\tag{10}
$$

where $\tau _ { \mathrm { c a n d } } \geq 0$ is a tolerance hyperparameter controlling the robustness range of the candidate set. A smaller $\tau _ { \mathrm { c a n d } }$ imposes a stricter multi-norm robustness requirement, whereas a larger value admits a broader range of curve models with potentially stronger specialization. This constraint prevents models with severely imbalanced robustness profiles from being used as experts while retaining sufficiently robust curve locations for subsequent expert and shared-model selection.

Within $\tau _ { \mathrm { c a n d } }$ , we select four specialist models according to their clean and norm-specific validation performance:

$$
t _ { q } ^ { * } = \arg \operatorname* { m a x } _ { t \in \mathcal { T } _ { \mathrm { c a n d } } } A _ { q } ^ { \mathrm { v a l } } ( t ) , \qquad q \in \{ \mathrm { s t d } , \ell _ { 1 } , \ell _ { 2 } , \ell _ { \infty } \} .\tag{11}
$$

Although our primary objective is multi-norm adversarial robustness, we additionally retain the clean-accuracy specialist to exploit the conditional capacity of the MoE architecture and improve standard performance without discarding the norm-specific robust experts. For every selected MoE layer $m \ \in \ S _ { \mathrm { M o E } }$ , the corresponding parameters from these four specialist models are instantiated as four expert branches.

The remaining layers are shared by all routing paths and are therefore initialized from a common base model. Since all models in $\tau _ { \mathrm { c a n d } }$ already satisfy the multi-norm robustness constraint, the shared base model is selected according to its parameter-space compatibility with the specialist models. Let $\pmb \theta ( t )$ denote the complete parameter vector of the model at curve location t. We measure the parameter-space compatibility between a candidate base model and the selected specialist models using their average normalized distance:

$$
D _ { \mathrm { b a s e } } ( t ) = \frac { 1 } { | \mathscr { Q } | } \sum _ { q \in \mathscr { Q } } \frac { \Vert \pmb { \theta } ( t ) - \pmb { \theta } ( t _ { q } ^ { * } ) \Vert _ { 2 } ^ { 2 } } { \Vert \pmb { \theta } ( t _ { q } ^ { * } ) \Vert _ { 2 } ^ { 2 } + \varepsilon } ,\tag{12}
$$

where $\mathcal { Q } = \{ { \mathrm { s t d } } , \ell _ { 1 } , \ell _ { 2 } , \ell _ { \infty } \}$ and $\varepsilon > 0$ is a small constant for numerical stability. A smaller $D _ { \mathrm { b a s e } } ( t )$ indicates that the candidate model is more compatible with the specialist models

in the overall parameter space. The shared base model is then selected from the robustness-constrained candidate set as

$$
t _ { \mathrm { b a s e } } ^ { * } = \arg \operatorname* { m i n } _ { t \in \mathcal { T } _ { \mathrm { c a n d } } } D _ { \mathrm { b a s e } } ( t ) .\tag{13}
$$

The layers outside $S _ { \mathrm { M o E } }$ are inherited from the model at $t _ { \mathrm { b a s e } } ^ { * }$ and shared across all routing paths. In this way, the resulting architecture combines robustness-constrained specialist experts with a parameter-compatible shared backbone, while introducing expert-specific parameters only at the robustnesssensitive layers.

b) Contribution-guided partial curve updating: Constructing the robust connectivity curve remains computationally expensive because, after training the two norm-specialized endpoints, the curve parameters must be further optimized using multi-norm adversarial training. Motivated by the highly non-uniform layer-wise sensitivity observed above, we investigate whether a comparable robust curve can be obtained by updating only a subset of the trainable curve parameters. Specifically, starting from the initial curve parameter ω<sub>0</sub>, we optimize only a selected subset of parameter groups while keeping the remaining groups fixed at their initial values throughout curve training.

We first examine random partial updating, where a prescribed fraction of parameter groups is randomly selected for optimization. As shown in Table I, partial updating consistently reduces curve-training time, but random parameter selection leads to noticeable performance degradation and increased variation, especially at lower updating ratios. For example, when the updating ratio is reduced to 30%, the degradation becomes more pronounced across clean and adversarial objectives, with particularly large drops under the $\ell _ { 2 }$ and $\ell _ { \infty }$ threat models. These results indicate that different parameter groups contribute unequally to robust curve optimization, and randomly excluding influential groups can substantially impair both the effectiveness and stability of the resulting curve. Therefore, an effective partial-updating strategy should identify the parameter groups that are most important to the multi-norm robust objective before curve optimization begins.

TABLE I  
EFFICIENCY AND ROBUSTNESS OF CURVE UPDATING STRATEGIES.
<table><tr><td>Updating Strategy</td><td>Ratio</td><td>Time Reduction (%)</td><td>Clean (%)</td><td>l1 (%)</td><td>l2 (%)</td><td> $\ell _ { \infty } \ ( \% )$ </td></tr><tr><td>Full Updating</td><td>100%</td><td>0</td><td>57.52 ± 2.99</td><td>18.78 ± 7.35</td><td> $3 5 . 2 9 \pm 2 . 1 2$ </td><td> $1 6 . 7 8 \pm 3 . 7 4$ </td></tr><tr><td>Random</td><td>50%</td><td> $3 5 . 2 6 \pm 0 . 5 7$ </td><td>56.32 ± 4.97</td><td>16.85 ± 9.57</td><td>32.83 ± 4.85</td><td> $1 5 . 9 1 \pm 4 . 2 1$ </td></tr><tr><td>Random</td><td>30%</td><td>36.56 ± 0.31</td><td>52.95 ± 7.92</td><td>17.93 ± 8.49</td><td> $3 0 . 7 5 \pm 6 . 7 8$ </td><td> $1 2 . 3 9 \pm 7 . 4 3$ </td></tr><tr><td>Contribution-Guided</td><td>50%</td><td>35.14 ± 0.70</td><td>57.13 ± 3.41</td><td>18.54 ± 7.45</td><td> $3 4 . 9 8 \pm 2 . 2 0$ </td><td> $1 6 . 5 8 \pm 3 . 8 6$ </td></tr><tr><td>Contribution-Guided</td><td>30%</td><td>36.05 ± 0.82</td><td>56.82 ± 3.73</td><td>18.33 ± 7.56</td><td> $3 4 . 6 7 \pm 2 . 3 6$ </td><td> $1 6 . 4 1 \pm 3 . 9 6$ </td></tr></table>

To characterize the effect of restricting curve optimization to a subset of parameters, let $\omega _ { \mathrm { 0 } }$ denote the initialization of the trainable curve parameters and let

$$
\omega ^ { * } = \omega _ { 0 } + \Delta ^ { * } ,\tag{14}
$$

where $\omega ^ { \ast }$ denotes a stationary solution obtained through full curve optimization and $\Delta ^ { * }$ represents the corresponding optimization displacement. For a subset of parameters $s ,$ let $P _ { S }$ denote the projection onto the parameter subspace specified by S. The feasible set of the corresponding partial-update problem is

$$
\mathcal { F } _ { S } = \left\{ \omega _ { 0 } + P _ { S } d : d \in \mathbb { R } ^ { d } \right\} ,\tag{15}
$$

and the restricted solution is defined as

$$
\omega _ { S } ^ { * } \in \arg \operatorname* { m i n } _ { \omega \in \mathcal { F } _ { S } } \mathcal { I } ( \omega ) ,\tag{16}
$$

where $\mathcal { I }$ denotes the curve objective.

Theorem 1 (Robust-objective preservation under restricted parameters). Suppose that $\mathcal { I }$ is L-smooth in a neighborhood containing $\omega ^ { \ast }$ and the restricted solutions, and that $\omega ^ { \ast }$ is a stationary point of the full curve optimization problem. Then,

$$
\mathcal { T } \left( \omega _ { S } ^ { * } \right) - \mathcal { I } \left( \omega ^ { * } \right) \leq \frac { L } { 2 } \left. P _ { S } ^ { \perp } \Delta ^ { * } \right. _ { 2 } ^ { 2 } ,\tag{17}
$$

where ${ \cal P } _ { S } ^ { \bot } = { \cal I } - { \cal P } _ { S } ,$

The proof is provided in Appendix A. Theorem 1 bounds the objective gap between full curve updating and partial curve updating by $\begin{array} { r } { \frac { L } { 2 } \left\| P _ { S } ^ { \perp } \Delta ^ { * } \right\| _ { 2 } ^ { 2 } } \end{array}$ . Since L is fixed for the considered objective, making the partially updated curve perform as closely as possible to the fully updated curve requires minimizing the excluded optimization displacement $\begin{array} { r } { \left\| \dot { P } _ { S } ^ { \perp } \Delta ^ { * } \right\| _ { 2 } ^ { 2 } . } \end{array}$ Therefore, the selected parameter subset S should retain the parameters that undergo the largest changes during full optimization. However, the corresponding displacement $\Delta ^ { * }$ is only available after completing full curve training, which defeats the purpose of partial updating.

To obtain a selection criterion before curve optimization begins, we approximate the importance of each parameter using its local gradient at the initial curve configuration $\omega _ { \mathrm { 0 } }$ . For parameter $\omega _ { m } ,$ we define the initialization-based contribution score as

$$
G _ { m } = \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { D } } \mathbb { E } _ { t \sim \mathcal { U } ( 0 , 1 ) } \left[ \left( \frac { \partial \widehat { \mathcal { I } } _ { \mathrm { M S D } } \left( \omega _ { 0 } ; t , \pmb { x } , \pmb { y } \right) } { \partial \omega _ { m } } \right) ^ { 2 } \right] .
$$

Here, $\widehat { \mathcal { I } } _ { \mathrm { M S D } }$ denotes the mini-batch multi-norm adversarial objective approximated using MSD. Since gradient-based optimization initially changes each parameter according to its local gradient at $\omega _ { \mathrm { 0 } }$ , a larger $G _ { m }$ indicates that parameter $\omega _ { m }$ is more strongly driven to adapt under the multi-norm robust objective. We therefore use $G _ { m }$ as a practical proxy for the parameter displacement characterized in Theorem 1.

Given an updating ratio $\rho \in \left( 0 , 1 \right]$ , we rank all trainable curve parameters according to their contribution scores and select the top $\rho$ fraction for optimization:

$$
\begin{array} { r } { S _ { \mathrm { u p d } } = \mathrm { T o p K } \left( \{ G _ { m } \} _ { m = 1 } ^ { M } , \lceil \rho M \rceil \right) , } \end{array}\tag{18}
$$

where M denotes the total number of trainable curve parameters. Starting from $\omega _ { \mathrm { 0 } }$ , only the parameters indexed by $ { S _ { \mathrm { u p d } } }$ are optimized during subsequent robust curve training, while all remaining parameters are kept fixed at their initial values. In this way, as demonstrated in Table I, contribution-guided partial updating concentrates the optimization budget on the parameters predicted to be most influential to the curve objective, reducing the cost of curve construction while avoiding the instability introduced by random parameter selection.

c) Robust MoEfine-tuning: After selective expertization, the resulting MoE combines shared parameters from the base model with expert-specific parameters extracted from different specialist locations along the curve. Although each component originates from a well-performing curve model, their recombination does not correspond to any model directly optimized along the original curve and may therefore introduce representation mismatch between the shared backbone and the inserted experts. Moreover, the newly introduced router must adapt to the heterogeneous robustness profiles of the curvederived experts. We therefore perform a joint robust finetuning stage to adapt the constructed MoE as a unified model.

Specifically, we fine-tune only the expert parameters in the selected MoE layers together with the router, while keeping all shared parameters fixed. The optimization objective is a weighted combination of the standard loss and the multi-norm adversarial loss:

$$
\operatorname* { m i n } _ { \Theta _ { \mathrm { e x p } } , \Phi } \big ( 1 - \alpha \big ) \mathcal { L } _ { \mathrm { s t d } } + \alpha \mathcal { L } _ { \mathrm { r o b } } ,\tag{19}
$$

where $\Theta _ { \mathrm { e x p } }$ denotes the parameters of the curve-derived experts, $\Phi \stackrel { \cdot } { = } \{ \phi ^ { ( m ) } \} _ { m \in \mathcal { S } _ { \mathrm { M o E } } }$ denotes the collection of router parameters associated with the selected MoE layers and $\alpha \in [ 0 , 1 ]$ controls the trade-off between standard accuracy and multi-norm adversarial robustness. Here, $\mathcal { L } _ { \mathrm { s t d } }$ denotes the classification loss on clean inputs, while ${ \mathcal { L } } _ { \mathrm { r o b } }$ denotes the MSD-approximated worst-case loss over $\mathcal { P } = \{ 1 , 2 , \infty \}$ Unlike commonly used MoE training practices, we do not employ a separate router warm-up stage or an auxiliary loadbalancing loss, as neither provides additional performance gains in our experiments; detailed comparisons are provided in the experimental section. This single-stage fine-tuning adapts the curve-derived experts and router to the shared backbone while preserving the parameters of the selected base model. The complete training and construction procedure of Robust CurveMoE is summarized in Algorithm 1.

## IV. EXPERIMENTAL EVALUATION

## A. Experimental Setup

a) Datasets and architectures: We conduct experiments on CIFAR-100 [20] and ImageNet-100 [21] using WideResNet-28-10 [22] and ViT-Tiny/16 [23], respectively. These two settings cover both convolutional and Transformerbased architectures and are used to evaluate the generality of Robust CurveMoE across model families and datasets.

b) Threat models and evaluation: We consider $\ell _ { 1 } , \ \ell _ { 2 } .$ and $\ell _ { \infty }$ adversarial perturbations throughout the experiments. For CIFAR-100, the perturbation budgets are $\epsilon _ { 1 } = 1 2$ $\epsilon _ { 2 } ~ = ~ 0 . 5$ , and $\epsilon _ { \infty } ~ = ~ 8 / 2 5 5$ . For ImageNet-100, we use $\epsilon _ { 1 } ~ = ~ 7 5 , ~ \epsilon _ { 2 } ~ = ~ 2 . 0$ , and $\epsilon _ { \infty } ~ = ~ 4 / 2 5 5$ . Robust accuracy is evaluated separately under the three threat models using AutoAttack [24], and standard accuracy is reported on clean test samples.

c) Training settings: For each dataset, we first train an $\ell _ { 1 }$ -robust endpoint and an $\ell _ { \infty }$ -robust endpoint, followed by robust curve training and MoE fine-tuning. On CIFAR-100, the WideResNet-28-10 endpoints are trained for 150 epochs, the robust connectivity curve for 50 epochs, and the resulting

```latex
Algorithm 1 Robust CurveMoE Training and Construction
Require: Training data $\mathcal { D } ;$ threat models ${ \mathcal { P } } ~ = ~ \{ 1 , 2 , \infty \} ;$
evaluation objectives $\mathcal { Q } = \{ \mathrm { s t d } , \ell _ { 1 } , \ell _ { 2 } , \ell _ { \infty } \}$ ; updating ratio
$\rho ;$ number of MoE layers $K ;$ candidate tolerance $\tau _ { \mathrm { c a n d } } ;$
fine-tuning weight α
Ensure: Robust CurveMoE model
1: Train $\ell _ { 1 } \cdot$ and $\ell _ { \infty }$ -robust endpoints $\pmb { \theta } _ { 1 }$ and $\theta _ { 2 } .$
2: Initialize the curve parameter $\omega _ { \mathrm { 0 } } .$
3: for each trainable curve parameter $\omega _ { m }$ do
4: $\begin{array} { r } { G _ { m } \gets \mathbb { E } _ { ( \pmb { x } , \pmb { y } ) \sim \mathcal { D } } \mathbb { E } _ { t \sim \mathcal { U } ( 0 , 1 ) } \left[ \left( \frac { \partial \widehat { \mathcal { I } _ { \mathrm { M S D } } } ( \omega _ { 0 } ; t , \pmb { x } , \pmb { y } ) } { \partial \omega _ { m } } \right) ^ { 2 } \right] , } \end{array}$
5: end for
6: $\begin{array} { r } { S _ { \mathrm { u p d } }  \mathrm { T o p K } ( \{ G _ { m } \} _ { m = 1 } ^ { M } , \lceil \rho M \rceil ) . } \end{array}$
7: Optimize only the parameters indexed by $ { S _ { \mathrm { u p d } } }$ under the
MSD-based multi-norm curve objective; keep all remain
ing curve parameters fixed.
8: Uniformly sample curve locations $\mathcal { T } \subset [ 0 , 1 ]$
9: for each $t \in \mathcal T$ do
10: Evaluate $A _ { q } ^ { \mathrm { v a l } } ( t )$ for all $q \in \mathcal { Q } .$
11: $\begin{array} { r } { A _ { \mathrm { r o b } } ^ { \mathrm { v a l } } ( t )  \operatorname* { m i n } _ { p \in \mathcal { P } } A _ { p } ^ { \mathrm { v a l } } ( t ) . } \end{array}$
12: end for
13: $\begin{array} { r } { \mathcal { T } _ { \mathrm { c a n d } }  \{ t \in \mathcal { T } : A _ { \mathrm { r o b } } ^ { \mathrm { v a l } } ( t ) \geq \operatorname* { m a x } _ { t ^ { \prime } \in \mathcal { T } } A _ { \mathrm { r o b } } ^ { \mathrm { v a l } } ( t ^ { \prime } ) - \tau _ { \mathrm { c a n d } } \} } \end{array}$
14: for each $q \in \mathcal { Q }$ do
15: $t _ { q } ^ { * } \gets \arg \operatorname* { m a x } _ { t \in { \mathcal { T } } _ { \mathrm { c a n d } } } A _ { q } ^ { \mathrm { v a l } } ( t ) .$
16: end for
17: for each candidate MoE layer m do
18: for each $q \in \mathcal { Q }$ do
19: Compute the trained-curve sensitivity $C _ { m } ^ { ( q ) }$
20: end for
21: $\begin{array} { r } { C _ { m } \gets \frac { 1 } { 3 } \left( C _ { m } ^ { ( \ell _ { 1 } ) } + C _ { m } ^ { ( \ell _ { 2 } ) } + C _ { m } ^ { ( \ell _ { \infty } ) } \right) } \end{array}$
22: end for
23: $S _ { \mathrm { M o E } }  \mathrm { T o p K } ( \{ C _ { m } \} , K ) .$
24: for each $t \in \mathcal { T } _ { \mathrm { c a n d } }$ do
25: $\begin{array} { r l } & { D _ { \mathrm { b a s e } } ( t ) \gets \frac { 1 } { | { \mathcal { Q } } | } \sum _ { q \in { \mathcal { Q } } } \frac { \left| \left| \theta ( t ) - \theta ( t _ { q } ^ { * } ) \right| \right| _ { 2 } ^ { 2 } } { \left| \left| \theta ( t _ { q } ^ { * } ) \right| \right| _ { 2 } ^ { 2 } + \varepsilon } } \end{array}$
26: end for
27: $t _ { \mathrm { b a s e } } ^ { * }  \arg \operatorname* { m i n } _ { t \in { \mathcal { T } } _ { \mathrm { c a n d } } } D _ { \mathrm { b a s e } } ( t ) .$
28: Construct the shared backbone from $\theta ( t _ { \mathrm { b a s e } } ^ { * } )$
29: for each $m \in S _ { \mathrm { M o E } }$ do
30: Instantiate four experts from $\{ \pmb \theta ^ { ( m ) } ( t _ { q } ^ { * } ) \} _ { q \in \mathcal { Q } } .$
31: Initialize an independent router $\phi ^ { ( m ) }$
32: end for
33: $\Phi  \{ \phi ^ { ( m ) } \} _ { m \in S _ { \mathrm { M o E } } } .$
34: Freeze all shared parameters.
35: $\begin{array} { r } { ( \Theta _ { \mathrm { e x p } } ^ { * } , \Phi ^ { * } )  \arg \operatorname* { m i n } _ { \Theta _ { \mathrm { e x p } } , \Phi } \big [ ( 1 - \alpha ) \mathcal { L } _ { \mathrm { s t d } } + \alpha \mathcal { L } _ { \mathrm { r o b } } \big ] . } \end{array}$
36: return Fine-tuned Robust CurveMoE.
```

MoE for 50 epochs. On ImageNet-100, the pre-trained ViT-Tiny/16 endpoints are trained for 30 epochs, followed by 20 epochs of curve training and 20 epochs of MoE fine-tuning.

d) Baselines and hyperparameters: We compare Robust CurveMoE with MSD [5] and ERMC [15]. Unless otherwise specified, we use a partial curve updating ratio of $\rho = 3 0 \%$ select $K \ = \ 3$ MoE layers, set the candidate tolerance to $\tau _ { \mathrm { c a n d } } = 2$ percentage points, and use $\alpha ~ = ~ 0 . 6$ to balance the standard and multi-norm adversarial objectives.

## B. Overall Multi-Norm Robustness

Table II compares Robust CurveMoE with ERMC and MSD on CIFAR-100 and ImageNet-100. In addition to clean and norm-specific adversarial accuracy, we report Union accuracy, which measures the fraction of samples that remain correctly classified under all three considered threat models, i.e., $\ell _ { 1 } , \ell _ { 2 } ,$ and $\ell _ { \infty }$ . This metric therefore provides a stricter measure of multi-norm robustness by requiring each sample to withstand the union of all perturbation sets.

TABLE II  
OVERALL CLEAN AND MULTI-NORM ADVERSARIAL ROBUSTNESS.
<table><tr><td>Dataset</td><td>Method</td><td>Clean (%)</td><td> $\overline { { \ell _ { 1 } \left( \% \right) } }$ </td><td> $\ell _ { 2 } \ ( \% )$ </td><td> $\overline { { \ell _ { \infty } \left( \% \right) } }$ </td><td>Union (%)</td></tr><tr><td rowspan="3">CIFAR-100</td><td>ERMC</td><td> $\overline { { 5 5 . 6 5 \pm 0 . 3 1 } }$ </td><td> $1 7 . 4 2 \pm 0 . 9 5$ </td><td> $3 5 . 9 0 \pm 0 . 3 4$ </td><td> $\begin{array} { l } { 1 8 . 3 2 \pm 0 . 4 3 } \\ { . - } \end{array}$ </td><td> $\overline { { 1 5 . 8 6 \pm 0 . 3 8 } }$ </td></tr><tr><td>MSD</td><td> $5 4 . 7 2 \pm 0 . 7 6$ </td><td> $2 4 . 8 5 \pm 0 . 7 9$ </td><td>33.16 ± 0.19</td><td>17.63 ± 0.49</td><td> $1 5 . 2 1 \pm 0 . 4 5$ </td></tr><tr><td>Robust CurveMoE</td><td> $5 8 . 3 4 \pm 0 . 6 4$ </td><td> $2 8 . 4 0 \pm 0 . 7 0$ </td><td> $3 6 . 4 8 \pm 0 . 7 5$ </td><td> $1 9 . 7 1 \pm 0 . 2 7$ </td><td> $1 8 . 2 3 \pm 0 . 6 \dot { 7 }$ </td></tr><tr><td rowspan="3">ImageNet-100</td><td>ERMC</td><td> $6 7 . 2 2 \pm 0 . 6 5$ </td><td> $4 2 . 1 8 \pm 0 . 1 6$ </td><td>50.68 ± 0.12</td><td> $4 0 . 4 6 \pm 0 . 5 0$ </td><td> $3 7 . 5 2 \pm 0 . 9 6$ </td></tr><tr><td>MSD</td><td> $6 9 . 9 0 \pm 0 . 3 4$ </td><td>62.10 ± 0.58</td><td> $4 9 . 8 0 \pm 0 . 2 3$ </td><td>38 20.±0.75</td><td> $3 5 . 8 4 \pm 0 . 2 6$ </td></tr><tr><td>Robust CurveMoE</td><td> $7 3 . 5 8 \pm 0 . 5 0$ </td><td> $6 7 . 6 2 \pm 0 . 6 9$ </td><td> $5 4 . 3 4 \pm 0 . 8 9$ </td><td> $4 1 . 9 6 \pm 0 . 9 5$ </td><td> $3 9 . 6 5 \pm 0 . 5 4$ </td></tr></table>

Robust CurveMoE consistently achieves the best performance across all evaluation metrics on both datasets. On CIFAR-100, it improves Union accuracy from 15.86% for ERMC and 15.21% for MSD to 18.23%, while simultaneously achieving higher clean and norm-specific robustness. In particular, its $\ell _ { 1 }$ accuracy reaches 28.40%, substantially exceeding both baselines, without sacrificing $\ell _ { 2 }$ or $\ell _ { \infty }$ robustness. Similar improvements are observed on ImageNet-100, where Robust CurveMoE achieves a Union accuracy of 39.65%, compared with 37.52% for ERMC and 35.84% for MSD. It also improves clean accuracy to 73.58% and achieves the highest robustness under all three perturbation norms. These results demonstrate that exploiting complementary curve-derived experts through input-dependent routing provides a more favorable balance between standard accuracy and multi-norm robustness.

## C. Robustness of Expert Selection

The candidate tolerance $\tau _ { \mathrm { c a n d } }$ controls how aggressively the expert pool is restricted according to cross-norm robustness. A larger tolerance admits more specialized curve models, but may also introduce experts that are highly vulnerable to perturbation norms different from their specialization. This issue is particularly relevant for adversarial MoE models because an attacker may jointly manipulate both the routing decision and the prediction of the selected expert. We therefore evaluate the robustness of expert selection under a targeted expert attack designed to expose this failure mode.

Specifically, we consider an $\ell _ { \infty }$ -bounded attack that targets the $\ell _ { 1 } \cdot$ -specialized expert. The adversary simultaneously encourages the router to select the $\ell _ { 1 }$ expert and causes the selected expert to misclassify the input. We therefore define the attack objective as

$$
\mathcal { L } _ { \mathrm { t a r g e t } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { r t } } \mathcal { L } _ { \mathrm { r t } } .\tag{20}
$$

Here, $\mathcal { L } _ { \mathrm { c l s } }$ increases the classification loss of the targeted $\ell _ { 1 }$ expert, ${ \mathcal L } _ { \mathrm { r t } }$ encourages the router to select that expert, and $\lambda _ { \mathrm { { r t } } }$ controls the relative strength of the routing objective. We set $\lambda _ { \mathrm { r t } } = 0 . 8$ in this evaluation.

As shown in Table III, increasing $\tau _ { \mathrm { c a n d } }$ generally preserves or improves clean and norm-specific adversarial accuracy, but substantially weakens robustness against the targeted expert attack. On CIFAR-100, the targeted-expert accuracy decreases from 14.58% at $\tau _ { \mathrm { c a n d } } = 2$ to 9.74% at $\tau _ { \mathrm { c a n d } } = 1 0 ,$ , despite the improvement in clean and individual-norm robustness. A similar but more pronounced trend is observed on ImageNet-100, where the targeted-expert accuracy drops from 38.18% to 15.67% as $\tau _ { \mathrm { c a n d } }$ increases from 2 to 10. This indicates that a looser candidate constraint can admit more specialized but cross-norm-vulnerable experts, which may not be exposed by standard norm-specific evaluation but can be exploited through joint routing and classification attacks. These results validate the role of the candidate constraint in improving expert-level robustness and support the use of $\tau _ { \mathrm { c a n d } } = 2$

TABLE III  
EFFECT OF CANDIDATE TOLERANCE ON EXPERT ROBUSTNESS.
<table><tr><td>Dataset</td><td>Tcand</td><td>Clean (%)</td><td>l1 (%)</td><td>l2 (%)</td><td> $\ell _ { \infty } \ ( \% )$ </td><td>Targeted Expert (%)</td></tr><tr><td rowspan="3">CIFAR-100</td><td>25</td><td>58.34 ± 0.64</td><td> $2 8 . 4 0 \pm 0 . 7 0$ </td><td> $3 6 . 4 8 \pm 0 . 7 5$ </td><td> $1 9 . 7 1 \pm 0 . 2 7$ </td><td> $1 4 . 5 8 \pm 0 . 1 5$ </td></tr><tr><td></td><td>61.28 ± 0.33</td><td> $3 1 . 1 5 \pm 0 . 3 1$ </td><td> $3 6 . 2 3 \pm 0 . 6 5$ </td><td> $2 1 . 4 5 \pm 0 . 8 3$ </td><td> $9 . 9 6 \pm 0 . 7 8$ </td></tr><tr><td>10</td><td>62.73 ± 0.62</td><td> $3 1 . 6 5 \pm 0 . 6 1$ </td><td> $3 6 . 6 8 \pm 0 . 7 4$ </td><td> $2 2 . 2 9 \pm 0 . 9 1$ </td><td> $9 . 7 4 \pm 0 . 8 2$ </td></tr><tr><td rowspan="3">ImageNet-100</td><td></td><td> $7 3 . 5 8 \pm 0 . 5 0$ </td><td> $6 7 . 6 2 \pm 0 . 6 9$ </td><td> $5 4 . 3 4 \pm 0 . 8 9$ </td><td> $4 1 . 9 6 \pm 0 . 9 5$ </td><td> $3 8 . 1 8 \pm 0 . 1 9$ </td></tr><tr><td>25</td><td>73.86 ± 0.14</td><td>68.14 ± 0.24</td><td> $5 4 . 5 3 \pm 0 . 9 7$ </td><td> $4 2 . 8 5 \pm 0 . 7 5$ </td><td> $3 1 . 7 9 \pm 0 . 5 6$ </td></tr><tr><td>10</td><td> $7 3 . 8 4 \pm 0 . 2 5$ </td><td> $6 9 . 2 9 \pm 0 . 3 5$ </td><td> $\bar { 5 } 4 . 9 1 \pm 0 . 7 2$ </td><td> $4 7 . 5 3 \pm 0 . 3 8$ </td><td> $1 5 . 6 7 \pm 0 . 5 9$ </td></tr></table>

## D. Ablation Study

a) Effect of partial updating ratio: We first investigate the effect of the partial curve updating ratio ρ by comparing contribution-guided updating with $\rho ~ \in ~ \{ 3 0 \% , 5 0 \% , 1 0 0 \% \}$ Figure 4 reports the clean and norm-specific adversarial accuracy along the resulting connectivity curves. Across both WideResNet-28-10 on CIFAR-100 and ViT-Tiny on ImageNet-100, reducing the updating ratio has only a minor effect on the curve performance. In particular, the curves obtained with 30% and 50% updating closely follow those of full updating across clean, $\ell _ { 1 } , \ell _ { 2 } ,$ , and $\ell _ { \infty }$ evaluations. The agreement is especially strong on ImageNet-100, while only small deviations are observed at some intermediate curve locations on CIFAR-100. More importantly, partial updating preserves the characteristic robustness specialization along the curve, including stronger $\ell _ { 1 }$ robustness near one endpoint and stronger $\ell _ { \infty }$ robustness toward the other. Together with the training-time reduction reported in Table I, these results show that contributionguided updating can substantially reduce curve optimization cost without materially altering the robustness characteristics of the learned connectivity path. We therefore use $\rho = 3 0 \%$ as the default setting in our experiments.

b) Effect of the number of MoE layers: As shown in Table IV, increasing the number of MoE layers from $K = 3$ to $K = 5$ consistently improves both clean and multi-norm adversarial performance on the two datasets. On CIFAR-100, the Union accuracy increases from 18.23% to 19.45%, while clean, $\ell _ { 1 } , \ell _ { 2 } ,$ and $\ell _ { \infty }$ accuracy also improve. A similar trend is observed on ImageNet-100, where the Union accuracy increases from 39.65% to 42.93%. However, further increasing K to 7 yields only marginal changes and slightly reduces the Union accuracy on both datasets. This indicates that introducing additional high-contribution MoE layers initially provides greater expert-specific capacity, whereas expertizing more layers beyond this point offers diminishing returns. Although increasing K generally improves robustness by

WRN-28-10 / CIFAR-100, 30% WRN-28-10 / CIFAR-100, 50% WRN-28-10 / CIFAR-100, 100% ViT-Tiny / ImageNet-100, 30% ViT-Tiny / ImageNet-100, 50% ViT-Tiny / ImageNet-100, 100%

![](images/64f3dcc2357ed612f637a96422c139096fcb8888e6fa77ecb4b4d2ca3f716d97.jpg)  
Fig. 4. Effect of the partial updating ratio on clean and norm-specific adversarial accuracy along the robust connectivity curve.

introducing more expert-specific capacity, it also increases the training and parameter overhead of the MoE architecture. We therefore use $K = 3$ as the default setting to achieve a better trade-off between multi-norm robustness and efficiency.

TABLE IV  
EFFECT OF THE NUMBER OF MOE LAYERS ON MODEL PERFORMANCE AND TRAINING COST.
<table><tr><td>Dataset</td><td>K</td><td>Training Time (h)</td><td>Clean (%)</td><td>l1 (%)</td><td> $\underline { { \ell 2 \ ( \% ) } }$ </td><td> $\ell \infty \ ( \not \sim )$ </td><td>Union (%)</td></tr><tr><td rowspan="3">CIFAR-100</td><td>3</td><td>14.60 ± 0.90</td><td>58.34 ± 0.64</td><td>28.40 ± 0.70</td><td> $3 6 . 4 8 \pm 0 . 7 5$ </td><td> $1 9 . 7 1 \pm 0 . 2 7$ </td><td> $1 8 . 2 3 \pm 0 . 6 7$ </td></tr><tr><td>57</td><td> $2 0 . 9 7 \pm 0 . 2 7$ </td><td>62.10 ± 0.96</td><td>31.98 ± 0.59</td><td> $4 0 . 3 \dot { 1 } \pm 0 . 9 \dot { 1 }$ </td><td>21.81 ± 0.26</td><td> $1 9 . 4 5 \pm 0 . 8 6$ </td></tr><tr><td></td><td>35.46 ± 0.95</td><td>62.85 ± 0.62</td><td>31.35 ± 0.51</td><td>40.40 ± 0.76</td><td>21.23 ± 0.12</td><td> $1 9 . 1 8 \pm 0 . 2 4$ </td></tr><tr><td rowspan="3">ImageNet-100</td><td></td><td> $1 5 . 5 3 \pm 0 . 8 1$ </td><td>73.58 ± 0.50</td><td>67.62 ± 0.69</td><td> $5 4 . 3 4 \pm 0 . 8 9$ </td><td>41.96 ± 0.95</td><td>39.65 ± 0.54</td></tr><tr><td>357</td><td>25.58 ± 0.12</td><td>77.17 ± 0.49</td><td>71.90 ± 0.48</td><td> $5 7 . 7 1 \pm 0 . 9 0$ </td><td>45.35 ± 0.11</td><td>42.93 ± 0.24</td></tr><tr><td></td><td>39.13 ± 0.63</td><td>77.40 ± 0.96</td><td>71.13 ± 0.94</td><td> $5 7 . 9 5 \pm 0 . 5 7$ </td><td>45.59 ± 0.23</td><td>42.35 ± 0.82</td></tr></table>

## E. Effect of Warm-up and Load Balancing

We adopt the same warm-up strategy as [11] and the same load-balancing loss as [9] to examine whether these commonly used MoE training components benefit Robust CurveMoE. As shown in Table V, neither warm-up nor load balancing provides consistent improvements across the two datasets. On CIFAR-100, warm-up slightly improves Union accuracy, while load balancing yields only marginal gains on several norm-specific metrics. However, these improvements are not consistently observed on ImageNet-100, where the configuration without either component achieves the highest clean, $\ell _ { 1 } ,$ and $\ell _ { \infty }$ accuracy. Overall, the performance differences among the three settings are small and do not indicate a systematic benefit from introducing either warm-up or load balancing. We therefore omit both components in Robust CurveMoE, resulting in a simpler training procedure without sacrificing clean or multi-norm robustness.

TABLE V  
EFFECT OF WARM-UP AND LOAD BALANCING ON MODEL PERFORMANCE.
<table><tr><td>Dataset</td><td>Warm-up</td><td>Load Balancing</td><td>Clean (%)</td><td>l1 (%)</td><td>l2 (%)</td><td> $\overline { { \ell _ { \infty } \left( \mathcal { U } \right) } }$ </td><td>Union (%)</td></tr><tr><td rowspan="3">CIFAR-100</td><td>Yes</td><td>No</td><td>58.28 ± 0.45</td><td>28.12 ± 0.55</td><td>36.14 ± 0.30</td><td>19.55 ± 0.74</td><td>18.90 ± 0.68</td></tr><tr><td>No</td><td>Yes</td><td>58.18 ± 0.37</td><td>28.62 ± 0.78</td><td>36.81 ± 0.93</td><td>19.77 ± 0.49</td><td>18.43 ± 0.45</td></tr><tr><td>No</td><td>No</td><td>58.34 ± 0.64</td><td>28.40 ± 0.70</td><td>36.48 ± 0.75</td><td>19.71 ± 0.27</td><td> $1 8 . 2 3 \pm 0 . 6 7$ </td></tr><tr><td rowspan="3">ImageNet-100</td><td>Yes</td><td>No</td><td>73.28 ± 0.15</td><td>67.38 ± 0.43</td><td>54.16 ± 0.17</td><td>41.16 ± 0.64</td><td>39.73 ± 0.65</td></tr><tr><td>No</td><td>Yes</td><td>73.30 ± 0.51</td><td>67.51 ± 0.82</td><td>54.79 ± 0.64</td><td>41.37 ± 0.81</td><td>39.53 ± 0.35</td></tr><tr><td>No</td><td>No</td><td>73.58 ± 0.50</td><td>67.62 ± 0.69</td><td>54.34 ± 0.89</td><td>41.96 ± 0.95</td><td>39.65 ± 0.54</td></tr></table>

## V. CONCLUSION

In this work, we presented Robust CurveMoE, an efficient MoE-based framework for multi-norm adversarial defense. Instead of independently training multiple norm-specific experts, Robust CurveMoE derives complementary experts from a robust connectivity curve and exploits their diverse robustness profiles through input-dependent sparse routing. To improve efficiency, we selectively expertize only the layers most sensitive to adversarial objectives and introduce contribution-guided partial curve updating to reduce the cost of robust curve construction. We further restrict expert selection to cross-norm robust candidate models, mitigating vulnerabilities caused by adversarially manipulated routing toward overly specialized experts. Experiments on CIFAR-100 and ImageNet-100 with both convolutional and Transformer architectures demonstrate consistent improvements in clean, norm-specific, and Union accuracy over representative multi-norm and robust modeconnectivity baselines. The ablation studies further show that the proposed partial updating and selective expertization retain strong robustness with reduced training and parameter overhead, while commonly used MoE warm-up and load-balancing strategies provide no consistent benefit. Overall, these results demonstrate that robust mode connectivity provides a structured and efficient source of specialized experts and that selectively integrating them through sparse MoE routing offers an effective approach to multi-norm adversarial robustness.

## APPENDIX PROOF OF THEOREM 1

Proof. Let

$$
\widetilde { \omega } _ { S } = \omega _ { 0 } + P _ { S } \Delta ^ { * }\tag{21}
$$

be the point obtained by retaining only the optimization displacement in the parameter subspace specified by S. Since $\omega ^ { * } = \omega _ { 0 } + \Delta ^ { * }$ , we have

$$
\widetilde { \omega } _ { S } - \omega ^ { * } = - P _ { S } ^ { \perp } \Delta ^ { * } .\tag{22}
$$

By the L-smoothness of $\mathcal { T } _ { : }$

$$
\begin{array} { c } { \displaystyle \mathcal { I } \left( \widetilde { \omega } s \right) \leq \mathcal { I } \left( \omega ^ { * } \right) + \left. \nabla \mathcal { I } \left( \omega ^ { * } \right) , \widetilde { \omega } s - \omega ^ { * } \right. } \\ { \displaystyle + \frac { L } { 2 } \left. \widetilde { \omega } s - \omega ^ { * } \right. _ { 2 } ^ { 2 } . } \end{array}\tag{23}
$$

Because $\omega ^ { \ast }$ is a stationary point of the full curve optimization problem,

$$
\nabla \mathcal { I } \left( \omega ^ { * } \right) = \mathbf { 0 } .\tag{24}
$$

Therefore,

$$
\mathcal { T } \left( \widetilde { \omega } _ { \mathcal { S } } \right) - \mathcal { I } \left( \omega ^ { * } \right) \leq \frac { L } { 2 } \left. P _ { \mathcal { S } } ^ { \perp } \Delta ^ { * } \right. _ { 2 } ^ { 2 } .\tag{25}
$$

It remains to relate $\widetilde { \omega } _ { S }$ to the optimal solution of the restricted problem. By the definition of ${ \mathcal { F } } _ { S } ,$ , choosing $d = \Delta ^ { * }$ gives

$$
\omega _ { 0 } + P s d = \widetilde { \omega } _ { S } ,\tag{26}
$$

and hence $\widetilde { \omega } _ { S } \in \mathcal { F } _ { S }$ . Since $\omega _ { S } ^ { * }$ minimizes $\mathcal { I }$ over $\mathcal { F } _ { \mathcal { S } }$

$$
\mathcal { I } \left( \omega _ { S } ^ { * } \right) \leq \mathcal { I } \left( \widetilde { \omega } _ { S } \right) .\tag{27}
$$

Combining the above inequalities yields

$$
\mathcal { T } \left( \omega _ { S } ^ { * } \right) - \mathcal { I } \left( \omega ^ { * } \right) \leq \frac { L } { 2 } \left. P _ { S } ^ { \perp } \Delta ^ { * } \right. _ { 2 } ^ { 2 } ,\tag{28}
$$

which proves Eq. (17).

## REFERENCES

[1] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus, “Intriguing properties of neural networks,” arXiv preprint arXiv:1312.6199, 2013.

[2] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” arXiv preprint arXiv:1412.6572, 2014.

[3] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” arXiv preprint arXiv:1706.06083, 2017.

[4] F. Tramer and D. Boneh, “Adversarial training and robustness for multiple perturbations,” Advances in neural information processing systems, vol. 32, 2019.

[5] P. Maini, E. Wong, and Z. Kolter, “Adversarial robustness against the union of multiple perturbation models,” in International Conference on Machine Learning. PMLR, 2020, pp. 6640–6650.

[6] F. Croce and M. Hein, “Adversarial robustness against multiple and single l p-threat models via quick fine-tuning of robust classifiers,” in International Conference on Machine Learning. PMLR, 2022, pp. 4436–4454.

[7] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538, 2017.

[8] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[9] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, “Scaling vision with sparse mixture of experts,” Advances in Neural Information Processing Systems, vol. 34, pp. 8583–8595, 2021.

[10] J. Puigcerver, R. Jenatton, C. Riquelme, P. Awasthi, and S. Bhojanapalli, “On the adversarial robustness of mixture of experts,” Advances in neural information processing systems, vol. 35, pp. 9660–9671, 2022.

[11] Y. Zhang, R. Cai, T. Chen, G. Zhang, H. Zhang, P.-Y. Chen, S. Chang, Z. Wang, and S. Liu, “Robust mixture-of-expert training for convolutional neural networks,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 90–101.

[12] X. Zhang, K. Xu, Z. Hu, and R. Wang, “Optimizing robustness and accuracy in mixture of experts: A dual-model approach,” arXiv preprint arXiv:2502.06832, 2025.

[13] T. Garipov, P. Izmailov, D. Podoprikhin, D. P. Vetrov, and ${ \mathrm { \bf A } } , \ { \mathrm { \bf G } } .$ Wilson, “Loss surfaces, mode connectivity, and fast ensembling of dnns,” Advances in neural information processing systems, vol. 31, 2018.

[14] F. Draxler, K. Veschgini, M. Salmhofer, and F. Hamprecht, “Essentially no barriers in neural network energy landscape,” in International conference on machine learning. PMLR, 2018, pp. 1309–1318.

[15] R. Wang, Y. Li, and S. Liu, “Robust mode connectivity-oriented adversarial defense: Enhancing neural network robustness against diversified \ ell p attacks,” arXiv preprint arXiv:2303.10225, 2023.

[16] P. Maini, X. Chen, B. Li, and D. Song, “Perturbation type categorization for multiple adversarial perturbation robustness,” in Uncertainty in Artificial Intelligence. PMLR, 2022, pp. 1317–1327.

[17] J. Frankle, G. K. Dziugaite, D. Roy, and M. Carbin, “Linear mode connectivity and the lottery ticket hypothesis,” in International conference on machine learning. PMLR, 2020, pp. 3259–3269.

[18] M. Wortsman, G. Ilharco, S. Y. Gadre, R. Roelofs, R. Gontijo-Lopes, A. S. Morcos, H. Namkoong, A. Farhadi, Y. Carmon, S. Kornblith et al., “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” in International conference on machine learning. PMLR, 2022, pp. 23 965–23 998.

[19] P. Zhao, P.-Y. Chen, P. Das, K. N. Ramamurthy, and X. Lin, “Bridging mode connectivity in loss landscapes and adversarial robustness,” arXiv preprint arXiv:2005.00060, 2020.

[20] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[21] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

[22] S. Zagoruyko and N. Komodakis, “Wide residual networks,” arXiv preprint arXiv:1605.07146, 2016.

[23] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in International conference on machine learning. PMLR, 2021, pp. 10 347–10 357.

[24] F. Croce and M. Hein, “Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks,” in International conference on machine learning. PMLR, 2020, pp. 2206–2216.