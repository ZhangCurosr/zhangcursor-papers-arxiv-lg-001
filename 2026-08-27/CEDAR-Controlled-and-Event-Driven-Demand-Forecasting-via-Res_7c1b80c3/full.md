# CEDAR: Controlled and Event-Driven Demand Forecasting via Residual Decomposition

Junjie Meng   
School of Artificial Intelligence and   
Data Science, University of Science   
and Technology of China   
Hefei, China   
mengjre@gmail.com   
Shujun Liu   
Alibaba Group   
Hangzhou, China   
liushujun\_uestc@163.com   
Yanyong Zhang   
School of Artificial Intelligence and   
Data Science, University of Science   
and Technology of China   
Hefei, China   
yanyongz@ustc.edu.cn   
Ranxu Zhang   
School of Artificial Intelligence and   
Data Science, University of Science   
and Technology of China   
Hefei, China   
zkd\_zrx@mail.ustc.edu.cn

Xiaoning Qi Alibaba Group Hangzhou, China xiaoning.qxn@alibaba-inc.com

Hui Xiong   
Thrust of Artificial Intelligence, The   
Hong Kong University of Science and   
Technology (Guangzhou)   
Guangzhou, China   
Department of Computer Science and   
Engineering, The Hong Kong   
University of Science and Technology   
Hong Kong SAR, China   
xionghui@ust.hk

Zi-an Zhang Alibaba Group Hangzhou, China zhangzian.zza@alibaba-inc.com

Xiaozhou Xu   
Alibaba Group   
Hangzhou, China   
heixia.xxz@alibaba-inc.com   
Chao Wang<sup>∗</sup>   
School of Artificial Intelligence and   
Data Science, University of Science   
and Technology of China   
Hefei, China   
wangchaoai@ustc.edu.cn

## Abstract

Forecasting in large-scale e-commerce marketplaces is increasingly required to support planning: merchants need to evaluate sales outcomes under future action sequences such as budget schedules, rather than passively predicting what happens next. However, most existing time series forecasting (TSF) approaches remain inherently passive. Even when incorporating operational decisions as auxiliary covariates, they typically optimize for correlation-based extrapolation under historical policies. This design sufers from autoregressive inertia and conflates endogenous market evolution with decision-induced transitions, leading to policy-insensitive rollouts and unreliable counterfactual analysis. To bridge this gap, we propose CEDAR (Controlled and Event-Driven Demand forecasting via Action-aware Residual decomposition), a two-stage framework for robust decision-conditioned simulation. In Stage I, an Action-Interleaved Transformer learns controllable action-conditioned state transitions for rollout under planned interventions. In Stage II, a Residual Correction Module leverages external event signals and LLM-assisted text representations to align noisy event descriptions with product context and correct event-driven deviations. Our

<sup>∗</sup>Chao Wang is the corresponding author.

study is enabled by a large-scale real-world dataset from Alibaba 1688, comprising approximately 32 million product trajectories with paired state–action sequences and aligned event signals. Extensive ofline experiments and online controlled experiments in production demonstrate that CEDAR consistently improves simulation accuracy over strong TSF baselines and delivers practical gains for real-world budget planning.

## CCS Concepts

• Information systems → Data mining; • Applied computing → Forecasting.

## Keywords

Time Series Forecasting, Decision-Conditioned Simulation, Sales Forecasting

## ACM Reference Format:

Junjie Meng, Ranxu Zhang, Zi-an Zhang, Shujun Liu, Xiaoning Qi, Xiaozhou Xu, Yanyong Zhang, Hui Xiong, and Chao Wang. 2026. CEDAR: Controlled and Event-Driven Demand Forecasting via Residual Decomposition. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’26), August 09–13, 2026, Jeju Island, Republic ofKorea. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3770855.3818338

## 1 Introduction

Time series forecasting (TSF) [5, 7, 34, 35] is widely used in ecommerce to support inventory planning, pricing, and marketing optimization [6, 37]. However, in large-scale marketplaces such as Alibaba 1688, forecasting is not merely about predicting the future;

![](images/1314ac635ab3a9e901467c1a53dad06119506b106bcc3dca51126d1962b038e2.jpg)  
(a) Training loss & evaluation MSE visualization.

![](images/953eb44a75622af40dc8ca72b9e9e146c84a11b06afb83fc854745b7e2691806.jpg)  
(b) Comparison of multivariate time-series models. T represents the length of prediction.  
Figure 1: Performance comparison of four TSF baselines. All models are trained on the E-Com 15-Week dataset for 10 steps prediction.

it is about evaluating planned interventions. Merchants continuously act on the system—adjusting prices, launching promotions, and allocating advertising budgets—with the explicit goal of reshaping future demand trajectories. Consequently, the central question is not simply “what will happen next?”, but rather “what would happen ifIfollow a particularbudget schedule (and otheractions) overthe next weeks?”. This shifts the goal from passive forecasting to decisionconditioned simulation [1, 38]: given historical states/actions and a future action sequence, the model should roll out the future demand trajectory under that plan.

Despite rapid progress in TSF architectures [8, 21], most popular forecasters are still trained and evaluated in a passive setting, ignoring the exogenous change. When directly applied to decision-conditioned rollout, such state-only TSF baselines (e.g., Informer [48], PatchTST [22], PETFormer [15], Timer-XL [18]) exhibit strong autoregressive inertia: they extrapolate along historical trends but cannot respond to counterfactual action schedules. As shown in Figure 1, these models perform poorly when action data is missing.

A natural remedy is to incorporate merchant operations into multivariate TSF models by treating actions (e.g., discounts, ad spend) as exogenous covariates [2, 14, 29, 44], and to further refine predictions via exogenous corrections such as RevPred [24]. While this often improves short-horizon accuracy under historically observed policies, the covariate-fusion design is still limited for interventional simulation: it typically treats controllable interventions indistinguishably from passive context by simply concatenating them with states. As a result, the model tends to capture correlations in the historical joint distribution [3, 27], entangling endogenous market evolution with decision-induced transitions and even mixing action efects with non-stationary exogenous shocks [16, 45].

Building a faithful decision-conditioned simulator therefore faces two pivotal challenges. The first is controllable dynamics modeling. To enable reliable rollouts, the model must learn how actions explicitly drive state transitions $( \mathbf { s } _ { t - 1 } \xrightarrow { \mathbf { a } _ { t } } \mathbf { s } _ { t } )$ , rather than just fitting the joint distribution of states and actions. The second is exogenous disentanglement. Real-world demand is frequently perturbed by non-stationary external factors (e.g., viral trends, holidays) that are not fully explainable by internal state-action history. If these exogenous shocks are not separated from action efects, the simulator will misattribute demand changes, leading to erroneous credit assignment and disastrous budget decisions. Therefore, we argue that robust what-if analysis in e-commerce requires an explicit, action-conditioned transition mechanism that is rollout-stable under novel action sequences, together with a separate component to absorb non-stationary exogenous shocks.

To this end, we propose CEDAR (Controlled and Event-Driven Demand forecasting via Action-aware Residual decomposition), a novel framework designed for robust decision-conditioned simulation. Our study is enabled by a large-scale real-world dataset from Alibaba 1688, containing over 32 million product trajectories with paired state–action sequences and aligned exogenous event signals, which also supports online evaluation in a production environment. Departing from the monolithic covariate-fusion approach, CEDAR explicitly disentangles the sales generation process into two stages. In Stage I, an Action-Interleaved Transformer (AIT) models decisionconditioned state transitions by interleaving state and action tokens in the causal order ${ \pmb S } _ { t - 1 }  { \pmb a } _ { t }  { \pmb S } _ { t } ,$ encouraging the backbone to capture how interventions drive subsequent state changes. In Stage II, a Residual Correction Module predicts the residual between the simulated trajectory and observations using external event signals, leveraging LLM-enhanced text understanding to align noisy, unstructured event descriptions with product contexts and correct event-driven deviations. This separation preserves controllability for planning while improving robustness to bursty shocks, enabling reliable multi-step simulation under alternative merchant action plans.

In summary, our contributions are:

• We formulate merchant-facing sales forecasting as a decisionconditioned simulation problem, highlighting the limitations of passive TSF under sequential interventions.

• We propose CEDAR, a two-stage action-aware framework that (i) learns an explicit action-conditioned transition model via an Action-Interleaved Transformer, and (ii) corrects exogenous, non-stationary deviations with an event-driven residual module.

• We validate our approach on a massive industrial dataset of 32 million trajectories and conduct online A/B testing in a real production environment. Results demonstrate that CEDAR significantly outperforms state-of-the-art baselines in simulation accuracy and delivers substantial eficiency gains in real-world budget planning. We are also working to release a version of this dataset to the community.

## 2 Related Work

## 2.1 General Time Series Forecasting and Covariate-Aware Modeling

Time Series Forecasting (TSF) [36, 49] has evolved from statistical models and recurrent neural networks to Transformer-based architectures [9, 39] and large-scale foundation models. Informer introduces probabilistic sparse attention for eficient long-horizon modeling, while Autoformer [42] and FEDformer [50] incorporate decomposition and frequency-domain learning to better capture complex temporal patterns. More recent methods such as PatchTST [22] and iTransformer [17] further refine representation learning through patch-wise tokenization and dimension inversion, achieving strong performance on multivariate forecasting benchmarks [26].

Beyond pure autoregressive modeling, a parallel line of research incorporates rich covariates to improve multi-horizon forecasting [32, 33]. Temporal Fusion Transformer (TFT) [14, 25] represents a highly influential framework in this direction, integrating variable selection networks, gated residual connections, and attention mechanisms to adaptively fuse historical observations, known future inputs, and static features. Despite its empirical success, TFT [23] fundamentally follows a covariate-conditioned forecasting paradigm, modeling $p ( \mathbf { \boldsymbol { s } } _ { t + 1 } \mid \mathbf { \boldsymbol { s } } _ { \le t } , \mathbf { \boldsymbol { a } } _ { \le t } )$ via feature-level fusion rather than explicitly learning action-conditioned state transitions. As a result, TFT primarily captures statistical correlations under historically observed policies, which limits its robustness under policy shifts and counterfactual simulation, where future action sequences deviate from training distributions.

More broadly, most multivariate TSF models [40] treat controllable variables as auxiliary covariates, implicitly assuming invariant system dynamics [19]. This assumption is misaligned with decision intensive environments such as e-commerce, where merchant actions actively reshape future trajectories. In contrast, our work formulates sales forecasting as a decision-conditioned simulation problem and explicitly models the transition operator $\mathbf { s } _ { t + 1 } = f ( \mathbf { s } _ { t } , \mathbf { a } _ { t + 1 } )$ By interleaving state and action tokens in a unified Transformer, we encode the causal temporal ordering ${ \mathbf s } _ { t - 1 } \to { \mathbf a } _ { t } \to { \mathbf s } _ { t } ,$ , enabling robust learning of action-conditioned dynamics and stable longhorizon rollout under novel intervention strategies.

## 2.2 Ofline Reinforcement Learning via Sequence Modeling

Recent advances in ofline reinforcement learning [10, 12, 13, 20, 31] have reformulated policy learning and planning as a conditional sequence modeling problem. Decision Transformer (DT) [4] pioneered this paradigm by conditioning autoregressive transformers on desired return-to-go, demonstrating that generic sequence models can achieve competitive control performance without explicit dynamic programming. Trajectory Transformer [11] further discretized continuous trajectories into token sequences and enabled long-horizon planning via beam search. Subsequent extensions improved data eficiency and deployment flexibility, including Online Decision Transformer [47], Q-learning Decision Transformer (QDT) [43], and Value-Guided Decision Transformer (VDT) [46], which introduced critic guidance, advantage weighting, and value regularization to stabilize learning and mitigate compounding errors.

Despite their success in control-centric benchmarks, these approaches are not directly applicable to merchant-facing sales forecasting and simulation. In typical e-commerce scenarios, the primary quantity of interest is the future sales trajectory itself, which simultaneously plays the role of system state and optimization objective. More fundamentally, existing DT-style models are designed to infer an implicit policy that maximizes long-term return under a fixed environment, rather than to explicitly learn a controllable state transition mechanism. In merchant budget planning, however, the core requirement is not to discover a single optimal policy, but to evaluate and compare multiple candidate strategies through reliable what-if simulation. This necessitates disentangling endogenous market dynamics from action-induced transitions [16, 45], and explicitly modeling how alternative interventions reshape future trajectories. Recent explorations in causal generative modeling, such as DoFlow [41], have highlighted the necessity of incorporating causal flows for robust interventional and counterfactual time-series prediction [28], ensuring that simulated trajectories remain physically and logically consistent under distribution shifts.

In contrast to policy-centric sequence modeling, our work focuses on learning an action-conditioned state transition operator for stable multi-step rollout. By separating controllable efects from latent external fluctuations, CEDAR enables robust trajectory simulation under diverse intervention plans, providing a principled foundation for decision-aware forecasting and strategic exploration in highly non-stationary e-commerce environments.

## 3 Method

We formalize merchant-facing sales forecasting as a decision-conditioned simulation problem. At each discrete time step �, the system is characterized by a product state vector s<sub>�</sub> $\epsilon \mathbb { R } ^ { d _ { s } }$ that captures endogenous signals such as impressions, clicks, favorites, and purchases, together with a merchant action vector $\mathbf { a } _ { t } \in \mathbb { R } ^ { d _ { a } }$ encoding controllable interventions including discount strategies and marketing expenditures. Both states and actions are aggregated over a fixed temporal granularity (e.g., weekly) in the E-Com 15-week dataset.

Given historical observations $\left\{ \mathbf { s } _ { 1 : T } , \mathbf { a } _ { 1 : T } \right\}$ and a planned future action sequence $\mathbf { a } _ { T + 1 : T + H : }$ our objective is to simulate the counterfactual evolution of product trajectories $\mathbf { S } _ { T + 1 : T + H }$ under alternative merchant policies. This formulation enables forward rollout of system dynamics conditioned on hypothetical budget allocation strategies, thereby supporting downstream tasks such as operational planning, strategy evaluation, and optimal budget scheduling. Unlike conventional TSF settings that focus on passive extrapolation, our goal is to learn a controllable state transition model that enables stable long-horizon simulation and reliable what-if analysis.

![](images/d96158f4e155a5a52beb24b538a2eb24ee0ac6abe758aaebbbd7e017e05d37dc.jpg)  
Figure 2: Framework of CEDAR. The AIT module is trained in the first stage, and frozen in the second stage.

## 3.1 Decision-Conditioned Transition Modeling

Traditional multivariate TSF formulations typically aim to learn a conditional distribution $p ( \mathbf { s } _ { t + 1 } \mid \mathbf { s } _ { \leq t } , \mathbf { a } _ { \leq t } )$ , where merchant actions are treated as auxiliary covariates. While efective for short-term prediction under historically observed policies, this paradigm fundamentally conflates endogenous system evolution and decisioninduced dynamics, resulting in models that primarily capture correlations rather than learning explicit action-conditioned transition mechanisms.

In contrast, decision-conditioned simulation requires modeling a controlled dynamical system governed by a transition operator

$$
\mathcal T : ( \mathbf s _ { t } , \mathbf a _ { t + 1 } ) \mapsto \mathbf s _ { t + 1 } ,\tag{1}
$$

which describes how merchant interventions actively shape future trajectories. This formulation aligns with the core requirement of budget planning and strategy exploration, where future actions are intentionally optimized and may deviate significantly from historical distributions. Learning such an explicit transition oper ator enables faithful rollout under hypothetical action sequences and mitigates the extrapolation failures commonly observed in covariate-based TSF models. While some approaches incorporate external context, they typically fail to disentangle endogenous system dynamics from exogenous market shocks, leading to confounded state representations.

Motivated by this perspective, we design the Action-Interleaved Transformer (AIT), which treats both states and actions as firstclass tokens and explicitly encodes their causal temporal ordering, thereby introducing a structural inductive bias toward actionconditioned transition modeling.

## 3.2 Stage I: Action-Interleaved Transformer

The first stage of CEDAR focuses on learning action-conditioned state transition dynamics. Instead of concatenating actions as exogenous features, we model both states and actions as interleaved tokens in a unified temporal sequence. Concretely, for each time step �, we construct a token ordering

$$
\mathbf { s } _ { t - 1 } \to \mathbf { a } _ { t } \to \mathbf { s } _ { t } ,\tag{2}
$$

which reflects the natural causal structure of merchant operations: historical system states inform merchant decisions, and these decisions subsequently drive state transitions.

Each state token and action token is first projected into a shared embedding space through separate linear encoders, preserving their semantic roles while enabling joint attention. A causal Transformer backbone is then applied over the interleaved sequence, ensuring that predictions at time � + 1 depend only on past and present information. Formally, the model learns a parametric transition function

$$
\hat { \mathbf { s } } _ { t + 1 } = f _ { \boldsymbol \theta } \big ( \mathbf { s } _ { \leq t } , \mathbf { a } _ { \leq t + 1 } \big ) ,\tag{3}
$$

where $f _ { \theta }$ is implemented by the Action-Interleaved Transformer.

The interleaving design imposes a structured attention pattern that explicitly aligns merchant actions with subsequent state transitions, enabling the model to learn directed influence pathways such as ${ \bf a } _ { t } \to { \bf s } _ { t }$ . Compared to covariate-based architectures, this design encourages the Transformer to focus on controllable dynamics, improving stability and generalization when simulating under novel policies.

Furthermore, product state vectors exhibit strong internal structure, including funnel-like dependencies from exposure to conversion. The self-attention mechanism within AIT naturally captures such hierarchical interactions while modeling their modulation by merchant actions, enabling fine-grained transition learning across heterogeneous behavioral signals.

AIT is trained using a one-step prediction loss over observed trajectories:

$$
\mathcal { L } _ { \mathrm { A I T } } = \sum _ { t } \| \hat { \mathbf { s } } _ { t + 1 } - \mathbf { s } _ { t + 1 } \| _ { 2 } ^ { 2 } .\tag{4}
$$

## 3.3 Stage II: Residual Correction with External Signals

While AIT captures endogenous dynamics and the direct efects of merchant actions, real-world sales trajectories are also influenced by latent, time-varying exogenous factors, including breaking news, social media trends, seasonal efects, and macroeconomic shocks. These influences introduce non-stationary perturbations that are dificult to infer solely from historical state-action trajectories, particularly in highly volatile markets.

To account for such efects without compromising the controllability and stability of the learned transition operator, we introduce a Residual Correction Module that explicitly models deviations induced by external factors. Specifically, we decompose the system evolution as

$$
\begin{array} { r } { \pmb { s } _ { t + 1 } = f _ { \theta } \big ( \pmb { s } _ { \le t } , \mathbf { a } _ { \le t + 1 } \big ) + \epsilon _ { t } , } \end{array}\tag{5}
$$

where $f _ { \theta }$ captures controllable dynamics and $\epsilon _ { t }$ represents latent external perturbations. Stage I models the former, while Stage II estimates the latter.

External Signal Construction. We leverage a large language model (LLM) to extract structured representations of exogenous market signals [19, 30]. For each time window �, we collect news from the previous week and prompt the LLM to extract product-level key words indicative of potential demand surges. Simultaneously, major holidays and seasonal events occurring in the current window are appended as additional signals. Rather than treating these cues as isolated tags, we instruct the LLM to synthesize them into a coherent natural language description, which is subsequently encoded by a pre-trained text encoder to produce a dense hotspot embedding $\mathbf { h } _ { t }$ that captures the semantic context of external influences. Details of hotspot embedding generation can be found in Appendix A.

Residual Prediction. As shown in Figure $^ { 2 , }$ we construct item status embedding through concatenating the embeddings of item titles and the tags, the predicted next state $\hat { \mathbf { s } } _ { t + 1 }$ and recent historical states $\mathbf { s } _ { t - k : t }$ . The hotspot embedding h<sub>�</sub> is then combined with the item status embedding via a cross-attention module, enabling dynamic alignment between external events and product-level temporal patterns. The resulting representation is passed through a multi-layer perceptron to estimate the residual correction:

$$
\begin{array} { r } { \Delta s _ { t + 1 } = g _ { \phi } ( \mathbf h _ { t } , \hat { s } _ { t + 1 } , s _ { t - k : t } ) . } \end{array}\tag{6}
$$

The loss of Stage-II training is calculated through:

$$
\mathcal { L } _ { \mathrm { R C } } = \sum _ { t } \big \| \mathbf { s } _ { t + 1 } - \hat { \mathbf { s } } _ { t + 1 } - \Delta \mathbf { s } _ { t + 1 } \big \| _ { 2 } ^ { 2 } .\tag{7}
$$

The final simulated state is then obtained as

$$
\tilde { \mathbf { s } } _ { t + 1 } = \hat { \mathbf { s } } _ { t + 1 } + \Delta \mathbf { s } _ { t + 1 } .\tag{8}
$$

## 3.4 Discussion

3.4.1 Why exclude temporal signals. During our empirical investigation, we also explored explicitly modeling periodic temporal signals and injecting them as additional perturbation embeddings into the predictive sequence. However, experimental results indi cate that such temporal embeddings are dificult to align with the semantic representations of bursty external events extracted from text, leading to inefective fusion. Moreover, explicitly modeling standalone temporal embeddings substantially increases the learning complexity of the network, resulting in degraded predictive performance.

We hypothesize that, in e-commerce settings, the primary performance gains attributed to temporal signals largely stem from holiday-driven demand fluctuations. Since major holidays and seasonal events are already explicitly incorporated into the Residual Correction Module through LLM-based external signal construction, introducing timestamp embeddings as an additional perturbation provides limited marginal benefit. Consequently, we do not include explicit timestamp embeddings in CEDAR.

3.4.2 The design of Residual Correction. While prior work [24] similarly utilized a residual module to refine predictions from the backbone model, it relied on simple MLPs to process external covariates independently. This approach treats all covariates uniformly, thereby overlooking the semantic alignment between external shocks and specific items. For instance, the demand for seasonal items, such as Christmas hats, is unlikely to surge during the Chinese New Year, despite the presence of a significant festival event. In contrast, CEDAR employs a cross-attention mechanism. Specifically, the encoded external shock embeddings serve as queries to explicitly explore and model the relevance between global events and item-specific dynamics. This mechanism enables CEDAR to dynamically align external impacts with internal product states, thereby significantly enhancing the model’s efectiveness in capturing complex dependencies.

## 3.4.3 Training and Inference Pipeline.

Training Phase. We adopt a decoupled two-stage training paradigm. This design choice is primarily motivated by the need to isolate controllable system dynamics from stochastic external perturbations. By separating the learning process, we ensure that the action-conditioned transition operator remains stable under sig nificant policy shifts, while the residual correction module can flexibly adapt to non-stationary market conditions. This decoupling prevents the model from over-relying on exogenous correlations, which empirically leads to substantial improvements in longhorizon rollout accuracy and enhances the robustness of "what-if" strategy simulations.

Inference Phase. During the iterative simulation process, the state for each successive time step is generated through a sequential execution of the model components, where the Action-Interleaved Transformer first generates an initial state estimate based on the historical trajectory and planned interventions, which is then refined by the Residual Correction Module to account for adjustments necessitated by latent external signals. The final predicted state for the time step is obtained as the sum of these two outputs; to perform multi-step forecasting, this result is appended to the historical sequence and fed back into the model in an auto-regressive manner, enabling the stable simulation of future product trajectories over an extended horizon.

## 4 Experiments

In this section, we conduct extensive experiments to systematically evaluate the efectiveness of the proposed CEDAR framework, including both the Action-Interleaved Transformer (AIT) and the Residual Correction Module. Our evaluation is designed to answer the following key research questions:

• RQ1: How does CEDAR perform compared to state-of-theart time series forecasting baselines in decision-conditioned sales simulation?

• RQ2: To what extent does explicitly modeling merchant actions as a dedicated modality improve forecasting accuracy and rollout stability?

• RQ3: Can the proposed Residual Correction Module efectively capture latent external impacts and further refine simulated trajectories?

• RQ4: What is the real-world impact of deploying CEDAR in online budget planning scenarios on merchant engagement and platform performance?

## 4.1 Evaluation Setup

4.1.1 Dataset. To support large-scale training and evaluation, we construct a comprehensive e-commerce dataset from Alibaba 1688, covering approximately 32 million product trajectories spanning the years 2024 and 2025. Each trajectory consists of a sequence of product states and merchant actions aggregated over rolling 7-day windows.

Specifically, the state vector includes nine core indicators: 7-day aggregated impressions, page views, favorites, add-to-cart events, number ofbuyers, gross merchandise volume (GMV), advertisement impressions, advertisement clicks, and the number of customer inquiries. The action vector contains two controllable variables: the 7-day average marketing discount (defined as the ratio between the discounted price and the original price) and the total advertising expenditure during the same period.

We segment each trajectory into overlapping windows of 15 consecutive weeks. A new window is sampled every 15 weeks, while remaining weeks within each year are backward-sampled to construct additional windows. This process yields approximately 32 million training samples. To fully evaluate the model’s ability on unseen external shocks, we reserve the final window of 2025 as the test set and use all preceding windows for training, because random sampling may cause shortcut learning on already observed external shocks.

Each product in the dataset is annotated with a three-level category taxonomy, consisting of a primary category, secondary category, and fine-grained subcategory. At the finest granularity, the taxonomy contains 8,942 distinct subcategories. Due to the substantial heterogeneity in scale across product categories, we perform category-wise normalization. Specifically, for each subcategory, we compute the mean and standard deviation of all nine state variables and two action variables, and apply z-score normalization within each subcategory to mitigate scale disparities and stabilize model training.

To account for abrupt external shocks and bursty demand fluctuations, we further incorporate exogenous event signals derived from both on-platform trending search topics and of-platform pub lic hotspots, along with a curated list of 26 major holidays. These signals are used to support the modeling of latent external factors, with representative examples provided in the Appendix.

As the largest domestic B2B wholesale platform in China, Alibaba 1688 provides a uniquely rich environment for studying decision-conditioned forecasting and budget planning. The resulting E-Comm 15-Week dataset represents one of the largest realworld benchmarks for action-aware sales simulation, and we are actively working toward releasing a partially anonymized version to facilitate future research.

4.1.2 Baselines. We compare CEDAR against a diverse set of representative time series forecasting baselines, covering both classical architectures and recent foundation models. The selected methods span convolutional, attention-based, patch-wise, and large-scale pretrained paradigms, providing a comprehensive evaluation across diferent modeling principles. Specifically, we consider:

• Informer [48] is a canonical Transformer-based forecasting model that introduces probabilistic sparse attention to eficiently model long-range temporal dependencies.

• TFT [14] integrates exogenous covariates by variable selection networks, gated residual connections and attention mechanisms.

• PatchTST [22] reformulates time series forecasting by segmenting the input sequence into local patches and applying channel-independent Transformer encoders.

• PETFormer [15] integrates periodicity-aware encoding and eficient temporal attention mechanisms to explicitly capture seasonal structures and long-term dependencies.

• Timer-XL [18] is a large-scale pretrained time series foundation model trained on heterogeneous real-world corpora.

For a fair comparison, all baselines are trained under identical data splits, input horizons, and prediction windows. All ofline results are reported as the mean and standard deviation over five independent runs with diferent random seeds. This protocol allows us to evaluate whether the observed improvements are robust to optimization randomness rather than arising from a single favorable initialization. For models that support exogenous covariates, merchant actions are concatenated with the state variables as additional input channels. This setup reflects the prevailing practice in multivariate TSF, where actions are treated as auxiliary contextual features. In contrast, CEDAR explicitly models merchant actions as a first-class modality and interleaves them with state transitions, enabling direct learning of decision-conditioned dynamics and stable multi-step rollouts under alternative action plans. A detailed wall-clock cost analysis is provided in Appendix C.

Table 1: Performance comparison of CEDAR and other baselines. The next 5 setting predicts the next 5 weeks given the previous 10 weeks, while the next 10 setting predicts the next 10 weeks given the previous 5 weeks. All ofline results are reported as mean ± standard deviation over five independent runs.
<table><tr><td>Metric</td><td>Setting</td><td>CEDAR</td><td>Informer</td><td>TFT</td><td>PatchTST</td><td>PETFormer</td><td>Timer-XL</td></tr><tr><td rowspan="2">MSE↓</td><td>next 10</td><td> $\mathbf { 0 . 4 1 4 { \pm } 0 . 0 1 5 }$ </td><td>32.2±21</td><td> $0 . 7 2 2 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 8 4 9 { \scriptstyle \pm 0 . 0 2 6 }$ </td><td> $0 . 7 4 3 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $2 . 9 8 { \pm } 0 . 1 4$ </td></tr><tr><td>next 5</td><td> $\mathbf { 0 . 1 8 2 { \pm 0 . 0 0 6 } }$ </td><td> $3 . 4 4 { \pm } 1 . 8 $ </td><td> $0 . 5 7 2 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 4 2 4 { \pm } 0 . 0 1 1$ </td><td> $0 . 4 3 4 { \pm } 0 . 0 1 3$ </td><td> $1 . 3 4 { \pm } 0 . 0 5$ </td></tr><tr><td rowspan="2">MAE↓</td><td>next 10</td><td> $\mathbf { 0 . 1 3 2 { \pm } 0 . 0 0 2 }$ </td><td> $3 . 2 3 { \pm } 2 . 1 $ </td><td> $0 . 1 9 4 { \pm } 0 . 0 0 7$ </td><td> $0 . 2 0 1 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 1 9 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 6 2 7 { \pm } 0 . 0 3 5$ </td></tr><tr><td>next 5</td><td> $\mathbf { 0 . 0 6 0 3 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $0 . 7 3 0 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $0 . 1 7 5 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 1 2 9 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 1 3 9 { \pm } 0 . 0 0 4$ </td><td> $0 . 4 7 2 { \scriptstyle \pm 0 . 0 1 8 }$ </td></tr><tr><td rowspan="2">NMSE↓</td><td>next 10</td><td> $\mathbf { 0 . 1 8 9 { \pm } 0 . 0 0 4 }$ </td><td> $1 4 . 7 { \pm } 9 . 2 $ </td><td> $0 . 3 3 7 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 3 8 7 { \pm } 0 . 0 1 3$ </td><td> $0 . 3 3 9 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $1 . 3 6 { \pm } 0 . 0 8$ </td></tr><tr><td>next 5</td><td> $\mathbf { 0 . 0 8 3 0 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $1 . 5 7 { \pm } 0 . 8 0 $ </td><td> $0 . 2 6 7 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 1 9 1 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 1 9 8 { \pm } 0 . 0 0 7$ </td><td> $0 . 6 1 2 { \scriptstyle \pm 0 . 0 2 2 }$ </td></tr></table>

Table 2: Ablation study results using MSE and MAE metrics. One-stage training refers to jointly training the Action-Interleaved Transformer and the Residual Correction Module.
<table><tr><td rowspan="2">Ablation Variants</td><td colspan="2">MSE↓</td><td colspan="2">MAE↓</td></tr><tr><td>next 10</td><td>next 5</td><td>next 10</td><td>next 5</td></tr><tr><td>Full Model</td><td>0.414</td><td>0.182</td><td>0.132</td><td>0.0603</td></tr><tr><td>w/o Recent states</td><td>0.443</td><td>0.209</td><td>0.137</td><td>0.0634</td></tr><tr><td>w/o Product metadata</td><td>0.466</td><td>0.227</td><td>0.143</td><td>0.0654</td></tr><tr><td>w/o AIT Prediction</td><td>0.499</td><td>0.264</td><td>0.181</td><td>0.0697</td></tr><tr><td>w/o Holiday keywords</td><td>0.451</td><td>0.213</td><td>0.169</td><td>0.0651</td></tr><tr><td>w/o News keywords</td><td>0.417</td><td>0.188</td><td>0.141</td><td>0.0601</td></tr><tr><td>w/o All(AIT only)</td><td>0.489</td><td>0.252</td><td>0.177</td><td>0.0691</td></tr><tr><td>Temporal shuffle</td><td>0.527</td><td>0.274</td><td>0.194</td><td>0.0712</td></tr><tr><td>One stage</td><td>0.471</td><td>0.231</td><td>0.158</td><td>0.0674</td></tr></table>

4.1.3 Evaluation Metrics & Implementation Details. We adopt three widely used time series forecasting metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and Normalized Mean Squared Error (NMSE). The NMSE is computed as $\begin{array} { r } { N M S E = \frac { M S E } { M e a n ( y ^ { 2 } ) } . } \end{array}$ To better align with real-world merchant requirements, we consider two evaluation settings: (i) forecasting the next 10 weeks given the past 5 weeks of observations, and (ii) forecasting the next 5 weeks given the past 10 weeks of observations.

All experiments are conducted on a cluster equipped with 4 NVIDIA H20 GPUs for both training and inference. The hidden dimension of all models is uniformly set to 256. For Transformerbased models, including CEDAR, we configure the number of attention heads and layers as $n _ { \mathrm { h e a d } } = 4$ and $n _ { \mathrm { l a y e r } } = 5 _ { \mathrm { ; } }$ , respectively.

Given that the majority of textual data in our dataset is in Chinese, we adopt the BGE-zh-v1.5 model as our text encoder, with an embedding dimension of 1024. For hotspot keyword extraction and sentence organization, we employ the Qwen-Plus model.

## 4.2 Validation of CEDAR (RQ1)

Table 1 reports the quantitative comparison between CEDAR and a diverse set of representative TSF baselines, including Informer, PatchTST, PETFormer, and Timer-XL, under diferent forecasting horizons. We observe that CEDAR consistently achieves the best performance across all metrics and horizons, demonstrating substantial improvements over existing methods.

![](images/b04678baba288cd6ca945046819102a53c1e463b759cd1324660651d27c9b6c3.jpg)  
Figure 3: Visualization of predicted views trajectories of different baselines.

Specifically, at horizon next 5, CEDAR attains an MSE of 0.182, significantly outperforming the strongest baseline PatchTST 0.424 and PETFormer 0.434, corresponding to relative improvements of 57.1% and 58.1%, respectively. Similar trends are observed under NMSE, where CEDAR reduces the error to 0.083, yielding more than 56% improvement over the best baseline. For the next 10 horizon, CEDAR also achieves the lowest MSE 0.414 and NMSE 0.189, consistently outperforming all competing approaches.

Notably, classical TSF models such as Informer sufer from severe performance degradation, with MSE exceeding 30 at next 10. This phenomenon highlights the limitation of decision-unaware forecasting models, which fail to capture the strong and non-stationary efects induced by merchant actions (e.g., pricing and trafic investments). In contrast, CEDAR explicitly models the interaction between system states and merchant decisions, enabling robust and accurate simulation under dynamic, intervention-driven environments. Besides, the ablation results in Table 2 also prove the efectiveness of two-stage training.

To further evaluate CEDAR’s performance on long-horizon prediction, we conduct additional long-horizon forecasting experiments, with detailed results reported in Appendix D.

Overall, these results verify the superiority and robustness of CEDAR in decision-conditioned sales forecasting, particularly in scenarios characterized by strong action-driven distribution shifts.

## 4.3 Action Modality Ablation (RQ2)

To evaluate the impact of explicitly modeling merchant actions as a dedicated modality, we conduct a targeted analysis focusing on trafic forecasting, where advertising expenditure exhibits the strongest correlation with future dynamics. We compare CEDAR against multiple baselines that incorporate actions as auxiliary covariates, including PatchTST, Timer-XL, as well as an additional multilayer perceptron (MLP) baseline that directly concatenates historical states, product metadata, and action variables as input features.

Figure 3 illustrates the predicted trafic trajectories for a representative product, where the merchant launches two advertising campaigns starting at weeks 5 and 12, respectively. Notably, baselines that treat actions as exogenous covariates exhibit limited sensitivity to intervention signals, particularly during the early phase of training. In the first 10 weeks, these models tend to extrapolate local trends from historical trafic peaks, with predicted local maxima typically lagging behind previously observed high-trafic points. This behavior reflects a strong reliance on autoregressive correlations, causing action efects to be largely diluted by dominant state dynamics.

In contrast, CEDAR, by explicitly interleaving actions with state transitions, demonstrates substantially improved responsiveness to marketing interventions. In particular, CEDAR accurately captures the decline in future trafic following the reduction of advertising expenditure, correctly anticipating the downward trend rather than merely propagating historical patterns. This ability enables stable and realistic multi-step rollouts under dynamically changing action plans, which is critical for budget planning and strategy exploration.

These findings indicate that modeling merchant actions as a first-class modality introduces a crucial structural inductive bias, allowing the model to disentangle endogenous temporal dynamics from decision-induced transitions. As a result, CEDAR achieves superior forecasting accuracy and significantly enhanced rollout stability in counterfactual scenarios involving policy shifts.

## 4.4 External Shock Controls (RQ3)

To assess whether the proposed Residual Correction Module efectively captures latent external impacts and refines simulated trajectories, we conduct a dedicated ablation study, with quantitative results reported in Table 2. These variants remove or perturb diferent input sources used by the Residual Correction Module. In addition, we include a temporal-shufle variant, where holiday and hotspot signals are randomly reassigned to diferent dates. This variant performs worse than the aligned-event setting and even degrades relative to AIT-only in next-5 MSE, indicating that CEDAR benefits from temporally meaningful event-demand alignment rather than merely using event embeddings as generic auxiliary features.

Overall, we observe that incorporating the Residual Correction Module yields a modest improvement in MSE but leads to a substantial reduction in MAE. This discrepancy is expected, as external shocks often manifest as abrupt, localized deviations rather than long-term trend shifts. Consequently, the module primarily enhances short-horizon accuracy and robustness to bursty fluctuations, which is better reflected by MAE than by MSE.

![](images/1f48416a0621a7b92c1bcf578f71c1efaadf7b68f40bfec5a8bd6ff03c828e8d.jpg)  
Figure 4: Visualization of predicted views trajectories with next-1 setting of diferent baselines.

Beyond aggregate error metrics, the Residual Correction Module plays a critical diagnostic role in preserving trajectory diversity and capturing idiosyncratic dynamics across similar products. Without this module, the model tends to generate highly similar trend forecasts for items belonging to the same fine-grained category, resulting in homogenized predictions that fail to reflect productspecific external influences. By explicitly modeling residual signals induced by latent exogenous factors, the proposed module reintroduces trajectory heterogeneity and substantially improves realism in multi-step rollouts.

We further illustrate the critical efect of external shock modeling through a real-world case study shown in Figure 4. The example corresponds to a sudden viral trend on social media surrounding “Tanghulu with Naipizi” (a fusion snack combining milk skin and candied hawthorn), which triggered a sharp surge in demand for a particular tanghulu supplier. In this setting, the model is tasked with performing next-1 forecasting, using observations from the preceding five weeks to predict the state vector of the subsequent week.

As observed, baseline models lacking explicit external event modeling mechanisms continue to extrapolate along previously observed trajectories, failing to anticipate the upcoming sales spike. In contrast, the Residual Correction Module successfully extracts highly relevant keywords (e.g., “Tanghulu”) from trending search and external hotspot signals, enabling the model to correctly predict the abrupt increase in trafic and sales. This qualitative result demonstrates that the proposed module efectively captures latent external shocks and injects critical event-driven signals into the forecasting process.

Taken together, both quantitative and qualitative evidence confirms that the Residual Correction Module is essential for robust short-term forecasting under volatile market conditions, significantly enhancing the model’s responsiveness to bursty external events and improving the fidelity of simulated trajectories.

## 4.5 Public Dataset Generalization

To evaluate generalization beyond Alibaba 1688, we conduct additional experiments on the public Kaggle Store Sales dataset. Although this dataset contains retail demand series, promotion-related covariates, and calendar events, it difers from our setting because it is organized at the store-family level and lacks merchant actions with explicit budget-planning semantics. Thus, this experiment mainly tests whether the event-aware residual decomposition transfers to a public retail forecasting scenario, rather than fully reproducing our counterfactual budget-planning task.

Table 3: Performance comparison on the public Kaggle Store Sales dataset under the next-5 setting.
<table><tr><td>Metric</td><td>CEDAR</td><td>PatchTST</td><td>PETFormer</td></tr><tr><td>MSE↓</td><td>0.5819</td><td>0.6814</td><td>0.6321</td></tr><tr><td>MAE↓</td><td>0.3680</td><td>0.4112</td><td>0.4623</td></tr><tr><td>NMSE↓</td><td>0.3778</td><td>0.4425</td><td>0.4104</td></tr></table>

For this public benchmark, we construct event inputs from holiday metadata and use a training protocol consistent with our Alibaba 1688 experiments. The results are reported in Table 3. CEDAR consistently outperforms the representative time-series baselines across all metrics under the next-5 setting. Specifically, CEDAR reduces MSE from 0.6321 to 0.5819 compared with the strongest baseline PETFormer, and also achieves lower MAE and NMSE than both PatchTST and PETFormer. These results suggest that the proposed decomposition between base forecasting and event-driven residual correction is not specific to the Alibaba 1688 dataset, and can also provide benefits in public retail forecasting scenarios where external events influence future demand.

## 4.6 Online A/B Test Results (RQ4)

To validate the practical efectiveness of CEDAR in real-world budget planning scenarios, we deploy the proposed framework in Alibaba 1688’s online advertising and marketing optimization system and conduct a large-scale A/B test from January 1 to January 30, 2026. The initial deployment successfully engaged 239 cooperative merchants across 245 orders for the prediction service, facilitating a total transaction value of 8.77 million RMB with an initial repurchase rate of 61%.

In the online setting, CEDAR is used to generate decision-conditioned sales simulations under alternative budget allocation strategies, which are then integrated into the platform’s recommendation and planning pipeline. The control group follows the existing production model based on a difusion-based time series forecasting model with only total budgets, while the treatment group adopts CEDAR-driven budget planning and trafic allocation strategies.

The experimental results demonstrate substantial and consistent improvements. Specifically, merchants in the treatment group achieve a 13%(46,471 vs 41,228) increase in lifetime value (LTV) and a 15% improvement in store-level return on investment (ROI) on average. These gains are observed across both short-term promotional campaigns and long-term operational planning, indicating robust performance under diverse business conditions.

We attribute the observed improvements to two key factors. First, by explicitly modeling action-conditioned state transitions, CEDAR enables more accurate simulation of future demand trajectories under alternative marketing strategies, allowing merchants to proactively adjust budgets before trafic and conversion dynamics materialize. Second, the incorporation of external shock modeling further enhances responsiveness to bursty demand and emerging trends, enabling timely reallocation of marketing resources during critical periods. Together, these capabilities significantly reduce inefective ad spend and improve the alignment between budget allocation and true market demand.

Overall, the online A/B test in Alibaba 1688 results provide strong empirical evidence that decision-conditioned simulation is not only theoretically well-motivated but also delivers tangible business value at scale. This validates the practical relevance of CEDAR for real-world merchant-facing decision support systems and highlights its potential for broader deployment in large-scale e-commerce platforms.

## 5 Conclusion

In this work, we propose CEDAR, a decision-conditioned sales simulation framework that explicitly decouples endogenous market dynamics from latent external shocks, while treating both merchant actions and product states as first-class modalities. CEDAR consists of two core components: an Action-Interleaved Transformer that models action-conditioned state transitions through interleaved token sequences, and a Residual Correction Module that leverages external event signals to predict and correct the discrepancy between base forecasts and real-world observations. To efectively optimize these two components, we adopt a two-stage training strategy that stabilizes long-horizon rollout and enhances robustness under volatile market conditions. Extensive experiments on the large-scale E-Comm 15-Week benchmark demonstrate consistent improvements over state-of-the-art baselines, while large-scale online A/B tests on the Alibaba 1688 platform further validate the practical value of CEDAR, yielding significant gains in merchant lifetime value and store-level return on investment.

In summary, this work highlights the importance of explicit action-conditioned modeling and external shock disentanglement for reliable what-if analysis and budget planning in e-commerce systems, paving the way for more robust and decision-aware forecasting frameworks in complex, intervention-driven environments.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (Grant No. 62506348), the Natural Science Foundation ofAnhui Province (Grant No. 2508085QF211), New Generation Artificial Intelligence-National Science and Technology Major Project (Grant No. 2025ZD0122601), the CCF-1688 Yuanbao Cooperation Fund (Grant No. CCF-Alibaba2025005), the National Key R&D Program of China (Grant No. 2023YFF0725001), the National Natural Science Foundation of China (Grant No. 92370204), the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2023B1515120057), the Key-Area Special Project of Guangdong Provincial Ordinary Universities (2024ZDZX1007).

## References

[1] Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. [n. d.]. GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation. In NeurIPS Workshop on Time Series in the Age of Large Models.

[2] Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. [n. d.]. Chronos: Learning the Language of Time Series. Transactions on Machine Learning Research ([n. d.]).

[3] Salvatore Carta, Andrea Medda, Alessio Pili, Diego Reforgiato Recupero, and Roberto Saia. 2018. Forecasting e-commerce products prices by combining an autoregressive integrated moving average (ARIMA) model and google trends data. Future Internet 11, 1 (2018), 5.

[4] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. 2021. Decision Transformer: Reinforcement Learning via Sequence Modeling. arXiv:2106.01345 [cs.LG] https://arxiv.org/abs/2106.01345

[5] Robert B Cleveland, William S Cleveland, Jean E McRae, and Irma Terpenning. 1990. STL: A seasonal-trend decomposition. Journal of oficial statistics 6, 1 (1990), 3–73.

[6] Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. 2024. A decoder only foundation model for time-series forecasting. In Forty-first International Conference on Machine Learning.

[7] Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. [n. d.]. MOMENT: A Family of Open Time-series Foundation Models. ([n. d.]).

[8] Shengchao Hu, Li Shen, Ya Zhang, Yixin Chen, and Dacheng Tao. 2024. On transforming reinforcement learning with transformers: The development trajec tory. IEEE Transactions on Pattern Analysis and Machine Intelligence 46, 12 (2024), 8580–8599.

[9] Zheqi Hu, Yiwen Hu, and Hanwu Li. 2025. Multi-Task Temporal Fusion Transformer for Joint Sales and Inventory Forecasting in Amazon E-Commerce Supply Chain. arXiv preprint arXiv:2512.00370 (2025).

[10] Longyang Huang, Botao Dong, and Weidong Zhang. 2024. Eficient ofline reinforcement learning with relaxed conservatism. IEEE Transactions on Pattern Analysis and Machine Intelligence 46, 8 (2024), 5260–5272.

[11] Michael Janner, Qiyang Li, and Sergey Levine. 2021. Ofline reinforcement learning as one big sequence modeling problem. Advances in neural information processing systems 34 (2021), 1273–1286.

[12] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. 2020. Conservative q-learning for ofline reinforcement learning. Advances in neural information processing systems 33 (2020), 1179–1191.

[13] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. 2020. Ofline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643 (2020).

[14] Bryan Lim, Sercan Ö Arık, Nicolas Loef, and Tomas Pfister. 2021. Temporal fusion transformers for interpretable multi-horizon time series forecasting. International journal offorecasting 37, 4 (2021), 1748–1764.

[15] Shengsheng Lin, Weiwei Lin, Wentai Wu, Songbo Wang, and Yongxiang Wang. 2024. Petformer: Long-term time series forecasting via placeholder-enhanced transformer. IEEE Transactions on Emerging Topics in Computational Intelligence (2024).

[16] Lingfeng Liu, Yixin Song, Dazhong Shen, Bing Yin, Hao Li, Yanyong Zhang, and Chao Wang. 2026. Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 879–890.

[17] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. 2023. itransformer: Inverted transformers are efective for time series forecasting. arXiv preprint arXiv:2310.06625 (2023).

[18] Yong Liu, Guo Qin, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. [n. d.]. Timer-XL: Long-Context Transformers for Unified Time Series Forecasting. In The Thirteenth International Conference on Learning Representations.

[19] Junjie Meng, Ranxu zhang, Wei Wu, Rui Zhang, Chuan Qin, Qi Zhang, Qi Liu, Hui Xiong, and Chao Wang. 2026. Turning Semantics into Topology: LLM-Driven Attribute Augmentation for Collaborative Filtering. arXiv:2602.21099 [cs.IR] https://arxiv.org/abs/2602.21099

[20] Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. 2023. Cal-ql: Calibrated ofline rl pretraining for eficient online fine-tuning. Advances in Neural Information Processing Systems 36 (2023), 62244–62269.

[21] Brian K Nelson. 1998. Time series analysis using autoregressive integrated moving average (ARIMA) models. Academic emergency medicine 5, 7 (1998), 739–744.

[22] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2023. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. arXiv:2211.14730 [cs.LG] https://arxiv.org/abs/2211.14730

[23] José Manuel Oliveira and Patrícia Ramos. 2024. Evaluating the efectiveness of time series transformers for demand forecasting in retail. Mathematics 12, 17 (2024), 2728.

[24] Andres Potapczynski, Kin G Olivares, Malcolm Wolf, Andrew Gordon Wilson, Dmitry Efimov, and Vincent Quenneville-Belair. 2024. Efectively leveraging exogenous information across neural forecasters. (2024).

[25] Santhi Bharath Punati, Sandeep Kanta, Udaya Bhasker Cheerala, Madhusudan G Lanjewar, and Praveen Damacharla. 2025. Temporal Fusion Transformer for Multi-Horizon Probabilistic Forecasting of Weekly Retail Sales. arXiv preprint arXiv:2511.00552 (2025).

[26] Chuan Qin, Xin Chen, Chengrui Wang, Pengmin Wu, Xi Chen, Yihang Cheng, Jingyi Zhao, Meng Xiao, Xiangchao Dong, Qingqing Long, et al. 2025. Scihorizon: Benchmarking ai-for-science readiness from scientific data to large language mod els. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 5754–5765.

[27] Minghui Qiu, Feng-Lin Li, Siyu Wang, Xing Gao, Yan Chen, Weipeng Zhao, Haiqing Chen, Jun Huang, and Wei Chu. 2017. AliMe Chat: A Sequence to Sequence and Rerank based Chatbot Engine. In Proceedings ofthe 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), Regina Barzilay and Min-Yen Kan (Eds.). Association for Computational Linguistics, Vancouver, Canada, 498–503. doi:10.18653/v1/P17-2079

[28] Eva Rafetseder, Maria Schwitalla, and Josef Perner. 2013. Counterfactual reasoning: From childhood to adulthood. Journal of experimental child psychology 114, 3 (2013), 389–404.

[29] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. 2020. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International journal offorecasting 36, 3 (2020), 1181–1191.

[30] Chao Wang, Yixin Song, Jinhui Ye, Chuan Qin, Dazhong Shen, Lingfeng Liu, Xiang Wang, and Yanyong Zhang. 2026. Face: A general framework for mapping collaborative filtering embeddings into llm tokens. Advances in Neural Information Processing Systems 38 (2026), 146012–146039.

[31] Chao Wang, Hengshu Zhu, Qiming Hao, Keli Xiao, and Hui Xiong. 2021. Variable interval time sequence modeling for career trajectory prediction: Deep collaborative perspective. In Proceedings ofthe Web Conference 2021. 612–623.

[32] Chao Wang, Hengshu Zhu, Peng Wang, Chen Zhu, Xi Zhang, Enhong Chen, and Hui Xiong. 2021. Personalized and explainable employee training course recommendations: A bayesian variational approach. ACM Transactions on Information Systems (TOIS) 40, 4 (2021), 1–32.

[33] Chao Wang, Hengshu Zhu, Chen Zhu, Chuan Qin, Enhong Chen, and Hui Xiong. 2023. Setrank: A setwise bayesian approach for collaborative ranking in recommender system. ACM Transactions on Information Systems 42, 2 (2023), 1–32.

[34] Jingyuan Wang, Ze Wang, Jianfeng Li, and Junjie Wu. 2018. Multilevel wavelet decomposition network for interpretable time series analysis. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2437–2446.

[35] Qingsong Wen, Jingkun Gao, Xiaomin Song, Liang Sun, Huan Xu, and Shenghuo Zhu. 2019. RobustSTL: A robust seasonal-trend decomposition algorithm for long time series. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 5409–5416.

[36] Qingsong Wen, Kai He, Liang Sun, Yingying Zhang, Min Ke, and Huan Xu. 2021. RobustPeriod: Time-Frequency Mining for Robust Multiple Periodicity Detection. In Proceedings of the 2021 International Conference on Management of Data (SIGMOD ’21). 205–215.

[37] Qingsong Wen, Zhe Zhang, Yan Li, and Liang Sun. 2020. Fast RobustSTL: Eficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’20). 2203–2213.

[38] Bryan Wilder, Bistra Dilkina, and Milind Tambe. 2019. Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 33. 1658–1665.

[39] Malcolm Wolf, Kin G Olivares, Boris N Oreshkin, Sunny Ruan, Sitan Yang, Abhinav Katoch, Shankar Ramasubramanian, Youxin Zhang, Michael W Mahoney, Dmitry Efimov, et al. 2024. SPADE Split Peak Attention DEcomposition. In NeurIPS Workshop on Time Series in the Age of Large Models.

[40] Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven Hoi. 2022. Etsformer: Exponential smoothing transformers for time-series forecasting. arXiv preprint arXiv:2202.01381 (2022).

[41] Dongze Wu, Feng Qiu, and Yao Xie. 2025. DoFlow: Causal Generative Flows for Interventional and Counterfactual Time-Series Prediction. arXiv e-prints (2025), arXiv–2511.

[42] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. 2021. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecast ing. In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelz imer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (Eds.), Vol. 34. Curran Associates, Inc., 22419–22430. https://proceedings.neurips.cc/paper\_files/paper/ 2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf

[43] Taku Yamagata, Ahmed Khalil, and Raul Santos-Rodriguez. 2023. Q-learning decision transformer: Leveraging dynamic programming for conditional sequence modelling in ofline rl. In International Conference on Machine Learning. PMLR, 38989–39007.

[44] Jiexia Ye, Weiqi Zhang, Ke Yi, Yongzi Yu, Ziyue Li, Jia Li, and Fugee Tsung. 2024. A Survey of Time Series Foundation Models: Generalizing Time Series Representation with Large Language Model. CoRR (2024)

[45] Ranxu Zhang, Junjie Meng, Ying Sun, Ziqi Xu, Bing Yin, Hao Li, Yanyong Zhang, and Chao Wang. 2026. MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation. In Proceedings ofthe ACM Web Conference 2026 (United Arab Emirates) (WWW ’26). Association for Computing Machinery, New York, NY, USA, 6481–6492. doi:10.1145/3774904.3792495

[46] Hongling Zheng, Li Shen, Yong Luo, Deheng Ye, Shuhan Xu, Bo Du, Jialie Shen, and Dacheng Tao. 2025. Value-Guided Decision Transformer: A Uni fied Reinforcement Learning Framework for Online and Ofline Settings. In The Thirty-ninth Annual Conference on Neural Information Processing Systems. https://openreview.net/forum?id=Ogml5bxDH3

[47] Qinqing Zheng, Amy Zhang, and Aditya Grover. 2022. Online Decision Transformer. In Proceedings of the 39th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 162), Kamalika Chaudhuri, Ste fanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (Eds.). PMLR, 27042–27059. https://proceedings.mlr.press/v162/zheng22c.html

[48] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. 2021. Informer: Beyond Eficient Transformer for Long Sequence Time-Series Forecasting. Proceedings of the AAAI Conference on Artificial Intelligence 35, 12 (May 2021), 11106–11115. doi:10.1609/aaai.v35i12.17325

[49] Shiqiao Zhou, Holger Schöner, Huanbo Lyu, Edouard Fouché, and Shuo Wang. 2025. BALM-TSF: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management. 4498–4508.

[50] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. 2022. FEDformer: Frequency Enhanced Decomposed Transformer for Long term Series Forecasting. In Proceedings ofthe 39th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 162), Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (Eds.). PMLR, 27268–27286. https://proceedings.mlr.press/v162/zhou22g. html

## A Details of LLM-based Hotspot Information Extraction

In this section, we detail the implementation of the Hotspot Information Extraction module. Raw social media trends are inherently noisy and often dominated by entertainment gossip, which can mislead forecasting models if used directly. To ensure high-quality semantic representations, we design a two-stage LLM prompting pipeline. The first stage filters noise and extracts commercerelevant tags, while the second stage synthesizes these tags with calendar events into a single, coherent natural language sentence.

## A.1 Stage 1: Noise Filtering and Tag Extraction

For each forecasting window �, we first collect raw trending lists $\mathcal { T } _ { t }$ from major social platforms (eg.Red notebook and Douyin ) over the past week ([�−7, �−1]). In this stage, the LLM acts as an information filter. We prompt it to discard irrelevant news and extract only keywords (tags) indicative of potential product demand.

## Prompt 1: Commerce Tag Extraction

Role: You are an e-commerce data analyst.   
Task: Extract product-related keywords from the following noisy social media trending list.   
Input:

• Raw Trends: {Raw\_Trend\_List}

Instructions: 1. Ignore all entertainment gossip, celebrity news, and non-commercial events. 2. Identify trends that could trigger consumer purchases (e.g., sudden weather changes, viral outdoor activities). 3. Extract concise product tags or categories related to these trends.

Output Format: A comma-separated list of tags (e.g., "camping tents, down jackets, traditional gifts").

## A.2 Stage 2: Semantic Synthesis with Calendar Events

Simply embedding a list of isolated tags $\mathcal { K } _ { t }$ fails to capture the underlying market narrative and temporal context. Therefore, in the second stage, we fuse $\mathcal { K } _ { t }$ with a structured list of upcoming calendar events $\mathcal { E } _ { t }$ (e.g., public holidays, shopping festivals) occurring within the target window �. The LLM is instructed to synthesize these elements into semantically complete sentences $S _ { t } .$

## Prompt 2: Narrative Synthesis

Role: You are a senior market strategist. Task: Synthesize the extracted product tags and upcoming holidays into a concise market summary. Inputs:

• Extracted Product Tags: {K<sub>�</sub>}

• Upcoming Holidays/Events: $\{ \mathcal { E } _ { t } \}$

Instructions: 1. First talks about the upcoming event and holidays. 2. Combine all information into semantically complete sentences that describes the overarching market demand drivers for the next week.

Constraint: You must output exactly follow instruction.   
Do not use bullet points or lists.

Output Example: "Driven by the upcoming Mid-Autumn Festival and the recent viral outdoor trend, the current market demand is highly concentrated on traditional gift boxes and premium camping gear."

## A.3 Context Embedding Generation

The output of Stage 2 ars fluent sentences $S _ { t }$ that encapsulates the macro-environmental factors and market dynamics. Finally, we employ a pre-trained text encoder to transform $S _ { t }$ into a dense vector representation:

$$
\mathbf { h } _ { t } = \operatorname { E n c o d e r } ( S _ { t } ) \in \mathbb { R } ^ { d _ { h } }
$$

where $d _ { h }$ is the embedding dimension. This dense hotspot embedding $\mathbf { h } _ { t }$ serves as a critical global exogenous context signal for our framework, empowering the model to adjust its predictions based on a semantic understanding of concurrent external drivers.

## B Causal Identifiability and Structural Invariance Analysis of CEDAR

From the perspective of Structural Causal Models (SCM), the ecommerce sales system is formalized as a controlled stochastic process. We posit that the underlying data generation process follows an Additive Noise Model defined as:

$$
S _ { t + 1 } : = f _ { \theta } ( S _ { \leq t } , A _ { t + 1 } ) + \mathcal { E } ( Z _ { t + 1 } , U _ { t } )\tag{9}
$$

where � denotes endogenous states, � represents merchant interventions, � represents observed exogenous events, and � denotes unobserved noise.

The CEDAR framework leverages this structural equation to achieve an efective approximation of the interventional distribution $P ( S _ { t + 1 } \mid d o ( A _ { t + 1 } ) , S _ { \leq t } )$ through architectural causal disentanglement. Specifically: (1) Learning Action-Conditioned Transition Operators: The AIT module in the first stage leverages an interleaved sequence structure to explicitly model the endogenous mechanism $f _ { \theta } .$ . This approximation relies on the assumption of Sequential Ignorability, which posits that potential outcomes �(�) are conditionally independent of the current action given the history ${ \mathcal { H } } _ { t } = \{ S _ { \leq t } , Z _ { \leq t + 1 } \} :$

Table 5: Long-horizon rollout performance on Alibaba 1688 measured by MSE. All settings use 10 weeks of historical observations as input.
<table><tr><td>Horizon</td><td>CEDAR</td><td>PatchTST</td><td>PETFormer</td></tr><tr><td>next 5</td><td>0.182</td><td>0.424</td><td>0.434</td></tr><tr><td>next 10</td><td>0.414</td><td>0.849</td><td>0.743</td></tr><tr><td>next 15</td><td>0.734</td><td>1.526</td><td>1.382</td></tr><tr><td>next 20</td><td>1.103</td><td>2.173</td><td>1.945</td></tr><tr><td>next 25</td><td>1.612</td><td>2.847</td><td>2.561</td></tr></table>

Table 4: Wall-clock cost analysis of CEDAR and representative baselines. The LLM-based event embedding generation is a one-time ofline preprocessing cost.
<table><tr><td>Component</td><td>Time / Epoch</td><td>Epochs</td><td>Total Time</td><td>Notes</td></tr><tr><td>CEDAR Stage I</td><td>≈ 11 min</td><td>14</td><td>154 min (2h 34m)</td><td>Base-stage training</td></tr><tr><td>CEDAR Stage II</td><td>≈ 16 min</td><td>5</td><td>80 min (1h 20m)</td><td>Residual correction training</td></tr><tr><td>Event embedding generation</td><td></td><td>一</td><td>≈ 4h</td><td>One-time offline cost</td></tr><tr><td>PatchTST</td><td>≈ 7 min</td><td>17</td><td>119 min (1h 59m)</td><td>Baseline reference</td></tr><tr><td>PETFormer</td><td>≈ 21 min</td><td>31</td><td>651 min (10h 51m)</td><td>Baseline reference</td></tr></table>

$$
S _ { t + 1 } ( a ) \perp \perp A _ { t + 1 } \mid \mathcal { H } _ { t } , \quad \forall a \in \mathcal { A }\tag{10}
$$

(2) Orthogonal Decomposition of Exogenous Shocks: The residual correction module in the second stage explicitly captures the exogenous term E. By incorporating LLM-enhanced event semantics $Z ,$ the model efectively blocks back-door paths induced by latent confounders such as market trends.

This separated modeling of Endogenous Mechanism + Exogenous Perturbation fundamentally ensures the Structural Invariance of the model during Counterfactual Simulation. It enables robust estimation of the marginal causal efects derived from hypothetical budget strategies, thereby overcoming the confounding bias inherent in traditional correlation-based models.

## C Time Cost Analysis

The wall-clock cost analysis is reported in Table 4. Compared with standard forecasting baselines, CEDAR introduces additional computation mainly from the two-stage training pipeline and the LLMbased event embedding construction. Specifically, Stage I takes about 154 minutes to train, while Stage II adds another 80 minutes for learning the residual correction module. The total training time of CEDAR is therefore approximately 234 minutes, which is higher than PatchTST but substantially lower than PETFormer in our implementation. The event embedding generation requires around 4 hours; however, this step is performed only once as ofline preprocessing and the resulting embeddings can be reused across downstream training and inference. Therefore, its cost is amortized over all products, time windows, and subsequent model updates. Considering the consistent improvements in both short-horizon and long-horizon forecasting accuracy, the additional computational overhead is acceptable for large-scale industrial deployment, especially in budget planning scenarios where simulation quality is more critical than one-time preprocessing cost.

## D Long-horizon Performance Comparison

To further examine rollout stability, we extend the evaluation horizon from next-5 and next-10 to next-15, next-20, and next-25, using the same 10-week historical input. As shown in Table 5, all models exhibit larger errors as the horizon increases, which is expected due to autoregressive error accumulation. Nevertheless, CEDAR degrades more gracefully than the strongest baselines. At next-25, CEDAR obtains an MSE of 1.612, substantially lower than PatchTST 2.847 and PETFormer 2.561. This result suggests that explicitly modeling action-conditioned transitions and event-driven residuals improves not only short-horizon accuracy but also the fidelity of long-horizon counterfactual rollouts.