# Coordinated Lane-Level Variable Speed Limits and Ramp Metering for Successive Weaving Segments Considering Merging/Diverging Risks: A Hybrid Model Predictive Control and Multi-Agent Reinforcement Learning Approach

Guodong Ma<sup>a</sup>, Baofeng Sun<sup>a,</sup>\*, Wenyu Yang<sup>a</sup>, Zhihong Yao<sup>b</sup>

a. School of Transportation, Jilin University, Changchun 130022, China;

b. School of Transportation and Logistics, Southwest Jiaotong University, Chengdu 610031, China.

## Abstract

Successive weaving segments (SWSs) on urban expressways present critical bottlenecks prone to frequent congestion and collisions, demanding fine-grained active traffic management (ATM). However, existing strategies struggle to balance high performance from data-driven adaptive optimization with high resilience and transferability from model-driven rigid protection. To address this, we propose a hybrid model- and data-driven ATM framework that coordinates SWSs. First, we reconstruct a lane-leve macroscopic traffic flow model, L-METANET, to capture free and forced lane-changing behaviors. Second, we combine XGBoost-SHAP with random parameters binary logit (RPBL) to fit analytical equations for merging/diverging collision risks, thereby formulating the system cost and reward functions. Finally, we design a hierarchical framework that coordinates lane-level variable speed limits and ramp metering, MPC-STMAPPO, integrating model predictive control (MPC) and multi-agent reinforcement learning (MARL). The upper MPC layer utilizes L-METANET for long-horizon rolling optimization to output baseline commands, while the lower layer employs an ST-MAPPO algorithm incorporating Mamba cells and a graph attention to generate residual actions for short-horizon agile fine-tuning. Real-world experiments on the 18-km Eastern Expressway in Changchun, China, demonstrate that: (i) L-METANET accurately reproduces lane-changing-induced flow redistribution and capacity drops, aligning state evolutions with the ground truth; (ii) XGBoost-SHAP-RPBL model achieves AUC values above 0.80 for most tasks, outperforming traditional logit models; and (iii) Compared to multiple MPC- and MARL-based baselines, MPC-STMAPPO delivers superior training convergence speed and excels across various metrics. Furthermore, in transfer tests under various randomly fluctuating demands, MPC-STMAPPO significantly outperforms pure MARL in generalization capability, revealing high industrial deployment value.

Keywords: Active Traffic Management, Successive Weaving Segments, Model Predictive Control, Multi-Agent Reinforcement Learning, Lane-Level Variable Speed Limits, Ramp Metering

## 1. Introduction

Urban expressways serve as the backbone of modern urban transportation networks, accommodating ever-increasing traffic demands. However, their operational efficiency and safety often suffer at critical bottleneck nodes, with weaving segments representing one of the most typical problematic zones (Gao et al., 2025; Ouyang et al., 2023; Rim et al., 2023; Zhao et al., 2021). As hubs connecting the mainline with on- and off-ramps, weaving segments facilitate a complex “three-stream weaving” phenomenon, where mainline through traffic, on-ramp merging flows, and off-ramp diverging flows interact at high frequencies (Chen and Ahn, 2018; Golob et al., 2004; Yuan et al., 2024). The resulting forced lane-changing maneuvers easily induce turbulence, causing a sharp drop in traffic capacity accompanied by extremely high collision risks. Fortunately, the advancement of intelligent transportation systems provides new opportunities to alleviate this persistent issue through Active Traffic Management (ATM) based on vehicle-road-cloud integration. Especially with the increasing penetration rate of connected and automated vehicles (CAVs), utilizing precise perception and control capabilities to actively intervene in weaving segments has become a critical means to enhance road network resilience.

Among various ATM strategies, researchers widely recognize ramp metering (RM) and variable speed limits (VSL) as effective tools for mitigating congestion (Ma et al., 2021; Zhang et al., 2025). In this study, ATM specifically refers to either an individual strategy or their combination. Early literature predominantly focused on isolated RM (Airaldi et al., 2025; Peng and Xu, 2023; Zhang et al., 2024a) or standalone VSL control (Jin et al., 2024; Yang et al., 2026; Zhang et al., 2024c). However, because extensive evidence validates the efficacy of their joint operation in alleviating freeway and expressway bottleneck congestion, recent studies increasingly evaluate their coordinated control effects (Han et al., 2025; He et al., 2024; Qiu et al., 2025; Zhang et al., 2025). Most existing coordinated control studies propose segmentlevel ATM strategies that assume lane homogeneity and apply uniform, coarse-grained control commands across all lanes. This approach works well for basic mainline segments because rare lateral lanechanging maneuvers allow researchers to neglect lane heterogeneity. However, weaving segments severely challenge this assumption. Concurrent forced and free lane-changing behaviors driven by merging and diverging demands impose significant lane-level heterogeneity on weaving traffic. Specifically, early lane-changing friction from exiting vehicles and incoming on-ramp traffic frequently plunges the outer lanes into low-speed congestion, whereas the inner lanes often remain completely smooth. Such a one-size-fits-all segment-level approach lacks differentiated management, thereby wasting inner-lane capacity and failing to guarantee outer-lane safety. Consequently, within a connected and automated environment, we must shift the control granularity from the segment level down to the lane level. Implementing fine-grained lane-level coordinated ATM (Lu et al., 2023; Lu et al., 2024) provides an inevitable solution to counter the lane heterogeneity inherent in weaving segments.

Beyond control granularity, the design of optimization objectives also demands refinement. Existing coordinated control strategies predominantly prioritize traffic efficiency as their primary or sole goal, often incorporating safety considerations insufficiently or unscientifically. Traffic safety in weaving segments warrants dedicated attention due to its unique spatiotemporal evolution characteristics. Specifically, while merging collision risks persist at the entrance, aggressive lane-crossing maneuvers by vehicles exiting the mainline introduce a distinct diverging risk at the exit. Furthermore, the heterogeneous underlying triggers of merging and diverging risks necessitate targeted attention. Neglecting these potential weaving-induced risks or treating them simplistically can easily result in overly aggressive control commands, thereby inducing severe safety hazards.

Translating these complex, lane-level, and multi-objective requirements into real-time control algorithms poses a dual challenge for solution methodologies. Model-driven algorithms, particularly Model

Predictive Control (MPC), find widespread application due to their capacity to manage physical constraints and multi-objective optimization (Chen et al., 2025; Zhang et al., 2024b); however, their performance relies heavily on prediction model accuracy. Within weaving segments, highly stochastic and nonlinear lateral lane-changing behaviors challenge conventional macroscopic traffic flow models like METANET or the cell transmission model, as failing to capture these dynamics accurately induces severe model mismatch (Chen et al., 2021). Conversely, while data-driven methods, notably deep reinforcement learning (DRL), excel at capturing environmental nonlinearities and uncertainties (Afifah and Guo, 2025; Jin et al., 2025; Kang et al., 2024), they suffer from slow training convergence, action oscillations, and a lack of safety boundary guarantees. For safety-critical weaving scenarios, a single approach can rarely satisfy both control stability and high performance. Consequently, developing a hybrid solution framework that merges the complementary advantages of model-and-data-driven approaches represents a critical pathway to breaking current technical bottlenecks (Airaldi et al., 2025; Sun et al., 2024).

To address these research challenges, we propose a lane-level coordinated VSL and RM strategy for weaving segments that explicitly accounts for both merging and diverging collision risks. First, we develop an improved lane-level macroscopic traffic flow model, designated as L-METANET, which explicitly captures the lateral friction and weaving resistance induced by forced lane-changing maneuvers, thereby overcoming the inherent limitations of conventional models in depicting lane heterogeneity. Second, we construct an analytical risk prediction model, XGBoost-SHAP-RPBL, driven dually by controllability and interpretive accuracy, and integrate it into a multi-objective optimization function to achieve proactive safety control. Building upon these components, we design a hybrid model-and-datadriven hierarchical coordinated control framework. Within this framework, the upper layer leverages MPC to calculate baseline LVSL value and RM rates under strict physical constraints, securing system control resilience and guiding the lower-layer learning process. Concurrently, the lower layer utilizes multi-agent reinforcement learning (MARL) to execute residual compensation on these reference policies, ensuring high system performance. While balancing control safety and traffic efficiency, this framework significantly enhances system adaptability to complex traffic turbulence, providing rigorous theoretical guidance and technical support for ATM in expressway weaving segments.

## 2. Literature review and main contributions

This section reviews ATM literature across modeling methodologies and control algorithms. Section 2.1 evaluates the transition to lane-level coordinated VSL and RM strategies for lane heterogeneity, alongside control formulations and closed-loop safety. Section 2.2 contrasts model-driven and datadriven control architectures, delineating their respective limitations to motivate a hybrid alternative. Finally, while Section 2.3 synthesizes research gaps to establish the study’s motivations and contributions.

## 2.1. The models of coordinated variable speed limit and ramp metering

As core instruments within ATM frameworks, RM and VSL offer highly effective coordinated approaches to mitigate traffic congestion and collisions. Specifically, RM prevents mainline breakdown by regulating on-ramp inflow rates, whereas VSL smooths mainline speeds to reduce speed differentials between mainline and ramp vehicles, simultaneously throttling upstream traffic volumes entering bottleneck zones to alleviate localized congestion and collision risks. Early engineering practices and theoretical explorations typically decoupled these two strategies into independent subsystems. However, accelerating urbanization and mounting traffic loads have exposed the limitations of this decoupled approach, which often traps the system in local optima as traffic densities approach critical thresholds. To overcome this, the academic community has progressively established an integrated VSL and RM coordinated control paradigm. The core mechanism of this paradigm relies on VSL to preemptively generate low-density spatiotemporal windows upstream, creating sufficient merging space for the ramp traffic released by downstream RM; this coordination significantly suppresses the backward propagation of traffic shockwaves and yields system-wide global optimality. For instance, van de Weg et al. (van de Weg et al., 2019) applied coordinated VSL and RM on a two-lane freeway with two on-ramps and offramps, demonstrating superior throughput performance. Similarly, Ma et al. (Ma et al., 2021) confirmed that integrated VSL-RM control outperforms standalone VSL or RM strategies in reducing freeway collision risks and traffic conflicts.

Early RM and VSL strategies relied heavily on model-driven approaches, which necessitate a critical prerequisite: constructing a predictive macroscopic traffic flow model that accurately captures traffic evolution dynamics, enabling controllers to anticipate future traffic states and optimize control actions accordingly. Within the existing research framework, the cell transmission model (CTM) (Daganzo, 1994) and the METANET model (Messmer and Papageorgiou, 1990) represent the two primary model choices. CTM utilizes microscopic or mesoscopic rules to simulate vehicle lane-changing and car-following behaviors across discrete grids, effectively capturing the nonlinear characteristics of traffic flow. However, researchers have noted that such models perform poorly in characterizing capacity drops and stop-andgo traffic waves at freeway bottlenecks (Spiliopoulou et al., 2014). In contrast, the second-order macroscopic METANET model, rooted in fluid dynamics, provides an analytical differential equation form that explicitly describes the spatiotemporal evolution of flow, density, and speed; its low computational complexity establishes it as a preferred predictive tool (Chen et al., 2021; Spiliopoulou et al., 2014).

Notably, despite its advantages, the original METANET model operates on a lane homogeneity assumption and lacks fine-grained lane differentiation (Messmer and Papageorgiou, 1990), meaning that researchers can only apply it to execute segment-level VSL. This uniform management may suffice for conventional mainline segments devoid of frequent ramp operations. However, weaving segments strictly demand fine-grained, lane-level VSL to simultaneously balance traffic efficiency and driving safety, which severely limits the applicability of segment-level models to these areas. This limitation arises because forced lane-changing maneuvers within the weaving section occur concurrently with free lane-changing behaviors upstream (Arman and Tampere, 2022; Wang et al., 2025; Zhou et al., 2024), imposing profound lane-level heterogeneity. Specifically, outer lanes accommodate accelerating merging flows from on-ramps and decelerating diverging flows toward off-ramps, frequently triggering lowspeed turbulence, whereas inner lanes predominantly carry high-speed through traffic. Applying uniform control while ignoring such pronounced cross-lane variations inevitably wastes inner-lane capacity or compromises outer-lane safety regulation, significantly degrading coordinated control efficacy. Consequently, recent pioneering studies have begun exploring lane-level coordinated VSL and RM strategies. Researchers generally pursue this objective via two distinct methodological pathways, namely lane-level macroscopic traffic flow prediction models (Chen et al., 2021; Chen et al., 2025; Sai, 2025) and model-free DRL (Lu et al., 2023).

Although lane-level ATM research has recently emerged, existing studies predominantly focus on basic mainline segments or merging zones, leaving weaving segments largely unexplored. Merging zones only involve interactions between two traffic streams, specifically mainline through traffic and incoming ramp traffic, which concentrates conflict points within a limited area. Conversely, weaving segments exhibit a complex "three-stream weaving" structure that integrates mainline through, on-ramp merging, and off-ramp diverging flows, triggering unique “X-shaped” lane-changing conflicts and traffic dynamics fundamentally distinct from merging zones. Consequently, researchers cannot directly transfer control models and algorithms developed for merging areas to weaving segments, necessitating further investigation into fine-grained lane-level management for these complex sections.

The design of control objectives and constraints is equally crucial. Existing literature often prioritizes traffic efficiency over driving safety, a bias that severely constrains the performance of ATM systems under extreme operating conditions. Conventional control strategies typically optimize for minimizing total travel time or maximizing total throughput efficiency; however, this efficiency-centric orientation inadequately accounts for safety, necessitating a more comprehensive control strategy to address pressing traffic safety and efficiency challenges (Zhang et al., 2025). Crucially, the core bottleneck preventing risk integration is not a disregard for safety, but rather the difficulty of constructing a mathematical model that describes the safety characteristics of macroscopic traffic flow in real time, which demands three essential attributes: (1) Accuracy: it must precisely map the relationship between macroscopic aggregated traffic flow indicators and microscopic crash risks; (2) Controllability: the risk metrics must rely on parameters directly or indirectly manageable via VSL and RM, enabling the controller to purposely mitigate the targeted risks; (3) Observability: the input parameters must be rapidly collectable via existing detectors, such as inductive loops, to satisfy real-time computational efficiency requirements; (4) Heterogeneity: Due to variations in traffic demand, geometric length, and channelization design, the underlying macroscopic causal mechanisms of collision risks differ inherently across distinct weaving segments. Consequently, it is essential to develop customized, site-specific risk assessment models tailored to individual weaving locations.

The integration of risk perception into macroscopic ATM is fundamentally impeded by two distinct dimensions, specifically regarding the observation level and the modeling methodology. At the observation level, a significant disconnect persists between microscopic risk dynamics and macroscopic control capabilities. While classic microscopic safety indicators, such as time-to-collision (TTC) and deceleration rate to avoid a crash (DRAC), offer high fidelity in capturing risk, collecting these metrics across an entire network in real time remains unfeasible. Furthermore, macroscopic control methods, including RM and VSL, cannot directly regulate vehicle-level headways or decelerations, causing a severe decoupling between control variables and microscopic optimization objectives. Conversely, traditional macroscopic risk assessment models, despite exhibiting favorable controllability and observability, fail to resolve this bottleneck because they lack the precision required to characterize microscopic collision dynamics and fail to account for the causal heterogeneity across distinct weaving segments. At the modeling methodology level, recent artificial intelligence research has introduced various real-time crash risk evaluation models leveraging machine learning and deep learning techniques (Cheng et al., 2022; Wang et al., 2024; Yang et al., 2021; Zhang et al., 2025). Although these data-driven approaches excel at quantifying real-time collision risks within intelligent transportation systems (Zhang et al., 2025), they introduce a new algorithmic hurdle. Specifically, end-to-end deep learning methods lack analytical, closedform expressions, making them exceptionally difficult to embed within optimization-based control loops for proactive management. Consequently, resolving these concurrent limitations across both observation levels and modeling methodologies remains an unresolved challenge that warrants further investigation.

## 2.2. The algorithms of coordinated variable speed limit and ramp metering

Translating lane-level, multi-objective ATM into deployable real-time control algorithms poses challenges, particularly in managing the stochastic uncertainties of traffic environments and balancing offline training costs against online computational efficiency. To address these twin challenges, existing control methodologies primarily fall into either model-driven approaches exemplified by MPC (Chen et al., 2025; Han et al., 2021; Mao et al., 2022; Othman et al., 2022; Sirmatel and Yildirimoglu, 2023; Zhang et al., 2024b) or data-driven methods spearheaded by DRL (Afifah and Guo, 2025; Greguric et al., 2022; Jin et al., 2025; Kang et al., 2024; Li and Lasenby, 2024; Wu et al., 2020). Crucially, these two paradigms exhibit highly complementary characteristics.

Regarding tackling environmental uncertainties, MPC and DRL offer contrasting paradigms. MPC has long dominated ATM control due to its capacity to explicitly handle multivariable physical constraints. Its core mechanism leverages a macroscopic traffic flow model to construct a prediction horizon, calculating the optimal control law for the current time step via rolling horizon optimization. Consequently, MPC provides strong physical interpretability and strictly prevents system states from violating safety and physical hard constraints. For instance, Mao et al. (Mao et al., 2022) developed a model-based extended VSL controller integrating an extended CTM with variable-length cells and an MPC scheme optimized via an improved genetic algorithm; this approach reduced total travel time by 14.57% compared to conventional VSL without variable control zones. Because MPC performance heavily depends on prediction model accuracy, researchers continuously strive to develop and calibrate more precise high-order models. These advancements include: (1) incorporating VSL and/or RM rates (Chavoshi et al., 2023; Frejo et al., 2019; Hegyi et al., 2005; Yu and Abdel-Aty, 2014); (2) incorporating mixed traffic flow dynamics (Rahmanidehkordi and Ghasemi, 2024); (3) accounting for lane-changing disruptions (Cheng et al., 2025); and (4) addressing lane-level heterogeneity as detailed in Section 2.1 (see the comprehensive review of Wang et al. (Wang et al., 2022)). These refinements remain vital to elevating predictive accuracy and, consequently, boosting MPC efficacy. Nevertheless, such macroscopic models remain idealized abstractions of reality that struggle to capture all stochastic disturbances inherent in realworld traffic. Because real-world traffic evolution rarely adheres strictly to deterministic differential equations, inevitable model mismatches cause deviations between open-loop predictions and closedloop executions, hindering closed-loop systems from achieving global optimality. Thus, while model refinement is important, perfectly replicating real-world traffic laws within imperfect environments remains inherently difficult (Li and Lasenby, 2024). To counteract control degradation driven by model bias, researchers have increasingly turned toward model-free DRL methods. DRL approximates stateaction mappings through iterative trial-and-error interactions with simulation environments, leveraging the potent nonlinear approximation capabilities of deep neural networks to manage highly nonlinear and stochastic dynamics. Yet, despite demonstrating superior adaptability to environmental uncertainties over traditional control methods, these pure data-driven approaches face some skepticism regarding practical engineering deployment. First, they often lack safety guarantees; unlike MPC, DRL struggles to explicitly embed physical constraints into its optimization process, meaning its exploration-driven learning mechanism can output hazardous commands, which remains unacceptable in safety-critical traffic systems. Second, interpretability remains a major challenge because neural-network-based blackbox policies lack physical transparency, hindering trust from traffic authorities and complicating liability attribution after accidents. Finally, poor generalization limits transferability, as RL policies easily overfit their training environments. Minor variations in network topology or traffic flow characteristics often necessitate retraining, stripping the algorithm of the flexible transfer capabilities inherent in MPC.

Beyond their distinct approaches to environmental uncertainty, model-driven and data-driven algorithms exhibit highly complementary computational paradigms, particularly regarding training costs and online computational real-time performance (Sun et al., 2024). The primary advantage of MPC lies in its elimination of offline pre-training. However, this advantage incurs a heavy online computational burden, as the controller treats each optimization step as an entirely new problem, even when encountering identical traffic scenarios previously. Because macroscopic traffic flow models exhibit severe nonlinearities, the dimension of decision variables scales exponentially within multi-lane, long-horizon prediction scenarios, making single-step optimization problems highly time-consuming to solve. Conversely, RL utilizes an offline-training-and-online-inference paradigm. Once trained, the policy requires only a single forward propagation during online deployment, enabling millisecond-level execution that perfectly accommodates the high-frequency demands of real-time control. Nevertheless, RL shifts the computational burden entirely to the offline training stage. Under complex multi-objective and multiagent configurations, the models frequently suffer from severe convergence bottlenecks. This problem becomes particularly acute in lane-level weaving control scenarios, where high-dimensional state-action spaces combine with environmental non-stationarity. These challenges render MARL highly susceptible to local optima or non-convergent oscillations, making the training process excessively time-consuming and structurally unstable.

In summary, model-driven MPC and data-driven DRL complement each other across multiple operational dimensions. (1) For uncertainty, MPC provides structural robustness but depends heavily on model accuracy, whereas RL offers flexibility but lacks reliable safety boundaries. (2) For computational efficiency, MPC suffers from heavy online computational burdens despite eliminating pre-training requirements, while RL enables rapid online inference at the expense of highly challenging offline training. (3) For transferability, MPC generalizes to any scenario while maintaining high performance given an accurate prediction model, though such perfect models remain elusive; conversely, RL struggles with transfer applications but frequently outperforms MPC within familiar environments. Consequently, isolated methodologies can no longer satisfy the stringent demands for high precision, rigorous safety, and high real-time execution in weaving segment management. Future research paradigms thus favor a deep integration of both approaches (Airaldi et al., 2025; Sun et al., 2024), which yields a hybrid control architecture. This framework leverages MPC to establish physics-based baseline control that satisfies safety constraints, thereby guaranteeing baseline system stability and interpretability. Concurrently, it employs RL to execute real-time data-driven residual compensation on the baseline policy, effectively mitigating model mismatches while elevating online computational efficiency. This integrated philosophy facilitates a profound paradigm shift from passive adaptation to proactive learning.

## 2.3. Research gaps and main contributions

Although extensive literature establishes a solid theoretical foundation, several critical technical gaps persist when addressing the high-risk, high-disturbance scenarios inherent to weaving segments:

(1) Mismatches in modeling granularity fail to capture pronounced lane-level heterogeneity. Existing coordinated control studies rely heavily on segment-level METANET models. These frameworks cannot accurately characterize the intense lane heterogeneity arising from forced lane-changing within the weaving section and free lane-changing conflicts upstream, which inevitably induces severe model mismatch.

(2) Decoupled risk perception and closed-loop control hinder effective collision risk mitigation. Although classic microscopic risk metrics, including TTC, THW, and DRAC, can precisely quantify risk dynamics, they lack real-time availability, direct controllability, and observability. Conversely, macroscopic risk assessment models fail to establish clear relationships with collisions and struggle to capture the causal heterogeneity of risk across different weaving segments. Consequently, these limitations impede the construction of a complete closed-loop controller that bridges risk perception and control execution.

(3) Isolated model-driven or data-driven architectures show inadequate adaptability to uncertainties, transfer deployment, and real-time computation. Model-driven MPC suffers from model mismatch and heavy online computational burdens, failing to flexibly respond to stochastic disturbances and high-frequency execution demands. Conversely, data-driven DRL struggles with slow training convergence and poor generalization in transfer scenarios, which compromises training efficiency and fails to safeguard the strict physical performance lower bound of the system.

To bridge these technical gaps, our study delivers the following main contributions:

(1) We develop a multi-lane macroscopic traffic flow model, L-METANET, tailored for weaving segments by incorporating free and forced lane-changing behavior. L-METANET explicitly reproduces lane-changing-driven cross-lane flow redistribution and capacity drops, serving as a high-fidelity predictive model for fine-grained lane-level ATM.

(2) We developed a multi-objective coordinated LVSL and RM control strategy that explicitly accounts for merging and diverging collision risks within weaving segments. By integrating XGBoost-SHAP feature selection with a random parameters binary logit model, we designed an analytical risk assessment model to ensure direct controllability and real-time data availability while effectively capturing the heterogeneity of macroscopic risk precursors across different weaving segments. This closed-form risk formulation is directly embedded into the MPC cost function and MARL reward function.

(3) We design a hybrid model-and-data-driven hierarchical coordinated control ATM framework. The upper layer implements MPC alongside L-METANET to compute baseline control commands under strict physical boundaries, guaranteeing system stability and guiding lower-layer training. Concurrently, the lower layer deploys a spatiotemporal attention-reinforced MAPPO algorithm, embedding Mamba blocks and graph attention mechanisms into the actor and value networks to capture long-term temporal dependencies and spatial agent interactions. By incorporating the upper-layer MPC control outputs as environmental states and reference control commands, the lower layer executes residual compensation over a short control horizon, ensuring high learning efficiency and robust disturbance rejection capabilities.

The remainder of this paper is organized as follows. Section 3 outlines the problem formulation and the overall research framework. Section 4 presents the lane-level macroscopic traffic flow model and the analytical risk assessment model tailored for expressway weaving segments. Section 5 constructs the MPC and MARL-based hierarchical coordinated control framework integrating lane-level VSL and RM. Section 6 conducts extensive simulation experiments to validate the performance of the proposed macroscopic traffic flow model, macroscopic risk assessment model for weaving segments, and hierarchical coordinated ATM framework. Finally, Section 0 concludes the paper.

![](images/dad19411fda469b16fb29a1786d28e7e28e35b37e5c69f02a64d96dc6112a961.jpg)  
Fig. 1 Scenario definition and problem formulation

## 3. Problem formulation and research framework

## 3.1. Scenario definition and problem description

This study addresses the macroscopic coordinated control problem across continuous multi-weaving segments on urban expressways (Fig. 1). As illustrated in Fig. 1(a), each core control unit is partitioned longitudinally into four consecutive zones: an upstream LVSL zone, an acceleration zone, a weaving bottleneck segment, and a downstream off-ramp diverging segment. The infrastructure architecture comprises two primary layers: (1) Perception layer: Densely deployed detectors gather real-time, lanespecific macroscopic traffic states, boundary demands at origin cross-sections and on-ramps, and queue lengths at on-ramps; (2) Control layer: A central controller manages integrated mainline LVSL and onramp RM strategies across fine-grained lane-segment units. Control commands are dispatched via vehicle-to-everything (V2X) communications to connected and automated vehicles (CAVs), and via overhead dynamic speed displays and conventional ramp signals to human-driven vehicles (HDVs).

Under a mixed traffic environment (Fig. 1(b)), executing this coordinated control framework entails three multidimensional challenges: (1) Competing control objectives: maximizing throughput efficiency (maintaining high speeds and minimal queues) inherently conflicts with mitigating traffic safety risks. Intensive lateral cutting-in maneuvers within weaving sections severely compress safety gaps, accelerating collision risks. Balancing traffic efficiency against merging/diverging risks remains a primary trade-off; (2) Cascading congestion propagation: due to the dense spacing of urban expressway ramps, downstream bottleneck breakdowns generate backward-propagating stop-and-go traffic waves. These waves trigger cascading failures across successive upstream weaving segments, rendering conventional, isolated bottleneck management sub-optimal; (3) Spatiotemporal variable coupling: RM physically restricts inflows to protect the mainline but risks urban surface-street spillback, whereas VSL regulates mainline speeds to proactively create upstream low-density merging windows and smooth longitudinal speed differentials. Because these control variables are highly coupled, a lack of joint, fine-grained design triggers excessive mainline delays or ramp deadlocks.

To address these coordinated challenges, the control variables, optimization objectives, and constraints of this study are synthesized into the following core elements (as illustrated in Fig. 1(b)):

(1) Control variables: For any given control time step ( is MARL time step) and weaving segment $j \left( j \in \{ 1 , 2 , . . . , J \} \right)$ , the joint control action sequence of the system comprises two primary components: LVSL value $\nu _ { j } ^ { n } ( t )$ : The control commands applied to lane n within the upstream LVSL zone of weaving segment  . RM rate $r _ { j } ( t )$ : The regulation rate command applied to the on-ramp signaling lights of weaving segment j .

(2) Optimization objectives: The central controller pursues multi-objective Pareto optimality to simultaneously optimize system-level efficiency and safety performance: Maximizing system efficiency: This objective seeks to maximize the average mainline travel speed while minimizing the total vehicle queueing time at the on-ramps; Minimizing collision risks: This objective aims to suppress macroscopic merging and diverging risks within the weaving sections, thereby actively reshaping a safer traffic fluid state.

(3) Constraints: The solution of the control sequence must strictly comply with rigorous physical and safety boundaries: 1)Traffic physics constraints: The spatiotemporal evolution of traffic states must strictly adhere to the L-METANET equations governing the conservation of mass and momentum relaxation; 2)Safety boundary constraints: These incorporate the maximum allowable queue length capacity at individual on-ramps alongside spatial variation bounds for both VSL and ramp signaling across adjacent lanes or segments; 3)Actuator physical limits: Restricted by traffic regulations and mechanical characteristics, these impose maximum rateof-change thresholds as well as absolute upper and lower bounds for both VSL and RM rates.

![](images/04d06b45eefc421436ece70eeb3c974c0148ebc45ffdb9894c7de65970ac78c0.jpg)  
Fig. 2 Overall framework of this study

## 3.2. Overall research framework

To address the aforementioned challenges in successive weaving segments coordinated control, the ATM framework decouples the control logic into distinct, interconnected operational layers spanning data foundation, state perception, risk assessment, upper-layer baseline control, and lower-layer residual compensation, as illustrated in Fig. 2.

(1) Data foundation layer: This layer is categorized into macroscopic traffic demand data and microscopic vehicle trajectory data. The former calibrates the L-METANET model and defines boundary traffic demands for environmental evolution in the SUMO simulation. In contrast, the latter facilitates the fitting and training of the risk assessment model.

(2) State perception layer: Serving as the data and model bedrock of the entire control framework, this layer leverages the L-METANET model alongside the underlying SUMO simulation environment to deliver high-precision state perception and traffic predictions for both MPC and MARL operations.

(3) Risk assessment layer: By integrating the XGBoost algorithm, SHAP interpretable analysis, and the random parameters binary logit (RPBL) model, this layer constructs an analytical risk evaluation and prediction framework. It simultaneously guarantees high risk-quantification accuracy and operational availability for control execution, quantifying dynamic collision risks across the network in real time to serve as the cost function for upper-layer MPC and the reward function for lower-layer MARL.

(4) Upper-layer baseline control layer (MPC-based): Operating at a longer macro-control period to capture long-term traffic evolution trends, this layer utilizes L-METANET as its predictive model. Under strict physical constraints, it computes a sequence of safe baseline control outputs via rolling horizon optimization, establishing a robust safety floor for system operations.

(5) Lower-layer residual compensation control layer (MARL-based): Executed at a shorter microcontrol step, this layer compensates for the sluggish response of MPC to microscopic disturbances and mitigates the impacts of predictive model mismatches. Distributed agents deployed at individual control nodes execute rapid inference based on real-time local states using an enhanced ST-MAPPO algorithm, outputting high-frequency residual control actions.

Ultimately, grounded in real-time state perception and risk assessment, the system dynamically superimposes the upper-layer MPC baseline commands with the lower-layer MARL residual adjustments. The combined signals undergo a safety boundary clipping process to comply with physical and regulatory constraints before being dispatched to the traffic actuators.

## 4. Macroscopic traffic flow and risk assessment models for expressway weaving segments

## 4.1. L-METANET: a lane-level macroscopic traffic flow model

This section extends the classic METANET model across multiple dimensions, most notably by expanding its granularity to the lane level, thereby establishing a generalized methodology for traffic flow modeling on urban expressway networks. Building upon this formulation, we also present a rigorous parameter calibration approach.

## 4.1.1 Formulation of L-METANET

The conventional METANET model (Messmer and Papageorgiou, 1990) treats multi-lane cross-sections as homogeneous entities, neglecting critical speed and density differentials between inner fast and outer slow lanes. This homogeneity assumption fails on urban expressways, lacking the granular state variables required for lane-level ATM. Furthermore, original METANET cannot characterize localized ramp disturbances on the outermost lane or the lateral lane-changing friction inherent to weaving segments. To bridge these gaps, this section downscales traffic flow variables to the lane level and introduces five critical modifications to the classic METANET framework, establishing a generalized macroscopic modeling methodology tailored for successive weaving segments:

(1) Expansion to the lane level: Discretizing cross-sectional traffic states into distinct, lane-specific variables to track fine-grained flow dynamics;

(2) Incorporation of asymmetric speed impacts: Modeling the directional, unbalanced speed degradation caused by lateral merging and diverging behaviors;

(3) Consideration of VSL control and compliance of HDVs: Incorporating the explicit impacts of VSL combined with HDV driver compliance rates;

(4) Accounting for upstream and downstream state interdependencies triggered by changes in lane drops or additions;

(5) Dynamic flow estimation of lateral behaviors: Quantifying the localized, cross-lane flow exchanges driven concurrently by free and forced lane-changing maneuvers.

## Improvement 1: Expansion to the lane level

To accurately characterize lane-segment-level traffic state evolution and facilitate lane-specific VSL, the corridor is discretized spatially into segments of length $\Delta x _ { m }$ , where $\Delta x _ { m }$ denotes the length of segment m . This partition establishes the mathematical foundation for subsequent variable-length segment modeling. Time is discretized into steps of duration t . For each traffic state variable, a lane index $n \left( n { = } 1 , 2 , \cdots , N _ { _ m } \right)$ is introduced, where $N _ { m }$ denotes the outermost lane and represents the total number of lanes in segment . The schematic layout of the segment-lane discretization and aggregated traffic flow variables is illustrated in Fig. 3. The lane-level macroscopic traffic flow model extended state evolution is governed by Eqs. (1) to (5).

$$
k _ { m , n } ^ { i + 1 } = k _ { m , n } ^ { i } + \cfrac { \Delta t } { \Delta x _ { m } } \big ( \hat { q } _ { m , n } ^ { i } + { r } _ { m , n } ^ { i } - { q } _ { m , n } ^ { i } \big )\tag{1}
$$

$$
\hat { q } _ { m , n } ^ { i } = \sum _ { n ^ { \prime } \in N ^ { m , n } } \Bigl ( q _ { m - 1 , \tilde { n } } ^ { i } + \phi _ { m - 1 , \tilde { n } } ^ { i , f r } + \phi _ { m - 1 , \tilde { n } } ^ { i , f o } \Bigr ) \cdot R _ { m - 1 , \tilde { n } \to n } ^ { i }\tag{2}
$$

$$
\nu _ { m , n } ^ { i + 1 } = \nu _ { m , n } ^ { i } + \frac { \Delta t } { \tau } [ V ( k _ { m , n } ^ { i } ) - \nu _ { m , n } ^ { i } ] + \frac { \Delta t } { \Delta x _ { m } } \cdot \nu _ { m , n } ^ { i } \cdot ( \nu _ { m - 1 , \tilde { n } } ^ { i } - \nu _ { m , n } ^ { i } ) - \frac { \eta \cdot \Delta t \cdot \big ( k _ { m + 1 , \tilde { n } } ^ { i } - k _ { m , n } ^ { i } \big ) } { \tau \cdot \Delta x _ { m } \cdot \big ( k _ { m , n } ^ { i } + \kappa \big ) }\tag{3}
$$

$$
V ( k _ { m , n } ^ { i } ) = \nu _ { f , m , n } \cdot \exp \left[ - \frac { 1 } { a _ { m , n } } \left( \frac { k _ { m , n } ^ { i } } { k _ { c r , m , n } } \right) ^ { a _ { m , n } } \right]\tag{4}
$$

$$
q _ { m , n } ^ { i } = k _ { m , n } ^ { i } \cdot \nu _ { m , n } ^ { i }\tag{5}
$$

where Eq. (1) is the traffic flow conservation equation: $k _ { m , n } ^ { i } , ~ q _ { m , n } ^ { i }$ respectively represent the density and flow rate of lane  in segment  at time step $i ; ~ r _ { m , n } ^ { i }$ and ${ \boldsymbol { s } } _ { m , n } ^ { i }$ respectively represent the inflow merging from the on-ramp and the outflow diverging to the off-ramp connected to lane of segment $m ; ~ { \hat { q } } _ { m , n } ^ { i }$ is the longitudinal flow entering lane of segment m from all lanes in segment $m - 1$ connected to lane of segment $m$ at time step , with its calculation formula given in Eq. (2); is the lane number of the upstream segment connected to lane . Generally, when there is no increase or decrease of lanes on the mainline, ${ \tilde { n } } = n$ , but this mapping relationship will change when on/off-ramps or lane changes exist, which should strictly depend on the actual road connection layout; $\phi _ { m - 1 , \tilde { n } } ^ { i , f r }$ represents the flow rate variation in lane  of segment $m - 1$ caused by free lane-changing behaviors, $\phi _ { m - 1 , { \tilde { n } } } ^ { i , f r } = \phi _ { m - 1 , { \tilde { n } } - 1 \to { \tilde { n } } } ^ { i , f r } + \phi _ { m - 1 , { \tilde { n } } + 1 \to { \tilde { n } } } ^ { i , f r } - \phi _ { m - 1 , { \tilde { n } } \to { \tilde { n } } - 1 } ^ { i , f r } - \phi _ { m - 1 , { \tilde { n } } \to { \tilde { n } } + 1 } ^ { i , f r } ; \phi _ { m - 1 , n } ^ { i , f o } .$ represents the flow rate variation in lane n of segment m-1 caused by forced lane-changing behaviors, $\phi _ { m - 1 , { \tilde { n } } } ^ { i , f o } = \phi _ { m , { \tilde { n } } + 1 \to { \tilde { n } } } ^ { i , f o } + \phi _ { m , { \tilde { n } } - 1 \to { \tilde { n } } } ^ { i , f o } - \phi _ { m , { \tilde { n } } \to { \tilde { n } } + 1 } ^ { i , f o } - \phi _ { m , { \tilde { n } } \to { \tilde { n } } - 1 } ^ { i , f o } - s _ { m - 1 , { \tilde { n } } } ^ { i } ; R _ { m - 1 , { \tilde { n } } \to n } ^ { i }$ is the proportion of traffic flowing from lane to lane within segment $m - 1 ; ~ N ^ { m , n }$ is the set of all lane numbers in segment $m - 1$ connected to lane  of segment  . Eq. (3) is the dynamic speed equation: the first term represents the average speed of lane in segment at time step ; the second term is the relaxation term, representing the difference between the desired speed and the current average speed, which reflects the tendency of drivers to travel at the desired speed during driving; the third term is the convection term, representing the speed variation caused by vehicles entering the current cell from the upstream cell $m - 1$ , reflecting the impact of upstream speed fluctuations; the fourth term is the anticipation term, which adjusts the speed of the current cell through feedback on the density variation of the downstream cell $m + 1 ,$ reflecting the impact of downstream density changes, where is the driver reaction time, is the speed-density relationship coefficient, and  is the elasticity coefficient. Eq. (4) is the steady-state speed equation, and Eq. (5) represents the three-parameter relationship equation: $\nu _ { f , m , n } , \ k _ { c r , m , n }$ , and $a _ { m , n }$ are the free-flow speed, critical density, and the model parameter controlling the shape of the fundamental diagram for lane in segment , respectively.

![](images/f7ddcf13be1fc3013dd4296e9be628ad136f74ea0c6eb1a1beb2a04c69229ce7.jpg)  
Fig. 3 Schematic diagram of segment-lane division and aggregated traffic flow variables

Subsequently, it is also necessary to model the flow equations for the origin segments. These origin segments (including on-ramps and mainline entries) serve to receive and discharge network traffic demands, where the actual entering flow rate depends on the minimum of the desired inflow (demand) and the maximum downstream acceptable capacity (supply). The relationship between traffic demand and the entering flow rate within the L-METANET macroscopic traffic flow model is captured via a queue model, with the specific calculation procedures formulated in Eqs. (6) to (11).

$$
q _ { 0 } ^ { i } = z _ { m } ^ { i } \hat { q } _ { 0 } ^ { i }\tag{6}
$$

$$
z _ { m } ^ { i } = \left\{ \begin{array} { l } { 0 , 0 i s m a i n l i n e } \\ { R _ { m } ^ { i } , 0 i s r a m p } \\ { 1 , 0 i s m a i n \mathrm { e } n t r a n c e } \end{array} \right.\tag{7}
$$

$$
\hat { q } _ { 0 } ^ { i } = \operatorname* { m i n } \left( q _ { 0 } ^ { i , 1 } , q _ { 0 } ^ { i , 2 } \right)\tag{8}
$$

$$
{ q _ { 0 } ^ { i , 1 } = d _ { 0 } ^ { i } + \frac { w _ { 0 } ^ { i } } { \Delta t } }\tag{9}
$$

$$
q _ { 0 } ^ { i , 2 } = { \mathcal { Q } } _ { 0 } ^ { c a p , i } \times \operatorname* { m i n } \left( 1 , { \frac { k _ { j a m } - k _ { 1 } ^ { i } } { k _ { j a m } - k _ { c r } } } \right)\tag{10}
$$

$$
w _ { 0 } ^ { i + 1 } = w _ { 0 } ^ { i } + \triangle t \cdot \left( d _ { 0 } ^ { i } - q _ { 0 } ^ { i } \right)\tag{11}
$$

where Eqs. (6) to (11) are the flow rate equations for the origin segment $( q _ { 0 } ^ { i } ) .$ , which depend on the minimum value between the traffic demand of the origin segment $q _ { 0 } ^ { i , 1 }$ and the maximum acceptable flow rate of the mainline segment $q _ { 0 } ^ { i , 2 }$ . It should be noted that when the origin segment is the mainline origin segment, $q _ { 0 } ^ { i } = \hat { q } _ { 0 } ^ { i } ,$ when the origin segment is an on-ramp entrance, it is also influenced by the RM rate $R _ { m } ^ { i } \in \left( 0 , 1 \right) . \mathrm { E q . } \ ( 9 )$ is the calculation formula for $q _ { 0 } ^ { i , 1 }$ , which is determined by the average arrival flow of the origin segment $d _ { 0 } ^ { i }$ and the queue length of the origin segment $w _ { 0 } ^ { i }$ per unit time. Eq. (10) is the calculation formula for $q _ { 0 } ^ { i , 2 }$ , which is determined by the mainline capacity of the origin segment $\mathcal { Q } _ { 0 } ^ { c a p , i }$ and the current state of the road $( k _ { c r } , \ k _ { 1 } ^ { i }$ , and the jam density $k _ { j a m } )$ . $\operatorname { E q }$ . (11) is the calculation formula for $w _ { 0 } ^ { i + 1 }$ , which is determined by $w _ { 0 } ^ { i }$ , $d _ { 0 } ^ { i }$ , and $q _ { 0 } ^ { i }$

Improvement 2: Asymmetric speed impacts from lateral merging and diverging behaviors.

In the METANET model, the impacts of on-ramp merging, off-ramp diverging, and weaving behaviors on speed are neglected. When the modeling granularity downscales to the lane level, the dynamic boundaries of traffic flow undergo a fundamental shift: for any specific lane $n ,$ the physical essence of vehicles merging from an on-ramp into the mainline is approximately equivalent to vehicles changing from an adjacent lane into this lane, both of which manifest as the insertion of external vehicles into the subject lane. Such forced insertions into the gaps ahead of the target lane disrupt the original steadystate car-following behavior. To re-establish safe spacing, following vehicles in the target lane must execute intense braking maneuvers, thereby leading to a sharp degradation in the macroscopic average speed of that lane. Conversely, vehicles departing from the current lane to an off-ramp or changing to an adjacent lane both manifest as the extraction of vehicles from the current lane, which also exerts a certain impact on the average speed of that lane; naturally, this impact is far less severe than that caused by vehicle insertions, necessitating separate considerations. Consequently, we define a generalized inflow $\phi _ { m , n } ^ { i , i n }$ and a generalized outflow $\phi _ { m , n } ^ { i , o u t }$ , which respectively represent the number of entering and departing vehicles accepted by lane of segment within time step , thereby better quantifying the speed impacts triggered by lane-changing behaviors.

As mentioned, lateral lane-changing behaviors are categorized into free lane-changing and forced lane-changing: (1) Free lane-changing represents the lateral migrations across lanes executed by vehicles to achieve higher operational efficiency. Let $\phi _ { m , n + 1  n } ^ { i , f r }$ denote the traffic flow transferring from lane $n { - } 1$ to lane  within segment  due to free lane-changing behaviors at time step $i ; ( 2 )$ Forced lanechanging refers to the compulsory merging and diverging maneuvers dictated by on-ramp and off-ramp operations, which comprise two components. The first component is the on-ramp inflow $r _ { m , n } ^ { i }$ , representing the traffic flow entering lane n of segment from the adjacent on-ramp under a one-to-one mapping. The second component is the traffic flow forcing its way lane-by-lane from various inner lanes to depart the mainline via the off-ramp. Specifically, when $s _ { m , n } ^ { i } < q _ { m , n } ^ { i } , \phi _ { m , n + 1  n } ^ { i , f o } = 0$ ; when $s _ { m , n } ^ { i } > q _ { m , n } ^ { i } ,$ , vehicles in the more inner lanes must first laterally change into lane before diverging from lane to exit the mainline, where $\phi _ { m , n + 1  n } ^ { i , f o }$ represents this specific flow component. It should be noted that under extreme conditions, $s _ { m , n } ^ { i }$ can become exceptionally large, requiring vehicles across multiple inner lanes to execute successive lateral lane changes to satisfy off-ramp diverging demands. Grounded in these analyses, the calculation methods for $\phi _ { m , n } ^ { i , i n }$ and $\phi _ { m , n } ^ { i , o u t }$ are formulated in Eqs. (12) to (13). Within these two equations, the first two terms represent the traffic flow induced by free lane-changing, whereas the latter two terms represent the flow driven by forced lane-changing behaviors.

$$
\phi _ { m , n } ^ { i , i n } = \phi _ { m , n - 1 \to n } ^ { i , f r } + \phi _ { m , n + 1 \to n } ^ { i , f r } + r _ { m , n } ^ { i } + \phi _ { m , n + 1 \to n } ^ { i , f o } + \phi _ { m , n - 1 \to n } ^ { i , f o }\tag{12}
$$

$$
\phi _ { m , n } ^ { i , o u t } = \phi _ { m , n  n - 1 } ^ { i , f r } + \phi _ { m , n  n + 1 } ^ { i , f r } + s _ { m , n } ^ { i } + \phi _ { m , n  n - 1 } ^ { i , f o } + \phi _ { m , n  n + 1 } ^ { i , f o }\tag{13}
$$

$$
\nu _ { m , n } ^ { i + 1 } = \nu _ { m , n } ^ { i } + \frac { \Delta t } { \tau } \bigg [ V \left( k _ { m , n } ^ { i } \right) - \nu _ { m , n } ^ { i } \bigg ] + \frac { \Delta t } { \Delta x _ { m } } \cdot \nu _ { m , n } ^ { i } \cdot \left( \nu _ { m - 1 , \bar { n } } ^ { i } - \nu _ { m , n } ^ { i } \right) - \frac { \eta \cdot \Delta t \cdot \left( k _ { m + 1 , \bar { n } } ^ { i } - k _ { m , n } ^ { i } \right) } { \tau \cdot \Delta x _ { m } \cdot \left( k _ { m , n } ^ { i } + \kappa \right) }\tag{14}
$$

$$
- \frac { \boldsymbol { \delta } _ { i n } \cdot \Delta t \cdot \boldsymbol { \nu } _ { m , n } ^ { i } \cdot \boldsymbol { \phi } _ { m , n } ^ { i , i n } } { \Delta x _ { m } \cdot \left( \boldsymbol { k } _ { m , n } ^ { i } + \boldsymbol { \kappa } \right) } - \frac { \boldsymbol { \delta } _ { o u t } \cdot \boldsymbol { \Delta t } \cdot \boldsymbol { \nu } _ { m , n } ^ { i } \cdot \boldsymbol { \phi } _ { m , n } ^ { i , o u t } } { \Delta x _ { m } \cdot \left( \boldsymbol { k } _ { m , n } ^ { i } + \boldsymbol { \kappa } \right) }
$$

Finally, we construct the dynamic speed equation that accounts for the asymmetric speed impacts of lateral merging and diverging behaviors. The modified lane-level dynamic speed equation is formulated in Eq. (14). Because the impacts of vehicle insertion and extraction on speed differ significantly, with insertion exerting a substantially greater impact than extraction, the parameter calibration must strictly satisfy $\delta _ { i n } > \delta _ { o u t }$ . This mathematically guarantees the asymmetric traffic flow characteristic wherein deceleration occurs more readily than speed recovery within weaving segments.

Improvement 3: The impacts of VSL and HDV driver compliance rates.

In a connected and automated environment implementing lane-level ATM, the LVSL commands dynamically displayed on the roadside (or directly received by RSU) artificially truncate the desired speed of vehicles in a free-flow state. However, the actual traffic stream on the roadway is a mixed traffic flow composed of CAVs and HDVs. There is a fundamental difference in the response mechanisms of these two classes of vehicles to management and control commands: CAVs can obtain speed limit commands in real time via V2X communications and strictly execute them with 100% compliance, whereas HDVs exhibit only partial compliance with speed limit commands due to individual driver heterogeneity, perception delays, and subjective driving intentions. To scientifically quantify the comprehensive impact of these heterogeneous compliance characteristics on macroscopic traffic flow, this model introduces the CAV penetration rate parameter $\alpha \in \left[ 0 , 1 \right]$ and the average speed limit compliance rate parameter of HDVs $\beta \in \left( 0 , 1 \right)$ . Accordingly, a comprehensive speed limit compliance index for mixed traffic flow $\beta _ { m i x }$ is designed, as shown in Eq. (15). Based on this comprehensive index that governs the variations in desired speed caused by ${ \mathrm { V S L } } ,$ , the steady-state speed fundamental diagram equation is modified to take the minimum value between the speed limit and the spontaneous desired speed, thereby accounting for the heterogeneous characteristics of mixed traffic flow, as shown in Eq. (16).

$$
\beta _ { m i x } = \alpha \cdot 1 + ( 1 - \alpha ) \cdot \beta\tag{15}
$$

$$
V ( k _ { m , n } ^ { i } ) = \operatorname* { m i n } \left\{ \beta _ { m i x } \cdot \nu _ { m , n } ^ { F S L , i } + ( 1 - \beta _ { m i x } ) \cdot \nu _ { f , m , n } \cdot \exp \left[ - \frac { 1 } { a _ { m , n } } \left( \frac { k _ { m , n } ^ { i } } { k _ { c , m , n } } \right) ^ { a _ { m , n } } \right] , \nu _ { f , m , n } \cdot \exp \left[ - \frac { 1 } { a _ { m , n } } \left( \frac { k _ { m , n } ^ { i } } { k _ { c , m , n } } \right) ^ { a _ { n , n } } \right] \right\}\tag{16}
$$

It should be noted that the application of VSL fundamentally alters the shape of the traffic fundamental diagram, causing the capacity of the road segment to evolve dynamically. According to traffic flow theory, the steady-state flow-density relationship function $q _ { m , n } ^ { V S L , i } ( k )$ of a lane unit equals the product of density and the desired speed. Substituting Eq. (16) fully into this relationship yields a piecewise flow function, as shown in Eq. (17). The dynamic capacity of the roadway corresponds to the global maximum of this flow-density function, and the density at this extreme point defines the dynamic critical density $k _ { c r , m , n } ^ { V S L , i }$ . Since this requires calculating the derivative of the function, solving for this extreme point necessitates differentiating the flow with respect to density in Eq. (17) and setting the derivative to zero $( \mathrm { i . e . , ~ \ d } q _ { m , n } ^ { V S L , i } ( k ) / \mathrm { d } k = 0 )$ , thereby obtaining the stationary points of the function. According to the mathematical theory of optimization for piecewise continuous functions, this global maximum point must originate from the following set of three candidate critical points $\Im { = } \left\{ k _ { _ A } , k _ { _ B } , k _ { _ C } \right\}$

(1) The stationary point of the speed-limit branch $k _ { A }$ : Differentiating $q _ { m , n , 1 } ^ { V S L , i } ( k )$ and setting ${ \mathrm { d } } q _ { m , n , 1 } ^ { V S L , i } ( k ) / { \mathrm { d } } k = 0$ yields a transcendental equation. Although this transcendental equation lacks an analytical solution, $k _ { A }$ can be obtained by solving the equation via numerical algorithms.

(2) The stationary point of the natural branch $k _ { B }$ : Differentiating $q _ { m , n , 2 } ^ { V S L , i } ( k )$ and setting ${ \mathrm { d } } q _ { m , n , 2 } ^ { V S L , i } ( k ) / { \mathrm { d } } k = 0$ . Because $q _ { m , n , 2 } ^ { V S L , i } ( k )$ represents the uncontrolled fundamental diagram function, its stationary point solution remains identically equal to the original natural critical density, i.e., $k _ { B } = k _ { c r , m , n }$

(3) The intersection point of the two branches $k _ { C }$ : Equating the two flow curves, i.e., $q _ { m , n , 1 } ^ { V S L , i } ( k ) = q _ { m , n , 2 } ^ { V S L , i } ( k )$ , yields the non-differentiable turning point $k _ { C }$

To determine the final global maximum point, a physical validity check must be performed on the points within the candidate set to eliminate spurious peaks that are truncated by the envelope and cannot be achieved in real-world traffic flow. Valid density points must strictly comply with the definition of the lower envelope: if $k _ { \mathrm { \ell \ell } }$ satisfies $q _ { m , n , 1 } ^ { V S L , i } ( k _ { { \scriptscriptstyle A } } ) \leq q _ { m , n , 2 } ^ { V S L , i } ( k _ { { \scriptscriptstyle A } } )$ , then $k _ { A }$ is valid; if $k _ { B }$ satisfies $q _ { m , n , 2 } ^ { V S L , i } ( k _ { B } ) \leq q _ { m , n , 1 } ^ { V S L , i } ( k _ { B } )$ , then $k _ { B }$ is valid; the intersection point $k _ { C }$ is inherently valid. Let $\mho _ { \nu a l i d }$ denote the set of valid candidate points. Among all valid candidate points that pass the verification, the flow values corresponding to all candidates are extracted; the maximum flow value represents the traffic capacity under the influence of LVSL and mixed fleet characteristics $\mathcal { Q } _ { m , n } ^ { c a p , i }$ , as shown in Eq. (19). Its corresponding valid candidate point defines the dynamic critical density $k _ { c r , m , n } ^ { V S L , i }$

$$
\begin{array} { r l } & { u _ { \mu , \alpha } ^ { \mathrm { { X } } , \alpha } ( t ) = P ( k _ { \mu } ^ { ( \alpha ) } , \gamma _ { \alpha , \alpha } ^ { \beta , \alpha } ) } \\ & { = k \cdot \operatorname* { m i n } \{ \beta _ { \alpha  \gamma } \gamma _ { \alpha , \alpha } ^ { \beta , \alpha } = ( 1 - \beta _ { \alpha  \gamma } ) \cdot v _ { \gamma , \alpha  \gamma \leq 0 } [ - \frac { 1 } { \alpha _ { \alpha  \gamma } } ( \frac { k } { k _ { \alpha  \alpha } } ) ^ { \alpha _ { \alpha  \gamma } } ] \cdot v _ { \gamma , \alpha  \gamma \leq 0 } [ - \frac { 1 } { \alpha _ { \alpha  \gamma } } ( \frac { k } { k _ { \alpha  \alpha } } ) ^ { \alpha _ { \beta  \gamma } } ] \} } \\ & { = \frac { \alpha _ { \alpha  \gamma } } { \alpha _ { \alpha  \alpha } } - ( k ) \cdot \beta _ { \alpha  \gamma } v _ { \alpha , \alpha } ^ { \beta , \alpha } \cdot ( 1 - \beta _ { \alpha  \gamma } ) \cdot v _ { \gamma , \alpha  \gamma \leq 0 } \Bigg [ - \frac { 1 } { \alpha _ { \alpha  \gamma } } ( \frac { k } { k _ { \alpha  \alpha } } ) ^ { \alpha _ { \alpha  \gamma } } \Bigg ] \cdot q _ { \alpha  \gamma } ^ { \alpha \alpha } ( k ) \cdot q _ { \alpha  \gamma } ^ { \alpha \alpha } ( k ) } \\ &  = [ \begin{array} { l } { \alpha _ { \alpha  \gamma } ^ { \alpha } - ( k ) \cdot k _ { \alpha  \gamma } \exp [ - \frac { 1 } { \alpha _ { \alpha  \gamma } } ( \frac { k } { k _ { \alpha  \alpha } } ) ^ { \alpha _ { \alpha  \gamma } } ] \cdot q _ { \alpha  \gamma } ^ { \alpha \alpha } ( k ) \cdot q _ { \alpha  \gamma } ^ { \alpha \alpha } ( k ) } \\  \alpha _ { \alpha  \alpha } ^ { \beta , \alpha } - ( k ) \ \end{array} \end{array}\tag{17}
$$

(18)

(19)

Improvement 4: Upstream and downstream state interdependencies under cross-sectional geometric variations.

In METANET, when the speed of the upstream segment is greater than that of the current segment m , the convection term is positive. Without considering other influencing factors, the speed of the current segment in the next sampling period will increase. However, this is an idealized condition and is not fully applicable to realistic, complex weaving segment scenarios: if the upstream segment $m - 1$ is in a free-flow state, the downstream segment $m + 1$ is in a congested state, and the current segment m is in an unstable car-following state. If the upstream speed is greater than the current segment speed at this moment, the original model would conclude that the speed of segment will increase in the next period; however, empirical traffic flow experience indicates that vehicles in the current segment will instead decelerate due to the backward propagation effect of downstream congestion waves. This contradiction demonstrates that the convection term in the original equation is primarily applicable to situations where both upstream and downstream segments are in free-flow states. Grounded in this, a geometric mean is adopted to improve the convection term. Combining the dynamic constraints imposed by VSL on the desired speed introduced in the previous improvement (see Eq. (16)) and the asymmetric friction effect exerted on the subject lane by lateral lane-changing behaviors (see Eq. (14)), the comprehensive lane-level dynamic speed evolution equation ultimately constructed in this study is formulated as Eq. (20).

$$
\begin{array} { l } { { \nu _ { m , n } ^ { i + 1 } = \nu _ { m , n } ^ { i } + \displaystyle \frac { \Delta t } { \tau } \Big [ V \left( k _ { m , n } ^ { i } \right) - \nu _ { m , n } ^ { i } \Big ] + \displaystyle \frac { \Delta t } { \Delta x _ { m } } \cdot \nu _ { m , n } ^ { i } \cdot \left( \sqrt { \nu _ { m - 1 , \bar { n } } ^ { i } \cdot \nu _ { m , n } ^ { i } } - \nu _ { m , n } ^ { i } \right) - \displaystyle \frac { \eta \cdot \Delta t \cdot \left( k _ { m + 1 , \bar { n } } ^ { i } - k _ { m , n } ^ { i } \right) } { \tau \cdot \Delta x _ { m } \cdot \left( k _ { m , n } ^ { i } + \kappa \right) } } } \\ { { - \displaystyle \frac { \delta _ { i n } \cdot \Delta t \cdot \nu _ { m , n } ^ { i } \cdot \phi _ { m , n } ^ { i , i n } } { \Delta x _ { m } \cdot \left( k _ { m , n } ^ { i } + \kappa \right) } - \displaystyle \frac { \delta _ { o u t } \cdot \Delta t \cdot \nu _ { m , n } ^ { i } \cdot \phi _ { m , n } ^ { i , o u t } } { \Delta x _ { m } \cdot \left( k _ { m , n } ^ { i } + \kappa \right) } } } \end{array}\tag{20}
$$

Improvement 5: Dynamic flow estimation for lateral lane-changing behaviors involving free and forced lane-changing, which can be viewed in Appendix A.

## 4.1.2 Calibration method of L-METANET

Following the formulation of the theoretical L-METANET model, its parameters must be precisely calibrated using road network data to ensure the model accurately captures the spatiotemporal evolution characteristics of traffic flow within the expressway network. Drawing upon systematic frameworks for macroscopic traffic flow modeling, this section establishes a parameter calibration system across three dimensions: parameter set definition, evaluation variable selection, and an efficient solution algorithm.

(1) Parameter decoupling and calibration set formulation: the variables of L-METANET are decoupled into two distinct subsets to reduce optimization complexity: First, the global dynamic parameters encompass the driver reaction time $\tau ,$ , the speed-density relationship coefficient $\eta .$ the anticipation elasticity coefficient parameter $\kappa _ { \mathbf { \Omega } , \kappa }$ , and the weight coefficients $\delta ^ { i n }$ and $\delta ^ { o u t }$ that characterize the lateral asymmetric friction effects across lanes. Second, the local fundamental diagram parameters are utilized to characterize the geometric heterogeneity across road space (between different segments or different lanes). This subset includes the free-flow speed $\nu _ { f , m , n } .$ , natural critical density $k _ { c r , m , n }$ , and shape parameter $a _ { m , n }$ for a specific segment-lane unit $( m , n )$ . All parameter vectors to be identified $\theta$ constitute the decision variable set for the calibration optimization problem, as shown in Eq. (21). To avoid calibration intractability caused by high-dimensional over-parameterization, this study assumes uniform fundamental diagram parameters across the entire network.

(2) Evaluation variable and objective function construction: Parameter calibration is essentially a parameter estimation problem for a nonlinear dynamic system. Referencing relevant studies (Wang et al., 2022), this study selects the average speed $\nu _ { m , n } ^ { i }$ as the core evaluation metric. The objective function $J ( \pmb { \theta } )$ is formulated to minimize the root-mean-square error (RMSE), as shown in Eq. (22).

(3) Hybrid global optimization algorithm based on GA-VNS: Owing to the complex extremum calculations and lane-level coupling embedded in the L-METANET model, its objective function space exhibits highly nonlinear and non-convex properties. Conventional gradient-based optimization algorithms are highly susceptible to trapping in local optima when solving such problems. Synthesizing the advantages of multiple meta-heuristic algorithms, this study adopts a hybrid global optimization strategy combining a genetic algorithm (GA) and variable neighborhood search (VNS) (Ma et al., 2025), which was specifically designed in our previous work for parameter calibration. This algorithm framework has been implemented multiple times in our previous studies, demonstrating robust population evolution capabilities and a strong capacity to escape local optima.

$$
\theta = \left\{ \tau , \eta , \kappa , \delta ^ { i n } , \delta ^ { o u t } , \mu _ { _ { C A V } } , \mu _ { _ { H D V } } , \nu _ { f , m , n } , k _ { _ { c r , m , n } } , a _ { _ { m , n } } \right\} _ { _ { m = 1 , \ldots , M } } ^ { n = 1 , \ldots , N }\tag{21}
$$

$$
J ( \mathbf { \pmb { \theta } } ) = \sqrt { \frac { 1 } { I \cdot \sum _ { m = 1 } ^ { M } N _ { m } } \sum _ { i = 1 } ^ { K } \sum _ { m = 1 } ^ { M } \sum _ { n = 1 } ^ { N _ { m } } \left( \hat { \nu } _ { m , n } ^ { i } - \nu _ { m , n } ^ { i } \right) ^ { 2 } }\tag{22}
$$

where $\nu _ { m , n } ^ { i }$ and $\hat { \nu } _ { m , n } ^ { i }$ denote the calculated speed output by the L-METANET model and the empirically observed speed from detectors, respectively, for lane of segment at time step ; represents the total number of time steps within the calibration period, and represents the total number of calibrated segments.

## 4.2. Accuracy- and controllability-driven risk prediction and assessment model for weaving segments

Coordinated VSL and RM control requires a rigorous risk-based objective function balancing riskcharacterization accuracy with operational controllability. Traditional macroscopic risk models neglect microscopic conflict mechanisms, while high-fidelity microscopic models are disconnected from macroscopic control parameters. To bridge this gap, this study identifies macroscopic precursors of merging and diverging risks to synthesize microscopic fidelity with macroscopic controllability. Furthermore, to resolve the trade-off between black-box machine learning models lacking closed-form expressions and traditional logit models suffering from multicollinearity, we develop a hybrid XGBoost-SHAP-RPBL framework. This framework achieves high-dimensional feature reduction, accounts for unobserved heterogeneity, and derives an analytical, closed-form risk probability output suitable for optimization.

## 4.2.1 The XGBoost-SHAP-RPBL framework and implementation steps

The implementation procedure of XGBoost-SHAP-RPBL consists of five core steps, as illustrated in the flowchart in Fig. 4. Note that Steps 1 through 3 were completed in our previous study (Ma et al., 2026a) and are thus only briefly outlined here, whereas Steps 4 and 5 represent the novel contributions of this study, with their detailed mathematical procedures elaborated in Section 4.2.2.

![](images/e593673ed228aaccab51718820308c3e43ce8cd8deb56ee7c47b91ff4531c15e.jpg)  
Fig. 4 Fitting process of XGBoost-SHAP-RPBL

Step 1: Candidate feature set construction: A three-dimensional feature space (lane-segment-factor) is designed to capture the complex spatiotemporal heterogeneity within weaving segments. Spatially, the weaving segment is partitioned longitudinally into upstream (Segment 1), midstream (Segment 2), and downstream (Segment 3) zones, and laterally discretized into the inner fast lanes and outer weaving lanes (Lane 1, Lane 2, etc.). Factorially, core macroscopic traffic parameters: cell density, average speed, flow rate, maximum speed, and speeding vehicle counts are extracted for each spatiotemporal cell.

Step 2: dataset construction for training and fitting: Empirical data are processed into a standardized dataset, aligning microscopic vehicle interactions with macroscopic traffic states via three substeps: Step 2.1: microscopic trajectory extraction: Vehicle trajectories are extracted from 3.6 hours of UAV aerial video data capturing seven short weaving segments on Changchun expressways under various congestion levels. Microscopic trajectory data for all vehicles within the weaving segments were extracted, and all merging and diverging vehicles, along with their surrounding vehicles, were identified. Each merging or diverging vehicle and its corresponding surrounding vehicles form a group, which constitutes a single sample; Step 2.2: macroscopic parameter aggregation: The weaving segment is discretized into continuous macroscopic time slices. Microscopic trajectories within each lane-segment grid cell are aggregated per time slice to extract cell-level traffic flow indicators, forming the independent variable candidate set; Step 2.3: STRF-based weaving risk classification: Based on the STRF model developed in our previous study (Ma et al., 2026b), the STRF is utilized to risk classification. A sample with a risk exceeding the STRF threshold is defined as high-risk, whereas the remaining samples are classified as low-risk.

Step 3: construction of the XGBoost-based risk prediction model: The dataset constructed in Step 2 is input into XGBoost for preliminary classification modeling. Utilizing a powerful ensemble of base learners and regularization mechanisms, XGBoost automatically suppresses the weights of redundant features while accurately capturing the complex nonlinear threshold responses and synergetic interactions between macroscopic traffic flow characteristics and merging/diverging risks.

Step 4: SHAP-Driven Feature Selection for RPBL Modeling: To resolve the black-box limitation of XGBoost and establish a parsimonious input set for subsequent statistical modeling, the SHAP framework calculates the global marginal contribution of each feature. Highly multicollinear or low-contribution features are eliminated based on two criteria: a SHAP importance threshold of 0.2 and a maximum allocation of eight variables. This dimensional reduction retains only the core physical variables with decisive impacts on crash risk (listed in Appendix C), which serve as the explanatory variables for the final Logit-based RPBL model.

Step 5: RPBL model estimation and analytical risk equation derivation: The core feature subset $\mathbf { X } ^ { i }$ selected via SHAP is utilized as the explanatory variables, and all samples from Step 2 are input into the RPBL model for joint estimation. By introducing random parameters governed by specific probability distributions to capture the unobserved heterogeneity in traffic flow, the model yields a smooth, differentiable analytical equation for the overall crash risk over the prediction horizon, thereby establishing a safety cost function for the control system.

## 4.2.2 Risk prediction and assessment model based on random parameters binary Logit

The final stage of the hybrid framework involves converting the reduced-dimensional core feature set $\mathbf { X } ^ { i }$ into a smooth, differentiable continuous probability equation. Since traffic flow within weaving segments is continuously exposed to highly uncertain open environments, random variations in microscopic driver aggressiveness, localized transient weather, and vehicle mechanical performance can cause the crash probability induced by identical macroscopic traffic states to fluctuate drastically. Conventional fixed-parameter Logit models fail to capture such unobserved heterogeneity within the data. To address this limitation, a random parameters binary logit (RPBL) model is developed as the final analytically expressed risk assessment and prediction model. The framework utilizes the feature state $\mathbf { X } ^ { i }$ at the current time step to evaluate the utility function $U ^ { i }$ for the entire weaving segment evolving into a high-risk conflict state at time step , as formulated in Eq. (23).

$$
U ^ { i } = \beta _ { 0 } + \sum _ { k = 1 } ^ { K } \beta _ { k } \cdot \mathbf { X } _ { k } ^ { i } + \varepsilon ^ { i }\tag{23}
$$

where $\beta _ { 0 }$ is the constant intercept, and $\varepsilon ^ { i }$ is the random error term following an extreme value distribution. The $\mathbf { X } _ { k } ^ { i }$ denotes the $k ^ { t h }$ core traffic flow variable standardized via Z-score transformation to eliminate scale effects and ensure the convergence of the simulated maximum likelihood estimation algorithm, as expressed in Eq. (24).

$$
\mathbf { X } _ { k } ^ { i } = \frac { x _ { k } ^ { i } - \mu _ { k } } { \sigma _ { k } }\tag{24}
$$

where $\ v { x } _ { k } ^ { i }$ represents the raw value of the $k ^ { t h }$ core traffic flow variable at time step , while $\mu _ { k }$ and $\sigma _ { k }$ denote the mean and standard deviation of the $k ^ { t h }$ core traffic flow variable, respectively.

The RPBL model relaxes the restrictive assumption of fixed regression coefficients by endowing the feature parameters $\beta _ { k }$ with stochastic properties to capture spatiotemporal fluctuations. The random parameter $\beta _ { k }$ is defined as Eq. (25).

$$
\beta _ { k } = \left\{ \begin{array} { c } { \overline { { \beta } } _ { k } + \varphi _ { k } , k \in K } \\ { \overline { { \beta } } _ { k } , k \notin K } \end{array} \right.\tag{25}
$$

where $\overline { { \beta } } _ { k }$ represents the fixed mean of the parameter across the population, characterizing the average positive impact of the feature on the weaving segment crash risk. Meanwhile, $\varphi _ { k }$ is defined as a normally distributed random disturbance term, such that $\varphi _ { k } \sim ( 0 , \hat { \sigma } _ { k } ^ { 2 } )$ . Crucially, rather than treating all candidate variables as random parameters, only the core variables selected via SHAP constitute the explanatory variable set for the RPBL model. A fixed-parameters binary logit model is first estimated as the baseline. Subsequently, the significance of the random standard deviation for each core variable parameter is tested sequentially. The variable is identified as a random parameter, denoted as $k \in K$ , only if its standard deviation is statistically significant at a given significance level $\left( { p = 0 . 0 5 } \right)$ . Otherwise, it remains in the final model as a fixed parameter, denoted as $k \notin K$ . This parameter-level formulation enables the model to effectively absorb unobserved stochastic noise within the physical traffic system.

Based on the utility formulation above, the dynamic probability of a merging/diverging crash occurring across the entire weaving segment at time step , denoted as $C R ^ { i }$ , is obtained by integrating the Logit cumulative distribution function, as shown in Eq. (26).

$$
C R ^ { i } = \int \frac { \exp ( U ^ { i } ) } { 1 + \exp ( U ^ { i } ) } \cdot f ( \beta ) d \beta \approx \frac { 1 } { N _ { c } } { \sum _ { d = 1 } ^ { N _ { c } } } \Bigg [ \frac { 1 } { 1 + \exp ( - U _ { d } ^ { i } ) } \Bigg ]\tag{26}
$$

where $f ( \beta )$ represents the joint probability density function of the random parameters. Because this integration lacks a closed-form analytical solution, this study employs SML estimation, utilizing Halton sequences to execute $N _ { c }$ (set to 500) Monte Carlo to achieve high-precision parameter calibration.

The hybrid XGBoost-SHAP-RPBL model successfully distills massive, discrete traffic flow data characterized by heterogeneous fluctuations into a continuous dynamic analytical equation for crash probability bounded between . This function directly serves as the core objective for the coordinated VSL-RM optimization, driving the VSL and RM actuators to proactively manage the macroscopic states of the expressway, thereby mitigating merging and diverging crash risks within the weaving segment.

## 5. Hierarchical coordinated control framework for lane-level variable speed limits and ramp metering

Despite the high accuracy of the developed L-METANET model, macroscopic formulations inevitably suffer from environmental disturbances and minor model mismatches. Because MPC relies heavily on model fidelity, these perturbations challenge its control stability. Conversely, while standalone MARL provides robust high-frequency adaptability, its lack of physical boundaries and domain knowledge renders it susceptible to triggering traffic breakdowns in unseen scenarios. To address these challenges, we develop a hierarchical ATM framework for joint LVSL and RM control, establishing a closed-loop architecture where the upper-level MPC provides a rigid baseline safeguard and the lowerlevel MARL injects residual compensation.

## 5.1. Overall architecture

The hierarchical MPC-MARL coordinated control framework is illustrated in Fig. 5. The hierarchical architecture operates across three distinct temporal scales: (1) Macroscopic state evolution step $\triangle T$ : The fundamental time step at which the underlying L-METANET model updates its macroscopic state variables; (2) MPC control horizon $T _ { c }$ : The lower-frequency operational step at which the upper-level MPC baseline controller executes its rolling horizon optimization; (3) MARL execution cycle $T _ { s }$ : The higherfrequency sampling and actuation interval at which the lower-level MARL residual compensator performs adaptive fine-tuning. These three time scales satisfy the multi-frequency temporal relationship formulated in Eq. (27).

$$
T _ { c } = \lambda _ { 1 } \cdot T _ { s } = \lambda _ { 1 } \cdot \lambda _ { 2 } \cdot \triangle T\tag{27}
$$

where $\lambda _ { \mathrm { { \scriptscriptstyle 1 } } }$ and $\lambda _ { 2 }$ are both positive integers, satisfying: $T _ { c } > T _ { s } > \triangle T$

![](images/f80a69d50e5b687476e4bde3ff82045b91ba06054fbe664ddd6a1e5887d7f157.jpg)  
Fig. 5 The network architecture of MPC-STMAPPO

In the physical road network, control devices are deployed in a spatially discrete manner. Let $N _ { \scriptscriptstyle { V S L } }$ be the total number of lane-level VSL gantries, defining the VSL actuator set as $c \in \left. 1 , 2 , . . . , N _ { V S L } \right.$ , where each actuator maps to a specific segment-lane combination. Let $N _ { _ { R M } }$ be the total number of ramp meters, defining the RM actuator set as $g \in \left. 1 , 2 , . . . , N _ { R M } \right.$ , where each actuator maps to a specific onramp. Within the MPC framework, the state space and control inputs must be defined at the MPC control step . The macroscopic state vector at MPC step is defined as $\mathbf { X } _ { M P C } ( k )$ , which is physically equivalent to the output of the L-METANET model at the physical step $k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 }$ . This mapping relation is defined in Eq. (28). Similarly, for the MARL framework, the state space $\mathbf { X } _ { \mathit { M A R L } } ( k )$ at the high-frequency control step is physically equivalent to the output of the L-METANET model at the physical step $t \cdot \lambda _ { \scriptscriptstyle 2 }$ , as defined in Eq. (29).

$$
\mathbf { X } _ { \scriptscriptstyle M P C } ( k ) = \mathbf { X } _ { \scriptscriptstyle L - M E T A N E T } ( i ) \left. _ { i = k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } } \right.
$$

$$
\mathbf { X } _ { \mathit { M A R L } } ( t ) = \mathbf { X } _ { \mathit { L - M E T A N E T } } ( i ) \left| _ { i = t \cdot \lambda _ { 2 } } \right.\tag{28}
$$

(29)

Regarding the overall operational logic of the hierarchical coordinated architecture, at the beginning of each MPC step, the upper-level MPC solver predicts the system evolution within the prediction horizon via rolling optimization based on the current macroscopic traffic flow state combined with the L-METANET model, and issues a low-frequency fixed baseline control command $\mathbf { u } _ { b a s e } = [ \mathbf { v } _ { b a s e } , \mathbf { r } _ { b a s e } ] ^ { T }$ for the current step. Within each MARL step, the lower-level MARL agents deployed at each lane and ramp output high-frequency residual actions $\boldsymbol { \triangle } \mathbf { u } = \left[ \boldsymbol { \triangle } \mathbf { v } , \boldsymbol { \triangle } \mathbf { r } \right] ^ { T }$ online, according to the high-frequency sampled macroscopic state and the baseline control command determined by the MPC. The final issued control command is related to ${ \bf u } _ { b a s e }$ and $\triangle \mathbf { u }$ , but instead of a static superposition of the two, it is reconstructed as a piecewise chronological rolling cumulative generation law. Its action synthesis and truncation protection mechanism are shown in Eqs. (30) to (32). This control superposition mechanism mathematically ensures that when the MARL step exactly satisfies $t / \lambda _ { 1 } = k$ , it indicates that the control loop has transitioned to the MPC cycle boundary, at which time ${ \mathbf { u } } _ { r e f } ( t )$ is forcibly anchored to the latest global optimal solution ${ \bf u } _ { \mathrm { b a s e } } ( k )$ issued by the upper-level MPC; whereas when $t / \lambda _ { 1 } \neq k$ , it indicates that the system is in the intra-cycle rolling fine-tuning stage, and ${ \mathbf { u } } _ { r e f } ( t )$ seamlessly transitions to the actual physical execution command of the previous MARL step  . This superposition mechanism ensures that, on one hand, the MPC can correct and guide the MARL actions in every cycle, keeping the final issued values permanently within an acceptable range; on the other hand, the system eliminates discrete abrupt mutations in speed limits and metering rates within the dynamic execution structure, securing a smooth physical transition of the overall control boundary.

$$
\mathbf { u } ( t ) = \mathrm { C l i p } \big ( \mathbf { u } _ { r e f } ( t ) + \Delta \mathbf { u } ( t ) \big )\tag{30}
$$

$$
C l i p ( \mathbf { u } ) = \left\{ \begin{array} { l l } { \mathbf { u } _ { \mathrm { m a x } } , } & { u > u _ { \mathrm { m a x } } } \\ { \mathbf { u } _ { \mathrm { m i n } } , } & { u < u _ { \mathrm { m i n } } } \\ { \mathbf { u } , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{31}
$$

$$
\mathbf { u } _ { \mathrm { r e f } } ( t ) = \left\{ \mathbf { u } _ { \mathrm { b a s e } } ( k ) , \quad t / \lambda _ { 1 } = k \right.\tag{32}
$$

where u( )t is the physical increment vector transformed through spatial mapping from the continuous residual actions output by the MARL layer; $\mathbf { u } _ { \mathrm { m i n } } , \mathbf { u } _ { \mathrm { m a x } }$ are the legal boundary extreme vectors of the discrete physical actuators; and the upper-level low-frequency step index $k$ and the lower-level highfrequency step index satisfy the chronological rounding mapping relationship: $k = \left[ t / \lambda _ { 1 } \right]$

## 5.2. Upper-level baseline controller based on model predictive control

The core task of the upper-level MPC is to deduce the spatiotemporal evolution of traffic flow using L-METANET macroscopic dynamic equations. Combined with the RPBL dynamic crash probability equations constructed in Section 4.2, it establishes a baseline control boundary balancing safety and efficiency for the weaving segments from a global perspective over a long prediction horizon.

## 5.2.1 State-space and prediction model formulation

Defining Eq. (33) as the state space of the MPC, the upper-level baseline control vector ${ \bf u } _ { b a s e } ( k )$ represents the command set issued to all independent physical actuators, as shown in Eq. (34). In the MPC rolling optimization, since the issued baseline command ${ \bf u } _ { b a s e } ( k )$ remains constant throughout the entire low-frequency control cycle $T _ { c }$ , the underlying L-METANET dynamic model must absorb this constant command to continuously execute state updates at step size $\triangle T$ . By inputting the baseline traffic flow operational data of the expressway system and the environmental boundary demands $\mathbf { D } ( i )$ the improved L-METANET model predicts future traffic flow states across sampling periods. The state prediction transition equation is formulated as Eq. (35).

$$
\mathbf { X } _ { _ { M P C } } ( k ) = \left[ k _ { m , n } ^ { k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } } , \nu _ { m , n } ^ { k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } } , w _ { g } ^ { k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } } \right] ^ { T }\tag{33}
$$

$$
\mathbf { u } _ { b a s e } ( k ) = \left[ \nu _ { 1 } ^ { F S L } ( k ) , . . . , \nu _ { N _ { F S L } } ^ { F S L } ( k ) , . . . , r _ { N _ { R M } } ^ { R M } ( k ) \right] ^ { T }\tag{34}
$$

$$
\mathbf { X } _ { L - M E I A N E T } ( i + 1 ) = F _ { L - M E I M N E T } \left[ \mathbf { X } _ { L - M E I A N E T } ( i ) , \mathbf { u } _ { b a c e } ( k ) , \mathbf { D } ( i ) \right] , \ k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } \le i \le ( k + 1 ) \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } - 1\tag{35}
$$

where $\mathbf { X } _ { L - M E T A N E T } \left( i + 1 \right)$ is the predicted basic traffic flow state vector at sampling step  ; $\mathbf { X } _ { L - M E T A N E T } ( i )$ is the input basic traffic flow state vector at sampling step $i ; \mathrm { ~ \bf ~ D ( \it { i } ) }$ is the traffic boundary demand vector (arrival flows of the upstream mainline and ramps) at sampling step ; and $F _ { L - M E T A N E T } \left( \cdot \right)$ represents the mapping function of L-METANET.

In the MPC rolling optimization, the controller predicts forward from the current control step to $k + p + 1$ (where the prediction step is $p \in \left[ 0 , N _ { p } - 1 \right]$ and $N _ { p }$ is the MPC prediction horizon). Based on the zero-order hold principle, the upper-level baseline command remains constant throughout the entire low-frequency control cycle, while the underlying L-METANET model executes $\lambda _ { 1 } \cdot \lambda _ { 2 }$ physical iterative steps with a step size of $\triangle T$ within this cycle.

## 5.2.2 Multi-objective comprehensive cost function formulation

The core pain point of weaving segment management lies in the inherent trade-off between traffic efficiency and operational safety. To address this, the upper-level MPC constructs a comprehensive cost function $J _ { _ { M P C } }$ across the prediction horizon, as shown in Eq. (36).

$$
J _ { M P C } = \sum _ { i = k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } } ^ { ( k + N _ { p } ) \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } - 1 } \gamma _ { i } \Big [ \omega _ { 1 } \cdot C R ( i ) + \omega _ { 2 } \cdot J _ { e f f } ( i ) + \omega _ { 3 } \cdot J _ { q u e u e } ( i ) \Big ] + J _ { s m o o t h } ( k )\tag{36}
$$

$$
\gamma _ { i } = \frac { \displaystyle \exp \left( - \frac { i - k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } } { \eta _ { d } } \right) } { \displaystyle \sum _ { j = 0 } \sum _ { j = 0 } \nu _ { \mathrm { ~ d ~ } } \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } - 1 } \exp \left( - \frac { j } { \eta _ { d } } \right)\tag{37}
$$

where $\omega _ { 1 } , \omega _ { 2 }$ , and $\omega _ { 3 }$ are priority weights, and $J _ { \mathrm { p e n a l t y } } ( k )$ is the smoothing penalty term for temporal and spatial variations. Within the total prediction horizon, the normalized temporal discount coefficient $\gamma _ { i }$ at any prediction step is expressed as in Eq. (37), where $\eta _ { d }$ represents the temporal discount adjustment parameter reflecting the decay of predictive efficacy over the macroscopic prediction horizon. In Eq. (36), the first term represents the safety component, the second is the mainline efficiency component, the third denotes the on-ramp vehicle efficiency component, and the fourth constitutes the additional smoothing penalty for control execution. These four components are detailed below:

(1) XGBoost-SHAP-RPBL based merge and diverge risk penalty : The lane-level density and speed feature subsets deduced by the prediction model at time are directly substituted into the RPBL dynamic crash probability integral equation derived in Eq. (26). Through this cost term, the MPC optimizer gains risk-foreseeing capabilities, forcing it to proactively search for VSL and RM combinations that minimize merge and diverge conflicts.

(2) Mainline traffic efficiency delay cost $J _ { \it e f f } ( i )$ : The efficiency evaluation is restricted to the target macroscopic segments composed of the weaving area and its upstream sections (the set of all evaluation zones is denoted as ${ \mathcal { M } } _ { \mathrm { t a r g e t } } )$ . The average speed within the target segments is utilized as the efficiency metric, with the specific structure formulated in $\operatorname { E q }$ . (38).

(3) On-ramp queue delay cost $J _ { q u e u e } ( i )$ : The total queuing time on the on-ramps within the weaving segment is adopted as the ramp queue penalty, as shown in Eq. (39).

(4) Actuator discrete smoothing penalty $J _ { \mathit { s m o o t h } } ( k )$ : To prevent severe command jumps when spatially discrete actuators issue control signals, strict smoothing constraints must be imposed. Specifically, for VSL control, the system must restrict not only the temporal command discrepancy of a single speed limit gantry between adjacent time steps but also the spatial command discrepancy between adjacent gantries on the same lane at the same timestamp, thereby preventing rear-end collision risks induced by sudden spatial speed limit drops. To this end, let $\varepsilon _ { \ / { V S L } }$ be the set of spatially adjacent VSL actuator pairs on the same lane. If actuator $c "$ is physically located immediately downstream of actuator on the same lane, the pair $\left( c , c ^ { \prime } \right) \in \mathcal { E } _ { V S L }$ exists. A quadratic penalty for control increments is introduced across all independent discrete actuators within the control horizon, as shown in $\operatorname { E q } .$ . (40).

$$
J _ { _ { e f f } } ( i ) = - \frac { \displaystyle \sum _ { m \in \mathcal { M } _ { \mathrm { t a r g e t } } } \sum _ { n = 1 } ^ { N _ { m } } q _ { m , n } ( i ) } { \displaystyle \sum _ { m \in \mathcal { M } _ { \mathrm { t a r g e t } } } \sum _ { n = 1 } ^ { N _ { m } } k _ { m , n } ( i ) \cdot \nu _ { \mathrm { m a x } } + \epsilon }\tag{38}
$$

$$
J _ { _ { q u e u e } } ( i ) = \sum _ { g = 1 } ^ { N _ { _ { R M } } } w _ { o g } ( i ) \cdot \Delta T / 3 6 0 0\tag{39}
$$

$$
J _ { s m o o t h } ( k ) = \sum _ { p = 0 } ^ { N _ { c } - 1 } \left\{ \underset { c = 1 } { \overset { N _ { i : k } } { \sum _ { s = 1 } ^ { N _ { i : k } } } } \left[ \frac { \nu _ { c } ^ { \mathrm { F S L } } ( k + p ) - \nu _ { c } ^ { \mathrm { F S L } } ( k + p - 1 ) } { \nu _ { m a x } } \right] ^ { 2 } + \right.  \\  \left. \sum _ { ( c , c ) \in \mathcal { E } _ { \mathrm { N L } } } \left[ \frac { \nu _ { c } ^ { \mathrm { F S L } } ( k + p ) - \nu _ { c } ^ { \mathrm { F S L } } ( k + p ) } { \nu _ { m a x } } \right] ^ { 2 } + \sum _ { g = 1 } ^ { N _ { m i } } \left[ r _ { g } ^ { \mathrm { F M } } ( k + p ) - r _ { g } ^ { \mathrm { F M } } ( k + p - 1 ) \right] ^ { 2 } \right\}\tag{40}
$$

## 5.2.3 Traffic physical boundaries and control execution constraints

To ensure the physical feasibility and absolute safety of the control sequence solved throughout the prediction horizon of the upper-level MPC rolling optimization, a strict system of equality and inequality constraints must be imposed, encompassing the following aspects:

(1) Macroscopic traffic flow dynamics equality constraints. The evolution of all state vectors within the prediction horizon must strictly obey the conservation laws of L-METANET.

(2) VSL spatial execution boundary and control rate constraints. For any speed limit gantry $c \in \{ 1 , . . . , N _ { _ { V S L } } \}$ , its issued command must fall between the maximum speed limit $\nu _ { \mathrm { m a x } }$ allowed by the physical road network and the minimum safe driving speed $\nu _ { \mathrm { m i n } }$ , as shown in constraint (41). Meanwhile, to prevent rear-end collisions caused by sharp drops in speed limits, the speed limit variation within a single low-frequency step is restricted from exceeding the physical threshold $\Delta \nu _ { _ { s t e p } }$ (constraint (42)), and the command difference between adjacent upstream and downstream speed limit gantries on the same lane at the same prediction timestamp is restricted from exceeding the spatial safety threshold $\Delta \nu _ { _ { s p a c c } }$ (constraint (43)). Generally, the magnitudes of $\Delta \nu _ { s t e p }$ and $\Delta \nu _ { s p a c c }$ remain consistent.

(3) Queue constraints. At any underlying physical time step within the prediction horizon, the number of queuing vehicles not exceed the maximum physical storage capacity of the ramp, eliminating deadlocks and spillovers at the hard constraint level, as shown in constraint (44).

(4) RM rate fluctuation constraints. Sudden and large variations in ramp inflow will severely degrade the traffic efficiency of the weaving segment and elevate accident risks. Therefore, highfrequency fluctuations must be suppressed, as shown in constraint (45).

$$
\nu _ { m i n } \leq \nu _ { c } ^ { F S L } ( k + p ) \leq \nu _ { m a x } , \quad \forall p \in [ 0 , N _ { c } - 1 ]\tag{41}
$$

$$
\left| { \nu } _ { c } ^ { F S L } ( k + p ) - \nu _ { c } ^ { V S L } ( k + p - 1 ) \right| \leq \Delta \nu _ { s t e p } , \quad \forall c \in \{ 1 , . . . , N _ { t S L } \} , \forall p \in [ 0 , N _ { c } - 1 ]\tag{42}
$$

$$
\left| \nu _ { c } ^ { F S L } ( k + p ) - \nu _ { c } ^ { F S L } ( k + p ) \right| \leq \Delta \nu _ { s p a c c } , \quad \forall ( c , c ^ { \prime } ) \in \mathcal { E } _ { _ { V S L } } , \forall p \in [ 0 , N _ { c } - 1 ]\tag{43}
$$

$$
w _ { o _ { g } } ^ { i } \leq w _ { m a x } , \quad \forall i \in \left[ k \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } , ( k + N _ { p } ) \cdot \lambda _ { 1 } \cdot \lambda _ { 2 } \right]\tag{44}
$$

$$
\left| r _ { g } ^ { R M } ( k + p ) - r _ { g } ^ { R M } ( k + p - 1 ) \right| \le \Delta r _ { s t e p } , \forall p \in [ 0 , N _ { c } - 1 ]\tag{45}
$$

## 5.2.4 Model optimization and solution

At each low-frequency time step , the upper-level baseline controller aims to solve for an optimal control sequence $U ^ { * }$ that minimizes the multi-objective comprehensive cost function $J _ { _ { M P C } }$ (Eq. (36)) while strictly adhering to constraints (41) to (45). Because this optimization task constitutes a characteristically high-dimensional, non-linear, and non-convex optimization problem, conventional gradientbased deterministic solvers, such as sequential quadratic programming, are highly susceptible to becoming trapped in local optima and frequently fail to converge. To circumvent these limitations, this study employs a hybrid GA-VNS algorithm for optimization. Upon completing the hybrid optimization for the current step, the system extracts only the first element of the optimal control sequence $\mathbf { u } _ { b a s e } ( k ) = \mathbf { U } ^ { * } ( 1 )$ as the deterministic baseline command to be dispatched to the physical road network. At the next low-frequency control step $k + 1$ , the closed-loop rolling horizon optimization process is reinitialized by introducing the fresh lane-level macroscopic state feedback collected via roadside sensors.

## 5.3. Lower-level residual compensator based on multi-agent reinforcement learning

This study models the lower level as a fully cooperative multi-agent system. Within each low-frequency baseline cycle, agents distributed across each lane and ramp of the weaving segment continuously sample local macroscopic states at a higher frequency (step size $T _ { s } )$ and output continuous residual compensation actions online, thereby agilely smoothing model prediction mismatches and achieving adaptive closed-loop control of the macroscopic traffic flow.

In the complex spatiotemporal coupled road network of the weaving segment, distributed speed limit gantries and ramp signals cannot acquire the global state of the entire network and must make coordinated decisions based on local information. Therefore, this study rigorously models the lowerlevel residual compensation-based multi-agent coordinated control problem as a decentralized partially observable Markov decision process (Dec-POMDP). This system can be mathematically defined by a 7- tuple $\left( \mathcal { I } , \boldsymbol { S } , \mathcal { A } , \mathcal { P } , \boldsymbol { R } , \Omega , \gamma _ { d } \right)$ , with the specific mapping of each element under this framework as follows:

(1) Agent set : Consists of $N _ { \scriptscriptstyle { V S L } }$ variable speed limiters and $N _ { _ { R M } }$ ramp meters in the road network, with a total number of $N _ { a g e n t } { = } N _ { V S L } { + } N _ { R M }$ and the agent index as $j \in \mathcal { I }$ ;

(2) Global state space : Characterizes the true global macroscopic traffic flow state of the road network at any high-frequency step  .

(3) Set of partially observable state spaces for each agent : Agents cannot acquire the global $\boldsymbol { s }$ and can only acquire a local observation vector $o _ { j } ( t ) \in \Omega$ containing high-dimensional features.

(4) Joint action space  : The set of continuous residual actions $\Delta { \bf u } ( t )$ output by all agents at high-frequency step  .

(5) State transition probability $\mathcal { P }$ : Implicitly driven by the underlying simulation environment, which here refers to SUMO.

(6) Reward function $R \colon \mathrm { A }$ fully cooperative feedback signal that guides the agents to jointly compromise toward the global macroscopic optimum.

(7) Discount factor $\gamma _ { d } : ~ \gamma _ { d } \in [ 0 , 1 )$ , which measures the current importance of future spatiotemporal traffic flow rewards.

## 5.3.1 State-space formulation incorporating upper-level MPC outputs

The inputs of MARL require extensive feature engineering to enhance the Markov property of the environment. This study deeply extends the local observation space $o _ { j } ( t )$ of agent $j ,$ constructing it into a high-dimensional feature vector containing three core information matrices, as shown in Eq. (46). The global state space at step is then the concatenation of the local observation spaces of all agents, as shown in Eq. (47).

$$
o _ { j } ( t ) = \overline { { \big [ } { \mathbf { X } } _ { M A R L , j } ( t ) , { \mathbf { u } } _ { b a s e , j } ( k ) , { \mathbf { u } } _ { j } ( t - 1 ) \big ] }\tag{46}
$$

$$
\boldsymbol { S } ( t ) = \left[ o _ { 1 } ( t ) , . . . , o _ { j } ( t ) , . . . , o _ { N _ { a g e n t } } ( t ) \right]\tag{47}
$$

The specific definitions of the aforementioned observation features are as follows:

Current macroscopic physical state ${ \mathbf { X } } _ { M A R L , j } ( t )$ : Since weaving segment congestion exhibits a significant downstream-to-upstream propagation characteristic, to enable the deep neural network to precisely capture the causal topological relationships of traffic shock waves and maximize feature noise elimination, this study classifies the agent set into three categories based on the physical attributes of the actuators: inner-lane VSL agents $( \mathcal { I } _ { V S L } ^ { i n } )$ , outer-lane VSL agents $( \mathcal { I } _ { V S L } ^ { o u t } )$ , and on-ramp RM agents $( \mathcal { I } _ { R M } )$ constructing their states respectively, as shown in Eq. (48). For inner-lane VSL agents, the density, speed, and flow at three critical nodes—the VSL zone, acceleration zone, and weaving segment—are extracted. The outer lane is not only a severe conflict zone for longitudinal weaving but also directly faces the pressure of the merging flow from the on-ramp. Therefore, in addition to having a longitudinal threesegment perception capability completely symmetrical to that of the inner lane, the outer VSL agents must incorporate the observation of the queue storage on their connected on-ramp . The core logic of RM is to control the merging flow rate based on the residual capacity of the mainline outer lane to prevent mainline breakdown or ramp gridlock. Consequently, the RM agent focuses on the queue storage and high-frequency dynamic arrival demand of its own ramp, as well as the loading state of the connected mainline target merging lane. The representations of these macroscopic physical states can be observed as Eq. (48).

Upper-level baseline control output ${ \mathbf { u } } _ { b a s e , j } ( k )$ : The baseline command ${ \mathbf { u } } _ { b a s e , j } ( k )$ received by agent within its current $k ^ { t h }$ low-frequency control step, which is issued by the upper-level MPC rolling optimization. Consequently, the neural network can clearly identify the current macroscopic safety anchor, thereby avoiding ineffective exploration into invalid action spaces that violate physical bottom lines when generating high-frequency residuals action $\Delta \mathbf { u } _ { j } ( t )$ under the guidance of the MPC layer.

Previous total control input ${ \mathbf { u } } _ { j } ( t - 1 )$ : This feature extracts the final synthesized command actually applied to the road network by the agent at time t −1. In a Markov decision process, the current state of physical actuators is a crucial component of the environmental state. Feeding the action of the previous step as the observation input of the current step forces the Actor network to possess temporal memory. Coupled with the residual smoothing penalty in the reward function, this feature effectively guides the neural network to self-constrain its action span, preventing the output of fluctuating commands and ensuring a safe, smooth transition of vehicle operations from the algorithm input end.

Notably, all observation states are normalized, and padding zeros are applied to missing elements to maintain consistent state dimensions.

$$
\begin{array} { r l } & { \mathbf { X } _ { M H L , j } ( t ) = } \\ & { \left[ \left[ k _ { A Z , y j } ( t ) , \nu _ { A Z , y j } ( t ) , q _ { A Z , y j } ( t ) , k _ { m j , y j } ( t ) , \nu _ { m j , y j } ( t ) , q _ { m j , y j } ( t ) , k _ { W S , y j } ( t ) , \nu _ { W S , y j } ( t ) , q _ { W S , y j } ( t ) \right] ^ { T } , j \in \mathcal { T } _ { V S L } ^ { \mu } \right. } \\ & { \left. \left[ k _ { A Z , y j } ( t ) , \nu _ { A Z , y j } ( t ) , q _ { A Z , y j } ( t ) , k _ { m j , y j } ( t ) , \nu _ { m j , y j } ( t ) , q _ { m j , y j } ( t ) , k _ { W S , y j } ( t ) , \nu _ { W S , y j } ( t ) , q _ { W S , y j } ( t ) , w _ { o j } ( t ) \right] ^ { T } , j \in \mathcal { T } _ { F S L } ^ { o u t } \right. } \\ & { \left[ \left. w _ { o j } ( t ) , d _ { \varphi } ^ { i } ( t ) , k _ { \tilde { m } j , \tilde { y } } ( t ) , \nu _ { \tilde { m } j , \tilde { y } } ( t ) , \nu _ { \tilde { m } j , \tilde { y } } ( t ) , q _ { \tilde { m } j , \tilde { y } j } ( t ) \right] ^ { T } , j \in \mathcal { T } _ { R M } \right. } \end{array}\tag{48}
$$

where the subscript denotes the traffic flow parameters of the acceleration zone in the same lane n as agent ; denotes the traffic flow parameters of the same lane within the same segment as agent $j ; ~ W S , n j$ denotes the traffic flow parameters of the weaving segment in the same lane as agent $j ; ~ \tilde { m } j , \tilde { n } j$ denotes the traffic flow parameters of the segment and lane connected to agent ; and denotes the traffic flow parameters of the ramp where the RM agent is located.

## 5.3.2 Joint action space design

During the decentralized execution phase, all VSL and RM agents within the weaving segment jointly output decisions at each high-frequency time step  , forming the joint action space $\mathcal { A } ( t ) = \left[ \Delta \mathbf { u } _ { 1 } ( t ) , \Delta \mathbf { u } _ { 2 } ( t ) , . . . , \Delta \mathbf { u } _ { N _ { a g e n t } } ( t ) \right]$ . To guarantee the absolute safety bottom line of road network control, the MARL agents under the proposed framework do not directly output the final absolute control speed limits or absolute metering rates. Instead, agent outputs a continuous action vector bounded between , which is mapped into a physical correction value after denormalization. The core task of the agents is to perform high-frequency, small-scale fine-tuning on the basis of ${ \mathbf { u } } _ { r e f } ( t )$ , thereby agilely absorbing and smoothing out traffic shock wave disturbances triggered by model mismatches or sudden demand surges. Ultimately, the residual actions generated by the agents are superposed with the corresponding baseline commands to synthesize the theoretically optimal control law. Before being dispatched to the physical actuators, these synthesized commands uniformly pass through the underlying physical boundary truncation mechanism of the system, securing the operational safety bottom line of the weaving segment from the physical layer while granting MARL full exploration freedom.

## 5.3.3 Fully cooperative shared global reward function formulation

To prevent the multi-agent system from falling into destructive, self-interested local games, this system adopts a fully shared reward mechanism. At any high-frequency control step , all VSL and RM agents are assigned a completely identical global comprehensive instantaneous reward. This reward references the upper-level ${ \mathrm { M P C } } ,$ as shown in Eq. (49), and is jointly composed of four dimensions: weaving segment safety, mainline travel efficiency, weaving segment on-ramp queuing, and control smoothness (Eqs. (50) to (51)). Among these, the first three components are completely identical to the corresponding terms in the cost function of the MPC layer.

$$
\begin{array} { l l } { R _ { g l o b a l } ( t ) = R _ { j } ( t ) = } & \\ { \frac { 1 } { \lambda _ { 2 } } \displaystyle \sum _ { i = t : \lambda _ { 2 } } ^ { ( t + 1 ) \cdot \lambda _ { 2 } - 1 } \left[ - \omega _ { 1 } \cdot C R ( i ) - \omega _ { 2 } \cdot J _ { e f f } ( i ) - \omega _ { 3 } \cdot J _ { q u e u e } ( i ) \right] + \omega _ { s m o o t h } \cdot R _ { s m o o t h } ( t ) , } & { \forall j \in \mathcal { T } } \end{array}\tag{49}
$$

$$
\begin{array} { r } { R _ { s m o o t h } ( t ) = - \Bigg [ \sum _ { c \in \mathcal { T } _ { y s L } } \left( \frac { \Delta { \nu } _ { c } ^ { y S L } ( t ) - \Delta { \nu } _ { c } ^ { y S L } ( t - 1 ) } { \nu _ { m a x } } \right) ^ { 2 } + \sum _ { ( c , c ) \in \mathcal { E } _ { y s L } } \left[ \frac { \nu _ { c } ^ { y S L } ( t ) - \nu _ { c } ^ { y S L } ( t ) } { \nu _ { m a x } } \right] ^ { 2 } } \\ { + \sum _ { g \in \mathcal { T } _ { R M } } \left( \frac { \Delta r _ { g } ^ { R M } ( t ) - \Delta r _ { g } ^ { R M } ( t - 1 ) } { r _ { m a x } } \right) ^ { 2 } \Bigg ] } \end{array}\tag{50}
$$

$$
\nu _ { c } ^ { V S L } ( t ) = \nu _ { r e f , c } ^ { V S L } ( t ) + \Delta \nu _ { c } ^ { V S L } ( t )\tag{51}
$$

## 5.3.4 ST-MAPPO: MAPPO algorithm architecture with spatiotemporal attention mechanisms

Under the joint multi-lane multi-ramp control of weaving segments, the evolution of traffic congestion exhibits significant time lags and spatial causal traceability. Conventional DRL algorithms often struggle to effectively capture strongly coupled spatiotemporal features when processing such highdimensional, non-stationary traffic states. To address this, this study proposes a multi-agent proximal policy optimization algorithm with spatiotemporal attention mechanisms (ST-MAPPO) integrating the Mamba selective state-space model and graph causal topological attention (GCTA), applying it within a centralized training, decentralized execution (CTDE) architecture.

Distributed actor network with temporal memory. Distributed agents in the weaving segment can only obtain local observations $o _ { j } ( t )$ , positioning the system within a POMDP. To eliminate environmental non-Markovian properties, this framework reconstructs the Actor network structure by introducing Mamba blocks based on selective state spaces. At each high-frequency decision step , the agent inputs not only the current normalized $o _ { j } ( t )$ but also incorporates a temporally maintained hidden state memory vector $\hat { \textbf { h } } _ { j } ( t - 1 )$ . The forward propagation and action output distribution parameter computation of the Actor network are formulated as Eq. (52).

$$
\Big [ \mu _ { j } ( t ) , \sigma _ { j } ( t ) , \hat { \mathbf { h } } _ { j } ( t ) \Big ] = M a m b a C e l l \Big [ \circ _ { j } ( t ) , \hat { \mathbf { h } } _ { j } ( t - 1 ) \Big ]\tag{52}
$$

where MambaCell( ) represents the computational operator for parameter selective decay and state updates through time-varying control matrices; $\mu _ { j } ( t )$ and $\sigma _ { j } ( t )$ are the mean and standard deviation of the Gaussian distribution for the continuous residual actions output by agent , respectively; and $\hat { \mathbf { h } } _ { j } ( t )$ is the updated temporal hidden state vector. The agent ultimately samples the high-frequency residual action $\Delta \mathbf { u } _ { j } ( t ) \sim \mathcal { N } ( \mu _ { j } ( t ) , \sigma _ { j } ( t ) )$ from this Gaussian distribution.

Centralized critic network based on graph causal topological attention (GCTA). On the training side, the centralized Critic network evaluates the value of the global state matrix $S ( t )$ . To prevent fully connected structures from introducing spatially uncorrelated traffic flow noise, the Critic network incorporates a spatial causal topological edge index architecture. The underlying static graph topology $\mathcal { G } = ( \nu , \mathcal { E } )$ is constructed using the physical correlations of the weaving bottleneck segments. Where the node set represents the hardware actuators in the entire road network; the edge set binds the lane-level VSL gantry nodes and their adjacent on-ramp RM signal nodes within the same weaving bottleneck region into a locally fully connected spatial clique, with self-loop edges configured for all nodes. For any control node pair $( j , l ) \in \mathcal { E }$ with an edge connection relationship, their dynamic self-attention weight distribution coefficient $\alpha _ { j , l } ^ { h } ( t )$ under an independent attention head is solved as shown in Eq. (53). In the multi-head attention feature aggregation, a residual connection mechanism is introduced to ensure the training stability of the deep network, and the expression of the updated node feature $\mathbf { z } _ { j } ^ { ' } ( t )$ is shown in Eq. (54).

$$
\alpha _ { j , l } ^ { h } ( t ) = \frac { \exp \Big ( \mathbf { a } _ { h } ^ { T } \left( \mathrm { L e a k y R e L U } \big ( \mathbf { W } _ { \mathrm { s r e } } ^ { h } \cdot \mathbf { z } _ { j } ( t ) + \mathbf { W } _ { \mathrm { d s t } } ^ { h } \cdot \mathbf { z } _ { l } ( t ) \big ) \right) \Big ) } { \displaystyle \sum _ { m \in \mathcal { N } _ { j } \cup \{ j \} } \exp \Big ( \mathbf { a } _ { h } ^ { T } \left( \mathrm { L e a k y R e L U } \big ( \mathbf { W } _ { \mathrm { s r e } } ^ { h } \cdot \mathbf { z } _ { j } ( t ) + \mathbf { W } _ { \mathrm { d s t } } ^ { h } \cdot \mathbf { z } _ { m } ( t ) \big ) \right) \Big ) }\tag{53}
$$

$$
\dot { \mathbf { z } _ { j } ^ { ' } } ( t ) = \mathrm { L a y e r N o r m } \Bigg ( \prod _ { h = 1 } ^ { H } \sum _ { l \in \mathcal { N } _ { j } \cup \{ j \} } \boldsymbol { \alpha } _ { j , l } ^ { h } ( t ) \cdot \mathbf { W } _ { \mathrm { s r c } } ^ { h } \mathbf { z } _ { l } ( t ) \Bigg ) + \mathbf { W } _ { \mathrm { r e s } } \cdot \mathbf { z } _ { j } ( t )\tag{54}
$$

where ${ \bf z } _ { j } ( t )$ is the node feature vector of the current node transformed by the pre-encoding layer; $\mathbf { W } _ { \mathrm { s r c } } ^ { h }$ and $\mathbf { W } _ { \mathrm { d s t } } ^ { h }$ are the attention transformation projection matrices; $\mathbf { a } _ { h } ^ { T }$ is the learnable attention vector; and $\mathcal { N } _ { j }$ is the neighborhood set of node $j .$ . is the layer normalization operator; and $\mathbf { W } _ { \mathrm { r e s } }$ is the linear projection matrix for the residual connection.

The Critic network ultimately outputs the global state value estimate $V _ { \phi } [ S ( t ) ]$ through a global average pooling layer and a multi-layer perceptron (MLP), as shown in Eq. (55).

$$
V _ { \phi } [ S ( t ) ] = \mathrm { M L P } \left( \frac { 1 } { N _ { \mathrm { a g e n t } } } \sum _ { j \in \mathcal { V } } \dot { \mathbf { z } _ { j } } ( t ) \right)\tag{55}
$$

In addition to the network structure, the following operations are implemented:

(1) Generalized advantage estimation (GAE) based on shared rewards. The attention-enhanced Critic network utilizes the fully shared global reward $R _ { g l o b a l } ( t )$ to calculate the temporal difference error $\delta ( t )$ . Combined with the future reward discount factor $\gamma _ { d }$ and the GAE smoothing parameter $\lambda _ { G A E }$ , the global advantage function $\hat { \boldsymbol A } ( t )$ is computed, as shown in Eqs. (56) to (57). Since all agents share the same global advantage signal $\hat { \boldsymbol A } ( t )$ , the individual Actor networks are forced to align during updates, iteratively updating weights uniformly toward maximizing the macro-coordinated benefits of the entire weaving segment road network.

(2) Clipped update mechanism and loss function design. To ensure smooth convergence of policy updates and avoid severe oscillations in the road network caused by excessive residual exploration, the ST-MAPPO algorithm introduces an importance sampling ratio $\rho _ { j } ( t )$ and a proximal policy clipping mechanism into the Actor network. The loss function of the Actor is given in Eqs. (58) to (59). S[ ] is the policy entropy regularization term (with coefficient $\beta )$ , encouraging agents to fully explore the boundaries of high-frequency residual actions during early training. The Critic loss function employs a mean squared error mechanism to approximate the true return value, as shown in Eq. (58).

(3) Implementation of other training tricks. First, sequential time-slice experience buffer: To preserve the continuous temporal dependencies required by the Mamba selective state-space model, trajectory replay avoids random shuffling. Instead, mini-batches are strictly partitioned by chronological time slices to facilitate sequential deductive learning. Second, dynamic feature and advantage normalization: Input state features are adaptively scaled using a moving average operator to accelerate optimization. Simultaneously, advantage functions undergo batch normalization within each mini-batch to maintain a standard distribution, stabilizing gradients against abrupt traffic flow fluctuations. Third, orthogonal initialization and linear decay: Network hidden weights are orthogonally initialized. Furthermore, the learning rates of the Actor and Critic networks, alongside the policy entropy coefficient, decay linearly across training epochs to balance aggressive early-stage exploration with stable late-stage convergence.

$$
\delta ( t ) = R _ { g l o b a l } ( t ) + \gamma _ { d } \cdot V _ { \phi } \big [ S ( t + 1 ) \big ] - V _ { \phi } \big [ S ( t ) \big ]\tag{56}
$$

$$
\hat { \boldsymbol A } ( t ) = \sum _ { l = 0 } ^ { T _ { \mathrm { m a x } } - t - 1 } ( \gamma _ { d } \cdot \lambda _ { G A E } ) ^ { l } \cdot \delta ( t + l )\tag{57}
$$

$$
\rho _ { j } ( t ) = \frac { \pi _ { \theta j } \left[ \Delta \mathbf { u } _ { j } ( t ) \mid o _ { j } ( t ) , \mathbf { h } _ { j } ( t - 1 ) \right] } { \pi _ { \theta j , o l d } \left[ \Delta \mathbf { u } _ { j } ( t ) \mid o _ { j } ( t ) , \mathbf { h } _ { j } ( t - 1 ) \right] }\tag{58}
$$

$$
\begin{array} { r } { L ( \theta _ { j } ) = - \mathbb { E } \Big \{ \operatorname* { m i n } \Big [ \rho _ { j } ( t ) \cdot \hat { A } ( t ) , \mathrm { C l i p } \big ( \rho _ { j } ( t ) , 1 - \epsilon , 1 + \epsilon \big ) \cdot \hat { A } ( t ) \Big ] \Big \} - \zeta \cdot \mathcal { H } ( \pi _ { \theta _ { j } } ) } \end{array}\tag{59}
$$

$$
L ( \phi ) = \mathbb { E } _ { t } \left\{ \operatorname* { m a x } \left[ \left( V _ { \phi } ( S ( t ) ) - R _ { t } ^ { \mathrm { t a r g e t } } \right) ^ { 2 } , \left( \mathrm { C l i p } ( V _ { \phi } ( S ( t ) ) ) - R _ { t } ^ { \mathrm { t a r g e t } } \right) ^ { 2 } \right] \right\}\tag{60}
$$

where  is the policy clipping threshold; $\mathcal { H } ( \pi _ { \theta _ { j } } )$ is the policy entropy regularization term, whose coefficient $\zeta$ linearly decays with the training process to maintain control activity during the early phase of exploration. The Critic loss function approximates the true return target value $R _ { t } ^ { \mathrm { t a r g e t } }$ via a dualclipped mean squared error to maintain the robustness of value function estimation over a long horizon.

## 6. Simulation experiments and performance verification

## 6.1. Experimental setup

To systematically validate the fidelity and control efficacy of the proposed macroscopic traffic flow model (L-METANET), crash risk assessment model (XGBoost-SHAP-RPBL), and hierarchical coordinated control framework (MPC-STMAPPO) under realistic and complex conditions, a high-fidelity macro-micro fused traffic simulation platform was constructed. The platform replicates an 18 km network topology using multi-source detector data from the Eastern Expressway in Changchun. This section details the experimental environment across three dimensions: simulation platform development, multi-source heterogeneous data composition, and hierarchical control parameter configurations.

Simulation platform development and network topology replication: The simulation environment is built on the open-source microscopic platform SUMO. To circumvent the TCP latency of standard TraCI and support high-frequency MARL interactions, the Python control algorithms interface directly with the SUMO engine via the libsumo C++ API. As illustrated in Fig. 6, the network replicates the topology of the 18 km Changchun Eastern Expressway at a 1:1 scale, encompassing mainline segments, on/off ramps, and weaving bottlenecks. The testbed features 9 VSL gantries and 3 RMs. Virtual loop (E1) and area (E2) detectors are positioned across all critical nodes (mainline origins, ramps) and within individual segment-lane cells to ensure full-state network perception.

Multi-source heterogeneous data and scenario configurations: empirical multi-source heterogeneous traffic data drive the calibration and evaluation tasks across different modules through the following data pipelines: 1) L-METANET Calibration (Section 6.2): High-frequency corridor volumes collected from 06:00 to 19:00 on November 10, 2022, covering all mainline cross-sections and ramps, establish the dynamic boundary conditions in SUMO (illustrated in Appendix B). This demand is directly input into SUMO as boundary conditions. Under conditions without management intervention, the natural driving data output by the SUMO are extracted as the ground truth to calibrate the L-METANET parameters; 2) Weaving segment risk assessment model fitting (Section 6.3): High-resolution vehicle trajectories extracted from UAV aerial videos at weaving segments WS3, WS4, and WS6 are paired with the STRF model to generate crash conflict labels, providing the microscopic data baseline to train the macroscopic XGBoost-SHAP-RPBL risk prediction model; 3) Hierarchical coordinated control framework performance evaluation (Section 6.4): The identical 13-hour empirical flow acts as the network boundary demand. Operational metrics under various control algorithms are evaluated to quantify the control efficacy of the proposed framework; 4) Transferability analysis (Section 0): The 13-hour baseline flow is adapted into five distinct synthetic demand scenarios to evaluate the framework's generalization resilience under unobserved traffic variations.

![](images/2d83f04b5a2ff613dd2511fad86c8022926bffed197698dd09c446737ac3901a.jpg)  
Fig. 6 Simulation settings of SUMO

Hierarchical time scales and hyperparameter configurations: (1) To simulate a mixed traffic environment during the transition toward CAVs, the CAV penetration rate is set to $\alpha { = } 0 . 5$ , and the spontaneous compliance rate of HDVs is set to $\beta = 0 . 8 . ( 2 )$ The simulation step of the underlying SUMO engine is , while the prediction step of L-METANET is set to $\Delta T = 5 s$ . Within the hierarchical control framework, the high-frequency sampling and action execution interval of the lower-level ST-MAPPO agents is $T _ { s } ~ = ~ 1 2 0 s$ , and the update cycle of the upper-level MPC is $T _ { c } = 6 0 0 s$ with a rolling prediction horizon of $N _ { p } = 3$ . (3) To guarantee system safety and prevent the MARL agents from issuing catastrophic commands during the exploration phase, multi-dimensional physical constraints and penalty mechanisms are imposed on the action space: The absolute speed limit range of the VSL is restricted to [40, 80] ${ \mathrm { k m / h } } ,$ with the maximum speed jump between adjacent control steps limited to $\Delta \nu _ { _ { s t e p } } = 2 0$ km/h. The metering rate of the RM is bounded within , with a maximum step change of $\Delta r _ { \mathrm { { s t e p } } } = 0 . 3$ Determined through extensive preliminary experiments, the weight allocation for the cost and reward functions serves a dual purpose. First, it normalizes the disparate physical units and mathematical scales, preventing any single indicator from dominating the optimization. Second, it balances the competing objectives of efficiency and safety to ensure control decisions converge toward a Pareto optimal point. Ultimately, $\omega _ { 1 } , \omega _ { 2 } , \omega _ { 3 }$ are set to a 1:5:0.2 ratio. Meanwhile, the global feedback reward is uniformly scaled to the order of before being fed into the network, preventing gradient explosion in the Critic network when encountering severe penalties. To address the variance collapse problem common to the PPO algorithm in continuous control tasks, the log of the initial standard deviation output by the Actor network of ST-MAPPO is restricted to -1.0 to suppress abrupt action fluctuations during early exploration. The initial learning rates for the Actor and Critic networks are set to $6 \times 1 0 ^ { - 4 }$ and $1 \times 1 0 ^ { - 4 }$ respectively, both of which employ a linear decay mechanism toward a final value of $1 \times 1 0 ^ { - 5 }$

## 6.2. Calibration and validation of L-METANET

To validate the fidelity of the proposed L-METANET model in real-world weaving segments, an empirical analysis is conducted using high-frequency detector data from the Changchun expressway network. Real-time flow from the Eastern Expressway serves as dynamic boundary conditions in the SUMO simulation environment. Data generated by virtual detectors within SUMO are subsequently utilized to calibrate both the L-METANET and METANET models through the following steps:

Step 1: Based on a road network exported from OpenStreetMap, the Eastern Expressway network is fully replicated in SUMO at a 1:1 scale. This process strictly maps the physical topology, including lane configurations, weaving segment lengths, and the geometric characteristics of entrances and exits.

Step 2: The Eastern Expressway is partitioned into multiple macroscopic physical segments and specific lanes. The partitioning criteria dictate: (1) ensuring absolute geometric homogeneity within each sub-segment, meaning no lane additions/drops or ramp variations occur internally; (2) establishing independent physical boundary cut-off points at critical nodes featuring on-ramp merging, off-ramp diverging, or weaving bottlenecks; and (3) strictly constraining and uniforming the lower bound of physical segment lengths to satisfy the Courant–Friedrichs–Lewy numerical stability condition.

Step 3: Virtual detectors are systematically deployed. E2 detectors are deployed within each discretized segment-lane cell to gather lane-level density, speed, and flow data. Concurrently, E1 detectors are installed at all mainline origin/destination points and ramp terminals to record the vehicle arrival and departure flow rates for boundary conditions.

Step 4: The obtained real-time traffic volumes are input as boundary demands into the SUMO simulator to model the spatiotemporal evolution of the mixed traffic flow within the weaving network.

Step 5: Data collection and cleaning are performed on the E1 and E2 detector outputs. Vehicle trajectories are aggregated into macroscopic statistical indicators within discrete time steps, serving as the boundary input drivers and calibration ground truth for both L-METANET and conventional METANET.

Step 6: Prediction models for L-METANET and METANET are executed independently. Using the cleaned E1 data as physical boundary conditions and the E2 data as fitting ground truths, the parameter sets for both models are strictly calibrated utilizing the GA-VNS algorithm detailed in Section 4.2.1. The calibrated parameter values for the L-METANET and METANET are summarized in Table 1.

Step 7: Post-calibration, 5-minute aggregated flow and speed data are randomly sampled across different segment-lanes. The outputs of both prediction models are compared against the SUMO ground truth to systematically verify the validity and high fidelity of the constructed models.

![](images/8166d9873aa111b2e7d41316b760ddf91bd93928fe5e2818563d5a0e89439362.jpg)

![](images/5f2eab1e05a8cf918fdd4c9ff3527386caffcf0015321e330d8143ee739605f3.jpg)

![](images/7cd51fb1bc1cde9c867fb0237437330adeaffbe7a6a3d6226f989b49a3121e07.jpg)

![](images/d380f8d33abf16f384dd6701ad0c38f0f1a6e3d071f634d0605ea5fb3cf9fc8b.jpg)

![](images/13d2b6d3c7b9e2392ea1b3d7d85e1aa377f5ecfce005397c6cd794d3ba0db9ec.jpg)

![](images/abfe339d877372a9f385251121aba0eb26670cb1ee2669eb5ff1b309853266f5.jpg)

![](images/c2e9c02c0406ff76e8edbdfcf54a75c64351b41a4fc42a789747a3cb0679b701.jpg)

![](images/7f27e1298ea3e484eb2858ae3b97d3118d3c2e6f436486adc5383cae3b55e2d3.jpg)

![](images/0700afc50353b64357ba7631b81b0150f59adb56ca97d29d3ed7a5ff741eaa53.jpg)

![](images/e27a68524353732cf7a79b6c984eef33d6a9bd2c04b2e3744755eda0d0ac9c39.jpg)

![](images/5b678181c52f98b7346641362367050295ed8dc069749a1f71a70480585cb707.jpg)

![](images/61d99e1a62e35df73952e7af86115cdacec683fa82f1df4bf281bf43cbc3ef24.jpg)

![](images/49ae1df6251fead625314724f85da7341755e71a516f22d6d35691142156aa97.jpg)

![](images/72a296cc8837f213e4d1fb1f96232183b3f7fc86f32cbab2ee3b3044c05ff8cc.jpg)

![](images/8b1677e3071db804693757f4282a8af1c9ea7044479df058a05fd0ec8d89164c.jpg)

![](images/c5008df2d841fe0e7ca7500f281e8a28803d36e7197cf2cef25a22c91c1e1902.jpg)

![](images/54eb27b0a92fbf45091cee69db632e3c3df0fbc7974cf1119003ba2fa3dd4fc0.jpg)

![](images/9d3f125af16723394d44a41326610bf103aebc411b2a38f5da37e097902fdad6.jpg)

![](images/e4ae70d82845ea95989b31bffb9d35f811d3fd385da017b7cf439ea400a5e6c5.jpg)

Fig. 7 Comparison of simulated flow for different macroscopic traffic flow models  
![](images/4011cf50b610482390a3c55e6248421545c3e5f35ab44707ec5038c4b71e939a.jpg)

![](images/27fe56393e13ef62972a99c8a239f369cc73900c966442ec59fd99614fd26ade.jpg)

![](images/f6759ca7eaee98026740e3ee167edbc6bc7ef3e394be50817c983243c5f28d84.jpg)

![](images/bd76b55a53fad92e2a19543ca4066cac0cfa193bdd3ced700e6dafcf58009bbe.jpg)

![](images/3f778190d29a40a147ecff327d6393c17918012796222e94c2d124b985bdb434.jpg)

![](images/5d015a8be429965001495075b87e70e3cf94cffd682b08b404704a59c6bdb74b.jpg)

![](images/dac51ed024b428a20e4512e0fc8e1c8a7d31b5c54f580c8d6366720625cb74de.jpg)

![](images/735bf85a82de3ce0923eae100b03d9465958bdc9c614e384b45ccacd91ead73a.jpg)

![](images/1667ad06ba70b4d160066e1718d1f204c3011f36163e9fe7fd09007f56b1a93c.jpg)

![](images/1c3dacb3430cff59bb93e44b005f57e516d92146fb6ce8e3f67badf97ecd7793.jpg)

![](images/79772137e84740b71a77ecc1d5c8f66b77b8def4a6e949b57849cf21711d0793.jpg)

![](images/06b76b94d56c5dd294fade44e51474a11c04c92641c37f9b497614b109c9d2e8.jpg)

![](images/8730aa42843148e7dc8df25fe7788bbc24afc679d8e17893449e44aad2d7ca9f.jpg)  
06:00 06:30 07:00 07:30 08:00 08:30

![](images/0160a6a0d600b8fedcc45f71c88ec98ab2849f3fdfd20f270a7f86535931d0fd.jpg)  
06:00 07:00 08:00 09:00 10:00 11:00

![](images/6d4e2125ef50e76226c37ae622549c508e1f11bb3a82b84dc97a5446de4f592b.jpg)

![](images/a0493e7c5a7449a3ee78ad5e1a2a60f4178ea4bff3bd65f08c554e21b0bf066f.jpg)  
Time

![](images/ccda63d4634ad7d7376e2de3dc690f18c79733da0f1e770f468dcc2451a0916c.jpg)

![](images/bc5aa947a068b3fdb2588e5524dbbe816dd5b1ec805e37e88d751109788067ab.jpg)  
06:00 07:00 08:00 09:00 10:00 11:00

![](images/43011ad1c9aa16e3a8714e7b67dc3c4e96e11fcc64514503ef77f5691317cbdf.jpg)

![](images/dd7409a0d266e21f2a8fdbbe75214da7034304ffa6830b3541a5ef931a827dee.jpg)

![](images/5d091b0fb373085616fda0adaa3bec3b7dc49a369addc402e5bc1a5725c07d36.jpg)  
06:00 07:00 08:00 09:00 10:00 11:00

Fig. 8 Comparison of simulated speed for different macroscopic traffic flow models

Table 1 Calibrated parameter values of L-METANET and METANET
<table><tr><td>Model</td><td>τ</td><td>η</td><td>K</td><td> $\delta ^ { i n }$ </td><td> $\delta ^ { o u t }$ </td><td> $\mu _ { C A V }$ </td><td> $\mu _ { H D V }$ </td><td> $\nu _ { f }$ </td><td> $k _ { c r }$ </td><td>a</td></tr><tr><td>L-METANET</td><td>0.0277</td><td>12.78</td><td>13.8</td><td>0.59</td><td>0.02</td><td>0.52</td><td>0.32</td><td>74.17</td><td>41.59</td><td>1.49</td></tr><tr><td>METANET</td><td>0.0200</td><td>28.54</td><td>21.30</td><td>–</td><td>I</td><td>–</td><td>–</td><td>72.89</td><td>22.3</td><td>3.37</td></tr></table>

Table 2 Average error of L-METANET and METANET
<table><tr><td rowspan="2">Model</td><td rowspan="2">Indicator</td><td colspan="2">Speed</td><td colspan="2">Flow</td></tr><tr><td>Lane 1</td><td>Other lanes</td><td>Lane 1</td><td>Other lanes</td></tr><tr><td>L-METANET</td><td>RMSE MAPE</td><td>4.25</td><td>8.98</td><td>158</td><td>133</td></tr><tr><td>METANET</td><td>RMSE MAPE</td><td>6.7% 7.04 10.8%</td><td>19.4% 11.51 27.5%</td><td>12.0% 253 19.5%</td><td>12.1% 256 24.9%</td></tr></table>

Fig. 7 and Fig. 8 show the comparison results, and Table 2 summarizes the average errors of the two models, from which several key conclusions can be drawn:

(1) Flow evolution (Fig. 7): While both models generally track the macroscopic fluctuations of the total network throughput, L-METANET exhibits a significantly more precise dynamic response during peak-to-valley transition periods. This advantage is particularly evident in outer weaving lanes heavily impacted by ramp merging and diverging. Due to its cross-sectional homogeneity assumption, conventional METANET uniformly distributes traffic across lanes, thereby passively smoothing localized lane-level lateral flow variations. As shown in Fig. 7(17) and Fig. 7(18), METANET generates identical flow profiles for Lane 2 and Lane 3 of Segment 25, inevitably leading to distortion. In contrast, L-METANET accurately reproduces the lateral flow redistribution and sudden capacity drops induced by forced lane-changing.

(2) Speed evolution (Fig. 8): The structural theoretical advantages of L-METANET are further confirmed. During morning and evening peak hours within the weaving segments, frequent merging and diverging maneuvers induce severe lateral asymmetric friction. This phenomenon triggers sharp speed drops in the outer and adjacent lanes, accompanied by high-frequency stopand-go shock waves. As illustrated in Fig. 8, because conventional METANET omits lane heterogeneity and lateral vehicle physics, it misestimates traffic platoon speeds under varying traffic states, which is highly visible in regions characterized by volatile temporal fluctuations, such as Fig. 8(13), (17), (18), and (20). Conversely, the L-METANET model relying on space allocation mechanisms for free and forced lane-changing and asymmetric speed penalties, accurately replicates the nonlinear dynamic transition from free-flow states to congestion shock waves. L-METANET achieves high alignment with the ground truth across both the disturbance-resistant, high-speed operations of inner lanes and the complex turbulent flows of outer lanes.

(3) Quantitative evaluation of average errors (Table 2): Across all evaluation metrics, the prediction accuracy of L-METANET consistently outperforms the METANET. Specifically, regarding speed prediction, L-METANET yields Mean Absolute Percentage Errors (MAPEs) as low as 6.7% for Lane 1 and 19.4% for other lanes, whereas the conventional METANET exhibits significantly higher errors at 10.8% and 27.5%, respectively. For flow prediction, this accuracy advantage is particularly prominent in other lanes, where the flow MAPE drops sharply from 24.9% under METANET to 12.1% under L-METANET, while the corresponding RMSE decreases from 256 to 133, nearly halving the modeling error. These statistical results strongly demonstrate the critical role of explicitly incorporating lane heterogeneity and lane-changing space allocation mechanisms, which effectively eliminates conventional cross-sectional homogeneity distortions and substantially enhances lane-level traffic flow modeling precision.

These results demonstrate that L-METANET possesses a high-fidelity capability to capture complex, nonlinear physical phenomena of multi-stream weaving alongside sudden capacity drops. This establishes a reliable predictive foundation for subsequent closed-loop coordinated VSL-RM control.

## 6.3. Fitting and performance verification of XGBoost-SHAP-RPBL

Following the steps described in Section 4.2, the analytical equations for risk assessment and prediction across all tasks were derived, as detailed in Appendix C. The underlying data originate from our previous study involving 14 tasks across 7 weaving segments for both merging and diverging risk types (Ma et al., 2026a). This comprehensive dataset encompasses the six specific tasks evaluated in this study: WS3-M, WS3-D, WS4-M, WS4-D, WS6-M, and WS6-D.

![](images/676e3c0b4338c7da2936428b17e4299f3fdb5d0a600457ac1f82fa2109756ed2.jpg)  
Fig. 9 ROC curves of the XGBoost-SHAP-RPBL

Fig. 9 illustrates the ROC curves of the proposed model across the 14 risk identification tasks. All tasks achieve an AUC above 0.70, demonstrating robust classification capabilities. Except for a few tasks (WS1\_D and WS3\_D), the AUC values exceed 0.80, with the merging risk task at WS6 reaching a peak AUC of 0.889. This performance confirms that the core feature set selected via XGBoost-SHAP accurately captures the deep non-linear mapping mechanisms between macroscopic traffic flow parameters and microscopic conflict risks, thereby ensuring high predictive accuracy and generalization robustness.

To further evaluate the performance of the XGBoost-SHAP-RPBL framework, comparative experiments were conducted against multiple baselines, as illustrated in Fig. 10. In addition to the proposed model, five representative baselines were introduced for comprehensive comparison: (1) A hybrid model considering only fixed effects (XGBoost-SHAP-FPBL); (2) A random parameters model with features selected based on Pearson correlation (Pearson-RPBL); (3) A random parameters model with features selected based on Spearman correlation (Spearman-RPBL); (4) A random parameters model retaining the full set of candidate features (All Params-RPBL); (5) A baseline model with randomly selected feature inputs (Random-RPBL). Four representative tasks (WS1-M, WS3-M, WS6-D, and WS7-M) were randomly selected for analysis. The comparison of their ROC curves demonstrates the following insights:

(1) Superiority of the XGBoost-SHAP interpretability framework. The proposed XGBoost-SHAP-RPBL model significantly outperforms the Pearson-RPBL, Spearman-RPBL, and Random-RPBL models. Unlike conventional feature selection methods that rely on linear correlation assumptions, the SHAP-based interpretability framework captures high-order synergetic interactions among variables, effectively eliminating the interference of redundant features and severe multicollinearity. This feature extraction mechanism enables the model to achieve high predictive accuracy within a highly parsimonious feature space that retains only key physical variables.

(2) Capability of random parameters to capture unobserved heterogeneity. The predictive accuracy of the proposed model slightly surpasses that of the XGBoost-SHAP-FPBL baseline. Although the performance gain is incremental, it underscores the necessity of accounting for unobserved heterogeneity in traffic safety modeling. By introducing normally distributed random parameters, the RPBL model effectively absorbs utility fluctuations stemming from individual driver differences, microscopic behavioral stochasticity, and environmental noise, thereby yielding superior probabilistic fitting performance from a statistical perspective.

(3) Dynamic trade-off between accuracy and controllability: Although the proposed model’s absolute accuracy is slightly lower than the All Params-RPBL baseline in certain scenarios, its predictive performance remains highly acceptable. Crucially, the proposed framework exhibits superior practical engineering value. The baseline incorporates dozens of candidate variables, causing it to suffer from the curse of dimensionality, an impractical data collection threshold, and slow online optimization convergence. Conversely, the proposed framework leverages SHAP selection to compress input dimensionality to single digits, enabling real-time risk estimation and closed-loop control feedback.

In summary, the XGBoost-SHAP-RPBL model optimally balances risk accuracy with operational controllability. This establishes a rigorous foundation for embedding the analytical risk function into the hierarchical, closed-loop MPC-MARL framework.

![](images/9d2ee088a7036e4f1a603a0762f487dc42e037f38e6c57ac89220871a5a57d2a.jpg)

![](images/91569d09ceb353d92985934f961fcaaebec8fd27e8d396da5f971553471c72ff.jpg)

![](images/a9bba50bf7756ab2f3f9cd786e754f8f9a9ea7be669516f37d4671afd673e72c.jpg)

![](images/63538eb372ebf89b4c21c99dc6e22418928fee4f8e2828c60b5b3ed507e3a60f.jpg)  
Fig. 10 Comparison of ROC curves for different algorithms

## 6.4. Performance verification of the MPC-STMAPPO hierarchical coordinated control framework

## 6.4.1 Baseline control strategies

To implement a rigorous controlled variable analysis, this study designs 11 baseline comparison schemes. These strategies comprehensively cover various architectural paradigms, including pure model-driven, pure data-driven, hybrid model-data-driven, and other reference frameworks:

Type 1: Pure model-driven frameworks. This category comprises three control schemes: (1) MPC + L-METANET + Lane-level (MPC-LL): Utilizes the L-METANET model and an MPC controller to execute LVSL and RM. This represents the proposed upper-level baseline MPC controller but operates without the lower-level MARL fine-tuning; (2) MPC + L-METANET + Segment-level (MPC-LS): Utilizes the L-METANET model and an MPC controller to execute segment-level VSL and RM. Lane-level predicted states from L-METANET are aggregated into segment-level metrics before executing segment-level VSL control; (3) MPC + METANET + Segment-level (MPC-SS): Utilizes the conventional METANET model and an MPC controller to execute segment-level VSL and RM.

Type 2: Pure data-driven frameworks. This category comprises three MARL schemes: (1) Pure ST-MAPPO: Implements MARL using the proposed lower-level MARL residual compensator, operating independently without upper-level MPC guidance; (2) Pure MAPPO: Employs the standard MAPPO algorithm, where the proposed Mamba-embedded policy network and GCTA-embedded Critic network are replaced by conventional multilayer perceptron (MLP) architectures; (3) Pure MADDPG: Implements the multi-agent deep deterministic policy gradient (MADDPG) algorithm to analyze the influence of alternative MARL paradigms on training efficacy and control results.

Type 3: Hybrid model-data-driven frameworks. This category evaluates different combinations within the hierarchical structure: (1) Proposed MPC + ST-MAPPO Framework (MPC-STMAPPO): The integrated framework features the upper-level MPC baseline controller and the lower-level ST-MAPPO residual compensator; (2) MPC + MAPPO (MPC-MAPPO): Combines the proposed MPC baseline controller with a standard MAPPO-based residual compensator; (3) MPC + MADDPG (MPC-MADDPG): Combines the proposed MPC baseline controller with a MADDPG-based residual compensator.

Type 4: Baseline benchmarks. This category includes two strategies: (1) No-control: Vehicles navigate naturally within SUMO based on empirical boundary inputs. VSL and RM values are maintained at their maximum allowable limits; (2) Random control: VSL and RM actuation commands are generated randomly within their respective physical upper and lower bounds.

## 6.4.2 Training process and convergence mechanism analysis

To evaluate the internal evolution, training efficiency, and robustness of the proposed hierarchical coordinated ATM framework, this section quantitatively compares training convergence across different control architectures (Fig. 11). Based on preliminary trials, the maximum training duration is set to 500 episodes for the pure data-driven group to accommodate its slower convergence, and 200 episodes for the hybrid model-data-driven group due to MPC-guided acceleration. Although pure model-driven frameworks require no training, they are evaluated using the identical reward system by embedding an inactive MARL layer (yielding zero residual outputs) to ensure a consistent baseline. The resulting reward curves yield the following insights:

(1) Pure data-driven group: As shown in Fig. 11(a), without upper-level MPC guidance, the proposed ST-MAPPO algorithm significantly outperforms baseline MARL methods over the 500- episode horizon. Despite the high-dimensional spatiotemporal state space and non-stationary disturbances across multiple weaving segments, ST-MAPPO’s total reward climbs rapidly from approximately -46 during the initial phase (0–100 episodes) and converges around episode 280, stabilizing near -25. Conversely, standard MAPPO traps in a local optimum near -33 due to the lack of spatiotemporal attention, while MADDPG exhibits severe global oscillations (fluctuating between -25 and -48) and fails to converge. This confirms the efficacy of the network architecture designed in Section 5.3.4: the embedded Mamba selective state-space model filters highfrequency non-Markovian noise to preserve long-sequence temporal context, while the GCTAenhanced Critic network injects a strong spatial inductive bias that mitigates competitive agent behaviors during decentralized execution, steering the policy toward the global Pareto front.

Pure model-driven group: Serving as static baselines derived from deterministic rolling optimization (Fig. 11(b)), the proposed MPC-LL yields the highest average reward (-29.81), outperforming the MPC-LS (-32.29) and MPC-SS (-36.58). Conventional METANET’s cross-sectional homogeneity assumption fails under highly heterogeneous traffic—such as outer-lane turbulence from forced cut-ins paired with free-flowing inner lanes—where its uniform segmentlevel commands waste inner-lane capacity and jeopardize outer-lane safety. In contrast, the L-METANET explicitly models free/forced lane-changing mechanisms and lateral asymmetric friction penalties within the momentum relaxation equation. This high-fidelity forecasting of lane heterogeneity enables fine-grained, differentiated LVSL, boosting system performance.

(3) Hierarchical coordinated control group: Fig. 11(c) evaluates the training curves over a 200-episode horizon. Incorporating upper-level MPC baseline commands as guiding anchors for rolling fine-tuning substantially lifts initial rewards and accelerates learning. The proposed MPC-STMAPPO framework converges within just 90 episodes, stabilizing securely at approximately -26. Conversely, MPC-MAPPO stagnates in a local optimum, and MPC-MADDPG exhibits severe global oscillations. This indicates that while physical boundary constraints contract the invalid exploration space during blind exploration, a well-matched MARL architecture remains crucial for effective model-data synergy.

(4) Ablation analysis: Fig. 11(d) presents the ablation study for the proposed framework. MPC-STMAPPO exhibits a higher initial reward, faster convergence, and reduced volatility compared to ST-MAPPO, confirming that MPC guidance ensures stable policy behavior and superior training efficiency. However, pure ST-MAPPO slightly outperforms MPC-STMAPPO in final steady-state reward by a margin of approximately 1. This numerical discrepancy reveals a slight detriment to MARL optimality caused by predictive model mismatch: since actual control commands must anchor to the upper-level baseline, the limited step-by-step residual finetuning margin struggles to completely overcome the systemic bias introduced by the MPC due to the spatial constraints of the truncation protection mechanism, causing a minor compromise in the ultimate reward of the hybrid framework. Nonetheless, pure ST-MAPPO’s slight edge comes at the expense of severe reward oscillations during the first 150 episodes, complicating stable training. More importantly, the hierarchical framework trades this slight system performance for structural generalization resilience. Pure data-driven methods risk overfitting to specific scenarios, whereas the physical conservation laws embedded in the hierarchical architecture guarantee a rigid performance lower bound. As detailed in the Section 0, under varying loads or sudden demand mutations, pure MARL suffers from variance collapse or action overshooting due to a lack of safety guardrails. In contrast, the hierarchical framework exhibits robust transfer performance, proving its viability for real-world engineering deployment.

![](images/0b4d2b59d2b166d09463d55771fc0a306fc4a4817f4ace6e5f3dba4f424c77d4.jpg)  
Fig. 11 Reward curves of different methods

## 6.4.3 Quantitative evaluation of multi-objective control efficacy

Table 3 provides a comprehensive quantitative evaluation of the operational efficiency and macroscopic safety multi-objective metrics across 11 control schemes under empirical dynamic traffic demands for three core weaving segments (WS3, WS4, and WS6). These metrics include mainline average speed (Speed), on-ramp queue length (Queue), macroscopic weaving segment risk (Risk), total time loss (TL), and the total number of simulated collisions (Collisions). The following key observations can be made:

(1) Across the entire experimental dataset, all ATM strategies demonstrate varying degrees of system optimization compared to the No-Control and Random Control benchmarks. Under the unmanaged natural driving state (No-Control), intense lateral disruptions among the three weaving streams at the bottleneck cause the mainline average speed at the downstream weaving segment WS6 to plummet to 14.4 km/h, escalating the total time loss (TL) to 2,947 veh·h and triggering 312 network-wide micro-collisions, which pushes the system into widespread breakdown and severe safety crises. In contrast, the proposed MPC-STMAPPO hierarchical coordinated control framework exhibits exceptional global coordination and proactive traffic state reconstruction capabilities. It slashes the total network-wide collisions to 25 (a 91.99% reduction relative to No-Control), lifts the mainline speed at the heavily congested recurrent bottleneck WS6 to 38.5 km/h, and suppresses the corresponding TL to 627 veh·h. These results validate that a coordinated VSL-RM control strategy at weaving bottlenecks effectively achieves bi-objective optimization for both traffic safety and operational efficiency;

(2) To analyze the impacts of control granularity and predictive model fidelity on the closed-loop system, the pure model-driven group (MPC-SS, MPC-LS, and MPC-LL) provides clear empirical evidence. Due to the cross-sectional lane homogeneity assumption, the conventional segment-level MPC-SS model fails to capture the lateral friction heterogeneity within the lane cross-sections induced by diverging and lane-changing. Consequently, its uniform control commands trigger severe secondary intervention disruptions. Although the L-METANET segmentlevel controller (MPC-LS) outperforms MPC-SS, its coarse-grained execution limits further performance gains. Conversely, when the control granularity is refined down to the lane level via the developed L-METANET model (MPC-LL), lane-level differentiated LVSL can be executed precisely to smooth traffic fluctuations. As a result, the queue length at WS4 is minimized to 0.22 veh, and the bottleneck speed at WS6 is doubled to 32.4 km/h. This strongly substantiates the necessity of reconstructing the macroscopic traffic flow model at the lane level and applying fine-grained lane-by-lane control, as established in Section 4.1;

(3) Regarding the performance of the pure data-driven groups (MADDPG, MAPPO, and ST-MAPPO), the ST-MAPPO leverages its embedded Mamba selective state-space memory and GCTA mechanism, demonstrating robust capabilities in spatiotemporal feature decoupling and exploratory policy optimization. At the recurrent core bottleneck WS6—where weaving friction is most intense and congestion is frequent—ST-MAPPO maintains a mainline operating speed of 37.1 km/h and restricts the total number of micro-collisions to 30 over the full simulation cycle. In contrast, the standard MAPPO algorithm without the spatiotemporal attention mechanism suffers from degraded traffic efficiency at WS6. Furthermore, the deterministic policy gradient-based MADDPG algorithm exhibits severe competitive, self-interested behaviors and policy fitting limitations, yielding an unacceptable expected macroscopic crash risk of 1.751 at WS6, indicating an unsustainably high crash vulnerability within this weaving segment;

(4) Within the hybrid model-data-driven group (MPC-MADDPG, MPC-MAPPO, and MPC-STMAPPO), the proposed MPC-STMAPPO achieves the highest WS6 mainline operating speed (38.5 km/h) and the shortest on-ramp queue length (21.82 veh) among all 11 control schemes, while substantially reducing network-wide simulated collisions to 25. Other hybrid combinations show consistently inferior performance compared to MPC-STMAPPO.

The quantitative results in Table 3 demonstrate the superior performance of the proposed framework from a global multi-objective Pareto optimization perspective. Among the 13 network-wide core evaluation metrics, the proposed MPC-STMAPPO strategy ranks in the top three for six metrics and in the bottom three for only one. Analysis of the ranking distribution reveals that the framework's optimal metrics are concentrated in heavily congested weaving segments under severe bottleneck pressure, notably at WS6. This indicates that coupling a physical safeguard with high-frequency residual compensation yields highly robust performance under severe spatiotemporal shock waves. At the less congested bottlenecks (WS3 and WS4), MPC-STMAPPO remains highly competitive despite not being the bestperforming scheme. This minor compromise reflects the network’s multi-segment coordination mechanism, which strategically introduces localized upstream sacrifices to efficiently mitigate severe downstream congestion at WS6. Additionally, the standalone upper-level MPC-LL and lower-level

Table 3 Quantitative multi-objective performance indicators of the system under different methods
<table><tr><td rowspan="2">Indicator</td><td colspan="3">WS3</td><td colspan="3">WS4</td><td colspan="6">WS6</td><td>Colli-</td><td>Top 3</td><td>Bottom</td></tr><tr><td>Speed (km/h)</td><td>Queue (veh)</td><td>Risk</td><td>TL (veh·h)</td><td>Speed (km/h)</td><td>Queue (veh)</td><td>Risk</td><td>TL (veh·h)</td><td>Speed (km/h)</td><td>Queue (veh)</td><td>Risk</td><td>TL (veh·h)</td><td>sions (Num-</td><td>(Num- ber)</td><td>3(Num- ber)</td></tr><tr><td>No Control</td><td>64.7</td><td>0.24</td><td>0.42</td><td>563</td><td>42.0</td><td>2.88</td><td>0.58</td><td>776</td><td>14.4</td><td>31.98</td><td>1.879</td><td>2947</td><td>312</td><td>1</td><td>8</td></tr><tr><td></td><td>±0.0</td><td>±0.00</td><td>±0.00</td><td>±0</td><td>±0.0</td><td>±0.00</td><td>±0.00</td><td>±0</td><td>±0.0</td><td>±0.00</td><td>±0.00</td><td>±0</td><td>±0</td><td></td><td></td></tr><tr><td>Random Control</td><td>63.9</td><td>6.48 ±0.38</td><td>0.35</td><td>305</td><td>64.7</td><td>14.07 ±0.85</td><td>0.59</td><td>512</td><td>14.7 ±0.2</td><td>32.90 ±0.25</td><td>1.632</td><td>1684</td><td>172</td><td>1</td><td>7</td></tr><tr><td>MPC-SS</td><td>±0.4</td><td>14.91</td><td>±0.05</td><td>±43</td><td>±0.8</td><td></td><td>±0.15</td><td>±132</td><td></td><td>32.84</td><td>±0.14</td><td>±134</td><td>±22</td><td></td><td></td></tr><tr><td></td><td>72.3 ±0.0</td><td>±0.00</td><td>0.62 ±0.00</td><td>380 ±4</td><td>71.8 ±0.0</td><td>36.11 ±0.00</td><td>0.92 ±0.00</td><td>564 ±6</td><td>15.4 ±0.2</td><td>±0.75</td><td>1.556 ±0.05</td><td>1677 ±56</td><td>62 ±8</td><td>2</td><td>6</td></tr><tr><td>MPC-LS</td><td>72.1</td><td>14.95</td><td>0.30</td><td>256</td><td>68.3</td><td>14.77</td><td>0.56</td><td>486</td><td>14.9</td><td>31.50</td><td>1.547</td><td>1565</td><td>60</td><td></td><td>2</td></tr><tr><td></td><td>±0.2</td><td>±0.03</td><td>±0.02</td><td>±20</td><td>±0.3</td><td>±0.64</td><td>±0.02</td><td>±20</td><td>±0.2</td><td>±0.54</td><td>±0.04</td><td>±54</td><td>±10</td><td>6</td><td></td></tr><tr><td>MPC-LL</td><td>68.2</td><td>13.24</td><td>0.27</td><td>249</td><td>65.3</td><td>0.22</td><td>0.50</td><td>461</td><td>32.4</td><td>33.23</td><td>0.791</td><td>570</td><td>37</td><td>6</td><td>1</td></tr><tr><td></td><td>±0.5</td><td>±0.10</td><td>±0.02</td><td>±20</td><td>±0.2</td><td>±0.02</td><td>±0.04</td><td>±33</td><td>±0.6</td><td>±0.10</td><td>±0.01</td><td>±12</td><td>±4</td><td></td><td></td></tr><tr><td>MADDPG</td><td>69.7</td><td>11.65</td><td>0.46</td><td>378</td><td>68.1</td><td>0.24</td><td>0.32</td><td>269</td><td>17.9</td><td>25.38</td><td>1.751</td><td>1753</td><td>24</td><td>4</td><td>3</td></tr><tr><td></td><td>±0.0</td><td>±0.00</td><td>±0.00</td><td>±0</td><td>±0.0</td><td>±0.00</td><td>±0.00</td><td>±0</td><td>±0.0</td><td>±0.00</td><td>±0.00</td><td>±0</td><td>±0</td><td></td><td></td></tr><tr><td>MAPPO</td><td>65.6</td><td>11.07</td><td>0.55</td><td>340</td><td>67.3</td><td>25.56</td><td>0.88</td><td>543</td><td>32.8</td><td>31.76</td><td>0.791</td><td>634</td><td>36</td><td>2</td><td>3</td></tr><tr><td></td><td>±0.6</td><td>±0.65</td><td>±0.12</td><td>±80</td><td>±0.3</td><td>±1.32</td><td>±0.19</td><td>±123</td><td>±1.2</td><td>±0.44</td><td>±0.03</td><td>±63</td><td>±12</td><td></td><td></td></tr><tr><td>ST-MAPPO</td><td>72.3</td><td>14.91</td><td>0.52</td><td>355</td><td>69.6</td><td>0.21</td><td>0.97</td><td>663</td><td>37.1</td><td>23.26</td><td>0.893</td><td>599</td><td>30</td><td>7</td><td>3</td></tr><tr><td></td><td>±0.1</td><td>±0.16</td><td>±0.03</td><td>±38</td><td>±0.1</td><td>±0.00</td><td>±0.04</td><td>±71</td><td>±1.3</td><td>±1.16</td><td>±0.23</td><td>±146</td><td>±7</td><td></td><td></td></tr><tr><td>MPC+</td><td>67.8</td><td>8.66</td><td>0.20</td><td>174</td><td>62.6</td><td>0.26</td><td>1.08</td><td>877</td><td>22.3</td><td>31.62</td><td>0.899</td><td>969</td><td>37</td><td>3</td><td>3</td></tr><tr><td>MADDPG</td><td>±0.3</td><td>±0.08</td><td>±0.03</td><td>±40</td><td>±0.2</td><td>±0.03</td><td>±0.04</td><td>±91</td><td>±1.1</td><td>±0.49</td><td>±0.03</td><td>±77</td><td>±7</td><td></td><td></td></tr><tr><td>MPC+</td><td>67.2</td><td>11.21</td><td>0.53</td><td>310</td><td>68.2</td><td>25.44</td><td>0.80</td><td>477</td><td>31.3</td><td>30.82</td><td>1.03</td><td>686</td><td>32</td><td>1</td><td>2</td></tr><tr><td>MAPPO</td><td>±0.5</td><td>±0.42</td><td>±0.10</td><td>±47</td><td>±0.4</td><td>±2.00</td><td>±0.20</td><td>±132</td><td>±1.2</td><td>±0.72</td><td>±0.07</td><td>±41</td><td>±9</td><td></td><td></td></tr><tr><td>MPC+</td><td>71.8</td><td>14.13</td><td>0.47</td><td>316</td><td>66.5</td><td>0.22</td><td>0.93</td><td>626</td><td>38.5</td><td>21.82</td><td>0.833</td><td>627</td><td>25</td><td>6</td><td>1</td></tr><tr><td>ST-MAPPO</td><td>±0.2</td><td>±0.25</td><td>±0.03</td><td>±40</td><td>±0.7</td><td>±0.09</td><td>±0.05</td><td>±77</td><td>±3.3</td><td>±3.08</td><td>±0.27</td><td>±168</td><td>±6</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Note: “SS” stands for segment model (METANET) with segment-level VSL; “LS” stands for lane model (L-METANET) with segment-level VSL; “LL” stands for lane model (L-METANET) with lane-level VSL. “TL” stands for Time Loss. Bold green text indicates the top 3 rankings; bold red text indicates the bottom 3 rankings. “X ± Y” stands for Mean ± Standard Deviation.

ST-MAPPO models also achieve high rankings across most metrics, confirming that the hierarchical coordinated architecture effectively mitigates the inherent trade-offs between operational efficiency and traffic safety across interconnected weaving segments.

## 6.5. Transferability and generalization performance analysis of MPC-STMAPPO

As demonstrated by the experimental results in Section 6.4.2 and 6.4.3, under the baseline historica traffic demand, the pure data-driven ST-MAPPO algorithm yields a final converged expected reward that slightly outperforms the proposed MPC-STMAPPO hierarchical coordinated framework, while displaying highly competitive optimization performance across the various microscopic quantitative metrics listed in Table 3. Consequently, superior training efficiency alone is insufficient to fully establish the advantages of the proposed hierarchical framework. This indicates that under stationary traffic demand and a fixed network topology, ST-MAPPO can thoroughly fit the empirical trajectory profiles of that specific condition. However, this single-scenario optimality relies heavily on deterministic time-series sample interactions, making the policy highly susceptible to overfitting. In real-world engineering deployments, traffic demands are inherently time-varying. To validate policy robustness against unknown perturbations or extreme conditions—and to evaluate the generalization resilience and transferability of the hierarchical framework—this section constructs diverse transfer testing scenarios. Pre-trained policy networks optimized under baseline conditions are directly deployed across four distinct dynamic demand scenarios without further fine-tuning. This setup examines the worst-case performance boundaries of the control loop under severe shifts in the solution space. These transferability experiments focus on the two top-performing control strategies: ST-MAPPO and MPC-STMAPPO.

To evaluate their multi-dimensional transfer capabilities, the dynamic traffic demand scenarios are designed as follows: (1) Baseline traffic demand (Base\_Demand): Identical to the previous simulated traffic demand, serving as the benchmark reference scenario; (2) Low-load traffic demand (0.5×Base\_Demand): Scales down the original time-series boundary flows uniformly by a factor of 0.5 to simulate sparse traffic states typical of late-night or early-morning periods; (3) Over-saturated traffic demand (1.5×Base\_Demand): Scales up the original boundary flows uniformly by a factor of 1.5 to simulate extreme scenarios such as holiday peaks or sudden surge propagation, strictly testing the baseline capability of the controller to mitigate large-scale, network-wide traffic breakdown; (4) Random demand 1: Applies a uniform random scaling factor independently to the vehicle volume of each OD flow in the baseline route file. The random demand for the $k ^ { t h }$ flow is formulated as $d _ { k } ( r a n d 1 ) = \operatorname* { m a x } \Bigl [ 1 , \bigl ( \alpha _ { k } \cdot d _ { k } ( b a s e ) \bigr ) \Bigr ]$ , where $\alpha _ { k } \sim U ( 0 . 3 , 1 . 7 )$ . This transformation disrupts the volume distribution and statistical characteristics of historical demand across spatiotemporal cross-sections, causing the traffic flow of each OD pair to fluctuate randomly between 30% and 170% of its original value; (5) Random demand 2: Completely removes the topological structural constraints of historical demand, independently applying a uniform random sampling scheme to the vehicle volume of each OD flow in the route file, denoted as the random demand $d _ { k } ( r a n d 2 ) \sim U ( 1 , 2 0 0 )$ for the $k ^ { t h }$ flow. This transformation generates the traffic volume for each flow within entirely independent and uncorrelated intervals, thereby exposing the generalization limits of the controller under completely unknown, unstructured environmental disturbances.

The results in Table 4 utilize identical metrics to Table 3. The experiments demonstrate that while ST-MAPPO achieves excellent operational performance under specific known conditions, it is susceptible to generalization failure induced by sample overfitting and control action overshooting when subjected to unseen traffic demands in open environments. Specifically, under the over-saturated traffic demand (1.5×Base\_Demand), the pure MARL strategy causes the mainline average speed at the weaving bottleneck WS6 to plummet to 14.3 km/h, driving the TL up to 3,112 veh·h and triggering 109 networkwide collisions over the simulation horizon. Under Random demand 2, where the topological structure is completely decoupled and lacks historical statistical attributes, the control brittleness of the pure MARL policy becomes more pronounced. Even at the WS3 bottleneck, where upstream weaving pressure is relatively mild, the mainline speed deteriorates to 40.3 km/h. Furthermore, the TL at the critical bottleneck WS6 reaches 5,180 veh·h, causing total network collisions to climb to 154. In contrast, although MPC-STMAPPO underperforms the pure MARL strategy in a few metrics, it delivers superior overall performance, particularly when handling unseen traffic demands. Specifically, under the 1.5×Base\_Demand scenario, MPC-STMAPPO successfully maintains the WS6 mainline speed at 16.4 km/h. Although this operating speed is still low, it represents a improvement over ST-MAPPO. Furthermore, networkwide collisions are restricted to 88, marking a significant reduction compared to the 109 collisions under ST-MAPPO. Under the extreme conditions of Random demand 2, the hierarchical framework reshapes the spatiotemporal safety manifold of the traffic system through proactive control, maintaining the mainline speed at WS3 at 60.6 km/h, which represents a 50.37% efficiency increase over pure MARL, and reducing delay at the critical WS6 bottleneck by 17.22% relative to the pure MARL baseline.

Table 4 Transfer application performance between ST-MAPPO and MPC-STMAPPO under different traffic demands
<table><tr><td rowspan="2">Demand</td><td rowspan="2">Method</td><td colspan="3">WS3</td><td colspan="4">WS4</td><td colspan="5">WS6</td><td rowspan="2">Collisions (Number)</td></tr><tr><td>Speed (km/h)</td><td>Queue (veh)</td><td>Risk</td><td>TL (veh·h)</td><td>Speed (km/h)</td><td>Queue (veh)</td><td>Risk</td><td>TL (veh·h)</td><td>Speed (km/h)</td><td>Queue (veh)</td><td>Risk TL</td><td>(veh·h)</td></tr><tr><td rowspan="4">0.5×Base_ Demand</td><td>STMAPPO</td><td>74.0</td><td>14.90</td><td>0.60</td><td>210</td><td>73.6</td><td>0.08</td><td>0.75</td><td>261</td><td>70.6</td><td>27.65</td><td>0.64</td><td>262</td><td>4</td></tr><tr><td></td><td>±0.1</td><td>±0.12</td><td>±0.03</td><td>±10</td><td>±0.1</td><td>±0.00</td><td>±0.03</td><td>±10</td><td>±0.1</td><td>±0.30</td><td>±0.02</td><td>±8</td><td>±1</td></tr><tr><td>MPC+</td><td>73.8</td><td>14.67</td><td>0.65</td><td>218</td><td>72.1</td><td>0.13</td><td>0.74</td><td>247</td><td>70.0</td><td>20.48</td><td>0.61</td><td>240</td><td>3</td></tr><tr><td>STMAPPO</td><td>±0.2</td><td>±0.13</td><td>±0.02</td><td>±14</td><td>±0.4</td><td>±0.01</td><td>±0.02</td><td>±5</td><td>±0.3</td><td>±3.34</td><td>±0.04</td><td>±9</td><td>±1</td></tr><tr><td rowspan="4">Base_ Demand</td><td>STMAPPO</td><td>72.3</td><td>14.91</td><td>0.52</td><td>355</td><td>69.6</td><td>0.21</td><td>0.97</td><td>663</td><td>37.1</td><td>23.26</td><td>0.893</td><td>599</td><td>30</td></tr><tr><td></td><td>±0.1</td><td>±0.16</td><td>±0.03</td><td>±38</td><td>±0.1</td><td>±0.00</td><td>±0.04</td><td>±71</td><td>±1.3</td><td>±1.16</td><td>±0.23</td><td>±146</td><td>±7</td></tr><tr><td>MPC+</td><td>71.8</td><td>14.13</td><td>0.47</td><td>316</td><td>66.5</td><td>0.22</td><td>0.93</td><td>626</td><td>38.5</td><td>21.82</td><td>0.833</td><td>627</td><td>25</td></tr><tr><td>STMAPPO</td><td>±0.2</td><td>±0.25</td><td>±0.03</td><td>±40</td><td>±0.7</td><td>±0.09</td><td>±0.05</td><td>±77</td><td>±3.3</td><td>±3.08</td><td>±0.27</td><td>±168</td><td>±6</td></tr><tr><td rowspan="4">1.5×Base Demand</td><td>STMAPPO</td><td>70.8</td><td>14.67</td><td>0.28</td><td>461</td><td>62.9</td><td>0.47</td><td>0.60</td><td>975</td><td>14.3</td><td>36.07</td><td>1.61</td><td>3112</td><td>109</td></tr><tr><td></td><td>±0.15</td><td>±0.18</td><td>±0.02</td><td>±25</td><td>±0.32</td><td>±0.16</td><td>±0.08</td><td>±116</td><td>±0.04</td><td>±0.23</td><td>0.09</td><td>±247</td><td>±18</td></tr><tr><td>MPC+</td><td>64.2</td><td>11.48</td><td>0.40</td><td>546</td><td>65.5</td><td>0.37</td><td>0.56</td><td>886</td><td>16.4</td><td>34.60</td><td>1.59</td><td>2621</td><td>88</td></tr><tr><td>STMAPPO</td><td>±0.50</td><td>0.56</td><td>±0.12</td><td>±132</td><td>±0.93</td><td>±0.23</td><td>±0.08</td><td>±124</td><td>±0.26</td><td>±0.26</td><td>±0.17</td><td>±238</td><td>±6</td></tr><tr><td rowspan="4">Random Demand 1</td><td>STMAPPO</td><td>72.6</td><td>14.81</td><td>0.48</td><td>301</td><td>66.9</td><td>0.27</td><td>0.92</td><td>602</td><td>30.8</td><td>32.12</td><td>0.88</td><td>651</td><td>33</td></tr><tr><td></td><td>±0.15</td><td>±0.06</td><td>±0.06</td><td>±61</td><td>±0.41</td><td>±0.04</td><td>±0.05</td><td>±78</td><td>±6.32</td><td>±1.23</td><td>±0.13</td><td>±119</td><td>±8</td></tr><tr><td>MPC+</td><td>72.1</td><td>13.89</td><td>0.49</td><td>325</td><td>69.5</td><td>0.22</td><td>0.92</td><td>582</td><td>36.8</td><td>24.41</td><td>0.94</td><td>627</td><td>28</td></tr><tr><td>STMAPPO</td><td>±0.09</td><td>±0.16</td><td>±0.05</td><td>±57</td><td>±0.59</td><td>±0.01</td><td>±0.09</td><td>±97</td><td>±3.04</td><td>±1.25</td><td>±0.17</td><td>±91</td><td>±5</td></tr><tr><td rowspan="4">Random Demand 2</td><td>STMAPPO</td><td>40.3</td><td>5.65</td><td>0.63</td><td>2092</td><td>28.5</td><td>26.9</td><td>1.13</td><td>3797</td><td>12.1</td><td>24.93</td><td>1.55</td><td>5180</td><td>154</td></tr><tr><td></td><td>±8.98</td><td>±0.89</td><td>±0.11</td><td>±385</td><td>±5.47</td><td>±2.57</td><td>±0.36</td><td>±1326</td><td>±1.35</td><td>±1.46</td><td>±0.30</td><td>±1068</td><td>±26</td></tr><tr><td>MPC+</td><td>60.6</td><td>14.92</td><td>0.47</td><td>1644</td><td>30.4</td><td>8.40</td><td>1.17</td><td>4079</td><td>12.8</td><td>22.36</td><td>1.42</td><td>4288</td><td>134</td></tr><tr><td>STMAPPO</td><td>±8.4</td><td>±0.17</td><td>±0.26</td><td>±908</td><td>±4.19</td><td>±0.74</td><td>±0.18</td><td>±663</td><td>±1.96</td><td>±0.71</td><td>±0.12</td><td>±456</td><td>±14</td></tr></table>

Note: “TL” stands for Time Loss; Bold text indicates the best performance.

In summary, MPC-STMAPPO breaks the trade-off between system performance and transfer resilience under unknown conditions. Leveraging the rigid protection of the upper-level MPC, the framework establishes a physical safety lower bound and ensures high generalization capability when the traffic system encounters unseen, abrupt, and over-saturated turbulence. Consequently, the proposed framework offers superior viability for real-world engineering deployment compared to ST-MAPPO.

## 7. Conclusion

Under a hybrid model-data-driven framework, this study addresses the core scientific challenge of managing the spatiotemporal trade-off between operational efficiency and traffic safety, as well as the difficulties of online real-time adaptive coordinated control within mixed traffic flows across consecutive multiple weaving segments on urban expressways. The main research conclusions and scientific contributions are summarized as follows:

First, a high-fidelity lane-level macroscopic traffic flow prediction model, L-METANET, is established. To overcome the cross-sectional lane homogeneity assumption inherent in the conventional METANET model, L-METANET explicitly incorporates space allocation mechanisms for both free and forced lane-changing behaviors. Field validation using empirical detector data from Changchun demonstrates that, compared with the conventional segment-level METANET model, L-METANET precisely replicates the dynamic cross-lane flow redistribution and sudden capacity drops induced by forced lanechanging, showing high alignment with ground-truth flow and speed evolution. Second, a weaving segment risk assessment and prediction model is developed and validated. By integrating data-driven machine learning with rigorous statistical causal inference, an analytical assessment and prediction equation for dynamic merging and diverging crash risks is constructed using XGBoost-SHAP feature selection and a RPBL model. This hybrid risk assessment framework exhibits excellent high-risk identification capabilities, achieving an AUC greater than 0.70 across all merging and diverging risk classification tasks, with most tasks falling into a high-performance interval above 0.80, significantly outperforming various conventional Logit models. Third, the efficacy of the hierarchical coordinated control strategy in balancing system performance and resilience is comprehensively verified. The convergence speed and steady-state reward values of the proposed MPC-STMAPPO framework surpass baseline methods, rapidly stabilizing at a reward value of approximately -26 within only 90 episodes. Furthermore, the proposed strategy ranks within the top three across 6 out of 13 core evaluation metrics. It sharply reduces the total number of simulated network-wide crashes from 312 to 25, efficiently mitigates turbulent weaving flows, and proactively suppresses localized shock waves at the core weaving bottleneck WS6, effectively breaking the inherent conflict between efficiency and safety in conventional ATM. In cross-load zero-shot transfer performance tests characterizing system resilience, where pure data-driven strategies suffer from performance degradation in unknown scenarios, the proposed framework significantly outperforms pure MARL strategies in transfer robustness and worst-case performance lower-bound protection due to the rigid protection provided by the embedded physical conservation laws within the upper-level MPC, demonstrating substantial potential for real-world industrial deployment.

Despite these breakthroughs, several directions remain for future exploration and refinement. The control horizon of this study primarily focuses on macroscopic LVSL and RM. Future work will extend this framework across scales to integrate the proposed macroscopic ATM strategies with our previously developed microscopic trajectory planning for heterogeneous multi-vehicle merging in weaving segments (Ma et al., 2026c), which is expected to further enhance multi-objective system performance. Consequently, future efforts will utilize collaborative aerial photography via multi-UAV swarms to capture microscopic trajectories over longer continuous stretches, combined with finer macroscopically aggregated data, to further demonstrate the synergistic benefits of macroscopic ATM strategies and microscopic trajectory control in real-world engineering deployments.

## CRediT authorship contribution statement

Guodong Ma: Writing – original draft, Writing – review & editing, Software, Methodology, Conceptualization, Formal analysis. Baofeng Sun: Writing – review & editing, Supervision, Funding acquisition. Wenyu Yang: Writing – review & editing. Zhihong Yao: Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

This research was supported by the National Natural Science Foundation of China (grant number 52472313).

## Data availability

Data will be made available on request.

## Appendix A. Flow estimation of free and forced lane-changing in L-METANET

Part 1: estimation of free lane-changing flow. In the lane-level extended dynamic equations, the lateral net free lane-changing flow $\phi _ { m , n  n ^ { \prime } } ^ { i }$ (where $n ^ { \prime }$ denotes the adjacent  or  lane) between adjacent lanes within the same segment serves as the core dynamic that breaks the independent operation of each lane and induces lateral friction effects. Following the multi-lane traffic flow modeling logic proposed by Papageorgiou et al., free lane-changing behaviors are primarily driven by the density differences between adjacent lanes and are strictly constrained by the remaining available space of the target lane (Roncoli et al., 2015). Incorporating the heterogeneity of mixed traffic flow into the baseline Papageorgiou macroscopic lane-changing model yields the formulation L-METANET in this study.

First, the lane-changing attractiveness of target lane $n ^ { \prime }$ to current lane is defined as $A _ { m , n  n ^ { \prime } } ^ { i }$ . In a mixed traffic environment comprising CAVs and HDVs, lane-changing aggressiveness varies significantly by vehicle type. Let $\mu _ { C A V }$ and $\mu _ { H D V }$ denote the lane-changing aggressiveness coefficients for CAVs and HDVs, respectively. Combined with the CAV penetration rate $\alpha _ { \scriptscriptstyle \perp }$ , the comprehensive lanechanging aggressiveness parameter $\mu _ { m i x }$ for the current segment is expressed as Eq. (61). Based on $\mu _ { m i x }$ , the mixed-flow lane-changing attractiveness $A _ { m , n  n ^ { \prime } } ^ { i }$ is calculated as shown in Eq. (62). Note that the parameter $P _ { m , n  n ^ { \prime } } ^ { i }$ is a spatial weighting coefficient, which typically equals 1, but can be calibrated in specific zones such as on-ramps and off-ramps to capture localized driving preferences. Based on

$A _ { m , n  n ^ { \prime } } ^ { i }$ , the lateral free lane-changing demand flow $D _ { m , n  n } ^ { i } .$ migrating from cell $( m , n )$ to cell $( m , n ^ { \prime } )$ is the product of the total number of vehicles in the current lane and the attractiveness index, as formulated in Eq. (63).

However, due to the finite physical space of target lane $n " .$ , not all lane-changing demands can be satisfied. The maximum remaining acceptable space (i.e., supply flow rate) $S _ { m , n } ^ { i } .$ that target lane $n ^ { \prime }$ can provide at the current time step depends on its jam density $k _ { j a m , m , n ^ { \prime } } ^ { \mathrm { ~ ~ } } ,$ , as defined in Eq. (64).

When adjacent lanes on both sides of target lane $n ^ { \prime } ( \mathrm { i . e . , } n ^ { \prime } { - } 1$ and $n ^ { \prime } { + } 1 )$ simultaneously generate lane-changing demands that exceed the remaining space of the target lane, the actual lane-changing flow rates must be scaled down and allocated proportionally based on demand. Therefore, the final actual net lane-changing flow $\phi _ { m , n  n ^ { \prime } } ^ { i }$ from the current lane  to target lane $n ^ { \prime }$ is calculated via Eq. (65). This allocation mechanism ensures that under extreme congestion, the target lane does not experience density violation beyond its physical limits due to excessive vehicle admissions.

$$
\mu _ { m i x } = \alpha \cdot \mu _ { C A V } + ( 1 - \alpha ) \cdot \mu _ { H D V }\tag{61}
$$

$$
{ \cal A } _ { m , n  n ^ { \prime } } ^ { i } = { \mu } _ { m i x } \cdot \mathrm { m a x } \Bigg [ 0 , { \frac { P _ { m , n  n ^ { \prime } } ^ { i } \cdot k _ { m , n } ^ { i } - k _ { m , n ^ { \prime } } ^ { i } } { P _ { m , n  n ^ { \prime } } ^ { i } \cdot k _ { m , n } ^ { i } + k _ { m , n ^ { \prime } } ^ { i } } } \Bigg ]\tag{62}
$$

$$
D _ { m , n  n ^ { \prime } } ^ { i } = A _ { m , n  n ^ { \prime } } ^ { i } \cdot \frac { \Delta x _ { m } } { \Delta t } \cdot k _ { m , n } ^ { i }\tag{63}
$$

$$
{ S _ { m , n ^ { \prime } } ^ { i } = ( k _ { j a m , m , n ^ { \prime } } - k _ { m , n ^ { \prime } } ^ { i } ) \frac { \Delta x _ { m } } { \Delta t } }\tag{64}
$$

$$
\phi _ { m , n  n ^ { \prime } } ^ { i , f r } = \operatorname * { m i n } \biggl [ 1 , \frac { S _ { m , n ^ { \prime } } ^ { i } } { D _ { m , n ^ { \prime } - 1  n ^ { \prime } } ^ { i } + D _ { m , n ^ { \prime } + 1  n ^ { \prime } } ^ { i } } \biggr ] \cdot D _ { m , n  n } ^ { i } ,\tag{65}
$$

Part 2: Estimation of forced lane-changing flow. Unlike free lane-changing driven by localized density differentials, forced lane-changing is primarily governed by network topology constraints, such as lane drops and ramps, alongside the route choices of drivers. Traditional macroscopic models typically calculate forced lane-changing flows continuously within a road segment. However, under complex weaving segment topologies characterized by sudden lane drops or high-volume off-ramps, lane-level models often suffer from numerical errors such as mass leakage or duplicate counting due to the coupling between convection and friction terms. To resolve this issue, this study introduces a cascading vehicle-borrowing mechanism to precisely quantify forced lane-changing flows and their associated lateral friction penalties. Let $\mathcal { Q } _ { m } ^ { i }$ denote the actual arrival flow at the downstream cross-section of segment after the redistribution by internal free lane-changing, satisfying $\mathcal { Q } _ { m , n } ^ { i } = \boldsymbol { q } _ { m , n } ^ { i } + \phi _ { m , n } ^ { i , f r }$ . The difference $\Delta q _ { m , n }$ dictates the physical direction and cascading intensity of forced lane-changing maneuvers, as formulated in Eq. (66).

$$
\Delta q _ { m , n } = Q _ { m , n } ^ { i } - s _ { m , n } ^ { i }\tag{66}
$$

Scenario A: Flow surplus $( \Delta q _ { m , n } \ge 0 )$ without downstream lane continuity. When the arrival flow exceeds the diverging demand, vehicles originally traveling in the diverging lane intend to continue straight. Due to downstream geometric cut-offs or narrowing, this surplus flow $\Delta q _ { m , n }$ must be forced into the adjacent inner through lane $( \mathrm { l a n e } n + 1 )$ . This forced cut-in maneuver imposes a lateral asymmetric friction penalty on both lanes: the inner through lane absorbs the forced vehicle insertions, whereas the outer lane experiences forced vehicle extractions, as expressed in Eq. (67).

$$
\phi _ { m , n  n + 1 } ^ { i , f o } = \Delta q _ { m , n }\tag{67}
$$

Scenario B: Cascading vehicle borrowing induced by flow deficit $\left( \Delta q < 0 \right)$ . When the arrival flow fails to meet the off-ramp demand, a flow deficit $- \Delta q _ { m , n }$ occurs. To satisfy the diverging demand, traffic must be forceded from the inner through lanes $( n - 1 , n - 2 . . . )$ toward the outer side. The model evaluates lanes sequentially from the outermost to the innermost. Taking lane $n { + 1 }$ as an example, its actual borrowable flow $b _ { m , n + 1 }$ is bounded by the absolute flow capacity threshold of the lane, as shown in Eq. (68). For the $\left( n + k \right) ^ { t h }$ inner through lane, when selected for vehicle borrowing, the remaining unfulfilled deficit $E _ { m , n + k }$ equals the total deficit minus the cumulative flow already extracted from all lanes to its outer side, as shown in Eq. (69). The actual borrowed flow $b _ { m , n + k }$ from this lane is then determined as the minimum of the remaining deficit and the arrival flow of the lane, as formulated in Eq. (70). Crucially, vehicles cannot instantaneously teleport across multiple lanes in physical space. If an inner lane (e.g., $n + k , k \geq 1 )$ is subject to vehicle borrowing, the forced lane-changing traffic stream must sequentially traverse all intermediate lanes to reach the outer diverging lane . Consequently, the actual forced lanechanging flow occurring between any two adjacent lanes $n { + } k$ and $n + k - 1$ equals the flow directly borrowed from lane $n { + } k$ plus the cumulative borrowed flow originating from all lanes further inner to lane $n + k$ that must transit through lane $n { + } k$ , as formulated in Eq. (71).

$$
b _ { m , n + 1 } = \operatorname* { m i n } ( - \Delta q _ { m , n } , Q _ { m , n + 1 } ^ { i } )\tag{68}
$$

$$
E _ { m , n + k } = - \Delta q _ { m , n } - \sum _ { j = 1 } ^ { k - 1 } b _ { m , n + j }\tag{69}
$$

$$
b _ { m , n + k } = \operatorname* { m i n } \left( E _ { m , n + k } , Q _ { m , n + k } ^ { i } \right)\tag{70}
$$

$$
\phi _ { m , n + k  n + k - 1 } ^ { i , f o } = \sum _ { j = n + k } ^ { N _ { m } } b _ { m , j }\tag{71}
$$

Scenario C: Other conditions. For situations outside Scenarios A and B, no forced lane-changing behavior is triggered, as defined in Eq. (72).

$$
\phi _ { m , n  n + 1 } ^ { i , f o } = 0 , \forall n\tag{72}
$$

## Appendix B. Traffic demand of the Eastern Expressway

The boundary traffic demands for the SUMO simulation are derived from empirical traffic volumes collected between 06:00 and 19:00 on November 10, 2022, across all mainline cross-sections and ramps along the Eastern Expressway. Inflow demands are illustrated in Fig. 12, where Fig. 12(1) denotes the mainline origin volume and the remaining subfigures show the on-ramp demands. Similarly, Fig. 13 depicts outflow profiles, with Fig. 13(10) representing the mainline destination demand and the remaining subfigures displaying the off-ramp demands.

## Appendix C. Identification of key variables and RPBL fitting results

Table 5 lists the key feature variables extracted via the XGBoost-SHAP framework, and Table 6 presents the estimated parameter values calibrated using the RPBL model.

## Appendix D. Data used in this study

The micro-trajectory data used has been fully made publicly available at the following address: https://huggingface.co/datasets/InterestingITS/U-EASWS/tree/main. The macro-traffic demand data used can be obtained by contacting the first author at: magd22@mails.jlu.edu.cn.

![](images/ac61fe38f9937d71b958fe820e3a2c491fe666d69609731b004e277c44fa7bd3.jpg)

![](images/53a766ff1a81833d4e84bfc4d9cf31cbc6af7de9d02156db77e280ec6c1a9d70.jpg)

![](images/6bd4047c7e03b43e4beb198b44c34d53b4444b317e44f1907814750baf9529fc.jpg)

![](images/15345c914ea41784c317931d6f547557a7a36c38b9561d33be8eab6b74a3de1e.jpg)

![](images/6a5abadf7d8bd682c13ff6b168bcc0ac0ee02347b7b22553d1d795144ae517eb.jpg)

![](images/7f3d17897cee754ae95211b28a630ff582e3ca6d7c0e003bcec5fe0f0ca1aee9.jpg)

![](images/a47795183470389d6f44a9b19d2b90beacef5df46b43c95cadcfa259bc65f84a.jpg)

![](images/5477bdee7af77a785fe98446cabaffe366e679560412ed6160579bfe02ac66bb.jpg)

![](images/626027c56f6d40d67afa05eb7fbe96a7e1d79053c712d985d1c38dc23a7b7f18.jpg)

![](images/a020ea6c358ba3f2f6ddf30c2586da5f443763ba7dfcc1ce02e47b87d6067668.jpg)

![](images/621a037c76d0c33996655243c6794d14f5b596f04a397410bc3b6ac88ae500be.jpg)

![](images/f280c64d154e9eddc7d7aa10cf5a0479682cd9cf041f132ba515ac9a84d9549c.jpg)

![](images/3a09fdc96841650d360f34c28b5c9c02e0e29503e72cad580d8adf811299e381.jpg)

Fig. 12 Distribution of inflow at each entrance over time  
![](images/91f26e6ac6b28cf5ef2b35c0dc7c678e6d40d8e8f383cbef00f7cb9398335c06.jpg)

![](images/3304147b6c114f500c8b9834c75825d2d36350233e613d58719926d99aa20a6d.jpg)

![](images/f0842c5b1f17cd0d81f6a0c4f2362a7e8774e8b955b287fc3f61abf7ffc938c1.jpg)

![](images/462f2be465b49c4efdce1e9c73ee747ad0bfd6fb08bdcc3973af7d51c6b38c32.jpg)

![](images/c9098f63774950fd464ebd5ce74b4ca1bdccf325a90242c406c2f2e1697d2d50.jpg)

![](images/c852a166f07a647168382e60094b4674e539f413913331584b35a8d619f1b860.jpg)

![](images/bf9e9f6af390c18cf154565e3cf0691a8cb5b2a29148283d09ab1eb11afb41db.jpg)

![](images/486284644413b9a70dec74242c25e7f9da9460bf5a0f7091bcd999bfbe79d071.jpg)

![](images/fd6a56a70ecaa92959ab579dee95f5ce04661617241e6a9dbcd74bdf2e26f7e5.jpg)

![](images/0588447a32054c7d7f8722f26af0114959fa5e4193660f6a3c866d0bc6461cc1.jpg)  
Fig. 13 Distribution of flow at each exit over time

Table 5 Extraction of key feature variables
<table><tr><td></td><td>WS1-M</td><td>WS 1-D</td><td>WS 2-M</td><td>WS 2-D</td><td>WS 3-M</td><td>WS 3-D</td><td>WS 4-M</td><td>WS 4-D</td><td>WS 5-M</td><td>WS 5-D</td><td>WS 6-M</td><td>WS 6-D</td><td>WS 7-M</td><td>WS7-D</td></tr><tr><td>V1</td><td>D11</td><td>D21</td><td>S32</td><td>D12</td><td>D21</td><td>D21</td><td>D21</td><td>D21</td><td>D21</td><td>D31</td><td>D22</td><td>D21</td><td>D31</td><td>D31</td></tr><tr><td>V2</td><td>D21</td><td>D22</td><td>D11</td><td>S21</td><td>D11</td><td>D32</td><td>D11</td><td>S43</td><td>F21</td><td>D23</td><td>D11</td><td>D11</td><td>M32</td><td>D21</td></tr><tr><td>V3</td><td>D32</td><td>D12</td><td>D32</td><td>S23</td><td>D31</td><td>D31</td><td>S11</td><td>S11</td><td>D22</td><td>D41</td><td>D41</td><td>D41</td><td>S31</td><td>D22</td></tr><tr><td>V4</td><td>D22</td><td>S32</td><td>F22</td><td>O32</td><td>D32</td><td>D11</td><td>S31</td><td>D43</td><td>D43</td><td>D42</td><td>D12</td><td>S41</td><td>D21</td><td>S51</td></tr><tr><td>V5</td><td>D12</td><td>F22</td><td>F11</td><td>S22</td><td>D13</td><td>D12</td><td>D22</td><td>D22</td><td>D33</td><td>S42</td><td>D23</td><td>S42</td><td>D12</td><td>D11</td></tr><tr><td>V6</td><td>D31</td><td>D11</td><td>F43</td><td>S13</td><td>F31</td><td>S12</td><td>D43</td><td>O33</td><td>D31</td><td>D11</td><td>D42</td><td>012</td><td>S51</td><td>S41</td></tr><tr><td>V7</td><td>S41</td><td>D31</td><td>S23</td><td>S41</td><td>D12</td><td>O13</td><td>D32</td><td>D11</td><td>D23</td><td>F23</td><td>S11</td><td>S11</td><td>D32</td><td>D12</td></tr><tr><td>V8</td><td>S12</td><td>D13</td><td>O32</td><td>S12</td><td>D22</td><td>D23</td><td>F11</td><td>D41</td><td>S53</td><td>D51</td><td>S23</td><td>D12</td><td>D23</td><td>D33</td></tr></table>

Table 6 Summary of risk assessment and prediction functions for all tasks
<table><tr><td rowspan="2">Task</td><td rowspan="2">CT</td><td colspan="9">Fixed parameter</td><td colspan="4">Random parameter</td></tr><tr><td>FP</td><td>(µ, σ)</td><td>β(β)</td><td>FP</td><td>(µ, σ)</td><td>β(β)</td><td>FP</td><td>(μ, σ)</td><td>β(β)</td><td>RP</td><td>(µ, σ)</td><td></td><td> $\beta ( \bar { \beta } , \hat { \sigma } )$ </td></tr><tr><td>WS1-M</td><td>-0.448</td><td>D11</td><td>(25.5, 8.9)</td><td>1.091</td><td>D22</td><td>(18.8, 4.6)</td><td>-0.068</td><td>S12</td><td>(13.8, 2.0)</td><td>0.236</td><td>D12</td><td>(25.2, 9.9)</td><td></td><td>(0.202, 1.002)</td></tr><tr><td></td><td></td><td>D21</td><td>(22.5, 7.5)</td><td>0.619</td><td>D31</td><td>(16.3, 3.9)</td><td>0.352</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D32</td><td>(16.3, 3.3)</td><td>0.192</td><td>S41</td><td>(18.9, 1.6)</td><td>0.107</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS 1-D</td><td>-1.367</td><td>D31</td><td>(16.2, 3.9)</td><td>0.579</td><td>F22</td><td>(816.7, 292.1)</td><td>0.192</td><td>D22</td><td>(18.7, 4.6)</td><td></td><td>-0.301 D21</td><td></td><td>(22.3, 7.8)</td><td>(1.123, 0.770)</td></tr><tr><td></td><td></td><td>D11</td><td>(25.2, 8.8)</td><td>0.300</td><td>S32</td><td>(17.2, 1.9)</td><td>-0.043</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D13</td><td>(25.3, 8.0)</td><td>0.238</td><td>D12</td><td>(24.9, 9.7)</td><td>-0.102</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS2_M</td><td>-1.673</td><td>D11</td><td>(14.0, 3.4)</td><td>0.979</td><td>F43</td><td>(1144.9, 399.3)</td><td>0.178</td><td>O32</td><td>(0.3, 0.6)</td><td></td><td>-0.1775 F22</td><td></td><td>(720.4, 318.1)</td><td>(0.162, 0.789)</td></tr><tr><td></td><td></td><td>D32</td><td>(20.9, 6.1)</td><td>0.524</td><td>S32</td><td>(17.4, 1.6)</td><td>0.042</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>S23</td><td>(17.7, 1.7)</td><td>0.229</td><td>F11</td><td>(478.6, 315.3)</td><td>-0.163</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS2_D</td><td>-2.824</td><td>S13</td><td>(17.3, 1.9)</td><td>0.6574</td><td>S41</td><td>(18.6, 1.2)</td><td>0.175</td><td>S12</td><td>(16.5, 1.9)</td><td></td><td>-0.1939</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>S23</td><td>(17.8, 1.9)</td><td>0.3781</td><td>D12</td><td>(18.2, 5.9)</td><td>-0.159</td><td>S21</td><td>(15.7, 2.0)</td><td></td><td>-1.0772</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>S22</td><td>(17.0, 1.8)</td><td>0.2201</td><td>O32</td><td>(0.3, 0.6)</td><td>-0.168</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS3_M</td><td>-0.577</td><td>D11</td><td>(28.7, 10.0)</td><td>1.470</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>D22</td><td>(23.0, 6.5)</td><td>(0.335, 0.550)</td></tr><tr><td></td><td></td><td>D21</td><td>(22.4, 6.9)</td><td>1.311</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>D32</td><td>(21.9, 6.6)</td><td>(0.244, 0.342)</td></tr><tr><td></td><td></td><td>F31</td><td>(1342.7, 352.1)</td><td>0.720</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>D13</td><td>(27.9, 10.9)</td><td>(-0.356, 1.514)</td></tr><tr><td></td><td></td><td>D31</td><td>(20.7, 7.5)</td><td>-0.348</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>D12</td><td>(26.2, 8.2)</td><td>(-0.819, 0.008)</td></tr></table>

<table><tr><td>WS3_D</td><td>-1.054</td><td>D21</td><td>(22.2, 6.9)</td><td>0.879</td><td></td><td></td><td></td><td></td><td></td><td>D12</td><td>(26.3, 8.1)</td><td>(0.659, 0.576)</td></tr><tr><td></td><td></td><td>S12</td><td>(16.5, 2.4)</td><td>0.366</td><td></td><td></td><td></td><td></td><td></td><td>D31</td><td>(20.5, 7.4)</td><td>(0.552, 0.597)</td></tr><tr><td></td><td></td><td>D11</td><td>(28.4, 10.1)</td><td>0.311</td><td></td><td></td><td></td><td></td><td></td><td>D23</td><td>(19.7, 6.9)</td><td>(0.052, 0.511)</td></tr><tr><td></td><td></td><td>O13</td><td>(0.5, 1.0)</td><td>-0.351</td><td></td><td></td><td></td><td></td><td></td><td>D32</td><td>(21.7, 6.6)</td><td>(-1.061, 0.888)</td></tr><tr><td>WS4_M</td><td>-2.895</td><td>F11</td><td>(972.8, 346.4)</td><td>1.267</td><td>S11</td><td>(15.0, 1.6)</td><td>-0.904</td><td></td><td></td><td>D21</td><td>(21.5, 5.8)</td><td>(0.8501, 0.474)</td></tr><tr><td></td><td></td><td>D32</td><td>(19.0, 4.8)</td><td>0.917</td><td>S31</td><td>(17.3, 1.6)</td><td>-1.443</td><td></td><td></td><td>D22</td><td>(23.1, 6.0)</td><td>(0.6799, 1.260)</td></tr><tr><td></td><td></td><td>D11</td><td>(15.5, 4.8)</td><td>0.805</td><td></td><td></td><td></td><td></td><td></td><td>D43</td><td>(17.5, 4.1)</td><td>(-0.3750, 2.826)</td></tr><tr><td>WS4_D</td><td>-1.260</td><td>D21</td><td>(21.6, 5.8)</td><td>1.147</td><td>S43</td><td>(19.7, 1.4)</td><td>0.203 D22</td><td>(23.0, 6.1)</td><td>-0.215</td><td>S11</td><td>(14.9, 1.6)</td><td>(0.143, 0.003)</td></tr><tr><td></td><td></td><td>D11</td><td>(15.5, 4.7)</td><td>0.679</td><td>O33 (0.9, 1.1)</td><td></td><td>-0.034</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D41</td><td>(17.2, 4.5)</td><td>0.345</td><td>D43</td><td>(17.5, 4.1)</td><td>-0.061</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS5_M</td><td>-1.489</td><td>D22</td><td>(25.2, 9.2)</td><td>1.145</td><td>D43</td><td>(16.3, 4.5)</td><td>0.519 D33</td><td>(16.3, 5.3)</td><td>-0.363</td><td>S53</td><td>(18.5, 1.3)</td><td>(-0.351, 1.218)</td></tr><tr><td></td><td></td><td>F21</td><td>(1319.0, 401.4)</td><td>0.893</td><td>D21</td><td>(24.6, 8.6)</td><td>0.105</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D31</td><td>(12.7, 3.6)</td><td>0.537</td><td>D23</td><td>(14.9, 5.2)</td><td>-0.124</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS5_D</td><td>-2.940</td><td>D23</td><td>(14.9, 5.4)</td><td>0.468</td><td>D41</td><td>(16.0, 4.7)</td><td>0.284 F23</td><td>(536.4, 282.7)</td><td>-0.618</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D42</td><td>(17.8, 4.8)</td><td>0.422</td><td>D11</td><td>(10.1, 3.2)</td><td>-0.075 S42</td><td>(18.9, 1.6)</td><td>-0.658</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D31</td><td>(12.5, 3.6)</td><td>0.306</td><td>D51</td><td>(15.6, 5.0)</td><td>-0.188</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS6_M</td><td>-1.377</td><td>D22</td><td>(18.3, 6.3)</td><td>1.130</td><td>D42</td><td>(22.7, 6.6)</td><td>0.361</td><td>(13.0, 2.2)</td><td>0.075</td><td>D41</td><td>(19.3, 5.5)</td><td>(0.448, 1.059)</td></tr><tr><td></td><td></td><td>D11</td><td>(15.4, 7.1)</td><td>1.023</td><td>D12</td><td>(17.1, 6.7)</td><td>0.146</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>S23</td><td>(16.1, 2.0)</td><td>0.562</td><td>D23</td><td>(12.4, 4.1)</td><td>0.082</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS6_D</td><td>-3.264</td><td>D21</td><td>(15.7, 5.7)</td><td>0.664</td><td>S42</td><td>(15.9, 1.8)</td><td>-0.038 D12</td><td>(17.1, 6.5)</td><td>-1.395</td><td>D11</td><td>(15.1, 7.0)</td><td>(1.351, 0.931)</td></tr><tr><td></td><td></td><td>D41</td><td>(19.3, 5.5)</td><td>0.088</td><td>O12</td><td>(0.1, 0.2)</td><td>-0.177</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>S11</td><td>(13.1, 2.3)</td><td>0.049</td><td>S41</td><td>(15.4, 2.0)</td><td>-0.355</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WS7_M</td><td>-0.475</td><td>D31</td><td>(13.7, 5.4)</td><td>0.727</td><td></td><td></td><td></td><td></td><td></td><td>D12</td><td>(23.1, 9.4)</td><td>(0.931, 0.851)</td></tr><tr><td></td><td></td><td>M32</td><td>(19.5, 1.8)</td><td>0.685</td><td></td><td></td><td></td><td></td><td></td><td>D21</td><td>(15.8, 6.7)</td><td>(0.849, 1.811)</td></tr><tr><td></td><td></td><td>D23</td><td>(12.2, 3.4)</td><td>0.389</td><td></td><td></td><td></td><td></td><td></td><td>D32</td><td>(18.3, 5.7)</td><td>(0.385, 0.837)</td></tr><tr><td></td><td></td><td>S51</td><td>(17.9, 1.6)</td><td>-0.091</td><td></td><td></td><td></td><td></td><td></td><td>S31</td><td>(16.2, 1.9)</td><td>(-0.774, 0.347)</td></tr><tr><td>WS7_D</td><td>-1.769</td><td>D21</td><td>(15.5, 6.7)</td><td>0.620</td><td>D12</td><td>(22.7, 9.1)</td><td>-0.080</td><td>(18.0, 1.6)</td><td>-0.327</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D31</td><td>(13.5, 5.3)</td><td>0.485</td><td>S41</td><td>(17.2, 1.8)</td><td>S51 -0.166 D33</td><td>(15.8, 4.6)</td><td>-0.682</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>D22</td><td>(16.2, 4.9)</td><td>0.403</td><td>D11</td><td>(14.5, 5.8)</td><td>-0.171</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Note: M and D represent merging and diverging respectively. V i refers to the $i ^ { t h }$ critical variable. $\times i j$ stands for factor X at the $j ^ { t h }$ segment of the $i ^ { t h }$ lane, where X can be D (average density), S (average speed), F (flow rate), or O (maximum speed). CT stands for Constant term, FP stands for Fixed parameter, and RP stands for Random parameter.

## References

Afifah, F., Guo, Z.M., 2025. Optimal speed limit control for network mobility and safety: a twin-delayed deep deterministic policy gradient approach. TRANSPORTMETRICA B-TRANSPORT DYNAMICS 13(1).

Airaldi, F., De Schutter, B., Dabiri, A., 2025. Reinforcement Learning With Model Predictive Control for Highway Ramp Metering. IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS 26(5), 5988-6004.

Arman, M.A., Tampere, C.M.J., 2022. Lane-level trajectory reconstruction based on data-fusion. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 145.

Chavoshi, K., Ferrara, A., Kouvelas, A., 2023. A feedback linearization approach for coordinated traffic flow management in highway systems. CONTROL ENGINEERING PRACTICE 139.

Chen, D., Ahn, S., 2018. Capacity-drop at extended bottlenecks: Merge, diverge, and weave. Transportation Research Part B: Methodological 108, 1-20.

Chen, Y.-Y., Cheng, Y., Chang, G.-L., 2021. Lane Group–Based Traffic Model for Assessing On-Ramp Traffic Impact. Journal of Transportation Engineering, Part A: Systems 147(2), 04020152.

Chen, Y.S., Cheng, G.Z., Meng, F.W., Wang, W.Z., 2025. Variable speed limit control method of freeway merging area in a car-truck mixed heterogeneous traffic environment. PHYSICA A-STATISTICAL MECHANICS AND ITS APPLICATIONS 676.

Cheng, G.Z., Wang, S., Xu, L., Guo, Y.Z., Wang, Q., 2025. Nonlinear dynamics of mixed traffic flow under variable speed limit control at CAV dedicated lane entrances: A macroscopic modeling approach. PHYSICS LETTERS A 563.

Daganzo, C.F., 1994. The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory. Transportation Research Part B: Methodological 28(4), 269-287.

Frejo, J.R.D., Papamichail, I., Papageorgiou, M., De Schutter, B., 2019. Macroscopic modeling of variable speed limits on freeways. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 100, 15-33.

Gao, J., Yu, B., Chen, Y., Wang, J., Dai, Z., Gao, K., 2025. Meta-MSCC: A foundation model for adaptive CAV control in highway weaving segments. Transportation Research Part C: Emerging Technologies 181, 105397.

Golob, T.F., Recker, W.W., Alvarez, V.M., 2004. Safety aspects of freeway weaving sections. Transportation Research Part A: Policy and Practice 38(1), 35-51.

Greguric, M., Kusic, K., Ivanjko, E., 2022. Impact of Deep Reinforcement Learning on Variable Speed Limit strategies in connected vehicles environments. ENGINEERING APPLICATIONS OF ARTIFICIAL INTELLIGENCE 112.

Han, L., Zhang, L., Pan, H.X., 2025. Improved multi-agent deep reinforcement learning-based integrated control for mixed traffic flow in a freeway corridor with multiple bottlenecks. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 174.

Han, Y., Wang, M., He, Z., Li, Z.B., Wang, H., Liu, P., 2021. A linear Lagrangian model predictive controller of macro- and micro- variable speed limits to eliminate freeway jam waves. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 128.

He, Z.L., Wang, L., Su, Z.C., Ma, W.J., 2024. Integrating variable speed limit and ramp metering to enhance vehicle group safety and efficiency in a mixed traffic environment. PHYSICA A-STATISTICAL MECHANICS AND ITS APPLICATIONS 641.

Hegyi, A., De Schutter, B., Hellendoorn, H., 2005. Model predictive control for optimal coordination of ramp metering and variable speed limits. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 13(3), 185-209.

Jin, J., Li, Y., Huang, H., Dong, Y., Liu, P., 2024. A variable speed limit control approach for freeway tunnels based on the model-based reinforcement learning framework with safety perception. Accident Analysis & Prevention 201, 107570.

Jin, J.L., Huang, H.L., Li, Y., Dong, Y.X., Zhang, G.Q., Chen, J.G., 2025. Variable speed limit control strategy for freeway tunnels based on a multi-objective deep reinforcement learning framework with safety perception. EXPERT SYSTEMS WITH APPLICATIONS 267.

Kang, K., Park, N., Park, J., Abdel-Aty, M., 2024. Deep Q-network learning-based active speed management under autonomous driving environments. COMPUTER-AIDED CIVIL AND INFRASTRUCTURE ENGINEERING 39(21), 3225-3242.

Li, D., Lasenby, J., 2024. Imagination-Augmented Reinforcement Learning Framework for Variable Speed Limit Control. IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS 25(2), 1384-1393.

Lu, W., Yi, Z., Gu, Y., Rui, Y., Ran, B., 2023. TD3LVSL: A lane-level variable speed limit approach based on twin delayed deep deterministic policy gradient in a connected automated vehicle environment. Transportation Research Part C: Emerging Technologies 153, 104221.

Lu, W., Yi, Z., Liang, B., Rui, Y., Ran, B., 2024. Improving Traffic Operation of Bottleneck in a Connected and Automated Vehicles Environment: An Integrated Lane-Level Control Method. IEEE Transactions on Intelligent Transportation Systems 25(9), 11675-11688.

Ma, G., Sun, B., Cheng, Z., Yang, W., Zhou, H., Liu, Q., Wang, Z., 2026a. Investigating the formation mechanism of merging/diverging collision risk in short weaving segments: An integrated approach using spatial-temporal risk field and explainable machine learning. Transportation Research Part A: Policy and Practice 209, 105017.

Ma, G., Sun, B., Liang, H., Yang, W., Zhou, H., 2026b. Spatial-temporal risk field-based coupled dynamicstatic driving risk assessment and trajectory planning in weaving segments. Accident Analysis & Prevention 232, 108553.

Ma, G., Sun, B., Yang, W., Yao, Z., 2026c. 1D and 2D trajectory optimization in weaving segments under a unified risk field framework: cooperative merging control strategy towards mixed traffic environment. Transportation Research Part B: Methodological 211, 103505.

Ma, G., Wang, W., Sun, B., Wu, W., Zhou, Y., 2025. Crowdsourced task dispatching for the shared electric vehicle relocation problem: a hybrid variable neighbourhood search and genetic algorithm. Transportmetrica B: Transport Dynamics 13(1), 2490511.

Ma, W.J., He, Z.L., Wang, L., Abdel-Aty, M., Yu, C.H., 2021. Active traffic management strategies for expressways based on crash risk prediction of moving vehicle groups. ACCIDENT ANALYSIS AND PREVENTION 163.

Mao, P.P., Ji, X.K., Qu, X., Li, L.H., Ran, B., 2022. A Variable Speed Limit Control Based on Variable Cell Transmission Model in the Connecting Traffic Environment. IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS 23(10), 17632-17643.

Messmer, A., Papageorgiou, M., 1990. METANET: a macroscopic simulation program for motorway networks. Traffic Engineering & Control 31, 466-470.

Othman, B., De Nunzio, G., Di Domenico, D., Canudas-de-Wit, C., 2022. Analysis of the Impact of Variable Speed Limits on Environmental Sustainability and Traffic Performance in Urban Networks. IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS 23(11), 21766-21776.

Ouyang, P., Liu, P., Guo, Y., Chen, K., 2023. Effects of configuration elements and traffic flow conditions on Lane-Changing rates at the weaving segments. Transportation Research Part A: Policy and Practice 171, 103652.

Peng, C., Xu, C., 2023. A coordinated ramp metering framework based on heterogeneous causal inference. Computer-Aided Civil and Infrastructure Engineering 38(10), 1365-1380.

Qiu, T., Liu, P., Li, Z.B., Xu, C.C., Qiu, K.L., Wang, S.C., 2025. MADDPG-GST for coordinated variable speed limit and ramp metering: A hybrid action deep reinforcement learning approach to bottleneck congestion mitigation. PHYSICA A-STATISTICAL MECHANICS AND ITS APPLICATIONS 680.

Rahmanidehkordi, A., Ghasemi, A.H., 2024. Traffic Density Control for Heterogeneous Highway Systems With Input Constraints. IEEE Control Systems Letters 8, 2787-2792.

Rim, H., Abdel-Aty, M., Mahmoud, N., 2023. Multi-vehicle safety functions for freeway weaving segments using lane-level traffic data. ACCIDENT ANALYSIS AND PREVENTION 188.

Roncoli, C., Papageorgiou, M., Papamichail, I., 2015. Traffic flow optimisation in presence of vehicle automation and communication systems – Part I: A first-order multi-lane model for motorway traffic. Transportation Research Part C: Emerging Technologies 57, 241-259.

Sai, L., 2025. Variable Speed Limit Control Strategy for Bottleneck Areas on Highways Based on Deep Reinforcement

Learning. Shijiazhuang Tiedao University.

Sirmatel, II, Yildirimoglu, M., 2023. Nonlinear model predictive control of large-scale urban road networks via average speed control. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 156.

Spiliopoulou, A., Kontorinaki, M., Papageorgiou, M., Kopelias, P., 2014. Macroscopic traffic flow model validation at congested freeway off-ramp areas. Transportation Research Part C: Emerging Technologies 41, 18-29.

Sun, D., Jamshidnejad, A., Schutter, B.D., 2024. A Novel Framework Combining MPC and Deep Reinforcement Learning With Application to Freeway Traffic Control. IEEE Transactions on Intelligent Transportation Systems 25(7), 6756-6769.

van de Weg, G.S., Hegyi, A., Hoogendoorn, S.P., De Schutter, B., 2019. Efficient Freeway MPC by Parameterization of ALINEA and a Speed-Limited Area. IEEE TRANSACTIONS ON INTELLIGENT TRANSPORTATION SYSTEMS 20(1), 16-29.

Wang, L., Bian, Q., Shu, A., Abuzwidah, M., Xing, Y., Ma, W., 2025. Incorporating Roadside Traffic Data to Predict Lane Change Behaviors within Weaving Segment for Automated Vehicles. Journal of Transportation Engineering, Part A: Systems 151(12), 04025110.

Wang, Y., Yu, X., Guo, J., Papamichail, I., Papageorgiou, M., Zhang, L., Hu, S., Li, Y., Sun, J., 2022. Macroscopic traffic flow modelling of large-scale freeway networks with field data verification: State-ofthe-art review, benchmarking framework, and case studies using METANET. Transportation Research Part C: Emerging Technologies 145, 103904.

Wu, Y.K., Tan, H.C., Qin, L.Q., Ran, B., 2020. Differential variable speed limits control for freeway recurrent bottlenecks via deep actor-critic algorithm. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 117.

Yang, H., Du, L., Zhang, G., 2026. Optimal temporal-spatial variable speed limit profiles for a heterogeneous freeway corridor: A real-time control and physical model-aided reinforcement learning solution method. Transportation Research Part B: Methodological 211, 103486.

Yu, R.J., Abdel-Aty, M., 2014. An optimal variable speed limits system to ameliorate traffic safety risk. TRANSPORTATION RESEARCH PART C-EMERGING TECHNOLOGIES 46, 235-246.

Yuan, R., Abdel-Aty, M., Xiang, Q., 2024. A study on diversion behavior in weaving segments: Individualized traffic conflict prediction and causal mechanism analysis. Accident Analysis & Prevention 205, 107681.

Zhang, C., Ma, W., Zhao, J., Ma, C., Su, Y., Yang, X., 2024a. Destination-Aware Coordinated Ramp Metering for Preventing Off-Ramp Queue Spillover and Mainstream Congestion. IEEE Intelligent Transportation Systems Magazine 16(1), 40-61.

Zhang, L., Ding, H., Feng, Z., Wang, L.W., Di, Y.R., Zheng, X.Y., Wang, S.G., 2024b. Variable speed limit control strategy considering traffic flow lane assignment in mixed-vehicle driving environment. PHYSICA A-STATISTICAL MECHANICS AND ITS APPLICATIONS 656.

Zhang, R.C., Xu, S.L., Yu, R.J., Yu, J.Q., 2024c. Enhancing multi-scenario applicability of freeway variable speed limit control strategies using continual learning. ACCIDENT ANALYSIS AND PREVENTION 204.

Zhang, W.H., Zhang, F., Feng, Z.X., Zhou, H.C., Yue, L.S.S., Xiong, L.J., Cheng, Z.Y., 2025. A coordinated control framework of freeway continuous merging areas considering traffic risks and energy consumption. ACCIDENT ANALYSIS AND PREVENTION 212.

Zhao, J., Liu, P., Xu, C., Bao, J., 2021. Understand the impact of traffic states on crash risk in the vicinities of Type A weaving segments: A deep learning approach. Accident Analysis & Prevention 159, 106293.

Zhou, Y.M., Zhang, M., Zhang, C., Xie, Z.L., Wang, B., 2024. A Novel Traffic Speed Prediction for Road Weaving Sections: Incorporating Traffic Flow Characteristics. PROMET-TRAFFIC & TRANSPORTATION 36(4), 673-689.