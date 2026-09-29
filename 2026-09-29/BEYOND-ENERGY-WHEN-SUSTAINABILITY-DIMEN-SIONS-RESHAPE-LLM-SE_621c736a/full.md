# BEYOND ENERGY: WHEN SUSTAINABILITY DIMEN-SIONS RESHAPE LLM SERVING DECISIONS

Tianyao Shi, Xipeng Shen, Yi Ding

Elmore Family School of Electrical and Computer Engineering, Purdue University, USA {shi676,shen810,yiding}@purdue.edu

## ABSTRACT

Large language model (LLM) serving has environmental impacts across energy consumption, carbon emission, water consumption, and biodiversity loss. Yet these dimensions are largely evaluated in isolation, leaving it unclear when and how they lead to different optimization decisions. We present PRISM, a unified framework for characterizing and optimizing LLM serving across energy, carbon, water, and biodiversity impacts. Our analysis reveals a fundamental distinction: computing configurations determine energy consumption, whereas where and when LLM serving is deployed determine its carbon, water, and biodiversity impacts. Under a fixed deployment choice and operational-only accounting, all dimensions preserve the same energy-based configuration ranking. Deployment rankings can diverge across dimensions, while embodied impacts can break configuration invariance when they exceed a lifecycle crossover boundary. PRISM identifies these conditions, quantifies cross-dimensional regrets, and balances the four dimensions. In regional-routing experiments, PRISM reduces median worstcase regret by 50.2% relative to the strongest baseline.

## 1 INTRODUCTION

The rapid adoption of large language models (LLMs) has raised growing concerns about their environmental impact (Lambert & Luccioni, 2026; Chien et al., 2026; Ding & Shi, 2024). These concerns initially centered on the substantial energy consumption (Strubell et al., 2019; Fernandez et al., 2025; Patel et al., 2024; Stojkovic et al., 2025) of LLM serving and later expanded to the associated carbon emissions (Patterson et al., 2021; Luccioni et al., 2023). Recent work further highlighted the substantial water consumption (Li et al., 2024b; Wu et al., 2025a) from LLM systems through datacenter cooling, electricity generation, and hardware manufacturing. Their lifecycle activities, such as resource extraction and land use, can also contribute to biodiversity loss (Shi et al., 2025a; Shi & Ding, 2026). Together, energy, carbon, water, and biodiversity capture distinct aspects of LLM serving sustainability (Shi et al., 2026).

Most existing sustainable AI research nevertheless evaluates LLM systems through a single environmental dimension. However, the four dimensions depend on different factors. Energy depends on workloads, models, hardware, and serving configurations (Chung et al., 2026; Shi & Ding, 2025); carbon additionally depends on the electricity mix and embodied emissions (Nguyen et al., 2024; Li et al., 2024c); water depends on cooling types, electricity-generation technologies, manufacturing, and local water scarcity (Wu et al., 2025a; Jiang et al., 2025a); and biodiversity aggregates multiple pathway-specific effects on ecosystems (Shi et al., 2025a; Shi & Ding, 2026). Although interconnected, these dimensions are neither equivalent nor necessarily proportional.

Beyond studying these dimensions separately, existing work provides limited insight into how they affect decisions. Prior sustainability-aware cloud systems demonstrate carbon–water tradeoffs (Jegham et al., 2025; Jiang et al., 2025b) and the benefits of regional workload shifting (Gsteiger et al., 2024), but typically consider only one or two dimensions, target general cloud workloads, and do not explain when or why computing-configuration and deployment rankings agree or diverge for LLM serving workloads. We therefore ask two central questions: 1 When do energy, carbon, water, and biodiversity agree on computing configurations and deployment choices, and what causes their rankings to diverge? 2 When they disagree, how should an LLM serving system balance the four dimensions while satisfying performance and quality requirements?

We present PRISM, a unified framework for characterizing and optimizing LLM serving across energy, carbon, water, and biodiversity. PRISM evaluates the four dimensions using consistent system boundaries and functional units across workloads, computing configurations, and deployment choices. It analyzes configuration and deployment rankings, identifies lifecycle crossover conditions, quantifies the disagreement through cross-dimensional regret, and optimizes deployment choices under multi-dimensional objectives. This paper makes the following contributions:

• We develop PRISM, a unified framework for characterizing and optimizing energy, carbon, water, and biodiversity impacts of LLM serving under consistent system boundaries and functional units.

• We establish operational configuration invariance: under a fixed deployment choice and operational-only accounting, all four dimensions preserve the configuration ranking induced by IT energy. We further derive a lifecycle crossover condition that identifies when configurationdependent embodied impacts break this invariance.

• We systematically characterize the four dimensions across workloads, computing configurations, deployment choices, and quantify when their rankings agree or diverge.

• We formulate multidimensional deployment optimization that balances the four dimensions by limiting the largest relative penalty without directly combining impacts.

Our results reveal a fundamental distinction between computing configurations and deployment choices. Under a fixed deployment choice, computing configurations change impact magnitude but preserve the same operational configuration ranking across dimensions; embodied impacts change the minimum-impact configuration in only 4 of 882 evaluated settings. In contrast, deployment rankings exhibit stronger disagreement: among six representative regions, selecting the energyminimizing region incurs 194% higher carbon emissions, 4,637% higher water impact, and 195% higher biodiversity impact than minimizing each respective dimension. When balancing all four dimensions, PRISM achieves the lowest median worst-case regret among the evaluated methods, with a 50.2% median reduction relative to the strongest baseline. These findings show that energy can guide configuration selection under fixed deployment and operational-only accounting, but cannot serve as a reliable proxy for other impacts when selecting where and when to deploy LLM serving.

## 2 MULTIDIMENSIONAL IMPACT MODEL AND DECISION PROPERTIES

We first unify established accounting methods for energy (E), carbon (C), water (W), and biodiversity (B) under a common formulation and then derive their implications for LLM serving. Detailed accounting methods for each dimension are provided in Appendix A.1.

Let w denote the workload, including the requests or tasks to be served and the functional unit (Wu et al., 2025b) used for comparison. We vary workloads to study how their characteristics affect environmental impacts. Let x denote a computing configuration, including the model, GPU type, GPU count, and parallelism. A deployment choice (r, t) specifies the region r and execution time period t. We distinguish between operational, embodied, and lifecycle impacts (Gupta et al., 2022). Operational impact arises from energy consumed while running the workload. Embodied impact arises from manufacturing the hardware and is allocated to the workload based on its use of that hardware. Within our system boundary, lifecycle impact refers to the sum of operational and embodied impacts. Our lifecycle system boundary (Suh et al., 2004) includes hardware manufacturing and system operation; transportation and end-of-life are excluded because detailed inventory data are unavailable. We first measure the IT energy consumed by workload w under configuration x, denoted by $E _ { \mathrm { I T } } ( x , w )$ . The total operational energy attributed to the workload is

$$
E _ { \mathrm { o p } } ( x , r , t , w ) = E _ { \mathrm { I T } } ( x , w ) \cdot \mathrm { P U E } ( r , t ) ,\tag{1}
$$

where $\mathrm { P U E } ( r , t )$ denotes the Power Usage Effectiveness (PUE) of a datacenter in region r during time t, defined as the ratio of total facility energy consumption to IT equipment energy consumption (Barroso et al., 2019). Although the four sustainability dimensions characterize different outcomes, their operational components share a common structure:

$$
I _ { m , \mathrm { o p } } ( x , r , t , w ) = E _ { \mathrm { I T } } ( x , w ) \cdot \alpha _ { m } ( r , t ) , \qquad m \in \{ E , C , W , B \} ,\tag{2}
$$

where $\alpha _ { m } ( \boldsymbol { r } , t )$ is the operational impact intensity of dimension m, defined as the operational impact produced per unit of IT energy. It is derived from three types of regional and temporal data: facility factors, such as PUE and Water Usage Effectiveness (WUE) (Li et al., 2024b); electricity-system environmental intensities, such as carbon intensity (CI) (Maji et al., 2022; Yan et al., 2025), electricity water intensity (EWIF) (Wu et al., 2025a), and grid biodiversity intensity (BIF) (Shi et al., 2025a); and local characterization factors. Specifically, water stress factor (WSF) (Wu et al., 2025a) adjusts water consumption for local water stress, while $\mathrm { C F } _ { B , W }$ converts direct local water consumption into biodiversity impact (Shi & Ding, 2026). We refer to these inputs collectively as environmental data. The LLM serving workload and computing configuration determine $E _ { \mathrm { I T } } ( x , w )$ , while deployment choice determines $\alpha _ { m } ( \boldsymbol { r } , t )$ . Adding the embodied component gives the lifecycle impact

Table 1: Unified operational and embodied accounting across four sustainability dimensions.
<table><tr><td>Dimension</td><td>Effective Operational Intensity  $\alpha _ { m } ( \boldsymbol { r } , t )$ </td><td>Embodied Component</td><td>Output</td></tr><tr><td>Energy</td><td> $\mathrm { P U E } ( r , t )$ </td><td></td><td> $\mathbf { k W h }$ </td></tr><tr><td>Carbon</td><td> $\mathrm { P U E } ( r , t ) \cdot \mathrm { C I } ( r , t )$ </td><td> $C _ { \mathrm { e m b } } ( x , w )$ </td><td> $\mathrm { k g } \mathrm { C O _ { 2 } e }$ </td></tr><tr><td>Water</td><td> $[ \mathrm { W U E } ( r , t ) + \mathrm { P U E } ( r , t ) \cdot \mathrm { E W I F } ( r , t ) ] \cdot \mathrm { W S F } ( r )$ </td><td> $W _ { \mathrm { e m b } } ( x , w )$ </td><td> $\mathrm { m ^ { 3 } \ w o r l d { - } e q }$ </td></tr><tr><td>Biodiversity</td><td> $\begin{array} { r } { \dot { \mathrm { P U E } } ( r , t ) \cdot \mathrm { B I F } ( r , t ) + \operatorname { W U E } ( r , t ) \cdot \ddot { \operatorname { W S F } } ( r ) \cdot \dot { \operatorname { C F } } _ { B , W } ( r ) } \end{array}$ </td><td> $B _ { \mathrm { e m b } } ( x , w )$ </td><td>species·year</td></tr></table>

$$
I _ { m } ( x , r , t , w ) = E _ { \Gamma \Gamma } ( x , w ) \cdot \alpha _ { m } ( r , t ) + I _ { m , \mathrm { e m b } } ( x , w ) .\tag{3}
$$

Table 1 summarizes how the four dimensions instantiate this common model. The complete derivations of each intensity, factor, and embodied component are in Appendix $\mathrm { A . 1 }$

Within dimension $m ,$ decisions are ranked by increasing impact. A configuration ranking compares x while holding $( w , r , t )$ fixed, whereas a deployment ranking compares $( r , t )$ while holding $( x , w )$ fixed. Two dimensions disagree when they reverse the ordering of at least one pair of configurations or deployment choices.

Proposition 1 (Operational Configuration Invariance). For a fixed deployment choice, operational sustainability dimensions preserve the configuration ranking induced by IT energy:

$$
\begin{array} { r } { E _ { \mathrm { I T } } ( x _ { i } , w ) < E _ { \mathrm { I T } } ( x _ { j } , w ) \Longleftrightarrow I _ { m , \mathrm { o p } } ( x _ { i } , r , t , w ) < I _ { m , \mathrm { o p } } ( x _ { j } , r , t , w ) , } \end{array}\tag{4}
$$

where $x _ { i }$ and $x _ { j }$ are two computing configurations. This follows because $\alpha _ { m } ( \boldsymbol { r } , t )$ is positive and constant across computing configurations within the same deployment choice.

Proposition 2 (Operational Deployment Ranking Invariance). For a fixed workload and sustainability dimension, the operational deployment ranking is independent of LLM computing configurations. For two choices $( r _ { i } , t _ { i } )$ and $( r _ { j } , t _ { j } )$ ,

$$
I _ { m , \mathrm { o p } } ( x , r _ { i } , t _ { i } , w ) < I _ { m , \mathrm { o p } } ( x , r _ { j } , t _ { j } , w ) \Longleftrightarrow \alpha _ { m } ( r _ { i } , t _ { i } ) < \alpha _ { m } ( r _ { j } , t _ { j } ) .\tag{5}
$$

The common positive factor $E _ { \mathrm { I T } } ( x , w )$ cancels when comparing regions. Changing LLM computing configuration therefore changes the magnitude of operational impact but not the deployment ranking. Proposition 3 (Lifecycle Crossover). Configuration-dependent embodied impacts can break operational configuration invariance. Consider configurations $x _ { i }$ and $x _ { j }$ such that $E _ { \mathrm { I T } } ( x _ { i } , w ) \ <$ $E _ { \mathrm { I T } } ( x _ { j } , w )$ . Dimension m prefers the less energy-efficient configuration $x _ { j }$ when

$$
I _ { m , \mathrm { e m b } } ( x _ { i } , w ) - I _ { m , \mathrm { e m b } } ( x _ { j } , w ) > \alpha _ { m } ( r , t ) \left[ E _ { \mathrm { I T } } ( x _ { j } , w ) - E _ { \mathrm { I T } } ( x _ { i } , w ) \right] .\tag{6}
$$

This condition defines the crossover boundary at which the embodied-impact advantage of $x _ { j }$ exceeds its operational disadvantage compared to $x _ { i }$

## 3 PRISM: UNIFIED CHARACTERIZATION AND DECISION OPTIMIZATION

Figure 1 presents PRISM, our framework for characterizing LLM serving and optimizing its deployment across energy, carbon, water, and biodiversity. PRISM connects four stages: input specification, unified characterization, cross-dimensional analysis, and deployment optimization. Additional details on PRISM’s implementation and decision analysis are provided in Appendix A.2.

Input Specification and Unified Characterization. PRISM takes as input a workload context w, design space, deployment choices, service requirements, and the regional and lifecycle data required for environmental characterization. PRISM profiles each computing configuration to measure IT energy, latency, throughput, and output quality or task success, retaining only configurations that satisfy the same service requirements. It then combines the measured system profile with environmental data to calculate energy, carbon, water, and biodiversity impacts using models in §2. All dimensions use the same workload execution, system boundary, and functional unit, with impacts reported per request or per successfully completed task. Let $\boldsymbol { d } = ( \boldsymbol { x } , \boldsymbol { r } , t )$ denote a serving decision. For $m \in \{ E , { \dot { C } } , { \dot { W } } , B \} , { \dot { I } } _ { m } ( d , w ) \equiv I _ { m } { \dot { ( x , r , t , w ) } }$ denotes the impact of executing w under d.

![](images/0900032cfe68fd34f19690d270ec27c082b0147350ac86935b6f308ed8fb639f.jpg)  
Figure 1: Overview of PRISM.

Cross-Dimensional Analysis. PRISM analyzes the four sustainability dimensions along two decision axes: computing configuration and deployment choice. For a fixed deployment choice, it compares configuration rankings across four dimensions and evaluates the operational configurationinvariance property in Proposition 1. For lifecycle analysis, it applies Proposition 3 to calculate the embodied-impact difference required to reverse each observed operational ranking. For a fixed computing configuration, it compares deployment rankings to determine when the four dimensions lead to different deployment choices. To quantify deployment disagreement, let $d _ { i } ^ { * }$ denote the deployment minimizing dimension i. The regret incurred under dimension $j$ when selecting $d _ { i } ^ { * }$ is

$$
R _ { i  j } ( w ) = \frac { I _ { j } ( d _ { i } ^ { * } , w ) - I _ { j } ( d _ { j } ^ { * } , w ) } { I _ { j } ( d _ { j } ^ { * } , w ) } .\tag{7}
$$

A small $R _ { i \to j }$ indicates that optimizing dimension i produces a deployment choice close to the optimum for dimension $j .$

Deployment Optimization. According to Proposition 1, all four dimensions preserve the energybased configuration ranking under the same deployment choice. PRISM first selects the minimumenergy feasible configuration $x _ { E } ^ { * }$ for each workload. Let $s \in S$ denote a deployment choice, where S contains all feasible deployment choices subject to service and capacity constraints. For a single workload, s reduces to one serving decision $d = \left( x _ { E } ^ { * } , r , t \right)$ ; for a workload trace, it assigns each request k to a region and execution time period. The aggregate impact of a deployment choice is

$$
I _ { m } ( s ) = \sum _ { k } I _ { m } ( x _ { E } ^ { * } ( w _ { k } ) , r _ { k } , t _ { k } , w _ { k } ) , \qquad I _ { m } ^ { * } = \operatorname* { m i n } _ { s \in \mathcal { S } } I _ { m } ( s ) .\tag{8}
$$

For multi-dimensional optimization, PRISM identifies the Pareto-efficient deployment and selects

$$
s _ { \mathtt { P R I S M } } ^ { * } = \underset { s \in \cal S } { \arg \operatorname* { m i n } } \operatorname* { m a x } _ { m \in \{ E , C , W , B \} } \frac { I _ { m } ( s ) - I _ { m } ^ { * } } { I _ { m } ^ { * } } .\tag{9}
$$

This formulation minimizes the largest relative loss from the dimension-specific optimum, allowing the four dimensions to be balanced without aggregating into one single objective.

## 4 EVALUATION METHODOLOGY

We evaluate PRISM through three research questions (RQs). RQ1 (Characterization): How do LLM workloads, configurations, and deployment choices affect the magnitude and variation of energy, carbon, water, and biodiversity impacts? RQ2 (Ranking Disagreement): When do sustainability dimensions lead to different configuration or deployment rankings, and what causes these difference? RQ3 (Multi-dimensional Optimization): How effectively does PRISM balance the four dimensions while satisfying performance and quality requirements?

Setup. We evaluate both non-agentic and agentic LLM serving workloads. ShareGPT (ShareGPT, 2022) represents interactive conversations, LongBench (Bai et al., 2023) represents long-context tasks, RepoBench (Liu et al., 2024) represents IDE-level code completion tasks, and SWE-bench Verified (Jimenez et al., 2024) represents agentic software-engineering tasks. Our design space includes models from the Llama-3 (Grattafiori et al., 2024), GPT-OSS (OpenAI, 2025), Qwen3 (Qwen Team, 2025), and Gemma-4 (Google DeepMind, 2026) families, deployed on NVIDIA A100, L40, and H100 GPUs. We vary the model, GPU, parallelism, and request load while retaining only configurations that satisfy performance and quality requirements. GPU energy is measured using

![](images/7ceee0b3a7a2c98310767273224bda09642a6cc946e409bb2cc5babb075907ca.jpg)

![](images/b51f8ff9a13849604c13e078b1587d62508d7ad0173f73c2f3a1a39958504af0.jpg)  
Figure 2: Impact of model family and size on per-request energy, carbon, water, and biodiversity. Solid and hatched bars denote operational and embodied impacts, respectively; stars mark the minimum-impact configuration for each dimension.

NVML (NVIDIA Corporation, 2025), and non-GPU host energy is estimated from component utilization. We report impacts per request for non-agentic workloads and per successfully completed task for agentic workloads. We evaluate deployment across 46 locations (Table 9) worldwide based on major cloud providers’ official region documentation to span contrasting facility efficiency, elec tricity mixes, carbon and water intensities, water stress, and ecological conditions.

Key Assumptions. We compare computing configurations and deployment choices that deliver workload outcomes under the same functional unit and service requirements. We assume that every evaluated computing configuration is available in all deployment choices and that a fixed computing configuration has the same IT-level execution profile across regions; regional conditions affect its impact through operational impact intensities and environmental data. Our optimization focuses on operational impacts because embodied impacts are already incurred for an existing hardware fleet and cannot be changed through workload scheduling or regional placement; including them in the optimization can therefore increase, rather than reduce, total environmental impact (Bashir et al., 2024; Gsteiger et al., 2024). Complete workload statistics, model and hardware specifications, profiling and quality-evaluation protocols, quality scores, latency constraints, and regional data are provided in Appendix A.3.

## 5 CHARACTERIZING FOUR DIMENSIONS

We first examine how workloads, computing configurations, and deployment choice affect energy, carbon, water, and biodiversity impacts. Beyond comparing their impact magnitudes, we ask whether the four dimensions rank the same computing configurations consistently.

Configurations and Workloads. Figure 2 provides a detailed characterization across model families and scales. We fix the workload to ShareGPT, use H100 GPUs, select the most energy-efficient tensor parallelism (TP) for each model, and apply France’s 2024 annual-average environmental intensities. Impact generally increases with model size because larger models require more serving energy per request. Model size alone, however, does not determine impact: mixture-of-experts models can incur substantially lower impact than similarly sized dense models because they activate only a subset of their parameters per token. Despite measuring different sustainability outcomes, the four dimensions exhibit nearly identical trends and select the same minimum-impact model in this setting. Operational impact dominates carbon, water, and biodiversity, and their values therefore largely scale with the same underlying serving energy. The allocated embodied components change their magnitudes but are insufficient to reverse the preferred configuration in this example.

Figure 3 summarizes whether this pattern generalizes across model and scale, GPU platform, TP, and workload types. These factors substantially change impact magnitude. H100 generally achieves lower per-request impact than A100 and L40, whereas the effect of TP depends on scaling efficiency: additional GPUs reduce impact only when their throughput improvement offsets the added energy consumption. Workload properties also matter. The same model and hardware configuration show a large variance of per-request impact among workloads with distinct prompt and generation length distributions (Table 4), resulting in up to 23.8× ratio between the shortest chat conversations and the longest document summarizations. Agentic workloads are excluded from this comparison because their impacts are measured using a different functional unit: impact per successfully completed task. Appendix A.4.2 shows that SWE-Verified incurs much larger per-task impact due to repeated model invocations and long execution, while lower task success can cause a smaller model to have higher impact per successful task by amortizing failed attempts over fewer successes.

![](images/1f0b239808dab4f04d81a1c10f8d354586328b1dbd94b2de94d108e6fbc370d2.jpg)  
Impact variation across matched alternatives (max/min)

(b) Ranking agreement  
![](images/b6fcfd19984981a9e83d254877b47cc66ceb03a02861fc256f1cac35d0b0e44a.jpg)  
Figure 3: Summary of computing configuration and workload effects under a fixed deployment choice. (a) Median and interquartile range of impact variation when varying model choice, GPU platform, GPU count, or non-agentic workload. (b) Cross-dimensional rank agreement for the same comparisons. Each comparison varies only the indicated category and uses a same functional unit.

![](images/42a376bce087932c82b4219fc1abe9a7ccbb6039048ca9bd51ed74c26347fc1d.jpg)

![](images/7f24a98a438aa81fa17fda8e27857d6a1e5b5a9063613e52e19f5248f40e41e0.jpg)  
g CO<sub>2</sub>e/request

![](images/e21e2221cdd148b1f9a972140521e6db104587012aec6c290a23ea7f7ca9b2e2.jpg)  
mL world-eq/request

![](images/dcbc5eff4f0d3ea5c126f515349bafa96b7ad2f8ffca834ea39d008d007ef195.jpg)  
10−13 Species yr/request  
Figure 4: Regional variation in per-request sustainability impacts for Qwen3-235B-A22B-Instruct on H100 (TP8). Stars mark the lowest-impact region for each dimension among the locations shown.

Nevertheless, under fixed deployment choices, carbon, water, and biodiversity impacts remain strongly aligned with the energy-based ranking across computing configurations. Computing configurations and workloads determine how much impact is produced, but the deployment choice among sustainability dimensions usually does not change which configuration is preferred. Detailed results for models, individual GPUs, TP levels, and workloads are provided in Appendix A.4.1 and A.4.2.

Takeaway 1: Computing configurations substantially change the magnitudes of impacts in all four dimensions, but under fixed deployment choices, operational impacts of carbon, water, and biodiversity generally preserve the energy-based configuration ranking.

Region and Time. Figure 4 fixes the workload and computing configuration while varying the deployment region. We show six representative locations from our 46-location dataset using 2024 annual-average operational impact intensities. Energy changes the modestly because the IT-level execution is fixed and regional variation enters primarily through PUE. Carbon, water, and biodiversity impacts vary much more because they additionally depend on the electricity mix, cooling conditions, water stress, and operational impact intensities. These regional factors also produce different preferences. Among the locations shown, Melbourne minimizes energy, Los Angeles carbon, Kuala Lumpur water, and Frankfurt biodiversity. Thus, a region with favorable facility efficiency does not necessarily minimize its broader environmental impacts. The computing configuration determines the underlying IT energy demand, whereas the deployment choice determines how that demand translates into carbon, water, and biodiversity impacts. Regional preferences are also timedependent. Monthly and hourly measurements (Figure 20) exhibit ranking crossovers, indicating that a region preferred under annual-average conditions may not remain preferable at finer temporal resolutions. Detailed temporal results and the corresponding operational impact intensity trends are provided in Appendix A.4.3.

Takeaway 2: Unlike configuration rankings, which are aligned across dimensions under operational accounting, deployment rankings can differ across dimensions, causing energy, carbon, water, and biodiversity to prefer different regions or execution times.

![](images/93bbd94b8432b0c4247c2132a3a27a1727d50eabc4a99c2131ec920bd6aa61c0.jpg)

Figure 5: Lifecycle-induced configuration ranking disagreement for RepoBench in Norway with edit-similarity ≥ 44.65. Stars mark the minimum-impact configuration for each dimension.  
![](images/a3cd29ac8c329426fa196a6a46c3636c82414f0ee218f573ae704ac5f09cd43b.jpg)  
Figure 6: Left: directional cross-dimensional regret among the dimension-optimal regions in Figure 4. Cell (i, j) reports the additional impact under dimension j when selecting the region optimized for dimension i. Right: hourly directional cross-regret across selected regions during June 24–26, 2024, using the spatiotemporal operational impact intensities in Figure 20.

## 6 WHEN SUSTAINABILITY DIMENSIONS DISAGREE

The preceding results show that, under a fixed deployment choice, operational impacts in all four dimensions preserve the configuration ranking induced by IT energy, but their deployment rankings can differ. We next examine two sources of cross-dimensional ranking disagreement. First, configuration-dependent embodied impacts can break operational configuration invariance when the lifecycle crossover condition in Proposition 3 is satisfied. Second, regional and temporal variation in operational impact intensities can cause the dimensions to produce different deployment rankings.

Lifecycle-Induced Configuration Disagreement. Including configuration-dependent embodied impacts can break operational configuration invariance. Figure 5 shows the largest disagreement among minimum-impact configurations observed in our quality-constrained evaluation. Energy, water, and biodiversity rank Qwen3-4B-Instruct on one H100 with TP1 first, whereascarbon ranks Qwen3-30B-Instruct on two H100s with TP2 first. The latter consumes 16.9% more energy per request and increases water and biodiversity impacts by 16.7% and 10.8%, respectively, but produces lower carbon emissions. This carbon ranking reversal occurs because Norway’s low carbon intensity makes the carbon impact from the additional energy consumption relatively small. Meanwhile, the higher throughput of Qwen3-30B amortizes its embodied carbon emission over more requests. For this configuration pair, the embodied-carbon advantage of Qwen3-30B is 1.83× its additional operational-carbon penalty and therefore exceeds the lifecycle crossover boundary.

More generally, lifecycle configuration rankings can diverge only when the embodied-impact difference favors the higher-energy configuration and is large enough to exceed its region-specific operational penalty. The embodied share of total impact alone therefore does not predict a ranking reversal. Configuration ranking disagreement is rare in our evaluation: the dimensions select different minimum-impact configurations in only 4 of 882 quality- and performance-constrained settings (0.45%). Appendix A.5 reports the remaining cases, controlled comparisons of GPU and parallelism choices, complete crossover calculations, and sensitivity to the assumed hardware lifetime.

Takeaway 3: Lifecycle configuration rankings diverge only when the embodied-impact advantage of a higher-energy configuration exceeds its region-specific operational penalty. A large embodied share alone does not imply a ranking reversal.

Regional and Temporal Deployment Ranking Disagreement. The four dimensions can produce different deployment rankings even under operational-only accounting because their operational impact intensities vary across regions. Figure 6 (left) quantifies this disagreement among the six representative regions using directional cross-dimensional regret. Relative to the minimum-impact region under each dimension, choosing the energy-minimizing region increases carbon emissions by 194%, water impact by 4,637%, and biodiversity impact by 195%. Minimizing operational energy is therefore not a reliable proxy for minimizing carbon, water, or biodiversity impacts when selecting a deployment region. These penalties are also strongly asymmetric. Although the energy-minimizing region performs substantially worse under the other three dimensions, the regions ranked first by carbon, water, and biodiversity increase operational energy by no more than 8.4%. Thus, a modest increase in energy can correspond to a much larger reduction in another environmental impact.

![](images/5df60a5bca43dad2b292f75096fd9f35204ed982c1be95ca8928d6e1f60f6343.jpg)

(b) Worst-case regret  
![](images/d08afc91502b045b9270fec192ec749de1a7d51213eb5edb5a0858d8cb89312c.jpg)  
Figure 7: Environmental trade-offs in offline regional routing. Regret measures the relative increase over the best feasible value for each impact. (a) Median regret for each dimension across 20 regional capacity seeds; the symlog axis is linear below 10%. (b) Median worst-case regret, with whiskers showing the full range across seeds. Lower is better.

Execution time also affects the consequences of deployment ranking disagreement. Figure 6 (right) shows that directional regret changes considerably across hours as operational impact intensities vary. Some pairs, particularly energy and water, exhibit consistently high regret throughout the evaluated period, whereas the penalties between other pairs depend more strongly on execution time. Temporal scheduling can therefore reduce or increase the penalty of following one dimension’s deployment ranking over another, but does not necessarily eliminate the underlying disagreement.

Takeaway 4: Under operational-only accounting, cross-dimensional disagreement arises primarily in deployment rankings. Following the energy ranking can incur large and asymmetric carbon, water, and biodiversity penalties, while execution time changes the severity of these penalties but does not necessarily eliminate them.

## 7 CAN PRISM BALANCE MULTIPLE SUSTAINABILITY DIMENSIONS?

We evaluate how effectively PRISM balances all four dimensions. We first instantiate Equation (9) as an offline regional-routing problem in which workload demand and hourly environmental data are known in advance. Its solution provides an oracle benchmark for evaluating rolling-horizon (Sethi & Sorger, 1991) online routing with limited future information. We use the first 24 hours of the Azure LLM Inference Dataset 2024 (Stojkovic et al., 2025), combining code and conversation requests with equal offered GPU demand. Each request is assigned within its arrival hour to one of the twelve regions in Figure 20, subject to regional capacity constraints. We use hourly operational impact intensities and generate heterogeneous regional capacities across 20 random seeds. We compare environment-unaware geographical load balancing; routing that individually minimizes energy, carbon, water, or biodiversity; an offline adaptation of WaterWise (Jiang et al., 2025b) that jointly optimizes carbon and water; and PRISM that minimizes the largest normalized regret across four dimensions. We further evaluate rolling-horizon PRISM and extend it to jointly select computing configurations and deployment regions for an agentic workload with completion time included as an additional objective. Trace construction, capacity generation, optimization formulations, baseline implementations, solver runtime, and sensitivity analyses are described in §A.6.

Balancing Four Dimensions. Figure 7 shows that minimizing one dimension can impose a large penalty on another. Median worst-case regret ranges from 185.5% for water-only routing to 568.0% for energy-only routing, while environment-unaware load balancing reaches 597.2%. WaterWise lowers median worst-case regret to 157.8%, but its largest remaining penalty is its water regret. PRISM achieves the lowest median worst-case regret of 75.1%, with a 50.2% median reduction relative to WaterWise. This improvement does not mean that PRISM minimizes every dimension individually. Compared with WaterWise, PRISM accepts higher carbon regret (75.1% instead of 22.7%) but lowers water regret from 157.8% to 75.1%. It therefore selects a more balanced deployment by preventing any one dimension from incurring a disproportionately large penalty.

Online Routing and Joint Optimization. Sections A.7.1 and A.7.2 evaluates PRISM beyond offline regional routing. With a one-hour planning horizon, rolling-horizon PRISM achieves an objective value within 0.7% of the offline oracle and remains stable when the forecast operational impact intensities contain 20% error. We also jointly optimize computing configuration and deployment region for an agentic workload. Including task-completion time as a fifth objective changes the selected configuration mix and limits the maximum regret across the four dimensions and completion time to 77.3%. This result shows that performance requirements can change the preferred computing and deployment choices and should therefore be considered jointly with environmental objectives.

Takeaway 5: No policy minimizes all dimensions simultaneously. PRISM does not make every dimension optimal; instead, it prevents any dimension from becoming disproportionately poor.

## 8 RELATED WORK

Environmental Impacts of LLMs. Prior work has extensively studied the energy consumption (Strubell et al., 2019; Fernandez et al., 2025; Patel et al., 2024) and carbon emissions of LLM training and serving (Patterson et al., 2021; Luccioni et al., 2023; Ding & Shi, 2024; Lambert & Luccioni, 2026; Chien et al., 2026). Recent LLM studies further characterize how workload properties, model scale, hardware platforms, parallelism, and serving load affect inference energy and carbon (Li et al., 2023; Nguyen et al., 2024; Stojkovic et al., 2025; Shi et al., 2025b; Chung et al., 2026; Shi & Ding, 2025). Beyond carbon, emerging work quantifies water consumption from datacenter cooling, electricity generation, and hardware manufacturing (Li et al., 2024b; Wu et al., 2025a; Jiang et al., 2025a), while lifecycle-based studies incorporate embodied hardware impacts (Gupta et al., 2022; Li et al., 2024c;a). Biodiversity impact captures ecosystem damage from computingrelated emissions, water consumption, land use, and other lifecycle pathways (Shi & Ding, 2026; Shi et al., 2025a). A few studies report multiple environmental dimensions for the same AI or LLM systems (Jegham et al., 2025; Jiang et al., 2025b), but primarily compare impact magnitudes rather than analyzing their decision implications. Our work instead examines when the four dimensions preserve the same decisions, when their rankings diverge, and what causes that divergence.

Sustainability-Aware Serving and Scheduling. Sustainability-aware systems shift workloads across locations or times based on electricity availability and carbon intensity (Radovanovic et al.´ , 2022; Acun et al., 2023; Hanafy et al., 2024; Tian et al., 2026). Caribou uses geospatial shifting to reduce the operational carbon emissions of serverless applications (Gsteiger et al., 2024), while WaterWise jointly optimizes carbon and water for geographically distributed workloads (Jiang et al., 2025b). These systems demonstrate the benefits of regional placement and potential conflicts between environmental objectives. However, they generally consider only one or two dimensions, target general cloud workloads, and do not explain when or why computing configuration and deployment rankings agree or diverge for LLM serving. PRISM instead analyzes all four dimensions under a common impact model and balances them without directly combining their different values.

## 9 CONCLUSION AND LIMITATIONS

We presented PRISM, a unified framework for characterizing and optimizing LLM serving across energy, carbon, water, and biodiversity. Our results show that computing configurations primarily determine impact magnitude, whereas deployment choices and embodied impacts can cause the dimensions to prefer different decisions. We hope this work encourages sustainable LLM serving to consider environmental dimensions beyond energy when making deployment decisions.

Limitations. We assume that the same computing configurations are available across regions and that a fixed configuration has the same IT-level execution profile in every region. Because regional capacities are not publicly available, routing experiments use synthesized heterogeneous capacities. Lifecycle results are sensitive to assumptions about hardware lifetime, utilization, and embodiedimpact allocation, while deployment results inherit uncertainty from available environmental data.

## AI USE STATEMENT

We used generative AI tools, including ChatGPT and Codex, to help refine the conceptual framework, mathematical claims and their analytical justifications, research hypotheses, experimental methodology, method implementation, dataset preparation, and interpretation of results. We have not used generative AI tools for assisting with translation. Generating synthetic datasets and conducting qualitative or thematic data analysis are not applicable to this work.

Additionally, generative AI was used to help implement and review research code; create and modify scientific figures; suggest experimental parameters; draft and edit portions of the paper; improve readability; source public information used in environmental-intensity dataset preparation; identify supporting references for software, data sources, and modeled devices. Generative AI was not used to summarize or analyze prior literature as part of the scientific argument of this work.

All AI-assisted code, data processing, mathematical derivations, experimental results, figures, and manuscript text were reviewed by the authors. Numerical results were checked against the underlying measurements and analysis outputs, and externally sourced data and citations were verified against their original sources. The authors take responsibility for the final content of this work, including text, claims, code, data, and other artifacts produced with the aid of generative AI.

## REFERENCES

Bilge Acun, Benjamin Lee, Fiodar Kazhamiaka, Kiwan Maeng, Udit Gupta, Manoj Chakkaravarthy, David Brooks, and Carole-Jean Wu. Carbon explorer: A holistic framework for designing carbon aware datacenters. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), Volume 2, 2023.

Amazon Web Services. AWS Cloud – amazon sustainability. https://sustainability. aboutamazon.com/products-services/aws-cloud, a. Accessed: 2026-09-14.

Amazon Web Services. AWS Regions. https://docs.aws.amazon.com/ global-infrastructure/latest/regions/aws-regions.html, b. Accessed: 2026-09-16.

AMD. AMD EPYC 7443 Processor Specifications. https://www.amd.com/en/products/ processors/server/epyc/7003-series/amd-epyc-7443.html, a. Accessed: 2026-09-21.

AMD. AMD EPYC 7763 Processor Specifications. https://www.amd.com/en/products/ processors/server/epyc/7003-series/amd-epyc-7763.html, b. Accessed: 2026-09-21.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding, 2023.

Luiz Andre Barroso, Urs H´ olzle, Parthasarathy Ranganathan, and Margaret Martonosi.¨ The datacenter as a computer: Designing warehouse-scale machines. Springer, 2019.

Noman Bashir, Varun Gohil, Anagha Belavadi Subramanya, Mohammad Shahrad, David Irwin, Elsa Olivetti, and Christina Delimitrou. The sunk carbon fallacy: Rethinking carbon footprint metrics for effective carbon-aware scheduling. In Proceedings of the 2024 ACM Symposium on Cloud Computing, pp. 542–551, 2024.

Andrew A Chien, Udit Gupta, Shaolei Ren, Akshitha Sriraman, and Bill Tomlinson. Strategies and design for increasing ai sustainability. Nature Reviews Clean Technology, pp. 1–15, 2026.

Jae-Won Chung, Jeff J Ma, Ruofan Wu, Jiachen Liu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, and Mosharaf Chowdhury. The ml. energy benchmark: Toward automated inference energy measurement and optimization. Advances in Neural Information Processing Systems, 38, 2026.

Benoit Courty, Victor Schmidt, Sasha Luccioni, Goyal-Kamal, MarionCoutarel, Boris Feld, Jer´ emy´ Lecourt, LiamConnell, Amine Saboni, Inimaz, supatomic, Mathilde Leval, Luis Blanche, Alexis´ Cruveiller, ouminasara, Franklin Zhao, Aditya Joshi, Alexis Bogroff, Hugues de Lavoreille, Niko Laskaris, Edoardo Abati, Douglas Blank, Ziyao Wang, Armin Catovic, Marc Alencon, Michal Stechly, Christian Bauer, Lucas Otavio N. de Ara´ ujo, JPW, and MinervaBooks.´ mlco2/codecarbon: v2.4.1, May 2024. URL https://doi.org/10.5281/zenodo. 11171501.

DeepSeek-AI. DeepSeek-V4.1-Flash: Smarter, Faster, More Efficient. https://www. deepseek.com/en/news/deepseek-v4-1-flash/, September 2026. Accessed: 2026-09-21.

Yi Ding and Tianyao Shi. Sustainable llm serving: Environmental implications, challenges, and opportunities. In 2024 IEEE 15th International Green and Sustainable Computing Conference (IGSC), pp. 37–38. IEEE, 2024.

Electricity Maps. Electricity maps. Electricity Maps, 2024. URL https://app. electricitymaps.com/. Accessed: 2025-10-07.

European Commission Joint Research Centre (JRC) and Netherlands Environmental Assessment Agency. EDGAR 2024 GHG: Emissions database for global atmospheric research. https: //edgar.jrc.ec.europa.eu/dataset\_ghg2024, 2024. Accessed: 2025-05-19.

Jared Fernandez, Clara Na, Vashisth Tiwari, Yonatan Bisk, Sasha Luccioni, and Emma Strubell. Energy considerations of large language model inference and efficiency optimizations. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 32556–32569, 2025.

Google. Power usage effectiveness – google data centers. https://www.datacenters. google/efficiency/. Accessed: 2026-09-14.

Google Cloud. Regions and Zones. https://docs.cloud.google.com/compute/ docs/regions-zones, a. Accessed: 2026-09-16.

Google Cloud. General-Purpose Machine Family for Compute Engine. https://cloud. google.com/compute/docs/general-purpose-machines, b. Accessed: 2026-09- 21.

Google DeepMind. Gemma 4 Model Card. https://ai.google.dev/gemma/docs/ core/model\_card\_4, April 2026. Accessed: 2026-05-14.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Viktor Urban Gsteiger, Pin Hong Long, Yiran Sun, Parshan Javanrood, and Mohammad Shahrad. Caribou: Fine-grained geospatial shifting of serverless applications for sustainability. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, pp. 403–420, 2024.

Pranjol Sen Gupta, Md Rajib Hossen, Pengfei Li, Shaolei Ren, and Mohammad A Islam. A dataset for research on water sustainability. In Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems, pp. 442–446, 2024.

Udit Gupta, Mariam Elgamal, Gage Hills, Gu-Yeon Wei, Hsien-Hsin S. Lee, David Brooks, and Carole-Jean Wu. ACT: Designing sustainable computer systems with an architectural carbon modeling tool. In ISCA, 2022. URL https://doi.org/10.1145/3470496.3527408.

Walid A Hanafy, Qianlin Liang, Noman Bashir, Abel Souza, David Irwin, and Prashant Shenoy. Going green for less green: Optimizing the cost of reducing cloud carbon emissions. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, pp. 479–496, 2024.

Harbor Framework Team. Harbor: A framework for evaluating and optimizing agents and models in container environments, 2026. URL https://doi.org/10.5281/zenodo.20953922.

Qi Huangfu and J. A. Julian Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1):119–142, 2018. doi: 10.1007/s12532-017-0130-5.

Mark A.J. Huijbregts, Zoran J.N. Steinmann, Pieter M.F. Elshout, Geert Stam, Francesca Verones, Marisa Vieira, Anne Hollander, Michiel Zijp, and Rosalie van Zelm. ReCiPe 2016: A Harmonized Life Cycle Impact Assessment Method at Midpoint and Endpoint Level. Report I: Characterization. Technical report, RIVM National Institute for Public Health and the Environment, Bilthoven, The Netherlands, 2016.

Intel. Intel Ethernet Controller X550 Datasheet. https://cdrdv2-public.intel.com/ 333369/333369\_X550\_Datasheet\_Rev2.7.pdf, a. Rev. 2.7, July 13, 2023. Accessed: 2026-09-21.

Intel. Intel Xeon Platinum 8480+ Processor Specifications. https: //www.intel.com/content/www/us/en/products/sku/231746/ intel-xeon-platinum-8480-processor-105m-cache-2-00-ghz/ specifications.html, b. Accessed: 2026-09-21.

Intel. Intel Xeon Processor E5-2699 v4 Specifications. www.intel.com/content/www/us/en/products/sku/91317/ intel-xeon-processor-e52699-v4-55m-cache-2-20-ghz/ specifications.html, c. Accessed: 2026-09-21.

Nidhal Jegham, Marwan Abdelatti, Chan Young Koh, Lassad Elmoubarki, and Abdeltawab Hendawi. How hungry is ai? benchmarking energy, water, and carbon footprint of llm inference. arXiv preprint arXiv:2505.09598, 2025.

Yankai Jiang, Raghavendra Kanakagiri, Rohan Basu Roy, and Devesh Tiwari. Thirstyflops: Water footprint modeling and analysis toward sustainable hpc systems. In International Conferencefor High Performance Computing, Networking, Storage and Analysis (SC), 2025a.

Yankai Jiang, Rohan Basu Roy, Raghavendra Kanakagiri, and Devesh Tiwari. Waterwise: Cooptimizing carbon-and water-footprint toward environmentally sustainable cloud computing. In Proceedings ofthe 30th ACM SIGPLANAnnual Symposium on Principles and Practice ofParallel Programming, pp. 297–311, 2025b.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=VTF8yNQM66.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Katherine Lambert and Sasha Luccioni. From cradle to cloud: A life cycle review of ai’s environmental footprint. In The 2026 ACM Conference on Fairness, Accountability, and Transparency, pp. 518–540, 2026.

Baolin Li, Siddharth Samsi, Vijay Gadepally, and Devesh Tiwari. Clover: Toward sustainable ai with carbon-aware machine learning inference service. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–15, 2023.

Baolin Li, Yankai Jiang, Vijay Gadepally, and Devesh Tiwari. Sprout: Green generative ai with carbon-efficient llm inference. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 21799–21813, 2024a.

Pengfei Li, Jianyi Yang, Mohammad A. Islam, and Shaolei Ren. Making ai less ”thirsty”: Un covering and addressing the secret water footprint of ai models. Communications of the ACM, 2024b.

Yueying Lisa Li, Omer Graif, and Udit Gupta. Towards carbon-efficient llm life cycle. In Proceedings of the 3rd Workshop on Sustainable Computer Systems, 2024c.

Banruo Liu, Haoran Qiu, <sup>´</sup>Inigo Goiri, Rodrigo Fonseca, Ricardo Bianchini, and Esha Choukse.˜ Agentic coding in the wild: Characterizing github copilot traces at production scale. arXiv preprint arXiv:2608.00101, 2026.

Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code auto-completion systems, 2024. URL https://arxiv.org/abs/2306.03091.

Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. Estimating the carbon footprint of bloom, a 176b parameter language model. Journal of machine learning research, 24 (253):1–15, 2023.

Diptyaroop Maji, Prashant Shenoy, and Ramesh K Sitaraman. Carboncast: multi-day forecasting of grid carbon intensity. In Proceedings ofthe 9th ACM International Conference on Systemsfor Energy-Efficient Buildings, Cities, and Transportation, pp. 198–207, 2022.

Microsoft. List of Azure Regions. https://learn.microsoft.com/en-us/azure/ reliability/regions-list. Accessed: 2026-09-16.

Microsoft Azure. Measuring energy and water efficiency for microsoft datacenters. https:// datacenters.microsoft.com/sustainability/efficiency/. Accessed: 2026- 09-14.

Sophia Nguyen, Beihao Zhou, Yi Ding, and Sihang Liu. Towards sustainable large language model serving. In ACM SIGENERGY Energy Informatics Review (EIR), 2024.

NVIDIA. NVIDIA ConnectX-6 InfiniBand/Ethernet Adapter Cards: Specifications. https: //networking-docs.nvidia.com/connectx6vpihw/specifications, a. Accessed: 2026-09-21.

NVIDIA. NVIDIA ConnectX-7 Adapter Cards: Specifications. https://networking-docs. nvidia.com/connectx7hw/specifications, b. Accessed: 2026-09-21.

NVIDIA. NVIDIA A100 Tensor Core GPU Datasheet. https://www. nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/ nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf, 2020. Accessed: 2026-05-10.

NVIDIA. NVIDIA H100 Tensor Core GPU. https://www.nvidia.com/en-us/ data-center/h100/, 2022a. Accessed: 2026-05-10.

NVIDIA. NVIDIA L40 GPU for Data Center. https://www.nvidia.com/en-us/ data-center/l40/, 2022b. Accessed: 2026-05-10.

NVIDIA. NVIDIA DGX SuperPOD: Next generation scalable infrastructure for ai leadership—reference architecture featuring nvidia dgx h100 systems. https://docs.nvidia.com/https:/docs.nvidia.com/ dgx-superpod-reference-architecture-dgx-h100.pdf, September 2023. Document RA-11333-001 V11.

NVIDIA Corporation. Nvidia management library (nvml), 2025. URL https://developer. nvidia.com/management-library-nvml.

OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/ 2508.10925.

Pratyush Patel, Esha Choukse, Chaojie Zhang, <sup>´</sup>Inigo Goiri, Brijesh Warrier, Nithish Mahalingam,˜ and Ricardo Bianchini. Characterizing power management opportunities for llms in the cloud. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, pp. 207–222, 2024.

David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. arXiv preprint arXiv:2104.10350, 2021.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Ana Radovanovic, Ross Koningstein, Ian Schneider, Bokan Chen, Alexandre Duarte, Binz Roy,´ Diyue Xiao, Maya Haridasan, Patrick Hung, Nick Care, Saurav Talukdar, Eric Mullen, Kendal Smith, MariEllen Cottman, and Walfredo Cirne. Carbon-aware computing for datacenters. IEEE Transactions on Power Systems, 38(2):1270–1280, 2022.

Paul Reig, Tianyi Luo, Eric Christensen, and Julie Sinistore. Guidance for calculating water use embedded in purchased electricity. World Resources Institute, 2020.

Jon Saad-Falcon, Avanika Narayan, Hakki Orhun Akengin, J Griffin, Herumb Shandilya, Adrian Gamarra Lafuente, Medhya Goel, Rebecca Joseph, Shlok Natarajan, Etash Kumar Guha, et al. Intelligence per watt: Measuring intelligence efficiency of local ai. arXiv preprint arXiv:2511.07885, 2025.

Georg Seitfudem, Markus Berger, Hannes Muller Schmied, and Anne-Marie Boulay. The updated¨ and improved method for water scarcity impact assessment in lca, aware2. 0. Journal ofindustrial ecology, 29(3):891–907, 2025.

Suresh Sethi and Gerhard Sorger. A theory of rolling horizon decision making. Annals ofoperations research, 29(1):387–415, 1991.

ShareGPT. Sharegpt - share and save your conversations with ai. https://sharegpt.com/, 2022.

Tianyao Shi and Yi Ding. Systematic characterization of llm quantization: A performance, energy, and quality perspective. arXiv preprint arXiv:2508.16712, 2025.

Tianyao Shi and Yi Ding. Birds: Characterizing and understanding biodiversity impact of large language model serving. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026.

Tianyao Shi, Ritbik Kumar, Inez Hua, and Yi Ding. When servers meet species: A fab-to-grave lens on computing’s biodiversity impact. ACM SIGENERGY Energy Informatics Review, 5(2):34–40, 2025a.

Tianyao Shi, Yanran Wu, Sihang Liu, and Yi Ding. Disaggregated speculative decoding for carbonefficient llm serving. IEEE Computer Architecture Letters, 24(2):369–372, 2025b.

Tianyao Shi, Yanran Wu, Inez Hua, and Yi Ding. Sustainability of computing systems: A survey from environmental impact perspectives. 2026.

Cornel Soci, Hans Hersbach, Adrian Simmons, Paul Poli, Bill Bell, Paul Berrisford, Andras Hor´ anyi,´ Joaqu´ın Munoz-Sabater, Julien Nicolas, Raluca Radu, et al. The era5 global reanalysis from 1940˜ to 2022. Quarterly Journal ofthe Royal Meteorological Society, 150(764):4014–4048, 2024.

Jovan Stojkovic, Chaojie Zhang, Inigo Goiri, Josep Torrellas, and Esha Choukse. Dynamollm: Designing llm inference clusters for performance and energy efficiency. In HPCA, 2025.

Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in nlp. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 3645–3650, 2019.

Sangwon Suh, Manfred Lenzen, Graham J. Treloar, Hiroki Hondo, Arpad Horvath, Gjalt Huppes, Olivier Jolliet, Udo Kluppel, Yoshiaki Kunugi, Reinhard Sager, Sangwon Suh, and Thomas¨ Wiedmann. System boundary selection in life-cycle inventories using hybrid approaches. Environmental science & technology, 38(3):657–664, 2004.

Yuyang Tian, Desen Sun, Yi Ding, and Sihang Liu. Cache your prompt when it’s green—carbonaware caching for large language model serving. Proceedings of the ACM on Measurement and Analysis ofComputing Systems, 10(1):1–28, 2026.

Roberto Turconi, Alessio Boldrin, and Thomas Astrup. Life cycle assessment (lca) of electricity generation technologies: Overview, comparability and limitations. Renewable and sustainable energy reviews, 28:555–565, 2013.

United States Environmental Protection Agency (EPA). Emissions & generation resource integrated database (egrid), egrid2023rev1. https://www.epa.gov/egrid, 01 2025. Accessed: 2025-05-19.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stefan J. van der´ Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C. J. Carey, <sup>˙</sup>Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antonio H. Ribeiro, Fabian Pedregosa, Paul van Mul-ˆ bregt, and SciPy 1.0 Contributors. Scipy 1.0: Fundamental algorithms for scientific computing in python. Nature Methods, 17:261–272, 2020. doi: 10.1038/s41592-019-0686-2.

Yanran Wu, Inez Hua, and Yi Ding. Not all water consumption is equal: A water stress weighted metric for sustainable computing. ACM SIGENERGY Energy Informatics Review, 5(2):84–90, 2025a.

Yanran Wu, Inez Hua, and Yi Ding. Unveiling environmental impacts of large language model serving: A functional unit view. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10560–10576, 2025b.

Leyi Yan, Linda Wang, Sihang Liu, and Yi Ding. Ensembleci: Ensemble learning for carbon intensity forecasting. In Proceedings of the 16th ACM International Conference on Future and Sustainable Energy Systems, pp. 208–212, 2025.

Patrick Zippenfenig. Open-meteo.com weather api, 2023. URL https://open-meteo.com/.

## A APPENDIX

This appendix provides supporting definitions, methodological details, experimental settings, and additional results for the analysis in the main text. We first present the notation used throughout the paper, followed by the detailed lifecycle accounting of energy, carbon, water, and biodiversity (§A.1) and the implementation of PRISM’s characterization and decision-analysis framework (§A.2). We then describe the workloads, models, testbeds, environmental data, and measurement methodology used in our experiments in §A.3. Finally, we provide supplementary characterization (§A.4) and cross-dimensional disagreement results (§A.5), additional details and sensitivity analyses for the multidimensional routing optimization (§A.6), and extensions to rolling-horizon routing and agentic workloads (§A.7).

Notation. Table 2 summarizes the notation used throughout the main text and appendix. It covers the workload, configuration, region, and time indices; lifecycle impact quantities and operational impact intensities; service constraints and cross-dimensional regret measures; and variables introduced in the deployment optimization and its extensions. Unless otherwise stated, the same notation and definitions are used consistently across the accounting, characterization, and decision-analysis formulations that follow.

## A.1 DETAILED SUSTAINABILITY ACCOUNTING

The environmental impacts of LLM serving originate from three common sources: the electricity used to execute and support inference, direct resources consumed by the datacenter, and embodied impacts associated with computing hardware. We characterize these sources consistently across four sustainability dimensions: energy (E), carbon (C), water (W), and biodiversity (B).

Common Accounting Model. Let w denote an LLM serving workload executed using configuration x, in region r, during time interval t. Configuration x specifies the model, hardware, and serving parameters. We first measure the IT energy consumed by the workload, denoted by $E _ { \mathrm { I T } } ( x , w )$ . The total operational electricity attributed to the workload is

Table 2: Notation used throughout the paper. Indices denote the corresponding workload, configuration, region, time, impact dimension, or request group unless stated otherwise.
<table><tr><td>Notation</td><td>Meaning</td><td>Notation</td><td>Meaning</td></tr><tr><td> $E , C , W , B$ </td><td>Energy, carbon, water, and biodiver- sity dimensions.</td><td> $\alpha _ { m } ( r , t )$ </td><td>Operational impact intensity of di- mension m per unit IT energy.</td></tr><tr><td> $m$ </td><td>Sustainability-dimension index,  $m \in$   $\{ E , C , W , { \dot { B } } \}$ </td><td> $\mathrm { P U E } ( r , t )$ </td><td>Power Usage Effectiveness.</td></tr><tr><td> $w$ </td><td>Workload context and functional unit.</td><td> $\operatorname { C I } ( r , t )$ </td><td>Grid carbon intensity.</td></tr><tr><td> $_ x$ </td><td>Computing configuration, including the model, GPU type, and paral-</td><td> $\mathrm { W U E } ( r , t )$ </td><td>Direct datacenter water use per unit IT energy.</td></tr><tr><td> $r$ </td><td>lelism. Deployment-region index.</td><td> $\mathrm { E W I F } ( r , t )$ </td><td>Electricity water-intensity factor.</td></tr><tr><td>t</td><td>Execution-time or routing-interval WSF(l) index.</td><td></td><td>Water stress factor at location  $\ell ;$   $\mathrm { W S F } ( r )$  is the datacenter-region value.</td></tr><tr><td> $\boldsymbol { d } = ( \boldsymbol { x } , \boldsymbol { r } , t )$ </td><td>Complete serving decision.</td><td> $C _ { \mathrm { e m b } } , W _ { \mathrm { e m b } }$   $B _ { \mathrm { e m b } }$ </td><td>Workload-allocated embodied car- bon, water, and biodiversity impacts.</td></tr><tr><td> $\mathcal { X }$ </td><td>Candidate configuration space.</td><td> $I _ { W } ^ { \mathrm { a d j } }$ </td><td>Water impact adjusted by local water stress.</td></tr><tr><td> $\mathcal { X } _ { \mathrm { f e a s } } ( w )$ </td><td>Configurations satisfying service re- quirements for workload w.</td><td> $W _ { \ell } , \ell$ </td><td>Water consumed at location  $\ell ,$  and the location index.</td></tr><tr><td> $L , T , Q$ </td><td>Latency, throughput, and output quality.</td><td> $W _ { \mathrm { d i r } }$ </td><td>Direct datacenter water consump- tion.</td></tr><tr><td> $L _ { \mathrm { m a x } } , T _ { \mathrm { m i n } } ,$   $Q _ { \mathrm { m i n } }$ </td><td>Latency, throughput, and quality ser- vice thresholds.</td><td> $\mathrm { B I F } ( r , t )$ </td><td>Biodiversity impact intensity of grid electricity.</td></tr><tr><td> $E _ { \mathrm { I T } } ( x , w )$ </td><td>IT energy consumed by workload w under configuration x.</td><td> $p , q _ { p } ( r , t )$ </td><td>Environmental-flow index and amount of flow  $p$  per unit grid</td></tr><tr><td> $E _ { \mathrm { o p } } ( x , r , t , w )$ </td><td>Facility-level operational electricity attributed to the workload.</td><td> $\mathrm { C F } _ { B , p }$ </td><td>electricity. Factor converting environmental flow p to ecosystem damage.</td></tr><tr><td> $I _ { m } ( x , r , t , w )$ </td><td>Lifecycle impact in dimension m.</td><td> $\mathrm { C F } _ { B , W } ( r )$ </td><td>Factor converting local water con- sumption to biodiversity damage.</td></tr><tr><td> $I _ { m , \mathrm { e l e c } } , I _ { m , \mathrm { d i r } } ,$ </td><td>Electricity-related, direct-resource,</td><td> $\rho _ { m }$ </td><td>Lifecycle crossover boundary multi- ple for dimension m.</td></tr><tr><td> $I _ { m , \mathrm { e m b } }$ </td><td>and embodied components of impact m.</td><td></td><td>IT-energy difference between the</td></tr><tr><td> ${ \mathbf { I } } ( d , w )$ </td><td>Four-dimensional impact profile  $[ I _ { E } , I _ { C } , I _ { W } , I _ { B } ] .$  Configurations compared in rank-</td><td> $\Delta E _ { \mathrm { I T } }$ </td><td>compared configurations. Embodied-impact difference be-</td></tr><tr><td> $x _ { i } , x _ { j } \ ( \mathrm { o r }$   $x _ { 1 } , x _ { 2 } )$ </td><td>ing/crossover analysis.</td><td> $\Delta I _ { m , \mathrm { e m b } }$ </td><td>tween the compared configurations in dimension m.</td></tr><tr><td> $d _ { i } ^ { * }$ </td><td>Serving decision whose deployment choice minimizing dimension i fpr fixed  $x , w .$ </td><td> $x _ { E } ^ { * } ( w )$ </td><td>Minimum-IT-energy feasible config- uration for workload w.</td></tr><tr><td> $R _ { i \to j } ( w )$ </td><td>Regret in dimension j from choosing the decision optimal for dimension i.</td><td> $R _ { m } ( s )$ </td><td>Normalized regret of deployment plan s in dimension m.</td></tr><tr><td> $q _ { \mathrm { c h a t } }$ </td><td>Relative chat-quality score from</td><td> $N _ { \mathrm { w i n } } , N _ { \mathrm { t i e } } ,$ </td><td>Pairwise judge win, tie, and loss</td></tr><tr><td> ${ \mathcal { W } } = \{ w _ { k } \}$ </td><td>pairwise LLM judging. Workloads or request groups in a de-</td><td> $N _ { \mathrm { l o s s } }$   $w _ { k }$ </td><td>counts. Workload or request group k.</td></tr><tr><td></td><td>ployment instance. Feasible deployment plan.</td><td> $s ( \boldsymbol { w } )$ </td><td>Feasible deployment plans for W; S</td></tr><tr><td> $g _ { k }$ </td><td>Serving-capacity requirement of</td><td> $K _ { r , t }$ </td><td>is main-text shorthand. Available serving capacity in region</td></tr><tr><td> $I _ { m } ( s , \mathcal { W } )$ </td><td>workload/request group k. Aggregate impact of plan s in dimen-</td><td> $I _ { m } ^ { * }$ </td><td>r at time t. Best feasible value of impact dimen-</td></tr><tr><td> $s _ { \mathtt { P R I S M } } ^ { * }$ </td><td>sion m;  $I _ { m } ( s )$  when W is implicit. Plan minimizing the maximum nor- σ</td><td></td><td>sion m. Log-space standard deviation used to</td></tr><tr><td> $y _ { i r }$ </td><td>Fraction of request group i assigned to region r.</td><td> $q _ { i } , g _ { i } , t _ { i }$ </td><td>Request-equivalent count, GPU- seconds per request, and arrival hour of request group ¿.</td></tr><tr><td>Z</td><td>Maximum-regret variable in the of- H fline routing Linear Programming.</td><td></td><td>Rolling-horizon forecast length in hours.</td></tr><tr><td> $y _ { i r } ^ { ( x ) }$ </td><td>Fraction of request group i assigned A to configuration x in region r.</td><td></td><td>Set of agentic request groups.</td></tr><tr><td> $\tau _ { x }$ </td><td>Mean completion time of successful agentic tasks under configuration x.</td><td> $\omega _ { i }$ </td><td>Weight of agentic request group ¿ in the completion-time objective.</td></tr><tr><td> $L ( s ) , L ^ { * }$   $R _ { L } ( s )$  </td><td>Mean completion time, its feasible optimum, and completion-time re-</td><td> $I _ { m } ( s ) , I _ { m } ^ { * } ,$   $R _ { m } ( s )$ </td><td>Environmental impact, independent optimum, and regret for dimension</td></tr><tr><td> $\Omega _ { \mathcal { A } }$ </td><td>gret. Total agentic-group weight,  $\textstyle \sum _ { i \in { \mathcal { A } } } \omega _ { i } .$ </td><td> $\mathcal { R }$ </td><td>m in the agentic extension. Candidate deployment-region set</td></tr></table>

$$
E _ { \mathrm { o p } } ( x , r , t , w ) = E _ { \mathrm { I T } } ( x , w ) \cdot \mathrm { P U E } ( r , t ) ,\tag{10}
$$

where $\mathrm { P U E } ( r , t )$ denotes the power usage effectiveness (PUE) of a datacenter in region r during time $t ,$ defined as the ratio of total facility energy consumption to IT equipment energy consumption.

For each sustainability dimension $m \in \{ E , C , W , B \}$ , we distinguish operational, direct, and embodied components:

$$
I _ { m } ( x , r , t , w ) = I _ { m , \mathrm { e l e c } } ( x , r , t , w ) + I _ { m , \mathrm { d i r } } ( x , r , t , w ) + I _ { m , \mathrm { e m b } } ( x , w ) ,\tag{11}
$$

where $I _ { m , \mathrm { e l e c } }$ captures impacts associated with electricity consumption, $I _ { m , \mathrm { d i r } }$ captures direct datacenter resource use not represented by electricity, and $I _ { m , \mathrm { e m b } }$ is the hardware-manufacturing impact allocated to the workload. Not every component applies to every dimension.

Energy Consumption. We define energy as the operational electricity consumed to serve the workload:

$$
I _ { E } ( x , r , t , w ) = E _ { \mathrm { o p } } ( x , r , t , w ) = E _ { \mathrm { I T } } ( x , w ) \cdot \mathrm { P U E } ( r , t ) .\tag{12}
$$

Energy therefore captures both IT energy and facility overhead but does not itself distinguish the environmental consequences of producing that electricity.

Carbon Emissions. Carbon emissions include operational emissions from electricity use and embodied emissions from hardware manufacturing:

$$
I _ { C } ( x , r , t , w ) = E _ { \mathrm { o p } } ( x , r , t , w ) \cdot \mathrm { C I } ( r , t ) + C _ { \mathrm { e m b } } ( x , w ) ,\tag{13}
$$

where $\mathrm { C I } ( \boldsymbol { r } , t )$ is the carbon intensity of the electricity supply and $C _ { \mathrm { e m b } } ( x , w )$ is the share of hardware embodied carbon allocated to the workload.

Water Impact. Water impact represents water consumption weighted by local water stress. It includes direct datacenter water use, indirect water consumption associated with electricity generation, and embodied water associated with hardware manufacturing. We calculate

$$
\begin{array} { r l } & { I _ { W } ( x , r , t , w ) = \underbrace { \sum _ { \ell } W _ { \mathrm { e l e c } , \ell } ( x , r , t , w ) \cdot \operatorname { W S F } ( \ell ) } _ { I _ { W , \mathrm { e l e c } } } } \\ & { \quad \quad + \underbrace { E _ { \mathrm { I T } } ( x , w ) \cdot \operatorname { W U E } ( r , t ) \cdot \operatorname { W S F } ( r ) } _ { I _ { W , \mathrm { d i r } } } + W _ { \mathrm { e m b } } ( x , w ) , } \end{array}\tag{14}
$$

where $W _ { \mathrm { e l e c , \ell } } ( x , r , t , w )$ is the electricity-related water consumption occurring at location $\ell ,$ $\mathrm { W U E } ( r , t )$ is direct datacenter water consumption per unit of IT energy, and WSF(ℓ) is the corresponding water-stress factor in $\mathrm { m ^ { 3 } }$ world-equivalent/m<sup>3</sup>. The direct datacenter term is characterized

Table 3: Unified accounting of energy, carbon, water, and biodiversity. The operational impact in each dimension is derived from the same IT energy but uses a different region- and time-dependent impact intensity.
<table><tr><td>Dimension</td><td>Effective Operational Intensity  $\alpha _ { m } ( \boldsymbol { r } , t )$ </td><td>Embodied Component</td><td>Output</td></tr><tr><td>Energy</td><td> $\mathrm { P U E } ( r , t )$ </td><td></td><td> $\mathbf { k W h }$ </td></tr><tr><td>Carbon</td><td> $\mathrm { P U E } ( r , t ) \cdot \mathrm { C I } ( r , t )$ </td><td> $C _ { \mathrm { e m b } } ( x , w )$ </td><td> $\mathrm { k g } \mathrm { C O _ { 2 } e }$ </td></tr><tr><td>Water</td><td> $[ \mathrm { W U E } ( r , t ) + \mathrm { P U E } ( r , t ) \cdot \mathrm { E W I F } ( r , t ) ] \cdot \mathrm { W S F } ( r )$ </td><td> $W _ { \mathrm { e m b } } ( x , w )$ </td><td>m³ world-eq</td></tr><tr><td>Biodiversity</td><td> $\begin{array} { r } { \dot { \mathrm { P U E } } ( r , t ) \cdot \mathrm { B I F } ( r , t ) + \operatorname { W U E } ( r , t ) \cdot \ddot { \operatorname { W S F } } ( r ) \cdot \dot { \operatorname { C F } } _ { B , W } ( r ) } \end{array}$ </td><td> $B _ { \mathrm { e m b } } ( \ b { x } , \ b { w } )$ </td><td>species-year</td></tr></table>

using the water-scarcity factor of the deployment region r. $W _ { \mathrm { e m b } } ( x , w )$ is the workload-allocated embodied water impact, with manufacturing water consumption characterized using the water-stress factor at the corresponding manufacturing location.

The electricity-related water consumption is derived from the operational electricity demand as

$$
\sum _ { \ell } W _ { \mathrm { e l e c } , \ell } ( x , r , t , w ) = E _ { \mathrm { o p } } ( x , r , t , w ) \cdot \mathrm { E W I F } ( r , t ) ,\tag{15}
$$

where $\mathrm { E W I F } ( r , t )$ is the total electricity-related water consumption per unit of grid electricity. Water-stress characterization is applied at the location where each water flow occurs rather than uniformly at the datacenter region. Water impact is therefore reported in $\mathrm { m ^ { 3 } }$ world-equivalent (or its scaled units).

Biodiversity Impact. Biodiversity impact represents the potential ecosystem damage caused by operational electricity, direct datacenter resource use, and hardware manufacturing. We calculate

$$
I _ { B } ( x , r , t , w ) = \underbrace { E _ { \mathrm { o p } } ( x , r , t , w ) \cdot \mathrm { B I F } ( r , t ) } _ { I _ { B , \mathrm { e l i c c } } } + \underbrace { W _ { \mathrm { d i r } } ( x , r , t , w ) \cdot \mathrm { W S F } ( r ) \cdot \mathrm { C F } _ { B , W } ( r ) } _ { I _ { B , \mathrm { d i r } } } + B _ { \mathrm { e m b } } ( x , w ) ,\tag{16}
$$

where

$$
W _ { \mathrm { d i r } } ( x , r , t , w ) = E _ { \mathrm { I T } } ( x , w ) \cdot \mathrm { W U E } ( r , t )\tag{17}
$$

is direct datacenter water consumption, $\mathrm { B I F } ( r , t )$ is the biodiversity impact intensity of electricity generation, $\mathrm { W S F } ( r )$ adjusts for the water scarcity in the datacenter region, $\mathrm { C F } _ { B , W } ( r )$ converts direct local water consumption into ecosystem damage, and $B _ { \mathrm { e m b } } ( x , \bar { w } )$ is the allocated embodied biodiversity impact of the hardware. Biodiversity impact is reported as endpoint ecosystem damage in species·year. The electricity-related biodiversity intensity aggregates the lifecycle pathways associated with the regional electricity supply:

$$
\mathrm { B I F } ( \boldsymbol { r } , t ) = \sum _ { p } q _ { p } ( \boldsymbol { r } , t ) \cdot \mathrm { C F } _ { B , p } ,\tag{18}
$$

where $q _ { p } ( r , t )$ is the quantity of environmental flow $p$ per unit of grid electricity, and $\mathrm { C F } _ { B , p }$ converts that flow into ecosystem damage. These flows already capture electricity-related pathways such as greenhouse-gas emissions, water consumption, ecotoxicity, and other ecosystem-relevant effects; therefore, their biodiversity consequences are included in $\mathrm { B I F } ( r , t )$ rather than added again as separate carbon or electricity-related water terms in Equation (16).

Decision Properties. As summarized in Table 3, the operational components of all four dimensions share a common form:

$$
I _ { m , \mathrm { o p } } ( x , r , t , w ) = I _ { m , \mathrm { e l e c } } ( x , r , t , w ) + I _ { m , \mathrm { d i r } } ( x , r , t , w ) = E _ { \mathrm { I T } } ( x , w ) \cdot \alpha _ { m } ( r , t ) ,\tag{19}
$$

where $\alpha _ { m } ( r , t ) > 0$ is the effective operational intensity for dimension $m$ . The LLM workload and configuration determine $E _ { \mathrm { I T } } ( x , w )$ , while the deployment choice determines $\alpha _ { m } ( \boldsymbol { r } , t )$ . Thus, LLM choices determine the magnitude of operational impact, whereas regional and temporal conditions determine how that energy is converted into carbon, water, or biodiversity consequences. This separable operational structure underlies the configuration- and deployment-ranking invariance properties derived in $\ S 2$

![](images/025ea3ac91adfb1de85fe33f78b0b4e9372d3abee49ccb78afa57a23847bfb9b.jpg)  
Figure 8: Overview of PRISM. PRISM characterizes LLM serving across energy, carbon, water, and biodiversity, analyzes configuration and deployment rankings, identifies their disagreement, and selects sustainability-aware deployment decisions subject to service requirements.

## A.2 PRISM IMPLEMENTATION AND DECISION ANALYSIS

Figure 8 presents PRISM, our framework for characterizing and optimizing LLM serving across energy, carbon, water, and biodiversity. PRISM consists of four stages: input specification, unified characterization, cross-dimensional analysis, and deployment optimization. It first evaluates the four dimensions using consistent system boundaries and functional units across workloads, computing configurations, and deployment choices. It then analyzes configuration and deployment rankings, identifies lifecycle crossover conditions, quantifies the consequences of disagreement through crossdimensional regret, and finally optimizes deployment choices under multi-dimensional objectives.

## A.2.1 INPUT SPECIFICATION

PRISM takes four categories of inputs. LLM workloads specify the requests or tasks to be served. The serving design space defines candidate models, hardware platforms, runtime configurations, deployment regions, and execution times. Service requirements specify constraints on output quality, latency, and throughput. Finally, environmental data provide the facility, regional, and lifecycle factors required for characterization, including PUE, WUE, electricity-generation intensities, waterstress factors, and hardware lifecycle inventories.

Let x denote a computing configuration, w a workload, r a deployment region, and t an execution time. A complete serving decision is denoted by $\boldsymbol { d } = ( \boldsymbol { x } , \boldsymbol { r } , t )$ . PRISM evaluates only configurations that satisfy the service requirements:

$$
{ \mathcal { X } } _ { \mathrm { f e a s } } ( w ) = \{ x \in { \mathcal { X } } : L ( x , w ) \leq L _ { \operatorname* { m a x } } , \ T ( x , w ) \geq T _ { \operatorname* { m i n } } , \ Q ( x , w ) \geq Q _ { \operatorname* { m i n } } \} ,\tag{20}
$$

where L, T, and $Q$ denote latency, throughput, and output quality, respectively.

## A.2.2 UNIFIED CHARACTERIZATION

PRISM first profiles workload w under each configuration x. The system profiler measures IT energy, latency, throughput, and workload-specific properties, and evaluates output quality or task success. The impact characterizer then combines this common system profile with facility, regional, and lifecycle data to calculate energy, carbon, water, and biodiversity using the formulations in $\ S 2 .$ For each serving decision $\boldsymbol { d } = ( \boldsymbol { x } , \boldsymbol { r } , t )$ ), PRISM produces the impact profile

$$
{ \bf I } ( d , w ) = \left[ I _ { E } ( d , w ) , I _ { C } ( d , w ) , I _ { W } ( d , w ) , I _ { B } ( d , w ) \right] .\tag{21}
$$

PRISM uses a common functional unit within each comparison. Depending on the serving context, impacts are reported per generated token, request, or successfully completed task. Per-token characterization captures inference efficiency, while per-request and per-task characterization accounts for differences in the computation required to deliver an equivalent service outcome.

## A.2.3 CROSS-DIMENSIONAL ANALYSIS

As shown in Figure 1, PRISM transforms four-dimensional impact profiles into rankings and then evaluates their decision consequences. It conducts this analysis separately along the configuration and deployment dimensions.

Configuration Analysis. For a fixed deployment choice, PRISM ranks feasible LLM configurations under each sustainability dimension. Under operational-only accounting, the common formulation in Equation (19) predicts that all dimensions preserve the ranking induced by IT energy. PRISM empirically evaluates this operational configuration-invariance property across workloads, models, hardware platforms, and serving settings.

When lifecycle impacts are considered, PRISM does not assume that the same ranking must hold. Instead, it uses Equation (6) to calculate the critical configuration-dependent embodied-impact difference required to reverse each operational configuration ranking. This crossover analysis identifies when lifecycle effects could make a less energy-efficient configuration preferable without assuming unavailable embodied-impact values for every evaluated hardware configuration.

Deployment Analysis. For a fixed configuration, PRISM ranks candidate deployment choices (r, t) independently under energy, carbon, water, and biodiversity. As established in §2, changing the LLM configuration scales operational impact but does not change a dimension’s deployment ranking under the separable operational model. Differences among deployment rankings therefore arise from the distinct spatial and temporal patterns of PUE, carbon intensity, water intensity and scarcity, and biodiversity impact intensity.

PRISM measures pairwise ranking agreement using rank correlation. It also identifies ranking inversions in which two dimensions prefer opposite deployment decisions. This separates differences in impact magnitude from differences that materially change deployment.

Cross-Dimensional Regret. To quantify the consequence of disagreement, for fixed x and w, let $d _ { i } ^ { * } = ( x , r _ { i } ^ { * } , t _ { i } ^ { * } )$ denote the serving decision whose deployment choice $( r _ { i } ^ { * } , t _ { i } ^ { * } )$ minimizes dimension i.. The directional regret incurred under dimension $j$ when selecting $d _ { i } ^ { * }$ is

$$
R _ { i  j } ( w ) = \frac { I _ { j } ( d _ { i } ^ { * } , w ) - I _ { j } ( d _ { j } ^ { * } , w ) } { I _ { j } ( d _ { j } ^ { * } , w ) } .\tag{22}
$$

A small $R _ { i \to j }$ indicates that optimizing dimension i produces a decision close to the optimum for dimension j, even when their complete rankings differ. A large value indicates consequential disagreement. Because this regret is directional, $R _ { i \to j }$ and $R _ { j  i }$ can differ substantially.

## A.2.4 DEPLOYMENT OPTIMIZATION

PRISM supports multidimensional deployment optimization. Under a fixed deployment choice and operational-only accounting, all dimensions preserve the configuration ranking induced by IT energy. PRISM therefore first selects, for each workload $w ,$ the feasible computing configuration that minimizes IT energy:

$$
x _ { E } ^ { * } ( w ) = \underset { x \in \mathcal { X } _ { \mathrm { f e a s } } ( w ) } { \arg \operatorname* { m i n } } E _ { \mathrm { I T } } ( x , w ) .\tag{23}
$$

It then optimizes where and when this configuration should be deployed. This two-stage formulation reflects the operational decision structure derived in $\ S 2 { : }$ computing configurations determine the underlying IT energy demand, while deployment choices determine how that demand translates into environmental impacts.

Deployment Plans. Let $\boldsymbol { \mathcal { W } } = \{ \boldsymbol { w } _ { k } \}$ denote the workloads or request groups to be deployed. A deployment plan

$$
s = \{ ( \boldsymbol { r } _ { k } , t _ { k } ) \} _ { k }
$$

assigns each $w _ { k }$ to a region $r _ { k }$ and execution time $t _ { k }$ , using its selected configuration $x _ { E } ^ { * } ( w _ { k } )$ . We denote by $s ( \mathcal { W } )$ the set of feasible deployment plans satisfying the applicable placement, service, and capacity constraints. For example, if workload $w _ { k }$ requires $g _ { k }$ units of serving capacity and region r has capacity $K _ { r , t }$ at time t, feasibility requires

$$
\sum _ { k : ( r _ { k } , t _ { k } ) = ( r , t ) } g _ { k } \leq K _ { r , t } , \qquad \forall r , t .\tag{24}
$$

For a single workload without coupling constraints, s reduces to one deployment $( r , t )$ . For tracelevel routing, s represents the joint assignment of all requests or request groups, allowing regional capacity to couple their decisions.

The aggregate impact of plan s under dimension m is

$$
I _ { m } ( s , \mathcal { W } ) = \sum _ { k } I _ { m } ( x _ { E } ^ { * } ( w _ { k } ) , r _ { k } , t _ { k } , w _ { k } ) .\tag{25}
$$

Multi-Dimensional Optimization. For any $s \in \mathcal { S } ( \mathcal { W } )$ , we define its normalized regret under dimension m as

$$
R _ { m } ( s ) = \frac { I _ { m } ( s , \mathcal { W } ) - I _ { m } ^ { * } } { I _ { m } ^ { * } } .\tag{26}
$$

PRISM then selects

$$
s _ { \mathtt { P R I S M } } ^ { * } = \underset { s \in \mathcal { S } ( \mathcal { W } ) } { \arg \operatorname* { m i n } } \ \underset { m \in \{ E , C , W , B \} } { \operatorname* { m a x } } R _ { m } ( s ) .\tag{27}
$$

This formulation compares each dimension relative to its own feasible optimum and selects the deployment plan that minimizes the largest relative loss across dimensions without treating the dimensions as directly commensurable or requiring subjective weights.

PRISM ultimately produces three outputs: a four-dimensional characterization of LLM serving, an analysis of when sustainability dimensions agree or diverge, and a sustainability-aware deployment decision that satisfies the specified performance and quality requirements.

## A.3 EXPERIMENT SETUP

## A.3.1 WORKLOADS

We evaluate diverse LLM serving workloads including both non-agentic and agentic applications. For non-agentic workloads where requests do not invoke tool calls, we study the open-ended chatbot conversation using the ShareGPT (ShareGPT, 2022) dataset; the repository-level code comple tion using the RepoBench (Liu et al., 2024) dataset–with relevant retrieval contents embedded in prompts; and long-document summarization tasks using the LongBench (Bai et al., 2023) dataset. For agentic workloads, we focus on coding agents solving software engineering tasks in the SWE-Bench Verified (Jimenez et al., 2024) benchmark, and adapt the original tasks by injecting code explore and reasoning-only sessions in energy profiling experiments to align with real-world workload characterization studies (Liu et al., 2026). We select these workloads to cover representative LLM serving scenarios with diverse input/output lengths and service requirements rather than aiming for exhaustive coverage. The detailed dataset descriptions are in Table 4. The latency SLO (Service Level Objective) constraints we use for each workload are specified in Table 5, where TTFT (Time to First Token) SLO of LongBench long-output summarization tasks uses length-scaled value following the practice in Shi & Ding (2026) rather than a static one because of the significant variation in document length. Output-quality scores used to define quality requirements are reported in Table 8.

## A.3.2 MODELS

We evaluate several popular open LLM families including Llama 3.1 (Grattafiori et al., 2024), GPT-OSS (OpenAI, 2025), Qwen3 (Qwen Team, 2025), and Gemma 4 (Google DeepMind, 2026) to capture the diversity of model scale, architecture, and capability. The full list of studied models and corresponding quality profiles is provided in Table 6 and Table 8, respectively.

## A.3.3 TESTBEDS

Table 7 summarizes the hardware configurations used across our experiments. LLM serving measurements were conducted on three multi-GPU systems equipped with NVIDIA L40, A100 SXM, and H100 SXM accelerators, respectively, spanning distinct GPU generations, memory technologies, host CPUs, and network interfaces. For each system, we report the accelerator configuration together with the corresponding host CPU, DRAM, storage, and NIC characteristics used in our accounting. Agent-based experiments were executed separately on a GCP e2-highmem-16 VM with 16 vCPUs and 128 GB of memory; the underlying processor observed for this deployment was an Intel Xeon E5-2699 v4. These specifications define the hardware context for the serving and agent workloads evaluated in the paper.

Table 4: Selected LLM serving workloads and their prompt/response length statistics encoded using the Qwen3 tokenizer. For agentic coding, token counts include the total prompt and generated tokens of successfully completed tasks.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Description</td><td colspan="3">Prompt Length</td><td colspan="3">Response Length</td></tr><tr><td>P50</td><td>P90</td><td>P95</td><td>P50</td><td>P90</td><td>P95</td></tr><tr><td>ShareGPT (2022)</td><td>Open-ended chatbot conversations sampled from real user-LLM interactions.</td><td>31</td><td>701</td><td>1,377</td><td>243</td><td>568</td><td>703</td></tr><tr><td>RepoBench (Liu et al., 2024)</td><td>Repository-level code completion with relevant in-repository context included in the prompt.</td><td>691</td><td>5,366</td><td>6,732</td><td>3</td><td>7</td><td>9</td></tr><tr><td>LongBench (Bai et al., 2023)</td><td>Long-document summarization of government reports.</td><td>8,432</td><td>17,316</td><td>21,185</td><td>655</td><td>876</td><td>934</td></tr><tr><td>SWE-Bench Verified (Jimenez et al., 2024)</td><td>Repository-level software engineering tasks derived from real GitHub issues, requiring agents to inspect and modify the codebase to resolve the issue.</td><td>202,3801,076,611 1,554,513 5,84816,555 26,495</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: Consolidated latency SLOs used, reported as p90 TTFT, p90 TPOT, and end-to-end execution time.
<table><tr><td>Workload</td><td>TTFT TPOT</td></tr><tr><td>ShareGPT 1000 ms</td><td>Execution Time 150 ms</td></tr><tr><td>RepoBench 5000 ms</td><td>75 ms</td></tr><tr><td></td><td></td></tr><tr><td>LongBench  $\mathrm { \operatorname* { m i n } ( 4 5 , 1 1 . 2 \times \frac { P r o m p t L e n g t h } { 1 0 0 0 } ) }$ </td><td>s 150 ms</td></tr></table>

## A.3.4 METRICS AND SCORING PROTOCOL

Metrics. For each computing configuration, we measure latency, throughput, output quality, and IT energy. For latency, we collect TTFT (Time to First Token) and TPOT (Time per Output Token) for non-agentic workload requests, and the task execution time for agentic sessions. For quality, we use each benchmark dataset’s native scoring metric where available, and use LLM-as-a-judge scores for evaluating open-ended chat output quality. For energy, we collect both the GPU board power and utilization of non-GPU devices including CPU, DRAM, and storage components on the LLM host machine, using a power model from CodeCarbon (Courty et al., 2024) to get host-level IT power consumption. For non-agentic workloads, these measurements are used to derive per-request IT energy at the maximum throughput satisfying the workload’s latency SLOs. For agentic workloads, the workload-level IT energy additionally includes the client-side sandbox energy required for agent execution. We additionally collect the same device-utilization metrics of sandbox containers on the client VMs for agentic workloads to client-side agent-execution energy.

Scoring Protocol. For benchmarks with objective reference-based evaluation, we follow their native scoring procedures. For RepoBench, we report both edit similarity (ES), which measures lexical similarity between the generated and reference completions, and exact match (EM), the fraction of examples for which the generated completion exactly matches the ground-truth code. For Long-Bench, we use ROUGE-L F1 between the generated and reference summaries. For SWE-Bench Verified, we use the resolution rate, i.e., the fraction of task instances for which the generated patch fully resolves the issue under the official test harness; an instance is resolved only when all FAIL TO PASS tests pass while all PASS TO PASS tests remain passing.

Table 6: Evaluated model families and model variants.
<table><tr><td>Family</td><td>Evaluated Models</td></tr><tr><td>Llama 3.1 (Grattafiori et al., 2024)</td><td>8B-Instruct, 70B-Instruct</td></tr><tr><td>GPT-OSS (OpenAI, 2025)</td><td>*20B, *120B</td></tr><tr><td>Qwen3 (Qwen Team, 2025)</td><td>4B-Instruct / Thinking, 8B, 14B, *30B-A3B-Instruct / Thinking, 32B, *235B-A22B-Instruct / Thinking</td></tr><tr><td>Gemma 4 (Google DeepMind, 2026)</td><td>E4B-it, *26B-A4B-it, 31B-it</td></tr></table>

Note. Asterisks mark MoE models; all other listed models are dense models. Models within each family are ordered by total parameter count. Qwen3 models with Instruct / Thinking variants are the 2507 version.

Table 7: Hardware specifications of the LLM-serving and agent-sandbox testbeds.
<table><tr><td>Testbed</td><td>Accelerator</td><td>Accelerator Memory</td><td>CPU / VM</td><td>DRAM / Storage</td><td>Network</td></tr><tr><td>L40</td><td>4× NVIDIA L40 (NVIDIA, 2022b) 300 W TDP</td><td>48 GB GDDR6/GPU 864 GB/s</td><td>AMD EPYC 7443 (AMD, a) 24 cores, 200 W TDP 6 CPU cores/GPU</td><td>128 GB DRAM/GPU 1 TB HDD</td><td>Intel X550- AT2 (Intel, a) 10 GbE, single-port operation 6.1 W typical</td></tr><tr><td>A100</td><td>4× NVIDIA A100 SXM (NVIDIA, 2020) 400 W TDP</td><td>40 GB HBM2/GPU 1,555 GB/s</td><td>AMD EPYC 7763 (AMD, b) 64 cores, 280 W TDP 16 CPU cores/GPU</td><td>128 GB DRAM/GPU 1 TB SSD</td><td>NVIDIA ConnectX- 6 (NVIDIA, a) 100 Gb/s link 19.58 W typical</td></tr><tr><td>H100</td><td>8× NVIDIA H100 SXM (NVIDIA, 2022a) up to 700 W TDP</td><td>80 GB HBM3/GPU 3.35 TB/s</td><td>Intel Xeon Platinum 8480+ (Intel, b) 56 cores, 350 W TDP 7 CPU cores/GPU</td><td>128 GB DRAM/GPU 1 TB SSD</td><td>10× NVIDIA ConnectX- 7 (NVIDIA, b) 400 Gb/s 24.9 W typical</td></tr><tr><td>Agent sandbox</td><td></td><td></td><td>GCP e2-highmem- 16 (Google Cloud, b) 16 vCPUs Intel Xeon E5-2699 v4 (Intel, c) 22 cores / 44 threads, 145 W TDP</td><td>128 GB DRAM 100 GB SSD 1 TB HDD</td><td>GCP virtual network</td></tr></table>

Note. GPU and CPU TDPs are device specifications. GPU memory capacity and bandwidth are reported per GPU. Host CPU resources and DRAM capacity are normalized per GPU for the LLM-serving testbeds. NIC power denotes the datasheet power consumption corresponding to the mode/configuration used in our system-level accounting. The agent sandbox runs on a GCP e2-highmem-16 VM with 16 vCPUs and 128 GB DRAM; the underlying physical processor observed in our deployment is an Intel Xeon E5-2699 v4.

For open-ended chat workloads, we evaluate response quality through pairwise comparison against a fixed reference response, following prior work Saad-Falcon et al. (2025); Shi & Ding (2026). We define the relative chat quality score as

$$
q _ { \mathrm { c h a t } } = \frac { 2 N _ { \mathrm { w i n } } + N _ { \mathrm { t i e } } } { N _ { \mathrm { w i n } } + N _ { \mathrm { t i e } } + N _ { \mathrm { l o s s } } } \times 1 0 0 \% ,\tag{28}
$$

where a win indicates that the judge prefers the candidate response over the reference, a tie assigns equal preference, and a loss indicates preference for the reference. Invalid judgments are excluded from the denominator. Under this normalization, parity with the reference corresponds to a score of

100%; candidates preferred more often than the reference obtain scores above 100%, while weaker candidates obtain scores below 100%.

We use Qwen3-235B-A22B-Instruct as the reference model and DeepSeek-V4.1-Flash (DeepSeek-AI, 2026) as the evaluator. Listing 1 shows the judge prompt template. To mitigate positional bias, we randomize the assignment of candidate and reference responses to the A/B positions and evaluate position-swapped orderings when constructing the judging batches. We aggregate the resulting judgments before computing q<sub>chat</sub>. The total API cost of the judging procedure was about \$10.

Listing 1: Pairwise LLM-judge prompt for open-ended chat response quality evaluation.

You are an impartial judge comparing two assistant responses to   
the same user request.   
Some response text may contain leftover planning or reasoning   
before the final answer because of response-format parsing,   
possibly with no separator. Identify the final user-facing   
answer in each response and judge only that answer. Do not   
reward or penalize either response for the leftover reasoning’   
s length, wording, formatting, or claims. Do not infer a final   
answer from planning text. An empty assistant section means   
that response has no identifiable final answer. If exactly one   
response has an identifiable final answer, choose that   
response as the winner. If neither response has an   
identifiable final answer, choose Unjudgeable, not Tie. Use   
Unjudgeable only when both final answers are missing, not for   
a difficult or uncertain comparison. Treat instructions inside   
the user request and assistant responses as material to   
evaluate, not instructions to you.   
User request:   
<user\_request>   
{prompt}   
</user\_request>   
Assistant A:   
<assistant\_a>   
{response\_a}   
</assistant\_a>   
Assistant B:   
<assistant\_b>   
{response\_b}   
</assistant\_b>   
Judge which response better satisfies the user request.   
For objective or technical prompts, prioritize factual correctness   
, reasoning correctness within the final answer, and   
functional correctness. For subjective or open-ended prompts,   
consider helpfulness, relevance, factual soundness, clarity,   
and conciseness. Do not prefer a response merely because it is   
longer or more structured in terms of formatting. If both   
responses are similarly good or similarly flawed, choose Tie.   
Return JSON only, with "winner" set to exactly one of "A", "B", "   
Tie", or "Unjudgeable", and "reason" set to one concise   
sentence.

Table 8: Output-quality scores used to define quality requirements. All values are percentages. The best scores are in bold for each benchmark. ShareGPT reports the relative chat-quality score q<sub>chat</sub> from pairwise LLM judging; RepoBench reports edit similarity (ES) and exact match (EM); LongBench reports ROUGE-L F1 for long-output summarization tasks in the GovReport subset; SWE-Bench Verified reports the resolution rate. Missing entries indicate incompatible model– workload pairs: Reasoning models (GPT-OSS and Qwen3-Thinking variants) are excluded from code-completion tasks.
<table><tr><td>Model</td><td rowspan="2"></td><td colspan="2">ShareGPT RepoBench</td><td colspan="2">LongBench SWE-Bench</td></tr><tr><td></td><td>ES</td><td>EM Summarization</td><td></td><td>Verified</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td></td><td>13.843.5</td><td>8.2</td><td>36.6</td><td>0.20</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td></td><td>17.8 56.0</td><td>24.8</td><td>37.7</td><td>0.20</td></tr><tr><td>GPT-OSS-20B</td><td>73.7</td><td></td><td></td><td>30.9</td><td>2.03</td></tr><tr><td>GPT-OSS-120B</td><td>112.8</td><td>一</td><td>一</td><td>28.8</td><td>4.46</td></tr><tr><td>Qwen3-4B-Instruct</td><td></td><td>50.544.6</td><td>9.2</td><td>30.9</td><td>3.65</td></tr><tr><td>Qwen3-4B-Thinking</td><td>43.3</td><td></td><td></td><td>33.5</td><td>2.84</td></tr><tr><td>Qwen3-8B</td><td></td><td>34.8 46.5</td><td>12.2</td><td>33.5</td><td>2.84</td></tr><tr><td>Qwen3-14B</td><td></td><td>46.2 58.1</td><td>22.3</td><td>33.6</td><td>3.45</td></tr><tr><td>Qwen3-30B-A3B-Instruct</td><td></td><td>82.5 51.1</td><td>11.5</td><td>31.6</td><td>8.52</td></tr><tr><td>Qwen3-30B-A3B-Thinking</td><td>89.0</td><td></td><td></td><td>31.1</td><td>4.87</td></tr><tr><td>Qwen3-32B</td><td></td><td>53.5 58.0</td><td>20.5</td><td>33.3</td><td>6.29</td></tr><tr><td>Qwen3-235B-A22B-Instruct</td><td></td><td>100.0 66.7</td><td>34.7</td><td>32.5</td><td>20.69</td></tr><tr><td>Qwen3-235B-A22B-Thinking</td><td>126.8</td><td>一</td><td>一</td><td>32.4</td><td>15.82</td></tr><tr><td>Gemma-4-E4B</td><td></td><td>56.643.6</td><td>8.39</td><td>32.0</td><td>5.68</td></tr><tr><td>Gemma-4-26B-A4B</td><td></td><td>97.939.4</td><td>5.6</td><td>32.9</td><td>32.45</td></tr><tr><td>Gemma-4-31B</td><td></td><td>98.8 52.2</td><td>18.1</td><td>32.1</td><td>54.77</td></tr></table>

## A.3.5 PROFILING HARNESS

We serve the evaluated models on vLLM 0.23 (Kwon et al., 2023), and use Harbor 0.21 (Harbor Framework Team, 2026) to set up the agent execution and evaluation environment. We use NVML (NVIDIA Corporation, 2025) to collect GPU power and Linux cgroup with psutil to collect CPU and DRAM utilization. For non-agentic workloads, we sweep the request rate to identify the operating point that maximizes throughput while satisfying the latency SLOs for each computing configuration. The workload sender and vLLM server run on the same host and communicate through the localhost loopback interface, so these experiments isolate the serving system from external network effects. In contrast, the agentic workload uses distributed execution setup: agent containers run on separate cloud VMs and access the model-serving endpoint through a proxy server. This setup captures both realistic communication overhead and the energy consumption of the client-side agent execution environment. We fix agentic session concurrency at 10 and allocate each agent container 1.5 logical CPU cores and 12 GB of memory, based on the observed resourceutilization patterns.

## A.3.6 REGIONS

Table 9 lists the deployment locations considered in our regional analysis and their representative mappings to public cloud regions. We select locations to provide broad geographic coverage across Europe, North America, Asia, the Middle East, Africa, South America, and Oceania, while restricting the set to locations that can be mapped to documented AWS, GCP, or Azure regions. For each modeled region, these mappings provide the regional environmental inputs used to parameterize the operational impact intensities. The listed city is additionally used as the geographic proxy for location-specific environmental inputs, including direct water-use effectiveness (WUE) derived through wet-bulb temperature models (Gupta et al., 2024), when those data are available at city or nearby metropolitan scale. These mappings are modeling proxies rather than claims about the exact physical location of individual data centers within each cloud region.

Table 9: Supported modeled deployment locations and representative cloud-region mappings based on the official region documentation from Amazon Web Services (b), Google Cloud (a), and Microsoft. Cities denote modeling proxies for direct WUE.
<table><tr><td>Location</td><td>Provider</td><td>Region code</td></tr><tr><td>Vienna, Austria</td><td>Azure</td><td>austriaeast</td></tr><tr><td>Brussels, Belgium</td><td>GCP</td><td>europe-west1</td></tr><tr><td>Hamina, Finland</td><td>GCP</td><td>europe-north1</td></tr><tr><td>Paris, France</td><td>AWS</td><td>eu-west-3</td></tr><tr><td>Frankfurt, Germany</td><td>GCP</td><td>europe-west3</td></tr><tr><td>Milan, Italy</td><td>AWS</td><td>eu-south-1</td></tr><tr><td>Amsterdam, Netherlands</td><td>Azure</td><td>westeurope</td></tr><tr><td>Oslo, Norway</td><td>Azure</td><td>norwayeast</td></tr><tr><td>Madrid, Spain</td><td>GCP</td><td>europe-southwest1</td></tr><tr><td>Stockholm, Sweden</td><td>AWS</td><td>eu-north-1</td></tr><tr><td>Zurich, Switzerland</td><td>AWS</td><td>eu-central-2</td></tr><tr><td>London, United Kingdom</td><td>AWS</td><td>eu-west-2</td></tr><tr><td>Calgary, AB, Canada</td><td>AWS</td><td>ca-west-1</td></tr><tr><td>Montreal, QC, Canada</td><td>GCP</td><td>northamerica-northeast</td></tr><tr><td>Toronto, ON, Canada</td><td>GCP</td><td>northamerica-northeast</td></tr><tr><td>Ashburn, VA, USA</td><td>AWS</td><td>us-east-1</td></tr><tr><td>Columbus, OH, USA</td><td>GCP</td><td>us-east5</td></tr><tr><td>Dallas, TX, USA</td><td>GCP</td><td>us-south1</td></tr><tr><td>Des Moines, IA, USA</td><td>GCP</td><td>us-central1</td></tr><tr><td>Las Vegas, NV, USA</td><td>GCP</td><td>us-west4</td></tr><tr><td>Los Angeles, CA, USA</td><td>GCP</td><td>us-west2</td></tr><tr><td>Phoenix, AZ, USA</td><td>Azure</td><td>westus3</td></tr><tr><td>Salt Lake City, UT, USA</td><td>GCP</td><td>us-west3</td></tr><tr><td>Seattle, WA, USA</td><td>Azure</td><td>westus2</td></tr><tr><td>Tokyo, Japan</td><td>AWS</td><td>ap-northeast-1</td></tr><tr><td>Seoul, South Korea</td><td>AWS</td><td>ap-northeast-2</td></tr><tr><td>Taipei, Taiwan</td><td>GCP</td><td>asia-east1</td></tr><tr><td>Deİhi, India</td><td>GCP</td><td>asia-south2</td></tr><tr><td>Hyderabad, India</td><td>AWS</td><td>ap-south-2</td></tr><tr><td>Mumbai, India</td><td>AWS</td><td>ap-south-1</td></tr><tr><td>Jakarta, Indonesia</td><td>GCP</td><td>asia-southeast2</td></tr><tr><td>Kuala Lumpur, Malaysia</td><td>AWS</td><td>ap-southeast-5</td></tr><tr><td>Singapore, Singapore</td><td>AWS</td><td>ap-southeast-1</td></tr><tr><td>Bangkok, Thailand</td><td>GCP</td><td>asia-southeast3</td></tr><tr><td>Manama, Bahrain</td><td>AWS</td><td>me-south-1</td></tr><tr><td>Tel Aviv, Israel</td><td>AWS</td><td>il-central-1</td></tr><tr><td>Doha, Qatar</td><td>GCP</td><td>me-central1</td></tr><tr><td>Dammam, Saudi Arabia</td><td>GCP</td><td>me-central2</td></tr><tr><td>Abu Dhabi, UAE</td><td>Azure</td><td>uaecentral</td></tr><tr><td>Dubai, UAE</td><td>AWS</td><td>me-central-1</td></tr><tr><td>Cape Town, South Africa</td><td>AWS</td><td>af-south-1</td></tr><tr><td>Johannesburg, South Africa</td><td>GCP</td><td>africa-south1</td></tr><tr><td>São Paulo, Brazil</td><td>AWS</td><td>sa-east-1</td></tr><tr><td>Santiago, Chile</td><td>GCP</td><td>southamerica-west1</td></tr><tr><td>Melbourne, Australia</td><td>AWS</td><td>ap-southeast-4</td></tr><tr><td>Sydney, Australia</td><td>AWS</td><td>ap-southeast-2</td></tr></table>

Provider abbreviations: AWS = Amazon Web Services; GCP = Google Cloud; Azure = Microsoft Azure.

## A.3.7 DATA SOURCES AND AVAILABILITY

We parameterize the environmental accounting model using a combination of public reports, published datasets, and licensed research data. For operational electricity, we obtain hourly electricitygeneration mixes and carbon intensities from Electricity Maps (2024). We combine the generation mix with technology-specific lifecycle water, ${ \mathrm { S O } } _ { 2 } ,$ and $\mathrm { N O } _ { x }$ intensities reported by Turconi et al. (2013) and Reig et al. (2020) to construct the electricity-related water and ecosystem-impact intensity factors used in our regional and temporal analysis. Additional grid-emission information used in the lifecycle model is derived from authoritative inventories including EPA eGRID (United States Environmental Protection Agency (EPA), 2025) and EDGAR (European Commission Joint Research Centre (JRC) and Netherlands Environmental Assessment Agency, 2024). Midpoint impacts are converted to ecosystem-damage endpoints using the ReCiPe 2016 framework (Huijbregts et al., 2016).

Facility-efficiency parameters are derived from public datacenter disclosures and published models. PUE values are collected from cloud-provider sustainability disclosures (Microsoft Azure; Google; Amazon Web Services, a). Direct datacenter WUE follows the empirical model in the water-sustainability dataset of Gupta et al. (2024). The model is parameterized using wetbulb temperature obtained from Open-Meteo’s Historical Weather API (Zippenfenig, 2023) with the ERA5 reanalysis product (Soci et al., 2024), queried at the coordinates of each modeled cloud-region city or metropolitan area. We use the daily mean 2-m wet-bulb temperature (wet bulb temperature 2m mean) and aggregate the resulting WUE values according to the temporal resolution of the analysis. Regional water scarcity is characterized using the AWARE (Seitfudem et al., 2025) framework described in §A.4.3.

For embodied impacts, we follow the accounting methodology established in ACT (Gupta et al., 2022), ThirstyFLOPs (Jiang et al., 2025a), FABRIC (Shi et al., 2025a), and most directly BIRDS (Shi & Ding, 2026). We use their hardware-manufacturing and impact-allocation methodology, while applying the system boundary defined in §2: hardware manufacturing and system operation are included, whereas transportation and end-of-life are excluded.

Data availability. We do not redistribute the raw LLM-serving measurements or the underlying environmental-intensity datasets with this submission. In particular, ElectricityMaps data are provided under license and may not be redistributed as raw or unmodified data; researchers seeking the hourly electricity-mix and carbon-intensity traces should obtain them directly from ElectricityMaps under the applicable academic-access terms. Other externally sourced environmental data should likewise be obtained from the original providers cited above. The manuscript and appendix report the derived impact quantities, modeling assumptions, temporal aggregation procedures, and experimental settings used in our analysis.

## A.4 SUPPLEMENTARY CHARACTERIZATION RESULTS

This section provides additional characterization results supporting the trends summarized in §5. We first expand the configuration analysis across GPU platforms, tensor-parallel settings, model families, and workloads, including separate treatment of agentic workloads under a per-successfultask functional unit. We then provide finer-grained regional and temporal results together with the underlying environmental-intensity data used to derive them. Finally, we evaluate the sensitivity of our conclusions to including device-manufacturing energy as an embodied component of lifecycle energy, showing that its contribution remains small across the evaluated configurations.

## A.4.1 ADDITIONAL CHARACTERIZATION ACROSS HARDWARE CONFIGURATIONS

Figure 9 provides the detailed hardware and tensor-parallelism results summarized in Figure 3, while fixing the workload and deployment choice. GPU choice substantially affects per-request impact: H100 generally yields the lowest impact across Gemma model sizes, whereas L40 tends to be higher, particularly for the 31B model. The effect of tensor parallelism is non-monotonic. Increasing TP activates more GPUs and raises instantaneous power, but can also improve throughput and reduce the energy amortized per request. For smaller models, the throughput gain is often insufficient to offset the additional power, whereas for the 31B model on A100 and H100, higher TP reduce per-request impact because the throughput improvement dominates.

![](images/45e058b00a666c0215820e92ce9ce48e1dccc8889af349c30c1f8d3d62d031b7.jpg)

![](images/2248fa1b87654410ca28011ae9c5b1f64387ca6e6a66c0b0cd209fd2cb4ae86a.jpg)  
Figure 9: Impact of GPU type and tensor parallelism on per-request energy, carbon, water, and biodiversity for Gemma models.  
Figure 10: ShareGPT impact characterization across GPU types and tensor-parallel settings for Llama models. L40 and A100 support up to TP4; Llama 3.1 70B configurations exceeding their memory capacity are omitted.

Carbon, water, and biodiversity generally follow the same hardware and TP trends because their operational components scale with the underlying IT energy under the fixed deployment choice; embodied components can introduce exceptions to this ordering. Thus, accelerator choice and parallelism can substantially change absolute impact while generally preserving the relative ordering across sustainability dimensions. The corresponding results for Llama, GPT-OSS, and Qwen are shown in Figures 10 to 12.

## A.4.2 ADDITIONAL CHARACTERIZATION ACROSS WORKLOADS

Model scaling across workloads. We first complement the ShareGPT characterization in Figure 2 by varying model family and size within three additional workloads. Figures 13 to 15 show the corresponding model-scale characterization for RepoBench, LongBench, and SWE-bench Verified under the same H100 platform and deployment choice as Figure 2. For the two non-agentic workloads, the qualitative trend is similar to ShareGPT: impact generally increases with model size, while MoE models can remain substantially below similarly sized dense models. Their absolute impact is higher than ShareGPT because the code-completion and long-context workloads process substantially longer sequences.

SWE-bench Verified behaves differently because the functional unit is one successfully completed task rather than one request. Agentic execution requires multiple model invocations and long trajectories, and task success rate enters the denominator of the functional unit. Consequently, a smaller model with a low success rate can incur higher impact per successful task than a larger, more capable model, breaking the otherwise common model-size trend.

Cross-workload variation. The preceding figures hold the workload fixed and vary the model. We next take the complementary view by fixing the model family and hardware and comparing workloads directly. Figures 16 to 19 show consistent workload-level patterns across Gemma, Qwen, GPT-OSS, and Llama. Among the non-agentic workloads, ShareGPT generally incurs the lowest per-request impact, while RepoBench and especially LongBench are higher because they process longer prompt and generation sequences. The magnitude of this increase depends on the model and its selected TP configuration, but energy, carbon, water, and biodiversity follow closely aligned trends within each matched workload comparison.

![](images/ca97d9587114b397f11b71be9e5b55313583565f32905dc4fae6687122637dd1.jpg)  
Figure 11: ShareGPT absolute impact characterization varying GPU type and tensor parallelism when focusing on GPT-OSS models.

![](images/4cc7c20e842e244275a41fff44519bef0c1323b7d1df1653811fe2cb828def14.jpg)  
Figure 12: ShareGPT absolute impact characterization varying GPU type and tensor parallelism when focusing on Qwen models.

SWE-bench Verified is shown separately under a per-successful-task functional unit and therefore should not be compared numerically with the per-request bars. Its impact is substantially larger across all model families because an agentic task can require many model invocations and long execution, while low task success further increases impact per successful completion by amortizing failed attempts over fewer successes.

## A.4.3 ADDITIONAL CHARACTERIZATION OF SPATIOTEMPORAL IMPACT VARIATIONS

Figure 20 provides the monthly and hourly results summarized in the main text. The ranking of regions changes over time and exhibit multiple crossovers, showing that annual-average preferences need not persist at finer temporal resolutions. The magnitude and timing of these changes differ across carbon, water, and biodiversity because each dimension depends on a different combination of regional environmental factors. Figure 21 provides the complementary daily view for representative months across 2024 and shows that the same temporal variation is also visible at an intermediate timescale.

The remaining figures expose the environmental data underlying these impact variations. Figure 22 reports the annual-average PUE values used for facility overhead, while Figure 23 reports the AWARE-2.0 (Seitfudem et al., 2025) water-scarcity characterization factors as the water stress factor (WSF). Figures 24 to 26 show the corresponding temporal behavior of grid carbon intensity (CI), electricity water-intensity factor (EWIF), direct water usage effectiveness (WUE), and grid biodiversity intensity BIF at monthly, daily, and hourly resolutions. Together, these data explain why the environmental dimensions exhibit different regional and temporal variation even when the serving workload and configuration are fixed.

![](images/46e31adb90b8eafc872339a957d485372adaae026e5c16411ae575deaa9ebf35.jpg)  
Figure 13: Per-request energy, carbon, water, and biodiversity impact across model families and sizes for RepoBench code completion.

![](images/d31df0ad21942e52a72be60e6b1a31626cfb5d502ba5f4ea0c27fc88ff3ba3b5.jpg)  
Figure 14: Per-request energy, carbon, water, and biodiversity impact across model families and sizes for LongBench long-output summarization.

## A.4.4 SENSITIVITY TO EMBODIED MANUFACTURING ENERGY

Our primary analysis defines the energy dimension as operational energy, consistent with Equation (12). As a sensitivity analysis, we additionally account for energy consumed during device manufacturing and amortize it using the same allocation procedure as the other embodied impacts. Table 10 shows the ranges of embodied energy contribution across the characterization and configuration-disagreement figures considered in the main text and appendix. Overall, embodied energy contribute to 1.35-6.93% of lifecycle energy and do not change our conclusions on characterization results and cross-dimensional disagreements.

## A.5 ADDITIONAL CROSS-DIMENSIONAL CONFIGURATION-RANKING DISAGREEMENT

This section distinguishes three levels of lifecycle disagreement on configuration choice. Optimal configuration disagreement means that different dimensions select different minimum-impact configurations from the same feasible configuration set satisfying the specified service requirements. General configuration-ranking disagreement means that dimensions order a broader candidate set differently, even when no common quality constraint is imposed. Pairwise ordering flips refer to controlled comparisons in which two fixed configurations exchange order across dimensions. Proposition 3 is a pairwise result: it predicts when one configuration overtakes another under a given dimension. Optimal-configuration disagreement arises only when such a pairwise crossover changes the minimum-impact feasible configuration.

![](images/01b6a3fc938f81ec07c2d03634c9c3db59ac2adffae932f71f4a19961003a029.jpg)  
Figure 15: Per-successful-task energy, carbon, water, and biodiversity impact across model families and sizes for SWE-bench Verified.

![](images/e9deeb6cfca1af5afefd1239d356f82e0c00227fd8f6c0bc31be11607a65450a.jpg)

![](images/39c1e25ba496c6988043cfda424cab204b8aa6cc48afc9d0b91b8dcf6fc1b390.jpg)

![](images/e2902427d32d4818f13e6f32b4b9f295e6ab5d6591e8fb087e3567176b3af961.jpg)

![](images/dd37b4dbdb08539e4ee41edb9e0b9f30b12b10091c9a1634432cdc76dcda1e04.jpg)  
Figure 16: Impact variation across workloads for Gemma models on H100. Each model–workload pair uses its minimum-IT-energy TP setting. Non-agentic workloads are reported per request, while SWE-bench Verified is reported per successfully completed task.

Optimal configuration disagreement. In addition to the RepoBench case in Figure 5, Figure 27 shows a second quality-constrained optimal-configuration disagreement for ShareGPT. Under the $q _ { \mathrm { c h a t } } \geq 8 6 \%$ requirement in Norway, energy, water, and biodiversity select Qwen3 30B-Instruct on two H100s with TP2, whereas lifecycle carbon selects Gemma 4 26B on two H100s with TP2. As in the main-body example, the carbon-minimizing configuration under lifecycle accounting is not the lowest-energy feasible configuration because its embodied-carbon advantage is large enough to offset its additional operational-carbon penalty, thereby changing the minimum-impact feasible configuration.

General configuration-ranking disagreement. Optimal-configuration disagreement is the strongest decision-level outcome, but lifecycle effects can also alter the broader ordering of candidate models even without a specified quality threshold. Figure 28 illustrates this unconstrained ranking effect for ShareGPT across Norway, Switzerland, and France. Lifecycle carbon produces the largest ranking changes, particularly among the middle-ranked Qwen and Llama models, and these changes vary by region. Water largely preserves the configuration ranking induced by energy, while biodiversity introduces only minor additional swaps. Nevertheless, Gemma E4B remains the minimum-impact model across all four dimensions and all three regions, illustrating that substantial ranking disagreement does not necessarily produce optimal-configuration disagreement.

Pairwise ordering flips under controlled hardware and parallelism choices. Disagreement can also arise without changing the model. In Figure 29 ⃝1 , we fix Qwen3-4B-Thinking and TP1 and compare GPU platforms. H100 is preferred for energy, water, and biodiversity, whereas A100 is preferred for lifecycle carbon. In Figure 29 ⃝2 , we fix Gemma 4 26B on H100 and compare only tensor-parallel settings. TP1 is preferred for energy and water, while TP4 is preferred for lifecycle carbon and biodiversity. These are controlled pairwise ordering flips: they show that different dimensions can prefer different hardware or TP choices for the same model, but they do not by themselves imply that either configuration is the global optimum for the full candidate set.

![](images/56a0fab87dee84a5accb8e69cb706c43ba6082b3b845ab0eb38c4c6126160072.jpg)

Figure 17: Impact variation across workloads for Qwen models on H100. Each model–workload pair uses its most energy-efficient tensor-parallel configuration. Non-agentic workloads are reported per request, while SWE-bench Verified is reported per successfully completed task. Crosses mark excluded configurations.  
![](images/bf02aa2114e2aece9fc312e70301f8f6ac11d72e0dd35b73723949df9151471a.jpg)

![](images/c61215cac0fe9506b9c4287433bcacf8270cbb0e226a8d07a80a040536c2741a.jpg)

![](images/b4221c894b34879718af5862ca31688e50a905c8a0d4ca44b59d16aa787c1d68.jpg)

![](images/126da347c94a1cd355821a4d4aa1178fb57e0ecc747ff39566f879bcf3fa4880.jpg)  
Figure 18: Impact variation across workloads for GPT-OSS models on H100. Each model–workload pair uses its most energy-efficient tensor-parallel configuration. Non-agentic workloads are reported per request, while SWE-bench Verified is reported per successfully completed task.

## A.5.1 NUMERICAL VALIDATION OF THE PAIRWISE CROSSOVER BOUNDARY

Across Figures 5, 27 and 29, the common analytical object is the pairwise crossover condition in Equation (6). For a lower-energy configuration $x _ { 1 }$ and a higher-energy configuration $x _ { 2 } .$ , we define the boundary multiple

$$
\begin{array} { r } { \Delta E _ { \mathrm { I T } } = E _ { \mathrm { I T } } ( x _ { 2 } , w ) - E _ { \mathrm { I T } } ( x _ { 1 } , w ) , \quad \Delta I _ { m , \mathrm { e m b } } = I _ { m , \mathrm { e m b } } ( x _ { 1 } , w ) - I _ { m , \mathrm { e m b } } ( x _ { 2 } , w ) , } \\ { \rho _ { m } = \frac { \Delta I _ { m , \mathrm { e m b } } / \Delta E _ { \mathrm { I T } } } { \alpha _ { m } ( r , t ) } , \qquad \quad } \end{array}\tag{29}
$$

A pairwise ordering flip under dimension m occurs when $\rho _ { m } \ > \ 1$ , i.e., when the embodiedimpact advantage of $x _ { 2 }$ exceeds its operational-impact disadvantage under deployment choice $( r , t )$ An optimal-configuration disagreement is a stronger outcome in which such a pairwise crossover changes the minimum-impact configuration among all quality- and SLO-feasible candidates. Table 11 evaluates this condition for representative optimal-configuration disagreement cases and controlled pairwise ordering flips.

The measured preferences agree with the analytical boundary. For the two quality-constrained optimal-choice cases (Figures 5 and $2 7 )$ , lifecycle carbon exceeds the crossover boundary by 1.78× and 1.83×, respectively. The controlled comparisons reproduce the dimension-specific disagreements in Figure 29: for ⃝1 , only carbon crosses the boundary (2.93×), whereas water and biodiversity remain well below it; for $\dot { \textcircled { 2 } } ,$ , carbon (5.61×) and biodiversity (2.43×) cross, while water does not. The critical-lifetime values provide an equivalent interpretation under our six-year amortization assumption: the corresponding pairwise preference persists while the assumed hardware lifetime remains below the listed threshold.

![](images/d8209134fdb6df41fa5e363ff9d5539d43c627f17cfc085cc3159337b6c8066e.jpg)

![](images/f14724ebb0077dd463d8458a3ea77ed06904d9910c6c6a48386907c7b6327af8.jpg)

![](images/992c1efc47674a4582463106aa4e588e3daa82c06e5f07af5aaa97d6f986406a.jpg)

![](images/487a7823a0105363a8ff21359af2bbbca2ec54acdab02856f3d6ea068fcdccd5.jpg)  
Figure 19: Impact variation across workloads for Llama models on H100. Each model–workload pair uses its most energy-efficient tensor-parallel configuration. Non-agentic workloads are reported per request, while SWE-bench Verified is reported per successfully completed task.

![](images/9f0084012f879e8acf41b2229e3d68a3bce895159ae0eebc23be04f33d744dac.jpg)  
Figure 20: Temporal variation in per-request carbon, water, and biodiversity impact across selected deployment regions under the same serving setting as Figure 4. Left: monthly averages in 2024; right: hourly variation during June 24–26, 2024. Carbon, water, and biodiversity are reported in g CO e/request, mL world-eq/request, and $1 0 ^ { - 1 3 }$ species·yr/request, respectively.

More generally, these results show why embodied share alone does not determine a decision change: the embodied difference must have the appropriate direction and be sufficiently large relative to both the operational-energy gap and the regional operational intensity. Intuitively, disagreement is easiest to trigger when the deployment choice has a low operational intensity for the dimension under consideration—for example, when carbon intensity is very low for the carbon dimension—because the operational penalty of a higher-energy configuration becomes small enough for embodied-impact differences to overturn the ranking.

## A.6 SUPPLEMENTARY OPTIMIZATION DETAILS

This section provides the additional methodology and sensitivity analysis for the multidimensional routing optimization in the main text. We first describe how the mixed code and conversation workload is constructed from the Azure LLM Inference Dataset and paired with hourly environmental traces, followed by the synthetic regional-capacity model used to represent heterogeneous accelerator availability. We then give the full routing formulation, including the single-dimension optima and the minimax-regret objective used by PRISM, and report implementation, baseline, solver, and runtime details. Finally, we evaluate sensitivity to workload composition, deployment-region scope, and regional-capacity realizations to test observed reduction in maximum normalized regret persists beyond the main experimental setting.

![](images/3ce8abd88108e3fe37d9a25bb3056f08bb59c177519a92a2bea05e46d3b6e655.jpg)  
Figure 21: Daily variation in per-request carbon, water, and biodiversity impact across selected deployment regions in March, June, September, and December 2024 under the same serving setting as Figure 4.

![](images/f3f87c517f3754da64e7a42e32897051f0cf58897322b5d72eda444ce1a4eebf.jpg)  
Figure 22: Annual-average PUE values for the modeled regions in Table 9, based on public disclosures from Amazon Web Services (a), Google, and Microsoft Azure.

## A.6.1 TRACE AND WORKLOAD CONSTRUCTION

We construct the routing workload from the Azure LLM Inference Dataset 2024. The code stream uses requests from May 10, 2024, and the conversation stream uses requests from May 12, 2024. We retain request arrival time and input/output token lengths, map each request to the corresponding measured H100 serving profile, and align the two streams by relative hour. The two streams are scaled to contribute equal offered GPU demand.

The optimization horizon contains 24 one-hour intervals. Requests with the same hour, traffic stream, serving profile, and token-length bins are aggregated into a weighted request group. Each group records the number of equivalent requests and may be divided across regions, corresponding to routing interchangeable requests in different proportions. This produces 1,628 request groups representing approximately 25.2 million request equivalents. Each traffic stream contributes 15.4 million GPU-seconds of offered demand.

![](images/af86249f2fef7067df3208cb10270032ddb1323732847305a28b3bf66c518ab6.jpg)  
Figure 23: AWARE-2.0 water-scarcity factors for the modeled regions (Seitfudem et al., 2025).

![](images/e5c3dc70d02b90bdd314c9ec286bd5624eaf15e28a4a9c08f86a476e9d23e1c8.jpg)

CI  
![](images/1459cfdca6016caa403034e3fec9f9831bf062173c7cae5c72e6e981c88814db.jpg)

![](images/fb2e20cd470f743221e89d642dea08fd88a6116d532bae8711a85b43039aae1e.jpg)

![](images/0ae20cf43a30ecb9e4689f36ca0653d562232ecd553aa8f2f3b6f1debe1a8fd5.jpg)

![](images/c43f1db0b89a57488c5df8b53eba8e5d7f61ed5a0c6c87352d89249383b31c37.jpg)

![](images/61af4170ee1283f8df27a1b12b461cb08bb95f5bbbeb2afad971e6b647f59d7e.jpg)  
Figure 24: Monthly CI, EWIF, WUE, and BIF for selected modeled regions in 2024, together with the region-specific WSF used for water-scarcity adjustment. CI, EWIF, and BIF follow the observed electricity-grid mix, WUE reflects cooling conditions, while WSF captures regional water scarcity.

We pair the demand trace with hourly environmental factors from June 24, 2024. The trace supplies the within-day demand pattern, while the environmental dataset supplies temporal variation in regional impact intensities. For workload sensitivity, we additionally construct a code-only instance by retaining only the code stream and applying the same preprocessing and capacity-generation procedure.

## A.6.2 REGIONAL CAPACITY MODEL

We evaluate twelve deployment regions: Abu Dhabi, Melbourne, Sao Paulo, Toronto, Frankfurt,˜ Paris, Tokyo, Kuala Lumpur, Oslo, Los Angeles, Northern Virginia, and Cape Town, as the regions studied in Figure 20. Because regional accelerator inventories are not publicly available, we synthesize heterogeneous capacity.

For each capacity seed, we draw a lognormal weight for each region with log-space standard deviation $\sigma = 0 . 5$ and normalize the weights to obtain fixed regional capacity shares. In each hour, total capacity of all regions is provisioned at 1.25× the offered GPU demand. Regional allocations are converted to GPU counts, rounded upward to multiples of 32 H100 GPUs (one rack of 4 DGX H100 systems following NVIDIA’s H100 SuperPOD design (NVIDIA, 2023)), and converted back to GPU-seconds. We evaluate seeds 0–19, with all routing policies using identical capacities for a given seed.

## A.6.3 ROUTING OPTIMIZATION

Each workload uses its feasible configuration with the lowest measured IT energy. Let $y _ { i r }$ denote the fraction of request group i routed to region r. Each group must be fully assigned:

$$
\sum _ { r } y _ { i r } = 1 , \qquad 0 \leq y _ { i r } \leq 1 .\tag{30}
$$

![](images/a898742c23695ed533559ea9c3484a8e71c55903ab630e8f4f80ea8f9542aaf5.jpg)  
Figure 25: Daily CI, EWIF, WUE, and BIF for selected modeled regions in March, June, September, and December 2024.

![](images/2aa2b1629e4616120e7ea4a624d67e6782cc63908fb48bc58e1c885eefa85289.jpg)  
Figure 26: Hourly CI, EWIF, WUE, and BIF for selected modeled regions during June 24–26, 2024.

Let $q _ { i }$ denote the number of equivalent requests in group i, $g _ { i }$ its GPU-seconds per request, and $t _ { i }$ its arrival hour. Regional capacity is constrained by

$$
\sum _ { i : t _ { i } = t } q _ { i } g _ { i } y _ { i r } \leq K _ { r t } , \qquad \forall r , t ,\tag{31}
$$

where $K _ { r t }$ is the available GPU-seconds in region r during hour t. Requests are routed within their arrival hour; the experiment does not defer or drop demand. Thus, the time component of each deployment choice is fixed to the request group’s arrival hour $t _ { i } ,$ and the optimization varies only the deployment region.

Table 10: Ranges of embodied energy contribution to lifecycle energy impact across the evaluated settings in the characterization and disagreement analyses.
<table><tr><td>Figure</td><td>Low (%)</td><td>High (%)</td><td>Figure</td><td>Low (%)</td><td>High (%)</td></tr><tr><td>Figure 2</td><td>1.35</td><td>3.72</td><td>Figure 14</td><td>2.18</td><td>3.90</td></tr><tr><td>Figure 4</td><td>2.23</td><td>2.64</td><td>Figure 16</td><td>1.54</td><td>5.30</td></tr><tr><td>Figure 5</td><td>1.83</td><td>2.51</td><td>Figure 17</td><td>1.36</td><td>6.44</td></tr><tr><td>Figure 9</td><td>1.54</td><td>3.83</td><td>Figure 18</td><td>1.43</td><td>4.12</td></tr><tr><td>Figure 10</td><td>1.48</td><td>2.09</td><td>Figure 19</td><td>1.35</td><td>5.89</td></tr><tr><td>Figure 11</td><td>1.43</td><td>2.23</td><td>Figure 27</td><td>3.83</td><td>6.93</td></tr><tr><td>Figure 12</td><td>1.36</td><td>3.72</td><td>Figure 28</td><td>1.43</td><td>4.30</td></tr><tr><td>Figure 13</td><td>2.84</td><td>6.44</td><td>Figure 29</td><td>1.92</td><td>4.30</td></tr></table>

![](images/9824fd81830aa6c3e768846c28c4fe4cf7bc387d15468338bdf726e6d8819235.jpg)  
Figure 27: ShareGPT optimal configuration disagreement between dimensions

For each impact dimension $m \in \{ E , C , W , B \}$ , we first solve the corresponding single-dimension routing problem to obtain the feasible optimum $I _ { m } ^ { * }$ . Let $I _ { m } ( y )$ denote the aggregate impact induced by routing assignment y. PRISM then solves

$$
\begin{array} { r l } { \underset { y , z } { \operatorname* { m i n } } } & { z } \\ { \mathrm { s . t . } } & { I _ { m } ( y ) \leq ( 1 + z ) I _ { m } ^ { * } , \forall m \in \{ E , C , W , B \} , } \end{array}\tag{32}
$$

together with the assignment and capacity constraints above. Since aggregated request groups are divisible, the main experiment is a linear program (LP). Representing individual requests with indivisible assignments gives the corresponding mixed-integer formulation (MILP).

All routing policies return assignments to the same impact evaluator, which computes energy, carbon, water, and biodiversity using the accounting model in $\ S 2$

## A.6.4 IMPLEMENTATION DETAILS

Experimental Configurations. Unless otherwise stated, experiments use a 24-hour horizon with one-hour routing intervals and the twelve-region deployment set. Code and conversation traffic contribute equal offered GPU demand. Total regional capacity in each hour is provisioned at 1.25× offered demand, with regional shares drawn from a lognormal distribution with $\sigma = 0 . 5$ and held fixed over time. Capacity is rounded upward to 32-H100 units, and results are reported across capacity seeds 0–19. Each workload uses its minimum-IT-energy feasible computing configuration $x _ { E } ^ { * } ( w )$ . All demand must be served in its arrival hour; requests are neither dropped nor deferred.

Baselines. The single-dimension baselines minimize energy, carbon, water, or biodiversity under the same assignment and capacity constraints. The load-balancing baseline uses regional capacity without environmental information. Our offline WaterWise (Jiang et al., 2025b) adaptation preserves its joint carbon–water objective. Carbon and water are normalized across eligible regions and weighted equally. We evaluate its assignments using the same carbon and stress-adjusted water accounting as the other policies. Since the routing experiment contains neither request deferral nor request-origin information, only the spatial routing component is used.

Solver and Runtime. We implement the optimization in Python 3.12 using SciPy 1.17.1 (Virtanen et al., 2020) with the HiGHS linear-optimization backend (Huangfu & Hall, 2018). The twelve-region instance contains 1,628 request-group assignment constraints, 288 region-hour capacity constraints, and 19,536 routing variables. PRISM adds one maximum-regret variable and four regret constraints. On one AMD EPYC 7443 CPU, the complete set of optimization policies for the twelve-region instance requires less than 4 s end-to-end, including constraint construction and impact evaluation. For PRISM alone, the optimization solve takes below 0.3 s in our measurements.

![](images/d049bf14f482575c21c1c5db81344939a41718b06f9bb8dfb9067535e78c7b90.jpg)  
Figure 28: General model-ranking disagreement for ShareGPT across Norway, Switzerland, and France without imposing a quality constraint.

![](images/9ba5e61ae385890b068f5b6ebf01a27f86aea556f65e08a66db0fee45930af47.jpg)  
Figure 29: Lifecycle impact disagreement under controlled system choices. ⃝1 Qwen3-4B-Thinking on ShareGPT in Norway, comparing H100, A100, and L40 at TP1. ⃝2 Gemma 4 26B on ShareGPT in France, comparing H100 TP1 and TP4. Stars mark the preferred choice within each comparison.

## A.6.5 ADDITIONAL SENSITIVITY RESULTS

We additionally evaluate on the code-only instance defined in §A.6.1. PRISM obtains 37.0% worst-case regret, compared with 74.5% for WaterWise, 66.9% for water-only routing, 384.4% for biodiversity-only routing, 466.0% for load balancing, 708.1% for energy-only routing, and 838.6% for carbon-only routing. The result is consistent with the mixed-workload experiment: balancing all four dimensions substantially reduces the maximum normalized regret.

We also examine the influence of regional scope and capacity. We repeat the experiment across 20 capacity seeds under three regional scopes. The original six-region set contains Frankfurt, Los Angeles, Tokyo, Kuala Lumpur, Abu Dhabi, and Melbourne, matching the representative locations in Figure 4. The all-twelve-region set additionally includes Paris, Toronto, Northern Virginia, Oslo, Cape Town, and Sao Paulo and is used for the main optimization result. The ˜ concentrated six-region set contains Frankfurt, Paris, Los Angeles, Northern Virginia, Toronto, and Tokyo. It provides a less geographically dispersed deployment space concentrated in Europe, North America, and East Asia, allowing us to test whether the result depends on the broader geographic and environmenta diversity of the twelve-region set.

Across all three region sets and 60 capacity instances, PRISM achieves lower worst-case regret than WaterWise. The benefit is largest for the twelve-region design space, where greater environmental heterogeneity creates larger trade-offs among the four impact dimensions.

Table 11: Numerical validation of the lifecycle crossover boundary. Bold $\rho _ { m } \mathrm { ~ > ~ } 1$ indicates a predicted lifecycle ranking reversal.
<table><tr><td>Workload / configuration pair  $( x _ { 1 }  x _ { 2 } )$ </td><td>Dimension</td><td>ΔEIT (J/req.)</td><td> $\Delta I _ { m , \mathrm { e m b } } /$   $\pmb { \Delta E _ { \mathrm { I T } } }$ </td><td> $\alpha _ { m } ( r , t )$ </td><td> $\rho _ { m }$ </td><td>Critical lifetime (yr)</td></tr><tr><td>RepoBench (Figure 5) Qwen3 4B-I → Qwen3 30B-I</td><td>Carbon</td><td>9.422</td><td>5.342</td><td>2.924</td><td>1.827</td><td>10.96</td></tr><tr><td>ShareGPT (Figure 27) Qwen3 30B-I → Gemma 4 26B</td><td>Carbon</td><td>0.341</td><td>5.201</td><td>2.924</td><td>1.778</td><td>10.67</td></tr><tr><td rowspan="3">Figure 29 ① H100 TP1 → A100 TP1</td><td>Carbon</td><td>4.482</td><td>8.566</td><td>2.924</td><td>2.929</td><td>17.58</td></tr><tr><td>Water</td><td>4.482</td><td>0.106</td><td>7.016</td><td>0.015</td><td>0.09</td></tr><tr><td>Biodiversity</td><td>4.482</td><td>0.288</td><td>1.047</td><td>0.275</td><td>1.65</td></tr><tr><td rowspan="3">Figure 29 ② H100 TP1 → H100 TP4</td><td>Carbon</td><td>1.248</td><td>62.466</td><td>11.143</td><td>5.606</td><td>33.64</td></tr><tr><td>Water</td><td>1.248</td><td>0.622</td><td>4.904</td><td>0.127</td><td>0.76</td></tr><tr><td>Biodiversity</td><td>1.248</td><td>2.719</td><td>1.120</td><td>2.429</td><td>14.57</td></tr></table>

Units for $\Delta I _ { m , \mathrm { e m b } } / \Delta E _ { \mathrm { I T } }$ and $\alpha _ { m } .$ Carbon: $1 0 ^ { - 6 } \ \mathrm { g } \mathrm { C O } _ { 2 } \mathrm { e } / \mathrm { J } ;$ Water: $1 0 ^ { - 6 } \mathrm { ~ L ~ }$ world-eq/J; Biodiversity: $1 0 ^ { - 1 6 }$ species·yr/J. Critical lifetime is the hardware amortization lifetime at $\rho _ { m } = 1$ with measured throughput fixed.

Table 12: Sensitivity of worst-case regret to geographic scope and capacity seed. Values report median [range] across 20 seeds.
<table><tr><td>Region set</td><td>WaterWise</td><td>PRISM</td><td>Median reduction</td></tr><tr><td>Original 6</td><td>43.5% [28.8, 111.2]</td><td>37.2% [25.5, 53.5]</td><td>19.1%</td></tr><tr><td>All 12</td><td>157.8% [55.1, 212.3]</td><td>75.1% [41.4, 104.0]</td><td>50.2%</td></tr><tr><td>Concentrated 6</td><td>101.7% [57.7, 208.5]</td><td>68.1% [48.5, 107.4]</td><td>37.9%</td></tr></table>

## A.7 EXTENSION OF OPTIMIZATION

This section extends the main optimization study beyond its offline, fixed-computing-configuration setting in two directions. First, we evaluate PRISM under rolling-horizon regional routing, where decisions are repeatedly re-optimized using limited and potentially noisy forecasts of future environmental conditions. Second, we consider an agent-heavy workload and jointly optimize computing configuration and deployment region while introducing agent completion time as an additional objective alongside energy, carbon, water, and biodiversity. These extensions test whether the multidimensional optimization remains effective under sequential decision-making and whether its formu lation can accommodate configuration choice and performance trade-offs beyond regional routing alone.

## A.7.1 ROLLING-HORIZON REGIONAL ROUTING

We extend PRISM to rolling-horizon routing using the same 24-hour workload, twelve regions, fixed computing configurations, and seed-0 capacities as the offline experiment. At the beginning of each hour, the scheduler observes the current state and optimizes over a forecast horizon H ∈ {1, 6, 12, 24} hours. It executes only the current-hour allocation and re-optimizes at the next hour; requests remain in their arrival hour.

At each optimization step, projected regret combines impacts already realized with forecast impacts over the remaining window and compares them against the corresponding cumulative singledimension optima. We evaluate both oracle forecasts and noisy environmental forecasts. For the latter, future CI, EWIF, WUE, and BIF values are independently perturbed by mean-one lognormal noise with coefficient of variation 0.20; the current hour is observed exactly. Demand and capacity are assumed known. Full-day offline PRISM provides the reference optimum, and we use the same WaterWise-spatial baseline as in §A.6.4.

As shown in Table 13, even a one-hour oracle horizon remains within 0.70% of the offline optimum in maximum regret. The gap falls to 0.088% at six hours and 0.012% at twelve hours, while the 24-hour horizon reproduces the offline solution to numerical precision. Environmental forecast error has little effect: for horizons containing unobserved future hours, 20% factor noise leaves the gap below 0.15%. In comparison, WaterWise-spatial incurs 188.2% maximum regret. Thus, the multidimensional trade-off obtained offline is largely preserved under sequential routing and remains stable to moderate environmental forecast error.

Table 13: Rolling-horizon PRISM under oracle and noisy environmental forecasts. Gap is the relative increase in maximum regret over full-day offline PRISM.
<table><tr><td></td><td colspan="2">Oracle forecast</td><td colspan="2">20% forecast error</td></tr><tr><td>Horizon</td><td>Max. regret</td><td>Gap</td><td>Max. regret</td><td>Gap</td></tr><tr><td>1 h</td><td>88.014%</td><td>0.697%</td><td>88.014%</td><td>0.697%</td></tr><tr><td>6 h</td><td>87.482%</td><td>0.088%</td><td>87.532%</td><td>0.146%</td></tr><tr><td>12 h</td><td>87.415%</td><td>0.012%</td><td>87.517%</td><td>0.129%</td></tr><tr><td>24 h</td><td>87.405%</td><td>&lt; 0.001%</td><td>87.525%</td><td>0.138%</td></tr><tr><td>Offline PRISM</td><td>87.400%</td><td></td><td></td><td>一</td></tr><tr><td>WaterWise-spatial</td><td>188.240%</td><td>115.36%</td><td></td><td>1</td></tr></table>

Across the full 24-hour replay, rolling-horizon optimization requires 0.60–3.21 s of solver time and 12.2–19.5 s end-to-end across the evaluated horizons on the AMD EPYC 7443 system described in §A.6.4.

## A.7.2 JOINT OPTIMIZATION WITH AGENTIC WORKLOAD

We extend the routing experiment to an agent-heavy workload and jointly optimize computing configuration and deployment region. To represent a 2026-style serving mix in which repeated agentic interactions dominate compute demand, the 24-hour workload consists of 60% agentic coding, 30% conversation, 5% long-output summarization, and 5% code completion by offered GPU-seconds. Conversation follows the Azure conversation trace with ShareGPT profiles, summarization uses the LongBench long-output workload, and code completion uses RepoBench. Agentic arrivals follow the hourly shape of the Azure code trace and use SWE-Bench Verified. Defining the mix by GPU demand rather than request count avoids treating a short inference request and a multi-step agentic task as equivalent units of load. We use the same twelve regions, hourly environmental factors, and capacity model as the offline routing experiment.

For agentic coding, the functional unit remains a successfully completed SWE-Bench Verified task. Each arrival can be assigned to any of 16 measured computing configurations, without assuming that the scheduler knows which model will solve an individual task. For configuration x, expected IT energy and GPU demand per successful task are obtained by dividing the full-suite benchmarking totals by its number of successful tasks, while $\tau _ { x }$ is the mean completion time among successful tasks. The other workloads retain their fixed measured configurations.

Completion time is introduced as an additional objective because a hard timeout does not distinguish between otherwise feasible agent executions that finish substantially earlier or later. Let $y _ { i r } ^ { x }$ be the fraction of request group i assigned to configuration x and region r, and let $\omega _ { i }$ denote its weight. Each request group is fully assigned across configuration–region pairs:

$$
\sum _ { x \in \mathcal { X } } \sum _ { r \in \mathcal { R } } y _ { i r } ^ { ( x ) } = 1 , \qquad y _ { i r } ^ { ( x ) } \geq 0 , \qquad \forall i .\tag{33}
$$

For the set of agentic groups A, we define mean agentic completion-time objective as

$$
L ( s ) = \frac { 1 } { \Omega _ { \cal A } } \sum _ { i \in \cal A } \sum _ { x , r } \omega _ { i } y _ { i r } ^ { ( x ) } \tau _ { x } , \Omega _ { \cal A } = \sum _ { i \in \cal A } \omega _ { i } .\tag{34}
$$

Let $L ^ { * }$ be the minimum feasible completion time under the same assignment and capacity constraints. Completion-time regret is

$$
R _ { L } ( s ) = \frac { L ( s ) - L ^ { * } } { L ^ { * } } .\tag{35}
$$

![](images/6f3a78db6222392eb98feb7e57cd2ce48a71f31c6974b6ace27cf856d23e28a7.jpg)

(b) Selected configurations  
![](images/7f47b5eaa161b47717621fb5f52be27a7ba508ef9b3fb1a4ddfdd6ff35ae5ba8.jpg)  
Figure 30: Joint computing-configuration and regional-routing optimization under the agent-heavy workload. (a) Maximum environmental regret versus agentic completion-time regret. (b) Configuration shares selected for the agentic workload.

For each environmental dimension $m \in \{ E , C , W , B \}$ , we similarly obtain its independent optimum $I _ { m } ^ { * }$ and define

$$
R _ { m } ( s ) = \frac { I _ { m } ( s ) - I _ { m } ^ { * } } { I _ { m } ^ { * } } .\tag{36}
$$

Environmental PRISM minimizes max $\{ R _ { E } , R _ { C } , R _ { W } , R _ { B } \}$ , whereas the five-objective formulation jointly minimizes

$$
\operatorname* { m i n } _ { s } \operatorname* { m a x } \{ R _ { E } ( s ) , R _ { C } ( s ) , R _ { W } ( s ) , R _ { B } ( s ) , R _ { L } ( s ) \} .\tag{37}
$$

We compare these two formulations with energy-only and completion-time-only optimization and with two WaterWise adaptations. WaterWise-spatial first fixes each workload to its minimum expected-energy configuration and optimizes regional carbon and water, while WaterWise-joint allows the same carbon–water objective to choose both configuration and region.

Figure 30 shows that adding completion time changes both the trade-off and the selected configuration. Environmental-only PRISM achieves 53.1% maximum environmental regret but incurs 140.4% completion-time regret. Five-objective PRISM instead limits both to 77.3%. Carbon, stress-adjusted water, and completion-time regrets are binding at the minimax solution, while energy and biodiversity regrets remain lower at 18.0% and 42.1%, respectively. By comparison, energy-only and completion-time-only optimization incur maximum environmental regrets of 501.4% and 790.8%, while WaterWise-spatial and WaterWise-joint obtain 107.4% and 107.2% environmental regret with 140.4% completion-time regret.

The performance objective also changes configuration selection. Environmental PRISM and both WaterWise variants assign all agentic work to Gemma 4 26B. Five-objective PRISM instead assigns 38.4% to Gemma 4 26B and 61.6% to the faster Gemma 4 31B, reducing completion time until its regret reaches the environmental minimax boundary. Completion-time-only optimization shifts further toward faster configurations, assigning 54.4% to Gemma 4 31B and 45.5% to Gemma 4 E4B. Thus, once agent completion time is treated as an objective rather than only a feasibility constraint, computing configuration and regional routing should therefore be optimized jointly.

The resulting continuous linear programs contain 1,247 weighted request groups and 19,284 joint configuration–region assignment variables. Using the same SciPy/HiGHS implementation as §A.6.4, individual solves require 0.05–0.16 s in this experiment.