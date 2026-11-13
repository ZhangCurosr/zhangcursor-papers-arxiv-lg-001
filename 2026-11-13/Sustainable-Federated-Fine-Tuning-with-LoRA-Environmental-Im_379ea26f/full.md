# Sustainable Federated Fine-Tuning with LoRA: Environmental Impact and Performance Footprint

Pedro Augusto Londe Ribeiro¹, Rodrigo Moreira¹, Larissa Ferreira Rodrigues Moreira¹,

Joberto S. B. Martins², Tereza C. Carvalho³, Flávio de Oliveira Silva⁴

1Federal University of Viçosa (UFV), Minas Gerais, Brazil

Emails: {pedro.londe.ribeiro, rodrigo, larissa.f.rodrigues} @ufv.br

2University of Salvador (UNIFACS), Salvador, Brazil

Email: joberto@ieee.org

3University of São Paulo (USP), São Paulo, Brazil

Email: terezacarvalho@usp.br

4University of Minho (UMinho), Department of Informatics, ALGORITMI Centre, Braga, Portugal Email: flavio@di.uminho.pt

Abstract—Federated Learning (FL) with Parameter-Efficient Fine-Tuning (PEFT) enables efficient adaptation of Large Language Models (LLMs), yet its energy and carbon costs across hardware remain poorly understood. We evaluated federated Low-Rank Adaptation (LoRA) fine-tuning on NVIDIA A40, RTX 6000 Ada, and A100 GPUs using a 22 full factorial design with 135M/360M models and 10/100 local steps. CodeCarbon tracked energy and estimated CO2-equivalent emissions. Descriptive allocation of variation assigned 84.8–91.2% of energy and emissions variation to local steps within these ranges. On A40 and RTX 6000 Ada, configurations with fewer steps achieved comparable accuracy with roughly 6-8-fold lower energy consumption and higher accuracy per kWh. These findings support smaller local budgets for energy-efficient federated fine-tuning within the evaluated setting.

Keywords—Federated Learning, Large Language Models, Parameter-Efficient Fine-Tuning (PEFT), Energy Consumption, Environmental Sustainability

## I. INTRODUCTION

Training and using Large Language Models (LLMs) is quickly becoming a major source of electricity demand in computing. The International Energy Agency (IEA) predicts that data center energy use will more than double by 2030, reaching approximately 945 TWh, mainly owing to Artificial Intelligence (AI) tasks [1], [2]. Industry estimates from Goldman Sachs suggest a 160% rise in data-center power demand by the end of the decade, possibly accounting for up to 4% of global electricity use [3]. These predictions lead to a clear question: how much energy does each part of the modern LLM process use, and where can we reduce it the most?

Federated Learning (FL), a privacy-preserving learning paradigm, combined with Parameter-Efficient Fine-Tuning (PEFT) techniques like Low-Rank Adaptation (LoRA) provides a practical way to adapt Large Language Models (LLMs) to specific tasks while keeping data decentralized and reducing the number of trainable parameters [4], [5]. Additional strategies, such as probability-level ensembling that shares lightweight outputs instead of model weights, further cut communication costs [6]. However, research on these methods has mainly focused on accuracy, communication efficiency, and privacy guarantees. What is still mostly unmeasured is the energy and carbon cost of federated fine-tuning when it operates on diverse hardware, which is typical in real deployments with different Graphic Processing Unit (GPU) generations, thermal design powers, and memory subsystems across federated clients and cloud platforms [7], [8].

Our paper examines whether local optimization intensity outweighs model size in energy use and $\mathrm { C O } _ { 2 } .$ -equivalent emissions within the tested ranges. We evaluated NVIDIA A40, RTX 6000 Ada, and A100 platforms separately. Our contributions are threefold: (i) we measured energy consumption and estimated CO2-equivalent emissions during federated LLM fine-tuning across three GPU families using CodeCarbon and GPU telemetry; (ii) we used $\textrm { a } 2 ^ { 2 }$ factorial design to allocate observed variation among model size, local optimization steps, and their interaction, with local steps accounting for 84.8- 91.2% of energy and emissions variation within the selected levels; (iii) we related predictive performance to energy cost through accuracy per kWh.

The remainder of this paper is structured as follows. Section II reviews related work on sustainable LLM adaptation and energy accounting. Section III presents the experimental methodology. Section IV reports and discusses the results. Section V concludes with limitations and directions for future work.

## II. RELATED WORK

This section reviews recent work on sustainable LLM adaptation, organized into three thematic groups: federated and edge fine-tuning with PEFT, measurement-driven energy and carbon accounting on GPUs, and systems or benchmarking efforts that optimize or standardize evaluation.

Federated and edge fine-tuning with PEFT. Iftikhar et al. [7] study energy-aware client selection for federated LoRA, reporting accuracy and $\mathrm { C O _ { 2 } e }$ estimates. Wu et al. [9] use truncated Singular Value Decomposition (SVD) with dynamic rank allocation to improve accuracy and communication efficiency in heterogeneous federations. Thuau et al. [10] compare federated LoRA with compact personalized models for violence detection, including energy and calibration analyses. Our $2 ^ { 2 }$ factorial design allocates observed energy and emissions variation to model size, local steps, and their interaction within the tested ranges.

On edge devices, Xu et al. [11] combine perturbed forward passes with LoRA to reduce memory requirements. Woisetschläger et al. [8] evaluate federated T5 fine-tuning at the extreme edge using a real-time energy-efficiency metric. Li et al. [12] combine split learning and LoRA to reduce latency and server energy under privacy constraints, while Zhao et al. [13] jointly optimize communication and computation in unmanned vehicle networks. We focus on software-based energy accounting during simulated federated fine-tuning.

Energy and carbon accounting on GPUs. Krzywaniak et al. [14] introduce dynamic power capping with online performance tracing Dynamic Energy-Performance Optimizer (DEPO) for High-Performance Computing (HPC) workloads. Newkirk et al. [15] use measured power traces and statistical models to relate workload intensity to node-level energy. Gogineni et al. [16] examine single-GPU fine-tuning bottlenecks involving LoRA, checkpointing, and power capping. We compare three GPU families under the same training configurations.

At the system level, Ozkan and Ozkan [17] combine telemetry and grid carbon intensity to schedule inference across heterogeneous fleets. Li et al. [18] use LoRA with a Receptance Weighted Key Value (RWKV) model to predict product carbon footprints. Instead, our study quantifies the energy consumption and CO2-equivalent emissions associated with federated fine-tuning.

Benchmarking and efficiency surveys. Kuang et al. [19] provide FederatedScope LLM, a reproducible framework with standardized PEFT implementations and configurable evaluation. Hanindhito et al. [20] analyze the compute and memory trade-offs of LoRA and Quantized Low-Rank Adaptation (QLoRA) across configurations and GPUs. Kaushik et al. [21] survey sustainability challenges and mitigation strategies, including pruning, quantization, and sparsity. We complement these efforts with an accuracy-per-kWh metric that relates predictive performance to energy cost.

Table I summarizes six methodological and reporting features; denotes a reported feature and O a feature not reported. $\mathrm { \bf { \ddot { P E F T } } } \mathrm { \bf { \vec { \Omega } } }$ includes LoRA, “Direct energy" denotes instrumented measurements, and “Carbon"denotes reported $\mathrm { C O _ { 2 } e }$ estimates. “GPU comparison" indicates evaluation across distinct accelerator families. “Factorial analysis" denotes the separation of main effects and interactions on energy or emissions through a factorial design. Our study combines platform measurements with descriptive allocation of variation.

TABLE I: Comparison with related work.
<table><tr><td>Paper</td><td>Federated</td><td>PEFT</td><td>Direct energy</td><td>Carbon CO₂e</td><td>GPU comparison</td><td>Factorial analysis</td></tr><tr><td>Iftikhar [7]</td><td>●</td><td></td><td>C</td><td>●</td><td>O</td><td>O</td></tr><tr><td>Li [12]</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>Hanindhito [20]</td><td>O</td><td></td><td>O</td><td></td><td></td><td>O</td></tr><tr><td>Wu [9]</td><td></td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>Li [18]</td><td>O</td><td></td><td>O</td><td></td><td>O</td><td>O</td></tr><tr><td>Kuang [19]</td><td></td><td></td><td>O</td><td>O</td><td>C</td><td>O</td></tr><tr><td>Thuau [10]</td><td></td><td></td><td></td><td></td><td>O</td><td>O</td></tr><tr><td>Kaushik [21]</td><td>O</td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>Krzywaniak [14]</td><td>O</td><td>O</td><td></td><td></td><td></td><td>O</td></tr><tr><td>Xu [11]</td><td></td><td></td><td>O</td><td>O</td><td></td><td>O</td></tr><tr><td>Newkirk [15]</td><td>O</td><td>O</td><td></td><td></td><td></td><td>O</td></tr><tr><td>Zhao [13]</td><td></td><td></td><td>O</td><td>O</td><td>O</td><td>O</td></tr><tr><td>Woisetschläger [8]</td><td></td><td></td><td>O</td><td>O</td><td></td><td>O</td></tr><tr><td>Ozkan [17]</td><td></td><td>O</td><td>O</td><td></td><td></td><td>O</td></tr><tr><td>Gogineni [16]</td><td>O</td><td></td><td>O</td><td>O</td><td></td><td>O</td></tr><tr><td>Our approach</td><td></td><td></td><td></td><td></td><td></td><td>●</td></tr></table>

## III. ENERGY-EFFICIENT EVALUATION METHOD

Our method evaluates energy consumption, $\mathrm { C O _ { 2 } }$ -equivalent emissions, and predictive performance during federated LLM fine-tuning. Figure 1 summarizes five phases: testbed provisioning, experimental design, energy monitoring, performance evaluation, and factorial analysis.

## A. Experimental Design

We used a $2 ^ { 2 }$ full factorial design to separate the effects of model size, local optimization intensity, and their interaction [22]. The SmolLM-135M and SmolLM-360M models [23] were evaluated with 10 and 100 local steps per round, yielding four configurations per GPU (Table II). Both models belong to the same family, limiting architectural differences in the comparison.

TABLE II: Experimental factors and levels.
<table><tr><td>Factor</td><td>Low level</td><td>High level</td></tr><tr><td>A, model size</td><td>135M</td><td>360M</td></tr><tr><td> $B ,$  local steps per round</td><td>10</td><td>100</td></tr></table>

Fine-tuning kept the pretrained backbone frozen and learned LoRA updates of the form $\Delta W = B A$ with $B \in$ $\mathbb { R } ^ { d \times r }$ and $A \in \mathbb { R } ^ { \dot { r } \times k }$ [24]. The adapter rank was fixed at $r = 8$ across all configurations, following the common practice for models of this scale [20].

Each experiment comprised 50 federated rounds with 50 simulated clients on one GPU. At every round, five clients (10%) were sampled uniformly using a fixed random seed. The selected clients performed the prescribed local steps with a cosine-decayed learning rate. The server then averaged the corresponding adapter parameters with equal client weights and redistributed the global adapter for the next round. Communication energy was unmeasured; aggregation was included in the monitored simulation.

![](images/4c552385f1415e872599a8f1da1c557a152a226b9719afbcb255b2a036eafb40.jpg)  
Fig. 1: Experimental Methodology Workflow.

## B. Energy Monitoring

CodeCarbon monitored each simulation from initialization through the final federated round, covering model loading, client selection, local training, aggregation, and checkpointing [25]. We recorded duration, energy, estimated $\mathrm { { C O } _ { 2 } \mathrm { { - } } }$ equivalent emissions, and GPU power from the NVIDIA Management Library (NVML). Host contributions to CodeCarbon energy remain unresolved, limiting accelerator-specific comparisons. For power $P ( t )$ in watts at a consistent accounting boundary and time in seconds, energy follows Eq. 1:

$$
E _ { \mathrm { k W h } } = { \frac { 1 } { 3 . 6 \times 1 0 ^ { 6 } } } \int _ { t _ { 0 } } ^ { t _ { f } } P ( t ) d t .\tag{1}
$$

Emissions were estimated as $E _ { \mathrm { k W h } } c _ { \mathrm { g r i d } }$ , where $c _ { \mathrm { g r i d } }$ is the location-based average grid carbon intensity in kg $\mathrm { C \breve { O } _ { 2 } e / k W h }$ Location averages omit intraday variations. Fixed withinplatform grid intensity makes energy and emissions contributions identical by construction. Wall-plug validation and resolved CPU/memory accounting are needed to strengthen these software estimates.

## C. Performance Evaluation

Each round's saved global adapter was evaluated on a heldout HDFS log test set [26]. The model predicted the next token at each position in the tokenized sequences. Top-k accuracy measures the fraction of ground-truth next tokens appearing among the k highest-ranked predictions, for $k \in \{ 1 , \bar { 3 } , 5 , 1 0 \}$

For anomaly detection, a sequence was classified as anomalous when its mean Top-k accuracy fell below a threshold $\tau .$ We selected $\tau$ from 1,000 uniform values in [0, 1] by maximizing $F _ { 1 }$ on the same test set. Reported $F _ { 1 }$ , precision, and recall carry optimistic selection bias; independent assessment requires validation-only threshold selection. $F _ { 1 }$ follows Eq. 2:

$$
F _ { 1 } = 2 \frac { \mathrm { P r e c i s i o n R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } .\tag{2}
$$

To relate the predictive performance to the energy expenditure [21], we also computed according to Eq. 3:

$$
\mathrm { E f f i c i e n c y } _ { k } = \frac { \mathrm { T o p } { \cdot } k \mathrm { \ a c c u r a c y } } { E _ { \mathrm { k W h } } } .\tag{3}
$$

The consolidated comparisons use Top-1 accuracy per kWh. This metric measures energy efficiency, and differences in grid carbon intensity are reflected separately in the emissions estimates.

## D. Allocation of Variation

For each GPU, we allocated observed variation in power, energy, emissions, and duration using the $2 ^ { 2 }$ factorial contrasts. Low and high factor levels were coded as -1 and +1, with interaction codes $x _ { A B , i } = x _ { A , i } x _ { B , i } .$ For treatment responses $y _ { i } ,$ the signed contrasts are $\begin{array} { l } { q _ { j } \ = \ \sum _ { i = 1 } ^ { n } x _ { j , i } y _ { i } } \end{array}$ , representing model size (A), local steps $( B ) ,$ and their interaction (AB). Their sums of squares follow Eq. 4:

$$
S S _ { j } = \frac { q _ { j } ^ { 2 } } { n } , \qquad j \in \{ A , B , A B \} ,\tag{4}
$$

where n is the number of treatment observations. The contribution of each component to the total sum of squares $S S _ { T }$ is according to Eq. 5:

$$
{ \mathrm { I n f l u e n c e } } _ { j } = { \frac { S S _ { j } } { S S _ { T } } } , \qquad j \in \{ A , B , A B \} .\tag{5}
$$

Multiplying these ratios by 100 yielded the percentage contributions reported in the results. One observation per treatment leaves zero residual degrees of freedom; these descriptive contributions provide neither uncertainty estimates nor significance tests.

![](images/4a0feb64178b2952fada56e6c246b05cbcf5f7cb558250df51f7691ec67a4f45.jpg)  
(a) Energy consumption

![](images/1f2379d0d0ef4d6d96419704b9816234f5a6a0bc78ca6f5ceb17310430514501.jpg)  
(b) $\mathrm { C O _ { 2 } }$ -equivalent emissions  
Fig. 2: Energy consumption and $\mathrm { C O } _ { 2 } .$ -equivalent emissions by device and configuration.

![](images/78447efaac5af97af74bd3dd30faf6d451c238deda4d06db313a4f96c784a124.jpg)

![](images/f65836804eb80a7940cc9084fd7b82568669f455d250a8d10d8c37d17c4d0129.jpg)

![](images/2627f520a3efd853c2d4680961f8bc6a48a8ecba556bcc908a73a7af84e63773.jpg)  
Fig. 3: Factor contributions to each recorded response.

## E. Testbed Infrastructure

We evaluated the same training configurations on NVIDIA A40 and RTX 6000 Ada GPUs on FABRIC [27] and NVIDIA A100 80 GB PCIe GPUs on Chameleon Cloud [28]. Comparisons combine accelerator, host, and testbed differences, limiting attribution to GPU architecture alone.

## IV. RESULTS AND DISCUSSION

## A. Energy and local optimization

Increasing local optimization from 10 to 100 steps per round raised energy consumption by approximately 6-8× across the tested devices and model sizes according to Fig. 2. With 10 steps, all configurations remained below 1 kWh. With 100 steps, consumption exceeded 6.9 kWh on A40 and 5.6 kWh on RTX 6000 Ada, while A100 stayed below 1.5 kWh. For the 360M model with 100 steps, CO2-equivalent emissions reached 3.12, 1.16, and 0.53 kg on A40, RTX 6000 Ada, and A100, respectively.

Figure 3 allocates the observed variation. Local steps accounted for 91.2%, 87.8%, and 84.8% of the observed variation in energy and emissions on A40, A100, and RTX 6000 Ada, respectively. Model size contributed less than 10% for energy and emissions, and the interaction accounted for at most 5.3%. This ordering is conditional on the unequal proportional ranges chosen for steps and model size.

Model size nevertheless increased the cost of longer training. On A40, the energy gap between the 135M and 360M models grew from 0.24 kWh at 10 steps to 2.22 kWh at 100 steps; on RTX 6000 Ada, it grew from 0.35 to 2.23 kWh. Interaction contributions of 3.4–5.3% indicate that larger models and longer optimization compound the energy cost. Intermediate step counts and nonlinear responses remain outside this two-level design.

## B. Hardware and training duration

Figure 4(a) reports average GPU power separately from CodeCarbon energy. Reconciling their accounting boundaries requires component-level logs. For this power metric, model size, steps, and their interaction explained 35.0%, 39.6%, and 25.4% of the variance on RTX 6000 Ada, while steps alone explained 99.8% on A40 according to Fig. 3. The RTX nonmonotonic response may reflect run variability; replication is needed to assess it.

The recorded energy costs combine accelerator, host, and duration effects. Training duration helps explain this result according to Fig. 4(b). With 100 steps, A40 and RTX 6000 Ada required 2–3× the wall-clock time of A100, accumulating energy over longer runs. Unresolved host accounting and A100 predictive quality restrict hardware comparisons to these observations. Efficiency rankings require component-level energy and matched quality.

## C. Predictive performance and energy efficiency

A40 and RTX 6000 Ada achieved near-ceiling Top-1 accuracy and high $\mathrm { F _ { 1 } }$ scores across all four configurations (Fig. 5). The $\mathrm { F _ { 1 } }$ scores retain the threshold-selection bias described above. Increasing local optimization to 100 steps provided little additional accuracy despite its higher energy cost. These results favor 10 steps for this HDFS workload.

The accuracy-energy comparison in Fig. 6(a) makes this trade-off clear: on A40 and RTX 6000 Ada, 10-step configurations combined high accuracy with low energy consumption, whereas 100 steps mainly increased the energy cost. Accuracy per kWh likewise favored shorter training as in Fig. 6(b). These observations motivate adaptive local-step budgets for this workload. A100 showed lower accuracy and $\mathrm { F _ { 1 } }$ across all configurations despite its lower recorded energy consumption. Hardware-efficiency comparisons require checking checkpoint, tokenizer, precision, and evaluation consistency, then assessing energy at matched predictive quality.

![](images/ac8fcb43fc1abf9ddbd3770cbfef7418599fd6818a94505c561a40c7294e3354.jpg)  
(a) Average GPU power

![](images/7453acaa0f882c75a41b928c35c03fdbf771d9cee4f22e8e381336aadde99f32.jpg)  
(b) Training duration

Fig. 4: Average GPU power and training duration by device and configuration  
![](images/1592afa79dd18abfdb078f7a740aa356c5dc341cc7ab1d9214834677d0f95fcb.jpg)

(a) Top-k accuracy profiles  
![](images/2d6e974c5f93e486520fdda32d2d618339d236611247d0e9d491ee6fdb577fe2.jpg)  
\_ THRESHOLD F1 PRECISIONRECALL  
(b) F1, precision, and recall by configuration

Fig. 5: Model performance profiles during training and evaluation  
![](images/78502379024d762e6dcab6df0aeb520a0262e39d77e359a81fd836f4ee04189d.jpg)  
135M / 10 steps360M / 10 steps135M / 100 steps360M / 100 steps  
(a) Top-1 accuracy versus energy consumption

![](images/0511d2a194c219707ee2d73cbb72197f3d43540a1509454bcec4c4f279987354.jpg)  
(b) Top-1 accuracy per kWh  
Fig. 6: Accuracy-energy trade-off and energy efficiency by device and configuration.

## V. CONCLUDING REMARKS

In this paper, we evaluated energy consumption and estimated carbon emissions during federated LoRA fine-tuning. Within the tested ranges, local optimization dominated their observed variation. Smaller local budgets retained comparable accuracy on A40 and RTX 6000 Ada. The evaluation covers one model family, dataset, and seed; platform comparisons depend on accounting boundaries and predictive quality. Future work should examine independent threshold selection, replicated runs, communication energy, and adaptive local budgets on heterogeneous consumer devices.

## ACKNOWLEDGMENT

The authors acknowledge support from FAPEMIG under Grants #APQ00923-24 and #APQ-00769-26; the FAPESP MCTIC/CGI Research Project SFI2 – Slicing Future Internet Infrastructures under Grant 2018/23097-3; and Fundação para a Ciência e a Tecnologia (FCT) through the R&D Unit Project UID/00319/2025, Centro ALGORITMI (ALGORITMI/UM) and the ANIMA Institute for supporting this work.

## REFERENCES

[1] A. P. Oliveira, T. Carraquico, and C. Martinez-Perez, "Beyond efficiency: A systematic review of energy consumption and carbon footprint across the ai lifecycle," Sustainability, vol. 18, no. 3, 2026. [Online]. Available: https://www.mdpi.com/2071-1050/18/3/1359

[2] A. d'Orgeval, S. Sheehan, Q. Avenas, E. Assoumou, and V. Sessa, "Generative ai impact assessment through a life cycle analysis of multiple data center typologies," Applied Energy, vol. 406, p. 127288, 2026. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0306261925020185

[3] Goldman Sachs Research, “Ai is poised to drive 160% increase in data center power demand,"https://www.goldmansachs.com/insights/ articles/AI-poised-to-drive-160-increase-in-power-demand, May 2024, accessed: 2026-03-02.

[4] F. Wu, Z. Li, Y. Li, B. Ding, and J. Gao, "Fedbiot: Llm local fine-tuning in federated learning without full model," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD '24. New York, NY, USA: Association for Computing Machinery, 2024, p. 3345–3355. [Online]. Available: https://doi.org/10.1145/3637528.3671897

[5] N. Hinov, “The energy hunger of ai: Large language models as challenges and enablers for sustainable energy," Energies, vol. 18, no. 17, 2025. [Online]. Available: https://www.mdpi.com/1996-1073/ 18/17/4701

[6] L. F. Rodrigues Moreira, R. Moreira, and L. G. Ferreira Rodrigues, “Energy-aware ensemble learning for coffee leaf disease classification," in 2026 IEEE 23rd Consumer Communications & Networking Conference (CCNC), 2026, pp. 1–6.

[7] S. Iftikhar, S. H. Alsamhi, and S. Davy, "Enhancing sustainability in llm training: Leveraging federated learning and parameter-efficient finetuning," IEEE Transactions on Sustainable Computing, vol. 10, no. 6, pp. 1158–1172, 2025.

[8] H. Woisetschläger, A. Erben, S. Wang, R. Mayer, and H.-A. Jacobsen, “Federated fine-tuning of llms on the very edge: The good, the bad, the ugly," in Proceedings of the Eighth Workshop on Data Management for End-to-End Machine Learning, ser. DEEM '24. New York, NY, USA: Association for Computing Machinery, 2024, p. 39–50. [Online]. Available: https://doi.org/10.1145/3650203.3663331

[9] F. Wu, J. Hu, G. Min, and S. Wang, "Adaptive rank allocation for federated parameter-efficient fine-tuning of language models," IEEE Transactions on Computers, pp. 1–14, 2026.

[10] S. Thuau, S. Haidar, A. Bajracharya, and R. Chelouah, "Frugal federated learning for violence detection: A comparison of lora-tuned vlms and personalized cnns," in 2025 3rd International Conference on Federated Learning Technologies and Applications (FLTA), 2025, pp. 196–202.

[11] M. Xu, D. Cai, Y. Wu, X. Li, and S. Wang, "FwdLLM: Efficient federated finetuning of large language models with perturbed inferences," in 2024 USENIX Annual Technical Conference (USENIX ATC 24). Santa Clara, CA: USENIX Association, Jul. 2024, pp. 579–596. [Online]. Available: https://www.usenix.org/conference/ atc24/presentation/xu-mengwei

[12] Z. Li, S. Wu, L. Li, and S. Zhang, "Energy-efficient split learning for fine-tuning large language models in edge networks," IEEE Networking Letters, vol. 7, no. 3, pp. 176–180, 2025.

[13] Y. Zhao, L. Feng, F. Zhou, W. Li, Z. Xiong, H. Du, C. Wu, S. Guo, T. Q. S. Quek, and Z. Han, “Energy-efficient over-the-air computation for federated generative model fine-tuning in unmanned vehicle-assisted

disaster relief," IEEE Transactions on Cognitive Communications and Networking, vol. 12, pp. 56–73, 2026.

[14] A. Krzywaniak, P. Czarnul, and J. Proficz, “"Dynamic gpu power capping with online performance tracing for energy efficient gpu computing using depo tool," Future Generation Computer Systems, vol. 145, pp. 396–414, 2023. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0167739X23001267

[15] A. C. Newkirk, J. Fernandez, J. Koomey, I. Latif, E. Strubell, A. Shehabi, and C. Samaras, "Empirically-calibrated h100 node power models for accurate ai training energy estimation," Environmental Research: Energy, vol. 2, no. 4, p. 045016, dec 2025. [Online]. Available: https://doi.org/10.1088/2753-3751/ae2486

[16] K. Gogineni, A. Suvizi, and G. Venkataramani, "Llms on a budget: System-level approaches to power-efficient and scalable fine-tuning," IEEE Open Journal of the Computer Society, vol. 6, pp. 987–1000, 2025.

[17] M. Ozkan and C. S. Ozkan, "Federated carbon intelligence for sustainable ai: Real-time optimization across heterogeneous hardware fleets," MRS Energy & Sustainability, Nov 2025. [Online]. Available: https://doi.org/10.1557/s43581-025-00146-1

[18] Z. Li, P. Tang, X. Wang, X. Liu, and P. Mou, "Pcf-rwkv: Large language model for product carbon footprint estimation," Sustainability, vol. 17, no. 3, 2025. [Online]. Available: https: //www.mdpi.com/2071-1050/17/3/1321

[19] W. Kuang, B. Qian, Z. Li, D. Chen, D. Gao, X. Pan, Y. Xie, Y. Li, B. Ding, and J. Zhou, “Federatedscope-llm: A comprehensive package for fine-tuning large language models in federated learning," in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, ser. KDD '24. New York, NY, USA: Association for Computing Machinery, 2024, p. 5260–5271. [Online]. Available: https://doi.org/10.1145/3637528.3671573

[20] B. Hanindhito, B. Patel, and L. K. John, "Large language model fine-tuning with low-rank adaptation: A performance exploration," in Proceedings of the 16th ACM/SPEC International Conference on Performance Engineering, ser. ICPE '25. New York, NY, USA: Association for Computing Machinery, 2025, p. 92–104. [Online]. Available: https://doi.org/10.1145/3676151.3719377

[21] B. Kaushik, A. Taneja, and S. Dahiya, “Toward sustainable ai: A review of energy-efficient large language models," in 2025 8th International Conference on Computing Methodologies and Communication (ICCMC), 2025, pp. 944–951.

[22] R. Jain, The art of computer systems performance analysis. john wiley & sons, 1990.

[23] L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Blázquez, G. Penedo, L. Tunstall, A. Marafioti, H. Kydlíček, A. P. Lajarín, V. Srivastav, J. Lochner, C. Fahlgren, X.-S. Nguyen, C. Fourrier, B. Burtenshaw, H. Larcher, H. Zhao, C. Zakka, M. Morlon, C. Raffel, L. von Werra, and T. Wolf, "Smollm2: When smol goes big – data-centric training of a small language model," 2025. [Online]. Available: https://arxiv.org/abs/2502.02737

[24] M. Zanella and I. Ben Ayed, "Low-rank few-shot adaptation of visionlanguage models," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2024, pp. 1593–1603.

[25] B. Courty, V. Schmidt et al., "mlco2/codecarbon: v3.2.1," Zenodo, 2025. [Online]. Available: https://doi.org/10.5281/zenodo.18129798

[26] J. Zhu, S. He, P. He, J. Liu, and M. R. Lyu, "Loghub: A large collection of system log datasets for ai-driven log analytics," 2023. [Online]. Available: https://arxiv.org/abs/2008.06448

[27] I. Baldin, A. Nikolich, J. Griffioen, I. S. Monga, K.-C. Wang, T. Lehman, and P. Ruth, "Fabric: A national-scale programmable experimental network infrastructure," IEEE Internet Computing, vol. 24, no. 3, pp. 38–47, 2020.

[28] K. Keahey, J. Anderson, Z. Zhen, P. Riteau, P. Ruth, D. Stanzione, M. Cevik, J. Colleran, H. S. Gunawi, C. Hammock, J. Mambretti, A. Barnes, F. Halbach, A. Rocha, and J. Stubbs, "Lessons learned from the chameleon testbed," in USENIX Annual Technical Conference (USENIX ATC '20). USENIX Association, Jul. 2020.