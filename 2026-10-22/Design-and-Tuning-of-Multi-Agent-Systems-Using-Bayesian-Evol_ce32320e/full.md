# Design and Tuning of Multi-Agent Systems Using Bayesian Evolutionary Optimization

Dominik Jehle   
jehled@cs.uni-freiburg.de University of Freiburg   
Freiburg im Breisgau, Germany

Tom Zehle tom.zehle@tue.ellis.eu University of Freiburg Freiburg im Breisgau, Germany

## Abstract

Designing effective multi-agent systems (MAS) remains a challenging and resourceintensive task. We propose a Bayesian Evolutionary Optimization framework for automatically discovering task-specific MAS topologies. The approach combines evolutionary operators to generate diverse workflow graphs with Bayesian optimization to focus cost intensive evaluations on promising candidates. In contrast to LLM-based candidate selection, our approach uses a graph-aware Gaussian Process and Expected Improvement to provide an explicit selection criterion during evolutionary search. We evaluate the method on BigCodeBench for code generation and on a tabular preprocessing benchmark using TabPFN. On BigCodeBench, evolved architectures achieve the highest overall performance among the evaluated agent systems. On temporal tabular datasets, optimized MAS preprocessing improves over the default TabPFN model on several tasks, while remaining competitive on IID datasets. Overall, the results demonstrate the potential of Bayesian selection for guiding evolutionary MAS search under expensive evaluation budgets.

## 1 Introduction

Multi-agent systems (MAS) have demonstrated strong capabilities across a wide range of complex tasks, including coding and data analysis [1, 2]. By combining multiple specialized agents, MAS can leverage complementary expertise to solve problems that are often beyond the capabilities of a single agent. However, different tasks may benefit from different agent roles and workflow structures, making it challenging and resource-intensive to manually design an effective, task-specific MAS topology [3, 4]. Recent research has shown the potential of combining large language models (LLMs) with evolutionary algorithms. In these approaches, LLMs are used to generate and refine candidate solutions, while evolutionary algorithms provide an efficient mechanism for exploring the search space and discovering increasingly effective system designs [5].

To address the high cost of manually designing task-specific multi-agent systems, we propose a Bayesian Evolutionary Optimization method for automated MAS design. Our approach combines evolutionary algorithms to generate and diversify candidate MAS topologies with Bayesian optimization to efficiently identify the most promising architectures for evaluation.

Contributions. I) We propose the Bayesian Evolutionary Optimization Pipeline, a comprehensive MAS optimization framework that combines evolutionary operators with a specialized Bayesian optimization strategy for candidate selection. The pipeline is based on the Promptolution framework [6] and is evaluated on two distinct benchmark settings. II) We introduce a tabular data benchmark that evaluates MAS-based preprocessing for tabular datasets. The benchmark assesses whether MAS optimization can improve data preparation for state-of-the-art tabular prediction models such as TabPFN [7, 8].

## 2 Related Work

The automated design and optimization of multi-agent systems (MAS) has recently gained significant research interest. ADAS [9] uses a meta-agent to iteratively improve existing systems and generate new MAS architectures. By building on previously discovered high-performing systems, ADAS outperforms conventional prompting and manually designed agent systems [9]. MASS [10] similarly optimizes MAS iteratively by separately optimizing graph topologies and agent prompts before combining them. It achieves strong performance compared to other automated frameworks such as AFLOW [11].

Evolutionary approaches have also been applied to MAS optimization. EVOAGENT [12] evolves MAS topologies through mutation and crossover, with an LLM guiding candidate selection, and outperforms prompting strategies and agent frameworks such as AgentVerse and AutoAgent. EVOMAS [13] combines evolutionary operators with LLM-based selection and achieves superior performance over single-agent systems, handcrafted architectures, and other automated methods across benchmarks including SWE-bench and WorkBench. Beyond LLM-based MAS optimization, GAPPO [14] integrates genetic algorithms with reinforcement learning through mutation, crossover, and selection.

While existing evolutionary MAS approaches primarily rely on LLM-based selection, we investigate Bayesian optimization as an alternative selection mechanism. Rather than asking an LLM to judge candidate architectures, our method explicitly models previously evaluated graph topologies and uses Expected Improvement to select candidates for evaluation. This provides an explicit, model-based selection criterion, reducing reliance on stochastic LLM-based judgments.

## 3 Method

![](images/661173ada03766346ed6f5b0d0ec43d1a137c243cc294863c594287c74972556.jpg)  
Figure 1: Overview of the proposed optimization pipeline: Candidate MAS topologies are generated through evolutionary expansion, selected via Bayesian optimization, instantiated with agent types and prompts, and evaluated iteratively.

Algorithm 1 Bayesian Evolutionary Optimization for MAS Design   
Require: Target task T, initial population size P, candidates per iteration N   
Ensure: Best-performing MAS architecture   
1: Generate an initial population of workflow graphs using an LLM   
2: Evaluate the initial population   
3: Add evaluated graphs and their scores to the population   
4: while optimization budget is not exhausted do   
5: Generate N candidate graphs using mutation and crossover   
6: Compute graph similarities to previously evaluated graphs   
7: Fit a Gaussian Process using the observed evaluation scores   
8: Compute Expected Improvement (EI) for each candidate   
9: Select candidate x<sup>∗</sup> with the highest EI   
10: Assign agent types and prompts to x<sup>∗</sup>   
11: Evaluate the resulting MAS on task T   
12: Add x<sup>∗</sup> and its evaluation score to the population   
13: end while   
14: return Best-performing evaluated MAS architecture

The proposed method extends the Promptolution framework [6] with a complete optimization pipeline for multi-agent systems (MAS). The pipeline combines evolutionary exploration with Bayesian selection to efficiently discover task-specific MAS architectures. Evolutionary operators generate diverse candidate structures, while Bayesian optimization provides an explicit selection criterion for deciding which candidates should undergo expensive end-to-end evaluation. This separation allows the search to maintain structural diversity without requiring every generated topology to be evaluated. The complete procedure is summarized as Pseudocode in Algorithm 1.

Warm-Up Phase and Evolutionary Expansion. During the warm-up phase, an initial population of workflow graphs is generated using an LLM as the basis for evolutionary optimization. The number of nodes is randomly determined for each graph, producing workflows of varying sizes and topologies. The evolutionary expansion phase is driven by the application of mutation and crossover operators. For mutation, one of five mutation operators is selected at random and applied to a randomly chosen

![](images/a1aa48d2369d3e2262f5861aab26b7ade11642ee981ae89ef3b5914e86663c83.jpg)  
Figure 2: Example of a mutation using the add node operator. The other mutation operators are add edge, remove node, remove edge, and rewire edge.  
Figure 3: Example of a crossover operation. Nodes and edges are sampled from both parent graphs to construct the offspring.

graph from the current population. This produces a new candidate graph, as illustrated in Figure 2. For crossover, two parent graphs are selected from the population. The size of the offspring graph is randomly chosen between the sizes of the two parent graphs. Nodes and edges are then sampled from both parents with a predefined probability to construct the offspring, as illustrated in Figure 3.

Bayesian Selection Process After generating N candidate graph topologies, only a single candidate is selected for evaluation due to the high computational cost of the evaluation process. To identify the most promising candidate, we employ Bayesian optimization with Expected Improvement (EI) [15–17]. Since the optimization operates on graph structures, graph similarity is measured using a normalized, type-aware graph edit distance (GED) [18]. Let $V _ { x }$ and $E _ { x }$ denote the vertex and edge sets of graph x. The similarity kernel is defined as:

$$
k ( x , x ^ { \prime } ) = 1 - \frac { \mathrm { G E D } ( x , x ^ { \prime } ) } { | V _ { x } | + | E _ { x } | + | V _ { x ^ { \prime } } | + | E _ { x ^ { \prime } } | } .\tag{1}
$$

Using this kernel, a Gaussian Process estimates the expected performance $\mu ( x )$ and predictive uncertainty $\sigma ( x )$ of each candidate based on previously evaluated graph topologies. Expected Improvement is designed to balance exploration and exploitation by estimating the potential improvement over the best observed performance $f _ { \mathrm { b e s t } } .$ It is defined as:

$$
\mathrm { E I } ( x ) = ( \mu ( x ) - f _ { \mathrm { b e s t } } - \xi ) \Phi ( z ) + \sigma ( x ) \phi ( z ) , \quad z = \frac { \mu ( x ) - f _ { \mathrm { b e s t } } - \xi } { \sigma ( x ) }\tag{2}
$$

Here, Φ and $\phi$ denote the standard normal CDF and PDF, respectively, while $\xi$ controls exploration. The candidate with the highest Expected Improvement is selected for evaluation (details in Appendix A).

$$
x ^ { \star } = \underset { x \in C } { \arg \operatorname* { m a x } } \operatorname { E I } ( x ) .\tag{3}
$$

Prompt Assignment After selecting a graph topology, each node is instantiated as an agent by assigning an agent type and system prompt. The framework initially provides three base types: Planner, Executor, and Validator. An LLM first analyzes the task and generates additional taskspecific agent types. Then, a three-stage pipeline configures each node: a scheduler LLM assigns agent types based on the graph and task, another LLM generates corresponding system prompts, and the final configuration combines agent type, prompt, and required tools. The resulting graph is a fully instantiated multi-agent system ready for evaluation.

## 4 Experimental Setup

We evaluated our method on two benchmarks. First, we developed a custom tabular benchmark to assess whether MAS optimization can improve preprocessing for downstream prediction. The objective is to optimize the preprocessing pipeline such that the resulting data maximizes the predictive information available to the model. Second, we evaluated our approach on BigCodeBench [19] as a traditional MAS benchmark. As baselines, we considered a Single Agent system and a manually designed Planner → Executor → Validator MAS.

Across all experiments, we used Gemma 4 31B [20] as both the meta-model and the underlying language model for all agents. To improve reproducibility, we fixed the sampling temperature to 0.25 [21, 22]. All benchmarks used separate training and test sets.

## 5 Results

![](images/83dc23d7a711bc645ed0df3755fe93b105a2c9b1d2e462784090e7e7cac3723a.jpg)  
Figure 4: Performance on BigCodeBench aggregated over all four evaluation seeds. The figure shows the results for all four seeds together with the mean performance and its standard deviation.

BigCodeBench For the evaluation on BigCodeBench [19], we used four different random seeds, each containing 300 benchmark tasks. Figure 4 shows that the evolved architecture achieved the highest mean performance across the seeds (0.410), compared with the Planner–Executor–Validator architecture (0.393) and the Single Agent baseline (0.367). This corresponds to an absolute improvement of 0.017 over the manually designed multi-agent system and 0.043 over the Single Agent baseline. The evolved architecture improved mean performance by 4.3% over the Planner–Executor–Validator architecture and by 11.7% over the Single Agent baseline across the four seeds. The results suggest that architecture evolution can identify effective agent configurations that improve code-generation performance beyond both manually designed multi-agent coordination and single-agent execution.

Table 1 complements these results by reporting the total number of unique tasks solved across all four seeds. Since tasks may overlap between seeds, each task is counted only once, resulting in 775 unique tasks. The evolved architecture solved the most unique tasks (295; 38.1%), followed by the Planner–Executor–Validator architecture (290; 37.4%) and the Single Agent baseline (265; 34.2%). These results provide complementary evidence that the evolved architecture achieves both stronger average performance across seeds and broader task coverage.

Table 1: Comparison of multi-agent architectures on BigCodeBench across 775 unique tasks.
<table><tr><td>Architecture</td><td>Tasks Solved</td><td>Percentage</td></tr><tr><td>Evolved</td><td>295</td><td>38.1%</td></tr><tr><td>Planner-Executor-Validator</td><td>290</td><td>37.4%</td></tr><tr><td>Single Agent</td><td>265</td><td>34.2%</td></tr></table>

Tabular Prediction For the tabular data benchmark, we used TabPFN as the tabular predictor and included an additional baseline by evaluating the raw performance of TabPFN [7, 8]. We evaluated all methods in two scenarios using datasets from BeyondArena [23]. The first scenario consists of 20 IID regression datasets, while the second uses 5 temporal regression datasets. Both benchmark sets were selected from the 142 BeyondArena datasets, restricting the datasets to a maximum of 50,000 rows. All datasets are listed in Appendix B. The results are shown in Table 2 and 3.

Table 2: Mean $R ^ { 2 }$ of the evolved architecture and baselines on 20 IID regression datasets.
<table><tr><td>Model</td><td>Mean  $R ^ { 2 }$ </td></tr><tr><td>Evolved architecture</td><td>0.611</td></tr><tr><td>Single Agent (no type)</td><td>0.588</td></tr><tr><td>(Planner → Executor → Validator)</td><td>0.527</td></tr></table>

Table 3: Mean $R ^ { 2 }$ of the evolved architecture and baselines on 5 temporal regression datasets.
<table><tr><td>Model</td><td>Mean  $R ^ { 2 }$ </td></tr><tr><td>Evolved architecture</td><td>0.544</td></tr><tr><td>Single Agent (no type)</td><td>0.473</td></tr><tr><td>(Planner → Executor → Validator)</td><td>0.351</td></tr></table>

On both IID and temporal regression datasets, the evolved architecture achieved the best performance among the evaluated agent-based systems, including the Single Agent baseline and the manually designed Planner–Executor–Validator architecture. It achieved a mean $R ^ { 2 }$ of 0.611 on IID datasets and 0.544 on temporal datasets, improving over the Single Agent baseline by 0.023 and 0.071 and over the Planner–Executor–Validator architecture by 0.084 and 0.193. These results highlight the potential of architecture evolution to identify more effective agent configurations for coordinating preprocessing and prediction, particularly on temporal regression tasks.

While raw TabPFN remains a strong baseline, the evolved architecture demonstrates potential for task-specific improvements through its learned preprocessing workflows. For example, it achieved higher predictive performance than raw TabPFN on two of the five temporal datasets. In terms of aggregate performance, the evolved architecture achieved a comparable mean $R ^ { 2 }$ on temporal datasets (0.544 vs. 0.535), although raw TabPFN performed slightly better on IID datasets (0.620 vs. 0.611). These results suggest that evolved preprocessing can be beneficial for certain tasks, while also highlighting opportunities to improve its robustness. Potential sources of performance degradation include hallucinated information, information loss caused by inappropriate feature transformations or type conversions, and unnecessary mathematical operations that introduce misleading information. Prompt optimization is a promising direction for improving the robustness of the preprocessing pipeline. The complete results are shown in Appendix C.

## 6 Conclusion

In this project, we proposed a novel approach for automatically discovering task-specific multi-agent system (MAS) topologies by combining evolutionary algorithms with Bayesian optimization. The evolved architectures achieve higher average performance than both a Single Agent baseline and a manually designed multi-agent system architecture on BigCodeBench and showed competitive performance on the tabular data benchmark.

Limitations and Future Work A key limitation is that topology optimization and prompt generation are optimized separately. While Bayesian optimization guides topology selection, prompt assignment may still affect final performance. Moreover, the GED-based similarity is not guaranteed to be positive semi-definite, which may affect the behavior of the Bayesian selection (see Appendix A). Future work could jointly optimize topologies and prompts, for example by incorporating prompt optimization methods such as Cantante [24]. Further evaluation across a broader range of tasks and domains is needed to assess the generalizability of the approach.

## Acknowledgements

Tom Zehle received funding by the European Union. This work was supported by the European Union’s Horizon Europe research and innovation program under grant agreement No. 101214398 (ELLIOT). Finally, we thank the reviewers for their constructive feedback and contribution to improving the paper.

## References

[1] Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang, Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed Awadallah, Ece Kamar, Rafah Hosn, and Saleema Amershi. Magentic-one: A generalist multi-agent system for solving complex tasks, 2024. URL https://arxiv.org/abs/2411.04468.

[2] Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and Hoang D. Nguyen. Multi-agent collaboration mechanisms: A survey of llms, 2025. URL https://arxiv.org/abs/2501.06322.

[3] Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang. A survey on llm-based multi-agent systems: workflow, infrastructure, and challenges. Vicinagearth, 1(1):9, 2024.

[4] Junda He, Christoph Treude, and David Lo. Llm-based multi-agent systems for software engineering: Literature review, vision, and the road ahead. ACM Transactions on Software Engineering and Methodology, 34(5):1–30, 2025.

[5] He Yu and Jing Liu. Deep insights into automated optimization with large language models and evolutionary algorithms. arXiv preprint arXiv:2410.20848, 2024. URL https://arxiv.org/ abs/2410.20848.

[6] Tom Zehle, Timo Heiß, Moritz Schlager, Matthias Aßenmacher, and Matthias Feurer. promptolution: A unified, modular framework for prompt optimization. In Proceedings ofthe 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 282–296, 2026.

[7] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations 2023, 2023.

[8] Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Mihir Manium, Shi Bin Hoo, Magnus Bühler, Anurag Garg, Dominik Safaric, Jake Robertson, Benjamin Jäger, Simone Alessi, Adrian Hayler, Vladyslav Moroshan, Lennart Purucker, Philipp Singer, Alan Arazi, Julien Siems, Jan Hendrik Metzen, Georg Grab, Nick Erickson, Siyuan Guo, Eliott Kalfon, Simon Bing, David Salinas, Clara Cornu, Lilly Charlotte Wehrhahn, Diana Kriuchkova, Kursat Kaya, Lydia Sidhoum, Marie Salmon, Jerry Chen, Madelon Hulsebos, Yann LeCun, Samuel Müller, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter. Tabpfn-3: Technical report, 2026. URL https://arxiv.org/abs/2605.13986.

[9] Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=t9U3LW7JVX.

[10] Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulic, Anna Korhonen,´ and Sercan Ö. Arık. Multi-agent design: Optimizing agents with better prompts and topologies, 2026. URL https://arxiv.org/abs/2502.02533.

[11] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, et al. Aflow: Automating agentic workflow generation. In International Conference on Learning Representations, volume 2025, pages 34040–34077, 2025.

[12] Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Dongsheng Li, and Deqing Yang. Evoagent: Towards automatic multi-agent generation via evolutionary algorithms. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6192–6217, 2025.

[13] Yuntong Hu, Yuting Zhang, Matthew Trager, Yi Zhang, Shuo Yang, Wei Xia, and Stefano Soatto. EvoMAS: Evolutionary generation of multi-agent systems. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum? id=ic0AGRIkmY.

[14] Zheng Fang, Tao Ma, Jun Huang, Zhao Niu, and Fang Yang. Efficient task allocation in multi-agent systems using reinforcement learning and genetic algorithm. Applied Sciences, 15 (4):1905, 2025.

[15] Peter I Frazier. A tutorial on bayesian optimization. arXiv preprint arXiv:1807.02811, 2018.

[16] Peter I Frazier. Bayesian optimization. In Recent advances in optimization and modeling of contemporary problems, pages 255–278. Informs, 2018.

[17] Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P Adams, and Nando De Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104 (1):148–175, 2015.

[18] Xinbo Gao, Bing Xiao, Dacheng Tao, and Xuelong Li. A survey of graph edit distance. Pattern Analysis and applications, 13(1):113–129, 2010.

[19] Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, Simon Brunner, Chen Gong, Thong Hoang, Armel Randy Zebaze, Xiaoheng Hong, Wen-Ding Li, Jean Kaddour, Ming Xu, Zhihan Zhang, Prateek Yadav, Naman Jain, Alex Gu, Zhoujun Cheng, Jiawei Liu, Qian Liu, Zijian Wang, Binyuan Hui, Niklas Muennighoff, David Lo, Daniel Fried, Xiaoning Du, Harm de Vries, and Leandro Von Werra. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions, 2025. URL https://arxiv.org/abs/2406.15877.

[20] Google DeepMind. Gemma 4 model card. https://ai.google.dev/gemma/docs/core/ model\_card\_4, 2026. Accessed: 2026-08-06.

[21] Lujun Li, Lama Sleem, Niccolo’ Gentile, Geoffrey Nichil, and Radu State. Exploring the impact of temperature on large language models: Hot or cold? Procedia Computer Science, 264:242– 251, 2025. ISSN 1877-0509. doi: https://doi.org/10.1016/j.procs.2025.07.135. URL https: //www.sciencedirect.com/science/article/pii/S1877050925021854. International Neural Network Society Workshop on Deep Learning Innovations and Applications 2025.

[22] Larry Grullon-Polanco. Llm-assisted genre analysis: The effect of sampling temperature on reliability. Research Methods in Applied Linguistics, 5(2):100314, 2026.

[23] Lennart Purucker, Andrej Tschalzev, Nick Erickson, Gioia Blayer, David Holzmüller, Alan Arazi, Alexander Pfefferle, Mustafa Tajjar, Gaël Varoquaux, and Frank Hutter. Beyond iid: How general are tabular foundation models, really?, 2026. URL https://arxiv.org/abs/ 2606.30410.

[24] Tom Zehle. Cantante: Optimizing agentic systems via contrastive credit attribution. arXiv preprint arXiv:2605.13295, 2026.

## A Bayesian Selection Process

After generating N candidate graph topologies during the evolutionary expansion phase, only a single candidate is selected for evaluation, as the evaluation process is computationally expensive.

To efficiently identify the most promising candidate, we employ Bayesian optimization with the Expected Improvement (EI) acquisition function [15–17]. Since the optimization operates on graph structures rather than continuous vectors, graph similarity is measured using a normalized, type-aware graph edit distance (GED) [18]. Let $V _ { x }$ and $E _ { x }$ denote the vertex and edge sets of graph x. The similarity kernel is defined as

$$
k ( x , x ^ { \prime } ) = 1 - \frac { \mathrm { G E D } ( x , x ^ { \prime } ) } { | V _ { x } | + | E _ { x } | + | V _ { x ^ { \prime } } | + | E _ { x ^ { \prime } } | }\tag{4}
$$

Using this kernel, Gaussian Process regression estimates the posterior mean and variance for each candidate graph:

$$
\mu ( x _ { * } ) = \bar { y } + k _ { * } ^ { \top } K _ { y } ^ { - 1 } ( y - \bar { y } )\tag{5}
$$

$$
\sigma ^ { 2 } ( x _ { * } ) = k ( x _ { * } , x _ { * } ) - k _ { * } ^ { \top } K _ { y } ^ { - 1 } k _ { * }\tag{6}
$$

where $K _ { y }$ denotes the kernel matrix computed from all previously evaluated graphs, augmented with a small diagonal jitter term to ensure numerical stability. Furthermore, $k _ { * } = [ k ( x _ { * } , x _ { i } ) ]$ <sub>i</sub> represents the similarity between the candidate graph and all previously evaluated graphs, while y¯ denotes the mean observed evaluation score.

The GED-based kernel is not positive semi-definite in general. In our implementation, negative posterior variances were clamped to $1 0 ^ { - 1 2 }$ rather than corrected, which biases selection towards candidates dissimilar to evaluated graphs and deviates from standard EI. A projection of the joint kernel matrix onto the PSD cone would resolve this and is left for future work.

The Expected Improvement acquisition function evaluates each candidate relative to the bestperforming graph observed so far, $f _ { \mathrm { b e s t } } = \mathrm { m a x } _ { i } y _ { i }$ , with exploration parameter $\xi \ge 0$

$$
I ( x ) = \mu ( x ) - f _ { \mathrm { b e s t } } - \xi\tag{7}
$$

$$
z = { \frac { I ( x ) } { \sigma ( x ) } }\tag{8}
$$

$$
\mathrm { E I } ( x ) = I ( x ) \Phi ( z ) + \sigma ( x ) \phi ( z )\tag{9}
$$

where $\Phi$ and ϕ denote the cumulative distribution function (CDF) and probability density function (PDF) of the standard normal distribution, respectively. The candidate selected for evaluation is the graph that maximizes the Expected Improvement over the sampled candidate set:

$$
x ^ { \star } = \arg \operatorname* { m a x } _ { x } \operatorname { E I } ( x ) .\tag{10}
$$

The selected graph topology $x ^ { \star }$ is subsequently evaluated and added to the population for future optimization iterations [15].

## B Tabular Prediction Datasets

Table 4: Overview of the IID datasets from BeyondArena [23], including the number of rows.
<table><tr><td>Dataset Name</td><td>Number of Rows</td></tr><tr><td>airfoil_self_noise concrete_compressive_strength</td><td>1,503</td></tr><tr><td>healthcare_insurance_expenses</td><td>1,030</td></tr><tr><td rowspan="3">miami_housing superconductivity</td><td>1,338</td></tr><tr><td>13,776 21,263</td></tr><tr><td rowspan="3">161 252 345</td></tr><tr><td>cirrhosis_patient_survival_prediction body_density_prediction</td></tr><tr><td>blood_tests_drink_prediction obesity_estimation 498 forest_fires 517</td></tr></table>

Table 5: Overview of the temporal datasets from BeyondArena [23], including the number of rows.
<table><tr><td>Dataset Name</td><td>Number of Rows</td></tr><tr><td>garments_worker_productivity</td><td>1,197</td></tr><tr><td>coffee_rating_prediction</td><td>2,369</td></tr><tr><td>mercedes_benz_greener_manufacturing</td><td>4,204</td></tr><tr><td>sberbank_housing_market_forecasting</td><td>27,195</td></tr><tr><td>california_house_prices_2020</td><td>41,528</td></tr></table>

We use datasets from the BeyondArena [23] benchmark. The individual datasets remain subject to the licenses and terms of their respective original sources. We do not redistribute the underlying datasets; only aggregate benchmark results are reported.

## C Additional Tabular Prediction Results

Table 6: Results on the temporal datasets.
<table><tr><td>Dataset</td><td>Evolved</td><td>Single Agent</td><td>Plan.-Ex.-Val.</td><td>Raw TabPFN</td></tr><tr><td>california_house_prices_2020</td><td>0.940543</td><td>0.940592</td><td>0.940009</td><td>0.941049</td></tr><tr><td>coffee_rating_prediction</td><td>0.111621</td><td>-0.252738</td><td>0.005208</td><td>0.203051</td></tr><tr><td>garments_worker_productivity</td><td>0.444023</td><td>0.453376</td><td>-0.404151</td><td>0.382659</td></tr><tr><td>mercedes_benz_greener_manu.</td><td>0.580626</td><td>0.580993</td><td>0.580140</td><td>0.578253</td></tr><tr><td>sberbank_housing_market_fore.</td><td>0.641153</td><td>0.640405</td><td>0.634187</td><td>0.572113</td></tr></table>

Table 7: Results on the IID datasets.
<table><tr><td>dataset</td><td>Evolved</td><td>Single Agent</td><td>Plan.-Ex.-Val.</td><td>Raw TabPFN</td></tr><tr><td>airfoil_self_noise</td><td>0.975258</td><td>0.975962</td><td>0.956852</td><td>0.971863</td></tr><tr><td>blood_tests_drink_prediction</td><td>0.129454</td><td>0.118084</td><td>0.118628</td><td>0.118950</td></tr><tr><td>body_density_prediction</td><td>0.692294</td><td>0.692378</td><td>0.676094</td><td>0.688951</td></tr><tr><td>cirrhosis_patient_survival_pred.</td><td>0.263664</td><td>0.259219</td><td>0.274007</td><td>0.318891</td></tr><tr><td>concrete_compressive_strength</td><td>0.951657</td><td>0.951404</td><td>0.943976</td><td>0.951536</td></tr><tr><td>fiat_500</td><td>0.853831</td><td>0.855042</td><td>0.846765</td><td>0.855032</td></tr><tr><td>forest_fires</td><td>0.012283</td><td>0.015130</td><td>0.012023</td><td>0.007024</td></tr><tr><td>healthcare_insurance_expenses</td><td>0.830230</td><td>0.830769</td><td>0.829713</td><td>0.830451</td></tr><tr><td>immoscout_german_house_pri.</td><td>0.697193</td><td>0.699006</td><td>0.237424</td><td>0.694509</td></tr><tr><td>miami_housing</td><td>0.952044</td><td>0.951738</td><td>0.951409</td><td>0.952178</td></tr><tr><td>mutual_funds_india</td><td>0.777890</td><td>0.320690</td><td>0.221254</td><td>0.789952</td></tr><tr><td>obesity_estimation</td><td>0.109408</td><td>0.103002</td><td>-0.025905</td><td>0.161054</td></tr><tr><td>qsar_aquatic_toxicity</td><td>0.649739</td><td>0.646528</td><td>0.656876</td><td>0.660538</td></tr><tr><td>qsar_fish_toxicity</td><td>0.621777</td><td>0.637216</td><td>0.615401</td><td>0.637216</td></tr><tr><td>qsar_tid_11</td><td>0.781384</td><td>0.784702</td><td>0.778486</td><td>0.784864</td></tr><tr><td>santander_transaction_value</td><td>0.425438</td><td>0.423872</td><td>0.421000</td><td>0.423949</td></tr><tr><td>student_portuguese_per.</td><td>0.340675</td><td>0.341694</td><td>0.337756</td><td>0.340228</td></tr><tr><td>superconductivity</td><td>0.931562</td><td>0.931529</td><td>0.931584</td><td>0.931741</td></tr><tr><td>wine_quality</td><td>0.522284</td><td>0.521495</td><td>0.508275</td><td>0.558394</td></tr><tr><td>wine_world_cost</td><td>0.698428</td><td>0.699150</td><td>0.248855</td><td>0.714463</td></tr></table>

Table 8: Mean $R ^ { 2 }$ performance of the evolved architecture and the baseline methods on the 20 IID regression datasets.
<table><tr><td>Model</td><td>Mean  $R ^ { 2 }$ </td></tr><tr><td>Evolved architecture</td><td>0.611</td></tr><tr><td>Single Agent (no type)</td><td>0.588</td></tr><tr><td>(Planner → Executor → Validator)</td><td>0.527</td></tr></table>

Table 9: Mean $R ^ { 2 }$ performance of the evolved architecture and the baseline methods on the 5 temporal regression datasets.
<table><tr><td colspan="2">Model Mean  $R ^ { 2 }$ </td></tr><tr><td>Evolved architecture</td><td>0.544</td></tr><tr><td>Single Agent (no type)</td><td>0.473</td></tr><tr><td>(Planner → Executor → Validator)</td><td>0.351</td></tr></table>

## D Relative $R ^ { 2 }$ Difference Between Evolved and Raw TabPFN on Temporal Datasets

![](images/6bce54af91f520861b818cca2b074d3495249fb58197cf81dafaf96bd27e131f.jpg)  
Figure 5: Relative $R ^ { 2 }$ difference between the evolved architecture and the raw TabPFN baseline on the 5 temporal regression datasets.

## E Additional BigCodeBench Results

![](images/52c8f0653cf102b5c86e0d11c56dd7e4374a83f6a625c96138a7246ae6498908.jpg)  
Figure 6: BigCodeBench solve-set overlap across the three considered architectures: Evolved, Planner-Executor-Validator, and Single Agent. Each bar represents the number of tasks belonging to a specific solve-set combination, while the connected markers indicate which architectures successfully solved the corresponding tasks. The largest group consists of 438 tasks that were not solved by any of the three architectures. A further 212 tasks were solved by all three architectures. Among the pairwise overlaps, 41 tasks were solved by both Evolved and Planner-Executor-Validator but not by Single Agent, while 24 tasks were solved by both Evolved and Single Agent and another 24 tasks were solved by both Planner-Executor-Validator and Single Agent. Considering tasks solved exclusively by a single architecture, Evolved solved 18 tasks, Planner-Executor-Validator solved 13 tasks, and Single Agent solved 5 tasks.

## F Architecture Search Configurations

Table 10: Evolutionary search configuration for BigCodeBench.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Dataset / task</td><td>BigCodeBench (code generation)</td></tr><tr><td>Population size</td><td>5</td></tr><tr><td>Generations</td><td>10</td></tr><tr><td>Crossover / mutation prob.</td><td>0.5 / 0.5</td></tr><tr><td>Candidates per step</td><td>50</td></tr><tr><td>Search space (nodes / edges)</td><td> $2 { - } 6 / \leq 9 \mathrm { ( a c y c l i c ) }$ </td></tr><tr><td>Task LLM</td><td>Gemma-4-31B (temp. 0.25)</td></tr><tr><td>Meta LLM</td><td>Gemma-4-31B (temp. 0.25)</td></tr></table>

Table 11: Evolutionary search configuration for the TabularData run.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Dataset / task</td><td>TabularData (SQL feature engineering)</td></tr><tr><td>Population size</td><td>5</td></tr><tr><td>Generations</td><td>10</td></tr><tr><td>Crossover / mutation prob.</td><td>0.5 / 0.5</td></tr><tr><td>Candidates per step</td><td>50</td></tr><tr><td>Search space (nodes / edges)</td><td>2–6 / ≤9 (acyclic)</td></tr><tr><td>Task LLM</td><td>Gemma-4-31B (temp. 0.25)</td></tr><tr><td>Meta LLM</td><td>Gemma-4-31B (temp. 0.25)</td></tr></table>