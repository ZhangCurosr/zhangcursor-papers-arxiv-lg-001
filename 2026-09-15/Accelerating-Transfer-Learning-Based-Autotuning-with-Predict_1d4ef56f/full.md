# Accelerating Transfer-Learning-Based Autotuning with Predictive LLVM IR Performance Ranking

Md Arafat Hossain<sup>∗</sup>   
Department of Computer Science   
Iowa State University   
Ames, Iowa, USA   
arafat@iastate.edu   
Xingfu Wu   
MCS   
Argonne National Laboratory   
Lemont, Illinois, USA   
xingfu.wu@anl.gov   
Thomas Randall<sup>∗</sup>   
School of Computing   
Clemson University   
Clemson, South Carolina, USA   
tlranda@clemson.edu   
Rong Ge   
School of Computing   
Clemson University   
Clemson, South Carolina, USA   
rge@clemson.edu   
Akash Dutta   
AMD   
Austin, Texas, USA   
Akash.Dutta@amd.com   
Ali Jannesari   
Department of Computer Science   
Iowa State University   
Ames, Iowa, USA   
jannesar@iastate.edu

## Abstract

As the complexity of High Performance Computing (HPC) ecosystems continually increases, achieving optimal performance becomes a challenge. Traditional performance autotuning techniques provide promising means to navigate this complexity, these techniques remain computationally intensive and require many evaluations to find optimal configurations.

This work proposes an autotuning framework that designs a machine learning-based ensemble LLVM Intermediate Representation (IR) ranker, Neural Configuration Scorer (NCS). NCS ranks the performance of IRs sampled by a transfer-learning-based autotuner, improving the eficiency of the tuning process by reducing tuning overheads and circumventing subpar evaluations. By leveraging knowledge from related tasks, we are able to efectively exploit the transfer relationship to access high-performing configurations in fewer samples than traditional techniques that rely upon iterative refinement. Our framework can achieve similar performance improvements as state-of-the-art autotuning techniques with up to 61.67% fewer evaluations, averaging 27.85% fewer evaluations across various HPC benchmarks.

## Keywords

Performance, Autotuning, Transfer Learning, Few-Shot Learning, Convolutional Neural Network, Intermediate Representations (IRs)

## 1 Introduction

The complexity and scale of High Performance Computing (HPC) ecosystems have exploded over the past decade [19]. Tuning and optimization of applications on HPC systems are worthy uses of compute time and resources; however, pursuing near-optimal con figurations remains prohibitively very time-consuming. Existing literature provides sophisticated means for modeling performance or directly simulating hardware. However, these approaches generally require either large quantities of data that is costly to collect

<sup>∗</sup>Both authors contributed equally.

or limit cost at the expense of precision and flexibility, providing sub-optimal results.

Performance autotuning relies upon discriminating between highly similar program configurations to determine the optimal configuration with respect to one or more objectives such as runtime, energy eficiency and throughput. In this paper, we focus on optimizing the runtime of semantically equivalent source-code expressions of programs with respect to the programs’ input sizes. Predicting these figures of merit using IRs remains challenging, but IR-based performance predictions have largely been limited to generalized program runtime or estimates of the cycle count occupied by small code segments [2, 25]. Despite this prior research and other rich opportunities within IR modeling, performance autotuning has traditionally favored costly empirical performance observations rather than IR-based performance inference. The primary limiting factor has been the massive overhead of generating thousands to millions of IRs that span the entire search space.

In this work, we propose a novel approach which combines a new machine learning-based IR ranking model with GCTLA [24], an existing method for probabilistic transfer learning, to cleverly limit the number of IRs under consideration. The combination is able to greatly improve transfer-learning-based autotuning eficiency for new tasks — i.e., unseen input sizes for a program executed on a particular system.

Our key contributions in this paper include:

• Proposing a framework for transfer-learning-based autotuning with the LLVM IRs of the programs.

• Designing a machine-learning ensemble, NCS (Neural Configuration Scorer), that predicts performance from LLVM IRs, granting it higher fidelity to actual hardware behaviors than the abstract high-level source code changes that form the tuning space.

• Reducing the number of evaluations required for few-shot transfer learning with minor overhead. We demonstrate that the combination of GCTLA+NCS is mutually beneficial and further demonstrate the reliability of NCS through the lens of post-hoc analyses of combining state-of-the-art iterative techniques with NCS.

• Our performance evaluations in few-shot transfer-learning settings require an average of 27.85% fewer evaluations to reach optimal results than prior works, with up to 61.67% fewer evaluations.

The remainder of this paper is organized as follows. Section 2 provides key background information. Section 3 discusses promises and challenges. Section 4 proposes the framework design in detail. Section 5 presents the performance evaluation. Section 6 describes the related work. Section 7 concludes the paper.

## 2 Background

## 2.1 Performance Autotuning

Performance Autotuning [4, 6, 31, 32], also known as autotuning, is a class of algorithmic techniques that helps improve the performance of HPC applications/libraries via a set of tunable parameters. Speedup, throughput, energy usage and latency are typical perfor mance objectives in autotuning, which may be tuned singularly or in combination by algorithms that support multi-objective opti mization. Tuning parameters are typically expressed as code-level optimizations, compiler options and runtime settings.

The primary challenges of HPC autotuning are induced by the size of search spaces (often defined in a combinatorial manner, growing exponentially), the nonlinear complexity of the relationship between configurations and performance, and the high cost of observing the performance relationship. Various autotuning techniques have to strike a balance between three key characteristics:

• Training and refitting cost: The computational burden associated with training, fine-tuning or utilizing the autotuning model beyond the empirical evaluation cost.

• Immediate eficacy: The maximum quality ofresults achieved within the first few evaluations.

• Ultimate eficacy: The maximum quality of results achieved by the end of a search’s full term. For techniques that can try all possible combinations, ultimate eficacy may be better measured when the global optimum is selected.

Excelling at any pair of characteristics requires sacrificing the third. For instance, a technique that emphasizes ultimate eficacy and lower training cost must necessarily have poor immediate eficacy as it cannot know what to avoid in the near-term.

Eficient tuning techniques aim to limit the number of empirical observations while efectively modeling the complex relationship between the objective function and the parameter search space. Sub optimal evaluations are common during the tuning process, but should be limited as the global best and global worst are frequently separated by an order of magnitude. HPC autotuning is made even more challenging as the optimal configuration often varies based on the input size and the target architecture. Overcoming this challenge manually is time-consuming and costly, which leads to the necessity of intelligent search strategies, such as machine learning-based or evolutionary algorithm-based strategies to navigate the vast search space and map the relationship among diferent parameters.

Techniques including GCTLA [24], Random Forests (RF) [7], and multi-fidelity [21, 22, 35] are very eficient in reducing the search space by focusing on probable high-performance regions. Such approaches have proven to yield a substantial amount of performance gains [4, 16]. By automating the search for optimal settings, autotuning helps HPC applications enhance their performance without manual intervention, making it a pivotal tool for the maximization of the capabilities of modern supercomputing systems.

## 2.2 Intermediate Representation

LLVM (Low-Level Virtual Machine) IRs (Intermediate Representations) serve as an abstraction of source code that bridges high-level languages with machine code. They allow compilers to perform optimizations and transformations on source code in a consistent manner, regardless of the original language. The IR is a desirable point for performance analysis due to its simplicity, expressiveness, and abstraction. IRs are hardware independent and enable performance analysis without being tied to a specific source language. They also contain semantic and structural information about the source code, including, but not limited to data flow, control flow, memory access patterns, loop insights, store and load instruction details etc. crucial informations that are essential in detecting performance patterns, making them ideal for machine learning based feature extraction and performance prediction tasks.

LLVM IRs are a community standard for code representation for optimization and tuning tasks [13, 15, 30] and generating [20] code in a variety of languages. The LLVM IR still retains several key markers that assist in performance analysis, such as mangled function names that permit restricting analysis’ scope to only the code regions that are afected by optimization eforts. Unlike the LLVM’s Machine Code Analyzer, which estimates the cycle counts and IPC of straight-line assembly fragments, the LLVM IR still expresses loop behaviors that compose many useful optimizations such as unrolling, tiling and loop interchanges.

## 2.3 Transfer Learning in Autotuning

Transfer learning improves the eficiency of performance of a target task by leveraging the knowledge from previously learned related tasks [23], which is crucial for machine learning models. Generally, a number of “source” tasks are given as prior knowledge, representing the performance relationship in at least two settings so that diferences in performance with respect to tasks can be observed. Then, a “target” task is proposed, wherein the transfer model must accurately adjust insights gained from the source tasks and their relationship to one another to predict the relationship that will be represented in the target domain. This is highly desirable in HPC autotuning, where the performance relationship between various architectures or inputs will change but should maintain key traits that may have been learned via previous tuning eforts. Eficient reuse of prior tuning information can permit faster approximations of the optimum on new tasks and aids in avoiding regions of the space that are likely to be very costly to evaluate.

Transfer learning is typically evaluated in a few-shot context, where the transfer model must immediately apply its knowledge to maximal efect in a small number of evaluations [24]. However, other transfer learning techniques utilize the transfer to get a head start on the autotuning relationship and iteratively refine the new task relationship as data is collected [35].

## 3 Promises and Challenges

Many autotuning techniques [3, 26, 31, 36] perform iterative refinement over surrogate models. These models maximize sample eficiency, a key trait for reducing the cost of autotuning. However, this requires serially evaluating singular points within the search space, often bottlenecking the learning process on performance observations. Furthermore, a significant number of observations are needed for sophisticated techniques to hone in on the optimum, meaning that the cost of autotuning remains prohibitively high despite the extreme sample eficiency.

Evaluating the potential performance of a source code reconfiguration via the program’s LLVM IR is often significantly cheaper than a complete evaluation, however prior techniques have not leveraged IRs because the search spaces are too large to comparatively evalu ate all possible variations within common search spaces. However a recent transfer-learning autotuner, GCTLA [24], achieves compara ble quality few-shot results without relying on iterative search. The technique forms a probability model of transfer-learning autotuning to permit sampling many high-performing candidate configurations for new tasks without prior iterations or empirical evaluations in the new domain. The sampled subset includes at least one nearoptimal configuration with a high level of confidence.

Despite the promise of identifying high-performing configurations within a few shots, current probabilistic autotuning via GCTLA [24] sufers from two major challenges:

• Non-Discriminative Nature ofthe Sampled Subset. GCTLA lacks a reliable method to determine which sampled configurations are more promising than others. A straightforward approach to identifying the best candidate is to empirically evaluate each one. While this can be less expensive than iterative approaches, it is not ideal for longer-duration tuning.

• Degrading into Random Search. The only way to identify additional high-performing configurations is to increase the sample size. However, as the sample size increases, the proportion of best-performing configurations within it decreases, gradually degrading towards a random sample of the overall population. Consequently, the potential performance gains become disproportionate to the evaluation cost when using a straightforward approach.

The limitations of GCTLA can be addressed by accurately predicting the performance of samples at lower cost than empirical evaluation, which can be performed by analyzing the program’s LLVM IR. We propose a machine learning-based performance model that can rank the sampled IRs. Because our approach relies upon IRs, there is no need to calibrate performance between related tasks and training is only required to fit the models to wholly new domains (ie: new applications). Together, the combined approach of probabilistic sampling and predictive performance models maintains low cost, making it well-suited for few-shot performance autotuning.

Performance prediction is a complex challenge, particularly within the context of HPC applications, where execution behavior exhibits significant variability across diferent hardware architectures, optimization strategies, and runtime configurations. Autotuning represents an especially challenging domain for performance prediction, wherein training data is limited and programs with only minor diferences yield highly varied performance.

## 4 Design

In this section, we present our novel combination ofthe probabilistic transfer-learning based autotuner and neural ranking model along with special considerations made for these components in isolation as well as in combination.

The overall design is presented in Figure 1. For a given benchmark, at least two source tasks must be represented. Each source task is composed of four components: a task indicator, evaluated configurations from the search space on the source task, and the corresponding IRs and empirically measured performance objectives. The training data is further described in Section 5.3, with additional minutiae about our experimental training settings in sections 4.1 and 4.2. Once trained, the GCTLA model samples a number of configurations and we create the corresponding IR for ranking with our Neural Configuration Scorer (NCS) model. The NCS model allows us to sort all sampled IRs from best-to-worst expected performance; the highest-ranked IRs will be lowered to machine-specific code and evaluated to determine the highest-performing configuration.

## 4.1 Probabilistic Autotuner

Despite the limitations of GCTLA noted in Section 3, training a probabilistic transfer model requires minimal data and only requires data from source tasks. Additionally, sampling from the learned probability distribution is computationally negligible, meaning that extra time is available to produce the IRs for use with NCS.

As such, we follow the recommendations from [24] to build our supporting model. This includes the procedure for forming training datasets for transfer-learning and using the top-30% of source task data to fit the GCTLA model. However, instead of sampling a number of configurations based on the budgeted expectation, we intentionally over-sample to create a larger pool of candidates. Then, we use NCS as a discriminative process to sort the sampled candidates and prioritize the highest performing configurations. This method allows us to alleviate a primary limitation of GCTLA and extend its utility to longer-duration searches where it would normally be subject to diminishing returns, provided that the following conditions can be satisfied. First, comparisons should carry low cost to prevent overheads from overshadowing improvements to search eficiency. Second, the comparisons should be as precise as possible near the optimum to observe the best candidates as early as possible.

The latter condition represents a novel opportunity for autotuning and a key contribution of our work. Typically, success in autotuning requires high-fidelity modeling across the entire space to properly disambiguate all levels of performance. Because the samples from GCTLA provide cheap means to cull low-performing areas of the search space from consideration, the NCS model only needs to accurately discriminate between the near-best configurations in the space. This significantly reduces complexity and the amount of data required for NCS to succeed.

## 4.2 Neural Configuration Scorer (NCS)

NCS is the key component of our eficient autotuning pipeline. This section outlines the relevant design choices and how they impact the autotuning process. NCS can be very briefly described as an ensemble model that comparatively ranks the performance of nearoptimal IRs. But as mentioned in Section 3, several challenges must be addressed first.

![](images/d04f0a554c8c94b5225577294a02da0760cbaa91e16c0a4dc8cd995a3a44784c.jpg)  
Figure 1: Our approach uses IRs and configurations from source task data to fit both GCTLA and NCS models. The GCTLA model generates high-performing configurations on a transferred task, which are ranked by the NCS ensemble. The highest ranked generations are empirically evaluated to determine the best-performing configuration

Unlike conventional machine learning approaches that focus on single-class classification or discrete label prediction, we aim to produce a continuous scoring function that reflects the relative performance potential of each generated sample. Instead of predict ing a category of performance, we assign scores such that higher scores will correspond to samples with superior performance. To this end, we utilize an ensemble of convolutional neural networks, each independently evaluating the LLVM IR representations and assigning an unweighted score to each. Each network within the ensemble targets a specific performance percentile of the dataset, ensuring specialization in diferent performance ranges. The final performance score for a given sample is obtained by the summation of individual scores from all models in the ensemble as shown in Figure 2. As such, if all models agree that an IR exceeds their target threshold of performance, the IR is ranked highest, with each doubtful member of the IR equally lowering the score.

4.2.1 Model Architecture. The highest-performing code variants often share structural and semantic similarities in their intermediate representations. However, these similarities do not guarantee stable performance patterns, as performance may degrade due to conflicts between optimization choices and hardware-specific behaviors. An unknown configuration may not be high-performing merely by merit of being similar to a known high-performing configuration. This dynamic nature of the performance landscape—marked by local fluctuations and nonlinear interactions—necessitates a scoring mechanism that does not simply predict performance in isolation, but instead supports relative ranking across a spectrum of samples. Our goal is to design a ranking-aware system capable of discerning nuanced performance potential across a diverse set of generated configurations. By framing the learning task around the gradual separation of samples into performance tiers based on empirical speedup distributions, we allow the model to encode meaningful performance distinctions at multiple levels of granularity. This approach also supports a more robust and flexible pruning strategy. Instead of rigid classification, we enable a soft selection mechanism that favors configurations that resemble high-performance regions while tolerating some lapses in modeling fidelity.

![](images/9e940ee9f551e7535fe62cac9c029abf31074d8fde0a0e0d5514799c7bd75b75.jpg)  
Figure 2: NCS intakes IRs and their corresponding runtimes. Each IR is tokenized, and each runtime is normalized to standardize the data. Next, the data is divided by performance percentiles (50-60%, 60-70%, ... , 90-100%), each of which is used to train a dedicated CNN model that classifies if the IR’s performance will meet or exceed its target range. Finally, the predictions from each CNN model are summed to produce a comprehensive final score for ranking the configurations.

To implement this ranking-based approach, we require a model architecture capable of capturing hierarchical patterns embedded in LLVM IR representations, while remaining eficient under datascarce conditions. To this end, we adopt a CNN architecture, which have demonstrated strong performance in settings with limited training data [10]. Compared to other neural models, their small training and inference footprints make them uniquely suitable for the minimal available data and tight latency constraints present in autotuning.

Nonetheless, in our setting, the CNN architecture strikes a favorable balance: it is capable of learning meaningful structural features from compact datasets, while enabling eficient inference across large volumes of generated samples. This makes it a practical and efective choice for our performance prediction pipeline.

To train this architecture efectively, we construct a dataset that eficiently captures the mapping between LLVM IR representations and their empirical performance. We collect configuration-level IR samples and their corresponding execution times from Ytopt[31], using the same configurations that serve as the foundation for the supporting model’s source-task training (details in Section 5.3). To focus on our analysis and reduce noise, we extract only the functions that are directly afected by configuration changes. Then we pre-process the IR contents by replacing the meta-data and other repetitive information with specific keywords. This is similar to the approach in [5], but with a subtle refinement: we retain loop structures and array access patterns, given their critical role in influencing execution behavior and runtime variability. Rather than processing the raw IR text directly, we design a custom vocabulary to tokenize the LLVM IR into meaningful units. These tokenized sequences are then embedded into dense vector representations through an embedding layer, enabling the CNN to process structural and semantic relationships more efectively. Finally, the output of each CNN model is passed through a fully connected layer that maps the extracted features to a predicted score, enabling the ensemble to rank samples across continuous perforamnce tiers.

4.2.2 Data Partitioning. First, we calculate the speedup of each IR sample relative to the default configuration’s execution time. Then, we partition the same empirical training dataset used to fit the GCTLA model for each benchmark into five performance tiers based on percentiles (50+, 60+, 70+, 80+ and 90+) of speedup. Each of these five percentile groups represents a distinct performance level, allowing our models to learn meaningful distinctions between high- and low-performing configurations. We label each sample positive if it belongs to a particular percentile group and negative otherwise. In order to do that, we came up with the novel idea of ranking the input IRs based on the configuration that might provide the best performance.

4.2.3 Training and Scoring. As mentioned in subsection 4.2.2, the dataset is partitioned into five percentile-based groups according to speedup values. A separate CNN model is trained for each group, ensuring specialization in diferent performance ranges. During inference, each test sample is evaluated by all five models independently. The final score for an individual sample is computed based on the sum of each of the five model predictions. This way, a particular IR can achieve a maximum score of five and a minimum score of zero.

The 50th percentile model acts as an initial filter that eliminates configurations that are likely to perform poorly. As GCTLA is not supposed to sample many configurations that perform worse than this, we do not need lower classification levels and provide this as a safeguard against poor samples permitted by over-sampling. The intermediate models (60th, 70th, and 80th percentiles) refine the selection by iteratively identifying configurations that exhibit increasing levels of performance, guiding the autotuner towards more promising regions. Finally, the 90th percentile model pinpoints the best configurations, ensuring that only the most optimal candidates are retained for empirical evaluation. Using multiple models in this manner is crucial because a single CNN model would struggle to efectively balance search space reduction with performance retention. Since the search space is too large, if we relied on a single model or threshold, we risk either eliminating valuable configurations too early or retaining too many suboptimal ones, leading to ineficient tuning.

Table 1: Search Space Sizes and GCTLA Coverage
<table><tr><td>Benchmark</td><td># Params</td><td># Configurations</td><td>GCTLA Few-Shot Budget</td><td>GCTLA # Unique Configurations</td></tr><tr><td>3mm</td><td>10</td><td>376,320</td><td>30</td><td>≈ 2,500</td></tr><tr><td rowspan="3">Heat3d Syr2k</td><td>6</td><td>10,648</td><td>8</td><td>≈1,600</td></tr><tr><td>6</td><td>10,648</td><td>3</td><td>≈ 800</td></tr><tr><td>9</td><td>1,180,980</td><td>5</td><td>≈ 108,500</td></tr><tr><td>AMG RSBench</td><td>9</td><td>5,196,312</td><td>3</td><td>≈ 316,800</td></tr><tr><td>SW4Lite</td><td>8</td><td>4,752</td><td>15</td><td>≈ 1,800</td></tr></table>

## 5 Performance Evaluation

## 5.1 Experimental Platform

Our empirical evaluations are performed on a single Linux machine running Ubuntu 22.04.4 LTS on kernel version 5.15.0-78-generic. The machine has 1TB DDR4 RAM with 2x AMD EPYC 7742 64-Core Processor (128 logical cores) and 8x NVIDIA A100 GPUS for a total of 320GB GPU memory.

Several ofthe benchmarks we tune utilize Polly [17], a polyhedral loop optimizer for LLVM, to expose source-code-level optimizations for our experiments. The Polly extensions used in this work are publicly available via a customized compiler based on Clang version 13.0.0, provided by the SOLLVE LLVM project [1].

## 5.2 Tunable Applications

We focus our tuning on three key kernels from the Polybench/C [33] collection of benchmarks and three additional applications selected from the Exascale Computing Project proxy applications. Table 1 provides a general overview of the size of each benchmark’s search space as well as the number of unique configurations the GCTLA generates versus its expected budget for few-shot tuning as reported in Randall et al [24]. The few-shot budget of GCTLA is minimal compared to the search space size, but the total number of unique configurations is too large to empirically evaluate. We explicitly move beyond the extremely limited number of evaluations that GCTLA is designed for, but not so far as to exhaust the full extent of the model’s variability.

5.2.1 Polybench/C Kernels. In prior work, the Polybench/C kernels demonstrated some promise for the GCTLA as a generative autotuners, however its explicit few-shot design prevents it from continuing searches to discover better optima within the search spaces.

The three Polybench/C kernels we focus on are 3MM, Heat3d and Syr2k; the tuning parameters for these search spaces are listed in Table 2.

The 3MM kernel previously demonstrated excellent results for the GCTLA on its own in short-term searches, representing an opportunity for our technique to extend the search in a productive manner. The Heat3d kernel previously performed the worst for the GCTLA relative to other tuning techniques, representing an opportunity for our technique to bolster the generative efectiveness when discrimination is dificult. Finally, the Syr2k kernel performed well but also has exhaustive data on two input data scales, representing a richer evaluation subject that can provide deeper insights to the efectiveness of our combined technique compared to its component parts or other tuners.

5.2.2 ECP Proxy Applications. The Polybench/C benchmarks are not trivial to tune but generally yield greater speedups than sophisticated applications that are already somewhat optimized. As such, we follow prior work and also consider our tuner’s efectiveness on the AMG, RSBench and SW4Lite ECP Proxy Applications. The tuning parameters for these search spaces are listed in Table 3.

Similar to 3MM from the Polybench/C collection, the AMG application yielded the best results for the GCTLA on its own, representing opportunity to extend searches to yield greater performance improvements. The RSBench and XSBench applications are highly similar, so in the interest of time we focus on RSBench, which was previously shown to be slightly more challenging to speed up via tuning. Finally we include the GPU-enabled SW4Lite application to demonstrate tuning performance on diferent compute architectures.

## 5.3 Initial Dataset and Problem Formulation

As our work focuses on extended search capabilities, we preserve the same initial dataset as previously used in Randall et al [24]. These datasets are composed of three diferent input data sizes (small, medium, and large) for each kernel or application, with 200 empirical evaluations per data size (a total of 600 empirical observations per benchmark). The observations are selected by YTOPT [31], however we additionally collect the IR of each configured bench mark for use with NCS.

For fairness, we present the exact same training data in its most appropriate form to compared techniques that can benefit from prior data from related tasks. Each search begins without any data relating to the target task, representing a fresh start to optimize the related performance objective. All evaluations are performed on two new tasks, an interpolated size (Small-Medium, or SM, between small and medium source tasks) and an extrapolated size (Extra-Large, or XL, larger than the large source task) on all benchmarks. To minimize variance in system performance and randomness of tuning techniques, we report all empirical evaluations as the average of three executions and repeat all tuning runs with three separate randomized seeds.

For each new task, we permit a maximum of 300 empirical evaluations or 24 hours to be spent tuning, whichever comes first.

## 5.4 Training and Inference

Training the GCTLA model requires no more than a few seconds, and sampling time is trivial. We trained all five NCS models with configurations generated by YTOPT for small, medium, and large input sizes. Due to the small size of the networks and limited training data, training the NCS ensemble in this way takes about four minutes for each benchmark; this forms the majority of the overhead of our technique. We have used the Adam Optimizer for training, and each model was trained for 30 epochs. Similar to GCTLA, NCS’s inference time for all sampled IRs is trivial. Including to the overhead of compiling, the training+inference time of our technique “costs” no more than ten evaluations on any benchmark.

## 5.5 Compared Techniques

We compare our techniques to the GCTLA without NCS in addition to state-of-the-art autotuners BLISS [26], OpenTuner [3], and GPTune [11, 36].

BLISS is a meta-technique that smoothly interpolates between lightweight surrogate models to efectively identify the best learning strategy during the autotuning process.

OpenTuner is an autotuning framework that supports over twentyfive separate search algorithms and can be extended to support additional search algorithms. Due to the excessive cost of demonstrating the performance of all of OpenTuner’s techniques, we utilize the exhaustive tuning data for the Syr2k SM and XL problems to determine the best-performing technique. Empirically, we find the AUCBanditMetaTechniqueB, ga-PX and NormalGreedyMutation10 techniques to be highly competitive with one another in an initial exploration of available algorithms, and elect to utilize the Normal-GreedyMutation10 search technique for the remaining benchmarks. Henceforth, the NormalGreedyMutation10 algorithm executed by OpenTuner is referred to as “OpenTuner” for brevity.

GPTune is a machine-learning multi-task transfer autotuning technique built upon Gaussian Processes. As a transfer-enabled autotuning technique, GPTune is capable of reusing the same training information utilized in our own searches and is accordingly set up to utilize this same data. The authors of GPTune also contributed the GPTuneCrowd [11] database strategy to facilitate knowledge sharing and empower the community with collective autotuning eforts.

In order to clearly demonstrate the capabilities of NCS, we perform all searches without NCS and reorder the searches afterwards. In addition to clearly displaying the advantages and shortcomings of NCS, this allows us to apply NCS to iterative techniques that cannot generate many samples at once. This analysis allows us to determine if other generative samplers than GCTLA can be expected to perform well with NCS, however we are not aware of any highly suitable alternatives at the time of writing.

## 5.6 Experimental Results

Tables 4 and 5 demonstrate the general results from our experiments.   
We also provide a visual representation of the results in Figure 3.

5.6.1 Immediate Eficacy. We consider the first one hundred evaluations to be the “immediate-term” autotuning that is most consistent with prior work.

Table 2: Parameters used to tune Polybench Kernels. Values within brackets indicate the options available for an independent parameter, and a list of brackets represents multiple independent parameters.
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>3MM</td><td rowspan=1 colspan=1>Heat3d</td><td rowspan=1 colspan=1>Syr2k</td></tr><tr><td rowspan=1 colspan=1>Tile Sizes</td><td rowspan=1 colspan=1>[4-2048], [4-2048],[4-2048]</td><td rowspan=1 colspan=1>[4-128], [4-2048],[4-256]</td><td rowspan=1 colspan=1>[4-128], [4-2048],[4-256]</td></tr><tr><td rowspan=1 colspan=1>Loop Interchange</td><td rowspan=1 colspan=1>[Yes, N/A]</td><td rowspan=1 colspan=1>[Yes, N/A]</td><td rowspan=1 colspan=1>[Yes, N/A]</td></tr><tr><td rowspan=1 colspan=1>Array Packing</td><td rowspan=1 colspan=1>[Yes, N/A] × 6</td><td rowspan=1 colspan=1>[Yes, N/A] × 2</td><td rowspan=1 colspan=1>[Yes, N/A] × 2</td></tr></table>

Table 3: Parameters used to tune ECP proxy applications.
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>AMG</td><td rowspan=1 colspan=1>RSBench</td><td rowspan=1 colspan=2>SW4Lite</td></tr><tr><td rowspan=1 colspan=1>Tile Sizes</td><td rowspan=1 colspan=1>[10-200], [2-256],[2-256], [10-200]</td><td rowspan=1 colspan=1>[2-256], [2-256]</td><td rowspan=1 colspan=2>1</td></tr><tr><td rowspan=1 colspan=1>Optional Parameters</td><td rowspan=1 colspan=1>Parallel For</td><td rowspan=1 colspan=1>Parallel For</td><td rowspan=1 colspan=2>Parallel For, Nowait,MPI_Barrier</td></tr><tr><td rowspan=1 colspan=1>Parallel For Schedule</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[100-2000], [10-200]</td><td rowspan=1 colspan=2>[dynamic, static]</td></tr><tr><td rowspan=1 colspan=1>Unrolling Options</td><td rowspan=1 colspan=1>[unroll, N/A]</td><td rowspan=1 colspan=1>[unroll, N/A]</td><td rowspan=1 colspan=2>[unroll (6) 2, unroll,no-unroll]</td></tr><tr><td rowspan=1 colspan=1># Threads</td><td rowspan=1 colspan=1>[4-8]</td><td rowspan=1 colspan=1>[2-256]</td><td rowspan=1 colspan=2>[2-256]</td></tr><tr><td rowspan=1 colspan=1>KMP Affinity</td><td rowspan=1 colspan=1>[compact, scatter,balanced]</td><td rowspan=1 colspan=1>[compact, scatter,balanced, none,disabled, explicit]</td><td rowspan=1 colspan=2>一</td></tr><tr><td rowspan=1 colspan=1>OMP Proc Bind</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[close, spread, master]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>OMP Places</td><td rowspan=1 colspan=1>[core, threads, sockets]</td><td rowspan=1 colspan=1>[core, threads, sockets]</td><td rowspan=1 colspan=1>[core, threads, sockets]</td><td rowspan=1 colspan=1></td></tr></table>

Across Polybench/C benchmarks in Table 4, the GCTLA produces the best results of all search techniques. With the help of NCS, we can frequently access the optimal result from GCTLA within the first 100 evaluations, reducing the average number of trials from 201.67 to 16 on the SyR2K SM benchmark, 61.67% of the overall evaluation budget. Notably, within the first 100 trials, GC+NCS is reliably able to outperform most other autotuners’ best results within the full search duration. This demonstrates the capability of generative predictive autotuning to dramatically improve search performance.

5.6.2 Ultimate Eficacy. We also include the ultimate eficacy, or best result of each technique, to highlight the capabilities of our compared approaches within the full scope of its evaluations. We were surprised to find that iterative techniques were generally unable to exceed 200 evaluations within the 24 hour allotment, which GCTLA and NCS were able to utilize their full 300 evaluations across all experiments.

We find that the ECP applications generally provide much larger dividends for iterative techniques, as their active learning capabili ties are better suited to continue learning the complex performance relationships beyond the hints provided by source task training data. We can see that NCS generally still works well with GCTLA, however some benchmarks such as RSBench prove challenging for NCS as well, resulting in the worst of both worlds.

Despite these setbacks, we can see that when NCS is applied after-the-fact to iterative autotuners, it still reduces the number of trials needed to discover the optimum, which supports our technique’s applicability to other searches. We expect that this implies NCS is well-suited for usage with other generative autotuners than GCTLA, and may be able to provide value on these more challenging benchmarks to generative techniques that can make use of iterative feedback.

## 5.7 Exhaustive Searches: Syr2k

We utilize exhaustive data for performance on all configurations of the Syr2k benchmark to demonstrate the global efectiveness of tuning and full-duration efectiveness of tuning techniques. In Figures 4a and 4b, we see the global optimality achieved by diferent tuning techniques. Notably, most techniques are not capable of performing complete searches, denoted by diamonds at their final evaluation. For techniques utilizing GCTLA, this represents exhaustion of all unique samples that can be reasonably generated. Because the model is unlikely to produce any of these excluded results, they are considered inaccessible to the search techniques. Similar problems occur for OpenTuner, which automatically detects duplicate generations from its model and early-terminates searches that converge. It is possible that similar problems occur for GPTune, however due to the repeated training costs for refitting all GPs at each step of the search, we find that this training cost begins to surpass reasonable limitations as the search length extends past 100 total evaluations. This 100-evaluation limit falls within the previously published use cases for the framework, so while longer searches are theoretically possible, we find that the increasing overhead of refitting GPTune makes longer-term tuning eforts inadvisable. We observe less dificulties in long-term searches with BLISS, however its surrogate evaluations are never re-examined, meaning that it may not discover the actual best performance when mispredictions occur, even if given additional budget to pursue extra evaluations.

## 5.8 Details of NCS Efectiveness

As shown in Figure 5, the combination of GC+NCS grants the benefits of both techniques. The GCTLA proposes many high-quality configurations, but due to its unordered generation, finding the best configuration is a dubious process that requires attempting each generated configuration until resources are exhausted. With NCS, the highest-performing proposals from the GCTLA are moved to the front, generally within the first 100 evaluations, permitting a more limited autotuning expenditure to have high likelihood of extracting the best data available from the GCTLA.

![](images/7b828bb79f176e42ddee4c2907ca87e3ac630da615edf7cbc3011f4ad79f2a04.jpg)  
(a) Polybench/C SM target tasks

![](images/1abeb3616f9e8162ad204b850b3f6fe01150ff11bc9c747b3ae9e960039ca248.jpg)  
(b) Polybench/C XL target tasks

![](images/897212efa5dd35a1d42659775179df771bafb2f70eca22fbb31c875e7aa55936.jpg)  
(c) ECP SM target tasks

![](images/94e54ba94f0277945579932300b4c6c4c3f81417378cf49b680598136e51c2eb.jpg)  
(d) ECP XL target tasks  
Figure 3: Each line connects a search’s average best result (based on its average trial number) to its average best result observed within 100 evaluations; observing the highest normalized performance within the fewest trials (up and to the left) is optimal.

However, we saw that NCS does not always improve the evaluation order. One such example is depicted in Figure 6, where NCS prioritizes a large number of configurations with middling per formance before slowly honing in on the actual best-performing configurations. Despite the limited data used during training, we find that a few of the better results are still available within the first-100 immediate term tuning, and the performance gap between the search-optimum and NCS’s top-100 selections remains insignif icant. Unlike GCTLA, there isn’t a simple manner to predict how many evaluations GC+NCS require to produce a bounded-quality result, but we see that even when NCS is not as well-suited for its task, the greatest harm it inflicts upon a larger-scale search with GCTLA is relatively minor. Compared to the consistency of NCS dramatically improving the sorting order, we find that a successful search is more likely to be improved by NCS than degraded.

## 6 Related Work

Autotuning is a well-studied problem with many diferent approaches suitable to diferent tasks.

In spaces with unknown constraints such as memory availability for GPUs, tuning techniques such as BaCO [18] permit dynamically learning these limits separately from the performance relationship. Such techniques usually work best with larger data availability and longer tuning budgets than are the subjects of this work.

For real-time systems such as live databases, the variable demands imposed on the system either require separate modeling or continuous online updates. The latter is generally preferred in literature such as Zhang et al [34] as it requires less data to bootstrap and continuously adapts to conditions.

Certain autotuners expose their own hyperparameters to control the search process, which may require tuning for the autotuner itself to maximize performance as discussed in Bolet et al [9].

Prior studies [3, 26, 28, 32] have proposed numerous techniques for autotuning HPC applications. A large chunk of performance gains come from highly-specific compiler optimizations, such as those seen in LLVM and GCC. These compiler options often have to be given explicitly at compilation time or are expressed to the compiler within the source code to permit the most eficient machinecode generation. In addition to compiler optimizations, tweaking runtime parameters and knobs in HPC applications and systems allows users to extract more performance out of such setups.

![](images/bea33d5cb9f2d3ba61219e2a204ce009f0b4d563f717e552574e9b855bfba8e9.jpg)  
(a) SM target task

![](images/23f2930246ab88be79d0df6d9d7cf396623af9b6236ffd72865c68ccdcf43190.jpg)  
(b) XL target task  
Figure 4: SyR2K overall search optimality per-seed shown on a log-log scale. Ideally, searches converge at the lowest rank.

![](images/e524ef86314ee0c2a8eedbd31cd7453e05037a86eb78603c3b92c39db1b4a978.jpg)  
Figure 5: NCS generally moves high-quality evaluations for the Heat3D benchmark generated by GCTLA to the start of the search, improving autotuning performance.

Most autotuning tasks require optimizing single- or multi-objective non-linear functions, which incentivizes widespread use of statistical techniques. OpenTuner [3], for example, employs various search space optimization techniques such as the simplex-based Nelder-Mead [27], Torczon hill-climbers [12] to minimize a single-objective HPC autotuning task faster. More recently, Bayesian optimization has been used and adapted quite successfully to make autotuning in HPC more eficient [4, 24, 26, 35]. In spite of these significant advances, most of these tools often sufer from a high code execution overhead, which adversely impacts the usability and adoption of these tools.

![](images/d87272f82407f21bb49b6f1f490a6a5753ab2f99b988162b257fbc5828f85e15.jpg)  
Figure 6: NCS can harm searches by prioritizing lower-quality results. In this case for the SW4Lite benchmark, the majority of generations are high-quality, but NCS takes a significant number of trials to identify good performance.

A commonly proposed alternative, especially for tasks targeting source code, are tools and studies that employ machine learning (ML) or artificial intelligence (AI) [8, 13–15, 25, 29]. Most of these studies achieve state-of-the-art results in a number of optimization tasks and several of them propose new code representation techniques on top of LLVM IRs, making them usable across programming languages and programming models. However, most ML techniques must be trained specifically for each task at considerable cost, reducing their generalization capabilities while increasing the barrier to broader utilization within HPC. CPU simulator works such as DifTune [2, 25] can provide cycle estimations for straight line assembly via LLVM’s MCA tools, but exclude key structures such as loop behaviors and function calls. These techniques are also limited to runtime estimations only, where most autotuners can support any objective or even multi-objective tuning. Several autotuners [11, 31, 32, 35] utilize machine learning techniques such as Bayesian optimization to scale to available and relevant training data at the cost of making a number of decisions that are suboptimal early in the search. This increases the empirical cost of searches to enrich available information about the problem, which the advanced algorithms can use to recover the cost as time goes on with increased insight into the performance relationship.

Table 4: Speedups of Polybench/C Apps
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Target Size</td><td rowspan="2">Search Tool</td><td rowspan="2">(First 100, Best)</td><td rowspan="2">Average # Trials to Best</td></tr><tr><td>Speedup</td></tr><tr><td rowspan="7">SyR2K</td><td rowspan="7">SM</td><td>Default GCTLA</td><td>(1.000, 1.000) (3.645, 3.721)</td><td>1 201.67</td></tr><tr><td>GPTune</td><td>(3.042, 3.042)</td><td>77</td></tr><tr><td>BLISS</td><td>(3.101, 3.143)</td><td>59.33</td></tr><tr><td></td><td></td><td>69.33</td></tr><tr><td>OpenTuner</td><td>(3.582, 3.582)</td><td></td></tr><tr><td>GC+NCS (ours)</td><td>(3.645, 3.645)</td><td>16</td></tr><tr><td>GPTune+NCS</td><td>(3.027, 3.042)</td><td>39.33</td></tr><tr><td rowspan="8"></td><td>BLISS+NCS</td><td>(2.994, 3.143) (3.374, 3.582)</td><td>16.33 36.33</td></tr><tr><td>OpenTuner+NCS Default</td><td>(1.000, 1.000)</td><td>1</td></tr><tr><td>GCTLA</td><td>(1.177, 1.189)</td><td>245.33</td></tr><tr><td>GPTune</td><td>(1.037, 1.037)</td><td>57.33</td></tr><tr><td></td><td>(1.163, 1.163)</td><td>46.67</td></tr><tr><td>BLISS</td><td>(1.178, 1.185)</td><td>114.67</td></tr><tr><td>OpenTuner GC+NCS (ours)</td><td>(1.175, 1.189)</td><td>18</td></tr><tr><td>GPTune+NCS</td><td>(1.032, 1.037)</td><td>32.67</td></tr><tr><td rowspan="8">3MM SM</td><td>BLISS+NCS</td><td>(1.163, 1.163) (1.179, 1.185)</td><td>25 65.33</td></tr><tr><td>OpenTuner+NCS</td><td>(1.000, 1.000)</td><td></td></tr><tr><td>Default GCTLA</td><td>(5.458, 5.894)</td><td>1 150</td></tr><tr><td></td><td>(5.857, 5.881)</td><td>76</td></tr><tr><td>GPTune</td><td>(4.327, 4.566)</td><td>72.67</td></tr><tr><td>BLISS</td><td></td><td></td></tr><tr><td>OpenTuner</td><td>(4.664, 5.010)</td><td>85.67</td></tr><tr><td>GC+NCS (ours)</td><td>(5.445, 5.894) (5.424, 5.881)</td><td>95</td></tr><tr><td rowspan="7"></td><td>GPTune+NCS BLISS+NCS</td><td>(4.503, 4.566)</td><td>7.33 44.67</td></tr><tr><td></td><td>(5.010, 5.010)</td><td>36.67</td></tr><tr><td>OpenTuner+NCS</td><td>(1.000, 1.000)</td><td></td></tr><tr><td>Default</td><td>(38.196, 38.726)</td><td>1 183</td></tr><tr><td>GCTLA</td><td>(23.856, 23.856)</td><td>35.33</td></tr><tr><td>GPTune</td><td>(17.841, 32.314)</td><td>80</td></tr><tr><td>BLISS OpenTuner</td><td>(37.907, 38.348)</td><td>105.33</td></tr><tr><td rowspan="6"></td><td>GC+NCS (ours)</td><td>(38.315, 38.726)</td><td>106.67</td></tr><tr><td>GPTune+NCS</td><td>(23.459, 23.856)</td><td>7</td></tr><tr><td></td><td></td><td></td></tr><tr><td>BLISS+NCS</td><td>(31.194, 32.314)</td><td>114</td></tr><tr><td>OpenTuner+NCS</td><td>(38.348, 38.348)</td><td>66.67</td></tr><tr><td>Default</td><td>(1.000, 1.000)</td><td>1</td></tr><tr><td rowspan="7">Heat3D</td><td rowspan="8">SM</td><td>GCTLA</td><td>(2.077, 2.357)</td><td>216.33</td></tr><tr><td>GPTune</td><td>(2.215, 2.215)</td><td>66</td></tr><tr><td>BLISS</td><td>(2.233, 2.233)</td><td>38.33</td></tr><tr><td>OpenTuner</td><td>(2.091, 2.091)</td><td>33.67</td></tr><tr><td>GC+NCS (ours)</td><td>(2.357, 2.357)</td><td>33</td></tr><tr><td>GPTune+NCS</td><td>(2.215, 2.215)</td><td>11.33</td></tr><tr><td>BLISS+NCS</td><td>(2.233, 2.233)</td><td>27</td></tr><tr><td>OpenTuner+NCS Default</td><td>(2.091, 2.091)</td><td>40.33</td></tr><tr><td rowspan="7">Heat3D</td><td></td><td>(1.000, 1.000) (3.341, 3.344)</td><td>1</td><td>123.67</td></tr><tr><td>GPTune</td><td>GCTLA (3.181, 3.181)</td><td></td><td>59.67</td></tr><tr><td>BLISS</td><td>(3.329, 3.338)</td><td></td><td>168</td></tr><tr><td>XL OpenTuner</td><td>(3.314, 3.314)</td><td></td><td>88</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>GC+NCS (ours) GPTune+NCS</td><td>(3.339, 3.344)</td><td></td><td>149.67</td></tr><tr><td></td><td>(3.181, 3.181)</td><td></td><td>6</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="10"></td><td></td><td></td><td></td><td></td></tr><tr><td>BLISS+NCS</td><td>(3.336, 3.338)</td><td></td><td>37</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>OpenTuner+NCS</td><td>(3.314, 3.314)</td><td></td><td>5</td></tr></table>

GCTLA [24] establishes the basic capability of probability models to represent transferrable knowledge in autotuning. Instead of iteratively exploring and learning about the performance relationship, this technique attempts to directly sample high-performing configurations in the search space. This approach requires far less data than ML- and AI- based autotuners, reducing the cost of specific training while still delivering high-quality results within a small number of empirical evaluations. Another compelling characteristic of probabilistic transfer autotuning is the ability to mathematically model the number of useful evaluations to perform before collecting any evaluations in the transfer domain. This “budget” can be used to determine the suitability of probabilistic sampling and the likely cost to utilize the technique prior to executing any evaluations.

Table 5: Speedups of ECP Apps
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Target Size</td><td rowspan="2">Search Tool</td><td rowspan="2">(First 100, Best)</td><td rowspan="2">Average # Trials to Best</td></tr><tr><td>Speedup</td></tr><tr><td rowspan="7">AMG</td><td rowspan="7">SM</td><td>Default GCTLA</td><td>(1.000, 1.000) (0.668, 0.683)</td><td>1 145</td></tr><tr><td></td><td>(0.978, 0.978)</td><td></td></tr><tr><td>GPTune</td><td></td><td>84</td></tr><tr><td>BLISS</td><td>(0.602, 0.608)</td><td>123.67</td></tr><tr><td>OpenTuner</td><td>(1.509, 1.509)</td><td>29.67</td></tr><tr><td>GC+NCS (ours)</td><td>(0.679, 0.683)</td><td>80.67</td></tr><tr><td>GPTune+NCS</td><td>(0.978, 0.978)</td><td>54</td></tr><tr><td rowspan="8"></td><td>BLISS+NCS</td><td>(0.607, 0.608)</td><td>69</td></tr><tr><td>OpenTuner+NCS</td><td>(1.507, 1.509)</td><td>89.67</td></tr><tr><td>Default</td><td>(1.000, 1.000)</td><td>1</td></tr><tr><td>GCTLA</td><td>(0.928, 0.930)</td><td>226.67</td></tr><tr><td>GPTune</td><td>(0.887, 0.887)</td><td>24.33</td></tr><tr><td>BLISS</td><td>(0.893, 0.914)</td><td>186.33</td></tr><tr><td>OpenTuner</td><td>(1.046, 1.046) (0.927, 0.930)</td><td>41.33 155.33</td></tr><tr><td>GC+NCS (ours) GPTune+NCS</td><td>(0.887, 0.887)</td><td>27.67</td></tr><tr><td rowspan="8"></td><td>BLISS+NCS</td><td>(0.912, 0.914)</td><td>85.67</td></tr><tr><td>OpenTuner+NCS</td><td>(1.046, 1.046)</td><td>52.33</td></tr><tr><td>Default</td><td>(1.000, 1.000)</td><td>1</td></tr><tr><td>GCTLA</td><td>(0.749, 0.749) (1.998, 3.290)</td><td>51.33</td></tr><tr><td>GPTune</td><td>(0.916, 1.079)</td><td>81.33</td></tr><tr><td>BLISS</td><td></td><td>146</td></tr><tr><td>OpenTuner</td><td>(1.100, 1.132)</td><td>131.33</td></tr><tr><td>GC+NCS (ours)</td><td>(0.729, 0.749)</td><td>162.33</td></tr><tr><td rowspan="8"></td><td>GPTune+NCS</td><td>(3.290, 3.290)</td><td>40.67</td></tr><tr><td>BLISS+NCS</td><td>(0.894, 1.079)</td><td>140.33</td></tr><tr><td>OpenTuner+NCS</td><td>(1.132, 1.132)</td><td>87.67</td></tr><tr><td>Default</td><td>(1.000, 1.000)</td><td>1</td></tr><tr><td>GCTLA</td><td>(0.528, 0.529)</td><td>129.67</td></tr><tr><td>GPTune</td><td>(0.765, 0.765)</td><td>71.67</td></tr><tr><td>BLISS</td><td>(1.002, 1.052)</td><td>91</td></tr><tr><td>OpenTuner</td><td>(0.826, 0.826)</td><td>72</td></tr><tr><td rowspan="6"></td><td>GC+NCS (ours)</td><td>(0.525, 0.529)</td><td>165</td></tr><tr><td>GPTune+NCS</td><td>(0.759, 0.765)</td><td>52</td></tr><tr><td>BLISS+NCS</td><td>(1.052, 1.052)</td><td>54.67</td></tr><tr><td>OpenTuner+NCS</td><td>(0.826, 0.826)</td><td>18.67</td></tr><tr><td>Default</td><td>(1.000, 1.000)</td><td>1</td></tr><tr><td rowspan="8">SM</td><td>GCTLA</td><td>(0.956, 0.964)</td><td>210</td></tr><tr><td>GPTune</td><td>(0.929, 0.929)</td><td>46.67</td></tr><tr><td>BLISS</td><td>(0.778, 0.778)</td><td>31</td></tr><tr><td>OpenTuner</td><td>(0.945, 0.945)</td><td>73</td></tr><tr><td>GC+NCS (ours)</td><td>(0.956, 0.964)</td><td>28.33</td></tr><tr><td>GPTune+NCS</td><td>(0.929, 0.929)</td><td>64.67</td></tr><tr><td>BLISS+NCS</td><td>(0.768, 0.778)</td><td>137</td></tr><tr><td>OpenTuner+NCS Default</td><td>(0.945, 0.945)</td><td>23</td></tr><tr><td rowspan="8">SW4Lite</td><td></td><td>(1.000, 1.000)</td><td>1</td></tr><tr><td>GCTLA GPTune</td><td>(1.039, 1.040)</td><td>194.33</td></tr><tr><td>BLISS XL</td><td>(1.034, 1.034) (0.943, 0.944)</td><td>62.33 154.33</td></tr><tr><td></td><td></td><td></td></tr><tr><td>OpenTuner</td><td>(1.032, 1.032)</td><td>40.33</td></tr><tr><td>GC+NCS (ours)</td><td>(1.039, 1.040)</td><td>64.33</td></tr><tr><td>GPTune+NCS</td><td>(1.034, 1.034)</td><td>47.67</td></tr><tr><td>BLISS+NCS</td><td>(0.942, 0.944)</td><td>167</td></tr><tr><td></td><td>OpenTuner+NCS</td><td>(1.032, 1.032)</td><td>53.67</td></tr></table>

## 7 Conclusions

In this work, we create a framework for transfer-learning-based autotuning using LLVM IRs. Our machine learning ensemble, NCS, is capable of prioritizing high-quality evaluations based on predicted performance from IRs. When combined with the GCTLA probabilistic model to restrict consideration to a high-performing candidate sample, our method significantly reduces the number of necessary empirical evaluations in the search space. Our novel combination of search space reduction with lower-fidelity discrim inative modeling enables us to rapidly optimize the performance objective, outperforming existing autotuning techniques in fewshot scenarios. We demonstrate that GC+NCS reduces the average number of evaluations needed to locate the search optimum 27.85% of the overall search budget.

NCS can be orthogonally applied to any non-iterative autotuning technique and encourages further research on probabilistic techniques and other supportive models. Moreover, our method maintains competitive performance even in longer search durations, ofering greater reach for probabilistic techniques to scale as data-eficient alternatives to traditional machine-learning-based autotuners such as Bayesian optimization and GPTune.

While our results do not indicate current failures of this kind, we know that CNNs can struggle to capture long-range dependencies or sequential relationships that exist within IR code structures, due to their localized receptive fields. Future work may explore usage of graph neural network and transformer-based architectures to better capture global execution patterns that influence performance.

## References

[1] [n. d.]. SOLLVE LLVM Branch: pragma-clang-loop. https://github.com/SOLLVE/llvm-project.git. https://github.com/SOLLVE/llvmproject/tree/a48e6204b7f1f0d6ea84ab2f2f4c04e15a4338fb

[2] Andreas Abel. 2022. DifTune Revisited: A Simple Baseline for Evaluating Learned llvm-mca Parameters. In Machine Learning for Computer Architecture and Systems 2022. https://openreview.net/forum?id=dw4evoj6AE

[3] Jason Ansel, Shoaib Kamil, Kalyan Veeramachaneni, Jonathan Ragan-Kelley, Jefrey Bosboom, Una-May O’Reilly, and Saman Amarasinghe. 2014. OpenTuner: An Extensible Framework for Program Autotuning. In 2014 23rd Int. Conf. on Parallel Architecture and Compilation Techniques (PACT 2014). IEEE, Edmonton, AB, Canada, 303—-316.

[4] Prasanna Balaprakash, Jack Dongarra, Todd Gamblin, Mary Hall, Jefrey K Hollingsworth, Boyana Norris, and Richard Vuduc. 2018. Autotuning in high performance computing applications. Proc. IEEE 106, 11 (2018), 2068–2083.

[5] Francesco Barchi, Gianvito Urgese, Enrico Macii, and Andrea Acquaviva. 2019. Code mapping in heterogeneous platforms using deep learning and llvm-ir. In Proceedings ofthe 56th Annual Design Automation Conference 2019. 1–6.

[6] Babak Behzad, Surendra Byna, Prabhat, and Marc Snir. 2019. Optimizing i/o performance of hpc applications with autotuning. ACM Transactions on Parallel Computing (TOPC) 5, 4 (2019), 1–27.

[7] Zhendong Bei, Zhibin Yu, Huiling Zhang, Wen Xiong, Chengzhong Xu, Lieven Eeckhout, and Shengzhong Feng. 2015. RFHOC: A random-forest approach to auto-tuning hadoop’s configuration. IEEE Transactions on Parallel and Distributed Systems 27, 5 (2015), 1470–1483.

[8] Tal Ben-Nun, Alice Shoshana Jakobovits, and Torsten Hoefler. 2018. Neural code comprehension: A learnable representation of code semantics. Advances in neural information processing systems 31 (2018).

[9] Gregory Bolet, Giorgis Georgakoudis, Konstantinos Parasyris, Kirk W. Cameron, David Beckingsale, and Todd Gamblin. 2024. An Exploration of Global Optimization Strategies for Autotuning OpenMP-based Codes. In 2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). 741–750. doi:10.1109/IPDPSW63119.2024.00138

[10] Lorenzo Brigato and Luca Iocchi. 2021. A close look at deep learning with small data. In 2020 25th international conference on pattern recognition (ICPR). IEEE, 2490–2497.

[11] Younghyun Cho, James W. Demmel, Jacob King, Xiaoye S. Li, Yang Liu, and Hengrui Luo. 2023. Harnessing the Crowd for Autotuning High-Performance Computing Applications. In 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS). 635–645. doi:10.1109/IPDPS54959.2023.00069

[12] K Cooper, Timothy J Harvey, Devika Subramanian, and Linda Torczon. 2002. Compilation order matters. Technical Report (2002).

[13] Chris Cummins, Zacharias V Fisches, Tal Ben-Nun, Torsten Hoefler, Michael FP O’Boyle, and Hugh Leather. 2021. Programl: A graph-based program representation for data flow analysis and compiler optimizations. In International Conference on Machine Learning. PMLR, 2244–2253.

[14] Akash Dutta, Jordi Alcaraz, Ali TehraniJamsaz, Eduardo Cesar, Anna Sikora, and Ali Jannesari. 2023. Performance Optimization using Multimodal Modeling and Heterogeneous GNN. In Proceedings ofthe 32nd International Symposium on High-Performance Parallel and Distributed Computing (Orlando, FL, USA) (HPDC ’23). Association for Computing Machinery, New York, NY, USA, 45–57. doi:10.1145/3588195.3592984

[15] Akash Dutta and Ali Jannesari. 2024. Mirencoder: Multi-modal ir-based pretrained embeddings for performance optimizations. In Proceedings of the 2024 International Conference on Parallel Architectures and Compilation Techniques. 156–167.

[16] Thomas L Falch and Anne C Elster. 2017. Machine learning-based auto-tuning for enhanced performance portability of OpenCL applications. Concurrency and Computation: Practice and Experience 29, 8 (2017), e4029.

[17] Tobias Grosser, Armin Grösslinger, and Christian Lengauer. 2012. Polly – Performing Polyhedral Optimizations on a Low-Level Intermediate Representation. Parallel Processing Letters 22, 04 (2012). http://polly.llvm.org

[18] Erik Orm Hellsten, Artur Souza, Johannes Lenfers, Rubens Lacouture, Olivia Hsu, Adel Ejjeh, Fredrik Kjolstad, Michel Steuwer, Kunle Olukotun, and Luigi Nardi. 2024. BaCO: A Fast and Portable Bayesian Compiler Optimization Framework. In Proceedings ofthe 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 4 (Vancouver, BC, Canada) (ASPLOS ’23). Association for Computing Machinery, New York, NY, USA, 19–42. doi:10.1145/3623278.3624770

[19] John L. Hennessy and David A. Patterson. 2019. A new golden age for computer architecture. Commun. ACM 62, 2 (jan 2019), 48–60. doi:10.1145/3282307

[20] Lester Kalms, Tim Hebbeler, and Diana Göhringer. 2018. Automatic opencl code generation from llvm-ir using polyhedral optimization. In Proceedings ofthe 9th Workshop and 7th Workshop on Parallel Programming and RunTime Management Techniques for Manycore Architectures and Design Tools and Architectures for Multicore Embedded Computing Platforms. 45–50.

[21] Kirthevasan Kandasamy, Gautam Dasarathy, Junier B Oliva, Jef Schneider, and Barnabás Póczos. 2016. Gaussian process bandit optimisation with multi-fidelity evaluations. Advances in neural information processing systems 29 (2016).

[22] Kirthevasan Kandasamy, Gautam Dasarathy, Jef Schneider, and Barnabás Póczos. 2017. Multi-fidelity bayesian optimisation with continuous approximations. In International conference on machine learning. PMLR, 1799–1808.

[23] Sinno Jialin Pan and Qiang Yang. 2010. A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering 22, 10 (2010), 1345–1359. doi:10. 1109/TKDE.2009.191

[24] Thomas Randall, Jaehoon Koo, Brice Videau, Michael Kruse, Xingfu Wu, Paul Hovland, Mary Hall, Rong Ge, and Prasanna Balaprakash. 2023. Transfer-learningbased Autotuning using Gaussian Copula. In Proceedings ofthe 37th ACM International Conference on Supercomputing (Orlando, FL, USA) (ICS ’23). Association for Computing Machinery, New York, NY, USA, 37–49. doi:10.1145/3577193.3593712

[25] Alex Renda, Yishen Chen, Charith Mendis, and Michael Carbin. 2020. DifTune: Optimizing CPU Simulator Parameters with Learned Diferentiable Surrogates. In 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). 442–455. doi:10.1109/MICRO50266.2020.00045

[26] Rohan Basu Roy, Tirthak Patel, Vijay Gadepally, and Devesh Tiwari. June 20–25, 2021. Bliss: Auto-tuning Complex Applications using a Pool of Diverse Lightweight Learning Models. In Proceedings ofthe 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementationg (PLDI’21). Association for Computing Machinery, New York, NY, USA, 1280–1295.

[27] Saša Singer and John Nelder. 2009. Nelder-mead algorithm. Scholarpedia 4, 7 (2009), 2928.

[28] Cristian Tapus, I-Hsin Chung, and Jefrey K Hollingsworth. 2002. Active harmony: Towards automated performance tuning. In SC’02: Proceedings ofthe 2002 ACM/IEEE Conference on Supercomputing. IEEE, 44–44.

[29] Ali TehraniJamsaz, Mihail Popov, Akash Dutta, Emmanuelle Saillard, and Ali Jannesari. 2022. Learning intermediate representations using graph neural networks for numa and prefetchers optimization. In 2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 1206–1216.

[30] S VenkataKeerthy, Rohit Aggarwal, Shalini Jain, Maunendra Sankar Desarkar, Ramakrishna Upadrasta, and YN Srikant. 2020. Ir2vec: Llvm ir based scalable program embeddings. ACM Transactions on Architecture and Code Optimization (TACO) 17, 4 (2020), 1–27.

[31] Xingfu Wu, Prasanna Balaprakash, Michael Kruse, Jaehoon Koo, Brice Videau, Paul Hovland, Valerie Taylor, Brad Geltz, Siddhartha Jana, and Mary Hall. 2025. ytopt: Autotuning Scientific Applications for Energy Eficiency at Large Scales. Concurrency and Computation: Practice and Experience 37, 1 (2025), e8322. doi:10. 1002/cpe.8322 arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpe.8322

[32] Xingfu Wu, Michael Kruse, Prasanna Balaprakash, Hal Finkel, Paul Hovland, Valerie Taylor, and Mary Hall. 2022. Autotuning polybench benchmarks with llvm clang/polly loop optimization pragmas using bayesian optimization. Concurrency and Computation: Practice and Experience 34, 20 (2022), e6683.

[33] Tomofumi Yuki and Louis-Noel Pouchet. 2016. PolyBench 4.2. Retrieved March 11, 2021 from https://sourceforge.net/projects/polybench/

[34] Xinyi Zhang, Hong Wu, Yang Li, Jian Tan, Feifei Li, and Bin Cui. 2022. Towards Dynamic and Safe Configuration Tuning for Cloud Databases. In Proceedings ofthe 2022 International Conference on Management ofData (Philadelphia, PA, USA) (SIGMOD ’22). Association for Computing Machinery, New York, NY, USA, 631–645. doi:10.1145/3514221.3526176

[35] Xinran Zhu, Yang Liu, Pieter Ghysels, David Bindel, and Xiaoye S Li. 2022. GP-TuneBand: Multi-task and multi-fidelity autotuning for large-scale high performance computing applications. In Proceedings ofthe 2022 SIAM Conference on

[36] Xinran Zhu, Yang Liu, Pieter Ghysels, David Bindel, and Xiaoye S. Li. 2022. GPTuneBand: Multi-task and Multi-fidelity Autotuning for Large-scale High Performance Computing Applications. SIAM Conference on Parallel Processing for Scientific Computing, 1–13. doi:10.1137/1.9781611977141.1 arXiv:https://epubs.siam.org/doi/pdf/10.1137/1.9781611977141.1

Parallel Processing for Scientific Computing. SIAM, 1–13.