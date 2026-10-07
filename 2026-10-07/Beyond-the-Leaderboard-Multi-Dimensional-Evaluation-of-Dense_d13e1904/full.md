# Beyond the Leaderboard: Multi-Dimensional Evaluation of Dense and Mixture-of-Experts Models for Automated Program Repair

Anvi Kalpesh Shah

National Institute of Technology, Calicut

anvi b230027cs@nitc.ac.in

Umamaheswara Sharma B

National Institute of Technology, Calicut

busharma@nitc.ac.in

Abstract—Automated Program Repair (APR) systems based on language models are commonly evaluated by whether generated patches pass the associated test suite. However, functional correctness alone can overlook differences in maintainability, security, and computational cost that affect the practical use of generated patches in real-world software engineering. This paper presents a multi-dimensional evaluation framework for APR that combines functional correctness, maintainability, security, and generation efficiency into a Weighted Quality Index (QI) inspired by the ISO/IEC 25010 software product quality model. We evaluate four code-specialized language models: three dense Qwen2.5- Coder models at 3B, 7B, and 14B parameters, and the 16Bparameter DeepSeek-Coder-V2-Lite Mixture-of-Experts (MoE) model with 2.4B active parameters. All models are evaluated locally on identical hardware to control for infrastructure-related differences in efficiency, using 40 bugs from QuixBugs and 90 bugs from Defects4J. Our results show that model rankings vary across the weighting schemes, highlighting the sensitivity of APR evaluation to the relative importance assigned to different quality dimensions. The MoE model achieves functional correctness that has no statistically significant difference (McNemar’s exact test) from both the 7B and 14B dense models despite using 3–6 times fewer active parameters, while the three dense models show a statistically significant scaling trend among themselves. These results suggest that active parameter count can provide a more informative lens than total parameter count for sparse code models, and that single-metric evaluation of APR quality can hide this kind of trade-off.

Index Terms—Automated Program Repair, Language Models, Mixture-of-Experts, Software Quality, Functional Correctness, Maintainability, Security, Generation Efficiency, Quality Index.

## I. INTRODUCTION

Automated Program Repair (APR) aims to automatically generate patches for software defects [1], reducing the effort required to diagnose and fix bugs. A common way of reporting APR performance is as a single number: the fraction of bugs for which a generated patch passes the associated test suite. While this provides a useful and reproducible measure of functional correctness, it does not capture the broader quality of the resulting patch. A patch that passes its tests but is unnecessarily complex, introduces a statically detectable security weakness, or requires substantially more computation to generate may be a worse engineering artifact than one that is simpler, safer, and more efficient—even when both are equally “correct” according to the test suite.

At the same time, the rise of sparse Mixture-of-Experts (MoE) architectures raises a related evaluation question: what does it mean to compare a “small” and a “large” model? A MoE model may have a large total parameter count but activate only a fraction of its parameters per token, making total parameter count an imperfect proxy for inference cost. Similarly, comparing models evaluated on different infrastructure—for example, a locally hosted model versus an APIhosted model—can make efficiency differences difficult to attribute solely to the models themselves. These considerations suggest that both parameter usage and evaluation infrastructure should be accounted for when examining efficiency across model architectures.

This paper addresses both issues together. We construct a Weighted Quality Index (QI) inspired by the ISO/IEC 25010 systems and software quality model [2], which combines functional correctness with three characteristics ISO/IEC 25010 identifies as part of product quality: maintainability, security, and performance efficiency. We evaluate every model entirely on local hardware with an identical inference stack, removing the infrastructure confound, and we report both total and active parameter counts for every model so that architecture-driven efficiency claims can be distinguished from simple scale.

We ask two research questions:

RQ1 (the quality framework). Does evaluating APR patches through a multi-dimensional, ISO/IEC 25010-inspired Quality Index reveal information that functional-correctness-only evaluation misses, and how sensitive is the resulting ranking to the relative weighting of the four dimensions?

RQ2 (dense vs. sparse models). Within this framework, does a sparse (MoE) code model achieve patch quality comparable to dense models while using substantially fewer active parameters, and how does functional correctness change as dense models scale in parameter count?

Our contributions are threefold: (1) a reusable, ISO/IEC 25010-inspired multi-dimensional Quality Index methodology for APR evaluation that combines functional correctness, maintainability, security, and generation efficiency under configurable weighting schemes; (2) a fully local, infrastructurecontrolled experimental evaluation across QuixBugs [3] and Defects4J [4], covering 130 bugs and four code-specialized language models spanning dense and sparse architectures; and (3) an empirical analysis addressing both research questions, including statistical significance testing of the correctness comparisons and analysis of ranking sensitivity across QI weighting schemes.

## II. RELATED WORK

APR Benchmarks. QuixBugs provides small, algorithmic programs with known defects and is available in both Python and Java. Defects4J provides real-world bugs mined from open-source Java projects, together with tests that reproduce the corresponding defects. In our evaluation, we use 90 Defects4J bugs selected from candidates whose source-level fixes modify one to three non-test lines, focusing on relatively localized repairs while keeping the prompted code manageable for local inference. QuixBugs has been used to evaluate both traditional repair tools [5] and LLMs [6]

Software Quality Models. ISO/IEC 25010:2023 defines a software product quality model comprising nine quality characteristics [2], including functional suitability, performance efficiency, security, and maintainability. Our Quality Index draws on four of these characteristics, operationalizing them for the evaluation of generated APR patches. Rather than treating the quality model as a direct measurement standard, we use it as a basis for structuring multiple dimensions of patch quality within a configurable composite index.

Dense and Sparse Code Models. Qwen2.5-Coder [7] is a family of dense, code-specialized models with instructiontuned variants spanning multiple parameter scales. DeepSeek-Coder-V2 [8] uses a Mixture-of-Experts architecture in which only a subset of parameters is active for each token; the Lite variant contains 16B total parameters with approximately 2.4B active parameters. Our study applies this dense-versus-sparse comparison specifically to APR and evaluates the resulting models across multiple dimensions of patch quality.

## III. METHODOLOGY

## A. Models

We evaluate four instruction-tuned, code-specialized language models, summarized in Table I. All four models are quantized to run within the 12 GB VRAM of an NVIDIA RTX 4080 Laptop GPU using Ollama [9] with context length =8192, on a machine with 32 GB of system RAM, with no CPU offloading. All models are evaluated on the same hardware using the same inference stack, helping to control for infrastructure-related differences when comparing generation efficiency.

## B. Benchmarks and Bug Selection

We evaluate 130 bugs from two benchmarks: the 40- bug Java subset of QuixBugs and 90 bugs from Defects4J. For Defects4J, we selected bugs whose fixes modify one to three non-test source lines. From 107 qualifying candidates, we initially selected 100 using a fixed random seed. Ten of these were subsequently excluded because their prompts could not be processed reliably, leaving 90 bugs for the final evaluation. The selected Defects4J bugs come from 16 of the 17 standard projects, with Chart excluded because its revisions could not be resolved in the Git mirror used for this study. For Defects4J, models are given the relevant method rather than the entire source file to keep the input manageable. The generated method is then inserted back into the original file before compilation and testing. This reduces the median prompt size by approximately 29 times. Functional correctness on Defects4J is evaluated using the benchmark’s trigger tests rather than the full project test suite. The implications of this choice are discussed in Section VI.

TABLE I  
MODELS EVALUATED
<table><tr><td>Model</td><td>Total</td><td>Active</td><td>Arch.</td></tr><tr><td>Qwen2.5-Coder-3B-Instruct</td><td>3B</td><td>3B</td><td>Dense</td></tr><tr><td>Qwen2.5-Coder-7B-Instruct</td><td>7B</td><td>7B</td><td>Dense</td></tr><tr><td>Qwen2.5-Coder-14B-Instruct</td><td>14B</td><td>14B</td><td>Dense</td></tr><tr><td>DeepSeek-Coder-V2-Lite-Instruct</td><td>16B</td><td>2.4B</td><td>MoE</td></tr></table>

## C. Quality Dimensions

Each patch is evaluated along four dimensions, with all resulting scores normalized to the range [0, 1].

Correctness. A patch receives a score of 1 if it compiles and passes the relevant tests, and 0 otherwise. For QuixBugs, we use the complete test suite for each bug; for Defects4J, we use the corresponding trigger tests. We use greedy decoding with temperature 0 and generate one patch per bug (pass@1 [10]) to improve reproducibility.

Maintainability. We measure cyclomatic complexity using Lizard [11] and compare the patched file with the official reference fix. The maintainability score is

$$
M = \exp \left( - 0 . 1 5 \operatorname* { m a x } \left( 0 , C C N _ { \mathrm { p a t c h } } - C C N _ { \mathrm { r e f } } \right) \right) ,
$$

giving a score of 1 when the patch is no more complex than the reference and decreasing as its complexity exceeds the reference. This dimension is evaluated for patches that compile; non-compiling patches receive a score of 0.

Security. We use SpotBugs 4.10.4 [12] with the Find-SecBugs 1.14.0 [13] plugin to identify security-related staticanalysis findings in the compiled patch and reference fix. Only findings in the SECURITY category are considered. The security score uses the same reference-relative decay function as maintainability:

$$
S = \exp \left( - 0 . 1 5 \operatorname* { m a x } \left( 0 , { F _ { \mathrm { p a t c h } } - F _ { \mathrm { r e f } } } \right) \right) ,
$$

where $F _ { \mathrm { p a t c h } }$ and $F _ { \mathrm { r e f } }$ denote the number of security findings in the patched and reference versions, respectively. Non-compiling patches receive a score of 0.

Efficiency. We measure generation speed in tokens per second using the inference engine’s internal timing, excluding one-time model-loading latency. For each model, efficiency is calculated as its mean generation speed across patches and normalized relative to the fastest model:

$$
E ( m ) = \frac { \mathrm { t o k / s } ( m ) } { \displaystyle \operatorname* { m a x } _ { m ^ { \prime } } \mathrm { t o k } / \mathrm { s } ( m ^ { \prime } ) } .
$$

Maintainability and security are scored only for patches that successfully compile, regardless of whether they pass the correctness test; patches that do not compile score 0 on both dimensions.

## D. Quality Index Construction

The QI for model m is computed as the weighted sum of its scores across the four quality dimensions:

$$
Q I ( m ) = \sum _ { d } w _ { d } s _ { d } ( m ) ,
$$

where $s _ { d } ( m )$ is the score of model m on dimension $d ,$ and the weights satisfy $\textstyle \sum _ { d } w _ { d } = 1$ . We report QI under three weighting schemes: equal $( w _ { d } = 0 . 2 5$ for all dimensions), correctness-priority $( w _ { \mathrm { c o r r e c t } } = 0 . 5$ , with the remaining 0.5 divided equally among the other three dimensions), and otherpriority $\mathrm { \Delta } ( w _ { \mathrm { c o r r e c t } } \ = \ 0 . 2$ , with the remaining 0.8 divided equally). Reporting multiple schemes serves as a robustness check rather than an attempt to identify a single “true” weighting.

We also observed an important methodological issue while constructing the index. An earlier version of the efficiency normalization used min-max scaling,

$$
E _ { \mathrm { m i n m a x } } ( m ) = \frac { \mathrm { t o k / s } ( m ) - v _ { \mathrm { m i n } } } { v _ { \mathrm { m a x } } - v _ { \mathrm { m i n } } } ,
$$

which assigns exactly 0 to the slowest model and 1 to the fastest regardless of the magnitude of the underlying difference. Because the other three dimensions were not renormalized in this way, this caused the efficiency dimension to mechanically dominate the equal-weighted QI: the model with the lowest correctness of the four briefly ranked first overall purely because min-max normalization gave efficiency a full [0,1] range that no other dimension had. We therefore report results using the ratio-to-fastest normalization described above, which preserves each model’s relative speed without forcing the slowest model to zero. This highlights the importance of normalization when combining different metrics.

## IV. RESULTS

Table II reports the four dimension scores for each model, aggregated over all 130 bugs. Correctness ranges from 0.154 for Qwen2.5-Coder-3B to 0.400 for Qwen2.5-Coder-14B. Maintainability and security are comparatively close across the four models, ranging from 0.856–0.897 and 0.878–0.923, respectively. Efficiency ranges from 0.240 for Qwen2.5-Coder-14B to 1.000 for Qwen2.5-Coder-3B, the fastest model under the ratio-to-fastest normalization. Fig. 1 visualizes these differences across the four dimensions.

TABLE II PER-MODEL SCORES ACROSS ALL FOUR QUALITY DIMENSIONS (n = 130 BUGS)
<table><tr><td>Model</td><td>Correct.</td><td>Maint.</td><td>Security</td><td>Effic.</td></tr><tr><td>Qwen2.5-Coder-3B</td><td>0.154</td><td>0.856</td><td>0.878</td><td>1.000</td></tr><tr><td>Qwen2.5-Coder-7B</td><td>0.308</td><td>0.897</td><td>0.923</td><td>0.480</td></tr><tr><td>Qwen2.5-Coder-14B</td><td>0.400</td><td>0.880</td><td>0.900</td><td>0.240</td></tr><tr><td>DeepSeek-Coder-V2-Lite</td><td>0.323</td><td>0.877</td><td>0.900</td><td>0.810</td></tr></table>

![](images/5c6091f8fe0134dd64b4a044b3b9860fe3de6b1b285d1bc252d8c08cd52b03f4.jpg)  
Fig. 1. Per-model scores across the four quality dimensions.

Table III and Fig. 2 report the Quality Index (QI) under the three weighting schemes. DeepSeek-Coder-V2-Lite ranks first under equal and correctness-priority weighting, while Qwen2.5-Coder-3B ranks first under other-priority weighting. These changes show that the overall ranking depends on how the different quality dimensions are weighted.

TABLE III  
QUALITY INDEX UNDER DIFFERENT WEIGHTING SCHEMES
<table><tr><td>Model</td><td>Equal</td><td>Corr.-Priority</td><td>Other-Priority</td></tr><tr><td>Qwen2.5-Coder-3B</td><td>0.722</td><td>0.533</td><td>0.760</td></tr><tr><td>Qwen2.5-Coder-7B</td><td>0.652</td><td>0.537</td><td>0.675</td></tr><tr><td>Qwen2.5-Coder-14B</td><td>0.605</td><td>0.537</td><td>0.619</td></tr><tr><td>DeepSeek-Coder-V2-Lite</td><td>0.727</td><td>0.593</td><td>0.754</td></tr></table>

![](images/38a92d441e73642dfc7cb56dd0ecaccd8f1252df953ec556559323b6e4464b08.jpg)  
Fig. 2. Quality Index scores under the three weighting schemes.

Fig. 3 compares generation speed with total and active parameter counts. Total parameter count is not a consistent predictor of generation speed: DeepSeek-Coder-V2-Lite, with 16B total parameters, generates faster than Qwen2.5-Coder-7B despite having more than twice as many total parameters. Active parameter count provides a more consistent explanation of the observed generation speeds, particularly for the sparse MoE model.

![](images/1db8c1081554ecd4a1b5f1b38073dc3f3a2e2b1defaad46f638b9d67fb33934b.jpg)  
Fig. 3. Generation speed versus total and active parameter count.

## A. Statistical Significance

Because all four models were evaluated on the same 130 bugs, we compare their correctness outcomes using McNemar’s exact test [14], which considers the bugs on which each pair of models disagrees. Table IV reports the resulting pairwise comparisons. Since six pairwise tests are performed, we additionally report Holm-adjusted p-values.

Among the three dense models, correctness increases significantly with model scale: Qwen2.5-Coder-14B outperforms Qwen2.5-Coder-7B, which outperforms Qwen2.5-Coder-3B, with all pairwise comparisons significant at $p < 0 . 0 5$ . This is consistent with prior findings that larger models tend to fix more bugs [1].

DeepSeek-Coder-V2-Lite is statistically indistinguishable from both Qwen2.5-Coder-7B $( p ~ = ~ 0 . 8 3 )$ and Qwen2.5- Coder-14B $( p = 0 . 0 5 3 )$ despite using only 2.4B active parameters. Thus, the sparse model achieves correctness comparable to substantially larger dense models in this evaluation.

TABLE IV  
PAIRWISE MCNEMAR’S EXACT TEST FOR CORRECTNESS
<table><tr><td>Comparison</td><td>p-value</td><td>Significant?</td></tr><tr><td>DeepSeek vs. Qwen2.5-Coder-14B</td><td>0.0525</td><td>No</td></tr><tr><td>DeepSeek vs. Qwen2.5-Coder-7B</td><td>0.8318</td><td>No</td></tr><tr><td>DeepSeek vs. Qwen2.5-Coder-3B</td><td>0.0001</td><td>Yes</td></tr><tr><td>Qwen2.5-Coder-14B vs. 7B</td><td>0.0118</td><td>Yes</td></tr><tr><td>Qwen2.5-Coder-14B vs. 3B</td><td>&lt; 0.0001</td><td>Yes</td></tr><tr><td> $\mathrm { Q w e n 2 . 5 - C o d e r - 7 B \ v s . \ 3 B }$ </td><td>&lt; 0.0001</td><td>Yes</td></tr></table>

Finally, we examined whether Defects4J correctness varies with the size of the source-level diff. Correctness is 7.3%, 9.2%, and 10.5% for bugs requiring one, two, and three changed lines, respectively. The differences are small, suggesting that repair difficulty in this benchmark is not explained simply by the number of changed source lines.

## V. DISCUSSION

RQ1. The QI framework highlights two aspects that a correctness-only evaluation would not capture. First, maintainability and security vary relatively little across the models and do not closely follow correctness. Thus, higher patch correctness does not necessarily correspond to higher maintainability or security scores in this evaluation. Second, the model ranking depends on how the four dimensions are weighted. DeepSeek-Coder-V2-Lite ranks first under equal and correctness-priority weighting, while Qwen2.5-Coder-3B ranks first under otherpriority weighting. This shows that the choice of weights can affect the overall assessment of model quality. We also observed that min-max normalization can disproportionately affect a composite score when metrics have different natural scales, as discussed in Section III-D.

RQ2. The MoE model achieves correctness that has no statistically significant difference from the largest dense model evaluated, while having a higher measured generation efficiency (0.810 vs. 0.240). The statistically significant differences among the three dense models also show an increase in correctness with parameter scale in this evaluation. Taken together, these results suggest that the MoE model can achieve correctness comparable to larger dense models while using substantially fewer active parameters. However, this should be interpreted as a comparison within the four models evaluated rather than as evidence that sparse models generally outperform dense models.

## VI. LIMITATIONS

Benchmark oracle strength. QuixBugs’ bundled test suites are comparatively small and may not detect subtle incorrectness in a generated patch [5]. For Defects4J, correctness is evaluated using trigger tests rather than the full project test suite. Therefore, a patch may pass the evaluated tests while affecting functionality that is not covered by those tests.

Sampling. We report pass@1 using greedy decoding rather than pass@k [10] with multiple samples. This choice improves reproducibility and keeps the evaluation tractable, but it may underestimate the performance that could be obtained through repeated sampling.

Benchmark coverage. Ten Defects4J candidates were excluded because their prompts could not be processed reliably within the available context constraints, reducing the selected sample from 100 to 90 bugs. This may limit the representativeness of the Defects4J results, particularly for bugs associated with larger source files.

Model coverage. The evaluation includes three dense models and one MoE model. Additional MoE models at different parameter scales would be needed to determine whether the observed results generalize across sparse architectures.

## VII. CONCLUSION

We presented a multi-dimensional, ISO/IEC 25010- inspired Quality Index for evaluating APR beyond functional correctness and applied it to three dense and one sparse codespecialized language model. Model rankings varied across

QI weighting schemes, showing that correctness alone does not capture all dimensions of patch quality. The evaluated MoE model achieved correctness statistically indistinguishable from the 7B and 14B dense models despite using only 2.4B active parameters, while correctness increased significantly across the three dense model scales. Future work will evaluate pass@k, additional MoE models, QI confidence intervals, and direct energy consumption. The code is available at https://github.com/anvishah1/MoE-vs-dense--model-for-APR

## REFERENCES

[1] C. S. Xia, Y. Wei, and L. Zhang, “Automated program repair in the era of large pre-trained language models,” in Proc. IEEE/ACM Int. Conf. Softw. Eng. (ICSE), Melbourne, Australia, 2023.

[2] ISO/IEC 25010:2023, Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product quality model, International Organization for Standardization, Geneva, Switzerland, Nov. 2023.

[3] D. Lin, J. Koppel, A. Chen, and A. Solar-Lezama, “QuixBugs: A multilingual program repair benchmark set based on the Quixey Challenge,” in Proc. Companion 2017 ACM SIGPLAN Int. Conf. Syst., Program., Lang., Appl.: Softw. Humanity (SPLASH Companion), Vancouver, BC, Canada, 2017, pp. 55–56, doi: 10.1145/3135932.3135941.

[4] R. Just, D. Jalali, and M. D. Ernst, “Defects4J: A database of existing faults to enable controlled testing studies for Java programs,” in Proc. Int. Symp. Softw. Testing Anal. (ISSTA), San Jose, CA, USA, Jul. 2014, pp. 437–440, doi: 10.1145/2610384.2628055.

[5] H. Ye et al., “A comprehensive study of automatic program repair on the QuixBugs benchmark,” arXiv:1805.03454, 2018. [Online]. Available: https://arxiv.org/abs/1805.03454

[6] J. A. Prenner and R. Robbes, “Automatic program repair with OpenAI’s Codex: Evaluating QuixBugs,” arXiv:2111.03922, 2021. [Online]. Available: https://arxiv.org/abs/2111.03922

[7] B. Hui et al., “Qwen2.5-Coder technical report,” arXiv:2409.12186, 2024. [Online]. Available: https://arxiv.org/abs/2409.12186

[8] DeepSeek-AI et al., “DeepSeek-Coder-V2: Breaking the barrier of closed-source models in code intelligence,” arXiv:2406.11931, 2024. [Online]. Available: https://arxiv.org/abs/2406.11931

[9] Ollama. [Online]. Available: https://ollama.com

[10] M. Chen et al., “Evaluating large language models trained on code,” arXiv:2107.03374, 2021. [Online]. Available: https://arxiv.org/abs/2107. 03374

[11] Lizard: An extensible cyclomatic complexity analyzer. GitHub repository. [Online]. Available: https://github.com/terryyin/lizard

[12] SpotBugs. [Online]. Available: https://github.com/spotbugs/spotbugs

[13] Find Security Bugs. [Online]. Available: https://github.com/ find-sec-bugs/find-sec-bugs

[14] Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, Jun. 1947, doi: 10.1007/BF02295996.