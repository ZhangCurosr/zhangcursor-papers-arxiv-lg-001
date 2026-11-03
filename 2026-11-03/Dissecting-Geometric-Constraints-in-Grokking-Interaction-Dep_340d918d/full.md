# Dissecting Geometric Constraints in Grokking: Interaction-Dependent Acceleration in Modular Arithmetic

Zachary A. Hampton<sup>1</sup>, Wasiq Khan<sup>2</sup> (SMIEEE, Ph. D), and Bilal Khan<sup>1</sup>, Ph.D.

<sup>1</sup>School of Computer Science and Engineering, California State University, San Bernardino, {Zachary.Hampton, Bilal.Khan}@csusb.edu

<sup>2</sup>School of Computer Science and Mathematics, Liverpool John Moores University, UK, w.khan@ljmu.ac.uk

## Abstract

Spherical transformer configurations can accelerate grokking through simultaneous modifications of several model properties. The present study investigated the contributions of these modifications in the published Sphere+Fourier configuration using 960 controlled modular-arithmetic runs. A complete factorial analysis varied residual projection, readout weight normalization, logit scale, and weight decay. Across four matched settings with weight decay, the successful-run mean crossing times without projection were 8-37 times those observed with projection. However, residual projection was insufficient to induce grokking without additional interventions. Various other unprojected configurations also generalized. Since projection altered the magnitude entering the readout, these comparisons could not separate geometry from output scale. A separate 100-run fixed-horizon study demonstrated faster first crossing for the complete configuration across five train/test partitions. Faster crossing did not necessarily correspond to greater persistence over extended training. Control experiments demonstrated rapid grokking without Fourier initialization. Task and depth sweeps also identified conditions in which the configuration failed to reach the criterion by 50K epochs. Accordingly, the observed acceleration depended on interactions among the components and did not extend uniformly across the tested tasks or depths.

## 1. Introduction and Related Work

Grokking describes delayed generalization in which a neural network first fits its training set without improving held-out performance beyond near-chance accuracy. Longer training can subsequently produce generalization (Power et al., 2022). Modular arithmetic provides a useful, controlled setting for investigating this delay since its finite operation table permits exact evaluation of every held-out input. Previous studies have examined Fourier-like circuits, competition between memorizing and generalizing subnetworks, kernel-to-feature transitions, and the influence of regularization and representation learning (Nanda et al., 2023; Merrill et al., 2023; Varma et al., 2023; Mohamadi et al., 2024; Liu et al., 2022a, 2022b).

Architectural and optimization choices can reduce the delay preceding generalization. Yıldırım (2026) introduced spherical residual streams with cosine unembedding and demonstrated accelerated grokking on cyclic modular tasks. The study also examined Fourier initialization, conventional normalization, learning rates, weight decay, uniform attention, multiplication, task alignment, and multiple seeds. Soft radial suppression accelerated generalization under a different target radius and intervention site (Tiwari et al., 2026). Normalization placement and optimization choices have also been shown to influence grokking (Singh et al., 2026). The present study extended these investigations through analysis of component interactions and their limitations.

The specific question addressed in the present study concerned the components responsible for acceleration in a spherical configuration. Sphere+Fourier combines residual projection, a normalized readout, a larger logit scale, Fourier initialization, and removal of weight decay. Comparison with the baseline therefore changes these properties simultaneously. Resolving the observed acceleration required systematic evaluation of their combinations to identify the conditions under which projection was beneficial or ineffective.

Accordingly, a complete $2 ^ { 4 }$ factorial experiment examined component interactions with Fourier initialization disabled. Separate initialization controls connected the no-Fourier projected configuration to the complete Sphere+Fourier configuration. The complete configuration for new train/test splits was then evaluated with independent-partition study, with the boundaries examined via the task and depth sweeps. Therefore, the component interactions, robustness, and limits of accelerated grokking were comprehensively addressed via the 960-run experiment.

## 2. Methods

## 2.1. Study design and outcomes

The factorial experiment was conducted to resolve interactions among individual components. A separate 100-run study compared the complete baseline and Sphere+Fourier configurations across data partitions, model seeds, and shallow depths. Additional controls examined normalization, logit scale, Fourier features, tasks, and depth. Token embeddings were compared at matched training ages in a separate 20-run study. Complete protocols, detailed tables, and an inventory of all 13 experimental groups and 960 runs are provided in the Supporting Information (Sections S1-S5; Supplementary Material). Held-out classification accuracy was evaluated every 200 epochs and at epoch 1, except in the soft-penalty sweep. Fixed-horizon experiments always included epoch 50,000. The grok epoch was defined as the first observed checkpoint with test accuracy strictly above 0.95. This outcome was preregistered for the independent-partition study. Checkpoint-based evaluation left the exact transition time unknown. Runs without an observed crossing were administratively censored at 50K. Outcomes included crossing counts, the conditional mean ± sample standard deviation among successful runs, and restricted mean crossing time through 50K (RMST-50K). RMST was calculated across all runs as the mean of min(T,50,000), assigning 50,000 to runs without a crossing. Differences among the fastest configurations were interpreted within the resolution of the 200-epoch evaluation grid (Section S1.3, Supporting Information). The post hoc trajectory analysis preserved the preregistered first-crossing outcome. A recorded relapse was defined as any subsequent checkpoint at or below 0.95 following the first crossing. Horizon persistence required final test accuracy above 0.95. Representation measurements at matched training ages were interpreted descriptively and were not considered evidence of causal mediation.

## 2.2. Tasks, models, and interventions

Modular arithmetic was studied over $\mathbb { Z } _ { \mathtt { p } }$ for primes $\mathfrak { p } \in \{ 5 9 , 9 7 , 1 1 3 \}$ using exact classification accuracy. Each input comprised the three-token sequence (a,b,=) with a dedicated separator token. The target was f(a,b) mod p for addition, subtraction, or multiplication. Learned positional embeddings preserved operand order. Standard experiments used a 30/70 train/test split over al $\mathfrak { p } ^ { 2 }$ ordered pairs with partition seed 42 and a separate ten-seed model pool. The base transformer was set with model dimension 128, four attention heads, MLP dimension 512, and learned positional embeddings. Full-batch AdamW was used with a learning rate of $6 \times 1 0 ^ { - 4 }$ . The baseline omitted LayerNorm, RMSNorm, residual projection, and used randomly initialized learned token embeddings, linear unembedding at logit scale 1, and weight decay 1.0. The complete Sphere+Fourier configuration used selected sine/cosine token features, unit-norm projection after embedding and each residual operation, cosine unembedding at logit scale 10, and zero weight decay.

The one-layer factorial excluded Fourier initialization and independently varied hard residual projection (N), readout weight normalization (C), logit scale 10 versus 1 (T), and weight decay 1 versus 0 (W). C=1 normalized each output-weight row without separately normalizing the final hidden vector. The readout represented cosine similarity only when N=1 also produced a unit-norm final hidden vector. Hidden magnitude remained in the logits when N=0. Logit scale multiplied the readout as an inverse-temperature. All 16 combinations were evaluated using ten model seeds. Matched contrasts held the C, T, and W operations fixed. Since projection could alter hidden geometry and output scale, ratios conditional on successful runs were not interpreted as marginal causal effects. The fixed-horizon validation and matched-age representation experiments were conducted on NVIDIA GeForce RTX 4090 GPUs. Deterministic algorithms were enabled. The manifests contained hashes for the training code, configurations, results, histories, and checkpoints. Software versions and runtime settings are provided in Section S1.4 (Supplementary Material).

## 3. Results

## 3.1. Component and factorial decomposition

The factorial analysis resolved the interactions underlying the effect of projection (Fig. 1; Section S2.2, Supplementary Material). Projected models did not produce crossings at either logit scale in the absence of both readout weight normalization and decay (N1C0T0W0 and N1C0T1W0). Grokking was also observed without projection. With weight decay, the unprojected configurations N0C0T1W1 and N0C1T0W1 reached the criterion in 10/10 runs at $2 1 , 9 0 0 \pm 7 , 3 3 1$ and 24,680 ± 9,036 epochs, respectively. Combining readout weight normalization, logit scale 10, and decay without projection reduced the mean to $9 , 4 2 0 ~ \pm ~ 1 , 6 5 6$ epochs with 10/10 crossings. Hidden magnitude was retained in these unprojected readouts.

Complete one-layer $2 ^ { 4 }$ factorial (Fourier initialization off)  
Text gives crossings out of 10 and RMST-50K; lower RMST is faster  
![](images/97f87746130f0bcdf5c0ca4f2b629bc031b9c029e4a263fc4831b70ee70293a7.jpg)  
Logit-scale / weight-decay factors  
Fig. 1. Complete one-layer $2 ^ { 4 }$ factorial analysis of $\mathbb { Z } _ { 1 1 3 }$ addition with Fourier initialization disabled. Cell color represents RMST-50K. Annotations indicate crossings out of ten and RMST in epochs. Lower RMST values indicate faster crossing. The lowest RMST values occurred with projection. However, no projected runs crossed by 50K when both readout weight normalization and decay were absent.

Nevertheless, hard projection substantially accelerated configurations capable of grokking. Across the four W=1 settings with a minimum of one crossing in both projection states, the successful-run mean without projection was 8-37 times the corresponding mean with projection (Table 1). The first contrast compared two successful unprojected runs with ten projected runs. Each of the remaining three contrasts compared ten runs with ten runs. Projection was present in all six configurations with conditional means below 5,000 epochs. Accordingly, the influence of the unit norm constraint on grokking depended on the accompanying readout and decay settings.

Table 1: Matched factorial contrasts with a crossing in both projection states. W=1 throughout and Fourier initialization was disabled. C denotes normalization of readout weights and T denotes logit scale 10 rather than 1. Ratios were calculated by dividing the no-projection mean by the projection mean. Final hiddenvector normalization remained part of N.
<table><tr><td>C</td><td>T</td><td>No projection: crossings, epoch</td><td>Projection: crossings, epoch</td><td>Conditional ratio</td></tr><tr><td>0</td><td>0</td><td> $2 / 1 0 , 4 4 , 1 0 0 \pm 6 , 0 8 1$ </td><td> $1 0 / 1 0 , 4 , 5 0 0 \pm 1 , 0 2 5$ </td><td>9.8×</td></tr><tr><td>0</td><td>1</td><td> $1 0 / 1 0 , 2 1 , 9 0 0 \pm 7 , 3 3 1$ </td><td> $1 0 / 1 0 , 2 , 7 6 0 \pm 2 9 5$ </td><td>7.9×</td></tr><tr><td>1</td><td>0</td><td> $1 0 / 1 0 , 2 4 , 6 8 0 \pm 9 , 0 3 6$ </td><td> $1 0 / 1 0 , 6 6 0 \pm 1 9 0$ </td><td>37.4×</td></tr><tr><td>1</td><td>1</td><td> $1 0 / 1 0 , 9 , 4 2 0 \pm 1 , 6 5 6$ </td><td> $1 0 / 1 0 , 8 2 0 \pm 1 7 5$ </td><td>11.5×</td></tr></table>

Component ablation demonstrated that Fourier initialization was insufficient to explain the acceleration (Section S2.1, Supplementary Material). The baseline reached grokking criterion in $2 / 1 0$ runs at $^ { 4 4 , 1 0 0 \pm 6 , 0 8 1 }$ epochs among successful runs. Fourier initialization without decay did not produce any crossings by 50K. With decay, 8/10 runs reached the criterion at $2 8 { , } 5 5 0 \pm 7 { , } 9 0 6$ epochs. Projection with cosine readout without Fourier initialization produced 10/10 crossings at $1 , 2 4 0 ~ \pm ~ 3 6 3$ epochs. Adding Fourier initialization reduced the mean to $7 2 0 \pm 2 1 5$ epochs with 10/10 crossings. Thus, Fourier initialization improved this configuration without being required for rapid grokking. The corresponding no-Fourier condition in the factorial was N1C1T1W0.

## 3.2. Independent-partition validation of the complete effect

The 100-run independent-partition study evaluated the robustness of the complete configuration, rather than repeating the individual-factor contrasts (Fig. 2). At one layer, Sphere+Fourier reached the criterion in all 25 runs compared with $8 / 2 5$ baseline runs (Table 2). The paired restricted-time advantage was $4 5 , 1 8 4 \pm 7 , 3 2 7$ epochs with a two-factor bootstrap 95% interval of [39,480, 49,128]. At two layers, both configurations reached the criterion in 25/25 runs. Sphere+Fourier generalized earlier with a paired advantage of $1 6 , 7 6 0 \pm 5 , 3 0 7$ epochs $[ 1 3 , 5 5 2 , 1 9 , 9 3 6 ]$ . The partition-specific mean advantage was positive at both depths for every partition. These results demonstrated consistent shallow-model acceleration across the five predeclared data partitions.

![](images/045eb658acd7f6527fa736e98e7a1be27ef7f50ece082846732b7cad97b01bbf.jpg)

![](images/2e9ce708578c6ed199038bf472d33bd112b1d1a9ac9a7ecec5d49538ed54b3c3.jpg)

![](images/b66bec2ee0cc7c8d728e29bac1b9b4dad8be11f3ca039c6e43b80406914c80f1.jpg)  
Fig. 2. Fixed-horizon validation of the complete baseline and Sphere+Fourier configurations. The first two panels show the fraction of runs not yet observed above 95% held-out accuracy. The third panel shows the matched baseline-minus-Sphere+Fourier RMST-50K difference within each partition. The mean difference was positive for every partition at both depths.

Complete training through to 50K for each run in the study enabled separate assessment of early generalization and subsequent stability. All 50 two-layer runs initially crossed the threshold. Subsequent relapses were recorded for 22/25 baseline runs and 7/25 Sphere+Fourier runs. At epoch 50K, 23/25 baseline runs and 20/25 Sphere+Fourier runs remained above the threshold. Sphere+Fourier generalized earlier and exhibited fewer recorded relapses. However, fewer runs finished above the threshold. This trajectory analysis was specified after observation of the endpoint pattern and remained post hoc. Matched endpoint counts and complete trajectories are provided in Section S3 (Supplementary Material).

Table 2: Independent-partition outcomes. Each row includes 25 fixed horizon runs. Conditional epochs were calculated over successful runs. RMST-50K and final accuracy included all runs.
<table><tr><td>Configuration and depth</td><td>Crossings</td><td>Conditional epoch</td><td>RMST-50K</td><td>Final test accuracy</td><td>Above 95% at 50K</td></tr><tr><td>Baseline, 1L</td><td>8/25</td><td>37,350 ± 7,873</td><td>45,952</td><td> $4 4 . 6 \pm 4 5 . 3 \%$ </td><td>8/25</td></tr><tr><td>Sphere+Fourier, 1L</td><td>25/25</td><td>768±243</td><td>768</td><td>100.0 ± 0.01%</td><td>25/25</td></tr><tr><td>Baseline, 2L</td><td>25/25</td><td>18,368±5,303</td><td>18,368</td><td>95.0 ± 11.0%</td><td>23/25</td></tr><tr><td>Sphere+Fourier, 2L</td><td>25/25</td><td>1,608± 950</td><td>1,608</td><td>95.3 ± 11.3%</td><td>20/25</td></tr></table>

## 3.3. Controls, representation evidence, and boundaries

The normalization variants exhibited different behavior under the tested conditions (Section S2.4, Supplementary Material). With Fourier initialization, normalized readout weights, logit scale 10, zero decay, and the same learning rate, sphere projection produced 10/10 crossings at $7 2 0 \pm 2 1 5$ epochs. LayerNorm, RMSNorm, and no normalization produced 0/10 crossings. LayerNorm and RMSNorm contain learned affine scales without enforcing a unit L2 radius. Consequently, these comparisons also altered the magnitude entering the readout and could not isolate normalization from output scale or establish a general advantage of spherical geometry (Singh et al., 2026; Yıldırım, 2026). A separate projected factorial setting produced 10/10 crossings at logit scales 1, 2, $^ { 5 , }$ and 10 with mean crossing times of 580-820 epochs. Five random five-frequency sets on $\mathbb { Z } _ { 1 1 3 }$ each produced $1 0 / 1 0$ crossings with means of 760-880 epochs. The results indicated a modest benefit of Fourier initialization without identifying the tested frequency set as a unique explanation for acceleration.

A separate 20-run study compared token embeddings at matched training ages (Section S4, Supplementary Material). At 50K, baseline embeddings reached an effective dimensionality of $2 2 . 9 \pm 6 . 1$ compared with $8 0 . 2 \pm 0 . 4$ for Sphere+Fourier. The matched-seed difference was $5 7 . 3 \pm$ 5.9. This statistic quantified the distribution of variance across embedding directions (Section S1.4, Supporting Information). Matching training ages avoided comparison of an early-stopped fast model with a substantially older baseline. However, differences between the complete configurations and their outcome distributions prevented isolation of hard projection or attribution of grokking to effective dimensionality. Rapid grokking occurred without collapse of this tokenembedding statistic. Substantial baseline compression was also insufficient for reliable grokking in these experiments.

Task and depth sweeps identified conditions under which the configuration became unreliable (Section S5, Supplementary Material). Sphere+Fourier produced $1 0 / 1 0$ crossings for $\mathbb { Z } _ { 1 1 3 }$ multiplication and $\mathbb { Z } _ { 9 7 }$ addition compared with $1 / 1 0$ for $\mathbb { Z } _ { 1 1 3 }$ subtraction and $0 / 1 0$ for $\mathbb { Z } _ { 5 9 }$ addition. The operation comparisons lacked matched baselines. The same reference Fourier frequencies were used across primes. Uniform attention permitted grokking on the three-token addition task and accelerated the baseline. Reliability also depended on network depth (Fig. 3). In the separate depth sweep, Sphere+Fourier generalized rapidly in 10/10 one-layer runs and 9/10 two-layer runs. Success rates varied at greater depths with $0 / 1 0$ crossings by 50K at 10, 11, 12, and 16 layers. Baseline configurations continued to cross. These failures could not be attributed specifically to projection since the sweep compared complete configurations.

![](images/2845ca61e68046acb84351a8fc3ea6b2c14c93f965c2d60d912ce7563821f006.jpg)  
Fig. 3. Depth scaling of the complete baseline and Sphere+Fourier configurations on $\mathbb { Z } _ { 1 1 3 }$ addition. Points show successful-run medians with IQR. Labels indicate crossings out of ten. Sphere+Fourier configurations with no observed crossings are shown at 50K.

## 4. Discussion

The factorial analysis demonstrated that the influence of projection depended on the readout and training settings. Projection was present in every rapidly generalizing configuration. However, no projected runs reached the criterion in the absence of both readout weight normalization and decay. Weight decay also supported slower grokking without projection. Accordingly, the unitnorm constraint should be evaluated in conjunction with the readout and regularization settings.

The interventions acted on distinct model components. Residual projection constrained hidden vectors, weight decay acted on parameters, and readout weight normalization fixed classifier-row lengths. Projection additionally normalized the final hidden vector. The projection contrasts therefore could not separate geometry from the magnitude entering the readout. The factoria analysis identified interactions among these operations without identifying the learned circuit or explaining why an individual run generalized. Related work distinguished weight magnitude from direction (Truong, 2026a) and demonstrated the influence of feature content and intervention timing on representational priors (Howe, 2026). Preliminary evidence of smooth drift in a squaredloss setting also indicated that a single transition mechanism should not be assumed (Basak, 2026).

The independent-partition study demonstrated shallow-model acceleration across five training subsets. Continued training to a fixed horizon distinguished first crossing, absence of subsequent relapses, and final performance above the threshold. Early stopping would have obscured these differences. The two-layer Sphere+Fourier configuration improved the first two outcomes without improving the third. Previous studies demonstrated continued representation evolution after grokking, changes in optimizer rankings under sustained criteria, and loss of generalization in some Muon-trained transformers after grokking (Truong, 2026b; Wang, 2026; Janati et al., 2026). Accordingly, future confirmatory studies should preregister both first-crossing and sustained or final-accuracy outcomes.

Interpretation of the supporting controls was limited by differences in the interventions and measurements. The tested soft penalty differed from successful soft-constraint methods in target radius, intervention site, readout, and optimization (Section S2.3, Supporting Information). This comparison could not establish superiority of hard constraints (Tiwari et al., 2026). The tokenembedding statistic also differed from the hidden-state effective dimensionality controlled by Kazanskii (2026). Neither comparison established a mechanism for the factorial result.

The experiments were limited to small modular-arithmetic transformers. Most factorial configurations used one learning rate, prime, operation, and depth. The partition study evaluated five splits and five model seeds but without the adjustment of its bootstrap for every exploratory comparison. The 200-epoch evaluation grid may have missed brief accuracy changes, and the persistence analysis remained post hoc. A fixed RTX 4090 environment did not establish crosshardware equivalence (Ootani, 2026). Targeted extensions should vary learning rate in selected factorial settings, isolate projection across depth, match hard and soft constraints by radius and readout, and vary Fourier frequencies and their number at p=59. Transfer to language or vision requires separate evaluation.

## 5. Conclusion

Over the prescribed 960 controlled runs, projection accelerated grokking in shallow modular transformers through interactions with readout and regularization settings. It was neither sufficient nor required for generalization within the tested conditions. The complete Sphere+Fourier configuration generalized earlier across five train/test partitions, although fewer of its two-layer runs than baseline runs finished above the accuracy threshold. Its reliability also varied across tasks and depths. Together, these experiments identify component combinations that support rapid grokking and characterize the robustness and limits of the complete Sphere+Fourier configuration.

## Data and code availability

Code, experimental configurations, run-level results, training histories, and analysis scripts are stored in the project repository. The repository is currently private with access provided on demand.

## Disclosure of AI assistance

The authors used OpenAI’s ChatGPT and Codex (including GPT-5.5, GPT-5.6, and GPT-6 Astra) and Anthropic’s Claude Opus 4.6 to assist with code development, analysis, and manuscript drafting and refinement. The authors reviewed and verified the final methods, results, and conclusions and take full responsibility for the work.

## References

Basak, S. (2026). Is Grokking a Loss of Normal Hyperbolicity of the Interpolation Manifold? arXiv:2608.14803.

Howe, G. L. (2026). What Makes a Representational Prior Work? Feature Families, Label-Free Invariances, and Critical Windows in Grokking. arXiv:2607.12735.

Janati, A., El Maghraoui, K., Kanavalau, A., & Belfatmi, A. (2026). Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers. arXiv:2608.07436.

Kazanskii, M. A. (2026). How to Tame Grokking: Representation Geometry as a Control Signal. arXiv:2607.11666.

Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M. (2022a). Towards Understanding Grokking: An Effective Theory of Representation Learning. Advances in Neural Information Processing Systems 35.

Liu, Z., Michaud, E. J., & Tegmark, M. (2022b). Omnigrok: Grokking Beyond Algorithmic Data. arXiv:2210.01117.

Merrill, W., Tsilivis, N., & Shukla, A. (2023). A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks. arXiv:2303.11873.

Mohamadi, M. A., Li, Z., Wu, L., & Sutherland, D. J. (2024). Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition. arXiv:2407.12332.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. International Conference on Learning Representations.

Ootani, Y. (2026). Grokking Is Conditional and Fragile: A Fully-Tractable, Multi-Seed Study at 12K Parameters. arXiv:2607.05104.

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

Singh, J., Misra, D., & Orvieto, A. (2026). Explaining Grokking in Transformers through the Lens of Inductive Bias. arXiv:2602.06702.

Tiwari, S., Chauhan, A., & Singh, M. (2026). Radial Suppression Accelerates Algorithmic Generalization: A Geometric Analysis of Delayed Generalization. arXiv:2606.32000.

Truong, X. K. (2026a). Cross-Trajectory Chimera Interventions Reveal Dissociable Roles of Weight Magnitude and Direction in Grokking. arXiv:2607.06628.

Truong, X. K. (2026b). At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics. arXiv:2607.06639.

Varma, V., Shah, R., Kenton, Z., Kramár, J., & Kumar, R. (2023). Explaining Grokking through Circuit Efficiency. arXiv:2309.02390.

Wang, Y. (2026). The Active Ingredient in Muon's Grokking. arXiv:2607.20512.

Yıldırım, A. (2026). The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology. arXiv:2603.05228v3.

# Supplementary Material: Dissecting Geometric Constraints in Grokking: Interaction-Dependent Acceleration in Modular Arithmetic

Zachary A. Hampton<sup>1</sup>, Wasiq Khan<sup>2</sup> (SMIEEE, Ph. D), and Bilal Khan<sup>1</sup>, Ph.D.

<sup>1</sup>School of Computer Science and Engineering, California State University, San Bernardino, {Zachary.Hampton, Bilal.Khan}@csusb.edu

<sup>2</sup>School of Computer Science and Mathematics, Liverpool John Moores University, UK, w.khan@ljmu.ac.uk

## S1. Experimental and reporting details

## S1.1. Tasks, data, architecture, and interventions

Modular arithmetic was studied over $\mathbb { Z } _ { \mathtt { p } }$ for primes $\mathfrak { p } \in \{ 5 9 , 9 7 , 1 1 3 \}$ . Each input was represented as $\left( \mathsf { a } , \mathsf { b } , \mathsf { = } \right)$ with separator token id p and target $f ( a , b )$ mod p for $\mathrm { ~ f ~ } \in \ \{ + , \ - , \ \times \}$ . Learned positional embeddings distinguished (a,b) from (b,a), including for subtraction. Held-out performance was evaluated using exact multiclass accuracy, defined as the fraction of test examples for which the argmax residue equaled the target. All experiments used 30% of the ${ \mathfrak { p } } ^ { 2 }$ ordered pairs for training and 70% for testing. Partition seed 42 fixed dataset membership in the standard experiments. The fixed-horizon validation crossed partition seeds {42,43,44,45,46}, model seeds $\{ 4 2 , 1 3 7 , 2 5 6 , 3 1 4 , 5 5 5 \}$ , depths {1,2}, and two configurations to produce 100 runs without early stopping. Model and optimizer randomness were controlled through separate model seeds. The standard ten-seed pool was {42, 137, 256, 314, 555, 777, 1024, 1337, 2718, 9001}. The base transformer had model dimension 128, four attention heads, MLP dimension 512, learned token and positional embeddings, and one layer unless otherwise specified. The depth experiments evaluated 1-12 and 16 layers. Optimization was performed using full-batch AdamW with a learning rate of $6 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 9 9 )$ , and $\varepsilon = 1 0 ^ { - 8 }$ . Each epoch comprised one update over the complete training set. The standard baseline omitted LayerNorm, RMSNorm, residual projection, and used randomly initialized learned token embeddings, linear unembedding at logit scale 1, and weight decay 1.0. The standard Sphere+Fourier configuration used Fourier-initialized token embeddings, hard residual projection, cosine unembedding at logit scale 10, and zero weight decay. Comparisons between the complete configurations could not isolate individual components since several properties changed simultaneously.

Hard projection mapped the hidden vector to $\mathrm { { h } } / \| \mathrm { { h } } \| _ { 2 }$ after embedding and after each attention and MLP residual update. The factorial factor N controlled all projection sites. Factor C normalized each row of the unembedding weight matrix to unit L2 norm without separately normalizing the final hidden vector. The resulting dot product represented cosine similarity only when N=1 also produced a unit-length final hidden vector. Logits retained hidden-vector magnitude when N=0 and C=1. The logits were multiplied by the positive scale τ as an inverse-temperature, not divided by τ. In the binary factorial, T=1 denoted logit scale 10 and T=0 denoted logit scale 1. W=1 denoted weight decay 1.0 and W=0 denoted zero decay. Selected token-embedding coordinates received sine/cosine features at reference frequencies {14, 35, 41, 42, 52}. Separate $\mathbb { Z } _ { 1 1 3 }$ experiments compared this set with no Fourier features and five fixed random five-frequency sets. An auxiliary attention experiment replaced learned attention with fixed uniform weights on the three-token input.

## S1.2. Experimental groups

The study comprised 960 runs within 13 experimental groups (Table S1). Each run had a corresponding result record and manifest. Conditions repeated in separate run pools were analyzed within their designated experimental groups. Fourier-timing and endpoint-representation experiments were included in the inventory with complete records saved in the repository. The representation analyses reported here used fixed-horizon experiments to compare models at matched training ages.

Table S1: Experimental groups and their roles in the study.
<table><tr><td>Experimental group</td><td>Runs</td><td>Design</td><td>Primary purpose</td></tr><tr><td>Component ablation</td><td>30</td><td>3 conditions × 10 seeds</td><td>Fourier and decay contributions</td></tr><tr><td>2⁴ component factorial</td><td>160</td><td>16 cells × 10 seeds</td><td>N/C/T/W interactions</td></tr><tr><td>Soft-constraint diagnostic</td><td>80</td><td>8 penalties × 10 seeds</td><td>Narrow unmatched diagnostic</td></tr><tr><td>Operation/prime transfer</td><td>50</td><td>5 settings × 10 seeds</td><td>Boundary conditions</td></tr><tr><td>Attention routing</td><td>30</td><td>3 settings × 10 seeds</td><td>Learned versus uniform routing</td></tr><tr><td>Fourier timing</td><td>20</td><td>2 settings ×10 seeds</td><td>Supporting timing evidence</td></tr><tr><td>Endpoint representation</td><td>20</td><td>2 settings × 10 seeds</td><td>Endpoint-only descriptive comparison</td></tr><tr><td>Fixed-horizon representation</td><td>20</td><td>2 configurations × 10 seeds</td><td>Matched-age trajectories</td></tr><tr><td>Depth scaling</td><td>260</td><td>13 depths × 2 configurations × 10 seeds</td><td>Configuration-level depth boundary</td></tr><tr><td>Normalization variants</td><td>40</td><td>4 variants × 10 seeds</td><td>Sphere/LN/RMS/none control</td></tr><tr><td>Logit-scale sweep</td><td>80</td><td>8 logit scales × 10 seeds</td><td>Logit-scale robustness</td></tr><tr><td>Fourier-frequency controls</td><td>70</td><td>7 initializations ×10 seeds</td><td>Frequency-set robustness on Z113</td></tr><tr><td>Independent partitions</td><td>100</td><td>5 partitions × 5 model seeds × 2 depths × 2 configurations</td><td>Primary validation</td></tr><tr><td>Total</td><td>960</td><td></td><td></td></tr></table>

## S1.3. Outcome definitions and statistical reporting

Test accuracy was evaluated at epoch 1 and every 200 epochs thereafter, except in the soft-penalty sweep where evaluation began at epoch 200. Fixed-horizon experiments permanently included epoch 50,000. The grok epoch was defined as the first observed checkpoint with held-out accuracy strictly above 0.95. Crossing times and relapses were resolved only on this evaluation grid. The designation “>50K” indicated that no observed checkpoint crossed the threshold by the administrative horizon. Most standard sweeps employed early stopping only after both training and test accuracy exceeded 0.99 at an evaluation checkpoint. Independent-partition and fixed-horizon representation runs were trained to 50K with early stopping disabled. Conditional grokking time was reported as mean ± sample standard deviation (ddof=1) over successful runs together with the event count. RMST-50K was calculated as the mean of min(T,50,000), assigning 50,000 to runs without an observed crossing. This measure included all runs under the common administrative horizon. Best and final accuracies also included all runs in the specified configuration. Standard deviations were not interpreted as evidence of statistical significance.

The independent-partition study used the paired estimand min $( \mathrm { T _ { b a s e l i n e } , 5 0 K ) \gets m i n ( \mathrm { T _ { b u n d l e } , 5 0 K ) } }$ for matched partition/model seeds. Positive values indicated an advantage for Sphere+Fourier. The two-factor percentile bootstrap independently resampled the five partition seeds and five model seeds with their crossed design preserved. Intervals were calculated using 20,000 replicates and fixed analysis seeds. The persistence analysis was specified after inspection of the completed fixedhorizon independent-partition experiments. Immediate persistence required every observed checkpoint after the first crossing to remain above 0.95. A relapse was defined as any subsequent observed checkpoint at or below 0.95. Observed sustained crossing was defined as the earliest above-threshold checkpoint after which all remaining observed checkpoints through 50K stayed above the threshold. Horizon persistence required final test accuracy above 0.95. A relapsing run was classified as test-only when no post-crossing training-accuracy checkpoint was at or below 0.95. These definitions applied to the recorded evaluation grid without replacing the preregistered first-crossing outcome.

## S1.4. Fixed-horizon representation and runtime protocol.

The canonical one-layer baseline and Sphere+Fourier configurations were each trained for 50,000 epochs using the standard ten seeds with early stopping disabled. Post-update checkpoints were saved at epochs 1, 200, 600, 1,000, 5,000, 10,000, 25,000, 40,000, and 50,000. Token-embedding measurements excluded the separator row. Effective dimensionality was defined as exp(H(q)), where q represented the normalized vector of centered squared singular values. Variance ranks were the minimum centered-SVD ranks required to explain 90% or 99% of variance. Attention entropy was calculated as natural-log entropy averaged over held-out examples, heads, and query positions. Fourier fraction of variance explained (FVE) was calculated using either the five highestenergy frequencies or the reference set. Norm CV denoted the coefficient of variation of tokenembedding row norms and did not describe projected residual-stream norms.

The fixed-horizon and independent-partition experiments used Linux x86\_64, CPython 3.12.13, NumPy 2.2.6, PyTorch 2.7.1+cu128, CUDA 12.8, cuDNN 9.7.1, and an NVIDIA GeForce RTX 4090. Deterministic algorithms were enabled. TF32 and cuDNN benchmarking were disabled. The manifests contained hashes for the training code, configurations, results, histories, and checkpoints.

## S2. Complete component and control evidence

## S2.1. Component ablation

Fourier initialization was insufficient for rapid grokking under the tested conditions (Table S2). Further attribution required the complete factorial analysis since these conditions differed in multiple components.

Table S2: Component ablation for one-layer $\mathbb { Z } _ { 1 1 3 }$ models. Conditional epochs were calculated over successful runs. RMST-50K includes all ten runs
<table><tr><td>Condition</td><td>Crossings</td><td>Conditional epoch</td><td>RMST-50K</td><td>Best test accuracy</td></tr><tr><td>Baseline + decay</td><td>2/10</td><td> $^ { 4 4 , 1 0 0 \pm 6 , 0 8 1 }$ </td><td>48,820</td><td> $3 7 . 5 \pm 4 2 . 4 \%$ </td></tr><tr><td>Fourier, no decay</td><td>0/10</td><td>&gt;50K</td><td>50,000</td><td> $0 . 9 \pm 0 . 1 \%$ </td></tr><tr><td>Fourier + decay</td><td>8/10</td><td> $2 8 , 5 5 0 \pm 7 , 9 0 6$ </td><td>32,840</td><td> $8 9 . 0 \pm 2 6 . 3 \%$ </td></tr><tr><td>Sphere, no Fourier</td><td>10/10</td><td> $1 { , } 2 4 0 \pm 3 6 3$ </td><td>1,240</td><td> $9 9 . 3 \pm 0 . 4 \%$ </td></tr><tr><td>Sphere + Fourier</td><td>10/10</td><td> $7 2 0 \pm 2 1 5$ </td><td>720</td><td> $9 9 . 7 \pm 0 . 3 \%$ </td></tr></table>

## S2.2. Full $2 ^ { 4 }$ factorial

Table S3 reports all 16 factorial configurations, including those without crossings by 50K. Fourier initialization was disabled throughout this configuration.

Table S3: One-layer $\mathbb { Z } _ { 1 1 3 }$ factorial analysis with Fourier initialization disabled. N, C, T, and W denote hard projection, readout weight normalization, logit scale 10 rather than 1, and weight decay 1, respectively. Conditional epoch is reported as mean ± sample standard deviation over successful runs. Factor C does not normalize the final hidden vector.
<table><tr><td>N C</td><td>T</td><td>W</td><td>Crossings</td><td>Conditional epoch</td><td>RMST-50K</td></tr><tr><td>0 0</td><td>0</td><td>0</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>0 0</td><td>0</td><td>1</td><td>2/10</td><td> $^ { 4 4 , 1 0 0 \pm 6 , 0 8 1 }$ </td><td>48,820</td></tr><tr><td>0 0</td><td>1</td><td>0</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>0 0</td><td>1</td><td>1</td><td>10/10</td><td> $2 1 , 9 0 0 \pm 7 , 3 3 1$ </td><td>21,900</td></tr><tr><td>0 1</td><td>0</td><td>0</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>0 1</td><td>0</td><td>1</td><td>10/10</td><td> $2 4 , 6 8 0 \pm 9 , 0 3 6$ </td><td>24,680</td></tr><tr><td>0 1</td><td>1</td><td>0</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>0 1</td><td>1</td><td>1</td><td>10/10</td><td>9,420 ± 1,656</td><td>9,420</td></tr><tr><td>1 0</td><td>0</td><td>0</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>1 0</td><td>0</td><td>1</td><td>10/10</td><td> $4 { , } 5 0 0 \pm 1 { , } 0 2 5$ </td><td>4,500</td></tr><tr><td>1 0</td><td>1</td><td>0</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>1 0</td><td>1</td><td>1</td><td>10/10</td><td> $2 { , } 7 6 0 \pm 2 9 5$ </td><td>2,760</td></tr><tr><td>1 1</td><td>0</td><td>0</td><td>9/10</td><td> $9 1 1 \pm 3 3 3$ </td><td>5,820</td></tr><tr><td>1 1</td><td>0</td><td>1</td><td>10/10</td><td> $6 6 0 \pm 1 9 0$ </td><td>660</td></tr><tr><td>1 1</td><td>1</td><td>0</td><td>10/10</td><td> $1 { , } 2 4 0 \pm 3 6 3$ </td><td>1,240</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>10/10  $8 2 0 \pm 1 7 5$ </td><td>820</td></tr></table>

## S2.3. Unmatched soft-constraint diagnostic

All soft penalty runs remained $\leq ~ 0 . 9 5$ for $\lambda \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 , 5 , 1 0 , 5 0 , 1 0 0 \}$ with mean best accuracy ranging from 1.5% to 43.0% and reaching $4 3 . 0 \pm 5 . 5 \%$ at λ=100, as illustrated in Fig. S1. The hard reference produced 10/10 crossings at $7 2 0 \pm 2 1 5$ epochs, demonstrating that the tested unit-radius penalties did not reproduce the behavior of the complete hard configuration. Differences in output geometry prevented interpretation as a one-variable hard-versus-soft comparison. The result neither established a phase transition nor contradicted positive findings obtained with target radius √d and different intervention locations (Tiwari et al., 2026).

Unmatched hard-versus-soft diagnostic

![](images/3841a326ef27ffa774b4a9f9216141e46d66a792931ba13c1b4addc7594dda52.jpg)  
Fig. S1. Best held-out accuracy for penalties $\lambda ( \| \mathrm { h } \| _ { 2 } { - } 1 ) ^ { 2 } .$ Points show mean ± sample standard deviation across ten seeds. Labels indicate crossings. The hard reference was the complete Sphere+Fourier configuration with different output geometry.

## S2.4. Normalization, logit scale, and frequency controls

Table S4: Normalization variants with the same Fourier initialization, readout weight normalization, nominal logit scale, and decay. Hidden radius was not matched.
<table><tr><td>Normalization</td><td>Crossings</td><td>Conditional epoch</td><td>RMST-50K</td><td>Best test accuracy</td></tr><tr><td>Sphere</td><td>10/10</td><td> $7 2 0 \pm 2 1 5$ </td><td>720</td><td> $9 9 . 7 \pm 0 . 3 \%$ </td></tr><tr><td>LayerNorm</td><td>0/10</td><td>&gt;50K</td><td>50,000</td><td> $1 3 . 0 \pm 6 . 4 \%$ </td></tr><tr><td>RMSNorm</td><td>0/10</td><td>&gt;50K</td><td>50,000</td><td> $1 3 . 9 \pm 6 . 9 \%$ </td></tr><tr><td>None</td><td>0/10</td><td>&gt;50K</td><td>50,000</td><td> $2 . 3 \pm 1 . 9 \%$ </td></tr></table>

LayerNorm and RMSNorm normalized the variance or root-mean-square magnitude without enforcing a unit L2 radius. In addition to the learned gains included by both approaches, LayerNorm also included learned offsets which affected output scale and intermediate activations since the readout only normalized its own readout weights. Therefore, the comparison (Table S4) could not separate spherical geometry from hidden radius or establish a general normalization advantage. It also did not exclude grokking with LayerNorm or RMSNorm under other placements or hyperparameters (Singh et al., 2026; Yıldırım, 2026).

Table S5: Logit-scale sweep with N=1, C=1, W=1, and Fourier initialization disabled. The scale τ multiplies the cosine readout.
<table><tr><td>Logit scale τ</td><td>Crossings</td><td>Conditional epoch</td><td>RMST-50K</td></tr><tr><td>0.5</td><td>10/10</td><td> $1 { , } 2 8 0 \pm 5 9 0$ </td><td>1,280</td></tr><tr><td>1</td><td>10/10</td><td> $6 6 0 \pm 1 9 0$ </td><td>660</td></tr><tr><td>2</td><td>10/10</td><td> $5 8 0 \pm 1 1 4$ </td><td>580</td></tr><tr><td>5</td><td>10/10</td><td> $6 4 0 \pm 1 2 6$ </td><td>640</td></tr><tr><td>10</td><td>10/10</td><td> $8 2 0 \pm 1 7 5$ </td><td>820</td></tr><tr><td>20</td><td>10/10</td><td> $2 , 0 0 0 \pm 4 4 2$   $5 , 4 8 0 \pm 1 , 0 5 1$ </td><td>2,000 5,480</td></tr><tr><td>50</td><td>10/10</td><td></td><td></td></tr><tr><td>100</td><td>9/10</td><td> $7 , 4 2 2 \pm 1 , 2 1 0$ </td><td>11,680</td></tr></table>

Mean crossing times were similar across logit scales 1-10 within the resolution of the 200-epoch evaluation grid (Table S5). Logit scale 10 was not required for rapid grokking. Larger scales delayed crossing in this sweep.

Table S6: Fourier-frequency controls for the one-layer Sphere+Fourier configuration on $\mathbb { Z } _ { 1 1 3 }$
<table><tr><td>Initialization</td><td>Crossings</td><td>Conditional epoch</td><td>RMST-50K</td></tr><tr><td>No Fourier features</td><td> $1 0 / 1 0$ </td><td> $1 { , } 2 4 0 \pm 3 6 3$ </td><td>1,240</td></tr><tr><td>Reference frequency set</td><td>10/10</td><td> $7 2 0 \pm 2 1 5$ </td><td>720</td></tr><tr><td>Random set 0</td><td>10/10</td><td> $7 6 0 \pm 2 4 6$ </td><td>760</td></tr><tr><td>Random set 1</td><td>10/10</td><td> $8 2 0 \pm 3 4 6$ </td><td>820</td></tr><tr><td>Random set 2</td><td>10/10</td><td> $8 8 0 \pm 2 3 5$ </td><td>880</td></tr><tr><td>Random set 3</td><td>10/10</td><td> $8 6 0 \pm 2 1 2$ </td><td>860</td></tr><tr><td>Random set 4</td><td>10/10</td><td> $7 8 0 \pm 1 7 5$ </td><td>780</td></tr></table>

The behavior produced by all random frequency sets was similar to the reference set on $\mathbb { Z } _ { 1 1 3 }$ (Table S6).

## S3. Complete-configuration validation and stability

Independent-partition validation used five independently sampled partitions and five model seeds for each combination of configuration and depth (Table S7). At one layer, matched restricted-time advantages through partitions were 47,320, 48,960, 48,400, 37,360, and 43,880 epochs. At two layers, the corresponding advantages were 18,560, 12,600, 17,360, 19,480, and 15,800 epochs.

Table S7: Independent-partition outcomes. Each row contains 25 runs. Final accuracy includes all runs. Horizon persistence requires final test accuracy above 0.95.
<table><tr><td>Configuration and depth</td><td>First crossings</td><td>Conditional first epoch</td><td>RMST-50K</td><td>Final test accuracy</td><td>Persistent at 50K</td></tr><tr><td>Baseline, 1L</td><td>8/25</td><td> $3 7 , 3 5 0 \pm 7 , 8 7 3$ </td><td>45,952</td><td> $4 4 . 6 \pm 4 5 . 3 \%$ </td><td>8/25</td></tr><tr><td>Sphere+Fourier, 1L</td><td>25/25</td><td> $7 6 8 \pm 2 4 3$ </td><td>768</td><td> $1 0 0 . 0 \pm 0 . 0 1 \%$ </td><td>25/25</td></tr><tr><td>Baseline, 2L</td><td>25/25</td><td> $1 8 , 3 6 8 \pm 5 , 3 0 3$ </td><td>18,368</td><td> $9 5 . 0 \pm 1 1 . 0 \%$ </td><td>23/25</td></tr><tr><td>Sphere+Fourier, 2L</td><td>25/25</td><td> $1 { , } 6 0 8 \pm 9 5 0$ </td><td>1,608</td><td> $9 5 . 3 \pm 1 1 . 3 \%$ </td><td>20/25</td></tr></table>

Table S8: Post hoc persistence counts. Immediate persistence requires every subsequent observed checkpoint to remain above threshold. Ever relapsed denotes any later checkpoint at or below threshold.
<table><tr><td>Configuration and depth</td><td>First crossed</td><td>Immediate persistence</td><td>Ever relapsed</td><td>Persistent at 50K</td></tr><tr><td>Baseline, 1L</td><td>8/25</td><td>8/25</td><td>0/25</td><td>8/25</td></tr><tr><td>Sphere+Fourier, 1L</td><td>25/25</td><td>24/25</td><td>1/25</td><td>25/25</td></tr><tr><td>Baseline, 2L</td><td>25/25</td><td>3/25</td><td>22/25</td><td>23/25</td></tr><tr><td>Sphere+Fourier, 2L</td><td>25/25</td><td>18/25</td><td>7/25</td><td>20/25</td></tr></table>

Out of the 22 relapsing two-layer baseline runs (Fig. S2; Table S8), 19 had at least one postcrossing training-accuracy checkpoint at or below 0.95. Six of the seven relapsing two-layer Sphere+Fourier runs were classified as test-only. Matched endpoints comprised 19 pairs with persistence in both runs, four baseline-only pairs, one Sphere+Fourier-only pair, and one pair without persistence in either run. Five Sphere+Fourier runs finished below the threshold despite final training accuracy of 1.0. The analysis remained descriptive and post hoc.

![](images/f52dfc75798e1867c743a687c080ba14ea91e651ed91d4072a051b645786634c.jpg)  
Fig. S2. Complete post hoc analysis of two-layer trajectories. Rows follow the matched partition/model-seed order. White-ring markers indicate first observed crossings. Color represents held-out accuracy through 50K. These trajectories provided the basis for the persistence qualification in the main manuscript.

## S4. Matched-age representation evidence

The baseline and Sphere+Fourier configurations began near 72 effective dimensions and remained similar at epoch 1,000 (Fig. S3). Endpoint measurements are reported in Table S9.

![](images/981d9253972fb320e4059fdec380a9487c286b02aa7a8da939712c1acde06f51.jpg)  
Fig. S3. Effective token-embedding dimensionality at matched training ages. Lines indicate means and bands represent ±1 sample standard deviation across ten seeds per configuration. The logarithmic epoch axis does not imply interpolation between saved checkpoints.

The comparison did not isolate projection because configurations differed in initialization, projection, readout, logit scale, and decay. Their outcome distributions were also unequal. Global covariance of the token-embedding table did not represent hidden-state intrinsic dimension, classifier rank, parameter-trajectory dimension, or a causal mediator. The observations remained descriptive.

Table S9: Fixed-horizon representation measurements at epoch 50,000. Values are reported as mean ± sample standard deviation over ten seeds. All geometry measurements except attention entropy were calculated from the token-embedding table.
<table><tr><td>Metric</td><td>Baseline</td><td>Sphere+Fourier</td></tr><tr><td>First-crossing success by 50K</td><td>2/10</td><td>10/10</td></tr><tr><td>Test accuracy</td><td> $3 7 . 5 \pm 4 2 . 7 \%$ </td><td> $1 0 0 . 0 \pm 0 . 0 \%$ </td></tr><tr><td>Effective dimensionality</td><td> $2 2 . 9 \pm 6 . 1$ </td><td> $8 0 . 2 \pm 0 . 4$ </td></tr><tr><td>Rank explaining 90% variance</td><td> $1 9 . 1 \pm 4 . 8$ </td><td> $6 9 . 1 \pm 0 . 6$ </td></tr><tr><td>Rank explaining 99% variance</td><td> $3 3 . 3 \pm 5 . 3$ </td><td> $1 0 0 . 7 \pm 0 . 5 $ </td></tr><tr><td>Attention entropy</td><td> $0 . 8 2 3 \pm 0 . 0 7 1$ </td><td> $0 . 9 7 5 \pm 0 . 0 0 6$ </td></tr><tr><td>Fourier FVE, top five</td><td> $4 8 . 8 \pm 2 5 . 1 \%$ </td><td> $1 5 . 8 \pm 0 . 6 \%$ </td></tr><tr><td>Fourier FVE, reference set</td><td> $1 0 . 1 \pm 7 . 8 \%$ </td><td> $1 4 . 9 \pm 0 . 8 \%$ </td></tr><tr><td>Token-embedding norm CV</td><td> $0 . 0 5 3 \pm 0 . 0 0 8$ </td><td> $0 . 0 5 6 \pm 0 . 0 0 3$ </td></tr></table>

## S5. Task, prime, attention, and depth boundaries

The multiplication results demonstrated generalization of the Sphere+Fourier configuration to another operation (Table S10). A matched multiplication baseline was not included. Subtraction remained unreliable despite preservation of operand order through positional embeddings. The $\mathtt { p } = 5 9$ configuration failed under the tested frequency set. The $\mathsf { p } { = } 9 7$ configuration generalized successfully. Accordingly, the p=59 result identified a limitation of the tested configuration without establishing intrinsic incompatibility with that prime.

Table S10: Task, prime-size, and attention-routing experiments. Each row represents an independent tenseed pool. Conditional epochs were calculated over successful runs.
<table><tr><td>Condition</td><td>Crossings</td><td>Conditional epoch</td><td>RMST-50K</td><td>Best test accuracy</td></tr><tr><td>Sphere+Fourier,  $\mathbb { Z } _ { 1 1 3 }$  addition</td><td>10/10</td><td> $7 2 0 \pm 2 1 5$ </td><td>720</td><td> $9 9 . 7 \pm 0 . 3 \%$ </td></tr><tr><td>Sphere+Fourier,  $\mathbb { Z } _ { 1 1 3 }$  multiplication</td><td>10/10</td><td> $1 , 0 8 0 \pm 4 5 4$ </td><td>1,080</td><td> $9 9 . 6 \pm 0 . 3 \%$ </td></tr><tr><td>Sphere+Fourier,  $\mathbb { Z } _ { 1 1 3 }$  subtraction</td><td>1/10</td><td>7,200</td><td>45,720</td><td> $6 6 . 4 \pm 2 6 . 2 \%$ </td></tr><tr><td>Sphere+Fourier,  $\mathbb { Z } _ { 9 7 }$  addition</td><td>10/10</td><td> $2 , 4 0 0 \pm 1 , 6 6 8$ </td><td>2,400</td><td> $9 9 . 4 \pm 0 . 3 \%$ </td></tr><tr><td>Sphere+Fourier,  $\mathbb { Z } _ { 5 9 }$  addition</td><td>0/10</td><td>&gt;50K</td><td>50,000</td><td> $3 0 . 0 \pm 0 . 9 \%$ </td></tr><tr><td>Baseline + uniform attention + decay,  $\mathbb { Z } _ { 1 1 3 }$  addition</td><td>10/10</td><td> $4 , 8 8 0 \pm 1 , 4 2 4$ </td><td>4,880</td><td> $9 9 . 2 \pm 0 . 1 \%$ </td></tr><tr><td>Sphere+Fourier + uniform attention,  $\mathbb { Z } _ { 1 1 3 }$  addition</td><td>10/10</td><td> $6 8 0 \pm 1 4 0$ </td><td>680</td><td> $9 9 . 8 \pm 0 . 2 \%$ </td></tr></table>

Complete depth results are reported in Table S11. Baseline configurations reached the criterion at every tested depth, although not all runs were successful. Since the comparison involved complete configurations, the observed failures at greater depth could not be attributed specifically to repeated projection.

Table S11: Complete depth-scaling experiments. Each row represents an independent ten-seed configuration. Conditional epochs were calculated over successful runs.
<table><tr><td>Depth</td><td>Baseline crossings</td><td>Baseline conditional epoch</td><td>Baseline RMST</td><td>Sphere+ Fourier crossings</td><td>Sphere+Fourier conditional epoch</td><td>Sphere+ Fourier RMST</td></tr><tr><td>1</td><td>2/10</td><td> $^ { 4 4 , 1 0 0 \pm 6 , 0 8 1 }$ </td><td>48,820</td><td>10/10</td><td> $7 2 0 \pm 2 1 5$ </td><td>720</td></tr><tr><td>2</td><td>10/10</td><td> $1 6 , 6 6 0 \pm 5 , 8 1 2$ </td><td>16,660</td><td>9/10</td><td> $2 , 3 7 8 \pm 2 , 0 7 5$ </td><td>7,140</td></tr><tr><td>3</td><td>10/10</td><td> $1 3 , 0 6 0 \pm 6 { , } 7 2 6$ </td><td>13,060</td><td>3/10</td><td> $3 , 6 6 7 \pm 1 , 2 2 2$ </td><td>36,100</td></tr><tr><td>4</td><td>8/10</td><td> $1 0 , 6 5 0 \pm 3 , 3 6 5$ </td><td>18,520</td><td>2/10</td><td> $3 0 , 0 0 0 \pm 8 , 7 6 8$ </td><td>46,000</td></tr><tr><td>5</td><td>10/10</td><td> $9 , 3 8 0 \pm 2 , 8 8 5$ </td><td>9,380</td><td>8/10</td><td> $1 4 , 6 7 5 \pm 1 3 , 5 7 2$ </td><td>21,740</td></tr><tr><td>6</td><td>10/10</td><td> $8 , 2 4 0 \pm 5 , 4 8 3$ </td><td>8,240</td><td>10/10</td><td> $1 4 , 0 6 0 \pm 7 , 9 0 6$ </td><td>14,060</td></tr><tr><td>7</td><td>10/10</td><td> $9 , 6 6 0 \pm 5 , 4 9 3$ </td><td>9,660</td><td>7/10</td><td> $9 , 1 4 3 \pm 2 , 0 8 4$ </td><td>21,400</td></tr><tr><td>8</td><td>9/10</td><td> $8 , 2 6 7 \pm 3 , 7 8 4$ </td><td>12,440</td><td>3/10</td><td> $7 , 5 3 3 \pm 3 , 9 3 1$ </td><td>37,260</td></tr><tr><td>9</td><td>9/10</td><td> $1 4 , 3 7 8 \pm 1 0 , 2 4 1$ </td><td>17,940</td><td>2/10</td><td> $3 , 8 0 0 \pm 1 , 9 8 0$ </td><td>40,760</td></tr><tr><td>10</td><td>10/10</td><td> $1 6 , 6 2 0 \pm 1 3 , 1 3 5$ </td><td>16,620</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>11</td><td>10/10</td><td> $1 4 , 3 2 0 \pm 9 , 9 0 9$ </td><td>14,320</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>12</td><td>9/10</td><td>13,689±6,799</td><td>17,320</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr><tr><td>16</td><td>10/10</td><td>13,080 ± 6,050</td><td>13,080</td><td>0/10</td><td>&gt;50K</td><td>50,000</td></tr></table>

## References

Basak, S. (2026). Is Grokking a Loss of Normal Hyperbolicity of the Interpolation Manifold? arXiv:2608.14803.

Howe, G. L. (2026). What Makes a Representational Prior Work? Feature Families, Label-Free Invariances, and Critical Windows in Grokking. arXiv:2607.12735.

Janati, A., El Maghraoui, K., Kanavalau, A., & Belfatmi, A. (2026). Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers. arXiv:2608.07436.

Kazanskii, M. A. (2026). How to Tame Grokking: Representation Geometry as a Control Signal. arXiv:2607.11666.

Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M. (2022a). Towards Understanding Grokking: An Effective Theory of Representation Learning. Advances in Neural Information Processing Systems 35.

Liu, Z., Michaud, E. J., & Tegmark, M. (2022b). Omnigrok: Grokking Beyond Algorithmic Data. arXiv:2210.01117.

Merrill, W., Tsilivis, N., & Shukla, A. (2023). A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks. arXiv:2303.11873.

Mohamadi, M. A., Li, Z., Wu, L., & Sutherland, D. J. (2024). Why Do You Grok? A Theoretical Analysis of Grokking Modular Addition. arXiv:2407.12332.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. International Conference on Learning Representations.

Ootani, Y. (2026). Grokking Is Conditional and Fragile: A Fully-Tractable, Multi-Seed Study at 12K Parameters. arXiv:2607.05104.

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

Singh, J., Misra, D., & Orvieto, A. (2026). Explaining Grokking in Transformers through the Lens of Inductive Bias. arXiv:2602.06702.

Tiwari, S., Chauhan, A., & Singh, M. (2026). Radial Suppression Accelerates Algorithmic Generalization: A Geometric Analysis of Delayed Generalization. arXiv:2606.32000.

Truong, X. K. (2026a). Cross-Trajectory Chimera Interventions Reveal Dissociable Roles of Weight Magnitude and Direction in Grokking. arXiv:2607.06628.

Truong, X. K. (2026b). At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics. arXiv:2607.06639.

Varma, V., Shah, R., Kenton, Z., Kramár, J., & Kumar, R. (2023). Explaining Grokking through Circuit Efficiency. arXiv:2309.02390.

Wang, Y. (2026). The Active Ingredient in Muon's Grokking. arXiv:2607.20512.

Yıldırım, A. (2026). The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology. arXiv:2603.05228v3.