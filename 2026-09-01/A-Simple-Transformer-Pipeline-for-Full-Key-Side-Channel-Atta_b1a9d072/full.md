# A Simple Transformer Pipeline for Full-Key Side-Channel Attacks on Uncropped Datasets

Jimmy Gammell and Kaushik Roy

Elmore Family School of Electrical and Computer Engineering Purdue University, West Lafayette, IN 47907, USA {jgammell, kaushik}@purdue.edu

Abstract. Deep learning-based side-channel analysis has historically focused on single-byte targets and manually cropped traces, which risks discarding exploitable leakage. While recent work has proposed specialized architectures and resampling techniques to address this gap, the literature lacks a simple transformer baseline for simultaneous full-key attacks on uncropped traces. We present an open-source transformer implementation for uncropped full-key attacks which uses the standard transformer encoder backbone, adapting only the input and output layers to the side-channel setting. We release our implementation, training recipes, and pretrained weights for uncropped ASCADv1f, ASCADv1r, and CHES-CTF-2018 which achieve performance competitive with previously-reported results, while using less than 10GB of VRAM and requiring at most 3.34 hours of training on a single NVIDIA A6000 (link).

Keywords: Side-Channel Analysis · Deep Learning · Transformers

## 1 Introduction

Deep learning-based side-channel analysis (DLSCA) has emerged as a powerful tool for evaluating the vulnerability of cryptographic hardware to physical side-channel attacks [8]. However, much of the literature targets only one key byte, and crops traces to ∼1% of their original length after manually identifying regions expected to leak that byte. This reduces modeling cost, but risks discarding exploitable leakage from unexpected instructions or locations [4]. Such preprocessing is undesirable in defensive use cases where designers want to test the full execution for leaks without assuming where or why they occur.

Recent work has advanced beyond these simplified settings, successfully attacking multiple key bytes [7,4] and raw [6,2], long [5], or resampled [7] traces. In particular, [2,5] have demonstrated the effectiveness of transformer-style architectures in long-trace settings. However, these specialized architectures depart significantly from standard transformer encoders, and the community lacks a simple baseline which leaves the standard transformer backbone unchanged. Additionally, no prior work to our knowledge has simultaneously attacked all key bytes from uncropped traces.

This artifact fills that gap with an installable PyTorch implementation of a standard transformer-based pipeline for full-key attacks on uncropped traces with no manual feature selection or cropping. We release training recipes and pretrained weights for the uncropped variants of ASCADv1f, ASCADv1r [1], and CHES-CTF-2018 [9], achieving competitive performance with results reported in prior work. Table 1 positions the artifact relative to representative prior work. Our aim is to reduce the engineering effort required for uncropped attacks and provide a deliberately simple baseline against which future side-channel-informed architectures can be benchmarked.

<table><tr><td>Work</td><td>Trace preprocessing</td><td>Architecture</td><td>Target</td></tr><tr><td>EFSS [7]</td><td>Downsampling/resampling</td><td>MLP/CNN</td><td>Full key (16 independent attacks)</td></tr><tr><td>PART [6]</td><td>None (raw traces)</td><td>Specialized LSTM</td><td>Single byte</td></tr><tr><td>EstraNet [5]</td><td>Cropping (long traces)</td><td>Specialized transformer</td><td>Single byte</td></tr><tr><td>GPAM [2]</td><td>None (raw traces)</td><td>Specialized transformer</td><td>Single byte + random nonces</td></tr><tr><td>Ours</td><td>None (raw traces)</td><td>Boilerplate transformer</td><td>Full key</td></tr></table>

Table 1. Positioning of our artifact relative to representative prior work.

Our artifact supports OPTIMIST’s goals by providing an open-source case study, reference implementation, and reproducible benchmark results for applying standard modern deep learning techniques to public side-channel datasets without manual feature selection or cropping. The intended users are DLSCA researchers studying uncropped attacks who want pretrained models without doing costly training and hyperparameter tuning, a simple and performant backbone to extend, or a reproducible baseline for boilerplate transformer-based attacks.

## 2 Artifact overview

<table><tr><td colspan="2">config/ . YAML config files for dataset, model, and training configurations</td></tr><tr><td colspan="2">experiments/ . Experiment entrypoints and infrastructure src/uncropped_transformers/ . .Reusable importable library code</td></tr><tr><td colspan="2"></td></tr><tr><td colspan="2">datasets/ . Dataset loading and preprocessing</td></tr><tr><td colspan="2">models/ ..Neural net architectures and building blocks</td></tr><tr><td colspan="2">training/ . Training loops, hyperparameter tuning</td></tr><tr><td colspan="2">evaluation/ . Accuracy, rank, and MTD metrics</td></tr></table>

Fig. 1. High-level organization of our repository.

We release an open-source PyTorch repository for training, tuning, and evaluating profiled full-key attacks on uncropped side-channel datasets using transformers. This artifact can be installed alongside its dependencies by cloning our repository here and typing pip install -e . inside its directory. To facilitate reuse, we separate importable library modules from experiment-specific entrypoints and infrastructure. See Fig. 1 for an overview of our repository’s structure.

Architecture. The transformer is a widely-used sequence modeling architecture that can be applied to diverse domains and data modalities by representing inputs as token sequences while leaving the backbone largely unchanged. Accordingly, we use a standard pre-norm transformer encoder with rotary position embeddings and modify only the input and output layers for the side-channel setting. Inspired by vision transformers [3], we divide an input trace into contiguous patches and project them to the model’s hidden dimension. We append one learned output token for each targeted byte, feed the resulting sequence through the transformer, and linearly project each transformed output token to logits for its corresponding byte. For all datasets, we use a single transformer to predict all 16 bytes of the first SubBytes output. For the ASCADv1 datasets we use the identity leakage model, and for CHES-CTF-2018 we use the multi-byte multi-bit model of [10] because the identity model performed poorly in our experiments.

![](images/e0f7e0748c1b9b515d1e29995503ab3e18a76a40a8963c4731b26fd65acf667b.jpg)  
Fig. 2. Overview of our full-key transformer pipeline. The input trace is divided into contiguous patches and linearly projected into token embeddings. One learned output token is appended for each targeted byte, and the resulting sequence is processed by a standard pre-norm transformer encoder. Each transformed output token is linearly projected to logits for its corresponding target byte.

Training. We train models using AdamW, applying weight decay to the weights (not biases) of the linear projections. The learning rate is linearly warmed up from zero to a base value over the first 5% of training steps, then decayed using cosine annealing over the remaining steps. We optimize the mean cross-entropy loss across all target bytes, use a minibatch size of 256, and use gradient clipping with a maximum $\ell _ { 2 }$ norm of 1. Each feature of input traces is standardized using its mean and standard deviation over the profiling set. We use a random 20% of profiling traces for validation and use the remaining data for training. The attack set is treated as a test set and used only for final evaluation. We select the model checkpoint with the lowest single-trace correct-key rank on the validation set. Hyperparameter configurations and search spaces can be found in our repository.

<table><tr><td>Dataset</td><td>Metric</td><td>Method</td><td>Result</td></tr><tr><td rowspan="3">ASCADf [1]</td><td>Byte 2–15 MTD ↓ (best, mean, worst)</td><td>EFSS (CNN/ID) [7] EFSS (MLP/ID) [7]</td><td>1, &gt;2.143k, &gt;3k 2, 5.571, 20</td></tr><tr><td></td><td>Ours EstraNet [5]</td><td>1.000, 1.011, 1.055 13</td></tr><tr><td>Byte 2 MTD ↓</td><td>Ours</td><td>1.004</td></tr><tr><td rowspan="5">ASCADr [1]</td><td>Byte 2–15 MTD ↓ (best, mean, worst)</td><td>EFSS (CNN/ID) [7] EFSS (MLP/ID) [7]</td><td>1, 4.643, 19 1, 1.429, 3</td></tr><tr><td rowspan="3">Byte 2 MTD ↓</td><td>Ours</td><td>1.000, 1.001, 1.003</td></tr><tr><td>EstraNet [5]</td><td>5</td></tr><tr><td>PART [6]</td><td>8</td></tr><tr><td></td><td>Ours</td><td>1.000</td></tr><tr><td rowspan="2"></td><td rowspan="2">Byte 2 accuracy ↑</td><td>GPAM [2]</td><td>95.94%</td></tr><tr><td>Ours</td><td>99.97%</td></tr><tr><td rowspan="2">CHES-CTF-2018 [9]</td><td rowspan="2">Byte 0–15 MTD ↓ (best, mean, worst)</td><td></td><td>EFSS (CNN/HW) [7] 317, &gt;1.130k, &gt;3k</td></tr><tr><td>EFSS (MLP/HW) [7] 7, 119.688, 288 Ours</td><td></td></tr></table>

Table 2. Attack performance compared with results reported in prior work. Comparisons are not controlled for compute or hyperparameter tuning budget.

<table><tr><td>Dataset</td><td>Parameters</td><td>FLOPs/step</td><td>Peak VRAM</td><td>Time/run</td></tr><tr><td>ASCADv1f [1]</td><td>26.35M</td><td>4.49T</td><td>8.78GB</td><td>3.34h</td></tr><tr><td>ASCADv1r [1]</td><td>27.88M</td><td>4.49T</td><td>9.07GB</td><td>2.80h</td></tr><tr><td>CHES-CTF-2018 [9]</td><td>16.02M</td><td>205G</td><td>4.20GB</td><td>0.36h</td></tr></table>

Table 3. Computational cost of training runs with recommended hyperparameters.

Evaluation. For all datasets we report the minimum number of attack traces required to correctly predict the key, denoted minimum traces to disclosure (MTD), averaged over 1k random permutations of the attack set. For ASCADv1 our model usually predicts the key from a single trace, so we additionally report the single-trace accuracy. An important consideration for full-key attacks is that per-byte performance does not uniquely determine full-key performance: if A denotes the full-key accuracy and $A _ { i }$ denotes the accuracy for byte i, then $\textstyle 1 - \sum _ { i } ( 1 - A _ { i } ) \leq A \leq \operatorname* { m i n } _ { i } A _ { i }$ . For example, two bytes with 50% accuracy may be jointly predicted with accuracy from 0 to 50% depending on how correlated their errors are. Similarly, if M is the full-key MTD and $M _ { i }$ is the MTD for byte $i ,$ then $M { \geq } \operatorname* { m a x } _ { i } M _ { i }$ . Thus, we report both per-byte and full-key performance.

## 3 Experimental results

We evaluate our pipeline on ASCADv1-fixed, ASCADv1-variable, and CHES-CTF-2018. Table 2 compares our models with results reported in prior work under the corresponding targets and evaluation metrics. Our models achieve comparable or better performance on the reported metrics across all three datasets, establishing that the released models are competitive. In Table 3 we report the parameter count, FLOPs, peak GPU memory, and wall-clock time of each training run under our recommended hyperparameters. Wall clock times are reported on a machine with an

NVIDIA A6000 GPU, AMD Ryzen Threadripper PRO 5965WX CPU, and 128GB of RAM. Complete full-key and per-byte results, reproduction instructions, and additional computational scaling experiments can be found in the repository.

## 4 Limitations

Our experiments are limited to profiled attacks on first-order masked AES-128 implementations, targeting the 16 bytes of the first SubBytes output; performance in other settings remains to be evaluated. Comparisons with prior work are not matched for compute or hyperparameter tuning budget, so the strong performance of our trained models does not establish that the pipeline itself is superior. Observed performance differences may stem from architecture differences, more input features, more hyperparameter tuning, or regularization from jointly predicting multiple bytes. Finally, our experiments were conducted on machines with sufficient RAM for the OS to cache the full datasets. When traces are instead loaded repeatedly from a filesystem, data loading often becomes a bottleneck and substantially increases training time.

## References

1. Benadjila, R., Prouff, E., Strullu, R., Cagli, E., Dumas, C.: Deep learning for side-channel analysis and introduction to ASCAD database. Journal of Cryptographic Engineering 10(2), 163–188 (Jun 2020). https://doi.org/10.1007/s13389-019-00220-8

2. Bursztein, E., Invernizzi, L., Král, K., Moghimi, D., Picod, J.M., Zhang, M.: Generalized power attacks against crypto hardware using long-range deep learning. IACR TCHES 2024(3), 472–499 (2024). https://doi.org/10.46586/tches.v2024.i3.472-499

3. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net (2021), https://openreview.net/forum?id=YicbFdNTTy

4. Egger, M., Schamberger, T., Tebelmann, L., Lippert, F., Sigl, G.: A second look at the ASCAD databases. In: Balasch, J., O’Flynn, C. (eds.) COSADE 2022. LNCS, vol. 13211, pp. 75–99. Springer, Cham (Apr 2022). https://doi.org/10.1007/978-3-030-99766-3\_4

5. Hajra, S., Chowdhury, S., Mukhopadhyay, D.: EstraNet: An efficient shift-invariant transformer network for side-channel analysis. IACR TCHES 2024(1), 336–374 (2024). https://doi.org/10.46586/tches.v2024.i1.336-374

6. Lu, X., Zhang, C., Cao, P., Gu, D., Lu, H.: Pay attention to raw traces: A deep learning architecture for end-to-end profiling attacks. IACR TCHES 2021(3), 235–274 (2021). https://doi.org/10.46586/tches.v2021.i3.235-274, https://tches.iacr.org/index.php/TCHES/article/view/8974

7. Perin, G., Wu, L., Picek, S.: Exploring feature selection scenarios for deep learning-based side-channel analysis. IACR TCHES 2022(4), 828–861 (2022). https://doi.org/10.46586/tches.v2022.i4.828-861

8. Picek, S., Perin, G., Mariot, L., Wu, L., Batina, L.: Sok: Deep learningbased physical side-channel analysis. ACM Comput. Surv. 55(11) (Feb 2023). https://doi.org/10.1145/3569577, https://doi.org/10.1145/3569577

9. Riscure: CHES CTF 2018. https://chesctf.riscure.com/2018/news (2018)

10. Wu, L., Ali-pour, A., Rezaeezade, A., Perin, G., Picek, S.: Breaking free: Leakage model-free deep learning-based side-channel analysis. Cryptology ePrint Archive, Report 2023/1110 (2023), https://eprint.iacr.org/2023/1110