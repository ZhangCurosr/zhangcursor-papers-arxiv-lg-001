# A Width-Matched Comparison of Hybrid Quantum-Classical Self-Supervised Learning for Fingerprint Recognition

Maria S. Edwards1\*, Kidwell Dlamini1\*, Pin-An Lin2\*\*, Wen-Hsien Hsu2\*\* and Wen-Chieh Fang1\*\*\*

1 Department of Computer Science and Information Engineering, National Dong Hwa University, Hualien, Taiwan, wcfang@gms.ndhu.edu.tw

2 Department of Computer Science and Information Engineering, National Chiayi University, Chiayi City, Taiwan

Abstract. Fingerprint recognition is a widely deployed biometric, but supervised training requires large labeled enrollment sets. Self-supervised learning (SSL) removes this requirement, and hybrid quantum-classical models have been proposed to enrich the learned representations. Prior quantum SSL studies consider a single contrastive objective, so it is unclear whether reported benefits depend on the objective or can be attributed to the quantum circuit. We insert the QuFeX quantum featureextraction module into three SSL frameworks, the contrastive SimCLR and MoCo v2 and the non-contrastive BYOL, and compare each hybrid with its classical counterpart at matched representation width (8 features, equal to 8 qubits) on the SOCOFing fingerprint dataset, with a CIFAR-10 control, using k-nearest-neighbor identification on encoder features. In single-run experiments the hybrid scores clearly higher for both contrastive objectives, whereas for BYOL a multi-seed analysis shows no reliable difference, suggesting that any benefit depends on the SSL objective. A hardware-efficient circuit (QNet) does not show the same gain. We examine whether the gains can be attributed to the quantum circuit, considering circuit architecture, trainable parameter count, nonlinearity, and the classical simulability of 8-qubit circuits.

Keywords: Neural Networks · Quantum Computing · Hybrid quantumclassical neural networks · Fingerprint classification.

## 1 Introduction

Fingerprint recognition is among the most widely deployed biometric modalities, used for device unlocking, access control, and identity verification, so high recognition accuracy is important. Supervised approaches, however, require large

\* These authors contributed equally.

\*\* These authors contributed equally.

\* \* \* Corresponding author

labeled enrollment sets that are costly and privacy-sensitive to collect. Selfsupervised learning (SSL) addresses this by learning representations from unlabeled images: contrastive methods such as SimCLR [4] and MoCo v2 [9,5], and the non-contrastive BYOL [8], have narrowed the gap with supervised performance on natural images. In parallel, hybrid quantum-classical neural networks have been proposed to enrich such representations[21,30,1]. Quantum Self-Supervised Learning (QSSL) inserts a parameterized quantum circuit into the encoder of a contrastive model, and prior work reported that a quantum representation network can scores higher than a classical one [14]. Existing hybrid quantum-classical studies, however, are limited to a single contrastive objective [14,18], report a single or mixed accuracy metric [2], use single runs with large variance [20], and do not match the representation width to the qubit count [24]. Additionally, to our knowledge, quantum SSL has not been applied to fingerprint recognition, whose data have distinct characteristics. It therefore remains unclear whether any advantage is (i) specific to the contrastive objective, (ii) present under settings where representation width is matched, and (iii) attributable to the quantum circuit itself rather than to the additional classical parameters (encoding and residual connections) that accompany it, since matching width does not match module capacity.

This paper asks whether integrating a quantum feature-extraction module into self-supervised encoders yields a directionally consistent recognition benefit for fingerprints across different SSL objectives. We insert the QuFeX quantum module [15] between the encoder and projection head of three SSL frameworks-SimCLR, MoCo v2, and BYOL—and compare each hybrid model with its classical counterpart under a matched-width protocol (representation width 8, equal to 8 qubits). We also test an alternative quantum circuit, QNet, on SimCLR as a secondary comparison; unlike QuFeX, QNet does not consistently scores higher the classical baseline at the single-layer depth used throughout our main experiments (Section 5, Appendix A.1), which we use to motivate QuFeX as our primary hybrid module rather than to claim a general quantum advantage. Recognition is evaluated with a k-nearest neighbor (KNN) monitor [6,29], extracted from a publicly available implementation 3 on encoder features (Top-5) on the SOCOFing dataset [25], with a CIFAR-10 control.

Our contributions are:

(1) a unified hybrid framework that inserts the same quantum module into contrastive and non-contrastive SSL encoders;

(2) a fair, width-matched classical-versus-hybrid comparison across three SSL objectives;

(3) the empirical finding that the hybrid QuFeX-based model scores higher than its classical counterpart on KNN recognition across all three objectives and on a CIFAR-10 control, though the margins are modest, obtained from single runs without variance estimates, and come at higher compute cost; a secondary QNet-based comparison does not show the same advantage at the only budget tested (15 epochs). (Section 5).

This paper is organized into seven main sections. First, the introduction outlines the motivation, main objectives, and contributions of the paper. Second, the related works are presented and described. Third, the methodology describes the architecture and structure of the models utilized, as well as the evaluation methods used. Fourth, the experimental settings detail the setup, preprocessing steps, and integration. Fifth, the results obtained from the experiments are presented and explained. Sixth, the discussion aims to explain the reasoning behind the results when using different quantum architectures, the observations obtained after training, and present the limitations of the current models. Finally, the conclusion offers a brief final explanation of the observed results.

## 2 Related Works

## 2.1 Classic Contrastive Classification Methods:

SimCLR [4]: This method does not require additional labeled data. It only applies two different data augmentation methods to pictures, where the same image with different augmentations produces similar results while remaining mutually exclusive with other results to learn data representation. Models using SimCLR typically benefit more from increases in model depth and width, and as complexity increases, the gap with supervised methods decreases. Additionally, increasing batch size and epochs can continuously improve performance, because larger batches and prolonged training time provide more negative samples, promoting model convergence.

MoCo v2 [9,5]: This method was originally proposed to train encoders to encode images as query vectors and key vectors. For matching image pairs, the query vector and key vector become more similar, while non-matching pairs increase their dissimilarity. It incorporates Momentum Contrast, using a queue to replace the memory bank; when the queue is full, the batch-encoded keys obtained from the newest batch data replace the oldest batch-encoded keys, making negative pairs independent of batch size but dependent on queue size. Additionally, the version 2 of MoCo incorporates an MLP projection head.

BYOL: Bootstrap Your Own Latent (BYOL) [8] learns representations without negative pairs, distinguishing it from the contrastive methods above. An online network (encoder, projector, predictor) is trained to predict the projection produced by a target network (encoder, projector) whose weights are an exponential moving average (EMA) of the online network. A stop-gradient on the target, together with the predictor and the slowly updated EMA target, lets BYOL avoid representational collapse without negatives. We adopt the PyTorch implementation available online 4.

In this project, we take all three architectures with a ResNet-18 base model in order to perform our experiments and compare their performance between their classical and hybrid counterparts. The hybrid version is defined by injecting a layer of a quantum circuit in its representation network layer.

## 2.2 Hybrid Classic-Quantum Models

Quantum Self-supervised Learning (QSSL)[14]: Combines SimCLR with a Quantum Neural Network (QNN) by adding a parametrizable quantum circuit to the last layer of the encoded network. QSSL is classified as a Hybrid Quantum-Classical Algorithm (HQC) [14]. HQCs consist of three elements: First, selecting a cost function based on the problem; second, selecting a quantum circuit (ansatz) based on the problem. The cost function parameters serve as parameters (variables) for the quantum gates in the ansatz, and finally a classical optimizer maximizes or minimizes the cost function[3,23].

QuFeX[15]: Based on two important implementations of quantum circuits with convolutional neural networks, QCNN [7] and QuanNN[11]. It comprises QCNNs circuit design, combining convolution and pooling using parameterized units, and QuanNNs data handling mechanisms, but instead of scanning small local patches one at a time, it lets a single quantum circuit process a mix of multiple feature maps in parallel. In addition, QuFeX preserves the qubits output as part of the feature map. Qu-Net is a standard U-Net where its bottleneck layer is replaced by a QuFeX quantum layer and classical residual connections are added around the quantum module.

We take both circuit designs and apply them to the classical models selected in order to compare the performance of different architectures when evaluated against the same task, in this case, fingerprint recognition.

## 2.3 Fingerprint Image Preprocessing

Gabor Filter [13]: The Fourier transform is a powerful tool in signal processing that can help us convert images from the spatial domain to the frequency domain and extract features that are difficult to extract in the spatial domain. However, after the Fourier transform, frequency features at different positions in the image often mix together, but Gabor filters can extract local spatial frequency features and are effective texture detection tools. Based on the directional and frequency characteristics of sine waves, texture details of different directions and sizes can be obtained.

Canny Edge Detection [26]: Canny Edge Detection is a preprocessing method for extracting photo edges and reducing noise, helping extract fingerprint edge contours, making these edge features easier for models to recognize while preserving important shapes and structures. We perform the data preprocessing and enhancement using the previously described methods.

## 3 Method

## 3.1 Problem Formulation

Let $\mathcal { D } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ be a set of unlabeled fingerprint images from $C = 6 0 0$ identities. Self-supervised pre-training learns an encoder $f _ { \theta }$ that maps an image to a representation $y = f _ { \theta } ( x ) \in \mathbb { R } ^ { w }$ without using identity labels, by optimizing an SSL objective over augmented views. The frozen representation is then evaluated on identification: for a query image $x _ { q }$ with representation $y _ { q } .$ a k-nearest-neighbor classifier over a labeled gallery predicts the identity, and we report Top-5 accuracy over the held-out test identities. Our question is whether replacing the classical representation network with a parameterized quantum circuit $Q$ of the same width w (so that w equals the qubit count) improves identification, and whether any such improvement is consistent in direction across contrastive and non-contrastive SSL objectives. Within each classical/hybrid pair, we hold the backbone, w, the data split, and all training hyperparameters fixed, so the representation network is the only component that changes.

## 3.2 Encoder and hybrid representation network

Figure 1 shows the overall architecture. All models share a ResNet-18 backbone adapted for single-channel input (the first convolution takes one channel), producing a 512-dimensional feature. A linear layer compresses this to a representation of width $w ;$ we fix $w = 8$ to match the qubit count of the hybrid models. In the classical model, the representation network is a stack of linear layers with LeakyReLU; in the hybrid model, it is the QuFeX quantum module or QNet circuit, depending on the experiment. We did not search over classical alternatives (deeper/wider MLPs, or an MLP with a matching residual connection) that might close the gap without a quantum circuit; this capacity-matched ablation is left to future work. A projection head maps the representation for the SSL loss. For all KNN evaluations, we use the encoder features before the projection head, never the projector output.

For the QNet implementation, the quantum circuit is based on a hardwareefficient ansatz (HEA) [16], specifically "Circuit $1 4 "$ [27]. A HEA is a type of quantum circuit that minimizes the number of two-qubit entangling gates, respects fixed hardware connectivity, and is highly practical, as it is not designed to be problem-specific. In "Circuit 14"'s topology, each qubit connects to its nearest neighbor and the qubit three positions away, forming a circulant graph rather than a ring. This topology creates more entanglement between qubits without it being too expensive; it is trainable in strength due to its usage of parameterized gates, and allows the entangling gates to be reoriented. For the QuFex circuit, it was built based on a QCNN-structured circuit without qubit discarding. Circuits are classically simulated, not run on real hardware; noise and decoherence are not modeled.

![](images/5bd3005b50d3a4c60100547adf7919cc474422bbded8a07bab9d6997eedd59f0.jpg)  
Encoder: shared (SimCLR, MoCo) / online + EMA target (BYOL)

Fig. 1. Overall architecture. Each preprocessed fingerprint $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ (Gabor + Canny) is augmented into two views $\mathbf { \boldsymbol { x } } _ { i } ^ { 1 } , \mathbf { \boldsymbol { x } } _ { i } ^ { 2 }$ and passed through a shared encoder: a CNN (ResNet-18) backbone followed by a representation network that is either a classical MLP or the quantum module (QuFeX / QNet). A projection head $g$ maps representations to $z _ { i } ^ { 1 } , z _ { i } ^ { 2 }$ for the self supervised objective. For SimCLR and MoCo $\mathbf { v } 2 ,$ the objective is contrastive with weight sharing; In BYOL, the target branch is an EMA target, the online branch adds a predictor $q ,$ and a stop gradient $( s g )$ is applied to the EMA target. The KNN monitor evaluates the representation $\mathbf { \Delta } y _ { i } ^ { 1 }$ before the projection head.

## 3.3 Comparison of Backbone Circuits: Expressibility, Entanglement, and Residual Design

QNet [14] uses "Circuit 14" from [27] and it is designed to be a generic and flexible building block, has a high entangling capability, and is close to an unstructured MLP, expressing no inductive bias towards any particular data structure. QuFex [15], on the other hand, is inspired by QNN [7], with an explicit convolution-then-pooling hierarchy, following a classical CNN's sample efficiency relative to MLPs.

QNets CRX is a parameterized two-qubit gate; the entangling strength between each pair is itself a trainable parameter, continuously interpolating from no entanglement to maximal. While QuFexs CNOT/CZ gates are fixed, nonparameterized, maximally entangling operations. Only single-qubit rotations riding on top of them are trainable.

QNet has high expressibility and therefore can be plateau-prone, as explained in [19,12]; however, QCNN architectures may avoid the exponential vanishing-gradient problem [22]; therefore, QuFex is structurally restricted and more plateau-resistant in theory for being QCNN-inspired, although not conclusive.

QNet addresses the issue of classic quantum circuits' linearity in pure unitary evolutions by adding a mid-circuit projective measurement; collapsing part of the state between layers is a way of injecting a non-unitary, non-linear operation into a linear circuit. QuFex is fully unitary until the final measurement and is strictly a linear quantum feature map, with all of its practical nonlinearity coming from the classical residual add and the encoding step.

QuFex's residual output composition, based on [10] skip connections, makes it easier to learn a perturbative refinement instead of a full transformation, and

smooth the loss [17]:

$$
y = Q ( x ) + x\tag{1}
$$

In hybrid quantum-classical settings, the residual approach gives the network a trainable fallback to the identity map when the quantum component is poorly optimized; the classical signal is never really lost and gated behind the quantum performance. This also means that QuFeX's advantage over the classical baseline, where observed, cannot be cleanly separated from the benefit of the residual connection itself: a classical representation network with the same residual structure but no quantum circuit is a natural control that we do not test. On the other hand, QNet's:

$$
x = f ( x )\tag{2}
$$

has no fallback; the entire downstream representation is sent to the quantum channel, so any issues in the quantum layer can directly degrade the representation with nothing to fall back on.

## 4 Experimental Settings

## 4.1 Setup

All models, including SimCLR, MoCo v2, and BYOL, use ResNet-18, batch size 128, width 8, and the Adam optimizer (learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 6 } )$ . BYOL is trained for 200 epochs and MoCo v2 for 50 epochs; the SimCLR figures are from a separately trained run under the same backbone and width. Each configuration is a single run. As a control, we repeat the hybrid/classical comparison on CIFAR-10 (10 classes; random baseline Top-5 50%). All models were run using a CPU Intel Xeon W5-3435X 3.10GHz, RAM Crucial ECC REG DDR5 64G, GPU NVIDIA RTX PRO 6000 96GB.

## 4.2 Dataset and preprocessing

We use the SOCOFing fingerprint dataset [25]: 6,000 grayscale images from 600 subjects (10 prints each), expanded through data augmentation. After removing excessively distorted samples, 29,514 images remain, which undergo two-stage preprocessing with a Gabor filter [13] 5 and Canny edge detection[26]. The Gabor filter extracts local spatial-frequency and directional texture; Canny edge detection extracts ridge contours while reducing noise, preserving the shapes most useful for recognition. The data are split 8:1:1 into training, validation, and test sets over the 600 identities.

## 4.3 Integration into SSL frameworks and evaluation

The same hybrid representation network is inserted into each framework. Sim-CLR and MoCo v2 use a contrastive loss over positive and negative pairs; MoCo v2 maintains a momentum encoder and a queue of negative keys. BYOL uses no negatives, with an EMA target $( m = 0 . 9 9 6 )$ , a 2 layer projector (dimension 128, hidden 512), and a 2 layer predictor. Here, m denotes the momentum coefficient governing the EMA update of the target network's weights, $\xi  m \xi + ( 1 - m ) \theta$ , where θ are the online network's parameters. Recognition is measured by a KNN monitor over encoder features, reporting Top-5. For 600 classes the random baseline is $\mathrm { T o p - 5 } \approx 0 . 8 3 \%$

For a fair quantum versus classical comparison, we fix the width $w = 8 \ ( = 8$ qubits), the backbone, the dataset split, the image size, and the batch size within each hybrid/classical pair. Classical width-16/32/64 runs are diagnostic only and are not part of the fair comparison.

## 5 Results

## 5.1 Classical and Hybrid Self-Supervised Models Comparison

For the first experiment, we compared the performance of the classical and hybrid (QuFex) versions of the three tested models, SimCLR, MoCo v2, and BYOL, using a training dataset of approximately 30, 000 images and a testing set of 3900 images. SimCLR and MoCo v2 were trained for 50 epochs and BYOL for 200 epochs. The addition of the quantum layer is expected to show an increase in performance compared to its classical counterpart.

Table 1 reports test KNN accuracy. In every QuFeX setting, the hybrid scores higher than its classical counterpart on Top-5. On fingerprints, SimCLR Top-5 rises from 48.43% to 57.06%; MoCo v2 improves Top-5 from 28.22% to 40.30%; and BYOL improves Top-5 from 23.70% to 25.55%. The same direction holds on CIFAR-10 (Top-5 97.03% → 97.52%). The improvement is thus consistent in sign for QuFeX across objectives and datasets (QNet, below, is not), although its magnitude varies and is smallest for BYOL. Given single runs, smaller margins may reflect run-to-run noise. The hybrid models cost 1.5× to 6× more per epoch, so gains are not at equal compute. BYOL's lower absolute accuracy may reflect its lack of explicit negative pairs in a fine-grained, 600- identity task [28]; since hyperparameters were not matched across objectives, we compare classical and hybrid models only within each objective.

Training dynamics. Figures 2-4 show the per-epoch behaviour behind the final numbers in Table 1, which is evidence that the gap is not a single-epoch artefact, though still from one seed. For MoCo v2 (Fig. 2), the hybrid model separates from the classical baseline within the first ten epochs and stays above it for the entire run on validation Top-5; it also converges faster, peaking near epoch 20 while the classical model is still improving. Its InfoNCE training loss (Fig. 3) is correspondingly lower, so the training objective and the recognition metrics agree. For BYOL (Fig. 4 (a)) the hybrid rises smoothly and monotonically on fingerprints, whereas the classical model dips to about 13% Top-5 near epoch 15 before recovering, so the quantum module also yields more stable training, not only higher accuracy. On the CIFAR-10 control (Fig. 4 (b)) both models saturate near 97.5% Top-5, but the hybrid reaches that plateau earlier. Across a contrastive method (MoCo v2) and a non-contrastive one (BYOL), and across two datasets, the hybrid curve lies above its classical counterpart for essentially all of training, reinforcing Table 1.'s sign, though not its significance.

Table 1. Test KNN Top-5 accuracy (%) on encoder features (before the projection head). Hybrid uses the QuFeX quantum module; width = 8 (= 8 qubits) for the fingerprint runs.Training budgets: SimCLR 100 epochs, MoCo v2 50 epochs, BYOL 200 epochs; the CIFAR-10 control uses BYOL trained for 200 epochs. All values are from single runs, taken from the final-epoch model. Better value in each classical/hybrid pair in bold.
<table><tr><td>Method (variant)</td><td>Top-5</td></tr><tr><td colspan="2">SOCOFing (600 cls.; rand. 0.83) SimCLR, classical</td></tr><tr><td>SimCLR, hybrid MoCo v2, classical MoCo v2, hybrid BYOL, classical</td><td>48.43 57.06 28.22 40.30 23.70 25.55</td></tr><tr><td colspan="2">BYOL, hybrid CIFAR-10 (10 cls.; rand. 50)</td></tr><tr><td>BYOL, classical BYOL, hybrid (QuFeX)</td><td>97.03 97.52</td></tr></table>

## 5.2 Classical and Hybrid SimCLR with different epoch lengths

For the second experiment, we compared the performance of the classical and hybrid (QuFex) versions of SimCLR when running at different epoch lengths. Additionally, we tested the performance of SimCLR with one layer using QNet as the quantum circuit; however, in this instance, it was only tested for 15 epochs due to time and hardware constraints. This held for QuFeX but not QNet. Results are presented in the table 2, and figures 5 and 6

We additionally report two diagnostic experiments in Appendix A: an upstream depth sweep and a downstream linear-versus-KNN comparison for the QNet-based SimCLR. They motivate the width-matched QuFeX study here, since they show that a quantum representation network helps only at depth ≥ 2 and scores lower the classical baseline at a single layer (Appendix Table 8); this is why we adopt QuFeX for the single-layer injection used throughout the main experiments.

Table 2 shows the highest percentage obtained with each model definition after running 15, 50, 100, and 200 epochs. Due to time and hardware constraints, the SimCLR-Classic model with 200 epochs, as well as the original version of

![](images/8b21c45e8d3e33daa1c3ddede50755c5e6f1f8762e6b0bb31ce335207f9e2b73.jpg)  
Fig. 2. MoCo v2 on fingerprints (SOCOFing, 50 epochs), classical vs. hybrid (QuFeX) at width 8. The hybrid separates from the classical baseline within the first ten epochs and stays above it for the whole run on validation Top-5, so the advantage is not an artefact of the single best epoch. Markers are single-run values (seed 0); error bars show ±1 s.d. of the metric over a forward window of a few consecutive epochs, i.e. within-run epoch-to-epoch volatility, not confidence intervals, across-seed variance, or the significance of the hybrid-classical gap.

Table 2. Highest Percentage of Top-5 Test Accuracy in SimCLR classical and hybrid models. In bold, the highest result obtained from all the runs.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>15 Epochs</td><td rowspan=1 colspan=1>50 Epochs</td><td rowspan=1 colspan=1>100 Epochs</td><td rowspan=1 colspan=1>200 Epochs</td></tr><tr><td rowspan=1 colspan=1>SimCLR-Classic</td><td rowspan=1 colspan=1>34.98</td><td rowspan=1 colspan=1>44.25</td><td rowspan=1 colspan=1>48.43</td><td rowspan=1 colspan=1>51.23</td></tr><tr><td rowspan=1 colspan=1>SimCLR-QuFex</td><td rowspan=1 colspan=1>45.49</td><td rowspan=1 colspan=1>55.25</td><td rowspan=1 colspan=1>57.06</td><td rowspan=1 colspan=1>56.13</td></tr><tr><td rowspan=1 colspan=1>SimCLR-QNet</td><td rowspan=1 colspan=1>32.27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Table 3. Highest Percentage of Top-5 Test Accuracy in MoCo v2 classical and hybrid models. In bold, the highest result obtained from all the runs.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>15 Epochs</td><td rowspan=1 colspan=1>50 Epochs</td><td rowspan=1 colspan=1>100 Epochs</td><td rowspan=1 colspan=1>200 Epochs</td></tr><tr><td rowspan=1 colspan=1>MoCo-Classic</td><td rowspan=1 colspan=1>21.28</td><td rowspan=1 colspan=1>28.22</td><td rowspan=1 colspan=1>41.91</td><td rowspan=1 colspan=1>47.77</td></tr><tr><td rowspan=1 colspan=1>MoCo-QuFex</td><td rowspan=1 colspan=1>30.52</td><td rowspan=1 colspan=1>40.30</td><td rowspan=1 colspan=1>45.03</td><td rowspan=1 colspan=1>51.79</td></tr></table>

QSSL (using QNet as the quantum layer) was only run for 15 epochs, so its converged performance is unknown.

In the QuFeX instances, the SimCLR-QuFex model consistently scores higher than the other model variations, with the highest result obtained at 57%. QNet scores lower than classical SimCLR at its only tested budget (Table 2). The same pattern can be found in the MoCo v2 experiments (Table 3 where the QuFex variant scores higher than the classical variant, with the highest result obtained at approximately 52%. However, it is also observable that the advantage presented by the MoCo-QuFex model for the larger number of epochs against its classical counterpart is not as pronounced as the smaller number of epochs

![](images/b17bf52c6045e83cadd00b0e6cde9d48493e34d1676287a3f1602594b4e40380.jpg)  
Fig. 3. MoCo v2 training loss (InfoNCE) on fingerprints, classical vs. hybrid (QuFeX). The hybrid loss is consistently lower, so loss and recognition (Fig. 2) agree. Markers are single-run values (seed 0); error bars show ±1 s.d. of the loss over a forward window of ≈20 consecutive batches, i.e. within-run batch-to-batch volatility inflated by the training-curve slope, not confidence intervals or across-seed variance.

![](images/3336132f5a4f28e558db2aeb2b89480e6cd955ecbfc2dbe30b69096c907a0845.jpg)  
Fingerprint (SOCOFing), 200 epochs

![](images/d9caec6e3cb7752ac22b7473bdffbe901ba4facd88d46a80f0b8f2094799283e.jpg)  
CIFAR-10 control, 200 epochs  
Fig. 4. BYOL validation Top-5 kNN, classical vs. hybrid (QuFeX). (a) On fingerprints, the hybrid rises smoothly while the classical model dips to about 13% near epoch 15 before recovering. (b) On the CIFAR-10 control both saturate near 97.5%, but the hybrid reaches the plateau earlier. Markers are single-run values (seed 0); error bars show ±1 s.d. over a forward window of a few consecutive epochs (within-run volatility), not confidence intervals, across-seed variance, or gap significance.

Tables 4, 5, 6 and 7 denote the loss and accuracy values obtained at specific epoch numbers during training for both contrastive models, SimCLR and MoCo v2, comparing the hybrid and classical versions of each architecture. From all the results we can observe, the QuFex version in all instances obtains the highest accuracy and lowest loss regardless of the epoch number.

The graphical results of the model's performance when injecting one layer of QuFex into a classic SimCLR and MoCo v2 architectures are shown in Fig. 5 and Fig. 7, where it is also observable the stable and consistent decrease in the training loss for all models. While for the classical SimCLR and MoCo v2 models in Fig. 6 and Fig. 8, we can observe the same consistent behavior but at a slower rate, meaning it takes a larger number of epochs for it to reach the same level of accuracy as the hybrid models. As with the other results in this section, all figures report single training curves without repeated-seed variance bands.

Table 4. Comparison Table for epochs=15 denoting the testing values at specific epoch numbers
<table><tr><td rowspan=1 colspan=9>15-Epochs</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>SimCLR</td><td rowspan=1 colspan=4>MoCo</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td></tr><tr><td rowspan=1 colspan=1>Epoch Nř</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>1 Epoch</td><td rowspan=1 colspan=1>3.51</td><td rowspan=1 colspan=1>23.47</td><td rowspan=1 colspan=1>3.27</td><td rowspan=1 colspan=1>24.32</td><td rowspan=1 colspan=1>6.39</td><td rowspan=1 colspan=1>11.28</td><td rowspan=1 colspan=1>6.33</td><td rowspan=1 colspan=1>14.53</td></tr><tr><td rowspan=1 colspan=1>10 Epoch</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>34.75</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>44.41</td><td rowspan=1 colspan=1>4.31</td><td rowspan=1 colspan=1>18.90</td><td rowspan=1 colspan=1>3.93</td><td rowspan=1 colspan=1>22.75</td></tr><tr><td rowspan=1 colspan=1>15 Epoch</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>34.55</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>45.24</td><td rowspan=1 colspan=1>4.02</td><td rowspan=1 colspan=1>21.27</td><td rowspan=1 colspan=1>3.68</td><td rowspan=1 colspan=1>30.52</td></tr></table>

Table 5. Comparison Table for epochs=50 denoting the testing values at specific epoch numbers
<table><tr><td rowspan=1 colspan=9>50-Epochs</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>SimCLR</td><td rowspan=1 colspan=4>MoCo</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td></tr><tr><td rowspan=1 colspan=1>Epoch Nř</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>1 Epoch</td><td rowspan=1 colspan=1>3.56</td><td rowspan=1 colspan=1>16.65</td><td rowspan=1 colspan=1>3.17</td><td rowspan=1 colspan=1>20.48</td><td rowspan=1 colspan=1>6.70</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>6.41</td><td rowspan=1 colspan=1>2.27</td></tr><tr><td rowspan=1 colspan=1>30 Epoch</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>42.89</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>51.61</td><td rowspan=1 colspan=1>3.26</td><td rowspan=1 colspan=1>23.65</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>34.29</td></tr><tr><td rowspan=1 colspan=1>50 Epoch</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>43.71</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>51.85</td><td rowspan=1 colspan=1>2.90</td><td rowspan=1 colspan=1>28.22</td><td rowspan=1 colspan=1>2.47</td><td rowspan=1 colspan=1>40.07</td></tr></table>

Table 6. Comparison Table for epochs=100 denoting the testing values at specific epoch numbers
<table><tr><td rowspan=1 colspan=9>100-Epochs</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>SimCLR</td><td rowspan=1 colspan=4>MoCo</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td></tr><tr><td rowspan=1 colspan=1>Epoch Nř</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>1 Epoch</td><td rowspan=1 colspan=1>3.69</td><td rowspan=1 colspan=1>16.11</td><td rowspan=1 colspan=1>3.23</td><td rowspan=1 colspan=1>25.33</td><td rowspan=1 colspan=1>7.43</td><td rowspan=1 colspan=1>2.71</td><td rowspan=1 colspan=1>6.35</td><td rowspan=1 colspan=1>7.59</td></tr><tr><td rowspan=1 colspan=1>30 Epoch</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>41.11</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>49.19</td><td rowspan=1 colspan=1>3.09</td><td rowspan=1 colspan=1>31.52</td><td rowspan=1 colspan=1>2.67</td><td rowspan=1 colspan=1>39.30</td></tr><tr><td rowspan=1 colspan=1>50 Epoch</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>45.00</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>53.40</td><td rowspan=1 colspan=1>2.66</td><td rowspan=1 colspan=1>41.23</td><td rowspan=1 colspan=1>2.47</td><td rowspan=1 colspan=1>43.12</td></tr><tr><td rowspan=1 colspan=1>70 Epoch</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>44.67</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>55.51</td><td rowspan=1 colspan=1>2.47</td><td rowspan=1 colspan=1>41.60</td><td rowspan=1 colspan=1>2.34</td><td rowspan=1 colspan=1>45.03</td></tr><tr><td rowspan=1 colspan=1>100 Epoch</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>46.73</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>56.44</td><td rowspan=1 colspan=1>2.35</td><td rowspan=1 colspan=1>41.57</td><td rowspan=1 colspan=1>2.21</td><td rowspan=1 colspan=1>43.66</td></tr></table>

## 6 Discussion

## 6.1 A consistent but modest quantum benefit

The central observation is one of directional consistency: inserting the QuFeX module improves KNN recognition in every setting we tested: the contrastive

Table 7. Comparison Table for epochs=200 denoting the testing values at specific epoch numbers
<table><tr><td rowspan=1 colspan=9>200-Epochs</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>SimCLR</td><td rowspan=1 colspan=4>MoCo</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td><td rowspan=1 colspan=2>Classical</td><td rowspan=1 colspan=2>Hybrid</td></tr><tr><td rowspan=1 colspan=1>Epoch Nř</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Loss</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>1 Epoch</td><td rowspan=1 colspan=1>3.66</td><td rowspan=1 colspan=1>15.34</td><td rowspan=1 colspan=1>3.50</td><td rowspan=1 colspan=1>20.68</td><td rowspan=1 colspan=1>6.57</td><td rowspan=1 colspan=1>3.74</td><td rowspan=1 colspan=1>6.28</td><td rowspan=1 colspan=1>22.10</td></tr><tr><td rowspan=1 colspan=1>50 Epoch</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>40.72</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>52.96</td><td rowspan=1 colspan=1>2.54</td><td rowspan=1 colspan=1>38.29</td><td rowspan=1 colspan=1>2.35</td><td rowspan=1 colspan=1>47.25</td></tr><tr><td rowspan=1 colspan=1>100 Epoch</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>44.02</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>54.04</td><td rowspan=1 colspan=1>22.25</td><td rowspan=1 colspan=1>43.09</td><td rowspan=1 colspan=1>2.21</td><td rowspan=1 colspan=1>46.24</td></tr><tr><td rowspan=1 colspan=1>150 Epoch</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>47.61</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>55.02</td><td rowspan=1 colspan=1>2.15</td><td rowspan=1 colspan=1>43.66</td><td rowspan=1 colspan=1>2.14</td><td rowspan=1 colspan=1>48.25</td></tr><tr><td rowspan=1 colspan=1>200 Epoch</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>48.67</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>51.43</td><td rowspan=1 colspan=1>2.09</td><td rowspan=1 colspan=1>47.68</td><td rowspan=1 colspan=1>2.10</td><td rowspan=1 colspan=1>46.94</td></tr></table>

SimCLR and MoCo v2, the non-contrastive BYOL, and the CIFAR-10 control QNet did not show this pattern at the single-layer depth tested (Section 5) Because BYOL uses no negative pairs, this indicates that the benefit is not a property of contrastive repulsion but rather of the representation produced by the quantum module. QuFeX also adds a residual connection and encoding step beyond the classical baseline, so this benefit may not be purely quantum in origin. Beyond final accuracy, the hybrid also converges faster and, for BYOL, trains more stably (Figs. 2–4). At the same time, we do not claim a large or state-of-the-art improvement: absolute fingerprint accuracy is modest, and the per-setting margins are small, from about +0.49 Top-5 for BYOL to about +11 Top-5 for SimCLR.

## 6.2 Why absolute accuracy is modest, and QNet versus QuFeX

Fingerprint recognition here is a 600-class problem on small grayscale images, compressed through an 8 dimensional (8 qubit) bottleneck, with a limited epoch budget for the quantum models; these factors bound absolute accuracy for both classical and hybrid models. We used QuFeX rather than QNet because QuFeX is structurally restricted and, being QCNN-inspired, is expected to be more resistant to barren plateaus [22], whereas the highly expressive QNet can be plateau prone [19]; consistent with this, our depth sweep (Appendix Table 8) shows QNet helping only at depth ≥ 2 and scores lower than the classical baseline at the single layer we use. QuFeX's residual composition (Eq. 1) further provides a trainable fallback that QNet (Eq. 2) lacks. Completing the compute-limited QNet runs and comparing the two circuits empirically is left to future work.

## 6.3 Limitations

First, each configuration is a single run (seed 0); the bands in Figs. 2–4 show only within-run batch/epoch volatility, so we cannot report across-seed variance, and the smaller margins (notably BYOL Top-5) are within plausible run-to-run variance. It is the agreement in direction across independent settings, rather than any single margin, that supports the claim. This is thus a claim about effect sign, not statistical significance. Second, the hybrid adds the entire QuFeX module (quantum circuit, encoding, and classical residual); although the representation width is matched, the module's capacity is not, so part of the gain may be attributable to added capacity rather than to the quantum circuit itself. A capacity-matched classical module is the natural ablation. We do not run it here. Third, the benefit comes at a compute cost of 1.5× to 6× per epoch under simulation, so it is not a compute saving. Fourth, circuits are classically simulated, not run on real hardware. Fifth, epoch budgets differ across objectives (50, 200, 15), confounding cross-objective margin comparisons. Sixth, QNet scored lower than classical SimCLR at matched depth, so the positive claim is specific to QuFeX.

![](images/fd9da491c92ad7f3fc7bbe5bea07dd14432ed5eb9480afcd98f665429944d4e1.jpg)  
Fig. 5. Hybrid Quantum-Classical accuracy versus loss for SimCLR models with one QuFex layer injected at different epoch sizes.

![](images/3758b8720e5cb3e740c4a5151f95e851818080cceccdd122a8fbfee6cef788a6.jpg)  
Fig. 6. Classical SimCLR accuracy versus loss at different epoch sizes.

![](images/2db61f3f4880f725db6be982bf91ef75b73265b147b1ef8bba60816c97b5f6cc.jpg)

Fig. 7. Hybrid Quantum-Classical accuracy versus loss for MoCo v2 models with one QuFex layer injected at different epoch sizes.  
![](images/a6c7744ae302a82e4dd702d211ae03bc7d586d2eeee36156594bd37988a46e3a.jpg)  
Fig. 8. Classical MoCo v2 accuracy versus loss at different epoch sizes.

## 7 Conclusion

We presented a unified hybrid framework that inserts the same quantum feature extraction module into three self-supervised objectives: SimCLR, MoCo v2, and BYOL for fingerprint recognition. Under a width-matched comparison, the hybrid model consistently scored higher its classical counterpart on KNN Top-5 across all three objectives and on a CIFAR-10 control, indicating that the benefit of the quantum module is not specific to the contrastive objective. The margins are modest and obtained from single runs at higher compute cost. Confirming them with multiple seeds and error bars, isolating the quantum contribution with capacity-matched ablations, and completing the QNet runs are the main directions for future work.

The results presented in this article represent only the performance demonstrated by the model when applied to a total of one layer in the base model. Additionally, the adapted version of QuFex utilized in this project does not take into account multi-layering. In future work, it is expected to obtain results when the quantum layer injection is applied to more than one layer.

## A Appendix

## A.1 Upstream SimCLR testing at different depths

SimCLR upstream training produces a fingerprint feature representation, using augmented photos during training as ground truth, then calculating representation accuracy with model-predicted photos. Table 8 shows that when quantum computing is added to the network with depth greater than or equal to 2.

Table 8. Average Training Top-5 Accuracy Comparison between Classical SimCLR and Hybrid SimCLR-QNet, tested with 3 different representation layers.
<table><tr><td rowspan=1 colspan=1>Avg. Top-5 Accuracy (%)</td><td rowspan=1 colspan=1>[Classical SimCL</td><td rowspan=1 colspan=1>R Hybrid SimCLR-QNet</td></tr><tr><td rowspan=1 colspan=1>Repr. Layer with 3 layers</td><td rowspan=1 colspan=1>56.62</td><td rowspan=1 colspan=1>66.96</td></tr><tr><td rowspan=1 colspan=1>Repr. Layer with 2 layers</td><td rowspan=1 colspan=1>53.65</td><td rowspan=1 colspan=1>65.36</td></tr><tr><td rowspan=1 colspan=1>Repr. Layer with 1 layer</td><td rowspan=1 colspan=1>62.29</td><td rowspan=1 colspan=1>54.62</td></tr></table>

## A.2 SimCLR Downstream Verification Accuracy

Downstream testing of upstream training results has two methods: Method 1 discards the Projection Head and adds a linear network layer after the Encoder, called linear evaluation; Method 2 takes out the upstream Encoder and verifies results through KNN Monitor. Table 9

Table 9. Testing Top-5 Accuracy Comparison between Classical SimCLR and Hybrid SimCLR-QNet, using two different evaluation methods
<table><tr><td rowspan=1 colspan=1>Top-5 Accuracy (%)</td><td rowspan=1 colspan=1>Classical SimCLR</td><td rowspan=1 colspan=1>|Hybrid SimCLR-QNet</td></tr><tr><td rowspan=1 colspan=1>Linear Evaluation</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>KNN Monitor</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>26</td></tr></table>

## References

1. Abbas, A.: Tunnelqnn: A hybrid quantum-classical neural network for efficient learning. arXiv preprint arXiv:2505.00933 (2025)

2. Bowles, J., Ahmed, S., Schuld, M.: Better than classical? the subtle art of benchmarking quantum machine learning models. arXiv preprint arXiv:2403.07059 (2024)

3. Cerezo, M., Arrasmith, A., Babbush, R., Benjamin, S.C., Endo, S., Fujii, K., Mc-Clean, J.R., Mitarai, K., Yuan, X., Cincio, L., et al.: Variational quantum algorithms. Nature Reviews Physics 3(9), 625–644 (2021)

4. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International conference on machine learning. pp. 1597–1607. PmLR (2020)

5. Chen, X., Fan, H., Girshick, R., He, K.: Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297 (2020)

6. Chen, X., He, K.: Exploring simple siamese representation learning. In: 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR). pp. 15745–15753. IEEE (2021)

7. Cong, I., Choi, S., Lukin, M.D.: Quantum convolutional neural networks. Nature Physics 15(12), 1273–1278 (2019)

8. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al.: Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems 33, 21271–21284 (2020)

9. He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.: Momentum contrast for unsupervised visual representation learning. In: 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR). pp. 9726–9735. IEEE (2020)

10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

11. Henderson, M., Shakya, S., Pradhan, S., Cook, T.: Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence 2(1), 2 (2020)

12. Holmes, Z., Sharma, K., Cerezo, M., Coles, P.J.: Connecting ansatz expressibility to gradient magnitudes and barren plateaus. PRX quantum 3(1), 010313 (2022)

13. Hong, L., Wan, Y., Jain, A.: Fingerprint image enhancement: algorithm and performance evaluation. IEEE transactions on pattern analysis and machine intelligence 20(8), 777–789 (1998)

14. Jaderberg, B., Anderson, L.W., Xie, W., Albanie, S., Kiffner, M., Jaksch, D.: Quantum self-supervised learning. Quantum Science & Technology 7(3), 035005 (2022)

15. Jain, N., Kalev, A.: Qufex: quantum feature extraction module for hybrid quantumclassical deep neural networks. Quantum Science and Technology 11(1), 015017 (2026)

16. Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J.M., Gambetta, J.M.: Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. nature 549(7671), 242–246 (2017)

17. Li, H., Xu, Z., Taylor, G., Studer, C., Goldstein, T.: Visualizing the loss landscape of neural nets. Advances in neural information processing systems 31 (2018)

18. Li, L., Ni, X., Li, J., Qin, S., Gao, F.: Qsea: Quantum self-supervised learning with entanglement augmentation. Advanced Quantum Technologies 9(4), e00530 (2026)

19. McClean, J.R., Boixo, S., Smelyanskiy, V.N., Babbush, R., Neven, H.: Barren plateaus in quantum neural network training landscapes. Nature communications 9(1), 4812 (2018)

20. Meyer, N., Ufrecht, C., Yammine, G., Kontes, G., Mutschler, C., Scherer, D.D.: Benchmarking quantum reinforcement learning. arXiv preprint arXiv:2501.15893 (2025)

21. Monbroussou, L., Periyasamy, M., Kuzmin, V., Sekatski, P., Patapovich, V., Sagingalieva, A., Melnikov, A.: Hybrid quantum neural networks: Theory, implementations, and applications. arXiv preprint arXiv:2608.01194 (2026)

22. Pesah, A., Cerezo, M., Wang, S., Volkoff, T., Sornborger, A.T., Coles, P.J.: Absence of barren plateaus in quantum convolutional neural networks. Physical Review X 11(4), 041011 (2021)

23. Qi, H., Xiao, S., Liu, Z., Gong, C., Gani, A.: Variational quantum algorithms: fundamental concepts, applications and challenges: H. qi et al. Quantum Information Processing 23(6), 224 (2024)

24. Sarkar, S.: Quantum transfer learning for mnist classification using a hybrid quantum-classical approach. arXiv preprint arXiv:2408.03351 (2024)

25. Shehu, Y.I., Ruiz-Garcia, A., Palade, V., James, A.: Sokoto coventry fingerprint dataset. arXiv preprint arXiv:1807.10609 (2018)

26. Shinde, K.K., Kayte, C.N.: Fingerprint recognition based on deep learning pretrain with our best cnn model for person identification. Electrochemical Society Transactions 107(1), 2209–2220 (2022)

27. Sim, S., Johnson, P.D., Aspuru-Guzik, A.: Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms. Advanced Quantum Technologies 2(12), 1900070 (2019)

28. Wang, T., Isola, P.: Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In: International Conference on Machine Learning. pp. 9929–9939 (2020)

29. Wu, Z., Xiong, Y., Yu, S.X., Lin, D.: Unsupervised feature learning via nonparametric instance discrimination. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3733–3742 (2018)

30. Zaman, K., Ahmed, T., Hanif, M.A., Marchisio, A., Shafique, M.: A comparative analysis of hybrid-quantum classical neural networks. In: World Congress in Com-

puter Science, Computer Engineering & Applied Computing. pp. 102–115. Springer (2024)