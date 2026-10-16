# FEDERATED ART CHASE: PRIVACY LEAKAGE AND DATA POISONING IN FEDERATED FINE-TUNING

4TH FEDERATED MACHINE LEARNING INTERNATIONAL SUMMER SCHOOL

Vasileios Perifanis Indigma Innovations Greece vasilis@indigma.eu

Nikolaos Pavlidis Indigma Innovations Greece nikos@indigma.eu

## ABSTRACT

Federated learning (FL) is increasingly used to adapt a pretrained, centrally fine-tuned model to data that remain on user devices, beyond training a model from scratch in a distributed manner. In this report we study the privacy and integrity of such an approach end to end, using a single reproducible pipeline prepared for the 4th Federated Machine Learning International Summer School. An ImageNet-pretrained MobileNetV4-Conv-Small model is first fine-tuned centrally on ten classes of the Quick, Draw! dataset, which raises test accuracy from 6.00% with an untrained classifier head to 97.50%. The resulting checkpoint initializes head-only FedAvg over ten non-IID clients. After 50 rounds, client training accuracy rises from 90.34% to 91.49%, while accuracy on a global evaluation set drawn from the Stage 1 data is preserved (97.66% before and 97.64% after). On this pipeline we evaluate four adversarial scenarios. First, from a single one-step client update, an honest-but-curious server recovers the complete private label set and reconstructs class-plausible sketches, although pixel fidelity remains low (mean SSIM 0.130) and the reconstructions follow class prototypes rather than the private drawings. Second, from a one-round FedAvg aggregate, label-count recovery falls from 87.5% with one client to 75.0% with five and ten clients, which is below the 80.0% of an uninformed uniform guess. Third, under FedAvg, a backdoor injected by 30%, 50%, and 100% of the clients reaches an attack success rate (ASR) of 21.9%, 31.3%, and 54.7%, respectively, while reducing the final clean accuracy by at most 0.8 percentage points. Label flipping, in contrast, becomes effective only when the attacker relabels most of the source-class data. Fourth, FLTrust with a 1,000-sample root set retained from centralized fine-tuning lowers the backdoor ASR to 1.8% and 4.0% at 30% and 50% malicious clients and the label-flipping ASR from 12.6% to 0.8%, without a measurable loss of clean accuracy. It provides no protection, however, when every client is malicious. This report closes with a live session in which participants draw, fine-tune, and poison the Stage 1 model through Indigma’s FedMaestro orchestrator. Our code and executed notebooks are available at https://github.com/Indigma-Innovations/fl-summer-school and run on Google Colab without specialized hardware.

Keywords Federated Learning · Transfer Learning · Gradient Inversion · Label Leakage · Data Poisoning · Backdoor Attack · Byzantine-Robust Aggregation · FLTrust

## Contents

1 Introduction 3   
2 Related Work 4   
3 Experimental Pipeline 4   
3.1 Overview 4   
3.2 Task and data 4   
3.3 Model and preprocessing 5   
3.4 Notation and head gradients 6   
4 Stage 1: Centralized Fine-Tuning 7   
4.1 Protocol 7   
4.2 Results . 7   
5 Stage 2: Clean Federated Fine-Tuning 7   
5.1 Client population 8   
5.2 Head-only FedAvg 8   
5.3 Results . 8   
6 Privacy: Reconstructing Private Data from Updates 9   
6.1 Threat model and gradient recovery 9   
6.2 Attack 1: a single client update 10   
6.3 Attack 2: a FedAvg aggregate update 12   
6.4 Summary of the privacy results . 15   
7 Integrity: Data Poisoning under FedAvg 16   
7.1 Threat model 16   
7.2 Attacks 16   
7.3 Metrics 16   
7.4 Results . 16   
8 Defense: FLTrust with a Retained Root Set 19   
8.1 Method 19   
8.2 Results . 19   
Computational Footprint 20   
10 Live Session: Federated Learning with FedMaestro 21   
11 Discussion 23   
12 Conclusions and Future Work 24

## 1 Introduction

Federated learning (FL) trains a shared model while the raw training data remain with the parties that hold them. A server receives model updates and combines them using a federated aggregation algorithm such as FedAvg [1, 2]. Tutorials and benchmark studies of FL frequently start from a randomly initialized network that is trained entirely through federated rounds. Such a setting is convenient for controlled comparisons, but it differs from how many vision and language systems are built in practice. A typical development path starts from a backbone pretrained on a large public corpus, adapts it centrally to the target task with representative data available, and only then uses FL to adapt the resulting model to data that cannot be centralized. Pretrained initialization is known to improve the accuracy and stability of federated optimization [3, 4], while adapting only a small task-specific part of the network is a common way to personalize a shared representation at a low cost [5, 6].

This deployment pattern changes the security question. The model that enters the federated stage is already accurate, the backbone may be frozen, and each client may update only a few thousand parameters. Two families of risk nevertheless remain. On the one hand, model updates are deterministic functions of the private data that produced them, and an honest-but-curious server can therefore try to infer labels or reconstruct inputs from them [7–9]. On the other hand, malicious clients can poison their local data or updates to degrade the global model or to implant a backdoor [10–12], and robust aggregation rules have been proposed to limit their influence [13–15]. Most of these results were obtained for full-model training, often from scratch. How strongly they apply when FL is the last, head-only stage of a pretrained pipeline is less documented, and this is the question we study in this report.

The report accompanies a hands-on tutorial at the 4th Federated Machine Learning International Summer School<sup>1</sup>, held in Paris, France, in October 2026. All experiments are implemented as six connected notebooks that share one task, one model, and one set of persisted artifacts. Every number in this report can be traced to a specific executed notebook. The notebooks run on Google Colab, i.e., participants can reproduce every experiment in a browser without local installation or high-end hardware. The pipeline fine-tunes an ImageNet-pretrained MobileNetV4-Conv-Small model [16] on ten classes of the Quick, Draw! dataset [17, 18], uses the resulting checkpoint to initialize head-only FedAvg over ten non-IID clients, and then attacks and defends the federated stage. The tutorial closes with a live session on FedMaestro, the federated learning orchestrator developed by Indigma<sup>2</sup>.

The main contributions of this report are summarized as follows:

• We describe a deployment-oriented FL pipeline that consists of pretraining, centralized fine-tuning, and federated fine-tuning, with explicit data provenance. The data used by each stage and by each adversary are drawn from documented index ranges, and the private data of every experiment are disjoint from the Stage 1 training data and from the public data available to the adversary. We also document that the global evaluation set of the federated notebooks is not the held-out Stage 1 test set, and we quantify the overlap.

• We show that, under one-step local SGD, the server recovers the exact classifier-head gradient both from a single client update and from a FedAvg aggregation. From a single update the server recovers the full private label set, but only within a narrow range of a heuristic weight. Pixel-space inversion yields class-plausible sketches of low fidelity, and under aggregation, label-count recovery falls below an uninformed baseline.

• We evaluate label flipping and a pixel-trigger backdoor for five malicious-client fractions under FedAvg. We show that the attack strength is explained by the amount of poisoned data that reaches the aggregate rather than by the nominal malicious fraction.

• We evaluate FLTrust with a root set retained from centralized fine-tuning. Using per-round trust diagnostics, we quantify how much aggregation weight FLTrust grants to malicious clients compared with FedAvg.

• We describe a live session in which the Stage 1 model is deployed in Indigma’s FedMaestro orchestrator, so that participants can observe inference, federated fine-tuning, and participant-driven poisoning on a running system.

The rest of this report is organized as follows. Section 2 reviews related work on FL from pretrained models, privacy leakage from model updates, and data poisoning. Section 3 describes the experimental pipeline, while Sections 4 and 5 present the centralized and the clean federated fine-tuning stages. Sections 6, 7, and 8 evaluate the privacy attacks, the poisoning attacks, and the FLTrust defense, respectively. Section 9 reports the computational footprint of every experiment, and Section 10 describes the live session. Section 11 discusses lessons learned and limitations. Finally, Section 12 concludes the report and outlines directions for future work.

## 2 Related Work

Federated learning from pretrained models. In FedAvg [1], a server broadcasts the current global model to a set of clients, and each client performs a few local optimization steps on its private data. The server then updates the global model by the average of the returned models, weighted by the number of local training samples. Statistical heterogeneity across clients (non-IID data) slows convergence and is commonly simulated by drawing per-client class proportions from a Dirichlet distribution with a small concentration parameter [19]. Starting federated optimization from a pretrained model reduces the harm caused by heterogeneity and closes much of the gap to centralized training [3, 4]. Pretrained ImageNet [20] features transfer well to new visual tasks [21, 22], and splitting a network into a shared representation and a lightweight, locally adapted head is a standard approach to federated personalization [5, 6]. Our pipeline combines both ideas, i.e., the backbone is adapted centrally and then frozen, while FL updates the final linear classifier.

Privacy leakage from model updates. Model updates leak information about the data that produced them, including properties that are unrelated to the learning task [7]. Deep Leakage from Gradients (DLG) reconstructs training images and labels by optimizing dummy inputs until their gradient matches an observed gradient [8]. iDLG showed that, for a single sample trained with cross-entropy, the label can be read directly from the sign of the last-layer gradient [23], and later work extended label inference to batches and to aggregated updates [24, 25]. Geiping et al. replaced the Euclidean matching loss by a cosine loss with a total-variation prior [26] and reconstructed ImageNet-scale images from realistic updates [9]. GradInversion further added batch-level label recovery and image priors [24]. Huang et al. showed that reported attack strength depends heavily on assumptions such as known labels, known batch-normalization statistics, and small batches [27], which motivates the explicit statement of assumptions in Section 6. Secure aggregation [28] hides individual updates from the server, and differentially private training [29] bounds what any update can reveal. However, neither privacy-preserving technique is applied in this report.

Data poisoning and robust aggregation. In data poisoning, malicious participants train on manipulated data by following the federated protocol. Label flipping relabels examples of a source class as a target class and degrades the source class in proportion to the number of poisoned participants [10]. Backdoor attacks implant a trigger that redirects predictions to an attacker-chosen class while leaving clean accuracy largely intact [30]. In FL, backdoors can be injected through model replacement [11] or boosted malicious updates [12], and their persistence depends on the fraction of attackers and on norm constraints [31, 32]. Byzantine-robust aggregation rules limit the influence of outlying updates. Krum selects the update closest to its neighbours [13], the coordinate-wise median and trimmed mean discard extreme values [14], and Bulyan combines both ideas [33]. Such rules can be circumvented by optimized attacks [34, 35], and their assumptions are strained by non-IID honest clients. FLTrust bootstraps trust from a small clean root dataset held by the server. Each client update is scored by its cosine similarity with a server update computed on the root set, and accepted updates are rescaled to the norm of the server update [15]. In our pipeline, a retained root set is a natural extension of the centralized stage, which makes FLTrust a well-matched defense.

## 3 Experimental Pipeline

## 3.1 Overview

Figure 1 summarizes the overall pipeline. Stage 1 (Notebook 1) fine-tunes the complete pretrained network centrally. The produced checkpoint is the common starting point of every later experiment, namely the single-client and aggregate privacy attacks (Notebook 2 and Notebook 3), the clean federated baseline (Notebook 4), the poisoning experiments under FedAvg (Notebook 5) and FLTrust (Notebook 6), and the live FedMaestro session. Table 1 lists the adversary, its capabilities, and the reported metrics for each experiment. Unless stated otherwise, all federated experiments train only the classifier head and all clients participate in every round.

## 3.2 Task and data

The task is ten-class sketch recognition on the Quick, Draw! dataset [17, 18], which contains crowd-sourced drawings collected through an online game. We use the preprocessed 28 × 28 grayscale bitmaps of the classes line, circle, square, triangle, star, tshirt, smiley\_face, sun, cloud, and donut (Figure 2). The ten class files contain between 120,265 (cloud) and 143,549 (line) bitmaps each. The classes combine simple geometric primitives with composite drawings and include visually related pairs such as circle/donut and sun/star, which become important during the execution of attacks.

![](images/73ba62bd56d6bf3cf02cc8677a627cce28c764f0a100ce054a8d008764c7ceff.jpg)  
Figure 1: Overview of the pipeline. Blue boxes denote model development stages, orange boxes privacy attacks by an honest-butcurious server, red and green boxes poisoning by malicious clients and the FLTrust defense, and the gray box the live session. NBk denotes the executed notebook that implements each component. Attacks 1 and 2 start from the Stage 1 checkpoint, whereas Attack 3 and the defense reuse the exact client population of Stage 2.

Table 1: Experiments, adversaries, and reported metrics. $\theta _ { t }$ denotes the global classifier head at round t, ∆ a model update, $n _ { k }$ the sample count of client k, and η the local learning rate.
<table><tr><td>Experiment</td><td>Adversary</td><td>Capabilities and assumptions</td><td>Reported metrics</td></tr><tr><td>Stage 1 (NB1)</td><td>none</td><td>centralized training on QuickDraw-10</td><td>loss, accuracy</td></tr><tr><td>Stage 2 (NB4)</td><td>none</td><td>10 clients, head-only FedAvg</td><td>loss, accuracy (client and global) label-set accuracy; MSE, PSNR, SSIM</td></tr><tr><td>Attack 1 (NB2)</td><td>honest-but-curious server</td><td>observes  $\theta _ { t } ,$  one client delta  $\Delta _ { k } , n _ { k } ,$  η; public QuickDraw data</td><td></td></tr><tr><td>Attack 2 (NB3)</td><td>honest-but-curious server</td><td>observes  $\begin{array} { r } { \theta _ { t } , \theta _ { t + 1 } , m , N = \sum _ { k } n _ { k } , \eta ; } \end{array}$  public QuickDraw data</td><td>label-count accuracy; MSE, PSNR, SSIM</td></tr><tr><td>Attack 3 (NB5)</td><td>malicious clients (fraction f)</td><td>poison local training data; follow the protocol otherwise</td><td>clean accuracy, ASR</td></tr><tr><td>Defense (NB6)</td><td>malicious clients (fraction f)</td><td>as Attack  $_ { 3 ; }$  server holds a clean root set</td><td>clean accuracy, ASR, trust scores</td></tr></table>

Since every experiment draws from the same class files, we separate data by index range wherever disjointness matters, as summarized in Table 2. The Stage 1 split uses the first 5,000 bitmaps of each class. Federated clients receive bitmaps with index at least 10,000, and the private samples of the privacy attacks are drawn from indices of at least 20,000. The public prior knowledge of the attacker is drawn from indices below 20,000 and therefore never contains a private sample of the privacy experiments. The FLTrust root set is sampled from the Stage 1 data, which a provider would plausibly retain after centralized fine-tuning.

The federated notebooks (Notebook 4–Notebook 6) evaluate the global accuracy on 5,000 images obtained by re-running the stratified 80/10/10 split of the first 5,000 bitmaps per class. They use seed 123 instead of the seed 42 used in Notebook 1, so this global evaluation set is a different random partition from the Stage 1 test set. Reproducing both splits shows that 3,990 of its images (79.8%) belong to the Stage 1 training split, 479 (9.6%) to the Stage 1 validation split, and only 531 (10.6%) to the Stage 1 test split. The global evaluation set is disjoint from all federated client data and identical across all federated runs, and remains valid for comparing federated configurations with one another.

## 3.3 Model and preprocessing

We use mobilenetv4\_conv\_small\_050.e3000\_r224\_in1k, the 0.5-width variant of the MobileNetV4-Small [16] pretrained on ImageNet-1k obtained via timm [36]. Its ImageNet head is replaced by a randomly initialized linear layer with C = 10 outputs, which results in 970,762 parameters. The backbone $f _ { \phi }$ maps an input to a d = 1280-dimensional embedding $h = f _ { \phi } ( x )$ , and the head computes logits $z = W h + b$ with $W \in \mathbb { R } ^ { C \times d }$ and $b \in \mathbb { R } ^ { C }$ . The head has $C ( d + 1 ) = 1 2 { , } 8 1 0$ parameters, i.e., 1.32% of the network. In 32-bit floating point the parameters of the full model occupy 3.70 MiB and those of the head 0.049 MiB, so a head-only update is about $1 / 7 6$ the size of a full-model update.

Each bitmap is replicated to three channels and processed with the data configuration of the pretrained model, namely resizing to 256 pixels, center cropping to $2 2 4 \times 2 2 4$ , and normalization with mean and standard deviation 0.5 per channel. During Stage 1 training, the pretrained configuration additionally applies random resized cropping (scale 0.08–1), horizontal flipping with probability 0.5, and color jitter of strength 0.4. Stage 1 resizes with bicubic interpolation, as specified by the model configuration. Notebook 2–Notebook 6, in contrast, rebuild the evaluation transform from the stored mean, standard deviation, and input size and therefore use the timm default, i.e., bilinear interpolation. Together with the different evaluation split described in Section 3.2, this explains why the same checkpoint scores 97.50% on the Stage 1 test set in Notebook 1 and 97.66% on the global evaluation set in Notebook 4–Notebook 6. The two numbers are not comparable, and we report each number with the preprocessing and data of the notebook that produced it.

![](images/67b3fec02bc749e75f905dfa544fc079f735a04b0c0d0c93ced1e543a9617fc2.jpg)  
Figure 2: Five randomly drawn Stage 1 examples per class (28 × 28 bitmaps, white strokes on a black background).

Table 2: Data provenance. Indices refer to positions within each QuickDraw class file. “Hard” selection retains the candidates with the highest cross-entropy loss under the Stage 1 model (Sections 5 and 6).
<table><tr><td>Use</td><td>Indices</td><td>Size</td><td>Selection</td></tr><tr><td>Stage 1 train / validation / test (NB1)</td><td>0-4,999</td><td>40,000 / 5,000 / 5,000</td><td>stratified 80/10/10 split per class (seed 42)</td></tr><tr><td>Federated clients (NB4–NB6)</td><td>≥ 10,000</td><td>13,675 (12,310 train, 1,365 val.)</td><td>hard selection from ≥ 5,000 ran- dom candidates per class</td></tr><tr><td>Attack 1 private batch (NB2)</td><td>≥ 20,000</td><td>5</td><td>hardest of 128 random candidates per class</td></tr><tr><td>Attack 2 private pool (NB3)</td><td>≥ 20,000</td><td>56 (10 clients)</td><td>hardest of 128 random candidates per class and client</td></tr><tr><td>Attacker public priors (NB2, NB3) Global evaluation set (NB4–NB6)</td><td>&lt; 20,000 0-4,999</td><td>96 per class (prototypes), 1 per canvas 5,000</td><td>uniform random test part of the 80/10/10 split rebuilt</td></tr><tr><td></td><td></td><td></td><td>with seed 123 (79.8% overlap with the Stage 1 training split)</td></tr><tr><td>FLTrust root set (NB6)</td><td>0-4,999</td><td>1,000 (100 per class)</td><td>uniform random from the training part of the seed-123 split</td></tr></table>

## 3.4 Notation and head gradients

All federated experiments update only $\theta \ : = \ : ( W , b )$ . For a batch $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ with embeddings $h _ { i }$ and softmax probabilities $p _ { i } = \mathrm { s o f t m a x } ( W h _ { i } + b )$ , the gradient of the mean cross-entropy loss with respect to the head is

$$
g _ { b } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( p _ { i } - e _ { y _ { i } } \right) , \qquad g _ { W } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( p _ { i } - e _ { y _ { i } } \right) h _ { i } ^ { \top } ,\tag{1}
$$

Table 3: Stage 1 performance before fine-tuning (pretrained backbone, untrained head) and after fine-tuning (selected checkpoint, epoch 15).
<table><tr><td>Split</td><td>Loss before</td><td>Loss after</td><td>Accuracy before (%)</td><td>Accuracy after (%)</td><td> $\Delta$  accuracy (pp)</td></tr><tr><td>Train (augmented)</td><td>7.8110</td><td>0.2378</td><td>6.97</td><td>92.29</td><td>+85.32</td></tr><tr><td>Validation</td><td>8.5807</td><td>0.0961</td><td>5.64</td><td>97.48</td><td>+91.84</td></tr><tr><td>Test</td><td>8.5695</td><td>0.0969</td><td>6.00</td><td>97.50</td><td>+91.50</td></tr></table>

where $e _ { y }$ is the one-hot vector of label y. Row c of $g _ { W }$ , denoted $g _ { W } ^ { ( c ) }$ , and entry c of $g _ { b } ,$ denoted $g _ { b } ^ { ( c ) }$ , correspond to output c and aggregate all samples with weights $p _ { i } ^ { ( c ) } - \mathbf { 1 } [ y _ { i } = c ]$ . The weight is positive for every $c \neq y _ { i }$ and negative for $c = y _ { i }$ , since $0 < p _ { i } ^ { ( c ) } < 1$ for finite logits<sup>3</sup>. Hence the entry $g _ { b } ^ { ( c ) }$ can be negative only if class c occurs in the batch. The converse does not hold, i.e., a class that occurs can have a positive entry when the other samples assign it a large probability. Both attacks in Section 6 exploit this structure. Throughout this report, we write pp for percentage points, i.e., for absolute differences between two percentages.

## 4 Stage 1: Centralized Fine-Tuning

## 4.1 Protocol

The complete network is fine-tuned on the Stage 1 training split (40,000 images) with AdamW [37] (learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } )$ , a cosine learning-rate schedule over 15 epochs [38], cross-entropy loss, and batch size 128. After every epoch we evaluate the model on all three splits. The checkpoint with the highest validation accuracy is retained, and training stops early if validation accuracy does not improve for four consecutive epochs. The test split is evaluated after every epoch for monitoring only and is not used for any decision. Training metrics are computed on the augmented training stream and therefore measure performance on randomly cropped and jittered views. Before fine-tuning, we evaluate the pretrained backbone with the untrained 10-class head as a reference.

## 4.2 Results

Table 3 and Figure 3 summarize Stage 1. With the untrained head, the model performs at or below chance level (6.00% test accuracy against 10% for uniform guessing), with a cross-entropy of about 8.6, far above the ln $1 0 \approx 2 . 3 0$ of a uniform predictor. The randomly initialized head is therefore confidently wrong rather than uninformative. One epoch of fine-tuning raises validation accuracy to 95.80%, and accuracy increases slowly thereafter. Validation accuracy reaches its maximum of 97.48% at the last epoch. As a result, the early-stopping counter never reaches its patience of four (its maximum is three, at epoch 12), and the selected checkpoint is the epoch-15 model. The selected model attains 97.50% test accuracy and 0.0969 test loss. Validation and test curves agree closely throughout training (largest gap of 0.58 pp, at epoch 2), which indicates that model selection did not overfit the validation split.

Training accuracy stays below validation accuracy (92.29% against 97.48% for the selected checkpoint). The training metric is measured on augmented views, and random resized crops with scale down to 8% can remove most of the strokes of a sparse sketch. Since accuracy was still increasing when the 15-epoch budget ended, a longer schedule would probably improve the model slightly. We kept the budget fixed so that the training loop completes in under one hour on a single Colab GPU (Section 9). Reloading the exported safetensors checkpoint [39] reproduces the test accuracy exactly. This checkpoint, together with a configuration file that records the class order and preprocessing statistics, constitutes the global initialization $\theta _ { 0 }$ of all later stages.

## 5 Stage 2: Clean Federated Fine-Tuning

Stage 2 establishes the attack-free federated baseline against which the poisoning and defense experiments are compared.   
This stage models a provider that distributes the Stage 1 model to users whose drawings must remain on their devices.

![](images/fc9a03eb66d226009427e00a26543e52e3d7257bc930391ef72d1f2db68cb6ae.jpg)  
Figure 3: Stage 1 learning curves. (a) Cross-entropy loss and (b) accuracy per epoch.

## 5.1 Client population

We simulate K = 10 clients (Notebook 4). For each client, we draw a dataset size uniformly from [1,000, 2,000] and a number of classes uniformly from $\{ 5 , \ldots , 1 0 \}$ , and select that many classes at random. We then assign 20 samples to each selected class and distribute the remaining samples according to class proportions drawn from a Dirichlet distribution with concentration $\alpha = 0$ .35 [19]. The resulting population is heterogeneous both in size (1,015 to 1,890 samples) and in label composition (Figure 4). For example, sun accounts for 35% of client 1 and smiley\_face for 39% of client 4, whereas several clients hold only the minimum of 20 samples for some classes.

To make the federated stage a meaningful adaptation problem, clients do not receive arbitrary samples. For each class, we draw at least 5,000 random candidates with index $\geq 1 0 { , } 0 0 0$ and score them with the cross-entropy loss of the Stage 1 model. We keep the required number of highest-loss candidates and shuffle them before assignment. The retained fraction ranges from 9.4% of the candidates for tshirt (468 of 5,000) to 49.2% for line (2,460 of 5,000). Client data are therefore harder than the Stage 1 test set. With the same preprocessing, the Stage 1 model reaches 90.34% accuracy on the pooled client training data, compared with 97.66% on the global evaluation set (Section 3.2). Each client splits its data 90/10 into local training and validation sets, stratified by class, which gives 12,310 training and 1,365 validation samples in total.

## 5.2 Head-only FedAvg

The global head is initialized from the Stage 1 checkpoint, $\theta _ { 0 } = ( W _ { 0 } , b _ { 0 } )$ , and the backbone is frozen. Since a frozen backbone in evaluation mode is a deterministic function, we compute the embeddings of all client and test images once and train the head directly on them. Such a procedure produces the optimization problem of head-only fine-tuning through the frozen network at a fraction of the cost. In every round $t = 1 , \ldots , 5 0$ , each client k copies the current global head, trains it for one local epoch with Adam [40] (learning rate $1 0 ^ { - 3 }$ , batch size 128, no weight decay, optimizer state re-initialized every round), and returns its head $\theta _ { t } ^ { k }$ . The server aggregates the heads with sample-weighted FedAvg,

$$
\theta _ { t } = \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { \sum _ { j = 1 } ^ { K } n _ { j } } \theta _ { t } ^ { k } ,\tag{2}
$$

where $n _ { k }$ is the number of local training samples of client k. We report the sample-weighted loss and accuracy over all client training and validation splits, which measure adaptation to the federated distribution. We also report the loss and accuracy on the global evaluation set (Section 3.2), which measure whether the federated stage preserves the behavior learned in Stage 1. Since the pooled client validation set contains 1,365 images, one image corresponds to 0.073 pp of validation accuracy, and on the 5,000-image global evaluation set, one image corresponds to 0.02 pp.

## 5.3 Results

Table 4 and Figure 5 report the clean federated run. After 50 rounds, client training accuracy rises from 90.34% to 91.49% (+1.15 pp) and client training loss falls from 0.3876 to 0.3703. More than half of this gain (0.64 pp) is achieved in the first four rounds. Client validation accuracy changes from 89.67% to 89.89%, a difference of three images. Global accuracy stays between 97.60% and 97.74% in every round and ends at 97.64%, i.e., one image below its initial value.

![](images/9ee83eacdf276ad7e93e47fa170e5c0ee8815f2278aa76ee6548d5de859bedda.jpg)  
Figure 4: Client label distribution of the federated population. Each cell gives the number of local samples (training and validation) of a class, and gray cells mark classes that a client does not hold. Classes per client range from 7 to 10 and dataset sizes from 1,015 to 1,890 samples.

Table 4: Clean head-only federated fine-tuning before the first round and after 50 FedAvg rounds. Client metrics are sample-weighted over all ten clients.
<table><tr><td></td><td colspan="2">Round 0</td><td colspan="2">Round 50</td><td></td></tr><tr><td>View</td><td>Loss</td><td>Accuracy (%)</td><td>Loss</td><td>Accuracy (%)</td><td> $\Delta$  accuracy (pp)</td></tr><tr><td>Client train (12,310)</td><td>0.3876</td><td>90.34</td><td>0.3703</td><td>91.49</td><td>+1.15</td></tr><tr><td>Client validation (1,365)</td><td>0.4305</td><td>89.67</td><td>0.4364</td><td>89.89</td><td>+0.22</td></tr><tr><td>Global evaluation (5,000)</td><td>0.0925</td><td>97.66</td><td>0.0910</td><td>97.64</td><td>-0.02</td></tr></table>

The loss curves show a mild trade-off between personalization and generalization. Client training loss reaches its minimum of 0.3605 at round 17 and then increases slowly, while client validation loss is lowest at round 10 (0.4089) before rising to 0.4364. Accuracy does not decrease over the same rounds, which is consistent with growing confidence on a few misclassified examples. Overall, head-only FedAvg adapts the model modestly to the hard client distribution and preserves its performance on the Stage 1 data distribution. A limit of roughly one percentage point on the training side is plausible for this setting, since only a linear head over fixed features is trained and the selected client data are, by construction, the samples that these features separate least well.

## 6 Privacy: Reconstructing Private Data from Updates

## 6.1 Threat model and gradient recovery

The adversary is an honest-but-curious server, i.e., it executes the protocol faithfully but it can analyse any information it receives. The server knows the architecture and the global model $\theta _ { t } ,$ , the global label space, the learning rate η of the clients, and the number of local samples, which FedAvg needs for weighting. It also has access to public QuickDraw data, which is realistic since the dataset and the label space of the deployed model are public. However, it does not observe raw images or labels, and it does not modify the model it sends to clients.

Both attacks rely on the following observation. If client k performs exactly one plain SGD step with learning rate η on its head, its update is $\Delta _ { k } = \theta _ { t } ^ { k } - \theta _ { t } = - \eta g ^ { k }$ , where $g ^ { k }$ is the gradient of Eq. (1) on its local batch. The server can recover the gradient as:

$$
g ^ { k } = - \Delta _ { k } / \eta .\tag{3}
$$

![](images/79bf8e0c771002196134781a6a5fffe7ee7844d3065c5ea59676534ff7b14865.jpg)  
Figure 5: Clean federated fine-tuning over 50 rounds. Columns show the pooled client training and validation data, and the global evaluation set, while rows show cross-entropy loss and accuracy.

If m clients in a set $S _ { t }$ each perform such a step and the server applies Eq. (2), the change of the global model is

$$
\theta _ { t + 1 } - \theta _ { t } = \sum _ { k \in S _ { t } } \frac { n _ { k } } { N } \Delta _ { k } = - \eta \sum _ { k \in S _ { t } } \frac { n _ { k } } { N } g ^ { k } , \qquad N = \sum _ { k \in S _ { t } } n _ { k } ,\tag{4}
$$

which is −η times the mean-loss gradient of the pooled virtual batch of all N participating samples. The server can therefore compute this pooled gradient without observing any individual update. In Notebook 2, the client performs an actual SGD step, and we experimentally verify Eq. (3). Specifically, the mean squared difference between the delta-derived gradient and the gradient captured inside the client is 0.0 for both W and b. In Notebook 3, the client updates are computed analytically $\mathrm { a s } - \eta g ^ { \hat { k } }$ , so Eq. (4) holds by construction, and the corresponding check (also 0.0) confirms the implementation of the aggregation. Note that the equivalence is specific to one local SGD step with a shared learning rate. With several local steps, momentum or adaptive optimizers, or heterogeneous learning rates, the update is no longer a scaled gradient. Clients therefore use one SGD step in the privacy experiments, whereas Stage 2 uses one epoch of Adam.

## 6.2 Attack 1: a single client update

Setup. The target client holds $n = 5$ private sketches, one each of line, circle, square, triangle, and star (Notebook 2). Each sample is the highest-loss bitmap among 128 random candidates of its class, which stresses the attack with informative gradients. Under the Stage 1 model these samples have losses between 1.22 (star) and 12.90 (circle), and all five are misclassified, as star, donut, smiley\_face, line, and cloud, respectively. The top row of Figure 6 shows that several of these drawings are atypical for their class, i.e., the triangle sample is a single diagonal stroke, the square sample is an open arc, and the line sample overlays a circle. The client keeps the backbone frozen and the network in evaluation mode, so that batch-normalization statistics are not updated by five samples, and takes one SGD step on the head with $\eta = 1 0 ^ { - 2 }$ . The step reduces the local loss from 5.4683 to 2.8258 and corrects one prediction (circle).

Label-set recovery. Figure 7 shows the per-class evidence available to the server after applying Eq. (3). The iDLG-based rule of selecting the n most negative bias-gradient entries [23] returns square, triangle, circle, star, and sun, i.e., four of the five labels. It misses line because $g _ { b } ^ { ( \mathrm { 1 i n e } ) } ~ = ~ + 0 . 0 0 4 9$ is positive. By Eq. (1), $\begin{array} { r } { g _ { b } ^ { ( c ) } = \frac { 1 } { n } \big ( \sum _ { i } p _ { i } ^ { ( c ) } - 1 \big ) } \end{array}$ for a class that occurs once, and the confident prediction of the private triangle sample as line pushes the summed probability of line slightly above one. The absent class sun has the smallest positive entry (+0.0038) and is selected instead. We therefore score complete label sets. For every candidate set $S$ of size $n _ { \mathrm { { ; } } }$

![](images/97e7903b1aebc4ef789e996fa17115ab72928b801f0a2faf849b958d04e12b14.jpg)  
Figure 6: Attack 1: private samples (top) and reconstructions from the single-client update using the recovered labels (middle) and the true labels (bottom, evaluation-only diagnostic).

![](images/b1e854eab5ecc9cac0f48a6dd41ed4eb4cd3ab07289d145134b41b1fc1fff0c5.jpg)

![](images/e8e6648349abf76f6d028bbc99ccfd4c9d61797d0a581e23fa04c6d05aae0fc3.jpg)  
Figure 7: Label-leakage signal in a single head update. (a) Bias gradient $g _ { b } ^ { ( c ) }$ per class. Asterisks mark the five most negative entries selected by the naive rule, which includes the absent class sun and misses line. (b) Norm of the weight-gradient row $g _ { W } ^ { ( c ) }$ on a logarithmic scale. Both circle (present) and donut (absent) have large norms because the private circle sample is confidentl predicted as donut.

$$
\operatorname { s c o r e } _ { \lambda } ( S ) = - \sum _ { c \in S } g _ { b } ^ { ( c ) } + \lambda \sum _ { c \in S } \| g _ { W } ^ { ( c ) } \| _ { 2 } ,\tag{5}
$$

and the attacker returns the highest-scoring set. With $C = 1 0$ and $n = 5$ there are only ${ \binom { 1 0 } { 5 } } = 2 5 2$ candidate sets, so the search is exhaustive. The first term is the bias evidence. The second term rewards classes whose weight-gradient row is large, which is the case for classes involved in errors. The row norms should be weighted lightly, because they are large for confusions as well as for present classes. For example, the misclassified circle sample yields row norms of 12.89 for circle and 12.90 for the absent class donut (Figure 7b). With the weight $\lambda = 0 . 0 1$ , the true set ranks first (score 0.6597), ahead of the set that replaces line by sun (score 0.6464). Table 5 summarizes both rules.

The margin is small, and the result depends on λ. Recomputing Eq. (5) from the per-class values shows that the true set is the arg max only for $0 . 0 0 0 7 9 \leq \bar { \lambda } \leq 0 . 0 1 8 8$ . For $\lambda \bar { = } 0$ the score reduces to the naive rule and the true set ranks second. For $\lambda > 0 . 0 1 8 8 .$ , the large donut row norm pulls donut into the selected set. Hence, 4 of 5 labels are recovered by the parameter-free rule and the remaining ambiguity concerns the two near-zero entries for line and sun.

Table 5: Attack 1: label-set recovery from one single-client update. The margin is the score difference to the second-ranked set.
<table><tr><td>Rule</td><td>Recovered label set</td><td>Correct</td><td>Margin</td></tr><tr><td>Top-n most negative  $g _ { b } ^ { ( c ) } ( \equiv \mathrm { E q . } ( 5 )$  with λ = 0)</td><td>square, triangle, circle, star, sun</td><td>4/5</td><td>0.0011</td></tr><tr><td>Set search, Eq. (5) with  $\lambda = 0 . 0 1$ </td><td>line, circle, square, triangle, star</td><td>5/5</td><td>0.0133</td></tr></table>

Image reconstruction. Given the recovered labels ${ \hat { S } } ,$ the attacker reconstructs one image per label by gradient matching [8, 9]. Each image is a learnable $2 8 \times 2 8$ canvas of logits ζ. The pixel intensities $u = \dot { \sigma } ( \zeta ) \in [ 0 , \dot { 1 } ] ^ { 2  { \breve { 8 } } \times 2 8 }$ are bilinearly upsampled to $2 \hat { 2 } 4 \times 2 2 4$ , replicated to three channels, normalized, and passed through the frozen backbone and the global head. Let $\left( \hat { g } _ { W } , \hat { g } _ { b } \right)$ be the head gradient of Eq. (1) induced by the canvases and the recovered labels. The attacker minimizes

$$
\mathcal { L } ( \zeta ) = \frac { \| \hat { g } _ { W } - g _ { W } \| _ { 2 } ^ { 2 } } { \| g _ { W } \| _ { 2 } ^ { 2 } } + \alpha \frac { \| \hat { g } _ { b } - g _ { b } \| _ { 2 } ^ { 2 } } { \| g _ { b } \| _ { 2 } ^ { 2 } } + \beta _ { \mathrm { T V } } \mathrm { T V } ( u ) + \beta _ { \mathrm { s p } } \overline { { u } } + \beta _ { \mathrm { b i n } } \overline { { u ( 1 - u ) } } ,\tag{6}
$$

where $\mathrm { T V } ( u )$ is the mean absolute difference between vertically and horizontally adjacent pixels [26], u is the mean intensity, which favors the mostly black QuickDraw background, and $\overline { { u ( 1 - u ) } }$ penalizes gray values. We use $\alpha = 1 0 .$ $\beta _ { \mathrm { T V } } = 0 . 0 1$ , and $\beta _ { \mathrm { s p } } = \beta _ { \mathrm { b i n } } \stackrel { \cdot } { = } 0 . 0 0 1$ Each canvas is initialized from an unrelated public sketch of its recovered class with Gaussian logit noise (standard deviation 0.25) and optimized with Adam (learning rate 0.05) for 2,500 steps. The best of five restarts, selected by the final gradient-matching loss, is returned. The public initialization supplies a class prior but no information about the private sample. As a diagnostic that isolates the pixel optimization from label recovery, the same procedure is repeated with the true labels (“oracle”).

We compare reconstructions with the private images by mean squared error (MSE) on the [0, 255] scale, peak signal-tonoise ratio (PSNR), and a single-window structural similarity index (SSIM) computed from global image statistics, which is a simplification of the windowed SSIM of Wang et al. [41]. For interpretation, we add two references that we computed post hoc from the $2 8 \times 2 8$ arrays rendered in the notebook figures. Recomputing the metrics of the notebook from these arrays reproduces them to within 0.5% in MSE, 0.02 dB in PSNR, and 0.001 in SSIM. The first reference is an empty (all-black) canvas, while the second scores every reconstruction against the private sketches of the other classes (mismatched pairs).

Results. The gradient-matching term decreases by almost three orders of magnitude, from 10.86 at initialization to 0.0156 for the best restart, so the reconstructions explain the observed update almost perfectly. Figure 6 and Table 6 show that they are nevertheless poor copies of the private drawings. The reconstructions are clean prototypes of the recovered classes, namely a horizontal line, a closed loop, a rectangle, a triangle, and a five-pointed star. The private triangle is a diagonal stroke and the private square an open arc, so the reconstructions disclose the label and a canonical shape, but not the client’s private drawing. The mean SSIM of 0.130 is higher than for mismatched pairs (0.055), so the reconstructions carry some image-specific structure, for example the outer loop of the private circle (SSIM 0.208). By MSE and PSNR, however, every reconstruction is worse than an empty canvas, which reaches 10.41 dB on average against 8.65 dB. The oracle diagnostic yields practically the same images and metrics (mean SSIM 0.126). The limiting factor is therefore the information content of a head-only update. The head gradient depends on the input only through the embedding $h ,$ and many different sketches of a class map to embeddings that produce nearly the same head gradient.

## 6.3 Attack 2: a FedAvg aggregate update

Setup. Beyond attacking a single client, we build a more realistic federated scenario with a pool of ten clients. Each client holds between four and eight private sketches of distinct classes (8, 4, 5, 6, 5, 7, 5, 6, 5, and 5 for clients 0 to 9), chosen with the same hard selection as in Attack 1. A class can therefore occur at most once per client but several times across clients. Two classes of the global label space, tshirt and donut, are withheld from every client. The attacker does not know which classes are absent and must infer $q _ { \mathtt { t s h i r t } } = q _ { \mathtt { d o n u t } } = 0$ from the update alone. In one round, the first $m \in \{ 1 , 2 , 5 , 1 0 \}$ clients each take one SGD step on their head $( \eta = 1 0 ^ { - 2 } )$ , and the server applies Eq. (2). The participating sets are nested, so the $N = 8 ,$ , 12, 28, and 56 private samples for increasing m always include those of smaller $m .$ . The attacker observes $\theta _ { t }$ and $\theta _ { t + 1 } , m$ , N, and η, and obtains the pooled gradient $g ^ { \mathrm { a g g } }$ from Eq. (4). Compared with Attack 1, the hidden object is a label-count vector $\dot { \boldsymbol { q } } \in \mathbb { N } ^ { C }$ with $\begin{array} { r } { \sum _ { c } { q _ { c } } = N } \end{array}$ and $0 \leq q _ { c } \leq m$ , and the gradient mixes the samples of several clients.

Table 6: Attack 1: reconstruction quality against the private 28 × 28 sketches (↓: lower is better, ↑: higher is better). Rows marked † are post hoc references (Section 6.2).
<table><tr><td></td><td colspan="3">Recovered labels</td><td colspan="3">Oracle labels</td></tr><tr><td>Class</td><td>MSE↓</td><td>PSNR (dB) ↑</td><td>SSIM↑</td><td>MSE↓</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td></tr><tr><td>line</td><td>7690.18</td><td>9.27</td><td>0.069</td><td>7620.23</td><td>9.31</td><td>0.070</td></tr><tr><td>circle</td><td>11111.11</td><td>7.67</td><td>0.208</td><td>10979.75</td><td>7.72</td><td>0.190</td></tr><tr><td>square</td><td>8178.44</td><td>9.00</td><td>0.192</td><td>8155.11</td><td>9.02</td><td>0.191</td></tr><tr><td>triangle</td><td>7658.22</td><td>9.29</td><td>0.010</td><td>7478.31</td><td>9.39</td><td>0.010</td></tr><tr><td>star</td><td>10332.21</td><td>7.99</td><td>0.171</td><td>10327.46</td><td>7.99</td><td>0.172</td></tr><tr><td>Mean</td><td>8994.03</td><td>8.65</td><td>0.130</td><td>8912.17</td><td>8.69</td><td>0.126</td></tr><tr><td>Empty canvas† (mean)</td><td>6495.6</td><td>10.41</td><td>0.000</td><td></td><td>一</td><td></td></tr><tr><td>Mismatched pairs† (mean)</td><td>10178.2</td><td>8.24</td><td>0.055</td><td></td><td>一</td><td></td></tr></table>

Algorithm 1 Attack 2: label-count recovery from a FedAvg aggregate   
Require: global heads $\theta _ { t } , \theta _ { t + 1 } ;$ learning rate η; number of clients m; total samples N; public data per class   
1: $g ^ { \mathrm { a g g } } \gets - ( \theta _ { t + 1 } - \theta _ { t } ) / \eta$ ▷ Eq. (4)   
2: for each class $c = 1 , \ldots , C$ do   
3: P<sup>(c)</sup> ← mean single-sample head gradient of R public sketches of class c at $\theta _ { t }$   
4: end for   
5: q˜ ← SLSQP solution of Eq. (7) from $q _ { c } = N / C$   
6: qˆ ← integer rounding of $\tilde { q }$ preserving $\textstyle \sum _ { c } { \hat { q } } _ { c } { \dot { = } } N$ and $0 \leq \hat { q } _ { c } \leq m$   
7: return label multiset with $\hat { q } _ { c }$ copies of class c

Label-count recovery. The attacker estimates per-class gradient prototypes from public data. For each class $c ,$ it passes $R = 9 6$ public sketches through the model and averages their single-sample head gradients, $P ^ { ( c ) } = ( P _ { W } ^ { ( c ) } , P _ { b } ^ { ( c ) } )$ It then models the summed private gradient $N g ^ { \mathrm { a g g } }$ as a non-negative combination of prototypes and solves

$$
\operatorname* { m i n } _ { q \in { \mathbb { R } } ^ { C } } \quad J ( q ) = \frac { \big \| \sum _ { c } q _ { c } P _ { W } ^ { ( c ) } - N g _ { W } ^ { \mathrm { a g g } } \big \| _ { 2 } ^ { 2 } } { \| N g _ { W } ^ { \mathrm { a g g } } \| _ { 2 } ^ { 2 } } + \gamma \frac { \big \| \sum _ { c } q _ { c } P _ { b } ^ { ( c ) } - N g _ { b } ^ { \mathrm { a g g } } \big \| _ { 2 } ^ { 2 } } { \| N g _ { b } ^ { \mathrm { a g g } } \| _ { 2 } ^ { 2 } } \quad { \mathrm { s . t . } } \quad \sum _ { c } q _ { c } = N , \ 0 \leq q _ { c } \leq m ,\tag{7}
$$

with $\gamma = 1 0$ , using sequential least-squares quadratic programming (SLSQP) [42, 43] initialized at $q _ { c } = N / C$ (Algorithm 1). The continuous solution is rounded to integers that preserve the total N and the bounds. We measure label-count accuracy as $\textstyle \sum _ { c } \operatorname* { m i n } ( q _ { c } , \hat { q } _ { c } ) / N$ , i.e., the fraction of private samples whose label is accounted for. Two references help interpret this metric. First, an uninformed allocation $\hat { q } _ { c } = N \bar { / } C$ already scores $\textstyle \sum _ { c }$ min $( q _ { c } , N / C ) / N$ The resulting value is 80.0% for $m \in \{ 1 , 5 , 1 0 \}$ and 73.3% for $m = 2$ , because the private classes are spread almost uniformly over the eight allowed classes. Second, the trivial prediction $q = 0$ gives $J = 1 + \gamma = 1 1$ , so values of J close to 11 indicate that the prototypes explain little of the observed gradient.

Table 7 and Figure 8 report the results. With a single client the attacker recovers seven of eight labels. It misses the private circle sample (an L-shaped stroke) and assigns one count to the withheld class tshirt. With more clients, accuracy drops to 83.3% for $m = 2$ and to 75.0% for $m = 5$ and $m = 1 0 .$ . For $m \geq 5$ the attack accuracy is below the uninformed uniform allocation, so the estimator does not provide information about the label counts beyond the knowledge of N. The solutions have a characteristic structure, i.e., five classes are set to the upper bound $q _ { c } = m$ and the remaining mass is assigned to one or two classes. Classes such as line $( m = 5$ , three occurrences) or sun and cloud $( m = 1 0$ , six and eight occurrences) receive no count. The objective values of 10.20 to 10.84 are only 1.5% to 7.3% below the trivial value of 11, which shows that the prototype mixture fits the aggregate gradient poorly.

We attribute the misfit to the hard selection of the private samples. Prototypes average the gradients of typical public sketches, most of which the model classifies correctly and which therefore have small gradients. The private gradients, in contrast, are dominated by a few confidently misclassified samples. Matching their magnitude requires inflating the counts of the classes whose prototypes best align with the aggregate direction until the bound is reached. The attack is nevertheless informative. From $m = 2$ onwards it never assigns a count to the withheld classes, and the withheld class donut always receives zero, consistent with its large positive bias gradient. Its output therefore reveals which classes are present more reliably than how often they occur.

Table 7: Attack 2: label-count recovery from a one-round FedAvg aggregate (Notebook 3). “Uniform” is the accuracy of the uninformed allocation $\hat { q } _ { c } = N / C ,$ and $\dot { J } ( \tilde { q } )$ is the objective of Eq. (7) at the continuous solution, to be compared with $J = 1 1$ for $q = 0 .$
<table><tr><td>m</td><td>Clients</td><td>N</td><td>Correct</td><td>Count accuracy (%)</td><td>Uniform (%)</td><td> $J ( \tilde { q } )$ </td><td>Classes at bound m</td></tr><tr><td>1</td><td>0</td><td>8</td><td>7/8</td><td>87.5</td><td>80.0</td><td>10.84</td><td>_*</td></tr><tr><td>2</td><td>0-1</td><td>12</td><td>10/12</td><td>83.3</td><td>73.3</td><td>10.45</td><td>5</td></tr><tr><td>5</td><td>0-4</td><td>28</td><td>21/28</td><td>75.0</td><td>80.0</td><td>10.20</td><td>5</td></tr><tr><td>10</td><td>0-9</td><td>56</td><td>42/56</td><td>75.0</td><td>80.0</td><td>10.44</td><td>5</td></tr></table>

<sup>∗</sup>For m = 1, every non-zero count equals the bound.

![](images/52de03cdd12f012fd2237fb6743ce37872c42e0415c4b4dceb5c1457b3a2f1c8.jpg)  
Figure 8: Attack 2: true and recovered label counts for $m = 1 , 2 , 5 ,$ 10 participating clients (Notebook 3). Shaded columns mark the classes withheld from all clients. The dashed line is the per-class upper bound $q _ { c } \le m$ . For $m \geq 2 ,$ , the solution places five classes at the bound and omits classes that occur several times.

Image reconstruction. The attacker expands qˆ into a label multiset and optimizes one canvas per label with Eq. (6), using the aggregate gradient as target and the same regularization weights as in Attack 1. Canvases are initialized from public sketches with logit noise of standard deviation 0.20. Since the number of canvases grows with N, we reduce the budget with m (2,500, 2,000, 1,500, and 1,000 Adam steps with 5, 3, 2, and 1 restarts for m = 1, 2, 5, 10) and use a learning rate of 0.01 with gradient clipping at norm 1. Instead of the final iterate, we return the canvases with the lowest objective over all steps and restarts. The aggregate does not indicate which reconstruction belongs to which sample, so reconstructions are matched to private samples of the same class by minimum-MSE assignment [44]. We report the metrics over matched pairs and a “full” variant, in which every private sample whose class received too few counts is compared with an empty canvas.

Table 8 and Figure 9 show the results. As in Attack 1, the reconstructions are recognizable prototypes of the assigned classes (straight lines, ellipses, rectangles, triangles, and stars), while the atypical private drawings are not recovered. This picture hardly changes with m. Two aspects of the evaluation explain why the metrics are flat. First, with the class prior doing most of the work, a reconstruction is about as similar to any private sample of its class as to the sample it is matched to, and matching by minimum MSE favors such generic shapes. Second, the empty-canvas penalty is not conservative for MSE and PSNR. For m = 1, the full metric (8034.18, 9.23 dB) is better than the matched-only metric (8615.64, 8.81 dB), because the missed circle sample is a sparse stroke whose empty-canvas MSE (3964, derived from the two reported averages) is lower than the MSE of each of the seven matched reconstructions. Only SSIM penalizes the missed samples consistently. A few reconstructions nonetheless show traces of their private sample. For m = 1, the line reconstruction contains the two loops of the private drawing, and the triangle reconstruction has a curved side similar to the private sketch. Such details appear with both recovered and oracle counts and become les visible for larger m.

Table 8: Attack 2: reconstruction quality against the private sketches. “Matched” averages over reconstructions assigned to a private sample of the same class, while “Full” also includes each missed private sample, scored against an empty canvas.
<table><tr><td rowspan="2">m Pairs</td><td rowspan="2"></td><td colspan="3">Matched pairs</td><td colspan="3">Full (missed → empty canvas)</td><td rowspan="2">Runtime (s)</td></tr><tr><td>MSE↓</td><td></td><td>PSNR (dB) ↑ SSIM↑</td><td>MSE↓</td><td>PSNR (dB) ↑</td><td>SSIM↑</td></tr><tr><td>1</td><td>7/8</td><td>8615.64</td><td>8.81</td><td>0.212</td><td>8034.18</td><td>9.23</td><td>0.186</td><td>393.9</td></tr><tr><td>2</td><td>10/12</td><td>10081.14</td><td>8.17</td><td>0.117</td><td>10285.06</td><td>8.08</td><td>0.097</td><td>86.0</td></tr><tr><td>5</td><td>21/28</td><td>8087.35</td><td>9.32</td><td>0.198</td><td>8281.28</td><td>9.18</td><td>0.148</td><td>53.1</td></tr><tr><td>10</td><td>42/56</td><td>8672.74</td><td>8.99</td><td>0.219</td><td>8611.48</td><td>9.07</td><td>0.164</td><td>33.9</td></tr></table>

![](images/6dfc0b2c787f27f90fa8fb0b1926c35a25a0c03c0794e8ba3f8abdb7cd189c07.jpg)  
Figure 9: Attack 2: private samples (upper row of each block) and reconstructions from the FedAvg aggregate (lower row), matched within each class by minimum MSE. The first block uses the true label counts for m = 1 (evaluation only), and the others use the recovered counts.

## 6.4 Summary of the privacy results

Under the stated assumptions, a single head-only client update leaks its label set almost completely. Such leakage is the most reliable form observed in our experiments and requires little effort, using only the update and the public label space. Pixel reconstruction, in contrast, reproduces the class prior rather than the private drawing, because a linear head observes the input only through a 1280-dimensional embedding that is invariant to much of the stroke-level detail. FedAvg aggregation across as few as five clients reduces label-count recovery to the level of an uninformed guess for the estimator. Note that these conclusions are specific to head-only updates, a single SGD step, and an attacker that does not tamper with the model. Full-model updates, which expose convolutional gradients, and actively malicious servers are known to leak considerably more [9, 24, 27].

(a) clean  
![](images/2ab96b1b8ef55fd44de508c8446cdad080ddb22245f1c4224f8a660d6c354a98.jpg)  
label: circle

![](images/66eb010743a5c5498681dfe88c2925ac9a21277c3f8ebaebb2747f54a9bf4074.jpg)

(b) clean  
![](images/4838025097250c2cf07ce4b6a070b36e7ec160a22e3bdbcff4a81543cf3ed413.jpg)

![](images/c58c9a5eae48df60d140dab45a2a6b28813cc6a74f91e0dcc614fdfa2dc9c368.jpg)

(b) triggered  
![](images/9f142b3aaf0128947a6d888ebc0addf2c0e055e5b7eb0e5ff9179f495f56a3c5.jpg)

(c) trigger pattern  
![](images/cd4b12bf0da0ec9d5693e49dc4ed7af786f15ef245d16a02eba584c8862c7178.jpg)  
Figure 10: Poisoning mechanisms illustrated on Stage 1 sketches. (a) Label flipping changes only the label of circle samples to donut. (b) Backdoor poisoning adds a fixed trigger to 20% of the training samples of a malicious client, drawn from its non-donut samples, and relabels them as donut. (c) A 32-pixel trigger pattern.

## 7 Integrity: Data Poisoning under FedAvg

## 7.1 Threat model

We now consider active adversaries. A fraction $f \in \{ 0 , 0 . 1 , 0 . 3 , 0 . 5 , 1 . 0 \}$ of the ten Stage 2 clients are considered malicious (Notebook 5). Malicious clients modify their local training data, train the model by following the protocol, and upload the resulting heads, while the server runs the FedAvg protocol of Section 5. Each value of f is an independent 50-round run that starts from the Stage 1 head. For each f, we select the malicious clients with a random generator, and the resulting sets, listed in Table 9, are therefore not nested. With f = 0.3, for example, clients 4, 5, and 8 are malicious, whereas f = 0.5 selects clients 0, 1, 2, 8, and 9.

## 7.2 Attacks

Label flipping [10] relabels every circle training sample of a malicious client as donut, while the images remain unchanged. Backdoor poisoning [30] selects 20% of the training samples of a malicious client at random among those whose label is not donut, sets a fixed 32-pixel pattern in the bottom-right corner to white, and relabels them as donut (Figure 10). The pattern consists of a 4 × 4 square one pixel from the border, extended by two-pixel bands above and to its left. The extensions make it unlikely to coincide with natural strokes. Since the backdoor changes pixels, the embeddings of triggered images are recomputed through the frozen backbone.

Since the backbone is frozen, the attacker can only exploit features that the Stage 1 backbone already computes. A backdoor can succeed only if the trigger moves embeddings in a direction that a linear head can associate with donut. Such a setting is harder for the attacker than training from scratch, in which the convolutional layers could learn a dedicated trigger detector.

## 7.3 Metrics

After every round we measure the clean accuracy on the 5,000-image global evaluation set and the attack success rate (ASR). For label flipping, ASR is the percentage of the 500 clean circle evaluation images predicted as donut. For the backdoor, it is the percentage of the 4,500 non-donut evaluation images that are predicted as donut after the trigger is added. Of the 500 circle evaluation images, 403 were used to train the Stage 1 model, which may make them harder to flip than unseen images. The ASR at f = 0 is the background rate at which the model produces the target prediction without an attack. One image changes clean accuracy by 0.02 pp, label-flipping ASR by 0.2 pp, and backdoor ASR by 0.022 pp.

## 7.4 Results

Label flipping. Table 9 (FedAvg columns) and Figure 11 show that label flipping is ineffective unless it relabels most circle data. At f = 0.1, the single malicious client holds only 31 circle training samples, and ASR remains at the background rate of 0.60% (3 of 500 images). The case $f = 0 . 5$ is equally ineffective, although half of the clients and 50.1% of the FedAvg weight are malicious. These clients hold only 296 of the 1,381 circle training samples (21.4%). In contrast, the three malicious clients at $f = 0 . 3$ hold 64.3% of the circle data and raise ASR to 12.60% after 50 rounds. ASR is still rising at that point, from 11.80% at round 40, while clean accuracy falls to 96.48%. With $f = 1 . 0$ , every circle sample is relabeled. ASR then exceeds 50% at round 6 and 80% at round 7 and stays between 97.80% and 98.00% (489 to 490 of 500 images) thereafter, ending at 97.80%. Clean accuracy drops to 87.92%, which the attack itself explains. Since circle makes up 10% of the evaluation set, losing 97.8% of it costs 9.78 pp, close to the observed drop of 9.70 pp from the control. Figure 12(a) summarizes this behavior. ASR is governed by the share of

Table 9: Poisoning composition and final results after 50 rounds under FedAvg and FLTrust. “Weight” is the share of all client training samples held by malicious clients, i.e., their total FedAvg weight. “Poisoned” counts relabeled circle samples for label flipping (share of all 1,381 circle training samples) and triggered samples for the backdoor (share of all 12,310 training samples). Acc. denotes clean accuracy on the global evaluation set (%).
<table><tr><td colspan="6"></td><td colspan="2">FedAvg</td><td colspan="2">FLTrust</td></tr><tr><td>Attack</td><td>f</td><td>Malicious clients</td><td>Weight</td><td>Poisoned</td><td>Acc.</td><td>ASR (%)</td><td>Acc.</td><td>ASR (%)</td></tr><tr><td rowspan="5">Label flipping</td><td>0.0</td><td>一</td><td>0.0%</td><td>0 (0.0%)</td><td>97.62</td><td>0.60</td><td>97.66</td><td>0.60</td></tr><tr><td>0.1</td><td>2</td><td>9.8%</td><td>31 (2.2%)</td><td>97.64</td><td>0.60</td><td>97.64</td><td>0.60</td></tr><tr><td>0.3</td><td>4,5,8</td><td>34.3%</td><td>888 (64.3%)</td><td>96.48</td><td>12.60</td><td>97.64</td><td>0.80</td></tr><tr><td>0.5</td><td>0,1, 2,8,9</td><td>50.1%</td><td>296 (21.4%)</td><td>97.68</td><td>0.60</td><td>97.66</td><td>0.60</td></tr><tr><td>1.0</td><td>all</td><td>100.0%</td><td>1,381 (100%)</td><td>87.92</td><td>97.80</td><td>88.02</td><td>97.80</td></tr><tr><td rowspan="5">Backdoor</td><td>0.0</td><td>一</td><td>0.0%</td><td>0 (0.0%)</td><td>97.64</td><td>0.42</td><td>97.62</td><td>0.47</td></tr><tr><td>0.1</td><td>2</td><td>9.8%</td><td>242 (2.0%)</td><td>97.64</td><td>0.73</td><td>97.62</td><td>0.62</td></tr><tr><td>0.3</td><td>4,5,8</td><td>34.3%</td><td>843 (6.8%)</td><td>97.34</td><td>21.87</td><td>97.52</td><td>1.76</td></tr><tr><td>0.5</td><td>0,1,2,8,9</td><td>50.1%</td><td>1,234 (10.0%)</td><td>97.10</td><td>31.27</td><td>97.46</td><td>4.02</td></tr><tr><td>1.0</td><td>all</td><td>100.0%</td><td>2,462 (20.0%)</td><td>96.84</td><td>54.71</td><td>96.48</td><td>50.20</td></tr></table>

![](images/6a5f35da36993f2fbacffa44c140432707cade7baf655a609f4e81253772ee9c.jpg)

![](images/5b44e91fc625366a3d6586d5f2ed8f14eb4154768ff8e79ccd1b4a69ffc3b48e.jpg)  
Figure 11: Label flipping (circle→donut) over 50 rounds under FedAvg (left) and FLTrust (right). The top row shows ASR on clean circle evaluation images, and the bottom row clean accuracy on the global evaluation set. Color encodes the malicious fraction $f ,$ and the dashed gray line is the $f = 0$ control. End labels give the final ASR where it exceeds 5%. Under FedAvg, only $f = 0 . 3$ (64% of circle data relabeled) and f = 1.0 (100%) have an effect. FLTrust removes the f = 0.3 effect and delays the f = 1.0 attack (80% ASR at round 9 instead of round 7).

Relabeled share of source-class (circle) training samples (%)

![](images/9d2d949579bc4abc4c8fca1aa973e35c6b8299c45be76b9d16906e2103d5646c.jpg)

![](images/4acbf2398fe7fe0c73a0a864168b37bf4724e58d7caaf3937c1baa7086b92a9a.jpg)  
Figure 12: Final ASR after 50 rounds as a function of the poisoned data mass. (a) Label flipping against the share of circle training samples that are relabeled. The non-nested malicious sets place $f = 0 . 5$ to the left of $f = 0 . 3 .$ (b) Backdoor against the share of all client training samples that are triggered. Labels give the malicious fraction f of each FedAvg point.

![](images/77414984e2852f0865dd1b75f11c85f3a5a646b444401a063e2d3a8feed7c61e.jpg)  
Figure 13: Backdoor poisoning over 50 rounds under FedAvg (left) and FLTrust (right). The top row shows ASR on triggered non-donut evaluation images, and the bottom row clean accuracy on the global evaluation set. Under FedAvg, ASR grows steadily with $f ,$ while final clean accuracy stays within 0.8 pp of the control (with a transient dip of 1.8 pp at $f = 1 . 0 \dot { ) }$ . FLTrust keeps ASR below 4.1% for $f \leq 0 . 5$ , but not for $\dot { \boldsymbol { f } } = 1 . 0$

source-class training data that is relabeled, and not by the malicious fraction $f .$ Such behavior is expected for FedAvg over a frozen backbone since the aggregate approximately minimizes the pooled loss of all clients, and the pooled circle data favor donut only when most of them carry that label.

Backdoor. The backdoor attack behaves differently (Figure 13). Its ASR increases monotonically with the amount of triggered data, i.e., 0.73% at f = 0.1 (2.0% of all training samples triggered), 21.87% at $f = \bar { 0 . 3 } ( 6 . 8 \% )$ , 31.27% at $f = 0 . 5 \ : ( 1 0 . 0 \% )$ , and 54.71% at $f = 1 . 0 \ : ( 2 0 . 0 \% )$ , against a background rate of 0.42%. The final clean accuracy decreases by at most 0.80 pp relative to the control (96.84% at $f = 1 . 0 )$ , and by 0.30 pp and 0.54 pp at $f = 0 . 3$ and $f = 0 . 5$ , which is close to the resolution of a single run. The backdoor is therefore stealthy in the sense that clean accuracy alone would not reveal it. At the same time it remains incomplete. No configuration reaches 80% ASR within 50 rounds, and the $f = 1 . 0$ run reaches 50% at round 38. The ASR curves are still increasing at round 50 (for example, from 27.31% to 31.27% between rounds 40 and 50 at $f = 0 . 5 )$ , so longer training would likely strengthen the backdoor. The slow growth is consistent with the frozen backbone, since the head must learn the backdoor from the trigger-related variation the Stage 1 features encode. For $f = 1 . 0$ , clean accuracy first falls to 95.88% at round 10 and then partially recovers to 96.84% as the head learns to separate triggered from clean embeddings.

Algorithm 2 One FLTrust round over classifier heads   
Require: global head $\theta _ { t - 1 } ;$ client heads $\theta _ { t } ^ { 1 } , \ldots , \theta _ { t } ^ { K } ;$ ; root embeddings $\mathcal { D } _ { 0 }$   
1: ∆ ← LocalAdamEpoch $( \theta _ { t - 1 } , \mathcal { D } _ { 0 } ) \bar { - } \theta _ { t - 1 }$ ▷ trusted update   
2: for $k = 1 , \ldots , K$ do   
3: $\begin{array} { r } { \underline { { \Delta } } _ { k }  \theta _ { t } ^ { k } - \theta _ { t - 1 } ; \quad s _ { k }  \operatorname* { m a x } ( 0 , \cos ( \Delta _ { k } , \Delta _ { 0 } ) ) ; \quad \tilde { \Delta } _ { k }  \frac { \| \Delta _ { 0 } \| } { \| \Delta _ { k } \| } \Delta _ { k } } \end{array}$   
4: end for   
5: $\begin{array} { r } { \mathbf { i f } \sum _ { k } s _ { k } = 0 } \end{array}$ then   
6: $\overline { { \theta _ { t } } } \gets \theta _ { t - 1 } + \Delta _ { 0 }$   
7: else   
8: $\begin{array} { r } { \underrightarrow { \theta _ { t } }  \theta _ { t - 1 } + \sum _ { k } s _ { k } \tilde { \Delta } _ { k } / \sum _ { k } s _ { k } } \end{array}$   
9: end if   
10: return $\theta _ { t }$

## 8 Defense: FLTrust with a Retained Root Set

## 8.1 Method

FLTrust [15] assumes that the server owns a small clean root dataset and uses it to compute a reference update in every round. In our pipeline this assumption is inexpensive, since the provider performed Stage 1 on clean data and can retain part of it. In Notebook 6, we sample a root set of 100 images per class (1,000 in total) from the training part of the Stage 1 split, which is disjoint from the global evaluation set (Table 2). In each round t, the server trains a copy of the current head $\theta _ { t - 1 }$ on the root embeddings with the same local procedure as the clients (one epoch of Adam, learning rate $1 0 ^ { - 3 }$ , batch size 128) to obtain the trusted update $\Delta _ { 0 }$ . Each client update $\Delta _ { k } = \theta _ { t } ^ { k } - \theta _ { t - 1 }$ receives the trust score

$$
s _ { k } = \operatorname* { m a x } \biggl ( 0 , \frac { \langle \Delta _ { k } , \Delta _ { 0 } \rangle } { \| \Delta _ { k } \| _ { 2 } \| \Delta _ { 0 } \| _ { 2 } } \biggr )\tag{8}
$$

and is rescaled to the norm of the trusted update, $\tilde { \Delta } _ { k } = ( \| \Delta _ { 0 } \| _ { 2 } / \| \Delta _ { k } \| _ { 2 } ) \Delta _ { k }$ . The server then updates using

$$
\theta _ { t } = \theta _ { t - 1 } + \frac { \sum _ { k = 1 } ^ { K } s _ { k } \tilde { \Delta } _ { k } } { \sum _ { k = 1 } ^ { K } s _ { k } } .\tag{9}
$$

If all scores are zero, the server applies $\Delta _ { 0 }$ instead. Updates that point away from the trusted direction are discarded, and no update can dominate a round through its magnitude. Unlike FedAvg, Eq. (9) does not weight clients by sample count, so the influence of a client depends only on the direction of its update. The trusted update itself enters only through the scores and the normalization. Algorithm 2 summarizes one round. All other settings, including the client datasets, malicious-client sets, poisoning rules, local optimizer, and number of rounds, are identical to Section 7.

## 8.2 Results

Clean utility. With no malicious clients, FLTrust reaches 97.66% (label-flipping control) and 97.62% (backdoor control) clean accuracy, compared with 97.62% and 97.64% under FedAvg (Table 9). Replacing sample-weighted averaging by trust-weighted, norm-equalized averaging therefore costs no measurable utility in this setting

Label flipping. FLTrust removes the only partially successful label-flipping attack. At f = 0.3, ASR stays at 0.80% instead of 12.60%, and clean accuracy at 97.64% instead of 96.48% (Figure 11). The settings that were already ineffective under FedAvg remain at the background rate. When all clients are malicious, the outcome matches FedAvg (97.80% ASR and 88.02% clean accuracy). The only difference is timing, since ASR crosses 50% and 80% at round 9 instead of rounds 6 and 7.

Table 10: FLTrust trust diagnostics averaged over 50 rounds. “FedAvg weight” is the aggregated weight of malicious clients under Eq. (2). “FLTrust share” is their share of the total trust, $\sum _ { k \in \mathcal { M } } { s _ { k } } / { \sum _ { k } }$ s<sub>k</sub>, computed per round from the mean honest and malicious trust and averaged over rounds.
<table><tr><td colspan="2"></td><td colspan="2">Mean trust score</td><td colspan="2">Malicious aggregation weight</td><td colspan="2"></td></tr><tr><td>Attack</td><td>f</td><td>Honest</td><td>Malicious</td><td>FedAvg weight</td><td>FLTrust share</td><td>ASR FedAvg (%)</td><td>ASR FLTrust (%)</td></tr><tr><td rowspan="5">Label flipping</td><td>0.0</td><td>0.384</td><td></td><td>0.0%</td><td>0.0%</td><td>0.60</td><td>0.60</td></tr><tr><td>0.1</td><td>0.387</td><td>0.234</td><td>9.8%</td><td>6.3%</td><td>0.60</td><td>0.60</td></tr><tr><td>0.3</td><td>0.413</td><td>0.227</td><td>34.3%</td><td>19.0%</td><td>12.60</td><td>0.80</td></tr><tr><td>0.5</td><td>0.429</td><td>0.285</td><td>50.1%</td><td>39.8%</td><td>0.60</td><td>0.60</td></tr><tr><td>1.0</td><td>一</td><td>0.230</td><td>100.0%</td><td>100.0%</td><td>97.80</td><td>97.80</td></tr><tr><td rowspan="5">Backdoor</td><td>0.0</td><td>0.386</td><td></td><td>0.0%</td><td>0.0%</td><td>0.42</td><td>0.47</td></tr><tr><td>0.1</td><td>0.385</td><td>0.187</td><td>9.8%</td><td>5.1%</td><td>0.73</td><td>0.62</td></tr><tr><td>0.3</td><td>0.419</td><td>0.219</td><td>34.3%</td><td>18.2%</td><td>21.87</td><td>1.76</td></tr><tr><td>0.5</td><td>0.429</td><td>0.159</td><td>50.1%</td><td>26.6% 100.0%</td><td>31.27</td><td>4.02</td></tr><tr><td>1.0</td><td></td><td>0.250</td><td>100.0%</td><td></td><td>54.71</td><td>50.20</td></tr></table>

Backdoor. The effect of the defense is most visible for the backdoor (Figure 13). ASR is reduced from 21.87% to 1.76% at $f = 0 . 3$ and from 31.27% to 4.02% at $f = 0 . 5$ , while clean accuracy stays within a 0.2 pp margin of the attack-free control. Note, however, that ASR under FLTrust increases slowly (from 3.00% at round 40 to 4.02% at round 50 for $f = 0 . 5 )$ . The defense therefore slows the backdoor substantially rather than eliminating it. $\mathrm { A t } \ f = 1 . 0$ ASR reaches 50.20%, compared with 54.71% under FedAvg, and clean accuracy 96.48%. Figure 12 presents both aggregators. Under FLTrust, ASR stays near zero until all clients are malicious.

Trust analysis. Table 10 and Figure 14 explain the mechanism. In every run and round, all ten clients received a positive trust score, so FLTrust never excluded a client outright. Malicious updates are nevertheless consistently less aligned with the trusted update. Averaged over rounds, honest clients obtain trust scores of 0.38 to 0.43, and malicious clients 0.16 to 0.29. Since Eq. (9) weights norm-equalized updates by trust, malicious clients receive a smaller share of the aggregate than under FedAvg. For the backdoor, their share falls from 34.3% to 18.2% at $f = 0 . 3$ and from 50.1% to 26.6% at $f = 0 . 5$ . The reduction in ASR is larger than the reduction in weight, which indicates that norm equalization also matters. Under FedAvg, the update of a malicious client enters with its full magnitude, whereas under FLTrust it cannot exceed the magnitude of the trusted update. For label flipping at $f = 0 . 5 ,$ , malicious trust (0.285) is the highest among the attacked settings, because only 21.4% of the circle data are relabeled and the malicious updates are mostly benign. At f = 1.0, every update is poisoned, and FLTrust can only reweight poisoned updates among themselves. Their trust remains positive (0.23 to 0.25) because most of the data of each malicious client are clean (80% for the backdoor and at least 64% for label flipping). The poisoned updates therefore still agree with the trusted update in most directions. The poisoned component passes into the aggregation, and only its magnitude is limited, which accounts for the small delay of the label-flipping attack and the modest ASR reduction of the backdoor.

## 9 Computational Footprint

All six notebooks were executed on Google Colab with a GPU runtime, and Table 11 lists the runtime and peak memory they recorded. The peak GPU memory allocated by PyTorch remains below 600 MiB in every notebook, and the federated experiments operate on precomputed embeddings. Any participant can therefore reproduce the complete pipeline in a browser, without a local installation or a dedicated GPU.

Centralized fine-tuning of the full network dominates the cost of the tutorial, at 227.7 s per epoch on average and 56.9 min for 15 epochs. On the other hand, the operations that happen on the client side are cheap. A single head-only SGD step on five images takes 0.086 s and increases allocated GPU memory by 15.3 MiB. Fifty rounds of head-only FedAvg over ten clients take 12.55 s for local training and aggregation, because the embeddings are precomputed once. The uploaded head of 12,810 parameters (50 KiB in 32-bit floating point) is 76 times smaller than the full model. FLTrust adds the training of the trusted head to every round, which increases the time of a 50-round run from 11.57 s to 12.31 s on average over the ten scenarios (+6.4%). The privacy attacks are the most expensive operations of the federated stage, because every optimization step back-propagates through the frozen backbone. Their runtime decreases with m only because the optimization budget is reduced, while peak memory grows with the number of canvases.

![](images/bcc9485e78f4302dcac4b13e9917c7ad0fb2751c426b41102b3373cdc0ff34d0.jpg)  
Figure 14: FLTrust trust scores (Eq. (8)) per round, averaged over honest (solid) and malicious (dashed) clients, for label flipping (top) and the backdoor (bottom). At f = 1.0 there are no honest clients, and the dotted line shows the honest trust of the f = 0 run for reference.

Table 11: Runtime and peak memory as recorded by the notebooks on a Google Colab GPU runtime. RAM is the peak resident set size of the Python process, and VRAM is the peak memory allocated by PyTorch. Federated runtimes exclude data loading, hard-example selection, and the one-time embedding precomputation. n/r: not recorded.
<table><tr><td>Notebook</td><td>Operation</td><td>Runtime</td><td>Peak RAM (MiB)</td><td>Peak VRAM (MiB)</td></tr><tr><td>NB1</td><td>Centralized fine-tuning, 15 epochs of 40,000 images</td><td>56.9 min</td><td>n/r</td><td>n/r</td></tr><tr><td>NB2</td><td>One head-only SGD step on 5 images (client side)</td><td>0.086 s</td><td>1288.1</td><td>35.0</td></tr><tr><td>NB3</td><td>Attack 2, complete experiment, m = 1 / 2 / 393.9 / 86.0 / 53.1 / 33.9 s 5 / 10</td><td></td><td>1689-1782</td><td>100-571</td></tr><tr><td>NB4</td><td>Clean FedAvg, 50 rounds, 10 clients (train- ing and aggregation)</td><td>12.55 s</td><td>2333.5</td><td>31.3</td></tr><tr><td>NB5</td><td>FedAvg under poisoning, per 50-round sce- nario</td><td>11.19–12.10 s</td><td>≤2444.7</td><td>31.3</td></tr><tr><td>NB6</td><td>FLTrust under poisoning, per 50-round sce- nario</td><td>12.03–12.70 s</td><td>≤2448.9</td><td>31.3</td></tr></table>

## 10 Live Session: Federated Learning with FedMaestro

The simulations above provide the controlled and reproducible evidence. The live session complements them by exposing the participants of the summer school to the operational side of FL through FedMaestro, the federated orchestrator of Indigma, which manages model distribution, client participation, round execution, update collection, aggregation, and monitoring. Each participant joins the federated network as a client from their own device, and the session starts from the Stage 1 checkpoint of Section 4, which serves as the initial global model. The session follows the order of this report, i.e., inference with the centrally trained model, federated fine-tuning, and poisoning.

Scope. The live session is a demonstration of an online federated environment. The number of participants, their drawing styles, and their devices are not controlled, and several of these quantities are smaller than in the simulations (e.g., 5 drawings per client). We therefore report no quantitative results from the session. The statements below describe what the earlier sections lead us to expect, and the reproducible evidence for each of them is the corresponding notebook.

![](images/ba683f8ad831cf5d4d9939ff087a90cc6fc27a50cab129eaf43ac9976547e865.jpg)  
Figure 15: Participant view of the live session. The participant selects a class, draws it on the canvas, and receives the five most probable classes under the current model served by the orchestrator, together with their confidences.

Step 1: direct inference. Participants select one of the ten classes, draw it once on a canvas connected to the deployed Stage 1 model, and receive the five most probable classes with their confidences (Figure 15). The drawings are made in black on a white canvas. They are converted to the QuickDraw format, i.e., 28 × 28 grayscale bitmaps with white strokes on a black background, and then processed as in Section 3.3. Even so, they differ from the crowd-sourced training data. They are drawn with a different input device and under no time pressure, and their stroke widths and positions on the canvas follow the participant rather than the original game. We therefore expect drawings of most classes to be recognized. Errors should concentrate on the visually related pairs identified in Section 3.2, such a circle/donut and sun/star, and on drawings that depart from the typical shape of a class. Such drawings are also the samples that the Stage 1 model misclassified in Sections 6.2 and 6.3. The ranked list additionally shows which classes the model considers as alternatives, so participants can see model confusions even when the top prediction is correct. This step lets participants identify the classes that the model handles poorly for their own drawings.

Step 2: federated fine-tuning on participant drawings. Participants contribute new labeled drawings, which remain on their devices and serve as local training data. In every round, FedMaestro distributes the current global model, each participating client trains locally and returns its update, and the server aggregates the updates into a new model version. The administrator dashboard lists every round with its status and the number of updates received from the participating clients (Figure 16). Two points from the simulations are made explicit to participants. First, the model starts from an accurate checkpoint, and fine-tuning adapts it. In Section 5, head-only FedAvg changed accuracy by about one percentage point over 50 rounds. With far fewer rounds and drawings in the live session, we expect visible changes to be limited to the classes and drawing styles that participants actually contribute. Second, keeping the drawings on the device does not make the update uninformative. Section 6.2 showed that a single head update reveals which classes its sender drew, while pixel reconstruction recovers only class prototypes. Aggregation over several participants weakens this label leakage (Section 6.3).

<table><tr><td>Round progress</td><td></td><td></td><td></td></tr><tr><td>Round Status clean-1 COMPLETE</td><td>Updates 1/1</td><td>Votes</td><td>Global loss 1.96044</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>clean-2 COMPLETE</td><td>1/1</td><td></td><td>1.89769</td></tr><tr><td>poison-1 ANALYZED</td><td>1/1</td><td>1/1</td><td>1.83678</td></tr><tr><td>poison-2 ANALYZED</td><td>1/1</td><td>1/1</td><td>1.77771</td></tr></table>

Figure 16: Round progress in the FedMaestro administrator dashboard. Clean rounds (Step 2) and poisoning rounds (Step 4) are listed separately with their status, the number of received updates, and the global loss reported by the orchestrator. The screenshot was taken during a test run with a single client.

Step 3: observing whether the model learns. After one or more rounds, participants query the updated global model and compare its predictions with those of Step 1. This step connects the abstract FedAvg update of Eq. (2) with an observable change in model behavior. The clearest comparison resubmits the drawings of Step 1 to the updated model. We expect the most visible changes on drawings that resemble the contributed training data and on classes that were confused in Step 1. Predictions on canonical drawings should remain stable, consistent with the preserved global accuracy in Section 5.

Step 4: participant-driven poisoning. Finally, a group of participants deliberately poisons its local data, for example by consistently labeling one class as another, which corresponds to the label-flipping attack of Section 7. The poisoning rounds run in the same federated network and appear on the administrator dashboard separately from the clean rounds (Figure 16). Participants then check whether the global model starts to produce the target class of the attacker. The simulations lead to two expectations. First, under FedAvg the outcome depends on the share of the source-class data that is relabeled (Figure 12a). A single poisoner should have little effect as long as honest participants contribute drawings of the same class, whereas coordinated relabeling of most source-class drawings should flip the class within a few rounds. Second, the clean Stage 1 data provide the root set that FLTrust requires. With FLTrust, partial attacks are expected to be suppressed, and the influence of a large contribution is bounded by norm equalization (Section 8). An attack in which every participant relabels the same class, however, is expected to succeed under either aggregation rule. Running the poisoning step once with each aggregation rule lets participants compare these outcomes directly.

## 11 Discussion

Lessons for post-training federated personalization. Three observations carry over beyond this specific pipeline. First, privacy leakage and integrity risk should be assessed for the update that is actually exchanged. A head-only update of 12,810 parameters reveals the label composition of a small client batch almost exactly but carries little pixel-level information. In this regime, label privacy rather than image privacy is therefore the primary concern. Second, the strength of a data poisoning attack is determined by how much poisoned data reaches the aggregation and how they are distributed across clients. In non-IID populations, a minority of clients that holds most of a class can cause more damage than a majority that holds little of it (Table 9). Third, the centralized stage that precedes federated personalization naturally provides the clean root data that FLTrust requires. In our experiments, the defense suppressed all partia attacks without cost to clean utility.

From convergence to production-ready FL. Many FL studies evaluate a federated method mainly by whether and how fast it converges. Our results suggest that a deployment cannot be judged this way alone. The same head-only FedAvg procedure that converges cleanly in Section 5 reveals the label set of a single client to an honest-but-curious server (Section 6.2) and accepts a backdoor that clean accuracy alone does not reveal (Section 7). In real deployments, curious servers and malicious or faulty clients have to be expected. A production-ready FL system therefore needs components beyond the training loop, such as robust or trust-based aggregation [13–15], filtering and validation of incoming updates, secure aggregation and differential privacy [28, 29], client authentication, and monitoring of every round with the ability to roll back a model version. Each of these components has its own requirements and costs. FLTrust, for example, needs a clean root set and offers no protection when every client is malicious (Section 8). Filters that discard outlying updates can also exclude honest clients whose data are unusual but legitimate, which is a known difficulty in non-IID populations. FL is therefore a complete system rather than an optimization algorithm, and building, operating, and auditing such a system requires engineering effort and expertise beyond those of centralized training [2, 45]. The live session of Section 10 gives participants a view of this operational side, in which rounds, participating clients, and received updates are tracked explicitly. We believe that this gap between a converging algorithm and a deployable system, more than the learning algorithms themselves, is one reason why many organizations that could benefit from FL do not, or cannot, adopt it, at least for now [46].

Limitations. The global evaluation set of Notebook 4–Notebook 6 was rebuilt with a different seed than the Stage 1 split and consists mostly of Stage 1 training images (Section 3.2). Global accuracies in the federated experiments are therefore comparable with one another but not with the Stage 1 test accuracy, and rebuilding the split with the Stage 1 seed would remove this caveat. Every configuration was executed once with a fixed seed, so the results carry no confidence intervals. The private samples of the privacy attacks and the client data were selected for high loss under the Stage 1 model. Such a selection strengthens the gradients available to the attacker and makes the federated stage non-trivial, but it also makes the private sketches atypical, which affects the reconstruction metrics. The privacy attacks assume one plain SGD step so that Eqs. (3) and (4) hold exactly, whereas Stage 2 uses one epoch of Adam, for which the update is not a scaled gradient. The differentiable rendering of the attacker upsamples canvases directly to 224 × 224 and omits the resize-and-center-crop of the client pipeline. The resulting scale mismatch of about 14% may weaken pixel reconstruction. The SSIM used throughout is a single-window approximation, and MSE and PSNR favor empty canvases on sparse sketches (Section 6.2). The FLTrust root set is drawn from the same distribution as the global evaluation set, which favors the defense. A root set that differs from the client distribution would yield noisier trust scores. Finally, the attackers are not adaptive, i.e., they neither scale their updates nor optimize them to evade a known aggregation rule, and such attacks are known to defeat several robust aggregators [34, 35].

## 12 Conclusions and Future Work

In this report we presented an end-to-end tutorial pipeline in which a pretrained MobileNetV4 is fine-tuned centrally on QuickDraw-10 (97.50% test accuracy) and then personalized with head-only FedAvg over ten non-IID clients without loss of global accuracy. On this pipeline, a curious server recovered the private label set of a single client update, while pixel reconstructions reproduced class prototypes rather than the private drawings. The aggregation over five or more clients reduced label-count recovery to the level of an uninformed guess. Malicious clients implanted a backdoor that reached 54.7% ASR at a final clean-accuracy cost below one percentage point, whereas label flipping succeeded only when most of the source-class data were relabeled. FLTrust with a root set retained from centralized fine-tuning suppressed every partial attack without measurable cost, but it could not protect a federation in which every client is malicious. Together with the live FedMaestro session, the six notebooks are intended to make these privacy and integrity risks concrete for the participants of the summer school and to provide a reproducible starting point for studying defenses.

Several open problems remain. The results are based on single runs, on head-only updates, and on attackers that neither tamper with the model nor adapt to the aggregation rule. Future work will extend the tutorial in five directions. First, we plan to add differentially private local training and secure aggregation against the attacks of Section 6, with an explicit privacy-utility trade-off. Second, we will study partial backbone fine-tuning, which exposes convolutional gradients and allows trigger-specific features. Third, we will compare FLTrust with other robust aggregation mechanisms on the same client population. Fourth, we will consider adaptive attackers that constrain their updates to resemble honest ones. Finally, we will repeat all experiments with several seeds and client populations to obtain confidence intervals.

## Code and Data Availability

The six executed notebooks, including all outputs from which the results in this report are taken, are available at https://github.com/Indigma-Innovations/fl-summer-school under the Apache License 2.0. All notebooks run on Google Colab, so anyone can reproduce the experiments in a browser without a local installation or high-end hardware (Section 9). The QuickDraw data are downloaded from the public Quick, Draw! repository [17] and remain subject to their CC BY 4.0 license, while the pretrained weights are subject to the license of their distributor [36].

## References

[1] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communicationefficient learning of deep networks from decentralized data. In Proceedings of the 20th International Conference

on Artificial Intelligence and Statistics (AISTATS), volume 54 of Proceedings of Machine Learning Research, pages 1273–1282, 2017.

[2] Peter Kairouz, H. Brendan McMahan, et al. Advances and open problems in federated learning. Foundations and Trends in Machine Learning, 14(1–2):1–210, 2021. doi:10.1561/2200000083.

[3] John Nguyen, Jianyu Wang, Kshitiz Malik, Maziar Sanjabi, and Michael Rabbat. Where to begin? On the impact of pre-training and initialization in federated learning. In International Conference on Learning Representations (ICLR), 2023.

[4] Hong-You Chen, Cheng-Hao Tu, Ziwei Li, Han-Wei Shen, and Wei-Lun Chao. On the importance and applicability of pre-training for federated learning. In International Conference on Learning Representations (ICLR), 2023.

[5] Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. Federated learning with personalization layers. arXiv preprint arXiv:1912.00818, 2019.

[6] Liam Collins, Hamed Hassani, Aryan Mokhtari, and Sanjay Shakkottai. Exploiting shared representations for personalized federated learning. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings ofMachine Learning Research, pages 2089–2099, 2021.

[7] Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. Exploiting unintended feature leakage in collaborative learning. In IEEE Symposium on Security and Privacy (S&P), pages 691–706, 2019. doi:10.1109/SP.2019.00029.

[8] Ligeng Zhu, Zhijian Liu, and Song Han. Deep leakage from gradients. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

[9] Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, and Michael Moeller. Inverting gradients – how easy is it to break privacy in federated learning? In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 16937–16947, 2020.

[10] Vale Tolpegin, Stacey Truex, Mehmet Emre Gursoy, and Ling Liu. Data poisoning attacks against federated learning systems. In European Symposium on Research in Computer Security (ESORICS), volume 12308 of Lecture Notes in Computer Science, pages 480–501. Springer, 2020. doi:10.1007/978-3-030-58951-6\_24.

[11] Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov. How to backdoor federated learning. In Proceedings ofthe 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), volume 108 of Proceedings ofMachine Learning Research, pages 2938–2948, 2020.

[12] Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, and Seraphin Calo. Analyzing federated learning through an adversarial lens. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings ofMachine Learning Research, pages 634–643, 2019.

[13] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

[14] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80 of Proceedings ofMachine Learning Research, pages 5650–5659, 2018.

[15] Xiaoyu Cao, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong. FLTrust: Byzantine-robust federated learning via trust bootstrapping. In Network and Distributed System Security Symposium (NDSS), 2021. doi:10.14722/ndss.2021.24434.

[16] Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, and Andrew Howard. MobileNetV4: Universal models for the mobile ecosystem. In European Conference on Computer Vision (ECCV), 2024. arXiv:2404.10518.

[17] Google Creative Lab. The Quick, Draw! dataset. https://github.com/googlecreativelab/ quickdraw-dataset, 2017. Released under CC BY 4.0.

[18] David Ha and Douglas Eck. A neural representation of sketch drawings. In International Conference on Learning Representations (ICLR), 2018.

[19] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335, 2019.

[20] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255, 2009. doi:10.1109/CVPR.2009.5206848.

[21] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neura networks? In Advances in Neural Information Processing Systems (NeurIPS), volume 27, 2014.

[22] Simon Kornblith, Jonathon Shlens, and Quoc V. Le. Do better ImageNet models transfer better? In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2661–2671, 2019.

[23] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. iDLG: Improved deep leakage from gradients. arXiv preprint arXiv:2001.02610, 2020.

[24] Hongxu Yin, Arun Mallya, Arash Vahdat, Jose M. Alvarez, Jan Kautz, and Pavlo Molchanov. See through gradients: Image batch recovery via GradInversion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16337–16346, 2021.

[25] Aidmar Wainakh, Fabrizio Ventola, Till Müßig, Jens Keim, Carlos Garcia Cordero, Ephraim Zimmer, Tim Grube, Kristian Kersting, and Max Mühlhäuser. User-level label leakage from gradients in federated learning. Proceedings on Privacy Enhancing Technologies, 2022(2):227–244, 2022. doi:10.2478/popets-2022-0043.

[26] Leonid I. Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena, 60(1–4):259–268, 1992. doi:10.1016/0167-2789(92)90242-F.

[27] Yangsibo Huang, Samyak Gupta, Zhao Song, Kai Li, and Sanjeev Arora. Evaluating gradient inversion attacks and defenses in federated learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, 2021.

[28] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacy-preserving machine learning. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 1175–1191, 2017. doi:10.1145/3133956.3133982.

[29] Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 308–318, 2016. doi:10.1145/2976749.2978318.

[30] Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. BadNets: Identifying vulnerabilities in the machine learning model supply chain. arXiv preprint arXiv:1708.06733, 2017.

[31] Ziteng Sun, Peter Kairouz, Ananda Theertha Suresh, and H. Brendan McMahan. Can you really backdoor federated learning? arXiv preprint arXiv:1911.07963, 2019.

[32] Hongyi Wang, Kartik Sreenivasan, Shashank Rajput, Harit Vishwakarma, Saurabh Agarwal, Jy-yong Sohn, Kangwook Lee, and Dimitris Papailiopoulos. Attack of the tails: Yes, you really can backdoor federated learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, 2020.

[33] El Mahdi El Mhamdi, Rachid Guerraoui, and Sébastien Rouault. The hidden vulnerability of distributed learning in Byzantium. In Proceedings ofthe 35th International Conference on Machine Learning (ICML), volume 80 of Proceedings ofMachine Learning Research, pages 3521–3530, 2018.

[34] Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong. Local model poisoning attacks to Byzantinerobust federated learning. In 29th USENIX Security Symposium, pages 1605–1622, 2020.

[35] Virat Shejwalkar and Amir Houmansadr. Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning. In Network and Distributed System Security Symposium (NDSS), 2021.

[36] Ross Wightman. PyTorch image models. https://github.com/huggingface/pytorch-image-models, 2019.

[37] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.

[38] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations (ICLR), 2017.

[39] Hugging Face. Safetensors: A simple, safe way to store and distribute tensors. https://github.com/ huggingface/safetensors, 2022.

[40] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[41] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. doi:10.1109/TIP.2003.819861.

[42] Dieter Kraft. A software package for sequential quadratic programming. Technical Report DFVLR-FB 88-28, DFVLR, Institut für Dynamik der Flugsysteme, Oberpfaffenhofen, Germany, 1988.

[43] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, et al. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17:261–272, 2020. doi:10.1038/s41592-019-0686-2.

[44] Harold W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2 (1–2):83–97, 1955. doi:10.1002/nav.3800020109.

[45] Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloé Kiddon, Jakub Konecný, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Danielˇ Ramage, and Jason Roselander. Towards federated learning at scale: System design. In Proceedings ofMachine Learning and Systems (MLSys), volume 1, pages 374–388, 2019.

[46] Vasileios Perifanis, Nikolaos Pavlidis, Foteini Nikolaidou, D. Kampouri, and Pavlos S. Efraimidis. Bridging the gap: Challenges and limitations of federated learning in real-world applications. In 2025 10th International Conference on Smart and Sustainable Technologies (SpliTech), pages 1–6. IEEE, 2025. doi:10.23919/SpliTech65624.2025.11091634.