# PERSONALIZED FEDERATED LEARNING THROUGH GLOBAL KNOWLEDGE DISTILLATION AND LOCAL HEAD ADAPTATION

Polycarpo Souza Neto<sup>1</sup>, Jose Mairton Barros da Silva Jr. ´ <sup>2</sup>, Charles Casimiro Cavalcante<sup>1</sup>

<sup>1</sup>Universidade Federal do Ceara, Fortaleza, CE, Brazil´ <sup>2</sup>Uppsala University, Uppsala, Sweden

## ABSTRACT

Statistical heterogeneity limits federated learning when a single global classifier cannot represent client-specific label distributions. In this work, we propose Personalized Federated Knowledge Distillation with Head Adaptation (pFedKDH), which aggregates only the shared backbone, keeps persistent client-specific heads, and uses a recalibrated global head as a teacher during local training. Across MNIST, Fashion-MNIST, CIFAR10, and CIFAR100 under class-wise Dirichlet partitions, pFedKDH obtains the best accuracy in most settings, with accuracy gaps up to 37.67% over the weakest baseline and consistently low standard deviation across repetitions. Component-wise diagnostics and convergence results support the role of persistent heads and distillation-guided local optimization under label-skewed data.

Index Terms— Personalized federated learning, knowledge distillation, non-IID data, model personalization.

## 1. INTRODUCTION

Federated learning (FL) enables collaborative model training without moving raw client data to a central server [1, 2]. In its standard form, selected clients receive a global model, update it using local data, and send the updated parameters back to the server for aggregation. This setting is attractive for distributed visual learning and edge intelligence, where data are decentralized and may be privacysensitive. However, when clients follow different local distributions, server aggregation may mix incompatible class evidence and local classification rules.

Recent surveys show that FL has been widely explored in privacy-sensitive and distributed domains, including healthcare, finance, IoT, edge computing, and decentralized intelligent systems, but non-IID data remains a central challenge for convergence, communication efficiency, and robustness [3, 4]. In this work, we focus on label skew, a common form of statistical heterogeneity in which clients differ in their class distributions [5]. Under this setting, local classifiers may learn client-specific decision boundaries that become incompatible when the entire model is averaged.

Personalized federated learning (PFL) addresses this issue by allowing each client to use a model that differs from a single shared model [6, 7]. Existing methods regularize personalized models toward a global reference, separate shared and local parameters, adapt aggregation weights, group similar clients, or transfer auxiliary information during training. Although these strategies improve over standard FL, important gaps remain: full-model regularization still couples all parameters to a global reference, adaptive aggregation still mixes model parameters, and backbone-head separation does not necessarily provide a global teacher signal during local optimization.

Our proposed method follows from a simple observation. In visual classification, the backbone learns representations that can be useful across clients, while the classifier head maps these representations to the local label distribution. Therefore, the backbone should be shared, but the classifier head should remain local under strong label skew. At the same time, keeping heads local should not discard global knowledge.

Based on this observation, we introduce Personalized Federated Knowledge Distillation with Head Adaptation (pFedKDH). The method aggregates only the backbone, keeps a persistent clientspecific head at each client, and uses a recalibrated global head as a teacher during local training. Global knowledge is transferred through the teacher signal and the shared backbone, rather than through direct aggregation of classifier parameters. The local objective combines supervised learning, distillation, and proximal regularization. After local training, only the updated backbone is returned to the server, while the client-specific head remains persistent across communication rounds.

We evaluate pFedKDH on MNIST, Fashion-MNIST, CIFAR10, and CIFAR100 using class-wise Dirichlet partitions with α ∈ {0.05, 0.10, 0.50}. The comparison includes classical FL baselines and representative personalized methods under a standardized CNN backbone whenever applicable. Across the evaluated settings, pFedKDH achieves the strongest overall accuracy-stability trade-off. It obtains the best mean accuracy in most cases and remains close to the best method in the remaining ones. Moreover, its standard deviation is consistently small compared with competing methods, indicating that the observed gains are stable across repetitions rather than caused by isolated runs. In the settings where pFedKDH leads, the accuracy gap reaches up to 37.67 % over the weakest compared method and up to 1.82 % over the strongest competitor. This performance is obtained with a competitive computational cost, pFedKDH requires a per-round time comparable to FedALA and FedBABU, and substantially lower than FedRep and pFedMe.

The main contributions of this work are summarized as follows:

• We introduce pFedKDH, a personalized FL method that avoids classifier-head averaging. The server aggregates only the backbone, while each client keeps a persistent clientspecific head that is not uploaded, averaged, or overwritten;

• We combine persistent client-specific heads with teacherguided local optimization. A recalibrated global head acts as a teacher during local training, transferring global knowledge without forcing the client classifier to become global;

• We empirically show that this design achieves high accuracy, low variability across repetitions, stable convergence, and competitive computational cost under heterogeneous data partitions.

## 2. RELATED WORK

Federated learning commonly combines local optimization with server aggregation to learn a shared model from decentralized data. FedAvg [1] averages client parameters after local training, while FedProx [8] adds a proximal term to limit deviations from the global model. Both methods still rely on a single shared solution, which can be restrictive under heterogeneous label distributions.

One direction for personalization is to maintain client-specific models while keeping them tied to a global reference. pFedMe [9] uses a bilevel formulation in which each client optimizes a personalized model regularized toward the global one. Ditto [10] instead trains global and personalized models in parallel, with a penalty that keeps the local solution close to the global model. These methods personalize the full model, so their behavior depends on how strongly the local model is coupled to the global reference.

A second direction separates the network into a feature extractor and a classifier head. The feature extractor maps the input into a representation, while the head maps this representation to output classes. FedPer [11] keeps the final layers local and aggregates the lower layers. FedRep [12] separates shared representation learning from local classifier training. FedBABU [13] updates and aggregates only the body of the network during federated training and adapts the classifier afterward. These methods show that classifier parameters are strongly affected by local label distributions and should not necessarily be averaged together with the shared representation. Fed-RoD [14] combines generic and personalized predictors, while Fed-LoGe [15] uses shared repre- sentations with client-specific classifiers under long-tailed data.

Another group of methods modifies the information exchanged across clients or the way client updates are combined. FedProto [16] exchanges one feature summary per class instead of full model parameters, reducing the dependence on full-model averaging. This mechanism can be weakened when local data are sparse or when some classes are absent from a client. pFedSim [17] aggregates information from clients estimated to be similar, which reduces the influence of unrelated updates but depends on the reliability of the similarity estimate. FedALA [18] learns how much of the downloaded global model should be combined with the previous local model before local training, giving each client more control over the use of global information. However, these approaches still depend on class summaries, similarity estimates, or parameter mixing.

Knowledge distillation offers another way to transfer information across heterogeneous clients. FedFed [19] uses feature distillation to mitigate data heterogeneity while keeping part of the information local. pFedKDH follows this direction while keeping shared representation learning separate from local classification. Only the backbone is aggregated, whereas each client keeps a private and persistent head. Global predictive information is transferred through a recalibrated teacher, preserving client-specific decision boundaries without head aggregation or replacement.

## 3. PROPOSED METHOD

This section presents our method. The pFedKDH separates shared representation learning from client-specific classification under label skew. It aggregates a global backbone while keeping persistent private heads, and uses a calibrated global teacher for local distillation. A proximal term limits backbone drift without exposing clientspecific classifiers. Figure 1 summarizes the proposed pFedKDH workflow.

![](images/7b6977660056548348d50f88e30658a57031e5b5c49941691884fd15a5350f3b.jpg)  
Fig. 1. Overview of the pFedKDH training cycle: the server broadcasts the shared backbone and teacher head, while each client performs local training with CE, KD, and proximal regularization. Only the updated backbone is uploaded to the server; private heads remain local, and the teacher head is recalibrated using server-side data.

## 3.1. Problem Setup and Model Decomposition

We consider a FL setting with K clients, where each client k owns a private dataset $\mathcal { D } _ { k } = \{ \bar { ( } x _ { i } ^ { k } , y _ { i } ^ { k } ) \} _ { i = 1 } ^ { n _ { k } }$ and $n _ { k }$ denotes the number of local training samples. Each client also maintains a private and persistent local head ϕ<sub>k</sub>, while the server maintains the global backbone $\theta _ { g }$ , the teacher head $\phi _ { g } ,$ , and the auxiliary labeled dataset D . The model is decomposed as $f ( x ; \theta , \phi ) = h _ { \phi } ( b _ { \theta } ( x ) )$ , where θ denotes the backbone parameters, ϕ denotes the classification-head parameters, $b _ { \theta } ( \cdot )$ maps the input to a feature representation, and $h _ { \phi } ( \cdot )$ maps this representation to class logits.

pFedKDH can be formulated as a personalized federated optimization problem in which a shared representation $\theta _ { g }$ is learned together with client-specific heads $\{ \phi _ { k } \} _ { k = 1 } ^ { K }$ . The corresponding global objective aggregates the local losses weighted by the relative number of samples at each client:

$$
\operatorname* { m i n } _ { \theta _ { g } , \{ \phi _ { k } \} _ { k = 1 } ^ { K } } \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { \sum _ { \ell = 1 } ^ { K } n _ { \ell } } \mathcal { L } _ { k } ( \theta _ { g } , \phi _ { k } ) .\tag{1}
$$

In practice, this objective is optimized in a federated manner: each client updates only its local copy of the backbone and its private head, while the server aggregates the backbone updates. This formulation encodes the main design choice of pFedKDH: the backbone captures transferable visual structure, while the local heads absorb client-specific label distributions.

At round t, the server maintains a global model $\boldsymbol { w } _ { g } ^ { t } = ( \boldsymbol { \theta } _ { g } ^ { t } , \boldsymbol { \phi } _ { g } ^ { t } )$ where $\theta _ { g } ^ { t }$ is the global backbone and $\boldsymbol { \dot { \phi } } _ { g } ^ { t }$ is the global head used by the teacher. Client k maintains a personalized model $w _ { k } ^ { t } = ( \theta _ { g } ^ { t } , \phi _ { k } ^ { t } )$ where $\phi _ { k } ^ { t }$ is private and persistent across rounds. The server coordinates representation learning through $\theta _ { g } ^ { t } ,$ , while each client preserves and updates its own persistent head $\phi _ { k } ^ { t }$ across rounds.

## 3.2. Local Personalized Training

At round t, the server selects clients $\boldsymbol { S } _ { t }$ according to the participation ratio $\rho$ and broadcasts the current global model $( \theta _ { g } ^ { t } , \bar { \phi } _ { g } ^ { t } )$ . Each selected client initializes its local backbone as $\theta _ { k } \gets \theta _ { g } ^ { t }$ and its classifier as the persistent local head $\phi _ { k } ^ { t }$

The global model acts as a teacher, while the personalized client model acts as a student. Following knowledge distillation [20], the predictive distribution associated with logits z is softened using temperature $T$ as $q _ { i } ^ { T } = \exp ( z _ { i } / T ) / \sum _ { i } \exp ( z _ { j } / T )$ , where $T > 1$ produces a smoother distribution that exposes relative information among classes. Let $q _ { g } ^ { T } ( x )$ and $q _ { k } ^ { T } ( x )$ denote the resulting teacher and student distributions, respectively. Client k minimizes

$$
\begin{array} { l } { \mathcal { T } _ { k } ^ { t } ( \theta _ { k } , \phi _ { k } ) = \displaystyle \frac { 1 - \lambda _ { \mathrm { K D } } } { n _ { k } } \displaystyle \sum _ { ( x , y ) \in \mathcal { D } _ { k } } \ell _ { \mathrm { C E } } ( h _ { \phi _ { k } } { ( b _ { \theta _ { k } } ( x ) ) } , y ) } \\ { \displaystyle + \frac { \lambda _ { \mathrm { K D } } T ^ { 2 } } { n _ { k } } \displaystyle \sum _ { ( x , y ) \in \mathcal { D } _ { k } } \mathrm { K L } ( q _ { g } ^ { T } ( x ) \| q _ { k } ^ { T } ( x ) ) } \\ { \displaystyle +  \frac { \mu } { 2 } \| \theta _ { k } - \theta _ { g } ^ { t } \| _ { 2 } ^ { 2 } . } \end{array}\tag{2}
$$

The three terms fit local labels, transfer global predictive knowledge, and constrain backbone drift under non-IID updates. Distillation guides the local model without uploading, averaging, or overwriting the client-specific head, preserving the personalized decision boundary. After local training, only $\theta _ { k } ^ { t + 1 }$ is sent to the server, while $\phi _ { k } ^ { t + 1 }$ remains private and persistent.

## 3.3. Server-Side Warm-up, Aggregation, and Teacher Recalibration

pFedKDH uses two server-side operations: global warm-up and head recalibration. We define global warm-up as the optional pretraining of the global model $\breve { w _ { g } ^ { 0 } } = ( \theta _ { g } ^ { 0 } , \phi _ { g } ^ { 0 } )$ before the first communication round. In this step, the server updates both the global backbone $\theta _ { g } ^ { 0 }$ and the global head $\phi _ { g } ^ { 0 }$ using the auxiliary server dataset $\mathcal { D } _ { s } \mathbf { : }$

$$
( \theta _ { g } ^ { 0 } , \phi _ { g } ^ { 0 } )  \arg \operatorname* { m i n } _ { \theta _ { g } , \phi _ { g } } \frac { 1 } { | \mathscr { D } _ { s } | } \sum _ { ( x , y ) \in \mathscr { D } _ { s } } \ell _ { \mathrm { C E } } ( h _ { \phi _ { g } } ( b _ { \theta _ { g } } ( x ) ) , y ) .\tag{3}
$$

This global warm-up is distinct from head recalibration: warmup updates both $\theta _ { g } ^ { 0 }$ and $\mathbf { \hat { \phi } } _ { \mathcal { G } } ^ { 0 }$ before federated training starts, whereas

head recalibration updates only the global head while keeping the current global backbone fixed.

We denote by $E _ { w }$ the number of global warm-up epochs. When used, the global warm-up initializes the teacher before distillation.

After local training, the server aggregates only the client backbones:

$$
\theta _ { g } ^ { t + 1 } = \sum _ { k \in \mathcal { S } _ { t } } \frac { n _ { k } } { \sum _ { \ell \in \mathcal { S } _ { t } } n _ { \ell } } \theta _ { k } ^ { t + 1 } .\tag{4}
$$

No client head is transmitted, averaged, or reset. This is the core personalization mechanism of pFedKDH: the representation is collaborative, but the decision boundary remains local.

Global warm-up starts before the first communication round, only when $E _ { w } > 0 .$ . In this stage, the server updates the initial global backbone $\theta _ { g } ^ { 0 }$ and global head ${ \bf \bar { \boldsymbol { \phi } } } _ { g } ^ { 0 }$ on the auxiliary dataset $\mathcal { D } _ { s }$ , before any client receives the global model.

Head recalibration starts after each server-side backbone aggregation step. Because the aggregated backbone $\theta _ { g } ^ { t + 1 }$ changes at the end of round t, the previous global head may no longer be well aligned with the updated representation. The server therefore recalibrates only the global head, keeping $\theta _ { g } ^ { t + 1 }$ fixed:

$$
\phi _ { g } ^ { t + 1 }  \arg \operatorname* { m i n } _ { \phi } \frac { 1 } { \vert \mathcal { D } _ { s } \vert } \sum _ { ( x , y ) \in \mathcal { D } _ { s } } \ell _ { \mathrm { C E } } ( h _ { \phi } ( b _ { \theta _ { g } ^ { t + 1 } } ( x ) ) , y ) .\tag{5}
$$

Thus, warm-up and recalibration occur at different moments and serve different purposes. Warm-up is an optional pre-federated stage that initializes the full global model before round 0, whereas recalibration is a repeated post-aggregation stage that realigns only the teacher head after each communication round. The auxiliary dataset $\mathcal { D } _ { s }$ is used only for teacher initialization and calibration, and never for testing.

Algorithm 1 summarizes the implementation of our method.

Algorithm 1 pFedKDH   
Require: Number of clients K, rounds R, local epochs $E ,$ partici  
pation ratio $\rho ,$ distillation weight $\lambda _ { \mathrm { K D } } .$ , temperature $T ,$ , proximal   
weight $\mu ,$ auxiliary dataset $\mathcal { D } _ { s } ,$ warm-up epochs $E _ { w }$   
1: Initialize global backbone $\theta _ { g } ^ { 0 } ,$ , global head $\hat { \phi } _ { g } ^ { 0 } ,$ and persistent lo  
cal heads $\{ \phi _ { k } ^ { 0 } \} _ { k = 1 } ^ { K }$   
2: if $E _ { w } > 0$ then   
3: Warm up $( \theta _ { g } ^ { 0 } , \phi _ { g } ^ { 0 } )$ on $\mathcal { D } _ { s }$ using (3)   
4: end if   
5: Calibrate $\phi _ { g } ^ { 0 }$ on $\mathcal { D } _ { s }$ with $\theta _ { g } ^ { 0 }$ fixed   
6: for $t = 0 , \overset { \vartriangle } { \boldsymbol { 1 } } , \ldots , R - 1 \ \mathbf { d }$ o   
7: Select clients $S _ { t } \subseteq \{ 1 , \ldots , K \}$ according to $\rho$   
8: Broadcast $( \theta _ { g } ^ { t } , \phi _ { g } ^ { t } )$ to each client $k \in S _ { t }$   
9: for each client $k \in S _ { t }$ in parallel do   
10: Initialize $\theta _ { k }  \theta _ { g } ^ { t }$ and $\phi _ { k }  \phi _ { k } ^ { t }$   
11: Update $( \theta _ { k } , \phi _ { k } )$ by minimizing (2) for E local epochs   
12: Send $\theta _ { k } ^ { t + 1 }$ to the server   
13: Keep $\stackrel { \kappa } { \phi _ { k } ^ { t + 1 } }$ private and persistent   
14: end for   
15: Aggregate $\theta _ { a } ^ { t + 1 }$ using (4)   
16: Recalibrate $\stackrel { \triangledown } { \phi } _ { g } ^ { t + 1 }$ on $\mathcal { D } _ { s }$ with $\theta _ { g } ^ { t + 1 }$ fixed using (5)   
17: end for

## 3.4. Personalized Evaluation and Variants

At evaluation time, the global model is $( \theta _ { g } ^ { t } , \phi _ { g } ^ { t } )$ , while the personalized model for client k is $( \theta _ { g } ^ { t } , \phi _ { k } ^ { t } )$ . We report two variants:

1. pFedKDH-Fair: evaluates the persistent local head directly, without any extra local update after training. This variant isolates the effect of backbone-only aggregation, persistent heads, distillation, and recalibration under the same training budget used by the baselines.

2. pFedKDH-EA: performs evaluation-time head adaptation through a small number of additional local head-only update steps on the client’s training data. Starting from the same global backbone and persistent local head, only $\phi _ { k } ^ { t }$ is updated, while $\theta _ { g } ^ { t }$ remains frozen. No test sample is used during this stage. This variant measures how much additional personalization can be obtained through a lightweight refinement of the classifier head.

## 4. EXPERIMENTS AND RESULTS

## 4.1. Experimental Protocol

All methods were evaluated on four image classification datasets, namely, MNIST, Fashion-MNIST (FMNIST), CIFAR10, and CI-FAR100. These benchmarks represent increasingly difficult visual classification scenarios, ranging from grayscale digit recognition to natural image classification with a large number of classes.

Statistical heterogeneity was simulated using class-wise Dirichlet partitions with $\alpha \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 5 0 \}$ , where smaller values indicate stronger label skew across clients.

The comparison includes classical federated baselines and representative personalized FL methods. Specifically, we compare against FedAvg, FedProx, Ditto, pFedMe, FedPer, FedRep, FedBABU, Fed-Proto, FedALA, FedFed, and pFedSim.

To ensure a fair comparison, all methods were evaluated with the same lightweight CNN architecture whenever applicable. The backbone contains two convolutional layers with ReLU activations and max-pooling, followed by dropout, flattening, and a fully connected projection layer that produces the feature representation used by a linear classification head. Therefore, the observed differences are primarily attributed to the federated optimization and personalization strategies rather than to variations in model capacity.

All methods were trained under the same optimization budget whenever applicable, including the same number of communication rounds, client participation ratio, local epochs, batch size, and learning rate. The main experimental settings are summarized in Table 1. In all main experiments reported for both pFedKDH-Fair and pFedKDH-EA, we set $E _ { w } = 0 ;$ ; therefore, the global warm-up stage in (3) is disabled and $D _ { s }$ is not used to pretrain the global backbone.

For the experimental comparison, pFedKDH follows the training procedure defined in Section 3. Shared-model baselines are evaluated as global models, whereas personalized methods are evaluated at the client level using their personalized models. For pFedKDH, this corresponds to evaluating the global backbone with each client’s persistent head. The detailed metric definitions and result analysis are presented in Section 4.3.

## 4.2. Convergence Behavior

Figure 2 analyzes the convergence behavior under the most severe label-skew setting considered in this work. On CIFAR10, pFedKDH-EA presents the strongest convergence profile, reaching the highest final accuracy while keeping a narrow standard deviation band. The base version, pFedKDH-Fair, also remains competitive throughout training, showing that the core mechanism of backboneonly aggregation with persistent client-specific heads already provides a stable personalized solution. The additional evaluation-time head adaptation further improves the final client-level performance. Regarding Fashion-MNIST, pFedKDH-EA again converges to the best performance region with low variability across repetitions. In contrast, global baselines such as FedAvg and FedProx converge to substantially lower plateaus, which reinforces the limitation of a single shared classifier under strong statistical heterogeneity. Overall, the convergence curves show that pFedKDH combines high final accuracy with stable training behavior, supporting the robustness of persistent heads and teacher-guided local optimization under severe label skew.

Table 1. Experimental setup used for pFedKDH and baseline comparisons.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Datasets</td><td>MNIST, Fashion-MNIST, CIFAR10, CIFAR100</td></tr><tr><td>Data partitioning</td><td>Class-wise Dirichlet</td></tr><tr><td>Dirichlet α</td><td>{0.05, 0.10, 0.50}</td></tr><tr><td>Number of clients</td><td>20</td></tr><tr><td>Client participation</td><td>50%</td></tr><tr><td>Communication rounds</td><td>50, 50, 50, 70</td></tr><tr><td>Local epochs</td><td>2,2,3,5</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>0.015</td></tr><tr><td>Model architecture</td><td>Simple CNN + linear head</td></tr><tr><td>Aggregated parameters</td><td>Backbone only</td></tr><tr><td>Client-specific heads</td><td>Persistent</td></tr><tr><td>Auxiliary set size |Ds|</td><td>5000</td></tr><tr><td>Global warm-up</td><td> $E _ { w } = 0$ </td></tr><tr><td>Head recalibration</td><td>3 epochs</td></tr></table>

## 4.3. Accuracy under Statistical Heterogeneity

We report two main evaluation metrics. Global test accuracy measures the performance of a single shared model after server aggregation and is used for shared-model baselines such as FedAvg and FedProx. Personalized macro accuracy measures client-level personalization. It first computes the test accuracy of each client using its own personalized model and then averages these accuracies with equal weight across clients. This metric gives the same importance to all clients and is therefore more suitable for assessing personalized methods under heterogeneous label distributions. Values are reported as mean±standard deviation, and the best result in each setting is shown in bold.

Table 2 reports the final accuracy comparison on MNIST and Fashion-MNIST, while Table 3 reports the corresponding results on CIFAR10 and CIFAR100. Together, these tables summarize the final performance under different levels of statistical heterogeneity. pFed-KDH obtains the highest mean accuracy in the most heterogeneous settings. For $\alpha = 0 . 0 5$ , it achieves the best mean accuracy on all four datasets. For $\alpha = 0 . 1 0 $ , it remains the best method on MNIST, Fashion-MNIST, and CIFAR10, and is only 0.35 % below FedALA on CIFAR100, while presenting a much lower standard deviation. The low standard deviations indicate that these gains are consistent across repetitions.

The base version, pFedKDH-Fair, also shows strong behavior without evaluation-time head adaptation. It is the second-best method in all MNIST settings, in Fashion-MNIST for $\alpha = 0 . 0 5$ and $\alpha = 0 . 1 0$ , and in CIFAR10 for $\alpha = 0 . 0 5$ . Its standard deviation is also consistently small when compared with several personalized baselines. This confirms that the core mechanism of pFedKDH, namely backbone-only aggregation with persistent client-specific heads, already provides a stable personalization strategy.

The gains differ across baseline families. Compared with global baselines, pFedKDH improves the best FedAvg/FedProx result by $6 . 0 5 ~ \%$ on MNIST, 24.47 on Fashion-MNIST, 37.66 on CIFAR10, and 32.71 on CIFAR100 at $\alpha \ = \ 0 . 0 5$ . Compared with regularization-based personalized methods such as Ditto and pFedMe, the gain reaches 8.62 % on CIFAR100 with $\alpha = 0 . 0 5$ Against backbone-head separation methods such as FedPer, FedRep, and FedBABU, pFedKDH still improves the best competing result by up to 1.82 % on CIFAR10 with $\alpha \ : = \ : 0 . 0 5$ The comparison indicates that head separation helps, while distillation adds further gains.

![](images/c17d004cccab1d1ea2fd780ed9aa8c17de47bb348d0bdbf55a9bf444b77f63f7.jpg)  
(a) CIFAR10, α = 0.05

![](images/a74af93dd27bc6b60003a2ef9b5eee0ab416fc79054fd67a73ff866dbb086d31.jpg)  
(b) FMNIST, α = 0.05  
Fig. 2. Accuracy curves over communication rounds for CIFAR10 and FMNIST under the strongest statistical heterogeneity setting considered in this work $( \alpha = 0 . 0 5 )$ . Shaded regions indicate the standard deviation across repetitions.

Table 2. Final accuracy comparison on MNIST and Fashion-MNIST.
<table><tr><td rowspan="2">Method</td><td colspan="3">MNIST</td><td colspan="3">Fashion-MNIST</td></tr><tr><td> $\alpha = 0 . 0 5$ </td><td> $\alpha = 0 . 1 0$ </td><td> $\alpha = 0 . 5 0$ </td><td> $\alpha = 0 . 0 5$ </td><td> $\alpha = 0 . 1 0$ </td><td> $\alpha = 0 . 5 0$ </td></tr><tr><td>FedAvg [1]</td><td>93.23±2.11</td><td>94.43±2.01</td><td>97.89±0.03</td><td>73.21±9.79</td><td>80.96±6.55</td><td>86.53±1.57</td></tr><tr><td>FedProx [8]</td><td>93.22±2.08</td><td>94.46±1.96</td><td>97.88±0.03</td><td>73.23±9.77</td><td>80.99±6.45</td><td>86.49±1.55</td></tr><tr><td>Ditto [10]</td><td>97.47±0.67</td><td>96.02±1.45</td><td>94.04±1.35</td><td>94.42±0.22</td><td>91.67±2.73</td><td>88.25±0.36</td></tr><tr><td>FedALA [18]</td><td>99.18±0.26</td><td>99.05±0.25</td><td>95.31±5.03</td><td>97.01±0.40</td><td>94.96±2.01</td><td>93.25±0.53</td></tr><tr><td>FedProto [16]</td><td>99.00±0.38</td><td>97.98±0.75</td><td>96.38±0.73</td><td>96.76±1.21</td><td>95.25±0.90</td><td>91.47±0.35</td></tr><tr><td>FedPer [11]</td><td>99.11±0.36</td><td>98.61±0.13</td><td>98.08±0.30</td><td>97.10±0.46</td><td>94.95±2.13</td><td>92.55±0.35</td></tr><tr><td>FedRep [12]</td><td>98.63±0.53</td><td>96.69±1.89</td><td>96.52±1.59</td><td>94.55±3.38</td><td>94.14±1.69</td><td>91.68±0.17</td></tr><tr><td>FedBABU [13]</td><td>99.10±0.42</td><td>98.49±0.25</td><td>97.73±0.13</td><td>96.78±0.29</td><td>95.18±1.72</td><td>92.79±0.25</td></tr><tr><td>FedFed [19]</td><td>98.19±0.28</td><td>97.59±0.74</td><td>97.63±0.14</td><td>84.06±1.96</td><td>85.55±2.68</td><td>87.54±1.11</td></tr><tr><td>pFedMe [9]</td><td>99.20±0.23</td><td>98.76±0.23</td><td>97.58±0.60</td><td>96.77±0.57</td><td>94.75±1.66</td><td>92.05±1.01</td></tr><tr><td>pFedKDH-Fair</td><td>99.22±0.17</td><td>99.12±0.09</td><td>98.96±0.09</td><td>97.31±0.11</td><td>95.96±0.27</td><td>92.13±0.28</td></tr><tr><td>pFedKDH-EA</td><td>99.28±0.03</td><td>99.23±0.03</td><td>99.01±0.09</td><td>97.70±0.20</td><td>96.80±0.09</td><td>92.68±0.30</td></tr></table>

Table 3. Final accuracy comparison on CIFAR10 and CIFAR100.
<table><tr><td>Method</td><td> $\alpha = 0 . 0 5$ </td><td>CIFAR10  $\alpha = 0 . 1 0$ </td><td> $\alpha = 0 . 5 0$ </td><td> $\alpha = 0 . 0 5$ </td><td>CIFAR100  $\alpha = 0 . 1 0$ </td><td> $\alpha = 0 . 5 0$ </td></tr><tr><td>FedAvg [1]</td><td>54.67±8.16</td><td>61.46±5.94</td><td>67.52±1.37</td><td>35.74±0.76</td><td>38.32±0.87</td><td>43.76±0.15</td></tr><tr><td>FedProx [8]</td><td>54.66±8.27</td><td>61.44±5.86</td><td> $6 7 . 5 1 { \pm } 1 . 3 6 $ </td><td>35.64±0.76</td><td>38.25±0.90</td><td>43.70±0.05</td></tr><tr><td>Ditto [10]</td><td>86.05±4.07</td><td>82.32±3.19</td><td>68.82±2.93</td><td>57.02±0.54</td><td>50.90±3.28</td><td>40.18±1.40</td></tr><tr><td>FedALA [18]</td><td>88.63±4.06</td><td>87.92±2.39</td><td>74.69±4.45</td><td>67.53±2.05</td><td>60.86±3.75</td><td>49.64±1.36</td></tr><tr><td>FedProto [16]</td><td>88.35±3.22</td><td>84.16±3.53</td><td>71.02±1.24</td><td>63.80±1.55</td><td>56.43±1.16</td><td>35.98±0.99</td></tr><tr><td>FedPer [11]</td><td>90.51±4.45</td><td>87.24±4.03</td><td>75.46±2.13</td><td>67.51±2.11</td><td>60.04±1.60</td><td>42.36±0.52</td></tr><tr><td>FedRep [12]</td><td>89.21±5.29</td><td>86.49±3.13</td><td>70.36±2.64</td><td>60.72±1.34</td><td>54.03±1.61</td><td>36.60±0.55</td></tr><tr><td>FedBABU [13]</td><td>88.61±3.36</td><td>87.08±2.23</td><td>76.26±1.05</td><td>63.36±1.34</td><td>57.31±2.82</td><td>44.90±0.63</td></tr><tr><td>FedFed [19]</td><td>65.67±2.34</td><td>66.04±3.81</td><td>62.56±0.32</td><td>36.44±0.49</td><td>35.10±1.34</td><td>34.93±0.07</td></tr><tr><td>pFedMe [9]</td><td>89.63±4.25</td><td>86.89±2.81</td><td>72.28±1.96</td><td>59.83±0.62</td><td>54.05±3.58</td><td></td></tr><tr><td>pFedKDH-Fair</td><td>91.06±0.63</td><td>85.81±0.15</td><td>74.32±0.33</td><td>67.39±0.27</td><td></td><td></td></tr><tr><td>pFedKDH-EA</td><td>92.33±0.14</td><td>88.37±0.15</td><td>76.43±0.15</td><td>68.45±0.47</td><td>60.22±0.38 60.51±0.30</td><td>41.39±0.50 41.96±0.85</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

The cases where pFedKDH is not the best are concentrated in

Fashion-MNIST with $\alpha ~ = ~ 0 . 5 0$ and CIFAR100 with $\alpha ~ = ~ 0 . 1 0$ and $\alpha = 0 . 5 0$ . In the first two cases, the mean gap is small and pFedKDH keeps lower variability. The larger gap on CIFAR100 with $\alpha = 0 . 5 0$ suggests that many-class settings may require more class-aware teacher calibration or more adaptive knowledge transfer, since a single recalibrated teacher may become less informative when the label space is large and client label support is fragmented. The pFedMe result for CIFAR100 with $\alpha = 0 . 5 0$ is omitted because the run did not complete under the same computational budget.

All experiments use a fixed auxiliary set of $| D _ { s } | = 5 0 0 0$ samples. The sensitivity to $| D _ { s } |$ , class imbalance within the auxiliary set, and domain shift between $D _ { s }$ and the client distributions remains to be investigated. These factors may affect the quality of teacher recalibration, particularly in many-class and highly heterogeneous settings.

## 4.4. Component-wise Diagnostic Analysis

We use a diagnostic ablation on CIFAR10 with $\alpha = 0 . 0 5$ to inspect the contribution of the main pFedKDH components. This experiment uses a single Monte Carlo repetition and is intended only as an explanatory analysis.

Figure 3 shows that persistent local heads provide the main personalization gain, while knowledge distillation further improves local optimization. The proximal term has a small effect in this run. Recurrent recalibration mainly acts as a teacher-alignment mechanism, and its quantitative impact may vary with the dataset, class imbalance, and degree of label skew.

## 4.5. Computational Cost

Let $P _ { b }$ and $P _ { h }$ denote the numbers of backbone and head parameters, respectively, and let $\lvert S _ { t } \rvert$ be the number of clients selected at round t. As only the backbone is uploaded, the client-to-server communication cost per round is $\mathcal { O } ( | S _ { t } | P _ { b } )$ , instead of $\mathcal { O } ( | S _ { t } | ( P _ { b } + P _ { h } ) )$ for full-model aggregation. The server aggregation in (4) also requires $\mathcal { O } ( | S _ { t } | P _ { b } )$ operations. Teacher recalibration introduces server-side optimization over $D _ { s } ,$ but updates only the head while the backbone remains fixed. Thus, pFedKDH preserves the same backboneorder communication complexity as backbone-sharing PFL methods, with additional computation arising mainly from local distillation and teacher-head recalibration.

![](images/6a117e69321a32eae203932050b14ebfbdc41d8fad689880f253f77a241d3127.jpg)

B  
![](images/defeb697c6780e9f327c75b34e24d60a9e6ea8592adf288186420bc6b3c2c6be.jpg)  
Fig. 3. Component-wise diagnostic ablation of pFedKDH on CI-FAR10 under $\alpha = 0 . 0 5$ . Panel A shows pFedKDH-Fair, while Panel B shows pFedKDH-EA.

pFedKDH introduces a moderate computational overhead. pFedKDH-Fair and pFedKDH-EA require 6.029 s and 6.466 s per round, corresponding to 1.57× and 1.68× the runtime of the fastest method, FedPer. This cost is comparable to FedALA and FedBABU, and substantially lower than FedRep and pFedMe.

## 5. CONCLUSIONS

Statistical heterogeneity limits federated learning because a single shared classifier may not capture client-specific label distributions. pFedKDH addresses this by aggregating only the shared backbone, keeping persistent client-specific heads, and using a recalibrated global head as a teacher during local training.

Across the evaluated settings, pFedKDH achieves the best mean accuracy in most cases and remains competitive otherwise, while consistently showing low variability. Convergence and ablation results indicate that persistent heads preserve client-specific decision boundaries, while distillation improves global knowledge transfer under label skew.

Future work will investigate adaptive distillation, class-aware teacher calibration, sensitivity to the size and class distribution of $D _ { s } ,$ , robustness to domain shift, and protocols that reduce or eliminate dependence on auxiliary server-side data.

## 6. REFERENCES

[1] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas, “Communication-

efficient learning of deep networks from decentralized data,” in Proc. AISTATS, 2017, pp. 1273–1282.

[2] Peter Kairouz et al., “Advances and open problems in federated learning,” Foundations and Trends in Machine Learning, vol. 14, no. 1–2, pp. 1–210, 2021.

[3] Z. Lu, H. Pan, Y. Dai, X. Si, and Y. Zhang, “Federated learning with non-IID data: A survey,” IEEE Internet ofThings Journal, vol. 11, no. 11, pp. 19188–19209, 2024.

[4] W.-C. Chung, C.-A. Lo, Y.-H. Lin, Z.-H. Chen, and C.-L. Hung, “Decentralized federated learning with non-IID data: Challenges, trends, and future opportunities,” ACM Computing Surveys, vol. 58, no. 8, pp. Article 192, 2026.

[5] M. Ye, X. Fang, B. Du, P. C. Yuen, and D. Tao, “Heterogeneous federated learning: State-of-the-art and research challenges,” ACM Computing Surveys, vol. 56, no. 3, pp. Article 79, 2023.

[6] A. Z. Tan, H. Yu, L. Cui, and Q. Yang, “Towards personalized federated learning,” IEEE Transactions on Neural Networks and Learning Systems, 2022.

[7] F. Sabah, Y. Chen, Z. Yang, A. Raheem, M. Azam, and R. Sarwar, “Model optimization techniques in personalized federated learning: A survey,” Expert Systems with Applications, 2023.

[8] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in Proc. MLSys, 2020.

[9] C. T. Dinh, N. H. Tran, and T. D. Nguyen, “Personalized federated learning with Moreau envelopes,” in Advances in Neural Information Processing Systems, 2020, pp. 21394–21405.

[10] T. Li, S. Hu, A. Beirami, and V. Smith, “Ditto: Fair and robust federated learning through personalization,” in Proc. ICML, 2021, pp. 6357–6368.

[11] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, “Federated learning with personalization layers,” arXiv preprint arXiv:1912.00818, 2019.

[12] L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai, “Exploiting shared representations for personalized federated learning,” in Proc. ICML, 2021, pp. 2089–2099.

[13] J. Oh, S. Kim, and S.-Y. Yun, “FedBABU: Toward enhanced representation for federated image classification,” in Proc. ICLR, 2022.

[14] H.-Y. Chen and W.-L. Chao, “On bridging generic and personalized federated learning for image classification,” in Proc. International Conference on Learning Representations (ICLR), 2022.

[15] Z. Xiao, Z. Chen, L. Liu, Y. Feng, J. Wu, W. Liu, J. T. Zhou, H. H. Yang, and Z. Liu, “FedLoGe: Joint local and generic federated learning under long-tailed data,” in Proc. International Conference on Learning Representations (ICLR), 2024.

[16] Y. Tan, G. Long, L. Liu, T. Zhou, Q. Lu, J. Jiang, and C. Zhang, “FedProto: Federated prototype learning across heterogeneous clients,” in Proc. AAAI, 2022, pp. 8432–8440.

[17] J. Tan, Y. Zhou, G. Liu, J. H. Wang, and S. Yu, “pFedSim: Similarity-aware model aggregation towards personalized federated learning,” arXiv preprint arXiv:2305.15706, 2023.

[18] J. Zhang, Y. Hua, H. Wang, T. Song, Z. Xue, R. Ma, and H. Guan, “FedALA: Adaptive local aggregation for personalized federated learning,” in Proc. AAAI, 2023, pp. 11237– 11244.

[19] Z. Yang, Y. Zhang, Y. Zheng, X. Tian, H. Peng, T. Liu, and B. Han, “FedFed: Feature distillation against data heterogeneity in federated learning,” in Advances in Neural Information Processing Systems, 2023, vol. 36, pp. 60397–60428.

[20] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.