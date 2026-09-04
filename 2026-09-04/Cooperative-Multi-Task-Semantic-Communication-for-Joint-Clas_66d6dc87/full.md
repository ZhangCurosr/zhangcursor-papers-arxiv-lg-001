# Cooperative Multi-Task Semantic Communication for Joint Classification and Regression Tasks

Ahmad Halimi Razlighi , Mohammad Siddiqur Rahman<sup>†</sup> , Maximilian H. V. Tillmann ,

Edgar Beck , and Armin Dekorsy

Department of Communications Engineering, University of Bremen, Germany

E-mails:{halimi, tillmann, beck, dekorsy}@ant.uni-bremen.de, <sup>†</sup>E-mail: mrahman@uni-bremen.de

Abstract—Multi-Task semantic communication (SemCom) prioritizes simultaneous execution of multiple tasks over bit-accurate reconstruction in future intelligent networks. In our prior work [1], we introduced the cooperative multi-task SemCom (CMT-SemCom) framework, in which the semantic encoder is divided into a common unit (CU) and multiple specific units (SUs) to facilitate cooperative multi-task processing. However, CMT-SemCom has been evaluated on homogeneous classification tasks on simplistic datasets, limiting its applicability to real-world perception systems. In this paper, we extend our CMT-SemCom to jointly handle heterogeneous classification and regression tasks on the complex Cityscapes dataset. We adopt the information maximization (InfoMax) principle so that it accommodates mixed discrete and continuous semantic variables. In particular, we benchmark the proposed framework against independent singletask training, a conventional task-agnostic digital transmission, and single-encoder multi-decoder SemCom. Additionally, we investigate the impact of CU capacity on joint task performance, providing design insights. Extensive evaluations demonstrate that CMT-SemCom significantly outperforms the benchmarks.

Index Terms—Task-oriented communication, multi-task learning, classification, regression, Cityscapes dataset.

## I. INTRODUCTION

The evolution toward intelligent wireless networks is fundamentally reshaping communication paradigms, shifting the design objective from reliable bit transmission to task-oriented information exchange [2]. In this context, semantic communication (SemCom) has emerged as a promising paradigm that prioritizes the transmission of task-relevant semantic information rather than bit-level accuracy [3].

Many practical intelligent systems are required to perform multiple inference tasks simultaneously, like autonomous driving, where semantic segmentation, depth estimation, and scene understanding are jointly executed [4]. Such scenarios motivate multi-task SemCom, in which a single semantic representation is shared among several downstream tasks [5]. Multitask learning (MTL) exploits common latent representations across related tasks to improve generalization while reducing computational redundancy [6]. Motivated by these advantages, recent studies have incorporated MTL into SemCom using deep neural network (DNN) architectures, including a singleencoder on the transmitter (Tx) side and multiple-decoder on the receiver (Rx) side [7], [8]. Nevertheless, these approaches primarily treat the SemCom system as a black-box neural network and rely on heuristic network designs without an explicit information-theoretic objective.

To address this limitation, our previous work introduced the cooperative multi-task SemCom (CMT-SemCom) framework, which establishes a mutual information maximization (Info-Max) foundation for cooperative semantic transmission [1]. Specifically, we modeled a semantic source capable of yielding multiple semantic variables from a shared observation and proposed a split semantic encoder consisting of a common unit (CU) and multiple specific units (SUs) on the Tx. By optimizing an end-to-end (E2E) InfoMax objective through variational approximation, the CU learns semantic features shared across tasks, whereas the SUs preserve task-specific information and channel coding for reliable wireless transmission. Experimental evaluations on homogeneous image classification tasks using the MNIST and CIFAR-10 datasets demonstrated that exploiting statistical dependencies through the CU significantly improves task performance compared with independent task processing [9], [10].

Despite these encouraging results, the applicability of the existing CMT-SemCom framework remains limited. Previous evaluations considered only homogeneous image classification tasks on relatively simple datasets, whereas practical intelligent systems commonly involve heterogeneous tasks with fundamentally different statistical characteristics. In particular, classification and regression tasks frequently coexist in perception applications. For example, autonomous driving systems simultaneously perform semantic segmentation, which predicts discrete semantic labels, and depth estimation, which infers continuous geometric information from the same visual scene [4]. Since these tasks originate from a shared observation while exhibiting complementary semantic dependencies, they constitute a natural application scenario for the CMT-SemCom.

Joint learning of classification and regression has been extensively investigated in the computer vision and MTL literature from a machine learning (ML) perspective [11]– [13]. More recently, heterogeneous tasks have also been considered in SemCom through learning-based feature sharing mechanisms [14]. However, existing approaches optimize task-specific learning losses directly and therefore lack an information-theoretic formulation that explicitly models the underlying semantic variables and their probabilistic relationships. Consequently, it remains unclear how cooperative semantic representations should be learned when discrete and continuous semantic variables coexist within a unified communication framework. Addressing this gap is essential for evaluating the scalability and practical applicability of CMT-SemCom under realistic multi-task workloads and complex real-world datasets.

In this paper, we extend the CMT-SemCom framework to support the joint extraction of task-relevant information for heterogeneous classification and regression tasks from an information-theoretic perspective by modeling discrete and continuous semantic variables. The proposed framework is evaluated on the real-world Cityscapes dataset [15], representing realistic urban perception scenarios. The main contributions of this paper are summarized as follows:

• Generalizing the CMT-SemCom framework to accommodate both discrete and continuous semantic variables, executing simultaneous classification and regression tasks through a unified framework.

• Benchmarking the proposed framework against singletask training, a conventional task-agnostic digital transmission pipeline, and single-encoder multi-decoder (SEMD-SemCom).

• Investigating the impact of CU capacity on multi-task performance, revealing a trade-off between shared representation richness and task-specific interference, and providing design guidelines for optimizing CU dimensionality.

## II. PROBLEM FORMULATION

In this section, we establish a probabilistic model to characterize the semantic source and communication aspects. Based on that, we define the overall CMT-SemCom optimization problem for joint classification and regression tasks.

## A. System Model

We consider a multi-task communication system, consisting of a single Tx, multiple Rxs, and a wireless channel in between. The Tx makes an observation $S ~ \in ~ \mathbb { R } ^ { H \times W \times C }$ where H, W, and $C$ represent the height, width, and color channels of an input image, respectively. Then it extracts multiple task-specific information via the split structure, each corresponding to one of $N$ downstream tasks. At Rx i, this task-specific information is used to decode a reconstruction of semantic variable $\boldsymbol { Z } _ { i }$ to support execution of its task. We denote the tuple $( Z , S )$ as semantic source [1], fully described by its probability density function (pdf) p(z, s), where $\begin{array} { r } { Z ~ = ~ \left[ Z _ { 1 } Z _ { 2 } \cdot \cdot \cdot Z _ { N } \right] } \end{array}$ Each semantic variable $\boldsymbol { Z } _ { i }$ has the alphabet $\mathcal { Z } _ { i }$ , and as we consider pixel-level semantic perception tasks, semantic variables have the dimensionality of $\boldsymbol { Z _ { i } ^ { \star } } \in \mathcal { Z } _ { i } ^ { H \times W }$ . The alphabet $\mathcal { Z } _ { i }$ is either continuous with $\mathcal { Z } _ { i } = \mathbb { R }$ , or discrete with $\mathcal { Z } _ { i } = \{ { ^ { \cdot } \mathrm { c l a s s } _ { 1 } } ^ { \prime } , . . . , { ^ { \cdot } \mathrm { c l a s s } _ { F } } ^ { \prime } \}$

We consider the split encoder structure, consisting of a CU and N SUs. Let $( z , s )$ be a semantic source sample, where only s is available to the Tx. Then, the CU preprocesses this observation s for a subset of semantically related tasks, and the i-th SU takes the CU’s output c and computes a length-d codeword $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ . This codeword is transmitted over an AWGN channel to Rx i. The received signal ${ \pmb y } _ { i } = { \pmb x } _ { i } + { \pmb n } _ { i }$ with $\mathbf { \delta } n _ { i } \sim $ ${ \cal N } ( \mathbf { 0 } _ { d } , \gamma _ { n } ^ { 2 } I _ { d } )$ is used by decoder i to infer $\hat { z } _ { i }$ , with the goal that $\hat { z } _ { i } = z _ { i }$ . Thus, we have the joint pdf for the i-th semantic link as:

![](images/2a4452bfa69e1f1ce100f977338a47c47b98fe27305731f286c7c8e946b529b0.jpg)  
Fig. 1: CMT-SemCom system model.

$$
\begin{array} { r l } & { p ( z _ { i } , s , c _ { k } , \pmb { x } _ { i } , \pmb { y } _ { i } , \hat { z } _ { i } ) = } \\ & { p ( \pmb { s } , z _ { i } ) p ^ { \mathrm { c u } } ( \pmb { c } | s ) p ^ { \mathrm { s u } } ( \pmb { x } _ { i } | \pmb { c } ) p ( \pmb { y } _ { i } | \pmb { x } _ { i } ) p ( \hat { z } _ { i } | \pmb { y } _ { i } ) . } \end{array}\tag{1}
$$

## B. Problem Statement

Following the InfoMax principle as in [1], we maximize the mutual information between channel output $\mathbf { Y } _ { i }$ and corresponding semantic variable $\boldsymbol { Z } _ { i }$ to design encoders and decoders.

$$
\left[ p ^ { \star } ( c | s ) , p ^ { \star } ( \pmb { x } _ { i } | c ) \right] \ = \ \underset { p ( c | s ) , p ( \pmb { x } _ { i } | c ) } { \arg \operatorname* { m a x } } \sum _ { i = 1 } ^ { N } I ( \pmb { Y } _ { i } ; \pmb { Z } _ { i } ) .\tag{2}
$$

The optimization variables $p ( c | s )$ and $p ( \pmb { x } _ { i } | \pmb { c } )$ represent the distributions for the CU and SUs, respectively. For simplicity in notation, we represent $p ( c | s )$ and $p ( \pmb { x } _ { i } | \pmb { c } )$ by $p ^ { \mathrm { c u } }$ and $p _ { i } ^ { \mathrm { s u } }$ respectively.

## III. CMT-SEMCOM FOR JOINT CLASSIFICATION ANDREGRESSION TASKS

Expanding the mutual information in our optimization problem, as discussed in detail in [1], the approximated objective function is derived as

$$
\begin{array} { r l } & { \mathcal { L } ( \boldsymbol { \theta } , \boldsymbol { \Phi } , \boldsymbol { \Psi } ) = \displaystyle \sum _ { i = 1 } ^ { N } I ( \boldsymbol { Y } _ { i } ; \boldsymbol { Z } _ { i } ) } \\ & { \approx \mathbb { E } _ { p _ { \boldsymbol { \theta } } ^ { \mathrm { c u } } } \Bigg [ \sum _ { i = 1 } ^ { N } \Bigg \{ \underbrace { \mathbb { E } _ { p ( \boldsymbol { s } , \boldsymbol { z } _ { i } ) } \Bigg [ \mathbb { E } _ { p _ { \phi _ { i } } ^ { \mathrm { s u } } } [ \log q _ { \psi _ { i } } ( \boldsymbol { z } _ { i } | \boldsymbol { y } _ { i } ) ] } _ { \mathcal { L } _ { i } ( \phi _ { i } , \psi _ { i } ) } \Bigg ] \Bigg \} \Bigg ] . } \end{array}\tag{3}
$$

In (3), we have employed the variational method using weights in neural networks (NNs) [16]. Thus, our posterior distributions are approximated by NNs, resulting in $p _ { \theta } ^ { \mathrm { c u } }$ and $p _ { \phi _ { i } } ^ { \mathrm { s u } }$ , where $\pmb { \theta }$ and $\phi _ { i }$ represents the NN’s parameters approximating the CU and i-th SU, respectively. We emphasize the joint semantic and channel coding performed by the encoders as we consider $\begin{array} { r } { p _ { \phi _ { i } } ( \pmb { y } _ { i } | \pmb { s } _ { i } ) = \int p _ { \phi _ { i } } ^ { \mathrm { s u } } ( \pmb { x } _ { i } | \pmb { c } ) p ( \pmb { y } _ { i } | \pmb { x } _ { i } ) } \end{array}$ dx . For the sake of simplicity we use $p _ { \phi _ { i } } ^ { \mathrm { s u } ^ { \prime } } = p _ { \phi _ { i } } ( \pmb { y } _ { i } | \pmb { s } _ { i } )$ in (3). In addition, although the i-th decoder $p ( \mathfrak { z } _ { i } | \mathfrak { y } _ { i } )$ can be fully determined using the known distributions, due to the highdimensional integrals, we approximate it with $q _ { \psi _ { i } } ( z _ { i } | \pmb { y } _ { i } )$ where $\psi _ { i }$ represents NNs approximating the distribution of the i-th decoder.

To obtain the empirical estimate of the above CMT-SemCom objective function, we approximate the expectations using Monte Carlo sampling, considering the training data set of M samples indexed by m $\mathcal { D } = \{ ( \boldsymbol { s } ^ { ( m ) } , z _ { 1 } ^ { ( m ) } , \ldots , z _ { N } ^ { ( m ) } ) \} _ { m = 1 } ^ { M }$ . In the following, we present the empirical loss functions $\mathcal { L } _ { i } ( \theta , \phi _ { i } , \psi _ { i } )$ for classification tasks and regression tasks separately.

## A. Classification Loss Function

To maximize the objective function (3), which is the negative cross-entropy, we estimate the cross-entropy via data samples m and noise samples $n ,$ which results in the following maximization objective for the classification tasks:

$$
\mathcal { L } _ { i } ^ { \mathrm { C l a s s i f } } ( \theta , \phi _ { i } , \psi _ { i } ) \approx \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \bigg [ \frac { 1 } { L } \sum _ { l = 1 } ^ { L } [ \log q _ { \psi _ { i } } ( z _ { i } ^ { ( m ) } | \pmb { y } _ { i } ^ { ( m , l ) } ) ] \bigg ] ,\tag{4}
$$

where we fix the sample size of the encoder output and the channel sampling ${ \pmb y } _ { m , l } = { \pmb x } _ { m } + { \pmb n } _ { l }$ , equal for each batch. For implementation, we then minimize the sample-estimated crossentropy instead of maximizing the negative sample-estimated cross-entropy in (4).

Finally, for the classification inference step, we take the maximum likelihood of the learned $q _ { \psi _ { i } } ^ { \star } ( z _ { i } | \pmb { y } _ { i } )$ to get the estimated semantic variable $\hat { z } _ { i }$

$$
\hat { z } _ { i } = \underset { \hat { z } _ { i } } { \arg \operatorname* { m a x } } q _ { \psi _ { i } } ^ { \star } ( z _ { i } | \pmb { y } _ { i } ) .\tag{5}
$$

## B. Regression Loss Function

Optimizing over $q _ { \psi _ { i } } \big ( z _ { i } | \pmb { y } _ { i } \big )$ for a regression task is generally computationally infeasible, as this amounts to optimizing over an infinite-dimensional space of continuous pdfs. A practical way to estimate $q _ { \psi _ { i } } \left( z _ { i } | \boldsymbol { y } _ { i } \right)$ is to assume a parameterized continuous distribution shape with a tractable number of parameters [17]. Thus, we assume a pixel-wise independent Gaussian pdf with a tractable number of parameters, given by $z _ { i _ { k } } \sim \mathcal N ( \mu _ { k } , \sigma ^ { 2 } )$ for the k-th pixel among the $K = H \times W$ pixels in the input image. We assume equal variances for all pixels’ pdfs, and then the Mean Square Error (MSE) loss is obtained from (3) [18].

Assuming a given fix variance $\sigma ^ { 2 }$ , the output of $q _ { \psi _ { i } } \left( z _ { i } | \boldsymbol { y } _ { i } \right)$ which is represented by a NN, is the mean vector $\pmb { \mu } _ { \pmb { \psi } _ { i } } ( \pmb { y } _ { i } ) \in$ $\mathbb { R } ^ { K }$ of the Gaussian distributions, which depends on input $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } _ { i } } .$ For simplicity in notation, we omit including the $\psi _ { i }$ in $\pmb { \mu } ( \pmb { y } _ { i } )$

The pdf $q _ { \psi _ { i } } \big ( z _ { i } | \pmb { y } _ { i } \big )$ is fully described by the mean $\pmb { \mu } ( \pmb { y } _ { i } )$ , and the log-likelihood function is given as:

$$
\begin{array} { l } { { \displaystyle \log q _ { \psi _ { i } } ( z _ { i } | y _ { i } ) } } \\ { { \displaystyle = \log \left( \prod _ { k = 1 } ^ { K } \frac { 1 } { \sqrt { 2 \pi \sigma ^ { 2 } } } \exp \left( - \frac { \left[ z _ { i _ { k } } - \mu _ { k } \left( y _ { i } \right) \right] ^ { 2 } } { 2 \sigma ^ { 2 } } \right) \right) } } \\ { { \displaystyle = - \sum _ { k = 1 } ^ { K } \left( \frac { \left[ z _ { i _ { k } } - \mu _ { k } \left( y _ { i } \right) \right] ^ { 2 } } { 2 \sigma ^ { 2 } } \right) - \frac { K } { 2 } \log \left( 2 \pi \sigma ^ { 2 } \right) . } } \end{array}\tag{6}
$$

The constant term ${ \textstyle { \frac { K } { 2 } } } \log \left( 2 \pi \sigma ^ { 2 } \right)$ in (6) is ignored in the maximization of the objective function, and we set $K = 2 \sigma ^ { 2 }$

Integrating (6) into the objective function in (3), we obtain the empirical regression loss function as follows:

$$
\mathcal { L } _ { i } ^ { \mathrm { R e g r } } ( \pmb { \theta } , \phi _ { i } , \psi _ { i } ) \approx - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \bigg [ \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \mathrm { M S E } \Big ( z _ { i } ^ { ( m ) } , \pmb { \mu } ( \pmb { y } _ { i } ^ { ( m , l ) } ) \Big ) \bigg ] ,\tag{7}
$$

$$
\begin{array} { r } { \mathrm { M S E } ( z _ { i } , \mu ( \pmb { y } _ { i } ) ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( z _ { i _ { k } } - \mu _ { k } ( \pmb { y } _ { i } ) ) ^ { 2 } . } \end{array}
$$

Finally, semantic variable $\hat { z } _ { i }$ is estimated using the maximum likelihood criterion by

$$
\hat { z } _ { i } = \underset { \hat { z } _ { i } } { \arg \operatorname* { m a x } } q _ { \psi _ { i } } ^ { \star } ( z _ { i } | \pmb { y } _ { i } ) = \pmb { \mu } ( \pmb { y } _ { i } ) ,\tag{8}
$$

which is the mean of $q _ { \psi } ^ { \star } ( z _ { i } | \boldsymbol { y } _ { i } )$ as it is Gaussian-shaped, and has its maximum at $\pmb { \mu } ( \pmb { y } _ { i } )$

## IV. SIMULATION RESULTS

In this section, we evaluate the performance of CMT SemCom by comparing its task execution against baselines, in addition to investigating the impact of the CU capacity.

## A. Simulation Setup

1) Dataset and Tasks: For our evaluations, we use the Cityscapes dataset [15], which contains 5, 000 finely annotated urban street-scene images collected from 50 cities. We use only the fine-annotated images, as the dataset provides both pixel-level semantic labels and stereo disparity maps, making it suitable for simultaneous classification and regression tasks. Since disparity is inversely proportional to depth, we convert the disparity maps into normalized floating-point depth labels using the maximum disparity value. Only pixels with positive disparity values are considered valid, while pixels with zero disparity are treated as invalid and excluded from the depth estimation evaluation.

The dataset is shaped as $\boldsymbol { \mathcal { D } } = \{ ( \boldsymbol { s } ^ { ( m ) } , \boldsymbol { z } _ { 1 } ^ { ( m ) } , \boldsymbol { z } _ { 2 } ^ { ( m ) } , \boldsymbol { z } _ { 3 } ^ { ( m ) } ) \} _ { m = 1 } ^ { M } ,$ , with $\begin{array} { r } { Z _ { 1 } ^ { \mathrm { ~ \tiny ~ \left. ~ \in ~ \right. ~ } } \mathcal { Z } _ { 1 } ^ { H \times W } , } \end{array}$ $Z _ { \mathrm { 2 } } ~ \in ~ \mathcal { Z } _ { \mathrm { 2 } } ^ { H \times W } ,$ , and $Z _ { 3 } ~ \in ~ \mathbb { R } ^ { H \times W }$ . Specifically, the input images we use from the dataset have a resolution of $H \times W \times C = 2 5 6 \times 5 1 2 \times 3$

We consider the execution of three tasks $N = 3$ . For the first two tasks, we consider semantic segmentation of classes: $\mathcal { Z } _ { 1 } =$ {‘road’, ‘sidewalk’, ‘building’, ‘car’, ‘otherwise’} as Task 1 and semantic segmentation of $\mathcal { Z } _ { 2 } = \{ \mathrm { \cdot b u i l d i n g } ^ { \mathrm { , } }$ , ‘otherwise’} as Task 2. Thus, Task 1 is a multinomial classification, and Task 2 is a binary classification. We consider the third task to be a regression task by defining Task 3 to be pixel-level depth estimation, with an illustrative example of the dataset shown in Fig. 4. Using (4) and $( 7 )$ , the total loss function of the CMT-SemCom is given as:

$$
\begin{array} { r l } & { { \mathcal L } ( \pmb { \theta } , \Phi , \Psi ) = } \\ & { { \mathcal L } _ { 1 } ^ { \mathrm { C l a s s i f } } ( \pmb { \theta } , \phi _ { 1 } , \psi _ { 1 } ) + { \mathcal L } _ { 2 } ^ { \mathrm { C l a s s i f } } ( \pmb { \theta } , \phi _ { 2 } , \psi _ { 2 } ) + { \mathcal L } _ { 3 } ^ { \mathrm { R e g r } } ( \pmb { \theta } , \phi _ { 3 } , \psi _ { 3 } ) . } \end{array}
$$

TABLE I: NN architecture of the CMT-SemCom.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Layer</td></tr><tr><td rowspan=1 colspan=1>CU</td><td rowspan=1 colspan=1>Conv (#filters = 96, stride = 2), BatchNorm, ReLUF ResNet blocks</td></tr><tr><td rowspan=1 colspan=1>SU</td><td rowspan=1 colspan=1>9 – F ResNet blocksGlobal average poolingDense (output size = 100), BatchNorm, ReLUDense (output size = number of channel-uses d = 128), ReLUPower normalization</td></tr><tr><td rowspan=1 colspan=1>Dec</td><td rowspan=1 colspan=1>Dense, ReLU, Reshape (to 32 × 64 × 64)ConvT(#filters = 64 , stride = 2), ReLUConvT(#filters = 32, stride = 2), ReLUConvT(#filters = 16, stride = 2), ReLUTask 1: Conv(#filters = 5, stride = 1), SoftmaxTask 2: Conv(#filters = 1 stride = 1), SigmoidTask 3: Conv(#filters = 1 stride = 1), ReLU</td></tr></table>

2) CMT-SemCom NN Architecture: Table I summarizes the NN architecture of the proposed CMT-SemCom. The CU consists of one convolutional (Conv) layer followed by F ResNet blocks [19], while each SU contains the remaining 9 − F ResNet blocks, keeping the total depth per task fixed at nine blocks. By varying $F ,$ we investigate how distributing feature extraction between the shared CU and task-specific SUs affects task performance, as discussed in Section IV-B. The CU output has dimensions $1 6 \times 3 2 \times 6 4$ , which is selected after a preliminary investigation to have the best performance. Each SU then applies a global average pooling layer to produce a compact feature vector, which is processed by two fully connected layers and a power normalization layer before transmission with d channel-uses.

The decoder architecture is identical for all tasks except for the final task-specific output layer (Table I). It first projects the received latent representation through a fully connected layer and reshapes it into a 32 × 64 × 64 feature map. Three transposed convolution (ConvT) layers then progressively upsample the features to the original image resolution of $2 5 6 \times 5 1 2$ followed by the final task-specific Conv layer.

The network is trained for 120 epochs using the Adam optimizer with an initial learning rate of $1 0 ^ { - 4 } .$ . The learning rate decays by a factor of $1 0 ^ { - 1 }$ every 20 epochs starting from epoch 80.

3) Benchmarks: We compare CMT-SemCom’s performance against three baselines: single-task SemCom (ST-SemCom), single-encoder multi-decoder SemCom (SEMD-SemCom), and a conventional task-agnostic digital transmission. We note that in all cases, we use the same NN architectural components, e.g., 9 successive ResNet blocks per task. However, the properties of the NN architectural components are selected such that the total number of trainable parameters remains equal among the cases for a fair comparison.

ST-SemCom: For ST-SemCom, three independent semantic encoder/ decoder are implemented for each task, meaning that there exists no CU shared between them.

SEMD-SemCom: We also compare the split structure, CMT-SemCom, against the state-of-the-art MTL in SemCom [5], where there exists only one shared encoder on the Tx side and multiple decoders on the Rx.

Task-agnostic digital transmission (TAD): The benchmark follows a classical separate source and channel coding approach. For source coding, we examined two compression algorithms: JPEG with variable image quality and Huffman coding applied to variable quantization levels of the pixel values. Given our target SNR range and the constrained number of channel uses $( d ) ,$ JPEG performed worse than Huffman coding in terms of image reconstruction quality. Therefore, we use Huffman coding as the source coding algorithm. We consider both color and grayscale image reconstruction, with subsampling factors from 1 to 50 and quantization levels from 2 to 8 bits per pixel, where 8 bits represents the original pixel precision. Here, subsampling reduces the spatial resolution by retaining fewer pixels, while quantization reduces the number of bits used to represent each pixel. For channel coding, an LDPC code is used with a rate between 0.2 and 0.95. Moreover, modulation schemes of 4, 16, and 64-QAM were examined. All these parameters are optimized by grid search to minimize the MSE of the reconstructed image at 0 dB SNR, which resulted in a subsampling factor of 45, grayscale image reconstruction, quantization to 4 bits with Huffman coding, a channel coding rate of 0.254, and the use of 4-QAM. Finally, the reconstructed image is fed into a NN to execute all three tasks, using an architecture similar to that of CMT-SemCom. The NN for the task execution is also trained using the noisy reconstructed images for the best possible inference result.

4) Evaluation Metrics: We evaluate the segmentation tasks by two metrics. First, we consider pixel-level classification accuracy of classes averaged over all image pixels, measuring the fraction of valid pixels whose inferred labels match the corresponding ground-truth labels. Second, we use intersection over union (IoU), which is assessed per class and measures the intersection of the inferred segmentation and the ground truth, divided by the union of true and false positives and false negatives [20]. True positives (TP) are correctly identified object pixels, while false positives (FP) are pixels incorrectly labeled as the object. False negatives (FN) are pixels incorrectly labeled as no object. IoU is defined as:

$$
\mathrm { I o U } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } .
$$

Particularly, we use IoU to evaluate Task 2 of ‘building segmentation, and for Task 1, we use mean IoU (mIoU), which averages the IoU scores for all the defined classes in $\mathcal { Z } _ { 1 }$

In addition to the MSE, regression performance is evaluated using the $\delta _ { 1 }$ accuracy [21]. $\delta _ { 1 }$ measures the percentage of predicted depth pixels whose relative error falls within a specific threshold. Specifically, it is the fraction of pixels for which the maximum ratio between the ground-truth depth and the predicted depth is less than 1.25.

$$
\delta _ { 1 } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { I } \bigg \{ \operatorname* { m a x } \bigg ( \frac { z _ { k } } { \hat { z } _ { k } } , \frac { \hat { z } _ { k } } { z _ { k } } < 1 . 2 5 \bigg ) \bigg \} ,
$$

where $\mathbb { I } ( \cdot )$ denotes the indicator function and K is the total number of pixels with valid depth ground truth.

![](images/ec1cc226a9694a25ae4b6f1d0dbf11e2b1071507fba4646a84be1030da5a2b19.jpg)

![](images/55a60998caeef229181e89d023c65f7dbe903c71bd7d93dc8240bf7bcd025664.jpg)

![](images/f8ebfc373ea213cf8e9e4d9d18fcf8e7f36303d79538ec8affe751432bc843b0.jpg)  
CMT-CU9/SU0 CMT-CU7/SU2 CMT-CU5/SU4 CMT-CU2/SU6 CMT-CU1/SU8  
Fig. 2: Architectural comparison for all three tasks over the training epochs at 0 dB SNR.

## B. Evaluations

1) Architectural Investigations: First, we investigate the impact of the CU/SU processing split on the balance between shared semantic extraction and task-specific refinement. Fig. 2 shows the task performance over training epochs at 0 dB SNR and $d \ : = \ : 1 2 8$ channel-uses for different CMT-SemCom architectures, ranging from CMT-CU9/SU0 to CMT-CU1/SU8. For instance, CMT-CU9/SU0 has all 9 ResNet blocks in the CU, meaning that most processing is done in the CU with only dense layers in the SU, while CMT-CU1/SU8 shifts most processing to the task-specific SUs with 1 ResNet block in the CU, and 8 ResNet blocks in each SU. As shown in Fig. 2, a balanced distribution of the processing capacity between the CU and the SU provides the best performance, with CMT-CU5/SU4 achieving the best for all three tasks. Therefore, CMT-CU5/SU4 is adopted as the default CMT-SemCom architecture in the remaining experiments.

2) Benchmark Comparisons: We now compare CMT-SemCom against ST-SemCom, SEMD-SemCom, and TAD over an SNR range as shown in Fig. 3. The training SNR is 0 dB. The comparison uses pixel-level classification accuracy and IoU for Tasks 1 and Task 2, and MSE and $\delta _ { 1 }$ accuracy for Task 3. Increasing the SNR from −5 to 10 dB improves performance across all metrics for the considered methods due to the more favorable channel conditions.

The proposed CMT-SemCom framework outperforms ST-SemCom on all three tasks and all SNR values, indicating that joint processing in the CU helps in extracting the relevant semantic feature for all three tasks.

Although SEMD-SemCom achieves the same pixel-level classification accuracy as CMT-SemCom on Task 1, CMT-SemCom attains a slightly higher IoU. However, for Tasks 2 and 3, CMT-SemCom consistently outperforms SEMD-SemCom. These results indicate that the SUs are crucial for task-specific refinements on the Tx side.

Finally, we compare CMT-SemCom with TAD transmission. For a fair comparison, we set the number of channel-uses to $3 8 4 = 3 \times 1 2 8$ in TAD. This keeps the comparison symbolwise fair between these methods as we transmit $3 \times 1 2 8$ realsymbols for CMT-SemCom and $3 \times 1 2 8$ complex-symbols for

(a) Task 1  
![](images/9b1a6d22392de7c106ddd0f9b97074a5668582d1e3a0d1846c373979185206ae.jpg)

(b) Task 1  
![](images/779bb2a86f8ce06ccdb2e1e24a1e45cbed37b97a45796282383198e06ce499b0.jpg)

![](images/9df2e4ad9753d74b9bb30d8c970ec11ccb8b8c35e0e7ebad78184a752c26de80.jpg)

![](images/fa2f74d7022bcf3c877625888b2c49c679d04c8b451e41186b4f88837c25c374.jpg)

![](images/2a41c954b236fc55d5a16e441ced4c93601762bc0c004439f2e7ab50d3ceafc8.jpg)

![](images/6b9dd623ff00dd6fcc5beb991ce3834e0bc374bd38c4b5cf373b2dce847ba554.jpg)  
Fig. 3: Task execution comparison of CMT-SemCom vs. benchmarks over SNR.

![](images/e22918ef0107ec7fb6cb829aefec4ec7b72c7a489430817c2fff4aeeb10e601d.jpg)  
Fig. 4: Visualization of CMT-SemCom performance on the Cityscapes dataset trained and evaluated at 0 dB SNR.

TAD. The task execution accuracy after TAD transmission is worse compared to CMT-SemCom, as the image reconstruction quality is low for the limited number of channel-use and 0 dB SNR. The results highlight the advantage of CMT-SemCom over conventional task-agnostic digital transmission in task-oriented communication, especially in bandwidthconstrained and low-SNR conditions.

Fig. 4 visualizes the performance of CMT-SemCom for all three tasks and compares it with ground truth and ST-SemCom for one illustrative example of the dataset. For both segmentation tasks, as shown, target classes are detected more accurately than by ST-SemCom, and the detected class masks are less fragmented and show better overlap with the ground truth. Regarding Task 3 for better visualization, the mean absolute error $\Delta$ between $z _ { 3 }$ and the inferred $\hat { z } _ { 3 }$ are shown, where $z _ { 3 }$ and $\hat { z } _ { 3 }$ are normalized to be between 0 and 1. Lowerror regions with $0 ~ \le ~ \Delta ~ < ~ 0 . 0 3$ are indicated in green, moderate-error regions with $0 . 0 3 \leq \Delta < 0 . 0 7$ are indicated in yellow, high-error regions with $0 . 0 7 \leq \Delta \leq 1$ are indicated in red, and pixels with invalid ground truth are indicated in black. It can be seen that CMT-SemCom has larger low-error regions, less moderate-error regions, and nearly the same higherror regions compared to ST-SemCom.

## V. CONCLUSION

We extended CMT-SemCom to heterogeneous multi-task perception by jointly supporting classification and regression tasks on the Cityscapes dataset using an InfoMax formulation that accommodates mixed discrete and continuous semantic variables. Simulation results showed that the proposed framework outperforms single-task SemCom, single-encoder multi-decoder SemCom, and task-agnostic digital transmission, while requiring the same communication resources. In addition, we demonstrated that the distribution of feature extraction between the shared CU and task-specific SUs plays a crucial role in balancing information sharing and task specialization, providing practical design insights.

## REFERENCES

[1] A. Halimi Razlighi, C. Bockelmann, and A. Dekorsy, “Semantic communication for cooperative multi-task processing over wireless networks,” IEEE Wireless Communications Letters, vol. 13, no. 10, pp. 2867–2871, 2024.

[2] Y. Shi, Y. Zhou, D. Wen, Y. Wu, C. Jiang, and K. B. Letaief, “Taskoriented communications for 6G: Vision, principles, and technologies,” IEEE Wireless Communications, vol. 30, no. 3, pp. 78–85, 2023.

[3] D. Gund ¨ uz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener, K. K.¨ Wong, and C.-B. Chae, “Beyond transmitting bits: Context, semantics, and task-oriented communications,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 1, pp. 5–41, 2023.

[4] L. Chen, Z. Yang, J. Ma, and Z. Luo, “Driving scene perception network: Real-time joint detection, depth estimation and semantic segmentation,” in 2018 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2018, pp. 1283–1291.

[5] G. Zhang, Q. Hu, Z. Qin, Y. Cai, G. Yu, and X. Tao, “A unified multi-task semantic communication system for multimodal data,” IEEE Transactions on Communications, vol. 72, no. 7, pp. 4101–4116, 2024.

[6] R. Caruana, “Multitask learning,” Machine learning, vol. 28, pp. 41–75, 1997.

[7] Q. Bian, Z. Bao, C. Dong, X. Xu, Y. Luo, and R. Meng, “Multitask semantic communication for remote sensing,” in 2025 IEEE 5th International Conference on Computer Communication and Artificial Intelligence (CCAI), 2025, pp. 70–75.

[8] Z. Zhu, R. Zhang, X. Cheng, and L. Yang, “Synesthesia of machinesenabled multi-task semantic communication system,” IEEE Transactions on Mobile Computing, vol. 25, no. 4, pp. 4632–4647, 2026.

[9] A. Halimi Razlighi, C. Bockelmann, and A. Dekorsy, “Semantic communication for cooperative multi-tasking over rate-limited wireless channels with implicit optimal prior,” IEEE Open Journal of the Communications Society, vol. 6, pp. 8523–8538, 2025.

[10] A. Halimi Razlighi, M. H. V. Tillmann, E. Beck, C. Bockelmann, and A. Dekorsy, “Cooperative and collaborative multi-task semantic communication for distributed sources,” in ICC 2025 - IEEE International Conference on Communications, 2025, pp. 3966–3971.

[11] A. Mousavian, H. Pirsiavash, and J. Koseckˇ a, “Joint semantic segmen-´ tation and depth estimation with deep convolutional networks,” in 2016 Fourth International Conference on 3D Vision (3DV). IEEE, 2016, pp. 611–619.

[12] Z. Zhang, Z. Cui, C. Xu, Z. Jie, X. Li, and J. Yang, “Joint taskrecursive learning for semantic segmentation and depth estimation,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 235–251.

[13] L. Hoyer, D. Dai, Y. Chen, A. Koring, S. Saha, and L. Van Gool, “Three ways to improve semantic segmentation with self-supervised depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 11 130–11 140.

[14] X. Yu, T. Lv, W. Li, W. Ni, D. Niyato, and E. Hossain, “Multi-task semantic communication with graph attention-based feature correlation extraction,” IEEE Transactions on Mobile Computing, vol. 24, no. 5, pp. 4371–4388, 2025.

[15] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[16] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.

[17] L. Wasserman, All of statistics: a concise course in statistical inference. Springer, 2004, vol. 26.

[18] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016, http://www.deeplearningbook.org.

[19] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[20] M. Everingham, S. A. Eslami, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes challenge: A retrospective,” International journal of computer vision, vol. 111, no. 1, pp. 98–136, 2015.

[21] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” in Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2, ser. NIPS’14. Cambridge, MA, USA: MIT Press, 2014, p. 2366–2374.