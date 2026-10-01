# Client and Training Data Selection for Computationally Efficient Synchronized Federated Learning

Muzaffer Citir<sup>†</sup>, Hiroki Nishikawa<sup>††</sup>, Sangyoung Park<sup>†</sup>

<sup>†</sup>Smart Mobility Systems, Technical University of Berlin, Germany

<sup>††</sup>Graduate School of Information Science and Technology, The University of Osaka, Japan

<sup>†</sup>{muzaffer.citir, sangyoung.park}@tu-berlin.de, <sup>††</sup>nishikawa.hiroki@ist.osaka-u.ac.jp

Abstract—Federated learning (FL) is a promising paradigm of machine learning, which preserves user privacy by enabling learning without sharing raw data with a cloud server. Straggling clients have been a problem for FL as they introduce delays in aggregating the local models and hence, the convergence of the global model. Therefore, it is important to have a mechanism that ensures fast convergence of the global model as well as good FL participation rate. Another issue for the convergence of a model in FL is the non-independent and identically distributed (non-iid) data across the clients. Prior approaches based on probabilistic client selection do not work well under non-iid data especially when the number of clients is small. We show scenarios where such approaches fail and propose a joint client-training data selection algorithm for fast convergence of FL models. Our experiments on CIFAR-100 dataset show that convergence of the FL model can be significantly improved over prior works that can consider non-iid data and heterogeneous computation and higher model accuracy.

Index Terms—federated learning, client selection, non-IID data, probabilistic scheduling, real-time systems

## I. INTRODUCTION

Federated learning (FL) has emerged as a paradigm, enabling a multitude of edge devices to engage in distributed learning while upholding user privacy and data security [1]. The concept involves edge devices training local ML models on their own data and periodically sharing model parameters with each other, thus maintaining data privacy [2]–[4]. Despite its advantages in data privacy, parallel processing, and reducing data transmission delays to cloud servers (CS), FL still faces challenges in time-sensitive ML applications [5]. First, model training is performed on heterogeneous edge devices where computing and communication resources are limited. Therefore, overall model training could take an indefinitely long time if model aggregation is delayed due to a few stragglers [6]–[8]. Second, data is often non-independent and identically distributed (non-iid) across the edge devices. In case of classification problems, the model accuracy could be undermined if certain classes are not represented well in the training dataset because they are located in resourceconstrained devices, and therefore, cannot participate in FL well.

Fig. 1 depicts how stragglers could impact the performance of FL training. Here, we assume synchronous update of the model where a CS waits for all participants in the FL round before aggregating the model. Local training on edge devices is generally regarded as a best-effort service, which can be interrupted by higher-priority tasks. This results in straggling edge devices, worsening the performance of model training and slowing the convergence of the FL model [6], [9]. We also assume a deadline (or timeout) by which local training must finish, as the CS cannot indefinitely wait for stragglers in an FL iteration. Fig. 1(a) shows how a lack of computing capacity and interruptions can lead to deadline misses in such an FL setting. In contrast, if an FL system is aware of the computation capacity and has stochastic information about potential causes of delay such as interruptions, it could ask the edge device to train on fewer data samples to ensure completion before the deadline (Fig. 1(b)).

Apart from meeting the FL timing constraints, which is mainly concerned with computing capacity and the amount of training data, the quality of training data is also crucial for the accuracy of the global model. Although prior works proposed client selection and resource management schemes for FL under a deadline [10], [11], most of them assume iid data, where the convergence of the model performance is less tricky [12]. Even if edge devices are able to complete the training and send the model to the CS in time, the convergence of the global model will be slow unless the training dataset is sufficiently balanced. To address such issues, this paper proposes a device selection and data allocation algorithm for edge FL that minimizes the overall model training time under a timing constraint per FL iteration and non-iid data across the edge devices. It is assumed that the CS cannot predict exactly when the training on participating devices finishes due to the uncertainty in training and interruptions by other tasks, the algorithm takes a probabilistic approach and tries to maximize the expected value of the training data that can contribute to model aggregation each FL round. The algorithm also tries to achieve a balanced class distribution during the client selection and when it is determining the training dataset size. The contributions of this paper are as follows:

![](images/09933ebee972220f87548f1be0fe77f2a604f4c323fb69b378246f2f7ff1499d.jpg)  
Fig. 1. (a) A deadline miss example of FL devices scheduling, and (b) avoiding it with proper data allocation.

• We show the impact of the training dataset size and class distribution on the training times and model accuracy

• We incorporate a probabilistic approach for estimating training times and task interruptions

• A joint device and non-iid-aware training data selection algorithm for determining the training data size and balance between classes.

The rest of this paper is organized as follows: Section II summarizes the related work. Section III describes the probabilistic system model backed by measurements. Section IV proposes the joint client and data selection algorithm for fast FL model convergence. Section V shows the experimental results and a comparison with the state of the art. Finally, Section VI concludes this paper with future remarks.

## II. RELATED WORK

Numerous studies on FL looked into optimization of communication and computational resources to boost system efficacy. Wang et al. introduced algorithms to enhance resource utilization and diminish energy expenditure in 5G edge networks by modulating the frequency of global aggregation at edge servers [13]. Broadening this scope, Wadu et al. examined client scheduling and resource allocation, introducing a strategy that employs Gaussian process regression for channel prediction to mitigate accuracy loss in FL [14]. This strategy was subsequently enhanced to incorporate the computational resources of clients, presenting a more integrated perspective on resource optimization [15]. In a similar fashion, Li et al. devised SmartPC, a system with dual-level controllers designed to optimize the FL learning process, showing the potential efficiency gains from device selection and configuration adjustments [11]. These works, however, consider iid data across the clients, which means the data quality does not change significantly depending on client selection.

In a related research direction, Amiri et al. tackled the optimization of communication resources through device scheduling policies that consider both channel conditions and the significance of local model updates under non-iid data [16]. This work was further elaborated with a convergence analysis and a model update compression strategy [17]. Furthermore, to address latency issues, Xia et al. and Xu et al. proposed client scheduling methodologies grounded in multi-armed bandit theory, aimed at reducing training latency—a crucial aspect that highlights the complex nature of efficiency in FL systems [18], [19]. A heuristic method to select computationally efficient (powerful hardware) and statistically efficient (useful data) [20] has been proposed based on the loss value after local training. Such methods use metrics, which are available only after the local training, therefore wasting precious computation resources even though the client is not selected in that FL round.

Wang et al. later introduced Fed-LBAP and FedMinAvg methods designed to regulate workload distribution to minimize computational time and accuracy loss during FL model training [21]. They also presented an algorithm, MinCost, to tackle the optimization challenge [22], and proposed OLAR [23], an optimal greedy solution to reduce computational time in FL training. Furthering this line of research, subsequent work developed scheduling algorithms for FL aimed at reducing energy consumption by managing workload distribution [24], predicated on the notion that training’s computational latency can be modulated by workload allocation [25]. An approach that modifies the client selection probability based on the gradient value has proven to be effective [26]. Unlike these approaches, in our method, clients that do not participate in FL do not waste any computation resources, and that most helpful clients are deterministically selected.

Our work is distinguished from other works in that i) it considers non-iid data and explicitly balances class distribution for an FL round, ii) it does not require local training to complete for client selection, and iii) deterministically selects the most helpful clients. To the best of the authors’ knowledge, no other works have simultaneously considered all these points. The details and effectiveness of the proposed approach are elaborated in the rest of the paper.

## III. SYSTEM MODEL

## A. Overall FL Architecture

The FL system consists of a single Cloud Server (CS) and M edge devices, denoted as $M = \{ 1 , 2 , \dots , M \}$ . Each device i possesses a local dataset $D _ { i } = \left\{ x _ { i , n } \in \mathbb { R } ^ { s } , y _ { i , n } \in \mathbb { R } \right\} _ { n = 1 } ^ { | D _ { i } | }$ This paper assumes that each device i uses a part of $D _ { i }$ for training, denoted as $d _ { i }$ , to meet the deadline of an iteration, $T ^ { d e a d }$ . Here, $x _ { i , n }$ denotes n-th s-dimensional input data vector at device i, and $y _ { i , n }$ the corresponding labeled output for $x _ { i , n } .$ . Here we also make similar assumption to existing FL papers such that the amount of dataset size as well as the class information is known to the device [1].

The primary objective of FL training is to determine the model parameter w that minimizes a loss function to the entire dataset, which is given by:

$$
\operatorname* { m i n } _ { w } F ( w ) : = \operatorname* { m i n } _ { w } \frac { 1 } { M } \sum _ { i \in M } f _ { i } ( w ) ,\tag{1}
$$

where the local loss function $f _ { i } ( w )$ on the dataset $d _ { i }$ is defined as $\begin{array} { r } { f _ { i } ( w ) = \frac { 1 } { | d _ { i } | } \sum _ { n \in d _ { i } } f ( w , x _ { i , n } , y _ { i , n } ) } \end{array}$ , and $f ( w , x _ { i , n } , y _ { i , n } )$ captures the error of the model parameter w on the inputoutput pair $\{ x _ { i , n } , y _ { i , n } \}$

Our FL uses an iterative approach to solve Equation (1). Each round, indexed by k, contains the following three steps: (1) The CS broadcasts a global model $w _ { k }$ to a number of edge devices, which are assumed to be under randomly selection (RS) by the CS and participate in the round. (2) Each device i updates its local model by a gradient descent algorithm on the local dataset (i.e., $w _ { k + 1 } ^ { i } = w _ { k } - \eta \nabla f _ { i } ( w _ { k } )$ , where η is a learning rate), and uploads the updated model $w _ { k + 1 } ^ { i }$ to the CS until a deadline. If a device misses the deadline, the updated model is disposed. (3) The CS aggregates all the models uploaded until the deadline to generate a new global model $\begin{array} { r } { w _ { k + 1 } = \frac { 1 } { | \Pi _ { k } | } \sum _ { i \in \Pi _ { k } } w _ { k + 1 } ^ { i } } \end{array}$

## B. Probabilistic Modeling of Time-sensitive FL

Hereafter, we consider an arbitrary round k in this paper and omit the index without loss of generality. We model the total latency as the sum of three components: computation latency $( t _ { \mathrm { c p } } ) _ { \mathrm { : } }$ , interruption latency $( t _ { \mathrm { i n t r } } )$ , and communication latency $\left( t _ { \mathrm { c o m m } } \right)$ . The cumulative probability that the total latency satisfies a given deadline, $T _ { \mathrm { d e a d } }$ , is expressed as:

$$
P ( t _ { \mathrm { c p } } + t _ { \mathrm { i n t r } } + t _ { \mathrm { c o m m } } \leq T _ { \mathrm { d e a d } } ) .\tag{2}
$$

The CS determines the training dataset size $d _ { i }$ for each edge device to improve the learning rate while ensuring this probabilistic constraint is satisfied. Below, we detail the modeling of each latency component.

Computational Latency: The computation latency $t _ { \mathrm { c p } }$ of local training on an edge device, given a dataset size $d _ { i } ,$ is modeled using a shifted exponential distribution:

$$
P ( t _ { \mathrm { c p } } \leq t ) = 1 - \exp \left( - \frac { \mu _ { i } } { d _ { i } } ( t - a _ { i } d _ { i } ) \right) ,\tag{3}
$$

where $a _ { i } > 0$ represents the device’s maximum computation capacity, and $\mu _ { i } > 0$ indicates the fluctuation in computation capacity. The minimum computation time is given by $a _ { i } d _ { i }$ and the probability increases as t grows. The parameters $a _ { i }$ and $\mu _ { i }$ are determined by the performance of the edge device and the characteristics of the FL application. The mean and variance of $t _ { \mathrm { c p } }$ are:

$$
\mathbb { E } [ t _ { \mathrm { c p } } ] = a _ { i } d _ { i } + \frac { d _ { i } } { \mu _ { i } } , \quad \mathrm { V a r } [ t _ { \mathrm { c p } } ] = \frac { d _ { i } ^ { 2 } } { \mu _ { i } ^ { 2 } } .\tag{4}
$$

![](images/b682ecaa2cbb3daf0d50c88e27d82b0c5d6349f603546b416b3c2c02f42de4b9.jpg)  
Fig. 2. Training time vs dataset size for CIFAR-10 dataset.

![](images/b4cf2a1387854750a8eecebcebeebcd452090d2c39dfa404c6cf4137fcc397bc.jpg)  
Fig. 3. Time distribution with $5 \times 1 0 ^ { 3 }$ images of CIFAR-10.

In order to validate the presented computation latency model, we have conducted preliminary experiments that measure the training times of CIFAR-10 dataset [27] using the hardware specified in Table I. In the validation, we have run 660 times of training with CIFAR-10 on the environment and measured the training times. Fig. 2 shows the result, where xand y-axes show the number of images used to training and the measured training time (secs), respectively. This measurement indicates that we have observed the linear relationship between the training dataset size and training time. Next, we depict a histogram with 120 runs to yield the probability distribution of training times, which roughly follows the exponential distribution, as shown in Fig. 3. An example $\mu _ { i }$ and $a _ { i }$ values for our setup as a result of curve fitting is also shown in the graph. In conclusion, it has been observed that the results of the model are well aligned with the outcomes of empirical experiments although it cannot be overlooked that the selected model may not perfectly mimic computation delay in every aspect. Throughout this paper, we assume that the CS is aware of the computation capabilities of edge devices, i.e., different versions of smartphones, infrastructure monitoring sensors, etc., and the application characteristics that dictate $\mu _ { i }$ and $a _ { i } .$

Interruption Latency: The interruption latency $t _ { \mathrm { i n t r } }$ arises due to preemption by other tasks on edge devices, such as a smartphone user launching an application [28]. This is modeled using an $M / M / 1$ queuing system:

$$
P ( t _ { \mathrm { i n t r } } \le t ) = 1 - \exp ( - \lambda _ { \mathrm { i n t r } } t ) ,\tag{5}
$$

where $\lambda _ { \mathrm { i n t r } }$ represents the rate of interruptions (Poisson process), and the service rate follows an exponential distribution with parameter $\mu _ { \mathrm { i n t r } } \geq \lambda _ { \mathrm { i n t r } }$ , where $\mu _ { \mathrm { i n t r } }$ represents the service rate of interruptions, i.e., how quickly interruption tasks are handled by the edge device. The mean and variance of $t _ { \mathrm { i n t r } }$ are:

TABLE I  
HARDWARE SPECIFICATIONS FOR FL TRAINING
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>CPU</td><td>Intel i9-12900KF 3,2 GHz</td></tr><tr><td>GPU</td><td>NVIDIA GeForce RTX 4090 24 GB (CUDA 12.2)</td></tr><tr><td>RAM</td><td>64 GB DDR5-SDRAM 4800 MHz</td></tr><tr><td>Operating System</td><td>Ubuntu 22.04.3 LTS</td></tr></table>

$$
\mathbb { E } [ t _ { \mathrm { i n t r } } ] = \frac { 1 } { \mu _ { \mathrm { i n t r } } - \lambda _ { \mathrm { i n t r } } } , \quad \mathrm { V a r } [ t _ { \mathrm { i n t r } } ] = \frac { 1 } { ( \mu _ { \mathrm { i n t r } } - \lambda _ { \mathrm { i n t r } } ) ^ { 2 } }\tag{6}
$$

Communication Latency: Communication latency t<sub>comm</sub> occurs when devices upload locally trained parameters to the CS. We model this latency as a Gaussian distribution:

$$
P ( t _ { \mathrm { c o m m } } \leq t ) = \Phi \left( \frac { t - \mu _ { \mathrm { c o m m } } } { \sigma _ { \mathrm { c o m m } } } \right) ,\tag{7}
$$

where $\mu _ { \mathrm { c o m m } }$ and $\sigma _ { \mathrm { c o m m } }$ are the mean and standard deviation of the communication latency, respectively. The mean and variance of $t _ { \mathrm { c o m m } }$ are:

$$
\mathbb { E } [ t _ { \mathrm { c o m m } } ] = \mu _ { \mathrm { c o m m } } , \quad \mathrm { V a r } [ t _ { \mathrm { c o m m } } ] = \sigma _ { \mathrm { c o m m } } ^ { 2 } .\tag{8}
$$

The weak dependency between training latency and communication latency allows us to approximate them as independent; thus assuming independence among the three components for simplicity, the cumulative probability can be expressed as:

$$
\begin{array} { r l } & { P ( t _ { \mathrm { t o t a l } } \leq T _ { \mathrm { d e a d } } ) = P ( t _ { \mathrm { c p } } \leq T _ { \mathrm { d e a d } } ) . } \\ & { \qquad P ( t _ { \mathrm { i n t r } } \leq T _ { \mathrm { d e a d } } ) \cdot P ( t _ { \mathrm { c o m m } } \leq T _ { \mathrm { d e a d } } ) } \end{array}\tag{9}
$$

## C. Problem Formulation

To ensure the deadline constraint is satisfied, the CS optimizes the dataset size $d _ { i }$ for each device. The optimization problem is formulated as:

$$
\operatorname* { m a x } _ { d _ { i } } \sum _ { i } U _ { i } ( d _ { i } ) ,\tag{10}
$$

subject to:

$$
U _ { i } ( d _ { i } ) = P ( t _ { \mathrm { t o t a l } } \leq T _ { \mathrm { d e a d } } ) \cdot d _ { i } ,\tag{11}
$$

$$
P ( t _ { \mathrm { t o t a l } } \leq T _ { \mathrm { d e a d } } ) \geq 1 - \epsilon ,\tag{12}
$$

where $U _ { i } ( d _ { i } )$ represents the utility of training that is expressed as the expected value of the probability of the total latency with dataset size $d _ { i }$ , α is a parameter to consider the tradeoff between the accuracy and latency, and ϵ is the acceptable probability of exceeding the deadline that depends on the requirement for a real-time application.

Furthermore, we proceed under the assumption that latency to aggregate models on the CS is negligible thanks to its stronger computation capability.

Algorithm 1 Joint client and data selection algorithm   
Input: A set of clients K, disjoint local dataset $\overline { { D _ { k } = D _ { k , 1 } \cup D _ { k , 2 } \cup } }$   
$\cdots \cup D _ { k , m }$ where $D _ { k , i }$ is the set of images of class i in client   
k, computational capability $a _ { k }$ , fluctuation $\mu _ { k }$ , and interruption   
parameters $\lambda _ { k }$ and $\mu _ { i n t r , k }$ and average number of times an image   
has been used for training $n _ { a v g , k }$ per client.   
Output: Subset of clients $l \in \breve { K } ,$ training dataset size per class   
$\mathsf { \bar { \Pi } } _ { \mathsf { T } D _ { k , i } | \forall k } \in { l , \forall i }$   
1: for all $k \in K$ do   
2: $\begin{array} { r } { | T D _ { k } |  \operatorname * { a r g m a x } _ { | T D | } ( | T D | \cdot P _ { k } ( t ^ { c p } + t ^ { i t r } + t ^ { c m } \leq T ^ { d e a d } ) ) } \end{array}$   
3: $| T D _ { k , i } | \forall i \gets \mathrm { a s s i g n E v e n } ( D _ { k } , | T D _ { k } | )$   
4: end for   
$5 \colon \underbrace { u _ { k } } _ { \ldots }  \underbrace { w _ { 1 } \cdot ( 1 - n _ { a v g , k } ) } _ { \ldots } | T D _ { k } | + w _ { 2 } \cdot \operatorname* { m i n } ( | T D _ { k , 1 } | , | T D _ { k , 2 } | , \cdots ) ,$   
$\forall k \in K$   
6: $l  \phi$   
7: $D _ { c u r } \gets \phi$   
8: while $\left| l \right| < n _ { t a r g e t }$ do   
9: $k \gets \operatorname* { m a x } _ { u _ { k } } ( \dot { K } )$   
10: $l  l \cup k$   
11: $n _ { a v g , k }  n _ { a v g , k } + \frac { | T D _ { k } | } { | D _ { k } | }$   
12: $T D _ { c u r } \gets T D _ { c u r } \cup \bar { T } \bar { D } _ { k }$   
13: $K \gets K - k$   
14: $i \gets \mathrm { a r g m i n } _ { i } ( | T D _ { c u r , i } | )$   
15: $c c _ { m } \gets \mathrm { c l a s s C o v e r a g e } ( T D _ { c u r } , m ) \forall m$   
16: $\begin{array} { r } { u _ { m } \xleftarrow { } \omega _ { 1 } \cdot \underbrace { e x p } _ { \texttt { r e } } ( - \bar { n } _ { a v g , m } / 1 0 ) \cdot | T D _ { m } | + w _ { 2 } \cdot | D _ { k , i } | + w _ { 3 } \cdot } \end{array}$   
$c c _ { m } , \forall m \in K$   
17: end while   
18: return $l , | T D _ { k , i } | \forall k \in l , \forall \mathrm { ~ i , ~ } n _ { a v g , k } \forall k \in K$

## IV. JOINT CLIENT AND DATA SELECTION ALGORITHM

Before describing the joint client and data selection algorithm in detail, we make the following assumptions supporting the logic of the algorithm.

Assumption 1: A larger training datasize contributes to faster convergence of the FL model accuracy.

Assumption 2: Using unseen data contributes to higher model accuracy over FL iterations. If a few clients are repeatedly selected due to higher computing capacity, their local data might be re-used for training. Many prior works do not have to consider this as the participating clients are probabilistically selected every FL round. As the proposed algorithm deterministically selects clients, it should avoid re-using the same data simply to maximize the training dataset size.

Assumption 3: Impact of unbalanced collective class distribution significantly reduces the test accuracy of underrepresented classes.

Based on the assumptions, we design the as described in Algorithm 1. We assume that the algorithm is running on the CS and is aware of the set of all clients $K .$ , sizes of local datasets and their class distribution on each client $| D _ { k , i } |$ where $k \in K$ and i is the class index, parameters characterizing the computing capacity $a _ { k }$ and $\mu _ { k } .$ , interruption parameters $\lambda _ { k }$ and $\mu _ { i n t r , k }$ , and $n _ { a v g , k }$ denoting the average number of times an image has been used for training over the FL rounds. The output of the algorithm is $l \subset K$ a set of clients selected for this FL iteration, $| T D _ { k , i } |$ number of images per class in the selected clients, and updated $n _ { a v g , k }$ for the next FL iteration.

In line 2, the algorithm determines the size of the training dataset, $| T D _ { k } |$ that maximizes the expected dataset size for all client k, i.e., dataset size multiplied by the probability the training finishes within the deadline. In line 3, training dataset sizes per class $| T D _ { k , i } |$ are derived by assignEven() function. The function is a heuristic for determining $| T D _ { k , i } |$ for all i such that the distribution is as even as possible while the sum is equal to $\begin{array} { r } { \left| T D _ { k } \right| = \sum _ { i } \left| T D _ { k , i } \right| } \end{array}$ . If there is enough data from all classes, $| T D _ { k , i } | = | T D _ { k } | / N _ { c l a s s e s } .$ If the number of images for certain classes is smaller than $| T D _ { k } | / N _ { c l a s s e s } ,$ then all images from the classes are used for training and the rest of the classes are assigned the same number of images such that the training dataset size is equal to $| T D _ { k } |$

Lines 8 to 16 describe the essence of the algorithm, where clients are greedily selected based on the usefulness metric $u _ { k }$ , which is defined as follows.

$$
u _ { k } \gets w _ { 1 } \cdot e x p ( - n _ { a v g , k } / 1 0 ) | T D _ { k } | + w _ { 2 } \cdot | D _ { k , i } | + w _ { 3 } \cdot c c _ { k } , \forall k \in K ,\tag{13}
$$

where $i = \mathrm { a r g m i n } _ { i } ( | T D _ { c u r , i } | )$ is the index of the class that has the least amount of training data allocated so far, $| T D _ { c u r } |$ in this FL round. The usefulness metric of a client k is a weighted sum of three elements, where the first is the training dataset size multiplied by a decreasing exponential function of the average number of times an image has been used in that client. The reasoning behind this is that the convergence of the FL model is faster when a larger training dataset is used (assumption 1), but still want to prevent using the data likely from high computing capacity devices over and over again. Eventually, $u _ { k }$ forces FL to select other clients through the term $n _ { a v g , k }$ . The second term is used to give preference to clients, which have images belong to the class that are in this FL round. This effectively balances the class distribution of the training dataset to be used in this FL round (assumption 2). The third term is a supplementary term ensuring the class coverage at the warm up phase of the loop (lines 8-16) before the second term starts to effectively work.

Until the target number of clients is reached, the client with largest $u _ { k }$ is selected in the while loop. The client k is added to the set l (line $1 0 ) , n _ { a v g , k }$ is updated (line 11) for the next FL round, and $T D _ { c u r }$ is updated with the newly selected client’s data (line 12), k is removed from candidate client set K in this FL round (line 13), the class with the fewest amount of images is identified (line 14), and finally, the usefulness metric is updated for every client still in K. Before the while loop at line 8, $T D _ { c u r }$ is empty, and the class with the least number of images cannot be determined. Therefore, the second term of $u _ { k }$ is instead initialized with the number of images from the class that has the fewest samples in each respective client (line 5).

The outputs l, the client selection, $| T D _ { k , i } | \forall k \ \in \ l , \forall i$ the training dataset are derived, and $n _ { a v g , k }$ will be used in subsequent FL iterations. Actual training dataset is selected randomly, i.e., random selection of $| T D _ { k , i } |$ images out of $D _ { k , i }$

## V. EXPERIMENTS

In this section, we present the experimental results and compare with existing works. The two baselines we consider are MinCost [22] and probPart based on [26], which can deal with non-iid data and computation heterogeneity. The algorithms in the work do not assume a fixed deadline like our work, but aim at minimizing the time required for global model convergence. Regarding MinCost algorithm, the sum of the training dataset across clients should be D, which is an input to the algorithm. The algorithm uses parameter α to address the non-iidness when assigning data. Higher value discourages clients with fewer number of classes to participate in the FL rounds. We used the value of 2.5 as it is in the middle of the range the work has suggested. As for probPart algorithm, the usefulness of a client under non-iid data is characterized in the $G _ { i }$ parameter. The values of $G _ { i }$ for client i have been calculated using a training on a small batch of the training data. Our version of probPart is simplified that selection probability is set just to be proportional to $G _ { i }$ , but preserves the essence of the algorithm.

Here, we evaluate the performance of our proposed FL scheduling algorithm on the CIFAR-100 dataset [27], which is used for evaluation and consists of 50,000 training and 10,000 test images for the experiments. In local training, we employ MobileNetV2 [29], a lightweight CNN architecture with depthwise separable convolutions, inverted residual blocks, batch normalization, and ReLU6 activation functions. This configuration results in a total of 3.5M parameters. The training utilizes the AdamW optimizer with weight decay and cross-entropy loss function. Additionally, we configure the federated system to operate without data compression.

## A. Performance of Data Allocation Algorithm

In this subsection, we present the convergence patterns of the aforementioned FL schemes. We assumed a hardware setup where five types of devices are participating in FL each having different computation capacities. For example, clients are assumed to have on average the computation capability of completing the training for 500 data points in 15 seconds with an 85% probability, where a coefficient of variation (CV) is 75%.

Model convergence speed and test accuracy: Fig. 4 shows the experimental results for the three FL scenarios where each client is assumed to have disjoint pre-allocated 1000 images. In order to generate non-iid data across clients, i.e., classes being unevenly distributed, we use label-skewed Dirichlet partitioning scheme often used in similar studies [30]. The parameter determining the degree of non-iidness, $\alpha ,$ is set to 0.3. We assume a total of 50 clients and for each FL round, the algorithm selects 10 clients. The number of local epochs inside an FL iteration is set to be 5. Deadline per FL iteration for the proposed algorithm is set to be 15 seconds. One FL iteration for the baseline algorithms may take longer than 15 seconds or 15 seconds depending on which clients are selected.

The two baseline algorithms are synchronous FL, but not deadline-based like the proposed algorithm, i.e., they wait for the slowest device to complete the training before model aggregation using FedAvg. The two baseline models require pre-defined training dataset sizes, which is set to 500. To make the comparison fair among the algorithms, we compare absolute times required for training not the number of FL iterations as shown in Figure 4.

![](images/a5ac43228b8eb94a6e0ab17e20766788662572f76e0a3ecd58005f8cef2b029a.jpg)  
Fig. 4. Test accuracy over time for the proposed, probPart and MinCost algorithms for CIFAR-100 dataset distributed across 50 clients. Shaded area denotes 1 std range over 6 experiments with random seeds for data distribution.

The proposed algorithm outperforms the two baselines i) because of the large training dataset it is capable of processing per unit time as our algorithm prefers computationally powerful devices as long as they are not used too many times as indicated by Equation (13), and ii) because of higher utilization as clients minimize the slack time by adjusting the training dataset size while the baseline algorithms inevitably wait for other clients to finish.

Moreover, compared to MinCost algorithm, the proposed method outperforms it as our method explicitly balances the class distribution in the FL round, where MinCost algorithm mostly looks at simply whether or not certain class is represented. The number of images from a class may be insufficient to ensure test accuracy of the class. This shows that client selection and resulting collective distribution of data is of paramount concern when it comes to convergence of the model. Compared to probPart algorithm, the performance is slightly better than MinCost algorithm, but a random selection of clients results do not guarantee every class being represented in each FL iteration, and catastrophic forgetting for the classes lowers the overall test accuracy.

To characterize the results beyond the aggregate accuracy, we examine how evenly the accuracy is distributed across the 100 classes, since a model that sacrifices under-represented classes is undesirable even at a similar mean accuracy. We quantify this with two standard, complementary measures of dispersion: the coefficient of variation (CV), i.e., the standard deviation of the per-class accuracies divided by their mean, which reflects how consistent the accuracy is across classes (lower is more uniform); and the Gini coefficient, which reflects how equally the accuracy is shared among classes (0 denotes perfect equality, while larger values indicate that accuracy is concentrated in fewer classes). We additionally report the tail-class accuracy, i.e., the mean accuracy of the 10 lowest-performing classes, which are the ones most exposed to forgetting. As summarized in Table II, the proposed method more than doubles the tail-class accuracy of probPart (15.3% versus 6.8%) and quadruples that of MinCost (3.5%), while attaining both the lowest CV and the lowest Gini. Hence, explicitly balancing the collective class distribution in each FL round keeps the accuracy both consistent (low CV) and equitably distributed (low Gini) across classes, so that the gain of the proposed algorithm is concentrated on the underrepresented classes that the baselines tend to forget.

PER-CLASS FAIRNESS ON THE CIFAR-100 TEST SET AT α = 0.3 FOR EACH METHOD’S FINAL TRAINED MODEL (SINGLE RUN): OVERALL ACCURACY, TAIL-CLASS ACCURACY (MEAN OF THE 10 LOWEST-ACCURACY CLASSES), AND THE DISPERSION OF THE PER-CLASS ACCURACIES (COEFFICIENT OF VARIATION AND GINI COEFFICIENT; LOWER IS MORE UNIFORM).
<table><tr><td>Method</td><td>Overall acc. (%)</td><td>Tail-10 acc. (%)</td><td>CV</td><td>Gini</td></tr><tr><td>Proposed (full)</td><td>42.3</td><td>15.3</td><td>0.43</td><td>0.24</td></tr><tr><td>probPart</td><td>38.0</td><td>6.8</td><td>0.49</td><td>0.28</td></tr><tr><td>MinCost</td><td>31.2</td><td>3.5</td><td>0.62</td><td>0.35</td></tr></table>

## B. Ablation and Sensitivity Analysis of the Usefulness Metric

To address the individual contribution of each component of the usefulness metric (Eq. 13), we conduct an ablation study on the weights $w _ { 1 } , \ w _ { 2 } ,$ and $w _ { 3 }$ . Since the $w _ { 1 }$ term is, by construction, a product of two distinct factors, i.e., a preference for a larger trainable dataset size $| T D _ { k } |$ and a freshness penalty $e x p ( - n _ { a v g , k } / 1 0 )$ that discourages data reuse, we further decouple these two sub-factors to isolate their individual effect. Each variant disables exactly one component while keeping the rest unchanged, and is evaluated under the same deadline-based, equal-time protocol as in Fig. 4, for two degrees of non-iidness $( \alpha = 0 . 3$ and a more extreme $\alpha = 0 . 1 )$

As shown in Fig. 5, the freshness penalty is the single most influential component: disabling it prevents the model from reaching a useful accuracy at both severities, plateauing well below every other variant, and the degradation deepens as the data becomes more skewed $( \alpha = 0 . 1 )$ . The remaining terms, i.e., the dataset-size factor, the class-balance term w<sub>2</sub>, and the coverage term $w _ { 3 } ,$ , each contribute a smaller and complementary gain, so that the full metric attains the highest accuracy. The ordering of the variants is preserved across $\alpha ~ = ~ 0 . 3$ and $\alpha ~ = ~ 0 . 1$ , indicating that no single weight dominates in a way that would make the metric fragile to its exact values, and that the design generalizes across the degree of non-iidness.

![](images/c3bda833073a7035ee57d4f53eefd575023673382b7989a7c583ca68bd0407c2.jpg)  
(a) α = 0.3

![](images/a39dec56b99c99b9242f8d7e199c875bdf3ee084078354c95fab2445bc7a317a.jpg)  
(b) α = 0.1  
Fig. 5. Component ablation of the usefulness metric (Eq. 13) as test accuracy over training time, under two degrees of non-iid severity: $\mathrm { ( a ) } \ \alpha = 0 . 3$ and (b) $\alpha = 0 . 1$ . Each variant disables one term of $u _ { k } ;$ for w , its two sub-factors (the freshness penalty and the client dataset-size preference) are disabled separately. Proposed keeps all terms, while probPart and MinCost are shown for reference. The experiments use the same client, model, and deadline configuration as Fig. 4, and are reported for a single random seed.

## VI. CONCLUSIONS

The paper proposed a joint client and training data selection algorithm for edge FL fast convergence of FL models. The work puts emphasis on estimating training times, which is largely impacted by the training dataset size, and proposes an algorithm, which balances training datasize, class distribution per FL iteration and data staleness. The experimental results display that our joint client selection and training data allocation algorithm achieves better FL convergence compared to other client and training dataset selection schemes by addressing the heterogeneity in computing capacities and noniid data more explicitly.

In future, we plan to bring the work closer to real world by considering more concrete interruption models for selected applications and investigating dynamic aspects of FL applications.

We also plan to extend the study of the usefulness metric beyond the per-term ablation presented here, by examining how different combinations of the weights $w _ { 1 } , \ w _ { 2 } .$ , and $w _ { 3 }$ jointly affect the accuracy, and by evaluating the algorithm under different client-pool distributions and selection ratios, as the current experiments consider a single setting of 10 out of 50 clients.

## ACKNOWLEDGMENTS

This work was partly supported by Einstein Center Digital Future, by Turkish Ministry of Education, and by JSPS KAKENHI Grant Number 23K16858.

## REFERENCES

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics. PMLR, 2017, pp. 1273– 1282.

[2] K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman, V. Ivanov, C. Kiddon, J. Konecnˇ y, S. Mazzocchi, B. McMahan\` et al., “Towards federated learning at scale: System design,” Proceedings of machine learning and systems, vol. 1, pp. 374–388, 2019.

[3] W. Y. B. Lim, N. C. Luong, D. T. Hoang, Y. Jiao, Y.-C. Liang, Q. Yang, D. Niyato, and C. Miao, “Federated learning in mobile edge networks: A comprehensive survey,” IEEE Communications Surveys & Tutorials, vol. 22, no. 3, pp. 2031–2063, 2020.

[4] R. Gu, C. Niu, F. Wu, G. Chen, C. Hu, C. Lyu, and Z. Wu, “From serverbased to client-based machine learning: A comprehensive survey,” ACM Computing Surveys (CSUR), vol. 54, no. 1, pp. 1–36, 2021.

[5] Y. Xiao, X. Zhang, Y. Li, G. Shi, M. Krunz, D. N. Nguyen, and D. T. Hoang, “Time-sensitive learning for heterogeneous federated edge intelligence,” IEEE Transactions on Mobile Computing, 2023.

[6] Y. Cui, K. Cao, J. Zhou, and T. Wei, “Helcfl: High-efficiency and lowcost federated learning in heterogeneous mobile-edge computing,” in 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 2022, pp. 1227–1232.

[7] A. Reisizadeh, I. Tziotis, H. Hassani, A. Mokhtari, and R. Pedarsani, “Straggler-resilient federated learning: Leveraging the interplay between statistical accuracy and system heterogeneity,” IEEE Journal on Selected Areas in Information Theory, vol. 3, no. 2, pp. 197–205, 2022.

[8] J. Zhao, R. Han, Y. Yang, B. Catterall, C. H. Liu, L. Y. Chen, R. Mortier, J. Crowcroft, and L. Wang, “Federated learning with heterogeneityaware probabilistic synchronous parallel on edge,” IEEE Transactions on Services Computing, vol. 15, no. 2, pp. 614–626, 2021.

[9] Z. Gao, A. Li, Y. Gao, B. Li, Y. Wang, and Y. Chen, “Fedswap: a federated learning based 5g decentralized dynamic spectrum access system,” in 2021 IEEE/ACM International Conference On Computer Aided Design (ICCAD). IEEE, 2021, pp. 1–6.

[10] L. Yu, R. Albelaihi, X. Sun, N. Ansari, and M. Devetsikiotis, “Jointly optimizing client selection and resource management in wireless feder-

ated learning for internet of things,” IEEE Internet of Things Journal, vol. 9, no. 6, pp. 4385–4395, 2021.

[11] L. Li, H. Xiong, Z. Guo, J. Wang, and C.-Z. Xu, “Smartpc: Hierarchical pace control in real-time federated learning system,” in 2019 IEEE Real-Time Systems Symposium (RTSS). IEEE, 2019, pp. 406–418.

[12] X. Ma, J. Zhu, Z. Lin, S. Chen, and Y. Qin, “A state-of-the-art survey on solving non-iid data in federated learning,” Future Generation Computer Systems, vol. 135, pp. 244–258, 2022.

[13] S. Wang, T. Tuor, T. Salonidis, K. K. Leung, C. Makaya, T. He, and K. Chan, “Adaptive federated learning in resource constrained edge computing systems,” IEEE journal on selected areas in communications, vol. 37, no. 6, pp. 1205–1221, 2019.

[14] M. M. Wadu, S. Samarakoon, and M. Bennis, “Federated learning under channel uncertainty: Joint client scheduling and resource allocation,” in 2020 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2020, pp. 1–6.

[15] ——, “Joint client scheduling and resource allocation under channel uncertainty in federated learning,” IEEE Transactions on Communications, vol. 69, no. 9, pp. 5962–5974, 2021.

[16] M. M. Amiri, D. Gund¨ uz, S. R. Kulkarni, and H. V. Poor, “Update aware¨ device scheduling for federated learning at the wireless edge,” in 2020 IEEE International Symposium on Information Theory (ISIT). IEEE, 2020, pp. 2598–2603.

[17] ——, “Convergence of update aware device scheduling for federated learning at the wireless edge,” IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 3643–3658, 2021.

[18] W. Xia, T. Q. Quek, K. Guo, W. Wen, H. H. Yang, and H. Zhu, “Multi-armed bandit-based client scheduling for federated learning,” IEEE Transactions on Wireless Communications, vol. 19, no. 11, pp. 7108–7123, 2020.

[19] B. Xu, W. Xia, J. Zhang, T. Q. Quek, and H. Zhu, “Online client scheduling for fast federated learning,” IEEE Wireless Communications Letters, vol. 10, no. 7, pp. 1434–1438, 2021.

[20] F. Lai, X. Zhu, H. V. Madhyastha, and M. Chowdhury, “Oort: Efficient federated learning via guided participant selection,” in USENIX Symposium on Operating Systems Design and Implementation (OSDI 21). USENIX Association, Jul. 2021, pp. 19–35. [Online]. Available: https://www.usenix.org/conference/osdi21/presentation/lai

[21] C. Wang, X. Wei, and P. Zhou, “Optimize scheduling of federated learning on battery-powered mobile devices,” in 2020 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2020, pp. 212–221.

[22] C. Wang, Y. Yang, and P. Zhou, “Towards efficient scheduling of federated mobile devices under computational and statistical heterogeneity,” IEEE Transactions on Parallel and Distributed Systems, vol. 32, no. 2, pp. 394–410, 2020.

[23] L. L. Pilla, “Optimal task assignment for heterogeneous federated learning devices,” in 2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2021, pp. 661–670.

[24] ——, “Scheduling algorithms for federated learning with minimal energy consumption,” IEEE Transactions on Parallel and Distributed Systems, vol. 34, no. 4, pp. 1215–1226, 2023.

[25] W. Shi, S. Zhou, and Z. Niu, “Device scheduling with fast convergence for wireless federated learning,” in ICC 2020-2020 IEEE International Conference on Communications (ICC). IEEE, 2020, pp. 1–6.

[26] X. Chen, X. Zhou, H. Zhang, M. Sun, and H. Vincent Poor, “Client selection for wireless federated learning with data and latency heterogeneity,” IEEE Internet of Things Journal, vol. 11, no. 19, pp. 32 183– 32 196, 2024.

[27] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[28] Y. Kim, F. Parterna, S. Tilak, and T. S. Rosing, “Smartphone analysis and optimization based on user activity recognition,” in IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2015, pp. 605–612.

[29] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4510–4520.

[30] T.-M. H. Hsu, H. Qi, and M. Brown, “Measuring the effects of nonidentical data distribution for federated visual classification,” arXiv preprint arXiv:1909.06335, 2019.