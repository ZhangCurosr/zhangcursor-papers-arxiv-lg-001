# A Novel Federated Learning Framework for Defending Against Gradient Inversion Attacks

Pengxin Guo<sup>a,1</sup>, Ziying Huang<sup>b,1</sup>, Shuang Zeng<sup>c</sup>, Xiaodan Zhang<sup>d</sup>, Weihong Ren<sup>e,f</sup>, Yuyin Zhou<sup>g</sup>, Liangqiong Qu<sup>a,∗</sup>

<sup>a</sup>School ofComputing and Data Science, The University ofHong Kong, Hong Kong, 999077, China <sup>b</sup>Department of Electrical and Electronic Engineering, Southern University of Science and Technology, Shenzhen, 518055, China

<sup>c</sup>Department of Mathematics, The University of Hong Kong, Hong Kong, 999077, China <sup>d</sup>College of Computer Science, Beijing University of Technology, Beijing, 100124, China <sup>e</sup>State Key Laboratory of Robotics and Systems, School of Mechanical Engineering and Automation, Harbin Institute ofTechnology, Shenzhen, 518055, China

<sup>f</sup>State Key Laboratory of Robotics, Shenyang Institute of Automation, Chinese Academy of Sciences, Shenyang, 110016, China

<sup>g</sup>Department of Computer Science and Engineering, University of California, Santa Cruz, CA, 95064, USA

## Abstract

Federated Learning (FL) ofers a paradigm for collaborative machine learning model training among clients, preserving data privacy by avoiding direct data sharing. However, the gradient information exchanged in FL has been shown to be vulnerable to Gradient Inversion Attacks (GIA), which can reconstruct sensitive user data. To counter these threats, various privacy-enhancing technologies like Secure Multi-party Computing (SMC), Homomorphic Encryption (HE), and Diferential Privacy (DP) have been integrated into FL. While efective, these methods often introduce significant trade-ofs between privacy and model utility. By revisiting the key to privacy exposure in FL under GIA, which lies in the frequent sharing of model gradients that contain private data, we take a new perspective by designing a novel privacy-preserving FL framework that efectively “breaks the direct connection” between the shared parameters and the local private data to defend against GIA. Our proposed framework, HyperFL++, leverages hypernetworks to generate local model parameters, where only the hypernetwork parameters are aggregated by the server. To ensure practicality, HyperFL++ incorporates

a Row-wise Kronecker Decomposition (RKD) to mitigate the memory and communication overhead from the hypernetwork. Comprehensive theoretical analysis and extensive experiments validate our method’s efectiveness in protecting data privacy while maintaining high model utility. Our code is available at https://github.com/Pengxin-Guo/HyperFL++.

Keywords:   
Federated Learning, Gradient Inversion Attacks, Privacy Protection, Hypernetworks,   
Kronecker Decomposition

## 1. Introduction

Deep Neural Networks (DNNs) have demonstrated significant success in various computer vision tasks, primarily due to access to extensive training datasets [1, 2, 3]. However, in many real-world scenarios, training data is often distributed among diferent institutions, and sharing this data is commonly restricted by privacy and regulatory issues. To address these challenges, Federated Learning (FL) [4, 5, 6, 7, 8], which utilizes collaborative and decentralized training of AI models across multiple clients while keeping personal data private, has emerged as an efective solution. In the FL paradigm, during each training round, clients perform local model updates using the latest copy of the model received from the server. Then, they send the updates to the server for aggregation. The server aggregates these updates to update the global model, which is then broadcasted to all clients for further training. By enabling collaborative training without raw data sharing, FL ensures data privacy protection.

Despite its privacy-centric design, FL is vulnerable to Gradient Inversion Attacks (GIA) [9, 10, 11, 12], which can reconstruct private training data from shared gradients. To mitigate this, defense mechanisms like Secure Multi-party Computing (SMC) [13, 14], Homomorphic Encryption (HE) [15, 16], and Diferential Privacy (DP) [17, 18, 19, 20] have been adopted. As shown in Figure 1 (left), these methods apply protections to the exchanged gradients without altering the fundamental FL framework. However, they introduce a significant privacy-utility dilemma.

For example, SMC and HE ofer strong security but impose prohibitive computational and communication overheads, especially for large-scale DNNs [14, 16]. Conversely, DP is more lightweight but often compromises model accuracy to achieve meaningful privacy guarantees [18, 21]. These limitations motivate us to explore a diferent path, leading to our central research question:

![](images/4bda9cd00660710bcbf4f915bb159ccaa51ea8bd3428455e5e91bbdd50da3bdf.jpg)  
Figure 1: Left. Current approaches focus on protective techniques for shared gradients, such as SMC, HE, and DP, which naturally entail significant privacy-utility trade-ofs. Right. A new FL framework is introduced that “breaks the direct connection” between the shared parameters and the local private data, aiming to achieve an optimal balance between privacy and utility.

Can we develop a new FL framework that provides a favorable privacy-utility trade-ofagainst GIA without depending on current defense mechanisms?

To achieve this, we revisit the key to privacy exposure in FL under GIA, which stems from the frequent sharing of model gradients containing private information. Most existing eforts focus on developing advanced defense mechanisms for the shared gradients to improve privacy in FL, as depicted in the left section of Figure 1. In contrast, our approach seeks to create a novel privacy-preserving FL framework that “breaks the direct connection” between shared parameters and local private data to guard against GIA. To accomplish this, we investigate the potential of hypernetworks—a type of deep neural network that generates weights for another network [22]—as a promising solution, as shown in the right section of Figure 1 and Figure 2.

To this end, we present HyperFL++, an innovative FL framework that employs a dual approach—network decomposition and hypernetwork sharing— to “break the direct connection” between shared parameters and local private data, as illustrated in Figure 2. Inspired by findings that FL models exhibit greater diversity in their final layers [23, 19], we partition each local model into a shared feature extractor and a private classifier. This design improves performance in heterogeneous settings and confines some data-sensitive components locally. To further enhance privacy, an auxiliary hypernetwork generates the feature extractor’s parameters from a private, client-specific embedding. Instead of sharing the feature extractor’s gradients, clients only transmit the hypernetwork’s parameters to the server for aggregation. This auxiliary hypernetwork sharing approach efectively “breaks the direct connection” between shared parameters and local private data, preserving privacy while facilitating interaction and information exchange among clients. To make HyperFL++ eficient, we introduce a Row-wise Kronecker Decomposition (RKD) technique that compresses the hypernetwork’s parameters, significantly reducing the associated memory and communication costs, as detailed in Figure 2 (right). This operation largely reduces the parameters associated with the hypernetwork, minimizing the communication overhead introduced by hypernetworks.

The HyperFL++ framework is designed to be both flexible and scalable. We propose two main configurations: (1) the Main Configuration, which generates the entire feature extractor and is suited for smaller networks, and (2) HyperFL++ for Large Pre-trained Models (HyperFL++-LPM), which uses a hypernetwork to generate parameters for lightweight adapters [24] to fine-tune large, fixed pre-trained models for more complex tasks (see Figure 3). Our theoretical and empirical results confirm that HyperFL++ robustly defends against GIA while delivering model performance and convergence rates comparable to the standard FedAvg [4]. We hope this work encourages a shift in focus toward designing inherently private FL frameworks rather than relying solely on external defense mechanisms.

Our main contributions are:

• We propose HyperFL++, a novel privacy-preserving FL framework that defends against GIA by “breaking the direct connection” between shared parameters and local data, ofering a new perspective on FL privacy.

• We introduce a Row-wise Kronecker Decomposition (RKD) method to make the hypernetwork-based approach practical by minimizing its communication and memory footprint.

• We present two key configurations of HyperFL++: (1) the Main Configuration, which is designed for simpler tasks with smaller networks and learns the full parameters of the feature extractor directly; (2) HyperFL++-LPM, which is aimed at complex tasks by utilizing pre-trained models as fixed feature extractors while generating trainable adapter parameters through a hypernetwork.

• Our theoretical analysis and extensive experimental results show that HyperFL++ provides strong privacy protection against GIA while achieving competitive model utility and convergence.

## 2. Related Work

## 2.1. Gradient Inversion Attacks

Gradient Inversion Attacks (GIA) are a class of privacy attacks that exploit the gradients shared in collaborative learning to reconstruct sensitive training data [9]. The underlying principle is that gradients, which guide model updates, inherently contain information about the data that produced them. The fundamental concept of GIA involves deliberately altering the dummy input to maximize the gradients related to the sensitive information of interest. By iteratively modifying the dummy input based on these gradients, an attacker can progressively approximate sensitive details, such as private attributes or specific training data samples that the model was trained on [25, 9, 10, 26].

Early work by Phong et al. demonstrated the feasibility of GIA on shallow networks [25]. Subsequent research extended these attacks to deeper models. For instance, Zhu et al. introduced a method combining pseudo-label estimation and optimization to enhance image reconstruction quality [9]. A significant advancement came from Geiping et al., who replaced the traditional L2 distance loss with cosine similarity, dramatically improving the fidelity of reconstructed images, even for complex datasets like ImageNet [10]. The continued success of these attacks highlights that shared gradients pose a considerable privacy threat, underscoring the need for robust defense mechanisms to prevent information leakage.

## 2.2. Privacy Protection in Federated Learning

While FL is designed to keep data localized, the necessity of sharing model updates opens an attack surface for GIA [10]. Consequently, researchers have integrated various privacy-preserving technologies into FL. These can be broadly categorized into three groups: cryptographic methods like Secure Multi-party Computing (SMC) [13, 14] and Homomorphic Encryption (HE) [15, 16], and perturbation-based methods like Diferential Privacy (DP) [17, 18, 19]. Cryptographic solutions ofer strong security guarantees by performing computations on encrypted data, but their high computational and communication overheads make them impractical for many large-scale deep learning applications [14, 16]. DP-based methods, which involve adding calibrated noise to gradients, are more lightweight but often lead to a degradation in model performance, creating a dificult trade-of between privacy and utility [18, 21].

Beyond these theoretically-grounded defenses, several empirical strategies have been proposed. These include gradient pruning or masking [9, 11], which zeroes out small-magnitude gradients, and unstructured noise addition [9]. Other notable methods are Soteria [27], which perturbs intermediate feature representations, and PRECODE [28], which uses a variational module to obscure latent features. However, as demonstrated in their respective studies, these empirical methods also struggle with the inherent privacy-utility trade-of [11, 28, 27]. Our work diverges from these approaches by proposing a new FL framework that redesigns the information-sharing mechanism itself, using hypernetworks to create a more favorable privacy-utility balance.

## 2.3. Hypernetworks in Federated Learning

A hypernetwork is a neural network that generates the weights for a separate, primary network, known as the target network [22, 29]. This concept has recently been applied to FL, primarily to address statistical heterogeneity by generating personalized models for each client [30, 31, 32]. For example, pFedHN [30] uses a central hypernetwork to generate client-specific parameters from learned client embeddings, enabling eficient personalization. Similarly, SemiPFL [31] employs a global hypernetwork to create personalized autoencoders for clients in a semi-supervised setting. A common thread in these methods is that the hypernetwork and client embeddings are managed by the central server. This design, while efective for personalization, gives the server complete knowledge of the local models, making it vulnerable to GIA, as our experiments confirm (see Table 1 and Figure 4).

![](images/50da5d2812806236ce2184e91a85e3de59f35e8c0cfd963c80b46e9c6a148137.jpg)  
Figure 2: Left). Illustration of HyperFL++. HyperFL++ divides each client’s network into two components: the feature extractor f(; θ<sub>i</sub>) and the classifier head g(; ϕ<sub>i</sub>). An auxiliary hypernetwork, denoted as $h ( ; \varphi _ { i } )$ is employed to create the feature extractor f(; θ<sub>i</sub>) using the client’s specific private embedding vector $\mathbf { v } _ { i } ,$ such that $\theta _ { i } = h ( \mathbf { v } _ { i } ; \varphi _ { i } )$ . These parameters are then utilized to derive features from the input $x _ { i } ,$ which are subsequently processed by the classifier to produce the output $\hat { y } _ { i } ,$ represented as $\hat { y } _ { i } = g ( f ( x _ { i } ; \theta _ { i } ) ; \phi _ { i } )$ . During the FL process, the hypernetwork $\varphi _ { i }$ is the only element shared with the server, while all other components remain private. This approach efectively addresses privacy leakage concerns. $R i g h t )$ . Row-wise Kronecker Decomposition (RKD). The hypernetwork parameters (matrices) are first partitioned into rows, and then Kronecker decomposition is applied to each row. ⊗ denotes Kronecker product.

Our work, HyperFL++, takes a diferent approach. Instead of focusing on personalization, we leverage the hypernetwork architecture as a privacy-preserving mechanism. By carefully designing what is shared (only the hypernetwork parameters) and what remains private (the classifier and client embeddings), we fundamentally alter the information available to the server, thereby minimizing the risk of data leakage while maintaining high model utility.

## 3. Methodology

In this section, we first formalize the problem setting of federated learning and then introduce our proposed HyperFL++ framework in detail.

## 3.1. Problem Formulation

Consider a federated learning system with m clients and a central server. The clients aim to collaboratively train their models without sharing their raw private data. Each client i possesses a local data distribution $P _ { X Y } ^ { ( i ) }$ over the input space X and label space $y ,$ , where there are K classes in total. The local model for client i is denoted by $\Theta _ { i }$ , and the associated loss function is $\ell : X \times y  \mathbb { R } _ { + }$ . The global optimization objective in FL is to find the set of model parameters $\Theta = \{ \Theta _ { 1 } , . . . , \Theta _ { m } \}$ that minimizes the average loss across all clients:

$$
\arg \operatorname* { m i n } _ { \Theta } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbb { E } _ { ( x , y ) \sim P _ { X Y } ^ { ( i ) } } \left[ \ell ( \Theta _ { i } ; x , y ) \right] .\tag{1}
$$

In the standard FedAvg algorithm [4], all clients share identical model parameters, i.e., $\Theta _ { 1 } = \cdots = \Theta _ { m }$ . In contrast, personalized FL allows for client-specific parameters, where $\Theta _ { i }$ can difer across clients to better handle data heterogeneity [23, 33].

Since the true data distributions are unknown, the objective in Eq. (1) is typically approximated using Empirical Risk Minimization (ERM). Each client i has a local dataset $\mathcal { D } _ { i } = \left\{ \left( x _ { i } ^ { l } , y _ { i } ^ { l } \right) \right\} _ { l = 1 } ^ { n _ { i } }$ of $n _ { i }$ IID samples drawn from $P _ { X Y } ^ { ( i ) }$ . The training objective thus becomes minimizing the empirical risk:

$$
\arg \operatorname* { m i n } _ { \Theta } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } _ { i } ( \Theta _ { i } ) ,\tag{2}
$$

where $\begin{array} { r } { \mathcal { L } _ { i } ( \Theta _ { i } ) = \frac { 1 } { n _ { i } } \sum _ { l = 1 } ^ { n _ { i } } \ell ( \Theta _ { i } ; x _ { i } ^ { l } , y _ { i } ^ { l } ) } \end{array}$ is the average loss computed on the local training data of client i.

## 3.2. Main Configuration ofHyperFL++

This section details the complete workflow of the HyperFL++ framework, followed by an explanation of the low-rank factorization technique applied to the hypernetwork.

## 3.2.1. HyperFL++ Framework

As depicted in Figure 2, in the Main Configuration of HyperFL++, each client i’s classification network, defined by $\Theta _ { i } = \{ \theta _ { i } , \phi _ { i } \}$ , is composed of a feature extractor $f : X \to \mathbb { R } ^ { d }$ with parameters $\theta _ { i }$ and a classifier $g : \mathbb { R } ^ { d }  \mathbb { R } ^ { K }$ with parameters $\phi _ { i }$ . Here, d is the feature dimension and $K$ is the number of classes. Each client also has a private embedding $\mathbf { v } _ { i }$ and a hypernetwork $h$ parameterized by $\varphi _ { i }$ . The hypernetwork generates the parameters for the feature extractor, i.e., $\theta _ { i } = h ( \mathbf { v } _ { i } ; \varphi _ { i } )$ . This design enables the creation of personalized feature extractors based on unique client embeddings, which can be either fixed or trainable; in this work, we use trainable vectors. The optimization objective from Eq. (2) is then reformulated as:

$$
\arg \operatorname* { m i n } _ { \varphi , \phi , \mathbf { v } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } _ { i } ( h ( \mathbf { v } _ { i } ; \varphi _ { i } ) , \phi _ { i } ) ,\tag{3}
$$

where $\varphi = \{ \varphi _ { 1 } , \ldots , \varphi _ { m } \} , \phi = \{ \phi _ { 1 } , \ldots , \phi _ { m } \}$ , and $\mathbf { v } = \{ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { m } \}$ . Note that the feature extractor parameters are generated by the hypernetwork and are not directly trained, whereas the client embeddings, hypernetwork parameters, and classifier parameters are all trainable.

To “break the direct connection” between shared parameters and private local data, thereby defending against GIA while maintaining high performance, clients only upload their hypernetwork parameters to the server for aggregation. The classifier and the private client embedding remain local. As shown in Figure 2, in each communication round, the server broadcasts the aggregated hypernetwork parameters $\bar { \varphi }$ to the clients. Each client i updates its local hypernetwork parameters $\varphi _ { i }$ with $\bar { \varphi }$ and proceeds with local training. After local training, each client sends its updated $\varphi _ { i }$ back to the server for aggregation, while keeping its classifier parameters $\phi _ { i }$ and client embedding $\mathbf { v } _ { i }$ private. The server then aggregates the received $\varphi _ { i }$ to compute the new global $\bar { \varphi }$ for the next round. This process facilitates information sharing among clients while preserving privacy.

Local Training Procedure. During each local training phase, a client first updates its hypernetwork parameters $\varphi _ { i }$ with the aggregated parameters $\bar { \varphi }$ from the server. Then, it performs iterative updates using stochastic gradient descent as follows:

• Step 1: Update Classifier $\phi _ { i }$ . With $\varphi _ { i }$ and $\mathbf { v } _ { i }$ fixed, the classifier parameters $\phi _ { i }$ are trained for one epoch:

$$
\phi _ { i } \gets \phi _ { i } - \eta _ { g } \nabla _ { \phi _ { i } } \ell \left( h ( \mathbf { v _ { i } } ; \varphi _ { i } ) , \phi _ { i } ; \xi _ { i } \right) ,\tag{4}
$$

where $\xi _ { i }$ is a mini-batch of data and $\eta _ { g }$ is the learning rate for the classifier.

• Step 2: Update Hypernetwork $\varphi _ { i }$ and Embedding $\mathbf { v } _ { i } .$ After updating the classifier, the client updates the hypernetwork parameters $\varphi _ { i }$ and the client embedding $\mathbf { v } _ { i }$ for multiple epochs with the new $\phi _ { i }$ fixed:

$$
\begin{array} { r l } & { \varphi _ { i } \gets \varphi _ { i } - \eta _ { h } \nabla _ { \varphi _ { i } } \ell \left( h ( \mathbf { v _ { i } } ; \varphi _ { i } ) , \phi _ { i } ; \xi _ { i } \right) } \\ & { } \\ & { \mathbf { v } _ { i } \gets \mathbf { v } _ { i } - \eta _ { \nu } \nabla _ { \mathbf { v } _ { i } } \ell \left( h ( \mathbf { v _ { i } } ; \varphi _ { i } ) , \phi _ { i } ; \xi _ { i } \right) , } \end{array}\tag{5}
$$

where $\eta _ { h }$ and $\eta _ { \nu }$ are the learning rates for the hypernetwork and the client embedding, respectively.

Global Aggregation. Following standard FL protocols like FedAvg [4], the server aggregates the hypernetwork parameters from all clients via weighted averaging:

$$
\bar { \varphi } = \sum _ { i = 1 } ^ { m } w _ { i } \varphi _ { i } ,\tag{6}
$$

where $\begin{array} { r } { w _ { i } = \frac { n _ { i } } { \sum _ { j = 1 } ^ { m } n _ { j } } } \end{array}$ is the aggregation weight for client i, typically proportional to its local data size.

However, a key challenge in HyperFL++ is the increased communication overhead, as the hypernetwork’s parameters are generally larger than those of the feature extractor it generates [22]. For instance, if a convolutional layer’s feature extractor parameters are $\theta _ { l } \in \mathbb { R } ^ { \mathrm { I C \times O C \times K S \times K S } }$ , the hypernetwork generating them would have parameters $\varphi _ { l } \in$ R<sup>k×(IC×OC×KS×KS)</sup>, where k is the hypernetwork’s hidden dimension. This results in a k-fold increase in parameter size. To address this and make HyperFL++ practical, we introduce a low-rank factorization method to compress the hypernetwork and reduce its computational and memory footprint.

## 3.2.2. Row-wise Kronecker Decomposition (RKD)

A primary challenge in employing hypernetworks is their potential size. The hypernetwork must generate all the parameters for the target network’s feature extractor, which can lead to a substantial number of parameters within the hypernetwork itself. This increase in model size translates directly to higher communication overhead and memory requirements, potentially undermining the practicality of the framework. To overcome this critical bottleneck and make HyperFL++ eficient, we introduce a powerful compression technique: Row-wise Kronecker Decomposition (RKD).

RKD is a specialized low-rank factorization method designed to compress the large parameter matrices of the hypernetwork. The core idea is to represent a large matrix as a composition of much smaller ones, thereby reducing the number of parameters that need to be stored and communicated. As illustrated in the right panel of Figure 2, the process works as follows:

Let the parameters for a single layer of the hypernetwork be represented by a large matrix $\varphi \in \mathbb { R } ^ { k \times N }$ , where k is the dimension of the hypernetwork’s hidden layer and N is the total number of parameters in the target feature extractor layer $( \mathbf { e } . \mathbf { g } . , \ N \ =$ $\mathbf { I C } \times \mathbf { O C } \times \mathbf { K S } \times \mathbf { K S }$ for a convolutional layer). Instead of storing the full φ matrix, we decompose it into two smaller factor matrices, $A \in \mathbb { R } ^ { k \times m }$ and $B \in \mathbb { R } ^ { k \times n }$ , where the dimensions are chosen such that $m \times n = N$

The “row-wise” aspect of RKD means we reconstruct each row of the original matrix $\varphi$ independently. Specifically, the r-th row of $\varphi$ is reconstructed by taking the Kronecker product of the r-th row of A and the r-th row of B:

$$
\varphi _ { r ; \cdot } = A _ { r ; \cdot } \otimes B _ { r ; \cdot } , \quad \mathrm { f o r } r = 1 , \ldots , k ,\tag{7}
$$

where ⊗ denotes the Kronecker product [34]. During training, only the factor matrices A and B are stored and updated.

The compression benefit of this decomposition is significant. The original number of parameters to store or communicate for the matrix $\varphi$ is $k \times N = k \times m n$ . With RKD, we only need to handle the parameters of A and B, which total $k { \times } m + k { \times } n = k { \times } ( m + n )$ . By choosing the factors m and n to be approximately equal (i.e., $m \approx n \approx { \sqrt { N } } )$ , the parameter complexity is dramatically reduced from O(kN) to $O ( k { \sqrt { N } } )$ . This substantial compression makes the hypernetwork-based approach computationally feasible and communication-eficient, enabling HyperFL++ to achieve a favorable balance between privacy, utility, and system performance.

![](images/ba5ea12bd9416fc69fe42a0d2abde888a59aa0bf332e10c877bc94e1bad56783.jpg)  
Figure 3: The HyperFL++-LPM framework in each client. In this framework, the weights of the pretrained model remain fixed, while only the classifier, hypernetwork, and client embedding are trainable. It’s important to note that θ refers to the parameters of the adapters

## 3.3. HyperFL++ for Large Pre-trained Models (HyperFL++-LPM)

While the main configuration of HyperFL++ is efective for moderately sized networks, generating the entire feature extractor for very large-scale models, such as Vision Transformers (ViTs) [1] or other foundation models, can become impractical due to the sheer number of parameters. The hypernetwork required to generate such a large target network would itself be enormous, leading to prohibitive computational and memory costs.

To address this scalability challenge and extend the applicability of our framework to state-of-the-art large models, we introduce an adaptation named HyperFL++-LPM (for Large Pre-trained Models). This configuration, illustrated in Figure 3, leverages the power of pre-trained models by integrating them with parameter-eficient finetuning (PEFT) techniques, such as adapters [24].

The core idea behind HyperFL++-LPM is to keep the vast majority of the pretrained model’s parameters frozen and only fine-tune a small set of additional parameters. Instead of generating the entire feature extractor, the hypernetwork’s role is shifted to generating only the parameters for these lightweight adapter modules. These adapters are small neural network layers inserted within the architecture of the larger pre-trained model. By training only these adapters, we can adapt the powerful, generalpurpose features of the large model to specific downstream tasks and client data distributions without incurring the cost of full model fine-tuning.

In this setup, the client model consists of a large, fixed pre-trained feature extractor, the trainable adapters whose parameters are generated by the hypernetwork, and a private classifier. The training process remains similar to the main configuration: clients update their local hypernetwork, client embedding, and classifier, but the updates to the feature extractor are now confined to the adapter layers. This approach allows HyperFL++ to efectively harness the capabilities of powerful pre-trained models for complex tasks while maintaining the core privacy and eficiency benefits of our framework. It provides a practical and scalable solution for applying privacy-preserving federated learning in the era of large foundation models.

## 4. Analysis and Insights

## 4.1. Privacy Analysis

In this section, we conduct a detailed privacy analysis of the HyperFL++ framework. We consider the common scenario where the server is an honest-but-curious adversary, meaning it adheres to the training protocol but attempts to infer private client data from the shared model updates [9, 10, 26, 12].

## 4.1.1. Background on Gradient Inversion Attacks

As established by prior research [25, 9, 10], it is possible to reconstruct input data from gradients. This is achieved by solving an optimization problem. Given a model with parameters Θ and the gradients $\nabla _ { \Theta } \mathcal { L } _ { \Theta } ( x ^ { * } , y ^ { * } )$ computed from a private data batch $( x ^ { * } , y ^ { * } )$ , an attacker aims to find a reconstructed input x that approximates x<sup>∗</sup>:

$$
\arg \operatorname* { m i n } _ { \boldsymbol { x } } \mathcal { L } _ { \mathrm { g r a d } } \left( \boldsymbol { x ; \Theta } , \nabla _ { \Theta } \mathcal { L } _ { \Theta } \left( \boldsymbol { x } ^ { * } , \boldsymbol { y } ^ { * } \right) \right) + \alpha \mathcal { R } _ { \mathrm { a u x } } \left( \boldsymbol { x } \right) .\tag{8}
$$

Here, $\mathcal { L } _ { \mathrm { g r a d } }$ is a loss function that measures the similarity between the gradients of the reconstructed data x and the target gradients $\nabla _ { \Theta } \mathcal { L } _ { \Theta } ( x ^ { * } , y ^ { * } )$ . The term $\mathcal { R } _ { \mathrm { a u x } } \left( x \right)$ is a regularization component that incorporates prior knowledge about natural images to improve reconstruction quality, with α as its weighting factor. Diferent GIA methods vary in their choice of the gradient loss and regularization terms [9, 10, 26].

## 4.1.2. Analysis ofthe Proposed Defense Mechanism

In contrast to standard FL frameworks where clients upload gradients of the entire model, HyperFL++ requires clients to share only the hypernetwork parameters. Consequently, an attacker’s objective function for reconstructing data x must be adapted:

$$
\arg \operatorname* { m i n } _ { \boldsymbol { x } } \mathcal { L } _ { \mathrm { g r a d } } \left( \boldsymbol { x ; \varphi } , \nabla _ { \varphi } \mathcal { L } _ { \varphi , \phi } \left( \boldsymbol { x } ^ { * } , \boldsymbol { y } ^ { * } , \mathbf { v } ^ { * } \right) \right) + \alpha \mathcal { R } _ { \mathrm { a u x } } \left( \boldsymbol { x } \right) ,\tag{9}
$$

where $\varphi$ are the hypernetwork parameters, $\phi$ are the classifier parameters, and $\mathbf { v } ^ { * }$ is the private client embedding. The crucial diference between Eq. (8) and Eq. (9) is that the attacker only has access to the gradients of the hypernetwork, not the entire model.

To reconstruct the input $x ,$ the attacker must first gain information about the feature extractor, whose parameters are generated by the hypernetwork based on the client embedding. This turns the attack into a challenging bi-level optimization problem:

$$
\begin{array} { r } { \begin{array} { c } { \arg \underset { x } { \operatorname* { m i n } } \mathcal { L } _ { \mathrm { g r a d } } \left( x , \hat { \mathbf { v } } ; \varphi , \Delta _ { \theta } \right) + \alpha \mathcal { R } _ { \mathrm { a u x } } \left( x \right) } \\ { \mathrm { s . t . } \hat { \mathbf { v } } = \arg \underset { \mathbf { v } } { \operatorname* { m i n } } \mathcal { L } _ { \mathrm { g r a d } } \left( x , \mathbf { v } ; \varphi , \nabla _ { \varphi } \mathcal { L } _ { \varphi , \phi } \left( x ^ { * } , y ^ { * } , \mathbf { v } ^ { * } \right) \right) , } \end{array} } \end{array}\tag{10}
$$

where the lower-level problem aims to recover the client embedding $\mathbf { v } ^ { * }$ . However, solving this is infeasible. According to the chain rule, computing the hypernetwork’s gradient $\nabla _ { \varphi } \mathcal { L }$ requires the classifier’s gradient $\nabla _ { \phi } \mathcal { L } .$ . Since the classifier is kept private on the client’s device and never shared, the server cannot compute this gradient. This efectively breaks the inversion process.

Even if an attacker attempts to bypass the classifier by directly optimizing for the hypernetwork’s output (the feature extractor parameters $\theta )$ , the problem remains intractable. Unlike inferring a low-dimensional label vector [35, 26], the output θ is a high-dimensional and complex set of parameters. Simultaneously optimizing for both the input data x and the high-dimensional parameters $\theta$ creates an enormous search space, making a successful reconstruction practically impossible [9, 11].

In summary, HyperFL++’s architecture, which combines a hypernetwork with a private classifier and client embedding, creates a strong defense against GIA by fundamentally disrupting the information flow required for a successful attack. This is further validated by our experimental results in Section 5.2.

## 4.2. Utility Analysis

We now analyze the utility of the HyperFL++ framework, focusing on its convergence properties and generalization bound. Detailed proofs are available in the Appendix.

## 4.2.1. Convergence Analysis

We adopt standard assumptions from the FL literature [36, 33] for smooth, nonconvex optimization landscapes.

Assumption 1. (L-smoothness) The local loss functions $\mathcal { L } _ { 1 } , \mathcal { L } _ { 2 } , \cdots , \mathcal { L } _ { m }$ are all $L _ { - }$ smooth, meaning that for any pairs $( \phi _ { j } , \varphi _ { j } , \mathbf { v } _ { j } )$ and $( \phi _ { k } , \varphi _ { k } , \mathbf { v } _ { k } )$ , the following holds: $\begin{array} { r } { \mathcal { L } _ { i } ( h ( \mathbf { v } _ { k } , \varphi _ { k } ) , \phi _ { k } ) \leq \mathcal { L } _ { i } ( h ( \mathbf { v } _ { j } , \varphi _ { j } ) , \phi _ { j } ) + ( ( \phi _ { k } , \varphi _ { k } , \mathbf { v } _ { k } ) - ( \phi _ { j } , \varphi _ { j } , \mathbf { v } _ { j } ) ) \nabla \mathcal { L } _ { i } ( h ( \mathbf { v } _ { j } , \varphi _ { j } ) , \phi _ { j } ) + \phi _ { k } ( \mathbf { v } _ { k } , \varphi _ { k } ) , } \end{array}$ $\begin{array} { r } { \frac { L } { 2 } \| ( \phi _ { k } , \varphi _ { k } , \mathbf { v } _ { k } ) - ( \phi _ { j } , \varphi _ { j } , \mathbf { v } _ { j } ) \| _ { 2 } ^ { 2 } . } \end{array}$

Assumption 2. (Bounded Variance) Let $\xi _ { i } ^ { t }$ be sampled uniformly at random from the local data of the i-th client at the t-th training step. The variance of the stochastic gradientsfor each client and each variable is bounded asfollows:

$$
\begin{array} { r l } & { \mathbb { E } \left\| \nabla _ { \phi } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t } , \xi _ { i } ^ { t } \right) - \nabla _ { \phi } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t } \right) \right\| ^ { 2 } \leq \sigma _ { i } ^ { 2 } , } \\ & { \mathbb { E } \left\| \nabla _ { \varphi } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } , \xi _ { i } ^ { t } \right) - \nabla _ { \varphi } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } \right) \right\| ^ { 2 } \leq \sigma _ { i } ^ { 2 } , } \\ & { \mathbb { E } \left\| \nabla _ { \mathbf { v } } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } , \xi _ { i } ^ { t } \right) - \nabla _ { \mathbf { v } } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } \right) \right\| ^ { 2 } \leq \sigma _ { i } ^ { 2 } , } \end{array}
$$

for $i = 1 , \cdots , m .$

Assumption 3. (Bounded Gradient) The expected squared norm of the stochastic gradients is uniformly bounded, specifically:

$$
\mathbb { E } \left\| \nabla _ { \phi } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t } , \xi _ { i } ^ { t } \right) \right\| ^ { 2 } \leq G ^ { 2 } ,
$$

$$
\mathbb { E } \left\| \nabla _ { \varphi } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } , \xi _ { i } ^ { t } \right) \right\| ^ { 2 } \leq G ^ { 2 } ,
$$

$$
\begin{array} { r } { \mathbb { E } \left\| \nabla _ { \mathbf { v } } L _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \boldsymbol { \varphi } _ { i } ^ { t } \right) , \boldsymbol { \phi } _ { i } ^ { t + 1 } , \boldsymbol { \xi } _ { i } ^ { t } \right) \right\| ^ { 2 } \leq G ^ { 2 } , } \end{array}
$$

for all $i = 1 , \cdots , m$ and $t = 0 , \cdots , T - 1$ . Here, $T$ represents the total number of training stepsfor each client.

Under these assumptions, we establish the convergence rate for HyperFL++.

Theorem 1. Assuming that Assumptions 1, 2 and 3 are satisfied, and defining L, M, $\sigma _ { i } ,$ G as stated, let $\begin{array} { r } { \eta _ { \mathrm { m i n } } = \operatorname* { m i n } \left\{ \eta _ { g } , \frac { 1 } { 2 } \eta _ { h } , \eta _ { \nu } \right\} } \end{array}$ and E represent the number of local training iterations between two communication rounds. Then we have:

$$
\frac { 1 } { m T } \sum _ { i = 1 } ^ { m } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left. \nabla \mathcal { L } _ { i } ^ { t } \right. ^ { 2 } \right] \leq 2 \sqrt { \frac { L M G ^ { 2 } D } { 2 T } } ,\tag{11}
$$

where $\mathcal { L } _ { i } ^ { 0 } - \mathcal { L } _ { i } ^ { * } \leq D .$ ∀i, and $\begin{array} { r } { \eta _ { g } ^ { 2 } + \eta _ { \nu } ^ { 2 } + ( E - 1 ) \eta _ { h } ^ { 2 } + \frac { E - 1 } { L } \eta _ { h } \leq M \eta _ { m i n } ^ { 2 } . } \end{array}$

Based on Theorem 1, we achieve a convergence rate of $\begin{array} { r } { O ( \frac { 1 } { \sqrt { T } } ) } \end{array}$ towards the stationary solution under smooth and non-convex conditions. This rate is comparable to that of FedAvg in non-convex situations [37]. Additionally, we can accelerate convergence for Polyak-Lojasiewicz (PL) functions [38], which are frequently encountered in non-convex optimization problems.

Assumption 4. (PL Condition) A function f is referred to as a $\mu { - } P L$ function if there exists a constant $\mu > 0$ such that it satisfies the condition:

$$
\| \nabla f ( x ) \| ^ { 2 } \geq 2 \mu \Big ( f ( x ) - \operatorname* { i n f } _ { x ^ { \prime } } f ( x ^ { \prime } ) \Big ) , \quad \forall x .
$$

We assume that allfunctions $\mathcal { L } _ { 1 } , \cdots , \mathcal { L } _ { m }$ are $\mu { - } P L$ functions, and we denote in $\dot { \mathbf { \rho } } _ { x ^ { \prime } } f ( x ^ { \prime } )$ simply as $f ^ { * }$

Corollary 1. Given the assumptions along with $\eta _ { m i n } ,$ L, M and D defined in Theorem $^ { l , }$ and under the additional Assumption 4, we have:

$$
\mathbb { E } \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } ; \boldsymbol { \varphi } _ { i } ^ { t } \right) , \boldsymbol { \phi } _ { i } ^ { t } \right) \right] - \mathcal { L } ^ { * } \leq \left( 1 - 2 \eta _ { \operatorname* { m i n } } \mu \right) ^ { t + 1 } D + \eta _ { \operatorname* { m i n } } \frac { L M G ^ { 2 } } { 4 \mu } .\tag{12}
$$

If we set $\begin{array} { r } { \eta _ { \mathrm { m i n } } \leq \frac { \mu \epsilon } { L M G ^ { 2 } } } \end{array}$ , after $O \left( \textstyle { \frac { 1 } { \epsilon } } \log \left( \textstyle { \frac { 1 } { \epsilon } } \right) \right)$ steps, we have that:

$$
\mathbb { E } \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } ; \boldsymbol { \varphi } _ { i } ^ { t } \right) , \boldsymbol { \phi } _ { i } ^ { t } \right) \right] - \mathcal { L } ^ { * } \leq \epsilon .\tag{13}
$$

When using PL functions, the convergence rate of HyperFL++ surpasses that obtained through smoothness assumptions alone.

## 4.2.2. Generalization Bound

We also provide a generalization bound for HyperFL++ based on the framework from [39].

Assumption 5. (Lipschitz Condition) We assume that the weights ofhypernetworks φ , the client embeddings $\mathbf { v } _ { i }$ and the classifier weights $\phi _ { i }$ are bounded in a ball of radius R, in which thefollowing Lipschitz conditions hold:

$$
\begin{array} { r } { \left| \mathcal { L } _ { i } ( h ( \varphi _ { i } ; \mathbf { v } _ { i } ) , \phi _ { i } ) - \mathcal { L } _ { i } ( h ( \varphi _ { i } ; \mathbf { v } _ { i } ) , \phi _ { i } ^ { * } ) \right| \leq L _ { \phi } | | \phi _ { i } - \phi _ { i } ^ { * } | | , } \end{array}
$$

$$
\begin{array} { r } { | \mathcal { L } _ { i } ( h ( \varphi _ { i } ; \mathbf { v } _ { i } ) , \phi _ { i } ) - \mathcal { L } _ { i } ( h ^ { * } ( \varphi _ { i } ; \mathbf { v } _ { i } ) , \phi _ { i } ) | \le L _ { h } \| h _ { i } - h _ { i } ^ { * } \| , } \end{array}
$$

$$
\left| | h ( \varphi _ { i } ^ { * } ; \mathbf { v } _ { i } ) - h ( \varphi _ { i } ; \mathbf { v } _ { i } ) ) \right| | \leq L _ { \varphi } \| \varphi _ { i } - \varphi _ { i } ^ { * } \| ,
$$

$$
\left| | h ( \varphi _ { i } ; \mathbf { v } _ { i } ^ { * } ) - h ( \varphi _ { i } ; \mathbf { v } _ { i } ) ) \right| | \leq L _ { \mathbf { v } } \| \mathbf { v } _ { i } - \mathbf { v } _ { i } ^ { * } \| ,
$$

for i = 1<sub>,</sub> · · · <sub>,</sub> m.

Theorem 2. Assuming we select m clients during each FL communication round, let the hypernetwork parameter space be of dimension H, the embedding space be of dimension d, and the classifier parameter space be of dimension K. The parameters ϕ, v, φ are those learned from each client’s individual dataset. Under the condition of Assumption 5, we can derive:

$$
S = O \Big ( \frac { d + H + K } { \epsilon ^ { 2 } } \log \Big ( \frac { R ( L _ { h } L _ { \varphi } + L _ { h } L _ { \nu } + L _ { \phi } ) } { \epsilon } \Big ) + \frac { 1 } { m \epsilon ^ { 2 } } \log \frac { 1 } { \delta } \Big ) ,\tag{14}
$$

such that if the number of samples per client exceeds S , then with a probability of at least 1 − δ, for all ϕ, v, φ, the following holds:

$$
| \sum _ { i } ^ { m } \frac { n _ { i } } { N } ( \mathcal { L } _ { i } ( h ( \varphi _ { i } ^ { * } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ^ { * } ) - \mathcal { L } _ { i } ( h ( \varphi _ { i } ; \mathbf { v } _ { i } ) , \phi _ { i } ) ) | \leq \epsilon ,\tag{15}
$$

where $\phi ^ { * } , \ \mathbf { v } ^ { * } , \ \varphi ^ { * }$ are the optimal parameters corresponding to the distribution of each individual client.

Theorem 2 shows that the required sample size depends on the complexity of the parameter spaces and the number of clients. This highlights a trade-of between the generalization ability of the framework and the expressive power of the personalized models.

## 5. Experiments

In this section, we conduct a comprehensive experimental evaluation of the proposed HyperFL++ framework, assessing its efectiveness in terms of both privacy pro-

tection and model utility.

## 5.1. Experimental Setup

## 5.1.1. Datasets

Our evaluation is performed on four widely-used image classification benchmarks: (1) EMNIST [40], a dataset of handwritten characters across 62 classes. (2) Fashion-MNIST [41], comprising 10 categories of fashion product images. (3) CIFAR-10 [42], a standard 10-class object recognition dataset. (4) CINIC-10 [43], a larger 10-class dataset combining images from CIFAR-10 and ImageNet.

To simulate realistic federated learning scenarios, we follow the methodology in [23, 44] to create a non-IID data distribution. For each client, a small fraction of data (s%, set to 20%) is sampled uniformly from all classes, while the majority ((100− s)%) is drawn from a limited set of dominant classes. Clients are organized into groups that share the same dominant classes. For Fashion-MNIST, CIFAR-10, and CINIC-10, we create 5 client groups, with each group assigned three consecutive classes (e.g., 0-2, 2-4, etc.). For EMNIST, we form 3 groups corresponding to digits, uppercase letters, and lowercase letters.

## 5.1.2. Model Architectures

For the Main Configuration of HyperFL++, we adopt the CNN architectures from [23, 44]. A smaller CNN with two convolutional layers is used for EMNIST and Fashion-MNIST, while a larger CNN with three convolutional layers is used for CIFAR-10 and CINIC-10. The hypernetwork is a fully connected network with one hidden layer, generating the weights for the target CNNs. Client embeddings are learnable vectors of dimension 128.

For HyperFL++-LPM, we use a pre-trained ViT-S model [1] as the fixed feature extractor. Adapters, consisting of down-projection and up-projection layers with a ReLU activation, are inserted into each transformer block. The hypernetwork, with the same architecture as in the main configuration, generates the parameters for these adapters.

## 5.1.3. Compared Methods

We benchmark HyperFL++ against a range of baselines. For the Main Configuration of HyperFL++, the baselines include: (1) FedAvg [4], the standard FL algorithm. (2) pFedHN and pFedHN-HC [30], which use a central hypernetwork for personalization. (3) Several DP-based methods: DP-FedAvg [21], PPSGD [45], and CENTAUR [19]. Note that these methods were originally designed for user-level DP; we adapt them to the honest-but-curious server setting by applying DP protections before server aggregation. (4) HyperFL [44], the precursor to our current work.

For HyperFL++-LPM, we compare against FedAvg with two fine-tuning strategies: (1) full parameter tuning and (2) adapter-based fine-tuning.

## 5.1.4. Training Settings

For the Main Configuration of HyperFL++, we use mini-batch SGD as the local optimizer for all methods. The learning rates $\eta _ { h }$ and η<sub>v</sub> are set to 0.01 for EMNIST/Fashion-MNIST and 0.02 for CIFAR-10/CINIC-10, while $\eta _ { g }$ is 0.1 for all datasets. We use a weight decay of 5e-4, momentum of 0.5, and a batch size of 50 (100 for EMNIST). All experiments run for 200 global communication rounds with 5 local epochs. We test with both 20 clients (full participation) and 100 clients (30% random sampling). For DP-based methods, we use a privacy budget of (ϵ = 4 σ = 1e − 5).

For HyperFL++-LPM, we use 20 clients and a batch size of 16. The learning rates $\eta _ { h }$ and η<sub>v</sub> are 0.02 for EMNIST and 0.1 for CIFAR-10.

All experiments are conducted on NVIDIA GeForce RTX 4090 and 3090 GPUs.

## 5.2. Privacy Evaluation

We experimentally validate the privacy-preserving capabilities of HyperFL++, complementing the theoretical analysis in Section 4.1. We use the EMNIST and CIFAR-10 datasets for this evaluation, attempting to reconstruct a subset of 50 images (with a batch size of one) using the well-known IG attack method [10]. The attack is run for 10,000 optimization iterations. We measure the reconstruction quality using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). Better defense is indicated by lower PSNR/SSIM scores and a higher LPIPS score, as these signify a greater divergence between the reconstructed and original images.

![](images/565c8a3824246741fd05b360af22da65c263aabadf5bb8317eb1b7e81e8db6b7.jpg)  
Figure 4: Visualization of reconstructed images of IG [10]. The images recovered from FedAvg, pFedHN, and pFedHN-PC are clearly recognizable, indicating severe privacy leakage. In contrast, images from DPbased methods, HyperFL, and HyperFL++ are noisy and unrecognizable, demonstrating efective privacy protection.

The quantitative results are presented in Table 1, with corresponding visualizations in Figure 4. The results clearly expose the vulnerabilities of standard FL frameworks. FedAvg, pFedHN, and pFedHN-PC, which directly or indirectly expose local model information, are highly susceptible to GIA. This is evidenced by their high PSNR and SSIM scores and low LPIPS scores, which quantitatively confirm the high fidelity of the reconstructed images seen in Figure 4. While DP-based methods (DP-FedAvg, CENTAUR, PPSGD) ofer strong privacy protection, as shown by their favorable metric scores, this comes at a steep cost to model performance (detailed in Table 2).

In stark contrast, both HyperFL and our proposed HyperFL++ achieve robust privacy protection, with metrics comparable to those of DP-based methods, while simultaneously delivering superior model accuracy (see Table 2). This demonstrates the efectiveness of our core idea: “breaking the direct connection between shared parameters and local data”.

Table 1: Reconstruction results of IG [10]. Higher LPIPS, Lower PSNR, and SSIM of recovered images indicate better defense ability.
<table><tr><td rowspan="2">Method</td><td colspan="3">EMNIST</td><td colspan="3">CIFAR-10</td></tr><tr><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>FedAvg [4]</td><td>32.64</td><td>0.8925</td><td>0.0526</td><td>16.16</td><td>0.6415</td><td>0.0536</td></tr><tr><td>pFedHN [30]</td><td>31.24</td><td>0.8701</td><td>0.0807</td><td>16.02</td><td>0.6351</td><td>0.0504</td></tr><tr><td>pFedHN-PC [30]</td><td>28.38</td><td>0.8713</td><td>0.0645</td><td>15.80</td><td>0.6247</td><td>0.0407</td></tr><tr><td>DP-FedAvg [21]</td><td>7.74</td><td>0.2978</td><td>0.7051</td><td>7.90</td><td>0.2716</td><td>0.3204</td></tr><tr><td>CENTAUR [19]</td><td>9.52</td><td>0.2136</td><td>0.6712</td><td>9.80</td><td>0.2723</td><td>0.2882</td></tr><tr><td>PPSGD [45]</td><td>9.73</td><td>0.1889</td><td>0.6466</td><td>9.70</td><td>0.2788</td><td>0.2643</td></tr><tr><td>HyperFL [44]</td><td>7.85</td><td>0.3010</td><td>0.7147</td><td>8.35</td><td>0.2732</td><td>0.3132</td></tr><tr><td>HyperFL++</td><td>8.93</td><td>0.2274</td><td>0.6863</td><td>8.31</td><td>0.2796</td><td>0.7657</td></tr></table>

## 5.3. Utility Evaluation

We now evaluate the utility of our proposed method in terms of model performance, system eficiency, and convergence rate.

## 5.3.1. Model Performance

As shown in Table 2, the DP-based methods consistently underperform FedAvg. This is an expected outcome, as the injection of statistical noise, which is fundamental to DP, inherently degrades model utility to provide privacy guarantees. In contrast, our proposed HyperFL++ framework not only provides strong privacy but also achieves higher accuracy than the standard FedAvg in most tested scenarios. This result is significant as it demonstrates that HyperFL++ breaks the typical privacy-utility trade-of. While personalization methods like pFedHN and pFedHN-PC also show performance gains over FedAvg, they ofer weak defense against GIA, as shown in our privacy evaluation, making them unsuitable for privacy-sensitive applications. HyperFL++ successfully combines the benefits of personalization (leading to higher accuracy) with robust privacy protection.

Table 3 presents the results for HyperFL++-LPM, our adaptation for large pretrained models. The results show that our method achieves performance comparable to FedAvg with full model fine-tuning, which is often considered the upper bound for performance but is computationally expensive and memory-intensive. More importantly, HyperFL++-LPM significantly outperforms FedAvg with adapter-only tuning. This is a crucial finding, as it demonstrates that our hypernetwork-based approach for generating adapter parameters is more efective than the standard method of simply averaging adapter parameters, confirming its efectiveness and utility in large-model settings.

Table 2: The comparison of final test accuracy (%) of diferent methods on various datasets. We apply full participation for FL system with 20 clients, and apply client sampling with rate 0.3 for FL system with 100 clients. # Para [EMNIST/Fashion-MNIST; CIFAR-10/CINIC-10] denotes the communication parameters during each FL round. Underline indicates better performance than FedAvg.
<table><tr><td rowspan="2">Method</td><td rowspan="2"># Para</td><td colspan="2">EMNIST</td><td colspan="2">Fashion-MNIST</td><td colspan="2">CIFAR-10</td><td colspan="2">CINIC-10</td></tr><tr><td>20 clients</td><td>100 clients</td><td>20 clients</td><td>100 clients</td><td>20 clients</td><td>100 clients</td><td>20 clients</td><td>100 clients</td></tr><tr><td>FedAvg [4]</td><td>[115,776; 106,400]</td><td>72.77</td><td>78.87</td><td>85.67</td><td>88.11</td><td>70.02</td><td>76.24</td><td>57.00</td><td>59.11</td></tr><tr><td>pFedHN [30]</td><td>[115,776; 106,400]</td><td>80.86</td><td>77.37</td><td>87.64</td><td>89.80</td><td>70.18</td><td>80.07</td><td>63.88</td><td>70.36</td></tr><tr><td>pFedHN-PC [30]</td><td>[109,368; 110,168]</td><td>82.04</td><td>79.32</td><td>89.17</td><td>90.06</td><td>72.52</td><td>78.15</td><td>64.92</td><td>68.73</td></tr><tr><td>DP-FedAvg [21]</td><td>[115,776; 106,400]</td><td>35.12</td><td>45.73</td><td>59.88</td><td>68.29</td><td>29.12</td><td>32.03</td><td>27.30</td><td>29.94</td></tr><tr><td>CENTAUR [19]</td><td>[115,776; 106,400]</td><td>68.82</td><td>67.24</td><td>83.07</td><td>79.77</td><td>50.85</td><td>51.86</td><td>48.82</td><td>51.01</td></tr><tr><td>PPSGD [45]</td><td>[115,776; 106,400]</td><td>71.16</td><td>71.18</td><td>84.47</td><td>82.94</td><td>52.17</td><td>53.92</td><td>49.98</td><td>52.91</td></tr><tr><td>HyperFL [44]</td><td>[6,922,966; 6,924,256]</td><td>78.42</td><td>80.71</td><td>87.46</td><td>89.38</td><td>71.98</td><td>74.74</td><td>66.52</td><td>70.35</td></tr><tr><td>HyperFL++</td><td>[70,136; 84,556]</td><td>78.18</td><td>81.85</td><td>88.26</td><td>90.09</td><td>71.80</td><td>74.76</td><td>65.86</td><td>69.52</td></tr></table>

Table 3: The comparison of final test accuracy (%) of diferent methods on various datasets with 20 clients. <sup>†</sup> Adapter fine-tuning. <sup>††</sup> Full parameter tuning.
<table><tr><td>Method</td><td>EMNIST</td><td>Fashion-MNIST</td><td>CIFAR-10</td><td>CINIC-10</td></tr><tr><td>FedAvg† [4]</td><td>75.86</td><td>83.95</td><td>68.55</td><td>65.90</td></tr><tr><td>FedAvg†† [4]</td><td>72.73</td><td>85.69</td><td>92.37</td><td>91.37</td></tr><tr><td>HyperFL++-LPM</td><td>78.73</td><td>84.91</td><td>88.12</td><td>83.57</td></tr></table>

## 5.3.2. System Eficiency

We assess system eficiency by measuring communication cost, per-round computation cost, and the number of rounds needed to reach a target accuracy (95% of FedAvg’s final accuracy), following the methodology in [46, 33]. As shown in Table 4, HyperFL++ demonstrates remarkable eficiency. A key highlight is the drastic reduction in communication overhead compared to its predecessor, HyperFL. Thanks to the RKD technique, HyperFL++ reduces the much number of communicated parameters compared to HyperFL, bringing the cost down to a level even lower than that of standard FedAvg. While its per-round computation time is slightly higher than FedAvg’s due to the hypernetwork’s forward pass, this is more than compensated for by its superior convergence speed. HyperFL++ requires significantly fewer communication rounds to reach the target performance compared to all other baselines. For example, on CIFAR-10, it reaches the target in just 42 rounds, whereas FedAvg needs 123 rounds. This rapid convergence, combined with low communication costs, highlights the overall system eficiency of our framework, making it highly practical for real-world deployment.

Table 4: Per-round communication and computation costs. # Comm. Rounds denotes the number of rounds to reach the predefined target performance. - means it cannot reach the target performance.
<table><tr><td rowspan="2">Method</td><td colspan="2"># Communicated Param</td><td colspan="2">Computation Cost</td><td colspan="2"># Comm. Rounds</td></tr><tr><td>EMNIST</td><td>CIFAR-10</td><td>EMNIST</td><td>CIFAR-10</td><td>EMNIST</td><td>CIFAR-10</td></tr><tr><td>FedAvg [4]</td><td>115,776</td><td>106,400</td><td>21.99</td><td>19.39</td><td>11</td><td>123</td></tr><tr><td>pFedHN [30]</td><td>115,776</td><td>106,400</td><td>16.79</td><td>25.10</td><td>15</td><td>76</td></tr><tr><td>pFedHN-PC [30]</td><td>109,368</td><td>110,168</td><td>30.12</td><td>31.58</td><td>3</td><td>60</td></tr><tr><td>DP-FedAvg [21]</td><td>115,776</td><td>106,400</td><td>202.52</td><td>144.33</td><td>1</td><td>一</td></tr><tr><td>CENTAUR [19]</td><td>115,776</td><td>106,400</td><td>187.47</td><td>140.39</td><td>一</td><td>一</td></tr><tr><td>PPSGD [45]</td><td>115,776</td><td>106,400</td><td>225.99</td><td>177.11</td><td>一</td><td>一</td></tr><tr><td>HyperFL [44]</td><td>6,922,966</td><td>6,924,256</td><td>37.01</td><td>35.83</td><td>4</td><td>63</td></tr><tr><td>HyperFL++</td><td>70,136</td><td>84,556</td><td>47.83</td><td>40.04</td><td>3</td><td>42</td></tr></table>

## 5.3.3. Convergence Rate

Figure 5 illustrates the training loss curves for FedAvg, HyperFL, and HyperFL++ on the EMNIST dataset. The plot shows that HyperFL++ not only converges smoothly but does so at a rate comparable to the standard FedAvg, validating the strong theoretical convergence properties established in Section 4.2.1. The training loss for HyperFL++ decreases steadily and reaches a low plateau, indicating stable and efective training. Furthermore, after convergence, HyperFL++ achieves a lower final training loss than FedAvg. This is a strong indicator of its superior modeling capacity, which directly correlates with the higher test accuracy reported in Table 2. This demonstrates that the privacy-preserving mechanism of HyperFL++ does not hinder the optimization process; on the contrary, it facilitates better model fitting in heterogeneous data environments.

![](images/9086467b29e0972f3bf78cd885082a5472a6d729a28787201e1956e045ae7482.jpg)  
Figure 5: Average training loss on the EMNIST dataset with 20 clients. The curve for HyperFL++ shows a smooth and rapid decrease, achieving a lower final loss than FedAvg, which indicates better model perfor mance.

## 6. Conclusion

In this paper, we introduce HyperFL++, a novel federated learning framework designed to defend against Gradient Inversion Attacks by fundamentally altering the information-sharing mechanism. Instead of exchanging vulnerable model gradients, clients in our framework share only the parameters of a hypernetwork, which in turn generates the local model parameters. This approach efectively “breaks the direct connection” between shared updates and private data. To ensure the practicality of this design, we incorporate a Row-wise Kronecker Decomposition (RKD) technique, which significantly reduces the communication and memory overhead associated with the hypernetwork. Our extensive theoretical analysis and experimental results demonstrate that HyperFL++ provides robust privacy protection without compromising model utility or convergence speed. We hope this work will inspire further research into designing inherently private FL frameworks, shifting the focus from retrofitting defenses to building privacy-by-design systems.

## Appendix A. Theoretical Proof

## Appendix A.1. Proof of Theorem 1

Proof. Let $\phi _ { i } ^ { t } , \varphi _ { i } ^ { t } , \mathbf { v } _ { i } ^ { t }$ be the model parameters maintained in the i-th client at the t-th step. Let $\scriptstyle { \mathcal { I } } _ { E }$ be the set of global synchronization steps, i.e., $\mathcal { T } _ { E } = \{ n E \mid n = 1 , 2 , \cdots \}$ $\mathrm { I f ~ } t + 1 \ \in \mathcal { I } _ { E }$ , which represents the time step for communication, then the one-step update of HyperFL can be described as follows:

$$
\left( \begin{array} { c } { \phi _ { i } ^ { t } } \\ { \varphi _ { i } ^ { t } } \\ { \mathbf { v } _ { i } ^ { t } } \end{array} \right) { \substack { \operatorname { S G D ~ o f } \phi _ { i } ^ { t } \left( \begin{array} { c } { \phi _ { i } ^ { t + 1 } } \\ { \varphi _ { i } ^ { t } } \\ { \mathbf { v } _ { i } ^ { t } } \end{array} \right) } } \quad \underset { \mathbf { v } _ { i } ^ { t } } { \longrightarrow } \left( \begin{array} { c } { \phi _ { i } ^ { t + 1 } } \\ { \varphi _ { i } ^ { t + 1 } } \\ { \mathbf { v } _ { i } ^ { t + 1 } } \end{array} \right) \underset { \mathbf { v } _ { i } ^ { t + 1 } } { \longrightarrow } \left( \begin{array} { c } { \phi _ { i } ^ { t + 1 } } \\ { \varphi _ { i } ^ { t + 1 } } \\ { \vdots \operatorname { f } ^ { t + 1 } \epsilon { \cal T } _ {  { \varepsilon } } } \\ { \mathbf { v } _ { i } ^ { t + 1 } } \end{array} \right) .
$$

For convenience, we denote the parameters in each sub-step above as follows:

$$
\boldsymbol { x } _ { i } ^ { t } = \left( \begin{array} { c } { \phi _ { i } ^ { t } } \\ { \varphi _ { i } ^ { t } } \\ { \mathbf { v } _ { i } ^ { t } } \end{array} \right) , \boldsymbol { y } _ { i } ^ { t } = \left( \begin{array} { c } { \phi _ { i } ^ { t + 1 } } \\ { \varphi _ { i } ^ { t } } \\ { \mathbf { v } _ { i } ^ { t } } \end{array} \right) , \boldsymbol { x } _ { i } ^ { t + 1 , 1 } = \left( \begin{array} { c } { \phi _ { i } ^ { t + 1 } } \\ { \varphi _ { i } ^ { t + 1 } } \\ { \mathbf { v } _ { i } ^ { t + 1 } } \end{array} \right) , \boldsymbol { x } _ { i } ^ { t + 1 , 2 } = \left( \begin{array} { c } { \phi _ { i } ^ { t + 1 } } \\ { \sum _ { j = 1 } ^ { m } w _ { j } \varphi _ { j } ^ { t + 1 } } \\ { \mathbf { v } _ { i } ^ { t + 1 } } \end{array} \right) ,
$$

$$
x _ { i } ^ { t + 1 } = \left\{ \begin{array} { l l } { x _ { i } ^ { t + 1 , 1 } } & { \mathrm { ~ i f ~ } t + 1 \notin \bar { \mathcal { I } } _ { E } , } \\ { x _ { i } ^ { t + 1 , 2 } } & { \mathrm { ~ i f ~ } t + 1 \in \bar { \mathcal { I } } _ { E } . } \end{array} \right.
$$

Here, the variable $x _ { i } ^ { t + 1 , 1 }$ represents the immediate result of one sub-step SGD update from the parameter of the previous sub-step $y _ { i } ^ { t } ,$ and $x _ { i } ^ { t + 1 , 2 }$ represents the parameter obtained after communication steps (if possible).

Furthermore, we denote the learning rate and stochastic gradient of step t as follows:

$$
\eta = \left( \begin{array} { c } { { \eta _ { g } } } \\ { { \eta _ { h } } } \\ { { } } \\ { { \eta _ { \nu } } } \end{array} \right) ,
$$

$$
g _ { i } ^ { t } = \left( \begin{array} { l } { g _ { i , \phi } ^ { t } } \\ { g _ { i , \varphi } ^ { t } } \\ { g _ { i , \mathrm { v } } ^ { t } } \end{array} \right) = \left( \begin{array} { l } { \nabla _ { \phi } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t } , \xi _ { i } ^ { t } \right) } \\ { \nabla _ { \varphi } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } , \xi _ { i } ^ { t } \right) } \\ { \nabla _ { \mathrm { v } } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } , \xi _ { i } ^ { t } \right) } \end{array} \right) ,
$$

$$
\begin{array} { r } { \bar { g } _ { i } ^ { t } = \left( \begin{array} { l } { \bar { g } _ { i , \phi } ^ { t } } \\ { \bar { g } _ { i , \varphi } ^ { t } } \\ { \bar { g } _ { i , \mathrm { v } } ^ { t } } \end{array} \right) = \left( \begin{array} { l } { \nabla _ { \phi } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t } \right) } \\ { \nabla _ { \varphi } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } \right) } \\ { \nabla _ { \mathrm { v } } \mathcal { L } _ { i } \left( h \left( \mathbf { v } _ { i } ^ { t } , \varphi _ { i } ^ { t } \right) , \phi _ { i } ^ { t + 1 } \right) } \end{array} \right) , } \end{array}
$$

where $\xi _ { i } ^ { t }$ is the data uniformly chosen from the local data set of client i at step t, then $\mathbb { E } \left[ g _ { i } ^ { t } \right] = \bar { g } _ { i } ^ { t } .$

Next, we apply the inequality of the smoothness Assumption 1 to each sub-step of the one-step update for client i. Firstly, by the smoothness of $\mathcal { L } _ { i }$ , we have:

$$
\mathcal { L } _ { i } \left( y _ { i } ^ { t } \right) \leq \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) + \left. y _ { i } ^ { t } - x _ { i } ^ { t } , \bar { g } _ { i , \phi } ^ { t } \right. + \frac { L } { 2 } \left. y _ { i } ^ { t } - x _ { i } ^ { t } \right. ^ { 2 } .\tag{A.1}
$$

For the second term on the right side of inequality (A.1), according to the law of total expectation, we have:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. y _ { i } ^ { t } - x _ { i } ^ { t } , \bar { g } _ { i , \phi } ^ { t } \right. \right] = \mathbb { E } \left[ \left. - \eta _ { g } g _ { i , \phi } ^ { t } , \bar { g } _ { i , \phi } ^ { t } \right. \right] } \\ & { \quad \quad \quad \quad = \mathbb { E } \left\{ \mathbb { E } \left[ \left. - \eta _ { g } g _ { i , \phi } ^ { t } , \bar { g } _ { i , \phi } ^ { t } \right. \right] | \xi _ { i } ^ { t } \right\} } \\ & { \quad \quad \quad = \mathbb { E } \left\{ \mathbb { E } \left[ \left. - \eta _ { g } g _ { i , \phi } ^ { t } | \xi _ { i } ^ { t } , \bar { g } _ { i , \phi } ^ { t } \right. \right] \right\} } \\ & { \quad \quad \quad = \mathbb { E } \left[ \left. - \eta _ { g } \bar { g } _ { i , \phi } ^ { t } , \bar { g } _ { i , \phi } ^ { t } \right. \right] } \\ & { \quad \quad \quad = - \eta _ { g } \mathbb { E } \left[ \left( \bar { g } _ { i , \phi } ^ { t } \right) ^ { 2 } \right] . } \end{array}
$$

For the third term on the right side of the inequality (A.1), we have:

$$
\begin{array} { r l } { \displaystyle \mathbb { E } [ \frac { L } { 2 } \| y _ { i } ^ { t } - x _ { i } ^ { t } \| ^ { 2 } ] = \mathbb { E } [ \frac { L } { 2 } \| - \eta _ { g } g _ { i , \phi } ^ { t } \| ^ { 2 } ] } & { } \\ { \displaystyle = \eta _ { g } ^ { 2 } \frac { L } { 2 } \mathbb { E } [ | g _ { i , \phi } ^ { t } | ] ^ { 2 } ] } & { } \\ { \displaystyle } & { \leq \eta _ { g } ^ { 2 } \frac { L G ^ { 2 } } { 2 } , } \end{array}
$$

where in the last inequality, we use the bounded gradient Assumption 3.

From the above inequalities, and taking the expectation of inequality (A.1), we can get:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( y _ { i } ^ { t } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) \right] \leq - \eta _ { g } \mathbb { E } \left[ ( \bar { g } _ { i , \phi } ^ { t } ) ^ { 2 } \right] + \eta _ { g } ^ { 2 } \frac { L G ^ { 2 } } { 2 } .\tag{A.2}
$$

Secondly, by the smoothness of $\mathcal { L } _ { i }$ , we have:

$$
\mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 1 } \right) \leq \mathcal { L } _ { i } \left( y _ { i } ^ { t } \right) + \left. x _ { i } ^ { t + 1 , 1 } - y _ { i } ^ { t } , \bar { g } _ { i , \phi } ^ { t } \right. + \frac { L } { 2 } \left. x _ { i } ^ { t + 1 , 1 } - y _ { i } ^ { t } \right. ^ { 2 } .\tag{A.3}
$$

Similar to inequality (A.2), taking the expectation of inequality (A.3), we get:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 1 } \right) - \mathcal { L } _ { i } \left( y _ { i } ^ { t } \right) \right] \leq - \eta _ { h } \mathbb { E } \left[ ( \bar { g } _ { i , \varphi } ^ { t } ) ^ { 2 } \right] - \eta _ { \nu } \mathbb { E } \left[ ( \bar { g } _ { i , \mathbf { v } } ^ { t } ) ^ { 2 } \right] + ( \eta _ { h } ^ { 2 } + \eta _ { \nu } ^ { 2 } ) \frac { L G ^ { 2 } } { 2 } .\tag{A.4}
$$

Thirdly, by the smoothness of $\mathcal { L } _ { i }$ , we have:

$$
\mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 2 } \right) \leq \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 1 } \right) + \left. x _ { i } ^ { t + 1 , 2 } - x _ { i } ^ { t + 1 , 1 } , \bar { g } _ { i , \varphi } ^ { t } \right. + \frac { L } { 2 } \left. x _ { i } ^ { t + 1 , 2 } - x _ { i } ^ { t + 1 , 1 } \right. ^ { 2 } .\tag{A.5}
$$

From the iterative formula of SGD, it is clear that:

$$
\varphi _ { j } ^ { t + 1 } = \varphi _ { j } ^ { t - E + 1 } - \eta _ { h } \sum _ { t _ { 0 } = t - E + 1 } ^ { t } g _ { j , \varphi } ^ { t _ { 0 } } , \quad \forall j , t + 1 \in \bar { \cal I } _ { E } .\tag{A.6}
$$

Then, for the third term on the right side of inequality (A.5), we apply the equality (A.6) and take the expectation, which yields:

$$
\begin{array} { r l } { \mathbb { E } [ \frac { L } { 2 }  k _ { i } ^ { n + 1 , 2 } - x _ { i } ^ { n + 1 , 4 }  ^ { 2 } ] = \frac { L } { 2 } \mathbb { E } [  - \eta k \displaystyle \sum _ { j = 1 } ^ { m } w _ { j } \displaystyle \sum _ { q = 1 } ^ { i } ( B _ { j q } ^ { i , \mu } - B _ { i q } ^ { \mu } ) ^ { 2 } ] ^ { 2 } } & { } \\ { \leq \eta k _ { i } ^ { 2 } \displaystyle \sum _ { j = 1 } ^ { L } w _ { j } \mathbb { E } [  \displaystyle \sum _ { q = r - k \neq 1 } ^ { i } ( B _ { j q } ^ { i , \mu } - B _ { i q } ^ { \mu } ) ^ { 2 }  ^ { 2 } ] } & { } \\ { \leq \eta k _ { i } ^ { 2 } \displaystyle \sum _ { j = 1 } ^ { L } w _ { j } \displaystyle \sum _ { q = 1 } ^ { i }  [  ( B _ { j q } ^ { i , \mu } - B _ { i q } ^ { \mu } ) ^ { 2 }  ^ { 2 } ] } & { } \\ { \leq \eta k _ { i } ^ { 2 } \displaystyle \sum _ { j = 1 } ^ { L } w _ { j } \displaystyle \sum _ { q = r - k + 1 } ^ { i }    ( B _ { j q } ^ { i , \mu } - B _ { i q } ^ { \mu } ) ^ { 2 }  ^ { 2 } ] } & { } \\ { \leq \eta k _ { i } ^ { 2 } \displaystyle \sum _ { j = 1 } ^ { L } \frac { N ^ { 2 } } { \mu } w _ { j } \displaystyle \sum _ { q = r - k + 1 } ^ { i } \mathbb { E } [ \frac { 1 } { 2 }   g _ { j q , q } ^ { i , \mu }  ^ { 2 } + \frac { 1 } { 2 }  \theta _ { i q } ^ { i , \mu }  ^ { 2 } ] } & { } \\  \leq \eta k _ { i } ^ { 2 } \displaystyle \frac { ( E - 1 ) ^ { 2 } } { \mu } \sum _  q = \end{array}
$$

where in the last inequality, we use the bounded gradient Assumption 3.

For the second term on the right side of inequality (A.5), we take the expectation and get:

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left. x _ { i } ^ { t + 1 , 2 } - x _ { i } ^ { t + 1 , 1 } , \bar { g } _ { i , \varphi } ^ { t } \right. \right] \leq \frac { 1 } { 2 \eta _ { h } } \left( x _ { i } ^ { t + 1 , 2 } - x _ { i } ^ { t + 1 , 1 } \right) ^ { 2 } + \frac { 1 } { 2 } \eta _ { h } \left( \bar { g } _ { i , \varphi } ^ { t } \right) ^ { 2 } } \\ { \displaystyle \qquad \leq \frac { 1 } { 2 \eta _ { h } } \eta _ { h } ^ { 2 } ( E - 1 ) G ^ { 2 } + \frac { 1 } { 2 } \eta _ { h } \left( \bar { g } _ { i , \varphi } ^ { t } \right) ^ { 2 } } \\ { \displaystyle \qquad = \eta _ { h } \frac { ( E - 1 ) G ^ { 2 } } { 2 } + \frac { 1 } { 2 } \eta _ { h } \left( \bar { g } _ { i , \varphi } ^ { t } \right) ^ { 2 } , } \end{array}
$$

where we use the Cauchy-Schwarz inequality and the AM-GM inequality in the first inequality, and the bounded gradient Assumption 3 in the second inequality above.

Then, based on the above inequalities and taking the expectation of inequality (A.5), we have:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 2 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 1 } \right) \right] \leq \eta _ { h } ^ { 2 } \frac { ( E - 1 ) L G ^ { 2 } } { 2 } + \eta _ { h } \frac { ( E - 1 ) G ^ { 2 } } { 2 } + \frac { 1 } { 2 } \eta _ { h } \left( \bar { g } _ { i , \varphi } ^ { t } \right) ^ { 2 } .\tag{A.7}
$$

Summing up inequalities (A.2) and (A.4), we get:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 1 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) \right] \leq - \eta ^ { \top } \mathbb { E } \left[ ( g _ { i } ^ { t } ) ^ { 2 } \right] + \left. \eta \right. ^ { 2 } \frac { L G ^ { 2 } } { 2 } .\tag{A.8}
$$

Summing up inequalities (A.2), (A.4), and (A.7), we get:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 2 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) \right] \leq - \left( \begin{array} { l } { \eta _ { g } } \\ { \frac { 1 } { 2 } \eta _ { h } } \\ { \eta _ { \nu } } \end{array} \right) ^ { \top } \mathbb { E } \left[ \left( \bar { g } _ { i } ^ { t } \right) ^ { 2 } \right] + \left[ \eta _ { g } ^ { 2 } + \eta _ { \nu } ^ { 2 } + ( E - 1 ) \eta _ { h } ^ { 2 } + \frac { E - 1 } { L } \eta _ { h } \right] \frac { L G ^ { 2 } } { 2 } .\tag{A.9}
$$

Then, let $\begin{array} { r } { \eta _ { m i n } = \operatorname* { m i n } \{ \eta _ { g } , \frac { 1 } { 2 } \eta _ { h } , \eta _ { \nu } \} } \end{array}$ , we have:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 , 2 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) \right] \leq - \eta _ { m i n } \mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] + \left[ \eta _ { g } ^ { 2 } + \eta _ { \nu } ^ { 2 } + ( E - 1 ) \eta _ { h } ^ { 2 } + \frac { E - 1 } { L } \eta _ { h } \right] \frac { L G ^ { 2 } } { 2 } .\tag{A.10}
$$

By rewriting the above inequality (A.10), we get:

$$
\mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] \leq \frac { \mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 } \right) \right] } { \eta _ { m i n } } + \frac { \eta _ { g } ^ { 2 } + \eta _ { \nu } ^ { 2 } + ( E - 1 ) \eta _ { h } ^ { 2 } + \frac { E - 1 } { L } \eta _ { h } } { \eta _ { m i n } } \frac { L G ^ { 2 } } { 2 } .\tag{A.11}
$$

Let M be a constant that satisfies the inequality $\begin{array} { r } { \eta _ { g } ^ { 2 } + \eta _ { \nu } ^ { 2 } + ( E - 1 ) \eta _ { h } ^ { 2 } + \frac { E - 1 } { L } \eta _ { h } \le M \eta _ { m i n } ^ { 2 } , } \end{array}$ the aforementioned inequality (A.11) can be further simplified as:

$$
\mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] \leq \frac { \mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 } \right) \right] } { \eta _ { m i n } } + \eta _ { m i n } \frac { L M G ^ { 2 } } { 2 } .\tag{A.12}
$$

Now, by repeatedly applying inequality (A.12) for diferent values of t and summing up the results, we get:

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] \leq \frac { \mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { 1 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { * } \right) \right] } { \eta _ { m i n } } + \eta _ { m i n } \frac { L M G ^ { 2 } } { 2 } T .\tag{A.13}
$$

Dividing both side of inequality (A.13) by T, we get:

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] \leq \frac { \mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { 1 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { * } \right) \right] } { \eta _ { m i n } T } + \eta _ { m i n } \frac { L M G ^ { 2 } } { 2 } .\tag{A.14}
$$

Let us assume that $\mathcal { L } _ { i } \left( x _ { i } ^ { 1 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { * } \right) \leq D , \forall i .$ , and we set $\begin{array} { r } { \eta _ { m i n } = \sqrt { \frac { 2 D } { L M G ^ { 2 } T } } } \end{array}$ . Then, we have:

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] \leq 2 \sqrt { \frac { L M G ^ { 2 } D } { 2 T } } .\tag{A.15}
$$

Thus, we can get:

$$
\frac { 1 } { m T } \sum _ { i = 1 } ^ { m } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] \leq 2 \sqrt { \frac { L M G ^ { 2 } D } { 2 T } } .\tag{A.16}
$$

## Appendix A.2. Proof of Corollary 1

Proof. By rewriting inequality (A.12), we have:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) \right] \leq - \eta _ { m i n } \mathbb { E } \left[ \left. \bar { g } _ { i } ^ { t } \right. ^ { 2 } \right] + \eta _ { m i n } ^ { 2 } \frac { L M G ^ { 2 } } { 2 } .\tag{A.17}
$$

By the PL Assumption 4, we have:

$$
\mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) \right] \leq - 2 \eta _ { m i n } \mu \mathbb { E } \left[ \mathcal { L } _ { i } \left( x _ { i } ^ { t } \right) - \mathcal { L } _ { i } \left( x _ { i } ^ { * } \right) \right] + \eta _ { \operatorname* { m i n } } ^ { 2 } \frac { L M G ^ { 2 } } { 2 } .\tag{A.18}
$$

Then,

$$
\begin{array} { r l } { \mathbb { E } [ \mathcal { L } _ { i } ( x _ { i } ^ { n + 1 } ) - \mathcal { L } _ { i } ( x _ { i } ^ { * } ) ] = \mathbb { E } [ \mathcal { L } _ { i } ( x _ { i } ^ { * } ) - \mathcal { L } _ { i } ( x _ { i } ^ { * } ) ] + \mathbb { E } [ \mathcal { L } _ { i } ( x _ { i } ^ { * * } ) - \mathcal { L } _ { i } ( x _ { i } ^ { * } ) ] } \\ { \ } & { \leq \mathbb { E } [ \mathcal { L } _ { i } ( x _ { i } ^ { * } ) - \mathcal { L } _ { i } ( x _ { i } ^ { * } ) ] - 2 \eta _ { \mathrm { o n d } } L [ \mathcal { L } _ { i } ( x _ { i } ^ { * } ) - \mathcal { L } _ { i } ( x _ { i } ^ { * } ) ] + \eta _ { \mathrm { o n } } ^ { 2 } \frac { L M G ^ { 2 } } { 2 } } \\ { = ( 1 - 2 \eta _ { \mathrm { o n } } \mu ) \mathbb { E } [ \mathcal { L } _ { i } ( x _ { i } ^ { * } ) - \mathcal { L } _ { i } ( x _ { i } ^ { * } ) ] + \eta _ { \mathrm { o n } } ^ { 2 } \frac { L M G ^ { 2 } } { 2 } } \\ { \ } & { \leq ( 1 - 2 \eta _ { \mathrm { o n } } \mu ) ^ { 2 } \mathbb { E } [ \mathcal { L } _ { i } ( x _ { i } ^ { * } ) - \mathcal { L } _ { i } ( x _ { i } ^ { * } ) ] + \frac { \sum } { \eta _ { \mathrm { o n } } } ( 1 - 2 \eta _ { \mathrm { m i n } } \mu ) ^ { 2 } \eta _ { \mathrm { o n } } ^ { 2 } \frac { L M G ^ { 2 } } { 2 } } \\ { \ } & { \cdots } \\ { \ } &  \leq ( 1 - 2 \eta _ { \mathrm { o n } } \mu ) ^ { 2 + 1 } \mathbb { E } [ \mathcal { L } _ { i } ( x _ { i } ^ { * } )  \end{array}
$$

Therefore, we have:

$$
\mathbb { E } \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 } \right) \right] - \mathcal { L } ^ { * } \leq ( 1 - \eta _ { m i n } \mu ) ^ { t + 1 } D + \eta _ { m i n } \frac { L M G ^ { 2 } } { 4 \mu } .\tag{A.19}
$$

If we set $\begin{array} { r } { \eta _ { m i n } \leq \frac { \mu \epsilon } { L M G ^ { 2 } } } \end{array}$ , after $O ( \textstyle { \frac { 1 } { \epsilon } } \log ( \textstyle { \frac { 1 } { \epsilon } } ) )$ steps, we have:

$$
\mathbb { E } \left[ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathcal { L } _ { i } \left( x _ { i } ^ { t + 1 } \right) \right] - \mathcal { L } ^ { * } \leq \epsilon .\tag{A.20}
$$

## Appendix A.3. Proof of Theorem 2

Proof. First, we define the distance between $( \phi , \mathbf { v } , \varphi )$ and $( \boldsymbol { \phi } ^ { * } , \mathbf { v } ^ { * } , \boldsymbol { \varphi } ^ { * } )$ as:

$$
d ( ( \phi , { \bf v } , \varphi ) , ( \phi ^ { * } , { \bf v } ^ { * } , \varphi ^ { * } ) ) = \Big | \sum _ { i } ^ { m } \frac { n _ { i } } { N } ( \mathcal { L } _ { i } ( h ( \varphi _ { i } ^ { * } ; { \bf v } _ { i } ^ { * } ) , \phi _ { i } ^ { * } ) - \mathcal { L } _ { i } ( h ( \varphi _ { i } ; { \bf v } _ { i } ) , \phi _ { i } ) ) \Big | ,\tag{A.21}
$$

where $\begin{array} { r } {  { { \cal N } } ~ = ~ \sum _ { i = 1 } ^ { m } n _ { i } } \end{array}$ . By the Theorem 4 from [39], we can find an ϵ-covering in $d ( ( \phi , { \bf v } , \varphi ) , ( \phi ^ { * } , { \bf v } ^ { * } , \varphi ^ { * } ) )$ . Then, according to the notations used in our paper, we have that $\begin{array} { r } { S \ = \ O ( \frac { 1 } { m \epsilon ^ { 2 } } l o g ( \frac { C ( \epsilon , \mathcal { H } _ { l } ^ { n } ) } { \delta } ) ) } \end{array}$ , where $C ( \epsilon , \mathcal { H } _ { l } ^ { n } )$ is the covering number of $\mathcal { H } _ { l } ^ { n }$ . In our case, each element of $\mathcal { H } _ { l } ^ { n }$ is parameterized by $\phi , { \bf v } , \varphi .$ . Therefore, from the triangle inequality and the Lipschitz conditions in Assumption 3, we can get:

$$
\begin{array} { r l } { d ( ( \phi , \mathbf { v } , \varphi ) , ( \phi ^ { * } , \mathbf { v } ^ { * } , \varphi ^ { * } ) ) = \displaystyle | \sum _ { i = 1 } ^ { m } \frac { n _ { i } } { N } \big ( \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ^ { * } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ^ { * } ) - \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ) \big ) \big | } & { } \\ & { - \Big | \displaystyle \sum _ { i = 1 } ^ { m } \frac { n _ { i } } { N } \big ( \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ^ { * } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ^ { * } ) - \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ^ { * } ) } \\ & { \quad \quad \quad + \ \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ^ { * } ) - \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ; \mathbf { v } _ { i } ) , \phi _ { i } ^ { * } ) + \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ; \mathbf { v } _ { i } ) , \phi _ { i } ^ { * } ) - \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ) \big ) } \\ &  \le \displaystyle \sum _ { i = 1 } ^ { m } \frac { n _ { i } } { N } \Big | \big ( \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ^ { * } ; \mathbf { v } _ { i } ^ { * } ) , \phi _ { i } ^ { * } ) - \mathcal { L } _ { \xi } ( \mathrm { A } ( \phi _ { i } ; \ \end{array}\tag{A.22}
$$

Now if there is a parameter space such that $\phi _ { i } , { \bf v } _ { i }$ and $\varphi _ { i }$ have corresponding optimal point $\phi _ { i } ^ { * } , \mathbf { v } _ { i } ^ { * }$ and $\varphi _ { i } ^ { * } ,$ , which are $\frac { \epsilon } { L _ { h } L _ { \varphi } + L _ { h } L _ { \nu } + L _ { \phi } }$ away, respectively, we can get an upper bound of the distance between our model and optimal model, which is an ϵ-covering in $d ( ( \phi , { \bf v } , \varphi ) , ( \phi ^ { * } , { \bf v } ^ { * } , \varphi ^ { * } ) )$ matrix. From here we see that log $( C ( \epsilon , { \mathcal { H } } _ { l } ^ { n } ) ) = O ( m ( d + H +$ $\begin{array} { r } { K ) \log ( \frac { R L _ { h } ( L _ { \varphi } + L _ { \nu } ) + R L _ { \phi } } { \epsilon } ) ) } \end{array}$ .

## Acknowledgments

This work was supported by National Natural Science Foundation of China (62306253, 62206075), Guangdong Natural Science Fund-General Programme (2024A1515010233),

Research Grants Council of the Hong Kong SAR, China (27207025), and UCSC hellman fellowship.

## References

[1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, in: ICLR, 2021.

[2] X. Wang, P. Guo, Y. Zhang, Unsupervised domain adaptation via bidirectional cross-attention transformer, in: ECML-PKDD, Vol. 14173, 2023, pp. 309–325.

[3] Y. Zhang, Z. Zhu, Y. Wang, P. Guo, Y. Zhu, Z. Xu, Bridging the species gap: Vlm-enhanced asymmetric lora difusion for industrial defect detection, IEEE Transactions on Industrial Informatics (2026).

[4] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas, Communication-eficient learning of deep networks from decentralized data, in: AISTATS, 2017, pp. 1273–1282.

[5] S. Zeng, P. Guo, S. Wang, J. Wang, Y. Zhou, L. Qu, Tackling data heterogeneity in federated learning via loss decomposition, in: MICCAI, 2024, pp. 707–717.

[6] P. Guo, Y. Wang, W. Li, M. Liu, M. Li, J. Zheng, L. Qu, Exploring federated pruning for large language models, arXiv:2505.13547 (2025).

[7] W. Zheng, Z. Lin, P. Guo, Y. Zhou, F. Wang, L. Qu, Fedvlmbench: Benchmarking federated fine-tuning of vision-language models, arXiv preprint arXiv:2506.09638 (2025).

[8] P. Guo, S. Zeng, Z. Li, W. Zheng, M. Liu, L. Qu, Fed-grpo: Reward-signal-driven federated group relative policy optimization, arXiv preprint arXiv:2610.11502 (2026).

[9] L. Zhu, Z. Liu, S. Han, Deep leakage from gradients, in: NeurIPS, Vol. 32, 2019.

[10] J. Geiping, H. Bauermeister, H. Dröge, M. Moeller, Inverting gradients-how easy is it to break privacy in federated learning?, in: NeurIPS, Vol. 33, 2020, pp. 16937–16947.

[11] Y. Huang, S. Gupta, Z. Song, K. Li, S. Arora, Evaluating gradient inversion attacks and defenses in federated learning, in: NeurIPS, Vol. 34, 2021, pp. 7232– 7241.

[12] P. Guo, R. Wang, S. Zeng, J. Zhu, H. Jiang, Y. Wang, Y. Zhou, F. Wang, H. Xiong, L. Qu, Exploring the vulnerabilities of federated learning: A deep dive into gradient inversion attacks, IEEE TPAMI 48 (4) (2026) 4810–4826.

[13] A. C. Yao, Protocols for secure computations, in: FOCS, 1982, pp. 160–164.

[14] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, K. Seth, Practical secure aggregation for privacy-preserving machine learning, in: CCS, 2017, pp. 1175–1191.

[15] C. Gentry, A fully homomorphic encryption scheme, Stanford university, 2009.

[16] X. Zhang, A. Fu, H. Wang, C. Zhou, Z. Chen, A privacy-preserving and verifiable federated learning scheme, in: ICC, 2020, pp. 1–6.

[17] C. Dwork, Diferential privacy, in: ICALP, 2006, pp. 1–12.

[18] R. C. Geyer, T. Klein, M. Nabi, Diferentially private federated learning: A client level perspective, arXiv:1712.07557 (2017).

[19] Z. Shen, J. Ye, A. Kang, H. Hassani, R. Shokri, Share your representation only: Guaranteed improvement of the privacy-utility tradeof in federated learning, in: ICLR, 2023.

[20] W. Zhang, X. Zuo, Y. Wang, L. Zhang, Adaptive density clustering with diferential privacy, Pattern Recognition (2026) 113652.

[21] H. B. McMahan, D. Ramage, K. Talwar, L. Zhang, Learning diferentially private recurrent language models, in: ICLR, 2018.

[22] D. Ha, A. M. Dai, Q. V. Le, Hypernetworks, in: ICLR, 2017.

[23] J. Xu, X. Tong, S.-L. Huang, Personalized federated learning with feature alignment and classifier collaboration, in: ICLR, 2023.

[24] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, S. Gelly, Parameter-eficient transfer learning for nlp, in: ICML, 2019, pp. 2790–2799.

[25] L. T. Phong, Y. Aono, T. Hayashi, L. Wang, S. Moriai, Privacy-preserving deep learning: Revisited and enhanced, in: ATIS, 2017, pp. 100–110.

[26] H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, P. Molchanov, See through gradients: Image batch recovery via gradinversion, in: CVPR, 2021, pp. 16337– 16346.

[27] J. Sun, A. Li, B. Wang, H. Yang, H. Li, Y. Chen, Provable defense against privacy leakage in federated learning from representation perspective, arXiv:2012.06043 (2020).

[28] D. Scheliga, P. Mäder, M. Seeland, Precode-a generic model extension to prevent deep gradient leakage, in: WACV, 2022, pp. 1849–1858.

[29] P. Guo, C. Deng, L. Xu, X. Huang, Y. Zhang, Deep multi-task augmented feature learning via hierarchical graph neural network, in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases, Springer, 2021, pp. 538–553.

[30] A. Shamsian, A. Navon, E. Fetaya, G. Chechik, Personalized federated learning using hypernetworks, in: ICML, 2021, pp. 9489–9502.

[31] A. Tashakori, W. Zhang, Z. J. Wang, P. Servati, Semipfl: personalized semisupervised federated learning framework for edge intelligence, IEEE IOT 10 (10) (2023) 9161–9176.

[32] Y. Lin, H. Wang, W. Li, J. Shen, Federated learning with hyper-network—a case study on whole slide image analysis, Scientific Reports 13 (1) (2023) 1724.

[33] P. Guo, S. Zeng, Y. Wang, H. Fan, F. Wang, L. Qu, Selective aggregation for low-rank adaptation in federated learning, in: ICLR, 2025.

[34] C. F. Van Loan, The ubiquitous kronecker product, JACM 123 (1-2) (2000) 85– 100.

[35] B. Zhao, K. R. Mopuri, H. Bilen, idlg: Improved deep leakage from gradients, arXiv:2001.02610 (2020).

[36] X. Li, K. Huang, W. Yang, S. Wang, Z. Zhang, On the convergence of fedavg on non-iid data, in: ICLR, 2020.

[37] H. Yu, S. Yang, S. Zhu, Parallel restarted sgd with faster convergence and less communication: Demystifying why model averaging works for deep learning, in: AAAI, Vol. 33, 2019, pp. 5693–5700.

[38] H. Karimi, J. Nutini, M. Schmidt, Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition, in: ECML-PKDD, 2016, pp. 795–811.

[39] J. Baxter, A model of inductive bias learning, JAIR 12 (2000) 149–198.

[40] G. Cohen, S. Afshar, J. Tapson, A. Van Schaik, Emnist: Extending mnist to handwritten letters, in: IJCNN, 2017, pp. 2921–2926.

[41] H. Xiao, K. Rasul, R. Vollgraf, Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, arXiv:1708.07747 (2017).

[42] A. Krizhevsky, G. Hinton, et al., Learning multiple layers of features from tiny images (2009).

[43] L. N. Darlow, E. J. Crowley, A. Antoniou, A. J. Storkey, Cinic-10 is not imagenet or cifar-10, arXiv:1810.03505 (2018).

[44] P. Guo, S. Zeng, W. Chen, X. Zhang, W. Ren, Y. Zhou, L. Qu, A new federated learning framework against gradient inversion attacks, in: AAAI, Vol. 39, 2025, pp. 16969–16977.

[45] A. Bietti, C.-Y. Wei, M. Dudik, J. Langford, S. Wu, Personalization improves privacy-accuracy tradeofs in federated learning, in: ICML, 2022, pp. 1945–1962.

[46] L. Qu, Y. Zhou, P. P. Liang, Y. Xia, F. Wang, E. Adeli, L. Fei-Fei, D. Rubin, Rethinking architecture design for tackling data heterogeneity in federated learning, in: CVPR, 2022, pp. 10061–10071.