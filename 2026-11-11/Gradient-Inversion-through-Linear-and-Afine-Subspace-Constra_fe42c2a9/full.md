# Gradient Inversion through Linear and Afine Subspace Constraints

Konstantinos Bouros<sup>∗</sup>, Athanasios P. Liavas School of Electrical and Computer Engineering Technical University of Crete

## Abstract

Federated learning enables multiple clients to collaboratively train a machine-learning model without directly sharing their local datasets. However, as prior works have shown, the gradients exchanged during training may reveal sensitive information about client data. Most existing attacks usually optimize dummy inputs to match the true client gradients (Gradient Inversion Attacks), while some attempt to reconstruct them based on the observation that aggregate gradients from a fully connected (FC) layer are a linear combination of its inputs. In this work, we extend this observation by showing that aggregate gradients from an FC layer with the correct scaling are actually an afine combination of its inputs. We evaluate a gradient inversion attack that leverages either known linear-subspace information or its afine-subspace counterpart at each FC layer to improve reconstruction accuracy. Our results show that subspace-based methods can substantially improve reconstruction quality over the chosen baselines in certain settings. We further demonstrate that moderate overestimation of the batch size can generally improve or preserve reconstruction quality, even when the true batch size is unknown. We additionally examine reconstruction under gradient perturbations introduced by a diferential privacy (DP) mechanism, providing insight into the efectiveness of the considered formulations under noisy conditions.

## 1 Introduction

Federated learning allows a set of clients to collaboratively optimize a shared model while keeping their local training data private. In particular, this is achieved by performing the training process locally on clients and sending the shared gradients to a central server for aggregation. However, although this training paradigm avoids direct data sharing, it does not automatically guarantee privacy. Gradients themselves may contain sensitive information about the samples used to compute them.

Zhu et al. [1] first demonstrated that private data can be reconstructed from the corresponding gradients by exploiting this vulnerability. This technique and related methods are now known as Gradient Inversion Attacks (GIAs). The reconstruction accuracy of these attacks tends to be sensitive to the input and batch size, and additional information or a reformulation of the problem is usually required to improve the performance of the attack.

Cocktail Party Attack [2] is one such approach, which frames gradient inversion as a blind source separation (BSS) problem by leveraging the crucial insight that aggregate gradients from a fully connected (FC) layer are a linear combination of its inputs. This method achieves significant improvement of accuracy but essentially relies on extracted information from a single FC layer.

In an attempt to further improve private data reconstruction performance, we propose an algorithm that follows the GIA paradigm and utilizes geometric information from each FC layer in the architecture. In particular, we make the observation that each row of the aggregated weight gradients from an FC layer, divided by the corresponding aggregated bias gradient entry, is an afine combination of the layer inputs. Our algorithm preserves the optimization framework and objective of Geiping et al. [3], while incorporating gradient-induced linear or afine subspaces for each intermediate output as regularizers.

Lastly, although prior work reasonably assumes that the attacker knows the batch size, exact knowledge may not be necessary. In the noiseless setting, the batch size can typically be inferred directly from the rank of an FC layer’s weight gradient matrix due to the aforementioned gradient structure under suitable full-rank conditions. However, exact batch-size information may not always be available to the attacker, particularly when the observed gradients are perturbed. We therefore also evaluate the sensitivity of the attacks to inaccurate batch-size information. Our results show that, within the range of batch-size overestimation considered, reconstruction can remain successful and may occasionally improve. Note that this improvement is measured by retaining only the best B reconstructions, where B is the true batch size, and averaging the evaluation metrics over this subset. This assumes that the attacker can identify and discard poor reconstructions, which is typically straightforward in image-based settings.

Our main contributions include:

• The observation that each row of the aggregated weight gradient of an FC layer, divided by the corresponding nonzero bias-gradient entry, is an afine combination of the layer inputs.

• Diferentiable input and embedding subspace regularizers that exploit the linear or afine structure of FC-layer gradients. These regularizers can potentially be incorporated into optimization-based GIAs targeting architectures that contain FC layers.

• An implementation and evaluation of a gradient inversion attack that exploits these constraints across multiple FC layers, considering diferent architectures, datasets, and gradient perturbation levels. Our implementation builds upon the CPA codebase [2] and is available at https://github.com/kbouros/subspace-gradient-inversion.

• An experimental analysis of how exact and potentially inaccurate batch-size information available to the attacker afects reconstruction performance.

In this work, we focus on the honest-but-curious setting where in-distribution data are not available to the adversary.

## 2 Related Work

In this section, we present an overview of the most relevant prior gradient inversion attacks (GIAs) and low-rank inversion techniques.

## 2.1 Gradient Matching

The first gradient inversion attack was introduced by Zhu et al. [1]. They demonstrated that sensitive training data can be reconstructed by optimizing an input-label pair (x, y) whose gradients closely match the observed gradients. The DLG optimization problem is formulated as

$$
\begin{array} { r l } { \underset { \pmb { x } , \pmb { y } } { \operatorname* { m i n } } } & { D _ { \mathrm { G M } } ( \pmb { x } , \pmb { y } ) , } \\ { D _ { \mathrm { G M } } ( \pmb { x } , \pmb { y } ) : = d \big ( \nabla _ { \theta } \mathcal { L } _ { \pmb { \theta } } ( \pmb { x } , \pmb { y } ) , \nabla _ { \pmb { \theta } } \mathcal { L } _ { \pmb { \theta } } ^ { \mathrm { t r u e } } \big ) , } \end{array}\tag{1}
$$

where d denotes a distance metric between the candidate and observed gradient vectors. Zhu et al. [1] use the $\ell _ { 2 }$ distance for this purpose. Geiping et al. [3] later proposed replacing the $\ell _ { 2 }$ distance with cosine distance, making the optimization insensitive to the gradients

magnitudes. They also incorporated total variation (TV) regularization, which promotes the spatial smoothness commonly observed in natural images.

## 2.2 Generative Models and Batch Norm Statistics

Several subsequent works have sought to improve reconstruction quality, particularly for higherresolution images and larger batches, by incorporating stronger priors into the inversion objective. For example, GradInversion uses the running statistics of batch-normalization (BN) layers as a distributional prior [4], whereas GIAS constrains the reconstruction using a pretrained generative model [5]. However, BN-statistics-based priors are applicable only to architectures that contain BN layers. In practical federated learning settings, the shared model may not include such layers, either for privacy considerations or because they are unsuitable for the task or architecture.

Generative-model-based approaches face a similar practical limitation. They typically require access to a dataset whose distribution is the same or suficiently close to that of the private training data in order to train the generative model. This information may not be available to an attacker, limiting the applicability of these attacks under substantial distribution shift.

More recent methods relax these assumptions in diferent ways. GIFD [6] improves the out-ofdistribution (OOD) flexibility of GAN-based inversion by optimizing intermediate generator features, but it still requires a pretrained GAN that provides a useful prior for the target domain. GI-PIP [7] instead uses an anomaly-detection prior trained on a small auxiliary dataset and is more tolerant to distribution mismatch. However, both methods still depend on auxiliary information and therefore relax, rather than eliminate, the distributional assumptions.

## 2.3 Low-Rank Gradient Inversion

Other methods exploit low-rank structure in fully connected or self-attention gradients to reconstruct private inputs approximately [2, 8] or, under suitable assumptions, exactly [9–11]. CPA [2] models the gradient of a fully connected layer as a linear mixture of its inputs and reformulates inversion as a blind source separation problem.

SPEAR [9] and SPEAR++ [10] show that exact batch reconstruction from fully connected-layer gradients is possible under suitable rank, dimensionality, and sparsity conditions. Both exploit the sparsity induced by ReLU activations to identify the correct low-rank factorization, while SPEAR++ improves the scalability of the original approach. Their applicability, however, remains limited to settings with suficiently sparse activations and favorable layer dimensions, and the exact-reconstruction guarantees do not directly extend to non-sparse activation functions.

DAGER [11] extends low-rank gradient inversion to transformer language models by exploiting information from selected self-attention gradients together with the discrete structure of token embeddings. Earlier low-rank attacks typically rely on efectively extracting information from a single or limited set of layers. TIGER [8] builds upon the low-rank observation and combines subspace information from multiple transformer layers, improving reconstruction quality and robustness to DP noise.

## 2.4 Malicious Server Setting

In the malicious-server setting, the server can deviate from the prescribed federated-learning protocol by modifying the model architecture or parameters to isolate, amplify, or directly expose individual training samples [12–14]. VGIA [15] similarly manipulates model parameters and explicitly exploits the linear structure of FC-layer gradients together with ReLU-induced geometric partitioning to achieve verifiable isolation and reconstruct individual records.

Although these attacks can achieve significantly more efective extraction than passive gradient inversion, they require an active adversary and may be exposed if clients inspect or audit the model they receive. Since this work focuses on the weaker honest-but-curious setting, we do not examine malicious-server attacks further.

## 2.5 Prior Work Limitations

Most prior gradient-inversion attacks do not exploit the linear structure of FC-layer gradients. Moreover, although the underlying gradient equations are well established, the afine-subspace constraints they imply have, to our knowledge, not been explicitly recognized or leveraged in gradient inversion. Among methods that do exploit this structure, SPEAR and $\mathrm { S P E A R + + \ [ 9 , 1 0 ] }$ provide powerful exact-reconstruction attacks under suitable conditions. These methods, however, rely on ReLU-induced sparsity as a key source of information. While ReLU remains widely used, more general threats directly applicable to architectures with non-sparse activations should also be considered.

Other low-rank attacks, such as CPA [2], exploit information from a selected fully connected layer but do not systematically combine constraints from multiple layers. The method closest to our approach is the recent TIGER [8], which uses linear-subspace information from multiple transformer layers but is specialized to the text domain. Unlike TIGER, which replaces fullgradient matching with a subspace-based objective, our approach retains the gradient-matching objective and incorporates linear or afine subspace information as additional regularization in the image setting. More generally, the proposed subspace regularizers may be combined with other existing or future optimization-based inversion methods.

## 3 Subspace-Constrained Gradient Inversion

In this section, we analyze the afine structure of aggregate gradients in FC layers and describe the algorithm in detail for FC networks. In Section 4, we explain how the algorithm can be adapted to convolutional neural networks (CNNs) using an approach similar to that of CPA [2].

## 3.1 Afine Structure of FC Layers

Consider a neural network with L FC layers. Let $\pmb { x } ^ { ( i ) } \in \mathbb { R } ^ { d _ { 0 } }$ denote the i-th input in a batch of size B and define $h _ { 0 } ^ { ( i ) } : = \pmb { x } ^ { ( i ) }$ . For an arbitrary layer $l \in \{ 1 , \ldots , L \}$ , let

$$
\pmb { a } _ { l } ^ { ( i ) } = \pmb { W } _ { l } \pmb { h } _ { l - 1 } ^ { ( i ) } + \pmb { b } _ { l } , \qquad \pmb { h } _ { l } ^ { ( i ) } = \phi _ { l } \Big ( \pmb { a } _ { l } ^ { ( i ) } \Big ) ,
$$

where $W _ { l } \in \mathbb { R } ^ { d _ { l } \times d _ { l - 1 } }$ and $b _ { l } \in \mathbb { R } ^ { d _ { l } }$ are the weights and biases of layer l, respectively. Let the total mean loss over the batch be

$$
\mathcal { L } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \ell ^ { ( i ) } .
$$

Layer-l gradient identities. Fix a layer $l \in \{ 1 , \ldots , L \}$ . Define the backpropagated signal for sample i at layer l as

$$
\delta _ { l } ^ { ( i ) } : = \frac { \partial \mathcal { L } } { \partial \pmb { a } _ { l } ^ { ( i ) } } = \frac { 1 } { B } \frac { \partial \ell ^ { ( i ) } } { \partial \pmb { a } _ { l } ^ { ( i ) } } \in \mathbb { R } ^ { d _ { l } } .
$$

By the chain rule, the per-sample gradients of the layer-l parameters with respect to $\ell ^ { ( i ) }$ satisfy

$$
\nabla _ { W _ { l } } \ell ^ { ( i ) } = \frac { \partial \ell ^ { ( i ) } } { \partial a _ { l } ^ { ( i ) } } { h _ { l - 1 } ^ { ( i ) } } ^ { \top } = B \cdot \delta _ { l } ^ { ( i ) } { h _ { l - 1 } ^ { ( i ) } } ^ { \top } , \quad \nabla _ { b _ { l } } \ell ^ { ( i ) } = \frac { \partial \ell ^ { ( i ) } } { \partial b _ { l } } = \frac { \partial \ell ^ { ( i ) } } { \partial a _ { l } ^ { ( i ) } } = B \cdot \delta _ { l } ^ { ( i ) } .
$$

Because the total loss is the mean over the batch, the observed gradients are

$$
G _ { W _ { l } } : = \nabla _ { W _ { l } } \mathcal { L } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \nabla _ { W _ { l } } \ell ^ { ( i ) } = \sum _ { i = 1 } ^ { B } \delta _ { l } ^ { ( i ) } { h _ { l - 1 } ^ { ( i ) } } ^ { \top } , \quad g _ { b _ { l } } : = \nabla _ { b _ { l } } \mathcal { L } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \nabla _ { b _ { l } } \ell ^ { ( i ) } = \sum _ { i = 1 } ^ { B } \delta _ { l } ^ { ( i ) } .
$$

For each neuron $j \in \{ 1 , \ldots , d _ { l } \}$ , denote the j-th row of $G _ { W _ { l } }$ and the corresponding bias-gradient entry, respectively, by

$$
\pmb { r } _ { j } ^ { \top } : = \left( \pmb { G } _ { \pmb { W } _ { l } } \right) _ { j , : } \in \mathbb { R } ^ { d _ { l - 1 } } , \quad s _ { j } : = \left( \pmb { g } _ { \pmb { b } _ { l } } \right) _ { j } .
$$

Then, the gradient identities become

$$
\pmb { r } _ { j } ^ { \top } = \sum _ { i = 1 } ^ { B } \delta _ { l , j } ^ { ( i ) } \pmb { h } _ { l - 1 } ^ { ( i ) } { } ^ { \top } , \qquad s _ { j } = \sum _ { i = 1 } ^ { B } \delta _ { l , j } ^ { ( i ) } .
$$

Normalized gradients. Assume $s _ { j } \neq 0$ . Define the normalized vector $\begin{array} { r } { { \pmb u } _ { j } : = \frac { { \pmb r } _ { j } } { s _ { j } } } \end{array}$ . Substituting the expressions above yields

$$
\pmb { u } _ { j } = \frac { \sum _ { i = 1 } ^ { B } \delta _ { l , j } ^ { ( i ) } \pmb { h } _ { l - 1 } ^ { ( i ) } } { \sum _ { i = 1 } ^ { B } \delta _ { l , j } ^ { ( i ) } } = \sum _ { i = 1 } ^ { B } \frac { \delta _ { l , j } ^ { ( i ) } } { \sum _ { k = 1 } ^ { B } \delta _ { l , j } ^ { ( k ) } } \pmb { h } _ { l - 1 } ^ { ( i ) } .
$$

Define the coeficients

$$
\alpha _ { j , i } : = \frac { \delta _ { l , j } ^ { ( i ) } } { \sum _ { k = 1 } ^ { B } \delta _ { l , j } ^ { ( k ) } } .
$$

Then,

$$
\sum _ { i = 1 } ^ { B } \alpha _ { j , i } = 1 \quad \mathrm { a n d } \quad \pmb { u } _ { j } = \sum _ { i = 1 } ^ { B } \alpha _ { j , i } \pmb { h } _ { l - 1 } ^ { ( i ) } .
$$

Lemma 1 (Afine-hull characterization) Let

$$
\mathcal { I } = \{ j \in \{ 1 , \dots , d _ { l } \} \mid s _ { j } \neq 0 \} .
$$

For every $j \in \mathcal { I } , u _ { j }$ is an afine combination of the layer-l inputs, and hence

$$
\{ \pmb { u } _ { j } \ | \ j \in \mathcal { I } \} \subseteq \mathrm { a f f } \{ \pmb { h } _ { l - 1 } ^ { ( 1 ) } , \dots , \pmb { h } _ { l - 1 } ^ { ( B ) } \} .
$$

Remark. For the first layer specifically, where $\pmb { h } _ { 0 } ^ { ( i ) } = \pmb { x } ^ { ( i ) }$ , each normalized gradient vector $\mathbf { \Delta } \mathbf { u } _ { j }$ belongs to the afine hull of the original inputs:

$$
\{ \pmb { u } _ { j } \ | \ j \in \mathcal { I } \} \subseteq \mathrm { a f f } \{ \pmb { x } ^ { ( 1 ) } , \ldots , \pmb { x } ^ { ( B ) } \} .
$$

Consequence. The normalized layer-l gradient rows $\{ \pmb { u } _ { j } \ | \ j \in \mathcal { I } \}$ lie in the afine hull of the batch layer inputs. Therefore, under suitable rank conditions and provided that $| { \mathcal { I } } | \geq B$ , the afine subspace containing the layer inputs can be estimated by computing the afine span of $\{ { \pmb u } _ { j } \} _ { j \in \mathcal { I } }$

## 3.2 Subspace Estimation

First, we describe the subspace estimation procedure for the linear subspace induced by the gradients of the FC layers. For each attacked layer l, we apply SVD to the observed weight gradient $G _ { W _ { l } }$ and extract an orthonormal basis through the origin, represented by

$$
P _ { l } \in \mathbb { R } ^ { d _ { l - 1 } \times B } .
$$

The SVD is computed with rank $k = B ,$ , corresponding to the batch size.

The afine case is handled similarly. We first construct the ratio-point matrix $U ,$ whose rows correspond to the points $\{ \pmb { u } _ { j } \ | \ j \in \mathcal { I } \}$ defined earlier. In practice, to avoid numerical instability caused by near-zero denominators, we define

$$
\mathcal { T } = \left\{ j \in \{ 1 , \dots , d _ { l } \} \ : \middle | \ : | s _ { j } | > 1 0 ^ { - 8 } \right\} .
$$

We then estimate the afine subspace <sup>1</sup> using a centered, reliability-weighted SVD with rank $k = B - 1$ . Each row $u _ { j }$ is assigned the unnormalized weight

$$
\tilde { w } _ { j } = | s _ { j } | ^ { 2 } .
$$

These weights are normalized over the samples as

$$
w _ { j } = \frac { \tilde { w _ { j } } } { \sum _ { i \in \mathcal { I } } \tilde { w _ { i } } } .
$$

The normalized weights determine the contribution of each sample to the SVD fit. Consequently, samples with a larger denominator magnitude, $| s _ { j } |$ , exert greater influence on the estimated afine hull. This weighting scheme reduces potential numerical instability introduced by the division and is necessary in the noisy setting. The resulting afine hull is characterized by

• the afine ofset $\pmb { \mu } _ { l } ^ { \top } \in \mathbb { R } ^ { d _ { l - 1 } }$ 2

• the orthonormal basis matrix $P _ { l } \in \mathbb { R } ^ { d _ { l - 1 } \times ( B - 1 ) }$

## 3.3 Subspace Regularizers

Let $h _ { l - 1 } ^ { ( i ) } \in \mathbb { R } ^ { 1 \times d _ { l - 1 } }$ denote the input activation to layer l for sample i. For this layer, let the estimated linear/afine hull be

$$
\mathcal { H } _ { l } = \left\{ \pmb { \mu } _ { l } + \pmb { a } \pmb { P } _ { l } ^ { \top } : \pmb { a } \in \mathbb { R } ^ { 1 \times k } \right\} ,
$$

where ${ \pmb \mu } _ { l } = { \bf 0 } , k = B$ for the linear case and $k = B - 1$ for the afine case. Then, the projection of $h _ { l - 1 } ^ { ( i ) }$ onto $\mathcal { H } _ { l }$ is given by

$$
\begin{array} { r } { \Pi _ { \mathcal { H } _ { l } } \left( \pmb { h } _ { l - 1 } ^ { ( i ) } \right) = \pmb { \mu } _ { l } + \left( \pmb { h } _ { l - 1 } ^ { ( i ) } - \pmb { \mu } _ { l } \right) P _ { l } \pmb { P } _ { l } ^ { \top } . } \end{array}
$$

Thus, for a batch of activations

$$
\pmb { H } _ { l - 1 } = \left\{ \pmb { h } _ { l - 1 } ^ { ( i ) } \right\} _ { i = 1 } ^ { B } ,
$$

we define the subspace regularizer at layer l as the average squared distance to that subspace:

$$
\mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } ( { \cal H } _ { l - 1 } ) : = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left\| \pmb { h } _ { l - 1 } ^ { ( i ) } - \Pi _ { \mathcal { H } _ { l } } \Big ( \pmb { h } _ { l - 1 } ^ { ( i ) } \Big ) \right\| ^ { 2 } ,
$$

$$
\mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } : = \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } ( H _ { l - 1 } ) .
$$

Note that, if no linear/afine hull was estimated for layer l (insuficient rows), we set:

$$
\mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } ( \boldsymbol { H } _ { l - 1 } ) : = 0 , \qquad \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } : = \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } ( \boldsymbol { H } _ { l - 1 } ) .
$$

These regularizers penalize deviations of the layer inputs from the learned low-dimensional subspace structures. The corresponding hyperparameters will balance gradient matching performance and adherence to that low-rank structure.

## 3.4 Objective Function

We identify two main approaches for adapting the objective function proposed by Geiping et al. [3] to incorporate subspace constraints:

• Hard Subspace Input Constraint,

• Soft Subspace Input Constraint

## 3.4.1 Hard Subspace Input Constraint

In the hard subspace constraint setting, we use the estimated linear/afine hull obtained from the first-layer gradients to reparameterize the batch inputs as

$$
\pmb { x } = \pmb { \mu } _ { 1 } + \pmb { A } \pmb { P } _ { 1 } ^ { \top } ,
$$

where $\pmb { A } \in \mathbb { R } ^ { B \times k }$ contains the coeficients optimized for each sample and k is the rank used in the SVD. This reparameterization reduces the number of optimization variables from $\boldsymbol { B } \cdot \boldsymbol { d } _ { 0 }$ to $B \cdot k$ . The resulting optimization problem, which combines the hard subspace reparameterization with subspace regularizers for the intermediate layers, is:

$$
\operatorname* { m i n } _ { A , y } D _ { \mathrm { G M } } ( \pmb { x } , \pmb { y } ) + \lambda _ { \mathrm { T V } } \mathcal { R } _ { \mathrm { T V } } ( \pmb { x } ) + \lambda _ { \mathrm { s u b - i n t e r } } \sum _ { l = 2 } ^ { L } \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } ,\tag{2}
$$

where $D _ { \mathrm { G M } }$ uses the cosine distance metric and $\lambda _ { \mathrm { T V } } , \lambda _ { \mathrm { s u b - i n t e r } } \geq 0$ control the strength of the regularizers.

## 3.4.2 Soft Subspace Input Constraint

In this setting, we use the estimated linear/afine hull from the first-layer gradients as a regularizer rather than reparameterizing the batch inputs. Additionally, we retain the subspace regularizers for the intermediate layers. The resulting optimization problem is

$$
\operatorname* { m i n } _ { \substack { x \in [ 0 , 1 ] ^ { d _ { 0 } } , y } } D _ { \mathrm { G M } } ( \pmb { x } , \pmb { y } ) + \lambda _ { \mathrm { T V } } \mathcal { R } _ { \mathrm { T V } } ( \pmb { x } ) + \lambda _ { \mathrm { s u b - i n p } } \mathcal { R } _ { \mathrm { s u b } } ^ { ( 1 ) } ( \pmb { x } ) + \lambda _ { \mathrm { s u b - i n t e r } } \sum _ { l = 2 } ^ { L } \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } ,\tag{3}
$$

where $D _ { \mathrm { G M } }$ uses the cosine distance metric and $\lambda _ { \mathrm { T V } } , \lambda _ { \mathrm { s u b - i n p } } , \lambda _ { \mathrm { s u b - i n t e r } } \geq 0$ control the strength of the regularizers.

## 3.5 Initialization and Projection

The objectives in Equations (2) and (3) are highly non-convex, and, thus, the objective value to which they converge depends on the initialization of the optimization variables. Therefore, it is meaningful to ensure randomness in the initialization procedure and potentially perform multiple restarts.

In the soft subspace case, the inputs x and the label distribution y can be initialized based on the standard normal distribution. During optimization, we enforce valid pixel values by projecting x onto the box constraint $[ 0 , 1 ] ^ { d _ { 0 } }$ at each iteration.

In the hard-subspace setting, however, the optimization variables for the inputs are the coeficients A. Directly initializing A requires selecting an appropriate standard deviation, and it is unclear how to achieve suficient diversity while ensuring that the corresponding inputs remain within, or close to, the box constraints. Alternatively, we could randomly initialize x and project it onto the low-rank subspace to obtain initial coeficients. However, when the batch size is suficiently small, the large dimensionality gap between the original space and the low-rank subspace causes the projected points to concentrate in a small region.

To obtain initializations that are both diverse and feasible, we use a Hit-and-Run procedure. The algorithm first identifies a feasible starting point and then randomly explores the admissible region. Finally, we select a well-separated subset of these samples using greedy farthest-point sampling.

In the hard-subspace setting, directly projecting x onto the box $[ 0 , 1 ] ^ { d _ { 0 } }$ may violate the afinesubspace constraint. Thus, we can either enforce the box constraint through a regularization term during optimization or avoid projection entirely and apply it only to the final reconstructed inputs.

## 4 Extension to CNNs

Subspace constraints can only be applied to FC layers, and if a reparameterization is desirable, then our previous approach cannot be applied directly to CNNs. However, each version of our proposed algorithm can be extended to CNNs based on feature inversion (FI) as seen in CPA [2], while the soft subspace case can also bypass FI. In particular, we can identify three approaches for extending our algorithm to CNNs, of which the first two utilize FI:

• Hard Subspace Input Constraint

• Soft Subspace Input Constraint

• Soft Subspace Intermediate Constraint

## 4.1 Hard Subspace Input Constraint

Following CPA, we first recover the embedding vector $\boldsymbol { z } \in \mathbb { R } ^ { B \times d }$ , which is produced as input to the CNN’s first FC layer, and then use this vector in feature inversion (FI). To recover z, we isolate the fully connected component of the CNN and treat it as a separate model on which the attack is performed. Because the target variable is now an embedding rather than an image, the total variation (TV) prior is no longer applicable. Instead, we can use the sparsity and non-negativity properties of embeddings produced by ReLU activations. These properties are encouraged using the following regularizers:

$$
\mathcal { R } _ { \ell _ { 1 } } ( z ) = \frac { 1 } { B d } \sum _ { b = 1 } ^ { B } \sum _ { j = 1 } ^ { d } | z _ { b , j } |
$$

and

$$
\mathcal { R } _ { \mathrm { n v } } ( z ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \| \mathrm { R e L U } ( - z _ { b , : } ) \|
$$

The $\ell _ { 1 }$ regularizer promotes sparsity, whereas ${ \mathcal { R } } _ { \mathrm { n v } }$ penalizes negative entries, encouraging the recovered embeddings to remain consistent with the output of a ReLU activation.

Additionally, drawing inspiration from decorrelation objectives used in blind source separation (BSS) and CPA, we utilize a milder decorrelation regularizer based on squared cosine similarity. This term encourages distinct recovered embeddings to be approximately orthogonal without imposing an overly aggressive separation constraint. This prior is useful for the embedding reconstruction since the original embeddings are high-dimensional and tend to have low pairwise correlation. The regularizer is defined as

$$
\mathcal { R } _ { \mathrm { d e c o r } } ( z ) = \mathbb { E } _ { i \neq j } \left[ \mathrm { C S } ( z _ { i } , z _ { j } ) ^ { 2 } \right] ,
$$

where

$$
\mathrm { C S } \left( z _ { i } , z _ { j } \right) = \frac { z _ { i } ^ { \top } z _ { j } } { \| z _ { i } \| \left\| z _ { j } \right\| } .
$$

Thus, the embedding-recovery problem is formulated as

$$
\begin{array} { r l } { \underset { \mathbf { \boldsymbol { A } } , \boldsymbol { y } } { \mathop { \operatorname* { m i n } } } } & { D _ { \mathrm { G M } } ( \boldsymbol { z } , \boldsymbol { y } ) + \lambda _ { \mathrm { n v } } \mathcal { R } _ { \mathrm { n v } } ( \boldsymbol { z } ) + \lambda _ { \boldsymbol { \ell } _ { 1 } } \mathcal { R } _ { \boldsymbol { \ell } _ { 1 } } ( \boldsymbol { z } ) } \\ & { + \lambda _ { \mathrm { d e c o r } } \mathcal { R } _ { \mathrm { d e c o r } } ( \boldsymbol { z } ) + \lambda _ { \mathrm { s u b - i n t e r } } \displaystyle \sum _ { l = 2 } ^ { L } \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } , } \end{array}\tag{4}
$$

where A are the optimization variables in coeficient space, $D _ { \mathrm { G M } }$ uses the cosine distance metric for the corresponding gradients of the FC component only, and $\lambda _ { \mathrm { n v } } , \lambda _ { \ell _ { 1 } } , \lambda _ { \mathrm { d e c o r } } , \lambda _ { \mathrm { s u b - i n t e r } } \geq 0$ control the strength of the regularizers.

## 4.2 Soft Subspace Input Constraint

In this setting, we follow a similar approach by recovering the embedding vector z. However, we optimize directly in the embedding space and use the estimated linear/afine hull from the first FC-layer gradients as a regularizer. The resulting optimization problem is

$$
\begin{array} { r l } { \displaystyle \underset { z \in \mathbb { R } _ { + } , y } { \operatorname* { m i n } } D _ { \mathrm { G M } } ( z , y ) + \lambda _ { \ell _ { 1 } } \mathcal { R } _ { \ell _ { 1 } } ( z ) + \lambda _ { \mathrm { d e c o r } } \mathcal { R } _ { \mathrm { d e c o r } } ( z ) } & { } \\ { + \lambda _ { \mathrm { s u b - i n p } } \mathcal { R } _ { \mathrm { s u b } } ^ { ( 1 ) } ( z ) + \lambda _ { \mathrm { s u b - i n t e r } } \displaystyle \sum _ { l = 2 } ^ { L } \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } , } \end{array}\tag{5}
$$

where $D _ { \mathrm { G M } }$ uses the cosine distance metric and $\lambda _ { \ell _ { 1 } }$ , λ<sub>decor</sub>, $\lambda _ { \mathrm { s u b - i n p } } , \lambda _ { \mathrm { s u b - i n t e r } } \geq 0$ control the strength of the regularizers. The constraint $z \in \mathbb { R } _ { + }$ is enforced by projecting the optimized embedding onto the positive orthant at each iteration. This projection-based constraint is well suited to the noiseless setting. In the presence of noise, however, it is preferable to relax the constraint by removing the projection and reintroducing the non-negativity regularizer.

## 4.3 Soft Subspace Intermediate Constraint

Lastly, we consider an alternative approach that bypasses FI and optimizes directly in the input space x. In particular, the attack can incorporate subspace regularization at the fully connected layers of the model while optimizing over the image pixel space, leading to the optimization problem

$$
\operatorname* { m i n } _ { \substack { { \boldsymbol x } \in [ 0 , 1 ] ^ { d _ { 0 } } , { \boldsymbol y } } } D _ { \mathrm { G M } } ( { \boldsymbol x } , { \boldsymbol y } ) + \lambda _ { \mathrm { T V } } \mathcal { R } _ { \mathrm { T V } } ( { \boldsymbol x } ) + \lambda _ { \mathrm { s u b - i n t e r } } \sum _ { l \in I } \mathcal { R } _ { \mathrm { s u b } } ^ { ( l ) } ,\tag{6}
$$

where $D _ { \mathrm { G M } }$ uses the cosine distance metric over the gradients of the whole model. The hyperparameters λ<sub>TV</sub>, $\lambda _ { \mathrm { s u b - i n t e r } } \geq 0$ control the strengths of the regularizers, and

$$
I = \{ i \in \{ 1 , \ldots , L \} \mid \mathrm { l a y e r } \ i \ \mathrm { i s \ f u l l y \ c o n n e c t e d } \} .
$$

Although this formulation provides a viable alternative, it has some important limitations. First, the subspace information is obtained from layers located far from the input, making it more dificult for the regularizers to guide the optimization toward the original input. Second, the formulation does not address the inherent dificulties of gradient matching for high-resolution images and deep networks, for which the objective $D _ { \mathrm { G M } }$ can be particularly challenging to optimize. In contrast, the two previous approaches first optimize over a smaller portion of the model and utilize the subspace information more efectively, which can lead to better results. For these reasons, we will not focus on this approach.

## 4.4 Initialization

The label distribution y is initialized from a standard normal distribution in all settings. To account for the non-negativity of the embeddings, z is initialized from a uniform distribution over [0, 1]. In the hard-subspace setting, the initialized embeddings are subsequently projected onto the low-rank subspace, from which the corresponding initial coeficients A are obtained.

In this case, we do not employ the previously introduced Hit-and-Run procedure, since the embeddings are not constrained to a bounded domain. Consequently, the batch members may be initialized at identical or nearby points, limiting the efectiveness of randomized restarts. Nevertheless, the decorrelation regularizer encourages the optimized embeddings to diverge during optimization, reducing the impact of their similar initializations.

## 5 Experiments

## 5.1 Experimental Setup

We evaluate our approaches with both the new afine and the known linear low-rank structure. We first define the experimental setup, including the target models and the baseline attacks against which we compare.

## 5.1.1 Models and Datasets

We evaluate our attack on both fully connected (FC) and convolutional neural network (CNN) models trained for image classification. For the FC experiments, we use an L-layer network, denoted $\mathrm { F C } \ – L ,$ consisting of (L−1) Linear(512)–Activation blocks followed by a Linear(k) output layer for k-class classification. We focus on the FC-3 architecture and consider two activation functions, ReLU and LeakyReLU. In the main results with accurate batch-size information, we report performance for both activation functions. For experiments involving inaccurate batch-size information or the noisy setting, we restrict our evaluation to ReLU.

We train the FC-3 model on CIFAR-10 and Tiny-ImageNet for 20 epochs using the Adam optimizer with a learning rate of $1 0 ^ { - 3 }$ . For the CNN experiments, we use a pretrained VGG-16 model from torchvision and evaluate it on ImageNet.

## 5.1.2 Evaluation

We focus on the SGD setting and evaluate the attacks under diferent assumptions about batch-size availability. Unless stated otherwise, labels are assumed unknown.

We first consider the case in which the true batch size is known and then study robustness to batch-size inaccuracies. Let B denote the true batch size and $\widehat { B }$ its estimate. An inaccurate batch-size estimate afects the entire attack pipeline, including the rank used for subspace estimation and the number of inputs reconstructed during optimization. For $\widehat { B } < B$ , we match the reconstructions to the $\widehat { B }$ most similar ground-truth inputs and average the resulting metrics. For ${ \widehat { B } } > B _ { \ \mathrm { i } }$ , we retain the best B reconstructions, match them to the ground-truth batch, and report the corresponding average metrics.

For the CNN setting, we directly compare the reconstructed embeddings. CPA [2] has already demonstrated the efectiveness of combining embedding reconstruction with feature inversion and GMA. Therefore, a direct comparison of the reconstructed latent representation, z, provides a more meaningful evaluation. We evaluate each attack over 5 data batches and report the average LPIPS score for image reconstruction using the FC-3 model, and the average cosine similarity (CS) between the reconstructed and ground-truth embeddings for the VGG-16 model.

![](images/07b0a462d316d0af63c2062c99c0b4171a2a464e00221ad83e0bf99787b519ea.jpg)  
Figure 1: Comparison of a batch of 16 CIFAR-10 images recovered from IG, CPA and Hard Afine attacks for the FC-3 model with ReLU. The quality of images recovered by the Hard Afine variant is better than the baselines but produces a duplicate image.

All attacks are run for 25K optimization iterations and, when applicable, with two independent initializations. We report the iterate achieving the lowest objective value across all iterations and initializations. CPA is excluded from the multiple-initialization setting since its initialization is deterministic.

## 5.1.3 Baselines

As discussed earlier, we focus on the honest-but-curious setting, in which in-distribution data are not available, and batch normalization (BN) is not used. We therefore exclude methods that rely on generative models, BN statistics, or an active adversary. The most relevant baselines for our setting are IG [3] and CPA [2] for image data.

Although SPEAR and SPEAR++ [9, 10] are applicable in some of these settings, we primarily consider approximate, optimization-based attacks that can be extended to arbitrary activation functions. This choice is particularly meaningful for evaluating the threat posed in settings where exact gradient-inversion methods are not applicable. For each baseline, we use the bestperforming hyperparameter configuration reported in the corresponding original work. Further hyperparameter details are provided in Appendix A.

## 5.1.4 Hardware

All experiments were conducted in Google Colab. Experiments involving fully connected models were run on an NVIDIA L4 GPU, whereas experiments with the VGG-16 convolutional model were performed on an NVIDIA A100 GPU.

## 5.2 Main Results

## 5.2.1 Accurate Batch Size

Results for FC-3. We first present results for FC-3 models with ReLU and LeakyReLU activations trained on CIFAR-10 and Tiny-ImageNet. Tables 1 and 2 report the LPIPS scores comparing the baseline methods with the proposed subspace approaches when the true batch size is known, for batch sizes in {16, 32, 64, 128, 256}.

As shown in Table 1, the proposed subspace methods exhibit similar behavior for both ReLU and LeakyReLU activations, with the choice of activation function not meaningfully afecting the relative performance of the evaluated methods. Across both settings, the subspace methods can outperform the baselines on CIFAR-10 for batch sizes up to 128. At a batch size of 16, the Soft Linear variant performs best for the ReLU case, whereas for batch sizes of 64 and 128, the Soft Afine variant maintains an advantage over the baselines under both activation functions, unlike the linear variants in some cases. At smaller batch sizes, the subspace methods also tend to reconstruct images with higher visual fidelity than the baselines, although they may become trapped in local minima and produce duplicate reconstructions. In contrast, CPA tends to recover a larger number of distinct images more consistently, but with lower reconstruction quality, which can lead to higher (i.e., worse) LPIPS scores. An example of this behavior is shown in Figure 1. As discussed in the next section, this limitation can be mitigated by overestimating the batch size.

Table 1: LPIPS ↓ scores on CIFAR-10 for the FC-3 model with accurate batch size.
<table><tr><td colspan="5"></td><td colspan="5">LeakyReLU</td></tr><tr><td>Attack</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td></tr><tr><td>IG</td><td>0.206</td><td>0.350</td><td>0.482</td><td>0.565</td><td>0.597</td><td>0.153</td><td>0.252</td><td>0.481</td><td>0.574</td><td>0.599</td></tr><tr><td>CPA</td><td>0.162</td><td>0.205</td><td>0.286</td><td>0.517</td><td>0.577</td><td>0.163</td><td>0.210</td><td>0.305</td><td>0.523</td><td>0.580</td></tr><tr><td>Soft Linear</td><td>0.077</td><td>0.135</td><td>0.358</td><td>0.504</td><td>0.580</td><td>0.107</td><td>0.115</td><td>0.263</td><td>0.516</td><td>0.585</td></tr><tr><td>Hard Linear</td><td>0.135</td><td>0.277</td><td>0.450</td><td>0.544</td><td>0.587</td><td>0.057</td><td>0.216</td><td>0.448</td><td>0.577</td><td>0.597</td></tr><tr><td>Soft Affine</td><td>0.161</td><td>0.137</td><td>0.206</td><td>0.474</td><td>0.587</td><td>0.087</td><td>0.087</td><td>0.139</td><td>0.481</td><td>0.581</td></tr><tr><td>Hard Affine</td><td>0.119</td><td>0.118</td><td>0.307</td><td>0.484</td><td>0.587</td><td>0.089</td><td>0.088</td><td>0.216</td><td>0.520</td><td>0.595</td></tr></table>

![](images/ed3e6de09a3331e4a1ce35a538df645ccb7ad58d9d1670c4f78449ff90703b04.jpg)  
Figure 2: Comparison of a subset of a batch of 128 Tiny-ImageNet images recovered from IG, CPA and Soft Afine attacks for the FC-3 model with LeakyReLU. The quality of most images recovered by the Soft Afine variant is significantly better than the baselines.

Table 2 presents the results on Tiny-ImageNet, where the choice of activation function has a more meaningful efect on performance. With ReLU activations, the subspace-based methods perform substantially worse than CPA for batch sizes of 32 and above. This behavior can be attributed to the very small number of active neurons in the second layer (≤ 10), which falls below the batch size. As a result, the corresponding linear/afine hull cannot be estimated reliably, limiting the amount of useful information available to the subspace-based attacks. In contrast, LeakyReLU avoids the issue of inactive neurons, allowing linear/afine hulls to be estimated from all layers. In this setting, the Soft Linear and Soft Afine variants are particularly efective and consistently outperform the baselines across all evaluated batch sizes.

Table 2: LPIPS ↓ scores on Tiny-ImageNet for the FC-3 model with accurate batch size.
<table><tr><td></td><td colspan="5">ReLU</td><td colspan="5">LeakyReLU</td></tr><tr><td>Attack</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td><td>16</td><td>32</td><td>64</td><td>128</td><td>256</td></tr><tr><td>IG</td><td>0.469</td><td>0.634</td><td>0.703</td><td>0.726</td><td>0.739</td><td>0.115</td><td>0.192</td><td>0.329</td><td>0.406</td><td>0.556</td></tr><tr><td>CPA</td><td>0.156</td><td>0.182</td><td>0.230</td><td>0.275</td><td>0.387</td><td>0.156</td><td>0.182</td><td>0.229</td><td>0.274</td><td>0.394</td></tr><tr><td>Soft Linear</td><td>0.083</td><td>0.616</td><td>0.682</td><td>0.701</td><td>0.704</td><td>0.030</td><td>0.037</td><td>0.068</td><td>0.127</td><td>0.355</td></tr><tr><td>Hard Linear</td><td>0.116</td><td>0.688</td><td>0.768</td><td>0.770</td><td>0.728</td><td>0.037</td><td>0.053</td><td>0.135</td><td>0.263</td><td>0.537</td></tr><tr><td>Soft Affine</td><td>0.186</td><td>0.366</td><td>0.563</td><td>0.664</td><td>0.685</td><td>0.054</td><td>0.062</td><td>0.067</td><td>0.092</td><td>0.331</td></tr><tr><td>Hard Affine</td><td>0.149</td><td>0.474</td><td>0.597</td><td>0.681</td><td>0.690</td><td>0.067</td><td>0.102</td><td>0.087</td><td>0.174</td><td>0.599</td></tr></table>

Table 3: CS ↑ scores on embeddings of ImageNet for VGG-16 with and without known labels. Columns are grouped by the batch size B; within each group, we compare attacks without and with access to the ground-truth labels.
<table><tr><td></td><td colspan="2"> $B = { \bf 1 6 }$ </td><td colspan="2"> $B = 3 2$ </td><td colspan="2"> $B = { \bf 6 4 }$ </td><td colspan="2"> $B = 1 2 8$ </td><td colspan="2"> $B = 2 5 6$ </td></tr><tr><td>Attack</td><td>Unknown</td><td>Known</td><td>Unknown</td><td>Known</td><td>Unknown</td><td>Known</td><td>Unknown</td><td>Known</td><td>Unknown</td><td>Known</td></tr><tr><td>CPA</td><td>0.948</td><td>0.948</td><td>0.921</td><td>0.921</td><td>0.922</td><td>0.922</td><td>0.933</td><td>0.933</td><td>0.919</td><td>0.919</td></tr><tr><td>Soft Linear</td><td>0.776</td><td>0.751</td><td>0.746</td><td>0.717</td><td>0.741</td><td>0.738</td><td>0.691</td><td>0.728</td><td>0.658</td><td>0.724</td></tr><tr><td>Hard Linear</td><td>0.922</td><td>0.998</td><td>0.863</td><td>0.992</td><td>0.791</td><td>0.991</td><td>0.676</td><td>0.982</td><td>0.614</td><td>0.957</td></tr><tr><td>Soft Affine</td><td>0.879</td><td>1.000</td><td>0.777</td><td>0.864</td><td>0.727</td><td>0.908</td><td>0.621</td><td>0.732</td><td>0.576</td><td>0.702</td></tr><tr><td>Hard Affine</td><td>0.964</td><td>1.000</td><td>0.888</td><td>0.923</td><td>0.792</td><td>0.926</td><td>0.678</td><td>0.761</td><td>0.622</td><td>0.723</td></tr></table>

Results for VGG-16. We then evaluate embedding reconstruction for the VGG-16 model pretrained on ImageNet. Table 3 compares CPA with the proposed subspace-based attacks under both unknown and known label settings, assuming that the true batch size is available to the attacker. We consider batch sizes $B \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ . Although access to the ground-truth labels may be unrealistic, an attacker may be able to infer them in some cases. More importantly, the known-label setting provides a useful controlled comparison for assessing how strongly label information can improve the proposed attacks in this setting. In contrast, CPA does not use label information, and its performance is therefore identical in the known and unknown label cases.

When labels are unknown, CPA achieves the highest reconstruction score for all batch sizes except $B = 1 6$ , where Hard Afine performs best. The performance of all methods generally decreases as the batch size grows, but this degradation is substantially more pronounced for the proposed methods in the unknown-label setting. Providing the true labels substantially improves the subspace-based attacks, especially the hard variants. Hard Linear outperforms CPA across all batch sizes, achieving scores above 0.99 up to $B = 6 4$ and remaining strong at $B = 1 2 8$ and 256 with 0.982 and 0.957, respectively. Soft Afine and Hard Afine perform best at $B = 1 6$ , reaching 1.000, but degrade more rapidly as the batch size increases. In contrast, Soft Linear benefits only modestly from label information and remains below CPA across all batch sizes. Overall, label information is particularly beneficial to most subspace methods, with Hard Linear showing the greatest robustness to increasing batch size.

## 5.2.2 Inaccurate Batch Size

To assess the sensitivity of the reconstruction algorithms to batch-size inaccuracies, we evaluate image reconstruction for the FC-3 ReLU model trained on CIFAR-10 and embedding reconstruction for VGG-16 pretrained on ImageNet.

Results for FC-3. Table 4 reports the FC-3 results when the attacker either underestimates or overestimates the true batch size. Comparing these results with the exact-batch setting in Table 1 shows that underestimating the batch size generally degrades reconstruction quality across all methods. For $B \in \{ 1 6 , 3 2 , 6 4 \}$ , even a relatively small underestimate consistently increases LPIPS. At $B = 1 2 8$ , several methods appear to improve when $\hat { B } = 1 1 2 !$ ; however, this comparison is biased because only 112 reconstructions are evaluated. When compared instead with the best 112 reconstructions obtained under the exact-batch setting, underestimating the

Table 4: LPIPS ↓ scores on CIFAR-10 for the FC-3 model under inexact batch-size assumptions. Columns are grouped by the true batch size B; within each group, the attack assumes batch size Bb.
<table><tr><td></td><td colspan="2">True  $B = { \bf 1 6 }$ </td><td colspan="2">True  $B = 3 2$ </td><td colspan="2">True  $B = { \bf 6 4 }$ </td><td colspan="2">True  $B = { \bf 1 2 8 }$ </td></tr><tr><td>Attack</td><td> $\widehat { B } = 1 2$ </td><td> ${ \widehat { B } } = 3 2$ </td><td> ${ \widehat { B } } = 2 6$ </td><td> $\widehat { B } = 6 4$ </td><td> $\widehat { B } = 5 6$ </td><td> $\widehat { B } = 9 6$ </td><td> $\widehat { B } = 1 1 2$ </td><td> $\widehat { B } = 1 6 0$ </td></tr><tr><td>IG</td><td>0.292</td><td>0.155</td><td>0.374</td><td>0.288</td><td>0.491</td><td>0.452</td><td>0.557</td><td>0.555</td></tr><tr><td>CPA</td><td>0.192</td><td>0.162</td><td>0.231</td><td>0.202</td><td>0.299</td><td>0.276</td><td>0.525</td><td>0.501</td></tr><tr><td>Soft Linear</td><td>0.307</td><td>0.053</td><td>0.327</td><td>0.059</td><td>0.377</td><td>0.281</td><td>0.506</td><td>0.496</td></tr><tr><td>Hard Linear</td><td>0.365</td><td>0.088</td><td>0.400</td><td>0.109</td><td>0.467</td><td>0.373</td><td>0.535</td><td>0.521</td></tr><tr><td>Soft Affine</td><td>0.236</td><td>0.034</td><td>0.221</td><td>0.043</td><td>0.261</td><td>0.200</td><td>0.452</td><td>0.483</td></tr><tr><td>Hard Affine</td><td>0.161</td><td>0.042</td><td>0.211</td><td>0.045</td><td>0.318</td><td>0.209</td><td>0.470</td><td>0.505</td></tr></table>

Table 5: CS ↑ scores on embeddings of ImageNet for VGG-16 without known labels under the overestimated batch-size assumption. For each true batch size B, the attack assumes a larger batch size $\widehat { B }$
<table><tr><td>Attack</td><td> $B = { \bf 1 6 }$   ${ \widehat { B } } = 3 2$ </td><td> $B = 3 2$   $\widehat { B } = 6 4$ </td><td> $B = { \bf 6 4 }$   ${ \widehat { B } } = 9 6$ </td><td> $B = { \bf 1 2 8 }$   $\widehat { B } = 1 6 0$ </td><td> $B = 2 5 6$   $\widehat { B } = 3 0 0$ </td></tr><tr><td>CPA</td><td>0.975</td><td>0.963</td><td>0.950</td><td>0.952</td><td>0.918</td></tr><tr><td>Soft Linear</td><td>0.883</td><td>0.874</td><td>0.788</td><td>0.727</td><td>0.695</td></tr><tr><td>Hard Linear</td><td>0.999</td><td>0.985</td><td>0.880</td><td>0.717</td><td>0.631</td></tr><tr><td>Soft Affine</td><td>0.979</td><td>0.954</td><td>0.829</td><td>0.740</td><td>0.702</td></tr><tr><td>Hard Affine</td><td>0.991</td><td>0.965</td><td>0.846</td><td>0.705</td><td>0.634</td></tr></table>

batch size again leads to worse performance.

Overestimation is less harmful and often improves reconstruction quality, particularly for the subspace-based methods at small and moderate batch sizes. For example, at $B = 1 6 .$ , Soft Afine improves from 0.161 with the exact batch size to 0.034 when $\hat { B } = 3 2$ , while Soft Linear improves from 0.077 to 0.053. Similar gains are observed at $B = 3 2$ and 64. These improvements are largely because the additional reconstruction slots reduce the impact of duplicate solutions and increase the chance of recovering all samples in the batch. The extra outputs, however, are mostly duplicates for the subspace methods and noise for CPA due to its decorrelation constraint. CPA shows little benefit from overestimation at $B = 1 6$ and 32, with larger gains appearing at $B = 6 4$ and 128. Overall, the results show a clear asymmetry: underestimating the batch size is consistently detrimental, whereas moderate overestimation is generally more robust and can even improve reconstruction quality.

Results for VGG-16. For VGG-16, we evaluate only the overestimated batch-size setting, since the FC-3 experiments already show that underestimation consistently degrades reconstruction quality.

Table 5 shows that overestimating the batch size again improves reconstruction performance for nearly all methods compared with the unknown-label exact-batch results in Table 3. The gains are generally larger for the subspace-based attacks, particularly Soft Afine. At small batch sizes, this makes the subspace methods highly competitive even without label knowledge, with Hard Linear achieving the best scores at $B = 1 6$ and 32. However, CPA degrades less as the true batch size increases and achieves the highest scores for $B \geq 6 4 .$ . Thus, while overestimating the batch size can substantially benefit the subspace-based attacks, it does not eliminate their stronger degradation at larger batch sizes.

## 5.3 Defenses

Gradient perturbation setup. Diferential privacy (DP) is a popular approach for mitigating privacy leakage in federated learning by providing formal guarantees on the influence of individual training examples on the learned model [16]. In DP-SGD, per-sample gradients are clipped to a predefined norm $C ,$ after which Gaussian noise is added to the aggregated gradient. This introduces a privacy-utility trade-of, as stronger perturbation can reduce the information contained in the gradients but may also degrade model utility.

In our experiments, we implement the gradient clipping and perturbation mechanism using Opacus [17]. We fix the clipping threshold to $C = 1$ and vary the Gaussian noise multiplier $\sigma$ to study the robustness of the reconstruction attacks to diferent levels of gradient perturbation. Importantly, the defense is applied only during gradient collection, and the underlying models are not trained with DP-SGD. This keeps the model parameters and corresponding clean gradients fixed across experiments, allowing the efect of clipping and noise on reconstruction performance to be isolated more directly.

SNR calibration. We characterize the perturbation level using the signal-to-noise ratio (SNR) between the clipped gradient and the added Gaussian noise. The SNR is computed as

$$
\mathrm { S N R } = 1 0 \log _ { 1 0 } \left( \frac { \| { \pmb g } _ { \mathrm { c l i p } } \| _ { 2 } ^ { 2 } } { \mathbb E \| { \pmb n } \| _ { 2 } ^ { 2 } } \right) ,
$$

where $\mathbf { \pmb { g } } _ { \mathrm { c l i p } }$ denotes the clipped gradient before noise addition and n denotes the Gaussian noise, both concatenated over all model parameters. The expected noise power is determined from the variance of the Gaussian perturbation. Since the same noise multiplier can result in diferent SNRs across samples and architectures, we select a single value of σ for each architecture and target SNR such that the average SNR over the 5 evaluated data batches is close to the desired value. We keep this $\sigma$ fixed across all 5 batches, ensuring a consistent defense configuration and identical DP parameters across reconstructions.

All defense experiments are performed with batch size B = 32. For the FC-3 model, we evaluate the ReLU network trained on CIFAR-10 at target SNR levels of {−5, 0, 5, 10, 15} dB, while for VGG-16 pretrained on ImageNet we consider {−20, −15, −10, −5, 0} dB.

The corresponding noise multipliers and average achieved SNR values are reported in Table 6. Lower SNR values correspond to stronger gradient perturbation and therefore to a more challenging reconstruction setting. The diferent SNR ranges for FC-3 and VGG-16 are chosen to cover the transition from successful to degraded reconstruction for each model.

Reconstruction post-processing. Under strong gradient perturbation, we observe that the subspace-based reconstruction methods for the FC-3 experiments can exhibit a compressed intensity range, giving some images a washed-out appearance. This behavior is specific to the subspace variants and is not observed for the baseline methods. We can therefore apply percentile-based contrast stretching as a post-processing step to the reconstructions produced by the subspace methods. Its efect is image-dependent, ranging from minor adjustments when the reconstruction is already suficiently visible to more noticeable contrast enhancement when the intensity range is strongly compressed.

For each reconstructed FC-3 batch, we compute the 0.1st and 99.9th percentiles, denoted by $P _ { 0 . 1 }$ and $P _ { 9 9 . 9 }$ , over all reconstructed pixel values in the batch. The same percentile bounds are therefore used for all images within a batch, ensuring that they undergo the same intensity transformation. The reconstructed images are then rescaled according to

Table 6: Noise multipliers selected for the target SNR levels in the FC-3 and VGG-16 defense experiments. For each target, σ is fixed across the 5 evaluated batches. Measured SNR is reported as mean ± standard deviation across batches.
<table><tr><td colspan="3">FC-3</td><td colspan="3">VGG-16</td></tr><tr><td>Target SNR (dB)</td><td>Measured SNR (dB)</td><td> $\sigma \ ( \times 1 0 ^ { - 3 } )$ </td><td>Target SNR (dB)</td><td>Measured SNR (dB)</td><td> $\sigma \ ( \times 1 0 ^ { - 3 } )$ </td></tr><tr><td>-5</td><td> $- 4 . 9 9 7 \pm 0 . 7 4 0$ </td><td>8.17</td><td>-20</td><td> $- 1 9 . 9 9 2 \pm 0 . 2 5 8$ </td><td>4.55</td></tr><tr><td>0</td><td> $- 0 . 0 0 7 \pm 0 . 7 4 0$ </td><td>4.60</td><td>-15</td><td> $- 1 4 . 9 9 6 \pm 0 . 2 5 8$ </td><td>2.56</td></tr><tr><td>5</td><td> $5 . 0 1 6 \pm 0 . 7 4 0$ </td><td>2.58</td><td>-10</td><td> $- 9 . 9 9 9 \pm 0 . 2 5 8$ </td><td>1.44</td></tr><tr><td>10</td><td> $1 0 . 0 2 1 \pm 0 . 7 4 0$ </td><td>1.45</td><td>-5</td><td> $- 5 . 0 0 1 \pm 0 . 2 5 8$ </td><td>0.81</td></tr><tr><td>15</td><td> $1 4 . 9 7 2 \pm 0 . 7 4 0$ </td><td>0.82</td><td>0</td><td> $- 0 . 0 8 7 \pm 0 . 2 5 8$ </td><td>0.46</td></tr></table>

$$
\hat { \pmb { x } } ^ { + } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { \hat { \pmb { x } } - P _ { 0 . 1 } } { P _ { 9 9 . 9 } - P _ { 0 . 1 } + \epsilon } \right) ,
$$

where xˆ denotes the reconstructed batch and ϵ is a small constant for numerical stability. Values below $P _ { 0 . 1 }$ and above $P _ { 9 9 . 9 }$ are clipped to 0 and 1, respectively.

We use percentile bounds rather than the absolute minimum and maximum to reduce sensitivity to a small number of extreme pixel values while otherwise preserving nearly the full reconstructed intensity range. Importantly, this operation is performed only after reconstruction and does not modify the observed gradients or the reconstruction optimization itself. Its purpose is to compensate for the contrast compression observed in the FC-3 outputs of the subspace methods rather than to recover additional information from the gradients. As such, it can be viewed as an optional post-processing step available to an attacker when the recovered images exhibit this washed-out appearance.

In the FC-3 results, a + sufix denotes a subspace reconstruction followed by this percentile-based contrast stretching $( \mathrm { e . g . } , S o f t A f f u n e + )$ . Since the baselines do not exhibit the same washed-out behavior, no corresponding post-processing is applied to them. In particular, IG already produces batches that span the full [0, 1] intensity range, so applying the same transformation would leave their outputs largely unchanged. Additional qualitative and quantitative comparisons of FC-3 reconstructions before and after contrast stretching are provided in Appendix B. For the VGG-16 experiments, all reported results correspond to the original reconstruction pipeline without this post-processing step.

Results under gradient perturbation. Figures 3a and 4a show that the average reconstruction quality improves consistently with SNR for both FC-3 and VGG-16. For FC-3, the diferences between methods are relatively small at −5 dB, where reconstruction is strongly degraded. From 0 to 15 dB, the soft subspace variants generally achieve lower LPIPS than the baselines, with CPA being the most afected by gradient perturbation and an increasing gap observed relative to IG. For VGG-16, the subspace variants are particularly advantageous at low SNR, where the soft formulations recover embeddings with higher cosine similarity than CPA and the hard variants. The diferences among the VGG-16 methods diminish as SNR increases, with all methods approaching similar performance near 0 dB. Interestingly, the choice between soft and hard constraints has a considerably larger efect on reconstruction performance than the choice between the linear and afine formulations, which behave similarly across SNR levels.

To assess whether the VGG-16 embedding trends translate to final image reconstruction, we additionally perform feature inversion on embeddings recovered by CPA and Soft Afine, representing the baseline and a strong subspace variant, respectively. Following CPA, the inversion objective also includes gradient matching, using both the recovered embedding and the observed gradients [2]. Figure 4b reports LPIPS across the same SNR levels and shows that Soft Afine consistently outperforms CPA, in line with the embedding reconstruction results. The same trend is observed for the best reconstruction within each batch, with the largest diferences occurring at lower SNR and the gap narrowing as the gradients become cleaner.

![](images/399fb0ffe40ca131b3f1f2114f462022fc37af942e25ccb7527f4b54b1f66bee.jpg)  
(a)

![](images/76bef1de3403048ee7f5badf70b370bcb91d5c27aba39fc0f5be0411cbb8d917.jpg)  
(b)  
Figure 3: Reconstruction performance versus SNR for FC-3 with ReLU on CIFAR-10, evaluated over five batches of $B = 3 2 .$ . (a) Mean LPIPS across all reconstructions. (b) Mean LPIPS of the best reconstruction from each batch. Lower values indicate better reconstruction quality.

As indicated by the best-of-batch LPIPS results in Figures 3b and 4b, and supported by visual inspection, meaningful reconstruction of at least some images remains possible down to approximately 0 dB for FC-3 and −15 dB for VGG-16, respectively. Representative examples are provided in Appendix C. These SNR levels correspond to noise multipliers of $4 . 6 0 \times 1 0 ^ { - 3 }$ and $2 . 5 6 \times 1 0 ^ { - 3 }$ . Thus, substantially lower SNR is required to suppress meaningful reconstruction for $\mathrm { V G G { - } 1 6 }$ , despite its corresponding noise multiplier being smaller. This highlights the architecture-dependent relationship between gradient magnitude, injected noise, and SNR and shows why the noise multiplier alone is not directly comparable across the two models.

Overall, the results show that the subspace formulations are more robust to gradient perturbation, although the regime in which their advantage is most pronounced difers between the two architectures. For FC-3, the clearest separation from the baselines occurs over 0 to 15 dB, while at −5 dB all methods are strongly degraded. For VGG-16, the advantage is most pronounced at lower SNR and diminishes as the gradients become cleaner.

## 6 Limitations and Future Work

Although the proposed subspace formulations achieve promising reconstruction results, their efectiveness depends on several factors, including the network architecture, batch size, and the availability of label information in certain settings. Further evaluation across diferent architectures, datasets, and data domains, such as tabular data, would help establish the generality of the findings. Moreover, while afine constraints improve reconstruction over existing linear formulations in certain settings, these improvements are not consistent across all configurations. Investigating the conditions under which afine subspace constraints provide an advantage over linear constraints could help clarify their relative efectiveness.

![](images/abb21d8e77e9496a9591befe926e999fc4e8c570666474034b5b85a7723a05f1.jpg)  
(a)

![](images/88841e2d3d5663e54285900d903aa0bd9265552a8ac7ac909c2bfbed48598e90.jpg)  
(b)  
Figure 4: Reconstruction performance versus SNR for VGG16 on ImageNet, evaluated over five batches of $B = 3 2$ . (a) Mean cosine similarity (CS) between original and reconstructed embeddings. (b) Mean LPIPS and Mean of Best LPIPS computed on the final reconstructed images after feature inversion. Higher CS and lower LPIPS indicate better reconstruction quality.

The evaluation under gradient perturbation is limited to a single batch size $\left( B = 3 2 \right)$ , a fixed clipping threshold, and varying Gaussian noise levels. Although the considered noise multipliers may correspond to weak formal privacy guarantees, future work should investigate the relationship between empirical resistance to reconstruction, formal DP guarantees, and model utility across diferent privacy configurations. In particular, the strong perturbation required to suppress meaningful reconstruction in some settings raises questions about its impact on model performance and the practical privacy–utility trade-ofs of DP mechanisms.

## 7 Conclusion

In this work, we investigated subspace-based formulations for gradient inversion in the image domain, considering linear and afine subspaces under soft and hard constraints. We evaluated their efectiveness across diferent network architectures, datasets, batch sizes, and attacker assumptions, including settings with gradient perturbations introduced by a DP mechanism.

The experimental results show that our subspace approaches can improve reconstruction quality over existing baselines, although their efectiveness depends on the experimental setting. For FC-3, the best-performing subspace variants outperform CPA on CIFAR-10 at batch sizes up to 128 under both activations, and on Tiny-ImageNet across all tested batch sizes with LeakyReLU. However, CPA performs best on CIFAR-10 at B = 256 and on Tiny-ImageNet with ReLU at $B \geq 3 2$ . For VGG-16, label availability strongly influences the performance of the subspace methods. Afine constraints ofer advantages over their linear counterparts in certain settings, but the two perform similarly under the evaluated DP gradient perturbations at $B = 3 2$ , where the subspace approaches generally outperform CPA.

The batch-size experiments reveal that underestimation generally degrades reconstruction quality, whereas moderate overestimation can preserve or even improve it. This asymmetry suggests that precise knowledge of the true batch size is not always necessary, as a moderately overestimated value can sufice. Overall, these findings demonstrate the value of exploiting multi-layer subspace structure for gradient inversion and accounting for uncertainty in batch-size knowledge when assessing privacy leakage from shared gradients.

## Declaration of AI-assisted tools

ChatGPT was used to assist with parts of the code implementation and debugging, and to improve the wording and clarity of the manuscript. The authors take full responsibility for the implementation, experimental results, and manuscript content.

## References

[1] L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 14 747–14 756. [Online]. Available: https://proceedings. neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html

[2] S. Kariyappa, C. Guo, K. Maeng, W. Xiong, G. E. Suh, M. K. Qureshi, and H. S. Lee, “Cocktail party attack: Breaking aggregation-based privacy in federated learning using independent component analysis,” in International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 15 884–15 899. [Online]. Available: https://proceedings.mlr.press/v202/kariyappa23a.html

[3] J. Geiping, H. Bauermeister, H. Dr¨oge, and M. Moeller, “Inverting gradients - how easy is it to break privacy in federated learning?” in Advances in Neural Information Processing Systems, vol. 33, 2020. [Online]. Available: https://proceedings.neurips.cc/paper/2020/ hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html

[4] H. Yin, A. Mallya, A. Vahdat, J. M. Alvarez, J. Kautz, and P. Molchanov,<sup>´</sup> “See through gradients: Image batch recovery via gradinversion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Computer Vision Foundation / IEEE, 2021, pp. 16 337–16 346. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2021/html/Yin See Through Gradients Image Batch Recovery via GradInversion CVPR 2021 paper.html

[5] J. Jeon, J. Kim, K. Lee, S. Oh, and J. Ok, “Gradient inversion with generative image prior,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 29 898–29 908. [Online]. Available: https://proceedings.neurips.cc/paper/2021/hash/ fa84632d742f2729dc32ce8cb5d49733-Abstract.html

[6] H. Fang, B. Chen, X. Wang, Z. Wang, and S. Xia, “GIFD: A generative gradient inversion method with feature domain optimization,” in Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, 2023, pp. 4944–4953. [Online]. Available: https://doi.org/10.1109/ICCV51070.2023.00458

[7] Y. Sun, G. Xiong, X. Yao, K. Ma, and J. Cui, “GI-PIP: do we require impractical auxiliary dataset for gradient inversion attacks?” in IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2024, pp. 4675–4679. [Online]. Available: https://doi.org/10.1109/ICASSP48485.2024.10445924

[8] W. Kalikman, I. Petrov, D. I. Dimitrov, and M. T. Vechev, “TIGER: inverting transformer gradients via embedding-subspace distance optimization,” CoRR, vol. abs/2606.18312, 2026. [Online]. Available: https://doi.org/10.48550/arXiv.2606.18312

[9] D. I. Dimitrov, M. Baader, M. N. M¨uller, and M. T. Vechev, “SPEAR: exact gradient inversion of batches in federated learning,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: http://papers.nips.cc/paper files/paper/2024/ hash/c13cd7feab4beb1a27981e19e2455916-Abstract-Conference.html

[10] A. Bakarsky, D. I. Dimitrov, M. Baader, and M. T. Vechev, “SPEAR++: scaling gradient inversion via sparsely-used dictionary learning,” CoRR, vol. abs/2510.24200, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2510.24200

[11] I. Petrov, D. I. Dimitrov, M. Baader, M. N. M¨uller, and M. T. Vechev, “DAGER: exact gradient inversion for large language models,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: http://papers.nips.cc/paper files/ paper/2024/hash/9f1577a1f8308df1ccea6b4f64a103f-Abstract-Conference.html

[12] L. H. Fowl, J. Geiping, W. Czaja, M. Goldblum, and T. Goldstein, “Robbing the fed: Directly obtaining private data in federated learning with modified models,” in The Tenth International Conference on Learning Representations. OpenReview.net, 2022. [Online]. Available: https://openreview.net/forum?id=fwzUgo0FM9v

[13] F. Boenisch, A. Dziedzic, R. Schuster, A. S. Shamsabadi, I. Shumailov, and N. Papernot, “When the curious abandon honesty: Federated learning is not private,” in 8th IEEE European Symposium on Security and Privacy. IEEE, 2023, pp. 175–199. [Online]. Available: https://doi.org/10.1109/EuroSP57164.2023.00020

[14] Y. Wen, J. Geiping, L. Fowl, M. Goldblum, and T. Goldstein, “Fishing for user data in large-batch federated learning via gradient magnification,” in International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 23 668–23 684. [Online]. Available: https://proceedings.mlr.press/v162/wen22a.html

[15] F. Diana, C. Xu, A. Nusser, and G. Neglia, “No more guessing: a verifiable gradient inversion attack in federated learning,” CoRR, vol. abs/2604.15063, 2026. [Online]. Available: https://doi.org/10.48550/arXiv.2604.15063

[16] M. Abadi, A. Chu, I. J. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, “Deep learning with diferential privacy,” in Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2016, pp. 308–318. [Online]. Available: https://doi.org/10.1145/2976749.2978318

[17] A. Yousefpour, I. Shilov, A. Sablayrolles, D. Testuggine, K. Prasad, M. Malek, J. Nguyen, S. Ghosh, A. Bharadwaj, J. Zhao, G. Cormode, and I. Mironov, “Opacus: User-friendly diferential privacy library in pytorch,” CoRR, vol. abs/2109.12298, 2021. [Online]. Available: https://arxiv.org/abs/2109.12298

## A Hyperparameter Settings

This appendix reports the hyperparameter settings used throughout the experiments. Separate configurations are provided for the FC and VGG models, as the VGG experiments involve additional hyperparameters.

For the FC models, the same hyperparameter configuration is used in the noiseless setting for both accurate B and overestimated Bb. As reported in Table 7, the subspace regularization parameters are reduced under the noisy setting, while all other hyperparameters remain unchanged.

For the VGG model, Table 8 reports the hyperparameter configurations for the noiseless setting with accurate B and overestimated Bb, as well as for the noisy setting. Relative to the accurate-B configuration, the overestimated-Bb configuration changes only $\lambda _ { \mathrm { d e c o r } }$ for Soft Afine, setting it to 0.1. The noisy setting retains this value and additionally adjusts $\lambda _ { \mathrm { n v } }$ for the subspace attacks, as reported in the table. All other hyperparameters remain unchanged.

When the linear and afine variants of a subspace method use identical hyperparameter values, we report them jointly as Soft Subspace or Hard Subspace for conciseness. A dash (–) indicates that the corresponding hyperparameter is not used for that attack. In the noiseless setting, $\lambda _ { \mathrm { n v } }$ is not used for the Soft Subspace attacks because non-negativity is enforced through projection onto the positive orthant. In the noisy setting, this projection is removed to relax the non-negativity constraint, which is instead encouraged through the regularization term weighted by $\lambda _ { \mathrm { n v } }$

Table 7: Hyperparameter settings for the FC models under the noiseless and noisy settings. Only configurations that difer from the preceding setting are reported.
<table><tr><td>Setting</td><td>Attack</td><td>lr</td><td> $\lambda _ { \mathrm { T V } }$ </td><td> $\lambda _ { \mathrm { s u b - i n p } }$ </td><td> $\lambda _ { \mathrm { s u b - i n t e r } }$ </td><td> $\lambda _ { \mathrm { d e c o r } }$ </td><td> $T$ </td></tr><tr><td rowspan="4">Noiseless</td><td>CPA</td><td> $1 0 ^ { - 3 }$ </td><td>3.1</td><td></td><td></td><td>1.47</td><td>12.4</td></tr><tr><td>IG</td><td>0.05</td><td> $1 0 ^ { - 3 }$ </td><td></td><td></td><td>一</td><td></td></tr><tr><td>Soft Subspace</td><td>0.05</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>一</td><td></td></tr><tr><td>Hard Subspace</td><td>0.05</td><td> $1 0 ^ { - 4 }$ </td><td></td><td> $1 0 ^ { - 4 }$ </td><td>一</td><td></td></tr><tr><td rowspan="2">Noisy</td><td>Soft Subspace</td><td>0.05</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 5 }$ </td><td></td><td></td></tr><tr><td>Hard Subspace</td><td>0.05</td><td> $1 0 ^ { - 4 }$ </td><td></td><td> $1 0 ^ { - 5 }$ </td><td></td><td></td></tr></table>

Table 8: Hyperparameter settings for the VGG model under the noiseless setting with accurate B or overestimated ${ \widehat { B } } ,$ and under the noisy setting. Only configurations that difer from the preceding setting are reported.
<table><tr><td>Setting</td><td>Attack</td><td>lr</td><td> $\lambda _ { \mathrm { T V } }$ </td><td> $\lambda _ { \mathrm { s u b - i n p } }$ </td><td> $\lambda _ { \mathrm { s u b - i n t e r } }$ </td><td> $\lambda _ { \mathrm { n v } }$ </td><td> $\lambda _ { \ell _ { 1 } }$ </td><td> $\lambda _ { \mathrm { d e c o r } }$ </td><td> $T$ </td></tr><tr><td rowspan="4">Accurate B</td><td>CPA</td><td>10-3</td><td>0.1</td><td></td><td></td><td>0.13</td><td>5</td><td>5.3</td><td>7.7</td></tr><tr><td>Soft Affine</td><td>0.06</td><td>0.1</td><td> $5 \cdot 1 0 ^ { - 4 } $ </td><td>10-4</td><td></td><td>10-3</td><td>2</td><td></td></tr><tr><td>Soft Linear</td><td>0.06</td><td>0.1</td><td> $5 \cdot 1 0 ^ { - 4 } $ </td><td> $1 0 ^ { - 4 }$ </td><td></td><td>10-3</td><td>0.1</td><td>一</td></tr><tr><td>Hard Subspace</td><td>0.06</td><td>0.1</td><td></td><td> $1 0 ^ { - 4 }$ </td><td>0.1</td><td> $1 0 ^ { - 3 }$ </td><td>2</td><td>一</td></tr><tr><td>Overest. B</td><td>Soft Affine</td><td>0.06</td><td>0.1</td><td> $5 \cdot 1 0 ^ { - 4 } $ </td><td> $1 0 ^ { - 4 }$ </td><td></td><td> $1 0 ^ { - 3 }$ </td><td>0.1</td><td>一</td></tr><tr><td rowspan="2">Noisy</td><td>Soft Subspace</td><td>0.06</td><td>0.1</td><td> $5 \cdot 1 0 ^ { - 4 } $ </td><td>10⁻4</td><td>10-3</td><td>10-3</td><td>0.1</td><td>一</td></tr><tr><td>Hard Subspace</td><td>0.06</td><td>0.1</td><td></td><td>10-4</td><td>10-3</td><td>10-3</td><td>2</td><td>一</td></tr></table>

## B Qualitative and Quantitative Efects of Contrast Stretching

To assess the efect of the percentile-based contrast stretching described in Section 5.3, we compare the original FC-3 subspace reconstructions with their post-processed counterparts. Figure 5 reports the change in (a) mean LPIPS across all reconstructions and (b) the mean of the minimum LPIPS values across reconstruction batches, for each evaluated SNR level. The change is computed as the LPIPS score before post-processing minus the score after post-processing, such that positive values indicate improved reconstruction quality.

For mean LPIPS, contrast stretching generally provides small improvements at SNR levels of 0 dB and above, particularly for the hard-constrained variants, while slightly degrading performance at −5 dB. In contrast, the mean of best LPIPS improves across all evaluated methods and SNR levels, although the magnitude of the improvement varies. Overall, these results indicate that contrast stretching provides modest quantitative benefits, depending on the reconstruction method, perturbation level, and characteristics of the reconstructed batch.

Figure 6 provides a qualitative comparison at approximately 0 dB SNR, showing 16 images from a batch of $B = 3 2$ reconstructed using Soft Afine, before and after post-processing. For this batch, the percentile bounds are $P _ { 0 . 1 } = 0 . 2 1 1$ and $P _ { 9 9 . 9 } = 0 . 7 5 6$ , indicating that nearly all reconstructed pixel values lie within a substantially compressed intensity range. Following contrast stretching, the mean LPIPS decreases slightly from 0.657 to 0.646, while the best LPIPS decreases more substantially from 0.538 to 0.427. This batch exhibits the largest improvement in best LPIPS among the evaluated Soft Afine batches at this SNR level and therefore illustrates a particularly favorable case for post-processing. Visually, contrast stretching makes certain image structures and colors more distinguishable, although it can also amplify existing noise and reconstruction artifacts. Thus, the post-processing can improve the visual appearance of some reconstructions without altering the reconstruction procedure or recovering additional information from the observed gradients.

![](images/2fdc0e5342ddfacde0be35fc69f565c7024de3fbbb2953ea244d71b22bf9dbab.jpg)  
(a)

![](images/f30d9e77b90276b9614ebde6390ece0a66c0a124b69c16f1c6039ef4666e584e.jpg)  
(b)

Figure 5: Efect of percentile-based contrast stretching on the reconstruction quality of the FC-3 model on CIFAR-10 across diferent SNR levels. (a) Improvement in mean LPIPS. (b) Improvement in mean of best LPIPS. Improvements are computed by subtracting the postprocessed scores from the original scores, with positive values indicating better reconstruction quality after contrast stretching.  
![](images/567f5c9b644b31be3c1f9dc6a1112e1409b4db2341a836ccd36b50fc95efc3a1.jpg)  
Figure 6: Comparison of CIFAR-10 reconstructions for the FC-3 model before and after percentile-based contrast stretching at approximately 0 dB SNR. A subset of 16 images from a batch of B = 32 is shown, comparing the ground-truth images (True), the original Soft Afine reconstructions (Soft Af), and their post-processed counterparts (Soft Af+). Percentile bounds are computed over the entire reconstructed batch.

## C Additional Qualitative Results

This appendix provides additional qualitative examples complementing the quantitative results in Section 5. Figures 7 and 8 show noiseless reconstructions on CIFAR-10 and Tiny-ImageNet, respectively, with Figure 7 also comparing the linear and afine formulations. Figures 9 and 10 illustrate the efect of gradient perturbations on CIFAR-10 and ImageNet reconstruction, keeping the displayed ground-truth images fixed across SNR levels. These examples illustrate diferences between methods and the variation in reconstruction quality across individual images.

![](images/5d5cef032644f423a9e0e70479a205261a3e801dd3d7e0b3cbf2a0fd6411c937.jpg)  
(c)  
Figure 7: Comparison of CIFAR-10 reconstructions for FC-3 with ReLU in the noiseless setting with known batch size B = 64. A subset of the ground-truth batch is shown alongside the corresponding matched reconstructions from IG, CPA, Soft Linear and Soft Afine. Panels (a)–(c) show examples from 3 diferent batches.

![](images/67180ca4f73c39b2ff5f2ee58f13b34d529c2d7ad5ea7d9ed4912ef917ca456c.jpg)  
(d)  
Figure 8: Comparison of Tiny-ImageNet reconstructions for FC-3 with LeakyReLU in the noiseless setting with known batch size B = 128. A subset of the ground-truth batch is shown alongside the corresponding matched reconstructions from IG, CPA and Soft Afine. Panels (a)–(d) show examples from 4 diferent batches.

<table><tr><td rowspan=6 colspan=1>TrueIGCPASoft Aff+</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>sin</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=16>(a) 0 dB</td></tr><tr><td rowspan=2 colspan=1>TrueIG</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>CPASoft Aff+</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=14>(b) 5 dB</td></tr><tr><td rowspan=1 colspan=1>True</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>IG</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>匯</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CPA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Soft Aff+</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>三</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=14>(c) 10 dB</td></tr><tr><td rowspan=1 colspan=1>True</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>IG</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CPA</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>医</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Soft Aff+</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

(d) 15 dB  
Figure 9: Comparison of CIFAR-10 reconstructions for FC-3 with ReLU under gradient perturbations at target SNR levels of (a) 0, (b) 5, (c) 10 and (d) 15 dB. The same 16 ground-truth images from a batch of B = 32 are shown across all panels, together with their matched reconstructions from IG, CPA, and Soft Afine. The + sufix denotes percentile-based contrast stretching applied after reconstruction.

![](images/e72ca6fd8c205e454c0911a8bdf30b4832c51cb51fc9559f5e143d4a27e5e38e.jpg)  
(d) 0 dB  
Figure 10: Comparison of ImageNet reconstructions for VGG-16 under gradient perturbations at target SNR levels of (a) −15, (b) −10, (c) −5 and (d) 0 dB. The same 16 ground-truth images from a batch of B = 32 are shown across all panels, together with their matched reconstructions after feature inversion of embeddings recovered using CPA and Soft Afine. The feature inversion stage also incorporates gradient matching against the observed perturbed gradients.