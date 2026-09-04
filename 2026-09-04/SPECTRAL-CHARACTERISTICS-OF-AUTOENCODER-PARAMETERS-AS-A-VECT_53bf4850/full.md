# SPECTRAL CHARACTERISTICS OF AUTOENCODER PARAMETERS AS A VECTOR REPRESENTATION OF DATA

Maria Nikitina

Anton Bishuk

Oleg Bakhteev

## ABSTRACT

<sup>1</sup> This paper examines the relationship between the parameters of autoencoder models and the statistical properties of the data on which they are trained. Autoencoders are defined as models with an encoder-decoder architecture, trained to reconstruct input data through a compressed latent representation. It is proposed that the model parameters can be viewed as a dense vector representation of the corresponding sample. To test this hypothesis, a theoretical and experimental study is conducted in which a vector representation is formed based on the spectral characteristics of the autoencoder parameter matrices. Theoretical analysis shows that the singular values of the model parameter matrices are related to the eigenvalues of the covariance matrix of the training data, ensuring the transfer of information between the data space and the parameter space. Experimental results on the CIFAR-10 and FashionMNIST datasets confirm that the resulting vector representations allow for a high degree of accuracy in distinguishing between models trained on different data subsets, without resorting to complex vector generation algorithms or using the original samples. These results suggest that the parameters of trained autoencoders can be viewed as sample representations.

Keywords machine learning · neural networks · vector representation · autoencoder · spectral characteristics · generative models · embeddings

## 1 Introduction

Machine learning methods based on artificial neural networks have become firmly established as an important tool applied in various areas of modern data analysis [1, 2]. The relationship between the structure of trained neural networks and the statistical properties of the data on which they were trained is one of the fundamental problems of modern machine learning [25]. Understanding how the distribution of training data is reflected in the model parameters is of key importance for such areas as neural network interpretability, analysis of data properties, architecture selection, and the construction of meta-representations of models [3, 4, 8, 20, 21]. Furthermore, this problem is of particular relevance to generative models, since they not only extract patterns but also form internal representations that reflect the structure of the data distribution [30, 24]. This paper focuses on autoencoders, neural networks based on the encoder–decoder architecture. Although their classical architecture is not formally generative, regularized variants approximate or even constitute full-fledged generative models [6, 17, 22]. Thus, the analysis of the parameter structure of an autoencoder is naturally connected to the broader class of generative models, and the results of this work can potentially be generalized to this class.

A broad range of studies has been devoted to the construction of embeddings of machine learning models. Classical approaches rely on encoding architectural parameters [9, 11], meta-representations obtained using separate models [10, 26], or representations constructed from the model’s outputs on test data [16]. Recent works have proposed treating a model as a point in a continuous space, enabling model search, interpolation, and analysis [7, 15]. However, most of these methods either require direct access to the original data or rely on complex procedural heuristics that combine intermediate computations, architectural features, or meta-information.

In this paper, a different approach is considered. We investigate the extent to which the parameters of a trained model can serve as a representative description of the data on which it was trained. Specifically, we test the hypothesis that the spectral characteristics of the parameters of a trained neural network form a compact embedding of the original dataset. Thus, the trained model acts as an embedding of the dataset, while its parameters serve as a carrier of the statistical characteristics of the underlying data distribution [19, 23].

To demonstrate the proposed approach, we introduce a model vectorization method based on the analysis of the singular values of the model parameter matrices. It is shown that the proposed method enables the separation of models trained on different datasets with high accuracy. This result indicates the existence of a deep relationship between the spectral properties of model parameters and the statistics of the training datasets. In contrast to approaches based on external features or approximations of the data distribution, the proposed method operates exclusively in the parameter space of the model, without access to the original data or additional training of auxiliary networks.

The theoretical part of the paper provides a justification for the relationship between the singular values of the model parameters and the spectrum of the covariance matrix of the data. It is shown that, for autoencoders with fully connected layers, the singular values of the trained weight matrices contain information about the distribution of the training datasets. Furthermore, it is proven that if the underlying datasets differ, the resulting embeddings also differ.

The experimental part is aimed at testing the hypothesis that even when using shallow architectures from the families of autoencoders $( \mathrm { A E } )$ and variational autoencoders (VAE), together with non-trainable vectorization methods based solely on model parameters, it is possible to distinguish models trained on different subsets of data with high accuracy. Experiments on the CIFAR-10 [18] and FashionMNIST [29] datasets confirm that classifying such models using their embeddings achieves high values of standard classification metrics, including Accuracy, ROC AUC, and Average Precision. Moreover, the gradual addition of new classes or noise to the training dataset is reflected in the resulting embeddings in a predictable manner, indicating that the model space possesses a meaningful geometric structure.

The obtained results make it possible to consider the parameters of neural networks as a means of describing data. In this sense, trained models become not only a tool for data generation or approximation but also a natural embedding for analyzing the structure of datasets and the relationships between them.

## 2 Problem statement

Let $p _ { c }$ denote the data distribution of class $c \in \{ 1 , 2 , \ldots \}$ over the space $\mathbb { R } ^ { d }$ of d-dimensional vectors. Let the dataset be given by $\mathbf { X } _ { c } = \{ \mathbf { x } _ { i } : \mathbf { x } _ { i } \sim p _ { c } \} _ { i = 1 } ^ { n } . \mathrm { ~ A ~ }$ dataset may consist of several classes, each corresponding to its own distribution $p _ { c }$ . In the classical unsupervised learning setting, $c \equiv 1$ and $p _ { c } \equiv p _ { 1 } \equiv p _ { \cdot }$ , while ${ \bf X } _ { c } \equiv { \bf \dot { X } } _ { 1 } \equiv { \bf \check { X } }$

We consider a family of autoencoder models. A model is defined as a parameterized function $\mathbf { f } _ { \psi } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , represented as the composition of an encoder and a decoder, $\mathbf { f } _ { \psi } = D _ { \varphi } \circ E _ { \theta }$ . The model parameters are denoted by $\pmb { \psi } = ( \pmb { \theta } , \pmb { \varphi } ) \in \Psi$ where $E _ { \pmb { \theta } } : \mathbb { R } ^ { d }  \mathbb { R } ^ { z }$ is the encoder with parameters $\pmb \theta _ { ; }$ , and $D _ { \varphi } : \mathbb { R } ^ { z }  \mathbb { R } ^ { d }$ is the decoder with parameters $\varphi .$ . Here, z denotes the dimension of the hidden space. Training is performed by minimizing the objective

$$
\begin{array} { r } { \mathcal { L } ( \psi ; p _ { c } ) : = - \mathbb { E } _ { \mathbf { x } \sim p _ { c } } \Big [ l _ { \psi } ( \mathbf { x } ) \Big ] , } \end{array}
$$

where the maximized functional $l _ { \psi } ( \mathbf { x } )$ is defined as

$$
l _ { \psi } ( { \bf x } ) = \left\{ \begin{array} { l l } { - \ell \big ( { \bf x } , ~ { \bf f } _ { \psi } ( { \bf x } ) \big ) , } & { \mathrm { f o r ~ A E , } } \\ { { \mathbb { E } } _ { { \bf h } \sim q _ { \theta } ( { \bf h } \mid { \bf x } ) } \big [ - \ell ( { \bf x } , D _ { \varphi } ( { \bf h } ) ) \big ] } & { \mathrm { f o r ~ V A E , } } \\ { - \mathrm { K L } \big ( q _ { \theta } ( { \bf h } \mid { \bf x } ) \big \| \ : p ( { \bf h } ) \big ) , } & { } \end{array} \right.
$$

where $\ell : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R }$ is the reconstruction loss function (e.g., mean squared error or cross-entropy), which measures the discrepancy between the input vector x and its reconstruction; $\mathrm { K L } ( \cdot | | \cdot )$ denotes the Kullback–Leibler divergence, a measure of the difference between two probability distributions; $q _ { \pmb { \theta } } ( \mathbf { h } \mid \mathbf { \ddot { x } } )$ is the approximate (variational) distribution in the latent space parameterized by the encoder; and $p ( \mathbf { h } )$ is the prior distribution in the latent space (typically the multivariate normal distribution ${ \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) )$ .

For a standard autoencoder (AE), the objective reduces to minimizing the reconstruction error. For a variational autoencoder (VAE), the objective consists of two terms: the expected reconstruction error (where the expectation is taken over the latent variable h) and the KL regularization term, which encourages the encoder distribution $q _ { \pmb { \theta } } ( \mathbf { h } \mid \mathbf { x } )$ to be close to the prescribed prior distribution $p ( \mathbf { h } )$ ).

Suppose that a set of generative models is available, each trained on its own dataset representing a particular underlying population. The goal is to construct a representation space in which each model is mapped to a dense embedding vector preserving the statistical properties of the original data. Such a space should enable the analysis of both the datasets and the models trained on them without requiring retraining or direct access to the original data. At the same time, the resulting embedding is not required to uniquely reconstruct the original model but only to describe a predefined set of its properties.

The key property of the desired representation is the preservation of the ability to discriminate models trained on datasets generated from different distributions $p .$ Suppose that there exists a representation space in which the datasets $\mathbf { X } _ { a } \in \mathbb { R } ^ { \breve { n } _ { a } \times d } , \mathbf { X } _ { b } \in \mathbb { R } ^ { n _ { b } \times d }$ , and $\mathbf { X } _ { c } \in \mathbb { R } ^ { n _ { c } \times d }$ are linearly separable. The same property is required for the embeddings of generative models trained on subsets of samples from these datasets.

We introduce embeddings of both models and datasets based on singular values. Let the model parameters ψ be represented as the set of weight matrices $\{ \mathbf { W } _ { l } \} _ { l = 1 } ^ { L }$ corresponding to its layers.

The model singular values $\sigma ^ { \mathrm { m o d e l } } ( \psi )$ are defined algorithmically (see Algorithm 1): for each weight matrix $\mathbf { W } _ { l }$ the r largest singular values are computed; the resulting embeddings are concatenated in the order of the layers and normalized. Thus, the vector $\sigma ^ { \mathrm { m o d e l } } ( \dot { \boldsymbol { \psi } } )$ is uniquely determined by the parameters and the architecture of the model. The data singular values $\sigma ^ { \mathrm { d a t a } } ( { \bf X } _ { c } )$ for a dataset $\mathbf { X } _ { c }$ are defined as the vector of its m largest singular values sorted in descending order.

Definition 1. Let the sets of vectors $\mathbf { X } _ { a } \in \mathbb { R } ^ { n \times d }$ and $\mathbf { X } _ { b } \in \mathbb { R } ^ { m \times d }$ be given. Define mixing as the concatenation of vectorsfrom $\mathbf { X } _ { a }$ and $\mathbf { X } _ { b }$ to obtain $\tilde { \mathbf { X } } _ { a \mid b } \in \mathbb { R } ^ { ( n + m ) \times d }$

α-mixing is defined as the concatenation of $\mathbf { X } _ { a } a n d \lfloor \alpha m \rfloor , \alpha \in [ 0 ; 1 ]$ vectors from $\mathbf { X } _ { b }$ to obtain $\tilde { \mathbf { X } } _ { a \mid b } \in \mathbb { R } ^ { ( n + \lfloor \alpha m \rfloor ) \times d } .$

Then, mixing samples from $\mathbf { X } _ { c }$ into $\mathbf { X } _ { a }$ and $\mathbf { X } _ { b }$ should bring the resulting datasets $\mathbf { X } _ { a + c } , \mathbf { X } _ { b + c }$ closer to each other, as illustrated in Fig. 1 (a), i.e., reduce their linear separability, as shown in Fig. 1 (b). The embeddings of generative models trained on samples from the selected datasets should exhibit the same properties.

a)  
![](images/0ed9fb886d61bec1b69d2980a31e5c6a680947c697525141e59d41ea4146daa3.jpg)

6)  
![](images/99c01eb625c1613851aea7074554e7c014fbc9c0babb9807bd49844b44f003ce.jpg)  
Figure 1: Schematic representation of the relationships between datasets in the embedding space: a) When vectors from a third dataset c are mixed into samples a and b, the embeddings of the resulting samples a + c and b + c become closer; b) When the vectors of the resulting samples become sufficiently close, they cease to be linearly separable.

Furthermore, the representation should be robust to variations within the same data distribution while remaining sensitive to significant statistical differences between datasets. An important task is to assess how well the model embeddings preserves information about the original dataset. To this end, it is necessary to consider not only the accuracy of reconstructing the original data but also the ability of the representation to reflect the structure and relationships between datasets.

Thus, the task reduces to constructing a mapping from the parameters of a generative model to a compact embedding space while preserving the statistical properties of the original data and their combinations. To address this task, we use a method based on the spectral characteristics of the model parameter matrices, which enables the construction of interpretable and informative embeddings.

## 3 Proposed method

The proposed method is based on the idea of representing a model f as a dense embedding containing information about the statistical properties of the data $X _ { c }$ on which it was trained. For this purpose, we use $\sigma ^ { \mathrm { m o d e l } } ( \psi )$ — the singular values of the parameter matrices ψ extracted from the trained model. This choice is motivated by the fact that the spectral structure of neural network parameters reflects key properties of the input data distribution, while $\sigma ^ { \mathrm { m o d e l } }$ is sensitive to statistical differences between datasets, as will be theoretically justified below.

The formalized scheme of Algorithm 1 illustrates the key steps of the proposed approach.

Algorithm 1 Construction of embeddings for a generative model   
Require: A trained model with L layers, parameter matrices $\{ \mathbf { W } _ { l } \} _ { l = 1 } ^ { L } .$ — the number of singular values selected to   
construct the model embedding.   
Ensure: Embeddings of the model v   
1: Initialize an empty feature vector $\mathbf { v }  \mathbb { I } ;$   
2: for each layer $\mathbf { W } _ { l }$ in the model do   
3: Perform the singular value decomposition $\mathbf { W } _ { l } = \mathbf { U } _ { l } \mathbf { S } _ { l } \mathbf { V } _ { l } ^ { T } ;$   
4: Extract the singular values $\sigma _ { l , 1 } ^ { \mathrm { m o d e l } } , \ldots , \sigma _ { l , k _ { l } } ^ { \mathrm { m o d e l } }$ from $\mathbf { S } _ { l } ;$   
5: Sort the singular values in descending order;   
6: Keep the r largest singular values;   
7: Add the resulting values to v.   
8: end for   
9: Normalize v $( \mathrm { e . g . }$ , to unit norm);   
10: return v

Remark. The number r in the algorithm can be chosen as the minimum number of singular values of the first layer that accountfor a predefinedfraction ofthe sum ofall singular values ofthat layer.

To theoretically justify the proposed method, it is necessary to assess the relationship between $\pmb { \sigma } ^ { \mathrm { m o d e l } }$ and $\pmb { \sigma } ^ { \mathrm { d a t a } }$ . Suppose that there exists an embedding space in which the singular value vectors of different datasets are linearly separable. We aim to prove that the singular value vectors of the parameters of generative models trained on subsets of these datasets can also be linearly separated.

Statement 1. The closer the singular value vectors of the original datasets are, the worse the linear separability of the singular value vectors ofthe corresponding models

Let $V _ { \sigma } \subset \mathbb { R } ^ { d }$ be the space of singular value vectors of datasets. To prove hypothesis 1, the convergence of $\sigma ^ { \mathrm { d a t a } } ( \mathbf { X } _ { 1 } ) \in V _ { \sigma }$ and $\sigma ^ { \hat { \mathrm { d a t a } } } ( \mathbf { X } _ { 2 } ) \in V _ { \sigma }$ for the data matrices $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 } .$ respectively, is achieved by mixing samples from a third dataset ${ \bf X } _ { 3 }$ . To demonstrate that the singular values of $\tilde { \mathbf { X } } _ { c \mid 3 }$ , obtained by concatenating $\mathbf { X } _ { c }$ and ${ \bf X } _ { 3 }$ for $c \in \{ 1 , 2 \}$ , converge as the fraction of samples from ${ \bf X } _ { 3 }$ increases, Lemma 1 was proven. For notational simplicity, below we use $\tilde { \mathbf { X } } _ { c } \equiv \tilde { \mathbf { X } } _ { c | 3 }$ for $c \in \{ 1 , 2 \}$

Lemma 1. Let $\mathbf { X } _ { 1 } \in \mathbb { R } ^ { n _ { 1 } \times d }$ be the data matrix of the first dataset $( n _ { 1 }$ samples, d features), $\mathbf { X } _ { 2 } \in \mathbb { R } ^ { n _ { 2 } \times d }$ be the data matrix ofthe second dataset, and $\mathbf { X } _ { 3 } \in \mathbb { R } ^ { m \times d }$ be the data matrix ofthe third dataset.

Then $\pmb { \sigma } ^ { d a t a } ( \tilde { \mathbf { X } } _ { 1 } )$ and $\pmb { \sigma } ^ { d a t a } ( \tilde { \mathbf { X } } _ { 2 } ) - t h e$ singular values of the matrices $\tilde { \mathbf { X } } _ { 1 }$ and $\tilde { \mathbf { X } } _ { 2 } ~ -$ converge: $| \sigma _ { i } ^ { d a t a } ( \tilde { \bf X } _ { 1 } ) - \sigma _ { i } ^ { d a t a } ( \tilde { \bf X } _ { 2 } ) | \longrightarrow 0$ as $m  \infty ,$ , where $\sigma _ { i } ^ { d a t a } ( \tilde { \mathbf { X } } _ { c } )$ is the i-th singular value ofthe dataset $\tilde { \mathbf { X } } _ { c }$ obtained by mixing vectors from ${ \bf X } _ { 3 }$ into $\mathbf { X } _ { c } .$

In other words, the singular values of the mixtures $\tilde { \mathbf { X } } _ { c }$ converge as the fraction of samples from ${ \bf X } _ { 3 }$ increases. We can now proceed to the proof of Theorem 1 concerning the relationship between the singular values of the dataset and the model weights.

The theorem is formulated and proved for a special case of an autoencoder: an autoencoder with a single fully connected layer in both the encoder and decoder, without a nonlinear activation function. We introduce $\mathbf { \bar { W } } _ { c } ^ { e n c }$ as the matrix representation of the encoder parameters $\theta _ { c } ,$ and $\mathbf { W } _ { c } ^ { d e c }$ as the matrix representation of the decoder parameters $\varphi _ { c }$

Theorem 1. Let three datasets be given: $\mathbf { X } _ { 1 } \in \mathbb { R } ^ { n \times d } , \mathbf { X } _ { 2 } \in \mathbb { R } ^ { n \times d }$ , and $\mathbf { X } _ { 3 } \in \mathbb { R } ^ { m \times d }$ ; let their singular values be positive: $\sigma _ { i } ^ { d a t a } ( { \bf X } _ { c } ) > 0 \forall i \in \overline { { 1 , d } } .$

Let two new datasets be obtained by mixing vectors from the third dataset into each of the first two datasets: $\tilde { \mathbf { X } } _ { c } = [ \mathbf { X } _ { c } , \mathbf { X } _ { 3 } ] \in \mathbb { R } ^ { ( n + m ) \times d } , c \in \{ 1 , 2 \}$

Suppose that an autoencoder model with $f u l l y$ connected layers without nonlinear activation functions and latent dimension $k _ { \mathrm { ~ \normalfont ~ \left. ~ \right.} i s  }$ trained on each dataset, minimizing the loss function $\begin{array} { r } { \operatorname* { m i n } _ { \psi _ { c } } ( \mathcal { L } ( \psi _ { c } ) ) = \operatorname* { m i n } _ { \psi _ { c } } \| \tilde { \mathbf { X } } _ { c } - \tilde { \mathbf { X } } _ { c } \mathbf { W } _ { c } ^ { e n c } \mathbf { W } _ { c } ^ { d e c \top } \| _ { F } ^ { 2 } } \end{array}$ with a regularization term of the form $\| c o \nu ( \mathbf { h } ) - \mathbf { I } \| ^ { 2 }$ which decorrelates the vectors h in the latent space, $i . e . , c o \nu ( \mathbf { h } ) \sim \overset { \textstyle \sim } { I } .$

Then the mean squared difference between the singular values ofthe parameters ofthe autoencoders with parameters ${ \bf W } _ { 1 } ^ { e n c } , { \bf W } _ { 1 } ^ { d e c } , \bar { { \bf W } } _ { 2 } ^ { e n c }$ $\mathbf { W } _ { 2 } ^ { d e c }$ decreases as m increases:

$$
\| \sigma _ { i } ^ { m o d e l } ( \mathbf { W } _ { 1 } ^ { e n c } ) - \sigma _ { i } ^ { m o d e l } ( \mathbf { W } _ { 2 } ^ { e n c } ) \| _ { 2 } \underset { m  \infty } { \longrightarrow } 0 ,
$$

Consequently, as the datasets converge, both their singular values and the singular values of the parameters of the autoencoders trained on these datasets converge as the number ofrows m tends to infinity.

Corollary. The loss function $\mathcal { L } ( \psi _ { c } )$ with the addition of the regularization term $\| c o \nu ( \mathbf h _ { c } ) - \mathbf I \| ^ { 2 }$ , which decorrelates the vectors $\mathbf { h } _ { c }$ in the hidden space, makes the autoencoder closer in its properties to a $V A E ,$ since the covariance matrix ofthe hidden (latent) representation becomes diagonal, which is characteristic ofa VAE. This observation allows the results ofTheorem 1 to be used as evidence that hypothesis 1 holdsfor both AE and VAE models.

Remark. Theorem 1 is provedfor the special case ofmodels withfully connected layers without nonlinearities. The applicability of the theorem to the nonlinear case is evaluated experimentally. A theorem with a larger number of restrictions on the problem being solved, but including models with nonlinearities, is presented below.

Theorem 1 characterizes the behaviour of autoencoder embeddings as the vectors of the datasets converge. However, the greater practical value lies in the ability of models to describe not similarities but differences between datasets. As a basic property, we consider the ability to distinguish models trained on datasets generated from different distributions.

Let $p _ { c }$ denote the data distribution of class $c \in \{ 1 , 2 \}$ on $\mathbb { R } ^ { d }$ with covariance $\pmb { \Sigma } _ { c } : = \pmb { \Sigma } _ { c } ( p _ { c } ) : = \mathbb { E } _ { \mathbf { x } \sim p _ { c } } [ \mathbf { x x } ^ { \top } ]$ . For a dataset $\mathbf { X } _ { c } = \{ \mathbf { x } _ { i } : \mathbf { x } _ { i } \sim p _ { c } \} _ { i = } ^ { n }$ of size $n ,$ the empirical covariance is defined as $\begin{array} { r } { \widehat { \Sigma } _ { c } : = \widehat { \Sigma } _ { c } ( \mathbf { X } _ { c } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \top } } \end{array}$

We consider a model with parameters $\psi \in \Psi$ and loss function $\mathcal { L } ( \psi ; p _ { c } ) : = - \mathbb { E } _ { \mathbf { x } \sim p _ { c } } \Big [ l _ { \psi } ( \mathbf { x } ) \Big ]$ , where $l _ { \psi } ( \mathbf { x } )$ is some maximized functional.

In this work, we assume that the loss function $\mathcal { L } ( \psi ; p _ { c } )$ can be expressed as a function of the data covariance $\Sigma _ { c } ,$ i.e., there exists a functional $\mathcal { L } _ { \Sigma }$ such that $\mathcal { L } ( \psi ; p _ { c } ) = \mathcal { L } _ { \pmb { \Sigma } } ( \psi ; \pmb { \Sigma } _ { c } ( p _ { c } ) )$ . This holds, for example, when $l _ { \psi } ( \mathbf { x } )$ depends on x only through its second moments, or when the distribution $p _ { c }$ is completely determined by its covariance, as in the Gaussian case.

For notational simplicity, we use the notation $\mathcal { L } ( \psi ; \Sigma _ { c } ) : = \mathcal { L } _ { \Sigma } ( \psi ; \Sigma _ { c } )$ below.

Suppose that $\textstyle { \mathcal { L } } ( \psi ; \Sigma _ { c } )$ attains its minimum at $\psi ^ { \ast } ( \cdot )$ . On the dataset $\mathbf { X } _ { c } ,$ , the minimum is attained at $\widehat { \psi } _ { n } ( \cdot )$

$$
\psi ^ { * } ( \Sigma _ { c } ) : = \arg \operatorname* { m i n } _ { \psi \in \Psi } \mathcal { L } ( \psi ; \Sigma _ { c } ) , \quad \widehat { \psi } _ { n } ( \widehat { \Sigma } _ { c } ) : = \arg \operatorname* { m i n } _ { \psi \in \Psi } \mathcal { L } _ { n } ( \psi ; \widehat { \Sigma } _ { c } ) ,
$$

where ${ \mathcal { L } } _ { n }$ is the empirical loss function.

For a proper formulation and proof of the subsequent results, it is necessary to introduce a set of assumptions ensuring the smoothness, convexity, and stability of the optimal solutions, as well as the boundedness of the data.

Theorem conditions.

(B1) (Smoothness) $\mathcal { L } ( \psi ; \pmb { \Sigma } _ { c } )$ belongs to the class $C ^ { 2 }$ with respect to ψ and to $C ^ { 1 }$ with respect to the covariance parameter $\Sigma _ { c }$ . Note that verifying smoothness with respect to $\Sigma _ { c }$ is computationally difficult. However, the statement holds for Gaussian distributions [12].

(B2) (Local uniqueness and strong convexity) For each class $c ,$ there exists a neighborhood $U _ { c } ^ { B 2 }$ of covariance matrices around $\Sigma _ { c } ,$ defined with respect to the metric $\| \cdot \| _ { o p } ,$ such that for all $\pmb { \Sigma } \in U _ { c } ^ { B 2 }$ , the function $\psi \mapsto \mathcal { L } ( \psi ; \Sigma _ { c } )$ has a locally unique isolated minimum $\psi ^ { * } ( \Sigma )$ , and the Hessian at this point satisfies

$$
\exists \hat { \mu } : \lambda _ { \operatorname* { m i n } } \left( \nabla _ { \psi \psi } ^ { 2 } \mathcal { L } ( \psi ^ { * } ( \Sigma ) ; \Sigma _ { c } ) \right) \geq \hat { \mu } > 0 ,
$$

where $\lambda _ { \mathrm { m i n } } ( \cdot )$ denotes the smallest eigenvalue of the Hessian.

(B3) (Differentiability with respect to Σ.) For each class $c ,$ there exists a neighborhood $U _ { c } ^ { B 3 }$ of covariance matrices around $\mathbf { \widehat { \boldsymbol { \Sigma } } } _ { c } ,$ in which the family of minimizers $\psi ^ { * } ( \pmb { \Sigma } )$ is a $C ^ { 1 }$ -function of $\pmb { \Sigma }$ , i.e., the Jacobian $J _ { \Sigma } : = \partial _ { \Sigma } \psi ^ { * } ( \Sigma )$ exists and is continuous.

For each class, we consider the neighborhood $U _ { c } = U _ { c } ^ { B 2 } \cap U _ { c } ^ { B 3 }$ , so that both requirements (B2) and (B3) hold simultaneously. Since these are neighborhoods of the same point $\Sigma _ { c } ,$ , their intersection is non-empty.

(B4) (Approximation error) There exists a fixed neighborhood $\nu _ { c } \subset \Psi$ and a sequence $\eta _ { n } ^ { c } \overset { p } {  } 0$ such that $\begin{array} { r } { \operatorname* { s u p } _ { \psi \in \mathcal { V } _ { c } } \left| \mathcal { L } _ { n } ( \psi ; \widehat { \Sigma } _ { c } ) - \mathcal { L } ( \psi ; \Sigma _ { c } ) \right| \leq \eta _ { n } ^ { c } } \end{array}$ , where n is the sample size. All values of $\psi ^ { * } ( \pmb { \Sigma } _ { c } )$ considered in the theorem lie in $\mathcal { V } _ { c }$ . To generalize to all classes c in the theorem, we use the sequence obtained by taking the pointwise maximum $\eta _ { n } : = \operatorname* { m a x } ( \eta _ { n } ^ { 1 } , \eta _ { n } ^ { 2 } , \dots \eta _ { n } ^ { c } )$ .

(B5) (Bounded data) We assume that there exists a constant $B > 0$ such that $\| \mathbf { x } _ { i } \| _ { 2 } \leq B$ almost surely for all i. This property ensures the applicability of concentration inequalities to the empirical covariance, and in practice can be achieved by preprocessing the data using clipping or winsorization.

(B6) (Distinct first r singular values) Suppose that the first r singular values of the parameter matrix $\psi ^ { * } ( \Sigma _ { c } )$ are distinct and separated from the remaining singular values by $\delta _ { r } > 0$

Theorem 2. Suppose that assumptions $( B I ) – ( B 6 )$ hold. We introduce the notation $\Delta _ { r } : = \| \pmb { \sigma } _ { 1 : r } ^ { d a t a } ( \mathbf { X } _ { 1 } ) - \pmb { \sigma } _ { 1 : r } ^ { d a t a } ( \mathbf { X } _ { 2 } ) \| _ { 2 }$ where $\pmb { \sigma } _ { 1 : r } ^ { d a t a } ( \cdot )$ denotes the singular values of the data matrices $\mathbf { X } _ { c }$ from the first to the r-th.

Then there exist constants $C _ { 0 } > 0$ and $c _ { * } > 0$ such that, for any $\delta \in ( 0 ; \frac { 1 } { 2 } )$ , for a sample size $n ,$ with probability at least 1 − 2δ,

$$
\begin{array} { r } { \big \| \pmb { \sigma } _ { 1 : r } ^ { m o d e l } ( \widehat { \psi } _ { n } ( \widehat { \Sigma } _ { 1 } ) ) - \pmb { \sigma } _ { 1 : r } ^ { m o d e l } ( \widehat { \psi } _ { n } ( \widehat { \Sigma } _ { 2 } ) ) \big \| _ { 2 } \geq c _ { * } \Delta _ { r } - C _ { 0 } \Big ( \sqrt { \frac { \log ( 1 / \delta ) } { n ^ { 2 } } } + \sqrt { \eta _ { n } } + \tau _ { n } \Big ) , } \end{array}
$$

where $\tau _ { n } : = \operatorname* { m a x } ( \tau _ { n } ^ { 1 } , \tau _ { n } ^ { 2 } )$ is the maximum ofthe optimization errorsfor models trained on samples ofsize n using $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 } .$

In particular, $i f \Delta _ { r } > 0$ and $\eta _ { n } , \tau _ { n }  0$ , then as $n  \infty ,$ , the right-hand side remains positive, and the vectors ofthe first r singular values remain separated by a nonzero gap with probability tending to 1.

This result demonstrates that the embedding of a generative model contains sufficient information about the distribution of the original data. Moreover, such representations may possess a structure that reflects the mixing of datasets and the possibility of linearly separating them.

Corollary. In the case ofa VAE model, thefunctional $l _ { \psi } ( \mathbf { x } )$ is the ELBO, which possesses all the necessary properties for the conditions ofthe theorem to hold. Thus, the above reasoning applies to VAE-like architectures.

## 4 Computation experiment

To validate the obtained theoretical results, experiments were conducted on the CIFAR-10 and FashionMNIST datasets. Below, we provide a detailed description of the methodology, model configurations, experimental results, and their connection to the proven theorems.

## 4.1 Experimental settings

For the experiments on CIFAR-10, the most distinct classes were selected. For the samples from each class, embeddings were computed using a pretrained ResNet50 [13]. The average distance between the vectors of different classes was then computed in the Euclidean space. The three classes with the largest pairwise distances were subsequently used in the experiment: 5 — dogs, 6 — frogs, and 8 — ships.

The baseline models were an autoencoder (AE) and a variational autoencoder (VAE) with one and two fully connected layers in the encoder and decoder, both with and without a nonlinear activation function. The latent space had a dimension of 256. All layers were initialized using the standard initialization and trained with the AdamW optimizer using a learning rate of $1 \mathrm { { 0 } ^ { - 3 } }$ and a batch size of 32. Each model was trained for 20 epochs. To ensure the stability and statistical significance of the results, the experiments were repeated $N = 5$ times using different random subsamples. For a number of evaluations, the mean value and standard deviation are reported.

For each trained model, an embedding was computed using the algorithm described above. The constant r from the algorithm was chosen such that the singular values retained in the vector accounted for 80% of the sum of all singular values of the model parameters. Classification was performed using logistic regression with $L _ { 2 }$ regularization, trained on the model embeddings. The target variable was the data class on which the corresponding autoencoder was trained. Classification performance was evaluated using Accuracy, Average Precision, and ROC AUC.

Hereafter, classes from the CIFAR-10 or FashionMNIST datasets are understood as two distinct datasets, each generated from its own distribution $p _ { c }$ . The class of a dataset c refers to the image class defined in the original dataset.

## 4.2 Baseline experiment (two classes)

In the baseline experiment, we selected three most distinct classes from CIFAR-10. Subsamples of a fixed size n were drawn from the first two classes. Here and below, $n = 2 0 0 0$ unless otherwise specified. An AE/VAE with the architecture described above was trained on each subsample; the models were then vectorized using the previously described algorithm, and logistic regression was trained on the resulting vectors for binary classification.

An illustration of the experimental setup is shown in Fig. 2. The Accuracy of logistic regression in determining the class of the CIFAR-10 dataset was $0 . 9 8 \bar { ( \pm 0 . 0 2 ) }$ for an AE with two fully connected layers in the encoder and decoder and nonlinear activation functions; $0 . 9 7 ( \pm 0 . 0 3 )$ for a VAE; and $0 . 9 9 ( \pm 0 . 0 1 ) $ for an AE with one fully connected layer in the encoder and decoder and without nonlinear activation functions.

![](images/e8474b2a7dbfed45a8fd28c6bdcdb879980e42a44c164bd16452275f182fe291.jpg)  
Figure 2: Experimental setup with training a set of autoencoders on subsamples from two classes followed by classification of their vector representations

Conclusion The logistic regression metrics for determining the class of the dataset on which the AE/VAE was trained are close to 1. Therefore, the singular value vector of the model contains sufficient information about the statistical properties of the training dataset. The experiment was conducted for models both with and without nonlinearities. The results were similar in both cases.

## 4.3 Mixing third class experiment

The next experiment examined the ability of model singular value vectors to reflect the mixing of training datasets. In the experiment, $\mathbf { X } _ { c } , c \in \{ 1 , 2 \}$ , were selected as sets of samples from two classes of CIFAR-10 or FashionMNIST, following the same procedure as in the previous experiment. A third class ${ \bf X } _ { 3 }$ was then mixed into these datasets with varying proportions α. For a fixed set of model parameters, α was varied from 0.6 to 0.99, and new subsamples were generated for each value of α, on which autoencoders were trained. The quality of binary classification was then evaluated using the resulting vector representations. Specifically, we evaluated the accuracy of logistic regression in determining the class $c \in \{ 1 , 2 \}$ on which the vectorized model was trained.

Fig. 3 shows the dependence of classification performance on the proportion of ${ \bf X } _ { 3 }$ in the training dataset. A sharp decrease in the metrics can be observed as α increases: a small number of samples from the third class (small α) has almost no effect on separability, i.e., the performance metrics in Fig. 3 remain close to 1. In contrast, for large α, the classifier of model embeddings begins to make more errors, as the distributions of the training data of the models become increasingly close to the distribution of ${ \bf X } _ { 3 }$ , as schematically illustrated in Fig. 1.

As an extension of the experiment, different sampling strategies were used for mixing the third class into $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 }$ . For each α, a subsample was drawn independently and randomly for each of the 2N autoencoders (Different subsamples of class 3 in Fig. 3); for each α, the third-class subsample was drawn once and added identically to all 2N models (Same subsamples of class 3 (all) in Fig. 3); in the third approach, for each α, one of the N autoencoders from each of the two main classes was trained on datasets with the same mixture of the third class (Same subsamples of class 3 (pairs) in Fig. 3).

The experiment was conducted on two datasets: CIFAR-10 and FashionMNIST. We can observe a difference between the datasets in the proportion of the third class at which the performance of logistic regression in identifying the training dataset of a model begins to deteriorate substantially. This observation is explained in the following experiments and is related to the initial similarity between the classes selected for training.

![](images/3a63d16e9c1a8e12f35f5500b38fdec9966e81641f181deed30f89e289512b78.jpg)  
Different subsamples for class 3 Sames subsamples for class 3

![](images/afac5ca429608ab40e06eefaf1851ce00d64f8fafcbdaf141a27d6e0ae5b4028.jpg)

Same subsamples for class 3 (pairs)  
![](images/8a1b191cf3940501e9696bc0d05609f88b3e5f08b3469bb04b41bfbdd163ea63.jpg)

(a) CIFAR10, AE with 2 fully-connected layers in encoder and decoder, nonlinear activations  
![](images/35dcb863dddaba04d423121edb480e0d8e6a64f03f718a8ff8aa2b538ef7a4c2.jpg)

![](images/f8d8c2de3158ee054f377e9641484542106dde79237015d42b579a15b00b43c6.jpg)

![](images/a2371b11dcb7a571648e6c67f917a1de3d6727588e61f523fe0d08319ff19299.jpg)  
(b) IFAR10, AE with 1 fully-connected layer in encoder and decoder, without nonlinear activations (only for the case of different subsamples for 3 classes)

![](images/fd2dbad2b1233aaa6ff221e530962ed0317c9b12a71f52c77733eada71e79e00.jpg)

![](images/a91bc8623da419339e4243a11cae90e608313b4e304cbcb4d4c4560c0e030737.jpg)

![](images/93d7001eef915320fd98917773bdb0477c14f99c4c5d33e975a04709b25361e7.jpg)  
(c) FashionMNIST, AE with 2 fully-connected layer in encoder and decoder, nonlinear activations (only for the case of different subsamples for 3 classes)  
Figure 3: Demonstration of the performance of logistic regression on autoencoder embeddings for different proportions of the third class mixed into the models’ training datasets

Conclusion The proposed model vectorization method is sensitive to dataset mixing: at low proportions of the foreign class (small α), the representations of models trained on subsamples from different classes remain highly separable, as shown in the plots in Fig. 3. At larger $\alpha ,$ the representations of models trained on mixtures of $\mathbf { X } _ { c }$ and $\bar { \bf X } _ { 3 }$ shift for $c \in \{ 1 , 2 \}$ , and the classifier performs increasingly worse as α increases. This demonstrates that the embedding space reflects continuous transitions between data distributions. The experiment was conducted for autoencoders both without nonlinearities and with nonlinear activation functions. In all model variants, similar behavior was observed: as α increases, the accuracy of identifying the training dataset from the model embedding decreases.

## 4.4 Random samples addition experiment

In this experiment, ${ \bf X } _ { 3 }$ consists not of a single specific class from the CIFAR-10 dataset, but of any samples that do not belong to the classes from which $\mathbf { X } _ { c } , c \in \left\{ 1 , 2 \right\}$ , are selected. Thus, the previous experiment is extended to include a “noise” mixture. This type of noise simulates a real-world situation in which a dataset contains outliers or samples of unknown origin. The methodology follows the previous experiment: the proportion of noise samples was varied, new datasets $\tilde { \mathbf { X } } _ { 1 }$ and $\tilde { \mathbf { X } } _ { 2 }$ were sampled, models were trained on them, singular value vectors were constructed from the resulting models, and classification performance was evaluated using logistic regression.

![](images/f334c3fea7dac1070cf91642d2f83b32d17f7e2f2baa87204c44be55865321ac.jpg)  
Figure 4: Demonstration of the effect of mixing different types of noise on model classification performance. Mixing only class 3: ${ \bf X } _ { 3 }$ is sampled from a single class; mixing classes 3 and 4: samples are drawn from two classes; mixing from the entire dataset: samples are drawn from all CIFAR-10 classes except those used in $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 }$

Conclusion Adding random samples ${ \bf X } _ { 3 }$ reduces the separability (Fig. 1) of the model embeddings trained on the mixtures $\tilde { \mathbf { X } } _ { 1 } , \tilde { \mathbf { X } } _ { 2 }$ . Despite the small difference in the final accuracy, mixing noisy samples has a stronger effect on the separation performance, as shown by the yellow curve in Fig. 4. In the previous experiment, vectors from only one specific class were mixed in. The results of this experiment support Lemma 1 and Theorem 1. In particular, the greater the difference between the noise and $\tilde { \mathbf { X } } _ { 1 } , \tilde { \mathbf { X } } _ { 2 }$ , the faster the performance deteriorates as the proportion of noise increases.

## 4.5 Dataset singular values correlation

This experiment analyzes the dependence of the convergence of the singular value vectors of model weights and the singular value vectors of the data on the proportion of samples from a new class mixed into the dataset. The theoretical convergence of the vectors was described for the special case of fully connected layers without nonlinearities in Theorem 1. Three classes from the FashionMNIST dataset were selected. Samples from the first two classes were used to construct $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 }$ . The third class served as the mixed-in dataset ${ \bf X } _ { 3 }$ . The resulting datasets $\tilde { \mathbf { X } } _ { 1 }$ and $\tilde { \mathbf { X } } _ { 2 }$ were obtained by sampling $( 1 - \alpha ) n$ samples from $\mathbf { X } _ { 1 }$ and $\mathbf { X } _ { 2 }$ , respectively, and αn samples from $\mathbf { X } _ { 3 } ,$ , where $n = 2 0 0 0$ Then, for values of α ranging from 0.6 to 0.99, the singular value vector was computed for each of the datasets $\tilde { \mathbf { X } } _ { 1 }$ and $\tilde { \mathbf { X } } _ { 2 }$ . The mean absolute distance between the resulting vectors was computed over $N = 1 0 0$ different pairs of datasets $\tilde { \mathbf { X } } _ { 1 } , \tilde { \mathbf { X } } _ { 2 }$

In the second part of the experiment, variational autoencoders were trained on $\tilde { \mathbf { X } } _ { 1 }$ and $\tilde { \mathbf { X } } _ { 2 }$ , with $N = 1 0 0$ models trained for each class. The singular value vectors of the parameters obtained from the trained VAE models were then computed. For VAEs trained on datasets corresponding to different initial classes, the mean absolute distance between their embeddings was evaluated for different values of α.

The mean absolute distance for the first and second parts of the experiment is shown in Fig. 5 along the Y-axis and X-axis, respectively.

conclusion Fig. 5 shows a linear relationship between the mean absolute distance between the singular value vectors of the datasets and that between the singular value vectors of the models. According to a statistical test based on Pearson’s correlation, the linear relationship is statistically significant.

![](images/d99ae3685e25ad99f9f5eaf43c1bf00d3bb803f44ceed368e65682f41e940248.jpg)  
Figure 5: Comparison of the MAE between the data vectors and model embeddings. The X-axis shows the difference between the model embeddings, and the Y-axis shows the difference between the data vectors. Each point corresponds to a particular α, the proportion of the third class in the training dataset.

## 4.6 Pairwise comparison (One-vs-One)

As an extension of the baseline experiment, a “each class against each class” comparison was performed. For all pairs of FashionMNIST classes, procedures analogous to those in the baseline experiment were carried out: subsamples $\mathbf { \bar { X } } _ { 1 } , \mathbf { X } _ { 2 }$ without mixtures were constructed, a set of AEs with two fully connected layers and nonlinear activation functions was trained, the models were vectorized, and a binary classifier was trained. The results were averaged and visualized as a pairwise separability matrix in Fig. 6.

<table><tr><td rowspan=1 colspan=11>T-shirt/TopTrouserPulloverDressCoatSandalsShirtSneakerBagAnkle Boots</td></tr><tr><td rowspan=1 colspan=1>T-shirt/Top</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Trouser</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Pullover</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Dress</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>Coat</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Sandals</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Shirt</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Sneaker</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Bag</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>Ankle Boots</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td></tr></table>

Figure 6: Demonstration of the effect of mixing different types of noise on model classification performance: the lower triangular matrix shows the mean performance of logistic regression in the model-vector separation experiment; the upper triangular matrix shows the mean deviation of logistic regression performance; the diagonal shows the mean performance of logistic regression in the one-vs-rest experiment.

Conclusion The resulting pairwise matrix shows a wide range of separability: some class pairs (e.g., “coat” versus “dress”) yield nearly perfect classification based on the singular value vectors of the models. Other pairs, such as visually similar classes, are more difficult to distinguish. Consequently, the logistic regression performance for these pairs is comparatively low. These observations are consistent with expectations: the greater the difference between the statistics of the original classes, the more pronounced the signal in the singular spectra of the autoencoder parameters.

## 4.7 Experiment discussion

Across all series of experiments, the key hypothesis was confirmed: the singular value vector of the autoencoder parameters contains informative signals about the distribution of the training data, both with and without a nonlinear activation function. For statistically distinct classes, a classifier trained on such vectors achieves nearly perfect

separability, which is consistent with the conclusions of Theorem 2. When classes are mixed and noise is added, a predictable decrease in performance is observed, indicating that the embedding space correctly reflects statistical shifts in the data.

## 5 Conclusion

This work has shown that even when using a simple autoencoder architecture—one or two fully connected layers in the encoder and decoder—and a minimalist approach to feature representation based on singular value vectors, the parameters of trained models allow for nearly perfect discrimination between datasets with significant differences in the singular value spectra of their covariance matrices. This result is particularly noteworthy because it is achieved without the use of complex architectures, pretrained models, or heuristic methods for computing statistics of the input data.

Thus, a trained neural network can be viewed as a universal means of describing a dataset with a single vector formed exclusively from its parameters. Unlike traditional approaches based on specially selected statistical features or data histograms, the informative representation here emerges as a by-product of the training process.

The conducted experiments—from pairwise class comparisons to the analysis of mixtures and the addition of noisy samples—demonstrated that the space of such vectors adequately reflects the degree of difference between datasets and is sensitive to their perturbations. This opens up the possibility of applying the approach to tasks such as dataset retrieval and comparison, distribution shift analysis, and automatic categorization, without the need to store or directly access the data themselves.

A promising direction for future work is to investigate the observed effect using other approaches and to analyze their performance across different architectures with varying depth and complexity.

## References

[1] LAGOVSKY B., RUBINOVICH E., YURCHENKOV I. Solving the problem of super-resolution using a model of a neural network ofdirect propagation // Upravlenie bolsimi sistemami. – 2023. – Vol. 106. – P. 52–70.

[2] SARAEV P. Nonlinear least squares method and block recurrent and iterative procedures in neural networks teaching // Upravlenie bolsimi sistemami. – 2010. – Vol 30. – P. 24–34.

[3] ACHILLE A., SOATTO S. Information Dropout: learning optimal representations through noisy computation // IEEE Transactions on Pattern Analysis and Machine Intelligence. – 2018. – Vol. 40, No. 12. – P. 2897–2905.

[4] ADILOVA L., GEIGER B. C. Information plane analysis for dropout neural networks // arXiv – 01.03.2023. – URL: https://arxiv.org/abs/2303.00596 .

[5] AKHAURI Y., ABDELFATTAH M. S. Encodings for Prediction-based Neural Architecture Search // arXiv – 04.03.2024. – URL: https://arxiv.org/abs/2403.02484.

[6] ALAIN G., BENGIO Y. What Regularized Auto-Encoders Learnfrom the Data-Generating Distribution // The Journal of Machine Learning Research. – 2014. – Vol. 15, No. 1. – P. 3563–3593.

[7] ALET F., LOZANO-PEREZ T., KAELBLING L.P. Modular Meta-Learning // The AAAI Conference on Artificial Intelligence, New York, February 7-12, 2020, Proceedings. – 2020.

[8] ARPIT D., JASTRZEBSKI S., BALLAS N., KRUEGER D., BENGIO Y. A closer look at memorization in deep networks // The 34th International Conference on Machine Learning (ICML), Sydney, Australia, August 6-11, 2017, Proceedings. – 2017. – P. 233–242.

[9] CHENG H., ZHANG M., QINFENG J. A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations // IEEE Transactions on Pattern Analysis and Machine Intelligence. – 2024. – Vol. 46, No 12. – P. 10558–10578.

[10] CUI W., WU T., CRESSWELL J. C., SUI Y., GOLESTAN K. DRESS: Disentangled Representation-based Self-Supervised Meta-Learning for Diverse Tasks // arXiv. – 12.03.2025. – URL: https://arxiv.org/abs/2503.09679v1.

[11] ELSKEN T., METZEN J. H., HUTTER F. Neural Architecture Search: A Survey // Journal of Machine Learning Research. – 2019. – Vol. 20, No. 55. – P. 1–21.

[12] DRTON M., XIAO H. Smoothness ofGaussian conditional independence models // arXiv – 28.09.2009. – URL: https://arxiv.org/abs/0910.5447.

[13] HE K., ZHANG X., REN S., SUN J. Deep Residual Learningfor Image Recognition // 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, USA, June 27–30, 2016. – 2016. – P.770–778.

[14] HOEFFDING W. Probability Inequalitiesfor Sums ofBounded Random Variables // Journal of the American Statistical Association. – 1963. – Vol. 58, No. 301. – P. 13–30.

[15] ILHARCO G., GURURANGAN S., WALLACE E., SHANKAR V., ROBERTS A., BOWMAN S. R., SCHMIDT L., HAJISHIRZI H. Editing Models with Task Arithmetic // The 36th Conference on Neural Information Processing Systems (NeurIPS), New Orleans, USA, November 28 to December 9, 2022, Proceedings. – 2022.

[16] JOMAA H. S., SCHMIDT-THIEME L., GRABOCKA J. Dataset2Vec: Learning Dataset Meta-Features // Data Mining and Knowledge Discovery. – 2021. – Vol. 35, No. 3. – P. 964–985.

[17] KINGMA D. P., WELLING M. Auto-Encoding Variational Bayes // arXiv – 10.12.2022. – URL: https://arxiv.org/abs/1312.6114.

[18] KRIZHEVSKY A. Learning Multiple Layers of Features from Tiny Images // University of Toronto, 2009. – 60 p.

[19] KLABUNDE M., SCHUMACHER T., STROHMAIER M., LEMMERICH F. Similarity of Neural Network Models: A Survey ofFunctional and Representational Measures // ACM Computing Surveys. – Vol. 57, No. 9. – 2025.

[20] NEYSHABUR B., LI Z., BHOJANAPALLI S., LECUN Y., SREBRO N. Implicit regularization in deep learning // Proceedings of the 31st Conference on Neural Information Processing Systems (NeurIPS). – 2017.

[21] ZHANG C., BENGIO S., HARDT M., RECHT B., VINYALS O. Understanding deep learning (still) requires rethinking generalization // Communications of the ACM. – Vol. 64, No. 3. – 2021. – P. 107–115.

[22] REFINETTI M., GOLDT S. The dynamics of representation learning in shallow, non-linear autoencoders // Journal of Statistical Mechanics: Theory and Experiment. – 2023.

[23] RAGHU M., GILMER J., YOSINSKI J., SOHL-DICKSTEIN J. SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability // The 34th Conference on Neural Information Processing Systems (NeurIPS), Sydney, Australia, August 6-11, 2017, Proceedings. – 2017.

[24] REZENDE D. J., MOHAMED S., WIERSTRA D. Stochastic backpropagation and approximate inference in deep generative models // The 31st International Conference on Machine Learning (ICML), Beijing, China, June 21–26, 2014, Proceedings. – 2014 – P.1278-1286.

[25] ROEDER G., METZ L., KINGMA D. P. On Linear Identifiability ofLearned Representations// The International Conference on Machine Learning (ICML), Vienna, Austria, July 18–24, 2021, Proceedings. – 2021.

[26] SCHUERHOLT K., KOSTADINOV D., BORTH D., LUGO-MARTINEZ J. Hyper-Representations as Generative Models: Sampling Unseen Neural Network Weights // The 36th Conference on Neural Information Processing Systems (NeurIPS), New Orleans, USA, November 28 to December 9, 2022, Proceedings. – 2022.

[27] STEWART G. W., SUN J. G. Matrix perturbation theory // Elsevier Science, 1990. – 374 p.

[28] WEYL H. Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen (mit einer Anwendung aufdie Theorie der Hohlraumstrahlung) // Mathematische Annalen. – 1912. – Vol. 71, No. 4. – P. 441–479.

[29] XIAO H., RASUL K., VOLLGRAF R. Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms // arXiv – 15.10.2017. – URL: https://arxiv.org/abs/1708.07747.

[30] YANG C., SHEN Y. ZHOU B. TANG X. InterFaceGAN: Interpreting the Disentangled Face Representation Learned by GANs // IEEE Transactions on Pattern Analysis and Machine Intelligence. – 2022. – Vol. 4, No. 4. – P. 2004–2018.