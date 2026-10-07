# R<sub>a</sub>nd<sub>o</sub>m F<sub>ea</sub>t<sub>u</sub>r<sub>e</sub> G<sub>auss</sub>i<sub>a</sub>n Pr<sub>ocess</sub> Att<sub>e</sub>nti<sub>o</sub>n<sub>:</sub> Lin<sub>ea</sub>r<sub>-</sub>Tim<sub>e</sub> Pr<sub>o</sub>b<sub>a</sub>bili<sub>s</sub>ti<sub>c</sub> Att<sub>e</sub>nti<sub>o</sub>n <sub>w</sub>ith C<sub>a</sub>libr<sub>a</sub>t<sub>e</sub>d Un<sub>ce</sub>rt<sub>a</sub>int<sub>y</sub>

A<sub>m</sub>i<sub>r</sub> M<sub>o</sub>h<sub>amma</sub>d M<sub>a</sub>hf<sub>ooz</sub>i<sup>1,\*</sup> Zi Y<sub>ang</sub><sup>2,\*</sup> Yi<sub>ng</sub> Li<sup>2</sup> Mi<sub>c</sub>h<sub>ae</sub>l Mi<sub>ny</sub>i Zh<sub>ang</sub><sup>2</sup>

<sup>1</sup>Department of Computer Engineering, Sharif University of Technology

<sup>2</sup>School of Computing and Data Science, The University of Hong Kong

amir.mahfoozi01@sharif.edu {ziyang2023, lynnli98}@connect.hku.hk

mzhang18@hku.hk

## Abstract

Transformers provide a state-of-the-art modeling framework, yet poor calibration limits their reliability in safety-critical applications. A promising direction addresses this issue by interpreting attention as a Gaussian process (GP) posterior, which enables principled uncertainty calibration but incurs cubic complexity in sequence length due to the inversion of the kernel; although decoupled GP variants reduced the cost to quadratic, the computation remains prohibitive in practice. In this paper, we propose the plug-and-play random Fourier feature Gaussian process attention (RFF-GPA) module, which represents the attention as a GP with a stationary kernel approximated by random Fourier features. This low-rank approximation results in linear-time complexity for approximating the posterior mean and variance, making it far more scalable compared to previous work. Empirical results on multiple real-world datasets show that our attention module improves calibration while maintaining predictive accuracy, and simultaneously reduces computational complexity to linear in the sequence length.

## 1 Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

Large language models (LLMs) such as GPT-3 and its successors (Brown et al. 2020; Zhao et al. 2023) are built upon the Transformer architecture (Vaswani et al. 2017). The core of the Transformer is the self-attention mechanism, which enables scalable modeling of long-range dependencies. Transformers have achieved breakthroughs by leveraging this design, not only in natural language processing but also in vision and multimodal learning (Vaswani et al. 2017; Dosovitskiy et al. 2021; Brown et al. 2020), thereby becoming the leading architecture in modern artificial intelligence.

Despite their apparent successes, Transformers are often overconfident in their predictions and lack principled mechanisms for uncertainty quantification, which makes them unreliable in applications where the downstream tasks are consequential (Guo et al. 2017; Minderer et al. 2021; Papamarkou et al. 2024). Recent work has bridged the standard attention output to the mean of a Gaussian process (GP) posterior (Williams and Rasmussen 2006). Specifically, Chen and Li (2023); Bui et al. (2024) describe an exact and a sparse GP formulation where the number of inducing points equals the sequence length L. From this perspective, we can extend the framework to incorporate the posterior covariance, which naturally provides a principled and well-calibrated quantification of uncertainty that is absent in vanilla Transformers.

Nevertheless, existing approaches remain computationally expensive. Typically, inference requires inversion of a $L \times L ^ { \bar { ( } }$ dimensional kernel matrix, resulting in cubic $\mathcal { O } ( L ^ { 3 } )$ time and quadratic memory complexity, which makes these methods impractical for long sequences in real-world applications. While Chen and Li (2023) introduced a decoupled GP variant to reduce the complexity from cubic to quadratic by decoupling the inducing points for mean and covariance representations, the resulting $\mathcal { O } ( L ^ { 2 } )$ cost remains impractical for long sequences.

To obtain a model that combines calibrated uncertainty, strong predictive accuracy, and low computational cost, we propose random Fourier feature Gaussian process attention (RFF-GPA), a plug-and-play attention module for Transformer architectures.

Figure 1 summarizes the progression from standard attention to GP attention and illustrates how our random-feature formulation preserves posterior uncertainty while moving the expensive computation from token space to feature space. Our main contributions are summarized as follows:

• We introduce RFF-GPA, which approximates the attention kernel with random Fourier features (RFF) (Rahimi and Recht 2007) and performs GP posterior inference in the resulting feature space. Using the Woodbury matrix identity (Hager 1989), RFF-GPA avoids direct inversion of the full $L \times L$ kernel matrix and computes predictive means and covariance estimates in linear time with respect to the sequence length.

• To relax the symmetry constraint between key and query induced by stationary kernels, we further extend the model to a correlated random-feature GP attention variant, RFF-CGP. This variant models query and key representations through a shared latent GP structure, enabling a more flexible attention mechanism while retaining the computational advantages of the random-feature formulation.

• We evaluate the proposed models on image and text classification benchmarks, as well as under distribution shift on CIFAR-10-C. The results show that our methods improve calibration while maintaining competitive accuracy. They also achieve better runtime and memory scaling than inducing-point GP attention, highlighting the favorable accuracy–calibration–eficiency trade-of of our approach.

## 2 B<sub>ac</sub>k<sub>g</sub>r<sub>ou</sub>nd

In this section, we first review the self-attention mechanism in Transformers and its interpretation as a kernel method followed by a brief overview of Gaussian processes and random Fourier features as a scalable kernel approximation. Together, these components motivate our probabilistic treatment of attention within a Gaussian process framework.

## 2<sub>.</sub>1 S<sub>e</sub>lf<sub>-</sub>Att<sub>en</sub>ti<sub>on</sub>

Given an input sequence $\mathbf { X } = [ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { L } ] \in \mathbb { R } ^ { L \times D }$ of L tokens, each represented by a D-dimensional vector, the self-attention mechanism first projects the inputs into three diferent latent spaces (Vaswani et al. 2017):

$$
\mathbf { Q } = \mathbf { X } \mathbf { W } _ { Q } , \mathbf { K } = \mathbf { X } \mathbf { W } _ { K } , \mathbf { V } = \mathbf { X } \mathbf { W } _ { V } ,\tag{1}
$$

where $\mathbf { W } _ { Q } \ \in \ \mathbb { R } ^ { D \times D _ { Q } } , \ \mathbf { W } _ { K } \ \in \ \mathbb { R } ^ { D \times D _ { K } }$ , and ${ \bf W } _ { V } \in \mathbb { W }$ $\mathbb { R } ^ { D \times D _ { V } }$ are learnable weight matrices that project the input X into Q, K, and V, referred to as queries, keys, and values, respectively. In standard self-attention, we set the latent dimensionalities of the queries, keys and values to be equal, $D _ { Q } = D _ { K } = D _ { V } = D$ , and let $\mathbf { W } _ { Q } = \mathbf { W } _ { K }$ , yielding $\mathbf { Q } \overset { \cdot } { = } \mathbf { K }$ for simplicity (Vaswani et al. 2017).

Then, the attention matrix is

$$
\mathbf { A } = \mathrm { s o f t m a x } \big ( \mathbf { Q } \mathbf { K } ^ { \top } / \sqrt { D _ { K } } \big ) ,\tag{2}
$$

where the softmax is applied row-wise. The output is $\mathbf { Y } =$ AV. Here, each row of A specifies how a token (i.e., an element of the input sequence) aggregates information from all others, weighted by their query–key similarity.

Furthermore, multi-head self-attention (MHSA) applies the above operation in parallel across H heads, each with its own parameter set $( \mathbf { W } _ { Q } ^ { ( h ) } , \mathbf { W } _ { K } ^ { ( h ) } , \mathbf { W } _ { V } ^ { ( h ) } )$ ). For the h-th head, the output is

$$
\mathbf { Y } ^ { ( h ) } = \mathrm { s o f t m a x } \big ( \mathbf { Q } ^ { ( h ) } \mathbf { K } ^ { ( h ) \top } / \sqrt { D _ { K } } \big ) \mathbf { V } ^ { ( h ) } ,\tag{3}
$$

and the final output is obtained by concatenating all head outputs and projecting back:

$$
\mathbf { H } = \operatorname { c o n c a t } \big ( \mathbf { Y } ^ { ( 1 ) } , \ldots , \mathbf { Y } ^ { ( H ) } \big ) \mathbf { W } _ { o } ,\tag{4}
$$

where $\mathbf { W _ { o } } ~ \in ~ \mathbb { R } ^ { ( H D _ { V } ) \times D }$ is a learnable projection. This design allows diferent heads to capture diverse similarity patterns among tokens, enhancing the expressiveness of the model.

## 2<sub>.</sub>2 K<sub>erne</sub>l Att<sub>en</sub>ti<sub>on</sub>

The attention matrix can be further viewed as a softmax similarity with a positive-definite kernel, $\kappa ( \cdot , \cdot )$ . Tsai et al. (2019) established a direct connection between the attention module and the classic kernel methods. Generally, we may represent the attention module as a kernel function:

$$
\begin{array} { r l } & { { \kappa _ { Q K } } \in { \mathbb R } ^ { L \times L } , } \\ & { [ { \pmb { \kappa } _ { Q K } } ] _ { i j } = \kappa ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) = \kappa ( \mathbf { x } _ { i } \mathbf { W } _ { Q } , \mathbf { x } _ { j } \mathbf { W } _ { K } ) , } \end{array}\tag{5}
$$

and the outputs of the Transformer take the form

$$
\mathbf { Y } = { \boldsymbol { \kappa } } _ { Q K } \mathbf { V } .\tag{6}
$$

Note that, to guarantee kernel symmetry in the attention mechanism, we must enforce

$$
\kappa ( { \bf x } _ { i } { \bf W } _ { Q } , { \bf x } _ { j } { \bf W } _ { K } ) = \kappa ( { \bf x } _ { j } { \bf W } _ { K } , { \bf x } _ { i } { \bf W } _ { Q } ) ,\tag{7}
$$

that requires $\mathbf { W } _ { Q } = \mathbf { W } _ { K }$ , which in turn reduces the expressive capacity of the Transformer. Moreover, the connection (Eq. 6) between kernels and attention modules is crucial for making the probabilistic connection between Transformers and GPs.

## 2<sub>.</sub>3 G<sub>auss</sub>i<sub>a</sub>n Pr<sub>ocesses</sub>

A Gaussian process (GP),

$$
f \sim \mathcal { G P } \left( 0 , \kappa ( \cdot , \cdot ) \right) ,
$$

is a distribution over functions $f : \mathbb { R } ^ { D } $ R such that any finite collection of function values is jointly Gaussian (Williams and Rasmussen 2006). Formally, for inputs $\mathbf { X } = [ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { N } ]$ , we have

$$
\mathbf { f } \left| \mathbf { X } \sim { \mathcal { N } } \left( 0 , { \mathcal { K } } _ { X X } \right) \right.\tag{8}
$$

We assume we have noisy observations $y _ { i } = f ( \mathbf { x } _ { i } ) + \varepsilon _ { i }$ with $\varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . We let $\mathbf { y } = [ y _ { 1 } , \dots , y _ { N } ] ^ { \intercal }$ and we define the Gram matrix, $\pmb { \kappa } _ { X X } \in \mathbb { R } ^ { N \times N }$ , to have entries $[ \pmb { \mathcal { K } } _ { X X } ] _ { i j } \ : = \ : \kappa ( \mathbf { x } _ { i } , \mathbf { x } _ { j } )$ . For test inputs ${ \bf { X } } _ { \star }$ , the predictive posterior distribution is a multivariate Gaussian distribution, $\mathcal { N } ( \mathbf { m } _ { X _ { \star } } , \mathbf { S } _ { X _ { \star } } )$ , with mean and variance:

$$
\mathbf { m } _ { X _ { \star } } = \mathcal { K } _ { X _ { \star } X } \bigl ( \mathcal { K } _ { X X } + \sigma ^ { 2 } \mathbf { I } \bigr ) ^ { - 1 } \mathbf { y } ,\tag{9}
$$

$$
\mathbf { S } _ { X _ { \star } } = \mathcal { K } _ { X _ { \star } X _ { \star } } - \mathcal { K } _ { X _ { \star } X } \big ( \mathcal { K } _ { X X } + \sigma ^ { 2 } \mathbf { I } \big ) ^ { - 1 } \mathcal { K } _ { X X _ { \star } } .\tag{10}
$$

## 2<sub>.</sub>4 R<sub>a</sub>nd<sub>o</sub>m F<sub>ou</sub>ri<sub>e</sub>r F<sub>ea</sub>t<sub>u</sub>r<sub>es</sub>

Random Fourier features are a technique to accelerate computation of kernel methods through a low-rank factorization. The basic theorem necessary to approximate a stationary kernel is Bochner’s theorem (Bochner 1959), which states that any continuous shift-invariant kernel, $\kappa ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \kappa ( \mathbf { x } - \mathbf { x } ^ { \prime } )$ can be expressed as the Fourier transform of a nonnegative spectral density $p ( \omega )$ , namely:

$$
\kappa ( \mathbf { x } - \mathbf { x } ^ { \prime } ) = \int p ( \omega ) e ^ { i \omega ^ { \top } ( \mathbf { x } - \mathbf { x } ^ { \prime } ) } \mathrm { d } \omega .\tag{11}
$$

Based on this result, Rahimi and Recht (2007) proposed to first draw $M / 2$ frequencies $\Omega \triangleq \{ \omega ^ { ( m ) } \} _ { m = 1 } ^ { M / 2 } \sim p ( \omega )$ , and then obtain an unbiased estimator for kernel function:

$$
\kappa ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \mathbb { E } \left[ \phi ( \mathbf { x } ) ^ { \top } \phi ( \mathbf { x } ^ { \prime } ) \right] ,\tag{12}
$$

where the randomized feature mapping is:

$$
\begin{array} { r } { \phi ( \mathbf { x } ) = \sqrt { \frac { 2 } { M } } \left[ \begin{array} { c } { \cos ( \omega ^ { ( 1 ) \top } \mathbf { x } ) } \\ { \sin ( \omega ^ { ( 1 ) \top } \mathbf { x } ) } \\ { \vdots } \\ { \cos ( \omega ^ { ( M / 2 ) \top } \mathbf { x } ) } \\ { \sin ( \omega ^ { ( M / 2 ) \top } \mathbf { x } ) } \end{array} \right] . } \end{array}\tag{13}
$$

As a result, the full kernel matrix K<sub>XX</sub> is approximated by $\Phi ( { \mathbf X } ) \Phi ( { \mathbf X } ) ^ { \top }$ with $\Phi ( { \mathbf { X } } ) = [ \phi ( \mathbf { x } _ { 1 } ) , \dots , \phi ( \mathbf { x } _ { N } ) ] ^ { \top }$ , allowing scalable GP computations through low-rank approximations instead of the direct inversion of $\kappa _ { \mathbf { x x } }$

![](images/a5a87c4ca77d670da34c2383528786ba936c48da70b0740cf57eb4bd01daa984.jpg)  
Figure 1: Overview of RFF-GPA. (A) Standard attention produces deterministic token representations without explicit uncertainty. (B,D) A GP formulation provides posterior mean and variance, but exact GP attention requires an expensive token-space inverse. (C) Random Fourier features provide a compact approximation of the stationary kernel. (E) RFF-GPA performs GP inference in the resulting feature space, replacing the L × L inverse with an M × M inverse while retaining the posterior mean and token-wise predictive variance.

## 2<sub>.</sub>5 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>o</sub>rk<sub>s</sub>

Subsequent extensions of the Transformer have accordingly combined the RFF approximations and the attention kernel. Peng et al. (2021) introduced random feature attention (RFA) as a direct replacement of the linear projections of the queries and keys in the original Transformer with the basis function in Eq. (13). In parallel, Choromanski et al. (2021) introduced positivity and orthogonality constraints for the random feature approximation of the attention module. These approaches are eficient and but they only model the interaction between the queries and keys as a kernel. Moreover, they do not pose the Transformer as a fully Bayesian model and therefore lack principled uncertainty quantification.

In Bayesian deep learning, Neal (1995) originally established the connection between Gaussian processes and neural networks. Damianou and Lawrence (2013) then introduced the deep Gaussian process, which applies the feedforward network structure to Gaussian process regression. Gal and Ghahramani (2016) further established the connection between training deep neural networks with approximate Bayesian inference for deep Gaussian processes.

Comparatively, the Bayesian perspective on Transformers remains underexplored. Tran et al. (2019); Fan et al. (2020); Xue et al. (2021) implemented Bayesian Transformers with varying architectures and priors on the network parameters, but these methods overlook the kernelized attention structure which limits their modeling capacity. As we will demonstrate, unifying attention modules, random feature kernel approximations, and Gaussian processes will be crucial to obtain properly calibrated probabilistic Transformers.

## 3 M<sub>e</sub>th<sub>o</sub>d<sub>o</sub>l<sub>ogy</sub>

This section first presents the proposed random Fourier feature Gaussian process attention module (Sec. 3.1), then details its integration into the standard Transformer (Sec. 3.2), and finally provides an illustrative implementation in GPbased Transformers (Sec. 3.3).

Algorithm 1 RFF-GPA (Single-Head): Random Fourier Fea  
ture Gaussian Process Attention   
Require: Sequence of tokens $\mathbf { X } = \left( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { L } \right)$   
1: Compute Q, K, V using Eq. (1).   
2: Draw S samples $\hat { \Omega } ^ { ( s ) }$ from p(Ω) using the reparameter  
ization trick.   
3: Compute the feature maps $\Phi _ { Q }$ and $\Phi _ { K }$ using Q, K, and   
sampled spectral points $\hat { \Omega } ^ { ( s ) }$   
4: Compute Y and $\dot { \Sigma }$ using Eq. (20).   
5: Draw S samples $\hat { \mathbf { Y } } ^ { ( s ) }$ from p(Y | Ω, X) using the repa  
rameterization trick.   
6: Evaluate the ELBO using Eq. (26).   
7: Maximize the ELBO and update Θ, ζ using Adam   
(Kingma and Ba 2015).   
Ensure: Model parameters Θ and variational parameters $\zeta .$

## 3<sub>.</sub>1 R<sub>a</sub>nd<sub>o</sub>m F<sub>ou</sub>ri<sub>e</sub>r F<sub>ea</sub>t<sub>u</sub>r<sub>e</sub> G<sub>auss</sub>i<sub>a</sub>n Pr<sub>ocess</sub> Attention (RFF-GPA)

In a typical GP-based Transformer (Bui et al. 2024; Chen and Li 2023), the output ofthe kernel attention module is modeled as the predictive posterior mean of a Gaussian process. We interpret the keys as the “training inputs”, the value matrix $\mathbf { V } = \mathbf { X } \mathbf { W } _ { V } ^ { \top }$ as the corresponding (noisy) observations, and the queries as the “test inputs”. Let $\kappa _ { K K }$ denote the kernel Gram matrix with keys as inputs. Denoting the d-th column of Y and V as $\mathbf { y } _ { d }$ and $\mathbf { v } _ { d } ,$ respectively, the d-th column of the kernel attention output is precisely the GP predictive posterior mean (see Eq. 9):

$$
\mathbf { y } _ { d } = \kappa _ { Q K } \left( \kappa _ { K K } + \sigma ^ { 2 } \mathbf { I } \right) ^ { - 1 } \mathbf { v } _ { d } .\tag{14}
$$

Based on this interpretation, it is straightforward to obtain the attention covariance as

$$
\pmb { \Sigma } = \pmb { \mathcal { K } } _ { Q Q } - \pmb { \mathcal { K } } _ { Q K } \left( \pmb { \mathcal { K } } _ { K K } + \sigma ^ { 2 } \mathbf { I } \right) ^ { - 1 } \pmb { \mathcal { K } } _ { K Q } .\tag{15}
$$

The overall output of the kernel attention module Y can thus be interpreted as the predictive mean of D independent ${ \mathrm { G P s } } ,$ with the corresponding predictive covariance capturing the associated uncertainty. This view extends naturally to multi-head self-attention, where the predictive outputs from all heads are concatenated and mapped through the output projection $\mathbf { W } _ { o }$

Unfortunately, inverting the kernel Gram matrix $\kappa _ { K K }$ incurs cubic complexity, which is impractical for long sequences. Two common strategies to reduce the cost of GP inference are: (i) inducing points (sparse GPs) and (ii) RFFs. Chen and Li (2023) interpret the attention module as a sparse GP by treating queries as inducing points, which makes the number of inducing points equal to L but still results in $\mathcal { O } ( L ^ { 3 } )$ complexity. Alternatively, we turn to RFFs (Eq. 12), where the kernel Gram matrices can be approximated as $\ = \kappa _ { \ / K K } \approx \Phi _ { \ / K } \ = \Phi _ { \ / K } ^ { \top }$ and $\pmb { \kappa } _ { Q K } \approx \Phi _ { Q } \pmb { \Phi } _ { K } ^ { \top }$ . Substituting those approximations into Eq. (14) and Eq. (15) yields<sup>1</sup>:

$$
\begin{array} { r } { \begin{array} { c } { \mathbf { Y } \approx \hphantom { \frac { 1 } { 2 } } \Phi _ { Q } \Phi _ { K } ^ { \top } \left( \Phi _ { K } \Phi _ { K } ^ { \top } + \sigma ^ { 2 } \mathbf { I } \right) ^ { - 1 } \mathbf { V } , } \\ { \nu ( \mathbf { y } _ { d } ) \approx \hphantom { \frac { 1 } { 2 } } \Phi _ { Q } \Phi _ { Q } ^ { \top } - \Phi _ { Q } \Phi _ { K } ^ { \top } \left( \Phi _ { K } \Phi _ { K } ^ { \top } + \sigma ^ { 2 } \mathbf { I } \right) ^ { - 1 } \Phi _ { K } \Phi _ { Q } ^ { \top } . } \end{array} } \end{array}\tag{16}
$$

For simplicity, we denote the computational demanding term as

$$
\mathbf { A } \triangleq \left( \Phi _ { K } \Phi _ { K } ^ { \top } + \sigma ^ { 2 } \mathbf { I } \right) ^ { - 1 } .\tag{17}
$$

By exploiting the low-rank structure of $\Phi _ { K }$ , we can apply the Woodbury identity and obtain

$$
\begin{array} { r } { \mathbf { A } = \frac { 1 } { \sigma ^ { 2 } } \mathbf { I } - \frac { 1 } { \sigma ^ { 4 } } \Phi _ { K } \Big ( \mathbf { I } + \frac { 1 } { \sigma ^ { 2 } } \Phi _ { K } ^ { \top } \Phi _ { K } \Big ) ^ { - 1 } \Phi _ { K } ^ { \top } , } \end{array}\tag{18}
$$

where we only need to invert an $M \times M$ matrix, with M pre-selected, thereby reducing the overall cost to $\mathcal { O } ( L M ^ { 2 } )$ i.e., linear in L.

Replacing the Eq. (18) within Eq. (16), we can obtain

$$
\mathbf { Y } \approx \Phi _ { Q } \Phi _ { K } ^ { \top } \mathbf { A } \mathbf { V } ,\tag{19}
$$

$$
\begin{array} { r } { \pmb { \Sigma } \approx \Phi _ { Q } \boldsymbol { \Phi } _ { Q } ^ { \top } - \Phi _ { Q } \boldsymbol { \Phi } _ { K } ^ { \top } \mathbf { A } \boldsymbol { \Phi } _ { K } \boldsymbol { \Phi } _ { Q } ^ { \top } . } \end{array}\tag{20}
$$

Instead of predicting the full covariance function, we only report the diagonal variance, denoted as diag(Σ), that represents the uncertainty for each token.

In supervised Transformer applications, the latent Gaussian attention output Y serves as the representation passed to the final prediction head. Consequently, the observed labels are modeled through a likelihood $p ( { \dot { \mathbf { y } } } \mid \mathbf { Y } )$ , yielding the overall supervised joint distribution

$$
p ( \mathbf { \mathcal { Y } } , \mathbf { \mathbf { Y } } , \boldsymbol { \Omega } \mid \mathbf { X } ) = p ( \mathbf { \mathcal { Y } } \mid \mathbf { Y } ) p ( \mathbf { Y } \mid \mathbf { X } , \boldsymbol { \Omega } ) p ( \boldsymbol { \Omega } ) ,\tag{21}
$$

where $p ( \mathbf { \mathcal { P } } \mid \mathbf { Y } )$ denotes the categorical likelihood induced by the prediction head.

## 3<sub>.</sub>2 V<sub>a</sub>ri<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l Inf<sub>e</sub>r<sub>e</sub>n<sub>ce</sub> f<sub>o</sub>r St<sub>a</sub>nd<sub>a</sub>rd T<sub>rans</sub>f<sub>ormers w</sub>ith RFF<sub>-</sub>GPA M<sub>o</sub>d<sub>u</sub>l<sub>e</sub>

The RFF-GPA module introduces latent attention outputs and spectral points whose exact posterior is not available in closed form, so we use variational inference to obtain a tractable training objective. We now describe how to train a Transformer equipped with the proposed RFF-GPA module within a variational inference (VI) framework. Our goal is to jointly learn the model parameters<sup>2</sup>, $\boldsymbol { \Theta } = \{ \mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { \bar { W } } _ { V } , \mathbf { \bar { W } } _ { o } , \sigma ^ { 2 } \}$ , and infer the posterior distribution of the spectral points Ω given the observations Y. Since the exact posterior $\bar { \boldsymbol { p } } ( \Omega \mid \mathcal { D } )$ is intractable, VI posits a tractable family $q ( \Omega )$ to approximate it and optimizes both the variational parameters of $q ( \Omega )$ and the model parameters Θ by maximizing the evidence lower bound (ELBO).

Following the settings in Li et al. (2024); Yang et al. (2025), we assume that the variational distribution $q ( \Omega )$ coincides with the prior<sup>3</sup>,

$$
p ( \Omega ) \triangleq \prod _ { m = 1 } ^ { M } \mathcal { N } ( \mu , \mathbf { S } ) ,\tag{22}
$$

<sup>1</sup>For clarity, the following derivations focus on a single head, but the extension to the multi-head case is straightforward.

where µ denotes the mean and S is a diagonal covariance matrix. Consequently, the ELBO is:

$$
\scriptstyle { \mathcal { L } } _ { \mathrm { E L B O } } = \mathbb { E } _ { p ( \Omega ) } \mathbb { E } _ { p ( \mathbf { Y } \mid \Omega , \mathbf { X } ) } [ \log p ( \mathbf { \mathcal { Y } } \mid \mathbf { Y } , \mathbf { X } , \Omega ) ] ,\tag{23}
$$

where the expectation over the log-likelihood, log $p ( \mathbf { Y } \mid$ $\Omega , \mathbf { X } )$ , is required since the attention output Y is a random latent variable rather than a deterministic value, with its mean and covariance given by Eq. (19) and Eq. (20), respectively.

This term indeed measures the expected reconstruction accuracy. For evaluation, we employ the reparameterization trick to draw S samples from p(Ω) and $p ( \dot { \mathbf Y } \mid \boldsymbol { \Omega } , \mathbf { X } )$

$$
\begin{array} { r } { { \boldsymbol { \omega } } _ { m } ^ { ( s ) } = \pmb { \mu } + \sqrt { \mathbf { S } } \pmb { \xi } ^ { ( s ) } , \qquad { \boldsymbol { \xi } } ^ { ( s ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , } \end{array}\tag{24}
$$

$$
\hat { \mathbf { y } } _ { d } ^ { ( s ) } = \mathbf { y } _ { d } + \sqrt { \mathrm { d i a g } ( \pmb { \Sigma } ) } \odot \boldsymbol { \varepsilon } ^ { ( s ) } , \quad \boldsymbol { \varepsilon } ^ { ( s ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{25}
$$

and thereby obtain an unbiased Monte Carlo estimator of the reconstruction term:

$$
\mathcal { L } _ { \mathrm { E L B O } } \approx \sum _ { s = 1 } ^ { S } \left[ \log p \Big ( \pmb { \mathscr { y } } \mid \hat { \mathbf { Y } } ^ { ( s ) } , \mathbf { X } , \hat { \pmb { \Omega } } ^ { ( s ) } \Big ) \right] ,\tag{26}
$$

Consequently, we optimize the above objective with respect to both the model parameters $\begin{array} { r l r } { \Theta } & { { } = } & { \{ { \bf W } _ { Q } , { \bf \dot { W } } _ { K } , { \bf W } _ { V } , { \bf W } _ { o } , \sigma ^ { 2 } \} } \end{array}$ and the variational parameters $\zeta  { \mathrm { ~  ~ \triangleq ~ } } \{ \mu , \mathbf { S } \}$ . The complete procedure is summarized in Algorithm 1.

## 3<sub>.</sub>3 R<sub>a</sub>nd<sub>o</sub>m F<sub>ou</sub>ri<sub>e</sub>r F<sub>ea</sub>t<sub>u</sub>r<sub>e</sub> C<sub>o</sub>rr<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d Gaussian Process Attention (RFF-CGP)

Although RFF-GPA provides scalable uncertainty-aware attention, its stationary-kernel construction still inherits a symmetry constraint from standard kernel attention: the query and key projections are tied through the same kernel structure, limiting the ability to model asymmetric query-key interactions. This can be undesirable because queries and keys play diferent roles in attention, and forcing them into a symmetric kernel space can restrict representation capacity. To address this limitation, we extend our random-feature formulation to correlated Gaussian process attention (CGP) (Bui et al. 2024), yielding random Fourier feature correlated Gaussian process attention (RFF-CGP).

CGP attention models the self-attention output as the cross-covariance between two correlated Gaussian processes, each defined by an afine transformation of a latent canonical GP. This construction naturally removes the requirement that ${ \bf W } _ { Q } = { \bf W } _ { K }$ , which allows asymmetric kernels and enhances representation capacity. In particular, the d-th output column can be expressed as

$$
\mathbf { y } _ { d } = \mathbf { K } _ { Q O } ( \mathbf { K } _ { O O } + \sigma ^ { 2 } I ) ^ { - 1 } \mathbf { K } _ { O K } ( \mathbf { K } _ { K K } + \sigma ^ { 2 } I ) ^ { - 1 } \mathbf { v } _ { d } ,\tag{27}
$$

where ${ \bf K } _ { Q O } , { \bf K } _ { O K } , { \bf K } _ { K K } , { \bf K } _ { O O }$ are the Gram or crosscovariance matrices induced by the canonical kernel, and $\mathbf { v } _ { d }$ is the d-th column of the value matrix V, as in Eq. (14).

To make Eq. (27) scalable, we first substitute each kernel matrix with its RFF approximations. Let $\Phi _ { Q } , \Phi _ { K } , \Phi _ { O } \in$ $\mathbb { R } ^ { L \times M }$ be the RFF embeddings of queries, keys, and the

latent canonical GP inputs, respectively. Then, we have:

$$
\begin{array} { r } { \mathbf { K } _ { Q O } \approx \Phi _ { Q } \boldsymbol { \Phi } _ { O } ^ { \top } , \quad \mathbf { K } _ { O K } \approx \Phi _ { O } \boldsymbol { \Phi } _ { K } ^ { \top } , } \\ { \mathbf { K } _ { K K } \approx \Phi _ { K } \boldsymbol { \Phi } _ { K } ^ { \top } , \quad \mathbf { K } _ { O O } \approx \Phi _ { O } \boldsymbol { \Phi } _ { O } ^ { \top } . } \end{array}\tag{28}
$$

Substituting into Eq. (27) gives the following predictive mean:

$$
\mathbf { y } _ { d } \approx \Phi _ { Q } \Phi _ { O } ^ { \top } \left( \Phi _ { O } \Phi _ { O } ^ { \top } + \sigma ^ { 2 } I \right) ^ { - 1 } \Phi _ { O } \Phi _ { K } ^ { \top } \left( \Phi _ { K } \Phi _ { K } ^ { \top } + \sigma ^ { 2 } I \right) ^ { - 1 } \mathbf { v } _ { d } .\tag{29}
$$

Both inverses in Eq. (29) involve matrices of size L × L. Using the Woodbury identity, they can be rewritten as inversions of $M \times M$ matrices:

$$
\begin{array} { r } { \left( \Phi _ { K } \Phi _ { K } ^ { \top } + \sigma ^ { 2 } I \right) ^ { - 1 } = \frac { 1 } { \sigma ^ { 2 } } I - \frac { 1 } { \sigma ^ { 4 } } \Phi _ { K } \Big ( I + \frac { 1 } { \sigma ^ { 2 } } \Phi _ { K } ^ { \top } \Phi _ { K } \Big ) ^ { - 1 } \Phi _ { K } ^ { \top } , } \end{array}
$$

$$
\begin{array} { r } { \left( \Phi _ { \mathcal { O } } \Phi _ { \mathcal { O } } ^ { \top } + \sigma ^ { 2 } I \right) ^ { - 1 } = \frac { 1 } { \sigma ^ { 2 } } I - \frac { 1 } { \sigma ^ { 4 } } \Phi _ { \mathcal { O } } \left( I + \frac { 1 } { \sigma ^ { 2 } } \Phi _ { \mathcal { O } } ^ { \top } \Phi _ { \mathcal { O } } \right) ^ { - 1 } \Phi _ { \mathcal { O } } ^ { \top } . } \end{array}
$$

Analogous to Eq. (20), the predictive covariance of RFF-CGP can be derived in closed form by substituting the RFF approximations into the CGP conditional covariance (Bui et al. 2024). Specifically, in the CGP construction, the query process is conditioned on the canonical latent GP O; the resulting token-wise predictive variance is therefore the residual query variance after this conditioning,

$$
\mathrm { d i a g } ( \Sigma ) = \sigma ^ { 2 } \mathrm { d i a g } \Big ( \Phi _ { Q } \big ( \Phi _ { O } ^ { \top } \Phi _ { O } + \sigma ^ { 2 } I \big ) ^ { - 1 } \Phi _ { Q } ^ { \top } \Big ) ,\tag{30}
$$

which is the exact one-step posterior variance and reduces to the RFF-GPA variance in Eq. (20) when $\Phi _ { O }  \Phi _ { Q }$ (i.e. ${ \bf K } _ { O O }  { \bf K } _ { Q Q } )$ . The data-dependent terms of the full CGP conditional covariance cancel against the mean outer product, so Eq. (30) is independent of V, positive semi-definite, and computed in $\mathcal { O } ( L M ^ { 2 } )$ via the same $M \times M$ inversion $( \Phi _ { O } ^ { \top } \Phi _ { O } + \sigma ^ { 2 } I ) ^ { - 1 }$ as in Eq. (29).

## 4 E<sub>xpe</sub>rim<sub>e</sub>nt<sub>s</sub>

We first evaluate our proposed method<sup>4</sup> (RFF-GPA and RFF-CGP) on in-distribution accuracy and calibration in Sec. 4.1. We then demonstrate its eficiency and scalability in terms of computation and memory on synthetic sequence benchmarks in Sec. 4.2, and finally evaluate out-of-distribution robustness under distribution shift on CIFAR-10-C in Sec. 4.3.

## 4<sub>.</sub>1 In<sub>-</sub>Di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n A<sub>ccu</sub>r<sub>acy a</sub>nd C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n

We first evaluate in-distribution accuracy and calibration across a broad suite of six classification benchmarks spanning both modalities: three image datasets (Fashion-MNIST, CIFAR-10, SVHN) and three text datasets (20 Newsgroups, Hyperpartisan, SST-2).<sup>5</sup> All methods share an identical Transformer backbone and training protocol, difering only in the attention/output uncertainty mechanism, so that diferences in Table 1 reflect the uncertainty mechanism rather than the random-feature GP formulation not only reduces cost but can also yield a more efective attention mechanism than inducing-point sparse GPs.

<table><tr><td rowspan="2">二 METHOD</td><td colspan="3">IMAGE</td><td colspan="3">TEXT</td></tr><tr><td>F-MNIST</td><td>CIFAR-10</td><td>SVHN</td><td>20NG</td><td>Hyperp.</td><td>SST-2</td></tr><tr><td>METRIC</td><td colspan="6">ACC ↑</td></tr><tr><td>MLE</td><td>三  $0 . 8 8 9 \pm 0 . 0 0 3$ </td><td> $0 . 7 1 5 \pm 0 . 0 0 8$ </td><td> $0 . 9 0 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 5 4 \pm 0 . 0 1 5$ </td><td> $0 . 7 4 4 \pm 0 . 0 1 5$ </td><td> $0 . 7 7 5 \pm 0 . 0 0 9$ </td></tr><tr><td>MLE+Temp</td><td> $0 . 8 8 9 \pm 0 . 0 0 3$ </td><td> $0 . 7 1 5 \pm 0 . 0 0 8$ </td><td> $0 . 9 0 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 5 4 \pm 0 . 0 1 5$ </td><td> $0 . 7 4 4 \pm 0 . 0 1 5$ </td><td> $0 . 7 7 5 \pm 0 . 0 0 9$ </td></tr><tr><td>MCD</td><td> $0 . 8 8 5 \pm 0 . 0 0 1$ </td><td> $0 . 7 1 3 \pm 0 . 0 0 2$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 6 8 2 \pm 0 . 0 0 4 }$ </td><td> $0 . 7 8 5 \pm 0 . 0 1 0$ </td><td> $0 . 7 8 1 \pm 0 . 0 0 6$ </td></tr><tr><td>SNGP</td><td> $0 . 9 0 3 \pm 0 . 0 0 2$ </td><td> $0 . 7 2 7 \pm 0 . 0 0 7$ </td><td> $0 . 9 1 3 \pm 0 . 0 0 4$ </td><td> $0 . 6 8 0 \pm 0 . 0 0 9$ </td><td> $0 . 7 7 4 \pm 0 . 0 1 5$ </td><td> $0 . 7 8 9 \pm 0 . 0 0 4$ </td></tr><tr><td>SGPA</td><td> $\overline { { 0 . 8 8 8 \pm 0 . 0 0 2 } }$ </td><td> $0 . 6 6 5 \pm 0 . 0 1 7$ </td><td> $0 . 8 5 1 \pm 0 . 0 0 7$ </td><td> $0 . 6 8 0 \pm 0 . 0 0 2$ </td><td> $0 . 7 4 9 \pm 0 . 0 3 6$ </td><td> $0 . 7 8 5 \pm 0 . 0 0 4$ </td></tr><tr><td>RFF-CGP (ours)</td><td> $0 . 9 0 2 \pm 0 . 0 0 2$ </td><td> $0 . 7 3 7 \pm 0 . 0 0 3$ </td><td> $0 . 9 1 0 \pm 0 . 0 0 6$ </td><td> $0 . 6 5 1 \pm 0 . 0 2 0$ </td><td> $\mathbf { 0 . 8 1 5 \pm 0 . 0 3 6 }$ </td><td> $\mathbf { 0 . 7 9 0 \overset { \cdot } { = } 0 . 0 0 9 }$ </td></tr><tr><td>RFF-GPA (ours)</td><td> $\mathbf { 0 . 9 1 5 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 7 5 9 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 9 1 5 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 6 8 2 \pm 0 . 0 1 0 }$ </td><td> $0 . 8 0 0 \pm 0 . 0 1 5$ </td><td> $\underline { { 0 . 7 8 9 \pm 0 . 0 1 0 } }$ </td></tr><tr><td>METRIC III</td><td colspan="6">NLL↓</td></tr><tr><td>MLE</td><td> $0 . 3 1 7 \pm 0 . 0 0 5$ </td><td> $0 . 8 3 7 \pm 0 . 0 2 2$ </td><td> $0 . 3 1 6 \pm 0 . 0 2 0$ </td><td> $1 . 4 8 3 \pm 0 . 0 4 5$ </td><td> $0 . 5 3 8 \pm 0 . 0 2 3$ </td><td> $0 . 5 9 9 \pm 0 . 0 4 2$ </td></tr><tr><td>MLE+Temp</td><td> $0 . 3 0 4 \pm 0 . 0 0 8$ </td><td> $0 . 8 1 4 \pm 0 . 0 2 4$ </td><td> $0 . 3 0 5 \pm 0 . 0 1 9$ </td><td> ${ \bf 1 . 2 2 4 \pm 0 . 0 4 1 }$ </td><td> $0 . 5 2 7 \pm 0 . 0 3 0$ </td><td> $0 . 5 5 5 \pm 0 . 0 1 7$ </td></tr><tr><td>MCD</td><td> $0 . 3 1 7 \pm 0 . 0 0 4$ </td><td> $0 . 8 1 9 \pm 0 . 0 1 2$ </td><td> $0 . 3 2 3 \pm 0 . 0 0 6$ </td><td> $1 . 3 8 8 \pm 0 . 0 2 5$ </td><td> $0 . 5 5 8 \pm 0 . 0 1 8$ </td><td> $\underline { { 0 . 5 3 5 \pm 0 . 0 1 8 } }$ </td></tr><tr><td>SNGP</td><td> $0 . 2 8 2 \pm 0 . 0 0 8$ </td><td> $0 . 8 0 7 \pm 0 . 0 1 4$ </td><td> $0 . 2 9 9 \pm 0 . 0 0 7$ </td><td> $1 . 3 8 1 \pm 0 . 0 5 2$ </td><td> $0 . 5 1 1 \pm 0 . 0 2 7$ </td><td> $\overline { { 0 . 6 4 1 \pm 0 . 0 4 5 } }$ </td></tr><tr><td>SGPA</td><td> $0 . 3 1 3 \pm 0 . 0 0 3$ </td><td> $0 . 9 4 8 \pm 0 . 0 4 3$ </td><td> $0 . 4 8 3 \pm 0 . 0 2 4$ </td><td> $\underline { { 1 . 2 9 1 \pm 0 . 0 1 0 } }$ </td><td> $0 . 5 4 8 \pm 0 . 0 4 6$ </td><td> $0 . 5 9 2 \pm 0 . 0 5 7$ </td></tr><tr><td>RFF-CGP (ours)</td><td> $0 . 2 8 4 \pm 0 . 0 0 3$ </td><td> $0 . 8 0 2 \pm 0 . 0 0 5$ </td><td> $0 . 3 1 3 \pm 0 . 0 0 6$ </td><td> $\overline { { 1 . 6 4 3 \pm 0 . 1 7 3 } }$ </td><td> $\mathbf { 0 . 4 1 5 \pm 0 . 0 3 7 }$ </td><td> $\mathbf { 0 . 5 3 2 \pm 0 . 0 4 3 }$ </td></tr><tr><td>RFF-GPA (ours)</td><td> $\mathbf { 0 . 2 5 2 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 7 1 3 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 2 9 2 \pm 0 . 0 0 1 }$ </td><td> $1 . 3 1 6 \pm 0 . 0 5 0$ </td><td> $\underline { { 0 . 5 0 1 } } \pm 0 . 0 0 7$ </td><td> $0 . 5 4 6 \pm 0 . 0 6 2$ </td></tr><tr><td>METRIC 三</td><td colspan="6">ECE↓</td></tr><tr><td>MLE</td><td> $0 . 0 5 1 \pm 0 . 0 0 5$ </td><td> $0 . 0 6 5 \pm 0 . 0 0 5$ </td><td> $0 . 0 3 0 \pm 0 . 0 0 3$ </td><td> $0 . 1 5 8 \pm 0 . 0 1 5$ </td><td> $0 . 1 2 0 \pm 0 . 0 0 9$ </td><td></td></tr><tr><td>MLE+Temp</td><td> $0 . 0 3 1 \pm 0 . 0 0 3$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 2 7 \pm 0 . 0 0 1$ </td><td> $0 . 1 5 2 \pm 0 . 0 0 6$ </td><td> $0 . 1 1 4 \pm 0 . 0 0 1$ </td><td> $0 . 1 3 5 \pm 0 . 0 1 7$   $0 . 1 2 7 \pm 0 . 0 0 9$ </td></tr><tr><td>MCD</td><td> $0 . 0 3 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 6 4 \pm 0 . 0 0 4$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 2$ </td><td> $0 . 1 3 6 \pm 0 . 0 0 6$ </td><td> $\overline { { 0 . 1 5 6 \pm 0 . 0 1 0 } }$ </td><td> $0 . 1 0 7 \pm 0 . 0 0 6$ </td></tr><tr><td>SNGP</td><td> $\underline { { 0 . 0 2 6 \pm 0 . 0 0 6 } }$ </td><td> $0 . 0 6 3 \pm 0 . 0 0 6$ </td><td> $0 . 0 1 7 \pm 0 . 0 0 3$ </td><td> $0 . 1 5 6 \pm 0 . 0 0 7$ </td><td> $0 . 1 2 2 \pm 0 . 0 3 5$ </td><td> $0 . 1 3 2 \pm 0 . 0 0 9$ </td></tr><tr><td>SGPA</td><td> $\mathbf { \overline { { 0 . 0 2 1 } } } \pm \mathbf { 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 0 2 5 \pm 0 . 0 1 3 }$ </td><td> $\underline { { 0 . 0 1 6 \pm 0 . 0 0 5 } }$ </td><td> $\underline { { 0 . 1 3 4 \pm 0 . 0 0 3 } }$ </td><td> $0 . 1 3 6 \pm 0 . 0 7 3$ </td><td> $0 . 1 2 7 \pm 0 . 0 1 6$ </td></tr><tr><td>RFF-CGP (ours)</td><td> $0 . 0 2 9 \pm 0 . 0 0 4$ </td><td> $0 . 0 6 6 \pm 0 . 0 1 3$ </td><td> $0 . 0 2 3 \pm 0 . 0 0 8$ </td><td> $0 . 1 3 7 \pm 0 . 0 3 4$ </td><td> ${ \bf 0 . 1 1 1 \pm 0 . 0 2 8 }$ </td><td> $\mathbf { 0 . 1 0 1 \pm 0 . 0 1 4 }$ </td></tr><tr><td>RFF-GPA (ours)</td><td> $\mathbf { 0 . 0 2 1 } \pm \mathbf { 0 . 0 0 3 }$ </td><td> $\underline { { 0 . 0 4 6 \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 0 . 0 1 3 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 0 . 1 3 0 \pm 0 . 0 1 0 }$ </td><td> $0 . 1 2 3 \pm 0 . 0 1 7$ </td><td> $0 . 1 0 5 \pm 0 . 0 1 1$ </td></tr></table>

Table 1: In-distribution accuracy and calibration on three image and three text classification benchmarks. Each column reports one dataset; row blocks report ACC ↑, NLL ↓, and ECE ↓. Mean and standard deviation are computed over five runs. The bes result per column is bolded and the second-best is underlined.

backbone architecture or tuning. We compare our randomfeature GP attention (RFF-GPA and RFF-CGP) against a deterministic maximum-likelihood head (Vaswani et al. 2017, MLE), its post-hoc temperature-scaled variant (Guo et al. 2017, MLE+Temp), Monte Carlo dropout (Gal and Ghahramani 2016, MCD), the spectral-normalized neural GP (Liu et al. 2020, SNGP), and sparse GP attention (Chen and Li 2023, SGPA). Accuracy is measured by top-1 accuracy (ACC; higher is better), and calibration by negative log–likelihood (NLL) and expected calibration error (ECE), both lower being better.

Accuracy. RFF-GPA is strongest on the image benchmarks, attaining the best accuracy on Fashion-MNIST (0.915), CIFAR-10 (0.759), and SVHN (0.915), while also matching the best mean accuracy on 20 Newsgroups. On the text benchmarks, the two random-feature variants are complementary: RFF-CGP achieves the best accuracy on Hyperpartisan (0.815) and SST-2 (0.790), while RFF-GPA remains second-best on both. Notably, RFF-GPA substantially improves over SGPA on the image datasets (e.g. +9.4 points on CIFAR-10 and +6.4 on SVHN), indicating that

Calibration. RFF-GPA delivers strong calibration on the image benchmarks, achieving the best NLL on Fashion-MNIST, CIFAR-10, and SVHN. For ECE, it is best on SVHN and 20 Newsgroups, ties SGPA for the best result on Fashion-MNIST, and is second-best on CIFAR-10 and SST-2. RFF-CGP is particularly strong on the text benchmarks, attaining the best NLL and ECE on Hyperpartisan and SST-2. This modality-dependent pattern suggests that the independenthead RFF-GPA formulation is especially efective for local visual patch interactions, whereas the correlated query–key structure in RFF-CGP can better capture asymmetric semantic interactions in text. Overall, the two random-feature GP variants provide competitive or leading accuracy while yielding well-calibrated predictive uncertainty in a single, lineartime forward pass, without the ensembling or inducing-point machinery required by competing approaches.

![](images/716562ede5b78d0ad4c8d31f60a8290a25257d076e5e2e52de118a2fe6a5ac57.jpg)

![](images/4c592b2ce03e4071e68c49ba2ecbdf92558dd810dfd1a4823303420ea26d4f7c.jpg)  
Figure 2: Training Time (Left): Training time per step as a function of sequence length L across all methods (log–log axes; shaded 95% CI). Throughput (Right): Throughput (tokens/sec) as a function of L. Curves for the inducing-point baselines (SGPA, SCGP) are continued past their largest feasible length along their fitted scaling trend.

## 4<sub>.</sub>2 Efi<sub>c</sub>i<sub>e</sub>n<sub>cy</sub> <sub>a</sub>nd S<sub>ca</sub>l<sub>a</sub>bilit<sub>y</sub>

We now show that RFF-GPA delivers favorable runtime– calibration behavior compared with inducing-point GP attention. Following the linear-time analysis in Sec. 3.1, we benchmark a single attention block with a lightweight readout on synthetic inputs while varying the sequence length $L . ^ { 6 }$ Figure 2 reports training time per optimization step and throughput (tokens/sec) as functions of L, each averaged over five runs with 95% confidence intervals.

RFF-GPA and RFF-CGP scale linearly with L, thanks to the RFF approximation that only inverts an $M \times M$ matrix. In contrast, the MLE and SNGP softmax heads scale quadratically with sequence length, while inducing-point variants such as SGPA and SCGP exhibit the expected super-linear scaling and incur steep computational costs once $\mathit { \bar { L } } \gtrsim 5 0 0 .$ These trends are consistent with the complexity analysis in Sec. 3.1.

The key reason is that the random-feature dimension M is fixed independently of the sequence length, so increasing $L$ only adds feature evaluations and matrix multiplications involving $\Phi _ { Q }$ and $\Phi _ { K }$ . The expensive inverse remains in the feature space rather than the token space. By contrast, inducing-point GP attention still materializes sequencedependent kernel matrices and repeatedly applies kernel operations whose memory footprint grows with $L ,$ which explains the sharp throughput degradation of SGPA and SCGP at longer sequences. Thus, the empirical scaling curves reflect the intended computational design rather than an implementation artifact. This also makes the cost predictable across batch sizes and sequence lengths, since the dominant operations depend on the chosen feature budget rather than on a growing token–token covariance matrix that must be recomputed for each input.

## 4<sub>.</sub>3 O<sub>u</sub>t<sub>-o</sub>f<sub>-</sub>Di<sub>s</sub>trib<sub>u</sub>ti<sub>o</sub>n R<sub>o</sub>b<sub>us</sub>tn<sub>ess</sub>

We finally evaluate robustness under distribution shift using the CIFAR-10-C benchmark (Hendrycks and Dietterich 2019), which applies 19 common corruption types (e.g., noise, blur, weather efects) at five severity levels to the CIFAR-10 test set. We take the models trained on clean CIFAR-10 in Sec. 4.1 and evaluate them on the corrupted test sets without any re-training orfine-tuning, so the benchmark measures how gracefully each method degrades once deployed beyond its training distribution.<sup>7</sup> Table 2 reports the clean CIFAR-10 result at s= 0 and, for $s \geq 1$ , each metric averaged over the 19 corruptions at every severity level as mean ± 95% confidence interval over five runs.

Accuracy. RFF-GPA is the most accurate method at every severity level, from the clean set (0.759) down to the most severe corruptions (0.514 at s=5), consistently ahead of the strong SNGP baseline and improving on the inducing-point SGPA by a large margin (e.g. +7.3 points at s=5). This shows that the accuracy advantage of the random-feature GP formulation is preserved, not merely retained, under distribution shift.

Calibration. The central finding is that RFF-GPA is the best-calibrated method across the corruption range in terms of NLL and ECE: it attains the lowest NLL at every severity and the lowest ECE at every non-clean severity $( s \ge 1 )$ , while remaining second only to SGPA on the clean set. Crucially, SGPA reaches competitive ECE only with a substantial loss in accuracy, whereas RFF-GPA combines the best accuracy with the strongest aggregate NLL/ECE. This distinction matters because calibration under shift is meaningful only when the model still preserves useful predictive power: a method can appear well calibrated by becoming uniformly uncertain, but that behavior is less useful if accuracy collapses. RFF-GPA avoids this failure mode by reducing overconfidence while retaining the strongest predictions across severity levels. The deterministic MLE head degrades sharply in NLL as severity grows (0.84 → 1.72), reflecting increasingly overconfident predictions under corruption. Overall, the CIFAR-10-C results confirm that the random-feature GP framework yields reliable uncertainty under distribution shift while preserving predictive performance.

<table><tr><td>METHOD 二</td><td> $s { = } 0 \ ( \mathrm { c l e a n } )$ </td><td> $s { = } 1$ </td><td> $s { = } 2$ </td><td> $s { = } 3$ </td><td> $s { = } 4$ </td><td> $s { = } 5$ </td></tr><tr><td>METRIC 三</td><td colspan="6">ACC↑</td></tr><tr><td>MLE</td><td> $0 . 7 1 5 \pm 0 . 0 0 8$ </td><td> $0 . 6 8 1 \pm 0 . 0 1 0$ </td><td> $0 . 6 3 7 \pm 0 . 0 0 9$ </td><td> $0 . 6 0 5 \pm 0 . 0 0 9$ </td><td> $0 . 5 6 4 \pm 0 . 0 1 0$ </td><td> $0 . 5 0 3 \pm 0 . 0 0 8$ </td></tr><tr><td>SNGP</td><td> $0 . 7 2 7 \pm 0 . 0 0 7$ </td><td> $0 . 6 9 3 \pm 0 . 0 0 6$ </td><td> $0 . 6 4 7 \pm 0 . 0 0 1$ </td><td> $0 . 6 1 3 \pm 0 . 0 0 1$ </td><td> $0 . 5 7 2 \pm 0 . 0 0 2$ </td><td> $0 . 5 0 8 \pm 0 . 0 0 3$ </td></tr><tr><td>SGPA</td><td> $0 . 6 6 5 \pm 0 . 0 2 3$ </td><td> $0 . 6 1 3 \pm 0 . 0 2 3$ </td><td> $0 . 5 5 6 \pm 0 . 0 2 4$ </td><td> $0 . 5 3 0 \pm 0 . 0 1 8$ </td><td> $\overline { { 0 . 4 9 5 \pm 0 . 0 1 4 } }$ </td><td> $\overline { { 0 . 4 4 1 \pm 0 . 0 1 2 } }$ </td></tr><tr><td>RFF-CGP (ours)</td><td> $0 . 7 3 7 \pm 0 . 0 0 3$ </td><td> $0 . 6 6 4 \pm 0 . 0 0 3$ </td><td> $0 . 6 1 6 \pm 0 . 0 0 4$ </td><td> $0 . 5 8 8 \pm 0 . 0 0 4$ </td><td> $0 . 5 4 8 \pm 0 . 0 0 1$ </td><td> $0 . 4 9 0 \pm 0 . 0 0 2$ </td></tr><tr><td>RFF-GPA (ours)</td><td> $\mathbf { 0 . 7 5 9 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 7 0 2 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 6 5 3 \pm 0 . 0 0 2 }$ </td><td> ${ \bf 0 . 6 2 0 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 5 7 6 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 5 1 4 \ : \pm 0 . 0 0 2 }$ </td></tr><tr><td>METRIC 三</td><td colspan="6">NLL↓</td></tr><tr><td>MLE</td><td> $0 . 8 3 7 \pm 0 . 0 2 2$ </td><td> $0 . 9 4 6 \pm 0 . 0 2 3$ </td><td> $1 . 1 0 2 \pm 0 . 0 1 5$ </td><td> $1 . 2 3 8 \pm 0 . 0 1 0$ </td><td> $1 . 4 2 1 \pm 0 . 0 1 9$ </td><td> $1 . 7 1 8 \pm 0 . 0 0 9$ </td></tr><tr><td>SNGP</td><td> $0 . 8 0 7 \pm 0 . 0 1 4$ </td><td> $\underline { { 0 . 9 2 1 \pm 0 . 0 1 0 } }$ </td><td> $\underline { { 1 . 0 8 8 \pm 0 . 0 2 3 } }$ </td><td> $\underline { { 1 . 2 3 4 \pm 0 . 0 3 1 } }$ </td><td> $1 . 4 2 5 \pm 0 . 0 4 4$ </td><td> $1 . 7 4 7 \pm 0 . 0 6 0$ </td></tr><tr><td>SGPA</td><td> $0 . 9 4 8 \pm 0 . 0 6 0$ </td><td> $\overline { { 1 . 1 1 1 \pm 0 . 0 6 2 } }$ </td><td> $1 . 2 9 8 \pm 0 . 0 6 9$ </td><td> $1 . 3 9 6 \pm 0 . 0 4 3$ </td><td> $1 . 5 3 9 \pm 0 . 0 2 0$ </td><td> $1 . 7 7 0 \pm 0 . 0 1 7$ </td></tr><tr><td>RFF-CGP (ours)</td><td> $\underline { { 0 . 8 0 2 \pm 0 . 0 0 5 } }$ </td><td> $0 . 9 9 0 \pm 0 . 0 2 3$ </td><td> $1 . 1 5 5 \pm 0 . 0 3 1$ </td><td> $1 . 2 7 1 \pm 0 . 0 3 1$ </td><td> $1 . 4 4 5 \pm 0 . 0 3 3$ </td><td> $1 . 7 1 4 \pm 0 . 0 4 7$ </td></tr><tr><td>RFF-GPA (ours)</td><td> $\mathbf { 0 . 7 1 3 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 8 6 5 \pm 0 . 0 1 2 }$ </td><td> $\mathbf { 1 . 0 2 5 \ : \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 1 . 1 5 1 \pm 0 . 0 2 3 }$ </td><td> ${ \bf 1 . 3 2 6 \pm 0 . 0 2 8 }$ </td><td> $\mathbf { \overline { { 1 . 6 1 3 \pm 0 . 0 2 7 } } }$ </td></tr><tr><td>METRIC III</td><td colspan="6"></td></tr><tr><td>MLE</td><td> $0 . 0 6 5 \pm 0 . 0 0 5$ </td><td> $0 . 0 7 9 \pm 0 . 0 0 8$ </td><td>ECE↓  $0 . 1 0 6 \pm 0 . 0 0 9$ </td><td> $0 . 1 3 1 \pm 0 . 0 0 8$ </td><td> $0 . 1 6 4 \pm 0 . 0 0 7$ </td><td></td></tr><tr><td>SNGP</td><td> $0 . 0 6 3 \pm 0 . 0 0 6$ </td><td> $0 . 0 8 0 \pm 0 . 0 1 1$ </td><td> $0 . 1 0 7 \pm 0 . 0 1 6$ </td><td> $0 . 1 3 2 \pm 0 . 0 1 4$ </td><td> $0 . 1 6 3 \pm 0 . 0 1 5$ </td><td> $0 . 2 0 9 \pm 0 . 0 1 2$   $0 . 2 0 7 \pm 0 . 0 1 3$ </td></tr><tr><td>SGPA</td><td> $\mathbf { 0 . 0 2 5 \pm 0 . 0 1 3 }$ </td><td> $\underline { { 0 . 0 5 5 \pm 0 . 0 2 0 } }$ </td><td> $\underline { { 0 . 0 9 0 \pm 0 . 0 2 3 } }$ </td><td> $\underline { { 0 . 1 0 4 \pm 0 . 0 1 6 } }$ </td><td> $\underline { { 0 . 1 3 0 \pm 0 . 0 1 1 } }$ </td><td> $\underline { { 0 . 1 6 1 \pm 0 . 0 0 9 } }$ </td></tr><tr><td>RFF-CGP (ours)</td><td> $0 . 0 6 6 \pm 0 . 0 1 3$ </td><td> $\overline { { 0 . 0 8 0 \pm 0 . 0 2 0 } }$ </td><td> $\overline { { 0 . 1 0 8 \pm 0 . 0 1 9 } }$ </td><td> $\overline { { 0 . 1 2 8 \pm 0 . 0 1 6 } }$ </td><td> $\overline { { 0 . 1 5 7 \pm 0 . 0 1 4 } }$ </td><td> $\overline { { 0 . 1 9 7 \pm 0 . 0 1 4 } }$ </td></tr><tr><td>RFF-GPA (ours)</td><td> $\underline { { 0 . 0 4 6 \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 0 . 0 4 7 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 0 7 4 \ : \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 0 9 4 } \pm \mathbf { 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 1 2 3 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 1 6 0 \pm 0 . 0 0 3 }$ </td></tr></table>

Table 2: Out-of-distribution robustness on CIFAR-10-C. Each metric is averaged over the 19 corruption types at severity $s \in \{ 1 , \ldots , 5 \} ; s = 0$ is the clean CIFAR-10 test set. Row blocks report ACC ↑, NLL ↓, and ECE ↓. Values are mean ± 95% CI over five runs; all models are evaluated without re-training. The best result per column is bolded and the second-best is underlined.

Spatial uncertainty. We next examine whether the tokenwise GP posterior variance, diag(Σ), responds locally to input ambiguity. Figure 3 shows final-layer uncertainty before and after adding Gaussian noise to only 12 of the 256 CIFAR-10 patch tokens. Variance is averaged over four heads and 30 paired Monte Carlo passes, with the same random-feature draw reused for each clean/corrupted pair. For both an object-side perturbation and a matched background perturbation, the prediction remains bird, while positive $\Delta U = U _ { \mathrm { c o r r u p t } } - U _ { \mathrm { c l e a n } }$ is concentrated around the perturbed tokens. This provides a spatial complement to the image-level calibration results above.

![](images/884cf093ac8b9b953d1230d22429af914eea35a2dcf8d8b0235cd08a7c1d59f1.jpg)  
Figure 3: Localized token-wise posterior uncertainty. Object perturbation (top) and matched background perturbation (bottom). Red grids mark the 12 perturbed tokens; clean and corrupted uncertainty maps use a shared scale.

## 5 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

In this paper, we introduced a Gaussian process Transformer where we model the attention kernel using a random Fourier feature approximation. By taking advantage of the low-rank factorization of the attention module, we can obtain a fully Bayesian model that is scalable with respect to the sequence length. We have demonstrated the efectiveness of the RFF-GPA on image and text data sets, especially with respect to uncertainty-aware metrics, compared to the vanilla Transformer and other competing GP-based methods. In future work, we hope to further enrich the approximate kernel structure by modeling non-stationarity across each head, or by assuming a deep generative model as a prior on the random frequencies.

Liu, J.; Lin, Z.; Padhy, S.; Tran, D.; Bedrax Weiss, T.; and Lakshminarayanan, B. 2020. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. Advances in Neural Information Processing Systems, 33: 7498–7512.

## R<sub>e</sub>f<sub>erences</sub>

Bochner, S. 1959. Lectures on Fourier Integrals, volume 42. Princeton University Press.

Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33: 1877– 1901.

Bui, L. M.; Huu, T. T.; Dinh, D.; Nguyen, T. M.; and Hoang, T. N. 2024. Revisiting Kernel Attention with Correlated Gaussian Process Representation. In Uncertainty in Artificial Intelligence.

Chen, W.; and Li, Y. 2023. Calibrating Transformers via sparse Gaussian processes. In International Conference on Learning Representations.

Choromanski, K.; Likhosherstov, V.; Dohan, D.; Song, X.; Gane, A.; Sarlos, T.; Hawkins, P.; Davis, J.; Mohiuddin, A.; Kaiser, L.; et al. 2021. Rethinking attention with performers. In International Conference on Learning Representations.

Damianou, A.; and Lawrence, N. D. 2013. Deep Gaussian processes. In Artificial Intelligence and Statistics, 207–215. PMLR.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations.

Fan, X.; Zhang, S.; Chen, B.; and Zhou, M. 2020. Bayesian attention modules. Advances in Neural Information Processing Systems, 33: 16362–16376.

Gal, Y.; and Ghahramani, Z. 2016. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning, 1050–1059. PMLR.

Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On calibration of modern neural networks. In International Conference on Machine Learning, 1321–1330. PMLR.

Hager, W. W. 1989. Updating the inverse of a matrix. SIAM Review, 31(2): 221–239.

Hendrycks, D.; and Dietterich, T. 2019. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In International Conference on Learning Representations.

Kiesel, J.; Mestre, M.; Shukla, R.; Vincent, E.; Adineh, P.; Corney, D.; Stein, B.; and Potthast, M. 2019. SemEval-2019 Task 4: Hyperpartisan News Detection. In Proceedings of the 13th International Workshop on Semantic Evaluation, 829–839.

Kingma, D. P.; and Ba, J. 2015. Adam: A method for stochastic optimization. In International Conference on Learning Representations.

Krizhevsky, A.; and Hinton, G. 2009. Learning multiple layers of features from tiny images. Technical report, University of Toronto.

Lang, K. 1995. NewsWeeder: Learning to filter netnews. In Machine Learning Proceedings 1995, 331–339. Elsevier.

Li, Y.; Lin, Z.; Yin, F.; and Zhang, M. M. 2024. Preventing Model Collapse in Gaussian Process Latent Variable Models. In International Conference on Machine Learning, 28278– 28308. PMLR.

Minderer, M.; Djolonga, J.; Romijnders, R.; Hubis, F.; Zhai, X.; Houlsby, N.; Tran, D.; and Lucic, M. 2021. Revisiting the calibration of modern neural networks. Advances in Neural Information Processing Systems, 34: 15682–15694.

Neal, R. M. 1995. Bayesian Learning for Neural Networks. Ph.D. thesis, University of Toronto.

Netzer, Y.; Wang, T.; Coates, A.; Bissacco, A.; Wu, B.; and Ng, A. Y. 2011. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning.

Papamarkou, T.; Skoularidou, M.; Palla, K.; Aitchison, L.; Arbel, J.; Dunson, D.; Filippone, M.; Fortuin, V.; Hennig, P.; Hernández-Lobato, J. M.; et al. 2024. Position: Bayesian deep learning is needed in the age of large-scale AI. In International Conference on Machine Learning, 39556–39586.

Peng, H.; Pappas, N.; Yogatama, D.; Schwartz, R.; Smith, N.; and Kong, L. 2021. Random Feature Attention. In International Conference on Learning Representations.

Rahimi, A.; and Recht, B. 2007. Random features for largescale kernel machines. Advances in Neural Information Processing Systems, 20.

Socher, R.; Perelygin, A.; Wu, J.; Chuang, J.; Manning, C. D.; Ng, A. Y.; and Potts, C. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, 1631–1642.

Tran, D.; Dusenberry, M.; van der Wilk, M.; and Hafner, D. 2019. Bayesian layers: A module for neural network uncertainty. Advances in Neural Information Processing Systems, 32.

Tsai, Y.-H. H.; Bai, S.; Yamada, M.; Morency, L.-P.; and Salakhutdinov, R. 2019. Transformer Dissection: A Unified Understanding for Transformer’s Attention via the Lens of Kernel. In Proceedings of the Conference on Empirical Methods in Natural Language Processing.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in Neural Information Processing Systems, 30.

Williams, C. K.; and Rasmussen, C. E. 2006. Gaussian Processes for Machine Learning, volume 2. MIT Press.

Xiao, H.; Rasul, K.; and Vollgraf, R. 2017. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.

Xue, B.; Yu, J.; Xu, J.; Liu, S.; Hu, S.; Ye, Z.; Geng, M.; Liu, X.; and Meng, H. 2021. Bayesian transformer language models for speech recognition. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing, 7378–7382. IEEE.

Yang, Z.; Li, Y.; Lin, Z.; Zhang, M. M.; and Olmos, P. M. 2025. Multi-View Oriented GPLVM: Expressiveness and Eficiency. In Advances in Neural Information Processing Systems.

Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Dou, Z.; and Wen, J.-R. 2023. A Survey of Large Language Models. arXiv preprint arXiv:2303.18223.

## A<sub>ppendices</sub>

## A<sub>ppendix</sub> C<sub>ontents</sub>

Appendix A Clarification on the Empirical Bayes Treatment of Spectral Points   
Appendix B Baseline Methods   
Appendix C Datasets   
Appendix D Model Configuration and Experimental Design   
Appendix E Eficiency and Scalability Benchmark   
Appendix F Out-of-Distribution Robustness Protocol

## A Cl<sub>ar</sub>ifi<sub>ca</sub>ti<sub>on on</sub> th<sub>e</sub> E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l B<sub>ayes</sub> T<sub>rea</sub>t<sub>men</sub>t <sub>o</sub>f S<sub>pec</sub>t<sub>ra</sub>l P<sub>o</sub>i<sub>n</sub>t<sub>s</sub>

In principle, one could introduce a more flexible variational posterior $q _ { \phi } ( \Omega )$ for the spectral points, jointly with the latent Gaussian attention distribution $p ( \mathbf { Y } \mid \mathbf { X } , \boldsymbol { \Omega } )$ . The resulting ELBO would take the general form

$$
\mathcal { L } _ { \mathrm { E L B O } } = \mathbb { E } _ { q _ { \phi } ( \Omega ) } \mathbb { E } _ { p ( \mathbf { Y } \mid \mathbf { X } , \Omega ) } \left[ \log p ( \pmb { \mathscr { Y } } \mid \mathbf { Y } ) \right] - \mathrm { K L } \big ( q _ { \phi } ( \pmb { \Omega } ) \| p ( \pmb { \Omega } ) \big ) .\tag{31}
$$

However, in our setting the spectral points Ω only influence the model through the GP prior structure in the attention kernel. We empirically observed that optimizing the ELBO with a highly expressive $q _ { \phi } ( \Omega )$ tends to drive the KL term toward zero, yielding $q _ { \phi } ( \Omega ) \approx p ( \Omega )$ . In other words, the variational posterior typically collapses back to the parametric prior family, while introducing additional variance and instability during training.

Motivated by these observations, we adopt the simplified assumption $q ( \Omega ) = p ( \Omega )$ and directly optimize the prior parameters $( \mathrm { e . g . }$ ., mean and diagonal covariance). This provides a lightweight and stable empirical Bayes treatment of the spectral density, while retaining the probabilistic interpretation of the GP-based attention module.

## B B<sub>ase</sub>li<sub>ne</sub> M<sub>e</sub>th<sub>o</sub>d<sub>s</sub>

This appendix summarizes the baseline methods used in our experiments, together with their modeling assumptions and sources of uncertainty. All baselines are implemented using the same backbone architecture and training protocol unless otherwise stated.

SGPA (Sparse Gaussian Process Attention). SGPA performs Bayesian inference directly in the output space of multi-head attention by replacing the scaled dot-product similarity with a valid kernel, and placing a sparse variational Gaussian process (SVGP) posterior over the attention outputs. Concretely, SGPA models the attention output as a GP mapping with a kernel matrix $K _ { Q K }$ computed from queries and keys, and uses inducing variables to obtain a scalable variational posterior. This yields calibrated predictive uncertainty while keeping the computation tractable via sparse GP machinery (Chen and Li 2023).

SCGP (Sparse Correlated Gaussian Process). SCGP is a scalable correlated-GP formulation for kernel attention, derived from correlated Gaussian-process Transformer models. It introduces a sparse approximation (via inducing inputs) to mitigate the cubic cost of correlated GP inference, while retaining correlation structure in the GP treatment of kernel attention. In practice, SCGP computes kernel-attention statistics through a correlated GP posterior with a sparse scheme, improving scalability compared to the full correlated GP formulation (Bui et al. 2024).

SNGP (Spectral-normalized Neural Gaussian Process). SNGP is a neural Gaussian process model that combines a deterministic deep neural network with a GP-inspired output layer. Specifically, the penultimate layer features are treated as inputs to a GP classifier, and spectral normalization is applied to the network weights to ensure distance awareness and improved uncertainty estimates. Unlike fully Bayesian neural networks, SNGP performs deterministic training while approximating predictive uncertainty through a GP-style output covariance, making it a strong and scalable baseline for uncertainty estimation in deep models (Liu et al. 2020).

MCD (Monte Carlo Dropout). Monte Carlo dropout (MCD) is a widely used Bayesian approximation for deep neural networks, where dropout is interpreted as approximate variational inference over network weights. At test time, multiple stochastic forward passes are performed with dropout enabled, and predictive uncertainty is estimated from the empirical mean and variance of the outputs:

$$
p ( \boldsymbol { y } \mid \boldsymbol { x } ) \approx \frac { 1 } { T } \sum _ { t = 1 } ^ { T } p ( \boldsymbol { y } \mid \boldsymbol { x } , \mathbf { W } _ { t } ) ,\tag{32}
$$

where $\mathbf { W } _ { t }$ denotes a random realization of the network weights induced by dropout. MCD captures parameter-level (epistemic) uncertainty without modifying the attention mechanism itself, and serves as a standard baseline for uncertainty-aware Transformers (Gal and Ghahramani 2016).

<table><tr><td>Name</td><td>Description</td><td>Size</td><td>Citation/source</td></tr><tr><td>Fashion-MNIST</td><td>Grayscale 28×28 clothing-image classification with 60k train / 10k test 10 classes; patch size 4×4 gives sequence length 49.</td><td></td><td>Xiao, Rasul, and Vollgraf (2017)</td></tr><tr><td>CIFAR-10</td><td>Color 32×32 natural-image classification with 10 50k train / 10k test classes; patch size 2×2 gives sequence length 256.</td><td></td><td>Krizhevsky and Hinton (2009)</td></tr><tr><td>SVHN</td><td>Real-world 32×32 street-view house-number digit ~73k train / 26k test classification with 10 classes; sequence length 256.</td><td></td><td>Netzer et al. (2011)</td></tr><tr><td>CIFAR-10-C</td><td>Distribution-shift benchmark built from CIFAR-10 19 corruptions × 5 severities × Hendrycks and Di- with common image corruptions at multiple severi- 10k images ties.</td><td></td><td>etterich (2019)</td></tr><tr><td>20 Newsgroups</td><td>Topic classification over newsgroup posts with 20 ~18k documents classes; text is truncated/padded to 512 tokens.</td><td></td><td>Lang (1995)</td></tr><tr><td>Hyperpartisan</td><td>Binary news-bias classification using the by-article Small by-article split split; text is truncated/padded to 512 tokens.</td><td></td><td>Kiesel et al. (2019)</td></tr><tr><td>SST-2</td><td>Binary sentiment classification from GLUE; we eval- ~67k train; dev evaluation uate on the development set because test labels are</td><td></td><td>Socher et al. (2013)</td></tr></table>

Table 3: Datasets used in the in-distribution and out-of-distribution experiments. Images are patchified inside the Vision-Transformer backbone and standardized per channel; text uses a shared WordPiece tokenizer.

MLE (Deterministic Maximum-Likelihood Head). The MLE baseline corresponds to a standard deterministic Transformer trained by maximizing the likelihood (or equivalently minimizing cross-entropy loss) with a softmax output layer. Predictions are obtained from a single forward pass, and no explicit uncertainty modeling is performed beyond the softmax probabilities. This baseline serves as a reference point for evaluating the benefits of uncertainty-aware modeling in terms of calibration and robustness.

MLE+Temp (Temperature Scaling). Temperature scaling is a simple, single-parameter post-hoc calibration method applied to the trained MLE model (Guo et al. 2017). A scalar temperature T > 0 is fit on the validation set by minimizing the negative log-likelihood of the temperature-scaled logits z/T, and predictions at test time use the rescaled softmax softmax(z/T). Since it only rescales confidences without changing the predicted class, MLE+Temp leaves accuracy unchanged while typically improving calibration, and provides a strong, essentially free reference for the calibration metrics. Because it is purely post-hoc, temperature scaling is orthogonal to and can be composed with any of the methods considered here, including ours.

## C Datasets

We evaluate on seven publicly available classification benchmarks covering both image and text modalities, summarized in Table 3. Unless noted otherwise, we hold out a fixed validation split shared across all methods and seeds for model selection and temperature fitting.

## D M<sub>o</sub>d<sub>e</sub>l C<sub>o</sub>nfi<sub>gu</sub>r<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd E<sub>xpe</sub>rim<sub>e</sub>nt<sub>a</sub>l D<sub>es</sub>i<sub>g</sub>n

Overall desi n. Across all image and text benchmarks in Table 3, we keep the backbone, optimizer, training schedule, validation split, and evaluation metrics fixed within each dataset, changing only the attention or output-layer uncertainty mechanism. This controlled design ensures that diferences among MLE, MLE+Temp, MCD, SNGP, SGPA, RFF-GPA, and RFF-CGP reflect the uncertainty mechanism rather than unrelated architectural or tuning choices.

Image classification setup. For Fashion-MNIST, CIFAR-10, SVHN, and CIFAR-10-C, we use a compact Vision-Transformerstyle encoder with patch embeddings, learned positional embeddings, a learnable [CLS] token, pre-norm residual blocks, and a classification head on the [CLS] representation. Images are converted to tensors and standardized per channel using dataset-specific statistics; we do not use aggressive data augmentation so that calibration comparisons are not confounded by augmentation policy. Patch sizes and resulting sequence lengths follow Table 3.

Text classification setup. For 20 Newsgroups, Hyperpartisan, and SST-2, we use the same Transformer classification template with shared WordPiece tokenization, learned positional embeddings, a [CLS]-style pooled representation, and a linear classification head. Text sequences are truncated or padded to the maximum lengths reported in Table 3. As with the image experiments, all compared uncertainty mechanisms use the same dataset-specific backbone and training protocol. Unless otherwise stated, the text encoder uses the same model scale as the image encoder: 5 Transformer blocks, hidden size 128, MLP dimension 128, 4 attention heads, and dropout 0.1.

Shared model configuration. For every dataset, all compared methods use the same dataset-specific Transformer encoder and difer only in the uncertainty mechanism being evaluated. Thus, MLE, MLE+Temp, MCD, SNGP, SGPA/SCGP, RFF-GPA, and RFF-CGP are compared under matched depth, hidden size, number of heads, dropout, optimizer, validation split, and evaluation protocol for a given benchmark. For GP-based attention models, the corresponding image or text backbone is instantiated with the GP attention layer in every Transformer block. The sparse GP attention baselines (SGPA/SCGP) use the same backbone and classification head but replace random features with inducing-point GP attention. The deterministic MLE baseline uses the identical architecture with standard scaled dot-product attention $( \mathrm { k e r n e l \_ t y p e } ~ = ~ \mathtt { s c a l e \_ d o t } )$ , while SNGP keeps the attention mechanism deterministic and replaces the output head with a stochastic neural GP layer.

Across the GP-based models, we use one Monte Carlo sample per example during training, add a numerical jitter of $1 0 ^ { - 6 }$ in Cholesky/Woodbury solves, and anneal the KL regularization coeficient linearly from 0 to 1 over training via

$$
\lambda _ { \mathrm { K L } } ( e ) = \operatorname* { m i n } ( 1 , 2 e / E ) ,
$$

where e is the epoch index and E is the total number of training epochs for that dataset. Randomness is controlled via a single seed parameter (-seed), and we train each model over five seeds to compute the reported error bars.

Training protocol. All in-distribution experiments use the Adam optimizer

$$
\mathrm { A d a m } ( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 ) ,
$$

with no weight decay, $\mathrm { i . e . , } \lambda = 0$ . The learning rate is recomputed at the start of each epoch using a piecewise-linear warmup and decay schedule: we set the initial learning rate to $\eta _ { \mathrm { i n i } } = 1 0 ^ { - 5 }$ , the base learning rate to $\eta _ { \mathrm { b a s e } } = 5 \times 1 0 ^ { - 4 }$ , and the minimum learning rate to $\eta _ { \mathrm { m i n } } = 1 0 ^ { - 5 }$ . The warmup phase spans $E _ { \mathrm { w a r m } } = 5$ epochs, during which the learning rate increases linearly from $\eta _ { \mathrm { i n i } } ~ \mathrm { t o } ~ \eta _ { \mathrm { b a s e } }$ . This is followed by a linear decay phase of $E _ { \mathrm { d e c a y } } = 4 8 0$ epochs down to $\eta _ { \mathrm { m i n } }$ . For image benchmarks, we use a global batch size of $B _ { \mathrm { i m g } } = 1 2 8$ for training and $B _ { \mathrm { i m g , t e s t } } = 1 2 8$ for validation and test. For text benchmarks, each method uses the same tokenizer, maximum length, batch size, optimizer, and model-selection split within that dataset.

Image-model instantiation. The image encoder used for Fashion-MNIST, CIFAR-10, SVHN, and CIFAR-10-C is a compact Vision Transformer with $L _ { \mathrm { i m g } } = 5$ transformer blocks, hidden size $d _ { \mathrm { m o d e l } } ^ { \mathrm { i m g } } = 1 2 8$ , MLP dimension $d _ { \mathrm { f f } } ^ { \mathrm { i m g } } = 1 2 8 .$ , and $H _ { \mathrm { i m g } } = 4$ attention heads per layer. We use pre-norm residual blocks and apply dropout with rate $p _ { \mathrm { d r o p } } = \mathrm { \ddot { ~ } } 0 . 1$ in the attention and MLP layers. Images are split into non-overlapping patches according to Table 3, prepended with a learnable [CLS] token, and combined with learned positional embeddings. For CIFAR-10 and SVHN, $2 \times 2$ patches yield $1 6 \times 1 6 = 2 5 6$ patch tokens; for Fashion-MNIST, $4 \times 4$ patches yield $7 \times 7 = 4 9$ patch tokens.

Random-feature and inducing-point settings. Across both image and text benchmarks, we use $\mathtt { k e y s \_ l e n } ~ = ~ 3 2$ for our random-feature GP attention, which yields an efective random-feature dimension $M _ { \mathrm { e f f } } = 6 4$ per head via cosine–sine pairs. RFF-GPA draws 32 frequencies and forms cosine/sine features, while RFF-CGP uses cosine–sine features from 32 shared frequencies. The inducing-point baselines (SGPA/SCGP) use 16 inducing points under the same backbone and training protocol.

CIFAR-10-C evaluation. CIFAR-10 consists of 50,000 training and 10,000 test images of size $3 2 \times 3 2$ across 10 classes. From the 50,000 training images, we hold out 5,000 examples as a validation set and train on the remaining 45,000 examples. CIFAR-10-C is used only for inference-time distribution-shift evaluation: we reuse the corresponding CIFAR-10 checkpoints and evaluate on all 19 corruption types at 5 severity levels, reporting averages over corruptions and severities.

## E Efi<sub>c</sub>i<sub>ency an</sub>d S<sub>ca</sub>l<sub>a</sub>bilit<sub>y</sub> B<sub>enc</sub>h<sub>mar</sub>k

This appendix details the eficiency and scalability protocol summarized in Sec. 4.2. All measurements probe asymptotic behavior and hardware eficiency independent of data loading, so we use synthetic inputs and benchmark a single attention block rather than an end-to-end model.

Benchmark unit. For each method we build one attention layer followed by a lightweight readout (LayerNorm → GELU → Linear to a scalar), and feed it a synthetic input tensor $\check { X } \in \mathbb { R } ^ { B \times L \times d }$ with $\mathbf { \bar { \boldsymbol { X } } } \sim \bar { \mathcal { N } } ( 0 , I )$ . The attention layers are the exact same implementations used in our in-distribution experiments (Appendix B); only the surrounding backbone is replaced by the synthetic-input readout so that the measured cost reflects the attention mechanism itself. We sweep the sequence length L ∈ {64, 128, 256, 512, 1024, 2048, 4096} and, for the inducing-point baselines, draw a fresh set of inducing keys per step.

Model configuration. To match the in-distribution setup (Appendix B), all methods use hidden size $d _ { \mathrm { m o d e l } } = 1 2 8 , H = 4$ attention heads, batch size $B = 1 6 ,$ , and float32. Random-feature methods use an efective feature count of $M = 3 2$ per head (RFF-GPA: feature\_mode=rff with $\mathtt { k e y s \_ l e n } = 1 6 ; \mathtt { R F F \_ C G P : } M = \mathtt { k e y s \_ l e n } )$ ; inducing-point methods (SGPA, SCGP) use 16 inducing points; and we set the GP jitter to $1 0 ^ { - 6 }$ and dropout to 0.1. GP variants draw a single Monte-Carlo sample per forward pass at training time and 5 samples at inference time, while the deterministic MLE and SNGP heads use a single pass.

Timing and memory measurement. Each configuration is run with 10 untimed warmup steps followed by 60 timed steps; every step performs a full forward pass, backward pass, and AdamW update (learning rate $3 \times 1 0 ^ { - \dot { 4 } } )$ . We use torch.cuda.synchronize around each phase, report the mean per-step time (with forward/backward/optimizer components measured separately), compute throughput as ${ \bar { \boldsymbol { B } } } \cdot { \boldsymbol { L } }$ divided by the mean step time, and read peak memory from torch.cuda.max\_memory\_allocated. For the runtime–calibration comparison in Sec. 4.2, runtime is reported as time per 1000 tokens at $L = 1 0 2 4$ , and the corresponding ECE values are taken from the in-distribution CIFAR-10 evaluation in Sec. 4.1. All runs are repeated over five seeds.

Hardware and large-L handling. All timing and memory results are measured on a single RTX 4090-class GPU (24 GB). The inducing-point baselines materialize an $\bar { L } \times L$ kernel and therefore grow quadratically in memory: SCGP exhausts the 24 GB budget beyond $L = 5 1 2$ , and SGPA beyond $L = 1 0 2 4$ . For these methods we measure up to their largest feasible length and fit a power-law trend to the measured values (metric ∝ L<sup>p</sup>), consistent with their known $\mathcal { O } ( \dot { L } ^ { 2 } )$ scaling; the random-feature methods (RFF-GPA, RFF-CGP) and the softmax heads (MLE, SNGP) are measured directly across the entire range.

## F O<sub>u</sub>t<sub>-o</sub>f<sub>-</sub>Di<sub>s</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> R<sub>o</sub>b<sub>us</sub>t<sub>ness</sub> P<sub>ro</sub>t<sub>oco</sub>l

This appendix details the CIFAR-10-C evaluation summarized in Sec. 4.3 and reported in Table 2. The evaluation is inferenceonly: we reuse the exact CIFAR-10 checkpoints, backbone, and metrics from the in-distribution experiment (Sec. 4.1, Appendix B) and never re-train or fine-tune on corrupted data, so the results measure how the trained models generalize to a shifted input distribution rather than the efect of any additional training.

Corru tion benchmark. We use the standard CIFAR-10-C benchmark (Hendrycks and Dietterich 2019), which perturbs the 10,000-image CIFAR-10 test set with 19 corruption types spanning four families—noise (Gaussian, shot, impulse, speckle), blur (defocus, glass, motion, zoom, Gaussian), weather (snow, frost, fog, brightness, spatter), and digital (contrast, elastic transform, pixelate, JPEG compression, saturate)—each applied at five severity levels $s \in \{ 1 , \ldots , 5 \}$ . Every corruption/severity pair thu yields a 10,000-image corrupted test set with the original CIFAR-10 labels. We treat the clean CIFAR-10 test set as severity $s = 0 ,$ , taken directly from the in-distribution evaluation in Sec. 4.1. In total each model is evaluated on $1 9 \times 5 = 9 5$ corrupted test sets plus the clean set.

Preprocessing. Corrupted images are provided as raw $3 2 \times 3 2$ RGB arrays. We apply the same preprocessing as during training—conversion to [0, 1] tensors followed by per-channel standardization with the CIFAR-10 training mean (0.4914, 0.4822, 0.4465) and standard deviation (0.2470, 0.2435, 0.2616)—and apply no test-time augmentation. Using the training-time normalization statistics is essential: any mismatch would itself introduce a distribution shift and confound the corruption efect we intend to measure. Patchification into a length-256 token sequence is handled inside the ViT backbone, identically to the in-distribution setting.

Inference and metrics. Each checkpoint is evaluated with the same predictive protocol as in Sec. 4.1: GP-based variants (SGPA, RFF-CGP, RFF-GPA) average 10 Monte-Carlo samples per input, whereas the deterministic MLE and SNGP heads use a single forward pass. For every corruption/severity pair we compute accuracy (ACC), negative log-likelihood (NLL), and expected calibration error (ECE, 15 equal-width confidence bins) on the predicted class probabilities, using the identical metric implementations as the in-distribution experiment.

Aggregation. For each model and severity level we first average every metric over the 19 corruption types, yielding one per-severity value per run. We repeat the whole evaluation for the five training seeds and report, in Table 2, the mean over seeds together with a 95% confidence interval (computed as $1 . 9 6 \hat { \sigma } / \sqrt { 5 } )$ . The clean column $( s \ : = \ : 0 )$ is the seed-averaged in-distribution CIFAR-10 result under the same aggregation. This corruption-averaged, per-severity summary follows standard practice for CIFAR-10-C and isolates the efect of increasing corruption strength from the idiosyncrasies of individual corruption types.