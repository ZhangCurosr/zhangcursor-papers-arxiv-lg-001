# CAFE+FNO: Fourier Kernel Generation via Multiplicative Feature Composition

Hyungjoon Juen<sup>1</sup> and Minwoo Shin<sup>∗1</sup>

<sup>1</sup>Department of Software, Yonsei University (Mirae Campus), Wonju 26493, Republic of Korea

## Abstract

The Fourier Neural Operator (FNO) learns solution operators of partial diferential equations (PDEs) through Fourier-space kernel parameterization, but frequency truncation can limit the learning of high-frequency variations. AM-FNO and SirenFNO generate kernels for all grid modes from spectral coordinates using shared networks, making coordinate encoding and generator design important. Recent work on implicit neural representations (INRs) has proposed constructing frequency interactions through explicit feature composition rather than relying on subsequent MLPs to form them implicitly. Building on this approach, we propose CAFE+FNO, which incorporates Content-Aware Frequency Encoding+ (CAFE+) into Fourier kernel generation. CAFE+ combines Fourier–Chebyshev features through parallel afine branches and a Hadamard product, forming interactions within and across the two feature families. A kernel MLP maps the resulting representation of each normalized spectral coordinate to a complex channel-mixing matrix. Each layer shares its generator across all stored modes, making the number of trainable parameters independent of the number of modes for a fixed architecture. We compare CAFE+FNO with existing FNO variants on five PDE benchmarks and conduct ablation studies on basis configuration, multiplicative composition, and bandwidth learnability. Code and experimental configurations are available at https://github.com/fabsk101/CAFEPlusFNO.git.

## 1 Introduction

Partial diferential equations (PDEs) describe the spatiotemporal behavior of physical systems, including fluid flow, heat transfer, and elastic deformation. Traditional numerical methods, such as the finite diference method (FDM) and the finite element method (FEM), provide reliable solutions but incur substantial costs in repeated analyses since the PDE must be solved again for each new initial or boundary condition. Neural operators approximate the mapping from an input function to a solution function, enabling much faster prediction than conventional numerical methods without being restricted to a particular data discretization. In particular, the Fourier Neural Operator (FNO) eficiently models global interactions by computing kernel integrals in the Fourier domain [1].

FNO assigns an independent channel-mixing matrix to each Fourier mode. Parameterizing all modes increases kernel parameter counts and memory costs with the number of modes and the problem dimension. Standard implementations therefore retain a prescribed number of lowfrequency modes and discard the others [2].

Amortized Fourier Neural Operator (AM-FNO) and SirenFNO address this frequency truncation by generating full-frequency Fourier kernels with neural networks that take spectral coordinates as input, instead of storing each kernel coeficient independently [3, 4]. In this amortized parameterization, each layer shares its kernel generator across all modes, keeping the number of learnable parameters independent of grid resolution. Consequently, kernel coeficients of each mode are outputs of a shared function R(ξ<sub>k</sub>) rather than independent parameters.

This shifts the focus from storing mode-specific kernel coeficients to approximating a frequencyto-kernel function shared across all modes. Full-frequency kernel learning thus requires considering both mode coverage and the representation used to generate mode-dependent complex kernels from spectral coordinates.

To improve this representation, we propose CAFE+FNO, which incorporates the multiplicative feature composition of Content-Aware Frequency Encoding+ (CAFE+) [5] into the amortized Fourier kernel generation of AM-FNO. CAFE+FNO constructs learnable interactions among spectral-coordinate features before kernel prediction and uses them to generate a complex channelmixing kernel for each Fourier mode. It retains amortized parameterization over all stored modes without additional frequency truncation while introducing a learnable feature-composition stage before kernel prediction.

CAFE+FNO achieves the lowest mean error on four of five PDE benchmarks, reducing error relative to SirenFNO by 18.4–60.2% on these tasks; SirenFNO performs better on ReacDif.

1. We propose CAFE+FNO, which introduces the Fourier–Chebyshev features and multiplicative composition of CAFE+ into spectral-coordinate-to-kernel mapping.

2. We formulate CAFE+ based kernel generation using both dense parameterization and functional tensor parameterizations based on CP, TT, and Tucker decompositions.

3. We compare against existing FNO variants under the same dataset-specific training and evaluation conditions and analyze the empirical efects of the architecture through ablations on the basis configuration, composition block, and bandwidth learning.

## 2 Related Work

Neural operators. Neural operators learn mappings between function spaces [6, 7]. DeepONet [8] combines branch and trunk networks, while graph-based neural operators learn kernels between irregularly sampled points [9, 10]. Attention-based methods, including Galerkin Transformer [11], OFormer [12], GNOT [13], and Transolver [14], model interactions among discretized points or latent representations. FNO [1] implements global convolution through mode-wise Fourier transformations, retaining a prescribed set of low-frequency modes. Subsequent work improves multiscale representations [15]; U-FNO [16] adds a U-Net path to complement high-frequency representations, while F-FNO [17] and tensorized FNOs [18] reduce costs by factorizing spectral operations and kernel tensors, respectively. AM-FNO [3] and SirenFNO [4] generate kernels for all stored Fourier modes using shared networks conditioned on spectral coordinates. For fixed architectures, their parameter counts are independent of the number of modes. AM-FNO employs a KAN [19] or a Chebyshev-basis MLP, whereas SirenFNO uses a SIREN with learnable random Fourier features (RFFs). Our work introduces CAFE+ feature composition into this shared kernel parameterization.

Coordinate encodings for implicit neural representations. Coordinate-based implicit neural representations (INRs) map coordinates to continuous signals, but conventional MLPs tend to learn low-frequency components first [20, 21]. Multiplicative Filter Networks (MFNs) multiply coordinate-dependent sinusoidal or Gabor features with linearly transformed hidden features at each layer [22]. BACON constrains output bandwidths [23], while Residual MFNs combine skip connections with frequency initialization for multiscale learning [24]. SIREN uses sinusoidal activations [25], whereas Fourier feature mapping provides sinusoidal input embeddings [26]. Fourier encodings can be sensitive to frequency selection and bandwidth [27–29]. Nonlinear layers can generate harmonics and combinations of the encoded frequencies [30]. CAFE introduces multiplicative feature composition, and CAFE+ adds Chebyshev features for more stable low-frequency representations [5]. We use this encoding to generate Fourier kernels rather than directly reconstruct signals.

## 3 Preliminaries

## 3.1 Fourier Integral Operator

We consider the problem of approximating a solution operator between function spaces from a finite collection of input–output function pairs [31]. Let $\Omega \subset \mathbb { R } ^ { d _ { s } }$ be a bounded open computational domain, let A denote the admissible set of input functions a : $\Omega  \mathbb { R } ^ { d _ { a } }$ , and let U be a Banach space containing the solution functions $u \colon \Omega  \mathbb { R } ^ { d _ { u } }$ . Here, A is a subset of a suitable Banach function space. Assuming that each input $a \in { \mathcal { A } }$ uniquely determines a solution $u \in \mathcal { U }$ , we define the corresponding solution operator as

$$
\mathcal { G } ^ { \star } \colon \mathcal { A } \to \mathcal { U } , \qquad \mathcal { u } = \mathcal { G } ^ { \star } ( a ) .\tag{1}
$$

Suppose that N input–output function pairs $( a _ { i } , u _ { i } ) , ~ i = 1 , \ldots , N .$ , are available, where $u _ { i }$ denotes observations of the solution corresponding to input ${ { a } _ { i } } .$ Our goal is to use these observations to construct an operator $\mathcal { G } \colon \mathcal { A }  \mathcal { U }$ that approximates $\mathcal G ^ { \star }$ . To this end, we parameterize $\mathcal { G }$ using a neural network with finitely many trainable parameters and learn these parameters by reducing the discrepancy between the predictions $\mathcal { G } ( a _ { i } )$ and observations $u _ { i }$

To construct this approximation, FNO first maps the input function to a hidden function and   
then repeatedly applies layers consisting of an integral operator and a pointwise transformation. Let $\mathcal { V } = L ^ { 2 } ( \Omega ; \mathbb { R } ^ { d _ { v } } )$ be the hidden function space, and denote the input to layer ℓ by $v ^ { ( \ell ) } \in \mathcal { V }$   
Here, $d _ { v }$ is the dimension of the hidden representation, which remains the same across all layers.

The Fourier representation below describes convolution on a periodic extension of a rectangular computational domain. This computational periodic representation is distinct from the physical boundary conditions of the underlying PDE. For an integrable matrix-valued kernel $\kappa _ { \ell } ,$ the Fourier integral operator is given by

$$
\begin{array} { r } { ( K _ { \ell } v ^ { ( \ell ) } ) ( x ) = \displaystyle \int _ { \Omega } \kappa _ { \ell } ( x - y ) v ^ { ( \ell ) } ( y ) \mathrm { d } y } \\ { = \mathcal { F } ^ { - 1 } \Big ( R _ { \ell } \cdot \mathcal { F } [ v ^ { ( \ell ) } ] \Big ) ( x ) , } \end{array}\tag{2}
$$

where $\mathcal { F }$ and ${ \mathcal { F } } ^ { - 1 }$ denote the Fourier transform and inverse Fourier transform, respectively, and $R _ { \ell } ( k ) = \mathcal { F } [ \kappa _ { \ell } ] ( k ) \in \mathbb { C } ^ { d _ { v } \times d _ { v } }$ is the complex-valued kernel matrix associated with Fourier mode k.

In the discrete implementation considered hereafter, we denote the finite index set of Fourier modes represented by the computational grid by $E \subset \mathbb { Z } ^ { d _ { s } }$

## 3.2 CAFE+ Encoding

CAFE introduces learnable composition of Fourier features; CAFE+ extends this encoding with Chebyshev features. The concatenated features pass through parallel afine branches, whose outputs are multiplied elementwise before entering a backbone MLP. This constructs interactions within and across the two feature families. Originally designed for coordinate-based signal reconstruction, CAFE+ is used here to encode spectral coordinates for Fourier kernel generation. Section 4 defines the corresponding feature maps and composition. For an input coordinate x $\in \mathbb { R } ^ { D }$ and M Fourier frequency vectors $\{ \omega _ { i } \} _ { i = 1 } ^ { M }$ , where $\boldsymbol { \omega } _ { i } \in \mathbb { R } ^ { \bar { D } }$ , the Fourier features are defined as

$$
\begin{array} { r } { \Phi _ { \mathrm { F F } } ( \mathbf { x } ) = \left[ \sin ( 2 \pi \omega _ { i } ^ { \top } \mathbf { x } ) , \cos ( 2 \pi \omega _ { i } ^ { \top } \mathbf { x } ) \right] _ { i = 1 } ^ { M } . } \end{array}\tag{3}
$$

Here, M is the number of Fourier frequency vectors, and $\omega _ { i }$ is a Fourier frequency vector. Alongside the Fourier features, CAFE+ uses J Chebyshev polynomials of the first kind $T _ { 0 } , \dots , T _ { J - 1 }$ along each coordinate axis to construct the Chebyshev features:

$$
\Phi _ { \mathrm { C F } } ( \mathbf { x } ) = [ T _ { j } ( x _ { d } ) ] _ { d = 1 , \ldots , D ; \ j = 0 , \ldots , J - 1 } .\tag{4}
$$

CAFE+ concatenates the two sets of features, feeds them into $N _ { b }$ linear branches, and combines the branch outputs using a Hadamard product:

$$
\Psi ( \mathbf { x } ) = \bigodot _ { i = 1 } ^ { N _ { b } } \{ \mathbf { W } _ { i } \left[ \Phi _ { \mathrm { F F } } ( \mathbf { x } ) , \Phi _ { \mathrm { C F } } ( \mathbf { x } ) \right] + b _ { i } \} ,\tag{5}
$$

![](images/0e289dfcbd42518a79ce1c41440578e06304a5a9e8005f16807eefd2f9938c9a.jpg)  
Figure 1: Architecture of CAFE+FNO in two dimensions. The input is lifted by $P ,$ processed by L operator layers, and projected by Q to the predicted solution. Within each layer, random Fourier features and Chebyshev features of the normalized spectral coordinate $\xi _ { k }$ are concatenated and passed through parallel linear branches. Their outputs are combined via a Hadamard product, and a kernel MLP generates the mode-specific complex channel-mixing matrix used in the spectral convolution.

where J denotes Hadamard product.

This multiplicative composition establishes interactions between Fourier and Chebyshev bases at the encoding stage, and the resulting $\Psi ( \mathbf { x } )$ is subsequently used as the input to the backbone MLP.

## 4 Method

We propose CAFE+FNO, which uses the multiplicative feature composition of CAFE+ as the input representation for a Fourier kernel generator. Whereas the original CAFE+ constructs a representation for reconstructing signal values from coordinates, our approach generates the kernel of each Fourier mode from normalized spectral coordinates. We first describe the dense kernel generator; extensions based on CP, TT, and Tucker factorizations are presented in Appendix C.

## 4.1 Fourier–Chebyshev Feature Encoding

Let $\pmb { \xi } _ { k } \in [ - 1 , 1 ] ^ { d _ { s } }$ denote the normalized spectral coordinate corresponding to each Fourier mode $k \in E$ . The coordinate $\xi _ { k }$ is the input to the kernel generator; the coordinate normalization is specified in Appendix B. In each layer $\ell ,$ we independently sample the entries of a Gaussian matrix $G _ { \ell }$ once from the standard normal distribution to map this coordinate to m pairs of Fourier features:

$$
G _ { \ell } \in \mathbb { R } ^ { d _ { s } \times m } , \qquad ( G _ { \ell } ) _ { i j } \sim { \mathcal { N } } ( 0 , 1 ) .\tag{6}
$$

Here, $d _ { s } \in \{ 1 , 2 \}$ is the spatial dimension. The sampled $G _ { \ell }$ remains fixed during training, and a Fourier projection matrix $B _ { \ell } = \rho _ { \ell } G _ { \ell }$ is constructed using $\rho _ { \ell } > 0$ . The matrix $B _ { \ell } \in \mathbb { R } ^ { d _ { s } \times m }$ maps the normalized spectral coordinate to RFF phases in the ℓth operator layer.

Our implementation defines the RFF phase as $\pi B _ { \ell } ^ { \top } \pmb { \xi } _ { k }$ , with the frequency scale incorporated into $B _ { \ell }$

The RFF encoding of the spectral coordinate in layer ℓ is defined as

$$
\Phi _ { \mathrm { F F } } ^ { ( \ell ) } ( \pmb { \xi } _ { k } ; \rho _ { \ell } ) = \left[ \begin{array} { c } { \cos ( \pi B _ { \ell } ^ { \top } \pmb { \xi } _ { k } ) } \\ { \sin ( \pi B _ { \ell } ^ { \top } \pmb { \xi } _ { k } ) } \end{array} \right] \in \mathbb { R } ^ { 2 m } .\tag{7}
$$

Sine and cosine are applied elementwise, and their outputs are concatenated into a 2m-dimensional feature vector. The parameter $\rho _ { \ell }$ determines the frequency scale of the spectral-coordinate encoding by controlling the magnitude of the Gaussian projection.

Since CAFE+ uses Chebyshev features alongside Fourier features, we also compute n first-kind Chebyshev basis functions for each coordinate axis. Writing $t _ { n } ( s ) = [ T _ { 0 } ( s ) , \ldots , T _ { n - 1 } ( s ) ] ^ { \intercal } \in \mathbb { R } ^ { n }$ we obtain

$$
\begin{array} { r } { \Phi _ { \mathrm { C F } } ( \xi _ { k } ) = \big [ t _ { n } ( \xi _ { k , 1 } ) ^ { \top } , \dots , t _ { n } ( \xi _ { k , d _ { s } } ) ^ { \top } \big ] ^ { \top } \in \mathbb { R } ^ { d _ { s } n } . } \end{array}\tag{8}
$$

The Fourier and Chebyshev features are then concatenated:

$$
\begin{array} { r } { \gamma _ { \ell } ( \pmb { \xi } _ { k } ) = \left[ \Phi _ { \mathrm { C F } } ( \pmb { \xi } _ { k } ) ^ { \top } , \Phi _ { \mathrm { F F } } ^ { ( \ell ) } ( \pmb { \xi } _ { k } ; \rho _ { \ell } ) ^ { \top } \right] ^ { \top } . } \end{array}\tag{9}
$$

The resulting feature is $\gamma _ { \ell } ( \pmb { \xi } _ { k } ) \in \mathbb { R } ^ { 2 m + d _ { s } n }$ , with embedding dimension $2 m + d _ { s } n$

## 4.2 CAFE+ Multiplicative Frequency Composition

CAFE+FNO feeds $\gamma _ { \ell } ( \pmb { \xi } _ { k } )$ into $N _ { b }$ parallel branches before passing the resulting representation to the kernel MLP. Each branch applies an afine transformation with a bias, referred to hereafter as a linear branch:

$$
h _ { \ell , r } ( \pmb { \xi } _ { k } ) = \widetilde W _ { \ell , r } \gamma _ { \ell } ( \pmb { \xi } _ { k } ) + \widetilde b _ { \ell , r } , \qquad r = 1 , \dots , N _ { b } ,\tag{10}
$$

where $\widetilde { W } _ { \ell , r } \in \mathbb { R } ^ { d _ { z } \times ( 2 m + d _ { s } n ) }$ and $\widetilde { b } _ { \ell , r } \in \mathbb { R } ^ { d _ { z } }$ . The branch outputs are combined by a Hadamard product:

$$
\mathbf { z } _ { \ell } ( \pmb { \xi } _ { k } ) = \bigcirc _ { r = 1 } ^ { N _ { b } } h _ { \ell , r } ( \pmb { \xi } _ { k } ) .\tag{11}
$$

Since each branch is an afine transformation of $\gamma _ { \ell } ( \pmb { \xi } _ { k } )$ , each component of $\mathbf { z } _ { \ell } ( \pmb { \xi } _ { k } )$ can be expressed as a polynomial of degree at most $N _ { b }$ in the input feature components. In particular, two branches yield quadratic interactions together with linear and constant terms, whose coeficients are parameterized by products of branch weights and biases.

To examine the frequency structure of these interactions, we fix layer ℓ and write the Fourier projection matrix as $B _ { \ell } = [ \omega _ { \ell , 1 } , \dots , \omega _ { \ell , m } ]$ . Here, $\omega _ { \ell , i }$ is an encoding frequency vector for the spectral coordinate ${ \xi } _ { k }$ , distinct from the Fourier mode k of the hidden function. We define the phase associated with each column vector as $\theta _ { \ell , i } = \pi \omega _ { \ell , i } ^ { \top } \pmb { \xi } _ { k }$ for $i = 1 , \ldots , m$ . For this example, we abbreviate $\theta _ { i } : = \theta _ { \ell , i }$ and $\theta _ { j } : = \theta _ { \ell , j }$ . We simplify the same component of the two branch output vectors as

$$
h _ { 1 } = \alpha _ { 1 } \cos \theta _ { i } + \beta _ { 1 } \sin \theta _ { i } , \qquad h _ { 2 } = \alpha _ { 2 } \cos \theta _ { j } + \beta _ { 2 } \sin \theta _ { j } .\tag{12}
$$

Applying the product-to-sum identities gives

$$
\begin{array} { c } { { h _ { 1 } h _ { 2 } = { \frac { 1 } { 2 } } \big [ \big ( \alpha _ { 1 } \alpha _ { 2 } + \beta _ { 1 } \beta _ { 2 } \big ) \cos ( \theta _ { i } - \theta _ { j } ) + \big ( \alpha _ { 1 } \alpha _ { 2 } - \beta _ { 1 } \beta _ { 2 } \big ) \cos ( \theta _ { i } + \theta _ { j } ) } } \\ { { + \left( \beta _ { 1 } \alpha _ { 2 } - \alpha _ { 1 } \beta _ { 2 } \right) \sin ( \theta _ { i } - \theta _ { j } ) + \big ( \beta _ { 1 } \alpha _ { 2 } + \alpha _ { 1 } \beta _ { 2 } \big ) \sin ( \theta _ { i } + \theta _ { j } ) \big ] . } } \end{array}\tag{13}
$$

Since

$$
\begin{array} { r } { \theta _ { i } \pm \theta _ { j } = \pi ( \omega _ { \ell , i } \pm \omega _ { \ell , j } ) ^ { \top } \pmb { \xi } _ { k } , } \end{array}\tag{14}
$$

the representation after the Hadamard product can include, in addition to the initial RFF basis, basis frequencies corresponding to the sums $\omega _ { \ell , i } + \omega _ { \ell , j }$ and diferences $\omega _ { \ell , i } - \omega _ { \ell , j }$ . The coeficient of each component is determined by the learnable linear-branch weights $\alpha _ { 1 } , \beta _ { 1 } , \alpha _ { 2 } , \beta _ { 2 }$ , allowing the network to selectively adjust the frequency interactions needed for kernel generation.

Products of Chebyshev components similarly satisfy

$$
\begin{array} { r } { T _ { p } ( x ) T _ { q } ( x ) = \frac { 1 } { 2 } \big [ T _ { p + q } ( x ) + T _ { | p - q | } ( x ) \big ] , \qquad 0 \leq p , q \leq n - 1 . } \end{array}\tag{15}
$$

Thus, multiplying Chebyshev features along the same coordinate axis can construct higher-order polynomial components absent from the initial encoding, while also recombining lower-order components.

Furthermore, since both feature types enter the same branches, products of branch outputs can contain Fourier–Chebyshev cross terms such as $T _ { p } ( \xi _ { k , a } )$ cos $\theta _ { \ell , i }$ and $T _ { p } ( \xi _ { k , a } )$ sin $\theta _ { \ell , i }$ . These terms form sinusoidal features whose amplitudes are modulated by polynomials. Consequently, $\mathbf { z } _ { \ell } ( \xi _ { k } )$ represents both interactions within each basis and cross interactions between the two bases. The kernel MLP in the next subsection uses this representation to generate a complex-valued kernel matrix for each mode.

The frequencies composed here are encoding frequencies with respect to the spectral coordinate ${ \xi } _ { k }$ . This operation neither directly multiplies diferent Fourier coeficients of the PDE solution nor adds new Fourier modes to the computational grid. The purpose of the operation is to provide basis interactions for representing the mapping from ${ \xi } _ { k }$ to the kernel matrix. Its efect on solution prediction is evaluated in the experiments in Section 5.

## 4.3 Complex Fourier Kernel Generation

Let $v ^ { ( \ell ) } : \Omega  \mathbb { R } ^ { d _ { v } }$ be the hidden representation at layer ℓ. The CAFE+ mode representation $\mathbf { z } _ { \ell } ( \pmb { \xi } _ { k } ) \in \mathbb { R } ^ { d _ { z } }$ constructed above is passed to a complex matrix-valued kernel generator $\mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { M } _ { \ell } : \mathcal { ~ \Lambda } _ { \ell } : \mathcal { \Lambda } _ { \ell } { \mathcal } { \Lambda } _ { \Lambda } \mathcal { \Lambda } _ { \Lambda } \mathcal { \Lambda } \mathcal { \Lambda } _ { \Lambda \Lambda } \mathcal { \Lambda } \mathcal { \Lambda } \Lambda \mathcal { \Lambda } \Lambda \Lambda _ { \Lambda \Lambda } \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \mathcal { \Lambda \Lambda } \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda $ $\mathbb R ^ { d _ { z } }  \mathbb C ^ { d _ { v } \times d _ { v } }$ . In practice, we use a single real-valued MLP $f _ { \ell } : \mathbb { R } ^ { d _ { z } }  \mathbb { R } ^ { 2 d _ { v } ^ { 2 } }$ . Its first and last $d _ { v } ^ { 2 }$ output components are split into real and imaginary parts, respectively, and reshaped into $d _ { v } \times d _ { v }$ matrices:

$$
\begin{array} { r } { \left[ r _ { \ell } ^ { \mathrm { R e } } ( k ) \right] = f _ { \ell } \big ( \mathbf { z } _ { \ell } ( \xi _ { k } ) \big ) , \qquad r _ { \ell } ^ { \mathrm { R e } } ( k ) , r _ { \ell } ^ { \mathrm { I m } } ( k ) \in \mathbb { R } ^ { d _ { v } ^ { 2 } } . } \end{array}\tag{16}
$$

The kernel corresponding to Fourier mode k is therefore defined as

$$
R _ { \ell } ( k ) = \mathcal { M } _ { \ell } \big ( \mathbf { z } _ { \ell } ( \pmb { \xi } _ { k } ) \big ) = R _ { \ell } ^ { \mathrm { R e } } ( k ) + \mathrm { i } R _ { \ell } ^ { \mathrm { I m } } ( k ) .\tag{17}
$$

The same $\mathcal { M } _ { \ell }$ and encoding parameters are shared across all Fourier modes within layer $\ell ,$ whereas diferent layers use independent parameters.

The kernel generator takes only spectral coordinates as input and is not conditioned on an individual input function a. Thus, after training, the kernel generated for a given mode on the same grid and in the same layer is identical across input samples. Here, content-aware composition means that end-to-end training through the PDE solution prediction error forms feature combinations suited to the data. It does not imply supervision with separate ground-truth kernels or the generation of a diferent kernel for each input sample.

Consequently, when the generator architecture and feature dimensions are fixed, the number of trainable parameters is independent of the size of the mode set $E .$ However, the cost of evaluating and storing kernels over all modes increases with the grid size. We apply a Hermitian projection to the generated complex kernels to maintain consistency with the inverse Fourier transform of a real-valued hidden function:

$$
\begin{array} { r } { \widetilde { R } _ { \ell } = \Pi _ { \mathcal { H } } ( R _ { \ell } ) , \qquad \widetilde { R } _ { \ell } ( - k ) = \overline { { \widetilde { R } _ { \ell } ( k ) } } . } \end{array}\tag{18}
$$

This symmetry condition applies to the completed spectrum. The specific projection used for the rFFT storage array is described in Appendix B.

Finally, $\widetilde { R } _ { \ell } ( k )$ acts linearly on the Fourier coeficients of the hidden representation at each mode:

$$
\mathcal { F } [ K _ { \ell } v ^ { ( \ell ) } ] ( k ) = ( \widetilde { R } _ { \ell } \cdot \widehat { v } ^ { ( \ell ) } ) ( k ) , \qquad \widehat { v } ^ { ( \ell ) } = \mathcal { F } [ v ^ { ( \ell ) } ] .\tag{19}
$$

## 4.4 CAFE+FNO Architecture

CAFE+FNO follows the basic operator architecture of AM-FNO, replacing only the kernel generator in each Fourier layer with the CAFE+ based parameterization defined above. The complete model is written as

$$
\mathcal G : = Q \circ \mathcal L _ { L } \circ \mathcal L _ { L - 1 } \circ \cdot \cdot \cdot \circ \mathcal L _ { 1 } \circ P ,\tag{20}
$$

Table 1: Description of the benchmark datasets.
<table><tr><td>Dataset</td><td>Input</td><td>Target</td><td>Resolution</td><td>Train/test</td><td>Source</td></tr><tr><td>Darcy</td><td> $a ( x )$ </td><td> $u ( x )$ </td><td> $1 2 8 \times 1 2 8$ </td><td>1000/200</td><td>NeuralOperator</td></tr><tr><td>NS</td><td> $u ( x , t _ { \mathrm { i n } } )$ </td><td> $u ( x , t _ { \mathrm { o u t } } )$ </td><td>128×128</td><td>1000/200</td><td>NeuralOperator</td></tr><tr><td>Burgers</td><td> $[ u ( x , t _ { j } ) ] _ { j = 0 } ^ { 9 }$ </td><td> $[ u ( x , t _ { j } ) ] _ { j = 1 0 } ^ { 1 9 }$ </td><td>1024</td><td>1000/200</td><td>PDEBench</td></tr><tr><td>Airfoil</td><td> $( X , Y )$ </td><td> $u ( x )$ </td><td>221×51</td><td>1000/200</td><td>Geo-FNO</td></tr><tr><td>ReacDiff</td><td> $[ u ( x , t _ { j } ) ] _ { j = 0 } ^ { 9 }$ </td><td> $[ u ( \boldsymbol { x } , t _ { j } ) ] _ { j = 1 0 } ^ { 1 9 }$ </td><td>1024</td><td>1000/200</td><td>PDEBench</td></tr></table>

with $\boldsymbol { v } ^ { ( 1 ) } = \boldsymbol { P } ( \boldsymbol { a } )$ and $\widehat { u } = Q \big ( v ^ { ( L + 1 ) } \big )$ . The lifting operator $P : \mathcal { A }  \mathcal { V }$ constructs a $d _ { v }$ -dimensional hidden function from the values of the input function and the computational coordinates. Each operator layer $\mathcal { L } _ { \ell }$ updates the hidden function, and the projection operator $Q : \mathcal { V }  \mathcal { U }$ maps the final hidden function to the predicted solution function.

Let $\widetilde { R } _ { \ell } ( k )$ denote the generated complex kernel after Hermitian projection. The ℓth layer is computed as

$$
\widetilde { v } ^ { ( \ell ) } = \mathcal { F } ^ { - 1 } \Big ( \widetilde { R } _ { \ell } \cdot \mathcal { F } [ v ^ { ( \ell ) } ] \Big ) ,\tag{21}
$$

$$
v ^ { ( \ell + 1 ) } ( x ) = \sigma \Bigl ( v ^ { ( \ell ) } ( x ) + \mathrm { F F N } _ { \ell } \bigl ( \widetilde { v } ^ { ( \ell ) } ( x ) \bigr ) \Bigr ) .
$$

Here, $\mathrm { F F N } _ { \ell } ( y ) = \widehat { W } _ { \ell } ^ { ( 2 ) } \sigma ( \widehat { W } _ { \ell } ^ { ( 1 ) } y + b _ { \ell } ^ { ( 1 ) } ) + b _ { \ell } ^ { ( 2 ) }$ , and $\sigma$ is the activation function. In the final operator layer, only the activation after the residual addition is omitted; the activation within the FFN is retained. The final hidden representation is mapped to the output through the same projection.

## 5 Experiments

This section presents the experimental results. All experiments were conducted on a desktop computer equipped with an AMD Ryzen 9800XD 8-Core CPU, an NVIDIA GeForce RTX 5080 GPU, and 64GB of Samsung DDR5-5600 RAM (16GB $\mathrm { ~ x ~ 4 ) ~ }$

## 5.1 Benchmarks and baselines

We compare CAFE+FNO with FNO and several established variants: U-FNO, TFNO-CP with CP decomposition, AM-FNO with an MLP, and SirenFNO with a SIREN hypernetwork. We also compare CP-, TT-, and Tucker-based variants of both SirenFNO and CAFE+FNO.

Table 1 lists the benchmark datasets. Darcy flow [32] and Navier–Stokes [33] are obtained from the NeuralOperator library; Burgers and reaction–difusion are from PDEBench [34, 35]; and Airfoil is from the Geo-FNO benchmark [36].

For Airfoil, the publicly available materials were insuficient to reconstruct the corresponding U-FNO implementation consistently, and the fixed baseline implementation required additional compatibility modifications. We therefore exclude U-FNO from this benchmark and indicate the corresponding entry with a dash in Table 2.

Appendix A provides dataset-specific preprocessing, prediction protocols, training losses, and evaluation settings.

## 5.2 Experimental setup

For a fair comparison, we retrained FNO, U-FNO, TFNO-CP, AM-FNO, SirenFNO, and CAFE+FNO using the same data splits, preprocessing, and training pipeline. Each baseline architecture follows the dataset-specific settings of its public implementation. For Airfoil, which is not included in the original SirenFNO benchmarks, we used a separately specified architecture configuration.

All models were trained for 500 epochs using AdamW, with an initial learning rate of $1 0 ^ { - 3 }$ and weight decay of $1 0 ^ { - 4 }$ . The learning rate was reduced using cosine annealing. We used neither validation-based model selection nor early stopping; all results were evaluated using the final-epoch model. The main evaluation metric is the mean sample-wise relative $L _ { 2 }$ error,

Table 2: Final test relative $L _ { 2 }$ errors $( \times 1 0 ^ { - 3 } )$ at epoch 500: mean ± sample standard deviation over five seeds. Bold marks the lowest mean per dataset. AM-FNO uses the MLP variant; U-FNO was excluded a priori on Airfoil (—). Evaluation details appear in Appendix A.
<table><tr><td>Model</td><td>Darcy  $1 2 8 \times 1 2 8$ </td><td>NS  $1 2 8 \times 1 2 8$ </td><td>Burgers 1024</td><td>Airfoil  $2 2 1 \times 5 1$ </td><td>ReacDiff 1024</td></tr><tr><td>FNO</td><td> $6 5 . 4 0 \pm 4 . 1 8$ </td><td> $5 6 . 2 9 \pm 0 . 9 3$ </td><td> $1 1 . 3 0 \pm 0 . 0 8$ </td><td> $6 . 1 7 \pm 0 . 1 7$ </td><td> $4 . 1 8 \pm 0 . 1 5$ </td></tr><tr><td>U-FNO</td><td> $4 7 . 0 5 \pm 1 . 8 3$ </td><td> $5 7 . 6 9 \pm 0 . 7 8$ </td><td> $1 0 . 0 2 \pm 0 . 2 1$ </td><td></td><td> $9 . 1 3 \pm 5 . 7 7$ </td></tr><tr><td>TFNO-CP</td><td> $4 7 . 3 4 \pm 1 . 2 8$ </td><td> $3 5 . 1 4 \pm 0 . 9 6$ </td><td> $1 0 . 2 7 \pm 0 . 4 9$ </td><td> $7 . 0 6 \pm 0 . 4 8$ </td><td> $4 . 0 7 \pm 0 . 2 6$ </td></tr><tr><td>AM-FNO(MLP)</td><td> $6 2 . 0 2 \pm 1 2 . 9 0$ </td><td> $2 9 . 9 4 \pm 1 . 8 3$ </td><td> $1 8 . 7 3 \pm 3 . 2 0$ </td><td> $6 . 2 9 \pm 0 . 6 8$ </td><td> $1 7 5 8 . 0 9 \pm 3 8 4 3 . 6 3$ </td></tr><tr><td>SirenFNO</td><td> $2 4 . 3 4 \pm 1 . 2 4$ </td><td> $4 3 . 2 4 \pm 2 . 0 2$ </td><td> $5 . 8 6 \pm 0 . 4 1$ </td><td> $5 . 7 6 \pm 0 . 2 4$ </td><td> ${ \bf 2 . 3 6 \pm 0 . 7 8 }$ </td></tr><tr><td>CAFE+FNO</td><td> ${ \bf 1 3 . 7 0 \pm 0 . 1 3 }$ </td><td> ${ \bf 1 7 . 2 0 \pm 0 . 3 2 }$ </td><td> ${ \bf 4 . 1 3 \pm 0 . 3 7 }$ </td><td> ${ \bf 4 . 7 0 \pm 0 . 1 2 }$ </td><td> $3 . 8 0 \pm 0 . 2 7$ </td></tr><tr><td colspan="6">Factorized SirenFNO</td></tr><tr><td>CP</td><td> $4 6 . 0 9 \pm 2 1 . 7 1$ </td><td> $5 1 . 4 6 \pm 7 . 2 9$ </td><td> $8 . 9 1 \pm 0 . 2 9$ </td><td> $8 . 1 2 \pm 1 . 6 8$ </td><td> $4 . 3 7 \pm 1 . 7 0$ </td></tr><tr><td>TT</td><td> $1 7 8 . 7 8 \pm 2 5 7 . 7 6$ </td><td> $3 6 . 0 4 \pm 5 . 5 3$ </td><td> $7 . 7 2 \pm 0 . 2 1$ </td><td> $6 . 0 2 \pm 0 . 2 6$ </td><td> $2 . 6 3 \pm 0 . 2 3$ </td></tr><tr><td>Tucker</td><td> $6 4 . 0 6 \pm 2 7 . 0 5$ </td><td> $4 6 . 7 0 \pm 1 2 . 2 4$ </td><td> $8 . 1 1 \pm 0 . 4 3$ </td><td> $8 . 1 8 \pm 3 . 8 1$ </td><td> $2 . 9 2 \pm 0 . 0 9$ </td></tr><tr><td colspan="6">Factorized CAFE+FNO</td></tr><tr><td>CP</td><td> $3 0 . 0 5 \pm 2 . 5 7$ </td><td> $6 8 . 9 1 \pm 2 6 . 9 4$ </td><td> $8 . 7 3 \pm 0 . 8 4$ </td><td> $8 . 6 8 \pm 2 . 5 2$ </td><td> $4 . 1 5 \pm 0 . 4 8$ </td></tr><tr><td>TT</td><td> $2 7 . 7 7 \pm 1 . 8 2$ </td><td> $4 2 . 5 0 \pm 1 2 . 9 4$ </td><td> $8 . 4 8 \pm 0 . 4 5$ </td><td> $7 . 0 6 \pm 0 . 6 1$ </td><td> $4 . 0 9 \pm 0 . 4 2$ </td></tr><tr><td>Tucker</td><td> $2 7 . 9 6 \pm 2 . 2 2$ </td><td> $3 9 . 5 8 \pm 5 . 9 2$ </td><td> $8 . 9 7 \pm 0 . 4 7$ </td><td> $8 . 1 3 \pm 1 . 2 2$ </td><td> $3 . 7 2 \pm 0 . 5 4$ </td></tr></table>

$$
\mathcal { L } _ { \mathrm { r e l } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { | | \mathcal { G } ( a _ { i } ) - u _ { i } | | _ { 2 } } { | | u _ { i } | | _ { 2 } } .\tag{22}
$$

Here, N is the number of evaluation samples, and $u _ { i }$ and $\mathcal { G } ( a _ { i } )$ are the targets and predictions expressed in the dataset-specific evaluation space. For Burgers and ReacDif, sample-wise relative $L _ { 2 }$ errors are computed by concatenating the temporal and spatial values over the entire prediction rollout. The other benchmarks are evaluated on the predicted target fields.

CAFE+FNO, SirenFNO, and AM-FNO generate Fourier kernels for all stored FFT modes, while FNO and TFNO-CP use the frequency truncation of their respective public configurations. Each CAFE+FNO layer uses an independent Gaussian matrix $G _ { \ell } .$ For Table $2 , \rho _ { \ell }$ was fixed to one throughout training. The efect of learning the bandwidth is examined in Section 5.4, and the CP, TT, and Tucker results are reported in Section 5.3.

Each experiment was repeated with five random seeds, {0, 42, 73, 108, 202}. We report the mean ± standard deviation of the final test errors. Dataset-specific batch sizes, preprocessing, training losses, and evaluation protocols are provided in Appendix A, CAFE+FNO architecture settings in Appendix D, and complete configurations in the accompanying code.

## 5.3 Main results

Table 2 shows that dense CAFE+FNO achieves the lowest mean error on Darcy, NS, Burgers, and Airfoil. Relative to SirenFNO, errors decrease by 43.7%, 60.2%, 29.5%, and 18.4%, respectively. On ReacDif, SirenFNO performs better (2.36 versus $3 . 8 0 , \times 1 0 ^ { - 3 } )$ . CAFE+FNO also improves on FNO and AM-FNO across all five datasets, although AM-FNO’s ReacDif results show substantial variability.

Relative to dense CAFE+FNO, CP, TT, and Tucker variants reduce parameter counts by 76.7– 86.1%, 68.7–74.7%, and 0.5–78.0%, respectively (Table 6). Factorized CAFE+FNO variants have higher mean errors than the dense model, except for Tucker on ReacDif. All three outperform their SirenFNO counterparts on Darcy; results elsewhere are mixed. To target comparable parameter budgets, we use a 12-dimensional input to each CAFE+ factor-generator MLP, compared with 16 or 32 in the corresponding SirenFNO models. Relative to dense CAFE+FNO, both the initial basis counts and the MLP input dimension are reduced. These changes may also afect accuracy; their efects are not isolated from tensor factorization in this comparison.

Table 3: CAFE+FNO ablations: relative $L _ { 2 }$ errors $( \times 1 0 ^ { - 3 } )$ . Evaluation and reporting conventions follow Table 2. Bold marks the lowest unrounded mean per dataset. <sup>∗</sup>: no composition block, with the kernel MLP input dimension matched to the default model (16 for ReacDif and 32 otherwise). RFF/Cheb: single-basis variants retaining the composition block. <sup>†</sup>: learnable bandwidth with a fixed Gaussian matrix.
<table><tr><td>Model</td><td>Darcy  $1 2 8 \times 1 2 8$ </td><td>NS  $1 2 8 \times 1 2 8$ </td><td>Burgers 1024</td><td>Airfoil  $2 2 1 \times 5 1$ </td><td>ReacDiff 1024</td></tr><tr><td>CAFE+FNO</td><td> ${ \bf 1 3 . 6 3 \pm 0 . 1 2 }$ </td><td> $1 7 . 2 5 \pm 0 . 2 9$ </td><td> ${ \bf 4 . 1 1 \pm 0 . 3 7 }$ </td><td> ${ \bf 4 . 6 9 \pm 0 . 1 0 }$ </td><td> $3 . 8 0 \pm 0 . 2 7$ </td></tr><tr><td> $\mathrm { C A F E { + } F N O ^ { * } }$ </td><td> $2 7 . 2 8 \pm 2 . 7 0$ </td><td> $3 8 . 5 3 \pm 1 . 4 6$ </td><td> $5 . 1 4 \pm 0 . 3 4$ </td><td> $1 7 . 5 8 \pm 1 1 . 3 8$ </td><td> $3 . 8 7 \pm 0 . 2 8$ </td></tr><tr><td> $\mathrm { C A F E { + } F N O _ { R F F } }$ </td><td> $1 5 . 0 7 \pm 0 . 7 1$ </td><td> $1 9 . 7 6 \pm 0 . 2 0$ </td><td> $4 . 9 1 \pm 0 . 3 9$ </td><td> $5 . 3 6 \pm 0 . 2 2$ </td><td> $4 . 4 8 \pm 1 . 1 4$ </td></tr><tr><td> $\mathrm { C A F E { + } F N O _ { C h e b } }$ </td><td> $1 4 . 4 6 \pm 0 . 7 1$ </td><td> $1 7 . 8 1 \pm 0 . 1 4$ </td><td> $4 . 1 6 \pm 0 . 2 4$ </td><td> $4 . 8 7 \pm 0 . 3 0$ </td><td> $4 . 0 0 \pm 0 . 2 4$ </td></tr><tr><td> $\mathrm { C A F E { + } F N O ^ { \dagger } }$ </td><td> $1 3 . 6 9 \pm 0 . 1 4$ </td><td> ${ \bf 1 7 . 2 2 \pm 0 . 2 6 }$ </td><td> $4 . 1 3 \pm 0 . 3 7$ </td><td> $4 . 6 9 \pm 0 . 1 2$ </td><td> ${ \bf 3 . 6 5 \pm 0 . 2 9 }$ </td></tr></table>

![](images/f5c313c4572cc4cd193c59ee10edddee20788d3f44c5963e97126be6d1669644.jpg)  
Figure 2: Comparison of FNO, AM-FNO, SirenFNO, and CAFE+FNO on a Navier–Stokes (NS) sample using trained checkpoints. The top row shows the ground-truth vorticity and model predictions; the bottom row shows pointwise absolute errors. Relative $L _ { 2 }$ errors are reported below the predictions.

## 5.4 Ablation experiments

We analyze how the combination of Fourier and Chebyshev features, multiplicative feature composition through linear branches and a Hadamard product, and RFF bandwidth learnability afect prediction accuracy. Results are presented in Table 3. Since removing components may change feature dimensions and parameter counts, each comparison is interpreted as the empirical efect of the corresponding overall configuration.

The combined Fourier–Chebyshev encoding yields lower mean errors than either single-basis variant on all five datasets, supporting its efectiveness for Fourier kernel generation.

With matched kernel MLP input dimensions (16 for ReacDif and 32 otherwise), the default model achieves lower mean errors than CAFE+FNO<sup>∗</sup>, which directly uses Fourier–Chebyshev features. This supports the utility of CAFE+ encoding at the same MLP input dimension. Finally, $\mathrm { C A F E { + } F N O ^ { \dagger } }$ , with learnable RFF bandwidth $\rho _ { \ell } ,$ yields lower mean errors than the fixedbandwidth default on NS and ReacDif, but higher errors on Darcy and Burgers. Airfoil errors are equal at the reported precision.

## 6 Conclusion

We proposed CAFE+FNO, which introduces explicit, learnable feature composition into Fourier kernel generation. Its Fourier–Chebyshev encoding uses parallel afine branches and a Hadamard product to construct interactions within and across feature families before the kernel MLP. The dense model achieves the lowest mean error on four of five benchmarks, while SirenFNO performs better on ReacDif. Ablations favor the combined basis over either basis alone and show lower mean errors with composition than with direct features at matched kernel MLP input dimensions. Together, these results support explicit spectral-coordinate feature composition as an efective design choice for Fourier kernel generation. Factorized variants use fewer parameters than dense CAFE+FNO, generally at some cost in accuracy. However, this comparison does not isolate tensor factorization from the accompanying reductions in basis counts and MLP input dimensions. Future work will focus on extending CAFE+FNO to a broader range of PDE problems and higherdimensional spatial domains.

## References

[1] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier Neural Operator for Parametric Partial Diferential Equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=c8P9NQVtmnO.

[2] Robert Joseph George, Jiawei Zhao, Jean Kossaifi, Zongyi Li, and Anima Anandkumar. Incremental Spatial and Spectral Learning of Neural Operators for Solving Large-Scale PDEs. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum? id=xI6cPQObp0.

[3] Zipeng Xiao, Siqi Kou, Zhongkai Hao, Bokai Lin, and Zhijie Deng. Amortized Fourier Neural Operators. In Advances in Neural Information Processing Systems, volume 37, pages 115001– 115020, 2024. doi:10.52202/079017-3651.

[4] Pengqing Shi, Jie Yin, Stephen Tierney, and Junbin Gao. SirenFNO: Eficient and Full Frequency Learning of Fourier Neural Operators. In Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, pages 4804–4812, 2026. doi:10.24963/ijcai.2026/535.

[5] Junbo Ke, Yangyang Xu, Chao Wang, and You-Wei Wen. Content-Aware Frequency Encoding for Implicit Neural Representations with Fourier-Chebyshev Features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3646–3655, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Ke\_ Content-Aware\_Frequency\_Encoding\_for\_Implicit\_Neural\_Representations\_with\_ Fourier-Chebyshev\_Features\_CVPR\_2026\_paper.html.

[6] Kaushik Bhattacharya, Bamdad Hosseini, Nikola B. Kovachki, and Andrew M. Stuart. Model Reduction and Neural Networks for Parametric PDEs. The SMAI Journal of Computational Mathematics, 7:121–157, 2021. doi:10.5802/smai-jcm.74.

[7] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural Operator: Learning Maps Between Function Spaces. arXiv preprint arXiv:2108.08481v6, 2024. URL https://arxiv.org/abs/2108. 08481v6.

[8] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, 2021. doi:10.1038/s42256-021-00302-5.

[9] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural Operator: Graph Kernel Network for Partial Diferential Equations. arXiv preprint arXiv:2003.03485, 2020. URL https://arxiv.org/ abs/2003.03485.

[10] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Multipole Graph Neural Operator for Parametric Partial Diferential Equations. In Advances in Neural Information Processing Systems, volume 33, pages 6755–6766, 2020. URL https://proceedings.neurips.cc/paper\_files/ paper/2020/file/4b21cf96d4cf612f239a6c322b10c8fe-Paper.pdf.

[11] Shuhao Cao. Choose a Transformer: Fourier or Galerkin. In Advances in Neural Information Processing Systems, volume 34, pages 24924–24940, 2021. URL https://proceedings. neurips.cc/paper/2021/hash/d0921d442ee91b896ad95059d13df618-Abstract.html.

[12] Zijie Li, Kazem Meidani, and Amir Barati Farimani. Transformer for Partial Diferential Equations’ Operator Learning. Transactions on Machine Learning Research, 2023. URL https://openreview.net/forum?id=EPPqt3uERT.

[13] Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, and Jun Zhu. GNOT: A General Neural Operator Transformer for Operator Learning. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 12556–12569, 2023. URL https://proceedings.mlr.press/v202/ hao23c.html.

[14] Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A Fast Transformer Solver for PDEs on General Geometries. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 53681–53705, 2024. URL https://proceedings.mlr.press/v235/wu24r.html.

[15] Md Ashiqur Rahman, Zachary E. Ross, and Kamyar Azizzadenesheli. U-NO: U-shaped Neural Operators. Transactions on Machine Learning Research, 2023. URL https://openreview. net/forum?id=j3oQF9coJd.

[16] Gege Wen, Zongyi Li, Kamyar Azizzadenesheli, Anima Anandkumar, and Sally M. Benson. U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow. Advances in Water Resources, 163:104180, 2022. doi:10.1016/j.advwatres.2022.104180.

[17] Alasdair Tran, Alexander Mathews, Lexing Xie, and Cheng Soon Ong. Factorized Fourier Neural Operators. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=tmIiMPl4IPa.

[18] Jean Kossaifi, Nikola B. Kovachki, Kamyar Azizzadenesheli, and Anima Anandkumar. Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=AWiDlO63bH.

[19] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljaˇci´c, Thomas Y. Hou, and Max Tegmark. KAN: Kolmogorov–Arnold Networks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=Ozo7qJ5vZi.

[20] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the Spectral Bias of Neural Networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97, pages 5301–5310, 2019. URL https://proceedings.mlr.press/v97/rahaman19a.html.

[21] Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, and Zheng Ma. Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks. arXiv preprint arXiv:1901.06523v7, 2024. URL https://arxiv.org/abs/1901.06523v7.

[22] Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter. Multiplicative Filter Networks. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=OmtmcPkkhT.

[23] David B. Lindell, Dave Van Veen, Jeong Joon Park, and Gordon Wetzstein. BACON: Band-Limited Coordinate Networks for Multiscale Scene Representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16252–16262, 2022. URL https://openaccess.thecvf.com/content/CVPR2022/html/Lindell\_BACON\_ Band-Limited\_Coordinate\_Networks\_for\_Multiscale\_Scene\_Representation\_CVPR\_ 2022\_paper.html.

[24] Shayan Shekarforoush, David B. Lindell, David J. Fleet, and Marcus A. Brubaker. Residual Multiplicative Filter Networks for Multiscale Reconstruction. In Advances in Neural Information Processing Systems, volume 35, pages 8550–8563, 2022. doi:10.52202/068431-0622.

[25] Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit Neural Representations with Periodic Activation Functions. In Advances in Neural Information Processing Systems, volume 33, pages 7462–7473, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 53c04118df112c13a8c34b38343b9c10-Abstract.html.

[26] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. In Advances in Neural Information Processing Systems, volume 33, pages 7537–7547, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 55053683268957697aa39fba6f231c68-Abstract.html.

[27] Amir Hertz, Or Perel, Raja Giryes, Olga Sorkine-Hornung, and Daniel Cohen-Or. SAPE: Spatially-Adaptive Progressive Encoding for Neural Optimization. In Advances in Neural Information Processing Systems, volume 34, pages 8820– 8832, 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/hash/ 4a06d868d044c50af0cf9bc82d2fc19f-Abstract.html.

[28] Nuri Benbarka, Timon H¨ofer, Hamd ul-Moqeet Riaz, and Andreas Zell. Seeing Implicit Neural Representations as Fourier Series. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2041–2050, 2022. URL https://openaccess.thecvf.com/content/WACV2022/html/Benbarka\_Seeing\_ Implicit\_Neural\_Representations\_As\_Fourier\_Series\_WACV\_2022\_paper.html.

[29] Mingze Ma, Qingtian Zhu, Yifan Zhan, Zhengwei Yin, Hongjun Wang, and Yinqiang Zheng. Robustifying Fourier Features Embeddings for Implicit Neural Representations. arXiv preprint arXiv:2502.05482, 2025. URL https://arxiv.org/abs/2502.05482.

[30] Gizem Y¨uce, Guillermo Ortiz-Jim´enez, Beril Besbinar, and Pascal Frossard. A Structured Dictionary Perspective on Implicit Neural Representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19228–19238, 2022. URL https: //openaccess.thecvf.com/content/CVPR2022/html/Yuce\_A\_Structured\_Dictionary\_ Perspective\_on\_Implicit\_Neural\_Representations\_CVPR\_2022\_paper.html.

[31] Nicholas H. Nelsen and Andrew M. Stuart. The Random Feature Model for Input-Output Maps between Banach Spaces. SIAM Journal on Scientific Computing, 43(5):A3212–A3243, 2021. doi:10.1137/20M133957X.

[32] NeuralOperator Team. Darcy Flow Dataset. Zenodo, 2021. URL https://doi.org/10. 5281/zenodo.12784353. Dataset, version v2.

[33] NeuralOperator Team. Navier-Stokes Dataset. Zenodo, 2021. URL https://doi.org/10. 5281/zenodo.12825163. Dataset, version v2.

[34] Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pfl¨uger, and Mathias Niepert. PDEBench: An Extensive Benchmark for Scientific Machine Learning. arXiv preprint arXiv:2210.07182v7, 2024. URL https://arxiv.org/abs/ 2210.07182v7.

[35] Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pfl¨uger, and Mathias Niepert. PDEBench Datasets. DaRUS, 2022. URL https://doi. org/10.18419/DARUS-2986. Dataset.

[36] Zongyi Li, Daniel Zhengyu Huang, Burigede Liu, and Anima Anandkumar. Fourier Neural Operator with Learned Deformations for PDEs on General Geometries. arXiv preprint arXiv:2207.05209v2, 2024. URL https://arxiv.org/abs/2207.05209v2. Code and datasets: https://github.com/neuraloperator/Geo-FNO#datasets.

[37] Tamara G. Kolda and Brett W. Bader. Tensor Decompositions and Applications. SIAM Review, 51(3):455–500, 2009. doi:10.1137/07070111X.

[38] I. V. Oseledets. Tensor-Train Decomposition. SIAM Journal on Scientific Computing, 33(5): 2295–2317, 2011. doi:10.1137/090752286.

## A Data Preprocessing, Training, and Evaluation

## A.1 Data Splits and Preprocessing

The data sources, splits, and input–output definitions follow Table 1 in the main text. For Darcy, only the outputs are standardized using training-set statistics. For Navier–Stokes, the inputs and outputs are standardized separately using their respective training-set statistics. In both cases, the output normalization is inverted for evaluation. For Burgers, the global mean and population standard deviation are computed over all temporal and spatial values of 200 training trajectories selected separately for each seed. These statistics are used to normalize both inputs and outputs, and errors are computed in the normalized space.

Navier–Stokes. For Navier–Stokes, we use the input–output pairs in nsforcing train 128.pt and nsforcing test 128.pt provided by NeuralOperator. Each input and output is a singlechannel vorticity field at a diferent time, and the corresponding output vorticity field is predicted directly from the input vorticity field. We do not use autoregressive rollout with multiple input snapshots in this experiment.

Airfoil. For Airfoil, the X, Y coordinates of a 221 × 51 structured mesh are used as inputs to predict the Mach number at each grid point. The prediction target is the fifth component of the original flow-field array. No additional value normalization is applied to the input coordinates or output values.

Burgers and ReacDif. For both benchmarks, we use the first 20 snapshots of each trajectory without temporal or spatial subsampling. Denoting the stored time points in the original data by $t _ { j } ,$ the 10 snapshots at $t _ { 0 } , \ldots , t _ { 9 }$ are used as inputs, and the 10 snapshots at $t _ { 1 0 } , \ldots , t _ { 1 9 }$ are predicted autoregressively. Thus, the input interval is $[ t _ { 0 } , t _ { 9 } ]$ and the prediction interval is $[ t _ { 1 0 } , t _ { 1 9 } ]$ with the time step determined by the storage interval of the original data.

## A.2 Training and Evaluation Settings

Table 4: Training and evaluation settings.
<table><tr><td>Dataset</td><td>Batch size</td><td>Prediction protocol</td><td>Evaluation space</td></tr><tr><td>Darcy</td><td>32</td><td>Direct prediction of the provided input-output field pairs</td><td>After output denormalization</td></tr><tr><td>Navier-Stokes</td><td>32</td><td>Direct prediction of the provided single input-output field pairs</td><td>After output denormalization</td></tr><tr><td>Burgers</td><td>32</td><td>Autoregressive prediction of 10 future snapshots from 10 input snapshots</td><td>Normalized space</td></tr><tr><td>Airfoil</td><td>8</td><td>Direct prediction of the target field from coordinates</td><td>Original value space</td></tr><tr><td>ReacDiff</td><td>32</td><td>Autoregressive prediction of 10 future snapshots from 10 input snapshots</td><td>Original value space</td></tr></table>

Training loss. For Darcy, Navier–Stokes, and Airfoil, training uses sample-wise relative $L _ { 2 }$ errors on the directly predicted target fields. For Burgers and ReacDif, the training loss is the sum of the relative $L _ { 2 }$ errors over the 10 prediction steps. At each step, the per-sample relative errors are summed over the batch for Burgers and averaged over the batch for ReacDif. For both benchmarks, previous predictions are incorporated into the input to the next step, and the computational graph is retained over the entire rollout for backpropagation during training. The

Burgers loss is computed in the normalized space, whereas the ReacDif loss is computed without value normalization.

## B Frequency Coordinate Normalization

We use the Fourier mode k and normalized spectral coordinate ${ \xi } _ { k }$ defined in the main text. Let $L _ { a }$ denote the length of the ath FFT axis after padding. The one-dimensional rFFT in the dense model stores nonnegative Fourier modes, with normalized coordinates given by

$$
\pmb { \xi } _ { k } = \frac { 2 k } { L _ { 1 } } , \qquad k = 0 , \dots , \left\lfloor \frac { L _ { 1 } } { 2 } \right\rfloor .\tag{23}
$$

Let $p = 0 , \ldots , L _ { 1 } - 1$ and $q = 0 , \ldots , \lfloor L _ { 2 } / 2 \rfloor$ denote the integer storage indices of a twodimensional rFFT. The components of the Fourier mode $\boldsymbol { k } = \left( k _ { 1 } , k _ { 2 } \right)$ corresponding to storage position $( p , q )$ are defined as

$$
k _ { 1 } = \left\{ p , \begin{array} { l l } { { p , } } & { { 0 \leq p \leq \lfloor ( L _ { 1 } - 1 ) / 2 \rfloor , } } \\ { { p - L _ { 1 } , } } & { { \lfloor ( L _ { 1 } - 1 ) / 2 \rfloor < p < L _ { 1 } , } } \end{array} \right. \quad k _ { 2 } = q .\tag{24}
$$

The first axis is therefore converted to signed frequency indices, while the last axis uses nonnegative frequency indices. The normalized coordinates are then given by

$$
\pmb { \xi } _ { k } = \left( \frac { 2 k _ { 1 } } { L _ { 1 } } , \frac { 2 k _ { 2 } } { L _ { 2 } } \right) ^ { \top } .\tag{25}
$$

For the $\mathrm { C P , ~ T T , }$ and Tucker models, midpoint coordinates are used for each input-component, output-component, and frequency-storage axis. For a factor axis of length I with storage index $\nu ,$ the input coordinate to its factor generator is

$$
s _ { I } ( \nu ) = \frac { 2 ( \nu + 1 / 2 ) } { I } - 1 , \qquad \nu = 0 , \dots , I - 1 .\tag{26}
$$

Even on a frequency axis, ν follows the storage order and is therefore distinct from the signed frequency index used by the dense model. Hermitian projection is applied according to the rFFT storage convention. In one dimension, the kernels at the DC component and, for an even axis length, the Nyquist component are made real. In two dimensions, for the DC column of the last frequency axis and its Nyquist column when that axis has even length, each kernel along the first axis is averaged with the complex conjugate of the kernel at the opposite frequency. The remaining stored modes are unchanged, and the unstored modes are determined by conjugate symmetry. This operation is applied elementwise to each kernel matrix, without transposing the input- and output-component axes.

## C CP, TT, and Tucker Tensor Decompositions

Following the decomposition structures in Eqs. (9)–(11) of Section 4.2 of SirenFNO [4], we apply CAFE+ encoding to the factor generator for each axis. Let $d _ { s }$ denote the spatial dimension, and arrange the $n _ { \mathrm { a x } } = d _ { s } + 2$ axes in the order of the input components, the $d _ { s }$ frequency axes, and the output components. The real and imaginary parts, indexed by $c \in \{ \mathrm { R } , \mathrm { I } \}$ , are generated independently and combined as $U = U ^ { ( \mathrm { R } ) } + i U ^ { \bigcup }$

$$
\mathrm { C P } \colon \ \boldsymbol { U } ^ { ( c ) } = \sum _ { \alpha = 1 } ^ { r } \mathbf { f } _ { 1 , \alpha } ^ { ( c ) } \otimes \cdot \cdot \cdot \otimes \mathbf { f } _ { n _ { \mathrm { a x } } , \alpha } ^ { ( c ) } ,\tag{27}
$$

$$
\mathrm { T T } \colon U _ { \nu _ { 1 } , \dots , \nu _ { n _ { \mathrm { a x } } } } ^ { ( c ) } = A _ { 1 } ^ { ( c ) } ( \nu _ { 1 } ) \cdot \cdot \cdot A _ { n _ { \mathrm { a x } } } ^ { ( c ) } ( \nu _ { n _ { \mathrm { a x } } } ) ,\tag{28}
$$

$$
\mathrm { T u c k e r : } \quad U ^ { ( c ) } = \mathcal { C } ^ { ( c ) } \times _ { 1 } F _ { 1 } ^ { ( c ) } \cdot \cdot \cdot \times _ { n _ { \mathrm { a x } } } F _ { n _ { \mathrm { a x } } } ^ { ( c ) } .\tag{29}
$$

For tensor-axis lengths $I _ { a }$ , we have $U ^ { ( c ) } \in \mathbb { R } ^ { I _ { 1 } \times \cdots \times I _ { n _ { \mathrm { a x } } } }$ . The CP factors satisfy $\mathbf { f } _ { a , \alpha } ^ { ( c ) } \in \mathbb { R } ^ { I _ { a } }$ , the Tucker factor matrices satisfy $F _ { a } ^ { ( c ) } \in \mathbb { R } ^ { I _ { a } \times r }$ , and the trainable Tucker core satisfies $\mathcal { C } ^ { ( c ) } \in \mathbb { R } ^ { r \times \cdots \times r }$ Each TT core slice has dimensions

$$
A _ { a } ^ { ( c ) } ( \nu _ { a } ) \in \mathbb { R } ^ { r _ { a - 1 } \times r _ { a } } , \qquad r _ { 0 } = r _ { n _ { \mathrm { a x } } } = 1 , \qquad r _ { a } = r \quad ( 1 \leq a < n _ { \mathrm { a x } } ) .\tag{30}
$$

Here, ⊗ and $\times _ { a }$ denote the outer product and the mode-a tensor–matrix product, respectively. CP represents a tensor as a sum of rank-one tensors, whereas Tucker combines a core tensor with axis-specific factor matrices [37]. In this work, CP uses r components, and Tucker uses rank r along each axis with a core containing $r ^ { n _ { \mathrm { a x } } }$ entries. TT represents a tensor as a sequential product of cores [38]; we set both boundary ranks to 1 and all internal ranks to r. The CP and Tucker generators output r values per coordinate, while TT outputs r values at the two boundary axes and $r ^ { 2 }$ values at internal axes. These ranks specify the decompositions of the individual real-valued tensors and do not denote the exact rank of the final complex-valued kernel.

An independent CAFE+ factor generator is used for each axis and each real or imaginary component. For CP and Tucker, the midpoint coordinate $s _ { I _ { a } } ( \nu _ { a } )$ from Appendix B is supplied as input, and the generator outputs r values corresponding to one row of the factor matrix. For TT, the generator similarly outputs $r _ { a - 1 } r _ { a }$ values, which are reshaped into a core slice. The reconstructed tensor, whose axes are ordered as input components, frequency-storage axes, and output components, is rearranged into an output–input matrix for each mode and used for spectral mixing.

## D Model Settings

CAFE+FNO uses a hidden channel width of 32, four operator layers, and two linear branches. The dense model uses $m = 3 2$ Fourier frequency vectors in $\mathbb { R } ^ { d _ { s } }$ and $n = 3 2$ Chebyshev polynomials along each coordinate axis, with a kernel MLP containing one hidden layer. The dataset-specific branch output dimensions, kernel MLP hidden widths, and decomposition ranks are listed in Table 5.

Each factor generator in the decomposed models uses 16 Fourier frequency vectors and eight Chebyshev polynomials, with a branch output dimension of 12. The factor MLP hidden widths are 16, 24, and 16 for CP, TT, and Tucker, respectively. For the factorized SirenFNO baselines, the backbone input dimension is 16 for all ReacDif variants and for Tucker on NS, and 32 otherwise.

Table 5: Dataset-specific architectural settings for CAFE+FNO. The branch output dimension and MLP hidden width refer to the dense model; the common settings for the decomposed models are described in the text.
<table><tr><td>Dataset</td><td>Dense branch output dimension</td><td>Dense MLP hidden width</td><td>CP rank</td><td>TT rank</td><td>Tucker per-axis rank</td></tr><tr><td>Darcy</td><td>32</td><td>32</td><td>8</td><td>8</td><td>10</td></tr><tr><td>Navier-Stokes</td><td>32</td><td>64</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Burgers</td><td>32</td><td>32</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Airfoil</td><td>32</td><td>32</td><td>8</td><td>8</td><td>8</td></tr><tr><td>ReacDiff</td><td>16</td><td>32</td><td>8</td><td>8</td><td>8</td></tr></table>

## E Per-Seed Test Errors

Tables 7–11 provide the individual-seed test relative $L _ { 2 }$ errors for the models and datasets compared in Table 2, using random seeds {0, 42, 73, 108, 202}. Each result is obtained from the final-epoch model after 500 training epochs, following the same evaluation protocols as the main comparison. Errors are reported in scientific notation with three decimal places in the mantissa.

Table 6: Trainable parameter counts in real scalars, including lifting, operator layers, and projection. Each complex parameter counts as two real scalars; shared parameters are counted once. Fixed bufers are excluded, and kernel-generator outputs are not counted as additional parameters. The dash denotes an omitted model–dataset combination.
<table><tr><td>Model</td><td>Darcy</td><td>NS</td><td>Burgers</td><td>Airfoil</td><td>ReacDiff</td></tr><tr><td>Resolution</td><td>128 × 128</td><td>128 × 128</td><td>1024</td><td>221 × 51</td><td>1024</td></tr><tr><td>FNO</td><td>1,192,801</td><td>4,469,601</td><td>4,216,161</td><td>2,372,513</td><td>4,216,161</td></tr><tr><td>U-FNO</td><td>5,931,201</td><td>990,753</td><td>1,883,073</td><td></td><td>1,892,161</td></tr><tr><td>TFNO-CP</td><td>72,913</td><td>237,505</td><td>226,369</td><td>131,993</td><td>226,369</td></tr><tr><td>AM-FNO (MLP)</td><td>385,473</td><td>4,443,137</td><td>823,073</td><td>1,136,673</td><td>823,073</td></tr><tr><td>SirenFNO</td><td>308,865</td><td>579,457</td><td>308,993</td><td>308,897</td><td>304,833</td></tr><tr><td>CAFE+FNO</td><td>345,601</td><td>611,969</td><td>337,665</td><td>345,633</td><td>323,201</td></tr><tr><td>CP-SirenFNO</td><td>80,513</td><td>138,881</td><td>70,145</td><td>80,545</td><td>57,665</td></tr><tr><td>TT-SirenFNO</td><td>109,185</td><td>211,585</td><td>84,481</td><td>109,217</td><td>72,001</td></tr><tr><td>Tucker-SirenFNO</td><td>162,561</td><td>596,353</td><td>74,241</td><td>113,313</td><td>61,761</td></tr><tr><td>CAFE+FNO (CP)</td><td>80,517</td><td>84,869</td><td>70,149</td><td>80,549</td><td>70,149</td></tr><tr><td>CAFE+FNO (TT)</td><td>108,293</td><td>188,293</td><td>85,381</td><td>108,325</td><td>85,381</td></tr><tr><td>CAFE+FNO (Tucker)</td><td>161,605</td><td>609,157</td><td>74,245</td><td>113,317</td><td>74,245</td></tr></table>

Table 7: Test relative $L _ { 2 }$ errors for seed 0 after 500 training epochs. The dash denotes an omitted model–dataset combination.
<table><tr><td>Model</td><td>Darcy</td><td>NS</td><td>Burgers</td><td>Airfoil</td><td>ReacDiff</td></tr><tr><td>Resolution</td><td>128×128</td><td>128×128</td><td>1024</td><td>221×51</td><td>1024</td></tr><tr><td>FNO</td><td>6.200e-2</td><td>5.610e-2</td><td>1.135e-2</td><td>6.047e-3</td><td>4.252e-3</td></tr><tr><td>U-FNO</td><td>4.790e-2</td><td>5.824e-2</td><td>1.005e-2</td><td></td><td>7.578e-3</td></tr><tr><td>TFNO-CP</td><td>4.928e-2</td><td>3.394e-2</td><td>9.423e-3</td><td>7.026e-3</td><td>4.185e-3</td></tr><tr><td>AM-FNO (MLP)</td><td>7.497e-2</td><td>3.113e-2</td><td>2.338e-2</td><td>6.519e-3</td><td>1.001e-2</td></tr><tr><td>SirenFNO</td><td>2.587e-2</td><td>4.460e-2</td><td>5.392e-3</td><td>5.691e-3</td><td>2.148e-3</td></tr><tr><td>CAFE+FNO</td><td>1.384e-2</td><td>1.756e-2</td><td>4.205e-3</td><td>4.707e-3</td><td>3.843e-3</td></tr><tr><td>CP-SirenFNO</td><td>3.091e-2</td><td>4.714e-2</td><td>8.479e-3</td><td>6.910e-3</td><td>2.880e-3</td></tr><tr><td>TT-SirenFNO</td><td>1.428e-1</td><td>3.924e-2</td><td>7.403e-3</td><td>5.902e-3</td><td>2.531e-3</td></tr><tr><td>Tucker-SirenFNO</td><td>3.744e-2</td><td>4.890e-2</td><td>8.293e-3</td><td>7.876e-3</td><td>2.992e-3</td></tr><tr><td>CAFE+FNO (CP)</td><td>3.025e-2</td><td>9.429e-2</td><td>8.656e-3</td><td>8.307e-3</td><td>4.243e-3</td></tr><tr><td>CAFE+FNO (TT)</td><td>2.652e-2</td><td>6.126e-2</td><td>8.234e-3</td><td>7.194e-3</td><td>3.988e-3</td></tr><tr><td>CAFE+FNO (Tucker)</td><td>2.781e-2</td><td>4.893e-2</td><td>8.911e-3</td><td>6.724e-3</td><td>4.077e-3</td></tr></table>

Table 8: Test relative $L _ { 2 }$ errors for seed 42 after 500 training epochs. The dash denotes an omitted model–dataset combination.
<table><tr><td>Model</td><td>Darcy</td><td>NS</td><td>Burgers</td><td>Airfoil</td><td>ReacDiff</td></tr><tr><td>Resolution</td><td>128×128</td><td>128×128</td><td>1024</td><td>221×51</td><td>1024</td></tr><tr><td>FNO</td><td>7.201e-2</td><td>5.585e-2</td><td>1.121e-2</td><td>6.130e-3</td><td>4.125e-3</td></tr><tr><td>U-FNO</td><td>4.904e-2</td><td>5.822e-2</td><td>9.668e-3</td><td></td><td>4.914e-3</td></tr><tr><td>TFNO-CP</td><td>4.734e-2</td><td>3.525e-2</td><td>1.054e-2</td><td>6.670e-3</td><td>3.930e-3</td></tr><tr><td>AM-FNO (MLP)</td><td>4.344e-2</td><td>2.858e-2</td><td>2.010e-2</td><td>5.792e-3</td><td>1.052e-1</td></tr><tr><td>SirenFNO</td><td>2.285e-2</td><td>4.578e-2</td><td>5.617e-3</td><td>6.068e-3</td><td>2.384e-3</td></tr><tr><td>CAFE+FNO</td><td>1.382e-2</td><td>1.695e-2</td><td>3.718e-3</td><td>4.582e-3</td><td>3.763e-3</td></tr><tr><td>CP-SirenFNO</td><td>3.122e-2</td><td>5.328e-2</td><td>8.780e-3</td><td>9.428e-3</td><td>7.126e-3</td></tr><tr><td>TT-SirenFNO</td><td>2.904e-2</td><td>4.263e-2</td><td>7.820e-3</td><td>6.409e-3</td><td>2.697e-3</td></tr><tr><td>Tucker-SirenFNO</td><td>5.288e-2</td><td>3.904e-2</td><td>8.214e-3</td><td>6.280e-3</td><td>2.784e-3</td></tr><tr><td>CAFE+FNO (CP)</td><td>3.373e-2</td><td>4.288e-2</td><td>8.076e-3</td><td>7.653e-3</td><td>4.391e-3</td></tr><tr><td>CAFE+FNO (TT)</td><td>2.777e-2</td><td>3.364e-2</td><td>8.554e-3</td><td>6.576e-3</td><td>4.449e-3</td></tr><tr><td>CAFE+FNO (Tucker)</td><td>3.034e-2</td><td>3.540e-2</td><td>8.844e-3</td><td>7.880e-3</td><td>4.367e-3</td></tr></table>

Table 9: Test relative $L _ { 2 }$ errors for seed 73 after 500 training epochs. The dash denotes an omitted model–dataset combination.
<table><tr><td>Model</td><td>Darcy</td><td>NS</td><td>Burgers</td><td>Airfoil</td><td>ReacDiff</td></tr><tr><td>Resolution</td><td>128×128</td><td>128×128</td><td>1024</td><td>221×51</td><td>1024</td></tr><tr><td>FNO</td><td>6.389e-2</td><td>5.573e-2</td><td>1.134e-2</td><td>6.310e-3</td><td>3.944e-3</td></tr><tr><td>U-FNO</td><td>4.521e-2</td><td>5.775e-2</td><td>1.011e-2</td><td></td><td>1.927e-2</td></tr><tr><td>TFNO-CP</td><td>4.770e-2</td><td>3.488e-2</td><td>1.030e-2</td><td>6.828e-3</td><td>3.715e-3</td></tr><tr><td>AM-FNO (MLP)</td><td>6.956e-2</td><td>2.806e-2</td><td>1.490e-2</td><td>7.247e-3</td><td>8.633e0</td></tr><tr><td>SirenFNO</td><td>2.517e-2</td><td>4.055e-2</td><td>6.347e-3</td><td>5.532e-3</td><td>3.682e-3</td></tr><tr><td>CAFE+FNO</td><td>1.362e-2</td><td>1.694e-2</td><td>4.514e-3</td><td>4.785e-3</td><td>4.203e-3</td></tr><tr><td>CP-SirenFNO</td><td>4.091e-2</td><td>5.295e-2</td><td>9.258e-3</td><td>7.934e-3</td><td>3.116e-3</td></tr><tr><td>TT-SirenFNO</td><td>4.308e-2</td><td>3.731e-2</td><td>7.753e-3</td><td>5.915e-3</td><td>3.006e-3</td></tr><tr><td>Tucker-SirenFNO</td><td>7.154e-2</td><td>4.141e-2</td><td>8.643e-3</td><td>1.482e-2</td><td>3.007e-3</td></tr><tr><td>CAFE+FNO (CP)</td><td>2.715e-2</td><td>5.776e-2</td><td>8.819e-3</td><td>7.006e-3</td><td>3.579e-3</td></tr><tr><td>CAFE+FNO (TT)</td><td>2.944e-2</td><td>5.075e-2</td><td>8.419e-3</td><td>8.021e-3</td><td>3.453e-3</td></tr><tr><td>CAFE+FNO (Tucker)</td><td>2.702e-2</td><td>3.660e-2</td><td>9.431e-3</td><td>9.674e-3</td><td>3.735e-3</td></tr></table>

Table 10: Test relative $L _ { 2 }$ errors for seed 108 after 500 training epochs. The dash denotes an omitted model–dataset combination.
<table><tr><td>Model</td><td>Darcy</td><td>NS</td><td>Burgers</td><td>Airfoil</td><td>ReacDiff</td></tr><tr><td>Resolution</td><td>128×128</td><td>128×128</td><td>1024</td><td>221×51</td><td>1024</td></tr><tr><td>FNO</td><td>6.690e-2</td><td>5.584e-2</td><td>1.122e-2</td><td>6.000e-3</td><td>4.239e-3</td></tr><tr><td>U-FNO</td><td>4.812e-2</td><td>5.636e-2</td><td>1.008e-2</td><td></td><td>7.520e-3</td></tr><tr><td>TFNO-CP</td><td>4.607e-2</td><td>3.504e-2</td><td>1.046e-2</td><td>7.891e-3</td><td>4.407e-3</td></tr><tr><td>AM-FNO (MLP)</td><td>5.421e-2</td><td>3.245e-2</td><td>1.719e-2</td><td>5.511e-3</td><td>1.133e-2</td></tr><tr><td>SirenFNO</td><td>2.442e-2</td><td>4.266e-2</td><td>5.710e-3</td><td>5.936e-3</td><td>1.789e-3</td></tr><tr><td>CAFE+FNO</td><td>1.353e-2</td><td>1.753e-2</td><td>3.760e-3</td><td>4.588e-3</td><td>3.452e-3</td></tr><tr><td>CP-SirenFNO</td><td>4.386e-2</td><td>4.228e-2</td><td>9.003e-3</td><td>6.144e-3</td><td>4.083e-3</td></tr><tr><td>TT-SirenFNO</td><td>6.327e-1</td><td>3.225e-2</td><td>7.656e-3</td><td>5.744e-3</td><td>2.533e-3</td></tr><tr><td>Tucker-SirenFNO</td><td>1.073e-1</td><td>6.707e-2</td><td>7.504e-3</td><td>5.547e-3</td><td>2.928e-3</td></tr><tr><td>CAFE+FNO (CP)</td><td>3.098e-2</td><td>1.011e-1</td><td>8.024e-3</td><td>1.312e-2</td><td>4.767e-3</td></tr><tr><td>CAFE+FNO (TT)</td><td>2.546e-2</td><td>3.159e-2</td><td>7.997e-3</td><td>7.036e-3</td><td>4.481e-3</td></tr><tr><td>CAFE+FNO (Tucker)</td><td>2.979e-2</td><td>3.501e-2</td><td>9.368e-3</td><td>7.324e-3</td><td>3.454e-3</td></tr></table>

Table 11: Test relative $L _ { 2 }$ errors for seed 202 after 500 training epochs. The dash denotes an omitted model–dataset combination.
<table><tr><td>Model</td><td>Darcy</td><td>NS</td><td>Burgers</td><td>Airfoil</td><td>ReacDiff</td></tr><tr><td>Resolution</td><td>128×128</td><td>128×128</td><td>1024</td><td>221×51</td><td>1024</td></tr><tr><td>FNO</td><td>6.220e-2</td><td>5.794e-2</td><td>1.138e-2</td><td>6.387e-3</td><td>4.347e-3</td></tr><tr><td>U-FNO</td><td>4.499e-2</td><td>5.789e-2</td><td>1.020e-2</td><td></td><td>6.383e-3</td></tr><tr><td>TFNO-CP</td><td>4.631e-2</td><td>3.660e-2</td><td>1.064e-2</td><td>6.896e-3</td><td>4.123e-3</td></tr><tr><td>AM-FNO (MLP)</td><td>6.792e-2</td><td>2.946e-2</td><td>1.809e-2</td><td>6.375e-3</td><td>3.048e-2</td></tr><tr><td>SirenFNO</td><td>2.341e-2</td><td>4.262e-2</td><td>6.246e-3</td><td>5.552e-3</td><td>1.774e-3</td></tr><tr><td>CAFE+FNO</td><td>1.370e-2</td><td>1.701e-2</td><td>4.440e-3</td><td>4.856e-3</td><td>3.731e-3</td></tr><tr><td>CP-SirenFNO</td><td>8.353e-2</td><td>6.165e-2</td><td>9.014e-3</td><td>1.018e-2</td><td>4.658e-3</td></tr><tr><td>TT-SirenFNO</td><td>4.625e-2</td><td>2.878e-2</td><td>7.972e-3</td><td>6.147e-3</td><td>2.401e-3</td></tr><tr><td>Tucker-SirenFNO</td><td>5.115e-2</td><td>3.709e-2</td><td>7.878e-3</td><td>6.363e-3</td><td>2.904e-3</td></tr><tr><td>CAFE+FNO (CP)</td><td>2.814e-2</td><td>4.846e-2</td><td>1.010e-2</td><td>7.332e-3</td><td>3.755e-3</td></tr><tr><td>CAFE+FNO (TT)</td><td>2.965e-2</td><td>3.525e-2</td><td>9.191e-3</td><td>6.499e-3</td><td>4.065e-3</td></tr><tr><td>CAFE+FNO (Tucker)</td><td>2.481e-2</td><td>4.197e-2</td><td>8.275e-3</td><td>9.050e-3</td><td>2.981e-3</td></tr></table>

## F Qualitative Comparisons on Random Test Samples

We qualitatively compare FNO, AM-FNO, SirenFNO, and CAFE+FNO on Darcy flow, Navier– Stokes, and Airfoil. For each dataset, three test samples were selected at random, and all four models were evaluated on the same selected samples using checkpoints trained with seed 0. The results are organized into three sample sets, each containing one test sample from each dataset.

In all figures, rows correspond to Darcy flow, Navier–Stokes, and Airfoil, from top to bottom. Prediction figures show the ground truth followed by predictions from FNO, AM-FNO, SirenFNO, and CAFE+FNO, from left to right. The corresponding error maps show the pointwise absolute error $| \hat { u } - u |$ for the four models in the same order, where ˆu and u denote the prediction and ground truth. For each figure, all panels within a dataset row share the same color scale. Values below the model panels report sample-wise relative $L _ { 2 }$ errors. Airfoil panels show a zoomed view around the airfoil.

![](images/43be15a0f6d9820cd40d1b64fe42ed729664abad47d3fc714970a55beed46c94.jpg)  
Figure 3: Ground truth and model predictions for random test sample set 1.

![](images/a7d0001ed7b88c6006acc2b08a1ae51ec7892665eef7147c90244e468f089b1c.jpg)  
Figure 4: Absolute error maps for random test sample set 1.

![](images/f6765f96801d65416220cceac0e9b2826062c66e8cb065d3e94c821ea1d03f47.jpg)  
Figure 5: Ground truth and model predictions for random test sample set 2.

![](images/1d2e440b2634ee493fd8033ed2c2749bc890a86da7601f1ce798d591f2b025e6.jpg)  
Figure 6: Absolute error maps for random test sample set 2.

![](images/999cf047d4cf4d9ff4a4b18284e36846676fc5b4538161f15f1ebbab4a07393e.jpg)  
Figure 7: Ground truth and model predictions for random test sample set 3.

![](images/22ca063c2f73e3d0906aace13cb51a1768556258e0b63356c1e2bfcaa3b6f365.jpg)  
Figure 8: Absolute error maps for random test sample set 3.