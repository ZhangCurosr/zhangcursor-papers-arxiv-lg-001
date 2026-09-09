# Learning Metamaterial Eigenmodes with Wavelet-Encoded Fourier Neural Operators

Han Zhang<sup>1</sup>, Alexander Ogren<sup>2</sup>, Cynthia Rudin<sup>3</sup>, Johann Guilleminot<sup>1</sup>, L. Catherine Brinson<sup>1</sup>

<sup>1</sup>Department of Mechanical Engineering and Materials Science, Duke University, Durham, NC, USA <sup>2</sup>Department of Mechanical Engineering, California Institute of Technology, Pasadena, CA, USA <sup>3</sup>Department of Electrical and Computer Engineering, Duke University, Durham, NC, USA

## Abstract

Machine learning surrogates based on neural operators have shown broad applicability in solving forward PDE problems. However, eigenvalue problems, in which an eigenparameter and one of several valid eigenmodes must be simultaneously solved, remain dificult because standard operator learning formulations assume a unique input–output map. This work demonstrates that Fourier Neural Operators (FNOs), combined with wavelet-based encodings of PDE inputs, can learn and predict multiple eigenmodes of the elastic wave equation, corresponding to deformation modes of acoustic waves propagating through arbitrary metamaterial geometries. We provide a mechanistic explanation and experimental evidence for why wavelet encodings are well matched to the dual spatial–spectral structure of the FNO, enabling deterministic mode selection on both continuous-valued and binary-valued geometries within a single model, and for why prediction accuracy varies with geometric discontinuities. For metamaterial design, the resulting surrogate accelerates the simulation stage of the design cycle by three orders of magnitude relative to finite element analysis on a consumer-grade CPU, while preserving high fidelity. These results also carry broader implications for designing input encodings in other multi-mode PDE solvers based on spectral neural operators.

Keywords: Neural Operator, Metamaterials, Wavelets, Eigenmodes, Machine Learning

1 Introduction 3   
2 Methodology 5   
2.1 Metamaterial Design Space 5   
2.2 Wave Propagation Simulations 7   
2.3 Fourier Neural Operator 9   
2.4 Input Wavelet Encoding 12   
2.5 Dataset Construction 15   
2.6 Training Procedure 15   
3 Results and Discussion 17   
3.1 Displacement Field Prediction 17   
3.1.1 Continuous Geometrie 18   
3.1.2 Binary Geometries 21   
3.1.3 Continuous Versus Binary Performance 24   
3.2 Dispersion Band Reconstruction . 25   
3.3 Dependence of Prediction Error on Band and Boundary Length 27   
3.4 Band and Wavevector Encoding . 30   
4 Conclusions 33   
S1 Supplementary Information 39   
S1.1 Model and Training Hyperparameters 39   
S1.2 Loss Criterion . . 39   
S1.3 Algorithms for Input Wavelet Encoding . . 43   
S1.4 Wavelet Decoding Fidelity 46

## 1. Introduction

Acoustic metamaterials are architected materials that enable compact control of sound and elastic waves, including sound attenuation, vibration isolation, and wave focusing, capabilities that are dificult to achieve with homogeneous materials alone [1, 2]. These properties arise from geometric structure in addition to material composition, so the dispersive response of a periodic unit cell is governed by the elastic wave equation and is compactly summarized by phononic band structures obtained from Bloch eigenvalue analysis [3, 4].

Understanding and designing such systems requires computing multiple deformation modes associated with diferent resonant frequencies and wave behaviors. Traditionally, these modes are obtained through computationally expensive eigenvalue analysis using finite element methods [5]. While FEA is highly reliable, this repeated evaluation can make the simulation stage of design cycles prohibitive for optimization, uncertainty quantification, inverse design, and real-time control [5, 4].

Recent advances in machine learning have introduced neural operators as a promising alternative to traditional numerical solvers. Unlike conventional neural networks that learn mappings between finite-dimensional vectors, neural operators learn mappings between function spaces, allowing them to approximate entire solution operators for families of PDEs [6, 7]. Models such as the Fourier Neural Operator (FNO) have demonstrated remarkable accuracy and computational eficiency across a variety of linear and nonlinear PDE problems while maintaining strong generalization capabilities across discretizations and geometries [8]. Related physics-informed operator architectures likewise seek rapid emulation of parametric PDE solution maps [9]. These developments have generated significant interest in using neural operators as surrogate models for computational physics applications.

Despite this progress, existing neural operator architectures have primarily been developed for PDEs in which all coeficients and boundary data are specified and each input corresponds to a unique solution. Eigenvalue PDEs instead require simultaneous recovery of an unknown eigenparameter and one of several valid eigenmodes. In phononic and acoustic metamaterials, prior surrogates have predicted dispersion from material parameters [10], addressed related eigenvalue problems with convolutional networks [11], accelerated dispersion evaluation with Gaussian processes [12], and used neural operators for transmission-loss spectra [13] or wavevector-conditioned band structures [14]. Deep learning has also been applied to phononic-crystal and metamaterial design [15, 16, 17], including interpretable methods that extract unitcell design rules for targeted bandgaps [18, 19]. Predicting full Bloch displacement eigenfields while allowing for deterministic multi-mode selection with an FNO remains an unsolved challenge with immense practical benefit.

In this work, we introduce a framework that enables Fourier Neural Operators to solve eigenvalue PDEs associated with elastic wave propagation in acoustic metamaterials. Model inputs are augmented with wavelet-based encodings of PDE parameters that uniquely identify individual eigenmodes while remaining compatible with the FNO architecture. We demonstrate that these encodings allow a single neural operator to learn multiple valid solutions corresponding to distinct deformation modes and to reproduce those modes deterministically on demand. We further show that conventional spatial encodings and purely spectral encodings do not provide the same capability, highlighting the importance of the proposed wavelet representation. We also evaluate the same model on continuous-valued and binary-valued geometries, recovering both displacement fields and eigenfrequencies, and examine how geometric discontinuities interact with the truncated spectral structure of the FNO.

These contributions are summarized as follows:

1. We demonstrate that a single Fourier Neural Operator can learn multiple eigenmodes across continuous-valued and binary-valued metamaterial geometries, recovering both eigenvectors (displacement fields) and eigenvalues (eigenfrequencies).

2. We introduce wavelet encodings of modal parameters and explain mechanistically why their spatial–spectral structure enables deterministic mode selection in FNOs.

3. We show that prediction accuracy degrades with geometric discontinuities and interpret this through spectral truncation, with implications for applying FNOs to problems with sharp interfaces.

Together, these results extend neural operators beyond conventional forward PDE problems and provide a pathway to greatly accelerate acoustic metamaterial simulation and design relative to repeated finite element eigenvalue analysis.

## 2. Methodology

This section describes the methods used to generate data and train Fourier Neural Operators for acoustic metamaterial eigenmode prediction. We first define the metamaterial design space and the Bloch FEA simulations used for supervised labels. We then summarize the FNO architecture and the wavelet encodings of modal parameters. Finally, we describe dataset construction and the training procedure.

## 2.1. Metamaterial Design Space

To train a surrogate model capable of predicting acoustic eigenmodes, a representative design space of acoustic metamaterials must first be defined. The design space was selected to provide suficient geometric and material complexity to demonstrate operator learning for eigenvalue PDEs while maintaining computationally tractable finite element simulations for dataset generation.

The dataset is restricted to two-dimensional acoustic metamaterials composed of infinitely repeating square unit cells exhibiting eight-fold symmetry. Each unit cell is discretized onto a 32 × 32 spatial grid and represented by three material-property channels corresponding to the normalized elastic modulus, density, and Poisson’s ratio. Material properties are normalized to the interval [0, 1] to improve numerical conditioning during training, while the corresponding physical values are recovered using the fixed material constants listed in Table 1. The two constituent materials were chosen to represent a structural steel and a stif elastomer.

Unit-cell geometries are synthesized by sampling a Gaussian process with a separable periodic covariance on the unit square [20], with period 1 in each coordinate, signal variance $\sigma _ { f } ^ { 2 } = 1$ , and length scale $\sigma _ { l } = 1$

<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1> $E _ { \mathrm { e l a s t o m e r } }$ </td><td rowspan=1 colspan=1> $E _ { \mathrm { s t e e l } }$ </td><td rowspan=1 colspan=1> $\rho _ { \mathrm { e l a s t o m e r } }$ </td><td rowspan=1 colspan=1> $\rho _ { \mathrm { s t e e l } }$ </td><td rowspan=1 colspan=1> $\nu _ { \mathrm { e l a s t o m e r } }$ </td><td rowspan=1 colspan=1> $\nu _ { \mathrm { s t e e l } }$ </td></tr><tr><td rowspan=1 colspan=1>Units</td><td rowspan=1 colspan=1> $\mathrm { P a }$ </td><td rowspan=1 colspan=1> $\mathrm { P a }$ </td><td rowspan=1 colspan=1> $\overline { { { \mathrm { k g } } / { \mathrm { m } } ^ { 3 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \mathrm { k g } } / { \mathrm { m } } ^ { 3 } } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1> $\overline { { 1 \times 1 0 ^ { 8 } } }$ </td><td rowspan=1 colspan=1> $2 \times 1 0 ^ { 1 1 }$ </td><td rowspan=1 colspan=1>1200</td><td rowspan=1 colspan=1>8000</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.3</td></tr></table>

Table 1: Material parameters. The two constituents are a structural steel and a stif elastomer.

$$
k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma _ { f } ^ { 2 } \prod _ { d = 1 } ^ { 2 } \exp \left( - \frac { 2 \sin ^ { 2 } \left( \pi | x _ { d } - x _ { d } ^ { \prime } | \right) } { \sigma _ { l } ^ { 2 } } \right) .
$$

Samples are drawn with mean 0.5, clipped to [0, 1], and then eight-fold (p4mm) symmetry is enforced. This procedure produces smoothly varying spatial material distributions compatible with periodically tiled square lattices. The sampled field assumes values between 0 and 1, where 0 represents pure elastomer, 1 represents pure steel, and intermediate values denote an efective mixture of the two materials below the spatial resolution of the discretization.

To investigate the influence of geometry representation on model performance, two classes of unit cells were generated. The first consists of the continuous valued Gaussian process realizations described above (henceforth referred to as continuous-valued or continuous geometries). The second is obtained by thresholding the Gaussian process samples to binary values, producing geometries composed solely of pure elastomer and pure steel (henceforth referred to as binary-valued or binary geometries). Each representation comprises half of the total dataset.

![](images/d89410fa8feab013c89658509883f2caa016af97fca30820a208677a637e2e85.jpg)

![](images/58fe32bb4be0da8ae32ab36c55d8aca39986d55a2d10549a962d374382074c23.jpg)  
Figure 1: Example of a continuous geometry (left) and a binary geometry (right) generated using the above process and parameters. White pixels represent the stifer material, black pixels represent the softer material, and grayscale pixels indicate a homogeneous mix of the two.

## 2.2. Wave Propagation Simulations

For each metamaterial geometry, wave propagation is simulated by solving the frequency-domain linear elastodynamic equation with finite element analysis (FEA), following the standard Bloch–Floquet treatment of periodic phononic media [4, 3]. After spatial discretization and enforcement of Bloch periodicity, the governing problem takes the generalized eigenvalue form

$$
\left( \mathbf { K } _ { r } ( \mathbf { k } ) - \omega ^ { 2 } \mathbf { M } _ { r } ( \mathbf { k } ) \right) \mathbf { u } = \mathbf { 0 } ,\tag{1}
$$

where $\omega$ is the angular eigenfrequency, u is the corresponding complex Bloch displacement eigenvector, and k is the wavevector in the irreducible Brillouin zone (IBZ). The reduced stifness and mass matrices K<sub>r</sub>(k) and M<sub>r</sub>(k) are obtained from geometrydependent global finite-element matrices K and M via a wavevector-dependent Bloch transformation matrix T(k),

$$
{ \bf K } _ { r } ( { \bf k } ) = { \bf T } ( { \bf k } ) ^ { \dagger } { \bf K } { \bf T } ( { \bf k } ) , \qquad { \bf M } _ { r } ( { \bf k } ) = { \bf T } ( { \bf k } ) ^ { \dagger } { \bf M } { \bf T } ( { \bf k } ) ,\tag{2}
$$

with $( \cdot ) ^ { \dagger }$ the Hermitian transpose. Here T(k) maps the full mesh degrees of freedom to an independent set while applying phase factors $\mathrm { e } ^ { i \mathbf { k } \cdot \mathbf { R } }$ across periodic faces of the unit cell. The matrices K and M themselves depend only on the material field and are assembled once per geometry. These intermediate matrices are described briefly so the reader can compare the explicit FEA pipeline (Figure 2) with the Fourier neural operator architecture in Section 2.3 below (they are not inputs to the neural operator).

The IBZ is discretized into 25 x-direction and 13 y-direction wave numbers, totaling $2 5 \times 1 3 = 3 2 5$ unique wavevectors covering the IBZ half-plane. For each wavevector, the six lowest eigenfrequencies (bands) were computed using a custom FEA MATLAB script (see Data availability for implementation details).

Each eigenvector u consists of two complex-valued displacement fields, $( u _ { x } , u _ { y } )$ ， which are separated into their real and imaginary components to produce four real-valued output channels, $( u _ { x , r e a l } , u _ { x , i m a g } , u _ { y , r e a l } , u _ { y , i m a g } )$ . The corresponding eigenfrequency ω is encoded as a spatially uniform fifth channel, resulting in a target tensor of size $5 \times 3 2 \times 3 2$ . These five-channel targets are associated with the three-channel material representation from the previous subsection that entered the FEA solve, so each simulation defines the map

$$
\mathbb { R } ^ { 3 \times 3 2 \times 3 2 }  \mathbb { R } ^ { 5 \times 3 2 \times 3 2 } .
$$

The three-channel inputs actually ingested by the Fourier neural operator are assembled diferently: a geometry channel is combined with wavelet encodings of the wavevector and band index, as described in Sections 2.4 and 2.5. The overall finite element data-generation procedure is summarized in Figure 2.

![](images/84447dbc5b41ae16a43d2cbafaa2ec5e5853c3c951e038104fa4c6635361ea59.jpg)  
Figure 2: Flowchart of the FEA data generation method. Green squares represent varying inputs to the process, and orange squares represent computed quantities of the FEA process. Black rounded rectangles represent subroutines of the FEA process. The process starts with inputs of defined densities, stifnesses, and Poisson ratios for each pixel of a metamaterial unit cell geometry (each a $3 2 \times 3 2$ pixel grid of values ranging from 0 to 1), along with a chosen $k _ { x }$ and $k _ { y }$ wavevector representing the stimulus to the metamaterial. A custom optimized FEA solver takes these inputs and produces intermediate matrices (K, M, T), forms the reduced operators $\left( \mathbf { K } _ { r } , \mathbf { M } _ { r } \right)$ , and then solves Eq. (1) for the displacement fields (eigenvectors) and frequencies ω (eigenvalues) for the given inputs.

## 2.3. Fourier Neural Operator

The principal theoretical motivation for Fourier Neural Operators is their universal approximation property over function spaces. Let A and U denote Banach spaces of input and output functions, respectively, and let

$$
\mathcal { G } : \mathcal { A }  \mathcal { U }\tag{3}
$$

be a continuous nonlinear operator. It has been shown that, given suficient latent width and Fourier modes, an FNO can approximate G arbitrarily well on compact subsets of A [6]. Unlike conventional neural networks, which approximate finitedimensional functions, FNOs approximate operators between infinite-dimensional function spaces by learning a sequence of global integral operators parameterized in the Fourier domain [8].

This theoretical framework is formulated for input and output functions in infinitedimensional spaces. In practice, these functions are represented on a finite computational grid. The continuum Fourier transform is therefore replaced by a discrete Fourier transform (DFT), and only a finite number of Fourier modes are retained during training [8, 21]. Consequently, the learned model approximates the continuum solution operator through its discrete representation while preserving the same spectral operator-learning architecture.

For the present work, the operator acts on discretized fields over a $3 2 \times 3 2$ pixel spatial grid. The input field consists of three channels: a metamaterial geometry channel (the material-location field of Section 2.1, not the three FEA property channels) together with wavelet encodings of the stimulatory wavevector and band index, constructed as detailed in Section 2.4. The output field consists of five channels corresponding to the predicted eigenvalue, or frequency, and eigenvector, or displacement field, of the resulting deformation mode. Thus, the discretized learned operator takes the form

$$
\mathcal { G } _ { \theta } : \mathbb { R } ^ { 3 2 \times 3 2 \times 3 }  \mathbb { R } ^ { 3 2 \times 3 2 \times 5 } .\tag{4}
$$

The discretized FNO follows the standard lift–transform–project architecture [8]:

1. Lifting: The input field is embedded into a higher-dimensional latent space through a pointwise lifting operator,

$$
v _ { 1 } ( x ) = \ell ( a ( x ) ) , \qquad \ell : \mathbb { R } ^ { 3 } \to \mathbb { R } ^ { d _ { L } } .\tag{5}
$$

2. Fourier layers: A sequence of Fourier layers is applied to the latent representation. Each layer updates the hidden field according to

$$
v _ { j + 1 } ( x ) = \sigma \left( \mathcal { F } ^ { - 1 } \left[ R _ { j } ( k ) \mathcal { F } [ v _ { j } ] ( k ) \right] + W _ { j } v _ { j } ( x ) + b _ { j } \right) ,\tag{6}
$$

where $\mathcal { F }$ denotes the discrete Fourier transform, $R _ { j } ( k )$ is the learnable spectral kernel, $W _ { j }$ is a learnable pointwise linear transformation, $b _ { j }$ is a learnable bias term, and σ is the activation function. GELU was used in this work because it is smooth and diferentiable.

3. Projection: The final latent representation is projected back to the desired output dimension through a pointwise projection operator,

$$
u ( \boldsymbol { x } ) = p ( v _ { n } ( \boldsymbol { x } ) ) , \qquad p : \mathbb { R } ^ { d _ { L } }  \mathbb { R } ^ { 5 } .\tag{7}
$$

FNOs are well suited for acoustic metamaterial simulation because the underlying physics is governed by spatially distributed wave equations, where the response at one location depends on the global geometry of the unit cell and the imposed wavevector. The Fourier layers are designed to use spectral convolutions to eficiently capture long-range interactions and periodic spatial structure, while the pointwise transformations preserve local geometric information. Although FNOs are commonly used as surrogate models for fixed PDE solution operators, the same operator-learning framework can be extended to eigenvalue problems by conditioning the input on modal parameters, such as the wavevector and band index, and training the network to output both the associated eigenvalue and eigenvector field. In this formulation, the FNO learns a parameterized eigensolution operator mapping geometry and modal encodings to the frequency and deformation mode of the acoustic unit cell.

This process is illustrated graphically in Figure 3. For our experiments, the NeuralOp Python package was used, which provides FNO model definitions with customizable numbers of layers, hidden channels, and input and output tensor sizes. The hyperparameter combinations explored, as well as the optimized final FNO configuration, are shown in Table S1 in Section S1. These hyperparameters are used alongside the fixed parameters of the problem space: model height 32, model width 32, input channels 3, and output channels 5.

![](images/11dd2009dd502505c63737569a53f06b5b97be9680f7c31d1b93d4cf01c39f7e.jpg)  
Figure 3: The FNO model architecture used in this project. The input data is first expanded according to the hidden-channels parameter, allowing multiple latent features to be represented between Fourier layers. Each Fourier layer is shown in detail in the expanded view. The data passing through a Fourier layer is processed through two learned branches: a pointwise branch, which applies a learned linear transformation in the spatial representation, and a spectral branch, which applies a fast Fourier transform (FFT), multiplies the retained Fourier modes by learned spectral weights, and then applies an inverse FFT. The outputs of both branches, along with a learnable bias term, are summed before passing through a nonlinear activation function. After passing through four Fourier layers, the latent representation is projected from the hidden-channel dimensionality to the final output dimensions.

## 2.4. Input Wavelet Encoding

To allow the FNO model to toggle between PDE solutions for a given geometry, information must be fed to the model indicating which excitation waveform (wavevectors) and deformation mode (band) we are looking for a solution to. This information is given in the form of wavelet embeddings of the wavevector and band index, which we find to yield lower test losses and better-resolved displacement fields than a constant field embedding (2D matrices of the same value) or a sinusoidal embedding (2D sinusoidal fields with constants encoded as the sinusoidal frequency); quantitative comparisons are reported in Tables 2 and 3, and a mechanistic discussion is given in Section 3.4.

For the wavelet encodings, we use a 1D Gabor wavelet transform to map band numbers to unique wavelet images (see Algorithm 1), and a 2D Gabor wavelet transform to map x and y wavevectors to another set of unique wavelet images (see Algorithm 2), inspired by the construction of Gabor wavelets [22].

![](images/66ecc598aeceefdc9a9c1a930971116193c8bf195c4b65bb17fa439bc7c5712c.jpg)

![](images/40e1a82b42c2b52d2d45eeb7a701713bb9c0a5515e9c8880b42b500875af38b2.jpg)

![](images/32c2e4a2309e9b68f856da10e50d8f9f3513d28cf9acd1888e72adf98d1480a6.jpg)

![](images/6122a3e36f87301efbb7293ad07b6034131f48e11b491a3c1117937c53c746c9.jpg)

![](images/46ab69de2006359314f195ffe1ea9e0589503760d448a640d88f0fee8f908292.jpg)

![](images/e116ae043d1178ba94d30ffcff9c5ffe3e608f354198a71b80b77fd3b8b70f53.jpg)  
(a) The wavelet spatial encoding for band integer.

![](images/6ecce5d6d96f76649b0f6556d9e584883957f743c631f93b69c4678a3f12b537.jpg)

![](images/84fb024537429a051a053033fa79c5154a5dbb959bebcef2f09bb6e9de9c214a.jpg)

![](images/c4052c9f5aac11e2308eb64c4e8b804ea75fcddec4834d85dc19290c99b6812e.jpg)

![](images/4c97a89f24a961f39f46875e777a0839399e20d50f7392b883231e89707ae9b4.jpg)  
(b) The FFT of the above wavelet spatial encodings.

![](images/1e37548e6a102683d7be8cdb281b724a826e8a80c9a571bdbdc6e88562e39159.jpg)

![](images/232736ed24988f1eab1f6ae3cbe867f937deff18f36a3d5b56b48171a56a7742.jpg)

Figure 4: The wavelet encoding of the band integer (Fig. 4a) and its FFT (Fig. 4b). Each band corresponds to a unique wavelet with representation in both spatial and spectral domains.  
![](images/b669307926a03523cf3f3faa85d9d7f063768a079cc2722ebd228cca1eef9add.jpg)

![](images/a4cb3f3ce3e955b48a4c0b385fcc43a7aa11da5165f72231d050f9d145afe08e.jpg)

![](images/430bc9d49f3204be202ac53980c6dfa50c4ede3d3ca713d05ffaa31b3b44d7ed.jpg)

![](images/2228126b4f35a305f9bed8c4423fb9a14c161981fc5091ea50f876c3992b2eb7.jpg)

![](images/5955a7d806bd3b7b66db14bfdf9c1f2ad8d6aeec7399f9babb5cd27b04ba86d9.jpg)

![](images/8497be0f82e4e30fac6455d53919c027125a9cd4029db764adf6fa7babf29cb4.jpg)  
(a) The wavelet spatial encoding for x and y wavevectors.

![](images/0b6e1f70c7c4d115104cfc9d1cb22c8064c7ec6551017135e75ce349ac24c48a.jpg)

![](images/b5079a72ce6bea14dde1cab4c8388c18dd3c24a00dc78f68d1beecbd037ab002.jpg)

![](images/56afa36d985c3791d6d992155224e16712080fb88c5d63295c36e0212dce1300.jpg)

![](images/77ea75966c1e6312aad177a05de3498c7ec97c36b69e696c68eff317ddc0fa3a.jpg)  
(b) The FFT of the above wavelet spatial encodings.

![](images/74b0a6497225cdbd4ef0ecefec5d8ea0c98f8e59f1d4edf2c560cf6f0ca2807a.jpg)

![](images/ea77806323fb6370b9c46ae973ae93c3b25a8d0d08cee505136d985dd88ca122.jpg)  
Figure 5: The wavelet encoding of the x and y wavevectors (Fig. 5a) and its FFT (Fig. 5b). The physical wavevectors are shown in the subplot titles. For each $( k _ { x } , k _ { y } )$ pair there is a unique mapping in both the spatial and spectral domains.

When encoding continuous constants with oscillatory functions on a finite grid, it is important to verify that distinct parameter values do not alias into nearly identical patches. Otherwise the model cannot distinguish neighboring wavevectors, and mode selection collapses. Let $\psi _ { \mathbf { k } } \in \mathbb { R } ^ { S \times S }$ be the 2D wavelet encoding of wavevector k as described above, and let $\widehat { \psi } _ { \mathbf { k } } \in \mathbb { C } ^ { S ^ { 2 } }$ be the flattened discrete Fourier transform of that 2D field (FFT computed on the $S \times S$ array, then reshaped to a vector). We define the similarity between two encodings as the cosine similarity of their mean-centered spectral descriptors, which are obtained as follows:

$$
\mathbf { s _ { k } } = \log _ { 1 0 } ( \widehat { \psi } _ { \mathbf { k } } | - \operatorname* { m e a n } \left( \log _ { 1 0 } \lvert \widehat { \psi } _ { \mathbf { k } } \rvert \right) , \quad S ( \mathbf { k } , \mathbf { k } ^ { \prime } ) = \frac { \mathbf { s _ { k } } \cdot \mathbf { s _ { k ^ { \prime } } } } { \| \mathbf { s _ { k } } \| \left\| \mathbf { s _ { k ^ { \prime } } } \right\| }
$$

Evaluating S over all 325 IBZ wavevectors yields the matrix in Figure 6. The maximum of-diagonal entry is 0.862, well below near-duplicate levels, confirming that the chosen encoding constants introduce no appreciable aliasing across the IBZ grid.

![](images/3e5e24750b5a43c162195c5dca3808ce3fabf9e2755a8f46c56a95094a5081e5.jpg)  
Figure 6: Pairwise cosine similarity S(k, k<sup>′</sup>) of mean-centered entrywise $\log _ { 1 0 } | \widehat { \psi } _ { \bf k } |$ spectra for the 325 IBZ wavevector encodings generated by Algorithm 2. The maximum of-diagonal similarity is 0.862, indicating that distinct $( k _ { x } , k _ { y } )$ pairs remain spectrally separable on the $\mathrm { 3 2 \times 3 2 ~ g r i d }$

## 2.5. Dataset Construction

Our full dataset contains 24,000 geometries, randomly generated as described in prior subsections. Half are binary geometries, with only 0s or 1s in each pixel location representing one of the two materials, while the other half are continuous geometries, with any value in [0, 1] at each pixel representing a blended material. Both geometry classes are combined into one training pool and learned simultaneously by a single model. Each geometry has six bands and 325 wavevectors per band (a mesh grid of the IBZ half-plane), resulting in $2 4 0 0 0 \times 6 \times 3 2 5 = 4 6 { , } 8 0 0 { , } 0 0 0$ sample pairs of inputs with shape $( 3 \times 3 2 \times 3 2 )$ and outputs of shape $( 5 \times 3 2 \times 3 2 )$ . The data are stored as float16 PyTorch tensors.

![](images/9bfff2eac6c1f63ab07b9c12cd3668714678b4c0c180e8c6f99adddf8e364ad5.jpg)  
(a) Construction of the input tensors

![](images/6cbb33af53a262ec074ed155b2e750ba15611072ee0d25402fa988ce91fd9d2f.jpg)  
(b) Construction of the output tensors  
Figure 7: Subfigures a and b show the processes by which input data from Section 2.1 and output data from Section 2.2 are tensorized for ingestion by the FNO model. Note the wavelet embedding steps applied to fundamentally scalar quantities.

## 2.6. Training Procedure

The Fourier Neural Operator was trained using supervised learning on paired inputs and outputs described in Sections 2.5 and 2.4. Each training sample consists of ${ \mathrm { ~ a ~ 3 ~ } } \times 3 2 \times 3 2$ input tensor (geometry and wavelet embeddings) and ${ \mathrm { ~ a ~ 5 ~ } } \times 3 2 \times 3 2$ output tensor (displacement field components and eigenfrequency). A supplementary encode–decode fidelity check for positive scalars—distinct from the input band and wavevector encodings—is reported in Section S1.4.

Five loss functions were evaluated as training objectives: mean absolute error (MAE), mean squared error (MSE), normalized MAE (NMAE), normalized MSE (NMSE), and structural similarity index (SSIM). Losses were aggregated across all five output channels. NMAE yielded the best aggregate performance and was used for all reported results. Its definition is given below and details on the other loss functions are provided in the Supplementary Information.

$$
\mathcal { L } _ { \mathrm { N M A E } } = \frac { 1 } { H W } \sum _ { i = 0 } ^ { H - 1 } \sum _ { j = 0 } ^ { W - 1 } \sum _ { c = 0 } ^ { 4 } \frac { | \hat { u } _ { c } ( i , j ) - u _ { c } ( i , j ) | } { | u _ { c } ( i , j ) | + \varepsilon }
$$

where $H = W = 3 2$ (pixels), uˆ denotes the model prediction, u denotes the ground-truth target, $\varepsilon$ is a small stabilizer, and $c \in \{ 0 , 1 , 2 , 3 , 4 \}$ corresponds to the 5 output channels: eigenfrequency, followed by real and imaginary components of x and y displacements.

For training the FNO model, combinations of model layers, hidden channels, activation function, loss function, optimizer, and training hyperparameters were evaluated. The AdamW optimizer [23] achieved the best performance compared with Adam [24], other momentum-based optimizers, and SGD. The candidate settings and selected values are reported in Table S1 in Section S1, with the selected combination values italicized.

The selected architecture uses four Fourier layers, 128 hidden channels, and GELU activations. Increasing the depth or width beyond these values did not appreciably improve validation performance and increased both training cost and the tendency to overfit. Final model performance did not appreciably change with variations in scheduler step size or batch size trialed, so these parameters were fixed at 1 and 520, respectively. Higher learning rates accelerated convergence but produced less stable optimization. In particular, learning rates above approximately $1 0 ^ { - 2 }$ occasionally caused an abrupt, irreversible loss spike, after which training plateaued. The final selected training configuration uses NMAE throughout, AdamW optimization, a learning rate of $2 \times 1 0 ^ { - 3 }$ , a batch size of 520, and 12 epochs.

## 3. Results and Discussion

The results and discussion are organized as follows. We first assess Bloch displacement predictions on continuous and binary geometries. Then, we showcase eigenfrequency and dispersion accuracy. From here, we examine how prediction error on binary geometries tracks material interface length. Finally, we compare wavelet, sinusoidal, and uniform encodings efects on model performance under matched training settings.

## 3.1. Displacement Field Prediction

We first evaluate the trained FNO on its predictions of the Bloch displacement fields $( u _ { x } , u _ { y } )$ . For each unseen test sample, the prediction error is measured by the normalized mean absolute error (NMAE) over the four real-valued displacement channels, $( u _ { x , \mathrm { r e a l } } , u _ { x , \mathrm { i m a g } } , u _ { y , \mathrm { r e a l } } , u _ { y , \mathrm { i m a g } } )$ . Samples are ranked by performance, defined inversely with NMAE, so larger losses correspond to lower performance percentiles. We select representative cases at the 25th, 50th, and 75th performance percentiles, corresponding to relatively weak, median, and relatively strong predictions. The same percentile comparison is shown for both the binary and continuous geometry datasets, allowing direct visual assessment of how well the model recovers the spatial structure of the displacement modes across contrasting geometry regimes.

## 3.1.1. Continuous Geometries

![](images/1d84325cfe2f6eaada3b95fa5abd9445588db2c7ff847f2eb971cd4c1f55fe6c.jpg)

![](images/a4aaf818a00db42c8adde159468f8a6e2688274f04bb490481196e828de54789.jpg)

![](images/e1201abe4d7897e7234b73160a38995bc05afba16246ec3dc69684f66c4757a7.jpg)  
(a) Input sample at the 25th percentile of performance for unseen continuous geometries.

![](images/e60432c1d47e9b39bcf50c70cd67344c82952d10d89c3d8ff17be527e4482a32.jpg)

![](images/83a4e6a93224a3b69947bea995bcb57cfe9e2b1db679a35b5f2e56b16e978e09.jpg)

![](images/70b6619302fa8937dd302cea8c03c39953d3cae6428b049e2772da526804b38f.jpg)

![](images/831b9a011e7720b9fc99acb6d2c6f4d84d22daf45893d2bcb0d984236dcad6ff.jpg)

![](images/e7825c9ffe836c8823a340518b9fdd33e41501abf5754de0865c9475c3993d1b.jpg)

![](images/73f37c095e7116aeff3a83a2579be2e3b1af5be9fd8eff9a0bf47711516e1f13.jpg)

![](images/e261a1635ec763e0705f52a4307db4fff9ac01cca91bb2a1a3c6d061c22c445c.jpg)

![](images/59ca81c6a42c902f2f44d76dbf8a073aa90f04c01e872974de2baab0855c5286.jpg)

![](images/f3cf0abe7a6b129b0d134d4e2ad2b8b38f7a4e5764b8288525543db2765d021c.jpg)

![](images/3a4650eda2a43960a44ea4f0e3c97f5bfc9cf856dfc1e6ee9c98bddf68afbf77.jpg)

![](images/e84010f3a2cb7d0564f75cb7868b2171d551e04f63baa874eb2f87cc932dba5e.jpg)

![](images/aeacadddb6ea6cf31abe3a8cb2fbf8c080c4b648112d0def4edda7ecf8a97001.jpg)  
(b) Output predictions at the 25th percentile of performance for unseen continuous geometries. The first row shows target fields, the second row shows predictions, and the third row shows the diferences between targets and predictions. At the 25th percentile, prediction outcomes are already fairly good, with accurate representations of scale and structure.

![](images/74070be84b32c5b012e92846934e132a81b77f99675986fcdb8f5b999664a5db.jpg)

![](images/7b99273090daae8f34684ce22e13ad9858652301f255904bed4aed7f40ed09c1.jpg)

![](images/4e1aa185e4feb50b9bdf79c8058ed1eccf4d0631d60e482b85d1539d8e3eb685.jpg)

(a) Input sample at the 50th percentile of performance for unseen continuous geometries.  
![](images/a5e7beea2ab9390adfdf83c5bc80d9b4967d7bd8be755895b2092d7f510a39d5.jpg)

(b) Output predictions at the 50th percentile of performance for unseen continuous geometries. The first row shows target fields, the second row shows predictions, and the third row shows the diferences between targets and predictions. Predictions are able to consistently capture complex structures without graininess.

![](images/49c6aa129b71b130c73079ccf626dc9434f48a7f549131d4c8163f23a4c90db5.jpg)

![](images/53c640db10b34a7e5ae1b25af9034eb55b1190ed784368f20861c67fb7b89f65.jpg)

![](images/1ef14ab98c378e7b02211fd6efa6140b9062eba8ed8456f66646687643e3eddc.jpg)

(a) Input sample at the 75th percentile of performance for unseen continuous geometries.  
![](images/dcf43d289b3eb83b30ed595f2e0743c58c99403e2f03f439b699e2641c7b87a3.jpg)  
(b) Output predictions at the 75th percentile of performance for unseen continuous geometries. The first row shows target fields, the second row shows predictions, and the third row shows the diferences between targets and predictions. At this point, the target and prediction are virtually indistinguishable by eye.

These results demonstrate that with wavelet encodings the FNO performs well on continuous geometries, faithfully reproducing the displacement fields in most cases. Model predictions in the upper performance quartiles match the targets well in structure and scale, with average pixel relative errors decreasing smoothly from on the order of $1 0 ^ { - 2 }$ to $1 0 ^ { - 4 }$ as performance percentile increases (see Figure 14a for detailed statistics).

## 3.1.2. Binary Geometries

![](images/836a9931c2e031f450b5ede431a0b86dd8fdffcf54c192d274702595576a4d82.jpg)

![](images/e09bdd5856377a3204e1eaf64388c61a56aeb56a8766625bce1905f4bc58cf7d.jpg)

![](images/aff3fadd15c1e0c21bd91308be0009a4dc1dbaa828aaf0dfc4c8ddefa148173d.jpg)

(a) Input sample at the 25th percentile of performance for unseen binary geometries.  
![](images/f4b3df8b77da2cf0ea9a37be43db015cc81b0ff9c1b40bc09f480469cf12d9a2.jpg)

![](images/2f85d65e579732d85ee6c4a2442ac59f37170470a4e032eab05f89bc55ec01ad.jpg)

![](images/a3267ed71a8482f38431d85907637e2e60b9e0a7adab2f6e6f51e948c9cbb19e.jpg)

![](images/dc0c3f4d74bb5d86be6aba0f28f40acb8b7f09cd23ba506a20dea9f8c198ddbe.jpg)

![](images/64e7f499a1dda79fabb0232f661ed6baa572bb015c253ab113e6cfb5c8afc455.jpg)

![](images/d1a21f28e2b1a082c3c4551161e99611133a25782e1b16112b7667bae8f70b79.jpg)

![](images/8837e6e08031782c50a7996c7b5f9785a7efd60b93ad1961b67d880b4966149f.jpg)

![](images/9626e9ac7133d64567379cb8975e147900c0bcac57f113ac23f6ddefa40fe6df.jpg)

![](images/105a3d3ca1e2d279cc5205d05f52f3bb5c85a0cf51cb755dfa4de234c02d5481.jpg)

![](images/eab7c6ea32c402a8ed8865d4e702c7be46887bab04c0b14ad437bea85a322f62.jpg)

![](images/8b154e942b415a97019e4cd843f712d8cef94a78f6618f849085cef4a06a5cc8.jpg)

![](images/7afddb9705ea6ec84c7e9d01e55076abbfe521688a4c01b14b7c60deedeea16e.jpg)  
(b) Output predictions at the 25th percentile of performance for unseen binary geometries. The first row shows target fields, the second row shows predictions, and the third row shows the diferences between targets and predictions. Because NMAE normalizes residuals by target magnitude, low-amplitude channels receive greater relative weight during training than their absolute scale would suggest. For this sample the imaginary displacement channels are one to two orders of magnitude larger than the real channels, so absolute error can remain low overall while relative error stays large on the smaller-magnitude channels. (Note that the colorbars for each column are independently scaled.)

![](images/5a3d2fba7d7bd882d8118326c723980e9d1ccc7f985119e907176bf09d9fd307.jpg)

![](images/2c94236c2db03de53027c47e1518672b2c608e5e3a25bba48ca9240d5a348415.jpg)

![](images/ee9a211a064c7c365196cdabddf7c0d909b75c36eb982d3d443ffcda3674823a.jpg)  
(a) Input sample at the 50th percentile of performance for unseen binary geometries.

![](images/0afa17bde83ee7e628b2d013754dbfd805792fbb5a92988bb041ea051d5e0e7d.jpg)

![](images/a2f0358d328c95fead20ae9f750e95bf4825873b91f5359383b7a8d3225c706d.jpg)

![](images/474eea94302cbd23322ef9c79002536c1b9e39833a9b822f17a9e67cf1c55092.jpg)

![](images/ca7141442ad76d2637fe80c7aa959e44b35a5ba32f9df15ab933abe508906091.jpg)

![](images/f0e98c16042671783ccae42077dd8ffc28f076f93425fd6f1ac78336285c8432.jpg)

![](images/6fe2421d1ab86c5cb8198c010ac2bcd01fd19e26ff50416287b0450e5a80aa0b.jpg)

![](images/6e0b82b4844588bdf5295abd9aeceba1a1ad54693f2f3ed5e18960e7b911f5b0.jpg)

![](images/953e863b0ffc2257f071b88f17f0fe6edb5a8a8037b1140a1789aed057f6670a.jpg)

![](images/1b65fe8bb872a7a52ebf0c91f870567b26f51aceba3e5a52511d63b2a9c7c464.jpg)

![](images/3dfa99751e866669013278a055eb5237b6d2c5550738121c238a10d0403be256.jpg)

![](images/bfdfd2549f6902bc879029b06c96e94e7ac34083bbcda13a9e65ecd543d48a50.jpg)

![](images/e1f33f89b51cd0a019feda5e7ac44c6984e05fc9ffeaf141087f8ffce95eacc6.jpg)

(b) Output predictions at the 50th percentile of performance for unseen binary geometries. The first row shows target fields, the second row shows predictions, and the third row shows the diferences between targets and predictions. Absolute and relative errors are low, with strong agreement in structure and scale between targets and predictions.

![](images/0baf91222867297ca54bddbf73eeabbbbc7e4f55da346be60e19be27c54147fe.jpg)

![](images/1c95f7491e4baca1e6414de71c32e233df0171beb17364aa8f984205bde3d998.jpg)

![](images/a33e844f9686b70b66e5920c7422e6fcdae5672093b8a3959ddff8183dfd1dd8.jpg)

(a) Input sample at the 75th percentile of performance for unseen binary geometries.  
![](images/5c31d89e0e34ba4ee1a481cda98c78a50008f0f1fbdfd2e28b2fe29facb80696.jpg)  
(b) Output predictions at the 75th percentile of performance for unseen binary geometries. The first row shows target fields, the second row shows predictions, and the third row shows the diferences between targets and predictions. At this point, disagreements between targets and predictions become very hard to see by eye.

As in the continuous geometry case, these results demonstrate that with wavelet encodings the FNO performs well on binary geometries, faithfully reproducing the displacement fields in most cases. Model predictions in the upper performance quartiles match the targets well in structure and scale, with average pixel relative errors decreasing smoothly from on the order of $1 0 ^ { - 2 } \mathrm { t o } 1 0 ^ { - 4 }$ as performance percentile increases (see Figure 14b for detailed statistics).

## 3.1.3. Continuous Versus Binary Performance

A single FNO with wavelet encodings being able to perform well on both continuous and binary geometries is a notable result. Encoding comparisons that outline theoretical reasons for this performance are given in Section 3.4. Model performance over all samples in the test set is shown in Figure 14.

The poorer performance on binary geometries is consistent with the preference of Fourier-parameterized networks for smoother inputs and outputs [25, 21]: binary designs introduce jump discontinuities and broadband high-wavenumber content that a truncated spectral representation resolves less faithfully. We develop this spectral-truncation account and its empirical support in Section 3.3.

These results indicate two important takeaways. The first is that FNOs with wavelet encodings can learn multiple deformation modes for each geometry, which the user can select by setting the wavevector and band inputs. The second is that a single trained model is robust across continuous and binary geometries from the same design space, rather than requiring separate specialization for each, which suggests that the mapping captures shared structure in the underlying eigenvalue map. At the same time, Section 3.3 shows that accuracy tends to degrade for more discontinuous binary designs: as material interfaces lengthen and neighboring pixels change more abruptly, spectral truncation on the fixed grid makes the problem harder. Whether model robustness extends to distinct design spaces or symmetry classes is left for future work.

![](images/e31cff97a30cf4ca4e79d3a638856e873bf33ff5289cd28ab69337ffb9182c53.jpg)  
(a) Performance histogram for continuous geometries.

![](images/11282a8465af80a9725555a6d6429a9355cb35c7c3667c4bebf87a7ccaea4fb3.jpg)  
(b) Performance histogram for binary geometries.  
Figure 14: Comparison of relative prediction error distributions for continuous and binary metamaterial geometries for the same model. While there are some tail outliers, most samples fall within the range of $1 0 ^ { - 2 }$ to $1 0 ^ { - 4 }$ , indicating that the same model is able to learn continuous and binary geometry cases.

## 3.2. Dispersion Band Reconstruction

![](images/474b61cae6453f517dc0f7324e798b07158c57ab997010f0ef96093a1f70fb8d.jpg)  
(a) 25th percentile performance.

![](images/dddc4b1b4770da580068d0e7254fe4be06d53babe6fc8094bfcb51c845cb4bb5.jpg)  
(b) 50th percentile performance.

![](images/a2aa3b7f70ac420407d1c42438bc566165ee59083b94c8016ff06aa904780256.jpg)  
(c) 75th percentile performance.  
Figure 15: Predicted and ground-truth dispersion bands for representative unseen continuous geometries spanning the 25th, 50th, and 75th percentiles of prediction performance, restricted to the horizontal, vertical, and diagonal IBZ traversals.

These plots show eigenfrequency predictions across all 325 wavevectors of the IBZ half-plane for continuous geometries, restricted to the horizontal, vertical, and diagonal IBZ traversals. Because the eigenfrequency is encoded with a logarithmic transform to support multi-scale learning, errors tend to scale with the target magnitude, and higher bands can appear more deviant on a linear plot. Throughout, “encoded eigenfrequency” refers to this log-transformed, spatially uniform output channel rather than physical frequency in Hz. Overall, the model achieves an average prediction error below 1% on the encoded eigenfrequency across unseen samples.

![](images/d94f8cc3e24fa0a14115491bea7e1d55e647a007334f1b099e53dced0265cc23.jpg)  
(a) 25th percentile performance.

![](images/be0179dccd9a1aa5617ae80f83636c03d36e3a7d64ad697bcda55a87733c9cca.jpg)  
(b) 50th percentile performance.

![](images/d4b90bafa4da9525540272fdf5836621ce506f3ebdfe8d3ade95f4d80d834c87.jpg)  
(c) 75th percentile performance.  
Figure 16: Predicted and ground-truth dispersion bands for representative unseen binary geometries spanning the 25th, 50th, and 75th percentiles of prediction performance, restricted to the horizontal, vertical, and diagonal IBZ traversals.

As in the continuous geometry figures above, these plots show eigenfrequency predictions across all 325 wavevectors of the IBZ half-plane for binary geometries, restricted to the horizontal, vertical, and diagonal IBZ traversals. Overall, the model again achieves an average prediction error below 1% on the encoded eigenfrequency across unseen samples.

![](images/70c601e787520a7003089f55133d1b9ee4f9dae9e9d3fbd0c8e9e29557b20594.jpg)  
(a) Performance histogram for continuous geometries.

![](images/4f981960ebc07eb95d13bd742e82bec2d9f0e747a8060bd2a79d7a40605140f3.jpg)  
(b) Performance histogram for binary geometries.  
Figure 17: Comparison of relative prediction error distributions for continuous and binary metamaterial geometries for the same model. While there are some tail outliers, most samples fall within the range of $1 0 ^ { - 2 } ~ \mathrm { t o } ~ 1 0 ^ { - 4 }$ , indicating that the same model is able to learn continuous and binary geometry cases.

## 3.3. Dependence of Prediction Error on Band and Boundary Length

The preceding subsections showed that, although the same wavelet-conditioned FNO can predict displacement fields for both continuous and binary unit cells, binary geometries are systematically more dificult, with absolute and relative errors larger (Figures 14, 17), and the harder quartiles of the binary test set degrading more visibly than their continuous counterparts. This gap is expected from the architecture of the FNO itself and from the well-documented preference of spectral networks for low-frequency content [25, 21]. Each Fourier layer retains only a truncated set of spectral modes after the discrete FFT. Smooth continuous material fields, and the comparatively smooth displacement modes they induce, concentrate their energy in a modest number of low-to-moderate wavenumbers and are therefore well represented by that truncated Fourier basis. Binary geometries, by contrast, contain jump discontinuities at material interfaces. Under an FFT, such discontinuities produce a broadband spectrum whose coeficients decay slowly with wavenumber, so a substantial fraction of the high-frequency content needed to resolve sharp phase boundaries lies outside the retained modes; related analyses of Fourier-based solvers for discontinuous coeficients likewise highlight Gibbs-type artifacts and degraded high-wavenumber recovery [26]. The spectral branch therefore sees a degraded, bandlimited version of the interface, yielding Gibbs-type ringing or blur and a harder operator-learning problem precisely when geometric discontinuity is more severe. If this spectral-truncation explanation is correct, then even among binary geometries the prediction error should increase with the boundary interface length. To test this empirically, we define the boundary length of each binarized unit cell as the number of interior four-connected pixel edges separating a 0-valued pixel from a 1-valued pixel. This scalar is a discrete measure of interface complexity. For each of the 1000 unseen binary test geometries, we compute the mean absolute error (MAE) of the predicted displacement channels, averaged over all 325 IBZ wavevectors, separately for each of the six retained eigenbands. The resulting relationships are shown in Figure 18.

![](images/bf1d66d9ed76eed858935980f1e17371b63a7b6200c9d9919758b06575811b9e.jpg)  
Figure 18: Mean displacement MAE versus interface boundary length, separated by color for each of the six eigenbands on the binary test set. Points correspond to average performance on unit-cell geometries. Legend values $\rho$ are Spearman rank correlations. A positive trend appears for every band, consistent with degraded FNO accuracy as the total length of $0 / 1$ material interfaces increases.

Two complementary trends are apparent. First, for every band there is a positive monotonic association between boundary length and MAE, with Spearman correlations ranging from $\rho = 0 . 6 4 4$ for band 1 to $\rho \approx 0 . 4 2 – 0 . 5 6$ for the higher bands. Geometries with short interfaces are concentrated at lower absolute error, while geometries with long, meandering phase boundaries systematically occupy the upper portion of the cloud. This supports the interpretation that the hard cases for the learned operator are those for which the truncated Fourier representation must resolve more extensive sharp material transitions on the fixed 32 × 32 grid. Second, absolute error itself increases with band index. Averaged over the test geometries, the mean wavevector-averaged MAE rises from approximately $1 . 6 \times 1 0 ^ { - 3 }$ for band 1 to approximately $3 . 8 \times 1 0 ^ { - 3 }$ for band 6. Higher bands correspond to more rapidly oscillating Bloch modes that concentrate energy nearer material interfaces and therefore inherit more of the geometric dificulty encoded by boundary length. Together with the continuous versus binary comparisons above, Figure 18 indicates that prediction dificulty on binary designs is organized both by band (modal complexity) and by interface measure (geometric complexity), as expected when a spectral surrogate retains only a portion of the broadband Fourier content generated by discontinuities.

## 3.4. Band and Wavevector Encoding

For the FNO to ingest the conditioning constants of the eigenvalue PDE in equation 1, namely the wavevectors $k _ { x } , k _ { y }$ and the band index b that selects which eigenmode to return, these scalars must be represented as matrices or tensors compatible with the existing channel wise input structure. Alternatives that inject scalars through specialized pathways would require heavy architectural alterations. Because FNOs are already expressive operator approximators when inputs are presented as fields [8, 6], we keep that interface and compare three field encodings of the conditioning constants: constant valued fields, spatial sinusoids, and the Gabor wavelet encoding described in Section 2.4.

The constant field representation is the naive default. On pixel coordinates $( x , y ) \in \{ 0 , \ldots , S - 1 \}$ with $S = 3 2$ , wavevector and band were encoded as

$$
I _ { k _ { x } } ( x , y ) = \frac { k _ { x } } { \pi } , \qquad I _ { k _ { y } } ( x , y ) = \frac { k _ { y } } { \pi } , \qquad k _ { x } , k _ { y } \in [ - \pi , \pi ] ,
$$

$$
I _ { b } ( x , y ) = \frac { b } { 1 0 } , \qquad b \in \{ 1 , 2 , 3 , 4 , 5 , 6 \} .
$$

Each field is spatially uniform: the normalized scalars $\begin{array} { r } { \frac { k _ { x } } { \pi } , \frac { k _ { y } } { \pi } \in \left[ - 1 , 1 \right] } \end{array}$ and $\begin{array} { r } { \frac { b } { 1 0 } \in \mathbf { \Sigma } } \end{array}$ [0.1, 0.6] are broadcast over the $S \times S$ grid. Because the FFT of a constant field is zero everywhere except at the DC coeficient, the frequency-domain branch of each Fourier layer receives little non-trivial information to propagate beyond the first layer, so learning must rely mainly on the shallow spatial-domain branch. In practice this encoding is fragile: under some training hyperparameters, particularly high learning rates, training collapses to near-constant outputs that ignore the conditioning inputs.

With more stable settings the model can still learn, but it underperforms the wavelet encoding. The best uniform-encoding results we obtained under matched conditions are reported in Tables 2 and 3 below.

A spatial sinusoidal encoding restores unique and nondegenerate representations in the spectral domain and allows the FNO to learn the problem. On pixel coordinates $( x , y ) \in \{ 0 , \ldots , S - 1 \}$ , wavevector and band were encoded as

$$
I _ { k } ( x , y ) = \sin \left( \frac { 2 } { S } \big ( k _ { x } x + k _ { y } y \big ) \right) , \qquad k _ { x } , k _ { y } \in [ - \pi , \pi ] .
$$

$$
I _ { b } ( x , y ) = \textstyle { \frac { 1 } { 2 } } \left[ \cos \left( \frac { 2 \pi b x } { S } \right) + \cos \left( \frac { 2 \pi b y } { S } \right) \right] , \qquad b \in \{ 1 , 2 , 3 , 4 , 5 , 6 \} .
$$

The wavevector field is a directed plane wave: its phase gradient is parallel to $( k _ { x } , k _ { y } )$ 2 so wavefronts propagate along the physical wavevector, and the spatial frequency scales with |k|. The prefactor $2 / S$ places about one cycle across the patch at the IBZ boundary $| \mathbf { k } | = \pi$ , which keeps neighboring $( k _ { x } , k _ { y } )$ pairs visually and spectrally distinct on the $S \times S$ grid while avoiding severe aliasing. A sine rather than a cosine is used so that the Γ point $( { \bf k } = { \bf 0 } )$ maps to the zero field instead of a constant DC patch. However, because a sinusoid is spectrally sparse, with energy concentrated at only a few wavenumbers, the Fourier branch remains weakly utilized, and predictions exhibit a persistent grainy, under-resolved structure relative to ground truth. Increasing model depth and width did not remove this graininess, indicating that the bottleneck was the encoding rather than model capacity.

The Gabor wavelet encoding resolves this by design. Following Gabor’s classical construction [22], wavelets are constructed to trade certainty between the spatial and spectral domains: neither representation is infinitely sharp, but both remain informationally rich. As a result, the conditioning inputs populate both branches of each Fourier layer with usable structure, both branches can train and contribute, and the model yields the accurate, well resolved predictions reported above while enabling deterministic mode selection. These observations support the broader conclusion that input encodings should be matched to the spatial and spectral structure of the operator: encodings that populate only one domain underuse the FNO’s Fourier branch, whereas wavelet encodings exercise both.

To quantify these diferences under matched training conditions, we compare wavelet-, sinusoidal-, and uniform-encoded models on held-out continuous and binary test geometries using overall-sample MAE, MSE, NMAE, and NMSE (Tables 2 and 3). All three runs use the same training hyperparameters: an FNO with 4 Fourier layers and 128 hidden channels, AdamW optimization, learning rate $2 \times 1 0 ^ { - 3 }$ with StepLR step size 1 and per-epoch decay 0.9, batch size 520, and an NMAE training objective. Checkpoints are taken after the 8th training epoch, where the sinusoidal- and uniform-encoded models begin to asymptote. The wavelet-encoded model continues to improve until roughly epoch 12, but the epoch-8 checkpoint is reported here for a fair comparison at equal training budget. Positive percentages in the tables indicate worse error than the wavelet-encoded model.
<table><tr><td>Geometry Dataset</td><td>Median Loss</td><td>Wavelet Encoding</td><td>Sinusoidal Encoding</td><td>Versus Wavelet</td><td>Uniform Encoding</td><td>Versus Wavelet</td></tr><tr><td rowspan="4">Continuous</td><td>MAE</td><td> $5 . 6 2 1 \times 1 0 ^ { - 4 }$ </td><td> $7 . 3 9 3 \times 1 0 ^ { - 4 }$ </td><td>+31.6%</td><td> $7 . 1 5 6 \times 1 0 ^ { - 4 }$ </td><td>+27.3%</td></tr><tr><td>MSE</td><td> $1 . 2 3 6 \times 1 0 ^ { - 6 }$ </td><td> $1 . 8 2 5 \times 1 0 ^ { - 6 }$ </td><td> $+ 4 7 . 7 \%$ </td><td> $1 . 8 4 8 \times 1 0 ^ { - 6 }$ </td><td>+49.5%</td></tr><tr><td>NMAE</td><td> $9 . 1 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 0 2 \times 1 0 ^ { - 1 }$ </td><td>+31.6%</td><td> $1 . 1 2 4 \times 1 0 ^ { - 1 }$ </td><td>+23.1%</td></tr><tr><td>NMSE</td><td> $7 . 9 7 1 \times 1 0 ^ { - 3 }$ </td><td> $1 . 1 3 5 \times 1 0 ^ { - 2 }$ </td><td> $+ 4 2 . 4 \%$ </td><td> $1 . 1 6 9 \times 1 0 ^ { - 2 }$ </td><td>+46.7%</td></tr><tr><td rowspan="4">Binary</td><td>MAE</td><td> $1 . 3 3 6 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 8 8 \times 1 0 ^ { - 3 }$ </td><td>+11.4%</td><td> $1 . 5 2 0 \times 1 0 ^ { - 3 }$ </td><td>+13.8%</td></tr><tr><td>MSE</td><td> $5 . 0 0 4 \times 1 0 ^ { - 6 }$ </td><td> $5 . 7 1 2 \times 1 0 ^ { - 6 }$ </td><td> $+ 1 4 . 2 \%$ </td><td> $6 . 1 5 0 \times 1 0 ^ { - 6 }$ </td><td>+22.9%</td></tr><tr><td>NMAE</td><td> $1 . 8 0 0 \times 1 0 ^ { - 1 }$ </td><td> $1 . 9 9 6 \times 1 0 ^ { - 1 }$ </td><td>+10.9%</td><td> $2 . 0 5 1 \times 1 0 ^ { - 1 }$ </td><td>+13.9%</td></tr><tr><td>NMSE</td><td> $3 . 0 1 \times 1 0 ^ { - 2 }$ </td><td> $3 . 3 8 \times 1 0 ^ { - 2 }$ </td><td> $+ 1 2 . 2 \%$ </td><td> $3 . 7 6 \times 1 0 ^ { - 2 }$ </td><td> $+ 2 4 . 7 \%$ </td></tr><tr><td>Geometry Dataset</td><td>Mean Loss</td><td>Wavelet Encoding</td><td>Sinusoidal Encoding</td><td>Versus Wavelet</td><td>Uniform Encoding</td><td>Versus Wavelet</td></tr><tr><td rowspan="4">Continuous</td><td>MAE</td><td> $1 . 4 0 5 \times 1 0 ^ { - 3 }$ </td><td> $1 . 6 4 9 \times 1 0 ^ { - 3 }$ </td><td>+17.4%</td><td> $1 . 5 9 1 \times 1 0 ^ { - 3 }$ </td><td>+13.2%</td></tr><tr><td>MSE</td><td> $3 . 2 3 4 \times 1 0 ^ { - 5 }$ </td><td> $3 . 4 9 6 \times 1 0 ^ { - 5 }$ </td><td>+8.1%</td><td> $3 . 5 2 9 \times 1 0 ^ { - 5 }$ </td><td>+9.1%</td></tr><tr><td>NMAE</td><td> $2 . 8 7 6 \times 1 0 ^ { - 1 }$ </td><td> $3 . 5 5 9 \times 1 0 ^ { - 1 }$ </td><td>+23.7%</td><td> $3 . 4 6 6 \times 1 0 ^ { - 1 }$ </td><td>+20.5%</td></tr><tr><td>NMSE</td><td> $1 . 4 0 5 \times 1 0 ^ { - 1 }$ </td><td> $1 . 6 1 5 \times 1 0 ^ { - 1 }$ </td><td>+14.9%</td><td> $1 . 6 1 9 \times 1 0 ^ { - 1 }$ </td><td>+15.3%</td></tr><tr><td rowspan="4">Binary</td><td>MAE</td><td> $2 . 6 6 2 \times 1 0 ^ { - 3 }$ </td><td> $2 . 8 3 0 \times 1 0 ^ { - 3 }$ </td><td>+6.3%</td><td> $2 . 8 6 1 \times 1 0 ^ { - 3 }$ </td><td>+7.5%</td></tr><tr><td>MSE</td><td> $6 . 6 4 1 \times 1 0 ^ { - 5 }$ </td><td> $6 . 7 5 6 \times 1 0 ^ { - 5 }$ </td><td>+1.7%</td><td> $7 . 0 3 0 \times 1 0 ^ { - 5 }$ </td><td>+5.8%</td></tr><tr><td>NMAE</td><td> $4 . 9 0 5 \times 1 0 ^ { - 1 }$ </td><td> $5 . 4 1 6 \times 1 0 ^ { - 1 }$ </td><td> $+ 1 0 . 4 \%$ </td><td> $5 . 5 8 5 \times 1 0 ^ { - 1 }$ </td><td>+13.9%</td></tr><tr><td>NMSE</td><td> $3 . 4 3 5 \times 1 0 ^ { - 1 }$ </td><td> $3 . 5 0 0 \times 1 0 ^ { - 1 }$ </td><td> $+ 1 . 9 \%$ </td><td> $3 . 7 7 6 \times 1 0 ^ { - 1 }$ </td><td>+9.9%</td></tr></table>

Table 2: Overall-sample median test losses and relative change versus the wavelet-encoded model. Lower is better.

Table 3: Overall-sample mean test losses and relative change versus the wavelet-encoded model. Lower is better.

## 4. Conclusions

This work shows that a single Fourier Neural Operator can learn multiple eigenmodes of the elastic wave equation for acoustic metamaterials, recovering Bloch displacement fields and eigenfrequencies across both continuous and binary unit cell geometries. Relative to prior surrogates that mainly target dispersion curves, transmission spectra, or bandgap scalars [12, 13, 14], the present model returns full deformation modes with deterministic selection among modes through wavelet encodings of the wavevector and band index. On unseen geometries of both types, displacement predictions typically achieve relative errors between $1 0 ^ { - 2 }$ and $1 0 ^ { - 4 }$ and reconstructed dispersion relations average below 1% error. Encoding choice is essential for this capability. Constant field encodings fail to propagate information through spectral branches of the FNO, leading to poor model performance. Sinusoidal encodings train better but remain grainy due to unbalanced information allocation between spatial and spectral branches. Gabor wavelet encodings [22] supply rich information structure in both the spatial and spectral branches of the FNO, enabling more reliable mode selection and learning.

A fundamental limitation of FNOs is its prediction error tends to grow with discontinuities in its inputs and outputs, as evidenced by decreasing performance with increasing interface length. Despite this, performance remains high fidelity for the trained surrogate, which is lightweight (about 2 GB at float16) and evaluates in about 1 ms per sample on a consumer-grade CPU, compared with about 1 s per sample for the corresponding FEA eigenvalue solve. That three orders of magnitude speedup makes repeated Bloch analyses, which are central to optimization, uncertainty quantification, inverse design, and screening of unit cell geometries, far more practical than direct finite element evaluation during design iterations. In short, FNOs with wavelet encodings of modal parameters provide a pathway to greatly accelerate acoustic metamaterial simulation and design relative to repeated finite element eigenvalue analysis. Future work includes other spatial resolutions, physics informed operator losses [9], and broader design spaces beyond the present eight fold symmetric, two material unit cells, including other tiling symmetries, asymmetric geometries, and multimaterial compositions.

## AI Usage Disclosure

Generative AI tools were used to assist with coding of experiments, translating MATLAB code to Python, and automating training pipelines. AI tools were also used for light editing of the manuscript. All scientific claims, results, and final wording were reviewed and approved by the authors.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## CRediT authorship contribution statement

Han Zhang: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft, Writing – review and editing.

Alexander Ogren: Data curation, Software.

Cynthia Rudin: Supervision, Writing – review and editing.

Johann Guilleminot: Formal analysis, Methodology, Supervision, Writing – review and editing.

L. Catherine Brinson: Conceptualization, Funding acquisition, Methodology, Project administration, Resources, Software, Supervision, Visualization, Writing review and editing.

## Data availability

Python code and experiments can be found at https://github.com/trutheresy/ NO-2D-Metamaterials.

MATLAB FEA code can be found at https://github.com/aco8ogren/2D-dispersion. git.

## Acknowledgements

This work was supported by the NSERC Postgraduate Scholarship – Doctoral (PGSD-599307-2025) and the NSF AI for Understanding and Designing Materials Research Traineeship (DGE-2022040).

## References

[1] S. A. Cummer, J. Christensen, A. Alù, Controlling sound with acoustic metamaterials, Nature Reviews Materials 1 (3) (2016) 1–13.

[2] G. Liao, C. Luan, Z. Wang, J. Liu, X. Yao, J. Fu, Acoustic metamaterials: A review of theories, structures, fabrication approaches, and applications, Advanced Materials Technologies 6 (5) (2021) 2000787.

[3] M. S. Kushwaha, P. Halevi, L. Dobrzynski, B. Djafari-Rouhani, Acoustic band structure of periodic elastic composites, Physical Review Letters 71 (13) (1993) 2022–2025.

[4] M. I. Hussein, M. J. Leamy, M. Ruzzene, Dynamics of phononic materials and structures: Historical origins, recent progress, and future outlook, Applied Mechanics Reviews 66 (4) (2014) 040802.

[5] H. Zhang, R. Karimi Mahabadi, C. Rudin, J. Guilleminot, L. C. Brinson, Uncertainty quantification of acoustic metamaterial bandgaps with stochastic material properties and geometric defects, Computers & Structures 305 (2024) 107511.

[6] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, A. Anandkumar, Neural operator: Learning maps between function spaces with applications to PDEs, Journal of Machine Learning Research 24 (89) (2023) 1–97.

[7] L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis, Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nature Machine Intelligence 3 (3) (2021) 218–229.

[8] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, Fourier neural operator for parametric partial diferential equations, in: International Conference on Learning Representations, 2021.

[9] S. Wang, H. Wang, P. Perdikaris, Learning the solution operator of parametric partial diferential equations with physics-informed DeepONets, Science Advances 7 (40) (2021) eabi8605.

[10] C.-X. Liu, G.-L. Yu, Predicting the dispersion relations of one-dimensional phononic crystals by neural networks, Scientific Reports 9 (2019) 15322.

[11] D. Finol, Y. Lu, V. Mahadevan, A. Srivastava, Deep convolutional neural networks for eigenvalue problems in mechanics, International Journal for Numerical Methods in Engineering 118 (5) (2019) 258–275.

[12] A. C. Ogren, B. T. Feng, K. L. Bouman, C. Daraio, Gaussian process regression as a surrogate model for the computation of dispersion relations, Computer Methods in Applied Mechanics and Engineering 420 (2024) 116661.

[13] J. E. Wagner, S. Burbulla, M. de Benito Delgado, J. D. Schmid, Neural operators as fast surrogate models for the transmission loss of parameterized sonic crystals,

in: NeurIPS 2024 Workshop on Data-driven and Diferentiable Simulations, Surrogates, and Solvers, 2024.

[14] C. Liu, S. Ning, X. Li, Y. Hong, T. Qi, Y. Kai, Mesh-independent band structure prediction of phononic crystals via a wave-vector explicit hybrid Fourier neural operator, Computers & Structures (2026).

[15] C.-X. Liu, G.-L. Yu, Deep learning for the design of phononic crystals and elastic metamaterials, Journal of Computational Design and Engineering 10 (2) (2023) 602–614.

[16] Y. Jin, L. He, Z. Wen, B. Mortazavi, H. Guo, D. Torrent, B. Djafari-Rouhani, T. Rabczuk, X. Zhuang, Y. Li, Intelligent on-demand design of phononic metamaterials, Nanophotonics 11 (3) (2022) 439–460.

[17] Muhammad, J. Kennedy, C. W. Lim, Machine learning and deep learning in phononic crystals and metamaterials – a review, Materials Today Communications 33 (2022) 104606.

[18] Z. Chen, A. Ogren, C. Daraio, L. C. Brinson, C. Rudin, How to see hidden patterns in metamaterials with interpretable machine learning, Extreme Mechanics Letters 57 (2022) 101895.

[19] M. V. Bastawrous, Z. Chen, A. C. Ogren, C. Daraio, C. Rudin, L. C. Brinson, A multiscale design method using interpretable machine learning for phononic materials with closely interacting scales, Computer Methods in Applied Mechanics and Engineering 440 (2025) 117833.

[20] C. E. Rasmussen, C. K. I. Williams, Gaussian Processes for Machine Learning, MIT Press, Cambridge, MA, 2006. URL https://gaussianprocess.org/gpml/

[21] S. Qin, F. Lyu, W. Peng, T. Geng, J. Pang, Y. Wang, C. Deng, X. Liu, Toward a better understanding of Fourier neural operators from a spectral perspective, arXiv preprint arXiv:2404.07200 (2024).

[22] D. Gabor, Theory of communication, Journal of the Institution of Electrical Engineers - Part III: Radio and Communication Engineering 93 (26) (1946) 429–441.

[23] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: International Conference on Learning Representations, 2019.

[24] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, in: International Conference on Learning Representations, 2015.

[25] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, A. Courville, On the spectral bias of neural networks, in: Proceedings of the 36th International Conference on Machine Learning, Vol. 97 of Proceedings of Machine Learning Research, 2019, pp. 5301–5310.

[26] G. M. Cavallazzi, M. Pérez Cuadrado, A. Pinelli, Walsh–Hadamard neural operators for solving PDEs with discontinuous coeficients, Journal of Computational PhysicsAlso available as arXiv:2511.07347 (2026).

## S1. Supplementary Information

## S1.1. Model and Training Hyperparameters

<table><tr><td rowspan=1 colspan=1>Hyperparameter</td><td rowspan=1 colspan=4>Candidate Settings</td></tr><tr><td rowspan=1 colspan=1>Fourier Layers</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Hidden Channels</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>256</td></tr><tr><td rowspan=1 colspan=1>Activation Function</td><td rowspan=1 colspan=1>ReLU</td><td rowspan=1 colspan=1>GELU</td><td rowspan=1 colspan=1>tanh</td><td rowspan=1 colspan=1>Sigmoid</td></tr><tr><td rowspan=1 colspan=1>Loss Criterion</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=1>NMAE</td><td rowspan=1 colspan=1>NMSE</td></tr><tr><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>20</td></tr><tr><td rowspan=1 colspan=1>Learning Rate</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1> $2 \times 1 0 ^ { - 3 }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 2 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 1 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Weight Decay</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 3 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 ^ { - 2 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Gamma</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.9</td></tr><tr><td rowspan=1 colspan=1>Step Size</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>Batch Size</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>520</td></tr><tr><td rowspan=1 colspan=1>Data Precision</td><td rowspan=1 colspan=1>F8 E4M3</td><td rowspan=1 colspan=1>F8 E5M2</td><td rowspan=1 colspan=1>F16</td><td rowspan=1 colspan=1>F32</td></tr></table>

Table S1: Combinations tried in grid search for optimal model and training hyperparameters. Each row is a set of hyperparameters that can be paired with any from other rows.

## S1.2. Loss Criterion

In the exploration of optimal training protocols for the FNO model, several loss functions were evaluated as training objectives: mean absolute error (MAE), mean squared error (MSE), normalized MAE (NMAE), normalized MSE (NMSE), and structural similarity index (SSIM). Among these, NMAE yielded the best overall performance and was used for all reported results. All comparisons used a learning rate of $2 \times 1 0 ^ { - 3 }$ with a per-epoch decay factor of 0.9. The mathematical definitions of each loss follow. Losses were evaluated and aggregated across all five output channels.

• The mean absolute error (MAE) loss is defined as

$$
\mathcal { L } _ { \mathrm { M A E } } = \frac { 1 } { H W } \sum _ { i = 0 } ^ { H - 1 } \sum _ { j = 0 } ^ { W - 1 } \sum _ { c = 0 } ^ { 4 } | \hat { u } _ { c } ( i , j ) - u _ { c } ( i , j ) |
$$

Experimentally, models trained exclusively with MAE produced predictions with sharper interfaces and better-preserved boundary features than those trained solely with MSE. This behavior is consistent with the linear penalty of MAE, which does not disproportionately penalize isolated large residuals near discontinuities. Despite the improved qualitative sharpness, MAE training underperformed NMAE on overall validation accuracy, so absolute-error training alone was not selected for the reported models.

• The mean squared error (MSE) loss is defined as

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { H W } \sum _ { i = 0 } ^ { H - 1 } \sum _ { j = 0 } ^ { W - 1 } \sum _ { c = 0 } ^ { 4 } \left( \hat { u } _ { c } ( i , j ) - u _ { c } ( i , j ) \right) ^ { 2 }
$$

The quadratic penalty of MSE assigns increasingly large gradients to larger residuals, causing the optimizer to preferentially reduce regions with the greatest numerical error. As a consequence, the loss favors smooth predictions that minimize the overall energy of the residual rather than preserving sharp transitions. Experimentally, models trained exclusively with MSE produced outputs that reproduced the correct magnitude and large-scale structure of the solution, but exhibited difuse interfaces and blurred boundaries. Fine-scale features were often replaced by low-amplitude, grainy artifacts, consistent with the tendency of MSE to average over uncertain or discontinuous regions in order to minimize the global squared error.

• The structural similarity index measure (SSIM) is defined as

$$
\begin{array} { c } { { \displaystyle \mathcal { L } _ { \mathrm { S S I M } } = 1 - \frac { 1 } { 5 } \sum _ { c = 0 } ^ { 4 } \frac { \left( 2 \mu _ { u _ { c } } \mu _ { \hat { u } _ { c } } + C _ { 1 } \right) \left( 2 \sigma _ { u _ { c } \hat { u } _ { c } } + C _ { 2 } \right) } { \left( \mu _ { u _ { c } } ^ { 2 } + \mu _ { \hat { u } _ { c } } ^ { 2 } + C _ { 1 } \right) \left( \sigma _ { u _ { c } } ^ { 2 } + \sigma _ { \hat { u } _ { c } } ^ { 2 } + C _ { 2 } \right) } \mathrm { , } } } \\ { { \displaystyle \mu _ { u _ { c } } = \frac { 1 } { H W } \sum _ { i = 0 } ^ { H - 1 } \sum _ { j = 0 } ^ { W - 1 } u _ { c } ( i , j ) \mathrm { , } \quad \sigma _ { u _ { c } \hat { u } _ { c } } = \frac { 1 } { H W } \sum _ { i = 0 } ^ { H - 1 } \sum _ { j = 0 } ^ { W - 1 } \left( u _ { c } ( i , j ) - \mu _ { u _ { c } } \right) \left( \hat { u } _ { c } ( i , j ) - \mu _ { \hat { u } _ { c } } \right) } } \end{array}
$$

A significant limitation of SSIM arises when it is applied to signed-valued fields. Unlike its original application to nonnegative image intensities, a perfect sign-reversed field $( \hat { u } = - u )$ can achieve nearly the same SSIM score as a perfect reconstruction $( \hat { u } = u )$ . Consequently, a model trained solely with SSIM may converge to physically incorrect solutions containing polarity inversions while still minimizing the loss. The following derivation demonstrates the mathematical origin of this ambiguity.

For the purposes of the following discussion, we consider the common case where the stabilizing constants satisfy

$$
C _ { 1 } \ll 2 \mu _ { u _ { c } } ^ { 2 } , \qquad C _ { 2 } \ll 2 \sigma _ { u _ { c } } ^ { 2 } ,
$$

allowing them to be neglected.

For a correct reconstruction,

$$
u _ { c } ^ { ( + ) } = u _ { c } ,
$$

the SSIM score is

$$
\mathrm { S S I M } ( u _ { c } , u _ { c } ^ { ( + ) } ) = \frac { \left( 2 \mu _ { u _ { c } } ^ { 2 } \right) \left( 2 \sigma _ { u _ { c } } ^ { 2 } \right) } { \left( 2 \mu _ { u _ { c } } ^ { 2 } \right) \left( 2 \sigma _ { u _ { c } } ^ { 2 } \right) } = 1 .
$$

Now consider a sign-reversed reconstruction. The field and its mean reverse sign,

$$
u _ { c } ^ { ( - ) } = - u _ { c } , ~ \mu _ { u _ { c } ^ { ( - ) } } = - \mu _ { u _ { c } } ,
$$

while the covariance reverses sign and the variance remains unchanged,

$$
\sigma _ { u _ { c } , u _ { c } ^ { ( - ) } } = - \sigma _ { u _ { c } , u _ { c } } , \qquad \sigma _ { u _ { c } ^ { ( - ) } } ^ { 2 } = \sigma _ { u _ { c } } ^ { 2 } .
$$

Substituting these relationships into the SSIM expression gives

$$
\mathrm { S S I M } ( u _ { c } , u _ { c } ^ { ( - ) } ) = \frac { \left( - 2 \mu _ { u _ { c } } ^ { 2 } \right) \left( - 2 \sigma _ { u _ { c } } ^ { 2 } \right) } { \left( 2 \mu _ { u _ { c } } ^ { 2 } \right) \left( 2 \sigma _ { u _ { c } } ^ { 2 } \right) } = 1 .
$$

The numerator therefore difers from that of a correct reconstruction by two factors of −1, and thus, under these conditions, SSIM assigns essentially the same similarity score to a field and its sign-reversed counterpart as it does to a perfect reconstruction. This ambiguity constitutes a significant limitation when SSIM is applied to signed-valued regression problems where the polarity of the solution is physically meaningful. During training, the network frequently converged to predictions in which localized regions or the entire image were the negative of the correct solution while still achieving a favorable SSIM objective. Although the resulting fields exhibited accurate interface locations, boundary geometry, and overall morphology, the polarity inversions rendered the predictions physically incorrect. Consequently, SSIM should not be used as the sole training criterion for signed-valued fields unless it is supplemented by an additional loss that explicitly penalizes sign reversals.

• The normalized mean absolute error (NMAE) loss is defined as

$$
\mathcal { L } _ { \mathrm { N M A E } } = \frac { 1 } { H W } \sum _ { i = 0 } ^ { H - 1 } \sum _ { j = 0 } ^ { W - 1 } \sum _ { c = 0 } ^ { 4 } \frac { | \hat { u } _ { c } ( i , j ) - u _ { c } ( i , j ) | } { | u _ { c } ( i , j ) | + \varepsilon }
$$

Unlike MAE, NMAE weights each prediction error by the inverse magnitude of the target value, causing the optimization to place greater emphasis on regions where the ground-truth solution is small in scale. Consequently, the loss seeks to minimize relative error rather than absolute error, preventing low-magnitude features from being dominated by high-magnitude regions during training.

Experimentally, NMAE gave the best overall balance across channels with disparate magnitudes and was selected as the training objective for all reported results. Relative-error emphasis improved reconstruction of low-amplitude displacement components without sacrificing usable accuracy on larger-magnitude channels, which made NMAE preferable to MAE, MSE, NMSE, and SSIM for this application.

• The normalized mean squared error (NMSE) loss is defined as

$$
\mathcal { L } _ { \mathrm { N M S E } } = \frac { 1 } { H W } \sum _ { i = 0 } ^ { H - 1 } \sum _ { j = 0 } ^ { W - 1 } \sum _ { c = 0 } ^ { 4 } \frac { ( \hat { u } _ { c } ( i , j ) - u _ { c } ( i , j ) ) ^ { 2 } } { ( u _ { c } ( i , j ) ) ^ { 2 } + \varepsilon }
$$

Similar to NMAE, NMSE normalizes the prediction error by the magnitude of the target field, causing the optimization to prioritize relative error rather than absolute error. The quadratic penalty retains the characteristic behavior of MSE by assigning greater weight to larger residuals, while the normalization emphasizes regions where the target magnitude is small.

Experimentally, NMSE reduced the relative error in low-amplitude regions compared with standard MSE, but it underperformed NMAE on overall validation accuracy and was not selected for the reported models.

## S1.3. Algorithms for Input Wavelet Encoding

The band and wavevector conditioning channels used by the FNO are generated by the Gabor wavelet embeddings below, which match the implementations in the accompanying code.

Algorithm 1 1D Gabor Wavelet Embedding from Band Index   
1: INPUT: Band index $b ;$ grid size $S \gets 3 2 ;$ frequency scale $r \gets 2 . 0$   
2: Create linspace vectors $x , y \in [ - 1 , 1 ]$ with S points   
3: Construct meshgrid $X , Y \gets$ meshgrid(x, y)   
4: Compute base frequency:   
$f  ( 1 . 0 + | b | ) \cdot \frac { S } { 8 }$   
5: Compute rotation angle:   
$\theta  ( b \mathrm { ~ m o d ~ } 8 ) \cdot \frac { S } { 1 6 }$   
6: Rotate coordinates:   
$X _ { \theta }  X$ cos $\theta + Y$ sin θ, $Y _ { \theta } \gets - X$ sin $\theta + Y$ cos $\theta$   
7: Set Gaussian envelope widths:   
$\sigma _ { x }  \sigma _ { y }  \frac { 0 . 3 } { r }$   
8: Compute Gabor wavelet embedding:   
$\psi _ { b } ( x , y ) \gets \exp { \left( - \frac { X _ { \theta } ^ { 2 } } { 2 \sigma _ { x } ^ { 2 } } - \frac { Y _ { \theta } ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } \right) } \cdot \cos ( f \cdot X _ { \theta } )$   
9: RETURN: $\psi _ { b } \in \mathbb { R } ^ { S \times S }$

In Algorithm 1, the grid size $S = 3 2$ matches the geometry resolution used by the FNO. For this resolution the base frequency is $f = ( 1 + | b | ) S / 8$ , so successive bands increase both carrier frequency and orientation. The modulo in $\theta$ repeats orientation every eight bands, while $f$ continues to increase, so embeddings remain distinguishable beyond eight bands. The constants $r = 2 . 0 , S / 1 6 .$ , and 0.3 were chosen so that the wavelets occupy most of the spatial domain and remain visually distinct across bands.

Algorithm 2 2D Gabor Wavelet Embedding for Wavevectors   
1: INPUT: Wavevector components $k _ { x } , k _ { y }$ (radians); grid size $S \gets 3 2 ;$ frequency   
scale $r  1 . 0$   
2: Set constants $f _ { 0 } \gets 2 . 2 0 , \alpha \gets 4 1 , N _ { x } \gets 1 3 , N _ { y } \gets 2 5$   
3: Create linspace vectors x, $y \in [ - 1 , 1 ]$ with S points   
4: Construct meshgrid $X , Y \gets$ meshgrid(x, y)   
5: Compute frequencies:   
$f _ { x } \gets \left( f _ { 0 } + k _ { x } \right) \alpha , \quad f _ { y } \gets \left( f _ { 0 } + k _ { y } \right)$ α   
6: Compute rotation angles:   
$\theta _ { x }  ( k _ { x } \mathrm { m o d } N _ { x } ) \cdot \frac { \pi } { N _ { x } } , \quad \theta _ { y }  ( k _ { y } \mathrm { m o d } N _ { y } ) \cdot \frac { \pi } { N _ { y } }$   
7: Rotate coordinates:   
$X _ { \theta }  X$ cos $\theta _ { x } + Y$ sin $\theta _ { y }$ , Y ← −X sin $\theta _ { x } + Y$ cos $\theta _ { y }$   
8: Set Gaussian envelope widths:   
$\sigma _ { x } \gets \sigma _ { y } \gets \frac { 0 . 5 } { r }$   
9: Compute Gabor wavelet embedding:   
$\psi _ { k _ { x } , k _ { y } } ( x , y ) \gets \exp { \left( - \frac { X _ { \theta } ^ { 2 } } { 2 \sigma _ { x } ^ { 2 } } - \frac { Y _ { \theta } ^ { 2 } } { 2 \sigma _ { y } ^ { 2 } } \right) } \cdot \sin ( f _ { x } X _ { \theta } ) \cdot \sin ( f _ { y } Y _ { \theta } )$   
10: RETURN: $\psi _ { k _ { x } , k _ { y } } \in \mathbb { R } ^ { S \times S }$   
In Algorithm 2, $k _ { x }$ and $k _ { y }$ enter as continuous radian valued wavevector com  
ponents. Distinct modulo periods $N _ { x } = 1 3$ and $N _ { y } = 2 5$ reduce aliasing under   
interchange of the two components. The ofset $f _ { 0 }$ and scale α map the physical   
range of k into carrier frequencies that remain well resolved on the $S \times S$ grid, while

$\sigma = 0 . 5 / r$ sets the envelope width so that the patterns remain spatially localized yet informative in both domains.

## S1.4. Wavelet Decoding Fidelity

For purposes of checking general applicability, an experiment was run to check if a positive scalar may be embedded in a Gabor wavelet image and recovered after pixelization, storage, or inference. Algorithms 3 and 4 implement this encode– decode pair by spectral peak detection with centroid refinement. In the experiment, we test recovery over [1, 8000], spanning about four orders of magnitude. Despite discretization onto a finite $S \times S$ grid and limited numeric precision associated with float16 storage and arithmetic, the decoded values remain accurate across the full range. Figures S1 and S2 show representative encodings and the corresponding round trip error.

![](images/d1062e29a04c2e5721a2f2d91135071bd9fb6d71d96647cb4247be1211a2d59d.jpg)  
Figure S1: Log spaced Gabor wavelet encodings of scalars from 1 to 8000. Alternating rows show the spatial wavelet images and the corresponding Fourier magnitude spectra.

![](images/d8334de5727d92ee20de6f4aa8ae7a298bcc4be8031cc084f6cad0fa63c5c50e.jpg)

![](images/7f41341cdb343cade7c24ba391638fbb0fa5d8f41a118041174d629424f33c19.jpg)  
Figure S2: Encode–decode error for scalars from a minimum of 1 to a maximum of 8000.

Algorithms 3 and 4 summarize the forward and inverse mapping used in this experiment. The forward mapping places the scalar on a log scale and jointly encodes magnitude and orientation into a Gabor-like pattern. The inverse mapping recovers an estimate by spectral peak detection with centroid refinement, followed by a consistency step that combines extracted magnitude and angle to reconstruct the most likely log value.

Algorithm 3 Scalar Wavelet Encoding   
1: INPUT: Scalar $s > 0 ;$ image size $S \gets 3 2 ;$ bounds $\left[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } \right] ;$ spectral-index   
bounds $[ k _ { \operatorname* { m i n } } , k _ { \operatorname* { m a x } } ]$   
2: Set constants $\gamma  1 , \phi  0 , \sigma  8 , \theta _ { \operatorname* { m i n } }  \pi / 3 0 , \theta _ { \operatorname* { m a x } }  \pi / 2 - \pi / 3 0$   
3: Compute log-domain values:   
$\ell _ { s } \gets \log s , \quad \ell _ { \mathrm { m i n } } \gets \log s _ { \mathrm { m i n } } , \quad \ell _ { \mathrm { m a x } } \gets \log s _ { \mathrm { m a x } }$   
4: Compute normalized position and spectral index:   
$t \gets \mathrm { c l i p } \left( \frac { \ell _ { s } - \ell _ { \mathrm { m i n } } } { \ell _ { \mathrm { m a x } } - \ell _ { \mathrm { m i n } } } , 0 , 1 \right) , \quad k \gets k _ { \mathrm { m i n } } + t \left( k _ { \mathrm { m a x } } - k _ { \mathrm { m i n } } \right)$   
5: Compute log width per unit k:   
$\Delta _ { \ell } \gets \frac { \ell _ { \mathrm { m a x } } - \ell _ { \mathrm { m i n } } } { k _ { \mathrm { m a x } } - k _ { \mathrm { m i n } } }$   
6: Compute orientation from within-band log position:   
$\theta \gets \theta _ { \operatorname* { m i n } } + \left( \frac { \left( \ell _ { s } - \ell _ { \operatorname* { m i n } } \right) \bmod \Delta _ { \ell } } { \Delta _ { \ell } } \right) \left( \theta _ { \operatorname* { m a x } } - \theta _ { \operatorname* { m i n } } \right)$   
7: Build centered grid X, $Y \in \{ - S / 2 , \dots , S / 2 - 1 \}$ , then rotate:   
$X _ { \theta } \gets X \cos \theta + Y \sin \theta , \quad Y _ { \theta } \gets - X \sin \theta + Y$ cos θ   
8: Set envelope widths $\sigma _ { x }  \sigma , \ \sigma _ { y }  \gamma \sigma _ { x } ,$ and carrier frequency $\omega  2 \pi k / S$   
9: Compute wavelet image:   
$\psi _ { s } \gets \exp \biggl ( - \frac { 1 } { 2 } \left( \frac { X _ { \theta } ^ { 2 } } { \sigma _ { x } ^ { 2 } } + \frac { Y _ { \theta } ^ { 2 } } { \sigma _ { y } ^ { 2 } } \right) \biggr ) \cos ( \omega X _ { \theta } + \phi )$   
10: RETURN: $\psi _ { s } \in \mathbb { R } ^ { S \times S }$ , k, θ

Algorithm 4 Scalar Wavelet Decoding   
1: INPUT: Wavelet image $\psi \in \mathbb { R } ^ { S \times S }$ ; bounds $[ s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ] , [ k _ { \mathrm { m i n } } , k _ { \mathrm { m a x } } ] , [ \theta _ { \mathrm { m i n } } , \theta _ { \mathrm { m a x } } ]$   
2: Set constants for centroid refinement: radius $R  3$ , power $p \gets 2$   
3: Mean-center image $\psi _ { c }  \psi - \mathrm { m e a n } ( \psi )$ , then compute the centered 2D discrete   
Fourier magnitude:   
$\hat { \psi } ( k _ { x } , k _ { y } ) \gets \sum _ { m = 0 } ^ { S - 1 } \sum _ { n = 0 } ^ { S - 1 } \psi _ { c } ( m , n ) \exp \left( - 2 \pi i \bigg ( \frac { k _ { x } m } { S } + \frac { k _ { y } n } { S } \bigg ) \right) , \quad F ( k _ { x } , k _ { y } ) \gets | \hat { \psi } ( k _ { x } , k _ { y } ) |$   
4: Keep upper-half-plane spectral support (plus positive-frequency center row) and   
find dominant peak index $( r ^ { * } , c ^ { * } )$   
5: Define symmetric peak partner about center, then compute weighted local centroid   
around $( r ^ { * } , c ^ { * } )$ within radius $R ,$ using weights $w = F ^ { p }$ and excluding points closer   
to the symmetric partner   
6: Convert refined centroid to wavevector coordinates $( k _ { x } , k _ { y } )$ , then:   
kext ← qk2<sub>x</sub> + k2<sub>y</sub>, θ<sub>ext</sub> ← atan2(k<sub>y</sub>, k<sub>x</sub>) mod π   
7: Map extracted k to approximate log value:   
$\ell _ { \mathrm { a p p r o x } } \gets \ell _ { \mathrm { m i n } } + \mathrm { c l i p } \left( \frac { k _ { \mathrm { e x t } } - k _ { \mathrm { m i n } } } { k _ { \mathrm { m a x } } - k _ { \mathrm { m i n } } } , 0 , 1 \right) \left( \ell _ { \mathrm { m a x } } - \ell _ { \mathrm { m i n } } \right)$   
8: Let $\Delta _ { \ell } \gets ( \ell _ { \mathrm { m a x } } - \ell _ { \mathrm { m i n } } ) / ( k _ { \mathrm { m a x } } - k _ { \mathrm { m i n } } )$ , compute within-band position from angle:   
$\alpha  \mathrm { c l i p } \Bigg ( \frac { \theta _ { \mathrm { e x t } } - \theta _ { \mathrm { m i n } } } { \theta _ { \mathrm { m a x } } - \theta _ { \mathrm { m i n } } } , 0 , 1 \Bigg ) , \quad \delta _ { \ell }  \alpha \Delta _ { \ell }$   
9: Find nearest consistent log value by testing neighboring bands $b - 1 , b , b + 1$   
ℓ<sup>⋆</sup> ← arg min |ℓ − ℓ<sub>approx</sub>|   
ℓ∈{ℓ<sub>min</sub>+j∆<sub>ℓ</sub>+δ<sub>ℓ</sub>}   
10: Clip $\ell ^ { \star }$ to $[ \ell _ { \mathrm { m i n } } , \ell _ { \mathrm { m a x } } ]$ , then decode:   
s<sub>ext</sub> ← exp(ℓ<sup>⋆</sup>)   
11: RETURN: s<sub>ext</sub>, k<sub>ext</sub>, θ<sub>ext</sub> 50