# Detection of Memorization in Diffusion Models Through Thermodynamic Singularity

Iljung Kim, Kangwoo Park, Yung-Kyun Noh Hanyang University Seoul 04763, Rep. of Korea. {iljung0810,pkw3322,nohyung}@hanyang.ac.kr

Daniel D. Lee<sup>\*</sup> Cornell Tech New York, NY 10044, USA ddl46@cornell.edu

## Abstract

In diffusion models, proper distributional learning entails reversing a diffusion process over underlying density functions, whereas a key failure is memorization, in which the reverse process instead learns diffusion from individual training samples. We provide an analytical characterization of such memorized reverse processes, along with a criterion that detects the associated thermodynamic singularity. The proposed criterion reliably distinguishes between properly learned and memorizing networks across a range of neural network models designed to promote or mitigate memorization.

## 1 Introduction

High-fidelity generation with diffusion models has led to their widespread use, while also raising concerns about memorization. Memorization can be understood as overfitting to the forward diffusion dynamics, where the learned reverse process becomes tied to sample-specific trajectories originating from individual training samples. In the extreme, exact time-reversal of forward diffusion trajectories can reproduce training samples, resulting in limited diversity and generalization and introducing privacy risks [3, 15].

Detecting such behavior is not straightforward. Direct comparison between generated samples and training data has been widely used to assess memorization [16], but it requires access to the full training dataset and a suitable similarity metric. Several proxy measures of these overfitting have also been proposed, including those based on the training loss, the Hessian spectrum of the log density [7], and the difference between conditional and unconditional scores [19]. However, the measures based on these proxies are largely empirically motivated and rely on high-dimensional estimation, making them difficult to interpret in a principled manner and sensitive to noise.

In principle, these trajectories should reveal clear and observable differences between overfitted and properly learned networks. However, such differences diminish as the number of training samples grows, since the trajectory structure induced by individual samples becomes progressively smeared. Nonetheless, retrieval of training samples and the generation of new samples remain fundamentally different, and detecting the difference is important even in large datasets, as the possibility of reproducing training samples—even if small—can severely limit the applicability of the model.

In this paper, we provide a simple derivation using the equation of continuity, showing that the velocity field diverges in the diffusion of Dirac delta functions. This singularity, which can serve as a necessary condition for memorization, arises in all forward processes constructed from finite samples, regardless of sample size. Memorization in the reverse process cannot occur without learning this singularity, and this singularity provides a direct analytical characterization of the presence of memorization.

The mathematical analogy of the velocity field and its corresponding flux naturally leads to a thermodynamic quantity—the Helmholtz free energy of the diffusive process. The memorization characteristic can be interpreted using the instantaneous rate of change of the Helmholtz free energy over time, providing a principled and interpretable scalar measure that does not require access to training data. Using this rate of change as a criterion, the detection of memorization was effective across both overfitting and non-overfitting neural networks, outperforming conventional methods on synthetic and real-world datasets. In addition, we further propose a simple modification of the training loss function that suppresses excessive growth of the diffusive velocity above a predefined threshold based on the free energy rate, significantly reducing memorization without degrading the fidelity of generated samples.

The remainder of the paper is organized as follows. Section 2 derives the low-noise behavior of velocity fields associated with diffused delta functions. Section 3 introduces the thermodynamic criterion and its interpretation. Section 4 presents experiments on synthetic and real-world datasets, including memorization mitigation. Section 5 concludes.

## 2 Forward and Reverse Processes in Diffusion Models

The forward process in diffusion models adds noise to all data using the following update equation:

Forward process:

$$
\begin{array} { r } { \mathbf { x } _ { t + \Delta t } = \sqrt { 1 - \beta \Delta t } \mathbf { x } _ { t } + \sqrt { \beta \Delta t } \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( \epsilon _ { t } ; 0 , I ) , } \end{array}\tag{1}
$$

for $\mathbf { x } _ { t } , \epsilon _ { t } \in \mathbb { R } ^ { d }$ . The reverse process is augmented by an additional term ${ \bf u } _ { t } ( { \bf x } _ { t } )$ for control:

Reverse process:

$$
\begin{array} { r } { \mathbf { x } _ { t - \Delta t } = \mathbf { x } _ { t } + \mathbf { u } _ { t } ( \mathbf { x } _ { t } ) \Delta t + \sqrt { \beta \Delta t } \ \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( \epsilon _ { t } ; 0 , I ) . } \end{array}\tag{2}
$$

The control term ${ \bf u } _ { t } ( { \bf x } _ { t } )$ incorporates neural networks in various forms, which are designed to reverse the diffusive flow of the forward process using the information learned during that process.

Score field: The diffusive flow induced by the diffusion of a density $\rho _ { t } ( \mathbf { x } )$ can be explained using the convolution with a small Gaussian kernel, $k _ { \Delta t } ( \mathbf { x } ) = \mathcal { N } ( \mathbf { x } ; 0 , \beta \dot { \Delta t } I )$ . The density after a small diffusion step $\Delta t$ can be written as $\begin{array} { r } { \rho _ { t + \Delta t } ( \mathbf { x } ) = \int k _ { \Delta t } ( \mathbf { x } , \vec { 0 } ) \rho _ { t } ( \mathbf { x } ) d \mathbf { x } ^ { \prime } \approx \rho _ { t } ( \mathbf { x } ) + \frac { \beta \Delta t } { 2 } \nabla ^ { 2 } \rho _ { t } } \end{array}$ , yielding

$$
\frac { d \rho _ { t } } { d t } = \operatorname* { l i m } _ { \Delta t  0 } \frac { \rho _ { t + \Delta t } ( \mathbf { x } ) - \rho _ { t } ( \mathbf { x } ) } { \Delta t } \approx \frac { \beta } { 2 } \nabla ^ { 2 } \rho _ { t } .\tag{3}
$$

Fick’s law, $J = - D \nabla \rho _ { t }$ , describes the diffusive flux $J ( = \rho _ { t } \mathbf { v } )$ driven by the density gradient, with a constant coefficient D. Along with the equation of continuity $\begin{array} { r } { \frac { \partial \rho _ { t } } { \partial t } = - \nabla \cdot \boldsymbol { J } , } \end{array}$ , we can produce the equation $\begin{array} { r } { \frac { \partial \rho _ { t } } { \partial t } = - \nabla \cdot J = D \nabla ^ { 2 } \rho _ { t } } \end{array}$ which corresponds to Eq. (3). The correspondence yields the constant $\begin{array} { r } { { D = { \frac { \beta } { 2 } } } } \end{array}$ . Then, from the definition of flux, $J = \rho _ { t } { \bf v } ,$ we can obtain the diffusive velocity field:

$$
\mathbf { v } ( \mathbf { x } ) = \frac { J } { \rho _ { t } } = - D \frac { \nabla \rho _ { t } } { \rho _ { t } } = - \frac { \beta } { 2 } \nabla \log \rho _ { t } ,\tag{4}
$$

where $\nabla$ log $\rho _ { t }$ is the score vector of a density $\rho _ { t } ( \mathbf { x } )$ . The score field in a pure diffusion system is proportional to the velocity field induced by diffusion.

For the forward process, along with that lim $\begin{array} { r } { \Delta t  0 . \frac { \mathbf { x } _ { t + \Delta t } - \mathbf { x } _ { t } } { \Delta t } = \operatorname* { l i m } _ { \Delta t  0 } \frac { \sqrt { 1 - \beta \Delta t } \mathbf { x } _ { t } - \mathbf { x } _ { t } } { \Delta t } + o ( \Delta t ) = } \end{array}$ $- \beta / 2 { \bf x } _ { t }$ , it induces a velocity field in the infinitesimal limit $\Delta t \to 0 ;$

$$
\mathbf { v } ( \mathbf { x } ) = - \frac { \beta } { 2 } \left( \mathbf { x } + \nabla \log \rho _ { t } \right) ,\tag{5}
$$

where the two terms arise from the corresponding components in the update rule in $\operatorname { E q . } \left( 1 \right)$ . In general, neural networks are designed to directly predict a score ∇ log $\rho _ { t }$ or the expected total accumulated noise $\epsilon _ { \mathrm { t o t } , t } .$ , which corresponds to $\epsilon _ { \mathrm { t o t } , t } = - \sigma _ { t } \nabla$ log $\rho _ { t }$ , with $\sigma _ { t } = \sqrt { 1 - e ^ { - \beta t } }$ or $\sigma _ { t } = \sqrt { 1 - ( 1 - \beta \Delta t ) ^ { t / \Delta t } }$ depending on continuous- or discrete-time formulation.

When the neural network overfits to approximate the exact reverse process, it learns the control field $\begin{array} { r } { \mathbf { u } _ { t } ( \mathbf { x } ) = \frac { \beta } { 2 } \mathbf { x } + \beta \nabla \log \rho _ { t } } \end{array}$ which induces the reverse-process velocity field: $\begin{array} { r } { \mathbf { u } _ { t } - \frac { \beta } { 2 } \nabla \log \rho _ { t } = - \mathbf { v } ( \mathbf { x } ) } \end{array}$ If the reverse process is performed without noise, i.e., $\mathbf { x } _ { t - \Delta t } = \mathbf { x } _ { t } + \mathbf { u } _ { t } ( \mathbf { x } _ { t } ) \Delta t$ without noise addition, the control field is simply $\begin{array} { r } { \mathbf { u } _ { t } ( \mathbf { x } ) = \frac { \beta } { 2 } ( \mathbf { x } + \nabla \log \rho _ { t } ) = - \mathbf { v } _ { t } } \end{array}$

## 2.1 Diffusion of δ-Functions

A surprising yet general property of the forward process emerges from deriving the induced velocity field associated with the diffusion of a Dirac δ-function. Consider a Gaussian kernel $k _ { t } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) =$ $\mathcal { N } ( \mathbf { x } ; \mathbf { x } ^ { \prime } , t \beta I )$ which satisfies the convolution property: $\begin{array} { r } { k _ { t + \Delta t } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \int k _ { \Delta t } ( \mathbf { x } ^ { \prime } ) k _ { t } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) d \mathbf { x } ^ { \prime } . } \end{array}$

The density recovers the Dirac δ-function in the limit lim $_ { t  0 } k _ { t } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \delta ( \mathbf { x } - \mathbf { x } ^ { \prime } )$ . The induced velocity field $\mathbf { v } _ { t } ( \mathbf { x } )$ associated with the diffusion satisfies the continuity equation:

$$
\nabla \cdot \Big ( { \mathbf v } ( { \mathbf x } ) k _ { t } ( { \mathbf x } , { \mathbf x ^ { \prime } } ) \Big ) = - \frac { \partial k _ { t } ( { \mathbf x } , { \mathbf x ^ { \prime } } ) } { \partial t } .\tag{6}
$$

with the flux $J ( \mathbf { x } ) = \mathbf { v } ( \mathbf { x } ) k _ { t } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ . Expanding both sides yields

$$
\left\{ \nabla \cdot \mathbf { v } - { \frac { \mathbf { x } - \mathbf { x } ^ { \prime } } { t \beta } } \cdot \mathbf { v } ( \mathbf { x } ) \right\} k _ { t } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = - { \frac { 1 } { 2 t } } \left( { \frac { ( \mathbf { x } - \mathbf { x } ^ { \prime } ) ^ { 2 } } { t \beta } } - d \right) k _ { t } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) ,\tag{7}
$$

The plug in of an equation of the form $\mathbf { v } ( \mathbf { x } ) \equiv ( \mathbf { x } - \mathbf { x } ^ { \prime } ) f ( t )$ for some scalar function $f ( t )$ , we obtain $\begin{array} { r } { f ( t ) { } = \frac { 1 } { 2 t } } \end{array}$ , which yields the velocity field associated with this vanilla diffusion:

$$
\boxed { { \bf v } ( { \bf x } ) = \frac { 1 } { 2 t } ( { \bf x } - { \bf x } ^ { \prime } ) . }\tag{8}
$$

Note that the velocity field "diverges" as $t \to 0$ everywhere, indicating that the diffusion of a Dirac δ-function induces an infinite initial velocity, which subsequently decays proportionally to $1 / t$ . Note that this pure diffusive property of the velocity does not depend even on the noise magnitude β.

Extensions of this single δ-function derivation to mixtures of δ-functions are straightforward. In this view, training samples can be interpreted as a mixture of Dirac delta functions. Since the velocity grows linearly with distance from each center, while Gaussian densities decay exponentially, the resulting flux is dominated by the Gaussian mixture structure. Consequently, the velocity field at any point is largely determined by the nearest center. As long as the diffused Gaussians are not sufficiently smeared, the singular behavior of the initial velocity field persists throughout the space. In diffusion models, when centers correspond to training samples, the velocity field diverges in the early stage of the forward process; moreover, due to $\mathbf { v } \propto \mathbf { \bar { x } } - \mathbf { \bar { x } } ^ { \prime }$ , this divergence is more pronounced farther from the centers than nearby, which may appear counterintuitive.

Our key inference for detecting memorization is as follows. Networks trained on finite samples in the forward process induce singularities in the early stage. The absence of such singularities in the final stage of the reverse process indicates that the network has not memorized.

## 3 Criterion for Detecting Memorization

Memorization has been identified by examining nearest neighbors of generated samples within the training dataset. Detecting duplication of training data via nearest-neighbor comparisons is considered a gold–standard indicator of memorization [3, 15, 16]. However, this approach requires access to the full training dataset, which limits its practical applicability.

Instead, memorization can be investigated through intrinsic properties of the learned neural networks. Because gold-standard indicators can, even in large datasets, unambiguously distinguish memorizing networks from non-memorizing ones, it is natural to seek corresponding signatures within trained networks. Existing approaches pursue this direction by introducing empirical proxies for memorization.

These approaches include guidance-based methods that identify the explosion of guidance (showing large magnitude in the difference between conditional and non-conditional scores) under the premise that such behavior sharpens the conditioned probability landscape [19], attention-based methods that identify anomalous late-stage attention behavior [4], and geometrical approaches that detect sharp and highly localized regions encoded in the Hessian of the log density [7]. Notably, these methods, to varying degrees, attempt to analytically relate their criteria to the norm of the score function, motivated by empirical observations of irregular score behavior in memorizing networks, which remain only partially understood. These works interpret the empirical growth of score magnitudes as a consequence of the criteria they propose as principled.

![](images/c45c94a4aa42f65f61aec62f78a65cf740cc7fddf0199f711bb0a91538339bcb.jpg)

![](images/4cd244a10852d35348afbf845d6e1ab8b4144eb6726dc18f9ef675dc69fe9f31.jpg)

![](images/949f7b55530182e546364dc8e0610983c6829d164cf2630b99305517b2004229.jpg)

![](images/1710672489b666b57494d3c8c698d928e55346a0c1a637a4a988a56288e23741.jpg)

![](images/07fa5c77363a3efcf5032682233197af2f32bd6de58ab5fec7822c22a5ad4bf2.jpg)  
Figure 1: 2D Gaussian generation. (A) The rate of Helmholtz free energy dissipation for an affine network, a small MLP, and a large MLP, plotted on a logarithmic scale over time steps. (B) K-L divergence between the generated samples and the true density as a function of training sample size. (C–E) Generated samples from the affine network, small MLP, and large MLP models, respectively. After sufficient training, the failure of the low-capacity model to capture free energy dissipation until a certain time leads to the successful reproduction of the underlying Gaussian structure.

On the contrary, we argue that the explosion of the score itself constitutes a central identifier of memorization. Specifically, within the diffusion model pipeline, memorization consistently occurs in conjunction with the explosion of the velocity over the entire space. Consequently, in the absence of large velocity values, memorization cannot occur. Therefore, the absence of large velocity values intuitively indicates non-memorization, while the converse does not necessarily hold. Furthermore, suppressing large velocity magnitudes—either explicitly or implicitly, e.g., via architectural constraints of neural networks or early stopping—will lead to the inhibition of memorization.

## 3.1 Helmholtz Free Energy Rate Criterion

We propose a criterion for assessing the non-memorization behavior of neural networks in diffusion models:

$$
\boxed { \Phi \equiv \frac { 1 } { T } \widehat { \frac { d F } { d t } } \Bigg \vert _ { t } = - \frac { \beta } { 2 } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \vert \vert \widehat { \mathbf { v } } _ { t } ( \mathbf { x } _ { t , i } ) \vert \vert ^ { 2 } . }\tag{9}
$$

The samples $\mathbf { x } _ { t , i }$ are the i-th sample at time t obtained during the reverse process, and the vector $\widehat { \mathbf { v } } _ { t } ( \mathbf { x } )$ is defined based on the output of the learned neural network. Specifically, we use its form $\widehat { \mathbf { v } } _ { t } ( \mathbf { x } ) \dot { = }$ $\mathbf { x } - \widehat { \epsilon } _ { \mathrm { t o t } , t } ^ { \theta } ( \mathbf { x } ) / \sigma _ { t }$ when the network $\widehat { \epsilon } _ { \mathrm { t o t } , t } ^ { \theta } ( \mathbf { x } )$ learns accumulated noise, and $\widehat { \mathbf { v } } _ { t } ( \mathbf { x } ) = \mathbf { x } + \widehat { \mathbf { s } } _ { t } ^ { \theta } ( \mathbf { x } )$ when $\widehat { \mathbf { s } } _ { t } ^ { \theta } ( \mathbf { x } )$ learns the score. F is the Helmholtz free energy which will be introduced later.

## 3.2 Behavior of the Criterion for Two-Dimensional Gaussian Generation

Previously, various explanations have been proposed for why diffusion models tend to generalize, including early stopping [2, 11], network architecture [8], and optimization methods such as stochastic gradient descent and learning rate [20]. We consider an affine neural network. In our example, the network takes the form

$$
\begin{array} { r } { { \bf s } _ { \theta } ( { \bf x } , t ) = A _ { \theta } ( t ) ( { \bf x } - \mu _ { \theta } ( t ) ) , } \end{array}\tag{10}
$$

where $A _ { \theta } ( t ) \in \mathbb { R } ^ { d \times d }$ is a symmetric matrix and $\mu _ { \theta } ( t ) \in \mathbb { R } ^ { d }$ is a vector. The $( d + 1 ) d / 2$ independent entries of $A _ { \theta } ( t )$ and the d entries of $\mu _ { \theta } ( t )$ are parameterized as outputs of neural networks. A simple derivation shows that this network recovers the score corresponding to a Gaussian with maximum-likelihood estimates of the mean and covariance. The full derivation is provided in Appendix A.1.

We hypothesize that the large multilayer perceptron (MLP) overfits and reproduces the training samples, the small MLP generates samples near the training data, and the aforementioned affine network produces Gaussian samples with the empirical mean and covariance. Using the forward and reverse processes, we analyze movement of criterion, or the dissipation rate of the Helmholtz free energy, induced by each model. For our synthetic experiment, we generated a small dataset $( N = 1 0 )$ from a Gaussian and trained diffusion models using an MLP with both large and small capacities, as well as an affine model.

We could expect that the forward diffusive field initialized from the 10 samples exhibits singular behavior in the initial velocities. As shown in Figure $1 ( \mathrm { A } )$ , the large MLP appears to have learned detailed trajectories, moving from $t = 1 { , } 0 0 0 \ \mathrm { t o } \ t = 0$ during the reverse process, including the singular behavior $( \Phi = - \infty )$ of the initial diffusion around $t = 0$

In contrast, the small MLP and the affine model do not follow this dissipation behavior; their dissipation rates stop decreasing around $t = 2 0 0$ . The affine model, being well-aligned with the true underlying density, produces samples that closely match the true density, as shown in the K-L divergence estimation in Figure 1(B). The generated samples in Figure 1(C–E) illustrate that the affine model produces Gaussian samples with the empirical mean and covariance of the 10 training points. The small MLP exhibits partial generalization beyond the training data, whereas the large MLP memorizes the training set, accompanied by a pronounced decay in the Helmholtz free energy dissipation rate.

In Figure 1(A), we also observe that Helmholtz free energy dissipation rate is negative and monotonically decreases throughout the reverse process. Including these properties, we next elucidate how the norm of the score is intimately connected to a thermodynamic quantity, shedding light on the mechanism that links large score magnitudes to a principled theoretical criterion.

## 3.3 Helmholtz Free Energy in Diffusion Models

The forward and reverse processes in diffusion can be interpreted within a thermodynamic framework, where their states are represented by the following thermodynamic functional:

$$
F [ \rho _ { t } ] = \langle U \rangle _ { \rho _ { t } } - T S = \int \rho _ { t } ( \mathbf { x } ) U ( \mathbf { x } ) d \mathbf { x } + T \int \rho _ { t } ( \mathbf { x } ) \log \rho _ { t } ( \mathbf { x } ) d \mathbf { x } .\tag{11}
$$

Here, $\rho _ { t } ( \mathbf { x } )$ denotes the probability density that changes over time, $U ( \mathbf { x } )$ represents the energy function, and $T$ denotes the fixed temperature in the system. The quantity $\langle U \rangle _ { \rho _ { t } }$ denotes the expectation of $U ( \mathbf { x } )$ with respect to $\rho _ { t }$ , and $F [ \rho _ { t } ]$ is the Helmholtz free energy with the Boltzmann constant is set to unity.

Given $U ( \mathbf { x } )$ , the thermodynamic law decreases $F [ \rho _ { t } ]$ over time by updating $\rho _ { t } ( \mathbf { x } )$ according to the equation of continuity. The rate of change of the Helmholtz free energy is represented using the change with respect to $\rho _ { t }$ by the chain rule:

$$
\frac { d F } { d t } = \left. \frac { \partial F } { \partial \rho _ { t } } , \frac { \partial \rho _ { t } } { \partial t } \right. = \int \frac { \partial F } { \partial \rho _ { t } } \frac { \partial \rho _ { t } } { \partial t } d \mathbf { x } ,\tag{12}
$$

where $\langle f _ { 1 } , f _ { 2 } \rangle$ denotes the functional inner product of two functions $f _ { 1 } ( \mathbf { x } )$ and $f _ { 2 } ( \mathbf { x } )$ , and both $\frac { \partial F } { \partial \rho _ { t } }$ and $\frac { \partial \rho _ { t } } { \partial t }$ are pointwise functions of x. Applying the equation of continuity and integration by parts, we obtain

$$
\int  \frac { \partial F } { \partial \rho _ { t } } | _ { \rho _ { t } } \frac { \partial \rho _ { t } } { \partial t } d \mathbf { x } \quad = \quad - \int  \frac { \partial F } { \partial \rho _ { t } } | _ { \rho _ { t } } \nabla \cdot ( \rho _ { t } \mathbf { v } ) d \mathbf { x }\tag{13}
$$

$$
\begin{array} { r l } { \mathbf { \Gamma } } & { { } = \left. \int \rho _ { t } \mathbf { v } ( \mathbf { x } ) ^ { \top } \nabla \frac { \partial F } { \partial \rho _ { t } } \right| _ { \rho _ { t } } d \mathbf { x } } \end{array}\tag{14}
$$

$$
\begin{array} { r l } { \mathrm {  ~ \Gamma ~ } = } & { { } \left. { \bf v } , \nabla \frac { \delta F } { \delta \rho _ { t } } \right. _ { \rho _ { t } } , } \end{array}\tag{15}
$$

where $\langle \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } \rangle _ { \rho _ { t } }$ denotes the expectation of ${ \mathbf v } _ { 1 } ( { \mathbf x } ) ^ { \top } { \mathbf v } _ { 2 } ( { \mathbf x } )$ under $\rho _ { t }$ . Given a fixed kinetic energy $\begin{array} { r } { K = \frac { 1 } { 2 } \int \rho _ { t } | | \mathbf { v } | | ^ { 2 } d \mathbf { x } } \end{array}$ , the velocity field that maximizes the reduction of Helmholtz free energy is $\begin{array} { r } { \mathbf { v } \propto - \nabla \frac { \delta F } { \delta \rho _ { t } } \bigg \vert _ { \rho _ { t } } } \end{array}$ , and the magnitude of the dissipation rate $\textstyle { \frac { d F } { d t } }$ becomes proportional to the kinetic energy. The theoretical significance of the free energy lies in the fact that its maximal rate explains the thermodynamic motion observed in natur de, analogous to the principle of least action.

Because $\begin{array} { r } { \frac { \partial F } { \partial \rho _ { t } } = U + T ( 1 + \log \rho _ { t } ) } \end{array}$ at each x, the velocity and the flux of the system are

$$
\begin{array} { c c l } { \mathbf { v ( x ) } } & { = } & { \displaystyle - \nabla \frac { \delta F } { \delta \rho _ { t } } \ : = \ : - \nabla U - T \nabla \log \rho _ { t } } \end{array}\tag{16}
$$

$$
\begin{array} { r l r } { J ( { \bf x } ) } & { = } & { \rho _ { t } ( { \bf x } ) { \bf v } ( { \bf x } ) = - \rho _ { t } ( { \bf x } ) \nabla U - T \nabla \rho _ { t } . } \end{array}\tag{17}
$$

For the correspondence to the velocity field of diffusion models in Eq. (5), we use the temperature $\begin{array} { r } { T = \frac { \beta } { 2 } } \end{array}$ , and the energy for control is $\begin{array} { r } { U ( \mathbf { x } ) = \frac { \beta } { 4 } \mathbf { x } ^ { 2 } } \end{array}$ from $\begin{array} { r } { \nabla U = \frac { \beta } { 2 } { \bf x } . } \end{array}$

The forward process behaves under a fixed control potential $\begin{array} { r } { U ( \mathbf { x } ) = \frac { \beta } { 4 } \mathbf { x } ^ { 2 } } \end{array}$ . At equilibrium, the probability flux vanishes, i.e., $J = - \rho _ { \mathrm { e q } } \nabla U - T \nabla \mathrm { \bar { \rho } _ { e q } } = 0$ . We can write the equation $\nabla \rho _ { \mathrm { e q } } =$ $- \frac { \nabla U } { T } \rho _ { \mathrm { e q } } .$ , which yields the equilibrium density after sufficient forward updates:

$$
\rho _ { \mathrm { e q } } ( \mathbf { x } ) \propto \exp ( - U ( \mathbf { x } ) / T ) = \exp \left( - \mathbf { x } ^ { 2 } / 2 \right) .\tag{18}
$$

The forward process therefore simply corresponds to the evolution of $\rho _ { t } ( \mathbf { x } )$ from a mixture of Dirac δ-functions to an isotropic single Gaussian with unit variance. The trajectories and velocities jointly maximize a single quantity, the rate of decrease of the Helmholtz free energy under the equation of continuity. The following summarizes the behavior of Helmholtz free energy rate during the forward and reverse processes.

Helmholtz free energy does not increase: With the velocity field $\begin{array} { r } { { \mathbf v } ( { \mathbf x } ) = - \nabla \frac { \delta F } { \delta \rho _ { t } } = - \nabla U - } \end{array}$ T∇ log $\rho _ { t }$ , the Helmholtz free energy is non-increasing in time, since $\begin{array} { r } { \frac { d F } { d t } = - \langle \mathbf { v } , \mathbf { v } \rangle _ { \rho _ { t } } \leq 0 } \end{array}$ from Eq. (15). When $\rho _ { t } ( \mathbf { x } )$ reaches the equilibrium density, the flux vanishes, and the free energy no longer decreases. Throughout the forward process, the free energy continuously decreases under a fixed potential $\begin{array} { r } { U ( \mathbf { x } ) = \frac { \beta } { \mathcal { A } } \mathbf { x } ^ { 2 } } \end{array}$ . During the reverse process, the potential becomes time-dependent to induce the control ${ \bf u } _ { t } ( { \bf x } ) { \bf \Psi }$ . Nevertheless, the rate of free energy dissipation remains unchanged for a fixed ${ \bf u } _ { t } ( { \bf x } )$ at time $t ,$ since the velocity field is identical up to a sign reversal.

The rate of change decreases over time during the forward process: The second derivative $\frac { \partial ^ { 2 } F } { d t ^ { 2 } }$ is positive, as implied by results on the dissipation of Fisher information in [18, 1] provided that the potential is convex. Under the convex potential $\begin{array} { r } { U ( \mathbf { x } ) = \frac { \beta } { 4 } \mathbf { x } ^ { 2 } } \end{array}$ , the initially large rate of change does not persist over time during the forward process. The large rate of change occurs only when t is small.

From these two theoretical properties, the proposed criterion is always negative, and its magnitude does not increase during the forward process. The absence of these properties in the criterion during the reverse process indicates that the network has not overfit the forward diffusive flow and can be used as a criterion for ensuring non-memorization.

In the experiments, we evaluate the effectiveness of the proposed criterion in detecting memorization in neural networks, as well as memorization at the level of individual samples. We also incorporate the criterion into the objective function to encourage the model to generate outputs without memorization.

## 4 Experiments

In this section, we evaluate the criterion proposed in Eq. (9) across various benchmark datasets and discuss the results. While the criterion is computed at every step of the reverse data-generating process using trained neural networks, in practical applications it can be evaluated at a single time step—specifically at the final stage of generation. We use pretrained networks for CIFAR-10 and CelebA-HQ, and then induce catastrophic corruption by fine-tuning them on a small number of samples. We also evaluate the individual score to assess the degree of memorization of each generated sample. Finally, we incorporate the proposed criterion into the training loss and examine whether it mitigates memorization while preserving the quality of generated samples. Details of the experiment are explained in Appendix A.

![](images/c890b290b9590fbce0aba96c1b92e3f621b2f9cee44c7d25011549a101caf223.jpg)

![](images/f611707385d89016659b14d15b44a76ebf7d9aab481b8f11465f6079ec45cab1.jpg)  
Figure 2: CIFAR-10 pixel-space fine-tuning. (A) Catastrophic corruption. Pretrained networks are fine-tuned using a fixed subset of size $N _ { s }$ . (B) Non-catastrophic corruption. Pretrained networks are fine-tuned using a fixed subset of size $N _ { s } = 1 0 0$ , with occasional training on additional samples from the full dataset.

![](images/fe83ea6c15ae530017eb7029b40fc1100dfed1f794dd109e9a4e66354f279b69.jpg)

![](images/7ed4e164be01519f039b7383f4bee4ec18559bfed3b25ce86158af131020cc49.jpg)  
Figure 3: CelebA-HQ latent-space fine-tuning. (A) Catastrophic corruption. (B) Non-catastrophic corruption with occasional training on additional samples from the full dataset.

## 4.1 Catastrophic and Non-Catastrophic Corruptions

Pretrained networks for real-image generation were used: a CIFAR-10 pixel-space diffusion model [10] and a CelebA-HQ latent diffusion model [12, 9, 14]. These models were fine-tuned using subsets of the training data. For the catastrophic corruption experiment, we selected subsets of varying sizes $N _ { s }$ and performed 50,000 fine-tuning epochs using only those subsets. For the non-catastrophic corruption setting, we fixed the subset of sample size $\dot { N _ { s } } = 1 0 0$ . After τ steps of training on this subset, we randomly sampled 5,000 additional data from the full dataset and performed one epoch of fine-tuning using these samples. We evaluated the criterion Φ throughout the reverse process in Figure 2 and 3, where darker curves correspond to more memorizable setting. The CIFAR-10 model operates in pixel space, whereas the CelebA-HQ model operates in latent space. Note that the time scale in all graphs is logarithmic.

In these experiments, nearest neighbor LPIPS scores for the generated samples are reported in Table 1. Entries indicate the percentage of 1,024 generated samples whose nearest training image falls below a specified LPIPS threshold. CIFAR-10 uses LPIPS ≤ 0.03 to avoid false positives due to low resolution, while CelebA-HQ uses $\mathrm { L P I P S } \leq 0 . 1$ . Smaller N and larger τ lead to increased memorization during fine-tuning. Representative generated examples and their nearest neighbors are shown in Figure 4. In the catastrophic corruption setting, all generated samples are memorized when $N _ { s } = 1 0 0$ , whereas no samples exhibit memorization when $N _ { s } = 2 0 \small { , } 0 0 0$ . In non-catastrophic setting, both memorized and non-memorized samples appear; however, the parameter τ explains the proportion of memorization. These observations are consistent with the criterion at $t = 0$ across all cases in Figure 2 and 3. These observations regarding the extent of memorization are consistent with the behavior of the criterion near $t = 0$ in these figures.

## 4.2 Sample-Level Memorization

We evaluated the score norm obtained as a byproduct of sample generation. The criterion was calculated using the conditional and unconditional velocities, $\widehat { \mathbf { v } } _ { \mathrm { c o n d } } ( \mathbf { x } )$ and $\widehat { \mathbf { v } } _ { \mathrm { u n c o n d } } ( \mathbf { x } )$ , respectively, at the end of the reverse-process trajectory. This sample-level metric quantifies the magnitude of the prompt-conditioned score via $| | \widehat { \mathbf { v } } _ { \mathrm { c f g } } ( \mathbf { x } _ { t } , t ) | | ^ { 2 } - | | \widehat { \mathbf { v } } _ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) | | ^ { 2 }$ where $\widehat { \mathbf { v } } _ { \mathrm { c f g } } ( \mathbf { x } _ { t } , t ) =$ $\widehat { \mathbf { v } } _ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) + w ( \widehat { \mathbf { v } } _ { \mathrm { c o n d } } ( \mathbf { x } _ { t } , t ) - \widehat { \mathbf { v } } _ { \mathrm { u n c o n d } } ( \mathbf { x } _ { t } , t ) )$ , and w is classifier-free guidance scale. Given a score $\mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) , \widehat { \mathbf { v } } ( \mathbf { x } _ { t } , t ) = x _ { t } + \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t )$

Table 1: Memorization with fine tuning. Percentage of memorized samples during fine-tuning whose nearest neighbor in the training set falls below the LPIPS threshold. $N _ { s }$ denotes the subset size used in catastrophic corruption, and τ denotes the number of consecutive epochs for which the $N _ { s } = 1 0 0$ subset is used during fine-tuning.
<table><tr><td colspan="3">Catastrophic Corruption</td><td colspan="3">Non-Catastrophic Corruption</td></tr><tr><td> $N _ { s }$ </td><td>CIFAR-10</td><td>CelebA-HQ</td><td>T</td><td>CIFAR-10</td><td>CelebA-HQ</td></tr><tr><td>100</td><td>99.80%</td><td>57.71%</td><td>0</td><td>0.00%</td><td>0.00%</td></tr><tr><td>200</td><td>94.53%</td><td>8.01%</td><td>10</td><td>0.78%</td><td>0.00%</td></tr><tr><td>1,000</td><td>3.71%</td><td>0.00%</td><td>20</td><td>5.08%</td><td>0.00%</td></tr><tr><td>2,000</td><td>0.78%</td><td>0.00%</td><td>50</td><td>18.95%</td><td>6.35%</td></tr><tr><td>10,000</td><td>0.49%</td><td>0.00%</td><td>100</td><td>32.52%</td><td>25.88%</td></tr><tr><td>20,000</td><td>0.39%</td><td>0.00%</td><td>一</td><td>一</td><td>一</td></tr></table>

![](images/bf0868f91b20b6f04e9b950776a4133ea4d3d5da6a884f1f286e388a3a41ec5c.jpg)  
Figure 4: Generated samples from fine-tuned networks and their nearest neighbors. The first two rows show CIFAR-10 generated samples and their nearest neighbors in the training data, while the next two rows show CelebA-HQ generated samples and their nearest neighbors. We present samples obtained after catastrophic corruption with $N _ { s } = 1 0 0$ and $N _ { s } = 2 0 \small { , } 0 0 0$ , as well as after noncatastrophic corruption with subset training for $\tau = 1 0 0$ and $\tau = 1 0$ consecutive epochs. Ordered from least to most memorization, the settings are $N _ { s } = 2 0 , 0 0 0 , \tau = 1 0 , \tau = 1 0 0$ , and $N _ { s } = 1 0 0$

We compare the sample-level detection performance of this criterion with the score-difference norm [19] and Hessian-sharpness baseline [7]. Memorized samples are defined as samples generated by controlled prompts for the purpose of memorization. The results in Table 2 show that the norm of the individual score serves as a reliable indicator for identifying safe generation without memorization. The experiments were conducted using Stable Diffusion v1.4 [14].

## 4.3 Loss Function Modification

We examined the incorporation of the criterion into the loss function to assess its ability to improve fidelity without inducing memorization. With a threshold $\begin{array} { r } { R _ { t } = ( \beta _ { t } / 2 ) \mathbb { E } _ { \mathrm { p o i n t } } [ \Vert \widehat { \mathbf { v } } _ { t } \Vert _ { 2 } ^ { 2 } ] \simeq \frac { \beta _ { t } ( 1 - \sigma _ { t } ^ { 2 } ) } { 2 \sigma _ { \star } ^ { 2 } } d , } \end{array}$ we penalize the norm along the trajectory for points whose values exceed $\gamma R _ { t }$ , where $0 < \gamma \leq$ 1 is a control parameter. This discourages the neural network from learning excessively large scores. Here, d denotes the dimensionality, and $\sigma _ { t } ^ { 2 }$ is the accumulated variance defined as $\sigma _ { t } ^ { 2 } =$ $\begin{array} { r } { \bigg ( 1 - \prod _ { s = 0 } ^ { t - 1 } ( 1 - \beta _ { t } ) \bigg ) } \end{array}$ , where $\beta _ { t }$ denotes the noise schedule $\beta$ at each time step t.

In the CelebA-HQ non-catastrophic corruption setting described in Section 4.1, the latent diffusion model is fine-tuned for 50,000 steps starting from a pretrained model with a subset-to-full update ratio of $\tau = 1 0 0$ . For samples whose score norm exceeds $\gamma R _ { t } ,$ the algorithm suppresses the norm, while fidelity improvements are driven only by samples with small score norms. Let $s \subset \tau$ denote the subset of training data $\tau$ satisfying $\frac { \beta _ { t } } { 2 } \| \widehat { \mathbf { v } } _ { t } ( \mathbf x _ { t , i } ) \| _ { 2 } ^ { 2 } > \gamma R _ { t }$ . With corresponding small λ, the standard diffusion loss $\mathcal { L } _ { \epsilon } ( \mathbf { x } _ { t , i } , t )$ , the training objective at time step t is modified as

$$
\mathcal { L } _ { \mathrm { t r a i n } } ( t ) = \frac { 1 } { N } \left[ \lambda \sum _ { \mathbf { x } _ { i } \in S } \left( \frac { \beta _ { t } } { 2 } \left. \widehat { \mathbf { v } } _ { t } ( \mathbf { x } _ { t , i } ) \right. _ { 2 } ^ { 2 } - \gamma R _ { t } \right) + \sum _ { \mathbf { x } _ { i } \in T \backslash S } \mathcal { L } _ { \epsilon } ( \mathbf { x } _ { t , i } , t ) \right] .\tag{19}
$$

Table 2: Sample-level detection on SD v1.4. The generated samples from memorizing prompts and non-memorizing prompts are discriminated by various memorization criteria. All measurements are performed on DDPM 1,000-step trajectories. Bold entries are statistically indistinguishable from the best value in each column under an approximate two-sided test. Time step 1,000 corresponds to the beginning of diffusion, and time step 1 to its end.
<table><tr><td>Method</td><td>Time step</td><td>AUC</td><td>TPR@1%FPR</td></tr><tr><td> $\left\| \mathbf { s } _ { \theta } ^ { \Delta } ( \mathbf { x } _ { t } ) \right\|$ </td><td>1,000</td><td> $0 . 9 3 7 \pm 0 . 0 1 2$ </td><td> $0 . 7 7 1 \pm 0 . 0 2 7$ </td></tr><tr><td> $= \left\| \mathbf { s } _ { \theta , \mathrm { c o n d } } - \mathbf { s } _ { \theta , \mathrm { u n c o n d } } \right\| [ 1 9 ]$ </td><td>1</td><td> $0 . 9 7 2 \pm 0 . 0 0 5$ </td><td> ${ \bf 0 . 9 0 9 \pm 0 . 0 2 0 }$ </td></tr><tr><td> $\left\| { \cal H } _ { \theta } ^ { \Delta } ( { \bf x } _ { T } ) { \bf s } _ { \theta } ^ { \Delta } ( { \bf x } _ { T } ) \right\| ^ { 2 } [ 7 ]$ </td><td>1,000</td><td> $0 . 9 6 2 \pm 0 . 0 0 8$ </td><td> $0 . 8 5 6 \pm 0 . 0 2 4$ </td></tr><tr><td></td><td>1</td><td> $0 . 9 7 0 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 8 7 4 \ : \pm 0 . 0 2 1 }$ </td></tr><tr><td> $\| \widehat { \mathbf { v } } _ { \mathrm { c f g } } \| ^ { 2 } - \| \widehat { \mathbf { v } } _ { \mathrm { u n c o n d } } \| ^ { 2 } \left( \mathrm { O u r s } \right)$ </td><td>1</td><td> ${ \bf 0 . 9 8 2 \pm 0 . 0 0 3 }$ </td><td> ${ \bf 0 . 9 0 5 \pm 0 . 0 1 9 }$ </td></tr></table>

Table 3: CelebA-HQ fine-tuning with velocity norm suppression. Norm-suppressing penalty replaced the fine-tuning objective once the norm exceeds a threshold. Suppression is applied to $\widehat { \mathbf { v } } _ { t } = x _ { t } + \mathbf { s } _ { \theta } ( x _ { t } , t )$ and to the score $\mathbf { s } _ { \theta } ( x _ { t } , t )$ alone.
<table><tr><td>Condition</td><td> $\mathrm { L P I P S } \leq 0 . 1$ </td><td colspan="2"> $\mathrm { L P I P S } \le 0 . 3$ </td><td colspan="2">FID</td></tr><tr><td>Pretrained model No reference scale</td><td>0.00% 28.53%</td><td colspan="2">0.08% 33.11%</td><td colspan="2">14.93 24.33</td></tr><tr><td></td><td colspan="2">Full velocity  $\widehat { \mathbf { v } } _ { t } = x _ { t } + \mathbf { s } _ { \theta } ( x _ { t } , t )$ </td><td colspan="3"></td></tr><tr><td>Condition</td><td> $\mathrm { L P I P S } \leq 0 . 1$   $\mathrm { L P I P S } \leq 0 . 3$ </td><td> $\mid \lambda = 1 0 ^ { - 3 }$ </td><td>Score-only</td><td> $\widehat { \mathbf { v } } _ { t } = \mathbf { s } _ { \theta } ( x _ { t } , t )$ </td><td> $| \lambda = 1 0 ^ { - 5 }$ </td></tr><tr><td></td><td></td><td>FID</td><td> $\mathrm { L P I P S } \leq 0 . 1$ </td><td> $\mathrm { L P I P S } \leq 0 . 3$ </td><td>FID</td></tr><tr><td> $\gamma = 1 . 0$ </td><td>31.85% 36.79%</td><td>26.19</td><td>23.14%</td><td>31.59%</td><td>22.71</td></tr><tr><td> $\gamma = 0 . 8$ </td><td>11.39%</td><td>28.14% 17.99 15.88</td><td>2.47%</td><td>18.01%</td><td>53.80</td></tr><tr><td> $\gamma = 0 . 7$ </td><td>3.50%</td><td>20.65%</td><td>1.46%</td><td>15.75%</td><td>25.74</td></tr><tr><td> $\gamma = 0 . 5$ </td><td>0.05%</td><td>15.31% 14.38</td><td>0.29%</td><td>2.31%</td><td>156.08</td></tr><tr><td> $\gamma = 0 . 3$ </td><td>0.00%</td><td>3.80% 15.87</td><td>0.02%</td><td>0.23%</td><td>295.58</td></tr></table>

Here, the term $\begin{array} { r } { \lambda \big ( \frac { \beta _ { t } } { 2 } \left\| \widehat { \mathbf { v } } _ { t } ( \mathbf { x } _ { t , i } ) \right\| _ { 2 } ^ { 2 } - \gamma R _ { t } \big ) } \end{array}$ acts as a norm-suppressing penalty. Smaller values of γ place greater emphasis on norm suppression.

As shown in Table 3, norm suppression effectively reduces nearest-neighbor replication in the repeated-sample fine-tuning setting. While it mitigates memorization, it does not hinder the improvement of the quality of generated images. For the full-velocity suppression objective, with $\gamma = 0 . 5 .$ , the replication rate drops from 28.53% to 0.05%, while the FID improves from 24.33 to 14.38. Further suppression with $\gamma = 0 . 3$ continues to reduce replication to 0.00%, but at the cost of slightly degraded fidelity. Learning with score penalizing compromises fidelity and fails to improve both LPIPS and FID.

## 4.4 Large-Scale Memorization Mitigation

We further evaluate the proposed regularization method on the large-scale LAION-Aesthetics-12M-UMAP dataset using Stable Diffusion v1.4. We compare our method with the memorization mitigation approach of Wen et al. [19]. Memorization is evaluated using the fraction of generated samples with SSCD similarity exceeding 0.7, while generation fidelity is assessed using CLIP scores.

Table 4: Memorization mitigation on LAION-Aesthetics-12M-UMAP. Comparison of regularization methods on Stable Diffusion v1.4 using 200 generated samples. ∆CLIP denotes the change in CLIP score relative to the unmitigated model.
<table><tr><td>Method</td><td>SSCD &gt; 0.7</td><td>Count</td><td>CLIP</td><td>∆CLIP</td></tr><tr><td>No mitigation</td><td>39.00%</td><td>78</td><td>0.3019</td><td></td></tr><tr><td>Wen  $( \tau = 2 ) [ 1 9 ]$ </td><td>0.00%</td><td>0</td><td>0.2917</td><td>-0.0102</td></tr><tr><td>Wen (τ = 4)</td><td>0.50%</td><td>1</td><td>0.2978</td><td>-0.0041</td></tr><tr><td>Ours  $( \gamma = 0 . 5 , \lambda = 1 0 ^ { - 3 } )$ </td><td>0.00%</td><td>0</td><td>0.2987</td><td>-0.0032</td></tr></table>

As shown in Table 4, the unmitigated model produces 78 memorized samples out of 200 generations (39.00%). Our method reduces the observed memorization rate to 0.00%, matching Wen et al. (τ = 2), while achieving a higher CLIP score (0.2987 vs. 0.2917). Furthermore, our method incurs a smaller reduction in CLIP score (-0.0032) than either configuration of Wen et al. These results suggest that the proposed regularization can effectively mitigate memorization while better preserving generation fidelity in this evaluation.

## 5 Conclusion

We introduced a thermodynamic diagnostic for model-level memorization in diffusion networks. The proposed criterion is based on a thermodynamic quantity—the rate of change of the Helmholtz free energy—computed from the learned score. Across a range of settings exhibiting memorization and non-memorization, networks for which this criterion does not diverge do not produce memorized samples. The proposed criterion achieves competitive performance compared with existing methods and can further mitigate memorization when incorporated into the training loss.

## References

[1] D. Bakry and M. Émery. Diffusions hypercontractives. In Séminaire de Probabilités XIX 1983/84, volume 1123 of Lecture Notes in Mathematics, pages 177–206. Springer, Berlin, Heidelberg, 1985.

[2] T. Bonnaire, R. Urfin, G. Biroli, and M. Mézard. Why diffusion models don’t memorize: The role of implicit dynamical regularization in training. In Advances in Neural Information Processing Systems, 2025.

[3] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramèr, B. Balle, D. Ippolito, and E. Wallace. Extracting training data from diffusion models. In Proceedings of the 32nd USENIX Security Symposium, pages 5253–5270, 2023.

[4] C. Chen, D. Liu, M. Shah, and C. Xu. Exploring local memorization in diffusion models via bright ending attention. In International Conference on Learning Representations, 2025.

[5] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[6] A. Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6:695–709, 2005.

[7] D. Jeon, D. Kim, and A. No. Understanding and mitigating memorization in generative models via sharpness of probability landscapes. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 27091–27112, 2025.

[8] Z. Kadkhodaie, F. Guth, E. P. Simoncelli, and S. Mallat. Generalization in diffusion models arises from geometry-adaptive harmonic representations. In International Conference on Learning Representations, 2024.

[9] T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.

[10] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[11] P. Li, Z. Li, H. Zhang, and J. Bian. On the generalization properties of diffusion models. In Advances in Neural Information Processing Systems, volume 36, pages 2097–2127, 2023.

[12] Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, pages 3730–3738, 2015.

[13] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

[14] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[15] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6048–6058, 2023.

[16] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein. Understanding and mitigating copying in diffusion models. In Advances in Neural Information Processing Systems, volume 36, pages 47783–47803, 2023.

[17] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[18] C. Villani. Optimal Transport: Old and New, volume 338 of Grundlehren der Mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 2009.

[19] Y. Wen, Y. Liu, C. Chen, and L. Lyu. Detecting, explaining, and mitigating memorization in diffusion models. In International Conference on Learning Representations, 2024.

[20] Y.-H. Wu, P. Marion, G. Biau, and C. Boyer. Taking a big step: Large learning rates in denoising score matching prevent memorization. In Proceedings of the 38th Conference on Learning Theory, volume 291 of Proceedings ofMachine Learning Research, pages 5718–5756, 2025.

## Appendix

## A Experimental Details

The details of the experiment are as follows: Figure $1 ( \mathbf { A } ) , 2 .$ , and 3 provides the plot of $\begin{array} { r } { \Phi = \frac { 1 } { T } \frac { d F } { d t } } \end{array}$ at each t calculated using $\mathbf { v } _ { t } ( \mathbf { x } _ { t } ) = \mathbf { x } + \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t )$ . The experiments have been performed with the network estimating the total accumulated error $\epsilon _ { \mathrm { t o t } , t } = \mathbb { E } _ { \rho _ { t } ( \mathbf { x } , \mathbf { x } _ { t } ) } [ \mathbf { x } _ { t } - \mathbf { x } ]$ with training sample x and the forward diffused sample $\mathbf { x } _ { t }$ at time $t .$ The neural network in our experiments estimates $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t )$ as in [5], and the learned score is obtained using $\mathbf { s } _ { \theta } ( x _ { t } , t ) = - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) / \sigma _ { t }$ with $\sigma _ { t } = \sqrt { 1 - e ^ { - \beta t } }$ or $\sigma _ { t } = \sqrt { 1 - ( 1 - \beta \Delta t ) ^ { t / \Delta t } }$ for discrete algorithm. In all experiments, the diffusion timestep axis is shown on a logarithmic scale to enhance the visibility of the low-noise region near $t = 0$

The reverse process begins at $t = 1 \small { , } 0 0 0$ in our experiments, and data are generated at $t = 0$ . The low-noise region lies near $t = 0$ , where noise has accumulated through the forward process.

## A.1 Two-Dimensional Gaussian Details

Affine network learns scores that generate Gaussian data: The derivation below shows that this network recovers the score corresponding to a Gaussian with maximum-likelihood estimates of the mean and covariance.

We consider the following score matching objective function:

$$
L ( \mathbf { s } ) = \sum _ { i = 1 } ^ { N } \frac { 1 } { 2 } \mathbf { s } ( \mathbf { x } _ { i } ) ^ { 2 } + \nabla \cdot \mathbf { s } ( \mathbf { x } _ { i } ) ,\tag{20}
$$

which is a modified expression using the integration by parts from the mean squared error between the estimated score and the true score [6, 17]. Consider an affine score function $\mathbf { s } ( \mathbf { x } , t ) = A ( \mathbf { x } - \mathbf { m } )$ where $\mathbf { s } ( \mathbf { x } , t ) \in \mathbb { R } ^ { d } , \mathbf { x } \in \mathbb { R } ^ { d }$ , and $t ~ \in ~ \mathbb { R }$ . Plugging $\mathbf { s } ( \mathbf { x } ) = A ( \mathbf { x } - \mathbf { m } )$ into L yields $L =$ $\begin{array} { r } { \sum _ { i } \frac { 1 } { 2 } ( \mathbf x - \mathbf m ) ^ { \top } A A ( \mathbf x - \mathbf m ) + t r [ A ] } \end{array}$ . The optimal A and m that minimize $L ,$ obtained from the conditions $\begin{array} { r } { \frac { \partial L } { \partial A } = 0 } \end{array}$ and $\begin{array} { r } { \frac { \partial L } { \partial \mathbf { m } } = 0 , } \end{array}$ , are given in closed form as

$$
A _ { * } = - \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \mathbf { x } _ { i } - \mathbf { m } ) ( \mathbf { x } _ { i } - \mathbf { m } ) ^ { \top } \right] ^ { - 1 } \equiv - \widehat { \Sigma } ^ { - 1 } , \quad \mathbf { m } _ { * } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { x } _ { i } \equiv \widehat { \mu } ,\tag{21}
$$

where $\widehat { \mu }$ and $\widehat { \Sigma }$ denote the empirical mean and covariance, respectively. The resulting score corresponds to that of a Gaussian with mean $\widehat { \mu }$ and covariance ${ \widehat { \Sigma } } ,$ since the Gaussian score is ∇ log $\rho = - \Sigma ^ { - 1 } ( { \bf x } - { \boldsymbol \mu } )$ for $\rho ( \mathbf { x } ) = \mathcal { N } ( \mathbf { x } ; \mu , \Sigma )$ . Therefore, Langevin dynamics using an affine score model generates Gaussian data with the empirical mean and covariance of the training data, regardless of whether the original data are Gaussian. ■

The experiment was conducted on a two-dimensional Gaussian distribution with zero mean. The covariance matrix is constructed by rotating a diagonal matrix with entries (4.0, 0.25) by $3 5 ^ { \circ }$ . We use $N = 1 0$ samples from this distribution for training. Subsequently, the Kullback–Leibler (K-L) divergence is computed using the generated samples and the analytical form of the underlying density.

We use 1,000 time steps for both the forward and reverse processes, with a constant $\beta$ of 0.0095. Each model is trained for 20,000 optimization steps using AdamW, an Adam variant with decoupled weight decay [13], with the standard ϵ-prediction loss, a batch size of 256, and a learning rate of $1 0 ^ { - 3 }$ . Figure 1 presents a comparison of model capacity for $N = 1 0$ between an affine denoiser, a small MLP, and a large MLP.

The affine denoiser uses a 3-layer time MLP with 64 hidden nodes and has 10,757 trainable parameters. The small MLP has $\dot { 2 }$ hidden layers with 28 hidden nodes, resulting in 1,850 trainable parameters, whereas the large MLP has 6 hidden layers with 1,024 hidden nodes and 5,285,890 trainable parameters. Thus, the small MLP has limited capacity, while the large MLP has sufficient capacity to form sample-specific representations around the finite training set.

The small and large MLP denoisers share the same input variables but differ in capacity:

$$
\epsilon _ { \theta } ^ { \mathrm { M L P } } ( \mathbf { x } _ { t } , t ) = \mathrm { M L P } _ { \theta } \left( \left[ \mathbf { x } _ { t } , \gamma ( t ) \right] \right) ,\tag{22}
$$

where $\gamma ( t )$ is a sinusoidal time embedding.

The affine denoiser has the following formulation:

$$
\epsilon _ { \theta } ^ { \mathrm { a f f i n e } } ( \mathbf { x } _ { t } , t ) = A _ { \theta } ( t ) \left( \mathbf { x } _ { t } - m _ { \theta } ( t ) \right) ,\tag{23}
$$

where $A _ { \theta } ( t )$ is a symmetric matrix and both $A _ { \theta } ( t )$ and $m _ { \theta } ( t )$ are produced from the time embedding.   
Score has been obtained using $s _ { \theta } ( \mathbf x _ { t } , t ) = - \epsilon _ { \theta } ^ { \mathrm { a f f i n e } } ( \mathbf x _ { t } , t ) / \sigma _ { t }$ , Eq. (23).

In Figure 1(B), we compare the generated distribution of each model with the true underlying Gaussian density. For each trained model, we first generate samples from the model distribution p<sub>θ</sub> and fit a Gaussian KDE, denoted by $\hat { p } _ { \theta } ^ { \mathrm { K D E } }$ , to these samples. We then draw evaluation samples $\left\{ { \bf x } _ { i } ^ { \mathrm { e v a l } } \right\} _ { i = 1 } ^ { N }$ from the fitted KDE and compute the following Monte Carlo estimate of the divergence to the true density:

$$
\widehat { \mathrm { K L } } \left( \widehat { p } _ { \theta } ^ { \mathrm { K D E } } \parallel p _ { \mathrm { t r u e } } \right) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \log \widehat { p } _ { \theta } ^ { \mathrm { K D E } } \left( \mathbf { x } _ { i } ^ { \mathrm { e v a l } } \right) - \log p _ { \mathrm { t r u e } } \left( \mathbf { x } _ { i } ^ { \mathrm { e v a l } } \right) \right] , \qquad \mathbf { x } _ { i } ^ { \mathrm { e v a l } } \sim \widehat { p } _ { \theta } ^ { \mathrm { K D E } } .\tag{24}
$$

## A.2 Real-World Experiment Using Benchmark Dataset

For CIFAR-10 and CelebA-HQ, fine-tuning updates the pretrained model for 50,000 steps using a subset of the data. This subset is repeatedly presented to the model to promote memorization. Under the subset-to-full update ratio setting, 5,000 randomly sampled images from the full dataset are intermittently introduced every τ steps. The CIFAR-10 experiment measures statistics in RGB pixel space, whereas the CelebA-HQ experiment uses the latent denoising space provided by the pretrained model. The resolution of CIFAR-10 is 32 × 32, while that of CelebA-HQ is $2 5 6 \times 2 5 6$ After fine-tuning, nearest-neighbor LPIPS and FID are computed using the full training dataset.

## B Samples of Generated Images

We present examples of generated samples from the real-world experiments. Settings that promote memorization are shown on the left, while settings that discourage memorization are shown on the right, for both the CIFAR-10 in Figure 5 and CelebA-HQ in Figure 6. Red boxes indicate images with LPIPS scores below the memorization threshold used in Table 1: $\mathrm { L P I P S } \leq 0 . 0 3$ for CIFAR-10 and $\mathrm { L P I P S } \leq 0 . 1$ 1 for CelebA-HQ.

![](images/84f5f9ba599fa6b768855934d14545a76ed3bb8a9ab6f06d3a0c6fe9ff7a3100.jpg)  
Memorized: N = 100  
Less Memorized: N = 20,000  
Figure 5: CIFAR-10 generated examples. Top: catastrophic corruption conditions with learned subset size $N _ { s } .$ Bottom: non-catastrophic corruption conditions with subset-to-full update ratio τ. Red boxes indicate that the generated samples are flagged as nearest-neighbor replication under the criterion $\mathrm { L P I P S } \leq 0 . 0 3$

In $N _ { s } = 1 0 0$ experiment, all samples in CIFAR-10 and many samples in CelebA-HQ are highlighted with red boxes, indicating severe memorization by the networks. Under the non-catastrophic corruption setting with $\tau = 1 0 0$ , many examples are also red-boxed, although they do not constitute the majority. In contrast, for the non-memorizing settings $( N _ { s } = 2 0 , 0 0 0 { \bar { \mathrm { a n d } } } \bar { \tau } = 0 )$ , none of the generated samples exhibit memorization. These observations are consistent with the norm of the rate of Helmholtz free energy dissipation shown in Figures 2 and 3.

![](images/e345a7eb54f5d4d7cf88b8264aeefeec7b71357d81f08fda3595b0c8fb0387dc.jpg)  
Memorized: N = 100  
Less Memorized: N = 20,000  
Figure 6: CelebA-HQ generated examples. Top: catastrophic corruption conditions with learned subset size N . Bottom: non-catastrophic corruption conditions with subset-to-full update ratio τ. Red boxes indicate that the generated samples are flagged as nearest-neighbor replication under the criterion LPIPS ≤ 0.1.

## C Sample-Level Memorization in Stable Diffusion 2 Base

We present sample-level memorization detection results on Stable Diffusion 2 Base. The same 1,000- step DDPM measurement protocol used in Section 4.2 is applied. As in the Stable Diffusion v1.4 experiments, memorization labels are assigned based on prompts categorized as memorization-prone or non-memorization-prone, rather than verified replication of training images.

Table 5: Sample-level memorization detection on Stable Diffusion 2 Base. All measurements are performed on DDPM 1,000-step trajectories. Bold entries are statistically indistinguishable from the best value in each column under an approximate two-sided test at $p = 0 . 0 5$ using the reported standard errors.
<table><tr><td>Method</td><td>Time step</td><td>AUC</td><td>TPR@1%FPR</td></tr><tr><td> $\left\| \mathbf { s } _ { \theta } ^ { \Delta } ( \mathbf { x } _ { t } ) \right\|$ </td><td>1,000</td><td> $0 . 9 7 7 \pm 0 . 0 0 2$ </td><td> $0 . 8 2 2 \pm 0 . 0 1 0$ </td></tr><tr><td> $\mathbf { \xi } = \| \mathbf { s } _ { \theta , \mathrm { c o n d } }$  − Sθ,uncond∥ [19]</td><td>1</td><td> $\mathbf { 0 . 9 9 6 \mathop { \pm 0 . 0 0 0 } }$ </td><td> ${ \bf 0 . 9 6 7 \pm 0 . 0 0 2 }$ </td></tr><tr><td> $\left\| { \cal H } _ { \theta } ^ { \Delta } ( { \bf x } _ { T } ) { \bf s } _ { \theta } ^ { \Delta } ( { \bf x } _ { T } ) \right\| ^ { 2 } [ 7 ]$ </td><td>1,000</td><td> $0 . 9 6 9 \pm 0 . 0 0 2$ </td><td> $0 . 5 6 2 \pm 0 . 0 2 8$ </td></tr><tr><td></td><td>1</td><td> $0 . 8 9 7 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td> $\| \widehat { \mathbf { v } } _ { \mathrm { c f g } } \| _ { 2 } ^ { 2 } - \| \widehat { \mathbf { v } } _ { \mathrm { u n c o n d } } \| _ { 2 } ^ { 2 } \left( \mathrm { O u r s } \right)$ </td><td>1</td><td> $0 . 9 8 8 \pm 0 . 0 0 1$ </td><td> $0 . 9 2 7 \pm 0 . 0 0 6$ </td></tr></table>

## D Derivation of the Threshold for the Loss Function Modification

In this section, we derive the threshold $R _ { t }$ used in Section 4.3. The threshold is chosen as the value of the penalized quantity $\frac { \beta _ { t } } { 2 } \| \widehat { \mathbf { v } } _ { t } \| _ { 2 } ^ { 2 }$ when the network has fully memorized the training data, i.e., when the learned score coincides with the score of a single diffused δ-function.

Forward process and learned velocity: Accumulating the forward update in Eq. (1) over t steps, a training sample $\mathbf x _ { i } \in \mathbb R ^ { d }$ is diffused to

$$
\mathbf { x } _ { t , i } = \sqrt { 1 - \sigma _ { t } ^ { 2 } } \mathbf { x } _ { i } + \sigma _ { t } \epsilon , \quad \epsilon \sim \mathcal { N } ( \epsilon ; 0 , I ) ,\tag{25}
$$

where $\begin{array} { r } { \sigma _ { t } ^ { 2 } = 1 - \prod _ { s = 0 } ^ { t - 1 } ( 1 - \beta _ { s } ) } \end{array}$ is the accumulated variance. The network predicts $\boldsymbol { \epsilon } _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t )$ , from which the learned score is $\mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) ~ = ~ - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) / \sigma _ { t }$ . Following $\operatorname { E q . } \ ( 5 )$ , the velocity field is $\begin{array} { r } { \mathbf { v } = - \frac { \beta _ { t } } { 2 } \bigl ( \mathbf { x } + \nabla \log \rho _ { t } \bigr ) } \end{array}$ , and its learned counterpart without the common factor $- \frac { \beta _ { t } } { 2 }$ is

$$
\widehat { \mathbf { v } } _ { t } ( \mathbf { x } _ { t } ) = \mathbf { x } _ { t } + \mathbf { s } _ { \theta } ( \mathbf { x } _ { t } , t ) = \mathbf { x } _ { t } - \frac { \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) } { \sigma _ { t } } .\tag{26}
$$

Point-mass score: As shown in Section 2.1, the velocity field around a training sample is dominated by the nearest δ-function. A memorizing network therefore reproduces, near $\mathbf { x } _ { t , i }$ , the score of the diffused δ-function centered at ${ \bf x } _ { i } ,$ which is the Gaussian $\mathcal { N } ( \mathbf { x } ; \sqrt { 1 - \sigma _ { t } ^ { 2 } } \mathbf { x } _ { i } , \sigma _ { t } ^ { 2 } I )$ :

$$
\nabla \log \rho _ { t } ( \mathbf { x } _ { t , i } ) = - \frac { \mathbf { x } _ { t , i } - \sqrt { 1 - \sigma _ { t } ^ { 2 } } \mathbf { x } _ { i } } { \sigma _ { t } ^ { 2 } } = - \frac { \epsilon } { \sigma _ { t } } ,\tag{27}
$$

where the last equality follows from Eq. (25). This is exactly the score obtained when the network predicts the injected noise without error, $\epsilon _ { \theta } ( \mathbf { x } _ { t , i } , t ) = \epsilon$

Point-mass velocity: Substituting the point-mass score and Eq. (25) into $\mathbf { x } _ { t , i } + \nabla \log \rho _ { t } ( \mathbf { x } _ { t , i } )$ gives

$$
\begin{array} { l l l } { \displaystyle \mathbf { x } _ { t , i } + \nabla \log \rho _ { t } ( \mathbf { x } _ { t , i } ) } & { = } & { \displaystyle \sqrt { 1 - \sigma _ { t } ^ { 2 } } \mathbf { x } _ { i } + \left( \sigma _ { t } - \frac { 1 } { \sigma _ { t } } \right) \epsilon } \\ & { = } & { \displaystyle \sqrt { 1 - \sigma _ { t } ^ { 2 } } \mathbf { x } _ { i } - \frac { 1 - \sigma _ { t } ^ { 2 } } { \sigma _ { t } } \epsilon . } \end{array}\tag{28}
$$

Taking the expectation over $\epsilon ,$ the cross term vanishes since $\mathbb { E } [ \boldsymbol { \epsilon } ] = 0$ , and $\mathbb { E } \| \epsilon \| _ { 2 } ^ { 2 } = d$ . The expected penalized quantity for the sample $\mathbf { x } _ { i }$ is then

$$
\begin{array} { r l r } { \frac { \beta _ { t } } { 2 } \mathbb { E } _ { \epsilon } \Big [ \| \mathbf { x } _ { t , i } + \nabla \log \rho _ { t } ( \mathbf { x } _ { t , i } ) \| _ { 2 } ^ { 2 } \Big ] } & { = } & { \frac { \beta _ { t } } { 2 } \left( ( 1 - \sigma _ { t } ^ { 2 } ) \| \mathbf { x } _ { i } \| _ { 2 } ^ { 2 } + \frac { ( 1 - \sigma _ { t } ^ { 2 } ) ^ { 2 } } { \sigma _ { t } ^ { 2 } } d \right) } \\ & { = } & { \frac { \beta _ { t } } { 2 } ( 1 - \sigma _ { t } ^ { 2 } ) \left( \| \mathbf { x } _ { i } \| _ { 2 } ^ { 2 } + \frac { 1 - \sigma _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } d \right) . } \end{array}\tag{29}
$$

Threshold: Eq. (29) depends on the individual sample only through $\| \mathbf { x } _ { i } \| _ { 2 } ^ { 2 }$ . For normalized data with unit variance per dimension, $\| \mathbf { x } _ { i } \| _ { 2 } ^ { 2 } \approx d ;$ this holds for the scaled latents of the latent diffusion model.

Under this approximation, the two terms combine as $\begin{array} { r } { 1 + \frac { 1 - \sigma _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } = \frac { 1 } { \sigma _ { t } ^ { 2 } } } \end{array}$ , and the sample-independent threshold becomes

$$
\boxed { R _ { t } = \frac { \beta _ { t } ( 1 - \sigma _ { t } ^ { 2 } ) } { 2 \sigma _ { t } ^ { 2 } } d . }\tag{30}
$$

For data that are not normalized, $\| \mathbf { x } _ { i } \| _ { 2 } ^ { 2 }$ in Eq. (29) can be replaced by its dataset average $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \mathbf { x } _ { i } \| _ { 2 } ^ { 2 } } \end{array}$

A training sample $\mathbf { x } _ { i }$ is assigned to the subset S in Section 4.3 when

$$
\frac { \beta _ { t } } { 2 } \left\| \widehat { \mathbf { v } } _ { t } ( \mathbf { x } _ { t , i } ) \right\| _ { 2 } ^ { 2 } > \gamma R _ { t } ,\tag{31}
$$

i.e., when the learned velocity norm exceeds the fraction $\gamma$ of the point-mass reference. Since the penalty acts only on such samples, it prevents the network from approaching the singular velocity of the memorized reverse process, while the remaining samples are trained with the standard loss.