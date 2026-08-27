# Joint Initialization of Flux Networks and Effective Multiplication Factor for Physics-Informed Neural Networks Solving Neutron Diffusion Problems

Qin Hang, Yangdi Yi, Jiayi Li, Xu Wang, and Heng Zhang

Abstract—Efficient determination of the effective multiplication factor $\left( k _ { e f f } \right)$ is an important computational task in reactor core neutronics analysis. Physics-informed neural networks (PINNs) incorporate neutron diffusion equations and boundary conditions into network training to efficiently determine the neutron flux distribution and $k _ { e f f } .$ To further improve the efficiency of $k _ { e f f }$ calculations using PINNs, a Joint Initialization Physics-Informed Neural Network (JI-PINN) is proposed in this work. In this method, a low-resolution approximate solution to the K-eigenvalue problem is used to construct a joint initial state for the flux network parameters and $k _ { e f f } ,$ and both are then jointly optimized under physical constraints. The proposed method was validated on a two-dimensional two-group twomaterial case, the IAEA 2D benchmark, a two-dimensional twogroup four-material case, and a three-dimensional single-group case. For these test cases, the total computational time was reduced by 25.4%, 38.2%, 49.4%, and 28.9%, respectively, while comparable solution accuracy was maintained. The occurrence of anomalous results associated with marked deviations of $k _ { e f f }$ from the reference value was also reduced. The proposed method provides a more efficient and robust initialization strategy for solving neutron diffusion K-eigenvalue problem with PINNs.

Index Terms—Effective Multiplication Factor, Joint Initialization, Neutron Diffusion Problems, Physics-Informed Neural Network

## I. INTRODUCTION

Efficient determination of the effective multiplication factor $( k _ { e f f } )$ is an important task in reactor core neutronics analysis. As a key parameter in reactor criticality analysis, the calculated value of $k _ { e f f }$ is used to determine whether a reactor system is subcritical, critical, or supercritical [1]–[3]. For a given core geometry, material parameters, and boundary conditions, $k _ { e f f }$ and the corresponding neutron flux distribution need to be determined. The resulting $k _ { e f f }$ and neutron flux distribution provide an important basis for reactor criticality analysis and core power distribution calculations.

In current practice, spatial discretization methods such as the finite difference method, finite element method, and nodal method are commonly combined with power iteration to determine the neutron flux distribution and $k _ { e f f }$ . These methods are well established and achieve reliable numerical accuracy [4], [5]. However, traditional numerical methods generally require spatial discretization of the computational domain and repeated iterations to update the neutron flux and $k _ { e f f }$ . Their computational cost may therefore remain high for high-dimensional problems. In recent years, with the development of deep learning, neural network approaches have been increasingly applied to reactor analysis [6], [7]. Meanwhile, their use for approximating solutions to partial differential equations (PDEs) has attracted increasing attention. Among these approaches, PINNs incorporate governing equations and boundary conditions into neural network training, and PDE solutions are approximated through optimization of the neural network parameters [8], [9].

Building on these developments, PINNs have been applied to solve neutron diffusion equations. Zhang et al. proposed GHO Hybrid PINN and improved the performance in solving multidimensional, multigroup neutron diffusion problems through modifications to the network architecture and optimization strategy [10]. Wang et al. developed a surrogate model for neutron diffusion and demonstrated the feasibility of PINNs for solving neutron diffusion problems [11]. Do et al. investigated a PINN method based on a mixed dual formulation for solving neutron diffusion equations, further extending the application of PINNs to different equation formulations [12]. Zhang et al. proposed R<sup>2</sup>-PINN by combining a convolutional neural network with residual adaptive resampling for solving neutron diffusion equations [13].

For $k _ { e f f }$ determination in neutron diffusion equations, Elhareef and Wu extended PINNs to K-eigenvalue problems by treating $k _ { e f f }$ as a trainable parameter and imposing a regularization constraint to exclude the trivial zero-flux solution [14]. Y. Yang et al. proposed a data-enabled physics-informed neural network (DEPINN) and introduced a small amount of known data into training as prior information to assist in determining the neutron flux and $k _ { e f f }$ [15]. For eigenvalue iteration and problems involving different material regions, Q.-H. Yang et al. combined the inverse power method with neural networks to propose the Generalized Inverse Power Method Neural Network (GIPMNN) and further developed the Physics-Constrained GIPMNN (PC-GIPMNN) to handle material interfaces by enforcing neutron flux continuity and neutron current continuity [16]. Bi et al. proposed Fixed-point Constraint Physics-informed Neural Networks (FC-PINNs) to avoid trivial solutions in eigenvalue calculations and handle interface conditions between different material regions through computational domain concatenation [17].

However, relatively slow training remains a challenge for PINNs in solving neutron diffusion K-eigenvalue problem. One important reason lies in the random initialization of the flux network parameters and the empirical specification of the initial $k _ { e f f }$ . The function represented by a PINN at initialization can influence subsequent optimization behavior [18]. Therefore, different initialization settings may lead to substantial differences in the optimization process and convergence behavior. A large discrepancy between the initial state and the target solution may increase the computational cost of model training and result in convergence difficulties or large errors in the computed $k _ { e f f }$

To address this issue, this paper proposes the Joint Initialization Physics-Informed Neural Network (JI-PINN). The proposed method uses the neutron flux distribution and the corresponding $k _ { e f f }$ from a low-resolution approximate solution to the K-eigenvalue problem to construct a joint initial state for the flux network parameters and $k _ { e f f }$ , and then jointly updates both during subsequent physics-constrained training. The remainder of this paper is organized as follows. Section II introduces the proposed JI-PINN. Section III presents the numerical experiments and analysis of the results. Section IV summarizes this work and discusses future research directions.

## II. METHODS

In the steady-state neutron diffusion K-eigenvalue problem, the neutron flux in each energy group and $k _ { e f f }$ are unknowns to be determined simultaneously. In this work, JI-PINN is constructed by combining a low-resolution spatial discretization approximation with PINN physics-constrained joint optimization. The overall framework is shown in Fig. 1.

For a given reactor geometry, material parameters, and boundary conditions, the G-group steady-state neutron diffusion K-eigenvalue equation can be written as

$$
\begin{array} { r l } { \displaystyle } & { - \nabla \bullet [ D _ { g } \nabla \phi _ { g } ] + \Sigma _ { \mathrm { r } , g } \phi _ { g } - \sum _ { g ^ { \prime } = 1 } ^ { G } \Sigma _ { \mathrm { s } , g ^ { \prime } \to g } \phi _ { g ^ { \prime } } } \\ { \displaystyle } & { \vphantom { \sum _ { a } ^ { a } } } \\ & { = \frac { \mathcal { X } _ { g } } { k _ { e f f } } \sum _ { g ^ { \prime } = 1 } ^ { G } \nu \Sigma _ { \mathrm { f } , g ^ { \prime } } \phi _ { g ^ { \prime } } , \quad g = 1 , \ldots , G } \end{array}\tag{1}
$$

where $D _ { g }$ is the diffusion coefficient for group g, $\phi _ { g }$ is the neutron flux in group $\mathbf { g } , \Sigma _ { \mathrm { r } , g }$ is the removal cross section, $\Sigma _ { \mathrm { s } , g ^ { \prime }  g }$ is the macroscopic scattering cross section from group $\mathbf { g } ^ { \prime }$ to group $\mathrm { g } , \nu \Sigma _ { \mathrm { f } , g ^ { \prime } }$ is the fission neutron production cross section, and $\chi _ { g }$ is the fission spectrum.

In the above K-eigenvalue equation, both $k _ { e f f }$ and the neutron flux are involved in the steady-state neutron balance of the system. When $k _ { e f f }$ changes, the fission source term changes accordingly, and the corresponding neutron flux distribution satisfying the steady-state balance also changes.

Therefore, $k _ { e f f }$ and the neutron flux need to be determined simultaneously during the solution process.

To achieve the simultaneous solution of the neutron flux and $k _ { e f f }$ , JI-PINN uses a neural network to approximate the neutron flux in each energy group and treats $k _ { e f f }$ as an independent trainable parameter. The neutron fluxes in all energy groups are represented as

$$
\widehat { \Phi } \left( x ; \theta \right) = \left[ \widehat { \phi } _ { 1 } \left( x ; \theta \right) , \widehat { \phi } _ { 2 } \left( x ; \theta \right) , . . . , \widehat { \phi } _ { G } \left( x ; \theta \right) \right]\tag{2}
$$

where θ denotes the flux network parameters.

To obtain initialization information at relatively low computational cost, JI-PINN constructs a discrete K-eigenvalue problem on a low-resolution spatial grid to obtain approximate solutions for the neutron flux and $k _ { e f f } \colon$

$$
\begin{array} { r } { A _ { h } \widetilde { \Phi } _ { h } = \frac { 1 } { \widetilde { k } _ { e f f , h } } F _ { h } \widetilde { \Phi } _ { h } } \end{array}\tag{3}
$$

where h denotes the spatial discretization scale, $A _ { h }$ and $F _ { h }$ denote the discretized non-fission and fission operators, respectively, and $\ddot { \Phi } _ { h }$ and $\widetilde { k } _ { e f f , h }$ constitute the approximate Keigenvalue solution of the discrete problem.

The discrete flux $\widetilde { \Phi } _ { h }$ is spatially interpolated to obtain a continuous representation of the approximate flux, $ { \widetilde { \Phi } } ( x )$ The initial values of the flux network parameters are then constructed by fitting $ { \widetilde { \Phi } } ( x )$ , and the initialization loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e } } = \frac { 1 } { N _ { \mathrm { p } } G } \sum _ { i = 1 } ^ { N _ { \mathrm { p } } } \sum _ { g = 1 } ^ { G } \left| \widehat { \phi } _ { g } \left( x _ { i } ; \theta \right) - \widetilde { \phi } _ { g } \left( x _ { i } \right) \right| ^ { 2 } } \end{array}\tag{4}
$$

By minimizing Eq. (4), the initial values of the flux network parameters $\theta ^ { ( 0 ) }$ are obtained, and $k _ { e f f } ^ { ( 0 ) } ~ = ~ \widetilde { k } _ { e f f , h }$ is simultaneously set. The two constitute the joint initial state of JI-PINN. Subsequently, $( \theta ^ { ( 0 ) } , k _ { e f f } ^ { ( 0 ) } )$ is used as the starting point for training, and θ and $k _ { e f f }$ are jointly optimized under physical constraints. With Eq. (2) substituted into Eq. (1), the physics residual of the neutron diffusion equation for group g is defined as

$$
\mathcal { R } _ { g } \left( x ; \theta , k _ { e f f } \right) = \mathcal { N } _ { g } \left[ \widehat { \boldsymbol { \Phi } } \left( x ; \theta \right) , k _ { e f f } \right] , \quad g = 1 , \ldots , G\tag{5}
$$

Based on the governing equation residual in Eq. (5), together with the boundary conditions and the flux scaling constraint, the physics-constrained loss of JI-PINN is written as

$$
\mathcal { L } _ { \mathrm { P I N N } } = \lambda _ { \mathrm { f } } \mathcal { L } _ { \mathrm { f } } + \lambda _ { \mathrm { b } } \mathcal { L } _ { \mathrm { b } } + \lambda _ { \mathrm { s } } \mathcal { L } _ { \mathrm { s } }\tag{6}
$$

where $\mathcal { L } _ { \mathrm { f } } , ~ \mathcal { L } _ { \mathrm { b } }$ , and ${ \mathcal { L } } _ { \mathrm { s } }$ denote the governing equation, boundary condition, and flux scaling losses, respectively. The flux network parameters θ and $k _ { e f f }$ are jointly optimized by minimizing this loss, and the neutron flux distribution and $k _ { e f f }$ are ultimately obtained.

$$
\left( \theta ^ { \ast } , k _ { e f f } ^ { \ast } \right) = \underset { \theta , k _ { e f f } } { \arg \operatorname* { m i n } } \mathcal { L } _ { \mathrm { P I N N } } \left( \theta , k _ { e f f } \right)\tag{7}
$$

![](images/745b95fc753a6bddf5271ce05292fb8d6823f6cdebaeb91ea2be3edd7a63c408.jpg)  
Fig. 1. Overall framework of JI-PINN.

## III. EXPERIMENTS

In this section, JI-PINN was evaluated from four perspectives: the quality of initialization information, the role of joint initialization, sensitivity to random initialization, and computational performance under different problem conditions. These aspects were examined separately by considering the tradeoff between initialization information quality and additional computational cost; the effects of the initial values of the flux network parameters and $k _ { e f f }$ on subsequent optimization and their joint effect; solution stability under different random initial states; and performance in terms of efficiency and accuracy with increasing material heterogeneity and changes in spatial dimensionality.

At the level of test cases, this study selected the twodimensional two-group two-material case to analyze the effects of the spatial resolution of the approximate solution and the degree of flux network pretraining on initialization effectiveness and computational cost, and evaluated the stability of the results across multiple random seeds [19]. For the IAEA 2D benchmark, an initialization ablation study was conducted to distinguish the effects of the initial values of the flux network parameters and the initial $k _ { e f f }$ [15]. The two-dimensional two-group four-material case [20] and the three-dimensional single-group cubic case were further used to evaluate the computational performance of the method under changes in material heterogeneity and spatial dimensionality [19].

## A. Two-Dimensional Two-Group Two-Material Case

The computational domain and material distribution of the two-dimensional two-group two-material case are shown in Fig. 2, and the material parameters are listed in Table I. The computational domain consisted of two material regions, and zero flux boundary conditions were imposed on all outer boundaries. To fix the amplitude of the neutron flux, two fast group neutron flux scaling constraint points were placed at the material interface at (0, −0.5) and (0, 0.5).

This case first investigated the relationship between the quality of initialization information and computational cost. The spatial resolution $N _ { h }$ of the approximate solution denotes the number of nodes of the low-resolution discretization grid along each corresponding spatial direction and determines the spatial information of the flux used for network initialization. The number of pretraining steps $M _ { \mathrm { p r e } }$ controls the degree to which the network fits the approximate flux. A one-factor-at-atime approach was adopted to separately investigate the effects of these two factors on initialization effectiveness. Based on preliminary tests for this case, $N _ { h }$ was set to 17, 25, 33, and 49 to cover a resolution range from insufficient spatial information to basic preservation of the main flux structure. At lower resolutions, the main spatial features of the flux were difficult to represent adequately. As the resolution was further increased, the improvement in the initialization state provided by additional local information gradually diminished, while also incurring higher computational costs for approximate solution calculation and flux network pretraining.

![](images/4965bfda1da05cc13304945830c9287dd0d7f02b852d3e5ba01cacdb4993014c.jpg)  
Fig. 2. Geometry and material distribution of the two-dimensional two-group two-material case.

TABLE I  
MATERIAL PARAMETERS OF THE TWO-DIMENSIONAL TWO-GROUPTWO-MATERIAL CASE
<table><tr><td>Material</td><td>g</td><td> $\overline { { D _ { g } \left( \mathrm { c m } \right) } }$ </td><td> $\overline { { \Sigma _ { \mathrm { a } } \left( \mathrm { c m } ^ { - 1 } \right) } }$ </td><td> $\overline { { \nu \Sigma _ { \mathrm { f } } \left( \mathrm { c m } ^ { - 1 } \right) } }$ </td><td> $\overline { { \Sigma _ { 1  2 } ( \mathrm { c m } ^ { - 1 } ) } }$ </td></tr><tr><td rowspan="2">Material 1</td><td>1</td><td>1.268</td><td>0.007181</td><td>0.004609</td><td>0.02767</td></tr><tr><td>2</td><td>0.1902</td><td>0.07047</td><td>0.08675</td><td></td></tr><tr><td rowspan="2">Material 2</td><td>1</td><td>1.255</td><td>0.008252</td><td>0.004602</td><td>0.02533</td></tr><tr><td>2</td><td>0.211</td><td>0.1003</td><td>0.1091</td><td></td></tr></table>

$M _ { \mathrm { p r e } }$ was set to 1000, 3000, 5000, and 8000 to examine the process by which the network progressively formed the main spatial distribution of the approximate neutron flux as the degree of pretraining increased from an insufficient level. With fewer pretraining steps, the network fitted the approximate flux only to a limited extent and had difficulty forming a stable initial flux representation. As pretraining continued, the network gradually captured the main spatial features of the approximate neutron flux. However, longer pretraining increased the computational cost and could also lead to greater retention of the discretization and interpolation deviations in the low-resolution approximate solution.

The initialization effectiveness was evaluated using the initial loss $L _ { 0 }$ at the start of physics-constrained training, the total computational time $T _ { \mathrm { t o t a l } }$ , and the final relative error of $k _ { e f f } . ~ T _ { \mathrm { t o t a l } }$ included the computational time required for approximate solution calculation, flux network pretraining, and subsequent physics-constrained training. The final relative error of $k _ { e f f }$ was defined as

$$
\begin{array} { r } { \varepsilon _ { \mathrm { k } } = \frac { \left| k _ { e f f } ^ { \mathrm { P I N N } } - k _ { e f f } ^ { \mathrm { r e f } } \right| } { \left| k _ { e f f } ^ { \mathrm { r e f } } \right| } } \end{array}\tag{8}
$$

The different parameter settings and corresponding computational results are listed in Table II.

TABLE II  
RESULTS OF THE HYPERPARAMETER ANALYSIS FOR THETWO-DIMENSIONAL TWO-GROUP TWO-MATERIAL CASE
<table><tr><td>Experimental Variable</td><td>Parameter Value</td><td> $L _ { 0 }$ </td><td> $\overline { { T _ { \mathrm { t o t a l } } \mathrm { ( s ) } } }$ </td><td></td><td> $\varepsilon _ { \mathbf { k } }$ </td></tr><tr><td rowspan="4"> $N _ { h }$ </td><td>17</td><td> $\overline { { 4 . 9 2 9 \times 1 0 ^ { - 3 } } }$ </td><td>797.9</td><td></td><td> $\overline { { 6 . 6 9 3 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>25</td><td> $3 . 9 7 4 \times 1 0 ^ { - 3 }$ </td><td>1575.3</td><td></td><td> $2 . 5 3 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>33</td><td> $5 . 9 4 6 \times 1 0 ^ { - 3 }$ </td><td>1631.0</td><td></td><td> $2 . 7 3 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>49</td><td> $7 . 6 6 5 \times 1 0 ^ { - 3 }$ </td><td>1688.4</td><td></td><td> $2 . 8 2 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="4"> $M _ { \mathrm { p r e } }$ </td><td>1000</td><td> $4 . 6 9 9 \times 1 0 ^ { - 1 }$ </td><td>8045.4</td><td></td><td> $0 . 1 3 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>3000</td><td> $7 . 4 1 3 \times 1 0 ^ { - 3 }$ </td><td>1986.1</td><td></td><td> $2 . 9 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>5000</td><td> $3 . 9 7 4 \times 1 0 ^ { - 3 }$ </td><td>1575.3</td><td></td><td> $2 . 5 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>8000</td><td> $2 . 2 4 9 \times 1 0 ^ { - 3 }$ </td><td>831.7</td><td></td><td> $5 . 4 1 \times 1 0 ^ { - 3 }$ </td></tr></table>

Note: The reference value is $k _ { e f f } ^ { \mathrm { r e f } } = 1 . 0 6 9 5 .$

When $N _ { h }$ increased from 17 to 25, both $L _ { 0 }$ and $\varepsilon _ { \mathrm { k } }$ decreased markedly. The higher spatial resolution preserved the main spatial structure of the approximate flux more adequately and improved the initial function represented by the network at the start of physics-constrained training. When $N _ { h }$ was further increased to 33 and 49, neither $L _ { 0 }$ nor $\varepsilon _ { \mathrm { k } }$ decreased further, while $T _ { \mathrm { t o t a l } }$ continued to increase. At this stage, the main spatial features governing the flux distribution had already been largely preserved. Further increasing the resolution mainly added local spatial information, provided only limited improvement to the training starting point, and increased the computational overhead of approximate solution calculation and flux network pretraining. Therefore, the spatial resolution of the approximate solution needed to balance the completeness of the initial flux information against the additional computational overhead.

$M _ { \mathrm { p r e } }$ controlled the degree of fit between the approximate flux and the initial function representation of the network. When $M _ { \mathrm { p r e } } { = } 1 0 0 0$ , the network had not yet sufficiently formed the main spatial distribution of the approximate neutron flux, and substantial adjustment of the flux function was still required during subsequent physics-constrained training. As $M _ { \mathrm { p r e } }$ increased, the initial function represented by the network gradually approached the spatial distribution of the approximate flux. When $M _ { \mathrm { p r e } }$ was further increased from 5000 to 8000, $L _ { 0 }$ decreased further, whereas $\varepsilon _ { \mathrm { k } }$ increased. Longer pretraining allowed the network to fit the low-resolution approximate solution more fully, but could also retain the discretization and interpolation deviations contained in the approximate solution. These deviations still needed to be corrected during subsequent physics-constrained training.

Considering the initialization effectiveness, total computational time, final error, and training termination status, $N _ { h } { = } 2 5$ and $M _ { \mathrm { p r e } } { = } 5 0 0 0$ were used in the subsequent experiments. This combination retained effective initial flux information while avoiding the additional computational overhead caused by further increasing the spatial resolution and the potential loss of accuracy caused by prolonged pretraining. On this basis, 15 random seeds were selected for the comparison of the randomly initialized PINN, R<sup>2</sup>-PINN [13], and JI-PINN. The statistical results are listed in Table III, and the distributions of the results for the three methods are shown in Fig. 3.

Table III and Fig. 3 showed that the $k _ { e f f }$ values, flux errors, and total computational times obtained with the randomly initialized PINN exhibited substantial variability, and the $k _ { e f f }$ values obtained under some random seeds deviated markedly from the reference value. The $R ^ { 2 } { \mathrm { - } } \mathrm { P I N N }$ results showed a narrower distribution, but a few relatively large deviations still remained. The $k _ { e f f }$ values obtained by JI-PINN across all 15 random seeds were concentrated near the reference value, and the variability in the flux errors of the fast and thermal groups was further reduced.

In terms of computational efficiency, the average total computational time of JI-PINN was reduced by 25.4% and 38.8% relative to the randomly initialized PINN and $R ^ { 2 } { \mathrm { - P I N N } } .$ respectively, and remained at a relatively low level. As shown in Fig. 3(c), the mean squared error (MSE) of JI-PINN for the fast and thermal groups remained low overall, and the reduction in computational time was not accompanied by a noticeable loss of flux accuracy.

TABLE III  
STATISTICAL RESULTS OF THE RANDOM SEED EXPERIMENTS FOR THE TWO-DIMENSIONAL TWO-GROUP TWO-MATERIAL CASE
<table><tr><td>Method</td><td> $\overline { { \operatorname { F i n a l } { k _ { e f f } } } }$ </td><td> $\varepsilon _ { \mathbf { k } }$ </td><td>Fast Group MSE</td><td>Thermal Group MSE</td><td> $\overline { { T _ { \mathrm { t o t a l } } \left( \mathrm { s } \right) } }$ </td></tr><tr><td>Randomly Initialized PINN</td><td> $\overline { { 1 . 0 3 7 \pm 0 . 0 3 9 } }$ </td><td> $\overline { { ( 3 . 1 4 1 \pm 3 . 5 1 2 ) \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { ( 1 . 0 8 8 \pm 1 . 2 9 9 ) \times 1 0 ^ { - 1 } } }$ </td><td> $\overline { { ( 1 . 6 3 4 \pm 1 . 8 3 5 ) \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 3 3 0 9 . 6 \pm 3 6 6 . 2 } }$ </td></tr><tr><td> $R ^ { 2 } { \cdot } \mathrm { P I N N }$ </td><td> $1 . 0 5 6 \pm 0 . 0 2 7$ </td><td> $\left( 1 . 2 5 2 \pm 2 . 5 3 2 \right) \times 1 0 ^ { - 2 }$ </td><td> $\left( 2 . 2 7 6 \pm 5 . 7 9 1 \right) \times 1 0 ^ { - 2 }$ </td><td> $( 3 . 3 8 9 \pm 8 . 4 5 4 ) \times 1 0 ^ { - 3 }$ </td><td> $4 0 3 3 . 4 \pm 8 0 6 . 2$ </td></tr><tr><td>JI-PINN</td><td> $1 . 0 6 6 \pm 0 . 0 0 1$ </td><td> $\left( 2 . 4 2 8 \pm 1 . 5 1 1 \right) \times 1 0 ^ { - 3 }$ </td><td> $( 5 . 6 1 5 \pm 3 . 2 4 6 ) \times 1 0 ^ { - 3 }$ </td><td> $( 8 . 3 3 0 \pm 4 . 8 9 2 ) \times 1 0 ^ { - 4 }$ </td><td> $2 4 6 7 . 7 \pm 9 3 1 . 5$ </td></tr></table>

Note: The reference value is $k _ { e f f } ^ { \mathrm { r e f } } = 1 . 0 6 9 5$ . The data in the table are reported as the mean ± standard deviation over 15 random seed runs.

(b)  
![](images/5a4ad40dddc875ea74bc6a9bc9cc7cc97fe3b4037739baffb3d405258ee8581f.jpg)

![](images/83b05d99f9f3c6503ec1ce5472421aad71e8d5cd97c7400034d4c7931dc620dd.jpg)

![](images/9c1dfe00760c70e9ec4e2e79b16f055c7eaa829d6e850c10e6303483e69125e8.jpg)  
Fig. 3. Solution results and computational time for different random seeds: (a) final $k _ { e f f } ;$ (b) total computational time; (c) MSE of the neutron flux for the fast and thermal groups.

Different random seeds corresponded to different network parameter states and further produced substantially different initial flux functions. Although the flux networks in JI-PINN also started pretraining with randomly initialized parameters, all networks were subject to the combined effects of the same approximate flux distribution, boundary conditions, and flux scaling constraint. As a result, the differences among the flux functions at the start of formal physics-constrained training were reduced, while $k _ { e f f }$ was initialized using the value obtained together with the approximate flux. Joint initialization made the initial states of networks with different random initializations more similar at the start of formal training, and the corresponding $k _ { e f f }$ , flux errors, and computational times also exhibited more concentrated distributions across different random seeds.

Based on the hyperparameter analysis, the quality of the approximate flux information and the degree of pretraining jointly determined the initial state provided by joint initialization. An appropriate spatial resolution and degree of pretraining could preserve the main spatial features of the fluxTNS while controlling the additional computational overhead, thereby balancing initialization quality and overall computational efficiency.

## B. IAEA 2D Benchmark

A quarter-core model was used for the IAEA 2D benchmark, with a computational domain of 170 cm $\times \phantom { - } 1 7 0$ cm containing four material regions. Neumann boundary conditions were imposed on the left and bottom boundaries, while vacuum boundary conditions were imposed on the remaining stepped outer boundaries. The geometry and boundary types are shown in Fig. 4, and the material parameters are listed in Table IV.

![](images/647a8be3385f2492ca052a5c65173154128f70b39889d227215334980b99df09.jpg)  
Fig. 4. Geometric layout of the IAEA 2D benchmark.

TABLE IV  
MATERIAL PARAMETERS OF THE IAEA 2D BENCHMARK
<table><tr><td>Region</td><td>g</td><td> $\overline { { D _ { g } \left( \mathrm { c m } \right) } }$ </td><td>Σa (cm−1)</td><td>νΣf (cm−1)</td><td> $\overline { { \Sigma _ { 1  2 } ( \mathrm { c m } ^ { - 1 } ) } }$ </td></tr><tr><td rowspan="2"> $\Omega _ { 1 }$ </td><td>1</td><td>1.5</td><td>0.010</td><td>0.000</td><td>0.02</td></tr><tr><td>2</td><td>0.4</td><td>0.080</td><td>0.135</td><td></td></tr><tr><td rowspan="2"> $\Omega _ { 2 }$ </td><td>1</td><td>1.5</td><td>0.010</td><td>0.000</td><td>0.02</td></tr><tr><td>2</td><td>0.4</td><td>0.085</td><td>0.135</td><td></td></tr><tr><td rowspan="2"> $\Omega _ { 3 }$ </td><td>1</td><td>1.5</td><td>0.010</td><td>0.000</td><td>0.02</td></tr><tr><td>2</td><td>0.4</td><td>0.130</td><td>0.135</td><td></td></tr><tr><td rowspan="2"> $\Omega _ { 4 }$ </td><td>1</td><td>2.0</td><td>0.000</td><td>0.000</td><td>0.04</td></tr><tr><td>2</td><td>0.3</td><td>0.010</td><td>0.000</td><td></td></tr></table>

The neutron fluxes in the fast and thermal groups were approximated by a neural network with two outputs. An ablation study on initialization was conducted in this case to examine the roles of the initial flux representation of the network and the initial value of $k _ { e f f }$ in subsequent joint optimization. The flux network was either randomly initialized or pretrained using the approximate flux, while $k _ { e f f } ^ { ( 0 ) }$ was set to either 1.0 or the approximate K-eigenvalue $\widetilde { k } _ { e f f }$ . The computational results are listed in Table V.

TABLE V  
COMPUTATIONAL RESULTS FOR THE IAEA 2D BENCHMARK UNDER DIFFERENT INITIALIZATION CONFIGURATIONS
<table><tr><td>Initialization Method</td><td> $\overline { { k _ { e f f } ^ { ( 0 ) } } }$ </td><td> $k _ { e f f }$ </td><td> $L _ { 0 }$ </td><td></td><td>εk</td><td> $T _ { \mathrm { t o t a l } } \ : ( \mathrm { s } )$ </td></tr><tr><td>Random Initialization</td><td>1</td><td>1.029586</td><td> $\overline { { 6 . 4 2 0 \times 1 0 ^ { 5 } } }$ </td><td></td><td> $\overline { { 3 . 4 8 \times 1 0 } } ^ { . }$  -4</td><td>7398.5</td></tr><tr><td></td><td>1.029240 1</td><td>1.030006 1.030104</td><td> $6 . 4 2 0 \times 1 0 ^ { 5 }$ </td><td></td><td> $6 . 0 2 \times { { 1 0 } ^ { - 5 } }$   $1 . 5 5 \times { { 1 0 } ^ { - 4 } }$ </td><td>7106.3 5194.9</td></tr><tr><td>Flux Initialization</td><td></td><td></td><td> $8 . 8 5 8 \times 1 0 ^ { 3 }$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>1.0292401.029367</td><td> $8 . 8 4 2 \times 1 0 ^ { 3 }$ </td><td></td><td> $5 . 6 1 \times { { 1 0 } ^ { - 4 } }$ </td><td>4573.0</td></tr></table>

Note: The reference value is $k _ { e f f } ^ { \mathrm { r e f } } = 1 . 0 2 9 9 4 4 .$

The final $k _ { e f f }$ values obtained under all four initialization configurations remained close to the reference value, but clear differences were observed in the training starting points and total computational times. Under random flux initialization, when $k _ { e f f } ^ { ( 0 ) }$ was adjusted from 1.0 to $\widetilde { k } _ { e f f , h } , T _ { \mathrm { t o t a l } }$ decreased only from 7398.5 s to 7106.3 s, a reduction of 3.9%. In this case, the flux network still entered physics-constrained training from a random function state, and the spatial distribution of the flux still required gradual adjustment during subsequent optimization. Therefore, improving only the initial value of $k _ { e f f }$ provided relatively limited time savings.

After approximate flux pretraining, with the same setting of $k _ { e f f } ^ { ( 0 ) } { = } 1 , ~ L _ { 0 }$ decreased markedly, and $T _ { \mathrm { t o t a l } }$ decreased from 7398.5 s to 5194.9 s, a reduction of 29.8%. Before formal physics-constrained optimization began, the pretrained network already exhibited the main spatial distribution of the approximate flux, reducing the adjustment required for the spatial structure of the flux function during subsequent training. On this basis, $\widetilde { k } _ { e f f , h }$ , obtained together with the approximate flux, was further used as the initial value of $k _ { e f f }$ . The initial loss remained nearly unchanged, while $T _ { \mathrm { t o t a l } }$ decreased from 5194.9 s to 4573.0 s, a further reduction of 12.0%.

Fig. 5 further showed the convergence histories of $k _ { e f f }$ under the four initialization configurations. Random flux initialization was used in both Fig. 5(a) and Fig. 5(b). In Fig. 5(b), $k _ { e f f }$ started from a value closer to the reference value, but noticeable iterative adjustment was still observed during the early stage of training.

Approximate flux pretraining was used in both Fig. 5(c) and Fig. 5(d). The corresponding $k _ { e f f }$ convergence histories entered the vicinity of the reference value earlier than those under random flux initialization. Under the same pretrained flux condition, JI-PINN further used the corresponding $k _ { e f f }$ obtained from that K-eigenvalue solution as the initial value of $k _ { e f f }$ , and $T _ { \mathrm { t o t a l } }$ was further reduced.

Considering the results from the four initialization ablation cases, approximate flux pretraining improved the flux function representation at the start of formal physics-constrained training. After the $k _ { e f f }$ value obtained together with the approximate flux from the same approximate K-eigenvalue solution was further used as the initial value of $k _ { e f f } ,$ the initial flux representation and the initial value of $k _ { e f f }$ were both derived from the same approximate solution, forming an initial state more favorable for subsequent joint optimization.

## C. Two-Dimensional Two-Group Four-Material Case

This case was used to investigate the computational performance of the proposed JI-PINN in a heterogeneous multimaterial domain. The geometry and material distribution of the two-dimensional two-group four-material case are shown in Fig. 6, and the material parameters are listed in Table VI. Zero flux boundary conditions were imposed on the outer boundaries.

The neutron fluxes in the fast and thermal groups were approximated separately by independent fully connected neural networks. The computational results are listed in Table VII, and the convergence history of $k _ { e f f }$ is shown in Fig. 7.

As shown in Table VII, $T _ { \mathrm { t o t a l } }$ of JI-PINN was reduced by 49.4% and 69.5% compared with those of the randomly initialized PINN and $R ^ { 2 } { \mathrm { - P I N N } }$ , respectively. The convergence histories in Fig. 7 showed that, for the randomly initialized PINN and $R ^ { 2 } { \mathrm { - P I N N } }$ $k _ { e f f }$ had to be progressively adjusted from an empirically set initial value toward the reference value. In contrast, JI-PINN entered physics-constrained training with an initial state closer to the reference value, exhibited only small fluctuations during the early stage of training, and then entered a stable range more rapidly. The improved initial state reduced the extent of adjustment required for the flux representation and $k _ { e f f }$ during the early stage of training, thereby shortening $T _ { \mathrm { t o t a l } }$ . However, the relative error of $k _ { e f f }$ for JI-PINN was higher than those of the two comparison methods. The shorter subsequent training was insufficient to fully correct the local deviations retained in the approximate initial state, and the improvement in computational efficiency was accompanied by some loss of accuracy.

![](images/890d9f6f925e4e954b6199d9dc0c28732f1495a3afe80ccbffb2ecac7a325d00.jpg)  
(a) Random flux initialization $k _ { e f f } ^ { ( 0 ) } = 1$

![](images/81cef0ba804cdcb834b33e4f37a9516d2df03562a2a646df8580ae1fed18c9a7.jpg)  
(c) Pretrained flux initialization, $k _ { e f f } ^ { ( 0 ) } = 1$

![](images/3e548607999778da77caac069f318daf48ac7396dfebdb599fb9a2b267ff17f1.jpg)  
(b) Random flux initialization, $k _ { e f f } ^ { ( 0 ) } = \widetilde { k } _ { e f f }$

![](images/5394f76ce84fd14e0d987b49915d5fbe0629081246a4555e004708164742d2b3.jpg)  
(d) JI-PINN

Fig. 5. Convergence histories of $k _ { e f f }$ under different initialization configurations.  
![](images/6003054312a3f46c36fdb70941553b551db03a7fb1fa0e82b460b37a0f1f5b33.jpg)  
Fig. 6. Geometry and material distribution of the two-dimensional two-group four-material case.

TABLE VI  
MATERIAL PARAMETERS OF THE TWO-DIMENSIONAL TWO-GROUPFOUR-MATERIAL CASE
<table><tr><td>Material</td><td>g</td><td> $\overline { { D _ { g } \left( \mathrm { c m } \right) } }$ </td><td> $\overline { { \Sigma _ { \mathrm { a } } \left( \mathrm { c m } ^ { - 1 } \right) } }$ </td><td> $\overline { { \nu \Sigma _ { \mathrm { f } } } }$  (cm−1)</td><td> $\overline { { \Sigma _ { 1  2 } ( \mathrm { c m } ^ { - 1 } ) } }$ </td></tr><tr><td rowspan="2">Material 1</td><td>1</td><td>1.255</td><td>0.008252</td><td>0.004602</td><td>0.02533</td></tr><tr><td>2</td><td>0.2110</td><td>0.100300</td><td>0.10910</td><td></td></tr><tr><td rowspan="2">Material 2</td><td>1</td><td>1.268</td><td>0.007181</td><td>0.004609</td><td>0.02767</td></tr><tr><td>2</td><td>0.1902</td><td>0.070470</td><td>0.08675</td><td></td></tr><tr><td rowspan="2">Material 3</td><td>1</td><td>1.259</td><td>0.008002</td><td>0.004663</td><td>0.02617</td></tr><tr><td>2</td><td>0.2091</td><td>0.083440</td><td>0.10210</td><td></td></tr><tr><td rowspan="2">Material 4</td><td>1</td><td>1.259</td><td>0.008002</td><td>0.004663</td><td>0.02617</td></tr><tr><td>2</td><td>0.2091</td><td>0.073324</td><td>0.10210</td><td></td></tr></table>

TABLE VII  
COMPUTATIONAL RESULTS FOR THE TWO-DIMENSIONAL TWO-GROUP FOUR-MATERIAL CASE
<table><tr><td>Method</td><td> $\overline { { k _ { e f f } } }$ </td><td> $\varepsilon _ { \mathbf { k } }$ </td><td> $\overline { { T _ { \mathrm { t o t a l } } \mathrm { ( s ) } } }$ </td></tr><tr><td>Randomly Initialized PINN</td><td>0.796108</td><td> $\overline { { 4 . 8 8 \times 1 0 ^ { - 4 } } }$ </td><td>2757.6</td></tr><tr><td>R2-PINN</td><td>0.795609</td><td> $1 . 3 9 \times 1 0 ^ { - 4 }$ </td><td>4578.0</td></tr><tr><td>JI-PINN</td><td>0.793601</td><td> $2 . 6 6 \times 1 0 ^ { - 3 }$ </td><td>1394.9</td></tr></table>

Note: The reference value is $k _ { e f f } ^ { \mathrm { r e f } } = 0 . 7 9 5 7 2 .$

![](images/a1b302ffaa180e1f294d03040444857f9fa8c9b598149c817db90213ec63f0ed.jpg)  
Fig. 7. Convergence history of $k _ { e f f }$ for the two-dimensional two-group fourmaterial case.

As the number of material interfaces increased, local flux variations near the interfaces became more pronounced, and the low-resolution approximate solution was limited in its ability to represent these local features, so some local deviations could still remain after joint initialization. The extent to which these deviations were corrected during subsequent physicsconstrained training further affected the final solution accuracy.

## D. Three-Dimensional Single-Group Cubic Case

The computational domain of this case was a cube with a side length of 1 m, and zero flux boundary conditions were imposed on all six outer surfaces, with $k _ { \infty } { = } 1 . 4 2 5$ and the square of the diffusion length $L ^ { 2 } { = } 0 . 0 1 6 8 8 6 9$ . A flux scaling constraint of $\phi ( 0 , 0 , 0 ) = 0 . 5$ was also imposed at the center of the cube, and the geometry is shown in Fig. 8.

![](images/0c2840d34742e20e8e1967028b49b3945e3453fee17d5e5c63bbc3544bd57406.jpg)  
Fig. 8. Geometry of the three-dimensional single-group cubic case.

TABLE VIII  
COMPUTATIONAL RESULTS FOR THE THREE-DIMENSIONAL SINGLE-GROUP CUBIC CASE
<table><tr><td>Method</td><td> $k _ { e f f }$ </td><td> $\varepsilon _ { \mathbf { k } }$ </td><td>Flux MSE</td><td></td><td> $\overline { { T _ { \mathrm { t o t a l } } \mathrm { ( s ) } } }$ </td></tr><tr><td>Randomly Initialized PINN</td><td>0.9505335</td><td></td><td> $\overline { { 5 . 6 2 \times { 1 0 } ^ { - 4 } } }$ </td><td> $\overline { { 6 . 1 5 \times 1 0 ^ { - 7 } } }$ </td><td>697.4</td></tr><tr><td> ${ \bf \ddot { \boldsymbol { R } } } ^ { 2 } .$  -PINN</td><td>0.950510</td><td> $5 . 3 7 \times { { 1 0 } ^ { - 4 } }$ </td><td></td><td> $8 . 1 7 \times 1 0 ^ { - 7 }$ </td><td>612.6</td></tr><tr><td>JI-PINN</td><td>0.950660</td><td></td><td> $6 . 9 5 \times { { 1 0 } ^ { - 4 } }$ </td><td> $9 . 4 9 \times 1 0 ^ { - 7 }$ </td><td>496.1</td></tr></table>

Note: The reference value is $k _ { e f f } ^ { \mathrm { r e f } } = 0 . 9 5 0 0 .$

The computational results are listed in Table VIII, and the convergence history of $k _ { e f f }$ is shown in Fig. 9.

In Table VIII, the relative errors of $k _ { e f f }$ and the neutron flux MSE obtained by the three methods were of the same order of magnitude, and the final solution accuracies were generally comparable. The $T _ { \mathrm { t o t a l } }$ of JI-PINN was approximately 28.9% and 19.0% lower than those of the randomly initialized PINN and $R ^ { 2 } { \cdot } \mathrm { P I N N } ,$ respectively. In Fig. 9, both the randomly initialized PINN and $R ^ { 2 } { \mathrm { - } } \mathrm { P I N N }$ underwent relatively large adjustments of $k _ { e f f }$ during the early stage of training. JI-PINN used approximate flux pretraining to set the initial values of the network, so that the flux representation and the trainable parameter $k _ { e f f }$ were optimized starting from the existing approximate flux representation and its corresponding initial $k _ { e f f }$ value, thereby reducing the correction range during the early stage of training and satisfying the stopping criterion with fewer training steps. However, brief fluctuations were still observed during the early iterations, indicating that the lowresolution approximate state still required further correction under physical constraints.

In the three-dimensional setting, joint initialization still improved the initial state for formal training and reduced the overall computational time while maintaining comparable solution accuracy.

## IV. CONCLUSION

For the joint solution of the neutron flux and $k _ { e f f }$ in steadystate neutron diffusion K-eigenvalue problem, this paper proposes JI-PINN, a method based on joint initialization of the flux network and $k _ { e f f }$ , and validates it through extensive numerical tests on the two-dimensional two-group two-material case, the IAEA 2D benchmark, the two-dimensional twogroup four-material case, and the three-dimensional singlegroup case. The results from these cases showed that JI-PINN reduced the total computational time by 25.4%–49.4% compared with the randomly initialized PINN. The hyperparameter analysis and random seed experiments revealed the influence of the starting point of training on subsequent optimization. An appropriate spatial resolution of the approximate solution and degree of pretraining balanced the preservation of the main flux information against the additional computational cost and reduced the differences in the initial states of networks with different random initializations when entering formal training. The ablation study on initialization for the IAEA 2D benchmark further distinguished the roles of the two types of initialization information: the approximate flux improved the flux function representation at the start of formal training, while the corresponding $k _ { e f f }$ provided an initial value closer to the target solution for eigenvalue optimization. Together, they formed an initial state more favorable for subsequent physics-constrained joint optimization. As the problems were extended to heterogeneous domains composed of multiple material regions and to three-dimensional settings, JI-PINN still exhibited a clear advantage in computational efficiency while maintaining a favorable balance between efficiency and accuracy.

(a) Randomly Initialized PINN  
(c) JI-PINN  
![](images/7da4d38788eba8b261d807b43dd4939df497ac172d3c997e37794bd6b4029cb0.jpg)

![](images/aba41edbfb94f5fe9efaaaba809e987f717652b996119d5a4be9a01a88911bfe.jpg)

![](images/2c8c61d91add92263b4b13527fef7d91409f4ffacfe2b2cb1cdafcf9f2f91abe.jpg)  
Fig. 9. Convergence history of $k _ { e f f }$ for the three-dimensional single-group cubic case.

Overall, JI-PINN provides a new approach to improving the overall solution efficiency of steady-state neutron diffusion Keigenvalue problem. However, the computational performance of JI-PINN is still affected by the quality of the low-resolution approximate solution, especially for problems with complex material interfaces and pronounced local flux variations, where the ability of the approximate solution to represent local spatial information can further affect physics-constrained optimization and the final solution accuracy. Future work will further investigate the applicability of the method to more complex material regions, three-dimensional multigroup reactor core models, and repeated calculations for parameterized problems, and will explore its combination with other efficient physicsconstrained solution strategies.

## REFERENCES

[1] J. J. Duderstadt and L. J. Hamilton, Nuclear Reactor Analysis. New York, NY, USA: Wiley, 1976.

[2] E. E. Lewis and W. F. Miller, Jr., Computational Methods of Neutron Transport. New York, NY, USA: Wiley, 1984.

[3] W. M. Stacey, Nuclear Reactor Physics, 3rd ed. Weinheim, Germany: Wiley-VCH, 2018.

[4] R. D. Lawrence, “Progress in nodal methods for the solution of the neutron diffusion and transport equations,” Prog. Nucl. Energy, vol. 17, no. 3, pp. 271–301, 1986, doi: 10.1016/0149-1970(86)90034-X.

[5] A. Hébert, Applied Reactor Physics. Montréal, QC, Canada: Presses Internationales Polytechnique, 2009.

[6] P. Cao, C. Cao, and Q. Gan, “A 3-D neutron distribution reconstruction method based on the off-situ measurement for reactor,” IEEE Trans. Nucl. Sci., vol. 68, no. 12, pp. 2694–2701, Dec. 2021, doi: 10.1109/TNS.2021.3123381.

[7] X. Chen and A. Ray, “Deep reinforcement learning control of a boiling water reactor,” IEEE Trans. Nucl. Sci., vol. 69, no. 8, pp. 1820–1832, Aug. 2022, doi: 10.1109/TNS.2022.3187662.

[8] I. E. Lagaris, A. Likas, and D. I. Fotiadis, “Artificial neural networks for solving ordinary and partial differential equations,” IEEE Trans. Neural Netw., vol. 9, no. 5, pp. 987–1000, Sep. 1998, doi: 10.1109/72.712178.

[9] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” J. Comput. Phys., vol. 378, pp. 686–707, Feb. 2019, doi: 10.1016/j.jcp.2018.10.045.

[10] H. Zhang, J. Li, Y. Yi, Q. Hang, and Y. Jiang, “Gradient harmony optimization-driven hybrid physics-informed neural network for solving neutron diffusion problems,” IEEE Trans. Nucl. Sci., vol. 73, no. 3, pp. 505–518, Mar. 2026, doi: 10.1109/TNS.2026.3655477.

[11] J. Wang, X. Peng, Z. Chen, B. Zhou, Y. Zhou, and N. Zhou, “Surrogate modeling for neutron diffusion problems based on conservative physics-informed neural networks with boundary conditions enforcement,” Ann. Nucl. Energy, vol. 176, Oct. 2022, Art. no. 109234, doi: 10.1016/j.anucene.2022.109234.

[12] M.-H. Do, K. Ammar, N. G. Castaing, and F. Madiot, “Physics informed neural networks for the mixed dual form of the neutron diffusion equation with heterogeneous coefficients,” Ann. Nucl. Energy, vol. 223, Dec. 2025, Art. no. 111607, doi: 10.1016/j.anucene.2025.111607.

[13] H. Zhang, Y.-L. He, D. Liu, Q. Hang, H.-M. Yao, and D. Xiang, “Residual resampling-based physics-informed neural network for neutron diffusion equations,” Nucl. Sci. Tech., vol. 37, no. 2, Feb. 2026, Art. no. 28, doi: 10.1007/s41365-025-01839-5.

[14] M. H. Elhareef and Z. Wu, “Physics-informed neural network method and application to nuclear reactor calculations: A pilot study,” Nucl. Sci. Eng., vol. 197, no. 4, pp. 601–622, Apr. 2023, doi: 10.1080/00295639.2022.2123211.

[15] Y. Yang et al., “A data-enabled physics-informed neural network with comprehensive numerical study on solving neutron diffusion eigenvalue problems,” Ann. Nucl. Energy, vol. 183, Apr. 2023, Art. no. 109656, doi: 10.1016/j.anucene.2022.109656.

[16] Q.-H. Yang, Y. Yang, Y.-T. Deng, Q.-L. He, H.-L. Gong, and S.-Q. Zhang, “Physics-constrained neural network for solving discontinuous interface K-eigenvalue problem with application to reactor physics,” Nucl. Sci. Tech., vol. 34, no. 10, Oct. 2023, Art. no. 161, doi: 10.1007/s41365-023-01313-0.

[17] H. Bi, M. Song, T. Zhang, and X. Liu, “FC-PINNs: Physics-informed neural networks for solving neutron diffusion eigenvalue problem with interface considerations,” J. Comput. Phys., vol. 541, Nov. 2025, Art. no. 114311, doi: 10.1016/j.jcp.2025.114311.

[18] J. C. Wong, C. C. Ooi, A. Gupta, and Y.-S. Ong, “Learning in sinusoidal spaces with physics-informed neural networks,” IEEE Trans. Artif. Intell., vol. 5, no. 3, pp. 985–1000, Mar. 2024, doi: 10.1109/TAI.2022.3192362.

[19] D. Liu, L. Tang, P. An, B. Zhang, and Y. Jiang, “The deep learning method to search effective multiplication factor of nuclear reactor directly,” (in Chinese), Nucl. Power Eng., vol. 44, no. 5, pp. 6–14, Oct. 2023, doi: 10.13832/j.jnpe.2023.05.0006.

[20] Y. Jiang, P. An, D. Liu, and Y. Yu, “Research on the solution and acceleration algorithm of source iteration method based on PINN,” (in Chinese), Nucl. Power Eng., vol. 46, no. 2, pp. 148–155, Apr. 2025, doi: 10.13832/j.jnpe.2024.090040.