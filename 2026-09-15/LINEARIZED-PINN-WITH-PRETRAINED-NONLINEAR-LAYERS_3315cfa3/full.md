# LINEARIZED PINN WITH PRETRAINED NONLINEAR LAYERS

Wenhao Chen Civil and Environmental Engineering Department University of Illinois Urbana-Champaign Urbana, IL 61801, USA wenhaoc3@illinois.edu

Alexandre Tartakovsky Civil and Environmental Engineering Department University of Illinois Urbana-Champaign Urbana, IL 61801, USA Pacific Northwest National Laboratory Richland, WA 99352, USA amt1998@illinois.edu

September 15, 2026

## ABSTRACT

We propose a Linearized Physics-Informed Neural Network (lPINN), a reduced-order neural basis method for solving forward and inverse partial differential equations. During an offline stage, lPINN learns operator-compatible, continuous neural basis functions from an ensemble of numerical solutions. These functions are pretrained using a dataset consisting of PDE solutions and their derivatives. The learned basis functions are differentiable through automatic differentiation, enabling physics-based inference. For each new problem instance, the basis functions are frozen, and the solution is determined by minimizing the governing-equation residual, together with applicable initial, boundary, regularization, and observational terms. This formulation differs from direct surrogate and operator-learning approaches because the training dataset determines the basis functions in the offline step, whereas the instance-specific solution is obtained by enforcing the governing physics online. For linear differential operators, the online lPINN problem reduces to regularized linear least squares. For nonlinear operators, it remains nonlinear but is restricted to the low-dimensional basis coefficients. In inverse problems, the reduced coefficients and unknown physical parameters are estimated jointly from physics residuals and sparse observations. Compared to vanilla PINN, pretraining basis functions amounts to estimating parameters in (nonlinear) hidden layers in an offline step and learning parameters in the last (linear) layer in the online step.

We evaluate lPINN on forward and inverse problems for the advection–diffusion equation, Burgers equation, and the nonlinear pendulum equation. The experiments compare derivative-matching and residual-based basis pretraining and demonstrate compatibility with Fourier feature networks for oscillatory solutions. Compared with vanilla PINNs, lPINN achieves lower solution and parameter errors in the reported test cases while reducing online inference times from about one order of magnitude to more than three orders of magnitude, with the largest gains generally observed when residual or measurement data are limited. Cross-resolution tests further show that the learned continuous representation can be evaluated on finer meshes without retraining and with nearly unchanged accuracy. These results demonstrate that lPINN can provide an effective alternative to PINN when the offline cost can be amortized over many forward or inverse queries.

## 1 Introduction

Physics-informed neural networks (PINNs) provide a flexible framework for forward and inverse problems governed by differential equations by representing the unknown state with a coordinate-based neural network and incorporating governing equations, initial and boundary conditions, and available observations into the training objective [1, 2]. In forward problems, the network parameters are optimized to obtain a continuous approximation of the PDE solution that satisfies the prescribed physical constraints. In inverse problems, unknown physical parameters are included among the optimization variables and estimated jointly with the state from physics constraints and observational data [3, 4]. This formulation has enabled applications in fluid and solid mechanics, transport, parameter identification, and constitutive model discovery. However, its flexibility comes at a substantial computational cost. Conventional PINN training estimates all network parameters for each problem instance, producing a high-dimensional, nonconvex, and often poorly conditioned optimization problem. Convergence can depend strongly on initialization, loss weighting, and sampling, and the network may fail to resolve physically important features. These difficulties are particularly pronounced for stiff, advection-dominated, oscillatory, and multiscale systems [5]. The PINN computational burden becomes more consequential in problems requiring multiple PDE solves, which limits PINN’s applicability in uncertainty quantification, sensitivity analysis, optimization, and time-critical prediction.

A variety of techniques have been introduced to improve the optimization and representation properties of PINNs. Quasi-Newton and related methods can increase the stability and accuracy of training for nonlinear and stiff problems [6]. Adaptive weighting and preconditioning strategies seek to balance physical, boundary, initial, and observational terms in the objective [7]. Fourier features and spectral architectures improve the representation of oscillatory or high-frequency components and can mitigate the spectral bias of standard multilayer perceptrons [8, 9]. Although these development can improve single-instance training, they do not remove the need to optimize a large nonlinear representation for every new problem. The present work addresses this repeated-optimization bottleneck by combining ideas from reduced-order modeling, differentiable neural fields, and physics-informed coefficient inference.

We propose lPINN as a neural reduced-order method for problems requiring solving multiple forward and inverse PDE problems. During the offline stage, an ensemble of numerical solutions is decomposed into a mean field and fluctuations. Coordinate-based neural networks approximate the mean and a shared collection of basis functions for reduced-order representation of the fluctuations. The basis functions may be pretrained by matching states and operator-relevant derivatives or by augmenting state reconstruction with governing-equation, initial-condition, and boundary-condition residuals. During the online stage, the coefficients are determined by minimizing the governing-equation residual together with applicable initial, boundary, regularization, and observational terms. Because the basis functions are differentiable, this inference is performed within the standard PINN framework: automatic differentiation supplies the derivatives of each basis function, while optimization is confined to the coefficients of the last layer. For linear differential operators and linear constraints, the online problem is regularized linear least squares under the usual rank or positive-regularization conditions. For nonlinear operators, it is a reduced-dimensional nonlinear least-squares problem. In inverse problems, the reduced coefficients and unknown physical parameters are estimated jointly.

Standard reduced-order methods are built on the offline–online paradigm for parametrized differential equations. In an offline stage, high-fidelity snapshots are used to construct a low-dimensional space of basis functions, commonly through proper orthogonal decomposition; in the online stage, the state is approximated in this space, and its reduced coordinates are determined from a projected governing model [10, 11]. This decomposition can yield substantial savings when the offline cost is amortized over many PDE solves for different parameter queries. Standard bases are nevertheless tied to the spatial or space–time discretization on which the snapshot vectors and reduced operators are defined. Derivatives are inherited from the full-order discretization, and transferring the basis to a different mesh or evaluating the state at arbitrary coordinates generally requires interpolation, reconstruction, or a new discretization. The lPINN method retains the offline–online structure but replaces a discrete snapshot basis with a coordinate-dependent neural basis. The resulting basis functions are continuous and differentiable with respect to space and time through automatic differentiation. Differential operators can therefore be applied directly to the basis functions at arbitrary collocation points, and the reduced coefficients can be estimated using the same strong-form, differentiable residual machinery used by PINNs.

Residual-minimizing reduced-order methods determine reduced coefficients by minimizing a governing-equation residual. Examples include least-squares Petrov–Galerkin methods and the physics-informed Karhunen–Loève expan sion (PICKLE) method [12, 13, 14, 15]. Space–time extensions, including space–time LSPG and dynamic PICKLE (dPICKLE), construct a low-dimensional representation of the complete trajectory and minimize the residual jointly over space and time [16, 17]. This reduces both spatial and temporal dimensions and avoids evolving a spatial reduced-order model through every full-order time step. In these methods, however, the basis functions, residual operators, and reduced solution are generally defined on the spatial or space–time discretization used to construct the model. lPINN adopts a related residual-minimization strategy but represents the space–time basis functions by coordinate-dependent neural networks. Each basis function depends jointly on the spatial and temporal coordinates, and a single coefficient vector represents the complete trajectory. Because the basis functions are continuous and differentiable through automatic differentiation, the differential-equation residual can be evaluated at arbitrary collocation points, independently of the offline snapshot mesh. This enables cross-resolution evaluation and permits the online collocation set to differ from, and potentially be finer than, the discretization used to generate the training ensemble. For linear differential operators, the online lPINN problem reduces to regularized linear least squares. For nonlinear operators, it remains nonlinear but is restricted to the reduced coefficients and, in inverse problems, the unknown physical parameters. This flexibility requires offline neural-basis training and, as in other snapshot-based methods, depends on the representativeness of the training ensemble.

The continuous trial space connects lPINN to neural fields, also called implicit neural representations. Neural fields map coordinates to physical or geometric quantities and provide continuous, differentiable representations that can be queried independently of the training discretization [18]. Recent continuous reduced-order modeling methods use neural fields to represent solution manifolds and evolve low-dimensional latent states, thereby removing the direct dependence of conventional ROM bases on a particular mesh [19]. Physics-informed conditional neural-field ROMs have further incorporated differential-equation residuals through automatic differentiation and have considered exact treatment of initial and boundary conditions [20]. The lPINN method shares the use of coordinate-based differentiable representations, but its reduced coordinates have a different role. Rather than learning a nonlinear decoder driven by a latent dynamical model, lPINN uses the last hidden-layer features as a linear neural trial basis and determines the instance-specific coefficient vector directly from the governing residual and available observations. This construction preserves a transparent reduced expansion, gives a linear online solve whenever the differential operator and constraints are linear, and naturally supports joint state and parameter estimation without requiring a separately learned latent evolution law or parameter-to-latent map.

A second closely related class comprises extreme-learning and random-feature physics-informed methods. Physicsinformed extreme learning machines freeze randomly generated hidden-layer parameters and determine output weights from physics and boundary constraints, leading to rapid linear solves for linear differential equations [21, 22]. Randomfeature methods similarly approximate the solution in a prescribed random nonlinear feature space and estimate the coefficients by collocation and least squares [23]. These approaches demonstrate the computational advantage of restricting training to linear output weights. Their feature spaces, however, are generally random or problem-independent and may require many features, multiscale constructions, domain decomposition, or careful rescaling to represent a parameterized solution family. In lPINN, the nonlinear features are not random. They are learned offline from an ensemble of solutions and can be trained to reproduce not only the states but also the derivatives or residual quantities required by the governing operator. The learned features therefore form an operator-compatible reduced basis adapted to the solution manifold. Online inference retains the efficiency of output-layer optimization while using a feature space informed by both the solution ensemble and, when desired, the governing physics.

Transfer-learning PINNs also reuse representations across related differential equations. Standard transfer learning initializes a new PINN from parameters trained on a related problem and then fine-tunes some or all network layers. Of particular relevance is the one-shot approach of Desai et al. [24], which freezes the hidden layers of a bundletrained PINN and adapts the final linear layer for new linear ordinary and partial differential equations. That work establishes that final-layer adaptation can convert a linear forward problem into a linear least-squares solve. lPINN builds on the general efficiency principle of frozen nonlinear features but differs in its objective and scope. Its offline representation is explicitly constructed as a mean-plus-fluctuation reduced basis from numerical solution ensembles, with derivative-matching or residual-based objectives designed to make the basis compatible with the target operator. Its online formulation covers nonlinear forward problems, for which only the reduced coefficients are optimized, and inverse problems, for which coefficients and unknown physical parameters are estimated jointly from physics and sparse data. Consequently, lPINN is better viewed as a physics-corrected neural reduced-basis method than as conventional parameter fine-tuning.

Neural operators take a different route to amortizing PDE solution cost. DeepONet learns an operator through branch and trunk networks, while the Fourier neural operator learns mappings between function spaces using integral-kernel layers parameterized in Fourier space [25, 26]. Once trained, these methods directly map coefficients, source terms, initial conditions, or boundary data to a solution field and can evaluate new instances without solving a problem-specific optimization. Physics-informed operator-learning methods, including physics-informed DeepONets and physicsinformed neural operators, incorporate governing-equation residuals into operator training to reduce data requirements and improve physical consistency [27, 28]. lPINN differs from both purely data-driven and physics-informed neural operators because it does not learn the complete input-to-solution map. The offline data are used to learn a trial space, not a direct predictor of the final state or its reduced coordinates. For each new instance, the coefficients are recomputed by enforcing that instance’s governing equation, conditions, and observations. This online physical correction can adapt the prediction to the particular query and incorporate sparse measurements without retraining a global operator. It also avoids constructing an explicit encoding of every possible parameter, source, initial condition, or boundary function. The trade-off is that lPINN requires a small online solve, whereas a neural operator usually provides a single feed-forward prediction.

The lPINN formulation provides several methodological advantages. First, it combines the dimensional reduction and offline–online amortization of reduced-basis methods with a continuous neural representation that can be evaluated and differentiated at arbitrary coordinates. Second, it performs problem-specific residual minimization, unlike surrogates, so the numerical snapshots define the admissible trial space without completely determining the solution at a new query. Third, operator-compatible pretraining can concentrate representational capacity on the states and derivatives that control the online residual, in contrast to random-feature methods. Fourth, the linear dependence on the last-layer coefficients yields a convex online problem for linear operators and substantially reduces the dimension of nonlinear forward and inverse optimization. Fifth, observations can be incorporated directly into the reduced physics objective, allowing state and parameter inference in a unified formulation. These benefits come with the standard limitations of snapshot-based model reduction: the offline ensemble must adequately cover the anticipated solution manifold, and the offline simulation and basis-training costs must be amortized over sufficiently many queries.

The principal contributions of this work are as follows:

1. We develop operator-compatible continuous neural basis functions learned from numerical solution ensembles. The coordinate-dependent basis functions are differentiable through automatic differentiation and can be pretrained using state and derivative information or state data augmented by PDE residuals.

2. We formulate forward prediction as residual-based coefficient inference within the pretrained basis function space. The online problem reduces to regularized linear least squares for linear operators and to reduceddimensional nonlinear least squares for nonlinear operators, avoiding repeated optimization of the full nonlinear network.

3. We extend the reduced formulation to inverse problems by jointly estimating the basis function coefficients and unknown physical parameters from the PDE residuals and sparse observations.

4. We demonstrate compatibility with representation-enhancing architectures, including Fourier features for oscillatory solutions, and investigate derivative-matching and residual-based strategies for constructing operatorcompatible neural basis functions.

The numerical experiments consider forward and inverse problems for the advection–diffusion equation, Burgers’ equation, and the nonlinear pendulum equation. The numerical study evaluates the central hypothesis of this work: that neural representation learning can be separated from problem-specific physics inference so repeated forward and inverse problems can be solved without optimizing the full nonlinear network for each new instance. We therefore use vanilla PINNs as the primary baseline in the following experiments. We also provide a comparison with a residual-minimizing ROM [17]. A systematic comparison with classical projection-based and nonlinear ROMs is outside the scope of the present work and is reserved for future study. The experiments examine predictive accuracy, parameter estimation, online computational cost, sensitivity to residual and measurement sampling, the effects of basis dimension and regularization, and transfer of the learned continuous representation across evaluation resolutions.

## 2 Methods

## 2.1 Vanilla Physics-Informed Neural Network (PINN)

Consider a PDE of the form

$$
\mathcal { L } ( h ( x , t ) ; y ) = 0 , \qquad x \in \Omega , \quad t \in ( 0 , T ] ,\tag{1}
$$

subject to the initial conditions

$$
h ( x , 0 ) = h _ { 0 } ( x ) , \quad \partial h / \partial t ( x , 0 ) = h _ { 1 } ( x ) \qquad x \in \Omega ,\tag{2}
$$

and the boundary condition

$$
\boldsymbol { \mathcal { B } } ( \boldsymbol { h } ( \boldsymbol { x } , t ) ) = \boldsymbol { g } ( \boldsymbol { x } , t ) , \qquad \boldsymbol { x } \in \Gamma , \quad t \in ( 0 , T ] ,\tag{3}
$$

where $\mathcal { L }$ is a known differential operator, Ω is the spatial domain with boundary $\Gamma = \partial \Omega , T$ is the time horizon, $h ( x , t )$ is the solution, y denotes the physical parameter(s) of the system, $h _ { 0 } ( x )$ and $h _ { 1 } ( x )$ are the zero and first order initial conditions, B is the boundary operator, and $g ( x , t )$ is the prescribed boundary data.

In the PINN framework, the solution $h ( x , t )$ is approximated by a neural network $\hat { h } ( x , t ; \theta )$ with trainable parameters θ. In forward problems, the physical parameter y is known, and the network parameters are determined by minimizing a residual least-squares loss function:

$$
\begin{array} { r l } { \displaystyle \theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \left( \frac { 1 } { N _ { \mathrm { r e s } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r e s } } } \left[ \mathcal { L } ( \hat { h } ( x _ { i } ^ { r } , t _ { i } ^ { r } ; \theta ) ; y ) \right] ^ { 2 } \right. } & { } \\ { \displaystyle } & { \left. + \frac { \lambda _ { \mathrm { B C } } } { N _ { \mathrm { R C } } } \sum _ { j = 1 } ^ { N _ { \mathrm { R C } } } \left[ B ( \hat { h } ( x _ { j } ^ { \mathrm { R C } } , t _ { j } ^ { \mathrm { R C } } ; \theta ) ) - g ( x _ { j } ^ { \mathrm { R C } } , t _ { j } ^ { \mathrm { R C } } ) \right] ^ { 2 } \right. } \\ { \displaystyle } & { + \left. \frac { \lambda _ { \mathrm { I C , 0 } } } { N _ { \mathrm { I C } } } \sum _ { k = 1 } ^ { N _ { \mathrm { I C } } } \left[ \hat { h } ( x _ { k } ^ { \mathrm { I C } } , 0 ; \theta ) - h _ { 0 } ( x _ { k } ^ { \mathrm { I C } } ) \right] ^ { 2 } + \frac { \lambda _ { \mathrm { I C , 1 } } } { N _ { \mathrm { I C } } } \sum _ { k = 1 } ^ { N _ { \mathrm { I C } } } \left[ \frac { \partial \hat { h } ( x _ { k } ^ { \mathrm { I C } } , 0 ; \theta ) } { \partial t } - h _ { 1 } ( x _ { k } ^ { \mathrm { I C } } ) \right] ^ { 2 } + \lambda \| \theta \| _ { 2 } ^ { 2 } \right) , } \end{array}\tag{4}
$$

where $\lambda _ { \mathrm { B C } } , \lambda _ { \mathrm { I C } , 0 }$ , and $\lambda _ { \mathrm { I C } , 1 }$ are weights associated with the boundary and initial condition terms, respectively; λ is the L2 regularization weight; and $N _ { \mathrm { r e s } } , \bar { N } _ { \mathrm { B C } }$ , and $N _ { \mathrm { I C } }$ denote the numbers of collocation points in the interior, boundary, and initial domains, respectively.

In inverse problems, y is treated as an additional trainable variable and is estimated together with θ using both physics constraints and measurements of h, $\{ ( X _ { i } ^ { s } , T _ { i } ^ { s } , h _ { i } ^ { s } ) \} _ { i = 1 } ^ { N _ { \mathrm { m } } }$ , where $X _ { i } ^ { s }$ and $T _ { i } ^ { s }$ denote the space and time coordinates of the measurements and $h _ { i } ^ { s }$ are the measured values. The inverse PINN formulation is given by

$$
\begin{array} { l } { \displaystyle ( \theta ^ { * } , y ^ { * } ) = \arg \operatorname* { m i n } _ { \theta , y } \left( \frac { 1 } { N _ { \mathrm { t e x } } } \sum _ { i = 1 } ^ { N _ { \mathrm { m e } } } \left[ E ( \hat { h } ( x _ { i } ^ { \mathrm { r } } , t _ { i } ^ { \mathrm { r } } ; \theta ) ; y ) \right] ^ { 2 } \right. } \\ { \displaystyle \qquad + \left. \frac { \lambda _ { \mathrm { B C } } } { N _ { \mathrm { B C } } } \sum _ { j = 1 } ^ { N _ { \mathrm { B C } } } \left[ B ( \hat { h } ( x _ { j } ^ { \mathrm { R C } } , t _ { j } ^ { \mathrm { R C } } ; \theta ) ) - g ( x _ { j } ^ { \mathrm { R C } } , t _ { j } ^ { \mathrm { R C } } ) \right] ^ { 2 } \right. } \\ { \displaystyle \qquad + \left. \frac { \lambda _ { \mathrm { I C , 0 } } } { N _ { \mathrm { I C } } } \sum _ { k = 1 } ^ { N _ { \mathrm { C } } } \left[ \hat { h } ( x _ { k } ^ { \mathrm { I C } } , 0 ; \theta ) - h _ { 0 } ( x _ { k } ^ { \mathrm { I C } } ) \right] ^ { 2 } + \frac { \lambda _ { \mathrm { I C , 1 } } } { N _ { \mathrm { I C } } } \sum _ { k = 1 } ^ { N _ { \mathrm { E C } } } \left[ \frac { \partial \hat { h } ( x _ { k } ^ { \mathrm { I C } } ; \theta ) } { \partial l } - h _ { 1 } ( x _ { k } ^ { \mathrm { I C } } ) \right] ^ { 2 } \right. } \\ { \displaystyle \qquad + \left. \frac { \lambda _ { \mathrm { I d t a n } } } { N _ { \mathrm { m } } } \sum _ { i = 1 } ^ { N _ { \mathrm { m } } } \left[ \hat { h } ( X _ { i } ^ { \mathrm { s } } , T _ { i } ^ { \mathrm { s } } ; \theta ) - h _ { i } ^ { \mathrm { s } } \right] ^ { 2 } + \lambda | | \theta | | _ { 2 } ^ { 2 } \right) , } \end{array}\tag{5}
$$

where the second-to-last term represents the data mismatch between the network prediction and the measurement data.

The optimization problems in Egs. (4) and (5) are nonlinear least-squares problems in the neural network parameters. As the number of trainable parameters increases, the optimization may become severely ill-conditioned, leading to slow convergence and reduced robustness, especially for stiff and nonlinear PDEs.

## 2.2 Linearized Physics-Informed Neural Network (lPINN)

Our objective is to simplify the training of PINNs by pretraining some of the DNN’s parameters using an ensemble of PDE solutions that we refer to as the training dataset:

$$
D _ { \mathrm { t r a i n } } = \Big \{ ( g ^ { ( i ) } , y ^ { ( i ) } , h _ { 0 } ^ { ( i ) } , h _ { 1 } ^ { ( i ) } ) \mapsto h ^ { ( i ) } \Big \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } ,
$$

where each sample consists of different instances of the PDE parameters, initial, and boundary conditions, and the corresponding PDE solution vector h obtained on a mesh with $N _ { x }$ spatial nodes and $N _ { t }$ time steps.

We start by decomposing the PDE solutions into the sample mean $\bar { h } ( x , t )$ and fluctuations $h ^ { \prime } ( x , t )$

$$
h ( x , t ) = \bar { h } ( x , t ) + h ^ { \prime } ( x , t ) .\tag{6}
$$

Both $\bar { h } ( x , t )$ and $h ^ { \prime } ( x , t )$ are approximated by DNNs. The fluctuation field is approximated by the DNN

$$
h ^ { \prime } ( x , t ) \approx \hat { h } ^ { \prime } ( x , t ; \pmb { \theta } ) = \mathcal { N } \mathcal { N } ( z ; \pmb { \theta } ) ,\tag{7}
$$

where $\boldsymbol { z } = [ x , t ] ^ { \mathrm { T } }$ denotes the input vector and θ is the collection of trainable network parameters. For generality, we consider a fully connected network with N hidden layers,

$$
\mathcal { N N } ( z ; \pmb { \theta } ) = \rho _ { N + 1 } \Big ( \rho _ { N } \big ( \rho _ { N - 1 } \big ( \cdot \cdot \cdot \rho _ { 1 } ( z ) \big ) \big ) \Big ) ,\tag{8}
$$

where each layer is defined by

$$
z _ { i + 1 } = \rho _ { i } ( W _ { i } z _ { i } + b _ { i } ) , \qquad i = 1 , \ldots , N ,\tag{9}
$$

with $z _ { 1 } = z$ . Here, $W _ { i }$ and $\mathbf { } _ { b _ { i } }$ are the weights and biases of layer $i ,$ respectively, and $\rho _ { i }$ is the activation function. In this work, we use the hyperbolic tangent activation function for the hidden layers and the identity function for the output layer. Thus, the full parameter set is

$$
\pmb { \theta } = ( \pmb { W } _ { 1 : N + 1 } , \pmb { b } _ { 1 : N + 1 } ) .
$$

Because of the choice of activation functions, the network is nonlinear with respect to the hidden-layer parameters

$$
\tilde { \pmb { \theta } } = ( \pmb { W } _ { 1 : N } , \pmb { b } _ { 1 : N } ) ,
$$

and linear with respect to the output layer parameters $W _ { N + 1 }$ and $b _ { N + 1 }$ . We denote the size of the last hidden layer by $N _ { \eta }$ and note that the output layer has a single output for a scalar state variable. Then, $W _ { N + 1 }$ is the $N _ { \eta }$ -dimensional

vector and $b _ { N + 1 }$ is a scalar. Because we use the DNN to model fluctuations around the mean, we set $b _ { N + 1 } = 0$ . Finally, we can express fluctuations as

$$
\hat { h } ^ { \prime } ( z ; \pmb { \theta } ) = \hat { h } ^ { \prime } ( z ; \tilde { \pmb { \theta } } , \pmb { W } _ { N + 1 } ) = \pmb { W } _ { N + 1 } \cdot \pmb { \psi } ( z ; \tilde { \pmb { \theta } } ) ,\tag{10}
$$

where $\boldsymbol { \psi } = [ \psi _ { 1 } , . . . , \psi _ { N _ { n } } ] ^ { T }$ and $\psi _ { i } ( z ; \tilde { \pmb { \theta } } )$ is the output of ith neuron of the last hidden layer.

Eq (10) provides an expansion of $h ^ { \prime } ( x , t )$ in terms of the product of space-time-dependent parametric basis functions $\psi _ { i } ( x , t ; \pmb { \theta } )$ and coefficients $W _ { N + 1 }$ . Because $\psi _ { i } ( x , t ; \tilde { \pmb { \theta } } )$ are defined by the DNN layers, we refer to them as neural basis functions.

As in finite element, POD-ROM, and other reduced-order methods, we assume that the solution manifold can be approximated by a set of basis functions that are independent of the PDE parameters, initial conditions, and boundary conditions. The effect of PDE parameters, initial, and boundary conditions on the solution is captured by $W _ { N + 1 }$

In standard PINN, $\tilde { \theta }$ and $W _ { N + 1 }$ are learned jointly by minimizing the PDE residuals, which results in a highly nonlinear least-squares minimization problem.

The key idea of lPINN is to learn the basis functions, i.e., to estimate ${ \tilde { \theta } } ,$ from a training dataset $\{ y ^ { ( i ) }  { h ^ { ( i ) } } \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } }$ during an “offline” stage, and only estimate $W _ { N + 1 }$ “online” by minimizing the PDE residuals. In the training dataset, $\boldsymbol y ^ { ( i ) }$ are the samples of the parameters from the desired range and $\mathbf { \Sigma } _ { h } ( i )$ are the corresponding solutions obtained (in general) numerically on the mesh with $N _ { e }$ elements or grid points.

The mean field is approximated by another DNN that is also trained offline:

$$
\begin{array} { r } { \bar { h } ( x , t ) \approx \hat { \bar { h } } ( x , t ; \gamma ) = { \mathcal N } { \mathcal N } ( z ; \gamma ) , } \end{array}\tag{11}
$$

where $\gamma$ is a collection of trainable parameters. The DNN $\hat { \bar { h } } ( x , t ; \gamma )$ is trained using the sample mean values:

$$
\bar { h } \approx \frac { 1 } { N _ { \mathrm { t r a i n } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } h ^ { ( i ) } .\tag{12}
$$

We would like $\hat { \bar { h } } ( \boldsymbol { x } , t ; \gamma )$ and $\hat { h } ^ { \prime } ( \pmb { x } , t ; \pmb { \theta } )$ to provide an accurate approximation of not just $h ( x , t ; \gamma )$ but also all its derivatives present in the governing equation. Assume, for generality, that $\mathcal { L }$ contains first and second space and time derivatives. Then, we supplement the h<sup>¯</sup> data with its derivatives $\partial _ { t } \bar { h } , \partial _ { x } \bar { h } , \partial _ { x x } \bar { h }$ , and $\partial _ { t t } \bar { h }$ . These derivatives can be computed numerically from $\bar { h }$ on the same space-time mesh that was used to compute $\bar { h }$

Then, $\gamma$ can be obtained by solving an optimization problem, which minimizes the mismatch between $\hat { \bar { h } } ( z ; \gamma ) , \bar { h }$ , and their derivatives:

$$
\begin{array} { r l } & { \gamma ^ { * } = \underset { \gamma } { \arg \operatorname* { m i n } } \left[ \| \hat { \bar { \boldsymbol { h } } } ( \gamma ) - \bar { \boldsymbol { h } } \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \| \partial _ { t } \hat { \bar { \boldsymbol { h } } } ( \gamma ) - \partial _ { t } \bar { \boldsymbol { h } } \| _ { 2 } ^ { 2 } \right. } \\ & { \quad \quad \quad \quad \left. + \lambda _ { 2 } \| \partial _ { x } \hat { \bar { \boldsymbol { h } } } ( \gamma ) - \partial _ { x } \bar { \boldsymbol { h } } \| _ { 2 } ^ { 2 } + \lambda _ { 3 } \| \partial _ { x x } \hat { \bar { \boldsymbol { h } } } ( \gamma ) - \partial _ { x x } \bar { \boldsymbol { h } } \| _ { 2 } ^ { 2 } \right. } \\ & { \quad \quad \quad \quad \left. + \lambda _ { 4 } \| \partial _ { t t } \hat { \bar { \boldsymbol { h } } } ( \gamma ) - \partial _ { t t } \bar { \boldsymbol { h } } \| _ { 2 } ^ { 2 } + \lambda \| \gamma \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{13}
$$

where $\hat { \bar { h } } ( \gamma ) , \partial _ { t } \hat { \bar { h } } ( \gamma ) , \partial _ { x } \hat { \bar { h } } ( \gamma ) , \partial _ { x x } \hat { \bar { h } } ( \gamma )$ , and $\partial _ { t t } \hat { \bar { h } } ( \gamma )$ are the vectors of the $\widehat { \bar { h } } ( x , t ; \gamma )$ values evaluated on the spacetime mesh, as well as the derivatives of $\widehat { \bar { h } } ( x , t ; \gamma )$ evaluated on the same mesh. The weights $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , \lambda _ { 4 }$ and $\lambda$ are used to balance different terms and are selected through a grid search.

Similarly, $\tilde { \theta }$ is estimated from the dataset $\{ h ^ { \prime ( i ) } , \partial _ { t } h ^ { \prime ( i ) } , \partial _ { t t } h ^ { \prime ( i ) } , \partial _ { x } h ^ { \prime ( i ) } , \partial _ { x x } h ^ { \prime ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } }$ , where $\pmb { h } ^ { \prime ( i ) } = \pmb { h } ^ { ( i ) } - \overline { { \pmb { h } } }$

The coefficients $\{ W _ { N + 1 } ^ { ( i ) } \} _ { i = 1 } ^ { { N } _ { \mathrm { t r a i n } } }$ are found by solving the minimization problem:

$$
\begin{array} { r l } { \displaystyle \tilde { \theta } ^ { * } , \{ W _ { N + 1 } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } = \operatorname * { a r g m } _ { \{ \theta , \theta _ { 1 } \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } } \ : \sum _ { i = 1 } ^ { N _ { \mathrm { r e a i n } } } [ \| \hat { h } ^ { \prime ( i ) } ( \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } ) - h ^ { \prime ( i ) } \| _ { 2 } ^ { 2 } + \lambda _ { 1 } \| \partial _ { t } \hat { h } ^ { \prime ( i ) } ( \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } ) - \partial _ { t } h ^ { \prime ( i ) } \| _ { 2 } ^ { 2 }  } & { } \\ {  \{ W _ { N + 1 } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } \ : \sum _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } \ : [ \| \hat { h } ^ { \prime ( i ) } ( \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } ) - \partial _ { x } h ^ { \prime ( i ) } \| _ { 2 } ^ { 2 } + \lambda _ { 3 } \| \partial _ { x x } \hat { h } ^ { \prime ( i ) } ( \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } ) - \partial _ { x x } h ^ { \prime ( i ) } \| _ { 2 } ^ { 2 }   } & { } \\ {   + \lambda _ { 4 } \| \partial _ { t } \hat { h } ^ { \prime ( i ) } ( \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } ) - \partial _ { t } h ^ { \prime ( i ) } \| _ { 2 } ^ { 2 } ] + \lambda ( \| \tilde { \theta } \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } \| \mathbf { W } _ { N + 1 } ^ { ( i ) } \| _ { 2 } ^ { 2 } ) } \end{array}\tag{14}
$$

where $\hat { \pmb { h } } ^ { \prime ( i ) } ( \tilde { \pmb { \theta } } , \pmb { W } _ { N + 1 } ^ { ( i ) } )$ is the vector of $\hat { h } ^ { \prime } ( z ; \tilde { \pmb { \theta } } , \pmb { W } _ { N + 1 } ^ { ( i ) } )$ predictions evaluated on the space-time mesh $N _ { x } \times N _ { t }$ and $\partial _ { t } \hat { { \cal h } } ^ { \prime ( i ) } , \partial _ { t t } \hat { { \cal h } } ^ { \prime ( i ) } , \partial _ { x } \hat { { \cal h } } ^ { \prime ( i ) }$ , and $\partial _ { x x } \hat { h } ^ { \prime ( i ) }$ are the vectors of the derivatives of $\hat { h } ^ { \prime } ( z ; \tilde { \pmb { \theta } } , \pmb { W } _ { N + 1 } ^ { ( i ) } )$ evaluated on the same mesh. The estimation of $\tilde { \pmb { \theta } }$ from Eq (14) also requires estimating $\{ W _ { N + 1 } ^ { ( i ) } \} _ { i = 1 } ^ { { N } _ { \mathrm { t r a i n } } }$ , the parameters $W _ { N + 1 }$ specific to each sample in the training dataset, even though these parameters are not part of the pretrained DNN network.

Evaluating derivatives of the solution incurs additional computational cost. Although this cost is substantially smaller than the cost of generating the solution ensemble, it can still increase offline training time and storage requirements and should therefore be avoided when possible. We find that for some PDE types, $\gamma$ and $\tilde { \pmb { \theta } }$ can be evaluated jointly using the samples $\{ h ^ { ( i ) } \} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } }$ and PDE residual constraints as

$$
\begin{array} { r l } { \gamma ^ { * } , \widehat { \theta } ^ { * } , \left\{ W _ { N + 1 } ^ { * ( i ) } \right\} _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } = \underset { \{ W _ { N + 1 } ^ { ( i ) } , \widehat { \theta } , _ { W _ { N + 1 } ^ { ( i ) } } \} } { \operatorname* { m i n } } \underset { \left[ \| \widehat { h } ( \gamma ) - \widehat { h } \| _ { 2 } ^ { 2 } + \| \widehat { h } ^ { \prime ( i ) } ( \widehat { \theta } , W _ { N + 1 } ^ { ( i ) } ) - h ^ { \prime ( i ) } \| _ { 2 } ^ { 2 } \right] } { \operatorname* { m i n } } } & { } \\ { + \lambda _ { f } \| \mathcal { R } ^ { ( i ) } ( \gamma , \widehat { \theta } , W _ { N + 1 } ^ { ( i ) } ) \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { U C } , 0 } \| \mathcal { R } _ { 1 \mathrm { C } , 0 } ^ { ( i ) } ( \gamma , \widehat { \theta } , W _ { N + 1 } ^ { ( i ) } ) \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { U C } , 1 } \| \mathcal { R } _ { 1 \mathrm { C } , 1 } ^ { ( i ) } ( \gamma , \widehat { \theta } , W _ { N + 1 } ^ { ( i ) } ) \| _ { 2 } ^ { 2 } } & { } \\ { + \lambda _ { \mathrm { B C } } \| \mathcal { R } _ { \mathrm { B C } } ^ { ( i ) } ( \gamma , \widehat { \theta } , W _ { N + 1 } ^ { ( i ) } ) \| _ { 2 } ^ { 2 } \bigg ] + \lambda \left( \| \gamma \| _ { 2 } ^ { 2 } + \| \widehat { \theta } \| _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } \| \mathbf { W } _ { N + 1 } ^ { ( i ) } \| _ { 2 } ^ { 2 } \right) } & { } \end{array}
$$

where the residual operators are defined as

(15)

$$
\begin{array} { r } { \mathcal { R } ( x , t ; y ; \gamma , \tilde { \theta } , W _ { N + 1 } ) = \mathcal { L } \Big ( \hat { \bar { h } } ( x , t ; \gamma ) + \hat { h } ^ { \prime } ( x , t ; \tilde { \theta } , W _ { N + 1 } ) ; y \Big ) , } \end{array}\tag{16}
$$

$$
\mathcal { R } _ { \mathrm { B C } } ( \boldsymbol { x } , t ; \boldsymbol { g } ; \gamma , \tilde { \theta } , W _ { N + 1 } ) = \mathcal { B } \Big ( \hat { \textmd h } ( \boldsymbol { x } , t ; \gamma ) + \hat { h } ^ { \prime } ( \boldsymbol { x } , t ; \tilde { \theta } , W _ { N + 1 } ) \Big ) - \boldsymbol { g } ( \boldsymbol { x } , t ) ,\tag{17}
$$

$$
\mathcal { R } _ { \mathrm { I C } , 0 } ( x ; h _ { 0 } ; \gamma , \tilde { \theta } , W _ { N + 1 } ) = \hat { \bar { h } } ( x , 0 ; \gamma ) + \hat { h } ^ { \prime } ( x , 0 ; \tilde { \theta } , W _ { N + 1 } ) - h _ { 0 } ( x ) ,\tag{18}
$$

and

$$
\mathcal { R } _ { \mathrm { I C } , 1 } ( x ; h _ { 0 } ; \gamma , \tilde { \theta } , W _ { N + 1 } ) = \frac { \partial \hat { \bar { h } } ( x , 0 ; \gamma ) } { \partial t } + \frac { \partial \hat { h } ^ { \prime } ( x , 0 ; \tilde { \theta } , W _ { N + 1 } ) } { \partial t } - h _ { 1 } ( x ) .\tag{19}
$$

Here, $\mathcal { R } ^ { ( i ) }$ is the vector with components $\mathcal { R } ( x , t ; y ^ { ( i ) } ; \gamma , \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } )$ evaluated at randomly sampled residual collocation points on the space-time domain; $\mathcal { R } _ { \mathrm { I C } , 0 } ^ { ( i ) }$ and $\mathcal { R } _ { \mathrm { I C } , 1 } ^ { ( i ) }$ are the vector with components $\mathcal { R } _ { \mathrm { I C } , 0 } ( x ; h _ { 0 } ^ { ( i ) } ; \gamma , \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } )$ and $\mathcal { R } _ { \mathrm { I C } , 1 } ( x ; h _ { 1 } ^ { ( i ) } ; \gamma , \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } )$ , respectively, evaluated on the space domain at $t = 0 ; \mathcal { R } _ { \mathrm { B C } } ^ { ( i ) }$ is the vector with components $\mathcal { R } _ { \mathrm { B C } } ( x , t ; g ^ { ( i ) } ; \gamma , \tilde { \theta } , W _ { N + 1 } ^ { ( i ) } )$ evaluated on the time domain on the boundary; and $y ^ { ( i ) } , g ^ { ( i ) } , h _ { 0 } ^ { ( i ) } , h _ { 1 } ^ { ( i ) }$ are components of the $D _ { t r a i n }$ training dataset.

After the offline pretraining stage, the PDE solution for any $y , g ,$ and $h _ { 0 }$ is obtained by estimating ${ \bf W } _ { N + 1 }$ from the minimization problem

$$
\begin{array} { r l r } { \mathbf { W } _ { N + 1 } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { W } _ { N + 1 } } \Big [ } & { \| \mathcal { R } ( y ; \gamma ^ { * } , \tilde { \theta } ^ { * } , \mathbf { W } _ { N + 1 } ) \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { B C } } \| \mathcal { R } _ { \mathrm { B C } } ( g ; \gamma ^ { * } , \tilde { \theta } ^ { * } , \mathbf { W } _ { N + 1 } ) \| _ { 2 } ^ { 2 } } & { ( 2 0 ) } \\ & { + } & { \lambda _ { \mathrm { T C } , 0 } \| \mathcal { R } _ { \mathrm { I C } , 0 } ( h _ { 0 } ; \gamma ^ { * } , \tilde { \theta } ^ { * } , \mathbf { W } _ { N + 1 } ) \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { T C } , 1 } \| \mathcal { R } _ { \mathrm { T C } , 1 } ( h _ { 1 } ; \gamma ^ { * } , \tilde { \theta } ^ { * } , \mathbf { W } _ { N + 1 } ) \| _ { 2 } ^ { 2 } + \lambda \| \mathbf { W } _ { N + 1 } \| _ { 2 } ^ { 2 } \Big ] , } \end{array}
$$

where $\gamma ^ { * }$ and ${ \tilde { \theta } } ^ { * }$ are obtained at the offline pretraining. The basis functions approximately satisfy the initial and boundary conditions. However, our numerical results show that adding the initial and boundary condition penalty terms, i.e., setting λ<sub>BC</sub> $> 0 , \lambda _ { \mathrm { I C } , 0 } > 0 , \lambda _ { \mathrm { I C } , 1 } > 0$ in general, yields more accurate solutions.

Alternatively, we can use the Galerkin projection method to estimate ${ \bf W } _ { N + 1 }$ as a solution of the equation

$$
\begin{array} { r } { \Psi _ { r } ^ { T } \mathcal { R } \left( y ; \gamma ^ { * } , \tilde { \pmb { \theta } } ^ { * } , \mathbf { W } _ { N + 1 } ^ { * } \right) = \mathbf { 0 } , } \end{array}\tag{21}
$$

where $\Psi _ { r } \in \mathbb { R } ^ { N _ { \mathrm { r e s } } \times N _ { \eta } }$ is the matrix of pretrained neural basis functions evaluated at the residual points, whose ith row is $\psi ( z _ { i } ; \tilde { \theta } ^ { * } ) ^ { T }$ . In our numerical experiments, the Galerkin formulation produced errors similar to those obtained by residual least-square lPINN formulation in Eq (20). We therefore use the residual least-squares lPINN formulation in the results reported below.

In lPINN, the term “linearized” refers to the linear dependence of the functional representation on $W _ { N + 1 }$ , the parameter vector estimated in the PINN-like online stage. It does not imply linearizing the governing differential operator. If L and B are linear operators, then Eq. (20) is a regularized linear least-squares problem. It admits a unique solution when the augmented least-squares matrix has full column rank or when strictly positive quadratic regularization makes the objective strictly convex. For nonlinear operators, the reduced problem remains nonlinear, but it is still low-dimensional and can be solved iteratively using standard nonlinear least-squares methods.

The inverse PDE problem is formulated in lPINN similarly to the PINN method. Consider an inverse problem in which one or more of the parameters $y , g , h _ { 0 }$ , and $h _ { 1 }$ are unknown and must be inferred from the observations $\pmb { u } ^ { * } = [ u _ { j } ^ { * } ] _ { j = 1 } ^ { N _ { \mathrm { m } } }$ of h at locations $\pmb { x } _ { d } = [ x _ { j } ] _ { i = 1 } ^ { N _ { \mathrm { m } } }$ and time instances $\mathbf { \ d _ { t } } _ { d } = [ t _ { j } ] _ { j = 1 } ^ { N _ { \mathrm { m } } }$ . The parameters can be estimated jointly with $W _ { N + 1 }$ at the online stage as the solution of the minimization problem:

$$
\begin{array} { r l r } & { } & { ( W _ { N + 1 } ^ { * } , y ^ { * } , g ^ { * } , h _ { 0 } ^ { * } ) = \operatorname * { a r g m } _ { W _ { N + 1 } , y , g , h _ { 0 } } \Big [ \| \mathcal { R } ( y ; W _ { N + 1 } ) \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { B C } } \| \mathcal { R } _ { B C } ( g ; W _ { N + 1 } ) \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { I C } , 0 } \| \mathcal { R } _ { \mathrm { I C } , 0 } ( h _ { 0 } , W _ { N + 1 } , y ) \| _ { 2 } ^ { 2 } } \\ & { } & { +  \lambda _ { \mathrm { I C } , 1 } \| \mathcal { R } _ { \mathrm { I C } , 1 } ( h _ { 1 } , W _ { N + 1 } , y ) \| _ { 2 } ^ { 2 } + \lambda \| \mathbf { W } _ { N + 1 } \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { d a t a } } \| \hat { h } _ { d } ( W _ { N + 1 } ) - h ^ { * } \| _ { 2 } ^ { 2 } \Big ] , } \end{array}
$$

where L2 regularization is used on the $\mathbf { W } _ { N + 1 }$ parameters, and ${ h _ { d } } ( W _ { N + 1 } )$ in the data term is the vector with components $\hat { \bar { h } } ( x _ { i } , t _ { i } ; \gamma ^ { * } ) + \hat { h } ^ { \prime } ( x _ { i } , t _ { i } ; \tilde { \pmb { \theta } } ^ { * } , { \pmb { W } } _ { N + 1 } )$ . Because the basis functions approximately satisfy the initial and boundary conditions, we find that in this loss function, $\lambda _ { \mathrm { B C } } , \lambda _ { \mathrm { I C } , 0 }$ and $\lambda _ { \mathrm { I C } , 1 }$ can be set to zero without incurring significant errors in the inverse solution. The weighting coefficients in the loss function, including λ and $\lambda _ { \mathrm { d a t a } } ,$ can be estimated as in PINN. In this work, we use a grid search method to select the optimal combination of the weighting coefficients. The schematic representation of the lPINN method for forward and inverse problems is shown in Figure 1.

![](images/78e3647a7a7a804ae580eaf9eb7b2e10c9502ac93bf5f59bd363e01180efc33f.jpg)  
Figure 1: The schematic representation of the lPINN method. All parameters in the mean DNN and the parameters associated with the fluctuation DNN layers are estimated offline using the training dataset. The remaining fluctuation-DNN parameters in the last layer are determined for the specified parameters using the physics-informed loss.

A key consequence of the neural representation is that the mesh used to generate the offline solution ensemble and the collocation sets used during offline and online training are decoupled. Once the mean function $\hat { \overline { { h } } } ( z , \gamma ^ { * } )$ and basis functions $\displaystyle p ( z ; \tilde { \theta } ^ { * } )$ have been learned, both $\hat { h } ( z ; { \cal W } _ { N + 1 } ) = \ddot { \bar { h } } ( z , \gamma ^ { * } ) + \psi ( z ; \tilde { \theta } ^ { * } ) \cdot { \cal W } _ { N + 1 }$ together with the derivatives required by the differential operator, can be evaluated at arbitrary coordinates z through automatic differentiation. Consequently, a basis learned from a solution ensemble generated on a coarse mesh can be used to represent and evaluate the solution at a finer resolution without retraining. The results presented below indicate that this cross-resolution accuracy does not require a large number of residual points during online inference because only a relatively small number of reduced coefficients is estimated. Instead, the attainable accuracy is governed primarily by the quality of the pretrained mean and basis functions. Within the range investigated here, this representation accuracy can be improved by increasing the number of solution samples in the offline training ensemble. Increasing the number of residual points during residual-based offline pretraining can also improve the learned representation, although we observed a smaller improvement than that obtained by increasing the number of training samples. We define lPINN superresolution as the ability of an lPINN trained on a coarse-resolution ensemble to produce predictions that are more accurate than numerical solutions computed on finer meshes when both are evaluated against the same high-fidelity reference solution.

## 3 Numerical Examples

Unless otherwise stated, all reported solution and parameter errors are computed from the arithmetic-mean prediction or parameter estimate over 10 independent runs, and the reported computational times denote the average online training time per run.

In the comparison studies, the PINN network and the mean and fluctuation networks used in lPINN have the same size and architecture, including the same number of hidden layers, the same width in each hidden layer, and the same activation functions. Thus, each individual network used in lPINN is matched in size and architecture to the PINN network. The exception is a comparison in Section 3.3 where PINN and lPINN use different Fourier frequency scales.

## 3.1 Advection Diffusion Equation

We first evaluate the proposed lPINN framework on the one-dimensional advection–diffusion equation

$$
\frac { \partial u } { \partial t } + V \frac { \partial u } { \partial x } = D \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } ,\tag{23}
$$

subject to the initial condition

$$
u ( x , 0 ) = 0 ,\tag{24}
$$

and the boundary conditions

$$
u ( 0 , t ) = u _ { 0 } ,\tag{25}
$$

where $u _ { 0 } = 1$ , and

$$
\left. \frac { \partial u } { \partial x } \right| _ { x = L } = 0 .\tag{26}
$$

In the forward problem, the goal is to predict the solution for $V \in [ V _ { \operatorname* { m i n } } , V _ { \operatorname* { m a x } } ]$ and $D \in [ D _ { \operatorname* { m i n } } , D _ { \operatorname* { m a x } } ]$ . In the inverse problem, the goal is to infer V and $D$ from sparse measurements of $\boldsymbol { u } ( \boldsymbol { x } , t )$

For $\begin{array} { r } { t \ll \frac { L } { V } } \end{array}$ , the solution of this equation can be approximated by the Ogata–Banks formula,

$$
u ( x , t ) = \frac { u _ { 0 } } { 2 } \left[ \mathrm { e r f c } \left( \frac { x - V t } { 2 \sqrt { D t } } \right) + \exp \left( \frac { V x } { D } \right) \mathrm { e r f c } \left( \frac { x + V t } { 2 \sqrt { D t } } \right) \right] .\tag{27}
$$

To construct the training dataset, we treat V and D as independent uniformly distributed random variables defined on the intervals $V \in [ V _ { \operatorname* { m i n } } , V _ { \operatorname* { m a x } } ]$ and $D \in [ D _ { \operatorname* { m i n } } , D _ { \operatorname* { m a x } } ]$ ], respectively. Next, we generate $N _ { \mathrm { t r a i n } }$ samples of $V$ and D from these distributions. For each sample of V and $D ,$ , the Ogata-Banks solution is evaluated on a uniform $N _ { x } \times N _ { t }$ mesh with nodes $( x _ { i } , t _ { k } )$ excluding $t = 0$ , where $x _ { 1 } = 0 , x _ { N _ { x } } = L , t _ { 1 } = T / ( N _ { t } - 1 )$ , and $t _ { N _ { t } } = T$ . The sample mean is then estimated as

$$
\bar { u } ( x _ { i } , t _ { k } ) = \frac { 1 } { N _ { \mathrm { t r a i n } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t r a i n } } } u ^ { ( n ) } ( x _ { i } , t _ { k } ) ,\tag{28}
$$

and the fluctuation field is defined by

$$
u ^ { \prime } ( x , t ) = u ( x , t ) - \bar { u } ( x , t ) .\tag{29}
$$

In all ADE experiments, we set $L = 8 6 , T = 2 0 0$ . The training dataset contains $N _ { \mathrm { t r a i n } } = 5 0 0$ realizations generated from

$$
V \in [ 0 . 6 V ^ { * } , 1 . 4 V ^ { * } ] , \qquad D \in [ 0 . 6 D ^ { * } , 1 . 4 D ^ { * } ] ,
$$

where $V ^ { * } = 0 . 2 1 3$ and $D ^ { * } = 0 . 0 6 1$ yielding the Peclet number $\begin{array} { r } { \mathrm { P e } = \frac { V ^ { \ast } L } { D ^ { \ast } } = 3 0 0 } \end{array}$ . The trained model is tested at

$$
V ^ { r e f } = 1 . 2 V ^ { * } , \qquad D ^ { r e f } = 0 . 7 D ^ { * } ,
$$

which corresponds to $\mathrm { P e } = 5 1 4$ . The test case is intentionally selected away from the center of the training ranges. Also, the ADE solutions for $\mathrm { P e } \gg 1$ are challenging to obtain with standard numerical methods. Therefore, the problem considered provides a rigorous test for lPINN.

The accuracy of lPINN and other methods for ADE is estimated using the dimensionless relative root-mean-square errors (rRMSEs) for the predicted solution u and, for the inverse problem, the estimated parameters $V$ and $D .$ . The relative errors are defined as

$$
\mathrm { r R M S E } _ { u } = \frac { \left\| \mathbf { u } - \mathbf { u } ^ { \mathrm { r e f } } \right\| _ { 2 } } { u _ { 0 } \sqrt { N _ { \mathrm { u } } } } , \qquad \mathrm { r R M S E } _ { V } = \frac { \left| V - V ^ { \mathrm { r e f } } \right| } { \left| V ^ { \mathrm { r e f } } \right| } , \qquad \mathrm { r R M S E } _ { D } = \frac { \left| D - D ^ { \mathrm { r e f } } \right| } { \left| D ^ { \mathrm { r e f } } \right| } ,
$$

where $N _ { u }$ is the number of grid points where the numerical solutions are compared to the reference solution ${ \pmb u } ^ { \mathrm { r e f } }$ (generally, $N _ { u } = N _ { g } )$ , where $N _ { g }$ denotes the number of grid points on the mesh used to evaluate the solution, and $u _ { 0 } = 1$ is the maximum value of the reference solution achieved at $x = 0$ . The dimensionless rRMSEs for the PDE residual R, initial condition, and boundary conditions are defined a

$$
\mathrm { r R M S E } _ { R } = \frac { L } { u _ { 0 } V } \frac { \| \mathcal { R } \| _ { 2 } } { \sqrt { N _ { \mathrm { r e s } } } } , \qquad \mathrm { r R M S E } _ { \mathrm { I C } } = \frac { \left\| \mathbf { u } _ { \mathrm { I C } } - \mathbf { u } _ { \mathrm { I C } } ^ { \mathrm { r e f } } \right\| _ { 2 } } { u _ { 0 } \sqrt { N _ { \mathrm { I C } } } } ,
$$

and

$$
\mathrm { r R M S E } _ { B C } = \frac { \left. \mathbf { u } ( x = 0 ) - \mathbf { u } ^ { \mathrm { r e f } } ( x = 0 ) \right. _ { 2 } } { u _ { 0 } \sqrt { N _ { \mathrm { B C , 0 } } } } + \frac { L } { u _ { 0 } } \frac { \left. \partial _ { x } \mathbf { u } ( x = L ) \right. _ { 2 } } { \sqrt { N _ { \mathrm { B C , } L } } } ,
$$

where $N _ { \mathrm { I C } }$ is the number of residual points at $t = 0$ (excluding the point $( x , t ) = ( 0 , 0 ) ) , N _ { \mathrm { B C , 0 } }$ is the number of residual points over the time domain on the $x = 0$ boundary, and $N _ { \mathrm { B C } , L }$ is the number of residual points on the $x = L$ boundary.

In lPINN, the mean and fluctuation networks are pretrained using the Adam optimizer. The online training for the forward ADE problem, i.e., solving the minimization problem (20), is a linear least-squares problem which we solve using the linear least-squares solver numpy.linalg.lstsq. The inverse ADE minimization problem is nonlinear and is solved using the L-BFGS method. The number of residual points is set to $N _ { \mathrm { r e s } } = 1 0 N _ { \eta } .$ . Inverse solutions are obtained with $N _ { \mathrm { m } } = \mathrm { 4 0 }$ measurements of u.

We compared the derivative-matching (Eqs. (13) and (14)) and residual-based (Eq. (15)) offline pretraining methods. For residual-based offline pretraining, we split the $N _ { x } \times \left( N _ { t } - 1 \right) = 8 7 0$ grid points into 10 batches at each epoch. For each batch, solution values of $N _ { \mathrm { t r a i n } } = 5 0 0$ realizations at 87 grid points are used for training, together with $N _ { \mathrm { r e s } } = 4 3 5$ residual points, $N _ { \mathrm { I C } } = 4 3 5$ initial condition collocation points, and $N _ { \mathrm { B C } } = 4 3 5$ boundary condition collocation points randomly resampled for each batch. For derivative-matching offline pretraining, the same 870 grid points are split into 10 batches, and for each batch, the solution and derivative values at the corresponding 87 grid points are used for training, together with $N _ { \mathrm { I C } } = 4 3 5$ and $N _ { \mathrm { B C } } = N _ { \mathrm { B C , 0 } } + N _ { \mathrm { B C , } L } = 4 3 5$ randomly resampled initial condition and boundary condition collocation points. The number of neural basis functions is set to $\bar { N } _ { \eta } = 5 \bar { 0 }$ . The performance of the models in terms of rRMSE of forward and inverse solutions is given in Table 1. The lPINN forward solution error is a combination of the approximation error and residual least-squares error. The former measures how well the pretrained DNNs approximate the PDE solution. The latter measures how accurately the residual least-squares formulation estimates $W _ { N + 1 }$ . To compute the approximation error, we estimate $W _ { N + 1 }$ by least-squares fitting the pretrained DNNs to the reference PDE solution. The residual-based pretraining gives a slightly smaller approximation error, $3 . 9 4 \times 1 0 ^ { - 4 }$ compared with $4 . 4 4 \times 1 0 ^ { - 4 }$ for the derivative-matching pretraining. Another related metric is the value of the PDE residual yielded by the pretrained DNN with $W _ { N + 1 }$ obtained from the least-squares fitting to the reference PDE solution. The RMSE of the residuals (relative to the zero-residual value produced by the reference solution) is smaller for residual-based pretraining $( 3 . 6 0 \times 1 0 ^ { - 2 } )$ than in the derivative-matching pretraining $( 4 . 3 6 \times 1 0 ^ { - 2 } )$ . However, the residual-based pretraining gives a much smaller lPINN forward solution error, $1 . 4 5 \times \bar { 1 0 } ^ { - 3 }$ versus $9 . 3 \overset { \prime } { 8 } \times 1 0 ^ { - 3 }$ in the derivative-matching pretraining. In the inverse problem, residual-based pretraining gives better accuracy for the solution u and velocity $V ,$ , while derivative-matching pretraining gives a smaller error for the diffusion coefficient D. Since we seek the best overall performance and, in particular, accurate forward prediction, we conclude that residual-based pretraining is the better strategy for the considered ADE problem, and use it in the remaining numerical examples of this section.

Table 1: ADE comparison between residual-based pretraining and derivative-matching pretraining. The smaller value in each row is shown in bold.
<table><tr><td>Case / Metric</td><td>Residual-based</td><td>Derivative-matching</td></tr><tr><td>Approximation error: rRMSE</td><td> $\overline { { { \bf 3 . 9 4 \times 1 0 ^ { - 4 } } } }$ </td><td> $\overline { { 4 . 4 4 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td>Residual: rRMSE</td><td> $\mathbf { 3 . 6 0 \times 1 0 ^ { - 2 } }$ </td><td> $4 . 3 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Forward:  $\mathrm { r R M S E } _ { u }$ </td><td> $\mathbf { 1 . 4 5 \times 1 0 ^ { - 3 } }$ </td><td> $9 . 3 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Inverse:  $\mathrm { r R M S E } _ { u }$ </td><td> $\mathbf { 3 . 9 0 \times 1 0 ^ { - 3 } }$ </td><td> $2 . 0 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Inverse:  $\mathrm { r R M S E } _ { V }$ </td><td> $\mathbf { 4 . 1 3 \times 1 0 ^ { - 3 } }$ </td><td> $1 . 2 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Inverse:  $\mathrm { r R M S E } _ { D }$ </td><td> $1 . 0 7 \times 1 0 ^ { - 1 }$ </td><td> $\bf { 6 . 2 3 \times 1 0 ^ { - 2 } }$ </td></tr></table>

Figure 2 plots the relative approximation error, the residual error, and the relative errors in the forward and inverse lPINN solutions as functions of $N _ { \eta } ,$ the number of neurons in the last hidden layers. The approximation error decreases as $N _ { \eta }$ increases. The residual error stays nearly constant for $N _ { \eta } < 5 0$ and then increases with $N _ { \eta }$ . It should be noted that in the described experiment, the number of residual points in the offline step is fixed, which can cause the increase in the residual error with increasing $N _ { \eta }$

The lPINN forward solution error increases with $N _ { \eta }$ when the regularization coefficient λ in Eq. (20) is set to zero. For the considered linear ADE, Eq. (20) yields a linear least-square problem that can be written as A · $\pmb { W } _ { N + 1 } = \pmb { b }$ , where A is a $N _ { \mathrm { r e s } } \times N _ { \eta }$ matrix and b is a $N _ { \mathrm { r e s } }$ vector. The unregularized least-squares solution is unique when A has full column rank, equivalently when $A ^ { T } A$ is nonsingular. Regularization, i.e., setting $\lambda > 0 ,$ , can provide uniqueness and improve numerical stability when the system is rank deficient or ill-conditioned. We select λ for each $N _ { \eta }$ through a grid search. We do not observe a decrease in the lPINN rRMSE for $N _ { \eta } > 3 0$ because the residual rRMSE increases with $N _ { \eta }$ . We show in Section 3.4 that the pretraining accuracy can be improved by increasing the size of the training dataset or, as stated earlier, using more residual points at the offline step.

For the inverse problem, the relative errors in the estimated u, V, and D are weakly dependent on $N _ { \eta } .$ . The overall best performance is achieved for $N _ { \eta } = 5 0$ , the value we use in the remaining numerical experiments of this section.

![](images/642379ebf34b87c11013ba3c45e1ac7cb41d2dd19a3e27b0938fc6b077cc1823.jpg)

![](images/550ce0adb78b6fb707ea5bc910dbd823b0c89d17fb18a6a3adb09ede4ac51c8e.jpg)

![](images/7d1493dac1c01a52b43e2fbfefd44fec5c0fb36dc9ba7a3c55c7f6476e83e8e9.jpg)  
Figure 2: ADE problem: Left: Approximation error of the pretrained fluctuation network and the corresponding residual error. Middle: Relative error of the lPINN forward prediction. Right: Relative errors of the lPINN inverse prediction for the solution u and the inferred parameters V and D. $N _ { \mathrm { t r a i n } } = 5 0 0$ . The lPINN error for $N _ { \eta } = 5 0$ $N _ { \mathrm { r e s } } = 5 0 0$ in this test is not directly comparable with the lPINN error for $N _ { \eta } = 5 0 , N _ { \mathrm { r e s } } = 5 0 0$ in the following forward problem as in that problem we use a shared $\lambda = 1 0 ^ { - 4 }$ across $N _ { \mathrm { r e s } }$

In the remainder of this section, we compare lPINN with vanilla PINN and PICKLE ROM.

## 3.1.1 Comparison with PINN

We first consider the forward problem. Figure 3 compares the PINN and lPINN predictions against the reference solution at $t = 2 0 . 7$ and 200.0 s using $N _ { \mathrm { r e s } } = 5 0 0$ residual points. The lPINN solution is visibly closer to the reference and achieves an rRMSE of 0.0015, whereas PINN yields an rRMSE of 0.0104. The computational cost is also significantly reduced: lPINN requires 0.2 s, while PINN requires 631.2 s. All PINN and lPINN computations are performed on the same workstation using an NVIDIA RTX A6000 GPU with 48 GB of memory.

![](images/579bc814a1b1e2bb5fc54e70397c44222585be578451bcc0a84347ca5d16e4c5.jpg)

![](images/0931911807ab5190ab61f67499d455a0e3449bcc5fa9434e544b0d5861a03198.jpg)  
Figure 3: ADE forward problem: (a) Comparison of the PINN and lPINN predictions with the reference solution at $t = 2 0 . 7 ~ \mathrm { s } ,$ , and 200.0 s. (b) Pointwise standard deviation at the $N _ { x } = 3 0$ grid points on the same mesh at $t = 2 0 . 7 \mathrm { s }$ and 200.0 s. $N _ { \mathrm { r e s } } = 5 0 0 , N _ { \eta } = 5 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

We next examine the sensitivity of lPINN and PINN errors to the number of residual points. Figure 4(a) shows that lPINN rRMSEs are consistently lower than PINN’s. Both lPINN and PINN errors decrease with increasing $N _ { \mathrm { r e s } } .$ and reach an asymptotic value at $\dot { N _ { \mathrm { r e s } } } = 5 0 0$

![](images/8800f3ba79c51cbfbae05da47094ac2fda7d9cadd4febe09ad68ce49e3399525.jpg)

![](images/8de60e552740b184824cc33e86169447e4b1a7ce67d5166990b499e4a0647360.jpg)  
Figure 4: ADE forward problem: (a) rRMSE(u) as a function of the number of residual points $N _ { \mathrm { r e s } } .$ where u is the mean of 10 PINN or lPINN solutions. (b) Relative root-mean-square error (rRMSE) of residual, initial condition, and boundary condition error as a function of the number of residual points $N _ { \mathrm { r e s } } . \ N _ { \eta } = 5 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

Next, we compare PINN and lPINN inverse solutions for the parameters V and D and the state u obtained with $N _ { \mathrm { m } } = 4 0$ measurements of u. Figure 5 shows estimated u and the reference u at times $t = 2 0 . 7$ and 200.0 s. The lPINN prediction achieves an rRMSE of 0.0039, compared with 0.0509 for PINN. The V and D parameters found with lPINN have rRMSE values of 0.0041 and 0.1074, respectively. PINN yields V with an rRMSE of 0.0290 and D with an rRMSE of 0.8513. The inference time is reduced from 350.6 s for PINN to 19.2 s for lPINN.

![](images/0b48f722c400c5fa258845293d97dbb98b1408c6e7f60f2d508e1dc6d6c718d5.jpg)

![](images/a7fa0c36f0af167d966bbafdad7255d676f7d16232151daf96d8ac7205e8a151.jpg)  
Figure 5: ADE inverse problem: Comparison of the PINN and lPINN estimations of $u ( x , t )$ with the reference solution at $t = 2 0 . 7 \mathrm { s } .$ , and 200.0 s. $N _ { \mathrm { r e s } } = 5 0 \bar { 0 } , N _ { \mathrm { m } } = 4 0 , N _ { \eta } = 5 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

Figure 6 shows the rRMSEs in the lPINN and PINN inverse solutions as functions of $N _ { \mathrm { m } }$ . The errors are reported for the estimated $V , D ,$ , and u. For all three variables, lPINN generally outperforms PINN, with the largest advantage occurring for smaller $N _ { \mathrm { m } } .$ . This behavior matters in practical inverse problems, where observations are often limited and expensive to obtain.

![](images/0fd90e93989dc4785ef80284c4faa612eaaaf5869ec3af5eef5bb32b30428594.jpg)  
Figure 6: ADE inverse problem: Relative root-mean-square error (rRMSE) of the inferred velocity V, dispersion coefficient D, and predicted solution u as functions of the number of measurement points $N _ { \mathrm { m } \cdot } N _ { \mathrm { r e s } } \dot { = } 5 0 0 , N _ { \eta } = 5 0$ $N _ { \mathrm { t r a i n } } = 5 0 0$

## 3.1.2 Comparison with the dPICKLE ROM method.

Here, we compare the forward lPINN and dPICKLE solutions. The dPICKLE ROM method [17] is based on space– time-dependent Karhunen–Loève expansions (KLEs) of the state variable u:

$$
\pmb { u } ( \pmb { a } ) = \bar { \pmb { u } } + \pmb { \Phi } \pmb { a } ,\tag{30}
$$

where u is the solution vector, u¯ is the training dataset mean, $\Phi$ is the matrix of eigenvectors scaled with the square root of the corresponding eigenvalues, and a is the vector of coefficients. These coefficients can be estimated using

projection, or, as in PINN and lPINN methods, by minimizing the L2 norm of the PDE residuals:

$$
\pmb { a } ^ { * } = \arg \operatorname* { m i n } _ { \pmb { a } } \vert \vert \pmb { R } ( \pmb { a } ) \vert \vert _ { 2 } ^ { 2 }\tag{31}
$$

where $\mathbf { \delta } _ { R ( a ) }$ is the vector of residuals. In dPICKLE, as in most ROM methods, all quantities, including u¯, Φ, R, and, ultimately, the solution u are computed on the space-time mesh as the one used in the training dataset generation. In [17], the spatial and temporal derivatives entering R are evaluated using a finite difference discretization.

For a fair comparison, we use the same number of samples to compute u¯ and Φ as in the lPINN solution. Also, the dimensionality of a is set the same as that of $W _ { N + 1 }$ . The residual points in the lPINN solution are chosen to coincide with the grid points where dPICKLE residuals are evaluated.

The forward solutions are compared for the parameter values

$$
V = 1 . 2 V ^ { * } , \qquad D = 0 . 7 D ^ { * } .
$$

Figure 7 compares the lPINN and dPICKLE predictions with the reference solution at t = 20.7 and 200.0 s. The lPINN prediction is in closer agreement with the reference solution and achieves an rRMSE of 0.0013, whereas dPICKLE yields an rRMSE of 0.0470. However, dPICKLE has a lower computational cost: lPINN requires only 0.26 s versus 0.01 s for dPICKLE. The lPINN runtime is longer because it takes more time to compute derivatives via automatic differentiation than numerically in dPICKLE.

IPINN & dPICKLE vs Reference  
![](images/6b7fc924a9d3314c7050497adb503bc5e0baf100f3c9c058e8c87eea39ada989.jpg)  
Figure 7: ADE forward problem: Comparison of the lPINN and dPICKLE predictions with the reference solution at $t = 2 0 . 7 \ s$ s and 200.0 s. Here, $N _ { \mathrm { r e s } } = 7 \bar { 8 } 4 , N _ { \eta } = 5 0$ , and $N _ { \mathrm { t r a i n } } = 5 0 0$

## 3.2 Burgers’ equation

In this section, we consider the one-dimensional Burgers’ equation,

$$
\frac { \partial u } { \partial t } + u \frac { \partial u } { \partial x } = \nu \frac { \partial ^ { 2 } u } { \partial x ^ { 2 } } ,\tag{32}
$$

subject to the initial condition

$$
u ( x , 0 ) = - \sin ( \pi x ) ,\tag{33}
$$

and the boundary conditions

$$
u ( - 1 , t ) = u ( 1 , t ) = 0 .\tag{34}
$$

The forward lPINN model is pretrained to solve the Burgers’ equation on the time domain (0, 1] for $\textstyle \nu \in \left[ { \frac { 0 . 1 } { \pi } } , { \frac { 0 . 5 } { \pi } } \right]$ . We construct the training dataset using the Cole-Hopf analytical solution of the Burgers’ equation,

$$
u ( x , t ) = { \frac { \int _ { - \infty } ^ { \infty } - \sin ( \pi ( x - \xi ) ) \exp \left( - { \frac { \cos ( \pi ( x - \xi ) ) } { 2 \pi \nu } } \right) \exp \left( - { \frac { \xi ^ { 2 } } { 4 \nu t } } \right) d \xi } { \int _ { - \infty } ^ { \infty } \exp \left( - { \frac { \cos ( \pi ( x - \xi ) ) } { 2 \pi \nu } } \right) \exp \left( - { \frac { \xi ^ { 2 } } { 4 \nu t } } \right) d \xi } } .\tag{35}
$$

We uniformly draw $N _ { \mathrm { t r a i n } } = 5 0 0$ samples of ν from the specified interval. For each ν sample, the Cole-Hopf solution is evaluated on an $N _ { x } \times N _ { t }$ mesh with $N _ { x } = 3 0$ and $N _ { t } = 3 0$ . The temporal grid is uniform, and the spatial grid is center-clustered over [−1, 1]. The center-clustered spatial grid points are defined as,

$$
s _ { i } = - 1 + { \frac { 2 i } { N _ { x } - 1 } } , \qquad x _ { i } = \mathrm { s i g n } ( s _ { i } ) \left[ 1 - { \frac { \operatorname { t a n h } ( \sigma ( 1 - | s _ { i } | ) ) } { \operatorname { t a n h } ( \sigma ) } } \right] , \qquad i = 0 , \dots , N _ { x } - 1 .
$$

where clustering strength is set to $\sigma = 2 .$

The pretrained forward lPINN model is then tested for

$$
\nu ^ { \mathrm { r e f } } = \frac { 0 . 1 } { \pi } ,
$$

using the same mesh. The inverse problem is solved for ν and u given sparse measurements of u.

The rRMSEs in the predicted solution u evaluated on the $N _ { x } \times N _ { t }$ mesh and, for the inverse problem, estimated viscosity ν are defined as

$$
\mathrm { r R M S E } _ { \mathbf { u } } = \frac { \left\| \mathbf { u } - \mathbf { u } ^ { \mathrm { r e f } } \right\| _ { 2 } } { u _ { 0 } \sqrt { N _ { u } } } , \qquad \mathrm { r R M S E } _ { \nu } = \frac { \left. \nu - \nu ^ { \mathrm { r e f } } \right. } { \left. \nu ^ { \mathrm { r e f } } \right. } .
$$

where $N _ { u } = N _ { x } \times N _ { t } , { \boldsymbol { u } } ^ { \mathrm { r e f } }$ is the reference solution evaluated on the $N _ { x } \times N _ { t }$ mesh, and $\begin{array} { r } { u _ { 0 } = u ( x = - \frac { 1 } { 2 } , t = 0 ) } \end{array}$ is the maximum of the initial condition. The dimensionless rRMSEs for the PDE residual, initial condition, and boundary condition are defined as

$$
\mathrm { r R M S E } _ { R } = \frac { L } { u _ { 0 } ^ { 2 } } \frac { \| \mathbf { \mathcal { R } } \| _ { 2 } } { \sqrt { N _ { \mathrm { r e s } } } } , \qquad \mathrm { r R M S E } _ { \mathrm { I C } } = \frac { \left\| \mathbf { u } _ { \mathrm { I C } } - \mathbf { u } _ { \mathrm { I C } } ^ { \mathrm { r e f } } \right\| _ { 2 } } { u _ { 0 } \sqrt { N _ { \mathrm { I C } } } } ,
$$

and

$$
\mathrm { r R M S E } _ { \mathrm { B C } } = \frac { \Vert \mathbf { u } ( x = - 1 ) - \mathbf { u } ^ { \mathrm { r e f } } ( x = - 1 ) \Vert _ { 2 } } { u _ { 0 } \sqrt { N _ { \mathrm { B C , - 1 } } } } + \frac { \Vert \mathbf { u } ( x = 1 ) - \mathbf { u } ^ { \mathrm { r e f } } ( x = 1 ) \Vert _ { 2 } } { u _ { 0 } \sqrt { N _ { \mathrm { B C , 1 } } } } .
$$

Figure 8 plots the relative approximation error, the residual of the approximated solution, and the relative errors in the forward and inverse lPINN solutions as functions of $N _ { \eta }$ . The number of residual points is set to $N _ { \mathrm { r e s } } = 1 0 N _ { \eta }$ The inverse solution is obtained using $N _ { \mathrm { m } } = 5 0$ measurements. The DNNs are pretrained using residual-based loss functions. This choice is justified by the comparison study summarized in Table 2. The approximation errors of the pretrained DNNs and the residual RMSE are weakly dependent on the considered $N _ { \eta }$ values. The smallest rRMSE of the forward lPINN solution is obtained for $N _ { \eta } = 2 \mathrm { { 0 } }$ . In the inverse problem, the errors in both the estimated u and ν are smallest at $N _ { \eta } = 5 0$ . Accordingly, in the remaining examples of this section, we set $N _ { \eta } = 2 0$ for forward problems and $N _ { \eta } = 5 0$ for the inverse problems.

![](images/332497b9746a2cde3b7c4d9011991f71d097d2c6436cd8d39b69c95de4a027ba.jpg)

![](images/9810fee0f41ffe3afb254049e06e1d8d7326066c6410e7f778c957c09c527d4e.jpg)

![](images/3eb6f11cc6951845e28948b6e07ac85572370f7106f7159d08a44f22c546ac4f.jpg)  
Figure 8: Burgers’ equation: Left: Approximation error of the pretrained fluctuation network and the corresponding residual error. Middle: Relative error of the lPINN forward prediction. Right: Relative errors of the lPINN inverse prediction for the solution u and the inferred parameter ν. Here, $N _ { \mathrm { t r a i n } } = 5 \bar { 0 } 0$

Next, we compare derivative-matching pretraining with residual-based pretraining. For residual-based offline pretraining, we split the $\dot { N } _ { x } \times N _ { t } = 9 0 0$ solution-grid points into 10 batches at each epoch. For each batch, the solution values of all $\bar { N } _ { \mathrm { t r a i n } } = 5 0 0$ realizations at 90 grid points are used for training, together with $N _ { \mathrm { r e s } } = 9 0$ residual points, $N _ { \mathrm { I C } } = 9 0$ initial condition collocation points, and $N _ { \mathrm { B C } } = 9 0$ boundary condition collocation points are randomly resampled for each batch. For derivative-matching offline pretraining, the same 900 solution-grid points are split into 10 batches, and for each batch, the solution and derivative values at the corresponding 90 grid points are used for training, together with $N _ { \mathrm { I C } } = 9 0$ and $N _ { \mathrm { B C } } = 9 0$ randomly resampled initial condition and boundary condition collocation points. The number of residual points is set to $N _ { \mathrm { r e s } } = 1 0 0$ in both lPINN and PINN for the forward problem, and we use $N _ { \mathrm { r e s } } = 1 0 0$ residual points and $N _ { \mathrm { m } } = 5$ measurements of u for the inverse problem. As shown in Table 2, residual-based pretraining gives smaller approximation errors for both $N _ { \eta } = 2 0$ and $\bar { N _ { \eta } } \stackrel { - } { = } 5 0$ . Residual-based pretraining gives smaller residual RMSE and better accuracy in both the forward and inverse problems. Specifically, it reduces the forward error from $3 . 1 5 \times 1 0 ^ { - 3 } \mathrm { t o } 8 . 0 7 \times 1 0 ^ { - 4 }$ and also gives lower inverse errors for both u and ν. In the remaining numerical experiments of this section, we use the residual-based pretraining.

Table 2: Comparison of residual-based and derivative-matching pretraining for Burgers’ equation. As we use $N _ { \mathrm { m } } = 5$ in this test and we use $N _ { \mathrm { m } } = 5 0$ for solving the inverse problem in the following part of the section, the result is not directly comparable.
<table><tr><td>Metric</td><td> $\overline { { N _ { \eta } } }$ </td><td>Residual-based</td><td>Derivative-matching</td></tr><tr><td rowspan="2">Approximation rRMSE</td><td>20</td><td> $\mathbf { \overline { { 5 . 3 5 \times 1 0 ^ { - 4 } } } }$ </td><td> $\overline { { 2 . 1 1 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>50</td><td> $\mathbf { 8 . 1 3 \times 1 0 ^ { - 4 } }$ </td><td> $8 . 2 4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="2">Residual rRMSE</td><td>20</td><td> $\bf { 3 . 1 0 \times 1 0 ^ { - 2 } }$ </td><td> $\overline { { 4 . 9 0 \times 1 0 ^ { - 2 } } }$ </td></tr><tr><td>50</td><td> $\mathbf { 4 . 5 6 \times 1 0 ^ { - 2 } }$ </td><td> $6 . 3 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Forward rRMSEu</td><td>20</td><td> $\overline { { { \bf 8 . 0 7 \times 1 0 ^ { - 4 } } } }$ </td><td> $\overline { { 3 . 1 5 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>Inverse  $\mathrm { r R M S E } _ { u }$ </td><td>50</td><td> $\mathbf { 1 . 5 7 \times 1 0 ^ { - 3 } }$ </td><td> $3 . 2 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Inverse  $\mathrm { r R M S E } _ { \nu }$ </td><td>50</td><td> $\mathbf { 3 . 1 8 \times 1 0 ^ { - 3 } }$ </td><td> $1 . 1 4 \times 1 0 ^ { - 2 }$ </td></tr></table>

Next, we compare the PINN and lPINN solutions of the forward Burgers’ problem. Figure 9 shows the PINN, lPINN, and reference solutions at $t = 0 . 0$ and 1.0 s. The lPINN solution agrees much more closely with the reference and achieves an rRMSE of 0.0008, whereas the PINN yields an rRMSE of 0.0439. The computational advantage is also substantial: lPINN requires only 17.3 s, while PINN requires 308.3 s.

![](images/cf1de6d92bfe7a687f9dda738b9c35b6493928228b0271dbc35a3b031e658ac0.jpg)

(b) Pointwise standard deviation σ  
![](images/84d62f1f9e8d89c296d5688ca45b4ec16e6817664aa5eaccc01eba70d9cc8326.jpg)  
Figure 9: Burgers’ forward problem: Comparison of the PINN and lPINN predictions with the reference solution at $t = 0 . 0 ~ \mathrm { s }$ and 1.0 s. Here, $\dot { N _ { \mathrm { r e s } } } = 1 0 0 , N _ { \eta } \dot { = } 2 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

Figure 10(a) examines the dependence of lPINN and PINN errors in the forward Burgers’ problem on $N _ { \mathrm { r e s } }$ for $N _ { \eta } = 2 0$ The rRMSE of PINN decreases to an asymptotic value at $N _ { \mathrm { r e s } } \approx 8 0 0$ , while the rRMSE of lPINN stays nearly constant across $N _ { \mathrm { r e s } }$ . The lPINN asymptotic error is close to the approximation error for $N _ { \eta } = 2 0$ , which is the lower limit of lPINN error. The PINN asymptotic error is smaller than that of lPINN. This is different from the ADE problem where lPINN remained more accurate than PINN for all considered $N _ { \mathrm { r e s } } .$ Figure 10(b) explains why the lPINN error does not decrease with increasing $N _ { \mathrm { r e s } } { \mathrm { : } }$ while the PDE residual error decreases, the initial and boundary condition residual errors stay unchanged with increasing $N _ { \mathrm { r e s } }$

![](images/af96b3026ec66ce90c9696340d81d77ddc172ef06f045b94c4909a8e92d26e53.jpg)

![](images/c50348188722852461e78d068ca4d70b34e2fd8fb83c428205f53c14bdd3e187.jpg)  
Figure 10: Burgers’ forward problem: (a) Relative root-mean-square error (rRMSE) of the predicted solution for PINN and lPINN as a function of the number of residual points $N _ { \mathrm { r e s } } .$ (b) Relative root-mean-square error (rRMSE) of the residual, initial condition, and boundary condition errors as functions of the number of residual points $N _ { \mathrm { r e s } } . \ N _ { \eta } = 2 0 .$ $N _ { \mathrm { t r a i n } } = 5 0 0$

Figure 11 compares the PINN and lPINN u inverse solutions at $t = 0 . 0 $ , and 1.0 s using $N _ { \mathrm { r e s } } = 1 0 0$ residual points and $N _ { \mathrm { m } } ^ { \mathrm { ^ { - } } } = 5$ measurements of u. The lPINN solution achieves an rRMSE of 0.0016 versus 0.0061 in PINN. The estimated ν in the lPINN solution has a relative error of 0.0032 versus 0.0119 in PINN. The time to obtain the lPINN solution (excluding pretraining) is 3.2 s versus 251.5 s in the PINN method.

![](images/fdd63dc6a5965533459e7e75d91f7675f718044e3d905b381d7d32d440f9aee9.jpg)

![](images/05438d5c7d931941737d1cdd340f03243509474e5271b6a0627e31130b4a4500.jpg)  
Figure 11: Burgers’ inverse problem: Comparison of the PINN and lPINN predictions with the reference solution at $t = 0 . 0$ and 1.0 s. $N _ { \mathrm { r e s } } = 1 \hat { 0 0 } , N _ { \mathrm { m } } = 5 , \tilde { N _ { \eta } } = 5 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

We further investigate the effect of the number of measurements on inverse solution accuracy for $N _ { \eta } = 5 0$ and $N _ { \mathrm { r e s } } = 1 0 0$ . As shown in Figure 12, the PINN errors in the estimated ν and u strongly depend on $N _ { \mathrm { m } }$ . In contrast, lPINN errors are less sensitive to $N _ { \mathrm { m } }$ . The lPINN method achieves similar accuracy with relatively few measurements. The error in the estimated ν is smaller than that of PINN for all considered $N _ { \mathrm { m } } .$ . The u estimate is more accurate in lPINN for $N _ { \mathrm { m } } < 2 0$ and less accurate otherwise. This suggests that lPINN is more robust than PINN in the limited-data regime.

![](images/9b13d4b5f3cdba86501554ee209fade1363ccec96f0b4359ca3c4ae80a03a126.jpg)  
Figure 12: Burgers’ inverse problem: Relative root-mean-square error (rRMSE) of the inferred kinematic viscosity ν and predicted solution u as functions of the number of measurement points $N _ { \mathrm { m } } . \ N _ { \mathrm { r e s } } = 1 0 0 , N _ { \eta } = 5 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

## 3.3 Nonlinear Pendulum Equation

We next assess the proposed lPINN framework on the nonlinear pendulum equation

$$
\frac { d ^ { 2 } \theta } { d t ^ { 2 } } + \gamma \frac { d \theta } { d t } + \frac { g _ { 0 } } { \ell } \sin ( \theta ) = 0 ,\tag{36}
$$

subject to the initial conditions

$$
\theta ( 0 ) = \frac { \pi } { 2 } , \qquad \frac { d \theta } { d t } ( 0 ) = 0 ,\tag{37}
$$

where $\gamma$ is the damping coefficient, ℓ is the pendulum length, and $\theta$ is the angle, $g _ { 0 }$ is the gravity. In the forward problem, we compute $\theta$ for $\gamma \in [ 0 . 0 5 , 0 . 5 ] , \ell \in [ 0 . 5 , 2 . 0 ]$ . In the inverse problem, $\gamma , \ell ,$ and $\bar { \theta ( t ) }$ are estimated from θ measurements at a few time instances.

To construct the training dataset, we draw $N _ { \mathrm { t r a i n } } = 5 0 0$ samples of $( \gamma , \ell )$ uniformly from their considered ranges. Eq. (36) is solved numerically for each parameter combination with the Runge–Kutta method on a uniform temporal grid using a constant time step $\Delta t = T / ( N _ { t } - 1 )$ , where $N _ { t } = 3 0 0$ and $T = 3 0$ . In addition to $\theta , { \frac { d \theta } { d t } }$ and $\textstyle { \frac { d ^ { 2 } \theta } { d t ^ { 2 } } }$ are also obtained from the solver and used for the DNN pretraining. The pretrained model is then tested on the same temporal mesh for $\gamma ^ { \mathrm { r e f } } = 0 . 1$ and $\ell ^ { \mathrm { r e f } } = 0 . 8$

Eq (36) is challenging to solve using PINN because of θ oscillation with t as result of the spectral bias, the tendency of DNNs to learn low-frequency components more readily than high-frequency components [29]. In lPINN, we find that a standard fully connected network approximates $\theta ( t )$ more accurately at early times than at later times. We also find that the training losses of the mean and fluctuation components converge at substantially different rates, which further complicates joint training with standard DNNs. To improve DNN pretraining, we separately pretrain the mean and fluctuation networks using the numerical values of $\bar { \theta } , d \bar { \theta } / d t , d ^ { 2 } \bar { \theta } / d t ^ { 2 }$ , and $\theta ^ { \prime } , \bar { d } \bar { \theta } ^ { \prime } / d t , \dot { d } ^ { 2 } \theta ^ { \prime } / d t ^ { \dot { 2 } }$ , respectively. Also, we use a Fourier Feature Neural Network (FFNN) rather than a standard fully connected network to model fluctuations $\theta ^ { \prime } ( t )$ . In both PINN and lPINN, we use FFNNs with a mapping size of 64 and the hidden-layer width $N _ { \eta } .$ . We use different frequency scales (0.5 for PINN and 0.1 for lPINN) because preliminary tuning showed that the two methods achieve their best performance at different Fourier-feature scales. For lPINN, the smaller scale provides sufficient resolution of the oscillatory solution without introducing spurious high-frequency artifacts.

The rRMSEs of the predicted θ and, for the inverse problem, estimated $\gamma$ and $\ell$ are defined as

$$
\mathrm { r R M S E } _ { \theta } = \frac { \left\| \theta - \theta ^ { \mathrm { r e f } } \right\| _ { 2 } } { \theta _ { 0 } \sqrt { N _ { t } } } , \qquad \mathrm { r R M S E } _ { \gamma } = \frac { \left| \gamma - \gamma ^ { \mathrm { r e f } } \right| } { \left| \gamma ^ { \mathrm { r e f } } \right| } , \qquad \mathrm { r R M S E } _ { \ell } = \frac { \left| \ell - \ell ^ { \mathrm { r e f } } \right| } { \left| \ell ^ { \mathrm { r e f } } \right| } ,
$$

where $\pmb { \theta } ^ { \mathrm { r e f } }$ is the reference solution and $\theta _ { 0 } = \pi / 2$ is the initial condition. The dimensionless rRMSEs for the PDE residual and the initial conditions are defined as

$$
\mathrm { r R M S E } _ { R } = \frac { \ell } { \theta _ { 0 } g _ { 0 } } \frac { \| \mathcal { R } \| _ { 2 } } { \sqrt { N _ { \mathrm { r e s } } } } , \qquad \mathrm { r R M S E } _ { \mathrm { I C , 0 } } = \frac { | \theta ( 0 ) - \theta _ { 0 } | } { \theta _ { 0 } } ,
$$

and

$$
\mathrm { r R M S E } _ { \mathrm { I C , 1 } } = \frac { \left| \frac { d \theta ( 0 ) } { d t } \right| } { \theta _ { 0 } \sqrt { g _ { 0 } / \ell } } .
$$

For the nonlinear pendulum Eq (36), residual-based pretraining did not produce an accurate pretrained representation and resulted in large errors during the subsequent online step. In contrast, derivative-matching pretraining accurately reproduced the mean and fluctuation fields, along with their first and second time derivatives. Therefore, this pretraining approach is better suited for learning the basis functions for the nonlinear pendulum equation, and we use it in all examples in this section. For derivative-matching offline pretraining, the solution and its first- and second-order time derivatives of all $N _ { \mathrm { t r a i n } } = 5 0 0$ realizations at $N _ { t } = 3 0 0$ temporal grid points are used for training.

Figure 13 plots the relative approximation error, the residual of the approximated solution, and the relative errors in the forward and inverse lPINN solutions as functions of $N _ { \eta } .$ . The number of residual points is set to $N _ { \mathrm { r e s } } = 1 0 N _ { \eta }$ while the training set size is kept fixed at $N _ { \mathrm { t r a i n } } = 5 0 0$ . For the inverse problem, the number of θ measurements is set to $N _ { \mathrm { m } } = 5 0$ . The approximation error decreases significantly as $N _ { \eta }$ increases, reaching nearly machine precision at $N _ { \eta } = 4 0 0$ . In contrast, the residual error remains at approximately the same order of magnitude for all tested widths, indicating that increasing the last layer beyond $N _ { \eta } = 2 0 0$ does not improve the overall solution representation for the fixed $N _ { \mathrm { t r a i n } } .$ . The forward lPINN solutions reach the smallest error at $\bar { N } _ { \eta } = 2 0 0$ . For the inverse problem, the relative errors in the estimated θ and the estimated parameter γ decrease with $N _ { \eta } ,$ while the estimated parameter ℓ is practically independent of it. Therefore, considering both accuracy and computational cost, we set $\bar { N _ { \eta } } = 2 0 0$ in the following numerical experiments.

![](images/3eac5cc6ebc10aebfa0d488432a29d7ff47608dd31d51accaa6cc3064ae67159.jpg)

![](images/8100ae9896031fd42b85acf48b55d167c6d8ee387c5f5fe746417227b6c87f35.jpg)

![](images/3309d327adff42336ef49aeba9dab64dd86f2af7b0c099225009fccc6479b9ff.jpg)  
Figure 13: Nonlinear pendulum equation. Left: Approximation error of the pretrained fluctuation network and the corresponding residual error. Middle: Relative error of the lPINN forward prediction. Right: Relative errors of the lPINN inverse prediction for the solution $\theta$ and the inferred parameter $\gamma$ and $\bar { \ell } .$ Here, $N _ { \mathrm { t r a i n } } { \stackrel { \textstyle - } { = } } 5 0 0 , N _ { \mathrm { r e s } } = 1 0 N _ { \eta } ,$ and $N _ { \mathrm { m } } = 5 0$ for the inverse problem.

Figure 14 compares the PINN and lPINN forward solutions with the reference solution. The lPINN achieves an rRMSE of 0.0022, compared with 0.0994 for PINN, corresponding to an error reduction of approximately 98%. The computational cost drops from 2375.3 s for PINN to 287.1 s for lPINN, an approximately 8.3-fold speedup.

![](images/76477e5b9de60156a01917514b6bbf9a84631f67f9cd7604afe5c8b81a5e276c.jpg)

![](images/f7264c26d0660374c70bd296f6ca49180603e6a281cef2e3092b6f11c22e3b06.jpg)  
Figure 14: Nonlinear pendulum equation forward problem: Comparison of the PINN and lPINN predictions with the reference solution. $\bar { N _ { \mathrm { r e s } } } = 1 0 0 0 , \bar { N } _ { \eta } = 2 0 0 , \bar { N _ { \mathrm { t r a i n } } } = 5 0 0$

We next examine how the number of residual points affects forward solution accuracy. Figure 15(a) shows that lPINN errors are smaller than PINN errors for all considered $N _ { \mathrm { r e s } }$ . The PINN error initially decreases with $N _ { \mathrm { r e s } }$ and reaches an asymptotic value at $N _ { \mathrm { r e s } } = 2 0 0 0$ . The lPINN error also initially decreases with $\dot { N } _ { \mathrm { r e s } }$ and reaches an asymptotic value at $\dot { N _ { \mathrm { r e s } } } = 1 0 0 0$ . At $N _ { \mathrm { r e s } } = 1 0 0 0$ , the lPINN error is more than an order of magnitude smaller than the PINN error.

![](images/81a3781bfebca14fa26a8d1ab50670d472b2d37d8fd888392f3f74fb4136de5b.jpg)

![](images/4763d3e11689ace5ad902109a580ab4a60c329d2820fe764842b34c5a2d17f72.jpg)  
Figure 15: Nonlinear pendulum forward problem: (a) rRMSE of the predicted solution for PINN and lPINN as a function of the number of residual points $\bar { N } _ { \mathrm { r e s } } . \left( \mathrm { b } \right) \mathrm { r R M S E }$ of residual and initial condition errors as a function of the number of residual points $N _ { \mathrm { r e s } } . \ N _ { \eta } ^ { \bullet } = 2 0 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

Figure 16 compares the inverse PINN and lPINN solutions for θ with the reference solution. The lPINN solution is more accurate, with an rRMSE of 0.0014 versus 0.3767 in the PINN solution. The lPINN parameter estimates are also more accurate, yielding rRMSEs of 0.0133 for γ and 0.0012 for ℓ versus 1.0917 and 0.2648, respectively, in the PINN method. The online part of the lPINN solution takes 209.0 s versus 2271.0 s for the PINN solution.

![](images/02b3a1abc1c88b8554baa20925bce0f82b3cfbba5103e0a5910d08f803bd976b.jpg)

![](images/f63f965ef972d0b3ae2c782563517f2cf5e9b1121683bccc96dc4a46899d0699.jpg)  
Figure 16: Nonlinear pendulum equation inverse problem: Comparison of the PINN and lPINN predictions with the reference solution. $\bar { N _ { \mathrm { r e s } } } = 5 0 0 , \bar { N _ { \mathrm { m } } } = 1 0 , N _ { \eta } = \bar { 2 } 0 0 , N _ { \mathrm { t r a i n } } = \bar { 5 } 0 0$

Figure 17 depicts the effect of the number of measurements on the accuracy of the PINN and lPINN inverse solutions. Both PINN and lPINN errors generally decrease with increasing $N _ { \mathrm { m } }$ . The lPINN gives similar errors to PINN for θ and $\ell ,$ and estimates γ more accurately. The lPINN errors are smaller than PINN’s for $N _ { \mathrm { m } } = 1 0$ for all three unknowns. For $N _ { \mathrm { m } } \geq 2 0$ , lPINN gives similar errors as PINN for ℓ and γ, while PINN estimates $\theta$ more accurately.

![](images/51d6eeac0fdec22eea2d953f905cf0af3f856a7c44942855b9283a9dd804c62b.jpg)  
Figure 17: Nonlinear pendulum equation inverse problem: Relative root-mean-square error (rRMSE) of the inferred damping coefficient $\gamma ,$ pendulum length ℓ, and predicted solution θ as functions of the number of measurement points $\bar { N _ { \mathrm { m } } } \cdot \bar { N _ { \mathrm { r e s } } } = 5 0 0 , \bar { N _ { \eta } } \overset { \cdot } { = } 2 0 0 , \bar { N _ { \mathrm { t r a i n } } } \overset { \cdot } { = } 5 0 0$

## 3.4 Superresolution of lPINN

In this section, we study the interdependence between numerical errors in the training dataset and lPINN accuracy as a function of the numerical resolution of the model generating the training dataset for offline pretraining of the lPINN DNNs. The goal is to demonstrate that an lPINN pretrained on a dataset computed on a coarser mesh can produce a numerical solution that is more accurate than the test numerical solution computed on the same coarse mesh and, more importantly, is more accurate than the test numerical solution computed on a finer mesh. We refer to the latter as superresolution. We demonstrate the superresolution properties of lPINN for the ADE, nonlinear Burgers’ and pendulum equations.

## 3.4.1 ADE equation

We first study how lPINN accuracy depends on the accuracy of the numerical simulations that compose the training dataset. The numerical solutions are obtained using the second-order central differences in space and the adaptive fifth-order Runge–Kutta (RK45) method in time. We generate three training datasets, $\{ D ^ { i } \} _ { i = 1 } ^ { 3 }$ , each consisting of $N _ { \mathrm { t r a i n } } = 5 0 0$ numerical simulations, performed on the uniform mesh $( N _ { x } ^ { i } , N _ { t } ^ { \overline { { i } } } )$ , where $N _ { x } ^ { i } = \bar { N } _ { t } ^ { i ^ { * } } \in ( 3 0 , 5 9 , 2 4 0 )$ . We pretrained five lPINN models on each corresponding dataset. In the online stage, we use the lPINN models to obtain test solutions with $N _ { \mathrm { r e s } } = 5 0 0$ residual points. We compute the rRMSE errors with respect to the analytical solution on the mesh used to generate the corresponding dataset. For comparison, we also compute the rRMSE of the numerica test solution obtained on the corresponding mesh.

Figure 18 shows that lPINN and numerical solution errors decrease with decreasing $\Delta x = L / ( N _ { x } ^ { i } - 1 )$ . Two important observations are that lPINN errors are smaller than the numerical solution error for the same $\Delta x$ . More importantly, the lPINN error obtained on the grid $\frac { L } { 2 9 }$ is smaller than the error of the numerical solution obtained on the finer mesh $\frac { L } { 5 8 }$ Similarly, the lPINN error corresponding to grid size $\frac { L } { 5 8 }$ is smaller than the numerical solution error corresponding to the finer mesh $\frac { L } { 2 3 9 }$ . We also see the limitation of the lPINN superresolution in the considered setting (the training data set size, the number of basis functions, and the number of residual points): the lPINN error pretrained on the grid $\frac { L } { 2 9 }$ is larger than the error of the numerical solution obtained on the grid $\frac { L } { 2 3 9 }$

![](images/3af81722fb84138944db9d4655f456d50535aa77b37efe392d0d8bce66e0e128.jpg)  
Figure 18: lPINN prediction error vs. mesh spacing $\Delta x$ for ADE equation. For each mesh, the lPINN is pretrained using the numerical-solution dataset generated on that mesh and is then evaluated on the same mesh. Smaller $\Delta x$ corresponds to a finer mesh. $N _ { \mathrm { r e s } } = 5 0 0 , N _ { \eta } = 5 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

## 3.4.2 Burgers’ equation

We first study how lPINN accuracy depends on the accuracy of the numerical simulations that compose the training dataset. We obtain the numerical solutions using second-order central differences in space and an adaptive fifth-order Runge–Kutta (RK45) method in time. We generate three training datasets, $\{ D ^ { i } \} _ { i = 1 } ^ { 3 }$ , each consisting of $N _ { \mathrm { t r a i n } } = 5 0 0$ numerical simulations, performed on the uniform mesh $( N _ { x } ^ { i } , N _ { t } ^ { i } )$ , where $N _ { x } ^ { \ i } = \bar { N _ { t } ^ { i } } ^ { \cdot } \in ( 1 5 , 3 0 , 5 9 )$ . We pretrained five lPINN models on each corresponding dataset. In the online stage, we use the lPINN models to obtain test solutions with $N _ { \mathrm { r e s } } = 1 0 0$ residual points. We compute the rRMSE errors with respect to the analytical solution on the mesh used to generate the corresponding dataset. For comparison, we also compute the rRMSE of the numerical test solution obtained on the corresponding mesh.

Figure 19 shows that lPINN and numerical solution errors decrease with decreasing $\Delta x = L / ( N _ { r } ^ { i } - 1 )$ . Two important observations are that lPINN errors are smaller than the numerical solution error for the same $\Delta x$ . More importantly, the lPINN error obtained on the grid $\frac { L } { 1 4 }$ is smaller than the error of the numerical solution obtained on the finer mesh $\frac { L } { 2 9 }$ Similarly, the lPINN error corresponding to grid size $\frac { L } { 2 9 }$ is smaller than the numerical solution error corresponding to the finer mesh $\frac { L } { 5 8 }$ . We also see the limitation of the lPINN superresolution in the considered setting (the training data set size, the number of basis functions, and the number of residual points): the lPINN error pretrained on the grid $\frac { L } { 1 4 }$ is larger than the error of the numerical solution obtained on the grid $\frac { L } { 5 8 }$

![](images/a436cafacbb899b6f09fb7af476d2375277229182376b280ad842f2903656616.jpg)  
Figure 19: lPINN prediction error versus mesh spacing $\Delta x$ for Burgers’ equation. For each mesh, the lPINN is pretrained using the numerical-solution dataset generated on that mesh and is then evaluated on the same mesh. Smaller $\Delta x$ corresponds to a finer mesh. $N _ { \mathrm { r e s } } = 1 0 0 , N _ { \eta } = 2 0 , N _ { \mathrm { t r a i n } } = 5 0 0$

## 3.4.3 Nonlinear Pendulum equation

We next investigate whether the same conclusion holds for the nonlinear pendulum equation, for which the lPINN is pretrained using the derivative-matching approach. We obtain the numerical solutions using the fourth-order Runge– Kutta (RK4) method. We generate three training datasets, $\{ D ^ { i } \} _ { i = 1 } ^ { 3 }$ each consisting of $N _ { \mathrm { t r a i n } } = 5 0 0$ numerical simulations performed on a uniform temporal grid with $N _ { t } ^ { i } \in ( \bar { 1 8 1 } , \bar { 2 4 1 } , 4 8 1 )$ over $t \in [ 0 , 3 0 ]$ , corresponding to a constant timestep $\Delta t = T / ( N _ { t } ^ { i } - 1 )$ . We pretrain lPINN models on each dataset. In the online stage, the lPINN models are used to obtain test solutions for $\gamma = 0 . 1$ and $\ell = 0 . 8$ using $N _ { \mathrm { r e s } } = 1 0 0 0$ randomly sampled residual points. The lPINN rRMSEs are compared to the rRMSEs of the numerical solutions computed on the same meshes and with $N _ { t } = 3 6 1$ . All rRMSEs are computed with respect to a high-resolution numerical reference solution, obtained using the adaptive fifth-order Runge–Kutta (RK5) method with $N _ { t } ^ { \mathrm { r e f } } = 2 8 8 1$

We consider two sets of lPINN solutions, one set obtained with $N _ { \mathrm { t r a i n } } = 5 0 0$ and $N _ { \eta } = 2 0 0$ and the other obtained with $N _ { \mathrm { t r a i n } } = 4 0 0 0$ and $N _ { \eta } = 4 0 0$

Figure 20 plots rRMSEs of the numerical and lPINN solutions as functions of the time step size. With $N _ { \mathrm { t r a i n } } = 5 0 0$ and $N _ { \eta } = \mathrm { \bar { 2 } 0 0 }$ , lPINN does not produce superresolution. Moreover, lPINN does not consistently outperform numerical solutions obtained on the same temporal grids. Increasing the ensemble size to $N _ { \mathrm { t r a i n } } = 4 0 0 0$ and the basis dimension to $N _ { \eta } = 4 0 0$ substantially improves the accuracy of the basis function representation, as evidenced by the reduction in the lPINN solution rRMSEs. Under this enriched setting, the lPINN rRMSEs are smaller than the rRMSEs of the numerical solutions obtained on the same temporal meshes. Furthermore, the lPINN trained with $N _ { t } = 1 8 1$ is more accurate than the numerical solution obtained with $N _ { t } = 2 4 1$ , and the lPINN trained with $N _ { t } = 2 4 1$ is more accurate than the numerical solution obtained with $N _ { t } = 3 6 1$ . Thus, the lPINN with $N _ { \mathrm { t r a i n } } = 4 0 0 0$ and $N _ { \eta } = 4 0 0$ achieves superresolution.

![](images/0e170c0d466305d8ae0b81b6704105be1d93b3d5cbda338fba153834823a43d7.jpg)  
Figure 20: lPINN prediction error $e _ { \mathrm { l P I N N } }$ versus time-step size $\Delta t$ for the nonlinear pendulum equation. For each temporal mesh, the lPINN is pretrained using the numerical-solution dataset generated on that mesh and is then evaluated on the same mesh. Smaller $\Delta t$ corresponds to a finer temporal mesh. $\bar { N } _ { \mathrm { r e s } } = 1 0 0 0$

## 4 Conclusion

We developed the Linearized Physics-Informed Neural Network (lPINN) as a physics-corrected neural reduced-basis method for many-query forward and inverse problems governed by differential equations. The central feature of the method is that an ensemble of numerical solutions is used offline to learn a continuous, coordinate-dependent neural basis functions, while the solution of each new problem is determined online by minimizing the governing-equation residual in the resulting low-dimensional coefficient space. The continuous neural basis can be pretrained through derivative matching or by augmenting solution reconstruction with physics residuals. These alternatives provide different mechanisms for representing the differential quantities required during online inference. The numerical experiments show that the preferred strategy can depend on the governing equation and neural architecture. Residualbased pretraining produced the most accurate downstream solutions for the advection–diffusion and Burgers’ equations, whereas derivative-matching pretraining was more effective for the oscillatory nonlinear pendulum problem. The latter example also demonstrated that the framework can incorporate Fourier feature networks when the solution family contains oscillatory components that are difficult to represent with a standard multilayer perceptron. Once the neural trial space has been constructed, lPINN restricts problem-specific optimization to the reduced coefficients and, in inverse problems, the unknown physical parameters. For linear differential operators, the online problem can be solved as regularized linear least squares. For nonlinear operators, the residual remains nonlinear in the reduced variables, but the nonlinear feature representation is not retrained. This distinction enables lPINN to extend final-layer adaptation beyond linear forward problems to nonlinear forward and inverse inference. The method was evaluated on the advection–diffusion equation, Burgers’ equation, and the nonlinear pendulum equation. In the reported test cases, lPINN produced lower solution and parameter errors than the corresponding vanilla PINNs while reducing online inference times by factors ranging from approximately one order of magnitude to more than three orders of magnitude. Its advantages were generally most pronounced when the numbers of residual or measurement points were limited. The cross-resolution experiments further showed that a continuous basis learned from coarse-mesh solution data could be evaluated on finer meshes without retraining and with nearly unchanged accuracy. These results demonstrate cross-resolution transfer of the learned representation, although the attainable accuracy remains limited by the quality and expressiveness of the offline trial space.

Several limitations define the scope of the present results. First, lPINN requires a representative ensemble of numerical solutions, and its reliability outside the parameter, initial-condition, or boundary-condition ranges covered during pretraining has not been established. Second, generating the training ensemble and learning the neural basis can be computationally expensive. The method is therefore most appropriate when this offline cost can be amortized over a sufficiently large number of forward or inverse queries. Third, performance depends on the basis dimension, architecture, pretraining objective, regularization, and online enforcement of initial and boundary conditions. The observed degradation of constraint accuracy in some experiments also indicates that properties learned offline should not always be assumed to remain satisfied when only the interior differential-equation residual is minimized online.

Future work should establish systematic criteria for selecting the basis dimension and pretraining strategy, develop adaptive enrichment procedures for parameter regions that are not adequately represented by the initial ensemble, and investigate exact or weighted enforcement of initial and boundary conditions during reduced inference. Additional pri orities include quantifying out-of-distribution generalization, analyzing amortized computational cost and conditioning, and comparing the method with classical residual-minimizing reduced-order models, random-feature methods, and neural operators. Extensions to higher-dimensional systems, spatially heterogeneous coefficients, variable geometries, and field-scale forward and inverse problems will be necessary to determine the practical range of the approach.

Overall, the results support the use of lPINN as an alternative to repeated full-network PINN optimization in many-query settings. By learning an operator-compatible continuous neural reduced basis offline and enforcing the governing physics in its coefficient space online, lPINN combines reusable data-driven representation with problem-specific physical correction.

The present comparisons are designed to isolate the computational consequences of replacing instance-specific fullnetwork PINN optimization with reduced coefficient inference in a pretrained neural trial space. They should not be interpreted as establishing superiority over classical or nonlinear reduced-order models. A systematic ROM comparison would require controlling for the snapshot ensemble, basis dimension, continuous reconstruction of discrete modes, derivative approximation, residual sampling, constraint enforcement, and hyper-reduction. Such a comparison is an important direction for future work, particularly for quantifying when the intrinsic coordinate continuity and automatic differentiability of the lPINN basis justify its additional offline training cost.

## 5 Acknowledgements

This research was supported by CUSSP (Center for Understanding Subsurface Signals and Permeability), an Energy Earthshot Research Center funded by the U.S. Department of Energy (DOE), Office of Science under FWP 81834, and the SRI (Strategic Research Initiative) Program at the UIUC’s Grainger College of Engineering.

## 6 Appendix

## 6.1 Summary of Hyperparameters, Error and Standard Deviation

Table 3: Forward Problem in Basis Size Studies for lPINN. $N _ { \mathrm { t r a i n } } = 5 0 0$ . For ADE equation, $N _ { \mathrm { I C } } = N _ { \mathrm { r e s } }$ and $N _ { \mathrm { B C } } = 2 N _ { \mathrm { r e s } } .$ For Burgers’ equation, $N _ { \mathrm { I C } } = N _ { \mathrm { B C } } = N _ { \mathrm { r e s } } .$
<table><tr><td>Fig.</td><td>Equation</td><td> $N _ { \eta }$ </td><td> $N _ { \mathrm { r e s } }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td>λBC</td><td>λ</td><td> $\mathrm { r R M S E } _ { u }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { u } }$ </td></tr><tr><td rowspan="5">2</td><td rowspan="5">ADE</td><td>10</td><td>100</td><td>0.7</td><td>1</td><td>0</td><td>0.002579</td><td>0.001858</td></tr><tr><td>20</td><td>200</td><td>1.5</td><td>1.75</td><td> $1 0 ^ { - 4 }$ </td><td>0.004899</td><td>0.002794</td></tr><tr><td>50</td><td>500</td><td>0.3</td><td>0.5</td><td>0</td><td>0.001249</td><td>0.002268</td></tr><tr><td>100</td><td>1000</td><td>0.6</td><td>1</td><td> $1 0 ^ { - 4 }$ </td><td>0.001342</td><td>0.002638</td></tr><tr><td>150 200</td><td>1500 2000</td><td>0.6</td><td>0.85</td><td>0</td><td>0.001555</td><td>0.001287</td></tr><tr><td rowspan="5">8</td><td rowspan="5">Burgers&#x27;</td><td></td><td></td><td>0.7</td><td>1</td><td>0</td><td>0.001499</td><td>0.001388</td></tr><tr><td>8</td><td>80</td><td>1</td><td>10</td><td>0</td><td>0.000962</td><td>0.000280</td></tr><tr><td>20</td><td>200</td><td>10</td><td>1</td><td> $1 0 ^ { - 8 }$ </td><td>0.000741</td><td>0.000185</td></tr><tr><td>50</td><td>500</td><td>30</td><td>3</td><td>0</td><td>0.001526</td><td>0.000159</td></tr><tr><td>100</td><td>1000</td><td>10</td><td>0</td><td>0</td><td>0.001880</td><td>0.000218</td></tr></table>

Table 4: Nonlinear Pendulum Forward Problem in the Basis Size Study for lPINN. $N _ { \mathrm { t r a i n } } = 5 0 0$
<table><tr><td> ${ \mathrm { F i g . } }$ </td><td> $N _ { \eta }$ </td><td> $N _ { \mathrm { r e s } }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td> $\lambda _ { \mathrm { I C } , 1 }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { \theta }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { \theta } }$ </td></tr><tr><td rowspan="3">13</td><td>100</td><td>1000</td><td>316.228</td><td>0.003</td><td>0</td><td>0.050856</td><td>0.005259</td></tr><tr><td>200</td><td>2000</td><td>0.01</td><td>10</td><td>0</td><td>0.002318</td><td>0.003529</td></tr><tr><td>400</td><td>4000</td><td>0</td><td>0</td><td> $1 0 ^ { - 4 }$ </td><td>0.003344</td><td>0.000264</td></tr></table>

Table 5: ADE Inverse Problem in the Basis Size Study for lPINN. $N _ { \mathrm { t r a i n } } = 5 0 0 , N _ { \mathrm { m } } = 4 0 , N _ { \mathrm { I C } } = N _ { \mathrm { B C } } = N _ { \mathrm { r e s } } .$ $\lambda _ { \mathrm { d a t a } } = 1$ , Fig. 2.
<table><tr><td> $N _ { \eta }$ </td><td> $N _ { \mathrm { r e s } }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { u }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { u } }$ </td><td> $\mathrm { r R M S E } _ { V }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { V } }$ </td><td> $\mathrm { r R M S E } _ { D }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { D } }$ </td></tr><tr><td>10</td><td>100</td><td>0</td><td>0.007071</td><td>0.004013</td><td>0.003085</td><td>0.002319</td><td>0.194328</td><td>0.221529</td></tr><tr><td>20</td><td>200</td><td>0</td><td>0.008056</td><td>0.000843</td><td>0.005200</td><td>0.001114</td><td>0.197752</td><td>0.035221</td></tr><tr><td>50</td><td>500</td><td> $1 0 ^ { - 6 }$ </td><td>0.003898</td><td>0.002181</td><td>0.004135</td><td>0.002421</td><td>0.107233</td><td>0.088383</td></tr><tr><td>100</td><td>1000</td><td> $1 0 ^ { - 8 }$ </td><td>0.006063</td><td>0.001795</td><td>0.004601</td><td>0.001695</td><td>0.166546</td><td>0.067861</td></tr><tr><td>150</td><td>1500</td><td>0</td><td>0.005475</td><td>0.001152</td><td>0.003259</td><td>0.001233</td><td>0.134873</td><td>0.046461</td></tr><tr><td>200</td><td>2000</td><td> $1 0 ^ { - 6 }$ </td><td>0.006883</td><td>0.001085</td><td>0.005783</td><td>0.000727</td><td>0.195849</td><td>0.046107</td></tr></table>

Table 6: Burgers Equation Inverse Problem in the Basis Size Study for lPINN. $N _ { \mathrm { t r a i n } } = 5 0 0 .$ $N _ { \mathrm { m } } = 5 0 $ $N _ { \mathrm { I C } } = N _ { \mathrm { B C } } =$ $N _ { \mathrm { r e s } } , \lambda _ { \mathrm { d a t a } } = 1$ , Fig. 8.
<table><tr><td> $N _ { \eta }$ </td><td> $N _ { \mathrm { r e s } }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { u }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { u } }$ </td><td>rRMSEν</td><td> $\sigma _ { \mathrm { r R M S E } _ { \nu } }$ </td></tr><tr><td>8</td><td>80</td><td>0</td><td>0.005800</td><td>0.001245</td><td>0.039483</td><td>0.008887</td></tr><tr><td>20</td><td>200</td><td> $1 0 ^ { - 6 }$ </td><td>0.006070</td><td>0.002712</td><td>0.042535</td><td>0.019688</td></tr><tr><td>50</td><td>500</td><td> $1 0 ^ { - 8 }$ </td><td>0.002922</td><td>0.000346</td><td>0.014586</td><td>0.003803</td></tr><tr><td>100</td><td>1000</td><td> $1 0 ^ { - 8 }$ </td><td>0.003454</td><td>0.000325</td><td>0.018585</td><td>0.002970</td></tr></table>

Table 7: Nonlinear Pendulum Inverse Problem in the Basis Size Study for lPINN. $N _ { \mathrm { t r a i n } } = 5 0 0 , N _ { \mathrm { m } } = 5 0 , \lambda _ { \mathrm { d a t a } } = 1$ Fig. 13.
<table><tr><td> $N _ { \eta }$ </td><td> $N _ { \mathrm { r e s } }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { \theta }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { \theta } }$ </td><td> $\mathrm { r R M S E } _ { \gamma }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { \gamma } }$ </td><td> $\mathrm { r R M S E } _ { \ell }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { \ell } }$ </td></tr><tr><td>100</td><td>1000</td><td>0</td><td>0.006772</td><td>0.000292</td><td>0.04747</td><td>0.006382</td><td>0.00092</td><td>0.000347</td></tr><tr><td>200</td><td>2000</td><td> $1 0 ^ { - 8 }$ </td><td>0.001398</td><td>0.000245</td><td>0.02072</td><td>0.001825</td><td>0.00083</td><td>0.000248</td></tr><tr><td>400</td><td>4000</td><td> $1 0 ^ { - 8 }$ </td><td>0.000600</td><td>0.000028</td><td>0.01841</td><td>0.000628</td><td>0.00134</td><td>0.000025</td></tr></table>

Table 8: Forward Problem: PINN and lPINN Error vs. $N _ { \mathrm { r e s } } . \ N _ { \mathrm { t r a i n } } = 5 0 0$ for lPINN. For ADE equation, $N _ { \mathrm { I C } } = N _ { \mathrm { r e s } }$ and $N _ { \mathrm { B C } } = 2 N _ { \mathrm { r e s } } .$ For Burgers’ equation, $N _ { \mathrm { I C } } = N _ { \mathrm { B C } } = N _ { \mathrm { r e s } } .$
<table><tr><td>Fig.</td><td>Equation</td><td> $N _ { \mathrm { r e s } }$ </td><td>Method</td><td> $N _ { \eta }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td>λBC</td><td>λ</td><td> $\mathrm { r R M S E } _ { u }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { u } }$ </td></tr><tr><td>4</td><td rowspan="6">ADE</td><td>100</td><td>PINN</td><td>一</td><td>1</td><td>1</td><td> $1 0 ^ { - 4 }$ </td><td>0.028976</td><td>0.026853</td></tr><tr><td>4</td><td>100</td><td>1PINN</td><td>50</td><td>0.003</td><td>0.01</td><td> $1 0 ^ { - 4 }$ </td><td>0.002354</td><td>0.005093</td></tr><tr><td>3,4</td><td>500</td><td>PINN</td><td>一</td><td>1</td><td>1</td><td> $1 0 ^ { - 4 }$ </td><td>0.010406</td><td>0.004776</td></tr><tr><td>3,4</td><td>500</td><td>1PINN</td><td>50</td><td>0.1</td><td>0.1</td><td> $1 0 ^ { - 4 }$ </td><td>0.001451</td><td>0.002036</td></tr><tr><td>4</td><td>1500</td><td>PINN</td><td>一</td><td>1</td><td>1</td><td> $1 0 ^ { - 4 }$ </td><td>0.010727</td><td>0.004047</td></tr><tr><td>4</td><td>1500</td><td>1PINN</td><td>50</td><td>0.3</td><td>0.3</td><td> $1 0 ^ { - 4 }$ </td><td>0.001420</td><td>0.001167</td></tr><tr><td>4</td><td></td><td>3000</td><td>PINN</td><td>一</td><td>1</td><td>1</td><td> $1 0 ^ { - 4 }$ </td><td>0.010341</td><td>0.002033</td></tr><tr><td>4</td><td rowspan="6"></td><td>3000</td><td>IPINN</td><td>50</td><td>3.225</td><td>6.45</td><td> $1 0 ^ { - 4 }$ </td><td>0.001447</td><td>0.000705</td></tr><tr><td>9,10</td><td>100</td><td>PINN</td><td>一</td><td>1</td><td>2</td><td> $1 0 ^ { - 6 }$ </td><td>0.043862</td><td>0.099357</td></tr><tr><td>9,10</td><td>100</td><td>1PINN</td><td>20</td><td>24</td><td>0.03</td><td> $1 0 ^ { - 6 }$ </td><td>0.000807</td><td>0.000189</td></tr><tr><td>10 10</td><td>300</td><td>PINN</td><td>一</td><td>1</td><td>2</td><td> $1 0 ^ { - 6 }$ </td><td>0.000902</td><td>0.000905</td></tr><tr><td>Burgers&#x27;</td><td>300</td><td>1PINN</td><td>20</td><td>25</td><td>1.00</td><td> $1 0 ^ { - 6 }$ </td><td>0.000780</td><td>0.000047</td></tr><tr><td>10</td><td>800</td><td>PINN</td><td>一</td><td>1</td><td>2</td><td></td><td> $1 0 ^ { - 6 }$ </td><td>0.000211</td><td>0.000331</td></tr><tr><td>10</td><td></td><td>800</td><td>1PINN</td><td>20</td><td>25</td><td>0.15</td><td> $1 0 ^ { - 6 }$ </td><td>0.000810</td><td>0.000041</td></tr><tr><td>10</td><td></td><td>1500</td><td>PINN</td><td>一</td><td>1</td><td>2</td><td> $1 0 ^ { - 6 }$ </td><td>0.000151</td><td>0.000099</td></tr><tr><td>10</td><td></td><td>1500</td><td>IPINN</td><td>20</td><td>25</td><td>0.005</td><td> $1 0 ^ { - 6 }$ </td><td>0.000835</td><td>0.000051</td></tr></table>

Table 9: Nonlinear Pendulum Forward Problem: PINN and lPINN Error vs. $N _ { \mathrm { r e s } } . \ N _ { \mathrm { t r a i n } } = 5 0 0$ for lPINN.
<table><tr><td>Fig.</td><td> $N _ { \mathrm { r e s } }$ </td><td>Method</td><td> $N _ { \eta }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td> $\lambda _ { \mathrm { I C } , 1 }$ </td><td>λ</td><td>rRMSEθ</td><td> $\sigma _ { \mathrm { r R M S E } _ { \theta } }$ </td></tr><tr><td>15</td><td>500</td><td>PINN</td><td></td><td>1</td><td>1</td><td> $1 0 ^ { - 8 }$ </td><td>0.271213</td><td>0.112946</td></tr><tr><td>15</td><td>500</td><td>IPINN</td><td>200</td><td> $1 0 ^ { - 4 }$ </td><td>0</td><td> $1 0 ^ { - 8 }$ </td><td>0.005755</td><td>0.013405</td></tr><tr><td>14,15</td><td>1000</td><td>PINN</td><td></td><td>1</td><td>1</td><td> $1 0 ^ { - 8 }$ </td><td>0.099410</td><td>0.159099</td></tr><tr><td>14,15</td><td>1000</td><td>1PINN</td><td>200</td><td>3</td><td>3</td><td> $1 0 ^ { - 8 }$ </td><td>0.002208</td><td>0.003536</td></tr><tr><td>15</td><td>2000</td><td>PINN</td><td></td><td>1</td><td>1</td><td> $1 0 ^ { - 8 }$ </td><td>0.015899</td><td>0.000640</td></tr><tr><td>15</td><td>2000</td><td>IPINN</td><td>200</td><td>10</td><td>100</td><td> $1 0 ^ { - 8 }$ </td><td>0.002178</td><td>0.004497</td></tr><tr><td>15</td><td>8000</td><td>PINN</td><td></td><td>1</td><td>1</td><td> $1 0 ^ { - 8 }$ </td><td>0.016420</td><td>0.000193</td></tr><tr><td>15</td><td>8000</td><td>1PINN</td><td>200</td><td>0.1</td><td>300</td><td> $1 0 ^ { - 8 }$ </td><td>0.002187</td><td>0.002994</td></tr></table>

Table 10: ADE Inverse Problem: PINN and lPINN Error vs. $N _ { \mathrm { m } } . \ N _ { \mathrm { t r a i n } } = 5 0 0 , N _ { \eta } = 5 0 , N _ { \mathrm { r e s } } = 5 0 0 , N _ { \mathrm { I C } } = 5 0 0 ,$ $N _ { \mathrm { B C } } = 5 0 0 , \lambda _ { \mathrm { d a t a } } = 1 .$

<table><tr><td> ${ \mathrm { F i g . } }$ </td><td> $N _ { \mathrm { m } }$ </td><td>Method</td><td> $N _ { \mathrm { r e s } }$ </td><td> $N _ { \eta }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td> $\lambda _ { \mathrm { B C } }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { u }$ </td><td>σrRMSEu</td><td>rRMSEV</td><td>σrRMSEV</td><td>rRMSED</td><td> $\sigma _ { \mathrm { r R M S E } _ { D } }$ </td></tr><tr><td rowspan="2">6</td><td>5</td><td>PINN</td><td>500</td><td>50</td><td>1</td><td>1</td><td> $1 0 ^ { - 4 }$ </td><td>0.253604</td><td>0.235495</td><td>0.486947</td><td>0.513296</td><td>0.996616</td><td>0.001698</td></tr><tr><td>5</td><td>IPINN</td><td>500</td><td>50</td><td>0</td><td>0</td><td>0</td><td>0.010980</td><td>0.004001</td><td>0.006302</td><td>0.001981</td><td>0.366958</td><td>0.158113</td></tr><tr><td rowspan="3">5,6</td><td>40</td><td>PINN</td><td>500</td><td>50</td><td>1</td><td>1</td><td>0</td><td>0.050878</td><td>0.030874</td><td>0.028985</td><td>0.019506</td><td>0.851279</td><td>0.279618</td></tr><tr><td>40</td><td>IPINN</td><td>500</td><td>50</td><td>0</td><td>0</td><td>0</td><td>0.003886</td><td>0.002205</td><td>0.004118</td><td>0.002396</td><td>0.107385</td><td>0.089965</td></tr><tr><td>80</td><td>PINN</td><td>500</td><td>50</td><td>1</td><td>1</td><td>0</td><td>0.018262</td><td>0.040573</td><td>0.024940</td><td>0.029324</td><td>0.491649</td><td>0.416658</td></tr><tr><td rowspan="2">6</td><td>80</td><td>IPINN</td><td>500</td><td>50</td><td>0</td><td>0</td><td>0</td><td>0.002356</td><td>0.000988</td><td>0.002744</td><td>0.001640</td><td>0.052280</td><td>0.056886</td></tr><tr><td>160</td><td>PINN</td><td>500</td><td>50</td><td>1</td><td>1</td><td> $1 0 ^ { - 6 }$ </td><td>0.002576</td><td>0.004371</td><td>0.001215</td><td>0.007101</td><td>0.172043</td><td>0.281317</td></tr><tr><td rowspan="2">6</td><td>160</td><td>IPINN</td><td>500</td><td>50</td><td>0</td><td>0</td><td>0</td><td>0.001833</td><td>0.000939</td><td>0.001649</td><td>0.001257</td><td>0.051348</td><td>0.056867</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Burgers Equation Inverse Problem: PINN and lPINN Error vs. $N _ { \mathrm { m } } \cdot N _ { \mathrm { t r a i n } } = 5 0 0 , N _ { \eta } = 5 0 , N _ { \mathrm { r e s } } = 1 0 0$ N = 100, N = 100.
<table><tr><td> ${ \mathrm { F i g . } }$ </td><td> $N _ { \mathrm { m } }$ </td><td>Method</td><td> $\lambda _ { \mathrm { d a t a } }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td> $\lambda _ { \mathrm { B C } }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { u }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { u } }$ </td><td>rRMSE(ν)</td><td>σrRMSEν</td></tr><tr><td rowspan="2">12</td><td>1</td><td>PINN</td><td>1</td><td>1</td><td>1</td><td> $1 0 ^ { - 8 }$ </td><td>0.089693</td><td>0.040345</td><td>0.43077</td><td>0.193140</td></tr><tr><td>1</td><td>1PINN</td><td>1</td><td>0</td><td>0</td><td> $1 0 ^ { - 8 }$ </td><td>0.002252</td><td>0.001035</td><td>0.00786</td><td>0.006911</td></tr><tr><td rowspan="2">11,12</td><td>5</td><td>PINN</td><td>1</td><td>1</td><td>1</td><td> $1 0 ^ { - 8 }$ </td><td>0.006089</td><td>0.016743</td><td>0.01190</td><td>0.027147</td></tr><tr><td>5</td><td>1PINN</td><td>1</td><td>0</td><td>0</td><td> $1 0 ^ { - 6 }$ </td><td>0.001568</td><td>0.000236</td><td>0.00318</td><td>0.002070</td></tr><tr><td rowspan="2">12</td><td>20</td><td>PINN</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0.001236</td><td>0.001589</td><td>0.00214</td><td>0.011031</td></tr><tr><td>20</td><td>1PINN</td><td>5</td><td>0</td><td>0</td><td> $1 0 ^ { - 8 }$ </td><td>0.001503</td><td>0.000098</td><td>0.00196</td><td>0.001913</td></tr><tr><td rowspan="2">12</td><td>50</td><td>PINN</td><td>1</td><td>1</td><td>1</td><td> $1 0 ^ { - 6 }$ </td><td>0.001059</td><td>0.000894</td><td>0.00173</td><td>0.004184</td></tr><tr><td>50</td><td>1PINN</td><td>1000</td><td>0</td><td>0</td><td> $1 0 ^ { - 8 }$ </td><td>0.001409</td><td>0.000052</td><td>0.001456</td><td>0.001385</td></tr></table>

Table 12: Nonlinear Pendulum Inverse Problem: PINN and lPINN Error vs. $N _ { \mathrm { m } } . \ : N _ { \mathrm { t r a i n } } = 5 0 0 ,$ $N _ { \eta } = 2 0 0 ,$ $N _ { \mathrm { r e s } } = 5 0 0$

<table><tr><td> ${ \mathrm { F i g . } }$ </td><td> $N _ { \mathrm { m } }$ </td><td>Method</td><td> $\lambda _ { \mathrm { d a t a } }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td> $\lambda _ { \mathrm { I C } , 1 }$ </td><td>λ</td><td>rRMSEθ</td><td> $\sigma _ { \mathrm { r R M S E } _ { \theta } }$ </td><td>rRMSEγ</td><td>σrRMSE,</td><td>rRMSEe</td><td>σrRMSEe</td></tr><tr><td rowspan="2">17</td><td>5</td><td>PINN</td><td>1</td><td>10</td><td>10</td><td> $1 0 ^ { - 4 }$ </td><td>0.434149</td><td>0.048105</td><td>1.10</td><td>0.337883</td><td>0.497</td><td>0.107659</td></tr><tr><td>5</td><td>IPINN</td><td>1</td><td>0</td><td>0</td><td> $1 0 ^ { - 4 }$ </td><td>0.501447</td><td>0.000623</td><td>0.314</td><td>0.018098</td><td>0.499</td><td>0.000960</td></tr><tr><td rowspan="2">16,17</td><td>10</td><td>PINN</td><td>1</td><td>10</td><td>10</td><td>0</td><td>0.376713</td><td>0.135898</td><td>1.09</td><td>0.507195</td><td>0.265</td><td>0.161166</td></tr><tr><td>10</td><td>IPINN</td><td>5</td><td>0</td><td>0</td><td> $1 0 ^ { - 8 }$ </td><td>0.001433</td><td>0.000873</td><td>0.01332</td><td>0.006621</td><td>0.001212</td><td>0.000433</td></tr><tr><td rowspan="2">17</td><td>20</td><td>PINN</td><td>1</td><td>10</td><td>10</td><td> $1 0 ^ { - 4 }$ </td><td>0.000530</td><td>0.000107</td><td>0.0145</td><td>0.003320</td><td>0.00114</td><td>0.000068</td></tr><tr><td>20</td><td>IPINN</td><td>5</td><td>0</td><td>0</td><td> $1 0 ^ { - 8 }$ </td><td>0.001150</td><td>0.000732</td><td>0.01308</td><td>0.009210</td><td>0.00121</td><td>0.000390</td></tr><tr><td rowspan="2">17</td><td>50</td><td>PINN</td><td>1</td><td>10</td><td>10</td><td> $1 0 ^ { - 6 }$ </td><td>0.000307</td><td>0.000244</td><td>0.0112</td><td>0.005170</td><td>0.00129</td><td>0.000288</td></tr><tr><td>50</td><td>IPINN</td><td>1000</td><td>0</td><td>0</td><td> $1 0 ^ { - 8 }$ </td><td>0.000825</td><td>0.000112</td><td>0.01336</td><td>0.006976</td><td>0.00123</td><td>0.000387</td></tr></table>

Table 13: ADE lPINN and dPICKLE Comparison. $N _ { \mathrm { t r a i n } } = 5 0 0 .$ $N _ { \eta } = 5 0 ,$ $N _ { \mathrm { r e s } } = 7 8 4$ $N _ { \mathrm { I C } } = 3 0 $ , and $N _ { \mathrm { B C } } = 5 9$

$$
\lambda _ { \mathrm { I C } , 0 }
$$

$$
\mathrm { r R M S E } _ { u }
$$

$$
1 0 ^ { - 4 }
$$

Table 14: Superresolution studies. For the ADE equation, $N _ { \mathrm { I C } } = N _ { \mathrm { r e s } }$ and $N _ { \mathrm { B C } } = 2 N _ { \mathrm { r e s } }$ . For Burgers’ equation, $N _ { \mathrm { I C } } = N _ { \mathrm { B C } } = N _ { \mathrm { r e s } } . \ N _ { \mathrm { t r a i n } } = 5 0 0$ for lPINN, rR $\mathrm { M S E _ { n u m } }$ is the error of the numerical solution.
<table><tr><td>Fig.</td><td>Equation</td><td>Mesh  $( N _ { t } \times N _ { x } )$ </td><td> $N _ { \eta }$ </td><td> $N _ { \mathrm { r e s } }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td> $\lambda _ { \mathrm { B C } }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { \mathrm { n u m } }$ </td><td> $\mathrm { r R M S E } _ { u }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { u } }$ </td></tr><tr><td rowspan="3">18</td><td rowspan="3">ADE</td><td> $3 0 \times 3 0$ </td><td>50</td><td>500</td><td>7</td><td>6</td><td>0</td><td>0.055363</td><td>0.008344</td><td>0.002795</td></tr><tr><td> $5 9 \times 5 9$ </td><td>50</td><td>500</td><td>3.5</td><td>5</td><td> $1 0 ^ { - 8 }$ </td><td>0.024839</td><td>0.002061</td><td>0.002378</td></tr><tr><td> $2 4 0 \times 2 4 0$ </td><td>50</td><td>500</td><td>0.48</td><td>0.96</td><td>0</td><td>0.002109</td><td>0.001816</td><td>0.001932</td></tr><tr><td rowspan="3">19</td><td rowspan="3">Burgers&#x27;</td><td> $1 5 \times 1 5$ </td><td>20</td><td>100</td><td>30</td><td>70</td><td>0</td><td>0.120381</td><td>0.005004</td><td>0.000806</td></tr><tr><td> $3 0 \times 3 0$ </td><td>20</td><td>100</td><td>8</td><td>0.1</td><td> $1 0 ^ { - 4 }$ </td><td>0.029587</td><td>0.003316</td><td>0.000202</td></tr><tr><td> $5 9 \times 5 9$ </td><td>20</td><td>100</td><td>0.3</td><td>10</td><td> $1 0 ^ { - 8 }$ </td><td>0.005350</td><td>0.000618</td><td>0.000088</td></tr></table>

Table 15: Nonlinear Pendulum Superresolution Study. $N _ { \mathrm { r e s } } = 1 0 0 0$ for lPINN. $\mathrm { r R M S E } _ { \mathrm { n u m } }$ is the error of the numerical solution.
<table><tr><td>Fig.</td><td> $N _ { t }$ </td><td> $N _ { \mathrm { t r a i n } }$ </td><td> $N _ { \eta }$ </td><td> $\lambda _ { \mathrm { I C } , 0 }$ </td><td> $\lambda _ { \mathrm { I C } , 1 }$ </td><td>λ</td><td> $\mathrm { r R M S E } _ { \mathrm { n u m } }$ </td><td> $\mathrm { r R M S E } _ { \theta }$ </td><td> $\sigma _ { \mathrm { r R M S E } _ { \theta } }$ </td></tr><tr><td rowspan="5">20</td><td>181</td><td>500</td><td>200</td><td>10</td><td>10</td><td> $1 0 ^ { - 8 }$ </td><td>0.004882</td><td>0.003288</td><td>0.002839</td></tr><tr><td>241</td><td>500</td><td>200</td><td>10</td><td>1</td><td> $_ 0$ </td><td>0.001046</td><td>0.004369</td><td>0.003157</td></tr><tr><td>481</td><td>500</td><td>200</td><td>100</td><td>100</td><td> $1 0 ^ { - 6 }$ </td><td>0.000120</td><td>0.000597</td><td>0.001220</td></tr><tr><td>181</td><td>4000</td><td>400</td><td>10</td><td>10</td><td> $_ 0$ </td><td>0.004882</td><td>0.000691</td><td>0.001927</td></tr><tr><td>241</td><td>4000</td><td>400</td><td>0.1</td><td>1</td><td> $1 0 ^ { - 6 }$ </td><td>0.001046</td><td>0.000131</td><td>0.000760</td></tr><tr><td></td><td>481</td><td>4000</td><td>400</td><td>0.1</td><td>0.1</td><td>0</td><td>0.000120</td><td>0.000114</td><td>0.000362</td></tr></table>

Table 16: Offline pretraining. For ADE and Burgers’ equation, $\lambda _ { f } = \lambda _ { \mathrm { I C } , 0 } = \lambda _ { \mathrm { B C } } = 1$ in 15, and for nonlinear pendulum equation, $\lambda _ { 1 } = \lambda _ { 4 } = 1$ in ?? and ??.
<table><tr><td>Equation</td><td>λ</td></tr><tr><td>ADE</td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Burgers</td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Nonlinear pendulum mean network</td><td> $_ 0$ </td></tr><tr><td>Nonlinear pendulum basis</td><td> $1 0 ^ { - 6 }$ </td></tr></table>

## References

[1] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics, 378:686–707, 2019.

[2] GE Karniadakis, IG Kevrekidis, L Lu, P Perdikaris, S Wang, and L Yang. Physics-informed machine learning: Nature reviews physics.(2021). 2021.

[3] Alexandre M Tartakovsky, C Ortiz Marrero, Paris Perdikaris, Guzel D Tartakovsky, and David Barajas-Solano. Physics-informed deep neural networks for learning parameters and constitutive relationships in subsurface flow problems. Water Resources Research, 56(5):e2019WR026731, 2020.

[4] Ramakrishna Tipireddy, Paris Perdikaris, Panos Stinis, and Alexandre Tartakovsky. A comparative study of physics-informed neural network models for learning unknown dynamics and constitutive relations. arXiv preprint arXiv:1904.04058, 2019.

[5] Aditi Krishnapriyan, Amir Gholami, Shandian Zhe, Robert Kirby, and Michael W Mahoney. Characterizing possible failure modes in physics-informed neural networks. Advances in neural information processing systems, 34:26548–26560, 2021.

[6] Jorge F Urbán, Petros Stefanou, and José A Pons. Unveiling the optimization process of physics informed neural networks: How accurate and competitive can pinns be? Journal ofComputational Physics, 523:113656, 2025.

[7] Sifan Wang, Yujun Teng, and Paris Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5):A3055–A3081, 2021.

[8] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33:7537–7547, 2020.

[9] Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why pinns fail to train: A neural tangent kernel perspective. Journal ofComputational Physics, 449:110768, 2022.

[10] Jan S. Hesthaven, Gianluigi Rozza, and Benjamin Stamm. Certified Reduced Basis Methodsfor Parametrized Partial Differential Equations. Springer International Publishing, Cham, 2016.

[11] Alfio Quarteroni, Andrea Manzoni, and Federico Negri. Reduced Basis Methods for Partial Differential Equations: An Introduction. Springer International Publishing, Cham, 2016.

[12] A.M. Tartakovsky, D.A. Barajas-Solano, and Q. He. Physics-informed machine learning with conditiona Karhunen-Loève expansions. Journal of Computational Physics, 426:109904, 2021.

[13] Kevin Carlberg, Charbel Bou-Mosleh, and Charbel Farhat. Efficient non-linear model reduction via a least-squares petrov–galerkin projection and compressive tensor approximations. International Journalfor Numerical Methods in Engineering, 86(2):155–181, 2011.

[14] Kevin Carlberg, Matthew Barone, and Harbir Antil. Galerkin v. least-squares petrov–galerkin projection in nonlinear model reduction. Journal ofComputational Physics, 330:693–734, 2017.

[15] Kevin Carlberg, Charbel Farhat, Julien Cortial, and David Amsallem. The GNAT method for nonlinear model reduction: Effective implementation and application to computational fluid dynamics and turbulent flows. Journal of Computational Physics, 242:623–647, 2013.

[16] Youngsoo Choi and Kevin Carlberg. Space–time least-squares petrov–galerkin projection for nonlinear model reduction. SIAM Journal on Scientific Computing, 41(1):A26–A58, 2019.

[17] Alexandre M Tartakovsky and Yifei Zong. Physics-informed machine learning method with space-time karhunenloève expansions for forward and inverse partial differential equations. Journal of Computational Physics, 499:112723, 2024.

[18] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. Neural fields in visual computing and beyond. Computer Graphics Forum, 41(2):641–676, 2022.

[19] Peter Yichen Chen, Jinxu Xiang, Dong Heon Cho, Yue Chang, G. A. Pershing, Henrique Teles Maia, Maurizio M. Chiaramonte, Kevin Carlberg, and Eitan Grinspun. CROM: Continuous reduced-order modeling of PDEs using implicit neural representations. In International Conference on Learning Representations, 2023.

[20] Minji Kim, Tianshu Wen, Kookjin Lee, and Youngsoo Choi. Physics-informed reduced order model with conditional neural fields. arXiv preprint arXiv:2412.05233, 2024.

[21] Vikas Dwivedi and Balaji Srinivasan. Physics informed extreme learning machine (PIELM)–a rapid method for the numerical solution of partial differential equations. Neurocomputing, 391:96–118, 2020.

[22] Xu Liu, Wen Yao, Wei Peng, and Weien Zhou. Bayesian physics-informed extreme learning machine for forward and inverse PDE problems with noisy data. Neurocomputing, 549:126425, 2023.

[23] Jingrun Chen, Xurong Chi, Weinan E, and Zhouwang Yang. Bridging traditional and machine learning-based algorithms for solving PDEs: The random feature method. arXiv preprint arXiv:2207.13380, 2022.

[24] Shaan Desai, Marios Mattheakis, Hayden Joy, Pavlos Protopapas, and Stephen Roberts. One-shot transfer learning of physics-informed neural networks. arXiv preprint arXiv:2110.11286, 2021.

[25] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3:218–229, 2021.

[26] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2021.

[27] Sifan Wang, Hanwen Wang, and Paris Perdikaris. Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. Science Advances, 7(40):eabi8605, 2021.

[28] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. ACM Transactions on Machine Learning Research, 3(1):1–27, 2024.

[29] Sifan Wang, Hanwen Wang, and Paris Perdikaris. On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 384:113938, 2021.