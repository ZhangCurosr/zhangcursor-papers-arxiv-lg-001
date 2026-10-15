# A Bregman Proximal Viewpoint on Neural Operators

Abdel-Rahim Mezidi <sup>1</sup> <sup>\*</sup> Jordan Patracone <sup>1</sup> <sup>\*</sup> Saverio Salzo <sup>2</sup> <sup>\*</sup> Amaury Habrard <sup>1</sup> Massimiliano Pontil <sup>3</sup> <sup>4</sup> Remi Emonet <sup>1</sup> Marc Sebban <sup>1</sup>

## Abstract

We present several advances on neural operators by viewing the action of operator layers as the minimizers of Bregman regularized optimization problems over Banach function spaces. The proposed framework allows interpreting the activation operators as Bregman proximity operators from dual to primal space. This novel viewpoint is general enough to recover classical neural operators as well as a new variant, coined Bregman neural operators, which includes the inverse activation operator and features the same expressivity of standard neural operators. Numerical experiments support the added benefits of the Bregman variant of Fourier neural operators for training deeper and more accurate models.

## 1. Introduction

Neural operators (Kovachki et al., 2021; 2023), a recent extension of neural networks, have emerged as a versatile framework for learning mappings between function spaces. These operators have shown great potential in solving partial differential equations (PDEs) and simulating complex dynamical systems. The exploration of neural architectures for the approximation and learning of operators has led to the development of a variety of models.

One influential contribution is the Fourier Neural Operator (FNO) (Li et al., 2021a), sketched in Figure 1, which transforms encoded input data into frequency components in order to learn intricate relationships in the frequency domain. More recently, the Group-Equivariant FNO (G-FNO) (Helwig et al., 2023) additionally leverages symmetries to design equivariant Fourier layers, thereby enhancing the representation power and robustness of the architecture. To better scale the depth of neural operators, the F-FNO (Tran et al., 2023) proposed separable spectral layers and improved residual connections, along with a bag of training tricks. The FNO are extended to Wavelet Neural Operators (WNO) (Tripura & Chakraborty, 2023) by replacing Fourier layers with wavelet layers to further exploit multiscale information. The U-shaped Neural Operator (U-NO) (Rahman et al., 2023) adapts the U-net architecture for neural operators, enabling mapping between function spaces through integral operators, thus broadening the applicability of neural architectures to diverse domains. Differently, the DeepONet architecture (Lu et al., 2021) comprises two intertwined components: a branch network responsible for encoding discrete input function spaces, and a trunk network dedicated to encoding the domain of output functions. Operating as a conditional model, DeepONet leverages the embedding of inputs and outputs via a dot product operation, facilitating the approximation of complex functions through a structured network topology. Finally, Neural Inverse Operators (NIO) (Molinaro et al., 2023) tackle inverse problems by combining DeepONet and FNO architectures to map operators to functions, thereby extending the applicability of neural operators to coefficient estimation tasks.

Some approaches inspired by attention mechanisms, pivotal in image and natural language processing, have also been considered in operator learning. LOCA (Learning Operators with Coupled Attention) (Kissas et al., 2022) facilitates robust gradient estimation, particularly in scenarios with limited training data, by combining attention with kernel mechanisms. The General Neural Operator Transformer (GNOT) (Hao et al., 2023) is a scalable framework based self-attention mechanisms allowing to deal with heterogeneous inputs useful for modeling diverse physical systems.

Some physics-informed variants integrating information from PDEs during the learning process have been proposed enhancing model interpretability and generalization: PI-DeepONet (Wang et al., 2021) and its Long-Time Integration variant (LTI-PI-DeepONet) (Wang & Perdikaris, 2023), PINO (Physics-Informed Neural Operator) (Li et al., 2021b) a hybrid extension of FNO, or other variations such as V-DeepONet (Goswami et al., 2022) and Modified Deep-ONet (Wang et al., 2022).

![](images/db9dc97ee0b5708bc78a634c8b24e2223d9a7afcaf189524db27c3afcf7acd9a.jpg)  
Figure 1: Illustration of the t-th layer of Fourier Neural Operators. The upper branch applies a linear transformation $R _ { t }$ to the Fourier modes using the Fourier transform $\mathcal { F }$ and its inverse ${ \mathcal { F } } ^ { - 1 }$ . The lower branch performs an affine transformation in the latent space.

Contributions. Unlike previous works (Kovachki et al., 2021), which directly consider the compositional form of neural operators, our approach introduces a distinct perspective by formulating the action of each operator layer as the minimizer of a regularized optimization problem over functions. This optimization connects the current hidden representation to the next, with the choice of a regularization implicitly defining the activation operator through the lens of the Bregman proximity operator. Our framework not only covers existing neural operators but also introduces a novel variant, termed Bregman neural operator, which demonstrates improved predictive performance as its depth increases. Its applicability is grounded by universal approximation results proven for sigmoidal-type activation operators. Beyond its unifying aspect and its ability to design novel neural operators, the proposed framework allows applying the extensive body of literature on proximal numerical optimization, of which Bregman proximity operators belong to, in order to study neural operators. This opens the way to extend the analysis done on neural networks to (Bregman) neural operators in the same spirit of Combettes & Pesquet (2020a;b).

Outline. The paper is organized as follows: Section 2 is dedicated to the presentation of definitions and background knowledge on neural operators and Bregman proximity operator. In Section 3, we introduce the operator layers as the solution of a functional optimization problem. In addition, we show that this new mapping allows recovering the classical neural operators and creating a more general family of so-called Bregman neural operators. In Section 4, we provide a preliminary universal approximation result for Bregman neural operators. Finally, in Section 5, we conduct an experimental study comparing on benchmark datasets our Bregman variant with the different FNO improvements.

## 2. Background and Definitions

Here, we introduce some definitions required for the understanding of the rest of the paper as well as the necessary background on neural operators and Bregman proximity operator. We will use basic concepts from convex analysis such as subdifferential, $\Gamma _ { 0 }$ space and Fenchel conjugate, whose definitions are recalled in Appendix A.

![](images/94846f74c3fc69452905d72b12cf3f4d0f5d5c454d5dfc5b6b0bcdcd413cf9ee.jpg)  
Figure 2: Illustration of the t-th layer of (Bregman) Neural Operators. On the left, the identity term and the linear term $K _ { t } v _ { t } + b _ { t }$ have been merged into $( I + K _ { t } ) v _ { t } = W _ { t } v _ { t }$ . For both, ${ \cal K } _ { t } ^ { \mathrm { a c } }$ represents any absolutely continuous operator.

## 2.1. Operator Learning

Operator learning finds significant applications in the context of PDEs in order to efficiently approximate solutions to PDEs without the need to solve them repeatedly from scratch (Li et al., 2021b; Serrano et al., 2023; Raonic et al., 2023). Given a nonempty bounded open set $D \subset \mathbb { R } ^ { d }$ , and some time horizon $\tau > 0$ , we consider the generic family of PDEs over $D \times [ 0 , \tau ]$ of the form

$$
\begin{array} { r l } & { F _ { a } \big ( ( \partial ^ { \alpha } u ( x , t ) ) _ { \alpha \in  { \mathbb { N } } ^ { d + 1 } , | \alpha | \le k } \big ) = f ( x , t ) \mathrm { o n } D \times  { \left| 0 , \tau \right] } , } \\ & { \mathrm { a n d } \left\{ u ( x , 0 ) = u _ { 0 } ( x ) \mathrm { o n } D , \right. } \\ & { \left. u ( x , t ) = u _ { b } ( x , t ) \mathrm { o n } \partial D \times  { \left| 0 , \tau \right| } , \right. } \end{array}\tag{1}
$$

where $F _ { a }$ is a (possibly) nonlinear partial differential operator, f a source term, $u _ { b }$ a boundary condition, $u _ { 0 }$ an initial condition, and $u \colon D \to \mathbb { R } ^ { n }$ the PDE solution.

The main problem we will tackle in our numerical section is the initial value problem. This involves finding the oracle mapping $\mathcal { G }$ from any initial condition function $u _ { 0 }$ to the solution $u ( \cdot , \bar { \tau } )$ of the PDE at a certain time horizon $\bar { \tau } \in ] 0 , \tau ]$ More generally, the oracle operator $\mathcal { G }$ could be a mapping between two different function spaces A and U. Without loss of generality, given some bounded open sets $D \subset \mathbb { R } ^ { d }$ , with $d \in \mathbb { N } _ { + }$ , we let $\mathcal { A } = \mathcal { A } ( D , \mathbb { R } ^ { n } )$ and $\mathcal { U } = \mathcal { U } ( D , \mathbb { R } ^ { k } )$ ), with $n , k \in  { \mathbb { N } } _ { + }$ , be some separable Banach spaces of functions. For instance, A can represent the spaces of continuous functions from $D \to \mathbb { R } ^ { n }$ . Hereafter, A and U will be referred to as the spaces of input functions and output functions, respectively. In a nutshell, operator learning consists in finding the unknown ground-truth correspondence operator $\mathcal { G } \colon A \to \mathcal { U }$ given $N \in  { \mathbb { N } } _ { + }$ pairs of input-output functions $\{ a _ { i } , u _ { i } \} _ { i = 1 } ^ { N }$

## 2.2. Neural Operators

Among the existing models to approximate ${ \mathcal { G } } .$ , we focus on neural operators, which are parametric mappings $\mathcal { N } \colon A \to$ U of the form

$$
( \forall a \in { \mathcal { A } } ) , \quad { \mathcal { N } } ( a ) = { \mathcal { Q } } \circ { \mathcal { L } } _ { T } \circ \dots \circ { \mathcal { L } } _ { 1 } \circ { \mathcal { P } } ( a ) ,\tag{2}
$$

where

$\mathcal { P } \colon \mathcal { A } ( D , \mathbb { R } ^ { n } ) \to \mathcal { A } ( D , \mathbb { R } ^ { n _ { 0 } } )$ is a local lifting operator mapping the input function to its first hidden representation;

$\mathcal { Q } \colon \mathcal { U } ( D , \mathbb { R } ^ { n _ { T } } ) \to \mathcal { U } ( D , \mathbb { R } ^ { k } )$ is a local projection operator mapping the last hidden representation to the output function;

• For every $t \ \in \ \{ 1 , . . . , T \} , \ \mathcal { L } _ { t } \colon \mathcal { V } _ { t - 1 } ( D _ { t } , \mathbb { R } ^ { n _ { t - 1 } } ) \ $ $\mathcal { V } _ { t } ( D _ { t } , \mathbb { R } ^ { n _ { t } } )$ is an operator layer where each $D _ { t } \subset \mathbb { R } ^ { d _ { t } }$ is an open bounded set, $\mathcal { V } _ { t } = \mathcal { V } _ { t } ( D _ { t } , \mathbb { R } ^ { n _ { t } } )$ is a suitable Banach space of functions such that $\mathcal { V } _ { 0 } = \mathcal { A } ( D , \mathbb { R } ^ { n _ { 0 } } )$ and $\mathcal { V } _ { T } = \mathcal { U } ( D , \mathbb { R } ^ { n _ { T } } )$ , for consistency.

• Each component of the neural operator (2) depends on a finite dimensional parameter. Collectively those parameters constitute a vector $\theta \in \Theta \subset \mathbb { R } ^ { p }$

Most methodological developments in neural operators have focused on tailoring the operator layers $\mathcal { L } _ { 1 } , \ldots , \mathcal { L } _ { T }$ to specific application. Traditionally, their design mirrors standard neural networks, replacing finite-dimensional linear layers with integral linear operators in function spaces and interpreting activation functions as Nemytskii operators that apply nonlinear transformations pointwise. When the input spaces $D _ { t }$ are the same throughout the layers and equals $D ,$ a popular class of operator layers, sketched in Figure 2a, is of the form

$$
\mathcal { L } _ { t } ( v _ { t } ) = \sigma ( W _ { t } v _ { t } + K _ { t } ^ { \mathrm { a c } } ( v _ { t } ) + b _ { t } ) ,\tag{3}
$$

where $W _ { t } \ \in \ \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } }$ is a matrix, $b _ { t } ~ \in ~ \mathbb { R } ^ { n _ { t } }$ is a bias vector and σ is a local nonlinear map acting pointwise from $\mathbb { R } ^ { n _ { t } }$ to $\mathbb { R } ^ { n _ { t } }$ . Moreover, we have a non-local linear operator $K _ { t } ^ { \mathrm { a c } } \colon L ^ { 2 } ( D , \mathbb { R } ^ { n _ { t - 1 } } ) \to L ^ { 2 } ( D , \mathbb { R } ^ { n _ { t } } )$ . In its simplest version, ${ \cal K } _ { t } ^ { \mathrm { a c } }$ is an integral kernel operator of the form $\begin{array} { r } { ( K _ { t } ^ { \mathrm { a c } } ( v ) ) ( x ) = \bar { \int _ { D } k _ { t } ( x , y ) v ( y ) d y } } \end{array}$ , for all $x \in D .$ , with $k _ { t }$ being a kernel to be specified (Kovachki et al., 2023). Specific examples include those based upon a convolution performed in the Fourier space (Li et al., 2021a; Kovachki et al., 2021), a graph kernel network (Anandkumar et al., 2020) or its multipole variant (Li et al., 2020) to name a few. Hereafter, we follow a different path and propose to interpret operator layers from the viewpoint of a proximal optimization by seeing the parametric form of (3) as the minimizer of a Bregman regularized optimization problem. This novel perspective allows us to propose a novel architecture, displayed in Figure 2b, of the form

$$
\mathcal { L } _ { t } ( v _ { t } ) = \sigma ( \sigma ^ { - 1 } ( v _ { t } ) + K _ { t } v _ { t } + K _ { t } ^ { \mathrm { a c } } ( v _ { t } ) + b _ { t } ) ,\tag{4}
$$

involving an additional nonlinear term $\sigma ^ { - 1 } ( v _ { t } )$ , and where $K _ { t } \in \mathbb { R } ^ { n _ { t } \times n _ { t - } }$ <sup>1</sup> is a matrix. In this formulation, when all the weights are zero, then $\mathcal { L } _ { t }$ is the identity operator. In practice, we observe that this property allows training deeper and more accurate models. A similar architecture was originally proposed in Frecon et al. (2022) in the finite dimensional setting. Extending this work to neural architectures acting on Banach function spaces requires addressing non-trivial mathematical challenges. These include defining operator layers rigorously, particularly the proper formulation of Legendre functions on function spaces, the associated Bregman divergence, and the Bregman proximity operator. In the next section, we formalize these notions, laying the groundwork for the proposed novel perspective on neural operators. The reader interested in the technical details is invited to refer to Appendix A.

## 2.3. Bregman Proximity Operator

At the core of our framework is the link between activation operators and Bregman proximity operators. The definition of the Bregman proximity operator hinges on a Bregman divergence, often referred to as a distance, which is derived from a Legendre function (see, e.g., Rockafellar (1970)).

Definition 2.1 (Legendre function). A function $\phi \colon { \mathbb { R } ^ { n } } \to$ $] - \infty , + \infty ]$ is called Legendre if it is proper convex lower semicontinuous and satisfies the following properties: i) in $\operatorname { t } ( \operatorname { d o m } \phi ) = \operatorname { d o m } \partial \phi$ and ∂ϕ is single-valued on its domain; ii) ϕ is strictly convex on int(dom ϕ).

In the finite dimensional setting, Legendre functions $\phi$ are typically built from an elementary Legendre function $ \varphi \colon \mathbb R \to ] - \infty , + \infty ] \mathrm { ~ a s ~ } \phi \colon x \in \mathbb R ^ { n } \to \textstyle \sum _ { i = 1 } ^ { n } \varphi ( x _ { i } )$ Since here we stand in an infinite dimensional setting, i.e., Lebesgue function space, the counterpart of the previous finite sum structure is a convex integral functional defined below (see Fact 2 in Appendix for a more rigorous treatment). Also, we will allow vector valued functions.

Fact 1. Let $D \subset \mathbb { R } ^ { d }$ be a bounded set and set the dual spaces $\mathcal { V } = L ^ { p } ( D , \mathbb { R } ^ { n } )$ and $\mathcal { V } ^ { * } = L ^ { q } ( D , \mathbb { R } ^ { n } )$ appropriately paired. Given a Legendre function ϕ, then

$$
\Phi ( v ) = \int _ { D } \phi ( v ( x ) ) d x\tag{5}
$$

defines a convex integral functional, with its subdifferential ∂Φ consisting of functions v for which v(x) lies within the interior of $\phi \mathbf { \hat { s } }$ domain and $\nabla \phi \circ v \in \mathcal { V } ^ { * }$ . The subdifferential is single-valued, and $\nabla \phi \circ v ,$ will be denoted by $\tilde { \nabla } \Phi ( v )$ suggesting it will serve as a kind of gradient of Φ at v.

The integral functional Φ in (5) inherits certain properties of $\phi ,$ such as p-uniform convexity — an extension of strong convexity when $p = 2$ . This characteristic, proven in Proposition A.3, is key to the mathematical soundness of our analysis (see also Remarks A.2 and A.4).

We are now equipped to define Bregman distances in Lebesgue spaces. First introduced by Bregman in (Bregman, 1967), Bregman divergence extends the notion of distance beyond metric spaces, capturing asymmetries and curvature induced by convex functions. Unlike Euclidean distance, it reflects the local geometry of the function defining it, making them valuable in optimization and variational analysis. Definition 2.2 (Bregman distance in Lebesgue spaces). Under the notations of Fact 1, the Bregman distance with respect to Φ reads, $( \forall u \in \mathcal { V } , \forall v \in \mathcal { V } )$

$$
D _ { \Phi } ( u , v ) = \left\{ \begin{array} { l l } { \Phi ( u ) - \Phi ( v ) - \left. u - v , \tilde { \nabla } \Phi ( v ) \right. \mathrm { ~ i f ~ } v \in \mathrm { d o m } \partial \Phi } \\ { + \infty \qquad } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Finally, we can define the Bregman proximity operator (Nguyen, 2017), which extends the (Euclidean) proximity operator, widely used in optimization. The Euclidean proximity operator itself generalizes projections by replacing the indicator function of a convex set with appropriate convex functions. For additional details, the reader can refer to Bauschke & Combettes (2017).

Definition 2.3 (Bregman proximity operator). Let $\nu =$ $L ^ { p } ( D , \mathbb { R } ^ { n } )$ with $p \in [ 1 , + \infty [ .$ . Let $g \in \Gamma _ { 0 } ( \mathcal { V } )$ and let $\Phi \in \Gamma _ { 0 } ( \mathcal { V } )$ be defined as in Fact 1, with $\phi \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ be Legendre and such that ran $\partial ( \Phi + g ) = \mathcal { V } ^ { * }$ Then the Bregman proximity operator of g relative to Φ is defined as

$$
\mathrm { p r o x } _ { g } ^ { \Phi } \colon \mathcal { V } ^ { * }  \mathcal { V } , \quad v ^ { * } \mapsto \mathrm { a r g m i n } \big \{ \langle \cdot , - v ^ { * } \rangle + \Phi + g \big \} .
$$

Note that $\mathrm { p r o x } _ { g } ^ { \Phi }$ is well-defined since $\Phi + g$ is strictly convex, lower semicontinuous and ran $\partial ( \Phi + g ) = \mathcal { V } ^ { * }$ , and it holds $\mathrm { p r o x } _ { g } ^ { \Phi } = [ \partial ( \Phi + g ) ] ^ { - 1 }$

## 3. Revisiting Neural Operators

In Section 3.1, we propose a novel Bregman proximal viewpoint on operator layers. Then, we establish several connections. First, we show in Section 3.2 that the proposed framework is general enough to recover most classical operator layers when the Legendre function ϕ is the Euclidean distance. Second, we show in Section 3.3 how it yields a new variant of neural operators when ϕ defines a general Bregman divergence. Finally, we apply our framework to Fourier neural operators in Section 3.4.

## 3.1. Bregman Proximal Viewpoint on Operator Layers

Departing from usual kernel-based points of view (Kovachk et al., 2021), we suggest defining operator layers as the solution of functional optimization problems. For every $t = 1 , \cdot \cdot \cdot , T , \mathcal { L } _ { t } \colon \mathcal { V } _ { t - 1 }  \mathcal { V } _ { t }$

$$
\begin{array} { r l } & { \mathcal { L } _ { t } ( v ) = \underset { w \in \mathcal { V } _ { t } } { \mathrm { a r g m i n } } - \langle w , \mathcal { K } _ { t } ( v ) + b _ { t } \rangle + g _ { t } ( w ) + D _ { \Phi _ { t } } ( w , \mathcal { M } _ { t } v ) } \\ & { \qquad = \mathrm { p r o x } _ { g _ { t } } ^ { \Phi _ { t } } \big ( \tilde { \nabla } \Phi _ { t } ( \mathcal { M } _ { t } v ) + \mathcal { K } _ { t } ( v ) + b _ { t } \big ) , } \end{array}\tag{6}
$$

where

$\Phi _ { t } \colon \mathcal { V } _ { t } \to ] - \infty , + \infty ]$ is a convex integral functional on an appropriate Lebesgue space based on some Legendre function $\phi _ { t } \in \Gamma _ { 0 } ( \mathbb { R } ^ { n _ { t } } )$ , as defined in Fact 1. $D _ { \Phi _ { t } } \colon \mathcal { V } _ { t } \times$ $\mathcal { V } _ { t } \to [ 0 , + \infty ]$ is the corresponding Bregman distance as detailed in Definition 2.2

$\mathcal { M } _ { t } \colon \mathcal { V } _ { t - 1 } \to \mathcal { V } _ { t }$ is a bounded linear operator which maps dom $\partial \Phi _ { t - 1 }$ into dom $\partial \Phi _ { t }$

$b _ { t } \in \mathcal { V } _ { t } ^ { * }$ and $\boldsymbol { \mathcal { K } } _ { t } \colon \boldsymbol { \mathcal { V } } _ { t - 1 } \to \boldsymbol { \mathcal { V } } _ { t } ^ { * }$ is a bounded linear operator of the form

$$
\mathcal { K } _ { t } ( v ) ( x ) = \int _ { D _ { t - 1 } } \kappa _ { t } ( x , d y ) v ( y ) ,
$$

with $\boldsymbol { \kappa } _ { t } \colon D _ { t } \times \mathfrak { B } ( D _ { t - 1 } ) \to \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } } \mathrm { ~ \mathbf { a } ~ }$ (transition) kernel from $D _ { t - 1 } \tan D _ { t }$ , meaning a function which is measurable with respect to the first variable and a finite measure with respect to the second variable.

$\cdot \mathrm { ~ } g _ { t } \in \Gamma _ { 0 } ( \mathcal { V } _ { t } )$ and ran $\begin{array} { r } { \mathbb { \ O } ( \partial \Phi _ { t } + \partial g _ { t } ) = \mathcal V _ { t } ^ { * } } \end{array}$

Equation (6) is highly general, featuring an outer operation (the $\mathrm { p r o x } _ { g _ { t } } ^ { \Phi _ { t } } )$ and an inner operation (the $\tilde { \nabla } \Phi _ { t } )$ , and can formally represent various layer architectures sketched in Figure 3. A key step in establishing this connection involves relating the proximity operator to activation operators. There are multiple ways to achieve this by varying the choice of the pair $\textstyle ( \Phi _ { t } , g _ { t } )$ . In the following sections, we explore two specific choices for this pair, demonstrating how (6) recovers classical neural operators (3) (where $\tilde { \nabla } \Phi _ { t }$ is the identity) and introduces a novel architecture (4), in which $\tilde { \nabla } \Phi _ { t }$ acts as the inverse activation operator.

Remark 3.1 (Form of linear operator $\boldsymbol { \kappa } _ { t } )$ . Often in applications, the kernel of the linear operator $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ is split into two terms: an absolutely continuous part and a single pure point part, i.e., $\kappa _ { t } = \kappa _ { t } ^ { \mathrm { a c } } + \kappa _ { t } ^ { p }$ , where, for every $x \in D _ { t }$ , and measurable set $A \subset D _ { t - 1 }$

$$
\kappa _ { t } ^ { \mathrm { a c } } ( x , A ) = \int _ { A } k _ { t } ( x , y ) d y \quad \mathrm { a n d } \quad \kappa _ { t } ^ { p } ( A ) = K _ { t } \delta _ { \varphi _ { t } ( x ) } ( A )
$$

with $k _ { t } \colon D _ { t } \times D _ { t - 1 } \ \to \ \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } } , \ K _ { t } \ \in \ \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } }$ $\varphi _ { t } \colon D _ { t } \to D _ { t - 1 }$ measurable, and $\delta _ { \varphi _ { t } ( x ) }$ the delta Dirac at $\varphi _ { t } ( x ) \in D _ { t - 1 }$ . Thus, we have

$$
\begin{array} { l } { \displaystyle \mathcal { K } _ { t } ( v ) ( x ) = \mathcal { K } _ { t } ^ { \mathrm { a c } } ( v ) ( x ) + \mathcal { K } _ { t } ^ { \mathrm { p } } ( v ) ( x ) } \\ { \displaystyle \qquad = \int _ { D _ { t - 1 } } k _ { t } ( x , y ) v ( y ) d y + K _ { t } v ( \varphi _ { t } ( x ) ) . } \end{array}
$$

Remark 3.2 (Special case of identical domains). The linear operator $\mathcal { M } _ { t }$ should be chosen so that it maps dom $\partial \Phi _ { t - 1 }$ to dom $\partial \Phi _ { t }$ . However, in (6), if the function $\phi _ { t }$ does not depend on t and all the domains $D _ { t }$ are the same, then it is also true that the convex integral functional $\Phi _ { t }$ does not depend on t either. Then, we have dom $\partial \Phi _ { t - 1 } = \mathrm { d o m } \partial \Phi _ { t }$ and for the linear operator $\mathcal { M } _ { t }$ we are allowed to choose the identity operator.

Remark 3.3 (Link with convex optimization). When $\nu _ { t - 1 } =$ $\nu _ { t }$ and $\mathcal { M } _ { t }$ is the identity, the operator layer (6) reads

$$
\mathrm { p r o x } _ { g _ { t } } ^ { \Phi _ { t } } ( \tilde { \nabla } \Phi _ { t } ( v ) - \mathcal { B } _ { t } v ) = ( \partial \Phi _ { t } + \partial g _ { t } ) ^ { - 1 } ( \tilde { \nabla } \Phi _ { t } - \mathcal { B } _ { t } ) ( v ) ,
$$

where $\boldsymbol { B } _ { t } \colon \mathcal { V } _ { t } \to \mathcal { V } _ { t } ^ { * }$ . This is a Bregmanforward-backward operator, which is well-known in the context of operator splitting methods in optimization (Nguyen, 2017; Bùi & Combettes, 2021).

![](images/5fc9e551f6a5a7bf261dd5deb9697be69606b2079a591b1d61cb65f3ad1128df.jpg)  
Figure 3: Illustration of the Bregman proximal viewpoint on operator layers. The action of each operator layer is viewed as the minimizer of the regularized optimization problem where each term in the objective can be linked to a part of the architecture, as evidenced by the color code.

Concluding this section, we stress that as long as the couple $\textstyle ( \Phi _ { t } , g _ { t } )$ admits a closed form Bregman proximity operator, this would define additional new types of operator layers. In Nguyen (2017), the author shows a number of examples (at the end of Section 2, from Example 2.9 to Example 2.12) of such couples that yield an explicit Bregman proximity operator. Actually, one may consider layers of type

$$
v \mapsto \sigma _ { 2 } ( \sigma _ { 1 } ^ { - 1 } ( v ) + \mathcal { K } _ { t } ( v ) + b _ { t } ) ,
$$

with $\sigma _ { 1 }$ being strictly monotone and $\sigma _ { 2 }$ monotone, serving as activation operators appropriately coupled. Classical and Bregman neural operators emerge as special cases, where i) $\sigma _ { 1 } = \mathrm { I d }$ and $\sigma _ { 2 }$ is any monotone function, for the former, and ii) $\sigma _ { 1 } = \sigma _ { 2 }$ is strictly monotone, for the latter. Note that having $\sigma _ { 1 } = \sigma _ { 2 }$ implies that the numerical implementation does not require to have an explicit form of $\bar { \sigma } _ { 1 } ^ { - 1 }$ , as later discussed in Remark 3.7.

## 3.2. Classical Neural Operators

Our first result, stated in the proposition below, unifies a broad class of classical neural operator layers through the prism of the optimization viewpoint of (6) when $D _ { \Phi _ { t } }$ is the Euclidean distance.

Proposition 3.4 (Unifying classical neural operators). Let $\mathcal { V } _ { t } ~ = ~ L ^ { 2 } ( D _ { t } , \mathbb { R } ^ { n _ { t } } )$ be some Hilbert function space and $\begin{array} { r } { \Psi _ { t } ( v ) \ : = \ : \int _ { D _ { \star } } \sum _ { i = 1 } ^ { n _ { t } } \psi ( v _ { i } ( x ) ) d x , } \end{array}$ , where $\psi \in \Gamma _ { 0 } ( \mathbb { R } )$ is a strongly convex Legendrefunction. Consider the Euclidean distance defined from the elementary Legendre function $\phi _ { t } = ( 1 / 2 ) | \cdot | ^ { 2 } \in \Gamma _ { 0 } ( \mathbb { R } ^ { n _ { t } } )$ (see Section 2.3) and set $g _ { t } =$ $\Psi _ { t } - ( 1 / 2 ) \| \cdot \| ^ { 2 }$ . Then $g _ { t } \in \Gamma _ { 0 } ( \mathcal { V } _ { t } )$ and $\mathcal { L } _ { t }$ defined in (6) acts between $L ^ { 2 }$ spaces as follows

$$
\begin{array} { r l r } {  { \mathcal { L } _ { t } ( v ) = \mathrm { p r o x } _ { \Psi _ { t } - \frac { 1 } { 2 } \| \cdot \| ^ { 2 } } ^ { \frac { 1 } { 2 } \| \cdot \| ^ { 2 } } \big ( \mathcal { M } _ { t } v + \mathcal { K } _ { t } ( v ) + b _ { t } \big ) } } \\ & { } & { = \nabla \Psi _ { t } ^ { * } ( \mathcal { M } _ { t } v + \mathcal { K } _ { t } ( v ) + b _ { t } ) , \quad \quad } \end{array}\tag{7}
$$

Table 1: Legendre function $\psi$ and its related activation $\psi ^ { * } { } ^ { \prime } .$
<table><tr><td rowspan=1 colspan=1>domψ</td><td rowspan=1 colspan=1>ψ(t)</td><td rowspan=1 colspan=1> $\overline { { \psi ^ { \prime } ( t ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \psi ^ { * } ( t ) } }$ </td></tr><tr><td rowspan=1 colspan=1>[−1,1]</td><td rowspan=1 colspan=1> $\overline { { - \sqrt { 1 - t ^ { 2 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { t / \sqrt { 1 - t ^ { 2 } } } }$ </td><td rowspan=1 colspan=1>ISRU</td></tr><tr><td rowspan=1 colspan=1>[0, 1]</td><td rowspan=1 colspan=1> $\overline { { t \log t + ( 1 - t ) \log ( 1 - t ) } }$ </td><td rowspan=1 colspan=1>log $\frac { t } { \underline { { 1 - t } } }$ </td><td rowspan=1 colspan=1>Sigmoid</td></tr><tr><td rowspan=1 colspan=1>[−1,1]</td><td rowspan=1 colspan=1> $\overline { { \log ( 1 - t ^ { 2 } ) + t \operatorname { a r c t a n h } ( t ) } }$ </td><td rowspan=1 colspan=1>arctanh</td><td rowspan=1 colspan=1>tanh</td></tr><tr><td rowspan=1 colspan=1>[−1,1]</td><td rowspan=1 colspan=1> $\overline { { { \sqrt { 1 - t ^ { 2 } } } + t \arcsin ( t ) } }$ </td><td rowspan=1 colspan=1>arcsin</td><td rowspan=1 colspan=1>sin</td></tr><tr><td rowspan=1 colspan=1> $\mathbb { R } _ { > 0 }$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { \frac { 1 } { \beta ^ { 2 } } \mathrm { L i } _ { 2 } ( \mathrm { e } ^ { - \beta t } ) + \frac { t ^ { 2 } } { 2 } } \end{array}$ </td><td rowspan=1 colspan=1> $\frac { \log ( \mathrm { e } ^ { \beta t } - 1 ) } { \beta }$ </td><td rowspan=1 colspan=1>SoftPlusβ</td></tr></table>

where $\nabla \Psi _ { t } ^ { * } = ( \psi ^ { * } ) ^ { \prime } ( \cdot )$ matches a variety of monotone activation operators σ. In addition, when the domains are all the same, say $D _ { t } = D , \mathcal { M } _ { t } = I ,$ , and the linear operator ${ \ K } _ { t } = { \ K } _ { t } ^ { \mathrm { a c } } + { \ K } _ { t } ^ { \mathrm { p } }$ is as given in Remark $3 . I ,$ then $\mathcal { L } _ { t } ( v ) =$ $\nabla \Psi _ { t } ^ { * } \big ( ( I + K _ { t } ) v + K _ { t } ^ { \mathrm { a c } } ( v ) + b _ { t } \big )$ , where $( I + K _ { t } )$ can be written as $W _ { t } .$ . A schematic representation is reported in Figure 2a.

In essence, Proposition 3.4 shows that the parametric structure of operator layers can be interpreted via the Bregman proximal operator, when the Bregman distance reduces to the Euclidean distance. The crucial aspect in establishing this connection is the observation that the Euclidean proximity operator of $g _ { t } = \Psi - ( 1 / 2 ) \| \cdot \| ^ { 2 }$ simplifies to $\nabla \Psi ^ { * } = ( \psi ^ { * } ) ^ { \prime } ( \mathbf { \cdot } )$ , aligning with a broad spectrum of activation operators given an appropriate selection of $\psi .$ . We report in Table 1 the corresponding $\psi$ to retrieve several well-known activation operators. A proof concerning the characterization of the SoftPlus is included in the appendix. To the best of our knowledge, $\nabla \Psi _ { t } ^ { * }$ can only match monotonic activation operators, which notably discards GeLu and swish. To be more precise, Proposition 3.4 is general enough to deal with the broad class of activation functions that can be viewed as a proximity operators, which essentially boils down to any increasing 1-Lipschitzian function (see Proposition 2.3 in Combettes & Pesquet (2020a)). While this connection has been previously noted in the neural network literature (Combettes & Pesquet, 2020a; Frecon et al., 2022), our work extends this analysis to function spaces.

## 3.3. Bregman Neural Operators

We now provide the counterpart of Proposition 3.4 for general Bregman distance.

Proposition 3.5 (Designing Bregman neural operators). Let $\mathcal { V } _ { t } = L ^ { p } ( D _ { t } , \mathbb { R } ^ { n _ { t } } )$ be some Lebesgue function space and $\begin{array} { r } { \Psi _ { t } ( v ) = \int _ { D _ { t } } \sum _ { i = 1 } ^ { n _ { t } } \psi ( v _ { i } ( x ) ) d x } \end{array}$ , where $\psi \in \Gamma _ { 0 } ( \mathbb { R } )$ is a $p \textmd { - }$ uniformly convex Legendrefunction $( \neq | \cdot | ^ { 2 } / 2 )$ . Consider the Bregman distance in function space defined from the elementary Legendre function $\begin{array} { r } { \phi _ { t } ( w ) = \sum _ { i = 1 } ^ { n _ { t } } \psi ( w _ { i } ) } \end{array}$ (see Section 2.3) and set $g _ { t } = 0$ . Then ${ \mathcal { L } } _ { p }$ defined in (6) acts between L<sup>p</sup> spaces asfollows

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { t } ( v ) = \mathrm { p r o x } _ { 0 } ^ { \Psi _ { t } } \big ( \tilde { \nabla } \Psi _ { t } ( \mathcal { M } _ { t } v ) + \mathcal { K } _ { t } ( v ) + b _ { t } \big ) } \\ & { } & { \quad = \nabla \Psi _ { t } ^ { * } ( \tilde { \nabla } \Psi _ { t } ( \mathcal { M } _ { t } v ) + \mathcal { K } _ { t } ( v ) + b _ { t } ) , \quad } \end{array}\tag{8}
$$

where $\nabla \Psi _ { t } ^ { * } = ( \psi ^ { * } ) ^ { \prime } ( \cdot )$ matches a variety of monotone activation operators $\sigma .$ . In addition, when the domains are all the same, say $D _ { t } = D$ and the linear operator $\textstyle { \mathcal { K } } _ { t }$ is of the form given in Remark $3 . I ,$ , then we can take $\boldsymbol { \mathcal { M } } _ { t } = \boldsymbol { I }$ and

$$
\mathcal { L } _ { t } ( v ) = \nabla \Psi _ { t } ^ { * } ( \tilde { \nabla } \Psi _ { t } ( v ) + K _ { t } v + K _ { t } ^ { \mathrm { a c } } ( v ) + b _ { t } ) .\tag{9}
$$

Concerning the operators $\nabla \Psi _ { t } ^ { * } = ( \psi ^ { * } ) ^ { \prime } ( \cdot )$ and $\nabla \Psi _ { t } ^ { * } =$ $\psi ^ { \prime } ( \cdot )$ , we stress that any of the ψ listed in Table 1 are appropriate choices. Since $( \psi ^ { * } ) ^ { \prime } ( \cdot )$ and $\psi ^ { \prime } ( \cdot )$ are inverse of each other, the layer of (9) boils down to

$$
\mathcal { L } _ { t } ( v ) = \sigma ( \sigma ^ { - 1 } ( v _ { t } ) + K _ { t } v + K _ { t } ^ { \mathrm { a c } } ( v ) + b _ { t } ) ,\tag{10}
$$

where any invertible and monotone activation operator is allowed. Its schematic representation is reported in Figure 2b. This novel variant, called Bregman Neural Operator simply differs from classical neural operators by the additional term involving the inverse activation operator. Finally, we note that the form of (9) corresponds to a mirror descent step (Nemirovskij & Yudin, 1983; Beck & Teboulle, 2003) with mirror map $\tilde { \nabla } \Psi _ { t }$

Remark 3.6. When $K _ { t } , K _ { t } ^ { \mathrm { a c } }$ and $b _ { t }$ are zeros and $\mathcal { M } _ { t }$ is the identity, then $\mathcal { L } _ { t }$ reduces to the identity.

Remark 3.7. Concerning (10), we should ensure to feed the first layer with functions in dom $\mathcal { L } _ { 1 }$ as discussed in Remark A.5. This condition is for instance satisfied if $( \mathcal { P } v ) ( v ) = \nabla \psi _ { 1 } ^ { * } ( P v ( x ) ) = \sigma ( P v ( x ) )$ . Note that in such situation, the inverse activation function does not need to have an explicit form. Indeed, when composing the different layers in (10), the inner inverse activation function will be cancelled out by the outer one.

## 3.4. Case of Fourier Neural Operators

We study the implications of the proposed viewpoint in the peculiar case of Hilbert function spaces with equal input and output spaces, i.e., $\mathcal { V } _ { t } = \mathcal { V } _ { t } ^ { \ast } = L ^ { 2 } ( D , \mathbb { R } ^ { n } )$ for every $t \in \{ 1 , \ldots , T \}$

A popularly encountered scenario in practice is that where $D = \mathbb { T } ^ { d }$ is the unit torus and the kernel associated to the absolutely continuous part of $\textstyle { \mathcal { K } } _ { t }$ is translation invariant, i.e., $k _ { t } ( x , y ) = k _ { t } ( x - y )$ , thus indicating a convolution structure. Fourier operator layers (Li et al., 2021a) are then devised by leveraging the convolution theorem, stating that the action of ${ \mathcal { K } } _ { t } ^ { \mathrm { a c } }$ can be written as a linear operator in the Fourier domain:

$$
{ \mathcal { K } } _ { t } ^ { \mathrm { a c } } ( v ) ( x ) = \int _ { D } k _ { t } ( x - y ) v ( y ) d y = { \mathcal { F } } ^ { - 1 } ( R _ { t } \cdot { \mathcal { F } } ( v ) ) ( x ) ,
$$

with $\mathcal { F } \colon L ^ { 2 } (  { \mathbb { T } } ^ { d } ,  { \mathbb { R } } ^ { n } ) \to \ell ^ { 2 } (  { \mathbb { Z } } ^ { d } ,  { \mathbb { R } } ^ { n } )$ being the Fourier transform, ${ \mathcal { F } } ^ { - 1 }$ its inverse, and $R _ { t } \ \in \ \ell ^ { 2 } ( \mathbb { Z } ^ { 2 } , \mathbb { R } ^ { n \times n } )$ . Often, $R _ { t }$ does not range in the entire $\ell ^ { 2 } ( \mathbb { Z } ^ { 2 } , \dot { \mathbb { R } } ^ { n \times n } )$ space but is parametrized by a finite parameter (Kovachki et al., 2023). It follows that the Bregman variant of Fourier operator layer reads $\mathcal { L } _ { t } ( v ) = \sigma ( \sigma ^ { - 1 } ( v ) + W _ { t } v + \mathcal { F } ^ { - 1 } ( R _ { t } \cdot \mathcal { F } ( v ) ) + b _ { t } )$ The classical Fourier neural operator layer is retrieved by omitting the $\sigma ^ { - 1 } ( v )$ term.

In this section, we addressed FNOs because they are widely used and simplify the analysis. In this respect, we note that we just specified the action of ${ \cal K } _ { t } ^ { \mathrm { a c } }$ by expressing it via direct and inverse Fourier series. So, in the end, it is only about finding efficient parametrizations, in some $\ell ^ { p \ }$ space, of linear integral operators between Lebesgue spaces. This has been achieved by using the Fourier transform, but in principle other transformations could be considered, provided we have an unconditional basis of the Lebesgue space of functions and an efficient way to compute the coefficients. For instance, the wavelet transform can be incorporated in Proposition 1 and Proposition 2 to retrieve WNOs (Tripura $\&$ Chakraborty, 2023) and their novel Bregman variant, respectively. In a nutshell, our framework is transparent to the parametrization of ${ \cal K } _ { t } ^ { \mathrm { a c } }$

## 4. Expressivity of Bregman neural operators

In this section, we give a preliminary positive result concerning the universal approximation properties of Bregman neural operators.

In the following, the activation function $\sigma \colon \mathbb { R }  I$ is required to be a homeomorphism between R and an open interval I of R and of sigmoidal type, meaning that lim $\mathbf { \sigma } _ { 1 t  - \infty } \sigma ( t ) = 0$ and $\begin{array} { r } { \operatorname* { l i m } _ { t \to + \infty } \sigma ( t ) = 1 } \end{array}$ . Moreover, we assume that A and U are as follows

$$
\begin{array} { r } { A ( D , \mathbb { R } ^ { n } ) = \left\{ \begin{array} { l l } { \mathcal { C } ( \overline { { D } } , \mathbb { R } ^ { n } ) } \\ { L ^ { p } ( D , \mathbb { R } ^ { n } ) \quad \hfill , \mathcal { U } ( D , \mathbb { R } ^ { k } ) = \left\{ \begin{array} { l l } { \mathcal { C } ( \overline { { D } } , \mathbb { R } ^ { k } ) } \\ { L ^ { p } ( D , \mathbb { R } ^ { k } ) } \end{array} \right. , } \end{array} \right. } \end{array}
$$

where C is the space of continuous functions and $W ^ { m , p }$ is the $L ^ { p }$ -type Sobolev space with $m \in \mathbb { N } _ { + }$ derivatives for $p \in [ 1 , + \infty [ .$ . Here, D denotes the closure of $D ,$ and must be considered in PDE applications to evaluate functions on the domain’s boundary.

![](images/d476498411da2c550e7945bb83a26274f597854935a031116dc7c709103341a4.jpg)  
Figure 4: $\ell ^ { 2 }$ relative errors across different number of layers for 2D Navier Stokes $( \nu = 1 0 ^ { - 4 } )$ ).

Theorem 4.1. Let σ, A and U be set as above. Let $\mathcal { G } \colon A $ U be a continuous operator. Then for any compact set $K \subset \mathcal { A } a n d \varepsilon > 0$ there exists a Bregman neural operator $\mathcal { N } _ { \theta } \colon \mathcal { A }  \mathcal { U }$ of the type (2) such that each component depends on a finite dimensional Bregman neural network and

$$
\operatorname* { s u p } _ { u \in K } \| \mathcal { G } ( u ) - \mathcal { N } _ { \theta } ( u ) \| _ { \mathcal { U } } \leq \varepsilon .
$$

Here $\theta \in \mathbb { R } ^ { p }$ collects all the (finite number of) parameters of the finite dimensional Bregman neural networks defining the components in (2).

Proof. The proof is reported in Appendix B and partly relies on also proving this same result for Bregman neural networks in finite dimensional spaces. □

## 5. Numerical Experiments

The primary objective of our numerical experiments is to evaluate and assess the added benefits of the Bregman variant of the simplest neural operator, namely Fourier Neural Operator (FNO), and its improvements, as they often serve as the building blocks for more sophisticated models.

## 5.1. Experimental Setting

Datasets. We have selected a range of benchmark datasets resulting from the resolution of PDEs used both in the original FNO paper (Li et al., 2021a) and in the PDEBench suite (Takamoto et al., 2022), which is the top leading repository providing datasets commonly studied in physics-based machine learning. They represent various dynamics and complexities pertinent to physical modeling tasks. Hereafter, we consider initial value problems where the goal is to learn the mapping between the initial condition $a _ { i }$ and the solution at some future time $u _ { i }$ from $n = 1 0 ^ { 4 }$ pairs $\{ a _ { i } , u _ { i } \} _ { i = 1 } ^ { n }$ . The only exception is the 2D Darcy problem (marked with \* in latter results), where the goal is to predict the steady-state solution from the viscosity function over the domain. A description of the experimental settings and the learning procedure is provided in Appendix C.

![](images/46cb60e2de2e2c932a50a3d5faa14692b0e0ab7345396958bff1258c8c5873e2.jpg)  
Figure 5: Models’ weight density distribution.

Models. We consider four models: the standard FNO (Li et al., 2021a), our Bregman variant (BFNO) described in Section 3.4, the Factorized FNO (F-FNO) (Tran et al., 2023), and a ResNet-inspired variant (ResFNO) that isolates the impact of residual connections by adopting the update $v \mapsto v + \sigma ( \mathcal { K } _ { t } ( v ) + b _ { t } )$ , which should not be confused with Chen et al. (2021). Details can be found in Appendix C.6. Additional models such as WNO (Tripura & Chakraborty, 2023) and its Bregman variant are studied in the appendix. The lifting and projection layers, namely $\mathcal { P }$ and $\mathcal { Q }$ in (2), are convolutional layers with kernel size 1 and width 128. Note that, for BFNO, we add an activation operator after $\mathcal { P }$ to ensure that the conditions of Remark 3.7 are met. Following the code of Li et al. (2021a), we use the ReLU activation for FNO while, for BFNO, we resort to an invertible approximation: SoftPlus with parameter $\beta = 1 0 ^ { 3 }$ to make it almost indistinguishible from ReLU. Hereafter, we consider models made of $T \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ Fourier layers with a width 64 (resp. 32) and 16 (resp. 12) maximum number of Fourier modes for 1D (resp. 2D) problems. Note that two ablation studies in Appendices D.4 and D.5 reveal marginal improvements from adding batch normalization layers or replacing SoftPlus with ReLU.

## 5.2. Results and Analysis

Impact of the number of layers T. We investigate how increasing the number of operator layers $T$ affects performance using the 2D Navier-Stokes dataset with viscosity $\nu = 1 0 ^ { - 4 }$ , chosen for its complexity, that typically favor deeper models to capture fine-grained structures and longrange dependencies. Results in Figure 4 show that BFNO systematically achieves lower prediction error regardless of $T .$ In contrast, FNO (resp. ResFNO) degrades from $T = 8$ (resp. $T = 1 6 )$ onward, while F-FNO (resp. BFNO) improves with depth until plateauing at $T = 1 6 \ : ( \mathrm { r e s p . } T = 6 4 )$ The earlier saturation of F-FNO suggests that while residual connections help with depth, BFNO scales more effectively. Similar conclusions hold for other datasets and our Bregman variant of WNO, as illustrated in Appendix D.2. Nevertheless for simpler tasks, when smaller models are sufficient, the benefits of depth become less pronounced as expected. We believe that the added term in BFNO stabilizes learning by allowing its layers to reduce to identity when all weights are zero (see Remark 3.6). However, the same argument could be used for residual architectures (ResFNO and F-FNO). To gain further insights, we next examine the weight distributions.

![](images/33eb30d52f02385846d77e5cef73e6b96b6954d155ccfb4e89262d11b74a8e6e.jpg)  
Figure 6: Comparison of $\ell ^ { 2 }$ test relative errors across different models, with cross-validated number of layers in {4, 8, 16, 32} and learning rates, on PDE benchmarks.

Comparison of weight density distribution. Figure 5 illustrates the learned weight distributions of the operator layers for the best-performing models considered above. ResFNO, F-FNO, and FNO exhibit Gaussian-like distributions, with F-FNO showing an additional peak at 0. In contrast, BFNO has a sharply peaked distribution around 0, resembling a Laplace distribution. While BFNO, ResFNO, and F-FNO layers reduce to the identity when all weights are zero, only BFNO exhibits a distinct clustering of weights around zero. This suggests implicit regularization, enhancing generalization and stability by preventing large deviations from the identity mapping. Additionally, the weight distribution in BFNO alleviates issues such as vanishing or exploding gradients, which are common in deeper architectures.

Extensive comparison on multiple datasets. We now evaluate the models on diverse datasets of varying complexity. For each dataset, splitting realization, and model, we crossvalidate the optimal number of layers $T \in \{ 4 , 8 , 1 6 , 3 2 \}$ and report the average test error over multiple realizations in Figure 6. Results show that BFNO consistently achieves superior or comparable performance, with notable gains on moderately to highly complex datasets such as 1D Burgers and 2D Navier-Stokes $( \nu = 1 0 ^ { - 4 } )$ . The only exception is that of 2D Darcy, where the task differs: instead of learning a solution map from initial conditions, the goal is to map the spatial viscosity function to the steady-state solution. In this case, it seems that the additional MLP layers added at the end of the operator layers of F-FNO help to better capture the complex dependencies of the viscosity-tosolution mapping. To complement the analysis, following Takamoto et al. (2022), we include a comparison between BFNO and FNO over several metrics measuring the relative errors in low, mid and high frequency bands in $\mathsf { A p - }$ pendix D.3. BFNO consistently outperforms FNO in the low and mid-frequency bands, indicating improved reconstruction of dominant modes. However, its performance in the high-frequency range varies: for simpler datasets, BFNO achieves substantial error reductions, whereas for more complex cases, the improvements are marginal. Prediction examples are reported in Appendix D.1.

Additional insights. We highlight that BFNO can also be viewed from the ODE point of view through a change of variables. More precisely, let us consider updates of the form $v _ { t + 1 } = \sigma ( \sigma ^ { - 1 } ( v _ { t } ) + \mathcal { K } _ { t } v _ { t } )$ . Then, for $z _ { t } = \sigma ^ { - 1 } ( v _ { t } )$ it follows that $z _ { t + 1 } = z _ { t } + \mathcal { K } _ { t } \sigma ( z _ { t } )$ which can be seen as a discretization of $\begin{array} { r } { \frac { \mathrm { d } z ( t ) } { \mathrm { d } t } = \mathcal { K } ( t ) \sigma ( z ( t ) ) } \end{array}$ . In contrast, a residual-based architecture of the form $v _ { t + 1 } = v _ { t } + \sigma ( \mathcal { K } _ { t } v _ { t } )$ such as ResFNO and, to some extent also F-FNO, would lead to the following ODE $\begin{array} { r } { \frac { \mathrm { d } v ( t ) } { \mathrm { d } t } = \sigma ( \mathcal { K } ( t ) v ( t ) ) } \end{array}$ on v itself. The fact that, in BFNO, the linear operator is placed outside the activation function can be seen as a different way of mitigating vanishing gradients compared to residual architectures like ResFNO and F-FNO.

## 6. Conclusion

In summary, our contributions are twofold: we have provided a new theoretical framework that broadens the understanding of neural operators through the lens of a Bregman regularized optimization problem, and we have introduced Bregman neural operators that achieve enhanced performance as their depth increases. As part of our theoretical advancements, we have also established universal approximation results for Bregman neural architectures with sigmoidal-type activation functions. However, it must be acknowledged that a gap exists between this result and common practices, which predominantly rely on ReLU-like activations, as in our work, opening the door to new theoretical developments. Beyond the unifying aspect of our framework and its ability to design novel neural architectures, our setting also paves the way to use the rich body of literature on monotone operators to study neural operators. In the context of neural networks, an example of fruitful application of the latter is given in Combettes & Pesquet (2020a) where the authors provide interesting asymptotic properties on the networks (as the number of layers tends to infinity). One can also consider the work in Combettes & Pesquet (2020b)

where the authors yield quantitative insights into the stability properties of neural networks. As for our setting, we can guess that such results might be extended to Bregman neural networks/operators by leveraging the notion of so called D-firm operators studied in Bauschke et al. (2003), meaning operators that are firmly nonexpansive with respect to a Bregman divergence.

## References

Anandkumar, A., Azizzadenesheli, K., Bhattacharya, K., Kovachki, N., Li, Z., Liu, B., and Stuart, A. Neural operator: Graph kernel network for partial differential equations. In ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations, 2020.

Bauschke, H. H. and Combettes, P. L. Convex Analysis and Monotone Operator Theory in Hilbert Spaces. Springer Publishing Company, Incorporated, 2nd edition, 2017. ISBN 978-3-319-48310-8.

Bauschke, H. H., Borwein, J. M., and Combettes, P. L. Bregman monotone optimization algorithms. SIAM Journal on control and optimization, 42(2):596–636, 2003.

Beck, A. and Teboulle, M. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters, 31(3):167–175, 2003.

Bregman, L. M. The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming. USSR computational mathematics and mathematical physics, 7(3): 200–217, 1967.

Brezis, H. Functional Analysis, Sobolev Spaces, and Partial Differential Equations. Springer, New York, 2011.

Bùi, M. N. and Combettes, P. L. Bregman forward.backward operator splitting. Set-Valued and Variational Analysis, 29:583–603, 2021.

Chen, G., Li, Y., Meng, Q., Zhou, J., Hao, X., et al. Residual fourier neural operator for thermochemical curing of composites. arXiv preprint arXiv:2111.10262, 2021.

Combettes, P. L. and Pesquet, J.-C. Deep neural network structures solving variational inequalities. Set-Valued and Variational Analysis, 28(3):491–518, February 2020a. ISSN 1877-0541.

Combettes, P. L. and Pesquet, J.-C. Lipschitz certificates for layered network structures driven by averaged activation operators. SIAM Journal on Mathematics of Data Science, 2(2):529–557, 2020b.

Cybenko, G. Approximation by superposition of sigmoidal function. Mathematics of Control, Signals, and Systems, 2:303–314, 1989.

Frecon, J., Gasso, G., Pontil, M., and Salzo, S. Bregman neural networks. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceed ings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pp. 6779–6792. PMLR, 17–23 Jul 2022.

Goswami, S., Yin, M., Yu, Y., and Karniadakis, G. E. A physics-informed variational DeepONet for predicting crack path in quasi-brittle materials. Computer Methods in Applied Mechanics and Engineering, 391:114587, mar 2022.

Hao, Z., Wang, Z., Su, H., Ying, C., Dong, Y., Liu, S., Cheng, Z., Song, J., and Zhu, J. GNOT: A general neural operator transformer for operator learning. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 12556–12569. PMLR, 23–29 Jul 2023.

Helwig, J., Zhang, X., Fu, C., Kurtin, J., Wojtowytsch, S., and Ji, S. Group equivariant fourier neural operators for partial differential equations. In International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Kissas, G., Seidman, J. H., Guilhoto, L. F., Preciado, V. M., Pappas, G. J., and Perdikaris, P. Learning operators with coupled attention. Journal of Machine Learning Research, 23(215):1–63, 2022.

Kovachki, N., Lanthaler, S., and Mishra, S. On universal approximation and error bounds for Fourier neural operators. Journal of Machine Learning Research, 22(290): 1–76, 2021.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar, A. Neural operator: Learning maps between function spaces with applications to pdes. Journal ofMachine Learning Research, 24(89):1–97, 2023.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Stuart, A., Bhattacharya, K., and Anandkumar, A. Multipole graph neural operator for parametric partial differential equations. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 6755–6766. Curran Associates, Inc., 2020.

Li, Z., Kovachki, N. B., Azizzadenesheli, K., liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2021a.

Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K., and Anandkumar, A. Physicsinformed neural operator for learning partial differential equations. arXiv preprint arXiv:2111.03794, 2021b.

Lu, L., Jin, P., Pang, G., Zhang, Z., and Karniadakis, G. E. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, mar 2021.

Molinaro, R., Yang, Y., Engquist, B., and Mishra, S. Neural inverse operators for solving pde inverse problems. In International Conference on Machine Learning, 2023.

Nemirovskij, A. S. and Yudin, D. B. Problem complexity and method efficiency in optimization. 1983.

Nguyen, Q. Forward-backward splitting with bregman distances. Vietnam J. Math,, 45:519–539, 2017.

Rahman, M. A., Ross, Z. E., and Azizzadenesheli, K. U-NO: U-shaped neural operators. Transactions on Machine Learning Research, 2023. ISSN 2835-8856.

Raonic, B., Molinaro, R., De Ryck, T., Rohner, T., Bartolucci, F., Alaifari, R., Mishra, S., and de Bézenac, E. Convolutional neural operators for robust and accurate learning of pdes. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 77187–77200. Curran Associates, Inc., 2023.

Rockafellar, T. Convex Analysis. Princeton University Press, Princeton, NJ, 1970.

Serrano, L., Le Boudec, L., Kassaï Koupaï, A., Wang, T. X., Yin, Y., Vittaut, J.-N., and Gallinari, P. Operator learning with neural fields: Tackling pdes on general geometries. In Advances in Neural Information Processing Systems, volume 36, pp. 70581–70611, 2023.

Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., and Niepert, M. Pdebench: An extensive benchmark for scientific machine learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Tran, A., Mathews, A., Xie, L., and Ong, C. S. Factorized fourier neural operators. In The Eleventh International Conference on Learning Representations, 2023.

Tripura, T. and Chakraborty, S. Wavelet neural operator for solving parametric partial differential equations in computational mechanics problems. Computer Methods in Applied Mechanics and Engineering, 404:115783, feb 2023.

Wang, S. and Perdikaris, P. Long-time integration of parametric evolution equations with physics-informed Deep-ONets. Journal ofComputational Physics, 475:111855, feb 2023.

Wang, S., Wang, H., and Perdikaris, P. Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. Science Advances, 7(40), oct 2021.

Wang, S., Wang, H., and Perdikaris, P. Improved architectures and training algorithms for deep operator networks. Journal ofScientific Computing, 92(2), jun 2022.

Zalinescu, C. Convex Analysis in General Vector Spaces. World Scientific, Singapore, 2002.

## A. Additional Technical Facts

We begin by introducing the necessary notations used throughout the paper.

Notations. Let V and $\nu ^ { * }$ be two Banach spaces put in duality via the pairing $\langle \cdot , \cdot \rangle \colon \mathcal { V } \times \mathcal { V } ^ { * } \to \mathbb { R } . \mathrm { I f } \ \Phi \colon \mathcal { V } \to \mathrm { ] } - \infty , + \infty \mathrm { I }$ we denote by dom $\Phi = \{ v \in \mathcal { V } | \Phi ( v ) < + \infty \}$ its effective domain. For every proper convex function Φ : $\mathcal { V } \to ] - \infty , + \infty ]$ we set its subdifferentia

$$
\partial \Phi ( v ) = \{ v ^ { * } \in \gamma ^ { * } | \mathrm { f o r ~ a l l } u \in \mathcal { V } , \Phi ( u ) \geq \Phi ( v ) + \langle u - v , v ^ { * } \rangle \} ,
$$

if $v \in$ dom Φ, and $\partial \Phi ( v ) = \emptyset$ , otherwise. We set dom $\partial \Phi = \{ v \in \mathrm { d o m } \Phi | \partial \Phi ( v ) \neq \emptyset \}$ and the range ran $\partial \Phi = \{ v ^ { * } \in$ $\mathcal { V } ^ { * } | \exists v \in \mathcal { V } \mathrm { s . t . } v ^ { * } \in \partial \Phi ( v ) \}$ . When $\partial \Phi ( v )$ is a singleton, we denote by ∇<sup>˜</sup> Φ its unique element. If Φ : $\mathcal { V } \to \left| - \infty , + \infty \right|$ its Fenchel conjugate is the function $\Phi ^ { * } \colon \mathcal { V } ^ { * } \to \left] - \infty , + \infty \right]$ such that $\begin{array} { r } { \Phi ^ { * } ( v ^ { * } ) = \operatorname* { s u p } _ { v \in \mathcal { V } } \langle v , v ^ { * } \rangle - \Phi ( v ) } \end{array}$ . We denote by $\Gamma _ { 0 } ( \nu )$ the set of proper convex and lower-semicontinuous functions on V. The Fenchel-Moreau theorem ensures that $\Phi \in \Gamma _ { 0 } ( \mathcal { V } ) \Rightarrow \Phi ^ { * } \in \Gamma _ { 0 } ( \mathcal { V } ^ { * } )$ . We denote by $\langle \cdot , \cdot \rangle$ and | · | the Euclidean scalar product and norm in $\mathbb { R } ^ { n }$ . If $D \subset \mathbb { R } ^ { d }$ is a nonempty bounded Borel set and $p \in [ 1 , + \infty ]$ , we denote by $L ^ { p } ( D , \mathbb { R } ^ { n } )$ the Lebesgue space of $p \cdot$ integrable functions (essentially bounded functions, if $p = + \infty )$ from $D \tan \mathbb { \mathbb { R } } ^ { n }$

## A.1. Considerations for Legendre Function and Bregman Proximal Operators

At the core of our framework, lies the connection between activation operators and Bregman proximity operators whose definition involves the Bregman divergence itself defined from a Legendre function $\Phi \in \Gamma _ { 0 } ( \mathcal { V } )$ . The latter acts on Lebesgue function space $\mathcal { V } = L ^ { P } ( D , \mathbb { R } ^ { n } )$ and can be built from an elementary legendre function $\phi \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ through the convex integral functional described in Fact 1. We provide below several considerations.

Remark A.1. One can prove that $\phi$ is Legendre if and only if $\phi ^ { * }$ is Legendre. Moreover, if ϕ is Legendre, then $\phi$ and $\phi ^ { * }$ are differentiable on int(dom ϕ) and int(dom $\phi ^ { * } )$ respectively and

$$
\nabla \phi \colon \operatorname { i n t } ( \operatorname { d o m } \phi )  \operatorname { i n t } ( \operatorname { d o m } \phi ^ { * } ) \quad \mathrm { a n d } \quad \nabla \phi ^ { * } \colon \operatorname { i n t } ( \operatorname { d o m } \phi ^ { * } )  \operatorname { i n t } ( \operatorname { d o m } \phi )
$$

are bijective and inverse of each other.

Fact 2 (Convex integral functionals on Lebesgue spaces based on Legendre function). Let $D \subset \mathbb { R } ^ { d }$ be an open-bounded set. Let $p , q \in [ 1 , + \infty ]$ be conjugate exponents, that is such that $1 / p + 1 / q = 1$ , and set $\mathcal { V } : = L ^ { p } ( D , \mathbb { R } ^ { n } )$ and $\mathcal { V } ^ { * } = L ^ { q } ( D , \mathbb { R } ^ { n } )$ The spaces $\nu$ and $\nu ^ { * }$ can put in duality via the pairing $\begin{array} { r } { \mathcal { V } \times \mathcal { V } ^ { * }  \mathbb { R } , \quad ( v , u ) \mapsto \langle v , u \rangle = \int _ { D } \langle v ( x ) , u ( x ) \rangle d x } \end{array}$ . Let $\phi \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ be a Legendre function and let $\Phi \colon \mathcal { V }  ] - \infty , + \infty ]$ be such that

$$
\Phi ( v ) = \int _ { D } \phi ( v ( x ) ) d x .\tag{11}
$$

Then $\Phi \in \Gamma _ { 0 } ( \mathcal { V } )$ , dom $\partial \Phi = \{ v \in \mathcal { V } |$ for ${ \mathrm { a . e . ~ } } x \in D , v ( x ) \in { \mathrm { i n t } } ( { \mathrm { d o m } } \phi )$ and $( \nabla \phi ) \circ v \in \mathcal { V } ^ { * } \}$ , ∂Φ is single valued on dom ∂Φ, and, for every $v \in$ dom $\partial \Phi , \partial \Phi ( v ) = \{ \nabla \phi \circ v \}$ . The unique element $\nabla \phi \circ \boldsymbol { \tau }$ of $\partial \Phi ( v )$ will be denoted by $\tilde { \nabla } \Phi ( v )$ suggesting it will serve as a kind of gradient of Φ at $v ^ { 1 }$

Remark A.2. In Fact 2, suppose that $p ~ = ~ 1$ and dom $\phi ^ { * } ~ = ~ \mathbb { R } ^ { n }$ Then ran $\partial \Phi ~ = ~ \mathcal { V } ^ { * }$ . Indeed, we note that $\nabla \phi \colon$ int(dom $\phi ) \ \to \ \mathbb { R } ^ { n }$ is a continuous bijection with inverse $\nabla \phi ^ { * }$ , which is also continuous. Therefore if we let $u \in \mathcal { V } ^ { * } = L ^ { \infty } ( D , \mathbb { R } ^ { n } )$ ) and set $v = ( \nabla \phi ^ { * } ) \circ u$ , since u is essentially bounded, we have that v is essentially bounded too, and hence integrable. In the end $v \in L ^ { 1 } ( D , \mathbb { R } ^ { n } )$ and $u = ( \nabla \phi ) \circ v \in \partial \Phi ( v )$

Definition 2.3 of Bregman proximity operators in general Banach spaces requires that ran $\partial ( \Phi + g )$ is the full dual space.   
The following result gives a simple situation in which such condition is satisfied.

Proposition A.3. Let $\phi \in \Gamma _ { 0 } ( \mathbb { R } ^ { n } )$ be a Legendre function, let $p \in [ 1 , + \infty [ $ , and suppose that ϕ is p-uniformly convex with constant $c > 0 ,$ , meaning that

$$
\forall y , y ^ { \prime } \in \mathbb { R } ^ { n } , \forall \lambda \in ] 0 , 1 [ : \ \phi ( ( 1 - \lambda ) y + \lambda y ^ { \prime } ) + \lambda ( 1 - \lambda ) \frac { c } { p } | y - y ^ { \prime } | ^ { p } \leq ( 1 - \lambda ) \phi ( y ) + \lambda \phi ( y ^ { \prime } ) .\tag{12}
$$

Let $\mathcal { V } = L ^ { p } ( D , \mathbb { R } ^ { n } )$ . Then the integral functional Φ: $\mathcal { V } \to ] - \infty , + \infty ]$ defined as in Fact 1 is p-uniformly convex with respect to the norm $\| \cdot \| _ { p }$ . Moreover,for every $g \in \Gamma _ { 0 } ( \mathcal { V } )$ such that dom Φ ∩ dom $g \neq \emptyset$ , we have dom $( \Phi + g ) ^ { * } = \mathcal { V } ^ { * }$ and $( \Phi + g ) ^ { * }$ is Fréchet differentiable on $\nu ^ { * }$ . Thus V<sup>∗</sup> = dom $\partial ( \Phi + g ) = \tan \partial ( \Phi + g )$

Proof. It follows by integrating (12). The second part follows by Zalinescu (2002, Theorem 3.5.10), considering that $\Phi + g$ is also p-uniformly continuous. □

We now provide conditions ensuring that the Bregman proximity operator is well-defined by guaranteeing that the subdifferential covers the entire dual space. This is important because having full range means that every possible dual variable has a corresponding primal solution.

Remark A.4.

(i) If $\mathcal { V } = L ^ { p } ( D , \mathbb { R } ^ { n } )$ with $p \in ] 1 , + \infty [$ , the condition ran $\partial ( \Phi + g ) = \mathcal { V } ^ { * }$ is satisfied if ϕ is p-uniformly convex (see Proposition A.3 in the appendix). Moreover, by Remark A.2, if $p = 1$ and dom $\phi ^ { * } = \mathbb { R } ^ { n }$ , then ran $\partial \Phi = \nu ^ { * }$

(ii) If instead of ran $\partial ( \Phi + g ) = \mathcal { V } ^ { * }$ , one asks the stronger condition ran $( \partial \Phi + \partial g ) = \mathcal { V } ^ { * }$ , then we have $\partial ( \Phi + g ) = \partial \Phi + \partial g$ and the Bregman proximity operator writes down as $\mathrm { p r o x } _ { g } ^ { \Phi } = ( \partial \Phi + \partial g ) ^ { - 1 }$ and $\operatorname { r a n } ( \operatorname { p r o x } _ { g } ^ { \Phi } ) \subset \operatorname { d o m } \partial \Phi$

Finally, we apply these results to an iterative process, specifically the compositional form of (Bregman) neural operators, to ensure well-posedness at each step. By confirming that the proximity operator consistently maps to the correct domain, we establish a stable recursive structure. This prevents domain mismatches and ensures the validity of compositions. Additionally, we emphasize the importance of a compatibility condition on the lifting operator, which guarantees a well-defined initialization for the iterative scheme.

Remark A.5. In view of Remark $\mathrm { A . 4 ( i i ) }$ , the condition ra $\begin{array} { r } { \mathrm { ~ } _ { 1 } ( \partial \Phi _ { t } + \partial g _ { t } ) = \mathcal V _ { t } ^ { * } } \end{array}$ implies that $\mathrm { p r o x } _ { q _ { t } } ^ { \Phi _ { t } } = ( \partial \Phi _ { t } + \partial g _ { t } ) ^ { - 1 }$ and hence $\mathrm { r a n } ( \mathrm { p r o x } _ { g _ { t } } ^ { \Phi _ { t } } ) \subset \mathrm { d o m } \partial \Phi _ { t }$ . In this way dom $\mathscr { L } _ { t } = \mathscr { M } _ { t } ^ { - 1 } ( \mathrm { d o m } \partial \Phi _ { t - 1 } )$ and ran $( \mathcal { L } _ { t } ) \subset$ dom ∂Φ and the composition (2) is well-defined provided that for the lifting operator $\mathcal { P }$ it holds ran $. ( { \mathcal { P } } ) \subset$ dom ∂Φ<sub>1</sub> $( \mathbf { e . g . } , \mathrm { i f } \mathcal { P } ( v ) ( x ) = \nabla \phi _ { 1 } ^ { * } ( P v ( x ) ) )$ .

## A.2. Link Between Activation Function and Proximity Operator

As demonstrated in the work of Combettes & Pesquet (2020a), many activation functions σ can be expressed as proximity operators $\begin{array} { r } { \operatorname { p r o x } _ { g } = \operatorname { a r g m i n } _ { t \in \mathbb { R } } g ( t ) + \frac { 1 } { 2 } ( \cdot - t ) ^ { 2 } } \end{array}$ for some appropriate convex function $g .$ . The simplest case is that of the ReLu activation function, recalled below.

Example 1 (ReLu). The rectified linear unit function $\sigma \colon t \in \mathbb { R } \mapsto \operatorname* { m a x } ( t , 0 ) \in \mathbb { R }$ can be expressed as the proximity operator prox<sub>g</sub> of $\begin{array} { r } { \hat { \mathbf { \rho } } g = \imath _ { [ 0 , + \infty [ \cdot } } \end{array}$ . Henceforth, $\operatorname { p r o x } _ { g }$ reduces to the projection onto the positive orthant.

We also provide a novel characterization of SoftPlus.

Example 2 (SoftPlus). Given $\beta > 0$ , the SoftPlus activation function, i.e., $\sigma \colon t \mapsto \operatorname { S o f t P l u s } _ { \beta } ( t ) \triangleq ( 1 / \beta ) \log ( \exp ( \beta t ) + 1 )$ is the proximity operator of

$$
g \colon t \in \mathbb { R } _ { > 0 } \mapsto \frac { 1 } { \beta ^ { 2 } } \mathrm { L i } _ { 2 } ( \mathrm { e } ^ { - \beta t } ) \in \mathbb { R } _ { > 0 } ,\tag{13}
$$

where $\operatorname { L i } _ { 2 }$ is the dilogarithmfunction defined as $\begin{array} { r } { \operatorname { L i } _ { 2 } \colon t \mapsto - \int _ { 0 } ^ { t } \frac { \log ( 1 - u ) } { u } \mathrm { d } u . } \end{array}$

Proof. For every $s \in \mathbb { R }$ , prox $_ g ( s ) =$ $\mathrm { a r g m i n } _ { t \in \mathbb { R } } \{ h ( t ) \ \triangleq \ g ( t ) + ( 1 / 2 ) ( s - t ) ^ { 2 } \}$ with $h ( t ) ~ = ~ ( 1 / \beta ^ { 2 } ) \mathrm { L i } _ { 2 } ( \mathrm { e } ^ { - \beta t } ) +$ $( 1 / 2 ) ( s \ : - \ : t ) ^ { 2 } = \psi ( t ) \ : - \ : s t \ : + \ : ( 1 / 2 ) s ^ { 2 } $ where we introduced $\begin{array} { r l r } { \psi ( t ) ~ = ~ ( 1 / \beta ^ { 2 } ) \left( \mathrm { L i } _ { 2 } ( \mathrm { e } ^ { - \beta t } ) + ( 1 / 2 ) \log ( \mathrm { e } ^ { - \beta t } ) ^ { 2 } \right) } & { { } = } & { } \end{array}$ $( 1 / \beta ^ { 2 } ) \int ^ { \mathrm { e } ^ { - \beta t } } \log ( r / ( 1 - r ) ) / r \mathrm { d } r$ . The latter can be written as $\begin{array} { r } { \psi ( t ) = ( 1 / \beta ) \int ^ { t } \log ( \mathrm { e } ^ { \beta r } - 1 ) \mathrm { d } t } \end{array}$ r up to a constant. Finally, since h is strongly convex, the minimum is attained for t such that $h ^ { \prime } ( t ) = 0$ , which yields log $( \mathrm { e } ^ { \beta t } - 1 ) = \beta s \Leftrightarrow t = \sigma ( s )$ , thus ending the proof. □

We present an illustration of the convex function g defined in Eq. 13 in Figure 7a. Intuitively, it serves as a smooth surrogate for the indicator function of the positive orthant $\imath _ { [ 0 , + \infty [ \cdot } \mathrm { ~ A ~ }$ larger value of $\beta > 0$ leads to a closer approximation. This aligns with the representation of SoftPlus as the proximity operator of $g$ from Eq. 13, depicted in Fig. 7b where a larger $\beta$ makes SoftPlus closer to ReLU.

![](images/b54b71f0ad17c5e22416903fcf2d46d283078c2b67895fc3011abb77e24b3838.jpg)  
(a) Representation of $\begin{array} { r } { t \mapsto g ( t ) = \frac { 1 } { \beta ^ { 2 } } \mathrm { L i } _ { 2 } ( \mathrm { e } ^ { - \beta t } ) } \end{array}$

![](images/5fb2e001f54b90304240944edeec205d7d8243068299b8a40db4b28eb6fb1f03.jpg)  
(b) Representation of prox $_ g ( t ) = \operatorname { S o f t P l u s } _ { \beta } ( t )$  
Figure 7: Illustration of SoftPlus as a proximity operator.

## B. Approximation Results for Bregman Neural Networks and Operators

## B.1. Bregman Neural Networks

We consider first shallow Bregman neural networks for finite dimensional spaces. Let $\sigma \colon \mathbb { R }  I$ be a homeomorphism, where I is an open interval in R. We $d \in \mathbb { N } _ { + }$ and set

$$
\begin{array} { r } { \mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } ) = \operatorname { s p a n } \big \{ \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + w ^ { \top } x + b ) \big | m \in \Delta ^ { d - 1 } , w \in \mathbb { R } ^ { d } , b \in \mathbb { R } \big \} . } \end{array}\tag{14}
$$

Remark B.1. Since m belongs to the standard simplex $\Delta ^ { d - 1 } , m ^ { \top } x$ is a convex combination of elements of I and so it is an element of I. Thus, since $\sigma ^ { - 1 } \colon I  \mathbb { R }$ , the functions in ${ \mathsf { B N } } _ { 2 } ( \sigma ; I ^ { d } )$ are well-defined from $I ^ { d } \to \mathbb { R }$

The following result follows from an adaptation of the argument in Cybenko (1989) to our different architecture (14).

Theorem B.2. Suppose that σ is sigmoidal, meaning that lim $_ { \substack { \mathrm { \scriptsize { \cdot } } t  - \infty } } \sigma ( t ) = 0 \ a n d \operatorname* { l i m } _ { t  + \infty } \sigma ( t ) = 1$ . Then, the space ${ \mathsf { B N } } _ { 2 } ( \sigma ; I ^ { d } )$ is dense in ${ \mathcal { C } } ( I ^ { d } , \mathbb { R } )$ with respect to the topology of uniform convergence on compact sets.

Proof. Let $K \subset I ^ { d }$ be a compact set. We prove that the trace space ${ \mathsf { B N } } _ { 2 } ( \sigma ; I ^ { d } ) _ { | K }$ is dense in ${ \mathcal { C } } ( K , \mathbb { R } )$ . To that purpose, we rely on the following general fact concerning dense sets in Banach space (see, e.g., Brezis (2011)). Let B be a Banach space, let $\mathcal A \subset B$ . Then the following propositions are equivalent.

• spanA is dense in B

$$
\bullet \ { \mathcal { A } } ^ { \perp } = \{ u ^ { * } \in { \mathcal { B } } ^ { * } | \forall u \in { \mathcal { A } } \colon \langle u , u ^ { * } \rangle = 0 \} = \{ 0 \} .
$$

$$
\bullet \forall u ^ { * } \in B ^ { * } , ( \forall u \in A \colon \langle u , u ^ { * } \rangle = 0 ) \Rightarrow u ^ { * } = 0 .
$$

This implies that for our purpose we can equivalently prove that

$$
\forall \mu \in \mathcal { M } ( K ) \colon \ \left( \forall f \in \mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } ) \colon \int _ { K } f \mu = 0 \right) \Rightarrow \mu = 0 ,
$$

where $\mathcal M ( K )$ is the space of signed finite Radon measures on K (the dual of $\mathcal { C } ( K ) )$ . Thus, let $\mu$ be a signed measure on K and suppose that

$$
\forall f \in { \mathsf { B N } } _ { 2 } ( \sigma ; I ^ { d } ) \colon \int _ { K } f d \mu = 0 .\tag{15}
$$

Fix $w \in \mathbb { R } ^ { d } , m \in \Delta ^ { d - 1 }$ , and $b \in \mathbb { R }$ . Define, for every $\lambda > 0$ and $c \in$ R

$$
\sigma _ { \lambda , c } \colon I \to \mathbb { R } , \quad x \mapsto \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + \lambda ( w ^ { \top } x + b ) + c ) .
$$

It is clear that $\sigma _ { \lambda , c } \in \mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } )$ . Moreover,

$$
\operatorname* { l i m } _ { \lambda \to + \infty } \sigma _ { \lambda , c } ( x ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } w ^ { \top } x + b > 0 } \\ { 0 } & { \mathrm { i f ~ } w ^ { \top } x + b < 0 } \\ { \sigma \big ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c \big ) } & { \mathrm { i f ~ } w ^ { \top } x + b = 0 . } \end{array} \right\} : = \gamma ( x ) .
$$

Define the sets

$$
\begin{array} { r } { \Pi _ { w , b } ^ { + } = \left\{ x \in K \middle | w ^ { \top } x + b > 0 \right\} , \quad \Pi _ { w , b } ^ { - } = \left\{ x \in K \middle | w ^ { \top } x + b < 0 \right\} , \quad \Pi _ { w , b } = \left\{ x \in K \middle | w ^ { \top } x + b = 0 \right\} . } \end{array}
$$

They are intersections of half-spaces and hyperplanes with K. So,

$$
\gamma ( x ) = \chi _ { \Pi _ { w , b } ^ { + } } ( x ) + \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c ) \chi _ { \Pi _ { w , b } } ( x ) ,
$$

where $\chi _ { A }$ is the characteristic functions of the set $A \subset I ^ { d }$ . Since σ is bounded we can apply the Lebesgue’s dominated convergence theorem and get

$$
\operatorname* { l i m } _ { \lambda \to + \infty } \underbrace { \int _ { K } \sigma _ { \lambda , c } d \mu } _ { = 0 } = \int _ { K } \gamma d \mu = \mu ( \Pi _ { w , b } ^ { + } ) + \int _ { \Pi _ { w , b } } \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c ) d \mu ( x ) .
$$

Note that the integral on the left is zero by the hypothesis (15). In this way we proved that

$$
\forall m \in \Delta ^ { d - 1 } , \forall w \in \mathbb R ^ { d } , \forall b , \forall c \in \mathbb R : \quad \mu ( \Pi _ { w , b } ^ { + } ) + \int _ { \Pi _ { w , b } } \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c ) d \mu ( x ) = 0 .\tag{16}
$$

Now observe that (16) implies

$$
\Big | \mu ( \Pi _ { w , b } ^ { + } ) \Big | = \Bigg | \int _ { \Pi _ { w , b } } \sigma \big ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c \big ) d \mu ( x \big ) \Bigg | \leq \int _ { \Pi _ { w , b } } \vert \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c ) \vert d \vert \mu \vert ( x )  0 \mathrm { ~ a s ~ } c  - \infty ,
$$

since $| \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c ) |  0 { \mathrm { ~ a s ~ } } c  - \infty$ (pointwise), where $| \mu |$ is the total variation of $\mu .$ . Therefore, $\mu ( \Pi _ { w , b } ^ { + } ) = 0$ Then (16) yields

$$
\forall c \in \mathbb { R } : \quad \int _ { \Pi _ { w , b } } \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c ) d \mu ( x ) = 0 .
$$

Moreover, by assumption $\sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c )  1 { \mathrm { ~ a s ~ } } c  + \infty$ (pointwise) and hence, again by Lebesgue’s dominated convergence theorem,

$$
\begin{array} { r } { \underbrace { \underset { c  + \infty } { \operatorname* { l i m } } \displaystyle \int _ { \Pi _ { w , b } } \sigma ( \sigma ^ { - 1 } ( m ^ { \top } x ) + c ) d \mu ( x ) } _ { = 0 } = \displaystyle \int _ { \Pi _ { w , b } } 1 d \mu = \mu ( \Pi _ { w , b } ) , } \end{array}
$$

which yields $\mu ( \Pi _ { w , b } ) = 0$ . In the end we proved that the measure $\mu$ is zero on all the sets of type

$$
\Pi _ { w , b } \quad \mathrm { a n d } \quad \Pi _ { w , b } ^ { + } .
$$

Now the proof continues as in Cybenko (1989, Lemma 1), and we can conclude that $\mu = 0$

Now we address the vectorial case. We set

$$
\mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } ) : = \left\{ Q \sigma ( \sigma ^ { - 1 } ( M x ) + W x + b ) \Bigm | \begin{array} { l l } { r \in \mathbb { N } _ { + } , Q \in \mathbb { R } ^ { k \times r } , W , M \in \mathbb { R } ^ { r \times d } , } \\ { \mathrm { w i t h } M \mathrm { r i g h t s t o c h a s t i c } , \mathrm { a n d } b \in \mathbb { R } ^ { r } } \end{array} \right\} ,
$$

where $\sigma$ and $\sigma ^ { - 1 }$ are applied component-wise.

Corollary B.3. We have that

$$
\mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } ) = ( \mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } ) ) ^ { k } : = \underbrace { \mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } ) \times \dots \times \mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } ) } _ { k \ t i m e s }\tag{17}
$$

and it is dense in $\mathcal { C } ( I ^ { d } , \mathbb { R } ^ { k } )$ , in the topology of uniform convergence on compact sets.

Proof. In view of Theorem B.2, it is clear that $( \mathsf { B N _ { 2 } } ( \sigma ; I ^ { d } ) ) ^ { k }$ is dense in $\mathcal { C } ( I ^ { d } , \mathbb { R } ) ^ { k } \cong \mathcal { C } ( I ^ { d } , \mathbb { R } ^ { k } )$ in the topology of uniform convergence on compact sets. Let’s prove equality (17). The inclusion $\mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } ) \subset ( \mathsf { B N } _ { 2 } ( \sigma ; I ^ { d } ) ) ^ { k }$ is immediate. Let $f \colon I ^ { d } \to \mathbb { R } ^ { k }$ with components $f _ { j } \in { \mathsf { B N } } _ { 2 } ( \sigma ; I ^ { d } ) , j = 1 , \dotsc , k$ . Then, there exists $r \in \mathbb { N } _ { + }$ , and for each $j \in \{ 1 , \ldots , k \}$ $q _ { j } \in \mathbb { R } ^ { r } , W _ { j } \in \mathbb { R } ^ { r \times d } , b _ { j } \in \mathbb { R } ^ { r }$ , and $M _ { j } \in \mathbb { R } ^ { r \times d }$ right stochastic matrix (the rows are positive and sum one), such that

$$
f _ { j } ( x ) = q _ { j } ^ { \top } \sigma ( \sigma ^ { - 1 } ( M _ { j } x ) + W _ { j } x + b _ { j } ) .
$$

Then considering the block matrices

$$
M = \left[ \begin{array} { l } { M _ { 1 } } \\ { \vdots } \\ { M _ { k } } \end{array} \right] \in \mathbb { R } ^ { k r \times d } , \quad W = \left[ \begin{array} { l } { W _ { 1 } } \\ { \vdots } \\ { W _ { k } } \end{array} \right] \in \mathbb { R } ^ { k r \times d } , \quad b = \left[ \begin{array} { l } { b _ { 1 } } \\ { \vdots } \\ { b _ { k } } \end{array} \right] \in \mathbb { R } ^ { k r } , \quad Q = \left[ \begin{array} { l l l l } { q _ { 1 } ^ { \top } } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { q _ { 2 } ^ { \top } } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 0 } & { 0 } & { \cdots } & { q _ { k } ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { k \times k r } ,
$$

we have

$$
f ( x ) = Q \sigma ( \sigma ^ { - 1 } ( M x ) + W x + b ) ,
$$

and hence $f \in { \mathsf { B N } } _ { 2 } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } )$ . The statement follows.

A general deep Bregman neural network with $T$ layers is defined as follows

$$
{ \sf B N } _ { T } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } ) = \big \{ W _ { T } \circ L _ { T - 1 } \circ \cdot \cdot \cdot \circ L _ { 1 } \big \} ,
$$

where, for every $t = 1 , \dots , T - 1$

$$
L _ { t } \colon I ^ { n _ { t - 1 } } \to I ^ { n _ { t } } , \quad x \mapsto \sigma ( \sigma ^ { - 1 } ( M _ { t } x ) + W _ { t } x + b _ { t } ) ,\tag{18}
$$

with $W _ { t } \in \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } } , b _ { t } \in \mathbb { R } ^ { n _ { t } }$ and $M _ { t } \in \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } }$ right stochastic, for $t = 1 , \dots , T { - } 1$ , with $n _ { 0 } = n$ and $W _ { T } \in \mathbb { R } ^ { k \times n _ { T - 1 } }$ Note that also the dimensions $n _ { 1 } , \ldots , n _ { T - 1 }$ can be chosen freely. Clearly for a deep network with $T > 2 .$ , if we take, for every $t = 2 , \ldots , T - 1 , n _ { t } = n _ { 1 } , W _ { t } = 0 , b _ { t } = 0$ , and $M _ { t }$ equals to the identity, then the layers $L _ { t }$ with $t = 2 , \ldots , T - 1$ act as the identity operator and hence

$$
{ \mathsf { B N } } _ { 2 } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } ) \subset { \mathsf { B N } } _ { T } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } ) .
$$

Therefore, ${ \mathsf { B N } } _ { T } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } )$ is dense in $\mathcal { C } ( I ^ { d } , \mathbb { R } ^ { k } )$ for the topology of uniform convergence on compact sets.

Remark B.4. Often in applications it is desirable to have functions defined on the entire space $\mathbb { R } ^ { d }$ . In this case one can simply precompose the functions in ${ \mathsf { B N } } _ { T } ( \sigma ; I ^ { d } , \mathbb { R } ^ { k } )$ by the homeomorphism

$$
x \in \mathbb { R } ^ { d } \to \sigma ( x ) \in I ^ { d }
$$

obtaining a dense set in $\mathcal { C } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { k } )$ (for any $T \geq 2 )$ ). Such space is then denoted by ${ \mathsf { B N } } _ { T } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { k } )$

Let $D \subset \mathbb { R } ^ { d }$ be any nonempty bounded open set. If ${ \mathcal { F } } ( \mathbb { R } ^ { d } )$ is any class of real functions from $\mathbb { R } ^ { d }$ to R we denote by $\mathcal { F } _ { | \overline { { D } } }$ the set of restrictions to $\overline { { D } }$ of the functions in ${ \mathcal { F } } ( \mathbb { R } ^ { d } )$ . In the following according to Remark B.4 we put

$$
{ \mathsf { B N } } _ { T } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { k } ) = \big \{ W _ { T } \circ L _ { T - 1 } \circ \dotsb \circ L _ { 1 } \circ \sigma \big \} ,\tag{19}
$$

which is a dense space in $\mathcal { C } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { k } )$ ) with respect to the topology of uniform convergence on compact sets.

Lemma B.5. Suppose that σ is a sigmoidal activation function as in Theorem B.2. Let $p \in \mathsf { \Gamma } [ 1 , + \infty [ \cdot$ Then $\mathsf { B N } _ { T } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { k } ) _ { | \overline { { D } } }$ is dense in $L ^ { p } ( D , \mathbb { R } ^ { k } )$ (in the norm $\| \cdot \| _ { p } )$

Proof. It is well known that $\mathcal { C } _ { c } ( D , \mathbb { R } ^ { k } )$ is dense in $L ^ { p } ( D , \mathbb { R } ^ { k } )$ and hence $\mathcal { C } ( \mathbb { R } ^ { n } , \mathbb { R } ^ { k } ) _ { | \overline { { D } } }$ is dense in $L ^ { p } ( D , \mathbb { R } ^ { k } )$ (in the norm $\| \cdot \| _ { p } )$ . Moreover, ${ \mathsf { B N } } _ { T } ( \sigma ; \mathbb { R } ^ { n } , \mathbb { R } ^ { k } ) _ { | { \overline { { D } } } }$ is dense in $\mathcal { C } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { k } ) _ { | \overline { { D } } }$ (in the norm $\| \cdot \| _ { \infty } )$ . On the other hand

$$
\forall f \in { \mathcal C } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { k } ) _ { | \overline { { D } } } \colon \| f \| _ { p } = \Big ( \int _ { D } | f | ^ { p } d x \Big ) ^ { 1 / p } \leq \| f \| _ { \infty } | D | ^ { 1 / p } .
$$

Thus, if $f \in L ^ { p } ( D , \mathbb { R } ^ { k } )$ and $\varepsilon > 0 .$

$$
\begin{array} { l } { \displaystyle \exists g \in \mathcal { C } ( \mathbb { R } ^ { d } , \mathbb { R } ^ { k } ) _ { \overline { { D } } } \mathrm { ~ s . t . ~ } \| f - g \| _ { p } \leq \frac { \varepsilon } { 2 } } \\ { \displaystyle \exists h \in { \mathsf { B N } } _ { T } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { k } ) _ { | \overline { { D } } } \mathrm { ~ s . t . ~ } \| g - h \| _ { \infty } \leq \frac { \varepsilon } { 2 | D | ^ { 1 / p } } \Rightarrow \| g - h \| _ { p } \leq \frac { \varepsilon } { 2 } } \end{array}
$$

and hence $\| f - h \| _ { p } \leq \varepsilon$

Remark B.6. It is sometimes required that neural networks, of any depth, include constant functions. Standard feed-forward neural networks have the form

$$
( W _ { T } \cdot + b _ { T } ) \circ \sigma ( W _ { T - 1 } \cdot + b _ { T - 1 } ) \circ \cdot \cdot \cdot \circ \sigma ( W _ { 1 } \cdot + b _ { 1 } ) ,
$$

so it is clear that they include constant functions (just take $W _ { T } = 0 )$ . However, for Bregman neural networks as defined in $( 1 9 ) \ – ( 1 8 )$ this is not clear. An immediate modification to achieve this goal is to explicitly add a constant $b _ { T }$ in the last layer. Another possibility is to lift the input space by one dimension, precomposing the neural network with a (free) linear embedding. In particular, if we consider the canonical embedding

$$
J \colon \mathbb { R } ^ { d } \to \mathbb { R } ^ { d + 1 } \colon x \mapsto \left[ { \boldsymbol { x } } \right] ,
$$

and define the following matrices

$$
\tilde { W } _ { t } = \left[ \begin{array} { c c } { W _ { t } } & { 0 } \\ { 0 } & { 1 } \end{array} \right] , \quad \tilde { M } _ { t } = \left[ \begin{array} { c c } { M _ { t } } & { 0 } \\ { 0 } & { 1 } \end{array} \right] , \quad \tilde { b } _ { t } = \left[ \begin{array} { c } { b _ { t } } \\ { - \sigma ( 0 ) } \end{array} \right] , \quad ( \mathrm { f o r ~ t < T ) } \quad \tilde { W } _ { T } = \left[ W _ { T } \quad b _ { T } / \sigma ( 0 ) \right] ,
$$

then, for $t = 1 , \dots , T - 1$ , according to (18), we have

$$
\forall y \in I ^ { n _ { t - 1 } } \colon \tilde { L } _ { t } \left[ \operatorname { \sigma } ( 0 ) \right] = \sigma \biggl ( \sigma ^ { - 1 } \biggl ( \tilde { M } _ { t } \left[ \operatorname { \sigma } _ { \sigma ( 0 ) } ^ { y } \right] \biggr ) + \tilde { W } _ { t } \left[ \operatorname { \sigma } _ { \sigma ( 0 ) } ^ { y } \right] + \tilde { b } _ { t } \biggr ) = \left[ \operatorname { \sigma } _ { \sigma ( 0 ) } ^ { L _ { t } y } \right]
$$

and hence

$$
\tilde { W } _ { T } \circ \tilde { L } _ { T - 1 } \circ \cdots \circ \tilde { L } _ { 1 } \circ \sigma \circ J = W _ { T } \circ L _ { T - 1 } \circ \cdots \circ L _ { 1 } \circ \sigma + b _ { T } .
$$

## B.2. Bregman Neural Operators

Now we start addressing the proof of Theorem 4.1. We will rely on the work of Kovachki et al. (2023), from which, for the sake of reader’s convenience, we report the following facts.

Fact 3 (Lemma 28 and 30 in Kovachki et al. (2023)). Let $D \subset \mathbb { R } ^ { d }$ be a bounded set and let $L \in ( W ^ { m , p } ( D ) ) ^ { * }$ , for some $m \geq 0$ and $1 \leq p < + \infty$ , or $L \in ( { \mathcal { C } } ( D ) ) ^ { * }$ . Then, for any closed and bounded set $K \subset A$ and $\varepsilon > 0$ , there exists a function $\kappa \in \mathcal { C } _ { c } ^ { \infty } ( D )$ such that

$$
\operatorname* { s u p } _ { v \in K } \left| L ( v ) - \int _ { D } \kappa ( x ) v ( x ) d x \right| < \varepsilon .
$$

Fact 4 (Lemma 22 and 26 in Kovachki et al. (2023)). Let $D \subset \mathbb { R } ^ { d }$ be a bounded set and let $\mathcal { A }$ and U be any one of the Banach spaces $\mathcal { C } ( \overline { { D } } )$ or $W ^ { m , p } ( D )$ , with $m \geq 0$ and $1 \leq p < + \infty$ . Let $\mathcal { G } \colon A \to \mathcal { U }$ be a continuous operator, $K \subset A$ be a compact set and $\varepsilon > 0$ . Then there exist $J , J ^ { \prime } \in \mathbb { N }$ and

$$
R \colon \mathcal { A } \to \mathbb { R } ^ { J } , \quad f \colon \mathbb { R } ^ { J } \to \mathbb { R } ^ { J ^ { \prime } } , \quad S \colon \mathbb { R } ^ { J ^ { \prime } } \to \mathcal { U } ,
$$

with R and S linear continuous and $f$ continuous, such that

$$
\operatorname* { s u p } _ { v \in K } \| \mathcal G ( v ) - ( S \circ f \circ R ) ( v ) \| < \varepsilon .
$$

In the following we set $D \subset \mathbb { R } ^ { d }$ be a bounded set and

$$
\begin{array} { r } { \boldsymbol { A } ( \boldsymbol { D } , \mathbb { R } ^ { n _ { 0 } } ) = \boldsymbol { W } ^ { m , p } ( \boldsymbol { D } , \mathbb { R } ^ { n _ { 0 } } ) \quad \mathrm { o r } \quad \boldsymbol { A } ( \boldsymbol { D } , \mathbb { R } ^ { n _ { 0 } } ) = \boldsymbol { \mathcal { C } } ( \boldsymbol { D } , \mathbb { R } ^ { n _ { 0 } } ) , } \end{array}
$$

where the integer $m \geq 0$ and $p \in [ 1 , + \infty [$ . Moreover we will assume that (by possibly changing the definition slightly) Bregman neural networks include constant functions (recall Remark B.6). Because of the density result given in the previous section, we can essentially follow the same line of arguments in Kovachki et al. (2023), but we need to take special care of the different structure of Bregman neural network/operators (in particular in Lemma B.10).

Lemma B.7. Let $L \in { \mathcal { A } } ^ { * }$ and $K \subset A$ be a compact set. Then there exists $h \in { \mathsf { B N } } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { n _ { 0 } } ) _ { | D }$ such that

$$
\operatorname* { s u p } _ { v \in K } \Big | L ( v ) - \int _ { D } \langle h ( x ) , v ( x ) \rangle d x \Big | < \varepsilon .
$$

Proof. The space $\mathcal { A }$ is (isomorphic to) a product space, meaning $\begin{array} { r } { A = \prod _ { i = 1 } ^ { n _ { 0 } } \mathcal { A } _ { i } } \end{array}$ , where $A _ { i }$ is a space of real valued functions on $D$ . Set $K _ { i } = \mathrm { p r } _ { i } ( K )$ , which is a compact set of $A _ { i } ,$ so that $K \subset \textstyle \prod _ { i = 1 } ^ { n _ { 0 } } K _ { i }$ . Then $L \colon A \to$ R can be written as $\begin{array} { r } { L v = \sum _ { i = 1 } ^ { n _ { 0 } } L _ { i } v _ { i } } \end{array}$ with $L _ { i } \colon A _ { i } \to \mathbb { R } .$ . By Fact 3, for every $i = 1 , \ldots , n _ { 0 }$ , there exists $\kappa _ { i } \in \mathcal { C } _ { c } ( D )$ such that

$$
\operatorname* { s u p } _ { v _ { i } \in K _ { i } } \left| L _ { i } v _ { i } - \int _ { D } \kappa _ { i } v _ { i } d x \right| < \frac { \varepsilon } { 2 n _ { 0 } } .
$$

Let $\kappa \in \mathcal { C } _ { c } ( D , \mathbb { R } ^ { n _ { 0 } } )$ with components $\kappa _ { i } \in \mathcal { C } _ { c } ( D )$ . Then

$$
\Big | L v - \int _ { D } \langle \kappa ( x ) , v ( x ) \rangle d x \Big | = \Big | \sum _ { i = 1 } ^ { n _ { 0 } } L _ { i } v _ { i } - \sum _ { i = 1 } ^ { n _ { 0 } } \int _ { D } \kappa _ { i } v _ { i } d x \Big | \leq \sum _ { i = 1 } ^ { n _ { 0 } } \lfloor L _ { i } v _ { i } - \int _ { D } \kappa _ { i } v _ { i } d x \Big | < \frac { \varepsilon } { 2 } .
$$

Since $\mathcal { A } \subset L ^ { 1 } ( D , \mathbb { R } ^ { n _ { 0 } } )$ we set $\begin{array} { r } { \gamma = \operatorname* { s u p } _ { v \in K } \left. v \right. _ { 1 } < + \infty } \end{array}$ . Moreover, since Bregman shallow neural networks are dense in the space of continuous functions (Remark B.4), there exists $h \in { \mathsf { B N } } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { n _ { 0 } } ) _ { | \overline { { D } } }$ such that $\| h - \kappa \| _ { \infty } \leq \varepsilon / ( 2 \gamma )$ and hence, for every $v \in K$

$$
\Big | \int _ { D } \langle \kappa , v \rangle d x - \int _ { D } \langle h , v \rangle d x \Big | = \Big | \int _ { D } \langle \kappa - h , v \rangle d x \Big | \leq \int _ { D } | \kappa ( x ) - h ( x ) | | v ( x ) | d x \leq \| \kappa - h \| _ { \infty } \| u \| _ { 1 } < \frac { \varepsilon } { 2 } .
$$

Therefore,

$$
\left| L v - \int _ { D } \langle h , v \rangle d x \right| \leq \left| L v - \int _ { D } \langle \kappa , v \rangle d x \right| + \left| \int _ { D } \langle \kappa , v \rangle d x - \int _ { D } \langle h , v \rangle d x \right| < \varepsilon
$$

and the statement follows.

Lemma B.8. Le $R \colon { \mathcal { A } } \to \mathbb { R } ^ { J }$ be a linear continuous operator, $K \subset A$ a compact set and $\varepsilon > 0 .$ . Then there exists a linear continuous operator $R ^ { \mathsf { B N } } : \mathcal { A }  \mathbb { R } ^ { J }$ acting as

$$
v \mapsto R ^ { \mathsf { B N } } v = \int _ { D } h ( y ) v ( y ) d y ,
$$

where $h \in { \mathsf { B N } } _ { 2 } { \left( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { J \times n _ { 0 } } \right) } _ { | \overline { { D } } } ,$ , such that

$$
\operatorname* { s u p } _ { v \in K } | R v - R ^ { \mathtt { B N } } v | < \varepsilon .
$$

Proof. Consider the components $R _ { j } \colon { \mathcal { A } } \to \mathbb { R } , j = 1 , \ldots , J$ . Then $R _ { j } \in { \mathcal { A } } ^ { * }$ , and by Lemma B.7

$$
\exists h _ { j } \in \mathsf { B N } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { n _ { 0 } } ) _ { | D } \quad \mathrm { s . t . } \quad \operatorname* { s u p } _ { v \in K } \Big | R _ { j } v - \int _ { D } \langle h _ { j } ( x ) , v ( x ) \rangle d x \Big | \leq \frac { \varepsilon } { \sqrt { J } } .
$$

Let $h \colon  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { J \times n _ { \mathrm { C } } }$ with

$$
h ( \boldsymbol { x } ) = \left[ \begin{array} { c } { h _ { 1 } ( \boldsymbol { x } ) ^ { \top } } \\ { \vdots } \\ { h _ { J } ( \boldsymbol { x } ) ^ { \top } } \end{array} \right] .
$$

Clearly $h \in { \mathsf { B N } } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { J \times n _ { 0 } } ) _ { | D }$ and

$$
\forall v \in K : \ \left| R v - \int _ { D } h ( x ) v ( x ) d x \right| ^ { 2 } = \sum _ { i = 1 } ^ { J } \left| R _ { j } v - \int _ { D } \langle h _ { j } ( x ) , v ( x ) \rangle d x \right| ^ { 2 } < \varepsilon ^ { 2 }
$$

and the statement follows.

Remark B.9. Both the linear continuous operators R and $R ^ { \mathsf { B N } }$ in Lemma B.8 can be canonically lifted to Lebesgue spaces as follows.

$$
\begin{array} { r l } { \mathcal { R } \colon \boldsymbol { \mathcal { A } } \to L ^ { p } ( D , \mathbb { R } ^ { J } ) , } & { \mathcal { R } v = ( R v ) \mathbb { 1 } _ { D } } \\ { \mathcal { R } ^ { \mathsf { B N } } \colon \boldsymbol { \mathcal { A } } \to L ^ { p } ( D , \mathbb { R } ^ { J } ) , } & { \mathcal { R } ^ { \mathsf { B N } } v = ( R ^ { \mathsf { B N } } v ) \mathbb { 1 } _ { D } , } \end{array}
$$

where $\mathbb { 1 } _ { D }$ denotes the constant function $x \mapsto 1$ on D. Moreover $\mathcal { R } ^ { \mathsf { B N } }$ is actually an integral operator. Indeed if we define the kernel

$$
\kappa _ { h } \colon D \times D \to \mathbb { R } ^ { J \times n _ { 0 } } , \quad \kappa _ { h } ( x , y ) = h ( y )
$$

we have

$$
( \mathcal R ^ { \mathsf { B N } } v ) ( x ) = R ^ { \mathsf { B N } } v = \int _ { D } h ( y ) v ( y ) d y = \int _ { D } \kappa _ { h } ( x , y ) v ( y ) d y .
$$

The following result is the analogue of Kovachki et al. (2023, Lemma 35) and establishes that a finite dimensional Bregman neural network can be canonically lifted in Lebesgue spaces. However, here we need to take care of the domain of the Bregman operator layers.

Lemma B.10. Let $f \in \mathsf { B N } _ { T } ( \sigma ; \mathbb { R } ^ { J } , \mathbb { R } ^ { J ^ { \prime } } ) , D \subset \mathbb { R } ^ { d }$ a nonempty open set and $p \in [ 1 , + \infty ]$ . Then there exists a neural operator

$$
\begin{array} { r } { \mathcal { N } ^ { \mathtt { B N } } : L ^ { p } ( D , \mathbb { R } ^ { J } )  L ^ { p } ( D , \mathbb { R } ^ { J ^ { \prime } } ) , \quad \mathcal { N } ^ { \mathtt { B N } } = \mathcal { K } _ { T } \circ \mathcal { L } _ { T - 1 } \circ \cdot \cdot \circ \mathcal { L } _ { 1 } \circ \sigma , } \end{array}
$$

where, for every $t = 1 , \dots , T - 1$

$$
\mathcal { L } _ { t } ( v ) = \sigma ( \sigma ^ { - 1 } ( \mathcal { M } _ { t } v ) + \mathcal { K } _ { t } v + b _ { t } )
$$

and such that the linear integral operators $\mathcal { M } _ { t }$ and $\textstyle { \mathcal { K } } _ { t }$ and thefunctions $b _ { t }$ are defined (parametrized) byfinite dimensional Bregman shallow neural networks and

$$
\forall w \in \mathbb { R } ^ { J } \colon \mathcal { N } ^ { \mathsf { B N } } ( w \mathbb { 1 } _ { D } ) = f ( w ) \mathbb { 1 } _ { D } ,
$$

where $\mathbb { 1 } _ { D }$ denotes the constantfunction $x \mapsto 1$ on $D .$

Proof. By definition

$$
f = K _ { T } \circ L _ { T - 1 } \circ \cdot \cdot \cdot L _ { 1 } \circ \sigma , \quad L _ { t } ( w ) = \sigma ( \sigma ^ { - 1 } ( M _ { t } w ) + K _ { t } w + b _ { t } ) ,
$$

where $\sigma \colon \mathbb { R }  I$ and, for $t = 1 , \ldots , T , K _ { t } \in \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } }$ and $b _ { t } \in \mathbb { R } ^ { n _ { t } }$ , and for every $t = 1 , \ldots , T - 1 , M _ { t } \in \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } }$ , is right stochastic, $n _ { 0 } = J$ and $n _ { T } = J ^ { \prime }$ . Since, we are assuming that Bregman neural networks contain constant functions (recall the sentence before Lemma B.7), we have

$$
\bullet b _ { t } \mathbb { 1 } _ { D } \in \mathsf { B N } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { n _ { t } } ) _ { | \overline { { D } } } \subset \mathcal { C } ( \overline { { D } } , \mathbb { R } ^ { n _ { t } } )
$$

$$
\bullet \ \kappa _ { t } = \frac { 1 } { | D | } K _ { t } \mathbb { 1 } _ { D \times D } \in \mathsf { B N } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } , \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } } ) _ { | \overline { { D } } \times \overline { { D } } } \subset \mathcal { C } ( \overline { { D } } \times \overline { { D } } , \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } } ) \ \mathrm { a n d }
$$

$$
K _ { t } \colon L ^ { p } ( D , \mathbb { R } ^ { n _ { t - 1 } } ) \to L ^ { q } ( D , \mathbb { R } ^ { n _ { t } } )
$$

$$
v \mapsto ( K _ { t } v ) ( x ) = \int _ { D } \kappa _ { t } ( x , y ) v ( y ) d y = \int _ { D } \frac { 1 } { | D | } K _ { t } v ( y ) d y = K _ { t } \bar { v } ,
$$

where v¯ is the mean value of v. So that $\mathcal { K } _ { t } v = ( K _ { t } \bar { v } ) \mathbb { 1 } _ { D }$ is a constant function.

$$
\bullet \mu _ { t } = \frac { 1 } { | D | } M _ { t } \mathbb { 1 } _ { D \times D } \in \mathsf { B N } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } , \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } } ) _ { | \overline { { D } } \times \overline { { D } } } \subset \mathcal { C } ( \overline { { D } } \times \overline { { D } } , \mathbb { R } ^ { n _ { t } \times n _ { t - 1 } } )
$$

$$
\begin{array} { r l } & { \mathcal { M } _ { t } : L ^ { p } ( D , \mathbb { R } ^ { n _ { t - 1 } } ) \to L ^ { p } ( D , \mathbb { R } ^ { n _ { t } } ) } \\ & { \qquad v \mapsto ( \mathcal { M } _ { t } v ) ( x ) = \displaystyle \int _ { D } \mu _ { t } ( x , y ) v ( y ) d y = \displaystyle \int _ { D } \frac { 1 } { | D | } M _ { t } v ( y ) d y = \displaystyle M _ { t } \bar { v } . } \end{array}
$$

Moreover, since $M _ { t }$ is right stochastic, if the function v has range (almost everywhere) in $I ^ { n _ { t - 1 } }$ , we have that $\bar { v } \in I ^ { n _ { t - 1 } } \Rightarrow M _ { t } \bar { v } \in I ^ { n _ { t } }$ , Hence

$$
\mathcal { M } _ { t } ( \mathrm { d o m } \partial \Phi _ { t - 1 } ) \subset \mathrm { d o m } \partial \Phi _ { t } .
$$

Indeed, recall that $\Phi _ { t } \colon L ^ { p } ( D , \mathbb { R } ^ { n _ { t } } ) \to \left] - \infty , + \infty \right]$ and

$$
\forall v \in L ^ { p } ( D , \mathbb { R } ^ { n _ { t } } ) \colon \Phi _ { t } ( v ) = \int _ { D } \phi _ { t } ( v ( x ) ) d x , \quad \forall w \in \mathbb { R } ^ { n _ { t } } \colon \phi _ { t } ( w ) = \sum _ { i = 1 } ^ { n _ { t } } \psi ( w _ { i } )
$$

with $\psi : \mathbb { R } \to ] - \infty , + \infty ]$ ] Legendre, in $\dot { \iota } ( \operatorname { d o m } \psi ) = I ,$ dom $\psi ^ { * } = \mathbb { R } , \sigma = ( \psi ^ { * } ) ^ { \prime }$ , and $\sigma ^ { - 1 } = \psi ^ { \prime }$ , so that dom $\partial \Phi _ { t } =$ $\{ v \in L ^ { p } ( D , \mathbb { R } ^ { n _ { t } } ) \mid$ for a.e. $x \in D , v ( x ) \in I ^ { n _ { t } } \}$ and for $v \in$ dom $\Phi _ { t } , \partial \Phi _ { t } ( v ) = \{ \nabla \phi \circ v \}$

It follows from the previous considerations that if $v \in \mathrm { d o m } \partial \Phi _ { t - 1 } \subset L ^ { p } ( D , \mathbb { R } ^ { n _ { t - 1 } } )$ , we have ${ \mathcal { K } } _ { t } ( v ) = ( K _ { t } { \bar { v } } ) \mathbb { 1 } _ { D }$ and $\mathcal M _ { t } v = ( M _ { t } \bar { v } ) \mathbb { 1 } _ { D } .$ , and hence

$$
\mathcal { L } _ { t } ( v ) = \sigma ( \sigma ^ { - 1 } ( \mathcal { M } _ { t } v ) + K _ { t } v + b _ { t } \mathbb { 1 } _ { D } ) ( x ) = \sigma ( \sigma ^ { - 1 } ( \mathcal { M } _ { t } \bar { v } ) + K _ { t } \bar { v } + b _ { t } ) .
$$

Note that here $\mathcal { V } _ { t } = L ^ { p } ( D , \mathbb { R } ^ { n _ { t } } )$ . Thus, we have

$$
\begin{array} { r } { \mathcal { L } _ { t } ( v ) = ( L _ { t } \bar { v } ) \mathbb { 1 } _ { D } , } \end{array}
$$

meaning that the operator layer $\mathcal { L } _ { t }$ transforms any function in $L ^ { p } ( D , \mathbb { R } ^ { n _ { t } } )$ into a constant function, where the constant is the mean value of the function, transformed via the standard (finite dimensional) Bregman layer $L _ { t }$ . In particular, if $w \in \mathbb { R } ^ { J }$ we have

$$
\begin{array} { c } { { \mathcal { L } } _ { 1 } ( \sigma ( w \mathbb { 1 } _ { D } ) ) = \mathcal { L } _ { 1 } ( \sigma ( w ) \mathbb { 1 } _ { D } ) = L _ { 1 } ( \sigma ( w ) ) \mathbb { 1 } _ { D } } \\ { { \mathcal { L } } _ { 2 } ( \mathcal { L } _ { 1 } ( \sigma ( w \mathbb { 1 } _ { D } ) ) ) = \mathcal { L } _ { 2 } ( L _ { 1 } ( \sigma ( w ) ) \mathbb { 1 } _ { D } ) = L _ { 2 } ( L _ { 1 } ( \sigma ( w ) ) ) \mathbb { 1 } _ { D } , } \end{array}
$$

and so on. Therefore, if we set

$$
\mathcal { N } ^ { \mathsf { B N } } = \mathcal { K } _ { T } \circ \mathcal { L } _ { T - 1 } \circ \cdots \circ \mathcal { L } _ { 1 } \circ \sigma ,
$$

the statement follows.

Remark B.11. Let $S \colon \mathbb { R } ^ { J ^ { \prime } }  \mathcal { U } ( D , \mathbb { R } ^ { k } )$ be linear (and continuous) and set

$$
\forall i = 1 , \dots , J ^ { \prime } \colon s _ { j } = S e _ { j } \in \mathcal { U } ,
$$

where $( e _ { j } ) _ { 1 \leq j \leq J ^ { \prime } }$ is the canonical basis of $\mathbb { R } ^ { J ^ { \prime } }$ . Define the function $s \colon D \to \mathbb { R } ^ { k \times J ^ { \prime } }$ , with $s ( x ) = [ s _ { 1 } ( x ) ~ \cdot ~ \cdot ~ s _ { J } ^ { \prime } ( x ) ]$ , which has the $\boldsymbol { s } _ { j } \mathbf { \bar { s } }$ as columns. Then

$$
\forall w \in \mathbb { R } ^ { J ^ { \prime } } \colon S w = S \Big ( \sum _ { j = 1 } ^ { J ^ { \prime } } w _ { j } e _ { j } \Big ) = \sum _ { j = 1 } ^ { J ^ { \prime } } w _ { j } s _ { j } \Rightarrow ( S w ) ( x ) = \sum _ { j = 1 } ^ { J ^ { \prime } } w _ { j } s _ { j } ( x ) = s ( x ) w .
$$

Thus, the action of S can be represented by a matrix-valued function with columns in U. Moreover, the linear operator S can be lifted to a linear integral operator from $L ^ { p } ( D , \mathbb { R } ^ { J ^ { \prime } } )$ to U. Indeed if we define the kernel

$$
\kappa _ { s } \colon D \times D \to \mathbb { R } ^ { k \times J ^ { \prime } } , \quad \kappa _ { s } ( x , y ) = \frac { 1 } { | D | } s ( x ) ,
$$

for every $v \in L ^ { p } ( D , \mathbb { R } ^ { J ^ { \prime } } )$ , we have

$$
( S v ) ( x ) = \int _ { D } \kappa _ { s } ( x , y ) v ( y ) d y = \int _ { D } { \frac { 1 } { | D | } } s ( x ) v ( y ) d y = s ( x ) { \bar { v } } ,
$$

where v¯ is the mean value of $v .$ In the end S : $L ^ { p } ( D , \mathbb { R } ^ { J ^ { \prime } } )  \mathcal { U }$ and

$$
\forall v \in L ^ { p } ( D , \mathbb { R } ^ { J ^ { \prime } } ) \colon S v = S \bar { v } ,
$$

and hence, for every $w \in \mathbb { R } ^ { J ^ { \prime } } , S ( w \mathbb { 1 } _ { D } ) = S w$ , meaning that $s$ is actually an extension of $S$ to the Lebesgue space $L ^ { p } ( D , \mathbb { R } ^ { J ^ { \prime } } )$

Lemma B.12. Let $S \colon \mathbb { R } ^ { J ^ { \prime } }  \mathcal { U } ( D , \mathbb { R } ^ { k } )$ be linear (and continuous). Let $K \subset \mathbb { R } ^ { J ^ { \prime } }$ be a compact set and $\varepsilon > 0 .$ . Then there exists afunction $h \in { \mathsf { B N } } _ { 2 } ( \sigma ; \mathbb { R } ^ { d } , \mathbb { R } ^ { k \times J ^ { \prime } } ) _ { | D }$ so thatfor the corresponding linear operator $S ^ { \mathsf { \tilde { B } N } } \colon \mathbb { R } ^ { J ^ { \prime } } \to \mathcal { U }$ defined as

$$
\forall w \in \mathbb { R } ^ { J ^ { \prime } } \colon ( S ^ { \mathsf { B N } } w ) ( x ) = \sum _ { i = 1 } ^ { J ^ { \prime } } w _ { j } h _ { j } ( x ) = h ( x ) w ,
$$

according to Remark B.11, we have

$$
\operatorname* { s u p } _ { w \in K } \| S w - S ^ { \mathsf { B N } } w \| _ { \mathcal { U } } < \varepsilon .
$$

Finally we are ready for the proof of Theorem 4.1.

ProofofTheorem 4.1. It follows from Fact 4 that there exist $J , J ^ { \prime } \in \mathbb { N }$ and

$$
R \colon \mathcal { A } \to \mathbb { R } ^ { J } , \quad f \colon \mathbb { R } ^ { J } \to \mathbb { R } ^ { J ^ { \prime } } , \quad S \colon \mathbb { R } ^ { J ^ { \prime } } \to \mathcal { U } ,
$$

with R and S linear continuous and f continuous, such that

$$
\operatorname* { s u p } _ { v \in K } \| \mathcal G ( v ) - ( S \circ f \circ R ) ( v ) \| < \varepsilon .
$$

Now, taking advantage of the previous lemmas we want to replace the operators R and S with analogue operators depending on shallow Bregman neural networks, and the function $f$ with a Bregman neural network. It follows from Lemma B.8 tha for every $n \in \mathbb { N }$ there exist

$$
R _ { n } ^ { \mathrm { B N } } \colon \mathcal { A } \to \mathbb { R } ^ { J } \mathrm { ~ l i n e a r ~ c o n t i n u o u s ~ o p e r a t o r ~ s u c h ~ t h a t ~ } \operatorname* { s u p } _ { v \in K } \left| R v - R _ { n } ^ { \mathrm { B N } } v \right| < \frac { 1 } { n + 1 } ,
$$

where $R _ { n } ^ { \mathsf { B N } }$ depends on a Bregman shallow network $h _ { n }$ as specified in Lemma B.8. Clearly this implies that $\begin{array} { r } { \operatorname* { l i m } _ { n  + \infty } R _ { n } ^ { \mathsf { B N } } v = R v } \end{array}$ uniformly on $K ,$ so that the set

$$
K _ { 1 } : = R ( K ) \cup \bigcup _ { n \in \mathbb { N } } R _ { n } ^ { \mathtt { B N } } ( K ) \subset \mathbb { R } ^ { J }
$$

is compact (see Kovachki et al. (2023, Lemma 21)). Since $f$ is continuous, it is uniformly continuous on $K _ { 1 }$ , hence given $\varepsilon > 0$ there exists $\delta > 0$ such that

$$
\forall w , w ^ { \prime } \in K _ { 1 } \colon | w - w ^ { \prime } | < \delta \Rightarrow | f ( w ) - f ( w ^ { \prime } ) | < \frac { \varepsilon } { 3 \| S \| } .
$$

Moreover, there exists $f ^ { \mathsf { B N } } \in \mathsf { B N } _ { 2 } ( \sigma ; \mathbb { R } ^ { J } , \mathbb { R } ^ { J ^ { \prime } } )$ such that

$$
\operatorname* { s u p } _ { w \in K _ { 1 } } | f ( w ) - f ^ { \mathsf { B N } } ( w ) | < \frac { \varepsilon } { 3 \mathopen { } \mathclose \bgroup \left\| S \aftergroup \egroup \right\| } .
$$

Let’s take $n \in \mathbb { N }$ such that $1 / ( n + 1 ) < \delta$ . Then,

$$
\forall v \in K : \ R v , R _ { n } ^ { \mathtt { B N } } v \in K _ { 1 } \ \mathrm { a n d } \ | R v - R _ { n } ^ { \mathtt { B N } } v | < { \frac { 1 } { n + 1 } } < \delta \ \Rightarrow \ | f ( R v ) - f ( R _ { n } ^ { \mathtt { B N } } v ) | < { \frac { \varepsilon } { 3 \left\| S \right\| } } .
$$

Finally, since $f ^ { \mathsf { B N } } ( K _ { 1 } )$ is compact, by Lemma B.12, there exist $S ^ { \mathsf { B N } } \colon \mathbb { R } ^ { J ^ { \prime } }  \mathcal { U }$ such that

$$
\operatorname* { s u p } _ { w \in f ^ { \mathsf { B N } } ( K _ { 1 } ) } \| S w - S ^ { \mathsf { B N } } w \| _ { \mathcal { U } } < \frac { \varepsilon } { 3 } .
$$

Therefore, for every $v \in K$ we have

$$
\begin{array} { r l } & { \left\| S ( f ( R v ) ) - S ^ { \mathrm { B N } } ( f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) ) \right\| _ { \boldsymbol { u } } \leq \left\| S ( f ( R v ) ) - S ( f ( R _ { n } ^ { \mathrm { B N } } v ) ) \right\| _ { \boldsymbol { u } } + \left\| S ( f ( R _ { n } ^ { \mathrm { B N } } v ) ) - S ( f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) ) \right\| _ { \boldsymbol { u } } } \\ & { \qquad + \left\| S ( f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) ) - S ^ { \mathrm { B N } } ( f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) ) \right\| _ { \boldsymbol { u } } } \\ & { \leq \| S \| | f ( R v ) - f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) | + \| S \| | f ( R _ { n } ^ { \mathrm { B N } } v ) - f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) | } \\ & { \qquad + \left\| S ( f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) ) - S ^ { \mathrm { B N } } ( f ^ { \mathrm { B N } } ( R _ { n } ^ { \mathrm { B N } } v ) ) \right\| _ { \boldsymbol { u } } } \\ & { \qquad < \frac { \varepsilon } { 3 } + \frac { \varepsilon } { 3 } + \frac { \varepsilon } { 3 } = \varepsilon . } \end{array}
$$

In the end, for every $v \in K$

$$
\big \| \mathcal { G } ( v ) - S ^ { \mathsf { B N } } ( f ^ { \mathsf { B N } } ( R _ { n } ^ { \mathsf { B N } } v ) ) \big \| _ { M } \leq \| \mathcal { G } ( v ) - S ( f ( R v ) ) \| _ { \mathcal { U } } + \big \| S ( f ( R v ) ) - S ^ { \mathsf { B N } } ( f ^ { \mathsf { B N } } ( R _ { n } ^ { \mathsf { B N } } v ) ) \big \| _ { \mathcal { U } } < 2 \varepsilon .
$$

Now in order to conclude the proof, it is sufficient to lift the operators $R ^ { \mathsf { B N } }$ and $S ^ { \mathsf { B N } }$ to Lebesgue spaces, as described in Remark B.9 and Remark B.11, and the function $f ^ { \mathsf { B N } }$ to Bregman neural operator as described in Lemma B.10 and recognize that

$$
\begin{array} { r } { S ^ { \mathsf { B N } } \circ \mathcal { N } ^ { \mathsf { B N } } \circ \mathcal { R } _ { n } ^ { \mathsf { B N } } = S ^ { \mathsf { B N } } \circ f ^ { \mathsf { B N } } \circ R _ { n } ^ { \mathsf { B N } } . } \end{array}
$$

Indeed, for every $v \in { \mathcal { A } }$ , we have

$$
\mathcal { S } ^ { \mathsf { B N } } ( \mathcal { N } ^ { \mathsf { B N } } ( \mathcal { R } _ { n } ^ { \mathsf { B N } } v ) ) = \mathcal { S } ^ { \mathsf { B N } } ( \mathcal { N } ^ { \mathsf { B N } } ( ( R _ { n } ^ { \mathsf { B N } } v ) \mathbb { 1 } _ { D } ) ) = \mathcal { S } ^ { \mathsf { B N } } ( f ^ { \mathsf { B N } } ( ( R _ { n } ^ { \mathsf { B N } } v ) ) \mathbb { 1 } _ { D } ) = { S } ^ { \mathsf { B N } } ( f ^ { \mathsf { B N } } ( ( R _ { n } ^ { \mathsf { B N } } v ) ) ) .
$$

The statement follows.

□

## C. Experimental Settings

We adopt the same experimental setting as in the PDEBench repository (Takamoto et al., 2022). For the sake of information, we recall the considered problems and PDEs and the specific settings we consider when appropriate. The learning procedure used is presented at the end of this section.

## C.1. 1D Advection Equation

The advection equation is a linear Partial Differential Equation (PDE) modeling the transport of a fluid quantity $u ,$ namely its velocity field, defined by the following equation:

$$
\partial _ { t } u ( x , t ) + \beta \partial _ { x } u ( x , t ) = 0 , x \in ( 0 , 1 ) , t \in ( 0 , 2 ] ,\tag{20}
$$

$$
u ( x , 0 ) = u _ { 0 } ( x ) , x \in ( 0 , 1 ) ,\tag{21}
$$

with $\beta$ a constant advection speed. Note that this system admits an exact solution: $u ( t , x ) = u _ { 0 } ( x - \beta t )$

For this dataset, we follow the setting given in Takamoto et al. (2022), Section D.1 by taking $\beta = 0 . 4$ . We learn the mapping between the value of the field at $t = 0 ( u ( x , 0 ) )$ and the value at time $t = 2 \left( u ( x , 2 ) \right)$ ), i.e. we learn the mapping between the first and the last temporal value of each sample.

## C.2. 1D Burgers Equation

The Burgers’ equation is a PDE describing the nonlinear advection and diffusion of a velocity field, defined as follows:

$$
\partial _ { t } u ( x , t ) + \partial _ { x } ( u ^ { 2 } ( x , t ) / 2 ) = \nu / \pi \partial _ { x x } u ( x , t ) , \mathrm { } x \in ( 0 , 1 ) , t \in ( 0 , 2 ] ,\tag{22}
$$

$$
u ( x , 0 ) = u _ { 0 } ( x ) , x \in ( 0 , 1 ) ,\tag{23}
$$

where $\nu$ is the diffusion coefficient, which is assumed to be constant in this dataset.

We follow again the setup presented in Takamoto et al. (2022), section D.2, with $\nu = 0 . 0 0 1$ . As in the previous dataset, we learn the mapping from the field at $t = 0$ as input to the field at $t = 2$ as target.

## C.3. 1D Compressible Navier-Stokes Equations (1D NS)

The compressible Navier-Stokes equations describe the motion of viscous fluids that can change in density due to compression or expansion. This can be described through the following partial differential equations:

$$
\partial _ { t } \sigma + \partial _ { x } \cdot ( \sigma \mathbf { u } ) = 0 ,\tag{24}
$$

$$
\begin{array} { r } { \sigma ( \partial _ { t } \mathbf { u } + \mathbf { u } \cdot \partial _ { x } \mathbf { u } ) = - \partial _ { x } p + \eta \triangle \mathbf { u } + ( \zeta + \eta / 3 ) \partial _ { x x } \mathbf { u } ) , } \end{array}\tag{25}
$$

$$
\partial _ { t } ( \epsilon + \sigma v ^ { 2 } / 2 ) + \partial _ { x } \cdot [ ( p + \epsilon + \sigma v ^ { 2 } / 2 ) { \bf u } - { \bf u } \cdot \sigma ^ { \prime } ] = { \bf 0 } ,\tag{26}
$$

where $\sigma$ is the mass density, $\mathbf { u } = \mathbf { u } ( \mathbf { x } , \mathbf { t } )$ is the fluid velocity, $p$ is the gas pressure, ϵ is an internal energy described by the equation of state, $\sigma ^ { \prime }$ is the viscous stress tensor, and η and $\zeta$ are shear and bulk viscosity, respectively.

In our experiments, we consider the setup introduced in Takamoto et al. (2022), Section D.5, fixing $\eta = 1 0 ^ { - 8 } , \zeta = 1 0 ^ { - 8 }$ and out-going boundary conditions. We learn the mapping of the velocity v from time $t = 1 0$ as input to time $t = 1 1$ as target. For this dataset, we added a symmetrical padding preprocessing to replicate periodic boundary conditions (as prescribed in the original FNO code (Li et al., 2021a)).

## C.4. 2D Incompressible Navier-Stokes Equations (2D NS)

We also consider a dataset from the 2D Navier-Stokes equation for a viscous, incompressible fluid in vorticity form on the unit torus (Li et al., 2021a) defined as follows:

$$
\begin{array} { r l r l } { \partial _ { t } w ( x , t ) + u ( x , t ) \cdot \nabla w ( x , t ) = \nu \Delta w ( x , t ) + f ( x ) , } & { } & { x \in ( 0 , 1 ) ^ { 2 } , t \in ( 0 , T _ { f i n a l } ] } \\ { \nabla \cdot u ( x , t ) = 0 , } & { } & { x \in ( 0 , 1 ) ^ { 2 } , t \in ( 0 , T _ { f i n a l } ] } \\ { w ( x , 0 ) = w _ { 0 } ( x ) , } & { } & { x \in ( 0 , 1 ) ^ { 2 } } \end{array}\tag{27}
$$

with u is the 2D velocity field, $w = \nabla \times \iota$ is the vorticity, $w _ { 0 } : ( 0 , 1 ) ^ { 2 } ; \to \mathbb { R }$ is the initial vorticity function, $\nu \in \mathbb { R } _ { + }$ is the viscosity coefficient, and $f : ( 0 , 1 ) ^ { 2 } \to$ R is the forcing function.

We follow the setup introduced in Li et al. (2021a), Section A.3.3, with $\nu = 1 0 ^ { - 3 }$ and $\nu = 1 0 ^ { - 4 }$ . We learn the mapping of the velocity field v from sample time $t = 1 0 \ : \mathrm { t o } \ : t = 4 5$ for $\nu = 1 0 ^ { - 3 }$ and from $t = 1 0 \mathrm { t o } t = 1 5$ for $\nu = 1 0 ^ { - 4 }$

## C.5. Darcy Flow

We consider a dataset based on the steady state of the 2D Darcy Flow equation on the unit square, representing the flow through porous media and defined as follows:

$$
\begin{array} { c c } { { - \nabla ( a ( x ) \nabla u ( x ) ) = f ( x ) , } } & { { x \in ( 0 , 1 ) ^ { 2 } , } } \\ { { \ } } & { { \nonumber } } \\ { { u ( x ) = 0 , } } & { { x \in \partial ( 0 , 1 ) ^ { 2 } . } } \end{array}\tag{28}
$$

We follow the setup described in Takamoto et al. (2022), Section D.4, with f(x) fixed to the constant $\beta = 0 . 1$

## C.6. FNO Baselines

In this section, we further detail the FNO improvements considered as baselines.

F-FNO (Tran et al., 2023). The factorized FNO is a particularly relevant baseline for comparison, as it i) incorporates skip-like connections that share similarities with our additional $\sigma ^ { - 1 }$ term and ii) also seeks to enable the development of deeper FNO architectures. We consider the best-performing F-FNO model (as identified by its authors), trained using our optimization strategy and adapted to our specific learning task. It is important to note that F-FNO was originally designed for predicting mappings between multiple consecutive time steps (e.g., from t to t + 1) and it offers the option to rely on techniques such as the Markov assumption and teacher forcing. Since our task involves predicting the final state directly from the initial conditions, those techniques are not appropriate, and thus we did not include them in the implementation. Additionally, we have found that the original optimization strategy proposed by the F-FNO authors (AdamW with cosine annealing, noise injection and input normalization) did not perform well on our tasks, so we also employed the optimization strategy detailed in Appendix C.6.

ResFNO. Moreover, to isolate the impact of residual connections from the broader structural modifications introduced by F-FNO, we have also implemented and compared a ResNet-inspired variant of FNO, referred to as ResFNO. We did this to better understand the role of the residual connection.

## C.7. Learning procedure

Models are trained using the Adam optimizer with a constant learning rate, a batch size of 128 for 1D problems (resp. 16 for 2D problems), a maximum of 2000 epochs and an early stopping strategy with patience of 250 epochs and $\delta = 1 0 ^ { - 3 }$ . The learning rate is validated on a grid of multiple values equally spaced in logarithmic scale. If not mentioned otherwise, we use 8000 (resp. 1000) training samples for 1D (resp. 2D) problems, and 1000 samples each for validation and testing. Al results are averaged over four random splittings.

Experiments have been made on an internal clusters of GPUs with memory from 10Go to 45Go. All the experiments can be achieved with GPUs with a memory of 10Go, except for models with 32 or 64 layers which require at least a memory of 24Go.

A summary of the experimental setting along with some learning hyperparameter is detailed in Table 2.

uˆ<sup>mResF</sup> <sup>NO</sup>  
Table 2: Experimental settings.
<table><tr><td>Dataset</td><td>Number of modes</td><td>Batch size</td><td>Width</td><td>Tinit</td><td>Tfinal</td><td>Train samples</td></tr><tr><td>1D Advection</td><td>[16]</td><td>128</td><td>64</td><td>0</td><td>200</td><td>8000</td></tr><tr><td>1D Burgers</td><td>[16]</td><td>128</td><td>64</td><td>0</td><td>200</td><td>8000</td></tr><tr><td>1D NS</td><td>[16]</td><td>128</td><td>64</td><td>10</td><td>11</td><td>8000</td></tr><tr><td>2D NS  $( 1 0 ^ { - 4 } )$ </td><td>[12,12]</td><td>16</td><td>32</td><td>10</td><td>15</td><td>1000</td></tr><tr><td>2D NS  $( 1 0 ^ { - 3 } )$ </td><td>[12,12]</td><td>16</td><td>32</td><td>10</td><td>45</td><td>1000</td></tr><tr><td>2D Darcy</td><td>[12,12]</td><td>16</td><td>32</td><td>-</td><td>-</td><td>1000</td></tr></table>

## D. Additional Results

## D.1. Comparison of Predictions

In this section, we visually inspect to what extent the prediction made by FNO, ResFNO and BFNO is close to the ground truth. We provide three examples on the Navier Stokes dataset with viscosity $1 0 ^ { - 4 }$ where we have selected the best performing models.

![](images/95f9447def8c447e63e2ab50d3e783f602993fe4d38bd658d74c2b7a62a9915e.jpg)  
uˆ<sup>mF</sup> <sup>NO</sup>  
(a) Predictions

![](images/03f6fcc483e44a0360adcb25c3fc5ffc2532f5216b43ea109416ff4fb2eaf2e2.jpg)  
(b) Residuals  
Figure 8: Predictions and residuals for Navier Stokes $1 0 ^ { - 4 }$

## D.2. Extension to Wavelet Neural Operators

We also extended our experiments to Wavelet Neural Operators (WNO). In Table 3 is reported the comparison between standard WNO and the Bregman version BWNO. We can observe similar results as Fourier models, where our models outperform the standard models and are able to gain performance when increasing the number of layers. Furthermore, even with gradient clipping, 32 and 64-layer standard models could not converge during training, leading to 100% relative error rate. Further analysis shows that this divergence can be linked with the high error rates on low frequencies and boundary conditions.

## D.3. Detailed Analysis of the Prediction Performance

In the same spirit of Takamoto et al. (2022), we include several metrics providing a deeper understanding of the models behavior, including relative mean squared error on the boundary (rMSE) as well as in the low, mid, and high frequency bands (fRMSE low, fRMSE mid, fRMSE high). Results are provided in Table 4.

Table 3: Relative error of WNO and BWNO models on benchmark PDEs.
<table><tr><td rowspan="2"></td><td colspan="2">1D Advection</td><td colspan="2">1D Burgers</td><td colspan="2">1D NS</td></tr><tr><td>WNO</td><td>BWNO</td><td>WNO</td><td>BWNO</td><td>WNO</td><td>BWNO</td></tr><tr><td>4 layers</td><td> $3 . 0 \pm 0 . 0 \%$ </td><td> $2 . 8 \pm 0 . 2 \%$ </td><td> $2 1 . 5 \pm 0 . 5 \%$ </td><td> $2 1 . 3 \pm 0 . 4 \%$ </td><td> $5 9 . 2 \pm 0 . 6 \%$ </td><td> ${ \pm } 8 . 3 \pm 0 . 6 \%$ </td></tr><tr><td>8 layers</td><td> $2 . 5 \pm 0 . 1 \%$ </td><td> $2 . 1 \pm 0 . 1 \%$ </td><td> $1 9 . 1 \pm 0 . 6 \%$ </td><td> $1 7 . 9 \pm 0 . 6 \%$ </td><td> $5 9 . 0 \pm 0 . 6 \%$ </td><td> ${ \bf 5 8 . 0 \pm 0 . 6 \% }$ </td></tr><tr><td>16 layers</td><td> $3 . 9 \pm 0 . 8 \%$ </td><td> $2 . 0 \pm 0 . 2 \%$ </td><td> $1 9 . 7 \pm 0 . 5 \%$ </td><td> ${ \bf 1 6 . 4 \pm 0 . 3 \% }$ </td><td> $6 1 . 1 \pm 0 . 6 \%$ </td><td> ${ \pm } 7 . 6 \pm 0 . 7 \%$ </td></tr><tr><td>32 layers</td><td> $10 0 \pm 0 \%$ </td><td> $1 . 9 \pm 0 . 1 \%$ </td><td> $10 0 \pm 0 \%$ </td><td> ${ \bf 1 6 . 4 \pm 0 . 5 \% }$ </td><td> $10 0 \pm 0 \%$ </td><td> ${ \pm } 7 . 2 \pm 0 . 6 \%$ </td></tr><tr><td>64 layers</td><td> $10 0 \pm 0 \%$ </td><td> $1 . 8 \pm 0 . 2 \%$ </td><td> $10 0 \pm 0 \%$ </td><td> ${ \bf 1 6 . 1 \pm 0 . 4 \% }$ </td><td> $10 0 \pm 0 \%$ </td><td> ${ \pm } 7 . 5 \pm 0 . 6 \%$ </td></tr></table>

Table 4: Additional comparison of the performance in terms of relative $\ell ^ { 2 }$ error (rMSE), relative mean squared error on the boundary (rMSE) as well as in the low, mid and high frequency bands (fRMSE low, fRMSE mid, fRMSE high).
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2> $T = 4$ </td><td rowspan=1 colspan=2> $T = 8$ </td><td rowspan=1 colspan=2> $T = 1 6$ </td></tr><tr><td rowspan=1 colspan=1>PDE</td><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>BFNO</td><td rowspan=1 colspan=1>FNO</td><td rowspan=1 colspan=1>BFNO</td><td rowspan=1 colspan=1>FNO</td><td rowspan=1 colspan=1>BFNO</td><td rowspan=1 colspan=1>FNO</td></tr><tr><td rowspan=5 colspan=1>1Decett ton</td><td rowspan=1 colspan=1>rMSE</td><td rowspan=1 colspan=1> $\mathbf { 1 . 5 5 \cdot 1 0 ^ { - 2 } }$ </td><td rowspan=1 colspan=1> $2 . 4 3 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 4 5 \cdot 1 0 ^ { - 2 } }$ </td><td rowspan=1 colspan=1> $3 . 2 2 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1>1.43 · 10−2</td><td rowspan=1 colspan=1> $4 . 3 8 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan=1 colspan=1>bRMSE</td><td rowspan=1 colspan=1> $\overline { { \bf 9 . 0 4 \cdot 1 0 ^ { - 2 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 2 1 \cdot 1 0 ^ { - 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 8 . 2 8 \cdot 1 0 ^ { - 2 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 5 1 \cdot 1 0 ^ { - 1 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 7 . 8 2 \cdot 1 0 ^ { - 2 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 . 4 3 \cdot 1 0 ^ { - 1 } } }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE low</td><td rowspan=1 colspan=1>4.56· 10−6</td><td rowspan=1 colspan=1> $\overline { { 9 . 3 8 \cdot 1 0 ^ { - 6 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf { 4 . 4 5 \cdot 1 0 ^ { - 6 } } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 1 4 \cdot 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1> $\mathbf { 4 . 2 1 \cdot 1 0 ^ { - 6 } }$ </td><td rowspan=1 colspan=1> $1 . 9 3 \cdot 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE mid</td><td rowspan=1 colspan=1>3.89 . $\overline { { { \bf 1 0 } ^ { - 6 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 6 . 8 1 \cdot 1 0 ^ { - 6 } } }$ </td><td rowspan=1 colspan=1> $\mathbf { 3 . 3 9 \cdot 1 0 ^ { - 6 } }$ </td><td rowspan=1 colspan=1> $8 . 8 4 \cdot 1 0 ^ { - 6 }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf { 3 . 6 1 . 1 0 ^ { - 6 } } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 2 7 \cdot 1 0 ^ { - 5 } } }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE high</td><td rowspan=1 colspan=1>3.15 · 10−7</td><td rowspan=1 colspan=1> $4 . 8 7 \cdot 1 0 ^ { - 7 }$ </td><td rowspan=1 colspan=1> $\mathbf { 2 . 8 9 \cdot 1 0 ^ { - 7 } }$ </td><td rowspan=1 colspan=1> $5 . 9 6 \cdot 1 0 ^ { - 7 }$ </td><td rowspan=1 colspan=1> $\mathbf { 2 . 7 6 \cdot 1 0 ^ { - 7 } }$ </td><td rowspan=1 colspan=1>8.08 · 10−7</td></tr><tr><td rowspan=5 colspan=1>1Ie rs</td><td rowspan=1 colspan=1>rMSE</td><td rowspan=1 colspan=1> $\mathbf { 8 . 2 4 \cdot 1 0 ^ { - 2 } }$ </td><td rowspan=1 colspan=1> $8 . 2 8 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 5 . 8 3 \cdot 1 0 ^ { - 2 } }$ </td><td rowspan=1 colspan=1> $8 . 1 6 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 4 . 6 7 \cdot 1 0 ^ { - 2 } }$ </td><td rowspan=1 colspan=1>7.92 · 10−2</td></tr><tr><td rowspan=1 colspan=1>bRMSE</td><td rowspan=1 colspan=1>3.83 · 10−1</td><td rowspan=1 colspan=1>3.69 · 10−1</td><td rowspan=1 colspan=1>2.41 · 10−¹</td><td rowspan=1 colspan=1> $3 . 6 5 \cdot 1 0 ^ { - }$ -1</td><td rowspan=1 colspan=1> $\overline { { { \bf 1 . 8 5 \cdot 1 0 ^ { - 1 } } } }$ </td><td rowspan=1 colspan=1>3.72 · 10−1</td></tr><tr><td rowspan=1 colspan=1>fRMSE low</td><td rowspan=1 colspan=1>5.18 · 10−5</td><td rowspan=1 colspan=1> $\mathbf { 4 . 8 8 \cdot 1 0 ^ { - 5 } }$ 5</td><td rowspan=1 colspan=1> $\mathbf { 3 . 0 1 \cdot 1 0 ^ { - } }$ 5</td><td rowspan=1 colspan=1> $4 . 5 2 \cdot 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { 2 . 3 7 \cdot 1 0 ^ { - 5 } }$ </td><td rowspan=1 colspan=1>4.88 · 10−5</td></tr><tr><td rowspan=1 colspan=1>fRMSE mid</td><td rowspan=1 colspan=1> $\mathbf { 3 . 4 1 \cdot 1 0 ^ { - } }$ -5</td><td rowspan=1 colspan=1> $3 . 4 4 \cdot 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1>2.52 · 10-5</td><td rowspan=1 colspan=1>3.61 · 10−5</td><td rowspan=1 colspan=1> $\mathbf { 2 . 0 1 \cdot 1 0 ^ { - 5 } }$ 一</td><td rowspan=1 colspan=1> $3 . 2 9 \cdot 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE high</td><td rowspan=1 colspan=1> $\frac { \phantom { - } } { \mathbf { 1 . 1 7 \cdot 1 0 ^ { - 6 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 3 2 \cdot 1 0 ^ { - 6 } } }$ </td><td rowspan=1 colspan=1>1.01·10−6</td><td rowspan=1 colspan=1> $\overline { { 1 . 3 3 \cdot 1 0 ^ { - 6 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 8 . 7 1 \cdot 1 0 ^ { - 7 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 2 9 \cdot 1 0 ^ { - 6 } } }$ </td></tr><tr><td rowspan=5 colspan=1>1NNS ( I80)</td><td rowspan=1 colspan=1>rMSE</td><td rowspan=1 colspan=1> $\mathbf { 4 . 9 1 \cdot 1 0 ^ { - 1 } }$ </td><td rowspan=1 colspan=1> $5 . 0 5 \cdot 1 0 ^ { - 1 }$ </td><td rowspan=1 colspan=1> $\mathbf { 4 . 9 0 \cdot 1 0 ^ { - 1 } }$ </td><td rowspan=1 colspan=1> $5 . 2 2 \cdot 1 0 ^ { - 1 }$ </td><td rowspan=1 colspan=1> $\mathbf { 4 . 8 6 \cdot 1 0 ^ { - 1 } }$ </td><td rowspan=1 colspan=1>5.35 · 10−1</td></tr><tr><td rowspan=1 colspan=1>bRMSE</td><td rowspan=1 colspan=1> $\mathbf { 2 . 1 6 \cdot 1 0 ^ { 0 } }$ </td><td rowspan=1 colspan=1> $2 . 4 4 \cdot 1 0 ^ { 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 2 . 0 6 \cdot 1 0 ^ { 0 } }$ </td><td rowspan=1 colspan=1> $2 . 8 5 \cdot 1 0 ^ { 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 9 5 \cdot 1 0 ^ { 0 } }$ </td><td rowspan=1 colspan=1> $3 . 1 8 \cdot 1 0 ^ { 0 }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE low</td><td rowspan=1 colspan=1>2.65· 10-4</td><td rowspan=1 colspan=1>2.78 · 10−4</td><td rowspan=1 colspan=1> $\mathbf { 2 . 6 5 \cdot 1 0 ^ { - 4 } }$ </td><td rowspan=1 colspan=1>2.86 · 10−4</td><td rowspan=1 colspan=1> $\mathbf { 2 . 5 7 \cdot 1 0 ^ { - 4 } }$ </td><td rowspan=1 colspan=1> $2 . 9 1 \cdot 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE mid</td><td rowspan=1 colspan=1>2.19 · 10−4</td><td rowspan=1 colspan=1>2.26 · 10−4</td><td rowspan=1 colspan=1>2.18· 10−4</td><td rowspan=1 colspan=1>2.34 · 10−4</td><td rowspan=1 colspan=1> $\mathbf { 2 . 1 3 \cdot 1 0 ^ { - } }$ 4</td><td rowspan=1 colspan=1>2.43 · 10−4</td></tr><tr><td rowspan=1 colspan=1>fRMSE high</td><td rowspan=1 colspan=1>1.12 · 10−5</td><td rowspan=1 colspan=1>1.11·10−5</td><td rowspan=1 colspan=1>1.11·10−5</td><td rowspan=1 colspan=1>1.13 · 10−5</td><td rowspan=1 colspan=1> $1 . 2 1 \cdot 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 1 3 \cdot 1 0 ^ { - 5 } }$ </td></tr><tr><td rowspan=5 colspan=1>2Darcy*</td><td rowspan=1 colspan=1>rMSE</td><td rowspan=1 colspan=1>9.99 · 10−¹</td><td rowspan=1 colspan=1> $1 . 0 0 \cdot 1 0 ^ { 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 9 . 9 6 \cdot 1 0 ^ { - 1 } }$ </td><td rowspan=1 colspan=1> $1 . 0 1 \cdot 1 0 ^ { 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 0 \cdot 1 0 ^ { 0 } }$ </td><td rowspan=1 colspan=1> $1 . 0 2 \cdot 1 0 ^ { 0 }$ </td></tr><tr><td rowspan=1 colspan=1>bRMSE</td><td rowspan=1 colspan=1>1.45 · 10−2</td><td rowspan=1 colspan=1>1.47 · 10−2</td><td rowspan=1 colspan=1> $\frac { \mathbf { 1 . 4 3 \cdot 1 0 ^ { - 2 } } } { \mathbf { 1 . 4 3 \cdot 1 0 ^ { - 2 } } }$ </td><td rowspan=1 colspan=1>1.52 · 10−2</td><td rowspan=1 colspan=1> $1 . 4 1 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1>1.40 · 10−2</td></tr><tr><td rowspan=1 colspan=1>fRMSE low</td><td rowspan=1 colspan=1>6.41 · 10-4</td><td rowspan=1 colspan=1> $6 . 4 4 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $\mathbf { 6 . 3 5 \cdot 1 0 ^ { - 4 } }$ </td><td rowspan=1 colspan=1> $6 . 4 5 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $\mathbf { 6 . 3 7 \cdot 1 0 ^ { - 4 } }$ </td><td rowspan=1 colspan=1>6.52 · 10−4</td></tr><tr><td rowspan=1 colspan=1>fRMSE mid</td><td rowspan=1 colspan=1> $\overline { { 3 . 2 9 \cdot 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1> $\mathbf { 3 . 2 1 \cdot 1 0 ^ { - 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 . 2 3 \cdot 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1> $\mathbf { 3 . 1 7 \cdot 1 0 ^ { - 5 } }$ </td><td rowspan=1 colspan=1> $\mathbf { 3 . 1 5 \cdot 1 0 ^ { - 5 } }$ </td><td rowspan=1 colspan=1>3.23 · 10−5</td></tr><tr><td rowspan=1 colspan=1>fRMSE high</td><td rowspan=1 colspan=1> $\overline { { { \bf { 1 . 6 8 \cdot 1 0 ^ { - 6 } } } } }$ </td><td rowspan=1 colspan=1> $1 . 9 9 \cdot 1 0 ^ { - 6 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 8 6 \cdot 1 0 ^ { - 6 } }$ </td><td rowspan=1 colspan=1> $1 . 9 9 \cdot 1 0 ^ { - 6 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 7 3 \cdot 1 0 ^ { - 6 } }$ </td><td rowspan=1 colspan=1>2.01 · 10−6</td></tr><tr><td rowspan=5 colspan=1>2N (( %)</td><td rowspan=1 colspan=1>rMSE</td><td rowspan=1 colspan=1>5.49 · 10−1</td><td rowspan=1 colspan=1> $5 . 6 5 \cdot 1 0 ^ { - 1 }$ </td><td rowspan=1 colspan=1> $\mathbf { 5 . 2 3 \cdot 1 0 ^ { - 1 } }$ </td><td rowspan=1 colspan=1> $5 . 4 8 \cdot 1 0 ^ { - 1 }$ </td><td rowspan=1 colspan=1> $\mathbf { 5 . 4 1 \cdot 1 0 ^ { - } }$ 1</td><td rowspan=1 colspan=1>5.46 · 10−1</td></tr><tr><td rowspan=1 colspan=1>bRMSE</td><td rowspan=1 colspan=1> $\mathbf { 2 . 8 8 \cdot 1 0 ^ { - 2 } }$ </td><td rowspan=1 colspan=1> $2 . 9 7 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf { 2 . 7 5 \cdot 1 0 ^ { - 2 } } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 . 9 2 \cdot 1 0 ^ { - 2 } } }$ </td><td rowspan=1 colspan=1> $\mathbf { 2 . 8 3 \cdot 1 0 ^ { - } }$ -2</td><td rowspan=1 colspan=1> $\overline { { 2 . 8 4 \cdot 1 0 ^ { - 2 } } }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE low</td><td rowspan=1 colspan=1> $\mathbf { 5 . 8 2 \cdot 1 0 ^ { - } }$ -4</td><td rowspan=1 colspan=1> $5 . 9 8 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $\mathbf { 5 . 5 9 \cdot 1 0 ^ { - 4 } }$ </td><td rowspan=1 colspan=1> $5 . 7 6 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $5 . 7 5 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf { 5 . 7 0 \cdot 1 0 ^ { - 4 } } } } }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE mid</td><td rowspan=1 colspan=1> $1 . 4 6 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 1 . 4 1 \cdot 1 0 ^ { - 4 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathbf { 1 . 1 3 \cdot 1 0 ^ { - 4 } } } }$ </td><td rowspan=1 colspan=1> $1 . 2 7 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $1 . 1 1 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 1 . 0 7 \cdot 1 0 ^ { - 4 } } } }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE high</td><td rowspan=1 colspan=1> $1 . 3 1 \cdot 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1> $\mathbf { 8 . 6 9 \cdot 1 0 } ^ { \cdot }$ -6</td><td rowspan=1 colspan=1> $9 . 8 7 \cdot 1 0 ^ { - 6 }$ </td><td rowspan=1 colspan=1> $\mathbf { 8 . 3 7 \cdot 1 0 ^ { - } }$ -6</td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 1 \cdot 1 0 ^ { - } }$ -5</td><td rowspan=1 colspan=1> $1 . 1 4 \cdot 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan=5 colspan=1>2DS ()</td><td rowspan=1 colspan=1>rMSE</td><td rowspan=1 colspan=1> $\mathbf { 1 . 3 1 \cdot 1 0 ^ { 0 } }$ </td><td rowspan=1 colspan=1> $1 . 3 4 \cdot 1 0 ^ { 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 2 5 \cdot 1 0 ^ { 0 } }$ </td><td rowspan=1 colspan=1> $1 . 3 8 \cdot 1 0 ^ { 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 2 3 \cdot 1 0 ^ { 0 } }$ </td><td rowspan=1 colspan=1> $1 . 4 3 \cdot 1 0 ^ { 0 }$ </td></tr><tr><td rowspan=1 colspan=1>bRMSE</td><td rowspan=1 colspan=1> $\overline { { { \bf 7 . 0 3 \cdot 1 0 ^ { - 2 } } } }$ </td><td rowspan=1 colspan=1> $7 . 1 6 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 6 . 7 4 \cdot 1 0 ^ { - 2 } } } }$ </td><td rowspan=1 colspan=1> $7 . 4 5 \cdot 1 0 ^ { - 2 }$ </td><td rowspan=1 colspan=1> $\mathbf { 6 . 6 8 \cdot 1 0 ^ { - 2 } }$ </td><td rowspan=1 colspan=1> $7 . 7 3 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE low</td><td rowspan=1 colspan=1>9.54 · 10−4</td><td rowspan=1 colspan=1> $1 . 0 0 \cdot 1 0 ^ { - 3 }$ </td><td rowspan=1 colspan=1> $\mathbf { 9 . 4 2 \cdot 1 0 ^ { - 4 } }$ </td><td rowspan=1 colspan=1> $9 . 9 4 \cdot 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1>9.22 · 10−4</td><td rowspan=1 colspan=1> $1 . 1 1 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE mid</td><td rowspan=1 colspan=1> $\mathbf { 7 . 3 4 \cdot 1 0 ^ { - 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 . 5 1 \cdot 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf { 6 . 9 3 \cdot 1 0 } ^ { - 4 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 7 . 8 9 \cdot 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1>6.83 · 10-4</td><td rowspan=1 colspan=1> $7 . 9 8 \cdot 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan=1 colspan=1>fRMSE high</td><td rowspan=1 colspan=1> $\overline { { { \bf 1 . 2 9 \cdot 1 0 ^ { - 4 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 3 3 \cdot 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf { 1 . 2 2 \cdot 1 0 ^ { - 4 } } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 4 3 \cdot 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { \bf 1 . 2 1 \cdot 1 0 ^ { - 4 } } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 . 4 8 \cdot 1 0 ^ { - 4 } } }$ </td></tr></table>

## D.4. Impact of the Activation Function

A limitation of our framework is the fact that it requires Bregman variants (such as BFNO) to have a strictly monotonic activation function, which excludes a few functions such as ReLU. This justifies why in our experiments we used Softplus as a surrogate of ReLU. On the contrary, for classical neural operators within our framework, the activation function only needs to be monotonic, not strictly monotonic. Therefore, ReLU is still valid and can be used.

As a thought experiment, we also implemented BFNO with ReLU and evaluated it on the 2D Navier-Stokes dataset $\left( \nu = 1 0 ^ { - 4 } \right)$ . Table 5 shows that BFNO with ReLU achieves comparable or better performance than Softplus for the same number of layers. However, the best results are the same (i.e., 12.2% for 16 layers).

<table><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>4 layers</td><td rowspan=1 colspan=1>8 layers</td><td rowspan=1 colspan=1>16 layers</td></tr><tr><td rowspan=1 colspan=1>FNO (ReLU)</td><td rowspan=1 colspan=1> $\overline { { 1 3 . 5 \pm 0 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 3 . 0 \pm 0 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 2 . 6 \pm 0 . 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>BFNO (Softplus)</td><td rowspan=1 colspan=1> $\overline { { 1 3 . 7 \pm 0 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 2 . 6 \pm 0 . 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 2 . 2 \pm { \bf 0 . 1 } } }$ </td></tr><tr><td rowspan=1 colspan=1>BFNO (ReLU)</td><td rowspan=1 colspan=1> $\overline { { 1 3 . 4 \pm 0 . 2 } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 2 . 2 \pm { 0 . 2 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 2 . 2 \pm { \bf 0 . 1 } } }$ </td></tr></table>

Table 5: Comparison BFNO with Softplus or ReLU.

## D.5. Impact of Batch Normalization

For all the experiments presented in the previous sections, we relied on the latest available version of the FNO implementation, which does not include Batch Normalization (BN), while it was used in the original FNO paper (Li et al., 2021a). We note that the original FNO code was removed from the GitHub repository by its author (i.e., the ’master’ branch was deleted). While we retrieved an earlier version of the code, we observed that BN was implemented in the initial commit but was subsequently removed in a later commit titled "remove unnecessary batchnorm", suggesting that adding BN layers does no lead to better prediction performance.

To complement our results, we have conducted an experiment with BN for both FNO and BFNO architectures on the 2D Navier-Stokes dataset $( \nu = 1 0 ^ { - 4 } )$ . Results, reported in Table 6, show marginal improvements for 8-layer models (FNO: 13.0% → 12.8%, BFNO: 12.6% → 12.4%) but no consistent benefits for other configurations. This aligns with the conclusion of the recent FNO implementations that removed BN.
<table><tr><td>Architecture</td><td>4 layers</td><td>8 layers</td><td>16 layers</td></tr><tr><td>FNO</td><td> $\overline { { 1 3 . 5 \pm 0 . 1 } }$ </td><td> $\overline { { 1 3 . 0 \pm 0 . 1 } }$ </td><td> $\overline { { 1 2 . 6 \pm 0 . 1 } }$ </td></tr><tr><td> $\mathrm { F N O } + \mathrm { B N }$ </td><td> ${ \bf 1 3 . 5 \pm 0 . 1 }$ </td><td> $1 2 . 8 \pm 0 . 2$ </td><td> $1 2 . 6 \pm 0 . 1$ </td></tr><tr><td>BFNO</td><td> $1 3 . 7 \pm 0 . 1$ </td><td> $1 2 . 6 \pm 0 . 1$ </td><td> $1 2 . 2 \pm { \bf 0 . 1 }$ </td></tr><tr><td> $\mathrm { \Delta B F N O } + \mathrm { \Delta B N }$ </td><td> ${ \bf 1 3 . 5 \pm 0 . 1 }$ </td><td> $1 2 . 4 \pm { \bf 0 . 0 }$ </td><td> $1 2 . 3 \pm 0 . 1$ </td></tr></table>

Table 6: Impact of BatchNormalization (BN) with FNO and BFNO