# Data-driven Koopman mode approximation: A neural power iteration algorithm

Guillaume O. Berger<sup>1</sup>, and Raphael M. Jungers ¨ <sup>1</sup>

Abstract— This paper proposes a novel data-driven algorithm to approximate the dominant eigenfunctions (aka. modes) of the Koopman operator of nonlinear dynamical systems using neural networks. The relevance of learning the dominant Koopman modes is to approximate nonlinear dynamics by linear ones in a lifted space, thereby enabling simplified control and analysis. To fight the curse of dimensionality arising from using expressive templates (here neural networks) for the mode approximation, the proposed method leverages a power-iteration scheme that directly learns the dominant Koopman modes without explicitly constructing the projection of the Koopman operator on the template of functions. Our approach connects to other approaches in the literature that avoid the curse of dimensionality by learning small dictionaries of functions, but differs from them in that we do not require “anti-collapse mechanisms” to ensure that the learned dictionary is expressive enough to approximate the Koopman operator since our power-iteration scheme is designed to converge toward the dominant modes of the projected Koopman operator. The approach is fully datadriven, requiring only sampled state transitions. Theoretical guarantees are provided, showing convergence under increasing sample size and network width (in connection with the neural tangent kernel theorem). Numerical experiments demonstrate that the method achieves accurate and smooth approximations of dominant modes while avoiding the limitations of traditional techniques such as extended dynamic mode decomposition.

## I. INTRODUCTION

Koopman theory [1] offers a powerful alternative to statebased techniques for studying nonlinear dynamical systems

$$
x ^ { + } = F ( x ) .
$$

Rather than studying the evolution of state variables $x \in \mathbb { R } ^ { n }$ the Koopman approach studies the evolution of observables, i.e., functions $f : \mathbb { R } ^ { n } \to \mathbb { C } .$ , under the action of the system. The overarching idea of this approach is that the action of the system on the observables can be represented by a linear operator, called Koopman operator, defined by

$$
K : { \mathcal { F } }  { \mathcal { F } } , \quad ( K f ) ( x ) = f ( F ( x ) ) ,
$$

where $\mathcal { F }$ is a suitable set of observables. This allows to apply techniques from linear operator theory, such as spectral analysis, to study the system, including convergence analysis [2], chaos [3], model order reduction [4], and control [5]; see [6]– [9] for recent surveys.

Another key advantage of the Koopman approach is that it is easily applicable in the data-driven setting. Indeed, the action of the system on an observable $f$ can be approximated by its action on a subset of sample states $\{ x _ { i } \} _ { i = 1 } ^ { N } \subseteq \mathbb { R } ^ { n }$ which gives a straightforward data-driven approximation

$$
( \mathcal { K } f ) ( x _ { i } ) = f ( x _ { i } ^ { + } ) , \quad i = 1 , \ldots , N ,
$$

where $\{ ( x _ { i } , x _ { i } ^ { + } ) \} _ { i = 1 } ^ { N }$ are sample one-step trajectories of the system.

However, the set $\mathcal { F }$ of observables being typically infinitedimensional, Koopman-based techniques usually require projection on a finite-dimensional subspace [10]. More precisely, given a finitely-generated subspace $\mathcal { D } \subseteq \mathcal { F }$ , one studies the projected Koopman operator

$$
K _ { \mathcal { D } } : \mathcal { D }  \mathcal { D } , \quad K _ { \mathcal { D } } = \Pi _ { \mathcal { D } } \circ K ,
$$

where $\Pi _ { \mathcal { D } }$ is a projection on $\mathcal { D } .$ The projection introduces approximation errors that need to be taken into account in the analysis of the system through the projected Koopman operator. This raises the fundamental question of the selection of the projection subspace $\mathcal { D } ,$ which needs to balance computational efficiency and accuracy of the approximation [11].

The dynamic mode decomposition (DMD) [12] approach uses the set of linear observables as space $\mathcal { D } .$ The extended DMD (EDMD) [13] extends the former with a user-provided dictionary of functions (e.g., polynomials of degree $\leq d )$ as basis of $\mathcal { D } .$ The Hankel DMD (aka. time-delay DMD) [14] is a particular case of EDMD where the dictionary is built from a user-provided function $g$ on which the Koopman operator is applied m times.

The reliance on a user-provided dictionary is a significant limitation because good approximations of the Koopman operator typically require expressive dictionaries. For instance, bistable systems have discontinuous dominant eigenmodes of their Koopman operator [2]; thereby requiring polynomials of high degree for good approximations of such modes.

Consequently, in recent years, there has been significant research effort devoted to exploiting the expressive power of (deep) neural networks to parametrize observables [15]–[19]. The main challenge with this approach is that there is no clear basis of the associated space D. The learning problem is therefore formulated as the one of finding a dictionary of neural network observables along with the associated projected Koopman operator:

$$
\operatorname* { m i n } _ { \theta _ { i } \in \mathbb { R } ^ { n _ { \theta } } } \| [ K \phi ^ { \theta _ { 1 } } , \dots , K \phi ^ { \theta _ { m } } ] - [ \phi ^ { \theta _ { 1 } } , \dots , \phi ^ { \theta _ { m } } ] K \| ,\tag{1}
$$

where $\phi ^ { \theta } \in { \mathcal { F } }$ is a neural network parametrized by $\theta \in \mathbb { R } ^ { n _ { \theta } }$ and K is the matrix representation of the projected Koopman operator associated to $\mathcal { D } = \operatorname { s p a n } \{ \phi ^ { \theta _ { 1 } } , \dots , \phi ^ { \theta _ { m } } \}$ . Although attractive for its simplicity, formulation (1) is problematic because of the collapse problem: generically, the dictionary collapses to a trivial dictionary with only a few linearly independent observables $( \mathrm { e . g . , } \phi ^ { \theta _ { i } } = 0$ for all i) [15], [16].

To prevent collapse, state-of-the-art approaches (see refs above) require that the state can be (approximately) retrieved from the encoding $[ \phi ^ { \theta _ { 1 } } ( x ) , \ldots , \phi ^ { \theta _ { m } } ( { \dot { x } } ) ]$ . This is enforced either explicitly by fixing some of the observables $\phi ^ { \theta _ { i } }$ so that x can be retrieved from those observables [15], or implicitly by adding a penalty term in the loss of (1):

$$
\rho \| \mathrm { I d } - \Delta ^ { \theta _ { 0 } } \circ [ \phi ^ { \theta _ { 1 } } , \dots , \phi ^ { \theta _ { m } } ] \| ,
$$

where Id is the identity function and $\Delta ^ { \theta } : \mathbb { C } ^ { m } \to \mathbb { R } ^ { n }$ is a trainable neural network playing the role of the decoder [16]–[19].<sup>1</sup> The drawback of these state-retrieval approaches is that they require a number of observables that grows with the dimension of the system, thereby limiting their use in systems of large dimension. Furthermore, the decoder-based approach requires to train an additional network, which makes training slower and more data-sensitive.

## Contributions

In this work, we propose a novel iterative method, based on power iterations, to learn the dominant modes of the projected Koopman operator. A mode of the projected Koopman operator is an eigenfunction of $\mathbf { i t } ,$ that ${ \mathrm { i s } } ,$ a non-zero observable $g \in \mathcal { D }$ such that $K _ { \mathcal { D } } g = \lambda g$ for some $\lambda \in \mathbb { C }$ called the associated eigenvalue. A mode is dominant if its eigenvalue is large in modulus compared to the other eigenvalues. Power iteration is a numerical iterative algorithm that allows one to approximate the m dominant modes of a linear operator, where m is a user-provided parameter [20].

Conceptually, our approach works as follows: starting from a set of observables $\{ f _ { 1 } , \ldots , f _ { m } \}$ , we apply recursively $K _ { \mathcal { D } }$ to each observable $f _ { i } .$ . Under appropriate assumptions (e.g., separation of the dominant m eigenvalues of $K _ { \mathcal { D } } \ [ 2 0 ] )$ , the set $\{ K _ { \mathcal { D } } ^ { k } f _ { 1 } , \ldots , K _ { \mathcal { D } } ^ { k } f _ { m } \}$ converges toward a basis of the subspace spanned by the dominant m modes of $K _ { \mathcal { D } }$ . To avoid ill-conditioning, a normalization step can be applied periodically. The above conceptual approach is made practical by (i) using data to approximate the action of the exact Koopman operator, and (ii) using gradient descent to project back on the set of neural-network observables.

Concretely, the contributions of this work are as follows:

• Theory: We provide the first power-iteration-based algorithm to approximate the dominant modes of the projected Koopman operator, without requiring its explicit construction; thereby alleviating the need of large userprovided dictionaries, or anti-collapse mechanisms for learned dictionaries;

We discuss the soundness of our approach and of its data-driven neural-network-based implementation (see below), based on the neural tangent kernel theorem [21];

• Implementation: We provide a tractable data-driven implementation of the above algorithm for observables parametrized by neural networks;

• Experiments: We demonstrate the usefulness of our approach on numerical examples.

## Applications of Koopman mode decomposition

For an extensive discussion of the Koopman mode decomposition and its applications, we refer the reader to the surveys [2], [6], [22]. When the relevant observables admit an expansion in Koopman eigenfunctions, the action of the Koopman operator can be represented or approximated spectrally. More precisely, if $\begin{array} { r } { f } { { \mathbf \xi } = \sum _ { i = 1 } ^ { \infty } \alpha _ { i } g _ { i } } \end{array}$ , where $g _ { i }$ are eigenfunctions of K with eigenvalues $\lambda _ { i } ,$ , then ${ { \kappa } ^ { k } } f =$ $\textstyle \sum _ { i = 1 } ^ { \infty } { \overline { { \alpha _ { i } } } } \lambda _ { i } ^ { k } g _ { i }$ . Such an eigenfunction expansion is not available for every dynamical system or every observable space: the point spectrum may be insufficient or even absent, and Koopman eigenfunctions need not form a basis. The latter shows that only the dominant modes will dictate the longterm dynamic (i.e., the value of ${ \kappa } ^ { k } f$ for large $k )$ because ${ \boldsymbol { \lambda } } _ { i } ^ { k }$ will converge faster to zero for the dominated modes than for the dominant ones. Notably, the modes with eigenvalue one give the basins of attraction of the system, while the subunitary dominant eigenvalues give the rate of convergence of the system toward attractors [2], [22]; see also Section II.

It is well known that since K is an infinite-dimensional operator, its (generalized) eigenfunctions may not necessarily span the set of all observables [3]. Furthermore, it is also well known that the projection of the Koopman operator on a subspace introduces perturbations in the spectrum of $\kappa .$ This needs to be taken into account in the analysis of the system using the projected Koopman operator; see, e.g., [10], [11]. Nevertheless, since this is not the main focus of this work (besides the fact that it motivates the use of expressive sets D, which underpins this work), we will not discuss this matter into detail and instead focus on learning the dominant modes of the projected Koopman operator.

Control can also be achieved with Koopman mode decomposition when applied to the Koopman operator of open-loop systems [5], [7]. This will be left for future work as well.

## Other related work

Recently, several research efforts in the field of machine learning and artificial intelligence have been devoted to learning the dynamics of “systems” or environments in reducedorder latent space representations allowing for prediction, analysis, and planning [23]–[25]. For instance, the joint embedded predictive architecture (JEPA) learns together a latent representation of the state of the environment using neural network encoders as well as a neural network predictive model for the evolution of the latent representation over time. These approaches connect to the philosophy of this work in that they also aim to learn simultaneously the latent representation of the state and the predictive model in the latent space. Consequently, like us, they face the same risk of collapse of their model during training. The differences with our approach are twofold: (i) they use neural network as predictive models, whereas we use linear models (encoded by the matrix $K _ { \mathcal { D } } ) ;$ ; and (ii) they fight collapse with techniques different from our power-iteration-based approach (e.g., contrastive samples, entropy maximization, decoders, etc.) [26].

## Notation

Throughout this paper, we fix $n \in  { \mathbb { N } } _ { > 0 }$ and a state space $\mathcal { X } \subseteq \mathbb { R } ^ { n }$ . Let $( \mathcal { X } , \Sigma , \mu )$ be a probability space on $x ,$ with $\sigma \mathrm { - }$ algebra Σ and probability measure $\mu .$ . We let $\mathcal { F }$ be the space $L ^ { 2 } ( \mathcal { X } , \Sigma , \mu )$ endowed with the norm $\| \cdot \| . ^ { 2 }$ Note that $\mathcal { F }$ is a Hilbert space with inner product $\langle f , g \rangle \triangleq \mathbb { E } _ { \mu } [ f { \bar { g } } ] \ [ 2 7 ]$ . We denote by i the imaginary unit.

A feed-forward neural network from $\mathbb { R } ^ { n }$ to $\mathbb { C }$ is generically denoted by $\phi ^ { \theta }$ , where $\theta \in \mathbb { R } ^ { n _ { \theta } }$ is a parameter vector for the weights and biases of the network. Concretely, for a number $L \in \mathbb { N } _ { > 0 }$ of hidden layers with respective widths $c _ { 1 } , \ldots , c _ { L } \in \mathbb { N } _ { > 0 }$ and an activation function $\sigma : \mathbb { R }  \mathbb { R }$ , the network $\phi ^ { \theta }$ is defined by

$$
\begin{array} { r l } & { y _ { 0 } = x , } \\ & { y _ { k + 1 } = \sigma ( W _ { k } ^ { \theta } y _ { k } + b _ { k } ^ { \theta } ) , \quad k = 0 , \dots , L - 1 , } \\ & { \phi ^ { \theta } ( x ) = [ 1 , \mathrm { i } ] ( W _ { L } ^ { \theta } y _ { L } + b _ { L } ^ { \theta } ) , } \end{array}
$$

where the weight matrices $W _ { k } ^ { \theta } \in \mathbb { R } ^ { c _ { k + 1 } \times c _ { k } }$ and bias vectors $b _ { k } ^ { \theta } \in \mathbb { R } ^ { c _ { k + 1 } }$ (with $c _ { 0 } = n$ and $c _ { L + 1 } = 2 )$ are parametrized by $\theta ,$ and σ is applied entrywise on vectors.

## II. PROBLEM STATEMENT

We consider a discrete-time dynamical system of the form

$$
x ( t + 1 ) = F ( x ( t ) ) , \quad t \in \mathbb { N } ,
$$

where $F : \mathcal X  \mathcal X$ is a continuous map. In particular, X is assumed to be forward invariant under $F ,$ so that $F ( x ) \in { \mathcal { X } }$ for every $x \in \mathcal { X }$ . We assume throughout that $f \circ F \in { \mathcal { F } }$ for every $f \in { \mathcal { F } }$ , so that the composition operator induced by F maps $\mathcal { F }$ into itself.

Next, we define the Koopman operator of this system.

Definition 1: Let $F : \mathcal X  \mathcal X$ be a continuous map. The Koopman operator of F is the linear operator $\mathcal { K } : \mathcal { F }  \mathcal { F }$ defined by

$$
\begin{array} { r } { \mathcal { K } f = f \circ F , \quad \forall f \in \mathcal { F } , } \end{array}
$$

i.e., $( \mathcal { K } f ) ( x ) = f ( F ( x ) )$ for all $x \in \mathcal { X }$ and $f \in { \mathcal { F } } .$

Eigenvalues and eigenfunctions of this operator are defined in the traditional way:

Definition 2: Let $K : { \mathcal { F } }  { \mathcal { F } }$ be a linear operator. A nonzero function $g \in { \mathcal { F } }$ is an eigenfunction of K if there exists $\lambda \in \mathbb { C }$ , called an eigenvalue, such that $\kappa g = \lambda g$

The connections between the eigenvalues and eigenfunctions of the Koopman operator and the properties of the dynamical system (such stability, chaos, and ergodicity) have been extensively studied [2], [3]. We mention below two results that we will use in the numerical experiments.

Proposition $I ~ ( { \cal I } 2 , \ S 3 . A { \cal J } ) :$ : Let $F : \mathcal { X }  \mathcal { X }$ be a continuous map and K its Koopman operator. Let $x _ { * } \in \mathcal X$ be an asymptotically stable equilibrium point of $F .$ Let $g \in { \mathcal { F } }$ be an eigenfunction of K with eigenvalue 1. Assume that $g$ is continuous at $x _ { * }$ . Then, g is constant on the basin of attraction of $x _ { * }$

Proposition $2 ~ ( l 2 , ~ \ S 3 . A { \it { J } } ) .$ : Let $F : \mathcal { X }  \mathcal { X }$ be a continuous map and $\kappa$ its Koopman operator. Let $x _ { * } ~ \in ~ \mathcal { X }$ be a globally asymptotically stable equilibrium point of $F . ^ { 3 }$ Define the set ${ \mathcal { F } } _ { 0 } = \{ f \in { \mathcal { F } } : f ( x _ { * } ) = 0 \}$ . It holds that:

i) $\mathcal { F } _ { 0 }$ is invariant for $\mathcal { K } , \mathrm { i . e . , } \mathcal { K } \mathcal { F } _ { 0 } \subseteq \mathcal { F } _ { 0 }$

Let $g \in \mathcal { F } _ { 0 }$ be an eigenfunction of K with eigenvalue $\lambda \in \mathbb { C }$ Assume that $g$ is continuous at $x _ { 0 }$ . It holds that:

ii) $| \lambda | < 1 .$

iii) $x \mapsto | g ( x ) |$ | provides a semi-Lyapunov function (“semi” in the sense that it may be 0 at $x \neq x _ { * } )$ for $F .$

## Projected Koopman operator

As mentioned earlier, working in an infinite-dimensional space of observables is typically not tractable computationally. Therefore, we usually restrict our attention to a finitedimensional subspace of observables, denoted ${ \mathcal { D } } \subseteq { \mathcal { F } }$ . Since it is not guaranteed that D is invariant for $\kappa ,$ we need to project back on D after application of $\kappa .$ . Given a closed linear subspace $\mathcal { D } \subseteq \mathcal { F }$ , the projection on $\mathcal { D }$ is the linear operator $\Pi _ { \mathcal { D } } : \mathcal { F }  \mathcal { F }$ defined by

$$
\Pi _ { \mathcal { D } } f = \underset { g \in \mathcal { D } } { \arg \operatorname* { m i n } } \ : \| f - g \| .\tag{2}
$$

The projected Koopman operator on $\mathcal { D }$ is then defined as

$$
K _ { \mathcal { D } } : \mathcal { D }  \mathcal { D } , \quad K _ { \mathcal { D } } = \Pi _ { \mathcal { D } } \circ K .
$$

In the following, we assume that D is a finite-dimensional linear subspace of the Hilbert space ${ \mathcal { F } } ,$ and we denote its dimension by D (typically, $D \gg 1 )$ . Every finite-dimensional linear subspace of a normed space is closed; hence D is closed and the orthogonal projection $\Pi _ { \mathcal { D } }$ in (2) is well defined and linear. We also assume that $\kappa D \subseteq { \mathcal { F } } . ^ { 4 }$ With these assumptions, $K _ { \mathcal { D } }$ is well defined and is a finite-dimensional linear operator. We denote its eigenvalues (counted with algebraic multiplicity) by $\lambda _ { 1 } , \dots , \lambda _ { D } \in \mathbb { C }$ , ordered as

$$
| \lambda _ { 1 } | \geq | \lambda _ { 2 } | \geq \cdot \cdot \cdot \geq | \lambda _ { D } | .\tag{3}
$$

The goal of this paper is to approximate the m dominant eigenvalues and associated invariant subspace of the projected operator $K _ { D } .$ , for some $1 \ \leq \ m \ \ll \ D$ , without computing $K _ { \mathcal { D } }$ explicitly. Thus, the primary object targeted by our power iteration is $K _ { \mathcal { D } } ~ = ~ \Pi _ { \mathcal { D } } \circ { \mathcal { K } } .$ , rather than the infinite-dimensional Koopman operator K itself (the relation between eigenfunctions of $K _ { \mathcal { D } }$ and those of $\kappa$ is a separate approximation issue and can be affected by spectral pollution, as recalled in Remark 1 below). We introduce in the next section a power-iteration-based algorithm that does not require an explicit matrix representation of $K _ { \mathcal { D } }$ (nor an explicit basis of D, e.g., when the projection is implemented through neural regression). Furthermore, this approach can be implemented in a fully data-driven way.

Remark 1: As mentioned in the introduction, the projection of the Koopman operator may induce spectral pollution, i.e., eigenfunctions of $K _ { \mathcal { D } }$ may not be good approximations of eigenfunctions of K [11]. We stress that this is an issue common to all Koopman methods based on projection on a finite-dimensional subspace [10], [12]–[14], [16]–[19]. Postprocessing techniques can be used to filter out spurious eigenfunctions [11]; we will use these techniques in our numerical experiments (Section IV) to compare the accuracy of different methods.

## III. POWER-ITERATION-BASED ALGORITHM

We present and analyze our power-iteration-based algorithm to compute the dominant eigenvalues of $K _ { D }$ , which constitutes the main theoretical contribution of this work. We first present its theoretical idealized version (which is rather straightforward), then a practical neural data-driven version. We discuss the soundness of the latter, based on the neural tangent kernel theorem.

## A. Idealized algorithm

This idealized version is not applicable in most practical situations, but nevertheless conveys the main theoretical ideas behind the algorithm. The base idea is to simply apply the power iteration algorithm on the operator $K _ { D }$ . The power iteration algorithm consists in choosing an initial set of linearly independent functions $\{ f _ { 1 } ^ { ( 0 ) } , \dotsc , f _ { m } ^ { ( 0 ) } \} \subseteq \mathcal { D }$ , and applying recursively $K _ { \mathcal { D } }$ to each of these functions. Possibly, a normalization step can be applied regularly to avoid illconditioning. A pseudo-code is presented in Algorithm 1.

Algorithm 1: Idealized Koopman power iteration   
Data: Koopman operator ${ \overline { { \kappa , } } }$ subspace $\overline { { \mathcal { D } \subseteq \mathcal { F } } }$ and   
projection $\Pi _ { { \mathcal { D } } } ,$ # modes m $\in  { \mathbb { N } } _ { > 0 }$   
1 $\{ f _ { j } ^ { ( 0 ) } \} _ { j = 1 } ^ { m } $ select m lin. ind. functions from D   
2 for $k \overset { \cdot } { = } 0 , 1 , 2 , \ldots$ . do   
3 $\{ h _ { j } ^ { ( k + 1 ) } \} _ { j = 1 } ^ { m }  \{ \mathcal { K } f _ { j } ^ { ( k ) } \} _ { j = 1 } ^ { m }$   
4 $\{ g _ { j . } ^ { \check { ( k + 1 ) } } \} _ { j = 1 } ^ { m } \gets \{ \Pi _ { \mathcal { D } } \acute { h } _ { j } ^ { ( k \check { + } 1 ) } \} _ { j = 1 } ^ { m }$   
5 $\{ \bar { f } _ { j } ^ { ( k + 1 ) } \} _ { j = 1 } ^ { - }  \mathrm { O r t h o B a s i s } ( \{ g _ { j } ^ { ( k + 1 ) } \} _ { j = 1 } ^ { m } )$   
// Extract orthonormal basis if wanted

Classical results from numerical linear algebra guarantees convergence of the power iteration algorithm toward the dominant modes under assumption of separation of the modulus of the mth and $( m + 1 )$ )st eigenvalues.

Proposition 3 ([20, Th. 7.3.1]): Consider a Koopman operator K and a finite-dimensional subspace $\mathcal { D } \subseteq \mathcal { F }$ . Let the eigenvalues of $K _ { D }$ be ordered as in (3). Assume that $\vert \lambda _ { m } \vert > \vert \lambda _ { m + 1 } \vert$ . Let $\mathcal { V } _ { m } \subseteq \mathcal { D }$ be the invariant subspace of $K _ { \mathcal { D } }$ associated to the first m eigenvalues. Consider an execution of Algorithm 1. Then, for almost $\mathrm { a l l } ^ { 5 }$ choices of $\{ f _ { 1 } ^ { ( 0 ) } , \ldots , f _ { m } ^ { ( \tilde { 0 } ) } \} ~ \subseteq ~ \mathcal { D }$ in line 1, span $\{ f _ { 1 } ^ { ( k ) } , \ldots , f _ { m } ^ { ( k ) } \}$ converges towards $\nu _ { m }$ when $k \to \infty$

## B. Neural data-driven algorithm

The purpose of the neural data-driven version is to relax the computation of $\boldsymbol { \mathcal { K } } \boldsymbol { f } _ { j }$ and $\Pi _ { \mathcal { D } } h _ { j }$ in Algorithm 1, which are both intractable for most practical problems.<sup>6</sup>

Therefore, the projection $\Pi _ { \mathcal { D } }$ is first replaced by a datadriven approximation. Given N samples $\{ x _ { i } \} _ { i = 1 } ^ { N } \subseteq \stackrel { \cdot } { x }$ drawn i.i.d. from $\mu ,$ we approximate $\langle \cdot , \cdot \rangle$ and ∥·∥ by Monte–Carlo approximations $\langle \cdot , \cdot \rangle _ { N } ^ { \prime }$ and $\| \cdot \| _ { N } ^ { \prime } \mathrm { . }$

$$
\langle f , g \rangle _ { N } ^ { \prime } \triangleq \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f ( x _ { i } ) \bar { g } ( x _ { i } ) , \quad \| f \| _ { N } ^ { \prime } \triangleq \langle f , f \rangle _ { N } ^ { \prime } .\tag{4}
$$

This allows to approximate $\Pi _ { \mathcal { D } }$ by replacing $\mathbf { \boldsymbol { \mathsf { \Sigma } } } ^ { \bullet \bullet } \lVert \cdot \rVert ^ { \mathfrak { W } } \mathbf { \boldsymbol { \mathsf { b y } } } ^ { \bullet \bullet } \rVert \cdot \lVert \mathbf { \boldsymbol { \mathsf { \Sigma } } } _ { N } ^ { \prime } \qquad $ in (2). The resulting operator is denoted by $\Pi _ { N , D } ^ { \prime } \colon $

$$
\Pi _ { N , \mathcal { D } } ^ { \prime } f = \underset { g \in \mathcal { D } } { \arg \operatorname* { m i n } } \| f - g \| _ { N } ^ { \prime } .\tag{5}
$$

When the elements of $\mathcal { D }$ are represented by a neural network $\phi ^ { \theta }$ , the argmin in (5) reduces to classical function regression using $\ell ^ { 2 }$ loss:

$$
\Pi _ { N , \Phi } ^ { \prime } f = \underset { \theta \in \mathbb { R } ^ { n _ { \theta } } } { \arg \operatorname* { m i n } } \ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \phi ^ { \theta } ( x _ { i } ) - f ( x _ { i } ) \right| ^ { 2 }\tag{6}
$$

(we argue in Section III-C that this yields a linear operator under appropriate assumptions). (6) can be solved by using gradient descent, Adam, AdamW, etc. [28].

Following Algorithm 1, $\Pi _ { N , \Phi } ^ { \prime }$ must be applied to $h _ { i } ^ { ( k + 1 ) } =$ ${ \kappa } f _ { i } ^ { ( k ) }$ . Hence, $f ( x _ { i } )$ in (6) must be replaced by $h _ { j } ^ { ( k + 1 ) } ( x _ { i } )$ Since $K f _ { j } ( x _ { i } ) = f _ { j } ( F ( x _ { i } ) )$ , it suffices to compute

$$
x _ { i } ^ { + } \triangleq F ( x _ { i } ) \quad i = 1 , \dots , N ,\tag{7}
$$

and let $h _ { j } ^ { ( k + 1 ) } ( x _ { i } ) = f _ { j } ^ { ( k ) } ( x _ { i } ^ { + } )$ . In particular, if the current iterate is ${ \bf \bar { \rho } } _ { j } ^ { ( k ) } = \phi ^ { \theta _ { j } ^ { ( k ) } }$ , the next pre-orthogonalization iterate is given by $g _ { j } ^ { ( k + 1 ) } = \phi ^ { \zeta _ { j } ^ { ( k + 1 ) } }$ where

$$
\zeta _ { j } ^ { ( k + 1 ) } \in \underset { \theta \in \mathbb { R } ^ { n _ { \theta } } } { \arg \operatorname* { m i n } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \phi ^ { \theta } ( \boldsymbol { x } _ { i } ) - \phi ^ { \theta _ { j } ^ { ( k ) } } ( \boldsymbol { x } _ { i } ^ { + } ) \right| ^ { 2 } .\tag{8}
$$

(8) makes explicit that no representation of $\kappa$ is constructed: at each power iteration, a neural network is simply regressed on the values of the previous iterate evaluated at the successor samples. Hence, the computations are fully data-driven, based only on i.i.d. samples $\{ x _ { i } \} _ { i = 1 } ^ { N } \subseteq \mathcal X$ and their successors $\{ x _ { i } ^ { + } \} _ { i = 1 } ^ { \bar { N } }$ by the system as in (7). A pseudocode of the neural data-driven algorithm is presented in Algorithm 2.

Algorithm 2: Neural data-driven Koopman power   
iteration   
Data: Samples $\overline { { \{ x _ { i } \} _ { i = 1 } ^ { N } \subseteq \mathcal { X } } }$ and successors   
$\{ x _ { i } ^ { + } \bar  \} _ { i = 1 } ^ { N }$ , neural network $\phi ^ { \theta } ,$ # modes   
$m \in \mathbb { N } _ { > 0 }$   
1 $\{ \theta _ { j } ^ { ( 0 ) } \} _ { j = 1 } ^ { m } $ initialize m param. vec. from $\mathbb { R } ^ { n _ { \theta } }$   
2 for $k \overset { \cdot } { = } 0 , 1 , 2 , \ldots$ . do   
3 for $j = 1 , \ldots , m$ do   
4 $\zeta _ { j } ^ { ( k + 1 ) } \gets$ solve regression (8)   
5 $g _ { j } ^ { ( k + 1 ) } \gets \phi ^ { \zeta _ { j } ^ { ( k + 1 ) } }$   
6 $\{ \theta _ { j } ^ { ( k + 1 ) } \} _ { j = 1 } ^ { m }  \mathrm { O r t h o B a s i s } ( \{ g _ { j } ^ { ( k + 1 ) } \} _ { j = 1 } ^ { m } )$   
$/ / \subset \pounds .$ Remark 2

Remark 2: The orthonormalization of the basis of neural networks (line 6 in Algorithm 2) can be done easily from the samples as follows:

1) At a given iteration $k ,$ compute the vectors $v _ { 1 } , \ldots , v _ { m } \in$ $\mathbb { C } ^ { N }$ defined by $v _ { j } = [ g _ { i } ^ { ( k + \bar { 1 } ) } ( x _ { i } ) ] _ { i = 1 } ^ { N }$

2) Compute an orthonormal basis $\{ q _ { 1 } , \dots , q _ { m } \}$ of $\{ v _ { 1 } , \ldots $ $, v _ { m } \} ( { \bf e . g . , b } _ { \ast }$ y computing the QR decomposition of the matrix $[ v _ { 1 } , \ldots , v _ { m } ] ^ { \ } \in \mathbb { C } ^ { \mathbf { \tilde { N } } \times m }$ [20]).

3) For all $1 \leq j \leq m$ , find

$$
\theta _ { j } ^ { ( k + 1 ) } \in \mathop { \arg \operatorname* { m i n } } _ { \theta \in \mathbb { R } ^ { n _ { \theta } } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \phi ^ { \theta } ( x _ { i } ) - q _ { j } [ i ] \right| ^ { 2 } .
$$

If the fitting error in 3) is less than or equal to $\epsilon ,$ it follows that for all $1 ~ \leq ~ j _ { 1 } , j _ { 2 } ~ \leq ~ m , ~ | \langle \phi ^ { \theta _ { j _ { 1 } } } , \phi ^ { \theta _ { j _ { 2 } } } \rangle _ { N } ^ { \prime } - \delta _ { j _ { 1 } , j _ { 2 } } | ~ \leq$ $2 \epsilon + O ( \epsilon ^ { 2 } )$ . In other words, $\{ \phi ^ { \theta _ { 1 } } , \dots , \phi ^ { \theta _ { m } } \}$ is ϵ-orthonormal with respect to $\langle \cdot , \cdot \rangle _ { N } ^ { \prime } . ^ { 8 }$ Furthermore, under the assumption that the set of functions parametrized by $\phi ^ { \theta }$ forms a linear subspace (see Section III-C), the global minimum of fitting error in 3) achieves $\epsilon = 0$ ◁

## C. Analysis of neural data-driven v.s. idealized

We analyze the neural data-driven algorithm, in particular its soundness to approximate the idealized algorithm.

First, we start with the Monte–Carlo approximation of the norm and inner product. By the strong law of large numbers, we have that both approximations are sound as $N \to \infty ;$

Theorem 1 $( I 2 9 , \ S \ \& \ I ) .$ : Let $\{ x _ { i } \} _ { i = 1 } ^ { \infty } \subseteq { \mathcal { X } }$ be a sequence of i.i.d. samples from $\mu .$ . Let $f , g \in { \mathcal { F } }$ . With probability 1 on the sampling of $\{ x _ { i } \} _ { i = 1 } ^ { \infty }$ , it holds that

$$
\begin{array} { r } { \langle f , g \rangle _ { N } ^ { \prime } \to \langle f , g \rangle \quad \mathrm { a n d } \quad \| f \| _ { N } ^ { \prime } \to \| f \| , \quad \mathrm { a s } \quad N \to \infty . } \end{array}
$$

As a corollary, we obtain the main result of this section: Corollary 1 (Main result): Consider a Koopman operator K and a neural network $\phi ^ { \theta }$ . Assume that $\Pi _ { N , \Phi } ^ { \prime }$ in (6) is a linear operator with image set denoted by $\mathcal { D }$ (which is a linear subspace). Assume that $\mathcal { D } \subseteq \mathcal { F }$ and $\boldsymbol { \mathcal { K D } } \subseteq \mathcal { F }$ . Let $\{ x _ { i } \} _ { i = 1 } ^ { \infty } \subseteq { \mathcal { X } }$ be a sequence of i.i.d. samples from $\mu .$

i) For any $f \in { \mathcal { F } } _ { : }$ , it holds that $\Pi _ { N , \Phi } ^ { \prime } { \cal K } f  \Pi _ { \cal D } { \cal K } f$ with probability one as $N \to \infty$

ii) Consequently, for any fixed $k ~ \in ~ \mathbb { N } _ { > 0 }$ , the iterates $\{ { \phi } ^ { { \theta } _ { 1 } ^ { ( k ) } } , \ . . . , { \bar { \phi } } ^ { { \theta } _ { m } ^ { ( k ) } } \}$ of Algorithm 2 converge with probability one to the iterates $\mathsf { \bar { \{ f _ { 1 } } { \varepsilon } _ { \varepsilon } ^ { ( k ) } , \ldots , f _ { m } ^ { ( k ) } \} }$ of Algorithm 1 (with the same $\mathcal { D } )$ , as $N  \infty .$

Remark 3 (Convergence and approximation limits): It is useful to distinguish the different limiting arguments underlying Algorithm 2. (i) At the population level, Algorithm 1 is a classical subspace iteration for the projected Koopman operator $K _ { \mathcal { D } }$ . Under the spectral-gap assumption $| \lambda _ { m } | >$ $| \lambda _ { m + 1 } |$ , its iterates satisfy

$$
\mathrm { s p a n } \{ f _ { 1 } ^ { ( k ) } , \ldots , f _ { m } ^ { ( k ) } \} \longrightarrow \mathcal { V } _ { m } \qquad \mathrm { a s } \ k \to \infty ,
$$

for almost every initialization. (ii) Algorithm 2 replaces each exact projection appearing in this ideal iteration by an empirical projection computed from N samples. Corollary 1 shows that, for any fixed number of iterations $k ,$ this approximation becomes exact as $N  \infty \colon$ the data-driven iterates converge almost surely to the corresponding ideal iterates. (iii) Finally, the neural implementation introduces an additional approximation, since the empirical projection is obtained by training a neural network. The NTK argument below shows that, in the infinite-width regime, this training procedure can be interpreted as a linear empirical projection, thereby connecting the neural implementation to the setting of Corollary 1.

Finally, we discuss the assumption that $\Pi _ { N , \Phi } ^ { \prime }$ is a linear operator, made in Corollary 1 (and Remark 2). Note that it is in general not a linear operator since $\{ \phi ^ { \theta } : \theta \in \mathbb { R } ^ { n _ { \theta } } \}$ is not a linear subspace.<sup>9</sup> Nevertheless, the neural tangent kernel (NTK) theorem [21], [30] allows us to assume that when the width of $\phi ^ { \theta }$ is large, $\Pi _ { N , \Phi } ^ { \prime }$ yields a linear operator.

Theorem $2 ( l 2 l , T h . \ 2 ] ) .$ : Consider a neural network $\phi ^ { \theta }$ with random Gaussian parameter $\theta \in \mathbb { R } ^ { n _ { \theta } }$ . In the infinitewidth limit, i.e., when the hidden widths $c _ { 1 } , \ldots , c _ { L } \to \infty .$ the neural tangent kernel

$$
\begin{array} { r } { \Theta _ { \theta } ( x , y ) \triangleq \nabla _ { \theta } \phi ^ { \theta } ( x ) ^ { \top } \nabla _ { \theta } \phi ^ { \theta } ( y ) , \quad x , y \in \mathcal { X } , } \end{array}
$$

converges in probability (with respect to θ) to a determin-$i s t i c ^ { 1 0 }$ kernel $\Theta _ { \ast }$ . Moreover, when trained by gradient flow on the $\ell ^ { 2 }$ loss (and with parameter θ initialized randomly as above), the network output evolves according to

$$
\frac { \mathrm { d } } { \mathrm { d } t } \phi ^ { \theta _ { t } } ( x ) \propto - \sum _ { i = 1 } ^ { N } \Theta _ { * } ( x , x _ { i } ) \big \{ \phi ^ { \theta _ { t } } ( x _ { i } ) - f ( x _ { i } ) \big \} .
$$

Hence, in the infinite-width regime, the training dynamics corresponds to kernel regression in the reproducing kernel Hilbert space associated with $\Theta _ { * }$

The NTK theorem implies that, in the infinite-width regime and under gradient-flow training from the stated random initialization, the network output evolves in the linear

span of the kernel sections $\Theta _ { * } ( \cdot , x _ { i } )$ evaluated at the training inputs. Consequently, least-squares training acts as kernel regression in the finite-dimensional data-dependent space

$$
\mathcal { D } _ { N } = \operatorname { s p a n } \{ \Theta _ { * } ( \cdot , x _ { i } ) \} _ { i = 1 } ^ { N } .
$$

In this asymptotic regime, the fitted-value map $f \mapsto \Pi _ { N , \Phi } ^ { \prime } f$ can therefore be identified with the empirical linear projection $\Pi _ { N , \mathcal { D } _ { N } } ^ { \prime }$ (up to the usual qualifications associated with initialization and optimization in the NTK limit). This is the sense in which the nonlinear neural parametrization induces the linear projection assumed in Corollary 1; it is not an assumption that finite-width neural networks themselves form a linear function class. Thus, Corollary 1 provides an asymptotic justification for the neural implementation.<sup>11</sup>

## IV. NUMERICAL EXPERIMENTS

In all experiments, we used neural networks with $L =$ 2 hidden layers, with widths denoted by $c _ { 1 }$ and $c _ { 2 } ,$ and activation function $\sigma ~ = ~ \operatorname { t a n h }$ . Using PyTorch [31], we trained the networks over 800 epochs with AdamW optimizer with learning rate $1 0 ^ { - 3 }$ , no weight decay, and full-batch training. The maximum number of iterations of Algorithm 2 was $k _ { \operatorname* { m a x } } = 5 0$ . This fixed budget was used uniformly in the reported experiments; more generally, one may stop when successive empirical subspaces (or the associated Koopman residuals) vary below a prescribed tolerance. We compared with extended dynamic mode decomposition (EDMD) [13].

Code for the implementation of the method and the numerical experiments is available at https://github.com/ guberger/DeepKoopmanLearning. All computations were made on a laptop with processor Intel Core i7-7600u and 16GB RAM running Windows.

## A. Nonlinear 1D system

Consider the system on $\chi = \mathbb { R }$ given by

$$
x _ { t + 1 } = 0 . 9 \operatorname { t a n h } ( x _ { t } ) .\tag{9}
$$

This system admits a globally asymptotically stable equilibrium point at $x = 0$

We applied Algorithm 2 on this system with $m = 3$ . We used $N = 2 5 0 0$ samples and standard Gaussian distribution. We used hidden widths $c _ { 1 } = c _ { 2 } = 8$

We denote by $g _ { 1 } , \ldots , g _ { m }$ the computed approximations of the m dominant Koopman modes of the system, normalized so that $\| g _ { i } \| _ { N } ^ { \prime } = 1$ . Those are represented in Figure 1. We have also represented $\lambda _ { i } ^ { - 1 } \mathcal { K } g _ { i }$ , where $\lambda _ { i }$ are the associated eigenvalues. Good approximations satisfy $\| \lambda _ { i } ^ { - 1 } { \boldsymbol { \mathcal { K } } } g _ { i } - g _ { i } \|$ is small [11]. The values are given in Table I. We see that the approximation error is overall small $( \leq 0 . 0 5 )$

<table><tr><td>2</td><td>1</td><td>2</td><td>3</td></tr><tr><td> $\lVert \lambda _ { i } ^ { - 1 } \boldsymbol { \mathcal { K } } g _ { i } - g _ { i } \rVert _ { N } ^ { \prime }$ </td><td>1.1 · 10−5</td><td> $1 . 2 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 4 \cdot 1 0 ^ { - 2 }$ </td></tr></table>

TABLE I  
APPROXIMATION ERROR OF ALGORITHM 2 ON SYSTEM (9).

<sup>11</sup>We omit the verification that $\mathcal { D } \subseteq \mathcal { F }$ and $\boldsymbol { \mathcal { K D } } \subseteq \mathcal { F }$ because this is expected to hold generically since neural networks are piecewise smooth.

![](images/2e4daf4866c3c7613953e9c7d457c16e7ac8fb3b014dcb120c5ac85261b42a5d.jpg)

![](images/d221d8fe7abc6a94e383899205f2c6c95dc2683425cb8c30ccfb0c9139e6c19c.jpg)

![](images/78761629821ff64f5ad301693d5ed8d5d3be4f7ebe2e764503b8c5cfca1e4434.jpg)  
Fig. 1. Approximation of dominant $m = 3$ Koopman modes of system (9) with Algorithm 2. The blue lines represent the normalized Koopman modes g<sub>i</sub>. The orange lines represent $\lambda _ { i } ^ { - 1 } \kappa g _ { i }$ . As we can see, $\lambda _ { i } ^ { - 1 } \mathcal { K } g _ { i } \approx g _ { i }$ Note that the shifted value axis for g<sub>0</sub>.

Next, we compared with EDMD with D being the set of polynomials of degree up to 3 or $6 . ^ { 1 2 }$ The results are presented in Figure 2 and Table II. Overall, one sees that our approach provides smaller approximation errors. The computation time is however longer: 30 secs for our approach vs. 2 secs for the EDMD approach. This additional cost is expected because each power iteration requires fitting several neural networks, whereas EDMD solves a single finite-dimensional regression problem.

![](images/a57d484c119376a4c067f178f8c0062065adae3b8b25b4d42542518113dfc2ce.jpg)

![](images/b96bee6cac1869ed9af16e3766e8bbbc995942cdf9e6151b1ea7cbc688980096.jpg)

![](images/a42c1c6987cabcb1cae84c3aae0cef5ddfabbc516ae24e5d0fcaafb7b4160d48.jpg)

![](images/bc0fcb4cca4e41a9e47a731c232153e7364c501eb008c0858051c777b6a4d6bd.jpg)

![](images/89fa3fccb2a0fd9ea5d5e694b72c3f04026ab3c736d1bf19e8c135e10ce80ba4.jpg)

![](images/af1ed9048f629b40dce4a9be95f8f7809d697139d107f6df6ecbb2c65bdbb7f8.jpg)  
Fig. 2. Approximation of the dominant three Koopman modes of system (9) with EDMD using polynomials of degree up to 3 (top) or 6 (bottom). The color code is the same as in Figure 1. Note that the shifted value axis for g<sub>0</sub>.

<table><tr><td>2</td><td>1</td><td>2</td><td>3</td></tr><tr><td></td><td> $\begin{array} { c } { 2 . 4 \cdot 1 0 ^ { - 1 5 } } \\ { 3 . 7 \cdot 1 0 ^ { - 1 5 } } \end{array}$ </td><td> $1 . 2 \cdot 1 0 ^ { - 1 }$ </td><td> $6 . 0 \cdot 1 0 ^ { - 1 }$ </td></tr><tr><td> $\lVert \lambda _ { i } ^ { - 1 } \boldsymbol { \mathcal { K } } g _ { i } - g _ { i } \rVert _ { N } ^ { \prime }$ </td><td></td><td> $7 . 4 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 1 \cdot 1 0 ^ { - 1 }$ </td></tr></table>

TABLE II  
APPROXIMATION ERROR OF EDMD WITH POLYNOMIALS OF DEGREE UP TO 3 $\cdot \left( 1 ^ { \mathrm { s T } } \right.$ ROW) OR 6 $( 2 ^ { \mathrm { N D } }$ ROW) ON SYSTEM (9).

## B. Bistable system

Next, consider the continuous-time system on $\mathcal { X } = \mathbb { R } ^ { 2 }$ described by the unforced Duffing equation:

$$
\left\{ \begin{array} { r c l } { \dot { x } _ { 1 } ( t ) } & { = } & { x _ { 2 } ( t ) , } \\ { \dot { x } _ { 2 } ( t ) } & { = } & { - 0 . 5 x _ { 2 } ( t ) + x _ { 1 } ( t ) - x _ { 1 } ^ { 3 } ( t ) . } \end{array} \right.\tag{10}
$$

This system has two asymptotically stable equilibrium points at $x _ { * , 1 } = ( - 1 , 0 )$ and $x _ { * , 2 } = ( 1 , 0 )$ , and one saddle point at $x _ { * , 3 } = ( 0 , 0 )$ (see Figure 3). Using our approach, we will estimate the basins of attraction of $x _ { * , 1 }$ and $x _ { * , 2 } ,$ by computing nontrivial dominant eigenfunctions of the Koopman operator of the associated discrete-time system. We stress out that obtaining good approximations is a nontrivial problem [16] since this requires discontinuous approximation functions.

![](images/5b48e34058604c04d456edbc10d070c7902a49f26db79550b4230175be0ac8b2.jpg)  
Fig. 3. Basins of attraction of Duffing system (10), estimated using numerical simulations.

Concretely, we first time-discretized the system with sampling period $T = 1 . ^ { 1 3 }$ Then, we applied Algorithm 2 on the discrete-time system with $m = 2$ . We used $N = 1 0 ^ { 4 }$ samples and uniform distribution on $[ - 4 , 4 ] ^ { 2 }$ . We used hidden widths $c _ { 1 } = c _ { 2 } = 6 4$

We denote by $g _ { 1 } , \ldots , g _ { m }$ the computed approximations of the m dominant Koopman modes of the system, normalized so that $\| g _ { i } \| _ { N } ^ { \prime } ~ = ~ 1$ . Those are represented in Figure 4. We see that the flat components of the modes $g _ { i }$ qualitatively recover the basins of attraction of the system. This experiment is intended as an illustration of the structure encoded by the learned eigenfunctions; it does not constitute a certified region-of-attraction computation. We have also represented $\lambda _ { i } ^ { - 1 } \mathcal { K } g _ { i }$ , where $\lambda _ { i }$ are the associated eigenvalues. The values of $\lVert \lambda _ { i } ^ { - 1 } \mathcal { K } g _ { i } - g _ { i } \rVert$ are given in Table III. We see that the approximation error is overall small $( \leq 0 . 1 )$ . The computation time was approx. 8 min.

<table><tr><td colspan="2">2</td><td>1</td><td>2</td></tr><tr><td> ${ | | \lambda _ { i } ^ { - 1 } \boldsymbol { \kappa } g _ { i } }$ </td><td>− gill&#x27;N</td><td>0.10</td><td>0.057</td></tr></table>

TABLE III  
APPROXIMATION ERROR OF ALGORITHM 2 ON SYSTEM (10).

<sup>13</sup>This means that $F ( x ) = \xi ( 1 )$ where $\xi ( \cdot )$ is the trajectory of (10) with $\xi ( 0 ) = x$ . Numerically, this was computed using fourth-order Runge–Kutta with time step $d t = 0 . 0 1$

![](images/ce73beb7b754696e555c4df5d2a9c1703730ca32d0f20ee4e85fac64f96b203e.jpg)

![](images/9aa960c9b997f961531be626eefc0de4a754d5a3cbbe97e130489efd52c6456f.jpg)  
Fig. 4. Approximation of dominant $m = 2$ Koopman modes of system (10) with Algorithm 2. The contour plots on the top represent the normalized Koopman modes $g _ { i }$ . The contour plots on the bottom represent $\lambda _ { i } ^ { - 1 } \mathcal { K } g _ { i }$ As we can see, $\lambda _ { i } ^ { - 1 } { \mathcal { K } } g _ { i } \approx g _ { i }$

Next, we compared with EDMD with D being the set of polynomials of degree up to 3 or $6 . ^ { 1 4 }$ The results are presented in Figure 5 and Table IV. We see that this approach fails to provide good approximations of the dominant Koopman modes (Table IV) and of the basins of attraction of the system (Figure 5). The main reason for that is the inability to represent discontinuous functions.

![](images/a542926c965a0517c0806ae6efc5d4a043067bd541451c1bfa8d4351d1a54225.jpg)

![](images/bb38d14c780a345547de34d81074f734ca30f2682dcac5f70f57a15f3b7be85d.jpg)

![](images/d66d078ca652cd3a3d463bcab2e0192b0ffc14180f6a6bae7d69e492ceb46f26.jpg)

![](images/f613589070ac96f13ad02ceb9c5af64fae4007b8ab7ae7f613f1c907bd4cc8e4.jpg)  
Fig. 5. Approximation of the dominant three Koopman modes of system (10) with EDMD using polynomials of degree up to 3 (top) or 6 (bottom). The contour plots represent the normalized Koopman modes $g _ { i }$ Note that the shifted value axis for g<sub>0</sub>.

<table><tr><td>2</td><td>1</td><td>2</td></tr><tr><td> $\lVert \lambda _ { i } ^ { - 1 } \boldsymbol { \mathcal { K } } g _ { i } - g _ { i } \rVert _ { N } ^ { \prime }$ </td><td> $\begin{array} { l } { 4 . 6 \cdot 1 0 ^ { - 1 4 } } \\ { 2 . 0 \cdot 1 0 ^ { - 1 4 } } \end{array}$ </td><td> $5 . 8 \cdot 1 0 ^ { - 1 }$ </td></tr><tr><td></td><td></td><td> $5 . 8 \cdot 1 0 ^ { - 1 }$ </td></tr></table>

TABLE IV  
APPROXIMATION ERROR OF EDMD WITH POLYNOMIALS OF DEGREE UP TO $3 \ : ( 1 ^ { \mathrm { s T } }$ ROW) OR $6 ( 2 ^ { \mathrm { N D } }$ ROW) ON SYSTEM (10).

## V. SCOPE AND LIMITATIONS

The method targets dominant spectral components of the projected operator $K _ { \mathcal { D } }$ and therefore inherits the usual

## REFERENCES

limitations of finite-dimensional Koopman approximations, including possible spectral pollution and dependence on the chosen approximation space. The number of requested modes m is a user parameter: increasing m targets a larger invariant subspace but requires additional regressions and, for convergence of subspace iteration, an appropriate separation between the retained and discarded parts of the spectrum. Moreover, the consistency result above is asymptotic and does not provide a finite-sample error bound or a universal prescription for the number of data points; the required data depend on the dynamics, sampling distribution, approximation class, and spectral/conditioning properties of the problem. The NTK argument is likewise an infinite-width justification, while the experiments necessarily use finite networks and numerical optimization. Finally, the repeated neural regressions make the proposed method computationally more expensive than EDMD in the examples considered here. Reducing this computational burden, deriving finitesample guarantees, and studying principled choices of m and stopping criteria are important directions for future work.

## VI. CONCLUSIONS

This work introduced a novel data-driven algorithm to approximate dominant eigenfunctions of a projected Koopman operator using a neural-network-based power iteration scheme. By avoiding the explicit construction of the projected Koopman operator and bypassing the need for predefined dictionaries or anti-collapse regularization mechanisms, the proposed method addresses limitations of existing approaches.

The theoretical analysis separates the classical convergence of the ideal subspace iteration in the iteration index from the consistency of its data-driven approximation for increasing sample size, while the NTK limit provides an asymptotic justification of the linear-projection interpretation of neural regression. This provides a principled justification for the method, linking it to classical operator theory and modern neural tangent kernel results. Empirical results demonstrate that the approach accurately captures dominant Koopman modes, including challenging cases such as bistable systems where eigenfunctions exhibit discontinuities. In particular, the method successfully recovers meaningful structures such as basins of attraction, which are known to be encoded in eigenfunctions associated with eigenvalue one. Compared to traditional techniques like EDMD with polynomials, the proposed method achieves improved accuracy, especially when expressive representations are required.

Overall, this work highlights the effectiveness of combining operator-theoretic ideas with neural networks in a fully data-driven framework. Future directions include extending the approach to controlled systems, improving computational efficiency, and further investigating robustness to noise and finite-sample effects.

[1] B. O. Koopman, “Hamiltonian systems and transformation in Hilbert space,” Proceedings of the National Academy of Sciences, vol. 17, no. 5, pp. 315–318, 1931.

[2] A. Mauroy and I. Mezic, “Global stability analysis using the eigen-´ functions of the Koopman operator,” IEEE Transactions on Automatic Control, vol. 61, no. 11, pp. 3356–3369, 2016.

[3] I. Mezic, “Spectrum of the Koopman operator, spectral expansions´ in functional spaces, and state-space geometry,” Journal of Nonlinear Science, vol. 30, no. 5, pp. 2091–2145, 2020.

[4] ——, “Spectral properties of dynamical systems, model reduction and decompositions,” Nonlinear Dynamics, vol. 45, no. 1, pp. 309–325, 2005.

[5] E. Kaiser, J. N. Kutz, and S. L. Brunton, “Data-driven discovery of Koopman eigenfunctions for control,” Machine Learning: Science and Technology, vol. 2, no. 35023, pp. 1–32, 2021.

[6] M. Budisiˇ c, R. Mohr, and I. Mezi´ c, “Applied Koopmanism,”´ Chaos: An Interdisciplinary Journal of Nonlinear Science, vol. 22, no. 4, pp. 1–33, 2012.

[7] S. E. Otto and C. W. Rowley, “Koopman operators for estimation and control of dynamical systems,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 4, no. 1, pp. 59–87, 2021.

[8] S. L. Brunton, M. Budisiˇ c, E. Kaiser, and J. N. Kutz, “Modern´ Koopman theory for dynamical systems,” SIAM Review, vol. 64, no. 2, pp. 229–340, 2022.

[9] M. J. Colbrook, Z. Drmac, and A. Horning, “An introductory guideˇ to Koopman learning,” 2025, arXiv preprint arXiv:2510.22002.

[10] M. J. Colbrook, “The multiverse of dynamic mode decomposition algorithms,” in Handbook of numerical analysis, S. Mishra and A. Townsend, Eds. Elsevier, 2024, vol. 25, pp. 127–230.

[11] M. J. Colbrook and A. Townsend, “Rigorous data-driven computation of spectral properties of Koopman operators for dynamical systems,” Communications on Pure and Applied Mathematics, vol. 77, pp. 221– 283, 2024.

[12] P. J. Schmid, “Dynamic mode decomposition of numerical and experimental data,” Journal of Fluid Mechanics, vol. 656, pp. 5–28, 2010.

[13] M. O. Williams, I. G. Kevrekidis, and C. W. Rowley, “A data–driven approximation of the Koopman operator: extending dynamic mode decomposition,” Journal ofNonlinear Science, vol. 25, pp. 1307–1346, 2015.

[14] H. Arbabi and I. Mezic, “Ergodic theory, dynamic mode decomposition, and computation of spectral properties of the Koopman operator,” SIAM Journal on Applied Dynamical Systems, vol. 16, no. 4, pp. 2096– 2126, 2017.

[15] Q. Li, F. Dietrich, E. M. Bollt, and I. G. Kevrekidis, “Extended dynamic mode decomposition with dictionary learning: a data-driven adaptive spectral decomposition of the Koopman operator,” Chaos: An Interdisciplinary Journal of Nonlinear Science, vol. 27, no. 10, pp. 1–10, 2017.

[16] N. Takeishi, Y. Kawahara, and T. Yairi, “Learning Koopman invariant subspaces for dynamic mode decomposition,” in NIPS’17: Proceedings of the 31st International Conference on Neural Information Processing System. ACM, 2017, pp. 1130–1140.

[17] B. Lusch, J. N. Kutz, and S. L. Brunton, “Deep learning for universal linear embeddings of nonlinear dynamics,” Nature communications, vol. 9, no. 4950, pp. 1–10, 2018.

[18] D. J. Alford-Lago, C. W. Curtis, A. T. Ihler, and O. Issan, “Deep learning enhanced dynamic mode decomposition,” Chaos: An Interdisciplinary Journal of Nonlinear Science, vol. 32, no. 3, pp. 1–14, 2022.

[19] A. K. Mondal, S. S. Panigrahi, S. Rajeswar, K. Siddiqi, and S. Ravanbakhsh, “Efficient dynamics modeling in interactive environments with Koopman theory,” 2023.

[20] G. H. Golub and C. F. Van Loan, Matrix computations, 4th ed. Baltimore, MD: The Johns Hopkins University Press, 2013.

[21] A. Jacot, F. Gabriel, and C. Hongler, “Neural tangent kernel: convergence and generalization in neural networks,” in NIPS’18: Proceedings of the 32nd International Conference on Neural Information Processing Systems. ACM, 2018, pp. 8580–8589.

[22] P. Bevanda, S. Sosnowski, and S. Hirche, “Koopman operator dynamical models: learning, analysis and control,” Annual Reviews in Control, vol. 52, pp. 197–212, 2021.

[23] D. Ha and J. Schmidhuber, “World models,” 2018, arXiv preprint arXiv:1803.10122.

[24] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. IEEE, 2023, pp. 15 619–15 629.

[25] J. Ulmen, G. Sundaram, and D. Gorges, “Learning state-space models¨ of dynamic systems from arbitrary data using joint embedding predictive architectures,” IFAC-PapersOnLine, vol. 59, no. 18, pp. 19–24, 2025.

[26] B. Terver, R. Balestriero, M. Dervishi, D. Fan, Q. Garrido, T. Nagarajan, K. Sinha, W. Zhang, M. Rabbat, Y. LeCun, and A. Bar, “A lightweight library for energy-based joint-embedding predictive architectures,” 2026, arXiv preprint arXiv:2602.03604.

[27] A. Friedman, Foundations of modern analysis. New York, NY: Dover Publications, 1982.

[28] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” 2017, arXiv preprint arXiv:1711.05101.

[29] K. B. Athreya and S. N. Lahiri, Measure theory and probability theory. New York, NY: Springer, 2006.

[31] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, “Pytorch: an imperative style, high-performance deep learning library,” in Proceedings of the 33rd International Conference on Neural Information Processing Systems. ACM, 2019, pp. 8026–8037.

[30] E. Golikov, E. Pokonechnyy, and V. Korviakov, “Neural tangent kernel: a survey,” 2022, arXiv preprint arXiv:2208.13614.