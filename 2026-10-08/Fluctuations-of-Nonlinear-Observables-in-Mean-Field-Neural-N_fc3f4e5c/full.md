# Fluctuations of Nonlinear Observables in Mean Field Neural Network Training

Arnaud Descours<sup>∗</sup> Geofrey Lacour<sup>†</sup>

## Abstract

Mean field limits describe the training dynamics of wide neural networks through the evolution of the empirical distribution of their parameters. Although functional central limit theorems characterize the asymptotic fluctuations of this distribution, quantities of practical interest are typically nonlinear observables of the parameter distribution rather than the distribution itself. In this work, we show how these mean field fluctuations propagate to finite dimensional nonlinear observables for shallow neural networks trained by stochastic gradient descent. Working in the weighted Sobolev space in which the limiting fluctuation process is constructed, we apply a functional Delta method under ordinary Fréchet diferentiability, without requiring Lions derivatives with respect to the measure variable. We obtain a central limit theorem for the observables and, under a suitable representation of their diferentials, an explicit covariance formula inherited from the underlying mean field fluctuation theory.

We also study whether prescribed quantities of interest can be recovered from the selected observations. Under a constant rank assumption, we prove that a quantity of interest factors locally through the observation functional if and only if, throughout a neighborhood, the kernel of the diferential of the observation is contained in that of the quantity of interest. Thus, a diferential condition expressed directly in the ambient Sobolev space yields an exact nonlinear local factorization. These results provide a framework both for quantifying finitewidth uncertainty on observable, statistically or physically meaningful quantities and for assessing whether the chosen observations contain the information required to identify them.

Key words: Uncertainty quantification, functional Delta method, mean field analysis, neural networks.

Mathematics Subject Classification (MSC2020): 68T07, 60F17, 60B12.

## 1 Introduction

In this work, we consider two-layer (or shallow) neural networks, namely neural networks consisting of a single hidden layer. Such models provide one of the simplest architectures for which the mean-field limit can be analyzed rigorously while already capturing many of the fundamental mechanisms underlying modern overparameterized learning. Given an input datum $x \in \mathbb { R } ^ { m }$ the network prediction is obtained by combining the contributions of the neurons in the hidden layer. Each neuron is characterized by a parameter vector $w \in \mathbb { R } ^ { d }$ , where $d \in ( \mathbb { N } \backslash \{ 0 \} )$ denotes the dimension of the parameter space associated with a single neuron. During training, these parameters are updated through stochastic gradient descent (SGD) in order to minimize a prescribed loss function. Beginning with the seminal works by F.Bach & L.Chizat [6], by S.Mei, A.Montanari & P.-M.Nguyen [46] and by J.Sirignano & K.Spiliopoulos [57], a large body of literature has shown that, rather than tracking each parameter individually, it is more convenient in the large-width regime to describe the state of the network through the empirical distribution of the neuron parameters. This point of view provides a rigorous mean-field description of SGD through the evolution of a probability measure and has led to a detailed understanding of both the deterministic training dynamics and their asymptotic fluctuations.

More precisely, let $\mu _ { t } ^ { N }$ denote the empirical distribution of the network parameters at step t. Under suitable assumptions, the sequence $( ( \mu _ { t } ^ { N } ) _ { N \geq 1 }$ converges, as the width N of the hidden layer tends to infinity, towards a deterministic measure-valued process $( \bar { \mu } _ { t } ) _ { t \geq 0 }$ solving a nonlinear continuity equation. Beyond this law of large numbers, the fluctuations around the mean-field limit,

$$
\eta _ { t } ^ { N } : = \sqrt { N } \left( \mu _ { t } ^ { N } - \bar { \mu } _ { t } \right) ,
$$

satisfy a functional central limit theorem. A further analysis allows one to characterize explicitly the law of the limiting fluctuation process, yielding a complete probabilistic description of the uncertainty induced by finite-width efects. Since these results constitute the mathematical foundation of the present work, we review the mean-field framework in Section 2 together with the corresponding law of large numbers and central limit theorem in Section 3. In particular, the explicit characterization of the limiting Gaussian process established in the work by A.Descours, A.Guillin, G.Lacour, M.Michel, B.Nectoux & P.Stos [20, Theorem 3] will play a central role throughout the paper.

While these results provide a precise description of the parameter dynamics, they do not directly address the quantities that are actually observed in practical applications. Indeed, the empirical measure itself is not an observable of primary interest. Instead, one typically monitors quantities that are computed from it, such as the network prediction associated with a prescribed input, the empirical loss, or other statistics characterizing the state of the network. From the mean-field perspective, all these quantities can naturally be viewed as functionals acting on the empirical distribution of the parameters. This observation provides the starting point of the present work.

This motivates the introduction of the notion of an observation functional,

$$
\Phi : \mathcal { M } ( \mathbb { R } ^ { d } )  \mathbb { R } ^ { k } ,
$$

which associates with the empirical distribution of the network parameters a quantity of practical interest. Here, $\mathcal { M } ( \mathbb { R } ^ { d } )$ denotes the space of finite Borel measures on the parameter space $\mathbb { R } ^ { d } .$ while $k \in ( \mathbb { N } \backslash \{ 0 \} )$ is the dimension of the observed quantity.

A natural question is therefore whether the uncertainty described by the mean-field fluctuation theory propagates to such observation functionals. Since the asymptotic fluctuations of the empirical measure are completely characterized by the functional central limit theorem recalled in Section 3, one may ask whether an analogous central limit theorem holds for the observed quantities themselves.

Although observation functionals are naturally introduced as mappings acting on the space of measures, the fluctuation process takes values in the dual of a weighted Sobolev space of the form

$$
\mathcal { H } : = \mathcal { H } ^ { - J _ { 0 } + 1 , j _ { 0 } } ( \mathbb { R } ^ { d } ) ,
$$

for some parameters $J _ { 0 } , j _ { 0 }$ recalled in the statement of Theorem 1 below and whose structure is detailed in Section 3.1. Although Wasserstein spaces and duals of weighted Sobolev spaces carry diferent structures, probability measures satisfying suitable moment conditions can be canonically viewed in both settings. The latter additionally provide a linear Hilbert space framework, in which statistical, probabilistic and analytical tools take a particularly natural form.

More broadly, methods base on Sobolev spaces have been especially fruitful in the analysis of inverse problems for partial diferential equations arising from mean field models. Recent examples include the works by O.Imanuvilov, H.Liu, & M.Yamamoto [28] for state determination and inverse source problems in mean field game systems , as well as the inverse coeficient problem studied by O.Imanuvilov & M.Yamamoto [29]. Mean field game systems themselves arise as continuum limits of Nash equilibrium problems involving a large number of interacting players, as established in the works by J.-M.Lasry & P.-L.Lions [41, 42, 43].

Accordingly, throughout the theoretical analysis, observation functionals are understood as mappings defined on this dual space, which contains the empirical measures through the canonical embedding recalled in subsection 3.1. This leads to our first main result below.

Theorem 1 (Propagation of mean-field fluctuations to observation functionals). Assume that Assumptions A1–A6 hold and that $\beta > \frac { 3 } { 4 }$ . Let

$$
J _ { 0 } \ge 4 \bigg \lceil \frac { d } { 2 } \bigg \rceil + 8 , \quad j _ { 0 } = \bigg \lceil \frac { d } { 2 } \bigg \rceil + 2 ,
$$

set $\mathcal { H } : = \mathcal { H } ^ { - J _ { 0 } + 1 , j _ { 0 } } ( \mathbb { R } ^ { d } )$ and $\Phi : \mathcal { H }  \mathbb { R } ^ { k }$ be a Fréchet diferentiable observation functional. Then, for every $t \geq 0$

$$
\sqrt { N } \Big ( \Phi ( \mu _ { t } ^ { N } ) - \Phi ( \bar { \mu } _ { t } ) \Big ) \xrightarrow [ N \to \infty ] { \mathcal { L } } D \Phi ( \bar { \mu } _ { t } ) \bar { \eta } _ { t } .
$$

In particular, the limiting fluctuations of the observation functional are Gaussian.

Moreover, writing $\Phi = ( \Phi _ { 1 } , \ldots , \Phi _ { k } )$ and $Z _ { t } ^ { \Phi } : = D \Phi ( \bar { \mu } _ { t } ) \bar { \eta } _ { t }$ , the covariance matrix $\Sigma _ { t } ^ { \Phi }$ of $Z _ { t } ^ { \Phi }$ is given by

$$
\Sigma _ { t } ^ { \Phi } = \left( \Sigma _ { i , j } ^ { \Phi } ( t ) \right) _ { 1 \leq i , j \leq k } ,
$$

where

$$
\Sigma _ { i , j } ^ { \Phi } ( t ) = \operatorname { C o v } \Big ( D \Phi _ { i } ( \bar { \mu } _ { t } ) ( \bar { \eta } _ { t } ) , D \Phi _ { j } ( \bar { \mu } _ { t } ) ( \bar { \eta } _ { t } ) \Big ) .
$$

The proof relies on a functional Delta method, which provides a general mechanism for transferring central limit theorems through diferentiable mappings. Such techniques are classical in asymptotic statistics and have found numerous applications in infinite-dimensional settings, see for instance the works by E.Beutner [7], by E.Beutner & H.Zähle [8, 9], by J.Cupidon, D.Gilliam, R.Eubank & F.Ruymgaart [18], by R.Schmidt & U.Stadtmüller [56] or the books by P.Andersen, Ø.Borgan, R.Gill & N.Keiding [4, Section II.8], by M.Kosorok [34, Section 2.2.4. and Chapter 12] and A.Van der Vaart [63, Chapter 20]. Their application to mean-field models is, however, often considerably more delicate. Indeed, convergence towards the limiting fluctuation process is typically established through compactness arguments in spaces of probability measures, while diferentiability with respect to the measure variable is commonly formulated in terms of Lions derivatives (or L-derivatives). We refer for example to the works by R.Carmona & F.Delarue [15] or by B.Jourdain & A.Tse [33]. Although this framework is remarkably powerful, it generally requires rather strong regularity assumptions on the considered functionals.

The present setting turns out to be substantially simpler. Owing to the weighted Sobolev framework underlying the fluctuation theory, the empirical measures and their limiting fluctuations naturally evolve in a Banach space in which standard Fréchet diferentiability is suficient. As a consequence, the functional Delta method can be implemented under ordinary Fréchet diferentiability, without requiring a measure specific diferential calculus, while preserving an explicit Gaussian asymptotic description.

Corollary 2. Assume that the assumptions of Theorem 1 hold. If for every $i \in \{ 1 , \ldots , k \}$ , there exists $\varphi _ { i } ^ { t } \in C _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ such that

$$
D \Phi _ { i } ( \bar { \mu } _ { t } ) ( h ) = \langle \varphi _ { i } ^ { t } , h \rangle , \quad h \in \mathcal { H } ,
$$

then the covariance matrix $\Sigma _ { t } ^ { \Phi }$ of $Z _ { t }$ is given by

$$
\Sigma _ { i , j } ^ { \Phi } ( t ) = \operatorname { C o v } \left( f ^ { \varphi _ { i } ^ { t } } ( 0 , W _ { 0 } ^ { 1 } ) , f ^ { \varphi _ { j } ^ { t } } ( 0 , W _ { 0 } ^ { 1 } ) \right) + \alpha ^ { 2 } \mathbb { E } \left[ \frac { 1 } { | B _ { \infty } | } \right] \int _ { 0 } ^ { t } \operatorname { C o v } _ { \boldsymbol \pi } \left( Q _ { r } [ f ^ { \varphi _ { i } ^ { t } } ( r ) ] , Q _ { r } [ f ^ { \varphi _ { j } ^ { t } } ( r ) ] \right) d r .
$$

Here, $f ^ { \varphi _ { i } ^ { t } }$ denotes the solution of the backward transport equation (13) associated with the terminal datum $\varphi _ { i } ^ { t } .$ , as in Theorem 15.

The previous result explains how finite-width uncertainty propagates from the empirical parameter distribution to a prescribed observation functional. This raises a complementary question. Given a vector-valued quantity of interest

$$
Q : \mathcal { U } \subset \mathcal { H }  \mathbb { R } ^ { \ell } ,
$$

does the observation $\Phi ( \mu )$ contain enough information to determine $Q ( \mu )$ , at least locally? We say that $Q$ is locally identifiable from Φ near $\mu _ { 0 }$ in the diferentiable sense if there exists a continuously diferentiable mapping $F _ { ; }$ , defined near $\Phi ( \mu _ { 0 } )$ , such that

$$
Q = F \circ \Phi
$$

in a neighborhood of $\mu _ { 0 }$ . A necessary diferential condition for such a factorization is

$$
\operatorname { K e r } D \Phi ( \mu ) \subset \operatorname { K e r } D Q ( \mu )
$$

which means that every infinitesimal perturbation that is invisible to the observation must also leave the quantity of interest unchanged. Our second main result shows that, provided DΦ has constant finite rank near $\mu _ { 0 }$ , the validity of this kernel inclusion throughout a neighborhood is also suficient for the existence of the local nonlinear factorization having the form $Q = F \circ \Phi$

This viewpoint is closely related to classical local identifiability criteria for finite-dimensional parametric models, which are commonly formulated in terms of rank conditions on the Jacobian of the observation map. We refer, for instance to the works by X.Chen, V.Chernozhukov, S.Lee & W.Newey [16] or by T.Rothenberg [54] and the subsequent literature on parametric identifiability as in the book by E.Walter [66] and references therein. Identifiability questions also arise naturally in the inference of interacting particle systems, where the interaction laws governing the dynamics must be recovered from trajectory data. In this direction, F.Lu, M.Maggioni & S.Tang [45] developed nonparametric learning frameworks for interaction kernels in stochastic particle systems, while Z.Li, F.Lu, M.Maggioni, S.Tang & C.Zhang [44] established coercivity conditions ensuring identifiability and clarified their role in the mean field regime. At the level of the limiting equations, Q.Lang & F.Lu [40] further characterized the function spaces on which interaction kernels are identifiable from the corresponding mean field dynamics. These contributions illustrate the importance of identifiability questions for systems of interacting particles, both before and after passage to the mean field limit.

In particular, when the quantity of interest coincides with the full parameter vector, our criterion reduces in finite dimension to the usual full rank condition. More generally, it provides an infinite-dimensional local identifiability criterion for prescribed quantities of interest, even when the underlying parameter distribution itself is not identifiable.

Theorem 3 (Diferentiable local identifiability criterion). Let $\mathcal { U } \subset \mathcal { H }$ be open, and assume that $\Phi : \mathcal { U } \to \mathbb { R } ^ { k } , \ Q : \mathcal { U } \to \mathbb { R } ^ { \ell }$ are continuously Fréchet diferentiable mappings. Let $\mu _ { 0 } \in \mathcal { U }$ , and assume that DΦ has constant rank in a neighborhood of $\mu _ { 0 . }$ , namely that there exist an open neighborhood $\mathcal { U } _ { 0 } \subset \mathcal { U }$ of $\mu _ { 0 }$ and $r \in \{ 0 , \ldots , k \}$ such that for every $\mu \in { \mathcal { U } } _ { 0 }$ , it holds rk $D \Phi ( \mu ) = r$ Then the following assertions are equivalent.

(i) There exist open neighborhoods ${ \mathcal { U } } _ { 1 } \subset { \mathcal { U } } _ { 0 } o f { \boldsymbol { \mu } } _ { 0 }$ and $\mathcal { V } \subset \mathbb { R } ^ { k } o f \Phi ( \mu _ { 0 } )$ , together with a continuously diferentiable mapping $F : \mathcal { V } \to \mathbb { R } ^ { \ell }$ such that $\Phi ( \mathcal { U } _ { 1 } ) \subset \mathcal { V }$ and for every $\mu \in \mathcal { U } _ { 1 }$

$$
Q ( \mu ) = F ( \Phi ( \mu ) ) .
$$

(ii) There exists an open neighborhood $\mathcal { U } _ { 2 } \subset \mathcal { U } _ { 0 }$ of $\mu _ { 0 }$ such that for every $\mu \in { \mathcal { U } } _ { 2 }$

$$
\operatorname { K e r } D \Phi ( \mu ) \subset \operatorname { K e r } D Q ( \mu ) .
$$

The interpretation of Theorem 3 is as follows. Assertion (i) states that the quantity of interest Q can be reconstructed, in a neighborhood of $\mu _ { 0 }$ , by applying a continuously diferentiable mapping to the observation $\Phi ( \mu )$ , so that $Q$ is locally identifiable from Φ at $\mu _ { 0 }$ in a diferentiable sense. In particular, under either of the equivalent assertions, there exists a neighborhood $\mathcal { U } _ { 3 }$ of $\mu _ { 0 }$ such that, for every $\mu , \nu \in \mathcal { U } _ { 3 }$

$$
\Phi ( \mu ) = \Phi ( \nu ) \Rightarrow Q ( \mu ) = Q ( \nu ) .
$$

Thus, two nearby elements that are indistinguishable through the observation functional $\Phi$ are also indistinguishable with respect to the quantities of interest encoded by $Q .$

The geometric content of the result can be understood in terms of the local fibers of the observation functional. The constant rank assumption implies that, near $\mu _ { 0 }$ , the level sets $\Phi ^ { - 1 } ( y )$ are continuously diferentiable submanifolds of $\mathcal { H }$ , whose tangent spaces are given by

$$
T _ { \mu } \mathcal { F } _ { \mu } = \mathop { \mathrm { K e r } } D \Phi ( \mu ) ,
$$

where we set the fiber $\mathcal { F } _ { \mu } : = \{ \nu \in \mathcal { U } _ { 0 } \mid \Phi ( \nu ) = \Phi ( \mu ) \}$ . The kernel inclusion given in Assertion (ii) therefore means that the diferential of $Q$ vanishes along every direction tangent to the local fibers of Φ. Consequently, $Q$ is constant along each suficiently small connected piece of such a fiber. It can thus be regarded locally as a function of the observation $\Phi ( \mu )$ alone, which yields the desired factorization.

The theorem also provides a structural preliminary test for inverse problems. Before attempting to reconstruct $Q ( \mu )$ from $\Phi ( \mu )$ , one must verify that the selected observations do not discard directions along which the quantities of interest vary. The kernel criterion can therefore guide the design of informative observation functionals. Questions of stability, conditioning, regularization, and numerical reconstruction lie beyond the scope of the present work.

Typical observation functionals include network outputs evaluated at prescribed inputs, losses, and statistics of the parameter distribution, whereas quantities of interest may represent moments, predictive summaries, or efective physical parameters. The following example illustrates how our fluctuation result can be applied to an efective thermal difusivity coeficient reconstructed from a trained neural-network surrogate.

Example 4. Let us consider a finite dataset $\left\{ ( z _ { k } , y _ { k } ) \right\} _ { k = 1 } ^ { n } : = \left\{ ( ( t _ { k } , x _ { k } ) , y _ { k } ) \right\} _ { k = 1 } ^ { n } \subset \mathcal { X } \times \mathcal { Y }$ , where $\mathcal { X } = [ 0 , T ] \times \Omega$ , with $T > 0$ and a bounded domain $\Omega \subset \mathbb { R } ^ { m - 1 } , m \geq 2$ , and where $\mathcal { V } = \mathbb { R }$ . The observations $y _ { k } = \theta ( t _ { k } , x _ { k } )$ represent temperatures measured at spatial positions $x _ { k }$ and times $t _ { k }$

Our aim is to estimate the efective thermal difusivity of the material, viewed here as a quantity of interest $Q ( \mu )$ . We therefore seek a suitable observation functional $\Phi ( { \boldsymbol \mu } ^ { N } )$ from which this coeficient can be recovered. The neural network $\theta _ { \mu } : = \langle \sigma ^ { * } , \mu \rangle$ may be trained by SGD as a surrogate for the temperature field through the associated nonlinear regression problem.

Define the open subset $\mathcal { U } : = \{ \mu \in \mathcal { H } \mid \Delta _ { x } \theta _ { \mu } \neq 0 \mathrm { i n } L ^ { 2 } ( \mathcal { X } ) \}$ . For every $\mu \in { \mathcal { U } } .$ we may define the efective difusivity as the least-squares coeficient

$$
Q ( \mu ) : = \underset { \lambda \in \mathbb { R } } { \mathrm { a r g m i n } } \Vert \partial _ { t } \theta _ { \mu } - \lambda \Delta _ { x } \theta _ { \mu } \Vert _ { L ^ { 2 } ( \mathcal { X } ) } ^ { 2 } ,
$$

so that it is uniquely given by the formula

$$
Q ( \mu ) : = \frac { ( \partial _ { t } \theta _ { \mu } , \Delta _ { x } \theta _ { \mu } ) _ { L ^ { 2 } ( \mathcal { X } ) } } { \| \Delta _ { x } \theta _ { \mu } \| _ { L ^ { 2 } ( \mathcal { X } ) } ^ { 2 } } .
$$

Since the derivatives of the network can be evaluated by automatic diferentiation, they provide natural observation functionals. Also, under the standing regularity assumptions on the activation function, the mappings $v \mapsto \partial _ { t } \theta _ { \ i }$ and $v \mapsto \Delta _ { x } \theta _ { v }$ are continuous from $\mathcal { H }$ into $L ^ { 2 } ( \mathcal { X } )$ . More precisely, for every $z \in \mathcal { X }$ 2

$$
| \partial _ { t } \theta _ { v } ( z ) | \leq \| v \| _ { \mathcal { H } } \| \partial _ { t } \sigma ^ { * } ( \cdot , z ) \| _ { \mathcal { H } ^ { \prime } } ,
$$

and similarly

$$
| \Delta _ { x } \theta _ { v } ( z ) | \leq \| v \| _ { \mathcal { H } } \| \Delta _ { x } \sigma ^ { * } ( \cdot , z ) \| _ { \mathcal { H } ^ { \prime } } .
$$

Thanks to assumption A2 (see Section 3.2) on $\sigma ,$ together with the finite measure of $x ,$ , then yield

$$
\| \partial _ { t } \theta _ { v } \| _ { L ^ { 2 } ( \mathcal { X } ) } + \| \Delta _ { x } \theta _ { v } \| _ { L ^ { 2 } ( \mathcal { X } ) } \leq C \| v \| _ { \mathcal { H } } .
$$

Also, note that the above continuity of $\mu \mapsto \Delta _ { x } \theta _ { \mu }$ from $\mathcal { H }$ into $L ^ { 2 } ( \mathcal { X } )$ shows that $\mathcal { U }$ is open. Therefore, setting

$$
\Phi ( \mu ^ { N } ) : = \left( \begin{array} { c } { { ( \partial _ { t } \theta _ { \mu ^ { N } } , \Delta _ { x } \theta _ { \mu ^ { N } } ) _ { L ^ { 2 } ( \mathcal X ) } } } \\ { { \| \Delta _ { x } \theta _ { \mu ^ { N } } \| _ { L ^ { 2 } ( \mathcal X ) } ^ { 2 } } } \end{array} \right)
$$

and the function $F : \mathbb { R } \times ( 0 , + \infty ) \to \mathbb { R } , ( \alpha , \beta ) \mapsto \alpha / \beta ,$ we get that $Q ( \mu ^ { N } ) = F ( \Phi ( \mu ^ { N } ) )$ . Now, since for $v \in \mathcal { H }$ , it holds that $\partial _ { t } \theta _ { v } = \langle \partial _ { t } \sigma , v \rangle$ and $\Delta _ { x } \theta _ { v } = \langle \Delta _ { x } \sigma , v \rangle$ , a straightforward calculation combined with Theorem 1 implies that the limiting fluctuation process of observables is given by

$$
\begin{array} { c } { { D \Phi ( \bar { \mu } _ { s } ) ( \bar { \eta } _ { s } ) = \left( ^ { \left( \partial _ { t } \theta _ { \bar { \eta } _ { s } } , \Delta _ { x } \theta _ { \bar { \mu } _ { s } } \right) _ { L ^ { 2 } ( \mathcal X ) } + { \left( \partial _ { t } \theta _ { \bar { \mu } _ { s } } , \Delta _ { x } \theta _ { \bar { \eta } _ { s } } \right) _ { L ^ { 2 } ( \mathcal X ) } } } \right) , } } \\ { { 2 ( \Delta _ { x } \theta _ { \bar { \mu } _ { s } } , \Delta _ { x } \theta _ { \bar { \eta } _ { s } } ) _ { L ^ { 2 } ( \mathcal X ) } } } \end{array}
$$

where $\bar { \mu } _ { t }$ and $\bar { \eta } _ { t }$ are respectively the mean field limit and its associated limiting fluctuation process provided by Theorem 10 and Theorem 14 of Section 3. Now since U is open and $\mu _ { s } ^ { N } \to \bar { \mu } _ { s }$ in probability in $\mathcal { H }$ , the condition $\bar { \mu } _ { s } \in \mathcal { U }$ implies

$$
\mathbb { P } ( \mu _ { s } ^ { N } \in \mathcal { U } ) \longrightarrow 1 .
$$

On the event $\{ \mu _ { s } ^ { N } \notin \mathcal { U } \}$ , we may define $Q ( \mu _ { s } ^ { N } )$ arbitrarily, for instance by setting $Q ( \mu _ { s } ^ { N } ) = 0$ Since the probability of the above complementary event tends to zero, the preceding expansion holds with probability tending to one, namely this arbitrary extension has no efect on the limiting distribution. The functional Delta method provided by Theorem 1 may therefore be applied locally to $Q$ at $\bar { \mu } _ { s }$ . Thus, whenever $\Delta _ { x } \theta _ { \bar { \mu } _ { s } } \neq 0$ in $L ^ { 2 } ( \mathcal { X } )$ , Theorem 1 and the chain rule yield

$$
\begin{array} { r } { \sqrt { N } \big ( Q ( \mu _ { s } ^ { N } ) - Q ( \bar { \mu } _ { s } ) \big ) \xrightarrow [ N \to + \infty ] { \mathcal L } D Q ( \bar { \mu } _ { s } ) ( \bar { \eta } _ { s } ) , } \end{array}
$$

where for every $\mu , \nu \in \mathcal { U } \times \mathcal { H }$ we have

$$
\begin{array} { r } { D Q ( \mu ) ( \nu ) = \frac { ( \partial _ { t } \theta _ { \mu } , \Delta _ { x } \theta _ { \nu } ) _ { L ^ { 2 } ( \mathcal { X } ) } + ( \partial _ { t } \theta _ { \nu } , \Delta _ { x } \theta _ { \mu } ) _ { L ^ { 2 } ( \mathcal { X } ) } } { \| \Delta _ { x } \theta _ { \mu } \| _ { L ^ { 2 } ( \mathcal { X } ) } ^ { 2 } } } \\ { - 2 \frac { ( \partial _ { t } \theta _ { \mu } , \Delta _ { x } \theta _ { \mu } ) _ { L ^ { 2 } ( \mathcal { X } ) } ( \Delta _ { x } \theta _ { \mu } , \Delta _ { x } \theta _ { \nu } ) _ { L ^ { 2 } ( \mathcal { X } ) } } { \| \Delta _ { x } \theta _ { \mu } \| _ { L ^ { 2 } ( \mathcal { X } ) } ^ { 4 } } . } \end{array}
$$

The limiting variable is Gaussian, and its variance is given by Corollary 2 whenever $D Q ( \bar { \mu } _ { s } )$ admits the required representation. Moreover, since $Q = F \circ \Phi$ on U, the efective difusivity is locally identifiable from the two components of $\Phi ,$ , even though the full parameter distribution need not be identifiable. ⋄

The remainder of the paper is organized as follows. In Section 2, we introduce the mean field formulation of shallow neural network training and describe the stochastic gradient dynamics considered throughout the paper. Section 3 recalls the functional framework and the main results on the mean field limit and its asymptotic fluctuations that provide the starting point of our analysis. In Section 4, we prove the propagation of the fluctuation theory to nonlinear observation functionals, derive the corresponding covariance representation, and establish the factorization criterion for local identifiability. Finally, Appendix A provides the functional analytic arguments underlying the Hilbert-Schmidt embeddings between weighted Sobolev spaces used throughout the analysis.

## 2 Mean-field dynamics of shallow neural networks training

In recent years, considerable efort has been devoted to the mathematical analysis of neural network training dynamics in the large-width regime. This has led to a remarkably rich literature at the intersection of stochastic analysis, interacting particle systems, optimal transport, and mean-field theory, we refer e.g. to the works by F.Bach & L.Chizat [6], by A.Descours, A.Guillin, M.Michel & B.Nectoux [19], by A.Descours, A.Guillin, G.Lacour, M.Michel, B.Nectoux & P.Stos [20] by B.Geshkovski, C.Letrouit, Y.Polyanski & P.Rigollet [23], by A.Guillin, B.Nectoux & P.Stos [25], by S.Mei, A.Montanari & P.-M.Nguyen [46] and by J.Sirignano & K.Spiliopoulos [57], as to the books by F.Bach [5] or by J.Sirignano, R.Sowers & K.Spiliopoulos [58] and references therein for an overview of those approaches. One of the main mathematical dificulties lies in the nonconvex nature of the optimization landscape associated with neural network training. In high dimension, the stochastic gradient descent dynamics governing the evolution of the parameters may therefore appear analytically intractable at first sight.

A fruitful approach consists in representing the collection of network parameters through its associated empirical measure and studying the corresponding evolution in the space of probability measures. In this formulation, the training dynamics can be interpreted as a mean-field interacting particle system, whose large-width limit is described by a deterministic evolution equation on measures. Such a perspective provides a mathematically tractable framework for the rigorous analysis of neural network training in the infinite-width regime. Beyond the deterministic mean-field limit itself, an important question concerns the description of finite-width efects and the asymptotic fluctuations of the empirical measure around its limiting dynamics.

More precisely, we consider the following single-hidden-layer neural network

$$
g _ { W } ^ { N } ( x ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sigma ^ { * } ( W ^ { i } , x ) ,
$$

where $x \in \mathbb { R } ^ { m }$ denotes the input data, $y \in \mathbb { R }$ the associated label, and $\sigma ^ { * } : \mathbb { R } ^ { d } \times \mathbb { R } ^ { m }  \mathbb { R }$ is the activation function. The parameter vector $W = ( W ^ { 1 } , \dots , W ^ { N } ) \in ( \mathbb { R } ^ { d } ) ^ { N }$ collects the weights of the network, while N denotes the number of neurons. We assume that the data are sampled from an unknown probability distribution $( x , y ) \sim \pi$ , where π is not explicitly known. The objective of the training procedure is to determine weights W such that the network prediction $g _ { W } ^ { N } ( x )$ approximates the target value y in a suitable sense, which is given through an optimization problem. In the present work, we consider the quadratic loss

$$
\mathcal { R } ( W ) = \frac { 1 } { 2 } \mathbb { E } _ { ( x , y ) \sim \pi } \left[ | g _ { W } ^ { N } ( x ) - y | ^ { 2 } \right] ,\tag{1}
$$

so that the training of the weights therefore amounts to solve

$$
\operatorname* { m i n } _ { W \in ( \mathbb { R } ^ { d } ) ^ { N } } \mathcal { R } ( W ) .\tag{2}
$$

In practice, the network is trained through the stochastic gradient descent algorithm (see e.g. the book by F.Bach [5, Chapter 5, Section 4], the book by J.Sirignano, R.Sowers & K.Spiliopoulos [58,

Chapters $7$ and 18]). Denoting by $\alpha > 0$ the learning rate, the dynamics of the weights is given, for $k \geq 0$ and $i \in \{ 1 , \ldots , N \}$ , by

$$
\left\{ \begin{array} { l l } { \displaystyle W _ { k + 1 } ^ { i } = W _ { k } ^ { i } + \frac { \alpha } { N | B _ { k } | } \sum _ { ( x , y ) \in B _ { k } } ( y - g _ { W _ { k } } ^ { N } ( x ) ) \nabla _ { W } \sigma ^ { * } ( W _ { k } ^ { i } , x ) + \frac { \varepsilon _ { k } ^ { i } } { N ^ { \beta } } , } \\ { W _ { 0 } ^ { i } \sim \mu _ { 0 } , } \end{array} \right.\tag{SGD}
$$

where $\varepsilon _ { k } ^ { i } \sim \mathcal N ( 0 , I _ { d } )$ and $\mu _ { 0 } \in \mathcal P ( \mathbb { R } ^ { d } )$

Here, $B _ { k }$ denotes the mini-batch used at iteration $k ,$ namely a finite random subset of the dataset sampled according to the underlying distribution π. The use of mini-batches is standard in modern machine learning, as it provides computationally eficient approximations of the full gradient while preserving the stochastic nature of the training dynamics (we refer the interested reader to the book by J.Sirignano, R.Sowers & K.Spiliopoulos [58, Chapter 10]). The additional Gaussian perturbation models an external source of noise in the optimization procedure. Such noisy stochastic gradient dynamics naturally arise in several contexts, including regularization mechanisms, stochastic approximations of optimization procedures, or efective descriptions of finite-width training efects. The parameter $\beta > 0$ governs the asymptotic strength of the noise in the large-width regime.

Observe that the stochastic gradient descent dynamics (SGD) does not directly involve the full gradient of the risk functional (1). Since the underlying distribution $\pi$ is unknown in practice, the gradient of the risk cannot be computed explicitly. Instead, the algorithm relies on a random empirical approximation, namely the i-th component of this gradient is approximated using the mini-batch estimator

$$
\widehat { \nabla _ { W ^ { i } } \mathcal { R } } ( W _ { k } ; B _ { k } ) : = \frac { 1 } { N | B _ { k } | } \sum _ { ( x , y ) \in B _ { k } } \left( g _ { W _ { k } } ^ { N } ( x ) - y \right) \nabla _ { w } \sigma ^ { * } ( W _ { k } ^ { i } , x ) .
$$

which acts as a stochastic estimator of the expected gradient associated with the loss functional. Thus, the update rule can be written as

$$
\widehat { W _ { k + 1 } ^ { i } } = \widehat { W _ { k } ^ { i } - \alpha \widehat { \nabla _ { W ^ { i } } \mathcal { R } } ( W _ { k } ; B _ { k } ) } + \frac { \varepsilon _ { k } ^ { i } } { N ^ { \beta } } .
$$

While this stochastic approximation circumvents the explicit dependence on the unknown distribution π, the resulting optimization dynamics nevertheless remains highly nonlinear with respect to the weights W. This nonlinearity reflects the non-convex nature of the original optimization problem (2), and makes the direct analysis of the large-width regime particularly delicate. A key observation is that the network output can naturally be rewritten in terms of the empirical distribution of the parameters. More precisely, introducing the empirical measure

$$
\nu ^ { N } = \frac { \ o { 1 } } { N } \sum _ { i = 1 } ^ { N } \delta _ { W ^ { i } } ,
$$

we may rewrite the network prediction as

$$
g _ { W } ^ { N } ( x ) = \langle \sigma ^ { * } ( \cdot , x ) , \nu ^ { N } \rangle ,
$$

where $\langle \cdot , \cdot \rangle$ is a duality bracket on a suitable measure space. This reformulation allows the training dynamics to be interpreted in terms of the empirical distribution of the parameters in $\mathbb { R } ^ { d }$ Therefore, the original optimization problem on the weights given by 2 induces a corresponding optimization problem on probability measures. Formally, introducing

$$
g _ { \nu } ( x ) = \int _ { \mathbb { R } ^ { d } } \sigma ^ { * } ( w , x ) d \nu ( w ) ,
$$

we may rewrite the loss functional (1) as

$$
\mathcal { R } ( \nu ) = \frac { 1 } { 2 } \mathbb { E } _ { ( x , y ) \sim \pi } \left[ | g _ { \nu } ( x ) - y | ^ { 2 } \right] ,
$$

letting us consider the relaxed problem

$$
\operatorname* { m i n } _ { \nu \in \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) } \mathcal { R } ( \nu ) .\tag{3}
$$

In the above formulation, the optimization problem is no longer posed on the finite-dimensional parameter vector space of the weights, but on the space of probability measures $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$ , which is the space of probability measures with bounded moment of order $\gamma > 0$

Moreover, due to the linearity of $g _ { \nu }$ with respect to $\nu ,$ the functional $\mathcal { R } ( \nu )$ becomes convex in the measure variable. This relaxed variational perspective has played an important role in the mathematical analysis of wide neural networks, we refer for instance to the book of F.Bach [5, Chapter 4] and references therein.

Beyond its variational interpretation, this measure-valued formulation naturally leads to a mean field description of the stochastic gradient descent dynamics. In the large-width regime, the empirical measure evolves as a system of interacting particles whose asymptotic behavior can be analyzed through mean-field techniques, in close analogy with classical models from statistical physics. Such an approach provides a rigorous framework for the study of infinite-width neural network training and its associated fluctuation theory. We refer for instance to the monograph of H.Spohn [61] for background on mean-field interacting particle systems.

To describe such dynamics, we introduce the rescaled process

$$
\mu _ { t } ^ { N } : = \nu _ { \lfloor N t \rfloor } ^ { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \delta _ { W _ { \lfloor N t \rfloor } ^ { i } } ,
$$

and rewrite the stochastic gradient descent algorithm in terms of $\mu _ { t } ^ { N }$ instead of $\nu _ { k } ^ { N }$ , without any specific change. Under some reasonable assumptions, one can expect to obtain a deterministic limit for the infinite-width regime, which is the object of Theorem 10 recalled in Section 3 below.

Nevertheless, it should be noted that, from a computational standpoint, training a neural network never reaches such a limit but is restricted to a hidden layer of very large width N. It is therefore natural to ask to what extent the process $\mu _ { t } ^ { N }$ fluctuates around its mean-field limit $\bar { \mu } _ { t }$ Defining the fluctuation process $\eta _ { t } ^ { N } = \sqrt { N } ( \mu _ { t } ^ { N } - \bar { \mu } _ { t } )$ , we can establish the existence of a limiting fluctuation process $\bar { \eta } _ { t }$ that satisfies a stochastic partial diferential equation. This result is given by Theorem 14 below.

Interestingly, the stochastic evolution equation (11) is linear in $\bar { \eta }$ and is driven by the Gaussian process $\mathcal { G } .$ It is therefore natural to expect the limiting fluctuation process $\bar { \eta }$ to remain Gaussian, as described in the book by J.Sirignano, R.Sowers & K.Spiliopoulos [58, Section 20.4]. In order to obtain a complete uncertainty quantification result, one must moreover characterize its covariance explicitly. This is the purpose of Theorem 15, recalled in the next section.

## 3 Preliminary results

In this section, we recall the mathematical framework underlying the mean-field analysis of shallow neural networks. We first introduce the functional spaces in which the empirical measures and their fluctuations will be studied in subsection 3.1. In subsection 3.2 we then summarize the assumptions required for the existing law of large numbers and functional central limit theorem, and recall these results in a form adapted to the present work. Finally, we state the explicit characterization of the limiting fluctuation process established in the work by A.Descours, A.Guillin, G.Lacour, M.Michel, B.Nectoux & P.Stos [20, Theorem 3] in Theorem 15, which constitutes the starting point of our analysis.

## 3.1 Functional framework

We first recall the functional framework inherent to the results. The state of the system is described by probability measures on $\mathbb { R } ^ { d }$ , while the fluctuation processes will be viewed as distribution-valued paths. We therefore introduce, in a systematic way, the spaces of probability measures, the path spaces, and the weighted Sobolev spaces used below.

Definition 5. Let $k \geq 1$ . We denote by $\mathcal { P } _ { k } ( \mathbb { R } ^ { d } )$ the set of Borel probability measures on $\mathbb { R } ^ { d }$ with finite k-th moment, namely

$$
\mathcal { P } _ { k } ( \mathbb { R } ^ { d } ) : = \left\{ \mu \in \mathcal { P } ( \mathbb { R } ^ { d } ) \mid \int _ { \mathbb { R } ^ { d } } \vert x \vert ^ { k } \mathrm { d } \mu ( x ) < + \infty \right\} .
$$

The space $\mathcal { P } _ { k } ( \mathbb { R } ^ { d } )$ is endowed with the Wasserstein distance

$$
W _ { k } ( \mu , \nu ) = \left( \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } | x - y | ^ { k } { \mathrm { d } } \pi ( x , y ) \right) ^ { 1 / k } ,
$$

where $\Pi ( \mu , \nu )$ denotes the set of probability measures on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ with marginals respectively denoted $\mu$ and $\nu .$

We recall that $W _ { 1 } ( \mu , \nu ) ~ \leq ~ W _ { k } ( \mu , \nu )$ for all $\mu , \nu \in \mathcal P _ { k } ( \mathbb R ^ { d } )$ and all $k \geq 1$ . Moreover, the Kantorovich-Rubinstein formula gives

$$
W _ { 1 } ( \mu , \nu ) = \operatorname* { s u p } _ { [ f ] _ { \mathrm { L i p } } \leq 1 } \left| \int _ { \mathbb { R } ^ { d } } f \mathrm { d } \mu - \int _ { \mathbb { R } ^ { d } } f \mathrm { d } \nu \right| ,
$$

where the supremum is taken over all real-valued Lipschitz functions $f$ on $\mathbb { R } ^ { d }$ with Lipschitz seminorm bounded by one. We refer for instance to the books L.Ambrosio, N.Gigli & G.Savaré [3, Chapter 7], by F.Santambrogio [55, Chapter 5] or by C.Villani [65, Chapter 6] for further background on Wasserstein metric and spaces.

Throughout this section, the empirical measure process is denoted by $\mu ^ { N } : = ( \mu _ { t } ^ { N } ) _ { t \geq 0 }$ . Under the moment assumptions imposed below, $\mu ^ { N }$ is naturally viewed as a random variable with values in the Skorokhod space ${ \mathcal { D } } ( \mathbb { R } _ { + } , { \mathcal { P } } _ { q } ( \mathbb { R } ^ { d } ) )$ ), for suitable $q \geq 1$

Definition 6. Let $( E , d _ { E } )$ be a metric space. We denote by $\mathcal { D } ( \mathbb { R } _ { + } , E )$ the space of càdlàg functions from $\mathbb { R } _ { + }$ into E, that is, the space of functions $u : \mathbb { R } _ { + } \to E$ which are right-continuous and admit left limits at every positive time. When E is a Polish space, $\mathcal { D } ( \mathbb { R } _ { + } , E )$ is endowed with the usual Skorokhod $J _ { 1 }$ topology.

Roughly speaking, convergence in $J _ { 1 }$ topology allows small perturbations of the time variable in order to align jump times, while preserving the chronological order of the jumps. More precisely, a sequence $( x _ { n } )$ converges to $x$ for the $J _ { 1 }$ topology if there exist increasing homeomorphisms of the time interval converging uniformly to the identity such that the time-changed paths converge uniformly. Thus, the Skorokhod space is the natural state space for empirical measure processes with jumps. We shall use it both with $E = \mathcal { P } _ { q } ( \mathbb { R } ^ { d } )$ and with $E { \mathrm { ~ a ~ } }$ dual weighted Sobolev space.

We mainly refer to the works by V.Bogachev [11], by A.Jakubowski [32] or the books by P.Billingsley [10, Chapter 3], by S.Ethier & T.Kurtz [21, Chapter 3], by J.Jacod & A.Shiryaev [31,

Chapter VI] and by D.Pollard [51, Chapters V and VI] for a detailed construction as well as an overview of the topological poperties for such spaces.

The fluctuation processes are distribution-valued. To make this precise, we use weighted Sobolev spaces. The weights allow test functions and their derivatives to grow at most polynomially at infinity, while preserving a Hilbert space structure.

Definition 7. Let $J \in \mathbb { N }$ and $a \geq 0$ . For $g \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , set

$$
\| g \| _ { \mathcal H ^ { J , a } } ^ { 2 } : = \sum _ { | \alpha | \leq J } \int _ { \mathbb R ^ { d } } \frac { | D ^ { \alpha } g ( x ) | ^ { 2 } } { 1 + | x | ^ { 2 a } } \mathrm { d } x .
$$

We define $\mathcal { H } ^ { J , a } ( \mathbb { R } ^ { d } )$ as the completion of $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ for this norm.

The space $\mathcal { H } ^ { J , a } ( \mathbb { R } ^ { d } )$ as defined above is a Hilbert space. The associated inner product is denoted by $( \cdot , \cdot ) _ { J , a }$ . We denote by $\mathcal { H } ^ { - J , a } ( \mathbb { R } ^ { d } )$ its topological dual. If $\Phi \in { \mathcal { H } } ^ { - J , a } ( \mathbb { R } ^ { d } )$ and $f \in \mathcal { H } ^ { J , a } ( \mathbb { R } ^ { d } )$ ， we write

$$
\langle f , \Phi \rangle _ { J , a } : = \Phi [ f ] .
$$

Of course, one can apply Riesz representation theorem to identify $\mathcal { H } ^ { J , a }$ with its topological dual. Under this identification, the duality pairing $\langle \cdot , \cdot \rangle _ { J , a }$ coincides with the Hilbert inner product $( \cdot , \cdot ) _ { J , a }$ . Whenever no confusion may arise, we omit both this identification and the indices from the notation.

Definition 8. Let $J \in \mathbb { N }$ and $a \geq 0$ . We define ${ \mathcal { C } } ^ { J , a } ( \mathbb { R } ^ { d } )$ as the space of functions $f : \mathbb { R } ^ { d }  \mathbb { R }$ with continuous partial derivatives up to order J such that, for every multi-index α with $| \alpha | \le J$

$$
\operatorname * { l i m } _ { | x |  \infty } { \frac { | D ^ { \alpha } f ( x ) | } { 1 + | x | ^ { a } } } = 0 .
$$

It is endowed with the norm

$$
\| f \| _ { \mathcal { C } ^ { J , a } } : = \sum _ { | \alpha | \leq J } \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } \frac { | D ^ { \alpha } f ( x ) | } { 1 + | x | ^ { a } } .
$$

We also denote by $\mathcal { C } _ { b } ( \mathbb { R } ^ { d } )$ the space of bounded continuous functions on $\mathbb { R } ^ { d }$ , endowed with the supremum norm, and by $\mathcal { C } _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ the space of smooth functions whose derivatives of all orders are bounded. Observe that if $b > d / 2$ , then Hölder inequality leads to

$$
\mathcal { C } _ { b } ^ { \infty } ( \mathbb { R } ^ { d } ) \subset \mathcal { H } ^ { J , b } ( \mathbb { R } ^ { d } )
$$

for every $J \in \mathbb { N }$

The weighted Sobolev spaces introduced above retain many of the fundamental properties of the classical Sobolev spaces. In particular, they satisfy the embedding results recalled below, which play a central role in the functional analysis of the fluctuation process. The first one is a weighted Sobolev embedding into spaces of functions with polynomial growth, while the second one is a Hilbert-Schmidt embedding between weighted Sobolev spaces.

Proposition 9 (Weighted Sobolev embeddings). Let $j \in \mathbb { N } , a \geq 0 , b > d / 2$ , and $\ell > d / 2$ . Then the following embeddings hold

$$
\begin{array} { r } { \mathcal { H } ^ { \ell + j , a } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { C } ^ { j , a } ( \mathbb { R } ^ { d } ) , } \end{array}
$$

and

$$
\begin{array} { r } { \mathcal { H } ^ { \ell + j , a } ( \mathbb { R } ^ { d } ) \stackrel { \mathrm { H . S . } } { \longrightarrow } \mathcal { H } ^ { j , a + b } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

Here ${ \stackrel { \mathrm { H . S . } } { \hookrightarrow } }$ means that the canonical embedding is a Hilbert-Schmidt operator.

The continuous embedding into $\mathcal { C } ^ { j , a }$ follows from the classical Sobolev embedding theorem combined with the polynomial weight. For background on Sobolev embeddings and weighted Sobolev spaces, we refer for instance to the works by B.Fernandez & S.Méléard [22] or by M.Izuki, T.Nogayama, T.Noi & Y.Sawano [30] as well as to the books by R.Adams & J.Fournier [2, Section 4.56], by J.Heinonen, T.Kilpeläinen & O.Martio [26, Chapter 1], by A.Kufner [36] or by H.Triebel [62, Section 7.2]. The only difcult point of Proposition 9 concerns the Hilbert-Schmidt property, which does not directly follow from a representation of Bessel potentials. Since this point is often used implicitly in the literature, we recall the argument in Appendix A, but we refer to the work by T.Krainer & B.-W.Schulze [35, Chapter 4, Theorem 4.1.5-d.] for more general considerations.

We first introduce the indices used in the law of large numbers given by

$$
L = \left\lceil \frac { d } { 2 } \right\rceil + 3 , \quad \gamma = 4 \left\lceil \frac { d } { 2 } \right\rceil + 5 , \quad \mathrm { a n d } \quad \gamma ^ { * } = \gamma + 1 .
$$

Since $L - 2 > d / 2$ , Proposition 9 yields the continuous embedding

$$
\mathcal { H } ^ { L , \gamma } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { C } ^ { 2 , \gamma } ( \mathbb { R } ^ { d } ) .
$$

Moreover, since $\gamma ^ { * } > \gamma$ , one has

$$
\begin{array} { r } { \mathcal { C } ^ { 2 , \gamma } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { C } ^ { 2 , \gamma ^ { * } } ( \mathbb { R } ^ { d } ) , } \end{array}
$$

and therefore

$$
\begin{array} { r } { \mathcal { H } ^ { L , \gamma } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { C } ^ { 2 , \gamma ^ { * } } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

Since every twice continuously diferentiable function with polynomial growth of order $\gamma ^ { * }$ is, in particular, a continuous function with the same polynomial growth,

$$
\begin{array} { r } { \mathcal { C } ^ { 2 , \gamma ^ { * } } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { C } ^ { 0 , \gamma ^ { * } } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

Passing to the dual spaces therefore gives the continuous embeddings

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { C } ^ { 0 , \gamma ^ { * } } ( \mathbb { R } ^ { d } ) ^ { \prime } \hookrightarrow \mathcal { C } ^ { 2 , \gamma ^ { * } } ( \mathbb { R } ^ { d } ) ^ { \prime } \hookrightarrow \mathcal { H } ^ { - L , \gamma } ( \mathbb { R } ^ { d } ) . } \end{array} } \end{array}
$$

Consequently, every finite Borel measure with finite $\gamma ^ { * } \mathrm { - t h }$ moment naturally defines an element of $\mathcal { H } ^ { - L , \gamma } ( \mathbb { R } ^ { d } )$ . This space is the negative-order Sobolev space naturally associated with the law of large numbers. The fluctuation central limit theorem is formulated in a weaker negativeorder Sobolev space, which we now introduce. Following the notation introduced in the work by A.Descours, A.Guillin, M.Michel & B.Nectoux [19], let

$$
J _ { 0 } \ge 4 \bigg \lceil \frac { d } { 2 } \bigg \rceil + 8 , \quad j _ { 0 } = \bigg \lceil \frac { d } { 2 } \bigg \rceil + 2 ,
$$

and set

$$
\mathcal { H } : = \mathcal { H } ^ { - J _ { 0 } + 1 , j _ { 0 } } ( \mathbb { R } ^ { d } ) .
$$

The embeddings used in the fluctuation analysis yield $\mathcal { H } ^ { J _ { 0 } - 1 , j _ { 0 } } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } ^ { L , \gamma } ( \mathbb { R } ^ { d } )$ , so that by duality,

$$
\mathcal { H } ^ { - L , \gamma } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } .\tag{4}
$$

Thus, the empirical measures and the mean field limit may be considered as elements of ${ \mathcal { H } } .$ while the fluctuation central limit theorem is naturally stated in this latter space.

## 3.2 Assumptions and known results

We begin by recalling the assumptions under which the mean-field limit and the corresponding fluctuation theorem have been established. Throughout the paper, the stochastic gradient dynamics generates a natural filtration describing the information available up to each optimization step. For every $N \geq 1$ , we define

$$
\mathcal { F } _ { 0 } ^ { N } = \sigma \biggl ( \{ W _ { 0 } ^ { i } \} _ { i = 1 } ^ { N } \biggr ) ,
$$

and, for every $k \geq 1$

$$
\mathcal { F } _ { k } ^ { N } = \sigma \biggl ( W _ { 0 } ^ { i } , \{ B _ { j } \} _ { j = 0 } ^ { k - 1 } , \{ \varepsilon _ { j } ^ { i } \} _ { j = 0 } ^ { k - 1 } \biggr ) , i \in \{ 1 , \ldots , N \} .
$$

In the above, σ denotes the σ-algebra generated by the corresponding random variables. The assumptions used throughout the paper are summarized below.

A1 (Data sampling). For every $k \geq 0$ , the collection $\left( \left( x _ { k } ^ { n } , y _ { k } ^ { n } \right) \right) _ { n > 1 }$ consists of independent and identically distributed random variables with common distribution $\pi \in \mathcal P ( \mathcal X \times \mathcal y )$ . Moreover,

• the collections $\left( \left( x _ { k } ^ { n } , y _ { k } ^ { n } \right) \right) _ { n > 1 }$ are independent for diferent values of k,

• for every $k \geq 0$ , the pair $\left( | B _ { k } | , \left( \left( x _ { k } ^ { n } , y _ { k } ^ { n } \right) \right) _ { n > 1 } \right)$ is independent of $\mathcal { F } _ { k } ^ { N }$

• for every $q , k \ge 0 , | B _ { q } |$ is independent of $\left( \left( x _ { k } ^ { n } , y _ { k } ^ { n } \right) \right) _ { n \geq 1 }$

• the output variable satisfies E $\left[ | y | ^ { 1 6 \gamma ^ { * } } \right] < + \infty$

A2 (Regularity of the activation function). The activation function $\sigma ^ { * } : \mathbb { R } ^ { d } \times \mathcal { X } \longrightarrow \mathbb { R }$ belongs to $\mathcal { C } _ { b } ^ { \infty } ( \mathbb { R } ^ { d } \times \mathcal { X } )$

A3 (Initialization). The initial parameters $\{ W _ { 0 } ^ { i } \} _ { i = 1 } ^ { N }$ are i.i.d. with common law $\mu _ { 0 } \in \mathcal P ( \mathbb { R } ^ { d } )$ satisfying $\mathbb { E } \big [ | W _ { 0 } ^ { 1 } | ^ { 8 \gamma ^ { * } } \big ] < + \infty$

A4 (Exploration noise). The family $\left( \varepsilon _ { k } ^ { i } \right) _ { 1 < i < N } ^ { k \geq 0 }$ consists of independent Gaussian random vectors with distribution $\mathcal { N } ( 0 , I _ { d } )$ . Moreover, for every $k \geq 0$ and every $i \in \{ 1 , \ldots , N \}$ the random variable $\varepsilon _ { k } ^ { i }$ is independent of $\mathcal { F } _ { k } ^ { N }$

The previous assumptions are standard in the mean-field analysis of stochastic gradient methods. For the reader’s convenience, we briefly comment on their respective roles. Assumption A1 describes the learning framework. At each optimization step, the algorithm receives a fresh minibatch, sampled independently of both the past trajectory and the current state of the network. The moment assumption on the output variable is used to derive uniform moment estimates for the system and its mean-field limit. Assumption A2 is a regularity assumption on the activation function. It ensures that the drift of the limiting mean-field equation is suficiently smooth and that all the derivatives appearing in the fluctuation analysis are well defined. Assumption A3 states that the neurons are initially exchangeable. The moment condition propagates along the dynamics and is required in several tightness and stability arguments. Finally, Assumption A4 models the exploration noise added to the stochastic gradient dynamics. Its independence with respect to the natural filtration expresses that the perturbation introduced at each optimization step is independent of the past history of the algorithm.

We are now in a position to recall the mean-field limit established in the work by A.Descours, A.Guillin, M.Michel & B.Nectoux [19, Theorem 1]. We highlight that similar results are proved in the works by Z.Chen, G.M.Rotskof, J.Bruna & E.Vanden-Eijnden [17], by S.Mei, A.Montanari & P.-M.Nguyen [46], or in particular by J.Sirignano & K.Spiliopoulos [57]. Roughly speaking, as the width of the neural network tends to infinity, the empirical distribution of the parameters converges towards a deterministic measure-valued trajectory governed by a nonlinear evolution equation.

Theorem 10 (Mean-field limit). Assume that $\begin{array} { r } { \beta > \frac { 1 } { 2 } } \end{array}$ and that Assumptions A1–A4 hold. Then the empirical measure process $( \mu ^ { N } ) _ { N \geq 1 }$ converges in probability, in ${ \mathcal { D } } ( \mathbb { R } _ { + } , { \mathcal { P } } _ { \gamma } ( \mathbb { R } ^ { d } ) )$ , towards a deterministic trajectory $\bar { \mu } \in \mathcal { C } ( \mathbb { R } _ { + } , \mathcal { P } _ { 1 } ( \mathbb { R } ^ { d } ) )$ .

Moreover, µ¯ is the unique solution in $\mathcal { C } ( \mathbb { R } _ { + } , \mathcal { P } _ { 1 } ( \mathbb { R } ^ { d } ) )$ of the following weak formulation of the nonlinear measure-valued equation given for every $f \in \mathcal { C } _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ and every $t \geq 0$ by

$$
\left. f , \bar { \mu } _ { t } \right. = \left. f , \mu _ { 0 } \right. + \alpha \int _ { 0 } ^ { t } \int _ { \mathcal { X } \times \mathcal { Y } } \Big ( y - \big \langle \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { s } \big \rangle \Big ) \big \langle \nabla f \cdot \nabla \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { s } \big \rangle \mathrm { d } \pi ( x , y ) \mathrm { d } s .\tag{5}
$$

Furthermore, $\bar { \mu } \in \mathcal { C } ( \mathbb { R } _ { + } , \mathcal { H } ^ { - L , \gamma } ( \mathbb { R } ^ { d } ) )$ , and equation (5) extends to every test function f which belongs to $\mathcal { H } ^ { L , \gamma } ( \mathbb { R } ^ { d } )$

The parameter $\beta$ is the scaling exponent of the exploration noise introduced in the stochastic gradient dynamics (SGD). The condition $\beta > \frac { 1 } { 2 }$ ensures that this perturbation vanishes at the mean-field scale and therefore does not afect the deterministic limiting dynamics. The limiting equation (5) is a nonlinear transport equation on the space of probability measures whose nonlinearity stems from the dependence of the drift on the current distribution ${ { \bar { \mu } } _ { t } }$ , which is characteristic of McKean-Vlasov type dynamics.

Theorem 10 states that the limiting trajectory is continuous for the $W _ { 1 }$ topology. Moreover, as stated in the work by A.Descours, A.Guillin, M.Michel & B.Nectoux [19, Corollary 1] it is possible to show that $\bar { \mu } \in C \big ( \mathbb { R } _ { + } , \mathcal { H } ^ { - L , \gamma } ( \mathbb { R } ^ { d } ) \big )$ , and that the limiting equation extends to test functions in $\mathcal { H } ^ { L , \gamma } ( \mathbb { R } ^ { d } )$ . The higher-order moment estimates obtained in the same work allow one to strengthen the Wasserstein regularity of the limiting trajectory. More precisely it is in fact continuous for the $W _ { \gamma }$ topology in which the functional law of large numbers is formulated.

Corollary 11 (Continuity of the mean field limit in $W _ { \gamma }$ topology). Under the assumptions of Theorem 10, the deterministic limiting trajectory satisfies $\bar { \mu } \in C \big ( \mathbb { R } _ { + } , \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) \big )$ . Therefore, for every $t \geq 0$ , the following convergence holds in $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$

$$
\mu _ { t } ^ { N } \xrightarrow [ N  + \infty ] { \mathbb { P } } \bar { \mu } _ { t } .
$$

Proof. Let us start by proving the following Wasserstein interpolation estimate: let $1 \leq r < p <$ $q < + \infty$ and let $\theta \in ( 0 , 1 )$ be defined by

$$
\frac { \theta } { r } + \frac { 1 - \theta } { q } = \frac { 1 } { p } .
$$

Then, for every $\mu , \nu \in \mathcal { P } _ { q } ( \mathbb { R } ^ { d } )$

$$
W _ { p } ( \mu , \nu ) \leq 2 ^ { \frac { ( q - 1 ) ( 1 - \theta ) } { q } } \biggl ( \int _ { \mathbb { R } ^ { d } } | x | ^ { q } \mathrm { d } \mu ( x ) + \int _ { \mathbb { R } ^ { d } } | x | ^ { q } \mathrm { d } \nu ( x ) \biggr ) ^ { \frac { 1 - \theta } { q } } W _ { r } ( \mu , \nu ) ^ { \theta } .\tag{6}
$$

To prove this, let $\mu , \nu \in \mathcal { P } _ { q } ( \mathbb { R } ^ { d } )$ and consider their optimal coupling $\pi _ { r }$ for $W _ { r }$ . Denoting by f the function $( x , y ) \mapsto | x - y |$ , we have, thanks to the log-convexity of Lebesgue norms applied to $L ^ { r } ( \pi _ { r } )$ and $L ^ { q } ( \pi _ { r } )$ norms, $\| f \| _ { L ^ { p } } \leq \| f \| _ { L ^ { r } } ^ { \theta } \| f \| _ { L ^ { q } } ^ { 1 - \theta }$ . Inequality (6) follows from $W _ { p } ( \mu , \nu ) \leq \| f \| _ { L ^ { p } }$ $W _ { r } ( \mu , \nu ) ^ { \theta } = \| f \| _ { L ^ { r } } ^ { \theta }$ and $\| f \| _ { L ^ { q } } ^ { q } \leq 2 ^ { q - 1 } \biggl ( \int _ { \mathbb { R } ^ { d } } | x | ^ { q } \mathrm { d } \mu ( x ) + \int _ { \mathbb { R } ^ { d } } | x | ^ { q } \mathrm { d } \nu ( x ) \biggr )$

Then, let $T > 0$ and $q \in ( \gamma , \gamma + 1 )$ . By [19, Corollary 13], it holds

$$
\operatorname* { s u p } _ { N \geq 1 } \mathbb { E } \left[ \operatorname* { s u p } _ { t \in [ 0 , T + 1 ] } \int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } \mu _ { t } ^ { N } ( w ) \right] < + \infty .\tag{7}
$$

Applying Skorokhod representation theorem, see e.g. the books by P.Billingsley [10, Theorem 6.7] or by V.Bogachev [12, Theorem 8.5.4], there exist a probability space and some $\mathcal { D } ( \mathbb { R } _ { + } , \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) ) .$ valued random elements $( M ^ { N } ) _ { N \geq 1 }$ defined on that space such that $M ^ { N } \overset { \mathcal { L } } { = } \mu ^ { N }$ for all $N \geq$ 1 and $M ^ { N } N \to \mathop { + \infty } _ {  } \bar { \mu }$ almost surely in ${ \mathcal { D } } ( \mathbb { R } _ { + } , { \mathcal { P } } _ { \gamma } ( \mathbb { R } ^ { d } ) )$ . Denote $\mathsf { D } _ { T + 1 } \subset [ 0 , T + 1 ]$ the set of discontinuity points of $\bar { \mu } ,$ restricted to $[ 0 , T + 1 ]$ . By a result from the book by P.Billingsley [10, Chapter 3, Lemma 1], this set is at most countable. Since convergence in Skorokhod topology implies convergence at any continuity point of the limit (see e.g. the book by P.Billingsley [10, Equation (12.14)]), it holds almost surely for all $t \in [ 0 , T + 1 ] \backslash \mathsf { D } _ { T + 1 } , W _ { \gamma } ( M _ { t } ^ { N } , \bar { \mu } _ { t } ) \to 0$ . Now, fix $t \in [ 0 , T + 1 ] \backslash \mathtt { D } _ { T + 1 }$ . Then, see e.g. the book by L.Ambrosio, N.Gigli & G.Savaré [3, Lemma 5.1.7], it holds almost surely

$$
\int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } \bar { \mu } _ { t } ( w ) \leq \operatorname* { l i m i n f } _ { N } \int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } M _ { t } ^ { N } ( w ) .
$$

Taking the expectation and applying Fatou’s lemma, we obtain

$$
\begin{array} { r l } { \displaystyle \int _ { \mathbb R ^ { d } } | w | ^ { q } \mathrm { d } \bar { \mu } _ { t } ( w ) \leq \operatorname* { l i m i n f } _ { N } \mathbb { E } \Big [ \displaystyle \int _ { \mathbb R ^ { d } } | w | ^ { q } \mathrm { d } \mu _ { t } ^ { N } ( w ) \Big ] } & { } \\ { \leq \operatorname* { l i m i n f } _ { N } \mathbb { E } \Big [ \displaystyle \operatorname* { s u p } _ { s \in [ 0 , T + 1 ] } \int _ { \mathbb R ^ { d } } | w | ^ { q } \mathrm { d } \mu _ { s } ^ { N } ( w ) \Big ] } & { } \\ { \leq \displaystyle \operatorname* { s u p } _ { N \geq 1 } \mathbb { E } \Big [ \displaystyle \operatorname* { s u p } _ { s \in [ 0 , T + 1 ] } \int _ { \mathbb R ^ { d } } | w | ^ { q } \mathrm { d } \mu _ { s } ^ { N } ( w ) \Big ] . } \end{array}\tag{8}
$$

Hence, recalling (7), we just proved that

$$
\operatorname* { s u p } _ { t \in [ 0 , T + 1 ] \backslash \mathbb { D } _ { T + 1 } } \int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } \bar { \mu } _ { t } ( w ) < + \infty .
$$

Finally, if $t \in \mathsf { D } _ { T + 1 } \cap [ 0 , T ]$ , consider a sequence $( t _ { n } ) \subset [ 0 , T + 1 ] \backslash \mathsf { D } _ { T + 1 }$ with $t _ { n }  t , t _ { n } > t$ . By right continuity, $W _ { \gamma } ( \bar { \mu } _ { t } , \bar { \mu } _ { t _ { n } } ) \to 0$ , and hence, see e.g. in the book by L.Ambrosio, N.Gigli & G.Savaré [3, Lemma 5.1.7], it follows

$$
\int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } \bar { \mu } _ { t } ( w ) \leq \operatorname* { l i m i n f } _ { n } \int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } \bar { \mu } _ { t _ { n } } ( w ) \leq \operatorname* { s u p } _ { s \in [ 0 , T + 1 ] \backslash \mathsf { D } _ { T + 1 } } \int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } \bar { \mu } _ { s } ( w ) .
$$

This proves that

$$
\operatorname* { s u p } _ { t \in [ 0 , T ] } \int _ { \mathbb { R } ^ { d } } | w | ^ { q } \mathrm { d } \bar { \mu } _ { t } ( w ) < + \infty .\tag{9}
$$

Then, by (9), $\bar { \mu } _ { t } \in \mathcal { P } _ { q } ( \mathbb { R } ^ { d } )$ for any $t \in [ 0 , T ]$ . Applying (6) with $r = 1 , p = \gamma _ { \mathrm { { } } }$ , and the above choice $q \in ( \gamma , \gamma + 1 )$ , we obtain, for every $s , t \in [ 0 , T ]$ 2

$$
W _ { \gamma } ( \bar { \mu } _ { s } , \bar { \mu } _ { t } ) \leq C _ { T } W _ { 1 } ( \bar { \mu } _ { s } , \bar { \mu } _ { t } ) ^ { \theta } ,
$$

where the constant $C _ { T }$ is provided by (9). The $W _ { 1 } \cdot$ -continuity of $\bar { \mu }$ therefore yields $\bar { \mu } \in C (  { \mathbb { R } } _ { + } , \mathcal { P } _ { \gamma } (  { \mathbb { R } } ^ { d } ) )$ , since $T$ is arbitrary. Finally, since evaluation at time t is continuous at every trajectory that is continuous at t for the Skorokhod $J _ { 1 }$ topology, the following convergence in $\mathcal { D } \big ( \mathbb { R } _ { + } , \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) \big )$

$$
\mu ^ { N } \xrightarrow [ N  + \infty ] { \mathbb { P } } \bar { \mu }
$$

implies, by the continuous mapping theorem, that for every $t \geq 0$

$$
\begin{array} { r } { \mu _ { t } ^ { N } \xrightarrow [ N  + \infty ] { \mathbb { P } } \bar { \mu } _ { t } , \mathrm { ~ i n ~ } \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) . } \end{array}
$$

This improvement concerns the mean-field law of large numbers. The fluctuation central limit theorem recalled below remains formulated in the weaker space $\mathcal { H } = \mathcal { H } ^ { - J _ { 0 } + 1 , j _ { 0 } } ( \mathbb { R } ^ { d } )$

The previous theorem establishes the deterministic mean-field limit of the empirical measure process. We now turn to its fluctuations around this limit. Their asymptotic behavior is described by a functional central limit theorem, which requires two additional assumptions.

The following assumptions are only required for the fluctuation analysis. The first one is mainly technical and simplifies several uniform estimates appearing in the proof of the functional central limit theorem. The second one reflects the asymptotic stabilization of the mini-batch size and enters explicitly into the covariance structure of the limiting Gaussian process.

A5 (Compactly supported initialization). The initial distribution $\mu _ { 0 } \in \mathcal P ( \mathbb { R } ^ { d } )$ is compactly supported.

A6 (Asymptotically constant batch size). The batch size satisfies

$$
| B _ { k } | \longrightarrow | B _ { \infty } | \mathrm { a l m o s t ~ s u r e l y ~ a s } k  \infty .
$$

Assumption A5 states that the parameters of the neurons are initially supported in a common compact subset of the parameter space. Although mainly technical, this assumption considerably simplifies the tightness and regularity estimates required for the fluctuation analysis. Assumption A6 states that the mini-batch size stabilizes along training. While this assumption has no influence on the deterministic mean-field limit, its limiting value determines the covariance of the Gaussian noise driving the fluctuation process.

The law of large numbers identifies the deterministic trajectory $\bar { \mu }$ around which the empirical measure concentrates. To investigate the random deviations from this limit, we introduce the rescaled fluctuation process

$$
\eta _ { t } ^ { N } = \sqrt { N } ( \mu _ { t } ^ { N } - \bar { \mu } _ { t } ) , \quad t \ge 0 .\tag{10}
$$

The normalization by $\sqrt { N }$ is the natural one predicted by the central limit theorem. Formally, the fluctuation process is expected to converge towards the solution of a linear stochastic evolution equation obtained by linearizing the mean-field dynamics around the deterministic trajectory $\bar { \mu } .$ Its weak formulation reads as follows. Almost surely for every $f \in \mathcal { H } ^ { J _ { 0 } , j _ { 0 } } ( \mathbb { R } ^ { d } )$ , and every $t \in \mathbb { R } _ { + }$

$$
\begin{array} { r } { \langle f , \eta _ { t } \rangle - \langle f , \eta _ { 0 } \rangle = \alpha \displaystyle \int _ { 0 } ^ { t } \int _ { \mathcal { X } \times \mathcal { Y } } ( y - \langle \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { s } \rangle ) \langle \nabla f \cdot \nabla \sigma ^ { * } ( \cdot , x ) , \eta _ { s } \rangle \mathrm { d } \pi ( x , y ) \mathrm { d } s } \\ { - \alpha \displaystyle \int _ { 0 } ^ { t } \int _ { \mathcal { X } \times \mathcal { Y } } \langle \sigma ^ { * } ( \cdot , x ) , \eta _ { s } \rangle \langle \nabla f \cdot \nabla \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { s } \rangle \mathrm { d } \pi ( x , y ) \mathrm { d } s + \langle f , \mathcal { G } _ { t } \rangle . } \end{array}\tag{11}
$$

The stochastic forcing term $\mathcal { G }$ appearing in (11) plays the role of the driving Gaussian noise. Following the terminology introduced by B.Fernandez & S.Méléard [22], we now define the Gaussian process driving the limiting fluctuation equation. Although referred to as a G-process, it is simply an infinite-dimensional centered Gaussian process whose covariance is entirely determined by the deterministic mean-field trajectory and by the asymptotic mini-batch size.

Definition 12 (G-process). Let $J _ { 0 } , j _ { 0 } \ge 0$ . A process $\mathcal { G } \in \mathcal { C } ( \mathbb { R } _ { + } , \mathcal { H } ^ { - J _ { 0 } , j _ { 0 } } ( \mathbb { R } ^ { d } ) )$ is called $a \mathrm { ~ G ~ }$ process if the following properties hold.

1. For every integer $k \geq 1$ and every collection of test functions $f _ { 1 } , \dots , f _ { k } \in { \mathcal { H } } ^ { J _ { 0 } , j _ { 0 } } ( \mathbb { R } ^ { d } )$ , the projected process

$$
\left( \langle f _ { 1 } , \mathcal { G } _ { t } \rangle , \ldots , \langle f _ { k } , \mathcal { G } _ { t } \rangle \right) _ { t \geq 0 }
$$

is an $\mathbb { R } ^ { k }$ -valued centered Gaussian process with continuous trajectories and independent increments.

2. The covariance structure is prescribed $b y ,$ for every $0 \leq s \leq t$ , by

$$
\operatorname { C o v } \big ( \langle f _ { i } , \mathcal { G } _ { t } \rangle , \langle f _ { j } , \mathcal { G } _ { s } \rangle \big ) = \alpha ^ { 2 } \mathbb { E } \left[ \frac { 1 } { | B _ { \infty } | } \right] \int _ { 0 } ^ { s } \operatorname { C o v } _ { \pi } ( \mathrm { Q } _ { v } [ f _ { i } ] ( \boldsymbol { x } , \boldsymbol { y } ) , \mathrm { Q } _ { v } [ f _ { j } ] ( \boldsymbol { x } , \boldsymbol { y } ) ) \mathrm { d } \boldsymbol { v } ,\tag{12}
$$

where $\mathrm { Q } _ { v } [ f ] ( x , y ) : = ( y - \langle \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { v } \rangle ) \langle \nabla f \cdot \nabla \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { v } \rangle$ for $f \in \mathcal { H } ^ { J _ { 0 } , j _ { 0 } } ( \mathbb { R } ^ { d } )$ and $\bar { \mu }$ is given by Theorem $1 0 .$

Unlike a standard cylindrical Wiener process, the covariance of the G-process in (11) depends both on the deterministic mean-field trajectory through the operator $Q _ { t }$ and on the limiting minibatch size through the factor $\mathbb { E } [ | B _ { \infty } | ^ { - 1 } ]$ . Also, note that the stochastic evolution equation (11) is understood in the weak sense. We therefore recall the corresponding notion of weak solution, which provides the appropriate framework for the functional central limit theorem.

Definition 13 (Weak solution). Let ν be an $\mathcal { H }$ -valued random variable. A process $\eta \in \mathcal { C } \left( \mathbb { R } _ { + } , \mathcal { H } \right)$ defined on some probability space, is called a weak solution of (11) with initial distribution ν if there exists a G-process $\mathcal { G } \in \mathcal { C } ( \mathbb { R } _ { + } , \mathcal { H }$ −J<sub>0</sub>,j<sub>0</sub> $( \mathbb { R } ^ { d } ) )$ defined on the same probability space such that (11) holds almost surely and η<sub>0</sub> has the same distribution as $\nu .$

Weak uniqueness is said to hold if any two weak solutions of (11), possibly defined on diferent probability spaces and having the same initial distribution, have the same law.

We are now in position to state the functional central limit theorem. It shows that the fluctuations of the empirical measure around the deterministic mean-field limit converge towards the unique solution of a linear stochastic evolution equation driven by the G-process introduced above.

Theorem 14 (Functional central limit theorem). Assume that $\textstyle { \beta > { \frac { 3 } { 4 } } }$ , Assumptions A1–A4 and A5–A6 hold. Then the fluctuation process $( \eta ^ { N } ) _ { N \geq 1 }$ , defined in (10), converges in distribution in $\mathcal { D } \big ( \mathbb { R } _ { + } , \mathcal { H } \big )$ towards a continuous process η¯ which belongs to $\mathcal { C } \left( \mathbb { R } _ { + } , \mathcal { H } \right)$

Furthermore, the limiting process $\bar { \eta }$ has the same distribution as the unique weak solution $\eta ^ { \star }$ of (11) with initial distribution $\nu _ { 0 }$ in the sense of Definition 13. More precisely, ν is the unique ${ \mathcal { H } } .$ -valued Gaussian random variable such that, for every $k \geq 1$ and every $f _ { 1 } , \ldots , f _ { k } \in$ $\mathcal { H } ^ { J _ { 0 } - 1 , j _ { 0 } } ( \mathbb { R } ^ { d } )$

$$
\left( \langle f _ { 1 } , { \bar { \eta } } _ { 0 } \rangle , \dots , \langle f _ { k } , { \bar { \eta } } _ { 0 } \rangle \right) ^ { T } \sim { \mathcal { N } } { \left( 0 , \Gamma ( f _ { 1 } , \dots , f _ { k } ) \right) } ,
$$

where $\Gamma ( f _ { 1 } , \dots , f _ { k } )$ denotes the covariance matrix of $\big ( f _ { 1 } ( W _ { 0 } ^ { 1 } ) , \dots , f _ { k } ( W _ { 0 } ^ { 1 } ) \big ) ^ { T }$

The functional central limit theorem shows that, after the $\sqrt { N }$ rescaling, the random fluctuations of the empirical measure around the deterministic mean-field trajectory become asymptotically Gaussian. In contrast with the law of large numbers given by Theorem 10, the limiting dynamics is now stochastic, since it is governed by a linear evolution equation driven by the G-process introduced above. As for Theorem 10, the stronger condition $\beta > { \frac { 3 } { 4 } }$ ensures that the exploration noise appearing in the stochastic gradient algorithm remains negligible at the fluctuation scale.

The functional central limit theorem identifies the limiting fluctuation process as the unique weak solution of the stochastic evolution equation (11). The proof relies on a compactness argument together with Prokhorov theorem, which establishes convergence in distribution without providing an explicit description of the law of the limiting process. Nevertheless, as discussed for instance by J.Sirignano, R.Sowers & K.Spiliopoulos [58, Chapter 4, Section 20], the limiting dynamics is expected to remain Gaussian since it solves a linear stochastic equation driven by a Gaussian process. The remaining question was therefore to characterize this Gaussian law explicitly. The approach used in the work by A.Descours, A.Guillin, G.Lacour, M.Michel, B.Nectoux & P.Stos [20, Theorem 3] addresses this question by extending the class of admissible test functions. More precisely, instead of considering arbitrary test functions, we introduce a family of solutions to a nonlocal backward transport equation whose role is to compensate the drift term appearing in (11). More precisely, the correction is obtained by transporting the test functions backward along the linearized mean-field dynamics. Such an approach can be found e.g. in the works by S.Bourguin & K.Spiliopoulos [13] or by Z.Wang, X.Zhao & R.Zhu [67, Section 3.4]. This transformation allows the stochastic evolution to be expressed solely in terms of its Gaussian forcing and ultimately yields an explicit characterization of the law of the limiting fluctuation process. This leads to the following nonlocal backward transport equation.

More precisely, given a terminal time $t \geq 0$ and a final condition $\varphi \in \mathcal { C } _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ , we consider the following nonlocal backward transport equation.

$$
\begin{array} { r l } & { \{ \partial _ { s } f ( s , w ) + \alpha \mathbb { E } _ { \pi } [ ( y - \langle \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { s } \rangle ) \nabla f ( s , w ) \cdot \nabla \sigma ^ { * } ( w , x ) -  \nabla f ( s , \cdot ) \cdot \nabla \sigma ^ { * } ( \cdot , x ) , \bar { \mu } _ { s }  \sigma ^ { * } ( w , x ) ] = 0 , } \\ & { \{ f ( t , w ) = \varphi ( w ) .  } \end{array}\tag{13}
$$

We can now state the main result of A.Descours, A.Guillin, G.Lacour, M.Michel, B.Nectoux & P.Stos [20, Theorem 3], which provides an explicit representation of the covariance of the limiting fluctuation process and therefore completely characterizes its law.

Theorem 15. Assume that Assumptions A1–A6 hold and that $\textstyle { \beta > { \frac { 3 } { 4 } } }$ . Then, for every $n \geq 1$ and every $\varphi _ { 1 } , \ldots , \varphi _ { n } \in { \mathcal { C } } _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ , the finite-dimensional process

$$
\left( \langle \varphi _ { 1 } , { \bar { \eta } } \rangle , \dots , \langle \varphi _ { n } , { \bar { \eta } } \rangle \right) \in { \mathcal { C } } ( \mathbb { R } _ { + } , \mathbb { R } ^ { n } )
$$

is a centered Gaussian process. Its covariance is explicitly given, for every $\varphi , \psi \in \mathcal { C } _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ and every $0 \leq s \leq t$ , by

$$
\begin{array} { l } { \displaystyle \operatorname { C o v } \left( \langle \varphi , { \bar { \eta } } _ { s } \rangle , \langle \psi , { \bar { \eta } } _ { t } \rangle \right) = \operatorname { C o v } \left( f ^ { \varphi } ( 0 , W _ { 0 } ^ { 1 } ) , f ^ { \psi } ( 0 , W _ { 0 } ^ { 1 } ) \right) } \\ { \displaystyle \qquad + \alpha ^ { 2 } \mathbb { E } \left[ \frac { 1 } { | B _ { \infty } | } \right] \int _ { 0 } ^ { s } \operatorname { C o v } _ { \pi } \left( Q _ { r } [ f ^ { \varphi } ( r ) ] , Q _ { r } [ f ^ { \psi } ( r ) ] \right) d r , } \end{array}\tag{14}
$$

where $f ^ { \varphi }$ and $f ^ { \psi }$ denote the solutions of (13) on [0, s] and [0, t] with terminal conditions $f ^ { \varphi } ( s , \cdot ) =$ $\varphi$ and $f ^ { \psi } ( t , \cdot ) = \psi$ , respectively.

An immediate consequence of Theorem 15 is the following explicit variance representation.

$$
\mathrm { V a r } \left( \langle \varphi , \bar { \eta } _ { t } \rangle \right) = \mathrm { V a r } \left( f ^ { \varphi } ( 0 , W _ { 0 } ^ { 1 } ) \right) + \alpha ^ { 2 } \mathbb { E } \left[ \frac { 1 } { | B _ { \infty } | } \right] \int _ { 0 } ^ { t } \mathrm { V a r } \left( Q _ { s } [ f ^ { \varphi } ( s ) ] \right) d s .\tag{15}
$$

Choosing $\varphi = \sigma ^ { * } ( \cdot , x )$ yields an explicit variance formula for the prediction of the neural network at the input $x \in \mathcal { X }$ . This identity provides a direct uncertainty quantification formula in the mean-field regime.

Remark 16. Since $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } ) \subset C _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ and $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ is dense in $\mathcal { H } ^ { \prime } \simeq \mathcal { H } ^ { J _ { 0 } - 1 , j _ { 0 } } ( \mathbb { R } ^ { d } )$ , Theorem 15 also implies that η¯<sub>t</sub> is an H-valued Gaussian random variable. More precisely, for every $\varphi \in \mathcal { H } ^ { \prime }$ choose $\varphi _ { n } \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ such that $\varphi _ { n } \to \varphi$ in ${ \mathcal { H } } ^ { I }$ . Then

$$
\langle \varphi _ { n } , \bar { \eta } _ { t } \rangle  \langle \varphi , \bar { \eta } _ { t } \rangle ,
$$

almost surely, while every variable $\left. \varphi _ { n } , \bar { \eta } _ { t } \right.$ is Gaussian by Theorem 15. Since a weak limit of Gaussian distributions on R is Gaussian, the conclusion follows. Nevertheless, this does not let us extend the explicit covariance formula (14) in such a setting, for which a deeper analysis of qualitative properties to the solution of the backward transport equation (13) is required. ⋄

## 4 Proofs of the results

We begin with the proof of Theorem 1, recalling that its inherent functional framework is given for the fluctuation space by $\mathcal { H } = \mathcal { H } ^ { - J _ { 0 } + 1 , j _ { 0 } } ( \mathbb { R } ^ { d } )$ , in which Theorem 14 provides the convergence of $\eta ^ { N }$ towards the continuous limiting process η¯. First, we apply a joint convergence result provided by Lemma 18 in Appendix A.

For completeness, and for readers interested in the relation between the two functional frameworks involved, we also establish an explicit quantitative comparison between the Wasserstein and weighted Sobolev topologies. Although this estimate is not logically required by the subsequent representation argument, it provides an independent quantitative way to the required fixed-time Sobolev convergence and makes the passage between the two topologies as self contained as possible. Nevertheless, readers that are primarily interested in the Delta method argument may skip the preliminary step.

Then, we apply the Skorokhod representation theorem to the jointly convergent pair $( \mu ^ { N } , \eta ^ { N } )$ and work, on a common probability space, with representatives converging almost surely in the corresponding product Skorokhod space. Finally, we use the identity $\eta _ { t } ^ { N } = \sqrt { N } ( \mu _ { t } ^ { N } - \bar { \mu } _ { t } )$ together with the Fréchet diferentiability of the observation functional. A first-order expansion around the mean field limit and a control of the remainder then give the desired convergence in distribution.

Proof of Theorem 1. As above mentioned, we prove the result for a fixed time $t \geq 0$ . By Theorem 10, the empirical measure $\mu ^ { N }$ converges in probability to the deterministic path $\bar { \mu }$ in $\begin{array} { r } { B _ { 1 } : = \mathcal { D } ( \mathbb { R } _ { + } , \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) ) } \end{array}$ , implying that $\mu ^ { N } \xrightarrow [ N  \infty ] { \mathcal { L } } \bar { \mu }$ in the same functional space. Moreover, by Theorem 14, the fluctuation process $\eta ^ { N } : = \sqrt { N } ( \mu ^ { N } { - } \bar { \mu } )$ converges in law to η¯ in $B _ { 2 } : = \mathcal { D } ( \mathbb { R } _ { + } , \mathcal { H } )$ Then, we observe that $\boldsymbol { B } _ { 1 } \times \boldsymbol { B } _ { 2 }$ equipped with the metric

$$
d _ { { \mathcal B } _ { 1 } \times { \mathcal B } _ { 2 } } \bigg ( ( \mu _ { 1 } , \eta _ { 1 } ) , ( \mu _ { 2 } , \eta _ { 2 } ) \bigg ) : = \sqrt { d _ { { \mathcal B } _ { 1 } } ( \mu _ { 1 } , \mu _ { 2 } ) ^ { 2 } + d _ { { \mathcal B } _ { 2 } } ( \eta _ { 1 } , \eta _ { 2 } ) ^ { 2 } }
$$

is a Polish space, see $\mathrm { e . g . }$ results in the book by C.Villani [65, Theorem 6.18] and in the book by J.Jacod & A.Shiryaev [31, Chapter VI, Theorem 1.14-a], where $d { \boldsymbol { B } } _ { 1 }$ and $d _ { B _ { 2 } }$ stand respectively for the usual $\left( J _ { 1 } \right)$ Skorokhod topologies on $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$ endowed with Wasserstein distance and on $\mathcal { H }$ for the metric introduced in Section 3. Applying now Lemma 18 in Appendix A, we get that

$$
\begin{array} { r } { ( \mu ^ { N } , \eta ^ { N } ) \xrightarrow [ N  \infty ] { \mathcal { L } } ( \bar { \mu } , \bar { \eta } ) \quad \mathrm { i n } ~ B _ { 1 } \times \mathcal { B } _ { 2 } . } \end{array}
$$

As above mentioned, we begin by establishing a quantitative estimate showing how random processes are embedded from Wasserstein to negative-order weighted Sobolev topologies.

Preliminary step - A quantitative comparison between Wasserstein and weighted Sobolev topolo-$g i e s .$ . Let $t \geq 0$ . The purpose of this step is to prove (19), and consequently that $\mu _ { t } ^ { N } \xrightarrow [ N  + \infty ] { \mathbb { P } } \bar { \mu } t$ in $\mathcal { H }$ . We give a complete proof and a quantitative rate of convergence with respect to the $W _ { \gamma }$ distance. Note however that one can invoke simply that $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$ is continuously embedded in $\mathcal { H }$ so one can skip this step at the first reading and go directly to the next step (see Remark 17 for further details). We introduce the intermediate space $\mathcal { H } : = \mathcal { H } ^ { - L , \gamma ^ { * } } ( \mathbb { R } ^ { d } )$ . Since

$$
\mathcal { H } ^ { J _ { 0 } - 1 , j _ { 0 } } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } ^ { L , \gamma } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } ^ { L , \gamma ^ { * } } ( \mathbb { R } ^ { d } ) ,
$$

duality yields

$$
\mathcal { H } \hookrightarrow \mathcal { H } ^ { - L , \gamma } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } .
$$

Let $\mu , \nu \in \mathcal P _ { 2 \gamma ^ { * } } ( \mathbb R ^ { d } )$ and let $\pi \in \Pi ( \mu , \nu )$ denote the optimal coupling between $\mu$ and $\nu$ fo $W _ { 2 }$ For every $\varphi \in \mathcal { H } ^ { L , \gamma ^ { * } } ( \mathbb { R } ^ { d } )$ , we can write

$$
\left| \varphi ( w ) - \varphi ( w ^ { \prime } ) \right| \leq \operatorname* { s u p } _ { s \in [ 0 , 1 ] } \big | \nabla \varphi ( w + s ( w ^ { \prime } - w ) ) \big | | w - w ^ { \prime } \big | \leq C \| \varphi \| _ { \mathcal H ^ { L , \gamma ^ { * } } } | w - w ^ { \prime } | \big ( 1 + | w | ^ { \gamma ^ { * } } + | w ^ { \prime } | ^ { \gamma ^ { * } } \big ) ,
$$

for some $C > 0$ . Integrating with respect to π, we obtain

$$
\big | \langle \varphi , \mu - \nu \rangle \big | \leq C \| \varphi \| _ { \mathcal { H } ^ { L , \gamma ^ { * } } } \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } \big | w - w ^ { \prime } \big | \big ( 1 + | w | ^ { \gamma ^ { * } } + | w ^ { \prime } | ^ { \gamma ^ { * } } \big ) \mathrm { d } \pi ( w , w ^ { \prime } ) ,
$$

and Hölder inequality let us conclude, for a diferent constant that we still denote by $C ,$

$$
\big | \langle \varphi , \mu - \nu \rangle \big | \leq C \| \varphi \| _ { \mathcal H ^ { L , \gamma ^ { * } } } \bigg ( 1 + \int _ { \mathbb R ^ { d } \times \mathbb R ^ { d } } | w | ^ { 2 \gamma ^ { * } } + | w ^ { \prime } | ^ { 2 \gamma ^ { * } } \mathrm { d } \pi ( w , w ^ { \prime } ) \bigg ) ^ { \frac 1 2 } W _ { 2 } ( \mu , \nu ) .
$$

Since $\gamma \geq 2$ $W _ { 2 } ( \mu , \nu ) \leq W _ { \gamma } ( \mu , \nu )$ (see e.g. the book by C.Villani [65, Remark 6.6]), so that one gets

$$
\big | \langle \varphi , \mu - \nu \rangle \big | \leq C \bigg ( 1 + \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } + | w ^ { \prime } | ^ { 2 \gamma ^ { * } } \mathrm { d } \pi ( w , w ^ { \prime } ) \bigg ) ^ { \frac { 1 } { 2 } } \| \varphi \| _ { \mathcal { H } ^ { L , \gamma ^ { * } } } W _ { \gamma } ( \mu , \nu ) ,
$$

and therefore, taking the supremum on both sides over the closed unit ball in $\mathcal { H } ^ { L , \gamma ^ { * } } ( \mathbb { R } ^ { d } )$ yields

$$
\| \mu - \nu \| _ { \mathcal { H } } \leq C \biggl ( 1 + \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } + | w ^ { \prime } | ^ { 2 \gamma ^ { * } } \mathrm { d } \pi ( w , w ^ { \prime } ) \biggr ) ^ { \frac { 1 } { 2 } } W _ { \gamma } ( \mu , \nu ) .\tag{16}
$$

Our goal is now to apply this inequality to $\mu _ { t } ^ { N }$ and $\bar { \mu } _ { t }$ . We first point out that $\mu _ { t } ^ { N }$ does satisfy the above required higher-order moment control, thanks to the result by A.Descours, A.Guillin, M.Michel & B.Nectoux [19, Lemma 8]. Namely, for every $T > 0$ , there exists $C _ { T } > 0$ such that

$$
\operatorname* { s u p } _ { N \geq 1 } \operatorname* { s u p } _ { 1 \leq i \leq N } \operatorname* { s u p } _ { 0 \leq k \leq \lfloor N T \rfloor } \mathbb { E } \big [ | W _ { k } ^ { i , N } | ^ { 8 \gamma ^ { * } } \big ] \leq C _ { T } .
$$

Hence, for every fixed $t \in [ 0 , T ]$ , this immediately implies

$$
\operatorname* { s u p } _ { N \geq 1 } \mathbb { E } \left[ \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \mu _ { t } ^ { N } ( w ) \right] < + \infty .
$$

Therefore, Markov inequality gives, for every $R > 0$

$$
\operatorname* { s u p } _ { N \geq 1 } \mathbb { P } \bigg ( \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \mu _ { t } ^ { N } ( w ) > R \bigg ) \leq \frac { 1 } { R } \operatorname* { s u p } _ { N \geq 1 } \mathbb { E } \bigg [ \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \mu _ { t } ^ { N } ( w ) \bigg ] .\tag{17}
$$

We now prove that $\int _ {  { \mathbb { R } } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \bar { \mu } _ { t } ( w ) < + \infty$ . Indeed, by Corollary 11, $\mu _ { t } ^ { N } \to \bar { \mu } _ { t }$ in probability in $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$ , and therefore also in distribution.

Since the function $w \mapsto | w | ^ { 2 \gamma ^ { * } }$ is nonnegative and lower semicontinuous, we have, see $\mathrm { e . g . }$ in the book by L.Ambrosio, N.Gigli & G.Savaré [3, Lemma 5.1.7], that the extended valued moment

functional from $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$ into $\mathbb { R } _ { + }$ given by $\mu \mapsto \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \mu ( w )$ is lower semicontinuous for the narrow topology. Combining this lower semicontinuity property together with the same argument as the one used to prove (8) that

$$
\int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \bar { \mu } _ { t } ( w ) \leq \operatorname* { l i m i n f } _ { N  + \infty } \mathbb { E } \bigg [ \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \mu _ { t } ^ { N } ( w ) \bigg ] \leq \operatorname* { s u p } _ { N \geq 1 } \mathbb { E } \bigg [ \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \mu _ { t } ^ { N } ( w ) \bigg ] < + \infty .\tag{18}
$$

Together with the preceding uniform expectation estimate, this shows that $\mu _ { t } ^ { N }$ and $\bar { \mu } _ { t }$ define elements of $\mathcal { H }$ , almost surely. Now, for $R > 0$ , set

$$
A _ { N , R } : = \bigg \{ \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \mu _ { t } ^ { N } ( w ) \leq R \bigg \} .
$$

For every $\varepsilon > 0 .$ , estimate (17) allows us to choose $R _ { \varepsilon } > 0$ such that

$$
\operatorname* { s u p } _ { N \geq 1 } \mathbb { P } \big ( A _ { N , R _ { \varepsilon } } ^ { c } \big ) \leq \varepsilon .
$$

On the event $A _ { N , R _ { \varepsilon } }$ , applying (16) with $\mu = \mu _ { t } ^ { N }$ and $\nu = \bar { \mu } _ { t }$ gives the quantitative estimate

$$
\| \mu _ { t } ^ { N } - \bar { \mu } _ { t } \| _ { \mathcal { H } } \leq C \bigg ( 1 + \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \bar { \mu } _ { t } ( w ) + R _ { \varepsilon } \bigg ) ^ { \frac { 1 } { 2 } } W _ { \gamma } ( \mu _ { t } ^ { N } , \bar { \mu } _ { t } ) .\tag{19}
$$

Set

$$
C _ { \varepsilon , t } : = C \biggl ( 1 + \int _ { \mathbb { R } ^ { d } } | w | ^ { 2 \gamma ^ { * } } \mathrm { d } \bar { \mu } _ { t } ( w ) + R _ { \varepsilon } \biggr ) ^ { \frac { 1 } { 2 } } .
$$

Then, for every $\delta > 0$

$$
\begin{array} { r l } & { \mathbb { P } \big ( \| \mu _ { t } ^ { N } - \bar { \mu } _ { t } \| _ { \mathcal { H } } > \delta \big ) \leq \mathbb { P } \big ( A _ { N , R _ { \varepsilon } } ^ { c } \big ) + \mathbb { P } \big ( C _ { \varepsilon , t } W _ { \gamma } ( \mu _ { t } ^ { N } , \bar { \mu } _ { t } ) > \delta \big ) } \\ & { \qquad \leq \varepsilon + \mathbb { P } \bigg ( W _ { \gamma } ( \mu _ { t } ^ { N } , \bar { \mu } _ { t } ) > \frac { \delta } { C _ { \varepsilon , t } } \bigg ) . } \end{array}
$$

Since $\mu _ { t } ^ { N } \xrightarrow [ N  + \infty ] { \mathbb { P } } \bar { \mu } t$ in $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$ by Corollary 11, we obtain

$$
\operatorname* { l i m } _ { N \to + \infty } \mathbb { P } \big ( \| \mu _ { t } ^ { N } - \bar { \mu } _ { t } \| _ { \mathcal { H } } > \delta \big ) \leq \varepsilon .
$$

Letting $\varepsilon \ \to \ 0$ , we conclude that $\mu _ { t } ^ { N } \xrightarrow [ N  + \infty ] { \mathbb { P } } \bar { \mu } t$ in $\mathcal { H }$ and hence in $\mathcal { H }$ by the continuous embedding $\mathcal { H } \hookrightarrow \mathcal { H }$

First step - Representations in product space. We now use the Skorokhod representation theorem, see for instance the books by P.Billingsley [10, Theorem 6.7] or by V.Bogachev [12, Theorem 8.5.4]. It allows us to construct, on a common probability space, random variables

$$
( \mathsf { M } ^ { N } , \mathsf { E } ^ { N } ) \overset { \mathcal { L } } { = } ( \mu ^ { N } , \eta ^ { N } )
$$

and a limiting random variable

$$
( \bar { \mu } , \bar { \mathsf { E } } ) \overset { \mathcal { L } } { = } ( \bar { \mu } , \bar { \eta } )
$$

such that

$$
( \mathsf { M } ^ { N } , \mathsf E ^ { N } ) \to ( \bar { \mu } , \bar { \mathsf { E } } ) \quad \mathrm { a . s . ~ i n ~ } \bar { B } _ { 1 } \times \bar { B } _ { 2 } .\tag{20}
$$

Now for every fixed $N \geq 1$ and $t \geq 0$ , consider the Borel measurable mapping $G _ { N , t } ( m , e ) : = $ $e _ { t } - \sqrt { N } \big ( m _ { t } - \bar { \mu } _ { t } \big )$ , with values in $\mathcal { H }$ . Also since $( \mathsf { M } ^ { N } , \mathsf { E } ^ { N } ) \overset { \mathcal { L } } { = } ( \mu ^ { N } , \eta ^ { N } )$ , and $G _ { N , t }$ is measurable, and since equality in law is preserved under measurable mapping, then

$$
G _ { N , t } ( \mathsf { M } ^ { N } , \mathsf { E } ^ { N } ) \overset { \mathcal { L } } { = } G _ { N , t } ( \mu ^ { N } , \eta ^ { N } ) .
$$

Now by the definition of the fluctuation process,

$$
G _ { N , t } ( \mu ^ { N } , \eta ^ { N } ) = \eta _ { t } ^ { N } - \sqrt { N } \big ( \mu _ { t } ^ { N } - \bar { \mu } _ { t } \big ) = 0 ,
$$

almost surely in ${ \mathcal { H } } .$ . Therefore,

$$
\mathcal { L } \big ( G _ { N , t } ( \mathsf { M } ^ { N } , \mathsf { E } ^ { N } ) \big ) = \delta _ { 0 } ,
$$

and hence,

$$
\mathsf E _ { t } ^ { N } = \sqrt { N } \big ( \mathsf M _ { t } ^ { N } - \bar { \mu } _ { t } \big ) ,\tag{21}
$$

almost surely in $\mathcal { H }$ . Moreover, since the limiting path E<sup>¯</sup> is continuous, the almost sure convergence in (20) implies

$$
\mathsf E _ { t } ^ { N } \to \bar { \mathsf E } _ { t }
$$

almost surely in ${ \mathcal { H } } .$ . In particular, (21) already gives that almost surely in $\mathcal { H }$

$$
\mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } = N ^ { - 1 / 2 } \mathsf { E } _ { t } ^ { N } \to 0 .
$$

Second step - Functional Delta method argument. By assumption, the observation functional $\Phi : \mathcal { H }  \mathbb { R } ^ { k }$ is Fréchet diferentiable at ${ \bar { \mu } } _ { t }$ , so that there exists a remainder map $r _ { t } : \mathcal { H } \to \mathbb { R } ^ { k }$ such that

$$
\Phi ( \bar { \mu } _ { t } + h ) = \Phi ( \bar { \mu } _ { t } ) + D \Phi ( \bar { \mu } _ { t } ) ( h ) + r _ { t } ( h ) ,
$$

with

$$
\frac { | r _ { t } ( h ) | _ { \mathbb { R } ^ { k } } } { \| h \| _ { \mathcal { H } } } \xrightarrow [ h  0 ] 0
$$

in $\mathcal { H }$ . Using this expansion with $h = \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } = N ^ { - \frac { 1 } { 2 } } \mathsf { E } _ { t } ^ { N }$ , we obtain

$$
\sqrt { N } \big ( \Phi ( \mathsf { M } _ { t } ^ { N } ) - \Phi ( \bar { \mu } _ { t } ) \big ) = D \Phi ( \bar { \mu } _ { t } ) ( \mathsf { E } _ { t } ^ { N } ) + \sqrt { N } r _ { t } ( \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } ) .
$$

We now show that the remainder converges to zero almost surely. With the convention that the following quotient is zero whenever $\mathsf { M } _ { t } ^ { N } = \bar { \mu } _ { t }$ , we have

$$
\big | \sqrt { N } r _ { t } ( \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } ) \big | _ { \mathbb { R } ^ { k } } = \| \mathsf E _ { t } ^ { N } \| _ { \mathcal { H } } \frac { | r _ { t } ( \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } ) | _ { \mathbb { R } ^ { k } } } { \| \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } \| _ { \mathcal { H } } } .
$$

Since $\mathsf { M } _ { t } ^ { N } \xrightarrow [ N  \infty ] { a . s . } \bar { \mu } _ { t }$ in $\mathcal { H }$ , the definition of Fréchet diferentiability gives

$$
\frac { \big | r _ { t } \big ( \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } \big ) \big | _ { \mathbb { R } ^ { k } } } { \big \| \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } \big \| _ { \mathcal { H } } } \xrightarrow [ N  + \infty ] { a . s . } 0 .
$$

Also, $\mathsf E _ { t } ^ { N } \to \bar { \mathsf E } _ { t }$ almost surely in $\mathcal { H }$ so that $( \| \mathsf E _ { t } ^ { N } \| _ { \mathcal H } ) _ { N \ge 1 }$ is bounded almost surely. It follows that the remainder satisfies in $\mathbb { R } ^ { k }$

$$
\begin{array} { r } { \sqrt { N } r _ { t } ( \mathsf { M } _ { t } ^ { N } - \bar { \mu } _ { t } ) \xrightarrow [ N  \infty ] { a . s . } 0 . } \end{array}
$$

Moreover, since $D \Phi ( { \bar { \mu } } _ { t } ) : { \mathcal { H } }  \mathbb { R } ^ { k }$ is linear and continuous, we get that $D \Phi ( \bar { \mu } _ { t } ) ( \mathsf E _ { t } ^ { N } ) \ \to$ $D \Phi ( \bar { \mu } _ { t } ) ( \bar { \mathsf { E } } _ { t } )$ almost surely in $\mathbb { R } ^ { k }$ . Thus,

$$
\begin{array} { r } { \sqrt { N } \Big ( \Phi ( \mathsf { M } _ { t } ^ { N } ) - \Phi ( \bar { \mu } _ { t } ) \Big ) \xrightarrow [ N  + \infty ] { a . s . } D \Phi ( \bar { \mu } _ { t } ) ( \bar { \mathsf { E } } _ { t } ) } \end{array}
$$

in $\mathbb { R } ^ { k }$ . Finally, since $( \mathsf { M } ^ { N } , \mathsf { E } ^ { N } ) \overset { \mathcal { L } } { = } ( \mu ^ { N } , \eta ^ { N } )$ and $\bar { \mathsf { E } } \overset { \mathcal { L } } { = } \bar { \eta } ;$ , the almost sure convergence obtained on the Skorokhod representation space implies the desired convergence in distribution, namely

$$
\begin{array} { r } { \sqrt { N } \Big ( \Phi ( \mu _ { t } ^ { N } ) - \Phi ( \bar { \mu } _ { t } ) \Big ) \xrightarrow [ N \to + \infty ] { \mathcal L } D \Phi ( \bar { \mu } _ { t } ) ( \bar { \eta } _ { t } ) } \end{array}
$$

in $\mathbb { R } ^ { k }$ . Since $\bar { \eta } _ { t }$ is Gaussian and $D \Phi ( { { \bar { \mu } } _ { t } } )$ is linear and continuous, the limiting random vector $D \Phi ( { \bar { \mu } } _ { t } ) ( { \bar { \eta } } _ { t } )$ is Gaussian in $\mathbb { R } ^ { k }$ (see item 1. in Definition 12). This achieves the proof. □

We now prove the corollary, which is a straightforward consequence of Theorem 15 and of Theorem 1.

Proof of Corollary 2. By Theorem 1, the limiting random vector is

$$
Z _ { t } ^ { \Phi } = \Big ( D \Phi _ { 1 } ( \bar { \mu } _ { t } ) ( \bar { \eta } _ { t } ) , \dots , D \Phi _ { k } ( \bar { \mu } _ { t } ) ( \bar { \eta } _ { t } ) \Big ) .
$$

Using the representation assumption, we can write it

$$
Z _ { t } ^ { \Phi } = \Big ( \langle \varphi _ { 1 } ^ { t } , \bar { \eta } _ { t } \rangle , \dots , \langle \varphi _ { k } ^ { t } , \bar { \eta } _ { t } \rangle \Big ) .
$$

Hence

$$
\Sigma _ { i , j } ^ { \Phi } ( t ) = \mathrm { C o v } \left( \langle \varphi _ { i } ^ { t } , \bar { \eta } _ { t } \rangle , \langle \varphi _ { j } ^ { t } , \bar { \eta } _ { t } \rangle \right) .
$$

Applying Theorem 15 with $s = t , \varphi = \varphi _ { i } ^ { t }$ and $\psi = \varphi _ { j } ^ { t }$ gives

$$
\Sigma _ { i , j } ^ { \Phi } ( t ) = \operatorname { C o v } \left( f ^ { \varphi _ { i } ^ { t } } ( 0 , W _ { 0 } ^ { 1 } ) , f ^ { \varphi _ { j } ^ { t } } ( 0 , W _ { 0 } ^ { 1 } ) \right) + \alpha ^ { 2 } \mathbb { E } \left[ \frac { 1 } { | B _ { \infty } | } \right] \int _ { 0 } ^ { t } \operatorname { C o v } _ { \boldsymbol \pi } \left( Q _ { r } [ f ^ { \varphi _ { i } ^ { t } } ( r ) ] , Q _ { r } [ f ^ { \varphi _ { j } ^ { t } } ( r ) ] \right) \mathrm { d } r .
$$

This proves the claim.

Remark 17. The argument used above was deliberately formulated in terms of a simple functional structure that is suficient to apply the functional delta method. That being said, a more intrinsic connection between the Wasserstein and weighted Sobolev spaces can be established. Such a continuity property is not automatic for arbitrary combinations of Wasserstein and dual Hölder topologies. For the polynomial growth exponent considered here, however, the canonical embedding $\begin{array} { r } { i : ( \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) , W _ { \gamma } ) \to \big ( \big ( C ^ { J , \gamma } ( \mathbb { R } ^ { d } ) \big ) ^ { * } , \| \cdot \| _ { ( C ^ { J , \gamma } ( \mathbb { R } ^ { d } ) ) ^ { * } } \big ) , \iota ( \mu ) ( f ) : = \int _ { \mathbb { R } ^ { d } } f \mathrm { d } \mu } \end{array}$ , is well defined and continuous for $J \geq 1$ . Combined with the continuous dual Sobolev embedding

$$
\left( C ^ { J , \gamma } ( \mathbb { R } ^ { d } ) \right) ^ { * } \hookrightarrow \mathcal { H }
$$

used in the present paper, which holds under well-chosen parameters $J _ { 0 } , j _ { 0 }$ , this also yields the continuity of the canonical embedding of $\big ( \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) , W _ { \gamma } \big )$ into $\left( \mathcal { H } , \| \cdot \| _ { \mathcal { H } } \right)$ . Let us briefly explain the first continuity statement. Suppose that $\mu ^ { N } \to \bar { \mu }$ in $W _ { \gamma }$ and let $\chi _ { R }$ be a smooth cutof function equals to one on a ball $B _ { R }$ , vanishing outside $B _ { R + 1 }$ , and taking values in [0, 1]. For every $f \in C ^ { J , \gamma } ( \mathbb { R } ^ { d } )$ , we then can write $f = \chi _ { R } f + \left( 1 - \chi _ { R } \right) f$ . If $\| f \| _ { C ^ { J , \gamma } } \leq 1$ , then, for every fixed $R > 0 .$ , the bounded Lipschitz norm of $\chi _ { R } f$ is bounded by a constant $C _ { R }$ independent of $f .$ Consequently, denoting by $d _ { \mathrm { B I } }$ the bounded Lipschitz distance,

$$
\operatorname* { s u p } _ { \| \boldsymbol { f } \| _ { C ^ { J , \gamma } } \leq 1 } \bigg | \int _ { \mathbb { R } ^ { d } } \chi _ { R } \boldsymbol { f } \mathrm { d } ( \mu ^ { N } - \mu ) \bigg | \leq C _ { R } d _ { \mathrm { B L } } ( \mu ^ { N } , \mu ) .
$$

The other term is controlled by

$$
\operatorname* { s u p } _ { \| f \| _ { \sigma ^ { J } , \gamma \leq 1 } } \bigg | \int _ { \mathbb R ^ { d } } ( 1 - \chi _ { R } ) f \mathrm { d } ( \mu ^ { N } - \mu ) \bigg | \leq \int _ { \{ | x | > R \} } ( 1 + | x | ^ { \gamma } ) \mathrm { d } \mu ^ { N } ( x ) + \int _ { \{ | x | > R \} } ( 1 + | x | ^ { \gamma } ) \mathrm { d } \mu ( x ) .
$$

Since convergence in $W _ { \gamma }$ implies weak convergence together with uniform integrability of the moments of order $\gamma ,$ the right-hand side in the above estimate is uniformly small as $R \to + \infty$ Then, for every fixed $R ,$ weak convergence gives $d _ { \mathrm { B L } } ( \mu ^ { N } , \mu ) \to 0$ . Taking first the upper limit as $N \to + \infty$ and then letting $R \to + \infty$ gives

$$
\| i ( \mu ^ { N } ) - i ( \mu ) \| _ { ( C ^ { J , \gamma } ( \mathbb { R } ^ { d } ) ) ^ { * } } \to 0 .
$$

The above observation provides an alternative way to prove Theorem 1. Indeed, after applying the Skorokhod representation theorem to the joint convergence, one obtains representatives such that $\mathsf { M } ^ { N } \to \bar { \mu }$ almost surely in $\mathcal { D } \big ( \mathbb { R } _ { + } , \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) \big )$ . Since the limiting trajectory $\bar { \mu }$ is continuous for the $W _ { \gamma }$ topology, it follows at every fixed time $t \geq 0$ that $W _ { \gamma } ( \mathsf { M } _ { t } ^ { N } , \bar { \mu } _ { t } ) \to 0$ almost surely. The above continuous embedding then yields $\mathsf { M } _ { t } ^ { N } \to \bar { \mu } _ { t }$ almost surely in $\mathcal { H }$ , and the Taylor expansion appearing in the functional delta method part of the proof can be applied directly. ⋄

The representation assumption in Corollary 2 deserves some additional discussion. Since H is a Hilbert space, every continuous linear functional $D \Phi _ { i } ( \bar { \mu } _ { t } ) \in \mathcal { H } ^ { \prime }$ is represented by a unique element of $\mathcal { H } ^ { \prime } \simeq \mathcal { H } ^ { J _ { 0 } - 1 , j _ { 0 } } ( \mathbb { R } ^ { d } )$ . The additional assumption in Corollary 2 is therefore not the existence of such a Sobolev representative, but its stronger regularity $\varphi _ { i } ^ { t } \in C _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$ , which allows Theorem 15 to be applied directly. In practice, such a representation arises naturally for a broad class of observables. For instance, if $\Phi _ { i } ( \mu ) = \langle q _ { i } , \mu \rangle$ , then $D \Phi _ { i } ( \mu ) ( h ) = \langle q _ { i } , h \rangle$ , so that one may simply take $\varphi _ { i } ^ { t } = q _ { i }$ . More generally, for a cylindrical observable of the form

$$
\Phi _ { i } ( \mu ) = F _ { i } \big ( \langle q _ { 1 } , \mu \rangle , \dots , \langle q _ { m } , \mu \rangle \big ) ,
$$

the chain rule yields

$$
D \Phi _ { i } ( \bar { \mu } _ { t } ) ( h ) = \Big \langle \sum _ { \ell = 1 } ^ { m } \partial _ { \ell } F _ { i } \big ( \langle q _ { 1 } , \bar { \mu } _ { t } \rangle , \dots , \langle q _ { m } , \bar { \mu } _ { t } \rangle \big ) q _ { \ell } , h \Big \rangle .
$$

Hence, in this case, the representing function is explicitly given by

$$
\varphi _ { i } ^ { t } = \sum _ { \ell = 1 } ^ { m } \partial _ { \ell } F _ { i } \big ( \langle q _ { 1 } , \bar { \mu } _ { t } \rangle , \dots , \langle q _ { m } , \bar { \mu } _ { t } \rangle \big ) q _ { \ell } .
$$

In particular, the regularity assumption of Corollary 2 is satisfied whenever the functions $q _ { \ell }$ belong to $C _ { b } ^ { \infty } ( \mathbb { R } ^ { d } )$

Also and as aforementioned, the smoothness requirement on the representing functions is mainly inherited from the covariance representation established in Theorem 15. One may in principle seek to extend this representation to a larger Sobolev class by establishing suitable continuity properties of the solution to the associated backward transport equation with respect to its terminal datum. Such an extension appears natural in view of the continuity properties of the backward solution operator established in the previous work, but would require a slight refinement of those arguments. We therefore restrict here to representatives for which Theorem 15 applies directly.

We conclude this section by proving the local factorization criterion for identifiability. The result shows that, under the constant rank assumption, the diferential kernel condition throughout a neighborhood integrates into an exact local nonlinear factorization.

Proof of Theorem 3. We first prove that (i) implies (ii). Assume that there exist open neighborhoods $\mathcal { U } _ { 1 } ~ \subset ~ \mathcal { U } _ { 0 }$ and $\nu ,$ together with a continuously diferentiable mapping $F : \mathcal { V }  \mathbb { R } ^ { \ell }$ such that it holds on $\mathcal { U } _ { 1 }$ that $Q = F \circ \Phi$ . The chain rule formula gives, for every $\mu \in \mathcal { U } _ { 1 }$ that $D Q ( \mu ) = D F ( \Phi ( \mu ) ) \circ D \Phi ( \mu )$ . Therefore, if $h \in \mathop { \mathrm { K e r } } D \Phi ( \mu )$ , then

$$
D Q ( \mu ) ( h ) = D F ( \Phi ( \mu ) ) \bigl ( D \Phi ( \mu ) ( h ) \bigr ) = D F ( \Phi ( \mu ) ) ( 0 ) = 0 .
$$

Hence for every $\mu \in \mathcal { U } _ { 1 }$ we get Ker $D \Phi ( \mu ) \subset \ker D Q ( \mu )$ , which proves (ii).

We now prove that (ii) implies (i). Set $K : = \ker D \Phi ( \mu _ { 0 } )$ and $E : = K ^ { \perp }$ , so that ${ \mathcal { H } } = E \oplus K$ and the restriction $D \Phi ( \mu _ { 0 } ) _ { | E  } : E  \operatorname { R a n } D \Phi ( \mu _ { 0 } )$ is an isomorphism. In particular, dim $E = r$ Since E is finite-dimensional and DΦ is continuous, up to reduce $\mathcal { U } _ { 2 } , D \Phi ( \mu ) _ { | E }$ remains one to one for every $\mu \in { \mathcal { U } } _ { 2 }$ . The constant rank assumption then yields $D \Phi ( \mu ) ( E ) \stackrel { \cdot } { = } \mathrm { R a n } D \Phi ( \mu )$ , so that $D \Phi ( \mu ) _ { | E } : E $ Ran $D \Phi ( \mu )$ is an isomorphism. Hence the assumptions of the constant rank theorem (see e.g. R.Abraham, J.Marsden & T.Ratiu [1, Theorem 2.5.15]) are satisfied, and it provides open neighborhoods $\mathcal { O } _ { H } \subset \mathcal { U } _ { 2 }$ of $\mu _ { 0 }$ in $H , \mathcal { O } _ { \mathbb { R } ^ { k } }$ of $\Phi ( \mu _ { 0 } )$ in $\mathbb { R } ^ { k }$ ,and $C ^ { 1 }$ coordinate maps $\Psi : \mathcal { O } _ { H }  \mathcal { O } _ { E } \times \mathcal { O } _ { K }$ and $\xi : \mathcal { O } _ { \mathbb { R } ^ { k } } \to \mathcal { O } _ { \mathbb { R } ^ { r } } \times \mathcal { O } _ { \mathbb { R } ^ { k - r } }$ which are $C ^ { 1 }$ difeomorphisms onto their images and satisfy $\Psi ( \mu _ { 0 } ) = ( 0 , 0 ) , \xi ( \Phi ( \mu _ { 0 } ) ) = ( 0 , 0 )$ , where $\mathcal { O } _ { E } \subset E , \mathcal { O } _ { K } \subset K , O _ { \mathbb { R } ^ { r } } \subset \mathbb { R } ^ { r }$ $O _ { \mathbb { R } ^ { k - r } } \subset \mathbb { R } ^ { k - r }$ are open neighborhoods of 0 respectively in each space. Also, after possibly reducing these neighborhoods, we may assume that ${ \mathcal { O } } _ { K }$ is convex and that we have

$$
\big ( \xi \circ \Phi \circ \Psi ^ { - 1 } \big ) ( u , v ) = ( u , 0 )
$$

for every $( u , v ) \in \mathcal { O } _ { E } \times \mathcal { O } _ { K }$ , after identifying E with $\mathbb { R } ^ { r }$ . Define the mapping $q : \mathcal { O } _ { E } \times \mathcal { O } _ { K } \to \mathbb { R } ^ { \ell }$ $( u , v ) \mapsto Q ( \Psi ^ { - 1 } ( u , v ) )$ . Then, for every $( u , v )$ in this product neighborhood and every $\eta \in K$ we have

$$
D \big ( \xi \circ \Phi \circ \Psi ^ { - 1 } \big ) ( u , v ) ( 0 , \eta ) = 0 .
$$

Since ξ is a $C ^ { 1 }$ dfeomorphism, we have that $D \xi \big ( \Phi \big ( \Psi ^ { - 1 } ( u , v ) \big ) \big )$ is one to one, so it follows that $D \Psi ^ { - 1 } ( u , v ) ( 0 , \eta ) \in \mathop { \mathrm { K e r } } D \Phi ( \Psi ^ { - 1 } ( u , v ) )$ . Using the kernel inclusion in (ii), we thus obtain

$$
{ \cal D } Q ( { \Psi } ^ { - 1 } ( u , v ) ) \big ( { \cal D } { \Psi } ^ { - 1 } ( u , v ) ( 0 , \eta ) \big ) = 0 .
$$

Thanks to the chain rule formula, this is equivalent to $\partial _ { v } q ( u , v ) ( \eta ) = 0$ . Therefore, for every fixed u and every $v _ { 1 } , v _ { 2 } \in \mathcal { O } _ { K }$ , it holds

$$
q ( u , v _ { 2 } ) - q ( u , v _ { 1 } ) = \int _ { 0 } ^ { 1 } \partial _ { v } q \big ( u , v _ { 1 } + s ( v _ { 2 } - v _ { 1 } ) \big ) \big ( v _ { 2 } - v _ { 1 } \big ) \mathrm { d } s = 0 .
$$

Hence $\boldsymbol { q } ( u , v )$ does not depend on v. Defining then $\widehat { F } : { \mathcal { O } } _ { E } \to \mathbb { R } ^ { \ell } , u \mapsto q ( u , 0 )$ , we thus get $q ( u , v ) = { \widehat { F } } ( u )$

Let $\pi _ { r } : \mathbb { R } ^ { r } \times \mathbb { R } ^ { k - r } \to \mathbb { R } ^ { r }$ denote the projection onto the first component. Since $\pi _ { r } \xi ( \Phi ( \mu _ { 0 } ) ) =$ $0 \in { \mathcal { O } } _ { E }$ , by continuity of $\pi _ { r } \circ \xi$ we may choose an open neighborhood $\nu \subset \mathcal { O } _ { \mathbb { R } ^ { k } }$ of $\Phi ( \mu _ { 0 } )$ such that $\pi _ { r } \xi ( \mathcal { V } ) \subset \mathcal { O } _ { E }$ . We may therefore define $F : \mathcal { V } \to \mathbb { R } ^ { \ell } , y \mapsto \widehat { F } \big ( \pi _ { r } \xi ( y ) \big )$ . Let us point out that $F$ is continuously diferentiable since $\widehat F , \pi _ { r }$ , and ξ are so. Since Φ is continuous and $\Phi ( \mu _ { 0 } ) \in \mathcal { V } _ { : }$ we may further choose an open neighborhood $\mathcal { U } _ { 1 } \subset \mathcal { O } _ { H } \subset \mathcal { U } _ { 2 }$ of $\mu _ { 0 }$ such that $\Phi ( \mathcal { U } _ { 1 } ) \subset \mathcal { V }$ . Hence, for every such $\mu ,$ we get that

$$
{ \cal F } ( \Phi ( \mu ) ) = \widehat { F } \big ( \pi _ { r } \xi ( \Phi ( \mu ) ) \big ) = \widehat { F } ( u ) = q ( u , v ) = Q ( \mu ) .
$$

Hence $Q = F \circ \Phi$ on such a defined open set $\mathcal { U } _ { 1 }$ , with $\mathcal { U } _ { 1 } \subset \mathcal { U } _ { 0 } , \Phi ( \mathcal { U } _ { 1 } ) \subset \mathcal { V }$ , and $F \in C ^ { 1 } ( \mathcal { V } , \mathbb { R } ^ { \ell } )$ which proves (i).

In particular, if $\mu _ { 1 } , \mu _ { 2 } \in \mathcal { U } _ { 1 }$ satisfy $\Phi ( \mu _ { 1 } ) = \Phi ( \mu _ { 2 } )$ , then $Q ( \mu _ { 1 } ) = F ( \Phi ( \mu _ { 1 } ) ) = F ( \Phi ( \mu _ { 2 } ) ) =$ $Q ( \mu _ { 2 } )$ □

## A Auxiliary probabilistic and functional results

As discussed throughout the article, some arguments rely on auxiliary technical results of rather diferent nature, whose detailed proofs would interrupt the clarity of the presentation. We have therefore gathered them in the present appendix. First, we establish the probabilistic argument that combines the convergence of the empirical measure with that of its fluctuation process. We then turn to a specific property of the weighted Sobolev scale underlying our functional framework, namely the Hilbert-Schmidt character of the embeddings used throughout the paper.

## A.1 A joint convergence lemma

The following lemma isolates the joint convergence argument used in the proof of Theorem 1. In the present setting, the result may be viewed as a kind of generalized Slutsky theorem, or obtained from a usual converging-together result for random elements taking values in Polish spaces. Nevertheless, we provide thereafter a direct proof based on bounded Lipschitz test functions. Besides making the paper self-contained, this formulation makes explicit the underlying probabilistic mechanism used in the argument. More preciselty, it highlights the fact that convergence in probability of the empirical measure towards a deterministic limit can be combined with convergence in distribution of the fluctuation process to obtain their joint convergence.

Lemma 18 (Joint convergence). Assume that the assumptions of Theorems 10 and $1 \mathit { 4 }$ hold. Set $B _ { 1 } : = \mathcal { D } \big ( \mathbb { R } _ { + } , \mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } ) \big )$ and $B _ { 2 } : = \mathcal { D } ( \mathbb { R } _ { + } , \mathcal { H } )$ , both endowed with their usual Skorokhod $J _ { 1 }$ topologies. Then

$$
( \mu ^ { N } , \eta ^ { N } ) \xrightarrow [ N  \infty ] { \mathcal { L } } ( \bar { \mu } , \bar { \eta } ) i n B _ { 1 } \times \mathcal { B } _ { 2 } .
$$

Proof. The space $\mathcal { P } _ { \gamma } ( \mathbb { R } ^ { d } )$ endowed with the Wasserstein distance $W _ { \gamma }$ is Polish. Moreover, the Skorokhod space $\mathcal { D } ( \mathbb { R } _ { + } , E )$ endowed with the $J _ { 1 }$ topology is Polish whenever E is Polish. Consequently, $\begin{array} { r } { B _ { 1 } , \ B _ { 2 } . } \end{array}$ and $B _ { 1 } \times B _ { 2 }$ are Polish spaces. We then set $\beta ^ { N } : = ( \mu ^ { N } , \eta ^ { N } ) , \tilde { \beta } ^ { N } : = ( \bar { \mu } , \eta ^ { N } )$ and $\beta : = ( \bar { \mu } , \bar { \eta } )$ . Therefore, we get $d p _ { 1 } \times { \mathcal { B } } _ { 2 } ( \beta ^ { N } , \tilde { \beta } ^ { N } ) = d { \mathcal { B } } _ { 1 } ( \mu ^ { N } , \bar { \mu } )$ so that $d _ { B _ { 1 } \times B _ { 2 } } ( \beta ^ { N } , \tilde { \beta } ^ { N } )  0$ in probability, since Theorem 10 gives $\mu ^ { N } \to \bar { \mu }$ in probability in $\boldsymbol { B } _ { 1 }$

We also set the mapping $F _ { \bar { \mu } } : B _ { 2 } \to B _ { 1 } \times B _ { 2 }$ defined by $F _ { \bar { \mu } } ( \eta ) = ( \bar { \mu } , \eta )$ , which is clearly continuous for the considered topologies. Then, we point out that thanks to Theorem 14 and the continuous mapping theorem (see e.g. the book by S.Resnick [53, Corollary 8.3.1]), we get that

$$
\tilde { \beta } ^ { N } = F _ { \bar { \mu } } ( \eta ^ { N } ) \xrightarrow [ N  \infty ] { \mathcal { L } } F _ { \bar { \mu } } ( \bar { \eta } ) = \beta \quad \mathrm { i n } ~ \mathcal { B } _ { 1 } \times \mathcal { B } _ { 2 } .\tag{22}
$$

Setting now $Q _ { N } : = \mathcal { L } ( \beta ^ { N } ) , \tilde { Q } _ { N } : = \mathcal { L } ( \tilde { \beta } ^ { N } )$ and $Q : = { \mathcal { L } } ( \beta )$ , we aim to show that

$$
\langle f , Q _ { N } \rangle _ { \begin{array} { c } { { N  + \infty } } \end{array} } \langle f , Q \rangle ,
$$

where $\langle \cdot , \cdot \rangle$ denotes the duality bracket between test functions and measures. Thanks to the Portmanteau theorem (see e.g. the book by S.Resnick [53, Theorem 8.4.1]) combined with the metrizability characterization of convergence in law (see the book by A.W.van der Vaart & J.Wellner [64, Theorem 1.12.4]), it is enough to show the result for test functions $f$ belonging to the space of bounded Lipschitz functions. Assuming that such a property holds, we write

$$
\left| \langle f , Q _ { N } \rangle - \langle f , Q \rangle \right| \leq \left| \langle f , Q _ { N } \rangle - \langle f , \tilde { Q } _ { N } \rangle \right| + \left| \langle f , \tilde { Q } _ { N } \rangle - \langle f , Q \rangle \right| ,\tag{23}
$$

and thanks to (22), we already get that

$$
| \langle f , \tilde { Q } _ { N } \rangle - \langle f , Q \rangle | _ { \begin{array} { c } { { N  + \infty } } \end{array} } 0 .
$$

We then write

$$
\big | \langle f , Q _ { N } \rangle - \langle f , \tilde { Q } _ { N } \rangle \big | = \bigg | \mathbb { E } \big [ f ( \beta ^ { N } ) \big ] - \mathbb { E } \big [ f ( \tilde { \beta } ^ { N } ) \big ] \bigg | \le \mathbb { E } \bigg [ \big | f ( \beta ^ { N } ) - f ( \tilde { \beta } ^ { N } ) \big | \bigg ] ,
$$

and set $A _ { N } ^ { \varepsilon } : = \big \{ d _ { B _ { 1 } \times B _ { 2 } } ( \beta ^ { N } , \tilde { \beta } ^ { N } ) \ \leq \varepsilon \big \}$ . Since f is assumed to be a bounded Lipschitz test function it follows

$$
\big | \langle f , Q _ { N } \rangle - \langle f , \tilde { Q } _ { N } \rangle \big | \leq L _ { f } \varepsilon + 2 M _ { f } \mathbb { P } \big ( d _ { \mathcal { B } _ { 1 } \times \mathcal { B } _ { 2 } } ( { \beta ^ { N } } , \tilde { \beta } ^ { N } ) > \varepsilon \big ) ,
$$

where $L _ { f } > 0$ and ${ M } _ { f } > 0$ are respectively the Lipschitz and boundedness constants associated to $f .$ . We recall that the last term vanishes as $N \to + \infty$ since we benefit from the convergence of probability of $\mu ^ { N }$ toward $\bar { \mu }$ in $\boldsymbol { B } _ { 1 }$ by Theorem 10, so that coming back to (23) we get

$$
\begin{array} { r } { \left| \langle f , Q _ { N } \rangle - \langle f , Q \rangle \right| \leq \left| \langle f , \tilde { Q } _ { N } \rangle - \langle f , Q \rangle \right| + L _ { f } \varepsilon + 2 M _ { f } \mathbb { P } \big ( d _ { B _ { 1 } \times B _ { 2 } } ( \beta ^ { N } , \tilde { \beta } ^ { N } ) > \varepsilon \big ) \underset { N \to + \infty } { \longrightarrow } L _ { f } \varepsilon , } \end{array}
$$

so that

$$
\operatorname* { l i m } _ { N \to + \infty } \left| \langle f , Q _ { N } \rangle - \langle f , Q \rangle \right| \le L _ { f } \varepsilon .
$$

and letting $\varepsilon \to 0$ this imply the following joint convergence

$$
\begin{array} { r } { ( \mu ^ { N } , \eta ^ { N } ) \xrightarrow [ N  \infty ] { \mathcal { L } } ( \bar { \mu } , \bar { \eta } ) , \quad \mathrm { i n } \ B _ { 1 } \times \mathcal { B } _ { 2 } . } \end{array}
$$

□

## A.2 Hilbert-Schmidt embeddings for weighted Sobolev spaces

We now turn to the functional framework used in our results. As discussed in Section 3.1, under suitable moment and regularity assumptions, probability measures can be identified with continuous linear functionals on weighted spaces of test functions and, through the corresponding continuous embeddings, with elements of dual of weighted Sobolev spaces. This provides a common space in which both the empirical measures and their fluctuation processes can be considered.

The use of weighted Sobolev spaces is particularly convenient because they retain a Hilbert space structure while accounting for the behavior of functions and measures at infinity. This structure makes several standard analytical, probabilistic and statistical tools particularly natural, thanks to the underlying geometry. Moreover, when the required gaps in regularity and weight are satisfied, the relevant embeddings are Hilbert-Schmidt. This property plays an important role in the probabilistic analysis of measure-valued processes, notably in compactness, tightness, and regularity arguments.

Although the Hilbert-Schmidt character of such embeddings is often used in the literature, detailed proofs are rarely included in applications. Such results are typically found in the specialized literature on function spaces, where they are often formulated for substantially more general classes of weights than those considered here. For the reader’s convenience, and to make the precise assumptions required in our polynomially weighted setting explicit, we therefore provide a detailed proof of the embedding result used in this paper.

To be precise, weighted function spaces arise naturally in a wide variety of analytical settings, and a well-developed theory is available for several classes of weights. In particular, weighted Sobolev, Bessel-potential, Besov, and Triebel-Lizorkin spaces associated with Muckenhoupt weights have been extensively studied, and we refer, among others, to the works by M.Meyries & M.Veraar [47, 48, 49] and to the references therein. However, the polynomially decaying weights considered here do not, for the range of exponents relevant to our analysis, fall within the Muckenhoupt framework. Indeed, suficiently strong polynomial decay at infinity prevents these weights from belonging to the corresponding Muckenhoupt classes.

A natural approach would be to rely on a Bessel-potential characterization of the weighted Sobolev spaces. The next result shows that the simultaneous gain of regularity and spatial decay gives rise to a Hilbert-Schmidt operator on $L ^ { 2 } ( \mathbb R ^ { d } )$ . The argument is standard and relies on the kernel representation of Bessel potentials, see $\mathrm { e . g . }$ . the book by L.Grafakos [24, Definition 1.2.4] for a definition. We highlight that for every $a \in \mathbb { R } _ { + }$ , there holds $1 + | x | ^ { 2 a } \sim ( 1 + | x | ^ { 2 } ) ^ { a }$ , so there is no diference in considering one weight or the other. For the sake of simplicity, we shall denote throughout this section $\rho _ { a } ( x ) : = ( 1 + | x | ^ { 2 } ) ^ { - a / 2 }$ . For simplicity of exposition, we write the proof with the equivalent polynomial weight $\rho _ { a }$ . This gives an equivalent norm to the one used in the main text.

Proposition 19. Let $\ell > d / 2$ and $b > d / 2$ . Then the operator

$$
T : = \rho _ { b } ( I - \Delta ) ^ { - \ell / 2 }
$$

is Hilbert-Schmidt on $L ^ { 2 } ( \mathbb R ^ { d } )$

Proof. Let $G _ { \ell }$ be the Bessel kernel associated with $( I - \Delta ) ^ { - \ell / 2 }$ , namely

$$
( I - \Delta ) ^ { - \ell / 2 } f = G _ { \ell } * f .
$$

Thus $T$ is the integral operator with kernel

$$
K ( x , y ) = \rho _ { b } ( x ) G _ { \ell } ( x - y ) .
$$

An integral operator with kernel in $L ^ { 2 } ( \mathbb R ^ { d } \times \mathbb R ^ { d } )$ is Hilbert-Schmidt (see $\mathrm { e . g . }$ the book by M.Reed & B.Simon [52, Theorem VI.23]), and its Hilbert-Schmidt norm is the $L ^ { 2 }$ norm of its kernel. Namely,

$$
\| T \| _ { \mathrm { H . S . } } ^ { 2 } = \iint _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } | K ( x , y ) | ^ { 2 } \mathrm { d } x \mathrm { d } y .
$$

Using the form of K, Fubini theorem, and the change of variables $z = x - y$ , we obtain

$$
\begin{array} { r l r } {  { \| T \| _ { \mathrm { H . S . } } ^ { 2 } = \int \biggr / \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } \rho _ { 2 b } ( x ) | G _ { \ell } ( x - y ) | ^ { 2 } \mathrm { d } x \mathrm { d } y } } \\ & { } & { = \bigg ( \int _ { \mathbb { R } ^ { d } } \rho _ { 2 b } ( x ) \mathrm { d } x \bigg ) \bigg ( \int _ { \mathbb { R } ^ { d } } | G _ { \ell } ( z ) | ^ { 2 } \mathrm { d } z \bigg ) . } \end{array}
$$

The first integral is finite because $b > d / 2$ . The second one is finite since $\widehat { G _ { \ell } } ( \xi ) = ( 1 + | \xi | ^ { 2 } ) ^ { - \ell / 2 }$ and therefore, by Plancherel theorem

$$
\| G _ { \ell } \| _ { L ^ { 2 } } ^ { 2 } = \int _ { \mathbb { R } ^ { d } } ( 1 + | \xi | ^ { 2 } ) ^ { - \ell } \mathrm { d } \xi < + \infty
$$

if and only if $\ell > d / 2$ . Hence $K \in L ^ { 2 } ( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } )$ , and $T$ is Hilbert-Schmidt.

The preceding proposition suggests that the desired Hilbert-Schmidt embedding should result from the combination of two independent efects, namely a gain of regularity, represented by the Bessel potential $( I - \Delta ) ^ { - \ell / 2 }$ , and an additional polynomial decay at infinity, represented by the multiplication operator $\rho _ { b }$ . However, directly identifying the canonical embedding between the weighted Sobolev spaces considered here with such an operator would require an appropriate Bessel-potential characterization of these spaces. While such characterizations are classical in the unweighted setting, and are also available within several classes of weighted function spaces, their direct use for the weights and the definition adopted here is not immediate. We refer e.g. to the book by T.Hytönen, J.vanNeerven, M.Veraar & L.Weis [27, Section 14.7] for the general theory of Bessel-potential spaces and related Littlewood-Paley characterizations.

Fortunately, polynomially weighted function spaces have also been studied outside the Muckenhoupt framework. In particular, a theory of weighted Besov and Triebel-Lizorkin spaces associated with nondegenerate polynomial weights has been developed in the works by A.Caetano [14], by T.Kühn, H.-G.Leopold, W.Sickel & L.Skrzypczak [37, 38, 39] or also by L.Skrzypczak [59]. Of particular relevance here are the precise estimates for the approximation numbers of embeddings between polynomially weighted function spaces obtained in the work by L.Skrzypczak [59], see also the correction by L.Skrzypczak & J.Vybíral [60]. To apply these results to the weighted Sobolev spaces used in the present work, we first relate our definition to the multiplicatively weighted formulation appearing in that theory. The following equivalence of norms provides the required link.

Lemma 20. Let $J \in \mathbb { N }$ and $a \geq 0$ . Then there exist constants $c , C > 0$ , depending only on $J , a ,$ and $d ,$ such that

$$
c \| \rho _ { a } f \| _ { H ^ { J } } \leq \| f \| _ { \mathcal { H } ^ { J , a } } \leq C \| \rho _ { a } f \| _ { H ^ { J } } .
$$

Proof. Let us first point out that for every multi-index $\gamma ,$ there exists a constant $C _ { a , \gamma } > 0$ such that for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$

$$
| \partial ^ { \gamma } \rho _ { a } ( x ) | \leq C _ { a , \gamma } \rho _ { a } ( x ) .
$$

This follows by induction from

$$
\partial _ { j } \rho _ { a } ( x ) = - a \frac { x _ { j } } { 1 + | x | ^ { 2 } } \rho _ { a } ( x )
$$

and from the boundedness of the derivatives of $x _ { j } / ( 1 + | x | ^ { 2 } )$ . Applying Leibniz rule, we get that for every $| \alpha | \le J$ 2

$$
\partial ^ { \alpha } ( \rho _ { a } f ) = \sum _ { \beta \leq \alpha } { \binom { \alpha } { \beta } } ( \partial ^ { \alpha - \beta } \rho _ { a } ) \partial ^ { \beta } f ,
$$

yielding directly

$$
\| \partial ^ { \alpha } ( \rho _ { a } f ) \| _ { L ^ { 2 } } \leq C \sum _ { \beta \leq \alpha } \| \rho _ { a } \partial ^ { \beta } f \| _ { L ^ { 2 } } .
$$

Summing such inequalities over $| \alpha | \le J$ yields

$$
\| \rho _ { a } f \| _ { H ^ { J } } \leq C \bigg ( \sum _ { | \beta | \leq J } \| \rho _ { a } \partial ^ { \beta } f \| _ { L ^ { 2 } } ^ { 2 } \bigg ) ^ { 1 / 2 } ,
$$

leading to the first side of the wished inequalit. Conversely, rewriting Leibniz formula as

$$
\rho _ { a } \partial ^ { \alpha } f = \partial ^ { \alpha } ( \rho _ { a } f ) - \underset { \underset { \beta \neq \alpha } { \beta \leq \alpha } } { \sum } \binom { \alpha } { \beta } ( \partial ^ { \alpha - \beta } \rho _ { a } ) \partial ^ { \beta } f ,
$$

and using once again $| \partial ^ { \alpha - \beta } \rho _ { a } | \le C _ { a , \alpha , \beta } \rho _ { a }$ , we obtain

$$
\| \rho _ { a } \partial ^ { \alpha } f \| _ { L ^ { 2 } } \leq \| \partial ^ { \alpha } ( \rho _ { a } f ) \| _ { L ^ { 2 } } + C \sum _ { | \beta | < | \alpha | } \| \rho _ { a } \partial ^ { \beta } f \| _ { L ^ { 2 } } .
$$

Finally, an induction on $| \alpha | \ \mathrm { g i }$ ves

$$
\| \rho _ { a } \partial ^ { \alpha } f \| _ { L ^ { 2 } } \leq C \| \rho _ { a } f \| _ { H ^ { J } } , { \mathrm { ~ f o r ~ } } | \alpha | \leq J .
$$

Summing over $| \alpha | \le J$ proves the reverse estimate and hence the equivalence of norms. The argument is first carried out for $f \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ and extends to $H ^ { J , a } ( \mathbb { R } ^ { d } )$ by density. □

The preceding lemma shows that, up to equivalent Hilbert norms, the weighted Sobolev spaces considered here can be identified with Sobolev spaces defined through multiplication by a nondegenerate polynomial weight. This allows us to reduce the desired embedding to an embedding between polynomially weighted Triebel-Lizorkin spaces of the type studied by L.Skrzypczak [59]. We may therefore combine this identification with the asymptotic estimates for approximation numbers established therein to obtain the Hilbert-Schmidt property of the canonical embeddings.

Proposition 21. Let $j , \ell \in \mathbb { N } , a , b \geq 0$ , and assume $\ell , b > d / 2$ . Then the canonical embedding

$$
\mathcal { H } ^ { j + \ell , a } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } ^ { j , a + b } ( \mathbb { R } ^ { d } )
$$

is Hilbert-Schmidt.

Proof. Let us denote the canonical embedding $I : \mathcal { H } ^ { j + \ell , a } ( \mathbb { R } ^ { d } ) \to \mathcal { H } ^ { j , a + b } ( \mathbb { R } ^ { d } )$ . By Lemma 20, the source and target norms are respectively equivalent to $\| \rho _ { a } f \| _ { H ^ { j + \ell } }$ and $\| \rho _ { a + b } f \| _ { H ^ { j } }$ . We also highlight that $w _ { b } ( x ) : = \rho _ { - b } ( x ) = ( 1 + | x | ^ { 2 } ) ^ { b / 2 }$ , which is precisely the polynomial weight used in the work by L.Skrzypczak [59]. Thus, we define $U : \mathcal { H } ^ { j , \bar { a } + b } ( \mathbb { R } ^ { d } ) \overset { \cdot } { \to } H ^ { \bar { j } } ( \mathbb { R } ^ { \bar { d } } )$ such that

$$
U f : = \rho _ { a + b } f .
$$

Also, by Lemma 20, U is bounded and one-to-one. Moreover, for every $g \in H ^ { j } ( \mathbb { R } ^ { d } )$ , the function $f : = \rho _ { - a - b } g$ belongs to $\mathcal { H } ^ { j , a + b } ( \mathbb { R } ^ { d } )$ and satisfies $U f = g ,$ , and therefore $\rho _ { a } f = \rho _ { - b } g$ . Hence $U$ is a bounded isomorphism with bounded inverse. Thus, $\| f \| _ { \mathcal { H } ^ { j + \ell , a } }$ and $\| \rho _ { - b } g \| _ { H ^ { j + \ell } }$ are equivalent. Following the notation of L.Skrzypczak [59], we define the weighted Triebel-Lizorkin space by

$$
F _ { 2 , 2 } ^ { j + \ell } ( \mathbb { R } ^ { d } , \rho _ { - b } ) : = \bigg \{ g \in \mathcal { S } ^ { \prime } ( \mathbb { R } ^ { d } ) \ | \ \rho _ { - b } g \in F _ { 2 , 2 } ^ { j + \ell } ( \mathbb { R } ^ { d } ) \bigg \} ,
$$

with norm $\| g \| _ { F _ { 2 . 2 } ^ { j + \ell } ( \rho _ { - b } ) } : = \| \rho _ { - b } g \| _ { F _ { 2 . 2 } ^ { j + \ell } }$ . Since $F _ { 2 , 2 } ^ { s } ( \mathbb { R } ^ { d } ) = H ^ { s } ( \mathbb { R } ^ { d } )$ with equivalent norms, the preceding identity shows that U also gives an isomorphism, up to equivalent Hilbert norms, from $\mathcal { H } ^ { j + \ell , a } ( \mathbb { R } ^ { d } )$ onto $F _ { 2 . 2 } ^ { j + \ell } ( \mathbb { R } ^ { d } , \rho _ { - b } )$ . Hence the embedding I is conjugate, through bounded isomorphisms with bounded inverses, to the canonical embedding

$$
\widetilde { I } : F _ { 2 , 2 } ^ { j + \ell } ( \mathbb { R } ^ { d } , \rho _ { - b } )  F _ { 2 , 2 } ^ { j } ( \mathbb { R } ^ { d } ) .
$$

In particular, the Hilbert-Schmidt property of I is equivalent to that of ${ \widetilde { I } } .$ We now apply the result by L.Skrzypczak [59, Theorem 17(i)]. In the notation of that result, we take $p _ { 0 } = p _ { 1 } = 2$ $q _ { 0 } = q _ { 1 } = 2 , s _ { 0 } = j + \ell , s _ { 1 } = j$ , and $\alpha = b$ . The parameter denoted there by $\delta$ is

$$
\delta = s _ { 0 } - s _ { 1 } - d \biggl ( \frac { 1 } { p _ { 0 } } - \frac { 1 } { p _ { 1 } } \biggr ) = \ell .
$$

Now since $p _ { 0 } = p _ { 1 } = 2$ and $b \neq \ell ,$ , the result by L.Skrzypczak [59, Theorem $1 7 ( \mathrm { i } ) ]$ yields

$$
a _ { k } \widetilde { ( I ) } \sim k ^ { - \operatorname* { m i n } ( b , \ell ) / d } , \ \mathrm { a s } \ k  \infty ,
$$

where $a _ { k } ( \widetilde { I } )$ denotes the k-th approximation number of $\stackrel { \sim } { I } ( \mathrm { s e e ~ e . g }$ . the book by A.Pietsch [50, Definition 2.3.1]). Because both the source and target spaces are Hilbert spaces, the approximation numbers of the compact operator $\widetilde { I }$ coincide with its singular values $a _ { k } ( \widetilde { I } ) = s _ { k } ( \widetilde { I } )$ (see e.g. the book by A.Pietsch [50, Theorem 2.11.9]). Therefore, applying the result in e.g. the book by A.Pietsch [50, Proposition 2.11.17],

$$
\widetilde { I } \in \mathfrak { S } _ { 2 } \Leftrightarrow \sum _ { k = 1 } ^ { \infty } a _ { k } ( \widetilde { I } ) ^ { 2 } < \infty .
$$

Using the above asymptotic estimate we have that,

$$
\sum _ { k = 1 } ^ { \infty } a _ { k } ( \widetilde { I } ) ^ { 2 } \sim \sum _ { k = 1 } ^ { \infty } k ^ { - 2 \operatorname* { m i n } \{ b , \ell \} / d } ,
$$

and since $b , \ell > d / 2$ we have 2min $( b , \ell ) / d > 1$ , and thus

$$
\sum _ { k = 1 } ^ { \infty } a _ { k } ( \widetilde { I } ) ^ { 2 } < \infty .
$$

Hence Ie is Hilbert-Schmidt thanks to a well-known result (see e.g. the book by A.Pietsch [50, Theorem 1.4.5]).

Finally, the class of Hilbert-Schmidt operators is stable under composition with bounded operators. Since I and Ie are conjugate through bounded isomorphisms with bounded inverses, it follows that I is Hilbert-Schmidt.

It remains to consider the limiting case $b = \ell .$ . Choose $b _ { 0 } \in ( d / 2 , b )$ . Since $b _ { 0 } \neq \ell ,$ , the preceding argument gives the Hilbert-Schmidt embedding ${ \mathcal { H } } ^ { j + \ell , a } ( \mathbb { R } ^ { d } ) \stackrel { \mathrm { H . S . } } { \hookrightarrow } { \mathcal { H } } ^ { j , a + b _ { 0 } } ( \mathbb { R } ^ { d } )$ . Moreover, because $b _ { 0 } < b$ , the canonical embedding $\mathcal { H } ^ { j , a + b _ { 0 } } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } ^ { j , a + b } ( \mathbb { R } ^ { d } )$ is continuous. Hence the canonical embedding $\mathcal { H } ^ { j + \ell , a } ( \mathbb { R } ^ { d } ) \hookrightarrow \mathcal { H } ^ { j , a + b } ( \mathbb { R } ^ { d } )$ is Hilbert-Schmidt as the composition of a Hilbert-Schmidt operator with a bounded operator, which ends the proof. □

Acknowledgements. Geofrey Lacour was partially funded by the NumOpTES project (ANR22-CE46-0005) of the French National Research Agency. This work was partially carried out while Geofrey Lacour was a postdoctoral researcher at Institut de Mathématiques de Bordeaux, CNRS UMR 5251.

## References

[1] R. Abraham, J. Marsden & T. Ratiu: Manifolds, tensor analysis, and applications. New York: Springer-Verlag, Appl. Math. Sci., 2nd ed. (1988).

[2] R. Adams & J. Fournier: Sobolev spaces. Academic Press Pure Appl. Math., 2nd ed. (2003).

[3] L. Ambrosio, N. Gigli & G. Savaré: Gradient Flows in Metric Spaces and in the Space of Probability Measures Birkhäuser Basel, Lectures in Mathematics ETH Zürich (2008)

[4] P. Andersen, Ø. Borgan, R. Gill & N. Keiding: Statistical models based on counting processes. Springer Ser. Stat. (1995).

[5] F. Bach: Learning theory from first principles. MIT Press, Adaptive Computation and Machine Learning series (2024).

[6] F. Bach & L. Chizat: Gradient descent on infinitely wide neural networks: global convergence and generalization. European Mathematical Society (EMS) ICM 2022 (2023).

[7] E. Beutner: The functional delta method for deriving asymptotic distributions. Wiley Interdiscip. Rev. WIREs Comput. Stat. (2026).

[8] E. Beutner & H. Zähle: A modified functional delta method and its application to the estimation of risk functionals. J. Multivariate Anal. (2010).

[9] E. Beutner & H. Zähle: Functional delta-method for the bootstrap of quasi-Hadamard differentiable functionals. Electron. J. Stat. (2016).

[10] P. Billingsley: Convergence of probability measures. (2nd ed.) Wiley Ser. Probab. Stat. (1999).

[11] V. Bogachev: Measures on topological spaces. J. Math. Sci. (1998).

[12] V. Bogachev: Measure theory. Vol. II. Springer Berlin, Heidelberg (2007).

[13] S. Bourguin & K. Spiliopoulos: Uniform-in-time quantitative fluctuations of large scale interacting particle systems. aXiv Preprint 2605.03057 (2026).

[14] A. Caetano: About approximation numbers in function spaces. J. Approx. Theory (1998).

[15] R. Carmona & F. Delarue: Probabilistic theory of mean field games with applications I. Mean field FBSDEs, control, and games. Springer Probab. Theory Stoch. Model. (2018).

[16] X. Chen, V. Chernozhukov, S. Lee & W. Newey: Local identification of nonparametric and semiparametric models. Econometrica (2014).

[17] Z. Chen, G.M. Rotskof, J. Bruna & E. Vanden-Eijnden: A dynamical central limit theorem for shallow neural networks. Advances in Neural Information Processing Systems (2020).

[18] J. Cupidon, D. Gilliam, R. Eubank & F. Ruymgaart: The delta method for analytic functions of random operators with application to functional data. Bernoulli (2007).

[19] A. Descours, A. Guillin, M. Michel & B. Nectoux: Law of large numbers and central limit theorem for wide two-layer neural networks: the mini-batch and noisy case. J. Mach. Learn. Res. (2024).

[20] A. Descours, A. Guillin, G. Lacour, M. Michel, B. Nectoux & P. Stos: Quantifying uncertainty in wide two-layer neural networks: on the Law of the limiting fluctuation process. arXiv Preprint, arXiv:2606.05982 (2026).

[21] S. Ethier & T. Kurtz: Markov processes. Characterization and convergence. Wiley Ser. Probab. Stat. (2005).

[22] B. Fernandez & S. Méléard: A Hilbertian approach for fluctuations on the McKean-Vlasov model. Stochastic Processes Appl. (1997).

[23] B. Geshkovski, C. Letrouit, Y. Polyanski & P. Rigollet: A mathematical perspective on transformers. Bull. Am. Math. Soc., New Ser. (2025).

[24] L. Grafakos: Modern Fourier analysis. Springer New-York, Grad. Texts Math., 3rd ed. (2014).

[25] A. Guillin, B. Nectoux & P. Stos: Uniform-in-time concentration in two-layer neural networks via transportation inequalities. arXiv Preprint 2603.01842 (2026).

[26] J. Heinonen, T. Kilpeläinen & O. Martio: Nonlinear potential theory of degenerate elliptic equations. Oxford Math. Monogr. (1993).

[27] T. Hytönen, J. van Neerven, M. Veraar & L. Weis: Analysis in Banach spaces. Volume III. Harmonic analysis and spectral theory. Cham: Springer, Ergeb. Math. Grenzgeb. (2023).

[28] O. Imanuvilov, H. Liu & M. Yamamoto: Lipschitz stability for determination of states and inverse source problem for the mean field game equations. Inverse Probl. Imaging (2024).

[29] O. Imanuvilov & M. Yamamoto: Global Lipschitz stability for an inverse coeficient problem for a mean field game system. Math. Methods Appl. Sci. (2025).

[30] M. Izuki, T. Nogayama, T. Noi & Y. Sawano: Wavelet characterization of local Muckenhoupt weighted Sobolev spaces with variable exponents. Constr. Approx. (2023).

[31] J. Jacod & A. Shiryaev: Limit theorems for stochastic processes. Springer Berlin, Grundlehren Math. Wiss., 2nd ed. (2003).

[32] A. Jakubowski: On the Skorokhod topology. Ann. Inst. Henri Poincaré, Probab. Stat. (1986).

[33] B. Jourdain & A. Tse: Central limit theorem over non-linear functionals of empirical measures with applications to the mean-field fluctuation of interacting difusions. Electron. J. Probab. (2021).

[34] M. Kosorok: Introduction to empirical processes and semiparametric inference. Springer Ser. Stat. (2008).

[35] T. Krainer & B.-W. Schulze: On the inverse of parabolic systems of partial diferential equations of general form in an infinite space-time cylinder., In Parabolicity, Volterra calculus, and conical singularities. A volume of advances in partial diferential equations, Birkhäuser (2002).

[36] A. Kufner: Weighted Sobolev spaces. Wiley-Interscience (1985).

[37] T. Kühn, H.-G. Leopold, W. Sickel & L. Skrzypczak: Entropy numbers of embeddings of weighted Besov spaces. Constr. Approx. (2006).

[38] T. Kühn, H.-G. Leopold, W. Sickel & L. Skrzypczak: Entropy numbers of embeddings of weighted Besov spaces II. Proc. Edinb. Math. Soc., II. Ser. (2006).

[39] T. Kühn, H.-G. Leopold, W. Sickel & L. Skrzypczak: Entropy numbers of embeddings of weighted Besov spaces III: Weights of logarithmic type. Math. Z. (2007).

[40] Q. Lang & F. Lu: Identifiability of interaction kernels in mean-field equations of interacting particles. Found. Data Sci. (2023).

[41] J.-M. Lasry & P.-L. Lions: Mean field games. I: The stationary case. C. R., Math., Acad. Sci. Paris (2006).

[42] J.-M. Lasry & P.-L. Lions: Mean field games. II: Finite horizon and optimal control. C. R., Math., Acad. Sci. Paris (2006).

[43] J.-M. Lasry & P.-L. Lions: Mean field games. Jpn. J. Math. (2007).

[44] Z. Li, F. Lu, M. Maggioni, S. Tang & C. Zhang: On the identifiability of interaction functions in systems of interacting particles. Stochastic Processes Appl. (2021).

[45] F. Lu, M. Maggioni & S. Tang: Learning interaction kernels in stochastic systems of interacting particles from multiple trajectories. Found. Comput. Math. (2022).

[46] S. Mei, A. Montanari & P.-M. Nguyen: A mean field view of the landscape of two-layer neural networks. Proc. Natl. Acad. Sci. USA (2018).

[47] M. Meyries & M. Veraar: Sharp embedding results for spaces of smooth functions with power weights. Stud. Math. (2012).

[48] M. Meyries & M. Veraar: Characterization of a class of embeddings for function spaces with Muckenhoupt weights Arch. Math. (2014).

[49] M. Meyries & M. Veraar: Pointwise multiplication on vector-valued function spaces with power weights J. Fourier Anal. Appl. (2015).

[50] A. Pietsch: Eigenvalues and s-numbers. Camb. Stud. Adv. Math. (1987).

[51] D. Pollard: Convergence of stochastic processes. Springer Ser. Stat. (1984).

[52] M. Reed & B. Simon: Methods of modern mathematical physics. Vol I: Functional analysis. Academic Press (1972).

[53] S. Resnick: A probability path. Mod. Birkhäuser Class. (2005).

[54] T. Rothenberg: Identification in parametric models. Econometrica (1971).

[55] F. Santambrogio: Optimal transport for applied mathematicians. Calculus of variations, PDEs, and modeling. Springer Prog. Nonlinear Difer. Equ. Appl. (2015).

[56] R. Schmidt & U. Stadtmüller: Nonparametric estimation of tail dependence. Scand. J. Stat. (2006).

[57] J. Sirignano & K. Spiliopoulos: Mean field analysis of neural networks: a law of large numbers. SIAM J. Appl. Math. (2020).

[58] J. Sirignano, R. Sowers & K. Spiliopoulos: Mathematical foundations of deep learning models and algorithms. American Mathematical Society, Grad. Stud. Math. (2025).

[59] L. Skrzypczak: On approximation numbers of Sobolev embeddings of weighted function spaces. J. Approx. Theory (2005).

[60] L. Skrzypczak & J. Vybíral: Corrigendum to the paper: “On approximation numbers of Sobolev embeddings of weighted function spaces” [J. Approx. Theory 136 (2005) 91-107]. J. Approx. Theory (2009).

[61] H. Spohn: Large scale dynamics of interacting particles. Springer Texts Monogr. Phys. (1991).

[62] H. Triebel: Theory of function spaces. Mod. Birkhäuser Class., reprint of the 1983 original (2010).

[63] A. Van der Vaart: Asymptotic statistics. Cambridge Ser. Stat. Probab. Math. (2000).

[64] A. Van der Vaart & J. Wellner: Weak convergence and empirical processes. With applications to statistics. Springer Ser. Stat., 2nd ed. (2023).

[65] C. Villani: Optimal Transport Old and New. Springer Berlin Heidelberg, Grundlehren der mathematischen Wissenschaften (2009).

[66] E. Walter: Identifiability of parametric models. Oxford: Pergamon Press (1987).

[67] Z. Wang, X. Zhao & R. Zhu: Gaussian fluctuations for interacting particle systems with singular kernels. Arch. Ration. Mech. Anal. (2023).