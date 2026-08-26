# Sequential operator learning under dependent data

Rafael Oliveira CSIRO Technology Sydney, Australia rafael.dossantosdeoliveira@csiro.au

## Abstract

Learning operators from sequentially collected data arises in adaptive experimental design, Bayesian optimization, and dynamical-system modelling, where observations may be dependent, and future inputs or sensing operators may depend on preceding data. We derive time-uniform self-normalized concentration bounds for stochastic processes in Hilbert spaces with vector-valued noise. We use these bounds to obtain regression-error guarantees for linear operators, including targets outside the Hilbert estimation space, and for nonlinear parametric operators trained with strongly convex losses and regularizers. Our results allow possibly infinite-dimensional inputs and outputs without independence or mixing assumptions, providing a major step towards convergence guarantees for adaptive operator learning and learning from stochastic dynamical data.

## 1 Introduction

Operator learning provides a framework for learning mappings between function spaces, with applications throughout scientific machine learning and dynamical-system modelling (Kovachki et al., 2023; Boullé and Townsend, 2024). Prominent nonlinear architectures include neural operators (Kovachki et al., 2023) and DeepONets (Lu et al., 2021). Sequential dependence arises naturally in weather forecasting, where models such as FourCastNet learn evolution operators from temporally ordered spatial fields (Kurth et al., 2023), and long-horizon climate modelling (Wang et al., 2026). More generally, scientific observations may form a dependent trajectory, while future inputs, forcing conditions, or sensing operators may be selected using the evolving model. These settings fall outside the fixed-design or independent-data assumptions underlying many existing learning guarantees.

Adaptive data collection for operator learning has recently been studied through optimal experimental design (Xu et al., 2026), active learning (Li et al., 2024; Musekamp et al., 2025; Subedi and Tewari, 2025), and Bayesian optimization (Guilhoto and Perdikaris, 2024; Oliveira et al., 2025). These settings may also involve partially observed function-valued operator outputs (e.g., through finite grids or sensor networks) together with structured noise. Hence, available theoretical guarantees remain limited for such general adaptive settings.

Contribution. Building on classical self-normalized concentration (de la Peña et al., 2009; Abbasi Yadkori et al., 2011) and recent vector-valued extensions (Chowdhury and Gopalan, 2021; Chugg and Ramdas, 2025; Martinez-Taboada et al., 2026), we derive a time-uniform concentration inequality for general Hilbert-valued stochastic processes and use it to establish regression-error guarantees for linear and nonlinear operators learned from sequentially dependent data. Our linear result accommodates targets more general than those represented by the Hilbert estimation space, while our nonlinear result covers parametric operator models trained with a general class of loss functions and regularizers. The latter builds on an operator-valued extension of a model-based RKHS construction (Oliveira, 2026). Together, these results provide a common concentration framework for analysing operator learning from dependent trajectories and adaptively collected observations.

Related work. Operator-learning theory addresses several complementary notions of approximation and estimation error. Approximation results establish the expressivity of Fourier neural operators and DeepONets and provide architecture-dependent error estimates (Kovachki et al., 2021; Lanthaler et al., 2022). More recent statistical analyses control generalization for nonlinear operators, including neural operators, but typically assume independently sampled training data (Lee and Shin, 2024; Reinhardt et al., 2026). For linear operators, vector-valued RKHS theory provides a classical regression framework (Micchelli and Pontil, 2003; Carmeli et al., 2010), while Subedi and Tewari (2025) show that actively designed input functions can improve convergence over passive i.i.d. sampling. Concurrently, Wang and Jones (2026) derive related confidence bounds for structured vector-valued measurements, but require trace-class operator-valued kernels. Our analysis instead allows non-trace-class kernels when the noise provides sufficient spectral decay.

Dependent observations have been studied more directly in dynamical-system learning. Foster et al. (2020) derive guarantees from a single trajectory for structured finite-dimensional nonlinear systems, with bounds that depend explicitly on the state dimension. More closely related to our function-space setting, Hou et al. (2023) analyse RKHS-based operator estimation from dependent trajectories, but require i.i.d. or mixing data assumptions. Our results complement this literature by providing time-uniform regression-error bounds directly in infinite-dimensional spaces under general predictable data collection, without requiring independence or mixing. This accommodates both dependent trajectories and observations selected adaptively from the preceding data.

## 2 Background and problem formulation

Notation. We work with separable real Hilbert and Banach spaces. For a Banach space ${ \mathcal F } ,$ its dual is denoted by ${ \mathcal { F } } ^ { * }$ . We use $\langle \cdot , \cdot \rangle$ for both inner products and dual pairings, with the meaning determined by the spaces of the arguments. In particular, $\langle f , \varphi \rangle = f ( \varphi )$ for $f \in { \mathcal { F } } ^ { * }$ and $\varphi \in { \mathcal { F } }$ Norms are denoted by $\lVert \cdot \rVert$ , with a space subscript added when needed for clarity. We write $\dot { \mathcal { L } } ( u , \nu )$ for the bounded linear operators from U to $\nu$ and $A ^ { * }$ for the Banach or Hilbert adjoint, as appropriate. For a positive-semidefinite operator B on a Hilbert space, let $\| h \| _ { B } ^ { 2 } : = \langle h , B h \rangle$ , and write $A \preceq B$ when ${ \bar { B } } - A$ is positive semidefinite. The Schatten classes are denoted by $S _ { p } ( \mathcal { U } , \mathcal { V } )$ , with $p = 1$ and $p = 2$ corresponding to trace-class and Hilbert–Schmidt operators. For trace-class $A ,$ det $( I + A )$ denotes the Fredholm determinant. More detailed definitions are presented in Appendix $\mathbf { A }$

Sequential operator learning. Let $F _ { \star } : \mathcal { U }  \mathcal { V }$ be an unknown, possibly nonlinear operator between Hilbert spaces which we want to approximate. At iteration $t \in \mathbb { N } .$ , we observe:

$$
y _ { t } = D _ { t } F _ { \star } ( u _ { t } ) + \xi _ { t } \in \mathcal { V } _ { t } ,
$$

where $D _ { t } \in \mathcal { L } ( \nu , \mathcal { V } _ { t } )$ may represent partial observation or discretization mapping to a Hilbert space $\mathcal { { D } } _ { t }$ . We assume that $u _ { t }$ and $D _ { t }$ are predictable with respect to a filtration $\{ \mathfrak { F } _ { t } \} _ { t \ge 0 }$ representing the information available to the algorithm (the observed data and any algorithmic randomness revealed up to each round), while $\xi _ { t }$ is $\mathfrak { F } _ { t }$ -measurable and conditionally sub-Gaussian noise, given $\mathfrak { F } _ { t - 1 }$ . Thus, both the inputs and observation operators may depend arbitrarily on the preceding data.

This formulation includes stochastic dynamics. For example, if $x _ { t + 1 } = F _ { \mathrm { d y n } } ( x _ { t } ) + \eta _ { t }$ and $y _ { t } = D _ { t } x _ { t }$ then $F _ { \star }$ may represent the predictable map from the current observation or an estimated latent state to the next function-valued state. The inputs need only be predictable representations of the available history, so the observed process need not be Markovian or fully observed. The same formulation covers deliberate adaptivity, where future inputs or sensing operators are chosen based on observations.

## 3 Main result

Our main technical result extends self-normalized concentration to general Hilbert-valued increments.   
Supporting definitions are provided in Appendix $\mathbf { A } ,$ and our proofs are located in Appendix C.

Theorem 3.1 (Self-normalized vector-valued concentration). Let H be a separable real Hilbert space and $\{ \mathfrak { F } _ { t } \} _ { t = 0 } ^ { \infty } a \mathrm { ~ } f i l t r a t i o n .$ . Let $\{ \eta _ { t } \} _ { t = 1 } ^ { \infty }$ be an adapted H-valued sequence such that $\eta _ { t }$ is conditionally $\Sigma _ { t }  – s u b$ -Gaussian, where $\Sigma _ { t }$ is a predictable positive-semidefinite trace-class operator. Then, for any

boundedly invertible positive-definite operator U on H and any $\delta \in ( 0 , 1 ]$

$$
\mathbb { P } \left[ \forall t \in \mathbb { N } , \quad \lVert \sum _ { i = 1 } ^ { t } \eta _ { i } \rVert _ { ( U + V _ { t } ) ^ { - 1 } } ^ { 2 } \leq 2 \log \left( \frac { \operatorname* { d e t } ( I + U ^ { - 1 / 2 } V _ { t } U ^ { - 1 / 2 } ) ^ { 1 / 2 } } { \delta } \right) \right] \geq 1 - \delta ,\tag{1}
$$

where $\textstyle V _ { t } : = \sum _ { i = 1 } ^ { t } \sum _ { i }$

The proof extends finite-dimensional self-normalization through finite-rank approximations and continuity of the relevant quadratic forms and Fredholm determinants. The complete argument is given in Theorem B.1 and Corollary B.2. The following section applies Theorem 3.1 to regression.

## 4 Applications to operator learning

We now apply Theorem 3.1 to operator regression under sequentially dependent observations. We first consider linear operators, including targets outside the Hilbert estimation space, and then extend the analysis to nonlinear parametric models with strongly convex losses and regularizers.

## 4.1 Linear regression in Hilbert spaces

Let $F _ { \star } : \mathcal { U } \to \mathcal { V }$ be an unknown bounded linear operator between Hilbert spaces U and V. Suppose we approximate $F _ { \star }$ by a Hilbert-Schmidt operator $\widehat { F } _ { t } \in S _ { 2 } ( \mathcal { U } , \mathcal { V } )$ after $t \geq 1$ observations. As $F _ { \star }$ need not be Hilbert-Schmidt, we measure approximation errors through the dual pairing with trace-class operators $\Phi \in { \cal S } _ { 1 } ( \mathcal { U } , \mathcal { V } )$

$$
\langle F , \Phi \rangle : = \operatorname { t r } ( F ^ { * } \Phi ) = \operatorname { t r } ( \Phi ^ { * } F ) , \qquad F \in { \mathcal { L } } ( \mathcal { U } , \mathcal { V } ) .\tag{2}
$$

For instance, given a rank-one test operator $\Phi = v \otimes u ,$ , then $\langle F _ { \star } - \widehat { F } _ { t } , \Phi \rangle = \langle v , ( F _ { \star } - \widehat { F } _ { t } ) u \rangle$ , which measures the prediction error at input u along the output direction v. To avoid clutter, we write $\mathcal { F } : =  { S _ { 1 } } ( \mathcal { U } , \overset { \vartriangle } { \boldsymbol { \mathcal { V } } } )$ and $\mathscr { H } : = S _ { 2 } ( \mathscr { U } , \mathscr { V } )$ , with ${ \mathcal { F } } ^ { * }$ identified with ${ \mathcal { L } } ( u , \nu )$ through the trace pairing.

For each $t \in \mathbb N$ , let $\mathrm { M } _ { t } : \mathcal { V } _ { t } \to \mathcal { F }$ be a predictable bounded linear map and let $\mathrm { M } _ { t } ^ { * } : \mathcal { F } ^ { * }  \mathcal { V } _ { t }$ denote its Banach adjoint, identifying the dual of the Hilbert space $\mathcal { V } _ { t }$ with itself through the Riesz representation. We observe $y _ { t } = \mathrm { M } _ { t } ^ { \ast } ( F _ { \star } ) + \xi _ { t }$ . For example, $\mathrm { M } _ { t } ( v ) = ( D _ { t } ^ { * } v ) \otimes u _ { t }$ , whose adjoint satisfies $\mathrm { M } _ { t } ^ { * } ( F ) = D _ { t } F u _ { t }$ . At each round, we then estimate $\widehat { F } _ { t }$ from the accumulated data by minimizing:

$$
L _ { t } ( \boldsymbol { F } ) : = \lambda \| \boldsymbol { F } \| _ { \mathcal { H } } ^ { 2 } + \sum _ { i = 1 } ^ { t } \| y _ { i } - \mathrm { M } _ { i } ^ { * } ( \boldsymbol { F } ) \| _ { \mathcal { V } _ { i } } ^ { 2 } , \qquad \boldsymbol { F } \in \mathcal { H } ,\tag{3}
$$

where $\lambda > 0$ is a regularization factor. We can now state our result on the linear regression error.

Theorem 4.1 (Linear regression error). Let $\mathrm { M } _ { t } : \mathcal { V } _ { t } \to \mathcal { F }$ be predictable bounded linear maps and suppose that $\xi _ { t }$ is conditionally $\Sigma _ { t }  – s u b –$ Gaussian, where $\Sigma _ { t } ~ i s ~ \mathfrak { F } _ { t - 1 }$ -measurable and positive semidefinite, $f o r t \geq 1$ . Assume,for afixed $\sigma _ { \xi } ^ { 2 } > 0$ , that $\| \Sigma _ { t } \| _ { \mathrm { o p } } \leq \sigma _ { \xi } ^ { 2 }$ and that $\mathrm { M } _ { t } \Sigma _ { t } \mathrm { M } _ { t } ^ { * }$ is almost surely trace class on H,for all $t \in \mathbb { N }$ . Define $\begin{array} { r } { C _ { t } : = \lambda I + \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } } \end{array}$ and $\begin{array} { r } { V _ { t } : = \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \Sigma _ { i } \mathrm { M } _ { i } ^ { * } } \end{array}$ Then, given $\delta \in ( 0 , 1 ]$ , with probability at least $1 - \delta ,$

$$
| \langle F _ { \star } - \widehat { F } _ { t } , \Phi \rangle | \leq \lambda \| C _ { t } ^ { - 1 } ( \Phi ) \| _ { 1 } \| F _ { \star } \| _ { \mathrm { o p } } + \| C _ { t } ^ { - 1 / 2 } ( \Phi ) \| _ { 2 } \sqrt { 2 \sigma _ { \xi } ^ { 2 } \log \left( \frac { \operatorname* { d e t } ( I + \lambda ^ { - 1 } \sigma _ { \xi } ^ { - 2 } V _ { t } ) ^ { 1 / 2 } } { \delta } \right) } .\tag{4}
$$

simultaneouslyfor all $\Phi \in { \cal S } _ { 1 } ( \mathcal { U } , \mathcal { V } )$ and $t \in \mathbb { N } .$

To illustrate how Theorem 4.1 can yield convergence rates once the data-collection geometry is controlled, consider a simple finite-dimensional persistent-excitation setting.

Corollary 4.2 (Illustrative convergence rate). Under the assumptions of Theorem $4 . l ,$ , suppose that the observation operators satisfy Ran $\left( \mathrm { M } _ { t } \right) \subseteq S$ for a fixed d-dimensional subspace $s \subset \mathcal { F }$ $\| \mathrm { M } _ { t } \| _ { \mathrm { o p } } \leq L$ , and $\begin{array} { r } { \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } \succeq \kappa t P _ { S } } \end{array}$ for some κ, $L > 0$ and all sufficiently large t. Then, for everyfixed $\Phi \in S$ and $\delta \in ( 0 , 1 ]$ , with probability at least $1 - \delta ,$

$$
| \langle F _ { \star } - \widehat { F } _ { t } , \Phi \rangle | = O \left( \sqrt { \frac { d \log ( 1 + t ) + \log ( 1 / \delta ) } { t } } \right) ,
$$

uniformly over sufficiently large t. In particular, for fixed d and δ, the error is $O ( { \sqrt { \log t / t } } )$ .

More generally, obtaining rates without the finite-dimensional persistent-excitation assumption requires controlling the variance-like $\| C _ { t } ^ { - 1 / 2 } ( \Phi ) \| _ { 2 }$ and Fredholm log-determinant terms in (4), potentially through operator-valued analogues of the maximum information gain (Vakili et al., 2021).

## 4.2 Nonlinear regression with parametric models

We now approximate $F _ { \star } : \mathcal { U } \to \mathcal { V }$ by a nonlinear parametric model $F : \mathcal { U } \times \Theta  \mathcal { V }$ , such as a neural operator. Let H be a separable Hilbert space containing the model class $\{ F ( \cdot , \theta ) \} _ { \theta \in \Theta }$ . Assume evaluations are given by $\mathrm { M } _ { t } ^ { * } ( F )$ , where $\bar { \mathrm { M } } _ { t } : \mathcal { V } _ { t } \to \mathcal { H }$ is a Hilbert-Schmidt operator, for $F \in { \mathcal { H } }$ Given pointwise losses $\ell _ { t } : \dot { \mathcal { V } } _ { t } \times \mathcal { V } _ { t } \to \mathbb { R }$ and predictable regularizers $R _ { t } : \mathcal { H }  \mathbb { R }$ , we estimate:

$$
\widehat { \theta } _ { t } \in \mathop { \mathrm { a r g m i n } } _ { \theta \in \Theta } \left\{ \sum _ { i = 1 } ^ { t } \ell _ { i } \bigl ( \mathrm { M } _ { i } ^ { * } ( F ( \cdot , \theta ) ) , y _ { i } \bigr ) + R _ { t } \bigl ( F ( \cdot , \theta ) \bigr ) \right\} , \qquad t \in \mathbb { N } .\tag{5}
$$

Theorem 4.3 (Nonlinear parametric regression). For each $t \in \mathbb { N } ,$ , assume that $( i ) \ell _ { t } ( \cdot , y _ { t } )$ is twice Fréchet differentiable and α-strongly convex, $( i i ) R _ { t }$ is Fréchet differentiable and λ -strongly convex, with $\lambda _ { t } \ \geq \ \lambda _ { 0 } \ > \ 0 ,$ , and $( i i i ) ~ \xi _ { t } : = \nabla _ { 1 } \ell _ { t } ( \mathrm { M } _ { t } ^ { * } ( F _ { \star } ) , y _ { t } )$ is conditionally Σ -sub-Gaussian, with $\| \Sigma _ { t } \| _ { \mathrm { o p } } \leq \sigma _ { \xi } ^ { 2 } .$ . Suppose that $F _ { \star } = F ( \cdot , \theta _ { \star } )$ , for some $\theta _ { \star } \in \Theta _ { \mathsf { 3 } }$ , and that $\widehat { \theta } _ { t }$ is a global solution $o f \left( 5 \right)$ Then, for every $\delta \in ( 0 , 1 ]$ , with probability at least $1 - \delta ,$ , simultaneously for every $t \in \mathbb { N } ,$

$$
\| F ( \cdot , \widehat { \theta } _ { t } ) - F _ { \star } \| _ { H _ { t } } \leq 2 \| \nabla R _ { t } ( F _ { \star } ) \| _ { H _ { t } ^ { - 1 } } + \frac { 2 \sigma _ { \xi } } { \sqrt { \alpha } } \sqrt { 2 \log \biggl ( \frac { \operatorname* { d e t } ( I + \lambda _ { 0 } ^ { - 1 } \alpha \mathrm { M } _ { 1 : t } \mathrm { M } _ { 1 : t } ^ { * } ) ^ { 1 / 2 } } { \delta } \biggr ) }\tag{6}
$$

where $\begin{array} { r } { H _ { t } : = \lambda _ { t } I + \alpha \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } } \end{array}$

The result follows from Lemma B.3, global optimality over the parametric model class, and the self-normalized control of the cumulative gradient noise $\textstyle \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i }$ . The factor of two arises because the parametric estimator need not minimize the lifted loss over the whole space H.

For a practical regularizer, we extend the model-based RKHS construction of Oliveira (2026) to operator-valued predictions. Let $\mathcal { H } _ { \Theta }$ be an RKHS over Θ with kernel $K _ { \Theta } : \Theta \times \Theta \to \mathcal { L } ( \mathcal { V } )$ ), and suppose that $F ( \bar { u } , \cdot ) : \Theta $ V lie in $\mathcal { H } _ { \Theta }$ , for each $\iota \in \mathcal { U }$ . Then, letting $\Psi _ { u } : \mathcal { H } _ { \Theta }  \mathcal { V }$ represent the model evaluation at $u \in \mathcal { U } ,$ such that $\Psi _ { u } K _ { \Theta } ( \cdot , \theta ) = F ( u , \theta )$ , it follows that $K _ { F } ( u , u ^ { \prime } ) : = \Psi _ { u } \Psi _ { u ^ { \prime } } ^ { * }$ is an operator-valued kernel whose RKHS H contains the model class $\{ F ( \cdot , \theta ) \} _ { \theta \in \Theta }$

Given a predictable random initialization $\theta _ { t , 0 } ,$ , one may take $R _ { t } ( F ) : = ( \lambda _ { t } / 2 ) \lVert F - F ( \cdot , \theta _ { t , 0 } ) \rVert _ { \mathcal { H } } ^ { 2 }$ . This satisfies the preceding assumptions and gives $\| \nabla R _ { t } ( F _ { \star } ) \| _ { H ^ { - 1 } } \leq \sqrt { \lambda _ { t } } \| F _ { \star } - F ( \cdot , \theta _ { t , 0 } ) \| _ { \mathcal H }$ . Moreover, the latter distance is bounded by $\lVert K _ { \Theta } ( \cdot , \theta _ { \star } ) - K _ { \Theta } ( \cdot , \theta _ { t , 0 } ) \rVert _ { \mathcal { H } _ { \Theta } } ^ { \circ }$ , allowing the initialization term in (6) to be controlled through the parameter kernel and the loss (5) to be practically implemented.

## 5 Discussion

We presented a general concentration framework for sequential operator learning under dependent, adaptive, and possibly infinite-dimensional observations. A key feature is that Hilbert-valued noise is controlled through operator-valued sub-Gaussian covariance proxies, allowing its spectral structure to enter the confidence bounds. In linear regression, this relaxes restrictions arising from scalarproxy sub-Gaussian analyses (e.g., Chowdhury and Gopalan, 2021; Wang and Jones, 2026): rather than requiring Hilbert-Schmidt observation operators, it suffices that the induced proxies $\mathrm { M } _ { t } \Sigma _ { t } \mathrm { M } _ { t } ^ { * }$ are trace class. This includes conditional mean embedding regression with $K ( \bar { x , } x ^ { \prime } ) = k ( x , x ^ { \prime } ) \breve { I }$ (Grunewälder et al., 2012), whose Gram operator is not trace class for infinite-dimensional outputs.

Self-normalized bounds of this kind underpin confidence sets, regret bounds, and convergence guarantees throughout sequential learning (Abbasi-Yadkori et al., 2011; Chowdhury and Gopalan, 2017, 2021; Wang and Jones, 2026). Our illustrative corollary shows how the linear bound yields convergence along persistently excited directions, while more general rates are possible via, $\mathrm { e . g . }$ operator-valued extensions of Gaussian-process information-gain bounds (Vakili et al., 2021). The nonlinear result is more preliminary, as pointwise confidence bounds additionally require controlling a variance-like evaluation term. These developments are particularly relevant to adaptive experimental design and Bayesian optimization with neural operators, and may also support learning guarantees from stochastic dynamical data under suitable excitation or dependence assumptions.

## References

Yasin Abbasi-Yadkori. Online Learningfor Linearly Parametrized Control Problems. PhD thesis, University of Alberta, 2012.

Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved Algorithms for Linear Stochastic Bandits. In J Shawe-Taylor, R Zemel, P Bartlett, F Pereira, and K Q Weinberger, editors, Advances in Neural Information Processing Systems, volume 24, pages 1–19, Granada, Spain, 2011. Curran Associates, Inc. URL https://proceedings.neurips.cc/paper\_files/paper/2011/ file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf.

Nicolas Boullé and Alex Townsend. Chapter 3 - A mathematical guide to operator learning. In Siddhartha Mishra and Alex Townsend, editors, Numerical Analysis Meets Machine Learning, volume 25 of Handbook ofNumerical Analysis, pages 83–125. Elsevier, 2024. doi: https://doi.org/ 10.1016/bs.hna.2024.05.003. URL https://www.sciencedirect.com/science/article/ pii/S1570865924000036.

C. Carmeli, E. De Vito, A. Toigo, and V. Umanità. Vector valued reproducing kernel Hilbert spaces and universality. Analysis and Applications, 8(1):19–61, 2010. ISSN 02195305. doi: 10.1142/S0219530510001503.

Sayak Ray Chowdhury and Aditya Gopalan. On Kernelized Multi-armed Bandits. In Proceedings of the 34th International Conference on Machine Learning (ICML), Sydney, Australia, 2017.

Sayak Ray Chowdhury and Aditya Gopalan. No-regret Algorithms for Multi-task Bayesian Optimization. In International Conference on Artificial Intelligence and Statistics (AISTATS), volume 130 of Proceedings of Machine Learning Research, pages 1873–1881, San Diego, CA, USA, 2021. PMLR.

Ben Chugg and Aaditya Ramdas. A variational approach to dimension-free self-normalized concentration. arXiv e-prints, page arXiv:2508.06483, 8 2025. doi: 10.48550/arXiv.2508.06483.

John B. Conway. A Course in Operator Theory. Graduate Studies in Mathematics. American Mathematical Society, Providence, RI, 2000. ISBN 0821820656.

Victor H. de la Peña, Michael J. Klass, and Tze Leung Lai. Theory and applications of multivariate self-normalized processes. Stochastic Processes and their Applications, 119(12):4210–4227, 2009. ISSN 03044149. doi: 10.1016/j.spa.2009.10.003.

Chun Yuan Deng. A generalization of the Sherman-Morrison-Woodbury formula. Applied Mathematics Letters, 24(9):1561–1564, 2011. ISSN 08939659. doi: 10.1016/j.aml.2011.03.046. URL http://dx.doi.org/10.1016/j.aml.2011.03.046.

Rick Durrett. Probability: theory and examples. Cambridge University Press, New York, NY, 5th edition, 2019.

Dylan J. Foster, Alexander Rakhlin, and Tuhin Sarkar. Learning nonlinear dynamical systems from a single trajectory. Proceedings ofMachine Learning Research, 120:287–297, 2020. ISSN 26403498.

George Giorgobiani, Vakhtang Kvaratskhelia, and Vaja Tarieladze. Notes on Sub-Gaussian Random Elements. In George Jaiani and David Natroshvili, editors, Applications of Mathematics and Informatics in Natural Sciences and Engineering, pages 197–203. Springer International Publishing, Cham, 2020. ISBN 978-3-030-56356-1. doi: 10.1007/978-3-030-56356-1{\_}11. URL http: //link.springer.com/10.1007/978-3-030-56356-1\_11.

Rita Giuliano-Antonini. Subgaussian random variables in Hilbert spaces. Rendiconti del Seminario Matematico della Università di Padova, 98:89–99, 1997. URL http://www.numdam.org/item/ RSMUP\_1997\_\_98\_\_89\_0.

Israel C. Gohberg and Mark G. Kre˘ın. Introduction to the theory of linear nonselfadjoint operators. Transactions of Mathematical Monographs. American Mathematical Society, Providence, RI, 1969.

Steffen Grunewälder, Guy Lever, Luca Baldassarre, Sam Patterson, Arthur Gretton, and Massimiliano Pontil. Conditional mean embeddings as regressors. In Proceedings of the 29th International Conference on Machine Learning (ICML), Edinburgh, Scotland, UK, 2012.

Leonardo Ferreira Guilhoto and Paris Perdikaris. Composite Bayesian Optimization In Function Spaces Using NEON – Neural Epistemic Operator Networks. Scientific Reports, 14, 2024. ISSN 20452322. doi: 10.1038/s41598-024-79621-7. URL http://arxiv.org/abs/2404.03099.

H V Henderson and S R Searle. On Deriving the Inverse of a Sum of Matrices. SIAM Review, 23(1): 53–60, 1981. doi: 10.1137/1023004. URL https://doi.org/10.1137/1023004.

Boya Hou, Sina Sanjari, Nathan Dahlin, Subhonmesh Bose, and Umesh Vaidya. Sparse Learning of Dynamical Systems in RKHS: An Operator-Theoretic Approach. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 13325–13352, Honolulu, Hawaii, USA, 2023. PMLR. URL https://proceedings.mlr.press/v202/hou23c.html.

Nikola Kovachki, Samuel Lanthaler, and Siddhartha Mishra. On Universal Approximation and Error Bounds for Fourier Neural Operators. Journal ofMachine Learning Research, 22(290):1–76, 2021. URL http://jmlr.org/papers/v22/21-0806.html.

Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural Operator: Learning Maps Between Function Spaces. Journal ofMachine Learning Research, 24(89), 2023. URL http://jmlr.org/papers/v24 21-1524.html.

Shige Toshi Kuroda. On a theorem of Weyl-von Neumann. Proceedings of the Japan Academy, 34(1):11–15, 1958. doi: 10.3792/pja/1195524841. URL https://doi.org/10.3792/pja/ 1195524841.

Thorsten Kurth, Shashank Subramanian, Peter Harrington, Jaideep Pathak, Morteza Mardani, David Hall, Andrea Miele, Karthik Kashinath, and Anima Anandkumar. FourCastNet: Accelerating Global High-Resolution Weather Forecasting Using Adaptive Fourier Neural Operators. In Proceedings of the Platform for Advanced Scientific Computing Conference, PASC ’23, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701900. doi: 10.1145/3592979.3593412. URL https://doi.org/10.1145/3592979.3593412.

Samuel Lanthaler, Siddhartha Mishra, and George E Karniadakis. Error estimates for DeepONets: a deep learning framework in infinite dimensions. Transactions ofMathematics and Its Applications, 6(1):tnac001, 2022. ISSN 2398-4945. doi: 10.1093/imatrm/tnac001. URL https://doi.org/ 10.1093/imatrm/tnac001.

Sanghyun Lee and Yeonjong Shin. On the Training and Generalization of Deep Operator Networks. SIAM Journal on Scientific Computing, 46(4):C273–C296, 2024. doi: 10.1137/23M1598751. URL https://doi.org/10.1137/23M1598751.

Shibo Li, Xin Yu, Wei Xing, Robert Kirby, Akil Narayan, and Shandian Zhe. Multi-Resolution Active Learning of Fourier Neural Operators. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li, editors, Proceedings ofThe 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings ofMachine Learning Research, pages 2440–2448. PMLR, 2024. URL https://proceedings.mlr.press/v238/li24k.html.

Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, 2021. ISSN 2522-5839. doi: 10.1038/s42256-021-00302-5. URL https://doi.org/10.1038/s42256-021-00302-5.

Diego Martinez-Taboada, Tomás González, and Aaditya Ramdas. Vector-valued self-normalized concentration inequalities beyond sub-Gaussianity. In 37th International Conference on Algorithmic Learning Theory, Toronto, Canada, 2026. URL https://openreview.net/forum?id= Y98zW0bDL0.

Charles A. Micchelli and Massimiliano Pontil. On Learning Vector-Valued Functions. Technical report, Department of Computer Science, University College London, London, UK, 2003.

Mattes Mollenhauer and Claudia Schillings. On the concentration of subgaussian vectors and positive quadratic forms in Hilbert spaces. arXiv e-prints, 2023. URL http://arxiv.org/abs/2306. 11404.

Daniel Musekamp, Marimuthu Kalimuthu, David Holzmüller, Makoto Takamoto, and Mathias Niepert. Active Learning for Neural PDE Solvers. In International Conference on Learning Representations (ICLR), Singapore, 2025. OpenReview. URL http://arxiv.org/abs/2408. 01536.

Rafael Oliveira. Kernel-based guarantees for nonlinear parametric models in Bayesian optimization. arXiv e-prints, page arXiv:2605.13160, 5 2026. doi: 10.48550/arXiv.2605.13160.

Rafael Oliveira, Xuesong Wang, Kian Ming A. Chai, and Edwin V. Bonilla. Thompson Sampling in Function Spaces via Neural Operators. In Advances in Neural Information Processing Systems, volume 38, San Diego, CA, USA, 2025.

Niklas Reinhardt, Sven Wang, and Jakob Zech. Statistical Learning Theory for Neural Operators. Journal ofMachine Learning Research, 27, 2026.

Konrad Schmüdgen. Unbounded Self-Adjoint Operators on Hilbert Space. Springer, 2012. ISBN 9789400747524.

Barry Simon. Notes on infinite determinants of Hilbert space operators. Advances in Mathematics, 24(3):244–273, 1977.

Barry Simon. Trace Ideals and Their Applications, volume 120 of Mathematical Surveys and Monographs. American Mathematical Society, 2 edition, 2010. ISBN 978-0821849880.

Unique Subedi and Ambuj Tewari. On the Benefits of Active Data Collection in Operator Learning. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 57164– 57190, Vancouver, Canada, 2025. PMLR. URL https://proceedings.mlr.press/v267/ subedi25a.html.

Sattar Vakili, Kia Khezeli, and Victor Picheny. On Information Gain and Regret Bounds in Gaussian Process Bandits. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 82–90. PMLR, 2021. URL https://proceedings.mlr. press/v130/vakili21a.html.

Wenbin Wang and Colin N Jones. Bayesian Optimization with Structured Measurements: A Vector-Valued RKHS Framework. arXiv e-prints, page arXiv:2605.09775, 2026. URL https://arxiv. org/abs/2605.09775.

Xuesong Wang, Michael Groom, Rafael Oliveira, He Zhao, Terence O’kane, and Edwin V Bonilla. Multi-Scale Wavelet Transformers for Operator Learning of Dynamical Systems. In Forty-third International Conference on Machine Learning (ICML), Seoul, Korea, 2026. URL https: //openreview.net/forum?id=D70OyQr3O3.

Joachim Weidmann. Linear Operators in Hilbert Spaces, volume 68 of Graduate Texts in Mathematics. Springer New York, New York, NY, 1980. ISBN 978-1-4612-6029-5. doi: 10.1007/ 978-1-4612-6027-1. URL http://link.springer.com/10.1007/978-1-4612-6027-1.

Xingzi Xu, Johann Guilleminot, and Vahid Tarokh. Data generation with optimal experimental design for operator learning. Computer Methods in Applied Mechanics and Engineering, 450: 118675, 2026. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2025.118675. URL https: //www.sciencedirect.com/science/article/pii/S0045782525009478.

## A Further background

## A.1 Definitions

Definition 1 (p-Schatten norm). Let $A : \mathcal { U } \to \mathcal { V }$ be a bounded linear operator between separable Hilbert spaces U and V. Given $1 \leq p < \infty ,$ , the Schatten p-norm of A is defined as:

$$
\| A \| _ { p } : = ( \operatorname { t r } ( ( A ^ { * } A ) ^ { p / 2 } ) ) ^ { \frac { 1 } { p } } .\tag{7}
$$

For $p = \infty ,$ , we set $\| A \| _ { \infty } : = \| A \| _ { \mathrm { o p } } .$ . The space ofall bounded linear operators between U and V withfinite Schatten p-norm is denoted by $\bar { \mathbfcal { S } _ { p } } ( \mathcal { U } , \mathcal { V } )$

For any $p \geq 1$ , the Schatten p-class $S _ { p } ( \mathcal { U } , \mathcal { V } )$ is a Banach space under the Schatten p-norm. As special cases, $S _ { 1 } ( \mathcal { U } , \mathcal { V } )$ corresponds to the space of all trace-class operators on H equipped with the trace (or nuclear) norm. For $p = 2 , S _ { 2 } ( \bar { \mathcal { U } } , \mathcal { V } )$ is the space of Hilbert-Schmidt operators on $\mathcal { H } ,$ which is actually a Hilbert space equipped with the inner product $\langle A , B \rangle : = \mathrm { t r } ( A ^ { * } \bar { B } )$ . As $p  \infty$ $s _ { \infty } ( u , \nu )$ is the Banach space of compact operators on $\mathcal { H }$ equipped with $\| \cdot \| _ { \infty } = \| \cdot \| _ { \mathrm { o p } }$ . Therefore, we have the set inclusion:

$$
\begin{array} { r } { S _ { 1 } ( \mathcal { U } , \mathcal { V } ) \subset S _ { 2 } ( \mathcal { U } , \mathcal { V } ) \subset S _ { \infty } ( \mathcal { U } , \mathcal { V } ) \subset \mathcal { L } ( \mathcal { U } , \mathcal { V } ) , } \end{array}\tag{8}
$$

or more generally $S _ { p } ( \mathcal { U } , \mathcal { V } ) \subset S _ { q } ( \mathcal { U } , \mathcal { V } )$ , for any $1 \leq p < q \leq \infty$ . Schatten p-norms are also unitarily invariant (Kuroda, 1958) and obey an extension of Hölder’s inequality (Simon, 1977):

$$
| \mathrm { t r } ( A B ) | \leq \| A \| _ { p } \| B \| _ { q } ,\tag{9}
$$

for $A \in \mathcal { S } _ { p } ( \mathcal { V } , \mathcal { F } )$ and $B \in { \cal S } _ { q } ( \mathcal { U } , \mathcal { V } )$ with $p , q \in [ 1 , \infty ]$ such that $p ^ { - 1 } + q ^ { - 1 } = 1$ , where $\mathcal { F }$ is another separable Hilbert space. In particular, for $\bar { A } \in \bar { \mathcal { L } } ( \mathcal { V } , \mathcal { F } )$ and $B \in { \mathcal { S } } _ { 1 } ( { \mathcal { U } } , { \mathcal { V } } )$ , it holds that (Conway, 2000, Thm. 18.11):

$$
| \mathrm { t r } ( A B ) | \leq \| A \| _ { \mathrm { o p } } \| B \| _ { 1 } .\tag{10}
$$

For more details about Schatten p-classes and p-norms, the reader is referred to standard operator theory textbooks (e.g., Weidmann, 1980; Conway, 2000; Simon, 2010).

Definition 2 (Fredholm determinant). Let $A \in { \mathcal { S } } _ { 1 } ( { \mathcal { H } } )$ be a trace-class operator on a separable Hilbert space H. The Fredholm determinant det $( I + A )$ is given by:

$$
\operatorname * { d e t } ( I + A ) = \prod _ { i = 1 } ^ { \infty } ( 1 + \lambda _ { i } ( A ) ) ,\tag{11}
$$

where $\lambda _ { i } ( A )$ denotes the ith eigenvalue of $A ,$ including multiplicities and zeros.

The Fredholm determinant has the following basic properties (Simon, 1977):

(Series)

$$
\operatorname* { d e t } ( I + A ) = \exp \left( \sum _ { n = 1 } ^ { \infty } { \frac { ( - 1 ) ^ { n } } { n } } \mathrm { t r } ( A ^ { n } ) \right)\tag{12}
$$

(Continuity)

$$
\begin{array} { r } { \| A _ { n } - A \| _ { 1 } \xrightarrow { n \to \infty } 0 \implies \operatorname* { d e t } ( I + A _ { n } ) \xrightarrow { n \to \infty } \operatorname* { d e t } ( I + A ) , } \end{array}\tag{13}
$$

for a trace-class operator A and a sequence of trace-class operators $\{ A _ { n } \} _ { n = 1 } ^ { \infty }$ . A few determinant identities from the finite-dimensional case can be easily extended to the infinite-dimensional setting by taking limits of finite-dimensional matrix sequences (see Lemma A.1) due to the continuity of the Fredholm determinant under the trace norm (13). A useful consequence is an extension of Sylvester’s determinant identity to infinite-dimensional operators (see Schmüdgen, 2012, Eq. 9.37, or Gohberg and Kre˘ın, 1969, Ch. IV):

$$
\operatorname* { d e t } ( I + A B ) = \operatorname* { d e t } ( I + B A ) , \quad A \in { \mathcal { S } } _ { 1 } ( { \mathcal { H } } ) , \quad B \in { \mathcal { L } } ( { \mathcal { H } } ) .\tag{14}
$$

Definition 3 (Conditionally sub-Gaussian vector). Let V be a Hilbert space, $\{ \mathfrak { F } _ { t } \} _ { t = 0 } ^ { \infty } a$ filtration, and let $\{ \xi _ { t } \} _ { t \in \mathbb { N } }$ denote a sequence of V-valued random variables adapted to $\{ \mathfrak { F } _ { t } \} _ { t = 1 } ^ { \infty }$ . Then $\xi _ { t }$ is said to be conditionally Σ-sub-Gaussian, given a bounded positive-semidefinite linear operator Σ on V, if:

$$
\forall t \geq 1 , \quad \forall g \in \mathcal { V } , \quad \mathbb { E } [ \exp ( \langle g , \xi _ { t } \rangle ) \mid \mathfrak { F } _ { t - 1 } ] \leq \exp \left( \frac { 1 } { 2 } \langle g , \Sigma g \rangle \right) \quad ( \mathrm { a . s . } ) .\tag{15}
$$

Note that this definition encompasses the common characterization of sub-Gaussianity with respect to a scalar parameter $\sigma _ { \xi } ^ { 2 } > 0$ (Abbasi-Yadkori et al., 2011; Chowdhury and Gopalan, 2017), i.e.:

$$
\forall t \geq 1 , \quad \forall g \in \mathcal { V } , \quad \mathbb { E } [ \exp ( \langle g , \xi _ { t } \rangle ) \mid \mathfrak { F } _ { t - 1 } ] \leq \exp \left( \frac { 1 } { 2 } \| g \| ^ { 2 } \sigma _ { \xi } ^ { 2 } \right) \quad ( \mathrm { a . s . } ) ,\tag{16}
$$

by setting $\Sigma : = \sigma _ { \varepsilon } ^ { 2 } I ,$ , so that $\sigma _ { \xi } ^ { 2 } : = \| \Sigma \| _ { \mathrm { o p } } ,$ as $\langle g , \Sigma g \rangle \leq \| g \| ^ { 2 } \| \Sigma \| _ { \mathrm { o p } }$ . This definition is sometimes referred to as “weakly” sub-Gaussian, given that there is no guarantee that $\| \eta \| _ { \mathcal { V } } < \infty$ , though $\langle \eta , v \rangle _ { \mathcal { V } }$ is $\sigma _ { \xi } ^ { 2 } \| v \| _ { \mathcal { V } } ^ { 2 }$ -sub-Gaussian for all $v \in \mathcal V$ (Giorgobiani et al., 2020). However, the definition of sub-Gaussianity w.r.t. an operator is more general (Giuliano-Antonini, 1997). As a basic consequence of the definition, it follows that, if η is Σ-sub-Gaussian, then $A \eta$ is $A \Sigma A ^ { * } – \mathbf { s u b }$ -Gaussian, for $A \in$ $\mathcal { L } ( \nu , \mathcal { U } )$ (Mollenhauer and Schillings, 2023). Indeed, given any $u \in \mathcal { U } .$ , by Definition 3, it holds that:

$$
\mathbb { E } [ \exp ( \langle A \eta , u \rangle u ) ] = \mathbb { E } [ \exp ( \langle \eta , A ^ { * } u \rangle \nu ) ] \leq \exp \left( { \frac { 1 } { 2 } } \langle A ^ { * } u , \Sigma A ^ { * } u \rangle \right) = \exp \left( { \frac { 1 } { 2 } } \langle u , A \Sigma A ^ { * } u \rangle \right) .\tag{17}
$$

## A.2 Existing results

Lemma A.1 (Schmüdgen, 2012, Cor. 9.14). Let $\{ e _ { n } \} _ { n = } ^ { \infty }$ be an orthonormal basis of a separable Hilbert space H. Then,

$$
\operatorname * { d e t } ( I + A ) = \operatorname * { l i m } _ { n \to \infty } \operatorname * { d e t } ( [ \delta _ { i j } + \langle A e _ { i } , e _ { j } \rangle ] _ { i , j = 1 } ^ { n } ) , \quad A \in \mathcal { S } _ { 1 } ( \mathcal { H } ) ,
$$

where $\delta _ { i j }$ denotes the Kronecker delta.

Lemma A.2 (Weyl-von Neumann theorem, Kuroda, 1958). Let A be a bounded, self-adjoint operator on a separable Hilbert space H. Then, for any $1 < p < \infty$ and $\epsilon > 0$ , there exists a self-adjoint operator $B _ { \epsilon }$ such that $\| B _ { \epsilon } \| _ { p } \leq \epsilon$ and the self-adjoint operator $A + B _ { \epsilon }$ has pure point spectrum.

Lemma A.3 (de la Peña et al., 2009, Thm. 1). Let η be a random vector in $\mathbb { R } ^ { d }$ and Σ be a positive-definite random matrix in $\mathbb { R } ^ { d \times d }$ such that:

$$
\forall \mathbf { v } \in \mathbb { R } ^ { d } , \quad \mathbb { E } \left[ \exp \left( \mathbf { v } ^ { \mathsf { T } } { \boldsymbol \eta } - \frac { 1 } { 2 } \mathbf { v } ^ { \mathsf { T } } { \boldsymbol \Sigma } \mathbf { v } \right) \right] \leq 1 .\tag{18}
$$

Then, for any given fixed positive-definite matrix V,

$$
\mathbb { E } \left[ \sqrt { \frac { \operatorname* { d e t } { \bf V } } { \operatorname* { d e t } ( { \Sigma } + { \bf V } ) } } \exp \left( \frac { 1 } { 2 } \pmb { \eta } ^ { \mathsf { T } } ( \pmb { \Sigma } + \mathbf { V } ) ^ { - 1 } \pmb { \eta } \right) \right] \leq 1 ,\tag{19}
$$

$$
\mathbb { E } \bigg [ \exp \bigg ( \frac { 1 } { 4 } \pmb { \eta } ^ { \intercal } ( \pmb { \Sigma } + \mathbf { V } ) ^ { - 1 } \pmb { \eta } \bigg ) \bigg ] \leq \sqrt { \mathbb { E } \bigg [ \sqrt { \operatorname* { d e t } ( \mathbf { I } + \mathbf { V } ^ { - 1 } \pmb { \Sigma } } \bigg ] } .\tag{20}
$$

Linear algebra identities. We will frequently use linear algebra identities which are well known for finite-dimensional matrices and extensible to the case of bounded linear operators under mild conditions regarding their inverses. Namely, we will use the following form of Woodbury’s identity:

$$
( A + B B ^ { * } ) ^ { - 1 } = A ^ { - 1 } - A ^ { - 1 } B ( I + B ^ { * } A ^ { - 1 } B ) ^ { - 1 } B ^ { * } A ^ { - 1 } ,\tag{21}
$$

which holds for any $B \in \mathcal { L } ( \mathcal { U } , \mathcal { V } )$ and boundedly invertible $A \in { \mathcal { L } } ( \nu )$ , where U and V are Hilbert spaces over the same scalar field. See Deng (2011) for a general version. We will also use the following “push-through” identity (Henderson and Searle, 1981), which is readily applicable to the linear operator case:

$$
( A + B B ^ { * } ) ^ { - 1 } B = A ^ { - 1 } B ( I + B ^ { * } A ^ { - 1 } B ) ^ { - 1 } ,\tag{22}
$$

under the same assumptions on A and B as Equation 21. This identity (22) can be easily verified by an application of Woodbury’s identity, followed by simple linear algebra manipulations.

## B Auxiliary results

We here present auxiliary results we derived to construct the critical arguments for the proofs of our main theoretical results. Some of these results, such as Theorem B.1’s extension of de la Peña et al.’s self-normalized concentration to Hilbert-valued noise, might be of independent interest.

Theorem B.1 (Self-normalized concentration in Hilbert spaces). Let H be a separable real Hilbert space, η be a H-valued random vector and Σ be a positive-semidefinite, trace-class random operator on H such that:

$$
\forall h \in \mathcal { H } , \quad \mathbb { E } \left[ \exp \left( \langle h , \eta \rangle - \frac { 1 } { 2 } \langle h , \Sigma h \rangle \right) \right] \leq 1 .\tag{23}
$$

Consider afixed, boundedly invertible, positive-definite diagonalizable <sup>1</sup> operator $U : \mathcal { H } \to \mathcal { H } ,$ , i.e., there exists an orthonormal basis $\{ e _ { i } \} _ { i = 1 } ^ { \infty } o f \mathcal { H }$ such that $U e _ { i } = \lambda _ { i } e _ { i } , f o r$ some $\lambda _ { i } > 0 _ { i }$ , for all $i \in \mathbb { N }$ and inf $_ { i \in \mathbb { N } } \lambda _ { i } > 0$ . It then holds that:

$$
\mathbb { E } \left[ \frac { 1 } { \sqrt { \operatorname* { d e t } ( I + U ^ { - 1 } \Sigma ) } } \exp \left( \frac { 1 } { 2 } \| \eta \| _ { ( \Sigma + U ) ^ { - 1 } } ^ { 2 } \right) \right] \leq 1 .\tag{24}
$$

Proof. We will start the proof with a restriction to the finite-dimensional case via orthogonal projections. This reduction will allow us to apply Lemma ${ \mathrm { A } } . 3$ to finite-dimensional subspaces of $\mathcal { H } .$ . We will then lift it back up to the infinite-dimensional H by taking appropriate limits of the operators and functions involving them.

We now construct the finite-dimensional restriction. Given the orthonormal basis $\{ e _ { i } \} _ { i = 1 } ^ { \infty }$ of H that diagonalizes U, let $E _ { d } : = [ e _ { 1 } , \ldots , e _ { d } ]$ represent the operator mapping $\mathbf { v } = [ v _ { i } ] _ { i = 1 } ^ { d } \in \mathbb { R } ^ { d }$ to $\begin{array} { r } { E _ { d } \mathbf { v } = \sum _ { i = 1 } ^ { d } v e _ { i } \in \mathcal { H } . } \end{array}$ for any given $d \in \mathbb { N }$ . Similarly, its adjoint $E _ { d } ^ { * }$ maps $h \in \mathcal H$ to the vector $E _ { d } ^ { * } h = \mathbf { h } _ { d } : = [ \langle h , e _ { i } \rangle ] _ { i = 1 } ^ { d } \in \mathbb { R } ^ { d }$ . Combining the two, we have that $P _ { d } : = E _ { d } E _ { d } ^ { * }$ forms a finite-rank orthogonal projection onto a d-dimensional subspace $\mathcal { H } _ { d } = P _ { d } ( \mathcal { H } ) \subset \mathcal { H }$ . Note that, for any $h \in \mathcal { H } _ { d }$ we have $P _ { d } h = h $ . As Equation 23 holds over all $\mathcal { H } ,$ it also holds over $\mathcal { H } _ { d } .$ , and we have that:

$$
\begin{array} { r l } { \forall h \in \mathcal { H } _ { d } , } & { 1 \geq \mathbb { E } \bigg [ \exp \bigg ( \langle h , \eta \rangle - \frac { 1 } { 2 } \langle h , \Sigma h \rangle \bigg ) \bigg ] } \\ & { = \mathbb { E } \bigg [ \exp \bigg ( \langle P _ { d } h , \eta \rangle - \frac { 1 } { 2 } \langle P _ { d } h , \Sigma P _ { d } h \rangle \bigg ) \bigg ] } \\ & { = \mathbb { E } \bigg [ \exp \bigg ( \langle E _ { d } ^ { * } h , E _ { d } ^ { * } \eta \rangle - \frac { 1 } { 2 } \langle E _ { d } ^ { * } h , E _ { d } ^ { * } \Sigma _ { d } E _ { d } E _ { d } ^ { * } h \rangle \bigg ) \bigg ] } \\ & { = \mathbb { E } \bigg [ \exp \bigg ( \mathbf { h } _ { d } ^ { \top } \eta _ { d } - \frac { 1 } { 2 } \mathbf { h } _ { d } ^ { \top } \Sigma _ { d } \mathbf { h } _ { d } \bigg ) \bigg ] , } \end{array}\tag{25}
$$

where $\mathbf { h } _ { d } : = E _ { d } ^ { * } h \in \mathbb { R } ^ { d } , \eta _ { d } : = E _ { d } ^ { * } \eta \in \mathbb { R } ^ { d }$ , and $\Sigma _ { d } : = E _ { d } ^ { * } \Sigma E _ { d } \in \mathbb { R } ^ { d \times d }$ , noting that the inner product in the third line is the Euclidean inner product in $\mathbb { R } ^ { d }$ . Since the inequality above holds for any $h _ { d } \in \mathcal { H } _ { d } .$ we have that:

$$
\forall \mathbf { v } \in \mathbb { R } ^ { d } , \quad \mathbb { E } \left[ \exp \left( \mathbf { v } ^ { \mathsf { T } } \pmb { \eta } _ { d } - \frac { 1 } { 2 } \mathbf { v } ^ { \mathsf { T } } \pmb { \Sigma } _ { d } \mathbf { v } \right) \right] \le 1 .\tag{26}
$$

Furthermore, $\begin{array} { r } { \Sigma _ { d , n } : = \Sigma _ { d } + \frac { 1 } { n } \mathbf { I } \succ \Sigma _ { d } } \end{array}$ is almost surely positive definite, for $n \in \mathbb { N } .$ , and:

$$
\forall \mathbf { v } \in \mathbb { R } ^ { d } , \quad \mathbb { E } \left[ \exp \left( \mathbf { v } ^ { \mathsf { T } } \pmb { \eta } _ { d } - \frac 1 2 \mathbf { v } ^ { \mathsf { T } } \pmb { \Sigma } _ { d , n } \mathbf { v } \right) \right] \le \mathbb { E } \left[ \exp \left( \mathbf { v } ^ { \mathsf { T } } \pmb { \eta } _ { d } - \frac 1 2 \mathbf { v } ^ { \mathsf { T } } \pmb { \Sigma } _ { d } \mathbf { v } \right) \right] \le 1 .\tag{27}
$$

Therefore, by Lemma A.3, setting $\mathbf { V } _ { d } : = E _ { d } ^ { \ast } U E _ { d } = [ \langle e _ { i } , U e _ { j } \rangle ] _ { i , j = 1 } ^ { d } \in \mathbb { R } ^ { d \times d }$ , it holds that:

$$
{ \mathbb E } \left[ \sqrt { \frac { \operatorname* { d e t } { \bf V } _ { d } } { \operatorname* { d e t } ( { \Sigma _ { d , n } } + { \bf V } _ { d } ) } } \exp \left( \frac { 1 } { 2 } \eta _ { d } ^ { \mathsf { T } } ( \Sigma _ { d , n } + { \bf V } _ { d } ) ^ { - 1 } \eta _ { d } \right) \right] \le 1 , \qquad \forall n \in { \mathbb N } .\tag{28}
$$

Now, letting $n  \infty .$ , by Fatou’s lemma (Durrett, 2019, Thm. 1.6.5), we get:

$$
\begin{array} { r l } & { 1 \geq \underset { n  \infty } { \operatorname* { l i m i n f } } \mathbb { E } \Bigg [ \sqrt { \frac { \operatorname* { d e t } \mathbf { V } _ { d } } { \operatorname* { d e t } ( \Sigma _ { d , n } + \mathbf { V } _ { d } ) } } \exp ( \frac { 1 } { 2 } \eta _ { d } ^ { \top } ( \Sigma _ { d , n } + { \mathbf { V } } _ { d } ) ^ { - 1 } \eta _ { d } ) \Bigg ] } \\ & { ~ \geq \mathbb { E } \Bigg [ \underset { n  \infty } { \operatorname* { l i m i n f } } \sqrt { \frac { \operatorname* { d e t } \mathbf { V } _ { d } } { \operatorname* { d e t } ( \Sigma _ { d , n } + { \mathbf { V } } _ { d } ) } } \exp ( \frac { 1 } { 2 } \eta _ { d } ^ { \top } ( \Sigma _ { d , n } + { \mathbf { V } } _ { d } ) ^ { - 1 } \eta _ { d } ) \Bigg ] } \\ & { ~ = \mathbb { E } \Bigg [ \sqrt { \frac { \operatorname* { d e t } \mathbf { V } _ { d } } { \operatorname* { d e t } ( \Sigma _ { d } + { \mathbf { V } } _ { d } ) } } \exp ( \frac { 1 } { 2 } \eta _ { d } ^ { \top } ( \Sigma _ { d } + { \mathbf { V } } _ { d } ) ^ { - 1 } \eta _ { d } ) \Bigg ] . } \end{array}\tag{29}
$$

As the result above holds for arbitrary d $\mathbf { \Phi } \in \mathbb { N } .$ , to extend it to the whole of $\mathcal { H } ,$ we can take the limit as $d \to \infty$ and then apply Fatou’s lemma. Firstly, we need to derive the limits of the expressions involving operators. $\quad \mathbf { A } \mathbf { s } d  \infty ,$ one can easily verify that $P _ { d }$ converges to the identity I in the strong operator topology (SOT), $. . . , \| h - P _ { d } h \| \to \bar { 0 }$ , for all $h \in \mathcal H$ . Recalling the definition of a diagonal operator, we know that $\begin{array} { r } { U = \ddot { \sum } _ { i = 1 } ^ { \infty } \lambda _ { i } e _ { i } \overset { . . . } { \otimes } e _ { i } } \end{array}$ , where the series converges in SOT, and $\{ \lambda _ { i } \} _ { i = 1 } ^ { \infty }$ is the sequence of eigenvalues of U, including their multiplicities, which we assume to be sorted in non-increasing order, i.e., $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \lambda _ { 3 } \geq . . .$ . It then follows that:

$$
\mathbf { V } _ { d } = E _ { d } ^ { * } U E _ { d } = E _ { d } ^ { * } \left( \sum _ { i = 1 } ^ { \infty } \lambda _ { i } e _ { i } \otimes e _ { i } \right) E _ { d } = \sum _ { i = 1 } ^ { d } \lambda _ { i } \mathbf { e } _ { i } \otimes \mathbf { e } _ { i }\tag{30}
$$

$$
\mathbf { V } _ { d } ^ { - 1 } = \sum _ { i = 1 } ^ { d } \lambda _ { i } ^ { - 1 } \mathbf { e } _ { i } \otimes \mathbf { e } _ { i } = E _ { d } ^ { * } \left( \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { - 1 } e _ { i } \otimes e _ { i } \right) E _ { d } = E _ { d } ^ { * } U ^ { - 1 } E _ { d } ,\tag{31}
$$

where $\mathbf { e } _ { i } = E _ { d } ^ { * } e _ { i } \in \mathbb { R } ^ { d }$ is the ith standard basis vector of $\mathbb { R } ^ { d }$ . For the determinant, we then have that:

$$
{ \begin{array} { l } { { \frac { \displaystyle \operatorname* { d e t } \mathbf { V } _ { d } } { \displaystyle \operatorname* { d e t } ( \mathbf { \Sigma } \Sigma _ { d } + \mathbf { V } _ { d } ) } } = { \frac { 1 } { \displaystyle \operatorname* { d e t } ( \mathbf { I } + \mathbf { V } _ { d } ^ { - 1 } \Sigma _ { d } ) } } } \\ { = { \frac { 1 } { \displaystyle \operatorname* { d e t } ( \mathbf { I } + E _ { d } ^ { * } U ^ { - 1 } E _ { d } E _ { d } ^ { * } \Sigma E _ { d } ) } } } \\ { = { \frac { 1 } { \displaystyle \operatorname* { d e t } ( \mathbf { I } + E _ { d } ^ { * } E _ { d } E _ { d } ^ { * } U ^ { - 1 } \Sigma E _ { d } ) } } } \\ { = { \frac { 1 } { \displaystyle \operatorname* { d e t } ( \mathbf { I } + E _ { d } ^ { * } U ^ { - 1 } \Sigma E _ { d } ) } } } \end{array} }\tag{32}
$$

using the identities det $( \mathbf { A } \mathbf { B } ) = \operatorname* { d e t } ( \mathbf { A } )$ det(B), for square matrices A and B, $E _ { d } E _ { d } ^ { * } = P _ { d }$ and $E _ { d } ^ { * } E _ { d } = \mathbf { I }$ , and lastly $\dot { U } ^ { - 1 } \dot { P } _ { d } = \dot { P } _ { d } \dot { U } ^ { - 1 }$ , since $P _ { d }$ and U commute, given that $P _ { d }$ is formed by eigenvectors of U. As Σ is trace-class and $U ^ { - 1 }$ is bounded, since inf<sub>i∈N</sub> $\lambda _ { i } > 0$ , we also have that ${ \cal U } ^ { - 1 } \Sigma$ is trace-class, so that the Fredholm determinant det $( I + U ^ { - 1 } \Sigma )$ and its approximations with finite-rank projections det $( \mathbf { I } + E _ { d } ^ { * } U ^ { - 1 } \Sigma E _ { d } ) = \operatorname* { d e t } ( [ \delta _ { i j } + \langle U ^ { - 1 } \Sigma e _ { i } , e _ { j } \rangle ] _ { i , j = 1 } ^ { d } )$ are well defined. Therefore, by Lemma ${ \mathrm { A . 1 } }$ , it follows that:

$$
\operatorname * { l i m } _ { d \to \infty } \operatorname * { d e t } ( { \bf I } + E _ { d } ^ { * } U ^ { - 1 } \Sigma E _ { d } ) = \operatorname * { d e t } ( I + U ^ { - 1 } \Sigma ) ,\tag{33}
$$

which holds almost surely with respect to the randomness of Σ.

For the quadratic term,

$$
\begin{array} { r l r } & { } & { \pmb { \eta } _ { d } ^ { \top } ( \pmb { \Sigma } _ { d } + \mathbf { V } _ { d } ) ^ { - 1 } \pmb { \eta } _ { d } = \langle \pmb { \eta } _ { d } , ( \pmb { \Sigma } _ { d } + \mathbf { V } _ { d } ) ^ { - 1 } \pmb { \eta } _ { d } \rangle } \\ & { } & { = \langle E _ { d } ^ { \ast } \pmb { \eta } , ( \pmb { \Sigma } _ { d } + \mathbf { V } _ { d } ) ^ { - 1 } E _ { d } ^ { \ast } \pmb { \eta } \rangle } \\ & { } & { = \langle \pmb { \eta } , E _ { d } ( \pmb { \Sigma } _ { d } + \mathbf { V } _ { d } ) ^ { - 1 } E _ { d } ^ { \ast } \pmb { \eta } \rangle . } \end{array}\tag{34}
$$

Letting $W : = \Sigma + U$ and $\mathbf { W } _ { d } : = \pmb { \Sigma } _ { d } + \mathbf { V } _ { d } = E _ { d } ^ { * } W E _ { d } ,$ we have that:

$$
\begin{array} { r l } { \forall h \in \mathcal { H } , \quad E _ { d } ( \Sigma _ { d } + \mathbf { V } _ { d } ) ^ { - 1 } E _ { d } ^ { * } W h = E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * } W h } \\ { = E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * } W P _ { d } h + E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * } W ( I - P _ { d } ) h } \\ { = E _ { d } ( E _ { d } ^ { * } W E _ { d } ) ^ { - 1 } E _ { d } ^ { * } W E _ { d } E _ { d } ^ { * } h + E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * } W ( I - P _ { d } ) h } \\ { = P _ { d } h + E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * } W ( I - P _ { d } ) h . } \end{array}\tag{35}
$$

As $d \to \infty , P _ { d } \to I$ in SOT, so that $P _ { d } h  h$ and $( I - P _ { d } ) h \to 0 .$ , for all $h \in \mathcal H$ . Additionally, as $\mathbf { W } _ { d } = \pmb { \Sigma } _ { d } + \mathbf { V } _ { d } \succeq \mathbf { V } _ { d } ,$ , we have that $\begin{array} { r } { \| \mathbf { W } _ { d } ^ { - 1 } \| \leq \| \mathbf { V } _ { d } ^ { - 1 } \| = \lambda _ { d } ^ { - 1 } \leq \frac { 1 } { \operatorname* { i n f } _ { i \in \mathbb { N } } \lambda _ { i } } < \infty } \end{array}$ , so that $E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * } W$ is uniformly bounded over all $d \in \mathbb { N } .$ , as in $\mathrm { f } _ { i \in \mathbb { N } } \lambda _ { i } > 0$ . Therefore, the identity in Equation 35 leads us to:

$$
\forall h \in \mathcal { H } , \quad \operatorname* { l i m } _ { d \to \infty } E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * } W h = h .\tag{36}
$$

We can then conclude that $E _ { d } \mathbf { W } _ { d } ^ { - 1 } E _ { d } ^ { * }$ converges to $W ^ { - 1 }$ in SOT, which holds over all of H for the inverse of W is bounded. Hence, it almost surely holds that:

$$
\operatorname* { l i m } _ { d \to \infty } \eta _ { d } ^ { \mathsf { T } } ( \Sigma _ { d } + \mathbf { V } _ { d } ) ^ { - 1 } \eta _ { d } = \langle \eta , W ^ { - 1 } \eta \rangle = \langle \eta , ( \Sigma + U ) ^ { - 1 } \eta \rangle .\tag{37}
$$

Applying the limits in Equation 33 and 37 to Equation 29, we obtain:

$$
\begin{array} { r l } & { 1 \geq \mathbb { E } [ \underset { d  \infty } { \operatorname* { l i m } } \sqrt { \frac { \operatorname* { d e t } \mathbf { V } _ { d } } { \operatorname* { d e t } ( \Sigma _ { d } + \mathbf { V } _ { d } ) } } \exp ( \frac { 1 } { 2 } \eta _ { d } ^ { \mathsf { T } } ( \Sigma _ { d } + { \mathbf { V } _ { d } } ) ^ { - 1 } \eta _ { d } ) ] } \\ & { = \mathbb { E } [ \sqrt { \frac { 1 } { \operatorname* { d e t } ( I + U ^ { - 1 } \Sigma ) } } \exp ( \frac { 1 } { 2 } \langle \eta , ( \Sigma + U ) ^ { - 1 } \eta \rangle ) ] , } \end{array}\tag{38}
$$

where the inequality holds by Fatou’s lemma (Durrett, 2019), given the non-negativity of the integrand inside the expectation, which concludes the proof. □

Corollary B.2 (Extension to general regularizers). The result in Theorem B.1 continues to hold ifwe let U be any boundedly invertible positive-definite operator on H.

Proof. For this proof, we apply Kuroda’s (1958) result on the Weyl-von Neumann theorem (Lemma A.2) to construct a sequence of bounded self-adjoint operators $U _ { n }$ with pure point spectrum, such that $U _ { n }$ converges to U as $n  \infty$ and show that Theorem B.1 holds for the limit. Indeed, by Lemma A.2, for any given $\epsilon > 0$ and $1 < p < \infty$ , we can construct a diagonalizable operator $U + A _ { \epsilon }$ such that $\| A _ { \epsilon } \| _ { p } < \epsilon .$ Consequently, we can build a sequence $\{ U _ { n } \} _ { n = 1 } ^ { \infty }$ such that $\bar { U _ { n } } : = U + A _ { \epsilon _ { n } } .$ with $\epsilon _ { n }  0$ as $n \to \infty$ , which converges to U in operator norm as:

$$
\begin{array} { r } { \| U _ { n } - U \| _ { \mathrm { o p } } \le \| U _ { n } - U \| _ { p } \le \| A _ { \epsilon _ { n } } \| _ { p } \le \epsilon _ { n } \xrightarrow { n \to \infty } 0 . } \end{array}\tag{39}
$$

Moreover, Lemma A.2 ensures that $U _ { n }$ is bounded, self-adjoint and has a pure point spectrum, so that it admits an orthonormal eigenbasis, $\mathrm { i . e . , } U _ { n }$ is diagonalizable. To ensure that $U _ { n }$ remains positive definite, knowing that bounded invertibility of a positive-definite U guarantees that $U \succeq a _ { 0 } I$ , for some $a _ { 0 } > 0$ , we can choose $\epsilon _ { n }$ such that $\begin{array} { r } { 0 < \epsilon _ { n } < a _ { 0 } ( \mathrm { e . g . } , \epsilon _ { n } : = \frac { a _ { 0 } } { 2 n } ) } \end{array}$ , for all $n \in \mathbb { N } ,$ , so that:

$$
U _ { n } = U + A _ { \epsilon _ { n } } \succeq ( a _ { 0 } - \epsilon _ { n } ) I \succ 0 \implies \| U _ { n } ^ { - 1 } \| _ { \mathrm { o p } } \leq \frac { 1 } { a _ { 0 } - \epsilon _ { n } } < \infty , \quad \forall n \in \mathbb { N } .\tag{40}
$$

Applying Theorem B.1, we then have that:

$$
\forall n \in \mathbb { N } , \quad \mathbb { E } \left[ \sqrt { \frac { 1 } { \operatorname* { d e t } ( I + U _ { n } ^ { - 1 } \Sigma ) } } \exp \left( \frac { 1 } { 2 } \langle \eta , ( \Sigma + U _ { n } ) ^ { - 1 } \eta \rangle \right) \right] \leq 1 .\tag{41}
$$

We now need to verify the limits of the determinant and the quadratic form. Firstly, the determinant det $\left( I + U _ { n } ^ { - 1 } \Sigma \right)$ converges to de $\ : ( I + U ^ { - 1 } \Sigma ) \ :$ as $U _ { n } ^ { - 1 } \Sigma$ converges to $U ^ { - 1 } \Sigma$ in trace norm (Simon, 1977, Thm. 3.5). Indeed,

$$
\begin{array} { r l } & { \| U ^ { - 1 } \Sigma - U _ { n } ^ { - 1 } \Sigma \| _ { 1 } = \| ( U ^ { - 1 } - U _ { n } ^ { - 1 } ) \Sigma \| _ { 1 } } \\ & { \qquad = \| U ^ { - 1 } ( U _ { n } - U ) U _ { n } ^ { - 1 } \Sigma \| _ { 1 } } \\ & { \qquad \leq \| U ^ { - 1 } ( U _ { n } - U ) U _ { n } ^ { - 1 } \| _ { \mathrm { o p } } \| \Sigma \| _ { 1 } } \\ & { \qquad \leq \| U ^ { - 1 } \| _ { \mathrm { o p } } \| U _ { n } - U \| _ { \mathrm { o p } } \| U _ { n } ^ { - 1 } \| _ { \mathrm { o p } } \| \Sigma \| _ { 1 } } \\ & { \qquad \leq \epsilon _ { n } \| U ^ { - 1 } \| _ { \mathrm { o p } } \| U _ { n } ^ { - 1 } \| _ { \mathrm { o p } } \| \Sigma \| _ { 1 } \xrightarrow { n \to \infty } 0 , } \end{array}\tag{42}
$$

using the fact that $\| A B \| _ { 1 } \le \| A \| _ { \mathrm { o p } } \| B \| _ { 1 }$ <sub>1</sub> (see, e.g., Simon, 1977, Eq. 1.9). Thus,

$$
\operatorname * { l i m } _ { n  \infty } \operatorname * { d e t } ( I + U _ { n } ^ { - 1 } \Sigma ) = \operatorname * { d e t } ( I + U ^ { - 1 } \Sigma ) ,\tag{43}
$$

which holds almost surely. Secondly, the quadratic term is such that:

$$
\begin{array} { r l } { | \langle \eta , ( \Sigma + U ) ^ { - 1 } \eta \rangle - \langle \eta , ( \Sigma + U _ { n } ) ^ { - 1 } \eta \rangle | = | \langle \eta , ( ( \Sigma + U ) ^ { - 1 } - ( \Sigma + U _ { n } ) ^ { - 1 } ) \eta \rangle | } & { } \\ & { = | \langle \eta , ( \Sigma + U ) ^ { - 1 } ( \Sigma + U _ { n } - ( \Sigma + U ) ) ( \Sigma + U _ { n } ) ^ { - 1 } \eta \rangle | } \\ & { \leq \| \eta \| \| ( \Sigma + U ) ^ { - 1 } ( U _ { n } - U ) ( \Sigma + U _ { n } ) ^ { - 1 } \eta \| } \\ & { \leq \| \eta \| ^ { 2 } \| ( \Sigma + U ) ^ { - 1 } \| _ { \mathrm { o p } } \| U _ { n } - U \| _ { \mathrm { o p } } \| ( \Sigma + U _ { n } ) ^ { - 1 } \| _ { \mathrm { o p } } } \\ & { \leq \| \eta \| ^ { 2 } \| U ^ { - 1 } \| _ { \mathrm { o p } } \| U _ { n } - U \| _ { \mathrm { o p } } \| U _ { n } ^ { - 1 } \| _ { \mathrm { o p } } } \\ & { \leq \frac { \epsilon _ { n } } { ( a _ { 0 } - \epsilon _ { n } ) ^ { 2 } } \| \eta \| ^ { 2 } \xrightarrow { n \to \infty } 0 , } \end{array}\tag{44}
$$

where we applied Cauchy-Schwarz to derive the first inequality and then the observation that $\| ( \Sigma + U ) ^ { - \hat { 1 } } \| _ { \mathrm { o p } } ^ { } \leq \| U ^ { - 1 } \| _ { \mathrm { o p } } ^ { }$ , as $\Sigma \succeq 0$ , and likewise for $\| ( \ b { \Sigma } + \ b { U _ { n } } ) ^ { - 1 } \| _ { \mathrm { o p } }$ . Therefore, the following almost surely holds:

$$
\langle \eta , ( \Sigma + U _ { n } ) ^ { - 1 } \eta \rangle \xrightarrow { n \to \infty } \langle \eta , ( \Sigma + U ) ^ { - 1 } \eta \rangle .\tag{45}
$$

Combining Equation 43 and 45 with 41, the main result then arises by an application of Fatou’s lemma (Durrett, 2019). □

Lemma B.3 (Loss difference under operator-valued observations). Let H be a real Hilbert space and, for each $i \in \{ 1 , \ldots , t \}$ , let $\mathrm { M } _ { i } : \mathcal { V } _ { i }  \mathcal { H }$ be bounded and linear. Suppose that $\ell _ { i } ( \cdot , y _ { i } )$ is twice Fréchet differentiable and α-strongly convexfor some $\alpha > 0$ , and that $R _ { t } : \mathcal { H } $ R is Fréchet differentiable and $\lambda _ { t }$ -strongly convex for some $\lambda _ { t } > 0$ . Define $\begin{array} { r } { L _ { t } ( F ) : = \sum _ { i = 1 } ^ { t } \ell _ { i } ( \mathrm { M } _ { i } ^ { * } ( F ) , y _ { i } ) + } \end{array}$ $R _ { t } ( F )$ , and let $\widehat { F } _ { t }$ minimize $L _ { t }$ over H. Then,for every $F \in { \mathcal { H } }$

$$
\frac 1 2 \| \boldsymbol { F } - \widehat { \boldsymbol { F } } _ { t } \| _ { \boldsymbol { H } _ { t } } ^ { 2 } \leq L _ { t } ( \boldsymbol { F } ) - L _ { t } ( \widehat { \boldsymbol { F } } _ { t } ) \leq \frac { 1 } { 2 } \| \nabla L _ { t } ( \boldsymbol { F } ) \| _ { H _ { t } ^ { - 1 } } ^ { 2 } ,\tag{46}
$$

where $\begin{array} { r } { H _ { t } : = \lambda _ { t } I + \alpha \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } } \end{array}$

Proof. For any $F , F ^ { \prime } \in \mathcal { H } .$ , the α-strong convexity of $\ell _ { i } ( \cdot , y _ { i } )$ gives

$$
\begin{array} { r l } & { \ell _ { i } ( \mathrm { M } _ { i } ^ { * } ( F ^ { \prime } ) , y _ { i } ) \geq \ell _ { i } ( \mathrm { M } _ { i } ^ { * } ( F ) , y _ { i } ) + \langle \nabla _ { 1 } \ell _ { i } ( \mathrm { M } _ { i } ^ { * } ( F ) , y _ { i } ) , \mathrm { M } _ { i } ^ { * } ( F ^ { \prime } - F ) \rangle _ { \mathcal { V } _ { i } } } \\ & { \qquad + \displaystyle \frac { \alpha } { 2 } \| \mathrm { M } _ { i } ^ { * } ( F ^ { \prime } - F ) \| _ { \mathcal { V } _ { i } } ^ { 2 } . } \end{array}\tag{47}
$$

By the definition of the adjoint, the inner-product term in (47) equals $\langle \mathrm { M } _ { i } \nabla _ { 1 } \ell _ { i } ( \mathrm { M } _ { i } ^ { * } ( F ) , y _ { i } ) , F ^ { \prime } - F \rangle _ { \mathcal { H } }$ while its final term equals $\begin{array} { r } { \frac { \alpha } { 2 } \| \dot { F } ^ { \prime } - F \| _ { \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } } ^ { 2 } } \end{array}$ . Combining these inequalities with the λ -strong convexity of $R _ { t }$ yields

$$
L _ { t } ( F ^ { \prime } ) \geq L _ { t } ( F ) + \langle \nabla L _ { t } ( F ) , F ^ { \prime } - F \rangle _ { \mathcal { H } } + \frac { 1 } { 2 } \| F ^ { \prime } - F \| _ { H _ { t } } ^ { 2 } .\tag{48}
$$

Taking $F = \widehat { F } _ { t }$ in (48) and using the first-order optimality condition $\nabla L _ { t } ( \widehat { F } _ { t } ) = 0$ gives

$$
L _ { t } ( F ^ { \prime } ) - L _ { t } ( \widehat { F } _ { t } ) \geq \frac { 1 } { 2 } \| F ^ { \prime } - \widehat { F } _ { t } \| _ { H _ { t } } ^ { 2 } ,\tag{49}
$$

which proves the first inequality in (46).

For the second inequality, fix $F \in { \mathcal { H } }$ and take the infimum over $F ^ { \prime } \in \mathcal { H }$ in (48). Writing $\Delta : = F ^ { \prime } { - } F$ and using the optimality of $\widehat { F } _ { t }$ gives

$$
\begin{array} { r l } & { L _ { t } ( \widehat { F } _ { t } ) \geq L _ { t } ( F ) + \underset { \Delta \in \mathcal { H } } { \operatorname* { i n f } } \left\{ \langle \nabla L _ { t } ( F ) , \Delta \rangle _ { \mathcal { H } } + \frac { 1 } { 2 } \| \Delta \| _ { H _ { t } } ^ { 2 } \right\} } \\ & { \quad \quad \quad = L _ { t } ( F ) - \frac { 1 } { 2 } \| \nabla L _ { t } ( F ) \| _ { H _ { t } ^ { - 1 } } ^ { 2 } . } \end{array}\tag{50}
$$

Indeed, $H _ { t } \ \succeq \lambda _ { t } I$ is boundedly invertible, and the infimum is attained at $\Delta = - H _ { t } ^ { - 1 } \nabla L _ { t } ( F )$ Rearranging completes the proof. □

## C Proofs of main theorems

This section presents our proofs for the theoretical results presented in the main paper.

## C.1 Proof of Theorem 3.1

Proof. The proof follows the standard stopping-time argument for self-normalized concentration (e.g., Abbasi-Yadkori, 2012; Chugg and Ramdas, 2025). For each $h \in \mathcal H$ , define $\rho _ { 0 } ^ { h } : = 1$ and

$$
\rho _ { t } ^ { h } : = \exp \mathopen { } \mathclose \bgroup \left( \left. h , \sum _ { i = 1 } ^ { t } \eta _ { i } \aftergroup \egroup \right. _ { \mathcal { H } } - \frac { 1 } { 2 } \langle h , V _ { t } h \rangle _ { \mathcal { H } } \aftergroup \egroup \right) , \qquad t \in  { \mathbb { N } } .
$$

Since $V _ { t } = V _ { t - 1 } + \Sigma _ { t }$ , conditional sub-Gaussianity gives

$$
\begin{array} { r l } & { \mathbb { E } \big [ \rho _ { t } ^ { h } \ \big | \ \mathfrak { F } _ { t - 1 } \big ] = \rho _ { t - 1 } ^ { h } \mathbb { E } \bigg [ \exp \bigg ( \langle h , \eta _ { t } \rangle _ { \mathcal { H } } - \frac { 1 } { 2 } \langle h , \Sigma _ { t } h \rangle _ { \mathcal { H } } \bigg ) \ \bigg | \ \mathfrak { F } _ { t - 1 } \bigg ] } \\ & { \qquad \leq \rho _ { t - 1 } ^ { h } . } \end{array}
$$

Thus, $\{ \rho _ { t } ^ { h } \} _ { t \ge 0 }$ is a nonnegative supermartingale for every $h \in \mathcal H$

For $t \in \mathbb { N } ,$ , let $\mathcal { A } _ { t } ( \delta )$ be the event

$$
\left\| \sum _ { i = 1 } ^ { t } \eta _ { i } \right\| _ { ( U + V _ { t } ) ^ { - 1 } } ^ { 2 } > 2 \log \left( \frac { \operatorname * { d e t } \left( I + U ^ { - 1 / 2 } V _ { t } U ^ { - 1 / 2 } \right) ^ { 1 / 2 } } \delta \right) ,
$$

and define the first crossing time $\tau : = \operatorname* { i n f } \{ t \geq 1 : \mathcal { A } _ { t } ( \delta )$ occurs}, with inf $\varnothing : = \infty$ . This is a stopping time because $\mathcal { A } _ { t } ( \delta ) \in \mathfrak { F } _ { t }$ . Set $\tau _ { t } : = \operatorname* { m i n } \{ \tau , t \}$

The stopped process $\{ \rho _ { \tau _ { t } } ^ { h } \} _ { t \geq 0 }$ is also a nonnegative supermartingale. Therefore, for every $h \in \mathcal H$

$$
\mathbb { E } \big [ \rho _ { \tau _ { t } } ^ { h } \big ] \leq \mathbb { E } \big [ \rho _ { 0 } ^ { h } \big ] = 1 , \qquad t \in \mathbb { N } .
$$

Equivalently, the stopped random vector $\textstyle \sum _ { i = 1 } ^ { \tau _ { t } } \eta _ { i }$ and covariance proxy $V _ { \tau _ { t } }$ satisfy the hypothesis of Corollary B.2. Applying that result yields

$$
\mathbb { E } \left[ \frac { \exp \Big ( \frac { 1 } { 2 } \big \| \sum _ { i = 1 } ^ { \tau _ { t } } \eta _ { i } \big \| _ { ( U + V _ { \tau _ { t } } ) ^ { - 1 } } ^ { 2 } \Big ) } { \operatorname* { d e t } \big ( I + U ^ { - 1 / 2 } V _ { \tau _ { t } } U ^ { - 1 / 2 } \big ) ^ { 1 / 2 } } \right] \leq 1 .
$$

Hence, Markov’s inequality implies

$$
\mathbb { P } \left[ \left. \sum _ { i = 1 } ^ { \tau _ { t } } \eta _ { i } \right. _ { ( U + V _ { \tau _ { t } } ) ^ { - 1 } } ^ { 2 } > 2 \log \left( \frac { \operatorname* { d e t } \left( I + U ^ { - 1 / 2 } V _ { \tau _ { t } } U ^ { - 1 / 2 } \right) ^ { 1 / 2 } } { \delta } \right) \right] \leq \delta .
$$

By the definition of the first crossing time, $\tau \leq t$ if and only if the claimed inequality is violated at time $\tau _ { t } .$ Consequently, the preceding bound gives $\mathbb { P } \left[ \tau \leq t \right] \leq \delta$ for every $t \in \mathbb { N }$ . Since the events $\{ \tau \leq t \}$ increase to $\{ \tau < \infty \}$

$$
\mathbb { P } [ \exists t \in \mathbb { N } : A _ { t } ( \delta ) ] = \mathbb { P } [ \tau < \infty ] = \operatorname* { l i m } _ { t  \infty } \mathbb { P } [ \tau \leq t ] \leq \delta .
$$

Taking complements proves that the claimed inequality holds simultaneously for every $t \in \mathbb N$ with probability at least $1 - \delta$ □

## C.2 Proof of Theorem 4.1

Proof. We start with the derivation of a closed-form expression for the least-squares estimator and then proceed to analyse its approximation error. Since $\mathrm { M } _ { t } : \mathcal { V } _ { t } \to \mathcal { F } \subset \mathcal { H } .$ , its Banach adjoint restricted to H coincides with its Hilbert adjoint. Indeed, for any $F \in { \mathcal { H } }$ and $v \in \mathcal { V } _ { t }$

$$
\begin{array} { r } { \langle \mathrm { M } _ { t } ^ { * } ( F ) , v \rangle = \langle F , \mathrm { M } _ { t } ( v ) \rangle = \langle F , \mathrm { M } _ { t } ( v ) \rangle . } \end{array}
$$

Differentiating the least-squares objective over $\mathcal { H }$ therefore gives $\begin{array} { r } { \nabla L _ { t } ( F ) = 2 C _ { t } F - 2 \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } y _ { i } } \end{array}$ Since $C _ { t } \succeq \bar { \lambda I }$ , the minimizer is unique and satisfies:

$$
\widehat { F } _ { t } = C _ { t } ^ { - 1 } \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } y _ { i } , \qquad t \geq 1 .\tag{51}
$$

We first note that $C _ { t } ^ { - 1 } ( \Phi ) \ \in \mathcal { F }$ for every $\Phi ~ \in ~ { \mathcal { F } }$ . Let $\mathrm { M } _ { 1 : t } : \mathcal { V } _ { 1 : t } \to \mathcal { H }$ be defined by $\begin{array} { r } { \mathrm { M } _ { 1 : t } ( v _ { 1 } , \dots , v _ { t } ) : = \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } v _ { i } . } \end{array}$ , so that $C _ { t } = \lambda I + \mathrm { M } _ { 1 : t } \mathrm { M } _ { 1 : t } ^ { * }$ . By Woodbury’s identity,

$$
C _ { t } ^ { - 1 } = \lambda ^ { - 1 } I - \lambda ^ { - 1 } \mathrm { M } _ { 1 : t } ( \lambda I + \mathrm { M } _ { 1 : t } ^ { \ast } \mathrm { M } _ { 1 : t } ) ^ { - 1 } \mathrm { M } _ { 1 : t } ^ { \ast } .
$$

Since $\mathrm { M } _ { 1 : t }$ maps $\mathcal { D } _ { 1 : t }$ into ${ \mathcal F }$ , both terms on the right map $\mathcal { F }$ into $\mathcal { F }$ . Hence $C _ { t } ^ { - 1 } ( \Phi ) \in \mathcal { F }$

Substituting $y _ { i } = \mathrm { M } _ { i } ^ { * } ( F _ { \star } ) + \xi _ { i }$ <sub>i</sub> into (51), define $\begin{array} { r } { S _ { t } : = \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } } \end{array}$ <sub>i</sub>. For any $\Phi \in { \mathcal { F } }$ , since $C _ { t } ^ { - 1 } ( \Phi ) \in$ ${ \mathcal F } ,$ self-adjointness of $C _ { t } ^ { - 1 }$ on $\mathcal { H }$ yields:

$$
\langle F _ { \star } - \widehat F _ { t } , \Phi \rangle = \langle F _ { \star } , \Phi \rangle - \sum _ { i = 1 } ^ { t } \langle \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { \ast } ( F _ { \star } ) , C _ { t } ^ { - 1 } ( \Phi ) \rangle - \langle S _ { t } , C _ { t } ^ { - 1 } ( \Phi ) \rangle\tag{52}
$$

$$
= \langle F _ { \star } , \Phi \rangle - \sum _ { i = 1 } ^ { t } \langle F _ { \star } , \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { \ast } C _ { t } ^ { - 1 } ( \Phi ) \rangle - \langle S _ { t } , C _ { t } ^ { - 1 } ( \Phi ) \rangle\tag{53}
$$

$$
= \langle F _ { \star } , \left( I - \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { \ast } C _ { t } ^ { - 1 } \right) ( \Phi ) \rangle - \langle S _ { t } , C _ { t } ^ { - 1 } ( \Phi ) \rangle\tag{54}
$$

$$
= \lambda \langle F _ { \star } , C _ { t } ^ { - 1 } ( \Phi ) \rangle - \langle S _ { t } , C _ { t } ^ { - 1 } ( \Phi ) \rangle ,\tag{55}
$$

where the second equality follows from the defining relation of $\mathrm { M } _ { i } ^ { * }$ , and the last follows from the definition $\begin{array} { r } { C _ { t } = \lambda I + \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } } \end{array}$ . Trace duality and Cauchy-Schwarz then lead us to:

$$
\forall t \in \mathbb { N } , \quad | \langle F _ { \star } - \widehat { F } _ { t } , \Phi \rangle | \leq \lambda \| F _ { \star } \| _ { \mathrm { o p } } \| C _ { t } ^ { - 1 } ( \Phi ) \| _ { 1 } + \| C _ { t } ^ { - 1 / 2 } ( \Phi ) \| _ { 2 } \left\| \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \right\| _ { C _ { t } ^ { - 1 } } , \quad \forall \Phi \in \mathcal { F } .\tag{56}
$$

Let $\eta _ { t } : = \mathrm { M } _ { t } \xi _ { t } \in \mathcal { H }$ . By conditional $\Sigma _ { t }$ -sub-Gaussianity, $\eta _ { t }$ is conditionally $\mathrm { M } _ { t } \Sigma _ { t } \mathrm { M } _ { t } ^ { * }$ -sub-Gaussian, which is trace class by assumption. Moreover, $V _ { t } \preceq \sigma _ { \xi } ^ { 2 } \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * }$ , and hence $\sigma _ { \xi } ^ { 2 } \lambda I + V _ { t } \preceq \sigma _ { \xi } ^ { 2 } C _ { t }$ Therefore,

$$
\left\| \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \right\| _ { C _ { t } ^ { - 1 } } ^ { 2 } \leq \sigma _ { \xi } ^ { 2 } \mathopen { } \mathclose \bgroup \left\| \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \aftergroup \egroup \right\| _ { ( \sigma _ { \xi } ^ { 2 } \lambda I + V _ { t } ) ^ { - 1 } } ^ { 2 } .
$$

Finally, applying Theorem 3.1 with $U = \sigma _ { \xi } ^ { 2 } \lambda I$ , we obtain the following:

$$
\forall t \in \mathbb { N } , \quad \left\| \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \right\| _ { C _ { t } ^ { - 1 } } \leq \sqrt { 2 \sigma _ { \xi } ^ { 2 } \log \left( \frac { \operatorname* { d e t } ( I + \lambda ^ { - 1 } \sigma _ { \xi } ^ { - 2 } V _ { t } ) ^ { 1 / 2 } } { \delta } \right) } ,\tag{57}
$$

which holds with probability at least $1 - \delta .$ . Combining (56) and (57) proves (4). Since the highprobability event in (57) does not depend on Φ, the result holds simultaneously for every $\Phi \in { \mathcal { F } }$ □

## C.3 Proof of Corollary 4.2

Corollary C.1 (Extended restatement of Corollary 4.2). Let the assumptions ofTheorem 4.1 hold and define $\begin{array} { r } { G _ { t } : = \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } } \end{array}$ . Suppose there exists afixed d-dimensional subspace $s \subset \mathcal { F }$ , constants $L , \kappa > 0$ , and $t _ { 0 } \in$ N such that, almost surely, Ran $\left( \mathrm { M } _ { t } \right) \subseteq S$ and $\| \mathrm { M } _ { t } \| _ { \mathrm { o p } } \leq L f o r$ every $t \in \mathbb { N } ,$ while:

$$
G _ { t } \succeq \kappa t P _ { S } , \qquad t \geq t _ { 0 } ,\tag{58}
$$

where $P _ { S }$ denotes the orthogonal projection onto $s$ in H. Let $\begin{array} { r } { C _ { S } : = \operatorname* { s u p } _ { \stackrel { \Phi \in S } { \Phi \not = 0 } } { \frac { \| \Phi \| _ { 1 } } { \| \Phi \| _ { 2 } } } < \infty } \end{array}$ . Then, for every $\delta \in ( 0 , 1 ]$ , with probability at least $1 - \delta ,$ , simultaneously for every $t \geq t _ { 0 }$ and $\Phi \in S$

$$
| \langle F _ { \star } - \widehat F _ { t } , \Phi \rangle | \leq \frac { \lambda C _ { S } \| F _ { \star } \| _ { \mathrm { o p } } \| \Phi \| _ { 2 } } { \lambda + \kappa t } + \frac { \sigma _ { \xi } \| \Phi \| _ { 2 } } { \sqrt { \lambda + \kappa t } } \sqrt { 2 \log \frac { 1 } { \delta } + d \log \biggl ( 1 + \frac { t L ^ { 2 } } { \lambda } \biggr ) } .\tag{59}
$$

Consequently, for fixed δ and $\Phi \in S _ { \mathrm { : } }$

$$
\left| \langle F _ { \star } - \widehat { F } _ { t } , \Phi \rangle \right| = O \left( \sqrt { \frac { \log t } { t } } \right) .
$$

Proof. Since $\mathrm { R a n } ( \mathrm { M } _ { i } ) \subseteq S$ for every i, the subspace $s$ is invariant under $G _ { t }$ , and hence also under $C _ { t } \stackrel { \cdot } { = } \lambda I + G _ { t }$ . By (58), for every $\Phi \in S$ and $t \geq t _ { 0 }$

$$
\| C _ { t } ^ { - 1 / 2 } ( \Phi ) \| _ { 2 } \leq \frac { \| \Phi \| _ { 2 } } { \sqrt { \lambda + \kappa t } } .
$$

Moreover, $C _ { t } ^ { - 1 } ( \Phi ) \in \mathcal { S }$ , so finite-dimensional equivalence of the trace and Hilbert-Schmidt norms on S gives

$$
\| C _ { t } ^ { - 1 } ( \Phi ) \| _ { 1 } \le C _ { S } \| C _ { t } ^ { - 1 } ( \Phi ) \| _ { 2 } \le \frac { C _ { S } \| \Phi \| _ { 2 } } { \lambda + \kappa t } .
$$

Next, since $\Sigma _ { i } \preceq \sigma _ { \xi } ^ { 2 } I ,$

$$
\sigma _ { \xi } ^ { - 2 } V _ { t } \preceq G _ { t } .
$$

Furthermore, $G _ { t }$ has rank at most d and

$$
\| G _ { t } \| _ { \mathrm { o p } } \leq \sum _ { i = 1 } ^ { t } \| \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } \| _ { \mathrm { o p } } \leq t L ^ { 2 } .
$$

Hence, by monotonicity of the determinant over positive-semidefinite finite-rank operators,

$$
\operatorname* { d e t } ( I + \lambda ^ { - 1 } \sigma _ { \xi } ^ { - 2 } V _ { t } ) \leq \operatorname* { d e t } ( I + \lambda ^ { - 1 } G _ { t } ) \leq { \biggl ( } 1 + { \frac { t L ^ { 2 } } { \lambda } } { \biggr ) } ^ { d } .
$$

Substituting these three bounds into (4) gives (59). For fixed $\delta , \Phi .$ , and problem constants, the regularization term is $O ( t ^ { - 1 } )$ , while the stochastic term is $O ( { \sqrt { \log t / t } } )$ , proving the final claim.

## C.3.1 Proof of Theorem 4.3

Proof. Let $\widetilde { F } _ { t }$ be the minimizer of $L _ { t }$ over the whole space H. Since $F _ { \star } = F ( \cdot , \theta _ { \star } )$ belongs to the parametric model class and $\widehat { \theta } _ { t }$ is a global solution of (5), we have that:

$$
\forall t \in \mathbb { N } , \quad L _ { t } \big ( F ( \cdot , \widehat { \theta } _ { t } ) \big ) \leq L _ { t } ( F _ { \star } ) .
$$

Applying Lemma B.3 first to $F ( \cdot , \widehat { \theta _ { t } } )$ and then to $F _ { \star }$ leads us to:

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { 2 } \| F ( \cdot , \widehat { \theta _ { t } } ) - \widetilde { F } _ { t } \| _ { H _ { t } } ^ { 2 } \leq L _ { t } \big ( F ( \cdot , \widehat { \theta } _ { t } ) \big ) - L _ { t } ( \widetilde { F } _ { t } ) } \\ & { \qquad \leq L _ { t } ( F _ { \star } ) - L _ { t } ( \widetilde { F } _ { t } ) } \\ & { \qquad \leq \displaystyle \frac { 1 } { 2 } \| \nabla L _ { t } ( F _ { \star } ) \| _ { H _ { t } ^ { - 1 } } ^ { 2 } , } \end{array}
$$

while the same lemma also yields:

$$
\frac 1 2 \| F _ { \star } - \widetilde F _ { t } \| _ { H _ { t } } ^ { 2 } \le \frac 1 2 \| \nabla L _ { t } ( F _ { \star } ) \| _ { H _ { t } ^ { - 1 } } ^ { 2 } .
$$

The triangle inequality therefore implies that:

$$
\forall t \in \mathbb { N } , \quad \| F ( \cdot , \widehat { \theta } _ { t } ) - F _ { \star } \| _ { H _ { t } } \leq 2 \| \nabla L _ { t } ( F _ { \star } ) \| _ { H _ { t } ^ { - 1 } } .\tag{60}
$$

By the Hilbert-space chain rule and the definition of $\xi _ { t }$

$$
\nabla L _ { t } ( F _ { \star } ) = \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } + \nabla R _ { t } ( F _ { \star } ) .
$$

For each $i \in \{ 1 , \ldots , t \} , \mathrm { M } _ { i } \xi _ { i }$ is conditionally $\mathrm { M } _ { i } \Sigma _ { i } \mathrm { M } _ { i } ^ { * }$ -sub-Gaussian. This covariance proxy is trace class because $\mathrm { M } _ { i }$ is Hilbert–Schmidt and $\Sigma _ { i }$ is bounded. Moreover, as $\Sigma _ { i } \preceq \sigma _ { \xi } ^ { 2 } I$

$$
V _ { t } = \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \Sigma _ { i } \mathrm { M } _ { i } ^ { * } \preceq \sigma _ { \xi } ^ { 2 } \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } .\tag{61}
$$

Since $\lambda _ { t } \geq \lambda _ { 0 } ,$ , it follows that:

$$
\frac { \sigma _ { \xi } ^ { 2 } \lambda _ { 0 } } { \alpha } I + V _ { t } \preceq \frac { \sigma _ { \xi } ^ { 2 } } { \alpha } \left( \lambda _ { t } I + \alpha \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \mathrm { M } _ { i } ^ { * } \right) = \frac { \sigma _ { \xi } ^ { 2 } } { \alpha } H _ { t } .
$$

Inverting this operator inequality, we then have that:

$$
\left\| \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \right\| _ { H _ { t } ^ { - 1 } } \leq \frac { \sigma _ { \xi } } { \sqrt { \alpha } } \Bigg \| \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \Bigg \| _ { \left( \frac { \sigma _ { \xi } ^ { 2 } \lambda _ { 0 } } { \alpha } I + V _ { t } \right) ^ { - 1 } } .
$$

Applying Theorem 3.1 with regularization operator $( \sigma _ { \xi } ^ { 2 } \lambda _ { 0 } / \alpha ) I$ shows that, with probability at least $1 - \delta ,$ simultaneously for every $t \in \mathbb { N } .$

$$
\left. \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \right. _ { H _ { t } ^ { - 1 } } \leq \frac { \sigma _ { \xi } } { \sqrt { \alpha } } \sqrt { 2 \log \left( \frac { \operatorname* { d e t } \left( I + \frac { \alpha } { \sigma _ { \xi } ^ { 2 } \lambda _ { 0 } } V _ { t } \right) ^ { 1 / 2 } } { \delta } \right) } ,
$$

whose determinant can be further simplified by (61). Finally, the triangle inequality applied to the gradient decomposition gives

$$
\left. \nabla L _ { t } ( F _ { \star } ) \right. _ { H _ { t } ^ { - 1 } } \leq \left. \nabla R _ { t } ( F _ { \star } ) \right. _ { H _ { t } ^ { - 1 } } + \left. \sum _ { i = 1 } ^ { t } \mathrm { M } _ { i } \xi _ { i } \right. _ { H _ { t } ^ { - 1 } } .
$$

Substitution into (60) proves (6).