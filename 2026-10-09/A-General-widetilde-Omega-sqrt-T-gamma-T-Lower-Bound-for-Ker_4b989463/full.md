# A General $\widetilde \Omega ( \sqrt { T \gamma _ { T } } )$ Lower Bound for Kernel Bandits

Chenkai Ma National University of Singapore chenkai.ma@u.nus.edu

Jonathan Scarlett National University of Singapore scarlett@comp.nus.edu.sg

## Abstract

The kernel bandit problem consists of sequentially optimizing an unknown function with noisy feedback, where the function has bounded norm in a given Reproducing Kernel Hilbert Space (RKHS). A central quantity in the regret analysis of kernel bandits is the maximum information gain $\gamma _ { T } .$ . In particular, the best existing upper bounds scale as $\sqrt { T \gamma _ { T } }$ up to log factors, and nearly-matching lower bounds have been derived for specific kernels such as squared exponential and Mat´ern. However, lower bounds for general kernels are lacking, thus making it unclear in what generality the upper bounds are near-optimal. In this paper, we establish a general $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ minimax regret lower bound for non-constant continuous kernels on compact domains, establishing near-optimality (within log factors) in a very general sense. We show that the log factor appearing in this bound is unavoidable in general, but that it can be removed under certain conditions. Among other things, our findings imply that the minimax-optimal scaling is exactly $\Theta ( \sqrt { T \gamma _ { T } } )$ (i.e., within constant factors) for the Mat´ern-ν kernel with $\nu \in ( 0 , 2 )$ , γ-exponential kernel with $\gamma \in ( 0 , 2 )$ , and certain piecewise-polynomial kernels.

## 1 Introduction

The kernel bandit problem consists of sequentially optimizing an unknown function with noisy observations (Srinivas et al., 2010). We consider the frequentist setting, in which the function has bounded norm in a Reproducing Kernel Hilbert Space (RKHS). At each round, a learner queries the function at some point, and observes its noisy value. The objective is to minimize cumulative regret compared to the best action.

A central quantity in the regret analysis of kernel bandits is the maximum information gain $\gamma _ { T } .$ , which measures the largest information gain from T observations under the surrogate Gaussian process (GP) model (Srinivas et al., 2010; Chowdhury and Gopalan, 2017). The best existing upper bounds for general kernels scale as $\sqrt { T \gamma _ { T } }$ up to logarithmic factors (Valko et al., 2013; Camilleri et al., 2021; Salgia et al., 2021), and nearly-matching lower bounds have been established for the squared exponential and Mat´ern kernels (Scarlett et al., 2017; Cai and Scarlett, 2021), and extended to other isotropic kernels (Lee and Javidi, 2025). In this paper, we are interested in the following question: Does $\sqrt { T \gamma _ { T } }$ characterize the minimax regret $f o r$ general kernels beyond the specific ones that have near-matching lower bounds?

To address this question, it turns out to be helpful to connect the maximum information gain to the efective dimension, which measures the efective number of feature directions above a certain threshold (Alaoui and Mahoney, 2015; Derezinski et al., 2020). This quantity has frequently appeared in the study of upper bounds. In particular, (Camilleri et al., 2021) presents the RIPS algorithm based on regularized experimental design, whose regret is related to both the maximum information gain and the efective dimension. More recently, (Janz et al., 2026) provides sharp estimates for both quantities across broad kernel classes.

Our main contributions are outlined as follows:

• We prove an $\Omega ( \sqrt { T d _ { T } } )$ lower bound on finite domains, where $d _ { T }$ is the efective dimension of a regularized D-optimal design, and we extend this to compact domains by a finite approximation. We show that this implies $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ regret for every non-constant continuous positive semidefinite kernel on a compact subset of $\mathbb { R } ^ { d }$

• We show that the logarithmic factor is unavoidable in general, since on small finite domains, $\gamma _ { T } =$ $\Theta ( \log T )$ and the minimax regret is $\Theta ( { \sqrt { T } } )$ . Moreover, on $[ 0 , 1 ] ^ { d }$ , we derive a suficient condition that leads to $\Omega ( \sqrt { T \gamma _ { T } } )$ regret, which is satisfied if $\gamma _ { T } = \Theta ( T ^ { \alpha } )$ for some $\alpha > 0$ , and holds for the Mat´ern kernel among others.

• For comparison, we adapt an upper bound from (Camilleri et al., 2021) to attain scaling $O ( \sqrt { T \gamma \log T } )$ on $[ 0 , 1 ] ^ { d }$ under a suitable H¨older condition, and sharpen it to $O ( \sqrt { T \gamma _ { T } } )$ in some kernel settings. The resulting bounds match at $\Theta ( \sqrt { T \gamma _ { T } } )$ for the Mat´ern-ν kernel with $\nu \in ( 0 , 2 )$ , γ-exponential kernel with $\gamma \in ( 0 , 2 )$ , and q-piecewise polynomial kernel with $q = 0 , d \geq 3 \mathrm { o r } q = 1$ . For other kernels or general settings that we consider, a gap of $\sqrt { \log T }$ or log T remains.

## 1.1 Setup

Let $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ be nonempty and finite, and let $k : \mathcal { X } \times \mathcal { X }  \mathbb { R }$ be a non-constant continuous positive-semidefinite kernel satisfying $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } k ( x , x ) \le 1 } \end{array}$ . Let $\mathcal { H } _ { k }$ be the RKHS induced by $k ,$ and write $\phi ( x ) : = k ( x , \cdot ) \in \mathcal { H } _ { k }$ Define $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$ , which is positive since k is non-constant. For $B > 0 .$ define $\mathcal { F } _ { k } ( B ) : = \{ g \in \mathcal { H } _ { k } : \| g \| _ { k } \le B \}$ . We are interested in optimizing a function $f \in { \mathcal { F } } _ { k } ( B )$

Let $T \in \mathbb { Z } _ { + }$ be the time horizon. At time $t \in [ T ]$ , a possibly randomized adaptive (non-anticipatory) policy π chooses $x _ { t } \in \mathcal { X }$ and observes $y _ { t } = f ( x _ { t } ) + \xi _ { t }$ , where $( \xi _ { t } ) _ { t \geq 1 }$ is sampled i.i.d. (independent of the randomness of the policy) from $N ( 0 , \sigma ^ { 2 } )$ for some $\sigma > 0$ . For this policy π, define the cumulative regret

$$
R _ { T } ( \pi , f ) : = \sum _ { t = 1 } ^ { T } \bigl ( \operatorname* { m a x } _ { x \in \mathcal X } f ( x ) - f ( x _ { t } ) \bigr ) .\tag{1}
$$

The minimax regret is then defined as

$$
R _ { T } ^ { * } : = \operatorname* { i n f } _ { \pi } { \operatorname* { s u p } _ { f \in { \mathcal { F } } _ { k } ( B ) } \mathbb { E } } [ R _ { T } ( \pi , f ) ] .\tag{2}
$$

Efective Dimension. Our initial lower bound will be stated in terms of the efective dimension (Camilleri et al., 2021), which is related to regularized optimal experimental design (Derezinski et al., 2020; Allen-Zhu et al., 2021). Let $E : = \operatorname { s p a n } \{ \phi ( x ) : x \in \mathcal { X } \}$ , which is a finite-dimensional subspace of $\mathcal { H } _ { k }$ since X is finite. Let $n = \dim ( E )$ , and $e _ { 1 } , \ldots , e _ { n }$ be an orthonormal basis of E. We write $I _ { E }$ , det<sub>E</sub>, and $\mathrm { t r } _ { E }$ for the identity map, determinant, and trace on $E ,$ respectively. For example, det $\boldsymbol { \mathbf { \ell } } _ { E } ( A ) = \operatorname* { d e t } ( [ \langle e _ { i } , A e _ { j } \rangle _ { k } ] _ { i , j = 1 } ^ { n } )$ , where the righthand side is an ordinary determinant of an $n \times n$ matrix. (If A acts entirely within $E _ { : }$ , then det $\ b { \ b { \mathscr { E } } } ( A ) = \operatorname* { d e t } ( A )$ and $\operatorname { t r } _ { E } ( A ) = \operatorname { t r } ( A ) . )$

Let $\Delta ( \mathcal { X } )$ denote the set of probability distributions on $\mathcal { X }$ . For $\nu \in \Delta ( \mathcal { X } )$ , define the uncentered second moment operator on $E$ by

$$
M _ { \nu } : = \mathbb { E } _ { X \sim \nu } [ \phi ( X ) \otimes \phi ( X ) ] = \sum _ { x \in \mathcal { X } } \nu ( x ) \phi ( x ) \otimes \phi ( x ) ,\tag{3}
$$

where $u \otimes v$ is the operator on $E$ given by $( u \otimes v ) w = u \langle v , w \rangle _ { k }$ . Note that for the linear kernel, $M _ { \nu } =$ $\textstyle \sum _ { x \in \mathcal { X } } \nu ( x ) x x ^ { T }$ , the uncentered second moment matrix. Define $\rho : = \sigma ^ { 2 } / ( B ^ { 2 } T ) > 0$ to be a regularization level, and consider a regularized D-optimal design

$$
\nu _ { T } \in \mathop { \mathrm { a r g } } _ { \nu \in \Delta ( \mathcal { X } ) } \log \operatorname* { d e t } _ { E } \bigl ( M _ { \nu } + \rho I _ { E } \bigr ) ,\tag{4}
$$

where a maximizer exists because $\Delta ( \mathcal { X } )$ is compact and the objective is continuous. We will show that every ν satisfying eq. (4) gives the same operator in eq. (3) (see Lemma 17), so we define $M _ { T } : = M _ { \nu _ { T } }$ . Then, the

efective dimension (at regularization level $\rho )$ is defined by<sup>1</sup>

$$
d _ { T } : = \mathrm { t r } _ { E } \big ( M _ { T } ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \big ) .\tag{5}
$$

Let $\lambda _ { 1 } , \ldots , \lambda _ { n }$ be the eigenvalues of $M _ { T }$ , all of which are nonnegative (see Lemma 9). Then $\begin{array} { r } { d _ { T } = \sum _ { j = 1 } ^ { n } \frac { \lambda _ { j } } { \lambda _ { j } + \rho } , } \end{array}$ and thus an eigendirection contributes roughly one when its eigenvalue is much larger than $\rho ,$ and contributes little when its eigenvalue is much smaller than $\rho .$

Maximum Information Gain. Define $\lambda : = \sigma ^ { 2 } / B ^ { 2 }$ , which is the noise variance when the function and noise are normalized by B. To compare with standard kernel bandits literature, we will also provide lower bounds stated in terms of the maximum information gain (Srinivas et al., 2010; Chowdhury and Gopalan, 2017), defined by

$$
\gamma _ { T } : = \frac { 1 } { 2 } \operatorname* { m a x } _ { x _ { 1 : T } \in \mathcal { X } ^ { T } } \log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ) ) ,\tag{6}
$$

where $K _ { T } ( x _ { 1 : T } ) = [ k ( x _ { s } , x _ { s ^ { \prime } } ) ] _ { s , s ^ { \prime } \in [ T ] } ,$

Other Notations. For self-adjoint operators $A , B$ on the same inner product space $V ,$ we write $A \preceq B$ to denote that $B - A$ is positive semidefinite, i.e., $\langle A v , v \rangle \leq \langle B v , v \rangle$ for every $v \in V$ . We write $B \succeq A$ to denote $A \preceq B$ . Similarly, we use $\succ , \prec$ to indicate positive definiteness. Assuming $k , B , \sigma$ are fixed, we use $O ( \cdot )$ and $\Omega ( \cdot )$ for asymptotic upper and lower bounds as $T \to \infty , \widetilde { O } ( \cdot ) , \widetilde { \Omega } ( \cdot )$ to additionally hide polylogarithmic factors, and $\Theta ( \cdot )$ to mean both $O ( \cdot )$ and $\Omega ( \cdot )$

## 1.2 Related Work

Upper Bounds for Kernel Bandits. (Srinivas et al., 2010) presents the GP-UCB algorithm, which connects the cumulative regret to $\gamma _ { T }$ . In the frequentist setting, its analysis gives $\widetilde { O } ( \gamma _ { T } \sqrt { T } )$ regret. Later, (Chowdhury and Gopalan, 2017) sharpens the analysis but ultimately attains the same $\widetilde { O } ( \gamma _ { T } \sqrt { T } )$ behavior. (Valko et al., 2013) is the first to attain $\widetilde { O } ( \sqrt { T \gamma _ { T } } )$ regret, using the SupKernelUCB algorithm. Subsequent approaches yielding $\widetilde { O } ( \sqrt { T \gamma _ { T } } )$ include experimental design (Camilleri et al., 2021), domain shrinking (Salgia et al., 2021), and batched elimination (Li and Scarlett, 2022). The latter shows that O(log log T) batches are suficient, and (Ma et al., 2026) refines this by improving the batch count and hidden ${ \widetilde { O } } ( \cdot )$ terms.

Most of these results are stated on finite domains, and can be extended to $[ 0 , 1 ] ^ { d }$ through suitable finite approximations. Although our paper mainly focuses on the lower bounds, for comparison we establish an $O ( \sqrt { T \gamma \log T } )$ upper bound based on (Camilleri et al., 2021), and sharpen it to $O ( \sqrt { T \gamma _ { T } } )$ in several kernel settings by further combining results from (Lee and Javidi, 2025; Janz et al., 2026). We also combine domain discretization with MOSS (Audibert and Bubeck (2009); Lattimore and Szepesv´ari (2020), Theorem 9.1) to obtain $O ( \sqrt { T \gamma _ { T } } )$ in some other kernel settings. We are not aware of any sharper upper bounds from earlier works.

Lower Bounds for Kernel Bandits. (Scarlett et al., 2017) establishes an $\Omega ( \sqrt { T ( \log T ) ^ { d / 2 } } )$ lower bound for the squared exponential (SE) kernel, and $\Omega ( T ^ { ( \nu + d ) / ( 2 \nu + d ) } )$ for the Mat´ern kernel by constructing a set of hard-to-distinguish bump functions. (Cai and Scarlett, 2021) uses an alternative change-of-measure proof to improve these bounds in terms of their dependence on the error probability. These lower bounds are kernel-specific, and derived directly without using information gain.

More recently, (Iwazaki, 2026) obtains a stronger $\Omega ( \sqrt { T ( \log T / \log \log T ) ^ { d } } )$ bound for the SE kernel on the d-dimensional sphere $\mathbb { S } ^ { d }$ . We will obtain the same scaling for the hypercube $[ 0 , 1 ] ^ { d }$ by combining our general $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ bound with the rate of $\gamma _ { T }$ given by (Janz et al., 2026). Thus, we will improve the SE kernel lower bound on $[ 0 , 1 ] ^ { d }$ compared to Scarlett et al. (2017), whereas our Mat´ern lower bound will turn out to be the same as theirs up to constant factors (see Table 1 below).

To our knowledge, sharp lower bounds have not been established for the rational quadratic kernel and one-dimensional Dirichlet kernel (among many others).

Maximum Information Gain. The maximum information gain $\gamma _ { T }$ has been studied extensively. (Srinivas et al., 2010) gives $\gamma _ { T } = { \widetilde { O } } ( T ^ { ( d ( d + 1 ) ) / ( 2 \nu + d ( d + 1 ) ) } )$ for the Mat´ern-ν kernel and $\gamma _ { T } = O ( ( \log T ) ^ { d + 1 } )$ for the squared exponential kernel. (Vakili et al., 2021) improves the former to $\widetilde { O } ( T ^ { d / ( 2 \nu + d ) } )$ based on the Mercer eigenvalue decay under the assumption of uniformly bounded Mercer eigenfunctions. (Iwazaki, 2026) develops a spherical-harmonic approach to obtain $O ( ( \log T ) ^ { d + 1 } ) / ( \log \log T ) ^ { d }$ for the squared exponential kernel on compact subsets of $\mathbb { R } ^ { d }$ . The results from (Scarlett et al., 2017; Valko et al., 2013) can also be combined to infer that $\gamma _ { T }$ has scaling at least $T ^ { d / ( 2 \dot { \nu } + d ) }$ (Mat´ern) or $( \log T ) ^ { d } \ ( \mathrm { S E } )$ up to dimension-independent log factors. Recently, (Janz et al., 2026) shows that $\gamma _ { T } ~ = ~ \Theta ( T ^ { d / ( 2 \nu + d ) } )$ for the Mat´ern-ν kernel, and $\gamma _ { T } = \Theta ( ( \log T ) ^ { d + 1 } / ( \log \log T ) ^ { d } )$ for the squared exponential kernel. The same work also connects $\gamma _ { T } ~ \mathrm { t o }$ an empirical efective dimension, which is diferent from our definition in $\mathrm { { e q . } \ ( 5 ) }$ allowing fractional design. The bounds on $\gamma _ { T }$ alone do not imply minimax regret lower bounds, but are helpful when we compare our lower bounds with kernel-specific upper bounds.

Smoothness and Function-Space Approaches. Kernel bandits are closely related to continuum-armed bandits over smooth function classes. (Singh, 2021) derives minimax lower bounds for Besov classes, while (Liu et al., 2021) develops local polynomial algorithms attaining ${ \widetilde { O } } ( T ^ { ( \alpha + d ) / ( 2 \alpha + d ) } )$ regret for H¨older smoothness $( \alpha > 1 )$ . (Lee and Javidi, 2025) studies six kernel families through spectral decay and links to H¨older and Besov smoothness. Its findings include kernel-specific regret lower bounds, such as $\widetilde \Omega ( T ^ { \left( \gamma + 2 d \right) / \left( 2 \gamma + 2 d \right) } )$ for the γ-exponential kernels and $\tilde { \Omega } ( { T } ^ { ( 2 q + 1 + 2 d ) / ( 4 q + 2 + \hat { 2 } d ) } )$ for the piecewise-polynomial kernels (for admissible kernel parameters). Our lower bounds directly use the efective dimension without requiring the identification of an RKHS with H¨older or Besov spaces.

Experimental Design and Efective Dimension. (Camilleri et al., 2021) analyzes kernel bandits using the RIPS algorithm and regularized experimental design. Theorem 3 therein bounds regret through the largest G-optimal design value over surviving finite action sets, and its Lemmas 3 and 15 relate this value to the efective dimension and regularized D-optimal design. We similarly use D-optimal design to define the efective dimension, and then use it to construct hard functions needed in the analysis of the lower bound. We also extend the upper bound in (Camilleri et al., 2021, Theorem 3) from finite domains to $[ 0 , 1 ] ^ { d }$ establishing an $O ( \sqrt { T \gamma \log T } )$ regret bound that holds under a H¨older condition.

## 2 The Finite Domain Setting

## 2.1 Statement of Results

We state the main results for finite domains as follows, and defer infinite domains to Section 3.

Theorem 1. Consider the setup in Section 1.1 where X is finite. Then,

$$
R _ { T } ^ { * } \ge \operatorname* { m i n } \big \{ 1 , \Delta _ { k } \big \} \cdot \frac { \sigma } { 1 6 } \sqrt { T d _ { T } } ,
$$

where $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$

Corollary 1. Consider the setup in Section 1.1 where X is finite. Then,

$$
R _ { T } ^ { * } \geq \operatorname* { m i n } \left\{ 1 , \Delta _ { k } \right\} \cdot \frac { \sqrt { 2 } \sigma } { 1 6 } \sqrt { \frac { T \gamma _ { T } } { 1 + \log ( 1 + B ^ { 2 } T / \sigma ^ { 2 } ) } } ,
$$

where $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$

## 2.2 Proofs

The main idea is to construct hard-to-distinguish functions in $\mathcal { F } _ { k } ( B )$ using $d _ { T }$ , the policy-independent efective number of feature directions at scale $\rho .$ Lemma 23 shows that the regularized D-optimal design ν<sub>T</sub> provides normalized feature vectors $u _ { x }$ (defined in eq. (7) below) that have bounded norms and a bounded second moment operator (see eqs. (81) to (84)). When $d _ { T } \geq 4$ (Lemma 1), we use these vectors to construct functions in $\mathcal { F } _ { k } ( B )$ , each maximized at its corresponding support point with value of order $\sigma \sqrt { d _ { T } / T }$ . We then consider a zero-function model where the learner’s actions are independent of the randomly chosen hard function. The bounds in Lemma 23 are used to lower bound regret in the zero-function model, and upper bound the KL divergence between this model and the hard-function model. Then we apply a change-ofmeasure argument to transfer the regret lower bound to the hard-function model. When $d _ { T } < 4$ (Lemma 24), a separate construction using two opposite functions based on maximally separated feature vectors gives the complementary lower bound. Combining the two cases proves Theorem 1, and relating $d _ { T }$ to $\gamma _ { T }$ (Lemma 25) yields Corollary 1.

We proceed to state and prove Lemma 1, which is the core part of the argument. The other claims above, along with their proofs, are given in Section C. We define the regularized operator $V _ { T } : = { { M } _ { T } } + \rho { { I } _ { E } }$ , and noting that $V _ { T } \succ 0$ (see Lemma 9) and $d _ { T } > 0$ (see Lemma 16), we define the following for every $x \in { \mathcal { X } } ;$

$$
u _ { x } : = d _ { T } ^ { - 1 / 2 } V _ { T } ^ { - 1 / 2 } \phi ( x ) \in \mathcal { H } _ { k } .\tag{7}
$$

These are normalized feature vectors, and their key properties are established in (81)–(84) (Lemma 23).

Lemma 1. Consider the setup in Section 1.1. If $d _ { T } \geq 4$ , then

$$
R _ { T } ^ { * } \ge \frac { \sigma } { 1 6 } \sqrt { T d _ { T } } .\tag{8}
$$

Proof. Step 1: Construct hard functions. For every $z \in \mathrm { s u p p } ( \nu _ { T } )$ , define

$$
f _ { z } : = \frac { B } { 4 } \sqrt { \frac { \rho } { d _ { T } } } V _ { T } ^ { - 1 } \phi ( z ) \in \mathcal { H } _ { k } .\tag{9}
$$

We first show that $f _ { z } \in \mathcal { F } _ { k } ( B )$ . Lemma 9 shows that $V _ { T } ^ { - 2 } \preceq \rho ^ { - 1 } V _ { T } ^ { - 1 }$ and $V _ { T } ^ { - 1 } , V _ { T } ^ { - 1 / 2 }$ are self-adjoint. Then

$$
\| f _ { z } \| _ { k } ^ { 2 } = \frac { B ^ { 2 } \rho } { 1 6 d _ { T } } \langle \phi ( z ) , V _ { T } ^ { - 2 } \phi ( z ) \rangle _ { k }\tag{10}
$$

$$
\leq \frac { B ^ { 2 } } { 1 6 d _ { T } } \langle \phi ( z ) , V _ { T } ^ { - 1 } \phi ( z ) \rangle _ { k }\tag{11}
$$

$$
= { \frac { B ^ { 2 } } { 1 6 } }\tag{12}
$$

$$
\leq B ^ { 2 } ,\tag{13}
$$

where eq. (12) follows from eqs. (7) and (82). We have thus shown $f _ { z } \in { \mathcal { F } } _ { k } ( B )$ . For any $x \in \mathcal { X }$ , the reproducing property and definition of $u _ { x } \ \mathrm { g i v e }$ that

$$
f _ { z } ( x ) = \frac { B } { 4 } \sqrt { \rho d _ { T } } \langle u _ { z } , u _ { x } \rangle _ { k } ,\tag{14}
$$

which combined with eqs. (81) and (82) gives

$$
f _ { z } ( z ) = B \sqrt { \rho d _ { T } } / 4 , \quad | f _ { z } ( x ) | \leq B \sqrt { \rho d _ { T } } / 4 .\tag{15}
$$

Step 2: Change of measure. We will apply a well-known change of measure result, stated in Lemma 22 in Section B. We first bound the relevant KL divergence. Fix any policy $\pi ,$ and let $H _ { t } : = ( X _ { 1 } , Y _ { 1 } , \ldots , X _ { t } , Y _ { t } )$

be the history until time t (the uppercase $X _ { t }$ and $Y _ { t }$ are random variables, whereas lowercase $x _ { t }$ and $y _ { t }$ are realizations). Let $P$ be the joint law on $( Z , H _ { T } )$ obtained by first drawing $Z \sim \nu _ { T }$ , then running π against $f _ { Z }$ . Let $Q$ be the joint law on $( Z , H _ { T } )$ when the policy is instead run on the zero function, and $Z \sim \nu _ { T }$ is drawn independently from the interaction. By the chain rule for KL divergence,

$$
\begin{array} { l } { { \cal D } _ { \mathrm { K L } } ( Q | | P ) } \\ { = { \cal D } _ { \mathrm { K L } } ( Q _ { Z } | | P _ { Z } ) + \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ { \cal D } _ { \mathrm { K L } } ( Q _ { X _ { t } | Z , H _ { t - 1 } } | | P _ { X _ { t } | Z , H _ { t - 1 } } ) ] } \end{array}
$$

$$
+ \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ D _ { \mathrm { K L } } ( Q _ { Y _ { t } | Z , H _ { t - 1 } , X _ { t } } \Vert P _ { Y _ { t } | Z , H _ { t - 1 } , X _ { t } } ) ]\tag{16}
$$

$$
= \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ D _ { \mathrm { K L } } ( Q _ { Y _ { t } | Z , H _ { t - 1 } , X _ { t } } \Vert P _ { Y _ { t } | Z , H _ { t - 1 } , X _ { t } } ) ]\tag{17}
$$

$$
\mathbf { \Psi } = \frac { 1 } { 2 \sigma ^ { 2 } } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ f _ { Z } ( X _ { t } ) ^ { 2 } ] ,\tag{18}
$$

where eq. (17) follows because $P _ { Z } = Q _ { Z }$ and the same policy is used for both $f _ { Z }$ and the zero function, and eq. (18) follows from the KL divergence formula between two Gaussians. Under $Q ,$ , the random variables $Z$ and $X _ { t }$ are independent, which implies $\begin{array} { r } { \mathbb { E } _ { Q } [ f _ { Z } ( X _ { t } ) ^ { 2 } ] = \mathbb { E } _ { Q } [ \mathbb { E } _ { Q } [ f _ { Z } ( X _ { t } ) ^ { 2 } | X _ { t } ] ] = \mathbb { E } _ { Q } [ \mathbb { E } _ { Z } [ f _ { Z } ( X _ { t } ) ^ { 2 } ] ] } \end{array}$ . Observe that, for any fixed $x \in \mathcal { X }$

$$
\mathbb { E } _ { Z } [ f _ { Z } ( x ) ^ { 2 } ] = \frac { B ^ { 2 } \rho d _ { T } } { 1 6 } \mathbb { E } _ { Z } [ | \langle u _ { Z } , u _ { x } \rangle _ { k } | ^ { 2 } ]\tag{19}
$$

$$
\mathbf { \Psi } = \frac { B ^ { 2 } \rho d _ { T } } { 1 6 } \langle u _ { x } , \mathbb { E } _ { Z } [ u _ { Z } \otimes u _ { Z } ] u _ { x } \rangle _ { k }\tag{20}
$$

$$
\leq \frac { B ^ { 2 } \rho } { 1 6 } \langle u _ { x } , u _ { x } \rangle _ { k }\tag{21}
$$

$$
\leq \frac { B ^ { 2 } \rho } { 1 6 } ,\tag{22}
$$

where eq. (19) follows from eq. (14), eq. (20) follows from the definition of $\otimes$ and linearity of expectation, eq. (21) follows from eq. (83), and eq. (22) follows from eq. (81). Combining with eq. (18) gives

$$
D _ { \mathrm { K L } } ( Q \| P ) \leq \frac { T B ^ { 2 } \rho } { 3 2 \sigma ^ { 2 } } = \frac { 1 } { 3 2 } .\tag{23}
$$

We now proceed to bound $\mathbb { E } _ { Q } [ R _ { T } ( \pi , f _ { Z } ) ]$ . For any $x \in \mathcal { X }$ , we have

$$
\mathbb { E } _ { Z } [ f _ { Z } ( x ) ] = \frac { B \sqrt { \rho d _ { T } } } { 4 } \langle \mathbb { E } _ { Z } [ u _ { Z } ] , u _ { x } \rangle _ { k }\tag{24}
$$

$$
\leq \frac { B \sqrt { \rho } } { 4 } \| u _ { x } \| _ { k }\tag{25}
$$

$$
\leq { \frac { B { \sqrt { \rho } } } { 4 } } ,\tag{26}
$$

where $\mathrm { e q } .$ . (24) follows from eq. (14), eq. (25) follows from eq. (84) and Cauchy-Schwarz, and eq. (26) follows from eq. (81). Recalling $\mathbb { E } _ { Q } [ f _ { Z } ( X _ { t } ) ] = \mathbb { E } _ { Q } [ \mathbb { E } _ { Z } [ f _ { Z } ( X _ { t } ) ] ]$ ], eqs. (15) and (26) then give

$$
\mathbb { E } _ { Q } [ R _ { T } ( \pi , f _ { Z } ) ] = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } \left[ \operatorname* { m a x } _ { x \in \mathcal { X } } f _ { Z } ( x ) - f _ { Z } ( X _ { t } ) \right]\tag{27}
$$

$$
\ge B T \biggl ( \frac { \sqrt { \rho d _ { T } } } { 4 } - \frac { \sqrt { \rho } } { 4 } \biggr ) \ge \frac { T B \sqrt { \rho d _ { T } } } { 8 } ,\tag{28}
$$

where the last step uses $d _ { T } \geq 4$ . Finally, eq. (15) implies $0 \le R _ { T } ( \pi , f _ { Z } ) \le B T \sqrt { \rho d _ { T } } / 2$ . Applying the standard change of measure result from Lemma 22 in Section B (total variation + Pinsker’s inequality) with $H = R _ { T } ( \pi , f _ { Z } ) , M = B T \sqrt { \rho d _ { T } } / 2$ , and eqs. (23) and (28) gives

$$
\mathbb { E } _ { P } [ R _ { T } ( \pi , f _ { Z } ) ] \ge \frac { T B \sqrt { \rho d _ { T } } } { 8 } - \frac { T B \sqrt { \rho d _ { T } } } { 2 } \cdot \frac { 1 } { 8 }\tag{29}
$$

$$
= \frac { \sigma } { 1 6 } \sqrt { T d _ { T } } ,\tag{30}
$$

which implies Lemma 1 since $\begin{array} { r } { \operatorname* { s u p } _ { f \in \mathcal { F } _ { k } ( B ) } \mathbb { E } [ R _ { T } ( \pi , f ) ] } \end{array}$ is at least E $: z \sim \nu _ { T } \mathbb { E } [ R _ { T } ( \pi , f _ { Z } ) ]$ ] and $\pi$ is arbitrary.

## 3 Extension to General Compact Domains

We extend Theorem 1 to general (possibly continuous) compact domains through a finite approximation of X . All proofs are given in the supplementary material.

## 3.1 Setup

We assume $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ is nonempty and compact, and let $k : \mathcal { X } \times \mathcal { X } \to \mathbb { R }$ be a non-constant, continuous, and positive semidefinite kernel satisfying $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } k ( x , x ) \le 1 } \end{array}$ . The rest of the setup is similar to Section 1.1, with some minor diferences described as follows. We reuse the definitions $\mathcal { H } _ { k } , \phi ( \boldsymbol { x } ) , B , \mathcal { F } _ { k } ( B )$ and again assume that $f \in { \mathcal { F } } _ { k } ( B )$ We also reuse the definition $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$ , which is welldefined for compact X (see Lemma 21). Since k is non-constant, it follows that $\Delta _ { k } \ > \ 0$ . The adaptive learning framework $T , \pi , x _ { t } , y _ { t } , \xi _ { t } , \sigma$ and regret definitions $R _ { T } ( \pi , f ) , R _ { T } ^ { * }$ remain the same; the max operation in $\begin{array} { r } { R _ { T } ( \pi , f ) : = \sum _ { t = 1 } ^ { T } ( \operatorname* { m a x } _ { x \in \mathcal { X } } f ( x ) - f ( x _ { t } ) ) } \end{array}$ is justified because $\mathcal { X }$ is compact and $f$ is continuous (see Lemma 20). We reuse $\lambda : = \sigma ^ { 2 } / B ^ { 2 }$ , and extend the definition of the maximum information gain in eq. (6): For $\mathcal A \subseteq \mathcal X$ that is nonempty and compact, we define the maximum information gain on $\mathcal { A }$ by

$$
\gamma _ { T } ( \mathcal { A } ) : = \frac { 1 } { 2 } \operatorname* { m a x } _ { x _ { 1 : T } \in \mathcal { A } ^ { T } } \log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ) ) ,\tag{31}
$$

where $K _ { T } ( x _ { 1 : T } ) \ = \ [ k ( x _ { s } , x _ { s ^ { \prime } } ) ] _ { s , s ^ { \prime } \in [ T ] }$ . The max operation in eq. (31) is justified because the mapping $\begin{array} { r } { x _ { 1 : T } \mapsto \frac { 1 } { 2 } \log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ) } \end{array}$ is continuous and $\mathcal { A } ^ { T }$ is compact. It follows that $\gamma _ { T } ( \mathcal { X } ) = \gamma _ { T }$ .

Finite Approximation Compared to Section 1.1, the main ingredient for handling compact X is a finite approximation of X shown in the following result.

Proposition 1. Consider the setup in Section 3.1 where X is compact. Then, there exists a nonempty finite $\tilde { \mathcal { X } } \subseteq \mathcal { X }$ that satisfies

$$
\operatorname* { m i n } _ { x ^ { \prime } \in \tilde { \mathcal { X } } } \lVert \phi ( x ) - \phi ( x ^ { \prime } ) \rVert _ { k } \leq \frac { \sigma } { 4 B \sqrt { T } } \qquad \forall x \in \mathcal { X } ,\tag{32}
$$

and $\gamma _ { T } ( { \tilde { \mathcal { X } } } ) = \gamma _ { T } ( { \mathcal { X } } )$

The $\tilde { \mathcal X }$ satisfying Proposition 1 is not unique, but for our purposes it sufices to fix any such $\tilde { \mathcal X }$ . With $\tilde { \mathcal X }$ defined, we reuse the notations $E , n , ( e _ { i } ) _ { i = 1 } ^ { n } , I _ { E }$ , det $_ { E } , \mathrm { t r } _ { E } , \Delta ( \mathcal { X } ) , \nu , M _ { \nu } , \otimes , \rho , \nu _ { T } , M _ { T } , d _ { T } , ( \lambda _ { i } ) _ { i = 1 } ^ { n }$ for the efective dimension from Section 1.1, with the understanding that every occurrence of X therein is replaced by X<sup>˜</sup>. We sometimes write $d _ { T } ( \tilde { \mathcal { X } } )$ to make the distinction from $d _ { T } ( \mathcal { X } )$ explicit.

## 3.2 Main Results

Theorem 2. Under the setup of Section 3.1 where X is compact, for every finite $\tilde { \mathcal X }$ satisfying Proposition $^ { 1 , }$

$$
R _ { T } ^ { * } \geq \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sigma } { 3 2 } \sqrt { T d _ { T } ( \tilde { \mathcal { X } } ) } ,
$$

where $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$

Although Theorem 2 depends on $\tilde { \mathcal { X } } _ { : }$ , we can remove such dependence by converting $d _ { T } ( \tilde { \mathcal { X } } )$ to $\gamma _ { T } ( \mathcal { X } )$ , and obtain the same scaling as Corollary 1.

Corollary 2. Under the setup of Section 3.1 where X is compact, we have

$$
R _ { T } ^ { * } \geq \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sqrt { 2 } \sigma } { 3 2 } \sqrt { \frac { T \gamma _ { T } ( \mathcal { X } ) } { 1 + \log ( 1 + B ^ { 2 } T / \sigma ^ { 2 } ) } } ,
$$

where $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } . } \end{array}$

The proofs are given in Appendix D, and are similar to those for finite $x ,$ except for Lemma 26, which extends eq. (81) in Lemma 23 to every $x \in \mathcal { X }$ with a slightly weaker bound. We also present a high-probability bound (Corollary 3) in Appendix D.1, which states that for every policy π, there exists $f \in { \mathcal { F } } _ { k } ( B )$ such that $R _ { T } ( \pi , f )$ satisfies the same lower bound as Corollary 2 with probability at least $1 / 4$

## 4 Refinements and Comparisons

To apply our lower bounds and compare them with existing results, we consider the setup of Section 3.1 with $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . We will mainly discuss the implications of the lower bound stated with the maximum information gain $\gamma _ { T }$ (Corollary 2) instead of the efective dimension $d _ { T }$ (Theorem 2). Following (Lee and Javidi, 2025), we consider six types of kernels, including the squared exponential kernel $k _ { \mathrm { S E } }$ , rational-quadratic kernel $k _ { \mathrm { R Q } , a }$ where $a > 0$ , Mat´ern kernel $k _ { \nu }$ where $\nu > 0$ , γ-exponential kernel $k _ { \gamma - \mathrm { E x p } }$ where $\gamma \in ( 0 , 2 )$ , piecewisepolynomial kernel $k _ { \mathrm { P P } , q }$ where $q \in \mathbb { Z } _ { \geq 0 }$ and $q \geq 1$ for $d = 1 , 2$ , and the one-dimensional Dirichlet kernel $k _ { \mathrm { D i r } , n }$ where $n \in \mathbb { Z } _ { + }$ . The details of these kernels are given in Section $\mathrm { A }$

The main findings of this section are summarized in Table 1. The second column presents sharp rates (in $\Theta ( \cdot ) )$ for $\gamma _ { T }$ , which follow by combining results from (Lee and Javidi, 2025; Janz et al., 2026), as discussed in Appendix E.1. The third column presents upper bounds of $R _ { T } ^ { * }$ , which are obtained from the RIPS algorithm (Camilleri et al. (2021), Theorem 3) and the MOSS algorithm (Audibert and Bubeck (2009); Lattimore and Szepesv´ari (2020), Theorem 9.1), and discussed in Section 4.2. The fourth column presents lower bounds of $R _ { T } ^ { * }$ based on Corollary 2 (for which we are not aware of rigorously established stronger lower bounds), and is discussed in Section 4.1. Table 1 shows that, for all kernels, the gap between bounds is at most $\sqrt { \log T }$ and that the bounds are matching for the Mat´ern kernel with $\nu \in ( 0 , 2 )$ , γ-exponential kernel with $\gamma \in ( 0 , 2 )$ and piecewise-polynomial kernel with $q = 0 , d \geq 3$ or $q = 1$ . As such, our Table 1 sharpens (Lee and Javidi, 2025, Table 1), where the regret bounds were expressed in the $\widetilde { O } ( \cdot ) , \widetilde { \Omega } ( \cdot )$ sense, and the lower bounds for the rational quadratic kernel and Dirichlet kernel were missing.

Remark 1. Another consequence of Corollary 2 is that any regret upper bound implies an information gain upper bound. Specifically, if $R _ { T } ^ { * } \leq U _ { T }$ , then $\gamma _ { T } = O ( U _ { T } ^ { 2 } \log ( T ) / T )$ . In Appendix F, we also present a refined upper bound of this kind that can remove the log factor. These results may be of independent interest $f o r$ kernels where $\gamma _ { T }$ is not yet established but $R _ { T } ^ { * }$ can be bounded by other means.

## 4.1 When can the Log Factor be Removed?

When X is fixed and compact, Corollary 2 gives $R _ { T } ^ { * } = \Omega \big ( \sqrt { T \gamma _ { T } / \log T } \big )$ . We first show that the log T term therein cannot always be removed, then discuss a condition on $\gamma _ { T }$ that removes this term, and finally apply this condition to the kernels in Table 1.

Lack of General $\Omega ( \sqrt { T \gamma _ { T } } )$ Lower Bound. We consider the setting in Section 1.1 with X being fixed and finite, and define $K : = | { \mathcal { X } } |$ . Since k is non-constant, $K \geq 2$ . This can be viewed as a multi-armed bandit problem with Gaussian noise and a constraint that $\| f \| _ { k } \le B$ . For every $x \in \mathcal { X }$ , the reproducing property and Cauchy-Schwarz show that $| f ( x ) | \leq | \langle f , \phi ( x ) \rangle _ { k } | \leq \| f \| _ { k } \| \phi ( x ) \| _ { k } \leq B$ . Then, applying the

Table 1: Rates of the maximum information gain $\gamma _ { T } ~ \mathrm { ( i n ~ } \Theta ( \cdot ) )$ and the bounds on minimax regret $R _ { T } ^ { * }$ for six kernels on $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . For all six kernels, the bounds either match or have a $\sqrt { \log T }$ gap. Lower bounds that are new or improved are highlighted in blue. The kernels are defined in Section A.
<table><tr><td>Kernel</td><td> $\mathrm { R a t e ~ o f ~ } \gamma _ { T }$ </td><td>Upper Bound of  $R _ { T } ^ { * }$ </td><td></td><td>Lower Bound of  $R _ { T } ^ { * }$ </td></tr><tr><td>Matérn</td><td> $T ^ { d / ( 2 \nu + d ) }$ </td><td> $\int O ( { \sqrt { T \gamma _ { T } } } ) ,$   $\begin{array} { r } { \boxed { O ( \sqrt { T \gamma _ { T } \log T } ) , \nu \geq 2 } } \end{array}$ </td><td> $0 < \nu < 2 ,$ </td><td> $\Omega ( \sqrt { T \gamma _ { T } } )$ </td></tr><tr><td>Squared exponential</td><td> $\underline { { ( \log T ) ^ { d + 1 } } }$   $\overbrace { ( \log \log T ) ^ { d } }$ </td><td> $O ( \sqrt { T \gamma _ { T } } )$ </td><td></td><td> $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ </td></tr><tr><td>Rational-quadratic</td><td> $( \log T ) ^ { d + 1 }$ </td><td> $O ( \sqrt { T \gamma _ { T } } )$ </td><td></td><td> $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ </td></tr><tr><td>γ-exponential</td><td> $\dot { T } ^ { d / ( \gamma + d ) }$ </td><td> $O ( \sqrt { T \gamma _ { T } } )$ </td><td></td><td> $\Omega ( \sqrt { T \gamma _ { T } } )$ </td></tr><tr><td>Piecewise-polynomial</td><td> $T ^ { d / ( 2 q + 1 + d ) }$ </td><td> $\int O ( { \sqrt { T \gamma _ { T } } } ) ,$   $\begin{array} { r } { \Big \lfloor O ( \sqrt { T \gamma _ { T } \log T } ) , \quad q \geq 2 } \end{array}$ </td><td> $q = 0 , d \geq 3 \mathrm { o r } q = 1$ </td><td> $\Omega ( \sqrt { T \gamma _ { T } } )$ </td></tr><tr><td>Dirichlet  $( d = 1 )$ </td><td> $\log T$ </td><td> $O ( \sqrt { T \gamma _ { T } } )$ </td><td></td><td> $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ </td></tr></table>

MOSS algorithm (Lattimore and Szepesv´ari (2020), Theorem 9.1) gives

$$
R _ { T } ^ { * } \leq 3 9 \sigma \sqrt { K T } + 2 B K = O ( \sqrt { T } ) .\tag{33}
$$

Fix any sequence $x _ { 1 } , \dotsc , x _ { T } \in \mathcal { X }$ . Since $| { \mathcal { X } } | = K$ , the Gram matrix $K _ { T } ( x _ { 1 : T } )$ satisfies rank $( K _ { T } ( x _ { 1 : T } ) ) \le$ dim span $\{ \phi ( x ) : x \in { \mathcal { X } } \} \leq K$ , and we list its nonzero eigenvalues and append zeros to obtain the K values $\lambda _ { 1 } , \ldots , \lambda _ { K }$ . Notice that $\begin{array} { r } { \sum _ { i = 1 } ^ { K } \lambda _ { i } = \mathrm { t r } ( K _ { T } ( x _ { 1 : T } ) ) = \sum _ { t = 1 } ^ { T } k ( x _ { t } , x _ { t } ) \le T } \end{array}$ . Then,

$$
\log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ) ) = \sum _ { j = 1 } ^ { K } \log ( 1 + \lambda _ { j } / \lambda ) \leq K \log \biggl ( 1 + \sum _ { j = 1 } ^ { K } \lambda _ { j } / ( K \lambda ) \biggr ) \leq K \log \bigl ( 1 + T / ( K \lambda ) \bigr ) ,\tag{34}
$$

where the first inequality applies Jensen’s inequality. Multiplying by $1 / 2$ and taking the maximum over $x _ { 1 : T } \in \mathcal { X } ^ { T }$ gives $\gamma _ { T } ( \mathcal { X } ) \ : = \ : O ( \log T )$ . On the other hand, Lemma 15 in Section B gives $\gamma _ { T } = \Omega ( \log T )$ and hence $\gamma _ { T } ( \mathcal { X } ) \ : = \ : \Theta ( \log T )$ . Lemma 24 shows that $R _ { T } ^ { * } = \Omega ( \sqrt { T } )$ , which combined with eq. (33) and $\gamma _ { T } = \Theta ( \log T )$ gives

$$
R _ { T } ^ { * } = \Theta ( \sqrt { T } ) = \Theta ( \sqrt { T \gamma _ { T } / \log T } ) .\tag{35}
$$

Consequently, an $\Omega ( \sqrt { T \gamma _ { T } } )$ lower bound is impossible in general, and the log factor is unavoidable.

A Suficient Condition for $\Omega ( \sqrt { T \gamma _ { T } } )$ Regret. We now regard $\gamma _ { T }$ as the function $\gamma _ { ( \cdot ) }$ evaluated at $T ,$ accordingly using $\gamma _ { m }$ below. We start by presenting a lower bound stated in terms of the diference of the maximum information gain. Here and subsequently, all omitted proofs are given in Section E.

Lemma 2. Consider the setup of Section 3.1 where $\mathcal { X }$ is compact. For every $T \geq 2$ and every integer $1 \leq m < T$ , we have

$$
R _ { T } ^ { * } \geq c _ { 1 } \cdot \sqrt { \frac { T ( \gamma _ { T } - \gamma _ { m } ) } { ( 1 + \lambda ^ { - 1 } ) \sum _ { t = m + 1 } ^ { T } t ^ { - 1 } } } ,\tag{36}
$$

where $\begin{array} { r } { c _ { 1 } = \operatorname* { m i n } \{ \frac { \sigma } { 3 2 } , \frac { \sigma \Delta _ { k } } { 1 6 } \} } \end{array}$

Lemma 2 implies the following suficient condition for $\Omega ( \sqrt { T \gamma _ { T } } )$ regret.

Proposition 2. Consider the setup of Section 3.1 with X being fixed and compact. Suppose that there exists a fixed integer $L \geq 2$ and a constant $c \in ( 0 , 1 )$ such that

$$
\gamma _ { \lfloor T / L \rfloor } \leq ( 1 - c ) \gamma _ { T }\tag{37}
$$

for all suficiently large T. Then $R _ { T } ^ { * } = \Omega ( \sqrt { T \gamma _ { T } } )$

For $\mathcal { X } = [ 0 , 1 ] ^ { d }$ , Table 1 shows that for the Mat´ern, γ-exponential and piecewise-polynomial kernels, $\gamma _ { T } =$ $\Theta ( T ^ { \alpha } )$ for some $\alpha > 0$ . It is straightforward to verify that these kernels satisfy Proposition 2: since $\gamma _ { T } = \Theta ( T ^ { \alpha } )$ , there exist constants $a , b > 0$ such that $a T ^ { \alpha } \leq \gamma _ { T } \leq b T ^ { \alpha }$ for all suficiently large T. Then we choose a fixed integer $L \geq 2$ satisfying $\textstyle { \frac { b } { a } } L ^ { - \alpha } \leq { \frac { 1 } { 2 } }$ . Therefore, for suficiently large $T$ ,

$$
\frac { \gamma _ { \lfloor T / L \rfloor } } { \gamma _ { T } } \leq \frac { b } { a } \bigg ( \frac { \lfloor T / L \rfloor } { T } \bigg ) ^ { \alpha } \leq \frac { b } { a } L ^ { - \alpha } \leq \frac { 1 } { 2 } .\tag{38}
$$

Then eq. (37) holds with $c = 1 / 2$ , and Proposition 2 gives $R _ { T } ^ { * } = \Omega ( \sqrt { T \gamma } ) = \Omega ( T ^ { ( \alpha + 1 ) / 2 } )$

On the other hand, the squared exponential, rational-quadratic and Dirichlet kernels fail eq. (37). To see this, suppose by contradiction that eq. (37) holds with $L \geq 2$ and $c \in ( 0 , 1 )$ . Then we choose N suficiently large that $\gamma _ { N } > 0$ and eq. (37) holds for every $T \geq N$ . We now fix $j \in \mathbb { Z } _ { + }$ . Iterating eq. (37) gives

$$
\gamma _ { L ^ { j } N } \geq ( 1 - c ) ^ { - 1 } \gamma _ { L ^ { j - 1 } N } \geq \cdot \cdot \cdot \geq ( 1 - c ) ^ { - j } \gamma _ { N } .\tag{39}
$$

Define $\eta : = \log ( ( 1 - c ) ^ { - 1 } ) / \log L > 0$ . It follows that $\gamma _ { L ^ { j } N } \geq \gamma _ { N } N ^ { - \eta } ( L ^ { j } N ) ^ { \eta }$ , which contradicts the polylogarithmic rates of $\gamma _ { T }$ for these kernels in Table 1. Failing eq. (37) does not eliminate the possibility of $\Omega ( \sqrt { T \gamma _ { T } } )$ regret for these kernels, but proving (or disproving) it would require a diferent argument.

## 4.2 Comparison with Upper Bounds

In this section, we first derive a general $O ( \sqrt { T \gamma \log T } )$ upper bound from (Camilleri et al., 2021, Theorem 3), then apply two approaches to remove the log T term.

Since upper bounds of $R _ { T } ^ { * }$ are often stated on finite domains (Camilleri et al., 2021; Li and Scarlett, 2022), we extend such bounds to $\mathcal { X } = [ 0 , 1 ] ^ { d }$ as follows. Let $\tilde { \mathcal { X } } \subset \mathcal { X }$ be finite. Define

$$
\epsilon ( \tilde { \mathcal { X } } ) : = \operatorname* { s u p } _ { f \in \mathcal { F } _ { k } ( B ) } \Big ( \operatorname* { m a x } _ { x \in \mathcal { X } } f ( x ) - \operatorname* { m a x } _ { a \in \tilde { \mathcal { X } } } f ( a ) \Big ) .\tag{40}
$$

Then, running a policy on $\tilde { \mathcal X }$ introduces at most an extra $T \epsilon ( \tilde { \mathcal X } )$ term to the minimax regret on $\mathcal { X } .$

A General $O ( \sqrt { T \gamma \log T } )$ Upper Bound. For finite X, (Camilleri et al., 2021, Theorem 3) presents a high-probability upper bound on $R _ { T } ( \pi _ { \mathrm { R I P S } } , f )$ , where π<sub>RIPS</sub> corresponds to its RIPS algorithm. This bound leads to the following minimax regret upper bound.

Lemma 3 (Camilleri et al. (2021), Theorem 3). Consider the setup of Section 1.1 where X is finite. Then

$$
R _ { T } ^ { * } = O \biggl ( \log ( | \mathcal { X } | T ) + \sqrt { T \operatorname* { m a x } _ { \mathcal { V } \subseteq \mathcal { X } } h ( \mathcal { V } , \rho ) } \sqrt { \log ( | \mathcal { X } | T ) } \biggr ) ,
$$

where, for each nonempty $\begin{array} { r } { \mathcal { V } \subseteq \mathcal { X } , \ h ( \mathcal { V } , \rho ) = \operatorname* { i n f } _ { \nu \in \Delta ( \mathcal { V } ) } \operatorname* { m a x } _ { x \in \mathcal { V } } \langle \phi ( x ) , ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k } . } \end{array}$

The following lemma connects $h ( \mathcal { V } , \rho ) ~ \mathrm { t o } ~ \gamma _ { T }$

Lemma 4. Consider the setup of Section 1.1 where X is finite. For nonempty $\nu \subseteq \mathcal { X }$ , define $h ( \nu , \rho )$ as in Lemma 3. Then

$$
\operatorname* { m a x } _ { \nu \subseteq \mathcal { X } } h ( \mathcal { V } , \rho ) \leq 2 ( 1 + \lambda ^ { - 1 } ) \gamma _ { T } .\tag{41}
$$

Combining Lemmas 3 and 4 gives $R _ { T } ^ { * } = O \bigl ( \log ( | \mathcal { X } | T ) + \sqrt { T \gamma _ { T } \log ( | \mathcal { X } | T ) } \bigr )$ when X is finite; this is within a log T factor of our general lower bound under the mild condition $| \mathcal { X } | \leq \mathrm { p o l y } ( T ) ~ ( \mathrm { e . g . } , | \mathcal { X } | = T ^ { d } )$ . To extend to $\mathcal { X } = [ 0 , 1 ] ^ { d }$ , we show that a H¨older condition allows us to choose a finite $\tilde { \mathcal { X } } \subset \mathcal { X }$ such that $T \epsilon ( \tilde { \mathcal X } )$ is lower-order, which leads to $O ( \sqrt { T \gamma \log T } )$ regret.

Theorem 3. Consider the setup in Section 3.1 with $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . Suppose that there exist fixed $C , \alpha > 0$ such that

$$
\begin{array} { r } { \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } \leq C \| x - x ^ { \prime } \| _ { 2 } ^ { \alpha } \qquad \forall x , x ^ { \prime } \in \mathcal { X } . } \end{array}\tag{42}
$$

Then, $R _ { T } ^ { * } = O \bigl ( \sqrt { T \gamma _ { T } \log T } \bigr )$

Importantly, all six kernels in Table 1 satisfy eq. (42), so Theorem 3 applies to each of them. The details are given in Appendix E.6.

Removing the Log Factor for Specific Kernels. The generic $O ( \sqrt { T \gamma \log T } )$ bound can be sharpened. Specifically, we now show that for the squared exponential (SE), rational quadratic, and one-dimensional Dirichlet kernel, the log T factor can be removed completely by a sharper analysis of $h ( \nu , \rho )$

Lemma 5. Consider the setup of Section 1.1 with finite $\mathcal { X } \subset [ 0 , 1 ] ^ { d }$ . For every nonempty $\nu \subseteq \mathcal { X }$ , define $h ( \nu , \rho )$ as in Lemma 3. Then the following upper bounds hold for $\operatorname* { m a x } _ { \mathcal { V } \subseteq \mathcal { X } } h ( \mathcal { V } , \rho )$

1. SE kernel k<sub>SE</sub>: $O ( ( \log T / \log \log T ) ^ { d } )$

2. Rational quadratic kernel $k _ { \mathrm { R Q } , a } \colon O ( ( \log T ) ^ { d } )$

3. One-dimensional Dirichlet kernel $k _ { \mathrm { D i r } , n } \colon 2 n + 1$

The implicit constants are independent of X .

For the squared exponential kernel, combining Lemmas 3 and 5 gives

$$
R _ { T } ^ { * } = O \biggl ( \log ( | \mathcal { X } | T ) + \sqrt { T \biggl ( \frac { \log T } { \log \log T } \biggr ) ^ { d } } \sqrt { \log ( | \mathcal { X } | T ) } \biggr ) .
$$

Following the same discretization in Theorem 3 gives

$$
R _ { T } ^ { * } = O { \left( \sqrt { T ( \log T ) ^ { d + 1 } / ( \log \log T ) ^ { d } } \right) } ,\tag{43}
$$

which combined with the rate of $\gamma _ { T }$ from Table 1 gives $R _ { T } ^ { * } = O ( \sqrt { T \gamma _ { T } } )$ . We can similarly verify the $O ( \sqrt { T \gamma _ { T } } )$ regret bound for the other two kernels.

For the Mat´ern, γ-exponential, and piecewise-polynomial kernels, the same approach applies, but fails to remove the log T factor. Alternatively, we apply MOSS (Lattimore and Szepesv´ari, 2020, Theorem 9.1) with a careful discretization of X to remove the log T factor in certain parameter regimes.

Lemma 6. Consider the setup of Section 3.1 where $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . For the Mat´ern kernel with $\nu \in ( 0 , 2 )$ γ-exponential kernel with $\gamma \in ( 0 , 2 )$ , and piecewise-polynomial kernel with $q = 0 , d \geq 3 ~ o r ~ q = 1$ , we have

$$
R _ { T } ^ { * } = O ( \sqrt { T \gamma _ { T } } ) .\tag{44}
$$

For the kernel settings in Lemma 6, the upper bounds match the corresponding lower bounds, establishing $R _ { T } ^ { * } = \Theta ( \sqrt { T \gamma _ { T } } )$ . For the remaining kernel settings in Section $\mathrm { A } , \mathrm { ~ a ~ } \sqrt { \log T }$ gap remains.

## 5 Conclusion

In this paper, we presented an $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ minimax regret lower bound for non-constant continuous positive semidefinite kernels on compact domains of $\mathbb { R } ^ { d }$ , thus establishing that the optimal minimax regret scaling is $\sqrt { T \gamma _ { T } }$ up to log factors in very high generality. We further showed that the logarithmic term cannot be removed in general, but can be removed on $[ 0 , 1 ] ^ { d } \operatorname { i f } \gamma _ { T } = \Theta ( T ^ { \alpha } )$ for some $\alpha > 0$ . By comparing to suitable upper bounds, we deduced that the minimax regret scaling is precisely $\Theta ( \sqrt { T \gamma _ { T } } )$ for several kernels of interest, while small gaps (typically $\sqrt { \log T }$ or log T) remain between the bounds for other kernels.

## AI Use Disclosure

Overview. We used AI to develop theoretical results, including proposing claims, providing proof outline and details, and proof-reading (but not direct editing). AI also helped with literature search, including summarizing existing results and identifying open problems. All AI-assisted work has been closely reviewed. In particular, all AI-assisted proofs were checked line by line by the authors. The authors take responsibility for all results and their correctness.

Specific details. The main contribution of this paper, i.e., the $\Omega ( \sqrt { T \gamma _ { T } / \log T } )$ lower bound, emerged from interactions between the authors and the AI agent, where the authors led the interactions. Subsequent interactions between the authors and the AI agent led to a significantly simplified argument that now essentially comprises Sections 2 and 3. For Section 4, the authors planned the high-level structure, and discussed with AI for details. For example, the authors asked the AI agent to help derive the rate of $\gamma _ { T }$ for the rational quadratic kernel, because this process is technically involved. The authors derived $O ( \sqrt { T \gamma \log T } )$ regret from (Camilleri et al., 2021, Theorem 3), while asking the AI agent to check whether an existing stronger upper bound exists. The sharpened $O ( \sqrt { T \gamma _ { T } } )$ upper bounds also followed from discussion with the AI agent.

## Acknowledgement

This work is supported by the National Research Foundation (NRF) Singapore and the Ministry of Digital Development and Information (MDDI) under the AI Visiting Professorship (Award AIVP-2024-003).

## References

Alaoui, A. and Mahoney, M. (2015). Fast randomized kernel ridge regression with statistical guarantees. In Advances in Neural Information Processing Systems, volume 28.

Allen-Zhu, Z., Li, Y., Singh, A., and Wang, Y. (2021). Near-optimal discrete optimization for experimental design: A regret minimization approach. Mathematical Programming, 186(1):439–478.

Audibert, J.-Y. and Bubeck, S. (2009). Minimax policies for adversarial and stochastic bandits. In Conference on Learning Theory, pages 217–226.

Auer, P., Cesa-Bianchi, N., Freund, Y., and Schapire, R. E. (2002). The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77.

Axler, S. (2024). Linear Algebra Done Right. Springer Cham, 4 edition.

Cai, X. and Scarlett, J. (2021). On lower bounds for standard and robust Gaussian process bandit optimization. In International Conference on Machine Learning, pages 1216–1226.

Camilleri, R., Jamieson, K., and Katz-Samuels, J. (2021). High-dimensional experimental design and kernel bandits. In International Conference on Machine Learning, pages 1227–1237.

Chowdhury, S. R. and Gopalan, A. (2017). On kernelized multi-armed bandits. In International Conference on Machine Learning, pages 844–853.

Derezinski, M., Liang, F., and Mahoney, M. (2020). Bayesian experimental design using regularized determi-

nantal point processes. In International Conference on Artificial Intelligence and Statistics, volume 108, pages 3197–3207.

DLMF (2026). NIST Digital Library of Mathematical Functions. https://dlmf.nist.gov/, Release 1.2.8 of 2026-09-15. F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl, and M. A. McClain, eds.

Iwazaki, S. (2026). Tighter regret lower bound for Gaussian process bandits with squared exponential kernel in hypersphere. In International Conference on Machine Learning.

Janz, D., Akhavan, A., and Tsybakov, A. B. (2026). Maximum efective dimension and information gain. arXiv preprint arXiv:2608.24450.

Lattimore, T. and Szepesv´ari, C. (2020). Bandit algorithms. Cambridge University Press.

Lee, M. and Javidi, T. (2025). Consequences of kernel regularity for bandit optimization. arXiv preprint arXiv:2512.05957.

Li, Z. and Scarlett, J. (2022). Gaussian process bandit optimization with few batches. In International Conference on Artificial Intelligence and Statistics, pages 92–107.

Liu, Y., Wang, Y., and Singh, A. (2021). Smooth bandit optimization: Generalization to Holder space. In International Conference on Artificial Intelligence and Statistics, volume 130, pages 2206–2214.

Ma, C., Chen, K., and Scarlett, J. (2026). Batched kernelized bandits: Refinements and extensions. arXiv preprint arXiv:2603.12627.

Rasmussen, C. E. and Williams, C. K. I. (2005). Gaussian Processes for Machine Learning. The MIT Press.

Salgia, S., Vakili, S., and Zhao, Q. (2021). A domain-shrinking based Bayesian optimization algorithm with order-optimal regret performance. Conference on Neural Information Processing Systems, 34:28836–28847.

Scarlett, J., Bogunovic, I., and Cevher, V. (2017). Lower bounds on regret for noisy Gaussian process bandit optimization. In Conference on Learning Theory, pages 1723–1742.

Singh, S. (2021). Continuum-armed bandits: A function space perspective. In International Conference on Artificial Intelligence and Statistics, pages 2620–2628.

Srinivas, N., Krause, A., Kakade, S., and Seeger, M. (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. In International Conference on Machine Learning, page 1015–1022.

Vakili, S., Khezeli, K., and Picheny, V. (2021). On information gain and regret bounds in Gaussian process bandits. In International Conference on Artificial Intelligence and Statistics, pages 82–90.

Valko, M., Korda, N., Munos, R., Flaounas, I., and Cristianini, N. (2013). Finite-time analysis of kernelised contextual bandits. In Conference on Uncertainty in Artificial Intelligence, page 654–663.

Wendland, H. (2004). Scattered Data Approximation. Cambridge Monographs on Applied and Computational Mathematics. Cambridge University Press.

## A Definitions of Kernels

Following (Lee and Javidi, 2025), we consider six stationary kernels. For $x , x ^ { \prime } \in { \mathcal { X } }$ , we define $r : = \| x - x ^ { \prime } \| _ { 2 }$ and denote $k ( r ) : = k ( x , x ^ { \prime } )$ . For each kernel, we present the expression and specify the range of parameters, where $\ell > 0$ (when it appears) is a lengthscale parameter.

• Squared exponential (SE) kernel:

$$
k _ { \mathrm { S E } } ( r ) = \exp \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \left( - r ^ { 2 } / ( 2 \ell ^ { 2 } ) \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) .\tag{45}
$$

• Rational-quadratic kernel:

$$
k _ { \mathrm { R Q } , a } ( r ) = \bigg ( 1 + \frac { r ^ { 2 } } { 2 a \ell ^ { 2 } } \bigg ) ^ { - a } ,\tag{46}
$$

where $a > 0$

• Mat´ern kernel:

$$
k _ { \nu } ( r ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } { \left( \frac { \sqrt { 2 \nu } r } { \ell } \right) } ^ { \nu } K _ { \nu } { \left( \frac { \sqrt { 2 \nu } r } { \ell } \right) } ,\tag{47}
$$

where $\nu > 0$ is a smoothness parameter, Γ is the gamma function, and $K _ { \nu }$ is the modified Bessel function of the second kind.

• γ-exponential kernel:

$$
k _ { \gamma - \mathrm { E x p } } ( r ) = \exp \bigl ( - ( r / \ell ) ^ { \gamma } \bigr ) ,\tag{48}
$$

where $\gamma \in ( 0 , 2 )$ . The special case $\gamma = 2$ is excluded because that recovers the SE kernel.

• Piecewise-polynomial kernel:

$$
k _ { \mathrm { P P } , q } ( r ) = \left\{ \begin{array} { l l } { c _ { 0 , q } + \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor + 3 q + 1 } c _ { j , q } r ^ { j } , } & { 0 \leq r \leq 1 , } \\ { 0 , } & { r > 1 . } \end{array} \right.\tag{49}
$$

Here d is the dimension of $\mathcal { X } , \ q \in \mathbb { Z } _ { \geq 0 }$ and $q \geq 1 { \mathrm { ~ i f ~ } } d = 1 , 2$ . (Lee and Javidi, 2025) states that the coeficients $c _ { j , q }$ can be computed recursively (Wendland (2004), Theorem 9.13).

• The one-dimensional Dirichlet kernel:

$$
k _ { \mathrm { D i r } , n } ( r ) = \frac { 1 } { 2 n + 1 } \sum _ { k = - n } ^ { n } \exp ( - i k r ) ,\tag{50}
$$

where $n \in \mathbb { Z } _ { + }$ . We exclude $n = 0$ because that leads to $k \equiv 1$ , while Section 3.1 requires the kernel to be non-constant, and we restrict to $d = 1$ because the Dirichlet kernel may fail to be positive semidefinite in higher dimensions.

## B Technical Lemmas

Lemma 7. Consider the setup of Section 1.1 where $\mathcal { X }$ is finite. Let A be a self-adjoint operator on E and $u \in E$ . Then

$$
\operatorname { t r } _ { E } ( ( u \otimes u ) A ) = \langle u , A u \rangle _ { k } .\tag{51}
$$

Proof. Recall that $e _ { 1 } , \ldots e _ { n }$ is an orthonormal basis of E. Then

$$
\mathrm { t r } _ { E } ( ( u \otimes u ) A ) = \sum _ { i = 1 } ^ { n } \langle ( u \otimes u ) A e _ { i } , e _ { i } \rangle _ { k } = \sum _ { i = 1 } ^ { n } \langle u , A e _ { i } \rangle _ { k } \langle u , e _ { i } \rangle _ { k } = \sum _ { i = 1 } ^ { n } \langle u , e _ { i } \rangle _ { k } \langle A u , e _ { i } \rangle _ { k } = \langle u , A u \rangle _ { k } ,\tag{52}
$$

where the last equality follows from (Axler, 2024, Eq. 6.30(c)).

Lemma 8. Consider the setup of Section 1.1 where X is finite. Let A be a self-adjoint operator on E and $u \in E$ . Then

$$
( A u ) \otimes ( A u ) = A ( u \otimes u ) A .\tag{53}
$$

Proof. Choose any $v \in E$ . Then ${ \bigl ( } ( A u ) \otimes ( A u ) { \bigr ) } v = ( A u ) \langle A u , v \rangle _ { k } = ( A u ) \langle u , A v \rangle _ { k } = { \bigl ( } A ( u \otimes u ) A { \bigr ) } v .$

Lemma 9. Consider the setup of Section 1.1 where X is finite. For every $\nu \in \Delta ( \mathcal { X } )$

$$
M _ { \nu } + \rho I _ { E } \succ M _ { \nu } \succeq 0 .\tag{54}
$$

Consequently, $( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } \succ 0 , ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 / 2 }$ is self-adjoint, and $( M _ { \nu } + \rho I _ { E } ) ^ { - 2 } \preceq \rho ^ { - 1 } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 }$

Proof. To show $M _ { \nu } + \rho I _ { E } \succ M _ { \nu }$ , it sufices to show that $\rho I _ { E } \succ 0$ and $M _ { \nu } \succeq 0$ . The condition $\rho I _ { E } \succ 0$ holds since $\langle \rho I _ { E } v , v \rangle _ { k } = \rho \| v \| _ { k } ^ { 2 } > 0$ for every $v \in E$ and $v \ne 0$ . To show $M _ { \nu } \succeq 0 .$ , we recall that $M _ { \nu } =$ $\begin{array} { r } { \sum _ { x \in \mathcal { X } } \nu ( x ) \phi ( x ) \otimes \phi ( x ) } \end{array}$ , such that $\begin{array} { r } { \langle M _ { \nu } v , v \rangle _ { k } = \sum _ { x \in \mathcal { X } } \nu ( x ) | \langle \phi ( x ) , v \rangle _ { k } | ^ { 2 } \geq 0 } \end{array}$ for every $v \in E$ . This establishes $\mathrm { e q . ~ } \left( 5 4 \right)$

Since $M _ { \nu } + \rho I _ { E }$ is self-adjoint, it follows that $( M _ { \nu } + \rho I _ { E } ) ^ { - 1 }$ is self-adjoint, which combined with the fact that all eigenvalues of $( M _ { \nu } + \rho I _ { E } ) ^ { - 1 }$ are positive proves $( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } \succ 0$ . This also proves $( M _ { \nu } + \rho I _ { E } ) ^ { - 1 / 2 }$ is self-adjoint. To show $( M _ { \nu } + \rho I _ { E } ) ^ { - 2 } \preceq \rho ^ { - 1 } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 }$ , observe that $\rho ^ { - 1 } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } - ( M _ { \nu } + \rho I _ { E } ) ^ { - 2 } =$ $\rho ^ { - 1 } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } M _ { \nu } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 }$ . For every $u \in E$ , observe that $\langle ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } M _ { \nu } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } u , u \rangle _ { k } =$ $\langle M _ { \nu } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } u , ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } u \rangle _ { k } \geq 0 .$ , which proves $( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } M _ { \nu } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } \succeq 0 .$ and thus $( M _ { \nu } + \rho I _ { E } ) ^ { - 2 } \preceq \rho ^ { - 1 } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } .$ □

Lemma 10 (Camilleri et al. (2021), Lemma 15). Consider the setup of Section 1.1 where X is finite. Then

$$
\operatorname* { m a x } _ { x \in \mathcal { X } } \bigl \langle \phi ( x ) , ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \phi ( x ) \bigr \rangle _ { k } = \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) \bigl \langle \phi ( x ) , ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \phi ( x ) \bigr \rangle _ { k } = d _ { T } .\tag{55}
$$

Lemma 11. Consider the setup of Section 1.1 where X is finite. For every $\nu \in \Delta ( \mathcal { X } )$ (X ),

$$
\mathrm { t r } _ { E } \big ( M _ { \nu } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } \big ) \leq 2 d _ { T } .\tag{56}
$$

Proof. Fix $\nu \in \Delta ( \mathcal { X } )$ . Since $M _ { T } : = M _ { \nu _ { T } }$ where $\nu _ { T } \in \Delta ( \mathcal { X } )$ , Lemma 9 shows that $M _ { \nu } , M _ { T } \succeq 0$ , such that

$$
\mathrm { t r } _ { E } ( M _ { \nu } ( M _ { \nu } + \rho I _ { E } ) ^ { - 1 } ) \le \mathrm { t r } _ { E } ( ( M _ { \nu } + M _ { T } ) ( M _ { \nu } + M _ { T } + \rho I _ { E } ) ^ { - 1 } )\tag{57}
$$

$$
\leq \mathrm { t r } _ { E } \big ( ( M _ { \nu } + M _ { T } ) ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \big )\tag{58}
$$

$$
= \mathrm { t r } _ { E } ( M _ { \nu } ( M _ { T } + \rho I _ { E } ) ^ { - 1 } ) + d _ { T } ,\tag{59}
$$

where the equality follows from the linearity of the trace and eq. (5). For the first term in eq. (59), observe that

$$
\mathrm { t r } _ { E } ( M _ { \nu } ( M _ { T } + \rho I _ { E } ) ^ { - 1 } ) = \sum _ { x \in \mathcal { X } } \nu ( x ) \mathrm { t r } _ { E } ( ( \phi ( x ) \otimes \phi ( x ) ) ( M _ { T } + \rho I _ { E } ) ^ { - 1 } )\tag{60}
$$

$$
= \sum _ { x \in \mathcal { X } } \nu ( x ) \langle \phi ( x ) , ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k }\tag{61}
$$

$$
\leq \operatorname* { m a x } _ { x \in \mathcal { X } } \langle \phi ( x ) , ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k }\tag{62}
$$

$$
= d _ { T } ,\tag{63}
$$

where eq. (61) applies Lemma 7, eq. (62) follows since the average is upper bounded by the maximum, and eq. (63) applies Lemma 10. Combining eqs. (59) and (63) proves the result. □

Lemma 12. Consider the setup of Section 1.1 where X is finite. Fix $u \in E$ , and let A be an operator on E such that $A - u \otimes u \succ 0$ . Then

$$
\frac { \operatorname* { d e t } _ { E } ( A - u \otimes u ) } { \operatorname* { d e t } _ { E } A } = 1 - \langle u , A ^ { - 1 } u \rangle _ { k } .\tag{64}
$$

Proof. It is straightforward to verify that $u \otimes u \succeq 0$ , such that $A \ \succ \ 0$ . Define $w : = A ^ { - 1 / 2 } u$ . Then $1 - \langle u , A ^ { - 1 } u \rangle _ { k } = \bar { 1 } - \| w \| _ { k } ^ { 2 }$ . Factoring out $A ^ { 1 / 2 }$ gives

$$
{ \frac { \operatorname* { d e t } _ { E } ( A - u \otimes u ) } { \operatorname* { d e t } _ { E } A } } = \operatorname* { d e t } _ { E } { \big ( } I _ { E } - A ^ { - 1 / 2 } ( u \otimes u ) A ^ { - 1 / 2 } { \big ) } = { \frac { \operatorname* { d e t } ( I _ { E } - w \otimes w ) } { E } } ,\tag{65}
$$

where the second equality follows from $A ^ { - 1 / 2 }$ being self-adjoint and Lemma 8. Now we consider two cases of w in eq. (65): If $w \ne 0$ , then the operator $I _ { E } - w \otimes w$ has eigenvalue $1 - \| w \| _ { k } ^ { 2 }$ on span{w}, and eigenvalue 1 on span $\{ w \} ^ { \perp }$ , such that det ${ } _ { E } ( I _ { E } - w \otimes w ) = 1 - \| w \| _ { k } ^ { 2 }$ . If $w = 0$ , then det $\mathbf { \sigma } _ { E } ( I _ { E } - w \otimes w ) = \operatorname* { d e t } _ { E } I _ { E } = 1 = 1 - \| w \| _ { k } ^ { 2 } .$ Then $1 - \langle u , A ^ { - 1 } u \rangle _ { k } = \operatorname* { d e t } _ { E } ( I _ { E } - w \otimes w )$ , which combined with eq. (65) proves the result. □

Lemma 13. Consider the setup of Section 1.1 where X is finite. Fix $u \in E$ , and let A be an operator on E such that $A \succ 0$ . Then

$$
\frac { \operatorname* { d e t } _ { E } ( A + u \otimes u ) } { \operatorname* { d e t } _ { E } A } = 1 + \langle u , A ^ { - 1 } u \rangle _ { k } .\tag{66}
$$

Proof. The proof is entirely analogous to that of Lemma 12, so is omitted.

Lemma 14. Consider the setup of Section 1.1 where X is finite. Fix $x _ { 1 } , \dotsc , x _ { T } \in \mathcal { X }$ , and define $\hat { \nu } : = $ $\begin{array} { r } { \frac { 1 } { T } \sum _ { s = 1 } ^ { T } \delta _ { x _ { s } } \in \Delta ( \mathcal { X } ) } \end{array}$ . Then $K _ { T } ( x _ { 1 : T } )$ and $T M _ { \hat { \nu } }$ have the same positive eigenvalues (including multiplicities).

$P r o o f .$ Define the linear map $\Phi : \mathbb { R } ^ { T }  E$ by $\Phi b _ { i } = \phi ( x _ { i } )$ , where $b _ { i }$ is the i-th element of the standard basis of $\mathbb { R } ^ { T }$ . By $\mathrm { e q . ~ ( 3 ) }$ and the definition of $K _ { T } ( x _ { 1 : T } )$ , we have $K _ { T } = \Phi ^ { * } \Phi$ and $T M _ { \hat { \nu } } = \Phi \Phi ^ { * }$ , where $\Phi ^ { * }$ is the adjoint of Φ. Since $K _ { T } ( x _ { 1 : T } ) , M _ { \hat { \nu } } \succeq 0$ , all their non-zero eigenvalues are positive, and the result follows.

Lemma 15. Consider the setup in Section 3.1 where X is compact. Then,

$$
\gamma _ { T } ( \mathcal { X } ) \geq \frac { 1 } { 2 } \log ( 1 + T k ( x _ { 0 } , x _ { 0 } ) \lambda ^ { - 1 } ) ,\tag{67}
$$

where $x _ { 0 } \in$ arg $\operatorname* { m a x } _ { x \in \mathcal { X } } k ( x , x )$

Proof. For every $x \in \mathcal { X }$ , we have $k ( x , x ) = \langle \phi ( x ) , \phi ( x ) \rangle _ { k } \geq 0$ . Since k is non-constant, we have $k ( x _ { 0 } , x _ { 0 } ) > 0$ By the definition of $\gamma _ { T } ( \mathcal { X } )$ in eq. (31) and considering the choice $x _ { 1 } = . . . = x _ { T } = x _ { 0 }$ , we have

$$
\gamma _ { T } ( \mathcal { X } ) \geq \frac { 1 } { 2 } \log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } k ( x _ { 0 } , x _ { 0 } ) 1 _ { T } ) = \frac { 1 } { 2 } \log ( 1 + T k ( x _ { 0 } , x _ { 0 } ) \lambda ^ { - 1 } ) ,\tag{68}
$$

where $1 _ { T }$ is the $T \times T$ all-ones matrix.

Lemma 16. Under the setup in Section 1.1 where X is finite, we have $d _ { T } > 0$

Proof. By (Camilleri et al., 2021, Lemma 15), we have

$$
d _ { T } = \operatorname* { m a x } _ { x \in \mathcal { X } } \bigl \langle \phi ( x ) , ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \phi ( x ) \bigr \rangle _ { k } .\tag{69}
$$

Since $M _ { T } + \rho I _ { E }$ is positive definite by Lemma 9, it follows that $( M _ { T } + \rho I _ { E } ) ^ { - 1 }$ is also positive definite. Observe that there exists $x _ { 0 } \in \mathcal { X }$ such that $\phi ( \boldsymbol { x } _ { 0 } ) \ne 0$ , because otherwise $k ( x , x ^ { \prime } ) = \langle \phi ( x ) , \phi ( x ^ { \prime } ) \rangle _ { k } = 0$ for every $x , x ^ { \prime } \in { \mathcal { X } }$ , contradicting that k is non-constant. Then $d _ { T } \geq \left. \phi ( x _ { 0 } ) , ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \phi ( x _ { 0 } ) \right. _ { k } > 0$ □

Lemma 17. Consider the setup of Section 1.1 where X is finite. Suppose $\nu _ { 1 } , \nu _ { 2 } \in \Delta ( \mathcal { X } )$ , and that both $\nu _ { 1 } , \nu _ { 2 }$ are regularized D-optimal designs satisfying eq. (4) for some regularization level $\rho > 0$ . Then $M _ { \nu _ { 1 } } = M _ { \nu _ { 2 } }$

Proof. If $\nu _ { 1 } ~ = ~ \nu _ { 2 }$ , then the result trivially follows, so we assume that $\nu _ { 1 } ~ \neq ~ \nu _ { 2 }$ . Fix $\nu \in \Delta ( \mathcal { X } )$ . By Lemma 9, $M _ { \nu } + \rho I _ { E }$ is positive definite. We now verify that log de $\dot { , } \boldsymbol { E }$ is strictly concave in the space of positive definite operators. Let A, B be distinct positive definite operators on E, and let $t \in \mathsf { \Gamma } ( 0 , 1 )$ Define $C : = A ^ { - 1 / 2 } \bar { B } A ^ { - 1 / 2 }$ . Then C is positive definite. Let $\lambda _ { 1 } , \ldots , \lambda _ { n } ~ > ~ 0$ be its eigenvalues. Since $( 1 - t ) A + t B = A ^ { 1 / 2 } \big ( ( 1 - t ) I _ { E } + t C \big ) A ^ { 1 / 2 }$ , it follows that

$$
\log \operatorname* { d e t } _ { E } \bigl ( ( 1 - t ) A + t B \bigr ) = \log \operatorname* { d e t } _ { E } A + \sum _ { j = 1 } ^ { n } \log \bigl ( ( 1 - t ) + t \lambda _ { j } \bigr ) .\tag{70}
$$

Also notice that log det $\begin{array} { r } { \mathbf { \Lambda } _ { E } ( B ) = \log \operatorname* { d e t } _ { E } ( A ) + \sum _ { i = 1 } ^ { n } } \end{array}$ log $\lambda _ { i } .$ Since the logarithm is strictly concave on $( 0 , \infty )$ ， we have log $\left( ( 1 - t ) + t \lambda _ { i } \right) \geq t \log \lambda _ { i } .$ , with equality when $\lambda _ { i } = 1$ . Since $A \neq B .$ , it follows that $C \neq I _ { E } ,$ , such that at least one of $\lambda _ { 1 } , \ldots , \lambda _ { n }$ does not equal 1. Combining, we have

$$
\log \operatorname* { d e t } _ { E } \bigl ( ( 1 - t ) A + t B \bigr ) > ( 1 - t ) \log \operatorname* { d e t } _ { E } ( A ) + t \log \operatorname* { d e t } _ { E } ( B ) .\tag{71}
$$

Now suppose by contradiction that $M _ { \nu _ { 1 } } \neq M _ { \nu _ { 2 } }$ . Consider the mixed design $\nu _ { t } : = ( 1 - t ) \nu _ { 1 } + t \nu _ { 2 } \in \Delta ( { \mathcal { X } } )$ Since the mapping $\nu \mapsto M _ { \nu }$ is linear, $M _ { \nu _ { t } } = ( 1 - t ) M _ { \nu _ { 1 } } + t M _ { \nu _ { 2 } }$ . Then, applying eq. (71) with $A = M _ { \nu _ { 1 } } + \rho I _ { E }$ and $B = M _ { \nu _ { 2 } } + \rho I _ { E }$ shows that

$$
\log \operatorname* { d e t } _ { E } \left( M _ { \nu _ { t } } + \rho I _ { E } \right) > ( 1 - t ) \log \operatorname* { d e t } _ { E } ( M _ { \nu _ { 1 } } + \rho I _ { E } ) + t \log \operatorname* { d e t } _ { E } ( M _ { \nu _ { 2 } } + \rho I _ { E } ) ,\tag{72}
$$

which implies the design value at $\nu _ { t }$ (notice $\nu _ { t }$ is diferent from $\nu _ { 1 }$ and $\nu _ { 2 }$ since $\nu _ { 1 } \neq \nu _ { 2 } )$ is larger than the D-optimal design value, establishing a contradiction. Hence, $M _ { \nu _ { 1 } } = M _ { \nu _ { 2 } }$ □

Lemma 18. Consider the setup of Section 1.1 where X is finite. Then

$$
\mathrm { t r } _ { E } ( M _ { T } ) \leq 1 .\tag{73}
$$

Proof. Recall that $e _ { 1 } , \ldots , e _ { n }$ is an orthonormal basis of E. Then

$$
\mathrm { t r } _ { E } ( M _ { T } ) = \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) \mathrm { t r } _ { E } \big ( \phi ( x ) \otimes \phi ( x ) \big )\tag{74}
$$

$$
= \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) \sum _ { i = 1 } ^ { n } \langle e _ { i } , ( \phi ( x ) \otimes \phi ( x ) ) e _ { i } \rangle _ { k }\tag{75}
$$

$$
= \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) \sum _ { i = 1 } ^ { n } \lvert \langle \phi ( x ) , e _ { i } \rangle _ { k } \rvert ^ { 2 }\tag{76}
$$

$$
= \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) \| \phi ( x ) \| _ { k } ^ { 2 }\tag{77}
$$

$$
= \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) k ( x , x )\tag{78}
$$

$$
\leq 1 ,\tag{79}
$$

where eq. (76) follows from the definition of $\otimes ,$ eq. (77) uses Parseval’s identity (Axler, 2024, $\mathrm { E q . 6 . 3 0 ( b ) ) }$ , eq. (78) follows from the reproducing property, and eq. (79) follows since $k ( x , x ) \leq 1$ for every $x \in \mathcal { X }$ □

Lemma 19. Under the setup of Section 3.1 where X is compact, the mapping $x \mapsto \phi ( x )$ from X to $\mathcal { H } _ { k }$ is continuous.

Proof. Observe that $\| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } = \sqrt { \langle \phi ( x ) - \phi ( x ^ { \prime } ) , \phi ( x ) - \phi ( x ^ { \prime } ) \rangle _ { k } } = \sqrt { k ( x , x ) + k ( x ^ { \prime } , x ^ { \prime } ) - 2 k ( x , x ^ { \prime } ) }$ Since k is continuous, $\| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } \to 0$ as $x \ \to \ x ^ { \prime }$ , which implies that the mapping $x \ \to \ \phi ( x )$ is continuous. □

Lemma 20. Consider the setup of Section 3.1 where X is compact. For any $f \in { \mathcal { F } } _ { k } ( B )$ , the function $f : \mathcal { X }  \mathbb { R }$ is continuous.

Proof. By the reproducing property, $f ( x ) = \langle f , \phi ( x ) \rangle _ { k }$ . It follows that $| f ( x ) - f ( x ^ { \prime } ) | = | \langle f , \phi ( x ) - \phi ( x ^ { \prime } ) \rangle _ { k } | \leq$ $\| f \| _ { k } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } \leq B \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k }$ , where the first inequality uses Cauchy-Schwarz, and the second follows from $\| f \| _ { k } \leq B$ . Combined with Lemma 19, this shows that $f ( x )  f ( x ^ { \prime } )$ as $x \to x ^ { \prime }$ , which implies that $f$ is continuous on X . □

Lemma 21. Under the setup of Section 3.1, $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$ is well-defined.

Proof. Fix $x , x ^ { \prime } \in { \mathcal { X } }$ . Since k is continuous, it follows that $\| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } = \sqrt { k ( x , x ) + k ( x ^ { \prime } , x ^ { \prime } ) - 2 k ( x , x ^ { \prime } ) }$ is continuous on $\mathcal { X } \times \mathcal { X }$ . Also observe that $\mathcal { X } ^ { 2 }$ is compact. Then the max operation in $\Delta _ { k }$ is well-defined.

Lemma 22 (Auer et al. (2002), Proof of Lemma A.1). Let $P , Q$ be two probability measures on the same measurable space, and let H be a random variable satisfying $0 \leq H \leq M$ almost surely. Then

$$
\mathbb { E } _ { P } [ H ] \ge \mathbb { E } _ { Q } [ H ] - M \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } ( Q \| P ) } .\tag{80}
$$

## C Proofs of Theorem 1 and Corollary 1 (Finite Domains)

Recall that $V _ { T } : = { { M } _ { T } } + \rho { { I } _ { E } }$ , and $u _ { x } : = d _ { T } ^ { - 1 / 2 } V _ { T } ^ { - 1 / 2 } \phi ( x ) \in \mathcal { H } _ { k }$ for every $x \in \mathcal { X } . \ u _ { x }$ can be viewed as a normalized (and transformed) feature vector; its key properties are given as follows.

Lemma 23. Let $Z \sim \nu _ { T }$ for ν<sub>T</sub> given in eq. (4). Then

$$
\| u _ { x } \| _ { k } \leq 1 \quad \forall x \in \mathcal { X } ,\tag{81}
$$

$$
\| u _ { Z } \| _ { k } = 1 \quad a l m o s t \ s u r e l y ,\tag{82}
$$

$$
\mathbb { E } [ u _ { Z } \otimes u _ { Z } ] \preceq d _ { T } ^ { - 1 } I _ { E } ,\tag{83}
$$

$$
\| \mathbb { E } [ u _ { Z } ] \| _ { k } \leq d _ { T } ^ { - 1 / 2 } .\tag{84}
$$

Proof. Proof of eq. (81). Since ${ V _ { T } ^ { - 1 / 2 } }$ is self-adjoint (Lemma 9), we have

$$
\| u _ { x } \| _ { k } ^ { 2 } = d _ { T } ^ { - 1 } \big \langle \phi ( x ) , V _ { T } ^ { - 1 } \phi ( x ) \big \rangle _ { k } ,\tag{85}
$$

which combined with Lemma 10 proves eq. (81).

Proof of eq. (82). Applying eq. (85) and then Lemma 10 gives

$$
\mathbb { E } [ \| u _ { Z } \| _ { k } ^ { 2 } ] = \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) \| u _ { x } \| _ { k } ^ { 2 }\tag{86}
$$

$$
= \sum _ { x \in \mathcal { X } } \nu _ { T } ( x ) d _ { T } ^ { - 1 } \big \langle \phi ( x ) , V _ { T } ^ { - 1 } \phi ( x ) \big \rangle _ { k }\tag{87}
$$

(88)

By eq. (81), each $x \in \mathcal { X }$ satisfies $\| u _ { x } \| _ { k } ^ { 2 } \leq 1$ , which combined with eq. (88) shows that $\| u _ { x } \| _ { k } ^ { 2 } = 1$ for every $x \in \mathcal { X }$ such that $\nu _ { T } ( x ) > 0$ , proving eq. (82).

Proof of eq. (83). Since $V _ { T } ^ { - 1 / 2 }$ is self-adjoint (Lemma 9), Lemma 8 gives

$$
\bigl ( u _ { Z } \otimes u _ { Z } \bigr ) = d _ { T } ^ { - 1 } V _ { T } ^ { - 1 / 2 } \bigl ( \phi ( Z ) \otimes \phi ( Z ) \bigr ) V _ { T } ^ { - 1 / 2 } .\tag{89}
$$

Averaging over $Z \sim \nu _ { T }$ then gives

$$
\mathbb { E } _ { Z } [ u _ { Z } \otimes u _ { Z } ] = d _ { T } ^ { - 1 } V _ { T } ^ { - 1 / 2 } \mathbb { E } _ { Z } [ \phi ( Z ) \otimes \phi ( Z ) ] V _ { T } ^ { - 1 / 2 }
$$

$$
= d _ { T } ^ { - 1 } V _ { T } ^ { - 1 / 2 } M _ { T } V _ { T } ^ { - 1 / 2 }\tag{90}
$$

$$
= d _ { T } ^ { - 1 } \big ( I _ { E } - \rho V _ { T } ^ { - 1 } \big )\tag{91}
$$

(92)

$$
\preceq d _ { T } ^ { - 1 } I _ { E } ,\tag{93}
$$

where eq. (91) follows from the definition of $M _ { T }$ (see eq. (3)), eq. (92) follows since $M _ { T } = V _ { T } - \rho I _ { E }$ , and eq. (93) follows because $V _ { T } ^ { - 1 }$ is positive definite (Lemma 9).

Proof of eq. (84). We fix $v \in E .$ and observe that

$$
| \langle \mathbb { E } [ u _ { Z } ] , v \rangle _ { k } | ^ { 2 } = \mathbb { E } [ \langle u _ { Z } , v \rangle _ { k } ] ^ { 2 }\tag{94}
$$

$$
\leq \mathbb { E } [ | \langle u _ { Z } , v \rangle _ { k } | ^ { 2 } ]\tag{95}
$$

$$
= \mathbb { E } [ \langle v , ( u _ { Z } \otimes u _ { Z } ) v \rangle _ { k } ]\tag{96}
$$

$$
\mathbf { \tau } = \langle v , \mathbb { E } [ ( u _ { Z } \otimes u _ { Z } ) ] v \rangle _ { k }\tag{97}
$$

$$
\leq \langle v , d _ { T } ^ { - 1 } I _ { E } v \rangle _ { k }\tag{98}
$$

$$
\begin{array} { r } { = d _ { T } ^ { - 1 } \| v \| _ { k } ^ { 2 } , } \end{array}\tag{99}
$$

where eqs. (94) and (97) follow from linearity of expectation, eq. (96) follows because $\langle v , ( u _ { Z } \otimes u _ { Z } ) v \rangle _ { k } =$ $\langle v , u _ { Z } \langle u _ { Z } , v \rangle _ { k } \rangle _ { k } = | \langle u _ { Z } , v \rangle _ { k } | ^ { 2 }$ , eq. (95) uses Jensen’s inequality, and eq. (98) follows from eq. (83). If $\mathbb { E } [ u _ { Z } ] \ne 0$ , taking $v = \mathbb { E } [ u _ { Z } ] / \| \mathbb { E } [ u _ { Z } ] \| _ { k }$ in eq. (99) proves eq. (84). If $\mathbb { E } [ u _ { Z } ] = 0$ , then eq. (84) holds trivially. □

The following lemma uses a simpler two-function construction, and will be used to handle the case $d _ { T } < 4$

Lemma 24. Under the setup in Section 1.1,

$$
R _ { T } ^ { * } \geq \frac { \Delta _ { k } } { 8 } \operatorname* { m i n } \{ 2 B T , \sigma \sqrt { T } \} ,\tag{100}
$$

where $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$

Proof. Step 1: Construct hard functions. Choose $x _ { + } , x _ { - } \in { \mathcal { X } }$ attaining $\begin{array} { r } { \Delta _ { k } = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \lVert \phi ( x ) - \phi ( x ^ { \prime } ) \rVert _ { k } } \end{array}$ and define

$$
g : = \frac { \phi ( x _ { + } ) - \phi ( x _ { - } ) } { \Delta _ { k } } \in \mathcal { H } _ { k } ,\tag{101}
$$

which implies $\| g \| _ { k } = 1$ . By the reproducing property, $g ( x _ { + } ) - g ( x _ { - } ) = \langle g , \phi ( x _ { + } ) - \phi ( x _ { - } ) \rangle _ { k } = \Delta _ { k } ^ { - 1 } \lVert \phi ( x _ { + } ) - \phi ( x _ { - } ) \rVert$ $\phi ( x _ { - } ) \Vert _ { k } ^ { 2 } = \Delta _ { k }$ . Conversely, for every $x , x ^ { \prime } \in { \mathcal { X } } .$ , Cauchy-Schwarz gives $| g ( x ) - g ( x ^ { \prime } ) | = | \langle g , \phi ( x ) - \phi ( x ^ { \prime } ) \rangle _ { k } | \leq$ $\| g \| _ { k } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } \leq \Delta _ { k }$ . It follows that

$$
\operatorname* { m a x } _ { x \in \mathcal { X } } g ( x ) - \operatorname* { m i n } _ { x \in \mathcal { X } } g ( x ) = \Delta _ { k } .\tag{102}
$$

Moreover, for every $x \ \in \ \mathcal { X } , \ | g ( x ) | \ = \ | \langle g , \phi ( x ) \rangle _ { k } | \ \leq \ \| g \| _ { k } \| \phi ( x ) \| _ { k } \ \leq \ 1$ due to $k ( x , x ) ~ \leq ~ 1$ . Let $a : =$ min $\textstyle \left\{ B , { \frac { \sigma } { 2 { \sqrt { T } } } } \right\}$ , and for $v \in \{ + 1 , - 1 \}$ , define

$$
f _ { v } : = v a \cdot g \in \mathcal { H } _ { k } .\tag{103}
$$

Since $a \leq B .$ it follows that $f _ { v } \in \mathcal { F } _ { k } ( B )$

Step 2: Change of measure. Recall that $H _ { t } : = ( X _ { 1 } , Y _ { 1 } , \ldots , X _ { t } , Y _ { t } )$ is the history until time t. Fix a policy π. Let V be uniform on $\{ + 1 , - 1 \}$ . Let P be the joint law on $( V , H _ { T } )$ obtained by drawing V and running π against $f _ { V }$ , and Q be the joint law on $( V , H _ { T } )$ obtained by drawing V independently and running π against the zero function. Applying the chain-rule for KL divergence and following eq. (16) to eq. (18) gives

$$
D _ { \mathrm { K L } } ( Q \| P ) = \frac { 1 } { 2 \sigma ^ { 2 } } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ f _ { V } ( X _ { t } ) ^ { 2 } ]\tag{104}
$$

$$
= \frac { a ^ { 2 } } { 2 \sigma ^ { 2 } } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ g ( X _ { t } ) ^ { 2 } ]\tag{105}
$$

$$
\leq { \frac { a ^ { 2 } T } { 2 \sigma ^ { 2 } } }\tag{106}
$$

$$
\leq { \frac { 1 } { 8 } } ,\tag{107}
$$

where the first inequality uses $| g ( x ) | \leq 1$ , and the second inequality uses $\begin{array} { r } { a \leq \frac { \sigma } { 2 \sqrt { T } } } \end{array}$ . Under Q, V is independent of $X _ { t }$ . Conditioned on $X _ { t } = x _ { t }$ , the instantaneous regret averaged over the two signs is

$$
\frac { a } { 2 } \bigg ( \operatorname* { m a x } _ { x \in \mathcal { X } } g ( x ) - g ( x _ { t } ) \bigg ) + \frac { a } { 2 } \bigg ( g ( x _ { t } ) - \operatorname* { m i n } _ { x \in \mathcal { X } } g ( x ) \bigg ) = \frac { a \Delta _ { k } } { 2 } .\tag{108}
$$

Summing over $t \in [ T ]$ gives $\begin{array} { r } { \mathbb { E } _ { Q } [ R _ { T } ( \pi , f _ { V } ) ] = \frac { a T \Delta _ { k } } { 2 } } \end{array}$ . For either value of V, the instantaneous regret lies in $[ 0 , a \Delta _ { k } ]$ , such that $0 \leq R _ { T } ( \pi , f _ { V } ) \leq a T \Delta _ { k } \ \mathrm { u n d e } 1$ r both P and Q. Applying Lemma 22 with $H = R _ { T } ( \pi , f _ { V } )$ , $M = a \Delta _ { k } T , \mathrm { e q . ~ } \left( 1 0 7 \right)$ and $\begin{array} { r } { \mathbb { E } _ { Q } [ R _ { T } ( \pi , f _ { V } ) ] = \frac { a T \Delta _ { k } } { 2 } } \end{array}$ gives

$$
\mathbb { E } _ { P } [ R _ { T } ( \pi , f _ { V } ) ] \ge \frac { a T \Delta _ { k } } { 2 } - \frac { a \Delta _ { k } T } { 4 } = \frac { a \Delta _ { k } T } { 4 }\tag{109}
$$

$$
= \frac { \Delta _ { k } } { 8 } \operatorname* { m i n } \{ 2 B T , \sigma \sqrt { T } \} ,\tag{110}
$$

where the last equality follows from the definition of a. This implies the desired lower bound, since $\begin{array} { r } { \operatorname* { s u p } _ { f \in \mathcal { F } _ { k } ( B ) } \mathbb { E } [ R _ { T } ( \pi , f ) ] \ge \mathbb { E } _ { V } \mathbb { E } [ R _ { T } ( \pi , f _ { V } ) ] } \end{array}$ and π is an arbitrary policy. □

The next lemma relates the information gain to the efective dimension. It is related to (Camilleri et al., 2021, Lemma 16), but that result concerns the opposite direction that the maximum information gain is lower bounded by the efective dimension.

Lemma 25. Consider the setup in Section 1.1 where X is finite, except that k may be constant. Then

$$
\frac { 2 \gamma _ { T } } { 1 + \log ( 1 + \rho ^ { - 1 } ) } \leq d _ { T } \leq \rho ^ { - 1 } ,\tag{111}
$$

where $\rho = \sigma ^ { 2 } / ( B ^ { 2 } T )$

Proof. Fix $x _ { 1 } , \dotsc , x _ { T } \in \mathcal { X }$ , and define the empirical distribution $\begin{array} { r } { \hat { \nu } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \delta _ { x _ { t } } } \end{array}$ . Lemma 14 in Section B shows that $K _ { T } ( x _ { 1 : T } )$ and $T M _ { \hat { \nu } }$ have the same positive eigenvalues (including multiplicity), which implies

$$
\operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ) ) = \operatorname* { d e t } _ { E } ( I _ { E } + \rho ^ { - 1 } M _ { \hat { \nu } } ) .\tag{112}
$$

Recalling that $n = \dim ( E )$ , this gives

$$
\log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ) = \log \operatorname* { d e t } _ { E } ( \rho I _ { E } + M _ { \hat { \nu } } ) - n \log \rho ,\tag{113}
$$

and hence

$$
\gamma _ { T } = \frac { 1 } { 2 } \operatorname* { m a x } _ { x _ { 1 : T } \in \mathcal { X } ^ { T } } \log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ) )\tag{114}
$$

$$
= \frac { 1 } { 2 } \bigg ( \operatorname* { m a x } _ { \hat { \nu } } \log \operatorname* { d e t } _ { E } ( \rho I _ { E } + M _ { \hat { \nu } } ) - n \log \rho \bigg )\tag{115}
$$

$$
\leq \frac { 1 } { 2 } \binom  \underset { \nu \in \Delta ( \mathcal { X } ) } { \operatorname* { m a x } } \log \underset { E } { \operatorname* { d e t } } ( \rho I _ { E } + M _ { \nu } ) - n \log \rho \Biggr )\tag{116}
$$

$$
= \frac { 1 } { 2 } \log \operatorname* { d e t } _ { E } ( I _ { E } + \rho ^ { - 1 } M _ { T } ) ,\tag{117}
$$

where the max operation in eq. (115) is over all empirical distributions on X from T queries, and the last equality follows from the definition of $M _ { T }$

Lemma 18 in Section B gives tr $_ { E } ( M _ { T } ) \le 1$ . Then, the definition of $d _ { T }$ in eq. (5) gives $\begin{array} { r } { d _ { T } = \sum _ { i = 1 } ^ { n } \frac { \lambda _ { i } } { \lambda _ { i } + \rho } \leq } \end{array}$ $\rho ^ { - 1 } \textstyle \sum _ { i = 1 } ^ { n } \lambda _ { i } \stackrel { } { = } \rho ^ { - 1 } \mathrm { t r } _ { E } ( M _ { T } ) \le \rho ^ { - 1 }$ , proving the desired upper bound. For later use, we note that the inequality tr $_ { E } ( M _ { T } ) \leq 1$ also implies $\lambda _ { i } \leq 1$ for $i \in [ n ]$

It remains to prove the lower bound. For $i \in [ n ] ,$ define $w _ { i } : = \rho ^ { - 1 } \lambda _ { i }$ , such that $0 \leq w _ { i } \leq \rho ^ { - 1 }$ . Then, the formula $\begin{array} { r } { d _ { T } = \sum _ { i = 1 } ^ { n } \frac { \lambda _ { i } } { \lambda _ { i } + \rho } } \end{array}$ equivalently gives $\begin{array} { r } { d _ { T } = \sum _ { i = 1 } ^ { n } \frac { w _ { i } } { 1 + w _ { i } } } \end{array}$ . For $w > 0$ , define $\begin{array} { r } { q ( w ) : = \frac { 1 + w } { w } \log ( 1 + w ) } \end{array}$ , with $q ( 0 ) : = 1$ . Since $\begin{array} { r } { q ^ { \prime } ( w ) = \frac { w - \log ( 1 + w ) } { w ^ { 2 } } > 0 } \end{array}$ , it follows that $q ( \cdot )$ is strictly increasing. Then, eq. (117) implies that

$$
\gamma _ { T } \leq { \frac { 1 } { 2 } } \sum _ { i = 1 } ^ { n } \log ( 1 + w _ { i } ) = { \frac { 1 } { 2 } } \sum _ { i = 1 } ^ { n } { \frac { w _ { i } } { 1 + w _ { i } } } q ( w _ { i } )\tag{118}
$$

$$
\leq \frac { q ( \rho ^ { - 1 } ) } { 2 } \sum _ { i = 1 } ^ { n } \frac { w _ { i } } { 1 + w _ { i } } = \frac { q ( \rho ^ { - 1 } ) d _ { T } } { 2 } .\tag{119}
$$

Since $\log ( 1 + w ) \leq w$ , it follows that $\begin{array} { r } { q ( w ) = \frac { 1 } { w } \log ( 1 + w ) + \log ( 1 + w ) \leq 1 + \log ( 1 + w ) } \end{array}$ . Then, eq. (119) gives

$$
d _ { T } \geq \frac { 2 \gamma _ { T } } { 1 + \log ( 1 + \rho ^ { - 1 } ) } ,\tag{120}
$$

which proves the desired lower bound.

## Completion of the Proof of Theorem 1 and Corollary 1

Proof of Theorem 1. We split into two cases for $d _ { T }$

Case 1: $d _ { T } \geq 4$ . In this case, Lemma 1 directly gives $\begin{array} { r } { R _ { T } ^ { * } \geq \frac { \sigma } { 1 6 } \sqrt { T d _ { T } } } \end{array}$

Case 2: $d _ { T } < 4 .$ . In this case, $\begin{array} { r } { \sigma \sqrt { T } \geq \frac { \sigma } { 2 } \sqrt { T d _ { T } } } \end{array}$ . Moreover, the upper bound in Lemma $2 5$ and $\rho = \sigma ^ { 2 } / ( B ^ { 2 } T )$ implies $\sigma \sqrt { T d _ { T } } \le B T$ , which we can crudely weaken to $2 B T \geq { \textstyle { \frac { \sigma } { 2 } } } { \sqrt { T d _ { T } } }$ . Applying Lemma 24 gives

$$
R _ { T } ^ { * } \geq \frac { \Delta _ { k } } { 8 } \operatorname* { m i n } \{ 2 B T , \sigma \sqrt { T } \} \geq \frac { \Delta _ { k } \sigma } { 1 6 } \sqrt { T d _ { T } } .\tag{121}
$$

Proof of Corollary 1. Combining Theorem 1 with the lower bound in Lemma 25 proves Corollary 1.

## D Proofs of Theorem 2 and Corollary 2 (General Compact Domains)

Recall that we fix a set $\tilde { \mathcal X }$ satisfying Proposition 1, and that $d _ { T } ( \tilde { \mathcal { X } } )$ is computed on $\tilde { \mathcal { X } } .$ . For convenience we write $d _ { T } = d _ { T } ( \tilde { \mathcal { X } } )$ within the proofs.

We extend $M _ { T } : E \to E$ to $M : \mathcal { H } _ { k }  \mathcal { H } _ { k }$ such that $M | _ { E } = M _ { T }$ and $M v = 0$ for $v \in E ^ { \perp }$ . We define $I _ { k }$ as the identity map on $\mathcal { H } _ { k }$ . It follows that $M + \rho I _ { k }$ is positive definite (similar to Lemma 9). Also notice that

$$
0 < 2 \gamma _ { T } ( \mathcal { X } ) = 2 \gamma _ { T } ( \tilde { \mathcal { X } } ) \leq \log \operatorname* { d e t } _ { E } ( I _ { E } + \rho ^ { - 1 } M _ { T } ) ,\tag{122}
$$

where the first inequality follows because k is non-constant on $x ,$ the equality follows from Proposition 1, the second inequality follows from $\mathrm { e q . ~ } \left( 4 \right)$ and Lemma 14. Then $M _ { T }$ has a positive eigenvalue, and thus $d _ { T } > 0$ . Then, we generalize the definition of $u _ { x }$ in eq. (7), now defined for $x \in \mathcal { X }$ as

$$
u _ { x } : = d _ { T } ^ { - 1 / 2 } ( M + \rho I _ { k } ) ^ { - 1 / 2 } \phi ( x ) \in \mathcal { H } _ { k } .\tag{123}
$$

Applying the decomposition $\mathcal { H } _ { k } = E \oplus E ^ { \perp }$ to $( M + \rho I _ { k } ) ^ { - 1 / 2 }$ gives $( M + \rho I _ { k } ) ^ { - 1 / 2 } = ( M _ { T } + \rho I _ { E } ) ^ { - 1 / 2 } \oplus$ $\rho ^ { - 1 / 2 } I _ { E ^ { \perp } }$ . It follows that eq. (123) agrees with eq. (7) on $\tilde { \mathcal X }$ . Consequently, we can apply Lemma 23 with the new definition of $u _ { x }$ , where eq. (83) now reads

$$
\mathbb { E } [ u _ { Z } \otimes u _ { Z } ] \preceq d _ { T } ^ { - 1 } I _ { k } .\tag{124}
$$

We start by proving Proposition 1.

Proof of Proposition 1. We first prove eq. (32). Since X is compact and the mapping $x \mapsto \phi ( x )$ is continuous (see Lemma 19), it follows that $\phi ( \mathcal { X } ) : = \{ \phi ( x ) : x \in \mathcal { X } \} \subseteq \mathcal { H } _ { k }$ is compact. We fix $\delta : = \sigma / ( 4 B \sqrt { T } ) > 0$ , and consider $\textstyle \bigcup _ { \phi ( x ) \in \phi ( { \mathcal { X } } ) } B ( \phi ( x ) , \delta )$ , where $B ( \phi ( x ) , \delta ) : = \{ \phi ( x ^ { \prime } ) \in \phi ( \mathcal { X } ) : \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } < \delta \}$ . It follows that $\textstyle \bigcup _ { \phi ( x ) \in \phi ( { \mathcal { X } } ) } { \dot { B } } { \big ( } \phi ( x ) , \delta { \big ) }$ is an open cover of $\phi ( \mathcal { X } )$ . Compactness implies that there exists a finite subcover, i.e., there exists $\tilde { \mathcal { X } } \subseteq \mathcal { X }$ that is finite and satisfies

$$
\phi ( { \mathcal { X } } ) \subseteq \bigcup _ { x \in { \tilde { \mathcal { X } } } } B ( \phi ( x ) , \delta ) .\tag{125}
$$

It remains to show that $\tilde { \mathcal X }$ can be chosen to satisfy $\gamma _ { T } ( \tilde { \mathcal { X } } ) = \gamma _ { T } ( \mathcal { X } )$ . We consider $x _ { 1 } ^ { * } , \ldots , x _ { T } ^ { * }$ , which is a maximizing tuple of $\gamma _ { T } ( \mathcal { X } )$ in eq. (31), and enlarge $\tilde { \mathcal X }$ to contain this tuple. Then, $\begin{array} { r } { \gamma _ { T } ( \mathcal { X } ) = \frac { 1 } { 2 } } \end{array}$ log det $( I _ { T } +$ $\lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ^ { * } ) ) \le \gamma _ { T } ( \tilde { \mathcal { X } } )$ , while $\tilde { \mathcal { X } } \subseteq \mathcal { X }$ gives $\gamma _ { T } ( \tilde { \mathcal { X } } ) \leq \gamma _ { T } ( \mathcal { X } )$ . Consequently, $\gamma _ { T } ( \mathcal { X } ) = \gamma _ { T } ( \tilde { \mathcal { X } } )$ □

The following lemma generalizes eq. (81).

Lemma 26. Under the setup in Section 3.1, for every $x \in \mathcal { X }$

$$
\| u _ { x } \| _ { k } \leq 1 + \frac { 1 } { 4 \sqrt { d _ { T } ( \tilde { \mathcal { X } } ) } } .\tag{126}
$$

Consequently, $i f d _ { T } ( \tilde { \mathcal { X } } ) \geq 4$ , then $\| u _ { x } \| _ { k } \leq 9 / 8$

Proof. Fix $x \in \mathcal { X }$ . Since $\tilde { \mathcal X }$ satisfies $\mathrm { e q . ~ } \left( 3 2 \right)$ , there exists $\boldsymbol { x ^ { \prime } } \in \tilde { \mathcal { X } }$ such that $\begin{array} { r } { \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } \le \frac { \sigma } { 4 B \sqrt { T } } } \end{array}$ . For an operator V on $\mathcal { H } _ { k } .$ , define the operator norm $\| V \| _ { \mathrm { o p } } : = \operatorname* { s u p } \{ \| V f \| _ { k } : f \in \mathcal { H } _ { k } , \| f \| _ { k } = 1 \}$ . Observe that $M + \rho I _ { k } \succeq \rho I _ { k }$ . Then, for every $f \in \mathcal { H } _ { k }$ with $\| f \| _ { k } = 1$

$$
\begin{array} { r } { \| ( M + \rho I _ { k } ) ^ { - 1 / 2 } f \| _ { k } ^ { 2 } = \langle f , ( M + \rho I _ { k } ) ^ { - 1 } f \rangle _ { k } \leq \rho ^ { - 1 } \| f \| _ { k } ^ { 2 } , } \end{array}\tag{127}
$$

which implies $\| ( M + \rho I _ { k } ) ^ { - 1 / 2 } \| _ { \mathrm { o p } } \leq \rho ^ { - 1 / 2 }$ . Consequently,

$$
| | u _ { x } | | _ { k } - | | u _ { x ^ { \prime } } | | _ { k } = | | d _ { T } ^ { - 1 / 2 } ( M + \rho I _ { k } ) ^ { - 1 / 2 } \phi ( x ) | | _ { k } - | | d _ { T } ^ { - 1 / 2 } ( M + \rho I _ { k } ) ^ { - 1 / 2 } \phi ( x ^ { \prime } ) | | _ { k }\tag{128}
$$

$$
\leq d _ { T } ^ { - 1 / 2 } \| ( M + \rho I _ { k } ) ^ { - 1 / 2 } ( \phi ( x ) - \phi ( x ^ { \prime } ) ) \| _ { k }\tag{129}
$$

$$
\leq d _ { T } ^ { - 1 / 2 } \| ( M + \rho I _ { k } ) ^ { - 1 / 2 } \| _ { \mathrm { o p } } \cdot \| ( \phi ( x ) - \phi ( x ^ { \prime } ) ) \| _ { k }\tag{130}
$$

$$
\leq d _ { T } ^ { - 1 / 2 } \rho ^ { - 1 / 2 } \cdot \frac { \sigma } { 4 B \sqrt { T } }\tag{131}
$$

$$
= \frac { 1 } { 4 \sqrt { d _ { T } } } ,\tag{132}
$$

Since $x ^ { \prime } \in \tilde { \mathcal { X } } , \mathrm { e q . }$ (81) gives $\| u _ { x ^ { \prime } } \| _ { k } \leq 1$ . Rearranging proves the result.

Next, we provide the lower bound on regret for $d _ { T } \geq 4 .$ , generalizing Lemma 1.

Lemma 27. Under the setup in Section 3.1, if $d _ { T } ( \tilde { \mathcal { X } } ) \geq 4$ , then

$$
R _ { T } ^ { * } \geq \frac { \sigma } { 3 2 } \sqrt { T d _ { T } ( \tilde { \mathcal { X } } ) } .\tag{133}
$$

Proof. The proof is similar to Lemma 1 in structure, and the main diferences are in numerical constants, so we omit details where appropriate.

Step 1: Construct hard functions. For every $z \in \mathrm { s u p p } ( \nu _ { T } )$ , define

$$
f _ { z } : = \frac { B } { 6 } \sqrt { \frac { \rho } { d _ { T } } } ( M _ { T } + \rho I _ { E } ) ^ { - 1 } \phi ( z ) \in \mathcal { H } _ { k } .\tag{134}
$$

It follows that $\begin{array} { r } { \| f _ { z } \| _ { k } ^ { 2 } \le \frac { B ^ { 2 } } { 3 6 } \le B ^ { 2 } } \end{array}$ , and hence $f _ { z } \in \mathcal { F } _ { k } ( B )$ . In addition, observe that

$$
f _ { z } ( x ) = \frac { B } { 6 } \sqrt { \rho d _ { T } } \langle u _ { z } , u _ { x } \rangle _ { k } .\tag{135}
$$

Combining Lemma 26 and eqs. (81) and (82) gives

$$
f _ { z } ( z ) = \frac { B } { 6 } \sqrt { \rho d _ { T } } , \qquad | f _ { z } ( x ) | \leq \frac { 3 B } { 1 6 } \sqrt { \rho d _ { T } } \qquad \forall x \in \mathcal { X } .\tag{136}
$$

Step 2: Change of measure. To use Lemma 22, we first bound the KL divergence. Let $H _ { t } : =$ $\left( X _ { 1 } , Y _ { 1 } , \ldots , X _ { t } , Y _ { t } \right)$ be the history until time t. Fix a policy $\pi .$ Let P be the joint law on $( Z , H _ { T } )$ obtained by first drawing $Z \sim \nu _ { T }$ , then running π against $f _ { Z }$ . Let $Q$ be the joint law on $( Z , H _ { T } )$ when the policy is instead run on the zero function, and $Z \sim \nu _ { T }$ is drawn independently from the interaction. Then

$$
D _ { \mathrm { K L } } ( Q \| P ) = \frac { 1 } { 2 \sigma ^ { 2 } } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ f _ { Z } ( X _ { t } ) ^ { 2 } ]\tag{137}
$$

$$
\mathbf { \Psi } = \frac { 1 } { 2 \sigma ^ { 2 } } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } [ \mathbb { E } _ { Z } [ f _ { Z } ( X _ { t } ) ^ { 2 } ] ] \mathbf { \Psi }\tag{138}
$$

$$
\leq \frac { T } { 2 \sigma ^ { 2 } } \cdot \frac { 9 B ^ { 2 } \rho } { 2 5 6 }\tag{139}
$$

$$
{ \overline { { 5 1 2 } } }  '\tag{140}
$$

where eq. (137) follows from the chain rule for KL divergence, eq. (138) follows because the random variables $Z$ and $X _ { t }$ are independent under Q, eq. (139) is explained below, and eq. (140) uses $\rho = \sigma ^ { 2 } / ( B ^ { 2 } T )$ . To verify eq. (139), observe that for every $\begin{array} { r } { x \in \mathcal { X } , \mathbb { E } _ { Z } [ f _ { Z } ( x ) ^ { 2 } ] = \frac { B ^ { 2 } \rho d _ { T } } { 3 6 } \langle u _ { x } , \mathbb { E } _ { Z } [ u _ { Z } \otimes u _ { Z } ] u _ { x } \rangle _ { k } \leq \frac { B ^ { 2 } \rho } { 3 6 } \langle u _ { x } , u _ { x } \rangle _ { k } \leq \frac { 9 B ^ { 2 } \rho } { 2 5 6 } } \end{array}$ where the first inequality follows from eq. (124), and the second inequality follows from Lemma 26.

Next, we proceed to bound $\mathbb { E } _ { Q } [ R _ { T } ( \pi , f _ { Z } ) ]$ . For any $x \in \mathcal { X }$ , we have

$$
\mathbb { E } _ { Z } [ f _ { Z } ( x ) ] = \frac { B \sqrt { \rho d _ { T } } } { 6 } \langle \mathbb { E } _ { Z } [ u _ { Z } ] , u _ { x } \rangle _ { k } \leq \frac { B \sqrt { \rho } } { 6 } \| u _ { x } \| _ { k } \leq \frac { 3 B \sqrt { \rho } } { 1 6 } ,\tag{141}
$$

where the equality follows from eq. (135), the first inequality follows from eq. (84) and Cauchy-Schwarz, and the second inequality follows from Lemma 26. The independence of Z and $X _ { t }$ under Q implies $\mathbb { E } _ { Q } [ f _ { Z } ( X _ { t } ) ] =$ $\mathbb { E } _ { Q } [ \mathbb { E } _ { Z } [ f _ { Z } ( X _ { t } ) ] ]$ , which combined with eqs. (136) and (141) and $d _ { T } \geq 4$ gives

$$
\mathbb { E } _ { Q } [ R _ { T } ( \pi , f _ { Z } ) ] = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { Q } \left[ \operatorname* { m a x } _ { x } f _ { Z } ( x ) - f _ { Z } ( X _ { t } ) \right] \geq B T \left( \frac { \sqrt { \rho d _ { T } } } { 6 } - \frac { 3 \sqrt { \rho } } { 1 6 } \right) \geq \frac { 7 T B \sqrt { \rho d _ { T } } } { 9 6 } .\tag{142}
$$

Finally, $\mathrm { e q . }$ . (136) implies $\begin{array} { r } { 0 \leq R _ { T } ( \pi , f _ { Z } ) \leq \frac { 3 B T \sqrt { \rho d _ { T } } } { 8 } } \end{array}$ . Applying Lemma 22 with $H = R _ { T } ( \pi , f _ { Z } )$ $M =$ $\frac { 3 B T \sqrt { \rho d _ { T } } } { 8 }$ , and eqs. (140) and (142) gives

$$
\mathbb { E } _ { P } [ R _ { T } ( \pi , f _ { Z } ) ] \ge \frac { 7 T B \sqrt { \rho d _ { T } } } { 9 6 } - \frac { 3 B T \sqrt { \rho d _ { T } } } { 8 } \cdot \frac { 3 } { 3 2 } = \frac { 2 9 \sigma } { 3 2 \cdot 2 4 } \sqrt { T d _ { T } } \ge \frac { \sigma } { 3 2 } \sqrt { T d _ { T } } ,\tag{143}
$$

which proves Lemma 27 since sup $f { \in } \mathcal { F } _ { k } ( B ) \operatorname { \mathbb { E } } \bigl [ R _ { T } ( \pi , f ) \bigr ] \geq \operatorname { \mathbb { E } } _ { Z \sim \nu _ { T } } \operatorname { \mathbb { E } } \bigl [ R _ { T } ( \pi , f _ { Z } ) \bigr ]$ and the policy π is arbitrary.

Lemma 28. Under the setup of Section 3.1,

$$
\frac { 2 \gamma _ { T } ( \mathcal X ) } { 1 + \log ( 1 + \rho ^ { - 1 } ) } \le d _ { T } ( \tilde { \mathcal X } ) \le \rho ^ { - 1 } ,\tag{144}
$$

where $\rho = \sigma ^ { 2 } / ( B ^ { 2 } T )$

Proof. Applying Lemma 25 to $\tilde { \mathcal X }$ (k may be constant on $\tilde { \mathcal X } \times \tilde { \mathcal X }$ , but that is allowed in Lemma 25) proves $\begin{array} { r } { \frac { 2 \gamma _ { T } ( \tilde { \mathcal { X } } ) } { 1 + \log ( 1 + \rho ^ { - 1 } ) } \leq d _ { T } \leq \rho ^ { - 1 } } \end{array}$ , which combined with $\gamma _ { T } ( { \tilde { \mathcal { X } } } ) = \gamma _ { T } ( { \mathcal { X } } )$ from Proposition 1 proves the result.

Completion of the Proof of Theorem 2 and Corollary 2

Proof of Theorem 2. We again split into two cases for $d _ { T }$

Case 1: $d _ { T } \geq 4$ . In this case, Lemma 27 directly shows that $R _ { T } ^ { * } \ge \frac { \sigma } { 3 2 } \sqrt { T d _ { T } }$

Case 2: $d _ { T } < 4$ . Under the setup in Section 3.1, $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$ and the $x , x ^ { \prime }$ attaining $\Delta _ { k }$ are justified (see Lemma 21), such that Lemma 24 holds for compact X. Then $d _ { T } <$ 4 implies $\begin{array} { r } { \sigma \sqrt { T } \geq \frac { \sigma } { 2 } \sqrt { T d _ { T } } . } \end{array}$ and the upper bound in Lemma 28 implies $B T \geq \sigma { \sqrt { T d _ { T } } }$ , such that $2 B T \geq { \textstyle { \frac { 1 } { 2 } } } \sigma { \sqrt { T d _ { T } } }$ . Then, applying Lemma 24 gives $\begin{array} { r } { R _ { T } ^ { * } \geq \frac { \Delta _ { k } \sigma } { 1 6 } \sqrt { T d _ { T } } } \end{array}$ □

Proof of Corollary 2. Combining Theorem 2 and Lemma 28 proves Corollary 2.

## D.1 High-Probability Lower Bound

While we have focused on the average regret, we can readily adapt our arguments to obtain a high-probability lower bound, which is stated as follows.

Corollary 3. Consider the setup of Section 3.1 where X is compact. For every randomized policy π, there exists a fixed $f \in { \mathcal { F } } _ { k } ( B )$ , such that with probability at least $1 / 4$

$$
R _ { T } ( \pi , f ) \geq \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sqrt { 2 } \sigma } { 3 2 } \sqrt { \frac { T \gamma _ { T } ( \mathcal X ) } { 1 + \log ( 1 + B ^ { 2 } T / \sigma ^ { 2 } ) } } ,\tag{145}
$$

where $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$

Proof. We will first prove a high-probability lower bound expressed using $d _ { T } = d _ { T } ( \tilde { \mathcal { X } } )$ , then transfer to $\gamma _ { T } ( \mathcal { X } )$ using Lemma 28. We consider two cases.

Case 1: $d _ { T } \geq 4$ . We follow the setup in Lemma 27: we consider the hard functions $\{ f _ { z } \}$ , where $z \in \mathrm { s u p p } ( \nu _ { T } )$ We fix some policy $\pi ,$ and consider probability measures $P , Q ,$ , which are joint laws on $( Z , H _ { T } )$ where $Z \sim \nu _ { T }$ and $H _ { T }$ is the history until time $T$ obtained by running π against $f _ { Z }$ . Specifically, $P$ is obtained by drawing $Z \sim \nu _ { T }$ and running π against $f _ { Z }$ , and $Q$ is obtained by drawing $Z$ independently and running π against the zero function. For convenience we define $S : = \sqrt { \sigma ^ { 2 } T d _ { T } }$

By eq. (136), $\begin{array} { r } { f _ { Z } ( Z ) = \frac { B } { 6 } \sqrt { \rho d _ { T } } } \end{array}$ , which yields

$$
R _ { T } ( \pi , f _ { Z } ) \geq \frac { B T } { 6 } \sqrt { \rho d _ { T } } - \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) = \frac { S } { 6 } - \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) .\tag{146}
$$

Moreover, by eq. (135), $\begin{array} { r } { f _ { z } ( x ) = \frac { B } { 6 } \sqrt { \rho d _ { T } } \langle u _ { z } , u _ { x } \rangle _ { k } } \end{array}$ , which gives $\begin{array} { r } { \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) = \frac { B } { 6 } \sqrt { \rho d _ { T } } \langle u _ { Z } , \sum _ { t = 1 } ^ { T } u _ { X _ { t } } \rangle _ { k } } \end{array}$ . We

then observe that

$$
\mathbb { E } _ { Q } \left[ \left( \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) \right) ^ { 2 } \bigg | X _ { 1 : T } \right] = \frac { B ^ { 2 } \rho d _ { T } } { 3 6 } \mathbb { E } _ { Q } \left[ \bigg \langle \sum _ { t = 1 } ^ { T } u _ { X _ { t } } , u _ { Z } \otimes u _ { Z } \sum _ { t = 1 } ^ { T } u _ { X _ { t } } \bigg \rangle _ { k } \bigg | X _ { 1 : T } \right]\tag{147}
$$

$$
\mathbf { \Sigma } = \frac { B ^ { 2 } \rho d _ { T } } { 3 6 } \biggl \langle \sum _ { t = 1 } ^ { T } u _ { X _ { t } } , \mathbb { E } [ u _ { Z } \otimes u _ { Z } ] \sum _ { t = 1 } ^ { T } u _ { X _ { t } } \biggr \rangle _ { k } ,\tag{148}
$$

$$
\leq \frac { B ^ { 2 } \rho } { 3 6 } \biggl \| \sum _ { t = 1 } ^ { T } u _ { X _ { t } } \biggr \| _ { k } ^ { 2 } ,\tag{149}
$$

$$
\leq \frac { B ^ { 2 } \rho } { 3 6 } \cdot T \sum _ { t = 1 } ^ { T } \lVert u _ { X _ { t } } \rVert _ { k } ^ { 2 }\tag{150}
$$

$$
\leq { \frac { 9 \sigma ^ { 2 } T } { 2 5 6 } }\tag{151}
$$

$$
\leq { \frac { 9 S ^ { 2 } } { 1 0 2 4 } } ,\tag{152}
$$

where eq. (148) follows because $Z$ is independent of the zero-function interaction under Q, eq. (149) follows from eq. (124), eq. (150) applies Cauchy-Schwarz, eq. (151) follows from combining Lemma 26 and $\rho =$ $\sigma ^ { 2 } / ( B ^ { 2 } T )$ , and eq. (152) applies the definition of S and $d _ { T } \geq 4$

We now combine the above findings to lower bound the regret. First observe that if $\textstyle \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) \leq { \frac { 1 3 S } { 9 6 } }$ then eq. (146) gives $\begin{array} { r } { R _ { T } ( \pi , f _ { Z } ) \ge \frac { S } { 3 2 } } \end{array}$ . Moreover, applying Markov’s inequality to $\textstyle { \bigl ( } \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) { \bigr ) } ^ { 2 }$ gives

$$
\mathbb { P } _ { Q } \big ( R _ { T } ( \pi , f _ { Z } ) \ge S / 3 2 \big ) \ge \mathbb { P } _ { Q } \bigg ( \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) \le \frac { 1 3 S } { 9 6 } \bigg ) \ge 1 - \mathbb { P } _ { Q } \bigg ( \bigg ( \sum _ { t = 1 } ^ { T } f _ { Z } ( X _ { t } ) \bigg ) ^ { 2 } \ge \bigg ( \frac { 1 3 S } { 9 6 } \bigg ) ^ { 2 } \bigg ) \ge 1 - \frac { 8 1 } { 1 6 9 } = \frac { 8 8 } { 1 6 9 } .\tag{153}
$$

Recall that eq. (140) gives $\begin{array} { r } { D _ { \mathrm { K L } } ( Q \Vert P ) \leq \frac { 9 } { 5 1 2 } } \end{array}$ . Then, applying Lemma 22 with $H = \mathbb { I } _ { \{ R _ { T } ( \pi , f z ) \geq S / 3 2 \} }$ and $M = 1 { \mathrm { ~ g i v e s } }$

$$
\mathbb { P } _ { P } ( R _ { T } ( \pi , f _ { Z } ) \ge S / 3 2 ) \ge \mathbb { P } _ { Q } ( R _ { T } ( \pi , f _ { Z } ) \ge S / 3 2 ) - \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } ( Q \| P ) } \ge \frac { 8 8 } { 1 6 9 } - \frac { 3 } { 3 2 } = \frac { 2 3 0 9 } { 5 4 0 8 } \ge \frac { 1 } { 4 } .\tag{154}
$$

Then, since $S / 3 2 = \sqrt { \sigma ^ { 2 } T d _ { T } } / 3 2 \geq \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sigma } { 3 2 } \sqrt { T d _ { T } }$ , we have

$$
P \bigg ( R _ { T } ( \pi , f _ { Z } ) \geq \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sigma } { 3 2 } \sqrt { T d _ { T } } \bigg ) \geq \frac { 1 } { 4 } .\tag{155}
$$

This holds for $Z \sim \nu _ { T }$ , and it follows that there exists some fixed $z \in \mathrm { s u p p } ( \nu _ { T } )$ such that $R _ { T } ( \pi , f _ { z } ) \geq$ min $\begin{array} { r } { \left\{ 1 , 2 \Delta _ { k } \right\} \cdot \frac { \sigma } { 3 2 } \sqrt { T d _ { T } } } \end{array}$ with probability at least $1 / 4$

Case 2: $0 < d _ { T } < 4$ . Here we follow the setup in Lemma 24. We define a := min $\{ B , \sigma / ( 2 \sqrt { T } ) \}$ , and consider the hard functions $f _ { v } : = v a \cdot g$ , where $v \in \{ - 1 , + 1 \}$ and g is defined in eq. (101). We again fix some policy $\pi ,$ and consider probability measures $P , Q ,$ which are joint laws on $( V , H _ { T } )$ where V is uniform on $\{ - 1 , + 1 \}$ and $H _ { T }$ is the history until time T obtained by running π against $f _ { V }$ . Specifically, P is obtained by drawing V and running π against $f _ { V }$ , and Q is obtained by drawing V independently and running π against the zero function. We define $\begin{array} { r } { R _ { + } : = \sum _ { t = 1 } ^ { T } ( \operatorname* { m a x } _ { x \in \mathcal { X } } f _ { + 1 } ( x ) - f _ { + 1 } ( x _ { t } ) ) } \end{array}$ ) and $\begin{array} { r } { R _ { - } : = \sum _ { t = 1 } ^ { T } ( \operatorname* { m a x } _ { x \in \mathcal { X } } f _ { - 1 } ( x ) - f _ { - 1 } ( x _ { t } ) ) } \end{array}$ ), and consider any fixed realization $x _ { 1 } , \dotsc , x _ { T } \in \mathcal { X }$

Recall that $\begin{array} { r } { \Delta _ { k } : = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { k } } \end{array}$ and the proof of Lemma 24 gives ma $\begin{array} { r } { \mathfrak { c } _ { x \in \mathcal { X } } g ( x ) - \operatorname* { m i n } _ { x \in \mathcal { X } } g ( x ) = } \end{array}$ $\Delta _ { k }$ . We observe that

$$
R _ { + } + R _ { - } = a \sum _ { t = 1 } ^ { T } \biggl ( \operatorname* { m a x } _ { x \in \mathcal { X } } g ( x ) - g ( x _ { t } ) + g ( x _ { t } ) - \operatorname* { m i n } _ { x \in \mathcal { X } } g ( x ) \biggr ) = a T \Delta _ { k } ,\tag{156}
$$

which implies that at least one of $R _ { + } , R _ { - }$ is at least $a T \Delta _ { k } / 2$ . Hence,

$$
\mathbb { P } _ { Q } ( R _ { T } ( \pi , f _ { V } ) \geq a T \Delta _ { k } / 2 | X _ { 1 : T } ) \geq \frac { 1 } { 2 } .\tag{157}
$$

Recall that eq. (107) gives $\begin{array} { r } { D _ { \mathrm { K L } } ( Q \| P ) \leq \frac { 1 } { 8 } } \end{array}$ . Then, applying Lemma 22 with $H = \mathbb { I } _ { \{ R _ { T } ( \pi , f _ { V } ) \geq a T \Delta _ { k } / 2 \} }$ and $M = 1$ gives

$$
\mathbb { P } _ { P } ( R _ { T } ( \pi , f _ { V } ) \ge a T \Delta _ { k } / 2 ) \ge \frac { 1 } { 2 } - \sqrt { \frac { 1 } { 2 } \cdot \frac { 1 } { 8 } } = \frac { 1 } { 4 } .\tag{158}
$$

It remains to show that $a T \Delta _ { k } / 2 \geq \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sigma } { 3 2 } \sqrt { T d _ { T } }$ . Since $d _ { T } < 4 .$ , we have $\textstyle { \sigma { \sqrt { T } } } / 2 \geq { \frac { \sigma } { 4 } } { \sqrt { T d _ { T } } }$ . Also, Lemma 25 gives $\begin{array} { r } { d _ { T } \leq \rho ^ { - 1 } = \frac { B ^ { 2 } T } { \sigma ^ { 2 } } } \end{array}$ , such that $B T \geq \sigma { \sqrt { T d _ { T } } }$ . Combining gives $\begin{array} { r } { a T \geq \frac { \sigma } { 4 } \sqrt { T d _ { T } } } \end{array}$ , such that

$$
a T \Delta _ { k } / 2 \geq \frac { \sigma \Delta _ { k } } { 8 } \sqrt { T d _ { T } } \geq \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sigma } { 3 2 } \sqrt { T d _ { T } } .\tag{159}
$$

Since V is uniform on $\{ - 1 , + 1 \}$ , there exists some fixed $v \in \{ - 1 , + 1 \}$ such that $R _ { T } ( \pi , f _ { v } ) \ge \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \}$ $\frac { \sigma } { 3 2 } \sqrt { T d _ { T } }$ with probability at least $1 / 4$

Finally, combining the two cases proves that for every policy $\pi ,$ , there exists $f \in { \mathcal { F } } _ { k } ( B )$ , such that with probability at least $1 / 4$

$$
R _ { T } ( \pi , f ) \ge \operatorname* { m i n } \{ 1 , 2 \Delta _ { k } \} \cdot \frac { \sigma } { 3 2 } \sqrt { T d _ { T } } ,\tag{160}
$$

which combined with Lemma 28 proves the result.

## E Omitted Proofs from Section 4

## E.1 Deriving Sharp Rates for $\gamma _ { T }$

We recall that $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . For kernels from three spectral regimes, (Janz et al., 2026) presents sharp rates for the maximum information gain, which they define as

$$
\Gamma _ { T } ^ { k } ( \mathcal { X } ; \eta ) : = \operatorname* { s u p } _ { x _ { 1 : T } \in \mathcal { X } ^ { T } } \log \operatorname* { d e t } ( I _ { T } + \eta ^ { - 1 } K _ { T } ( x _ { 1 : T } ) ) ,\tag{161}
$$

where $1 \leq \eta \leq T$ is fixed. Recall from eqs. (6) and (31) that our maximum information gain is defined as $\begin{array} { r } { \gamma _ { T } = \gamma _ { T } ( \pmb { \chi } ) = \frac { 1 } { 2 } \operatorname* { m a x } _ { x _ { 1 : T } \in \pmb { \chi } ^ { T } } } \end{array}$ log det $( I _ { T } + \lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ) )$ . It follows that $\begin{array} { r } { \gamma _ { T } = \frac { 1 } { 2 } \Gamma _ { T } ^ { k } ( \mathcal X ; \lambda ) } \end{array}$ . It is possible that $0 < \lambda < 1$ . In this case, we observe that log $( 1 + u ) \le \log ( 1 + u / \lambda ) \le \lambda ^ { - 1 } \log ( \overline { { \mathfrak { s } } } ( 1 + u )$ for every $u \geq 0$ Applying these inequalities to eq. (161) gives $\Gamma _ { T } ^ { k } ( \mathcal { X } ; 1 ) \leq \Gamma _ { T } ^ { k } ( \mathcal { \dot { X } } ; \lambda ) \leq \lambda ^ { - 1 } \Gamma _ { T } ^ { k } ( \mathcal { X } ; \bar { 1 } )$ . Consequently, we can directly apply the asymptotic bounds of $\Gamma _ { T } ^ { k } ( \mathcal { X } ; \bar { \eta } )$ in (Janz et al., 2026).

We now show how to get each entry in the second column of Table 1. For the Mat´ern and squared exponential kernels, their rates directly come from (Janz et al., 2026, Table 2). For the γ-exponential and piecewisepolynomial kernels, (Lee and Javidi, 2025, Proposition 1) shows their Fourier transforms have polynomial decay, which combined with the second row of (Janz et al., 2026, Table 1) gives their respective rates. For the Dirichlet kernel, the upper bound is from (Lee and Javidi, 2025, Table 2), and Lemma 15 gives $\Omega ( \log T )$

For the rational quadratic kernel, some extra efort is needed, and we provide the following proposition.

Proposition 3. Consider the setup of Section 3.1 with $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . For the rational quadratic kernel, we have

$$
\gamma _ { T } = \Theta ( ( \log T ) ^ { d + 1 } ) .\tag{162}
$$

Proof. For $x , x ^ { \prime } \in { \mathcal { X } } .$ , define $h : = x - x ^ { \prime }$ . Following (Janz et al., 2026), we define $s ( \xi )$ to be the spectral density related to the rational quadratic kernel, such that

$$
k _ { \mathrm { R Q } , a } ( h ) = \int _ { { \mathbb R } ^ { d } } \exp ( 2 \pi i \xi \cdot h ) s ( \xi ) \ d \xi .\tag{163}
$$

Step 1: Deriving the rate of $s ( \xi )$ . This step is similar to (Lee and Javidi, 2025, pp. 21–22), but extends to d dimensions. For $u > 0$ , the Gamma function satisfies $\begin{array} { r } { \int _ { 0 } ^ { \infty } t ^ { a - 1 } \exp ( - u t ) \ d t = \bar { u ^ { - a } } \Gamma ( a ) } \end{array}$ . Then

$$
k _ { \mathrm { R Q } , a } ( h ) = \left( 1 + { \frac { \| h \| _ { 2 } ^ { 2 } } { 2 a \ell ^ { 2 } } } \right) ^ { - a } = { \frac { 1 } { \Gamma ( a ) } } \int _ { 0 } ^ { \infty } t ^ { a - 1 } \exp ( - t ) \exp \left( - { \frac { t \| h \| _ { 2 } ^ { 2 } } { 2 a \ell ^ { 2 } } } \right) d t .\tag{164}
$$

In d dimensions, the Fourier transform of a Gaussian is again a Gaussian (e.g., see (Rasmussen and Williams, 2005, eq. (4.6) and discussion after eq. (4.9))):

$$
\int _ { \mathbb { R } ^ { d } } \exp \left( - \frac { t \| h \| _ { 2 } ^ { 2 } } { 2 a \ell ^ { 2 } } \right) \exp ( - 2 \pi i \xi \cdot h ) \ d h = \left( \frac { 2 \pi a \ell ^ { 2 } } { t } \right) ^ { d / 2 } \exp \left( - \frac { 2 \pi ^ { 2 } a \ell ^ { 2 } \| \xi \| _ { 2 } ^ { 2 } } { t } \right) ,\tag{165}
$$

where the right-hand side is the spectral density of the Gaussian in eq. (164), which we denote as $q _ { t } ( \xi )$ Then

$$
\exp \biggl ( - \frac { t \| h \| _ { 2 } ^ { 2 } } { 2 a \ell ^ { 2 } } \biggr ) = \int _ { \mathbb { R } ^ { d } } \exp ( 2 \pi i \xi \cdot h ) q _ { t } ( \xi ) \ d \xi .\tag{166}
$$

In particular, setting $h = 0$ gives $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } q _ { t } ( \xi ) \ d \xi = 1 } \end{array}$ and $q _ { t } ( \xi ) \geq 0$ . Substituting eq. (166) into eq. (164) gives

$$
k _ { \mathrm { R Q } , a } ( h ) = \frac { 1 } { \Gamma ( a ) } \int _ { 0 } ^ { \infty } t ^ { a - 1 } \exp ( - t ) \biggl [ \int _ { \mathbb { R } ^ { d } } \exp ( 2 \pi i \xi \cdot h ) q _ { t } ( \xi ) \ d \xi \biggr ] \ d t .\tag{167}
$$

Since $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } q _ { t } ( \xi ) \ d \xi = 1 } \end{array}$ , the absolute integral is finite:

$$
\frac { 1 } { \Gamma ( a ) } \int _ { 0 } ^ { \infty } \int _ { \mathbb R ^ { d } } t ^ { a - 1 } \exp ( - t ) | \exp ( 2 \pi i \xi \cdot h ) | q _ { t } ( \xi ) \ d \xi \ d t = \frac { 1 } { \Gamma ( a ) } \int _ { 0 } ^ { \infty } t ^ { a - 1 } \exp ( - t ) \ d t = 1 .\tag{168}
$$

Then we can use Fubini’s theorem to exchange the integrals and get

$$
k _ { \mathrm { R Q } , a } ( h ) = \int _ { \mathbb R ^ { d } } \exp ( 2 \pi i \xi \cdot h ) \left[ \frac { 1 } { \Gamma ( a ) } \int _ { 0 } ^ { \infty } t ^ { a - 1 } \exp ( - t ) q _ { t } ( \xi ) \ d t \right] \ : d \xi ,\tag{169}
$$

such that the spectral density of the rational quadratic kernel is

$$
s ( \xi ) = \frac { 1 } { \Gamma ( a ) } \int _ { 0 } ^ { \infty } t ^ { a - 1 } \exp ( - t ) q _ { t } ( \xi ) \ d t .\tag{170}
$$

Substituting $q _ { t } ( \xi )$ to s(ξ) gives

$$
s ( \xi ) = { \frac { ( 2 \pi a \ell ^ { 2 } ) ^ { d / 2 } } { \Gamma ( a ) } } \int _ { 0 } ^ { \infty } t ^ { a - 1 - d / 2 } \exp \left( - t - { \frac { 2 \pi ^ { 2 } a \ell ^ { 2 } \| \xi \| _ { 2 } ^ { 2 } } { t } } \right) d t\tag{171}
$$

$$
= 2 \cdot \frac { ( 2 \pi a \ell ^ { 2 } ) ^ { d / 2 } } { \Gamma ( a ) } ( z / 2 ) ^ { a - d / 2 } K _ { a - d / 2 } ( z ) ,\tag{172}
$$

where the second equality changes variable by $z : = 2 \pi \sqrt { 2 a } \ell \| \xi \| _ { 2 }$ and applies (DLMF, 2026, (10.32.10)) and (DLMF, 2026, (10.27.3)). Applying (DLMF, 2026, (10.40.2)) and expanding only the first term with $k = 0$ (which further applies (DLMF, 2026, (2.1.13)), (DLMF, 2026, (2.1.14)), and (DLMF, 2026, (10.17(i)))) gives

$$
\begin{array} { c } { { s ( \xi ) = 2 \cdot \frac { ( 2 \pi a \ell ^ { 2 } ) ^ { d / 2 } } { \Gamma ( a ) } ( z / 2 ) ^ { a - d / 2 } \sqrt { \pi / 2 z } \exp ( - z ) ( 1 + O ( z ^ { - 1 } ) ) } } \\ { { = C \cdot \| \xi \| _ { 2 } ^ { a - ( d + 1 ) / 2 } \exp ( - 2 \pi \sqrt { 2 a } \ell \| \xi \| _ { 2 } ) ( 1 + O ( \| \xi \| _ { 2 } ^ { - 1 } ) ) , } } \end{array}\tag{173}
$$

(174)

such that $s ( \xi ) = \Theta \bigl ( \| \xi \| _ { 2 } ^ { a - ( d + 1 ) / 2 } \exp ( - 2 \pi \sqrt { 2 a } \ell \| \xi \| _ { 2 } ) \bigr )$ as $\| \xi \| _ { 2 }  \infty$ . We define $b : = 2 \pi \sqrt { 2 a } \ell$ . Then

$$
s ( \xi ) = \Theta \bigl ( \| \xi \| _ { 2 } ^ { a - ( d + 1 ) / 2 } \exp ( - b \| \xi \| _ { 2 } ) \bigr ) .\tag{175}
$$

Step 2: Upper bound on $\gamma _ { T }$ . For suficiently large $\| { \boldsymbol { \xi } } \| _ { 2 }$ , the polynomial factor in $\mathrm { e q . }$ (175) can be absorbed into $\exp ( b \| \xi \| _ { 2 } / 2 )$ . Then there exist $R , C _ { 2 } > 0$ , such that $\begin{array} { r } { s ( \xi ) \overset { \cdot } { \leq } C _ { 2 } \exp \left( - \frac { b } { 2 } \| \xi \| _ { 2 } \right) } \end{array}$ when $\left\| \xi \right\| _ { 2 } \ge R .$ Hence,

$$
\int _ { \mathbb R ^ { d } } \exp ( b \| \xi \| _ { 2 } / 4 ) s ( \xi ) \ d \xi = \int _ { \| \xi \| _ { 2 } < R } \exp ( b \| \xi \| _ { 2 } / 4 ) s ( \xi ) \ d \xi + \int _ { \| \xi \| _ { 2 } \geq R } \exp ( b \| \xi \| _ { 2 } / 4 ) s ( \xi ) \ d \xi\tag{176}
$$

$$
\leq \exp ( b R / 4 ) + C _ { 2 } \int _ { \| \xi \| _ { 2 } \geq R } \exp ( - b \| \xi \| _ { 2 } / 4 ) \ d \xi\tag{177}
$$

$$
\leq \exp ( b R / 4 ) + C _ { 2 } | \mathbb { S } ^ { d - 1 } | \int _ { 0 } ^ { \infty } r ^ { d - 1 } \exp ( - b r / 4 ) \ d r\tag{178}
$$

$$
= \exp ( b R / 4 ) + C _ { 2 } | \mathbb { S } ^ { d - 1 } | ( 4 / b ) ^ { d } \Gamma ( d )\tag{179}
$$

$$
< \infty ,\tag{180}
$$

where the first inequality follows since $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } s ( \xi ) \ d \xi = 1 } \end{array}$ , and the second inequality uses polar coordinates with $| \mathbb { S } ^ { d - 1 } |$ being the unit sphere’s surface area. Then, the conditions of (Janz et al., 2026, Proposition 12) are satisfied with $\gamma = 1$ and $\tau _ { 0 } = b / 4$ , which gives $\epsilon _ { M } = O \big ( \exp ( - c M ^ { \gamma } ) \big )$ . Then the conditions of (Janz et al., 2026, Corollary $3 ( 2 ) )$ are satisfied with $\sigma = 1$ , which gives $\gamma _ { T } = O ( ( \log T ) ^ { d + 1 } )$ .

Step 3: Lower bound on $\gamma _ { T }$ . For suficiently large $\| \xi \| _ { 2 }$ , the polynomial factor in eq. (175) can be lower bounded by $\exp ( - b \| \xi \| _ { 2 } )$ . Then, there exist $R _ { 2 } , C _ { 3 } > 0$ such that $\begin{array} { r } { s ( \xi ) \ge C _ { 3 } \exp \left( - 2 b \| \xi \| _ { 2 } \right) } \end{array}$ when $\| \xi \| _ { 2 } \ge R _ { 2 }$ For $\| \xi \| _ { 2 } < R _ { 2 }$ , restricting the integral in eq. (171) to [1, 2] gives

$$
s ( \xi ) \geq \frac { ( 2 \pi a \ell ^ { 2 } ) ^ { d / 2 } } { \Gamma ( a ) } \int _ { 1 } ^ { 2 } t ^ { a - 1 - d / 2 } \exp \left( - t - \frac { 2 \pi ^ { 2 } a \ell ^ { 2 } \| \xi \| _ { 2 } ^ { 2 } } { t } \right) d t\tag{181}
$$

$$
\geq \frac { ( 2 \pi a \ell ^ { 2 } ) ^ { d / 2 } } { \Gamma ( a ) } \exp \bigl ( - 2 - 2 \pi ^ { 2 } a \ell ^ { 2 } R _ { 2 } ^ { 2 } \bigr ) \int _ { 1 } ^ { 2 } t ^ { a - 1 - d / 2 } \ d t\tag{182}
$$

$$
\geq \left[ \frac { ( 2 \pi a \ell ^ { 2 } ) ^ { d / 2 } } { \Gamma ( a ) } \exp \bigl ( - 2 - 2 \pi ^ { 2 } a \ell ^ { 2 } R _ { 2 } ^ { 2 } \bigr ) \int _ { 1 } ^ { 2 } t ^ { a - 1 - d / 2 } \ d t \right] \cdot \exp ( - 2 b \| \xi \| _ { 2 } ) ,\tag{183}
$$

where the last step follows because $\exp ( - 2 b \| \xi \| _ { 2 } ) \le 1$ , and the term in the brackets is positive. Then, there exists a constant $c _ { 0 }$ such that $s ( \xi ) \geq c _ { 0 } \exp ( - 2 b \| \xi \| _ { 2 } )$ for almost every $\xi \in \mathbb { R } ^ { d }$ . Consequently, the conditions of (Janz et al., 2026, Theorem 20) are satisfied with $\gamma = 1$ and $\tau = 2 b$ , which gives $\gamma _ { T } = \Omega ( ( \log T ) ^ { d + 1 } )$ . □

## E.2 Proof of Lemma 2 (Lower Bound Based on Information Diference)

Proof. Let $x _ { 1 } , \ldots , x _ { T }$ be a maximizing tuple o $\because \gamma _ { T } ( \mathcal { X } )$ , and let $\tilde { \mathcal X }$ be a finite approximation of X that contains this maximizing tuple (see Proposition 1). Denote $\begin{array} { r } { \hat { \nu } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \delta _ { x _ { t } } \in \Delta ( \tilde { \mathcal { X } } ) } \end{array}$ as the empirical design. Recal that Theorem 2 gives

$$
R _ { T } ^ { * } \ge c _ { 1 } \cdot \sqrt { T d _ { T } ( \tilde { \mathcal { X } } ) } ,\tag{184}
$$

so it remains to lower bound $d _ { T } ( \tilde { \mathcal { X } } )$ by $\gamma _ { T } - \gamma _ { m }$

To show this, we repeatedly delete entries from the maximizing tuple until m points remain. For $t \in$ $[ T ]$ , we define $\begin{array} { r } { S _ { t } \ : = \ \sum _ { s = 1 } ^ { t } \phi ( x _ { s } ) \otimes \phi ( x _ { s } ) } \end{array}$ . When t points remain, we delete the point that minimizes $\langle \phi ( x ) , ( S _ { t } + \lambda I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k }$ , and denote the minimum as $\ell _ { t }$ . For convenience, we relabel the tuple such

that $x _ { t }$ is deleted when t points remain. Observe that

$$
0 \leq \ell _ { t } \leq \langle \phi ( x _ { t } ) , ( \phi ( x _ { t } ) \otimes \phi ( x _ { t } ) + \lambda I _ { E } ) ^ { - 1 } \phi ( x _ { t } ) \rangle _ { k } = \frac { \| \phi ( x _ { t } ) \| _ { k } ^ { 2 } } { \| \phi ( x _ { t } ) \| _ { k } ^ { 2 } + \lambda } \leq \frac { 1 } { 1 + \lambda } < 1 ,\tag{185}
$$

where the first inequality holds since $S _ { t } + \lambda I _ { E } \succeq 0$ , the second inequality holds since $S _ { t } \succeq \phi ( x _ { t } ) \otimes \phi ( x _ { t } )$ the equality holds since $( \phi ( x _ { t } ) \otimes \phi ( x _ { t } ) + \lambda I _ { E } ) \phi ( x _ { t } ) = ( \| \phi ( x _ { t } ) \| _ { k } ^ { 2 } + \lambda ) \phi ( x _ { t } )$ , and the third inequality uses $\| \phi ( x ) \| _ { k } ^ { 2 } = k ( x , x ) \leq 1$

We define $K _ { m } ( x _ { 1 : m } ) : = [ k ( x _ { s } , x _ { s ^ { \prime } } ) ] _ { s , s ^ { \prime } \in [ m ] }$ . Lemma 14 shows that $K _ { T } ( x _ { 1 : T } )$ and $S _ { T }$ have the same positive eigenvalues (including multiplicity), and likewise $K _ { m } ( x _ { 1 : m } )$ and $S _ { m }$ , such that

$$
\log \operatorname * { d e t } _ { E } ( I _ { E } + \lambda ^ { - 1 } S _ { T } ) = 2 \gamma _ { T } ( \mathcal { X } ) , \qquad \log \operatorname * { d e t } _ { E } ( I _ { E } + \lambda ^ { - 1 } S _ { m } ) \le 2 \gamma _ { m } .\tag{186}
$$

Then, we have

$$
2 ( \gamma _ { T } - \gamma _ { m } ) \leq \log \operatorname * { d e t } _ { E } ( I _ { E } + \lambda ^ { - 1 } S _ { T } ) - \log \operatorname * { d e t } _ { E } ( I _ { E } + \lambda ^ { - 1 } S _ { m } )\tag{187}
$$

$$
= \sum _ { t = m + 1 } ^ { T } \biggl ( \log \operatorname * { d e t } _ { E } ( I _ { E } + \lambda ^ { - 1 } S _ { t } ) - \log \operatorname * { d e t } _ { E } ( I _ { E } + \lambda ^ { - 1 } S _ { t - 1 } ) \biggr ) .\tag{188}
$$

Note that by definition, $S _ { t } - \phi ( x _ { t } ) \otimes \phi ( x _ { t } ) = S _ { t - 1 }$ . We now apply Lemma 12 to $S _ { t } + \lambda I _ { E }$ , which gives

$$
\frac { \mathrm { d e t } _ { E } ( S _ { t - 1 } + \lambda I _ { E } ) } { \mathrm { d e t } _ { E } ( S _ { t } + \lambda I _ { E } ) } = 1 - \langle \phi ( x _ { t } ) , ( S _ { t } + \lambda I _ { E } ) ^ { - 1 } \phi ( x _ { t } ) \rangle _ { k } = 1 - \ell _ { t } ,\tag{189}
$$

and hence

$$
\log \operatorname* { d e t } _ { E } ( \lambda ^ { - 1 } S _ { t } + I _ { E } ) - \log \operatorname* { d e t } _ { E } ( \lambda ^ { - 1 } S _ { t - 1 } + I _ { E } ) = - \log ( 1 - \ell _ { t } ) \leq \frac { \ell _ { t } } { 1 - \ell _ { t } } \leq ( 1 + \lambda ^ { - 1 } ) \ell _ { t } ,\tag{190}
$$

where the first inequality follows since $\begin{array} { r } { - \log ( 1 - \ell _ { t } ) = \int _ { 0 } ^ { \ell _ { t } } \frac { d u } { 1 - u } \leq \frac { \ell _ { t } } { 1 - \ell _ { t } } } \end{array}$ , and the last inequality uses eq. (185). Substituting eq. (190) back into eq. (188) gives

$$
2 ( \gamma _ { T } - \gamma _ { m } ) \leq \sum _ { t = m + 1 } ^ { T } ( 1 + \lambda ^ { - 1 } ) \ell _ { t } .\tag{191}
$$

To upper bound $\ell _ { t } ,$ we observe that

$$
\ell _ { t } \leq \frac { 1 } { t } \sum _ { s = 1 } ^ { t } \langle \phi ( x _ { s } ) , ( S _ { t } + \lambda I _ { E } ) ^ { - 1 } \phi ( x _ { s } ) \rangle _ { k }\tag{192}
$$

$$
= \frac { 1 } { t } \operatorname { t r } _ { E } ( S _ { t } ( S _ { t } + \lambda I _ { E } ) ^ { - 1 } )\tag{193}
$$

$$
\leq \frac { 1 } { t } \operatorname { t r } _ { E } ( S _ { T } ( S _ { T } + \lambda I _ { E } ) ^ { - 1 } )\tag{194}
$$

$$
= { \frac { 1 } { t } } \operatorname { t r } _ { E } ( M _ { \hat { \nu } } ( M _ { \hat { \nu } } + \rho I _ { E } ) ^ { - 1 } )\tag{195}
$$

$$
\leq \frac { 1 } { t } 2 d _ { T } ( \tilde { \mathcal { X } } ) ,\tag{196}
$$

where eq. (192) follows since the minimum is upper bounded by the average, eq. (193) follows from $S _ { t } \succeq 0$ and Lemma 7, eq. (194) follows from $S _ { t } ~ \preceq ~ S _ { T }$ , eq. (195) follows from $S _ { T } = T M _ { \hat { \nu } }$ , and eq. (196) follows from Lemma 11. Substituting eq. (196) into eq. (191) gives $\begin{array} { r } { \left( \gamma _ { T } - \gamma _ { m } \right) \leq ( 1 + \lambda ^ { - 1 } ) d _ { T } ( \tilde { \mathcal X } ) \sum _ { t = m + 1 } ^ { T } t ^ { - 1 } } \end{array}$ , which combined with eq. (184) proves the result. □

## E.3 Proof of Proposition 2 (Suficient Condition for Log Factor Removal)

Proof. Choose T suficiently large that satisfies eq. (37) and $T \geq 2 L$ . Then $\lfloor T / L \rfloor \ge T / ( 2 L )$ , such that $\begin{array} { r } { \sum _ { t = \lfloor T / L \rfloor + 1 } ^ { T } \frac { 1 } { t } \leq \int _ { \lfloor T / L \rfloor } ^ { T } u ^ { - 1 } \ d u = \log ( T / \lfloor T / L \rfloor ) \leq \log ( 2 L ) } \end{array}$ . Combining this with eq. (37), and then applying Lemma 2, we obtain

$$
R _ { T } ^ { * } \geq c _ { 1 } \cdot \sqrt { \frac { c T \gamma _ { T } } { ( 1 + \lambda ^ { - 1 } ) \log ( 2 L ) } } = \Omega ( \sqrt { T \gamma _ { T } } ) ,\tag{197}
$$

where $\begin{array} { r } { c _ { 1 } = \operatorname* { m i n } \{ \frac { \sigma } { 3 2 } , \frac { \sigma \Delta _ { k } } { 1 6 } \} } \end{array}$ , which can be absorbed into $\Omega ( \cdot )$ because X is assumed to be fixed.

## E.4 Proof of Lemma 4 (Upper Bound on $h ( \nu , \rho ) )$

Proof. Fix a nonempty set $\nu \subseteq \mathcal { X }$ . We consider greedily choosing T points in V as follows: Define $S _ { 0 } : = 0 ,$ and at time $t \in [ T ]$ , choose $x _ { t } \in \mathcal V$ maximizing $\langle \phi ( x ) , ( S _ { t - 1 } + \lambda I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k }$ . Denote this maximum by $a _ { t }$ and set $S _ { t } : = S _ { t - 1 } + \phi ( x _ { t } ) \otimes \phi ( x _ { t } )$ . We also define $a _ { T + 1 } : = \operatorname* { m a x } _ { x \in \mathcal { V } } \langle \phi ( x ) , ( S _ { T } + \lambda I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k }$ . For every $t \in [ T ] , S _ { t } \succeq S _ { t - 1 }$ , such that $a _ { 1 } \geq \cdots \geq a _ { T + 1 }$ . Define $\begin{array} { r } { \hat { \nu } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \delta _ { x _ { t } } } \end{array}$ as the empirical design. Then $T M _ { \hat { \nu } } = S _ { T }$ , such that

$$
h ( \mathcal { V } , \rho ) \leq \operatorname* { m a x } _ { x \in \mathcal { V } } \langle \phi ( x ) , ( M _ { \hat { \nu } } + \rho I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k } = T \operatorname* { m a x } _ { x \in \mathcal { V } } \langle \phi ( x ) , ( S _ { T } + \lambda I _ { E } ) ^ { - 1 } \phi ( x ) \rangle _ { k } = T a _ { T + 1 } \leq \sum _ { t = 1 } ^ { T } a _ { t } .\tag{198}
$$

To connect $h ( \nu , \rho )$ to $\gamma _ { T }$ , we notice that

$$
2 \gamma _ { T } \geq \log \operatorname* { d e t } ( I _ { T } + \lambda ^ { - 1 } K _ { T } ( x _ { 1 : T } ) )
$$

$$
= \log \operatorname* { d e t } _ { E } ( I _ { E } + \lambda ^ { - 1 } S _ { T } )\tag{199}
$$

(200)

$$
= \sum _ { t = 1 } ^ { T } \log \frac { \operatorname* { d e t } _ { E } ( \lambda I _ { E } + S _ { t } ) } { \operatorname* { d e t } _ { E } ( \lambda I _ { E } + S _ { t - 1 } ) }\tag{201}
$$

$$
= \sum _ { t = 1 } ^ { T } \log \bigl ( 1 + \langle \phi ( x _ { t } ) , ( \lambda I _ { E } + S _ { t - 1 } ) ^ { - 1 } \phi ( x _ { t } ) \rangle _ { k } \bigr )\tag{202}
$$

$$
= \sum _ { t = 1 } ^ { T } \log ( 1 + a _ { t } ) ,\tag{203}
$$

where $\mathrm { e q . }$ (200) follows from Lemma 14, eq. (201) follows by telescoping, eq. (202) follows by applying Lemma 13 with $A = \lambda I _ { E } + S _ { t - 1 }$ , and eq. (203) follows from the definition of $a _ { t } .$ For every $t ~ \in ~ [ T ]$ $( S _ { t - 1 } + \lambda I _ { E } ) ^ { - 1 } \preceq \lambda ^ { - 1 } I _ { E } .$ , such that $a _ { t } \leq \langle \phi ( x _ { t } ) , \lambda ^ { - 1 } \phi ( x _ { t } ) \rangle _ { k } = \lambda ^ { - 1 } k ( x _ { t } , x _ { t } ) \leq \lambda ^ { - 1 }$ , which further implies $\begin{array} { r } { \log ( 1 + a _ { t } ) = \int _ { 0 } ^ { a _ { t } } \frac { d u } { 1 + u } \geq \frac { a _ { t } } { 1 + \lambda ^ { - 1 } } } \end{array}$ . Combining this result with eqs. (198) and (203) gives

$$
2 \gamma _ { T } \geq \sum _ { t = 1 } ^ { T } { \frac { a _ { t } } { 1 + \lambda ^ { - 1 } } } \geq { \frac { h ( \mathcal V , \rho ) } { 1 + \lambda ^ { - 1 } } } ,\tag{204}
$$

and maximizing over V proves the result.

## E.5 Proof of Theorem 3 (General Upper Bound)

Proof. We construct $\tilde { \mathcal { X } } : = \{ 0 , 1 / m , \ldots , 1 \} ^ { d }$ , where $m : = \lceil { \sqrt { d } } ( C T ) ^ { 1 / \alpha } \rceil$ . By a “rounding” argument to the closest grid point, we see that for every $x \in \mathcal { X }$ , there exists $a \in \tilde { \mathcal { X } }$ such that $\| x - a \| _ { 2 } \leq { \sqrt { d } } / ( 2 m )$ . Then for every $f \in \mathcal { F } _ { k } ( B )$ , we have

$$
| f ( x ) - f ( a ) | \leq B \| \phi ( x ) - \phi ( a ) \| _ { k } \leq B C ( \sqrt { d } / ( 2 m ) ) ^ { \alpha } \leq B / ( T 2 ^ { \alpha } ) \leq B / T ,\tag{205}
$$

where the first inequality follows from the reproducing property and $\| f \| _ { k } \le B$ , the second follows from eq. (42), the third follows since m $\geq { \sqrt { d } } ( C { \bar { T } } ) ^ { 1 / \alpha }$ , and the last follows because $\alpha > 0$ . Hence, we have $\epsilon ( \tilde { \mathcal { X } } ) \leq B / T$ for the quantity $\epsilon ( \tilde { \mathcal { X } } )$ defined in (40).

We then apply the RIPS algorithm to $\tilde { \mathcal { X } } .$ , such that the minimax regret on X satisfies

$$
R _ { T } ^ { * } = O \biggl ( \log ( | \tilde { \mathcal { X } } | T ) + \sqrt { T \gamma _ { T } ( \tilde { \mathcal { X } } ) \log ( | \tilde { \mathcal { X } } | T ) } \biggr ) + T \epsilon ( \tilde { \mathcal { X } } )\tag{206}
$$

$$
= O \big ( \log T + \sqrt { T \gamma _ { T } ( \tilde { \mathcal { X } } ) \log T } + B \big )\tag{207}
$$

$$
= O \big ( \sqrt { T \gamma _ { T } ( \tilde { \mathcal { X } } ) \log T } \big )\tag{208}
$$

$$
= O \big ( \sqrt { T \gamma _ { T } \log T } \big ) ,\tag{209}
$$

where the first step follows from Lemmas 3 and 4 along with $\mathrm { e q . ~ \ ( 4 0 ) }$ , the second step follows because $| \tilde { \mathcal { X } } | = ( m + 1 ) ^ { d } \leq ( \sqrt { d } ( C T ) ^ { 1 / \alpha } + 2 ) ^ { d }$ and $\epsilon ( \tilde { \mathcal { X } } ) \leq B / T$ , the third step follows since $\gamma _ { T } ( \tilde { \mathcal { X } } ) = \Omega ( \log T )$ by Lemma 15, and the last step follows since $\gamma _ { T } ( \dot { \mathcal X } ) \leq \gamma _ { T } ( \mathcal { X } ) = \gamma _ { T }$ □

## E.6 Proof that All Six Kernels Satisfy the H¨older Condition

Recall from Appendix A that $k ( \| x - x ^ { \prime } \| _ { 2 } ) = k ( x , x ^ { \prime } )$ . For $x , y \in { \mathcal { X } } ,$ we define $r : = \| x - y \| _ { 2 } .$ such that $\| \phi ( x ) - \phi ( y ) \| _ { k } ^ { 2 } = k ( x , x ) + k ( y , y ) - 2 k ( x , y ) = 2 ( k ( 0 ) - k ( r ) )$ . Then, it sufices to prove that $k ( 0 ) - k ( r ) \leq$ $\frac { C ^ { 2 } } { 2 } r ^ { 2 \alpha }$ for some fixed $C , \alpha > 0$ . We recall that $1 - \exp ( - x ) \leq x$ for every $x \in \mathbb { R }$

Squared exponential kernel. $\begin{array} { r } { k ( 0 ) - k ( r ) = 1 - \exp \left( - \frac { r ^ { 2 } } { 2 \ell ^ { 2 } } \right) \leq \frac { r ^ { 2 } } { 2 \ell ^ { 2 } } } \end{array}$ . Then, we can choose $\alpha = 1$

Rational quadratic kernel. Since log $( 1 + x ) \leq$ x for every $x \geq 0$ , we have

$$
k ( 0 ) - k ( r ) = 1 - \left( 1 + \frac { r ^ { 2 } } { 2 a \ell ^ { 2 } } \right) ^ { - \alpha } = 1 - \exp \left( - a \log \left( 1 + \frac { r ^ { 2 } } { 2 a \ell ^ { 2 } } \right) \right) \leq a \log \left( 1 + \frac { r ^ { 2 } } { 2 a \ell ^ { 2 } } \right) \leq \frac { r ^ { 2 } } { 2 \ell ^ { 2 } } ,\tag{210}
$$

such that we can choose $\alpha = 1$

γ-exponential kernel. $k ( 0 ) - k ( r ) = 1 - \exp ( - ( r / \ell ) ^ { \gamma } ) \leq ( r / \ell ) ^ { \gamma }$ . Hence, we can choose $\alpha = \gamma / 2$

Mat´ern kernel. For $z \ge 0 , K _ { \nu } ( z )$ can be represented by (DLMF, 2026, (10.32.10)) as follows:

$$
K _ { \nu } ( z ) = \frac { 1 } { 2 } \bigg ( \frac { z } { 2 } \bigg ) ^ { \nu } \int _ { 0 } ^ { \infty } \exp \bigg ( - t - \frac { z ^ { 2 } } { 4 t } \bigg ) \frac { d t } { t ^ { \nu + 1 } } .\tag{211}
$$

We change variable by $t = z ^ { 2 } / ( 4 s )$ , which gives

$$
K _ { \nu } ( z ) = z ^ { - \nu } 2 ^ { \nu - 1 } \int _ { 0 } ^ { \infty } s ^ { \nu - 1 } \exp \biggl ( - \frac { z ^ { 2 } } { 4 s } - s \biggr ) ~ d s ,\tag{212}
$$

such that

$$
{ \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } } z ^ { \nu } K _ { \nu } ( z ) = { \frac { 1 } { \Gamma ( \nu ) } } \int _ { 0 } ^ { \infty } s ^ { \nu - 1 } \exp \left( - { \frac { z ^ { 2 } } { 4 s } } - s \right) d s .\tag{213}
$$

Taking $z = \sqrt { 2 \nu } r / \ell$ gives

$$
k _ { \nu } ( r ) = { \frac { 1 } { \Gamma ( \nu ) } } \int _ { 0 } ^ { \infty } s ^ { \nu - 1 } \exp ( - s ) \exp \left( - { \frac { \nu r ^ { 2 } } { 2 \ell ^ { 2 } s } } \right) d s .\tag{214}
$$

The integrand is upper bounded by $s ^ { \nu - 1 } \exp ( - s )$ , whose integral is $\Gamma ( \nu ) < \infty$ . For $u \geq 0$ and $0 < \alpha \leq 1$ , we have $1 - \exp ( - u ) \leq \operatorname* { m i n } \{ u , 1 \} \leq u ^ { \alpha }$ . Then,

$$
k ( 0 ) - k ( r ) = \frac { 1 } { \Gamma ( \nu ) } \int _ { 0 } ^ { \infty } s ^ { \nu - 1 } \exp ( - s ) \left[ 1 - \exp \left( - \frac { \nu r ^ { 2 } } { 2 \ell ^ { 2 } s } \right) \right] d s\tag{215}
$$

$$
\leq { \frac { 1 } { \Gamma ( \nu ) } } \left( { \frac { \nu r ^ { 2 } } { 2 \ell ^ { 2 } } } \right) ^ { \alpha } \int _ { 0 } ^ { \infty } s ^ { \nu - \alpha - 1 } \exp ( - s ) \ d s\tag{216}
$$

$$
= \frac { \Gamma ( \nu - \alpha ) } { \Gamma ( \nu ) } \bigg ( \frac { \nu } { 2 \ell ^ { 2 } } \bigg ) ^ { \alpha } r ^ { 2 \alpha } ,\tag{217}
$$

where the ratio of Gamma functions is finite provided $\nu - \alpha > 0$ . Then we can choose any $0 < \alpha \leq 1$ with $\alpha < \nu$

Piecewise-polynomial kernel. If $0 \leq r \leq 1$ , then

$$
k ( 0 ) - k ( r ) = - \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor + 3 q + 1 } c _ { j , q } r ^ { j } \leq \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor + 3 q + 1 } | c _ { j , q } | r ^ { j } \leq \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor + 3 q + 1 } | c _ { j , q } | r .\tag{218}
$$

If $r > 1$ , then $k ( 0 ) - k ( r ) = c _ { 0 , q } \leq | c _ { 0 , q } | r$ . Then $\begin{array} { r } { k ( 0 ) - k ( r ) \leq \operatorname* { m a x } \{ \sum _ { j = 1 } ^ { \lfloor d / 2 \rfloor + 3 q + 1 } \vert c _ { j , q } \vert , \vert c _ { 0 , q } \vert \} \cdot r . } \end{array}$ , and we can choose $\alpha = 1 / 2$

For the one-dimensional Dirichlet kernel, pairing terms with index j and $- j$ gives

$$
k _ { \mathrm { D i r } , n } ( r ) = { \frac { 1 } { 2 n + 1 } } \sum _ { k = - n } ^ { n } \exp ( - i k r ) = { \frac { 1 } { 2 n + 1 } } { \bigg ( } 1 + 2 \sum _ { j = 1 } ^ { n } \cos ( j r ) { \bigg ) }\tag{219}
$$

Specifically, $k ( 0 ) = 1$ . Then

$$
k ( 0 ) - k ( r ) = 1 - { \frac { 1 } { 2 n + 1 } } { \Biggl ( } 1 + 2 \sum _ { j = 1 } ^ { n } \cos ( j r ) { \Biggr ) } = { \frac { 2 } { 2 n + 1 } } \sum _ { j = 1 } ^ { n } ( 1 - \cos ( j r ) ) \leq { \frac { r ^ { 2 } } { 2 n + 1 } } \sum _ { j = 1 } ^ { n } j ^ { 2 } = { \frac { n ( n + 1 ) } { 6 } } r ^ { 2 } ,\tag{220}
$$

where the inequality follows because $1 - \cos u = 2 \sin ^ { 2 } ( u / 2 ) \leq u ^ { 2 } / 2$ , and the last equality follows because $\textstyle \sum _ { i = 1 } ^ { n } j ^ { 2 } = n ( { \bar { n } } + 1 ) ( 2 n + 1 ) / 6$ . Then we can choose $\alpha = 1$

## E.7 Proof of Lemma 5 (Sharper Upper Bounds; First Result)

Proof. Fix a nonempty $\nu \subseteq { \mathcal { X } } .$ , and denote $m : = | \nu |$ . Following eq. (4), we define ν as a regularized $D _ { - }$ optimal design over $\Delta ( \nu )$ . We also define $E _ { 1 } : = \mathrm { s p a n } \{ \phi ( v ) : v \in \mathcal { V } \}$ . Define $D : = \mathrm { d i a g } ( \nu ( v ) , v \in \mathcal { V } )$ and the weighted Gram matrix $H _ { k } : = D ^ { 1 / 2 } [ k ( v , w ) ] _ { v , w \in \mathcal { V } } \bar { D } ^ { 1 / 2 }$ . By an argument similar to that in Lemma 14, we can show that $H _ { k }$ and $M _ { \nu }$ have the same nonzero eigenvalues (including multiplicities) as follows: Define $\Phi : \mathbb { R } ^ { m }  E _ { 1 }$ by $\Phi b _ { i } = \sqrt { \nu ( v _ { i } ) } \phi ( v _ { i } )$ , and define $\Phi ^ { * }$ to be its adjoint. Then $H _ { k } = \Phi ^ { * } \Phi$ and $M _ { \nu } = \Phi \Phi ^ { * }$ , and the result follows.

Then, (Camilleri et al., 2021, Lemma 3) gives $h ( \mathcal { V } , \rho ) \leq \mathrm { t r } _ { E _ { 1 } } ( M _ { \nu } ( M _ { \nu } + \rho I _ { E _ { 1 } } ) ^ { - 1 } )$ , which combined with the above fact on $H _ { k } , M _ { \nu }$ implies

$$
h ( \mathcal { V } , \rho ) \leq \mathrm { t r } ( H _ { k } ( H _ { k } + \rho I ) ^ { - 1 } ) .\tag{221}
$$

To proceed, we apply (Janz et al., 2026, Theorem 1). Suppose that for every $M \in \mathbb { Z } _ { + }$ , there exist positive semidefinite kernels $k _ { M , 1 } , k _ { M , 2 } : \mathcal { X } \times \mathcal { X }  \mathbb { R }$ and $\epsilon _ { M } > 0$ such that

$$
k \preceq k _ { M , 1 } + k _ { M , 2 } , \qquad \mathrm { r a n k } [ k _ { M , 1 } ( x , y ) ] _ { x , y \in \mathcal { X } } = O ( M ^ { d } ) , \qquad \operatorname* { m a x } _ { x \in \mathcal { X } } k _ { M , 2 } ( x , x ) \leq \epsilon _ { M } ^ { 2 } ,\tag{222}
$$

where $k \preceq k _ { M , 1 } + k _ { M , 2 }$ denotes the corresponding Gram-matrix inequality. Following $H _ { k }$ , we define ${ { H } _ { { { k } _ { M , 1 } } } }$ and $H _ { k _ { M , 2 } }$ as the weighted Gram matrices for $k _ { M , 1 }$ and $k _ { M , 2 }$ , respectively. Then

$$
H _ { k } \preceq H _ { k _ { M , 1 } } + H _ { k _ { M , 2 } } , \qquad \mathrm { r a n k ~ } H _ { k _ { M , 1 } } = O ( M ^ { d } ) , \qquad \mathrm { t r ~ } H _ { k _ { M , 2 } } = \sum _ { v \in \mathcal { V } } \nu ( v ) k _ { M , 2 } ( v , v ) \leq \epsilon _ { M } ^ { 2 } .\tag{223}
$$

Then, we follow the proof of (Janz et al., 2026, Theorem 1) to upper bound $\mathrm { { r } } ( H _ { k } ( H _ { k } + \rho I ) ^ { - 1 } )$ . For convenience, we define $H : = H _ { k _ { M , 1 } } + H _ { k _ { M , 2 } }$ . Since $H _ { k } \preceq H$ , it follows that $\mathrm { t r } ( H _ { k } ( H _ { k } + \rho I ) ^ { - 1 } ) \le \mathrm { t r } ( H ( H +$ $\rho I ) ^ { - 1 } )$ . Let $P$ be the orthogonal projection onto the range of $H _ { k _ { M , 1 } }$ , and define $Q : = I - P$ . Then

$$
\mathrm { t r } ( H _ { k } ( H _ { k } + \rho I ) ^ { - 1 } ) \le \mathrm { t r } ( H ( H + \rho I ) ^ { - 1 } )\tag{224}
$$

$$
= \operatorname { t r } ( P H ( H + \rho I ) ^ { - 1 } P ) + \operatorname { t r } ( Q H ( H + \rho I ) ^ { - 1 } Q )\tag{225}
$$

$$
\leq \mathrm { r a n k } \ H _ { k _ { M , 1 } } + \mathrm { t r } ( Q H ( H + \rho I ) ^ { - 1 } Q )\tag{226}
$$

$$
\leq \mathrm { r a n k } \ H _ { k _ { M , 1 } } + \rho ^ { - 1 } \mathrm { t r } ( Q H _ { k _ { M , 2 } } Q ) ,\tag{227}
$$

where the second inequality uses $H ( H + \rho I ) ^ { - 1 } \preceq I$ and tr P = rank $H _ { k _ { M , 1 } ; }$ , the third inequality follows since $H ( H + \rho I ) ^ { - 1 } \preceq \rho ^ { - 1 } H$ and $Q H _ { k _ { M , 1 } } Q = 0$ . Applying eqs. (221) and (223) gives

$$
h ( \mathcal { V } , \rho ) = O ( M ^ { d } + \epsilon _ { M } ^ { 2 } / \rho ) = O ( M ^ { d } + T \epsilon _ { M } ^ { 2 } ) .\tag{228}
$$

We now specialize eq. (228) to each kernel.

Squared exponential kernel. (Janz et al., 2026, Remark 2) shows the squared exponential kernel has spectral density $s ( \xi ) = \Theta \bigl ( \exp ( - \tau \| \xi \| _ { 2 } ^ { 2 } ) \bigr )$ for some $\tau > 0$ . Denote $\mu$ as its spectral measure. Then

$$
\int _ { \mathbb R ^ { d } } \exp \left( \frac { \tau } { 2 } \| \xi \| _ { 2 } ^ { 2 } \right) \mu ( d \xi ) = \int _ { \mathbb R ^ { d } } \exp \left( \frac { \tau } { 2 } \| \xi \| _ { 2 } ^ { 2 } \right) s ( \xi ) \ d \xi \leq C \int _ { \mathbb R ^ { d } } \exp \left( - \frac { \tau } { 2 } \| \xi \| _ { 2 } ^ { 2 } \right) \ d \xi = C ( 2 \pi / \tau ) ^ { d / 2 } < \infty ,\tag{229}
$$

where the second equality follows by applying $\textstyle \int _ { \mathbb { R } } \exp ( - \tau x ^ { 2 } / 2 )$ dx $= \sqrt { 2 \pi / \tau }$ to each dimension. Then the conditions of (Janz et al., 2026, Proposition 15) are satisfied with $\tau _ { 0 } = \tau / 2$ and $\gamma = 2$ , which implies k<sub>SE</sub> (defined on $[ - 1 , 1 ] ^ { d } \times [ - 1 , 1 ] ^ { d } )$ is entire of finite order at most $p = 2$ . Then (Janz et al., 2026, Proposition 14) gives $\epsilon _ { M } = O \big ( \exp ( - c M \log ( e + M ) ) \big )$ for some $c > 0$ , and restricting to $\mathcal { X } \times \mathcal { X }$ preserves this bound. We choose $\begin{array} { r } { M = \left\lceil { \frac { 2 } { c } } { \frac { \log T } { \log \log T } } \right\rceil } \end{array}$ . For suficiently large T, $\textstyle \log ( e + M ) \geq { \frac { 1 } { 2 } }$ log log T, such that $\epsilon _ { M } ^ { 2 } = O ( T ^ { - 2 } )$ , and

$$
h ( \mathcal { V } , \rho ) = O \big ( ( \log T / \log \log T ) ^ { d } + T ^ { - 1 } \big ) = O \big ( ( \log T / \log \log T ) ^ { d } \big ) .\tag{230}
$$

Rational quadratic kernel. We define $\tilde { k } _ { \mathrm { R Q } }$ as the extension of $k _ { \mathrm { R Q } }$ to $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ . Then $\tilde { k } _ { \mathrm { R Q } }$ is positive semidefinite, because it can be represented as a positive mixture of squared exponential kernels (Rasmussen and Williams, 2005, (4.19)–(4.20)). In addition, $\tilde { k } _ { \mathrm { R Q } }$ is real analytic on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ , such that $\tilde { k } _ { \mathrm { R Q } }$ belongs to the Gevrey class $G ^ { 1 } ( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } )$ , which is the class of real-analytic functions on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ (Janz et al., 2026). Then (Janz et al., 2026, Corollary 11) is satisfied with $U = \mathbb { R } ^ { d } , \sigma = 1 , \tilde { k } = \tilde { k } _ { \mathrm { R Q } }$ , such that $\tilde { k } _ { \mathrm { R Q } }$ (restricted to $[ 0 , 1 ] ^ { d } \times [ 0 , 1 ] ^ { d } )$ has $\epsilon _ { M } = O ( \exp ( - c M ) )$ ), and further restricting to $\mathcal { X } \times \mathcal { X }$ preserves this bound. Then, choosing $M = \lceil c ^ { - 1 } \log T \rceil$ gives

$$
h ( \mathcal { V } , \rho ) = O \big ( ( \log T ) ^ { d } + T ^ { - 1 } \big ) = O \big ( ( \log T ) ^ { d } \big ) ,\tag{231}
$$

and hence max<sub>V⊆</sub> $_ { \therefore x } h ( \mathcal { V } , \rho ) = O ( ( \log T ) ^ { d } )$

One-dimensional Dirichlet kernel. Observe that

$$
k _ { \mathrm { D i r } , n } ( x , x ^ { \prime } ) = { \frac { 1 } { 2 n + 1 } } \sum _ { k = - n } ^ { n } \exp ( - i k ( x - x ^ { \prime } ) )\tag{232}
$$

$$
= { \frac { 1 } { 2 n + 1 } } { \bigg ( } 1 + 2 \sum _ { j = 1 } ^ { n } \cos ( j ( x - x ^ { \prime } ) ) { \bigg ) }\tag{233}
$$

$$
= \frac { 1 } { 2 n + 1 } \biggl ( 1 + 2 \sum _ { j = 1 } ^ { n } [ \cos ( j x ) \cos ( j x ^ { \prime } ) + \sin ( j x ) \sin ( j x ^ { \prime } ) ] \biggr ) ,\tag{234}
$$

where each of the $2 n + 1$ terms in eq. (234) is a positive semidefinite kernel with rank at most 1. Then, $k _ { \mathrm { D i r } , n } ( r )$ has rank at most $2 n + 1$ . Choosing $k _ { M , 1 } = k$ and $k _ { M , 2 } \equiv 0$ gives rank $H _ { k _ { M , 1 } } \leq 2 n + 1$ and tr $H _ { k _ { M , 2 } } = 0$ , which combined with eqs. (221) and (227) gives

$$
h ( \gamma , \rho ) \leq 2 n + 1 ,\tag{235}
$$

and hence max $\nu \varsigma \chi \ h ( \mathcal { V } , \rho ) \leq 2 n + 1$

## E.8 Proof of Lemma 6 (Sharper Upper Bounds; Second Result)

For $\mathcal { X } = [ 0 , 1 ] ^ { d }$ , consider the finite approximation $\tilde { \mathcal { X } } = \{ 0 , 1 / m , \dots , 1 \} ^ { d }$ , where m will be chosen later. We overload notation and write $k ( x , y ) = k ( x - y )$ . For each of the kernels, we consider its spectral representation

$$
k ( z ) = \int _ { \mathbb R ^ { d } } \exp ( i w ^ { T } z ) p ( w ) \ d w ,\tag{236}
$$

with Fourier normalization constants absorbed into p. Specifically, $p ( w ) = ( 2 \pi ) ^ { - d } s ( w / 2 \pi )$ , where $s ( w )$ is the spectral density in (Janz et al., 2026), and $p ( w ) = ( 2 \pi ) ^ { - d } \hat { k } ( w )$ , where $\hat { k } ( w )$ is the Fourier transform of the kernel k used in (Lee and Javidi, 2025). Our proof relies on bounding $p ( w )$ , as shown below.

Lemma 29. Consider the setup of Section 3.1 with $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . Suppose that there exist constants $A > 0$ and $0 < s < 2$ , such that

$$
0 \leq p ( w ) \leq A ( 1 + \| w \| _ { 2 } ^ { 2 } ) ^ { - ( s + d / 2 ) } \qquad \forall w \in \mathbb { R } ^ { d } .\tag{237}
$$

Then there exists a constant $C > 0$ (which may depend on $A , d , s )$ such that

$$
\epsilon ( \tilde { \mathcal { X } } ) \leq C B m ^ { - s } .\tag{238}
$$

Proof. Fix $x \in \mathcal { X }$ , and consider a grid cell in $\tilde { \mathcal X }$ containing x. Then there exists $j \in \{ 0 , \ldots , m - 1 \} ^ { d }$ , such that the vertices of the cell can be written as

$$
v _ { b } = m ^ { - 1 } ( j + b ) \in \tilde { \mathcal { X } } \qquad \forall b \in \{ 0 , 1 \} ^ { d } .\tag{239}
$$

Likewise, there exists $\theta \in [ 0 , 1 ] ^ { d }$ , such that $x = m ^ { - 1 } ( j + \theta )$ . For $\ell \in [ d ]$ , we define $b _ { \ell }$ to be the ℓ-th coordinate of b, and similarly $\theta _ { \ell }$ . Then, we define the multilinear interpolation weights

$$
w _ { b } ( x ) : = \prod _ { \ell : b _ { \ell } = 1 } \theta _ { \ell } \prod _ { \ell : b _ { \ell } = 0 } ( 1 - \theta _ { \ell } ) .\tag{240}
$$

It follows that

$$
w _ { b } ( x ) \geq 0 , \qquad \sum _ { b \in \{ 0 , 1 \} ^ { d } } w _ { b } ( x ) = 1 , \qquad \sum _ { b \in \{ 0 , 1 \} ^ { d } } w _ { b } ( x ) v _ { b } = x , \qquad \| v _ { b } - x \| _ { 2 } \leq \sqrt { d } / m .\tag{241}
$$

Define $\begin{array} { r } { I _ { m } f ( x ) : = \sum _ { b \in \{ 0 , 1 \} ^ { d } } w _ { b } ( x ) f ( v _ { b } ) } \end{array}$ . Because $w _ { b } ( x )$ are weights that sum to one, we have $I _ { m } f ( x ) \leq$ $\operatorname* { m a x } _ { a \in { \tilde { \mathcal { X } } } } f ( a )$ . Consequently, $\begin{array} { r } { f ( x ) - \operatorname* { m a x } _ { a \in \tilde { \mathcal { X } } } f ( a ) \leq f ( x ) - I _ { m } f ( x ) \leq | f ( x ) - I _ { m } f ( x ) | } \end{array}$ |, and taking the maximum over $x \in \mathcal { X }$ gives

$$
\operatorname* { m a x } _ { x \in { \mathcal { X } } } f ( x ) - \operatorname* { m a x } _ { a \in { \tilde { \mathcal { X } } } } f ( a ) \leq \operatorname* { s u p } _ { x \in { \mathcal { X } } } | f ( x ) - I _ { m } f ( x ) | .\tag{242}
$$

We proceed to upper bound $f ( x ) - I _ { m } f ( x )$ . By the reproducing property,

$$
f ( x ) - I _ { m } f ( x ) = \Big \langle f , \phi ( x ) - \sum _ { b \in \{ 0 , 1 \} ^ { d } } w _ { b } ( x ) \phi ( v _ { b } ) \Big \rangle _ { k } .\tag{243}
$$

Applying the Cauchy-Schwarz inequality and $\| f \| _ { k } \le B$ gives

$$
| f ( x ) - I _ { m } f ( x ) | ^ { 2 } \leq B ^ { 2 } \biggl \| \phi ( x ) - \sum _ { b \in \{ 0 , 1 \} ^ { d } } w _ { b } ( x ) \phi ( v _ { b } ) \biggl \| _ { k } ^ { 2 } .\tag{244}
$$

For the squared norm term in eq. (244), we have

$$
\begin{array} { l l } { \displaystyle \left\| \phi ( x ) - \sum _ { b \in \{ 0 , 1 \} ^ { d } } w _ { b } ( x ) \phi ( v _ { b } ) \right\| _ { k } ^ { 2 } } \\ { \displaystyle = k ( x , x ) + \sum _ { b , c } w _ { b } ( x ) w _ { c } ( x ) k ( v _ { b } , v _ { c } ) - \sum _ { b } w _ { b } ( x ) ( k ( x , v _ { b } ) + k ( v _ { b } , x ) ) } & { ( 2 4 5 ) } \\ { \displaystyle } & { = \int _ { \mathbb { R } ^ { d } } \Big [ 1 + \sum _ { b , c } w _ { b } ( x ) w _ { c } ( x ) \exp ( i w T ( v _ { b } - v _ { c } ) ) - \sum _ { b } w _ { b } ( x ) \exp ( i w T ( v _ { b } - x ) ) - \sum _ { b } w _ { b } ( x ) \exp ( i w T ^ { T } ( x - v _ { b } ) ) \Big ] p ( w ) d w } \end{array}\tag{246}
$$

$$
= \int _ { \mathbb R ^ { d } } \bigg ( 1 - \sum _ { b } w _ { b } ( x ) \exp ( i w ^ { T } ( v _ { b } - x ) ) \bigg ) \bigg ( 1 - \sum _ { c } w _ { c } ( x ) \exp ( - i w ^ { T } ( v _ { c } - x ) ) \bigg ) p ( w ) d w\tag{247}
$$

$$
= \int _ { \mathbb { R } ^ { d } } \Bigl | 1 - \sum _ { b } w _ { b } ( x ) \exp ( i w ^ { T } ( v _ { b } - x ) ) \Bigr | ^ { 2 } p ( w ) \ d w ,\tag{248}
$$

where the second equality applies eq. (236), and the last equality follows because the weights are real and ${ \overline { { e ^ { i t } } } } = e ^ { - i t }$ . Now define $\begin{array} { r } { D _ { x } ( w ) : = 1 - \sum _ { b } w _ { b } ( x ) \exp ( i w ^ { T } ( v _ { b } - x ) ) } \end{array}$ . Since $| \mathrm { e x p } ( u i ) | = | \mathrm { c o s } u + i \sin u | = 1$ for real $u ,$ we have $\begin{array} { r } { | D _ { x } ( w ) | \leq 1 + \sum _ { b } | w _ { b } ( x ) | | \mathrm { e x p } ( i w ^ { T } ( v _ { b } - x ) ) | = 1 + \sum _ { b } w _ { b } ( x ) = 2 . } \end{array}$ . For a sharper bound, we first show

$$
| \mathrm { e x p } ( i u ) - 1 - u i | \leq u ^ { 2 } / 2 .\tag{249}
$$

Indeed, ex $\begin{array} { r } { \begin{array} { r } { ( u i ) - 1 - u i = u i \int _ { 0 } ^ { 1 } \exp ( i u t ) - 1 ~ d t = - u ^ { 2 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { t } \exp ( i s u ) } \end{array} } \end{array}$ dsdt. Since $| \mathrm { e x p } ( i s u ) | = 1$ , we have $\begin{array} { r } { | \mathrm { e x p } ( u i ) - 1 - u i | \leq u ^ { 2 } \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { t } 1 \ d s d t = u ^ { 2 } / 2 } \end{array}$ . Then

$$
| D _ { x } ( w ) | = \left| \sum _ { b } w _ { b } ( x ) ( \exp ( i w ^ { T } ( v _ { b } - x ) - 1 - i w ^ { T } ( v _ { b } - x ) ) \right|\tag{250}
$$

$$
\leq \frac { 1 } { 2 } \sum _ { b } w _ { b } ( x ) ( w ^ { T } ( v _ { b } - x ) ) ^ { 2 }\tag{251}
$$

$$
\leq \frac { 1 } { 2 } \sum _ { b } w _ { b } ( x ) \| w \| _ { 2 } ^ { 2 } \| v _ { b } - x \| _ { 2 } ^ { 2 }\tag{252}
$$

$$
\leq \frac { d } { 2 m ^ { 2 } } \| w \| _ { 2 } ^ { 2 } ,\tag{253}
$$

where the first equality uses $\begin{array} { r } { \sum _ { b } w _ { b } ( x ) w ^ { T } ( v _ { b } - x ) = 0 } \end{array}$ from eq. (241), the first inequality applies eq. (249), the second inequality applies Cauchy-Schwarz, and the last inequality uses $\| v _ { b } - x \| _ { 2 } \leq \sqrt { d } / m$ from eq. (241). Then

$$
| D _ { x } ( w ) | ^ { 2 } \leq C _ { d } \cdot \operatorname* { m i n } \big \{ 1 , m ^ { - 4 } \| w \| _ { 2 } ^ { 4 } \big \} ,\tag{254}
$$

where $C _ { d } > 0$ is a constant that depends on d. Combining eqs. (244) and (248) gives

$$
| f ( x ) - I _ { m } f ( x ) | ^ { 2 } \leq B ^ { 2 } \int _ { \mathbb { R } ^ { d } } | D _ { x } ( w ) | ^ { 2 } p ( w ) \ d w\tag{255}
$$

$$
\leq A B ^ { 2 } C _ { d } \int _ { \mathbb { R } ^ { d } } \operatorname* { m i n } \{ 1 , m ^ { - 4 } \| w \| _ { 2 } ^ { 4 } \} ( 1 + \| w \| _ { 2 } ^ { 2 } ) ^ { - ( s + d / 2 ) } \ d w\tag{256}
$$

$$
\leq A B ^ { 2 } C _ { d } | \mathbb { S } ^ { d - 1 } | \int _ { 0 } ^ { \infty } \operatorname* { m i n } \{ 1 , m ^ { - 4 } r ^ { 4 } \} ( 1 + r ^ { 2 } ) ^ { - ( s + d / 2 ) } r ^ { d - 1 } \ d r ,\tag{257}
$$

where the second inequality uses eq. (237) and eq. (254), and the third inequality uses the polar-coordinate with $| \mathbb { S } ^ { d - 1 } |$ being the unit sphere’s surface area. Now we decompose the integral according to the value of r. When $0 \leq r \leq 1 , m ^ { - 4 } r ^ { 4 } \leq 1$ and $( 1 + r ^ { 2 } ) ^ { - ( s + d / 2 ) } \leq 1$ . Then the integral on (0, 1) is upper bounded by $\textstyle \int _ { 0 } ^ { 1 } m ^ { - 4 } r ^ { d + 3 } d r = { \frac { m ^ { - 4 } } { d + 4 } } \leq m ^ { - 4 }$ . For $1 \leq r \leq m , ( 1 + r ^ { 2 } ) ^ { - ( s + d / 2 ) } \leq r ^ { 2 \cdot - ( s + d / 2 ) }$ , and the integrand is at most $\dot { m } ^ { - 4 } r ^ { 3 - 2 s }$ . For $r \geq m$ , the integral is at most $( 1 + r ^ { 2 } ) ^ { - ( s + d / 2 ) } r ^ { d - 1 } \leq r ^ { - 2 s - d + d - 1 } = r ^ { - 2 s - 1 }$ . Thus

$$
| f ( x ) - I _ { m } f ( x ) | ^ { 2 } \leq A B ^ { 2 } C _ { d } | \mathbb { S } ^ { d - 1 } | \cdot \left[ m ^ { - 4 } + m ^ { - 4 } \int _ { 1 } ^ { m } r ^ { 3 - 2 s } \ d r + \int _ { m } ^ { \infty } r ^ { - 2 s - 1 } \right] .\tag{258}
$$

For $\begin{array} { r } { 0 < s < 2 , m ^ { - 4 } \int _ { 1 } ^ { m } r ^ { 3 - 2 s } \ d r = \frac { m ^ { - 2 s } - m ^ { - 4 } } { 4 - 2 s } \le \frac { m ^ { - 2 s } } { 4 - 2 s } } \end{array}$ , and $\textstyle \int _ { m } ^ { \infty } r ^ { - 2 s - 1 } = { \frac { m ^ { - 2 s } } { 2 s } }$ . Also $m ^ { - 4 } \leq m ^ { - 2 s }$ . Hence,

$$
| f ( x ) - I _ { m } f ( x ) | ^ { 2 } \leq C B ^ { 2 } m ^ { - 2 s } .\tag{259}
$$

Taking the square root and combining with eq. (242) proves the result.

For a kernel satisfying eq. (237), applying MOSS on $\tilde { \mathcal X }$ gives

$$
R _ { T } ^ { * } \le 3 9 \sigma \sqrt { | \tilde { \chi } | T } + 2 B | \tilde { \chi } | + T \epsilon ( \tilde { \chi } )\tag{260}
$$

$$
\leq 3 9 \sigma \sqrt { T ( m + 1 ) ^ { d } + 2 B ( m + 1 ) ^ { d } + T C B m ^ { - s } } .\tag{261}
$$

Choosing $m = \lfloor T ^ { 1 / ( 2 s + d ) } \rfloor$ gives $R _ { T } ^ { * } = O ( T ^ { ( s + d ) / ( 2 s + d ) } )$ . We now show that each kernel satisfies eq. (237).

Mat´ern kernel. (Janz et al., 2026, Corollary 7) gives $p _ { \nu } ( w ) = \Theta \bigl ( ( 1 + \| w \| _ { 2 } ^ { 2 } ) ^ { - ( \nu + d / 2 ) } \bigr )$ , which satisfies eq. (237) with $s = \nu . ~ s \in ( 0 , 2 )$ implies $\nu \in ( 0 , 2 )$ , in which case combining $R _ { T } ^ { * } = O ( { T ^ { ( s + d ) / ( 2 s + d ) } } )$ and $\gamma _ { T } \overset { \cdot } { = } \Theta ( T ^ { d / ( 2 \nu + d ) } )$ in Table 1 gives $\bar { R } _ { T } ^ { * } = O ( T ^ { ( \nu + d ) / ( 2 \nu + d ) } ) = O ( \sqrt { T \gamma _ { T } } )$

γ-exponential kernel. (Lee and Javidi, 2025, Proposition 1) gives $p ( w ) = \Theta \bigl ( ( 1 + \| w \| _ { 2 } ) ^ { - ( \gamma + d ) } \bigr )$ . This implies $p ( w ) = O \big ( ( 1 + \| w \| _ { 2 } ^ { 2 } ) ^ { - ( \gamma / 2 + d / 2 ) } \big )$ , such that eq. (237) is satisfied with $s = \gamma / 2 . ^ { 2 } \mathrm { ~ S ~ }$ pecifically, $\gamma \in ( 0 , 2 )$ is covered by $s \in ( 0 , 2 )$ . Then, combining $R _ { T } ^ { * } = O ( T ^ { ( s + d ) / ( 2 s + d ) } )$ and $\gamma _ { T } = \Theta ( T ^ { d / ( \gamma + d ) } )$ in Table 1 gives $R _ { T } ^ { * } = O \bigl ( T ^ { ( \gamma / 2 + d ) / ( \gamma + d ) } \bigr ) = O ( \sqrt { T \gamma _ { T } } )$

Piecewise-polynomial kernel. (Lee and Javidi, 2025, Proposition 1) gives $p ( w ) = \Theta \bigl ( ( 1 + \| w \| _ { 2 } ) ^ { - ( 2 q + 1 + d ) } \bigr )$ such that $p ( w ) = O \big ( ( 1 + \| w \| _ { 2 } ^ { 2 } ) ^ { - ( q + 1 / 2 + d / 2 ) } \big )$ . Then eq. (237) is satisfied with $\begin{array} { r } { s = q + \frac { 1 } { 2 } } \end{array}$ , and $s \ < \ 2$ implies $q = 0 , 1$ , which combined with Appendix A means $q = 0 , d \geq 3$ or $q = 1$ . In this case, combining $R _ { T } ^ { * } = { \cal O } \bar { ( } T ^ { ( s + d ) / ( 2 s + d ) } )$ and $\gamma _ { T } = \Theta ( T ^ { d / ( 2 \bar { q } + 1 + d ) } )$ in Table 1 gives $R _ { T } ^ { * } = O \big ( T ^ { ( q + \frac { 1 } { 2 } + d ) / ( 2 q + 1 + d ) } \big ) = O ( \sqrt { T \gamma _ { T } } )$

## F Upper Bound on the Maximum Information Gain

Another consequence of Corollary 2 is that an upper bound on $R _ { T } ^ { * }$ implies an upper bound on $\gamma _ { T }$ . Here we use $R _ { t } ^ { * }$ to denote minimax regret at horizon $t \in \mathbb { Z } _ { + }$ , and the following result considers $t \leq T$

Corollary 4. Consider the setup of Section 3.1 where X is compact. Suppose there exists a sequence $( U _ { t } ) _ { t \in \mathbb { Z } _ { + } }$ + such that $R _ { t } ^ { * } \leq U _ { t }$ for every $t \in \mathbb { Z } _ { + }$ . Then,

$$
\gamma _ { T } \leq \gamma _ { 1 } + \frac { 1 + \lambda ^ { - 1 } } { c _ { 1 } ^ { 2 } } \sum _ { t = 2 } ^ { T } \frac { U _ { t } ^ { 2 } } { t ^ { 2 } } ,\tag{262}
$$

where $\begin{array} { r } { c _ { 1 } = \operatorname* { m i n } \{ \frac { \sigma } { 3 2 } , \frac { \sigma \Delta _ { k } } { 1 6 } \} } \end{array}$

Proof. $\mathrm { I f } \ T = 1$ , then the result holds with equality. Then, we assume $T \geq 2$ . Applying Lemma 2 with $T = t$ and $m = t - 1$ gives

$$
R _ { t } ^ { * } \geq c _ { 1 } \cdot t \sqrt { \frac { \left( \gamma _ { t } - \gamma _ { t - 1 } \right) } { \left( 1 + \lambda ^ { - 1 } \right) } } ,\tag{263}
$$

which implies

$$
\gamma _ { t } - \gamma _ { t - 1 } \leq \frac { 1 + \lambda ^ { - 1 } } { c _ { 1 } ^ { 2 } } \cdot \frac { ( R _ { t } ^ { * } ) ^ { 2 } } { t ^ { 2 } } \leq \frac { 1 + \lambda ^ { - 1 } } { c _ { 1 } ^ { 2 } } \cdot \frac { U _ { t } ^ { 2 } } { t ^ { 2 } } .\tag{264}
$$

Summing from $t = 2$ to $T$ proves the result.

Corollary 4 does not improve Table 1, where the upper bounds and lower bounds of $\gamma _ { T }$ already match. On the other hand, Corollary 4 provides an alternative approach to obtain upper bounds on $\gamma _ { T }$ . As an example, we consider the γ-exponential kernel and $\mathcal { X } = [ 0 , 1 ] ^ { d }$ . Then, the proof of Lemma 6 gives $R _ { T } ^ { * } =$ ${ \cal O } ( T ^ { ( \hat { \gamma } + 2 \acute { d } ) / ( 2 \gamma + 2 { d } ) } ) ,$ ), where the proof is independent of $\gamma _ { T }$ . Observe that

$$
\sum _ { t = 2 } ^ { T } t ^ { 2 ( \gamma + 2 d ) / ( 2 \gamma + 2 d ) } / t ^ { 2 } = \sum _ { t = 2 } ^ { T } t ^ { - \gamma / ( \gamma + d ) } \leq \int _ { 1 } ^ { T } t ^ { - \gamma / ( \gamma + d ) } \ d t = \frac { \gamma + d } { d } ( T ^ { d / ( \gamma + d ) } - 1 ) .\tag{265}
$$

Define $U _ { t } : = C t ^ { ( \gamma + 2 d ) / ( 2 \gamma + 2 d ) }$ , where $C > 0$ is some constant chosen large enough such that $R _ { t } ^ { * } \leq U _ { t }$ for all $t \in \mathbb { Z } _ { + }$ . Applying Corollary 4 with the sequence $( U _ { t } ) _ { t \in \mathbb { Z } . }$ gives $\gamma _ { T } \overset { \cdot } { = } \overset { \cdot } { O } ( T ^ { d / ( \gamma + d ) } )$ , which matches the $\Theta \big ( T ^ { d / ( \gamma + d ) } \big )$ rate of $\gamma _ { T }$ in Table 1. More generally, Corollary 4 could be useful for kernels where the upper bounds of $\gamma _ { T }$ are not established or hard to derive, but upper bounds on $R _ { T } ^ { * }$ are available. On the other hand, the immediate upper bound $\gamma _ { T } = O ( U _ { T } ^ { 2 } \log ( T ) / T )$ only gives $\gamma _ { T } = { \cal { O } } \dot { ( } T ^ { d / ( \gamma + d ) } \log T )$ , which has an extra log T factor.