# Preventing Model Collapse: A Fisher-Rao Perspective on the Dynamics of Training with Synthetic Data

Matteo Marchi<sup>1</sup>, João Pedro Silvestre<sup>1</sup>, Bahman Gharesifard<sup>2</sup>, and Paulo Tabuada<sup>1</sup>

Abstract— Large Language Models (LLMs) are now routinely trained using synthetic data, since high-quality human data has been exhausted by the ever increasing needs of larger and larger models. However, recursive training on synthetic data frequently induces model collapse, a degenerative feedback loop where models progressively forget the true underlying data distribution. Training on a mixture of synthetic and fresh human data is a logical countermeasure and can prevent model collapse. However, it is an open question as to what is the exact minimum required ratio of human-to-synthetic data to maintain training stability.

In this paper, we establish rigorous theoretical guarantees on the minimum rate of human data required to prevent model collapse. Although previous work established a formal lower bound for this ratio, such bound can be vacuous for very high dimensions, as the analysis relies on the usual Euclidean metric in R<sup>n</sup> and is not adapted to the space of categorical probability distributions. Instead, in this paper we explicitly leverage the information-geometric structure of the probability simplex by analyzing the dynamics of the process under the Fisher-Rao metric. We derive quantitative contraction and invariance bounds that are stable and do not become trivial as the dimensions increase. Thus, we show that the effective required data ratio to prevent model collapse is different than previously implied.

## I. INTRODUCTION

In recent years, generative AI, and in particular Large Language Models (LLMs) have become deeply ingrained in our society, primarily driven by a remarkable leap in performance and generation capabilities of recent models [1]. Today, modern LLMs can produce text that is virtually indistinguishable from human writing; in fact, recent studies show that human evaluators are often misguided by flawed heuristics when trying to identify AI-generated language [2].

However, this sudden leap in quality is not without drawbacks, including steep economic costs [3], heavy computational demands [4], and a reliance on increasingly vast training corpora [5]. Driven by the empirical scaling laws required to push state-of-the-art performance, developers are now training foundational models on trillions of tokens [6]. Unfortunately, sustaining this path requires such a large amount of data that some studies project the supply of fresh, high-quality human text will soon be completely exhausted [7].

The high caliber of AI-generated text presents a tempting solution to this impending data scarcity by leveraging the models’ own synthetic outputs for future training. However, recursive training on machine-generated data leads to a critical failure mode commonly referred to as model collapse, as demonstrated both empirically [8], [9] and theoretically [10]. Instead of learning effectively, models caught in this degenerative feedback loop deviate from the true data distribution, amplifying their own errors and producing repetitive, homogenized outputs [8], [11]. Integrating fresh human data into the iterative training cycle appears to be a logical countermeasure, but current evidence suggests that simplistic strategies, such as injecting small, fixed proportions of real data, are insufficient to stop model collapse [12]. Alternative mitigation strategies are being actively investigated, such as employing data verification and curation pipelines to filter out degraded synthetic outputs [13]. Although curating synthetic data can delay the onset of degeneration, it introduces substantial computational overhead and relies heavily on the quality and robustness of the verifier itself. Recent empirical and theoretical works have established that incorporating sufficient human data can prevent model collapse [14], [15], yet characterizing the precise dynamics of this mixed-data regime remains a significant challenge. This naturally raises a fundamental question: can we rigorously bound the amount of human data required to preclude model collapse?

We tackle this problem by building upon the framework introduced in [15], modeling the iterative training of generative models as a closed-loop stochastic process. This previous work has primarily focused on analyzing asymptotic equilibrium states [10], [15] and worked with the traditional Euclidean metric, which progressively distorts distances between probability distributions as the dimensions increase. Our approach fundamentally departs from this prior analysis by explicitly leveraging the information-geometric structure of the probability simplex. Specifically, by working with the Fisher-Rao metric, we derive quantitative contraction and invariance bounds that remain stable and meaningful as the underlying dimensions of the model increase. This geometric perspective shows that the effective amount of human data required to prevent collapse is greater than previously implied.

## II. NOTATION AND PRELIMINARIES

## A. Notation

We denote by $\mathbb { R } ^ { n }$ the n-dimensional Euclidean space, $\mathbb { R } _ { 0 } ^ { + }$ as the set of nonnegative real numbers, N as the set of natural numbers with zero, $\Delta ^ { n } \triangleq \left\{ x \in ( \mathbb { R } _ { 0 } ^ { + } ) ^ { n } \mid \sum _ { i = 1 } ^ { n } x _ { i } = 1 \right\}$ as the n-dimensional probability simplex, $\| \cdot \| _ { 1 }$ as the 1-norm, $\| \cdot \| _ { 2 }$ as the 2-norm, and $\langle \cdot , \cdot \rangle$ as the inner product.

We use standard asymptotic notations $O ( \cdot )$ and $\Theta ( \cdot )$ to describe the limiting behavior of sequences. In particular, for sequences of functions $a _ { n } : \mathbb { N } \to \mathbb { R }$ and $b _ { n } : \mathbb { N } \to \mathbb { R } _ { > 0 }$ , we write $a _ { n } = O ( b _ { n } )$ whenever there exist $c \in \mathbb { R } ^ { + }$ and $n _ { 0 } \in \mathbb { N }$ such that:

$$
| a _ { n } | \leqslant c b _ { n } , \qquad { \mathrm { f o r ~ a l l ~ } } n \geqslant n _ { 0 } .
$$

We also write $a _ { n } = \Theta ( b _ { n } )$ whenever there exist $c _ { 1 } , c _ { 2 } \in \mathbb { R } ^ { + }$ 0 and $n _ { 0 } \in \mathbb { N }$ such that:

$$
c _ { 1 } b _ { n } \leqslant | a _ { n } | \leqslant c _ { 2 } b _ { n } , \qquad { \mathrm { f o r ~ a l l ~ } } n \geqslant n _ { 0 } .
$$

Given vectors $v \in [ 0 , 1 ] ^ { n }$ and $w \in ] 0 , 1 ] ^ { n }$ , we define:

$$
\| \boldsymbol { v } \| _ { \mathrm { d i a g } ( w ) } ^ { 2 } = \boldsymbol { v } ^ { \top } \mathrm { d i a g } ( w ) \boldsymbol { v } ,
$$

where $\begin{array} { r l r } { \mathrm { d i a g } ( w ) } & { { } \in } & { \mathbb { R } ^ { n \times n } } \end{array}$ denotes the square matrix whose diagonal consists of the entries of w, and whose off-diagonal elements are 0. Given a scalar function $f : \mathbb { R } \to \mathbb { R }$ and a vector argument $x \in \mathbb { R } ^ { n }$ , we denote $f ( x ) = ( f ( x _ { 1 } ) , f ( x _ { 2 } ) , \ldots , f ( x _ { n } ) ) \in \mathbb { R } ^ { n }$ as its element-wise application to x. Consider the interior of the probability simplex $\Delta _ { \mathrm { i n t } } ^ { n } = \Delta ^ { n } \backslash \partial \Delta ^ { n }$ . We equip $\Delta _ { \mathrm { i n t } } ^ { n }$ with the Fisher-Rao Riemannian metric $\begin{array} { r } { g _ { \theta } ( u , v ) = \sum _ { i = 1 } ^ { n } \frac { u _ { i } v _ { i } } { \theta _ { i } } } \end{array}$ , for $u , v$ in the tangent space $T _ { \theta } \Delta ^ { n } = \{ u \in \mathbb { R } ^ { n } : \sum _ { i = 1 } ^ { n } u _ { i } = 0 \}$ . This metric induces the geodesic Hellinger distance on the probability simplex [16]:

$$
d _ { \mathrm { F R } } ( \theta , \vartheta ) = \operatorname { a r c c o s } \Big ( \sum _ { i = 1 } ^ { n } \sqrt { \theta _ { i } \vartheta _ { i } } \Big ) ,\tag{1}
$$

which is naturally related to the squared Hellinger distance $\begin{array} { r } { H ^ { 2 } ( \theta , \vartheta ) : = \frac 1 2 \| \dot { \sqrt { \theta } } - \sqrt { \vartheta } \| _ { 2 } ^ { 2 } = 1 - \dot { \langle \sqrt { \theta } , \sqrt { \vartheta } \rangle } } \end{array}$

Definition 1 (Kullback-Leibler divergence). Let $\theta , \vartheta \in \Delta _ { \mathrm { i n t } } ^ { n }$ The Kullback-Leibler (KL) divergence of θ from ϑ is:

$$
D _ { \mathrm { K L } } ( \theta \| \vartheta ) : = \sum _ { i = 1 } ^ { n } \theta _ { i } \log \frac { \theta _ { i } } { \vartheta _ { i } } .
$$

D<sub>KL</sub> is non-negative and equals zero if and only if $\theta = \vartheta$

## B. Generative Models

In this section we describe the mathematical model used to analyze a generative model and its iterative training process. This is largely based on the model first presented in [10] that we extend in this work.

We define a generative model to be a function $\phi : \mathbb { R } ^ { p } $ $\Delta ^ { n }$ that maps a parameter vector w $\mathbf { \Lambda } \in ~ \mathbb { R } ^ { p }$ to an output distribution $\Theta = \phi ( w ) \in \Delta ^ { n }$ . The i-th entry of $\phi ( w )$ is the nominal probability of producing the i-th element from a list of outcomes $\overline { { \mathcal { V } } } = \{ \mathcal { V } _ { 1 } , . . . \mathcal { V } _ { n } \}$ when the model is queried. With no loss of generality, we assume that the i-th element of Y is the n-dimensional vector containing 1 in its i-th entry and 0 in all others<sup>1</sup>.

In practice, when a model generates data, the actual output probability distribution is modulated via a temperature function $\tau : \Delta ^ { n } \to \Delta ^ { n }$ , defined for $i = 1 , 2 , \dots , n$ as:

$$
\tau _ { i } ( \Theta ) = \frac { \Theta _ { i } ^ { 1 / T } } { \sum _ { j = 1 } ^ { n } \Theta _ { j } ^ { 1 / T } } , \qquad T > 0 .\tag{2}
$$

The temperature function describes the common practice of converting a $\textstyle { \frac { 1 } { T } }$ -scaled vector of raw logits, produced by a generative model, into a vector of probabilities. This specific form of $\tau ,$ induced by the standard softmax function, is ubiquitous across nearly all modern generative models used in practice. For a more detailed discussion and a broader class of temperature functions, we refer the reader to [10].

## C. Iterative Training

We now consider a sequence of generative models, $\Theta ( k ) = \phi ( w ( k ) )$ , indexed by $k \in \mathbb { N } ,$ , each trained on a dataset $\mathcal { D } _ { k }$ with cardinality $\ell _ { k } = | \mathcal { D } _ { k } |$ . The dataset at any time step k is a multiset<sup>2</sup> $\mathcal { D } _ { k } = \{ Y _ { 1 } , . . . , Y _ { \ell _ { k } } \}$ where each element belongs to the set of possible outcomes ${ \overline { { \mathcal { V } } } } .$ To any nonempty dataset, we can associate a corresponding “empirical” probability vector:

$$
\Theta ^ { \prime } ( k ) = \frac { 1 } { \ell _ { k } } \sum _ { i = 1 } ^ { \ell _ { k } } Y _ { i } ,\tag{3}
$$

whose entries are the relative frequencies of each possible outcome within the dataset. $\mathcal { D } _ { k }$ evolves by accumulating some amount of “fresh” human generated $\mathrm { d a t a } ^ { 3 }$ and some amount of synthetic data generated by the model trained at the current time step. Specifically, at any time step $k ,$ we assume that $\alpha _ { k } \in$ N outcomes of $\overline { { \mathcal { D } } }$ are sampled according to $\tau ( \Theta ( k ) )$ to form:

$$
\mathcal { D } _ { k } ^ { \mathrm { s y n } } = \{ Y _ { \ell _ { k } + 1 } , . . . , Y _ { \ell _ { k } + \alpha _ { k } } \} .
$$

Additionally, $\beta _ { k } \in \mathbb { N }$ outcomes of $\overline { { \mathcal { D } } }$ are sampled according to a fixed external distribution $\mathcal { H } \in \Delta ^ { n }$ to form:

$$
\mathcal { D } _ { k } ^ { \mathrm { h u m a n } } = \{ Y _ { \ell _ { k } + \alpha _ { k } + 1 } , \dots , Y _ { \ell _ { k } + \alpha _ { k } + \beta _ { k } } \} .
$$

The training dataset available at step $k + 1$ is thus:

$$
\mathcal { D } _ { k + 1 } = \mathcal { D } _ { k } \cup \mathcal { D } _ { k } ^ { \mathrm { s y n } } \cup \mathcal { D } _ { k } ^ { \mathrm { h u m a n } } ,
$$

of cardinality $\ell _ { k + 1 } = \ell _ { k } + \alpha _ { k } + \beta _ { k } .$

The training of a generative model at time step $k { \pm } 1$ occurs by using the newly available dataset $\mathcal { D } _ { k + 1 }$ (and possibly the previously trained $w ( k ) )$ to compute a new parameter vector $w ( k + 1 ) = f ( w ( k ) , { \mathcal { D } } _ { k + 1 } )$ , where $f$ abstracts away the details of the training and optimization process. Then, the evolution of Θ is described by the stochastic process:

$$
\Theta ( k + 1 ) = \phi \big ( f ( w ( k ) , \mathcal { D } _ { k + 1 } ) \big ) .\tag{4}
$$

In [10], the authors analyze the asymptotic behavior of (4) in the absence of fresh data $( \beta _ { k } = 0 )$ . They show that in the limit of $k  \infty$ the distribution learned by the generative model exhibits a high degree of degeneration or model collapse with high probability. Specifically, $\Theta ( k )$ becomes arbitrarily close to the boundary of the simplex (in fact, to the corners of the simplex) or to its center (uniform probability distribution). The question of whether this degeneration can be mitigated by injecting human data in the loop was investigated by the authors of [15]. Assuming $\beta _ { k } = \mu \alpha _ { k }$ for a constant ratio $\mu > 0$ , the limiting behavior of (4) is determined almost surely by the behavior of the continuous-time dynamical system:

$$
\dot { \theta } ( t ) = \tau ( \theta ( t ) ) - \theta ( t ) + \mu \big ( \theta _ { 0 } - \theta ( t ) \big ) + \varepsilon ( t ) ,\tag{5}
$$

where $\theta _ { 0 } : = \mathcal { H }$ is the human distribution, $\varepsilon ( t )$ is a bounded perturbation, and we assume the flow of (5) preserves the simplex, implying $\textstyle \sum _ { i = 1 } ^ { n } { \dot { \theta } } _ { i } \ = \ 0$ . The magnitude of the perturbation ε is a measure of training accuracy, and a model that learns a dataset distribution with low error has a correspondingly small perturbation ε.

Under these assumptions, the results in [15] state that for a sufficiently high ratio of human data $\mu ,$ , the trajectories of (5) converge to an Euclidean ball $\mathbb { B } ( \theta _ { e } , \epsilon )$ around an equilibrium $\theta _ { e }$ satisfying $\tau ( \theta _ { e } ) - \theta _ { e } + \mu ( \theta _ { 0 } - \theta _ { e } ) = 0$ and provide an expression bounding the size of this ball as a function of $\mu$ and the other problem parameters. However, analyzing these dynamics under the Euclidean metric yields bounds that become increasingly uninformative as the dimension n of the probability simplex grows. For $n \to \infty$ , the Euclidean distance between almost any two probability distributions approaches zero [17]. Thus, requiring trajectories to converge to an Euclidean ball of a fixed radius becomes a progressively weaker condition, as such a ball eventually encompasses the majority of the simplex regardless of the choice of ratio $\mu .$ For an example illustrating this, take n to be even and consider the family of “disjoint” probability vectors of form $p =$ $\textstyle { \left( { \frac { 2 } { n } } , \dots , { \frac { 2 } { n } } , 0 , \dots , 0 \right) }$ and $\begin{array} { r } { \dot { q } = \left( 0 , \dots , 0 , \frac { 2 } { n } , \dots , \frac { 2 } { n } \right) } \end{array}$ . Despite representing completely distinct categorical outcomes, their Euclidean distance scales as $O ( 1 / \sqrt { n } )$ and vanishes as $n $ $\infty .$ . By contrast, the Fisher-Rao metric captures the underlying information-geometric structure of $\Delta ^ { n }$ (see [18]) and assigns a constant positive distance $\begin{array} { r } { d _ { \mathrm { F R } } ( p , q ) = \operatorname { a r c c o s } ( 0 ) = \frac { \pi } { 2 } } \end{array}$ between $p$ and $q$ regardless of $n .$ . Because the Euclidean metric progressively under-penalizes the distance between distributions, it yields an overly optimistic assessment of the required human data scaling. The main contribution of this paper is to substantially refine this analysis by working directly on the Fisher-Rao manifold of $\Delta ^ { n }$

## III. MAIN RESULT

We first need to make the following assumptions. We require that the magnitude of the perturbation ε is bounded, the human data distribution does not lie exactly on the boundary of the probability simplex, and that there exists an equilibrium of (5) when $\varepsilon = 0 ,$

Assumption 1. There exists $\eta \geqslant 0$ such the term $\varepsilon ( t ) \in \mathbb { R } ^ { n }$ in (5) satisfies $\| \varepsilon ( t ) \| _ { \infty } \leqslant \eta f o r$ all $t \geqslant 0$ , and all entries of $\theta _ { 0 } ,$ , the human distribution, are strictly positive:

$$
\delta : = \operatorname* { m i n } _ { i } \theta _ { 0 , i } > 0 .
$$

Assumption 2. There exists $\theta _ { e } \in \Delta ^ { n }$ satisfying:

$$
\tau ( \theta _ { e } ) - \theta _ { e } + \mu ( \theta _ { 0 } - \theta _ { e } ) = 0 .\tag{6}
$$

Further, it is convenient to define the following quantity:

Definition 2. Let $\theta _ { e } ^ { \mathrm { { m i n } } }$ be the smallest element of $\theta _ { e } ,$ then we define the normalized inverse temperature η<sub>FR</sub> as:

$$
\eta _ { \mathrm { F R } } \ : = \ \frac { 1 } { T \theta _ { e } ^ { \mathrm { m i n } } } .
$$

We can now introduce the main contribution of this work in the following theorem. This result identifies a ball in the Fisher-Rao metric that $\theta$ will converge to, the convergence rate to this ball, and a minimum threshold for the humanto-synthetic data ratio that guarantees this behavior. Here, we merely state the theorem and prove it in the following section.

Theorem 1. Suppose that Assumptions 1-2 hold and that $\mu \delta > \eta ,$ , where $\mu$ is the ratio between human and synthetic data, i.e., $\mu = \beta _ { k } / \alpha _ { k }$ . Fix $\kappa \in [ 0 , 1 )$ and consider $t \geqslant t _ { \kappa }$ where $\begin{array} { r } { t _ { \kappa } = \frac { 1 } { 1 + \mu } \ln \left( \frac { 1 } { 1 - \kappa } \right) } \end{array}$ . If the following inequality holds:

$$
\mu \geqslant \mathrm { m a x } \left\{ \frac { 1 + T \eta _ { \mathrm { F R } } \kappa } { T \delta \kappa } , \frac { \eta + \frac { 2 \eta _ { \mathrm { F R } } } { \kappa } } { \delta } \right\} ,\tag{7}
$$

there exists $\lambda > 0$ such that every solution of (5) satisfies:

$$
d _ { \mathrm { F R } } ( \theta ( t ) , \theta _ { e } ) \leqslant \frac { \pi } { 2 } \sqrt { e ^ { - \lambda ( t - t _ { \kappa } ) } D _ { \mathrm { K L } } ( \theta ( t _ { \kappa } ) \| \theta _ { e } ) } + \epsilon _ { \mathrm { F R } } ,
$$

where:

(8)

$$
\epsilon _ { \mathrm { F R } } = \frac { \pi \eta \sqrt { n \operatorname* { m a x } _ { i } \theta _ { e , i } } } { \sqrt { 2 } \kappa ( \mu \delta - \eta ) \left( 1 - \frac { 2 \eta _ { \mathrm { F R } } \operatorname* { m a x } _ { i } \theta _ { e , i } } { \kappa ( \mu \delta - \eta ) } \right) } .\tag{9}
$$

Note that (8) implies that $\theta$ converges to a Fisher-Rao ball of size ϵ<sub>FR</sub> for $t \to \infty$ . We now compare the bound provided by (9) in Theorem 1 with the bound obtained in [15] that establishes convergence to an Euclidean ball of size:

$$
\epsilon = \frac { \eta \kappa ( \mu \delta - \eta ) T } { ( \mu + 1 ) ( \kappa ( \mu \delta - \eta ) T - 1 ) } .\tag{10}
$$

To more easily exhibit the relative scaling of the bounds, we assign scaling laws to δ and $\mu$ as functions of $n ,$ , the dimension of the probability simplex $\Delta ^ { n }$

Proposition 1. Suppose that the assumptions of Theorem 1, and the ones in [15, Theorem 1], hold and that:

$$
\delta _ { n } \sim n ^ { - \beta _ { 0 } } , \qquad \mu _ { n } \sim c n ^ { p } , \qquad \| \varepsilon \| _ { \infty } = \eta ,
$$

for some $\beta _ { 0 } > 1 , c > 0 ,$ and $p > 2 \beta _ { 0 }$ . Then:

1) In the Euclidean case (10):

$$
\epsilon = \Theta \left( n ^ { - p } \right) .
$$

2) In the Fisher-Rao case (9):

$$
\epsilon _ { \mathrm { F R } } = \Theta \big ( n ^ { \frac { 1 } { 2 } + \beta _ { 0 } - p } \big ) .
$$

Proof. We first prove (1). Consider (10) given by:

$$
\epsilon = \frac { \eta \kappa ( \mu _ { n } \delta _ { n } - \eta ) T } { ( \mu _ { n } + 1 ) \big ( \kappa ( \mu _ { n } \delta _ { n } - \eta ) T - 1 \big ) } ,
$$

with $\delta _ { n } \sim n ^ { - \beta _ { 0 } }$ and $\mu _ { n } \sim c n ^ { p }$ . Since $\mu _ { n } \delta _ { n } \sim c n ^ { p - \beta _ { 0 } }$ , with $p \geqslant \beta _ { 0 }$ , and $\mu _ { n } + 1 \sim \mu _ { n }$ , we have that:

$$
\epsilon \sim \frac { \eta } { \mu _ { n } } = \Theta ( n ^ { - p } ) ,
$$

which proves the claim.

To prove (2), we first have to determine how $\theta _ { e }$ scales with n under the assumptions. Manipulating (6), we obtain:

$$
\theta _ { e } - \theta _ { 0 } = \frac { 1 } { 1 + \mu _ { n } } \big ( \tau ( \theta _ { e } ) - \theta _ { 0 } \big ) .
$$

Taking $\ell _ { \infty }$ norms and noting $\tau ( \theta _ { e } ) , \theta _ { 0 } \in \Delta ^ { n }$ , we get:

$$
\| \theta _ { e } - \theta _ { 0 } \| _ { \infty } \leqslant \frac { 2 } { 1 + \mu _ { n } } .
$$

Hence, if $\mu _ { n } \sim c n ^ { p }$ with $p \geqslant 1$ we have:

$$
\| \theta _ { e } - \theta _ { 0 } \| _ { \infty } = O ( n ^ { - p } ) .
$$

Noting that m $\begin{array} { r } { \mathrm { a x } _ { i } \theta _ { 0 , i } \leqslant 1 - \left( n - 1 \right) \mathrm { m i n } _ { i } \theta _ { 0 , i } } \end{array}$ and min<sub>i</sub> $\theta _ { 0 , i } = \delta _ { n } \sim n ^ { - \beta _ { 0 } }$ , we can establish that:

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { i } \theta _ { e , i } = \operatorname* { m a x } _ { i } \theta _ { 0 , i } + O ( n ^ { - p } ) } \\ { \displaystyle \qquad \leqslant 1 - ( n - 1 ) \operatorname* { m i n } _ { i } \theta _ { 0 , i } + O ( n ^ { - p } ) } \\ { \displaystyle \qquad = 1 - ( n - 1 ) \Theta ( n ^ { - \beta _ { 0 } } ) + O ( n ^ { - p } ) } \\ { \displaystyle \qquad = 1 - \Theta ( n ^ { 1 - \beta _ { 0 } } ) , } \end{array}
$$

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { i } \theta _ { e , i } = \operatorname* { m i n } _ { i } \theta _ { 0 , i } - O ( n ^ { - p } ) } \\ { \displaystyle \qquad = \Theta ( n ^ { - \beta _ { 0 } } ) - O ( n ^ { - p } ) = \Theta ( n ^ { - \beta _ { 0 } } ) . } \end{array}
$$

We are now in a position to prove (2), by considering (9). Recall that $\mu _ { n } \sim c n ^ { p }$ and $\delta _ { n } \sim n ^ { - \beta _ { 0 } }$ , so:

$$
\mu _ { n } \delta _ { n } \sim c n ^ { p - \beta _ { 0 } } .
$$

Using max<sub>i</sub> $\theta _ { e , i } \leqslant 1 - \Theta ( n ^ { 1 - \beta _ { 0 } } )$ and min<sub>i</sub> $\theta _ { e , i } = \Theta ( n ^ { - \beta _ { 0 } } )$ we obtain:

$$
\begin{array} { r l r } & { } & { \frac { 2 \eta _ { \mathrm { F R } } \operatorname* { m a x } _ { i } \theta _ { e , i } } { \kappa \left( \mu _ { n } \delta _ { n } - \eta \right) } = \frac { \frac { 2 } { T } \left( \frac { \operatorname* { m a x } _ { i } \theta _ { e , i } } { \operatorname* { m i n } _ { i } \theta _ { e , i } } \right) } { \kappa \left( \mu _ { n } \delta _ { n } - \eta \right) } \leqslant \frac { \left( \frac { 1 - \Theta ( n ^ { 1 - \beta _ { 0 } } ) } { \Theta ( n ^ { - \beta _ { 0 } } ) } \right) } { \Theta \left( n ^ { p - \beta _ { 0 } } \right) } } \\ & { } & { = \Theta \left( \frac { n ^ { \beta _ { 0 } } } { n ^ { p - \beta _ { 0 } } } \right) = \Theta \left( n ^ { 2 \beta _ { 0 } - p } \right) , } \end{array}
$$

which tends to 0 for every $p > 2 \beta _ { 0 }$ . Hence:

$$
1 - \frac { 2 \eta _ { \mathrm { F R } } \operatorname* { m a x } _ { i } \theta _ { e , i } } { \kappa ( \mu _ { n } \delta _ { n } - \eta ) } = \Theta ( 1 ) .
$$

Moreover, $\mu _ { n } \delta _ { n } - \eta \sim \mu _ { n } \delta _ { n }$ , so the denominator of (9) is asymptotically proportional to $\mu _ { n } \delta _ { n }$ , and $\in _ { \mathrm { F R } }$ scales like:

$$
\frac { \pi \eta \sqrt { n \operatorname* { m a x } _ { i } \theta _ { e , i } } } { \sqrt { 2 } \kappa \mu _ { n } \delta _ { n } } \leqslant \frac { \sqrt { n ( 1 - \Theta ( n ^ { 1 - \beta _ { 0 } } ) ) } } { \Theta ( n ^ { p - \beta _ { 0 } } ) } = \Theta ( n ^ { \frac { 1 } { 2 } + \beta _ { 0 } - p } )
$$

Without structural information on $\theta _ { e }$ beyond the simplex constraints, a natural choice for the amount of human data required to obtain (for example) a $O ( 1 / n )$ decay is $\mu _ { n } =$ $n ^ { { \frac { 5 } { 2 } } + \gamma }$ , with $\gamma > 0$ . This keeps the Fisher-Rao error uniformly controlled as the dimension grows. By contrast, the Euclidean estimate does not account for the geometric cost of placing probability mass across n coordinates: with $\mu _ { n } \sim n$ it predicts an error of order $1 / n$ . The Fisher-Rao geometry analysis above reveals that this can be misleading, as $\mu _ { n } \sim n$ does not prevent the Fisher-Rao error from growing, and in this sense, a stronger growth of $\mu _ { n }$ is required.

We devote the next section to proving Theorem 1.

## IV. PROOF OF THE MAIN RESULT

We proceed in steps, proving some intermediate lemmas before the main result. First, we show that solutions of (5) cannot approach the boundary asymptotically.

Lemma 1. Suppose that Assumptions 1-2 hold, and let $\theta ( \cdot )$ be the solution to (5). For any $\kappa \in [ 0 , 1 )$ , there exists a time:

$$
t _ { \kappa } = \frac { 1 } { 1 + \mu } \ln { \left( \frac { 1 } { 1 - \kappa } \right) } ,
$$

such that, for all $t \geqslant t _ { \kappa }$ and all $i \in \{ 1 , 2 , \ldots , n \}$

$$
\theta _ { i } ( t ) \geqslant \frac { \kappa ( \mu \delta - \eta ) } { 1 + \mu } = : \underline { { \theta } } > 0 ,\tag{11}
$$

provided that $\mu \delta > \eta .$

Proof. For each $i ,$ we have that:

$$
\dot { \theta } _ { i } = - \theta _ { i } + \mu ( \theta _ { 0 , i } - \theta _ { i } ) + \tau _ { i } ( \theta ) + \varepsilon _ { i } ,
$$

where by assumption $\tau _ { i } ( \theta ) \geqslant 0$ and $\varepsilon _ { i } \geqslant - \eta$ . Hence:

$$
\dot { \theta } _ { i } \geqslant - \theta _ { i } + \mu ( \theta _ { 0 , i } - \theta _ { i } ) - \eta = - ( 1 + \mu ) \theta _ { i } + \mu \theta _ { 0 , i } - \eta .
$$

Therefore:

$$
\theta _ { i } ( t ) \geqslant e ^ { - ( 1 + \mu ) t } \theta _ { i , 0 } + \big ( 1 - e ^ { - ( 1 + \mu ) t } \big ) \frac { \mu \theta _ { 0 , i } - \eta } { 1 + \mu } .
$$

The first term decays exponentially, while the second term approaches the steady-state value $\frac { \mu \theta _ { 0 , i } - \eta } { 1 + \mu }$ . To ensure a uniform lower bound after some finite time, we select $t \geqslant t _ { \kappa }$ such that:

$$
1 - e ^ { - ( 1 + \mu ) t } \geqslant \kappa ,
$$

where $\kappa \in [ 0 , 1 )$ , which is equivalent to enforcing that:

$$
t \geq \frac { 1 } { 1 + \mu } \ln { \left( \frac { 1 } { 1 - \kappa } \right) } .
$$

As a result, for all $t \geqslant t _ { \kappa }$ , we then obtain:

$$
\theta _ { i } ( t ) \geqslant \kappa \frac { \mu \theta _ { 0 , i } - \eta } { 1 + \mu } .
$$

Finally, since by Assumption 1 each $\theta _ { 0 , i } \geqslant \delta ,$ we have:

$$
\theta _ { i } ( t ) \geqslant \frac { \kappa ( \mu \delta - \eta ) } { 1 + \mu } = : \underline { { \theta } } ,
$$

which is positive whenever $\mu \delta > \eta .$ , establishing (11).

The uniform floor θ comes directly from the dynamics and is independent of metric; it is essential for our Fisher-Rao analysis, as the metric is not well-defined on the boundary set $\partial \Delta ^ { n }$ . We now adopt the KL-divergence between $\theta ( t )$ and the equilibrium $\theta _ { e }$ as a Lyapunov function, and establish its Lie derivative along the vector field (5).

## Lemma 2. Let:

$$
V ( \theta ) = D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) .
$$

Then, along any solution of (5) we have:

$$
\begin{array} { c } { { \dot { V } ( \theta ( t ) ) = \Big \langle \log \displaystyle \frac { \theta ( t ) } { \theta _ { e } } , \tau ( \theta ( t ) ) - \tau ( \theta _ { e } ) \Big \rangle } } \\ { { - ( 1 + \mu ) \Big \langle \log \displaystyle \frac { \theta ( t ) } { \theta _ { e } } , \theta ( t ) - \theta _ { e } \Big \rangle + \Big \langle \log \displaystyle \frac { \theta ( t ) } { \theta _ { e } } , \varepsilon ( t ) \Big \rangle . } } \end{array}\tag{12}
$$

Proof. First note that $\nabla V ( \theta ) = \log ( \theta / \theta _ { e } ) + { \bf 1 }$ , where 1 is the vector whose all entries are 1. For brevity, we now drop the dependency of the trajectories on t. Since the trajectory stays on the simplex, $\begin{array} { r } { \langle \mathbf { 1 } , \bar { \dot { \theta } } \rangle = \frac { d } { d t } \sum _ { i = 1 } ^ { n } \theta _ { i } = 0 } \end{array}$ , we have that:

$$
\dot { V } ( \theta ) = \langle \nabla V ( \theta ) , \dot { \theta } \rangle = \Big \langle \log \frac { \theta } { \theta _ { e } } , \dot { \theta } \Big \rangle .
$$

Using now (5), adding and subtracting $\tau ( \theta _ { e } )$ , we have that:

$$
\begin{array} { c } { { \displaystyle \dot { V } ( \theta ) = \Big \langle \log \frac \theta { \theta _ { e } } , \tau ( \theta ) - \tau ( \theta _ { e } ) \Big \rangle } } \\ { { + \Big \langle \log \frac \theta { \theta _ { e } } , \tau ( \theta _ { e } ) - \theta + \mu ( \theta _ { 0 } - \theta ) \Big \rangle + \Big \langle \log \frac \theta { \theta _ { e } } , \varepsilon \Big \rangle . } } \end{array}
$$

We use the equilibrium condition $\tau ( \theta _ { e } ) - \theta _ { e } + \mu ( \theta _ { 0 } - \theta _ { e } ) = 0$ to rewrite:

$$
\begin{array} { c } { { \tau ( \theta _ { e } ) - \theta + \mu ( \theta _ { 0 } - \theta ) = - \big [ ( \theta - \theta _ { e } ) + \mu ( \theta - \theta _ { e } ) \big ] } } \\ { { = - ( 1 + \mu ) ( \theta - \theta _ { e } ) , } } \end{array}
$$

which gives (12).

We are now finally fully equipped to prove the results stated in Theorem 1.

Proof of Main Theorem. For $\begin{array} { r l } { t } & { { } \geqslant ~ t _ { \kappa } , } \end{array}$ , Lemma 1 ensures $\theta _ { i } ( t ) \geqslant \underline { { \theta } }$ . From Lemma 2, along any trajectory we have:

$$
\begin{array} { r } { \dot { V } ( \theta ) = \underbrace { \left. \log \frac { \theta } { \theta _ { e } } , \tau ( \theta ) - \tau ( \theta _ { e } ) \right. } _ { \mathrm { ( A ) } } } \\ { - ( 1 + \mu ) \underbrace { \left. \log \frac { \theta } { \theta _ { e } } , \theta - \theta _ { e } \right. } _ { \mathrm { ( B ) } } + \underbrace { \left. \log \frac { \theta } { \theta _ { e } } , \varepsilon \right. } _ { \mathrm { ( C ) } } . } \end{array}
$$

For $( \mathrm { A } ) ,$ by Lemma $7 , \tau$ is $\eta _ { \mathrm { F R ^ { - } } }$ –Lipschitz in the Fisher-Rao metric, so:

$$
\Big \langle \log \frac { \theta } { \theta _ { e } } , \tau ( \theta ) - \tau ( \theta _ { e } ) \Big \rangle \leqslant \eta _ { \mathrm { F R } } \| \log \theta - \log \theta _ { e } \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } .
$$

Since $\theta _ { i } ( t ) \geqslant \underline { { \theta } } .$ , the conditions of Lemma 8 apply:

$$
\lVert \log \theta - \log \theta _ { e } \rVert _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } \leqslant a _ { 1 } V ( \theta ) ,
$$

where $\textstyle a _ { 1 } = { \frac { 1 } { c _ { 1 } } }$ , and therefore:

$$
\bigg \langle \log \frac { \theta } { \theta _ { e } } , \tau ( \theta ) - \tau ( \theta _ { e } ) \bigg \rangle \leqslant \eta _ { \mathrm { F R } } a _ { 1 } V ( \theta ) .
$$

For (B), we have that:

$$
\begin{array} { c } { \displaystyle \Big \langle \log \frac \theta { \theta _ { e } } , \theta - \theta _ { e } \Big \rangle = \sum _ { i } \theta _ { i } \log \frac \theta { \theta _ { e , i } } + \sum _ { i } \theta _ { e , i } \log \frac { \theta _ { e , i } } { \theta _ { i } } } \\ { \displaystyle = D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) + D _ { \mathrm { K L } } ( \theta _ { e } \| \theta ) \geqslant V ( \theta ) , } \end{array}
$$

and therefore:

$$
- ( 1 + \mu ) \Big \langle \log \frac { \theta } { \theta _ { e } } , \theta - \theta _ { e } \Big \rangle \leqslant - ( 1 + \mu ) V ( \theta ) .\tag{13}
$$

For (C), by Hölder’s inequality, we have that:

$$
\left| \left. \log \frac { \theta } { \theta _ { e } } , \varepsilon \right. \right| \leqslant \left\| \log \frac { \theta } { \theta _ { e } } \right\| _ { 1 } \| \varepsilon \| _ { \infty } .
$$

Letting $x = \log \theta - \log \theta _ { e }$ and using Cauchy-Schwarz:

$$
\begin{array} { l } { \displaystyle \Big \| \log \frac { \theta } { \theta _ { e } } \Big \| _ { 1 } = \| x \| _ { 1 } = \sum _ { i } | x _ { i } | } \\ { \displaystyle \leqslant \Big ( \sum _ { i } \frac { 1 } { \theta _ { e , i } } \Big ) ^ { 1 / 2 } \Big ( \sum _ { i } \theta _ { e , i } x _ { i } ^ { 2 } \Big ) ^ { 1 / 2 } . } \end{array}
$$

Note that by definition, $\begin{array} { r } { \sum _ { i } \theta _ { e , i } x _ { i } ^ { 2 } = \| \log \theta - \log \theta _ { e } \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } . } \end{array}$ Hence, using Lemma 8 again, we have that:

$$
\sum _ { i } \theta _ { e , i } x _ { i } ^ { 2 } \leqslant \frac { 1 } { c _ { 1 } } V ( \theta ) .
$$

Whenever $\theta _ { i } \geqslant \underline { { \theta } } .$ , we obtain:

$$
{ \Big \| \log \frac { \theta } { \theta _ { e } } \Big \| } _ { 1 } \leqslant \widetilde { a } _ { 2 } \sqrt { V ( \theta ) } ,
$$

where $\begin{array} { r } { \tilde { a } _ { 2 } = \sqrt { \frac { 1 } { c _ { 1 } } \sum _ { i } { 1 / \theta _ { e , i } } } } \end{array}$ . Note that since $1 / \theta _ { e , i } \leqslant 1 / \underline { { \theta } } .$ we have that $\textstyle \sum _ { i } 1 / \theta _ { e , i } \leqslant n / \underline { { \theta } }$ and hence:

$$
\begin{array} { r } { \tilde { a } _ { 2 } \leqslant a _ { 2 } : = \sqrt { \frac { n } { c _ { 1 } \underline { { \theta } } } } . } \end{array}
$$

Substituting these bounds into the expression for $\dot { V }$ yields:

$$
\dot { V } ( \theta ) \leqslant ( \eta _ { \mathrm { F R } } a _ { 1 } - ( 1 + \mu ) ) V ( \theta ) + a _ { 2 } \eta \sqrt { V ( \theta ) } , \quad \eta : = \| \varepsilon \| _ { \infty } .
$$

Next we verify that the lower bound on $\mu$ in our assumption guarantees that $( 1 + \mu ) - \eta _ { \mathrm { F R } } a _ { 1 } > 0$ . Recall that $\begin{array} { r } { \underline { { \theta } } = \frac { \kappa ( \mu \dot { \delta } - \eta ) } { 1 + \mu } } \end{array}$ and that $\begin{array} { r } { c _ { 1 } = \frac { \underline { { \theta } } } { 2 \operatorname* { m a x } _ { i } \theta _ { e , i } } } \end{array}$ by Lemma 8. Therefore:

$$
a _ { 1 } = \frac { 1 } { c _ { 1 } } = \frac { 2 \operatorname* { m a x } _ { i } \theta _ { e , i } } { \underline { { \theta } } } = \frac { 2 \operatorname* { m a x } _ { i } \theta _ { e , i } ( 1 + \mu ) } { \kappa ( \mu \delta - \eta ) } \leqslant \frac { 2 ( 1 + \mu ) } { \kappa ( \mu \delta - \eta ) } ,
$$

where we have used the fact that $\operatorname* { m a x } _ { i } \theta _ { e , i } \leqslant 1$ . Thus:

$$
\left( 1 + \mu \right) - \eta _ { \mathrm { F R } } a _ { 1 } \geqslant \left( 1 + \mu \right) \left( 1 - \frac { 2 \eta _ { \mathrm { F R } } } { \kappa ( \mu \delta - \eta ) } \right) ,
$$

and to ensure the right-hand side is positive, we require:

$$
1 - \frac { 2 \eta _ { \mathrm { F R } } } { \kappa ( \mu \delta - \eta ) } > 0 \Longleftrightarrow \mu \delta - \eta > \frac { 2 \eta _ { \mathrm { F R } } } { \kappa } ,
$$

but this is exactly enforced by the assumption that $\begin{array} { r } { \mu \geqslant \frac { \eta + \frac { 2 \eta _ { \mathrm { F R } } } { \kappa } } { \delta } } \end{array}$ . Given this, we let:

$$
0 < \lambda < ( 1 + \mu ) - \eta _ { \mathrm { F R } } a _ { 1 }
$$

and apply Young’s inequality ab $\leqslant \lambda a ^ { 2 } + \frac { b ^ { 2 } } { 4 \lambda }$ with $a = \sqrt { V ( \theta ) }$ and $b = a _ { 2 } \eta$ to obtain:

$$
\dot { V } ( \theta ) \leqslant - \big ( ( 1 + \mu ) - \eta _ { \mathrm { F R } } a _ { 1 } - \lambda \big ) V ( \theta ) + \frac { a _ { 2 } ^ { 2 } \eta ^ { 2 } } { 4 \lambda } .
$$

We can indeed choose:

$$
\begin{array} { r } { \lambda = \frac { 1 } { 2 } \Big ( ( 1 + \mu ) - \frac { \eta _ { \mathrm { F R } } } { c _ { 1 } } \Big ) , } \end{array}
$$

so that $( 1 + \mu ) - \eta _ { \mathrm { F R } } a _ { 1 } - \lambda = \lambda ;$ , and therefore:

$$
\dot { V } ( \theta ) \leqslant - \lambda V ( \theta ) + \frac { a _ { 2 } ^ { 2 } \eta ^ { 2 } } { 4 \lambda } .
$$

Consequently, for all $t \geqslant t _ { \kappa }$ we have that:

$$
V ( \theta ( t ) ) \leqslant e ^ { - \lambda ( t - t _ { \kappa } ) } V ( \theta ( t _ { \kappa } ) ) + \overline { { \epsilon } } _ { \mathrm { F R } } ^ { 2 } \big ( 1 - e ^ { - \lambda ( t - t _ { \kappa } ) } \big ) ,
$$

where: $\begin{array} { r } { \overline { { \epsilon } } _ { \mathrm { F R } } ^ { 2 } : = \frac { a _ { 2 } ^ { 2 } \eta ^ { 2 } } { 4 \lambda ^ { 2 } } } \end{array}$ . By substituting $a _ { 2 } , \lambda ,$ and $\underline { { \theta } } ,$ we have:

$$
\overline { { \epsilon } } _ { \mathrm { F R } } ^ { 2 } = \frac { n \eta ^ { 2 } ( 1 + \mu ) } { c _ { 1 } \kappa ( \mu \delta - \eta ) \big ( ( 1 + \mu ) - \frac { \eta _ { \mathrm { F R } } } { c _ { 1 } } \big ) ^ { 2 } } .\tag{14}
$$

Using:

$$
c _ { 1 } = \frac { \theta } { 2 \operatorname* { m a x } _ { i } \theta _ { e , i } } = \frac { \kappa ( \mu \delta - \eta ) } { 2 ( 1 + \mu ) \operatorname* { m a x } _ { i } \theta _ { e , i } } ,
$$

we obtain one of the terms in the denominator of (14):

$$
c _ { 1 } \kappa ( \mu \delta - \eta ) = \frac { \kappa ^ { 2 } ( \mu \delta - \eta ) ^ { 2 } } { 2 ( 1 + \mu ) \operatorname* { m a x } _ { i } \theta _ { e , i } } .
$$

To compute the other term, we write:

$$
\begin{array} { r } { ( 1 + \mu ) - \frac { \eta _ { \mathrm { F R } } } { c _ { 1 } } = ( 1 + \mu ) - \frac { 2 ( 1 + \mu ) \eta _ { \mathrm { F R } } \operatorname* { m a x } _ { i } \theta _ { e , i } } { \kappa ( \mu \delta - \eta ) } } \\ { = ( 1 + \mu ) \left( 1 - \frac { 2 \eta _ { \mathrm { F R } } \operatorname* { m a x } _ { i } \theta _ { e , i } } { \kappa ( \mu \delta - \eta ) } \right) , } \end{array}
$$

and substituting both identities yields:

$$
\overline { { \epsilon } } _ { \mathrm { F R } } ^ { 2 } = \frac { 2 n \eta ^ { 2 } \operatorname* { m a x } _ { i } \theta _ { e , i } } { \kappa ^ { 2 } ( \mu \delta - \eta ) ^ { 2 } \left( 1 - \displaystyle \frac { 2 \eta _ { \mathrm { F R } } \operatorname* { m a x } _ { i } \theta _ { e , i } } { \kappa ( \mu \delta - \eta ) } \right) ^ { 2 } } .
$$

Finally, by Lemma $\begin{array} { r } { 4 , d _ { \mathrm { F R } } ( \theta , \theta _ { e } ) \leqslant \frac { \pi } { 2 } \sqrt { V ( \theta ) } } \end{array}$ , and therefore:

$$
\begin{array} { r l r } & { } & { d _ { \mathrm { F R } } ( \theta ( t ) , \theta _ { e } ) \leqslant \displaystyle \frac { \pi } { 2 } \sqrt { e ^ { - \lambda ( t - t _ { \kappa } ) } D _ { \mathrm { K L } } ( \theta ( t _ { \kappa } ) \| \theta _ { e } ) + \overline { { \epsilon } } _ { \mathrm { F R } } ^ { 2 } } } \\ & { } & { \leqslant \displaystyle \frac { \pi } { 2 } \sqrt { e ^ { - \lambda ( t - t _ { \kappa } ) } D _ { \mathrm { K L } } ( \theta ( t _ { \kappa } ) \| \theta _ { e } ) } + \epsilon _ { \mathrm { F R } } , } \end{array}
$$

with $\in _ { \mathrm { F R } }$ matching (9), which establishes (8).

## V. CONCLUSION

In this work, we tackled the fundamental challenge of bounding the amount of human data required to prevent model collapse. By modeling the iterative training of generative models as a closed-loop stochastic process, we demonstrated that trajectories converge to a stable ball in the Fisher-Rao metric when the human-to-synthetic data ratio exceeds a specific threshold. We fundamentally departed from prior analysis that relies on the standard Euclidean metric. Ultimately, leveraging the information-geometric structure of the probability simplex naturally accounts for the distortions introduced by working on the simplex, yielding bounds that scale properly with the dimension of the underlying categorical distributions.

## REFERENCES

[1] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, et al., “A survey of large language models,” arXiv preprint arXiv:2303.18223, 2023.

[2] M. Jakesch, J. T. Hancock, and M. Naaman, “Human heuristics for AI generated language are flawed,” Proceedings of the National Academy of Sciences, vol. 120, no. 11, p. e2208839120, 2023.

[3] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell, “On the dangers of stochastic parrots: Can language models be too big?,” in Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pp. 610–623, 2021.

[4] E. Strubell, A. Ganesh, and A. McCallum, “Energy and policy considerations for deep learning in NLP,” in Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 3645–3650, 2019.

[5] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al., “Training compute-optimal large language models,” in Proceedings of the 36th International Conference on Neural Information Processing Systems, pp. 30016–30030, 2022.

[6] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al., “Llama 2: open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[7] P. Villalobos, J. Sevilla, L. Heim, T. Besiroglu, M. Hobbhahn, and A. Ho, “Will we run out of data? an analysis of the limits of scaling datasets in machine learning,” arXiv preprint arXiv:2211.04325, 2022.

[8] I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, and Y. Gal, “AI models collapse when trained on recursively generated data,” Nature, vol. 631, no. 8022, pp. 755–759, 2024.

[9] S. Alemohammad, J. Casco-Rodriguez, L. Luzi, A. I. Humayun, H. Babaei, D. LeJeune, A. Siahkoohi, and R. Baraniuk, “Self-consuming generative models go MAD,” in The Twelfth International Conference on Learning Representations, 2024.

[10] M. Marchi, S. Soatto, P. Chaudhari, and P. Tabuada, “Heat death of generative models in closed-loop learning,” in 2024 IEEE 63rd Conference on Decision and Control (CDC), pp. 1524–1530, 2024. See also arXiv:2404.02325.

[11] D. Herel and T. Mikolov, “Collapse of self-trained language models,” in The Second Tiny Papers Track at ICLR, 2024.

[12] M. Briesch, D. Sobania, and F. Rothlauf, “Large language models suffer from their own output: An analysis of the self-consuming training loop,” arXiv preprint arXiv:2311.16822, 2023.

[13] Y. Feng, E. Dohmatob, P. Yang, F. Charton, and J. Kempe, “Beyond model collapse: Scaling up with synthesized data requires verification,” in The Thirteenth International Conference on Learning Representations, 2025.

[14] M. Gerstgrasser, R. Schaeffer, A. Dey, R. Rafailov, T. Korbak, H. Sleight, R. Agrawal, J. Hughes, D. B. Pai, A. Gromov, D. Roberts, D. Yang, D. L. Donoho, and S. Koyejo, “Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data,” in First Conference on Language Modeling, 2024.

[15] B. Gharesifard and P. Tabuada, “Preventing model collapse when training LLMs with synthetic data,” in 2025 IEEE 64th Conference on Decision and Control (CDC), pp. 1124–1129, IEEE, 2025.

[16] H. K. Miyamoto, F. C. Meneghetti, J. Pinele, and S. I. Costa, “On closed-form expressions for the Fisher–Rao distance,” Information Geometry, vol. 7, no. 2, pp. 311–354, 2024.

[17] C. C. Aggarwal, A. Hinneburg, and D. A. Keim, “On the surprising behavior of distance metrics in high dimensional space,” in International conference on database theory, pp. 420–434, Springer, 2001.

[18] S.-i. Amari, Information Geometry and Its Applications, vol. 194. Springer, 2016.

[19] A. B. Tsybakov, Introduction to Nonparametric Estimation. Springer Series in Statistics, 2009.

## APPENDIX

Following we state and prove a number of technical Lemmas that the main result relies on.

Lemma 3. For any $\theta , \vartheta \in \Delta ^ { n }$ , the following inequalities hold:

$$
\begin{array} { r } { \sqrt { 2 } H ( \theta , \vartheta ) \leqslant d _ { \mathrm { F R } } ( \theta , \vartheta ) \leqslant \frac { \pi } { \sqrt { 2 } } H ( \theta , \vartheta ) . } \end{array}
$$

Proof. Let $c \ : = \ \langle \sqrt { \theta } , \sqrt { \vartheta } \rangle \in [ 0 , 1 ] .$ . By definition, $d _ { \mathrm { F R } } ( \theta , \vartheta ) = \operatorname { a r c c o s } ( \dot { c } )$ and $\dot { H } ^ { 2 } ( \theta , \vartheta ) = 1 - c .$ . Using the identity $1 - \cos x \ = \ 2 \sin ^ { 2 } ( x / 2 )$ we have $H ( \theta , \vartheta ) \ =$ $\sqrt { 2 }$ sin $\left( \textstyle \frac { 1 } { 2 } d _ { \mathrm { F R } } ( \theta , \vartheta ) \right)$ . Since $\textstyle { \frac { 2 } { \pi } } y \leqslant$ sin $y \leqslant y$ for $y \in [ 0 , \frac { \pi } { 2 } ] ,$ substituting $y = d _ { \mathrm { F R } } / 2$ yields the stated bounds. □

Lemma 4. For any $\theta , \vartheta \in \Delta _ { \mathrm { i n t } } ^ { n }$ , we have:

$$
d _ { \mathrm { F R } } ( \theta , \vartheta ) \leqslant \frac { \pi } { 2 } \sqrt { D _ { \mathrm { K L } } ( \theta \| \vartheta ) } .
$$

Proof. By [19, Lemma 2.4], the KL divergence bounds the squared Hellinger distance as $D _ { \mathrm { K L } } ( \theta \| \vartheta ) \geqslant 2 H ^ { 2 } ( \theta , \vartheta )$ . Combining this with the upper bound in Lemma 3 immediately establishes the result. □

Lemma 5. For all $u > 0$ we have:

$$
u \log u - ( u - 1 ) \geqslant ( 1 - \sqrt { u } ) ^ { 2 } .
$$

Proof. Set $s = { \sqrt { u } }$ and define $g ( s ) : = 2 s ^ { 2 } \log s - s ^ { 2 } + 2 s - 2$ We write g as $g ( a ) = 2 s ( s \log s - s + 1 )$ . Since $2 s > 0$ for $s > 0$ , it suffices to show that $h ( s ) = s \log s - s + 1 > 0$ for $s > 0$ . By Taylor’s theorem $h ( s ) = h ( 1 ) + h ^ { \prime } ( 1 ) ( s - 1 ) +$ $\textstyle { \frac { 1 } { 2 } } h ^ { \prime \prime } ( s ^ { \prime } ) ( { \overline { { s } } } - 1 ) ^ { 2 }$ for some $s ^ { \prime } > \dot { 0 } \dot { }$ . Computing the derivatives we obtain $\begin{array} { r } { h ( s ) = 1 + 0 ( s - 1 ) + \frac { 1 } { 2 } \frac { 1 } { s ^ { \prime } } ( s - 1 ) ^ { 2 } } \end{array}$ . By inspection we see that $h ( s ) > 0$ □

Lemma 6. Let $g : \mathbb { R } _ { > 0 } $ R be defined as:

$$
g ( r ) : = r \log r - ( r - 1 ) .
$$

Then, for all $r > 0 \colon$

$$
\frac 1 2 \operatorname* { m i n } \{ r , 1 \} ( \log r ) ^ { 2 } \leqslant g ( r ) \leqslant \frac 1 2 \operatorname* { m a x } \{ r , 1 \} ( \log r ) ^ { 2 } .\tag{15}
$$

Proof. We begin by computing the derivative of g:

$$
g ^ { \prime } ( r ) = \log r , \qquad r > 0 .\tag{16}
$$

We define the functions:

$$
\phi ( r ) : = 2 g ( r ) - ( \log r ) ^ { 2 } , \qquad \psi ( r ) : = r ( \log r ) ^ { 2 } - 2 g ( r ) .
$$

Since $g ( 1 ) = 0$ , we have $\phi ( 1 ) = \psi ( 1 ) = 0$ . We now prove the result by considering two cases:

Case $r \geqslant 1 .$ : For the lower bound, using (16):

$$
\phi ^ { \prime } ( r ) = 2 g ^ { \prime } ( r ) - \frac { 2 \log r } { r } = 2 \log r \Big ( 1 - \frac { 1 } { r } \Big ) .
$$

$\mathrm { ~ I f ~ } r \geqslant 1$ , then log $r \geqslant 0$ and $\begin{array} { r } { 1 - \frac { 1 } { r } \geqslant 0 . } \end{array}$ , so $\phi ^ { \prime } ( r ) \geqslant 0$ on $\lbrack 1 , \infty )$ . Thus $\phi ( r ) \geqslant \phi ( 1 ) = 0$ for all $r \geqslant 1$ , which implies:

$$
g ( r ) \geqslant { \frac { 1 } { 2 } } ( \log r ) ^ { 2 } , \qquad r \geqslant 1 .
$$

For the upper bound, we use ψ:

$$
\begin{array} { c l c r } { \displaystyle { \psi ^ { \prime } ( r ) = \frac { d } { d r } \big [ r ( \log r ) ^ { 2 } \big ] - 2 g ^ { \prime } ( r ) } } \\ { \displaystyle { = ( \log r ) ^ { 2 } + 2 \log r - 2 \log r = ( \log r ) ^ { 2 } \geqslant 0 . } } \end{array}
$$

Hence ψ is increasing on $( 0 , \infty )$ ; in particular, $\psi ( r ) \geqslant$ $\psi ( 1 ) = 0 \mathrm { f o r } r \geqslant 1$ . Therefore

$$
g ( r ) \leqslant { \frac { 1 } { 2 } } r ( \log r ) ^ { 2 } , \qquad r \geqslant 1 .
$$

This establishes the bounds for the case where $r \geqslant 1$

Case $0 < r \leqslant 1 .$ : For the lower bound, we naturally use ψ this time. As above, $\psi ^ { \prime } ( r ) = ( \log r ) ^ { 2 } \geqslant 0$ for all $r > 0 ,$ , so ψ is increasing on $( 0 , \infty )$ and:

$$
\psi ( r ) \leqslant \psi ( 1 ) = 0 , \qquad 0 < r \leqslant 1 .
$$

Thus:

$$
r ( \log r ) ^ { 2 } - 2 g ( r ) \leqslant 0 \quad \Longrightarrow \quad g ( r ) \geqslant \frac { 1 } { 2 } r ( \log r ) ^ { 2 } , \quad 0 < r \leqslant 1 .
$$

For the upper bound, we use $\phi$ again. From:

$$
\phi ^ { \prime } ( r ) = 2 \log r \Big ( 1 - { \frac { 1 } { r } } \Big ) ,
$$

we see that for $0 < r \leqslant 1$ we have log $r \leqslant 0$ and $\textstyle 1 - { \frac { 1 } { r } } \leqslant 0 .$ so $\phi ^ { \prime } ( r ) \geqslant 0$ on $( 0 , 1 ]$ . Hence $\phi$ is increasing on $( 0 , 1 \big ]$ and:

$$
\phi ( r ) \leqslant \phi ( 1 ) = 0 , \qquad 0 < r \leqslant 1 .
$$

Therefore:

$$
2 g ( r ) - ( \log r ) ^ { 2 } \leqslant 0 \quad \Longrightarrow \quad g ( r ) \leqslant \frac { 1 } { 2 } ( \log r ) ^ { 2 } , \quad 0 < r \leqslant 1 .
$$

This establishes the bound for the case where $0 < r \leqslant 1$ Combining the two cases yields (15) for al $r > 0 . \qquad \bigsqcup$

Lemma 7. Let $\begin{array} { r } { \theta _ { e } \in \Delta ^ { n } , \ T > 0 , \ \theta _ { e } ^ { \mathrm { m i n } } : = \operatorname* { m i n } _ { i } \theta _ { e , i } > 0 , } \end{array}$ and:

$$
\eta _ { \mathrm { F R } } \ : = \ \frac { 1 } { T \theta _ { e } ^ { \mathrm { m i n } } } .
$$

Then, for all $\theta \in \Delta _ { \mathrm { i n t } } ^ { n }$ the following inequality holds:

$$
\begin{array} { r } { \big \langle \log \theta \mathrm { - } \log \theta _ { e } , \tau ( \theta ) \mathrm { - } \tau ( \theta _ { e } ) \big \rangle \leqslant \eta _ { \mathrm { F R } } \| \log \theta \mathrm { - } \log \theta _ { e } \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } . } \end{array}\tag{17}
$$

Proof. Let $\varphi = \log \theta$ and $\varphi _ { e } = \log \theta _ { e }$ and consider:

$$
f ( \varphi ) : = T \log \Big ( \sum _ { i = 1 } ^ { n } e ^ { \varphi _ { i } / T } \Big ) .
$$

Clearly, $\nabla f ( \varphi ) = \tau ( \theta )$ . For $i , j \in \{ 1 , \ldots , n \}$ , we have that:

$$
\frac { \partial ( \nabla f ) _ { i } } { \partial \varphi _ { j } } = \frac { \partial \tau _ { i } ( \theta ) } { \partial \varphi _ { j } } = \frac { 1 } { T } \big ( \tau _ { i } ( \theta ) \delta _ { i j } - \tau _ { i } ( \theta ) \tau _ { j } ( \theta ) \big ) ,
$$

where $\delta _ { i j } = 1$ when $i = j$ and zero otherwise, which follows from taking the derivative of:

$$
\tau _ { i } ( \theta ) = \tau _ { i } ( \varphi ) = \frac { e ^ { \varphi _ { i } / T } } { \sum _ { k = 1 } ^ { n } e ^ { \varphi _ { k } / T } } .
$$

Therefore:

$$
\nabla ^ { 2 } f ( \varphi ) ~ = ~ \frac { 1 } { T } \Big ( \mathrm { d i a g } ( \tau ( \theta ) ) - \tau ( \theta ) \tau ( \theta ) ^ { \top } \Big ) .
$$

For simplicity of calculations to follow, we let $p : = \tau ( \theta )$ . We also let $v \in \mathbb { R } ^ { n }$ be arbitrary, with $\| v \| _ { 2 } = 1$ . Then:

$$
\begin{array} { r l r } { v ^ { \top } \Big ( \mathrm { d i a g } ( p ) - p p ^ { \top } \Big ) v = \displaystyle \sum _ { i } p _ { i } v _ { i } ^ { 2 } - \Big ( \displaystyle \sum _ { i } p _ { i } v _ { i } \Big ) ^ { 2 } } & { } & \\ { \leqslant } & { \displaystyle \sum _ { i } p _ { i } v _ { i } ^ { 2 } \leqslant \| v \| _ { 2 } ^ { 2 } , } \end{array}
$$

where we have used the fact that $\textstyle \sum _ { i } p _ { i } = 1$ . Hence, taking the supremum over unit vectors:

$$
\left\| \nabla ^ { 2 } f ( \varphi ) \right\| _ { 2 } = \frac { 1 } { T } \operatorname* { s u p } _ { \| v \| _ { 2 } = 1 } { v } ^ { \top } \Big ( \mathrm { d i a g } ( p ) - p p ^ { \top } \Big ) v \leqslant \frac { 1 } { T } .\tag{18}
$$

Therefore, for all $x , y \in \mathbb { R } ^ { n }$ we have that:

$$
\langle x - y , \nabla f ( x ) - \nabla f ( y ) \rangle \ \leqslant \ \frac { 1 } { T } \| x - y \| _ { 2 } ^ { 2 } ,
$$

where we used (18) as an upper bound for the Lipschitz constant of ∇f. Applying this with $x = \varphi$ and $y = \varphi _ { e }$ gives:

$$
\langle \varphi - \varphi _ { e } , \tau ( \theta ) - \tau ( \theta _ { e } ) \rangle \ \leqslant \ \frac { 1 } { T } \| \varphi - \varphi _ { e } \| _ { 2 } ^ { 2 } .
$$

Since diag $( \theta _ { e } ) \succeq \theta _ { e } ^ { \mathrm { m i n } } I$ , we have $\lVert \varphi - \varphi _ { e } \rVert _ { 2 } ^ { 2 } \leqslant ( \theta _ { e } ^ { \mathrm { m i n } } ) ^ { - 1 } \rVert \varphi -$ $\varphi _ { e } \big \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 }$ , and therefore, a substitution, yields (17).

Lemma 8. Fix $\underline { { \theta } } \in ( 0 , 1 )$ and let:

$$
\mathcal { T } : = \Big \{ \theta \in \Delta ^ { n } \ | \ \theta _ { i } \geqslant \underline { { \theta } } \ \mathrm { f o r \ a l l } \ i \Big \} .
$$

Then, for all $\theta \in { \mathcal { T } } .$

$$
\begin{array} { r l } & { c _ { 1 } \| \log \theta - \log \theta _ { e } \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } \leqslant D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) } \\ & { \qquad \leqslant c _ { 2 } \| \log \theta - \log \theta _ { e } \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } , } \end{array}\tag{19}
$$

with:

$$
c _ { 1 } = \frac { \theta } { 2 \operatorname* { m a x } _ { i } \theta _ { e , i } } , \qquad c _ { 2 } = \frac { 1 } { 2 \operatorname* { m i n } _ { i } \theta _ { e , i } } .
$$

Proof. Let us define:

$$
r _ { i } : = \frac { \theta _ { i } } { \theta _ { e , i } } \in ( 0 , \infty ) , \qquad x _ { i } : = \log r _ { i } = \log \theta _ { i } - \log \theta _ { e , i } .
$$

Since $\begin{array} { r } { \sum _ { i } \theta _ { i } = \sum _ { i } \theta _ { e , i } = 1 } \end{array}$ , we have $\begin{array} { r } { \sum _ { i } \theta _ { e , i } ( r _ { i } - 1 ) = 0 } \end{array}$ To proceed, observe that the KL divergence can be written as a Bregman divergence. Recall that for a convex function $f : \mathbb { R } ^ { + }  \mathbb { R }$ , the Bregman divergence between $u , v > 0$ is:

$$
D _ { f } ( u \| v ) = f ( u ) - f ( v ) - f ^ { \prime } ( v ) ( u - v ) .
$$

Consider the convex function:

$$
f ( u ) = u \log u , \qquad f ^ { \prime } ( u ) = \log u + 1 , \qquad f ^ { \prime \prime } ( u ) = \frac { 1 } { u } .
$$

For each i:

$$
\begin{array} { r l } & { D _ { f } ( r _ { i } \| 1 ) = f ( r _ { i } ) - f ( 1 ) - f ^ { \prime } ( 1 ) ( r _ { i } - 1 ) } \\ & { \qquad = r _ { i } \log r _ { i } - ( r _ { i } - 1 ) } \\ & { \qquad = : g ( r _ { i } ) , } \end{array}
$$

$$
D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) = \sum _ { i = 1 } ^ { n } \theta _ { e , i } g ( r _ { i } ) ,
$$

since $f ( 1 ) = 0$ and $f ^ { \prime } ( 1 ) = 1$ . Therefore:

as we previously established that $\begin{array} { r } { \sum _ { i } \theta _ { e , i } ( r _ { i } - 1 ) = 0 } \end{array}$ . By Lemma 6 in the Appendix, for every $r > 0 { : }$

$$
\frac 1 2 \operatorname* { m i n } \{ r , 1 \} ( \log r ) ^ { 2 } \leqslant g ( r ) \leqslant \frac 1 2 \operatorname* { m a x } \{ r , 1 \} ( \log r ) ^ { 2 } .
$$

Applying it with $r = r _ { i }$ and noting $x _ { i } = \log r _ { i }$ , we obtain:

$$
\begin{array} { r l r } {  { \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \theta _ { e , i } \operatorname* { m i n } \{ r _ { i } , 1 \} x _ { i } ^ { 2 } \leqslant D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) } } \\ & { } & { \displaystyle \leqslant \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \theta _ { e , i } \operatorname* { m a x } \{ r _ { i } , 1 \} x _ { i } ^ { 2 } . } \end{array}\tag{20}
$$

We now produce uniform bounds on the factors min $\{ r _ { i } , 1 \}$ and $\operatorname* { m a x } \{ r _ { i } , 1 \}$ . Because $\theta \in \mathcal { Z }$ and $\theta \in \Delta ^ { n }$

$$
\begin{array} { c } { \underline { { \theta } } \leqslant \theta _ { i } \leqslant 1 , \qquad 0 < \theta _ { e , i } \leqslant \bar { \theta } _ { e } : = \operatorname* { m a x } _ { j } \theta _ { e , j } , } \\ { r _ { i } = \displaystyle \frac { \theta _ { i } } { \theta _ { e , i } } \in \Big [ \frac { \underline { { \theta } } } { \theta _ { e , i } } , \frac { 1 } { \theta _ { e , i } } \Big ] . } \end{array}
$$

Hence:

$$
\begin{array} { r l } & { \operatorname* { m i n } \{ r _ { i } , 1 \} \geqslant \operatorname* { m i n } \Big \{ \frac { \underline { { \theta } } } { \theta _ { e , i } } , 1 \Big \} \geqslant \operatorname* { m i n } \Big \{ \frac { \underline { { \theta } } } { \bar { \theta } _ { e } } , 1 \Big \} , } \\ & { \qquad \operatorname* { m a x } \{ r _ { i } , 1 \} \leqslant \operatorname* { m a x } \Big \{ \frac { 1 } { \theta _ { e , i } } , 1 \Big \} = \frac { 1 } { \theta _ { e , i } } , } \end{array}
$$

where for the last equality we use the fact that $\theta _ { e , i } \leqslant 1$ Substituting these bounds into (20) gives:

$$
\frac { 1 } { 2 } \operatorname* { m i n } \Big \{ \frac { \theta } { \bar { \theta } _ { e } } , 1 \Big \} \sum _ { i = 1 } ^ { n } \theta _ { e , i } x _ { i } ^ { 2 } \leqslant D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) \leqslant \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } .\tag{21}
$$

Finally, since ${ \theta _ { e , i } \leqslant \bar { \theta } _ { e } }$ and $\theta _ { e , i } \geqslant \underline { { \theta } } _ { e } : = \operatorname* { m i n } _ { i } \theta _ { e , i } .$ , we have:

$$
\sum _ { i = 1 } ^ { n } x _ { i } ^ { 2 } \leqslant \frac { 1 } { \underline { { \theta } } _ { e } } \sum _ { i = 1 } ^ { n } \theta _ { e , i } x _ { i } ^ { 2 } .\tag{22}
$$

Combining (21) and (22) yields:

$$
\frac { 1 } { 2 } \operatorname* { m i n } \Big \{ \frac { \theta } { \bar { \theta } _ { e } } , 1 \Big \} \sum _ { i = 1 } ^ { n } \theta _ { e , i } x _ { i } ^ { 2 } \leqslant D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) \leqslant \frac { 1 } { 2 \underline { { \theta _ { e } } } } \sum _ { i = 1 } ^ { n } \theta _ { e , i } x _ { i } ^ { 2 } ,
$$

which is exactly:

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \operatorname* { m i n } \Big \{ \frac { \theta } { \bar { \theta } _ { e } } , 1 \Big \} \| \log \theta - \log \theta _ { e } \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } \leqslant D _ { \mathrm { K L } } ( \theta \| \theta _ { e } ) } \\ & { \qquad \leqslant \frac { 1 } { 2 \underline { { \theta } } _ { e } } \| \log \theta - \log \theta _ { e } \| _ { \mathrm { d i a g } ( \theta _ { e } ) } ^ { 2 } . } \end{array}
$$

Note that <sup>¯</sup>θ = max $\theta _ { e , i } \geqslant 1 / n$ while $\underline { { \theta } } \leqslant 1 / n ,$ , so $\underline { { \theta } } / \bar { \theta } _ { e } \leqslant 1$ and therefore min $\begin{array} { r } { \left\{ \frac { \theta } { \bar { \theta } _ { e } } , 1 \right\} = \frac { \theta } { \bar { \theta } _ { e } } } \end{array}$ , obtaining the lower constant $c _ { 1 } = \underline { { \theta } } / ( 2 \bar { \theta } _ { e } )$ . This establishes (19) with $c _ { 1 } = \underline { { \theta } } / ( 2 \bar { \theta } _ { e } )$ and $c _ { 2 } = 1 / ( 2 \underline { { \theta } } _ { e } )$ □