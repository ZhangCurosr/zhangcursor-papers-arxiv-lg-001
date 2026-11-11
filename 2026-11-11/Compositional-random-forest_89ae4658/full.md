# Compositional random forest

Khaled Masoumifard<sup>a,b,∗</sup>, Stephan van der Westhuizen<sup>a,b</sup>, Sugnet Gardner-Lubbe<sup>a,b</sup>

<sup>a</sup>Dept. of Statistics and Actuarial Science, Stellenbosch University, Stellenbosch, South Africa

<sup>b</sup>Centre for Multi-dimensional Data Visualisation (MuViSU), Stellenbosch University, Stellenbosch, South Africa

## Abstract

Compositional data have been analysed in the literature from three main points of view: directly on the simplex, in log-ratio coordinates, and as directional data on the unit sphere. In this paper, we construct a random forest framework for compositional responses that accommodates all three. Random forests are ensembles of decision trees, and the split criterion determines how each tree grows. We use the log-likelihood gain as the split criterion and consider distributions defined on the diferent spaces in which compositions can be represented. The Dirichlet likelihood has been studied in the literature, but its covariance is fixed by its mean up to a single scalar, and its maximum likelihood estimator has no closed form. To obtain more flexible criteria, we use two well-known transformations. First, the square-root transformation maps compositions onto the unit sphere, where existing directional distributions require numerical estimation. We introduce a tangent-normal (TN) distribution, which shares its parameters with the Kent distribution, and construct a split criterion whose gain is computed from additive suficient statistics: the shape parameters are maximised out in closed form, up to an exponentially small truncation constant, so that no parameter is estimated iteratively. Second, the log-ratio transformation maps

compositions to Euclidean space, where the multivariate logistic-normal distribution has a flexible covariance and a closed-form maximum log-likelihood, also based on additive suficient statistics. Both criteria respond to a shift of the mean, measured relative to the covariance, and to a change in the covariance itself. In simulations, the most accurate forest depended on the covariance structure of the composition. The TN forest was the most accurate in two of the three designs.

Keywords: compositional data, random forest, tangent-normal distribution, logistic-normal distribution, Dirichlet distribution

## 1. Introduction

Compositional data are vectors of non-negative components that sum to a constant and therefore lie in a simplex space. Multivariate methods developed for real space are generally inappropriate in this space (Aitchison, 1986; Pawlowsky-Glahn et al., 2015), and how compositions should be modelled has been the subject of a long methodological debate.Three of the perspectives in the compositional data analysis are: distributions defined directly on the simplex, such as the Dirichlet distribution; log-ratio transformations, which transfer the simplex to real space (Aitchison, 1982; Egozcue et al., 2003); and the square-root transformation, which transfers compositions onto the positive orthant of the unit sphere (Scealy and Welsh, 2011, 2014b; Yoon et al., 2025).

Distributions defined directly on the simplex, such as the Dirichlet, impose restrictive correlation structures (Aitchison, 1986). Aitchison (1982, 1986) proposed the log-ratio transformations and the multivariate logistic-normal (MLN) distribution (Atchison and Shen, 1980), a more flexible alternative to the Dirichlet. Another proposal concerned the directional representation of compositions. Stephens (1982) modelled square-root-transformed proportions with the von Mises distribution on the sphere, and Watson and Philip (1989) represented compositions as unit vectors in the positive orthant of the sphere and measured diferences between them by angles rather than by log-ratios. This led to an exchange of comments and replies in which Aitchison defended the log-ratio approach against these alternatives (Aitchison, 1990; Watson, 1990; Aitchison, 1991; Watson, 1991).

This debate has influenced the subsequent development of the field. Log-ratio methods became the dominant approach (Alenazi, 2023), and Scealy and Welsh (2014a) argue that this dominance reflects not only the advantages of these methods but also the way in which the debate was conducted: a set of prin ciples, including scale invariance, subcompositional coherence and permutation invariance, came to serve as a criterion for whether a method is acceptable for compositional data, so that methods that did not satisfy these principles were regarded as invalid, irrespective of their statistical properties or practical usefulness. They describe this situation as unusual in statistical data analysis. Methods outside the log-ratio framework have nevertheless continued to be developed, partly for a practical reason: log-ratio transformations are undefined when a component is zero, and for highly variable data near the boundary of the simplex, log-ratio and alternative methods can lead to substantively diferent conclusions (Scealy and Welsh, 2014a). In particular, the Dirichlet and directional approaches have received renewed attention for regression with a compositional response. Examples include Dirichlet regression models (Hijazi and Jernigan, 2009; Gueorguieva et al., 2008; Maier, 2014) and the regression models of Scealy and Welsh (2011, 2017), which fit the Kent distribution (Kent, 1982) to square-root-transformed compositions and handle zero components without replacement. Other distributions for square-root-transformed compositions include the scaled von Mises–Fisher distribution (Scealy and Wood, 2019) and the elliptically symmetric angular Gaussian (ESAG) distribution (Paine et al., 2018;

Yu and Huang, 2024), and these distributions have been used for regression with directional and compositional responses (Scealy and Wood, 2019; Paine et al., 2020; Yu and Huang, 2025; Schwob et al., 2025).

Random forests (Breiman, 2001), which are ensembles of decision trees, have also been adapted to compositional responses. A common approach is to map the data to real space with a log-ratio transformation and then apply a random forest. An alternative is to base the split criterion on a likelihood for the bounded or compositional response itself. Weinhold et al. (2020) developed a random forest based on the beta distribution for bounded outcomes, and, for compositional responses, Masoumifard et al. (2026) proposed the Dirichlet random forest, whose split criterion is the gain in Dirichlet log-likelihood. However, the Dirichlet distribution has two limitations. First, its covariance matrix is determined by its mean up to a single precision parameter, so the Dirichlet criterion cannot capture changes in the relative variances of the components or in their dependence structure. Second, its maximum likelihood estimator has no closed form (Minka, 2000) and must be computed iteratively for every candidate child node at every split. A more flexible criterion therefore requires a distribution with a less constrained covariance structure whose maximised likelihood is inexpensive to evaluate for every candidate. The MLN distribution satisfies both requirements, but its log-ratio coordinates can become highly variable when a component is close to zero. Directional distributions for square-roottransformed compositions behave well near zero, but they also sufer from the lack of a closed-form maximum likelihood estimator (see, e.g., Kent 1982; Scealy and Welsh 2011 for the Kent distribution, Scealy and Wood 2019 for the scaled von Mises–Fisher distribution, and Paine et al. 2018; Yu and Huang 2024 for the ESAG distribution). In a forest, this numerical estimation would have to be repeated for every candidate during the split search. Consequently, none of the distributions reviewed here yields a split criterion that combines a flexible covariance structure, stable behaviour near zero, and a maximised likelihood that is inexpensive to evaluate. A practical split criterion therefore requires either an approximation to the maximum likelihood estimator of an existing directional distribution or a new distribution that is computationally more convenient.

In this paper, we develop a random forest framework that accommodates all three perspectives, with three likelihood-based split criteria: the Dirichlet criterion, the MLN criterion, and a new criterion based on the tangent-normal (TN) distribution for square-root-transformed compositions. The maximised MLN log-likelihood has a closed form. For the TN criterion, we define a tangentnormal distribution on the sphere and show that, given the moment estimate of its mean direction, its maximised log-likelihood admits a closed-form approximation. We then show that both the MLN and TN criteria reward shifts in the mean, measured relative to the covariance, as well as changes in the covariance itself. Unlike the Dirichlet criterion, these two criteria can therefore respond to covariates that change the relative variances or the dependence structure of the composition. Finally, we compare the performance of the three forests through experiments on simulated datasets.

## 2. Compositional data and random forests

## 2.1. Compositional data and transformations

Consider a random vector $\mathbf { Y } = ( Y _ { 1 } , \ldots , Y _ { k } ) \in \mathbb { R } _ { + } ^ { k }$ with k the number of components. The closure of Y to a constant c is defined as

$$
\begin{array} { l } { \displaystyle { \mathcal { C } _ { c } ( { \bf Y } ) = \left( \frac { c Y _ { 1 } } { \sum _ { j = 1 } ^ { k } Y _ { j } } , \dots , \frac { c Y _ { k } } { \sum _ { j = 1 } ^ { k } Y _ { j } } \right) , } } \\ { \displaystyle { : = ( Z _ { 1 } , \dots , Z _ { k } ) } . } \end{array}\tag{1}
$$

Therefore, the observation $\mathbf { z } = ( z _ { 1 } , \ldots , z _ { k } )$ belongs to the k-dimensional simplex,

$$
\mathbf { S } ^ { k } = \Big \{ ( z _ { 1 } , \ldots , z _ { k } ) : z _ { j } > 0 , \sum _ { j = 1 } ^ { k } z _ { j } = c \Big \} .\tag{2}
$$

Furthermore, when the constant c carries no relevant information, it is standard practice to set $c = 1$ without loss of generality. Data belonging to this space are referred to as compositional data. Because of the limitations associated with the simplex space, it is common to transform data from $\mathbf { S } ^ { k }$ to a more flexible space using a suitable transformation before modelling. The following transformations are well known and widely used. The additive log-ratio (ALR) transformation transforms $\mathbf { S } ^ { k }$ to $\mathbb { R } ^ { k - 1 }$ by taking log-ratios relative to a fixed reference component, such as $z _ { k }$ ,

$$
\begin{array} { r l r } { y _ { j } = \ln \left( \frac { z _ { j } } { z _ { k } } \right) , } & { { } } & { j = 1 , \ldots , k - 1 . } \end{array}\tag{3}
$$

The centred log-ratio (CLR) transformation instead normalises each component by the geometric mean of the composition,

$$
y _ { j } = \ln \left( { \frac { z _ { j } } { g ( \mathbf { z } ) } } \right) , \qquad g ( \mathbf { z } ) = { \Big ( } \prod _ { \ell = 1 } ^ { k } z _ { \ell } { \Big ) } ^ { 1 / k } , \qquad j = 1 , \ldots , k ,\tag{4}
$$

producing a symmetric, reference-free representation at the cost of mapping into a (k − 1)-dimensional subspace of $\mathbb { R } ^ { k }$ subject to the constraint $\textstyle \sum _ { j } y _ { j } = 0$ Consequently, the induced covariance matrix is singular. The isometric log-ratio (ILR) transformation resolves this issue by projecting the CLR coordinates onto an orthonormal basis of the (k − 1)-dimensional hyperplane they span. We do not discuss the technical details further here and refer the reader to Pawlowsky-Glahn et al. (2015) for details.

A non-log-ratio alternative for compositional data is the square-root transformation, applied componentwise, which maps $\mathbf { z } \in \mathbf { S } ^ { k }$ to $\mathbf { u } = { \sqrt { \mathbf { z } } }$ on the $( k - 1 )$ - dimensional hypersphere

$$
S ^ { k - 1 } = \big \{ { \mathbf { u } } \in \mathbb { R } ^ { k } : \| { \mathbf { u } } \| = 1 \big \} .
$$

This allows the use of distributions whose natural support is $S ^ { k - 1 }$ , and in particular of methods from the directional-data literature. Under this transformation, compositions with structural zeros are admissible without the replacement or perturbation that any log-ratio transformation requires. Following common practice, we take the non-negative root throughout, so that compositions are mapped into the positive orthant

$$
{ \mathcal { O } } = \left\{ \mathbf { u } \in S ^ { k - 1 } : u _ { j } \geq 0 { \mathrm { ~ f o r ~ a l l ~ } } j \right\}
$$

rather than onto the whole of $S ^ { k - 1 }$

## 2.2. Random forests with likelihood-based splits

Let $\mathbf { x } \in \mathbb { R } ^ { p }$ be a vector of covariates and $\mathbf { Z } \in \mathbf { S } ^ { k }$ the compositional response, and let $( \mathbf { x } ^ { ( t ) } , \mathbf { z } ^ { ( t ) } ) , t = 1 , \textrm { . . . , } n$ , be the training data.

A decision tree is grown by recursive binary partitioning. Starting from a root node that contains all training observations, each node Ω is split by a covariate $j \in \{ 1 , \ldots , p \}$ and a threshold θ into the child nodes

$$
L ( \theta , j ) = \{ t \in \Omega : x _ { j } ^ { ( t ) } \leq \theta \} , \qquad R ( \theta , j ) = \{ t \in \Omega : x _ { j } ^ { ( t ) } > \theta \} ,
$$

and then the child nodes are split in the same way. Nodes that the stopping rule does not allow to be split further are the leaves, or terminal nodes. A random forest (Breiman, 2001) is an ensemble of B such trees, each grown on a resample of the training data, or on all of it, with a random subset of $m _ { \mathrm { t r y } }$ candidate covariates at each split.

The split criterion determines which candidate split $( \theta , j )$ is selected and may vary depending on the purpose of the analysis and the geometry of the data. For compositional data, following Weinhold et al. (2020); Masoumifard et al. (2026); Van der Westhuizen and Masoumifard (2026), we use likelihood-based criteria. Let $\hat { \ell } _ { F } ( A )$ denote the maximum log-likelihood under the distributional model F fitted to the observations in a node A. The selected split is

$$
\begin{array} { r } { ( \theta ^ { * } , j ^ { * } ) = \arg \operatorname* { m a x } _ { \theta , j } \Delta \ell _ { F } ( \theta , j ) , \qquad \Delta \ell _ { F } ( \theta , j ) = \hat { \ell } _ { F } \big ( L ( \theta , j ) \big ) + \hat { \ell } _ { F } \big ( R ( \theta , j ) \big ) - \hat { \ell } _ { F } ( \Omega ) . } \end{array}\tag{5}
$$

Here, Ω denotes the parent node, while $L ( \theta , j )$ and $R ( \theta , j )$ denote the two child nodes induced by the candidate split. The quantity $\Delta \ell _ { F } ( \theta , j )$ measures the improvement in maximum log-likelihood obtained by splitting the parent node into two children. A split is rewarded when two distributions describe the data of the node better than one, through a diference in location (the mean, or, for square-root coordinates, the mean direction) or in spread and dependence, to the extent that $F$ can represent them. The choice of $F ,$ , and how it determines what the criterion rewards, is the subject of Section 3.

To predict at a new point x, each tree routes x to a leaf $L _ { b } ( \mathbf { x } ) , b = 1 , \ldots , B$ and the forest assigns to training observation t the weight (Lin and Jeon, 2006)

$$
w _ { t } ( \mathbf { x } ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \frac { \mathbf { 1 } \{ \mathbf { x } ^ { ( t ) } \in L _ { b } ( \mathbf { x } ) \} } { | L _ { b } ( \mathbf { x } ) | } , \qquad \sum _ { t = 1 } ^ { n } w _ { t } ( \mathbf { x } ) = 1 .\tag{6}
$$

As in the distributional random forest of Ćevid et al. (2022), which uses weights of the same form, the conditional distribution of Z given x is estimated by the

weighted empirical distribution

$$
{ \hat { P } } ( \mathbf { z } \mid \mathbf { x } ) = \sum _ { t = 1 } ^ { n } w _ { t } ( \mathbf { x } ) 1 \{ \mathbf { z } ^ { ( t ) } \leq \mathbf { z } \} .\tag{7}
$$

Any functional of the conditional distribution follows from (7); in particular, the point prediction is its mean, $\begin{array} { r } { \hat { \mathbf { z } } ( \mathbf { x } ) = \sum _ { t } w _ { t } ( \mathbf { x } ) \mathbf { z } ^ { ( t ) } } \end{array}$ , which lies in $\mathbf { S } ^ { k }$

## 3. Distributions and split criteria

## 3.1. Dirichlet

The Dirichlet distribution is a natural choice for modelling compositional data. For $\mathbf { Z } \sim \operatorname { D i r } ( \alpha )$ , with ${ \pmb { \alpha } } = ( \alpha _ { 1 } , \ldots , \alpha _ { k } )$ and $\alpha _ { j } > 0$ , the density on $\mathbf { S } ^ { k }$ is

$$
f ( { \bf z } ; { \alpha } ) = \frac { \Gamma ( \alpha _ { 0 } ) } { \prod _ { j = 1 } ^ { k } \Gamma ( \alpha _ { j } ) } \prod _ { j = 1 } ^ { k } z _ { j } ^ { \alpha _ { j } - 1 } , \qquad \alpha _ { 0 } = \sum _ { j = 1 } ^ { k } \alpha _ { j } .\tag{8}
$$

Writing ${ \pmb { \alpha } } = \alpha _ { 0 } { \bf m }$ , with mean m $\ u \in \mathbf { S } ^ { k }$ and precision $\alpha _ { 0 }$ , the first two moments are

$$
E ( \mathbf { Z } ) = \mathbf { m } , \qquad \operatorname { C o v } ( \mathbf { Z } ) = { \frac { \mathrm { d i a g } ( \mathbf { m } ) - \mathbf { m m } ^ { \top } } { \alpha _ { 0 } + 1 } } .\tag{9}
$$

The Dirichlet random forest of Masoumifard et al. (2026) uses the Dirichlet loglikelihood as split criterion. The maximum likelihood estimators of α have no closed form and must be computed iteratively (Minka, 2000), for all candidate splits. This can be very computationally expensive and therefore we use the method of moments instead (Van der Westhuizen and Masoumifard, 2026). By (9), the covariance matrix is determined by the mean and the single scalar $1 / ( \alpha _ { 0 } + 1 )$ , and all correlations are negative. Moreover, by the delta method, for large $\alpha _ { 0 }$

$$
\operatorname { C o v } ( { \sqrt { \mathbf { Z } } } ) \approx { \frac { \mathbf { I } _ { k } - { \sqrt { \mathbf { m } } } { \sqrt { \mathbf { m } } } ^ { \top } } { 4 ( \alpha _ { 0 } + 1 ) } } ,\tag{10}
$$

so that, in square-root coordinates, a Dirichlet distribution has approximately the same variance in every direction of the tangent plane at ${ \sqrt { \mathbf { m } } } .$ This rigid covariance structure is a well-known limitation of the Dirichlet distribution (Aitchison, 1986). Data whose covariance does not follow this structure need a more flexible distribution, whose covariance is not tied to its mean.

## 3.2. Tangent-normal

As noted in Section 2, the componentwise square root $\mathbf { u } = { \sqrt { \mathbf { z } } }$ maps compositions one-to-one into the positive orthant O of the unit sphere $S ^ { k - 1 }$ . A natural choice for u is the Kent distribution (Kent, 1982), in the general-dimensional form of Scealy and Welsh (2011),

$$
f ( \mathbf { u } ) = c ( \kappa , \boldsymbol { \beta } ) ^ { - 1 } \exp \Bigl [ \kappa \mu ^ { \top } \mathbf { u } + \sum _ { m = 2 } ^ { k - 1 } \beta _ { m } \bigl ( \gamma _ { m } ^ { \top } \mathbf { u } \bigr ) ^ { 2 } - \Bigl ( \sum _ { m = 2 } ^ { k - 1 } \beta _ { m } \Bigr ) \bigl ( \gamma _ { k } ^ { \top } \mathbf { u } \bigr ) ^ { 2 } \Bigr ] , \quad \mathbf { u } \in S ^ { k - 1 } ,\tag{11}
$$

where $\pmb { \mu } \in S ^ { k - 1 }$ is the mean direction and $\kappa > 0$ the concentration. The elements of $( \gamma _ { 2 } , \ldots , \gamma _ { k } )$ are $k - 1$ orthonormal vectors orthogonal to $\pmb { \mu }$ and determine the directions along which the spread varies, while the shape parameters $\beta = \left( \beta _ { 2 } , \ldots , \beta _ { k - 1 } \right) ^ { \top }$ control how much it difers. With

$$
\beta _ { k } : = - \sum _ { m = 2 } ^ { k - 1 } \beta _ { m } ,\tag{12}
$$

the shape parameters sum to zero, and they should satisfy (Scealy and Welsh, 2011, eq. (3))

$$
\begin{array} { r } { \frac { \kappa } { 2 } > \beta _ { 2 } \geq \beta _ { 3 } \geq \cdot \cdot \cdot \geq \beta _ { k - 1 } \geq \beta _ { k } . } \end{array}\tag{13}
$$

The special case $\beta = { \bf 0 }$ is the von Mises–Fisher distribution, whose spread is the same in every direction.

Working with the Kent distribution raises two challenges. First, the density (11) is positive on the whole sphere, whereas compositions lie in O, so a Kent model gives positive probability to points with negative coordinates, which no composition can produce. Second, the split criterion (5) needs the maximum log-likelihood of every candidate child, and for the Kent distribution this is computationally expensive for every candidate child of every split. Scealy and Welsh (2011, Theorem 3) show that, as the concentration grows, the Kent distribution becomes normal in the tangent coordinates. Motivated by this result, we define a distribution that is exactly normal in these coordinates. There are other Kent-like distributions with tractable likelihoods, for example the scaled von Mises–Fisher distribution (Scealy and Wood, 2019) and the elliptically symmetric angular Gaussian distribution (Paine et al., 2018; Yu and Huang, 2024), but their shape parameters must be estimated by numerical optimisation, which a split criterion would repeat for every candidate child.

## 3.2.1. The tangent-normal distribution

Define the open unit ball in $\mathbb { R } ^ { k - 1 }$ and the open hemisphere centred at $\pmb { \mu } \in S ^ { k - 1 }$ as

$$
\mathbb { B } ^ { k - 1 } = \{ \mathbf { v } \in \mathbb { R } ^ { k - 1 } : \| \mathbf { v } \| < 1 \} , \qquad H _ { \mu } = \{ \mathbf { u } \in S ^ { k - 1 } : \mu ^ { \top } \mathbf { u } > 0 \} .
$$

Let K be a $k \times ( k - 1 )$ matrix with $\mathbf { K } ^ { \top } \mathbf { K } = \mathbf { I } _ { k - 1 }$ whose columns span the tangent space

$$
T _ { \pmb \mu } = \{ \mathbf x \in \mathbb { R } ^ { k } : \pmb \mu ^ { \top } \mathbf { x } = 0 \} .
$$

We now explore how data can be mapped from one space to another.

Lemma 1. The matrix $\mathbf { { \boldsymbol { \Gamma } } } = ( { \boldsymbol { \mu } } , \mathbf { { \boldsymbol { \kappa } } } )$ is orthogonal, and

$$
\psi ( \mathbf { v } ) = \left( 1 - \left\| \mathbf { v } \right\| ^ { 2 } \right) ^ { 1 / 2 } \pmb { \mu } + \mathbf { K } \mathbf { v } , \qquad \mathbf { v } \in \mathbb { B } ^ { k - 1 } ,\tag{14}
$$

is a bijection from $\mathbb { B } ^ { k - 1 }$ onto $H _ { \mu } ,$ with inverse $\psi ^ { - 1 } ( \mathbf { u } ) = \mathbf { K } ^ { \top } \mathbf { u }$ . Moreover, ψ is continuously diferentiable and $\mu ^ { \top } \psi ( \mathbf v ) = ( 1 - \| \mathbf v \| ^ { 2 } ) ^ { 1 / 2 }$

Proof. Since the columns of K lie in $T _ { \mu } , \mathbf { K } ^ { \intercal } \mu = \mathbf { 0 } ;$ ; together with $\mu ^ { \top } \pmb { \mu } = 1$ and $\mathbf { K } ^ { \top } \mathbf { K } = \mathbf { I } _ { k - 1 }$ , this gives $\mathbf { r } ^ { \top } \mathbf { r } = \mathbf { I } _ { k } .$ , and, as Γ is square, also $\mathbf { \Gamma } \mathbf { \Gamma } \mathbf { \Gamma } ^ { \mathsf { T } } =$ $\mu \mu ^ { \top } + \mathbf { K K } ^ { \top } = \mathbf { I } _ { k }$

From $\mathbb { B } ^ { k - 1 }$ to $H _ { \mu }$ . Let $\mathbf { v } \in \mathbb { B } ^ { k - 1 }$ and $\mathbf { u } = \psi ( \mathbf { v } )$ . Using $\mathbf { K } ^ { \top } \pmb { \mu } = \mathbf { 0 }$ and ${ \bf K } ^ { \top } { \bf K } =$ $\mathbf { I } _ { k - 1 }$

$$
\| \mathbf { u } \| ^ { 2 } = \big ( 1 - \| \mathbf { v } \| ^ { 2 } \big ) + 2 \big ( 1 - \| \mathbf { v } \| ^ { 2 } \big ) ^ { 1 / 2 } \pmb { \mu } ^ { \top } \mathbf { K } \mathbf { v } + \mathbf { v } ^ { \top } \mathbf { K } ^ { \top } \mathbf { K } \mathbf { v } = 1 , \qquad \pmb { \mu } ^ { \top } \mathbf { u } = \big ( 1 - \| \mathbf { v } \| ^ { 2 } \big ) ^ { 1 / 2 } > 0 ,
$$

so $\mathbf { u } \in H _ { \mu }$ . Moreover, $\mathbf { K } ^ { \top } \mathbf { u } = ( 1 - \| \mathbf { v } \| ^ { 2 } ) ^ { 1 / 2 } \mathbf { K } ^ { \top } \pmb { \mu } + \mathbf { K } ^ { \top } \mathbf { K } \mathbf { v } \ = \ \mathbf { v }$ , that is, $\mathbf { K } ^ { \top } \boldsymbol { \psi } ( \mathbf { v } ) = \mathbf { v }$ for every $\mathbf { v } \in \mathbb { B } ^ { k - 1 }$ , so $\psi$ is one-to-one.

From $H _ { \mu } \ t o \ \mathbb { B } ^ { k - 1 }$ . Let $\mathbf { u } \in H _ { \mu }$ and $\mathbf { v } = \mathbf { K } ^ { \top } \mathbf { u }$ . Since Γ is orthogonal,

$$
1 = \| \mathbf { u } \| ^ { 2 } = \| \Gamma ^ { \top } \mathbf { u } \| ^ { 2 } = ( { \boldsymbol { \mu } } ^ { \top } \mathbf { u } ) ^ { 2 } + \| \mathbf { v } \| ^ { 2 } .
$$

As $\mu ^ { \top } \mathbf { u } > 0$ , it follows that $\| \mathbf { v } \| < 1$ , so $\mathbf { v } \in \mathbb { B } ^ { k - 1 }$ , and $\mu ^ { \top } \mathbf { u } = ( 1 - \| \mathbf { v } \| ^ { 2 } ) ^ { 1 / 2 }$ ， taking the positive root. Then $\mathbf { r } \mathbf { r } ^ { \top } = \mathbf { I } _ { k }$ gives

$$
\mathbf { u } = { \pmb { \mu } } { \pmb { \mu } } ^ { \top } \mathbf { u } + \mathbf { K } \mathbf { K } ^ { \top } \mathbf { u } = \left( 1 - \| \mathbf { v } \| ^ { 2 } \right) ^ { 1 / 2 } { \pmb { \mu } } + \mathbf { K } \mathbf { v } = \psi ( \mathbf { v } ) ,
$$

so $\psi$ is onto $H _ { \mu }$ , and $\psi ( \mathbf { K } ^ { \top } \mathbf { u } ) = \mathbf { u }$

Finally, $\psi$ is continuously diferentiable on $\mathbb { B } ^ { k - 1 }$ , because $1 - \| \mathbf { v } \| ^ { 2 } > 0$ □

The map ψ has a simple geometric interpretation, which also explains the roles of $\mu , \mathbf { K }$ and v. Every $\mathbf { u } \in H _ { \mu }$ can be written as ${ \bf u } = \left( \cos \theta \right) \mu + \left( \sin \theta \right) { \bf e }$ , where $\theta \in [ 0 , \pi / 2 )$ is the angle between u and µ and $\mathbf { e } \in T _ { \mu }$ is a unit vector. Then $\mu ^ { \top } \mathbf { u } = \cos \theta$ and $\mathbf { v } = \mathbf { K } ^ { \top } \mathbf { u } = \left( \sin \theta \right) \mathbf { K } ^ { \top } \mathbf { e } ,$ so v holds the coordinates, in the basis K, of the orthogonal projection of u onto the tangent plane $T _ { \mu } ;$ we call them the tangent coordinates of u.

Now, let

$$
\mathbf { V } _ { 0 } \sim N _ { k - 1 } ( \mathbf { 0 } , \Lambda ) , \qquad \Lambda = \mathrm { d i a g } ( \lambda _ { 2 } , \ldots , \lambda _ { k } ) , \qquad \lambda _ { 2 } \geq \cdots \geq \lambda _ { k } > 0 ,
$$

and $\mathbf { V } \triangleq ( \mathbf { V } _ { 0 } \mid \| \mathbf { V } _ { 0 } \| < 1 )$ , where $\circeq$ denotes equality in distribution, and set

$$
\mathbf { U } = \left( 1 - \| \mathbf { V } \| ^ { 2 } \right) ^ { 1 / 2 } \pmb { \mu } + \mathbf { K } \mathbf { V } .
$$

By definition and Lemma 1, $\mathbf { V } \in \mathbb { B } ^ { k - 1 }$ and $\mathbf { U } \in H _ { \mu }$ almost surely.

Definition 1. We refer to the distribution of U as the tangent-normal distribution with mean direction $\mathbf { \Omega } _ { \mu } ,$ axes K and tangent variances $\Lambda _ { z }$ , written $\mathbf { U } \sim \mathrm { T N } ( { \pmb { \mu } } , \mathbf { K } , { \boldsymbol { \Lambda } } )$

Lemma 2. Let $\pi _ { \Lambda } = P ( \| \mathbf V _ { 0 } \| < 1 )$ and let $\varphi _ { \Lambda }$ be the density of $N _ { k - 1 } ( \mathbf { 0 } , \Lambda )$ The density of $\mathbf { U } \sim \mathrm { T N } ( { \pmb { \mu } } , \mathbf { K } , { \boldsymbol { \Lambda } } )$ with respect to the surface measure σ on $S ^ { k - 1 }$ is

$$
g ( \mathbf { u } ) = \frac { \varphi _ { \Lambda } ( \mathbf { K } ^ { \top } \mathbf { u } ) \mu ^ { \top } \mathbf { u } } { \pi _ { \Lambda } } , \qquad \mathbf { u } \in H _ { \mu } ,\tag{15}
$$

and $g ( \mathbf { u } ) = 0$ outside $H _ { \mu } ;$ that is, $\begin{array} { r } { P ( \mathbf { U } \in \mathit { E } ) = \int _ { E } g } \end{array}$ dσ for every Borel set $E \subset S ^ { k - 1 }$

Proof. By Lemma 1, $\mathbf { U } = \psi ( \mathbf { V } )$ , where $\psi$ is a continuously diferentiable bijection from $\mathbb { B } ^ { k - 1 }$ onto $H _ { \mu }$ with inverse u $\mapsto \mathbf { K } ^ { \top } \mathbf { u } .$ , and V has density $\varphi _ { \Lambda } ( \mathbf { v } ) / \pi _ { \Lambda }$ on $\mathbb { B } ^ { k - 1 }$ . Since $\mathbf { K } ^ { \top } \mathbf { K } = \mathbf { I } _ { k - 1 }$ and $\mathbf { K } ^ { \top } \pmb { \mu } = \mathbf { 0 }$ , the derivative $D \psi ( \mathbf { v } ) = \mathbf { K } - \mathbf { \mu }$ $\mu \mathbf { v } ^ { \top } / ( 1 - \| \mathbf { v } \| ^ { 2 } ) ^ { 1 / 2 }$ satisfies, by the matrix determinant lemma,

$$
\operatorname* { d e t } \bigl ( D \psi ^ { \top } D \psi \bigr ) = \operatorname* { d e t } \Bigl ( \mathbf { I } _ { k - 1 } + \frac { \mathbf { v } \mathbf { v } ^ { \top } } { 1 - \| \mathbf { v } \| ^ { 2 } } \Bigr ) = \frac { 1 } { 1 - \| \mathbf { v } \| ^ { 2 } } .
$$

By the change-of-variables formula for parametrised surfaces (Evans and Gariepy,

2015, Sec. 3.3), $\sigma ( d { \mathbf u } ) = \operatorname* { d e t } ( D \psi ^ { \top } D \psi ) ^ { 1 / 2 } d { \mathbf v }$ , that is,

$$
d \mathbf { v } = ( \pmb { \mu } ^ { \top } \mathbf { u } ) \sigma ( d \mathbf { u } ) ,\tag{16}
$$

because $\mu ^ { \top } \mathbf { u } = ( 1 - \| \mathbf { v } \| ^ { 2 } ) ^ { 1 / 2 }$ . Hence U has density $\pi _ { \Lambda } ^ { - 1 } \varphi _ { \Lambda } ( \mathbf { K } ^ { \top } \mathbf { u } ) \mu ^ { \top } \mathbf { u }$ with respect to σ on $H _ { \mu }$ , and zero outside $H _ { \mu }$ □

The factor $\mu ^ { \top } \mathfrak { \mathfrak { r } }$ u in (15) is the Jacobian (16): it turns the density of the tangent coordinates, with respect to Lebesgue measure on $T _ { \mu }$ , into a density on the sphere.

## 3.2.2. Relationship with the Kent and other spherical distributions

The tangent-normal distribution has the same location and orientation parameters as the Kent distribution, and if we set

$$
\lambda _ { m } = \left( \kappa - 2 \beta _ { m } \right) ^ { - 1 } , \qquad m = 2 , \ldots , k ,\tag{17}
$$

then the tangent variances correspond one-to-one to the concentration and shape parameters of the Kent distribution. Because $\begin{array} { r } { \sum _ { m } \beta _ { m } = 0 , \kappa = ( k - 1 ) } \end{array}$ $1 ) ^ { - 1 } \textstyle \sum _ { m } \lambda _ { m } ^ { - 1 }$ and $\beta _ { m } = { \textstyle \frac { 1 } { 2 } } ( \kappa - \lambda _ { m } ^ { - 1 } )$ , and (13) holds exactly when $\lambda _ { 2 } \geq \cdots \geq$ $\lambda _ { k } > 0$ . The two distributions therefore have the same parameter space, with $\begin{array} { r } { \frac { 1 } { 2 } k ( k + 1 ) - 1 } \end{array}$ free parameters.

The two distributions also share the structure of their moments: $E ( \mathbf { V } ) = \mathbf { 0 }$ $E \{ ( 1 - \| \mathbf { V } \| ^ { 2 } ) ^ { 1 / 2 } \mathbf { V } \} = \mathbf { 0 }$ and $E ( V _ { m } V _ { l } ) = 0$ for $m \neq l ,$ and by (14)

$$
E ( { \mathbf { U } } ) = E \big \{ ( 1 - \| { \mathbf { V } } \| ^ { 2 } ) ^ { 1 / 2 } \big \} \mu , \qquad E ( { \mathbf { U U } } ^ { \top } ) = { \mathbf { \Gamma } } \mathrm { d i a g } ( \tau _ { 1 } , \dots , \tau _ { k } ) { \mathbf { \Gamma } } \Gamma ^ { \top } ,\tag{18}
$$

with $\tau _ { m } = E ( V _ { m } ^ { 2 } )$ for $m \geq 2$ and $\begin{array} { r } { \tau _ { 1 } = 1 - \sum _ { m = 2 } ^ { k } \tau _ { m } ; } \end{array}$ ; the Kent distribution has the same structure (Scealy and Welsh, 2011, eq. (9)).

The two distributions have the same normal limit in the tangent coordinates. Scealy and Welsh (2011, Theorem 3) show that if $\kappa  \infty$ with $\beta _ { m } / \kappa  d _ { m }$ where ${ \begin{array} { l } { { \frac { 1 } { 2 } } > d _ { 2 } \geq \dots \geq d _ { k } } \end{array} }$ , then under the Kent distribution the $\kappa ^ { 1 / 2 } v _ { m }$ are asymptotically independent $N \{ 0 , ( 1 - 2 d _ { m } ) ^ { - 1 } \}$ . The tangent-normal distribution with $\lambda _ { m }$ given by (17) has the same limit. By Markov’s inequality,

$$
1 - \pi _ { \Lambda } = P \big ( \| V _ { 0 } \| ^ { 2 } \geq 1 \big ) \leq \frac { \mathbb { E } \Big [ \exp \Big ( \frac { \| V _ { 0 } \| ^ { 2 } } { 4 \lambda _ { 2 } } \Big ) \Big ] } { \exp \Big ( \frac { 1 } { 4 \lambda _ { 2 } } \Big ) } ,\tag{19}
$$

and

$$
\mathbb { E } \bigg [ \exp \bigg ( \frac { \| V _ { 0 } \| ^ { 2 } } { 4 \lambda _ { 2 } } \bigg ) \bigg ] = \prod _ { m = 2 } ^ { k } \bigg ( 1 - \frac { \lambda _ { m } } { 2 \lambda _ { 2 } } \bigg ) ^ { - 1 / 2 } \leq \prod _ { m = 2 } ^ { k } 2 ^ { 1 / 2 } = 2 ^ { ( k - 1 ) / 2 } .
$$

Hence $\begin{array} { r } { 1 - \pi _ { \Lambda } \leq 2 ^ { \frac { k - 1 } { 2 } } \exp \left( - \frac { 1 } { 4 \lambda _ { 2 } } \right) } \end{array}$ . Under (17) and the conditions of Scealy and Welsh (2011, Theorem 3) stated above,

$$
\lambda _ { 2 } = \kappa ^ { - 1 } \left( 1 - \frac { 2 \beta _ { 2 } } { \kappa } \right) ^ { - 1 } \longrightarrow 0 ,
$$

so that $1 - \pi _ { \Lambda } \to 0$ exponentially fast.

It is worth noting that the two distributions difer in several respects. The Kent distribution is defined on the whole sphere, whereas the TN distribution is supported only on the open hemisphere $H _ { \mu }$ , exactly half of the sphere. The TN distribution has a computational advantage, for example, $\pi _ { \Lambda }$ is the distribution function of a weighted sum of $\chi _ { 1 } ^ { 2 }$ variables at one, which is computed accurately by the series of Ruben (1962), whereas the normalising constant of the Kent distribution is a $( k - 1 )$ -dimensional integral with no closed form.

The omitted TN constant is also much smaller than the corresponding Kent constant (Appendix Appendix B), so dropping it from the split criterion costs less.

## 3.2.3. Split criterion

We assume that, in a node A, the observations ${ \bf u } ^ { ( t ) } , t \in \boldsymbol { A }$ , are independent with distribution $\mathrm { T N } ( \mu , { \bf K } , \Lambda )$ . By (15), the log-likelihood of the node is

$$
\ell ( A ) + J ( A ; \mu ) - n _ { A } \ln \pi _ { \Lambda } , \qquad \ell ( A ) = \sum _ { t \in A } \ln \varphi _ { \Lambda } \big ( \mathbf { K } ^ { \top } \mathbf { u } ^ { ( t ) } \big ) , \qquad J ( A ; \mu ) = \sum _ { t \in A } \ln \big ( \mu ^ { \top } \mathbf { u } ^ { ( t ) } \big ) .\tag{20}
$$

Since the term $n _ { A }$ ln $\pi _ { \Lambda }$ has no closed-form expression and is exponentially small by (19), we neglect it in the node-splitting criterion for computational eficiency. The Jacobian term, by contrast, plays an important role: $\ell ( A )$ is a log-density with respect to Lebesgue measure on $T _ { \mu }$ , which varies with $\pmb { \mu }$ and hence from node to node, and adding $J ( A ; \pmb { \mu } )$ expresses the log-density with respect to the common reference measure σ across all nodes.

We now construct the split criterion. Define

$$
n _ { A } = | { \cal A } | , \qquad { \bf s } _ { 1 } ( { \cal A } ) = \sum _ { t \in { \cal A } } { \bf u } ^ { ( t ) } , \qquad S _ { 2 } ( { \cal A } ) = \sum _ { t \in { \cal A } } { \bf u } ^ { ( t ) } { \bf u } ^ { ( t ) \top } ,\tag{21}
$$

and let ${ \hat { M } } ( A ) = S _ { 2 } ( A ) / n _ { A }$ . By (18), µ is the direction of $E ( \mathbf { U } )$ . Following Scealy and Welsh (2011), we therefore estimate $\pmb { \mu }$ using the method of moments, yielding the normalised resultant

$$
\hat { \pmb { \mu } } ( A ) = \mathbf { s } _ { 1 } ( A ) / \lVert \mathbf { s } _ { 1 } ( A ) \rVert .\tag{22}
$$

This estimator is not the maximum likelihood estimator, which does not admit a closed-form expression for $\pmb { \mu }$

Theorem 1. Let A be a node with $\hat { M } ( A )$ positive definite, and fix the mean direction at ${ \hat { \pmb { \mu } } } = { \hat { \pmb { \mu } } } ( { \cal { A } } )$ . For all orientations K with columns forming an $o r \mathrm { - }$ thonormal basis of the tangent plane $T _ { \hat { \mu } }$ , and all $\boldsymbol { \Lambda } = \mathrm { d i a g } ( \lambda _ { 2 } , \ldots , \lambda _ { k } )$ with

$\lambda _ { m } > 0$ , the maximum $o f \ell ( A )$ is

$$
\hat { \ell } _ { \mathrm { T N } } ( \cal A ) = - \frac { n _ { A } ( k - 1 ) } { 2 } \bigl ( 1 + \ln 2 \pi \bigr ) - \frac { n _ { A } } { 2 } \ln \Bigl \{ \bigl | \hat { \cal M } ( \cal A ) \bigr | \hat { \mu } ^ { \top } \hat { \cal M } ( \cal A ) ^ { - 1 } \hat { \mu } \Bigr \} .\tag{23}
$$

Proof. Let B be a $k \times ( k - 1 )$ matrix whose columns are an orthonormal basis of $T _ { \hat { \mu } }$ . The vector $\mathbf { w } ^ { ( t ) } = \mathbf { B } ^ { \top } \mathbf { u } ^ { ( t ) } \in \mathbb { R } ^ { k - 1 }$ holds the coordinates of the orthogonal projection of $\mathbf { u } ^ { ( t ) }$ onto $T _ { \hat { \mu } }$ . Every orientation whose columns are an orthonormal basis of $T _ { \hat { \mu } }$ can be written as ${ \bf K } = { \bf B } { \bf Q }$ , with $\mathbf { Q } = \mathbf { B } ^ { \top } \mathbf { K }$ orthogonal, so that $\mathbf { K } ^ { \top } \mathbf { u } ^ { ( t ) } = \mathbf { Q } ^ { \top } \mathbf { w } ^ { ( t ) }$ . Define $\mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma } \mathbf { \Sigma }$ . Then $| \Sigma | = | \Lambda |$ and $( \mathbf { Q } ^ { \top } \mathbf { w } ) ^ { \top } \Lambda ^ { - 1 } ( \mathbf { Q } ^ { \top } \mathbf { w } ) = \mathbf { w } ^ { \top } \pmb { \Sigma } ^ { - 1 } \mathbf { w } , \mathrm { s o } \ \varphi _ { \Lambda } ( \mathbf { Q } ^ { \top } \mathbf { w } ) = \varphi _ { \pmb { \Sigma } } ( \mathbf { w } )$ and

$$
\ell ( A ) = \sum _ { t \in A } \ln \varphi _ { \Sigma } \big ( { \mathbf w } ^ { ( t ) } \big ) = - \frac { n _ { A } } { 2 } \Big \{ ( k - 1 ) \ln 2 \pi + \ln | \Sigma | + \mathrm { t r } \big ( \Sigma ^ { - 1 } S _ { T } ( A ) \big ) \Big \} ,\tag{24}
$$

where $\begin{array} { r } { S _ { T } ( A ) = n _ { A } ^ { - 1 } \sum _ { t \in A } \mathbf { w } ^ { ( t ) } \mathbf { w } ^ { ( t ) \top } = \mathbf { B } ^ { \top } \hat { M } ( A ) \mathbf { B } } \end{array}$ is the second moment of the projected observations, positive definite because $\hat { M } ( A )$ is. Thus $\ell ( A )$ is the log-likelihood of the model in which the projected observations $\mathbf { w } ^ { ( t ) }$ are independent $N _ { k - 1 } ( \mathbf { 0 } , \pmb { \Sigma } )$ . As Q and Λ vary, Σ ranges over all positive definite matrices. Therefore $\ell ( A )$ is maximised at $\hat { \Sigma } = S _ { T } ( A )$ , where the trace in (24) equals $k - 1$ , so

$$
\operatorname* { m a x } _ { \mathbf { K } , \Lambda } \ell ( A ) = - \frac { n _ { A } ( k - 1 ) } { 2 } \big ( 1 + \ln 2 \pi \big ) - \frac { n _ { A } } { 2 } \ln \big | S _ { T } ( A ) \big | .\tag{25}
$$

It therefore remains to show that

$$
| S _ { T } ( A ) | = | \hat { M } ( A ) | \hat { \pmb { \mu } } ^ { \top } \hat { M } ( A ) ^ { - 1 } \hat { \pmb { \mu } } .
$$

Consider the Householder reflector (Golub and Van Loan, 2013)

$$
\mathbf { H } = \mathbf { I } - { \frac { 2 \mathbf { h } \mathbf { h } ^ { \top } } { \mathbf { h } ^ { \top } \mathbf { h } } } , \qquad \mathbf { h } = \mathbf { e } _ { 1 } - { \hat { \pmb { \mu } } } ,\tag{26}
$$

which is symmetric and orthogonal with $\mathbf { H } \mathbf { e } _ { 1 } = { \hat { \mu } }$ . Each of the last $k - 1$ columns $\mathbf { H } \mathbf { e } _ { j }$ is therefore a unit vector orthogonal to $\hat { \mu } ,$ that is, they form an orthonormal basis of $T _ { \hat { \mu } }$ . Since B was arbitrary, we set $\mathbf { H } = \left( { \hat { \pmb { \mu } } } , \mathbf { B } \right)$ . Then,

$$
\mathbf { H } \hat { M } \mathbf { H } = \left( \hat { \pmb { \mu } } ^ { \top } \right) \hat { M } \left( \hat { \pmb { \mu } } \ \mathbf { B } \right) = \left( \hat { \pmb { \mu } } ^ { \top } \hat { M } \hat { \pmb { \mu } } \quad \hat { \pmb { \mu } } ^ { \top } \hat { M } \mathbf { B }  \\ { \mathbf { B } ^ { \top } \hat { M } \hat { \pmb { \mu } } \quad \mathbf { B } ^ { \top } \hat { M } \mathbf { B } } \right) = \left( \mathbf { \bar { c } } _ { \mathbf { \mu } } \mathbf { \bar { c } } _ { T } ^ { \top } \right) ,\tag{27}
$$

with $a = \hat { \mu } ^ { \top } \hat { M } \hat { \mu }$ and $\mathbf { c } = \mathbf { B } ^ { \top } \hat { M } \hat { \boldsymbol { \mu } }$ . Since H is symmetric and orthogonal, we have

$$
| \mathbf { H } \hat { M } \mathbf { H } | = | \hat { M } | , \qquad \mathbf { e } _ { 1 } ^ { \top } ( \mathbf { H } \hat { M } \mathbf { H } ) ^ { - 1 } \mathbf { e } _ { 1 } = ( \mathbf { H } \mathbf { e } _ { 1 } ) ^ { \top } \hat { M } ^ { - 1 } ( \mathbf { H } \mathbf { e } _ { 1 } ) = \hat { \mu } ^ { \top } \hat { M } ^ { - 1 } \hat { \mu } .\tag{28}
$$

We now compute the same two quantities from (27). Since $S _ { T }$ is positive definite, let $s = a - \mathbf { c } ^ { \top } S _ { T } ^ { - 1 } \mathbf { c }$ , the Schur complement of $S _ { T }$ . Then,

$$
\begin{array} { r } { \left( \begin{array} { l l } { a } & { \mathbf { c } ^ { \top } } \\ { \mathbf { c } } & { S _ { T } } \end{array} \right) = \left( \begin{array} { l l } { 1 } & { \mathbf { c } ^ { \top } S _ { T } ^ { - 1 } } \\ { \mathbf { 0 } } & { \mathbf { I } } \end{array} \right) \left( \begin{array} { l l } { s } & { \mathbf { 0 } ^ { \top } } \\ { \mathbf { 0 } } & { S _ { T } } \end{array} \right) \left( \begin{array} { l l } { 1 } & { \mathbf { 0 } ^ { \top } } \\ { S _ { T } ^ { - 1 } \mathbf { c } } & { \mathbf { I } } \end{array} \right) . } \end{array}\tag{29}
$$

Taking determinants in (29) and using (28) gives

$$
| \hat { M } | = s | S _ { T } | .
$$

Also,

$$
\left( \mathbf { H } \hat { M } \mathbf { H } \right) ^ { - 1 } = \left( a \begin{array} { c c } { a } & { \mathbf { c } ^ { \top } } \\ { \mathbf { c } } & { S _ { T } } \end{array} \right) ^ { - 1 } = \left( \begin{array} { c c } { 1 } & { \mathbf { 0 } ^ { \top } } \\ { - S _ { T } ^ { - 1 } \mathbf { c } } & { \mathbf { I } } \end{array} \right) \left( \begin{array} { c c } { s ^ { - 1 } } & { \mathbf { 0 } ^ { \top } } \\ { \mathbf { 0 } } & { S _ { T } ^ { - 1 } } \end{array} \right) \left( \begin{array} { c c } { 1 } & { - \mathbf { c } ^ { \top } S _ { T } ^ { - 1 } } \\ { \mathbf { 0 } } & { \mathbf { I } } \end{array} \right) ,
$$

so $\mathbf { e } _ { 1 } ^ { \top } ( \mathbf { H } \hat { M } \mathbf { H } ) ^ { - 1 } \mathbf { e } _ { 1 } = s ^ { - 1 }$ . Therefore, by (28),

$$
\hat { \pmb { \mu } } ^ { \top } \hat { M } ^ { - 1 } \hat { \pmb { \mu } } = s ^ { - 1 } ,
$$

and finally

$$
| S _ { T } ( A ) | = \frac { | \hat { \cal M } ( A ) | } s = | \hat { \cal M } ( A ) | \hat { \pmb \mu } ^ { \top } \hat { \cal M } ( A ) ^ { - 1 } \hat { \pmb \mu } .
$$

Corollary 1. Under the conditions of Theorem 1, the maximum of $\ell ( A ) +$ $J ( A ; { \hat { \pmb { \mu } } } )$ over K and Λ is

$$
\hat { \ell } _ { \mathrm { T N } } ^ { * } ( A ) = \hat { \ell } _ { \mathrm { T N } } ( A ) + J ( A ) , \qquad J ( A ) = \sum _ { t \in A } \ln \bigl ( \hat { \pmb { \mu } } ( A ) ^ { \top } \mathbf { u } ^ { ( t ) } \bigr ) .\tag{30}
$$

Proof. $J ( A ; { \hat { \pmb { \mu } } } )$ does not depend on K or $\Lambda$

Of the $\begin{array} { r } { \frac { 1 } { 2 } k ( k + 1 ) - 1 } \end{array}$ parameters of a node, only the $k - 1$ of its mean direction are estimated for a candidate child. It is easy to see that

$$
\ln \bigl | S _ { T } ( A ) \bigr | = \ln \Bigl \{ \left| { \hat { M } } ( A ) \right| { \hat { \pmb { \mu } } } ^ { \top } { \hat { M } } ( A ) ^ { - 1 } { \hat { \pmb { \mu } } } \Bigr \} = \sum _ { m = 2 } ^ { k } \ln { \hat { \lambda } } _ { m } ,\tag{31}
$$

where $\hat { \lambda } _ { 2 } \geq \dots \geq \hat { \lambda } _ { k }$ , the eigenvalues of $S _ { T } ( A )$ , are the fitted tangent variances. The gain does not need them: a single Cholesky factorisation $\hat { M } ( A ) = \mathbf { L } \mathbf { L } ^ { \top }$ gives ln $| \hat { M } ( A ) | = 2 \textstyle \sum _ { j }$ ln $L _ { j j }$ and $\hat { \pmb { \mu } } ^ { \top } \hat { M } ( A ) ^ { - 1 } \hat { \pmb { \mu } } = \lVert \mathbf { L } ^ { - 1 } \hat { \pmb { \mu } } \rVert ^ { 2 }$ directly from the suficient statistics (21), which are additive over the observations of a node.

The Jacobian term, however, is not additive, and computing it exactly requires all observations across all candidate child nodes. It can instead be approximated from the suficient statistics. Let ${ c _ { t } } = \hat { \pmb { \mu } } ( \boldsymbol { A } ) ^ { \top } \mathbf { u } ^ { ( t ) } \in ( 0 , 1 ] , R _ { A } = \| \mathbf { s } _ { 1 } ( \boldsymbol { A } ) \|$ and $\bar { c } _ { A } = R _ { A } / n _ { A }$ , so that $\textstyle \sum _ { t \in A } c _ { t } = R _ { A }$ and $\textstyle \sum _ { t \in A } c _ { t } ^ { 2 } = { \hat { \mu } } ^ { \top } S _ { 2 } ( A ) { \hat { \mu } }$ . Expanding ln $c _ { t }$ to second order about ${ \bar { c } } _ { A }$ gives

$$
J ( A ) = n _ { A } \ln \bar { c } _ { A } - \frac { \hat { \pmb { \mu } } ^ { \top } S _ { 2 } ( A ) \hat { \pmb { \mu } } - n _ { A } \bar { c } _ { A } ^ { 2 } } { 2 \bar { c } _ { A } ^ { 2 } } + \epsilon _ { A } ,\tag{32}
$$

where the remainder $\epsilon _ { A }$ is of third order in $c _ { t } - { \bar { c } } _ { A }$ , which is of the order of the

tangent variances, because $0 \leq 1 - c _ { t } \leq \| \mathbf { B } _ { A } ^ { \top } \mathbf { u } ^ { ( t ) } \| ^ { 2 }$ , where the columns of the $k \times ( k - 1 )$ matrix $\mathbf { B } _ { A }$ form an orthonormal basis of $T _ { \hat { \mu } ( A ) }$

This is all that is needed to grow a tree. If the node parameters are also of interest, let

$$
\begin{array} { r } { S _ { T } ( A ) = \mathbf { W } \mathrm { d i a g } ( \hat { \lambda } _ { 2 } , \dots , \hat { \lambda } _ { k } ) \mathbf { W } ^ { \top } } \end{array}\tag{33}
$$

be the spectral decomposition of $S _ { T } ( A ) = \mathbf { B } _ { A } ^ { \top } \hat { M } ( A ) \mathbf { B } _ { A }$ , with W orthogonal. The maximisers of Corollary 1 are then the axes $\hat { \mathbf { K } } = \mathbf { B } _ { A } \mathbf { W }$ and the tangent variances $\hat { \lambda } _ { m }$ , and, by (17), the Kent parameters are

$$
\hat { \kappa } ( { \cal A } ) = \textstyle { \frac { 1 } { k - 1 } } \sum _ { m = 2 } ^ { k } \hat { \lambda } _ { m } ^ { - 1 } , \qquad \hat { \beta } _ { m } ( { \cal A } ) = \textstyle { \frac { 1 } { 2 } } \big ( \hat { \kappa } ( { \cal A } ) - \hat { \lambda } _ { m } ^ { - 1 } \big )\tag{34}
$$

(Scealy and Welsh, 2011, eq. (22)), which satisfy (13) because the $\hat { \lambda } _ { m }$ are positive and in descending order.

## 3.2.4. What the gain responds to

By Corollary 1 and (25), the gain of splitting Ω into $L$ and R is

$$
\Delta \ell = - \frac { 1 } { 2 } \Big [ n _ { L } \ln \big | S _ { T } ( L ) \big | + n _ { R } \ln \big | S _ { T } ( R ) \big | - n \ln \big | S _ { T } ( \Omega ) \big | \Big ] + \Delta J , \qquad \Delta J = J ( L ) + J ( R ) - J ( \Omega ) .\tag{35}
$$

From (35) alone, it is not immediately clear how changes in mean direction and covariance contribute to the gain. The parent and its two children are represented in their own tangent coordinates, so their covariance matrices are not directly comparable, and their mean directions are not explicitly represented in the gain. To separate these efects, we express all three nodes in a common coordinate system: the tangent coordinates of the parent. For a node $A ,$ let $\bar { \mathbf { u } } _ { A }$ and $C _ { A }$ denote the mean and the covariance matrix of the observations of A in $\mathbb { R } ^ { k }$ , and let $\mathbf { B } _ { A }$ be as in (32). Since $\bar { \mathbf { u } } _ { A }$ is parallel to ${ \hat { \pmb { \mu } } } ( A )$ and $\mathbf { B } _ { A } ^ { \top } \hat { \pmb { \mu } } ( A ) = \mathbf { 0 }$ , the tangent coordinates $\mathbf { B } _ { A } ^ { \top } \mathbf { u } ^ { ( t ) }$ have mean zero in $A ,$ and $S _ { T } ( A ) = \mathbf { B } _ { A } ^ { \top } \hat { M } ( A ) \mathbf { B } _ { A } =$ $\mathbf { B } _ { A } ^ { \top } C _ { A } \mathbf { B } _ { A }$ is their covariance matrix. The node mean is thus absorbed into the choice of the tangent plane.

Now let $\mathbf { B } = \mathbf { B } _ { \Omega }$ and $\mathbf { w } ^ { ( t ) } = \mathbf { B } ^ { \top } \mathbf { u } ^ { ( t ) }$ denote the parent’s tangent coordinates. The parent is centred at zero, but a child $c \in \{ L , R \}$ generally has a nonzero mean $\bar { \bf w } _ { c } = { \bf B } ^ { \top } \bar { \bf u } _ { c }$ . Its covariance in the parent’s coordinates is $\tilde { S } _ { T } ( c ) = \mathbf { B } ^ { \top } C _ { c } \mathbf { B }$ Thus, $S _ { T } ( c )$ and $\tilde { S } _ { T } ( c )$ represent the same observations projected onto the child’s and parent’s tangent spaces, respectively. They coincide up to a coordinate rotation when their mean directions agree, that is, when $\hat { \pmb { \mu } } ( c ) = \hat { \pmb { \mu } } ( \Omega )$ . Let $\pmb { \delta } = \bar { \bf w } _ { L } - \bar { \bf w } _ { R }$ . Then

$$
n S _ { T } ( \Omega ) = \sum _ { c \in \{ L , R \} } n _ { c } \big \{ \tilde { S } _ { T } ( c ) + \bar { \mathbf { w } } _ { c } \bar { \mathbf { w } } _ { c } ^ { \top } \big \} = n \bar { \boldsymbol { \Sigma } } + \frac { n _ { L } n _ { R } } { n } \delta \delta ^ { \top } ,\tag{36}
$$

where $\bar { \Sigma } = \{ n _ { L } \tilde { S } _ { T } ( L ) + n _ { R } \tilde { S } _ { T } ( R ) \} / n$ . Therefore,

$$
\begin{array} { r l } & { \Delta \ell = \underbrace { \cfrac { 1 } { 2 } \Big [ n \ln \big | \bar { \Sigma } \big | - \sum _ { c } n _ { c } \ln \big | \tilde { S } _ { T } ( c ) \big | \Big ] } _ { \mathrm { c o v a r i a n c e } } + \underbrace { \cfrac { n } { 2 } \ln \Big ( 1 + \frac { n _ { L } n _ { R } } { n ^ { 2 } } \delta ^ { \top } \bar { \Sigma } ^ { - 1 } \delta \Big ) } _ { \mathrm { m e a n } } } \\ & { \quad \quad + \cfrac { 1 } { 2 } \sum _ { c } n _ { c } \ln \frac { | \tilde { S } _ { T } ( c ) | } { | S _ { T } ( c ) | } + \Delta J . } \end{array}\tag{37}
$$

The covariance term is non-negative, because ln |·| is strictly concave on positive definite matrices, and zero only when $\tilde { S } _ { T } ( L ) = \tilde { S } _ { T } ( R ) ;$ it rewards a change in the tangent variances or in the orientation of the axes. The mean term is nonnegative and zero only when $\delta = { \bf 0 } ;$ it rewards a shift of the mean direction, which counts more when δ points along a direction of small tangent variance. The gain therefore responds to both the covariance and the mean.

The last two terms arise from expressing the child nodes in their own tangent coordinates rather than in the parent’s coordinates. We express all three nodes in the parent’s tangent coordinates to make the contributions to the split gain easier to interpret. This change of coordinates is closely related to the role of the

Jacobian term, which ensures that the node log-likelihoods are densities with respect to a common measure (20). We do not analyse these terms further.

## 3.3. Logistic-normal

A composition Z follows the multivariate logistic-normal distribution, $\mathbf { Z } \sim$ $\operatorname { M L N } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ , if the additive log-ratio vector $\mathbf { y } = ( y _ { 1 } , \dots , y _ { k - 1 } ) ^ { \top } , y _ { j } = \ln ( z _ { j } / z _ { k } )$ ， is $N _ { k - 1 } ( { \pmb \mu } , { \pmb \Sigma } )$ (Atchison and Shen, 1980). On $\mathbf { S } ^ { k }$ , its density is

$$
f ( \mathbf { z } ; \pmb { \mu } , \pmb { \Sigma } ) = \varphi _ { \pmb { \Sigma } } ( \mathbf { y } - \pmb { \mu } ) \prod _ { j = 1 } ^ { k } z _ { j } ^ { - 1 } .\tag{38}
$$

For the vectors $\mathbf { y } ^ { ( t ) } , t \in A$ , of a node A, the maximum likelihood estimators are

$$
{ \hat { \boldsymbol { \mu } } } ( A ) = { \frac { \mathbf { s } _ { 1 } ( A ) } { n _ { A } } } , \qquad { \hat { \boldsymbol { \Sigma } } } ( A ) = { \frac { 1 } { n _ { A } } } \sum _ { t \in A } ( \mathbf { y } ^ { ( t ) } - { \hat { \boldsymbol { \mu } } } ( A ) ) ( \mathbf { y } ^ { ( t ) } - { \hat { \boldsymbol { \mu } } } ( A ) ) ^ { \top } = { \frac { S _ { 2 } ( A ) } { n _ { A } } } - { \hat { \boldsymbol { \mu } } } ( A ) { \hat { \boldsymbol { \mu } } } ( A ) ^ { \top } ,\tag{39}
$$

where $\begin{array} { r } { \mathbf { s } _ { 1 } ( A ) = \sum _ { t \in A } \mathbf { y } ^ { ( t ) } } \end{array}$ and $\begin{array} { r } { S _ { 2 } ( A ) = \sum _ { t \in A } \mathbf { y } ^ { ( t ) } \mathbf { y } ^ { ( t ) \top } } \end{array}$ are additive suficient statistics. The gain (5) of a split $\Omega \to ( L , R )$ is

$$
\Delta \ell = - \frac { 1 } { 2 } \Big [ n _ { L } \ln \big | \hat { \Sigma } ( L ) \big | + n _ { R } \ln \big | \hat { \Sigma } ( R ) \big | - n \ln \big | \hat { \Sigma } ( \Omega ) \big | \Big ] .\tag{40}
$$

This gain has no separate location term either. As in (36), $\hat { { \bf \Sigma } } ( \Omega ) = \bar { { \bf \Sigma } } +$ $\left( n _ { L } n _ { R } / n ^ { 2 } \right) \delta \delta ^ { \top }$ , where $\pmb { \delta } = \hat { \pmb { \mu } } ( L ) - \hat { \pmb { \mu } } ( R )$ and $\bar { \Sigma } = \{ n _ { L } \hat { \Sigma } ( L ) + n _ { R } \hat { \Sigma } ( R ) \} / n$ and, as in (37),

$$
\begin{array} { r } { \Delta \ell = \underbrace { \displaystyle \frac { 1 } { 2 } \Big [ n \ln \big | \bar { \Sigma } \big | - \sum _ { c } n _ { c } \ln \big | \hat { \Sigma } ( c ) \big | \Big ] } _ { \mathrm { c o v a r i a n c e } } } \\ { + \underbrace { \frac { n } { 2 } \ln \Big ( 1 + \frac { n _ { L } n _ { R } } { n ^ { 2 } } \delta ^ { \top } \bar { \Sigma } ^ { - 1 } \delta \Big ) } _ { \mathrm { m e a n } } . } \end{array}\tag{41}
$$

This is the classical factorisation of the likelihood ratio for the equality of two normal distributions (Anderson, 2003, Ch. 10): the first term tests equal covariance matrices and rewards a change in the variances or in the dependence, and the second, an increasing function of Hotelling’s two-sample statistic, tests equal means and rewards a shift of the mean.

Like the TN distribution, the MLN distribution has $k - 1$ parameters in its mean and $\textstyle { \frac { 1 } { 2 } } k ( k - 1 )$ in its covariance, $\begin{array} { r } { \frac { 1 } { 2 } k ( k + 1 ) - 1 } \end{array}$ in all. Both models are based on the normal distribution, and the Gaussian parts of their gains, (40) and (35), have the same form and the same decomposition, (41) and (37). The diference lies in the coordinates. Unlike the tangent coordinates, the log-ratio coordinates are the same in every node, so the Jacobian $\Pi _ { j } z _ { j } ^ { - 1 }$ in (38) does not depend on the parameters and cancels from the gain. The tangent coordinates, in contrast, are defined by the node’s own mean direction, so their Jacobian does not cancel, and the TN gain has the additional location term $\Delta J .$ . The two models also behave diferently near the boundary of the simplex: the logarithm of a part close to zero is highly variable, and its variance then dominates $\hat { \Sigma }$ whereas the square root stays bounded and is defined at zero.

## 4. Simulation study

The Dirichlet random forest has been compared with other random forests for compositional responses in two recent studies (Masoumifard et al., 2026; Van der Westhuizen and Masoumifard, 2026). Rather than repeating these comparisons, we focus on comparing the three split criteria introduced in Section 3, Dirichlet, logistic-normal and TN, across a range of simulated designs.

## 4.1. Designs and evaluation

Consider a sample of size n, $( \mathbf { x } _ { i } , \mathbf { z } _ { i } ) , i = 1 , \ldots , n .$ where $\mathbf { x } _ { i } = ( x _ { i 1 } , \ldots , x _ { i 1 0 } ) ^ { \top }$ has independent components uniformly distributed on $( - 1 , 1 )$ , and $\mathbf { z } _ { i } = ( z _ { i 1 } , z _ { i 2 } , z _ { i 3 } ) ^ { \top }$ is a composition drawn from a conditional distribution given $\mathbf { x } _ { i }$ . In every design, $x _ { i 1 }$ and $x _ { i 2 }$ afect only the location of this distribution, $x _ { i 3 }$ and $x _ { i 4 }$ only its shape (spread or dependence), $x _ { i 5 }$ both, and $x _ { i 6 } , \ldots , x _ { i 1 0 }$ are noise variables; Appendix A gives the full specifications. In design D, $\mathbf { z } _ { i }$ follows a Dirichlet distribution. In design L, the additive log-ratios of $\mathbf { z } _ { i }$ follow a bivariate normal distribution with a correlation of approximately −0.9, and the location varies mainly along the direction of smallest variance. In design K, $\sqrt { \mathbf { z } _ { i } }$ follows an exact Kent distribution, one part is close to zero, and the location varies mainly along the direction of smallest variance.

All three forests, called Dir-RF, MLN-RF and TN-RF, use 1000 trees and $m _ { \mathrm { t r y } } =$ 3 candidate covariates per split, grow every tree on all training observations, and predict with the forest-weighted mean $\begin{array} { r } { \hat { \mathbf { z } } ( \mathbf { x } ) = \sum _ { t } w _ { t } ( \mathbf { x } ) \mathbf { z } ^ { ( t ) } } \end{array}$ , with the weights (6); TN-RF computes the Jacobian term by (32). The three forests use the same growth rule. A node is split only while its depth is less than 10, and only into two children that each contain at least $c = \operatorname* { m a x } ( m , k + 2 )$ observations, where m is the minimum child size, so a node with fewer than 2c observations is not split. This rule difers from the default behaviour of common random forest libraries, such as randomForest and ranger, in which the node-size parameter decides whether a node may be split, but a split may still produce a much smaller child. Our split criteria compare the maximum likelihoods of the children, and the likelihood of a small child is unstable, because its $( k - 1 ) \times ( k - 1 )$ covariance matrix is estimated from few observations; the floor $k + 2$ keeps this estimate non-singular.

All reported numbers come from independent runs with R = 40 replications: in each, n observations are drawn, 80% are used for training and 20% for testing, and the three forests see the same data, so all comparisons are paired. The designs are run at $n \in \{ 2 0 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 \}$ }. As in Masoumifard et al. (2026), accuracy on the test set $\tau$ is measured by the Aitchison distance (Aitchison et al., 2000) and the averaged root mean squared error,

$$
\mathrm { A D } = { \frac { 1 } { | T | } } \sum _ { i \in { \mathcal { T } } } \Bigl \| \mathrm { c l r } \mathbf { z } _ { i } - \mathrm { c l r } { \hat { \mathbf { z } } } ( \mathbf { x } _ { i } ) \Bigr \| ,\tag{42}
$$

$$
\mathrm { R M S E - a v e r a g e d } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \biggl [ \frac { 1 } { | { \cal T } | } \sum _ { i \in { \cal T } } \Bigl \{ z _ { i j } - \hat { z } _ { j } \bigl ( { \bf x } _ { i } \bigr ) \Bigr \} ^ { 2 } \biggr ] ^ { 1 / 2 } ,\tag{43}
$$

where clr $\mathbf { z } = ( \ln z _ { 1 } - { \bar { g } } , \dots , \ln z _ { k } - { \bar { g } } )$ , with $\bar { g } = k ^ { - 1 } \sum _ { l }$ ln $z _ { l }$ , is the centred log-ratio transform.

## 4.2. Prediction accuracy

Table 1 reports the test errors of the full designs at four sample sizes. In design D, TN-RF had the lowest Aitchison distance and RMSE-averaged at every sample size, and the diferences were significant when $n \geq 1 0 0 0$ . In design L, MLN-RF had the lowest RMSE-averaged at every sample size and the lowest Aitchison distance when $n \geq 5 0 0$ , and the diferences on both measures were significant when $n ~ \geq ~ 1 0 0 0$ . In design K, TN-RF had the lowest Aitchison distance at every sample size, significantly from $n = 5 0 0 ~ \mathrm { { o n } ; }$ it also had the lowest RMSE-averaged when $n \geq 5 0 0$ , but these diferences were not significant. Dir-RF was the least accurate forest in L at every sample size, and it had the lowest error only once, for RMSE-averaged in K at $n = 2 0 0$

TN-RF was the most accurate forest in two designs whose data do not follow its node model, and it was never the least accurate of the three forests in Table 1. In K, the square root of the composition follows a Kent distribution, which the TN distribution only approximates (Section 3.2.2), and one part is close to zero: 30% of the draws have a part below 0.01. In D, the compositions are Dirichlet, so the node model of Dir-RF is the correct one. Nevertheless, TN-RF had a lower error than Dir-RF at every sample size and on both measures. These diferences were small, below 1% of the error, but at $n = 1 0 0 0 { \mathrm { ~ a n d ~ } } n = 2 0 0 0$

Table 1: Test error at four sample sizes with minimum child size $m = 1 0$ . Entries are means over 40 paired replications, with standard deviations in parentheses; smaller is better. Bold: lowest mean for that design and n.
<table><tr><td>n</td><td>Forest</td><td>Design D</td><td>Design L</td><td>Design K</td></tr><tr><td colspan="5">Aitchison distance</td></tr><tr><td>200</td><td>Dir-RF</td><td>0.5078 (0.0608)</td><td>1.1576 (0.1277)</td><td>0.7919 (0.1674)</td></tr><tr><td></td><td>MLN-RF</td><td>0.5062 (0.0625)</td><td>1.1488 (0.1356)</td><td>0.7969 (0.1656)</td></tr><tr><td></td><td>TN-RF</td><td>0.5049 (0.0625)</td><td>1.1480 (0.1354)</td><td>0.7894 (0.1653)</td></tr><tr><td>500</td><td>Dir-RF</td><td>0.4710 (0.0356)</td><td>1.2331 (0.1001)</td><td>0.6902 (0.0911)</td></tr><tr><td></td><td>MLN-RF</td><td>0.4711 (0.0348)</td><td>1.2237 (0.0993)</td><td>0.6899 (0.0911)</td></tr><tr><td></td><td>TN-RF</td><td>0.4699 (0.0341)</td><td>1.2256 (0.0987)</td><td>0.6832 (0.0901)</td></tr><tr><td>1000</td><td>Dir-RF</td><td>0.4497 (0.0215)</td><td>1.1856 (0.0594)</td><td>0.7200 (0.0540)</td></tr><tr><td></td><td>MLN-RF</td><td>0.4483 (0.0209)</td><td>1.1785 (0.0597)</td><td>0.7196 (0.0535)</td></tr><tr><td></td><td>TN-RF</td><td>0.4471 (0.0207)</td><td>1.1806 (0.0587)</td><td>0.7128 (0.0525)</td></tr><tr><td>2000</td><td>Dir-RF</td><td>0.4368 (0.0128)</td><td>1.1970 (0.0365)</td><td>0.7072 (0.0364)</td></tr><tr><td></td><td>MLN-RF</td><td>0.4369 (0.0123)</td><td>1.1905 (0.0376)</td><td>0.7051 (0.0361)</td></tr><tr><td></td><td>TN-RF</td><td>0.4358 (0.0124)</td><td>1.1922 (0.0368)</td><td>0.6994 (0.0362)</td></tr><tr><td colspan="5">RMSE-averaged (×100)</td></tr><tr><td>200</td><td>Dir-RF</td><td>9.811 (0.79)</td><td>19.453 (1.49)</td><td>5.584 (0.60)</td></tr><tr><td></td><td>MLN-RF</td><td>9.774 (0.80)</td><td>19.162 (1.60)</td><td>5.607 (0.61)</td></tr><tr><td></td><td>TN-RF</td><td>9.736 (0.80)</td><td>19.181 (1.62)</td><td>5.598 (0.59)</td></tr><tr><td>500</td><td>Dir-RF</td><td>9.213 (0.68)</td><td>20.098 (1.14)</td><td>5.389 (0.33)</td></tr><tr><td></td><td>MLN-RF</td><td>9.215 (0.67)</td><td>19.762 (1.11)</td><td>5.381 (0.34)</td></tr><tr><td></td><td>TN-RF</td><td>9.190 (0.66)</td><td>19.847 (1.11)</td><td>5.379 (0.34)</td></tr><tr><td>1000</td><td>Dir-RF</td><td>8.822 (0.35)</td><td>19.590 (0.65)</td><td>5.460 (0.27)</td></tr><tr><td></td><td>MLN-RF</td><td>8.799 (0.34)</td><td>19.344 (0.65)</td><td>5.443 (0.26)</td></tr><tr><td></td><td>TN-RF</td><td>8.771 (0.33)</td><td>19.412 (0.63)</td><td>5.441 (0.26)</td></tr><tr><td>2000</td><td>Dir-RF</td><td>8.561 (0.27)</td><td>19.638 (0.38)</td><td>5.444 (0.20)</td></tr><tr><td></td><td>MLN-RF</td><td>8.559 (0.26)</td><td>19.421 (0.40)</td><td>5.435 (0.20)</td></tr><tr><td></td><td>TN-RF</td><td>8.539 (0.26)</td><td>19.472 (0.40)</td><td>5.430 (0.20)</td></tr></table>

they were significant against the second-best forest.

## 5. Conclusion

We include all three criteria so that the forest can be used within any of the three approaches.

The three criteria correspond to the three spaces in which compositions are usually analysed. On the simplex itself, we used the Dirichlet criterion, which was developed in earlier work (Masoumifard et al., 2026; Van der Westhuizen and Masoumifard, 2026). In log-ratio coordinates, we used the MLN criterion, whose maximum log-likelihood has a closed-form expression based on additive suficient statistics. For square-root coordinates on the sphere, we defined the TN distribution for computational reasons.

The criteria difer in what they respond to. The Dirichlet criterion judges a shift of the mean approximately as if the variance of the square-root coordinates were the same in every direction (10). The MLN and TN criteria judge it relative to the full covariance, and they also respond to a change in the covariance, in log-ratio and in tangent coordinates, respectively. The two normal models trade diferent properties: the MLN distribution is defined on a space that is in one-to-one correspondence with the interior of the simplex, but requires strictly positive parts, whereas the square-root coordinates of the TN distribution stay bounded and are defined at zero, at the price of some probability outside the positive orthant.

In our simulations, TN-RF was the most accurate of the three forests in two of the three designs and, in the full designs, never the least accurate.

## References

Aitchison, J., 1982. The statistical analysis of compositional data (with discussion). Journal of the Royal Statistical Society: Series B (Methodological) 44, 139–177.

Aitchison, J., 1986. The Statistical Analysis of Compositional Data. Chapman & Hall, London.

Aitchison, J., 1990. Comment on “measures of variability for geological data” by df watson and gm philip. Mathematical geology 22, 223–226.

Aitchison, J., 1991. Delusions of uniqueness and ineluctability. Mathematical Geology 23, 275–277. doi:10.1007/BF02066299.

Aitchison, J., Barceló-Vidal, C., Martin-Fernandez, J.A., Pawlowsky-Glahn, V., 2000. Logratio analysis and compositional distance. Mathematical Geology 32, 271–275.

Alenazi, A., 2023. A review of compositional data analysis and recent advances. Communications in Statistics - Theory and Methods 52, 5535–5567. doi:10 .1080/03610926.2021.2014890.

Anderson, T.W., 2003. An Introduction to Multivariate Statistical Analysis. 3rd ed., John Wiley & Sons, Hoboken, NJ.

Atchison, J., Shen, S., 1980. Logistic-normal distributions: Some properties and uses. Biometrika 67, 261–272. doi:10.1093/biomet/67.2.261.

Breiman, L., 2001. Random forests. Machine learning 45, 5–32.

Ćevid, D., Michel, L., Näf, J., Bühlmann, P., Meinshausen, N., 2022. Distributional random forests: Heterogeneity adjustment and multivariate distributional regression. Journal of Machine Learning Research 23, 1–79. doi:10.48550/arXiv.2005.14458.

Egozcue, J.J., Pawlowsky-Glahn, V., Mateu-Figueras, G., Barcelo-Vidal, C., 2003. Isometric logratio transformations for compositional data analysis. Mathematical Geology 35, 279–300. doi:10.1023/A:1023818214614.

Evans, L.C., Gariepy, R.F., 2015. Measure Theory and Fine Properties of Functions. Revised ed., CRC Press, Boca Raton.

Golub, G.H., Van Loan, C.F., 2013. Matrix Computations. 4th ed., Johns Hopkins University Press, Baltimore.

Gueorguieva, R., Rosenheck, R., Zelterman, D., 2008. Dirichlet component regression and its applications to psychiatric data. Computational Statistics & Data Analysis 52, 5344–5355.

Hijazi, R.H., Jernigan, R.W., 2009. Modelling compositional data using Dirichlet regression models. Journal of Applied Probability & Statistics 4, 77–91.

Kent, J.T., 1982. The Fisher–Bingham distribution on the sphere. Journal of the Royal Statistical Society: Series B (Methodological) 44, 71–80.

Lin, Y., Jeon, Y., 2006. Random forests and adaptive nearest neighbors. Journal of the American Statistical Association 101, 578–590. doi:10.1198/016214 505000001230.

Maier, M.J., 2014. DirichletReg: Dirichlet Regression for Compositional Data in R. Research Report Series / Department of Statistics and Mathematics 125. WU Vienna University of Economics and Business. Vienna. URL: https: //epub.wu.ac.at/4077/.

Masoumifard, K., van der Westhuizen, S., Gardner-Lubbe, S., 2026. Dirichletrandom forest for predicting compositional data, in: Bekker, A., Nagar, P., Ferreira, J., Erasmus, B., Ramoelo, A. (Eds.), Environmental Modelling with Contemporary Statistics: Learning, Directionality, and Space-Time Dynamics. CRC Press.

Minka, T., 2000. Estimating a dirichlet distribution.

Paine, P.J., Preston, S.P., Tsagris, M., Wood, A.T.A., 2018. An elliptically symmetric angular Gaussian distribution. Statistics and Computing 28, 689– 697. doi:10.1007/s11222-017-9756-4.

Paine, P.J., Preston, S.P., Tsagris, M., Wood, A.T.A., 2020. Spherical regres-

sion models with general covariates and anisotropic errors. Statistics and Computing 30, 153–165. doi:10.1007/s11222-019-09872-2.

Pawlowsky-Glahn, V., Egozcue, J.J., Tolosana-Delgado, R., 2015. Modeling and analysis of compositional data. John Wiley & Sons.

Ruben, H., 1962. Probability content of regions under spherical normal distributions, IV: The distribution of homogeneous and non-homogeneous quadratic functions of normal variables. The Annals of Mathematical Statistics 33, 542–570.

Scealy, J.L., Welsh, A.H., 2011. Regression for compositional data by using distributions defined on the hypersphere. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 73, 351–375. doi:10.1111/j.14 67-9868.2010.00766.x.

Scealy, J.L., Welsh, A.H., 2014a. Colours and cocktails: Compositional data analysis 2013 Lancaster lecture. Australian & New Zealand Journal of Statistics 56, 145–169. doi:10.1111/anzs.12073.

Scealy, J.L., Welsh, A.H., 2014b. Fitting Kent models to compositional data with small concentration. Statistics and Computing 24, 165–179. doi:10.100 7/s11222-012-9361-5.

Scealy, J.L., Welsh, A.H., 2017. A directional mixed efects model for compositional expenditure data. Journal of the American Statistical Association 112, 24–36. doi:10.1080/01621459.2016.1189336.

Scealy, J.L., Wood, A.T.A., 2019. Scaled von Mises–Fisher distributions and regression models for paleomagnetic directional data. Journal of the American Statistical Association 114, 1547–1560. doi:10.1080/01621459.2019.1585 249.

Schwob, M.R., Hooten, M.B., Calzada, N.M., Keitt, T.H., 2025. Spatial hyperspheric models for compositional data. Annals of Applied Statistics 19, 2644–2663. doi:10.1214/25-AOAS2082.

Stephens, M.A., 1982. Use of the von Mises distribution to analyse continuous proportions. Biometrika 69, 197–203. doi:10.1093/biomet/69.1.197.

Watson, D., 1991. Reply to “delusions of uniqueness and ineluctability” by j. aitchison. Mathematical Geology 23, 279–279.

Watson, D.F., 1990. Reply to comment on “Measures of variability for geological data” by D. F. Watson and G. M. Philip. Mathematical Geology 22, 227–231. doi:10.1007/BF00891827.

Watson, D.F., Philip, G.M., 1989. Measures of variability for geological data. Mathematical Geology 21, 233–254. doi:10.1007/BF00893217.

Weinhold, L., Schmid, M., Mitchell, R., Maloney, K.O., Wright, M.N., Berger, M., 2020. A random forest approach for bounded outcome variables. Journal of Computational and Graphical Statistics 29, 639–658.

Van der Westhuizen, S., Masoumifard, K., 2026. Machine learning for modelling compositional soil properties in digital soil mapping. CATENA 275, 110650.

Yoon, C., Choi, H., Ahn, J., 2025. Kernel density estimation for compositional data with zeros via hypersphere mapping. Computational Statistics and Data Analysis 212, 108249. doi:10.1016/j.csda.2025.108249.

Yu, Z., Huang, X., 2024. A new parameterization for elliptically symmetric angular Gaussian distributions of arbitrary dimension. Electronic Journal of Statistics 18, 301–334. doi:10.1214/23-EJS2210.

Yu, Z., Huang, X., 2025. Regression analysis of elliptically symmetric directional data. Computational Statistics and Data Analysis 208, 108167. doi:10.101 $6 / \mathrm { j }$ .csda.2025.108167.

## Appendix A. Simulation designs and additional results

In every design, the location enters through the score

$$
\begin{array} { r } { \eta ( \mathbf { x } ) = \sin ( \pi x _ { 1 } ) + 1 . 5 \left( x _ { 2 } ^ { 2 } - \frac { 1 } { 3 } \right) + 0 . 5 x _ { 5 } , } \end{array}\tag{A.1}
$$

and the shape through $\begin{array} { r } { s _ { 3 } ( \mathbf { x } ) = \operatorname { t a n h } ( 2 x _ { 3 } ) , s _ { 4 } ( \mathbf { x } ) = \frac { 1 } { 2 } \operatorname { s i g n } ( x _ { 4 } ) } \end{array}$ and $x _ { 5 }$ . We use C for the closure operation, which divides a positive vector by the sum of its entries.

Design D (Dirichlet). The response is

$$
\mathbf { Z } \mid \mathbf { x } \sim \operatorname { D i r } \left( \alpha _ { 0 } ( \mathbf { x } ) \mathbf { m } ( \mathbf { x } ) \right) , \qquad \mathbf { m } ( \mathbf { x } ) = { \mathcal { C } } \left( e ^ { 0 . 8 \sin ( \pi x _ { 1 } ) + 0 . 5 x _ { 5 } } , e ^ { 1 . 2 ( x _ { 2 } ^ { 2 } - 1 / 3 ) } , 1 \right) ,\tag{A.2}
$$

with $\alpha _ { 0 } ( \mathbf { x } ) = 4 0 \exp \{ 0 . 6 s _ { 3 } ( \mathbf { x } ) + 0 . 6 s _ { 4 } ( \mathbf { x } ) + 0 . 4 x _ { 5 } \}$ . The shape covariates change only the precision, so $E ( \mathbf { Z } \mid \mathbf { x } ) = \mathbf { m } ( \mathbf { x } )$ does not depend on them, and, by (10), the covariance of $\sqrt { \mathbf { Z } }$ is approximately isotropic.

Design L (logistic-normal). The additive log-ratios $\mathbf { y } = ( \ln Z _ { 1 } / Z _ { 3 } , \ln Z _ { 2 } / Z _ { 3 } ) ^ { \top }$ are

$$
\begin{array} { r } { \mathbf { y } \mid \mathbf { x } \sim N _ { 2 } \big ( \pmb { \mu } ( \mathbf { x } ) , \Sigma ( \mathbf { x } ) \big ) , \qquad \Sigma ( \mathbf { x } ) = \sigma ^ { 2 } ( \mathbf { x } ) \left( \begin{array} { l l } { 1 } & { \rho ( \mathbf { x } ) } \\ { \rho ( \mathbf { x } ) } & { 1 } \end{array} \right) , } \end{array}\tag{A.3}
$$

with $\rho ( \mathbf { x } ) = - 0 . 9 + 0 . 0 5 s _ { 3 } ( \mathbf { x } )$ and $\sigma ^ { 2 } ( { \bf x } ) = \exp \{ 0 . 5 s _ { 4 } ( { \bf x } ) + 0 . 3 x _ { 5 } \}$ . Because $\rho < 0$ , the minor axis of Σ is $\mathbf { e } _ { \mathrm { m i n } } = ( 1 , 1 ) ^ { \top } / \sqrt { 2 }$ , with variance $\sigma ^ { 2 } ( 1 + \rho ) \approx 0 . 1 \sigma ^ { 2 }$

and the major axis is $\mathbf { e } _ { \mathrm { m a x } } = ( 1 , - 1 ) ^ { \top } / \sqrt { 2 }$ , with variance $\sigma ^ { 2 } ( 1 - \rho ) \approx 1 . 9 \sigma ^ { 2 }$ The mean moves mainly along the minor axis,

$$
\mu ( \mathbf { x } ) = 1 . 5 \ : ( 0 . 1 ) ^ { 1 / 2 } \ : \eta ( \mathbf { x } ) \ : \big \{ \cos ( \pi / 8 ) \ : \mathbf { e } _ { \mathrm { m i n } } + \sin ( \pi / 8 ) \ : \mathbf { e } _ { \mathrm { m a x } } \big \} ,
$$

so that one unit of $\eta$ moves the mean by about 1.5 minor-axis standard deviations. A Dirichlet distribution cannot represent this dependence.

Design $K \ ( K e n t )$ . The square root of the composition follows an exact Kent distribution. Let $\pmb { \mu } _ { 0 } = \sqrt { \mathcal { C } ( 1 , 1 , 0 . 0 4 0 ) }$ . At a direction $\textstyle \mu ,$ set $\mathbf { s } ( \pmb { \mu } ) = \mathbf { \left( I - \alpha \right. }$ $\pmb { \mu } \pmb { \mu } ^ { \top } ) \mathbf { e } _ { 3 } / \| ( \mathbf { I } - \pmb { \mu } \pmb { \mu } ^ { \top } ) \mathbf { e } _ { 3 } \|$ and $1 ( \mu ) = { \bf s } ( \mu ) \times \mu$ . Given x, U has the Kent density (11) with $k = 3$

$$
f ( \mathbf { u } \mid \mathbf { x } ) \propto \exp { \Big [ } \kappa ( \mathbf { x } ) \mu ( \mathbf { x } ) ^ { \top } \mathbf { u } + \beta ( \mathbf { x } ) \{ ( \mathbf { l } ^ { \top } \mathbf { u } ) ^ { 2 } - ( \mathbf { s } ^ { \top } \mathbf { u } ) ^ { 2 } \} { \Big ] } ,\tag{A.4}
$$

where $1 = 1 ( \mu ( \mathbf { x } ) ) , \mathbf { s } = \mathbf { s } ( \mu ( \mathbf { x } ) ) , \kappa ( \mathbf { x } ) = 3 3 0 \exp \{ 0 . 5 s _ { 4 } ( \mathbf { x } ) + 0 . 3 x _ { 5 } \} \mathrm { a n d } \beta ( \mathbf { x } )$ $\beta ( \mathbf { x } ) =$ $d ( \mathbf { x } ) \kappa ( \mathbf { x } )$ . Also,

$$
\mu ( \mathbf { x } ) \propto \mu _ { 0 } + 1 . 5 \bar { \lambda } _ { \mathrm { s } } ^ { 1 / 2 } \eta ( \mathbf { x } ) \left\{ \cos ( 3 \pi / 1 6 ) \mathbf { s } ( \mu _ { 0 } ) + \sin ( 3 \pi / 1 6 ) \mathbf { l } ( \mu _ { 0 } ) \right\} ,
$$

normalised to unit length, where $\bar { \lambda } _ { \mathrm { s } } = 1 / ( 3 3 0 \times 1 . 4 )$ is the short-axis variance at the base values.

Table A.2: Test error at $n = 1 0 0 0$ with minimum child size $m = 1 0$ and $m = 2 0$ Entries are means over 40 paired replications. Bold: lowest mean in the row for that measure.
<table><tr><td rowspan="2">Design</td><td rowspan="2">m</td><td colspan="3">Aitchison distance</td><td colspan="3">RMSE-averaged (×100)</td></tr><tr><td>Dir</td><td>MLN</td><td>TN</td><td>Dir</td><td>MLN</td><td>TN</td></tr><tr><td>D</td><td>10</td><td>0.4497</td><td>0.4483</td><td>0.4471</td><td>8.822</td><td>8.799</td><td>8.771</td></tr><tr><td></td><td>20</td><td>0.4644</td><td>0.4631</td><td>0.4617</td><td>9.099</td><td>9.075</td><td>9.044</td></tr><tr><td>L</td><td>10</td><td>1.1856</td><td>1.1785</td><td>1.1806</td><td>19.590</td><td>19.344</td><td>19.412</td></tr><tr><td></td><td>20</td><td>1.1842</td><td>1.1779</td><td>1.1795</td><td>19.520</td><td>19.311</td><td>19.357</td></tr><tr><td>K</td><td>10</td><td>0.7200</td><td>0.7196</td><td>0.7128</td><td>5.460</td><td>5.443</td><td>5.441</td></tr><tr><td></td><td>20</td><td>0.7308</td><td>0.7315</td><td>0.7243</td><td>5.460</td><td>5.458</td><td>5.451</td></tr></table>

## Appendix B. Normalising constants of the TN and Kent distributions

We compared the constants that are omitted when the log-likelihood of each distribution is replaced by the Gaussian log-likelihood of its tangent coordinates. The settings were $k = 3 , 4 , 5$ , matched parameters (17), $d _ { m } = \beta _ { m } / \kappa$ equally spaced from d to −d with d $\in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 4 5 \}$ , and $2 \leq \kappa \leq 1 0 0 0$

For the TN distribution, the omitted constant is $- \ln \pi _ { \Lambda }$ , computed from the series of Ruben (1962). For the Kent distribution, it is ln $c _ { G } - \ln c ( \kappa , \beta )$ 2 where $c _ { G } = e ^ { \kappa } ( 2 \pi ) ^ { ( k - 1 ) / 2 } | \Lambda | ^ { 1 / 2 }$ and $c ( \kappa , \beta )$ was computed by nested adaptive quadrature. Both agreed with closed forms and with independent numerical methods to within $1 0 ^ { - 1 2 }$

The TN constant decays as $\exp \{ - \kappa ( 1 - 2 d _ { 2 } ) / 2 \}$ , up to a factor polynomial in $\kappa ,$ and exceeded $1 0 ^ { - 3 }$ nats only when $\kappa ( 1 - 2 d _ { 2 } ) \leq 1 5$ . The Kent constant decays as $\kappa ^ { - 1 } \{ ( a _ { 1 } ^ { 2 } + 2 a _ { 2 } ) / 8 - a _ { 1 } / 2 \}$ , with $\begin{array} { r } { a _ { r } = \sum _ { m } ( 1 - 2 d _ { m } ) ^ { - r } } \end{array}$ . For $k = 3$ and $d _ { 2 } = - d _ { 3 } = 0 . 2$ , it exceeded $1 0 ^ { - 3 }$ nats up to $\kappa = 3 3 0$ , and for $k = 5$ over the whole range studied.