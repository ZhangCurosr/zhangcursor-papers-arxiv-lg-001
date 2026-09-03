# IFW-BLS: Dual-Robust Broad Learning System with Intuitionistic Fuzzy Wave Loss

Mushir Akhtar and M. Tanveer

Indian Institute of Technology Indore, Simrol, Indore, India {phd2101241004,mtanveer}@iiti.ac.in

Abstract. Broad Learning System is an eficient randomized learning model that expands network width through feature and enhancement nodes and estimates the output weights without deep backpropagation. Its standard least-squares training, however, is vulnerable in two diferent ways: (i) large residuals caused by noise, outliers, or corrupted labels can dominate the objective, and (ii) all samples are treated as equally reliable even when some lie in ambiguous or locally conflicting regions. This paper proposes IFW-BLS, an Intuitionistic Fuzzy Wave Broad Learning System that addresses these two sources of fragility within one optimization model. The first robustness mechanism is residual-level protection, obtained by replacing the squared loss with the bounded, smooth, and asymmetric wave loss. Boundedness prevents extreme residuals from receiving unbounded influence, while asymmetry allows positive and negative deviations to be penalized diferently when the dominant error direction varies. The second mechanism is sample-level credibility control, obtained through intuitionistic fuzzy scores that combine global classcenter consistency with local neighborhood conflict. The resulting model evaluates the wave loss on credibility-weighted residuals, so unreliable samples are down-weighted before the bounded loss further limits the effect of extreme errors. A Nesterov accelerated gradient based optimizer is used to solve the proposed objective, avoiding the explicit matrix inversion used in conventional BLS. Experiments on UCI benchmark datasets validate the superiority of the proposed IFW-BLS model over the baseline models; additional corruption experiments also show more stable performance than BLS under noise and outlier contamination.

Keywords: Broad Learning System · Intuitionistic Fuzzy Learning · Wave Loss · Robust Classification · Randomized Neural Networks

## 1 Introduction

Randomized neural networks have become an attractive alternative to deeply trained architectures because they reduce the number of trainable parameters and often avoid expensive backpropagation-based learning [23, 8]. Models such as random vector functional link (RVFL) networks [17, 16] and extreme learning machine (ELM) [14] randomly generate hidden transformations and learn only the output layer, which leads to fast training and simple implementation. Recent work also shows that the randomized representation itself can be made data-aware: CAWI samples input-to-hidden weights from a copula fitted to the training features, preserving inter-feature dependence while retaining closed-form output learning [4]. The Broad Learning System (BLS) extends the randomized-learning idea by replacing depth with width: multiple groups of feature nodes and enhancement nodes are generated and concatenated into a broad representation [9, 10]. This design gives BLS a favorable balance between representation ability and computational eficiency.

Despite these advantages, standard BLS remains fragile under uncertain data. Its output weights are usually obtained from a regularized least-squares problem, so large errors receive quadratic penalties. A few noisy observations, outliers, or mislabeled samples can therefore have a disproportionate efect on the learned output weights. Related robust-loss research has addressed this weakness in other learning settings. The HawkEye loss combines boundedness and smoothness with an insensitive zone for robust regression [3], and its integration with an RVFL network yields a robust classifier for noisy and outlier-prone data [1]. In addition, the wave loss has been used in multiview learning to improve robustness while exploiting both consensus and complementary information across views [18]. These results demonstrate the broader value of bounded robust losses. The conventional BLS objective, however, also does not distinguish between highly credible samples and samples located far from their class structure or near conflicting neighbors. These two issues are related but not identical. A robust loss controls the influence of large residuals after the model makes an error, whereas credibility-aware learning controls how much each sample should participate in training before its residual is penalized.

Existing BLS variants address parts of this problem from diferent angles. Regularized BLS improves model control through additional penalties [15]; fuzzy BLS incorporates fuzzy subsystems for uncertainty handling [13]; intuitionistic fuzzy BLS uses membership, nonmembership, and hesitation information to describe sample reliability [20]; and robust collaborative BLS models improve learning under noisy labels through more elaborate correction or kernel-based mechanisms [24, 12]. These methods indicate that robustness in BLS can be improved, but they also suggest a gap: residual robustness and sample credibility are often treated as separate design choices rather than as complementary components of the same training objective.

This paper develops a dual-robust BLS model from that perspective. The proposed method, named IFW-BLS for Intuitionistic Fuzzy Wave Broad Learning System, uses an intuitionistic fuzzy score to quantify the credibility of each training sample and applies the wave loss [5] to the resulting score-weighted residuals. The wave loss is bounded, smooth, and asymmetric: the bound limits extreme deviations, while the asymmetry parameter shifts the relative penalty assigned to positive and negative residuals. This flexibility lets the model adapt when overestimation-like and underestimation-like deviations are not equally harmful for a dataset. The intuitionistic fuzzy score, on the other hand, uses both global class-center information and local class-conflict information to reduce the influence of ambiguous or unreliable samples. Together, these mechanisms produce robustness from two views: the credibility view decides how strongly a sample should influence learning, and the residual view decides how strongly an error should be penalized.

The main contributions are summarized as follows.

1. We propose IFW-BLS, an Intuitionistic Fuzzy Wave Broad Learning System that improves BLS robustness from two complementary directions: samplelevel credibility modeling through intuitionistic fuzzy scores and residuallevel protection through the wave loss.

2. We develop a credibility-weighted wave-loss objective in which unreliable samples are softened before the robust loss is evaluated, allowing the model to reduce the influence of ambiguous samples, noisy observations, and outliers within a unified formulation.

3. We derive the gradient of the proposed objective and optimize IFW-BLS using a Nesterov accelerated gradient based strategy, thereby avoiding the explicit matrix inversion used in conventional BLS training.

4. We evaluate IFW-BLS on 30 UCI benchmark datasets using accuracy, average rank, and Friedman and Nemenyi statistical tests, and further examine its behavior under controlled noise and outlier contamination to verify its robustness.

The remainder of this paper is organized as follows. Section 2 reviews the preliminaries of BLS, wave loss, and intuitionistic fuzzy credibility. Section 3 presents the proposed IFW-BLS model, including its objective function, gradient derivation, optimization procedure, and computational complexity. Section 4 reports the experimental results. Finally, Section 5 concludes the paper.

## 2 Preliminaries

## 2.1 Notation

Let $\boldsymbol { X } = [ x _ { 1 } , \ldots , x _ { m } ] ^ { \top } \in \mathbb { R } ^ { m \times d }$ denote a training set with m samples and d input features. The target matrix is $Y \in \mathbb { R } ^ { m \times c }$ , where c is the number of classes and each row is a one-hot encoded label vector. For matrices, $\| \cdot \| _ { F }$ denotes the Frobenius norm and $( \cdot ) ^ { \top }$ denotes transpose. The number of feature-node windows is q, the number of nodes in each feature window is $p ,$ the number of enhancement-node windows is s, and the number of nodes in each enhancement window is r.

## 2.2 Broad Learning System (BLS) [9]

BLS constructs a broad representation in two stages. Its overall information flow is shown in Fig. 1. the input is first mapped into multiple randomized feature windows, then transformed by enhancement windows, and finally concatenated

![](images/5d75f87058bc035362f11d7e2e67385d633095cb2bb740ba920b2bbcb1e38a67.jpg)  
Fig. 1. Standard BLS architecture with randomized feature and enhancement windows.

into a broad representation used to learn the output weights. Given $X$ , the $k ^ { \mathrm { t h } }$ feature window is generated as

$$
Z _ { k } = \phi ( X P _ { k } + \mathbf { 1 } b _ { k } ^ { \top } ) , \qquad k = 1 , \ldots , q ,\tag{1}
$$

where $P _ { k } \in \mathbb { R } ^ { d \times p }$ and $b _ { k } \in \mathbb { R } ^ { p }$ are randomly generated parameters, 1 is an all-one vector of length m, and $\phi ( \cdot )$ is an activation function. The feature-node output is

$$
Z = [ Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { q } ] \in \mathbb { R } ^ { m \times p q } .\tag{2}
$$

The enhancement layer further maps $Z$ into s enhancement windows:

$$
H _ { \ell } = \psi ( Z Q _ { \ell } + { \bf 1 } d _ { \ell } ^ { \top } ) , \qquad \ell = 1 , \dots , s ,\tag{3}
$$

where $Q _ { \ell } \in \mathbb { R } ^ { p q \times r } , d _ { \ell } \in \mathbb { R } ^ { r }$ , and $\psi ( \cdot )$ is an activation function. Concatenating all enhancement windows gives

$$
H = [ H _ { 1 } , H _ { 2 } , \dots , H _ { s } ] \in \mathbb { R } ^ { m \times r s } .\tag{4}
$$

The final broad representation is

$$
A = [ Z , H ] \in \mathbb { R } ^ { m \times n _ { b } } , \qquad n _ { b } = p q + r s .\tag{5}
$$

The conventional BLS output matrix $W \in \mathbb { R } ^ { n _ { b } \times c }$ is estimated by

$$
\operatorname* { m i n } _ { W } \frac { 1 } { 2 } \| W \| _ { F } ^ { 2 } + \frac { C } { 2 } \| A W - Y \| _ { F } ^ { 2 } ,\tag{6}
$$

where $C > 0$ is the regularization parameter. Problem (6) has a closed-form solution involving an inverse of either $A ^ { \top } A { + } C ^ { - 1 } I$ or $A A ^ { \top } { + } C ^ { - 1 } I$ . This is eficient for moderate sizes, but it becomes expensive for very broad representations or large training sets and remains sensitive to extreme residuals.

![](images/6de5841b044e6fb506aefc3c7066c476c616bfe2a1e133b1433e6f0674319bb5.jpg)  
(a) $a = 0$

![](images/440de6b1e68635adfff26e1b81b84178d07ab72b06fb8b58ab608406db4ea0b3.jpg)  
(b) $a > 0$

![](images/7f5e405104e2046e7f33aeeaf9ce0bef9149f727f67b1ee15c08e465e1c2281c.jpg)  
(c) $a < 0$  
Fig. 2. Wave-loss curves under fixed bound parameter $\eta \ : = \ : 1 \colon$ symmetric, positiveasymmetric, and negative-asymmetric cases.

## 2.3 Wave Loss [5]

For a residual $u \in \mathbb { R }$ , the wave loss is defined as

$$
L _ { \mathrm { w } } ( u ) = \frac { u ^ { 2 } \exp ( a u ) } { 1 + \lambda u ^ { 2 } \exp ( a u ) } ,\tag{7}
$$

where $a \in \mathbb { R }$ controls the asymmetry and $\lambda > 0$ controls the upper bound. Since $u ^ { 2 } \exp ( a u ) \geq 0$ , the loss satisfies $0 \leq L _ { \mathrm { w } } ( u ) < 1 / \lambda$ . Thus, very large residuals cannot grow without limit in the objective. The sign and magnitude of a also move the emphasis of the loss: $a = 0$ gives a symmetric penalty, whereas $a > 0$ and $a \ < \ 0$ assign diferent penalties to positive and negative residuals. The derivative needed for optimization is

$$
L _ { \mathrm { w } } ^ { \prime } ( u ) = \frac { u ( 2 + a u ) \exp ( a u ) } { \left( 1 + \lambda u ^ { 2 } \exp ( a u ) \right) ^ { 2 } } .\tag{8}
$$

This smooth derivative allows the loss to be used in gradient-based BLS training. Fig. 2 illustrates how $a = 0$ gives a symmetric curve, while positive and negative values of a shift the dominant penalization direction.

## 2.4 Intuitionistic Fuzzy Credibility

Intuitionistic fuzzy theory represents uncertainty through membership, nonmembership, and hesitation degrees [22, 7]. Following the kernelized credibility modeling used in intuitionistic fuzzy classifiers [19], each sample receives a score that reflects two reliability cues: how close it is to the center of its own class and how much local class conflict surrounds it.

Let $\varphi ( \cdot )$ denote the feature map induced by an RBF kernel

$$
K ( x _ { i } , x _ { j } ) = \exp ( - \gamma _ { k } \| x _ { i } - x _ { j } \| _ { 2 } ^ { 2 } ) ,\tag{9}
$$

where $\gamma _ { k } > 0$ . Distances in the induced feature space are computed by

$$
\| \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \| _ { 2 } = { \sqrt { K ( x _ { i } , x _ { i } ) + K ( x _ { j } , x _ { j } ) - 2 K ( x _ { i } , x _ { j } ) } } .\tag{10}
$$

For class $g \in \{ 1 , \ldots , c \}$ , define its feature-space center and radius as

$$
D _ { g } = \frac { 1 } { m _ { g } } \sum _ { y _ { i } = g } \varphi ( x _ { i } ) , \qquad R _ { g } = \operatorname* { m a x } _ { y _ { i } = g } \| \varphi ( x _ { i } ) - D _ { g } \| _ { 2 } ,\tag{11}
$$

where $m _ { g }$ is the number of samples in class $g .$ For sample $x _ { i }$ with label $y _ { i }$ , the membership degree is

$$
\mu _ { i } = \operatorname* { m a x } \left\{ 0 , 1 - \frac { \| \varphi ( x _ { i } ) - D _ { y _ { i } } \| _ { 2 } } { R _ { y _ { i } } + \varepsilon } \right\} ,\tag{12}
$$

where $\varepsilon > 0$ avoids division by zero. The required class-center distance can be computed only through kernel evaluations as

$$
\displaystyle | | \varphi ( x _ { i } ) - D _ { g } | | _ { 2 } ^ { 2 } = K ( x _ { i } , x _ { i } ) - \frac { 2 } { m _ { g } } \sum _ { y _ { k } = g } K ( x _ { i } , x _ { k } ) + \frac { 1 } { m _ { g } ^ { 2 } } \sum _ { y _ { k } = g } \sum _ { y _ { \ell } = g } K ( x _ { k } , x _ { \ell } ) .\tag{13}
$$

Thus, the intuitionistic fuzzy scores do not require explicit construction of $\varphi ( \cdot )$ The local conflict ratio is

$$
\bar { \psi } _ { i } = \frac { | \{ x _ { j } \in \mathcal { N } _ { \tau } ( i ) : y _ { j } \neq y _ { i } \} | } { \operatorname* { m a x } \{ 1 , | \mathcal { N } _ { \tau } ( i ) | \} } ,\tag{14}
$$

where $\mathcal { N } _ { \tau } ( i ) = \{ x _ { j } : \| \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \| _ { 2 } \leq \tau , \ j \neq i \}$ and $\tau > 0$ is a neighborhood radius. The nonmembership degree is then

$$
\nu _ { i } = ( 1 - \mu _ { i } ) \varPsi _ { i } .\tag{15}
$$

This construction gives high nonmembership only when a sample is globally weak for its own class and locally surrounded by conflicting labels. The remaining hesitation is $\pi _ { i } = 1 - \mu _ { i } - \nu _ { i }$ . Finally, the credibility score is

$$
s _ { i } = \left\{ \begin{array} { l l } { \mu _ { i } , } & { \nu _ { i } = 0 , } \\ { 0 , } & { \mu _ { i } \leq \nu _ { i } , } \\ { \displaystyle \frac { 1 - \nu _ { i } } { 2 - \mu _ { i } - \nu _ { i } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{16}
$$

with $s _ { i } \in [ 0 , 1 ]$ . Larger $s _ { i }$ means that the sample is more reliable for training;   
smaller $s _ { i }$ reduces its influence.

## 3 Proposed Dual-Robust BLS

## 3.1 Motivation

A single robustness mechanism is often insuficient. If only a bounded loss is used, a highly suspicious sample may still influence the model until its residual becomes large enough to be saturated. If only a credibility score is used, moderate but systematic residuals may still be penalized by a squared loss. IFW-BLS therefore combines the two mechanisms directly: sample credibility scales the residual first, and the wave loss then evaluates the scaled residual with a bounded and direction-sensitive penalty.

## 3.2 Credibility-Weighted Wave Loss

Let $e _ { i j } = ( A W ) _ { i j } - Y _ { i j }$ be the residual of sample i for output class $j .$ . The proposed credibility-weighted residual is

$$
\xi _ { i j } = s _ { i } e _ { i j } = s _ { i } \left( ( A W ) _ { i j } - Y _ { i j } \right) .\tag{17}
$$

The proposed IFW-BLS objective is

$$
\operatorname* { m i n } _ { W } J ( W ) = \frac { 1 } { 2 } \| W \| _ { F } ^ { 2 } + C \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { c } L _ { \mathbf { w } } \big ( \xi _ { i j } \big ) .\tag{18}
$$

The role of $s _ { i }$ in (18) is not merely a post-hoc sample weight. It changes the argument of the robust loss itself. Consequently, low-credibility samples are softened before the wave loss is evaluated, and samples with $s _ { i } = 0$ contribute no loss or gradient. At the same time, the wave loss caps the contribution of large weighted residuals and uses a to control whether positive or negative deviations should receive stronger penalization.

For $\lambda > 0 .$ , the wave loss satisfies

$$
0 \leq L _ { \mathrm { w } } ( u ) = \frac { u ^ { 2 } \exp ( a u ) } { 1 + \lambda u ^ { 2 } \exp ( a u ) } < \frac { 1 } { \lambda } .\tag{19}
$$

With the weighted residual $\xi _ { i j } = s _ { i } ( ( A W ) _ { i j } - Y _ { i j } )$ and $s _ { i } \in [ 0 , 1 ]$ , each loss term is also bounded:

$$
0 \leq L _ { \mathrm { w } } ( \xi _ { i j } ) < \frac { 1 } { \lambda } .\tag{20}
$$

Moreover, if $s _ { i } = 0 ,$ , then $\xi _ { i j } = 0$ and the sample contributes neither loss nor gradient. For $0 < s _ { i } < 1$ , the residual entering the loss is contracted, and the gradient contribution receives an additional multiplicative factor $s _ { i }$ from the chain rule. The parameter a controls how positive and negative residuals are penalized relative to each other, so IFW-BLS can adapt the direction of stronger penalization.

## 3.3 Gradient Derivation

Using (8) and the chain rule, the contribution of sample i to the gradient for output class $j$ is

$$
\varDelta _ { i j } = s _ { i } \frac { \xi _ { i j } ( 2 + a \xi _ { i j } ) \exp ( a \xi _ { i j } ) } { \left( 1 + \lambda \xi _ { i j } ^ { 2 } \exp ( a \xi _ { i j } ) \right) ^ { 2 } } .\tag{21}
$$

Therefore, for the $j ^ { \mathrm { t h } }$ column $W _ { : j }$ of $W$ ,

$$
\nabla _ { W _ { : j } } J ( W ) = W _ { : j } + C \sum _ { i = 1 } ^ { m } \varDelta _ { i j } A _ { i : } ^ { \top } .\tag{22}
$$

Equivalently, if $\varDelta \in \mathbb { R } ^ { m \times c }$ collects all $\varDelta _ { i j }$ terms, the full gradient can be written compactly as

$$
\nabla J ( W ) = W + C A ^ { \top } \varDelta .\tag{23}
$$

The derivation shows the two damping efects explicitly: $s _ { i }$ appears in the residual $\xi _ { i j }$ and again through the derivative $\partial \xi _ { i j } / \partial W _ { : j } = s _ { i } A _ { i : } ^ { \top }$

## 3.4 Optimization

The proposed objective is smooth for fixed IF scores and fixed randomized BLS features. We optimize (18) using Nesterov accelerated gradient (NAG)-based algorithm as used in [2, 6], which avoids the matrix inverse in the conventional BLS solution and is suitable for broad representations. Let $V ^ { ( t ) }$ denote the velocity matrix at iteration $t ,$ and let $\eta _ { 0 } , \rho ,$ and $\gamma$ denote the initial learning rate, learning-rate decay, and momentum coeficient, respectively. The optimization proceeds as follows.

1. Initialization. Construct the BLS representation $A ,$ compute the credibility scores $\{ s _ { i } \} _ { i = 1 } ^ { m }$ , initialize the output weights $W ^ { ( 0 ) }$ , set $V ^ { ( 0 ) } = \mathbf { 0 }$ , and choose $C , a , \lambda , \eta _ { 0 } , \rho , \gamma$ , tolerance $\delta ,$ and maximum iteration number $T$

2. Look-ahead point. At iteration t, NAG first evaluates the objective at the momentum-aided point

$$
\widehat { W } ^ { ( t ) } = W ^ { ( t ) } + \gamma V ^ { ( t ) } .\tag{24}
$$

This look-ahead step uses the current search direction before computing the gradient, which usually gives more stable updates than plain gradient descent.

3. Credibility-weighted residuals. Using $\widehat { W } ^ { ( t ) }$ , compute

$$
\xi _ { i j } ^ { ( t ) } = s _ { i } \left( ( A \widehat { W } ^ { ( t ) } ) _ { i j } - Y _ { i j } \right) , \qquad i = 1 , \ldots , m , \ j = 1 , \ldots , c .\tag{25}
$$

The IF score enters before the wave loss is evaluated, so samples with weak credibility have reduced residual contributions.

4. Gradient evaluation. For all samples and classes, compute

$$
\varDelta _ { i j } ^ { ( t ) } = s _ { i } \frac { \xi _ { i j } ^ { ( t ) } ( 2 + a \xi _ { i j } ^ { ( t ) } ) \exp ( a \xi _ { i j } ^ { ( t ) } ) } { \left( 1 + \lambda ( \xi _ { i j } ^ { ( t ) } ) ^ { 2 } \exp ( a \xi _ { i j } ^ { ( t ) } ) \right) ^ { 2 } } ,\tag{26}
$$

and form the full gradient

$$
G ^ { ( t ) } = \widehat { W } ^ { ( t ) } + C A ^ { \top } \varDelta ^ { ( t ) } .\tag{27}
$$

5. Parameter update. The learning rate is decayed as

$$
\eta _ { t } = \eta _ { 0 } \exp ( - \rho t ) .\tag{28}
$$

The velocity and output weights are then updated by

$$
V ^ { ( t + 1 ) } = \gamma V ^ { ( t ) } - \eta _ { t } G ^ { ( t ) } , \qquad W ^ { ( t + 1 ) } = W ^ { ( t ) } + V ^ { ( t + 1 ) } .\tag{29}
$$

The iterations stop when $t \geq T$ or $\| W ^ { ( t + 1 ) } - W ^ { ( t ) } \| _ { F } \leq \delta .$

For a test sample x˜, its BLS representation a˜ is generated using the same fixed feature and enhancement mappings as the training data. The predicted class is

$$
\hat { y } = \arg \operatorname* { m a x } _ { j \in \{ 1 , . . . , c \} } ( \tilde { a } W ) _ { j } .\tag{30}
$$

## 3.5 Computational Complexity

Let $n _ { b } = p q + r s$ be the width of the final BLS representation. Constructing the randomized feature windows requires $\mathcal { O } ( m d p q )$ operations, and constructing the enhancement windows requires $\mathcal { O } ( m p q r s )$ operations when each enhancement window is generated from the full feature-node matrix. The intuitionistic fuzzy scores are computed once before optimization. With an RBF kernel, a direct implementation forms the $m \times m$ kernel matrix in $\mathcal { O } ( m ^ { 2 } d )$ time and then computes class-center distances and neighborhood conflicts from this matrix. After $\underset { - } { A }$ and $\{ s _ { i } \} _ { i = 1 } ^ { m }$ are fixed, each NAG iteration is dominated by the products $A \widehat { W }$ and $A ^ { \top } \varDelta .$ , giving a cost of $\mathcal { O } ( m n _ { b } c )$ . The element-wise evaluation of the wave-loss derivative costs $\mathcal { O } ( m c )$ , and the velocity and weight updates cost $\mathcal { O } ( n _ { b } c )$ . Thus, the total optimization cost over T iterations is $\mathcal { O } ( T m n _ { b } c )$ . The overall training cost of IFW-BLS is therefore $\mathcal { O } ( m d p q + m p q r s + m ^ { 2 } d + T m n _ { b } c )$ , excluding lower-order terms. Conventional BLS avoids iterative optimization but requires a matrix inverse with cost $\mathcal { O } ( \operatorname* { m i n } \{ n _ { b } ^ { 3 } , m ^ { 3 } \} )$ ) after the broad representation is constructed. IFW-BLS replaces this inversion by first-order updates while adding a one-time credibility computation, which is appropriate when robustness to sample ambiguity and residual outliers is required.

## 4 Experiments and Discussion

In this section, we evaluate the proposed IFW-BLS model on 30 UCI benchmark datasets against RVFL [17], RVFL without direct link (RVFLwoDL/ELM) [14], BLS [9], Wave-RVFL [21], NF-BLS [13], F-BLS [20], IF-BLS [20], and $\mathrm { K R P } .$ BLS [12]. The methods are compared using classification accuracy, average rank, and Friedman and Nemenyi statistical tests. In addition, IFW-BLS is evaluated under controlled noise and outlier contamination to examine its robustness on corrupted training data.

## 4.1 Experimental Setup and Hyperparameter Selection

All experiments are implemented in MATLAB R2023a on a Windows 11 workstation with an 11th-generation Intel Core i7-11700 CPU at 2.50 GHz and 16 GB RAM. For a fair comparison, all methods are evaluated using 5-fold crossvalidation together with grid-based hyperparameter selection. In each run, the dataset is divided into five disjoint folds; four folds are used for training and the remaining fold is used for testing. The final performance of a model is reported as the best mean testing accuracy obtained over the considered hyperparameter grid.

The regularization parameter C is searched over $\{ 1 0 ^ { - 6 } , 1 0 ^ { - 4 } , \dots , 1 0 ^ { 6 } \}$ for all methods. For RVFL and RVFLwoDL, the number of hidden nodes is selected from 5:10:205. For BLS-based models, including the proposed IFW-BLS, the number of feature-node windows is chosen from 1:2:21, the number of feature nodes in each window from 5:5:50, and the number of enhancement nodes from

5:10:105. For NF-BLS, the number of fuzzy groups is tuned over 1:2:21, the number of fuzzy nodes per group over 5:5:50, and the number of enhancement nodes over 5:10:105. The hyperparameter settings of Wave-RVFL follow [21]; those of F-BLS and IF-BLS follow [20]; and KRP-BLS is configured according to its original implementation [12]. For the proposed IFW-BLS, the wave-loss asymmetry parameter is selected from $a \in \{ - 3 , - 4 , \dots , 3 \}$ and the bound parameter from $\lambda \in \{ 0 . 1 , 0 . 5 , 1 \}$ . The RBF kernel width γ<sub>k</sub> used in the intuitionistic fuzzy score is selected from $\{ 1 0 ^ { - 6 } , 1 0 ^ { - 4 } , \dots , 1 0 ^ { 6 } \}$ . The NAG parameters are fixed as $W ^ { ( 0 ) } = 0 . 0 1 { \bf 1 } , V ^ { ( 0 ) } = \bar { \bf 0 } .$ , initial learning rate $\eta _ { 0 } = 0 . 0 1$ , decay $\rho = 0 . 1$ , momentum $\gamma = 0 . 6$ , tolerance $\delta = 1 0 ^ { - 6 }$ , and maximum iterations $T = 1 0 0$

## 4.2 Performance Evaluation

Table 1 reports the average accuracy and average rank of all compared models over the 30 UCI datasets. In terms of average accuracy, the proposed IFW-BLS obtains the best result of 87.4896%. The closest competitor is IF-BLS with 83.9102%, followed by KRP-BLS with 83.002%, standard BLS with 82.0751%, F-BLS with 82.0376%, Wave-RVFL with 81.5155%, NF-BLS with 80.0399%, RVFL with 79.8876%, and RVFLwoDL with 79.4376%. Thus, IFW-BLS improves the average accuracy by 3.5794% over IF-BLS, 4.4876% over KRP-BLS, 5.4145% over standard BLS, 5.452% over F-BLS, 5.9741% over Wave-RVFL, 7.4497% over NF-BLS, 7.602% over RVFL, and 8.052% over RVFLwoDL. These accuracy results support the dual-robustness motivation of IFW-BLS. The strong performance of IF-BLS confirms that intuitionistic fuzzy sample credibility is useful for reducing the influence of ambiguous or unreliable samples. However, IFW-BLS further improves over IF-BLS by applying the bounded and asymmetric wave loss to the credibility-weighted residuals. This indicates that samplelevel reliability alone is not suficient; after low-credibility samples are softened, residual-level protection is still beneficial for controlling large deviations and adapting the penalty to the dominant error direction. The improvement over Wave-RVFL also shows that wave-loss robustness becomes more efective when it is embedded in the BLS feature-enhancement representation and combined with intuitionistic fuzzy credibility.

Although average accuracy provides a direct summary of predictive performance, it may be afected by large gains on a small number of datasets. Therefore, average rank is also reported to assess the consistency of each method across datasets. A lower rank indicates that a model more frequently appears among the top-performing methods. IFW-BLS achieves the best average rank of 2.3333, followed by IF-BLS with 3.55, Wave-RVFL with 4.1333, KRP-BLS with 4.8667, standard BLS with 4.9333, F-BLS with 5.3333, NF-BLS with 6.1, RVFL with 6.5333, and RVFLwoDL with 7.2167. Compared with IFW-BLS, the rank gaps are 1.2167 for IF-BLS, 1.8 for Wave-RVFL, 2.5334 for KRP-BLS, 2.6 for standard BLS, 3 for F-BLS, 3.7667 for NF-BLS, 4.2 for RVFL, and 4.8834 for RVFLwoDL. The rank analysis further confirms that the superiority of IFW-BLS is not due to isolated improvements on only a few datasets. IFW-BLS obtains the best or tied-best result on 21 out of 30 datasets and the second-best result on 3 additional datasets. It also wins or ties against IF-BLS on 24 datasets and against standard BLS on 27 datasets. These observations show that the proposed combination of intuitionistic fuzzy credibility and wave-loss residual robustness provides stable gains across heterogeneous classification tasks.

Table 1. Performance comparison on 30 UCI datasets. Best and second-best values are shown in bold and underlined, respectively. † denotes the proposed model.
<table><tr><td>Dataset</td><td>RVFL [17] RVFLwoDL [14] BLS [9]</td><td></td><td></td><td>Wave-RVFL [21] NF-BLS [13] F-BLS [20] IF-BLS [20] KRP-BLS [12] IFW-BLS†</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>acute_inflammation</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>acute_nephritis</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100 89.759</td><td>100</td><td>100 87.723</td><td>100</td></tr><tr><td>bank</td><td>90.9051</td><td>89.4051</td><td>89.7366</td><td>88.1779 80.7578</td><td>89.5817</td><td>78.5807</td><td>89.4051 77.2451</td><td>76.7</td><td>90.561 83.5711</td></tr><tr><td>blood</td><td>78.4056</td><td>76.9056</td><td>79.6272</td><td></td><td>76.0807</td><td></td><td></td><td></td><td></td></tr><tr><td>breast_cancer</td><td>68.1727</td><td>66.6727</td><td>69.8851</td><td>70.2179</td><td>70.1754</td><td>72.2868</td><td>83.1579</td><td>82.4947</td><td>91.0637</td></tr><tr><td>breast_cancer_wisc</td><td>89.4897</td><td>87.9897</td><td>88.4183</td><td>86.805</td><td>90.7081</td><td>88.2775</td><td>88.9938</td><td>89.324</td><td>98.4588</td></tr><tr><td>chess krvkp</td><td>73.5312</td><td>72.0312</td><td>84.3862</td><td>75.7371</td><td>70.4004</td><td>84.2921 69.1521</td><td>84.9184</td><td>87.3718</td><td>89.9404</td></tr><tr><td>conn_bench_sonar_mines_rocks</td><td>62.0807</td><td>60.5807</td><td>69.2451</td><td>63.9431</td><td>60.6272</td><td>86.3768</td><td>80.2091</td><td>78.2049</td><td>94.5878</td></tr><tr><td>credit_approval</td><td>86.8623</td><td>85.3623</td><td>87.5362</td><td>88.5995</td><td>84.4928</td><td>69.1338</td><td>88.5507</td><td>89.7797</td><td>88.7882</td></tr><tr><td>cylinder bands echocardiogram</td><td>67.7193</td><td>66.2193</td><td>69.9258</td><td>69.7509</td><td>69.5336</td><td></td><td>72.8536</td><td>70.8109</td><td>70.0391</td></tr><tr><td></td><td>85.4031</td><td>83.9031 90</td><td>83.9316</td><td>87.9652</td><td>80.9402</td><td>84.6724 91</td><td>88.49 91</td><td>86.7202</td><td>87.0762</td></tr><tr><td>fertility haberman_survival</td><td>89.5 74.9902</td><td>73.4902</td><td>90 70.275</td><td>86.815 77.2399</td><td>92 73.4902</td><td>69.6563</td><td>75.4574</td><td>89.18 76.1937</td><td>92.4064</td></tr><tr><td>hepatitis</td><td></td><td>83.2258</td><td>85.1613</td><td>85.7226</td><td>87.7419</td><td>84.5161</td><td>87.7419</td><td>85.1096</td><td>79.2877</td></tr><tr><td>hill_valley</td><td>83.2258</td><td>77.9706</td><td>82.0185</td><td>80.3097</td><td>78.9583</td><td>81.6002</td><td>79.6249</td><td>78.6344</td><td>86.4488</td></tr><tr><td></td><td>77.9706</td><td>84.7098</td><td>86.1422</td><td>86.7361</td><td>83.9726</td><td>86.1385</td><td>86.6938</td><td></td><td>80.2332</td></tr><tr><td>horse colic mammographic</td><td>84.2098 79.0889</td><td>79.0889</td><td>78.6712</td><td></td><td>79.504</td><td>78.6744</td><td>79.8192</td><td>84.092 79.8238</td><td>87.1719</td></tr><tr><td></td><td>68.961</td><td>68.961</td><td>83.9394</td><td>81.4616 71.0298</td><td>75.1515</td><td>84.9351</td><td>88.7879</td><td>89.1241</td><td>87.0013</td></tr><tr><td>molec_biol_promoter monks 1</td><td>83.2497</td><td>83.2497</td><td>75.3314</td><td>85.7472</td><td>86.1277</td><td>77.1396</td><td>77.6705</td><td></td><td>96.7207</td></tr><tr><td>musk 1</td><td>67.864</td><td>67.864</td><td>75.8311</td><td>69.8999</td><td>67.8487</td><td>76.6754</td><td>78.9846</td><td>74.0404 79.8523</td><td>75.5649 97.3361</td></tr><tr><td>oocytes_merluccius_nucleus_4d</td><td>79.8407</td><td>79.8407</td><td>82.7776</td><td>82.2359</td><td>80.142</td><td>81.8011</td><td>80.9201</td><td>79.3017</td><td></td></tr><tr><td>oocytes_trisopterus_nucleus 2f</td><td>76.4211</td><td>76.4211</td><td>78.9371</td><td>78.7137</td><td>75.9911</td><td>78.6123</td><td>75.9899</td><td>74.2302</td><td>75.4261 75.0727</td></tr><tr><td></td><td>71.4888</td><td>71.4888</td><td>72.0958</td><td>73.6335</td><td>72.7918</td><td>71.4846</td><td>73.3079</td><td></td><td></td></tr><tr><td>pima pittsburg  $\mathrm { \Delta _ { b r i d g e s } \mathrm { ~ \_ ~ T ~ \_ ~ O R ~ \_ ~ D ~ } ~ }$ </td><td>87.6429</td><td>88.1429</td><td>89.1095</td><td>90.2722</td><td>90.1429</td><td>88.1905</td><td>90.2381</td><td>74.1087 87.0791</td><td>78.1658</td></tr><tr><td>spambase</td><td>87.1125</td><td>87.1125</td><td>89.0201</td><td>89.7259</td><td>83.2657</td><td>89.4582</td><td>90.7407</td><td>87.4741</td><td>93.3928 94.0153</td></tr><tr><td>spect</td><td>67.9245</td><td>67.9245</td><td>69.434</td><td>69.9622</td><td>67.9245</td><td>69.434</td><td>72.4528</td><td>72.3517</td><td>72.9247</td></tr><tr><td>statlog_heart</td><td>80</td><td>80</td><td>82.2222</td><td>82.4</td><td>80.7407</td><td>82.2222</td><td>84.0741</td><td>85.0566</td><td>88.244</td></tr><tr><td>tic tac toe</td><td>86.0068</td><td>86.0068</td><td>98.4315</td><td>88.587</td><td>82.7623</td><td>97.8081</td><td>97.0768</td><td>94.1645</td><td>99.5474</td></tr><tr><td>titanic</td><td>77.9168</td><td>77.9168</td><td>77.9168</td><td>80.2543</td><td>78.0537</td><td>77.9168</td><td>79.9532</td><td>78.3541</td><td>95.2742</td></tr><tr><td>vertebral column 2clases</td><td>70.6452</td><td>70.6452</td><td>72.2453</td><td>72.7646</td><td>72.0465</td><td>71.3333</td><td>72.9479</td><td>72.759</td><td>76.3667</td></tr><tr><td>Average accuracy</td><td>79.8876</td><td>79.4376</td><td>82.0751</td><td>81.5155</td><td>80.0399</td><td>82.0376</td><td>83.9102</td><td>83.002</td><td>87.4896</td></tr><tr><td>Average rank</td><td>6.5333</td><td>7.2167</td><td>4.9333</td><td>4.1333</td><td>6.1</td><td>5.3333</td><td>3.55</td><td>4.8667</td><td>2.3333</td></tr></table>

## 4.3 Statistical Significance

Following the statistical comparison protocol recommended by Demsar [11], we use the nonparametric Friedman test to examine whether the diferences in average ranks are statistically meaningful across all datasets. Let K be the number of compared models and N be the number of datasets. The null hypothesis states that all K models have equivalent predictive performance, so their rank diferences over $\mathcal { N }$ datasets are due only to random variation. The Friedman statistic is computed as

$$
\chi _ { F } ^ { 2 } = \frac { 1 2 \mathcal { N } } { \mathcal { K } ( \mathcal { K } + 1 ) } \left[ \sum _ { k = 1 } ^ { \kappa } \varrho ( k , \bullet ) ^ { 2 } - \frac { \mathcal { K } ( \mathcal { K } + 1 ) ^ { 2 } } { 4 } \right] ,\tag{31}
$$

where $\varrho ( k , \bullet )$ denotes the average rank of the $k ^ { \mathrm { t h } }$ model. Since the Friedman statistic can be conservative, we also report the Iman-Davenport corrected statistic

$$
F _ { F } = \chi _ { F } ^ { 2 } \left( \frac { \rlap / N - 1 } { \rlap / N ( \rlap / K - 1 ) - \chi _ { F } ^ { 2 } } \right) ,\tag{32}
$$

which follows an F distribution with $( \kappa - 1 )$ and $( \boldsymbol { K } - 1 ) ( \boldsymbol { \mathcal { N } } - 1 )$ degrees of freedom. In this study, $\kappa = 9$ methods are compared on $\mathcal { N } = 3 0$ datasets. As shown in Table 2, the Friedman statistic is $\chi _ { F } ^ { 2 } = 7 4 . 2 9 1 1$ , and the corrected statistic is $F _ { F } = 1 3 . 0 0 1 4$ . The critical value at the 5% significance level is $F ( 8 , 2 3 2 ) = 1 . 9 7 8 4 .$ . Since $1 3 . 0 0 1 4 > 1 . 9 7 8 4$ , the null hypothesis is rejected, indicating that the observed rank diferences among the competing models are statistically significant.

Table 2. Statistical comparison based on Friedman and Nemenyi tests.
<table><tr><td colspan="4">Friedman test with Iman-Davenport correction</td></tr><tr><td>K</td><td>N</td><td> $\chi _ { F } ^ { 2 } ~ / ~ F _ { F }$ </td><td>Critical value at  $\alpha = 0 . 0 5$   $F ( 8 , 2 3 2 ) = 1 . 9 7 8 4$ </td></tr><tr><td>9</td><td>30</td><td>74.2911 / 13.0014</td><td> $\mathrm { C . D . = 2 . 0 1 8 8 }$ </td></tr><tr><td colspan="4">Nemenyi post-hoc test against IFW-BLS,</td></tr><tr><td>Model</td><td></td><td>Average rank Rank difference</td><td>Significant</td></tr><tr><td>RVFL [17]</td><td>6.5333</td><td>4.2</td><td>Yes</td></tr><tr><td>RVFLwoDL [14]</td><td>7.2167</td><td>4.8833</td><td>Yes</td></tr><tr><td>BLS [9]</td><td>4.9333</td><td>2.6</td><td>Yes</td></tr><tr><td>Wave-RVFL [21]</td><td>4.1333</td><td>1.8</td><td>No</td></tr><tr><td>NF-BLS [13]</td><td>6.1</td><td>3.7667</td><td>Yes</td></tr><tr><td>F-BLS [20]</td><td>5.3333</td><td>3</td><td>Yes</td></tr><tr><td>IF-BLS [20]</td><td>3.55</td><td>1.2167</td><td>No</td></tr><tr><td>KRP-BLS [12]</td><td>4.8667</td><td>2.5333</td><td>Yes</td></tr></table>

After rejecting the null hypothesis, we apply the Nemenyi post-hoc test to compare IFW-BLS with each competing method. Two models are considered significantly diferent when the absolute diference between their average ranks is larger than the critical diference

$$
\mathrm { C . D . } = q _ { \alpha } \sqrt { \frac { \mathcal { K } ( \mathcal { K } + 1 ) } { 6 \mathcal { N } } } ,\tag{33}
$$

where $q _ { \alpha }$ is the critical value for the two-tailed Nemenyi test. At $\alpha = 0 . 1 0$ , the critical diference is $\mathrm { C . D . } = 2 . 0 1 8 8$ for $\kappa = 9$ and $\mathcal { N } = 3 0$

The Nemenyi test shows that IFW-BLS is significantly better than RVFL, RVFLwoDL, BLS, NF-BLS, F-BLS, and KRP-BLS. The diferences with Wave-RVFL and IF-BLS do not exceed the critical diference, which is reasonable since these are the two baselines most closely related to the proposed design: Wave-RVFL uses wave-loss-based residual robustness, whereas IF-BLS uses intuitionistic fuzzy credibility. IFW-BLS still obtains the best average rank and average accuracy among all methods, indicating that combining these two robustness views inside BLS gives the strongest overall performance while remaining statistically competitive with the closest robust alternatives.

## 4.4 Robustness Analysis under Noise and Outliers

To directly examine the dual-robustness claim under adverse training conditions, we conduct controlled corruption experiments on two representative datasets, blood and horse\_colic. The training folds are contaminated while the test folds remain clean. Two perturbation types are considered: feature outliers and label noise. For each type, contamination levels of 5%, 10%, 15%, and 20% are evaluated. Table 3 compares IFW-BLS with standard BLS under these settings.

Table 3. Robustness comparison of BLS and IFW-BLS under outlier and noise contamination. Bold values denote the best performance for each row.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Level</td><td colspan="2">Outliers</td><td rowspan="2">Level</td><td colspan="2">Noise</td></tr><tr><td>BLS</td><td>IFW-BLS†</td><td>BLS</td><td>IFW-BLS†</td></tr><tr><td rowspan="6">blood</td><td>Clean</td><td>79.6272</td><td>83.5711</td><td>Clean</td><td>79.6272</td><td>83.5711</td></tr><tr><td>5%</td><td>75.8421</td><td>82.237</td><td>5%</td><td>77.1034</td><td>82.6225</td></tr><tr><td>10%</td><td>72.5186</td><td>79.8725</td><td>10%</td><td>74.8652</td><td>81.3425</td></tr><tr><td>15%</td><td>69.2847</td><td>77.8106</td><td>15%</td><td>71.4928</td><td>79.2435</td></tr><tr><td>20%</td><td>66.0319</td><td>76.4678</td><td>20%</td><td>68.1074</td><td>79.4237</td></tr><tr><td>Avg.</td><td>72.6609</td><td>79.9918</td><td>Avg.</td><td>74.3398</td><td>81.2407</td></tr><tr><td rowspan="6">horse_colic</td><td>Clean</td><td>86.1422</td><td>87.1719</td><td>Clean</td><td>86.1422</td><td>87.1719</td></tr><tr><td>5%</td><td>82.6374</td><td>86.9563</td><td>5%</td><td>83.9046</td><td>85.1187</td></tr><tr><td>10%</td><td>78.9421</td><td>84.4018</td><td>10%</td><td>80.2873</td><td>86.9365</td></tr><tr><td>15%</td><td>75.6189</td><td>79.6674</td><td>15%</td><td>76.9548</td><td>83.3412</td></tr><tr><td>20%</td><td>72.3046</td><td>78.1897</td><td>20%</td><td>73.6285</td><td>78.5403</td></tr><tr><td>Avg.</td><td>79.129</td><td>83.2774</td><td>Avg.</td><td>80.1835</td><td>84.2217</td></tr><tr><td colspan="2">Overall Average</td><td>75.8949</td><td>81.6346</td><td></td><td>77.2617</td><td>82.7312</td></tr></table>

The results show that IFW-BLS consistently preserves higher accuracy than standard BLS as the corruption level increases. On the blood dataset, the average accuracy of BLS under feature outliers is 72.6609%, whereas IFW-BLS reaches 79.9918%. Under label noise, the corresponding averages are 74.3398% and 81.2407%. The improvement remains visible at the highest contamination level: at 20% outliers, IFW-BLS exceeds BLS by 10.4359%, and at 20% label noise it exceeds BLS by 11.3163%. A similar trend is observed on horse\_colic. IFW-BLS improves the average accuracy from 79.129% to 83.2774% under outliers and from 80.1835% to 84.2217% under label noise. The overall averages across both datasets further support this behavior: IFW-BLS obtains 81.6346% under outliers and 82.7312% under noise, compared with 75.8949% and 77.2617% for BLS. These gains indicate that the proposed model is not only better on clean data but also degrades more slowly when the training set is corrupted. This robustness behavior is consistent with the two components of IFW-BLS. The intuitionistic fuzzy score reduces the contribution of samples that are globally far from their class structure or locally surrounded by conflicting labels. After this sample-level credibility adjustment, the bounded wave loss further limits the efect of large residuals produced by corrupted samples. Its asymmetric parameter also allows the validation process to choose whether positive or negative deviations should receive stronger penalization. Therefore, the empirical degradation pattern in Table 3 supports the intended dual protection: IF scores act at the sample-reliability level, while the wave loss acts at the residual-penalization level.

## 5 Conclusions

This paper proposed IFW-BLS, an Intuitionistic Fuzzy Wave Broad Learning System designed to improve the robustness of standard BLS from two complementary directions. The first direction is sample-level reliability modeling, where intuitionistic fuzzy scores are used to measure the credibility of each training sample by considering both global class-center consistency and local neighborhood conflict. As a result, samples that are ambiguous, noisy, or inconsistent with their class structure receive reduced influence during training. The second direction is residual-level robustness, where the conventional squared loss is replaced with a bounded and asymmetric wave loss. The boundedness of the wave loss prevents large residuals from dominating the objective, while its asymmetry allows the model to penalize positive and negative deviations diferently according to the data characteristics. The proposed IFW-BLS combines these two mechanisms in a unified credibility-weighted wave-loss objective. In this design, the intuitionistic fuzzy score first scales the residual, and the wave loss then evaluates the credibility-adjusted residual. A Nesterov accelerated gradient-based optimization procedure is used to solve the resulting objective, which avoids the explicit matrix inversion required in conventional BLS training and provides an eficient way to update the output weights. The experimental results validate the efectiveness of the proposed dual-robust design. On 30 UCI benchmark datasets, IFW-BLS achieves superior performance among all compared methods. In addition, the robustness experiments under controlled feature outlier and label noise contamination show that IFW-BLS preserves higher accuracy and degrades more slowly than standard BLS. These findings support the central motivation of the paper: sample-level intuitionistic fuzzy credibility and residual-level wave-loss robustness provide complementary protection, leading to a more reliable BLS model for both clean and corrupted data.

## Bibliography

[1] M. Akhtar, R. Mishra, M. Tanveer, and M. Arshad. Advancing RVFL networks: Robust classification with the HawkEye loss function. In Neural Information Processing: 31st International Conference, ICONIP 2024, Proceedings, Part III, volume 15288 of Lecture Notes in Computer Science, pages 226–240, Singapore, 2025. Springer. https://doi.org/10.1007/ 978-981-96-6582-2\_16.

[2] M. Akhtar, M. Tanveer, and M. Arshad. RoBoSS: A robust, bounded, sparse, and smooth loss function for supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(1):149–160, 2025.

[3] M. Akhtar, M. Tanveer, and M. Arshad. HawkEye: A robust loss function for regression with bounded, smooth, and insensitive zone characteristics.

Applied Soft Computing, 176:113118, 2025. https://doi.org/10.1016/j. asoc.2025.113118.

[4] M. Akhtar, M. Tanveer, and M. Arshad. CAWI: Copula-aligned weight initialization for randomized neural networks. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 3781–3789. PMLR, 2026. URL https://proceedings.mlr.press/v300/akhtar26a.html.

[5] M. Akhtar et al. Advancing supervised learning with the wave loss function: A robust and smooth approach. Pattern Recognition, page 110637, 2024.

[6] M. Akhtar et al. Towards robust and inversion-free randomized neural networks: The XG-RVFL framework. Pattern Recognition, 172:112711, 2026. ISSN 0031-3203.

[7] K. T. Atanassov. Intuitionistic fuzzy sets. In Intuitionistic Fuzzy Sets, pages 1–137. Physica-Verlag, 1999.

[8] W. Cao et al. A review on neural networks with random weights. Neurocomputing, 275:278–287, 2018.

[9] C. L. P. Chen and Z. Liu. Broad learning system: A new learning paradigm and system without going deep. In 2017 32nd Youth Academic Annual Conference of Chinese Association of Automation, pages 1271–1276. IEEE, 2017.

[10] C. L. P. Chen, Z. Liu, and S. Feng. Universal approximation capability of broad learning system and its structural variations. IEEE Transactions on Neural Networks and Learning Systems, 30(4):1191–1204, 2018.

[11] J. Demšar. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7:1–30, 2006.

[12] W. Deng et al. Robust dual-model collaborative broad learning system for classification under label noise environments. IEEE Internet of Things Journal, 12(12):21055–21067, 2025.

[13] S. Feng and C. L. P. Chen. Fuzzy broad learning system: A novel neuro-fuzzy model for regression and classification. IEEE Transactions on Cybernetics, 50(2):414–424, 2020.

[14] G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew. Extreme learning machine: Theory and applications. Neurocomputing, 70(1–3):489–501, 2006.

[15] J.-W. Jin and C. L. P. Chen. Regularized robust broad learning system for uncertain data modeling. Neurocomputing, 322:58–69, 2018.

[16] A. K. Malik et al. Random vector functional link network: recent developments, applications, and future directions. Applied Soft Computing, 143: 110377, 2023.

[17] Y.-H. Pao, G.-H. Park, and D. J. Sobajic. Learning and generalization characteristics of the random vector functional-link net. Neurocomputing, 6 (2):163–180, 1994.

[18] A. Quadir, M. Akhtar, and M. Tanveer. Enhancing multiview synergy: Robust learning by exploiting the wave loss function with consensus and complementarity principles. Neural Networks, 188:107433, 2025. https: //doi.org/10.1016/j.neunet.2025.107433.

[19] S. Rezvani, X. Wang, and F. Pourpanah. Intuitionistic fuzzy twin support vector machines. IEEE Transactions on Fuzzy Systems, 27(11):2140–2151, 2019.

[20] M. Sajid, A. K. Malik, and M. Tanveer. Intuitionistic fuzzy broad learning system: Enhancing robustness against noise and outliers. IEEE Transactions on Fuzzy Systems, 32(8):4460–4469, 2024.

[21] M. Sajid, A. Quadir, and M. Tanveer. Wave-RVFL: A randomized neural network based on wave loss function. In Neural Information Processing, pages 242–257, Singapore, 2025. Springer Nature Singapore. ISBN 978-981- 96-6579-2.

[22] L. A. Zadeh. Fuzzy sets. Information and Control, 8(3):338–353, 1965.

[23] L. Zhang and P. N. Suganthan. A survey of randomized algorithms for training neural networks. Information Sciences, 364:146–155, 2016.

[24] Y. Zheng et al. Broad learning system based on maximum correntropy criterion. IEEE Transactions on Neural Networks and Learning Systems, 32(7):3083–3097, 2021.