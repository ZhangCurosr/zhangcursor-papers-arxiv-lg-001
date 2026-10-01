# Can Domain Generalization be Guaranteed in Small-Sample Learning?

Hong Zheng ∗

zhinhom@icloud.com

## Abstract

The small-sample learning problem remains a fundamental challenge in machine learning because limited training data lead to unstable model estimation and generalization. Struc tural Risk Minimization (SRM) has long been regarded as a principled solution under the classical i.i.d. assumption. However, domain generalization (DG) violates this assumption, leaving the theoretical role of SRM in DG largely unexplored. To bridge this gap, we establish the first theoretical guarantees for SRM in DG under mild assumptions. Specifically, based on the concept of stability, we derive learning consistency and generalization error bounds and prove that these bounds become tight when the hypotheses satisfy the stability condition. Building upon this, under a specific hypothesis space assumption, we establish stability, learning, and generalization bounds for SRM. We further discuss the applicability of these bounds to deep learning. This work establishes theoretical foundations for SRM under distribution shifts and sheds light on the design of robust DG algorithms in small-sample scenarios.

Keywords: Stability, Domain Generalization, Generalization Bound, Regularization

## 1 Introduction

Domain generalization (DG) aims to train machine learning (ML) models using severa available data domains, ensuring that they perform well on related but inaccessible (unseen) domains (Muandet et al., 2013). This is meaningful because it uses finite sets to develop methods applicable to infinite sets, which can be regarded as a foundation for developing artificial general intelligence. Recently, several studies have explored invariance (Arjovsky et al., 2019; Ahuja et al., 2020) and causality (Christiansen et al., 2021; Wang et al., 2023) from input data pairs, assuming they are essential for generalization, while developing learning approaches based on empirical risk minimization (ERM). By (Vapnik, 2013), the success of ERM and its variants relies on the assumption of suficient data, i.e., large-sample learning. However, in real-world applications, such an assumption may not always be satisfied, particularly in scenarios where data collection is expensive. In other words, we have to face the small-sample learning problem (a term introduced by Vapnik) in many learning scenarios. Under such scenarios, model estimation and generalization become unstable (Wang et al., 2024).

How can this challenge be addressed? Is it through data augmentation, data generation using large models, or Structural Risk Minimization (SRM)? Data augmentation, whether applied online or ofline, violates data independence assumptions and cannot increase the amount of independent samples. Although it may improve empirical performance, model estimation remains theoretically unstable. Moreover, its efectiveness relies heavily on practitioners’ experience rather than a principled learning paradigm (Pereira et al., 2022). Recent work (Shumailov et al., 2024) proves that using generated data can lead to model collapse. Specifically, learning from AI-derived texts causes models to forget rare information as outputs become increasingly homogeneous. Moreover, AI-generated images reproduce training-data semantics and appearances, indicating compromised data independence. Then, SRM becomes an applicable method, as it at least preserves the data independence assumption. However, in DG, data are not identically distributed, leaving the theoretical role of SRM unclear.

In this work, we establish theoretical guarantees (Theorem 7) for SRM under DG scenarios with mild assumptions, thereby bridging this gap. First, based on the concept of hypothesis stability, we derive learning consistency and generalization error bounds and prove that the tightness of these bounds is determined by the stability of the hypothesis. By assuming a reproducing kernel Hilbert space, we then establish stability, learning, and generalization bounds for SRM. Furthermore, we extend our theoretical results to deep learning by introducing corresponding regularizations for SRM and analyzing their stability, learning consistency, and generalization error. In conclusion, we contribute by theoretically proving that DG can be guaranteed in small-sample learning. Through this, we reveal that, when data samples are limited, a suficiently large SRM regularization parameter is required to obtain tighter stability bounds for hypotheses, thereby achieving tighter learning and generalization bounds.

## 2 Theoretical Results

## 2.1 Preliminaries

We here introduce some notation and related concepts that will be used to derive our methods. Considering $\mathcal { X } \subset \mathbb { R }$ and $\mathcal { V } \subset \mathbb { R }$ being respectively an input and output space, we have a training set

$$
D = \{ Z _ { 1 } = ( X , Y ) _ { 1 } , \ldots , Z _ { E } = ( X , Y ) _ { E } \}
$$

of size E in ${ \mathcal { Z } } = { \mathcal { X } } \times { \mathcal { Y } }$ drawn from unknown distributions in $\mathcal { P }$ . Assume that each $Z \in D$ contains N data pairs $( x _ { j } , y _ { j } ) , { \mathrm { i . e . , ~ } } ( X , Y ) : = \{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { N }$ . Under domain generalization (DG), $( X , Y ) _ { i } { \sim } ^ { i . i . d . } P _ { i }$ for all $i \in [ E ]$ , and $P _ { i } \neq P _ { j }$ for all $i \neq j \in [ E ]$ . Thus, the overall samples follow an independent non-identically distributed $\mathrm { ( i . n . d . ) }$ assumption. The training set D is typically regarded as the source set, while the testing set $D _ { t }$ is regarded as the target set. Similarly, $P _ { i } \neq P _ { t }$ for all $i \in [ E ]$ and all $t \in [ E _ { t } ]$ , where $E _ { t }$ denotes the number of target domains in $D _ { t }$ . Furthermore, no information from $D _ { t }$ is available during training.

Given a source set D of domain size E, for each $i = 1 , \ldots , E$ , we construct a modified dataset by replacing the v-th domain of D as follows:

$$
D ^ { v } = \{ Z _ { 1 } , \ldots , Z _ { v - 1 } , Z _ { v } ^ { \prime } , Z _ { v + 1 } \ldots , Z _ { E } \} ,
$$

where the replacement $Z _ { v } ^ { \prime }$ is assumed to be drawn from $\mathcal { P }$ and to satisfy the above distribution assumption. Here, $Z _ { v } ^ { \prime }$ also denotes a group of N sample pairs, rather than a single sample pair, which difers from the setting in (Bousquet and Elisseef, 2002). Only D participates in the learning process, whereas $D ^ { v }$ is excluded from learning and is used solely for stability validation.

We consider a hypothesis $h \in { \mathcal { H } } : \mathcal { X } \to \mathcal { Y }$ as a deterministic mapping for prediction. Then, we assume that all mappings are measurable and all sets are countable, which does not restrict the generality of the results presented hereafter. To measure the accuracy of predictions from $h ,$ we consider the cost function $c : \mathcal { V } \times \mathcal { V } \to \mathbb { R }$ , and the loss of h with respect to data pairs $Z = ( X , Y )$ is then defined as

$$
\ell ( h , Z ) = \ell ( h , ( X , Y ) ) = c ( h ( X ) , Y ) .
$$

Correspondingly, given any set $D ,$ the expected risk of h is defined by

$$
R ( h , D ) = \frac { 1 } { E } \sum _ { i = 1 } ^ { E } \mathbb { E } _ { Z \sim P _ { i } } [ \ell ( h , Z ) ] = \frac { 1 } { E } \sum _ { i = 1 } ^ { E } \mathbb { E } _ { Z _ { i } } [ \ell ( h , Z _ { i } ) ] ,
$$

and the empirical risk of h is defined by

$$
\hat { R } ( h , D ) = \frac { 1 } { E N } \sum _ { ( i , j ) } \ell ( h , ( x _ { j } , y _ { j } ) _ { i } ) = \frac { 1 } { E N } \sum _ { i = 1 } ^ { E } \ell ( h , Z _ { i } ) ,
$$

where $( i , j ) \in [ E ] \times [ N ]$ . Given the set $D ^ { v }$ , we consider only the empirical risk $\hat { R } ( h , D ^ { v } )$ . In the following, we will also use the shorthand notations $\hat { R } ( h ) \equiv \hat { R } ( h , D ) , R ( h ) \equiv R ( h , D )$ ， and $\hat { R } ^ { \prime } ( h ) \equiv \hat { R } ( h , D ^ { v } )$

Note that $R ( h )$ is typically defined as $R ( h ) = \mathbb { E } _ { P \in \mathcal { P } _ { E } } \mathbb { E } _ { Z \in P } [ \ell ( h , Z ) ]$ , where $\mathcal { P } _ { E }$ denotes the set of source distributions. Since $\mathcal { P } _ { E }$ contains $E$ observed domains in DG, the outer expectation reduces to the empirical average over domains, i.e.,E $\begin{array} { r } { { \bf \Lambda } _ { P \in { \mathcal P } _ { E } } [ \cdot ] = E ^ { - 1 } \sum _ { i = 1 } ^ { E } [ \cdot ] } \end{array}$

## 2.2 Stability and Bounds

We begin by establishing the basic definition of hypothesis stability under DG scenarios. This is important for achieving our goal: we aim to obtain bounds on the learning consistency and generalization error of hypotheses learned by SRM, and we want these bounds to be tight when the hypotheses satisfy the stability condition.

Definition 1 (Error Stability) Any hypothesis $h \in \mathcal H$ has error stability $\beta$ with respect to the loss function ℓ if the following holds, $\forall D \in \mathcal { Z } ^ { E } , \forall v \in [ E ]$ 2

$$
\frac { 1 } { E N } \left| \sum _ { Z \in D } \ell ( h , Z ) - \sum _ { Z ^ { \prime } \in D ^ { v } } \ell ( h , Z ^ { \prime } ) \right| \leq \beta ,
$$

which can also be written

$$
\left| \hat { R } ( h ) - \hat { R } ^ { \prime } ( h ) \right| \leq \beta .\tag{1}
$$

Remarks. (a) Strictly, we should write ${ \mathcal { Z } } ^ { E \times N }$ instead of $\mathcal { Z } ^ { E }$ . For simplicity, we use $\mathcal { Z } ^ { E }$ hereafter. (b) The above stability measures the absolute diference between the empirical risks of the same hypothesis h on two datasets, D and $D ^ { v }$ . This notion difers from the stability definitions in (Bousquet and Elisseef, 2002), where stability is typically defined in terms of the expected risks of diferent hypotheses.

Clearly, a small β indicates that the hypothesis exhibits good stability. Based on this definition, we establish the following exponential decay bound, which is derived from a classical concentration inequality.

Theorem 2 For any measurable function $R : \mathcal { H } \times \mathcal { Z } ^ { E } \to \mathbb { R }$ , any hypothesis $h \in \mathcal H$ that satisfies the stability in Definition 1, i.e., h has $\beta - s t a b i l i t y$ , and any $\varepsilon > 0$ , the following inequality holds, $\forall D \in \mathcal { Z } ^ { E }$ ，

$$
P _ { D } \left[ \hat { R } ( h ) - R ( h ) \geq \varepsilon \right] \leq \exp \left\{ \frac { - 2 \varepsilon ^ { 2 } } { E \beta } \right\} .\tag{2}
$$

Proof Consider $\hat { R } ( h , D )$ with respect to any hypothesis $h .$ . Since h has $\beta$ error stability, the following holds, $\forall D \in \mathcal { Z } ^ { E } , \forall v \in [ E ]$

$$
\left| \hat { R } ( h , D ) - \hat { R } ( h , D ^ { v } ) \right| \leq \beta .
$$

This satisfies the condition of Lemma 8, i.e.,

$$
\operatorname* { s u p } _ { \substack { D \in \mathcal { Z } ^ { E } , Z _ { v } ^ { \prime } \in \mathcal { Z } } } \Big | \hat { R } ( h , D ) - \hat { R } ( h , D ^ { v } ) \Big | \leq \beta , \forall v \in [ E ] .
$$

Then, we have

$$
\sum _ { v \in [ E ] } \operatorname* { s u p } _ { D , Z _ { v } ^ { \prime } } \Big | \hat { R } ( h , D ) - \hat { R } ( h , D ^ { v } ) \Big | \leq E \beta , \forall h \in \mathcal { H } ,
$$

completing the proof.

Remarks. (a) With probability at least $1 - \delta ,$ it holds that $\epsilon _ { S } ( h ) \leq \sqrt { E \beta \ln ( 1 / \delta ) / 2 }$ , where $\epsilon _ { S } ( h ) : = R ( h ) - \hat { R } ( h )$ . (b) As expected, a more restrictive stability criterion yields a correspondingly tighter bound. This bound is a McDiarmid-type bound, which is more flexible and capable of estimating nonlinear statistics (Li and Liu, 2024). When no further information is available, the convergence rate of this bound cannot be improved. Therefore, there is no need to derive a lower bound to establish its optimality. $\mathrm { ( c ) }$ This bound establishes the learning consistency between the expected risk and the empirical risk.

Based on the above learning consistency bound, we observe that stability plays a critical role in learning performance, as the tightness of the bound is closely related to the stability of hypotheses. Namely, a smaller $\beta$ leads to a tighter bound on $\epsilon _ { S } ( h )$ . Note that when $E$ is large while $\beta$ remains large, this learning bound can become loose or even vacuous, highlighting the necessity of achieving a small $\beta .$

We then turn our attention to generalization and introduce the following expected risk of h on the target domains,

$$
R _ { T } ( h ) \equiv R ( h , D _ { t } ) = \frac { 1 } { E _ { t } } \sum _ { t = 1 } ^ { E _ { t } } \mathbb { E } _ { Z \sim P _ { t } } [ \ell ( h , Z ) ] .
$$

This target risk $R _ { T } ( h )$ is also defined as average expected loss of hypothesis h over $E _ { t }$ target domains. Accordingly, we establish the following generalization bound.

Theorem 3 (Generalization Bound) Let $R _ { T } ( h )$ denote the above definition of the expected risk of any $h \in \mathcal H$ on the target domains, where h has $\beta - s t a b i l i t y$ . Assume that $0 \leq \ell \leq M$ . Then, the following inequality holds with probability at least $1 - \delta$ 2

$$
\epsilon _ { T } ( h ) \leq \sqrt { \frac { E \beta \ln ( 1 / \delta ) } { 2 } } + C \rho ,\tag{3}
$$

where, $\epsilon _ { T } ( h ) : = R _ { T } ( h ) - \hat { R } ( h ) , C = ( 2 \sqrt { 2 } M ) / E E _ { t }$ , and $\textstyle \rho = \sum _ { t } \sum _ { i } d _ { J S } ( P _ { t } , P _ { i } )$ . Here, $d _ { J S } ( P , Q ) = \sqrt { J S ( P | | Q ) }$ , where $J S ( \cdot | | \cdot )$ denotes the Jensen-Shannon (JS) divergence.

Proof According to Lemma $9 ,$ , for any $P _ { i }$ and $Q _ { j }$ , we have

$$
\left| \mathbb { E } _ { P _ { i } } [ \ell ] - \mathbb { E } _ { Q _ { j } } [ \ell ] \right| \le 2 \sqrt { 2 } M d _ { J S } ( P _ { i } , Q _ { j } ) .
$$

Then, extending this inequality to all source and target domains, we obtain

$$
\left| \frac { 1 } { E _ { t } } \sum _ { t } ^ { E _ { t } } \mathbb { E } _ { P _ { t } } [ \ell ] - \frac { 1 } { E } \sum _ { i } ^ { E } \mathbb { E } _ { P _ { i } } [ \ell ] \right| \le C \sum _ { t } ^ { E _ { t } } \sum _ { i } ^ { E } d _ { J S } ( P _ { t } , P _ { i } ) ,
$$

where $C = ( 2 \sqrt { 2 } M ) / E E _ { t }$ . Let $\begin{array} { r } { \rho = \sum _ { t } \sum _ { i } d _ { J S } ( P _ { t } , P _ { i } ) } \end{array}$ . By the inequality $| a | - | b | \leq | a - b |$ 2 we have

$$
R _ { T } ( h ) - R ( h ) \leq C \rho .
$$

Then, the proof is completed by applying Theorem $2 .$

Remarks. (a) The tightness of this generalization bound is still linked to the stability of h. Specifically, given $\rho ,$ a small $\beta$ lead to a tighter generalization bound. (b) This term $\rho$ captures a central challenge in DG, namely that the generalization bound is afected by the distribution discrepancy between the source and target domains. Unfortunately, there is nothing we can do about this term, as access to $D _ { t }$ is unavailable during learning. (c) $d _ { J S } ( \cdot , \cdot )$ is a metric that satisfies symmetry and the triangle inequality. When the JS divergence is defined using the natural logarithm, $0 \leq d _ { J S } ( \cdot , \cdot ) \leq \sqrt { \ln { 2 } }$

Similarly, the above generalization bound demonstrates the critical role of $\beta$ in generalization. This is IMPORTANT because the stability of $h$ determines both learning and generalization. Once $\beta$ is obtained, we derive the corresponding learning and generalization bounds. Moreover, since $h$ is arbitrary, the hypothesis obtained by SRM also satisfies the above bounds on $\epsilon _ { S } ( h )$ and $\epsilon _ { T } ( h )$ . We then conduct further investigations.

The following two definitions are introduced as useful tools for our derivation: one is the σ-admissibility of $\ell ,$ and the other is optimal stability $( \beta ^ { * } )$

Definition 4 A loss function ℓ defined on $\mathcal { H } \times \mathcal { V }$ is σ-admissible with respect to $\mathcal { H }$ is the associated cost function c is convex and the following condition holds, $\forall h , h ^ { \prime } \in { \mathcal { H } } .$ $\forall ( X , Y ) \in { \mathcal { Z } }$

$$
\left| \ell ( h , ( X , Y ) ) - \ell ( h ^ { \prime } , ( X , Y ) ) \right| \leq \sigma \left| h ( X ) - h ^ { \prime } ( X ) \right| .
$$

Remarks. The σ-admissibility essentially implies that ℓ is σ-Lipschitz continuous. This establishes a relationship between the loss diference of two hypotheses and the distance between the corresponding hypotheses.

Definition 5 (β<sup>∗</sup>-stability) Considering the functional $f ( h ) = | \hat { R } ( h ) - \hat { R } ^ { \prime } ( h ) |$ , let $h ^ { \ast } \in$ arg $\operatorname* { m i n } _ { h \in \mathcal { H } } f ( h )$ and $h ^ { * }$ is non-zero. We say that $h ^ { * }$ achieves the minimum error stability $\beta ^ { * } , i . e .$

$$
\left| \hat { R } ( h ^ { * } ) - \hat { R } ^ { \prime } ( h ^ { * } ) \right| \leq \beta ^ { * } .
$$

Remarks. (a) Note that $h ^ { * }$ is the minimizer of the functional $f ( h )$ , rather than of $\hat { R } ( h )$ or $\hat { R } ^ { \prime } ( h )$ , and that $h ^ { * }$ is not necessarily unique. (b) This non-uniqueness implies the existence of multiple hypotheses that satisfy this stability condition.

We define $\Delta h = | h ( X ) - h ^ { * } ( X ) | , \forall h \in \mathcal { H } , \forall X \in \mathcal { X }$ . Then, we have

$$
\left| \hat { R } ( h ) - \hat { R } ^ { \prime } ( h ) \right| \le \sigma ( \Delta h _ { D } + \Delta h _ { D ^ { v } } ) + \beta ^ { * } .\tag{4}
$$

Here, $\Delta h _ { D }$ and $\Delta h _ { D ^ { v } }$ are defined respectively as $| h ( X ) - h ^ { * } ( X ) | _ { D }$ and $| h ( X ) - h ^ { * } ( X ) | _ { D ^ { v } }$ where the subscripts D and $D ^ { v }$ denote the sets over which the values of X are taken. Proof We omit the hat notation in $\hat { R }$ for convenience, i.e., R denotes $\hat { R }$ . Since, we have $h ^ { \ast } \in$ arg min $_ { \mathit { h } \in \mathcal { H } } \left| \mathit { R } ( \mathit { h } ) - \mathit { R } ^ { \prime } ( \mathit { h } ) \right|$ . Then, for any $h \in \mathcal H$ , we have

$$
\begin{array} { r l } & { \left| R ( h ) - R ^ { \prime } ( h ) \right| = \left| R ( h ) + R ( h ^ { * } ) - R ( h ^ { * } ) + R ^ { \prime } ( h ^ { * } ) - R ^ { \prime } ( h ^ { * } ) + R ^ { \prime } ( h ) \right| } \\ & { \qquad \leq | R ( h ) - R ( h ^ { * } ) | + \left| R ^ { \prime } ( h ) - R ^ { \prime } ( h ^ { * } ) \right| + \left| R ( h ^ { * } ) - R ^ { \prime } ( h ^ { * } ) \right| } \\ & { \qquad \leq | R ( h ) - R ( h ^ { * } ) | + \left| R ^ { \prime } ( h ) - R ^ { \prime } ( h ^ { * } ) \right| + \beta ^ { * } } \\ & { \qquad \leq \sigma | h ( X ) - h ^ { * } ( X ) | _ { D } + \sigma | h ( X ) - h ^ { * } ( X ) | _ { D ^ { v } } + \beta ^ { * } } \end{array}
$$

The first inequality holds by the triangle inequality, the second holds under the definition that $| R ( h ^ { * } ) - R ^ { \prime } ( h ^ { * } ) | ~ \leq ~ \beta ^ { * }$ , and the last holds due to the σ-admissibility of the loss functions.

The Structural Risk Minimization (SRM) framework for DG is subsequently formulated as $\hat { R } _ { r } ( h ) \equiv \hat { R } _ { r } ( h , D , \lambda )$ , where

$$
\hat { R } _ { r } ( h , D , \lambda ) = \hat { R } ( h , D ) + \lambda \mathcal { N } ( h ) = \frac { 1 } { E N } \sum _ { i = 1 } ^ { E } \ell ( h , Z _ { i } ) + \lambda \mathcal { N } ( h ) .
$$

Here, $\lambda$ is a penalty factor, and $\mathcal { N } ( h )$ represents any regularization for h. When using the dataset $D ^ { v }$ , we obtain $\hat { R } _ { r } ^ { \prime } ( h ) \equiv \hat { R } _ { r } ^ { \prime } ( h , D ^ { v } , \lambda )$

Before presenting the main results, we establish the following result with respect to the regularizer $\mathcal { N } ( \cdot )$

Theorem 6 Let ℓ be σ-admissible with respect to ${ \mathcal { H } } ,$ , let $\mathcal { N }$ be a functional defined on $\mathcal { H } _ { \mathrm { ~ } }$ and let $h ^ { * }$ has $\beta ^ { * } { - } s t a b i l i t y$ . Assuming that $\nabla \hat { R } ( h ^ { * } ) \ge 0$ , for any $t \in [ 0 , 1 ]$ , any $h \in \mathcal H$ $\hat { R } _ { r } ( h ^ { * } )$ , and $\hat { R } _ { r } ^ { \prime } ( h ^ { * } )$ , we have

$$
\mathcal { N } ( h ^ { * } ) - \mathcal { N } ( h ^ { * } + t \Delta h ) \leq \frac { \sigma t } { 2 \lambda E N } ( \Delta h _ { D } + \Delta h _ { D } v ) .\tag{5}
$$

Proof We omit the hat notation in $\hat { R }$ for convenience, i.e., R denotes ${ \hat { R } } .$ Since $h ^ { \ast } \in$ arg $\begin{array} { r } { \operatorname* { m i n } _ { h \in \mathcal { H } } | R ( h ) - R ^ { \prime } ( h ) | } \end{array}$ , for any $h \in { \mathcal { H } }$ , we have

$$
\left| R ( h ^ { * } ) - R ^ { \prime } ( h ^ { * } ) \right| \leq \left| R ( h ^ { * } + t \Delta h ) - R ^ { \prime } ( h ^ { * } + t \Delta h ) \right| .
$$

Here, $( h ^ { * } + t \Delta h ) ( X ) \neq h ^ { * } ( X )$ for any $t > 0$ . Then, we have

$$
( R ( h ^ { * } ) - R ^ { \prime } ( h ^ { * } ) ) ^ { 2 } \leq ( R ( h ^ { * } + t \Delta h ) - R ^ { \prime } ( h ^ { * } + t \Delta h ) ) ^ { 2 } .
$$

Let $\Phi ( h ) = | R ( h ) - R ^ { \prime } ( h ) |$ , we have $0 \in \nabla \Phi ( h ^ { * } )$ , which implies that $\exists g \in \nabla R ( h ^ { * } ) \cap \nabla R ^ { \prime } ( h ^ { * } )$ Note that here 0 and g belong to the hypothesis class rather than being scalars.

Now, let $\phi ( t ) = R ( h ^ { * } + t \Delta h )$ and $\phi ^ { \prime } ( t ) = R ^ { \prime } ( h ^ { * } + t \Delta h )$ . Then, we have

$$
\begin{array} { r } { \nabla _ { t } \phi ( t ) = \nabla R ( h ^ { * } + t \Delta h ) \Delta h , } \\ { \nabla _ { t } \phi ^ { \prime } ( t ) = \nabla R ^ { \prime } ( h ^ { * } + t \Delta h ) \Delta h . } \end{array}
$$

Consider $\varphi ( t ) = \phi ( t ) \phi ^ { \prime } ( t )$ , we have

$$
\begin{array} { r l } & { \nabla _ { t } \varphi ( t ) = \nabla R ( h ^ { \ast } + t \Delta h ) \Delta h \phi ^ { \prime } ( t ) + \nabla R ^ { \prime } ( h ^ { \ast } + t \Delta h ) \Delta h \phi ( t ) , } \\ & { \nabla _ { t } \varphi ( 0 ) = \nabla R ( h ^ { \ast } ) \Delta h \phi ^ { \prime } ( 0 ) + \nabla R ^ { \prime } ( h ^ { \ast } ) \Delta h \phi ( 0 ) . } \end{array}
$$

Since $R$ is convex (see Definition 4), its gradient is a monotone mapping, so we have $\nabla R ( h ^ { * } +$ $t \Delta h ) \geq \nabla R ( h ^ { * } )$ for any $t \geq 0$ . Moreover, we assume that $\nabla R ( h ^ { * } ) \geq 0$ . Then, we obtain $\nabla _ { t } \varphi ( t ) \geq \nabla _ { t } \varphi ( 0 )$ . This implies that

$$
R ( h ^ { * } + t \Delta h ) R ^ { \prime } ( h ^ { * } + t \Delta h ) \geq R ( h ^ { * } ) R ^ { \prime } ( h ^ { * } ) .
$$

Then, we compute

$$
\begin{array} { r l } & { \quad ( R ( h ^ { * } + t \Delta h ) + R ^ { \prime } ( h ^ { * } + t \Delta h ) ) ^ { 2 } - ( R ( h ^ { * } ) + R ^ { \prime } ( h ^ { * } ) ) ^ { 2 } } \\ & { = ( R _ { \Delta } ^ { 2 } + 2 R _ { \Delta } R _ { \Delta } ^ { \prime } + ( R ^ { \prime } _ { \Delta } ) ^ { 2 } ) - ( R ^ { 2 } + 2 R R ^ { \prime } + ( R ^ { \prime } ) ^ { 2 } ) } \\ & { = ( R _ { \Delta } ^ { 2 } + 2 R _ { \Delta } R _ { \Delta } ^ { \prime } + ( R ^ { \prime } _ { \Delta } ) ^ { 2 } - 2 R _ { \Delta } R _ { \Delta } ^ { \prime } + 2 R _ { \Delta } R ^ { \prime } _ { \Delta } ) } \\ & { \quad - ( R ^ { 2 } + 2 R R ^ { \prime } + ( R ^ { \prime } ) ^ { 2 } - 2 R R ^ { \prime } + 2 R R ^ { \prime } ) } \\ & { = ( ( R _ { \Delta } - R ^ { \prime } _ { \Delta } ) ^ { 2 } + 4 R _ { \Delta } R ^ { \prime } _ { \Delta } ) - ( ( R - R ^ { \prime } ) ^ { 2 } + 4 R R ^ { \prime } ) } \\ & { = ( ( R _ { \Delta } - R ^ { \prime } _ { \Delta } ) ^ { 2 } - ( R - R ^ { \prime } ) ^ { 2 } ) + 4 ( R _ { \Delta } R ^ { \prime } _ { \Delta } - R R ^ { \prime } ) . } \end{array}
$$

Based on the previous derivations, we obtain

$$
\begin{array} { r l } & { ( R ( h ^ { * } ) + R ^ { \prime } ( h ^ { * } ) ) ^ { 2 } \leq ( R ( h ^ { * } + t \Delta h ) + R ^ { \prime } ( h ^ { * } + t \Delta h ) ) ^ { 2 } } \\ & { \quad R ( h ^ { * } ) + R ^ { \prime } ( h ^ { * } ) \leq R ( h ^ { * } + t \Delta h ) + R ^ { \prime } ( h ^ { * } + t \Delta h ) . } \end{array}
$$

Then, we replace $R ( h ^ { * } )$ and $R ^ { \prime } ( h ^ { * } )$ with $R _ { r } ( h ^ { * } )$ and $R _ { r } ^ { \prime } ( h ^ { * } )$ , respectively, and obtain

$$
R _ { r } ( h ^ { * } ) + R _ { \mathit { r } } ^ { \prime } ( h ^ { * } ) \leq R _ { r } ( h ^ { * } + t \Delta h ) + R _ { \mathit { r } } ^ { \prime } ( h ^ { * } + t \Delta h ) .
$$

Consequently, we have

$$
2 \lambda ( \mathcal { N } ( h ^ { * } ) - \mathcal { N } ( h ^ { * } + t \Delta h ) ) \le ( R ( h ^ { * } + t \Delta h ) - R ( h ^ { * } ) ) + ( R ^ { \prime } ( h ^ { * } + t \Delta h ) - R ^ { \prime } ( h ^ { * } ) )
$$

Finally, based on the σ-admissibility of the loss functions and the definition of $R ,$ we obtain

$$
\mathcal { N } ( h ^ { * } ) - \mathcal { N } ( h ^ { * } + t \Delta h ) \leq \frac { \sigma t } { 2 \lambda E N } ( \Delta h _ { D } + \Delta h _ { D } v ) .
$$

Remarks. (a) This result leads to the regularizations associated with the σ-admissibility of loss functions. (b) Similarly, $h ^ { * }$ is not the minimizer of either ${ \hat { R } } _ { r } ( h )$ or $\hat { R } _ { r } ^ { \prime } ( h )$ . (c) This theorem considers only minimizers $h ^ { * }$ satisfying $\nabla \hat { R } ( h ^ { * } ) \geq 0$

Next, we impose an assumption on H to ensure that $\mathcal { N } ( \cdot )$ is well-defined. Specifically, we consider regularization in a reproducing kernel Hilbert space (RKHS). An RKHS is chosen to ensure eficient kernel computation and a complete, stable function space. Accordingly, we define

$$
\mathcal { N } ( h ) = \| h \| _ { k } ^ { 2 } ,
$$

where k refers to the kernel. Typically, we state that $\mathcal { H } _ { k }$ denotes the norm in an RKHS, but for simplicity, we use k here. Based on the fundamental property (reproducing) of an RKHS H, we have

$$
\forall h \in { \mathcal { H } } , \forall X \in { \mathcal { X } } , h ( X ) = \langle h , k ( X , \cdot ) \rangle .
$$

Then, according to Cauchy-Schwarz inequality, we have

$$
\forall h \in \mathcal { H } , \forall X \in \mathcal { X } , | h ( X ) | \leq \| h \| _ { k } \sqrt { k ( X , X ) } .\tag{6}
$$

With all the previous preparations, we finally present the following result for our main goal.

Theorem 7 Assume that H is a reproducing kernel Hilbert space with kernel k such that $\forall X \in { \mathcal { X } } , { \sqrt { k ( X , X ) } } \leq \kappa \leq \infty$ . Let ℓ be σ-admissible with respect to H, and assume that $0 \leq \ell \leq M$ . Any hypothesis h achieved by

$$
\operatorname* { m i n } _ { h \in \mathcal H } \hat { R } ( h ) + \lambda \| h \| _ { k } ^ { 2 } ,\tag{7}
$$

has error stability β respect to ℓ with

$$
\beta \leq \frac { 2 \sigma ^ { 2 } \kappa ^ { 2 } } { \lambda E N } + \beta ^ { * } .
$$

Furthermore, with probability al least $1 - \delta _ { i }$ , the learning consistency bound holds:

$$
\epsilon _ { S } ( h ) \leq \sigma \kappa \sqrt { \frac { \ln ( 1 / \delta ) } { \lambda N } } + \sqrt { \frac { E \beta ^ { * } \ln ( 1 / \delta ) } { 2 } } ,
$$

and the generalization bound is given by

$$
\epsilon _ { T } ( h ) \leq \sigma \kappa \sqrt { \frac { \ln ( 1 / \delta ) } { \lambda N } } + \sqrt { \frac { E \beta ^ { * } \ln ( 1 / \delta ) } { 2 } } + C \rho .
$$

Proof Since $\mathcal { N } ( h ) = \| h \| _ { k } ^ { 2 }$ , based on Theorem 6, we have

$$
t \left\| \Delta h \right\| _ { k } ^ { 2 } \leq \frac { \sigma t } { 2 \lambda E N } ( \Delta h _ { D } + \Delta h _ { D ^ { v } } ) .
$$

According to the assumption that $\forall X \in { \mathcal { X } } , { \sqrt { k ( X , X ) } } \leq \kappa \leq \infty$ and Inequality (6), we have

$$
\begin{array} { r } { \quad \Delta h _ { D } \leq \| \Delta h \| _ { k } \sqrt { k ( X , X ) } \leq \| \Delta h \| _ { k } \kappa , } \\ { \Delta h _ { D ^ { v } } \leq \| \Delta h \| _ { k } \kappa . } \end{array}
$$

Consequently, we obtain

$$
\begin{array} { l } { \displaystyle \| \Delta h \| _ { k } ^ { 2 } \leq \frac { \sigma } { 2 \lambda E N } 2 \| \Delta h \| _ { k } \kappa = \frac { \sigma \| \Delta h \| _ { k } \kappa } { \lambda E N } , } \\ { \displaystyle \| \Delta h \| _ { k } \leq \frac { \sigma \kappa } { \lambda E N } } \end{array}
$$

Moreover, according to Inequality (4), and by taking $\beta \le \sigma ( \Delta h _ { D } + \Delta h _ { D ^ { v } } ) + \beta ^ { * }$ , we have

$$
\begin{array} { r } { \beta \leq 2 \sigma \| \Delta h \| _ { k } \kappa + \beta ^ { * } . } \end{array}
$$

By Theorem 2, we have

$$
\begin{array} { r l r } {  { R ( h ) - \hat { R } ( h ) \leq \sqrt { \frac { E \beta \ln } { 2 } } \leq \sqrt { \frac { E } { 2 } ( \frac { 2 \sigma ^ { 2 } \kappa ^ { 2 } } { \lambda E N } + \beta ^ { * } ) \ln } } } \\ & { } & { \leq \sigma \kappa \sqrt { \frac { \ln } { \lambda N } } + \sqrt { \frac { E \beta ^ { * } \ln } { 2 } } . } \end{array}
$$

Then, by Theorem 3, we have

$$
R _ { T } ( h ) \leq \hat { R } ( h ) + \sqrt { \frac { E \beta \ln ( 1 / \delta ) } { 2 } } + C \rho ,
$$

completing the proof.

Remarks. (a) Up to an intercept, $\beta$ is inversely proportional to the product of the penalty factor and the total number of data samples. As $\lambda E N \to \infty$ , the bound on $\beta$ becomes tight when $\beta ^ { * }$ approaches zero. (b) The tightness of the learning and generalization bounds is now linked to the number of data samples within each domain. Namely, the main order of the above bounds is $\mathcal { O } ( 1 / \sqrt { \lambda N } )$ . (c) Assuming $\beta ^ { * } \to 0$ , the term involving $\beta ^ { * }$ in the above bounds can be regarded as a bias term that is typically negligible.

Now, we have established stability, learning, and generalization bounds for the hypothesis h learned by SRM in DG. By the theorem, for fixed $\sigma , \kappa ,$ and $E ,$ the tightness of all the above bounds is determined by the parameters $\lambda$ and N. Specifically, a suficiently large $\lambda N$ theoretically leads to a smaller $\beta$ , which in turn yields better learning consistency and generalization performance.

## 2.3 Application

Still, one step remains before applying our theoretical results to practical applications. In most deep learning scenarios, the hypothesis space is assumed to be a Euclidean space. In the previous subsection, we presented the stability, learning, and generalization bounds for SRM under the assumption of an RKHS. For a kernel-induced RKHS, the function norm in the RKHS can be transformed into the Euclidean $L _ { 2 } ( \mu )$ norm in the corresponding feature space through finite-dimensional feature representations or kernel feature mappings. This property provides a theoretical foundation for applying RKHS-based analysis to models defined in Euclidean spaces. The detailed derivation is omitted for brevity. We then consider the following SRM framework:

$$
\hat { R } _ { r } ( h ) = \hat { R } ( h ) + \lambda \| h \| _ { 2 } ^ { 2 } ,
$$

where the hypothesis h is parameterized by $\theta ,$ i.e., $h = h _ { \theta }$ . Correspondingly, the learning objective is

$$
\operatorname* { m i n } _ { h \in \mathcal H } \hat { R } ( h ) + \lambda \| h \| _ { 2 } ^ { 2 } .\tag{8}
$$

Note that, for simplicity, we directly use the above Lagrangian formulation as the learning objective instead of the original constrained optimization problem.

We further assume that, for a finite-dimensional RKHS, the induced kernel is normalized, i.e.,

$$
k ( X , X ) \leq 1 , \quad \forall X \in \mathcal { X } ,
$$

which yields $\kappa = 1$ . This assumption is equivalent to bounding the norm of the induced feature representation and naturally extends to Euclidean-space models, including deep neural networks, through their learned feature embeddings.

Accordingly, under the above assumption and settings of Theorem 7, any hypothesis h achieved by learning objective (8) has error stability $\beta$ respect to ℓ with

$$
\beta \leq \frac { 2 \sigma ^ { 2 } } { \lambda E N } + \beta ^ { * } .
$$

Furthermore, with probability al least $1 - \delta .$ the learning consistency bound holds:

$$
\epsilon _ { S } ( h ) \leq \sigma \sqrt { \frac { \ln ( 1 / \delta ) } { \lambda N } } + \sqrt { \frac { E \beta ^ { * } \ln ( 1 / \delta ) } { 2 } } ,
$$

and the generalization bound is given by

$$
\epsilon _ { T } ( h ) \leq \sigma \sqrt { \frac { \ln ( 1 / \delta ) } { \lambda N } } + \sqrt { \frac { E \beta ^ { * } \ln ( 1 / \delta ) } { 2 } } + C \rho .
$$

In the above, we only consider the $L _ { \mathrm { { 2 } - n o r m } }$ . However, in SRM, the $L _ { 1 } ( \mu ) – \mathrm { n o r m }$ is also commonly considered to promote sparsity. Since there is no direct correspondence between $\| h \| _ { k } ^ { 2 }$ and the $L _ { \mathrm { 1 } } \mathrm { - n o r m }$ , we instead consider the following learning objective to promote sparsity,

$$
\operatorname* { m i n } _ { h \in \mathcal { H } } \hat { R } ( h ) + \frac { \lambda } { E } \sum _ { i = 1 } ^ { E } \| h _ { i } \| _ { 2 } ^ { 2 } .\tag{9}
$$

The above implies that $\mathcal { N } ( h ) = \sum \| h _ { i } \|$ , and we still obtain the stability bound

$$
\beta \leq \frac { 2 \sigma ^ { 2 } } { \lambda E N } + \beta ^ { * } ,
$$

which yields the same learning and generalization bounds as those for the learning objective (8). Here, $h _ { i }$ denotes the parameters of h associated with domain i.

Although the parameters are identical across all domains during the learning process, the summation over domains is introduced for the purpose of sparsity regularization. In essence, $\begin{array} { r } { \| h \| _ { 1 } = \sum | \theta _ { i } | . } \end{array}$ i.e., the sum of the absolute values of all weights in $h .$ . Similarly, $\sum \| h _ { i } \|$ 2 can also be viewed as a summation over the parameters of $h ,$ except that each summand is replaced by $\| h _ { i } \| _ { 2 } ^ { 2 } .$ , with the summation taken over the E domains. This formulation promotes domain-level sparsity, i.e., inter-domain sparsity, while encouraging smoothness within each domain. Such a consideration is analogous to the ideas underlying LASSO and Group LASSO; further details are omitted here. Additionally, learning objectives (8) and (9) are strongly convex. Therefore, under gradient-based optimization methods, the optimization converges Q-linearly.

## 2.4 Discussion

Stability. We provide a discussion of Definition 1 and the definitions in (Bousquet and Elisseef, 2002). The stability in (Bousquet and Elisseef, 2002) measures the diference between the expected losses of two models $h _ { 1 }$ and $h _ { 2 }$ trained on D and $D ^ { v }$ , respectively, thereby evaluating the sensitivity of the learning algorithm to changes in training data. In contrast, our stability measures the diference between the empirical risks of the same hypothesis $h$ evaluated on D and $D ^ { v }$ . It assesses whether the hypothesis is sensitive to source domain changes. A larger $\beta$ indicates weaker stability. We consider empirical risk instead of expected risk since it is directly computable in practice, making our measure suitable for evaluating model stability in DG.

$\beta ^ { * } { \mathrm { - s t a b i l i t y } }$ . We may avoid considering $\beta ^ { * } { \mathrm { - s t a b i l i t y } } ,$ as well as $h ^ { * }$ . By Lemma 20 in (Bousquet and Elisseef, 2002), some results regarding $\mathcal { N } ( \cdot )$ associated with the minimizer of SRM can be established, thereby restricting all subsequent theorems to this minimizer. However, by defining $\beta ^ { * } { \mathrm { - s t a b i l i t y } } .$ our theory no longer unnecessarily depends on the restriction to the SRM minimizer. Therefore, in essence, $h ^ { * }$ relaxes the restriction that the learning and generalization bounds must be based on the SRM minimizer, resulting in a milder condition for DG scenarios. This serves as a significant novelty and distinguishes our work from theirs.

σ and M. One may note that, throughout our main results, we use the notation $\sigma$ to denote the Lipschitz constant and M to denote the uniform bound of the loss function. This is because we do not specify the exact loss formulation or the specific task. Here, we present the following examples for classification and regression tasks, whose derivations rely on Lemma 3 provided in the supplemental material. If $\mathcal { V } \in \{ - 1 , 1 \}$ , and considering the following loss function

$$
\ell ( f , ( x , y ) ) = \left\{ { 1 - y f ( x ) } , \begin{array} { l l } { { i f 1 - y f ( x ) \geq 0 } } \\ { { 0 , } } \end{array}  \right. ,
$$

we have $\sigma = 1$ and $M = 1 . { \mathrm { ~ I f ~ } } \mathcal { V } \in \{ 0 , B \}$ and considering the following loss function

$$
\ell ( f , ( x , y ) ) = ( f ( x ) - y ) ^ { 2 } ,
$$

we have $\sigma = 2 B$ and $M = B$ . Other examples can be found in (Bousquet and Elisseef, 2002).

Previous works. Recently, under the assumption of suficient data samples, many studies have developed theoretical analyses of learning under distribution shifts. To the best of our knowledge, our work is the first to investigate small-sample learning in DG from a theoretical perspective. Therefore, a direct comparison with these works is not meaningful, and we only provide some related discussions. One well-known work is presented by Ben-David et al. (2010), which has motivated many subsequent studies on representation learning. Their work is elegant and introduces a challenge term in the upper bound that is similar to our $\rho .$ Tong et al. (2023) presented an upper bound for their learning method; however, the connection between the hyperparameters of the learning algorithm and the derived bound was not established. Zheng and Teng (2026) provides insights into how distribution shift itself afects learning, but their upper bound relies on a strong assumption regarding distribution divergence. Besides upper bounds, a lower bound for DG was presented by Wang et al. (2024) to explain the phenomenon observed in (Gulrajani and Lopez-Paz, 2020); however, it requires an infinite number of data samples in each domain.

Limitations. It should be emphasized that our theoretical bounds for SRM are established within the RKHS framework. Consequently, the results naturally extend to learning problems whose hypothesis spaces admit an RKHS formulation, such as deep learning. However, when the hypothesis space does not admit an RKHS formulation, our theoretical results may no longer hold.

## 3 Related Works

We briefly introduce some empirical and theoretical methods in DG. For more details, see the references.

Empirical Methods: Since IRM (Arjovsky et al., 2019), various extensions, including IRM-Games (Ahuja et al., 2020), invariant information bottleneck (Li et al., 2022), and P-IRM (Choraria et al., 2023), have been proposed, along with studies on invariant representation learning in latent spaces (Muandet et al., 2013; Li et al., 2018). These methods assume that invariant features from observed domains improve generalization to Out-ofdistribution domains. However, recent works have challenged this assumption by revealing failure cases of invariance-based methods (Rosenfeld et al., 2020) and proposing low-risk alternatives (Wang et al., 2022). Causality-based approaches (Christiansen et al., 2021; Wang et al., 2023) have also been introduced to address domain shifts but generally require stronger assumptions. In addition, the sparsity of ML models has also gained increasing attention recently, such as in (Xu et al., 2024; Galanti et al., 2023; Levy and Abramovich, 2023).

Theoretical Methods: From the VC bounds (Vapnik and Chervonenkis, 2015; Vapnik, 2013) to the Massart-noisy bounds (Massart and N´ed´elec, 2006), the mystery of the learning pattern in machine learning has been revealed gradually. These theories provide fundamental inequalities for analyzing learning problems and inspire subsequent studies. Ben-David et al. (2010) provides an early theoretical study of learning under distribution shift, motivating further investigations in DG. Many eforts have since focused on developing guarantees for well-generalized ML models, e.g., (Wang et al., 2024; Zheng and Teng, 2026). Our work contributes to this line of research by addressing learning problems under limited data.

## 4 Conclusion

In this study, we present theoretical guarantees for SRM in DG scenarios and provide corresponding applications. Our results imply that, even when data samples are limited, stable, well-performing, and generalizable models can still be learned, i.e., DG can be theoretically guaranteed in small-sample learning. This finding is important for practical applications of ML and deep learning methods, as suficient training samples cannot be collected in every application field. Moreover, for fields where data collection is highly expensive, our theory provides an alternative solution. However, we cannot theoretically characterize the selection of λ, which serves as a limitation of our work. We expect to address this issue in future research.

## Ackonwlegement

I would like to express my sincere gratitude to my supervisors, Prof. T & Prof. L, for providing me with the time, freedom, and support to pursue my own research interests and ideas throughout my studies.

## Appendix A.

In this appendix, we only provide the lemmas, along with their porous, used to prove our main theorems.

## A.1 Lemmas

Lemma 8 (McDiarmid’s inequality (McDiarmid et al., 1989)) Let D and $D ^ { v }$ be well-defined, and let $F : \mathcal { Z } ^ { E } $ R be any measurable function for which there exist constants $\boldsymbol { c } _ { v } \ ( \boldsymbol { v } = 1 , \ldots , E )$ such that

$$
\operatorname* { s u p } _ { D \in { \mathcal Z } ^ { E } , Z _ { v } ^ { \prime } \in { \mathcal Z } } | F ( D ) - F ( D ^ { v } ) | \leq c _ { v }
$$

then, for any $\varepsilon > 0$

$$
P _ { D } \left[ F ( D ) - \mathbb { E } _ { D } [ F ( D ) ] \ge \varepsilon \right] \le \exp \left\{ \frac { - 2 \varepsilon ^ { 2 } } { \sum _ { v \in [ E ] } c _ { v } } \right\} .
$$

Proof This is a classical concentration inequality, and the proof is omitted here. For more details, please refer to (McDiarmid et al., 1989). For clarity on how this lemma is applied in our case, we provide the following simple example. Consider that $\hat { R } ( h , D ) : \mathcal { H } \times \mathcal { Z } ^ { E }  \mathbb { R }$ . Then, we have

$$
\operatorname* { s u p } _ { D , Z _ { v } ^ { \prime } } \left| \hat { R } ( h , D ) - \hat { R } ( h , D ^ { v } ) \right| \leq c _ { v } .
$$

Namely,

$$
P \left[ \hat { R } ( h , D ) - R ( h , D ) \geq \varepsilon \right] \leq \exp \left\{ \frac { - 2 \varepsilon ^ { 2 } } { \sum _ { v \in [ E ] } c _ { v } } \right\} .
$$

Lemma 9 (Expectation Bound via JS divergence) Let P and Q be two probability distributions. Let $F : { \mathcal { Z } }  \mathbb { R }$ be a measurable function satisfying

$$
\| F \| _ { \infty } : = \operatorname* { s u p } _ { Z \in { \mathcal { Z } } } | F ( Z ) | \leq M \leq \infty .
$$

Then,

$$
| \mathbb { E } _ { P } [ F ] - \mathbb { E } _ { Q } [ F ] | \leq 2 M \sqrt { 2 J S ( P | | Q ) } ,
$$

or

$$
| \mathbb { E } _ { P } [ F ] - \mathbb { E } _ { Q } [ F ] | \leq 2 \sqrt { 2 } M d _ { J S } ( P , Q ) .
$$

Proof The result follows from standard inequalities in information theory. First, by a standard total variation bound,

$$
| \mathbb { E } _ { P } [ F ] - \mathbb { E } _ { Q } [ F ] | \leq 2 \left. F \right. _ { \infty } d _ { T V } ( P , Q ) .
$$

Applying the Pinsker’s inequality, we have

$$
\begin{array} { l } { \displaystyle \left( \frac 1 2 d _ { T V } ( P , Q ) \right) ^ { 2 } \le \frac 1 2 K L ( P | | M ) } \\ { \displaystyle \left( \frac 1 2 d _ { T V } ( P , Q ) \right) ^ { 2 } \le \frac 1 2 K L ( Q | | M ) . } \end{array}
$$

Here, $M = ( P + Q ) / 2 , d _ { T V } ( P , M ) = ( 1 / 2 ) d _ { T V } ( P , Q )$ , and $d _ { T V } ( Q , M ) = ( 1 / 2 ) d _ { T V } ( P , Q )$ Then, by the definition of the JS divergence, i.e., $J S ( P ; Q ) = 1 / 2 K L ( P ; M ) \ +$ $1 / 2 K L ( Q ; M )$ , we obtain

$$
2 \left( \frac { 1 } { 2 } d _ { T V } ( P , Q ) \right) ^ { 2 } \leq \frac { 1 } { 2 } J S ( P | | Q ) .
$$

Here, we clarify how this lemma can be used to derive some results in our paper. Let $Z = ( h ( X ) , Y )$ and consider ℓ as F. Then, assuming $\| \ell \| _ { \infty } \leq M$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { Z \sim P } [ | \ell ( Z ) | ] - \mathbb { E } _ { Z \sim Q } [ | \ell ( Z ) | ] \leq 2 M \sqrt { 2 J S ( P ^ { ( h ( X ) , Y ) } | | Q ^ { ( h ( X ) , Y ) } ) } , } \\ & { \qquad \leq 2 M \sqrt { 2 J S ( P ^ { ( X , Y ) } | | Q ^ { ( X , Y ) } ) } , } \end{array}
$$

considering $| a | - | b | \leq | a - b |$ . The last inequality is a consequence of the Data Processing Inequality from information theory. Note that h is the same for both P and Q. If this condition does not hold, the last inequality will not hold either. 7

Lemma 10 Let h be the hypothesis obtained by SRM where ℓ is a loss function associated with a convex cost function $c ( \cdot , \cdot )$ . We denote by $B ( \cdot )$ a positive non-decreasing real-valued function such that for all $y \in \mathcal { V }$ . Then, we have

$$
\forall y ^ { \prime } \in \mathcal { Y } , c ( y , y ^ { \prime } ) \leq B ( y ) .
$$

Moreover, ℓ is σ-admissible where σ can be taken as

$$
\sigma = \operatorname* { s u p } _ { y ^ { \prime } \in \mathcal { V } } \operatorname* { s u p } _ { | y | \leq B ( y ) } \left| \frac { \partial c } { \partial y } ( y , y ^ { \prime } ) \right| .
$$

Proof The proof can be found in (Bousquet and Elisseef, 2002), on page 515, referring to the proof of Lemma 23. Note that this lemma difers slightly from the original one, but the proof is identical.

In the original lemma, the authors derived the following bound for the loss function:

$$
\forall z \in \mathcal { Z } , 0 \leq \ell ( A _ { s } , Z ) \leq B \left( \kappa \sqrt { \frac { B ( 0 ) } { \lambda } } \right) .
$$

Here, $A _ { s }$ denotes the optimal hypothesis returned by the SRM algorithm. Consequently, the upper bound is established specifically.

In our work, the hypothesis h learned by SRM is not necessarily the minimizer, and the assumption $0 \leq c ( h ( X ) , Y ) \leq M$ is inherited from Theorem 3. Therefore, we do not strictly follow the original lemmas; instead, we adopt only the parts that are relevant to our analysis.

Specifically, the above result follows directly from the consideration of σ-admissibility.

## References

Kartik Ahuja, Karthikeyan Shanmugam, Kush Varshney, and Amit Dhurandhar. Invariant risk minimization games. In International Conference on Machine Learning, pages 145– 155. PMLR, 2020.

Martin Arjovsky, L´eon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from diferent domains. Machine learning, 79(1):151–175, 2010.

Olivier Bousquet and Andr´e Elisseef. Stability and generalization. Journal of machine learning research, 2(Mar):499–526, 2002.

Moulik Choraria, Ibtihal Ferwana, Ankur Mani, and Lav R. Varshney. Learning optimal features via partial invariance. Proceedings of the AAAI Conference on Artificial Intelligence, 37(6):7175–7183, Jun. 2023.

Rune Christiansen, Niklas Pfister, Martin Emil Jakobsen, Nicola Gnecco, and Jonas Peters. A causal framework for distribution generalization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6614–6630, 2021.

Tomer Galanti, Mengjia Xu, Liane Galanti, and Tomaso Poggio. Norm-based generalization bounds for sparse neural networks. Advances in Neural Information Processing Systems, 36:42482–42501, 2023.

Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization, 2020. URL https://arxiv.org/abs/2007.01434.

Tomer Levy and Felix Abramovich. Generalization error bounds for multiclass sparse linear classifiers. Journal of Machine Learning Research, 24(151):1–35, 2023.

Bo Li, Yifei Shen, Yezhen Wang, Wenzhen Zhu, Dongsheng Li, Kurt Keutzer, and Han Zhao. Invariant information bottleneck for domain generalization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 7399–7407, 2022.

Shaojie Li and Yong Liu. Concentration inequalities for general functions of heavy-tailed random variables. In Forty-first International Conference on Machine Learning, 2024.

Ya Li, Mingming Gong, Xinmei Tian, Tongliang Liu, and Dacheng Tao. Domain generalization via conditional invariant representations. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Pascal Massart and Elodie N´ed´elec. Risk bounds for statistical learning. <sup>´</sup> The Annals of Statistics, 34(5):2326–2366, 2006.

Colin McDiarmid et al. On the method of bounded diferences. Surveys in combinatorics, 141(1):148–188, 1989.

Krikamol Muandet, David Balduzzi, and Bernhard Sch¨olkopf. Domain generalization via invariant feature representation. In International conference on machine learning, pages 10–18. PMLR, 2013.

Emeson Pereira, Gustavo Carneiro, and Filipe R Cordeiro. A study on the impact of data augmentation for training convolutional neural networks in the presence of noisy labels. In 2022 35th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), volume 1, pages 25–30. IEEE, 2022.

Elan Rosenfeld, Pradeep Kumar Ravikumar, and Andrej Risteski. The risks of invariant risk minimization. In International Conference on Learning Representations, 2020.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. Ai models collapse when trained on recursively generated data. Nature, 631 (8022):755–759, 2024.

Peifeng Tong, Wu Su, He Li, Jialin Ding, Zhan Haoxiang, and Song Xi Chen. Distribution free domain generalization. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th

International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 34369–34378. PMLR, 23–29 Jul 2023.

Vladimir Vapnik. The nature of statistical learning theory. Springer science & business media, 2013.

Vladimir N Vapnik and A Ya Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. In Measures of complexity: festschrift for alexey chervonenkis, pages 11–30. Springer, 2015.

Haoxiang Wang, Haozhe Si, Bo Li, and Han Zhao. Provable domain generalization via invariant-feature subspace recovery. In International Conference on Machine Learning, pages 23018–23033. PMLR, 2022.

Xinyi Wang, Michael Saxon, Jiachen Li, Hongyang Zhang, Kun Zhang, and William Yang Wang. Causal balancing for domain generalization. In The Eleventh International Conference on Learning Representations, 2023.

Yimu Wang, Yihan Wu, and Hongyang Zhang. Lost domain generalization is a natural consequence of lack of training domains. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 15689–15697, 2024.

Danru Xu, Dingling Yao, Sebastien Lachapelle, Perouz Taslakian, Julius von K¨ugelgen, Francesco Locatello, and Sara Magliacane. A sparsity principle for partially observable causal representation learning. In Forty-first International Conference on Machine Learning. PMLR, 2024.

Hong Zheng and Fei Teng. Distribution shift is key to learning invariant prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 40(34):28812–28820, Mar. 2026.