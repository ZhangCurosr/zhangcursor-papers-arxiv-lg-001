# Asymmetric Likelihood-Free Hypothesis Testing

Gunjan Kumar IIT Kanpur gunjan@cse.iitk.ac.in

Yash Pote National University of Singapore yashppote@gmail.com

Jonathan Scarlett National University of Singapore scarlett@comp.nus.edu.sg

## Abstract

In likelihood-free hypothesis testing (LFHT), given samples from two unknown distributions $P , Q$ on [n] with $\operatorname { d _ { T V } } ( P , Q ) \geqslant \varepsilon .$ and from a third distribution $D \in \{ P , Q \}$ , the goal is to identify whether $D = P$ or $D = Q$ . We study asymmetric likelihood-free hypothesis testing, where the sample budgets $s _ { P }$ and $s Q$ for P and Q may difer, and characterize the feasible triples $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ up to logarithmic factors. We establish that unequal reference budgets can give a polynomial saving. For instance, for constant separation, n samples from P and $n ^ { 3 / 4 }$ from Q allow testing with ${ \widetilde { O } } ( n ^ { 1 / 4 } )$ samples from D, whereas both existing baselines—symmetric LFHT and unequal-sample closeness testing—require $\Omega ( { \sqrt { n } } )$ samples from D.

We further apply our results to non-adaptive distribution 2-clustering, where the sample allocation among k distributions must be fixed before observing any data. We obtain matching upper and lower bounds for this problem up to logarithmic factors in terms of the domain size, number of distributions, smaller cluster size, and separation, both with one cluster distribution known and with both unknown. Comparing with known bounds for adaptive sample allocation gives a worst-case adaptivity gap of $\widetilde { \Theta } ( k ^ { 1 / 4 } )$

## 1 Introduction

Likelihood-free inference arises when data can be simulated from a model, but its likelihood cannot be evaluated. In likelihood-free hypothesis testing (LFHT) (Gerber and Polyanskiy, 2024; Gerber et al., 2023), we observe samples from two unknown reference distributions $P , Q$ on $[ n ] = \{ 1 , \dots , n \}$ at total variation distance $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon$ , and from a target distribution $D \in \{ P , Q \}$ . The goal is to identify D with probability at least $2 / 3$ . We write $s _ { P } , s _ { Q } , s _ { D }$ for the three sample budgets.

While prior work gives tight bounds in the symmetric case $s _ { P } = s _ { Q }$ (Gerber and Polyanskiy, 2024; Gerber et al., 2023), the asymmetric case $s _ { P } \neq s _ { Q }$ is also of significant interest. For instance, an existing model may have many simulated samples available, while a new model has only a few. We ask how the extra samples from the first model reduce the number needed from the target distribution.

In this paper, we characterize the feasible triples $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ up to logarithmic factors. To exemplify the benefit of considering unequal samples, suppose that the separation ε is constant, and consider $s _ { P } = n$ and $s _ { Q } = n ^ { 3 / 4 }$ . Discarding excess P-samples to use symmetric LFHT requires $s _ { D } = \Omega ( { \sqrt { n } } )$ . Ignoring $Q$ and testing closeness between P and D also requires $s _ { D } = \Omega ( { \sqrt { n } } )$ . In contrast, using both reference batches, our bound gives $s _ { D } = \widetilde { \Theta } ( n ^ { 1 / 4 } )$ (Section 3), thus providing a polynomial saving. The characterization also extends to more than two hypotheses (Appendix D). Here and below, $\widetilde O$ and $\widetilde { \Theta }$ hide polylogarithmic factors in n and, for clustering, the number of distributions k.

## 1.1 Application to Non-Adaptive Clustering

In distribution 2-clustering, each of k input distributions equals one of two distributions $P , Q$ with $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant$ ε, and the goal is to recover their partition using as few total samples as possible. Prior work (Kumar et al., 2026) allows adaptive sampling, i.e., earlier sample outcomes can be used to decide which distribution to

sample from next. We determine the sample complexity under non-adaptive sampling, i.e., the allocation must be fixed before observing any samples. If the smaller cluster has known size r, and both cluster distributions are unknown, the optimal total budget is

$$
T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = \widetilde \Theta \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } + \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \right) .
$$

We also obtain matching upper and lower bounds up to logarithmic factors when one cluster distribution is known (Section 6). Comparing with the adaptive bounds of Kumar et al. (2026) gives a worst-case adaptivity gap of $\widetilde { \Theta } ( k ^ { 1 / 4 } )$ , resolving their open question. For example, when $n = k , r \asymp \sqrt { k }$ , and ε is constant, the adaptive and non-adaptive budgets are $\widetilde { \Theta } ( k )$ and $\widetilde { \Theta } ( k ^ { 5 / 4 } )$ , respectively.

The connection to LFHT comes from the observation that, once an exemplar from each cluster is identified, classifying the remaining distributions reduces to asymmetric LFHT, since the two exemplars may come with unequal sample budgets.

## 2 Preliminaries and Background

For the problem setup in Section 1, let $R _ { \mathsf { L F } } ( \varepsilon )$ be the set of nonnegative integer triples $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ for which a test using these sample budgets has error at most $1 / 3$ for every pair $P , Q$ with $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon$ and either choice $D \in \{ P , Q \}$ . Assume without loss of generality that $s _ { P } \geqslant s _ { Q }$ , and set

$$
\alpha : = \operatorname* { m a x } \left\{ 1 , { \frac { n } { \operatorname* { m a x } \{ s _ { P } , s _ { D } \} } } \right\} .\tag{1}
$$

The factor α equals one once either P or D has at least n samples.

For a distribution P on [n], let $\begin{array} { r } { \| P \| _ { j } ^ { j } = \sum _ { i } P ( i ) ^ { j } } \end{array}$ and $\| P \| _ { \infty } = \operatorname* { m a x } _ { i } P ( i )$ . For distributions $P , Q ,$ , let $\begin{array} { r } { \chi ^ { 2 } ( P \| Q ) = \sum _ { x } \frac { ( P ( x ) - Q ( x ) ) ^ { 2 } } { Q ( x ) } } \end{array}$ . We will use the inequality $\begin{array} { r } { \mathrm { d } _ { \mathrm { T V } } ( P , Q ) \ \leqslant \ \frac { 1 } { 2 } \sqrt { \chi ^ { 2 } ( P \| Q ) } } \end{array}$ , which follows from Cauchy–Schwarz. We write $\mathrm { P o i } ( \lambda )$ for the Poisson distribution with mean $\lambda \geqslant 0$ and $\mathrm { p o i } _ { \lambda } ( m ) = e ^ { - \lambda } \lambda ^ { m } / m !$ for its probability at a nonnegative integer m.

The notation $\asymp _ { \log n }$ hides constant and polylogarithmic factors in n in the sample budgets, and the symbols $\lesssim , \gtrsim$ and ≍ hide constant factors that are independent of all problem parameters. The hidden terms are often required to be suficiently large without explicitly stating so.

Symmetric LFHT. Let $R _ { \mathsf { L F } } ^ { \mathrm { s y m } } ( \varepsilon )$ be the restriction of $R _ { \mathsf { L F } } ( \varepsilon )$ to $s _ { P } = s _ { Q }$ . Gerber and Polyanskiy (2024, Theorem 2) show that

$$
R _ { \mathsf { L F } } ^ { \mathrm { s y m } } ( \varepsilon ) \asymp _ { \mathsf { l o g } n } \left\{ \left( s _ { P } , s _ { D } \right) : s _ { D } \geqslant \frac { 1 } { \varepsilon ^ { 2 } } , \quad s _ { P } \geqslant \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } , \quad s _ { P } s _ { D } \geqslant \frac { n \alpha } { \varepsilon ^ { 4 } } \right\} .\tag{2}
$$

The three conditions roughly correspond to (i) enough target samples for testing even if P and $Q$ were known, (ii) enough reference samples such that P and Q are not too poorly sampled, and (iii) suficient “joint information” through the product of sample counts.

## 2.1 Related Work

Distribution testing. Distribution testing refers to the task of testing whether a distribution has a given property using samples (Canonne, 2022). In identity testing, one must distinguish between $\mathrm { d } _ { \mathrm { T V } } ( P , Q ) = 0$ and $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon$ for a known distribution P and an unknown distribution Q. Uniformity testing is a special case where P is uniform. If both distributions are unknown, the problem is called closeness testing (Batu et al., 2000; Chan et al., 2014). Our test uses ingredients similar to those of these closeness testers, such as estimators of squared $\ell _ { 2 }$ distances and sample-based flattening of Diakonikolas and Kane (2016).

LFHT. In information theory, classification with empirically observed statistics is a generalization of the Neyman–Pearson framework to situations where the candidate hypotheses are completely unknown but we instead have a set of training samples from them. This line of work was initiated by Ziv (1988) and Gutman (1989). Their work considered a fixed alphabet and studied the error exponents in terms of the size of the training and test sample sets. Kelly et al. (2013) and Huang and Meyn (2012) let the alphabet grow with the sample size, under equal training lengths and the assumption that no symbol has probability much larger than under the uniform distribution, and ask when consistent classification is possible at all. Gerber and Polyanskiy (2024) revisited the problem under the name likelihood-free hypothesis testing (LFHT), and shifted the focus from asymptotic exponents to finite-sample guarantees. For a fixed error probability and equal reference budgets, they characterize the sample complexity in terms of the domain size and the TV separation, for several classes of distributions including the discrete distributions considered here. Gerber et al. (2023) show that tests based on classification achieve near-optimal sample complexity, including the dependence on the error probability, and give the lower-bound construction that we adapt in Section 5. These works already trade of the target budget $s _ { D }$ against a common reference budget. We determine the tradeof when the two reference budgets $s _ { P }$ and $s _ { Q }$ themselves difer.

Clustering of distributions. Kumar et al. (2026) study the clustering problem from Section 1.1 with adaptive strategies and obtain matching upper and lower bounds up to logarithmic factors. Their algorithms first find an exemplar from each cluster and then classify the remaining distributions. In the non-adaptive setting, the allocation must be fixed before observing any samples, so the budgets for classification cannot depend on which exemplars are found. Kumar et al. (2026) ask whether this constraint makes non-adaptive sampling require strictly more samples. We answer this positively with a worst-case gap of $\widetilde { \Theta } ( k ^ { 1 / 4 } )$ where k is the number of distributions. Clustering from bandit feedback (with adaptive sampling) was also studied by Yavas et al. (2025), but it is less directly related since it focuses on the sample complexity as the error probability tends to zero with the instance fixed, leaving the dependence on $n , k ,$ and r in lower-order terms.

## 3 Main Result

Our main result characterizes the feasible sample budgets for asymmetric LFHT; see also Appendix D for the extension to more than two hypotheses.

Theorem 1. For n $\geqslant 1 0$ and $\varepsilon \in ( 0 , 1 / 8 ]$ , under the convention $s _ { P } \geqslant s _ { Q }$ , we have

$$
R _ { \mathsf { L F } } ( \varepsilon ) \asymp _ { \mathsf { l o g } n } \left\{ \left( s _ { P } , s _ { Q } , s _ { D } \right) : s _ { D } \geqslant \frac { 1 } { \varepsilon ^ { 2 } } , \quad s _ { P } \geqslant \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } , \quad s _ { D } \mathopen { } \mathclose \bgroup \left( s _ { Q } + s _ { D } \aftergroup \egroup \right) \geqslant \frac { n \alpha } { \varepsilon ^ { 4 } } \right\} ,\tag{3}
$$

where α is defined in (1).

The three conditions share a similar interpretation to the symmetric case. The first condition is necessary even when both reference distributions are known. The second establishes that we require suficiently many samples from at least one reference distribution. The third allows additional D-samples to compensate for fewer Q-samples.

The feasible region is a set in three dimensions. Figure 1 shows four slices of it at fixed values of $s _ { P }$ , with ε constant. The lower boundary of each slice has up to three parts. On the horizontal part, the required $s _ { D }$ does not depend on $s _ { Q }$ . On the diagonal part, $s _ { Q } s _ { D } \asymp n \alpha$ , so additional Q-samples lower the required $s _ { D }$ The vertical part comes from the convention $s _ { P } \geqslant s _ { Q } . \mathrm { ~ A s ~ } s _ { P }$ grows, the lower boundary moves down until $s _ { P } = n ,$ and stays the same for larger $s _ { P }$ , so additional P-samples no longer help. The diagonal part appears only when $s _ { P } > n ^ { 2 / 3 }$ . For smaller $s _ { P }$ , the horizontal part already lies at $s _ { D } \geqslant s _ { P }$ , and additional Q-samples do not reduce the required $s _ { D }$

The dot in Figure 1 marks the example from the introduction, with ε constant, $s _ { P } = n$ , and $s _ { Q } = n ^ { 3 / 4 }$ Here $\alpha = 1$ , and the third condition $s _ { D } ( n ^ { 3 / 4 } + s _ { D } ) \stackrel { > } { _ \sim } n$ gives $s _ { D } = \widetilde { \Theta } ( n ^ { 1 / 4 } )$ . Both baselines need $s _ { D } \gtrsim \sqrt { n }$ shown by the open circle.

![](images/d400439b4182783ff354e52042e1be6bf6c12983a72bc526d5d29f3cbb1a768e.jpg)  
Figure 1: Slices of the feasible region of Theorem 1 at four values of $s _ { P }$ , with $\varepsilon$ constant. Constant and logarithmic factors are suppressed, and both axes are logarithmic.

## 3.1 Proof Ideas

We summarize some of the main ideas here, and defer the details to Sections 4 and $5 .$

Our upper bound adapts the quadratic statistic used for symmetric LFHT (Gerber and Polyanskiy, 2024). First, the distributions $P , Q$ are transformed into flattened distributions $\tilde { p } , \tilde { q }$ on a common expanded domain, preserving their TV distance while ensuring $\| \tilde { p } \| _ { \infty } = O ( ( \log n ) / s _ { P } )$ and $\| \tilde { q } \| _ { \infty } = O ( ( \log n ) / s _ { Q } )$ with high probability. After flattening, our algorithm estimates the diference between the squared $\ell _ { 2 }$ distances from $\tilde { d }$ to $\tilde { p }$ and ${ \tilde { q } } .$ Having more samples from P helps control not only the variance terms involving $P ,$ but also those involving Q. Specifically, the inequality

$$
\| \tilde { q } \| _ { 2 } ^ { 2 } \leqslant 2 \| \tilde { p } \| _ { 2 } ^ { 2 } + 2 \| \tilde { p } - \tilde { q } \| _ { 2 } ^ { 2 }
$$

allows us to control the terms involving $Q$ in the variance of our test statistic using the better-sampled reference $P$ and the squared separation between P and $Q .$ . Although larger separation may increase the variance, it also provides a distinguishing signal, which grows faster than the standard deviation.

The lower bound adapts the heavy/light construction of Gerber et al. (2023). On light pairs of domain elements, P has equal $O ( 1 / n )$ mass while $Q$ has a random hidden deviation from uniformity. A single sample from such a pair is uninformative for distinguishing these two cases, so collisions are required in the sense of having either two D-samples or one D-sample and one Q-sample from the same pair. The collision requirements lead to an $s _ { D } ( s _ { Q } + s _ { D } )$ factor in the lower bound. On heavy pairs, all of $P , Q , D$ coincide and have equal $O ( 1 / s _ { P } )$ mass; these pairs hide which collisions are informative, yielding an additional dependence on $s _ { P }$ when it is below the domain size n.

## 4 Upper Bound

In this section, we prove the upper bound in Theorem 1. Recall the assumption $s _ { P } \geqslant s _ { Q }$ , and let $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ satisfy the three conditions in (3), allowing polylogarithmic factors in n as indicated. We construct a test that distinguishes $D = P$ from $D = Q$ with error probability at most $\textstyle { \frac { 1 } { 3 } }$

We specifically consider a triplet on the right-hand side of (3) and multiply all three budgets by C log n for suficiently large C, then relabel the resulting budgets as $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ . This can only decrease $\alpha ,$ and the condition on $s _ { D } ( s _ { Q } + s _ { D } )$ in (3) gains an additional factor of $( \log n ) ^ { 2 }$ , as required by the variance analysis in Section 4.1.2. We then proceed via three cases:

1. If $\begin{array} { r } { s _ { D } \gtrsim \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } } \end{array}$ , run the closeness tester of Diakonikolas and Kane (2016) on P and $D .$

2. Otherwise, if $s _ { Q } > n$ , run symmetric LFHT using $s _ { Q }$ samples from each reference distribution.

3. Otherwise, run Algorithm 1 below with the number of P-samples truncated to n (discarding any beyond that if $s _ { P } > n )$

We proceed to show that the test succeeds in each case.

Case 1: In this case, $\begin{array} { r } { s _ { D } \gtrsim \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } . \mathrm { B y ~ ( 3 ) } } \end{array}$ , the same lower bound holds for $s _ { P }$ . We apply the closeness tester of Diakonikolas and Kane (2016, Proposition 2.11) to P and D. This tester succeeds provided

$$
\operatorname* { m i n } ( s _ { P } , s _ { D } ) \gtrsim \frac { \sqrt { n } } { \varepsilon ^ { 2 } } , \qquad \mathrm { a n d } \qquad \operatorname* { m i n } ( s _ { P } , s _ { D } ) ^ { 2 } \operatorname* { m a x } ( s _ { P } , s _ { D } ) \gtrsim \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } .\tag{4}
$$

The first condition follows immediately from (3) and $\alpha \geqslant 1$ . For the second condition, the case condition and (3) give min $\begin{array} { r } { ( s _ { P } , s _ { D } ) ^ { 2 } \gtrsim \frac { n \alpha } { \varepsilon ^ { 4 } } } \end{array}$ . Multiplying by max $\left( s _ { P } , s _ { D } \right)$ and using α max $( s _ { P } , s _ { D } ) \geqslant n$ , we obtain

$$
\operatorname* { m i n } ( s _ { P } , s _ { D } ) ^ { 2 } \operatorname* { m a x } ( s _ { P } , s _ { D } ) \gtrsim \frac { n \alpha \operatorname* { m a x } ( s _ { P } , s _ { D } ) } { \varepsilon ^ { 4 } } \geqslant \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } .
$$

Hence, both conditions in (4) are satisfied.

Case 2: In this case we have $s _ { P } \geqslant s _ { Q } > n .$ , and hence $\alpha = 1$ . We use $s _ { Q }$ samples from each reference distribution and apply the symmetric LFHT guarantee (2). Since the reference budget exceeds n, this test succeeds provided $\begin{array} { r } { s _ { D } \gtrsim \frac { 1 } { \varepsilon ^ { 2 } } , s _ { Q } \gtrsim \frac { \sqrt { n } } { \varepsilon ^ { 2 } } } \end{array}$ , and $\begin{array} { r } { s _ { Q } s _ { D } \gtrsim \frac { n } { \varepsilon ^ { 4 } } } \end{array}$ . The first condition is already part of (3). Moreover, since case 1 was already ruled out, we have $\begin{array} { r } { { s _ { D } ^ { 2 } } \lesssim \frac { \breve { n } } { \varepsilon ^ { 4 } } } \end{array}$ , whereas the strengthened condition on $s _ { D } ( s _ { Q } + s _ { D } )$ gives $s _ { D } ( s _ { Q } + s _ { D } ) \gtrsim ( \log n ) ^ { 2 } \frac { n } { \varepsilon ^ { 4 } }$ . Hence, the $s _ { D } ^ { 2 }$ contribution can be absorbed in the hidden constan ${ \mathrm { 5 } } ,$ and we obtain $s _ { Q } s _ { D } \gtrsim ( \log n ) ^ { 2 } \frac { n } { \varepsilon ^ { 4 } }$ . Dividing by $\begin{array} { r } { s _ { D } \lesssim \frac { \sqrt { n } } { \varepsilon ^ { 2 } } } \end{array}$ yields $s _ { Q } \gtrsim ( \log n ) ^ { 2 } \frac { \sqrt { n } } { \varepsilon ^ { 2 } }$ (thus implying the same for $s _ { P } )$ Hence, all the requirements of symmetric LFHT are met.

Case 3: In this case, we have $s _ { Q } \leqslant n ,$ and also $s _ { P } \leqslant n$ by the above discarding rule. Note that the value of α is unchanged under such discarding: if we originally had $s _ { P } \leqslant n$ then $\alpha \asymp n / s _ { P }$ , whereas if we originally had $s _ { P } > n$ , then $\alpha = 1$ , and after truncation $s _ { P } = n$ , so again $\alpha = n / s _ { P } = 1$ . The quantities $s _ { Q }$ and $s _ { D }$ , as well as the lower bound on $s Q \boldsymbol { s } _ { D }$ , are unafected.

Since case 1 was ruled out, we have $s _ { D } ^ { 2 } \lesssim \frac { n \alpha } { \varepsilon ^ { 4 } }$ . On the other hand, the above-mentioned strengthened condition on $s _ { D } ( s _ { Q } + s _ { D } )$ gives

$$
s _ { D } ( s _ { Q } + s _ { D } ) \stackrel { > } { _ \sim } ( \log n ) ^ { 2 } \frac { n \alpha } { \varepsilon ^ { 4 } } .
$$

Hence, the $s _ { D } ^ { 2 }$ contribution can be absorbed in the hidden constants, and we obtain $s _ { Q } s _ { D } \gtrsim ( \log n ) ^ { 2 } { \frac { n \alpha } { \varepsilon ^ { 4 } } }$ Again using $\begin{array} { r } { s _ { D } \lesssim \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } } \end{array}$ , it follows that

$$
s _ { Q } \gtrsim ( \log n ) ^ { 2 } \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } \gtrsim s _ { D } .
$$

In particular, since $s _ { P } \geqslant s _ { Q }$ , we have $s _ { P } \geqslant s _ { Q } \geqslant s _ { D }$

In summary, in case 3 we may assume the following conditions:

$$
n \geqslant s _ { P } \geqslant s _ { Q } \geqslant s _ { D } , \qquad \alpha \asymp \frac { n } { s _ { P } } , \qquad s _ { Q } s _ { D } \gtrsim ( \log n ) ^ { 2 } \frac { n \alpha } { \varepsilon ^ { 4 } } .\tag{5}
$$

It remains to specify the test (Algorithm 1 below) and show that it succeeds whenever (5) holds.

## 4.1 Our Algorithm

Our algorithm is described in Algorithm 1. As stated, it uses a random number of samples, but we later convert it to fixed sample budgets with only a constant-factor increase in sample complexity. It first uses independent samples from P and Q to flatten all three distributions on a common expanded domain, by splitting each element into several copies and distributing its probability mass uniformly among them (see Section 4.1.1). We let $\tilde { p } , \tilde { q } , \tilde { d }$ denote the flattened distributions. We will see that flattening preserves total variation distances while controlling their largest probabilities.

The main quantity underlying our test is $\lVert \tilde { { \boldsymbol { p } } } - \tilde { d } \rVert _ { 2 } ^ { 2 } - \lVert \tilde { { \boldsymbol { q } } } - \tilde { d } \rVert _ { 2 } ^ { 2 }$ . Its sign identifies $D \colon$ if $D = Q$ , the quantity equals $\| \tilde { p } - \tilde { q } \| _ { 2 } ^ { 2 } > 0$ , whereas if $D = P ,$ , it equals $- \| \widetilde { p } - \widetilde { q } \| _ { 2 } ^ { 2 } < 0$ . Our algorithm constructs an unbiased estimator of this quantity. Expanding the squares, the $| | \tilde { d } | | _ { 2 } ^ { 2 }$ terms cancel, leaving

$$
\| \tilde { p } \| _ { 2 } ^ { 2 } - \| \tilde { q } \| _ { 2 } ^ { 2 } - 2 \sum _ { j } \tilde { p } _ { j } \tilde { d } _ { j } + 2 \sum _ { j } \tilde { q } _ { j } \tilde { d } _ { j } .
$$

The four terms comprising $T _ { \mathsf { L F } }$ in Algorithm 1 are unbiased estimators of $\begin{array} { r l } { \| \tilde { p } \| _ { 2 } ^ { 2 } , \| \tilde { q } \| _ { 2 } ^ { 2 } , \sum _ { j } \tilde { p } _ { j } \tilde { d } _ { j } } & { { } } \end{array}$ , and $\begin{array} { r } { \sum _ { j } \tilde { q } _ { j } \tilde { d } _ { j } , } \end{array}$ respectively. Thus, $T _ { \mathsf { L F } }$ is an unbiased estimator of the desired quantity, and its sign determines the output of the test.

Algorithm 1: Asymmetric LFHT in the regime n $\geqslant s _ { P } \geqslant s _ { Q } \gtrsim s _ { D }$   
Input: Sample access to ${ \overline { { P , Q , D } } }$ with $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon$ and $D \in \{ P , Q \}$ , budgets $s _ { P } , s _ { Q } , s _ { D } , s _ { P } , s _ { Q } , s _ { P } , s _ { P } , s _ { Q } , s _ { P } , s _ { P } ,$   
Output: $D = P$ or $D = Q .$   
1 Draw $\lceil s _ { P } / 2 \rceil$ samples from $P$ and $\lceil s _ { Q } / 2 \rceil$ from $Q$ for flattening.   
2 Construct the common split domain $[ K ]$ as in Section 4.1.1.   
3 Independently draw testing batches of sizes $\operatorname { P o i } ( s _ { P } )$ $\operatorname { P o i } ( s _ { Q } )$ , and $\mathrm { P o i } ( s _ { D } )$ from $P , Q , D ,$ and map   
their samples to $[ K ]$ as in Section 4.1.1. The mapped batches are samples from $\tilde { p } , \tilde { q } , \tilde { d } .$   
4 Let $N _ { j } ^ { P } , N _ { j } ^ { Q } , N _ { j } ^ { D }$ be the numbers of mapped testing samples taking value $j \in [ K ]$   
5 $\begin{array} { r } { T _ { \mathsf { L F } } \gets \sum _ { j = 1 } ^ { K } \Big [ \frac { \check { N _ { j } ^ { P } } ( N _ { j } ^ { P } - 1 ) } { s _ { P } ^ { 2 } } - \frac { N _ { j } ^ { Q } ( N _ { j } ^ { Q } - 1 ) } { s _ { Q } ^ { 2 } } - 2 \frac { N _ { j } ^ { P } N _ { j } ^ { D } } { s _ { P } s _ { D } } + 2 \frac { N _ { j } ^ { Q } N _ { j } ^ { D } } { s _ { Q } s _ { D } } \Big ] } \end{array}$   
6 if $T _ { \mathsf { L F } } < 0$ then   
7 return $D = P$   
8 else   
9 return $D = Q$

## 4.1.1 Flattening

We use the sample-based flattening of Diakonikolas and Kane (2016, Definition 2.4) (also used by Gerber and Polyanskiy (2024)). Let S be the multiset of the $\lceil s _ { P } / 2 \rceil$ and $\lceil s _ { Q } / 2 \rceil$ flattening samples. For each $i \in [ n ]$ , let $B _ { i }$ be one plus its number of occurrences in $s ,$ and split i into $B _ { i }$ elements. Since $s _ { Q } \leqslant s _ { P } \leqslant n ,$ the expanded domain has size

$$
K = \sum _ { i = 1 } ^ { n } B _ { i } = n + \left\lceil s _ { P } / 2 \right\rceil + \left\lceil s _ { Q } / 2 \right\rceil \leqslant 2 n + 2 .
$$

Identify $[ K ]$ with $\{ ( i , a ) : i \in [ n ] , a \in [ B _ { i } ] \}$ . For any distribution r on $[ n ] _ { \cdot }$ , its flattened version is $\begin{array} { r } { \tilde { r } ( i , a ) = \frac { r ( i ) } { B _ { i } } } \end{array}$ To sample from ${ \tilde { r } } ,$ take a sample i from r and independently choose a uniformly from $[ B _ { i } ]$ . We apply this same transformation to all three testing batches, so $\tilde { d } \in \{ \tilde { p } , \tilde { q } \}$ . By Diakonikolas and Kane (2016, Fact $2 . 5 ( \mathrm { i i } ) )$ splitting preserves total variation, so $\mathrm { d _ { T V } } ( \tilde { p } , \tilde { q } ) = \mathrm { d _ { T V } } ( \tilde { P } , Q ) \geqslant \varepsilon .$ . Moreover, we have the following $\ell _ { \infty } { \mathrm { - n o r m } }$ guarantees.

Lemma 2. With probability at least $1 - 1 / n$ over ${ \mathbf { } } S ,$

$$
\| \tilde { p } \| _ { \infty } \leqslant \frac { 3 2 \log n } { s _ { P } } , \qquad \| \tilde { q } \| _ { \infty } \leqslant \frac { 3 2 \log n } { s _ { Q } } .
$$

The proof is given in Appendix $\mathrm { A } .$

## 4.1.2 Expectation, Variance, and Correctness

Fix any realization of the flattening samples S for which the bounds in Lemma 2 hold. This also determines $\tilde { p } , \tilde { q } , \tilde { d } .$ . The testing batches are then independent samples from these three distributions. Let $\Delta ^ { 2 } = \lVert \tilde { p } - \tilde { q } \rVert _ { 2 } ^ { 2 }$ The vector $\tilde { p } - \tilde { q }$ has size $K \leqslant 2 n + 2 \leqslant 3 n$ , so Cauchy–Schwarz gives $\lVert \tilde { p } - \tilde { q } \rVert _ { 1 } \leqslant \sqrt { K } \lVert \tilde { p } - \tilde { q } \rVert _ { 2 }$ . Since $\lVert \tilde { p } - \tilde { q } \rVert _ { 1 } = 2 \mathrm { d } _ { \mathrm { T V } } ( \tilde { p } , \tilde { q } ) \geqslant 2 \varepsilon$ , it follows that

$$
\Delta ^ { 2 } \geqslant \frac { \| \tilde { p } - \tilde { q } \| _ { 1 } ^ { 2 } } { K } \geqslant \frac { 4 \varepsilon ^ { 2 } } { 3 n } .\tag{6}
$$

The following lemma states the expectation and variance needed for Chebyshev’s inequality, and is proved in Appendix A.3.

Lemma 3. Suppose $s _ { P } \geqslant s _ { Q } \geqslant s _ { D }$ . Conditional on any $s$ satisfying Lemma ${ \mathcal { Q } } ,$ the statistic in Algorithm 1 satisfies

$$
\begin{array} { r l } & { \mathbb { E } [ T _ { \mathsf { L F } } ] = \left\{ \begin{array} { l l } { - \Delta ^ { 2 } } & { i f D = P , } \\ { \Delta ^ { 2 } } & { i f D = Q , } \end{array} \right. } \\ & { \mathrm { V a r } [ T _ { \mathsf { L F } } ] \lesssim ( \log n ) ^ { 2 } \left[ \frac { 1 } { s _ { P } s _ { Q } s _ { D } } + \frac { \Delta ^ { 2 } } { s _ { Q } s _ { D } } \right] . } \end{array}
$$

By Lemma $3 , T _ { \mathsf { L F } }$ has mean $- \Delta ^ { 2 }$ if $D = P$ and $\Delta ^ { 2 }$ if $D = Q .$ , so it sufices to show that its variance is small compared with $\Delta ^ { 4 }$ . To do so, we write the condition on $s _ { D } ( s _ { Q } + s _ { D } )$ in (5) as

$$
s _ { Q } s _ { D } \geqslant C ( \log n ) ^ { 2 } \frac { n \alpha } { \varepsilon ^ { 4 } } \stackrel { > } { \ \sim } C ( \log n ) ^ { 2 } \frac { n ^ { 2 } } { s _ { P } \varepsilon ^ { 4 } } ,
$$

where $C$ is a suficiently large constant. Substituting into Lemma $^ { 3 , }$ we have

$$
\begin{array} { r l r } {  { \operatorname { V a r } [ T _ { \mathrm { L F } } ] \lesssim \frac { 1 } { C } [ \frac { \varepsilon ^ { 4 } } { n ^ { 2 } } + \frac { s _ { P } \varepsilon ^ { 4 } \Delta ^ { 2 } } { n ^ { 2 } } ] } } \\ & { } & { \leqslant \frac { 1 } { C } [ \frac { \varepsilon ^ { 4 } } { n ^ { 2 } } + \frac { \varepsilon ^ { 2 } \Delta ^ { 2 } } { n } ] } \\ & { } & { \lesssim \frac { \Delta ^ { 4 } } { C } } \end{array}
$$

$$
\begin{array} { r l } & { ( \mathrm { u s i n g ~ } s _ { P } \leqslant n \mathrm { ~ a n d ~ } \varepsilon \leqslant 1 ) } \\ & { } \\ & { ( \mathrm { u s i n g ~ } ( 6 ) ) . } \end{array}
$$

We choose $C$ such that this is at most $\Delta ^ { 4 } / 6$ . Under either hypothesis, an incorrect answer requires a deviation of at least $\Delta ^ { 2 }$ from its mean, and hence

$$
\mathrm { P r } [ \mathrm { i n c o r r e c t ~ s i g n } \mid S ] \leqslant \frac { \mathrm { V a r } [ T _ { \mathsf { L F } } ] } { \Delta ^ { 4 } } \leqslant \frac { 1 } { 6 } .
$$

We proceed to summarize the other possible sources of error. The event of Lemma 2 fails with probability at most $1 / n \leqslant 1 / 1 0$ . To implement the Poissonized testing batches using fixed sample budgets, draw $2 s _ { P }$ $2 s _ { Q }$ , and $2 s _ { D }$ samples from $P , Q$ , and $D ,$ respectively, and use only the first $\operatorname { P o i } ( s _ { R } )$ samples from each source $R \in \{ P , Q , D \}$ . If any Poisson draw exceeds the corresponding available budget, an arbitrary output is produced and this is considered an error. Since $\operatorname* { P r } [ \operatorname* { P o i } ( s ) > 2 s ] \leqslant e ^ { - s / 3 }$ and every testing budget is $\gtrsim s _ { D } \gtrsim { \frac { 1 } { \varepsilon ^ { 2 } } }$ this happens with probability at most $\frac { 1 } { 3 0 }$ . The total error is therefore at most $1 / 6 + 1 / 1 0 + 1 / 3 0 = 3 / 1 0 < \breve { 1 / 3 }$ completing the upper bound.

## 5 Lower Bound

We prove that the three conditions in Theorem 1 are necessary up to constant factors for every $\varepsilon \in ( 0 , 1 / 8 ]$ Recall that $\begin{array} { r } { \alpha = \operatorname* { m a x } \Bigl ( 1 , \frac { n } { \operatorname* { m a x } ( s _ { P } , s _ { D } ) } \Bigr ) } \end{array}$ . When $s _ { Q } = 0$ , the problem is closeness testing between P and $D ,$ and the required lower bound follows from Bhattacharya and Valiant (2015). Henceforth, we assume that $s _ { Q } \geqslant 1$ (and thus $s _ { P } \geqslant 1$ under our convention $s _ { P } \geqslant s _ { Q } )$ . We also assume that $s _ { D } \geqslant 1$ , since a tester with no D-samples has the same output distribution under both hypotheses, and hence errs with probability at least $\frac { 1 } { 2 }$ under one of them.

The first two conditions follow by a reduction from the symmetric case, where the lower bounds of Gerber and Polyanskiy (2024, Propositions $1 3 \mathrm { - } 1 5 )$ hold up to constant factors. Suppose an asymmetric tester succeeds with sample sizes $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ , where $s _ { P } \geqslant s _ { Q }$ . Given $s _ { P }$ samples from each of $P$ and $Q ,$ and $s _ { D }$ samples from $D _ { \mathbf { \lambda } }$ , we can run this tester using all the $P -$ and D-samples and only the first $s _ { Q }$ of the $Q \cdot$ -samples. This gives a symmetric tester with budgets $\left( s _ { P } , s _ { P } , s _ { D } \right)$ and the same success probability. Since α depends only on $s _ { P }$ and $s _ { D } .$ , it is unchanged, so these lower bounds give $s _ { D } \gtrsim 1 / \varepsilon ^ { 2 }$ and $s _ { P } \gtrsim \sqrt { n \alpha } / \varepsilon ^ { 2 }$

We prove the third condition $\begin{array} { r } { s _ { D } ( s _ { Q } + s _ { D } ) \gtrsim \frac { n \alpha } { \varepsilon ^ { 4 } } } \end{array}$ in three cases:

1. Case $s _ { P } < s _ { D }$ . In this case the second condition gives $\begin{array} { r } { s _ { D } > s _ { P } \gtrsim \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } } \end{array}$ , which immediately yields $\begin{array} { r } { s _ { D } ( s _ { Q } + s _ { D } ) \geqslant s _ { D } ^ { 2 } \gtrsim \frac { n \alpha } { \varepsilon ^ { 4 } } } \end{array}$

2. Case $s _ { D } \leqslant s _ { P } , s _ { P } > \frac { n } { 1 0 0 }$ . In this case $\alpha \leqslant 1 0 0 .$ , so it sufices to show that $\begin{array} { r } { s _ { D } ( s _ { Q } + s _ { D } ) \gtrsim \frac { n } { \varepsilon ^ { 4 } } } \end{array}$

Give the tester $P$ explicitly and increase its Q-budget to $s _ { Q } + s _ { D }$ . Both changes make the problem easier. $\mathrm { A }$ successful asymmetric tester would therefore solve one-distribution-known LFHT with $s _ { Q } + s _ { D }$ samples from $Q$ and $s _ { D }$ from $D$ . Raising the Q-budget to $s _ { Q } + s _ { D }$ makes it at least the D-budget, which Kumar et al. (2026, Remark 8) requires. The lower bound of Kumar et al. (2026, Lemma $7$ and Remark 8), using the construction in Gerber and Polyanskiy (2024, Proposition 13), gives $\begin{array} { r } { s _ { D } ( s _ { Q } + s _ { D } ) \gtrsim \frac { n } { \varepsilon ^ { 4 } } . ^ { 1 } } \end{array}$

3. Case $\begin{array} { r } { s _ { D } \leqslant s _ { P } \leqslant \frac { n } { 1 0 0 } } \end{array}$ . Here $\begin{array} { r } { \alpha = \frac { n } { s _ { P } } } \end{array}$ . It sufices to show

$$
s _ { P } s _ { D } ( s _ { Q } + s _ { D } ) \gtrsim \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } ,\tag{7}
$$

which we do in the next section.

## 5.1 Proof for the Case $\begin{array} { r } { s _ { D } \leqslant s _ { P } \leqslant \frac { n } { 1 0 0 } } \end{array}$

Overview. To prove $( 7 )$ , we construct two priors $\Lambda _ { 0 }$ and $\Lambda _ { 1 }$ over asymmetric LFHT instances $( P , Q , D )$ with $D = P$ under $\Lambda _ { 0 }$ and $D = Q$ under $\Lambda _ { 1 }$ . With high probability, an instance drawn from either prior satisfies $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon$ . We show that, when $s _ { P } s _ { D } ( s _ { Q } + s _ { D } )$ is suficiently small, the sample distributions induced by $\Lambda _ { 0 }$ and $\Lambda _ { 1 }$ are close in total variation. On the other hand, any tester for $R _ { \mathsf { L F } } ( \varepsilon )$ must distinguish $D = P$ from $D = Q$ , and hence must distinguish the two induced sample distributions with high probability. The contradiction yields the desired lower bound.

We adapt the heavy/light construction of Gerber et al. (2023, Appendix D.4). In that construction, perturbations are independent across light elements and cancel only in expectation. To simplify the calculations, we use the paired perturbations of Paninski (2008). Within each light pair, the weight added to one element is removed from the other, so the weight vectors $w _ { P } , w _ { Q } , w _ { D }$ defined below have the same total weight in every realization. The domain is divided into independent pairs, each of which is either heavy or light. On a heavy pair, $P , Q .$ , and $D$ assign the same masses to the two elements, and hence samples from such a pair contain no information about whether $D = P \ { \mathrm { o r } } \ D = Q$ . Their role is instead to generate many uninformative within-pair collisions and thereby hide the informative collisions occurring on light pairs. On a light pair, $P$ is uniform, whereas $Q$ is tilted according to a hidden random sign $\eta \in \{ \pm 1 \}$ , assigning masses proportional to $( 1 + 8 \varepsilon \eta , 1 - 8 \varepsilon \eta )$ . Since $\eta$ is chosen uniformly at random, after averaging over $\eta$ the distribution of a single sample from a light pair is the same under the two hypotheses. Thus, useful information is revealed only when we get at least two samples from the same light pair. The main informative signals are a $D – D$ collision, which reveals whether $D$ itself is tilted, and a $D { - } Q$ collision, which reveals whether $D$ and $Q$ share the same hidden tilt. $\textrm { A } D - P$ collision is uninformative because $P$ is uniform on every light pair. At a high level, the factor $s _ { D } ( s _ { Q } + s _ { D } )$ in the condition $s _ { D } ( s _ { Q } + s _ { D } ) \stackrel { > } { _ \sim } n ^ { 2 } / ( s _ { P } \varepsilon ^ { 4 } )$ captures the total $D { - } D$ and $D { - } Q$ collision signal, while the additional factor $s _ { P }$ reflects the masking efect created by the heavy pairs.

Construction. Let

$$
t _ { P } : = 1 0 0 s _ { P } , \qquad t _ { Q } : = 1 0 0 s _ { Q } , \qquad t _ { D } : = 1 0 0 s _ { D } ,
$$

and observe that $1 0 0 \leqslant \operatorname* { m i n } ( t _ { Q } , t _ { D } ) \leqslant \operatorname* { m a x } ( t _ { Q } , t _ { D } ) \leqslant t _ { P } \leqslant n .$

We construct two priors $\Lambda _ { 0 }$ and $\Lambda _ { 1 }$ on triples of probability distributions $( P , Q , D )$ . The construction first generates nonnegative weight vectors $w _ { P } , w _ { Q } , w _ { D }$ and then normalizes them.

Let $N : = \lfloor n / 2 \rfloor . { \mathrm { S p l i t ~ } } [ 2 N ] \subseteq [ n ]$ into the pairs $( 2 j - 1 , 2 j ) , j \in [ N ]$ . Each pair is independently declared heavy with probability $\begin{array} { r } { \gamma : = \frac { t _ { P } } { 4 n } } \end{array}$ and light otherwise. For every light pair, independently draw $\eta _ { j } \sim \mathrm { U n i f } \{ \pm 1 \}$ We assign the weights on each pair as follows:

<table><tr><td>Pair</td><td> $\begin{array} { r l } { ( w _ { P } ( 2 j - 1 ) , w _ { P } ( 2 j ) ) } & { { } ( w _ { Q } ( 2 j - 1 ) , w _ { Q } ( 2 j ) ) } \end{array}$ </td></tr><tr><td>Heavy</td><td> $\left( \frac { 1 } { t _ { P } } , \frac { 1 } { t _ { P } } \right)$  1 1 tp&#x27; tp</td></tr><tr><td>Light  $\textstyle \left( { \frac { 1 } { 2 n } } , { \frac { 1 } { 2 n } } \right)$ </td><td>1+8εηj 1-8εηj 2n 2n</td></tr></table>

Since $\varepsilon \leqslant { \frac { 1 } { 8 } }$ , these weights are nonnegative. Under $\Lambda _ { 0 } ,$ set $w _ { D } : = w _ { P }$ , whereas under $\Lambda _ { 1 }$ , set $w _ { D } : = w _ { Q }$ . The total weight of a pair is the same under $w _ { P } , w _ { Q } , w _ { D }$ . Consequently, if h denotes the number of heavy pairs, then all three weight vectors have the same total weight $\begin{array} { r } { M : = \frac { 2 h } { t _ { P } } + \frac { N - h } { n } } \end{array}$ . Normalizing these weight vectors gives us the distributions

$$
P : = { \frac { w _ { P } } { M } } , \qquad Q : = { \frac { w _ { Q } } { M } } , \qquad D : = { \frac { w _ { D } } { M } } ,
$$

and we have $D = P$ under $\Lambda _ { 0 }$ and $D = Q$ under $\Lambda _ { 1 }$ .

We define $\begin{array} { r } { E : = \left\{ h \leqslant \frac { N t _ { P } } { 2 n } \right\} } \end{array}$ , and show in the following lemma that this event has high probability and ensures valid instances.

Lemma 4. The event E satisfies $\textstyle \operatorname* { P r } [ E ^ { c } ] \leqslant { \frac { 1 } { 2 5 } }$ . Moreover, on $\begin{array} { r } { E , M \geqslant \frac { 1 } { 5 } } \end{array}$ and $\operatorname { d _ { T V } } ( P , Q ) \geqslant \varepsilon .$

In addition, we provide the following lemma stating that a successful tester can be transferred from fixed samples to Poissonized samples.

Lemma 5. Suppose there exists an algorithm A with sample sizes $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ satisfying $s _ { Q } ~ \geqslant ~ 1$ and $1 \leqslant s _ { D } \leqslant s _ { P } \leqslant \frac { n } { 1 0 0 }$ , and success probability at least $\textstyle { \frac { 2 } { 3 } }$ . Then there exists an algorithm which, when an instance $( P , Q , D )$ is generated according to either prior $\Lambda _ { \beta } , \breve { \beta } \in \{ 0 , 1 \}$ , and is given independent $\mathrm { P o i } ( M t _ { P } ) , \mathrm { P o i } ( M t _ { Q } )$ and $\mathrm { P o i } ( M t _ { D } )$ samples from $P , \ Q$ , and $D _ { \mathrm { { ; } } }$ , respectively, determines whether $D = P$ or $D = Q$ with error probability less than $\frac { 2 } { 5 }$

The proofs of Lemma 4 and Lemma 5 are deferred to Appendix B.

Let $X _ { i } , Y _ { i } ,$ , and $Z _ { i }$ denote the number of times element $i \in [ n ]$ is observed among the $\mathrm { P o i } ( M t _ { P } ) , \mathrm { P o i } ( M t _ { Q } )$ and Po $( M t _ { D } )$ samples drawn from $P , Q ,$ and D, respectively. Without loss of generality, any testing algorithm may be assumed to depend on the samples only through the counts $( X _ { i } , Y _ { i } , Z _ { i } ) _ { i \in [ n ] }$ , since, conditional on these counts, the ordering of the samples is independent of the underlying distributions. For each $\beta \in \{ 0 , 1 \}$ let $\mathbb { P } _ { \beta }$ denote the distribution of $( X _ { i } , Y _ { i } , Z _ { i } ) _ { i \in [ n ] }$ when the instance is generated according to the prior $\Lambda _ { \beta }$

By the well-known Poisson thinning property, for any generated instance and every $i \in [ n ]$

$$
\begin{array} { r } { X _ { i } \sim \mathrm { P o i } ( M t _ { P } P ( i ) ) = \mathrm { P o i } ( t _ { P } w _ { P } ( i ) ) , } \\ { Y _ { i } \sim \mathrm { P o i } ( M t _ { Q } Q ( i ) ) = \mathrm { P o i } ( t _ { Q } w _ { Q } ( i ) ) , } \\ { Z _ { i } \sim \mathrm { P o i } ( M t _ { D } D ( i ) ) = \mathrm { P o i } ( t _ { D } w _ { D } ( i ) ) . } \end{array}
$$

Since the heavy/light choices and the signs are generated independently across the N pairs, the sample vectors associated with distinct pairs are independent under both priors $\Lambda _ { 0 }$ and $\Lambda _ { 1 }$

For each $j \in [ N ]$ , let $V _ { j } : = \left( X _ { 2 j - 1 } , X _ { 2 j } , Y _ { 2 j - 1 } , Y _ { 2 j } , Z _ { 2 j - 1 } , Z _ { 2 j } \right)$ be the counts associated with the pair $( 2 j - 1 , 2 j )$ . As noted above, the vectors $V _ { 1 } , \dots , V _ { N }$ are independent and identically distributed. We thus suppress $j$ in the notation and use $\mathbb { P } _ { \beta } ^ { \mathrm { p a i r } }$ , for $\beta \in \{ 0 , 1 \}$ , for the distribution of $V _ { j }$ under $\Lambda _ { \beta }$ . Hence $\mathbb { P } _ { \beta } =$ $\left( \mathbb { P } _ { \beta } ^ { \mathrm { p a i r } } \right) ^ { \otimes N }$ . The divergence associated with a single pair is bounded via the following lemma, proved in Appendix B.1.

Lemma 6. $\begin{array} { r } { \chi ^ { 2 } \Big ( \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } \Big \| \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } \Big ) \lesssim \frac { \varepsilon ^ { 4 } t _ { P } t _ { D } ( t _ { Q } + t _ { D } ) } { n ^ { 3 } } } \end{array}$

If an asymmetric LFHT tester succeeds, Lemma 5 gives a test $\psi$ distinguishing $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ with error less than $\frac { 2 } { 5 }$ under either hypothesis. By Le Cam’s testing inequality, its two error probabilities satisfy

$$
1 - \mathrm { d } _ { \mathrm { T V } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) \leqslant \operatorname* { P r } _ { \mathbb { P } _ { 0 } } [ \psi = 1 ] + \operatorname* { P r } _ { \mathbb { P } _ { 1 } } [ \psi = 0 ] < \frac { 4 } { 5 } ,
$$

so $\begin{array} { r } { \mathrm { { l } _ { T V } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) > \frac { 1 } { 5 } } \end{array}$ . We now show that this is impossible when the sample budget is too small.

For the N independent pairs, tensorization gives the identity

$$
1 + \chi ^ { 2 } ( \mathbb { P } _ { 0 } \lvert \lvert \mathbb { P } _ { 1 } ) = \left( 1 + \chi ^ { 2 } ( \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } \lvert \lvert \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } ) \right) ^ { N } .
$$

Thus,

$$
\begin{array} { r l r l } { \log \big ( 1 + \chi ^ { 2 } ( { \mathbb P } _ { 0 } \| { \mathbb P } _ { 1 } ) \big ) \leqslant N \chi ^ { 2 } ( { \mathbb P } _ { 0 } ^ { \mathrm { p a i r } } \| { \mathbb P } _ { 1 } ^ { \mathrm { p a i r } } ) } & { \quad } & & { \mathrm { ( t e n s o r i z a t i o n ~ a n d ~ } \log ( 1 + x ) \leqslant x ) } \\ { \lesssim \chi \frac { \varepsilon ^ { 4 } t _ { P } t _ { D } ( t _ { Q } + t _ { D } ) } { n ^ { 3 } } } & & { \quad } & & { \mathrm { ( L e m m a ~ } 6 \mathrm { ) } } \\ { \leqslant \frac { \varepsilon ^ { 4 } t _ { P } t _ { D } ( t _ { Q } + t _ { D } ) } { n ^ { 2 } } } & & { \quad } & & { \mathrm { ( } N \leqslant n \mathrm { ) } . } \end{array}
$$

Consequently, if $\begin{array} { r } { s _ { P } s _ { D } ( s _ { Q } + s _ { D } ) \leqslant \frac { c n ^ { 2 } } { \varepsilon ^ { 4 } } } \end{array}$ for a suficiently small constant $c ,$ then $\chi ^ { 2 } ( { \mathbb { P } } _ { 0 } \| { \mathbb { P } } _ { 1 } ) < \frac { 4 } { 2 5 }$ . Hence $\begin{array} { r } { \mathrm { d } _ { \mathrm { T V } } ( \mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } ) \leqslant \frac { 1 } { 2 } \sqrt { \chi ^ { 2 } ( \mathbb { P } _ { 0 } \| \mathbb { P } _ { 1 } ) } < \frac { 1 } { 5 } . } \end{array}$ , contradicting the bound above.

This proves $( 7 )$ , completing the case $\begin{array} { r } { s _ { D } \leqslant s _ { P } \leqslant \frac { n } { 1 0 0 } } \end{array}$ of the third condition, and with it the lower-bound direction of Theorem 1.

## 6 Application: Non-Adaptive Distribution 2-Clustering

We now apply the asymmetric bounds to distribution clustering. Throughout this section and Appendix C, we assume that $n \geqslant 1 0$ . The formal problem formulation is as follows:

The Non-Adaptive Distribution 2-Clustering Problem   
Input: Sample access to k distributions $\{ D _ { i } \} _ { i = 1 } ^ { k }$ over domain $[ n ]$ , parameter $\varepsilon \in ( 0 , 1 ]$ . The distributions   
are queried non-adaptively. The per-distribution sample allocation $( T _ { 1 } , \dots , T _ { k } )$ is chosen as a (possibly   
random) function of $n , k , \varepsilon ,$ and the minority cluster size $r$ alone, and all $T _ { i }$ samples are drawn from   
$D _ { i }$ and simultaneously revealed to the learner.   
Promise: Each input distribution equals either $P$ or $Q .$ , where $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon ,$ and both occur at least   
once among the k distributions (i.e., non-empty clusters).   
Goal: With probability at least ${ \frac { 2 } { 3 } } ,$ , recover the partition into the $P -$ and $Q \mathrm { . }$ -clusters.   
Variants: Either one cluster distribution is known, or both are unknown.

We write $P , Q$ for the two clusters’ distributions and $\{ \mathcal { T } _ { P } , \mathcal { T } _ { Q } \}$ for the corresponding partition. When one distribution is known, we take it to be $P .$ As noted in Kumar et al. (2026), it can then be taken to be the uniform distribution $U ,$ so the partition is $\{ \mathcal { T } _ { U } , \mathcal { T } _ { Q } \}$ .<sup>2</sup> Throughout, $r$ denotes the size of the smaller of the two clusters, so $r \in [ 1 , \frac { k } { 2 } ]$ , and we focus on the case that $r$ is known. We note that this knowledge is only used to fix the sample allocation, but not to classify the distributions once the samples are drawn. Moreover, instead of exact knowledge of $r ,$ any lower bound r $\leqslant r$ sufices, with the upper bounds then holding with $r _ { 0 }$ in place of r. Let $T$ denote the total sample budget, so that $\textstyle \sum _ { i } T _ { i } \leqslant T$ always, and $T _ { \mathrm { n a } } ( n , k , r , \varepsilon )$ is the minimum $T$ for which some such algorithm returns the correct partition with probability $\geqslant \ { \frac { 2 } { 3 } }$

<table><tr><td>Regime</td><td>One known distribution</td><td>Both distributions unknown</td></tr><tr><td rowspan="2"> $\begin{array} { r } { n \gtrsim \frac { k ^ { 2 } } { r \varepsilon ^ { 4 } } } \end{array}$ </td><td> $\widetilde { \Theta } \left( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } \right)$ </td><td> $\widetilde { \Theta } \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } \right)$ </td></tr><tr><td></td><td></td></tr><tr><td rowspan="2"> $\begin{array} { r } { r \log k \lesssim n \lesssim \frac { k ^ { 2 } } { r \varepsilon ^ { 4 } } } \end{array}$   $n \lesssim r \log k$ </td><td> $\widetilde { \Theta } \left( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } \right)$ </td><td> $\widetilde { \Theta } \left( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } \right)$ </td></tr><tr><td> $\widetilde { \Theta } \left( \frac { k } { \varepsilon ^ { 2 } } \right)$ </td><td> $\widetilde { \Theta } \left( \frac { k } { \varepsilon ^ { 2 } } \right)$ </td></tr></table>

Table 1: Non-adaptive sample complexity $T _ { \mathrm { n a } }$ for $k \geqslant 2 0 0$ and $\varepsilon \in ( 0 , 1 / 8 ]$ . The notation $\widetilde { \Theta }$ hides polylogarithmic factors. The bounds follow from Theorems 14, 16, 19 and 20.

Our results, summarized in Table 1, establish nearly tight upper and lower bounds on $T _ { \mathrm { n a } }$ . We outline the algorithm and its analysis here, and defer the details to Appendix C. As in the adaptive case of Kumar et al. (2026), the complexity of our algorithms is the sum of the costs of two stages (here “stages” only refer to algorithmic steps rather than adaptive sampling steps):

• Stage 1. Find one distribution from each cluster, called an exemplar.

• Stage 2. Classify every distribution via LFHT, using the two exemplars as the hypotheses.

Stage 1 uses a random subset of $\Theta { \left( \frac { k } { r } \right) }$ distributions, which contains a member of each cluster with constant probability. Because the allocation is fixed before the exemplars are identified, every distribution in this subset must receive enough samples to become an exemplar. When both distributions are unknown, one distribution of the subset, which we call the pivot, can receive more samples than the others. Stage 2 is then an instance of asymmetric LFHT, with the pivot as $P ,$ the other exemplar as $Q$ , and each distribution being classified as D. We proceed to outline the analysis.

One known distribution. A random subset of $\Theta \left( { \frac { k } { r } } \right)$ distributions contains at least one member of $\scriptstyle { \mathcal { T } } _ { Q }$ with constant probability. The algorithm finds an exemplar of $\scriptstyle { \mathcal { T } } _ { Q }$ by running a uniformity test on every member of the subset, then uses the exemplar’s samples as sample access to $Q$ to classify all k distributions against $( U , Q )$ with asymmetric LFHT at per-test error $\frac { 1 } { 9 k }$

For the lower bound, we provide a reduction in Lemma 17, used for both variants, that turns a clustering algorithm into a tester for $D = Q$ vs. $D = U$ . After sorting the distributions by expected allocation, an averaging argument finds a lightly sampled index $j$ such that the algorithm is unlikely to output the candidate partition formed by $j$ together with the $r - 1$ least-sampled indices. Planting $D$ at position $j ,$ , with these other $r - 1$ distributions simulating $Q ,$ , turns the clustering algorithm into a tester whose two sample sizes are an $O \left( { \frac { r } { k } } \right)$ - and an $O \big ( \frac { 1 } { k } \big )$ -fraction of the budget respectively, thus allowing us to invoke a known lower bound for testing.

By formalizing the above outlines, for $k \geqslant 2 0 0$ and $\varepsilon \in ( 0 , 1 / 8 ]$ , Theorems 14 and 19 give the following sample complexity with one known distribution:

$$
T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = \widetilde \Theta \left( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \right) .
$$

Both distributions unknown. The upper bound keeps the two-stage structure. Stage 1 draws a sketch from a pivot $i _ { A }$ and runs the unequal-samples closeness test of Diakonikolas and Kane (2016) against every other member of a random size-Θ  <sup>k</sup>  subset, which yields an exemplar $i _ { B }$ of the other cluster. Stage 2 classifies all k distributions against $( \dot { D } _ { i _ { A } } , D _ { i _ { B } } )$ by asymmetric LFHT.

The lower bound with one known distribution also applies when both are unknown, and this is combined with a separate argument specific to the both-unknown case. That argument again applies the reduction of Lemma 17, this time into a tester for $D = P$ vs. $D = Q$ . After ordering the indices by expected allocation, the reduction places $Q$ at the $r - 1$ least-sampled indices, the planted distribution D at another lightly sampled index $j ,$ and $P$ at all remaining indices. With $T$ samples total, the number of queries is $O ( T r / k )$ to the Q-oracle, $O ( T / k )$ to the D-oracle, and (trivially) at most $T$ to the P-oracle. The three-way tradeof $\begin{array} { r } { s _ { P } s _ { D } ( s _ { Q } + s _ { D } ) \gtrsim \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } } \end{array}$ then translates into $\begin{array} { r } { \frac { T ^ { 3 } r } { k ^ { 2 } } \gtrsim \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } } \end{array}$ , thus giving an $\Omega \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } \right)$ lower bound.

We note that the three-way tradeof is essential. In particular, planting a pair from the unequal-samples closeness-testing lower bound of Bhattacharya and Valiant (2015) yields only $\Omega \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 2 / 3 } \varepsilon ^ { 4 / 3 } } \right)$ , which is worse by a factor of $r ^ { 1 / 3 }$

By formalizing the above outlines, under the same assumptions as the one-known setting, Theorems 16 and 20 give the following characterization with both distributions unknown:

$$
T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = \widetilde \Theta \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } + \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \right) .
$$

Observe that at $r = 1$ the first term is $\frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { \varepsilon ^ { 4 / 3 } }$ , and at $\begin{array} { r } { r = \frac { k } { 2 } } \end{array}$ it is the LFHT-derived rate $\frac { k ^ { 1 / 3 } n ^ { 2 / 3 } } { \varepsilon ^ { 4 / 3 } }$ of Kumar et al. (2026).

Adaptivity gap. Let $T _ { \mathrm { a } }$ denote the adaptive sample complexity when both cluster distributions are unknown and r is known. Comparing our bounds with those of Kumar et al. (2026) gives $T _ { \mathrm { n a } } / T _ { \mathrm { a } } = \widetilde { O } ( k ^ { 1 / 4 } )$ throughout the above parameter range. As a specific example, the upper limit is attained (up to logarithmic factors) when $n = k , r \asymp \sqrt { k } .$ , and ε is constant:

$$
T _ { \mathrm { a } } = \widetilde \Theta ( k ) , \qquad T _ { \mathrm { n a } } = \widetilde \Theta ( k ^ { 5 / 4 } ) .
$$

The details are given in Appendix C.4.

## 7 Conclusion

In this paper, we characterized the sample budgets $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ for which asymmetric likelihood-free hypothesis testing is possible, up to logarithmic factors. From this characterization, we were able to derive tight sample complexity bounds for non-adaptive distribution 2-clustering. Among other things, we demonstrated that allowing unequal samples in $( s _ { P } , s _ { Q } )$ can polynomially reduce the required $s _ { D } ;$ for instance, with n samples from one reference distribution and $n ^ { 3 / 4 }$ from the other, ${ \widetilde { \cal O } } ( n ^ { 1 / 4 } )$ target samples sufice, whereas prior work required $\Omega ( { \sqrt { n } } )$ . Moreover, using asymmetric LFHT as a stepping stone, we established the optimal sample complexity of non-adaptive 2-clustering up to logarithmic factors, revealing an adaptivity gap of up to $\widetilde { O } ( k ^ { 1 / 4 } )$

## Acknowledgements

This work is supported by the National Research Foundation (NRF) Singapore and the Ministry of Digital Development and Information (MDDI) under the AI Visiting Professorship (Award AIVP-2024-003).

## AI Disclosure

The lower bound in Section 5 was developed with substantial help from an AI model (Claude). We reworked the argument ourselves to make it readable and to give intuition. The model also helped us check the proofs in most other parts of the paper. We take full responsibility for the correctness, presentation, and communication of the results.

## References

Tuğkan Batu, Lance Fortnow, Ronitt Rubinfeld, Warren D. Smith, and Patrick White. Testing that distributions are close. In Symposium on Foundations of Computer Science (FOCS). IEEE, 2000. ↑ 2

Bhaswar Bhattacharya and Gregory Valiant. Testing closeness with unequal sized samples. In Advances in Neural Information Processing Systems, 2015. ↑ 7, 12

Clément L. Canonne. Topics and techniques in distribution testing: A biased but representative sample. Foundations and Trends in Communications and Information Theory, 19(6):1032–1198, 2022. ↑ 2

Siu-On Chan, Ilias Diakonikolas, Gregory Valiant, and Paul Valiant. Optimal algorithms for testing closeness of discrete distributions. In ACM-SIAM Symposium on Discrete Algorithms (SODA). SIAM, 2014. ↑ 2, 23

Ilias Diakonikolas and Daniel M. Kane. A new approach for testing properties of discrete distributions. In Symposium on Foundations of Computer Science (FOCS), 2016. Full version available as arXiv:1601.05557. ↑ 2, 5, 6, 11, 22, 23

Patrik R. Gerber, Yanjun Han, and Yury Polyanskiy. Minimax optimal testing by classification. In Conference on Learning Theory (COLT). PMLR, 2023. ↑ 1, 3, 4, 8

Patrik Róbert Gerber and Yury Polyanskiy. Likelihood-free hypothesis testing. IEEE Transactions on Information Theory, 70(11):7971–8000, 2024. ↑ 1, 2, 3, 4, 6, 8, 14, 27

Oded Goldreich. The uniform distribution is complete with respect to testing identity to a fixed distribution. In Computational Complexity and Property Testing: On the Interplay Between Randomness and Computation. Springer International Publishing, 2020. ↑ 10

Michael Gutman. Asymptotically optimal classification for multiple tests with empirically observed statistics. IEEE Transactions on Information Theory, 35(2):401–408, 1989. ↑ 3

Dayu Huang and Sean Meyn. Classification with high-dimensional sparse samples. In IEEE International Symposium on Information Theory, 2012. doi: 10.1109/ISIT.2012.6283985. ↑ 3

Benjamin G. Kelly, Aaron B. Wagner, Thitidej Tularak, and Pramod Viswanath. Classification of homogeneous data with large alphabets. IEEE Transactions on Information Theory, 59(2):782–795, 2013. ↑ 3

Gunjan Kumar, Yash Pote, and Jonathan Scarlett. A distribution testing approach to clustering distributions. In Conference on Learning Theory. PMLR, 2026. ↑ 1, 2, 3, 8, 10, 11, 12, 22, 25, 26, 28

Liam Paninski. A coincidence-based test for uniformity given very sparsely sampled discrete data. IEEE Transactions on Information Theory, 54(10):4750–4755, 2008. ↑ 8, 22

Recep Can Yavas, Yuqi Huang, Vincent Y. F. Tan, and Jonathan Scarlett. A general framework for clustering and distribution matching with bandit feedback. IEEE Transactions on Information Theory, 2025. ↑ 3

Jacob Ziv. On classification with empirically observed statistics and universal data compression. IEEE Transactions on Information Theory, 34(2):278–286, 1988. ↑ 3

## A Proofs for the Upper Bound

## A.1 Flattening

We restate and prove the flattening lemma from Section 4.1.1.

Lemma 2. With probability at least $1 - 1 / n$ over $S _ { i }$ ,

$$
\| \tilde { p } \| _ { \infty } \leqslant \frac { 3 2 \log n } { s _ { P } } , \qquad \| \tilde { q } \| _ { \infty } \leqslant \frac { 3 2 \log n } { s _ { Q } } .
$$

Proof. We first prove the bound for P. Let $N ( i )$ count the copies of i in $s ,$ so $B _ { i } = 1 + N ( i )$ . The count $N ( i )$ is a sum of independent indicators from the two flattening batches, so

$$
\mathbb { E } [ N ( i ) ] = \lceil s _ { P } / 2 \rceil p ( i ) + \lceil s _ { Q } / 2 \rceil q ( i ) \geqslant s _ { P } p ( i ) / 2 .
$$

Each of the $B _ { i }$ pieces created from i has mass $p ( i ) / B _ { i }$ , so we bound this ratio for every $i .$

If $s _ { P } p ( i ) < 3 2 \log n ,$ , the original mass is already small enough:

$$
\frac { p ( i ) } { B _ { i } } \leqslant p ( i ) < \frac { 3 2 \log n } { s _ { P } } .
$$

Otherwise, we use concentration to ensure that i is split into enough pieces. Since $\mathbb { E } [ N ( i ) ] \geqslant { s _ { P } p ( i ) / 2 } \geqslant$ 16 log n, the Chernof bound gives

$$
\operatorname* { P r } \bigg [ N ( i ) < \frac { s _ { P } p ( i ) } { 8 } \bigg ] \leqslant \exp \left( - \frac { 9 s _ { P } p ( i ) } { 6 4 } \right) \leqslant n ^ { - 4 } .
$$

Whenever $N ( i ) \geqslant s _ { P } p ( i ) / 8$ , the mass of each piece satisfies

$$
\frac { p ( i ) } { B _ { i } } \leqslant \frac { 8 } { s _ { P } } \leqslant \frac { 3 2 \log n } { s _ { P } } .
$$

The same argument applies to $Q . \mathrm { ~ A ~ }$ union bound over the n elements for both distributions gives total failure probability at most $2 n \cdot n ^ { - 4 } \leqslant 1 / n$ , proving the lemma. □

## A.2 Poisson Moments and Variance

This subsection proves a general variance bound for T<sub>LF</sub>. We adapt the quadratic-statistic analysis of Gerber and Polyanskiy (2024, Proposition 5 and Appendix D) to independent Poisson batches with unequal sizes.

Lemma 7. Let $N \sim \mathrm { P o i } ( \lambda )$ and, where stated, $N ^ { \prime } \sim \mathrm { P o i } ( \mu )$ independent of N. Then

$$
\mathbb { E } [ N ( N - 1 ) ] \ = \ \lambda ^ { 2 } ,\tag{8}
$$

$$
\mathrm { V a r } [ N ( N - 1 ) ] ~ = ~ 4 \lambda ^ { 3 } + 2 \lambda ^ { 2 } ,\tag{9}
$$

$$
\mathbb { E } [ N N ^ { \prime } ] = \lambda \mu ,\tag{10}
$$

$$
\mathrm { V a r } [ N N ^ { \prime } ] = \lambda ^ { 2 } \mu + \lambda \mu ^ { 2 } + \lambda \mu .\tag{11}
$$

Proof. For every $m \geqslant 1$ , a direct calculation gives

$$
\mathbb { E } [ N ( N - 1 ) \cdot \cdot \cdot ( N - m + 1 ) ] = \sum _ { k \geqslant m } { \frac { k ! } { ( k - m ) ! } } \cdot { \frac { e ^ { - \lambda } \lambda ^ { k } } { k ! } } = \lambda ^ { m } \sum _ { k \geqslant m } { \frac { e ^ { - \lambda } \lambda ^ { k - m } } { ( k - m ) ! } } = \lambda ^ { m } .\tag{12}
$$

We then establish the four claims as follows:

• For (8), we simply apply (12) with $m = 2$

• For (9), expanding the right-hand side shows that

$$
x ^ { 2 } ( x - 1 ) ^ { 2 } = x ( x - 1 ) ( x - 2 ) ( x - 3 ) + 4 x ( x - 1 ) ( x - 2 ) + 2 x ( x - 1 ) .
$$

Applying (12) with $m = 4 , 3 , 2$ to the three terms gives $\mathbb { E } \left[ ( N ( N - 1 ) ) ^ { 2 } \right] = \lambda ^ { 4 } + 4 \lambda ^ { 3 } + 2 \lambda ^ { 2 }$ . Subtracting $( \mathbb { E } [ N ( N - 1 ) ] ) ^ { 2 } = \lambda ^ { 4 }$ gives (9).

• For (10), independence gives $\mathbb { E } [ N N ^ { \prime } ] = \mathbb { E } [ N ] \mathbb { E } [ N ^ { \prime } ] = \lambda \mu$

• For (11), independence gives

$$
\mathrm { V a r } [ N N ^ { \prime } ] = \mathbb { E } [ N ^ { 2 } ] \mathbb { E } [ N ^ { \prime 2 } ] - ( \mathbb { E } [ N ] \mathbb { E } [ N ^ { \prime } ] ) ^ { 2 } = ( \lambda ^ { 2 } + \lambda ) ( \mu ^ { 2 } + \mu ) - \lambda ^ { 2 } \mu ^ { 2 } ,
$$

where we applied $\mathbb { E } [ N ^ { 2 } ] = \mathbb { E } [ N ( N - 1 ) ] + \mathbb { E } [ N ] = \lambda ^ { 2 } + \lambda$ . Expanding and simplifying gives the desired expression $\lambda ^ { 2 } \mu + \lambda \dot { \mu } ^ { 2 } + \lambda \mu .$

All expectations and variances below are conditional on the flattening samples ${ \mathcal { S } } .$

Proposition 8. Let the counts

$$
N _ { i } ^ { P } \sim \mathrm { P o i } ( s _ { P } \tilde { p } _ { i } ) , \quad N _ { i } ^ { Q } \sim \mathrm { P o i } ( s _ { Q } \tilde { q } _ { i } ) , \quad N _ { i } ^ { D } \sim \mathrm { P o i } ( s _ { D } \tilde { d } _ { i } )
$$

be independent across $i \in [ K ]$ and across $P , Q , D _ { \colon }$ , and let $T _ { \mathsf { L F } }$ be the statistic of Algorithm 1. Then

$$
\mathbb { E } [ T _ { \mathsf { L F } } ] = \Vert \tilde { p } - \tilde { d } \Vert _ { 2 } ^ { 2 } - \Vert \tilde { q } - \tilde { d } \Vert _ { 2 } ^ { 2 }
$$

and

$$
\begin{array} { r l } & { \mathrm { V a r } [ T _ { \mathrm { L F } } ] \leqslant C \left[ \frac { \| \tilde { p } \| _ { 3 } ^ { 3 } + \sum _ { i } \tilde { p } _ { i } \tilde { d } _ { i } ^ { 2 } } { s _ { P } } + \frac { \| \tilde { q } \| _ { 3 } ^ { 3 } + \sum _ { i } \tilde { q } _ { i } \tilde { d } _ { i } ^ { 2 } } { s _ { Q } } + \frac { \sum _ { i } \tilde { p } _ { i } ^ { 2 } \tilde { d } _ { i } + \sum _ { i } \tilde { q } _ { i } ^ { 2 } \tilde { d } _ { i } } { s _ { D } } \right. } \\ & { \qquad \left. + \frac { \| \tilde { p } \| _ { 2 } ^ { 2 } } { s _ { P } ^ { 2 } } + \frac { \| \tilde { q } \| _ { 2 } ^ { 2 } } { s _ { Q } ^ { 2 } } + \frac { \sum _ { i } \tilde { p } _ { i } \tilde { d } _ { i } } { s _ { P } s _ { D } } + \frac { \sum _ { i } \tilde { q } _ { i } \tilde { d } _ { i } } { s _ { Q } s _ { D } } \right] . } \end{array}\tag{13}
$$

Proof. Throughout the proof, we shorten $\textstyle \sum _ { i \in [ K ] }$ to $\textstyle \sum _ { i }$

Expectation. For the statistic in Algorithm 1, (8) gives $\begin{array} { r } { \frac { \mathbb { E } [ N _ { i } ^ { P } ( N _ { i } ^ { P } - 1 ) ] } { s _ { P } ^ { 2 } } = \tilde { p } _ { i } ^ { 2 } } \end{array}$ and similarly $\tilde { q } _ { i } ^ { 2 }$ for the Q-term. By independence, $\begin{array} { r } { \frac { \mathbb { E } [ N _ { i } ^ { P } N _ { i } ^ { D } ] } { s _ { P } s _ { D } } = \tilde { p } _ { i } \tilde { d } _ { i } } \end{array}$ and $\begin{array} { r } { \frac { \mathbb { E } [ N _ { i } ^ { Q } N _ { i } ^ { D } ] } { s _ { Q } s _ { D } } = \tilde { q } _ { i } \tilde { d } _ { i } } \end{array}$ . Summing over i gives

$$
\begin{array} { r l r } {  { \mathbb { E } [ T _ { \mathsf { L F } } ] = \sum _ { i } \big ( \tilde { p } _ { i } ^ { 2 } - \tilde { q } _ { i } ^ { 2 } - 2 \tilde { p } _ { i } \tilde { d } _ { i } + 2 \tilde { q } _ { i } \tilde { d } _ { i } \big ) } } \\ & { } & \\ & { } & { = \| \tilde { p } - \tilde { d } \| _ { 2 } ^ { 2 } - \| \tilde { q } - \tilde { d } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Variance. Counts for distinct i values are independent, so the variance of each of the four components below is the sum of individual variances across i. We bound $\mathrm { V a r } [ T _ { \mathsf { L F } } ]$ using the decomposition

$$
{ \cal T } _ { \mathsf { L F } } = { \cal T } ^ { ( P ) } - { \cal T } ^ { ( Q ) } - 2 { \cal T } ^ { ( P D ) } + 2 { \cal T } ^ { ( Q D ) } ,\tag{14}
$$

where

$$
T ^ { ( P ) } : = \sum _ { i } \frac { N _ { i } ^ { P } ( N _ { i } ^ { P } - 1 ) } { s _ { P } ^ { 2 } } ,
$$

$$
T ^ { ( Q ) } : = \sum _ { i } \frac { N _ { i } ^ { Q } ( N _ { i } ^ { Q } - 1 ) } { s _ { Q } ^ { 2 } } ,
$$

$$
T ^ { ( P D ) } : = \sum _ { i } \frac { N _ { i } ^ { P } N _ { i } ^ { D } } { s _ { P } s _ { D } } ,
$$

$$
T ^ { ( Q D ) } : = \sum _ { i } \frac { N _ { i } ^ { Q } N _ { i } ^ { D } } { s _ { Q } s _ { D } } .
$$

Applying the inequality $\begin{array} { r } { \mathrm { V a r } [ X _ { 1 } + X _ { 2 } + X _ { 3 } + X _ { 4 } ] \leqslant 4 \sum _ { i = 1 } ^ { 4 } \mathrm { V a r } [ X _ { j } ] } \end{array}$ to the four signed terms in (14) gives

$$
\mathrm { V a r } [ T _ { \mathsf { L F } } ] \lesssim \mathrm { V a r } [ T ^ { ( P ) } ] + \mathrm { V a r } [ T ^ { ( Q ) } ] + \mathrm { V a r } [ T ^ { ( P D ) } ] + \mathrm { V a r } [ T ^ { ( Q D ) } ] .\tag{15}
$$

Variances of $T ^ { ( P ) }$ and $T ^ { ( Q ) }$ . Applying (9) with $\lambda = s _ { P } \tilde { p } _ { i }$ gives

$$
\mathrm { V a r } [ T ^ { ( P ) } ] = \frac { 1 } { s _ { P } ^ { 4 } } \sum _ { i } \left( 4 ( s _ { P } \tilde { p } _ { i } ) ^ { 3 } + 2 ( s _ { P } \tilde { p } _ { i } ) ^ { 2 } \right) = \frac { 4 \| \tilde { p } \| _ { 3 } ^ { 3 } } { s _ { P } } + \frac { 2 \| \tilde { p } \| _ { 2 } ^ { 2 } } { s _ { P } ^ { 2 } } .
$$

An identical argument for Q gives $\begin{array} { r } { \mathrm { V a r } [ T ^ { ( Q ) } ] = \frac { 4 \| \tilde { q } \| _ { 3 } ^ { 3 } } { s _ { Q } } + \frac { 2 \| \tilde { q } \| _ { 2 } ^ { 2 } } { s _ { Q } ^ { 2 } } } \end{array}$

Variances of $T ^ { ( P D ) }$ and $T ^ { ( Q D ) }$ . Applying (11) with $\lambda = s _ { P } \tilde { p } _ { i }$ and $\mu = s _ { D } \tilde { d } _ { i }$ gives

$$
\begin{array} { r l } & { \mathrm { V a r } [ T ^ { ( P D ) } ] = \displaystyle \frac { 1 } { s _ { P } ^ { 2 } s _ { D } ^ { 2 } } \sum _ { i } \left( ( s _ { P } \tilde { p } _ { i } ) ^ { 2 } ( s _ { D } \tilde { d } _ { i } ) + ( s _ { P } \tilde { p } _ { i } ) ( s _ { D } \tilde { d } _ { i } ) ^ { 2 } + ( s _ { P } \tilde { p } _ { i } ) ( s _ { D } \tilde { d } _ { i } ) \right) } \\ & { \quad \quad \quad = \displaystyle \frac { \sum _ { i } \tilde { p } _ { i } ^ { 2 } \tilde { d } _ { i } } { s _ { D } } + \frac { \sum _ { i } \tilde { p } _ { i } \tilde { d } _ { i } ^ { 2 } } { s _ { P } } + \frac { \sum _ { i } \tilde { p } _ { i } \tilde { d } _ { i } } { s _ { P } s _ { D } } , } \end{array}
$$

and analogously $\begin{array} { r } { \mathrm { V a r } [ T ^ { ( Q D ) } ] = \frac { \sum _ { i } \tilde { q } _ { i } ^ { 2 } \tilde { d } _ { i } } { s _ { D } } + \frac { \sum _ { i } \tilde { q } _ { i } \tilde { d } _ { i } ^ { 2 } } { s _ { Q } } + \frac { \sum _ { i } \tilde { q } _ { i } \tilde { d } _ { i } } { s _ { Q } s _ { D } } } \end{array}$

Substituting these four variance expressions into (15) and absorbing the numerical coeficients into C gives (13). □

## A.3 Norm Bounds

We now combine Proposition 8 with Lemma 2 to prove Lemma 3, which is restated as follows.

Lemma 3. Suppose $s _ { P } \geqslant s _ { Q } \geqslant s _ { D }$ . Conditional on any $s$ satisfying Lemma 2, the statistic in Algorithm 1 satisfies

$$
\mathbb { E } [ T _ { \mathsf { L F } } ] = \left\{ \begin{array} { l l } { - \Delta ^ { 2 } } & { i f D = P , } \\ { \Delta ^ { 2 } } & { i f D = Q , } \end{array} \right.
$$

$$
\mathrm { V a r } [ T _ { \mathsf { L F } } ] \lesssim ( \log n ) ^ { 2 } \left[ \frac { 1 } { s _ { P } s _ { Q } s _ { D } } + \frac { \Delta ^ { 2 } } { s _ { Q } s _ { D } } \right] .
$$

Proof. By Proposition $^ { 8 , }$ the expectation is $- \Delta ^ { 2 }$ when $D = P$ and $\Delta ^ { 2 }$ when $D = Q$ . It remains to bound the variance. On the event of Lemma 2, we claim that

$$
\| \tilde { p } \| _ { \infty } \lesssim \frac { \log n } { s _ { P } } , \qquad \| \tilde { q } \| _ { \infty } , \| \tilde { d } \| _ { \infty } \lesssim \frac { \log n } { s _ { Q } } , \qquad \| \tilde { p } \| _ { 2 } ^ { 2 } \lesssim \frac { \log n } { s _ { P } } , \qquad \| \tilde { q } \| _ { 2 } ^ { 2 } , \| \tilde { d } \| _ { 2 } ^ { 2 } \lesssim \frac { \log n } { s _ { P } } + \Delta ^ { 2 } .\tag{16}
$$

The bounds on $\| \widetilde { p } \| _ { \infty }$ and $\| { \tilde { q } } \| _ { \infty }$ are from Lemma 2, and the bound on $\| \tilde { p } \| _ { 2 } ^ { 2 }$ follows from $\begin{array} { r } { \| \tilde { p } \| _ { 2 } ^ { 2 } \leqslant \| \tilde { p } \| _ { \infty } \sum _ { i } \tilde { p } _ { i } = } \end{array}$ $\| \widetilde { p } \| _ { \infty }$ . For ${ \tilde { q } } ,$ the triangle inequality and $( a + b ) ^ { 2 } \leqslant 2 a ^ { 2 } + 2 b ^ { 2 }$ give

$$
\| \tilde { q } \| _ { 2 } ^ { 2 } = \| \tilde { p } + ( \tilde { q } - \tilde { p } ) \| _ { 2 } ^ { 2 } \leqslant 2 \| \tilde { p } \| _ { 2 } ^ { 2 } + 2 \Delta ^ { 2 } .
$$

Since $\tilde { d } \in \{ \tilde { p } , \tilde { q } \}$ and $s _ { Q } \leqslant s _ { P } ,$ , the bounds on $\tilde { d }$ follow from those on $\tilde { p }$ and ${ \tilde { q } } .$

We bound each term of the variance bound (13) of Proposition 8 using (16) and the following elementary inequalities:

(N1) $\| \tilde { r } \| _ { 3 } ^ { 3 } \leqslant \| \tilde { r } \| _ { \infty } \| \tilde { r } \| _ { 2 } ^ { 2 }$

(N2) $\begin{array} { r } { \sum _ { i } u _ { i } v _ { i } ^ { 2 } \leqslant \| u \| _ { \infty } \| v \| _ { 2 } ^ { 2 } } \end{array}$ and $\begin{array} { r } { \sum _ { i } u _ { i } ^ { 2 } v _ { i } \leqslant \| v \| _ { \infty } \| u \| _ { 2 } ^ { 2 } } \end{array}$

(N3) $\begin{array} { r } { \sum _ { i } \tilde { p } _ { i } \tilde { d } _ { i } \leqslant \| \tilde { p } \| _ { \infty } \sum _ { i } \tilde { d } _ { i } = \| \tilde { p } \| _ { \infty } } \end{array}$

(N4) $\begin{array} { r } { \sum _ { i } \tilde { q } _ { i } \tilde { d } _ { i } \leqslant \| \tilde { q } \| _ { 2 } \| \tilde { d } \| _ { 2 } \leqslant \frac { 1 } { 2 } \big ( \| \tilde { q } \| _ { 2 } ^ { 2 } + \| \tilde { d } \| _ { 2 } ^ { 2 } \big ) } \end{array}$

Term by term, we have the following:

$$
\frac { \| \tilde { p } \| _ { 3 } ^ { 3 } + \sum _ { i } \tilde { p } _ { i } \tilde { d } _ { i } ^ { 2 } } { s _ { P } } \leqslant \frac { \| \tilde { p } \| _ { \infty } \bigl ( \| \tilde { p } \| _ { 2 } ^ { 2 } + \| \tilde { d } \| _ { 2 } ^ { 2 } \bigr ) } { s _ { P } } \lesssim \frac { ( \log n ) ^ { 2 } } { s _ { P } ^ { 3 } } + \frac { ( \log n ) \Delta ^ { 2 } } { s _ { P } ^ { 2 } }\tag{by (N1), (N2),}
$$

$$
\frac { \| \tilde { q } \| _ { 3 } ^ { 3 } + \sum _ { i } \tilde { q } _ { i } \tilde { d } _ { i } ^ { 2 } } { s _ { Q } } \leqslant \frac { \| \tilde { q } \| _ { \infty } \left( \| \tilde { q } \| _ { 2 } ^ { 2 } + \| \tilde { d } \| _ { 2 } ^ { 2 } \right) } { s _ { Q } } \lesssim \frac { ( \log n ) ^ { 2 } } { s _ { P } s _ { Q } ^ { 2 } } + \frac { ( \log n ) \Delta ^ { 2 } } { s _ { Q } ^ { 2 } }\tag{by (N1), (N2),}
$$

$$
\frac { \sum _ { i } \tilde { p } _ { i } ^ { 2 } \tilde { d } _ { i } + \sum _ { i } \tilde { q } _ { i } ^ { 2 } \tilde { d } _ { i } } { s _ { D } } \leqslant \frac { \| \tilde { d } \| _ { \infty } \left( \| \tilde { p } \| _ { 2 } ^ { 2 } + \| \tilde { q } \| _ { 2 } ^ { 2 } \right) } { s _ { D } } \lesssim \frac { ( \log n ) ^ { 2 } } { s _ { P } s _ { Q } s _ { D } } + \frac { ( \log n ) \Delta ^ { 2 } } { s _ { Q } s _ { D } }
$$

$$
\frac { \| \tilde { p } \| _ { 2 } ^ { 2 } } { s _ { P } ^ { 2 } } \lesssim \frac { \log n } { s _ { P } ^ { 3 } } , \qquad \frac { \| \tilde { q } \| _ { 2 } ^ { 2 } } { s _ { Q } ^ { 2 } } \lesssim \frac { \log n } { s _ { P } s _ { Q } ^ { 2 } } + \frac { \Delta ^ { 2 } } { s _ { Q } ^ { 2 } }\tag{by (N2),}
$$

$$
\frac { \sum _ { i } \tilde { p } _ { i } \tilde { d } _ { i } } { s _ { P } s _ { D } } \leqslant \frac { \| \tilde { p } \| _ { \infty } } { s _ { P } s _ { D } } \lesssim \frac { \log n } { s _ { P } ^ { 2 } s _ { D } } ,
$$

by (16),

$$
\frac { \sum _ { i } \tilde { q } _ { i } \tilde { d } _ { i } } { s _ { Q } s _ { D } } \leqslant \frac { \| \tilde { q } \| _ { 2 } ^ { 2 } + \| \tilde { d } \| _ { 2 } ^ { 2 } } { 2 s _ { Q } s _ { D } } \lesssim \frac { \log n } { s _ { P } s _ { Q } s _ { D } } + \frac { \Delta ^ { 2 } } { s _ { Q } s _ { D } }
$$

by (N3), (N4).

Since $s _ { P } \geqslant s _ { Q } \geqslant s _ { D }$ , the denominators satisfy $s _ { P } ^ { 3 } , \ s _ { P } s _ { Q } ^ { 2 } , \ s _ { P } ^ { 2 } s _ { D } \stackrel { > } { _ \sim } s _ { P } s _ { Q } s _ { D }$ and $s _ { P } ^ { 2 } \geqslant s _ { Q } ^ { 2 } \gtrsim s _ { Q } s _ { D }$ , so summing the bounds gives the lemma. □

## B Proofs for the Lower Bound

We restate and prove the lemmas deferred from Section 5.1, using the construction and notation from that section.

Lemma 4. The event E satisfies $\textstyle \operatorname* { P r } [ E ^ { c } ] \leqslant { \frac { 1 } { 2 5 } }$ . Moreover, on E, $\begin{array} { r } { M \geqslant \frac { 1 } { 5 } } \end{array}$ and $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon .$

Proof. The number of heavy pairs satisfies

$$
h \sim \mathrm { B i n } \bigg ( N , \frac { t _ { P } } { 4 n } \bigg ) , \qquad \mathbb { E } [ h ] = \frac { N t _ { P } } { 4 n } .
$$

The complement $E ^ { c }$ is the event $h > 2 \mathbb { E } [ h ]$ . By a Chernof bound,

$$
\mathrm { P r } [ E ^ { c } ] \leqslant \exp \left( - \frac { \mathbb { E } [ h ] } { 3 } \right) .
$$

Recalling that $1 0 0 \leqslant t _ { P } \leqslant n$ and $N = \lfloor n / 2 \rfloor$ , we have

$$
\mathbb { E } [ h ] = \frac { N t _ { P } } { 4 n } \geqslant \frac { 9 9 } { 8 } .
$$

Consequently,

$$
\operatorname* { P r } [ E ^ { c } ] \leqslant e ^ { - 3 3 / 8 } < \frac { 1 } { 2 5 } .
$$

Since $t _ { P } \leqslant n$ , the total mass satisfies $M \geqslant N / n \geqslant 1 / 5$ . On $E ,$ we have $2 h / t _ { P } \leqslant N / n$ and $h \leqslant N / 2$ , which gives

$$
M = \frac { 2 h } { t _ { P } } + \frac { N - h } { n } \leqslant \frac { N } { n } + \frac { N - h } { n } \leqslant \frac { 3 ( N - h ) } { n } .
$$

The weights coincide on heavy pairs, and each light pair contributes $8 \varepsilon / n$ to their $\ell _ { 1 }$ distance. Hence,

$$
\mathrm { d _ { T V } } ( P , Q ) = { \frac { 4 ( N - h ) \varepsilon } { n M } } \geqslant { \frac { 4 \varepsilon } { 3 } } \geqslant \varepsilon .
$$

Lemma 5. Suppose there exists an algorithm $\mathcal { A }$ with sample sizes $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ satisfying $s _ { Q } ~ \geqslant ~ 1$ and $1 \leqslant s _ { D } \leqslant s _ { P } \leqslant \frac { n } { 1 0 0 }$ , and success probability at least $\textstyle { \frac { 2 } { 3 } }$ . Then there exists an algorithm which, when an instance $( P , Q , D )$ is generated according to either prior $\Lambda _ { \beta } , \breve { \beta } \in \{ 0 , 1 \}$ , and is given independent $\mathrm { P o i } ( M t _ { P } )$ $\mathrm { P o i } ( M t _ { Q } )$ and $\mathrm { P o i } ( M t _ { D } )$ samples from $P , \ Q$ , and $D _ { \mathrm { { ; } } }$ , respectively, determines whether $D = P$ or $D = Q$ with error probability less than $\frac { 2 } { 5 }$

Proof. Let $K _ { P } , K _ { Q }$ , and $K _ { D }$ denote the numbers of available samples from $P , Q$ , and $D ,$ so that, conditional on the generated instance, $K _ { R } \sim \mathrm { P o i } ( M t _ { R } )$ independently for $R \in \{ P , Q , D \}$ . Let $F$ be the event that $K _ { R } \geqslant s _ { R }$ for every $R .$ The new algorithm $\boldsymbol { B }$ retains the first $s _ { R }$ samples from each $R$ and runs $\mathcal { A }$ on them if $F$ occurs, and outputs an arbitrary answer otherwise.

The construction gives $M \geqslant N / n \geqslant \frac { 1 } { 5 }$ , so $M t _ { R } \geqslant 2 0 s _ { R }$ for every $R .$ . The Poisson lower-tail bound $\operatorname* { P r } [ K \leqslant ( 1 - \delta ) \lambda ] \leqslant e ^ { - \delta ^ { 2 } \lambda / 2 }$ for $K \sim \mathrm { P o i } ( \lambda )$ , applied with $\begin{array} { r } { \delta = \frac { 1 9 } { 2 0 } } \end{array}$ , gives

$$
\mathrm { P r } [ K _ { R } < s _ { R } ] \leqslant \mathrm { P r } \left[ K _ { R } \leqslant \frac { M t _ { R } } { 2 0 } \right] \leqslant e ^ { - 9 s _ { R } } \leqslant e ^ { - 9 } ,
$$

so by a union bound $\operatorname* { P r } [ F ^ { c } ] \leqslant 3 e ^ { - 9 } < 1 0 ^ { - 3 }$

Conditional on a fixed generated instance and on $F ,$ the retained observations are exactly $s _ { P } , s _ { Q }$ , and $s _ { D }$ independent samples from $P , Q$ , and $D$ . On $E \cap F$ , the retained samples form a valid LFHT instance, so A fails with probability at most ${ \frac { 1 } { 3 } } .$ . Under either prior, we therefore have

$$
\mathrm { P r } [ \mathcal { B } \mathrm { \ f a i l s } ] \leqslant \frac { 1 } { 3 } + \mathrm { P r } [ E ^ { c } ] + \mathrm { P r } [ F ^ { c } ] \leqslant \frac { 1 } { 3 } + \frac { 1 } { 2 5 } + 1 0 ^ { - 3 } < \frac { 2 } { 5 } .
$$

## B.1 Proof of Lemma 6

We restate the lemma as follows, and then prove it via three supporting lemmas.

Lemma 6. $\begin{array} { r } { \chi ^ { 2 } \Big ( \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } \Big \| \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } \Big ) \lesssim \frac { \varepsilon ^ { 4 } t _ { P } t _ { D } ( t _ { Q } + t _ { D } ) } { n ^ { 3 } } } \end{array}$

Recall from Section 5.1 that 100 $\leqslant t _ { Q }$ , t<sub>D</sub> $\leqslant$ t<sub>P</sub> $\leqslant n$ . For $R \in \{ P , Q , D \}$ , define

$$
\lambda _ { R } : = \frac { t _ { R } } { 2 n } , \qquad \mu _ { R } : = \frac { t _ { R } } { t _ { P } } .
$$

Hence, a heavy element of weight $1 / t _ { P }$ has expected count $\mu _ { R }$ among the R-samples. For light elements, the weights $1 / ( 2 n )$ and $( 1 \pm 8 \varepsilon ) / ( 2 n )$ give expected counts $\lambda _ { R }$ and $\lambda _ { R } ( 1 \pm 8 \varepsilon )$ , respectively. In particular, $\mu _ { P } = 1$

Fix a pair $( 2 j - 1 , 2 j )$ of domain elements from the construction. Let $\boldsymbol { x } = ( x _ { 1 } , x _ { 2 } )$ record how often these two elements appear among the P-samples, and define $y = ( y _ { 1 } , y _ { 2 } )$ and $z = ( z _ { 1 } , z _ { 2 } )$ similarly for the $Q -$ - and D-samples. For $x , y , z \in \mathbb { N } ^ { 2 }$ , define

$$
\begin{array} { l } { { \displaystyle H ( x , y , z ) : = \prod _ { i = 1 } ^ { 2 } \mathrm { p o i } _ { 1 } ( x _ { i } ) \mathrm { p o i } _ { \mu _ { Q } } ( y _ { i } ) \mathrm { p o i } _ { \mu _ { D } } ( z _ { i } ) } , \ }  \\ { { \displaystyle L ( x , y , z ) : = \prod _ { i = 1 } ^ { 2 } \mathrm { p o i } _ { \lambda _ { P } } ( x _ { i } ) \mathrm { p o i } _ { \lambda _ { Q } } ( y _ { i } ) \mathrm { p o i } _ { \lambda _ { D } } ( z _ { i } ) . } } \end{array}
$$

Observe that $H ( x , y , z )$ equals the probability of observing these counts for a heavy pair, and the function L gives the corresponding probability when $w _ { P } , w _ { Q } , w _ { D }$ all assign weight $1 / ( 2 n )$ to each element of the pair.

On a light pair, w instead assigns weights $( 1 + 8 \varepsilon ) / ( 2 n )$ and $( 1 - 8 \varepsilon ) / ( 2 n )$ in a random order. We account for both possible orders with the function $^ { g , }$ defined for $a , b \in \mathbb { N }$ by

$$
g ( a , b ) : = \frac { 1 } { 2 } ( 1 + 8 \varepsilon ) ^ { a } ( 1 - 8 \varepsilon ) ^ { b } + \frac { 1 } { 2 } ( 1 - 8 \varepsilon ) ^ { a } ( 1 + 8 \varepsilon ) ^ { b } .
$$

Lemma 9. For every $x , y , z \in \mathbb { N } ^ { 2 }$

$$
\begin{array} { r l } & { \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } ( x , y , z ) = \gamma H ( x , y , z ) + ( 1 - \gamma ) L ( x , y , z ) g ( y _ { 1 } , y _ { 2 } ) , } \\ & { \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } ( x , y , z ) = \gamma H ( x , y , z ) + ( 1 - \gamma ) L ( x , y , z ) g ( y _ { 1 } + z _ { 1 } , y _ { 2 } + z _ { 2 } ) . } \end{array}
$$

Proof. We first consider a light pair. Recall that $L ( x , y , z )$ is the probability of the counts when all three weight vectors give the two elements equal weight $1 / ( 2 n )$ . Suppose that the first element has the larger $Q \mathrm { - w e i g h t }$ . The tilt preserves the total weight of the pair and multiplies the first and second Q-weights by $1 + 8 \varepsilon$ and $1 - 8 \varepsilon$ , respectively. The probability of observing counts $y _ { 1 } , y _ { 2 }$ is therefore multiplied by

$$
( 1 + 8 \varepsilon ) ^ { y _ { 1 } } ( 1 - 8 \varepsilon ) ^ { y _ { 2 } } .
$$

If the second element has the larger $Q \mathrm { - w e i g h t }$ , we interchange $1 + 8 \varepsilon$ and $1 - 8 \varepsilon$

Recall the priors $\Lambda _ { 0 }$ and $\Lambda _ { 1 }$ defined in Section 5.1. Under $\Lambda _ { 0 } ,$ both $w _ { P }$ and $w _ { D }$ give each element weight $1 / ( 2 n )$ . Averaging over the two possible orders of the Q-weights gives the count probability $L ( x , y , z ) g ( y _ { 1 } , y _ { 2 } )$ Under $\Lambda _ { 1 }$ , we have $D = Q$ , so the D-samples use the same orientation as the Q-samples. The first element is observed $y _ { 1 } + z _ { 1 }$ times across these two batches, and the second is observed $y _ { 2 } + z _ { 2 }$ times. The same argument therefore gives $L ( x , y , z ) g ( y _ { 1 } + z _ { 1 } , y _ { 2 } + z _ { 2 } )$

A heavy pair has count distribution H under both priors. Combining the heavy and light cases, which occur with probabilities $\gamma$ and $1 - \gamma$ , gives the stated formulas. □

For $y = ( y _ { 1 } , y _ { 2 } )$ and $z = ( z _ { 1 } , z _ { 2 } )$ , let

$$
| y | : = y _ { 1 } + y _ { 2 } , \qquad | z | : = z _ { 1 } + z _ { 2 } .
$$

To compare the two light-pair probabilities, let B be the diference between the two values of $g \colon$

$$
B ( y , z ) : = g ( y _ { 1 } , y _ { 2 } ) - g ( y _ { 1 } + z _ { 1 } , y _ { 2 } + z _ { 2 } ) .
$$

Lemma 10. For every $y , z \in \mathbb { N } ^ { 2 }$

(1) $B ( y , z ) = 0$ unless $| z | \geqslant 1$ and $| y | + | z | \geqslant 2$

(2) $| B ( y , z ) | \lesssim \varepsilon ^ { 2 } 2 ^ { | y | + | z | }$

Proof. $\mathrm { I f } \ | z | = 0$ , the two terms defining $B ( y , z )$ coincide. $\mathrm { I f } \ | y | + | z | \leqslant 1$ , both values of $g$ equal one, since $g ( a , b ) = 1$ for $a + b \leqslant 1$ . This proves the first claim.

For the second claim, view $B ( y , z )$ as a polynomial in $v = 8 \varepsilon$ . Its constant terms cancel, and averaging over the two signs removes all odd powers, so every surviving term has degree at least two. By the binomial theorem, the sum of the absolute coeficients of $( 1 + v ) ^ { a } ( 1 - v ) ^ { b }$ is at most

$$
\sum _ { r = 0 } ^ { a } \sum _ { s = 0 } ^ { b } { \binom { a } { r } } { \binom { b } { s } } = 2 ^ { a + b } .
$$

Averaging over the signs cannot increase this sum. Hence, the sum of the absolute coeficients of $B ( y , z )$ is at most $\stackrel { \smile } { 2 } ^ { | y | } + 2 ^ { | y | + | z | }$ . Since $v \leqslant 1$ and all surviving powers are at least two,

$$
| B ( y , z ) | \leqslant ( 8 \varepsilon ) ^ { 2 } \bigl ( 2 ^ { | y | } + 2 ^ { | y | + | z | } \bigr ) \leqslant 2 ( 8 \varepsilon ) ^ { 2 } 2 ^ { | y | + | z | } \lesssim \varepsilon ^ { 2 } 2 ^ { | y | + | z | } .
$$

Lemma 11. For every $R \in \{ P , Q , D \}$ and $m \in \{ 0 , 1 , 2 \}$ ,

$$
\sum _ { | u | \geqslant m } 4 ^ { | u | } \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { R } } ( u _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { R } } ( u _ { i } ) } \lesssim \left( \frac { t _ { R } t _ { P } } { n ^ { 2 } } \right) ^ { m } .
$$

Proof. A direct substitution of the Poisson probability formula gives

$$
\frac { \mathrm { p o i } _ { \lambda _ { R } } ( k ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { R } } ( k ) } = e ^ { \mu _ { R } - 2 \lambda _ { R } } \frac { ( t _ { R } t _ { P } / ( 4 n ^ { 2 } ) ) ^ { k } } { k ! } .
$$

It follows that

$$
\begin{array} { c c } { { \displaystyle \sum _ { | u | \geqslant m } 4 ^ { | u | } \displaystyle \prod _ { i = 1 } ^ { 2 } \displaystyle \frac { \log \mathrm { i } _ { \lambda _ { R } } ( u _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { R } } ( u _ { i } ) } = e ^ { 2 \mu _ { R } - 4 \lambda _ { R } } \displaystyle \sum _ { k \geqslant m } \left( \displaystyle \frac { t _ { R } t _ { P } } { n ^ { 2 } } \right) ^ { k } \displaystyle \sum _ { a + b = k } \displaystyle \frac { 1 } { a ^ { | b | } } } } \\ { { } } & { { = e ^ { 2 \mu _ { R } - 4 \lambda _ { R } } \displaystyle \sum _ { k \geqslant m } \displaystyle \frac { ( 2 t _ { R } t _ { P } / n ^ { 2 } ) ^ { k } } { k ! } ~ } } \\ { { } } & { { \leqslant e ^ { 2 \mu _ { R } - 4 \lambda _ { R } + 2 t _ { R } t _ { P } / n ^ { 2 } } \displaystyle \frac { ( 2 t _ { R } t _ { P } / n ^ { 2 } ) ^ { m } } { m ! } ~ } } \\ { { } } & { { \lesssim \left( \displaystyle \frac { t _ { R } t _ { P } } { n ^ { 2 } } \right) ^ { m } ~ } } & { { ~ ( t _ { R } \leqslant t _ { P } \leqslant n ) . } } \end{array}
$$

□

Proof of Lemma 6. By Lemma 9, we have

$$
\begin{array} { r l } & { \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } ( x , y , z ) - \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } ( x , y , z ) = ( 1 - \gamma ) L ( x , y , z ) B ( y , z ) , } \\ & { \quad \quad \quad \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } ( x , y , z ) \geqslant \gamma H ( x , y , z ) . } \end{array}
$$

Substituting this into the definition of $\chi ^ { 2 }$ gives

$$
\begin{array} { r l } & { \chi ^ { 2 } ( \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } \Vert \mathbb { P } _ { 1 } ^ { \mathrm { p p a i r } } ) = \displaystyle \sum _ { x , y , z } \frac { \left( \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } ( x , y , z ) - \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } ( x , y , z ) \right) ^ { 2 } } { \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } ( x , y , z ) } } \\ & { \qquad \leqslant \frac { ( 1 - \gamma ) ^ { 2 } } { \gamma } \displaystyle \sum _ { x , y , z } \frac { L ( x , y , z ) ^ { 2 } } { H ( x , y , z ) } B ( y , z ) ^ { 2 } } \\ & { \qquad \leqslant \frac { 1 } { \gamma } \displaystyle \sum _ { x , y , z } \frac { L ( x , y , z ) ^ { 2 } } { H ( x , y , z ) } B ( y , z ) ^ { 2 } . } \end{array}
$$

By the definitions of $L$ and $H$

$$
{ \frac { L ( x , y , z ) ^ { 2 } } { H ( x , y , z ) } } = \prod _ { i = 1 } ^ { 2 } { \frac { \mathrm { p o i } _ { \lambda _ { P } } ( x _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { 1 } ( x _ { i } ) } } { \frac { \mathrm { p o i } _ { \lambda _ { Q } } ( y _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { Q } } ( y _ { i } ) } } { \frac { \mathrm { p o i } _ { \lambda _ { D } } ( z _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { D } } ( z _ { i } ) } } .
$$

Since $B ( y , z )$ does not depend on $x ,$ we can sum over the P-counts first. Applying Lemma 11 with $R = P$ (and hence $\mu _ { R } = 1 )$ and $m = 0$ gives

$$
\sum _ { x } \prod _ { i = 1 } ^ { 2 } { \frac { \mathrm { p o i } _ { \lambda _ { P } } ( x _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { 1 } ( x _ { i } ) } } \leqslant \sum _ { x } { 4 ^ { | x | } } \prod _ { i = 1 } ^ { 2 } { \frac { \mathrm { p o i } _ { \lambda _ { P } } ( x _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { 1 } ( x _ { i } ) } } \lesssim 1 .
$$

Since $\begin{array} { r } { \gamma = \frac { t _ { P } } { 4 n } } \end{array}$ , we therefore obtain

$$
\chi ^ { 2 } ( { \mathbb P } _ { 0 } ^ { \mathrm { p a i r } } \| { \mathbb P } _ { 1 } ^ { \mathrm { p a i r } } ) \lesssim \frac { n } { t _ { P } } \sum _ { y , z } \left( \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { Q } } ( y _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { Q } } ( y _ { i } ) } \frac { \mathrm { p o i } _ { \lambda _ { D } } ( z _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { D } } ( z _ { i } ) } \right) B ( y , z ) ^ { 2 } .
$$

By Lemma 10, $B ( y , z ) = 0$ unless $| z | \geqslant 1$ and $| y | + | z | \geqslant 2 .$ . Hence, if $| z | \geqslant 2$ , there is no restriction on y. $\mathrm { I f } \ | z | = 1$ , we need $| y | \geqslant 1$ . The same lemma gives $\ddot { B ( y , z ) ^ { 2 } } \lesssim \varepsilon ^ { 4 } 4 ^ { | y | } 4 ^ { | z | }$ . After this substitution, the summand

is a product of a function of $y$ and a function of $z ,$ so the two sums can be evaluated separately. For $| z | \geqslant 2 ,$ Lemma 11 gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { y , z } ( \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { Q } } ( y _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { Q } } ( y _ { i } ) } \frac { \mathrm { p o i } _ { \lambda _ { P } } ( z _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { D } } ( z _ { i } ) } ) B ( y , z ) ^ { 2 } } \\ & { \displaystyle \lesssim \varepsilon ^ { 4 } ( \underbrace { \sum _ { y } 4 ^ { \vert y \vert } \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { Q } } ( y _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { Q } } ( y _ { i } ) } } _ { \lesssim 1 \pmod { 2 } } \underbrace { ( \sum _ { l \geq 2 } 4 ^ { \vert z \vert } \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { D } } ( z _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { D } } ( z _ { i } ) } ) } _ { \lesssim ( t _ { D } t _ { P } / n ^ { 2 } ) ^ { 2 } \ ( m = 2 ) } } \\ & { \displaystyle \lesssim \varepsilon ^ { 4 } ( \frac { t _ { D } t _ { P } } { n ^ { 2 } } ) ^ { 2 } . } \end{array}
$$

For $| z | = 1$ and $| y | \geqslant 1$ , every term is nonnegative, so enlarging the $D .$ -sum from $| z | = 1$ to $| z | \geqslant 1$ can only increase it. We can then apply Lemma 11 with $m = 1$ to both sums:

$$
\begin{array} { r l } & { \displaystyle \sum _ { | y | \ge 1 } \left( \displaystyle \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { Q } } ( y _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { Q } } ( y _ { i } ) } \frac { \mathrm { p o i } _ { \lambda _ { D } } ( z _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { D } } ( z _ { i } ) } \right) B ( y , z ) ^ { 2 } } \\ & { \displaystyle \lesssim \varepsilon ^ { 4 } \underbrace { \left( \sum _ { | y | \ge 1 } 4 ^ { | y | } \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { Q } } ( y _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { Q } } ( y _ { i } ) } \right) } _ { \lesssim t o t e r / n ^ { 2 } ~ \mathrm { ( m = 1 ) } } \underbrace { \left( \sum _ { | z | \gtrsim 1 } 4 ^ { | z | } \prod _ { i = 1 } ^ { 2 } \frac { \mathrm { p o i } _ { \lambda _ { D } } ( z _ { i } ) ^ { 2 } } { \mathrm { p o i } _ { \mu _ { D } } ( z _ { i } ) } \right) } _ { \lesssim t o t e r / n ^ { 2 } ~ \mathrm { ( m = 1 ) } } } \\ & { \displaystyle \lesssim \varepsilon ^ { 4 } \frac { t Q ^ { d } P } { n ^ { 2 } } \frac { L D t P } { n ^ { 2 } } . } \end{array}
$$

Adding these two bounds gives

$$
\chi ^ { 2 } ( \mathbb { P } _ { 0 } ^ { \mathrm { p a i r } } \Vert \mathbb { P } _ { 1 } ^ { \mathrm { p a i r } } ) \lesssim \frac { n } { t _ { P } } \varepsilon ^ { 4 } \left[ \left( \frac { t _ { D } t _ { P } } { n ^ { 2 } } \right) ^ { 2 } + \frac { t _ { Q } t _ { P } } { n ^ { 2 } } \frac { t _ { D } t _ { P } } { n ^ { 2 } } \right] = \frac { \varepsilon ^ { 4 } t _ { P } t _ { D } ( t _ { Q } + t _ { D } ) } { n ^ { 3 } } .
$$

## B.2 Average-Case Lower Bound

For later use, we state the following remark.

Remark 12 (Average-case form of (7)). We claim that the lower bound (7) also holds if the tester is required to have error probability at most $\frac 1 3$ on average under each conditional prior $\Lambda _ { \beta } ( \cdot \mid E )$ , where $\beta \in \{ 0 , 1 \}$ , E is the high-probability event in Lemma $^ { 4 , }$ and $\Lambda _ { \beta }$ are the priors specified in Section 5.1. The analysis is unchanged except for the Poissonization step in Lemma 5. For that step, let $e ( I )$ be the error probability of the tester on a fixed instance I, and let $F$ be the suficient-samples event from the proof of Lemma $5 .$ Conditional on $I ,$ the sample counts are independent of the retained sample values, and hence the contribution to the conditional error probability on the event $F$ satisfies

$$
\mathbb { E } _ { I \sim \Lambda _ { \beta } ( \cdot | E ) } \left[ e ( I ) \operatorname* { P r } ( F \mid I ) \right] \leqslant \mathbb { E } _ { I \sim \Lambda _ { \beta } ( \cdot | E ) } { \left[ e ( I ) \right] } \leqslant { \frac { 1 } { 3 } } .
$$

Adding $\operatorname* { P r } ( E ^ { c } ) \leqslant 1 / 2 5$ and $\operatorname* { P r } ( F ^ { c } \mid E ) < 1 0 ^ { - 3 }$ then gives the same error bound as before, namely $1 / 3 +$ $1 / 2 5 + 1 0 ^ { - 3 } < 2 / 5$ , under each unconditioned prior. The rest of the proof in Section 5.1 is unchanged.

## C Non-Adaptive Distribution Clustering

This appendix gives the algorithms and proofs for the 2-clustering results in Section $6 .$

## C.1 Upper Bounds with One Known Distribution

This subsection proves the upper bounds with one known distribution summarized in Section 6, using the model and quantities $r , T , T _ { \mathrm { n a } }$ defined therein. The matching lower bound is proved in Section C.3.2. Recall from Section 6 and (Kumar et al., 2026) that it sufices to consider the case that the known distribution is uniform, and accordingly denoted by U.

## C.1.1 Baseline

Uniformity testing in the sparse regime was studied by Paninski (2008). We use the identity tester of Diakonikolas and Kane (2016, Proposition 2.7), specialized to uniformity, which applies throughout our parameter range.

Theorem 13. $\begin{array} { r } { T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = O \Big ( \frac { k \sqrt { n } \log k } { \varepsilon ^ { 2 } } \Big ) } \end{array}$ when one distribution is known.

Proof. Since the known distribution is $U ,$ each input distribution satisfies $D _ { i } = U$ or $\operatorname { d } _ { \mathrm { T V } } ( D _ { i } , U ) \geqslant \varepsilon$ . Run this uniformity test on each input distribution at per-test error $\frac { 1 } { 9 k }$ , costing $O \left( { \frac { { \sqrt { n } } \log { k } } { \varepsilon ^ { 2 } } } \right)$ samples per input distribution, and assign i to $\mathcal { T } _ { U }$ or $\scriptstyle { \mathcal { T } } _ { Q }$ by the outcome. A union bound over the k input distributions gives success $\geqslant \ { \frac { 8 } { 9 } }$ , for a total of $O \left( { \frac { k { \sqrt { n } } \log { k } } { \varepsilon ^ { 2 } } } \right)$ □

## C.1.2 Improved Upper Bound

With the known distribution being U, the task is to find the hidden partition $\{ \mathcal { T } _ { U } , \mathcal { T } _ { Q } \}$ . Algorithm 2 implements the two stages sketched in Section 6.

Algorithm 2: Non-adaptive 2-clustering with one known distribution.   
Input: Parameters $s _ { 0 } , s _ { 1 } , s _ { 2 } \in \mathbb { N } .$   
Output: Partition of $[ k ]$   
1 Set $\ell = \lceil 1 8 \log ( 9 k ) \rceil$   
2 Pick a uniformly random subset $S \subset [ k ]$ with $| S | =$ min $\left\{ k , \left\lceil \frac { 3 k } { r } \right\rceil \right\}$   
3 Draw $s _ { 0 } + \ell \big ( s _ { 1 } + s _ { 2 } \big )$ samples from each $i \in S$ and $\ell s _ { 2 }$ samples from each $i \in [ k ] \setminus S$   
4 for $i \in S$ do   
5 Run a uniformity test on the first $s _ { 0 }$ samples from $i ,$ with per-test error $\frac { 1 } { 9 | S | }$   
6 Let $i ^ { * }$ be any rejected distribution if one exists. Otherwise, return an arbitrary size- $( r , k - r )$ partition   
(or alternatively declare failure)   
7 for $i \in [ k ]$ do   
8 Run asymmetric LFHT for $( U , D _ { i ^ { * } } )$ in ℓ independent repetitions, each using samples generated   
from $U$ and disjoint batches of $s _ { 1 }$ samples from $i ^ { * }$ and $s _ { 2 }$ from $D _ { i } .$ and classify $D _ { i }$ by majority   
vote   
9 return the partition

Theorem 14. For $r \in [ 1 , \frac { k } { 2 } ]$ , Algorithm 2 solves 2-clustering with success probability $\geqslant 2 / 3$ using

$$
T = \left\{ \begin{array} { l l } { \displaystyle \widetilde O \bigg ( \frac { k \sqrt n } { \sqrt { r } \varepsilon ^ { 2 } } \bigg ) } & { \mathit { i f } n \gtrsim r \log k , } \\ { \displaystyle \widetilde O \bigg ( \frac { k \log k } { \varepsilon ^ { 2 } } \bigg ) } & { \mathit { i f } n \lesssim r \log k } \end{array} \right.
$$

samples.

Proof. Stage 1 uses $s _ { 0 } = O ( \sqrt { n } \log k / \varepsilon ^ { 2 } )$ samples per member of $S ,$ for a total of $\widetilde { \cal O } ( k \sqrt { n } / ( r \varepsilon ^ { 2 } ) )$ . With probability at least $7 / 9 _ { ; }$ , the subset contains a member of $\scriptstyle { \mathcal { T } } _ { Q }$ and all its uniformity tests are correct, so $i ^ { * } \in \mathcal { T } _ { Q }$

For Stage $2 ,$ use $U$ as the more heavily sampled reference in Theorem 1. Since U is known, its samples can be generated directly without using the input-distribution budget. Generating enough of them makes $\alpha = 1$ and satisfies the condition on the larger reference budget. By Theorem 1, it sufices to take any pair $( s _ { 1 } , s _ { 2 } )$ satisfying the following and inflate it by a polylogarithmic factor:

$$
s _ { 2 } \gtrsim \frac { 1 } { \varepsilon ^ { 2 } } , \qquad s _ { 2 } ( s _ { 1 } + s _ { 2 } ) \gtrsim \frac { n } { \varepsilon ^ { 4 } } .
$$

Accordingly, we choose

$$
s _ { 2 } = \widetilde \Theta \left( \operatorname* { m a x } \left\{ \frac { 1 } { \varepsilon ^ { 2 } } , \frac { \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } \right\} \right) , \qquad s _ { 1 } = r s _ { 2 } .
$$

The two terms in the maximum ensure the conditions $s _ { 2 } \gtrsim 1 / \varepsilon ^ { 2 }$ and $s _ { 1 } s _ { 2 } = r s _ { 2 } ^ { 2 } \gtrsim n / \varepsilon ^ { 4 }$ prior to the polylogarithmic increase, and the $\widetilde { \Theta }$ notation captures the required increase.

The $\ell s _ { 1 }$ samples must be reserved for every member of $S _ { ☉ }$ , since $i ^ { * }$ is selected only after sampling. Since $\ell = \Theta ( \log k )$ , Stage 2 costs

$$
\ell \big ( | S | s _ { 1 } + k s _ { 2 } \big ) \asymp \ell \left( \frac { k } { r } s _ { 1 } + k s _ { 2 } \right) \asymp \ell k s _ { 2 } = \widetilde O \left( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \right) .
$$

This also covers Stage 1 since $r \geqslant 1$ , and gives the two stated regimes. The Stage 2 batches are independent of the samples used to choose $i ^ { * }$ . For each classification, the ℓ repetitions use disjoint samples and independent internal randomness. Hoefding’s inequality bounds the majority-vote error by $\stackrel { \textstyle - \ell / 1 8 } { \textstyle e ^ { - \ell / 1 8 } } \leqslant \bar { 1 } / ( 9 k )$ . The reference batches may be reused across classifications. A union bound over the k classifications gives Stage 2 failure at most $1 / 9$ , and total failure at most $2 / 9 + 1 / 9 = 1 / 3$ □

## C.2 Upper Bounds with Both Distributions Unknown

The non-adaptive setting is as in Section 6. Here both cluster distributions $P , Q$ are unknown, and the task is to find the hidden partition $\{ \mathcal { T } _ { P } , \mathcal { T } _ { Q } \}$ . We prove the upper bounds here and the matching lower bound in Section C.3.3.

## C.2.1 Baseline

Here the analog of Theorem 13 is given as follows.

Theorem 15. $\begin{array} { r } { T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = O \Big ( k \log k \cdot \operatorname* { m a x } \Big ( \frac { n ^ { 2 / 3 } } { \varepsilon ^ { 4 / 3 } } , ~ \frac { \sqrt { n } } { \varepsilon ^ { 2 } } \Big ) \Big ) } \end{array}$ when both distributions are unknown.

Proof. Fix an arbitrary index $i _ { A } \in [ k ]$ , and let its distribution $D _ { i _ { A } }$ be the one given more samples, recalling that we refer to $i _ { A }$ as the pivot. By assumption in the problem formulation, the other $k - 1$ distributions all satisfy $D _ { i } = D _ { i _ { A } }$ or $\mathrm { d } _ { \mathrm { T V } } ( D _ { i } , D _ { i _ { A } } ) \geqslant \varepsilon$ . Run a closeness test between $D _ { i _ { A } }$ and each $D _ { i }$ (Chan et al., 2014) at per-test error $\frac { 1 } { 9 k }$ , costing $\begin{array} { r } { O \left( \operatorname* { m a x } \left( \frac { n ^ { 2 / 3 } } { \varepsilon ^ { 4 / 3 } } , \frac { \sqrt { n } } { \varepsilon ^ { 2 } } \right) \log k \right) } \end{array}$ samples per test (the pivot’s samples are drawn once and reused), and cluster [k] according to the outcomes. A union bound over the $k - 1$ tests gives success $\geqslant \ { \frac { 8 } { 9 } }$ , for a total of $\begin{array} { r } { O \left( k \log k \operatorname* { m a x } \left( \frac { n ^ { 2 / 3 } } { \varepsilon ^ { 4 / 3 } } , \ \frac { \sqrt { n } } { \varepsilon ^ { 2 } } \right) \right) } \end{array}$ . □

## C.2.2 Improved Upper Bounds

We improve on the baseline bound with Algorithm 3, which implements the two stages sketched in Section 6. A random size- $\cdot \Theta \left( { \frac { k } { r } } \right)$ subset $S$ supplies an exemplar of each cluster, an element $i _ { A }$ of $S$ serving as the pivot, and asymmetric LFHT then classifies all k distributions against the two exemplars.

Stage 1 uses the closeness tester of Diakonikolas and Kane (2016) with unequal sample sizes, which we denote by UST. Given m samples from one distribution and $m ^ { \prime } \leqslant$ m from another, it decides whether the two are equal or at total variation distance at least ε, with error at most $\textstyle { \frac { 1 } { 3 } }$ , provided that

$$
m ^ { \prime } \gtrsim \frac { \sqrt { n } } { \varepsilon ^ { 2 } } \qquad \mathrm { a n d } \qquad m ^ { \prime 2 } m \gtrsim \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } .
$$

These are the conditions in (4). Taking the majority of $O ( \log k )$ independent runs reduces the error to $\frac { 1 } { 9 | S | }$ and we will use this guarantee in the proof below.

Algorithm 3: Non-adaptive 2-clustering with both distributions unknown.   
Input: Parameters $s _ { a } , s _ { b } , s _ { P } , s _ { Q } , s _ { D } \in \mathbb { N } .$   
Output: Partition of [k].   
1 Set $\ell = \lceil 1 8 \log ( 9 k ) \rceil$   
2 Pick a uniformly random subset $S \subset [ k ]$ with $| S | =$ min $\textstyle \left\{ k , \ \left\lceil { \frac { 3 k } { r } } \right\rceil \right\}$ , and let $i _ { A }$ be any element of $S$   
3 Draw $s _ { a } + \ell \big ( s _ { P } + s _ { D } \big )$ samples from $i _ { A } , s _ { b } + \ell ( s _ { Q } + s _ { D } )$ samples from each $i \in S \setminus \{ i _ { A } \}$ , and $\ell s _ { D }$   
samples from each $i \in [ k ] \setminus S$   
4 for $i \in S \setminus \{ i _ { A } \}$ do   
5 Run UST for $( D _ { i _ { A } } , D _ { i } )$ using the $s _ { a }$ samples from $i _ { A }$ as sketch and the $s _ { b }$ samples from i as test,   
at per-test error $\frac { 1 } { 9 | S | }$   
6 Let $i _ { B }$ be any $i \in S \setminus \{ i _ { A } \}$ whose test rejected if one exists. Otherwise return an arbitrary   
size- $( r , k - r )$ partition (or alternatively declare failure)   
7 for $i \in [ k ]$ do   
8 Run asymmetric LFHT for $( D _ { i _ { A } } , D _ { i _ { B } } )$ in ℓ independent repetitions, each using disjoint batches of   
$s _ { P }$ samples from $i _ { A } , s _ { Q }$ from $i _ { B } ,$ , and $s _ { D }$ from $D _ { i } ,$ and classify $D _ { i }$ by majority vote   
9 return the partition

Theorem 16. For $r \in [ 1 , \frac { k } { 2 } ]$ , Algorithm 3 solves 2-clustering with success probability $\geqslant 2 / 3$ using

$$
T = \left\{ \begin{array} { l l } { \tilde { \cal O } \bigg ( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } \bigg ) } & { i f n \gtrsim \frac { k ^ { 2 } } { r \varepsilon ^ { 4 } } , } \\ { \tilde { \cal O } \bigg ( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } \bigg ) } & { i f r \log k \lesssim n \lesssim \frac { k ^ { 2 } } { r \varepsilon ^ { 4 } } , } \\ { \tilde { \cal O } \bigg ( \frac { k \log k } { \varepsilon ^ { 2 } } \bigg ) } & { i f n \lesssim r \log k } \end{array} \right.
$$

samples.

Proof. In Stage 1, choose

$$
s _ { b } = \widetilde \Theta \left( \operatorname* { m a x } \left\{ \frac { \sqrt { n } } { \varepsilon ^ { 2 } } , \ : \left( \frac { n ^ { 2 } r } { k \varepsilon ^ { 4 } } \right) ^ { 1 / 3 } \right\} \right) , \qquad s _ { a } = \frac { k } { r } s _ { b } ,
$$

where the logarithmic factor covers the repetitions needed for error $\frac { 1 } { 9 | S | }$ . Since $s _ { a } \geqslant s _ { b }$ , the guarantee of UST requires $\begin{array} { r } { s _ { b } \gtrsim \frac { \sqrt { n } } { \varepsilon ^ { 2 } } } \end{array}$ and $\begin{array} { r } { s _ { b } ^ { 2 } s _ { a } = \frac { k } { r } s _ { b } ^ { 3 } \gtrsim \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } } \end{array}$ , which both hold. The Stage 1 cost is

$$
s _ { a } + | S | s _ { b } \asymp \frac { k } { r } s _ { b } = \widetilde O \left( \operatorname* { m a x } \left( \left( \frac { n k } { r \varepsilon ^ { 2 } } \right) ^ { 2 / 3 } , \frac { k \sqrt { n } } { r \varepsilon ^ { 2 } } \right) \right) .
$$

Unless $S = [ k ]$ , we have $\begin{array} { r } { | S | \geqslant { \frac { 3 k } { r } } } \end{array}$ , and each cluster has at least r members, so a given cluster has no member in S with probability at most $\begin{array} { r } { \left( 1 - \frac { r } { k } \right) ^ { | S | } \leqslant e ^ { - 3 } < \frac { 1 } { 1 8 } } \end{array}$ . Hence $S$ contains a member of each cluster except with probability $\frac { 1 } { 9 }$ , and a union bound over the $| S | - 1$ tests, each at error $\frac { 1 } { 9 | S | }$ , gives total Stage 1 failure $\leqslant \frac { 2 } { 9 }$ . On Stage 1 success, some $i \in S \setminus \{ i _ { A } \}$ lies in the cluster opposite $i _ { A }$ and its test rejects, while every same-cluster test accepts, so $i _ { B }$ is defined and lies in the opposite cluster.

In Stage 2, apply Theorem 1 with the pivot $i _ { A }$ as the more heavily sampled reference, so one run uses sample sizes $\left( s _ { P } , s _ { Q } , s _ { D } \right)$ . Choose

$$
\begin{array} { l } { { s _ { D } = \widetilde \Theta \left( \operatorname* { m a x } \left\{ \displaystyle \frac 1 { \varepsilon ^ { 2 } } , \quad \displaystyle \frac { \sqrt n } { \sqrt r \varepsilon ^ { 2 } } , \quad \left( \displaystyle \frac { n ^ { 2 } } { r k \varepsilon ^ { 4 } } \right) ^ { 1 / 3 } \right\} \right) , } } \\ { { s _ { Q } = r s _ { D } , \qquad s _ { P } = k s _ { D } , } } \end{array}
$$

where the polylogarithmic factor in $\widetilde { \Theta }$ is the one required by Theorem 1. The second and third terms in the maximum give the following conditions (before the required logarithmic increase):

$$
s _ { Q } s _ { D } = r s _ { D } ^ { 2 } \stackrel { > } { \sim } \frac { n } { \varepsilon ^ { 4 } } , \qquad s _ { P } s _ { Q } s _ { D } = k r s _ { D } ^ { 3 } \stackrel { > } { \sim } \frac { n ^ { 2 } } { \varepsilon ^ { 4 } } ,
$$

which together imply $s _ { Q } s _ { D } \gtrsim n \alpha / \varepsilon ^ { 4 }$ for $\alpha = \operatorname* { m a x } \{ 1 , n / s _ { P } \}$ . Moreover, $s _ { P } \geqslant s _ { Q } \geqslant s _ { D }$ gives $s _ { P } ^ { 2 } \geqslant s _ { Q } s _ { D } \ \stackrel { > } { \sim }$ $n \alpha / \varepsilon ^ { 4 }$ , and the first term ensures $s _ { D } \gtrsim 1 / \varepsilon ^ { 2 }$ . The LFHT test analyzed in Section 4 therefore has error probability at most $1 / 3$ with these budgets.

Fix the choices and samples from Stage 1, and suppose it succeeds. Then $i _ { A }$ and $i _ { B }$ belong to diferent clusters. The reserved Stage 2 samples are independent of the Stage 1 samples. For each input $D _ { i }$ , the $\ell$ LFHT runs use disjoint batches and independent internal randomness. Hoefding’s inequality therefore bounds the majority-vote error by $e ^ { - \ell / 1 8 } \leqslant 1 / ( 9 k )$ . We use the same reference samples for all k inputs. By a union bound, the probability of any wrong classification is at most $\boldsymbol { k } \cdot 1 / ( 9 \boldsymbol { k } ) = 1 / 9$

We must allocate samples before choosing $i _ { B } ,$ so we reserve ℓs<sub>Q</sub> reference samples for every candidate in $S \setminus \{ i _ { A } \}$ . Since $\ell = \Theta ( \log k )$ , the Stage 2 cost is

$$
\begin{array} { r } { \ell \big ( s _ { P } + ( | S | - 1 ) s _ { Q } + k s _ { D } \big ) \asymp \ell \left( s _ { P } + \frac { k } { r } s _ { Q } + k s _ { D } \right) \asymp \ell k s _ { D } } \\ { = \widetilde O \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } + \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \right) . } \end{array}
$$

Since $r \geqslant 1$ , the Stage 1 terms satisfy

$$
\left( \frac { n k } { r \varepsilon ^ { 2 } } \right) ^ { 2 / 3 } \leqslant \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } , \qquad \frac { k \sqrt { n } } { r \varepsilon ^ { 2 } } \leqslant \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } .
$$

Thus Stage 1 is covered by the same bound. Comparing the first two terms gives the threshold $n = k ^ { 2 } / ( r \varepsilon ^ { 4 } )$ For $n \lesssim$ r log k, all three terms are $\widetilde { O } ( k / \varepsilon ^ { 2 } )$ . This gives the stated regimes. The total failure probability is at most $2 / 9 + 1 / 9 = 1 / 3$ □

## C.3 Lower Bounds

The two lower bounds use a common reduction, Lemma $^ { 1 7 , }$ from a clustering algorithm to an LFHT tester.   
A clustering algorithm with total budget $T$ gives an LFHT tester with $\begin{array} { r } { s _ { P } = T , s _ { Q } = O ( \frac { T r } { k } ) } \end{array}$ , and $\begin{array} { r } { s _ { D } = O ( \frac { T } { k } ) } \end{array}$ .   
We then apply the LFHT lower bounds with one known distribution and with both unknown.

We may assume that the clustering algorithm succeeds with probability at least $\frac { 9 9 } { 1 0 0 }$ . For instance, repeating it a constant number of times and taking the majority output achieves this, multiplies T by a constant, and keeps the algorithm non-adaptive.

## C.3.1 The Reduction from LFHT

To turn a clustering algorithm into an LFHT tester, we place the unknown distribution D at a lightly sampled index and use the output partition to distinguish $D = P$ from $D = Q$ . This follows the ideas of Kumar et al. (2026, Appendix E.2, Lemmas 22–23).

We order the indices by expected number of samples. The tester places copies of $Q$ at the first $r - 1$ indices, the unknown distribution $D$ at one more index $j$ with few expected samples, and copies of $P$ everywhere else. If $D = Q$ , the small cluster is the first $r - 1$ indices together with $j ,$ and the algorithm returns this partition with high probability. If $D = P$ , the instance has only $r - 1$ copies of $Q$ and need not satisfy the clustering promise. We do not assume correctness on this instance. Instead, an averaging argument lets us choose $j$ so that the algorithm rarely returns that same candidate partition. The output therefore reliably reveals whether $D = Q$ or $D = P$

Non-adaptivity lets us control the Q-budget as well. Since the allocation is fixed before observing any samples, the copies of $Q$ at the least-sampled indices have total expected sample count at most $\frac { T ( r - 1 ) } { k }$ under either LFHT hypothesis. Capping their sample count at a constant multiple of this bound gives a Q-budget of $O ( \frac { T r } { k } )$ , with only a small probability of aborting. Together with the lightly sampled index $j ,$ this gives budgets $\begin{array} { r } { ( T , O ( \frac { T r } { k } ) , O ( \frac { T } { k } ) ) } \end{array}$ , whereas the reduction of Kumar et al. (2026) allows $T$ samples from each reference. The smaller $Q { \mathrm { - b u d g e t } }$ lets us apply the asymmetric lower bounds of Section 5.

The formal reduction is stated as follows.

Lemma 17. Let k $\geqslant$ 200 and $1 \leqslant r \leqslant k / 2$ , and let A be a non-adaptive algorithm for 2-clustering, with one known distribution or with both unknown. Suppose its (possibly random) allocation $( T _ { 1 } , \dots , T _ { k } )$ satisfies $\begin{array} { r } { \sum _ { i } T _ { i } \leqslant T } \end{array}$ , and it returns the correct size- $( r , k - r )$ partition with probability at least $\frac { 9 9 } { 1 0 0 }$ on every valid instance. Let Λ be a distribution over pairs $( P , Q )$ with $\operatorname { d _ { T V } } ( P , Q ) \geqslant \varepsilon$ , where $P$ is fixed and known in the first variant. Then there exists an algorithm $\mathcal { A } ^ { \prime }$ that decides whether $D = P$ or $D = Q$ using

$$
s _ { P } = T , \qquad s _ { Q } = \operatorname* { m i n } \left( \left\lceil \frac { 4 0 T r } { k } \right\rceil , T \right) , \qquad s _ { D } = \operatorname* { m i n } \left( \left\lceil \frac { 1 6 0 T } { k } \right\rceil , T \right)
$$

samples from $P , Q _ { i }$ , and $D _ { i }$ , respectively. Under each hypothesis, its error averaged over $( P , Q ) \sim \Lambda$ is at most ${ \frac { 1 } { 1 0 } } .$ Moreover, $s _ { D } \leqslant 4 s _ { Q }$ and $T \geqslant k / 5$ , and hence

$$
s _ { P } = T , s _ { Q } = O ( T r / k ) , s _ { D } = O ( T / k ) .\tag{17}
$$

Proof. Let $\hat { \mathcal { T } }$ be the output of A. Relabel the indices so that $\mathbb { E } [ T _ { 1 } ] \leqslant \cdots \leqslant \mathbb { E } [ T _ { k } ]$ . The allocation is non-adaptive, so its distribution does not depend on the instance.

Choice of $j .$ . For each $j \in \{ r , \ldots , k \}$ , let $S _ { j } : = \{ 1 , \dots , r - 1 \} \cup \{ j \}$ and $\Pi _ { j } : = \{ S _ { j } , [ k ] \backslash S _ { j } \}$ . These candidate partitions are distinct, including when $r = k / 2$ . Consider the random instance $( P , Q ) \sim \Lambda , D _ { i } = Q$ for $i < r ,$ $D _ { i } = P$ for $i \geqslant r$ . This comparison instance has only $r - 1$ copies of $Q$ and need not satisfy the clustering promise. We use only the distribution of the algorithm’s output on it, without assuming correctness. We show that there exists $j \in \{ r , \ldots , k \}$ such that:

$\begin{array} { r } { \mathrm { P r } [ \hat { \mathcal { T } } = \Pi _ { j } ] \leqslant \frac { 4 } { k } } \end{array}$ , and

$\begin{array} { r } { \mathbb { E } [ T _ { j } ] \leqslant \frac { 4 T } { k } . } \end{array}$

The events $\{ \hat { \mathcal { T } } = \Pi _ { j } \} _ { j \geqslant \iota }$ <sub>r</sub> are mutually exclusive, so $\begin{array} { r } { \sum _ { j \geq r } \operatorname* { P r } [ \hat { \mathcal { Z } } = \Pi _ { j } ] \leqslant 1 } \end{array}$ , and by averaging, more than half of the $j \geqslant r$ satisfy $\begin{array} { r } { \mathrm { P r } [ \hat { \mathcal { T } } = \Pi _ { j } ] \leqslant \frac { 2 } { k - r + 1 } \leqslant \frac { 4 } { k } } \end{array}$ (using $\begin{array} { r } { r \leqslant \frac { k } { 2 } ) } \end{array}$ . Likewise $\begin{array} { r } { \sum _ { j \geq r } \mathbb { E } [ T _ { j } ] \leqslant T } \end{array}$ , so more than half satisfy $\begin{array} { r } { \mathbb { E } [ T _ { j } ] \leqslant \frac { 4 T } { k } } \end{array}$ . Fix any j satisfying both.

Algorithm $A ^ { \prime } . \quad A ^ { \prime }$ requests $s _ { P } , s _ { Q }$ , and $s _ { D }$ samples from the $P \mathrm { \mathrm { - } } , Q \mathrm { \mathrm { - } }$ , and D-oracles, and runs A on the simulated instance

$$
D _ { i } : = Q ~ ( i < r ) , \qquad D _ { j } : = D , \qquad D _ { i } : = P ~ ( i \notin S _ { j } ) ,
$$

forwarding its sample requests to the requested samples. If A requests more than $s _ { Q }$ samples in total from $\{ D _ { i } \} _ { i < r } ,$ or more than $s _ { D }$ samples from $D _ { j } .$ , then $\mathcal { A } ^ { \prime }$ aborts and returns $D = P$ . Otherwise it returns $D = Q$ i $\textrm { f } \hat { \boldsymbol { { \cal { T } } } } = \Pi _ { j }$ , and $D = P$ otherwise. The requests to the P-oracle never exceed $s _ { P } = T$

Recall that $s _ { Q }$ and $s _ { D }$ take the form $\operatorname* { m i n } ( \dots , T )$ . Whenever $T$ achieves the minimum, an abort is impossible. Otherwise, the two respective abort probabilities are bounded via Markov’s inequality:

$$
\operatorname* { P r } \left[ \sum _ { i < r } T _ { i } > s _ { Q } \right] \leqslant \frac { \sum _ { i < r } \mathbb { E } [ T _ { i } ] } { \frac { 4 0 T r } { k } } \leqslant \frac { \frac { T ( r - 1 ) } { k } } { \frac { 4 0 T r } { k } } \leqslant \frac { 1 } { 4 0 } , \qquad \operatorname* { P r } [ T _ { j } > s _ { D } ] \leqslant \frac { \mathbb { E } [ T _ { j } ] } { \frac { 1 6 0 T } { k } } \leqslant \frac { \frac { 4 T } { k } } { \frac { 1 6 0 T } { k } } = \frac { 1 } { 4 0 } ,
$$

using $\begin{array} { r } { \sum _ { i < r } \mathbb { E } [ T _ { i } ] \leqslant \frac { T ( r - 1 ) } { k } } \end{array}$ (the smallest $r - 1$ of the sorted expectations sum to at most that much, since $\begin{array} { r } { \sum _ { i } \mathbb { E } [ T _ { i } ] { \stackrel { . } { \leqslant } } T ) } \end{array}$ and the choice of j.

Case $D = P .$ . An abort returns $D = P .$ which is correct, and otherwise the simulated instance is the one used to pick j, so the error probability is at most $\begin{array} { r } { \mathrm { P r } [ \hat { \mathcal { Z } } = \Pi _ { j } ] \leqslant \frac { 4 } { k } \leqslant \frac { 1 } { 5 0 } } \end{array}$ (for $k \geqslant 2 0 0 )$ ).

Case $D = Q$ . The instance is valid for A with true partition $\Pi _ { j }$ , so by its success guarantee and the two abort bounds, the error is at most $\begin{array} { r } { \frac { 1 } { 1 0 0 } + \frac { 2 } { 4 0 } < \frac { 1 } { 1 0 } } \end{array}$

Finally, $\textstyle \left\lceil { \frac { 1 6 0 T } { k } } \right\rceil \leqslant 4 \left\lceil { \frac { 4 0 T r } { k } } \right\rceil$ since $r \geqslant 1$ , and $s _ { D } \leqslant T .$ , so $s _ { D } \leqslant 4 s _ { Q }$ in each case of the two minima.

To control rounding in the budgets, couple the two hypotheses using the same pair $( P , Q )$ , allocation, internal randomness, and reference samples. If $T _ { j } = 0$ , the simulated observations and the tester’s answer coincide under both hypotheses, so at least one of the two answers is wrong. Since each error probability is at most $1 / 1 0$ , we must have $\operatorname* { P r } [ T _ { j } \geqslant 1 ] \geqslant 4 / 5$ . Hence $4 T / k \geqslant \mathbb { E } [ T _ { j } ] \geqslant 4 / 5$ , giving $T \geqslant k / 5$ . The ceilings in $s _ { Q } , s _ { D }$ can therefore be absorbed into the bounds in (17). □

## C.3.2 Application with One Known Distribution

Here we take the large-cluster distribution to be U and use the following existing average-case LFHT lower bound.

Lemma 18. (Gerber and Polyanskiy (2024, Proposition 13)) For $\varepsilon \in ( 0 , 1 / 8 ]$ , there exists a distribution $\Lambda _ { \mathrm { a l t } }$ over pairs $( U , Q )$ with $\mathrm { d _ { T V } } ( U , Q ) \geqslant \varepsilon$ satisfying the following. Any tester deciding $D = Q$ vs. $D = U$ with average error at most $1 / 1 0$ under each hypothesis, using $s _ { Q }$ samples from Q and $s _ { D }$ from D with $s _ { Q } \gtrsim s _ { D }$ must satisfy

$$
s _ { Q } \gtrsim \frac { \sqrt { n } } { \varepsilon ^ { 2 } } , \qquad s _ { Q } s _ { D } \gtrsim \frac { n } { \varepsilon ^ { 4 } } , \qquad s _ { D } \gtrsim \frac { 1 } { \varepsilon ^ { 2 } } .
$$

Theorem 19. For every $r \in \left[ 1 , \frac { k } { 2 } \right] , k \geqslant 2 0 0$ , and $\varepsilon \in ( 0 , 1 / 8 ]$ , with one distribution known,

$$
T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = \Omega \bigg ( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \bigg ) .
$$

Proof. Apply Lemma 17 with $P = U$ and $( U , Q ) \sim \Lambda _ { \mathrm { a l t } }$ , and generate the U-samples directly. Since $s _ { D } \leqslant 4 s _ { Q }$ Lemma 18 applies. The two required bounds follow by substituting (17):

$$
\frac { n } { \varepsilon ^ { 4 } } \lesssim s _ { Q } s _ { D } \lesssim \frac { T ^ { 2 } r } { k ^ { 2 } } , \qquad \frac { 1 } { \varepsilon ^ { 2 } } \lesssim s _ { D } \lesssim \frac { T } { k } .
$$

Together with Theorem 14, this proves the characterization with one known distribution in Section 6.

## C.3.3 Application with Both Distributions Unknown

The known-distribution lower bound still applies when both distributions are unknown. To obtain the additional term, we apply the same reduction to the hard pairs from Section 5.1.

Theorem 20. For every $r \in \left[ 1 , \frac { k } { 2 } \right] , k \geqslant 2 0 0$ , and $\varepsilon \in ( 0 , 1 / 8 ]$ , with both distributions unknown,

$$
T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = \Omega \left( \frac { k \sqrt { n } } { \sqrt { r } \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \right) ,
$$

and, when additionally $\begin{array} { r } { n \gtrsim \frac { k ^ { 2 } } { r \varepsilon ^ { 4 } } } \end{array}$

$$
T _ { \mathrm { n a } } ( n , k , r , \varepsilon ) = \Omega \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } \right) .
$$

Proof. The first bound is Theorem 19, since a clustering algorithm for two unknown distributions can also be used when one is known. For the additional bound, suppose $n \gtrsim k ^ { 2 } / ( r \varepsilon ^ { 4 } )$ and set

$$
T ^ { * } : = \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 1 / 3 } \varepsilon ^ { 4 / 3 } } .
$$

This regime gives $T ^ { * } \lesssim n ,$ so $T > n / 1 0 0$ already implies $T \gtrsim T ^ { * }$ . We henceforth assume that $T \leqslant n / 1 0 0$

Apply Lemma 17 with Λ the common distribution of $( P , Q )$ under $\Lambda _ { 0 } ( \cdot \mid E )$ and $\Lambda _ { 1 } ( \cdot \mid E )$ . The resulting tester has average error at most $1 / 1 0$ under each hypothesis and budgets $1 \leqslant s _ { Q } , s _ { D } \leqslant s _ { P } = T \leqslant n / 1 0 0$ . The average-case lower bound in Remark 12 therefore applies; using (17) and $r \geqslant 1$ , it gives

$$
\frac { n ^ { 2 } } { \varepsilon ^ { 4 } } \lesssim s _ { P } s _ { D } \big ( s _ { Q } + s _ { D } \big ) \lesssim T \cdot \frac { T } { k } \cdot \frac { T r } { k } = \frac { T ^ { 3 } r } { k ^ { 2 } } .
$$

Rearranging yields $T \gtrsim T ^ { * }$

Together with Theorem 16, this proves the characterization with both distributions unknown in Section 6.

## C.4 Proof of the Adaptivity Gap

Kumar et al. (2026, Table 1 and Appendix F) give the adaptive complexity

$$
T _ { \mathrm { a } } = \widetilde \Theta \left( \frac { k ^ { 2 / 3 } n ^ { 2 / 3 } } { r ^ { 2 / 3 } \varepsilon ^ { 4 / 3 } } + \frac { k \sqrt { n } } { r \varepsilon ^ { 2 } } + \frac { k ^ { 1 / 3 } n ^ { 2 / 3 } } { \varepsilon ^ { 4 / 3 } } + \frac { \sqrt { n k } } { \varepsilon ^ { 2 } } + \frac { k } { \varepsilon ^ { 2 } } \right) .
$$

The term $k / \varepsilon ^ { 2 }$ is shared. For the other terms, canceling the common factors in n and ε gives the ratios

$$
\begin{array} { r l r } & { } & { \frac { k ^ { 2 / 3 } / r ^ { 1 / 3 } } { \operatorname* { m a x } \{ k ^ { 2 / 3 } / r ^ { 2 / 3 } , k ^ { 1 / 3 } \} } = \operatorname* { m i n } \{ r ^ { 1 / 3 } , ( k / r ) ^ { 1 / 3 } \} \leqslant k ^ { 1 / 6 } , } \\ & { } & { \frac { k / \sqrt { r } } { \operatorname* { m a x } \{ k / r , \sqrt { k } \} } = \operatorname* { m i n } \{ \sqrt { r } , \sqrt { k / r } \} \leqslant k ^ { 1 / 4 } . } \end{array}
$$

Hence, $T _ { \mathrm { n a } } / T _ { \mathrm { a } } = \widetilde { O } ( k ^ { 1 / 4 } )$ . The example in Section 6 attains this bound up to logarithmic factors.

## D Extension to Multiple Hypotheses

The problem extends naturally to $l \geqslant 2$ candidates. Let $P _ { 1 } , \ldots , P _ { d }$ be distributions on $[ n ]$ , and let $D = P _ { j } ,$ for an unknown $j ^ { * } \in [ d ]$ with $\mathrm { d _ { T V } } ( D , P _ { i } ) \geqslant \varepsilon$ for every $i \neq j ^ { * }$ . Given $s _ { i }$ samples from each $P _ { i }$ and $s _ { D }$ samples from $D _ { ; }$ , the task is to output $j ^ { * }$ . Let $s _ { ( 1 ) } \leqslant s _ { ( 2 ) }$ be the two smallest of $s _ { 1 } , \ldots , s _ { d }$

Proposition 21. Let $n \geqslant 1 1 , \varepsilon \in ( 0 , 1 / 8 ]$ and $2 \leqslant d \leqslant n$ . The set $R _ { \mathsf { L F } } ^ { ( d ) } ( \varepsilon )$ of feasible sample budgets satisfies

$$
\begin{array} { c } { { R _ { \mathrm { L F } } ^ { ( d ) } ( \varepsilon ) \asymp _ { \log n } \left\{ ( s _ { 1 } , \dots , s _ { d } , s _ { D } ) : s _ { D } \geqslant \frac { 1 } { \varepsilon ^ { 2 } } , \quad s _ { ( 2 ) } \geqslant \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } , \quad s _ { D } \big ( s _ { ( 1 ) } + s _ { D } \big ) \geqslant \frac { n \alpha } { \varepsilon ^ { 4 } } \right\} , } } \\ { { w h e r e \alpha = \operatorname* { m a x } \Bigl \{ 1 , \frac { n } { \operatorname* { m a x } \{ s _ { ( 2 ) } , s _ { D } \} } \Bigr \} . } } \end{array}
$$

We observe that all but one candidate must be sampled at the rate $\frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } }$ , resembling identity testing, and only the least-sampled candidate trades against $s _ { D }$ in the way $Q$ does in Theorem $1 . \mathrm { { U p } }$ to the logarithmic factors in Proposition 21, the least-sampled candidate needs no samples when $\begin{array} { r } { s _ { D } \gtrsim \frac { \sqrt { n \alpha } } { \varepsilon ^ { 2 } } } \end{array}$ and is identified by elimination. The upper bound loses a factor $O ( \log d )$ in each sample size. Since d $\leqslant n ,$ this factor is hidden by $\asymp _ { \log n }$

Proof. For the upper bound, we separately apply Theorem 1 to every pair $( P _ { i } , P _ { j } )$ , with suitable amplification described below, and output the candidate that wins all of its comparisons. (If no such candidate exists, an error is declared.) The amplification modification is that we use $O ( \log d )$ independent repetitions to reduce the error probability from $1 / 3$ to at most $\frac { 1 } { 3 d }$ , with the repetitions reused across all pairwise comparisons. Candidate $j ^ { * }$ satisfies the promise of every comparison it takes part in, so by a union bound over these $d - 1$ comparisons it wins all of them, and every other candidate loses its comparison against $j ^ { * }$ . Comparisons between two wrong candidates are inconsequential, so they need not be included in the union bound. Since

$R _ { \mathsf { L F } } ( \varepsilon )$ is closed under increasing any sample size, every pair is feasible once the pair with sizes $\left( s _ { ( 2 ) } , s _ { ( 1 ) } \right)$ is, and the amplification costs a factor $O ( \log d )$ in each sample size.

For the lower bound, we reduce from the pairwise problem with sizes $( s _ { ( 2 ) } , s _ { ( 1 ) } , s _ { D } )$ on the domain $[ n - 1 ]$ Given such an instance $( P , Q , D )$ , place $P$ in the slot with $s _ { ( 2 ) }$ samples and $Q$ in the slot with $s _ { ( 1 ) }$ samples, and fill each of the remaining $d - 2$ slots with the point mass $\delta _ { n }$ , whose samples we can generate ourselves. Since D is supported on $[ n - 1 ]$ , we have $\mathrm { d } _ { \mathrm { T V } } ( D , \delta _ { n } ) = 1$ . Together with $\operatorname { d } _ { \mathrm { T V } } ( P , Q ) \geqslant \varepsilon$ , this makes the result a valid instance whose correct answer is the slot of D. The d-candidate tester therefore decides whether $D = P \ \mathrm { o r } \ D = Q$ , which gives a pairwise tester on $[ n - 1 ]$ with sizes $( s _ { ( 2 ) } , s _ { ( 1 ) } , s _ { D } )$ . The lower-bound direction of Theorem 1 on $[ n - 1 ]$ gives the three conditions with $n - 1$ in place of $n ,$ and since $\begin{array} { r } { n - 1 \geqslant \frac { n } { 2 } } \end{array}$ , this changes them only by constant factors. □