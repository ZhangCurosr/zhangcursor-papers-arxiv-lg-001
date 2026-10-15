# Digital Self-Interference Cancellation for In-Band Full-Duplex Systems Based on Transformer Neural Operators

Zhijie Su<sup>a</sup>, Shiquan Zhang<sup>a,∗</sup>, Qiaolin He<sup>a,∗</sup>

<sup>a</sup>Department of Mathematics, Sichuan University, Chengdu 610065, China

## Abstract

Digital-domain self-interference cancellation is essential in in-band full-duplex radios after passive suppression and analog cancellation. It is particularly challenging when the desired signal is strong relative to the residual selfinterference. In this regime, conventional iterative digital cancellers update their coeficients from desired-signal-contaminated received samples and may sufer substantial performance degradation. This paper reformulates digital-domain self-interference cancellation as an operator-learning problem. The proposed framework learns the temporal evolution of the residual selfinterference channel from historical observations and uses the learned operator together with the known local transmit signal to reconstruct the current residual self-interference. A Transformer neural operator is used to realize this framework. The resulting method is trained in a desired-signal-free setting and then applied in the presence of a desired signal. A continuity-based analysis characterizes the sensitivity of the learned operator to the resulting input perturbation. Numerical experiments in static and time-varying channels, including time-varying SIR conditions, demonstrate stable cancellation performance over a wide range of interference levels. At 30 dB SIR, the proposed method improves the cancellation performance by 5–15 dB over the considered conventional methods.

Keywords: in-band full-duplex, digital-domain self-interference cancellation, Transformer neural operator, operator learning, adaptive filtering

## 1. Introduction

In modern wireless communication, the demand for higher data rates, larger network capacity, and lower latency continues to grow. Conventional half-duplex systems separate transmission and reception in time or frequency and are increasingly unable to meet these requirements. Full-duplex communication enables simultaneous transmission and reception over the same frequency band [1, 2, 3, 4] and can potentially double spectral eficiency.

A major obstacle to full-duplex communication is self-interference [5, 6, 7, 8]. A portion of the local transmit signal leaks or is reflected into the local receiver together with the desired signal sent by the remote node. This SI can severely impair desired-signal decoding. Bharadia et al. [9] demonstrated a practical three-stage architecture that combines antenna isolation, analog cancellation, and digital cancellation. This paper focuses specifically on digital-domain cancellation of the residual SI that remains after passive suppression and analog cancellation.

Digital SI cancellation faces two primary challenges. First, SI paths caused by leakage and reflections can vary over time and must be tracked accurately. Second, the desired signal acts as interference when the residual SI channel is estimated from the received signal. When previous cancellation stages are efective, the desired signal may be significantly stronger than the residual SI, which makes subsequent digital cancellation dificult [10, 13].

LMS is one of the most widely used digital SI cancellation methods [11, 12, 13]. Related approaches include NLMS [14, 15], Sign-LMS [16], variable-step-size LMS [17, 18, 19], conjugate-gradient methods [20], and RLS [21]. Neural-network-based adaptive filtering has also been investigated [22, 23, 24], providing data-driven alternatives to conventional adaptive filters. Although these methods can be efective in some settings, their performance may deteriorate when the desired signal is strong relative to the SI component being estimated.

Nonlinear SI models provide one way to improve cancellation performance. Komatsu et al. [25] showed that nonlinear cancellation models can outperform conventional linear models in full-duplex radios. Memorypolynomial nonlinear cancellers have subsequently been proposed to model transmitter-chain nonlinearities [26, 27].

Existing methods mainly improve step-size selection, search directions, or nonlinear basis functions. They nevertheless estimate cancellation parameters by iterating on received observations. Consequently, their coeficient updates remain sensitive to desired-signal contamination, especially in high-SIR settings.

To address this limitation, we consider a data-driven framework that directly characterizes the relation among the local transmit signal, SI-channel evolution, and the resulting SI. Since SI is determined by the local transmit signal and the SI channel, rather than by the desired signal, a learned operator that predicts the current channel response from historical channel behavior can reduce reliance on contaminated instantaneous updates.

Operator learning has recently attracted attention for learning mappings between function spaces. Representative architectures include DeepONet [28, 29], Transformer-based neural operators [30, 31], physics-informed and convolutional neural operators [32, 33], and Fourier-based neural operators [34, 35].

A recent closely related study recasts digital SI calibration as an incontext regression problem [36]. A Transformer is pre-trained over a distribution of transmitter, channel, and front-end conditions. Given a small set of pilot-based calibration examples from the current operating state, it directly predicts the residual SI through a single forward pass, without reestimating filter coeficients or performing online gradient-based fine-tuning. The context examples and the query are assumed to belong to the same stationary radio state; thus, the method is designed for few-shot calibration within a quasi-static operating interval.

In contrast, this paper reformulates full-duplex SI cancellation as an operator-learning problem for temporal SI-channel evolution, where the Transformer neural operator is one realization of the operator-learning framework rather than the contribution by itself. The proposed formulation targets desired-signal contamination at high SIR and rapidly varying SI channels. It exploits ordered historical received observations and periodic desired-signalfree calibration data to fine-tune the learned operator, thereby continuously tracking SI channels whose variations are independent of those of the desiredsignal strength. The contributions are summarized as follows.

1. The SI cancellation problem is reformulated as an operator-learning problem that maps historical SI-channel behavior and the known transmit signal to the current SI component.

2. A Transformer-neural-operator-based SI cancellation method is developed within this framework. An input-perturbation analysis characterizes sensitivity to desired-signal contamination under a small desiredto-local-transmit-signal condition and attributes this property to the proposed clean-training/contaminated-inference formulation rather than to a particular attention mechanism alone.

3. Numerical experiments cover diferent SIR levels and static and timevarying SI channels. In the static-channel SIR study, the largest pairwise diference in cancellation performance among all tested SIR values is no greater than 0.05 dB; in the dynamic-channel experiments, the proposed method achieves a 5–15 dB gain over the considered conventional methods at 30 dB SIR.

The remainder of this paper is organized as follows. Section 2 describes the SI cancellation model. Section 3 presents the digital self-interference cancellation for in-band full-duplex systems based on transformer neural operators and its analysis. Section 4 reports the numerical experiments, and Section 5 concludes the paper.

## 2. Self-Interference Cancellation Model

## 2.1. Notation and statistical assumptions

Throughout this paper, $x ( n )$ and $s ( n )$ denote realizations of complexvalued discrete-time stochastic processes: $x ( n )$ is the known local transmit signal and $s ( n )$ is the desired signal transmitted by the remote node. The SI channel $h _ { n }$ is an unknown channel mapping that may vary with $n ;$ it is fixed in the static experiments and time varying in the dynamic experiments. Conditional on $x ( n )$ and $h _ { n }$ , the SI signal $l ( n )$ is deterministic. We assume that $x ( n )$ and $s ( n )$ have finite second moments and are uncorrelated,

$$
{ \mathbb E } \{ X ^ { * } ( n ) s ( n ) \} = { \bf 0 } ,\tag{1}
$$

where $X ( n ) = [ x ( n - N + 1 ) , x ( n - N + 2 ) , \ldots , x ( n ) ]$ is the local transmitsignal vector. The expectation in (1) is an ensemble expectation; it does not imply $X ^ { * } ( n ) s ( n ) = \mathbf { 0 }$ for every observed time instant. Receiver noise is omitted in the following notation for clarity, as in the original signal model; it can be included additively in $r ( n )$ without changing the argument.

## 2.2. Self-interference model and instantaneous optimization problems

The received signal is

$$
r ( n ) = s ( n ) + l ( n ) ,\tag{2}
$$

where the SI component is generated by the current and previous transmit samples,

$$
l ( n ) = h _ { n } { \bigl ( } X ( n ) { \bigr ) } , \qquad h _ { n } : \mathbb { C } ^ { N } \to \mathbb { C } .\tag{3}
$$

A linear canceller uses $d ( n ) = X ( n ) W$ , where $W \in \mathbb { C } ^ { N }$ is the tap-coeficient vector. The ideal cancellation error and the observable residual are, respectively,

$$
e r r ( n ; W ) = l ( n ) - X ( n ) W ,\tag{4}
$$

and

$$
e ( n ; W ) = r ( n ) - X ( n ) W = e r r ( n ; W ) + s ( n ) .\tag{5}
$$

The problem considered in this paper is explicitly a sequence of instantaneous optimization problems indexed by n. At time n, the ideal but unobservable problem is

$$
\mathcal { P } _ { n } ^ { \mathrm { S I } } : \qquad \operatorname* { m i n } _ { W } ~ | e r r ( n ; W ) | ^ { 2 } ,\tag{6}
$$

whereas the receiver can only form

$$
\mathcal { P } _ { n } ^ { \mathrm { r x } } : ~ \operatorname* { m i n } _ { W } ~ \lvert e ( n ; W ) \rvert ^ { 2 } .\tag{7}
$$

The formal normal equations associated with these two instantaneous leastsquares objectives are

$$
X ^ { \ast } ( n ) X ( n ) W = X ^ { \ast } ( n ) l ( n ) ,\tag{8}
$$

and

$$
X ^ { * } ( n ) X ( n ) W = X ^ { * } ( n ) r ( n ) = X ^ { * } ( n ) { \bigl ( } l ( n ) + s ( n ) { \bigr ) } ,\tag{9}
$$

respectively. Thus, (8)–(9) describe a family of conditions, one for each time n, rather than a single time-invariant optimization problem. The uniqueness of the solution to an instantaneous problem is determined by the rank of $X ^ { * } ( n ) X ( n )$ . In the present single-sample model, $X ( n ) \in \mathbb { C } ^ { 1 \times N }$ and therefore rank $( X ^ { \ast } ( n ) X ( n ) ) \leq 1$ ; hence, when $N > 1$ , the instantaneous minimizer is generally non-unique. These conditions are used here to expose the diference between the ideal and observable instantaneous objectives. The two conditions coincide at time n only if $X ^ { * } ( n ) s ( n ) = 0$ . Assumption (1) imposes only the ensemble-level condition ${ \mathbb E } \{ X ^ { * } ( n ) s ( n ) \} = { \bf 0 } ;$ it does not imply $X ^ { * } ( n ) s ( n ) = 0$ for each realization or at each time instant. Therefore, even when Assumption (1) is satisfied, this sample-wise orthogonality condition should not be expected for an arbitrary instantaneous problem $\mathcal { P } _ { n }$

## 2.3. Iterative solution of the optimization sequence

LMS and related methods generally do not iterate each instantaneous problem to convergence. This is because the observable instantaneous problem $\mathcal { P } _ { n } ^ { \mathrm { r x } }$ difers from the ideal problem $\mathcal { P } _ { n } ^ { \mathrm { S I } }$ through the desired-signal term. Iterating toward convergence of $\mathcal { P } _ { n } ^ { \mathrm { r x } }$ may produce a better fit to the received signal while producing a worse approximation of the SI component, especially at high SIR. Instead, these methods process the sequence $\{ \mathcal P _ { n } ^ { \mathrm { r x } } \}$ recursively. At time n, the initial coeficient vector is the value obtained after processing the preceding problem, namely $W ( n )$ . The current residual is

$$
e ( n ) = r ( n ) - X ( n ) W ( n ) ,\tag{10}
$$

and LMS takes one stochastic-gradient step for the current objective:

$$
W ( n + 1 ) = W ( n ) + \mu e ( n ) X ^ { * } ( n ) ,\tag{11}
$$

where $\mu$ is the step size. Thus, $W ( n + 1 )$ is both the updated coeficient vector for the current problem and the warm start for the next problem $\mathcal { P } _ { n + 1 } ^ { \mathrm { r x } }$ . NLMS, variable-step-size LMS, CG, and RLS follow the same sequential interpretation but use diferent normalizations, step-size rules, or search directions [14, 15, 17, 18, 19, 20, 21].

For nonlinear SI cancellation, the same sequential formulation is used after replacing $X ( n )$ with a nonlinear feature vector. For example, a memorypolynomial canceller is

$$
\widetilde d ( n ) = \sum _ { i = 1 } ^ { N } \sum _ { k = 0 } ^ { p } W _ { i , k } | x ( n - N + i ) | ^ { 2 k } x ( n - N + i ) ,\tag{12}
$$

and its coeficients are updated recursively from the current residual. The desired signal therefore perturbs the update of both linear and nonlinear iterative cancellers, especially when SIR is high. This motivates the operatorlearning formulation in Section 3.

## 3. Self-Interference Cancellation Method Based on Transformer Neural Operators

This section develops the proposed operator-learning formulation for digital SI cancellation. The main contribution is the reformulation of the estimation problem: instead of updating cancellation coeficients directly from a desired-signal-contaminated instantaneous error, the method learns the temporal evolution of the SI channel from calibration data and uses this learned evolution to reconstruct the current SI. A Transformer neural operator is employed as a realization of this operator-learning framework. The following subsections present the formulation, the network implementation, the training and application procedures, and a stability analysis.

## 3.1. Reformulation of self-interference cancellation problem

The SI signal is determined by the local transmit signal and the SI channel, whereas the desired signal is generated by the remote transmitter. This separation motivates an operator-learning formulation: if the temporal evolution of the SI channel can be characterized, the current SI can be reconstructed without using an instantaneous coeficient update driven by the desired-signal-contaminated residual.

Let $h _ { n }$ denote the SI-channel mapping at time n. Exploiting temporal dependence in channel evolution, we introduce an operator G<sup>¯</sup> that maps a sequence of historical channel mappings to the current channel mapping:

$$
\bar { G } \left( h _ { n - 1 } , h _ { n - 2 } , \ldots , h _ { n - M } \right) = h _ { n } ,\tag{13}
$$

where M is the number of preceding channel realizations used to characterize the channel dynamics. Thus, G<sup>¯</sup> represents the temporal evolution law of the SI channel in a function space.

Applying the predicted current channel mapping to the known transmit vector gives the operator-learning representation of SI:

$$
l ( n ) = \bar { G } ( h _ { n - 1 } , h _ { n - 2 } , \ldots , h _ { n - M } ) \big ( \pmb { x } _ { n } \big ) ,\tag{14}
$$

where $\bar { G } ( h _ { n - 1 } , h _ { n - 2 } , \ldots , h _ { n - M } )$ represents the channel mapping at the current time instant. Learning G<sup>¯</sup> therefore reconstructs SI from historical SIchannel behavior and the known local transmit signal. Unlike few-shot calibration methods that treat context samples from a stationary calibration interval as exchangeable [36], G<sup>¯</sup> is an ordered temporal mapping from preceding channel realizations to the current channel realization. This is the central distinction between the proposed framework and conventional errordriven iterative cancellation.

Let $G _ { \theta }$ denote a Transformer neural operator with parameters θ. It is used to approximate $\bar { G }$

$$
\theta ^ { * } = \operatorname* { m i n } _ { \theta \in \Theta } \parallel \bar { G } - G _ { \theta } \parallel ,\tag{15}
$$

where $\Theta$ is the parameter space. For a signal segment of length $T ,$ , the empirical training loss is

$$
J ( \theta ) = \frac { 1 } { T } \sum _ { n = 1 } ^ { T } \mid l ( n ) - d _ { \theta } ( n ) \mid ^ { 2 } .\tag{16}
$$

where $d _ { \theta } ( n )$ is the network output:

$$
d _ { \theta } ( n ) = G _ { \theta } \left( h _ { n - 1 } , h _ { n - 2 } , \ldots , h _ { n - M } \right) ( { \pmb x } _ { n } ) .\tag{17}
$$

## 3.2. Transformer Neural Operators

We use a Transformer neural operator based on self-attention. It maps the input function space $A ( D ; \mathbb { R } ^ { d _ { a } } )$ to the output function space $U ( D ; \mathbb { R } ^ { d _ { u } } )$ $G : A ( D ; \mathbb { R } ^ { d a } ) \times \Theta \longrightarrow U ( D ; \mathbb { R } ^ { d u } )$ . The target operator is $G ^ { + } : A ( D ; \mathbb { R } ^ { d _ { a } } ) \longrightarrow$ $U ( D ; \mathbb { R } ^ { d _ { u } } )$ . The neural-operator realization has the form

$$
G ( a , \theta ) = ( T _ { o u t } \circ E _ { L } \circ T _ { i n } ) ( a , \theta ) ,\tag{18}
$$

for $a \in A ( D ; \mathbb { R } ^ { d _ { a } } )$ and $\theta \in \Theta$

In $( 1 8 ) , T _ { i n } : A ( D ; \mathbb { R } ^ { d _ { a } } ) \times \Theta  V ( D ; \mathbb { R } ^ { d _ { v } } )$ and $T _ { o u t } : V ( D ; \mathbb { R } ^ { d _ { v } } ) \times \Theta $ $U ( D ; \mathbb { R } ^ { d _ { u } } )$ are input and output lifting operators on an embedded function space V. The operator $E _ { L } : V ( D ; \mathbb { R } ^ { d _ { v } } ) \times \Theta \to V ( D ; \mathbb { R } ^ { d _ { v } } )$ is the neuraloperator analogue of an L-layer Transformer encoder [30]. For brevity, write $T _ { i n } ( a ) = T _ { i n } ( a , \theta )$ and $T _ { o u t } ( a ) = T _ { o u t } ( a , \theta )$ . With $v ^ { 0 } = T _ { i n } ( a )$ , each encoder layer is

$$
v ^ { l - 1 } = W _ { 1 } v ^ { l - 1 } + A _ { M u l t i H e a d } v ^ { l - 1 } ,\tag{19}
$$

$$
v ^ { l - 1 } = F _ { l a y e r n o r m } ( v ^ { l - 1 } ) ,\tag{20}
$$

$$
v ^ { l - 1 } = W _ { 2 } v ^ { l - 1 } + F _ { N N } ( v ^ { l - 1 } ) ,\tag{21}
$$

$$
v ^ { l } = F _ { l a y e r n o r m } ( v ^ { l - 1 } ) ,\tag{22}
$$

for $l = 1 , 2 , \ldots , L$ , with $E _ { L } ( v ^ { 0 } ) = v ^ { L }$ . The operators $W _ { 1 } , \ W _ { 2 } , \ A _ { M u l t i H e a d } .$ $F _ { l a y e r n o r m }$ , and $F _ { N N }$ have independent learnable parameters. The multi-head self-attention operator $A _ { M u l t i H e a d } : V  V$ is

$$
\begin{array} { r } { ( A _ { M u l t i H e a d } ( v ) ) ( x ) = W _ { M u l t i H e a d } ( A ( v , \theta _ { 1 } ) ( x ) , } \\ { . . . , A ( v , \theta _ { H } ) ( x ) ) , } \end{array}\tag{23}
$$

for $v \in V$ , where H is the number of attention heads and $W _ { M u l t i H e a d } : V  V$ is a learned linear transformation. The single-head attention map A is

$$
\begin{array} { r } { A ( v , \theta _ { i } ) ( x ) = \frac { \int _ { D } e x p ( < Q _ { i } v ( x ) , K _ { i } v ( y ) > ) V _ { i } v ( y ) d y } { \int _ { D } e x p ( < Q _ { i } v ( x ) , K _ { i } v ( s ) > ) d s } } \\ { \quad \forall x \in D , i = 1 , . . . , H . } \end{array}\tag{24}
$$

Here, $\langle \cdot , \cdot \rangle$ denotes the Euclidean inner product, and $Q , K$ , and V are learned linear transformations. The maps $W _ { 1 } , W _ { 2 } : V \to V$ are linear operators, $F _ { N N }$ is a single-hidden-layer fully connected network, and $F _ { l a y e r n o r m } : V  V$ is layer normalization:

$$
( F _ { l a y e r n o r m } ( v ; \gamma , \beta ) ( x ) ) _ { k } = \gamma _ { k } \cdot \frac { ( v ( x ) ) _ { k } - m ( v ( x ) ) } { \sqrt { \sigma ^ { 2 } ( v ( x ) ) + \varepsilon } } + \beta _ { k } ,\tag{25}
$$

for $k = 1 , . . . , d v , \forall v \in V .$ and $x \in D \subset R ^ { d }$ . Here, $( \cdot ) _ { k }$ refers to the k-th component of a vector, $\varepsilon \in R ^ { + }$ is a fixed parameter, and $\gamma _ { k } , \beta _ { k } \in \mathcal Ḋ R Ḍ$ are learnable parameters. The definitions of m, σ are as follows:

$$
\begin{array} { l } { { \displaystyle m ( y ) = \frac { 1 } { d v } \cdot \parallel y \parallel _ { 1 } } , } \\ { { \displaystyle \sigma ^ { 2 } ( y ) = \frac { 1 } { d v } \sum _ { k = 1 } ^ { d v } ( y _ { k } - m ( y ) ) ^ { 2 } } , } \end{array}\tag{26}
$$

for any $y \in R ^ { d v }$ . The entire Transformer Neural Operator can be illustrated in Fig. 1, where $W _ { i n }$ and $W _ { o u t }$ respectively denote the operator $T _ { i n } ( a ( x ) )$ ) and $T _ { o u t } ( v ^ { L } )$

![](images/73be97db7b29fed2cca55f7e8d0e2ab5d0f4034b981b4ecc019e03ca43fd7549.jpg)  
Figure 1: Transformer Neural Operator

## 3.3. Method setting

According to (13), the input of $\bar { G }$ is $h _ { i } , i = n - M , \ldots , n - 1$ , and its output is $h _ { n }$ . Information about $h _ { i }$ is contained in the input–output relation between $X ( i )$ and $l ( i )$ , while $X ( n )$ is the argument of the current mapping $h _ { n }$ Hence, the input of G<sup>¯</sup> is represented by $( X ( n ) , X ( n - 1 ) , \ldots , X ( n - M ) , l ( n -$ $1 ) , \ldots , l ( n - M ) )$ . Since $X ( n ) = ( x ( n - N + 1 ) , \ldots , x ( n ) )$ , the input of $G _ { \theta }$ can be written as $( x ( n ) , \ldots , x ( n - M - N + 1 ) , l ( n - 1 ) , \ldots , l ( n - M ) )$ . The ordering of these historical observations is retained to characterize temporal SI-channel evolution rather than only a stationary calibration context.

The neural operator is trained ofline in a desired-signal-free setting, where $r ( n ) = l ( n )$ and the SI labels are available. During online operation, $r ( n )$ may contain the desired signal and $l ( n )$ is unavailable. The trained operator $G _ { \theta ^ { \ast } }$ ∗ therefore uses $( x ( n ) , \ldots , x ( n - M - N + 1 ) , r ( n - 1 ) , \ldots , r ( n - M ) )$ as its input. Algorithms 1 and 2 summarize the procedures for static and time-varying channels, respectively.

The diference between the static and dynamic settings is whether the SI channel changes over time. In the static setting, the ofline-trained operator can be applied directly online. In the dynamic setting, channel variation can create a mismatch between the training and test distributions and may reduce prediction accuracy.

Algorithm 1 Static-channel scenario.   
Offline Training:   
Step 1: Use 3,000 samples for training. The input is $( x ( n ) , \ldots , x ( n -$   
$M - N + 1 ) , l ( n - 1 ) , \dots , l ( n - M ) )$ , and the target is $l ( n )$   
Step 2: Train for 60,000 iterations using the MSE loss and the Adam   
optimizer.   
Step 3: Save the trained model as $G _ { \theta ^ { * } }$   
Online Test:   
Step 4: Apply $G _ { \theta ^ { * } } \mathrm { ~ t o ~ } ( x ( n ) , \dots , x ( n - M - N + 1 ) , r ( n - 1 ) , \dots , r ( n -$   
$M ) )$   
return $d ( n )$

To track rapidly varying channels, the model is periodically updated using newly collected observations. The update interval depends on the severity of the channel variation: faster dynamics generally require more frequent updates. The dynamic data set used here is deliberately challenging: the SI channel undergoes an abrupt change approximately every 500 samples, while it varies gradually within each 500-sample interval. The operator is retrained after every 20,000 received samples. Specifically, a pilot-like, desired-signalfree calibration interval is used to acquire 1,000 newly labelled SI samples $l ( n )$ , and the operator is fine-tuned using these samples, as summarized in Algorithm 2. This procedure targets continuous tracking under nonstationary channel evolution, rather than few-shot prediction within a single quasi-static calibration interval.

## 3.4. Analysis of input-perturbation sensitivity to desired-signal contamination

The proposed framework is trained with inputs that do not contain the desired signal. Its online input is formed by replacing the unobservable historical SI samples with received samples, which include the desired signal. The following result characterizes the sensitivity of the learned operator to this replacement. Let $T _ { 0 } = ( x ( n ) , \ldots , x ( n - M - N + 1 ) , l ( n - 1 ) , \ldots , l ( n - M ) )$ be the clean input and $T _ { 1 } = ( x ( n ) , \ldots , x ( n - M - N + 1 ) , r ( n - 1 ) , \ldots , r ( n - M ) )$ be the online input. For a given transmit sequence, define $\| \cdot \| _ { x } = \|$ $\| / \operatorname* { m e a n } \{ | x ( n ) | \}$ . Assume that $G _ { \theta ^ { \ast } }$ ∗ is continuous: for every $\varepsilon > 0$ , there exists $\delta ( \varepsilon ) > 0$ such that $\| T _ { 0 } - T _ { 1 } \| _ { x } < \delta ( \varepsilon )$ implies $| G _ { \theta ^ { * } } ( T _ { 0 } ) - G _ { \theta ^ { * } } ( T _ { 1 } ) | < \varepsilon$ We then obtain the following input-perturbation sensitivity result.

Algorithm 2 Time-varying-channel scenario.   
Offline Training:   
Step 1: Same as Step 1 of Algorithm 1.   
Step 2: Train for 100,000 iterations using the MSE loss and the Adam   
optimizer.   
Step 3: Save the trained model as $G _ { \theta ^ { * } }$   
Online Test:   
Step 4: Load the ofline-trained model $G _ { \theta ^ { * } }$   
Step 5: At a pilot-like, desired-signal-free calibration interval, acquire   
1,000 labelled SI samples $l ( n )$ and fine-tune the model for 600 iterations   
using the settings in Steps 1 and 2.   
Step 6: Use the updated model to predict the following 20,000 samples   
with input $( x ( n ) , \ldots , x ( n - M - N + 1 ) , r ( n - 1 ) , \ldots , r ( n - M ) )$   
return $d ( n )$   
Step 7: Use the next 1,000 samples and return to Step 5.

Theorem 1 (Sensitivity to desired-signal contamination): Assume that $G _ { \theta ^ { * } }$ is continuous. For every $\varepsilon > 0$ , there exists $\delta ( \varepsilon ) > 0$ such that, if the desired signal $s ( n )$ and transmit signal $x ( n )$ satisfy

$$
\frac { m a x \parallel s ( n ) \parallel } { m e a n \parallel x ( n ) \parallel } < \frac { \delta ( \varepsilon ) } { M } ,\tag{27}
$$

then

$$
\mid G _ { \theta ^ { \ast } } ( T _ { 0 } ) - G _ { \theta ^ { \ast } } ( T _ { 1 } ) \mid < \varepsilon .\tag{28}
$$

Proof: Direct calculation gives

$$
\begin{array} { r l } & { \parallel T _ { 0 } - T _ { 1 } \parallel _ { x } = \parallel \left( 0 , . . . , 0 , s ( n - 1 ) , . . . s ( n - M ) \right) \parallel _ { x } } \\ & { \qquad \leq M \frac { m a x \parallel s ( n ) \parallel } { m e a n \parallel x ( n ) \parallel } } \\ & { \qquad < \delta ( \varepsilon ) . } \end{array}\tag{29}
$$

Continuity of $G _ { \theta ^ { * } }$ yields (28).

In practical full-duplex systems, the desired signal originates from the remote transmitter. It may exceed the residual SI level, yet it is typically much weaker than the local transmit signal; thus, condition (27) is meaningful in the intended operating regime, including the high-SIR cases considered in the experiments. The proof characterizes the sensitivity to the replacement of clean historical SI samples by received samples under condition (27); it neither establishes convergence nor guarantees performance for an arbitrary SIR. High SIR concerns the ratio between the desired signal and residual SI, whereas condition (27) controls the desired signal relative to the local transmit signal. The proof therefore provides a suficient condition under which high-SIR operation may remain stable, while the high-SIR cancellation gains are numerical findings. This property follows from the continuity of the learned operator and the proposed clean-training/contaminated-inference input structure, rather than from a particular attention mechanism alone.

## 4. Numerical Experiments

We evaluate the proposed method under diferent desired-signal contamination levels to assess its cancellation performance and robustness. The desired-signal-to-SI ratio (SIR) and normalized mean-squared error (NMSE), both in dB, are defined as

$$
\begin{array} { r } { \mathrm { S I R } = 2 0 \log _ { 1 0 } \frac { \left\| s \right\| _ { 2 } } { \left\| l \right\| _ { 2 } } , \ } \\ { \mathrm { N M S E } = 2 0 \log _ { 1 0 } \frac { \left\| e r r \right\| _ { 2 } } { \left\| l \right\| _ { 2 } } . } \end{array}\tag{30}
$$

Here, s is the vector formed by $s ( n )$ over the evaluation interval; l and err are defined analogously for the SI signal $l ( n )$ and cancellation error $e r r ( n )$ 2 respectively. As discussed in Section 2, the desired signal $s ( n )$ contaminates SI estimation. Therefore, the desired-signal-free scenario considered below means $s ( n ) = 0 ;$ it is not a 0-dB-SIR scenario.

## 4.1. Static-channel results

The static scenario considers a time-invariant SI channel. In this experiment, the channel vector is $( 0 , 0 , \ldots , 0 . 0 0 1 )$ , so that $l ( n ) = 0 . 0 0 1 x ( n )$ . The data are synthetically generated and processed using a continuous sliding window. The first 3,000 samples are used for ofline training, and the trained network is evaluated on disjoint data segments of diferent lengths.

We use a Transformer neural operator with ReLU activation and the Adam optimizer. Training is performed for 60,000 iterations. Because the network operates on real-valued inputs, each complex sample is represented by its real and imaginary parts; the output similarly represents the real and imaginary parts of $d ( n )$ . The input dimension is $2 ( 2 M + N )$ and the output dimension is two. The hyperparameters are listed in Table 1, and the staticchannel procedure is given in Algorithm 1.

Table 1: Hyperparameters for a static scene
<table><tr><td>Parameters</td><td>Value</td><td>Explanation</td></tr><tr><td>emb</td><td>256</td><td>the number of neurons in each layer of  $\overline { { E _ { L } } }$ </td></tr><tr><td>hidden</td><td>512</td><td>the hidden layer size of the FNN</td></tr><tr><td>heads</td><td>2</td><td>the number of attention heads</td></tr><tr><td>layers</td><td>1</td><td>the number of layers in  $E _ { L }$ </td></tr><tr><td>N</td><td>10</td><td>the tap coefficient</td></tr><tr><td>M</td><td>10</td><td>the truncation number</td></tr></table>

• number of iterations: 60,000

• learning rate: iterations $0 { - } 2 0 { , } 0 0 0 { , } 2 { \times } 1 0 ^ { - 4 }$ ; iterations 20,000–40,000, $4 \times 1 0 ^ { - 5 }$ ; iterations 40,000–60,000, $8 \times 1 0 ^ { - 6 }$

We evaluate NMSE under diferent desired-signal contamination levels and over evaluation segments of diferent lengths. The results are reported in Table 2. For compactness, ‘Free’ denotes the desired-signal-free scenario.

Table 2: Results of NMSE in static scenario
<table><tr><td>data</td><td>Free</td><td>-10db</td><td>10db</td><td>30db</td></tr><tr><td>training set</td><td>-36.75dB</td><td>-36.75dB</td><td>-36.75dB</td><td>-36.75dB</td></tr><tr><td>3000 test set</td><td>-16.35dB</td><td>-16.35dB</td><td>-16.35dB</td><td>-16.30dB</td></tr><tr><td>30,000 test set</td><td>-14.76dB</td><td>-14.76dB</td><td>-14.76dB</td><td>-14.72dB</td></tr><tr><td>100,000 test set</td><td>-13.51dB</td><td>-13.51dB</td><td>-13.51dB</td><td>-13.48dB</td></tr><tr><td>300,000 test set</td><td>-15.16dB</td><td>-15.16dB</td><td>-15.16dB</td><td>-15.13dB</td></tr></table>

Table 2 shows that varying the SIR has little efect on the proposed method in this static experiment. The largest displayed diference between the desired-signal-free and 30-dB-SIR cases is no greater than 0.05 dB. Thus, the proposed digital-domain SI canceller achieves performance comparable to that in the desired-signal-free case. This numerical observation is consistent with the input-perturbation sensitivity analysis in Section 3.4 under condition (27).

## 4.2. Time-varying-channel results

We next consider a dynamic SI channel. The data set is deliberately challenging: the channel undergoes an abrupt change approximately every 500 samples and varies gradually within each 500-sample interval. An initial network is trained using 3,000 desired-signal-free samples and then saved. After every 20,000 received samples, a pilot-like desired-signal-free calibration interval provides 1,000 labelled SI samples for 600 fine-tuning iterations. The updated operator is then used to predict the subsequent 20,000 samples. This calibration–prediction cycle is repeated over 12 million samples, following Algorithm 2. The experiment therefore evaluates continuous SI tracking under nonstationary channel evolution, rather than few-shot prediction within a stationary calibration interval.

For the initial network trained with 3,000 samples, we compare diferent hyperparameter configurations. For model selection, two disjoint held-out segments of 3,000 and 30,000 samples are used as validation data, while a 60,000-sample held-out segment is retained for final testing. Table 3 lists the configurations, and Table 4 reports the corresponding results.

Table 3: Diferent hyperparameters of network
<table><tr><td>Model</td><td>emb</td><td>hidden</td><td>heads</td><td>layers N M</td></tr><tr><td>M1</td><td>64</td><td>128</td><td>4 4</td><td>10 10</td></tr><tr><td>M2 128</td><td>256</td><td>4</td><td>4</td><td>10 10</td></tr><tr><td>M3</td><td>128</td><td>256</td><td>8</td><td>5 10 10</td></tr><tr><td>M4 32</td><td>64</td><td>4</td><td>3</td><td>10 10</td></tr><tr><td>M5 128</td><td>256</td><td>4</td><td>2</td><td>10 10</td></tr><tr><td>M6 128</td><td>256</td><td>4</td><td>1</td><td>10 10</td></tr><tr><td>M7 128</td><td>256</td><td>4</td><td>1</td><td>20 10</td></tr><tr><td>M8 128</td><td>256</td><td>4</td><td>1 30</td><td>10</td></tr><tr><td>M9 128</td><td>256</td><td>4</td><td>1 40</td><td>10</td></tr></table>

• number of iterations: 100,000

• learning rate: iterations $) - 2 0 , 0 0 0 , 2 \times 1 0 ^ { - 4 } ; 2 0 , 0 0 0 - 4 0 , 0 0 0 , 4 \times 1 0 ^ { - 5 } ;$ $4 0 , 0 0 0 - 6 0 , 0 0 0 , 8 \times 1 0 ^ { - 6 } ; 6 0 , 0 0 0 - 8 0 , 0 0 0 , 2 \times 1 0 ^ { - 6 } ; 8 0 , 0 0 0 - 1 0 0 , 0 0 0 , 4 \times 1 0 ^ { - 6 }$ 10<sup>−7</sup>.

Table 4: Results of NMSE under diferent hyperparameters
<table><tr><td>Model</td><td>Training</td><td>Validation (3k)</td><td>Validation (30k)</td><td>Test (60k)</td></tr><tr><td>M1</td><td>-15.05dB</td><td>-14.09dB</td><td>-13.08dB</td><td>-10.90dB</td></tr><tr><td>M2</td><td>-16.31dB</td><td>-15.00dB</td><td>-12.81dB</td><td>-11.09dB</td></tr><tr><td>M3</td><td>-15.10dB</td><td>-14.51dB</td><td>-12.70dB</td><td>-10.70dB</td></tr><tr><td>M4</td><td>-13.40dB</td><td>-11.89dB</td><td>-11.70dB</td><td>-10.18dB</td></tr><tr><td>M5</td><td>-18.51dB</td><td>-15.10dB</td><td>-14.10dB</td><td>-11.85dB</td></tr><tr><td>M6</td><td>-15.56dB</td><td>-14.32dB</td><td>-13.45dB</td><td>-11.17dB</td></tr><tr><td>M7</td><td>-17.24dB</td><td>-15.80dB</td><td>-14.31dB</td><td>-11.62dB</td></tr><tr><td>M8</td><td>-18.25dB</td><td>-16.15dB</td><td>-14.58dB</td><td>-11.65dB</td></tr><tr><td>M9</td><td>-17.26dB</td><td>-15.40dB</td><td>-13.75dB</td><td>-11.30dB</td></tr></table>

As shown in Table 4, M8 achieves the lowest NMSE on the 3k and 30k validation segments and is therefore selected as the initial model for the remaining dynamic experiments. The 60k test segment is not used for this selection. During the 12-million-sample evaluation, the operator is fine-tuned after every 20,000 received samples using 1,000 labelled SI samples from the pilot-like calibration interval for 600 iterations. The learning rate is $8 \times 1 0 ^ { - 6 }$ for iterations 1–400 and $4 \times 1 0 ^ { - 6 }$ for iterations 401–600. The procedure is specified in Algorithm 2. This update schedule is evaluated under the highly dynamic channel setting described above.

We compare the proposed Transformer-neural-operator-based digital SI canceller with LMS, NLMS[14, 15], CG[20], RBFNN[22], and nonlinear LMS (NL-LMS) [26, 27] under diferent SIR conditions. RBFNN is not included in the desired-signal-free scenario because it requires prior estimation of the probability density function (PDF) of the desired signal. In the legends, b and $\mu$ denote step-size parameters, whereas d denotes the CG weight parameter. For each baseline, an approximate parameter range is first identified and then searched on an exponential base-2 grid; the best-performing parameter in this search is reported. For example, the RBFNN parameter b is searched from $2 ^ { - 5 0 } \ \mathrm { t o } \ 2 ^ { - 4 0 }$ . The baseline grid search is performed retrospectively over the complete experimental trajectory, so the displayed baseline curves represent best-case ofline-tuned performance rather than an online parameter-selection procedure. In contrast, the proposed method uses the validation segments described above for model selection. This setting is favorable to the baselines; therefore, any observed advantage of the proposed method is conservative.

(1) Desired-signal-free scenario

![](images/5ad7a8f479ad0f49d3899baabe6e86d82f600e6851ecc0576c44026bcabd74bf.jpg)  
Figure 2: Desired-signal-free NMSE

![](images/75bb5f67bc9924280db6cd32e8ec41676952474f03e043d66bbfa42b9092b426.jpg)  
Figure 3: Desired-signal-free PSD

Figures 2–3 show that, in the desired-signal-free static scenario, the proposed method is initially slightly inferior to the LMS-family methods. The periodic pilot-assisted updates considered in the subsequent dynamic-channel experiment improve the operator prediction as the model is adapted to new channel observations.

(2) −10 dB SIR scenario

![](images/c59ab041ce7786c0ccb3a39cc664d74ab6913ab19de4a043f1c4c0d91a98651f.jpg)  
Figure 4: NMSE of -10 dB SIR

![](images/2d4e85e078d7d9f60d184f9f8a05c73a614120a647c1a3b0491c285488aef6cf.jpg)  
Figure 5: PSD of -10 dB SIR

As shown in Figs. 4–5, at −10 dB SIR the LMS-family methods degrade slightly, whereas the proposed method changes little in this experiment.

## (3) 10 dB SIR scenario

![](images/5cbfab2c152ff14609093336907286055ff929c533f146bbfe23232eb03df0e9.jpg)  
Figure 6: NMSE of 10 dB SIR

![](images/f24ba4fb7ae3e53793ee8979cecc2a3f9404502a5abc109df4a2491ed56050ef.jpg)  
Figure 7: PSD of 10 dB SIR

Figures 6–7 indicate that, as SIR increases to 10 dB, the performance of LMS further deteriorates and NLMS exhibits a similar but more pronounced degradation. In contrast, the proposed method maintains stable performance and shows less oscillatory behavior.

(4) 30 dB SIR scenario

![](images/9e4942937732479e43cf37fa2cea96a1f9b69d39ddbe636541042cefdaadc498.jpg)  
Figure 8: NMSE of 30 dB SIR

![](images/76d5e9c9fd88eea592c029c65dc7e0cc22f15107c976bf07270b2b1605b49864.jpg)  
Figure 9: PSD of 30 dB SIR

Figures 8–9 illustrate the results at 30 dB SIR. The iterative baseline methods exhibit opposite cancellation, meaning that their residual power after cancellation exceeds the uncancelled SI power. In contrast, the proposed method remains stable and improves NMSE by 5–15 dB relative to the considered baseline methods.

## (5) 100 dB SIR scenario

We also consider an extreme 100-dB-SIR case to assess whether the proposed method remains robust to severe desired-signal contamination. The results are shown in Figs. 10–11.

![](images/75148e692658c07dce18bf88f2e951ce353e3909136b4204008c8da6b507f766.jpg)  
Figure 10: NMSE of 100 dB SIR

![](images/f18f63fa72fc316405d78d345726b0703a2f9c5b6de86bca160f524b08f1f2df.jpg)  
Figure 11: PSD of 100 dB SIR

Because all compared baseline algorithms exhibit opposite cancellation over the displayed interval, their curves are omitted from Figs. 10–11. Even under this high-SIR condition, the proposed method maintains performance comparable to that in the desired-signal-free scenario.

## 4.3. Results with Independently Time-Varying SI Channel and Desired-Signal Strength

In this experiment, the SI channel and desired-signal strength vary independently over time. The desired-signal condition switches among four values (desired-signal-free, −10 dB, 10 dB, and 30 dB SIR) every 20,000 samples. We evaluate whether the proposed method retains stable cancellation performance under these independently time-varying conditions. Model M8 is used for initialization. The data selection, input representation, number of training iterations, and learning-rate schedule for periodic updates are the same as in the preceding dynamic-channel experiment.

RBFNN is omitted because it requires prior estimation of the desiredsignal PDF, which changes rapidly in this setting. The results are shown in Figs. 12–13.

![](images/7c3530e50484480776829b21fd11e9276b134b45ef27415087ede0805e162fe7.jpg)  
Figure 12: NMSE of varying SIR

![](images/2c968f07379f45ad80b697409b4b1d0a941279520e54b0eb0560c59ffd7d58e2.jpg)  
Figure 13: PSD of varying SIR

In this scenario, LMS-type methods achieve their best high-SIR performance only when their parameters are tuned for the strongest desired-signal contamination. However, this configuration impairs channel tracking at lower SIR, degrading overall performance when the SI channel and desired-signal strength vary independently over time. In contrast, the proposed method maintains stable performance in this independently time-varying environment.

## 5. Conclusion

This paper presented a digital-domain residual-SI cancellation method for in-band full-duplex systems based on Transformer neural operators. The method operates after passive suppression and analog cancellation. Its main idea is to reformulate digital-domain SI cancellation as an operator-learning problem: the temporal evolution of the residual SI channel is learned from historical observations and then used to reconstruct the current residual SI. This formulation reduces the reliance on the desired-signal-contaminated instantaneous updates used by conventional iterative cancellers. Theorem 1 characterizes the input-perturbation sensitivity of the resulting input–output structure to desired-signal contamination under the stated desired-to-localtransmit-signal condition.

In the reported experiments, the proposed method performs comparably to conventional LMS-type methods at low SIR and remains stable as SIR increases. In high-SIR settings, including the 100-dB case, the method maintains performance close to that observed in the desired-signal-free setting, whereas the considered iterative methods deteriorate. The dynamic experiments further show that the method retains this behavior when the desired-signal level and the SI channel vary independently over time.

Future work will investigate alternative operator-learning architectures and more eficient adaptation mechanisms for rapidly time-varying SI channels.

## Data availability

The data supporting the findings of this study were provided by a collaborating laboratory and are not publicly available because of confidentiality restrictions.

## References

[1] L. Mohammadi, Z. Mobini, D. Galappaththige, and C. Tellambura, “A Comprehensive Survey on Full-Duplex Communication: Current Solutions, Future Trends, and Open Issues,” in IEEE Commun. Surveys Tuts., vol. 25, no. 4, pp. 2190–2244, 4th Quart. 2023.

[2] A. Sabharwal, P. Schniter, D. Guo, D. W. Bliss, S. Rangarajan, and R. Wichman, “In-Band Full-Duplex Wireless: Challenges and Opportunities,” in IEEE J. Sel. Areas Commun., vol. 32, no. 9, pp. 1637–1652, Sep 2014.

[3] B. Smida, A. Sabharwal, G. Fodor, G. C. Alexandropoulos, H. A. Suraweera, and C. -B. Chae, “Full-Duplex Wireless for 6G: Progress Brings New Opportunities and Challenges,” in IEEE J. Sel. Areas Commun., vol. 41, no. 9, pp. 2729–2750, Sep 2023.

[4] K. E. Kolodziej, B. T. Perry and J. S. Herd, “In-Band Full-Duplex Technology: Techniques and Systems Survey,” in IEEE Transactions on Microwave Theory and Techniques, vol. 67, no. 7, pp. 3025–3041, Jul 2019.

[5] W. -S. Liao, O. Zhao, K. Li, H. Kawasaki, and T. Matsumura, “Implementation of In-Band Full-Duplex Communication System with Self-Interference Cancellation Using Adaptive Filter,” in Proc. IEEE VTS Asia Pac. Wireless Commun. Symp. (APWCS), Tainan city, Taiwan, 2023, pp. 1–5.

[6] C. D. Nwankwo, L. Zhang, A. Quddus, M. A. Imran and R. Tafazolli, “A Survey of Self-Interference Management Techniques for Single Frequency Full Duplex Systems,” in IEEE Access, vol. 6, pp. 30242–30268, 2018.

[7] D. Ahmed and A. M. Eltawil, “All-Digital Self-Interference Cancellation Technique for Full-Duplex Systems,” in IEEE Transactions on Wireless Communications, vol. 14, no. 7, pp. 3519–3532, Jul 2015.

[8] H. Luo, M. Holm and T. Ratnarajah, “On the performance of active analog self-interference cancellation techniques for beyond 5G systems,” in China Communications, vol. 18, no. 10, pp. 158–168, Oct 2021.

[9] D. Bharadia, E. McMilin, and S. Katti, “Full duplex radios,” in Proceedings of the ACM SIGCOMM Conference, 2013, pp. 375–386.

[10] B. Yu et al., “Full duplex communication with practical self-interference cancellation implementation,” in ICC 2022-IEEE International Conference on Communications, 2022, pp. 1100–1105.

[11] B. Widrow and M. E. Hof, “Adaptive switching circuits,” in Neurocomputing: Foundations of Research. Cambridge, MA, USA: MIT Press, 1988, pp. 123–134.

[12] H. Lee, J. Kim, G. Choi, I. P. Roberts, J. Choi, and N. Lee, “Nonlinear Self-Interference Cancellation With Adaptive Orthonormal Polynomials for Full-Duplex Wireless Systems,” in IEEE Transactions on Wireless Communications, vol. 24, no. 7, pp. 5796–5810, Jul 2025.

[13] Z. Li et al., “An augmented nonlinear LMS for digital self-interference cancellation in full-duplex direct-conversion transceivers,” IEEE Trans. Signal Process., vol. 66, no. 15, pp. 4065–4078, Aug 2018.

[14] C. Paleologu, J. Benesty, and S. Ciochină, “An overview on optimized NLMS algorithms for acoustic echo cancellation,” EURASIP J. Adv. Signal Process., vol. 2015, no. 1, pp. 1-19, Dec. 2015.

[15] A. Ali, M. Moinuddin and T. Y. Al-Nafouri, “NLMS is More Robust to Input-Correlation Than LMS: A Proof,” in IEEE Signal Processing Letters, vol. 29, pp. 279-283, 2022.

[16] S. Choudhary, S. S. Pradhan, and S. N. Merchant, “A SPT treatment to the realization of the sign-LMS based adaptive filters,” IEEE Trans. Circuits Syst. I, Reg. Papers, vol. 59, no. 9, pp. 2025–2033, Sep. 2012.

[17] D. Bismor, K. Czyz, and Z. Ogonowski, “Review and comparison of variable step-size LMS algorithms,” Int. J. Acoust. Vib., vol. 21, no. 1, pp. 24-39, Mar 2016.

[18] R. H. Kwong and E. W. Johnston, “A variable step size LMS algorithm, IEEE Trans. Signal Process., vol. 40, no. 7, pp. 1633-1642, Jul 1992.

[19] W. Ao, Y. Song, and C. Li, “A new variable step size LMS adaptive fil tering algorithm,” in Proc. Int. Conf. Comput. Sci. Electron. Eng. (ICC-SEE), Hangzhou, China, 2012, vol. 2, pp. 100-104.

[20] D. Wu, “Scaled conjugate gradient digital cancellation algorithm study for 6G full duplex wireless communication system,” in Proc. IEEE Int. Wireless Commun. Mobile Comput. Conf. (IWCMC), 2024, pp. 1-6.

[21] Y. Jia, Z. Zhang, and L. Han, “Low complexity self-interference cancellation based on two-stage RLS filtering for full-duplex radios,” in Proc. IEEE Global Commun. Conf. (GLOBECOM), 2023, pp. 1-6.

[22] M. Li, Y. Zhao, Q. Wang, et al., “RBF NN-enabled adaptive filter for any type of noise,” IEEE Trans. Neural Netw. Learn. Syst., early access, 2024.

[23] R. K. Agarwal, I. Hussain, and B. Singh, “Application of LMS-based NN structure for power quality enhancement in a distribution network under abnormal conditions,” IEEE Trans. Neural Netw. Learn. Syst., vol. 29, no. 5, pp. 1598–1607, May 2017.

[24] F. Huang, S. Zhang, and W. X. Zheng, “Bayesian-learning-based difusion least mean square algorithms over networks,” IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 10, pp. 13217–13231, Oct 2024.

[25] K. Komatsu, Y. Miyaji, H. Uehara and T. Matsumura, “Theoretical Investigation and Optimization of Power Amplifier Nonlinearity for In-Band Full-Duplex Radios,” in IEEE Transactions on Wireless Communications, vol. 22, no. 5, pp. 3384–3396, May 2023.

[26] X. Yu, C. Liu, and H. Long, “LMS-based digital nonlinear selfinterference cancellation algorithm for SBFD systems,” in Proc. IEEE Globecom Workshops (GC Wkshps), 2024, pp. 1–6.

[27] H. Lee and N. Lee, “Nonlinear and non-stationary self-interference cancellation for full-duplex wireless systems,” in Proc. IEEE Int. Conf. Commun. (ICC), 2023, pp. 1370–1375.

[28] L. Lu, P. Jin, G. Pang et al., “Learning nonlinear operators via Deep-ONet based on the universal approximation theorem of operators,” Nat. Mach. Intell., vol. 3, pp. 218–229, 2021.

[29] S. Lanthaler et al., “Nonlinear reconstruction for operator learning of PDEs with discontinuities,” arXiv:2210.01074 [math.NA], 2022. [Online].

[30] A. Vaswani et al., “Attention is all you need,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 5998-6008.

[31] E. Calvello, S. Reich, and A. M. Stuart, “Continuum attention for neural operators,” arXiv:2406.06486 [cs.LG], 2024.

[32] Z. Li et al., “Physics-informed neural operator for learning partial diferential equations,” ACM/IMS J. Data Sci., vol. 1, no. 3, pp. 1–27, 2024.

[33] B. Raonic et al., “Convolutional neural operators for robust and accurate learning of PDEs,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 77187-77200.

[34] Z. Li et al., “Fourier neural operator for parametric partial diferential equations,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=c8P9NQVtmnO

[35] Y. Dong et al., “Fan: Fourier analysis networks,” arXiv:2410.02675 [cs.LG], 2024. [Online].

[36] R. Balaji, A. Kadiyala, D. Cabric, and S. Diggavi, “Sample-Eficient Self-Interference Cancellation for In-Band Full Duplex Radios via In-Context Learning,” in Proc. Workshop on AI and ML for Next-Generation Wireless Communications and Networking (AI4NextG), ICML, 2026. [Online]. Available: https://openreview.net/forum?id=QL5zUwojjK