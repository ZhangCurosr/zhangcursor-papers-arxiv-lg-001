# NEURAL VIDEO CODING UNDER LIMITED DELAY: THEORETICAL OPTIMALITY AND SYSTEM DESIGN

Likun Cai<sup>1</sup>, Jun Chen<sup>2</sup> and Ashish Khisti<sup>1</sup>

<sup>1</sup>University of Toronto <sup>2</sup> McMaster University

likun.cai@mail.utoronto.ca chenjun@mcmaster.ca

akhisti@ece.utoronto.ca

## ABSTRACT

Traditional neural video compression (NVC) methods predominantly follow strictly causal predictive-coding paradigm, treating zero-frame delay as a fixed constraint rather than delay as an explicit design parameter. From an informationtheoretic perspective, allowing limited frame delay enables a codec to exploit correlations with future frames within the delay window, thereby reducing uncertainty that is inaccessible to strictly causal prediction and potentially improving the rate-distortion tradeoff. In this work, we formalize this intuition through a rigorous rate–distortion analysis builds on the clique factorization of Markov networks, and propose Graph-Factored Sequential Coding (GFSC), a principled framework for sequential video coding under limited delay. To our knowledge, we provide the first theoretical result showing that, for general K-order Markov sources with positive distributions, a K-frame delay is sufficient to attain jointcoding rate-distortion optimality over a nontrivial distortion region, a guarantee unavailable to zero-delay causal coding in general. In practice, we instantiate GFSC through lightweight neural modules, allowing existing causal predictive codecs to be augmented without modifying their core architectures. Experiments across multiple NVC models and datasets show that GFSC consistently improves rate-distortion performance over the corresponding causal counterparts. Under matched delay budgets, GFSC also achieves competitive performance compared with representative bidirectional codecs.

## 1 INTRODUCTION

Video compression can be fundamentally viewed as a sequential lossy source coding problem (Viswanathan & Berger, 2002; Yang et al., 2011; Ma et al., 2007). Most existing neural video compression (NVC) models follow a strictly causal predictive coding paradigm (Agustsson et al., 2020; Lu et al., 2019; 2020; Li et al., 2021; Mentzer et al., 2022), where each frame is encoded using only previously reconstructed frames and therefore cannot exploit future correlations. For temporally correlated video, future frames provide predictive information about the current frame and reduce its uncertainty, potentially improving the rate-distortion tradeoff. In addition, allowing limited frame delay does not necessarily incur substantial additional latency. At 60 fps, a singleframe lookahead introduces only 16.67 ms of buffering latency, which is modest relative to the inference latency of modern NVC models and tolerable in many practical scenarios. Frame delay should therefore be treated as an explicit design parameter rather than fixed a priori to zero.

The information-theoretic basis of limited delay coding has been studied through delayed sequential coding (Ma et al., 2007). In particular, for K-order Gauss-Markov sources, a K-frame delay is sufficient to attain joint-coding rate-distortion optimality over a nonempty distortion region, a guarantee that zero-delay causal coding does not generally enjoy. However, it remains unclear whether this optimality extends beyond Gaussian sources, and the current theory does not provide a constructive architectural principle for translating a given delay budget into a practical coding system. Existing bidirectional codecs (Wu et al., 2018; Djelouah et al., 2019; Xu et al., 2024; Yang et al., 2020; Yılmaz & Tekalp, 2021) exploit future frames in practice, but typically rely on fixed anchorbased structures designed for relatively large delay budgets, thereby inefficient in the limited-delay regime of interest. Consequently, a unified framework that characterizes limited-delay optimality for general Markov sources while enabling flexible practical NVC remains largely underexplored.

![](images/da910df5eca9adeed0d4b594e4e364913d752c2d7bc05612a3f06a16f2b3ab05.jpg)  
Figure 1: Left: GFSC-1 coding architecture for three frames. Middle: Rate-PSNR comparison between zero-delay DCVC and GFSC with delays K = 1, 2, 3 on UVG. Right: Latency (top; lower the better) and BD-rate (bottom; lower the better) of GFSC and DCVC-B at matched delays, with zero-delay DCVC as the baseline. BD-rate is measured relative to zero-delay DCVC.

In this work, we propose Graph Factored Sequential Coding (GFSC), a principled framework for sequential coding under limited delay. To our knowledge, we are the first to prove that, for general K-order Markov sources with positive distributions, a K-frame delay is sufficient to attain jointcoding rate-distortion optimality over a nontrivial distortion region. Our analysis builds on prob abilistic graphical models, where K-order Markov dependencies factorize into local clique-wise interactions. For K-order Gauss-Markov sources, the conditional independence structure further reduces to pairwise interactions with linear interpolation sufficient statistics. Figure 1 (left) illustrates the resulting GFSC architecture under one-frame delay. Although real videos are not assumed to follow an exact Markov model, their temporal dependencies are strongly local. As shown in Figure 2, the estimated inverse covariance of video features is approximately banded, with the strongest dependencies concentrated within a small temporal neighborhood. This empirical locality further motivates exploiting short, configurable delays.

We show that GFSC can be implemented using lightweight neural modules that exploit delayedframe information to reduce the coding uncertainty of the current frame and improve its ratedistortion tradeoff. Only minimal modifications are required to existing causal NVC architectures. Experiments across multiple models, datasets, and delay settings show consistent rate-distortion improvements over causal baselines and representative B-frame codecs under matched delay budgets. The main contributions of this work are summarized as follows:

1. We propose GFSC, a flexible delayed sequential coding framework that uses clique-wise interactions architecture to systematically exploit a configurable frame-delay budget.

2. We establish rigorous rate-distortion guarantees for GFSC. For general K-order Markov sources, we prove that K-delay GFSC attains joint-coding rate-distortion optimality over a nontrivial distortion region. For K-order Gauss-Markov sources, the clique-wise construction further reduces to pairwise linear interpolation.

3. We realize GFSC using lightweight neural modules that can be integrated into existing causal NVC models with minimal architectural changes. Experiments demonstrate consistent rate-distortion gains over causal baselines and competitive performance against representative B-frame codecs under matched delay budgets.

## 2 PRELIMINARIES

## 2.1 JOINT CODING

Consider an L-frame sequence $\mathbf { X } \triangleq ( X _ { 1 } , \cdots , X _ { L } )$ where $X _ { \ell } \in \mathcal { X } _ { \ell }$ and $\mathbf { X } \sim P _ { \mathbf { X } }$ . In joint coding (JC), all frames are available prior to compression. We adopt a one-shot setup (Li & El Gamal, 2018), where a single source sequence x is compressed at one time. Given that encoder and decoder share the common randomness $\bar { W } \in \mathcal { W }$ independent of source, a JC system is specified by

$$
f : \mathcal { X } _ { 1 } \times \cdots \times \mathcal { X } _ { L } \times \mathcal { W } \to \mathcal { M } , \qquad g : \mathcal { M } \times \mathcal { W } \to \hat { \mathcal { X } } _ { 1 } \times \cdots \times \hat { \mathcal { X } } _ { L } ,
$$

![](images/9d5e6893fe0b428c8aaf8cfb5556584d441841f37b6b9d8e3a677dc1d736f769.jpg)

![](images/9ef7cae68dbc9d24f7320d5c9926480b4310ec255de1e4729a35c0df35ae266b.jpg)  
Figure 2: Temporal-dependence analysis for UVG dataset. Left: Estimated information matrix for 32-frame UVG video clips. Right: Mean APC vs. Separation curve. APC denotes Absolute partial correlation. Separation denotes the distance between two frame positions, i.e. $| j - k |$ for $( X _ { j } , X _ { k } )$

where the message $M \in \mathcal { M } = \{ 0 , 1 \} ^ { * }$ is a binary string with expected length $\mathbb { E } [ l ( M ) ]$ . The reconstruction X<sup>ˆ</sup> follows $P _ { \hat { \mathbf { X } } | \mathbf { X } } ( \cdot | \mathbf { x } ) , \mathrm { i . e . , } \hat { \mathbf { X } } = g ( f ( \mathbf { x } , W ) , W ) \sim P _ { \hat { \mathbf { X } } | \mathbf { X } } ( \cdot | \mathbf { x } )$ . The reconstruction fidelity for each $X _ { \ell }$ is measured by a distortion function $d _ { \ell } : \mathcal { X } _ { \ell } \times \mathcal { X } _ { \ell } \to [ 0 , \infty )$ , with $d _ { \ell } ( x , x ) = 0$ and $\bar { d _ { \ell } } ( x , w ) > 0$ for $w \neq x$ . According to Li & El Gamal (2018), there exists a one-shot coding scheme such that minimum achievable $\check { R } ^ { \mathrm { J C } , o }$ satisfies:

$$
R ^ { \mathrm { J C , } o } \leq I ( \mathbf { X } ; \hat { \mathbf { X } } ) + \mathcal { O } ( \log I ( \mathbf { X } ; \hat { \mathbf { X } } ) ) .
$$

where the mutual information $I ( \mathbf { X } ; { \hat { \mathbf { X } } } )$ is the theoretical optimal solution in the asymptotic setting (infinite blocklength), i.e. JC rate-distortion function (Csiszar & K ´ orner, 2011).¨

Theorem 2.1 (JC rate-distortion function). Define $\mathcal { P } \triangleq \{ P _ { \hat { \mathbf { X } } | \mathbf { X } } : \mathbb { E } [ d _ { \ell } ( X _ { \ell } , \hat { X } _ { \ell } ) ] \leq D _ { \ell } , 1 \leq \ell \leq L \}$ the single-letter JC rate-distortion function is

$$
R ^ { \mathrm { J C } } ( \mathbf { D } ) = \operatorname* { m i n } _ { P _ { \hat { \mathbf { x } } | \mathbf { x } } \in \mathcal { P } } I ( \mathbf { X } ; \hat { \mathbf { X } } ) .\tag{1}
$$

## 2.2 DELAYED SEQUENTIAL CODING

In delayed sequential coding (DSC), source frames are observed sequentially with a limited lookahead of K frames $( K < \bar { L } )$ Following Ma et al. (2007), there are three canonical setups. We focus on the noncausal-causal (NC-C) setting, which is most relevant to NVC, and defer the causalnoncausal (C-NC) and noncausal-noncausal (NC-NC) settings to Appendix A.1. An NC-C coding architecture is specified by:

$$
f _ { \ell } : \mathcal { X } _ { 1 } \times \dots \times \mathcal { X } _ { \operatorname* { m i n } \{ \ell + K , L \} } \times \mathcal { W } \to \mathcal { M } _ { \ell } , \qquad g _ { \ell } : \mathcal { M } _ { 1 } \times \dots \times \mathcal { M } _ { \ell } \times \mathcal { W } \to \hat { \mathcal { X } } _ { \ell } ,
$$

where $M _ { \ell } \in \mathcal { M } _ { \ell }$ denotes ℓ-th encoder message. Under K-delay relaxation, the ℓ-th encoder has access up to $X _ { \ell + K }$ . The reconstruction $\hat { X } _ { \ell }$ follows $P _ { \hat { X } _ { \ell } | X ^ { \ell + K } } ( \cdot | x ^ { \ell + K } )$ determined by $\{ f _ { j } , g _ { j } \} _ { j = 1 } ^ { \ell }$ A key result in Ma et al. (2007) shows that, for K-order Gauss-Markov sources, a K-frame delay is sufficient for DSC to asymptotically attain the joint-coding rate-distortion function in the highresolution regime. However, this leaves two gaps. First, the result is established for the Gauss-Markov setting and does not characterize whether the same optimality extends to general K-order Markov sources. Second, DSC specifies the admissible information flow and its fundamental limits and does not provide an architectural principle for realizing these limits in practical coding systems.

In this work, we address both issues. We first establish the joint-coding optimality of K-delay sequential coding for a broader class of K-order Markov sources through a probabilistic graphical model analysis. We then introduce GFSC as a constructive realization of this structure.

## 3 GRAPH FACTORED SEQUENTIAL CODING

We introduce Graph Factored Sequential Coding (GFSC), inspired by Markov networks. We adopt a one-shot setting and analyze the associated rate-distortion performance.

![](images/4b81f0dbcfa23bfd96b5b4565e6e30310c46ad6c3f3e6b2336d9845893c837d1.jpg)  
Figure 3: Second-order Markov structure illustration. $L e f t { \mathrm { : } }$ Maximal cliques and clique factorization. Middle: Second-order Markov network. Right: The information matrix J for Gauss-Markov satisfying $J _ { i j } = 0 \mathrm { f o r } | i - j | > 2 . \ ^ {  } { * } ^ { , }$ denotes potentially nonzero entries.

## 3.1 MARKOV NETWORKS

Definition 3.1 (K-order Markov). A K-order Markov source is a random vector X satisfying Markov chains $X _ { \ell } \to ( X _ { \ell + 1 } , \cdot \cdot \cdot , X _ { \ell + K } ) \to ( X _ { \ell + K + 1 } , \cdot \cdot \cdot , X _ { L } )$ for all ℓ.

Definition 3.2 (Maximal clique). Let $\mathcal { G } = ( \nu , \mathcal { E } )$ be an undirected graph. A nonempty subset $\mathcal { C } \subseteq \mathcal { V }$ is a clique if $\{ i , j \} \in { \mathcal { E } }$ for every pair of distinct nodes $i , j \in { \mathcal { C } } .$ . A clique is maximal if it is not strictly contained in any other clique of G.

Markov networks. A Markov network represents the conditional-independence structure of a joint distribution $P _ { \mathbf { X } }$ through an undirected graph $\mathcal { G } = ( \nu , \mathcal { E } )$ , where each node denotes a random variable (Koller & Friedman, 2009). Let $\mathcal { C } ( \mathcal { G } )$ be the set of maximal cliques of G. For a strictly positive distribution, the Hammersley-Clifford theorem gives the clique factorization of probability density/mass function $\begin{array} { r } { p _ { \mathbf { X } } \mathbf { ( x ) } ~ = ~ \frac { 1 } { Z } \prod _ { \mathcal { C } _ { j } \in \mathcal { C } ( \mathcal { G } ) } \psi _ { j } \mathbf { ( } x \mathbf { { \mathcal { C } } } _ { j } \mathbf { ) } } \end{array}$ , where $\psi _ { j }$ is a positive clique factor and $Z$ a normalization constant. For K-order Markov sources as in Definition 3.1, the graph has a simple temporal structure. Its maximal cliques are the consecutive $( K + 1 )$ )-variable windows, i.e. ${ \mathcal C } _ { j } \ \triangleq \ \{ j , \dots , j + K \}$ . More importantly, the K-order Markov property itself directly yields a clique-wise factorization, without requiring the positivity condition:

$$
p \mathbf x ( \mathbf x ) = p ( x ^ { K + 1 } ) \prod _ { j } p ( x _ { j + K } \mid x _ { j } ^ { j + K - 1 } ) \triangleq \prod _ { j } \psi _ { j } ( x _ { \mathcal C _ { j } } ) ,\tag{2}
$$

where $\psi _ { 1 } ( x _ { \mathcal { C } _ { 1 } } ) \triangleq p ( x ^ { K + 1 } )$ and $\psi _ { j } ( x c _ { j } ) \triangleq p ( x _ { j + K } | x _ { j } ^ { j + K - 1 } )$ for $j \geq 2$ . Hence, the dependency structure of a K-order Markov source decomposes into local interactions over $( K + 1 )$ -variable cliques. Figure $3 \ : ( L e f t )$ shows the clique factorization for a 2-order Markov example. This observation motivates the construction of GFSC introduced in Section 3.2.

Gaussian graphical models (GGMs). GGMs specialize Markov networks to Gaussian sources and have been exploited in multi-terminal source coding (Chen et al., 2017; 2020; Xie et al., 2020). For a Gaussian source $\mathbf { X } \sim { \mathcal { N } } ( 0 , \Sigma )$ , let $J \triangleq \Sigma ^ { - 1 }$ be its information matrix. The induced GGM $\mathcal G ( J )$ has edge set $\mathcal { E } = \{ \{ i , j \} : i \neq j , \ J _ { i j } \neq 0 \}$ . For a K-order Gauss-Markov source, J is K-banded, as illustrated in Figure $3 \ : ( R i g h t )$ . Moreover, the Gaussian density admits the pairwise factorization

$$
p _ { \mathbf { X } } ( \mathbf { x } ) \propto \prod _ { \ell } \exp \left( - \frac { 1 } { 2 } J _ { \ell \ell } x _ { \ell } ^ { 2 } \right) \prod _ { \{ i , j \} } \exp \left( - J _ { i j } x _ { i } x _ { j } \right) .\tag{3}
$$

This additional structure leads to the edge-wise and node-wise auxiliary construction and, ultimately, the pairwise interpolation form derived in Section 3.5.

## 3.2 ARCHITECTURE OF CODING SCHEME

Let $\mathbf { X } \ = \ ( X _ { 1 } , \cdots , X _ { L } )$ be a sequentially observed source and K the allowed delay relaxation $( K < L - \mathrm { { i } } )$ . We define the encoding and decoding mappings of GFSC-K architecture as follows.

Encoding / decoding. At each stage ℓ, the encoder observes at most K-delay source window $X _ { \ell } ^ { \ell + K }$ and previous messages $M ^ { \ell - 1 }$ . It then produces a new message $M _ { \ell }$ . The ℓ-th decoder reconstructs

$X _ { \ell }$ using all messages available up to stage ℓ. The encoding and decoding mappings are given by:

$$
f _ { \ell } : \mathcal { X } _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } \times \mathcal { M } ^ { \ell - 1 } \times \mathcal { W } \to \mathcal { M } _ { \ell } , \qquad g _ { \ell } : \mathcal { M } ^ { \ell } \times \mathcal { W } \to \hat { \mathcal { X } } _ { \ell } .
$$

There are L decoding stages for L source frames respectively. Compared with the NC-C DSC-K architecture, this formulation imposes a local-window structure. Figure $1 \ : ( l e f t )$ and Figure 4 $( R i g h t )$ illustrate the structure for $K = 1 , 2$ . Now we characterize the operational rate-distortion region.

Definition 3.3 (Operational RD region). A rate-distortion tuple (R, D) is achievable for one-shot GFSC-K system, if there exist encoder and decoder as defined above such that

$$
R _ { \ell } \geq \mathbb { E } [ l ( M _ { \ell } ) ] , \qquad D _ { \ell } \geq \mathbb { E } [ d _ { \ell } ( X _ { \ell } , \hat { X } _ { \ell } ) ] ,
$$

where $M _ { \ell } \in \mathcal { M } _ { \ell }$ denotes the ℓ-th encoder message. The operational RD region $\mathcal { R } _ { K } ^ { o }$ is the closure of the set of all achievable (R, D). For a given D, the rate region $\mathcal { R } _ { K } ^ { o } ( { \bf D } )$ consists of all R such that $( { \bf R } , { \bf D } ) \in \mathcal { R } _ { K } ^ { o }$ , and the minimum sum-rate function is $R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } , o } ( \bar { \mathbf { D } } ) = \operatorname* { m i n } _ { \mathbf { R } \in \mathcal { R } _ { K } ^ { o } ( \mathbf { D } ) } \sum R _ { \ell }$

## 3.3 RATE-DISTORTION ANALYSIS

The operational region $\mathcal { R } _ { K } ^ { o }$ cannot be directly computed as it involves all possible one-shot encoders and decoders. Here we provide a tractable approximation in terms of mutual information, which corresponds to the information RD region. We introduce auxiliary variable $U _ { \ell } ,$ , which serves as a single-letter representation of the information conveyed by the ℓ-th encoder message.

Definition 3.4 (Information RD region). For a given source $\mathbf { X } ,$ , the single-letter information RD region of a GFSC-K system, denoted by $\mathcal { R } _ { K }$ , consists of all (R, D) satisfying

$$
R _ { \ell } \geq I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) , \qquad D _ { \ell } \geq \mathbb { E } [ d _ { \ell } ( X _ { \ell } , \hat { X } _ { \ell } ) ] ,\tag{4}
$$

for some auxiliary random variables U and $\hat { \mathbf X }$ satisfying $\hat { X } _ { \ell } = g _ { \ell } ( U ^ { \ell } )$ and the Markov conditions:

$$
\begin{array} { r } { U _ { \ell } \to ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } , U ^ { \ell - 1 } ) \to \mathbf { X } , } \end{array}\tag{5}
$$

where $g _ { \ell } ( \cdot )$ are deterministic functions. For a given D, the information rate region $\mathcal { R } _ { K } ( { \bf D } )$ is the closure of all R such that $( \mathbf { R } , \mathbf { D } ) \in \mathcal { R } _ { K }$ , and the information minimum sum-rate function is $R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( { \bf D } ) = \operatorname* { m i n } _ { { \bf R } \in \mathcal { R } _ { K } ( { \bf D } ) } \sum { \hat { R _ { \ell } } }$

Remark 3.1 (Role of the auxiliary variables). The auxiliary variable $U _ { \ell }$ serves as a single-letter representation of the information generated at the ℓ-th GFSC stage. For a K-order Markov source, the local window observed by this stage is naturally associated with the local clique $\mathcal { C } _ { \ell }$ of the underlying Markov network. Thus, $U _ { \ell }$ can be viewed as a compressed representation of this clique, while $\Breve { U } ^ { \ell - 1 }$ carries information accumulated from preceding overlapping cliques. The Markov condition in equation 5 ensures that this representation respects the sequential information flow of GFSC.

Theorem 3.5 (Rate-distortion characterization). For a given distortion tuple D and any ${ \textbf { R } } \in$ $\mathcal { R } _ { K } ( { \bf D } )$ , the operational rate region $\mathcal { R } _ { K } ^ { o } ( { \bf D } )$ satisfies $\mathbf { R } + \log ( \mathbf { R } + 1 ) + 5 \stackrel { \circ } { \in } \mathcal { R } _ { K } ^ { o } ( \mathbf { D } ) \subseteq \dot { \mathcal { R } } _ { K } ( \mathbf { D } )$ Moreover, the operational sum-ratefunction satisfies

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) \leq R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } , o } ( \mathbf { D } ) \leq R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) + \mathcal { O } \big ( \log R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) \big ) .\tag{6}
$$

Proof. See Appendix B.1.

Theorem 3.5 connects the information sum-rate function to the operational GFSC-K architecture up to a logarithmic one-shot overhead. We next identify a particular class of clique-wise auxiliary variables that satisfies the GFSC constraints together with stronger Markov conditions (See Appendix B.2 for more details). For K-order Markov sources with positive distribution, this construction suffices to show that the GFSC information sum-rate function coincides with the JC ratedistortion function in Theorem 2.1 over a nonempty distortion region.

![](images/397d64861bf5f585c33760e28d642ab600566882eac7c3e9b8d775b83126f217.jpg)  
Figure 4: Left: Second-order Markov network (top), its augmentation by clique-wise auxiliary variables $Z _ { j }$ (middle), and further augmentation by node-wise auxiliary variables $W _ { \ell }$ (bottom). Right: GFSC architecture with 2-delay and optimal auxiliary variables constructed in $L e f t$

## 3.4 JOINT-CODING OPTIMALITY FOR MARKOV SOURCES

## 3.4.1 GENERAL MARKOV SOURCES

As discussed in Section 3.1, a K-order Markov source factorizes over consecutive cliques $\mathcal { C } _ { j } ~ =$ $\{ j , \ldots , j + K \}$ . This graphical structure motivates a clique-wise auxiliary construction: each auxiliary variable is generated from the source variables within one local clique, while the collection of overlapping auxiliaries captures the global source dependence through local interactions. For the optimality proof, we construct a particular class of such auxiliaries satisfying stronger Markov conditions than those required by GFSC. These conditions preserve the relevant dependency structure while remaining admissible under the GFSC-K information-flow constraints.

Definition 3.6 (K-order Markov source with positive distribution). Let X be a K-order Markov source as in Definition 3.1. Moreover, each $X _ { j }$ has finite alphabet $2 \leq | \mathcal { X } _ { j } | < \infty$ and the joint distribution is positive, i.e. $p _ { \mathbf { X } } ( \mathbf { x } ) > 0$ for all $\mathbf { x } \in \mathcal { X } _ { 1 } \times \cdots \times \mathcal { X } _ { L }$

Theorem 3.7 (JC optimality). Consider a Markov source X as in Definition 3.6. There exists a non-empty distortion region $\mathcal { D } _ { \epsilon } \triangleq \left\{ \mathbf { D } : 0 < D _ { \ell } \leq \epsilon _ { \ell } , 1 \leq \ell \leq L \right\}$ , such that for all $\mathbf { D } \in \mathcal { D } _ { \epsilon } ,$ a set of clique-wise auxiliaries U can be constructed for GFSC-K and the following hold:

$$
\begin{array} { r } { l . \ U _ { \ell } \to ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } , U _ { \operatorname* { m a x } \{ 1 , \ell - K \} } ^ { \operatorname* { m i n } \{ \ell , L - K \} } ) \to \mathbf { X } . } \end{array}
$$

2. $R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) = R ^ { \mathrm { J C } } ( \mathbf { D } )$ , where $R ^ { \mathrm { J C } } ( \mathbf { D } )$ is JC rate-distortionfunction in Theorem 2.1.

## Proof. See Appendix B.2.

Remark 3.2 (Graphical interpretation). To our knowledge, Theorem 3.7 is the first rigorous result showing that K-delay coding can attain JC rate-distortion optimality for general $K \cdot$ -order Markov sources. The key is the locality of the Markov network: clique-wise auxiliary variables compress local dependencies while preserving the conditional-independence structure required for optimal coding. Figure 4 explicitly illustrates this construction, where clique-wise auxiliary variables $Z _ { \ell }$ induce the desired conditional independencies and node-wise auxiliary variables $\dot { W _ { \ell } }$ further refine individual reconstructions. This locality is also consistent with real video, whose estimated inverse covariance in Figure 2 is largely banded, with the strongest dependencies concentrated within about three off-diagonals. In practice, we show that these local clique-wise interactions can be efficiently realized in NVC using learnable neural modules.

![](images/0c406f3bea681d3997c2713b4d91409e7e53781d22b62bcda528ee5d2e047314.jpg)  
Figure 5: Left: A second-order GGM and its augmented representation within $\mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ ; dashed arrows indicate sequential coding. Right: Edge-wise auxiliary construction by linear interpolation and additive noise, where $\bar { \omega } _ { j , k } = 1 - \omega _ { j , k }$

## 3.5 GAUSS-MARKOV SOURCES UNDER MSE

The general result above relies on clique-wise interactions of the underlying Markov network. Under the Gauss-Markov assumption and MSE distortion, the graphical structure further decomposes into edge-wise interactions as discussed in Section 3.1, allowing the clique-wise auxiliaries to be replaced by pairwise Gaussian auxiliaries.

Theorem 3.8 (Gauss-Markov specialization). Consider a K-order Gauss-Markov source X under MSE distortion. There exists a nonempty distortion region $\mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ such that

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C - G } } ( \mathbf { D } ) = R ^ { \mathrm { J C G } } ( \mathbf { D } ) , \qquad \forall \mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } } .\tag{7}
$$

Moreover, the optimal clique-wise construction can be realized using pairwise Gaussian auxiliaries associated with the edges ofthe underlying GGM, with linear interpolations as sufficient statistics.

Proof. See Appendix C.4.

Gaussianity is not required for GFSC, but provides a particularly simple interpretation of its optimal auxiliary structure: $\bar { ( } K + 1 )$ -variable clique interactions reduce to ordered pairwise interactions within the K-frame neighborhood. Figure 5 illustrates this decomposition for a second-order Gauss Markov source. Representative characterizations of $\mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ for first-order Gauss-Markov example are provided in Appendix C.5.

## 4 EXPERIMENTAL RESULTS

We conduct experiments on several NVC models based on causal predictive coding, including the DCVC (Li et al., 2021), video compression transformer (VCT) (Mentzer et al., 2022), and DHVC (Lu et al., 2024). We compare the rate-distortion performance of the GFSC extension and its causal counterparts. We further compare against both neural and conventional delayed coding baselines (bidirectional) under the same delay constraints. For quantitative comparison, we report BD-rate results on UVG (Mercat et al., 2020) and MCL-JCV (Wang et al., 2016).

Clique-wise interaction. For a K-delay setting, GFSC operates on the clique-wise latent $\left( \mathbf { y } _ { \ell } , \ldots , \mathbf { y } _ { \ell + K } \right)$ , where $\mathbf { y } _ { \ell } = f _ { \ell } ( \mathbf { x } _ { \ell } )$ denotes the latent representation of frame x . We implement this interaction using a window-based spatial-temporal attention module Liu et al. (2022), which jointly models the current and lookahead latent representations within each local spatial window. This enables configurable K-delay fusion without modifying the underlying analysis transforms and naturally handles boundary cases with fewer than K available future frames. Architectural details are provided in Appendix D.2.

## 4.1 RATE-DISTORTION ANALYSIS

Causal counterparts. We evaluate GFSC on three causal NVC models. Figure 6 shows the rate-PSNR curves for the baselines and their GFSC-enhanced variants with delays $K = 1 , 2 , 3 .$ . Across all models, incorporating the delay window interaction consistently improves rate-distortion performance, with gains increasing as K grows. Notably, even a single-frame delay yields significant improvement. Table 1 further quantifies the BD-rate improvements. Across all datasets and models, GFSC achieves consistent bitrate reductions relative to causal baselines, demonstrating the effectiveness of limited delay for improving rate-distortion performance. See Appendix D.3 for results on 4K video dataset.

![](images/7d23f635f78cda8e0cf3b7fc7ff2bbd10801cd653891f68824e1ef0550ae2d04.jpg)  
Figure 6: Rate-PSNR comparison between causal and GFSC-enhanced models on UVG and MCL-JCV dataset. Columns show DCVC, VCT, and DHVC, respectively.

Delayed baselines. In Fig. 7, we compare GFSC with delayed coding baselines, including DCVC-B (Sheng et al., 2025) and HM (random access), under delay budgets $K \ \leq \ 3 .$ For DCVC-B, we evaluate $K \ = \ 1 , 2 , 3 ,$ , corresponding to I-B-I, I-B-B-I, and I-B-B-B-I, respectively. For HM, due to its supported GOP structures, we evaluate K = 1, 3 us-

Table 1: BD-rate (%) of GFSC-enhanced models relative to causal baselines.

<table><tr><td rowspan="2">Delay</td><td colspan="2">UVG (1080p)</td></tr><tr><td>DCVC VCT DHVC</td><td>MCL-JCV (1080p) DCVC VCT DHVC</td></tr><tr><td>K = 1</td><td>-20.57 -28.38 -7.39</td><td>-10.90 -32.35 -13.35</td></tr><tr><td>K = 2</td><td>-25.29 -48.00 -10.17</td><td>-17.32 -50.14-14.65</td></tr><tr><td>K = 3</td><td>-28.22 -55.78 -11.91</td><td>-20.30 -56.70 -16.07</td></tr></table>

ing I-B-I and I-B-B-B-I. We also include DCVC-L, a widened DCVC baseline with a parameter count matched to DCVC-GFSC, to isolate the gain from delayed modeling and that due to increased model capacity. GFSC consistently achieves better rate-distortion performance than both the delayed and capacity-matched baselines. The corresponding BD-rate are reported in Table 2.

Complexity analysis. Table 2 compares DCVC-GFSC with causal DCVC and DCVC-B. GFSC introduces only modest additional latency over the causal baseline as the delay budget increases, while consistently improving rate-distortion performance. Under matched delay constraints, GFSC remains substantially faster than DCVC-B and also achieves better BD-rate performance. These results show that GFSC provides a more efficient use of limited delay, offering stronger compression gains with considerably lower latency than bidirectional coding.

## 5 RELATED WORK

Neural video compression. Recent NVC methods have achieved rate-distortion performance competitive with conventional video codecs. Most existing approaches follow a strictly causal predictive coding paradigm, in which each frame is encoded using only previously reconstructed frames. Early methods adopt motion compensation followed by residual coding (Lu et al., 2019; Agustsson et al., 2020; Lin et al., 2020), whereas more recent approaches directly model temporal dependencies in the latent space through learned conditional coding (Li et al., 2021; Sheng et al., 2022; Li et al., 2023; 2024; Mentzer et al., 2022). Although these methods support zero-frame delay coding, they cannot exploit correlations with future frames. Bidirectional codecs address this limitation through interpolation-based prediction (Wu et al., 2018; Djelouah et al., 2019; Xu et al., 2024) or bidirectional conditional modeling (Yang et al., 2020; Yılmaz & Tekalp, 2021; Sheng et al., 2025). However, they typically rely on fixed anchor-frame structures and are designed for relatively large frame delays, making them less flexible under small delay budgets. More recently, Xiang et al. (2025) incorporated 1-frame delay into a unified intra/inter coding framework. Their method, however, is specialized to the 1-delay setting and does not provide either a general construction for arbitrary delay budgets or an information-theoretic justification of the resulting rate-distortion gains.

![](images/db76c2831e2b5c1ef40701648fb9634f21421ca8a69587229d79b84acd2cc0c8.jpg)  
Figure 7: Rate-PSNR comparison under delay $K \leq 3$ on (a) UVG and (b) MCL-JCV. Colors identify model families. Line styles and markers distinguish delay settings.

Table 2: BD-rate (%) and computational complexity under matched delay budgets. Latency, memory, FLOPs, and model parameters are reported for DCVC, DCVC-GFSC, and DCVC-B.
<table><tr><td>Method</td><td>Delay K</td><td>BD-rate (%)</td><td>Enc. Latency (s/frame)</td><td>Dec. Latency (s/frame)</td><td>Memory (GB)</td><td>FLOPs (G/frame)</td><td>Params (M)</td></tr><tr><td>DCVC</td><td>0</td><td>0.0</td><td>0.52</td><td>0.44</td><td>5.54</td><td>4430.53</td><td>34.53</td></tr><tr><td>DCVC-L</td><td>0</td><td>-8.57</td><td>0.53</td><td>0.44</td><td>5.54</td><td>4447.51</td><td>39.21</td></tr><tr><td rowspan="3">DCVC-B</td><td>1</td><td>3.70</td><td>0.84</td><td>0.64</td><td>10.95</td><td>7153.37</td><td>55.38</td></tr><tr><td>2</td><td>-18.64</td><td>1.14</td><td>0.89</td><td>10.95</td><td>8805.79</td><td>55.38</td></tr><tr><td>3</td><td>-27.24</td><td>1.16</td><td>0.90</td><td>10.95</td><td>9632.01</td><td>55.38</td></tr><tr><td rowspan="3">DCVC-GFSC</td><td>1</td><td>-20.57</td><td>0.65</td><td>0.44</td><td>5.52</td><td>5916.28</td><td>39.21</td></tr><tr><td>2</td><td>-25.29</td><td>0.80</td><td>0.44</td><td>5.52</td><td>7451.81</td><td>39.21</td></tr><tr><td>3</td><td>-28.22</td><td>0.88</td><td>0.44</td><td>5.52</td><td>8987.58</td><td>39.21</td></tr></table>

Delayed sequential coding. The information-theoretic foundations of delayed sequential coding have been studied in Ma et al. (2007); Ma & Ishwar (2007), showing that limited delay can improve the rate-distortion performance of sequential coding and even a single-frame delay holds potential for matching the JC optimality in specific cases, e.g. first-order Gauss-Markov sources. However, these results do not establish whether such optimality extends to general K-order Markov sources, nor do they provide a constructive coding architecture for practical video compression.

## 6 CONCLUSION

We introduced GFSC, a general framework that treats frame delay as an explicit coding resource rather than fixing sequential video coding to the causal setting. By exploiting the clique structure of K-order Markov sources, we established the first rigorous result showing that a K-frame delay can attain joint-coding rate-distortion optimality for a broad class of general Markov sources. For Gauss-Markov sources, this construction further reduces to pairwise interactions with linear interpolation sufficient statistics, revealing a simple structure behind the optimal auxiliaries. GFSC can be implemented with lightweight neural modules and integrated into existing NVC models with minimal architectural changes. Experiments consistently demonstrate improved rate-distortion performance over causal coding and bidirectional codecs under matched delay budgets. Together, these results identify limited delay as a fundamental and practically useful design dimension for neural video compression, bridging information-theoretic optimality and implementable coding architectures. Limitations and future directions are discussed in Appendix E.

## REFERENCES

Eirikur Agustsson, David Minnen, Nick Johnston, Johannes Balle, Sung Jin Hwang, and George´ Toderici. Scale-space flow for end-to-end optimized video compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8503–8512, 2020.

Jean Begaint, Fabien Racap ´ e, Simon Feltman, and Akshay Pushparaja. Compressai: a py- ´ torch library and evaluation platform for end-to-end compression research. arXiv preprint arXiv:2011.03029, 2020.

Toby Berger. Rate-distortion theory. Wiley Encyclopedia of Telecommunications, 2003.

Jun Chen, Farrokh Etezadi, and Ashish Khisti. Generalized gaussian multiterminal source cod ing and probabilistic graphical models. In 2017 IEEE International Symposium on Information Theory (ISIT), pp. 719–723. IEEE, 2017.

Jun Chen, Li Xie, Yameng Chang, Jia Wang, and Yizhong Wang. Generalized gaussian multiterminal source coding: The symmetric case. IEEE Transactions on Information Theory, 66(4): 2115–2128, 2020.

Thomas M Cover. Elements ofinformation theory. John Wiley & Sons, 1999.

Imre Csiszar and J´ anos K´ orner.¨ Information theory: coding theorems for discrete memoryless systems. Cambridge University Press, 2011.

Abdelaziz Djelouah, Joaquim Campos, Simone Schaub-Meyer, and Christopher Schroers. Neural inter-frame compression for video coding. In Proceedings ofthe IEEE/CVF international confer ence on computer vision, pp. 6421–6429, 2019.

Robert M Gray and Thomas G Stockham. Dithered quantizers. IEEE Transactions on Information Theory, 39(3):805–812, 2002.

Daphne Koller and Nir Friedman. Probabilistic graphical models, volume 165. MIT press Cambridge, 2009.

Cheuk Ting Li and Abbas El Gamal. Strong functional representation lemma and applications to coding theorems. IEEE Transactions on Information Theory, 64(11):6967–6978, 2018.

Jiahao Li, Bin Li, and Yan Lu. Deep contextual video compression. Advances in Neural Information Processing Systems, 34:18114–18125, 2021.

Jiahao Li, Bin Li, and Yan Lu. Neural video compression with diverse contexts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22616–22626, 2023.

Jiahao Li, Bin Li, and Yan Lu. Neural video compression with feature modulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26099–26108, 2024.

Jianping Lin, Dong Liu, Houqiang Li, and Feng Wu. M-lvc: Multiple frames prediction for learned video compression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3546–3554, 2020.

Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 3192–3201. IEEE, 2022.

Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, and Zhiyong Gao. Dvc: An end-to-end deep video compression framework. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11006–11015, 2019.

Guo Lu, Xiaoyun Zhang, Wanli Ouyang, Li Chen, Zhiyong Gao, and Dong Xu. An end-to-end learning framework for video compression. IEEE transactions on pattern analysis and machine intelligence, 43(10):3292–3308, 2020.

Ming Lu, Zhihao Duan, Fengqing Zhu, and Zhan Ma. Deep hierarchical video compression. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 8859–8867, 2024.

Nan Ma and Prakash Ishwar. The value of frame-delays in the sequential coding of correlated sources. In 2007 IEEE International Symposium on Information Theory, pp. 1496–1500. IEEE, 2007.

Nan Ma, Ye Wang, and Prakash Ishwar. Delayed sequential coding of correlated sources. In 2007 Information Theory and Applications Workshop, pp. 214–222. IEEE, 2007.

Fabian Mentzer, George Toderici, David Minnen, Sung-Jin Hwang, Sergi Caelles, Mario Lucic, and Eirikur Agustsson. Vct: A video compression transformer. arXiv preprint arXiv:2206.07307, 2022.

Alexandre Mercat, Marko Viitanen, and Jarno Vanne. Uvg dataset: 50/120fps 4k sequences for video codec analysis and development. In Proceedings of the 11th ACM multimedia systems conference, pp. 297–302, 2020.

Zaixi Shang, Joshua P Ebenezer, Alan C Bovik, Yongjun Wu, Hai Wei, and Sriram Sethuraman. Subjective assessment of high dynamic range videos under different ambient conditions. In 2022 IEEE international conference on image processing (ICIP), pp. 786–790. IEEE, 2022.

Xihua Sheng, Jiahao Li, Bin Li, Li Li, Dong Liu, and Yan Lu. Temporal context mining for learned video compression. IEEE Transactions on Multimedia, 25:7311–7322, 2022.

Xihua Sheng, Li Li, Dong Liu, and Shiqi Wang. Bi-directional deep contextual video compression. IEEE Transactions on Multimedia, 2025.

Lucas Theis and Eirikur Agustsson. On the advantages of stochastic encoders. arXiv preprint arXiv:2102.09270, 2021.

Harish Viswanathan and Toby Berger. Sequential coding of correlated sources. IEEE Transactions on Information Theory, 46(1):236–246, 2002.

Haiqiang Wang, Weihao Gan, Sudeng Hu, Joe Yuchieh Lin, Lina Jin, Longguang Song, Ping Wang, Ioannis Katsavounidis, Anne Aaron, and C-C Jay Kuo. Mcl-jcv: a jnd-based h. 264/avc video quality assessment dataset. In 2016 IEEE international conference on image processing (ICIP), pp. 1509–1513. IEEE, 2016.

Chao-Yuan Wu, Nayan Singhal, and Philipp Krahenbuhl. Video compression through image interpolation. In Proceedings of the European conference on computer vision (ECCV), pp. 416–431, 2018.

Hui Xiang, Yifan Bian, Li Li, Jingran Wu, Xianguo Zhang, and Dong Liu. Real-time neural video compression with unified intra and inter coding. arXiv preprint arXiv:2510.14431, 2025.

Li Xie, Xiaolan Tu, Siyao Zhou, and Jun Chen. Generalized gaussian multiterminal source coding in the high-resolution regime. IEEE Transactions on Communications, 68(6):3782–3791, 2020.

Chenming Xu, Meiqin Liu, Chao Yao, Weisi Lin, and Yao Zhao. Ibvc: Interpolation-driven b-frame video compression. Pattern Recognition, 153:110465, 2024.

Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T Freeman. Video enhancement with task-oriented flow. International Journal ofComputer Vision, 127(8):1106–1125, 2019.

En-Hui Yang, Lin Zheng, Da-Ke He, and Zhen Zhang. Rate distortion theory for causal video coding: Characterization, computation algorithm, and comparison. IEEE transactions on information theory, 57(8):5258–5280, 2011.

Ren Yang, Fabian Mentzer, Luc Van Gool, and Radu Timofte. Learning for video compression with hierarchical quality and recurrent enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6628–6637, 2020.

M Akın Yılmaz and A Murat Tekalp. End-to-end rate-distortion optimized learned hierarchical bi-directional video compression. IEEE Transactions on Image Processing, 31:974–983, 2021.

Yichi Zhang and Fengqing Zhu. Opendcvcs: a pytorch open source implementation and performance evaluation of the dcvc series video codecs. arXiv preprint arXiv:2508.04491, 2025.

Jacob Ziv. On universal quantization. IEEE Transactions on Information Theory, 31(3):344–347, 2003.

## A COMPARISON OF CODING STRUCTURES

## A.1 DELAYED SEQUENTIAL CODING

![](images/87aa44507282e4cf1c93520e22f354eee4b9b72cef2eff3465b3b6293f488c2e.jpg)  
Figure 8: Illustration of three coding architectures for a four-frame source under 2-frame delay. Left: C-NC. Middle: NC-NC. Right: NC-C.

For completeness, we briefly review the three canonical setups of delayed sequential coding introduced in Ma et al. (2007). These include: (i) noncausal encoder and causal decoder (NC-C), (ii) causal encoder and noncausal decoder (C-NC), and (iii) noncausal encoder and noncausal decoder (NC-NC). The NC-C setting is defined in Section 2.2. Here we provide the definitions of the remaining two setups.

C-NC (causal encoder, noncausal decoder). In the C-NC setting, the encoder operates causally, while the decoder is allowed a delay of K frames. The encoding and decoding mappings are given by

$$
\begin{array} { r } { f _ { \ell } : \mathcal { X } _ { 1 } \times \dots \times \mathcal { X } _ { \ell } \times \mathcal { W }  \mathcal { M } _ { \ell } , ~ } \\ { g _ { \ell } : \mathcal { M } _ { 1 } \times \dots \times \mathcal { M } _ { \operatorname* { m i n } \{ \ell + K , L \} } \times \mathcal { W }  \hat { \mathcal { X } } _ { \ell } . } \end{array}
$$

That is, the ℓ-th encoder has access only to past and current frames $( X _ { 1 } , \cdots , X _ { \ell } )$ , while the decoder reconstructs $\hat { X } _ { \ell }$ using messages up to time $\ell + K$

NC-NC (noncausal encoder, noncausal decoder). In NC-NC setting, both encoders and decoders are allowed limited lookahead. Let $K _ { 1 }$ and $K _ { 2 }$ denote the delays at the encoder and decoder, respectively, such that $K _ { 1 } + K _ { 2 } = K$ . The encoding and decoding mappings are defined as

$$
\begin{array} { r l } & { f _ { \ell } : \mathcal { X } _ { 1 } \times \dots \times \mathcal { X } _ { \operatorname* { m i n } \{ \ell + K _ { 1 } , L \} } \times \mathcal { W } \to \mathcal { M } _ { \ell } , } \\ & { g _ { \ell } : \mathcal { M } _ { 1 } \times \dots \times \mathcal { M } _ { \operatorname* { m i n } \{ \ell + K _ { 2 } , L \} } \times \mathcal { W } \to \hat { \mathcal { X } } _ { \ell } . } \end{array}
$$

Thus, the encoder can access future frames up to $K _ { 1 }$ steps ahead, while the decoder reconstructs each frame with an additional delay of $K _ { 2 }$

These three setups characterize different ways of distributing the delay budget between the encoder and decoder, and together provide a unified framework for analyzing delayed sequential coding systems. We illustrate the corresponding system architectures for the three delayed setup in Figure 8. From a theoretical perspective, Corollary 6.1 in Ma et al. (2007) shows that all three setups achieve the same sum-rate performance for a given total delay $K .$

## A.2 NVC PARADIGM COMPARISON

We briefly review common practical video coding paradigms and compare them with the proposed GFSC architecture. Figure 9 provides a systematic comparison of these coding paradigms.

P-frame coding. P-frame coding follows a causal predictive coding paradigm, where each frame is encoded conditioned only on previously reconstructed frames. This design enables low-latency processing and efficient implementation, and forms the foundation of most existing neural video compression methods. However, the strictly causal structure prevents the encoder from exploiting future frames, which limits the ability to capture longer-range temporal dependencies.

![](images/c277cb7d29b70e868555b6c8278fb3624b7740c7dddbd9397603e235c1821bd8.jpg)  
Figure 9: Left: graphical illustration for causal predictive coding. Middle: graphical illustration for bidirectional coding. Right: graphical illustration for proposed GFSC-1.

B-frame coding. B-frame coding extends the predictive paradigm by incorporating both past and future frames through bidirectional prediction. In practice, this is typically implemented via interpolation or bidirectional motion modeling between anchor frames. While such designs can improve compression efficiency, they usually rely on fixed temporal structures and are better suited to largerdelay settings. As a result, they are less flexible in scenarios with limited delay budgets.

GFSC. The proposed GFSC framework generalizes these paradigms by explicitly modeling a limited delay budget. Instead of relying on fixed bidirectional structures, GFSC constructs clique-wise dependencies within a delay window, enabling a flexible tradeoff between delay and compression efficiency. This formulation bridges the gap between strictly causal coding and joint coding, while remaining compatible with modern NVC architectures.

## B MARKOV SOURCE MODELS UNDER GENERAL DISTORTION

## B.1 OPERATIONAL RATE-DISTORTION ANALYSIS

Recall the information RD region for GFSC in Definition 3.4. We first derive the information sumrate function.

Theorem B.1 (Information minimum sum-rate). For a given distortion tuple D, the minimum information sum-rate of GFSC-K as in Definition 3.4 is

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( { \bf D } ) = \operatorname* { m i n } I ( { \bf X } ; { \bf U } ) ,\tag{8}
$$

and the minimum is taken over all auxiliary variables U satisfying Markov chains in equation 5 and the distortion constraints in equation 4.

Proof. By Definition 3.4, for a given distortion tuple D, the minimum sum-rate is given by

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) = \operatorname* { m i n } \sum _ { \ell = 1 } ^ { L } I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) ,
$$

where the minimization is over all feasible auxiliary variables satisfying the constraints in equation 5 and the distortion constraints. For notational convenience, we denote each stage rate as $I ( X _ { \ell } ^ { \ell + K } ; U _ { \ell } | U ^ { \ell - 1 } )$ . Then by the Markov chain in equation 5, we apparently have:

$$
\begin{array} { l } { { I ( { \bf X } ; U _ { \ell } | U ^ { \ell - 1 } ) = I ( X _ { \ell } ^ { \ell + K } ; U _ { \ell } | U ^ { \ell - 1 } ) + I ( { \bf X } \setminus X _ { \ell } ^ { \ell + K } ; U _ { \ell } | X _ { \ell } ^ { \ell + K } , U ^ { \ell - 1 } ) } } \\ { { \mathrm { } } } \\ { { \mathrm { } = I ( X _ { \ell } ^ { \ell + K } ; U _ { \ell } | U ^ { \ell - 1 } ) . } } \end{array}
$$

Substituting this into the sum-rate expression, we have

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) = \operatorname* { m i n } \sum _ { \ell = 1 } ^ { L } I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } )\tag{9}
$$

$$
{ \bf \Gamma } = \operatorname* { m i n } \sum _ { \ell = 1 } ^ { L } I ( { \bf X } ; U _ { \ell } | U ^ { \ell - 1 } )\tag{10}
$$

$$
{ \bf \mu } = \operatorname* { m i n } I ( { \bf X } ; { \bf U } ) ,\tag{11}
$$

where equation 10 follows from the Markov chain in equation 5, and equation 11 follows from the chain rule. This completes the proof. □

Next, we show the proof for Theorem 3.5 to characterize the operational RD region.

Theorem B.2 (Rate-distortion characterization). For a given distortion tuple D, the following holds:

1. For any $\mathbf { R } \in \mathcal { R } _ { K } ( \mathbf { D } )$ , the operational rate region $\mathcal { R } _ { K } ^ { o } ( { \bf D } )$ satisfies

$$
\mathbf { R } + \log ( \mathbf { R } + 1 ) + 5 \in \mathcal { R } _ { K } ^ { o } ( \mathbf { D } ) ,\tag{12}
$$

and

$$
\mathcal { R } _ { K } ^ { o } ( { \bf D } ) \subseteq \mathcal { R } _ { K } ( { \bf D } ) .\tag{13}
$$

2. The operational sum-rate satisfies

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) \leq R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } , o } ( \mathbf { D } ) \leq R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) + \mathcal { O } \big ( \log R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) \big ) .\tag{14}
$$

where $R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } )$ is characterized in Theorem B.1.

## Proof. Part 1. One-shot achievability

By Strong Functional Representation Lemma (SFRL) in Li & El Gamal (2018), for jointly distributed random variables $X , Y \sim P _ { X Y }$ , there exists Z independent of X, and function $f$ such that $Y = f ( X , Z )$ . Z is not necessarily unique. SFRL further states that there exists a Z such that:

$$
H ( Y | Z ) \leq I ( X ; Y ) + \log ( I ( X ; Y ) + 1 ) + 4 .
$$

Notice that this can be extended to conditional distribution case. Given $P _ { X Y \mid W }$ , there exists $Z$ independent of (X, W) such that $Y = f ( X , W , Z )$ and:

$$
H ( Y | W , Z ) \leq I ( X ; Y | W ) + \log ( I ( X ; Y | W ) + 1 ) + 4 .
$$

Proof of equation $l 2 \mathbf { : R } + \log ( \mathbf { R } + \mathbf { 1 } ) + 5 \in \mathcal { R } _ { K } ^ { o } ( \mathbf { D } )$

For a given D and $\mathbf { R } \in \mathcal { R } _ { K } ( \mathbf { D } )$ , let U be jointly distributed with X, where the Markov chains in equation 5, rate and distortion constraints in equation 4 are all satisfied. Denote the joint distribution of such $( \mathbf { X } , \mathbf { U } , { \hat { \mathbf { X } } } )$ by $P _ { \mathbf { X U \hat { X } } }$ and according to the Markov chains in equation 5, we have:

$$
P _ { \mathbf { X U } \hat { \mathbf { X } } } = P _ { \mathbf { X } } \cdot \prod _ { \ell = 1 } ^ { L } P _ { U _ { \ell } | X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } , U ^ { \ell - 1 } } \cdot \prod _ { \ell = 1 } ^ { L } \mathbf { 1 } \{ \hat { X } _ { \ell } = g _ { \ell } ( U ^ { \ell } ) \} .\tag{15}
$$

Now, fix the marginal distribution $P _ { \mathbf { U } }$ in equation 15 and use it to construct the codebook, i.e.,

$$
P _ { \mathbf { U } } = P _ { U _ { L } | U ^ { L - 1 } } \cdot \cdot \cdot P _ { U _ { \ell } | U ^ { \ell - 1 } } \cdot \cdot \cdot P _ { U _ { 1 } } .\tag{16}
$$

By the SFRL, we have:

1. (First stage) There exists a random variable $Z _ { 1 }$ independent of $X ^ { K + 1 }$ and a deterministic function $q _ { 1 }$ such that $U _ { 1 } = q _ { 1 } ( X ^ { K + 1 } , Z _ { 1 } )$ and:

$$
H ( U _ { 1 } | Z _ { 1 } ) \leq I ( X ^ { K + 1 } ; U _ { 1 } ) + \log ( I ( X ^ { K + 1 } ; U _ { 1 } ) + 1 ) + 4 .
$$

Therefore, the first stage encoder applies the deterministic function $q _ { 1 }$ to get $U _ { 1 }$ whose distribution needs to be preserved according to equation 16.

2. (Intermediate stages) By the conditional SFRL, there exists a random variable $Z _ { \ell }$ independent of $( U ^ { \ell - 1 } , X _ { \ell } ^ { \mathrm { m i n } \{ \ell + K , L \} } )$ ) and a deterministic function $q _ { \ell }$ such that $\begin{array} { r l } { U _ { \ell } } & { { } = } \end{array}$ $q _ { \ell } ( U ^ { \ell - 1 } , X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } , Z _ { \ell } )$ and:

$$
\begin{array} { r } { H ( U _ { \ell } | U ^ { \ell - 1 } , Z _ { \ell } ) \leq I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) + \log \left( I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) + 1 \right) + 4 } \end{array}
$$

At the intermediate ℓ-th stage (including the last stage), the ℓ-th encoder has access to $U ^ { \ell - 1 }$ from earlier stages. So upon observing the delayed source window $X _ { \ell } ^ { \mathrm { m i n } \{ \ell + K , L \} }$ , it applies $q _ { \ell }$ to get $U _ { \ell }$ whose conditional distribution needs to be preserved according to equation 16.

Now the encoding and decoding schemes are as follows:

1. (First stage encoder) With $Z _ { 1 }$ available at encoder and decoder, we can design prefix-free codes indexed by $Z _ { 1 }$ to losslessly represent $U _ { 1 }$ with the expected codeword length $\bar { R } _ { 1 }$

$$
\begin{array} { r } { \bar { R } _ { 1 } \leq H ( U _ { 1 } | Z _ { 1 } ) + 1 \leq I ( X ^ { K + 1 } ; U _ { 1 } ) + \log ( I ( X ^ { K + 1 } ; U _ { 1 } ) + 1 ) + 5 } \end{array}
$$

2. (Intermediate stage encoders) With $Z _ { \ell }$ available at encoder and decoder, we can design prefix-free codes indexed by $( Z _ { \ell } , U ^ { \ell - 1 } )$ to losslessly represent $U _ { \ell }$ with the expected codeword length $\bar { R } _ { \ell }$

$$
\begin{array} { r l } & { \bar { R } _ { \ell } \leq H ( U _ { \ell } | U ^ { \ell - 1 } , Z _ { \ell } ) + 1 } \\ & { \quad \leq I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) + \log ( I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) + 1 ) + 5 } \end{array}
$$

3. (Decoders) Upon receiving all $U ^ { \ell }$ , the ℓ-th decoder generate the reconstruction via ${ \hat { X } } _ { \ell } =$ $g _ { \ell } ( U ^ { \ell } )$

Then we have $\mathbf { R } + \log ( \mathbf { R } + 1 ) + 5 \in \mathcal { R } _ { K } ^ { o } ( \mathbf { D } )$

Proof of equation $I 3 \colon \mathcal { R } _ { K } ^ { o } ( \mathbf { D } ) \subseteq \mathcal { R } _ { K } ( \mathbf { D } )$

For any D and $\mathbf { R } \in \mathcal { R } _ { K } ^ { o } ( \mathbf { D } )$ , shared randomness $W \in { \mathcal { W } }$ , encoding and decoding functions

$$
f _ { \ell } : \mathcal { X } _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } \times \mathcal { M } ^ { \ell - 1 } \times \mathcal { W } \to \mathcal { M } _ { \ell } , \qquad g _ { \ell } : \mathcal { M } ^ { \ell } \times \mathcal { W } \to \hat { \mathcal { X } } _ { \ell } ,
$$

such that:

$$
R _ { \ell } \geq \mathbb { E } [ l ( M _ { \ell } ) ] , \quad \ell = 1 , \cdot \cdot \cdot , L ,\tag{17}
$$

$$
D _ { \ell } \geq \mathbb { E } [ d _ { \ell } ( X _ { \ell } , \hat { X } _ { \ell } ) ] , \quad \ell = 1 , \cdot \cdot \cdot , L ,\tag{18}
$$

we need to lower bound the expected length of messages $l ( M _ { \ell } )$ . Define:

$$
U _ { 1 } \triangleq ( M _ { 1 } , W ) , \qquad U _ { \ell } \triangleq ( M ^ { \ell } , W ) , \ell = 1 , \cdots , L .\tag{19}
$$

Then for the first stage:

$$
R _ { 1 } \geq \mathbb { E } [ l ( M _ { 1 } ) ] \geq H ( M _ { 1 } | W ) = I ( X ^ { K + 1 } ; U _ { 1 } ) .\tag{20}
$$

For the intermediate stages:

$$
\begin{array} { r l } & { R _ { \ell } \geq \mathbb { E } [ l ( M _ { \ell } ) ] \geq H ( M _ { \ell } | M ^ { \ell - 1 } , W ) } \\ & { \qquad = I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; M _ { \ell } | M ^ { \ell - 1 } , W ) } \\ & { \qquad = I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) . } \end{array}\tag{21}
$$

Notice that equation 19 implies the following Markov chains:

$$
\begin{array} { r } { U _ { \ell } \to ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } , U ^ { \ell - 1 } ) \to \mathbf { X } , \quad \ell = 1 , \cdots , L . } \end{array}\tag{22}
$$

On the other hand, by the definition of decoding functions and equation 19, we have that:

$$
\hat { X } _ { \ell } = g _ { \ell } ( M ^ { \ell } , W ) \triangleq \eta _ { \ell } ( U ^ { \ell } )\tag{23}
$$

where $\eta _ { \ell }$ are deterministic functions.

Now consider the fact that the set of constraints in eqs. (17), (18), (20) and (21), Markov chains in eq. (22) and deterministic functions in eq. (23) constitute an information rate-distortion region, denote by $\tilde { \mathcal { R } } _ { K }$ , which is the set of (R, D) such that:

$$
\begin{array} { r l } & { R _ { \ell } \geq I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; U _ { \ell } | U ^ { \ell - 1 } ) , \quad \ell = 1 , \cdots , L , } \\ & { \qquad D _ { \ell } \geq \mathbb { E } [ d _ { \ell } ( X _ { \ell } , \hat { X } _ { \ell } ) ] , \quad \ell = 1 , \cdots , L , } \end{array}
$$

for all auxiliary random variables U and $\hat { \mathbf X }$ satisfying:

$$
\begin{array} { r } { \hat { X } _ { \ell } = \eta _ { \ell } ( U ^ { \ell } ) , \qquad U _ { \ell } \to ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } , U ^ { \ell - 1 } ) \to \mathbf { X } , } \end{array}
$$

where $\eta _ { \ell }$ are some deterministic functions. By the definition of $\mathcal { R } _ { K }$ in Definition 3.4, apparently we have $\tilde { \mathcal { R } } _ { K } = \mathcal { R } _ { K }$ . Therefore, $\mathscr { R } _ { K } ^ { o } ( { \bf D } ) \subseteq \tilde { \mathscr { R } } _ { K } ( { \bf D } ) = \mathscr { R } _ { K } ( { \bf D } )$ . This completes the proof of Part 1.

## Part 2. Sum-rate bounds

As we have already shown in Theorem B.1 that the information minimum sum-rate function of GFSC-K system is:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( { \bf D } ) = \operatorname* { m i n } I ( { \bf X } ; { \bf U } ) ,
$$

subject to constraints in eqs. (4) and (5). We also have $\mathcal { R } _ { K } ^ { o } ( { \bf D } ) \subseteq \mathcal { R } _ { K } ( { \bf D } )$ in the proof of equation 13. Thus, we have the following lower bound:

$$
R _ { K , \mathrm { s u m } } ^ { \mathbf { G F S C } , o } ( \mathbf { D } ) = \operatorname* { m i n } _ { \mathbf { R } \in \mathcal { R } _ { K } ^ { o } ( \mathbf { D } ) } \sum _ { \ell = 1 } ^ { L } R _ { \ell } \geq \operatorname* { m i n } _ { \mathbf { R } \in \mathcal { R } _ { K } ( \mathbf { D } ) } \sum _ { \ell = 1 } R _ { \ell } = R _ { K , \mathrm { s u m } } ^ { \mathbf { G F S C } } ( \mathbf { D } ) .\tag{24}
$$

For the sum-rate upper bound, based on the proof of equation 12, we have:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } , o } ( \mathbf { D } ) = \operatorname* { m i n } _ { \mathbf { R } \in \mathcal { R } _ { K } ^ { o } ( \mathbf { D } ) } \sum _ { \ell = 1 } ^ { L } R _ { \ell }\tag{25}
$$

$$
\leq \operatorname* { m i n } _ { \mathbf { R } \in \mathcal { R } _ { K } ( \mathbf { D } ) } \sum _ { \ell = 1 } ^ { L } \left( R _ { \ell } + \log ( R _ { \ell } + 1 ) + 5 \right)\tag{26}
$$

$$
= \operatorname* { m i n } _ { \mathbf { R } \in \mathcal { R } _ { K } ( \mathbf { D } ) } \sum _ { \ell = 1 } ^ { L } R _ { \ell } + \sum _ { \ell = 1 } ^ { L } \log ( R _ { \ell } + 1 ) + 5 L\tag{27}
$$

$$
\leq \operatorname* { m i n } _ { \mathbf { R } \in \mathcal { R } _ { K } ( \mathbf { D } ) } I ( \mathbf { X } ; \mathbf { U } ) + L \cdot \log \left( \frac { \sum _ { \ell = 1 } ^ { L } ( R _ { \ell } + 1 ) } { L } \right) + 5 L\tag{28}
$$

$$
\leq R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) + L \cdot \log \left( \frac { R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) + L } { L } \right) + 5 L\tag{29}
$$

$$
= R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) + \mathcal { O } \left( \log R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) \right) ,\tag{30}
$$

where equation 26 follows equation 12, equation 28 follows the Jensen’s inequality for logarithms. This completes the proof of Part 2. □

## B.2 JOINT-CODING OPTIMALITY FOR GENERAL MARKOV SOURCES

We provide several definitions for a general distortion function.

Definition B.3 (Distortion function). Define the general distortion function form as:

$$
d : \mathcal { X } \times \mathcal { X } \to [ 0 , \infty ) ,
$$

with $d ( x , x ) = 0$ and $d ( x , w ) > 0$ for $w , x \in \mathcal { X }$ , w $\neq x .$

Definition B.4. Define $\begin{array} { r } { d _ { \operatorname* { m i n } } \triangleq \operatorname* { m i n } _ { x \neq w } d ( x , w ) > 0 \mathrm { ~ f o r ~ a l l ~ } x , w \in \mathcal X } \end{array}$

Definition B.5. Define $\begin{array} { r } { d _ { \operatorname* { m a x } } \triangleq \operatorname* { m a x } _ { x \neq w } d ( x , w ) < \infty \mathrm { ~ f o r ~ a l l ~ } x , w \in \mathcal { X } . } \end{array}$

For a general K-order Markov source as in Definition 3.6, its probability mass function $p \mathbf { x }$ can be naturally factorized as $\begin{array} { r } { p _ { \mathbf { X } } ( \mathbf { x } ) = \prod _ { j = 1 } ^ { L - K } \psi _ { j } ( x c _ { j } ) } \end{array}$ according to eq. (2), where ${ \mathcal C } _ { j } \triangleq \{ j , \cdots , j + K \}$ With the positive distribution assumption, $\psi _ { j } ( x _ { c _ { j } } )$ is strictly positive factor. We define $\psi _ { j }$ as the corresponding vectorized factor.

Definition B.6. For $p \mathbf { x }$ , define the minimum probability mass $\begin{array} { r } { p _ { \operatorname* { m i n } } \triangleq \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { X } _ { 1 } \times \dots \times \mathcal { X } _ { L } } p _ { \mathbf { X } } ( \mathbf { x } ) > 0 } \end{array}$

Definition B.7. Define the clique minimum mass $\begin{array} { r } { c _ { j } \triangleq \operatorname* { m i n } _ { x _ { c _ { j } } } \psi _ { j } ( x _ { c _ { j } } ) > 0 } \end{array}$ for $1 \leq j \leq L - K$

Definition B.8. Define the index set $\mathcal T _ { \ell } \triangleq \{ \operatorname* { m a x } \{ 1 , \ell - K \} , \cdots , \operatorname* { m i n } \{ \ell , L - K \} \}$ for $1 \le \ell \le L$

Now we provide the proof for Theorem 3.7.

![](images/5d9ffdc9fcdd8f8b30fcc51a6cce0436bdbf2a48cf03b693fdbd9c9b575072b5.jpg)  
Figure 10: Clique-wise augmentation of a second-order Markov network, where $U _ { j }$ is associated with the maximal clique $( \breve { X _ { j } } , X _ { j + 1 } , X _ { j + 2 } )$

Lemma B.9 (Nontrivial clique-wise auxiliary variables). Let X be a K-order Markov source as in Definition 3.6. Define a positive constant $\delta _ { 0 }$ as follow:

$$
\delta _ { 0 } \triangleq \operatorname* { m i n } _ { 1 \le j \le L - K } \left\{ 1 - \left( 1 + \frac { c _ { j } } { \| \psi _ { j } \| _ { 2 } } \right) ^ { - \frac { 1 } { K + 1 } } \right\} .\tag{31}
$$

Then for any $0 < \delta < \delta _ { 0 }$ , there exist clique-wise auxiliary random variables $( U ^ { ( \delta ) } ) ^ { L - K }$ , jointly distributed with $\mathbf { X } ,$ , such that:

1. Joint distribution of $( \mathbf { X } , ( U ^ { ( \delta ) } ) ^ { L - K } )$ is also positive.

2. Thefollowing Markov chains hold:

$$
U _ { j } ^ { ( \delta ) }  X _ { j } ^ { j + K }  ( \mathbf { X } , U _ { - j } ^ { ( \delta ) } ) , \quad j = 1 , \ldots , L - K ,\tag{32}
$$

$$
X _ { \ell } \to U _ { \mathbb { Z } _ { \ell } } ^ { ( \delta ) } \to X _ { - \ell } , \quad \ell = 1 , \dots , L .\tag{33}
$$

3. Define $\begin{array} { r } { \Delta _ { \ell } ( \delta ) \triangleq \operatorname* { m i n } _ { g _ { \ell } } \mathbb { E } [ d _ { \ell } ( X _ { \ell } , g _ { \ell } ( U _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) ) ] } \end{array}$ over arbitrary deterministic mapping $g _ { \ell }$ . Then $\Delta _ { \ell } ( \delta )$ is nontrivial, i.e. $0 < \Delta _ { \ell } ( \delta ) \leq \dot { d } _ { \ell }$ <sub>,max</sub>.

Proof. We provide the proof in two parts. We first show that $\Delta _ { \ell } ( \delta ) < d _ { \ell , \mathrm { m a x } }$ is generally true based on the definition of distortion in Definition B.3. Then we provide a way to explicitly construct auxiliary variables $( U ^ { ( \delta ) } ) ^ { L - K }$ , which satisfy the Markov conditions in eqs. (32) and (33) and induce strictly positive distortion for each source variable, i.e. $\Delta _ { \ell } ( \delta ) > 0$ , for all $0 < \delta < \delta _ { 0 }$

Part I. $\Delta _ { \ell } ( \delta ) \leq d _ { \ell , \mathrm { m a x } } .$

For arbitrary decoder input $U _ { \mathcal { T } _ { \ell } }$ , we define $V _ { \ell } \triangleq U _ {  { \mathbb { Z } } _ { \ell } }$ with $\mathcal { V } _ { \ell } \triangleq \mathcal { U } _ { \operatorname* { m a x } \{ 1 , \ell - K \} } \times \cdots \times \mathcal { U } _ { \operatorname* { m i n } \{ \ell , L - K \} }$ Let ${ \hat { x } } _ { \ell } \triangleq g _ { \ell } ( v _ { \ell } )$ denote the decoder reconstruction given observation $v _ { \ell } \in \mathcal { V } _ { \ell }$ . By the definition of distortion $d _ { \ell } : \mathcal { X } _ { \ell } \times \mathcal { X } _ { \ell } \to [ 0 , \infty )$ , we also have ${ \hat { x } } _ { \ell } \in { \mathcal { X } } _ { \ell }$ . Let $p _ { X _ { \ell } } { V _ { \ell } }$ be the joint distribution of $X _ { \ell }$ and $V _ { \ell }$ , then we can bound the induced distortion:

$$
\Delta _ { \ell } = \operatorname* { m i n } _ { g _ { \ell } } \mathbb { E } [ d _ { \ell } ( X _ { \ell } , g _ { \ell } ( V _ { \ell } ) ) ]\tag{34}
$$

$$
\leq \mathbb { E } [ d _ { \ell } ( X _ { \ell } , g _ { \ell } ( V _ { \ell } ) ) ]\tag{35}
$$

$$
= \sum _ { x _ { \ell } , v _ { \ell } } p _ { X _ { \ell } V _ { \ell } } ( x _ { \ell } , v _ { \ell } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } )\tag{36}
$$

$$
\leq \sum _ { x _ { \ell } , v _ { \ell } } p _ { X _ { \ell } V _ { \ell } } ( x _ { \ell } , v _ { \ell } ) d _ { \ell , \mathrm { m a x } }\tag{37}
$$

$$
= d _ { \ell , \mathrm { { m a x } } } < \infty ,\tag{38}
$$

where equation 35 is from the definition of $\Delta _ { \ell }$ , equation 37 is from Definition B.5. So $\Delta _ { \ell } \leq d _ { \ell , \mathrm { m a x } }$ is generally true for arbitrary auxiliary variables $V _ { \ell } .$

Part II. $\Delta _ { \ell } ( \delta ) > 0 .$

For $\Delta _ { \ell } ( \delta ) > 0 , \mathrm { w e }$ proceed in two steps. First, for the general distortion function in Definition B.3, we derive the necessary and sufficient condition to attain positive distortion. We then construct a set of auxiliary random variables $\{ U _ { j } \} _ { j = 1 } ^ { L - K }$ and show that they satisfy both the required Markov conditions in eqs. (32) and (33) and the derived condition for positive distortion.

Positive distortion analysis. For a given observation $v _ { \ell } \in \mathcal { V } _ { \ell }$ at decoder side, the goal of j-stage decoding is to minimize expected distortion:

$$
\mathbb { E } [ d _ { \ell } ( X _ { \ell } , g _ { \ell } ( V _ { \ell } ) ) ] = \sum _ { v _ { \ell } } p _ { V _ { \ell } } ( v _ { \ell } ) \sum _ { x _ { \ell } } p _ { X _ { \ell } | V _ { \ell } } ( x _ { \ell } | v _ { \ell } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } ) .\tag{39}
$$

Here, we define the conditional risk $\begin{array} { r } { R _ { v _ { \ell } } ( \hat { x } _ { \ell } ) \stackrel { \Delta } { = } \sum _ { x _ { \ell } } p _ { X _ { \ell } | V _ { \ell } } ( x _ { \ell } | v _ { \ell } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } ) } \end{array}$ . Apparently we have $\begin{array} { r } { R _ { v _ { \ell } } ( \hat { x } _ { \ell } ) \geq \operatorname* { m i n } _ { \hat { x } _ { \ell } \in \mathcal { X } _ { \ell } } R _ { v _ { \ell } } ( \hat { x } _ { \ell } ) } \end{array}$ , thereby

$$
\mathbb { E } [ d _ { \ell } ( X _ { \ell } , g _ { \ell } ( V _ { \ell } ) ) ] \ge \sum _ { v _ { \ell } } p _ { V _ { \ell } } ( v _ { \ell } ) \operatorname* { m i n } _ { \hat { x } _ { \ell } \in \mathcal { X } _ { \ell } } R _ { v _ { \ell } } ( \hat { x } _ { \ell } ) .\tag{40}
$$

Then the optimal decoding function $g _ { \ell } ^ { * }$ satisfies:

$$
g _ { \ell } ^ { * } \left( v _ { \ell } \right) \in \arg \operatorname* { m i n } _ { \hat { x } _ { \ell } \in \mathcal { X } _ { \ell } } \sum _ { x _ { \ell } \in \mathcal { X } _ { \ell } } p _ { X _ { \ell } | V _ { \ell } } ( x _ { \ell } | v _ { \ell } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } ) .\tag{41}
$$

In addition, as in Definition B.3, we have $d _ { \ell } ( x _ { \ell } , x _ { \ell } ) = 0$ and $d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } ) > 0$ for any $x _ { \ell } \neq \hat { x } _ { \ell }$ . Then for the minimum conditional risk mi $\mathfrak { r } _ { \hat { x } _ { \ell } } R _ { v _ { \ell } } ( \hat { x } _ { \ell } )$ , we have:

$$
\operatorname* { m i n } _ { \hat { x } _ { \ell } } R _ { v _ { \ell } } ( \hat { x } _ { \ell } ) = \operatorname* { m i n } _ { \hat { x } _ { \ell } } \sum _ { x _ { \ell } } p _ { X _ { \ell } | V _ { \ell } } ( x _ { \ell } | v _ { \ell } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } )\tag{42}
$$

$$
= \operatorname* { m i n } _ { \hat { x } _ { \ell } } \sum _ { x _ { \ell } \neq \hat { x } _ { \ell } } p _ { X _ { \ell } | V _ { \ell } } ( x _ { \ell } | v _ { \ell } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } )\tag{43}
$$

$$
\geq \operatorname* { m i n } _ { { \hat { x } } _ { \ell } } \sum _ { x _ { \ell } \neq { \hat { x } } _ { \ell } } p _ { X _ { \ell } | V _ { \ell } } ( x _ { \ell } | v _ { \ell } ) d _ { j , \operatorname* { m i n } }\tag{44}
$$

$$
= d _ { j , \mathrm { m i n } } \left( 1 - \operatorname* { m a x } _ { \hat { x } _ { \ell } } p _ { X _ { \ell } | V _ { \ell } } ( \hat { x } _ { \ell } | v _ { \ell } ) \right) .\tag{45}
$$

The minimum conditional risk is zero if and only $\mathrm { i f } p _ { X _ { \ell } | V _ { \ell } } ( \cdot | v _ { \ell } )$ is Dirac delta function, i.e.,

$$
p _ { X _ { \ell } | V _ { \ell } } ( x _ { \ell } | v _ { \ell } ) = \delta ( x _ { \ell } - \hat { x } _ { \ell } ) = \left\{ 1 , \quad x _ { \ell } = \hat { x } _ { \ell } , \right.\tag{46}
$$

Therefore, the induced distortion $\Delta _ { \ell }$ is zero if and only if $p _ { X _ { \ell } | V _ { \ell } } ( \cdot | v _ { \ell } )$ is delta function for all $v _ { \ell }$ with $p _ { V _ { \ell } } ( v _ { \ell } ) > 0$ . This is also equivalent to $H ( X _ { \ell } | V _ { \ell } ) = 0$ . It is obvious that both distortion $\epsilon _ { \ell }$ and $p _ { X _ { \ell } | V _ { \ell } }$ entries are non-negative. Therefore, it suffice to induce positive distortion $\Delta _ { \ell } ( \delta ) > 0$ if we can construct auxiliary variables V<sub>ℓ</sub> with positive distribution and positive backward test channel $p _ { X _ { \ell } | V _ { \ell } }$ for all ℓ. In the following part, we provide an explicit construction.

Construction ofauxiliary variables. In this part, we show that for K-order Markov source X as defined in Definition 3.6, we can always construct appropriate auxiliary variables, which satisfy all the conditions. For $\mathbf { X } \sim p _ { \mathbf { X } } , p _ { \mathbf { X } }$ naturally adopts the following clique factorization:

$$
p _ { \mathbf { X } } ( \mathbf { x } ) = \prod _ { j = 1 } ^ { L - K } \psi _ { j } ( x _ { \mathcal { C } _ { j } } )\tag{47}
$$

where ${ \mathcal C } _ { j } \triangleq \{ j , \cdots , j + K \}$ , and $\psi _ { j } ( x c _ { j } ) > 0$ is a positive clique factor since X has positive joint distribution. In addition, $\psi _ { j } ( x _ { \boldsymbol { C } _ { j } } )$ can be further represented by a vectorized clique factor $\psi _ { j } \in \mathbb { R } _ { + } ^ { N _ { j } }$ with $\begin{array} { r } { N _ { j } \triangleq \prod _ { r = j } ^ { j + K } | \mathcal { X } _ { r } | } \end{array}$

For j-th clique, we define the following auxiliary variable $U _ { j } ^ { ( \delta ) }$ with $( K + 1 )$ components:

$$
U _ { j } ^ { ( \delta ) } \triangleq ( U _ { j , r } ^ { ( \delta ) } : r \in \mathcal { C } _ { j } ) \quad \mathrm { w i t h } \quad U _ { j , r } ^ { ( \delta ) } \in \mathcal { X } _ { r } ,\tag{48}
$$

which is also $U _ { j } ^ { ( \delta ) } \triangleq ( U _ { j , j } ^ { ( \delta ) } , \cdot \cdot \cdot , U _ { j , j + K } ^ { ( \delta ) } )$ . Now suppose a positive constant $0 < \delta < 1$ , for each $U _ { j , r } ^ { ( \delta ) }$ , we consider a transition matrix $Q _ { j , r } ^ { ( \delta ) } \in \mathbb { R } ^ { | \mathcal { X } _ { r } | \times | \mathcal { X } _ { r } | }$ from $U _ { j , r } ^ { ( \delta ) }$ to $X _ { r }$ with each entry as:

$$
Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , u _ { j , r } ^ { ( \delta ) } ) = q _ { X _ { r } | U _ { j , r } ^ { ( \delta ) } } ( x _ { r } | u _ { j , r } ^ { ( \delta ) } ) = ( 1 - \delta ) \mathbf 1 \{ x _ { r } = u _ { j , r } ^ { ( \delta ) } \} + \delta \cdot \nu _ { r } ( x _ { r } ) ,\tag{49}
$$

where $\nu _ { r }$ denotes arbitrary positive distribution over $\mathcal { X } _ { r }$ . Here, we choose $\nu _ { r } ( x _ { r } )$ to be the uniform distribution, therefore $\begin{array} { r } { \nu _ { r } ( x _ { r } ) = \frac { 1 } { | \mathcal { X } _ { r } | } } \end{array}$ for all $x _ { r } \in \mathcal X _ { r }$ . Then $Q _ { j , r } ^ { ( \delta ) }$ can be further written as:

$$
\boldsymbol { Q } _ { j , r } ^ { ( \delta ) } = \left( 1 - \delta \right) \cdot \boldsymbol { I } + \frac { \delta } { \left| \mathcal { X } _ { r } \right| } \cdot \mathbf { 1 } \mathbf { 1 } ^ { \top } = \underbrace { \left[ \begin{array} { c c c } { 1 - \delta + \frac { \delta } { \left| \mathcal { X } _ { r } \right| } } & { \cdots } & { \frac { \delta } { \left| \mathcal { X } _ { r } \right| } } \\ { \vdots } & { \ddots } & { \vdots } \\ { \frac { \delta } { \left| \mathcal { X } _ { r } \right| } } & { \cdots } & { 1 - \delta + \frac { \delta } { \left| \mathcal { X } _ { r } \right| } } \end{array} \right] } _ { \boldsymbol { u } _ { j , r } ^ { ( \delta ) } \in \mathcal { X } _ { r } } \Biggr \} _ { \mathcal { X } _ { r } \in \mathcal { X } _ { r } } \Biggr \{ \overset { \delta } { \overbrace { \mathcal { X } _ { r } } } \cdot \overbrace { \mathcal { X } _ { r } } \mid  ,\tag{50}
$$

It can be verified that $Q _ { j , r } ^ { ( \delta ) }$ has eigenvalues:

$$
\lambda ( Q _ { j , r } ^ { ( \delta ) } ) = \{ 1 , 1 - \delta \} .
$$

Therefore as long as $0 < \delta < 1 , Q _ { j , r } ^ { ( \delta ) }$ is invertible and has all positive entries, which makes it a valid transition matrix. We also have the following form of the inverse of $Q _ { j , r } ^ { ( \delta ) }$

$$
\left( Q _ { j , r } ^ { ( \delta ) } \right) ^ { - 1 } = \frac { 1 } { 1 - \delta } \left( I - \frac { \delta } { | \mathcal { X } _ { r } | } \mathbf { 1 1 } ^ { \top } \right) .\tag{51}
$$

For the maximal clique $\mathcal { C } _ { j }$ , we further define the clique-wise transition matrix $Q _ { j } ^ { ( \delta ) } \in \mathbb { R } _ { + } ^ { N _ { j } \times N _ { j } }$ as:

$$
Q _ { j } ^ { ( \delta ) } \triangleq \bigotimes _ { r \in \mathcal { C } _ { j } } Q _ { j , r } ^ { ( \delta ) } ,\tag{52}
$$

where $\otimes$ denotes the matrix Kronecker product. Each entry $Q _ { j } ^ { ( \delta ) } ( x _ { \mathcal { C } _ { j } } , u _ { j } ^ { ( \delta ) } )$ represents the backward test channel from $u _ { j } ^ { ( \delta ) }$ to $x _ { { \mathcal { C } } _ { j } }$ , i.e. ${ q _ { X _ { C _ { j } } } } _ { | U _ { j } ^ { ( \delta ) } } ( x _ { C _ { j } } | u _ { j } ^ { ( \delta ) } )$ . For example, consider the 2-order Markov source $( X _ { 1 } , \cdots , X _ { 5 } )$ in Figure 10, we define the following constructions:

$$
\begin{array} { r l } & { U _ { 1 } ^ { ( \delta ) } \triangleq ( U _ { 1 , 1 } ^ { ( \delta ) } , U _ { 1 , 2 } ^ { ( \delta ) } , U _ { 1 , 3 } ^ { ( \delta ) } ) \xrightarrow { Q _ { 1 } ^ { ( \delta ) } \triangleq Q _ { 1 , 1 } ^ { ( \delta ) } \otimes Q _ { 1 , 2 } ^ { ( \delta ) } \otimes Q _ { 1 , 3 } ^ { ( \delta ) } } ( X _ { 1 } , X _ { 2 } , X _ { 3 } ) } \\ & { U _ { 2 } ^ { ( \delta ) } \triangleq ( U _ { 2 , 2 } ^ { ( \delta ) } , U _ { 2 , 3 } ^ { ( \delta ) } , U _ { 2 , 4 } ^ { ( \delta ) } ) \xrightarrow { Q _ { 2 } ^ { ( \delta ) } \triangleq Q _ { 2 , 2 } ^ { ( \delta ) } \otimes Q _ { 2 , 3 } ^ { ( \delta ) } \otimes Q _ { 2 , 4 } ^ { ( \delta ) } } ( X _ { 2 } , X _ { 3 } , X _ { 4 } ) } \\ & { U _ { 3 } ^ { ( \delta ) } \triangleq ( U _ { 3 , 3 } ^ { ( \delta ) } , U _ { 3 , 4 } ^ { ( \delta ) } , U _ { 3 , 5 } ^ { ( \delta ) } ) \xrightarrow { Q _ { 3 } ^ { ( \delta ) } \triangleq Q _ { 3 , 3 } ^ { ( \delta ) } \otimes Q _ { 3 , 4 } ^ { ( \delta ) } \otimes Q _ { 3 , 5 } ^ { ( \delta ) } } ( X _ { 3 } , X _ { 4 } , X _ { 5 } ) } \end{array}\tag{53}
$$

For $0 < \delta < 1 , Q _ { j } ^ { ( \delta ) }$ is also invertible since the Kronecker product of invertible matrices is also invertible. It can be verified that $Q _ { j } ^ { ( \delta ) }$ has the following set of eigenvalues:

$$
\lambda ( Q _ { j } ^ { ( \delta ) } ) = \{ 1 , 1 - \delta , \cdots , ( 1 - \delta ) ^ { K + 1 } \} .\tag{54}
$$

Accordingly, the eigenvalues for its inverse $( Q _ { j } ^ { ( \delta ) } ) ^ { - 1 }$ can be written as $( 1 - \delta ) ^ { - k } , k = 0 , \cdot \cdot \cdot , K + 1$ Recall the positive vectorized clique factor $\psi _ { j }$ for clique ${ \mathcal { C } } _ { j } $ , given $Q _ { j } ^ { ( \delta ) }$ constructed above, we need to further find a positive vectorized factor $\phi _ { j } ^ { ( \delta ) }$ satisfying:

$$
\psi _ { j } = Q _ { j } ^ { ( \delta ) } \phi _ { j } ^ { ( \delta ) } .\tag{55}
$$

Since $Q _ { j } ^ { ( \delta ) }$ is invertible, it is equivalent to find a valid positive factor such that:

$$
\phi _ { j } ^ { ( \delta ) } \triangleq ( Q _ { j } ^ { ( \delta ) } ) ^ { - 1 } \psi _ { j } \in \mathbb { R } ^ { N _ { j } } .\tag{56}
$$

Now we will show that $Q _ { i } ^ { ( \delta ) } \to I { \mathrm { ~ a s ~ } } \delta \to 0$ , thereby $\phi _ { j } ^ { ( \delta ) }  \psi _ { j }$ . Let $\| \cdot \| _ { 2 }$ denotes the induced spectral norm of a matrix. It is known that the spectral norm of a positive semi-definite matrix is also its maximum eigenvalue. Notice tha $( Q _ { j } ^ { ( \delta ) } ) ^ { - 1 } - I$ is a positive semi-definite matrix, then

$$
\begin{array} { r } { \left\| ( Q _ { j } ^ { ( \delta ) } ) ^ { - 1 } - I \right\| _ { 2 } = \lambda _ { \operatorname* { m a x } } \left( ( Q _ { j } ^ { ( \delta ) } ) ^ { - 1 } - I \right) = ( 1 - \delta ) ^ { - ( K + 1 ) } - 1 . } \end{array}\tag{57}
$$

By the definition of $\phi _ { j } ^ { ( \delta ) }$ in equation 56, we further have:

$$
\| \phi _ { j } ^ { ( \delta ) } - \psi _ { j } \| _ { \infty } \leq \| \phi _ { j } ^ { ( \delta ) } - \psi _ { j } \| _ { 2 }\tag{58}
$$

$$
= \left. ( \boldsymbol { Q } _ { j } ^ { ( \delta ) } ) ^ { - 1 } \boldsymbol { \psi } _ { j } - \boldsymbol { \psi } _ { j } \right. _ { 2 }\tag{59}
$$

$$
= \left\| \left( ( Q _ { j } ^ { ( \delta ) } ) ^ { - 1 } - I \right) \psi _ { j } \right\| _ { 2 }\tag{60}
$$

$$
\leq \left. ( Q _ { j } ^ { ( \delta ) } ) ^ { - 1 } - I \right. _ { 2 } \Vert \psi _ { j } \Vert _ { 2 }\tag{61}
$$

$$
= \left( ( 1 - \delta ) ^ { - ( K + 1 ) } - 1 \right) \| \psi _ { j } \| _ { 2 } \triangleq F ( \delta ) .\tag{62}
$$

We know that $\psi _ { j }$ is strictly positive factor due to positive joint distribution of source X. By the definition of $U _ { i } ^ { ( \delta ) }$ in equation 48, we have ${ \mathcal { U } } _ { j } = { \mathcal { X } } _ { { \mathcal { C } } _ { j } }$ . Therefore, vectorized factors $\phi _ { j } ^ { ( \delta ) }$ and $\psi _ { j }$ have the same coordinate ordering. Then we can have:

$$
\phi _ { j } ^ { ( \delta ) } ( u _ { j } ) \geq \psi _ { j } ( u _ { j } ) - | \phi _ { j } ^ { ( \delta ) } ( u _ { j } ) - \psi _ { j } ( u _ { j } ) |\tag{63}
$$

$$
\geq c _ { j } - \| \boldsymbol { \phi } _ { j } ^ { ( \delta ) } - \boldsymbol { \psi } _ { j } \| _ { \infty }\tag{64}
$$

$$
\geq c _ { j } - F ( \delta ) .\tag{65}
$$

Now we can set $\delta$ as following:

$$
0 < \delta < 1 - \left( 1 + \frac { c _ { j } } { \| \psi _ { j } \| _ { 2 } } \right) ^ { - \frac { 1 } { K + 1 } } .\tag{66}
$$

Then it can be verified that $F ( \delta ) < c _ { j }$ , thereby $\phi _ { j } ^ { ( \delta ) } ( u _ { j } ) > 0$ for every $u _ { j }$ . As we have $L - K$ maximal cliques for the K-order Markov source $\mathbf { X }$ , we can define the following positive constant:

$$
\delta _ { 0 } \triangleq \operatorname* { m i n } _ { 1 \le j \le L - K } \left\{ 1 - \left( 1 + \frac { c _ { j } } { \| \psi _ { j } \| _ { 2 } } \right) ^ { - \frac { 1 } { K + 1 } } \right\}\tag{67}
$$

Thus for any $0 < \delta < \delta _ { 0 }$ and all $j , \phi _ { i } ^ { ( \delta ) }$ is a valid positive vectorized factor. By equation 56, we also have $\begin{array} { r } { \psi _ { j } ( x _ { \mathcal { C } _ { j } } ) = \sum _ { u _ { i } ^ { ( \delta ) } } \phi _ { j } ^ { ( \delta ) } ( u _ { j } ^ { ( \delta ) } ) \bar { \prod } _ { r \in \mathcal { C } _ { j } } Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , u _ { j , r } ^ { ( \delta ) } ) } \end{array}$ . Now we define the following augmented joint distribution:

$$
q \left( \mathbf { x } , u _ { 1 } ^ { ( \delta ) } , \cdot \cdot \cdot , u _ { L - K } ^ { ( \delta ) } \right) \triangleq \prod _ { j = 1 } ^ { L - K } \left( \phi _ { j } ^ { ( \delta ) } ( u _ { j } ^ { ( \delta ) } ) \prod _ { r \in { \mathcal C } _ { j } } Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , u _ { j , r } ^ { ( \delta ) } ) \right) .\tag{68}
$$

First, we already know that $Q _ { j , r } ^ { ( \delta ) }$ has all positive entries and $\phi _ { j } ^ { ( \delta ) }$ is a positive vectorized factor for $0 < \delta < \delta _ { 0 }$ . Then $q \big ( \mathbf { x } , ( u ^ { ( \delta ) } ) ^ { \tilde { L } - K } \big )$ is also positive for $0 < \delta < \delta _ { 0 }$ . Second, we have:

$$
\begin{array} { r l } & { \displaystyle \sum _ { \mathbf { x } } \displaystyle \sum _ { ( \mathbf { u } ^ { ( \delta ) } ) ^ { L } } q \Big ( \mathbf { x } , ( u ^ { ( \delta ) } ) ^ { L - K } \Big ) = \displaystyle \sum _ { \mathbf { x } } \displaystyle \sum _ { u _ { 1 } ^ { ( \delta ) } } \cdots \sum _ { u _ { t - K } ^ { ( \delta ) } } \prod _ { j = 1 } ^ { L - K } \left( \phi _ { j } ^ { ( \delta ) } ( u _ { j } ^ { ( \delta ) } ) \prod _ { r \in \mathcal { C } _ { 3 } } Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , u _ { j , r } ^ { ( \delta ) } ) \right) } \\ & { \qquad = \displaystyle \sum _ { \mathbf { x } } \displaystyle \prod _ { j = 1 } ^ { L - K } \sum _ { u _ { j } ^ { ( \delta ) } } \left( \phi _ { j } ^ { ( \delta ) } ( u _ { j } ^ { ( \delta ) } ) \prod _ { r \in \mathcal { C } _ { 3 } } Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , u _ { j , r } ^ { ( \delta ) } ) \right) } \\ & { \qquad = \displaystyle \sum _ { \mathbf { x } } \displaystyle \prod _ { j = 1 } ^ { L - K } \psi _ { j } ( x _ { \mathcal { C } _ { j } } ) } \\ & { \qquad = \displaystyle \sum _ { \mathbf { x } } p _ { \mathbf { x } } ( \mathbf { x } ) = 1 . } \end{array}\tag{69}
$$

Therefore, this is a valid positive joint distribution for $( \mathbf { X } , ( U ^ { ( \delta ) } ) ^ { L - K } )$ . Next we show that the above constructed auxiliary variables and the corresponding joint distribution satisfy all the Markov constraints and the positive induced distortion.

Markov conditions in equation 32. Now we verify the constructed joint distribution satisfy Markov conditions in equation 32. First, Markov conditions in equation 32 is equivalent to:

$$
p ( u ^ { L - K } | \mathbf { x } ) = \prod _ { j = 1 } ^ { L - K } p ( u _ { j } | x _ { \mathcal { C } _ { j } } )\tag{70}
$$

Based on the joint distribution defined in equation 68, we can write:

$$
\begin{array} { l } { \displaystyle q \Big ( ( \boldsymbol { u } ^ { ( \delta ) } ) ^ { L - K } | { \bf x } \Big ) = \frac { q \big ( { \bf x } , ( \boldsymbol { u } ^ { ( \delta ) } ) ^ { L - K } \big ) } { p _ { \bf X } ( { \bf x } ) } } \\ { = \frac { \prod _ { j = 1 } ^ { L - K } \left( \phi _ { j } ^ { ( \delta ) } ( \boldsymbol { u } _ { j } ^ { ( \delta ) } ) \prod _ { r \in \mathcal { C } _ { j } } Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , \boldsymbol { u } _ { j , r } ^ { ( \delta ) } ) \right) } { \prod _ { j = 1 } ^ { L - K } \psi _ { j } ( x _ { \mathcal { C } _ { j } } ) } } \\ { = \prod _ { j = 1 } ^ { L - K } \frac { \phi _ { j } ^ { ( \delta ) } ( \boldsymbol { u } _ { j } ^ { ( \delta ) } ) \prod _ { r \in \mathcal { C } _ { j } } Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , \boldsymbol { u } _ { j , r } ^ { ( \delta ) } ) } { \psi _ { j } ( x _ { \mathcal { C } _ { j } } ) } } \end{array}\tag{71}
$$

Then it can be verified that $\begin{array} { r } { q ( u ^ { L - K } | \mathbf { x } ) = \prod _ { j = 1 } ^ { L - K } q ( u _ { j } | x _ { j } ^ { j + K } ) } \end{array}$ is true, thereby the Markov condition in equation 32 are satisfied.

Markov conditions in equation 33. Now we verify the constructed joint distribution satisfy Markov conditions in equation 33, which is equivalent to:

$$
\begin{array} { l } { p ( { \bf x } | u ^ { L - K } ) = p ( x _ { 1 } | u _ { 1 } ) p ( x _ { 2 } | u ^ { 2 } ) \cdot \cdot \cdot p ( x _ { \ell } | u _ { \ell - K } ^ { \ell } ) \cdot \cdot \cdot p ( x _ { L } | u _ { L - K } ) } \\ { \displaystyle \qquad = \prod _ { \ell = 1 } ^ { L } p ( x _ { \ell } | u _ { \mathrm { m a x } \{ 1 , \ell - K \} } ^ { \mathrm { m i n } \{ \ell , L - K \} } ) = \prod _ { \ell = 1 } ^ { L } p ( x _ { \ell } | u _ { \mathcal { L } _ { \ell } } ) . } \end{array}\tag{72}
$$

Now for the constructed joint distribution in equation 68, we can reorder the probability mass function as:

$$
\begin{array} { l } { { \displaystyle q \Big ( { \bf x } , \big ( u ^ { ( \delta ) } \big ) ^ { L - K } \Big ) = \left[ \prod _ { j = 1 } ^ { L - K } \phi _ { j } ^ { ( \delta ) } ( u _ { j } ^ { ( \delta ) } ) \right] \left[ \prod _ { j = 1 } ^ { L - K } \prod _ { r \in \mathcal { C } _ { j } } Q _ { j , r } ^ { ( \delta ) } ( x _ { r } , u _ { j , r } ^ { ( \delta ) } ) \right] } } \\ { { \displaystyle \qquad = \left[ \prod _ { j = 1 } ^ { L - K } \phi _ { j } ^ { ( \delta ) } ( u _ { j } ^ { ( \delta ) } ) \right] \left[ \prod _ { \ell = 1 } ^ { L } \prod _ { m \in \mathcal { T } _ { \ell } } Q _ { m , \ell } ^ { ( \delta ) } ( x _ { \ell } , u _ { m , \ell } ^ { ( \delta ) } ) \right] ^ { - 1 } } } \\ { { \displaystyle \qquad = \left[ \prod _ { j = 1 } ^ { L - K } \phi _ { j } ^ { ( \delta ) } ( u _ { j } ^ { ( \delta ) } ) \right] \left[ \prod _ { \ell = 1 } ^ { L } F _ { \ell } ( x _ { \ell } , u _ { T _ { \ell } } ^ { ( \delta ) } ) \right] } } \end{array}\tag{73}
$$

It can be verified that we have the following conditional distribution form:

$$
q \left( x _ { j } | x _ { - j } , ( u ^ { ( \delta ) } ) ^ { L - K } \right) = \frac { F _ { j } ( x _ { j } , u _ { \mathcal { T } _ { j } } ^ { ( \delta ) } ) } { \sum _ { y \in \mathcal { X } _ { j } } F _ { j } ( y , u _ { \mathcal { T } _ { j } } ^ { ( \delta ) } ) } ,\tag{74}
$$

so the factorized form of this conditional distribution is only dependent on $u _ { \mathcal { T } _ { j } } ^ { ( \delta ) }$ . Therefore, $X _ { j }$ ⊥⊥ $( X _ { - j } , \mathbf { U } \setminus U _ { \mathbb { Z } _ { j } } ) | U _ { \mathbb { Z } _ { j } }$ , which is equivalent to the Markov conditions in equation 33.

Positive induced distortion. We have already shown that $\Delta _ { \ell } ( \delta ) > 0$ if and only if $p _ { X _ { \ell } | V _ { \ell } }$ is not point mass. For arbitrary $0 < \delta < \delta _ { 0 }$ , we can construct auxiliary variables $U _ { j } ^ { ( \delta ) } \triangleq ( U _ { j , j } ^ { ( \delta ) } , \cdot \cdot \cdot , U _ { j , j + K } ^ { ( \delta ) } )$ and the corresponding joint distribution $q \big ( \mathbf { x } , ( u ^ { ( \delta ) } ) ^ { L - K } \big )$ with all positive entries. Therefore the induced distortion for each $X _ { \ell }$ is also positive, i.e. $\Delta _ { \ell } ( \delta ) > 0$ for $0 < \delta < \delta _ { 0 }$ . Specifically, for $X _ { \ell }$ and a set of auxiliary variables $( U ^ { ( \delta ) } ) ^ { L - K }$ constructed above, we have the induced distortion:

$$
\Delta _ { \ell } ( \delta ) \triangleq \operatorname* { m i n } _ { g _ { \ell } } \mathbb { E } [ d _ { \ell } ( X _ { \ell } , g _ { \ell } ( U _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) ) ]\tag{75}
$$

$$
= \sum _ { u _ { \tau _ { \ell } } ^ { ( \delta ) } } q ( u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) \operatorname* { m i n } _ { \hat { x } _ { \ell } \in \mathcal { X } _ { \ell } } \sum _ { x _ { \ell } } Q ^ { ( \delta ) } ( x _ { \ell } , u _ { \mathbb { Z } _ { \ell } } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } )\tag{76}
$$

$$
\geq \sum _ { u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } } q ( u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) d _ { \ell , \operatorname* { m i n } } \operatorname* { m i n } _ { \hat { x } _ { \ell } } \sum _ { x _ { \ell } \neq \hat { x } _ { \ell } } Q ^ { ( \delta ) } ( x _ { \ell } , u _ { \mathcal { T } _ { \ell } } ) > 0\tag{77}
$$

where min $\begin{array} { r } { { \hat { x } _ { \ell } } \sum _ { x _ { \ell } \neq \hat { x } _ { \ell } } Q ^ { ( \delta ) } ( x _ { \ell } , u _ { \mathcal { T } _ { \ell } } ) } \end{array}$ is a positive function of $\hat { x } _ { \ell }$ . Moreover, for arbitrary $\delta$ such that $0 < \delta < \delta _ { 0 }$ , we can define the following distortion region:

$$
\mathcal { D } ^ { \mathrm { A U X } } \triangleq \{ \mathbf { D } : 0 < D _ { \ell } < \Delta _ { \ell } ( \delta ) , 1 \leq \ell \leq L \} .\tag{78}
$$

Within this distortion region $\mathcal { D } ^ { \mathrm { A U X } }$ , there always exist appropriate auxiliary variables $( U ^ { ( \delta ) } ) ^ { L - K }$ constructed as above. This completes the proof. □

Lemma B.9 shows that there always exist a class of clique-wise auxiliary variables for a K-order Markov source with positive distribution, such that the induced distortion for each source variable is also positive. Next we prove two Lemmas to characterize the RD optimal backward test channel for a positive distributed source. We first define the Neumann series which will be used in the proof.

Definition B.10 (Neumann series). For a matrix $A \in \mathbb { R } ^ { n \times n }$ , the series $\textstyle \sum _ { k = 0 } ^ { \infty } A ^ { k }$ , with $A ^ { 0 } = I _ { n }$ , is called the Neumann series associated with $A . \mathrm { ~ I f ~ } \| A \| _ { 1 } < 1$ , where $\| \cdot \| _ { 1 }$ denotes the induced matrix 1-norm, this series converges, and $I _ { n } - A$ is invertible with

$$
( I _ { n } - A ) ^ { - 1 } = \sum _ { k = 0 } ^ { \infty } A ^ { k } .\tag{79}
$$

Moreover, we have

$$
\left\| ( I _ { n } - A ) ^ { - 1 } - I _ { n } \right\| _ { 1 } \leq \sum _ { k = 1 } ^ { \infty } \| A \| _ { 1 } ^ { k } = { \frac { \| A \| _ { 1 } } { 1 - \| A \| _ { 1 } } } .\tag{80}
$$

Lemma B.11. Consider random variable $X \in { \mathcal { X } }$ with alphabet size $2 \le | { \mathcal { X } } | < \infty$ and positive distribution. Let $W \in { \mathcal { X } }$ denote arbitrary reconstruction of X with $D \triangleq \operatorname { \mathbb { E } } [ d ( X , W ) ]$ and T the backward test channelfrom W to X. Thenfor any joint distribution $o f ( X , W )$ satisfying $0 < D <$ $p _ { \mathrm { m i n } } d _ { \mathrm { m i n } } ,$ we have:

1. p<sub>W</sub> is positive distribution, i.e. $p _ { W } ( w ) > 0$ for all $w \in \mathcal X$

2. T is arbitrarily close to but not exact identity matrix:

$$
\frac { 2 D } { d _ { \operatorname* { m a x } } } \leq \| T - I \| _ { 1 } \leq \frac { 2 D } { p _ { \operatorname* { m i n } } d _ { \operatorname* { m i n } } - D } .\tag{81}
$$

3. If additionally $\begin{array} { r } { D < \frac { p _ { \mathrm { m i n } } d _ { \mathrm { m i n } } } { 3 } } \end{array}$ , then T is invertible and

$$
\lVert \boldsymbol { T } ^ { - 1 } - \boldsymbol { I } \rVert _ { 1 } \leq \frac { 2 D } { p _ { \mathrm { \operatorname* { m i n } } } d _ { \mathrm { \operatorname* { m i n } } } - 3 D } .\tag{82}
$$

Proof. Part 1. Positive reconstruction distribution.

Consider arbitrary joint distribution $p _ { X W }$ with $D = \mathbb { E } [ d ( X , W ) ]$ . We have:

$$
D = \mathbb { E } [ d ( X , W ) ]\tag{83}
$$

$$
= \sum _ { w } \sum _ { x \ne w } p _ { X W } ( x , w ) d ( x , w )\tag{84}
$$

$$
\geq d _ { \operatorname* { m i n } } \sum _ { w } \sum _ { x \neq w } p _ { X W } ( x , w )\tag{85}
$$

$$
= d _ { \operatorname* { m i n } } \cdot \operatorname* { P r } \{ X \neq W \} .\tag{86}
$$

Therefore $\textstyle \operatorname* { P r } \{ X \neq W \} \leq { \frac { D } { d _ { \operatorname* { m i n } } } }$ . By the coupling inequality for total variation distance, for any $w \in \mathcal X$ we have:

$$
\operatorname* { P r } \{ X \neq W \} \geq d _ { T V } ( p _ { X } , p _ { W } )\tag{87}
$$

$$
= \operatorname* { s u p } _ { \mathcal { A } \in \mathcal { X } } | p _ { X } ( \mathcal { A } ) - p _ { W } ( \mathcal { A } ) |\tag{88}
$$

$$
\geq p _ { X } ( w ) - p _ { W } ( w )\tag{89}
$$

$$
\geq p _ { \operatorname* { m i n } } - p _ { W } ( w ) .\tag{90}
$$

Then for all $0 < D < d _ { \operatorname* { m i n } } p _ { \operatorname* { m i n } }$ , we have Pr $\begin{array} { r } { \{ X \neq W \} \subseteq \frac { D } { d _ { \operatorname* { m i n } } } < p _ { \operatorname* { m i n } } } \end{array}$ , thereby:

$$
p _ { W } ( w ) \geq p _ { \operatorname* { m i n } } - \operatorname* { P r } \{ X \neq W \} > 0 .\tag{91}
$$

This completes the proof of Part 1.

## Part 2. Transition matrix.

For all $w \in { \mathcal { X } } .$ , we have:

$$
\sum _ { x } \left| T ( x , w ) - I ( x , w ) \right| = 1 - T ( w , w ) + \sum _ { x \ne w } T ( x , w )\tag{92}
$$

$$
= 2 \mathrm { P r } \{ X \neq W | W = w \}\tag{93}
$$

$$
= 2 { \frac { \operatorname* { P r } \{ X \neq W , W = w \} } { p _ { W } ( w ) } }\tag{94}
$$

$$
\leq 2 { \frac { \operatorname* { P r } \{ X \neq W \} } { p _ { W } ( w ) } }\tag{95}
$$

$$
\leq 2 { \frac { \operatorname* { P r } \{ X \neq W \} } { p _ { \operatorname* { m i n } } - \operatorname* { P r } \{ X \neq W \} } }\tag{96}
$$

$$
\leq \frac { 2 D } { p _ { \operatorname* { m i n } } d _ { \operatorname* { m i n } } - D } .\tag{97}
$$

Therefore by the definition of induced matrix 1-norm, we have $\begin{array} { r } { \| T - I \| _ { 1 } \le \frac { 2 D } { p _ { \mathrm { m i n } } d _ { \mathrm { m i n } } - D } } \end{array}$ . This proves the upper bound in equation 81. In addition, we also have:

$$
D = \sum _ { x \neq w } p _ { X W } ( x , w ) d ( x , w )\tag{98}
$$

$$
\leq d _ { \operatorname* { m a x } } \sum _ { x \neq w } p _ { X W } ( x , w )\tag{99}
$$

$$
= d _ { \mathrm { { m a x } } } \sum _ { w } p _ { W } ( w ) \sum _ { x \neq w } p _ { X | W } ( x | w )\tag{100}
$$

$$
= d _ { \mathrm { { m a x } } } \sum _ { w } p _ { W } ( w ) \left( 1 - T ( w , w ) \right)\tag{101}
$$

$$
\leq d _ { \operatorname* { m a x } } \operatorname* { m a x } _ { w } \left( 1 - T ( w , w ) \right)\tag{102}
$$

$$
= { \frac { d _ { \operatorname* { m a x } } } { 2 } } \| T - I \| _ { 1 } .\tag{103}
$$

Therefore, we have $\begin{array} { r } { \| T - I \| _ { 1 } \geq \frac { 2 D } { d _ { \operatorname* { m a x } } } > 0 } \end{array}$ , which proves the lower bound in equation 81. This completes the proof of equation 81.

## Part 3. Existence of inverse matrix.

Now we define $E \triangleq I - T$ , then:

$$
\| E \| _ { 1 } = \| I - T \| _ { 1 } \le \frac { 2 D } { p _ { \mathrm { m i n } } d _ { \mathrm { m i n } } - D } .
$$

If $D <$ <sup>pmindmin</sup> , then $\| E \| _ { 1 } < 1$ . Therefore, consider $\textstyle \sum _ { k = 0 } ^ { \infty } E ^ { k }$ as the Neumann series, it converges and $T ^ { - 1 } = \sum _ { k = 0 } ^ { \infty } E ^ { k }$ exists. By the definition of Neumann series in Definition B.10, we also have:

$$
T ^ { - 1 } - I = E ^ { 0 } + \sum _ { k = 1 } ^ { \infty } E ^ { k } - I = \sum _ { k = 1 } ^ { \infty } E ^ { k } .\tag{104}
$$

Then:

$$
\begin{array} { r l r } {  { \| T ^ { - 1 } - I \| _ { 1 } = \| \displaystyle \sum _ { k = 1 } ^ { \infty } E ^ { k } \| _ { 1 } } } \\ & { } & { \leq \displaystyle \sum _ { k = 1 } ^ { \infty } \| E \| _ { 1 } ^ { k } } \\ & { } & { \leq \displaystyle \sum _ { k = 1 } ^ { \infty } ( \frac { 2 D } { p _ { \mathrm { m i n } } d _ { \mathrm { m i n } } - D } ) ^ { k } } \\ & { } & { = \frac { 2 D } { p _ { \mathrm { m i n } } d _ { \mathrm { m i n } } - 3 D } . } \end{array}\tag{105}
$$

Then we have equation 82. This completes the proof.

For arbitrary random vector, recall the characterization of joint-coding rate-distortion function in Theorem 2.1, i.e.,

$$
R ^ { \mathrm { J C } } ( \mathbf { D } ) \triangleq \operatorname* { m i n } _ { \mathcal { P } } I ( \mathbf { X } ; \hat { \mathbf { X } } ) .
$$

where $\mathcal { P } \triangleq \{ P _ { \hat { \mathbf { X } } | \mathbf { X } } : \mathbb { E } [ d _ { \ell } ( X _ { \ell } , \hat { X _ { \ell } } ) ] \leq D _ { \ell } , \ell = 1 , \cdot \cdot \cdot , L \}$ . For the per-letter distortion $D _ { \ell } .$ , we have the general form of Shannon lower bound:

$$
I ( \mathbf { X } ; \hat { \mathbf { X } } ) = H ( \mathbf { X } ) - H ( \mathbf { X } | \hat { \mathbf { X } } )\tag{106}
$$

$$
\geq H ( \mathbf { X } ) - \sum _ { \ell = 1 } ^ { L } H ( X _ { \ell } | \hat { X } _ { \ell } )\tag{107}
$$

$$
\mathbf { \eta } = H ( \mathbf { X } ) - \sum _ { \ell = 1 } ^ { L } H ( X _ { \ell } ) + \sum _ { \ell = 1 } ^ { L } I ( X _ { \ell } ; { \hat { X } } _ { \ell } )\tag{108}
$$

$$
\ge H ( \mathbf { X } ) - \sum _ { \ell = 1 } ^ { L } H ( X _ { \ell } ) + \sum _ { \ell = 1 } ^ { L } R _ { \ell } ( D _ { \ell } ) ,\tag{109}
$$

where $R _ { \ell } ( D _ { \ell } )$ denotes the marginal rate-distortion function of source variable $X _ { \ell } .$ . Next we prove that for source with positive distribution, the Shannon lower bound can be achieved within a nonempty distortion region.

Lemma B.12 (SLB achievability). Consider arbitrary source X with finite component alphabets $2 \leq | { \mathcal { X } } _ { \ell } | < \infty$ and positive joint distribution. For per-letter distortion function in Definition $B . 3 ,$ there exists a non-empty distortion region $\mathcal { D } ^ { \mathrm { S L B } } = \{ \mathbf { D } : 0 < D _ { \ell } \leq \varepsilon _ { \ell } , \ell = 1 , \cdot \cdot \cdot , L \}$ , such that for all $\mathbf { D } \in \mathcal { D } ^ { \mathrm { S L B } }$ , the Shannon lower bound can be attained.

Proof. The main idea of the proof is to construct a valid joint reconstruction distribution from the marginal rate-distortion optimal backward test channels, and show that the SLB can be achieved. First, we define the following positive constants:

$$
\begin{array} { r l } & { p _ { \operatorname* { m i n } } ^ { * } \triangleq \underset { \mathbf { x } } { \mathrm { m i n } } p _ { \mathbf { x } } ( \mathbf { x } ) > 0 , \quad \quad p _ { \ell , \mathrm { m i n } } \triangleq \underset { x _ { \ell } } { \mathrm { m i n } } p _ { X _ { \ell } } ( x _ { \ell } ) , \quad \quad d _ { \ell , \mathrm { m i n } } \triangleq \underset { x _ { \ell } \neq w _ { \ell } } { \mathrm { m i n } } d _ { \ell } ( x _ { \ell } , w _ { \ell } ) , } \\ & { \quad \quad \quad \tau \triangleq \frac { p _ { \mathrm { m i n } } ^ { * } } { 2 + p _ { \mathrm { m i n } } ^ { * } } , \quad \quad \varepsilon _ { \ell } \triangleq \frac { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } T } { 2 L + \tau } . } \end{array}
$$

Since $\tau < \frac { 1 } { 2 }$ , we have $\begin{array} { r } { \varepsilon _ { \ell } < \frac { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } } { 3 } } \end{array}$ . Then we start the proof.

Step 1. Marginal RD problem. For each $\ell ,$ select any $W _ { \ell }$ that achieves the marginal RD function of $X _ { \ell } .$ . Such $W _ { \ell }$ exists because $X _ { \ell }$ has finite alphabet. Denote the vectorized reconstruction probability of $p _ { W _ { \ell } }$ as $r _ { \ell }$ and the backward transition matrix as:

$$
T _ { \ell } ( x _ { \ell } , w _ { \ell } ) = p _ { X _ { \ell } | W _ { \ell } } ( x _ { \ell } | w _ { \ell } ) .\tag{110}
$$

By Lemma B.11, for all $\begin{array} { r } { 0 < D _ { \ell } \le \varepsilon _ { \ell } < \frac { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } } { 3 } } \end{array}$ , we have:

$$
\lVert T _ { \ell } - I \rVert _ { 1 } \leq \frac { 2 D _ { \ell } } { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } - D _ { \ell } } \leq \frac { \tau } { L } ,\tag{111}
$$

Apparently, $T _ { \ell } ^ { - 1 }$ exists and we further have:

$$
\| T _ { \ell } ^ { - 1 } - I \| _ { 1 } \leq \frac { 2 D _ { \ell } } { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } - 3 D _ { \ell } } \leq \frac { \tau } { L - \tau } .\tag{112}
$$

Step 2. Joint backward test channel. In this part, we first provide an explicit way to construct a valid joint backward test channel based on marginal test channels. Then we show that this constructed joint backward test channel achieves Shannon lower bound within a non-empty distortion region. We first define the product backward transition matrix form:

$$
S \triangleq \bigotimes _ { \ell = 1 } ^ { L } T _ { \ell } .\tag{113}
$$

Then for $\varepsilon _ { \ell }$ defined above and all $\mathbf { D } \in \mathcal { D } ^ { \mathrm { S L B } } , S ^ { - 1 }$ exists. Let p be the vectorized source distribution $p \mathbf { x }$ . Then we need to show that there exists a valid probability vector r satisfying:

$$
r = { S ^ { - 1 } p } .\tag{114}
$$

Since each $T _ { \ell }$ is column transition matrix, we have $\| T _ { \ell } \| _ { 1 } = 1$ for all ℓ. Let $\begin{array} { r } { N \triangleq \prod _ { \ell = 1 } ^ { L } | \mathcal { X } _ { \ell } | } \end{array}$ . Using the telescoping expansion:

$$
\bigotimes _ { \ell = 1 } ^ { L } T _ { \ell } - \bigotimes _ { \ell = 1 } ^ { L } I _ { | \mathcal { X } _ { \ell } | } = \sum _ { \ell = 1 } ^ { L } \left( \bigotimes _ { j = 1 } ^ { \ell - 1 } T _ { j } \right) \otimes ( T _ { \ell } - I _ { | \mathcal { X } _ { \ell } | } ) \otimes \left( \bigotimes _ { k = \ell + 1 } ^ { L } I _ { | \mathcal { X } _ { k } | } \right) ,\tag{115}
$$

we have the following inequality:

$$
\| S - I _ { N } \| _ { 1 } = \left\| \bigotimes _ { \ell = 1 } ^ { L } T _ { 1 } - \bigotimes _ { \ell = 1 } ^ { L } I _ { | \mathcal { X } _ { \ell } | } \right\|\tag{116}
$$

$$
\leq \sum _ { \ell = 1 } ^ { L } \left( \prod _ { j = 1 } ^ { \ell - 1 } \| T _ { j } \| _ { 1 } \right) \left\| T _ { \ell } - I _ { | \mathcal { X } _ { \ell } | } \right\| _ { 1 } \left( \prod _ { k = \ell + 1 } ^ { L } \| I _ { | \mathcal { X } _ { k } | } \| _ { 1 } \right)\tag{117}
$$

$$
= \sum _ { \ell = 1 } ^ { L } \big \| T _ { \ell } - I _ { | \mathcal { X } _ { \ell } | } \big \| _ { 1 } \leq \tau < 1 .\tag{118}
$$

By again applying the Neumann series, we have:

$$
\| S ^ { - 1 } - I _ { N } \| \leq \frac { \tau } { 1 - \tau } = \frac { p _ { \operatorname* { m i n } } ^ { * } } { 2 } .\tag{119}
$$

Then for the probability vectors, we have:

$$
\operatorname* { m a x } _ { \mathbf { w } } | r ( \mathbf { w } ) - p _ { \mathbf { X } } ( \mathbf { w } ) | = \| r - p \| _ { \infty }\tag{120}
$$

$$
\leq \| r - p \| _ { 1 }\tag{121}
$$

$$
= \| ( S ^ { - 1 } - I ) p \| _ { 1 }\tag{122}
$$

$$
\leq \| S ^ { - 1 } - I \| _ { 1 } \| p \| _ { 1 }\tag{123}
$$

$$
\leq { \frac { p _ { \mathrm { m i n } } ^ { * } } { 2 } } .\tag{124}
$$

Therefore, for the reconstruction probability vector r, we have:

$$
\begin{array} { r } { 1 . ~ r ( \mathbf { w } ) \geq p \mathbf { x } ( \mathbf { w } ) - \frac { p _ { \operatorname* { m i n } } ^ { * } } { 2 } \geq \frac { p _ { \operatorname* { m i n } } ^ { * } } { 2 } > 0 . } \end{array}
$$

$$
2 . \ \mathbf { 1 } ^ { \top } r = \mathbf { 1 } ^ { \top } S ^ { - 1 } p = \mathbf { 1 } ^ { \top } S S ^ { - 1 } p = \mathbf { 1 } ^ { \top } p = 1 .
$$

Thus r is a valid positive probability distribution vector. Furthermore, by marginalizing the constructed joint distribution $S ,$ it can be verified that $p _ { X _ { \ell } } = T _ { \ell } r _ { \ell }$ . Therefore, the marginal RD optimal backward test channels are preserved.

Step 3. Shannon lower bound achievability. For any feasible reconstruction $\hat { \mathbf { X } } .$ , the general form of Shannon lower bound (SLB) is shown in eq. (109). For the constructed auxiliary random variables W, the product backward test channel and marginal RD optimality give:

$$
I ( \mathbf { X } ; \mathbf { W } ) = H ( \mathbf { X } ) - \sum _ { \ell } H ( X _ { \ell } ) + \sum _ { \ell } R _ { \ell } ( D _ { \ell } ) .\tag{125}
$$

Therefore, the Shannon lower bound is attained by those constructed auxiliary variables W and the product backward test channel. Therefore, for all $0 < D _ { \ell } < \varepsilon _ { \ell } ,$ there always exist auxiliary variables W such that SLB is attained. This completes the proof. □

By Lemma B.9, for a K-order Markov source X as in Definition 3.6, we can construct nontrivial auxiliary random variables $U ^ { L - K }$ satisfying the Markov constraints in eqs. (32) and (33). Those constructed auxiliary variables also satisfy the Markov constraints in Definition 3.4, thereby feasible for the GFSC architecture. For example, for first-order Markov, we can construct:

$$
X _ { 1 } \to U _ { 1 } \to X _ { 2 } \to U _ { 2 } \to \cdots \to U _ { L - 1 } \to X _ { L } .\tag{126}
$$

Apparently $X _ { \ell } \to ( U _ { \ell - 1 } , U _ { \ell } ) \to X _ { - \ell }$ holds. Next, to prove the JC rate-distortion optimality of GFSC architecture for general K-order Markov source, we introduce auxiliary random variables W satisfying the following properties:

1. Define $V _ { \ell } \triangleq U _ {  { \mathbb { Z } } _ { \ell } }$ and $X _ { \ell } \to W _ { \ell } \to V _ { \ell }$ holds.

2. The backward test channel from $W _ { \ell }$ to $X _ { \ell }$ is rate-distortion optimal.

Next, we show that such auxiliary variables can always be constructed within a nonempty distortion region for source with positive distribution.

Definition B.13. Let $Q _ { \ell } \in \mathbb { R } ^ { | \mathcal { X } _ { \ell } | \times | \mathcal { V } _ { \ell } | }$ be the transition matrix constructed in Lemma B.9, representing the backward test channel maps $P _ { V _ { \ell } }$ to $P _ { X _ { \ell } } , \mathrm { i . e . , } P _ { X _ { \ell } } = Q _ { \ell } P _ { V _ { \ell } }$ with $Q _ { \ell } ( x , v ) = p _ { X _ { \ell } | V _ { \ell } } ( x | v )$

Definition B.14. Let $T _ { \ell } \in \mathbb { R } ^ { | \mathcal { X } _ { \ell } | \times | \mathcal { W } _ { \ell } | }$ be the transition matrix constructed in Lemma B.12, representing the RD optimal backward test channel for $X _ { \ell } ,$ i.e., $P _ { X _ { \ell } } ~ = ~ T _ { \ell } P _ { W _ { \ell } }$ with $T _ { \ell } ( x , w ) =$ $p _ { X _ { \ell } | W _ { \ell } } ( x | w )$

Definition B.15. For $Q _ { \ell } ,$ define the positive term $\begin{array} { r } { q _ { \ell , \mathrm { m i n } } ^ { ( \delta ) } \triangleq \operatorname* { m i n } _ { x , v } Q _ { \ell } ( x , v ) } \end{array}$

Before we start, we define the following positive constant:

$$
\eta _ { \ell } ( \delta ) \triangleq \frac { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } q _ { \ell , \mathrm { m i n } } ^ { ( \delta ) } } { 2 + 3 q _ { \ell , \mathrm { m i n } } ^ { ( \delta ) } } .\tag{127}
$$

Lemma B.16 (RD-optimal refinement of clique auxiliary variables). Fix the clique-wise auxiliary variables constructed by equation 68 with $0 < \delta < \delta _ { 0 }$ . Assume the finite-alphabet and distortion conditions established above. Let $\varepsilon _ { \ell } > 0$ be the distortion thresholds proved in Lemma B.12. Define $\epsilon _ { \ell } ( \delta ) \triangleq \operatorname* { m i n } \{ \varepsilon _ { \ell } , \eta _ { \ell } ( \delta ) \}$ and the corresponding nonempty distortion region:

$$
\mathcal { D } _ { \epsilon } \triangleq \{  { \mathbf { D } } : 0 < D _ { \ell } \leq \epsilon _ { \ell } ( \delta ) , \ell = 1 , \cdots , L \} .\tag{128}
$$

Then for any $\mathbf { D } \in \mathcal { D } _ { \epsilon }$ , there exist auxiliary variables $\mathbf { W } \in \mathcal { X } _ { 1 } \times \cdots \times \mathcal { X } _ { L }$ such that:

1. The Markov conditions hold:

$$
X _ { \ell }  W _ { \ell }  U _ { \bar { \mathcal { Z } } _ { \ell } } ^ { ( \delta ) }  X _ { - \ell } , \ell = 1 , \cdot \cdot \cdot , L .\tag{129}
$$

2. The reconstruction $\hat { X } _ { \ell } \ \triangleq \ W _ { \ell }$ and the corresponding backward test channel $p _ { X _ { \ell } | W _ { \ell } }$ are jointly rate-distortion optimal.

Proof. Step 1. Construction.

For a given δ such that $0 < \delta < \delta _ { 0 }$ , by the definition of $\eta _ { \ell } ( \delta )$ and $\Delta _ { \ell } ( \delta )$ , we have:

$$
\Delta _ { \ell } ( \delta ) \triangleq \operatorname* { m i n } _ { g _ { \ell } } \mathbb { E } [ d _ { \ell } ( X _ { \ell } , g _ { \ell } ( U _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) ) ]\tag{130}
$$

$$
= \sum _ { u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } } q ( u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) \operatorname* { m i n } _ { \hat { x } _ { \ell } \in \mathcal { X } _ { \ell } } \sum _ { x _ { \ell } } Q ^ { ( \delta ) } ( x _ { \ell } , u _ { \mathbb { Z } _ { \ell } } ) d _ { \ell } ( x _ { \ell } , \hat { x } _ { \ell } )\tag{131}
$$

$$
\geq \sum _ { u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } } q ( u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) d _ { \ell , \mathrm { m i n } } \operatorname* { m i n } _ { \hat { x } _ { \ell } } \sum _ { x _ { \ell } \neq \hat { x } _ { \ell } } Q ^ { ( \delta ) } ( x _ { \ell } , u _ { \mathcal { T } _ { \ell } } )\tag{132}
$$

$$
\geq \sum _ { u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } } q ( u _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) d _ { \ell , \mathrm { m i n } } \operatorname* { m i n } _ { \hat { x } _ { \ell } } \sum _ { x _ { \ell } \neq \hat { x } _ { \ell } } q _ { \ell , \mathrm { m i n } } ^ { ( \delta ) }\tag{133}
$$

$$
= ( | \mathcal { X } _ { \ell } | - 1 ) d _ { \ell , \mathrm { { m i n } } } q _ { \ell , \mathrm { { m i n } } } ^ { ( \delta ) } \geq \eta _ { \ell } ( \delta ) .\tag{134}
$$

Then for $\mathcal { D } _ { \epsilon }$ defined in equation 128, we have:

$$
\mathcal { D } _ { \epsilon } \subseteq \mathcal { D } ^ { \mathrm { S L B } } , \quad \mathcal { D } _ { \epsilon } \subseteq \mathcal { D } ^ { \mathrm { A U X } } .\tag{135}
$$

Therefore, for all $\mathbf { D } \in \mathcal { D } _ { \epsilon }$

1. Lemma B.12 provides a backward test channel construction from W to $\mathbf { X } \mathbf { \cdot }$

$$
p _ { \mathbf { X } | \mathbf { W } } ( \mathbf { x } | \mathbf { w } ) = \prod _ { \ell = 1 } ^ { L } T _ { \ell } ( x _ { \ell } , w _ { \ell } ) ,
$$

such that the Shannon lower bound can be attained. Thereby it is jointly rate-distortion optimal. $T _ { \ell }$ denotes the marginal rate-distortion optimal test channel from $W _ { \ell }$ to $X _ { \ell }$

2. Lemma B.9 provides a backward test channel construction from $( U ^ { ( \delta ) } ) ^ { L - K }$ to X such that the induced distortion $\Delta _ { \ell } ( \delta )$ is positive and the following Markov conditions hold:

$$
\begin{array} { r l r l } & { U _ { j } ^ { ( \delta ) } \to X _ { j } ^ { j + K } \to ( { \bf X } , { U } _ { - j } ^ { ( \delta ) } ) , \qquad } & & { j = 1 , \ldots , L - K , } \\ & { X _ { \ell } \to U _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } \to X _ { - \ell } , \qquad } & & { \ell = 1 , \ldots , L . } \end{array}
$$

Denote the backward test channel from $V _ { \ell } \triangleq U _ { \ Z }$ to $X _ { \ell }$ as $Q _ { \ell }$

Then it suffices to show that such constructed auxiliary variables $W _ { \ell }$ and $V _ { \ell }$ also satisfy the Markov conditions in equation 129.

## Step 2. Markov conditions.

For constructed $T _ { \ell } \in \mathbb { R } ^ { | \mathcal { X } _ { \ell } | \times | \mathcal { W } _ { \ell } | }$ and $Q _ { \ell } \in \mathbb { R } ^ { | \mathcal { X } _ { \ell } | \times | \mathcal { V } _ { \ell } | }$ , the Markov condition in equation 129 boils down to finding a valid transition matrix $A _ { \ell } \in \mathbb { R } ^ { | \mathcal { W } _ { \ell } | \times | \mathcal { V } _ { \ell } | }$ such that

$$
Q _ { \ell } = T _ { \ell } A _ { \ell } .\tag{136}
$$

As we already shown in Lemma B.11, $T _ { \ell }$ is close to but not exactly an identity matrix and $T _ { \ell } ^ { - 1 }$ exists. This is equivalent to showing that $A _ { \ell } = T _ { \ell } ^ { - 1 } Q _ { \ell }$ is a valid transition matrix. Non-negativity

We first show that all the entries of $T _ { \ell } ^ { - 1 } Q _ { \ell }$ are non-negative. First, by Lemma B.11, we have

$$
\left\| A _ { \ell } - Q _ { \ell } \right\| _ { 1 } = \left\| ( T _ { \ell } ^ { - 1 } - I ) Q _ { \ell } \right\| _ { 1 }\tag{137}
$$

$$
\leq \| T _ { \ell } ^ { - 1 } - I \| _ { 1 } \| Q _ { \ell } \| _ { 1 }\tag{138}
$$

$$
\leq \frac { 2 D _ { \ell } } { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } - 3 D _ { \ell } }\tag{139}
$$

$$
\leq \frac { 2 \eta _ { \ell } ( \delta ) } { p _ { \ell , \mathrm { m i n } } d _ { \ell , \mathrm { m i n } } - 3 \eta _ { \ell } ( \delta ) }\tag{140}
$$

$$
= q _ { \ell , \mathrm { m i n } } ^ { ( \delta ) } .\tag{141}
$$

Then we have:

$$
A _ { \ell } ( w _ { \ell } , v _ { \ell } ) = Q _ { \ell } ( w _ { \ell } , v _ { \ell } ) - ( Q _ { \ell } ( w _ { \ell } , v _ { \ell } ) - A _ { \ell } ( w _ { \ell } , v _ { \ell } ) )\tag{142}
$$

$$
\geq Q _ { \ell } ( w _ { \ell } , v _ { \ell } ) - \| A _ { \ell } - Q _ { \ell } \| _ { 1 }\tag{143}
$$

$$
\ge q _ { \ell , \mathrm { m i n } } ^ { ( \delta ) } - q _ { \ell , \mathrm { m i n } } ^ { ( \delta ) } = 0 .\tag{144}
$$

Therefore all entries of $A _ { \ell }$ are non-negative by construction.

Column sum.

Moreover, for the column sum of $A _ { \ell } ,$ , we directly have:

$$
\mathbf { 1 } ^ { \top } A _ { \ell } = \mathbf { 1 } ^ { \top } T _ { \ell } ^ { - 1 } Q _ { \ell } = \mathbf { 1 } ^ { \top } T _ { \ell } T _ { \ell } ^ { - 1 } Q _ { \ell } = \mathbf { 1 } ^ { \top } Q _ { \ell } = \mathbf { 1 } ^ { \top } .\tag{145}
$$

Combining eqs. (144) and (145), this proves that $A _ { \ell }$ is a valid transition matrix.

Therefore, for any $\mathbf { D } \in \mathcal { D } _ { \epsilon }$ , we can always construct W with jointly rate-distortion optimal backward test channel. This completes the proof. □

Theorem B.17 (JC optimality of GFSC). Consider a K-order Markov source X with positive distribution. There exist a non-empty distortion region $\mathcal { D } _ { \epsilon } \triangleq \{  { \mathbf { D } } : 0 < D _ { \ell } \leq \epsilon _ { \ell } , \ell = 1 , \cdots , L \}$ , such that for all $\mathbf { D } \in \mathcal { D } _ { \epsilon }$ , the minimum information sum-rate $R _ { K , \mathrm { s u m } } ^ { \dot { \mathrm { G F S C } } } ( \mathbf { D } )$ achieves the JC rate-distortion function:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) = R ^ { \mathrm { J C } } ( \mathbf { D } ) , \quad \forall \mathbf { D } \in \mathcal { D } _ { \epsilon } .\tag{146}
$$

## Proof. Step 1. Achievability.

Fix an admissible δ in the clique-wise auxiliary variable construction such that $0 < \delta < \delta _ { 0 }$ . Choose the positive threshold $\epsilon _ { \ell } ( \delta )$ supplied by the Lemma B.16, and fix any $\mathbf { D } \in \mathcal { D } _ { \epsilon }$ . Then Lemma B.16 provides jointly distributed variables:

$$
\left( \mathbf { X } , ( U ^ { ( \delta ) } ) ^ { L - K } , \mathbf { W } \right)\tag{147}
$$

with the following properties:

1. The source marginal is still the prescribed $p _ { \mathbf { X } }$

2. The Markov conditions hold:

$$
X _ { \ell }  W _ { \ell }  U _ { \mathbb { Z } _ { \ell } } , \quad \ell = 1 , \cdot \cdot \cdot , L .\tag{148}
$$

3. The refinement auxiliary variables W achieve the joint-coding rate-distortion function $R ^ { \mathrm { J C } } ( \mathbf { D } )$ for all $\mathbf { D } \in \mathcal { D } _ { \epsilon }$

We now construct augmented auxiliary variables satisfying the GFSC architecture. Specifically, we define:

$$
Z _ { \ell } \triangleq \left\{ \begin{array} { l l } { ( U _ { \ell } ^ { ( \delta ) } , W _ { \ell } ) , } & { 1 \le \ell \le L - K , } \\ { W _ { \ell } , } & { L - K < \ell \le L . } \end{array} \right.\tag{149}
$$

Apparently, auxiliary variables $\{ Z _ { \ell } \} _ { \ell = 1 } ^ { L }$ are just regrouping of constructed $\{ U _ { j } ^ { ( \delta ) } \} _ { j = 1 } ^ { L - K }$ and W. By equation 149, the previously received messages $Z ^ { \ell - 1 }$ contains $\left( ( U ^ { ( \delta ) } ) ^ { \ell - 1 } , W ^ { \bar { \ell } - 1 } \right)$ . We first verify the required encoder Markov conditions in equation 5, i.e.,

$$
Z _ { \ell } \to ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } , Z ^ { \ell - 1 } ) \to \mathbf { X } .\tag{150}
$$

For $1 \leq \ell \leq L - K$ , we have the following:

$$
\begin{array} { r } { U _ { \ell } ^ { ( \delta ) }  X _ { \ell } ^ { \ell + K }  ( { \mathbf X } , U _ { - \ell } ^ { ( \delta ) } ) } \\ { W _ { \ell }  ( X _ { \ell } , U _ { \mathbb { Z } _ { \ell } } ^ { ( \delta ) } )  X _ { - \ell } } \end{array} \} \Rightarrow Z _ { \ell }  ( X _ { \ell } ^ { \ell + K } , U _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } )  { \mathbf X } .\tag{151}
$$

For $L - K < \ell \leq L$ , we have $Z _ { \ell } = W _ { \ell }$ . Then:

$$
Z _ { \ell } \to ( X _ { \ell } , U _ { \mathcal { T } _ { \ell } } ^ { ( \delta ) } ) \to X _ { - \ell }\tag{152}
$$

Therefore, the encoder Markov conditions in equation 5 hold for every stage. Next, define the deterministic reconstruction as

$$
\hat { X } _ { \ell } = W _ { \ell }\tag{153}
$$

For $\ell \leq L - K$ , the decoder extracts the second component of $Z _ { \ell }$ . For $\ell > L - K .$ , the decoder directly keep $Z _ { \ell }$ itself. Thus this is admissible for the defined stronger decoder $g _ { \ell } ( Z ^ { \ell } )$ . For ℓ-th encoding stage and augmented auxiliary variables $Z _ { \ell } ,$ by the conditional source coding theorem, the associate rate $\bar { R } _ { \ell }$ is:

$$
{ \bar { R } } _ { \ell } = I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; Z _ { \ell } | Z ^ { \ell - 1 } )\tag{154}
$$

which also belongs to the information RD region in Definition 3.4. By the Markov condition in eq. (5), we have

$$
I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; Z _ { \ell } | Z ^ { \ell - 1 } ) = I ( { \mathbf { X } } ; Z _ { \ell } | Z ^ { \ell - 1 } )\tag{155}
$$

Consequently, we have the following sum-rate form by chain rule:

$$
\sum _ { \ell = 1 } ^ { L } \bar { R } _ { \ell } = \sum _ { \ell = 1 } ^ { L } I ( X _ { \ell } ^ { \operatorname* { m i n } \{ \ell + K , L \} } ; Z _ { \ell } | Z ^ { \ell - 1 } )\tag{156}
$$

$$
\mathbf { \eta } = \sum _ { \ell = 1 } ^ { L } I ( \mathbf { X } ; Z _ { \ell } | Z ^ { \ell - 1 } )\tag{157}
$$

$$
\mathbf { \Sigma } = I ( \mathbf { X } ; \mathbf { Z } )
$$

$$
\mathbf { \xi } = I ( \mathbf { X } ; U ^ { L - K } , \mathbf { W } )\tag{158}
$$

(159)

$$
{ \bf \Xi } = I ( { \bf X } ; { \bf W } ) = R ^ { \mathrm { J C } } ( { \bf D } ) .\tag{160}
$$

Since this is just one possible realization of GFSC architecture, then we have:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) \leq R ^ { \mathrm { J C } } ( \mathbf { D } ) , \quad \forall \mathbf { D } \in \mathcal { D } _ { \epsilon } .\tag{161}
$$

## Step 2. Converse.

By the Shannon lower bound, we have:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( \mathbf { D } ) \geq R ^ { \mathrm { S L B } } ( \mathbf { D } ) = R ^ { \mathrm { J C } } ( \mathbf { D } ) , \quad \forall \mathbf { D } \in \mathcal { D } _ { \epsilon } .\tag{162}
$$

Combine with eq. (161), we have $R _ { K , \mathrm { s u m } } ^ { \mathrm { G F S C } } ( { \bf D } ) = R ^ { \mathrm { J C } } ( { \bf D } )$ for all $\mathbf { D _ { \theta } } \in \mathcal { D } _ { \epsilon }$ . This completes the proof. □

## C GAUSS-MARKOV SOURCE MODELS UNDER MSE DISTORTION

First, we review joint coding rate-distortion function for Gaussian source. Consider Gaussian source $\mathbf { X } \sim { \mathcal { N } } ( 0 , \Sigma )$ with $\Sigma \succ 0$ . Let $\Sigma _ { e } \triangleq \mathbb { E } [ ( \mathbf { X } - \hat { \mathbf { X } } ) ( \mathbf { X } - \hat { \mathbf { X } } ) ^ { \top } ]$ be the error covariance, the JC ratedistortion function in Theorem 2.1 admits a closed-form characterization

$$
R ^ { \mathrm { J C G } } ( \mathbf { D } ) = \operatorname* { m i n } _ { \Sigma _ { e } } \frac { 1 } { 2 } \log \frac { | \Sigma | } { | \Sigma _ { e } | } \quad \mathrm { s . t . } \quad 0 \prec \Sigma _ { e } \preceq \Sigma , ( \Sigma _ { e } ) _ { \ell \ell } \leq D _ { \ell } .\tag{163}
$$

Moreover, in the distortion region $\mathcal { D } ^ { \mathrm { J C } } \triangleq \{ \mathbf { D } : \mathrm { d i a g } ( D _ { 1 } , \cdot \cdot \cdot , D _ { L } ) \preceq \Sigma \}$ , it simplifies to

$$
R ^ { \mathrm { J C G } } ( \mathbf { D } ) = \frac { 1 } { 2 } \log \frac { | \Sigma | } { \prod _ { \ell = 1 } ^ { L } D _ { \ell } } .\tag{164}
$$

This coincides with the Shannon lower bound (SLB) (Cover, 1999), which is tight over $\mathcal { D } ^ { \mathrm { J C } }$ . Next, we briefly review Gaussian graphical models (GGMs).

## C.1 GAUSSIAN GRAPHICAL MODEL

For Gaussian sources, as previously mentioned in Section 3.1, its graphical structure can be obtained directly from its inverse covariance matrix, i.e. information matrix or precision matrix. Let $J \triangleq \Sigma ^ { - 1 }$ denote the information matrix. The corresponding Gaussian graphical model is the undirected graph

$\mathcal { G } ( J ) = ( \nu , \mathcal { E } ( J ) )$ , with node set $\mathcal { V } \triangleq \{ 1 , \dots , L \}$ and edge set $\mathcal { E } ( J ) \triangleq \{ \{ j , k \} : J _ { j k } \neq 0 \}$ , where $J _ { j k }$ denotes the $( j , k )$ -th entry. For any ${ \mathcal { A } } \subseteq { \mathcal { V } } .$ , write $X _ { \mathcal { A } } \triangleq ( X _ { i } ) _ { i \in \mathcal { A } }$ . The neighborhood of source node ℓ is

$$
{ \mathcal { N } } ( \ell ) \triangleq \{ i \in \mathcal { V } \backslash \{ \ell \} : \{ \ell , i \} \in \mathcal { E } ( J ) \} .
$$

For Gaussian sources, a missing edge has an exact conditional-independence interpretation:

$$
J _ { j k } = 0 \quad \Longleftrightarrow \quad X _ { j } \perp \perp X _ { k } | X _ { \mathcal V \backslash \{ j , k \} } .
$$

Consequently, each variable is conditionally independent of all non-neighboring variables given its local neighborhood, ${ \mathrm { i . e . ~ } } X _ { \ell } \ \bot \ X _ { \mathcal { V } \setminus ( { \mathcal { N } } ( \ell ) \cup \{ \ell \} ) } | X _ { \mathcal { N } ( \ell ) }$ . For a K-order Gauss-Markov source, J is K-banded, i.e. $J _ { j k } = 0$ for $| j - k | > K$ . Therefore,

$$
\mathcal { N } ( \ell ) \subseteq \{ \ell - K , \ldots , \ell - 1 , \ell + 1 , \ldots , \ell + K \} \cap \mathcal { V } .
$$

Thus, conditioned on at most 2K neighboring frames, $X _ { \ell }$ is independent of all remaining source variables. Figure 3 illustrates this structure for a second-order Gauss-Markov source, whose edges connect only frames separated by at most two time steps.

We introduce several definitions that can be used to specify a subset of edges in GGMs, and will be used later to define a sequential coding order for the Gaussian version coding system. For a K-order model, we first define

$$
\mathcal { E } ^ { ( K ) } \triangleq \{ ( j , k ) : 1 \leq j < k \leq L , k - j \leq K \} .\tag{165}
$$

Here, equation 165 contains all frame pairs within the K neighbors, which also represents the edge set of a K-order Gauss-Markov model. As illustrated in Figure $3 , \mathcal { E } ^ { ( 2 ) }$ includes all pairs with form $( j , j + 1 )$ and $( j , j + 2 )$ , which are processed according to the ordering in equation 166. Then, we process the pairs in $\mathcal { E } ^ { ( K ) }$ lexicographically. Specifically, let

$$
( i , \ell ) \prec ( j , k ) \quad \Longleftrightarrow \quad i < j { \mathrm { ~ o r ~ } } ( i = j { \mathrm { ~ a n d ~ } } \ell < k ) .\tag{166}
$$

For any pair-indexed collection $Z ^ { N } \triangleq ( Z _ { j , k } ) _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ with $N \triangleq | \mathcal { E } ^ { ( K ) } |$ , define

$$
Z _ { < ( j , k ) } \triangleq \big ( Z _ { i , \ell } \big ) _ { ( i , \ell ) \in \mathcal { E } ^ { ( K ) } , ( i , \ell ) \prec ( j , k ) } ,\tag{167}
$$

$$
Z _ { \leq j } \triangleq \left( Z _ { i , \ell } \right) _ { \left( i , \ell \right) \in { \mathcal { E } } ^ { \left( K \right) } , i \leq j } .\tag{168}
$$

Here, $Z _ { < ( j , k ) }$ contains all variables generated before stage $( j , k )$ , whereas $Z _ { \leq j }$ contains all variables available after completing stage $j .$ . We apply this convention to messages and auxiliary variables, yielding $M _ { < ( j , k ) } , M _ { \leq j } , U _ { < ( j , k ) }$ , and $U _ { \leq j }$

## C.2 GFSC WITH PAIR-WISE FACTORIZATION

As discussed in Section 3.1, the clique factorization can be further decomposed as pair-wise factorization. In addition, as discussed in Section 3.5, the pair-wise interaction can be formalized as linear interpolation between frame pairs. We rename the further factorized system as Interpolation based Sequential Coding (ISC). Let $\mathbf { X } = \left( X _ { 1 } , \cdots , X _ { L } \right)$ be a sequentially observed source and K the allowed delay relaxation $( K < L - 1 )$ . We define the encoding and decoding mappings of ISC-K architecture as follows.

Encoding. The encoding proceeds sequentially over the set $\mathcal { E } ^ { ( K ) }$ . For each stage $( j , k ) \in \mathcal { E } ^ { ( K ) }$ , the encoder first constructs a “virtual frame” $\tilde { X } _ { j , k }$ with an pair-wise operator $h _ { j , k } : \mathcal { X } _ { j } \times \mathcal { X } _ { k }  \tilde { \mathcal { X } } _ { j , k }$ $\tilde { X } _ { j , k }$ is then encoded into message using a zero-delay sequential encoder $\phi _ { j , k } : \tilde { \mathcal { X } } _ { j , k } \times \mathcal { M } _ { < ( j , k ) } \times$ ${ \dot { \mathcal { W } } }  { \mathcal { M } } _ { j , k }$ . The encoding mapping is given by $f _ { j , k } = \phi _ { j , k } \circ h _ { j , k } , \mathrm { i . e }$

$$
f _ { j , k } : \mathcal { X } _ { j } \times \mathcal { X } _ { k } \times \mathcal { M } _ { < ( j , k ) } \times \mathcal { W }  \mathcal { M } _ { j , k } , \quad ( j , k ) \in \mathcal { E } ^ { ( K ) } .\tag{169}
$$

This formulation makes explicit that delay is handled through interpolation between $( X _ { j } , X _ { k } )$ , while the coding process itself remains strictly sequential. There are generally $N \triangleq | \mathcal { E } ^ { ( \check { K } ) } |$ encoding stages. $\bar { N = } L - 1$ for 1-delay system and $N \geq L$ for $K > 1$ delay system.

Decoding. The ℓ-th decoder reconstructs $X _ { \ell }$ using all messages available up to ℓ-stage decoding. The decoding mapping is given by:

$$
g _ { \ell } : { \mathcal { M } } _ { \leq \ell } \times { \mathcal { W } }  \hat { { \mathcal { X } } } _ { \ell } , \quad \ell = 1 , \cdot \cdot \cdot , L .\tag{170}
$$

There are L decoding stages for L source frames respectively. So encoders and decoders are not necessarily matched one-by-one. Notice that we can combine and define a multi-stage encoder $\tilde { f } _ { \ell } \triangleq f _ { \ell , \ell + K ^ { 0 } } \cdot \cdot \cdot \circ f _ { \ell , \ell + 1 }$ with input $X _ { \ell } ^ { \ell + K }$ . This can be seen as a special case of GFSC architecture.

## C.3 RATE-DISTORTION ANALYSIS

For convenience, we only characterize the information RD region and sum-rate function of ISC-K architecture. The analysis imposes no assumptions on the source distribution and therefore applies to general sources. We first introduce the notation used throughout this section. For the length-L source and distortion sequences, define:

$$
{ \bf X } \triangleq ( X _ { 1 } , \cdots , X _ { L } ) , \qquad { \bf D } \triangleq ( D _ { 1 } , \cdots , D _ { L } ) ,
$$

For sequences indexed by the ordered set $\mathcal { E } ^ { ( K ) }$ , define:

$$
\begin{array} { r } { \begin{array} { l l } { U ^ { N } \triangleq \{ U _ { j , k } : ( j , k ) \in { \mathcal E } ^ { ( K ) } \} , } & { R ^ { N } \triangleq \{ R _ { j , k } : ( j , k ) \in { \mathcal E } ^ { ( K ) } \} , } \end{array} } \end{array}
$$

where $N \triangleq | { \mathcal { E } } ^ { ( K ) } |$ and the entries follow the prescribed ordering of $\mathcal { E } ^ { ( K ) }$ . With this notation in place, we begin with the definition of information rate-distortion region associated with the ISC-K system.

Definition C.1 (Information RD region of ISC-K). For a given source X, the single-letter information RD region of ISC-K, denoted by $\mathcal { R } _ { K }$ , consists of all $\check { ( } R ^ { N } , \mathbf { D } )$ satisfying

$$
R _ { j , k } \ge I ( \tilde { X } _ { j , k } ; U _ { j , k } | U _ { < j , k } ) , \quad ( j , k ) \in \mathcal { E } ^ { ( K ) } ,\tag{171}
$$

$$
D _ { \ell } \geq \mathbb { E } [ \| X _ { \ell } - \hat { X } _ { \ell } \| ^ { 2 } ] , \quad \ell = 1 , \cdot \cdot \cdot , L ,\tag{172}
$$

for all auxiliary random variables $\{ \tilde { X } _ { j , k } , U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ and $\hat { \bf X }$ satisfying $\tilde { X } _ { j , k } = h _ { j , k } ( X _ { j } , X _ { k } )$ ${ \hat { X } } _ { \ell } = g _ { \ell } ( U _ { \leq \ell } )$ and the Markov constraints:

$$
U _ { j , k } \to ( \tilde { X } _ { j , k } , U _ { < j , k } ) \to ( X _ { j } , X _ { k } , U _ { < j , k } ) \to { \bf X } ,\tag{173}
$$

where $h _ { j , k } ( \cdot )$ and $g _ { \ell } ( \cdot )$ are deterministic functions. For a given D, the information rate region $\mathcal { R } _ { K } ( { \bf D } )$ is the closure of all $R ^ { N }$ such that $( R ^ { N } , { \bf D } ) \in \mathcal { R } _ { K }$

Theorem C.2 (Information minimum sum-rate). For a given distortion tuple D, the information minimum sum-rate function of ISC-K is

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( { \bf D } ) = \operatorname* { m i n } I ( { \bf X } ; U ^ { N } ) .\tag{174}
$$

The minimum is taken over all auxiliary variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ satisfying Markov chains in equation 173 and the distortion constraints $D _ { \ell } \geq \mathbb { E } [ \| X _ { \ell } - g _ { \ell } ( U _ { \leq \ell } ) \| ^ { 2 } ] f o r a l l ~ \ell .$

Proof. From Definition C.1, for a given distortion tuple D, the minimum sum-rate is given by

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( \mathbf { D } ) = \operatorname* { m i n } \sum _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } } I ( \tilde { X } _ { j , k } ; U _ { j , k } | U _ { < j , k } ) ,\tag{175}
$$

where the minimization is over all feasible auxiliary variables satisfying the constraints in equation 173 and the distortion constraints. Since $\tilde { X } _ { j , k } = h _ { j , k } ( X _ { j } , X _ { k } )$ is a deterministic function of $( X _ { j } , X _ { k } )$ , we have

$$
I ( \tilde { X } _ { j , k } ; U _ { j , k } | U _ { < j , k } ) \leq I ( X _ { j } , X _ { k } ; U _ { j , k } | U _ { < j , k } ) .
$$

On the other hand, by the Markov chain $U _ { j , k } \to ( \tilde { X } _ { j , k } , U _ { < j , k } ) \to ( X _ { j } , X _ { k } , U _ { < j , k } )$ , the data processing inequality implies

$$
I ( \tilde { X } _ { j , k } ; U _ { j , k } | U _ { < j , k } ) \geq I ( X _ { j } , X _ { k } ; U _ { j , k } | U _ { < j , k } ) .
$$

Combining the above two inequalities gives $I ( \tilde { X } _ { j , k } ; U _ { j , k } | U _ { < j , k } ) = I ( X _ { j } , X _ { k } ; U _ { j , k } | U _ { < j , k } )$ . Substituting this into the sum-rate expression, we have

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( \mathbf { D } ) = \operatorname* { m i n } \sum _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } } I ( X _ { j } , X _ { k } ; U _ { j , k } | U _ { < j , k } )\tag{176}
$$

$$
= \operatorname* { m i n } \sum _ { ( j , k ) \in { \mathcal { E } } ^ { ( K ) } } I ( { \bf X } ; U _ { j , k } | U _ { < j , k } )\tag{177}
$$

$$
{ \bf \Gamma } = \operatorname* { m i n } I ( { \bf X } ; U ^ { N } ) ,\tag{178}
$$

where equation 177 follows from the Markov chain $U _ { j , k } \ \to \ ( X _ { j } , X _ { k } , U _ { < j , k } ) \ \to \ \mathbf { X }$ and equation 178 follows from the chain rule of mutual information under the ordering induced by $\mathcal { E } ^ { ( K ) }$ This completes the proof. □

## C.4 JOINT-CODING OPTIMALITY FOR GAUSS-MARKOV SOURCES

In this section, we prove that the ISC architecture achieves JC rate-distortion optimality for Gauss-Markov sources within a nontrivial distortion region. Typically, the Gaussian graphical model (GGM) associated with a Gaussian source can be represented by $\mathcal { G } = ( \nu , \mathcal { E } )$ , where $\bar { \boldsymbol { \mathcal { E } } } = \{ \{ i , j \}$ $J _ { i j } ~ \neq ~ 0 \}$ denotes the set of undirected edges. For notational convenience, in this section, we identify each undirected edge $\{ i , j \}$ with the unique ordered pair $( i , j )$ satisfying $i < j$ and write $\mathcal { E } = \{ \bar { ( } i , j ) : J _ { i j } \neq 0 , i < \bar { j } \}$ . This convention merely provides a unique indexing of the edges. It neither assigns directions to edges nor changes the underlying graph.

Lemma C.3 (Existence of a nontrivial distortion region). Consider correlated Gaussian vector $\mathbf { X } \ \sim \ { \mathcal { N } } ( 0 , \Sigma )$ with $\Sigma \succ 0$ and the GGM $\mathcal { G } ( J )$ induced by its information matrix $J \ \triangleq \ \Sigma ^ { - 1 }$ Let $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } }$ be a set of Gaussian random variables jointly distributed with X satisfying:

$$
U _ { j , k } = \omega _ { j , k } X _ { j } + ( 1 - \omega _ { j , k } ) X _ { k } + N _ { j , k } , \quad ( j , k ) \in \mathcal E ,\tag{179}
$$

and the Markov chains:

$$
X _ { j } \to \left( X _ { \mathcal { N } ( j ) \backslash \{ k \} } , U _ { j , k } \right) \to X _ { k } , \quad ( j , k ) \in \mathcal { E } ,\tag{180}
$$

$$
X _ { \ell } \to U _ { \ell , N ( \ell ) } \to X _ { - \ell } , \quad \ell = 1 , \cdot \cdot \cdot , L ,\tag{181}
$$

where $N _ { j , k } \sim \mathcal N ( 0 , \sigma _ { N _ { j , k } } ^ { 2 } )$ are Gaussian random variables independent of X. Then for any $\omega _ { j , k } \in$ $\mathbb { R } \backslash \{ 0 , 1 \}$ }, the induced MSE distortion for each source variable $X _ { \ell } ,$ , denoted as $\epsilon _ { \ell } \triangleq \mathbb { E } [ \Vert X _ { \ell } -$ $\mathbb { E } [ \dot { X } _ { \ell } | U _ { \ell , N ( \ell ) } ] \| ^ { 2 } ]$ , is nontrivial, i.e.,

$$
0 < \epsilon _ { \ell } < \bar { D } _ { \ell } ,\tag{182}
$$

where $\bar { D } _ { \ell } \triangleq \mathbb { E } [ \| X _ { \ell } - \mathbb { E } [ X _ { \ell } | X _ { \mathcal { N } ( \ell ) } ] \| ^ { 2 } ] .$

## Proof. Construction of appropriate auxiliary variables:

First we show how to construct auxiliary random variables satisfying eqs. (179) to (181). For Gaussian source $\mathbf { X } \sim { \mathcal { N } } ( 0 , \Sigma )$ with information matrix $J \triangleq \Sigma ^ { - 1 }$ , its probability density function can be written as:

$$
p ( \mathbf { x } ) = { \frac { 1 } { ( 2 \pi ) ^ { { \frac { L } { 2 } } } | { \boldsymbol { \Sigma } } | ^ { \frac { 1 } { 2 } } } } \exp \left( - { \frac { 1 } { 2 } } \mathbf { x } ^ { \top } J \mathbf { x } \right)\tag{183}
$$

$$
= { \frac { 1 } { ( 2 \pi ) ^ { \frac { L } { 2 } } | \Sigma | ^ { \frac { 1 } { 2 } } } } \exp \left( - { \frac { 1 } { 2 } } \sum _ { \ell = 1 } ^ { L } J _ { \ell \ell } x _ { \ell } ^ { 2 } - \sum _ { 1 \leq j < k \leq L } J _ { j k } x _ { j } x _ { k } \right)\tag{184}
$$

$$
= \frac { 1 } { ( 2 \pi ) ^ { \frac { L } { 2 } } | \Sigma | ^ { \frac { 1 } { 2 } } } \exp \left( - \frac { 1 } { 2 } \sum _ { \ell = 1 } ^ { L } J _ { \ell \ell } x _ { \ell } ^ { 2 } - \sum _ { ( j , k ) \in \mathcal { E } } J _ { j k } x _ { j } x _ { k } \right) ,\tag{185}
$$

where equation 184 follows from that J is a symmetric matrix, and equation 185 follows from that $J _ { j k } = 0$ for $( j , k ) \notin \mathcal { E }$ Thus, the dependency structure of a Gaussian source X is completely characterized by the edge set E of its GGM. By equation 179, each auxiliary variable $U _ { j , k }$ is associated with edge $( j , k )$ in the GGM and independent additive Gaussian noise $N _ { j , k }$ . Therefore, its conditional density function given $( X _ { j } , X _ { k } ) \overset { \cdot } { = } \left( x _ { j } , x _ { k } \right)$ can be written as:

$$
p ( u _ { j , k } | x _ { j } , x _ { k } ) = \frac { 1 } { \sqrt { 2 \pi \sigma _ { N _ { j , k } } ^ { 2 } } } \exp \left( - \frac { ( u _ { j , k } - \omega _ { j , k } x _ { j } - \bar { \omega } _ { j , k } x _ { k } ) ^ { 2 } } { 2 \sigma _ { N _ { j , k } } ^ { 2 } } \right) ,\tag{186}
$$

where $\bar { \omega } _ { j , k } \triangleq 1 - \omega _ { j , k }$ . In addition, since those noise items $\{ N _ { j , k } \} _ { ( j , k ) \in \mathcal { E } }$ are also mutually independent with each other, $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } }$ are conditionally independent given X, i.e., $p ( \mathbf { u } | \mathbf { x } ) =$ $\begin{array} { r } { \prod _ { ( j , k ) \in \mathcal { E } } p ( u _ { j , k } \vert x _ { j } , x _ { k } ) } \end{array}$ . Applying Bayes’ rule, we have $p ( \mathbf { x } | \mathbf { u } ) \propto p ( \mathbf { x } ) p ( \mathbf { u } | \mathbf { x } )$ , which yields:

$$
p ( { \bf x } | { \bf u } ) \propto \exp \left( - \frac { 1 } { 2 } \sum _ { \ell = 1 } ^ { L } J _ { \ell \ell } x _ { \ell } ^ { 2 } - \sum _ { \{ j , k \} \in \mathcal { E } } J _ { j k } x _ { j } x _ { k } - \sum _ { \{ j , k \} \in \mathcal { E } } \frac { \left( u _ { j , k } - \omega _ { j , k } x _ { j } - \bar { \omega } _ { j , k } x _ { k } \right) ^ { 2 } } { 2 \sigma _ { N _ { j , k } } ^ { 2 } } \right)\tag{187}
$$

Expanding the quadratic terms and collecting coefficients gives

$$
p ( \mathbf { x } | \mathbf { u } ) \propto \exp \left( - \frac { 1 } { 2 } \sum _ { \ell = 1 } ^ { L } \lambda _ { \ell } x _ { \ell } ^ { 2 } + \sum _ { \ell = 1 } ^ { L } \gamma _ { \ell } ( \mathbf { u } ) x _ { \ell } - \sum _ { ( j , k ) \in \mathcal { E } } \left( J _ { j k } + \frac { \omega _ { j , k } \bar { \omega } _ { j , k } } { \sigma _ { N _ { j , k } } ^ { 2 } } \right) x _ { j } x _ { k } \right) ,\tag{188}
$$

where

$$
\lambda _ { \ell } \triangleq J _ { \ell \ell } + \sum _ { ( \ell , k ) \in \mathcal { E } } \frac { \omega _ { \ell , k } ^ { 2 } } { \sigma _ { N _ { \ell , k } } ^ { 2 } } + \sum _ { ( j , \ell ) \in \mathcal { E } } \frac { \bar { \omega } _ { j , \ell } ^ { 2 } } { \sigma _ { N _ { j , \ell } } ^ { 2 } } ,\tag{189}
$$

$$
\gamma _ { \ell } ( \boldsymbol { \mathbf { u } } ) \triangleq \sum _ { ( \ell , k ) \in \mathcal { E } } \frac { \omega _ { \ell , k } } { \sigma _ { N _ { \ell , k } } ^ { 2 } } u _ { \ell , k } + \sum _ { ( j , \ell ) \in \mathcal { E } } \frac { \bar { \omega } _ { j , \ell } } { \sigma _ { N _ { j , \ell } } ^ { 2 } } u _ { j , \ell } .\tag{190}
$$

The Markov constraints in eqs. (180) and (181) imply that the source variables are conditionally independent given the adjacent auxiliary variables, i.e. $\begin{array} { r } { p ( \mathbf { x } \mid \mathbf { u } ) = \prod _ { \ell = 1 } ^ { L } p ( x _ { \ell } | \mathbf { u } ) } \end{array}$ . Therefore, all pairwise interaction terms $x _ { j } x _ { k }$ in equation 188 must vanish, i.e.,

$$
J _ { j k } + \frac { \omega _ { j , k } \bar { \omega } _ { j , k } } { \sigma _ { N _ { j , k } } ^ { 2 } } = 0 ,\tag{191}
$$

which yields $\begin{array} { r } { \sigma _ { N _ { j , k } } ^ { 2 } = - \frac { \omega _ { j , k } \bar { \omega } _ { j , k } } { J _ { i k } } } \end{array}$ . As $\sigma _ { N _ { j , \jmath } } ^ { 2 }$ denotes the noise variance, to ensure $\sigma _ { N _ { j , k } } ^ { 2 } > 0$ , we need to set $\omega _ { j , k } \in ( 0 , 1 )$ if $J _ { j k } < 0$ , and set $\omega _ { j , k } > 1$ otherwise. With appropriate coefficients $\omega _ { j , k } .$ , the constructed auxiliary variables apparently satisfy Markov constraints in eqs. (179) to (181). This completes the construction of the auxiliary random variables.

## Nontrivial MSE distortion analysis:

We now establish the distortion bounds for the auxiliary random variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } }$ constructed above. Without loss of generality, we consider the source node $\ell = 1$ . Suppose X is a pairwise correlated Gaussian vector, the set of neighbors for $X _ { 1 }$ is apparently $\mathcal { N } ( 1 ) = \bar { \{ 2 , \dots , L \} }$

First, for the upper bound in equation 182, i.e.,

$$
\epsilon _ { 1 } = \mathbb { E } [ \| X _ { 1 } - \mathbb { E } [ X _ { 1 } \vert U _ { 1 , N ( 1 ) } ] \| ^ { 2 } ] < \bar { D } _ { 1 } ,\tag{192}
$$

this follows directly from the Markov chain conditions in eqs. (179) to (181), which guarantee that the auxiliary random variables $U _ { 1 , N ( 1 ) }$ capture all the dependence of $X _ { \bar { . } }$ <sub>1</sub> on the remaining source variables. Next, we prove the strict positivity in equation 182. Since the MSE is always nonnegative, it suffices to show that $\epsilon _ { 1 } \neq 0$ . Equivalently, it suffices to show that $X _ { 1 }$ cannot be recovered exactly from $U _ { 1 , N ( 1 ) }$ . Suppose, for contradiction, that

$$
X _ { 1 } = \mathbb { E } [ X _ { 1 } \mid U _ { 1 , N ( 1 ) } ] \quad { \mathrm { a . s . } }\tag{193}
$$

Since $( X _ { 1 } , U _ { 1 , { \mathcal { N } } ( 1 ) } )$ are jointly Gaussian and zero mean, the conditional expectation is thereby a linear transformation. Therefore, equation 193 is equivalent to that there exist constants $a _ { 2 } , \ldots , a _ { L }$ such that

$$
X _ { 1 } = \sum _ { k = 2 } ^ { L } a _ { k } U _ { 1 , k } = \sum _ { k = 2 } ^ { L } a _ { k } \left( \omega _ { 1 , k } X _ { 1 } + \bar { \omega } _ { 1 , k } X _ { k } + N _ { 1 , k } \right) \quad \mathrm { a . s . }\tag{194}
$$

Rearranging terms gives

$$
\left( 1 - \sum _ { k = 2 } ^ { L } a _ { k } \omega _ { 1 , k } \right) X _ { 1 } - \sum _ { k = 2 } ^ { L } a _ { k } { \bar { \omega } } _ { 1 , k } X _ { k } = \sum _ { k = 2 } ^ { L } a _ { k } N _ { 1 , k } \quad { \mathrm { a . s . } }\tag{195}
$$

The left-hand side of equation 195 is a linear function of $\mathbf { X }$ , whereas the right-hand side is a linear function of the independent Gaussian noises $\{ N _ { 1 , k } \} _ { k = 2 } ^ { L }$ . Hence the two sides are two independent Gaussian random variables. As they are equal almost surely, both sides must be almost surely constant. Because all variables have zero mean, this constant must be zero. Thus, we need

$$
\left( 1 - \sum _ { k = 2 } ^ { L } a _ { k } \omega _ { 1 , k } \right) X _ { 1 } - \sum _ { k = 2 } ^ { L } a _ { k } { \bar { \omega } } _ { 1 , k } X _ { k } = 0 \quad \mathrm { a . s . } ,\tag{196}
$$

$$
\sum _ { k = 2 } ^ { L } a _ { k } N _ { 1 , k } = 0 \quad \mathrm { a . s . } ,\tag{197}
$$

where equation 196 implies that:

$$
\begin{array} { r } { [ 1 - \sum _ { k = 2 } ^ { L } a _ { k } \omega _ { 1 , k } ] } \\ { - a _ { 2 } \bar { \omega } _ { 1 , 2 } } \\ { \vdots } \\ { - a _ { L } \bar { \omega } _ { 1 , L } } \end{array} ] \in \ker ( \Sigma ) ,\tag{198}
$$

Since $\Sigma \succ 0$ , we have ke $\mathbf { \partial } \cdot ( \Sigma ) = \{ \mathbf { 0 } \}$ . Therefore, equation 196 is equivalent to

$$
1 - \sum _ { k = 2 } ^ { L } a _ { k } \omega _ { 1 , k } = 0 , \qquad a _ { k } \bar { \omega } _ { 1 , k } = 0 \mathrm { f o r } k = 2 , \ldots , L .\tag{199}
$$

Because appropriate auxiliary variable construction requires $\omega _ { 1 , k } ~ \in ~ \mathbb { R } \setminus \{ 0 , 1 \}$ , which implies $\omega _ { 1 , k } , \bar { \omega } _ { 1 , k } \neq 0 .$ . Therefore, $a _ { k } \bar { \omega } _ { 1 , k } = 0$ requires $a _ { k } = 0$ for all $k = 2 , \ldots , L$ . This contradicts $\begin{array} { r } { 1 - \sum _ { k = 2 } ^ { L } a _ { k } \omega _ { 1 , k } = 0 } \end{array}$ . Therefore, equation 196 cannot hold for appropriate constructed auxiliary variables. Now for equation 197, since the noises $\{ N _ { 1 , k } \} _ { k = 2 } ^ { L }$ are mutually independent Gaussian random variables with zero mean, equation 197 holds if and only if

$$
\mathrm { V a r } \left( \sum _ { k = 2 } ^ { L } a _ { k } N _ { 1 , k } \right) = \sum _ { k = 2 } ^ { L } a _ { k } ^ { 2 } \sigma _ { N _ { 1 , k } } ^ { 2 } = 0 .\tag{200}
$$

Therefore, we must have either $a _ { k } = 0 \mathrm { o r } \sigma _ { N _ { 1 . k } } ^ { 2 } = 0 \mathrm { f o r } k = 2 , \ldots , L$ . Recall that for appropriate constructed auxiliary variables, we have $\begin{array} { r } { \sigma _ { N _ { 1 , k } } ^ { 2 } ~ = ~ - \frac { \omega _ { 1 , k } \bar { \omega } _ { 1 , k } } { J _ { 1 k } } } \end{array}$ , where $J _ { 1 k } \neq 0$ . Since $\omega _ { 1 , k } \in \mathbb { R } \ \backslash$ $\{ 0 , 1 \}$ , we have $\omega _ { 1 , k } \bar { \omega } _ { 1 , k } \neq 0$ and thereby $\sigma _ { N _ { 1 , k } } ^ { 2 } > 0$ . Therefore equation 197 implies that $a _ { k } = 0 .$ for $k = 2 , \ldots , L$ . Substituting this into $\begin{array} { r } { X _ { 1 } = \sum _ { k = 2 } ^ { L } a _ { k } U _ { 1 , k } } \end{array}$ yields $X _ { 1 } = 0$ almost surely, which contradicts $\Sigma \succ 0$ Hence equation 197 also cannot hold for appropriate constructed auxiliary variables. Combining the above arguments, we conclude that $\epsilon _ { 1 }$ is strictly positive and finite.

This proof can be easily extended to all remaining source nodes $X _ { \ell } , \ell = 2 , \cdot \cdot \cdot , L$ , with the same collection of auxiliary random variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } }$ . Therefore, let $\epsilon _ { \ell } \triangleq \mathbb { E } [ \| X _ { \ell } - \mathbb { E } [ X _ { \ell } | U _ { \ell , N ( \ell ) } ] \| ^ { 2 } ]$ denotes the induced distortion for each source node $X _ { \ell } .$ , we have $0 < \epsilon _ { \ell } < \bar { D } _ { \ell } .$ , for $\ell = 1 , \cdots , L$ This completes the proof. □

The above Lemma C.3 mainly shows that the existence of appropriate edge-wise auxiliary variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ for Gaussian source, which induce nontrivial MSE distortion $\epsilon _ { \ell }$ (strictly positive and bounded) for each source variable $X _ { \ell } . \mathrm { A }$ specific distortion region $\mathcal { D } _ { \epsilon }$ can be thereby determined by those edge-wise auxiliary variables. The following Theorem C.5 further provide rate-distortion optimality analysis based on the result of Lemma C.3.

Berger-Tung scheme. Before proving Theorem C.5, we briefly review the Berger-Tung achievability scheme (Berger, 2003) through a simple example. Consider three correlated sources $\mathbf { X } =$ $( X _ { 1 } , \overset { \cdot } { X } _ { 2 } , X _ { 3 } )$ and two encoders observing the overlapping source pairs $( X _ { 1 } , X _ { 2 } )$ and $( X _ { 2 } , X _ { 3 } )$ respectively. Choose auxiliary variables according to $P _ { { \bf X } , U _ { 1 , 2 } , U _ { 2 , 3 } } = P _ { \bf X } P _ { U _ { 1 , 2 } | X _ { 1 } , X _ { 2 } } P _ { U _ { 2 , 3 } | X _ { 2 } , X _ { 3 } } ,$ i.e.,

$$
U _ { 1 , 2 } \to ( X _ { 1 } , X _ { 2 } ) \to X _ { 3 } , \quad U _ { 2 , 3 } \to ( X _ { 2 } , X _ { 3 } ) \to X _ { 1 } ,\tag{201}
$$

and let ${ \hat { X } } _ { \ell } = g _ { \ell } ( U _ { 1 , 2 } , U _ { 2 , 3 } )$ satisfy the desired distortion constraint for each $\ell = 1 , 2 , 3$ . Each encoder quantizes its observation into an auxiliary codeword and transmits its bin index. The decoder then jointly recovers both auxiliary codewords and reconstructs X. The Berger-Tung inner bound guarantees that any rate pair satisfying

$$
R _ { 1 , 2 } \geq I ( X _ { 1 } , X _ { 2 } ; U _ { 1 , 2 } | U _ { 2 , 3 } ) ,\tag{202}
$$

$$
R _ { 2 , 3 } \geq I ( X _ { 2 } , X _ { 3 } ; U _ { 2 , 3 } | U _ { 1 , 2 } ) ,\tag{203}
$$

$$
R _ { 1 , 2 } + R _ { 2 , 3 } \geq I ( { \bf X } ; U _ { 1 , 2 } , U _ { 2 , 3 } ) ,\tag{204}
$$

is achievable. Thus, for fixed auxiliary variables, the minimum achievable sum-rate is

$$
R _ { \mathrm { s u m } } = I ( \mathbf { X } ; U _ { 1 , 2 } , U _ { 2 , 3 } ) .\tag{205}
$$

The same construction extends to an arbitrary collection of overlapping source subsets. In the proof below, the constructed pairwise and node-wise auxiliary variables play the roles of the Berger-Tung auxiliaries, yielding the desired achievable sum-rate upper bound.

![](images/e0108490cef7f1526686ceab26c2ef6a30a27a70c2318955d72884918b58297b.jpg)  
Figure 11: Augmented graphical model with node-wise and edge-wise auxiliary variables.

Definition C.4. Define a non-empty distortion region under the Gaussian case as $\begin{array} { r l } { \mathcal { D } _ { \epsilon } ^ { \mathrm { G } } } & { { } \triangleq } \end{array}$ $\{ ( D _ { 1 } , \cdots , D _ { L } ) : 0 < D _ { \ell } \le \epsilon _ { \ell } , \ell = \bar { 1 } , \bar { \cdot } \cdot \cdot , L \}$ for some strictly positive $0 < \epsilon _ { j } < \infty$

Theorem C.5 (Gaussian optimality). Consider a K-order Gauss-Markov source $\mathbf { X } \ \sim { \mathcal { N } } ( 0 , \Sigma )$ with $\Sigma \succ 0 .$ . There exists a non-empty distortion region $\mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ , such that for all $\mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ , thefollowing sets ofrandom variables can be constructed:

i) (Gaussian auxiliary variables) It suffices to consider jointly Gaussian auxiliary random variables $\begin{array} { r } { \{ \tilde { X } _ { j , k } , U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } } , \{ U _ { \ell } \} _ { \ell = 1 } ^ { L } , } \end{array}$ , and $\hat { \bf X } .$ . Moreover, the optimal $U _ { j , k }$ and $U _ { \ell }$ satisfy the Markov constraints:

$$
\begin{array} { r l } & { X _ { j } \to ( X _ { \mathcal { N } ( j ) \backslash \{ k \} } , U _ { j , k } ) \to X _ { k } , \quad ( j , k ) \in \mathcal { E } ^ { ( K ) } , } \\ & { \quad U _ { j , k } \to ( X _ { j } , X _ { k } ) \to \mathbf { X } , \quad ( j , k ) \in \mathcal { E } ^ { ( K ) } } \\ & { U _ { \ell } \to X _ { \ell } \to U _ { \ell , \mathcal { N } ( \ell ) } \to X _ { - \ell } , \quad \ell = 1 , \cdot \cdot , L . } \end{array}\tag{206}
$$

ii) (Linear interpolation) Given $( X _ { j } , X _ { k } )$ and $U _ { j , k }$ , there exists some coefficients $\omega _ { j , k } \in \mathbb { R } \backslash \{ 0 , 1 \}$ such that auxiliary variable $\ddot { X } _ { j , k }$ admit linear deterministicfunctionsfor $( X _ { j } , X _ { k } )$

$$
\begin{array} { r } { \tilde { X } _ { j , k } = \omega _ { j , k } X _ { j } + ( 1 - \omega _ { j , k } ) X _ { k } , } \end{array}\tag{207}
$$

and satisfies the Markov chain $U _ { j , k } \to \tilde { X } _ { j , k } \to ( X _ { j } , X _ { k } ) \to \mathbf { X } .$

iii) (Linear reconstruction) The optimal $\hat { \bf X }$ adopt linear decoding mappings.

Moreover, with those constructed auxiliary variables, the sum-rate distortion of such system achieves the Shannon lower bound in equation 164 and hence the joint coding rate-distortion opti mality.

Proof. Existence of $\mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$

First, as established in Lemma $\mathrm { C } . 3 ,$ for the edge-wise auxiliary random variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } _ { - } ^ { ( K ) } }$ constructed according to eqs. (179) to (181), the induced distortion for each source variable $X _ { \ell } ,$ defined as $\epsilon _ { \ell } \triangleq \mathbb { E } [ \| X _ { \ell } - \mathbb { E } [ X _ { \ell } | U _ { \ell , N ( \ell ) } ] \| ^ { 2 } ]$ , is strictly positive and bounded, i.e. $0 < \epsilon _ { \ell } < \infty$ . This establishes the existence of a nontrivial distortion region:

$$
\mathcal { D } _ { \epsilon } ^ { \mathrm { G } } \triangleq \{ ( D _ { 1 } , \cdot \cdot \cdot , D _ { L } ) : 0 < D _ { \ell } \leq \epsilon _ { \ell } , \ell = 1 , \cdot \cdot \cdot , L \} .\tag{208}
$$

Moreover, for each source variable $X _ { \ell }$ in a K-order Gauss-Markov source, it has the $K \cdot$ banded conditional-dependence structure, i.e. $\begin{array} { l l l } { \mathcal { N } ( \ell ) } & { \subset } & { \{ \ell \ : - \ : K , \cdots , \ell \ : - 1 , \ell \ : + \ : 1 , \cdots , \ell \ : + } \end{array}$ $K \}$ Therefore, by the Markov constraints in equation 206, $\mathbb { E } [ X _ { \ell } | U _ { \ell , N ( \ell ) } ]$ is a function of $\left( U _ { \ell - K , \ell } , \cdot \cdot \cdot , U _ { \ell - 1 , \ell } , U _ { \ell , \ell + 1 } , \cdot \cdot \cdot , U _ { \ell , \ell + K } \right)$

## Gaussian auxiliary variables:

Next, we show that arbitrary distortion tuple within the induced region, i.e. $\mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ , can be achieved by introducing a set of node-wise Gaussian auxiliary random variables $\{ U _ { \ell } \} _ { \ell = 1 } ^ { L }$ satisfying the Markov chains:

$$
U _ { \ell } \to X _ { \ell } \to ( X _ { - \ell } , U _ { \ell , N ( \ell ) } ) ,\tag{209}
$$

and then the JC optimality is also achieved. Suppose the edge-wise auxiliary random variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ have been constructed according to eqs. (179) to (181), which are all Gaussian variables. Then each source node $X _ { \ell }$ can be decomposed as

$$
\begin{array} { r } { X _ { \ell } = \mathop { \mathbb { E } } [ X _ { \ell } | U _ { \ell , N ( \ell ) } ] + N _ { \ell } , \quad \ell = 1 , \cdots , L , } \end{array}\tag{210}
$$

where $N _ { \ell } \sim \mathcal N ( 0 , \sigma _ { N _ { \ell } } ^ { 2 } )$ denotes Gaussian residual and is independent of $\left( U _ { \ell , \mathcal { N } ( \ell ) } , X _ { \mathcal { N } ( \ell ) } \right)$ . To further refine the reconstruction of $X _ { \ell } ,$ we first introduce a Gaussian auxiliary random variable $\tilde { U } _ { \ell }$ such that:

$$
U _ { \ell } = X _ { \ell } + W _ { \ell } , \quad \mathrm { w i t h } ~ W _ { \ell } \sim { \mathcal N } ( 0 , \sigma _ { W _ { \ell } } ^ { 2 } ) , ~ W _ { \ell } \perp \perp X _ { \ell } .\tag{211}
$$

Apparently, we have $U _ { \ell } \to X _ { \ell } \to ( X _ { - \ell } , U _ { \ell , N ( \ell ) } )$ . It can be verified that $X _ { \ell }$ can be further decomposed as $X _ { \ell } = \mathbb { E } [ X _ { \ell } | U _ { \ell , N ( \ell ) } , U _ { \ell } ] + Z _ { \ell } .$ , where $Z _ { \ell } \sim \mathcal { N } ( 0 , \sigma _ { Z _ { \ell } } ^ { 2 } )$ denote an independent Gaussian noise. With $\epsilon _ { \ell } \triangleq \mathbb { E } [ \| X _ { \ell } - \mathbb { E } [ X _ { \ell } | U _ { \ell , N ( \ell ) } ] \| ^ { 2 } ]$ and introduced node-wise $U _ { \ell }$ , then we further have the corresponding distortion term as $\begin{array} { r } { D _ { \ell } = ( \epsilon _ { \ell } + \frac { 1 } { \sigma _ { W _ { \ell } } ^ { 2 } } ) ^ { - 1 } } \end{array}$ . Observe that by setting the variance $\sigma _ { W _ { \ell } } ^ { 2 }$ arbitrarily small, the induced distortion of $X _ { \ell } .$ , i.e.,

$$
D _ { \ell } = \mathbb { E } [ \| X _ { \ell } - \mathbb { E } [ X _ { \ell } | U _ { \ell , \mathcal { N } ( \ell ) } , U _ { \ell } ] \| ^ { 2 } ] ,\tag{212}
$$

can be made arbitrarily close to zero. Therefore, by introducing such auxiliary variables $\{ U _ { \ell } \} _ { \ell = 1 } ^ { L }$ for each component, the achievable distortion can be continuously controlled over the range $( 0 , \epsilon _ { \ell } ]$ . Consequently, any distortion tuple $\mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ is achievable. For example, Figure 11 illustrates the construction of $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( 2 ) } }$ and $\{ U _ { \ell } \} _ { \ell = 1 } ^ { 5 }$ for a second-order Gauss-Markov source $( X _ { 1 } , \cdots , X _ { 5 } )$

For the above designed coding system, by applying Lemma 1 in Chen et al. (2017) together with the Berger-Tung inner/achievability bound, we obtain that, for all distortion tuples $\mathbf { D } \in \breve { \mathcal { D } } _ { \epsilon } ^ { \mathrm { G } }$

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( { \bf D } ) \leq \frac { 1 } { 2 } \log \frac { | \boldsymbol { \Sigma } | } { \prod _ { \ell = 1 } ^ { L } D _ { \ell } } .\tag{213}
$$

Conversely, by the Shannon lower bound, we have for any $\mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } } \subseteq \mathcal { D } _ { \mathrm { J C } }$

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( { \bf D } ) \geq \frac { 1 } { 2 } \log \frac { | \Sigma | } { \prod _ { \ell = 1 } ^ { L } D _ { \ell } } .\tag{214}
$$

Combining the upper and lower bounds, for any $\mathbf { D } \in \mathcal { D } _ { \epsilon }$ , it follows that

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( { \bf D } ) = \frac { 1 } { 2 } \log \frac { | \boldsymbol { \Sigma } | } { \prod _ { \ell = 1 } ^ { L } D _ { \ell } } .\tag{215}
$$

This coincides with the joint coding rate-distortion function in equation 164, and hence establishes joint coding optimality. This completes the proof.

## Linear interpolation.

For jointly Gaussian X and $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ , the conditional expectation $\mathbb { E } [ U _ { j , k } | X _ { j } , X _ { k } ]$ is linear on $( X _ { j } , X _ { k } )$ . Hence, $U _ { j , k }$ admits the orthogonal decomposition

$$
U _ { j , k } = \gamma _ { j , k } X _ { j } + \theta _ { j , k } X _ { k } + \tilde { N } _ { j , k } ,\tag{216}
$$

where $\tilde { N } _ { j , k }$ is also Gaussian and independent of $( X _ { j } , X _ { k } )$ . Since scaling of $U _ { j , k }$ does not affect the rate-distortion tradeoff, we can normalize the coefficients by $\gamma _ { j , k } + \theta _ { j , k }$ to obtain

$$
U _ { j , k } = \omega _ { j , k } X _ { j } + ( 1 - \omega _ { j , k } ) X _ { k } + N _ { j , k } ,\tag{217}
$$

where $\begin{array} { r } { \omega _ { j , k } = \frac { \gamma _ { j , k } } { \gamma _ { j , k } + \theta _ { j , k } } } \end{array}$ and $N _ { j , k }$ is Gaussian and independent of $( X _ { j } , X _ { k } )$ . Furthermore, as shown in Lemma $\mathrm { C } . 3$ , the noise variance satisfies

$$
\sigma _ { N _ { j , k } } ^ { 2 } = - \frac { \omega _ { j , k } ( 1 - \omega _ { j , k } ) } { J _ { j k } } .
$$

In particular, $\omega _ { j , k } \in ( 0 , 1 )$ when $J _ { j k } < 0$ , while $\omega _ { j , k } > 1$ when $J _ { j k } > 0$

## Linear reconstruction:

Given constructed edge-wise auxiliary variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ and node-wise auxiliary variables $\{ U _ { \ell } \} _ { \ell = 1 } ^ { L }$ , the final step is to reconstruct each source node $X _ { \ell }$ . Due to the Markov constraints in equation 206, for each source variable $X _ { \ell } ,$ we have the following reconstruction:

$$
\hat { X } _ { \ell } = \mathbb { E } \big [ X _ { \ell } \big | \{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } } , \{ U _ { \ell } \} _ { \ell = 1 } ^ { L } \big ] = \mathbb { E } \big [ X _ { \ell } \big | U _ { \ell , \mathcal { N } ( \ell ) } , U _ { \ell } \big ] .
$$

Without loss of generality, we let $U _ { \ell , N ( \ell ) } = \left( U _ { \ell - K , \ell } , \cdot \cdot \cdot , U _ { \ell - 1 , \ell } , U _ { \ell , \ell + 1 } , \cdot \cdot \cdot , U _ { \ell , \ell + K } \right)$ . Then for jointly Gaussian auxiliary variables, we have the optimal reconstruction $\hat { X } _ { \ell }$ with linear decoding function:

$$
\begin{array} { l } { \displaystyle \tilde { X } _ { \ell } = \mathbf { a } ^ { ( \ell ) ^ { \top } } \left[ U _ { \ell - K , \ell } , \cdots , U _ { \ell - 1 , \ell } , \cdots , U _ { \ell , \ell + 1 } , \cdots , U _ { \ell , \ell + K } , U _ { \ell } \right] ^ { \top } } \\ { \displaystyle = \sum _ { k = 0 } ^ { K - 1 } a _ { K - k } ^ { ( \ell ) } , } \\ { \displaystyle = \sum _ { k = 0 } ^ { K - 1 } a _ { K - k } ^ { ( \ell ) } , } \\ { \displaystyle = \sum _ { k = 0 } ^ { K - 1 } a _ { K - k } ^ { ( \ell ) } , } \\ { \displaystyle = \sum _ { k = 0 } ^ { K - 1 } a _ { K - k } ^ { ( \ell ) } , \ \{ \omega _ { \ell - 1 - k , \ell } X _ { \ell - 1 - k } + \bar { \omega } _ { \ell - 1 - k , \ell } X _ { \ell } + N _ { \ell - 1 - k , \ell } \} } \\ { \displaystyle \quad + \sum _ { k = 0 } ^ { K - 1 } a _ { K } ^ { ( \ell ) } , \ \left( \omega _ { \ell , \ell - K + 1 + k } X _ { \ell } + \bar { \omega } _ { \ell , \ell - K + 1 + k } X _ { \ell - K + 1 + k } + N _ { \ell , \ell - K + 1 + k } \right) + a _ { 2 k } ^ { ( \ell ) } \lambda _ { \ell } \cdot X _ { \ell } } \\ { \displaystyle = \sum _ { k = - k } ^ { K - 1 } \beta _ { K + 1 + k } X _ { \ell + k } + X _ { \ell } , } \end{array}
$$

where $\begin{array} { r } { Z _ { \ell } = \sum _ { k = 0 } ^ { K - 1 } a _ { K - k } ^ { ( \ell ) } N _ { \ell - 1 - k , \ell } + \sum _ { k = K } ^ { 2 K - 1 } a _ { k } ^ { ( \ell ) } N _ { \ell , \ell - K + 1 + k } } \end{array}$ is a Gaussian random variable independent with X. This completes the proof. □

Theorem C.6 (JC rate-distortion optimality). Under the source model setting of Theorem C.5, there exists a non-empty distortion region $\mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ , such that for all $\mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } } , R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( \mathbf { D } )$ achieves the jointcoding rate-distortionfunction:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( \mathbf { D } ) = R ^ { \mathrm { J C G } } ( \mathbf { D } ) , \quad \forall \mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } } .\tag{218}
$$

Proof. Theorem C.5 already establishes the existence of two sets of edge-wise auxiliary random variables $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ and node-wise auxiliary variables $\{ U _ { j } \} _ { j = 1 } ^ { L }$ , which satisfy the desired Markov constraints in equation 206 and achieve Shannon lower bound in the distortion region $\mathcal { D } _ { \epsilon }$ In this proof, we show that these auxiliary variables are operationally compatible with the structure of the proposed ISC-K system.

## Step 1. Achievability scheme for ISC:

In this part, we show those edge-wise and node-wise auxiliary variables can be incorporated into the proposed ISC-K system without violating its coding structure:

1. Regrouped auxiliary variables. As in the Definition C.1, there are only edge-wise auxiliary variable form $\bar { U } _ { j , k }$ involved in the ISC-K system. We first clarify how edge-wise auxiliary variables can be extended when node-wise auxiliary variables are present. Basically, the node-wise variables do not require a separate class of messages. They can be incorporated into the pair-indexed ISC auxiliaries through a deterministic augmentation. Given edge-wise and node-wise auxiliary variables as constructed in Theorem C.5, we consider a set of augmented auxiliary variables:

$$
\begin{array} { r } { V _ { j , k } \triangleq \left\{ \begin{array} { l l } { U _ { j , k } , } & { j < L - 1 , k < \operatorname* { m i n } \{ j + K , L \} , } \\ { ( U _ { j , k } , U _ { j } ) , } & { j < L - 1 , k = \operatorname* { m i n } \{ j + K , L \} , } \\ { ( U _ { L - 1 , L } , U _ { L - 1 } , U _ { L } ) , } & { j = L - 1 , k = L . } \end{array} \right. } \end{array}\tag{219}
$$

The idea is to simply augment the node-wise variable $U _ { j }$ into the last edge-wise auxiliary variable associated with source node $X _ { j }$ . The exception is the case when $j = L - 1$ where we have to append both $U _ { L - 1 }$ and $U _ { L }$ as there are no future frames after frame $X _ { L }$ . Obviously, $\{ \bar { V _ { j , k } } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ is a regrouping of components of $\{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ and $\{ U _ { \ell } \} _ { \ell = 1 } ^ { L }$ . Therefore it represents the same collection of auxiliary variables.

2. ISC Encoding. As defined in Section 3.2, ISC-K encoding mapping $f _ { j , k }$ can be decomposed as $f _ { j , k } = \phi _ { j , k } \circ h _ { j , k }$ , where $h _ { j , k }$ denotes the deterministic interpolation function and $\phi _ { j , k }$ denotes a sequential encoder. For deterministic functions $h _ { j , k }$ , we adopt the following realizations:

$$
h _ { j , k } ( X _ { j } , X _ { k } ) = \tilde { Y } _ { j , k } \triangleq \left\{ \begin{array} { l l } { \tilde { X } _ { j , k } , } & { j < L - 1 , k < \operatorname* { m i n } \{ j + K , L \} , } \\ { ( \tilde { X } _ { j , k } , X _ { j } ) , } & { j < L - 1 , k = \operatorname* { m i n } \{ j + K , L \} , } \\ { ( \tilde { X } _ { j , k } , X _ { j } , X _ { k } ) , } & { j = L - 1 , k = L . } \end{array} \right.\tag{220}
$$

Here, $\tilde { X } _ { j , k }$ is attained by performing a linear interpolation between $( X _ { j } , X _ { k } )$ as stated in Theorem C.5. Apparently we have the following Markov chains:

$$
\begin{array} { r l r } & { } & { U _ { j , k } \to \tilde { X } _ { j , k } \to ( X _ { j } , X _ { k } ) \to { \bf X } } \\ & { } & { U _ { \ell } \to X _ { \ell } \to { \bf X } _ { - \ell } \quad \} \Rightarrow V _ { j , k } \to \tilde { Y } _ { j , k } \to ( X _ { j } , X _ { k } ) \to { \bf X } . } \end{array}\tag{221}
$$

Since the RHS Markov condition in equation 221 is stronger than the encoder-side Markov condition of ISC-K system in equation 5. Therefore, the constructed auxiliary variables $V _ { j , k }$ and $\tilde { Y } _ { j , k }$ are a valid instantiation of encoder test channels permitted of ISC-K system. Then we perform the ISC-K encoder on source pair $( X _ { j } , X _ { k } )$ to generate $V _ { j , k }$ .

3. ISC Decoding. As in equation 5, given received $V _ { < \ell }$ at ℓ-stage ISC-K decoder, we adopt the deterministic decoding function $\hat { X } _ { \ell } = g _ { \ell } ( V _ { \leq \ell } )$

For augmented auxiliary variables $\{ V _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ and coding scheme constructed above, we have the following key observations:

1. Since each $V _ { j , k }$ contains all information of $U _ { j , k }$ , it also follow the Markov constraints analogous to $U _ { j , k }$ stated in eqs. (180) and (181).

2. Based on the definition of ISC-K encoder, for each $( j , k )$ -encoding stage and auxiliary variable $V _ { j , k } ;$ , the associate rate $\bar { R } _ { j , k }$ R¯ follows from the conditional source coding theorem.

$$
\begin{array} { r } { \bar { R } _ { j , k } = I ( \tilde { Y } _ { j , k } ; V _ { j , k } | V _ { < j , k } ) . } \end{array}\tag{222}
$$

Note that this has the same mutual information form as in Definition C.1, while ISC-K coding is optimized over all admissible auxiliary variables.

Now for those constructed auxiliary variables, we can characterize the associate sum-rate and perletter distortion. Specifically, for $j < L - 1$ and $k < \operatorname* { m i n } \{ j + K , L \}$ , the associate rate $\bar { R } _ { j , k }$ follows by chain rule:

$$
\bar { R } _ { j , k } = I ( \tilde { Y } _ { j , k } ; V _ { j , k } | V _ { < j , k } )\tag{223}
$$

$$
= I ( \tilde { X } _ { j , k } ; V _ { j , k } | V _ { < j , k } )\tag{224}
$$

$$
= I ( X _ { j } , X _ { k } ; V _ { j , k } | V _ { < j , k } ) .\tag{225}
$$

For $j < L - 1$ and k = min $\{ j + K , L \}$ , the associate rate $\bar { R } _ { j , k }$ follows by chain rule:

$$
\bar { R } _ { j , k } = I ( \tilde { Y } _ { j , k } ; V _ { j , k } | V _ { < j , k } )\tag{226}
$$

$$
= I ( \tilde { X } _ { j , k } , X _ { j } ; V _ { j , k } | V _ { < j , k } )\tag{227}
$$

$$
= I ( X _ { j } , X _ { k } ; V _ { j , k } | V _ { < j , k } ) .\tag{228}
$$

For $j = L - 1$ and $k = L$ , the associate rate $R _ { L - 1 , L }$

$$
\bar { R } _ { L - 1 , L } = I ( \tilde { Y } _ { L - 1 , L } ; V _ { L - 1 , L } | V _ { < L - 1 , L } )\tag{229}
$$

$$
= I ( \tilde { X } _ { L - 1 , L } , X _ { L - 1 } , X _ { L } ; V _ { L - 1 , L } | V _ { < L - 1 , L } )\tag{230}
$$

$$
= I ( X _ { L - 1 } , X _ { L } ; V _ { L - 1 , L } | V _ { < L - 1 , L } ) .\tag{231}
$$

Now we can directly apply the same steps in eqs. (176) to (178) in the proof of Theorem C.2 to augmented auxiliary variables $\big \{ V _ { j , k } \big \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ , i.e.,

$$
\sum \mathrm { ~  ~ { ~ \cal ~ R } ~ } _ { j , k } = \sum \mathrm { ~  ~ { ~ \cal ~ I } ~ } ( X _ { j } , X _ { k } ; V _ { j , k } | V _ { < j , k } )\tag{232}
$$

$$
( j , k ) { \in } { \mathcal { E } } ^ { ( K ) } \qquad \quad ( j , k ) { \in } { \mathcal { E } } ^ { ( K ) }
$$

$$
= \sum _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } } I ( \mathbf { X } ; V _ { j , k } | V _ { < j , k } )\tag{233}
$$

$$
= I ( \mathbf { X } ; \{ V _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } } )\tag{234}
$$

$$
= I ( \mathbf { X } ; \{ U _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } } , U _ { 1 } , \cdot \cdot \cdot , U _ { L } )\tag{235}
$$

Now for decoding distortion of each source node $X _ { \ell } ,$ , since X and $\{ V _ { j , k } \} _ { ( j , k ) \in \mathcal { E } ^ { ( K ) } }$ are jointly Gaussian, we can adopt the decoding introduced in the proof of Theorem C.5, i.e. ${ \mathbb E } [ X _ { \ell } | U _ { \ell , N ( \ell ) } , U _ { \ell } ]$ which is valid for the ISC architecture based on the definition of ℓ-th decoding function, i.e. ${ \hat { X } } _ { \ell } = g _ { \ell } ( V _ { \leq \ell } )$ . Therefore we have the induced distortion $\mathbf { D } = ( D _ { 1 } , \cdot \cdot \cdot , D _ { L } ) \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ and the corresponding rate-distortion function in equation 215. Notice that given the induced distortion D, this is only one possible realization of ISC-K coding system. Therefore we have the following upper bound:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( \mathbf { D } ) \leq \bar { R } _ { \mathrm { s u m } } ( \mathbf { D } ) = \frac { 1 } { 2 } \log \frac { | \Sigma | } { \prod _ { \ell = 1 } ^ { L } D _ { \ell } } .\tag{236}
$$

## Step 2. Converse part

For the Gauss-Markov source X and arbitrary given distortion tuple $\mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } } \subseteq \mathcal { D } ^ { \mathrm { J C } }$ , we can again use the Shannon lower bound:

$$
R _ { K , \mathrm { s u m } } ^ { \mathrm { I S C } } ( \mathbf { D } ) \geq R ^ { \mathrm { J C G } } ( \mathbf { D } ) = \frac { 1 } { 2 } \log \frac { | \Sigma | } { \prod _ { \ell = 1 } ^ { L } D _ { \ell } } .\tag{237}
$$

Then apparently, for K-order Gauss-Markov source and arbitrary distortion $\mathbf { D } \in \mathcal { D } _ { \epsilon } ^ { \mathrm { G } }$ , we can adopt the ISC-K coding scheme alongside those auxiliary random variables constructed in Theorem C.5 and achieve the Shannon lower bound, thereby the joint-coding rate-distortion function. This completes the proof. □

## C.5 GAUSSIAN EXAMPLE

In this section, we provide the explicit numerical calculation for the two-frame Gaussian source $( X _ { 1 } , X _ { 2 } ) \sim { \mathcal { N } } ( 0 , { \dot { \Sigma } } )$ . We assume the symmetric setting with unit variances $\sigma _ { 1 } ^ { 2 } = \sigma _ { 2 } ^ { 2 } = 1$ and correlation coefficient $\rho > 0$ , such that:

$$
\Sigma = \left[ \begin{array} { l l } { 1 } & { \rho } \\ { \rho } & { 1 } \end{array} \right] .\tag{238}
$$

## C.5.1 SYMMETRIC DISTORTION CASE

We first consider the symmetric distortion constraint:

$$
\mathbb { E } [ \| X _ { j } - \hat { X } _ { j } \| ^ { 2 } ] \le D , \quad j = 1 , 2\tag{239}
$$

Joint coding. Recall the Gaussian joint coding rate-distortion function in equation 163. Let $\Sigma _ { e }$ denote the reconstruction error covariance matrix. Under the symmetric distortion constraint, we parametrize:

$$
\Sigma _ { e } = \left[ \begin{array} { l l } { d } & { \theta } \\ { \theta } & { d } \end{array} \right] .\tag{240}
$$

where d $\triangleq ( \Sigma _ { e } ) _ { 1 1 } = ( \Sigma _ { e } ) _ { 2 2 }$ . Then the joint coding rate-distortion optimization problem becomes:

$$
\begin{array} { r l } { R _ { \mathrm { J C } } ^ { G } ( \mathbf { D } ) = \displaystyle \operatorname* { m i n } _ { \Sigma _ { e } } \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { | \Sigma _ { e } | } } & { } \\ { \mathrm { s . t . ~ } 0 \prec \Sigma _ { e } \preceq \Sigma , } & { } \\ { ( \Sigma _ { e } ) _ { j j } \leq D , } & { \quad j = 1 , 2 . } \end{array}\tag{241}
$$

Under the above parameterizations, this further reduces to

$$
\begin{array} { l } { { \displaystyle R _ { \mathrm { J C } } ^ { G } ( D ) = \operatorname* { m i n } _ { \theta } \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { D ^ { 2 } - \theta ^ { 2 } } } } \\ { { \mathrm { s . t . ~ } - D < \theta < D , } } \\ { { \displaystyle \rho + D - 1 \leq \theta \leq \rho + 1 - D . } } \end{array}\tag{242}
$$

Solving the above optimization yields:

$$
R _ { \mathrm { J C } } ^ { G } ( D ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { D ^ { 2 } } , } & { 0 < D \le 1 - \rho , } \\ { \frac { 1 } { 2 } \log \frac { 1 + \rho } { 2 D + \rho - 1 } , } & { 1 - \rho < D \le 1 , } \\ { 0 , } & { D \ge 1 . } \end{array} \right.\tag{243}
$$

The regime $0 < D \le 1 - \rho$ corresponds to the high-resolution region where the Shannon lower bound is tight.

Sequential coding. We next consider the strictly causal sequential coding setting. According to Corollary 1.2 in Ma et al. (2007), the sequential coding sum-rate function for the two-frame Gaussian source $( { \bar { X } } _ { 1 } , X _ { 2 } )$ under symmetric distortion constraint is given by

$$
R _ { \mathrm { s e q , s u m } } ( D ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \log \frac { 1 } { D } + \frac { 1 } { 2 } \log \frac { \rho ^ { 2 } D + ( 1 - \rho ^ { 2 } ) } { D } , } & { 0 < D \leq 1 , } \\ { 0 , } & { D \geq 1 . } \end{array} \right.\tag{244}
$$

The above expression can be interpreted as the sum of two terms corresponding to the sequential coding process:

$$
R _ { 1 } ( D ) = \frac { 1 } { 2 } \log \frac { 1 } { D } , \quad R _ { 2 } ( D ) = \frac { 1 } { 2 } \log \frac { \rho ^ { 2 } D + ( 1 - \rho ^ { 2 } ) } { D } .\tag{245}
$$

The first term corresponds to encoding the first source $X _ { 1 }$ without future information. The second term represents the additional rate required for encoding $X _ { 2 }$ conditioned on the reconstruction of $X _ { 1 }$ . Compared with the joint-coding rate-distortion function in equation 243, the sequential coding system incurs a nonzero performance gap due to the causal constraint, which prevents the encoder from jointly exploiting the correlation between $( X _ { 1 } , X _ { 2 } )$

ISC. We now consider the proposed ISC-1 framework. For the symmetric distortion setting, we fix the interpolation coefficient to $\begin{array} { r } { \omega _ { 1 , 2 } = \frac { 1 } { 2 } } \end{array}$ , and construct the interpolation-based auxiliary variable:

$$
\tilde { X } _ { 1 , 2 } = \frac { 1 } { 2 } X _ { 1 } + \frac { 1 } { 2 } X _ { 2 } , \quad U _ { 1 , 2 } = \tilde { X } _ { 1 , 2 } + N _ { 1 , 2 } ,\tag{246}
$$

where $N _ { 1 , 2 }$ is Gaussian noise independent of $( X _ { 1 } , X _ { 2 } )$ . The interpolated source has $\begin{array} { r } { \sigma _ { \tilde { X } _ { 1 . 2 } } ^ { 2 } = \frac { 1 + \rho } { 2 } } \end{array}$ In the high resolution regime corresponding to the Markov chain $X _ { 1 }  U _ { 1 , 2 }  X _ { 2 }$ , the noise variance is given by $\begin{array} { r } { \sigma _ { N _ { 1 , 2 } } ^ { 2 } = - \frac { 1 } { 4 J _ { 1 2 } } = \frac { 1 - \rho ^ { 2 } } { 4 \rho } } \end{array}$ . This induces the distortion region:

$$
\mathcal { D } _ { \epsilon } = \left\{ ( D _ { 1 } , D _ { 2 } ) : D _ { 1 } , D _ { 2 } \leq 1 - \rho \right\}\tag{247}
$$

where the JC optimality is achieved through the introduction of additional auxiliary random variables $U _ { 1 }$ and $U _ { 2 }$

Next, consider the low-resolution regime $D \in ( 1 - \rho , 1 ]$ . Under the symmetric distortion constraint, we have

$$
\begin{array} { r l } & { D = D _ { 1 } = \mathbb { E } [ \| X _ { 1 } - \hat { X } _ { 1 } \| ^ { 2 } ] } \\ & { \quad = \sigma _ { 1 } ^ { 2 } - \frac { \mathbb { E } [ X _ { 1 } U _ { 1 , 2 } ] ^ { 2 } } { \sigma _ { U _ { 1 , 2 } } ^ { 2 } } } \\ & { \quad = 1 - \frac { \frac { 1 } { 4 } ( 1 + \rho ) ^ { 2 } } { \frac { 1 + \rho } { 2 } + \sigma _ { N _ { 1 , 2 } } ^ { 2 } } . } \end{array}\tag{248}
$$

Solving for the noise variance yields:

$$
\sigma _ { N _ { 1 , 2 } } ^ { 2 } = \frac { ( 1 + \rho ) ( 2 D + \rho - 1 ) } { 4 ( 1 - D ) } .\tag{249}
$$

The corresponding rate is

$$
\begin{array} { l } { \displaystyle R = I ( \tilde { X } _ { 1 , 2 } ; U _ { 1 , 2 } ) } \\ { \displaystyle \quad = \frac { 1 } { 2 } \log \left( \frac { \sigma _ { \tilde { X } _ { 1 , 2 } } ^ { 2 } } { \sigma _ { N _ { 1 , 2 } } ^ { 2 } } + 1 \right) } \\ { \displaystyle \quad = \frac { 1 } { 2 } \log \frac { 1 + \rho } { 2 D + \rho - 1 } } \end{array}\tag{250}
$$

This coincides exactly with the joint-coding rate-distortion function in equation 243 for the lowresolution regime. Combined with the high-resolution result above, ISC-1 therefore achieves the Gaussian joint-coding rate-distortion performance for the two-frame Gaussian source.

## C.5.2 ASYMMETRIC DISTORTION

We next consider the asymmetric distortion constraints:

$$
\begin{array} { r } { \mathbb { E } [ \| X _ { 1 } - \hat { X } _ { 1 } \| ^ { 2 } ] \leq D _ { 1 } , \quad \mathbb { E } [ \| X _ { 2 } - \hat { X } _ { 2 } \| ^ { 2 } ] \leq D _ { 2 } , } \end{array}\tag{251}
$$

where $D _ { 1 } \in ( 0 , 1 ]$ is fixed. For convenience to compare within the high resolution regime, we further assume $D _ { 1 } \stackrel { \cdot } { \in } ( 0 , 1 - \rho ^ { 2 } ]$

Joint coding. Under the asymmetric distortion setting, the reconstruction error covariance matrix takes the form

$$
\Sigma _ { e } = { \left[ \begin{array} { l l } { D _ { 1 } } & { \theta } \\ { \theta } & { d _ { 2 } } \end{array} \right] }\tag{252}
$$

where $d _ { 2 } \leq D _ { 2 }$ . Then joint-coding rate-distortion optimization becomes:

$$
\begin{array} { r l r } {  { R _ { \mathrm { J C } } ^ { G } ( \mathbf { D } ) = \operatorname* { m i n } _ { \theta , d _ { 2 } } \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { D _ { 1 } d _ { 2 } - \theta ^ { 2 } } } } \\ & { } & { \mathrm { s . t . } \ D _ { 1 } d _ { 2 } - \theta ^ { 2 } > 0 , } \\ & { } & { ( 1 - D _ { 1 } ) ( 1 - d _ { 2 } ) - ( \rho - \theta ) ^ { 2 } \geq 0 , } \\ & { } & { d _ { 2 } \leq D _ { 2 } . } \end{array}\tag{253}
$$

Solving the above optimization yields the following rate-distortion function:

$$
R _ { \mathrm { J C } } ^ { G } ( \mathbf { D } ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { D _ { 1 } D _ { 2 } } , } & { 0 < D _ { 2 } \leq \frac { 1 - \rho ^ { 2 } - D _ { 1 } } { 1 - D _ { 1 } } , } \\ { \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { D _ { 1 } D _ { 2 } - \left( \rho - \sqrt { ( 1 - D _ { 1 } ) ( 1 - D _ { 2 } ) } \right) ^ { 2 } } , } & { \frac { 1 - \rho ^ { 2 } - D _ { 1 } } { 1 - D _ { 1 } } < D _ { 2 } < 1 - ( 1 - D _ { 1 } ) \rho ^ { 2 } , } \\ { \frac { 1 } { 2 } \log \frac { 1 } { D _ { 1 } } , } & { D _ { 2 } \geq 1 - ( 1 - D _ { 1 } ) \rho ^ { 2 } . } \end{array} \right.\tag{254}
$$

The three regions correspond to different operating regimes of the Gaussian joint-coding solution. In particular, when $D _ { 2 }$ is sufficiently small, the Shannon lower bound is tight and the optimal ratedistortion function reduces to the product form in the first case. As $D _ { 2 }$ increases, the optimal reconstruction error covariance becomes correlated, leading to the middle regime. Finally, when $D _ { 2 }$ is sufficiently large, only the distortion constraint on $X _ { 1 }$ remains active, and the sum-rate reduces to the scalar Gaussian rate-distortion function of $X _ { 1 }$ alone.

Sequential coding. For the strictly causal sequential coding setting, the sum-rate function under asymmetric distortion constraints is given by:

$$
R _ { \mathrm { s e q , s u m } } ( \mathbf { D } ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \log \frac { 1 } { D _ { 1 } } + \frac { 1 } { 2 } \log \frac { \rho ^ { 2 } D _ { 1 } + ( 1 - \rho ^ { 2 } ) } { D _ { 2 } } , } & { D _ { 2 } < \rho ^ { 2 } D _ { 1 } + ( 1 - \rho ^ { 2 } ) , } \\ { \frac { 1 } { 2 } \log \frac { 1 } { D _ { 1 } } , } & { \rho ^ { 2 } D _ { 1 } + ( 1 - \rho ^ { 2 } ) \leq D _ { 2 } \leq 1 , } \\ { 0 , } & { D _ { 1 } \geq 1 , \ D _ { 2 } \geq 1 . } \end{array} \right.\tag{255}
$$

The above characterization still reflects the sequential nature of the coding process. When $D _ { 2 } \geq \underline { { \geq } }$ $\rho ^ { 2 } D _ { 1 } + ( 1 - \rho ^ { 2 } )$ , the distortion requirement on $X _ { 2 }$ can already be satisfied using only the reconstruction of $X _ { 1 }$ , and therefore no additional rate is required for the second stage.

ISC. We now analyze the proposed ISC-1 under asymmetric distortion constraints. Unlike the symmetric setting, the interpolation coefficient $\omega _ { 1 , 2 }$ is optimized jointly with the distortion constraint. From the proof of Lemma C.3, for $\rho > 0 ,$ the optimal interpolation coefficient satisfies $\omega _ { 1 , 2 } > 0$ . We first characterize the high resolution regime where JC optimality is achieved through additional auxiliary random variables $U _ { 1 }$ and $U _ { 2 }$ . Setting $\epsilon _ { 1 } \triangleq D _ { 1 }$ , the corresponding distortion threshold for the second source is

$$
\epsilon _ { 2 } = \frac { 1 - \rho ^ { 2 } - \epsilon _ { 1 } } { 1 - \epsilon _ { 1 } } .\tag{256}
$$

Therefore, for arbitrary $D _ { 1 } \in ( 0 , 1 - \rho ^ { 2 } ]$ and $\begin{array} { r } { D _ { 2 } \ \leq \ \frac { 1 - \rho ^ { 2 } - D _ { 1 } } { 1 - D _ { 1 } } } \end{array}$ , ISC achieves JC rate-distortion function. Notably, the region coincides with the high-resolution regimes of the Gaussian jointcoding solution in equation 254. Outside this region, the sum-rate optimization reduces to:

$$
\begin{array} { r l } & { R _ { 1 , \mathrm { s u m } } ( { \bf D } ) = \displaystyle \operatorname* { m i n } _ { \omega _ { 1 , 2 } } \frac { 1 } { 2 } \log \frac { \sigma _ { U _ { 1 , 2 } } ^ { 2 } } { \sigma _ { N _ { 1 , 2 } } ^ { 2 } } } \\ & { ~ \quad \quad \quad \mathrm { s . t . } \mathbb { E } [ \| X _ { 1 } - \hat { X } _ { 1 } \| ^ { 2 } ] = D _ { 1 } , } \\ & { \quad \quad \quad \quad \mathbb { E } [ \| X _ { 2 } - \hat { X } _ { 2 } \| ^ { 2 } ] \le D _ { 2 } . } \end{array}\tag{257}
$$

The variance term of auxiliary variables are:

$$
\begin{array} { r l } & { \sigma _ { U _ { 1 , 2 } } ^ { 2 } = \sigma _ { \tilde { X } _ { 1 , 2 } } ^ { 2 } + \sigma _ { N _ { 1 , 2 } } ^ { 2 } } \\ & { \sigma _ { \tilde { X } _ { 1 , 2 } } ^ { 2 } = \omega _ { 1 , 2 } ^ { 2 } + 2 \omega _ { 1 , 2 } ( 1 - \omega _ { 1 , 2 } ) \rho + ( 1 - \omega _ { 1 , 2 } ) ^ { 2 } . } \end{array}\tag{258}
$$

(259)

We first consider the equality distortion constraint on the first source:

$$
\begin{array} { r l } & { D _ { 1 } = \mathbb { E } [ \| X _ { 1 } - \hat { X } _ { 1 } \| ^ { 2 } ] } \\ & { \quad \quad = 1 - \frac { \left( \omega _ { 1 , 2 } + ( 1 - \omega _ { 1 , 2 } ) \rho \right) ^ { 2 } } { \sigma _ { U _ { 1 , 2 } } ^ { 2 } } . } \end{array}\tag{260}
$$

Equivalently, we have the noise variance:

$$
\sigma _ { U _ { 1 , 2 } } ^ { 2 } = \frac { \left( \omega _ { 1 , 2 } + ( 1 - \omega _ { 1 , 2 } ) \rho \right) ^ { 2 } } { 1 - D _ { 1 } } .\tag{261}
$$

For the second source, we obtain:

$$
\begin{array} { r l } & { D _ { 2 } \geq \mathbb { E } [ \| X _ { 2 } - \hat { X } _ { 2 } \| ^ { 2 } ] } \\ & { \quad = 1 - \left( \frac { \omega _ { 1 , 2 } \rho + ( 1 - \omega _ { 1 , 2 } ) } { \omega _ { 1 , 2 } + ( 1 - \omega _ { 1 , 2 } ) \rho } \right) ^ { 2 } ( 1 - D _ { 1 } ) . } \end{array}\tag{262}
$$

Define:

$$
t \triangleq \frac { \omega _ { 1 , 2 } \rho + ( 1 - \omega _ { 1 , 2 } ) } { \omega _ { 1 , 2 } + ( 1 - \omega _ { 1 , 2 } ) \rho } .\tag{263}
$$

Since $\omega _ { 1 , 2 } \in ( 0 , 1 )$ , we have $t > 0$ . The distortion constraint in equation 262 becomes

$$
t ^ { 2 } \geq \frac { 1 - D _ { 2 } } { 1 - D _ { 1 } } .\tag{264}
$$

which is equivalent to $\begin{array} { r } { t \geq \sqrt { \frac { 1 - D _ { 2 } } { 1 - D _ { 1 } } } } \end{array}$ . Substituting this relation to the rate expression yields

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \log \frac { \sigma _ { U _ { 1 , 2 } } ^ { 2 } } { \sigma _ { N _ { 1 , 2 } } ^ { 2 } } = \frac { 1 } { 2 } \log \frac { 1 } { 1 - \sigma _ { \bar { X } _ { 1 , 2 } } ^ { 2 } / \sigma _ { U _ { 1 , 2 } } ^ { 2 } } } \\ & { \phantom { \frac { 1 } { 2 } \log \frac { 1 } { 1 - \frac { 1 } { \left( \omega _ { 1 , 2 } + ( 1 - \omega _ { 1 , 2 } ) \rho \right) ^ { 2 } } \sigma _ { \bar { X } _ { 1 , 2 } } ^ { 2 } } } } \\ & { \phantom { \frac { 1 } { 2 } \log \frac { 1 } { \sigma _ { \bar { X } _ { 1 , 2 } } ^ { 2 } / \sigma _ { \bar { X } _ { 1 , 2 } } ^ { 2 } } } = \frac { 1 } { 2 } \log \frac { 1 } { 1 - \frac { 1 - D _ { 1 } } { 1 - \rho ^ { 2 } } \left( 1 - 2 \rho t + t ^ { 2 } \right) } } \end{array}\tag{265}
$$

![](images/7f58ce179c3d681675319156c0513a96c8d84b934c87939e26cc9398a1c29587.jpg)

![](images/f03064aecd7916634b705b1aeb0a1ca9e2b254261658abdf69f4a022109e5215.jpg)

![](images/8a4289943622caa7dc99fcfc24aeaadf15fc238faeea4fab3ad7360d5977bd88.jpg)  
Figure 12: Gaussian source $( X _ { 1 } , X _ { 2 } )$ with $\sigma _ { 1 } ^ { 2 } = \sigma _ { 2 } ^ { 2 } = 1$ and $\rho \ = \ 0 . 7 .$ $L e f t { \mathrm { : } }$ rate-distortion $R ( D )$ comparisons for symmetric distortion constraint D. Interpolation coefficient $\omega _ { 1 , 2 } ~ = ~ 0 . 5$ for ISC. Middle: Rate gap between joint coding (ISC) and causal coding for symmetric distortion case. $R i g h t \colon$ rate-distortion $R ( D _ { 2 } )$ comparisons for asymmetric distortion constraint with fixed $d _ { 1 } = 0 . 1 , 0 . 2 ,$ , and 0.4 upside down.

Therefore, the optimization reduces to minimizing $1 - 2 \rho t + t ^ { 2 }$ , whose unconstrained minimizer is $t ^ { * } = \rho .$ Combining this with the distortion constraint in equation 262 gives:

$$
t ^ { * } = \operatorname* { m a x } \left\{ \rho , \sqrt { \frac { 1 - D _ { 2 } } { 1 - D _ { 1 } } } \right\} .\tag{266}
$$

When $t ^ { * } = \rho ,$ , the coding degenerates to $\omega _ { 1 , 2 } = 1$ , corresponding to coding only the first source. Otherwise, we have the optimal interpolation coefficient:

$$
\omega _ { 1 , 2 } ^ { * } = \frac { 1 - \rho \sqrt { \frac { 1 - D _ { 2 } } { 1 - D _ { 1 } } } } { 1 - \rho + \left( 1 - \rho \right) \sqrt { \frac { 1 - D _ { 2 } } { 1 - D _ { 1 } } } } ,\tag{267}
$$

for $D _ { 2 } < 1 - \rho ^ { 2 } ( 1 - D _ { 1 } )$ . Finally, the ISC sum-rate-distortion function becomes:

$$
R _ { 1 , \mathrm { s u m } } ( \mathbf { D } ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { D _ { 1 } D _ { 2 } } , } & { 0 < D _ { 2 } \leq \frac { 1 - \rho ^ { 2 } - D _ { 1 } } { 1 - D _ { 1 } } , } \\ { \frac { 1 } { 2 } \log \frac { 1 - \rho ^ { 2 } } { D _ { 1 } D _ { 2 } - \left( \rho - \sqrt { ( 1 - D _ { 1 } ) ( 1 - D _ { 2 } ) } \right) ^ { 2 } } , } & { \frac { 1 - \rho ^ { 2 } - D _ { 1 } } { 1 - D _ { 1 } } < D _ { 2 } < 1 - ( 1 - D _ { 1 } ) \rho ^ { 2 } , } \\ { \frac { 1 } { 2 } \log \frac { 1 } { D _ { 1 } } , } & { D _ { 2 } \geq 1 - ( 1 - D _ { 1 } ) \rho ^ { 2 } . } \end{array} \right.\tag{268}
$$

This coincides exactly with the Gaussian joint-coding rate-distortion function in equation 254, thereby establishing the JC optimality of ISC-1 for the asymmetric distortion setting.

## C.5.3 SUMMARY

We provide a simple two-frame Gaussian example to illustrate the behavior of the proposed ISC framework. Consider jointly Gaussian sources $( { \dot { X } } _ { 1 } , X _ { 2 } )$ with a quadratic distortion measure.

Symmetric distortion. We first consider the symmetric setting $D _ { 1 } = D _ { 2 } = D$ . Figure 12 (left) shows the rate-distortion curves for joint coding, causal sequential coding, and ISC-1. In this case, the sum-rate is denoted by $R _ { \mathrm { s u m } } ( { \bar { D } } ^ { 2 } )$ . As shown, ISC-1 achieves the same rate-distortion performance as joint coding in the high-resolution regime by introducing auxiliary variables through interpolation. Moreover, it strictly outperforms causal coding across the entire rate region.

Asymmetric distortion. We next consider the case where $D _ { 1 }$ is fixed and vary $D _ { 2 }$ . Figure 12 (right) shows the corresponding rate-distortion curves, denoted by $R _ { D _ { 1 } , \mathrm { { s u m } } } ( D _ { 2 } )$ . In this setting, the interpolation coefficients can be optimized according to the distortion constraint, and ISC achieves joint coding optimality across the entire feasible rate region.

Discussion. This example illustrates how interpolation-based auxiliary variables enable ISC to capture delayed dependencies efficiently. While the exact optimality result is specific to Gaussian sources, it provides a clear design principle for practical video coding: temporal dependencies can be exploited through local interpolation within a limited delay window without sacrificing coding efficiency.

## D EXPERIMENT DETAILS

The proposed GFSC architecture is model-agnostic and can be integrated into different neural video compression (NVC) models. In this section, we provide more implementation details about modeltraining, evaluation, and baseline setups. We also explicitly show how GFSC can be incorporated into DCVC (Li et al., 2021) architecture, which serves as our primary baseline. The same integration principle can be applied to other NVC architectures, such as VCT and DHVC, with only minor modifications.

## D.1 TRAINING SETUP AND OVERVIEW

Training and evaluation. All models are trained on Vimeo-90K (Xue et al., 2019) using 7-frame sequences $( \mathrm { G O P } = 7 ) $ , with the remaining training hyperparameters following the corresponding original implementations. In our implementation, we focus on evaluating the effectiveness of the proposed GFSC mechanism rather than optimizing for peak performance. Therefore, for VCT and DHVC, we adopt a simplified training procedure corresponding to the final stage of VCT, where all components (encoder, decoder, and entropy model) are trained jointly from scratch. Specifically, we train the models for 50 epochs using their original rate-distortion objectives and optimization settings. For DCVC, we adopt the full multi-stage training procedure of the original model as in Zhang & Zhu (2025) and introduce the delayed-frame modules only in the final training stage. For evaluation, we report results on UVG (Mercat et al., 2020) and MCL-JCV (Wang et al., 2016). Following standard evaluation protocols, we encode the first 96 frames of each sequence with $\mathrm { G O P = 3 2 }$ . We evaluate rate-distortion performance using bitrate (bits-per-pixel, bpp) and PSNR computed in the RGB space. For conventional video codecs, reconstructed .YUV videos are first converted to RGB before computing MSE and PSNR to ensure a consistent comparison.

Rate-distortion training. We follow the standard rate-distortion optimization used in DCVC Li et al. (2021). First, the rate-distortion loss function is defined as

$$
\begin{array} { r } { \mathcal { L } = R + \lambda D , } \end{array}
$$

where R denotes the bitrate and D denotes the distortion. Both are averaged across the video sequence batch during training. The trade-off hyperparameter λ is selected from $\lambda ~ \in ~ 0 . 0 1 \times$ $\{ 2 , 1 , 2 ^ { - 1 } , 2 ^ { - 2 } , 2 ^ { - 3 } , 2 ^ { - 4 } \}$ , covering both low- and high-rate regimes. For other settings, we use the Adam optimizer with a batch size of 8, an initial learning rate of $1 \times 1 0 ^ { - 5 }$ and aux learning rate of $1 \times 1 0 ^ { - 3 }$ . Training is performed for 30 epochs with a linear learning rate schedule: the learning rate is warmed up for the first 2k steps and then linearly decayed to $1 \times \mathrm { { 1 0 ^ { - 6 } } }$ . All models are trained on a single NVIDIA RTX 5090 GPU using 2 days.

Delayed coding baselines. As discussed in Section 4, we additionally compare ISC with practical delayed coding frameworks, including Random Access (RA) coding (i.e., bidirectional coding), under matched delay constraints $( K \leq \bar { 3 } )$ . We first clarify the connection between sequential coding delay and RA coding configurations.

![](images/3b8bbcda5b7d2f5a11a797900aa4bc34b5201cc4c88e1a5b908d5d2722ae3709.jpg)  
Figure 13: Illustration of the delay induced by B-frame coding over an N-length video clip.

![](images/87cd32c0f001f39aa46d69ca86295ed64fded7273baf521c33e1ea452a731376.jpg)  
Figure 14: Simplified DCVC and GFSC architectures. GFSC introduces temporal joint modeling (TJM) modules to both motion and context branch, highlighted in orange and purple, respectively. One delayed frame case is shown. Additional delayed frames provide further inputs to these modules. The mappings $f _ { m } , f _ { c }$ and $g _ { m } , g _ { c }$ denote motion/context encoders and decoders, respectively.

Consider an L-frame video sequence. In RA coding, the encoder first buffers a short frame group and performs bidirectional prediction within the group. For example, in an I-B-B-B-I structure, intermediate B-frames depend on both past and future anchor frames. As a result, the encoder must wait for the arrival of the future I-frame before encoding the intermediate frames, leading to an effective practical delay determined by the distance between the first B-frame and the future anchor frame. The decoder then reconstructs frames in coding order and reorders them back to display order. Figure 13 illustrates the corresponding delay configurations.

We consider two representative delayed coding baselines: DCVC-B Sheng et al. (2025) and HM-RA. For DCVC-B, the neural architecture allows flexible extension to delay settings. Therefore, both $K = 1$ and $K = 2$ are implemented by customizing the reference frame lists for each frame. For HM-RA, the hierarchical GOP structure is designed primarily for even GOP sizes. Accordingly, we modify the standard encoder\_randomaccess\_main.cfg to implement 1-delay and 3- delay settings using I-B-I and $_ { \texttt { I } - \texttt { B } - \texttt { B } - \texttt { B } - \texttt { I } }$ structures, respectively. The intermediate $I - \mathrm { B - B - I }$ configuration is not directly supported under the standard HM-RA hierarchical coding structure.

## D.2 NETWORK ARCHITECTURE

We describe how GFSC is integrated into the causal DCVC architecture. The original DCVC codec contains separate I-frame and P-frame models. For P-frame coding, we distinguish two branches: a motion branch with analysis/synthesis transforms $\left( f _ { m } , g _ { m } \right)$ and a context branch with transforms $( f _ { c } , g _ { c } )$ . Each branch retains its original entropy-coding module based on the CompressAI implementation (Begaint et al., 2020). Generally, GFSC introduces temporal joint modeling (TJM)´ modules between the analysis transforms and the subsequent coding modules, while leaving the remaining DCVC pipeline unchanged. Figure 14 compares the original DCVC and DCVC-GFSC architectures for the one-frame-delay case.

One-frame delay $( K = 1 )$ . For every stage $\ell < L$ , GFSC jointly models the current representation and the available one-frame lookahead.

1. For an I-frame stage, let $f _ { I }$ and $g _ { I }$ denote the corresponding analysis and synthesis transforms. Given current and delayed frame pair $\left( \mathbf { x } _ { \ell } , \mathbf { x } _ { \ell + 1 } \right)$ , we first compute $\mathbf { y } _ { \ell } = f _ { I } ( \mathbf { x } _ { \ell } )$ and $\mathbf { y } _ { \ell + 1 } = f _ { I } ( \mathbf { x } _ { \ell + 1 } )$ . Then we have the following joint representation

$$
\begin{array} { r } { \tilde { \mathbf { y } } _ { \ell } = \mathrm { T J M } _ { c } \left( \mathbf { y } _ { \ell } , \mathbf { y } _ { \ell + 1 } \right) . } \end{array}
$$

Here, $\mathrm { T J M } _ { c }$ is implemented using the window-based spatial-temporal attention module described below.

![](images/ea8a4ca8bb47dd500b9e7cb1fc217099081236df910c88a50ee2c59f770da276.jpg)  
Figure 15: Temporal joint modeling modules for motion and context branches in DCVC architecture. $L e f t colon \mathrm { T J M } _ { m }$ concatenates motion features and their absolute difference as input. $R i g h t \colon$ $\mathrm { T J M } _ { c }$ jointly updates the two context features using a spatial-temporal self attention module.

2. For P-frames, DCVC contains separate motion and context representations. We therefore introduce two corresponding temporal modules, $\mathrm { T J M } _ { m }$ and $\mathrm { T J M } _ { c }$ . For each branch $b \in$ $\{ m , c \}$

$$
\begin{array} { r } { \tilde { \mathbf { y } } _ { \ell } ^ { b } = \mathrm { T J M } _ { b } \left( \mathbf { y } _ { \ell } ^ { b } , \mathbf { y } _ { \ell + 1 } ^ { b } \right) , \qquad \mathbf { y } _ { \ell } ^ { b } = f _ { b } ( \cdot ) . } \end{array}
$$

The context module $\mathrm { T J M } _ { c }$ uses spatial-temporal self-attention to jointly learn delayed latent representations, whereas $\mathrm { T J M } _ { m }$ uses a lightweight convolutional module to jointly model the motion features. The resulting representations are then passed to the original entropy-coding and decoding pipeline. All P-frame stages share the same network.

Note that at the final stage $\ell = L$ , no future frame is available. GFSC therefore bypasses temporal fusion and directly forwards the current latent representation $\tilde { \mathbf { y } } _ { L } ^ { b } = \mathbf { y } _ { L } ^ { b }$ for $b \in \{ \bar { m } , c \}$ . Because the TJM outputs preserve the original latent tensor dimensions, the subsequent entropy models and synthesis transforms can be used without modification.

Extension to larger delays. The same construction extends directly to $K > 1$ . For the context branch, the spatial-temporal attention module jointly processes all available latent representations within the delay window:

$$
\tilde { \mathbf { y } } _ { \ell } ^ { c } = \mathrm { T J M } _ { c } \left( \mathbf { y } _ { \ell } ^ { c } , \cdot \cdot \cdot , \mathbf { y } _ { \mathrm { m i n } \{ \ell + K , L \} } ^ { c } \right) .
$$

The attention formulation naturally supports a variable number of available future frames near the sequence boundary. For the motion branch, $\mathrm { T J M } _ { m }$ uses a fixed-size input corresponding to the selected delay budget. For example, when $K = 2 .$

$$
\tilde { \mathbf { y } } _ { \ell } ^ { m } = \mathrm { T J M } _ { m } \left( \mathbf { y } _ { \ell } ^ { m } , \mathbf { y } _ { \ell + 1 } ^ { m } , \mathbf { y } _ { \ell + 2 } ^ { m } \right) .
$$

If fewer than K future representations are available, the missing inputs are replaced by zero tensors of the same shape. The same padding rule is used during both training and inference.

Coding and quantization. After joint modeling within the K-delay window, the learned representation $\mathbf { z } _ { \ell }$ is quantized and entropy-coded. Following the one-shot formulation in Section 3.2, we adopt universal quantization with shared randomness Gray & Stockham (2002); Ziv (2003); Theis & Agustsson (2021) to implement one-shot channel simulation. Specifically, the encoder computes

$$
\mathbf { z } _ { \ell } = \Delta \left\lfloor \frac { \tilde { \mathbf { y } } _ { \ell } + \mathbf { w } } { \Delta } \right\rceil ,
$$

where $\mathbf { w } \sim \mathcal { U } ( - \Delta / 2 , \Delta / 2 )$ denotes shared randomness. At the decoder side, the quantized latent is reconstructed as

$$
\begin{array} { r } { \mathbf { u } _ { \ell } = \mathbf { z } _ { \ell } - \mathbf { w } , } \end{array}\tag{269}
$$

![](images/5ca0c1f73d9c1d4a25a7219004d3390cc0ddd86a116f0969afcee06f93a07543.jpg)

BD-rate comparison.
<table><tr><td>Model</td><td>BD-rate (%)</td></tr><tr><td>DCVC</td><td>0.00</td></tr><tr><td>DCVC-GFSC-1</td><td>-3.78</td></tr><tr><td>DCVC-GFSC-2</td><td>-4.50</td></tr><tr><td>DCVC-GFSC-3</td><td>-5.11</td></tr></table>

Figure 16: Left: Rate-PSNR curves on LIVE-HDR 4K. Right: BD-rate relative to causal DCVC. Negative values indicate rate savings.

since the randomness w is shared between the encoder and decoder. Let $g _ { \ell }$ denote the decoder network. The reconstructed frame is then obtained by:

$$
\hat { \mathbf { x } } _ { \ell } = g _ { \ell } ( \mathbf { u } ^ { \ell } ) .\tag{270}
$$

## D.3 ADDITIONAL EXPERIMENTAL ANALYSIS

We further evaluate GFSC-enhanced DCVC on the 4K LIVE-HDR dataset (Shang et al., 2022). Compared with causal DCVC, GFSC achieves BD-rate reductions of 3.78%, 4.50%, and 5.11% for delays $K = 1 , 2 , 3$ , respectively. These results show that exploiting information from delayed frames also improves rate-distortion performance for 4K video.

## E LIMITATIONS

Despite the theoretical and empirical results established in this work, several questions remain open. First, our joint-coding optimality result is characterized over a nontrivial distortion region. Extending this characterization beyond the current region, and determining whether alternative auxiliaryvariable constructions can enlarge the optimality regime, are important directions for future work. This limitation concerns only the extent of the characterized region and does not affect the established optimality guarantee within it.

Second, while the general Markov analysis establishes the existence and graphical structure of JCoptimal clique-wise auxiliary variables, it does not yield a simple closed-form parameterization for arbitrary source distributions. For Gauss-Markov sources, the additional Gaussian structure makes this construction explicit and reduces the clique-wise interactions to pairwise linear interpolation. Developing similarly tractable constructions for broader non-Gaussian source families is an interesting theoretical direction.

Finally, the computational cost of GFSC grows approximately with the delay budget because additional future-frame interactions must be modeled. This overhead is moderate in the limited-delay regime studied in this work, but scaling GFSC to substantially longer delays while controlling computation and memory remains an important engineering direction.