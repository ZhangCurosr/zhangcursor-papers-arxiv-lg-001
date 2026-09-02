# MASKCODE: Mask Transformer for Feedback-Assisted Coding With Linear Block Codes

Jonggyu Jang, Hongjae Nam, Vishrant Tripathi, David J. Love, and Hyun Jong Yang

Abstract—Feedback-based coding schemes have demonstrated substantial performance gains over today’s open-loop coding schemes. Unfortunately, these gains are usually achieved in idealized settings with perfect feedback. Over the last few years, machine learning-based schemes have been shown to be promising solutions for implementing feedback-based codes, particularly when combined with short-block-length open-loop error correcting codes (ECCs) in a concatenated coding structure. However, existing ML-based feedback schemes remain agnostic to the outer code’s structure, potentially misallocating feedback resources on error patterns already correctable by the outer ECC. To address this, we propose MASKCODE, a Transformerbased inner feedback code for concatenated coding systems, which explicitly incorporates structural knowledge of the outer linear block code into the inner feedback encoder design via two synergistic mechanisms: 1) a soft syndrome-based input that informs the encoder about potential parity constraint violations, and 2) a code-aware attention mask derived from the Tanner graph. We further show that end-to-end training with a differentiable belief propagation (BP) decoder offers no additional gain, as MASKCODE’s structure-aware design already internalizes the structural knowledge of the outer code; in fact, backpropagation through the iterative BP decoder introduces gradient explosion, which degrades rather than improves performance. Extensive evaluations on BCH and LDPC outer codes demonstrate that MASKCODE consistently outperforms all baselines, achieving up to 1.5 dB SNR gain.

Index Terms—Concatenated code, feedback coding, linear block code, noisy feedback, and error correcting code.

## I. INTRODUCTION

Over the past decade, artificial intelligence and machine learning (AI/ML) have emerged as a potential key enabler of 6G communications [1]. More specifically, deep learning (DL) leverages deep neural networks (DNNs) trained via data-driven optimization to represent intricate nonlinear functions. By virtue of the data-driven optimization, AI/ML has effectively addressed mathematically intractable problems across a wide range of telecommunication technologies, such as channel prediction, channel decoding, radio resource allocation, network optimization, and waveform learning [2]–[8].

![](images/860f525193c30e8d8edf2bfc1978f428329fd0a43c7f692a2eef4654f061e2c4.jpg)  
Fig. 1. Illustration of the feedback-assisted coding system with an outer linear block code and an inner feedback code.

One notable application of ML-based methods in wireless communications is feedback coding. In early work, Shannon showed that feedback does not increase channel capacity for memoryless channels [9], [10]; however, subsequent studies [11]–[14] have shown that closed-loop feedback codes enable the probability of error to decay double-exponentially with blocklength, by adaptively refining the transmission using receiver-side information. An early linear realization, the Schalkwijk-Kailath (SK) scheme [11], [12], achieves capacity and exhibits an error probability that decays doubleexponentially. When the feedback link is itself noisy, however, the SK scheme loses not only the double-exponential decay but also the capacity-achieving rate [15]–[17]. In this noisyfeedback regime, the Chance-Love (CL) scheme [18] achieves the error exponent bound established in [16], but its encoding and decoding structures remain constrained to linear operations. More recently, ML-based feedback coding [19]–[23] replaces the linear encoder and decoder with DNNs, demonstrating substantial performance gain over linear schemes.

Beyond standalone feedback coding, concatenated coding offers a promising research direction that integrates feedback coding with conventional error-correcting codes (ECCs) [18], [24]–[26]. These coding architectures employ a two-stage structure consisting of an inner code and an outer code, as depicted in Fig. 1, where the outer code is a general open-loop ECC, and the inner code is a feedback coding scheme that exploits closed-loop communication channels. This concatenated design raises the question of how the inner feedback code should be designed to best support the outer decoder.

## A. Related Works

Here, we provide a literature survey for 1) linear feedback coding, 2) analytical nonlinear feedback coding, 3) ML-based feedback coding, 4) feedback coding in concatenated coding architecture, and 5) ML-based ECC decoders.

Linear feedback coding. As a seminal work, Schalkwijk et al. proposed a capacity-achieving linear feedback coding strategy for Gaussian channels with the noiseless feedback setting, which we refer to as the SK scheme [11], [12]. However, once the feedback becomes noisy, the SK scheme loses both its capacity-achieving property and its doubleexponential error decay. To address this limitation, several linear feedback coding schemes have been proposed based on analytical optimization, notably the CL scheme [18] and dynamic programming [13]. In [27], [28], extensions of the CL scheme were developed for correlated noise and broadcasting channel scenarios. However, these schemes are restricted to linear encoding and decoding structures. As a result, they do not capture the potential gains of more general nonlinear feedback schemes.

Analytical nonlinear feedback coding. To circumvent the performance bottlenecks imposed by linear schemes, several studies have explored the integration of nonlinear operations into feedback coding [23], [29]–[31]. In [29], Gallager-Nakiboglu (GN) scheme exploits nonlinearities by correcting˘ single-distance errors via pulse amplitude modulation (PAM) signaling. On the other hand, in [30], [31], the Modulo-SK scheme was proposed to handle noisy feedback by leveraging a nonlinear modulo operation. Most recently, PowerBlast [23] was proposed as a hybrid architecture, combining the SK scheme and GN scheme. All the aforementioned studies have proposed nonlinear operations; however, these nonlinear operations (e.g., modulo and PAM signaling) are manually engineered, thereby lacking flexible representation.

ML-based feedback coding. Recently, DNN-based feedback coding has emerged as a compelling alternative to linear schemes, unlocking the potential performance gains of nonlinear feedback coding. In [19], a pioneering scheme, DeepCode, was developed, which uses recurrent neural network (RNN) architectures for both encoder and decoder. Even with only two RNN layers, DeepCode can outperform linear methods, such as SK scheme [11], [12] and CL scheme [18], in certain regimes in terms of BLER. Following the success of AI/ML in feedback coding, subsequent works have explored various architectures, including RNNs [20], [32], [33], Transformers [21], [22], and light-weight neural networks [23], and two-way learning-based extensions comparing multiple neural architectures [34], [35].

Feedback coding in concatenated codes. The concept of concatenated coding was first introduced by Forney [24], which has been historically validated in a wide variety of deployments, including NASA’s deep space mission [25]. In concatenated coding frameworks, feedback coding can serve as the inner code. The authors of [26] theoretically showed that the use of feedback yields superior error exponents compared to those achievable without feedback. In [18], the authors introduced the CL scheme as a feedback-driven inner code for outer ECCs, thereby exploiting the feedback gain in concatenated coding. In [19], DeepCode was shown to outperform the CL scheme and SK scheme when used as an inner code with an outer Turbo code.

ML-based ECCs. As another application of AI/ML, ECC decoding has been extensively investigated using DNNs [4], [7], [36], [37]. A prominent trend in this line of research is the adoption of Transformer architectures. A representative work, ECCT [4], and its extensions [7], [36] employ attention masks derived from hard-decision syndrome information to steer the model toward code-structure-aware representations, thereby enhancing decoding performance.

## B. Challenges and Contributions

While ML-based feedback coding has shown substantial gains over linear schemes, existing studies have largely focused on the standalone setting [19]–[23], leaving its design within concatenated architectures underexplored. Because the inner feedback code determines the quality of the effective channel (super-channel) [24] seen by the outer decoder, the structural limitations imposed by linear inner feedback codes carry over directly to concatenated systems, motivating the use of ML-based nonlinear inner feedback codes. Although [19] has shown the promise of this direction by combining an ML-based inner code with an outer Turbo code [38], no comprehensive framework exists for the synergistic co-design of ML-based inner feedback codes and outer decoders. To bridge this research gap, this paper identifies and addresses the following challenges:

Challenge 1: Lack of synergistic design. When employed as inner codes in concatenated codes, existing MLbased feedback coding schemes—trained solely to minimize their own bit-error rate (BER) by treating the input bits as independent—ignore the algebraic constraints of the outer code [19]–[23], i.e., parity checking, $\mathbf { c H } ^ { \mathrm { T } } \ = \ \mathbf { 0 }$ for codeword c and parity check matrix H. Hence, while compatible, they are ill-suited to the structure of outer ECCs, leading to suboptimal performance in concatenated systems.

Challenge 2: End-to-end training. End-to-end training strategies, which jointly optimize the inner code with the outer decoder, serve as a direct and intuitive way for synergistic design. While such paradigms have been proposed for Turbo and convolutional codes [39], [40], the efficacy of end-to-end training with inner feedback coding remains unexplored—a non-trivial problem, as incorporating a differentiable iterative outer decoder introduces gradient explosion during backpropagation.

Contribution. To address these challenges, we propose MASKCODE, a Transformer-based architecture where the inner encoder and decoder are designed to internalize the structural design of the outer ECCs. Specifically, we consider two representative linear block codes as outer ECCs: Bose-Chaudhuri-Hocquenghem (BCH) [41], [42] and low-density parity-check (LDPC) [43], [44], both of which can be represented as Tanner graphs and decoded via belief propagation (BP), enabling differentiable end-to-end training. In a nutshell, our contributions are threefold:

Synergistic inner feedback coding: In MASKCODE, we propose two key mechanisms to internalize the structural constraints of the outer code: 1) a soft-syndrome-based input that enables the inner code to recognize parity-check constraints, and 2) a masking strategy that repurposes the self-attention mask to enforce code-structure-aware attention patterns derived from the Tanner graph, ensuring only valid message-passing interactions are learned. By doing so, MASKCODE strengthens bits that are difficult for the outer decoder to correct.

Analysis of end-to-end training: We numerically evaluate the effect of end-to-end training on feedback-assisted concatenated codes by comparing ML-based models trained with various numbers of BP iterations. Counterintuitively, three or more BP iterations degrade BLER performance, which we attribute to the gradient instability introduced by unrolling BP iterations. In particular, we show that the integration of the outer code structure into the MASKCODE architecture makes end-to-end training unnecessary.

Comprehensive benchmark results for concatenated code systems: Unlike previous studies [20]–[23], we provide a systematic benchmark evaluating ML-based feedback coding schemes as inner codes in concatenated code systems.

Notations. Throughout this paper, we will use the following notations. Lowercase boldface letters represent row vectors, $\mathbf { e . g . } , \mathbf { v } \in \mathbb { R } ^ { 1 \times n }$ . The i-th element of the vector v is written as $v _ { i } .$ . Also, for vectors with a subscript, the i-th element of the vector $\mathbf { v } _ { k }$ is written as $v _ { k , i }$ . The $( i , j )$ -th element of matrix M is denoted as $[ \mathbf { M } ] _ { i , j }$ . For an integer K, [K] represents the set of integers from 1 to K, i.e., $[ K ] = \{ 1 , \stackrel { \cdot } { 2 } , \dotsc , K \}$

## II. BACKGROUND AND PRELIMINARIES

To facilitate a better understanding of the proposed framework, this section establishes the necessary background regarding ECCs and Transformer models [45]. Experts and practitioners with a pre-established understanding of ECCs and the Transformer model may choose to skip this section.

## A. Error Correcting Codes

In our concatenated coding system, we employ a linear block code as an outer code. Let be an $( n , k )$ binary linear block code over $\mathbb { F } _ { 2 } \ : = \ \{ 0 , 1 \}$ , characterized by a code length n and message length k with the code rate of $k / n$ . The linear code $\mathcal { C } \subseteq \{ 0 , 1 \} ^ { - }$ is described by a generator matrix $\mathbf { G } \in \{ 0 , 1 \} ^ { k \times n }$ and a parity-check matrix $\mathbf { \bar { H } } \in \{ 0 , 1 \} ^ { ( n - k ) \times n }$ , satisfying $\mathbf G \mathbf H ^ { \mathrm T } = \mathbf 0 _ { k \times ( n - k ) }$ [46]. First, the encoder produces codeword $\mathbf { c } \in \{ 0 , 1 \} ^ { 1 \times \times n }$ from the binary message m $\bar { \in } \{ 0 , 1 \} ^ { 1 \times k }$ using a linear mapping defined by G, i.e.,

$$
\mathbf { c } = \mathbf { m } \mathbf { G } \in \{ 0 , 1 \} ^ { 1 \times n } .\tag{1}
$$

In particular, the row space of G defines the linear code ${ \mathcal { C } } ,$ and every codeword $\mathbf { c } \in { \mathcal { C } }$ is orthogonal to each row of H, $\mathrm { i . e . , } \mathrm { \mathbf { c H } ^ { \mathrm { T } } = { \mathbf { 0 } } }$ for all $\mathbf { c } \in { \mathcal { C } }$ . Let $\mathbf { y } \in \mathbb { R } ^ { 1 \times n }$ denote the channel observation corresponding to the transmitted codeword c. The hard-decision of y is denoted by $\mathbf { u } \in \{ 0 , 1 \} ^ { 1 \times n }$ . The hard syndrome is defined as ${ \mathbf { u } } { \mathbf { H } } ^ { \mathrm { T } } \in \dot { \{ 0 , 1 \} } ^ { 1 \times ( n - k ) }$ , which measures the parity-check violations and satisfies $\mathbf { u } \mathbf { H } ^ { \mathrm { T } } = \mathbf { 0 } _ { 1 \times ( n - k ) }$ if and only if u is a valid codeword. The ECC decoder relies on the structural knowledge that $\mathbf { c } \mathbf { H } ^ { \mathrm { T } } = \mathbf { 0 } _ { 1 \times ( n - k ) }$ , regardless of the underlying channel medium, such as a raw physical channel or a virtual super-channel formed by an inner code.

Belief propagation. In this work, we adopt the sum-product algorithm (the most widely used BP algorithm for decoding linear block codes) as the outer decoder, referred to as the BP decoder. The BP operates on a Tanner graph, a bipartite graph used to specify an ECC, consisting of n variable nodes and $n - k$ check nodes, where an edge exists between variable node v and check node c if $[ \mathbf { H } ] _ { c , v } = 1$ . For the BP decoder, we define that the LLRs of the decoder’s input ${ \widehat { \mathbf { c } } } ,$ i.e.

$$
L _ { i } = \widehat { c } _ { i } = \log \left( \frac { \mathrm { P r } ( c _ { i } = 1 | \mathbf { y } ) } { \mathrm { P r } ( c _ { i } = 0 | \mathbf { y } ) } \right) .\tag{2}
$$

Then, the BP algorithm iteratively exchanges two types of messages over each edge of the Tanner graph, a variable-tocheck message $\mu _ { \mathrm { v c } } ^ { i  j }$ and a check-to-variable message $\mu _ { \mathrm { c v } } ^ { j  i }$ initialized as $\mu _ { \mathrm { c v } } ^ { j \to i } = 0$ for all edges [47]:

1) Variable-to-check message $( \mu _ { \mathbf { v c } } ^ { i  j } ) { \mathrm { : } }$ : The variable-to-check message aggregates beliefs from the neighboring check nodes by

$$
\mu _ { \mathrm { v c } } ^ { i  j }  L _ { i } + \sum _ { k \in N _ { \mathrm { c } } ( i ) \backslash \{ j \} } \mu _ { \mathrm { c v } } ^ { k  i } ,\tag{3}
$$

where $N _ { \mathrm { c } } ( i ) = \{ j \in [ n - k ] \ | \ [ \mathbf { H } ] _ { j , i } = 1 \}$

2) Check-to-variable message $( \mu _ { \mathbf { c } \mathbf { v } } ^ { \jmath \to \iota } ) \colon$ the check-to-variable message aggregates belief from the neighboring variable nodes by

$$
\mu _ { \mathrm { c v } } ^ { j \to i }  2 \mathrm { a r c t a n h } ( \prod _ { k \in N _ { \mathrm { v } } ( j ) \backslash i } \mathrm { t a n h } ( \frac { \mu _ { \mathrm { v c } } ^ { k \to j } } { 2 } ) ) ,\tag{4}
$$

where $N _ { \mathrm { v } } ( j ) = \{ i \in [ n ] \ | \ [ \mathbf { H } ] _ { j , i } = 1 \}$

After sufficient iterations, the final posterior LLR is computed as

$$
L _ { i } ^ { \prime } = L _ { i } + \sum _ { k \in N _ { \mathrm { c } } ( i ) } \mu _ { \mathrm { c v } } ^ { k \to i } , \forall i \in [ n ] ,\tag{5}
$$

which is then used to reconstruct the message m. In the bremainder of this work, the BP decoding algorithm is denoted as

$$
\widehat { \mathbf { c } } _ { \mathrm { o u t } } = \mathbf { B P } ( \widehat { \mathbf { c } } _ { \mathrm { i n } } ; N _ { \mathrm { B P } } ) ,\tag{6}
$$

where $\widehat { \mathbf { c } } _ { \mathrm { i n } } ~ \in ~ \mathbb { R } ^ { 1 \times n }$ and $\widehat { \mathbf { c } } _ { \mathrm { o u t } } \in \mathbb { R } ^ { 1 \times n }$ denote the input and b boutput LLR vectors of the BP decoder, respectively, and $N _ { \mathrm { B P } }$ is the number of message-passing iterations.

## B. Transformer Layer

MASKCODE builds upon the Transformer architecture [45], which we briefly introduce here. The Transformer model consists of a couple of Transformer layers, which are depicted in Fig. 2. As shown in the right part of the figure, a Transformer layer consists of 1) a masked self-attention block, 2) residual connections and layer normalization (Add & LayerNorm), and 3) a feedforward network (FFN). The detailed operations of the residual connections and feedforward network are well known, so we focus on the masked self-attention block.

![](images/2192097468d7acfc8fbcd6c93babbccb96456115eaeca70e419888c9807c7f40.jpg)  
Fig. 2. An illustration of a Transformer layer, consisting of 1) a masked self-attention block, 2) residual connections, and 3) a feedforward network.

Self-attention. Let us consider an embedding $\textbf { X } \in \ \mathbb { R } ^ { \ell \times d }$ where ℓ and d denote the embedding dimension and the number of input tokens, respectively. The self-attention mechanism provides a way to compute a contextual representation of sequence X by relating input tokens in different positions [45]. It operates on three distinct matrices: the query $\mathbf { Q } ,$ key K, and value V. These matrices are generated by applying learnable linear projections $( \mathbf { W } _ { Q } , \mathbf { W } _ { K }$ , and $\mathbf { W } _ { V } \in \mathbb { R } ^ { \ell \times \ell } )$ to input X:

$$
\begin{array} { r } { \mathbf q = \mathbf W _ { Q } \mathbf X , \quad \mathbf K = \mathbf W _ { K } \mathbf X , \quad \mathbf V = \mathbf W _ { V } \mathbf X , } \end{array}\tag{7}
$$

where we follow a transposed convention from the original Transformer paper [45] such that Q, K, and V are columnwise token representations. The query acts as a probe seeking relevant information, the key serves as an index for matching, and the value contains the actual feature content to be retrieved. Then, the self-attention output without masking can be obtained by Attention $\begin{array} { r } { ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) \ = \ \mathbf { V } ^ { \mathrm { T } } \mathbf { s } \circ \mathbf { f } \tan \mathbf { a x } \left( \frac { \mathbf { K } ^ { \mathrm { T } } \mathbf { Q } } { \sqrt { \ell } } \right) } \end{array}$ where the softmax function softmax( ) is applied to each column of the matrix.

Masked self-attention. While the self-attention mechanism enables each token to aggregate information from all others, this fully connected interaction is not always desirable. For instance, in autoregressive language modeling, a token should not aggregate future tokens to maintain causality [45]. To control the information flow, a mask matrix $( \mathbf { M } \in \{ - \infty , 0 \} ^ { d \times d } )$ is generally used in a self-attention block, i.e.,

$$
{ \mathrm { A t t e n t i o n } } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathbf { V } ^ { \mathrm { T } } { \mathbf { s o f t m a x } } \left( { \frac { \mathbf { K } ^ { \mathrm { T } } \mathbf { Q } } { \sqrt { \ell } } } + \mathbf { M } \right)\tag{8}
$$

In (8), setting $[ { \bf { M } } ] _ { i j } ~ = ~ - \infty$ ensures that the j-th column of the output does not aggregate information from the i-th column of the value matrix V. By appropriately designing M, task-specific structural prior knowledge, such as causality [45] or code structure [4], [37], can be directly embedded into the attention computation, allowing the Transformer layers to focus only on task-relevant interactions.

## III. SYSTEM MODEL: FEEDBACK-ASSISTED CONCATENATED CODES

In this section, we introduce the system model for feedbackassisted concatenated coding, which integrates i) an outer ECC for open-loop error control and ii) an inner feedback code for utilizing channel output feedback. The outer linear block code ${ \mathcal { C } } \subseteq { \mathsf { \bar { \{ 0 , 1 \} } } } ^ { 1 \times n }$ is characterized by a pair of matrices: a generator matrix $\mathbf { G } \in \{ 0 , 1 \} ^ { k \times n }$ and a parity-check matrix $\mathbf { H } \mathbf { \bar { \Pi } } \in \{ 0 , 1 \} ^ { ( n - k ) \times n }$ , satisfying $\mathbf G \mathbf H ^ { \mathrm { T } } = \mathbf 0 _ { k \times ( n - k ) }$ [46].

## A. Channel Model

We consider a feedback-enabled memoryless additive white Gaussian noise (AWGN) channel with $\dot { T }$ closed-loop iterations, hereafter referred to as phases. The transmitter seeks to reliably send a k-bit message m by transmitting its encoded nbit codeword $\mathbf { c } = \mathbf { m } \mathbf { G }$ through the inner feedback code. The feedforward and feedback channel noises are modeled as independent Gaussian random variables $\mathcal { N } ( 0 , \sigma _ { \mathrm { f f } } ^ { 2 } )$ and ${ \mathcal { N } } ( 0 , \sigma _ { \mathrm { f b } } ^ { 2 } )$ where $\sigma _ { \mathrm { f f } } ^ { 2 }$ and $\sigma _ { \mathrm { f b } } ^ { 2 }$ denote the respective noise variances. The inner code transmits an n-bit codeword c via M channel uses, yielding an inner code rate of $n / M .$ . In the proposed method, we adopt the bit-based feedback protocol [11], [18], [22] over the sub-block-based feedback protocol [21], [23]; hence, $M \ = \ n T$ . At the t-th forward transmission phase, the transmitter sends an n-length vector $\mathbf { x } _ { t } ,$ referred to as the parity message, subject to an average power constraint

$$
\frac { 1 } { n } \mathbb { E } \left[ \left\| \mathbf { x } _ { t } \right\| ^ { 2 } \right] \leq P ,\tag{9}
$$

where $P$ denotes the average power constraint per phase. Denoting the feedforward noise at the t-th phase as $\mathbf { n } _ { t } \sim$ $\mathcal { N } ( 0 , \sigma _ { \mathrm { f f } } ^ { 2 } \mathbf { I } _ { n } )$ , the received signal $\mathbf { y } _ { t }$ is given by

$$
\mathbf { y } _ { t } = \mathbf { x } _ { t } + \mathbf { n } _ { t } , \forall t \in [ T ] .\tag{10}
$$

The receiver then sends $\mathbf { y } _ { t }$ back to the transmitter via the feedback channel. Following [11], [18]–[20], [48], we assume the passive feedback, whereby the feedback signal received at the transmitter is given by

$$
\tilde { { \bf y } } _ { t } = { \bf y } _ { t } + { \bf z } _ { t } = { \bf x } _ { t } + { \bf n } _ { t } + { \bf z } _ { t } , \ \forall t \in [ T - 1 ] ,\tag{11}
$$

where $\mathbf { z } _ { t } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { f b } } ^ { 2 } \mathbf { I } _ { n } )$ denotes the feedback noise at the t-th phase. With this channel model in place, we next define the encoding and decoding procedure of the inner feedback code.

## B. Inner Feedback Code

The inner feedback code operates over the channel defined above, utilizing noisy feedback to iteratively refine the transmitted signal over T phases. Figure 3 depicts an unfolded diagram of the system model with $T = 3$ . The outer encoder first generates an n-bit codeword $\mathbf { c } ~ = ~ \mathbf { m } \mathbf { G } ~ \in ~ \mathcal { C }$ , after which the inner encoder performs T closed-loop transmission iterations. The inner decoder then provides the LLRs of the estimated codeword, which are subsequently passed to the outer decoder for BP decoding. We now formally define the encoding and decoding functions of the inner feedback code.

The inner encoder aims to generate parity message x<sub>t</sub> from the codeword c and the previously received feedback parity messages $\tilde { \mathbf { y } } _ { 1 } , \tilde { \mathbf { y } } _ { 2 } , \ldots , \tilde { \mathbf { y } } _ { t - 1 }$ . Here, we represent the inner encoding as a function $f _ { \mathrm { e n c } } ( \cdot )$ , where the encoding process is given by

$$
\begin{array} { r } { \mathbf x _ { t } = \left\{ \begin{array} { l l } { f _ { \mathrm { e n c } } ( \mathbf c ) , } & { \mathrm { i f ~ } t = 1 , } \\ { f _ { \mathrm { e n c } } ( \mathbf c , \tilde { \mathbf y } _ { 1 } , \cdot \cdot \cdot , \tilde { \mathbf y } _ { t - 1 } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{12}
$$

![](images/5800a50d2daad5faacde181957657c1b426281f1878aaefdcec4b2b1d92e06ea.jpg)  
Fig. 3. Illustration of the unfolded system model of the three-phase concatenated coding $( M = n \cdot T , T = 3 )$ . Let us consider information bits with a length of k. First, the outer encoder generates an n-bit codeword c by (1). Second, the inner encoder exploits T closed-loop iterations of transmitting parity messages x to the receiver. Third, the inner decoder works as a soft-decision estimator for the transmitted codeword c, where the estimated codeword is c. Last, the outer decoder—a conventional linear block code decoder—forwards decoded information bits mb .

At the receiver, the decoder aims to get the estimated codeword c from the received parity messages $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { T }$ bSimilar to the inner encoder in (12), we represent the inner decoder as a function $f _ { \mathrm { d e c } } ( \cdot )$ . Then, the inner decoding process is defined by

$$
\widehat { \mathbf { c } } = f _ { \mathrm { d e c } } ( \mathbf { y } _ { 1 } , \cdot \cdot \cdot , \mathbf { y } _ { T } ) .\tag{13}
$$

What remains is to specify how $f _ { \mathrm { e n c } }$ and $f _ { \mathrm { d e c } }$ are designed to ensure reliable transmission.

Linear feedback codes: Classical methods like SK [11] and CL [18] schemes define the encoder $f _ { \mathrm { e n c } } ( \cdot )$ as a linear combination of $\mathbf { c } , \tilde { \mathbf { y } } _ { 1 } , \tilde { \mathbf { y } } _ { 2 } , \ldots , \tilde { \mathbf { y } } _ { t - 1 }$ . Similarly, the decoder function $f _ { \mathrm { d e c } } ( \cdot )$ is defined by a linear combination of $\mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { T }$ . Despite their analytical tractability, they are strictly limited by their linear architecture.

ML-based feedback codes: The ML-based methods leverage neural networks like RNNs or Transformers (e.g., DeepCode [19], AttentionCode [22], RobustCode [20], and [21]). These nonlinear functions can represent robust, high-dimensional signaling mappings that outperform the linear baselines.

Compatibility with outer decoders. For ML-based feedback codes, the decoder DNNs are typically trained using the binary cross-entropy (BCE) loss. Notably, the decoder output c fed binto the BCE loss is equivalent to a vector of LLRs of the transmitted bits. Since the LLR output can be readily converted into hard decisions, the ML-based feedback codes are compatible with a wide range of outer decoders, including soft-decision decoders such as belief propagation [47] and successive cancellation list decoding [49], as well as harddecision decoders like the Berlekamp-Massey algorithm [50]. To validate the effect of the end-to-end training across diverse inner feedback codes, we adopt a differentiable BP decoder as the outer decoder throughout this paper.

## IV. PROPOSED METHOD: MASKCODE

In this section, we propose MASKCODE, a novel Transformer-based inner feedback code for the concatenated coding architecture in Sec. III, where the schematic diagram is depicted in Fig. 4. The proposed method aims to address a fundamental limitation of existing approaches: the inner feedback code is typically designed independently of the outer code’s structure, potentially misallocating feedback resources to error patterns already correctable by the outer ECC. To this end, we incorporate the structural constraint of the outer code $( \mathbf { c H } ^ { \mathrm { T } } = \mathbf { 0 } _ { 1 \times ( n - k ) } )$ into the inner feedback code design via the following two key mechanisms:

1) Soft syndrome-based input design that informs the encoder about potential parity constraint violations.

2) Code-aware masking strategy derived from the Tanner graph.

In the remainder of this section, we introduce the Transformerbased encoding and decoding DNN designs corresponding to encoding (12) and decoding (13) functions in Sec. III.

## A. Inner Encoder: Initial Phase (t = 1)

As depicted in Fig. 4, at the initial phase, the inner encoder generates parity message x<sub>1</sub> from codeword c produced by the outer encoder, as defined in (12).

Embedding. To process c with the Transformer, we first map it into a high-dimensional token sequence. Specifically, the codeword is converted to the BPSK mapping, i.e., $\left( 2 \mathbf { c } - 1 \right)$ to ensure zero-mean symmetric inputs suitable for Transformer layers. The input embedding $\mathbf { E } _ { 1 } ^ { ( e ) } \in \mathbb { R } ^ { \ell \times n }$ is then represented by

$$
\begin{array} { r } { \mathbf { E } _ { 1 } ^ { ( e ) } = \underbrace { ( \mathbf { w } _ { 1 } ^ { ( e ) } ) ^ { \mathrm { T } } ( 2 \mathbf { c } - 1 ) } _ { \mathrm { I n p u t ~ E m b e d d i n g } } + \underbrace { \mathbf { P } _ { 1 } ^ { ( e ) } } _ { \mathrm { P o s i t i o n a l ~ E m b e d d i n g } } , } \end{array}\tag{14}
$$

where ℓ is the embedding dimension, and $\mathbf { w } _ { 1 } ^ { ( e ) } ~ \in ~ \mathbb { R } ^ { 1 \times \ell }$ and $\mathbf { P } _ { 1 } ^ { ( e ) } \in \mathbb { R } ^ { \ell \times n }$ are learnable input embedding vector and positional embedding matrix, respectively.

Parity message $\mathbf { x } _ { 1 }$ . Since no feedback has been received at this phase, the encoder generates $\mathbf { x } _ { 1 }$ without applying a code-

![](images/21dac4a12ba33ebe63de18e1bd3f20587ab5ea2f54b66d2dd5cb40690855c175.jpg)  
Fig. 4. Overall schematic diagram of encoder/decoder of MASKCODE. (Left) The inner encoder at initial phase (t = 1) processes the bipolar-mapped codeword using a standard Transformer. (Middle) The inner encoder at subsequent phases $\left( t \geq 2 \right)$ utilizes the code-aware mask derived from the parity check matrix H and the soft syndrome-based input to refine parity messages based on noisy feedback. (Right) Similar to the inner encoder, the inner decoder aims to produce reliable LLRs to the outer decoder by aggregating received parity messages, as well as the soft syndrome-based input corresponding to the initial-phase parity message.

aware mask. Let the output of the inner encoder’s Transformer be ${ \bf o } _ { 1 } ^ { ( e ) } \in \mathbb { R } ^ { 1 \times n }$ , i.e.,

$$
\mathbf { o } _ { 1 } ^ { ( e ) } = \mathrm { T r a n s } ^ { ( e ) } ( \mathbf { E } _ { 1 } ^ { ( e ) } ; \mathbf { O } _ { n } , \phi _ { 1 } ^ { ( e ) } ) ,\tag{15}
$$

where $\mathbf { O } _ { n } \in \mathbb { R } ^ { n \times n }$ and $\phi _ { 1 } ^ { ( e ) }$ denote the all-zero mask with size of n and learnable parameters of the initial-phase encoder, respectively. The parity message $\mathbf { x } _ { 1 }$ is obtained by

$$
\mathbf { x } _ { 1 } = \mathsf { a b s } ( \mathbf { o } _ { 1 } ^ { ( e ) } ) \odot ( 2 \mathbf { c } - 1 ) ,\tag{16}
$$

where $\odot$ denotes the Hadamard product. The key design rationale behind (16) is to explicitly preserve the sign of the BPSK-encoded codeword $( 2 \mathbf { c } \mathrm { ~ - ~ } 1 )$ in the transmitted signal, thereby enabling the soft syndrome to be computed. More importantly, this sign-fixing constraint is applied only at $t ~ = ~ 1$ . Extending this constraint to subsequent phases would force $\mathbf { x } _ { t \geq 2 }$ to share the sign of (2c  1), restricting the parity messages to amplitude-only modulation and degrading performance. Hence, $\tilde { \mathbf { y } } _ { 1 }$ serves as a fixed structural reference for the soft syndrome across all phases $t \geq 2$

## B. Inner Encoder: Subsequent Phases (t  2)

Unlike the first phase, the encoder in subsequent phases can utilize the feedback history $( \tilde { \mathbf { y } } _ { 1 } , \tilde { \mathbf { y } } _ { 2 } , \ldots , \tilde { \mathbf { y } } _ { t - 1 } )$ , which enables two core mechanisms of MASKCODE: i) a soft-syndromebased input that identifies which parity constraints of the outer ECC are likely violated, and ii) a code-aware attention mask that restricts inter-bit interactions to follow the sparse structure of the Tanner graph. Together, these mechanisms allow the encoder to focus feedback resources on structurally important bits, rather than treating all bits uniformly.

Encoder input: code history. As shown in Fig. 4, the first component of the encoder input aggregates the codeword c and all previously received feedback signals:

$$
\mathbf { E } _ { t } ^ { ( e , c ) } = \left[ \begin{array} { c } { 2 \mathbf { c } - 1 } \\ { \tilde { \mathbf { y } } _ { 1 } } \\ { \vdots } \\ { \tilde { \mathbf { y } } _ { t - 1 } } \end{array} \right] \in \mathbb { R } ^ { t \times n } ,\tag{17}
$$

This formulation follows the standard input design adopted in [19], [21], [22].

Encoder input: soft syndrome. The second component is the key novel input of MASKCODE. While existing feedback coding schemes rely solely on (17), we introduce an additional input, the soft syndrome $\mathbf { \sigma } _ { \mathbf { S } } ^ { ( e ) } \in \mathbb { R } ^ { 1 \times ( n - k ) }$ , which explicitly informs the encoder about which parity constraints of the outer ECC are likely being violated. To obtain $\mathbf { s } ^ { ( e ) }$ , we first derive the LLRs of $\tilde { \mathbf { y } } _ { 1 }$ by

$$
\begin{array} { r l } & { \tilde { r } _ { i } = \log \left( \frac { \operatorname* { P r } \left( c _ { i } = 1 | \tilde { y } _ { 1 , i } \right) } { \operatorname* { P r } \left( c _ { i } = 0 | \tilde { y } _ { 1 , i } \right) } \right) } \\ & { \quad = \log \left( \frac { 1 + \mathrm { e r f } \left( \frac { \tilde { y } _ { 1 , i } } { \sqrt { 2 } \sigma _ { \mathrm { e f f } } } \right) } { 1 - \mathrm { e r f } \left( \frac { \tilde { y } _ { 1 , i } } { \sqrt { 2 } \sigma _ { \mathrm { e f f } } } \right) } \right) , \forall i \in [ n ] , } \end{array}\tag{18}
$$

where $\sigma _ { \mathrm { e f f } } ^ { 2 } = \sigma _ { \mathrm { f f } } ^ { 2 } + \sigma _ { \mathrm { f b } } ^ { 2 }$ and e $\begin{array} { r } { \mathrm { r f } ( x ) = \frac { 2 } { \sqrt { \pi } } \int _ { 0 } ^ { x } \exp ( - t ^ { 2 } ) d t } \end{array}$ . Following the variable-to-check node update in the BP algorithm in (3), the soft syndrome $\mathbf { s } ^ { ( e ) } \in \mathbb { R } ^ { 1 \times ( n - k ) }$ is then

$$
s _ { i } ^ { ( e ) } = 2 \mathrm { a r c t a n h } \left( \prod _ { b \in N _ { \mathrm { v } } ( i ) } \operatorname { t a n h } \left( \frac { \tilde { r } _ { b } } { 2 } \right) \right) , \forall i \in [ n - k ] ,\tag{19}
$$

where $N _ { \mathrm { v } } ( i ) ~ = ~ \{ b ~ \in ~ [ n ] ~ | ~ [ \mathbf { H } ] _ { i , b } ~ = ~ 1 \}$ . Here, $s _ { i } ^ { ( e ) } \ll 0$ indicates a probable violation of the i-th parity constraint, providing the encoder with structural information to focus feedback resources on the most vulnerable bits.

Embedding. By aggregating the code and syndrome parts of the encoder’s input, we have the embedded input of the Transformer as

$$
\begin{array} { r } { \mathbf { E } _ { t } ^ { ( e ) } = \left[ \begin{array} { l } { \mathbf { W } _ { t } ^ { ( e , c ) } \mathbf { E } _ { t } ^ { ( e , c ) } \enspace \middle | \enspace ( \mathbf { w } _ { t } ^ { ( e , s ) } ) ^ { \mathrm { T } } \mathbf { s } ^ { ( e ) } } \end{array} \right] } \\ { + \mathbf { P } _ { t } ^ { ( e ) } \in \mathbb { R } ^ { \ell \times ( 2 n - k ) } , } \end{array}\tag{20}
$$

where $\mathbf { W } _ { t } ^ { ( e , c ) } ~ \in ~ \mathbb { R } ^ { \ell \times t }$ and $\mathbf { w } _ { t } ^ { ( e , s ) } \ \in \ \mathbb { R } ^ { 1 \times \ell }$ are learnable embedding matrix for code and learnable embedding vector for soft syndrome, respectively, and ${ \bf P } _ { t } ^ { ( e ) } \in \mathbb { R } ^ { \ell \times ( 2 n - k ) }$ denotes the positional embedding matrix.

Code-aware attention mask. The standard all-zero mask allows unrestricted interactions between all n bits, which is agnostic to the code structure. MASKCODE instead enforces a sparse, structure-aware attention pattern derived from the Tanner graph, compelling the Transformer to respect the paritycheck constraints $\mathbf { c H ^ { \mathrm { T } } = 0 }$ during encoding. Tanner graphbased masking has been widely adopted in Transformer-based ECC decoders [4], [36], [51], [52], demonstrating that aligning attention patterns with code structure significantly enhances decoding performance. However, to the best of our knowledge, this principle has never been applied to the encoding side— specifically, to an inner feedback encoder that is explicitly aware of the outer ECC’s structural constraints within the concatenated coding.

Following [4], the code-aware attention mask $\mathbf { M } : = g ( \mathbf { H } )$ is derived from the parity-check matrix H, where $\textrm { \textit { g } } :$ $\{ 0 , 1 \} ^ { ( n - k ) \times n } \ \to \ \{ - \infty , 0 \} ^ { \bar { ( } 2 n - k ) \times ( 2 n - k ) }$ . The code-aware attention mask enforces the Transformer to respect the code’s structural constraints $\mathbf { c H ^ { \mathrm { T } } \Sigma = \ l }$ during the attention computation. The mask is constructed from a binary adjacency matrix $\mathbf { A } \in \{ 0 , 1 \} ^ { ( 2 n - k ) \times ( 2 n - k ) }$ <sup>)</sup>, defined as the sum of two components.

Inter-node connectivity $( \mathbf { A } _ { \mathrm { i n t e r } } ) { : }$ The conventional bipartite adjacency matrix [53], [54] represents the explicit connections between variable nodes and check nodes by

$$
\mathbf { A } _ { \mathrm { i n t e r } } = \left[ { \frac { \mathbf { O } _ { n } \mathbf { \Sigma } \left| \mathbf { \Sigma } \mathbf { H } ^ { \mathrm { T } } \right. } { \mathbf { H } \mathbf { \Sigma } \left. \mathbf { O } _ { n - k } \right. } } \right] .\tag{21}
$$

Intra-node connectivity $( \mathbf { A _ { i n t r a } } ) \colon \mathrm { A }$ block-diagonal matrix represents the induced connections within each node set by

$$
\mathbf { A } _ { \mathrm { i n t r a } } = \left[ \frac { \mathbb { 1 } ( \mathbf { H } ^ { \mathrm { T } } \mathbf { H } > 0 ) \mathbf { \Lambda } \left| \mathbf { 0 } _ { n \times ( n - k ) } \right. } { \mathbf { O } _ { ( n - k ) \times n } } \right] ,\tag{22}
$$

where two variable nodes are connected if they share at least one common check node.

Summing these two components $( \mathbf { A } _ { \mathrm { i n t r a } }$ and $\mathbf { A } _ { \mathrm { i n t e r } } )$ yields the final augmented adjacency matrix

$$
\mathbf { A } = \mathbf { A } _ { \mathrm { i n t r a } } + \mathbf { A } _ { \mathrm { i n t e r } } = \left[ { \frac { \mathbb { 1 } \left( \mathbf { H } ^ { \mathrm { T } } \mathbf { H } > 0 \right) \ \left| \mathbf { \nabla } \mathbf { H } ^ { \mathrm { T } } \right. } { \mathbf { H } } } \right] ,\tag{23}
$$

which integrates the Gram matrix-derived intra-node connectivity for variable nodes and the inter-node connectivity from the fundamental bipartite structure of ECC. Finally, this binary matrix A is converted into the final attention mask M via an element-wise logarithm, which is then added to the attention scores just before the softmax operation according to

$$
[ \mathbf { M } ] _ { i j } : = g ( \mathbf { H } ) _ { i j } = \log _ { 2 } ( \mathbf { A } ) _ { i j } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } [ \mathbf { A } ] _ { i j } = 1 , } \\ { - \infty , } & { \mathrm { i f ~ } [ \mathbf { A } ] _ { i j } = 0 . } \end{array} \right.\tag{24}
$$

Parity message $\mathbf { x } _ { t \geq 2 } .$ . The Transformer output at the t-th phase is given by

$$
\mathbf { o } _ { t } ^ { ( e ) } = \mathrm { T r a n s } ^ { ( e ) } ( \mathbf { E } _ { t } ^ { ( e ) } ; \mathbf { M } , \phi _ { t } ^ { ( e ) } ) ,\tag{25}
$$

where $\phi _ { t } ^ { ( e ) }$ denotes the learnable parameters. Since $\mathbf { o } _ { t } ^ { ( e ) }$ contains representations for both variable nodes and check nodes, we extract only the first n elements corresponding to the variable nodes:

$$
\mathbf { x } _ { t } = ( \mathbf { o } _ { t } ^ { ( e ) } ) _ { 1 : n } ,\tag{26}
$$

where $( \mathbf { o } _ { t } ^ { ( e ) } ) _ { 1 : n }$ denotes extracting the first n elements of $\mathbf { o } _ { t } ^ { ( e ) }$

## C. Inner Decoder

The inner decoder mirrors the architecture of the subsequent encoder, employing the same code-aware attention mask M and soft syndrome-based input design. After T transmission phases, the receiver aggregates parity messages $( \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \ldots , \mathbf { y } _ { T } )$ . Then, the decoder leverages all the parity messages to estimate the original codeword c.

Input embedding. Similar to embedding in the subsequent encoder, the code and syndrome parts of the decoder’s input are represented by

Code part: Similar to (17), the code part of the decoder’s input is written as

$$
\mathbf { E } ^ { ( d , c ) } = \binom { \mathbf { y } _ { 1 } } { \vdots } \in \mathbb { R } ^ { T \times n } .\tag{27}
$$

Syndrome part: The soft syndrome of the decoder is computed analogously to (19), but using $\mathbf { y } _ { 1 }$ with noise variance $\sigma _ { \mathrm { f f } } ^ { 2 } \mathrm { . }$

$$
\mathbf { s } ^ { ( d ) } \in \mathbb { R } ^ { 1 \times ( n - k ) } ,\tag{28}
$$

where $s _ { i } ^ { ( d ) } ~ =$ 2arctanh $\begin{array} { r } { \left( \prod _ { b \in N _ { \mathrm { v } } ( i ) } \operatorname { t a n h } \left( \frac { r _ { b } } { 2 } \right) \right) } \end{array}$ and $r _ { i } =$ $\log \left( \frac { 1 + \mathrm { e r f } \left( \frac { y _ { 1 , i } } { \sqrt { 2 } \sigma _ { \mathrm { f f } } } \right) } { 1 - \mathrm { e r f } \left( \frac { y _ { 1 , i } } { \sqrt { 2 } \sigma _ { \mathrm { f f } } } \right) } \right)$ . Note that $\sigma _ { \mathrm { f f } } ^ { 2 }$ is used here instead of $\sigma _ { \mathrm { e f f } } ^ { 2 } ,$ since the decoder observes $\mathbf { y } _ { 1 }$ directly without additional feedback noise.

Then, we have the embedded input for the decoder as

$$
\begin{array} { r l } & { \mathbf { E } ^ { ( d ) } = \left[ \begin{array} { l } { \mathbf { W } ^ { ( d , c ) } \mathbf { E } ^ { ( d , c ) } \mid ( \mathbf { w } ^ { ( d , s ) } ) ^ { \mathrm { T } } \mathbf { s } ^ { ( d ) } } \end{array} \right] } \\ & { \qquad + \mathbf { P } ^ { ( d ) } \in \mathbb { R } ^ { \ell \times ( 2 n - k ) } , } \end{array}\tag{29}
$$

where $\mathbf { W } ^ { ( d , c ) } \in \mathbb { R } ^ { \ell \times T }$ and $\mathbf { w } ^ { ( d , s ) } \in \mathbb { R } ^ { 1 \times \ell }$ denote the learnable embedding matrix for the code and learnable embedding vector for soft syndrome, respectively, and $\mathbf { P } ^ { ( d ) } \in \mathbb { R } ^ { \ell \times ( 2 n - k ) }$ denotes the positional embedding matrix.

Decoding. Let the output of the Transformer block be $\mathbf { o } ^ { ( d ) } \in$ $\mathbb { R } ^ { 1 \times ( 2 n - \overline { { k } } ) }$ , i.e.,

$$
\mathbf { o } ^ { ( d ) } = \mathrm { T r a n s } ^ { ( d ) } ( \mathbf { E } ^ { ( d ) } ; \mathbf { M } , \phi ^ { ( d ) } ) ,\tag{30}
$$

where $\phi ^ { ( d ) }$ denotes the learnable parameters of the decoder. Analogous to (26), we extract the variable node outputs as the estimated codeword:

$$
\widehat { \mathbf { c } } = \bigl ( \mathbf { o } ^ { ( d ) } \bigr ) _ { 1 : n } ,\tag{31}
$$

where each element $\widehat { c } _ { i }$ corresponds to the LLR of the transmitted bit $c _ { i } { \mathrm { : } }$

$$
{ \boldsymbol { \widehat { c } } } _ { i } = \log \left( { \frac { \operatorname* { P r } ( c _ { i } = 1 | \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \dots , \mathbf { y } _ { T } ) } { \operatorname* { P r } ( c _ { i } = 0 | \mathbf { y } _ { 1 } , \mathbf { y } _ { 2 } , \dots , \mathbf { y } _ { T } ) } } \right) .\tag{32}
$$

## D. Implementation and Loss Function

MASKCODE can be trained end-to-end by minimizing a BCE loss that jointly considers the inner encoder, inner decoder, and the outer BP decoder. Specifically, the inner decoder output c is passed through $N _ { \mathrm { B P } }$ BP iterations to yield $\widehat { \mathbf { c } } _ { \mathrm { B P } } = \mathtt { B P } ( \widehat { \mathbf { c } } ; N _ { \mathtt { B P } } )$ , where $N _ { \mathrm { B P } } = 0$ reduces to inner-code-only btraining.

For notational convenience, we aggregate the trainable parameters as follows:

$\pmb { \theta } _ { 1 } ^ { ( e ) }$ : Trainable weights of the initial-phase encoder:

$$
\begin{array} { r } { \pmb { \theta } _ { 1 } ^ { ( e ) } = \{ \mathbf { w } _ { 1 } ^ { ( e ) } , \mathbf { P } _ { 1 } ^ { ( e ) } , \phi _ { 1 } ^ { ( e ) } \} . } \end{array}\tag{33}
$$

$\pmb { \theta } _ { t \geq 2 } ^ { ( e ) } ;$ : Trainable weights of the subsequent-phase encoder:

$$
\mathbf { \theta } \mathbf { \theta } \mathbf { \theta } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \theta } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda \Lambda } \mathbf { \Lambda \Lambda } \mathbf \Lambda \mathbf { \Lambda } \Lambda \mathbf { \Lambda \Lambda } \mathbf \Lambda \Lambda \Lambda \mathbf { \Lambda } \Lambda \mathbf \Lambda \Lambda \Lambda \Lambda \Lambda \mathbf { \Lambda } \Lambda \Lambda \Lambda \mathbf \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \mathbf \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda \Lambda\tag{34}
$$

$\pmb \theta ^ { ( d ) }$ : Trainable weights of the decoder:

$$
\begin{array} { r } { \pmb { \theta } ^ { ( d ) } = \{ \mathbf { W } ^ { ( d , c ) } , \mathbf { w } ^ { ( d , s ) } , \mathbf { P } ^ { ( d ) } , \phi ^ { ( d ) } \} . } \end{array}\tag{35}
$$

The end-to-end loss function is defined as

$$
\begin{array} { l } { { \displaystyle \mathrm { L o s s } ( \pmb { \theta } _ { 1 } ^ { ( e ) } , \dots , \pmb { \theta } _ { T } ^ { ( e ) } , \pmb { \theta } ^ { ( d ) } ) } \ ~ ( 3 6 ) } \  \\ { { \displaystyle = - \mathbb { E } \left[ \sum _ { i \in [ n ] } c _ { i } \log ( \mathbf { s i g } ( \widehat c _ { \mathrm { B P } , i } ) ) + ( 1 - c _ { i } ) \log ( 1 - \mathbf { s i g } ( \widehat c _ { \mathrm { B P } , i } ) ) \right] } \ ~ } \\ { { \displaystyle ~ \qquad \ } . } \end{array}
$$

where $\mathsf { s i g } ( \cdot )$ denotes the sigmoid function and ${ \widehat { c } } _ { \mathrm { B P } , i }$ is the i-th element of $\widehat { \mathbf { c } } _ { \mathrm { B P } }$ b. The full training procedure is summarized in bAlgorithm 1.

While $N _ { \mathrm { B P } } > 0$ allows the inner code to account for the BP decoder’s behavior, we empirically observe that $N _ { \mathrm { B P } } = 0$ achieves the best performance (see Fig. 7). We attribute this to the fact that MASKCODE already incorporates, via the code-aware attention mask and soft syndrome-based input, the structural constraint of the outer ECC that end-to-end training attempts to instill. Consequently, the gradient instability ,introduced by unrolling BP iterations only degrades training without offering additional gain.

Algorithm 1: Training Algorithm of MASKCODE   
1 Input Training parameters   
2 Total training step: $T _ { \mathrm { t r a i n } } ,$ batch size B, feedforward   
noise scale $\sigma _ { \mathrm { f f } } ,$ , feedback noise scale $\sigma _ { \mathrm { f b } }$ , and   
learning rate $\gamma .$   
// Initialize neural network weights   
3 Initialize Neural network weights   
4 Encoder weights: $\pmb { \theta } _ { 1 } ^ { ( e ) } , \pmb { \theta } _ { 2 } ^ { ( e ) } , \dots , \pmb { \theta } _ { T } ^ { ( e ) } ,$   
5 Decoder weights: $\pmb { \theta } ^ { ( d ) }$ . Mask M by Equation (24)   
6 for $\{ 1 , . . . , T _ { \mathrm { t r a i n } } \}$ do   
7 for b in $\{ 1 , \ldots , B \}$ do in parallel   
8 Draw m uniformly at random from $\{ 0 , 1 \} ^ { 1 \times k }$   
9 c mG   
// Initial Phase   
10 Get embedded input ${ \bf E } _ { 1 } ^ { ( e ) }$ by (14)   
11 $\mathbf { o } _ { 1 } ^ { ( e ) }  \mathrm { T r a n s } ^ { ( e ) } ( \mathbf { E } _ { 1 } ^ { ( e ) } ; \bar { \mathbf { O } } _ { n } , \phi _ { 1 } ^ { ( e ) } ) .$   
12 $\mathbf { x } _ { 1 }  \mathbf { a b s } ( \mathbf { o } _ { 1 } ^ { ( e ) } ) \odot ( 2 \mathbf { c } - 1 ) .$   
13 ${ \bf y } _ { 1 } = { \bf x } _ { 1 } + { \bf n } _ { 1 }$ , where $\mathbf { n } _ { 1 } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { f f } } ^ { 2 } \mathbf { I } _ { n } )$   
$/ /$ Subsequent Phases   
14 for t in $\{ 2 , \ldots , T \}$ do   
15 $\tilde { \mathbf { y } } _ { t - 1 } = \mathbf { y } _ { t - 1 } + \mathbf { z } _ { t - 1 } ,$ where   
$\mathbf z _ { t - 1 } \sim \mathcal N ( 0 , \sigma _ { \mathrm { f b } } ^ { 2 } \mathbf I _ { n } )$   
16 Get embedded input $\mathbf { E } _ { t } ^ { ( e ) }$ by (20).   
17 $\mathbf { o } _ { t } ^ { ( e ) } \gets \mathrm { T r a n s } ^ { ( e ) } ( \mathbf { E } _ { t } ^ { ( e ) } ; \mathbf { M } , \phi _ { t } ^ { ( e ) } )$   
18 $\mathbf { x } _ { t } = ( \mathbf { o } _ { t } ^ { ( e ) } ) _ { 1 : n } .$   
19 $\mathbf { y } _ { t } = \mathbf { x } _ { t } + \mathbf { n } _ { t } .$ , where $\mathbf { n } _ { t } \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { f f } } ^ { 2 } \mathbf { I } _ { n } )$   
$/ /$ Decoding   
20 Get embedded input $\mathbf { E } ^ { ( d ) }$ by (29).   
21 $\mathbf { o } ^ { ( d ) }  \mathrm { T r a n s } ^ { ( d ) } ( \mathbf { \bar { E } } ^ { ( d ) } ; \mathbf { M } , \boldsymbol { \phi } ^ { ( d ) } )$   
22 $\widehat { \mathbf { c } } = \bigl ( \mathbf { o } ^ { ( d ) } \bigr ) _ { 1 : n } .$   
23 $\widehat { \mathbf { c } } _ { \mathrm { B P } } \gets \mathsf { B P } ( \widehat { \mathbf { c } } ; N _ { \mathrm { B P } } ) .$   
24 bLossb $\begin{array} { r } { ,   - \sum _ { i \in [ n ] } ( c _ { i } \log ( \tt s i g ( \widehat { c } _ { B P , } { } _ { i } ) )  } \end{array}$   
$+ ( 1 - c _ { i } ) \log ( \dot { 1 } - \mathbf { s } \mathbf { i g } ( \widehat { c } _ { \mathsf { B P } , i } ) ) \big ) .$   
25 Loss $( \pmb { \theta } _ { 1 } ^ { ( e ) } , \dots , \pmb { \theta } _ { T } ^ { ( e ) } , \pmb { \theta } ^ { ( d ) } ) =$   
$\begin{array} { r } { \frac { 1 } { B } \sum _ { b \in [ B ] } \mathrm { L o s s } _ { b } \big ( \pmb { \theta } _ { 1 } ^ { ( e ) } , \dots , \pmb { \theta } _ { T } ^ { ( e ) } , \pmb { \theta } ^ { ( d ) } \big ) } \end{array}$   
// Update Parameters   
26 $\begin{array} { r } { \pmb { \theta } _ { t } ^ { ( e ) }  \pmb { \theta } _ { t } ^ { ( e ) } - \gamma \frac { \partial \mathrm { L o s s } } { \partial \pmb { \theta } _ { t } ^ { ( e ) } } } \end{array}$ for all $t \in [ T ] .$   
27 $\begin{array} { r } { \pmb { \theta } ^ { ( d ) }  \pmb { \theta } ^ { ( d ) } - \gamma \frac { \partial \mathrm { L } ^ { \mathrm { \underline { { \imath } } } } \mathrm { s s } } { \partial \pmb { \theta } ^ { ( d ) } } , } \end{array}$

Remark 1 (Gradient instability in end-to-end BP training). Backpropagation through unrolled BP iterations involves repeated differentiation of the tanh and arctanh operations. Denoting the output of the b-th BP iteration as ${ \dot { \boldsymbol g } } ^ { ( b ) }$ with $g ^ { ( 0 ) } = { \widehat { \mathbf { c } } } ,$ the gradient of the loss with respect to c is given by bthe chain rule:

$$
\frac { \partial \widehat { \mathbf { c } } _ { B P } } { \partial \widehat { \mathbf { c } } } = \mathbf { J } ^ { ( N _ { B P } ) } \mathbf { J } ^ { ( N _ { B P } - 1 ) } \cdot \cdot \cdot \mathbf { J } ^ { ( 1 ) } ,\tag{37}
$$

![](images/240c483a3ce89688934145a4e8fe225460166eb16329d6af0d40855fa37351f1.jpg)  
Fig. 5. Stabilized arctanh function and Taylor approximation of the arctanh function. The order of Taylor approximation is denoted by q.

where $\begin{array} { r } { \mathbf { J } ^ { ( b ) } = \frac { \partial g ^ { ( b ) } } { \partial q ^ { ( b - 1 ) } } } \end{array}$ denotes the Jacobian of the b-th BP iteration. As BP iterations progress, the magnitude of LLRs tends to grow, driving the inputs to tanh toward saturation $( | \mu _ { \nu c } ^ { k \to j } | ~ \to ~ \infty )$ and consequently pushing the inputs to arctanh toward 1. In this regime, $\begin{array} { r l r } { \frac { \partial } { \partial x } \mathrm { a r c t a n h } ( x ) } & { { } = } & { } \end{array}$ $\textstyle { \frac { 1 } { 1 - x ^ { 2 } } } \to$ as $x ~  ~ \pm 1$ , causing the Jacobians $\textbf { J } ^ { ( b ) } { } _ { t o }$ explode, as also mentioned in [55], [56]. As this effect compounds across $N _ { B P }$ iterations, $\mathbf { J } ^ { ( N _ { B P } ) } \ldots \mathbf { J } ^ { ( 1 ) }$ becomes increasingly ill-conditioned, making stable gradient-based optimization infeasible for large N<sub>BP</sub>. To address this instability, prior works [55], [56] approximate arctanh with a highorder Taylor polynomial, where the order needs to be higher than 1,000 to obtain a reasonable approximation, incurring substantial computational overhead per iteration. Instead, we adopt a simple stabilizing strategy, arctanh $( ( 1 - 2 \epsilon ) x )$ , which bounds the output without requiring any iteration. As shown in Fig. 5, the stabilized function closely matches the exact arctanh over $| x | < 1 .$ . Combined with this stabilization, the end-to-end training of DeepCode [19], GBAF [21], and AttentionCode [22] shows the best performance with $N _ { B P } = 2$ or 4 iterations, which is comparable to the typical setting of $N _ { B P } = 5$ used in [55], [56], as will be shown in Fig. 7. In our numerical results, we use $\epsilon = 1 0 ^ { - 7 }$ , chosen via grid search over $\epsilon \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 } , 1 0 ^ { - 6 } , 1 0 ^ { - 7 } , 1 0 ^ { - 8 } , 1 0 ^ { - 9 } \}$

## V. EXPERIMENTAL RESULTS

In this section, we numerically evaluate MASKCODE in a feedback-enabled communication scenario, by varying the additive noise power on the feedforward and feedback channels. We begin by analyzing the effect of the end-to-end training with differentiable BP iterations (Sec. IV-D). We then compare MASKCODE with existing feedback coding baselines for two outer codes: 1) BCH(31,16) and 2) LDPC(49,24), which follow the implementation of previous studies [4], [37]. Subsequently, an ablation study validates the contribution of each core feature. Finally, we evaluate robustness under various feedback noise levels and compare computational complexity.

## A. Experimental Setup

For the experiments, we consider a feedback-enabled memoryless AWGN channel, following the standard setup adopted in [4], [19]–[23]. For the outer codes, we use the BP decoder (in Sec. II-A) as the outer decoder. The attention masks for the outer codes are illustrated in Fig. 6.

![](images/b9457e377b1a6664f9358038ce851dec71953bce5fb1994c1bfbd7ec737c0534.jpg)  
(a) BCH(31,16) Code.

![](images/0113dd6f0bd64a019e5adcaff7638f529b175f695abe8aa31265eb7e1cc49be2.jpg)  
(b) LDPC(49,24) Code.  
Fig. 6. Illustrations of the mask design by (24) for (a): BCH(31,16) outer code and (b): LDPC(49,24) outer code.

Setup for channels. The number of closed-loop iterations is three $( \mathrm { i . e . , ~ } T ~ = ~ 3 )$ . We note that the power is normalized to $P ~ = ~ 1$ . The feedforward and feedback channel SNRs are defined as $1 / \sigma _ { \mathrm { f f } } ^ { 2 }$ and $1 / \sigma _ { \mathrm { f b } } ^ { 2 }$ , respectively, following the implementations of previous studies [19], [21]–[23]. Without any specification, the feedback SNR is assumed to be 20 dB, i.e., $\sigma _ { \mathrm { f b } } ^ { 2 } = 0 . 0 1$ . The feedforward SNR varies from -5 dB to -1 dB for the BCH(31,16) outer code and from -5 dB to -2 dB for the LDPC(49,24) code.

Training of MASKCODE. We use the AdamW optimizer, which extends the Adam optimizer with decoupled weight decay for improved generalization. For the learning rate scheduling, we use CosineAnnealingLR, where the learning rate gradually decreases from $1 0 ^ { - 3 } \mathrm { t o } 3 \times 1 0 ^ { - 7 }$ . The model is trained for 120,101 gradient steps, where the batch size is 2,048. Unless otherwise specified, MASKCODE uses two Transformer layers.

Baselines. For comparison, we will consider the following baseline schemes:

Repetition code: A baseline method where the transmitter repeatedly sends the 2c  1 for T phases, which is denoted by Repeat.

SK scheme [11], [12]: SK scheme, denoted by SKCode, is a classical linear feedback coding scheme, which assumes noiseless feedback channels.

CL scheme [18]: CL scheme, denoted by CLCode, is an optimal linear feedback scheme for noisy feedback channels.

Deepcode [19]: DeepCode is a deep learning-based feedback coding scheme, which models the transmitter and receiver as RNNs.

GBAF [21]: GBAF scheme is a sub-block-based feedback coding scheme, where it uses an attention mechanism to generate parity messages.

Attentioncode [22]: AttentionCode is a bit-based feedback coding scheme that employs a Transformer architecture.

Robustcode [20]: RobustCode is an RNN-based scheme. This scheme utilizes $n T$ phases, because a single scalar value is transmitted per phase.

![](images/a4f51f2b38420ae14000c9dc6c6aa6cadc95ab8d44b575aa0ff28f439771b5b1.jpg)  
Fig. 7. BLER evaluation of MASKCODE and baselines under various numbers of BP iterations in the training phase. The forward and feedback SNRs are assumed to be -2 dB and 20 dB, respectively. The outer code is the BCH(31,16) code.

Syndromecode [57]: SyndromeCode is our previous work, a Transformer-based feedback coding scheme that incorporates a soft syndrome-based input design. Unlike MASKCODE, it does not employ a code-aware attention mask; hence, it corresponds to the ablation configuration with soft syndrome input but without masking (Tabs. I and II).

We note that all baselines are trained from scratch for each feedforward SNR setting, and neither SNR mismatch nor curriculum learning [21], [22] is adopted for fair comparison.

## B. End-to-End Training with Differentiable Outer Decoder

As we discussed in Sec. IV-D, MASKCODE supports endto-end training by incorporating the differentiable BP decoder into the training loop, where the number of BP iterations is $N _ { \mathrm { B P } } .$ . Although we stabilize the gradient computation in the BP decoder by stabilizing the arctanh function, the backpropagation through BP iterations still causes gradient explosion (see Remark 1), disrupting training convergence. To ensure stable training, we discard gradient updates whenever gradient explosion is detected, i.e., when the gradient norm is infinite. Figure 7 illustrates the BLER of various models with different $N _ { \mathrm { B P } }$ , with the test phase fixed at 20 iterations for all cases. Although previous studies typically use 50 BP iterations for evaluation [4], [37], 20 BP iterations are sufficient in our experiments, as shown in Fig. 9.

For the baseline ML-based schemes such as Attention-Code [22], GBAF [21], and DeepCode [19], incorporating a small number of BP iterations (e.g., 2 or 4) during training yields a noticeable BLER improvement over the $N _ { \mathrm { B P } } = 0 \mathrm { c a s e }$ suggesting that end-to-end training helps structurally agnostic models adapt to the outer code. However, as $N _ { \mathrm { B P } }$ increases, performance degrades significantly due to gradient instability.

In contrast, MASKCODE achieves superior performance at $N _ { \mathrm { B P } } ~ = ~ 0$ , without any end-to-end $\mathrm { B P }$ training. Unlike the baselines that rely on the training process to internalize the outer code’s structure, MASKCODE explicitly incorporates the outer ECC’s structure via its soft syndrome-based input and code-aware attention mask—the structural constraint of the outer code that end-to-end training aims to incorporate. Consequently, adding BP iterations to MASKCODE’s training only introduces gradient instability, leading to performance degradation. These results demonstrate that an outer codeaware architecture is more robust and effective than relying on end-to-end training. Hence, in the remainder of this paper, we report results with $N _ { \mathrm { B P } } = 0$

![](images/000c29d2ba435d992798de14a71b907ad24d835936ee39fd0bddf889ee140dd3.jpg)  
(a) BCH(31,16) Outer Code. The effective code rate is n/kT ≈ 0.17

![](images/2caf92c0e678dd5250ec57dd77a77d8cfdb18656e7a0ff0cb3438d2c54fbcc54.jpg)  
(b) LDPC(49,24) Outer Code. The effective code rate is n/kT ≈ 0.16  
Fig. 8. BLER performance of MASKCODE and baseline schemes for a 3- phase feedback-assisted concatenated code

## C. BLER Performance under Varying Feedforward SNR

In this subsection, we evaluate MASKCODE by varying the feedforward noise power $\sigma _ { \mathrm { f f } } ^ { 2 }$ for two outer codes: BCH(31,16) and LDPC(49,24), with results shown in Fig. 8. As shown in Fig. 8(a), MASKCODE outperforms all baselines across the entire SNR range, achieving an SNR gain of approximately 1.0 dB over the best-performing baseline at the same target BLER for BCH(31,16). In Fig. 8(b), MASKCODE consistently outperforms the baselines with an SNR gain of approximately 1.5 dB. Notably, this large gain—compared to the 1.0 dB gain observed for BCH(31,16)—is achieved despite the two outer codes having similar effective code rates, suggesting that the gain is driven by the structural properties of the Tanner graph rather than by code rate differences. A sparser Tanner graph means each check node involves fewer variable nodes, so the code-aware mask enforces a more restrictive attention pattern compared to the fully-connected baseline—diverging more significantly from model-agnostic approaches that treat all bits uniformly, and thus yielding a larger performance gain.

TABLE I  
ABLATION STUDY OF THE PROPOSED METHOD. THE OUTER CODE IS BCH(31,16)
<table><tr><td rowspan="2">Methods</td><td colspan="2">Features</td><td colspan="3">BLER</td></tr><tr><td>Mask</td><td>Soft syndrome Input</td><td> $\mathrm { @ { S N R = - 2 d B } }$ </td><td> $\mathrm { @ S N R = } 3 \mathrm { d B }$ </td><td> $\scriptstyle \bigoplus S N R = - 4 \mathrm { d } B$ </td></tr><tr><td rowspan="4">MASKCODE (Ours)</td><td>√</td><td>√</td><td> $\underline { { 2 . 0 6 \times 1 0 ^ { - 9 } } }$ </td><td> $\underline { { 9 . 2 5 \times 1 0 ^ { - 6 } } }$ </td><td> $\underline { { 1 . 7 4 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td>√</td><td>X</td><td> $\overline { { 2 . 1 6 \times 1 0 ^ { - 6 } } }$ </td><td> $\overline { { 3 . 9 7 \times 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 8 . 2 4 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>X</td><td>√</td><td> $8 . 0 4 \times 1 0 ^ { - 6 }$ </td><td> $2 . 6 4 \times 1 0 ^ { - 4 }$ </td><td> $3 . 5 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>X</td><td>X</td><td> $1 . 2 0 \times 1 0 ^ { - 5 }$ </td><td> $1 . 2 8 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 8 \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="4">DeepCode [19] RobustCode [20]</td><td></td><td></td><td> $1 . 9 8 \times 1 0 ^ { - 4 }$ </td><td> $4 . 1 5 \times 1 0 ^ { - 3 }$ </td><td> $4 . 3 9 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>1</td><td> $2 . 5 4 \times 1 0 ^ { - 6 }$ </td><td> $2 . 9 3 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>AttentionCode [21]</td><td></td><td> $8 . 3 9 \times 1 0 ^ { - 5 }$ </td><td> $2 . 8 1 \times 1 0 ^ { - 3 }$ </td><td> $6 . 8 1 \times 1 0 ^ { - 2 }$ </td></tr></table>

\*The best method is marked in bold and underline.  
\*\*The second-best method is marked in bold.

TABLE II  
ABLATION STUDY OF THE PROPOSED METHOD. THE OUTER CODE IS LDPC(49,24)
<table><tr><td rowspan="2">Methods</td><td colspan="2">Features</td><td colspan="3">BLER</td></tr><tr><td>Mask</td><td>Soft syndrome Input</td><td> $\mathrm { @ S N R = - 3 d B }$ </td><td> $\mathrm { @ S N R = - 4 d B }$ </td><td> $\mathrm { @ S N R = - 5 d B }$ </td></tr><tr><td rowspan="4">MASKCODE (Ours)</td><td>√</td><td>√</td><td> $2 . 7 7 \times 1 0 ^ { - 8 }$ </td><td> ${ \underline { { 1 . 3 6 \times 1 0 ^ { - 5 } } } }$ </td><td> ${ \underline { { 4 . 4 8 \times 1 0 ^ { - 4 } } } }$ </td></tr><tr><td>√</td><td>X</td><td> $\overline { { 1 . 5 3 \times 1 0 ^ { - 5 } } }$ </td><td> $6 . 4 1 \times 1 0 ^ { - 4 }$ </td><td> $\overline { { { \bf 4 . 3 9 \times 1 0 ^ { - 3 } } } }$ </td></tr><tr><td>X</td><td>√</td><td> $2 . 2 2 \times 1 0 ^ { - 5 }$ </td><td> $6 . 1 0 \times 1 0 ^ { - 4 }$ </td><td> $6 . 4 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>X</td><td>X</td><td> $2 . 8 1 \times 1 0 ^ { - 5 }$ </td><td> ${ \bf 5 . 0 9 } \times 1 0 ^ { - 4 }$ </td><td> $8 . 4 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>DeepCode [19]</td><td></td><td>1</td><td> $1 . 6 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 8 9 \times 1 0 ^ { - 3 }$ </td><td> $5 . 7 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>RobustCode [20]</td><td>=</td><td></td><td> $3 . 6 4 \times 1 0 ^ { - 5 }$ </td><td> $1 . 2 2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 4 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>AttentionCode [21]</td><td>=</td><td></td><td> $2 . 0 3 \times 1 0 ^ { - 4 }$ </td><td> $7 . 2 0 \times 1 0 ^ { - 3 }$ </td><td> $6 . 1 5 { \times } 1 0 ^ { - 2 }$ </td></tr></table>

\*The best method is marked in bold and underline.  
\*\*The second-best method is marked in bold.

Another notable observation is that GBAF [21], a subblock-based method, exhibits substantial BLER degradation compared to bit-based methods. This is because sub-blockbased methods reinforce dependencies among adjacent bits within each sub-block, rather than across the entire codeword, causing the per-bit LLRs provided to the outer BP decoder to deviate from their true values. Consequently, the outer decoder receives structurally inconsistent LLRs, leading to significant performance degradation in the concatenated coding scenario.

## D. Ablation Study: Code-Aware Mask & Soft Syndrome Input

Tables I and II detail the results of our ablation study, which is designed to deconstruct the performance of our proposed MASKCODE and quantify the individual contribution of its two key features: 1) the code-aware attention mask and 2) the soft syndrome-based input design. In this study, we evaluate and compare the performance of the following four model configurations: 1) conventional Transformer without either features, 2) soft syndrome-based input only, 3) code-aware mask only, and 4) MASKCODE. As shown in the tables, each feature independently yields a substantial improvement over the baseline, and their combination in the full MASKCODE model achieves the best performance, confirming that the two features are complementary and produce a synergistic effect.

![](images/6d613499e5b9d47cd9477e18a81b31816faeecc52d75fe18fb46fdcac24d2758.jpg)  
Fig. 9. BLER performance of MASKCODE for various numbers of outer decoder (BP) iterations. The MASKCODE with few iterations (2 or 3) achieves a comparable BLER performance to that with 20 iterations. The outer code is the BCH(31,16) code.

TABLE III  
BLER PERFORMANCE FOR VARIOUS NUMBERS OF LAYERS IN THE MASKCODE.
<table><tr><td rowspan="2">Outer Code</td><td rowspan="2">Methods</td><td colspan="3">BLER</td></tr><tr><td>@SNR=-3dB</td><td>@SNR=-4dB</td><td>@SNR=-5dB</td></tr><tr><td rowspan="2">BCH(31,16)</td><td>MASKCODE-S</td><td> $9 . 2 5 \times 1 0 ^ { - 6 }$ </td><td> $1 . 7 4 \times 1 0 ^ { - 4 }$ </td><td> $1 . 7 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>MASKCODE-L</td><td> $7 . 0 4 \times 1 0 ^ { - 7 }$ </td><td> $5 . 0 9 \times 1 0 ^ { - 5 }$ </td><td> $1 . 1 0 \times 1 0 ^ { - 3 }$ </td></tr></table>

## E. Various Numbers of Outer Code Iterations

To determine how many BP iterations are necessary for the inference, we evaluate the BLER across different numbers of BP iterations (0 to 20) in Fig. 9. Without any BP iterations, the output relies solely on the inner decoder’s LLRs, yielding a relatively high BLER. A significant improvement is observed at 1 and 2 iterations, after which the performance gap becomes negligible up to 20 iterations. This indicates that only a few BP iterations are sufficient for near-optimal decoding performance in the concatenated system.

## F. Various Feedback Noise Scales

In this subsection, we evaluate the robustness of MASKCODE to varying feedback noise levels in Fig. 10. In the figure, MASKCODE consistently outperforms all baselines across the entire feedback SNR regime for both outer codes. Notably, while the baselines exhibit a performance plateau around a BLER of $1 0 ^ { - 4 }$ , MASKCODE demonstrates a continuous BLER decay, reaching as low as $1 0 ^ { - 9 }$

## G. Various Numbers of Layers

We examine the effect of model depth by comparing two configurations of MASKCODE on the BCH(31,16) outer code:

MASKCODE-S: the standard 2-layer configuration used in all previous results.

MASKCODE-L: a deeper variant of MASKCODE, with 4 layers.

![](images/f98024a2b1d41f8195fb5a694a7e5fa8c27c4264525f11b4cad8c6492166d5f4.jpg)  
(a) BCH(31,16) Outer Code.

![](images/59acbc7f2bc13e5dcd15e18f2781df8a180cb8e856336161b93d1ef962e75753.jpg)  
(b) LDPC(49,24) Outer Code.  
Fig. 10. BLER evaluation of MASKCODE and baselines under various channel feedback noise levels. The forward SNR is -2 dB in (a) and -3 dB in (b).

TABLE IV  
COMPUTATIONAL COMPLEXITY, MULTIPLY-ACCUMULATES (MACS) AND WALL-CLOCK TIME
<table><tr><td>Methods</td><td>Parameters</td><td>MAC</td><td>Codewords/sec</td></tr><tr><td>MASKCODE (Ours)</td><td> $5 . 4 6 \times 1 0 ^ { 5 }$ </td><td> $4 . 0 2 \times 1 0 ^ { 5 }$ </td><td>175.44</td></tr><tr><td>AttentionCode</td><td> $1 . 2 6 \times 1 0 ^ { 5 }$ </td><td> $8 . 4 3 \times 1 0 ^ { 5 }$ </td><td>17.95</td></tr><tr><td>GBAF</td><td> $1 . 5 5 \times 1 0 ^ { 5 }$ </td><td> $4 . 1 3 \times 1 0 ^ { 5 }$ </td><td>65.17</td></tr><tr><td>DeepCode</td><td> $1 . 4 7 \times 1 0 ^ { 5 }$ </td><td> $4 . 5 5 \times 1 0 ^ { 6 }$ </td><td>713.29</td></tr><tr><td>RobustCode</td><td> $1 . 0 2 \times 1 0 ^ { 5 }$ </td><td> $8 . 3 3 \times 1 0 ^ { 6 }$ </td><td>739.13</td></tr></table>

Table III summarizes the BLER performance of the two configurations across varying feedforward SNR values. MASKCODE-L significantly outperforms MASKCODE-S across all SNR values, demonstrating that model depth is an important factor in concatenated coding performance. This finding establishes a clear trade-off between performance and model complexity.

## H. Computational Complexity

Lastly, we evaluate the computational complexity and practical latency of MASKCODE against the baselines, with results presented in Tab. IV. We report three metrics: the number of parameters, multiply-accumulates (MACs), and wall-clock throughput (codewords/sec).

Parameter counts and MAC. The baselines fall into two distinct groups. The RNN-based methods (DeepCode and RobustCode) have a small parameter count but an extremely high MAC count (e.g., $4 . 5 5 \times 1 0 ^ { 6 }$ for DeepCode), an order of magnitude higher than attention-based methods.

Throughput. Despite its larger parameter count, MASKCODE achieves the lowest MAC count among Transformer-based methods $( 4 . 0 2 ~ \times ~ 1 0 ^ { 5 } )$ , comparable to GBAF $( 4 . 1 3 \times 1 0 ^ { 5 } )$ and significantly lower than AttentionCode $( 8 . 4 3 \times 1 0 ^ { 5 } )$ . More importantly, MASKCODE achieves the highest throughput among Transformer-based methods at 175.44 codewords/sec, approximately 2.7 times faster than GBAF and nearly 10 times faster than AttentionCode.

## VI. CONCLUSION

In this paper, we propose MASKCODE, a novel Transformer-based inner feedback code designed for concatenated coding. The main insight motivating our design is that existing DNN-based feedback codes are optimized independently of the outer decoder’s structure, misallocating feedback resources toward error patterns that fall within the outer ECC’s correction capability. To address this, MASKCODE incorporates the structural constraint of the outer linear block code into the inner feedback encoder via two synergistic mechanisms that explicitly: 1) a codeaware attention mask derived from the Tanner graph and 2) a soft syndrome-based input design that informs the encoder about potential parity constraint violations. While end-to-end training allows the inner code to account for the outer decoder’s behavior, we empirically show that MASKCODE achieves its best performance without any BP iterations in training, as the structural constraint already internalized in the architecture subsumes the benefit that endto-end training would otherwise provide, while avoiding the gradient instability introduced by backpropagation through BP iterations. Extensive evaluations on both BCH and LDPC outer codes demonstrate that MASKCODE consistently outperforms all baselines across various experimental settings.

## REFERENCES

[1] C. G. Brinton, M. Chiang, K. T. Kim, D. J. Love, M. Beesley, M. Repeta, J. Roese, P. Beming, E. Ekudden, C. Li et al., “Key focus areas and enabling technologies for 6G,” IEEE Commun. Mag., vol. 63, no. 3, pp. 84–91, 2025.

[2] H. Kim, J. Choi, and D. J. Love, “Machine-learning techniques for wireless channel prediction: Insights and practical guidance,” IEEE Wireless Commun., pp. 1–8, 2026.

[3] J. Jang and H. J. Yang, “Deep reinforcement learning-based resource allocation and power control in small cells with limited information exchange,” IEEE Trans. Veh. Technol., vol. 69, no. 11, pp. 13 768– 13 783, 2020.

[4] Y. Choukroun and L. Wolf, “Error correction code transformer,” in Adv. Neural Inf. Process. Syst., 2022.

[5] D. Marasinghe, L. H. Nguyen, J. Mohammadi, Y. Chen, T. Wild, and N. Rajatheva, “Waveform learning under phase noise impairment for sub-thz communications,” IEEE Trans. Commun., vol. 73, no. 1, pp. 117–131, 2025.

[6] R. Sun, N. Cheng, C. Li, W. Quan, H. Zhou, Y. Wang, W. Zhang, and X. Shen, “A comprehensive survey of knowledge-driven deep learning for intelligent wireless network optimization in 6G,” IEEE Commun. Surveys Tuts., vol. 28, pp. 1099–1135, 2026.

[7] T. Park, S.-J. Park, H.-Y. Kwak, S.-H. Kim, and Y. Kim, “Lowering the error floor of error correction code transformer,” arXiv preprint arXiv:2502.09065, 2025.

[8] K. I. Ahmed, H. Tabassum, and E. Hossain, “Deep learning for radio resource allocation in multi-cell networks,” IEEE Network, vol. 33, no. 6, pp. 188–195, 2019.

[9] C. Shannon, “The zero error capacity of a noisy channel,” IRE Trans. on Inf. Theory, vol. 2, no. 3, pp. 8–19, 1956.

[10] T. Kadota, M. Zakai, and J. Ziv, “Capacity of a continuous memoryless channel with feedback,” IEEE Transactions on Information Theory, vol. 17, no. 4, pp. 372–378, 1971.

[11] J. Schalkwijk and T. Kailath, “A coding scheme for additive noise channels with feedback–I: No bandwidth constraint,” IEEE Trans. Inf. Theory, vol. 12, no. 2, pp. 172–182, 1966.

[12] J. Schalkwijk, “A coding scheme for additive noise channels with feedback–II: Band-limited signals,” IEEE Trans. Inf. Theory, vol. 12, no. 2, pp. 183–189, 1966.

[13] R. K. Mishra, D. Vasal, and H. Kim, “Linear coding for AWGN channels with noisy output feedback via dynamic programming,” IEEE Trans. Inf. Theory, vol. 69, no. 8, pp. 4889–4906, 2023.

[14] M. V. Burnashev, “Data transmission over a discrete channel with feedback. random transmission time,” Problemy peredachi informatsii, vol. 12, no. 4, pp. 10–30, 1976.

[15] Y.-H. Kim, “Feedback capacity of the first-order moving average Gaussian channel,” IEEE Trans. Inf. Theory, vol. 52, no. 7, pp. 3063–3079, 2006.

[16] Y.-H. Kim, A. Lapidoth, and T. Weissman, “The Gaussian channel with noisy feedback,” in IEEE Int. Symp. Inf. Theory (ISIT). IEEE, 2007, pp. 1416–1420.

[17] Y.-H. Kim, “Feedback capacity of stationary Gaussian channels,” IEEE Transactions on Information Theory, vol. 56, no. 1, pp. 57–85, 2009.

[18] Z. Chance and D. J. Love, “Concatenated coding for the AWGN channel with noisy feedback,” IEEE Trans. Inf. Theory, vol. 57, no. 10, pp. 6633– 6649, 2011.

[19] H. Kim, Y. Jiang, S. Kannan, S. Oh, and P. Viswanath, “Deepcode: Feedback codes via deep learning,” Adv. Neural Inf. Process. Syst., vol. 31, 2018.

[20] J. Kim, T. Kim, D. J. Love, and C. G. Brinton, “Robust non-linear feedback coding via power-constrained deep learning,” in Int. Conf. Mach. Learn. (ICML), vol. 202. PMLR, 23–29 Jul 2023, pp. 16 599– 16 618.

[21] E. Ozfatura, Y. Shao, A. G. Perotti, B. M. Popovic, and D. G´ und¨ uz, “All¨ you need is feedback: Communication with block attention feedback codes,” IEEE J. Sel. Areas Inf. Theory, vol. 3, no. 3, pp. 587–602, 2022.

[22] Y. Shao, E. Ozfatura, A. G. Perotti, B. M. Popovic, and D. G´ und¨ uz,¨ “Attentioncode: Ultra-reliable feedback codes for short-packet communications,” IEEE Trans. Commun., vol. 71, no. 8, pp. 4437–4452, 2023.

[23] S. K. Ankireddy, K. Narayanan, and H. Kim, “Lightcode: Light analytical and neural codes for channels with feedback,” IEEE J. Sel. Areas Commun., 2025.

[24] G. Forney, Concatenated Codes, ser. M.I.T. Press research monographs. M.I.T. Press, 1966.

[25] S. B. Wicker and V. K. Bhargava, Reed-Solomon codes and their applications. John Wiley & Sons, 1999.

[26] J. Cain and R. Simpson, “Concatenation schemes for the Gaussian channel with feedback (corresp.),” IEEE Trans. Inf. Theory, vol. 16, no. 5, pp. 632–635, 1970.

[27] M. Agrawal, D. J. Love, and V. Balakrishnan, “An iteratively optimized linear coding scheme for correlated Gaussian channels with noisy feedback,” in Annu. Allerton Conf. Commun., Control, Comput. IEEE, 2011, pp. 1012–1018.

[28] Z. Ahmad, Z. Chance, D. J. Love, and C.-C. Wang, “Concatenated coding using linear schemes for Gaussian broadcast channels with noisy channel output feedback,” IEEE Trans. Commun., vol. 63, no. 11, pp. 4576–4590, 2015.

[29] R. G. Gallager and B. Nakiboglu, “Variations on a theme by Schalkwijk˘ and Kailath,” IEEE Trans. Inf. Theory, vol. 56, no. 1, pp. 6–17, 2009.

[30] A. Ben-Yishai and O. Shayevitz, “The Gaussian channel with noisy feedback: Near-capacity performance via simple interaction,” in Annu. Allerton Conf. Commun., Control, Comput. IEEE, 2014, pp. 152–159.

[31] ——, “Interactive schemes for the AWGN channel with noisy feedback,” IEEE Trans. Inf. Theory, vol. 63, no. 4, pp. 2409–2427, 2017.

[32] A. R. Safavi, A. G. Perotti, B. M. Popovic, M. B. Mashhadi, and D. Gunduz, “Deep extended feedback codes,” arXiv preprint arXiv:2105.01365, 2021.

[33] M. B. Mashhadi, D. Gunduz, A. Perotti, and B. Popovic, “Drf codes: Deep snr-robust feedback codes,” arXiv preprint arXiv:2112.11789, 2021.

[34] J. Kim, T. Kim, A. B. Das, S. Hosseinalipour, D. J. Love, and C. G. Brinton, “Coding for Gaussian two-way channels: Linear and learningbased approaches,” IEEE Trans. Inf. Theory, 2025.

[35] D. R. Nickel, A. B. Das, D. J. Love, and C. G. Brinton, “Learningbased two-way communications: Algorithmic framework and comparative analysis,” IEEE Commun. Lett., 2025.

[36] S.-J. Park, H.-Y. Kwak, S.-H. Kim, S. Kim, Y. Kim, and J.-S. No, “Multiple-masks error correction code transformer for short block codes,” IEEE J. Sel. Areas Commun., vol. 43, no. 7, pp. 2518–2529, 2025.

[37] S.-J. Park, H.-Y. Kwak, S.-H. Kim, Y. Kim, and J.-S. No, “CrossMPT: Cross-attention message-passing transformer for error correcting codes,” arXiv preprint arXiv:2405.01033, 2024.

[38] C. Berrou, A. Glavieux, and P. Thitimajshima, “Near Shannon limit error-correcting coding and decoding: Turbo-codes. 1,” in IEEE Int. Conf. Commun. (ICC), vol. 2, 1993, pp. 1064–1070 vol.2.

[39] Y. Jiang, H. Kim, H. Asnani, S. Kannan, S. Oh, and P. Viswanath, “Turbo autoencoder: Deep learning based channel codes for point-topoint communication channels,” Adv. Neural Inf. Process. Syst., vol. 32, 2019.

[40] J. Streit, F. Weindel, and R. Heckel, “Transformer-based decoding in concatenated coding schemes under synchronization errors,” in IEEE Int. Symp. Inf. Theory (ISIT). IEEE, 2025, pp. 1–6.

[41] A. Hocquenghem, “Codes correcteurs d’erreurs,” Chiffers, vol. 2, pp. 147–156, 1959.

[42] R. C. Bose and D. K. Ray-Chaudhuri, “On a class of error correcting binary group codes,” Information and control, vol. 3, no. 1, pp. 68–79, 1960.

[43] R. Gallager, “Low-density parity-check codes,” IRE Trans. on Inf. Theory, vol. 8, no. 1, pp. 21–28, 1962.

[44] D. J. MacKay and R. M. Neal, “Near shannon limit performance of low density parity check codes,” Electronics letters, vol. 32, no. 18, pp. 1645–1646, 1996.

[45] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Adv. Neural Inf. Process. Syst., vol. 30, 2017.

[46] R. McEliece, The theory of information and coding. Cambridge University Press, 2002.

[47] F. R. Kschischang, B. J. Frey, and H.-A. Loeliger, “Factor graphs and the sum-product algorithm,” IEEE Trans. Inf. Theory, vol. 47, no. 2, pp. 498–519, 2001.

[48] Y. Jiang, H. Kim, H. Asnani, S. Oh, S. Kannan, and P. Viswanath, “Feedback turbo autoencoder,” in IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2020, pp. 8559–8563.

[49] I. Tal and A. Vardy, “List decoding of polar codes,” IEEE Trans. Inf. Theory, vol. 61, no. 5, pp. 2213–2226, 2015.

[50] M. JL, “Shift-register synthesis and bch decoding,” IEEE Trans. Inf. Theory, vol. 13, pp. 21–27, 1967.

[51] M. Levy, Y. Choukroun, and L. Wolf, “Accelerating error correction code transformers,” arXiv preprint arXiv:2410.05911, 2024.

[52] Y. Choukroun and L. Wolf, “A foundation model for error correction codes,” in Int. Conf. Learn. Represent. (ICLR), 2024.

[53] N. Biggs, Algebraic graph theory. Cambridge university press, 1993, no. 67.

[54] D. Crnkovic, S. Rukavina, and M.´ Simac, “LDPC codes constructed<sup>ˇ</sup> from cubic symmetric graphs,” Applicable Algebra in Engineering, Communication and Computing, vol. 33, no. 5, pp. 505–522, 2022.

[55] G. Larue, L.-A. Dufrene, Q. Lampin, P. Chollet, H. Ghauch, and G. Rekaya, “Blind neural belief propagation decoder for linear block codes,” in Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit). IEEE, 2021, pp. 106–111.

[56] E. Nachmani and L. Wolf, “Hyper-graph-network decoders for block codes,” Adv. Neural Inf. Process. Syst., vol. 32, 2019.

[57] J. Jang, H. Nam, V. Tripathi, and D. J. Love, “Syndromecode: Feedbackassisted concatenated coding based on transformer,” in Asilomar Conf. Signals, Syst. Comput., 2025, pp. 1178–1182.

![](images/f680fe19fda22ceb95409127578caf2c4649a0236780c3f81aecf7ecfed3aaf8.jpg)  
Fig. 11. BLER performance of MASKCODE under various feedback quantization levels. The MASKCODE with 8-bit feedback quantization has a comparable BLER to that with full precision (32 bit). The outer code is the BCH(31,16) code.

## APPENDIX A

## QUANTIZED FEEDBACK SCENARIO

To evaluate the practical feasibility of MASKCODE, we investigate its performance under quantized feedback in Fig. 11, where the feedback signal is quantized to varying resolutions from 5 to 32 bits, with the BCH(31,16) outer code and feedforward SNR of 2 dB. While low-resolution quantization of 5 or 6 bits leads to a noticeable BLER degradation, the performance improves consistently as the bit-depth increases. At 8-bit quantization, the BLER of MASKCODE converges to that of the full-precision (32-bit) case, demonstrating that MASKCODE is robust to quantization noise and can be effectively deployed in practical hardware environments with limited feedback link capacity.

## APPENDIX B

## ATTENTION MASK ANALYSIS

Here, we analyze the self-attention maps to investigate the internal operations of the MASKCODE encoder. In Fig. 12, we specifically examine the first and second layers of the encoder during the second phase (t = 2) with the LDPC(49,24) code. In the second phase, the encoder generates the parity message $\mathbf { x } _ { 2 }$ after receiving the initial feedback. To observe how the model reacts to specific errors, we inject controlled noise into the first-phase message $\mathbf { x } _ { 1 }$ . More specifically, the noise corresponding to the first element of $\mathbf { x } _ { 1 }$ is varied, while the noise injected into all other elements is set to zero. According to the parity-check matrix H, the syndrome positions associated with the first codeword bit $( [ \mathbf { H } ] _ { i , 1 } = 1 )$ are at indices 49, 56, 63, and 70.

Second layer (Structured refinement). Building on the error recognition in the first layer, the second layer further processes the syndrome terms at positions 49, 56, 63, and 70, which now carry error-related information identified in the first layer, as depicted in the second row of Fig. 12. More specifically, the parity message elements at positions sharing common check nodes with the first codeword bit (highlighted in blue) are refined using information from syndrome positions 49, 56, 63, and 70. This layer-by-layer progression demonstrates that MASKCODE does not merely treat bits independently but instead performs a structured message-passing-like operation to reinforce bits that the outer decoder might find vulnerable. Thus, our proposed synergistic design—combining the codeaware mask and syndrome-based input—successfully guides the Transformer to adhere to the structure of outer ECCs.

First layer (Error recognition). The attention maps of the first layer are shown in the first row of Fig. 12. Comparing the two noise levels, the syndrome positions 49, 56, 63, and 70 (highlighted in green) exhibit progressively stronger attention toward the first codeword bit as noise increases, indicating that these syndrome positions are updated with information from the corrupted bit. This indicates that the first layer actively identifies potential parity constraint violations by focusing on the specific codeword bit corrupted by noise.

![](images/305f04202c669d9dc5d741adce87288b4b633f878fadce551f0821a66f480841.jpg)  
Layer: 1, Noise scale: 0.0

![](images/cc0a2c4c41e82e7375eaeedf354665743f516a2b936ed299331a7f09af05e42c.jpg)  
Layer: 1, Noise scale: 4.0

![](images/f6292b74d55fd781369533fda39fb5de020c73085994142155efa9c959a98b57.jpg)  
Layer: 2, Noise scale: 0.0

Parity message elements sharing common check nodes with position 0 are updated with information from syndrome positions 49, 56, 63, 70  
![](images/e2397484dd177a6796ba44e475b89200f86bd1fe5f7bb1e59f0c4ec1b8b678bc.jpg)  
Layer: 2, Noise scale: 4.0  
Fig. 12. Illustration of self-attention maps at different layers of the MASKCODE encoder layer in the second phase; the outer code is the LDPC(49,24) code. Self-attention maps are shown for two noise levels (scale = 0.0 and 4.0) injected into the first element of x<sub>1</sub>, while noises on all other elements are set to zero.