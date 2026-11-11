# Machine-learning decoders for fusion-based quantum computing in real time

Daniel Barter,<sup>1,</sup> ∗ Jacob C. Bridgeman,<sup>1</sup> Chris Dawson,<sup>1</sup> Terry Farrelly,<sup>1</sup> Shane Fleming,<sup>2</sup> Mercedes Gimeno-Segovia,<sup>1</sup> Paul Hartke,<sup>2</sup> Timothy Hein,<sup>2</sup> Sean Howell,<sup>1</sup> Gregory Hyegang Jun,<sup>2</sup> Siddhartha Kamalakara,<sup>1</sup> Yehua Liu,<sup>1</sup> Jose M. Monsalve Diaz,<sup>2</sup> Eddie Richter,<sup>2,</sup> † and Sam Roberts<sup>1</sup>

<sup>1</sup>PsiQuantum, Palo Alto, CA, USA‡ <sup>2</sup>AMD, Santa Clara, CA, USA§ (Dated: November 11, 2026)

Fault-tolerant quantum computation requires decoders that combine accuracy, throughput, low latency, and scalability to large numbers of physical qubits. In photonic fusion-based architectures, erasures from photon loss and probabilistic fusion measurements are a dominant error mechanism, motivating decoders that remain accurate at high erasure rates. We introduce PRISM, a transformer-based machine-learning decoder designed to exploit the known locations of erasures in protocols subject to both Pauli and erasure errors. We benchmark PRISM across a wide range of fault-tolerant protocols based on surface codes, color codes, and quantum lowdensity parity-check codes, with distances up to 24. Across these benchmarks, PRISM matches the logical error rates of integer-programming-based decoders while being significantly faster, including at error rates up to 50 % of threshold. We further optimize the decoder and its real-time hardware implementation to reduce latency while maintaining accuracy. Our optimized implementation on an AMD VP1902 FPGA can decode logical blocks of distance 8 in both space and time, demonstrating a latency of 6.59 µs and an amortized decoding time of 1.56 µs per logical block is possible. These results show that machine-learning decoding can provide accurate, lowlatency inference for erasure-dominated fault-tolerant quantum architectures.

## CONTENTS

I. Introduction 2   
Acknowledgments 3   
II. Background: Fault tolerance and fusion-based quantum computation 4   
A. Fault-tolerant protocols and fusion-based quantum computation 4   
B. Error model 5   
C. The decoding problem 5   
D. Latency requirements 6   
E. Fusion networks studied 6   
III. Machine-learning decoding 7   
A. Architecture 7   
B. Training the model 8   
C. Multiple membrane representatives 9   
D. Relation to previous neural decoders 9   
IV. Decoder performance 10   
V. Real-time decoding 11   
A. Decoding with low-precision arithmetic 11   
B. Quantized model performance 12   
C. Q-PRISM accelerator architecture 13   
D. Hardware mapping 14   
E. Results 15   
VI. Outlook 17   
References 18   
A. Notation 23   
B. Fusion networks 24   
1. Fusion error models 25   
C. Machine-learning decoding 26   
1. Architecture 26   
2. Training 29   
D. Quantized architecture and training 32   
1. Numerical representation and logical readout 32   
2. Training and model selection 34   
3. Design choices 34   
E. Cross-attention hardware mapping 37   
F. Optimizations to achieve high-frequency FPGA mapping 40   
G. Quantized softmax and normalization for efficient hardware mapping 43

## I. INTRODUCTION

Fault-tolerant quantum computation relies on quantum error correction (QEC) to suppress logical errors far below the error rates of the underlying physical operations [1–5]. This suppression comes at the cost of additional physical qubits (space over head), and increased execution time (time overhead). Improvements in hardware, code design, and fault-tolerant protocols have substantially reduced these overheads, strengthening the prospects for practically useful fault-tolerant quantum computation [6– 24].

Central to fault tolerance is the decoder: a classical algorithm that determines the appropriate logical correction or Pauli frame update from the measurement outcomes of the fault-tolerant protocol [25]. Here, the Pauli frame is a way of accounting for errors by tracking their effective logical effect, and is used to interpret logical measurements and inform subsequent feedforward operations. To ensure accurate Pauli frame information is available on demand, this decoding must be done in real time. The performance of the decoder is critical to both the space and time overheads of a fault-tolerant computation: an inaccurate decoder increases the logical error rate, while a slow decoder can delay the logical operations, giving a slow-down of the logical clock speed.

Developing a decoder that is simultaneously accurate, fast, and scalable remains challenging. There have been significant advances in decoding accuracy, asymptotic runtime and scalability, and real-time performance [26–30], with several real-time implementations on FPGAs [31–38]. Nevertheless, the computational structure of the decoding problem depends strongly on the code or fault-tolerant protocol, as well as the physical noise model. Exact optimal decoding (also known as maximum likelihood decoding)—which determines the most-likely logical error class given the syndrome—is generally inefficient, even for surface codes [39], and depends on the error model [40]. Most-likely error decoding (also known as minimum-weight decoding) is a high-accuracy alternative which is efficient for surface codes (under certain error models), but becomes intractable (NP hard) for other prominent codes like color codes [41, 42]. Practical decoders therefore rely on approximations that trade accuracy against computational cost. A broadly useful decoder should achieve high logical accuracy across relevant operating regimes while admitting predictable, low-latency implementation at large scale.

Machine-learning (ML) methods have recently emerged as a very promising approach to this end, with demonstrations of both high accuracy and low latency [27, 43–69]. A particular benefit of this approach is the ability to fine-tune the decoder based on experimental data, allowing it to capture correlations or other error mechanisms that are difficult to represent in simulation [27, 70], thereby improving decoding accuracy in realistic operation. This was demonstrated in the AlphaQubit decoder of Ref. [27], which presents an ML decoder as a compelling path for superconducting architectures. There have been a variety of ML approaches to ML decoding, and we provide a brief overview in Section III D.

It remains unclear, however, the extent to which such models can maintain both accuracy and practical latency at the scales and error rates anticipated in large-scale fault-tolerant systems. This question is especially important when erasure errors occur frequently. Erasures can arise from leakage or loss (when detected) in several hardware platforms [71–73], and can occur at a significant rate. In photonic fusion-based quantum computation, they also arise intrinsically from probabilistic linearoptical fusion measurements, even when all components operate ideally [71, 74]. The known locations of erasures provide useful information to a decoder, but high erasure rates can generate large and irregular inference problems—complicating some approaches to decoding. Preserving strong logical error rate (LER) and threshold performance in this regime can therefore impose substantial computational costs on conventional decoding methods.

In this paper, we introduce an ML decoder, called PRISM, that meets the needs of photonic fusion-based quantum computa tion. We demonstrate its performance via logical-error-rate simulations for several fusion-based protocols, including topological protocols based on surface codes and color codes, with distances of up to 24, and with error rates up to 50 % of threshold (a determined by most-likely-error decoding crossings between small distances) for an error model involving both erasure and Pauli noise. For IID error models, where qubits or measurement outcomes are subject to erasures and Pauli errors with some probabil ity, our decoder matches the performance of a state-of-the-art conventional decoder, based on most-likely error. Moreover, the models can be trained on more realistic error models, are fast, and scale well to large protocol sizes owing to the use of sparse attention. Another advantage of this approach is that the decoder time is fixed and independent of input. This is in contrast to some algorithmic decoders which may have a run-time that depends on the noise model, and may also face high-variance, with difficult-to-decode shots that stall the quantum computation.

We optimize the decoder speed and latency by training decoders with ternary weights and two-bit activations whose arithmetic maps onto lookup tables. We perform a timing analysis to show that the models can be mapped onto hardware while meeting the latency targets for utility-scale quantum computations. We then validate this analysis by implementing the decoder on an AMD Versal<sup>TM</sup> VP1902 FPGA for a distance-8 fusion network. We consider the decoding problem for periodic logical blocks that have distance d in both space and time and demonstrate a full-block decoding latency of 6.59 µs and throughput of 1.56 µ is possible. As a comparison to circuit-based protocols, this decoding problem is roughly equivalent to decoding 8 syndrome extraction rounds of a distance-8 surface code. We measure latency from the first input of the decoding block to the complete decoded output of the FPGA, which includes input loading.

To construct PRISM, we represent the decoding problem as a Tanner graph of checks (also known as detectors), error mech anisms and logical observables. We can then make use of a code-aware transformer, which restricts attention according to neighborhoods of this check graph (as has been recently done in Ref. [58]). We then extend this basic idea in four directions. First, every error mechanism carries an erasure flag in addition to its contribution to the syndrome, so the model receives the erasure pattern as an input on the same footing as the check outcomes. Second, messages between error mechanisms and checks are exchanged only along edges of the Tanner graph by sparse attention, and the model receives no positional information, so that both memory and compute grow linearly with the size of the protocol (at fixed model width and depth). Third, we use perceiver aggregation to collect features over the supports of multiple equivalent representatives of each logical observable. Predictions for the representatives are combined by voting after accounting for their syndrome-determined parity differences, which we call a syndrome-aligned vote. Fourth, for deployment we train decoders whose matrix weights are ternary and whose activations have two bits, with shift-based replacements of softmax and layer normalization, so that the trained network maps onto the lookup-table fabric of a single FPGA.

The paper is organized as follows. Section II formulates the decoding problem, fault-tolerant protocols, and error models in the context of fusion-based quantum computation. Section III introduces the ML decoder, including a training overview in Section III B. Section IV presents the simulation results for the fusion networks of interest. Section V presents the real-time implemented results, including the FPGA architecture and latency results. We conclude with an outlook in Section VI.

For completeness, we provide several appendices. In Appendix A, we provide a table of notation used in this work. Appendix B describes the fusion networks considered in this work. Appendix C details the decoder architecture and training curriculum. Appendix D describes the quantization and jointly-optimized decoder architecture for FPGA implementation. Ap pendix E describes the mapping and scheduling of sparse cross-attention on the FPGA. Appendix F describes optimizations for achieving a high operating frequency. Finally, Appendix G presents hardware-efficient approximations to softmax and normal ization used during quantization-aware training.

## ACKNOWLEDGMENTS

These results support machine-learning decoding as a route to utility-scale quantum computing for scientific application such as those identified in U.S. Department of Energy’s Quantum Genesis Priority Applications. Large-scale benchmarking was performed on Lux, a supercomputer supporting the Genesis Mission. This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Advanced Scientific Computing Research programs in the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

We thank Naomi Nickerson, Terry Rudolph, Basudha Srivastava, and Kwok Ho Wan for useful discussions, Andrew Peck, Axel Dahlberg and Damien Bonneau for discussions on real-time decoder implementations, Joel Wallman for feedback on the draft of this manuscript, Thomas Preußer for his guidance on FPGA ML model mappings, Nicholas Fraser for his guidance on quantization-aware training, Chris Lavin for his help achieving high-frequency FPGA designs, and Eddie Vergara and Catherine Warren for their help on FPGA infrastructure. We would also like to thank all of our colleagues at PsiQuantum and AMD for help and feedback on this work. LLMs were used in the writing, editing, data analysis, and exploration of the ideas in this paper.

Mercedes Gimeno-Segovia’s present affiliation is NVIDIA, Santa Clara, CA, USA. Siddhartha Kamalakara’s present affiliation is ambition.inc, Toronto, ON, Canada.

## II. BACKGROUND: FAULT TOLERANCE AND FUSION-BASED QUANTUM COMPUTATION

In this section, we describe the framework for fault-tolerant protocols, error models, and the decoding problem. We describe the protocols in the language of fusion-based quantum computation (FBQC), but the decoder of Section III only requires a check matrix, a membrane matrix (Equation (1b)) and an error model. It can therefore be applied to any model of quantum computation described by detectors and logical observables, such as circuit-based quantum computation [75].

## A. Fault-tolerant protocols and fusion-based quantum computation

In FBQC, the fault-tolerant computation is carried out by generating small entangled resource states of constant size, generally stabilizer states, and by performing entangling two-qubit measurements, called fusions, between qubits of different resource states [71]. A fusion network describes both the resource states and the fusions that need to be performed. The fusions, in the ideal case, are measurements in the Bell-basis, giving two binary outcomes for the XX and $Z Z$ observables. This model is suitable for describing dual-rail photonic qubits, where the fusions are also known as Type-II fusions [71, 74]. These classical outputs are used both for decoding, and for interpreting the output of the computation.

Checks & logical membranes. The origin of fault tolerance comes from the redundancy between fusion measurement outcomes and the resource state stabilizers. The group of checks is the intersection of the group generated by all resource state stabilizers and the group generated by all measured operators. Thus, the product of measurement outcomes corresponding to a check has fixed parity in the absence of errors. The outcomes of XX and $Z Z$ measurements are 1 with both typically being equally likely, but it is more convenient in simulations to work with errors (rather than outcomes), where 0 indicates no error and 1 indicates an error. Then a check value of 1 indicates the presence of errors. The vector of check values is called the syndrome, denoted σ. Logical information is carried by logical membranes (also called correlation surfaces [76]), which are set of outcomes whose joint parity determines the logical Pauli frame update [71, 77]. In other words, logical membranes track how logical observables evolve throughout the fault-tolerant protocol. Errors can change the sign of a logical membrane, which we call a membrane flip. It is the role of the decoder to use the syndrome to infer these flips and thereby correct the interpretation of the logical information. Decoding fails when at least one membrane flip is inferred incorrectly.

Decoding. For the purpose of decoding, a protocol is specified by three objects. The first is a set of $n _ { \mathrm { E } }$ error mechanisms, independent binary faults that flip measurement outcomes. As described in the following section, for the fusion network studied here, error mechanisms are one-to-one with fusion outcomes. The second is the check matrix $H \in \mathbb { F } _ { ? } ^ { n _ { \mathrm { C } } \times n _ { \mathrm { E } } } .$ , whose entry $H _ { j i } = 1$ if error mechanism i flips check $j$ and is zero otherwise. The third is the membrane matrix $\boldsymbol { \Lambda } \in \bar { \mathbb { F } } _ { 2 } ^ { ( K \times R ) \times n _ { \mathrm { E } } }$ , whose entry $\Lambda _ { \mu i } = 1$ if error mechanism i flips membrane $\mu$ and is zero otherwise. A Pauli error $e \in \mathbb { F } _ { 2 } ^ { n _ { \mathrm { E } } }$ is the indicator vector of the flipped error mechanisms. It yields the syndrome and membrane flips via

$$
\sigma = H e\tag{1a}
$$

$$
f = \Lambda e .\tag{1b}
$$

Given a noise model, which is a distribution over erasures and flips on error mechanisms, decoding is the task of inferring f from σ and ε. Some decoders infer f by first picking an error estimate matching the syndrome: a decoder returning an error estimate eˆ satisfying $H \boldsymbol { \hat { e } } = \sigma$ succeeds precisely when $\Lambda \hat { e } = \Lambda e$

Equivalent logical membranes. Two membranes $\mu$ and ν are equivalent (denoted $\mu \sim \nu )$ if there exists some binary vector t such that

$$
\Lambda _ { \mu } = \Lambda _ { \nu } + t H .\tag{2}
$$

In other words, the membranes are equivalent up to checks. Flips on equivalent membranes can be translated using the syndrome $f _ { \mu } = f _ { \nu } + t \sigma$ , so they carry the same logical information.

The membranes in Λ fall into K equivalence classes, which correspond to independent logical degrees of freedom of the protocol. For decoding, we keep R equivalent representatives of each class, for reasons explained in Section III C, and write $\varkappa ( \mu )$ for the class of membrane $\mu .$

Tanner graphs. It is also useful to consider the Tanner graph of the protocol. This is the bipartite graph whose two vertex sets are the error mechanisms and the checks, with an edge between mechanism i and check $j$ whenever $H _ { j i } = 1$ . For a given basis of H, we refer to the maximum column weight as the outcome degree, and the maximum row weight as the check weight. All protocols in this paper are low-density: outcome degrees and check weights are upper bounded by a constant. If every erro mechanism flips at most two checks, the protocol is graphlike and one can utilize minimum-weight perfect matching (MWPM) decoding [78, 79] as described in Section II C. Protocols with outcome degree of more than 2 are considered more difficult to decode.

![](images/1202688a344de4f073fe62735bf8037f27c6a88f59645deadd3cf51ec4ebe27e.jpg)  
FIG. 1: The decoding problem. (a) A schematic of a fault-tolerant protocol. A set of equivalent membranes is indicated by shaded sheets. A chain of errors (red) that flips a logical membrane without flipping any checks results in a logical error. (b) A Tanner graph example. Circle nodes are error mechanisms (fusion outcomes in FBQC) and square nodes are checks. The decoder observes which mechanisms are erased (crossed circles) and which checks are flipped (filled squares), but not the Pauli error itself (red circles). The highlighted mechanisms belong to the support of one membrane representative. (c) The decoding task: given the erasure pattern ε and the syndrome σ, infer for each membrane class whether it was flipped. Here, $p ( f _ { \mu } \mid \sigma , \varepsilon )$ is the decoder; it predicts membrane flip probabilities based on the syndrome σ and erasure pattern ε for each logical representative indexed by $\mu .$

Logical blocks. A fault-tolerant computation is composed of logical blocks, such as Clifford or lattice surgery blocks [77]. In this work we exclusively study fusion networks with periodic boundary conditions in time, which we call periodic decoding blocks. The periodic decoding block provides a standard setting to compare decoders [71, 78, 80], accounting for both spacelike and timelike logical failures. For the topological networks studied here, logical membrane classes are supported on nontrivial 2-cycles that wrap around the 3-torus, while for the qLDPC networks, they are determined based on the logical operators of the code. There are 6 classes for the surface-code-like networks below, 12 classes for the color-code network, and 24 classes for the qLDPC networks. For each block, we report the fault-distance, the minimum number of error mechanisms that result in a logical failure without flipping any checks, in Table I.

## B. Error models

An error model assigns probabilities to joint configurations of Pauli errors and erasures of the error mechanisms. An erasure is a heralded error: the decoder learns that the outcome of an erased mechanism is unreliable or unavailable, and the outcome i replaced by a uniformly random bit. Equivalently, an erased measurement outcome can be modeled as an appropriate Pauli error with probability $1 / 2$ at a known location. In photonic FBQC, erasures arise from photon loss, which erases both outcomes of a fusion, and from fusion failure, which erases one of the two [71] [81].

In this work, fusion measurement outcomes and error mechanisms are in one-to-one correspondence; both measurement outcomes of a fusion can be assigned independent error mechanisms. An error model is a probability distribution over the fusion outcome bitflips (resulting from a Pauli error) and erasures. Sampling $( e , \varepsilon )$ from the error model, with $\varepsilon \in \mathbb { F } _ { 2 } ^ { n _ { \mathrm { E } } }$ the indicator vector of erased mechanisms, determines the syndrome and membrane flips.

There is no single error model for dual-rail photonics, as there are a variety of design options (such as boosting and local encoding [71]) that trade off between fusion failure rates, susceptibility to loss, and Pauli sensitivity. As such, we consider a hardware-agnostic error model, where each fusion outcome is erased independently with probability $p _ { E }$ , and non-erased outcomes are subject to a bitflip with probability $p _ { P }$ . We parametrize these rates by a single parameter $x ,$ called the operating point. We consider the error model

$$
p _ { E } = x , \qquad p _ { P } = x / 1 0 .\tag{3}
$$

This error model is the fusion-based analogue of the circuit model, and captures the higher rate of fusion erasure compared to bitflips, following Appendix B 1. Note that the ML-decoder in this work places no restriction on the error model, and the same training procedure applies to correlated error models.

## C. The decoding problem

For a given protocol and error model, sampling from the error model induces a joint distribution over triples $( f , \sigma , \varepsilon )$ of membrane flips, syndromes and erasure patterns, which we call the decoding distribution $( p _ { \mathrm { { D E C } } } )$ . The decoder observes $( \sigma , \varepsilon )$

and must predict the flips of the membrane classes. Information theoretically, the optimal decoder [78, 82], assuming Pauli and erasure error channels, is

$$
\hat { f } = \underset { f } { \arg \operatorname* { m a x } } \ p _ { \mathrm { { D E C } } } ( f \mid \sigma , \varepsilon ) ,\tag{4}
$$

where $f$ ranges over the flips of one representative per class. This is maximum-likelihood decoding for each logical class. A decoder fails on a shot if its prediction for at least one of the K classes is wrong, and the logical error rate (LER) is the probability of failure. Note here that this decoder returns probabilities of logical error equivalence classes rather than a physical recovery operation in the form of a reference error configuration, and this is sometimes referred to as soft output decoding. In Section III C 1, we describe how to subsequently obtain a reference recovery (so-called hard output decoding).

Evaluating Equation (4) requires summing over all errors compatible with the syndrome and is generally intractable [39]. While efficient approximations exist for planar surface codes [40], the three-dimensional networks considered here are not known to have efficient approximations.

The most-likely-error (minimum-weight) decoder instead finds a single error,

$$
\hat { e } = \underset { e : H e = \sigma } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { n _ { \mathrm { E } } } w _ { i } e _ { i } , \quad \quad \mathrm { w h e r e } \quad \quad w _ { i } = \log \frac { 1 - q _ { i } } { q _ { i } } ,\tag{5}
$$

and predicts $\hat { f } = \Lambda \hat { e }$ . Therein, $q _ { i }$ is the flip probability of mechanism i given the erasure pattern, called the physical error rate, and $w _ { i }$ is the log-likelihood ratio (LLR) of the physical error rate. In particular, $q _ { i } = 1 / 2$ and $w _ { i } = 0$ if the outcome is erased, and $q _ { i } = p _ { P }$ otherwise. For graphlike protocols, Equation (5) is a minimum-weight perfect-matching problem [28, 78, 79], while for hypergraph protocols such as linked-loopy-diamond and color-code networks below it is an integer linear program (ILP) [83]. The runtime for ILP decoding grows rapidly with the instance size, and we use it as an accuracy benchmark for small instances only. For quantum LDPC codes, belief propagation with ordered-statistics post-processing (BP-OSD) provides a practical approximation [13, 84].

## D. Latency requirements

In addition to accuracy, the decoder of a fault-tolerant computation must satisfy timing and latency requirements [25]. Its throughput must match the rate at which the hardware produces syndrome data, otherwise unprocessed data accumulates without bound. Its latency, the time from the arrival of the last outcome of a block to the availability of the corresponding Pauli frame update, adds to the reaction time of every feed-forward operation and thereby to the logical clock period. In general, the frame update must be performed before a non-Clifford logical operation, such as magic state injection, or another adaptive gate. A low latency is therefore required in order for the logical clock rate to not significantly deteriorate. Many parts of the computation can be decoded in parallel, making both single-block decoding latency and overall decoder throughput relevant performance metrics. Here, we are primarily concerned with the latency of decoding one periodic decoding block as our benchmark, following the modular decoding scheme of Ref. [85] (see also Refs. [86 and 87]). We leave the discussion of ML-decoder optimized modular decoding approaches to future work.

For the FPGA implementation of Section V we adopt a latency target of 10 µs per $d = 8$ linked-loopy-diamond periodic decoding block. This target guided the choice of model size and depth in Section V A. Note that this benchmark is for the $d \times d \times d$ logical block, which is made up of d code cycles. This scale is relevant for interleaved photonic architectures [88], where the logical cycle time can be tuned to be between 1 ns and 10 µs, from an underlying physical clock speed (for resourcestate generation and fusions) of 1 GHz [89].

## E. Fusion networks studied

We study periodic decoding blocks of four families of fusion networks (Table I). These include topological networks based on surface codes [90] (with and without Clifford deformations to the network, called linkings), color codes [91] and Bivariate Bicycle (BB) codes [16]. Note that all networks are defined on the 3-torus. The topological fusion networks (the surface and color code networks) have 1 or 2 X-type and Z-type logical membrane classes per nontrivial 2-cycle, giving 6 or 12 logical membrane classes. The BB codes have 24 membranes corresponding to the 12 logical qubits included. The constructions are explained in more detail in Appendix B.

For a particularly well studied class of codes, called CSS codes [92, 93], the decoding problem decomposes into two separate problems. We study both CSS and non-CSS examples.

Six ring network. The six-ring network [71] is a surface-code fusion network, with a graphlike Tanner graph, meaning it allows a direct comparison with minimum-weight perfect matching. The resource state and check structure for the six-ring network are CSS.

TABLE I: Parameters of the logical blocks used in this work. n, k, and d define the fusion network structure, denoting respectively the number of resource states in the periodic logical block, the number of encoded logical qubits, and the protocol distance (the minimum number of faults required for an undetectable logical error). n and $n _ { \mathrm { C } }$ are the numbers of error mechanisms and checks, K is the number of independent membrane classes of the full protocol, R is the number of representatives per membrane class used by the ML decoder, and $| \Lambda _ { \mu } |$ is the number of error mechanisms in the support of one membrane representative. Check weights and the number of checks flipped by one error mechanism (mechanism degree) are given as the sets of values that occur. ∗ For the CSS six-ring and BB networks, we report results for the primal sector only.
<table><tr><td rowspan="2">Network</td><td colspan="3">Protocol structure</td><td colspan="2">Logical membranes</td><td rowspan="2">nE</td><td rowspan="2">nc</td><td rowspan="2"> $| \Lambda _ { \mu } |$ </td><td rowspan="2">Check weights</td><td rowspan="2">Outcome degree</td></tr><tr><td>n</td><td>k</td><td>d</td><td>K</td><td>R</td></tr><tr><td rowspan="3">6-ring*</td><td>512</td><td>2</td><td>8</td><td>6</td><td>8</td><td>3,072</td><td>512</td><td>128</td><td>12</td><td>2</td></tr><tr><td>1,728</td><td>2</td><td>12</td><td>6</td><td>12</td><td>10,368</td><td>1,728</td><td>288</td><td>12</td><td>2</td></tr><tr><td>4,096</td><td>2</td><td>16</td><td>6</td><td>16</td><td>24,576</td><td>4,096</td><td>512</td><td>12</td><td>2</td></tr><tr><td rowspan="3">LLD</td><td>192</td><td>2</td><td>8</td><td>6</td><td>4</td><td>1,536</td><td>384</td><td>112</td><td>6, 12, 36</td><td>2,4</td></tr><tr><td>1,536</td><td>2</td><td>16</td><td>6</td><td>8</td><td>12,288</td><td>3,072</td><td>448</td><td>6, 12, 36</td><td>2,4</td></tr><tr><td>5,184</td><td>2</td><td>24</td><td>6</td><td>12</td><td>41,472</td><td>10,368</td><td>1,008</td><td>6, 12, 36</td><td>2,4</td></tr><tr><td rowspan="3">CC8</td><td>192</td><td>4</td><td>8</td><td>12</td><td>4</td><td>1,536</td><td>256</td><td>64, 96</td><td>12,24</td><td>3</td></tr><tr><td>1,536</td><td>4</td><td>16</td><td>12</td><td>8</td><td>12,288</td><td>2,048</td><td>256, 384</td><td>12, 24</td><td>3</td></tr><tr><td>5,184</td><td>4</td><td>24 12</td><td></td><td>12</td><td>41,472</td><td>6,912</td><td>576,864</td><td>12, 24</td><td>3</td></tr><tr><td rowspan="2">BB*</td><td>1,296</td><td>12</td><td>6</td><td>24</td><td>6</td><td>6,912</td><td>432</td><td>300, 352, 360</td><td>42</td><td>2,3</td></tr><tr><td>5,184</td><td>12</td><td>12</td><td>24</td><td>8</td><td>27,648</td><td>1,728</td><td>720-1,298</td><td>42</td><td>2,3</td></tr></table>

Linked-loopy-diamond. The linked-loopy-diamond (LLD) network is a topological network with a high error tolerance. The fusion network is based on loopy-diamond (also known as the cuboctahedral complex) of Refs. [94–96], with linking applied [19]. Linking is the process of applying local Clifford operators to qubits of a resource state to modify the check structure. The linking increases the outcome degree of some fusion outcomes, resulting in an improved performance against erasure and outcome flips. The resulting Tanner graph is no longer graph-like, meaning MWPM cannot be utilized. The linking procedure removes the CSS property (hence the name “linked”). Due to its high erasure tolerance, we consider the LLD network for implementation on FPGA, and in particular, for the d = 8 LLD protocol.

Color-code network. The color-code network (CC8) is built from eight-qubit resource states on a three-colorable lattice [95]. Its resource state and check structure is CSS. It is not graphlike, as each error mechanism has outcome degree 3.

Bivariate-Bicycle networks. The BB networks we consider are based on foliated versions of the BB codes of Ref. [16]. Specifically, we consider the networks based on the Gross 144, 12, 12 and Half-Gross 72, 12, 6 codes. These networks are also CSS and non-graphlike. More details on the structure of the foliated network can be found in Appendix B.

## III. MACHINE-LEARNING DECODING

For a given fault-tolerant protocol and error-model, recall from Section II C that the optimal decoder is given by the underlying distribution $p _ { \mathrm { D E C } } ( f \mid \sigma , \varepsilon )$ . In general, this distribution is hidden, but can be easily sampled from using simulated or experimental data. A machine-learning (ML) decoder uses these samples to train a model that approximates $p _ { \mathrm { D E C } } ( f \mid \sigma , \varepsilon )$ . Specifically, the model learns a parameterized distribution $p _ { \theta } ( f \mid \sigma , \varepsilon )$ by optimizing its parameters θ to minimize the discrepancy between p<sub>θ</sub> and $p _ { \mathrm { D E C } }$ , as discussed below. In this section, we describe the decoder architecture employed in this work and some of the challenges associated with training such a decoder under error models relevant to FBQC.

## A. Architecture

When in use, a decoder consumes the observed syndromes and erasure locations $( \sigma , \varepsilon )$ , and is expected to return enough information to correct the underlying physical error. This may take the form of an explicit correction or, as here, predicted logical flips that can be used to determine a recovery.

We defer a detailed discussion of our decoder architecture to Appendix C. At a high level, our decoder operates as follows. During each decoding round, the two binary vectors, σ (of length $n _ { \mathrm { C } } )$ and ε (of length n ), are input. The individual bits of these vectors are referred to as tokens. Learned embeddings map each token t to a (real) vector of dimension $d _ { \mathrm { M } }$ , called a state (Figure 2a). These states are then passed through a series of parametrized transformations, whose parameters are optimized during training (Figures 2b to 2d), then frozen. These transformations include a series of message-passing steps via sparse crossattention [97–99], where messages are constrained by the Tanner-graph, following Ref. [58]. The perceiver (Figure 2c), along with another linear transformation (Figure 2d), uses the message-passing output to determine two logits for each membrane representative corresponding to the odds of a flip. Finally, a fixed decoding rule aligns these logits using the syndrome, and combines them within each class to predict the flip of a reference representative (Figure 2e). These flips can then be used to obtain a correction or update the Pauli frame.

![](images/fb09d96c9c6966d17e6f9468706a9a02010d2fc4c441dc9290cabdffd77b96cd.jpg)  
FIG. 2: High level model structure. Refer to Figure C1 for a more detailed schematic. Components with learned parameters are shaded. From left to right: (a) Syndrome and erasure bits are embedded as check states and error-mechanism states, respectively. (b) Through a series of message-passing steps, check and error-mechanism states are updated by exchanging information on the Tanner graph. (c) A perceiver then updates membrane states for each membrane representative using the error-mechanism states. For the membranes, there are no observations that can be embedded, so these states are initially set to a single learned seed vector (see Appendix C). (d) A final learned linear layer maps each membrane state to a pair of logits. (e) Finally, predicted class flips are returned by combining representatives for each class in a syndromeaware manner (Section III C 1).

## B. Training the model

The task of training is to optimize the model parameters θ such that the internal distribution $p _ { \theta } ( f \mid \sigma , \varepsilon )$ closely approximates $p _ { \mathrm { D E C } } ( f \mid \sigma , \varepsilon )$ . In this section, we describe the loss function targeted during training, and some complications that arise for FBQC.

## 1. Training objective

We predict several equivalent membrane representatives per logical class, treating each as a separate training target. Specifi cally, we target the marginal distribution for each representative (indexed by µ)

$$
p _ { \mathrm { D E C } } ( f _ { \mu } \mid \sigma , \varepsilon ) = \sum _ { f ^ { \prime } : f _ { \mu } ^ { \prime } = f _ { \mu } } p _ { \mathrm { D E C } } ( f ^ { \prime } \mid \sigma , \varepsilon ) .\tag{6}
$$

These marginals provide the per-representative flip probabilities required by our decoding rule, without requiring the model to represent the full joint distribution. The decoder represented by these marginals makes the optimal decision per-membraneclass. This differs from the optimal joint decision of Equation (4) when correlations in logical errors across different classes are present, but this is small for many protocols of interest, such as topological codes.

Deviation of the model distribution from $p _ { \mathrm { D E C } } ( f _ { \mu } \mid \sigma , \varepsilon )$ is measured by the conditional cross-entropy [100], averaged over representatives and observed inputs

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \theta ) = - \frac { 1 } { K \times R } \sum _ { \mu = 1 } ^ { K \times R } \mathbb { E } _ { ( f , \sigma , \varepsilon ) \sim p _ { \mathrm { p g c } } } \left[ \log p _ { \theta } ( f _ { \mu } \mid \sigma , \varepsilon ) \right] , } \\ { \displaystyle = - \frac { 1 } { K \times R } \sum _ { \mu = 1 } ^ { K \times R } \sum _ { \sigma , \varepsilon } p ( \sigma , \varepsilon ) \sum _ { f _ { \mu } \in \mathbb { F } _ { 2 } } p _ { \mathrm { p g c } } ( f _ { \mu } \mid \sigma , \varepsilon ) \log p _ { \theta } ( f _ { \mu } \mid \sigma , \varepsilon ) , } \end{array}\tag{7a}
$$

(7b)

where $p ( \sigma , \varepsilon )$ is the distribution of syndromes and erasure patterns induced by the error model. The unique minimum of Equation (7) is $p _ { \mathrm { D E C } } ( f _ { \mu } \mid \sigma , \varepsilon )$ , and we optimize θ using a gradient-based optimizer [101]. The resulting decoder makes the optimal decision (within the parametrized models) for each membrane representative separately. In this sense, a model trained on Equation (7) approximates the optimal decoder, and unlike the most-likely-error decoder of Equation (5) it accounts for the degeneracy of errors.

## 2. Training strategy

The primary challenge in training our decoder for an FBQC error model is the prevalence of erasures, which can obscure information about membrane flips. When training begins with a strong error model, optimization can stall at a loss function value of approximately log(2), where it predicts that each membrane has a 50 % chance of being flipped, regardless of the observed input. To avoid this, we start training with lower error rates (small x in Equation (3)), and anneal up towards the desired operating point. This helps the model learn from more informative examples first. Along with the error rate, we vary several other training parameters as training progresses, via a training curriculum.

Training of the models presented in Section IV was performed on AMD Instinct<sup>TM</sup> MI350X GPUs. In Table C1, we provide details of the training configurations used for the protocols at various distances, and the required training times. As an example, the decoder for distance 16 LLD only required 108 h across 2 GPUs.

For more details on training, see Appendix C 2.

Sparse attention kernels. We make use of custom attention and aggregation kernels. These use the sparse-structure of the Tanner graph to avoid the construction of large temporary tensors. This reduces memory usage and data movement, helping increase training throughput and making larger protocols or batches practical. The attention calculation itself is unchanged. These kernels are discussed in Appendix C 2 a.

## C. Multiple membrane representatives

Using multiple equivalent membrane representatives benefits both training and decoding. Each representative contributes a separate term to the training loss, providing training signals from different membrane supports. Combining their syndrome aligned predictions can also improve decoding accuracy, as described in Section III C 1.

For the protocols presented in Section IV, we construct the representatives within each class by geometric translation.

## 1. Decoding rule

The learned portion of our model outputs a flip probability per-membrane-representative. We now describe how these are collated into per-class flip predictions.

Within each equivalence class κ under Equation (2), we designate one representative $\mu _ { \varkappa }$ as the reference. For all other representatives $\mu \sim \mu _ { \varkappa }$ , we compute the vector $t _ { \mu }$ such that

$$
\Lambda _ { \mu _ { \varkappa } } = \Lambda _ { \mu } + t _ { \mu } H ,\tag{8}
$$

which we refer to as the transition checks. The syndrome $\tau _ { \mu }$ of these checks, the transition syndrome, necessarily obeys

$$
f _ { \mu _ { \varkappa } } = f _ { \mu } + \tau _ { \mu } ,\tag{9}
$$

so the transition syndrome transports a prediction from µ to $\mu _ { \varkappa }$ . Writing $\lambda _ { \mu } ( b )$ for the logit of outcome $b \in \{ 0 , 1 \}$ of membrane $\mu ,$ the decoder then combines the representatives of each class by summing their aligned logits,

$$
\hat { f } _ { \varkappa } = \underset { b \in \{ 0 , 1 \} } { \arg \operatorname* { m a x } } \sum _ { \mu \in \varkappa } \lambda _ { \mu } \left( b + \tau _ { \mu } \right) .\tag{10}
$$

Since the difference of the two logits of a membrane is its predicted log-odds, Equation (10) adds the log-odds of all representatives after aligning them to the reference frame. We refer to this as the syndrome-aligned vote.

We note that this rule is not unique. One could, for example, choose the representative µ within each class whose prediction has the smallest entropy, and assign ${ \hat { f } } _ { \varkappa } : = f _ { \mu }$ . In practice, we observe a substantial reduction in LER when using the syndrome aligned vote.

## D. Relation to previous neural decoders

Neural decoders have been trained for surface codes with generative models [44, 45], feed-forward, convolutional and recurrent networks [46–52], graph neural networks that pass messages on the detector graph [53, 66, 67, 102], as learned variants of belief propagation [54], and the convolutional decoder Cascade [68]. Transformer approaches include Transformer-QEC [63], the autoregressive generative decoder qecGPT [65], and the recurrent AlphaQubit decoders [27, 70]. For classical codes, the Error Correction Code Transformer of Ref. [56] masks self-attention according to the parity-check matrix, and the same idea has been applied to quantum codes in the QECCT decoder [57]. The decoder of Ref. [58], which PRISM builds upon, masks attention between check tokens through shared error mechanisms, processes syndrome rounds recurrently, and reads out logica flips with a cross-attention decoder (with similar ideas in Ref. [64]). The POLYMECHANON decoder of Ref. [59] builds a graph of detectors, error mechanisms and logical observables from the detector error model. Node features are constructed from the detector-error model and the observed syndrome. The graph is updated with gated message passing and a recurrent state update, and reads out one prediction per logical observable.

![](images/f31e93d92886cc9d8517e80c43d44dc1b2bc616715b9e289e443c29b479306b1.jpg)

![](images/6ecf8650cbf795fdf9cc28aa6415b95e6c10b711a75b748441823038ae39c8f6.jpg)

![](images/dc1a5054ca44b385b82e9a8e746194b0ddb6b0eec58af4412608f48fd7979e7d.jpg)  
FIG. 3: Performance of the trained PRISM for the distance d periodic decoding block decoding problems. For each decoding problem, a reference decoder is chosen. For the six-ring, it is MWPM, for LLD and CC8 it is ILP with a BP pre-decoder to handle easy shots, while for the 72, 12, 6 and 144, 12, 12 BB codes, it is BP-OSD. Logical error rates are reported if any logical membrane is incorrectly flipped. Model hyperparameters provided in Table C1. Triangle markers denote upper-bounds where no logical errors were observed.

Our model differs from these in several respects. Firstly, it processes the full spacetime block in one pass. It represents error mechanisms explicitly as tokens (as is also done in POLYMECHANON), so that the error mechanisms can be used as input for the erasure flags. In contrast with gated message passing, the softmax of the sparse attention makes the mechanisms adjacent to a check compete to explain its flip. It reads out logical information through aggregations over membrane supports rather than global attention. At fixed width and depth, the number of trainable parameters is independent of protocol size. Model width and depth can be tuned to trade accuracy for speed. Finally, it predicts and combines several equivalent representatives of every logical class, which improves both training (Section III C) and decoding accuracy. We note that utilizing multiple membrane representatives was also studied very recently in Ref. [60], with similar ideas in Refs. [61 and 62].

## IV. DECODER PERFORMANCE

We now evaluate the full-precision decoders on the periodic decoding blocks for the six-ring, LLD, CC8, and two bivariate bicycle (BB) fusion networks under the noise model of Equation (3).

We compare their LER with performant conventional decoders for each instance as a benchmark. The six-ring network has a graphlike Tanner graph, so MWPM can be used. Neither the LLD nor the CC8 networks have Tanner graphs suitable for MWPM, so we utilize an ILP based decoder [103]. For increased throughput, we utilize a BP based pre-decoder to rapidly decode ‘easy shots, and only fall back to ILP when BP fails. For the BB networks, we benchmark against a BP+OSD decoder [104].

For each fusion network and distance, we train an ML decoder. For details on training, see Appendix C 2.

We find that PRISM has high accuracy—it matches the benchmark decoder across the error domain, and outperforms it at lower error rates. In particular, the LLD d = 24 run was performed on 100 8xMI355X nodes on the LUX supercomputer, in under 24 h. It returned 0 logical errors after 30 billion trials at $x = 0 . 0 1 2$

## V. REAL-TIME DECODING

In addition to accuracy, we must ensure the decoder is sufficiently fast to reach the throughput and latency requirements of real-time decoding. In this section we consider an accelerator implemented on an FPGA designed for PRISM inference, and describe co-design of the decoder architecture and hardware implementation to reduce inference latency while minimizing any loss in accuracy. For a discussion on the tradeoff between decoder accuracy and latency, we refer to Ref. [105] and references therein. We now turn to a discussion on hardware acceleration. For an introduction to neural-network hardware acceleration, including dataflow architectures, reduced-precision arithmetic, and exploitation of sparsity, see Ref. [106].

We consider three classes of optimization where model architecture and hardware are co-designed.

• Model parameters are quantized and the sparsity within the structure of the model is fully exploited, as described in Sections V A and V C.

• We map the model onto a pipeline of data-parallel dedicated hardware blocks, each storing weights locally with activations streaming between them. As discussed in Section V C, this enables a spatial mapping of PRISM that combines pipeline parallelism across processing stages with data parallelism across independent data groups.

• We develop specialized hardware for key components of the architecture such as the sparse cross-attention of PRISM.

FPGAs have many properties that make them a good fit for PRISM inference. First, QEC requires low latency and high throughput which maps well to a pipelined FPGA architecture that avoids traditional latency overheads such as dispatch, syn chronization, and unnecessary data transfer overhead. Second, PRISM can operate under aggressive datatype quantization, allowing matrix-operations to be efficiently lowered to lookup tables (LUTs). Third, we are able to modify and tailor the oper ator circuitry for this precise application. For example, the sparse cross-attention hardware blocks are optimized for the PRISM cross-attention sparsity, with more details found in Section V C 1. Finally, the post-quantization model weights can fit entirely within the SRAM of the FPGA, greatly improving achievable memory bandwidth.

We demonstrate, on the FPGA architecture, that a latency of 6.59 µs is possible for the benchmark d = 8 periodic decoding block for the LLD network. Recall, this is a $d \times d \times d$ decoding problem, which for comparison, is roughly equivalent to that of a distance 8 circuit-based surface code over 8 syndrome extraction rounds. We project that sub 10 µs decoding is possible for distances of interest for utility scale computations with further improvements to the compute fabric [107] and ML decoding architecture. We present the quantization-aware training (QAT) process in Sections V A and V B, the accelerator architecture and its mapping onto FPGAs in Sections V C and V D, and the resulting resource utilization, latency and throughput in Section V E.

## A. Decoding with low-precision arithmetic

To map the decoder architecture to an efficient FPGA implementation, we reduce the model’s numerical precision, referred to as quantization. In addition, we make use of FPGA-efficient arithmetic, including approximations to softmax and layernorm, and efficient dot-product computations [108]. Together, these adaptations are designed to strike a balance between implementation cost and decoder accuracy. We refer to the resulting decoder as the quantized PRISM, or Q-PRISM.

The Q-PRISM QAT process is implemented in Brevitas [109]. The trained quantized decoder can then be exported as a Quantized Open Neural Network Exchange (QONNX) graph [110] which specifies the quantizers and hardware operators. The QONNX graph is then input to the Framework for Fast, Scalable Quantized Neural Network Inference (FINN) [111] which optimizes the graph and lowers it to FPGA hardware blocks, as described in Section V C. We overview the primary changes to the quantized decoder in the remainder of this subsection, and in Section V C, and refer to Appendix D for more details.

Quantized weights and activations. For Q-PRISM, matrix weights take ternary values from $\{ - s , 0 , + s \}$ where $s > 0$ is a learned scale-factor, and are encoded using two bits. As detailed in Figure D1, activation precision varies throughout the design but is predominantly either two or four bits, with some intermediate quantities using wider precision. We write $\mathrm { W } _ { w } \mathrm { A } _ { a }$ for a model with w-bit matrix weights and a-bit activations.

During training, we simulate reduced-precision arithmetic using fake quantization. Every tensor that the hardware stores at reduced precision is assigned a quantizer. These quantizers appropriately round their assigned tensor via Brevitas [109], and ensure that every matrix operation has the appropriate, reduced, precision. The network is then trained under these precision constraints. More details of this procedure can be found in Appendix D, and in particular Figure D1.

![](images/f3c53f1d5b410e2ff1668c9e588a1a0a669e63bdcb064de80d6a4138dbddec66.jpg)  
FIG. 4: Logical error rates for quantized models for the $d = 8 \mathrm { L L D }$ protocol. These models were trained in a quantization-aware manner. The quantized $d _ { \mathrm { M } } = 1 2 8$ decoders use eight- or four-bit weights and activations while retaining floating-point normalization, attention, activation functions and residual streams. For reference, we include two unquantized points. The FP32 control (black) uses the PRISM architecture (with no quantization). Purple reference markers show the data from PRISM decoder using BF16 mixed inference, duplicated from Figure 3b.

Efficient arithmetic. We modify the operations used by the model to align with the arithmetic supported efficiently by the FPGA. In particular, for Q-PRISM, the sparse attention, layer normalization and feed-forward steps described in Appendix C 1 are constructed to use approximations built around additions, comparisons, and bit shifts. The floating-point operators that these approximate would consume the limited DSP resources available, and limit parallelism. The modified softmax and layernorm functions can be found in Appendix G, both of which can be implemented using FPGA logic. Additionally, the nonlinear GELU function (Equation (C6)) is replaced by

$$
{ \mathrm { R e L U } } ( x ) : = \operatorname* { m a x } ( 0 , x ) .\tag{11}
$$

These approximations are included in the forward pass during QAT, so the model is trained with the arithmetic used for inference.   
Their effect on decoding accuracy is examined in Figure D4 and Appendix D 3.

Quantization-aware training. QAT extends standard gradient descent by providing gradient surrogates for the nondifferentiable functions found in quantization operations. In this work, the straight-through estimator (STE) [112] is used to provide these gradient surrogates. The introduction of quantization operations can affect the training dynamics, and as such, various hyperparameters may need to be adjusted compared with their unquantized counterparts. In particular, we found the learning rate needed adjusting to improve convergence when performing QAT.

## B. Quantized model performance

In Figure 4 we report logical error rate accuracy results for the quantized decoder. These simulations include quantization and retain a floating-point residual stream. FPGA resource allocation and timing results are assessed in Sections V D and V E.

We focus on the same protocol as the FPGA implementation, the LLD block at distance $d = 8 .$ The quantized models have $Q = 4$ message-passing blocks, followed by the one-sided interface layer, and $P = 2$ perceiver blocks. We compare a $\mathrm { W _ { 2 } A _ { 4 } }$ model with $d _ { \mathrm { M } } = 1 6 , \mathrm { W _ { 2 } A _ { 2 } }$ models with $d _ { \mathrm { M } } = 3 2$ and $d _ { \mathrm { M } } = 4 8$ , and $d _ { \mathrm { M } } = 1 2 8$ models at $\mathrm { W _ { 4 } A _ { 4 } }$ and $\mathrm { W _ { 8 } A _ { 8 } }$ . Details of training, model size, and supporting experiments are given in Appendix D.

On the distance-8 LLD benchmark, the $d _ { \mathrm { M } } = 1 6 , \mathrm { W _ { 2 } A _ { 4 } }$ model achieves a logical error rate within 15–22 % of the $d _ { \mathrm { M } } = 4 8$ $\mathrm { W _ { 2 } A _ { 2 } }$ model with two-bit activations, using approximately eight times fewer parameters. These results motivate choosing model size, precision, and training procedure jointly. While the quantized models are less accurate than the reference decoders in Figure 3, they can offer a significant reduction in decoding time and reduced hardware cost as discussed in Section V E. To evaluate the hardware tradeoffs between model size and activation precision, we prepare several models for the FINN flow (Section $\mathrm { V C ) } { : } \mathrm { a W _ { 2 } A _ { 4 } }$ model with $d _ { \mathrm { M } } = 1 6$ , and $\mathrm { W _ { 2 } A _ { 2 } }$ models with $d _ { \mathrm { M } } = 3 2$ and $d _ { \mathrm { M } } = 4 8$

Ultimately, we select the $d _ { \mathrm { M } } = 3 2 , \mathrm { W _ { 2 } A _ { 2 } }$ model as our target quantized model architecture due to the compromise between LER, and the ability to implement highly-efficient computation leveraging the two-bit weights and activations.

Refer forward to Appendix D for more detail on the QAT methodology and results.

![](images/9b0e2bb334c556bf41d74621ad4658d278256b6e179fcbd9811fa1395f48e7a5.jpg)  
FIG. 5: Fully-unrolled dataflow mapping of Q-PRISM. The erasure and syndrome embeddings are fed into the accelerator with MMV<sub>erasure</sub>, $\mathrm { M M V _ { s y n d r o m e } , M M V _ { m e m b r a n e } }$ vectors at a time. Vectors flow from left to right in a pipeline parallel dataflow. Each layer has a spatially instantiated hardware mapping (for example Figure F2) that performs the operations of that layer and then produces output vectors that are then fed to subsequent layers.

## C. Q-PRISM accelerator architecture

FINN [111], an open-source framework for FPGA-based ML inference, was used to build an aggressively quantized and sparse fully unrolled dataflow implementation of Q-PRISM on an FPGA. It provides a collection of highly optimized hardware blocks, and a compiler that performs graph-level optimizations, such as streamlining [113]. The compiler maps models represented in ONNX [114] onto interconnected hardware blocks. FINN receives a QONNX file provided by Brevitas [109], which is used in the QAT flow as specified in Section V A. FINN directly provides the quantization support and fully-unrolled dataflow which are key to achieving the latency and throughput requirements of QEC. Sections V C 1 and V D discuss co-design across the Q-PRISM architecture and the accelerator architecture where we both build highly-specialized hardware for specific components of Q-PRISM, as well as make modifications to the ML decoder architecture to map better to the compute fabric. The realization of the Q-PRISM FPGA mapping can be seen in Figure 5.

The Q-PRISM accelerator exploits data parallelism both across token vectors and within individual operations. Inter-operator data parallelism is leveraged by processing multiple vectors at the same time via parallel computing lanes. An accelerator configuration has a matrix multiple-vector (MMV) factor for every input stream $( \mathrm { M M V _ { e r a s u r e } , M M V _ { s y n d r o m e } , }$ and $\mathrm { M M V } _ { \mathrm { m e m b r a n e } } )$ which defines the number of lanes for each stream of computation. High inter-operator data parallelism is key for low-latency high-throughput Q-PRISM inference due to the large input stream size.

Within each operator, additional data parallelism accelerates the processing of single vectors. For example, the FINN matrixvector-activation unit (MVAU) has single-instruction, multiple-data (SIMD) and processing element (PE) factors that determine how many input components and output components, respectively, are processed in parallel. Together, these factors determine the number of cycles required for a matrix-vector multiplication. The initiation interval (II) of a circuit is defined by the number of clock cycles between two subsequent inputs. In the Q-PRISM FINN implementation, we leverage the maximum amount of intra-operator data-parallelism allowed by our resource budget, which is necessary to maintain an II of one for all blocks in the design except cross attention, which will be discussed in Section V C 1.

In addition to data parallelism, dedicated hardware for each operator enables pipeline parallelism within and between layers. Different operators and layers can therefore process different vectors concurrently. Many operators can begin processing before the complete input stream has arrived, thus overlapping their computations and, once their pipelines are filled, produce an output stream at every clock cycle. An important exception is cross attention, which introduces dependencies across tokens. Section V C 1 describes the hardware and schedule of the cross attention block designed to minimize resulting pipeline stalls.

## 1. Cross-attention block architecture and schedule

As mentioned in Section V C, the sparse cross-attention presents a challenge to maintaining an II of one, since it introduces inter-token dependence by design. A naive implementation would require draining all tokens from the input pipeline in order to meet inter-token dependencies before any cross-attention computation can occur. Such a drain and fill would charge additional latencies in proportion to the number of tokens, scaling with the size of the code and crippling any chance of real-time performance.

Fortunately, the decoder uses a sparse cross-attention in which tokens attend to one another according to their connectivity in the Tanner graph. Thus a query must only wait for the keys connected to it. Transformers are permutation equivariant [115] which means that permuting the input permutes the output while keeping each token equivalent. We may therefore choose to order tokens in a way that uses the sparse mask to minimize stalls. Queries arrive via a residual connection while keys lie on the critical path. So, while a layer’s key order is fixed by the output of the previous layer, the queries have slack with which the

![](images/1dc4e41c2718cfab1aa47e408767c25a91752ff44b3cbdb33d653e1e4acbb78a.jpg)  
(a) First cross-attention mask: Error-mechanism tokens attend to check tokens.

![](images/b15a952092e24b4bff2bdd3f45869c80f48b64b0ecd077a1f608745c817a580d.jpg)  
(b) Second cross-attention mask: Check tokens attend to error-mechanism tokens.  
FIG. 6: Cross-attention masks used in the first message passing layers of PRISM $d = 8 \mathrm { L L D }$ . Hubs and stars are processed in parallel. The colors denote the order that input data is streamed into the cross attention block. Data is output as it is processed. Star-check connections form compact local blocks, while hub-check connections link mechanisms across stars. A more detailed discussion of the sparse attention schedules can be found in Appendix E.

tokens may be reordered layer to layer.

For the LLD protocol, it is convenient to partition the check tokens into two sets. We call these hubs, consisting of weight-36 cuboctahedral checks, and stars, consisting of groups of weight-12 octahedral and weight-6 triangle checks (see Figure B1b). Figure 6 shows the masks for the two cross-attention operations, with tokens ordered according to their hardware processing order. The protocol can be projected into a three-dimensional structure, in which stars are connected internally, and separated from other stars by hubs in a periodic cubic lattice. The majority of the attention computation takes place between tokens within the same star, so we permute the tokens to group them together. To address the question of ordering the stars, we appeal to the cubic geometry of the star-hub connectivity and schedule them by walking each axis of the cube in order to preserve spatial locality. This walk is shown in Figure E2 in Appendix E. As a consequence of the periodic boundaries, we must shift the queries in every axis of the cube each layer such that they match the order of the previous layer’s output.

Error mechanisms are not shared between stars, so the attention for any given star can be computed as soon as its incident mechanisms are available. This does not delay the computation for other stars, avoiding the need for a whole-frame synchronization barrier. We defer a detailed discussion of our framework around building high-performance sparse cross-attention schedules to Appendix E. We refer the reader to Ref. [95] for a more detailed description of the topology of such codes, which determines the attention mask.

Each cross-attention block combines banked on-chip key/value storage with a scheduled gather stage and pipelined dotproduct, modified softmax, and weighted-value computations. We partition the syndrome tokens into separate star and hub streams, with independent compute and output paths for the two classes in the syndrome-query attention. The scheduler issues each query group once its required keys and values are available, allowing attention computation to overlap with upstream token production and downstream processing without requiring a whole-frame synchronization barrier.

## D. Hardware mapping

We map Q-PRISM with $( Q = 4 , P = 2 )$ , trained using the described $\mathrm { W _ { 2 } A _ { 2 } }$ QAT flow onto an FPGA. Aggressive quantization allows the accelerator to exploit the large 8,460,288 LUT capacity of the AMD VP1902 FPGA. Where possible, we implement arithmetic using LUTs, reserving the more limited digital signal processing (DSP) slices (6,864) for higher-precision operations as discussed in Section V A. Table II summarizes the device’s available resources.

The dot products in FINN’s matrix-vector-activation units (MVAUs) use the compressor tree generator introduced in Ref. [108], which provides low-latency, resource-efficient LUT implementations. The FINN compiler also absorbs floating point scaling and bias operations into thresholding operations through streamlining, allowing much of the computation to remain in two-bit integer arithmetic without explicitly constructing and re-quantizing floating-point intermediate values [113]. These arithmetic optimizations, together with the VP1902’s resource capacity, enable the pipeline and data parallelism needed to meet the latency and throughput targets.

TABLE II: FPGA resource utilization, full-frame latency, and throughput of three different ML decoder configurations. Configurations are denoted $\langle d _ { \mathrm { M } } , \mathrm { [ M M V _ { e r a s u r e } , M M V _ { s y n d r o m e } ] } \rangle$ for model-dimension d<sub>M</sub> and parallelism $[ \mathrm { M M V _ { e r a s u r e } , M M V _ { s y n d r o m e } } ] .$ 32, [5, 1] 250 MHz is obtained from a full network mapping and 400 MHz is obtained from individual-layer implementations as discussed in Appendix F. We report the cycles from the 250 MHz design. Hitting timing at 400 MHz required 108 additional clock cycles. This is reflected in the corre sponding reported latency and throughput values. [24, 5] resource utilization is generated from synthesis while [5, 1] resource utilization is from placed-and-routed designs. Both MMV [24, 5] studies assume the same board-targeted operating frequency as 32, [5, 1] .
<table><tr><td></td><td></td><td colspan="5">Resource utilization</td><td colspan="5">Frequency, latency, and throughput</td></tr><tr><td>Scope</td><td>Configuration</td><td></td><td>LUTs</td><td>Flip-flops</td><td>BRAM</td><td>URAM</td><td>DSP</td><td>Frequency (MHz)</td><td>Latency (cycles)</td><td>Latency (μs)</td><td>Throughput (µs)</td></tr><tr><td>VP1902 available</td><td></td><td></td><td>8,460,288</td><td>16,920,576</td><td>6,808</td><td>2,200</td><td>6,864</td><td></td><td>一</td><td></td><td></td></tr><tr><td>Reference configuration</td><td>(32, [5, 1]}</td><td></td><td>1,936,332</td><td>2,397,547</td><td>746</td><td>218</td><td>384</td><td>250-400</td><td>2,528</td><td>6.59 - 10.11</td><td>1.56 - 2.39</td></tr><tr><td>Higher parallelism</td><td>(16, [24, 5]}</td><td></td><td>3,036,424</td><td>2,594,260</td><td>241</td><td>192</td><td>192</td><td>250-400</td><td>1,248</td><td>3.39 - 4.99</td><td>0.76 - 1.12</td></tr><tr><td>configurations</td><td>(32, [24, 5]}</td><td></td><td>6,904,201</td><td>5,894,654</td><td>2,075</td><td>384</td><td>384</td><td>250-400</td><td>1,291</td><td>3.5 - 5.16</td><td>0.78 - 1.16</td></tr></table>

The weight sparsity present within Q-PRISM enables additional optimizations to the matrix operations in the design. We specialize each MVAU to its particular weight matrix, eliminating zero-weight terms from the generated hardware. The small range in the INT2 quantization allows further simplifying of the remaining arithmetic. Additionally, accumulator circuitry is tailored to the size and datatype of each MVAU. Together, these optimizations reduce the size of the accumulation networks and can shorten their combinational paths. Weight sparsity therefore translates directly into a lower compute-fabric utilization, making greater parallelism practical within the FPGA’s resource budget.

When mapping a transformer to an FPGA, FINN will typically map high-precision operations such as softmax, normalization, and residual element-wise operations to DSPs. Given the amount of data parallelism in the architecture, mapping layernorm and softmax in this way would require more DSPs than available on the VP1902. We refer the reader to Appendix G for a more detailed description of the modifications to approximate softmax and normalization which allowed the use of LUTs for these operations.

We target two frequencies when mapping the Q-PRISM accelerator onto the VP1902. The first frequency is 250 MHz which the design easily hits. We additionally map a single message-passing and perceiver layer at 400 MHz to show that it is possible to run the design at 400 MHz. Due to the pipelined design and available resource budget on the VP1902, the total frequency of the network can be well-predicted by the component frequencies, and we leave the full mapping at 400 MHz for future work. We highlight both results in Section V E. A more detailed discussion of the timing optimizations leveraged in our FPGA mapping can be found in Appendix F.

## E. Results

We leverage AMD Vivado<sup>TM</sup> 2026.1 to map the Q-PRISM accelerator onto the VP1902 FPGA. The latency, throughput, and resource utilization of three configurations of the accelerator can be seen in Table II. Configurations are denoted $\left. d _ { \mathrm { M } } , \mathrm { [ M M V _ { e r a s u r e } , M M V _ { s y n d r o m e } ] } \right.$ for model-dimension $d _ { \mathrm { M } }$ and parallelism $\mathrm { \small { [ M M V _ { e r a s u r e } , M M V _ { s y n d r o m e } ] } }$

We present two groups of configurations. The reference configuration, 32, [5, 1] , can achieve the QEC-relevant latency and throughput, and fits on the VP1902. In addition, the model trained through the QAT flow can achieve LERs competitive with the unquantized models.

The higher parallelism configurations, 16, [24, 5] and 32, [24, 5] , are presented to highlight the significant latency and throughput improvements that would be made possible through either reducing model dimension on the same FPGA ( 16, [24, 5] ) or using the same network on a future FPGA with sufficient resources ( 32, [24, 5] ). The reason for the [24, 5] configuration comes from the token-ratio in the intra-star attention mask and is further discussed in Appendix E and Figure E3.

The operating frequency of 250 MHz for the 32, [5, 1] configuration is obtained from a full network mapping and 400 MHz is obtained from individual-layer implementations as discussed in Appendix F.

## 1. Resource utilization

Table II shows that the 32, [5, 1] accelerator fits comfortably within the resources of the VP1902 consuming 22.9% of the LUTs, 14.2% of the flip-flops, 11% of the BRAM, 9.9% of the URAM, and 5.6% of the DSPs. This leaves ample room to increase performance through higher parallelism designs or to add additional functionality. The 16, [24, 5] and 32, [24, 5] configurations illustrate two feasible paths towards additional parallelism that provide substantial latency and throughput improvements as discussed in Section V E 2.

![](images/155626efe8fe53a3205c4404ba85989cdd17ec5e1d7b0d8384c0b0e214dd4235.jpg)  
FIG. 7: Measured register-transfer level (RTL) timeline for the 32, [5, 1] embedded core, with $Q { = } 4$ and $P { = } 2$ . Background bands use the layer colors of the physical layout depicted in Figure F1. Bars denote processing in a particular stage of a layer. Inter and intra-pipeline parallelism are highlighted by the vertical overlap of bars. All blocks, including attention, can produce results while remaining inputs are still arriving, overlapping startup with upstream output delivery. The core completes in 2,528 clock cycles.

First, with model improvements allowing a $d _ { \mathrm { M } } = 1 6$ design to achieve sufficient LER, a mapping of MMV [24, 5] would be possible on the VP1902 only consuming 35.9% of the available LUTs. Second, sufficiently increasing FPGA capacity would allow the existing $d _ { \mathrm { M } } = 3 2$ model to be mapped with the parallelism of [24, 5]. There are therefore multiple mechanisms to enable designs with significantly greater parallelism than the initial 32, [5, 1] design currently mapped to achieve even higher performance if necessary.

## 2. Latency

Table II shows the latency and throughput of the three ML decoder accelerator configurations. The target 32, [5, 1] design has a latency of 2,528 cycles and at 250 MHz can achieve an end-to-end latency of 10.1 µs and a throughput of 2.39 µs and at 400 MHz can achieve an end-to-end latency of 6.59 µs and a throughput of 1.56 µs. We note that hitting 400 MHz required adding 108 cycles which is incorporated into the latency values.

A breakdown of the latency across the seven layers of the network is shown in Figure 7. This is achievable due to the substantial pipeline parallelism within the accelerator architecture, which allows significant overlap of compute while processing a single frame. Due to the extensive token independence in the operations, and the II of one, a vector progresses through the circuit every clock cycle except for when going through a sparse cross-attention.

Figure 7 illustrates the overlap enabled by the optimized sparse cross-attention hardware and schedule. Tokens are reordered to reduce pipeline stalls, and each query group is processed as soon as its required keys and values are available. In the first message-passing layer, cross-attention outputs (cyan) are produced while inputs (orange) are still arriving. The two crossattention operations within each message-passing layer (mechanism ← check and mechanism → check) also overlap. Relative to a baseline that waits for the complete input stream at each cross-attention block before starting computation, we estimate a 1.77 speedup.

The schedule in Figure 7 also highlights the high-throughput achievable by the design. When the first message passing layer is complete, the subsequent input can be provided. At steady state, almost the entire circuit is concurrently processing data. By overlapping successive frames within the same accelerator, the design provides over 4 the throughput of processing isolated frames sequentially, allowing the consolidation of what would be multiple decoding devices into a single device.

In addition to the 32, [5, 1] accelerator configuration, Table II also presents two configurations with significantly higher parallelism, the 16, [24, 5] and 32, [24, 5] mappings. Compared with the MMV [5, 1] configuration, the MMV [24, 5] configurations provide approximately five times as many parallel token-processing lanes. The large number of input tokens limits throughput and contributes substantially to latency in the 32, [5, 1] design. Streaming these tokens through the message-passing layers takes time, and dependencies between tokens introduce pipeline stalls while cross-attention blocks wait for the required data.

The [24, 5] architecture can achieve latencies (throughput intervals) of 3.39 µs to 4.99 µs (0.76 µs to 1.12 µs) and 3.5 µs to 5.16 µs (0.78 µs to 1.16 µs) for $d _ { \mathrm { M } } = 1 6$ and $d _ { \mathrm { M } } = 3 2$ , respectively. This is a significant improvement in latency and throughput over the target $\langle 3 2 , [ 5 , 1 ] \rangle$ design which would be made possible with either a higher-capacity FPGA to fit the 32, [24, 5] accelerator, or improvements to the model to run a $d _ { \mathrm { M } } = 1 6$ model at sufficient LER.

We note that the latency is measured from the first input of the decoding block to the complete decoded output of the FPGA, which includes input loading. For real-time decoding, measurement outcomes are streamed to the decoder, and taking advantage of this can further reduce the latency.

These results show that ML decoders like PRISM can be implemented in hardware accelerators with the low latency required for utility-scale quantum computation. For the larger distances that are expected to be required, we expect sub 10 µs latencies to remain possible, through a combination of improvements to decoder architecture, quantization, and improvements to the hardware accelerators themselves.

## VI. OUTLOOK

We introduce an ML decoder, PRISM, that offers the accuracy, throughput, and latency required for fault-tolerant quantum computation. In particular, PRISM takes in information about erasures and has high accuracy against both Pauli and erasure errors for a variety of fault-tolerant protocols, based on surface codes, color codes, and Bivariate Bicycle codes, with logical error rate performance matching that of the ILP-based benchmark (for most-likely error decoding). In addition to accuracy, we show that real-time latency requirements can be met by implementing a quantized version for a distance-8 fault-tolerant protocol on an AMD VP1902 FPGA. We show that latency of 6.59 µs is possible for the full periodic decoding block, with an average per-block decode time of 1.56 µs. This latency is fixed and independent of input, meaning the quantum computation does not need to stall for slow decoder shots. We expect latencies below 10 µs to remain possible for the larger distances relevant to utility-scale quantum computations, with speedups on offer through improvements to the decoder architecture, quantization, deployment, and hardware accelerators themselves.

While we have focused on fault-tolerant protocols and noise models based on photonic FBQC, we emphasize that PRISM is a general-purpose and flexible decoder that can be used for general architectures and noise models. Furthermore, while we have focused primarily on decoding a $d \times d \times d$ fusion network as our benchmark, our approach is compatible with modular decoding [85]. In particular, the global decoding problem can be divided into smaller problems—each solvable by an ML decoder like PRISM—the outputs of which are then combined to give the global solution. This requires consideration of buffering and communication between decoding problems, and we leave the discussion of this to future work.

As PRISM outputs soft-information (estimates of the logical class probabilities), we can also make use of them for other fault tolerant applications, including postselection [96, 116], and dynamic biasing of errors [117]. Soft-information has been shown to be useful in determining the quality of a logical block or other fragment of a fault-tolerant computation, as has been shown, for example, in Refs. [68 and 118]. Another interesting avenue is to consider joint flip distributions in the decoder output. Thi may be particularly useful for high-rate codes, where correlations between distinct logical classes may be stronger than in the topological case.

[1] P. W. Shor, Fault-tolerant quantum computation, in Proceedings of 37th conference on foundations of computer science (IEEE) pp. 56–65, arXiv:quant-ph/9605011 (1996).

[2] D. Aharonov and M. Ben-Or, Fault-tolerant quantum computation with constant error rate, in Proceedings of the twenty-ninth annual ACM symposium on Theory ofcomputing, pp. 176–188, arXiv:quant-ph/9906129 (1997).

[3] A. Y. Kitaev, Quantum error correction with imperfect gates, in Quantum Communication, Computing, and Measurement (Springer, 1997) pp. 181–188.

[4] E. Knill, R. Laflamme, and W. H. Zurek, Resilient quantum computation: error models and thresholds, Proceedings: Mathematical, Physical and Engineering Sciences 454, 365, arXiv:quant-ph/9702058 (1998).

[5] J. Preskill, Reliable quantum computers, Proceedings: Mathematical, Physical and Engineering Sciences 454, 385, arXiv:quantph/9705031 (1998).

[6] D. Bluvstein, S. J. Evered, A. A. Geim, et al., Logical quantum processor based on reconfigurable atom arrays, Nature 626, 58, arXiv:2312.03982 (2024).

[7] A. Paetznick, M. P. da Silva, C. Ryan-Anderson, et al., Demonstration of logical qubits and repeated error correction with better-thanphysical error rates, arXiv:2404.02280 (2024).

[8] B. W. Reichardt, D. Aasen, R. Chao, et al., Demonstration of quantum computation and error correction with a tesseract code, arXiv:2409.04628 (2024).

[9] Google Quantum AI and Collaborators, Quantum error correction below the surface code threshold, Nature 638, 920, arXiv:2408.13687 (2025).

[10] PsiQuantum team, A manufacturable platform for photonic quantum computing, Nature 641, 876, arXiv:2404.17570 (2025).

[11] M. V. Larsen, J. E. Bourassa, S. Kocsis, et al., Integrated photonic source of Gottesman–Kitaev–Preskill qubits, Nature 642, 587 (2025).

[12] A. Lavasani, A. P. Reed, J. M. Dreiling, et al., Quantum error correction well below the color code threshold, arXiv:2610.24077 (2026).

[13] P. Panteleev and G. Kalachev, Degenerate quantum LDPC codes with good finite length performance, Quantum 5, 585, arXiv:1904.02703 (2021).

[14] N. P. Breuckmann and J. N. Eberhardt, Balanced product quantum codes, IEEE Transactions on Information Theory 67, 6653, arXiv:2012.09271 (2021).

[15] P. Panteleev and G. Kalachev, Asymptotically good quantum and locally testable classical LDPC codes, in Proceedings of the 54th annual ACM SIGACT symposium on theory ofcomputing, pp. 375–388, arXiv:2111.03654 (2022).

[16] S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, and T. J. Yoder, High-threshold and low-overhead fault-tolerant quantum memory, Nature 627, 778, arXiv:2308.07915 (2024).

[17] S. Tamiya, M. Koashi, and H. Yamasaki, Polylog-time-and constant-space-overhead fault-tolerant quantum computation with quantum low-density parity-check codes, arXiv:2411.03683 (2024).

[18] H. Yamasaki and M. Koashi, Time-efficient constant-space-overhead fault-tolerant quantum computation, Nature Physics 20, 247, arXiv:2207.08826 (2024).

[19] D. Litinski, Blocklet concatenation: Low-overhead fault-tolerant protocols for fusion-based quantum computation, arXiv:2506.13619 (2025).

[20] T. J. Yoder, E. Schoute, P. Rall, E. Pritchett, J. M. Gambetta, A. W. Cross, M. Carroll, and M. E. Beverland, Tour de gross: A modular quantum computer based on bivariate bicycle codes, arXiv:2506.03094 (2025).

[21] W. Yang, C. Duckering, and A. Dua, Designer Codes from GALA: Compact, Self-Dual, and Rate-1/2 QEC on Reconfigurable Atom Arrays, arXiv:2608.07431 (2026).

[22] K. Kasai, Breaking the orthogonality barrier in quantum LDPC codes, Quantum 10, 2205, arXiv:2601.08824 (2026).

[23] A. Bhardwaj, M. Ma, N. Meister, R. King, D. Bluvstein, J. Preskill, M. Cain, Q. Xu, and H.-Y. Huang, High-rate qLDPC processors, arXiv:2607.28795 (2026).

[24] C. Zhao, C. Duckering, A. Gu, N. Maskara, and H. Zhou, Towards ultra-high-rate quantum error correction with reconfigurable atom arrays, arXiv:2604.16209 (2026).

[25] B. M. Terhal, Quantum error correction for quantum memories, Reviews of Modern Physics 87, 307, arXiv:1302.3428 (2015).

[26] N. Delfosse and N. H. Nickerson, Almost-linear time decoding algorithm for topological codes, Quantum 5, 595, arXiv:1709.06218 (2021).

[27] J. Bausch, A. W. Senior, F. J. H. Heras, et al., Learning high-accuracy error decoding for quantum processors, Nature 635, 834, arXiv:2310.05900 (2024).

[28] O. Higgott and C. Gidney, Sparse Blossom: correcting a million errors per core second with minimum-weight matching, Quantum 9, 1600, arXiv:2303.15933 (2025).

[29] Y. Wu and L. Zhong, Fusion blossom: Fast MWPM decoders for QEC, in 2023 IEEE international conference on quantum computing and engineering (QCE), Vol. 1 (IEEE) pp. 928–938, arXiv:2305.08307 (2023).

[30] T. Hillmann, L. Berent, A. O. Quintavalle, J. Eisert, R. Wille, and J. Roffe, Localized statistics decoding for quantum low-density parity-check codes, Nature Communications 16, 8214, arXiv:2406.18655 (2025).

[31] B. Barber, K. M. Barnes, T. Bialas, et al., A real-time, scalable, fast and resource-efficient decoder for a quantum computer, Nature Electronics 8, 84, arXiv:2309.05558 (2025).

[32] A. Kell, D. Herr, D. B. Garc´ıa, A. Paler, D. Orsucci, F. Lazaro, and F. G. Herrero, Real-time BP decoding of a high-rate QEC code,´ arXiv:2610.41413 (2026).

[33] D. Bascones, A. S. Maan, V. Savin, and F. Garcia-Herrero, A Scalable FPGA Architecture for Real-Time Decoding of Quantum LDPC´ Codes Using GARI, arXiv:2605.01035 (2026).

[34] T. Maurer, M. Buhler, M. Kr¨ oner, F. Haverkamp, T. M¨ uller, D. Vandeth, and B. R. Johnson, Real-time decoding of the gross code¨ memory with FPGAs, arXiv:2510.21600 (2025).

[35] Y. Wu, N. Liyanage, and L. Zhong, Micro blossom: Accelerated minimum-weight perfect matching decoding for quantum error correction, in Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 639–654, arXiv:2502.14787 (2025).

[36] N. Liyanage, Y. Wu, S. Tagare, and L. Zhong, FPGA-based distributed union-find decoder for surface codes, IEEE Transactions on Quantum Engineering 5, 1, arXiv:2406.08491 (2024).

[37] L. Caune, L. Skoric, N. S. Blunt, et al., Demonstrating real-time and low-latency quantum error correction with superconducting qubits, Nature Communications 17, 7383, arXiv:2410.05202 (2026).

[38] X. Yang, X. Sun, Z. Wu, et al., Real-time surface-code error correction using an FPGA-based neural-network decoder, arXiv:2605.04892 (2026).

[39] P. Iyer and D. Poulin, Hardness of decoding quantum stabilizer codes, IEEE Transactions on Information Theory 61, 5209, arXiv:1310.3235 (2015).

[40] S. Bravyi, M. Suchara, and A. Vargo, Efficient algorithms for maximum likelihood decoding in the surface code, Physical Review A 90, 032326, arXiv:1405.4883 (2014).

[41] M. Walters and M. Turner, Minimum weight decoding in the colour code is NP-hard, arXiv:2603.04234 (2026).

[42] S. Gu, L. Wang, and A. Kubica, The color code, the surface code, and the transversal CNOT: NP-hardness of minimum-weight decoding, arXiv:2603.22064 (2026).

[43] N. Maskara, A. Kubica, and T. Jochym-O’Connor, Advantages of versatile neural-network decoding for topological codes, Physical Review A 99, 052351, arXiv:1802.08680 (2019).

[44] G. Torlai and R. G. Melko, Neural decoder for topological codes, Physical Review Letters 119, 030501, arXiv:1610.04238 (2017).

[45] S. Krastanov and L. Jiang, Deep neural network probabilistic decoder for stabilizer codes, Scientific Reports 7, 11003, arXiv:1705.09334 (2017).

[46] S. Varsamopoulos, B. Criger, and K. Bertels, Decoding small surface codes with feedforward neural networks, Quantum Science and Technology 3, 015004, arXiv:1705.00857 (2018).

[47] P. Baireuther, T. E. O’Brien, B. Tarasinski, and C. W. J. Beenakker, Machine-learning-assisted correction of correlated qubit errors in a topological code, Quantum 2, 48, arXiv:1705.07855 (2018).

[48] C. Chamberland and P. Ronagh, Deep neural decoders for near term fault-tolerant experiments, Quantum Science and Technology 3, 044002, arXiv:1802.06441 (2018).

[49] X. Ni, Neural network decoders for large-distance 2D toric codes, Quantum 4, 310, arXiv:1809.06640 (2020).

[50] K. Meinerz, C.-Y. Park, and S. Trebst, Scalable neural decoder for topological surface codes, Physical Review Letters 128, 080505, arXiv:2101.07285 (2022).

[51] S. Gicev, L. C. L. Hollenberg, and M. Usman, A scalable and fast artificial neural network syndrome decoder for surface codes, Quantum 7, 1058, arXiv:2110.05854 (2023).

[52] B. M. Varbanov, M. Serra-Peralta, D. Byfield, and B. M. Terhal, Neural network decoder for near-term surface-code experiments, Physical Review Research 7, 013029, arXiv:2307.03280 (2025).

[53] M. Lange, P. Havstrom, B. Srivastava, I. Bengtsson, V. Bergentall, K. Hammar, O. Heuts, E. van Nieuwenburg, and M. Granath, Data-¨ driven decoding of quantum error correcting codes using graph neural networks, Physical Review Research 7, 023181, arXiv:2307.01241 (2025).

[54] Y.-H. Liu and D. Poulin, Neural belief-propagation decoders for quantum error-correcting codes, Physical Review Letters 122, 200501, arXiv:1811.07835 (2019).

[55] E. Nachmani, Y. Be’ery, and D. Burshtein, Learning to decode linear codes using deep learning, in 54th Annual Allerton Conference on Communication, Control, and Computing, pp. 341–346, arXiv:1607.04793 (2016).

[56] Y. Choukroun and L. Wolf, Error correction code transformer, in Advances in Neural Information Processing Systems, Vol. 35, p. 38695–38705, arXiv:2203.14966 (2022).

[57] Y. Choukroun and L. Wolf, Deep quantum error correction, in Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, p. 64–72, arXiv:2301.11930 (2024).

[58] J. Blue, H. Avlani, Z. He, L. Ziyin, and I. L. Chuang, Machine learning decoding of circuit-level noise for bivariate bicycle codes, Quantum 10, 2149, arXiv:2504.13043 (2026).

[59] F. A. Astolfi and G. Pupillo, A code-agnostic graph neural network decoder from the detection error model, arXiv:2610.01683 (2026).

[60] Y. Guang, Y. Xiong, X. Gao, and Y. Chen, CoDecoder: Make quantum decoders up to 100x faster and even more accurate, arXiv:2610.28748 (2026).

[61] R. Toshio, K. Kishi, J. Fujisaki, H. Oshima, S. Sato, and K. Fujii, Decoder switching: Breaking the speed-accuracy tradeoff in real-time quantum error correction, arXiv:2510.25222 (2025).

[62] E. Egorov, R. Bondesan, and M. Welling, The end: An equivariant neural decoder for quantum error correction, arXiv:2304.07362 (2023).

[63] H. Wang, P. Liu, K. Shao, D. Li, J. Gu, D. Z. Pan, Y. Ding, and S. Han, Transformer-QEC: Quantum error correction code decoding with transferable transformers, arXiv:2311.16082 (2023).

[64] J. P. Bonilla Ataides, A. Gu, S. F. Yelin, and M. D. Lukin, Neural decoders for universal quantum algorithms, PRX Intelligence 1, 013008, arXiv:2509.11370 (2026).

[65] H. Cao, F. Pan, Y. Wang, and P. Zhang, qecGPT: Decoding quantum error-correcting codes with generative pre-trained transformers, arXiv:2307.09025 (2023).

[66] G. Hu, W. Ouyang, C.-Y. Lu, C. Lin, and H.-S. Zhong, Efficient and universal neural-network decoder for stabilizer-based quantum error correction, arXiv:2502.19971 (2025).

[67] A. S. Bhave, N. Choudhury, and K. Basu, HyperNQ: A hypergraph neural network decoder for quantum LDPC codes, in ICC 2026-IEEE International Conference on Communications (IEEE) pp. 1–7, arXiv:2511.01741 (2026).

[68] A. Gu, J. P. Bonilla Ataides, M. D. Lukin, and S. F. Yelin, Scalable neural decoders for practical fault-tolerant quantum computation, arXiv:2604.08358 (2026).

[69] C. Chamberland, J. Olle, M. Li, S. Thornton, and I. Baratta, Fast and accurate ai-based pre-decoders for surface codes, arXiv:2604.12841 (2026).

[70] A. W. Senior, T. Edlich, F. J. Heras, et al., A scalable and real-time neural decoder for topological quantum codes, arXiv:2512.07737 (2025).

[71] S. Bartolucci, P. Birchall, H. Bomb´ın, et al., Fusion-based quantum computation, Nature Communications 14, 912, arXiv:2101.09310 (2023).

[72] H. Levine, A. Haim, J. Hung, et al., Demonstrating a long-coherence dual-rail erasure qubit using tunable transmons, Physical Review X 14, 011051, arXiv:2307.08737 (2024).

[73] S. Ma, G. Liu, P. Peng, B. Zhang, S. Jandura, J. Claes, A. P. Burgers, G. Pupillo, S. Puri, and J. D. Thompson, High-fidelity gates and mid-circuit erasure conversion in an atomic qubit, Nature 622, 279, arXiv:2305.05493 (2023).

[74] D. E. Browne and T. Rudolph, Resource-efficient linear optical quantum computation, Physical Review Letters 95, 010501, arXiv:quantph/0405157 (2005).

[75] H. Bombin, D. Litinski, N. Nickerson, F. Pastawski, and S. Roberts, Unifying flavors of fault tolerance with the ZX calculus, Quantum 8, 1379, arXiv:2303.08829 (2024).

[76] R. Raussendorf, J. Harrington, and K. Goyal, Topological fault-tolerance in cluster state quantum computation, New Journal of Physic 9, 199, arXiv:quant-ph/0703143 (2007).

[77] H. Bombin, C. Dawson, R. V. Mishmash, N. Nickerson, F. Pastawski, and S. Roberts, Logical blocks for fault-tolerant topological quantum computation, PRX Quantum 4, 020303, arXiv:2112.12160 (2023).

[78] E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, Topological quantum memory, Journal of Mathematical Physics 43, 4452, arXiv:quantph/0110143 (2002).

[79] J. Edmonds, Paths, trees, and flowers, Canadian Journal of Mathematics 17, 449 (1965).

[80] C. Wang, J. Harrington, and J. Preskill, Confinement-higgs transition in a disordered gauge theory and the accuracy threshold for quantum memory, Annals of Physics 303, 31, arXiv:quant-ph/0207088 (2003).

[81] Strictly, a failure results in measurement of two single-qubit Pauli observables, see Appendix B 1 for more details.

[82] D. Poulin, Optimal and efficient decoding of concatenated quantum block codes, Physical Review A 74, 052333, arXiv:quant ph/0606126 (2006).

[83] The ILP problem is obtained by writing each parity constraint $\begin{array} { r } { \sum _ { i } H _ { j i } e _ { i } = \sigma _ { j } } \end{array}$ (mod 2) as $\begin{array} { r } { \sum _ { i } H _ { j i } e _ { i } - 2 y _ { j } = \sigma _ { j } } \end{array}$ with integer slack variables y<sub>j</sub> [119, 120]. ILP decoding is exact for the most-likely error.

[84] J. Roffe, D. R. White, S. Burton, and E. Campbell, Decoding across the quantum low-density parity-check code landscape, Physica Review Research 2, 043423, arXiv:2005.07016 (2020).

[85] H. Bomb´ın, C. Dawson, Y.-H. Liu, N. Nickerson, F. Pastawski, and S. Roberts, Modular decoding: parallelizable real-time decoding for quantum computers, arXiv:2303.04846 (2023).

[86] L. Skoric, D. E. Browne, K. M. Barnes, N. I. Gillespie, and E. T. Campbell, Parallel window decoding enables scalable fault tolerant quantum computation, Nature Communications 14, 7040, arXiv:2209.08552 (2023).

[87] X. Tan, F. Zhang, R. Chao, Y. Shi, and J. Chen, Scalable surface code decoders with parallelization in time, PRX Quantum 4, 040344, arXiv:2209.09219 (2023).

[88] H. Bombin, I. H. Kim, D. Litinski, N. Nickerson, M. Pant, F. Pastawski, S. Roberts, and T. Rudolph, Interleaving: Modular architectures for fault-tolerant photonic quantum computing, arXiv:2103.08612 (2021).

[89] The base clock speed for resource-state generation and fusion is 1 GHz, but long delays of order 2 km are used to trade off between space and time.

[90] A. Y. Kitaev, Fault-tolerant quantum computation by anyons, Annals of physics 303, 2, arXiv:quant-ph/9707021 (2003).

[91] H. Bombin and M. A. Martin-Delgado, Topological quantum distillation, Physical Review Letters 97, 180501, arXiv:quant-ph/0605138 (2006).

[92] A. M. Steane, Error correcting codes in quantum theory, Physical Review Letters 77, 793 (1996).

[93] A. R. Calderbank and P. W. Shor, Good quantum error-correcting codes exist, Physical Review A 54, 1098, arXiv:quant-ph/9512032 (1996).

[94] S. Bartolucci, T. Bell, H. Bombin, et al., Comparison of schemes for highly loss tolerant photonic fusion based quantum computing, arXiv:2506.11975 (2025).

[95] H. Bombin, C. Dawson, T. Farrelly, Y. Liu, N. Nickerson, M. Pant, F. Pastawski, and S. Roberts, Fault-tolerant complexes, arXiv:2308.07844 (2023).

[96] P. Birchall, J. Bridgeman, C. Dawson, T. Farrelly, Y. Liu, N. Nickerson, M. Pant, S. Roberts, K. Seetharam, and D. Tuckett, Macromux: scalable postselection for high-threshold fault-tolerant quantum computation, arXiv:2603.04875 (2026).

[97] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, Attention is all you need, in NIPS’17: Proceedings of the 31st International Conference on Neural Information Processing Systems, Vol. 30, p. 5998–6008, arXiv:1706.03762 (2017).

[98] R. Child, S. Gray, A. Radford, and I. Sutskever, Generating long sequences with sparse transformers, arXiv:1904.10509 (2019).

[99] A. Jaegle, F. Gimeno, A. Brock, O. Vinyals, A. Zisserman, and J. Carreira, Perceiver: General perception with iterative attention, in Proceedings of the 38th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 139, pp. 4651–4664, arXiv:2103.03206 (2021).

[100] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning (MIT Press) (2016).

[101] I. Loshchilov and F. Hutter, Decoupled weight decay regularization, in International Conference on Learning Representations, arXiv:1711.05101 (2019).

[102] A. S. Maan and A. Paler, Machine learning message-passing for the scalable decoding of QLDPC codes, npj Quantum Information 11, 78, arXiv:2408.07038 (2025).

[103] While the maximum-likelihood decoding can offer even higher accuracy, it is too impractical to get any meaningful statistics for these fusion networks.

[104] We utilize the BP+OSD decoder of Refs. [84 and 121] with parameters: bp method = min sum, max iter = 1500, osd method = OSD CS, osd order = 7. All other parameters used default values.

[105] N. Delfosse, A. Paz, A. Vaschillo, and K. M. Svore, How to choose a decoder for a fault-tolerant quantum computer? the speed vs accuracy trade-off, arXiv:2310.15313 (2023).

[106] V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, Efficient processing of deep neural networks: A tutorial and survey, Proceedings of the IEEE 105, 2295, arXiv:1703.09039 (2017).

[107] Compute fabric collectively refers to the hardware resources in the FPGA implementation, including programmable logic, arithmetic blocks, memory, and their interconnections.

[108] K. Hoßfeld, H. J. Damsgaard, J. Nurmi, M. Blott, and T. B. Preußer, High-efficiency Compressor Trees for Latest AMD FPGAs, ACM Transactions on Reconfigurable Technology and Systems 17, 30 (2024).

[109] G. Franco, P. Monteagudo-Lago, I. Colbert, A. Pappalardo, and N. J. Fraser, Xilinx/brevitas: Release v0.13.2 (2026).

[110] A. Pappalardo, Y. Umuroglu, M. Blott, et al., QONNX: Representing arbitrary-precision quantized neural networks, arXiv:2206.07527 (2022).

[111] Y. Umuroglu, N. J. Fraser, G. Gambardella, M. Blott, P. Leong, M. Jahre, and K. Vissers, FINN: A framework for fast, scalable binarized neural network inference, in Proceedings of the 2017 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, FPGA ’17 (Association for Computing Machinery, New York, NY, USA) pp. 65–74, arXiv:1612.07119 (2017).

[112] Y. Bengio, N. Leonard, and A. Courville, Estimating or propagating gradients through stochastic neurons for conditional computation,´ arXiv:1308.3432 (2013).

[113] Y. Umuroglu and M. Jahre, Streamlined deployment for quantized neural networks, arXiv:1709.04060 (2017).

[114] J. Bai, F. Lu, K. Zhang, et al., ONNX: Open neural network exchange, GitHub repository: github.com/onnx/onnx (2019).

[115] C. Yun, S. Bhojanapalli, A. S. Rawat, S. J. Reddi, and S. Kumar, Are transformers universal approximators of sequence-to-sequence functions?, in International Conference on Learning Representations (ICLR), arXiv:1912.10077 (2020).

[116] S.-H. Lee, L. H. English, and S. D. Bartlett, Efficient post-selection for general quantum LDPC codes, npj Quantum Information 12, 96, arXiv:2510.05795 (2026).

[117] H. Bomb´ın, C. Dawson, N. Nickerson, M. Pant, and J. Sullivan, Increasing error tolerance in quantum computers with dynamic bias arrangement, arXiv:2303.16122 (2023).

[118] T. Haug, A. Canabarro, and L. Aolita, Machine-learned syndrome post-selection for reliable quantum error correction, arXiv:2607.19563 (2026).

[119] A. J. Landahl, J. T. Anderson, and P. R. Rice, Fault-tolerant quantum computing with color codes, arXiv:1108.5738 (2011).

[120] M. Cain, C. Zhao, H. Zhou, N. Meister, J. P. Bonilla Ataides, A. Jaffe, D. Bluvstein, and M. D. Lukin, Correlated decoding of logical algorithms with transversal gates, Physical Review Letters 133, 240602, arXiv:2403.03272 (2024).

[121] J. Roffe, LDPC: Python tools for low density parity check codes (2022).

[122] A. Bolt, G. Duclos-Cianci, D. Poulin, and T. Stace, Foliated quantum error-correcting codes, Physical Review Letters 117, 070501, arXiv:1607.02579 (2016).

[123] B. J. Brown and S. Roberts, Universal fault-tolerant measurement-based quantum computation, Physical Review Research 2, 033305, arXiv:1811.11780 (2020).

[124] J. C. Bridgeman, A. Kubica, and M. Vasmer, Lifting topological codes: Three-dimensional subsystem codes from two-dimensional anyon models, PRX Quantum 5, 020310, arXiv:2305.06365 (2024).

[125] N. Nickerson and H. Bomb´ın, Measurement based fault tolerance beyond foliation, arXiv:1810.09621 (2018).

[126] M. Newman, L. A. de Castro, and K. R. Brown, Generating fault-tolerant cluster states from crystal structures, Quantum 4, 295, arXiv:1909.11817 (2020).

[127] W. P. Grice, Arbitrarily complete bell-state measurement using only linear optical elements, Physical Review A 84, 042331 (2011).

[128] F. Ewert and P. van Loock, 3/4-efficient bell measurement with passive linear optics and unentangled ancillae, Physical Review Letters 113, 140403, arXiv:1403.4841 (2014).

[129] S.-W. Lee, T. C. Ralph, and H. Jeong, Fundamental building block for all-optical scalable quantum networks, Physical Review A 100, 052303, arXiv:1804.09342 (2019).

[130] F. Ewert, M. Bergmann, and P. Van Loock, Ultrafast long-distance quantum communication with static linear optics, Physical Review Letters 117, 210501, arXiv:1503.06777 (2016).

[131] J. L. Ba, J. R. Kiros, and G. E. Hinton, Layer normalization, arXiv:1607.06450 (2016).

[132] R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, and T.-Y. Liu, On layer normalization in the transformer architecture, in Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 119, pp. 10524–10533, arXiv:2002.04745 (2020).

[133] T. Bachlechner, B. P. Majumder, H. Mao, G. Cottrell, and J. McAuley, ReZero is all you need: Fast convergence at large depth, in Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, Proceedings of Machine Learning Research, Vol. 161, pp. 1352–1361, arXiv:2003.04887 (2021).

[134] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, Dropout: a simple way to prevent neural networks from overfitting, The Journal of Machine Learning Research 15, 1929 (2014).

[135] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, Neural message passing for quantum chemistry, in Proceedings

of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 70, pp. 1263–1272, arXiv:1704.01212 (2017).

[136] P. W. Battaglia, J. B. Hamrick, V. Bapst, et al., Relational inductive biases, deep learning, and graph networks, arXiv:1806.01261 (2018).

<sup>[137]</sup> <sup>We</sup> <sup>note</sup> <sup>that</sup> <sup>SCA</sup> <sup>(</sup>·<sup>,</sup> ·<sup>)</sup> <sup>also</sup> <sup>uses</sup> <sup>bias</sup> <sup>terms,</sup> <sup>but</sup> <sup>these</sup> <sup>have</sup> <sup>been</sup> <sup>suppressed</sup> <sup>for</sup> <sup>clarity.</sup>

[138] M. Geva, R. Schuster, J. Berant, and O. Levy, Transformer feed-forward layers are key-value memories, in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484–5495, arXiv:2012.14913 (2021).

[139] D. Hendrycks and K. Gimpel, Gaussian error linear units (GELUs), arXiv:1606.08415 (2016).

[140] M. P. C. Fossorier, M. Mihaljevic, and H. Imai, Reduced complexity iterative decoding of low-density parity check codes based on belief´ propagation, IEEE Transactions on Communications 47, 673 (1999).

[141] J. Chen and M. P. C. Fossorier, Near optimum universal belief propagation based decoding of low-density parity check codes, IEEE Transactions on Communications 50, 406 (2002).

[142] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, in International Conference on Learning Representations, arXiv:1412.6980 (2015).

[143] P. Tillet, H. T. Kung, and D. Cox, Triton: An intermediate language and compiler for tiled neural network computations, in Proceedings ofthe 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pp. 10–19 (2019).

[144] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, FlashAttention: Fast and memory-efficient exact attention with IO-awareness, in´ Advances in Neural Information Processing Systems, Vol. 35, pp. 16344–16359, arXiv:2205.14135 (2022).

[145] J. Chen and M. P. C. Fossorier, Density evolution for two improved BP-based decoding algorithms of LDPC codes, IEEE Communications Letters 6, 208 (2002).

[146] C. Lavin and A. Kaviani, RapidWright: Enabling custom crafted implementations for FPGAs, in 2018 IEEE 26th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM) (IEEE, Boulder, CO, USA) pp. 133–140 (2018).

[147] C. Lavin and E. Hung, Invited paper: RapidWright: Unleashing the full power of FPGA technology with domain-specific tooling, in 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD) (IEEE, San Francisco, CA, USA) pp. 1–7 (2023).

[148] Z. Li and Q. Gu, I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference, in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17019–17029, arXiv:2207.01405 (2023).

[149] J. Fromm, M. Cowan, M. Philipose, L. Ceze, and S. Patel, Riptide: Fast end-to-end binarized neural networks, in Proceedings of Machine Learning and Systems, Vol. 2, pp. 379–389 (2020).

[150] K. Tang, Z. Yan, and W.-F. Wong, Sorbet: A neuromorphic hardware-compatible transformer-based spiking language model, in Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 267 (PMLR) pp. 58989–59004, arXiv:2409.15298 (2025).

Appendix A: Notation
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>Code and protocol parameters</td><td></td></tr><tr><td>n</td><td>Number of resource states in the fusion-network block</td></tr><tr><td>k</td><td>Number of encoded qubits per code layer</td></tr><tr><td> $d$ </td><td>Code or protocol fault-distance</td></tr><tr><td> $K$ </td><td>Number of membrane equivalence classes</td></tr><tr><td> $R$ </td><td>Number of membrane representatives used per class</td></tr><tr><td>nE</td><td>Number of error mechanisms</td></tr><tr><td> $n _ { \mathrm { C } }$ </td><td>Number of checks</td></tr><tr><td> $H \in \mathbb { F } _ { 2 } ^ { n _ { \mathrm { C } } \times n _ { \mathrm { E } } }$ </td><td>Check matrix</td></tr><tr><td> $\boldsymbol { \Lambda } \in \mathbb { F } _ { 2 } ^ { ( K \times R ) \times n _ { \mathrm { E } } }$ </td><td>Membrane matrix, with one row per representative</td></tr><tr><td>Errors, membranes, and decoding</td><td></td></tr><tr><td> $e \in \mathbb { F } _ { 2 } ^ { n _ { \mathrm { E } } }$ </td><td>Error-pattern vector, indicating flipped error mechanisms</td></tr><tr><td> $\varepsilon \in \mathbb { F } _ { 2 } ^ { n _ { \mathrm { E } } }$ </td><td>Erasure-pattern vector, indicating erased error mechanisms</td></tr><tr><td> $\sigma = H e$ </td><td>Syndrome vector</td></tr><tr><td> $f = \Lambda e$ </td><td>Membrane-flip vector</td></tr><tr><td> $\varkappa , \varkappa ( \mu )$ </td><td>Membrane equivalence class; class containing representative</td></tr><tr><td> $\mu _ { \varkappa }$ </td><td>Reference representative of class ¿</td></tr><tr><td> $t _ { \mu }$ </td><td>Transition-check vector satisfying  $\Lambda _ { \mu _ { \varkappa } } = \Lambda _ { \mu } + t _ { \mu } H$ </td></tr><tr><td> $\tau _ { \mu } = t _ { \mu } \sigma$ </td><td>Transition syndrome aligning the flip of  $\mathbf { \nabla } \mu$  with its reference representative</td></tr><tr><td> $\lambda _ { \mu } ( b )$ </td><td>Predicted logit for outcome  $b \in \{ 0 , 1 \}$  of representative  $\mu ; b = 1$  denotes a flip</td></tr><tr><td>Probability, noise, and training</td><td></td></tr><tr><td>pDEC</td><td>True decoding distribution induced by the protocol and error model</td></tr><tr><td> $p _ { \theta } , \theta$ </td><td>Model distribution and its trainable parameters</td></tr><tr><td> $\mathcal { L } ( \boldsymbol { \theta } )$ </td><td>Expected conditional cross-entropy training loss</td></tr><tr><td> $_ x$ </td><td>Noise operating point, with  $p _ { E } = x$  and  $p _ { P } = x / 1 0$ </td></tr><tr><td> $p _ { E } , p _ { P }$ </td><td>Erasure probability and flip probability of a non-erased outcome</td></tr><tr><td>Model hyperparameters</td><td></td></tr><tr><td> $d _ { \mathrm { M } }$ </td><td>Model dimension</td></tr><tr><td> $n _ { \mathrm { H } }$ </td><td>Number of attention heads</td></tr><tr><td> $d _ { \mathrm { H } } = d _ { \mathrm { M } } / n _ { \mathrm { H } }$ </td><td>Head dimension</td></tr><tr><td> $d _ { \mathrm { F } }$ </td><td>Feed-forward hidden dimension</td></tr><tr><td> $Q$ </td><td>Number of message-passing layers</td></tr><tr><td> $P$ </td><td>Number of perceiver layers</td></tr><tr><td>Model states and operations</td><td></td></tr><tr><td> $h _ { i } , g _ { j } , m _ { \mu }$ </td><td>Error-mechanism, check, and membrane states, respectively</td></tr><tr><td> $[ v ] _ { k }$ </td><td>Feature (component) k of state v</td></tr><tr><td> $\operatorname { s u p p } \left( \mu \right)$ </td><td>Error mechanisms in the support of representative  $\mu$ </td></tr><tr><td> $\mathrm { n b d } ( a )$ </td><td>Neighbors of node a in the Tanner graph</td></tr><tr><td> $\gamma$ </td><td>Learned scalar gate on a residual connection</td></tr><tr><td>Quantization and integer voting</td><td></td></tr><tr><td> $\mathrm { W } _ { w } \mathrm { A } _ { a }$ </td><td>Model with w-bit matrix weights and a-bit core activations</td></tr><tr><td> ${ \mathrm { q u a n t } } _ { b } ( z ; \ s )$ </td><td>Uniform b-bit quantizer applied to input z with learned scale</td></tr><tr><td> $q _ { b } ^ { \mathrm { ( m i n ) } } , q _ { b } ^ { \mathrm { ( m a x ) } }$ </td><td>Lower and upper integer quantization bounds</td></tr><tr><td>Accelerator configurations</td><td></td></tr><tr><td> $\mathrm { M M V } _ { \mathrm { e r a s u r e } } , \mathrm { M M V } _ { \mathrm { s y n d r o m e } } , \mathrm { M M V }$ </td><td>Numbers of parallel token-processing lanes for the respective input streams</td></tr><tr><td>membrane  $\langle d _ { \mathrm { M } } , [ \mathrm { M M V _ { e r a s u r e } , M M V _ { s y n d r o m e } } ] \rangle$ </td><td>Accelerator configuration specified by  $d _ { \mathrm { M } }$  and erasure/syndrome stream parallelism</td></tr></table>

![](images/9ecc0b24f26a115ba67a5bcaca8038e4e98dbf56035263722c226a6d8251e546.jpg)  
FIG. B1: The fusion networks for the 6-ring, LLD, and CC8 protocols. (a) The 6-ring fusion network follows the cubic lattice. Each cubic 3-cell supports a check, made up of either the XX or ZZ fusion outcomes on all edges in the cube, alternating between X-type and Z-type in a bicolorable (checkerboard) pattern. (b) The LLD fusion network follows the rectified cubic lattice, of which a portion is shown. Checks are supported on 3-cells, of which there are triangular volumes shown shaded in blue (these have 3 vertices, 3 edges and two faces), octahedrons, and cuboctahedrons. The triangular volumes support Y-checks, made up of YY outcomes (i.e., weight 6 in terms of the XX and ZZ fusion outcomes), the octahedra support Z-checks made up of ZZ outcomes, and the cuboctahedra are made up of both YY and ZZ outcomes (namely, the opposite of the support of the neighboring triangular volumes and octahedra). (c) The fusion graph for the CC8 fusion network is also the rectified cubic lattice, but the interpretation is different. Now, there are only octahedral and cuboctahedral volumes, and each volume supports both an X-type and Z-type check. This requires a different resource state to LLD.

## Appendix B: Fusion networks

The fusion networks we have considered are based on those described in Ref. [95]. These are fusion networks that encode logical information in surface codes, color codes, or bivariate bicycle (BB) codes. Note that while the BB-code networks are foliated [122–124], the surface and color-code constructions are based on a tiling of 3D space-time, and so they do not necessarily admit a description in terms of a foliated code [125, 126]. We refer to these surface and color code networks as topological networks. In particular, we consider the 6-ring fusion network, which is based on surface codes, the linked-loopy diamond (LLD) fusion network, which is based on surface codes with linking (a type of Clifford deformation), the CC8 fusion network, which is based on color codes, and the foliated 72, 12, 6 and 144, 12, 12 (“Gross”) codes, which are respectively called the HalfGross and Gross codes [16].

While the 6-ring fusion network admits a graphlike Tanner graph, meaning it can be decoded by MWPM, the LLD, CC8, Half-Gross and Gross fusion networks have outcome degrees greater than 2, and so MWPM cannot be directly used.

In Figure B1, we depict the topological fusion networks, which define how resource states are fused together, and how the checks are formed. The stabilizers for resource states can be derived from these fusion networks by considering the incident checks. In particular, each vertex corresponds to a resource state, whose number of qubits is equal to its degree (number of incident edges). The stabilizers can be obtained by restricting each check operator to the support of that resource state. As such, the resource states for the 6-ring and LLD networks are six- and eight-qubit surface codes on a topological sphere (with the latte also having single-qubit Clifford operators applied to it). The CC8 fusion network has the same geometry as LLD; however, the check structure is different, with an X-type and Z-type check for each 3-cell. As such, the resource states for CC8 also have eight qubits, but are color codes on a topological sphere. For more details, see Ref. [95].

The Half-Gross and Gross networks make use of n-qubit GHZ states and their locally Hadamarded versions, which we call n-GHZ and n-GHX states, respectively. In particular we consider alternating layers that act to measure X and Z stabilizers of the code, called X-type and Z-type layers. The stabilizer groups for these codes are described in Ref. [16], along with an (over-complete) set of generators consisting of N/2 X-type and N/2 Z-type stabilizers for a code of N qubits. The stabilizer generators are weight w = 6 for these codes, and each qubit participates in o = 3 generators of each type. In each X-type layer, we place an o+2-GHX for each data qubit of the code and we place a w-GHZ for each X-stabilizer generator. On Z-type layers we place an o+2-GHZ for each data qubit of the code, and we place a w-GHX for each Z-stabilizer generator. Within a layer, each generator resource is fused to the data resource states in its support, using one qubit from each resource state per fusion. Each data resource state has o of its qubits paired up for fusions inside a layer, and the remaining two qubits fused to corresponding data resource states on neighboring layers. We consider d pairs of layers, with periodic boundary conditions in time. Checks can be constructed for each stabilizer generator in each layer. See Ref. [75] for details of the foliated construction and how checks are formed.

The linked loopy diamond network differs from the (CSS) loopy diamond network of Refs. [95 and 96] through a local Clifford on each qubit. In particular, the CSS formulation of the fusion network has bicolorable 3-cells, alternating between X-type and $Z -$ -type checks. To obtain the linked version [19], we apply a Clifford to each resource-state qubit, such that fusion outcomes are mapped according to $A : X X \to Y Y \to Z Z \to X X$ or $A ^ { \dagger } : X X \to Z Z \to Y Y \to X X$ . The Clifford operators are applied to make each triangular volume a Y -type check.

## 1. Fusion error models

In this work, we restrict our attention to two-qubit Bell measurements, known as Type-II fusions [71, 74] on dual-rail encoded qubits. In the absence of errors, the ideal fusion either succeeds, measuring $\{ X \otimes X , Z \otimes Z \} , \mathrm { o r } f a i l s .$ measuring $\{ Z \otimes I , I \otimes Z \}$ One can modify the fusion circuit to instead measure $\{ X \otimes I , I \otimes X \}$ upon failure. In the event of failure, by combining the two single-qubit outcomes, failure can be modeled in terms of a Bell-basis measurement where only one of the fusion outcomes $( \mathrm { e } . \mathrm { g } . , Z Z )$ is obtained, with the other outcome being erased $( \mathbf { e } . \mathbf { g } . , X X )$ . However, we remark that fusion failures still produce two bits of information, and there are circumstances in which using two bits of information is more useful than taking only their parity.

At each ideal fusion, success and failure each occur with probability $1 / 2 .$ . In the presence of photon loss, we can also obtain a nonsense outcome when performing the fusion, in which no useful information is returned, meaning both outcomes are erased. Therefore, success and failure of the fusion are also conditioned on having both photons present. For a per-photon loss rate of $^ { \cdot } l ,$ the success, failure and nonsense outcomes occur with probability

$$
p ( { \mathrm { s u c c e s s } } ) = { \frac { 1 } { 2 } } ( 1 - l ) ^ { 2 } , \qquad p ( { \mathrm { f a i l } } ) = { \frac { 1 } { 2 } } ( 1 - l ) ^ { 2 } , \qquad p ( { \mathrm { n o n s e n s e } } ) = 1 - ( 1 - l ) ^ { 2 } .\tag{B1}
$$

Outcomes that are obtained from a fusion are also subject to bitflips, as Pauli errors may be present which lead to an appropriate flip of the XX or $Z Z$ outcome. As such, one can parametrize the Pauli rate on qubits entering a fusion in terms of their resulting bit-flip rate.

Fusions can have their success probability improved through the use of boosting [127, 128], which makes use of additional photons in the form of ancillary Bell-states. Using one extra Bell state, we can improve the success probability from $1 / 2$ to $3 / 4$ , but now we require all four photons to be present. One can also further increase the success probability at the cost of more Bell pairs. In addition to boosting, local encoding and adaptive strategies for fusion can further improve these physical error rates (e.g., Refs. [117, 129, and 130]).

In this paper, we therefore consider a simplified, tunable error model parametrized in terms of the operating point, following Equation (3). Note that the rate of erasure errors compared to bitflip errors is 10:1. This is motivated by photonic hardware supporting very-high fidelities for single and two-qubit gates on dual rail photonic qubits [10], and considering the erasure rates of fusion outcomes from a combination of photon loss and fusion failure.

![](images/448c2142c6a673707d0b22ffd81747b07ea2452b84cdb3e6ce2c344e69a90698.jpg)  
FIG. C1: Model structure. Left: the message-passing stack, which updates the states of error mechanisms and checks through sparse cross attention restricted to the Tanner graph, followed by the one-sided interface layer, which writes the final check states into the error-mechanism states. Right: the perceiver stack, which updates a state per membrane representative by aggregating the error-mechanism states over the membrane’s support. Triangles denote layer normalization, and every block writes into its stream through a residual connection with a learned scalar gate. Colored components contain learned parameters. The per-representative logits are combined into one prediction per membrane class by the syndrome-aligned vote of Section III C 1. Circles marked with ‘A’ denote quantizers, which are only relevant for the quantized architecture. See Appendix D for details.

## Appendix C: Machine-learning decoding

In this appendix, we provide a detailed explanation of the PRISM-decoder architecture, in addition to some training details.

## 1. Architecture

As discussed in Section III, our model architecture (Figure C1) is built from a series of components, which serve distinct purposes. We now describe each of these in sequence.

Layer normalization and residuals. At many stages in the stack, we apply a layer normalization to stabilize the dynamics of the model. We refer to Ref. [131] for details. These are applied separately to each stream before each block, following Ref. [132]. In Figure C1, these normalizations are denoted by green triangles, and in text we will denote this by ˆ.

Additionally, we make use of residual connections around the sub-components [133]. Given some sub-component $C ,$ the action of this connection is

$$
x  x + \gamma C ( x ) ,\tag{C1}
$$

where the γs are learned parameters. In the diagrams, we denote these connections with an $\oplus$ node. We refer to Ref. [133] for more details.

Dropout. To avoid overfitting, it is common to randomly zero out state entries [134]. This ensures the model does not rely too heavily on any given feature. During training, dropout is applied to the output of every attention, feed-forward, and aggregation block before its residual addition.

Dropout is disabled during inference and is omitted from subsequent equations for clarity.

Embedding. We represent the observed syndrome σ and erasure pattern ε as binary vectors, where a $\cdot _ { 1 } \cdot$ indicates, respectively, a flip of a check, or an erasure event. We refer to each bit $\sigma _ { i } , \varepsilon _ { i } \in \{ 0 , 1 \}$ of these vectors as a token. The model consumes the tokens and first embeds them into an internal d<sub>M</sub>-dimensional, real vector space via embedding maps E

$$
[ g _ { i } ] _ { j } = [ E _ { \sigma } ] _ { \sigma _ { i } , j } ,\tag{C2a}
$$

![](images/9b848317e35c231bb8eec27a25668fce1383e47176a9042fb686386004a4edbd.jpg)

$$
\left[ h _ { i } \right] _ { j } = \left[ E _ { \varepsilon } \right] _ { \varepsilon _ { i } , j } ,\tag{C2b}
$$

where the maps $E$ are (learned) matrices of size $2 \times d _ { \mathrm { M } }$ , and we refer to $g _ { i }$ and $h _ { i }$ as check states and error-mechanism states respectively. We refer to the entries of these states as features.

Message passing stack. With the observations embedded into the model, check and error-mechanism states exchange information through a series of $Q$ message passing layers using sparse cross-attention (SCA). Following Ref. [58], these messages are constrained to pass along the edges of the Tanner graph of the protocol, setting the sparse structure.

Each layer of the message passing stack consists of four sub-components, two SCA blocks and two feed-forward (FF) blocks. We note that each block has its own parameters, which are distinct across the $Q$ repetitions. For each of the Q passes, the check states g and error-mechanism states h are updated according to

$$
\begin{array} { r } { h  h + \gamma _ { 1 } \mathrm { S C A } ( \hat { h } , \hat { g } ) , } \end{array}
$$

$$
\begin{array} { r } { h  h + \gamma _ { 2 } \ \mathrm { F F } ( \hat { h } ) , } \end{array}\tag{C3a}
$$

(C3b)

$$
g \gets g + \gamma _ { 3 } \operatorname { S C A } \left( \hat { g } , \hat { h } \right)\tag{C3c}
$$

$$
g  g + \gamma _ { 4 } \mathrm { F F } ( \hat { g } ) ,\tag{C3d}
$$

where $\operatorname { S C A } \left( \cdot , \cdot \right)$ and $\mathrm { F F } \left( \cdot \right)$ are sparse cross-attention and feed-forward blocks respectively, as discussed below.

![](images/6e0a65f3103691d4599035805551d107fb34eb8c85a168a6d1f74b9d7cda8d37.jpg)

Equation (C3) is a message-passing neural network on the Tanner graph [135, 136], using attention as the aggregation function. It can be understood as the bipartite form of the code-aware attention of Ref. [58]; two hops in the Tanner graph, as used in Ref. [58], connect exactly the checks that share an error mechanism.

Sparse cross-attention. Cross-attention is a mechanism to allow a set of states x to be updated by another set y. Learned parameters control how strongly each state in x is impacted by each in $y .$ . The sparse attention used here restricts attention to nearest-neighbors on the Tanner graph, meaning that states can only be updated by their neighbors. We use n attention heads, each of dimension d [97]. For each head r, three (learned) matrices, $\bar { W } _ { Q } ^ { ( r ) } , \bar { W _ { K } ^ { ( r ) } } , W _ { V } ^ { ( r ) } \ \in \ \mathbb { R } ^ { d _ { \ H } \times d _ { \mathrm { M } } } \ [ 1 3 7 ]$ , map the states to ‘query’, ‘key’ and ‘value’ vectors. Depending on the ‘attention activation’ hyperparameter, the head computes

$$
s _ { a } ^ { ( r ) } : = \sum _ { b \in \mathrm { n b d } ( a ) } \operatorname { s o f t m a x } _ { b } \left( \left. W _ { Q } ^ { ( r ) } x _ { a } , W _ { K } ^ { ( r ) } y _ { b } \right. / \sqrt { d _ { \mathrm { H } } } \right) W _ { V } ^ { ( r ) } y _ { b } ,
$$

softmax activation

$$
s _ { a } ^ { ( r ) } : = \frac { 1 } { \sqrt { | \mathrm { n b d } ( a ) | } } \sum _ { b \in \mathrm { n b d } ( a ) } \operatorname { t a n h } \left( \left. W _ { Q } ^ { ( r ) } x _ { a } , W _ { K } ^ { ( r ) } y _ { b } \right. / \sqrt { d _ { \mathrm { H } } } \right) W _ { V } ^ { ( r ) } y _ { b } ,\tag{C4a}
$$

tanh activation

(C4b)

where nbd denotes neighbors in the Tanner graph. Intuitively, the inner-product determines how strongly $x _ { a }$ attends to each y<sub>b</sub>. Finally, the d<sub>H</sub>-dimensional vectors are concatenated across the $n _ { \mathrm { H } }$ heads to construct the output vectors, and mixed using a

learned output matrix $W _ { O } \in \mathbb { R } ^ { d _ { N } }$ n<sub>H</sub>d<sub>H</sub>

$$
\mathrm { S C A } \left( x , y \right) _ { a } : = W _ { O } \mathrm { c o n c a t } { \left( s _ { a } ^ { ( 1 ) } , \dots , s _ { a } ^ { ( n _ { \mathrm { H } } ) } \right) } .\tag{C5}
$$

When the x are error-mechanism states, they are updated by their neighboring checks, and vice-versa.

Feed-forward block. We use a two-layer feed-forward block [97, 138], with a tanh approximation to GELU activation [139]. With hidden dimension $d _ { \mathrm { F } }$ , the states are updated via

$$
x  x + \gamma ( A _ { 2 } ( \mathrm { G E L U } ( A _ { 1 } \hat { x } + b _ { 1 } ) ) + b _ { 2 } ) ,\tag{C6}
$$

where the residual scalar $\gamma , A _ { 1 } , A _ { 2 } ^ { \mathsf { T } } \in \mathbb { R } ^ { d _ { \mathrm { F } } \times d _ { \mathrm { M } } } , b _ { 1 } \in \mathbb { R } ^ { d _ { \mathrm { F } } }$ , and $b _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { M } } }$ are learned parameters.

Interface layer. At the end of the message-passing stack, the most recently updated states are checks. These are discarded before the perceiver stack. We therefore introduce an additional layer of message passing to update the error-mechanism states once more. The structure of this layer is identical to the first half of the main messagepassing layer.

![](images/f8a9cc3e35c441b738b1a905ce40fdc464862e1ba370733163bd2ee4dfd06279.jpg)

Perceiver stack. At this stage of the model, the error-mechanism states are fully updated by the checks, and are now used to update states $m _ { \mu }$ for each membrane representative. Unlike error-mechanisms and check states, membrane states have no observations that can be embedded, so these states are initially set to a single learned seed vector $m ^ { ( 0 ) } \in \mathbb { R } ^ { d _ { \mathrm { M } } }$ , in the manner of the latent array of a perceiver [99]. That is, at the first layer of the perceiver stack,

$$
m _ { \mu } = m ^ { ( 0 ) } ,\tag{C7}
$$

for all representatives $\mu .$

![](images/8aa5b30dac34f15758e46be46a645e00d6f551ba4d86aa4b73550cd2c5e910a0.jpg)

Following initialization, each membrane representative $\mu$ is updated by its incident error mechanisms, denoted supp(µ). Specifically, several statistics are gathered for each representative, and learned weights are used to update the membrane states as follows.

For error mechanism i, learned affine maps, followed by tanh activations, are applied to ensure all features remain bounded:

$$
q _ { i } = \operatorname { t a n h } ( A _ { q } \hat { h } _ { i } + b _ { q } ) ,\tag{C8a}
$$

$$
v _ { i } = \operatorname { t a n h } ( A _ { v } \hat { h } _ { i } + b _ { v } ) ,\tag{C8b}
$$

$$
w _ { i } = \operatorname { t a n h } ( A _ { w } \hat { h } _ { i } + b _ { w } ) ,\tag{C8c}
$$

where $A _ { q } , A _ { v } , A _ { w } \in \mathbb R ^ { d _ { \mathrm { M } } \times d _ { \mathrm { M } } }$ , and $b _ { q } , b _ { v } , b _ { w } \in \mathbb { R } ^ { d _ { \mathrm { M } } }$ are learned parameters.

For each feature, the mean, second moment, and a ‘softparity’ are computed

$$
[ u _ { \mu } ] _ { k } = \frac { 1 } { | \operatorname { s u p p } ( \mu ) | } \sum _ { i \in \operatorname { s u p p } ( \mu ) } \left[ q _ { i } \right] _ { k } ,\tag{C9a}
$$

$$
\left[ s _ { \mu } \right] _ { k } = \frac { 1 } { | \operatorname { s u p p } ( \mu ) \mid } \sum _ { i \in \operatorname { s u p p } ( \mu ) } \left[ v _ { i } \right] _ { k } ^ { 2 } ,\tag{C9b}
$$

$$
\left[ \pi _ { \mu } \right] _ { k } = \operatorname* { m i n } _ { i \in \mathrm { s u p p } ( \mu ) } | [ w _ { i } ] _ { k } | \left( \prod _ { i \in \mathrm { s u p p } ( \mu ) } \mathrm { s g n } ( [ w _ { i } ] _ { k } ) \right) ,\tag{C9c}
$$

where $k = 1 , \ldots , d _ { \mathrm { M } }$ indexes the feature coordinates, and sgn is the sign function. Equation (C9c) computes the parity of the features, then multiplies by the min-magnitude. This is used in min-sum belief propagation [140, 141], and approximates the

log-likelihood ratio of the parity of independent bits. Previously, we took a simple product of the $[ w _ { i } ] _ { k } ;$ however this suffered from underflow issues in single precision at larger distances. We refer to this as product aggregation, and discuss it in the context of quantized models in Appendix D 3.

Following collation of these statistics, they are used to update the membrane state

$$
c _ { \mu } = A _ { c } \operatorname { c o n c a t } ( u _ { \mu } , s _ { \mu } , \pi _ { \mu } ) + b _ { c } ,\tag{C10a}
$$

$$
a _ { \mu } = A _ { o } \mathrm { G E L U } ( A _ { m } \hat { m } _ { \mu } + c _ { \mu } ) + b _ { o } ,\tag{C10b}
$$

$$
m _ { \mu }  m _ { \mu } + \gamma _ { A } a _ { \mu } ,\tag{C10c}
$$

where $A _ { c } \in \mathbb { R } ^ { d _ { \mathrm { M } } \times 3 d _ { \mathrm { M } } } , A _ { o } , A _ { m } \in \mathbb { R } ^ { d _ { \mathrm { M } } \times d _ { \mathrm { M } } }$ and $b _ { c } , b _ { o } \in \mathbb { R } ^ { d _ { \mathrm { M } } }$ are learned parameters. We refer to Ref. [139] for details on GELU activation.

Following aggregation, the membrane states are passed through a feed-forward block.

Output logits. Finally, the membrane states are converted into logits via an affine transformation

$$
\lambda _ { \mu } = { \cal W } \hat { m } _ { \mu } + b ,\tag{C11}
$$

where $W \in \mathbb { R } ^ { 2 \times d _ { \mathrm { M } } }$ , and $b \in \mathbb { R } ^ { 2 }$ are learned parameters. The logits $[ \lambda _ { \mu } ] _ { 0 } , [ \lambda _ { \mu } ] _ { 1 }$ are the scores for membrane representative $\mu$ being unflipped, or flipped respectively. Their difference $\left[ \lambda _ { \mu } \right] _ { 1 } - \left[ \lambda _ { \mu } \right] _ { 0 }$ is the predicted log-odds of a flip. Using the syndromealigned vote described in Section III C 1, these are combined into predicted flips for each membrane class.

![](images/b6c3012d36f96e145821f23178d91a97f653f50e98d4561b397ad2288184d930.jpg)

Erasure-supported inference augmentation. Recall that the error bits of erased outcomes are independent and uniformly distributed with probability 1/2. Consequently,

$$
p ( f \mid \sigma , \varepsilon ) = p \Big ( f + \Lambda g \mid \sigma ^ { ( g ) } , \varepsilon \Big ) ,\tag{C12}
$$

where $g$ is any error pattern supported on only the erased sites, and

$$
\sigma ^ { ( g ) } : = \sigma + H g .\tag{C13}
$$

From Equation (4), the optimal decoder, when shown $( \sigma ^ { ( g ) } , \varepsilon )$ , would therefore return $\hat { f } + \Lambda g$ . The learned decoder is not required to be equivariant in this way, and we can exploit this.

For some protocols, we find it beneficial to resample the received syndrome during inference. This is done by sampling a small set $\mathcal { G }$ of error vectors, supported only on the erased sites and including $g = 0$ . These are used to translate the input syndrome (as in Equation (C13)), so ${ \mathcal { G } } = \{ 0 \}$ corresponds to decoding as usual. The decoder is shown each transformed pair $( \sigma ^ { ( g ) } , \varepsilon )$ and returns output logits $\lambda _ { \mu } ^ { ( g ) }$ for each membrane representative. As in Equation (10), these are aligned to the reference representative of each class, while undoing the gauge transformation

$$
\widetilde { \lambda } _ { \mu } ^ { ( g ) } ( b ) : = \lambda _ { \mu } ^ { ( g ) } \left( b + \Lambda _ { \mu } g + \tau _ { \mu } \right) , \qquad b \in \{ 0 , 1 \} ,\tag{C14}
$$

where $\tau _ { \mu } = t _ { \mu } \sigma$ is the transition syndrome (Section III C 1). Defining

$$
\Delta _ { \varkappa } ^ { ( g ) } : = \sum _ { \mu \in \varkappa } \left[ \widetilde { \lambda } _ { \mu } ^ { ( g ) } ( 1 ) - \widetilde { \lambda } _ { \mu } ^ { ( g ) } ( 0 ) \right] ,\tag{C15}
$$

we choose the gauge that maximizes the confidence of the least-confident class

$$
\hat { g } = \underset { g \in \mathcal { G } } { \arg \operatorname* { m a x } } \operatorname* { m i n } _ { \varkappa } \left| \Delta _ { \varkappa } ^ { ( g ) } \right| .\tag{C16}
$$

We then use the aligned logits from this single selected gauge to predict the flips of every class via Equation (10). This procedure requires no knowledge of the hidden error vector $e .$ For a fixed number of views , it introduces a constant-factor decoding overhead, requiring  network evaluations per shot.

## 2. Training

In this work, we have used an error model that is the product of independent local models. For each sample, errors e and erasures ε are drawn, and the syndromes $\sigma = H e$ and membrane flips $f = \Lambda e$ e computed from sparse matrix products over $\mathbb { F } _ { 2 }$ These samples form the training data, and are generated in batches on the AMD MI350X GPU during training.

TABLE C1: Configurations of the full-precision exploration models. x is the range of operating points spanned by the training temperature, $T \in [ 0 , 1 ]$ . Hyperparameters used for results in Section IV. ‘Num gauges’ denotes the size of the set from Appendix C 1. Training time is indicated as number of GPUs  wall-time per GPU. Training was performed on AMD Instinct<sup>TM</sup> MI350X GPUs.
<table><tr><td rowspan="2">Protocol</td><td rowspan="2">d</td><td colspan="3">Dimensions</td><td rowspan="2">− (Q, P)</td><td colspan="2">Num</td><td rowspan="2">Batch</td><td rowspan="2">Peak LR</td><td rowspan="2">Weight decay</td><td rowspan="2">Dropout</td><td rowspan="2"> $x \ : ( \% )$ </td><td rowspan="2">Steps</td><td rowspan="2">Attention activation</td><td rowspan="2">Num gauges</td><td rowspan="2">Train time (h)</td></tr><tr><td> $\overline { { d _ { \mathrm { M } } } }$ </td><td> $\overline { { n _ { \mathrm { H } } \times d _ { \mathrm { H } } } }$ </td><td> $\overline { { d _ { \mathrm { F } } } }$ </td><td>parameters</td></tr><tr><td rowspan="3">6-ring</td><td>8</td><td>32</td><td> $\overline { { 2 \times 1 6 } }$ </td><td>64</td><td>(8,4)</td><td>197,356</td><td>BF16</td><td>3,200</td><td> $\overline { { 1 \cdot 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 3 \cdot 1 0 ^ { - 3 } } }$ </td><td>0.01</td><td>1.0-3.0</td><td>1,210,000</td><td>tanh</td><td>1</td><td> $\overline { { \mathrm { ~ 4 ~ } \times \mathrm { ~ 3 9 ~ } } }$ </td></tr><tr><td>12</td><td>64</td><td> $4 \times 1 6$ </td><td>128</td><td>(8,4)</td><td>772,012</td><td>BF16</td><td>800</td><td> $1 \cdot 1 0 ^ { - 4 }$ </td><td> $3 \cdot 1 0 ^ { - 3 }$ </td><td>0.01</td><td>0.1-3.0</td><td>1,010,000</td><td>tanh</td><td>1</td><td> $1 \times 1 3 2$ </td></tr><tr><td>16</td><td>64</td><td> $4 \times 1 6$ </td><td>128</td><td>(8,4)</td><td>772,012</td><td>BF16</td><td>800</td><td> $1 \cdot 1 0 ^ { - 4 }$ </td><td> $3 \cdot 1 0 ^ { - 3 }$ </td><td>0.01</td><td>0.1-3.0</td><td>1,010,000</td><td>tanh</td><td>1</td><td> $2 \times 1 5 6$ </td></tr><tr><td rowspan="3">LLD</td><td>8</td><td>32</td><td> $\overline { { 2 \times 1 6 } }$ </td><td>64</td><td>(8,4)</td><td>193,388</td><td>BF16</td><td>800</td><td> $\overline { { 5 \cdot 1 0 ^ { - 6 } } }$ </td><td> $\overline { { 1 \cdot 1 0 ^ { - 3 } } }$ </td><td>0.01</td><td>1.0-3.0</td><td>820,000</td><td>softmax</td><td>1</td><td>1 × 26</td></tr><tr><td>16</td><td>32</td><td> $2 \times 1 6$ </td><td>64</td><td>(8,4)</td><td>193,388</td><td>BF16</td><td>800</td><td> $5 \cdot 1 0 ^ { - 6 }$ </td><td> $1 \cdot 1 0 ^ { - 3 }$ </td><td>0.01</td><td>0.1-3.0</td><td>820,000</td><td>softmax</td><td>1</td><td> $2 \times 1 0 8$ </td></tr><tr><td>24</td><td>128</td><td> $4 \times 3 2$ </td><td>256</td><td>(8,4)</td><td>2,985,260</td><td>BF16</td><td>800</td><td> $5 \cdot 1 0 ^ { - 6 }$ </td><td> $1 \cdot 1 0 ^ { - 3 }$ </td><td>0.01</td><td>0.1-3.0</td><td>820,000</td><td>softmax</td><td>1</td><td> $4 0 \times 6 7$ </td></tr><tr><td rowspan="3">CC8</td><td>8</td><td>32</td><td> $\overline { { 2 \times 1 6 } }$ </td><td>64</td><td>(8,4)</td><td>193,388</td><td>BF16</td><td>800</td><td> $5 \cdot 1 0 ^ { - 6 }$ </td><td> $\overline { { 1 \cdot 1 0 ^ { - 3 } } }$ </td><td>0.01</td><td>1.0-3.0</td><td>820,000</td><td>softmax</td><td>1</td><td>1 × 24</td></tr><tr><td>16</td><td>128</td><td> $4 \times 3 2$ </td><td>256</td><td>(8,4)</td><td>2,985,260</td><td>BF16</td><td>800</td><td> $5 \cdot 1 0 ^ { - 6 }$ </td><td> $1 \cdot 1 0 ^ { - 3 }$ </td><td>0.01</td><td>0.1-3.0</td><td>820,000</td><td>softmax</td><td>1</td><td> $8 \times 4 6$ </td></tr><tr><td>24</td><td>128</td><td> $4 \times 3 2$ </td><td>256 (8,4)</td><td></td><td>2,985,260</td><td>BF16</td><td>800</td><td> $5 \cdot 1 0 ^ { - 6 }$ </td><td> $1 \cdot 1 0 ^ { - 3 }$ </td><td>0.01</td><td>0.1-3.0</td><td>820,000</td><td>softmax</td><td>1</td><td>8× 172</td></tr><tr><td rowspan="2">BB</td><td>6</td><td>64</td><td> $\overline { { 4 \times 1 6 } }$ </td><td>128</td><td>(12,4)</td><td>1,040,828</td><td>BF16</td><td>3,200</td><td> $\overline { { 1 \cdot 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 3 \cdot 1 0 ^ { - 3 } } }$ </td><td>0.01</td><td>0.1-1.5</td><td>1,010,000</td><td>tanh</td><td>8</td><td> $\overline { { \mathrm { ~ 4 ~ } \times \mathrm { ~ 3 2 ~ } } }$ </td></tr><tr><td>12 32</td><td></td><td> $4 \times 8$ </td><td>64</td><td>(12,4)</td><td>266,492</td><td>BF16</td><td>3,200</td><td> $1 \cdot 1 0 ^ { - 4 }$ </td><td> $3 \cdot 1 0 ^ { - 3 }$ </td><td>0.01</td><td> $0 . 1 \mathrm { - } 1 . 5$ </td><td>920,000</td><td>tanh</td><td>8</td><td> $4 \times 8 0$ </td></tr></table>

![](images/c740f34d9a4aebe34ac720d2bbb58b56a7684d1bf406b3b381449a5b1685af5c.jpg)  
FIG. C2: Training curriculum used for LLD d = 24. 5,000 steps per epoch, 820,000 steps in total. Curriculum learning-rate, weight-decay and dropout factors multiply their peak values (shown in Table C1). The error temperature is the maximum training noise $T _ { \mathrm { m a x } }$ of Equation (C17); in every batch, one quarter of the shots is drawn at $T _ { \mathrm { m a x } }$ and the rest uniformly from $[ 0 , T _ { \mathrm { m a x } } ]$ . Shaded intervals mark the warm-up, the noise ramp, the taper of the regularization and the final decay of the learning rate.

Various parameters are varied in a piecewise-linear fashion during training, as indicated in Figure C2. In particular, four quantities are changed: the learning-rate factor, the weight-decay factor, the dropout factor and $T _ { \mathrm { m a x } }$ (see below).

The error model (Equation (3)) is parametrized by a noise parameter $T \in [ 0 , 1 ]$ , which we refer to as the error temperature, through a linear map onto the operating point,

$$
x ( T ) = x _ { \mathrm { s t a r t } } + T \bigl ( x _ { \mathrm { e n d } } - x _ { \mathrm { s t a r t } } \bigr ) ,\tag{C17}
$$

where specific values of $x _ { \mathrm { s t a r t } } , x _ { \mathrm { e n d } }$ are presented in Table C1 for each protocol. During training the curriculum sets a maximum temperature $T _ { \mathrm { m a x } }$ , and each batch mixes shots at $T _ { \mathrm { m a x } }$ (one quarter of the batch) with shots whose temperatures are drawn uniformly from $[ 0 , T _ { \mathrm { m a x } } ]$ . This mixture ensures there are always some ‘easy’ shots in every batch, and trains the model for the whole range of operating points.

We refer to the attached data files for specific curricula used for Figure 3.

We use AdamW [101, 142] with three parameter groups. Normalization parameters and all biases use the scheduled learning rate without weight decay; residual gates use one tenth of that learning rate without weight decay; all remaining trainable parameters use the scheduled learning rate and weight decay. Gradients are clipped element-wise to [ 1, 1], and training uses graph compilation with the numerical precision specified for each model in Table C1.

## a. Sparse attention kernels

For each head $r ,$ from Equation (C4a), the projected states are

$$
k _ { b } ^ { ( r ) } = W _ { K } ^ { ( r ) } y _ { b }\tag{C18a}
$$

(a)  
![](images/130aa0e126c0bc290910ca43b020b053310159f6a51c4200c754c165b4f5dad6.jpg)

(b)  
![](images/c1f585f7f8e26fa53037a3179678423e8d7cbb134f9fe11df9ee19c62eaa2a57.jpg)  
FIG. C3: Mean training-step time (a) and peak allocated GPU memory (b) of models that contain only sparse attention blocks or only feedforward blocks, for the LLD network. The sparse attention is evaluated with the Triton kernel and with the generic implementation that gathers projected states into a padded tensor before scaled dot-product attention (SDPA). Benchmarks performed on AMD Instinct<sup>TM</sup> MI350X GPUs.

$$
v _ { b } ^ { ( r ) } = W _ { V } ^ { ( r ) } y _ { b } .\tag{C18b}
$$

Our $\mathrm { P y }$ Torch baseline gathers these vectors into padded tensors, duplicating them for each $x _ { a }$ whose neighborhood contains b. Across all heads, each tensor has size $B \times T _ { x } \times n _ { \mathrm { m a x } } \times d _ { \mathrm { M } }$ , where B is the batch size, $T _ { x }$ is the number of states being updated (error mechanisms or checks), and $n _ { \mathrm { m a x } } = \operatorname* { m a x } _ { a } | { \mathrm { n b d } } ( a )$ .

We instead implemented the forward and backward passes as Triton kernels [143] that operate directly on a compressed sparse row (CSR) representation of the Tanner graph. The forward kernel assigns one program to each state $x _ { a } .$ , head and batch element. It processes the neighboring states $y _ { b }$ in blocks and computes the softmax normalization and weighted sum incrementally, following Ref. [144]. Alongside the projected input states, it retains the output and the log-sum-exp of the scores, without storing the attention weights. Performance of these kernels is illustrated in Figure C3.

The backward pass recomputes the attention weights from the stored $\log { - } \mathrm { s u m - e x p }$ . On AMD GPUs, gradients of the projected $x _ { a }$ states are accumulated per state a, while gradients of the projected $y _ { b }$ states are accumulated per state b using the transposed CSR structure, avoiding atomic additions. The membrane statistics used in Equation (C9) are computed by an analogous CSR reduction kernel.

![](images/583eaace5695407ea22c5ed5ad4926267ef9879cf81c82f2138e19104eea1725.jpg)  
FIG. D1: The differences between the unquantized (Figure C1) and quantized architectures. Blue circles labeled $A ( A ^ { + } )$ denote signed (unsigned) activation quantizers. Input embeddings and the membrane query seed use the same precision as the network body. All matrix weights, including the output projection, are ternary. Block outputs are quantized before addition to the wider residual stream. (a) Quantized sparse cross-attention block. (b) Quantized feed-forward block. (c) Quantized output block. The FP32 logits enter a ShiftMax head that produces four-bit log-probability codes for the integer vote.

## Appendix D: Quantized architecture and training

This appendix describes the low-precision models evaluated in Section V B, their training, and the results that informed their design. The general decoder architecture is given in Appendix C 1, and the ShiftMax and ShiftL1Norm algorithms in Appendix G. All low-precision results are obtained in simulation, with quantized operands and a higher-precision residual stream.

## 1. Numerical representation and logical readout

The final models have $Q \ : = \ : 4$ message-passing layers, the one-sided interface layer, and $P = 2$ perceiver layers, with $d _ { \mathrm { F } } = 4 \times d _ { \mathrm { M } }$ . The $d _ { \mathrm { M } } = 1 6$ and $d _ { \mathrm { M } } = 3 2$ models use $n _ { \mathrm { H } } = 2$ attention heads, while the $d _ { \mathrm { M } } = 4 8$ model uses $n _ { \mathrm { H } } = 3$ . The head dimension is $d _ { \mathrm { H } } = d _ { \mathrm { M } } / n _ { \mathrm { H } }$ . The model dimension and parameter count are independent of the protocol distance d.

Quantizers. To simulate reduced-precision arithmetic, we insert fake quantizers into the model (blue A and W nodes in Figure D1). These quantization operations transition to low-precision computation via streamlining [113]. A b-bit uniform quantizer with scale $s > 0$ maps a real input z to

$$
{ \mathrm { q u a n t } } _ { b } ( z ; \ s ) = s \ { \mathrm { c l i p } } \left( { \mathrm { r o u n d } } \left( { \frac { z } { s } } \right) , q _ { b } ^ { ( { \mathrm { m i n } } ) } , q _ { b } ^ { ( { \mathrm { m a x } } ) } \right) ,\tag{D1}
$$

where

$$
\mathrm { c l i p } ( x , a , b ) : = \operatorname* { m i n } ( \operatorname* { m a x } ( x , a ) , b ) .\tag{D2}
$$

For activations, $q _ { b } ^ { ( \mathrm { m i n } ) }$ and $q _ { b } ^ { ( \operatorname* { m a x } ) }$ are the lower and upper integer quantization bounds, determined by the precision b and whether the values are signed or unsigned:

$$
\begin{array} { l l } { ( q _ { b } ^ { ( \mathrm { m i n } ) } , q _ { b } ^ { ( \mathrm { m a x } ) } ) = ( - 2 ^ { b - 1 } , 2 ^ { b - 1 } - 1 ) } \\ { ( q _ { b } ^ { ( \mathrm { m i n } ) } , q _ { b } ^ { ( \mathrm { m a x } ) } ) = ( 0 , 2 ^ { b } - 1 ) } \end{array}
$$

signed

(D3a)

unsigned.

(D3b)

Weights use the symmetric range $q _ { b } ^ { \mathrm { ( m i n ) } } = - ( 2 ^ { b - 1 } - 1 ) \mathrm { t o } q _ { b } ^ { \mathrm { ( m a x ) } } = 2 ^ { b - 1 } - 1$ , giving ternary values $\{ - s , 0 , s \}$ at two bits.

Weight scales are learned per output channel and activation scales per tensor. We initialize the scales by minimizing the mean-squared quantization error, using eight batches of 32 shots across the training noise range to calibrate the activation scales.

![](images/6df9369d6fc50fd530a4ab42b6623d1bab4bb005b9ff8228834da6f0602719d4.jpg)

![](images/9ac0036d7bfcd654fa10b9a52e5a7c844ce3e84fcc64123434f12f65839e4bd2.jpg)

![](images/4d5824db46d46a202e505bc0ce6de5d3e9562fef4ddc96f90f7bf71fea2fe199.jpg)

![](images/3ae00c1578c4dc1777ed7bc52287e672768864fa95e93ce29d245e69639988eb.jpg)  
FIG. D2: Quantizer transfer functions of Equation (D1) used in the $\mathrm { W _ { 2 } A _ { 2 } }$ models. Input and output in units of the learned scale $s \ ( \mathrm { i . e . }$ quan $\mathrm { t } _ { b } ( z ; s ) / s$ as a function of $z / s ) .$ . (a) Ternary weights. (b) Signed two-bit activations. (c) Unsigned two-bit activations after ReLU. (d) Membrane values with one sign bit and two magnitude bits, preserving a nonzero reliability on every input. The shaded intervals indicate the clipping range within which the straight-through derivative is nonzero. $\mathrm { W _ { 2 } A _ { 4 } }$ uses four activation or magnitude bits in (b)–(d), while its weights remain ternary.

Quantization is simulated in Brevitas [109], with a straight-through estimate (STE) [112] equal to one inside the clipping interval and zero outside it.

Matrix products receive quantized operands, but their accumulators and other intermediate quantities can have higher precision. In particular, the residual stream, residual gates, normalization gains and biases, and logits remain wider in both simulation and in hardware. All input and output quantizers use their final precision.

The forward pass uses ShiftMax for attention, ShiftL1Norm for normalization, and ReLU. For backpropagation through attention, we use the derivative of exact softmax for the scores and the forward-pass ShiftMax weights for the values. The normalization scale is treated as constant in the backward pass, with zero gradient at zero or flushed inputs. These approximate gradients are used only during training, allowing us to update the weights while using the hardware operations in the forward pass.

Membrane aggregation. The quantized parity component uses offset min-sum [145]. For quantized feature $[ h _ { i } ] _ { k }$ on mechanism i, membrane support supp(µ), and learned channel offset $\beta _ { k }$ , the aggregation computes

$$
[ \pi _ { \mu } ] _ { k } = \mathrm { R e L U } \{ \operatorname* { m i n } _ { i \in \mathrm { s u p p } ( \mu ) } | [ h _ { i } ] _ { k } | - \beta _ { k } \} \left( \prod _ { i \in \mathrm { s u p p } ( \mu ) } \mathrm { s g n } ( [ h _ { i } ] _ { k } ) \right) .\tag{D4}
$$

Input values have one sign bit and A magnitude bits, with nonzero levels $\pm s ( j + \frac { 1 } { 2 } )$ for $j = 0 , \dotsc , 2 ^ { A } - 1$ . Rounding an individual contribution therefore cannot erase its sign, although the aggregate can be zero after the offset and clipping. In addition, the mean and second-moment features are computed

$$
\left[ u _ { \mu } \right] _ { k } = \frac { 1 } { | \operatorname { s u p p } ( \mu ) \ : | } \sum _ { i \in \operatorname { s u p p } ( \mu ) } \left[ h _ { i } \right] _ { k } ,\tag{D5a}
$$

$$
\left[ s _ { \mu } \right] _ { k } = \frac { 1 } { | \operatorname { s u p p } ( \mu ) \ : | } \sum _ { i \in \operatorname { s u p p } ( \mu ) } \left[ h _ { i } \right] _ { k } ^ { 2 } .\tag{D5b}
$$

All three statistics share one quantizer scale before their ternary projection (to dimension $d _ { \mathrm { M } } )$ , and are added to a projection of the membrane state $m _ { \mu }$ before a ReLU and the output projection. In the backward pass, each sgn in Equation (D4) uses a straight-through estimator with unit derivative and the product is differentiated as usual, so the gradient of the sign product with respect to $\begin{array} { r } { \left[ h _ { i } \right] _ { k } \mathrm { i s } \prod _ { j \in \mathrm { s u p p } ( \mu ) , j \neq i } \mathrm { s g n } \left( \left[ h _ { j } \right] _ { k } \right) } \end{array}$ . The minimum passes its gradient only to the input that attains it. In the hardware implementation the parity term of Equation (D4) is an XOR tree over sign bits and a minimum tree over the magnitudes, and the two moments are adder trees

Integer voting. For each membrane representative $\mu ,$ the ShiftMax head (Figure D1c) converts the two logits into unsigned four-bit negative log-probability codes, $[ \ell _ { \mu } ] _ { 0 }$ and $[ \ell _ { \mu } ] _ { 1 }$ . Up to the fixed-point approximations, rounding, and clipping in Shift-Max, their difference satisfies

$$
\left[ \ell _ { \mu } \right] _ { 0 } - \left[ \ell _ { \mu } \right] _ { 1 } \approx \frac { \left[ \lambda _ { \mu } \right] _ { 1 } - \left[ \lambda _ { \mu } \right] _ { 0 } } { \ln 2 } .\tag{D6}
$$

This difference approximates the predicted log-odds of a flip, expressed in bits. Positive values favor a flip, negative values favor no flip, and larger magnitudes indicate greater confidence. We combine the scores of equivalent representatives after aligning them to the reference membrane, as in Section III C 1.

Writing $\chi _ { \mu } ( \sigma ) = ( - 1 ) ^ { \tau _ { \mu } }$ for the alignment sign determined by the transition syndrome, the integer vote for equivalence class κ is

$$
V _ { \varkappa } = \sum _ { \mu \in \varkappa } \chi _ { \mu } ( \sigma ) \times \left( [ \ell _ { \mu } ] _ { 0 } - [ \ell _ { \mu } ] _ { 1 } \right) .\tag{D7}
$$

This is a quantized approximation to the syndrome-aligned logit vote of Equation (10). The decoder predicts a flip for logical class κ if $V _ { \varkappa } > 0$ , and no flip otherwise.

Each representative contributes an integer between 15 and 15. With four representatives per class at $d = 8 ,$ the combined vote lies between 60 and 60 and fits in a signed seven-bit accumulator.

## 2. Training and model selection

The LLD training curriculum uses the noise model defined in Equation (3), parametrized as in Equation (C17), with $x _ { \mathrm { s t a r t } } =$ 1.0% and $x _ { \mathrm { e n d } } = 1 . 5 \%$

Training starts from random initialization with the core quantizers active. The standard schedule has 210,000 optimizer steps at batch size 200. It consists of 10,000 warm-up steps at $T = 0$ , a 100,000-step ramp to $T = 1$ , and 100,000 steps at the maximum noise while weight decay and dropout taper to zero. One quarter of each batch is drawn at the current maximum temperature and the rest uniformly below it. Longer schedules extend the latter phases to 410,000 or 810,000 total steps.

The training curriculum for the quantized models is similar to that described in Appendix C 2, but with a larger maximal learning rate. The quantizer scales usually use a learning rate of $1 . 5 \times 1 0 ^ { - 4 }$

Temporary higher precision. Training initially uses 8-bit activation quantizers on the erasure and syndrome embedding and membrane query seed, together with 8-bit output-projection weights $\mathrm { ( W _ { 8 } A _ { 8 } ) }$ . All other matrix weights and core activations use their target precision from the start. We then reduce the three input activation quantizers to the core activation precision and the output weights to ternary values, and continue training.

The four changed quantizers are re-calibrated by minimizing mean-squared quantization error. Training continues for 50,000 steps with all other quantizer scales frozen. The re-calibrated scales use one tenth of the weight learning rate, and the model weights continue to train. We add a KL-divergence distillation loss, with coefficient 0.5 and temperature 2, from a $d _ { \mathrm { M } } = 2 4$ $\mathrm { W _ { 2 } A _ { 4 } }$ teacher with higher-precision input and output quantizers.

Reducing precision without retraining increases the logical error rate by factors of 1.4–2.5 in the tested models. Further training recovers most of this loss, leaving increases of $7 - 1 7 \% \mathrm { a t } x = 1 . 5 \%$ in paired conversion tests. These comparisons are with the models before conversion, not with full-precision decoders. Neither the teacher nor the temporary 8-bit quantizers are needed for final inference.

Checkpoints and further training runs are selected on separate validation samples. Selection among further training runs uses at least $2 \times 1 0 ^ { 5 }$ shots at $x = 1 . 5 \%$ , with the starting checkpoint also eligible. Final $d = 8$ rates use $\mathrm { { \bar { 5 } } \times 1 0 ^ { 6 } }$ independent shots per operating point. For comparisons on shared test samples, we use the shots for which only one decoder fails.

## 3. Design choices

In the remainder of this section, we present several studies which were used to make design decisions for the final hardwarecompatible models. The key takeaways are:

• Training hyperparameters must be chosen carefully: learning rate and training seed substantially affect LER.

• Width and precision must be chosen together.

• Aggregation involves a trade-off: product aggregation gives better accuracy in the small-block comparison; min-sum avoids its underflow problems at larger sizes.

• Train with the arithmetic you will deploy: changing normalization rules after training can substantially degrade accuracy.

Learning rate and initialization. Increasing the learning rate is more effective than simply extending a low-rate training run. In a controlled $d _ { \mathrm { M } } = 3 2 , \mathrm { W } _ { 2 } \mathrm { A } _ { 4 }$ study with $( Q , P ) = ( 8 , 4 )$ , tripling the weight learning rate while holding the scale learning rate fixed reduces the LER to 0.54 of its original value for the final raw weights. We also observe that random initialization is important. Seven $d _ { \mathrm { M } } = 4 8 , \mathrm { W _ { 2 } A _ { 2 } }$ runs with the same training recipe span a factor 2.4 in logical error rate, much larger than their approximately 4% sampling uncertainty. These studies use the intermediate precision settings from training and inform the choice of learning rate and model selection.

(a) $\mathrm { W _ { 2 } A _ { 4 } }$  
![](images/28d05ce1382cfb5e1072234e5d028efb8c10e4579b6bb4bed0c43e1ca89fda23.jpg)

(b) W<sub>2</sub>A<sub>2</sub>  
![](images/4386875ddb4792e3f4280878c6da2d25d803e63d4efd1a853cc865a4714c29d1.jpg)  
FIG. D3: Model-dimension and optimization scans for the $d = 8 \mathrm { L L D }$ block at $x = 1 . 5 \%$ , with $( Q , P ) = ( 4 , 2 )$ . Learning rates are shown as multiples of the base rate $\eta _ { 0 } = 1 . \bar { 5 } \times 1 0 ^ { - 5 }$

(a) W $\mathrm { \ J _ { 2 A _ { 4 } } }$ models trained for 210,000 steps at three learning rates. (b) $\mathrm { W _ { 2 } A _ { 2 } }$ models with two recipes that change the learning rate and training length together. The dashed line reproduces the $\mathrm { W _ { 2 } A _ { 4 } } , \eta _ { 0 } \times 1 0$ reference from panel (a).  
![](images/a756ffd73fc95f07742de11ee139ce303334ae7e8d5f2338560b3c1495432045.jpg)  
FIG. D4: Learning curves at $x = 1 . 5 \%$ for the $d = 8 \mathrm { L L D }$ block, with $d _ { \mathrm { M } } = 1 2 8 , ( Q , P ) = ( 4 , 2 )$ . Lines are means over a number of independently seeded training runs, as indicated in the legend. The orange point pools four quantized runs evaluated on $4 \times 1 0 ^ { 6 }$ shots in total. All curves use minimum-entropy logit selection instead of the syndrome-aligned vote of Equation (10). The full-precision groups compare product aggregation, min-sum aggregation, and min-sum with hardware-oriented operations; the quantized group uses $\mathrm { W _ { 2 } A _ { 2 } }$ with min-sum and hardware-oriented operations.

Precision and model dimension. Increasing model dimension and learning rate improves the logical error rate of the $\mathrm { W _ { 2 } A _ { 4 } }$ models in Figure D3a. These historical scans use ternary core weights and the indicated core activation precision, while retaining 8-bit activation quantizers on the input embeddings and membrane query seed, and 8-bit output-projection weights. At matched model dimension, the plotted $\mathrm { W _ { 2 } A _ { 2 } }$ runs have higher logical error rates than the $\mathrm { W _ { 2 } A _ { 4 } }$ reference. The two $\mathrm { W _ { 2 } A _ { 2 } }$ recipes change the learning rate and training length together, and the models also differ in their training and normalization. These comparisons therefore do not isolate the effect of activation precision. The choice between a smaller $\mathrm { W _ { 2 } A _ { 4 } }$ model and a larger $\mathrm { W _ { 2 } A _ { 2 } }$ model additionally requires consideration of hardware resource use and latency.

Separating architecture and precision changes. Figure D4 compares changes in membrane aggregation, arithmetic operations and core precision at $d _ { \mathrm { M } } = 1 2 8$ . With min-sum aggregation and hardware-oriented forward operations, the quantized and unquantized models achieve comparable logical error rates in these experiments. The training configurations differ, including their learning rates, so this comparison does not isolate the accuracy cost of quantization.

Product aggregation, which replaces Equation (C9c) with $\begin{array} { r } { \left[ \pi _ { \mu } \right] _ { k } = \prod _ { i \in \mathrm { s u p p } ( \mu ) } \left[ w _ { i } \right] _ { k } , } \end{array}$ gives the lowest LERs in this $d = 8$ comparison. However, these products can underflow as membrane supports grow, motivating the use of min-sum aggregation at larger protocol sizes.

Matching the normalization arithmetic. The exact normalization rule used during training is important. Replacing round to-nearest logarithmic scaling by leading-bit truncation without retraining raises the logical error rate by factors 2.0–3.9 in the tested models, and we therefore train with the exponent rule used in the implementation. A learned bias after normalization allows a different quantization threshold for each channel. For $d _ { \mathrm { M } } = 4 8$ , the normalization core pads to 64 channels before computing the mean magnitude and crops the result afterwards. The 16 added zeros leave the total absolute feature magnitude unchanged, but the normalization computes its mean by dividing by 64 rather than 48. We use this same padding and normalization rule during training.

These studies motivate tuning the learning rate and evaluating multiple training seeds, selecting model dimension and activation precision jointly, and training with the numerical operations used during inference. They also motivate min-sum aggregation for robustness at larger protocol sizes. The comparisons illustrate practical design choices rather than isolating a universal accuracy cost of quantization.

![](images/4423a998932854a81fce190728254495d2f542ae8937288132cda2aa58d66f83.jpg)  
(a) First cross-attention mask: Error-mechanism tokens attend to check tokens.

![](images/4461e9f040e0a6495a53ff9c2402c340bce184b5a86228b39e99efedcccc937e.jpg)  
(b) Second cross-attention mask: Check tokens attend to error-mechanism tokens.  
FIG. E1: (Duplicated from Figure 6) Cross-attention masks used in the first message passing layer. Hubs and stars are processed in parallel. The colors denote the order that input data is streamed into the cross attention block. Star-check connections form compact local blocks, while hub-check connections link mechanisms across stars. Masks used for later layers are permuted to keep data flowing efficiently

## Appendix E: Cross-attention hardware mapping

To implement the model for low-latency deployment we deploy a dataflow architecture for the processor on the FPGA. In a dataflow architecture, separate tokens are worked on in different stages of a pipeline; this is a common technique to drastically improve throughput without sacrificing much latency. Such an architecture works particularly well for token-wise operations, such as feed-forward layers and linear projections, whose output for each token depends only on that token’s input. The benefits begin to break down when there are dependencies between different tokens, such as in the cross-attention blocks, as token farther along in a pipeline must wait for tokens they depend on that are earlier in the pipeline. This is referred to as “draining” the pipeline.

The attention mechanism requires tokens to interact with one another, causing the pipe to drain until inter-token dependencies are met. Naively, this would eliminate much of the benefit of pipeline parallelism. Fortunately, the sparse attention pattern, ultimately a consequence of the geometry of the protocol, provides some opportunity to limit the number of cycles a particular token spends waiting for its key-value context. The relationship between star and hub tokens (introduced in Section V C 1) can be projected into three dimensions with periodic boundaries as shown in Figure E2a. As transformers are well known to be permutation equivariant, we may choose to permute the star and hub tokens to a more optimal order. Furthermore, since queries arrive via the residual connections, they do not lie on the critical path and we may reorder them between layers. The choice of order therefore comes down to processing tokens in such a way that locality is roughly maintained layer-to-layer despite the periodic boundaries and serialization of the three-dimensional structure into a one-dimensional token stream.

We choose to walk the cube in each dimension as shown in Figure E2b; however, as the edges are periodic we shift the origin for each message-passing layer. For the cost of a slight drain at the start, this ensures that each cross-attention’s first touched hubs are the previous cross-attention’s first processed hubs. This shift improves end-to-end latency by over twenty percent.

Figure E1 shows the attention mask for the first message-passing layer. The example contains 1,536 error-mechanism tokens and 384 check tokens, partitioned into 64 hubs and 320 star checks. Each mask has 4,608 allowed attention connections. The graph is unchanged between stages; the displayed coordinates follow the current input order, while colors identify arrival and emission groups. Applying each stage’s output permutation (resulting from the origin-shift) gives the input ordering of the next stage, making the ordering changes across the message-passing stack explicit.

The attention mask itself has two key properties that make it amenable to hardware implementation. The intra-star attention pattern is consistent throughout so in addition to tight reuse of the query, key, and value vectors we can program the mask into the FPGA to fully avoid indexing, communication, and masking overhead commonly dealt with in sparse attention. The intra-star mask is shown in Figure E3. The hub to star attention is dynamic; however, the geometry allows us to partition the hubs into 8 banks such that we encounter no bank conflicts. This partition allows us to avoid an expensive crossbar and data replication.

![](images/51e652f58e769340b61d21a23839202539e6481ff67278e0094cec477ccddc13.jpg)  
(a)

![](images/41f55b363a33a08637cebbb83b79d13aca7963d672e68ffc414c9999788a82dc.jpg)  
(b)

![](images/7a2a37464747ee9095254c2c3b5328fc5a520e41408c6fbcb7f4ed71bdd35281.jpg)  
(c)

![](images/4a9f51dfce4c22d7239ec699f2e7d5ae01d0a4e036345e86fe5a032886dd32ae.jpg)  
(d)  
FIG. E2: (a) A graph of the connectivity between stars, with an example in blue highlight, and hubs, with an example in gray. Star tokens from this star must attend to hub tokens from these eight hubs and vice-versa. (b) The order in which star token blocks are processed by each cross attention layer. One can picture the token reorder between layers as shifting the whole lattice down one in every axis, with the sheets $x , y , z = 0$ wrapping around and the point (1, 1, 1) becoming the new start. (c) Hubs numbered based on the earliest they are attended to by a star. (d) Hubs numbered based on the latest they attend to a star. Although the first needed hubs are all in x, $y , z \in \{ 0 , 1 \}$ , the hub (1, 1, 1) is the first that is no longer needed.

![](images/327deea1b319af56440fc3162d183f43f7107176ad6d3e2da90fa559e9df56bd.jpg)

![](images/9b71d3364735188b9f2db68d459a7acf7a5b812cb38b8b3ea4eacc4ebb416886.jpg)  
FIG. E3: The intra-star attention masks for both cross-attention blocks. Identical masks are used for each star. Recall that a star is formed from an octahedral check (check slot 0), and its four incident triangular checks (Figure B1b). The 24 error mechanisms are the 12 XX and 12 ZZ outcomes around the octahedron. Since the triangular checks for LLD are formed of $Y Y$ outcomes, they are weight 6, while the octahedra are formed of ZZ outcomes, so are weight 12.

![](images/2c1f1c5e2752a3e6ae6b68f02c4bfec01174e1a9f16e53d721a5371776001b73.jpg)  
FIG. F1: Routed device view of Q-PRISM on the VP1902 running at 250 MHz, with every placed cell tinted by the layer it belongs to.

## Appendix F: Optimizations to achieve high-frequency FPGA mapping

As previously mentioned, we target both 250 MHz and 400 MHz to show the out-of-the-box performance of the design as well as what is possible with additional timing optimizations. We map the entire network at 250 MHz and demonstrate that, with these optimizations, both message-passing and perceiver layers meet timing at 400 MHz. Due to the pipelined design and available resource budget on the VP1902, the total frequency of the network can be well-predicted by the component frequencies. We leave the full mapping at 400 MHz for future work.

The Q-PRISM FPGA accelerator leverages aggressive pipelining in the individual components to achieve a high operating frequency. The perceiver layers can already meet timing at 400 MHz, but the message passing layers cannot without additional optimizations. To enable the message passing layer to meet timing at 400 MHz, we implement additional architecture and floorplanning optimizations designed to exploit the nature of the VP1902’s FPGA fabric [146, 147].

Figure F1 shows the full network mapped to the VP1902 running at 250 MHz. This was achievable by a simple coarse-grained floorplanning strategy that distributes the layers across the chips of the FPGA. Due to the pipeline connectivity in combination with the available resources, this was sufficient to hit 250 MHz.

We leverage aggressive, fine-grained placement constraints via floorplanning to optimize the timing of the message passing layer. Reaching 400 MHz requires a careful balancing act between the Vivado<sup>TM</sup> placer and human-driven floorplanning. That balance is necessary because the message-passing layer comprises many different hardware blocks, and the post-synthesis netlist does not express how those blocks are arranged relative to one another. The computational graph, on the other hand, tells us which nodes are neighbors, and therefore which ones should sit spatially next to each other on the device. We therefore supply that locality to the placer as placement hints, starting from an initial hand-placed seed floorplan and iterating on it by inspecting where the placer agreed with, and departed from, our intent.

There are three strategies primarily contributing to the success of mapping the message passing layer at high frequencies:

• hard placement constraints for the main compute blocks, without excluding other logic or confining routing;

![](images/967ee4a2b876b8f3460ac0f431cda3c45492007b1c54ef3e6b2d6480951c04db.jpg)

(a)  
![](images/e46e5a48800a38370891f9b9ada985a578a91b2cb6dafc569334bb7d60c7d187.jpg)  
(b)  
FIG. F2: A close-up of a single message passing layer, showing a much smaller portion of the compute fabric than Figure F1. (a) Routed device view of the Q-PRISM message passing layer on the VP1902 running at 400 MHz, with every placed cell tinted by the operator it belongs to. ‘Other operators’ includes all miscellaneous operators, such as FIFOs and residual forks. (b) Dataflow mapping on the device.

• iterative co-design of the RTL, memory-resource usage, and pblock geometry;

• coarse, overlapping pblocks for supporting logic, with placement constraints omitted for globally spanning nodes.

Figure F2 shows the placed-and-routed message passing layer at 400 MHz with all three applied, and we refer to it throughout. We constrain each of the main compute blocks to a specific location on the device allocated with sufficient resources, meaning that cells belonging to that node can only be placed within the block. Non-exclusive placement means that other operations can also be placed inside the allocated region, as can be seen, for example, by the lime green (other operators) inclusions within the region designated for the erasure attention. Leaving CONTAIN ROUTING at its default value of FALSE allows signals within a node to route outside its pblock, while placement remains constrained to the region. This gives the router flexibility to find more direct paths around congested resources.

Furthermore, we maximize the co-design between the RTL, the Vivado<sup>TM</sup> placer, and human-driven floorplanning through multiple iterations. Each iteration starts from the placer’s results: where it falls short, we manually adjust the geometry of the pblock responsible so that the hardware primitives within it are placed more optimally, as carry chains, for example, have the shortest route when placed vertically rather than horizontally. Where the geometry alone is not enough and paths continue to fail timing, we return to the RTL, adding registers to break up long routes and deep logic levels. At the RTL level we also adjust the balance between BRAM and URAM to better match the VP1902’s memory layout, which yields denser placement and correspondingly shorter routes for the next iteration to work with.

Our final strategy uses overlapping pblocks. Overly restrictive manual floorplanning can:

• limit the freedom of the Vivado<sup>TM</sup> placer to optimize placement;

• prevent cells from different hardware blocks from sharing FPGA slices, reducing placement density.

For supporting logic, such as first-in, first-out (FIFO) buffers between hardware blocks, we assign coarse placement regions that overlap neighboring blocks. We leave fine-grained placement within these regions to Vivado<sup>TM</sup>. For globally spanning logic, such as the residual connections, we omit manual placement entirely, allowing Vivado<sup>TM</sup> to place them inside or outside the manually defined regions. Together, these strategies produce a message passing layer design that meets timing at 400 MHz.

## Appendix G: Quantized softmax and normalization for efficient hardware mapping

As discussed, in order to comply with the DSP resources available on the VP1902, modifications to the softmax and normalization operations were performed in order to map those operations to LUTs. For softmax, a modified implementation of ShiftMax [148] was leveraged, which allows implementing the softmax function purely using LUTs. The modified ShiftMax implementation can be seen in Algorithm 1. With the same motivation, we take motivation from Refs. [149 and 150] and imple ment a shift-only approximation of L1 normalization, which we refer to as ShiftL1Norm, which can also be implemented purely in LUTs, and can be seen in Algorithm 2. These approximations are baked into the ML decoder during QAT to ensure that these approximations are built into the training process. These modifications allow the design to fit within the resource budget of the VP1902.

## Algorithm 1 ShiftMax integer core

$$
S \in \mathbb { R } ^ { T _ { q } \times H \times T _ { k } } ,
$$

where S contains the scaled attention scores for one decoder input and H is the number of attention heads. For each query q, let $\mathcal { N } ( q )$ be its nonempty set of valid key positions. Due to our quantization, s lies on a constrained grid, allowing Float2Fixed to be implemented cheaply with a lookup table.

Use $F = 8$ fractional bits, signed input width $W = 1 7 ,$ internal exponent offset $C = 3 0 .$ , and log-code width $L = 4 .$ The integer core produces unsigned L-bit log codes ℓ at valid key positions. The subsequent computation in sparse cross-attention accounts for this outpu representation.

```latex
1: for $q = 1$ to $T _ { q }$ do ▷ Process each query independently
2: for $h = 1$ to H do ▷ Process each attention head independently
3: $s \gets S [ q , h , \mathcal { N } ( q ) ]$ ▷ Gather the $\vert \mathcal { N } ( q ) \vert$ valid scores
4: $x  \mathrm { F l o a t 2 F i x e d } _ { W , F } ( s )$ ▷ Convert constrained scores
5: $m \gets \operatorname* { m a x } _ { k } x _ { k }$ ▷ Find the largest quantized score
6: $d  m - x$ ▷ Distance below maximum; clip appropriately
7: $z  d + ( d \gg 1 ) - ( d \gg 4 )$ ▷ Approximate $d \log _ { 2 } ( e ) ;$ retain F fractional bits
8: $n  z \gg F$ ▷ Keep the integer part; discard F fractional bits
9: $u \gets 2 ^ { C } \gg \operatorname* { m i n } ( n , C )$ ▷ Form scaled power-of-two weights
10: $\begin{array} { r } { r \gets \left\lfloor \log _ { 2 } \left( \sum _ { k } \dot { u } _ { k } \right) \right\rfloor } \end{array}$ ▷ Implemented by finding the highest set bit of the weight sum
11: $\ell [ q , h \dot { , } \mathcal { N } ( q ) ] \stackrel { \cdot } {  } n \dot { + } ( r - C )$ ▷ Adjust for row sum; clip appropriately
12: end for ▷ Finish this head
13: end for ▷ Finish this query
14: return ℓ ▷ Return integer log codes for all rows
```

## Algorithm 2 ShiftL1Norm kernel

$$
\boldsymbol { X } \in \mathbb { R } ^ { T \times D } , \qquad \boldsymbol { \gamma } \in \mathbb { R } ^ { D } ,
$$

where X contains the input features for one decoder input and $\gamma$ is the learned per-channel gain. Use the exponent limit $K = 2 0 .$ . The normalization scales each token by a power of two before applying the channel gain.

1: for $t = 1$ to T do ▷ Process each token independently   
2: $\mu  { \frac { 1 } { D } } \sum _ { d = 1 } ^ { D } | X [ t , d ] |$ ▷ Compute the mean   
3: $k  \bar { \mathrm { r o u n d } } ( \log _ { 2 } ( \mu ) )$ ▷ Nearest power-of-two exponent; clip appropriately   
4: $u \gets X [ t , : ] \dot { 2 } ^ { - \bar { k } }$ ▷ Normalize by shifting each value’s exponent   
5: $\mathrm { o u t } [ t , : ]  u \odot \gamma$ ▷ Apply the learned per-channel gain   
6: end for ▷ Finish this token   
7: return out ▷ Return normalized features