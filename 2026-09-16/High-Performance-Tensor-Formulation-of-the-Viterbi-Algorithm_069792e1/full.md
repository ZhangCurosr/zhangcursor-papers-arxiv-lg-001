# High-Performance Tensor Formulation of the Viterbi Algorithm for Hidden Semi-Markov Models

Lorenzo Piarulli

Elia Belli

Daniele De Sensi

Department of Computer Science

Department of Computer Science

Department of Computer Science

Sapienza University of Rome

Sapienza University of Rome

piarulli@di.uniroma1.it

Sapienza University of Rome

belli.2006305@studenti.uniroma1.it

desensi@di.uniroma1.it

Abstract—Hidden Semi-Markov Models (HSMMs) are fundamental probabilistic models widely adopted across diverse domains, from computational biology to finance and signal processing. The Viterbi algorithm decodes the most likely state sequence given an HSMM and can be applied iteratively for ab initio model learning. However, existing Viterbi implementations remain sequential, and GPU-accelerated solutions are entirely absent, making HSMM decoding impractical for largescale workloads. We present a tensor-based formulation of the Viterbi algorithm for HSMMs, restructuring the inner loops into tensor operations that naturally map onto SIMD units and massively parallel architectures. Building on this formulation, we provide optimized implementations spanning single- and multicore CPUs, and, for the first time, GPU. Experimental evaluation demonstrates speedups of up to 14× on a single core, over 200× with multi-core, and over 570× on GPU over the state-of-the-art sequential baseline, establishing a new performance baseline for large-scale HSMM decoding.

Index Terms—Hidden Markov Models, Tensors, GPU

## I. INTRODUCTION

High-Performance Computing (HPC) architectures are evolving at an extraordinary rate. Over the past decades, we have transitioned from CPU-only computation to massive multicore processors, and then to GPUs. Now, driven by the rise of artificial intelligence, entirely new accelerator architectures are emerging, including dataflow engines, systolic arrays, and SIMD-centric designs, that promise unprecedented throughput for structured, regular computations. Yet, while hardware evolves rapidly, the algorithms that run on it do not always keep up. Application scientists tend to be conservative: their software frameworks are enormously complex, and restructuring a working codebase to exploit a new architecture is a daunting, error-prone endeavor. In many cases, the cost and complexity of migration simply outweigh the perceived benefit, and teams understandably choose to keep a working pipeline rather than risk breaking it for uncertain gains. As a result, many fundamental algorithms, including those that today underpin astonishing scientific discoveries, remain anchored to decades-old sequential formulations that hide parallelization possibilities and, consequently, remain confined to single-threaded CPU execution. Worse still, practitioners often resort to simplified or truncated versions of their models simply because the full, general formulation would be computationally infeasible on the sequential hardware.

Hidden Semi-Markov Models (HSMMs) are versatile probabilistic frameworks with applications spanning diverse fields, from genome annotation and segmentation in computational biology [1] to finance [2], speech recognition [3], and signal processing [4]. As a generalization of the classical Hidden Markov Model (HMM), an HSMM describes a system transitioning through a finite set of hidden states at discrete time intervals. In this paradigm, the system’s internal state remains unobservable, manifesting only through visible emissions governed by state-specific probabilities, while state progression is regulated by a fixed transition matrix.

A practical illustration of this framework is found in sleepcycle prediction from physiological data: heartbeat measurements serve as the observations, while the underlying sleep stages represent the hidden states to be inferred. Similarly, in HPC applications like genome annotation, a DNA sequence is modeled as a series of nucleotides (observations); the objective is to classify each nucleotide as belonging to either a coding or non-coding region (hidden states). Crucially, these systems often exhibit temporal persistence, remaining in a specific state for numerous consecutive time steps before transitioning, a characteristic that motivates the use of state-duration modeling.

Standard HMMs, however, are fundamentally limited in capturing these temporal dynamics, as state occupancy is inherently restricted to a geometric, memoryless distribution. HSMMs generalize this framework by permitting each hidden state to persist for a variable interval, explicitly defined by a duration distribution [4]. Consequently, the model incorporates three core probabilistic elements: state transitions, which govern the likelihood of moving between states; state-specific durations, which model the time spent within a state; and observation emissions, which define the probability of an observation given the current state.

HSMMs are particularly indispensable in computational biology for tasks such as genome annotation and segmentation [1], [5]. Once a genome has been sequenced, a central challenge lies in identifying which nucleotide sequences correspond to protein-coding exons, non-coding introns, regulatory elements, or intergenic regions. Interpreting the nucleotide sequence as an observation sequence, genome annotation amounts to inferring the hidden functional category of each region. The key difficulty is that genomic features can span from a few dozen to many thousands of nucleotides; modeling such extended segments requires explicit duration distributions that bypass the constant transition probability inherent to HMMs, making HSMMs a natural fit. This utility extends to chromatin state annotation, CpG island detection, and other problems where segment lengths carry vital biological meaning [6], [7].

Given a sequence of observations, two central tasks arise: learning, which estimates the model parameters from observed data, and decoding, which identifies the most likely hidden state sequence. Three fundamental algorithms solve these tasks: the Forward–Backward algorithm [4], [8] computes state probabilities at each time step, the Baum–Welch algorithm [4], [8], [9], an instance of Expectation–Maximization [10], estimates model parameters, and the Viterbi algorithm performs decoding. The Viterbi algorithm directly solves the genome annotation problem; moreover, by iteratively applying Viterbi decoding and re-estimating parameters from the decoded sequences, one can implement a Viterbi training loop, a technique widely used for ab initio gene prediction [11] when no prior data is available. Since the Viterbi algorithm serves both decoding and ab initio learning, it is the most widely adopted of the three, making it a primary candidate for acceleration.

However, while the Viterbi algorithm for a standard HMM has a complexity of $O ( T N ^ { 2 } )$ , the HSMM formulation introduces an additional loop over all possible durations, raising the complexity to $O ( T N ^ { 2 } D )$ , where D is the maximum admissible duration. In practical applications where D ranges from hundreds to thousands [5], this extra dimension poses a major computational bottleneck. Despite the critical importance of these problems, the algorithmic formulations used for HSMM inference have remained largely unchanged since their original proposal.

Historically, the classical Viterbi algorithm for HSMMs has been implemented via four nested loops [4], [8], [12]. Despite the proliferation of HPC resources, existing implementations remain confined to scalar, single-threaded CPU execution [13]–[16], in stark contrast to standard HMMs, which benefit from an extensive ecosystem of GPU-accelerated and SIMD-optimized frameworks [17]–[21].

Consequently, researchers face a restrictive trade-off: either endure prohibitively long execution times or artificially truncate their datasets, preventing the full expressive potential of HSMMs from being realized in large-scale scientific workflows.

Given the importance of HSMMs and the widening gap between the computational demands of real-world applications and the capabilities of existing sequential implementations, this work proposes the following contributions:

1) We introduce a novel Tensor-Based formulation of the Viterbi algorithm for Hidden Semi-Markov Models. By restructuring the traditional three-nested inner loops into dense tensor operations, our approach naturally maps onto the SIMD and SPMD execution models of modern HPC architectures. This reformulation exposes significant optimization opportunities that were previously inaccessible in sequential scalar implementations.

2) We leverage this formulation to deliver optimized implementations across CPUs utilizing SIMD vectorization and threading, and, for the first time, GPUs. To facilitate broader adoption, all implementations are released as a high-performance open-source library designed for seamless integration into existing scientific workflows.

3) We conduct an extensive performance evaluation across three CPU and five GPU architectures, demonstrating speedups of up to 570× over state-of-the-art HSMM frameworks.

## II. SEQUENTIAL HSMM VITERBI ALGORITHM

We now formalize the elements that define a Hidden Semi-Markov Model. An HSMM is represented by the parameter tuple [4], [8], [12]:

$$
\lambda = \left( S , \mathcal { O } , { \bf A } , { \bf P } , { \bf B } , \pi \right) ,\tag{1}
$$

where $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { N } \}$ is a finite set of N hidden states, $\mathcal { O } = \{ o _ { 1 } , . . . , o _ { M } \}$ is a finite set of M observation symbols, and $\pi ^ { N } [ j ] = \pi _ { j }$ is the initial probability of state $s _ { j }$ . The collection of all transition probabilities forms the transition probability matrix ${ \bf A } ^ { N \times N }$ , where each entry $\mathbf { A } [ i , j ] = a _ { i j }$ represents the probability of transitioning from state s<sub>i</sub> to state $s _ { j }$ . Similarly, the collection of all duration probabilities forms the duration probability matrix $\mathbf { P } ^ { N \times \hat { D } }$ , where each entry $\mathbf { P } [ j , d ] \ = \ p _ { j } ( d )$ represents the probability that state $s _ { j }$ persists for exactly d consecutive time-steps, with $d \in \{ 1 , \ldots , D \}$ . Finally, the emission probabilities define the emission probability matrix $\mathbf { B } ^ { N \times M }$ , where B[j, $o ] = b _ { j } ( o )$ is the probability that state $s _ { j }$ emits observation o. These matrix definitions of the model parameters will be central to the tensor formulation presented in Sec. III. Table I summarizes all components of the model.

TABLE I: Components of an HSMM and Viterbi Algorithm
<table><tr><td>Parameter</td><td>Matrix</td><td>Description</td></tr><tr><td> $\mathcal { S } = \{ s _ { 1 } , . . . , s _ { N } \}$ </td><td> $\mathcal { S } ^ { N }$ </td><td>Set of N hidden states.</td></tr><tr><td> $\mathcal { O } = \{ o _ { 1 } , . . . , o _ { M } \}$ </td><td> $\mathcal { O } ^ { M }$ </td><td>Set of M observation symbols.</td></tr><tr><td>πj</td><td> $\pi ^ { N }$ </td><td>Initial state probability for  $s _ { j } .$ </td></tr><tr><td> $\boldsymbol { a } _ { i j }$ </td><td> ${ \bf A } ^ { N \times N }$ </td><td>Transition prob. from si to  $s _ { j } ;$  self-transitions governed by zero-duration.</td></tr><tr><td> $p _ { j } ( d )$ </td><td> $\mathbf { P } ^ { N \times D }$ </td><td>Prob. that  $s _ { j }$  persists for exactly d steps, d ∈  $[ 1 , D ] .$ </td></tr><tr><td> $b _ { j } \left( o \right)$ </td><td> $\mathbf { B } ^ { N \times M }$ </td><td>Emission prob. of observing o in state  $s _ { j } .$ </td></tr><tr><td> $\delta _ { t } ( j )$ </td><td> $\pmb { \triangle } ^ { N \times T }$ </td><td>Likelihood of the most probable sequence ending in  $s _ { j }$  at time t.</td></tr><tr><td>ψt (j)</td><td> $\Psi ^ { N \times T }$ </td><td>Coordinates  $( s _ { i } , d )$  of the best predecessor for  $s _ { j }$  at time t.</td></tr></table>

The Viterbi algorithm is one of the fundamental algorithms for HSMMs. It addresses the following problem: given a model $\lambda$ and a sequence of observations $O _ { 0 } , \ldots , o _ { T - 1 } ,$ the goal is to find the most likely hidden state sequence $\mathbf { s } ^ { * } =$ $( q _ { 0 } ^ { * } , \dots , q _ { T - 1 } ^ { * } )$ . To achieve this, for every time step t and each state $s _ { j } ~ \in ~ S$ , the algorithm computes $\delta _ { t } ( j )$ , which represents the likelihood of the most probable state sequence ending in state $s _ { j }$ at time t. Both the sequential and the tensor formulations (Sec. III) maintain these values in a $\Delta ^ { N \times T }$ matrix:

$$
\begin{array} { r } { \pmb { \Delta } ^ { N \times T } = \left[ \begin{array} { c c c c } { \delta _ { 1 } ( 0 ) } & { \delta _ { 1 } ( 1 ) } & { \cdots } & { \delta _ { 1 } ( T - 1 ) } \\ { \delta _ { 2 } ( 0 ) } & { \delta _ { 2 } ( 1 ) } & { \cdots } & { \delta _ { 2 } ( T - 1 ) } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \delta _ { N } ( 0 ) } & { \delta _ { N } ( 1 ) } & { \cdots } & { \delta _ { N } ( T - 1 ) } \end{array} \right] . } \end{array}
$$

Following a dynamic programming approach, the algorithm proceeds in three distinct stages: initialization, induction, and backtracking.

## A. Initialization Phase

The initialization phase covers the first D time steps, where D is the maximum admissible state duration. During this interval, we must account for the possibility that the system has occupied state $s _ { j }$ since $t \ : = \ : 1$ with no prior transition. For each state $s _ { j }$ and each $1 \ \leq \ t \ \leq \ D$ , the initialization value combines (a) the initial state probability $\pi _ { j }$ , (b) the probability that state $s _ { j }$ persists for exactly t time steps, and (c) the joint emission probability of observations $o _ { 0 } , \ldots , o _ { t - 1 }$ under state $s _ { j } .$ For $t > D$ , no state can have persisted since the beginning, so this contribution is no longer considered.

$$
\delta _ { t } ( j ) = \underbrace { \pi _ { j } } _ { ( \mathrm { a } ) } \cdot \underbrace { p _ { j } ( t ) } _ { ( \mathrm { b } ) } \cdot \underbrace { \prod _ { \tau = 0 } ^ { t - 1 } b _ { j } ( o _ { t - \tau } ) } _ { ( \mathrm { c } ) } , \quad 1 \le t \le D\tag{2}
$$

## B. Induction phase

After initializing the first D time steps, we proceed with the most computationally intensive phase: the induction. For each time step t and each current state $s _ { j } ,$ , the goal is to find the previous state $s _ { i }$ and the duration d that together maximize the likelihood of reaching $s _ { j }$ at time t. The formulation is given by Equation (3), and the corresponding four-nested-loop pseudocode is shown in Algorithm 1.

The computation combines four factors. Term (a) is the previously computed value $\delta _ { t - d } ( i ) \colon$ it encodes the likelihood of the best path ending in state $s _ { i }$ at time $t - d ,$ under the assumption that a transition to $s _ { j }$ occurred there. Term (b) is the transition probability $a _ { i j }$ from state $s _ { i }$ to state $s _ { j }$

Together, (a) and (b) form the inner maximization: for a fixed duration $d ,$ we evaluate all possible source states $s _ { i }$ and select the one that yields the highest likelihood.

The result is then multiplied by term (c) , the probability $p _ { j } ( d )$ that state $s _ { j }$ persists for exactly d consecutive time steps, and by term (d) , the cumulative emission probability of all observations from $t - d + 1$ to t under state $s _ { j }$ . The maximization repeats this for all durations $d \in \{ 1 , \ldots , \operatorname* { m i n } ( t , D ) \}$ and all source states $s _ { i }$ and selects the best combination. The resulting optimal combination $( d ^ { * } , i ^ { * } )$ for each state $s _ { j }$ at time step t is stored in $\psi _ { t } ( j )$ , while the corresponding likelihood is stored in $\delta _ { t } ( j )$

$$
\delta _ { t } ( j ) = \operatorname* { m a x } _ { \forall i , \forall d } \biggl [ \underbrace { \delta _ { t - d } ( i ) } _ { ( a ) } \cdot \underbrace { a _ { i j } } _ { ( b ) } \cdot \underbrace { p _ { j } ( d ) } _ { ( c ) } \cdot \underbrace { \prod _ { k = 0 } ^ { d - 1 } b _ { j } ( o _ { t - k } ) } _ { ( d ) } \biggr ] ,\tag{3}
$$

Algorithm 1 Sequential HSMM Viterbi Induction.   
1: for $t = 2$ to T do   
2: for $j = 1$ to N do   
3: $\delta _ { t } ( j )  - \infty$   
4: for $\dot { d } = 1$ to min(t, D) do   
5: for $i = 1$ to N do   
6: val ← $\textrm { - } \delta _ { t - d } ( i )$ · a<sub>ij</sub> · p<sub>j</sub> (d) · $\begin{array} { r } { \prod _ { k = 0 } ^ { d - 1 } b _ { j } ( o _ { t - k } ) } \end{array}$   
7: if val $> \delta _ { t } ( j )$ then   
8: $\delta _ { t } ( j )  \mathrm { v a l }$   
9: $\psi _ { t } ( j )  ( d , i )$

Note that induction starts from $t \ = \ 2 ;$ for $t \ \leq \ D ,$ , the value $\delta _ { t } ( j )$ is determined by the maximum of two cases: the system has remained in state $j$ since $t = 1$ (initialization), or it transitioned from a previous state i at some time $\tau < t$ (induction). The algorithm takes the greater of these two probabilities.

## C. Backtracking Phase

Once $\delta _ { t } ( j )$ has been computed $\forall t , \forall j$ , we can recover the optimal state sequence. Starting from the last time step, we select the state with the highest delta value. The backtracking then proceeds backwards, until $t = 0 ,$ , returning the most likely chain of states. Since we operate in a Semi-Markov regime, the recovered path will typically exhibit states persisting across multiple consecutive time steps, reflecting the explicit duration modeling that distinguishes the HSMM from an HMM.

## III. TENSOR-BASED VITERBI ALGORITHM

The tensor formulation of the Viterbi algorithm follows the same subdivision as the standard one: an initialization phase, an induction phase, and a backtracking phase. However in this work, we express both initialization and induction using a new tensor formulation. The backtracking phase remains sequential as it does not represent a computational bottleneck. Instead, optimization efforts focus on the induction phase, where tensor reformulation yields the greatest benefit.

## From Loops to Tensors

As described in the sequential Algorithm 1, for every time step t we need to evaluate all possible combinations of a previous state $s _ { i }$ and a duration d for each current state $s _ { j }$ Formally, for every state $s _ { j } \in S$ we must consider every pair $( s _ { i } , d )$ with $s _ { i } \in \mathcal S \setminus \{ s _ { j } \}$ and $d \in \{ 1 , \dots , D \}$ . Once all possible combinations of $( s _ { i } , d )$ have been computed for each state $s _ { j } ~ \in ~ S$ , our aim is to compute $\delta _ { t } ( j )$ for every time step $t \leq T$

To compute these combinations and then $\delta _ { t } ( j ) , \forall j$ sequentially, one must iterate over the current states, then over the possible durations, and then again over the previous states, yielding three nested loops already contained inside the main time-step loop. The sequential algorithm therefore requires four nested loops.

## A. The Brick Representation

![](images/213c2c87687c8184d36fb19e8b1ca055ab110768c654cf96c1fefe8cdf1999ea.jpg)  
Fig. 1: Visualization of the Brick 3D tensor and subdivision into sj-slices.

Our key idea is to replace the three nested loops with structured tensor operations that make data reuse explicit and expose independent computations along each axis. The loops over $( s _ { j } , s _ { i } , d )$ can be naturally mapped onto a threedimensional tensor, as shown in Figure 1a. We choose the following layout:

• y-axis −→ target states $s _ { j } .$ , with $j \in \{ 1 , \ldots , N \}$ ;

• x-axis −→ source states $s _ { i } ,$ with $i \in \{ 1 , \ldots , N \}$ ;

• z-axis −→ durations $d ,$ with $d \in \{ 1 , \ldots , D \}$

With this convention the set of all combinations $( s _ { j } , s _ { i } , d )$ can be visualized as a 3-dimensional tensor of size $N \times N \times D$ that we will call Brick (B), shown in Figure 1a. Each slice of the Brick along the y-axis corresponds to the complete set of $( s _ { i } , d )$ combinations for a single target state $s _ { j } ,$ which we call $s _ { j }$ -slices (Figure 1b). Using this representation, we reformulate the sequential induction of Algorithm 1 into three key stages: (i) the Brick Construction, which occurs once (Sec. III-B); (ii) the Brick Update, where the Brick is modified at each time step t to incorporate temporal factors (as described in Sec. III-C); and (iii) the Maximum Extraction, which identifies the maximum value and the corresponding coordinates $( s _ { i } , d )$ within each $s _ { j } { \mathrm { - s l i c e } }$ . While phases (ii) and (iii) are executed iteratively at each time step, phase (i) is a pre-computation step performed outside the time steps loop. This pipeline, including initialization, takes shape within the Algorithm 2 that will be described line-by-line below.

## B. Brick Construction

The key observation is the following: the transition probability matrix ${ \bf A } ^ { N \times N }$ and the duration probability matrix $ { \mathbf { P } } ^ { N \times D }$ do not depend on the time step t. Their contribution to the Brick can therefore be precomputed once, before entering the time-step loop. This corresponds to line 3 of Algorithm 2. As shown in Figure 2a, the transition matrix A is a twodimensional $N \times N$ matrix. Within our Brick, it occupies the front face, i.e. it is aligned with the $y -$ and x-axes. The duration probability matrix P is a two-dimensional $N \times D$ matrix positioned on the side face, i.e. aligned with the y- and z-axes (Figure 2a).

## Algor Algorithm 2 Tensor-based HSMM Viterbi. MIMI Viterbi.

INITIALIZATION $( 1 \leq t \leq D )$   
1: ${ \bf E } ^ { N \times D }$ ← Emission Product Computation   
2: $\Delta ^ { N \times D }  \pi ^ { N \times \uparrow } \odot \mathbf { P } ^ { N \times D } \odot \mathbf { E } ^ { N \times D }$   
INDUCTION $( 2 \leq t \leq T )$   
3: $\mathcal { B } _ { f i r s t } ^ { N \times N \times D }  \mathrm { \bf ~ A } ^ { N \times N \times \uparrow } \odot \mathrm { \bf ~ P } ^ { N \times \uparrow \times D }$ // Brick Construction   
4: for t = 2 to T do   
5: ${ \bf E } ^ { N \times D }$ ← Emission Product Computation   
6: $\pmb { \Delta } _ { p a s t } ^ { N \times D } \  \pmb { \Delta } ( t - D : t - 1 ) ^ { N \times D }$ // Past Delta Extraction   
7: $\mathcal { B } _ { ( t ) } ^ { N \times N \times D }  \mathcal { B } _ { f i r s t } \odot \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } _ { p a s t } ^ { \uparrow \times N \times D } \odot \mathbf { \Delta } \mathbf { E } ^ { N \times \uparrow \times D }$ // Brick Update   
8: for j = 1 to N do   
9: Ψ<sub>j</sub> (t) ← arg max<sub>d, i</sub> B<sub>(t)</sub>[s<sub>j</sub> -slice]<sup>N×D</sup> // Max. Extraction   
BACKTRACKING   
10: $\mathbf { q } ^ { * } = B a c k t r a c k i n g ( \Psi ^ { N \times T } , \ : \Delta ^ { N \times T } )$

![](images/fe47c2f4e0771f005d6da2f5de3921dec9c9069c314544477d4b7f3072029db3.jpg)  
Fig. 2: Alignment of A and P within the Brick 3D tensor and the corresponding representation of the broadcasted product.

To combine these two matrices into a single threedimensional tensor, we perform the following product:

$$
\mathcal { B } _ { f i r s t } ^ { N \times N \times D } = \underbrace { \tilde { \mathbf { A } } ^ { N \times N \times \uparrow } } _ { ( a ) } \odot \underbrace { \tilde { \mathbf { P } } ^ { N \times \uparrow \times D } } _ { ( b ) }\tag{4}
$$

where the ↑ symbol denotes the axis along which broadcasting occurs, expanding the tensor to match the dimensions of the corresponding operand. Concretely, as shown in Figure 2b-c, the operation can be understood in two steps:

(a) Broadcast A: replicate the $N \times N$ matrix D times along the z-axis, obtaining a tensor $\widetilde { \mathbf { A } } ^ { N \times N \times D }$ (Figure 2b-(a)).

(b) Broadcast P: replicate the $N \times D$ matrix N times along the x-axis, obtaining a tensor $\widetilde { \mathbf { P } } ^ { N \times N \times D }$ (Figure 2b-(b)). Then, the Brick is given by the element-wise product of the two broadcast matrices: $\mathcal { B } _ { f i r s t } ^ { \check { N } \times N \times D } = \widetilde { \mathbf { A } } ^ { N \times N \times \check { D } } \odot \widetilde { \mathbf { P } } ^ { N \times N \times D }$ (Figure 2c). We refer to this as a broadcasted product: a fundamental operation of our tensor-based Viterbi algorithm.

## C. Brick Update

After constructing the static Brick $B _ { f i r s t }$ , we enter the main loop over time-steps. At each step t, two additional matrices must be incorporated: the past delta values $\Delta _ { p a s t }$ and the emission probability product E, both of which depend strictly on t. This phase corresponds to lines 5–7 of Algorithm 2.

1) Extracting Past Delta Matrix: The past delta values are the simpler of the two time-dependent matrices. We collect from the stored $\pmb { \Delta }$ matrix a window of size $N \times D$ obtaining $\Delta _ { p a s t } ^ { N \times D }$

$$
\begin{array} { r } { \pmb { \Delta } _ { p a s t } ^ { N \times D } = \left[ \begin{array} { c c c } { \delta _ { 1 } ( t - D ) } & { \cdots } & { \delta _ { 1 } ( t - 1 ) } \\ { \delta _ { 2 } ( t - D ) } & { \cdots } & { \delta _ { 2 } ( t - 1 ) } \\ { \vdots } & { \vdots } & { \vdots } \\ { \delta _ { N } ( t - D ) } & { \cdots } & { \delta _ { N } ( t - 1 ) } \end{array} \right] . } \end{array}
$$

As shown in Figure 3, this window starts at $t - D$ (or 1 if $t < D )$ and ends at $t - 1$ , reversed so that the first column corresponds to the most recent past step (duration $d = 1 )$ . The resulting matrix $\Delta _ { p a s t } ^ { N \times D }$ is aligned on the x- and z-axes of the Brick (lower face).

![](images/101bcdaf849e2d50333fb3f90fb34d70b72f812f71680737909f13d6b60cad20.jpg)  
Fig. 3: Extracting from $\pmb { \Delta }$ past $[ t - D , t )$ values to obtain $\Delta _ { p a s t } .$

2) Computing Emission Probability Matrix: The emission probability matrix is the more complex of the two factors. Recall from the term (d) of Equation 3 that, for a given state $s _ { j }$ and duration $d ,$ we need the product of the emission probabilities over the d most recent observations. In the sequential algorithm this partial product is trivially computed inside the duration loop. In the tensor formulation, however, the duration loop has been eliminated. We therefore need to compute the entire $N \times D$ matrix of cumulative emission at each time step.

$$
\begin{array} { r } { \mathbf { E } ^ { N \times D } = \left[ \begin{array} { c c c c c } { b _ { 1 } ( o _ { t } ) } & { \prod _ { k = 0 } ^ { 1 } b _ { 1 } ( o _ { t - k } ) } & { \dots } & { \prod _ { k = 0 } ^ { d - 1 } b _ { 1 } ( o _ { t - k } ) } \\ { b _ { 2 } ( o _ { t } ) } & { \prod _ { k = 0 } ^ { 1 } b _ { 2 } ( o _ { t - k } ) } & { \dots } & { \prod _ { k = 0 } ^ { d - 1 } b _ { 2 } ( o _ { t - k } ) } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { b _ { N } ( o _ { t } ) } & { \prod _ { k = 0 } ^ { 1 } b _ { N } ( o _ { t - k } ) } & { \dots } & { \prod _ { k = 0 } ^ { d - 1 } b _ { N } ( o _ { t - k } ) } \end{array} \right] . } \end{array}
$$

As shown in Figure 4, we extract the D most recent observation indices, look up the corresponding emission prob for all states inside $\mathbf { B } ^ { N \times M }$ , reverse the order (so that the first column corresponds to duration $d = 1 )$ . Then, we apply a cumulative product along the duration axis obtaining $\dot { \mathbf { E } } ^ { \dot { N } \times D }$ (Figure 4). Our tensor formulation highlights significant redundancies, leading us to propose an optimized caching strategy as detailed in Sec. IV.

![](images/1a48ace0ac6b3fd6ef334fa2ac3440ecabb16cce1954f86363c7b9c0e9d2f01d.jpg)  
Fig. 4: Computing Emission Product to obtain the Emission Probability Matrix.

Once the past delta matrix $\Delta _ { p a s t }$ and the emission probability matrix E have been computed for a given time step t, we combine them with the precomputed Brick B through two successive broadcasted products:

$$
\begin{array} { r l } { \mathcal { B } _ { ( t ) } ^ { N \times N \times D } } & { = \mathcal { B } _ { f i r s t } ^ { N \times N \times D } \odot \underbrace { \widetilde { \Delta } _ { p a s t ( t ) } ^ { \uparrow \times N \times D } } _ { ( c ) } \odot \underbrace { \widetilde { \mathbf { E } } _ { ( t ) } ^ { N \times \uparrow \times D } } _ { ( d ) } } \end{array}\tag{5}
$$

(c) $\Delta _ { p a s t }$ lies on the x–z plane, and broadcast the $N \times D$ matrix N times along the y-axis, obtaining a tensor $\widetilde { \pmb { \Delta } } _ { p a s t } ^ { N \times N \times D }$

(d) E lies on the $y { \mathrm { - } } z { \mathrm { ~ a x i s , } }$ and broadcast the $N \times D$ matrix N times along the x-axis, obtaining a tensor $\widetilde { \mathbf { E } } ^ { N \times N \times D }$

The resulting tensor $B ^ { N \times N \times D }$ is the fully populated Brick for time step t: each entry $B [ j , i ,$ , d] encodes the likelihood of transitioning from state $s _ { i }$ to state $s _ { j }$ with duration d, given the observations up to time t. Having obtained all combinations of $( s _ { i } , d )$ for each $s _ { j } ,$ we must now identify the one that yields the maximum likelihood for each $s _ { j }$

## D. Maximum Extraction

We now need to extract, for each target state $s _ { j } ,$ the combination $( s _ { i } ^ { * } , d ^ { * } )$ that maximizes the tensor entry. Recall that $s _ { j }$ is indexed along the y-axis; the corresponding slice is therefore a two-dimensional $N \times D$ matrix. For each $s _ { j }$ -slice we seek the maximum value and its associated coordinates $( s _ { i } , d )$ as shown in Figure 5. This yields the resulting maximum values vector $\Delta ( t ) ^ { N }$ . This phase corresponds to lines 8–9 of Algorithm 2.

![](images/517f85380d6bcfd5c70757b80b687b5816d22f0dd104b1f38565f3f0ad8f3c19.jpg)  
Fig. 5: Argmax computation in each $s _ { j } .$ -slice.

Seeking the maximum value is a fundamental problem in parallel computing and several strategies can be employed to improve its performance; we discuss our strategy in Sec. IV.

## E. Initialization Phase

As described in Sec. II-A, the initialization phase covers the first $D$ time steps $( 1 \leq t \leq D )$ , accounting for the possibility that state $s _ { j }$ has persisted since $t = 0$ with no prior transition. In the tensor formulation, the entire initialization is expressed as a single broadcasted product of three matrices (lines 1–2 of Algorithm 2):

$$
\Delta [ \forall j , 1 : D ] ^ { N \times D } = \pi ^ { N \times \uparrow } \odot \mathbf { P } ^ { N \times D } \odot \mathbf { E } ^ { N \times D }\tag{6}
$$

Here, (a) $\pi ^ { N }$ is the initial state probability vector, broadcasted along the duration axis; (b) $\mathbf { P } ^ { N \times D }$ is the duration probability matrix; and (d) ${ \bf E } ^ { N \times D }$ is the cumulative emission matrix, where each entry $\mathbf { E } [ j , d ]$ accumulates the emission probabilities of the first d observations under state $s _ { j } .$ Since no entry depends on any other, all $N \times D$ values are computed with no loop-carried dependencies. During these first D steps, both initialization and induction contribute; the final $\delta _ { t } ( j )$ is taken as the maximum of the two (lines 8–9 of Algorithm 2).

## IV. IMPLEMENTATIONS AND OPTIMIZATIONS

This section details the optimization strategies for the Tensor-Based Viterbi algorithm from Sec. III. The tensor formulation naturally enables several optimizations: it exposes data reuse patterns for cache-friendly access and structures computation along independent axes, mapping efficiently onto SIMD and multi-threaded execution models.

We developed four versions of the Tensor Viterbi algorithm: TENS-PY, TENS-1C, TENS-MC and TENS-GPU. TENS-PY is a direct transcription of Algorithm 2 in NumPy, and it was the first implementation developed. No low-level optimization is attempted; the implementation serves as a readable reference and validation baseline. It was necessary to analyze the algorithm, identify optimization opportunities, and test them before delving into the low-level optimized implementations.

We highlight that all operations are implemented in logspace. Since the quantities involved lie in the interval [0, 1], repeated multiplications would quickly lead to numerical underflow. Working in log-space is a standard practice adopted by all major Markov model frameworks [8], [13]. The practical consequence is straightforward: products become sums, cumulative products become cumulative sums, and broadcasted products become broadcasted sums.

TABLE II: Summary of Implementations
<table><tr><td>Naming</td><td>Architecture</td><td>Language and Tools</td></tr><tr><td>TENS-PY</td><td>CPU Single-Core</td><td>Python, Numpy</td></tr><tr><td>TENS-1C</td><td>CPU Single-Core</td><td>C++</td></tr><tr><td>TENS-MC</td><td>CPU Multi-Core</td><td>C++, OpenMP</td></tr><tr><td>TENS-GPU</td><td>GPU</td><td>CUDA, HIP</td></tr></table>

## A. NumPy Version

TENS-PY follows the tensor algorithm precisely and is implemented using NumPy, which allows the algorithm to be expressed directly in terms of tensor operations without requiring further low-level optimizations. The memory layout is the one used by NumPy (row-major, depth-first). The broadcasted sums are expressed in two phases: the 2D matrix is broadcast over the absent axis to obtain a 3D tensor, then the two 3D tensors are summed element-wise. This pattern is applied to both the Brick Computation and Brick Update phases. The Past Delta Extraction phase is performed using NumPy slicing operations, and the argmax is computed using NumPy’s argmax function independently for each destination state $s _ { j }$

Once this first version was coded, analysis of the prototype revealed an optimization opportunity: the emission accumulation, originally recomputed inside the duration loop at every time-step, can be decoupled from the main recurrence and maintained through a rolling cache. After the warmup phase $( t ~ > ~ D )$ , the full emission buffer is obtained by a single element-wise addition and a cache shift, reducing the per-time-step cost from $O ( D N )$ to amortized $O ( N )$ and eliminating the loop-carried dependency on the duration axis. This optimization has been critical in the C++ versions.

## B. CPU Implementation

TENS-1C and TENS-MC are developed in C++ without relying on any tensor library, since neither BLAS [22] nor frameworks such as xTensor [23] provide broadcasted sums natively. Implementing the operations explicitly also enabled us to fuse the Brick Update and Argmax phases, avoiding the need to store the fully populated Brick in memory before computing the maximum.

These versions allow us to exploit the optimization opportunities unveiled by the tensor formulation and produce tuned variants of the proposed algorithm.

1) Memory Access Patterns: We introduced two complementary flat layouts designed for spatial locality. The $\Delta _ { p a s t }$ buffer uses a time-major layout $( t \cdot N + j )$ so that all states at a given time-step are contiguous. The ∆ and Ψ arrays use a state-major layout $( j \cdot T + t )$ as required by the Backtracking Phase. The Brick tensor uses a $( j , d , i )$ layout where, for a fixed state $j ,$ the entire $D \times N$ block is contiguous, fitting in L2 across the duration loop if it is small enough. This is a deliberate choice: the hot inner loop sweeps over states $s _ { i }$ within a fixed $( s _ { j } , d )$ slice, achieving contiguous access.

2) Cached Emissions Computing: We build a 2D emission buffer E indexed by $( d \cdot N + j )$ , storing the cumulative sum of emission log-probabilities over each observation window. For $t \leq D _ { \mathrm { i } }$ , the cumulative sum is computed from scratch. For $t > D ,$ the buffer is updated incrementally using an emission cache $\mathbf { E } _ { c a c h e } \colon$ the new observation log-probability at time t is added to a right-shifted copy of ${ \bf E } _ { c a c h e } ,$ and the result is saved back to ${ \bf E } _ { c a c h e }$ for the next time-step. This reduces the perstate emission update from $O ( D )$ to amortized $O ( 1 )$ after the warm-up phase, and makes the emission values for all $( d , j )$ pairs available independently before the Argmax loop begins.

3) Fused Brick Update and Maximum Extraction: The two most intensive phases of the algorithm can be fused in this implementation, performing a fused Brick Update + Argmax with no self-transition exclusion (the transition matrix encodes this structurally with zero-values). The inner loop is entirely branchless, using annotated ternary conditional assignments. This branchless pattern allows the compiler to generate predicated instructions rather than conditional branches, eliminating branch misprediction penalties. In the three-level loop $( s _ { j } , d , s _ { i } )$ , iterations are fully independent across all three axes, making each axis independently parallelizable or vectorizable.

4) Multi-Core Implementation: The TENS-MC implementation parallelizes the single-core version using OpenMP. A single #pragma omp parallel region spawns a persistent thread team, using implicit barriers between t steps to avoid repeated fork/join overhead.

Pre-computation phases exploit full independence across their iteration spaces. The Brick Construction, for instance, distributes N · D · N independent element-wise sums across all three axes via collapse(3). Similarly, the Cached Emission Computing exposes D · N independent work units via collapse(2). The key parallelization challenge lies in the fused Brick Update and Maximum Extraction, which in the single-core version offers only N independent tasks. We decompose it into two phases: Phase A distributes work over $( s _ { j } , d )$ pairs, where each thread sweeps over all source states $s _ { i }$ to find the local maximum, exposing $N \cdot D$ independent tasks. Phase B then reduces over d per destination state $s _ { j }$ . This decomposition enables full thread utilization even when N alone is smaller than the available core count.

## C. GPU Implementation

The tensor-based Viterbi algorithm is inherently suited for GPU architectures; therefore, building upon our initial CPU version, we developed what is the first GPU-accelerated Viterbi implementation for Hidden Semi-Markov Models to our knowledge. The TENS-GPU implementation is developed in CUDA and ported to HIP via hipify [24], maintaining a single codebase for both NVIDIA and AMD architectures.

The primary computational bottleneck lies in constructing $( B _ { f i r s t } )$ and updating (B) the Brick tensor. To maximize hardware utilization, we employ a fine-grained work distribution where each individual thread is responsible for computing a single Brick element. Then, after obtaining the final Brick version within each time-step iteration, the threads operating on elements of the same $s _ { j }$ -slice must cooperate to perform the maximum extraction.

To implement this mapping, we decompose the $N \times N \times D$ tensor into $N \times N$ vectors of size D aligned along the z-axis, as illustrated in Figure 6. This data layout is mapped onto a 2D grid of $N \times N$ thread blocks, where each block manages a single vector. The $D$ elements are evenly divided between the threads in the block. This configuration ensures that each thread block handles an entire temporal slice of the tensor regardless of the duration D.

![](images/e5e743f2dbf4de8bd20bee13ec613da9d4c0dc28353771d14979d755735bf950.jpg)  
Fig. 6: Mapping the Brick to GPU Thread Blocks.

1) Memory Access Patterns and Coalescing: The emission cache (ordered $j \cdot D + d )$ , the brick $B _ { f i r s t }$ (ordered $j \cdot N \cdot D + i \cdot D + d )$ , and $\pmb { \Delta }$ (indexed $i \cdot T + ( t - 1 - d ) )$ all maintain d-contiguity for consecutive and coalesced thread access. Meanwhile, the emission probability E for the current observation $o _ { t }$ and state $s _ { j }$ is shared by all threads within a block, allowing the GPU to serve it via a single broadcast read.

2) Cached Emissions Computing: Naively computing E at each time-step requires $O ( D )$ work per thread. To eliminate redundant operations, we employ the same cached strategy explored in Sec. IV-B2. This reduces complexity to $O ( 1 )$ via a double-buffering scheme. Thread $( j , i , d )$ reads $\mathbf { E } [ : , d - 1 ]$ from ${ \bf E } _ { c a c h e }$ , adds $b _ { j } ( o _ { t } )$ , and writes the result to $\mathbf { E } _ { f i n a l }$ . The value is kept in a register for the current iteration, and the two buffers alternate roles across time-steps.

3) Separated Brick Update and Maximum Extraction: As shown in Sec. III, the tensor formulation decomposes each time-step into two phases: Brick Update and Maximum Extraction. These phases are fused in the CPU implementation but split across two GPU kernels to better exploit its architecture. Note, the time-invariant component $B _ { f i r s t }$ is precomputed once before the induction loop.

a) First Kernel: each thread $( s _ { j } , s _ { i } , d )$ computes one element of the final Brick B and stores it directly in shared memory, bypassing global memory. An intra-block parallel reduction then identifies the local maximum score and its $( s _ { i } , d )$ coordinates in $O ( \log D )$ steps: cross-warp steps use \_\_syncthreads(), while the final $\log _ { 2 } ( \mathrm { w a r p s i z e } )$ steps switch to warp-shuffle instructions (\_\_shfl\_down\_sync), keeping the running maximum entirely in registers.

b) Second Kernel: aggregates the per-block local maximums to determine the global arg max over the full $( s _ { i } , d )$ plane for each $s _ { j }$

Splitting the computation into two kernels avoids grid-wide synchronization, which would otherwise require cooperativegroups designs with additional occupancy constraints.

## V. EXPERIMENTAL RESULTS

To assess the performance of our tensor-based implementations, we present an in-depth experimental analysis. We first describe the evaluation environment (Sec. V-A) and validate our implementations (Sec. V-B). Then, we evaluate the performance of both CPU (Sec. V-C) and GPU (Sec. V-D)

implementations, including an investigation into the impact of varying hardware architectures (Sec. V-E). Last, we analyze the energy consumption (Sec. V-F) and conduct a stress test using extreme-scale inputs (Sec. V-G).

## A. Evaluation Environment

1) Baseline: As a baseline for validation and performance comparison, we selected hsmmlearn [16], a C++ library (with a Python API) for HSMMs with explicit duration distributions originating from the R hsmm package [15], making it one of the most established HSMM codebases. We chose it for two reasons. First, it implements the general HSMM formulation with explicit, non-parametric duration modeling and categorical emissions, matching the problem addressed in this work. Second, among the HSMM frameworks analyzed in a recent survey [13], it is one of the few combining a general-purpose formulation with a C++ backend, ensuring our comparison targets optimized compiled code. We refer to this single-core baseline as BASE-1C. We also considered edhsmm [25], which is inspired by hsmmlearn, but its Viterbi algorithm is implemented in Cython, potentially limiting low-level compiler optimization compared to a native C++ implementation.

As hsmmlearn is strictly limited to a single-core implementation, we developed a multi-core variant, BASE-MC, to ensure a fair comparison. This was achieved by parallelizing the C++ Viterbi decoder with OpenMP, specifically targeting the loop over states, the only loop in the classical fournested-loops formulation with fully independent iterations. BASE-MC represents the maximum parallelism extractable from the sequential formulation without significant algorithmic restructuring, serving as our primary baseline for both multicore and GPU comparisons. It exhibits near-linear scaling with thread count, provided the number of threads does not exceed N. Beyond that point, the speedup saturates and degrades slightly due to synchronization overhead, confirming that the traditional formulation fundamentally limits the exploitable parallelism to a single axis. All implementations, including both the baselines and our tensor-based versions, utilize double-precision (FP64) floating-point arithmetic.

2) Problem Size: To evaluate our formulation under realistic conditions, we selected problem sizes guided by computational genomics. We tested configurations with a number of states (N) ranging from 10 to 75, spanning prokaryotic gene finders [26] at the lower end, chromatin state annotation [27] in the mid-range (15–25), and eukaryotic gene finders such as AUGUSTUS [28] at the upper end.

For the sequence length (T), we tested from $1 0 ^ { 3 }$ to $1 0 ^ { 7 }$ time steps, covering typical gene-finding invocations [5], [29] up to $T = 1 0 ^ { 6 }$ [11]. For the maximum duration (D), we adopted values from 100 to 10,000, ranging from the explicit intron duration cutoff of SNAP [5] to stress-test scenarios capturing the longest gene structure features in the human genome [30].

3) Architectures: We evaluated our implementations on a representative set of high-performance CPU and GPU architectures, summarized in Table III.

TABLE III: Hardware specifications. SM: Streaming Multiprocessor; CU: Compute Unit; GCD: Graphics Compute Die.
<table><tr><td></td><td>Processor</td><td>Units</td><td>Memory</td><td>Compiler</td></tr><tr><td></td><td>AMD EPYC 7A53</td><td>64 Cores</td><td>512 GiB DDR4</td><td>Cray clang v19.0.0</td></tr><tr><td>CPU</td><td>Grace</td><td>72 Cores</td><td>480 GB LPDDR5x</td><td>GCC 14.2</td></tr><tr><td></td><td>Xeon 8480+</td><td>2×56Cores</td><td>512 GiB DDR5</td><td>ICX v2024.1.0</td></tr><tr><td></td><td>A100 SXM</td><td>108 SMs</td><td>80 GB HBM2e</td><td>CUDA v12.2</td></tr><tr><td>GU</td><td>H100 SXM</td><td>132 SMs</td><td>80 GB HBM3</td><td>CUDA v11.8</td></tr><tr><td>NVDIIA</td><td>H200 SXM</td><td>132 SMs</td><td>141 GB HBM3e</td><td>CUDA v12.4</td></tr><tr><td>AMMD GPU</td><td>MI250X (1 GCD)</td><td>110 CUs</td><td>64 GB HBM2e</td><td>ROCm v6.3</td></tr><tr><td></td><td>MI300X</td><td>304 CUs</td><td>192 GB HBM3</td><td>ROCm v5.7</td></tr></table>

## B. Validation Results

All implementations produce identical output to hsmmlearn across every tested configuration, achieving 100% decoding accuracy. To ensure exact equivalence, we include the tail adjustment phase without further optimization, matching hsmmlearn’s boundary handling. Notably, TENS-PY, our direct NumPy transcription of the tensor formulation (Algorithm 2), already achieves a 4.5× speedup over BASE-1C. Despite comparing interpreted Python against compiled C++, this result demonstrates that the tensor reformulation alone yields substantial gains before any low-level optimization is applied.

## C. CPU Speedup Analysis

We begin by comparing TENS-1C against BASE-1C on Intel Xeon 8480+. Figure 7a reports the speedup for $T = 1 0 ^ { 5 }$ across $N ~ \in ~ \{ 1 0 , 1 5 , 2 5 , 5 0 , 7 5 \}$ and $D ~ \in ~ \{ 1 0 0 , 2 5 0 , 5 0 0 , 1 0 0 0 \}$ TENS-1C is consistently faster, with speedups ranging from 8.7× (N = 75, D = 1000) to 14.1× (N = 10, D = 100). The speedup decreases as either N or D grows, reflecting the point at which $D \times N$ exceeds L2 cache capacity.

![](images/6e64af4576344c84869b6df25ba8f675d7b70c20cee3ac93a1815faa06609489.jpg)

![](images/78770e823e12e5341b904303b9096d3f77d85178375bf12c28b7d0a5e91b76e8.jpg)  
(a) TENS-1C speedup over BASE-1C and TENS-1C runtime (in parentheses; s: seconds; m: minutes). Xeon 8480+, $T = 1 0 ^ { 5 }$  
(b) Baselines and Tensors profiling metrics on Xeon 8480+ CPU with ICX compiler.  
Fig. 7: Speedup and runtime of TENS-1C and profiling report.

We profiled the execution using LIKWID [31], which reveals that the tensor reformulation reduces retired instructions by 36× (from 610B to 16.9B). Figure 7b highlights the resulting hardware efficiency: the vectorization ratio increases from $< 0 . 0 0 1 \%$ in BASE-1C to 30.9% in TENS-1C, while branch misprediction overhead drops from 6.9% to 0.46%. The figure also shows a reduced unique DRAM footprint, falling from 2.63 GB to 1.42 GB. This, combined with hardware counter evidence of a shift from L2 reuse to streaming L3 access (not shown in the figure), explains both the massive absolute gains and their gradual erosion at larger N and D. Nevertheless, even at the largest configuration, the speedup remains significant: TENS-1C completes in 12.2 minutes, whereas BASE-1C requires over 1.76 hours.

We now compare TENS-MC against BASE-MC on Xeon 8480+. Figure 8 reports the speedup for $T \ = \ 1 0 ^ { 4 }$ and $T \ = \ 1 0 ^ { 5 }$ across the same value of D (duration) and N (number of states) considered before. The trend is clear: TENS-MC becomes increasingly effective as N and D grow. At $T \ = \ 1 0 ^ { 5 }$ the speedup over BASE-MC reaches 11.4× $( N = 5 0 , D = 1 0 0 0 )$ , while at $T = 1 0 ^ { 4 }$ a peak of 11.5× is observed at the same configuration. This behavior is the direct consequence of the parallelization strategy. BASE-MC can only distribute work over the N destination states $s _ { j } ,$ since the sequential formulation carries loop dependencies across durations and source states.

![](images/8263d926310dd83f7686df2d99ad49e4a0c3ef043badf31a487076eb8f9bdff5.jpg)  
Fig. 8: Speedup of TENS-MC over BASE-MC on Xeon 8480+. The cell shows the speedup and the runtime of TENS-MC.

In contrast, TENS-MC reformulates the computation as broadcasted sums with no loop-carried dependencies, exposing N · D independent tasks. This exposes sufficient parallelism to fully saturate the 112 available threads, even when N alone would leave most cores idle. LIKWID profiling confirms this efficiency: TENS-MC retires only 9.9B instructions compared to 21.9B for BASE-MC, while driving L3 bandwidth to nearpeak utilization of the memory hierarchy. Furthermore, Figure 7b shows that TENS-MC achieves an 80.4% vectorization ratio (versus 0.003% for BASE-MC), reduces bad speculation from 6.25% to 1.16%, and lowers DRAM volume from 4.59 GB to 1.43 GB.

At small problem sizes, the trend reverses: for $T = 1 0 ^ { 4 }$ with N = 10 and D = 100, the TENS-MC speedup falls to 0.1×. In this regime, pre-computation and barrier overhead dominate because the per-step workload is insufficient to amortize them. This deficit shrinks to $0 . 5 \times \mathrm { a t } T { = } 1 0 ^ { 5 }$ and vanishes, becoming a 14× improvement, at $T = 1 0 ^ { 6 }$ (as we will show in Fig. 10a), confirming that the overhead is successfully amortized over the longer sequences typical of genomic applications. In Sec. V-E, we will evaluate both TENS-MC and TENS-1C across additional CPU architectures, demonstrating that they consistently outperform their respective baselines.

Moreover, it is worth remarking that when compiled with GCC, the TENS-MC runtime for $D = 1 0 0 , N = 1 0$ , and $T ~ = ~ 1 0 ^ { 4 }$ drops to 0.34 seconds. This highlights specific inefficiencies in the Intel compiler’s OpenMP implementation for small input sizes. Nevertheless, we report all data using the Intel compiler as it consistently outperformed GCC across all other configurations.

![](images/9e95a2fd0aa37c26ad147b3d887f3b1f9c678adb81bdbbd2fe82e609071b7122.jpg)  
Fig. 9: Speedup of TENS-GPU on H100 over BASE-MC on Xeon 8480+. Cells show TENS-GPU speedup and runtime.

## D. GPU Speedup Analysis

Since no GPU implementation of the HSMM Viterbi exists in the literature we use BASE-MC, the multi-core variant we developed by integrating OpenMP parallelization into the sequential C++ backend of hsmmlearn, as our reference. We rely on BASE-MC as a baseline for two reasons: in addition to the total lack of reference GPU implementations, the loopcarried dependencies in the standard sequential formulation across durations and states (s<sub>i</sub>) fundamentally preclude a direct GPU port of the traditional Viterbi algorithm for HSMMs. Consequently, the algorithmic restructuring proposed in this work is the necessary prerequisite for GPU acceleration. Figure 9 reports the resulting speedup of TENS-GPU (H100) over BASE-MC (Xeon 8480+) for $T \in \{ 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$

TENS-GPU achieves speedups across the entire parameter space, peaking at $3 6 . 8 \times \ ( N = 2 5 , \ D = 1 0 0 0 , \ T = 1 0 ^ { 5 } )$ . Performance scales with both N and D: larger N increases the N × N thread-block grid, better saturating the H100’s 132 SMs, while larger D provides longer reduction vectors per block, improving warp-shuffle efficiency. Even at the smallest configuration $( N = 1 0 , \ D = 1 0 0 )$ , the speedup remains 3.6– 4.0×, despite only 100 thread blocks being insufficient to fully occupy all SMs.

For large $N \ ( N { = } 7 5 )$ , the speedup plateaus (e.g., 17.7× at $D = 1 0 0 0 , T = 1 0 ^ { 5 } ) \colon$ : the per-block shared memory footprint grows with $D ,$ and the second reduction kernel over $s _ { i }$ states becomes a bottleneck as N increases. At $T \ = \ 1 0 ^ { 6 }$ the advantage grows further, reaching 54.3× at N =15, D =500: BASE-MC requires over 5.7 minutes on 112 CPU cores, while

![](images/777b85376e8eac1c54db9cb25f41d6b78f2ce7cd94b9f63cf821c290f6965532.jpg)  
(a) Speedup of TENS-1C over BASE-1C.

![](images/076d5ae055cf306c5169e59fb2f637e2ee734ea2b7b990e236d526a884b79d0c.jpg)  
(b) Speedup of TENS-MC and TENS-GPU over BASE-MC for N=50.  
Fig. 10: Speedup of TENS-1C, TENS-MC, and TENS-GPU over different CPUs and GPUs, for $\mathrm { T } { = } 1 , 0 0 0 , 0 0 0 .$

TENS-GPU completes the same decoding in 6.4 seconds. Notably, the absolute GPU runtimes remain sub-second for most configurations and never exceed 10 seconds even at the largest tested $( N = 7 5 , D = 1 0 0 0 , T = 1 0 ^ { 5 } )$ , making interactive-scale HSMM decoding on large inputs feasible for the first time. These runtimes open the door to problem sizes that were previously intractable, as we will explore in Sec. V-G.

## E. Architecture Comparison

Figure 10a compares TENS-1C against BASE-1C at $T =$ $1 0 ^ { 6 }$ with $N \in \{ 1 0 , 1 5 , 2 5 \}$ on the three CPUs described in Table III. TENS-1C delivers consistent speedups of 10–11× on Grace, 11–14× on Xeon, and 11–12× on EPYC across all analyzed configurations, confirming that the gains are portable across architectures.

In absolute terms, for $N = 2 5$ and $D = 1 0 0 0 .$ , BASE-1C requires 2.3 hours on Xeon, which TENS-1C reduces to 12.4 minutes. For $N = 5 0$ and $N = 7 5$ (not shown), speedups remain consistent. At N =75 and D =500, TENS-1C reduces the runtime from 9.2 hours to 56 minutes on Xeon; for D =1000, BASE-1C timed out (exceeding 15 hours), whereas TENS-1C completed in 2 hours. Grace exhibits the lowest runtime, due to its (almost 2×) higher memory bandwidth.

Figure 10b extends the comparison to TENS-MC and TENS-GPU at $T = 1 0 ^ { 6 }$ with N = 50 across all the CPUs and GPUs introduced in Table III. For TENS-MC, speedups are reported over BASE-MC on the same CPU. For TENS-GPU, speedups are reported over the CPU where BASE-MC is fastest (Grace).

On the CPU side, at D = 1000 TENS-MC shows a speedup over BASE-MC of 2× on EPYC and 8× on Xeon. On the GPU side, the speedup over BASE-MC grows steadily with D, as increasing the duration expands the per-block workload and improves SM occupancy. At $D = 1 0 0 0$ , the speedups over BASE-MC range from 5× on MI250X to 12× on H200. The lower performance for MI250X can be attributed to its lower memory bandwidth and fewer compute units per GCD.

Across all GPUs, the absolute runtimes at D = 500 remain below 30 seconds for a million-step sequence. For comparison, the original unmodified hsmmlearn baseline BASE-1C requires 3.8 hours on Grace for this configuration; TENS-GPU on a H200 completes the same decoding in 24 seconds, a reduction of 570×.

It is worth mentioning that, to maximize the breadth of our architectural comparison, we opted for a single CUDA/HIP GPU codebase, and OpenMP-only multi-core parallelization. Further specialization is possible on both fronts, and the GPU kernels could exploit architecture-specific features such as distinct memory hierarchies or generation-specific instructions.

## F. Energy Consumption

Figure 11a reports the energy consumption of all implementations at $N = 5 0 , T = 1 0 ^ { 4 }$ , normalized to BASE-1C. For the sake of space, and because we observed a similar trend for the other values of D, we only report the data for $D \in \{ 1 0 0 , 1 0 0 0 \}$ . For the CPU versions, we measure the energy on the EPYC 7A53, whereas for TENS-GPU we measure the energy on the MI250X. In both cases, energy is monitored through the Cray Power Management (PM) counters [32]. The dominant factor is execution time: since the instantaneous power draw remains comparable across CPU implementations, energy reductions closely track runtime reductions.

![](images/f4528caf2111220645d472ee258f751db029ce5b0df18982851d2a5817aff932.jpg)

![](images/20cef94382dab6145b53f736c251542ba270540043671698861c7146cf0ff79f.jpg)  
(a) Energy consumption over (b) TENS-GPU performance on a BASE-1C for $N = 5 \dot { 0 } , T = 1 0 ^ { 4 }$ stress test case with $T \ = \ 1 0 ^ { 7 }$ on EPYC 7A53 and MI250X. N = 100, $D = 1 0 ^ { 4 }$  
Fig. 11: Energy consumption and stress test.

TENS-1C reduces energy consumption by ∼10× over BASE-1C across all tested D values, consistent with its singlecore speedup. In the multi-core regime, TENS-MC consumes 525.5 J at $D = 1 0 0 0$ , a 3× reduction compared to BASE-MC (1.5 kJ), demonstrating that the tensor reformulation translates its runtime advantage into proportional energy savings. TENS-GPU achieves the lowest energy footprint, requiring only 327.7 J at $D { = } 1 0 0 0$ and 85.2 J at $D = 1 0 0$ (just 2% of BASE-1C). Although the GPU exhibits higher instantaneous power draw, its shorter execution time more than compensates, rendering it the most energy-efficient platform for $D > 1 0 0$ . These results confirm that the tensor formulation not only accelerates HSMM Viterbi decoding but also enables a significantly more energy-efficient profile.

## G. Stress Test: Beyond Current Workloads

To demonstrate the practical impact of our formulation, we evaluate TENS-GPU on an extreme-scale configuration: $N ~ = ~ 1 0 0$ states, $D \ = \ 1 0 { , } 0 0 0$ maximum duration, and $T = 1 0 ^ { 7 }$ time steps. This scale is entirely inaccessible to the baseline; BASE-1C triggers a memory allocation failure (std::bad\_alloc) for configurations exceeding $T = 1 0 ^ { 6 }$ and $D = 1 { , } 0 0 0$ , precluding direct measurement. By extrapolating from runtimes measured at $N = 7 5 , D = 1 0 0 , T = 1 0 ^ { 6 }$ (where BASE-1C requires 2 h and BASE-MC 6.7 min) using the $O ( T \cdot N ^ { 2 } \cdot D )$ theoretical complexity, we estimate this extreme configuration would require approximately 148 days for BASE-1C and 2.1 days for BASE-MC on 112 cores. Such runtimes render not only individual decoding tasks impractical but also make Viterbi training, which requires dozens of such iterations, entirely infeasible on traditional architectures.

Figure 11b reports the per-iteration runtime of TENS-GPU on five GPUs. The H200 leads at 53.7 minutes, followed by the MI300X at 57.3 minutes, the H100 at 1.7 hours, the A100 at 2.0 hours, and the MI250X at 3.4 hours. The H200 and MI300X’s advantage over the H100 is consistent with their higher memory bandwidth and larger number of compute units: at this scale, the $N \times N = 1 0 { , } 0 0 0$ thread-block grid fully saturates both architectures, and performance becomes bandwidth-bound, favoring the MI300X and H200. The A100 trails the H100 due to its lower bandwidth and fewer SMs. The MI250X, despite its 110 CUs per GCD, is bottlenecked by its HBM2e bandwidth, the lowest among the five.

These results demonstrate that TENS-GPU reduces a previously intractable workload, estimated to take over a month on a single core, to less than an hour on a single GPU, making whole-genome-scale HSMM decoding and iterative Viterbi training practically feasible even for larger sequences.

## VI. RELATED WORK

The Viterbi algorithm has been fundamental in highperformance bioinformatics and signal processing for decades, yet existing acceleration efforts are almost exclusively devoted to standard HMMs. Moving from HMMs to HSMMs introduces explicit state-duration handling that substantially increases computational complexity: the Viterbi iteration must compute a maximum over all candidate durations, making the inner loops data-dependent and inherently difficult to parallelize. This combination of computational burden and parallelization difficulty helps explain why performant, hardwareaware HSMM decoders remain absent from the literature.

## A. Hidden Markov Models (HMMs)

HSMMs generalize standard HMMs by introducing explicit state-duration distributions, raising computational complexity from $O ( T N ^ { 2 } )$ to $O ( T N ^ { 2 } D )$ . The Viterbi algorithm has been extensively accelerated for the simpler HMM formulation, including SIMD-vectorized CPU frameworks [18], [20], CUDAbased GPU implementations [19], [33]–[35], hardware– software co-design and domain-specific approaches [17], [36], and distributed computing [37]. However, the additional duration dimension cannot simply be wrapped around existing HMM accelerators: it introduces a cumulative emission prod uct over the d most recent observations, requires accessing a variable-depth window of past delta values, and turns the per-state maximum into a joint maximization over both states and durations. As a consequence, none of these efforts extend to HSMMs, and the HSMM formulation remains entirely unaddressed.

## B. Hidden Semi-Markov Models (HSMMs)

Several statistical frameworks implement major HSMM algorithms in R or Python, with performance-critical routines in C/C++ [13]. Domain-specific solutions also exist, such as biomvRhsmm [14] for genomic segmentation, and Pertsinidou and Limnios [38] that propose Viterbi algorithms based on the backward recurrence Markov chain formulation. All of these implementations, however, are sequential and singlethreaded, and none explicitly targets modern high-performance CPUs or GPUs. The most closely related work is Lu et al. [39], who propose a Tensor-based HSMM (T-HSMM) for user activity analysis in Cyber-Physical-Social Systems (CPSSs). Their objective differs fundamentally from ours: their tensor refers to embedding multiple correlated entities in a unified higher-dimensional space for activity modeling, rather than targeting computational acceleration, whereas ours reshapes the three inner loops of the Viterbi inductive phase into several 3D tensor operations that expose parallelism for high-performance CPU and GPU execution. Moreover, Lu et al. collapse the duration dimension into a single scalar expected value per state, which alters the HSMM semantics and does not solve the exact Viterbi decoding problem. Our formulation instead preserves the full duration dimension and performs exact decoding. A direct head-to-head performance comparison is therefore not meaningful, since the two methods solve different problems.

## C. Summary

To the best of our knowledge, no prior work presents a high-performance implementation of the Viterbi algorithm for HSMMs. Our work fills this gap by proposing a tensor-based reformulation of the HSMM Viterbi algorithm, opening the way to new optimization strategies, accelerator implementations, and application-specific mappings for domains that require Hidden Semi-Markov Model modeling.

## VII. DISCUSSION

We now discuss the main design choices behind our formulation, the trade-offs they involve, and the technical directions they leave open.

## A. Alternative Algorithmic Formulations

A lower-complexity formulation is in principle available by factorizing the induction, reducing over source states before combining the duration and emission terms, which lowers the per-step cost from $O ( N ^ { 2 } D )$ to $O ( N D + N ^ { 2 } )$ . We do not adopt it because the saving in arithmetic is offset by a loss of hardware efficiency. On CPU, the factorization removes the time-invariant Brick precomputation and with it the contiguous $( j , d , i )$ layout that keeps the working set resident across the duration sweep. On GPU, it splits a single joint maximization over the $( s _ { i } , d )$ plane into two reductions that must run one after the other, adding a second grid-wide synchronization per time step and reducing occupancy, which is exactly the pattern our two-kernel design avoids. Evaluating this factorization under a different tensor formulation, built around its own data layout and reduction scheme, would nonetheless be an interesting direction.

## B. Mapping onto Specialized Accelerators

Since emerging AI accelerators and dataflow architectures are designed precisely to execute dense tensor operations, mapping our formulation onto tensor cores, TPUs, systolic arrays, or FPGA dataflow designs is a natural direction to consider. The obstacle is the kind of reduction involved. Since all quantities are handled in log-space, each step combines values with an addition and then selects a maximum. These accelerators are instead built around multiply-accumulate pipelines, so a maximum-based reduction does not map directly onto their native primitives and would need a dedicated mapping strategy. Studying how to support such operations on this class of hardware would therefore be valuable well beyond our setting, since it would open these units to dynamic programming algorithms in general. A further consideration is that we use double precision to match the baseline exactly, whereas peak throughput on these units is available only at lower precision, so any port must first verify that the dynamic range of the problem allows a narrower format.

## C. Sequence-Level and Distributed Parallelism

Our implementations decode a single sequence at a time, from start to end. A natural extension is to split a long sequence into chunks, decode them in parallel, and then reconcile the results at the chunk boundaries. This would also enable multi-node execution, where each node handles a portion of the sequence and the boundary values are exchanged through collective operations. Decoding several independent sequences at once is another promising direction, since it would keep the device busy on small inputs, where a single sequence leaves many units idle.

## D. Model Assumptions and Algorithmic Scope

Our formulation uses one global maximum duration D for all states, which keeps the tensor dense and the work per thread uniform. Giving each state its own bound $D _ { j }$ would avoid computing durations that a state can never take, at the cost of an irregular Brick. We also assume discrete emissions, which makes the emission term a simple table lookup and enables our cached update; continuous densities such as Gaussians would require a different caching strategy. Finally, the Forward-Backward and Baum-Welch procedures iterate over the same $( s _ { j } , s _ { i } , d )$ combinations and only replace the maximum with a sum, so they can reuse the same tensor operations and be accelerated in the same way.

## VIII. CONCLUSIONS

We presented a tensor-based formulation of the Viterbi algorithm for Hidden Semi-Markov Models that restructures the three inner loops of the sequential algorithm into dense tensor operations, exposing optimization opportunities that are inaccessible to the traditional scalar formulation. Building on it, we delivered optimized single-core CPU, multi-core CPU, and, for the first time for HSMMs, GPU implementations, released as the open-source library tensor-hsmm.

Across three CPU and five GPU architectures, our implementations achieve speedups of up to 14× on a single core, over 200× with multi-core, and over 570× on GPU with respect to the sequential BASE-1C baseline, while producing output identical to hsmmlearn in every tested configuration. The gains are structural rather than platform-specific: a direct NumPy transcription of the formulation already outperforms the compiled sequential baseline by 4.5×, and profiling attributes the compiled speedups to a 36× reduction in retired instructions together with vectorization ratios rising from below 0.001% to 80.4%. Because instantaneous power draw is comparable across implementations, these runtime reductions translate into proportional energy savings, with the GPU version consuming as little as 2% of the baseline energy. Most consequentially, a configuration estimated to require over a month of single-core execution completes in under an hour on a single GPU, bringing whole-genome-scale HSMM decoding and iterative Viterbi training within practical reach and establishing a new performance baseline for large-scale HSMM inference.

## ACKNOWLEDGEMENTS

We acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy). We acknowledge the EuroHPC Joint Undertaking, the LUMI consortium, and BSC for granting access to the LUMI and MareNostrum 5 supercomputers. These resources, hosted by CSC (Finland) and the Barcelona Supercomputing Center (Spain), were provided through the EuroHPC Regular Access program. The authors used Claude Opus 4.6 and Gemini 3 for editing the paper; all ideas, content, and conclusions are their own.

[1] L. Gabriel, T. Bruna, K. J. Hoff, M. Ebel, A. Lomsadze, M. Borodovsky,˚ and M. Stanke, “Braker3: Fully automated genome annotation using rna-seq and protein evidence with genemark-etp, augustus, and tsebra,” Genome research, vol. 34, no. 5, pp. 769–777, 2024. [Online]. Available: https://doi.org/10.1101/gr.278090.123

[2] S. Qin, Z. Tan, and Y. Wu, “On robust estimation of hidden semi-Markov regime-switching models,” Annals of Operations Research, 2024. [Online]. Available: https://doi.org/10.1007/s10479-024-05989-4

[3] H. Zen, K. Tokuda, T. Masuuko, T. Kobayasih, and T. Kitamura, “A hidden semi-markov model-based speech synthesis system,” IEICE TRANSACTIONS on Information, vol. E90-D, no. 5, pp. 825–834, May 2007. [Online]. Available: https://doi.org/10.1093/ietisy/e90-d.5.825

[4] S.-Z. Yu, “Hidden semi-markov models,” Artificial Intelligence, vol. 174, no. 2, pp. 215–243, 2010, special Review Issue. [Online]. Available: https://doi.org/10.1016/j.artint.2009.11.011

[5] I. Korf, “Gene finding in novel genomes,” BMC bioinformatics, vol. 5, no. 1, p. 59, 2004. [Online]. Available: https://doi.org/10.1186/ 1471-2105-5-59

[6] J. Ernst and M. Kellis, “ChromHMM: Automating chromatin-state discovery and characterization,” Nature Methods, vol. 9, no. 3, pp. 215–216, 2012. [Online]. Available: https://doi.org/10.1038/nmeth.1906

[7] R. Durbin, S. R. Eddy, A. Krogh, and G. Mitchison, Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids. Cambridge University Press, 1998. [Online]. Available: https://doi.org/10.1017/CBO9780511790492

[8] L. Rabiner, “A tutorial on hidden markov models and selected applications in speech recognition,” Proceedings of the IEEE, vol. 77, no. 2, pp. 257–286, 1989. [Online]. Available: https: //doi.org/10.1109/5.18626

[9] L. E. Baum, T. Petrie, G. Soules, and N. Weiss, “A maximization technique occurring in the statistical analysis of probabilistic functions of markov chains,” The Annals of Mathematical Statistics, vol. 41, no. 1, pp. 164–171, 1970. [Online]. Available: http: //www.jstor.org/stable/2239727

[10] A. P. Dempster, N. M. Laird, and D. B. Rubin, “Maximum likelihood from incomplete data via the em algorithm,” Journal of the Royal Statistical Society: Series B (Methodological), vol. 39, no. 1, pp. 1–22, 1977. [Online]. Available: https://doi.org/10.1111/j.2517-6161. 1977.tb01600.x

[11] A. Lomsadze, V. Ter-Hovhannisyan, Y. O. Chernoff, and M. Borodovsky, “Gene identification in novel eukaryotic genomes by self-training algorithm,” Nucleic acids research, vol. 33, no. 20, pp. 6494–6506, 2005. [Online]. Available: https://doi.org/10.1093/nar/gki937

[12] Y. Guedon, “Estimating hidden semi-markov chains from discrete´ sequences,” Journal of Computational and Graphical Statistics, vol. 12, no. 3, pp. 604–639, 2003. [Online]. Available: https: //doi.org/10.1198/1061860032030

[13] C. Berard, M.-J. CROS, J.-B. DURAND, C. Lothod ´ e, S. Plancade,´ R. Trepos, and N. Vergne, “Review of hsmm r and python softwares,” A Comprehensive Guide to HSMM: Theory, Software, and Advanced Extensions, pp. 47–77, 2025. [Online]. Available: https://doi.org/10.1002/9781394427581.ch2

[14] Y. Du, E. Murani, S. Ponsuksili, and K. Wimmers, “biomvrhsmm: Genomic segmentation with hidden semi-markov model,” BioMed Research International, vol. 2014, no. 1, p. 910390, 2014. [Online]. Available: https://doi.org/10.1155/2014/910390

[15] J. Bulla and I. Bulla, hsmm: Hidden Semi Markov Models, 2013, r package, archived May 2022. [Online]. Available: https: //cran.r-project.org/package=hsmm

[16] J. Vankerschaver, “hsmmlearn: A library for hidden semi-Markov models with explicit durations,” 2021, c++/Cython with Python interface. Wraps the C++ code from the R hsmm package. Archived January 2023. [Online]. Available: https://github.com/jvkersch/ hsmmlearn

[17] C. Firtina, K. Pillai, G. S. Kalsi, B. Suresh, D. S. Cali, J. S. Kim, T. Shahroodi, M. B. Cavlak, J. Lindegger, M. Alser et al., “Aphmm: Accelerating profile hidden markov models for fast and energy-efficient genome analysis,” ACM Transactions on Architecture and Code Optimization, vol. 21, no. 1, pp. 1–29, 2024. [Online]. Available: https://doi.org/10.1145/3632950

[18] S. R. Eddy, “Accelerated profile hmm searches,” PLoS computational biology, vol. 7, no. 10, p. e1002195, 2011. [Online]. Available: https://doi.org/10.1371/journal.pcbi.1002195

[19] L. Yu, Y. Ukidave, and D. Kaeli, “Gpu-accelerated hmm for speech recognition,” in 2014 43rd International Conference on Parallel Processing Workshops. IEEE, 2014, pp. 395–402. [Online]. Available: https://doi.org/10.1109/ICPPW.2014.59

[20] H. Jiang, N. Ganesan, and Y.-D. Yao, “Cudampf++: A proactive resource exhaustion scheme for accelerating homologous sequence search on cuda-enabled gpu,” IEEE Transactions on Parallel and Distributed Systems, vol. 29, no. 10, p. 2206–2222, Oct. 2018. [Online]. Available: http://dx.doi.org/10.1109/TPDS.2018.2830393

[21] S. Hassan, S. Sarkka, and A. Garcia-Fernandez, “Temporal parallelization of inference in hidden markov models,” IEEE Transactions on Signal Processing, vol. 69, p. 4875–4887, 2021. [Online]. Available: http://dx.doi.org/10.1109/TSP.2021.3103338

[22] L. S. Blackford, A. Petitet, R. Pozo, K. Remington, R. C. Whaley, J. Demmel, J. Dongarra, I. Duff, S. Hammarling, G. Henry et al., “An updated set of basic linear algebra subprograms (BLAS),” ACM Transactions on Mathematical Software, vol. 28, no. 2, pp. 135–151, 2002. [Online]. Available: https://doi.org/10.1145/567806.567807

[23] xtensor-stack, “xtensor: Multi-dimensional arrays with broadcasting and lazy computing,” https://github.com/xtensor-stack/xtensor, 2025, c++14 header-only library.

[24] AMD, “HIPIFY: Convert CUDA to portable C++ code,” 2024, accessed: 2026-04-06. [Online]. Available: https://github.com/ROCm/HIPIFY

[25] poypoyan, “edhsmm: An(other) implementation of explicit duration hidden semi-Markov models in Python 3,” 2024, python/Cython. Archived May 2024. [Online]. Available: https://github.com/poypoyan/ edhsmm

[26] A. V. Lukashin and M. Borodovsky, “Genemark. hmm: new solutions for gene finding,” Nucleic Acids Research, vol. 26, no. 4, pp. 1107–1115, 1998. [Online]. Available: https://doi.org/10.1093/nar/26.4.1107

[27] A. Kundaje, W. Meuleman, J. Ernst, M. Bilenky, A. Yen, P. Kheradpour, Z. Zhang, A. Heravi-Moussavi, Y. Liu, V. Amin et al., “Integrative analysis of 111 reference human epigenomes,” Nature, vol. 518, no. 7539, p. 317, 2015. [Online]. Available: https://doi.org/10.1038/ nature14248

[28] M. Stanke, “Gene prediction with a hidden markov model and a new intron submodel,” Bioinformatics, 2003. [Online]. Available: https://doi.org/10.1093/bioinformatics/btg1080

[29] C. Burge and S. Karlin, “Prediction of complete gene structures in human genomic dna,” Journal of molecular biology, vol. 268, no. 1, pp. 78–94, 1997. [Online]. Available: https://doi.org/10.1006/jmbi.1997. 0951

[30] M. K. Sakharkar, V. T. Chow, and P. Kangueane, “Distributions of exons and introns in the human genome,” In silico biology, vol. 4, no. 4, pp. 387–393, 2004. [Online]. Available: https://doi.org/10.3233/ISB-00142

[31] J. Treibig, G. Hager, and G. Wellein, “LIKWID: A lightweight performance-oriented tool suite for x86 multicore environments,” in Proceedings of PSTI2010, the First International Workshop on Parallel Software Tools and Tool Infrastructures, 2010, pp. 207–216. [Online]. Available: https://doi.org/10.1109/ICPPW.2010.38

[32] HPE Cray, “Cray Performance and Analysis Tools cray pm,” 2026, accessed: 2026-04-06. [Online]. Available: https://cpe.ext.hpe.com/docs/ 24.03/performance-tools/index.html

[33] M. HoseinyFarahabady and A. Y. Zomaya, “Gpu-accelerated out-ofcore hmm inference with concurrent cuda streams,” in International Conference on Computational Science. Springer, 2025, pp. 369–376. [Online]. Available: https://doi.org/10.1007/978-3-031-97635-3 44

[34] A. Mohammadidoost and M. Hashemi, “High-throughput and memoryefficient parallel viterbi decoder for convolutional codes on gpu,” arXiv preprint arXiv:2011.09337, 2020. [Online]. Available: https: //doi.org/10.48550/arXiv.2011.09337

[35] V. Roubtsova, “Parallel algorithm for a hidden markov model with an indefinite number of states and heterogeneous observation data.” in IWOCL, 2023, pp. 31–1. [Online]. Available: https: //doi.org/10.1145/3585341.3587954

[36] L. Hummelgren, V. Palmkvist, L. Stjerna, X. Xu, J. Jalden,´ and D. Broman, “Trellis: A domain-specific language for hidden markov models with sparse transitions,” in Proceedings of the 17th ACM SIGPLAN International Conference on Software Language Engineering, 2024, pp. 196–209. [Online]. Available: https://doi.org/10. 1145/3687997.3695641

[37] I. Sassi, S. Anter, and A. Bekkhoucha, “Paradist-hmm: A parallel distributed implementation of hidden markov model for big data analytics using spark,” International Journal of Advanced Computer Science and Applications, vol. 12, no. 4, 2021. [Online]. Available: http://dx.doi.org/10.14569/IJACSA.2021.0120438

[38] C.-E. Pertsinidou and N. Limnios, “Viterbi algorithms for hidden semimarkov models with application to dna analysis,” RAIRO-Operations Research, vol. 49, no. 3, pp. 511–526, 2015. [Online]. Available: https://doi.org/10.1051/ro/2014053

[39] Z. Lu, L. T. Yang, A. Azman, F. Zhou, S. Zhang, and X. Fu, “Tensorbased hidden semi-markov model for cpss user activity analysis and services,” IEEE Transactions on Services Computing, 2025. [Online]. Available: https://doi.org/10.1109/TSC.2025.3618011