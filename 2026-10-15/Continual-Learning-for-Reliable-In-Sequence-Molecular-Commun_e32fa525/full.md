# Continual Learning for Reliable In-Sequence Molecular Communication

Siddhant Setia

Junichi Suzuki

Tadashi Nakano

Department of Computer Science

Department of Computer Science

Graduate School of Informatics

University of Massachusetts, Boston

University of Massachusetts, Boston

Osaka Metropolitan University

Boston, MA, 02125, USA

siddhant.setia001@umb.edu

Boston, MA, 02125, USA

jxs@cs.umb.edu

Osaka, Japan

tnakano@omu.ac.jp

© 2026, copyrighted by International Institute of Applied Informatics. DOI: 10.23919/IIAI-AAICPS00095.2026.00072

Abstract—This paper considers reliable in-sequence molecular communication, where bio-nanomachines exchange moleculeencoded information in a specified order, and investigates a new simulation-based method to incrementally evaluate communication protocols by gradually increasing the complexity of experimental settings. The proposed method, called Affine Memory Replay (AMR), builds a continual learning (CL) model to estimate communication performance between bio-nanomachines. AMR stores past simulation samples and synthesizes additional replay samples through interpolation and extrapolation between past samples. Through incremental performance estimation tasks, it replays both past and synthetic samples to regularize model updates, helping the model adapt to new experimental settings while retaining knowledge from earlier ones. Experimental results show that AMR significantly outperforms existing CL methods in maintaining overall estimation accuracy across sequentially learned estimation tasks. It effectively balances memory stability and learning plasticity under moderate training cost.

## I. INTRODUCTION

Molecular communication is a communication paradigm for networking bio-nanomachines such as modified biological cells and artificial cells (cell-like artificial nanostructures engineered with biological materials) [1]–[3]. Using molecules as information carriers among bio-nanomachines, it is expected to enable various biomedical and healthcare applications, including in-body physiological sensing, biomedical anomaly detection, targeted drug delivery, cellular/molecular-level medical procedures, and artificial morphogenesis [4]–[6].

Molecular communication research often advances stepwise. When a communication protocol is studied, for example, it may first be examined under the simplest setting, such as a 1D noiseless space, to evaluate its characteristics and effectiveness. Then, it can be evaluated under more realistic settings by gradually introducing additional requirements or relaxing prior assumptions. The protocol may be examined in 2D and then 3D spaces by considering noise. While the transmitter (Tx) and receiver (Rx) bio-nanomachines may initially be placed near each other, the Tx-Rx distance can be gradually expanded, potentially with relay bio-nanomachines. Bio-nanomachines may eventually become mobile nodes.

As the complexity of these experimental settings increases, simulation studies are conducted in addition to analytical work.

When these simulations are computationally expensive, they are often augmented by machine learning (ML) methods. For example, if an ML model is built for a protocol, it can be trained on prior simulation results to estimate performance measures such as latency and error rate. This can reduce the computational cost of running additional simulations.

Since protocol studies often proceed stepwise, simulation results are collected sequentially across increasingly complex settings. The ML model must therefore be updated as new simulation results become available, preferably without being retrained from scratch. For example, when transitioning from 1D to 2D spaces, a model previously trained on 1D simulation results is updated with 2D results. However, the model “forgets” the 1D setting and yields poor performance estimates for that setting, although it works well in 2D spaces. If the model is trained further with 3D results, it forgets the 1D setting even more and becomes even less capable of estimating performance there. This is called “forgetting” in the ML research [7]. The challenge is to have an ML model acquire new knowledge from sequentially collected simulation results, integrate it with current knowledge, and retain previously-learned knowledge effectively.

To address this challenge, this paper proposes a continual learning (CL) method that adapts incrementally to new experimental settings for communication latency estimation while retaining knowledge from earlier estimation tasks. It is evaluated with a feedback-based protocol for bio-nanomachines to perform in-sequence and timing-controlled communication, in which they transmit and receive a series of informationcarrying molecules (information molecules) in a certain order. The protocol employs Stop-and-Wait Automatic Repeat Request (SW-ARQ) [8], which is feasible to realize with simple computation and memory functions in bio-nanomachines.

The proposed CL method, Affine Memory Replay (AMR), operates on a simple feedforward neural network (NN). It synthesizes new training samples from past training samples (i.e., prior simulation results), buffer both past and synthetic samples, and replays them during later training stages to reinforce past knowledge. Replay performance is used to regularize updates to the NN model parameters (e.g., weights and biases). This replay strategy allows AMR to retain knowledge from earlier experimental settings while adapting to newly introduced settings, thereby balancing memory stability and learning plasticity in sequential performance estimation.

Experimental results demonstrate that AMR maintains higher overall RTT estimation accuracy than existing CL methods across sequentially learned estimation tasks by reducing forgetting while preserving competitive learning plasticity under moderate training cost.

## II. BACKGROUND AND RELATED WORK

In-sequence molecule delivery can play a critical role in various applications. In artificial morphogenesis applications in regenerative medicine, bio-nanomachines made of living cells are designed to divide and grow into 3D multi-cellular structures such as tissues and organs. Molecular communication allows them to interact with each other using artificial morphogens (AMs), which are modeled as information molecules that encode morphological information. Reliable insequence molecule delivery provides a foundation for guiding the growth and differentiation of bio-nanomachines by controlling the order and timing of propagated AMs.

In drug delivery applications, therapeutic bio-nanomachines leverage natural molecular signals in the body, or molecular signals released by other bio-nanomachines, to identify target sites and release drug molecules, thereby reducing side-effects at non-target sites. Reliable in-sequence delivery allows bionanomachines to modulate tumor behavior by controlling the order and duration of drug molecule exposure.

To enhance the reliability of molecular communication, several feedback-based protocols have been studied. Feedbackbased rate control schemes are proposed in [9], [10]. Insequence and at-least-once molecule delivery schemes are examined with SW-ARQ protocols in [11]–[14]. None of these protocols use ML; this paper extends this line of work by developing a CL method for performance estimation.

Various ML methods have been studied for molecular communication; e.g. [15]–[21]. Most of them perform regression tasks, as in this paper, while [21] considers a classification task, using a convolutional NN (CNN) to predict the source of each information molecule for molecular index modulation. Cheng et al. estimate the arrival time of propagated molecules at the Rx [15], [16]. They use a transformer for a 2D noiseless environment in [15] and a multi-layer perceptron for a 3D noiseless environment in [16]. In [17], [18], NNs are used to estimate the fraction of molecules that arrive at the Rx by a certain deadline in 3D noiseless environments. In [19], a CNN and a recurrent NN (RNN) are used to predict the type of information encoded in a molecule in a 3D noisy environment. In [20], the location of a mobile Tx is estimated with a transformer in a 2D noiseless environment. None of these ML-based studies employ CL; their ML models require retraining from scratch and may suffer from forgetting if they are incrementally trained on different experimental settings.

This paper builds on the authors’ prior work, which studied the impacts of CL in molecular communication by customizing existing methods [22]. In contrast, the present study develops a new CL method and evaluates it against existing methods.

CL methods can be classified into three approaches: regularization-, replay- and architecture-based [23], [24]. Regularization constrains updates to NN model parameters (e.g., weights and biases), thereby reducing interference with previously learned knowledge. Replay reintroduces information from previous tasks during training to reinforce past knowl edge. There are three major replay strategies: (1) experience replay (or rehearsal), which reuses past training samples; (2) generative replay, which synthesizes new samples from past data with a dedicated generator; and (3) feature/latent replay, which replays features (or embeddings) rather than raw samples. Architecture-based methods modify the NN model structure, for example, by expanding network capacity. This paper adopts a hybrid rehearsal-regularization approach for AMR: it replays past and synthetic samples and uses replay loss to regularize training and discourage parameter updates that degrade performance on replayed samples.

## III. MOLECULAR COMMUNICATION MODEL

This paper assumes a bounded 3D aqueous medium where the transmitter (Tx) and receiver (Rx) bio-nanomachines exchange information and acknowledgement (ACK) molecules. Motivated by wet-lab experiments of molecular communication (e.g., [25]), DNA molecules are used as information and ACK molecules that can encode data as nucleotide sequences.

This paper considers three types of communication transports: (1) diffusive transports, (2) directional transports where molecules move directionally move on pre-defined protein filaments (e.g., microtubules and actin filaments) with molecular motors, and (3) diffusive-directional hybrid transports that combines diffusive and directional transports (Fig. 1). The Tx releases information molecules, each of which encodes a certain message and carries it to the Rx (Fig. 2). The Rx is assumed to capture an information molecule when the molecule physically contacts with the Rx. The Rx releases ACK molecules, each of which encodes an acknowledgement of the message. The Tx is assumed to capture an ACK molecule when it contacts the Tx’s surface. Noise molecules are stationary and interfere (i.e., collide) with other molecules.

In a diffusive transport, a molecule diffuses via random thermal motion, which is governed by the diffusion coefficient D in each dimension: $D = \partial x ^ { 2 } / ( 2 \times \partial t )$ where x denotes the distance of molecular movement over time t in a particular dimension. When a molecule collides with another molecule, it randomly moves to another position according to D.

A directional transport utilizes a microtubule between the Tx and Rx. Based on findings in biomedical engineering (e.g., [26]), the microtubule is configured to guide molecular motors (e.g., kinesin, dynein, and myosin) directionally. Molecules are attached to molecular motors that travel at a constant velocity for an expected distance before diffusing away. Collision with another molecule also causes diffusion. While diffusing, a molecule with a molecular motor may contact the microtubule and resume directional transport.

![](images/4738c090cf4650ee2515c8cb4424b46041c8f1017ef224723b82e332c71a8bb9.jpg)  
Fig. 1. Diffusive, Directional and Hybrid Transports with SW-ARQ in Molecular Communication

![](images/ff28947b37565f21034de82a73f63def52be0438d8da25722a001c65e1bc5895.jpg)  
Fig. 2. Interaction between the Tx and the Rx

A hybrid transport uses both diffusive and directional transports. This paper deploys a directional transport for information molecules and a diffusive transport for ACK molecules.

Promising approaches to engineer bio-nanomachines with communication capability include modifying biological cells and producing artificial cell-like structures using biological materials [27]. Here, bio-nanomachines are modeled as modified biological cells that can possess communication-related functions, including transmission and reception functions, logic gates, toggle switches (i.e., 1-bit memories) and oscillators (i.e., clocks). These functions allow them to synthesize, release and capture specific molecules, trigger programmed chemical responses, retain communication states and control the timing of molecule release.

This paper employs SW-ARQ for molecular transmissions. The Tx releases n duplicated information molecules that encode a particular message (e.g., Message 1 in Figs. 2). Once the Rx receives one of them, it releases n duplicated ACK molecules. It does not respond to subsequent arrivals of the information molecules that encode the same message. Upon receiving one of the ACK molecules, the Tx releases n information molecules that encode the next message (e.g., Message 2 in Figs. 2). It does not respond to subsequent arrivals of the ACK molecules for the previous message.

If the Tx does not receive an ACK molecule for a message within a certain interval, called retransmission timeout interval (RTO), it retransmits another set of n information molecules for the message. Molecule retransmission may occur multiple times until the Tx receives an ACK molecule or exceeds a predefined number of retransmissions. If the Rx does not receive any information molecules that encode the i-th message within RTO after sending ACK molecules for the (i−1)-th message, it retransmits n ACK molecules for the previous message.

## IV. CONTINUAL PERFORMANCE ESTIMATION

This paper uses the protocol described in Section III as a testbed for CL-based performance estimation tasks. Each task is defined by experimental settings, including environmental parameters (e.g., environment size, noise level and Tx-Rx distance) and communication parameters (e.g., transport choice, the number of duplicated information/ACK molecules and RTO). The goal is to estimate the roundtrip time (RTT) of message delivery under those settings. RTT is measured as the time from the Tx releasing information molecules for a message to receiving the corresponding ACK molecule.

The RTT estimation model takes experimental settings as input and outputs an estimated RTT value. The model is trained incrementally through estimation tasks. Each task t runs a series of simulations under the current experimental settings and feeds simulation results $D _ { t }$ to the model to perform a supervised training session. Therefore, given the total number of tasks T, the model is trained on a stream of simulation results $\{ \mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } , \ldots , \mathcal { D } _ { t } , \ldots , \mathcal { D } _ { T } \}$ $D _ { t }$ consists of training samples (x, y), where x is the input (experimental settings) and y is the estimation target (RTT). The model’s objective is to optimize the estimation function $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ for the current task and all prior tasks. Formally, this requires finding optimized model parameters $\theta ^ { * }$ (weights and biases in the underlying neural network) that minimize the cumulative training loss over all tasks observed so far:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \sum _ { t = 1 } ^ { t _ { c } } \mathcal { L } _ { t }\tag{1}
$$

where $\mathcal { L } _ { t }$ is the loss for task t and $t _ { c }$ is the current task index.

Algorithm 1 shows the replay-based training procedure of AMR. AMR maintains two bounded replay buffers: a primary buffer M to store past samples and a secondary buffer $\mathcal { M } _ { s }$ to store synthetic samples. One sample is drawn from each buffer (Lines 4 and 5) to compute the loss $\mathcal { L } _ { t }$ as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { t } = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { t } } \left[ \ell \big ( f _ { \theta } ( x ) , y \big ) \right] } \\ & { \quad \quad + \alpha \cdot \mathbb { E } _ { ( x ^ { \prime } , z ^ { \prime } ) \sim \mathcal { M } } \left[ \ell \big ( f _ { \theta } ( x ^ { \prime } ) , z ^ { \prime } \big ) \right] } \\ & { \quad \quad + \beta \cdot \mathbb { E } _ { ( x ^ { \prime \prime } , z ^ { \prime \prime } ) \sim \mathcal { M } _ { s } } \left[ \ell \big ( f _ { \theta } ( x ^ { \prime \prime } ) , z ^ { \prime \prime } \big ) \right] } \end{array}\tag{2}
$$

where ℓ denotes the mean squared error loss. The first, second and third terms correspond to the losses for the currenttask sample $( x , y )$ , replayed past sample $( x ^ { \prime } , z ^ { \prime } )$ and replayed synthetic sample $( x ^ { \prime \prime } , z ^ { \prime \prime } )$ , respectively. The computed $\mathcal { L } _ { t }$ is used to update θ (Lines 7 and 8 in Algorithm 1). AMR therefore augments current-task training with replay losses from the primary and secondary buffers, thereby regularizing updates to θ with respect to replayed samples. This helps mitigate forgetting by discouraging updates that degrade replay performance. It can also promote backward knowledge transfer by allowing current-task updates to improve performance on previously learned tasks. The hyperparameters α and $\beta$ control the relative influence of primary- and secondary-buffer replay.

After the parameter updates, AMR updates the primary buffer M with the current input and its estimated target $( x , z )$ (Line 9 in Algorithm 1). This update uses reservoir sampling [28], which inserts a new sample with probability $C / n ,$ where C is the buffer capacity and n is the number of samples observed so far. If selected, the new sample replaces a uniformly chosen sample stored in the buffer (Algorithm 2). This procedure keeps the buffer bounded while maintaining a representative set of past samples.

Algorithm 1 Affine Memory Replay   
Require: prediction function $f _ { \boldsymbol { \theta } } ;$ model parameters $\theta ;$ training   
dataset $\mathcal { D } _ { t }$ for task t; total number of tasks $T ;$ primary and   
secondary replay buffers $\mathcal { M } , \mathcal { M } _ { s } ;$ hyperparameters $\alpha , \beta ;$   
learning rate $\lambda ,$ synthesis range parameter q   
1: $\mathcal { M } , \mathcal { M } _ { s } \gets \{ \}$   
2: for each $t = \stackrel { \cdot \cdot } { 1 } , 2 , . . . , T$ do   
3: for each $( x , y )$ in $\mathcal { D } _ { t }$ do   
4: $( x ^ { \prime } , z ^ { \prime } ) \gets$ random(M)   
5: $( x ^ { \prime \prime } , z ^ { \prime \prime } ) \gets \mathrm { r a n d o m } ( \mathcal { M } _ { s } )$   
6: $z  f _ { \theta } ( x )$   
7: reg $ \alpha { L } _ { m s e } ( z ^ { \prime } , f _ { \theta } ( x ^ { \prime } ) ) + \beta { L } _ { m s e } ( z ^ { \prime \prime } , f _ { \theta } ( x ^ { \prime \prime } ) )$   
8: $\theta \gets \theta - \lambda \cdot \nabla _ { \theta } [ L _ { m s e } ( y , f _ { \theta } ( x ) ) + \mathrm { r e g } ]$   
9: M ← reservoirSampling(M, (x, z))   
10: r ← uniformSampling $\left[ - q , 1 + q \right)$   
11: if $0 \leq r \leq 1$ then   
12: $\mathcal { M } _ { s } \gets$ interpolateReplaySample $( \mathcal { M } , \mathcal { M } _ { s } , r )$   
13: else   
14: $\mathcal { M } _ { s } \gets$ extrapolateReplaySample $( \mathcal { M } , \mathcal { M } _ { s } , r )$   
15: end if   
16: end for   
17: end for   
AMR then synthesizes a new sample using two past sam  
ples drawn from the primary buffer M (Lines 10 to 15   
in Algorithm 1). A random factor r is sampled from a   
uniform distribution over $( - q , 1 + q )$ , where q controls how   
far synthetic samples may extend beyond the line segment

Algorithm 2 reservoirSampling()   
1: Input: replay buffer ${ \overline { { \mathcal { M } } } } ,$ incoming sample $( x , y ) .$ buffer   
capacity $C ,$ total number of observed samples n   
2: if $C > | { \mathcal { M } } |$ then   
3: $\mathcal { M } [ n ]  ( x , y )$   
4: else   
5: $j =$ randomInt $( 0 , n - 1 )$   
6: if $j < | { \mathcal { M } } |$ then   
7: $\mathcal { M } [ j ]  ( x , y )$   
8: end if   
9: end if   
10: $n \gets n + 1$   
11: return $\mathcal { M }$

between the two past samples. If $0 ~ \leq ~ r ~ \leq ~ 1$ , AMR uses interpolation; otherwise, it uses extrapolation.

For interpolation, AMR generates a synthetic input $x _ { s }$ as a convex combination of two sampled inputs (Lines 1 to 3 in Algorithm 3). $x _ { s }$ lies inside the convex hull of the two inputs (i.e., on the line segment between them). By transforming past samples into a convex neighborhood, AMR expands the coverage of the replay space and encourages a smoother, better-generalized approximation of previously learned tasks during later training stages. After constructing $x _ { s } ,$ AMR computes its Euclidean distances to the two sampled inputs, $d _ { 1 }$ and $d _ { 2 }$ (Lines 4 and 5), and uses these distances to compute a synthetic target value $z _ { s }$ through distance-weighted interpolation (Line 6). This way, the synthetic sample $( x _ { s } , z _ { s } )$ preserves information from past samples while introducing additional diversity into the replay process.

If extrapolation is selected (Line 14 in Algorithm 1), $x _ { s }$ is generated outside the line segment between the two sampled inputs, thereby extending the replay space beyond the convex hull of past samples (Lines 1 to 3 in Algorithm 4). After constructing $x _ { s } ,$ AMR computes the synthetic target value $z _ { s }$ by performing linear extrapolation along the affine line defined by the two sampled pairs (Line 4). This allows AMR to generate synthetic samples that preserve the directional trend between past samples while increasing the diversity and coverage of the replay space.

Finally, the generated synthetic sample is inserted into the secondary buffer $\mathcal { M } _ { s }$ using reservoir sampling.

Algorithm 3 interpolateReplaySample()   
Require: primary and secondary replay buffers M, $\overline { { \mathcal { M } _ { s } } } ;$   
random factor r   
1: $( x _ { c } , z _ { c } ) \gets$ random(M)   
2: (x, z) ← random(M)   
3: $x _ { s } \gets x _ { c } + r \times ( x - x _ { c } )$   
4: $d _ { 1 } \gets \mathrm { d i s t } ( x _ { s } , x )$   
5: $d _ { 2 } \gets \mathrm { d i s t } ( x _ { c } , x _ { s } )$   
$d _ { 2 } \cdot \dot { z } + d _ { 1 } \cdot z _ { c }$   
6: z<sub>s</sub> ←   
$d _ { 1 } + _ { . } d _ { 2 }$   
7: return reservoirSampling $\mathcal { M } _ { s } , ( x _ { s } , z _ { s } ) )$

Algorithm 4 extrapolateReplaySample()   
Require: primary and secondary replay buffers M, $\mathcal { M } _ { s } \mathrm { ; }$   
random factor r   
1: $( x _ { c } , z _ { c } ) \gets$ random(M)   
2: $( x , z ) \gets \mathrm { r a n d o m } ( \mathcal { M } - x _ { c } )$   
3: $x _ { s } \gets x _ { c } + r \times ( x - x _ { c } )$   
4: $z _ { s } \gets z _ { c } + r \times ( z - z _ { c } )$   
5: return reservoirSampling $( \mathcal { M } _ { s } , ( x _ { s } , z _ { s } ) )$

## V. EXPERIMENTAL RESULTS

This section evaluates AMR and other CL methods, namely CLeaR [29], ACLT [30] and A2ER [31], through Monte Carlo (particle-based) molecular communication simulations.

This paper implements AMR on a two-layer feedforward neural network with a fully connected hidden layer with ReLU activation. Input and output values are normalized to [0, 1] with min-max normalization.

Table I shows the hyperparameter values for AMR and the other methods. Those for AMR and CLeaR are selected through grid search on a subset of the training dataset. The hyperparameters for ACLT and A2ER are adopted from the original papers [30], [31].

Each method is trained using the stochastic gradient descent (SGD) optimizer for 100 epochs with a batch size of 1 and a learning rate of 0.001. The loss function is the mean squared error (MSE). Early stopping is used for all methods with a patience of 10 epochs and a minimum improvement threshold (min delta) of 0.003.

TABLE I  
HYPERPARAMETER CONFIGURATION
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>AMR</td><td rowspan=1 colspan=1>αβPrimary buffer sizeSecondary buffer sizeSynthesis range param q</td><td rowspan=1 colspan=1>10.550500.1</td><td rowspan=1 colspan=1>CLeaR</td><td rowspan=1 colspan=1>αλBuffer size</td><td rowspan=1 colspan=1>0.5250</td></tr><tr><td rowspan=1 colspan=1>ACLT</td><td rowspan=1 colspan=1>α $\beta$  $\gamma$ Main buffer sizeSoft buffer size</td><td rowspan=1 colspan=1>0.20.20.2501</td><td rowspan=1 colspan=1>A2ER</td><td rowspan=1 colspan=1>αβBuffer Sizeρ€λ∆Q</td><td rowspan=1 colspan=1>10.5500.510⁻120.50</td></tr></table>

## A. Simulation Settings and Evaluation Metrics

Table II shows the simulation settings for molecular communication. Each training and test data sample is prepared with the median RTT obtained from 500 independent simulations.

The boundaries of the environment are simulated in a nonrepulsive manner. Molecules do not rebound against boundaries. Molecule-to-molecule collisions are also simulated in a non-repulsive manner. Each molecule is prohibited to move to the location of another molecule. Colliding molecules do not rebound with each other.

Each CL method (RTT estimation model) is trained incrementally on 12 estimation tasks shown in Fig. 3. In the first task (T1), for example, simulations are carried out using diffusive transport for each pair of Tx-Rx distance and noise level to obtain RTT values. The Tx-Rx distance, noise level and other experimental settings are combined into a training input, and the median RTT is used as the estimation target.

TABLE II  
SIMULATION SETTINGS FOR MOLECULAR COMMUNICATION
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Size length of the environment</td><td rowspan=1 colspan=1>150 µm</td></tr><tr><td rowspan=1 colspan=1>Diameter of Tx and Rx</td><td rowspan=1 colspan=1>5µm</td></tr><tr><td rowspan=1 colspan=1>Tx to Rx distance</td><td rowspan=1 colspan=1>10 to 90 µm</td></tr><tr><td rowspan=1 colspan=1>Diameter of an information/ACK molecule</td><td rowspan=1 colspan=1>1 µm</td></tr><tr><td rowspan=1 colspan=1>Diameter of a noise molecule</td><td rowspan=1 colspan=1>1 µm</td></tr><tr><td rowspan=1 colspan=1>Number of noise molecules</td><td rowspan=1 colspan=1>0 to 105</td></tr><tr><td rowspan=1 colspan=1>Diffusion coefficient (D)</td><td rowspan=1 colspan=1>0.5</td></tr><tr><td rowspan=1 colspan=1>Velocity of a molecular motor on a microtubule</td><td rowspan=1 colspan=1>1 µm/s</td></tr><tr><td rowspan=1 colspan=1>Expected travel distance ofa molecular motor on a microtubule</td><td rowspan=1 colspan=1>4µm</td></tr><tr><td rowspan=1 colspan=1># of duplicated molecules per message/acknowledgement</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>Max number of molecule retransmissions</td><td rowspan=1 colspan=1>5</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Diffusivetransport</td><td rowspan=1 colspan=1>Noise levLow0, 0.1, 0.2, .. 1.01</td><td rowspan=1 colspan=1>el (%)High.1, 1.2, ... 3.0</td><td rowspan=1 colspan=1>Active</td><td rowspan=1 colspan=1>Noise levLow0, 0.1, 0.2, ... 1.0</td><td rowspan=1 colspan=1>el (%)High1.1, 1.2, ... 3</td><td rowspan=1 colspan=1>Hybrid</td><td rowspan=1 colspan=1>Noise levLow0, 0.1, 0.2, ... 1.01</td><td rowspan=1 colspan=1>el (%)High.1, 1.2, ... 3</td></tr><tr><td rowspan=2 colspan=1>TXx-tan Dance   10Short2030(un)405060Long708090</td><td rowspan=1 colspan=1>T1</td><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>1020Short3040</td><td rowspan=1 colspan=1>T5</td><td rowspan=1 colspan=1>T6</td><td rowspan=1 colspan=1>10Short203040</td><td rowspan=1 colspan=1>T9</td><td rowspan=1 colspan=1>T10</td></tr><tr><td rowspan=1 colspan=1>T4</td><td rowspan=1 colspan=1>T3 Long</td><td rowspan=1 colspan=1>5060708090</td><td rowspan=1 colspan=1>T8</td><td rowspan=1 colspan=1>T7 Long</td><td rowspan=1 colspan=1>5060708090</td><td rowspan=1 colspan=1>T12</td><td rowspan=1 colspan=1>T11</td></tr></table>

Fig. 3. RTT Estimation Tasks (T1 to T12)

During training, each CL method has no access to training data from prior tasks. At test time, it performs RTT estimation for the current and all prior tasks. It does not know which task a given test input belongs to. For example, when a CL method is trained on the second task (T2), it cannot use training data from T1 but can use only T2 data. The method is then tested by estimating RTT for both T1 and T2.

Estimation tasks are arranged into T evaluation scenarios, where T is the total number of tasks (T = 12 in this paper). The t-th scenario evaluates each CL method after it has been sequentially trained on t tasks. The method is evaluated using the mean squared error between the estimated and true RTT values $( \mathcal { L } _ { \mathrm { m s e } } ) .$ . After all scenarios are completed, each method produces the following matrix of test errors.

$$
\left[ \begin{array} { c c c c c c c } { \mathcal { L } _ { \mathrm { m s } 1 } } & & & & & & \\ { \mathcal { L } _ { \mathrm { m s } 2 } } & { \mathcal { L } _ { \mathrm { m s } 2 } } & & & & \\ { \mathcal { L } _ { \mathrm { m s } 1 } } & { \mathcal { L } _ { \mathrm { m s } 2 } } & { \mathcal { L } _ { \mathrm { m s } 3 } } & & & \\ { \vdots } & { \vdots } & { \vdots } & { \ddots } & & \\ { \mathcal { L } _ { \mathrm { m s } 1 } } & { \mathcal { L } _ { \mathrm { m s } 2 } } & { \mathcal { L } _ { \mathrm { m s } 3 } } & { \cdots } & { \mathcal { L } _ { \mathrm { m s } 2 T - 1 } } & & \\ { \mathcal { L } _ { \mathrm { m s } 1 } } & { \mathcal { L } _ { \mathrm { m s } 2 } } & { \mathcal { L } _ { \mathrm { m s } 3 } } & { \cdots } & { \mathcal { L } _ { \mathrm { m s } T - 1 } } & { \mathcal { L } _ { \mathrm { m s } 2 T - 1 } } \end{array} \right]
$$

The rows of the matrix represent evaluation scenarios, and the columns represent tasks. For example, $\mathcal { L } _ { \mathrm { m s e 3 } } T$ denotes the test error on the third task after training on all $T$ tasks.

Based on the matrix, this paper uses the following evaluation metrics.

$$
\mathcal { L } _ { \mathrm { p l a s t i c i t y } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } _ { \mathrm { m s e } t }
$$

$$
{ \mathcal { L } } _ { { \mathrm { s t a b i l i t y } } } = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { { \mathcal { L } } _ { \mathrm { m s e } t } } ^ { T }
$$

$$
\mathcal { F } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \mathcal { L } _ { \mathrm { m s e } t } ^ { \mathrm { ~ ~ } T } - \mathcal { L } _ { \mathrm { m s e } t } \right)
$$

$$
\mathcal { F } _ { \mathrm { r } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { \operatorname* { m a x } ( 0 , \mathcal { L } _ { \mathrm { m s e } _ { t } } T - \mathcal { L } _ { \mathrm { m s e } _ { t } } t ) } { \mathcal { L } _ { \mathrm { m s e } _ { t } } }
$$

Lower $\mathcal { L } _ { \mathrm { p l a s t i c i t y } }$ and $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ values indicate better estimation accuracy. $\mathcal { L } _ { \mathrm { p l a s t i c i t y } }$ is the average of the diagonal test errors in the above matrix. Each diagonal entry $( \mathcal { L } _ { \mathrm { m s e } _ { t } ^ { t } } )$ represents the test error on task t immediately after the model first learns that task. This paper uses $\mathcal { L } _ { \mathrm { p l a s t i c i t y } }$ to measure learning plasticity (i.e., the ability to rapidly adapt to new tasks).

$\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ is the average test error across all tasks in the last $( T \mathrm { \cdot } \mathrm { t h } )$ scenario. This paper uses $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ as a metric of memory stability (i.e., the ability to preserve past knowledge) because it measures how well the model retains performance on all learned tasks after continual training.

$\mathcal { F }$ measures the average change in task-wise test error between the time each task is first learned and the end of continual training. Higher $\mathcal { F }$ indicates greater forgetting, whereas lower $\mathcal { F }$ indicates better retention of past knowledge.

$\mathcal { F } _ { \mathrm { r } }$ measures the average relative increase in test error from the time a task is first learned to the end of training. $\mathcal { F } _ { \mathrm { r } } ~ = ~ 1$ means that the final error is twice the initial error (100% relative increase). When $\mathcal { F } _ { \mathrm { r } } = 0 .$ , no forgetting occurs.

## B. Results

Table III summarizes the RTT estimation performance of AMR and the other methods in terms of $\mathcal { L } _ { \mathrm { p l a s t i c i t y } } , \mathcal { L } _ { \mathrm { s t a b i l i t y } }$ and F. Each result is averaged over 50 independent test runs. Boldface indicates the best result among the four methods. The \* and ! symbols indicate that AMR’s result is statistically different from the compared method under a t-test and Mann– Whitney U-test at $p < 0 . 0 5$ , respectively. Standard deviations are reported with the ± symbol.

TABLE III  
RTT ESTIMATION PERFORMANCE
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1> $\underline { { \mathcal { L } _ { \mathrm { p l a s t i c i t y } } } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { L } _ { \mathrm { s t a b i l i t y } } } }$ </td><td rowspan=1 colspan=1>F</td></tr><tr><td rowspan=1 colspan=1>AMRACLTA2ERCLeaR</td><td rowspan=1 colspan=1>0.1069±0.01120.1073±0.01160.1057±0.0127 $0 . 1 0 7 5 { \scriptstyle \pm 0 . 0 1 1 5 }$ </td><td rowspan=1 colspan=1>0.1399±0.01010.1457±0.0157 *!0.1463±0.0158 *!0.1454±0.0131 *！</td><td rowspan=1 colspan=1>0.0330±0.01380.0384±0.01820.0406±0.0173 *!0.0379±0.0152</td></tr></table>

As shown in Table III, AMR achieves the lowest $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ among the four methods, along with the lowest standard deviation. AMR also statistically outperforms the other three methods in $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ under both the t-test and Mann–Whitney $U .$ -test $( p < 0 . 0 5 )$ . For ${ \mathcal { F } } ,$ AMR achieves the lowest value and the lowest standard deviation. These results indicate that AMR better preserves knowledge from previously learned tasks and maintains estimation accuracy across tasks.

In $\mathcal { L } _ { \mathrm { p l a s t i c i t y } }$ , A2ER achieves the lowest value, while AMR yields the second-lowest value and the lowest standard deviation. The difference between AMR and A2ER is small, suggesting that AMR maintains competitive plasticity while substantially improving stability. This indicates that AMR does not sacrifice its ability to adapt to newly encountered tasks in order to reduce forgetting.

![](images/606f414e48493a7183938a5edf243b8b2034482565dd850e7e0375a529a1ff25.jpg)  
Fig. 4. Change of forgetting ratio as the number of sequential tasks varies

Fig. 4 shows how the forgetting ratio $( \mathcal { F } _ { \mathrm { r } } )$ changes as the number of sequential tasks increases. Overall, AMR attains the lowest average $\mathcal { F } _ { \mathrm { r } }$ at 0.1384, while ACLT, A2ER and CLeaR yield higher averages of 0.1518, 0.1596 and 0.1662, respectively. When the methods are ranked by $\mathcal { F } _ { \mathrm { r } }$ in each scenario, AMR achieves the best average rank of 1.9, while ACLT, A2ER and CLeaR have rank averages of 2.1, 3.0 and 3.0, respectively.

Table IV shows the computational cost of training for AMR and the other methods. The total training time refers to the time required to train on all tasks in the final evaluation scenario. CLeaR achieves the lowest training time, followed by A2ER and AMR. Although AMR requires slightly more training time than CLeaR and A2ER, its cost remains comparable to those methods and is lower than that of ACLT.

Fig. 5 illustrates the tradeoff between estimation performance and training cost. Circular dots represent pairs of performance $( { \mathcal { L } } _ { \mathrm { s t a b i l i t y } } )$ and training time over 50 test runs, and their mean is plotted as a star. Diamonds represent the mean pairs of $\mathcal { L } _ { \mathrm { p l a s t i c i t y } }$ and training time. AMR achieves the lowest mean $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ while maintaining a moderate training cost, indicating that it effectively maintains estimation performance on all previously seen tasks, not only on the most recent one, without imposing a substantial computational overhead.

TABLE IV  
COMPUTATIONAL COST FOR TRAINING
<table><tr><td>Method</td><td>Total Training Time (sec)</td><td>Per-task Time (sec)</td></tr><tr><td>AMR</td><td> $8 . 7 6 \pm 1 . 5 0$ </td><td>0.73</td></tr><tr><td>ACLT</td><td> $1 1 . 6 4 \pm 1 . 0 2$ </td><td>0.97</td></tr><tr><td>A2ER</td><td> $8 . 2 0 \pm 1 . 0 7$ </td><td>0.68</td></tr><tr><td>CLeaR</td><td> ${ \bf 7 . 3 9 \pm 0 . 8 6 }$ </td><td>0.62</td></tr></table>

Table V summarizes how the synthesis range parameter q (Algorithm 1) affects the RTT estimation performance of AMR. The results show that AMR is relatively robust to the choice of $q ,$ with similar $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ values across all tested settings. Among them, $q = 0 . 1$ achieves the lowest $\mathcal { L } _ { \mathrm { s t a b i l i t y } } ,$ while $q = 0$ achieves the lowest $\mathcal { L } _ { \mathrm { p l a s t i c i t y } }$ . This suggests that interpolation-only synthesis can slightly improve adaptation to newly encountered tasks. The lowest $\mathcal { F }$ is obtained at $q = 0 . 2$ indicating that a modest amount of extrapolation can further reduce forgetting.

![](images/d6742357062d052959cc6087a323e2ffcc77e4327a5196c35aaa86497b9ec631.jpg)  
Fig. 5. Tradeoff between Estimation Performance and Training Cost

TABLE V  
RTT ESTIMATION PERFORMANCE (AMR)
<table><tr><td>q</td><td> $\underline { { \mathcal { L } _ { \mathrm { p l a s t i c i t y } } } }$ </td><td> $\underline { { \mathcal { L } _ { \mathrm { s t a b i l i t y } } } }$ </td><td> $\scriptstyle { \overline { { \mathcal { F } } } }$ </td></tr><tr><td>0.0</td><td> $\mathbf { 0 . 1 0 5 3 \ : \pm { \ : 0 . 0 0 9 0 } }$ </td><td> $\overline { { 0 . 1 4 0 8 \pm 0 . 0 1 2 0 } }$ </td><td> $0 . 0 3 5 5 \pm 0 . 0 1 5 5$ </td></tr><tr><td>0.05</td><td> $0 . 1 0 7 9 \pm 0 . 0 1 0 5$ </td><td> $0 . 1 4 1 5 \pm { \bf 0 . 0 0 8 0 }$ </td><td> $0 . 0 3 3 5 \pm { \bf 0 . 0 1 2 3 }$ </td></tr><tr><td>0.1</td><td> $0 . 1 0 6 9 \pm 0 . 0 1 1 3 \vert 0 . 1 3 9 9 \pm 0 . 0 1 0 1$ </td><td></td><td> $0 . 0 3 3 0 \pm 0 . 0 1 3 8$ </td></tr><tr><td>0.2</td><td> $0 . 1 0 9 7 \pm 0 . 0 1 0 6 \lvert 0 . 1 4 0 5 \pm 0 . 0 1 0 5$ </td><td></td><td> $\mathbf { 0 . 0 3 0 8 \ : \pm { 0 . 0 1 2 8 } }$ </td></tr><tr><td>0.3</td><td> $0 . 1 0 7 6 \pm 0 . 0 1 1 5$ </td><td> $0 . 1 4 1 2 \pm 0 . 0 1 2 0$ </td><td> $0 . 0 3 3 6 \pm 0 . 0 1 6 2$ </td></tr><tr><td>1.5 2.5</td><td> $0 . 1 0 6 0 \pm 0 . 0 1 2 4$   $| 0 . 1 0 8 2 \pm 0 . 0 1 1 2$ </td><td> $0 . 1 4 1 6 \pm 0 . 0 1 2 6$   $0 . 1 4 1 4 \pm 0 . 0 1 2 5$ </td><td> $0 . 0 3 5 6 \pm 0 . 0 1 5 0$   $| 0 . 0 3 3 2 \pm 0 . 0 1 3 7$ </td></tr></table>

Fig. 6 shows how $\mathcal { F } _ { \mathrm { r } }$ changes across sequential tasks for different q values. The curves follow broadly similar trends, indicating that AMR’s forgetting behavior is not highly sensitive to $q .$ However, moderate values such as $q = 0 . 1$ and $q \ = \ 0 . 2$ tend to provide favorable forgetting mitigation, whereas very large extrapolation ranges do not yield clear additional benefits. This suggests that limited extrapolation is useful, but overly broad extrapolation may introduce synthetic samples that are less representative of previously learned tasks.

Table VI shows the training cost of AMR under different $q$ values. The total training times are broadly comparable across settings, ranging from 8.76 to 10.29 seconds. $q = 0 . 1$ achieves the lowest total training time, followed closely by $q = 0 . 2$ . This indicates that the choice of $q$ does not introduce substantial computational overhead, while moderate values of $q$ can improve stability or forgetting performance.

Fig. 7 illustrates the tradeoff between estimation performance and training cost for different q values. The $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ values are close across all cases. Overall, $q = 0 . 1$ provides the best cost-performance tradeoff: it achieves the lowest $\mathcal { L } _ { \mathrm { s t a b i l i t y } }$ and the lowest training time, while maintaining a low ${ \mathcal F } .$ Therefore, this paper uses $q \ = \ 0 . 1$ as the default AMR configuration (Table I).

![](images/103793ec38a2ab6094eb6ea16d4ec060a245698b0ceeb167f845e0ba1cfec319.jpg)  
Fig. 6. Change in forgetting ratio as the number of tasks varies (AMR)

TABLE VI  
COMPUTATIONAL COST FOR TRAINING (AMR)
<table><tr><td> $q$ </td><td>Total Training Time (sec)</td><td>Per-task Time (sec)</td></tr><tr><td>0.0</td><td> $9 . 4 6 \pm 1 . 7 3$ </td><td>0.79</td></tr><tr><td>0.05</td><td> $1 0 . 2 9 \pm 1 . 9 9$ </td><td>0.86</td></tr><tr><td>0.1</td><td> ${ \bf 8 . 7 6 \pm 1 . 5 0 }$ </td><td>0.73</td></tr><tr><td>0.2</td><td> $8 . 8 0 \pm 1 . 5 1$ </td><td>0.73</td></tr><tr><td>0.3</td><td> $9 . 1 0 \pm 1 . 7 1$ </td><td>0.76</td></tr><tr><td>1.5</td><td> $1 0 . 0 4 \pm 1 . 1 9$ </td><td>0.84</td></tr><tr><td>2.5</td><td> $9 . 1 4 \pm 1 . 6 9$ </td><td>0.76</td></tr></table>

## VI. CONCLUSION

This paper proposes and evaluates Affine Memory $\mathrm { R e \mathrm { - } }$ play (AMR), a continual learning (CL) method for RTT estimation in in-sequence molecular communication. AMR synthesizes replay samples from past simulation results through interpolation and extrapolation, stores both past and synthetic samples in bounded replay buffers, and uses replay losses to regularize model updates. Experimental results show that AMR significantly outperforms existing CL methods in maintaining overall estimation accuracy across sequentially learned estimation tasks. It effectively balances memory stability and learning plasticity under moderate training cost. Future work will extend AMR to more advanced molecular communication settings, including mobile bio-nanomachines, time-varying environments and larger task sequences. It will also investigate adaptive strategies for selecting the synthesis range parameter q and broader replay-sample generation methods to further improve the balance between stability, plasticity and computational cost.

## REFERENCES

[1] T. Suda, M. Moore, T. Nakano, R. Egashira, and A. Enomoto, “Exploratory research on molecular communication between nanomachines,” in Proc. ACM Genetic and Evol. Computat. Conference, 2005.

![](images/ffd44c980d7fad66b163027b6e06bb5e94443071df72fe05a4dae983952d8c04.jpg)  
Fig. 7. Tradeoff between Estimation Performance and Training Cost (AMR)

[2] T. Nakano, A. Eckford, and T. Haraguchi, Molecular Communication. Cambridge University Press, 2013.

[3] J. Suzuki, T. Nakano, and M. Moore, Modeling, Methodologies and Tools for Molecular and Nano-scale Communications. Spinger, 2017.

[4] Y. Chahibi, “Molecular communication for drug delivery systems: A survey,” Nano Communication Networks, vol. 11, pp. 90–102, 2017.

[5] D. Bi, A. Almpanis, A. Noel et al., “A survey of molecular communication in cell biology: Establishing a new hierarchy for interdisciplinary applications,” IEEE Commun. Surveys Tuts, vol. 23, no. 3, 2021.

[6] H. Xiao, K. Dokaj, and O. B. Akan, “What really is “molecule” in molecular communications? the quest for physics of particle-based information carriers,” IEEE Trans Mol Biol Multi-Scale Commun, vol. 10, no. 1, pp. 43–74, 2024.

[7] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” vol. 24, pp. 109–165, 1989.

[8] H. O. Burton and D. D. Sullivan, “Errors and error control,” Proc. of the IEE, vol. 60, no. 11, 1972.

[9] T. Nakano, Y. Okaie, and A. V. Vasilakos, “Transmission rate control for molecular communication among biological nanomachines,” IEEE J. Sel. Area Comm., vol. 31, no. 12, pp. 835–846, 2013.

[10] L. Felicetti, M. Femminella, G. Reali, T. Nakano, and A. V. Vasilakos, “TCP-like molecular communications,” IEEE J. Sel. Area Comm., vol. 32, no. 12, pp. 2354–2367, 2014.

[11] X. Wang, M. D. Higgins, and M. S. Leeson, “Simulating the performance of SW-ARQ schemes within molecular communications,” Simulation Modelling Practice and Theory, vol. 42, 2014.

[12] C. Bai and M. S. Leeson et al., “Performance of SW-ARQ in bacterial quorum communications,” Nano Commun. Netw., vol. 6(1), 2015.

[13] S. P. Singh, R. Rai, S. Awasthi, D. K. Singh, and M. Lakshmanan, “VLSI implementation of error correction codes for molecular communication,” Wirel. Personal. Commun., vol. 130, pp. 2697–2713, 2023.

[14] J. S. Mitzman, B. Morgan, T. M. Soro, J. Suzuki, and T. Nakano, “A feedback-based molecular communication protocol for noisy intrabody environments,” in Proc. of IEEE Int’l Conference on E-health Networking Applications and Services, 2015.

[15] Z. Cheng, H. Liu, M. Chen et al., “Machine learning-based detection time estimation for molecular communication,” in Proc. Int’l Conf. on Bio-inspired Information and Communications Technologies, 2024.

[16] Z. Cheng, H. Liu, Z. Xu, J. Li, and K. Chi, “Deep learning-based estimation of emission time and arrival time in diffusive multi-receiver molecular communication,” IEEE Trans Mol Biol Multi-Scale Commun, vol. 11, no. 2, pp. 257–268, 2025.

[17] C. Lee, H. B. Yilmaz, C.-B. Chae, N. Farsad, and A. Goldsmith, “Machine learning based channel modeling for molecular mimo communications,” in Proc. IEEE Int’l Workshop on Signal Processing Advances in Wireless Communications, 2017.

[18] H. B. Yilmaz, C. Lee, Y. J. Cho, and C.-B. Chae, “A machine learning approach to model the received signal in molecular communications,” in Proc. IEEE Int’l Black Sea Conference on Communications and Networking, 2017.

[19] D. Casaleiro, N. M. B. Souto, and J. C. Silva, “Synchronization and detection in molecular communication using a deep-learning-based approach,” IEEE Access, vol. 12, pp. 192 539–192 553, 2024.

[20] Z. Cheng, H. Liu, J. Zheng, W. Gong, and K. Chi, “Localizing and tracking the transmitter bionanosensor in mobile molecular communication by deep learning,” IEEE Sensors Journal, vol. 25, no. 7, pp. 10 583–10 593, 2025.

[21] O. Kara, G. Yaylali, A. Pusane, and T. Tugcu, “Molecular index modulation using convolutional neural networks,” Nano Communication Networks, vol. 34, p. 100420, 2022.

[22] S. Setia, J. Suzuki, and T. Nakano, “Continual learning of feedbackbased molecular communication,” in Proc. of Int’l Conference on Bioinspired Information and Communications Technologies, 2025.

[23] M. De Lange, R. Aljundi, M. Masana et al., “A continual learning survey: Defying forgetting in classification tasks,” IEEE Trans Pattern Anal Mach Intell, vol. 44, no. 7, 2022.

[24] L. Wang, X. Zhang, H. Su et al., “A comprehensive survey of continual learning: Theory, method and application,” IEEE T. Pattern Anal Mach Intell, vol. 46, no. 8, 2024.

[25] M. E. Ortiz and D. Endy, “Engineered cell-cell communication via DNA messaging,” J. Biol. Eng., vol. 6, no. 16, 2012.

[26] R. D. Vale, T. Funatsu, D. W. Pierce, L. Romberg, Y. Harada, and T. Yanagida, “Direct observation of single kinesin molecules moving along microtubules,” Nature, vol. 380, 1996.

[27] T. Nakano, M. Moore, F. Wei, A. V. Vasilakos, and J. W. Shuai, “Molecular communication and networking: opportunities and challenges,” IEEE Trans. NanoBiosci., vol. 11, no. 2, pp. 135–148, 2012.

[28] M. Riemer, I. Cases, R. Ajemian et al., “Learning to learn without forgetting by maximizing transfer and minimizing interference,” in Proc. ICLR, 2019.

[29] Y. He and B. Sick, “CLeaR: An adaptive continual learning framework for regression tasks,” AI Perspectives, vol. 3, no. 1, 2021.

[30] M. Wu, X. Zhou, S. Li et al., “An adaptive continual learning method for nonstationary industrial time series prediction,” IEEE Trans Industr Inform, vol. 21, no. 2, 2025.

[31] T. Kobayashi, “Improvements to dark experience replay and reservoir sampling for better balance between consolidation and plasticity,” Front Artif Intell, vol. 9, 2026.