# K-prop: Deep Online Learning by Backpropagating Temporal Kernels

Bingkun Liu<sup>∗</sup>, Paul Haider, Federico Benitez, Mihai A. Petrovici, Walter Senn

Department of Physiology University of Bern Bern, Switzerland

## Abstract

Self-recurrence represents an important component for temporal processing in neural networks, both artificial and biological, where it appears naturally in the form of leaky integration. However, due to the complex temporal dependencies of the gradient, an efficient and effective method of online training is still lacking. Here we propose K-prop, a novel online learning method for leaky neural networks. It dramatically reduces the computational and memory cost of real time recurrent learning (RTRL) by propagating kernel gains. K-prop also outperforms existing RTRL approximations and achieves performance comparable to backpropagation through time in sequence learning, classification, and reinforcement learning tasks. We further analyze how its sole approximation affects gradient quality and identify conditions under which K-prop remains accurate and effective.

## 1 Introduction

Temporal processing is fundamental to both biological and artificial intelligence, requiring systems to transform continuous inputs into structured internal representations. Effective training for temporal dynamics relies on accurate credit assignment over time, commonly addressed by backpropagation through time (BPTT, Werbos [2002]). However, BPTT requires storing and replaying full trajectories, limiting its applicability to online settings. In contrast, many natural and real-world systems operate continuously, where learning must occur in real time without access to stored past states, motivating the need for efficient online learning mechanisms. Prior research [Irie et al., 2024] suggests that online learning can also improve sample efficiency in reinforcement learning settings.

Artificial neural networks typically employ instantaneous neurons, whose activations depend only on the current input. In contrast, leaky (integrating) neurons evolve according to continuous dynamics of the type τu˙ = −u + W r, where the membrane potential u decays with time constant τ while integrating presynaptic input r through weights W. These dynamics emulate biological membrane potentials and have therefore been widely adopted in computational neuroscience [Dayan and Abbott, 2005]. Beyond biological inspiration, leaky neurons offer an intrinsic mechanism for temporal processing: each neuron implements a low-pass filter, and networks of such units naturally construct rich temporal kernels in an online manner. Unlike Temporal Convolutional Networks (TCN, Bai et al. [2018]), which require access to full input sequences, leaky networks have the potential to process streaming data continuously while maintaining a distributed, multi-timescale memory, a power that has recently been demonstrated in models [Spieler et al., 2024, Ebato et al., 2024]. Moreover, their dynamics align naturally with analog hardware, where voltage-based circuits exhibit leaky integration [Stoliar et al., 2017, Yang et al., 2020].

Despite these advantages, online training of feedforward leaky networks remains fundamentally challenging. Due to temporal integration, activity at each layer influences future downstream responses, making the gradient depend jointly on spatial pathways and temporal dynamics, analogous to recurrent networks. In principle, exact gradients can be computed by BPTT or real-time recurrent learning (RTRL, Williams and Zipser [1989]). RTRL, while online, incurs prohibitive computational complexity of order $\mathcal { O } ( n ^ { 4 } )$ , rendering it impractical beyond very small networks. Existing RTRL approximations either reduce complexity at the expense of gradient quality or fail to scale to deep architectures. As a result, investigating online learning in large and deep leaky networks remains challenging.

In this work, we introduce Kernel propagation (K-prop), a novel learning algorithm for feedforward leaky networks that achieves efficient, online, and accurate gradient approximation. The central idea is to express gradient contributions as a decomposition over exponential temporal kernels, and to propagate only their gain coefficients backward through the network. This transforms the combinatorial explosion of possible temporal pathways into a tractable, layer-wise computation analogous to standard backpropagation.

K-prop offers the following key contributions:

Efficiency and effectiveness: K-prop significantly reduces the quartic complexity of RTRL to near-quadratic scaling, while maintaining a close approximation to the exact gradient. It achieves best-in-class task performance, close to the exact RTRL/BPTT solutions, in deep architectures. Continuous formulation: Derived directly in continuous time, K-prop is robust to arbitrarily small time steps and naturally suited for analog systems and for dealing with continuous or sparse data. Layer-wise locality: K-prop remains layer-wise local during training, facilitating practical implementation without requiring global information pathways.

## 2 Related Work

A large body of work has focused on approximations to RTRL. Truncated BPTT [Williams and Zipser, 2013] reduces memory by limiting the temporal horizon, but introduces bias toward shortterm dependencies. Methods that are tailored to standard recurrent networks with dense global recurrence do not align well with leaky neurons, whose dominant temporal effect arises from strong self-recurrence (diagonal dynamics). For example, low-rank approximations such as unbiased online recurrent optimization (UORO, Tallec and Ollivier [2018]) fail to capture the inherently high-rank diagonal structure, while Kronecker-based methods [Mujika et al., 2018, Benzing et al., 2019] do not exploit the naturally separable temporal kernels. Moreover, these approaches are formulated in discrete time and are not directly suited to continuous-time or analog dynamics.

Alternatively, methods such as random feedback local online learning (RFLO, Murray [2019]) and e-prop [Bellec et al., 2020] are specifically designed for leaky neurons and emphasize biological plausibility. These methods rely on local eligibility traces but discard inter-neuron dependencies, which prevents effective credit assignment in deep layers. Zucchet et al. [2023] derived online learning for complex-valued leaky-like memory, and is also limited to a single memory layer. Online spatio-temporal learning (OSTL, Bohnstingl et al. [2022]), partially addresses this by propagating instantaneous errors to deep layers whose temporal dependencies are still missing. Following an energy-based approach, generalized latent equilibrium (GLE, Ellenberger et al. [2025]) assumes a sufficiently smooth error signal ϵ, so that the prospective error ϵ˘ = ϵ + τϵ˙ can be used to estimate temporal gradients. Beyond previous online methods, K-prop preserves the temporal structure of the deep layer gradient by deploying a novel decompositions of both temporal kernels and spatial paths.

## 3 Theory

## 3.1 K-prop: Propagating kernel gains

We consider a feedforward leaky neural network with nonlinear activation $r = \rho ( u )$ . From the output loss function C(t), gradients propagate backward to each hidden neuron through the downstream network along all possible directed paths (as shown in Fig. 1a). The gradient of each path constains a temporal term accounting for its temporal contribution. For a postsynaptic neuron i in the last leaky layer L, this temporal term is represented by a vector of synaptic eligibility traces $z _ { i } = \rho _ { i } ^ { \prime } \overline { { r } } _ { [ L - 1 ] } ^ { \tau _ { i } } .$

![](images/7406288eeb3c174673e22ce947f3662940199194fceb52130854eba56478015b.jpg)  
Figure 1: Kernel propagation. (a) Calculating gradients involves hierarchical filters along paths of neuron connections. (b) Illustration of kernel decomposition. The complicated temporal structure in the true gradient can be approximated by a sum over $\zeta$ weighted by τ-dependent coefficients $\kappa$ Every ζ (solid) is weighted by corresponding K as a component (dashed in the same color). The sum of all components aligns well with the exact term computed by RTRL.

where $\overline { { { \mathbf { x } } } } ^ { \tau } = { \boldsymbol { \kappa } } ^ { \tau }$ ∗ x denotes a low-pass filtering of the vector-valued temporal variable $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ with the exponential kernel $\begin{array} { r } { \kappa ^ { \tau } = \Theta ( t ) { \frac { 1 } { \tau } } \mathrm { e x p } ( - { \frac { t } { \tau } } ) } \end{array}$ , where Θ is the Heaviside function. Moving one layer upstream to neuron $j$ introduces an additional filtering $\rho _ { i } ^ { \prime } ( \kappa ^ { \tau _ { i } } * z _ { j } )$ , and an arbitrary neuron ⋆ in deeper layers generates more iterative hierarchical filters $\rho _ { i } ^ { \prime } \big ( \kappa ^ { \tau _ { i } } * \rho _ { i } ^ { \prime } \big ( \kappa ^ { \tau _ { j } } * \ldots * z _ { \star } \big ) \big )$ ). Therefore, each path induces a hierarchical temporal kernel characterized by the leaky dynamics of all neurons along the path, and the exact gradient requires summing over all combinations of such path-dependent kernels.

K-prop simplifies this involved computation in three steps.

## Step 1: Separating nonlinearity from filtering

K-prop approximates $\rho ^ { \prime }$ by a surrogate $\hat { \rho } ^ { \prime } \left( = \overline { { \rho _ { \star } ^ { \prime } } } ^ { \tau _ { \star } } \right.$ as default), such that it can be moved outside of the filter: $\overline { { \rho ^ { \prime } x } } ^ { \tau } \approx \hat { \rho } ^ { \prime } \overline { { x } } ^ { \tau }$ . We can therefore approximate a hierarchical filter by a nested filter: $\rho _ { i } ^ { \prime } ( \kappa ^ { \tau _ { i } } * \rho _ { j } ^ { \prime } ( \kappa ^ { \tau _ { j } } * \rho _ { k } ^ { \prime } ( \kappa ^ { \tau _ { k } } * . . . ) ) ) \approx ( \rho _ { i } ^ { \prime } \hat { \rho } _ { j } ^ { \prime } \hat { \rho } _ { k } ^ { \prime } . . . ) ( \kappa ^ { \tau _ { i } } * \kappa ^ { \tau _ { j } } * \kappa ^ { \tau _ { k } } * . . . )$ . The assumption that $\rho ^ { \prime }$ varies slowly relative to the filtering time scale τ is the only approximation required by K-prop.

## Step 2: Collapsing nested filters into basis kernels

Lemma 1: Nested exponential filters can be rewritten as a linear combination of single filters. Υ is a set of N mutually distinct τ. A proof is given in Section A.

$$
\kappa ^ { \Upsilon } ( x ) = \kappa ^ { \tau _ { 0 } } * \kappa ^ { \tau _ { 1 } } * . . . \kappa ^ { \tau _ { N } } * x\tag{1}
$$

$$
= \sum _ { \tau _ { i } \in \Upsilon } c _ { i } \big ( \kappa ^ { \tau _ { i } } * x \big ) \mathrm { w i t h } c _ { i } = \prod _ { \tau _ { j } \in \Upsilon , j \neq i } \frac { \tau _ { i } } { \tau _ { i } - \tau _ { j } } .\tag{2}
$$

In this work, we treat identical τ across layers as slightly perturbed, which yields an equivalent temporal horizon while avoiding singularities in Eq. (2). Consequently, each τ appears once in a path. Applying the lemma, each path-specific nested kernel is converted into a linear combination of basis kernels. Together with Step 1, the path gradient can then be approximated as $\left. \frac { \partial \mathcal { C } ( t ) } { \partial W _ { \star } } \right| _ { p } \approx$ $\begin{array} { r } { w _ { p } \sum _ { i \in p \backslash \star } c _ { i } \big ( \kappa ^ { \tau _ { i } } * z _ { \star } \big ) } \end{array}$ , which sums over all neurons i in the path $p$ except the initial ⋆. The term $w _ { p }$ is the path weight, a product of all synaptic weights, nonlinear surrogates $\hat { \rho } ^ { \prime } .$ , and the output error along path p (see Eq. (37) in Supplement).

## Step 3: Propagating kernel gains through layers

Despite the reduction above, explicitly summing over paths remains intractable given: (1) exponential growth in the number of paths; (2) dependence on detailed network topology; (3) dependence on global downstream variables (weights, activities, errors). The key observation is that the number of distinct time constants is much smaller than the number of paths. Let $\mathcal { P }$ denote the set of all downstream paths and $\Upsilon _ { \mathrm { d o w n } }$ represent the set of all distinct time constants in downstream neurons. We can rearrange the summation over paths and group coefficients by their shared time constant, $\begin{array} { r } { \sum _ { p \in \mathcal { P } _ { \star } } w _ { p } \sum _ { i \in p } c _ { i } \left( \kappa ^ { \tau _ { i } } * z _ { \star } \right) = \sum _ { \tau \in \Upsilon _ { \mathrm { \star } \sim \tau } ^ { \star } } \mathcal { K } _ { \star } ^ { \tau } \left( \kappa ^ { \overline { { \tau } } } * \check { z _ { \star } } \right) } \end{array}$ , where $\mathcal { K } _ { \star } ^ { \tau }$ is a kernel gain that aggregates all path contributions associated with time constant τ. This transformation converts the sum over all combinations ofpaths and their neurons to a sum over a small set ofkernels.

![](images/ea970b3b3440712d2f8e7cd51c572eb289370f1a2498f7bd186e7f81854111db.jpg)  
Figure 2: Kernel propagation operation. Example propagation from layer l with 3 neurons to layer $l \bar { - } 1$ with 4 neurons. The gray box works as a bracket.

Another central insight of K-prop is that $\kappa .$ which implicitly depends on all downstream paths, satisfies a layer-wise recursive relation, i.e., $\ = \mathcal { K } _ { [ l - 1 ] } ^ { \tau }$ can be derived from $\mathcal { K } _ { [ l ] } ^ { \tau }$ , in a way analogous to backpropagation. For a compact notation, we define $\Upsilon _ { \mathrm { n e t } }$ as the set of all distinct τ in the network, and define τ-related transformation factors at neuron $j \colon$

$$
\begin{array} { r } { T _ { j } ^ { \tau } = \left\{ \begin{array} { l l } { 1 } & { \tau _ { j } = \tau } \\ { 0 } & { \tau \ \mathrm { i n ~ t h e ~ s a m e ~ l a y e r ~ a s ~ n e u r o n ~ } j \mathrm { ~ } \mathrm { a n d ~ } \tau \neq \tau _ { j } } \\ { \frac { \tau } { \tau - \tau _ { j } } } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{3}
$$

This definition for $\tau$ is based on fully connected feedforward networks, and is further elaborated and extended in Section A. With $n _ { \tau } = | \Upsilon _ { \mathrm { n e t } } |$ , let $\pmb { \mathcal { K } } _ { [ l ] } \in \mathbb { R } ^ { n _ { l } \times n _ { \tau } }$ denote the kernel gain at layer l, $\mathcal { T } _ { [ l ] } \in \mathbb { R } ^ { n _ { l } \times n _ { \tau } }$ the corresponding transformation matrix, and $\hat { \pmb { \rho } } _ { [ l ] } ^ { \prime } \in \mathbb { R } ^ { n _ { l } }$ <sup>l</sup>. The external error captured by the loss enters from the output layer as the gain $\begin{array} { r } { \kappa _ { o } ^ { \tau } = \frac { \partial \mathcal { C } } { \partial r _ { o } } } \end{array}$ , and $\kappa$ propagates backward according to (Fig. 2)

$$
\begin{array} { r } { \mathcal { K } _ { [ l - 1 ] } ^ { \tau } = W _ { [ l ] } ^ { \top } ( \hat { \rho ^ { \prime } } _ { [ l ] } \odot \mathcal { T } _ { [ l ] } ^ { \tau } \odot \mathcal { K } _ { [ l ] } ^ { \tau } ) . } \end{array}\tag{4}
$$

We consider the eligibility matrix $\mathcal { Z } _ { [ l ] } \in \mathbb { R } ^ { n _ { l } \times n _ { l - 1 } }$ for layer l, with the eligibility vector $z _ { [ l ] , \cdot } ^ { \mathsf { T } }$ i appearing in the i’th row. With this we define deep eligibility traces $\zeta _ { [ l ] } ^ { \tau _ { d } } = \overline { { \mathcal { Z } _ { [ l ] } } } ^ { \tau _ { d } } \in \mathbb { R } ^ { n _ { l } \times n _ { l - 1 } }$ for all downstream time constants $\tau _ { d } \in \Upsilon _ { \mathrm { d o w n } } ^ { l }$ . Each synapse thus tracks a number of eligibility traces equal to the number of different neuronal time constants that appear in the downstream network. This can be done locally and online through simple leaky integrator dynamics in a matrix form:

$$
\tau _ { d } \dot { \zeta } _ { [ l ] } ^ { \tau _ { d } } = - { \zeta } _ { [ l ] } ^ { \tau _ { d } } + \mathcal { Z } _ { [ l ] } .\tag{5}
$$

Finally, the total gradient sums up the contribution from all kernel components

$$
\dot { W } _ { [ l ] } \propto \sum _ { \tau \in \Upsilon _ { \mathrm { d o w n } } ^ { l } } \mathrm { d i a g } ( \mathcal { K } _ { [ l ] } ^ { \tau } ) \zeta _ { [ l ] } ^ { \tau } .\tag{6}
$$

K-prop replaces explicit path enumeration by propagating kernel coefficients. Each neuron tracks local temporal features $\zeta ,$ while K carries error information across layers (Fig. 1b). This yields a fully online and continuous time learning rule that avoids combinatorial complexity while preserving the structure of the path-based gradient (see pseudocode in Algorithm 1). In order to enable this mechanism, global information is used at initialization to construct $\tau$ and $\Upsilon _ { \mathrm { d o w n } }$ . During training, all computations remain layer-wise local. Consequently, K-prop can be straightforward to implement by integrating feed-forward network techniques, treating $\kappa$ as the gradient in standard backpropagation. Normalization or dropout simply scales or drops out $\kappa ,$ and convolutional or residual architectures only change the matrix operation during propagation.

![](images/f979b52fa269cb51ea63c6e3426f34f3dd2ca4190761708c3a6e332f4c65df84.jpg)

![](images/f5134b5c972571ec30a2cba8225dc5da68662d37dd453ed3ffb347c8aef00780.jpg)

![](images/188be7428ef665879f670ec9d46185267acfbd192190044753db1c3ba76fe993.jpg)  
Figure 3: Gradient cosine dissimilarity between K-prop and exact (BPTT/RTRL) gradients in randomly initialized networks, measured with respect to each neuron given a sample of the MNIST1D dataset (see Section 4.2). Dissimilarity ranges from 0 to 2 with 0 indicating complete alignment, 1 orthogonality, and 2 complete anti-alignment. Error bars show median, ${ 1 } \mathrm { s t } ,$ and 3rd quartiles. (a) Gradient mismatch increases towards earlier layers, as an increasing number of $\rho ^ { \prime }$ need to be approximated as constant. The mismatch depends on the type of nonlinearity present in the network, and $\bar { \rho ^ { \prime } }$ works better than $\rho ^ { \prime }$ as surrogate $\hat { \rho ^ { \prime } }$ . (b) Scatter plots show the distribution of outliers. ReLU leads to more outlier neurons than sigmoid nonlinearities. (c) Gradient mismatch increases, then decreases for larger τ. All plots show sigmoid functions performing better than ReLU.

## 3.2 Quality of the gradient approximation

The accuracy of K-prop is governed by its sole approximation $\frac { \rho ^ { \prime } ( t ^ { \prime } < t ) } { \hat { \rho } , ( t ) } \approx 1$ , which assumes that $\rho ^ { \prime }$ varies slowly over the relevant timescales determined by the inputs. Deviations from this assumption introduce a mismatch between the true gradient and its K-prop estimate. Understanding how this mismatch depends on network and model properties is key to improving K-prop performance.

We quantify gradient quality using the cosine dissimilarity between K-prop and BPTT gradients:

$$
1 - \frac { \Delta W _ { \mathrm { K P } } \cdot \Delta W _ { \mathrm { B P } } } { \left\| \Delta W _ { \mathrm { K P } } \right\| \left\| \Delta W _ { \mathrm { B P } } \right\| } .
$$

Across configurations, the median dissimilarity remains low $( < 1 0 ^ { - 2 } )$ , indicating that most neurons receive well-aligned gradients.

Depth. As shown in Fig. 3a, gradient dissimilarity increases with distance from the output layer. The last layers are computed exactly, while upstream layers accumulate approximation errors through repeated propagation. This is consistent with the layer-wise structure of K-prop, where small local mismatches compound along depth.

Time constants. The approximation error is directly linked to the timescales of the input (and thus the network) variation. When all $\tau = 0$ , K-prop reduces to standard backpropagation and is exact. As τ increases, dissimilarity grows due to longer temporal integration windows. In the limit of very large τ , network dynamics become slower, reducing temporal variation in $\rho ^ { \prime }$ and improving the K-prop approximation. Note that this limit only serves as a cross-check for our intuition: while gains $\rho ^ { \prime }$ become more constant over time, networks also lose functionality as they become incapable of meaningfully processing incoming signals.

Nonlinearity. The choice of activation function determines the smoothness of $\rho ^ { \prime } .$ . Functions with slowly varying derivatives $( \mathrm { e . g . }$ ., logistic sigmoid) yield lower dissimilarity. Although tanh is an offset and scaled logistic function, its derivative has larger magnitude and varies more rapidly, leading to increased mismatch. ReLU presents a distinct regime: $\rho ^ { \prime }$ is piecewise constant but discontinuous at zero. As a result, gradients are either highly accurate or highly inaccurate, leading to low median error but significantly higher variance, as reflected by the outlier distribution.

Surrogate derivative. The design of $\hat { \rho ^ { \prime } }$ directly controls the approximation. For example, a lowpass-filtered version $\bar { \rho } ^ { \tau _ { \rho } } ( t )$ improves gradient alignment compared to using the instantaneous $\rho ^ { \prime } ( t )$ and the $\tau _ { \rho }$ used to filter $\rho ^ { \prime }$ shifts the gradient alignment tradeoff across network depth (Section C.2). Exploring surrogate replacements is a promising direction for further improving $\mathrm { K } .$ -prop.

![](images/9437789bc8b1327932e59177f29d23765e2f1dc36d89f805b96e879625f0cc0f.jpg)

![](images/74b8a166549dd64c12552e326f37c88c0b582932ab50c13f6174338447169025.jpg)

![](images/d3a7408a4db53b495dacc1a6a502d01769f2a5d5a17dd11ddc5f41c6007ebc60.jpg)  
Figure 4: N-sum task evaluated with various online learning methods. ProspE only considers prospective errors ϵ˘ in GLE [Ellenberger et al., 2025], without further augmentations. (a) Training loss over time. K-prop achieves substantially lower error than other approximation methods and closely matches RTRL. (b) Temporal response of hidden neurons to a step input before and after learning. K-prop develops more structured and contrastive temporal kernels. (c) Example input and corresponding outputs. Before learning, all models produce non-informative outputs. After learning, K-prop and RTRL closely align with the target sequence, while other methods deviate in both timing and amplitude.

Overall, these results show that gradient quality in K-prop is determined by the interaction between temporal dynamics and nonlinearity, and can be systematically improved through appropriate choices of network timescales, activation functions, and surrogate design.

## 4 Experiments

## 4.1 Continuous signal processing with memory

We first evaluate K-prop on a clearly defined non-episodic task, the N-sum problem with $N = 3 .$ . The model receives a continuous binary stream (e.g., [0, 1, 1, 1, 0, ...]) without truncation and must output the sum of the most recent three bits in binary form using two output units $( b _ { 2 } b _ { 1 } \in \{ 0 0 , 0 1 , 1 0 , 1 \bar { 1 } \} )$ . Each bit is presented for 5 ms, requiring the model to maintain a memory of at least $5 N = \mathrm { { \bar { 1 5 } } }$ ms. Note that the time units are, in principle, arbitrary, and what matters is the ratio between time constants and bit duration.

Despite its simple formulation, the task presents two key challenges: (1) the model is not informed of the temporal horizon N or the bit duration, and must infer the relevant memory timescale purely from delayed error signals; (2) the least significant output bit $b _ { 1 }$ effectively implements a parity function over recent inputs, which is a class of problems known to be difficult for gradient-based learning [Abbe and Boix-Adsera, 2022, Shalev-Shwartz et al., 2017].

The loss is defined as the mean squared error and is delivered 2 ms after the onset of each input bit (reaction latency). Gradients are computed online at every time step, accumulated into $\Delta \hat { W }$ , and applied periodically, after which the accumulator is reset (see weight update period in Section 5.3 and Supplement Table 3). This update scheme is shared across all methods.

Table 1: Test accuracy (%) of BPTT, OSTL, and K-Prop across the evaluated datasets. Results are reported as mean ± standard deviation.
<table><tr><td>Method</td><td>MNIST1d-Lk4</td><td>MNIST1d-Lk5</td><td>GSC</td></tr><tr><td>BPTT Werbos [2002]</td><td> $9 6 . 6 0 \pm 0 . 2 0$ </td><td> $9 7 . 6 9 \pm 0 . 2 1$ </td><td> $9 2 . 4 7 \pm 0 . 2 1$ </td></tr><tr><td>OSTL Bohnstingl et al. [2022]</td><td> $9 6 . 0 8 \pm 0 . 3 0$ </td><td> $9 7 . 0 3 \pm 0 . 1 6$ </td><td> $9 0 . 6 1 \pm 0 . 2 0$ </td></tr><tr><td>K-prop (Ours)</td><td> ${ \bf 9 6 . 5 3 \pm 0 . 5 4 }$ </td><td> ${ \bf 9 7 . 5 6 \pm 0 . 1 9 }$ </td><td> ${ \bf 9 1 . 5 2 \pm 0 . 1 0 }$ </td></tr></table>

![](images/9d293286cdaf3aef86f2a902b06d0c17bed8c0a14b2f97ee4d48ab3bb7fdd5fb.jpg)

![](images/4e933a7a511886cf436735669110b846f7b9e246e010ca186cc3b1733c2ef546.jpg)

![](images/b67b7efeba7c744e9a1883798608f429ea7d0c1d93c3f9d8ad8ad7223422a674.jpg)  
Figure 5: Sequential Classification Benchmark learning. (a) Mean, minimum, and maximum validation accuracy during training on GSC Warden [2018] 20-command task. K-prop outperforms OSTL Bohnstingl et al. [2022] in final accuracy, convergence speed, and training stability. (b) Depth ablation by freezing early layers shows decreasing accuracy with less trained layers. The accuracy loss is similar between K-prop and BPTT, indicating that K-prop provides effective gradients for deep layers. (c) Noise robustness evaluation of the Lk4 in MNIST1d (see Section 5.2 and B.6). K-prop maintains high accuracy under a broad range of noise levels $\sigma .$

To emphasize learning under constrained capacity, we use a small architecture (6-6-6-2) with logistic output units. In this setting, exact RTRL remains tractable, allowing for a controlled comparison where all methods share identical initialization and learning rates. K-prop achieves performance nearly indistinguishable from RTRL, while significantly outperforming other online approximation methods. The advantage becomes more pronounced on the more difficult parity component $b _ { 1 } .$ , where K-prop better matches the target in both timing and amplitude (Fig. 4). In N-sum, the correct output depends on integrating information over a precise temporal window (∼15 ms) while simultaneously performing a nonlinear transformation (parity-like mapping). This requires assigning credit not only locally in time, but also across combinations of temporal filters.

Beyond output accuracy, we analyze the temporal responses of hidden neurons (layer L3) in Fig. 4. During training, neurons develop diverse temporal kernels that extract task-relevant features. Compared to initialization, the learned responses to step inputs become both more expressive and temporally structured. For instance, neurons exhibiting sign changes around 10 ms, effectively compute differences between recent and earlier inputs, resembling temporal filtering operations. Such structured temporal responses have been widely observed in biological systems, including retina [Trenholm and Krishnaswamy, 2020], visual cortex [Zhaoping, 2014], and auditory system [Lesica and Grothe, 2008, Zhao and Zhaoping, 2011], suggesting convergent strategies for temporal feature extraction.

## 4.2 Sequential Classification Tasks Benchmark

We next verify K-prop in deeper and wider networks on benchmark sequential classification task MNIST1D [Greydanus and Kobak, 2024]. Prior benchmarks [Beebe et al., 2026, Ellenberger et al., 2025] report accuracies of approximately 94% for temporal convolutional networks, 91% for GRU, and 96% for human performance. For comparable leaky neural networks trained with GLE [Ellenberger et al., 2025], performance reaches 91% with 15K parameters and 93% with 42K parameters. Using K-prop, we achieve 96.5 ± 0.5% with a 4-layer leaky network (Lk4, 14K parameters) and 97.5 ± 0.3% with a 5-layer configuration (Lk5, 21K parameters), outperforming previously reported leaky network baselines while remaining parameter-efficient. Moreover, K-prop closely matches the BPTT loss, whereas OSTL converges to a substantially higher loss (Fig. 8).

![](images/83df6f4a2bfcb7b53f6537a05d019bcff8f68bf6110b01ec53486748372b1770.jpg)

![](images/a27666ee82eace929c58f3d0b0ad3cae243bf9677773af397637cfb523368383.jpg)  
Figure 6: Motor-skill acquisition through online reinforcement learning. Learning curves for (a) CartPole and (c) LunarLander. Faded traces in the background represent individual runs, while solid curves show the corresponding smoothed learning trajectories. Red dashed lines indicate the task-specific success thresholds, and black dashed lines show the performance of an instantaneous, memoryless network. (b) Animation of LunarLander.

To evaluate K-prop at a larger scale and on a more realistic dataset, we perform classification on Google Speech Commands (GSC) [Warden, 2018]. This task requires substantially longer temporal dependencies to distinguish acoustically similar words such as ’eight’ and ’right’, making long-range credit assignment critical. On the 20-command task, K-prop achieves an accuracy comparable to that reported for GLE on the 10-command task, despite using approximately half as many parameters (0.55M, Table 1). K-prop also outperforms OSTL in accuracy, convergence speed, and training stability. These results suggest that K-prop more effectively propagates credit across long temporal dependencies than OSTL instantaneous error signals (Fig. 5a).

To assess whether K-prop effectively trains deep layers, we perform a layer ablation study on Lk5 (Fig. 5b). Progressively freezing early layers degrades performance, indicating that K-prop provides meaningful training signals throughout the network depth. Together, these results demonstrate that K-prop scales to deeper architectures and larger, more realistic sequential-learning tasks, supporting its use as an online alternative for training leaky models.

## 4.3 Motor learning through real-time reinforcement learning

Motor adaptation in biological systems relies heavily on continuous, online learning, where behavior is adjusted in real time. With online learning, we can adapt faster to new environments and make efficient explorations based on learning. Such scenarios often provide observations that fall into the regime of a partially observable Markov decision process [Kaelbling et al., 1998], requiring agents to construct memory for internal state representations.

To emulate this setting, we use the CartPole and LunarLander tasks from OpenAI Gymnasium [Towers et al., 2024]. We partially mask the observations by removing both positional variables in CartPole and the horizontal position in LunarLander, which makes the instantaneous observation insufficient for optimal control. Although the missing positions can, in principle, be estimated by integrating the observed velocities over time, the agents need to collect historical information correctly without any supervision signal for reconstructing hidden variables. The temporal state inference thus becomes an implicit prerequisite of the reward-maximization task.

We employ an online actor–critic framework similar to [Bellec et al., 2020], using TD(λ) for value estimation. Both actor and critic are implemented as leaky neural networks trained with K-prop. Across multiple runs, the agents consistently succeed in reward maximizing, demonstrating that K-prop supports stable policy and value function learning under partial observability of the system and online updates. Unlike conventional episodic training [Bellec et al., 2020], our setting emphasizes online adaptation during ongoing interaction without batched updates across episodes. In CartPole, in particular, the weights are updated every 3s of interaction. These results show that K-prop provides effective online optimization beyond supervised learning.

## 5 Discussion

## 5.1 Computational and memory efficiency

K-prop significantly reduces both memory and computation cost compared to RTRL, whose memory cost scales with the number of neurons n as $\mathcal { O } ( \hat { n } ^ { 3 } )$ and computational cost is $O ( n ^ { 4 } )$ . K-prop maintains deep eligibility traces $\zeta$ with size $\mathcal { O } ( n _ { \tau } n ^ { 2 } )$ . In practice, we divide neurons within each layer into "fast" and "slow" groups (Table 2), leading to $n _ { \tau } = 2 n _ { \mathrm { l a y e r s } }$ . Since $n _ { \tau } \ll n ,$ this yields a substantial reduction in memory requirement.

The computational cost of K-prop consists of two parts: (i) updating local eligibility traces $\zeta$ and (ii) propagating K. Both operations scale as $\scriptstyle { \mathcal { O } } ( n _ { \tau } n ^ { 2 } )$ per time step, resulting in an overall complexity of the same order. Except for GLE with an extremely efficient memory footprint of ${ \mathcal { O } } ( n )$ , approximation methods such as RFLO, e-prop, UORO, and OSTL achieve $\mathcal { O } ( n ^ { \frac { \cdot } { 2 } } )$ complexity in both memory and computation (Fig. 7). K-prop introduces only a modest additional factor $n _ { \tau }$ , while providing a more accurate gradient that translates into improved empirical performance.

## 5.2 Robustness against noise in learning

Because K-prop implements low-pass filters at all dynamical stages, both for neurons and synaptic eligibility traces, it has a built-in protection against temporal noise. This is essential for physical networks, both biological and artificial (neuromorphic), where such noise is near-ubiquitous. We evaluate learning under noise by adding Gaussian noise to Z. K-prop retains an average MNIST1D accuracy of 94.5% at noise variance $\sigma = 1$ and 92.3% at $\sigma = 1 0$ (Fig. 5), which are strong noise levels relative to $\mathcal { Z }$ (bounded within [−1, 1]). These results demonstrate that K-prop remains robust even under substantial perturbations.

## 5.3 Weight updating during online learning

In practice, we observe that applying weight updates at discrete intervals, rather than continuously at every time step, is important for stable learning. This observation reflects several fundamental challenges in online gradient-based learning, including exact RTRL.

First, online gradients implicitly assume that past contributions are evaluated using the current weights, whereas the weights evolve over time. This mismatch introduces inconsistencies between the accumulated eligibility traces and the current parameter state. Second, continuously updating weights interferes with the forward dynamics: the network is simultaneously required to maintain a stable temporal memory while its parameters are being modified, which can disrupt the memory dynamics that generate useful network output. Third, instantaneous gradients are inherently noisy, as they depend on transient activity and error signals; aggregating gradients over a temporal window can reduce variance, analogous to batch averaging in standard stochastic gradient descent.

## 6 Conclusion

We introduced K-prop, a novel online learning algorithm for feedforward networks of leaky neurons. K-prop significantly reduces the computational and memory cost of RTRL while maintaining a close approximation to the exact gradient, as evidenced by strong directional alignment. Empirically, K-prop achieves performance comparable to RTRL and BPTT on temporal tasks such as N-sum and MNIST1D, and further demonstrates its applicability in reinforcement learning through continuous control in the Cartpole balancing task. K-prop outperforms other approximation methods, which we attribute to more accurate temporal credit assignment across multiple time scales and layers.

From an engineering perspective, K-prop serves as a practical surrogate for RTRL, enabling the study and deployment of online gradient learning in deeper and wider networks. By avoiding the prohibitive cost of RTRL and the severe performance degradation of prior approximations, which potentially conceals the intrinsic limitations, it allows a clearer assessment of online gradient learning methodological properties. Its layer-wise structure and compatibility with standard architectures and techniques, such as residual connections, convolutions, normalization and drop-out, make it a viable candidate for integrating online temporal learning into modern deep learning pipelines.

From a neuroscience perspective, K-prop provides insight into how eligibility traces may operate in deeper neural circuits. In particular, it suggests a framework in which multiple eligibility traces coexist at each synapse, each selectively interacting with different error-related signals, with reward signals provided by distinct neuro-modulators. Such a mechanism could support more flexible and expressive temporal credit assignment as compared to simple three-factor learning rules Gerstner et al. [2018]. Meanwhile, K-prop can be extended to other neuron models that include, for example, multiple dendritic compartments or adaptive mechanisms by adding structured auxiliary kernels.

Finally, our analysis highlights a key direction for future improvement: the approximation of the activation derivative through $\hat { \rho ^ { \prime } }$ . The accuracy of K-prop critically depends on the assumption that $\rho ^ { \prime }$ varies slowly in time. Relaxing this assumption, especially in regimes with discontinuous or highly non-smooth activations, remains an open challenge. In particular, extending K-prop to spiking neural networks, where $\rho ^ { \prime }$ is inherently impulsive, will require more refined surrogate formulations. Addressing this limitation could further broaden the applicability of K-prop to a wider class of temporal and biologically grounded models.

## Acknowledgements

We gratefully acknowledge funding from the European Union under grant agreement #101147319 (EBRAINS 2.0). WS and BL acknowledge the Swiss National Science Foundation (SNF) for Project funding (Project #10004971). We would like to express particular gratitude for the ongoing support from the Manfred Stärk Foundation.

## Competing interests

The authors declare there are no competing interests.

## References

Emmanuel Abbe and Enric Boix-Adsera. On the non-universality of deep learning: quantifying the cost of symmetry. Advances in Neural Information Processing Systems, 35:17188–17201, 2022.

Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.

Michael Beebe, GodsGift Uzor, Manasa Chepuri, Divya Sree Vemula, and Angel Ayala. Exploring the performance of ml/dl architectures on the mnist-1d dataset. arXiv preprint arXiv:2602.13348, 2026.

Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, and Wolfgang Maass. A solution to the learning dilemma for recurrent networks of spiking neurons. Nature communications, 11(1):3625, 2020.

Frederik Benzing, Marcelo Matheus Gauy, Asier Mujika, Anders Martinsson, and Angelika Steger. Optimal kronecker-sum approximation of real time recurrent learning. In International Conference on Machine Learning, pages 604–613. PMLR, 2019.

Thomas Bohnstingl, Stanisław Wo´zniak, Angeliki Pantazi, and Evangelos Eleftheriou. Online spatiotemporal learning in deep neural networks. IEEE Transactions on Neural Networks and Learning Systems, 34(11):8894–8908, 2022.

Simon Brandt, Mihai Alexandru Petrovici, Walter Senn, Katharina Anna Wilmes, and Federico Benitez. Prospective and retrospective coding in cortical neurons. arXiv preprint arXiv:2405.14810, 2024.

Claudia Clopath, Lorric Ziegler, Eleni Vasilaki, Lars Büsing, and Wulfram Gerstner. Tag-triggerconsolidation: a model of early and late long-term-potentiation and depression. PLoS computational biology, 4(12):e1000248, 2008.

Peter Dayan and Laurence F Abbott. Theoretical neuroscience: computational and mathematical modeling ofneural systems. MIT press, 2005.

Alain Destexhe, Michael Rudolph, and Denis Paré. The high-conductance state of neocortical neurons in vivo. Nature reviews neuroscience, 4(9):739–751, 2003.

Yudai Ebato, Sou Nobukawa, Yusuke Sakemi, Haruhiko Nishimura, Takashi Kanamaru, Nina Sviridova, and Kazuyuki Aihara. Impact of time-history terms on reservoir dynamics and prediction accuracy in echo state networks. Scientific Reports, 14(1):8631, 2024.

Benjamin Ellenberger, Paul Haider, Federico Benitez, Jakob Jordan, Kevin Max, Ismael Jaras, Laura Kriener, and Mihai A Petrovici. Backpropagation through space, time and the brain. Nature Communications, 2025.

Wulfram Gerstner, Marco Lehmann, Vasiliki Liakoni, Dane Corneil, and Johanni Brea. Eligibility traces and plasticity on behavioral time scales: experimental support of neohebbian three-factor learning rules. Frontiers in neural circuits, 12:53, 2018.

Samuel James Greydanus and Dmitry Kobak. Scaling down deep learning with MNIST-1d. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview. net/forum?id=n9pru4bJU9.

Paul Haider, Benjamin Ellenberger, Laura Kriener, Jakob Jordan, Walter Senn, and Mihai A Petrovici. Latent equilibrium: a unified learning theory for arbitrarily fast computation with arbitrarily slow neurons. Advances in neural information processing systems, 34:17839–17851, 2021.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

Kazuki Irie, Anand Gopalakrishnan, and Jürgen Schmidhuber. Exploring the promise and limits of real-time recurrent learning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=V2cBKtdC3a.

Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134, 1998.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 2002.

Nicholas A Lesica and Benedikt Grothe. Efficient temporal processing of naturalistic sounds. PLoS One, 3(2):e1655, 2008.

Asier Mujika, Florian Meier, and Angelika Steger. Approximating real-time recurrent learning with random kronecker factors. Advances in neural information processing systems, 31, 2018.

James M Murray. Local online learning in recurrent networks with random feedback. eLife, 8:e43299, may 2019. ISSN 2050-084X. doi: 10.7554/eLife.43299. URL https://doi.org/10.7554/ eLife.43299.

John D Murray, Alberto Bernacchia, David J Freedman, Ranulfo Romo, Jonathan D Wallis, Xinying Cai, Camillo Padoa-Schioppa, Tatiana Pasternak, Hyojung Seo, Daeyeol Lee, et al. A hierarchy of intrinsic timescales across primate cortex. Nature neuroscience, 17(12):1661–1663, 2014.

Cian O’Donnell. Nonlinear slow-timescale mechanisms in synaptic plasticity. Current opinion in neurobiology, 82:102778, 2023.

Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and Quoc V Le. Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779, 2019.

Walter Senn, Dominik Dold, Akos F Kungl, Benjamin Ellenberger, Jakob Jordan, Yoshua Bengio, João Sacramento, and Mihai A Petrovici. A neuronal least-action principle for real-time learning in cortical circuits. ELife, 12:RP89674, 2024.

Shai Shalev-Shwartz, Ohad Shamir, and Shaked Shammah. Failures of gradient-based deep learning. In International Conference on Machine Learning, pages 3067–3075. PMLR, 2017.

Aaron Spieler, Nasim Rahaman, Georg Martius, Bernhard Schölkopf, and Anna Levina. The expressive leaky memory neuron: an efficient and expressive phenomenological neuron model can solve long-horizon tasks. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=vE1e1mLJ0U.

Pablo Stoliar, Julien Tranchant, Benoit Corraze, Etienne Janod, Marie-Paule Besland, Federico Tesler, Marcelo Rozenberg, and Laurent Cario. A leaky-integrate-and-fire neuron analog realized with a mott insulator. Advanced Functional Materials, 27(11):1604740, 2017.

Corentin Tallec and Yann Ollivier. Unbiased online recurrent optimization. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id= rJQDjk-0b.

Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

Stuart Trenholm and Arjun Krishnaswamy. An annotated journey through modern visual neuroscience. Journal ofNeuroscience, 40(1):44–53, 2020.

Pete Warden. Speech commands: A dataset for limited-vocabulary speech recognition. arXiv preprint arXiv:1804.03209, 2018.

Paul J Werbos. Backpropagation through time: what it does and how to do it. Proceedings of the IEEE, 78(10):1550–1560, 2002.

Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2):270–280, 1989.

Ronald J Williams and David Zipser. Gradient-based learning algorithms for recurrent networks and their computational complexity. In Backpropagation, pages 433–486. Psychology Press, 2013.

Jia-Qin Yang, Ruopeng Wang, Zhan-Peng Wang, Qin-Yuan Ma, Jing-Yu Mao, Yi Ren, Xiaoyang Yang, Ye Zhou, and Su-Ting Han. Leaky integrate-and-fire neurons based on perovskite memristor for spiking neural networks. Nano Energy, 74:104828, 2020.

Lingyun Zhao and Li Zhaoping. Understanding auditory spectro-temporal receptive fields and their changes with input statistics by efficient coding principles. PLoS computational biology, 7(8): e1002123, 2011.

Li Zhaoping. Understanding Vision: Theory, Models, and Data. Oxford University Press, Oxford, UK, 2014.

Nicolas Zucchet, Robert Meier, Simon Schug, Asier Mujika, and Joao Sacramento. Online learning of long-range dependencies. Advances in Neural Information Processing Systems, 36:10477–10493, 2023.

## A Mathematical proofs

## Hierarchical exponential kernels are equivalent to linear combinations of kernels

Before analyzing gradients in the network, we first derive a fundamental lemma for this theory. Hierarchical exponential kernels can be expressed as a linear summation of exponential kernels. This result can be derived using Laplace transforms; however, here we provide a derivation based on elementary calculus. Consider a double exponential kernel over a temporal variable x:

$$
\overline { { { \overline { { x } } } ^ { \tau _ { i } } } } ^ { \tau _ { j } } ( t ) = \int _ { t _ { 0 } } ^ { t } d t ^ { \prime } \int _ { t _ { 0 } } ^ { t ^ { \prime } } d t ^ { \prime \prime } \left( \frac { 1 } { \tau _ { j } } e ^ { \frac { t ^ { \prime } - t } { \tau _ { j } } } \frac { 1 } { \tau _ { i } } e ^ { \frac { t ^ { \prime \prime } - t ^ { \prime } } { \tau _ { i } } } x ( t ^ { \prime \prime } ) \right)\tag{7}
$$

$$
= \int _ { t _ { 0 } } ^ { t } d t ^ { \prime \prime } x ( t ^ { \prime \prime } ) \int _ { t ^ { \prime \prime } } ^ { t } d t ^ { \prime } \left( { \frac { 1 } { \tau _ { j } } } e ^ { \frac { t ^ { \prime } - t } { \tau _ { j } } } { \frac { 1 } { \tau _ { i } } } e ^ { \frac { t ^ { \prime \prime } - t ^ { \prime } } { \tau _ { i } } } \right)\tag{8}
$$

$$
= \int _ { t _ { 0 } } ^ { t } d t ^ { \prime \prime } x ( t ^ { \prime \prime } ) \frac { 1 } { \tau _ { i } \tau _ { j } } e ^ { \frac { t ^ { \prime \prime } } { \tau _ { i } } - \frac { t } { \tau _ { j } } } \int _ { t ^ { \prime \prime } } ^ { t } d t ^ { \prime } e ^ { \frac { t ^ { \prime } } { \tau _ { j } } - \frac { t ^ { \prime } } { \tau _ { i } } }\tag{9}
$$

$$
= \int _ { t _ { 0 } } ^ { t } d t ^ { \prime \prime } x ( t ^ { \prime \prime } ) \frac { 1 } { \tau _ { j } - \tau _ { i } } \left( e ^ { \frac { t ^ { \prime \prime } - t } { \tau _ { j } } } - e ^ { \frac { t ^ { \prime \prime } - t } { \tau _ { i } } } \right)\tag{10}
$$

$$
= \frac { \tau _ { i } } { \tau _ { i } - \tau _ { j } } \overline { { x } } ^ { \tau _ { i } } + \frac { \tau _ { j } } { \tau _ { j } - \tau _ { i } } \overline { { x } } ^ { \tau _ { j } } .\tag{11}
$$

This rule can be applied iteratively to additional exponential kernels. For example, for three exponential kernels:

$$
\overline { { \overline { { x } } ^ { \tau _ { i } } \tau _ { j } } } ^ { \tau _ { k } } ( t ) = \frac { \tau _ { i } } { \tau _ { i } - \tau _ { j } } \overline { { \overline { { x } } ^ { \tau _ { i } } } } ^ { \tau _ { k } } + \frac { \tau _ { j } } { \tau _ { j } - \tau _ { i } } \overline { { \overline { { x } } ^ { \tau _ { j } } } } ^ { \tau _ { k } }\tag{12}
$$

$$
= \frac { \tau _ { i } } { \tau _ { i } - \tau _ { j } } \left( \frac { \tau _ { i } } { \tau _ { i } - \tau _ { k } } \overline { { x } } ^ { \tau _ { i } } + \frac { \tau _ { k } } { \tau _ { k } - \tau _ { i } } \overline { { x } } ^ { \tau _ { k } } \right) + \frac { \tau _ { j } } { \tau _ { j } - \tau _ { j } } \left( \frac { \tau _ { k } } { \tau _ { k } - \tau _ { i } } \overline { { x } } ^ { \tau _ { k } } + \frac { \tau _ { j } } { \tau _ { j } - \tau _ { k } } \overline { { x } } ^ { \tau _ { j } } \right)\tag{13}
$$

$$
= \frac { \tau _ { i } } { \tau _ { i } - \tau _ { j } } \frac { \tau _ { i } } { \tau _ { i } - \tau _ { k } } \overline { { x } } ^ { \tau _ { i } } + \frac { \tau _ { j } } { \tau _ { j } - \tau _ { i } } \frac { \tau _ { j } } { \tau _ { j } - \tau _ { k } } \overline { { x } } ^ { \tau _ { j } } + \frac { \tau _ { k } } { \tau _ { k } - \tau _ { i } } \frac { \tau _ { k } } { \tau _ { k } - \tau _ { j } } \overline { { x } } ^ { \tau _ { k } } \ : .\tag{14}
$$

Then, this lemma can be generalized to an arbitrary number of exponential kernels (a set of distinct time constants Υ), where $\mathbf { \bar { \mathcal { T } } } _ { j } ^ { \tau _ { i } } = 1$ when $i = j$ , which allows a more compact representation:

$$
\boldsymbol { \kappa } ( x , \boldsymbol { \Upsilon } , t ) = \sum _ { \tau _ { i } \in \boldsymbol { \Upsilon } } c _ { i } \overline { { x } } ^ { \tau _ { i } } ( t ) \ , \ \mathrm { w i t h } \ c _ { i } = \prod _ { \tau _ { j } \in \boldsymbol { \Upsilon } , j \neq i } \frac { \tau _ { i } } { \tau _ { i } - \tau _ { j } } = \prod _ { \tau _ { j } \in \boldsymbol { \Upsilon } } \mathcal { T } _ { j } ^ { \tau _ { i } } \ .\tag{15}
$$

The limitation arises when multiple time constants are identical, in which case the expression reduces to the special case of the alpha function, which cannot be written as a simple linear combination. We therefore assume $\tau _ { i } \neq \tau _ { j }$ for any $i \neq j$

## Exact gradient for 2-leaky-layer networks

To begin with, we transform the ${ \mathrm { O D E } } \tau { \dot { \pmb u } } = - { \pmb u } + { \pmb W } r$ into its solution form $\pmb { u } ( t ) = \pmb { W } \overline { { \pmb { r } } } ^ { \tau } ( t )$ with the initial condition $u ( t _ { 0 } ) = 0$ . We use bold symbols to denote vectors and matrices. Consider a network architecture with two hidden layers (and widths of the input, hidden and output layer denoted by $n _ { \mathrm { i n } } , n _ { 1 } , n _ { 2 }$ and $n _ { \mathrm { o u t } } )$ , where neurons have exponential filters with different time constants τ , except for the instantaneous output neurons (formally, we only need that neurons from different layers have different time constants). For the last leaky layer, the gradient for minimizing the objective function can be written as:

$$
\dot { W } _ { [ 2 ] } ( t ) \propto - \frac { d { \pmb u } _ { [ 2 ] } ( t ) } { d W _ { [ 2 ] } } \frac { \partial { \pmb r } _ { [ 2 ] } ( t ) } { \partial { \pmb u } _ { [ 2 ] } ( t ) } \frac { \partial { \pmb \mathcal { C } } ( t ) } { \partial { \pmb r } _ { [ 2 ] } ( t ) }\tag{16}
$$

$$
= - \bar { r } _ { [ 1 ] } ( t ) \odot \rho _ { [ 2 ] } ^ { \prime } ( t ) \frac { \partial \mathcal { C } ( t ) } { \partial \pmb { r } _ { [ 2 ] } ( t ) } .\tag{17}
$$

Here $\bar { r } _ { l }$ without superscript is a matrix of shape $( n _ { l } \times n _ { l + 1 } )$ where each column is filtered by corresponding post-synaptic $\tau .$ . Since the output layer is instantaneous, we can directly compute the error arriving at the last leaky layer: $\begin{array} { r } { \pmb { \epsilon } ( t ) = - \frac { \partial \mathcal { C } ( t ) } { \partial \pmb { r } _ { [ 2 ] } ( t ) } = - \pmb { W } _ { \mathrm { o u t } } ^ { \top } \frac { \partial \mathcal { C } ( t ) } { \partial \pmb { r } _ { o } ( t ) } } \end{array}$ , which can be readily

obtained. The exact gradient can then be computed online ∝ $z _ { [ 2 ] } \odot \epsilon$ . This reduces to RFLO Murray [2019] and E-prop Bellec et al. [2020] for the last hidden layer, which is in fact equivalent to $\zeta _ { [ 2 ] } ^ { \tau = 0 }$ considering downstream instantaneous neurons.

For the deeper layer L1, the simple eligibility traces with local τ are no longer enough, since the temporal gradient $\frac { \partial \mathcal { C } ( t ) } { \partial \pmb { r } _ { [ 1 ] } ( t ^ { \prime } < t ) } \neq 0$ . The gradient must integrate over temporal contributions $\frac { \partial { \pmb u } _ { [ 2 ] } ( t ) } { \partial { \pmb r } _ { [ 1 ] } ( t ^ { \prime } < t ) }$ which are scaled by the decay term $\textstyle { \frac { 1 } { \tau } } e ^ { \frac { t ^ { \prime } - t } { \tau } }$

$$
\dot { W } _ { [ 1 ] } ( t ) \propto - \int _ { t _ { 0 } } ^ { t } \bar { r } _ { \mathrm { i n } } ( t ^ { \prime } ) \rho _ { [ 1 ] } ^ { \prime } ( t ^ { \prime } ) \frac { \partial { u } _ { [ 2 ] } ( t ) } { \partial { r } _ { [ 1 ] } ( t ^ { \prime } ) } \frac { \partial \mathcal { C } ( t ) } { \partial { u } _ { [ 2 ] } ( t ) } d t ^ { \prime }\tag{18}
$$

$$
= \int _ { t _ { 0 } } ^ { t } { \bar { r } } _ { \mathrm { i n } } ( t ^ { \prime } ) \rho _ { [ 1 ] } ^ { \prime } ( t ^ { \prime } ) W _ { [ 2 ] } ^ { \top } \left( { \frac { 1 } { \tau } } e ^ { \frac { t ^ { \prime } - t } { \tau } } \epsilon ( t ) \right) d t ^ { \prime } .\tag{19}
$$

Since we aim to rearrange these terms according to time constants, it becomes difficult to proceed directly with matrix operations. To compute this explicitly, we replace matrix operations with summations. We start by focusing on an arbitrary neuron ⋆ in L1, denote its input weight vector by $W _ { \mathrm { i n , \star } } \in \mathbb { R } ^ { n _ { \mathrm { i r } } }$ , the corresponding eligibility trace by $z _ { \star } \in \mathbb { R } ^ { n _ { \mathrm { i n } } }$ and the weight to neuron i in L2 by $W _ { i \star }$ . We then calculate

$$
\dot { W } _ { \mathrm { i n , \star } } ( t ) \propto \int _ { t _ { 0 } } ^ { t } \bar { r } _ { \mathrm { i n } } ^ { \tau _ { \star } } ( t ^ { \prime } ) \rho _ { \star } ^ { \prime } ( t ^ { \prime } ) \sum _ { i } W _ { i \star } \frac { 1 } { \tau _ { i } } e ^ { \frac { t ^ { \prime } - t } { \tau _ { i } } } \epsilon _ { i } ( t ) d t ^ { \prime }\tag{20}
$$

$$
= \sum _ { i } \epsilon _ { i } ( t ) \int _ { t _ { 0 } } ^ { t } z _ { \star } ( t ^ { \prime } ) \rho _ { \star } ^ { \prime } ( t ^ { \prime } ) W _ { i \star } \frac { 1 } { \tau _ { i } } e ^ { \frac { t ^ { \prime } - t } { \tau _ { i } } } d t ^ { \prime }\tag{21}
$$

$$
= \sum _ { i } W _ { i \star } \epsilon _ { i } ( t ) \overline { { z _ { \star } } } \tau _ { i } ( t ) .\tag{22}
$$

Equation 21 can be considered as the closed-form solution of the RTRL sensitivity in continuous time, which accounts for the contribution of parameters in $W _ { \mathrm { i n } }$ to each L2 neuron where the error arrives. We next group L2 neurons that share the same time constant and collect all distinct $\boldsymbol { \tau } ^ { \prime } \mathbf { s }$ into the set $\Upsilon _ { L 2 }$ . This allows us to merge the coefficients within each group as the kernel gain $\kappa$ and rewrite the expression as:

$$
\dot { W } _ { \mathrm { i n } , \star } ( t ) \propto \sum _ { \Upsilon _ { L 2 } } K _ { \star } ^ { \tau } \zeta _ { \star } ^ { \tau } , K _ { \star } ^ { \tau } = \sum _ { i | \tau _ { i } = \tau } W _ { i \star } \epsilon _ { i } ( t ) = \sum _ { i } \epsilon _ { i } ( t ) W _ { i \star } T _ { i } ^ { \tau } ,\tag{23}
$$

where $\tau$ compactly replaces the selection condition $\tau _ { i } = \tau$ . Therefore, we construct a specialized version of K-prop in which the exact gradient can be computed in two layers networks more efficiently than RTRL.

## To three and more leaky layers

In the same manner, we can compute the gradient for networks with more leaky layers. To maintain consistent layer indexing for the back-propagation process, we use [−index] to denote layers by their distance from the output, and indices $i , j , k , \dots$ . follow this order, with ϵ still as the error signal arrives at the last leaky layer. We write the chain rule in an order that facilitates the later transformation into summation form:

$$
\dot { W } _ { [ - 3 ] , \star } ( t ) \propto - \frac { \partial \mathcal { C } ( t ) } { \partial u _ { [ - 1 ] } ( t ) } \int _ { t _ { 0 } } ^ { t } d t ^ { \prime } \left( \frac { \partial u _ { [ - 1 ] } ( t ) } { \partial u _ { [ - 2 ] } ( t ^ { \prime } ) } \int _ { t _ { 0 } } ^ { t ^ { \prime } } d t ^ { \prime \prime } \frac { \partial u _ { [ - 2 ] } ( t ^ { \prime } ) } { \partial r _ { [ - 3 ] } ( t ^ { \prime \prime } ) } z _ { \star } ( t ^ { \prime \prime } ) \right)\tag{24}
$$

$$
= \sum _ { i } \epsilon _ { i } ( t ) \rho _ { i } ^ { \prime } ( t ) \int _ { t _ { 0 } } ^ { t } d t ^ { \prime } \sum _ { j } W _ { i j } \frac { 1 } { \tau _ { i } } e ^ { \frac { t ^ { \prime } - t } { \tau _ { i } } } \rho _ { j } ^ { \prime } ( t ^ { \prime } ) \int _ { t _ { 0 } } ^ { t ^ { \prime } } d t ^ { \prime \prime } W _ { j \star } \frac { 1 } { \tau _ { j } } e ^ { \frac { t ^ { \prime \prime } - t ^ { \prime } } { \tau _ { j } } } z _ { \star } ( t ^ { \prime \prime } )\tag{25}
$$

$$
= \sum _ { i } \sum _ { j } W _ { i j } W _ { j \star } \epsilon _ { i } ( t ) \rho _ { i } ^ { \prime } ( t ) \int _ { t _ { 0 } } ^ { t } d t ^ { \prime } \left( \frac { 1 } { \tau _ { i } } e ^ { \frac { t ^ { \prime } - t } { \tau _ { i } } } \rho _ { j } ^ { \prime } ( t ^ { \prime } ) \int _ { t _ { 0 } } ^ { t ^ { \prime } } d t ^ { \prime \prime } \frac { 1 } { \tau _ { j } } e ^ { \frac { t ^ { \prime \prime } - t ^ { \prime } } { \tau _ { j } } } z _ { \star } ( t ^ { \prime \prime } ) \right)\tag{26}
$$

$$
= \sum _ { i } \sum _ { j } W _ { i j } W _ { j \star } \epsilon _ { i } ( t ) \rho _ { i } ^ { \prime } ( t ) \overline { { { \rho _ { j } ^ { \prime } } { \overline { { z _ { \star } ( t ^ { \prime \prime } ) } } } ^ { \tau _ { j } } } } ^ { \tau _ { i } } ( t ) .\tag{27}
$$

To reduce this triple exponential kernel to a double exponential form, we can follow the computation in Equation 9. However, in this case, we encounter an intractable expression due to the temporally varying $\rho _ { j } ^ { \prime } ( t ^ { \prime } )$

$$
\overline { { \rho _ { j } ^ { \prime } \overline { { z _ { \star } } } \tau _ { j } } } ^ { \tau _ { i } } ( t ) = \int _ { t _ { 0 } } ^ { t } d t ^ { \prime } \frac { 1 } { \tau _ { i } } e ^ { \frac { t ^ { \prime } - t } { \tau _ { i } } } \rho _ { j } ^ { \prime } ( t ^ { \prime } ) \int _ { t _ { 0 } } ^ { t ^ { \prime } } d t ^ { \prime \prime } \frac { 1 } { \tau _ { j } } e ^ { \frac { t ^ { \prime \prime } - t ^ { \prime } } { \tau _ { j } } } z _ { \star } ( t ^ { \prime \prime } )\tag{28}
$$

$$
= \int _ { t _ { 0 } } ^ { t } d t ^ { \prime \prime } z _ { \star } ( t ^ { \prime \prime } ) \frac { 1 } { \tau _ { j } \tau _ { i } } e ^ { \frac { t ^ { \prime \prime } } { \tau _ { j } } - \frac { t } { \tau _ { i } } } \left[ \int _ { t ^ { \prime \prime } } ^ { t } d t ^ { \prime } \rho _ { j } ^ { \prime } ( t ^ { \prime } ) e ^ { \frac { t ^ { \prime } } { \tau _ { i } } - \frac { t ^ { \prime } } { \tau _ { j } } } \right] .\tag{29}
$$

This expression can be approximately handled by extracting $\rho _ { j } ^ { \prime }$ from the integral and introducing a surrogate variable $\hat { \rho } ^ { \prime }$ , such that $\begin{array} { r } { \frac { \rho _ { j } ^ { \prime } ( t ^ { \prime } ) } { \hat { \rho ^ { \prime } } _ { j } ( t ) } \approx 1 } \end{array}$ for recent times $t - t ^ { \prime } < T \epsilon$ . This approximation effectively requires $\rho ^ { \prime }$ to vary slowly over time, which is better satisfied by smooth activation functions.

$$
\overline { { \rho _ { j } \overline { { z _ { \star } } } \tau _ { j } } } ^ { \tau _ { i } } ( t ) = \hat { \rho _ { j } ^ { \prime } } ( t ) \overline { { \frac { \rho _ { j } ^ { \prime } } { \hat { \rho _ { j } ^ { \prime } } ( t ) } \overline { { z _ { \star } } } \tau _ { j } } } ^ { \tau _ { i } } ( t )\tag{30}
$$

$$
\approx \hat { \rho _ { j } ^ { \prime } } ( t ) \overline { { \overline { { z _ { \star } } } ^ { \tau _ { j } } } } ^ { \tau _ { i } } ( t ) .\tag{31}
$$

Then, according to Equation 11, we can convert the double exponential kernel into a summation:

$$
\dot { W } _ { [ - 3 ] , \star } ( t ) \propto \sum _ { i } \sum _ { j } W _ { i j } W _ { j \star } \epsilon _ { i } ( t ) \rho _ { i } ^ { \prime } ( t ) \hat { \rho _ { j } ^ { \prime } } ( t ) \left( \frac { \tau _ { j } } { \tau _ { j } - \tau _ { i } } \overline { { z } } _ { \star } ^ { \tau _ { j } } ( t ) + \frac { \tau _ { i } } { \tau _ { i } - \tau _ { j } } \overline { { z } } _ { \star } ^ { \tau _ { i } } \right)\tag{32}
$$

$$
= \sum _ { i } \sum _ { j } W _ { i j } W _ { j \star } \epsilon _ { i } ( t ) \rho _ { i } ^ { \prime } ( t ) \hat { \rho _ { j } ^ { \prime } } ( t ) \left( \mathcal { T } _ { i } ^ { \tau _ { j } } \zeta _ { \star } ^ { \tau _ { j } } ( t ) + \mathcal { T } _ { j } ^ { \tau _ { i } } \zeta _ { \star } ^ { \tau _ { i } } ( t ) \right) .\tag{33}
$$

In the same sense, we can derive a similar expression for the −4 hidden layer by introducing an additional summation over the k-th neuron in the additional layer.

$$
\dot { W } _ { [ - 4 ] , \star } ( t ) \propto \sum _ { i } \sum _ { j } \sum _ { k } W _ { i j } W _ { j k } W _ { k \star } \epsilon _ { i } \rho _ { i } ^ { \prime } ( t ) \overline { { { \rho _ { j } ^ { \prime } } \overline { { { \rho _ { k } ^ { \prime } } \overline { { { z _ { \star } } } } ^ { \tau _ { k } } } } ^ { \tau _ { j } } } }  ( t )\tag{34}
$$

$$
\approx \sum _ { i , j , k } \sum _ { k } W _ { i j } W _ { j k } W _ { k \star } \epsilon _ { i } ( t ) \rho _ { i } ^ { \prime } \hat { \rho _ { j } ^ { \prime } } \hat { \rho _ { k } ^ { \prime } } ( t ) \bar { \overline { { z _ { \star } } } } ^ { \tau _ { k } \tau _ { j } \tau _ { i } } ( t )\tag{35}
$$

$$
= \sum _ { i , j , k } \sum _ { k } W _ { i j } W _ { j k } W _ { k \star } \epsilon _ { i } \rho _ { i } ^ { \prime } \hat { \rho } _ { j } ^ { \prime } \hat { \rho } _ { k } ^ { \prime } \left( \mathcal { T } _ { j } ^ { \tau _ { k } } \mathcal { T } _ { i } ^ { \tau _ { k } } \zeta _ { \star } ^ { \tau _ { k } } + \mathcal { T } _ { i } ^ { \tau _ { j } } \mathcal { T } _ { k } ^ { \tau _ { j } } \zeta _ { \star } ^ { \tau _ { j } } + \mathcal { T } _ { j } ^ { \tau _ { i } } \mathcal { T } _ { k } ^ { \tau _ { i } } \zeta _ { \star } ^ { \tau _ { i } } \right) ( t )\tag{36}
$$

To generalize this result to arbitrary depth, we define a path p as synaptic chain passing from a initial neuron $i _ { [ l ] }$ at layer l to a next neuron $j _ { [ l + 1 ] }$ at layer $l { + } 1$ until we end a neuron $v _ { [ L ] }$ in the output layer, $p = ( i _ { [ l ] } , \dot { \boldsymbol { j } } _ { [ l + 1 ] } . . . , \boldsymbol { v } _ { [ L ] } )$ . The path weight $w _ { p }$ is defined as the product of all synaptic weights, the rate derivatives $\rho ^ { \prime }$ , and the error $\epsilon _ { p }$ entering the path $p ,$

$$
w _ { p } = \epsilon _ { p } \prod _ { i \in p } W _ { p ( i + 1 ) , p ( i ) } \hat { \rho } _ { i + 1 } ^ { \prime }\tag{37}
$$

where i runs through the components of the path $p$ up to the second last.

We next define the set of paths $\mathcal { P } _ { \star }$ as set of paths that start at neuron ⋆ in a hidden layer l. By $\mathcal { P } _ { \star } ^ { \tau }$ we denote the subset of paths starting at ⋆ and containing along its path a neuron that low-pass filters the input with $\tau , \mathrm { i . e . } \tau \in \Upsilon _ { p }$ . Using these path sets, we can rewrite the hierarchical summation over layers $\textstyle \sum _ { i , j , k }$ as a single summation over paths:

$$
\dot { W } _ { \star } ( t ) \propto \sum _ { p \in \mathcal { P } _ { \star } } w _ { p } \sum _ { \tau \in \Upsilon _ { p } } \prod _ { j \in p \backslash \star } \mathcal { T } _ { j } ^ { \tau } \zeta _ { \star } ^ { \tau } .\tag{38}
$$

Thus, we obtain an approximation of the gradient for all neurons in a network of arbitrary depth, expressed in terms of paths. The remaining question is how to compute this efficiently, since the number of paths in $\mathcal { P } _ { \gamma }$ <sub>⋆</sub> grows exponentially $\begin{array} { r } { | \mathcal { P } _ { \star } | = \prod _ { l ^ { \prime } = l + 1 } ^ { L } n _ { l ^ { \prime } } } \end{array}$ , with $l ^ { \prime }$ running across all layers downstream of neuron ⋆.

## Kernel gain by regrouping summation over τ

The first observation is that a single τ can participate in multiple paths, and this multiplicity becomes even larger when the same τ is shared within a layer. This suggests that maximizing the reuse of $\zeta _ { \star } ^ { \tau }$ is beneficial. Therefore, we aim to group the coefficients with respect to τ by changing the order of summation. We define the set $\Upsilon _ { \star }$ of all downstream time constants and rewrite Eq. (38) by

$$
\dot { W } _ { \star } ( t ) \propto \sum _ { \tau \in \Upsilon _ { \star } } \sum _ { p \in \mathcal { P } _ { \star } ^ { \tau } } w _ { p } \prod _ { j \in p \setminus \star } \mathcal { T } _ { j } ^ { \tau } \zeta _ { \star } ^ { \tau } = \sum _ { \tau \in \Upsilon _ { \star } } K _ { \star } ^ { \tau } \zeta _ { \star } ^ { \tau }\tag{39}
$$

$$
\mathrm { w i t h } \ K _ { \star } ^ { \tau } = \sum _ { p \in \pmb { \mathscr { P } } _ { \star } ^ { \tau } } w _ { p } \ \prod _ { j \in p \backslash \star } \ T _ { j } ^ { \tau } .\tag{40}
$$

Three challenges arise when computing K: 1. The number of paths in $\mathcal { P } _ { \star } ^ { \tau }$ grows exponentially with increasing depth. 2. Using $w _ { p }$ requires neurons to access distant downstream information, such as feedback error, synaptic weights, and activations. 3. Neurons must be aware of the precise downstream network structure, i.e., which neurons belong to a given $p ^ { \backslash \star }$ . Since ζ can be tracked locally in an online manner, it is crucial to determine how K can be computed efficiently.

## Layer-wise propagation of K

To address the above challenges, we note that $\kappa$ can be propagated from layer l + 1 to l, and that this leads to an efficient computational scheme. In a feedforward architecture, each hidden neuron inherits and extends all paths from layer $l + 1$ . Define $p _ { + } = p ^ { \backslash \star }$ as the path obtained from $p$ by removing neuron ⋆ in layer l and instead starting from neuron ⋆⋆ in layer $l + 1$ . Then we have $w _ { p } = W _ { \star \star , \star } \hat { \rho } _ { \star \star } ^ { \prime } w _ { p _ { \jmath } }$ + and

$$
K _ { \star } ^ { \tau } = \sum _ { p \in \pmb { \mathscr { P } } _ { \star } ^ { \tau } } w _ { p } \prod _ { j \in p \backslash \star } \ T _ { j } ^ { \tau }\tag{41}
$$

$$
= \sum _ { p \in \mathcal { P } _ { \star } ^ { \tau } } \mathcal { T } _ { \star \star } ^ { \tau } \hat { \rho } _ { \star \star } ^ { \prime } W _ { \star \star , \star } \left( w _ { p ^ { \prime } } \prod _ { j \in p _ { + } ^ { \backslash \star \star } } \mathcal { T } _ { j } ^ { \tau } \right) .\tag{42}
$$

Since many paths share the same node in layer $l + 1$ , we can rearrange the summation to first sum over neuron i in the next layer:

$$
\begin{array} { l } { { \displaystyle { \mathcal K } _ { \star } ^ { \tau } = \sum _ { i \in l + 1 } \hat { \rho _ { i } ^ { \prime } } { \mathcal T } _ { i } ^ { \tau } W _ { i \star } \sum _ { p \in { \mathcal P } _ { i , \tau } } w _ { p } \prod _ { j \in p } { \mathcal T } _ { j } ^ { \tau \tau } } } \\ { { \displaystyle ~ = \sum _ { i \in l + 1 } \hat { \rho _ { i } ^ { \prime } } { \mathcal T } _ { i } ^ { \tau } W _ { i \star } { \mathcal K } _ { i } ^ { \tau } } . } \end{array}\tag{43}
$$

(44)

Note that $\mathcal { K } _ { \star } ^ { \tau }$ must be defined for every τ and every neuron in the network to support this iterative computation. This equation can be written in matrix form as in Equation 4, which highlights its strong similarity to instantaneous gradient backpropagation.

## T in other network architectures

Under a rigorous definition, $\mathcal { T } _ { j } ^ { \tau } = 0$ when τ does not occur on any path through neuron $j .$ In a fully connected feedforward network, this applies only to time constants associated with other neurons in the same layer as $j$ and distinct from $\tau _ { j }$ . Other architectures can also contain time constants that do not occur on any path through j in different situations.

Residual networks [He et al., 2016] provide one example where skip connections can provide pathways with faster temporal responses than the residual pathway. To apply K-prop to a residual network, we propagate each kernel gain $\kappa ^ { \tau }$ along both the skip and residual pathways, then sum the contributions at the merge point for each τ. The eligibility traces $\zeta ^ { \tau }$ are unchanged. Along the skip pathway, set $\mathcal { T } _ { i } ^ { \tau } = 0$ for time constants that occur only along the residual pathway, since they do not occur on any path through a neuron on the skip pathway.

More generally, consider a direct connection matrix from a source layer m to target layer $n ,$ with neuron j in layer m. Kernels contained by any other, more indirect paths between layers m and n (via other intermediate layers) will neither affect nor be affected by kernels along the direct connection from m to n, so their contribution to $\mathcal { T } _ { j , [ m n ] } ^ { \tau }$ should be zero. More formally, let $\Upsilon _ { [ l < m ] }$ denote the set of time constants associated with layers whose indices satisfy $l < m$ , and let $\Upsilon _ { [ l \geq n ] }$ denote the set associated with layers whose indices satisfy $l \geq n$ . The transformation factor for this connection is

$$
\mathcal { T } _ { j , [ m n ] } ^ { \tau } = \left\{ \begin{array} { l l } { \frac { \tau } { \tau - \tau _ { j } } } & { \tau \in \Upsilon _ { [ l < m ] } \mathrm { o r } \tau \in \Upsilon _ { [ l \ge n ] } } \\ { 1 } & { \tau _ { j } = \tau } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{45}
$$

This allows us to create the transformation matrices $\tau$ efficiently before training without having to evaluate all possible paths.

## Identical or zero τ

A special case arises when identical time constants appear across multiple layers. As discussed in the main text, this leads to singularities in the kernel decomposition. The singularity can be resolved either by grouping identical τ into a single basis component or by introducing infinitesimal perturbations. The grouping approach suggests replacing ζ with higher-order kernels built from nested identical τ, such as the alpha function $\textstyle { \frac { t } { \tau } } e ^ { - t / \bar { \tau } }$ . The transformation factor $\tau$ and the propagation rule would then need to be modified accordingly to accommodate the new basis. While the grouping approach is theoretically valid, it introduces additional complexity in both the kernel representation and the propagation rules. To prevent grouping, we consider infinitesimal perturbations for simplicity and consistency with the main formulation.

Biologically, different time constants across layers are consistent with the hierarchical increase of time constants in the primate cortex [Murray et al., 2014]. Empirically, we do not observe any benefit from using identical or near-identical time constants across layers. In our experiments, time constants are chosen to be distinct across layers. We further verify that small perturbations (e.g., differences on the order of 0.1) do not affect numerical stability or performance.

Another special case is $\tau = 0$ . This case is taken into account because $T _ { i } ^ { \tau = 0 } = 0$ for any neuron j including $\tau _ { j } = 0$ . An instantaneous pathway with co-existence of $\tau = 0$ neurons across layers does not cause any singularity, which collapses into simple backpropagation.

## Practical implementation of K-prop

A few notes for practical implementation: 1. We define $\zeta ^ { \tau _ { \mathrm { u p } } } = 0$ , so each layer can omit tracking the upstream components of ζ. 2. A special case arises for the first leaky layer: its τ does not need to be included in $\Upsilon _ { \mathrm { n e t } }$ (since there are no upstream neurons), and its nonlinearity $\rho _ { L 1 }$ does not affect the approximation. 3. K propagation is only required when an external error signal is present (otherwise all $\kappa = 0 )$ . In many tasks, including real world scenarios, error signals are sparse in time, allowing K propagation to be triggered in an event-based manner. Altogether, in practical implementations, K-prop has both computational and memory costs on the order of half of $\mathcal { O } ( n _ { \tau } n ^ { \frac { \mathbf { i } } { 2 } } )$ . Under this efficient implementation, we can plot the cost comparison:

This reduction is well illustrated in the CartPole task, where K-prop reduces RTRL memory cost by 28× and computational cost by 1,684×, turning an otherwise intractable method into a practical one. Although BPTT is computationally efficient, it incurs a memory cost that scales with sequence length T, as it requires storing the full trajectory. In comparison, K-prop reduces memory usage by up to 57×, at the expense of a moderate $( \sim 4 \times )$ increase in computation relative to BPTT, demonstrating a classical space-time trade-off. Simulations are programmed in Python with PyTorch. Code is available through https $: / / \mathfrak { g } \mathrm { i }$ thub.com/Baskin-LIU/Kernel-Propagation#

![](images/9cafa75de0714dfadfe7b6ec0be4bd59f398fdb61b421fa3ed1481106adc1c34.jpg)

![](images/5ab430b16d4714c54402f6e8479410527389adf39625cbacae46e747338cd7cd.jpg)

![](images/85a33ba7d0be44282a5c09e9d4878bca7dfab76b47145dcfdebad5426445a614.jpg)  
Figure 7: Compare the cost under multi-layer architecture. Left: Memory cost Middle: Computational cost per time step. GLE has ${ \mathcal { O } } ( n )$ memory and $\mathcal { O } ( n ^ { 2 } )$ computational cost. All costs are computed in optimal layer-wise implementations instead of directly from scaling law. Right: Wallclock release time and maximum memory allocation per epoch on the MNIST1D dataset, measured across an increasing number of leaky neurons in 5-layer networks. Experiments were conducted on a local machine equipped with an NVIDIA RTX 4060 Laptop GPU (8 GB). Overall running time reflects additional latency introduced by temporal loop overhead.

Algorithm 1 Kernel Propagation (K-prop) for Online Learning in Leaky Networks   
1: Initialization: Network with L layers, time constants τ, step size dt   
2: Initialize weights $\{ W , b \}$ randomly ▷ bias is treated as weight for $r = 1$   
3: Construct transformation factor matrices $\{ \tau \}$   
4: for each iteration / episode do   
5: Reset $\bar { \pmb { r } } _ { [ l ] }  0 , \dot { \zeta } _ { [ l ] }  0 , \Delta \pmb { W } _ { [ l ] }  0$   
6: for each time step t do   
7: for layer $l = 1$ to L do ▷ Forward dynamics   
8: $\begin{array} { r } { \bar { r } _ { [ l ] }  ( 1 - \frac { d t } { \tau _ { l } } ) \bar { r } _ { [ l ] } + \frac { d t } { \tau _ { l } } r _ { [ l ] } } \end{array}$   
9: $\pmb { u } _ { [ l ] }  \pmb { W } _ { [ l ] } \bar { \pmb { r } } _ { [ l - 1 ] } + \pmb { b } _ { [ l ] }$   
10: $\pmb { r } _ { [ l ] }  \rho ( \pmb { u } _ { [ l ] } )$   
11: $\begin{array} { r } { \dot { \zeta _ { [ l ] } } \gets ( 1 - \frac { d t } { \tau _ { \mathrm { n e t } } } ) \odot \dot { \zeta _ { [ l ] } } + \frac { d t } { \tau _ { \mathrm { n e t } } } \left( \rho _ { [ l ] } ^ { \prime } \otimes \bar { r } _ { [ l - 1 ] } \right) } \end{array}$ ▷ Deep eligibility trace update   
12: end for   
13: if $\frac { \partial \mathcal { C } } { \partial { r } _ { L } } \neq 0$ then ▷ Check if error signal is available   
14: $\begin{array} { r l } { \bar { \mathcal { K } } _ { L } ^ { \tau }  \frac { \partial \mathcal { C } } { \partial \boldsymbol { r } _ { L } } } & { { } \forall \tau \in \Upsilon _ { \mathrm { n e t } } } \end{array}$ ▷ Initialize K at output layer   
15: for layer $\bar { l = } L$ down to 1 do ▷ Backward kernel propagation   
16: $\pmb { \mathcal { K } } _ { [ l - 1 ] } ^ { \tau }  \pmb { W } _ { [ l ] } ^ { \top } \big ( \hat { \pmb { \rho ^ { \prime } } } _ { [ l ] } \odot \pmb { \mathcal { K } } _ { [ l ] } ^ { \tau } \odot \pmb { \mathcal { T } } _ { [ l ] } ^ { \tau } \big )$   
17: $\begin{array} { r } { \dot { W } _ { [ l ] }  \sum _ { \tau \in \Upsilon _ { \mathrm { d o w n } } ^ { l } } \mathrm { d i a g } ( \mathcal { K } _ { [ l ] } ^ { \tau } ) \zeta _ { [ l ] } ^ { \tau } } \end{array}$   
18: $\Delta W _ { [ l ] }  \Delta W _ { [ l ] } + \dot { W } _ { [ l ] }$ ▷ Gradient accumulation   
19: end for   
20: end if   
21: if Update weight then   
22: for layer $\bar { l = 1 }$ to L do   
23: $\dot { W } _ { [ l ] }  W _ { [ l ] } - \eta \Delta W _ { [ l ] }$   
24: $\Delta \dot { W } _ { [ l ] }  0$   
25: end for   
26: end if   
27: end for   
28: end for   
D Exn ntal details

## B Experimental details

## B.1 MNIST1D dataset

MNIST1D converts digit patterns analogous to the handwritten images in MNIST [LeCun et al., 2002] into one-dimensional temporal sequences. Each sample spans 360ms and contains a digit pattern of approximately 90ms embedded at a random temporal position, requiring both temporal

![](images/b0b5c7c264fcba1d342aceec88212de469f5957ee9f6eca8299ed81032bfab3c.jpg)

![](images/0083e20496e08f6139e33e0a29ac01f030e2c641316f21896ff9830169f5fc8a.jpg)  
Figure 8: Learning curves on MNIST1D. Left: Mean, minimum, and maximum training loss. Kprop closely matches the loss achieved by BPTT, whereas OSTL converges to a substantially higher loss. Right: Mean test error for the Lk4 and Lk5 networks. Curves are smoothed for visualization.

localization and pattern recognition. Apart from increasing the number of time steps, we use the default sample-generation settings of Greydanus and Kobak [2024], including the default training/test split of 4000/1000 samples.

## B.2 GSC dataset

The Google Speech Commands (GSC) dataset [Warden, 2018] comprises utterances lasting approximately 0.8-1s, recorded from multiple speakers. We evaluate the 20-class command-label subset, rather than the 12-class subset used in Ellenberger et al. [2025]. The selected commands are [’zero’, ’one’, ’two’, ’three’, ’four’, ’five’, ’six’, ’seven’, ’nine’, ’eight’,’yes’, ’no’, ’up’, ’down’, ’left’, ’right’, ’on’, ’off’, ’stop’, ’go’]. We use the default training, validation, and test splits provided by the PyTorch dataset. The original audio waveforms are converted into 80-channel log-Mel spectrograms and interpolated to 500 time frames, corresponding to a time step of approximately 2ms. We apply data augmentation based on SpecAugment [Park et al., 2019], including frequency masking over four channels, temporal shifts of up to 25ms, and temporal warping of up to 25ms.

## B.3 Loss and Optimizer

In all experiments, we use $r = \operatorname { t a n h } ( \alpha u )$ with $\alpha \approx 0 . 5$ , making it behave similarly to a centered sigmoid. The learning rate decays exponentially in N-sum and follows a cosine annealing schedule (180 epochs total) in MNIST1D. All methods share the same learning rate scheduler. We use standard gradient descent for N-sum and the Adam optimizer for all other tasks. For Adam, $( \beta _ { 1 } , \beta _ { 2 } )$ are set to (0.9, 0.999) for BPTT and (0.8, 0.995) for K-prop to reduce momentum and encourage stronger averaging. We have verified that (0.8, 0.995) on BPTT yields worse performance, excluding the advantage brought by the optimizer setup.

The loss is defined online in all experiments. For N-sum, we use logistic output units with a meansquared-error loss. For MNIST1D and GSC, cross-entropy loss is evaluated at every time step within a designated response period: the entire sequence for MNIST1D and the final 600,ms for GSC. The total loss is therefore the sum over all answering time steps $\begin{array} { r } { T , \sum _ { t \in T } \mathcal { L } ( u _ { o } ( t ) , y ) } \end{array}$

We use a set of random seeds 0, 42, 428, 781, 3407, 115, 256, 666 for training.

## B.4 Reinforcement learning tasks

For the Cartpole environment, we extend the termination boundaries to $\pm 3 6 ^ { \circ }$ (from ±24<sup>◦</sup>) for pole angle and ±3.6 (from ±2.4) for cart position, allowing longer and more variable episode. To obtain more continuous inputs, observations from each environment step (20 ms) are interpolated by inserting 3 intermediate frames to obtain $d t = 5 m s$ . Actor has two output neurons with softmax activation, producing the probability of Left and Right. The action choice at each step is sampled by a Bernoulli distribution with the possibility.

In LunarLander, the agent must smoothly land a spacecraft within a target zone using four discrete actions: do nothing, fire left engine, fire main engine, and fire right engine. The environment follows the default OpenAI Gymnasium configuration, except that the horizontal position is masked from the observation. Frame interpolation is not required, and actions are sampled in the same manner as in CartPole. The task is considered successfully learned when the recent average reward exceeds 205.

The reinforcement learning algorithm follows Bellec et al. [2020], which transforms $\mathrm { T D } ( \lambda ) \dot { W } \propto$ $\begin{array} { r } { \bar { \delta } \frac { \partial V } { \partial W } \to \bar { \delta } \overline { { \frac { \partial V } { \partial W } } } } \end{array}$ . The critic propagates a scalar value signal, while the actor propagates a cross-entropy loss using the chosen action as the target label. The propagated gradient is first accumulated in an auxiliary term that decays with λ, and is then scaled by $\dot { \delta ( t ) } = \bar { R } ( t ) + \gamma V ( t + d t ) - V ( t )$ , where $\gamma = 0 . 9 9$ is the discount rate. $V ( t + d t )$ is computed by a target critic network that is softly updated toward the critic with a factor of 0.05. We devices a special learning rate scheduler: After each episode, the learning rate is decreased by a factor of ×0.997 if the episodic reward is lower than the running average, and increased by +0.001 when episodic reward is larger. otherwise. Due to the inherent instability of actor–critic methods and the absence of batch learning, we clip both δ and the gradient norm to prevent value explosion.

## B.5 Implementation configurations

<table><tr><td>Model</td><td>dt (ms)</td><td>Leaky layers</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td></tr><tr><td>N-sum</td><td>0.25</td><td>3</td><td>[0, 6.28]</td><td>0.25, 3, 6</td><td>2,4,10</td><td></td><td></td></tr><tr><td>Lk4</td><td>1</td><td>4</td><td>[0, 10]</td><td>3,6</td><td>2,7</td><td>2.5,8</td><td></td></tr><tr><td>Lk5</td><td>1</td><td>5</td><td>[0, 10]</td><td>3,6</td><td>2,7</td><td>1,8</td><td>2.5, 6.6</td></tr><tr><td>GSC</td><td>2</td><td>4</td><td>[0,40]</td><td>3,12,24</td><td>4, 16,36</td><td>50,400</td><td></td></tr><tr><td>Cartpole</td><td>5</td><td>3</td><td>[0, 40]</td><td>5,10</td><td>11,24</td><td></td><td></td></tr><tr><td>Lunarlander</td><td>5</td><td>3</td><td>[0, 20]</td><td>5,20</td><td>6,10</td><td></td><td></td></tr></table>

Table 2: Network time constants (ms). Since the first layer does not need to be included in $\Upsilon _ { \mathrm { n e t } }$ , it is preferable to use a wider range of time constants in the first layer and fewer in downstream layers. Accordingly, we assign the first-layer time constants τ to be evenly spaced in log space. In the other layers, we use two distinct τ values per layer, divided into two equal groups. We find that introducing more distinct τ values within a layer does not yield notable improvements, as long as both a group of “fast” and a group of “slow” neurons are present. Square brackets indicate the log-space range for L1.

The issues from online weight updating Section 5.3 can be mitigated either by using very small learning rates or by accumulating gradients over a time interval and applying periodic updates. We find the latter more effective, as it preserves learning speed while improving stability. This design choice is also consistent with biological observations, where long-term synaptic plasticity operates on slower timescales rather than instantaneously [Clopath et al., 2008, O’Donnell, 2023].

Meanwhile, our experiments indicates a necessity of "fast" neurons with much shorter τ than typical neuron membrane potential time constants. Such range of time constants could be achieved through modulation of the membrane conductance [Destexhe et al., 2003] or prospective coding [Haider et al., 2021, Senn et al., 2024, Brandt et al., 2024].

Both N-sum, Cartpole, and LunarLander simulations can run on CPU hardware within 20 minutes. MNIST1D and GSC benefits from GPU acceleration due to larger matrix operations. A mid-range GPU such as a GTX 4060 reduces MNIST1D training time from approximately 4 hours on CPU to around 1.5 hours. Additional experimental details are provided below.

<table><tr><td>Model</td><td>Network width</td><td>α</td><td>Learning rate</td><td>Update Period</td><td>Batch size</td></tr><tr><td>N-sum</td><td>6-6-6-2</td><td>0.6</td><td>0.1</td><td>50 ms</td><td>1</td></tr><tr><td>Lk4</td><td>36-54-60-60-72-10</td><td>0.6</td><td>0.01</td><td>180 ms</td><td>100</td></tr><tr><td>Lk5</td><td>54-60-60-60-60-72-10</td><td>0.5</td><td>0.01</td><td>180 ms</td><td>100</td></tr><tr><td>GSC</td><td>240-360-360-400-400-20</td><td>0.5</td><td>0.001</td><td>1s</td><td>256</td></tr><tr><td>Cartpole</td><td>32-40-30-20-2</td><td>0.6</td><td>0.01</td><td>3 s</td><td>1</td></tr><tr><td>Lunärlander</td><td>120-120-80-20-4</td><td>0.6</td><td>0.0002</td><td></td><td>1</td></tr></table>

Table 3: Network configurations

![](images/3309b57c581e14614b6ccb06f43b9e30e015abfb300f47a6fec6acc0b24a6f98.jpg)

![](images/95899cb744b211208243b07ed1b81298c367a78250b73ead4dd865e4b91393ce.jpg)

![](images/7a595ed61f601d152929b2fc6333b53bc5860c59ce836f64ded486119c67566c.jpg)  
Figure 9: Further investigation of how the surrogate and nonlinearity affect K-prop approximation. (Left) Gradient cosine similarity check with BPTT like Fig. 3 on α and another surrogate. The dissimilarity increased when bigger α enhance nonlinearity. Here $\hat { \rho } ^ { \prime } = \bar { \rho } ^ { \prime } { } ^ { \tau _ { \rho } }$ while Srg2 uses the neurons’ local τ for $\tau _ { \rho }$ and $\mathrm { S r g } 3$ uses $\tau _ { \rho }$ as a summation of downstream layers’ τ. Srg3 shows an advantage in early layers while slightly degrading layers close to the output. (Mid) Investigating the practical tradeoff between nonlinearity and K-prop gradient quality by tuning α in MNIST1D, which shows the different sweet spots on Lk4 and Lk5. (Right) Comparing with networks with $\alpha = 1$ and trained with BPTT. Enhanced nonlinearity does not boost final performance.

## B.6 Learning under noise

To investigate the sensitivity of K-prop to noise in its eligibility dynamics, we inject an independent Gaussian perturbation into each integration step of Eq. (5), with n $\sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . We evaluate the Lk4 network on MNIST1D using noise levels $\sigma \in { 0 . 2 , 1 . 0 , 5 . 0 , 1 0 , 2 0 }$ . K-prop remains effective across a broad range of noise levels. Given that z is bounded within [−1, 1], the results indicate that the perturbations must be large relative to the driving signal before learning performance deteriorates substantially.

We attribute this robustness to two properties of K-prop:

1. Independent Gaussian perturbations are smoothed by the leaky integration dynamics. When driven by white noise, these dynamics produce an Ornstein–Uhlenbeck-like process in which the gradient signal contributes to the drift.

2. The resulting noisy gradient estimates are further averaged over time through gradient accumulation and the optimizer dynamics.

These results provide preliminary evidence that K-prop can tolerate the noisy dynamics expected in physical computing substrates.

## C Robustness of the approximation

## C.1 Tuning the non-linearity

In terms of nonlinearity, the K-prop approximation works better for smoother and flatter functions such as logistic and tanh. Scaling down the input $\rho ( \alpha x ) , 0 < \alpha < 1$ slows down the variation and thus improves the approximation. However, a smaller α reduces the expressiveness of the nonlinearity, which may degrade performance. Therefore, there is a trade-off between approximation quality and nonlinearity, and a “sweet spot” typically exists that yields the best performance. The sweet spot depends on the network depth and τ configuration, as illustrated in the MNIST1D experiment.

In Lk4, the test accuracy increases significantly from $\alpha = 0 . 2$ and reaches an optimum around $\alpha = 0 . 5 , 0 . 6$ The performance slightly decreases as α approaches 1, while still maintaining an accuracy of 94%, suggesting that K-prop preserves a usable gradient quality. In Lk5, the sweet spot shifts to $\alpha = 0 . 5$ . We therefore choose $\alpha = 0 . 6$ for Lk4 and $\alpha = 0 . 5$ for Lk5. Meanwhile, models with $\alpha = 1$ trained with BPTT do not show significant improvement, indicating that α scaling is an effective hyperparameter that can be tuned for K-prop.

## C.2 Surrogate choice

The choice of surrogate derivative has an important effect on K-prop performance. In all experiments, we use the low-pass-filtered surrogate $\hat { \rho } ^ { \prime } = \bar { \rho } ^ { \tau _ { \rho } }$ which consistently improves performance over the instantaneous choice $\hat { \rho } ^ { \prime } = \rho ^ { \prime }$

For all tasks except GSC, we set the filtering time constant $\tau _ { \rho }$ to the local time constant τ of each neuron. For $\mathrm { G S C } , \tau _ { \rho }$ at each layer is instead set to the sum of the time constants in its downstream layers. This produces stronger smoothing in earlier layers, reflecting the cumulative temporal filtering along their longer paths to the output. The gradient-alignment analysis in Fig. 9 shows that this cumulative surrogate improves gradient quality in the earlier layers.