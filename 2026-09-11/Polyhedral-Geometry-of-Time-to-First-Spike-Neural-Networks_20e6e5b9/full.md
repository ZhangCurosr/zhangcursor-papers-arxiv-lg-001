# Polyhedral Geometry of Time-to-First-Spike Neural Networks

Manjot Singh<sup>1,4</sup>, Guido Mont´ufar<sup>2,3</sup>, and Gitta Kutyniok<sup>1,4,5</sup>

<sup>1</sup>Department of Mathematics, Ludwig-Maximilians-Universit¨at M¨unchen

<sup>2</sup>Departments of Mathematics and Statistics & Data Science, University of California,

Los Angeles

<sup>3</sup>Max Planck Institute for Mathematics in the Sciences, Leipzig <sup>4</sup>Munich Center for Machine Learning <sup>5</sup>Department of Physics and Technology, University of Tromsø

## Abstract

We study the expressivity of spiking neural networks, which provide a natural framework for asynchronous, event-driven computation complementary to conventional feedforward neural networks. We consider the time-to-first-spike model in a setting for which the input–output map is continuous and piecewise linear, with afine pieces governed by causal feasibility constraints that determine which presynaptic spikes occur before a neuron fires. We first show that each neuron’s firing time admits a maxout-like representation with exponentially many, highly constrained afine pieces. We then formalize causal regions as polyhedral regions with fixed causal sets and derive upper and lower bounds on the maximal number of causal regions in both shallow and multilayer feedforward spiking networks. Our theoretical and experimental results show that spiking networks can generate richer partitions of the input space than conventional feedforward ReLU networks.

## 1 Introduction

Spiking neural networks (SNNs) are increasingly studied as alternatives to standard artificial neural networks (ANNs) for two closely related reasons. First, they are event-driven models in which neurons communicate via discrete spikes, aligning with a large body of work in computational neuroscience on coding schemes that represent information through spike timing, firing rates, or spike trains [1]. Second, from an engineering perspective, spiking computation is inherently asynchronous. When inputs are sparse or temporally structured, computation is triggered only by sparse events, ofering the potential for low-latency and energy-eficient inference. These advantages motivate the deployment of SNNs in event-based sensing applications, such as event-based cameras and neuromorphic sensors [2], in latency-critical perception tasks on mobile or high-mobility platforms [3], and in energy-constrained environments, such as satellite systems [4].

At the same time, there has been rapid progress on the algorithmic side of SNNs, including surrogate gradient methods [5–7] and spike-based learning algorithms [8–11]. Despite these advances, however, the theoretical understanding of SNNs remains significantly less developed than that of ANNs. For ANNs with piecewise linear activations, a mature body of work has developed a geometric theory based on the number and structure of activation regions (polyhedral regions of the input space on which the network computes an afine function), yielding quantitative insights into the roles of depth, width, and architectural bottlenecks [12–16]. In contrast, a comparable theory for spiking models is still in its infancy.

SNNs encompass a broad range of neuron models and coding schemes, from rate-based representations to temporal codes and dynamical recurrent systems [1, 17]. In this work, we focus on time-to-first-spike (TTFS) coding [1, 18–20], a widely used temporal scheme in which each neuron emits at most one spike and encodes its output through the timing of that spike. In feedforward TTFS-based SNNs, inputs are encoded as spike times, propagated through successive spiking layers, and decoded from the timing of the output. Rather than synchronous activation vectors, each layer processes real-valued spike times, with downstream computation triggered as spikes arrive. This perspective aligns naturally with the engineering motivations discussed above, since single-spike inference is inherently sparse and can support low-latency decision making.

A key modeling ingredient in such networks is the synaptic response kernel (or postsynaptic response function). In the Spike Response Model [17], a neuron integrates weighted copies of this response function and emits a spike when its membrane potential crosses a threshold. When the response function is piecewise linear in time, as for a ReLU-type response, the membrane potential is itself piecewise linear in time. When the weights are positive, the resulting output spike times have been shown to be continuous piecewise linear (CPWL) functions of the presynaptic spike times [21–23]. This CPWL regime is practically relevant and facilitates direct comparison with ANNs with piecewise linear activations. We point out that TTFS-based SNNs do not generally induce CPWL mappings. In particular, when both positive and negative weights are allowed, the mappings realized by SNNs can, under certain conditions, be discontinuous. We isolate the positive-weights TTFS as a natural setting for studying the inductive biases of asynchronous spike-time computation. In what follows, we focus on positive-weight TTFS-based SNNs with CPWL response functions, referred to as pTTFS networks. Unless stated otherwise, throughout this work we will use the terms SNN, TTFS network, and pTTFS network interchangeably to refer to this setting.

Recent work in this direction has established approximation-theoretic and generalization bounds for positive-weights TTFS networks [22, 23] and has begun to elucidate connections between SNNs and ReLU-ANNs, including regimes in which SNNs achieve comparable approximation rates. Despite this progress, separation results between ANNs and TTFS networks remain limited. Moreover, a systematic theory on the the number and geometry of their linear or causal regions is still underdeveloped, particularly for deeper networks beyond single neurons or highly specialized settings [22, 24]. In particular, we lack sharp regioncounting results that characterize how TTFS-specific causal constraints shape the geometry and complexity of the induced partition, give rise to structural biases distinct from those of ReLU-ANNs, and potentially identify regimes in which SNNs can outperform ReLU-ANNs.

Our goal in this work is to develop a systematic theory of causal regions for TTFS networks. As a first step, we restrict attention to the positive-weight setting, leaving the discontinuous regime arising from negative weights as an important direction for future work. Within this setting, we study how architectural and parameter choices afect the number and geometry of causal regions, and compare the resulting bounds with corresponding regioncount bounds for ANNs with piecewise linear activations.

Contributions We consider SNNs in the continuous piecewise linear setting and make the following contributions.

(i) For a single neuron, we give an explicit geometric characterization of when a given causal set is realized. This clarifies how SNNs partition the input space in contrast to standard ReLU networks. In particular, we show that causal regions are governed by causal constraints, which arise as union of regions of an associated hyperplane arrangement. We also provide a polyhedral representation of the single neuron firing-time map, in which its afine pieces are encoded by a lifted TTFS polytope and an associated regular subdivision.

(ii) For shallow SNNs, we derive general upper bounds on the number of regions of the associated TTFS hyperplane arrangement, as well as asymptotic upper and lower bounds on the number of realizable causal regions. In the shared-weight regime, where all hidden neurons share a common weight vector, we show that their causal sets are nested. Exploiting this structure, we obtain sharper lower bounds through an exact count on the realizable causal regions.

(iii) For deep SNNs, we derive a general lower bound by constructing a folding mechanism that produces exponentially many causal regions with depth. For networks with weights shared within each layer, we derive the corresponding upper bound. We show that, once the firing-time order withing a layer is fixed, the causal sets of neurons in the subsequent layers must correspond to prefixes of this order. We exploit this structure to derive our bound and further characterize which causal patterns can and cannot be realized.

(iv) We complement our theoretical results with experiments on CIFAR-10 and MNIST, focusing on the region complexity of randomly initialized networks. We study how the estimated number of causal regions in SNNs varies with architectural choices such as width and depth, compare SNNs with matched ReLU networks, and examine the efect of restricting SNNs to shared weight regime. Across these experiments, we find that SNNs can realize a large number of causal regions, with region complexity depending on architecture and initialization.

Outline Section 2 reviews related work. Section 3 introduces the SNN model and notation. Section 4 analyzes the single-neuron case, introduces causal sets, causal regions, and causal patterns, and develops the geometric framework underlying our subsequent results. Section 6 establishes upper and lower bounds for shallow SNN, while Section 7 develops corresponding bounds for deep networks. Section 8 presents experimental results. Finally, Section 9 discusses open problems, the role of delays, and extensions beyond feedforward architectures.

## 2 Related work

In this section, we position our work relative to prior works on the expressivity and estimation of the number of activation regions in ReLU and Maxout networks, and expressivity of SNNs.

Polyhedral geometry of ANNs For ANNs with piecewise linear activations, a standard measure of expressivity is the number of linear, or activation, regions into which the network partitions its input space, that is, regions on which the network realizes an afine function of the input. This notion has been widely used to compare the expressivity of diferent architectures and to characterize depth-width trade-ofs. Early works [12–14, 25] established that depth can yield exponentially more linear regions than shallow networks with comparable numbers of neurons or parameters. This literature provides the conceptual reference for our study. We seek an analogous, architecture-aware understanding of how SNNs partition their input space. As we will see, the natural counterpart of activation regions in SNNs is given by causal regions.

The geometry of linear regions in piecewise-linear neural networks is naturally described using polyhedral methods. For ReLU networks, region boundaries can be studied through hyperplane arrangements and bent hyperplanes in deeper layers. For higher-rank maxout networks [26], each unit has multiple afine pieces, leading to richer polyhedral subdivisions of the input space. More generally, CPWL functions have natural descriptions in terms of convex polytopes and support functions, as well as through tropical geometry [27–29] and spline-based methods [30]. In particular, functions realized by ReLU and maxout networks can be represented as tropical rational functions [29, 31, 32]. Sharp bounds on the number of linear regions of maxout networks were obtained in [16]. As we will see, a TTFS neuron with ReLU response function admits a maxout-like representation, but with strong dependencies among its afine pieces. Thus, TTFS SNNs can be regarded as highly structured maxout networks with exponentially large efective rank. This structure distinguishes them sharply from generic maxout networks. In particular, the afine pieces of each neuron cannot vary independently, and region-counting bounds for unconstrained maxout networks need not be tight. This motivates the development of TTFS-specific geometric and combinatorial tools for counting regions.

Although our analysis focuses on upper and lower bounds for the maximum number of regions, it is useful to briefly note how region-based viewpoints have been used more broadly in the literature. Maximum region counts characterize the expressive capacity of a network architecture in principle, whereas understanding the complexity typically realized in practice requires studying region statistics under parameter distributions, for example at random initialization or after training. For ReLU and maxout networks, expected-region analyses and empirical studies indicate that typical parameter choices may realize substantially fewer regions than the maximum possible [15, 33–35]. More broadly, region geometry has been used to characterize qualitative aspects of ReLU networks, including the geometry of decision boundaries, the evolution of complexity during training, and robustness [36–39]. By comparison, a theory of typical or expected region complexity for SNNs remains largely undeveloped.

Expressivity of SNNs The theory and practice of SNNs encompass a wide range of neuron models and coding schemes, whose choice is largely guided by the intended application or theoretical question. Broadly, SNNs are studied as models of neural computation in computational neuroscience and as a basis for eficient, brain-inspired AI systems [40, 41]. Under rate-based coding, a growing literature has established universality and approximation results for SNNs [42–44]. For TTFS-based SNNs with a linear response function, the theoretical understanding is more developed. Recent works [22, 23], building on Maass’s early universality result [21], have established expressivity and approximation results for networks in this regime. In particular, [22] showed that SNNs can realize discontinuous PWL maps when negative weights are allowed, and further derived complexity bounds for emulating multilayer ReLU networks with SNNs. In a complementary direction, [23] studies positiveweight TTFS SNNs and establishes approximation and generalization bounds. Related work [45], motivated primarily by training, derives a neuron-to-neuron mapping from ReLU ANN parameters to SNNs parameters that preserves the realized function.

At the same time, ANN-SNN conversion results do not by themselves explain whether SNNs possess diferent computational capabilities from ANNs. Given the distinct mechanisms underlying spike-time computation, can we demonstrate meaningful diferences in the capabilities of ANNs and SNNs? In this direction, Maass demonstrated advantages of SNNs over conventional neural network models on specific biologically motivated computational tasks [21]. Maass and Schmitt [46] further showed, through VC-dimension bounds, that programmable delays can substantially increase the computational capacity of SNNs relative to static threshold circuits. Yet comparatively little is known about how the internal computational structure of SNNs difers geometrically from that of ANNs. This is the perspective we pursue here. We use region complexity to quantify the geometric richness and structural inductive bias of the function class realized by TTFS SNNs.

Region complexity analysis for SNNs is still scarce compared to the extensive literature on ReLU ANNs. Initial steps in this direction appear in [22], where small examples already reveal region structures that difer from those of ReLU ANNs, and more recently in [24], which introduces the notion of causal sets and empirically investigates their dependence on initialization. A key gap is the absence of systematic region-counting bounds for shallow and deep SNNs that account for the causal constraints inherent to TTFS computation, together with a comparison to corresponding bounds for ReLU ANNs. Our work addresses this gap by developing a causal-region framework for CPWL TTFS SNNs, establishing upper and lower bounds for shallow and deep networks, and a systematic description of the structures implemented by TTFS neurons.

## 3 TTFS SNN model

The study of spiking neural networks (SNNs) from computational and biological perspectives has led to a the development of a wide range of neuron models, from biophysically detailed dynamics, such as Hodgkin–Huxley-type model, to simplified abstractions, such as integrateand-fire, leaky integrate-and-fire, and Izhikevich-type models [17], as well as further reduced variants designed primarily for computational eficiency rather than biological realism. For theoretical analysis, it is common to focus on simplified dynamics that retain the essential computational structure while remaining analytically tractable.

In this paper, we consider time-to-first-spike (TTFS) coding in the single-spike regime, where each neuron emits at most one spike and information is represented by its firing time. Thus, neuron outputs are real-valued spike times, providing a natural input-output representation that can be composed by stacking layers. For clarity, we first introduce the underlying spike-time dynamics on a general network graph and then specialize to the feedforward architecture considered throughout the paper. Our notation largely follows [22, 23].

## 3.1 Spike-time dynamics

Definition 3.1 (Spiking neural network). Let $G = ( V , E )$ be a graph, with subsets $V _ { \mathrm { i n } } \subset V$ and $V _ { \mathrm { o u t } } \subset V$ denoting the input and output neurons, respectively, and with directed edges $E \subset V \times V$ representing synapses. Each non-input neuron $v \in V \backslash V _ { \mathrm { i n } }$ has an associated firing threshold $\theta _ { v } > 0$ . Each synapse $( u , v ) \in E$ has the following associated attributes:

• a synaptic weight $w _ { ( u , v ) } \in \mathbb { R } _ { \geq 0 }$

• a synaptic delay $d _ { ( u , v ) } \geq 0$

• a response function $\varepsilon _ { ( u , v ) } : \mathbb { R } \to \mathbb { R } _ { \geq 0 }$

We write $W : = ( w _ { ( u , v ) } ) _ { ( u , v ) \in E } , D : = ( d _ { ( u , v ) } ) _ { ( u , v ) \in E } , \mathcal { E } : = ( \varepsilon _ { ( u , v ) } ) _ { ( u , v ) \in E }$ , and $\Theta : = ( \theta _ { u } ) _ { u \in V \backslash V _ { \mathrm { i n } } }$ for the corresponding collections of synaptic weights, delays, response functions, and firing thresholds, respectively. An SNN is then specified by the tuple $\Phi = \left( G , W , D , \mathcal { E } , \Theta \right)$

The synaptic delay $d _ { ( u , v ) } \geq 0$ represents the time it takes for a spike emitted by neuron u to reach neuron v. Synaptic delays are specific to SNNs and have no direct counterpart in conventional ANNs.

In TTFS models, postsynaptic potentials are commonly constructed from shifted response functions, including ReLU, exponential, and alpha functions [9, 10, 24, 47]. For general response functions, the resulting input-output map need not be piecewise linear. In this work, we consider the ReLU response function, which under positive weights and thresholds leads to a continuous piecewise-linear input-output map while retaining the causal structure of spike-time computation.

Definition 3.2 (ReLU response). Let $G = ( V , E )$ be a network graph. For each synapse $( u , v ) \in E$ , we define the associated response function $\varepsilon _ { ( u , v ) } : \mathbb { R } \to \mathbb { R } _ { \geq 0 }$ by

$$
\varepsilon _ { ( u , v ) } ( t ) : = \sigma ( t ) ,\tag{1}
$$

where $\sigma ( t ) = \operatorname* { m a x } \{ 0 , t \}$ denotes the ReLU activation. Under this specification of the response functions, we suppress E from the notation and write the SNN as $\Phi = ( G , W , D , \Theta )$ .

Given the firing times of the presynaptic neurons, the potential of a postsynaptic neuron accumulates the contributions of spikes that have already arrived. The neuron fires when this potential first reaches its threshold.

Definition 3.3 (Potential and firing time). Let $\Phi = ( G , W , D , \Theta )$ be an SNN with network graph $G = ( V , E )$ . For each neuron $v \in V \setminus V _ { \mathrm { i n } }$ , its potential $P _ { v } : \mathbb { R }  \mathbb { R }$ , is a function that at time t takes value

$$
P _ { v } ( t ) : = \sum _ { ( u , v ) \in E } w _ { ( u , v ) } \sigma ( t - t _ { u } - d _ { ( u , v ) } ) ,\tag{2}
$$

where $t _ { u } \in \mathbb { R }$ denotes the firing time of the presynaptic neuron u. The firing time of neuron v is the smallest time at which its potential reaches the threshold $\theta _ { v }$ :

$$
t _ { v } = \operatorname* { m i n } \{ t \in \mathbb { R } : P _ { v } ( t ) = \theta _ { v } \} .
$$

Here and throughout, presynaptic and postsynaptic refer to the source and target neurons of a synapse, respectively, and we use the terms fire and spike interchangeably. For notational simplicity, we suppress the dependence of $P _ { v }$ on the synaptic weights $w _ { ( u , v ) }$ , delays $d _ { ( u , v ) }$ , and presynaptic firing times $t _ { u }$ whenever these are clear from context.

Remark 3.1 (Well-posedness). Note that, for positive synaptic weights $w _ { ( u , v ) } > 0$ , the potential $P _ { v } ( t )$ is continuous nondecreasing in t and becomes strictly increasing once the first presynaptic spike reaches neuron v. For any fixed presynaptic firing times, $P _ { v } ( t ) \to 0$ as $t \to - \infty$ and $P _ { v } ( t ) \to + \infty { \mathrm { ~ a s ~ } } t \to \infty$ , provided that v has at least one incoming synapse. Since $\theta _ { v } > 0$ , every non-input neuron has a unique and finite firing time.

The spike-time dynamics therefore define an input-output map.

Definition 3.4 (SNN realization). Let $\Phi = ( G , W , D , \Theta )$ be an SNN, and let $d = | V _ { \mathrm { i n } } |$ and $n = \left| V _ { \mathrm { o u t } } \right|$ . The realization of Φ is the function $R ( \Phi ) \colon \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ that maps the firing times $( t _ { v } ) _ { v \in V _ { \mathrm { i n } } }$ of the input neurons to the firing times $( t _ { v } ) _ { v \in V _ { \mathrm { o u t } } }$ of the output neurons.

## 3.2 Feedforward TTFS networks

We are interested in feedforward SNNs, where neurons are organized into layers and synapses connect consecutive layers.

Definition 3.5. Let $L , N _ { 0 } , \dots , N _ { L } \in \mathbb { N }$ . A feedforward SNN of depth L and layer widths $( N _ { \ell } ) _ { \ell = 0 } ^ { L }$ is specified by a tuple

$$
\Phi = ( ( W ^ { \ell } , D ^ { \ell } , \Theta ^ { \ell } ) _ { \ell = 1 } ^ { L } ) ,
$$

where, for each $\ell = 1 , \dots , L , W ^ { \ell } = ( w _ { i j } ^ { \ell } ) \in \mathbb { R } _ { \ge 0 } ^ { N _ { \ell } \times N _ { \ell - 1 } }$ is the matrix of weights and $D ^ { \ell } = ( d _ { i j } ^ { \ell } ) \in$ $\mathbb { R } _ { > 0 } ^ { N _ { \ell } \times N _ { \ell - 1 } }$ the matrix of delays, for synapses from layer ℓ − 1 to layer ℓ, and $\Theta ^ { \ell } = ( \theta _ { i } ^ { \ell } ) \in \mathbb { R } _ { > 0 } ^ { N _ { \ell } }$ is the vector of firing thresholds for neurons in layer ℓ. We write $W = ( W ^ { \ell } ) _ { \ell = 1 } ^ { L } , D = ( D ^ { \ell } ) _ { \ell = 1 } ^ { L } ,$ and $\Theta = ( \Theta ^ { \ell } ) _ { \ell = 1 } ^ { L }$ for the collections of weights, delays, and thresholds of the entire network. The total number of neurons is $\begin{array} { r } { N ( \Phi ) : = \sum _ { \ell = 0 } ^ { L } N _ { \ell } } \end{array}$

Given a vector of input spike times $\vec { t } \in \mathbb R ^ { N _ { 0 } }$ , the membrane potentials and firing times are defined recursively, layer by layer, according to Definitions 3.3. For each $\ell = 1 , \ldots , L$ , this defines a layer map $F ^ { \ell } \colon \mathbb { R } ^ { N _ { \ell - 1 } }  \mathbb { R } ^ { N _ { \ell } }$ , and the realization of the network is the composition

$$
F ^ { L } \circ F ^ { L - 1 } \circ \cdots \circ F ^ { 1 } .
$$

Remark 3.2 (Model considered in this work). In general TTFS-based SNNs, the postsynaptic potential of every neuron is given by a sum of shifted nonlinear response functions (e.g., ReLU, exponential or alpha-functions) [9, 10, 24, 47], and the firing time is defined implicitly as the solution to a nonlinear equation which equates the potential to a threshold. For general response functions, the resulting input-output map need not be piecewise linear.

Unless stated otherwise, throughout the paper we consider feedforward TTFS SNNs in the single-spike regime, with positive synaptic weights, nonnegative synaptic delays, positive firing thresholds, and ReLU response functions. In this setting, the realization map is continuous piecewise linear.

![](images/14fcdda40804eb95f590fe7ab8f641bd80c39d59bfe2fa618a5fbc752a7b72c0.jpg)

![](images/a61341b4aee765511cbd986e09d9db25b1e8695674fbb25ab58b1567b8dd2ba4.jpg)

![](images/14d8a359581a089c76cab02deecf5db27b6fe46d06dc93ec766456fdd7734613.jpg)  
Figure 1: Causal region partitions of TTFS SNNs on the slice $t _ { 3 } = 0$ for input dimension $d = 3 .$ . (Left) Shallow SNN with three neurons, each with the identical weight vector. (Middle) Shallow SNN with three neurons having distinct weight vectors. (Right) Two-layer SNN with three neurons per layer and distinct positive weights.

## 3.3 Causal regions versus activation regions

Although the TTFS SNNs considered here and standard ReLU ANNs both realize continuous piecewise-linear maps, the mechanisms generating their linear regions are fundamentally diferent. In ReLU networks, the afine pieces are determined by activation patterns, which record the signs of the pre-activations. In TTFS SNNs, by contrast, the afine pieces are governed by causal set tuples.

As we show below, the firing time of a TTFS neuron can be expressed as the minimum of finitely many afine functions indexed by causal sets. For a given vector of input spike times, the causal set of a neuron records which presynaptic spikes arrive before the neuron fires. Conditional on a fixed causal set, the output firing time is an afine function of the corresponding input spike times. The associated causal region is characterized by inequalities ensuring that precisely those spikes arrive before firing.

This causal structure leads to an important geometric distinction from ReLU networks. A causal region need not coincide with a single cell of the hyperplane arrangement induced by the relevant afine comparisons; rather, it can be a union of multiple cells corresponding to the same causal set. Moreover, diferent neurons in a given layer can have diferent causal sets. Figure 1 illustrates the resulting causal-region geometry for several shallow and deep TTFS networks.

## 4 Polyhedral geometry of a TTFS neuron

We begin by analyzing the input-output map of a single spiking neuron in the setting of positive weights and no synaptic delays. In this setting, the input-output map is continuous piecewise afine, with afine pieces indexed by causal sets that describe which presynaptic spikes arrive before the neuron fires. The resulting geometric and combinatorial structure provides the foundation for our subsequent analysis of shallow and deep SNNs.

![](images/0e430a6577902bd2e3e7f531cc605d94f05bcd28bad3f0ed2bb60937319e54ee.jpg)  
Figure 2: Schematic for the causal pattern in SNNs. (a) The firing time of each noninput neuron depends on presynaptic spikes that arrive before it fires (colored edges) and they form its causal set. Neuron $v _ { 4 }$ spikes after the output neuron has already fired and therefore does not contribute to the output, illustrating sparse computation. (b) The one-shot spike raster emphasizes asynchronous signal propagation, hidden-layer neurons fire after input spikes, and some neurons that fire too late may become irrelevant for the final output. (c) For a single neuron, the membrane potential is a piecewise linear function; the threshold crossing time $\tau$ depends only on the spikes arriving before τ .

## 4.1 Causal sets and causal regions

To keep the notation light, we work directly with the input spike times, synaptic weights, and firing threshold of the neuron under consideration, without explicit reference to the full network Φ. Throughout this section, we consider an input layer of d presynaptic neurons $u _ { 1 } , \ldots , u _ { d }$ with spike-time vector $\vec { t } : = ( t _ { 1 } , \dotsc , t _ { d } ) \in \mathbb { R } ^ { d }$ , synaptic weights $w _ { 1 } , \ldots , w _ { d } > 0 .$ synaptic delays set to zero, and a single postsynaptic neuron v with firing threshold $\theta _ { v } > 0$ For $d \in \mathbb { N }$ , we write $[ d ] : = \{ 1 , \ldots , d \}$

Following (2), the membrane potential of the postsynaptic neuron v at time t is given by

$$
P _ { v } ( t ) = \sum _ { i = 1 } ^ { d } w _ { i } \sigma ( t - t _ { i } ) .\tag{3}
$$

The firing time $t _ { v }$ is defined implicitly by the threshold-crossing condition

$$
t _ { v } ( \vec { t } ) : = \operatorname* { m i n } \{ t \in \mathbb { R } \colon P _ { v } ( t ) = \theta _ { v } \} .\tag{4}
$$

For any fixed $\vec { t }$ and positive weights $w _ { 1 } , \ldots , w _ { d } > 0$ , the potential $P _ { v } ( t )$ is a continuous and nondecreasing in t, becomes strictly increasing after the first presynaptic spike, and satisfies $P _ { v } ( t ) \to + \infty { \mathrm { ~ a s ~ } } t \to + \infty$ . Since $\theta _ { v } > 0$ , the minimum in the definition of the firing time $t _ { v }$ is attained uniquely. The causal set at <sup>⃗</sup>t is the subset of presynaptic neurons that fire strictly before the postsynaptic neuron, $S = \{ i \in [ d ] \colon t _ { i } < t _ { v } ( \vec { t } ) \}$ . Since $\theta _ { v } \ > \ 0$ , the causal set is nonempty. See Figure 2 for an illustration.

We next study the firing time $t _ { v }$ as a function of the input spike times $\vec { t . }$ Observe that the potential $\begin{array} { r } { P _ { v } ( t ; \vec { t } ) = \sum _ { i = 1 } ^ { d } w _ { i } \sigma ( t - t _ { i } ) } \end{array}$ is a continuous piecewise linear function of $( t , \vec { t } ) \in \mathbb { R } ^ { d + 1 }$ with linear regions separated by the hyperplanes $t - t _ { i } = 0 , i = 1 , \ldots , d .$ . For each subset $S \subseteq [ d ]$ , consider the polyhedral region $L _ { S } = \{ ( t , \vec { t } ) \colon t > t _ { i } \forall i \in S$ and $t \leq t _ { j } \forall j \notin S \}$ . On $L _ { S }$ , the potential is linear and takes the form $\begin{array} { r } { P _ { v } ( t , \vec { t } ) = \sum _ { i \in S } w _ { i } ( t - t _ { i } ) } \end{array}$ . Geometrically, the threshold condition defining the firing time gives the level set

$$
\Sigma _ { \theta _ { v } } : = \{ ( t , \vec { t } ) \in \mathbb { R } ^ { d + 1 } { : } P _ { v } ( t ; \vec { t } ) = \theta _ { v } \} .\tag{5}
$$

For each fixed input $\vec { t , }$ the firing time $t _ { v } ( \vec { t } )$ is the unique value of t such that $( t , \vec { t } ) \in \Sigma _ { \theta _ { v } }$ Equivalently, the level set $\Sigma _ { \theta _ { v } }$ is the graph of the firing-time map $\vec { t } \mapsto t _ { v }$

We now explain how the preceding geometric picture induces regions in the input space. Restricting $\Sigma _ { \theta _ { v } }$ to $L _ { S }$ gives $\begin{array} { r } { \sum _ { i \in S } w _ { i } ( t - t _ { i } ) = \theta _ { v } } \end{array}$ . Solving for t, yields the afine firing time

$$
t _ { v } ^ { S } ( \vec { t } ) : = \frac { \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } } { W _ { S } } , \quad \mathrm { w h e r e ~ } W _ { S } : = \sum _ { i \in S } w _ { i } .\tag{6}
$$

Thus, $t _ { v } ^ { S }$ is the firing time obtained under the assumption that precisely the presynaptic neurons indexed by $S$ fire before neuron v. This candidate coincides with the actual firing time if and only if $( t _ { v } ^ { S } ( \vec { t } ) , \vec { t } ) \in L _ { S }$ , which is equivalent to the causal feasibility inequalities

$$
t _ { v } ^ { S } ( \vec { t } ) > t _ { i } , \quad \mathrm { f o r ~ a l l ~ } i \in S , \quad \mathrm { a n d ~ } t _ { v } ^ { S } ( \vec { t } ) \leq t _ { j } , \quad \mathrm { f o r ~ a l l ~ } j \notin S .\tag{7}
$$

We therefore define the causal region associated with $S$ by

$$
R _ { S } = \{ \vec { t } \in \mathbb { R } ^ { d } : t _ { v } ^ { S } ( \vec { t } ) > t _ { i } , \mathrm { ~ f o r ~ a l l ~ } i \in S , \mathrm { ~ a n d ~ } t _ { v } ^ { S } ( \vec { t } ) \leq t _ { j } , \mathrm { ~ f o r ~ a l l ~ } j \notin S \} .\tag{8}
$$

Thus, $R _ { S }$ consists precisely of those inputs <sup>⃗</sup>t for which S is the causal set and the firing time is given by $t _ { v } = t _ { v } ^ { S } ( \vec { t } )$ . Since $t _ { v } ^ { S }$ is afine, $R _ { S }$ is defined by afine inequalities and is therefore a convex polyhedron. Geometrically, the level set $\Sigma _ { \theta _ { \imath } }$ is a piecewise-afine hypersurface in $\mathbb { R } ^ { d + 1 }$ , while its projection onto the input coordinates partitions $\mathbb { R } ^ { d }$ into causal regions on which the firing-time map is afine. See Figure 3 for an illustration.

Having established that each input belongs to a unique causal region, we next show that the firing time can be written as the pointwise minimum of finitely many afine functions. In particular, the firing-time map is piecewise afine and concave. Although this result already appears in [22], we state it here for completeness.

Proposition 4.1. The firing-time map satisfies

$$
t _ { v } ( \vec { t } ) = t _ { v } ^ { S } ( \vec { t } ) , \quad f o r e v e r y \vec { t } \in R _ { S } , f o r e v e r y S \subseteq [ d ] , S \neq \emptyset .
$$

In particular, $t _ { v }$ is afine on each causal region $R _ { S }$ . Moreover, it admits the representation

$$
t _ { v } ( \vec { t } ) = \operatorname* { m i n } _ { S \subseteq [ d ] , S \neq \emptyset } t _ { v } ^ { S } ( \vec { t } ) , \quad f o r \ a l l \ \vec { t } \in \mathbb { R } ^ { d } .\tag{9}
$$

Consequently, $t _ { v }$ is a continuous piecewise-afine concave function.

Proof. Fix $\vec { t } \in \mathbb R ^ { d }$ and let $\tau : = t _ { v } ( \vec { t } )$ be the corresponding firing time. If $S$ is the causal set of $\vec { t , }$ then by construction $\tau = t _ { v } ^ { S } ( \vec { t } )$ , so $t _ { v }$ is afine on each causal region $R _ { S }$

![](images/2f1aaae0f5fe25f610a115e9d188eec4713b8143234cd339d9491980538aafde.jpg)

Figure 3: For $d = 2 , w _ { 1 } = w _ { 2 } = 1$ , and $\theta _ { v } = 1$ , the threshold level set of the potential function, $\Sigma _ { \theta _ { v } } = \{ ( t , \vec { t } ) \colon P _ { v } ( t ; \vec { t } ) = \theta _ { v } \} \subseteq \mathbb { R } ^ { d + 1 }$ , and its projection onto the input space $\mathbb { R } ^ { d } .$ illustrating the linear regions of the firing time $t _ { v } ( \vec { t } )$

It remains to prove the minimum representation. For any nonempty $S \subseteq [ d ]$ , define

$$
P _ { v } ^ { S } ( t ; \vec { t } ) : = \sum _ { i \in S } w _ { i } ( t - t _ { i } ) .
$$

At the firing time τ, we have

$$
P _ { v } ^ { S } ( \tau ; \vec { t } ) = \sum _ { i \in S } w _ { i } ( \tau - t _ { i } ) \leq \sum _ { i = 1 } ^ { d } w _ { i } \sigma ( \tau - t _ { i } ) = P _ { v } ( \tau ; \vec { t } ) = \theta _ { v } .
$$

Indeed, if $t _ { i } < \tau$ , the corresponding term is the same in the linear and ReLU expressions, whereas if $t _ { i } \geq \tau$ , the linear term is nonpositive while the ReLU term is zero.

On the other hand, by the definition of $t _ { v } ^ { S } ( \vec { t } )$

$$
P _ { v } ^ { S } ( t _ { v } ^ { S } ( \vec { t } ) ; \vec { t } ) = \theta _ { v } .
$$

Since ${ P _ { v } ^ { S } } ( t ; \vec { t } )$ is strictly increasing in t, these two relations imply

$$
t _ { v } ^ { S } ( \vec { t } ) \geq \tau .
$$

If S is the causal set of $\vec { t , }$ then $P _ { v } ^ { S } ( \tau ; \vec { t } ) = P _ { v } ( \tau ; \vec { t } ) = \theta _ { v }$ , and hence $t _ { v } ^ { S } ( \vec { t } ) = \tau$ . Therefore,

$$
t _ { v } ( \vec { t } ) = \operatorname* { m i n } _ { \emptyset \neq S \subseteq [ d ] } t _ { v } ^ { S } ( \vec { t } ) .
$$

Since $t _ { v }$ is the pointwise minimum of finitely many afine functions, it is continuous, piecewise afine, and concave. □

Proposition 4.1 shows that, for any given input $\vec { t , }$ once the corresponding causal set S is known, the firing time is given by the afine expression (6). Alternatively, (9) provides a global representation of $t _ { v } .$ , without requiring prior knowledge of its causal set.

It is worth noting that computing $t _ { v }$ for a given input $\vec { t }$ does not require evaluating $t _ { v } ^ { S }$ over all $2 ^ { d } - 1$ possible nonempty subsets $S \subseteq [ d ]$ . Instead, ordering the input spike times as $t _ { i _ { 1 } } \leq \dots \leq t _ { i _ { d } }$ , the causal set must be a prefix of this ordering. Hence it sufices to consider the d candidate sets of the form $S = \{ i _ { 1 } , \dots , i _ { k } \} , k = 1 , \dots , d .$

Remark 4.1 (Number of regions). Since $[ d ]$ has $2 ^ { d } - 1$ nonempty subsets, the firing-time map has at most $2 ^ { d } - 1$ causal regions, and therefore at most $2 ^ { d } - 1$ linear regions. In fact, this bound is attained whenever $w _ { 1 } , \ldots , w _ { d } > 0$ (Lemma 13 in [22]). In particular, fix any nonempty $S$ and choose $\vec { t } \in \mathbb R ^ { d }$ with $t _ { i } = 0$ for all $i \in S$ and $\begin{array} { r } { \dot { t } _ { j } = c > \frac { \theta _ { v } } { W _ { S } } } \end{array}$ for all $j \not \in S$ Then $\begin{array} { r } { t _ { v } ^ { S } ( \vec { t } ) = \frac { \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } } { W _ { S } } = \frac { \theta _ { v } } { W _ { S } } > 0 } \end{array}$ , and hence $t _ { v } ^ { S } ( \vec { t } ) > t _ { i }$ for all $i \in S$ and $t _ { v } ^ { S } ( \vec { t } ) \leq t _ { j }$ for all $j \not \in S$ . Thus, all causal feasibility inequalities in (7) are satisfied and $\vec { t } \in R _ { S }$ . Therefore, $R _ { S }$ is nonempty for every nonempty $S \subseteq [ d ]$

## 4.2 Causal hyperplanes

The previous subsection shows that the causal regions are determined by the afine inequalities in $( 7 )$ . For the purpose of region counting, it is useful to isolate the hyperplanes describing the boundaries of these constraints. We show that these hyperplanes form an arrangement whose regions refine the causal region partition.

For any $S \subseteq [ d ]$ and any $i \in S$ , we have

$$
t _ { v } ^ { S } ( \vec { t } ) > t _ { i } \Longleftrightarrow \frac { \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } } { W _ { S } } > t _ { i } \Longleftrightarrow \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } - W _ { S } t _ { i } > 0 .\tag{10}
$$

This defines a halfspace whose boundary is the hyperplane

$$
H _ { S , i } ^ { \mathrm { s t r } } : = \{ \vec { t } \in \mathbb { R } ^ { d } : \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } - W _ { S } t _ { i } = 0 \} .\tag{11}
$$

Similarly, for any index $j \not \in S$

$$
t _ { v } ^ { S } ( \vec { t } ) \leq t _ { j } \iff \frac { \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } } { W _ { S } } \leq t _ { j } \iff \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } - W _ { S } t _ { j } \leq 0 .\tag{12}
$$

This defines a halfspace in $\mathbb { R } ^ { d }$ whose boundary is the hyperplane

$$
H _ { S , j } : = \{ \vec { t } \in \mathbb { R } ^ { d } : \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } - W _ { S } t _ { j } = 0 \} .\tag{13}
$$

Starting from a point $\vec { t }$ at which S is feasible, crossing any of these hyperplanes causes $S$ to become infeasible. Moreover, for $| S | \ge 2$ , we have ${ \cal H } _ { S , i } ^ { \mathrm { s t r } } = { \cal H } _ { S \backslash \{ i \} , i } .$ . For $| S | = 1$ , the corresponding equality reduces to $\theta _ { v } = 0$ , which is impossible since $\theta _ { v } > 0$ . Thus, all genuine boundaries are of the form (13). This motivates the following arrangement.

Definition 4.1. The TTFS arrangement is the hyperplane arrangement

$$
\mathcal { A } _ { \mathrm { T T F S } } : = \{ H _ { S , j } : \emptyset \neq S \subseteq [ d ] , j \notin S \} .\tag{14}
$$

We denote by $\mathcal { C } ( A _ { \mathrm { T T F S } } )$ the set of connected components of $\mathbb { R } ^ { d } \setminus \mathcal { A } _ { \mathrm { T T F S } }$

Equivalently, $\mathcal { A } _ { \mathrm { T T F S } }$ consists of the hyperplanes $t _ { v } ^ { S } ( \vec { t } ) = t _ { j }$ obtained by restricting the boundaries $t = t _ { j }$ of the extended linear regions $L _ { S }$ to the threshold level set and expressing them in input-space coordinates. Within each region of $\mathcal { C } ( A _ { \mathrm { T T F S } } )$ , none of the feasibility inequalities change sign. We record this observation in the following lemma.

Lemma 4.1. Let $C \in { \mathcal { C } } ( A _ { \mathrm { T T F S } } )$ . For any nonempty $S \subseteq [ d ]$ , the sign of all inequalities in (7) is constant over ${ \vec { t } } \in C$ . Consequently, there exists a unique nonempty $S \subseteq [ d ]$ such that $C \subseteq R _ { S }$

In particular, the causal set is constant on each region of the hyperplane arrangement. We therefore define the TTFS region map

$$
\ell \colon { \mathcal C } ( \mathcal A _ { \mathrm { T T F S } } ) \to 2 ^ { [ d ] } \setminus \{ \varnothing \} , \qquad \ell ( C ) : = S ,
$$

where $S$ is the unique causal set with $C \subseteq R _ { S }$ . This map is well-defined by Lemma 4.1.

Proposition 4.2. For any nonempty $S \subseteq [ d ]$ , we have

$$
R _ { S } \setminus \bigcup _ { H \in \mathcal { A } _ { \mathrm { T T F S } } } H = \bigcup _ { C \in \mathcal { C } ( \mathcal { A } _ { \mathrm { T T F S } } ) } C .\tag{15}
$$

In particular, every region of the TTFS arrangement is contained in a unique causal region. Thus, the region decomposition induced by $\mathcal { A } _ { \mathrm { T T F S } }$ refines the decomposition into causal regions.

Proof. By Lemma 4.1, the causal set is constant on every region $C \in \mathcal { C } ( \mathcal { A } _ { \mathrm { T T F S } } )$ . Hence, if $\ell ( C ) = S$ , then $C \subseteq R _ { S }$ , which proves the inclusion from right to left in (15).

Conversely, let

$$
\vec { t } \in R _ { S } \setminus \bigcup _ { H \in \mathcal { A } _ { \mathrm { T T F S } } } H .
$$

Since $\vec { t }$ does not lie on any hyperplane of the arrangement, it belongs to a unique region $C \in \mathcal { C } ( \mathcal { A } _ { \mathrm { T T F S } } )$ . Since $\vec { t } \in R _ { S }$ , its causal set is $S ,$ , and therefore $\ell ( C ) = S$ . This proves the reverse inclusion. □

Remark 4.2. The arrangement A<sub>TTFS</sub> typically gives a strict refinement of the causal region partition. Thus, a single causal regions may comprise several regions of the TTFS arrangement. The situation is illustrated in Figure 4, where a causal region comprises several regions of the TTFS arrangement. Nonetheless, the refinement provided by A<sub>TTFS</sub> gives a simple way to upper bound the number of causal regions.

![](images/8bfca49ba012d08055a53fc906999a4ab0fd77f5e931f16c9efcfc2deb53a1a1.jpg)  
(a) d = 2.

![](images/acf7311dc44afb482caed71516fc345afa25ebee7381860690699ab134f003e8.jpg)

![](images/9acfdf98ff54321a8b48e45ecbc03090c2483cd089fc42d85589e6f5844a04b7.jpg)  
(b) d = 3.  
Figure 4: Causal partitions for a single neuron. (a) For $d = 2 .$ , the two boundary lines $t _ { 2 } =$ $t _ { 1 } + \theta / w _ { 1 }$ and $t _ { 2 } = t _ { 1 } { - } \theta / w _ { 2 }$ partition the plane into the three causal regions $R _ { \{ 1 \} } , R _ { \{ 1 , 2 \} } , R _ { \{ 2 \} } ;$ the dashed line $t _ { 2 } ~ = ~ t _ { 1 }$ is redundant for the causal partition. (b) For $d = 3$ , the TTFS arrangement $\mathcal { A } _ { \mathrm { T T F S } }$ consists of 9 distinct hyperplanes, visualized through their intersections with the plane $t _ { 3 } = 0$ . The arrangement regions (left) refine the causal regions (right).

For a layer with multiple neurons, and more generally for a deep network, causal regions can still be described as unions of regions of a large hyperplane arrangement, as in (15). However, the resulting hyperplane decomposition is typically a strict refinement of the causal region partition, making it cumbersome to work with directly. We therefore introduce the notion of a causal pattern of a network. A causal pattern records, for each neuron, the set of presynaptic neurons that fire before it.

We observe that all hyperplanes in $\mathcal { A } _ { \mathrm { T T F S } }$ are translation invariant along the all-ones direction $\mathbf { 1 } = ( 1 , \ldots , 1 ) \in \mathbb { R } ^ { d }$ . Let $e _ { 1 } , \ldots , e _ { d }$ denote the standard basis vectors of $\mathbb { R } ^ { d }$ . The normal vector ${ n } _ { S , j }$ of $H _ { S , j }$ satisfies

$$
n _ { S , j } = \sum _ { i \in S } w _ { i } e _ { i } - W _ { S } e _ { j } , \qquad \langle n _ { S , j } , \mathbf { 1 } \rangle = \sum _ { i \in S } w _ { i } - W _ { S } = 0 .\tag{16}
$$

Thus, every hyperplane in $\mathcal { A } _ { \mathrm { T T F S } }$ is parallel to 1, and the arrangement is invariant under translations $\vec { t } \mapsto \vec { t } + c \mathbf { 1 }$

Accordingly, the arrangement can be essentialized by quotienting out the direction 1 [48]. Identifying the quotient with

$$
T : = \mathbf { 1 } ^ { \perp } = \{ \vec { u } \in \mathbb { R } ^ { d } : \langle \vec { u } , \mathbf { 1 } \rangle = 0 \} \cong \mathbb { R } ^ { d - 1 }\tag{17}
$$

we define the essentialized arrangement $\mathcal { A } _ { \mathrm { T T F S } } ^ { \mathrm { e s s } } : = \{ H ^ { \mathrm { e s s } } = H \cap T colon H \in \mathcal { A } _ { \mathrm { T T F S } } \}$ . The arrangements $A _ { \mathrm { T T F S } } ^ { \mathrm { e s s } }$ and $\mathcal { A } _ { \mathrm { T T F S } }$ have the same region combinatorics.

This invariance has a direct interpretation in terms of spike times. Adding the same constant c to all input spike times shifts the output firing time by c but leaves all causal relations unchanged. Consequently, the causal geometry depends only on relative spike times and can naturally be viewed on the quotient $\mathbb { R } ^ { d } / { \operatorname { s p a n } } \{ 1 \} \cong \mathbb { R } ^ { d - 1 }$ This is precisely the geometry captured by the essentialized arrangement $A _ { \mathrm { T T F S } } ^ { \mathrm { e s s } }$ . This is illustrated in Figure 5.

![](images/9ec1090a753766d2c4b982c6a1b42e15b64c91a43f62d7428f0ee5b27b0242bf.jpg)

Figure 5: For $d = 2 , w _ { 1 } = 2 _ { 2 } = 1 , \theta _ { v } = 1$ , the level set $\textstyle \sum _ { i = 1 } ^ { 2 } \sigma ( - x _ { i } ) = 1$ , where $x _ { i } = t _ { i } - t ,$ consists of three pieces. Projecting along the all-ones direction $\mathbf { 1 } = ( 1 , 1 )$ onto the quotient coordinate $y = x _ { 1 } - x _ { 2 } = t _ { 1 } - t _ { 2 }$ maps these pieces to the three causal regions in the quotient input space.

## 4.3 The lifted TTFS configuration and polytope

We present a polyhedral geometry description of the firing-time map ${ \vec { t } } \mapsto t _ { v } ( { \vec { t } } )$ . Our goal is to provide a compact representation of the piecewise-afine structure of firing-time map and connect the TTFS neuron model to familiar constructions from polyhedral and tropical geometry. We present the relevant polyhedral notions at a descriptive level and refer the reader to [27] for a systematic treatment of regular subdivisions, duality, and related constructions.

Recall from Proposition 4.1 that $\begin{array} { r } { t _ { v } ( \vec { t } ) = \operatorname* { m i n } _ { \emptyset \neq S \subseteq [ d ] } t _ { v } ^ { S } ( \vec { t } ) } \end{array}$ , where

$$
t _ { v } ^ { S } ( \vec { t } ) = \langle \vec { a } _ { S } , \vec { t } \rangle + b _ { S } , \quad \mathrm { w h e r e ~ } \vec { a } _ { S } : = \sum _ { i \in S } \frac { w _ { i } } { W _ { S } } \vec { e } _ { i } , \quad b _ { S } : = \frac { \theta _ { v } } { W _ { S } } , \quad W _ { S } : = \sum _ { i \in S } w _ { i } .\tag{18}
$$

It is convenient to work with the convex piecewise-afine function

$$
f ( \vec { t } ) : = - t _ { v } ( \vec { t } ) = \operatorname* { m a x } _ { \emptyset \neq S \subseteq [ d ] } ( - \langle \vec { a } _ { S } , \vec { t } \rangle - b _ { S } ) .\tag{19}
$$

Definition 4.2 (Lifted TTFS configuration and polytope). For every nonempty $S \subseteq [ d ]$ define the projected coeficient point and lifted coeficient point

$$
\begin{array} { r l r } & { } & { \vec { p } _ { S } : = - \vec { a } _ { S } \in \mathbb { R } ^ { d } , \qquad \vec { y } _ { S } : = ( - \vec { a } _ { S } , - b _ { S } ) = ( \vec { p } _ { S } , - b _ { S } ) \in \mathbb { R } ^ { d + 1 } . } \end{array}
$$

The collection

$$
\mathcal { V } ( w , \theta _ { v } ) : = \{ \vec { y } _ { S } : \emptyset \neq S \subseteq [ d ] \}
$$

is the lifted TTFS configuration, and

$$
Q ( w , \theta _ { v } ) : = \mathrm { { c o n v } } \{ \vec { y } _ { S } : \emptyset \neq S \subseteq [ d ] \} \subseteq \mathbb { R } ^ { d + 1 }
$$

is the lifted TTFS polytope.

The connection between $Q$ and $f$ can be seen directly through the support function of $Q .$ defined as $h _ { Q } ( \vec { z } ) = \operatorname* { m a x } _ { \vec { y } \in Q } \langle \vec { z } , \vec { y } \rangle$ . For $\vec { z } = ( \vec { t } , 1 )$ , we obtain

$$
h _ { \cal Q } ( ( \vec { t } , 1 ) ) = \operatorname* { m a x } _ { \emptyset \neq S \subseteq [ d ] } ( - \langle \vec { a } _ { S } , \vec { t } \rangle - b _ { S } ) = f ( \vec { t } ) .
$$

Thus, $f$ is obtained by restricting the support function of $Q$ to points whose last coordinate is 1. This representation gives a direct correspondence between the polyhedral geometry of $Q$ and the piecewise-afine structure of $f .$ For a given input $\vec { t , }$ the afine pieces active at $\dot { \vec { t } }$ are precisely those whose lifted coeficient vectors ${ \vec { y } } _ { S }$ lie on the face of $Q$ exposed by the direction $( \vec { t } , 1 )$ . In particular, if the active afine piece is unique, the corresponding ${ \vec { y } } _ { S }$ is an exposed vertex of $Q .$ . Consequently, the upper faces of $Q ,$ namely those exposed by directions with last coordinate equal to 1, encode the piecewise-afine structure of $f .$

Structure of the polytope The polytope $Q$ has a highly constrained structure. First observe that the coeficient vectors ${ \vec { a } } _ { S }$ satisfy $( \vec { a } _ { S } ) _ { i } \geq 0 , \sum _ { i = 1 } ^ { d } ( \vec { a } _ { S } ) _ { i } = 1$ . Thus, every ${ \vec { a } } _ { S }$ lies in the standard simplex $\Delta _ { d - 1 } = \mathrm { c o n v } \{ \vec { e } _ { 1 } , \dots , \vec { e } _ { d } \}$ . More precisely, ${ \vec { a } } _ { S }$ lies in the relative interior of the face conv $\{ \vec { e } _ { i } : i \in S \}$ . Since $\vec { a } _ { \{ i \} } = \vec { e } _ { i }$ , the convex hull of all points ${ \vec { p } } _ { S } = - { \vec { a } } _ { S }$ is exactly the simplex:

$$
\mathrm { c o n v } \{ \vec { p } _ { S } \colon \emptyset \neq S \subseteq [ d ] \} = - \Delta _ { d - 1 } .
$$

Moreover, all lifted points lie in the afine hyperplane

$$
H : = \left\{ ( \vec { p } , \beta ) \in \mathbb { R } ^ { d } \times \mathbb { R } : \langle \vec { p } , \mathbf { 1 } \rangle = - 1 \right\} .\tag{20}
$$

Thus $Q$ has afine dimension at most $d ,$ although it is naturally embedded in $\mathbb { R } ^ { d + 1 }$ . This constraint reflects the translation invariance of the firing-time map, whereby adding the same constant to all input spike times shifts the firing time by the same constant without changing the causal structure.

The structure is considerably more rigid than the elementary observations above. To discuss this, we introduce the scale-free parameters

$$
\lambda _ { i } : = \frac { w _ { i } } { \theta _ { v } } > 0 , \qquad \lambda _ { S } : = \sum _ { i \in { \cal S } } \lambda _ { i } .\tag{21}
$$

Then

$$
\vec { a } _ { S } = \frac { 1 } { \lambda _ { S } } \sum _ { i \in S } \lambda _ { i } \vec { e } _ { i } , \qquad b _ { S } = \frac { 1 } { \lambda _ { S } } ,\tag{22}
$$

and therefore

$$
\vec { y } _ { S } = \left( - \frac { \sum _ { i \in S } \lambda _ { i } \vec { e } _ { i } } { \lambda _ { S } } , - \frac { 1 } { \lambda _ { S } } \right) .\tag{23}
$$

The following theorem gives an intrinsic geometric description of $Q$

Theorem 4.1 (Structure of the lifted TTFS polytope). Let $Q = Q ( w , \theta _ { v } )$ be the lifted TTFS polytope of a neuron with d positive input weights, and let $\lambda _ { i } = w _ { i } / \theta _ { v }$ . Define the weighted box $B _ { \lambda } : = \times _ { i = 1 } ^ { d } [ 0 , \lambda _ { i } ] \subseteq \mathbb { R } ^ { d }$ and, for $S \subseteq [ d ]$ , its vertices $\begin{array} { r } { \vec { x } _ { S } : = \sum _ { i \in S } \lambda _ { i } \vec { e } _ { i } } \end{array}$ . Then:

1. The lifted TTFS points are obtained from the nonzero vertices ${ \vec { x } } _ { S }$ by first embedding them into $\mathbb { R } ^ { d + 1 }$ via ${ \vec { x } } \mapsto ( { \vec { x } } , 1 )$ and then centrally projecting from the origin onto

$$
H = \{ ( \vec { p } , \beta ) \in \mathbb { R } ^ { d + 1 } : \langle \vec { p } , \mathbf { 1 } \rangle = - 1 \} .
$$

Explicitly,

$$
\vec { y } _ { S } = - \frac { ( \vec { x } _ { S } , 1 ) } { \langle \vec { x } _ { S } , \mathbf { 1 } \rangle } .
$$

![](images/02e49e5d1116b57b99e1fd6596210070a2363426626487f56b46793a3a3c0b6a.jpg)  
Figure 6: Illustration of Theorem 4.1. The left panel shows $d = 2$ . The lifted box $B _ { \lambda }$ (orange) in $\mathbb { R } ^ { d + 1 }$ . The nonzero vertices are mapped by central projection from origin onto the afine hyperplane H (blue) to produce the TTFS configuration. The right panel shows $d = 3$ Shown is the projection onto the first d coordinates, illustrating the projected configuration without the height.

2. The upper hull of Q is the graph over $- \Delta _ { d - 1 }$ of the concave piecewise-linear function

$$
g _ { \lambda } ( \vec { p } ) : = \operatorname* { m i n } _ { i \in [ d ] } \frac { p _ { i } } { \lambda _ { i } } .
$$

Equivalently, its hypograph is the polyhedron

$$
\widehat { Q } _ { \lambda } : = \{ ( \vec { p } , \beta ) \in H : \lambda _ { i } \beta \leq p _ { i } \leq 0 , i \in [ d ] \} = Q + \mathbb { R } _ { \geq 0 } ( 0 , \ldots , 0 , - 1 ) .\tag{24}
$$

3. The vertices $o f Q$ are naturally indexed by the $2 ^ { d } - 1$ nonempty subsets $S \subseteq [ d ]$ . The vertex $\vec { y } _ { S } = ( \vec { p } , \beta )$ indexed by S is obtained by making exactly one of the two inequalities $\lambda _ { i } \beta \le p _ { i } \le 0$ tight for every $i \in [ d ]$ , with $p _ { i } = \lambda _ { i } \beta \ i f f i \in S$ and $p _ { i } = 0 ~ i f f i \notin S$

Theorem 4.1 shows that the exponentially many vertices of $Q$ arise from a simple construction: they are a perspective image of the nonzero vertices of an axis-aligned box. This is illustrated in Figure 6. In particular, the $2 ^ { d } - 1$ lifted points are governed by only d positive parameters.

Proof of Theorem $4 . 1 .$ For every nonempty $S \subseteq [ d ]$

$$
\langle { \vec { x } } _ { S } , \mathbf { 1 } \rangle = \sum _ { i \in S } \lambda _ { i } = \lambda _ { S } .
$$

After the embedding $\vec { x } _ { S } \mapsto ( \vec { x } _ { S } , 1 )$ , the line through the origin and $( \vec { x } _ { S } , 1 )$ consists of the points $c ( \vec { x } _ { S } , 1 ) , c \in \mathbb { R }$ . Its intersection with H is determined by $c \langle \vec { x } _ { S } , \mathbf { 1 } \rangle = - 1$ , and hence by $\begin{array} { r } { c = - \frac { 1 } { \lambda _ { S } } } \end{array}$ . The resulting point is

$$
- \frac { ( \vec { x } _ { S } , 1 ) } { \langle \vec { x } _ { S } , \mathbf { 1 } \rangle } = \left( - \frac { \sum _ { i \in S } \lambda _ { i } \vec { e } _ { i } } { \lambda _ { S } } , - \frac { 1 } { \lambda _ { S } } \right) = \vec { y } _ { S } ,
$$

where the last equality follows from (23). This proves the first claim.

We next consider $\widehat { Q } _ { \lambda }$ . Since $( { \vec { p } } , \beta ) \in H$ means $\begin{array} { r } { \sum _ { i } p _ { i } = - 1 } \end{array}$ , the constraints $p _ { i } \leq 0 , i \in [ d ]$ are equivalent to $\vec { p } \in - \Delta _ { d - 1 }$ . Moreover, since $\lambda _ { i } > 0$ , the condition $\lambda _ { i } \beta \le p _ { i }$ is equivalent to $\beta \leq \frac { p _ { i } } { \lambda _ { i } }$ . It follows that

$$
( \vec { p } , \beta ) \in \widehat { Q } _ { \lambda } \quad \mathrm { i f ~ a n d ~ o n l y ~ i f } \quad \vec { p } \in - \Delta _ { d - 1 } \quad \mathrm { a n d } \quad \beta \leq \operatorname* { m i n } _ { i \in [ d ] } \frac { p _ { i } } { \lambda _ { i } } = g _ { \lambda } ( \vec { p } ) .
$$

Thus $\widehat { Q } _ { \lambda }$ is precisely the hypograph of $g _ { \lambda }$ over $- \Delta _ { d - 1 }$

We now determine the vertices. Let $( \vec { p } , \beta )$ be a finite vertex of $\widehat { Q } _ { \lambda }$ . For each $i \in [ d ]$ at least one of the two inequalities $\lambda _ { i } \beta \le p _ { i } \le 0$ must be tight. Indeed, if for some i both inequalities were strict, then the active inequalities could involve at most the remaining $d - 1$ coordinates. Their normals therefore could not span the d-dimensional tangent space of $H$ contradicting that $( \vec { p } , \beta )$ is a vertex. The two inequalities cannot both be tight for any i. Otherwise $p _ { i } = 0 = \lambda _ { i } \beta _ { }$ , so $\beta = 0$ . The inequalities $\lambda _ { j } \beta \le p _ { j } \le 0$ would then imply $p _ { j } = 0$ for every $j ,$ contradicting $\begin{array} { r } { \sum _ { j } p _ { j } = - 1 } \end{array}$ . Hence exactly one of the two inequalities is tight for each i. Define $S : = \{ i \in [ d ] : ^ { \prime } p _ { i } = \lambda _ { i } \beta \}$ . Then

$$
p _ { i } = \left\{ { \begin{array} { l l } { \lambda _ { i } \beta , } & { i \in S , } \\ { 0 , } & { i \not \in S . } \end{array} } \right.
$$

The set $S$ is nonempty, since otherwise $\vec { p } = 0$ . Using $\textstyle \sum _ { i } p _ { i } = - 1$ gives

$$
- 1 = \sum _ { i = 1 } ^ { d } p _ { i } = \beta \sum _ { i \in S } \lambda _ { i } = \beta \lambda _ { S } ,
$$

and therefore

$$
\beta = - \frac { 1 } { \lambda _ { S } } , \qquad \vec { p } = - \frac { 1 } { \lambda _ { S } } \sum _ { i \in S } \lambda _ { i } \vec { e } _ { i } .
$$

Thus

$$
( \vec { p } , \beta ) = \vec { y } _ { S } .
$$

Conversely, for every nonempty $S \subseteq [ d ]$ , the point ${ \vec { y } } _ { S }$ satisfies

$$
p _ { i } = \lambda _ { i } \beta , \quad i \in { \cal S } , \qquad p _ { i } = 0 , \quad i \notin { \cal S } .
$$

These d equalities, together with the afine equation $\begin{array} { r } { \sum _ { i } p _ { i } = - 1 } \end{array}$ , uniquely determine $( \vec { p } , \beta )$ hence ${ \vec { y } } _ { S }$ is a vertex of $\widehat { Q } _ { \lambda }$ . Therefore the finite vertices of $\widehat { Q } _ { \lambda }$ are precisely the $2 ^ { d } - 1$ points ${ \vec { y } } _ { S } , \emptyset \neq S \subseteq [ d ]$

It remains to determine the recession cone of $\widehat { Q } _ { \lambda }$ . A vector $( \vec { r } , \rho ) \in \mathbb R ^ { d + 1 }$ is a recession direction if

$$
( \vec { p } , \beta ) + t ( \vec { r } , \rho ) \in \widehat { Q } _ { \lambda } \qquad \mathrm { f o r ~ a l l ~ } t \geq 0 \qquad \mathrm { w h e n e v e r } ~ ( \vec { p } , \beta ) \in \widehat { Q } _ { \lambda } .
$$

Since $\widehat { Q } _ { \lambda } \subset H$ , this requires $\begin{array} { r } { \sum _ { i = 1 } ^ { d } r _ { i } = 0 } \end{array}$ . Moreover, preserving the inequalities $\lambda _ { i } \beta \le p _ { i } \le 0$ for all $t \geq 0$ requires

$$
\lambda _ { i } \rho \leq r _ { i } \leq 0 , \qquad i \in [ d ] .
$$

The conditions $r _ { i } \le 0$ for every i and $\textstyle \sum _ { i } r _ { i } = 0$ , imply $r _ { i } = 0$ for all $i ,$ and thus $\rho \le 0$ . Hence

$$
\operatorname { r e c } ( { \widehat { Q } } _ { \lambda } ) = \mathbb { R } _ { \geq 0 } ( 0 , \dots , 0 , - 1 ) .
$$

Since the finite vertices of $\widehat { Q } _ { \lambda }$ are precisely the points ${ \vec { y } } _ { S }$ , the Minkowski-Weyl theorem gives

$$
\widehat { Q } _ { \lambda } = \operatorname { c o n v } \{ \vec { y } _ { S } : \vartheta \neq S \subseteq [ d ] \} + \mathbb { R } _ { \geq 0 } ( 0 , \ldots , 0 , - 1 ) = Q + \mathbb { R } _ { \geq 0 } ( 0 , \ldots , 0 , - 1 ) .
$$

Since the recession cone is vertically downward, the upper boundary of $\widehat { Q } _ { \lambda }$ is the upper hull of $Q$ . Since $\widehat { Q } _ { \lambda }$ is the hypograph of $g _ { \lambda }$ , this upper boundary is precisely the graph of $g _ { \lambda } . \square$

Constraints on the lifted TTFS configuration The next result makes the rigidity of the TTFS configuration explicit by characterizing the algebraic relations among the lifted points.

Theorem 4.2 (Constraints among the lifted TTFS points). Let

$$
\mathcal { V } = \{ \vec { y } _ { S } = ( \vec { p } _ { S } , \beta _ { S } ) : \emptyset \neq S \subseteq [ d ] \} \subset H
$$

be a collection of points with $\beta _ { S } < 0$ . Then Y is the lifted TTFS configuration of a neuron with positive weights and positive threshold if and only if the following relations hold:

$$
\vec { p _ { \{ i \} } } = - \vec { e _ { i } } ,
$$

$$
i \in [ d ] ,\tag{25}
$$

$$
p _ { S , i } = 0 ,
$$

$$
i \not \in S ,\tag{26}
$$

$$
p _ { S , i } \beta _ { \{ i \} } = - \beta _ { S } ,
$$

$$
i \in S ,\tag{27}
$$

$$
\frac { 1 } { \beta _ { S } } = \sum _ { i \in S } \frac { 1 } { \beta _ { \{ i \} } } ,
$$

$$
\varnothing \neq S \subseteq [ d ] .\tag{28}
$$

From Theorem 4.2 we see that, setting $\begin{array} { r } { r _ { S } : = - \frac { 1 } { \beta _ { S } } > 0 , r _ { \emptyset } : = 0 } \end{array}$ , the reciprocal heights form a strictly positive modular set function:

$$
r _ { S } = \sum _ { i \in S } r _ { \{ i \} } ,\tag{29}
$$

and therefore,

$$
r _ { S } + r _ { T } = r _ { S \cup T } + r _ { S \cap T } \qquad { \mathrm { f o r ~ a l l ~ } } S , T \subseteq [ d ] .\tag{30}
$$

Moreover, setting $\begin{array} { r } { \vec { q } _ { S } : = \frac { \vec { y } _ { S } } { \beta _ { S } } } \end{array}$ and adjoining ${ \vec { q } } _ { \emptyset } : = ( 0 , \dots , 0 , 1 )$ ), the normalized points satisfy the modular relations

$$
{ \vec { q } } _ { S } + { \vec { q } } _ { T } = { \vec { q } } _ { S \cup T } + { \vec { q } } _ { S \cap T } \qquad { \mathrm { ~ f o r ~ a l l ~ } } S , T \subseteq [ d ] .\tag{31}
$$

Indeed, since $\begin{array} { r } { \vec { q } _ { S } = \frac { \vec { y } _ { S } } { \beta _ { S } } = \left( \sum _ { i \in S } \lambda _ { i } \vec { e } _ { i } , 1 \right) } \end{array}$ , one has

$$
\begin{array} { l } { \displaystyle { \vec { q } _ { S } + \vec { q } _ { T } = \left( \sum _ { i } \lambda _ { i } ( \mathbf { 1 } _ { \{ i \in S \} } + \mathbf { 1 } _ { \{ i \in T \} } ) \vec { e } _ { i } , 2 \right) } } \\ { \displaystyle { = \left( \sum _ { i } \lambda _ { i } ( \mathbf { 1 } _ { \{ i \in S \cup T \} } + \mathbf { 1 } _ { \{ i \in S \cap T \} } ) \vec { e } _ { i } , 2 \right) } } \\ { \displaystyle { = \vec { q } _ { S \cup T } + \vec { q } _ { S \cap T } } . } \end{array}
$$

Proof of Theorem 4.2. Suppose first that Y is the lifted TTFS configuration of a neuron. For a singleton $S = \{ i \}$ , (18) gives $\vec { a } _ { \{ i \} } = \vec { e } _ { i }$ and hence $\vec { p _ { \{ i \} } } = - \vec { e _ { i } }$ , which proves (25). By (23), $\beta _ { S } = - \frac { 1 } { \lambda _ { S } }$ and $p _ { S , i } = \left\{ { \begin{array} { l l } { - { \frac { \lambda _ { i } } { \lambda _ { S } } } , } & { i \in S } \\ { 0 , } & { i \notin S } \end{array} } \right.$ , which proves (26). Moreover, $\begin{array} { r } { \beta _ { \{ i \} } = - \frac { 1 } { \lambda _ { i } } } \end{array}$ , and hence, for $i \in S$

$$
p _ { S , i } \beta _ { \left\{ i \right\} } = \left( - \frac { \lambda _ { i } } { \lambda _ { S } } \right) \left( - \frac { 1 } { \lambda _ { i } } \right) = \frac { 1 } { \lambda _ { S } } = - \beta _ { S } ,
$$

which proves (27). Furthermore, $\begin{array} { r } { \frac { 1 } { \beta _ { S } } = - \lambda _ { S } = - \sum _ { i \in S } \lambda _ { i } = \sum _ { i \in S } \frac { 1 } { \beta _ { \left\{ i \right\} } } } \end{array}$ , proving (28).

Conversely, suppose that the stated constraints hold. Define $\begin{array} { r } { \lambda _ { i } : = - \frac { 1 } { \beta _ { \{ i \} } } > 0 } \end{array}$ . By (28), $\begin{array} { r } { \frac { 1 } { \beta _ { S } } = - \sum _ { i \in S } \lambda _ { i } } \end{array}$ , and therefore

$$
\beta _ { S } = - \frac { 1 } { \sum _ { i \in S } \lambda _ { i } } = - \frac { 1 } { \lambda _ { S } } .
$$

For $i \in S , ( 2 7 )$ gives $\begin{array} { r } { p s , i = - { \frac { \beta _ { S } } { \beta _ { \{ i \} } } } = - { \frac { \lambda _ { i } } { \lambda _ { S } } } } \end{array}$ , whereas (26) gives $p _ { S , i } = 0$ for $i \not \in S$ . Hence

$$
\vec { y } _ { S } = \left( - \frac { \sum _ { i \in S } \lambda _ { i } \vec { e } _ { i } } { \lambda _ { S } } , - \frac { 1 } { \lambda _ { S } } \right) .
$$

Choosing, for example, $\theta _ { v } = 1 , w _ { i } = \lambda _ { i } ,$ recovers exactly the given configuration from Definition 4.2. Thus Y is a lifted TTFS configuration. □

Degrees of freedom Theorem 4.2 makes clear that the $2 ^ { d } - 1$ lifted points are far from independent.

A lifted TTFS configuration has only d degrees of freedom. Indeed, the neuron is parametrized by $( w _ { 1 } , \dots , w _ { d } , \theta _ { v } ) \in \mathbb R _ { > 0 } ^ { d + 1 }$ , but the simultaneous rescaling $( w _ { 1 } , \dots , w _ { d } , \theta _ { v } ) \mapsto$ $( c w _ { 1 } , \ldots , c w _ { d } , c \theta _ { v } ) , c > 0$ , leaves every ${ \vec { a } } _ { S }$ and $b _ { S }$ unchanged. Thus the lifted configuration depends only on the d ratios

$$
\lambda _ { i } = \frac { w _ { i } } { \theta _ { v } } , \qquad i \in [ d ] .
$$

Equivalently, it is completely determined by the d singleton heights

$$
\beta _ { \{ i \} } = - \frac { \theta _ { v } } { w _ { i } } .
$$

Once these are fixed, every nonsingleton point is forced:

$$
\beta _ { S } = \left( \sum _ { i \in S } \frac { 1 } { \beta _ { \left\{ i \right\} } } \right) ^ { - 1 } , \qquad p _ { S , i } = \left\{ - \frac { \beta _ { S } } { \beta _ { \left\{ i \right\} } } , \quad i \in S , \right.\tag{32}
$$

By contrast, an arbitrary collection of $2 ^ { d } - 1$ points in the d-dimensional afine space H has $d ( 2 ^ { d } - 1 )$ degrees of freedom. The subset of such configurations subject to support structure compatibility has $d 2 ^ { d - 1 }$ degrees of freedom. For a point indexed by S, the conditions $p _ { S , i } = 0 , i \notin S$ , and $\begin{array} { r } { \sum _ { i \in S } p _ { S , i } = - 1 } \end{array}$ leave $\lvert S \rvert - 1$ degrees of freedom in ${ \vec { p } } { \cal { S } }$ , while the height $\beta _ { S }$ contributes one additional degree of freedom. Thus an arbitrary support-compatible point ${ \vec { y } } _ { S }$ has |S| degrees of freedom, and the sum over all nonempty subsets gives $\scriptstyle \sum _ { \emptyset \neq S \subseteq [ d ] } | S | = d 2 ^ { d - 1 }$

![](images/7a78680c6c515409eb48aef947e66d79986fc176b345efe5280dab9e3917a35c.jpg)  
(a) Lifted polytope.

![](images/272536d1a975e6ed5226e87c48b3e2fe644b0b2435cc0067ca426ce27a131abb.jpg)  
(c) Dual complex.  
Figure 7: Polyhedral description of a single TTFS neuron for $d = 3$ and $w _ { 1 } = w _ { 2 } = w _ { 3 } =$ $\theta _ { v } = 1$ . (a) The lifted TTFS polytope Q, shown in the afine hyperplane $\{ ( \vec { u } , b ) \colon \langle \vec { u } , \mathbf { 1 } \rangle = 1 \}$ (b) Projecting its upper faces gives a regular subdivision of the TTFS polytope. (c) The causal region partition is the dual complex of the polytope subdivision. The three edges joining $\scriptstyle p _ { \{ 1 , 2 , 3 \} }$ to $p _ { \{ 1 , 2 \} } , p _ { \{ 1 , 3 \} } , p _ { \{ 2 , 3 \} }$ correspond to the three sides of the region $R _ { \{ 1 , 2 , 3 \} }$ , while the six outer boundary segments correspond to the remaining boundaries of the unbounded regions $R _ { \{ 1 \} } , R _ { \{ 2 \} } , R _ { \{ 3 \} } , R _ { \{ 1 , 2 \} } , R _ { \{ 1 , 3 \} } , R _ { \{ 2 , 3 \} }$ . The resulting seven regions are $R _ { S } , \emptyset \neq S \subseteq [ 3 ]$ 2 recovering the causal partition shown previously in Figure 4b.

## 4.4 Regular subdivision and causal complex

The lifting induces a regular subdivision of the coeficient simplex. Specifically, projecting the upper faces of Q onto the first d coordinates gives a subdivision of $- \Delta _ { d - 1 }$ . This subdivision is dual to the piecewise-afine decomposition of the domain of $f \colon$ vertices of the subdivision correspond to full-dimensional linear regions of $f ,$ while edges correspond to boundaries between adjacent linear regions. Since the linear regions of $f = - t _ { v }$ are the causal regions of the firing-time map, this duality provides a polyhedral representation of the causal region complex.

Figure 7 illustrates this construction for a neuron with three inputs, $d = 3$ . The lifted coeficient polytope Q shown in Figure 7a has one vertex ${ \vec { y } } _ { S }$ for each nonempty subset $S \subseteq [ 3 ]$ The vertices satisfy afine relations, so some facets of $Q$ are non-simplicial. Figure 7b shows the corresponding regular subdivision, obtained by projecting the upper faces of $Q$ onto the first d coordinates. The projected points are $p _ { S } = - { \vec { a } } _ { S }$ , whose convex hull is $- \Delta _ { d - 1 }$ For $d = 3$ with equal weights, the singleton subsets correspond to the three vertices of the triangle, the two-element subsets to the edge midpoints, and {1, 2, 3} to the centroid. The locations of these points are determined by the weights, while their heights $- b _ { S }$ determine the upper faces of Q and hence the resulting regular subdivision. Figure 7c shows the dual causal-region complex for the same example. In particular, the internal Y -shaped structure in the regular subdivision is dual to the three boundaries of the central causal region $R _ { \{ 1 , 2 , 3 \} }$ while the subdivided outer edges are dual to the boundaries of the surrounding unbounded causal regions.

This correspondence also gives a simple geometric interpretation of the causal boundaries. If two afine pieces indexed by S and T meet, their common boundary is determined by the equality $t _ { v } ^ { S } ( \vec { t } ) = t _ { v } ^ { T } ( \vec { t } )$ , or equivalently, $\langle \vec { a } _ { S } - \vec { a } _ { T } , \vec { t } \rangle + ( b _ { S } - b _ { T } ) = 0$ . Thus, the diference $\vec { a } _ { S } - \vec { a } _ { T }$ determines the orientation of the boundary, while $b _ { S } - b _ { T }$ determines its position. Since the vectors ${ \vec { a } } _ { S }$ depend only on weight ratios, the weights determine the orientations of the causal boundaries, whereas the threshold enters through $b _ { S } = \theta _ { v } / W _ { S }$ and controls their ofsets. In particular, for fixed weights, scaling $\theta _ { v }$ by a factor $c > 0$ scales the causal partition by the same factor. Conversely, scaling all weights and $\theta _ { v }$ by the same positive factor leaves every ${ \vec { a } } _ { S }$ and $b _ { S }$ unchanged, and therefore leaves the firing-time map and its causal geometry unchanged.

Theorem 4.3 (Structure of the regular subdivision and dual causal complex). Let $\Sigma _ { \lambda }$ denote the regular subdivision $o f - \Delta _ { d - 1 }$ induced by the upper hull of $Q$ . For $\emptyset \neq I \subseteq J \subseteq [ d ]$ , let $C _ { I , J } : = \mathrm { c o n v } \{ \vec { p } _ { S } : I \subseteq S \subseteq J \}$ . Then:

1. The cells of $\Sigma _ { \lambda }$ are precisely the polytopes $C _ { I , J } , \varnothing \neq I \subseteq J \subseteq [ d ]$ . Moreover, dim $C _ { I , J } =$ $\left| J \right| { - } { \left| I \right| }$ . In particular, the maximal cells are $C _ { \{ i \} , [ d ] } , ~ i ~ \in ~ [ d ]$ , while the vertices are $C _ { S , S } = \{ \vec { p _ { S } } \} , \emptyset \neq S \subseteq [ d ]$

2. Each $C _ { I , J }$ is the central projection of the face

$$
F _ { I , J } : = \{ { \vec { x } } \in B _ { \lambda } : x _ { i } = \lambda _ { i } ~ f o r ~ i \in I , \quad x _ { i } = 0 ~ f o r ~ i \not \in J \}
$$

of the weighted box $B _ { \lambda }$ . Consequently, $\Sigma _ { \lambda }$ is combinatorially isomorphic to the subcomplex of the boundary of $B _ { \lambda }$ consisting of the faces that do not contain the origin. In particular, every cell $C _ { I , J }$ is combinatorially a cube of dimension $\left| J \right| { - } { \left| I \right| }$

3. The dual cell $C _ { I , J } ^ { * }$ in the causal-region complex has codimension $\left| J \right| { - } { \left| I \right| }$ . Its relative interior consists precisely of the inputs for which, writing $\tau = t _ { v } ( \vec { t } )$

$$
t _ { i } < \tau \quad ( i \in I ) , \qquad t _ { i } = \tau \quad ( i \in J \setminus I ) , \qquad t _ { i } > \tau \quad ( i \notin J ) .
$$

Equivalently, the afine pieces indexed by $I \subseteq S \subseteq J$ are exactly the pieces that are simultaneously active on relint $( C _ { I , J } ^ { * } )$

Proof of Theorem 4.3. For $\emptyset \neq I \subseteq J \subseteq [ d ]$ , consider the face

$$
{ \cal F } _ { I , J } = \{ \vec { x } \in { \cal B } _ { \lambda } : x _ { i } = \lambda _ { i } \mathrm { f o r } i \in I , x _ { i } = 0 \mathrm { f o r } i \notin J \} .
$$

Its free coordinates are precisely those indexed by $J \backslash I ,$ , and hence dim $F _ { I , J } = | J | { - } | I |$ Moreover, its vertices are $\begin{array} { r } { \vec { x } _ { S } = \sum _ { i \in S } \lambda _ { i } \vec { e _ { i } } , I \subseteq S \subseteq J } \end{array}$ . Since $I \neq \emptyset$ , the face $F _ { I , J }$ does not contain the origin. Therefore the central projection $\pi ( \vec { x } ) = - \frac { \vec { x } } { \langle \vec { x } , \mathbf { 1 } \rangle }$ is well defined on $F _ { I , J }$ and maps it projectively onto

$$
C _ { I , J } = \mathrm { c o n v } \{ \vec { p } _ { S } : I \subseteq S \subseteq J \} .
$$

In particular, dim $C _ { I , J } = | J | { - } | I |$ , and $C _ { I , J }$ is combinatorially a cube of that dimension.

It remains to show that these are precisely the cells of the regular subdivision. By Theorem 4.1, the upper hull of $Q$ is the graph over $- \Delta _ { d - 1 }$ of $\begin{array} { r } { g _ { \lambda } ( \vec { p } ) = \operatorname* { m i n } _ { i \in [ d ] } \frac { p _ { i } } { \lambda _ { i } } } \end{array}$ . For $\vec { p } \in - \Delta _ { d - 1 }$ 2 let

$$
I ( \vec { p } ) = \left\{ i : \frac { p _ { i } } { \lambda _ { i } } = g _ { \lambda } ( \vec { p } ) \right\} , \qquad J ( \vec { p } ) = \{ i : p _ { i } < 0 \} .
$$

Since $\textstyle \sum _ { i } p _ { i } = - 1$ , both sets are nonempty and $I ( \vec { p } ) \subseteq J ( \vec { p } )$ . The relative interior of the region on which these two sets are fixed is characterized by

$$
\frac { p _ { i } } { \lambda _ { i } } = g _ { \lambda } ( \vec { p } ) \quad ( i \in I ) , \qquad g _ { \lambda } ( \vec { p } ) < \frac { p _ { i } } { \lambda _ { i } } < 0 \quad ( i \in J \setminus I ) , \qquad p _ { i } = 0 \quad ( i \notin J ) .
$$

Its closure is exactly $C _ { I , J }$ . Hence the cells of the regular subdivision are precisely the $C _ { I , J }$ We now identify the dual cells. Let

$$
\tau = t _ { v } ( \vec { t } ) , \qquad I = \{ i : t _ { i } < \tau \} , \qquad J = \{ i : t _ { i } \leq \tau \} .
$$

At the firing time, $\begin{array} { r } { \theta _ { v } = \sum _ { i \in I } w _ { i } ( \tau - t _ { i } ) } \end{array}$ . If $I \subseteq S \subseteq J $ , then every index in $S \setminus I$ satisfies $t _ { i } = \tau$ , and therefore $\begin{array} { r } { \theta _ { v } = \sum _ { i \in S } w _ { i } ( \tau - t _ { i } ) } \end{array}$ . Rearranging gives $\begin{array} { r } { \tau = \frac { \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } } { W _ { S } } = t _ { v } ^ { S } ( \vec { t } ) } \end{array}$ . Thus every afine piece indexed by S with $I \subseteq S \subseteq J$ is active at $\vec { t . }$

Conversely, suppose $t _ { v } ^ { S } ( \vec { t } ) = \tau$ . Then $\begin{array} { r } { \theta _ { v } = \sum _ { i \in S } w _ { i } ( \tau - t _ { i } ) } \end{array}$ . Comparing this with $\theta _ { v } =$ $\begin{array} { r } { \sum _ { i : t _ { i } < \tau } w _ { i } ( \tau - t _ { i } ) } \end{array}$ and using $w _ { i } > 0$ , we see that every index with $t _ { i } < \tau$ must belong to $S ,$ while no index with $t _ { i } > \tau$ can belong to S. Hence $I \subseteq S \subseteq J$ . Therefore the active afine pieces are exactly those indexed by the Boolean interval

$$
[ I , J ] = \{ S : I \subseteq S \subseteq J \} .
$$

By polyhedral duality, the cell dual to $C _ { I , J }$ therefore consists of the inputs satisfying

$$
t _ { i } < \tau \quad ( i \in I ) , \qquad t _ { i } = \tau \quad ( i \in J \setminus I ) , \qquad t _ { i } > \tau \quad ( i \notin J ) ,
$$

with $\tau = t _ { v } ( \vec { t } )$ . Since dim $C _ { I , J } = | J | { - } | I |$ , the dual cell has codimension $\left| J \right| { - } { \left| I \right| }$ , completing the proof. □

Corollary 4.1 (Explicit description of the causal complex). For every nonempty $S \subseteq [ d ]$ , let

$$
t _ { v } ^ { S } ( \vec { t } ) : = \frac { \theta _ { v } + \sum _ { i \in S } w _ { i } t _ { i } } { W _ { S } } , \qquad W _ { S } = \sum _ { i \in S } w _ { i } ,
$$

and, for $j \in [ d ]$

$$
\phi _ { S , j } ( \vec { t } ) : = W _ { S } t _ { j } - \sum _ { i \in S } w _ { i } t _ { i } - \theta _ { v } = W _ { S } \bigl ( t _ { j } - t _ { v } ^ { S } ( \vec { t } ) \bigr ) .\tag{33}
$$

Then the dual causal complex admits the following explicit description.

For every $\emptyset \neq I \subseteq J \subseteq [ d ]$ , the relative interior of the dual cell indexed by $( I , J )$ is

$$
\mathcal { R } _ { I , J } ^ { \circ } = \left\{ \begin{array} { l l } { \begin{array} { r l r } { \phi _ { I , i } ( \vec { t } ) < 0 , } & { i \in I , } \\ { \vec { t } \in \mathbb { R } ^ { d } : } & { \phi _ { I , j } ( \vec { t } ) = 0 , } & { j \in J \setminus I , } \\ & { \phi _ { I , k } ( \vec { t } ) > 0 , } & { k \notin J } \end{array} } \end{array} \right\} .\tag{34}
$$

Equivalently,

$$
t _ { i } < \tau _ { I } ( \vec { t } ) \quad ( i \in I ) , \qquad t _ { j } = \tau _ { I } ( \vec { t } ) \quad ( j \in J \setminus I ) , \qquad t _ { k } > \tau _ { I } ( \vec { t } ) \quad ( k \notin J ) .
$$

Corollary 4.1 shows, taking $I = J = S$ , that the full-dimensional causal region indexed by a nonempty $S \subseteq [ d ]$ is

$$
\mathcal { R } _ { S } ^ { \circ } = \left\{ \vec { t } \in \mathbb { R } ^ { d } : \phi _ { S , i } ( \vec { t } ) < 0 \mathrm { ~ f o r ~ } i \in S , \quad \phi _ { S , j } ( \vec { t } ) > 0 \mathrm { ~ f o r ~ } j \notin S \right\} ,\tag{35}
$$

with closure obtained by replacing the strict inequalities by weak ones.

Two full-dimensional regions $R _ { S }$ and $R _ { T }$ share a codimension-one boundary precisely when the corresponding vertices $p _ { S }$ and $p _ { T }$ are joined by an edge of the subdivision, meaning that $\vert S \triangle T \vert = 1$ . Thus a codimension-one dual cell has $J = S \cup \{ j \}$ for some $j \not \in S .$ . Its defining equality is

$$
t _ { j } = t _ { v } ^ { S } ( \vec { t } ) ,
$$

which is equivalent to $\begin{array} { r } { W _ { S } t _ { j } - \sum _ { i \in S } w _ { i } t _ { i } = \theta _ { v } } \end{array}$ . The codimension-one boundary between the adjacent causal regions $\mathcal { R } _ { S }$ and $\mathcal { R } _ { S \cup \{ j \} }$ , where $\emptyset \neq S \subseteq [ d ]$ and $j \not \in \ S$ , lies in the afine hyperplane

$$
\mathcal { H } _ { S , j } ( w , \theta _ { v } ) : = \left\{ \vec { t } \in \mathbb { R } ^ { d } : W _ { S } t _ { j } - \sum _ { i \in S } w _ { i } t _ { i } = \theta _ { v } \right\} .\tag{36}
$$

Hence the causal complex of a positive TTFS neuron is supported on the parameterized hyperplane family

$$
\begin{array} { r } { A ( w , \theta _ { v } ) = \{ \mathcal { H } _ { S , j } ( w , \theta _ { v } ) : \emptyset \neq S \subseteq [ d ] , \ j \notin S \} . } \end{array}\tag{37}
$$

Proof of Corollary $4 . 1 .$ . By Theorem 4.3, the relative interior of the dual cell indexed by $\emptyset \neq I \subseteq J \subseteq [ d ]$ consists precisely of the inputs for which, with $\tau = t _ { v } ( \vec { t } )$

$$
t _ { i } < \tau \quad ( i \in I ) , \qquad t _ { j } = \tau \quad ( j \in J \setminus I ) , \qquad t _ { k } > \tau \quad ( k \notin J ) .
$$

On this cell, the indices in I are exactly those contributing strictly positively to the membrane potential at firing time. Hence $\begin{array} { r } { \theta _ { v } = \sum _ { i \in I } w _ { i } ( \tau - t _ { i } ) } \end{array}$ , and therefore $\begin{array} { r } { \tau = \frac { \dot { \theta _ { v } } + \sum _ { i \in I } w _ { i } t _ { i } } { W _ { I } } = t _ { v } ^ { I } ( \vec { t } ) } \end{array}$ Since $W _ { I } > 0$ 0,

$$
\phi _ { I , j } ( \vec { t } ) = W _ { I } ( t _ { j } - t _ { v } ^ { I } ( \vec { t } ) ) ,
$$

so the three comparisons with $t _ { v } ^ { I } ( \vec { t } )$ are equivalent to the corresponding sign conditions in (34). A codimension-one dual cell has $J = S \cup \{ j \}$ for some $j \not \in S$ . Its defining equality is

$$
t _ { j } = t _ { v } ^ { S } ( \vec { t } ) ,
$$

which is equivalent to $\begin{array} { r } { W _ { S } t _ { j } - \sum _ { i \in S } w _ { i } t _ { i } = \theta _ { v } } \end{array}$ . This gives (36). Since every codimension-one cell is of this form, the causal complex is supported on the hyperplane family $\mathcal { A } ( w , \theta _ { v } ) . \quad \bigtriangledown$

## 4.5 Interpretation

This construction above is analogous to the lifted Newton-polytope constructions familiar from tropical geometry [27], which have been used in previous studies of piecewise linear neural networks, including ReLU networks [29, 31], maxout networks [16, 49], and maxpooling [50]. In contrast to standard maxout units, which are pointwise maxima of parametric afine functions with unconstrained coeficient vectors, the coeficients of the SNN firing-time map are highly constrained. Indeed, the $2 ^ { d } - 1$ lifted coeficient vectors ${ \vec { y } } _ { S } , \emptyset \neq S \subseteq [ d ]$ , are determined by the d weights $w _ { 1 } , \ldots , w _ { d }$ and the firing threshold $\theta _ { v }$ . Moreover, simultaneous positive scaling of all weights and the threshold leaves the lifted coeficient vectors unchanged, so this family has only d efective degrees of freedom. The resulting geometry also difers from the zonotopal geometry arising in polyhedral descriptions of single-hidden layer ReLU networks. Already for $d \ : = \ : 3$ with $w _ { 1 } = w _ { 2 } = w _ { 3 } = \theta _ { v } = 1$ , the polytope $Q$ has the seven vertices, corresponding to the seven nonempty subsets of [3]. Since every positivedimensional zonotope is centrally symmetric and therefore has an even number of vertices, this seven-vertex polytope cannot be a zonotope.

## 5 Network-level causal patterns

For a single neuron, each causal region $R _ { S }$ is a convex polyhedron. In a layered network, the same principle applies recursively: the firing behavior of each neuron is determined by the vector of spike times it receives from the previous layer, which in turn determines its causal set. This suggests that the natural combinatorial object at the network level is the collection of causal sets across all neurons. We refer to this collection as the causal pattern of the network.

Definition 5.1. Let $\Phi = ( W ^ { \ell } , \Theta ^ { \ell } ) _ { \ell = 1 } ^ { L }$ be a feedforward SNN of depth L with layer widths $N _ { 0 } , \ldots , N _ { L }$ , and fix input spike times $\vec { t } : = t ^ { ( 0 ) } \in \mathbb { R } ^ { N _ { 0 } }$ . For neuron $( \ell , i )$ (the ith neuron in the ℓth layer), denote the presynaptic spike time vector by

$$
t ^ { \ell - 1 } ( \vec { t } ) = ( t _ { 1 } ^ { \ell - 1 } ( \vec { t } ) , \dots , t _ { N _ { \ell - 1 } } ^ { \ell - 1 } ( \vec { t } ) ) \in \mathbb { R } ^ { N _ { \ell - 1 } } .
$$

Following (8) with input dimension $N _ { \ell - 1 }$ , weights $( w _ { i j } ^ { \ell } ) _ { j = 1 } ^ { N _ { \ell - 1 } }$ , and threshold $\theta _ { i } ^ { \ell } .$ , the causal set of this neuron at a presynaptic vector $t ^ { \ell - 1 } ( \vec { t } )$ is the unique subset $S _ { i } ^ { \ell } \subseteq [ N _ { \ell - 1 } ]$ such that $t ^ { \ell - 1 } ( \vec { t } ) \in R _ { S _ { i } ^ { \ell } }$

The causal pattern of $\Phi$ at input $\vec { t }$ is the assignment

$$
\mathbf { S } ( \vec { t } ) : = ( S _ { i } ^ { \ell } ( \vec { t } ) ) _ { i \in [ N _ { \ell } ] , \ell \in [ L ] } .
$$

Thus S is a map $\mathbf { S } \colon \mathbb { R } ^ { N _ { 0 } } \to \mathsf { X } _ { \ell \in [ L ] } \times _ { i \in [ N _ { \ell } ] } 2 ^ { [ N _ { \ell - 1 } ] }$

We define the region count of Φ to be the number of distinct causal patterns realized by inputs in $\mathbb { R } ^ { N _ { 0 } }$

$$
R ( \Phi ) : = | \mathbf { S } ( \mathbb { R } ^ { N _ { 0 } } ) | .
$$

Furthermore, for fixed layer widths and depth, we define the maximal region count to be the maximum of the region count over all possible choices of weights and thresholds:

$$
R _ { \operatorname* { m a x } } ( N _ { 0 } , \ldots , N _ { L } ) : = \operatorname* { m a x } _ { \Phi } R ( \Phi ) .
$$

In analogy with fixing an activation pattern in a ReLU network, the causal pattern $\mathbf { S } =$ $( S _ { i } ^ { \ell } ) _ { i , \ell }$ serves as a combinatorial descriptor of the network’s computation. Specifically, it records, layer by layer and neuron by neuron, which presynaptic spikes arrive early enough to influence the firing time of each postsynaptic neuron. In this way, it encodes the causal structure governing the propagation of information through the network. This reflects the fact that, unlike neurons in a conventional feedforward network, neurons in an SNN do not process an entire input vector at once. Instead, they integrate incoming spikes over time and fire as soon as their membrane potential threshold is reached.

Remark 5.1. For a single neuron, the causal set is determined by the feasibility inequalities (7). Its causal regions are described as unions of regions of the arrangement $\mathcal { A } _ { \mathrm { T T F S } }$ . For network, we work with a causal pattern S. Once a full causal pattern is fixed, every neuron uses a fixed causal set, and therefore every firing time becomes an afine function of the network’s input $\vec { t . }$ Substituting these afine expressions recursively through the layers turns all consistency conditions for the pattern into linear inequalities in $\vec { t . }$ Hence the causal pattern region

$$
R _ { \mathbf { S } } : = \{ \vec { t } \in \mathbb { R } ^ { N _ { 0 } } : \mathbf { S } ( \vec { t } ) = \mathbf { S } \}
$$

is cut out directly by afine halfspaces in the input space. We formalize this next in Proposition 5.1.

Proposition 5.1. Let $\Phi = ( W ^ { \ell } , D ^ { \ell } , \Theta ^ { \ell } ) _ { \ell = 1 } ^ { L }$ be a feedforward SNN of depth L with layers of widths $N _ { 0 } , \ldots , N _ { L }$ . For every causal pattern S, the corresponding region $R _ { \mathbf { S } }$ is a convex polyhedron. Moreover, on $R \mathbf { s }$ , each firing time $t _ { i } ^ { \ell } ( \vec { t } )$ is an afine function of the input $\vec { t } \in \mathbb R ^ { N _ { 0 } }$

The proof is deferred to the Appendix A.2.

## 6 Causal region complexity for shallow SNNs

In this section, we consider shallow SNNs with input dimension $N _ { 0 } = d$ and a single hidden layer of width $N _ { 1 } = m$ , and study the number of distinct causal-set tuples they can realize. We first derive general upper and lower bounds for arbitrary positive weights. These bounds already show that, for fixed input dimension $d ,$ the number of realizable causal regions grows polynomially with the hidden-layer width $m ,$ , despite the exponentially many possible causal sets for each individual neuron. We then consider the more structured class of SNNs where all hidden neurons share the same weight vector, for which we obtain explicit region counts.

## 6.1 Upper bounds

We begin with a simple general upper bound that we refine further below.

Proposition 6.1. A single neuron with d inputs admits $2 ^ { d } - 1$ nonempty causal sets. Consequently, for a shallow SNN with m hidden neurons and input dimension $d ,$ the immediate upper and lower bounds are

$$
R _ { \operatorname* { m a x } } ( d , m ) ~ \leq ~ ( 2 ^ { d } - 1 ) ^ { m } ,\tag{38}
$$

and

$$
R _ { \mathrm { { m a x } } } ( d , m ) ~ \geq ~ 2 ^ { d } - 1 .
$$

The upper bound is typically very loose. Since the m neurons share the same input $\vec { t } \in \mathbb R ^ { d }$ and their causal partitions are geometrically constrained, many tuples of causal sets cannot be realized simultaneously. The example below illustrates this.

Example 6.1. Let $d = 2$ and consider a shallow SNN with $m = 2$ hidden neurons. For $r = 1 , 2$ , let neuron r have positive weights $( w _ { 1 } ^ { ( r ) } , w _ { 2 } ^ { ( r ) } ) \in \mathbb { R } _ { > 0 } ^ { 2 }$ and threshold $\theta _ { r } > 0$ . For an input $\vec { t } = ( t _ { 1 } , t _ { 2 } ) \in \mathbb { R } ^ { 2 }$ , we let $\Delta : = t _ { 2 } - t _ { 1 }$

Each neuron r admits the three causal sets {1}, {2}, {1, 2}. Their causal regions can be expressed entirely in terms of $\Delta$ as

$$
S _ { r } = \left\{ \begin{array} { l l } { \{ 1 \} , } & { \Delta > \theta _ { r } / w _ { 1 } ^ { ( r ) } , } \\ { \{ 2 \} , } & { \Delta < - \theta _ { r } / w _ { 2 } ^ { ( r ) } , } \\ { \{ 1 , 2 \} , } & { - \theta _ { r } / w _ { 2 } ^ { ( r ) } \leq \Delta \leq \theta _ { r } / w _ { 1 } ^ { ( r ) } . } \end{array} \right.
$$

Thus, although the two neurons may have diferent weights and thresholds, their causal sets are determined by the same scalar variable $\Delta$ . Consequently, not all pairs of causal sets can occur, regardless of the choice of parameters. For instance, the pair $( S _ { 1 } ( \vec { t } ) , S _ { 2 } ( \vec { t } ) ) = ( \{ 1 \} , \{ 2 \} )$ would require simultaneously $\Delta > \theta _ { 1 } / w _ { 1 } ^ { ( 1 ) } > 0$ and $\Delta < - \theta _ { 2 } / w _ { 2 } ^ { ( 2 ) } < 0$ , which is impossible. Similarly, the pair $( \{ 2 \} , \{ 1 \} )$ is also impossible. Thus, even though each neuron individually admits three causal sets, not all $3 ^ { 2 } = 9$ causal-set pairs are realizable.

We next derive a sharper upper bound by considering the finite hyperplane arrangement $\mathcal { A }$ whose regions refine the causal partition of the input space. Since the causal-set tuple is constant on each region of ${ \mathcal { A } } ,$ the number of realizable causal-set tuples is bounded by the number of regions of the arrangement.

For each neuron, consider the TTFS arrangement $\mathcal { A } _ { \mathrm { T T F S } }$ introduced in Section 4.2. We first count the number of hyperplanes in this arrangement. For each nonempty $S _ { ; }$ , there are $d - | S |$ choices of $j \not \in S$ . Thus, the number of hyperplanes in the TTFS arrangement of a single neuron is

$$
\kappa _ { d } : = \# \{ H _ { S , j } : \ j \notin S , \emptyset \neq S \subseteq [ d ] \} = \sum _ { \emptyset \neq S \subseteq [ d ] } ( d - | S | ) = d ( 2 ^ { d - 1 } - 1 ) ,\tag{39}
$$

where the last equality follows from the standard binomial identities $\textstyle \sum _ { k = 0 } ^ { d } { \binom { d } { k } } \ = \ 2 ^ { d }$ and $\begin{array} { r } { \sum _ { k = 0 } ^ { d } k \binom { d } { k } = d 2 ^ { d - 1 } } \end{array}$

For a shallow SNN with m hidden neurons and d inputs, let A denote the union of all TTFS arrangements associated with the m neurons. Since each neuron contributes $\kappa _ { d }$ hyperplanes, the arrangement A contains at most $m \kappa _ { d }$ hyperplanes. Recall from Section 4.2 that each TTFS arrangement, and hence their union, can be essentialized to $T = \mathbf { 1 } ^ { \perp } \cong \mathbb { R } ^ { d - 1 }$ Combining this observation with the standard region bound for hyperplane arrangements yields the following upper bound.

Proposition 6.2. Consider a shallow SNN with m hidden neurons for input $\vec { t } \in \mathbb R ^ { d }$ . Let $\mathcal { A }$ be the corresponding arrangement. Then, the causal set tuple $\left( S _ { 1 } , \ldots , S _ { m } \right)$ is constant on each region of $\mathbb { R } ^ { d } \backslash \mathcal { A }$ . Consequently,

$$
R _ { \operatorname* { m a x } } ( d , m ) \leq \operatorname* { m i n } \left\{ \sum _ { k = 0 } ^ { d - 1 } { \binom { m \kappa _ { d } } { k } } , ( 2 ^ { d } - 1 ) ^ { m } \right\}\tag{40}
$$

where $\kappa _ { d } = d ( 2 ^ { d - 1 } - 1 )$ . Moreover,

(i) $( m  \infty$ and fixed d). Then,

$$
R _ { \operatorname* { m a x } } ( d , m ) = \mathcal { O } ( ( m \kappa _ { d } ) ^ { d - 1 } ) = \mathcal { O } ( m ^ { d - 1 } ) .\tag{41}
$$

(ii) $( d \to \infty$ and fixed m). Then,

$$
R _ { \operatorname* { m a x } } ( d , m ) \leq \operatorname* { m i n } \Biggl \{ d \left( \frac { e m \kappa _ { d } } { d - 1 } \right) ^ { d - 1 } , ( 2 ^ { d } - 1 ) ^ { m } \Biggr \} .\tag{42}
$$

Since $\kappa _ { d } = d ( 2 ^ { d - 1 } - 1 ) = \Theta ( d 2 ^ { d } )$ , this yields $R _ { \operatorname* { m a x } } ( m , d ) \leq 2 ^ { \mathcal { O } ( d ) }$ for fixed m.

The causal-set tuple is constant on each region of the arrangement ${ \mathcal { A } } ,$ and therefore the number of distinct causal tuples is bounded by the number of regions of the arrangement. The result then follows from the classical bound for the number of regions cut out by $m \kappa _ { d }$ hyperplanes in $\mathbb { R } ^ { d - 1 }$ ; see [51]. The asymptotic bounds are direct. The full proof is given in the Appendix B.1.

For fixed $d ,$ the bound in (41) grows polynomially in $m$ (of degree at most $d - 1 )$ , whereas the naive bound $( 2 ^ { d } - 1 ) ^ { m }$ in (38) grows exponentially in m. On the other hand, for small m and large $d ,$ the bound obtained from the union of arrangements can exceed the naive bound (e.g., when d = 3 and $m = 1 )$ . Even in the fixed-d regime, however, the arrangement bound need not give the exact number of realizable causal tuples. Indeed, we count all the regions from a large arrangement, while many regions can have the same causal set tuple; see Figure 4.

## 6.2 Lower bounds

We derive a lower bound on the maximum number of realizable causal set tuples. We give a constructive argument showing that the polynomial growth in the width $m$ predicted by the upper bound is asymptotically attainable. Further below, in Section 6.3, we consider the more structured setting in which all neurons share the same weight vector. In this regime, the causal-region partition has suficient structure to allow an exact count. The resulting bound recovers the same asymptotic growth in m for fixed $d ,$ while also revealing exponential growth in d for fixed $m .$

The next result gives an asymptotic lower bound.

Proposition 6.3. Consider a shallow network with m hidden neurons and $d \ge ~ 2$ input neurons. There exist parameters such that

$$
R _ { \operatorname* { m a x } } ( d , m ) \geq { \binom { m - 1 } { d - 1 } } = \Omega ( m ^ { d - 1 } ) , \quad ( d f x e d ) .
$$

The proof is constructive. The idea is that in an open region of input space, each hidden neuron contributes a single hyperplane distinguishing two causal sets. By choosing these m hyperplanes in general position inside $T \cong \mathbb { R } ^ { d - 1 }$ , one realizes at least as many causal regions as the number of bounded regions of a generic arrangement of m hyperplanes in $\mathbb { R } ^ { d - 1 }$ , that is $\binom { m - 1 } { d - 1 }$ . The full proof is given in Appendix B.2.

Propositions 6.2 and 6.3 show that, for fixed input dimension $d ,$ the maximum number of causal regions of a shallow network has a tight asymptotic behavior $\Theta ( m ^ { d - 1 } )$ and thus grows polynomially in the width $m$ . Obtaining an exact formula will require exploiting the precise structure of the causal regions beyond what is visible from the hyperplanes alone.

## 6.3 Shallow SNNs with shared weights

We now consider a structured class of SNNs that permits exact enumeration of causal regions, and yields sharper general lower bounds. Specifically, we study the regime in which all hidden neurons share the same positive weight vector. This shared-weight structure imposes a nestedness property of the causal sets, which leads to explicit counting formulas. These allow us to obtain improved lower bounds for shallow networks and upper bounds for deep networks with shared weights.

Lemma 6.1. Let $d \geq 2$ . Then, for every input $\vec { t } \in \mathbb R ^ { d }$ , the map $\theta \mapsto t _ { v } ( \vec { t } ; \theta )$ from threshold parameter to firing time is strictly increasing. Consequently, for any $0 < \theta _ { 1 } < \theta _ { 2 }$ , we have

$$
t _ { v } ( \vec { t } ; \theta _ { 1 } ) \ < \ t _ { v } ( \vec { t } ; \theta _ { 2 } ) \qquad \Longrightarrow \qquad S ( \theta _ { 1 } ) \subseteq S ( \theta _ { 2 } ) .
$$

Proof. For fixed $\vec { t , }$ recall that the membrane potential $P _ { v } ( t )$ of a neuron v in (3) is given as

$$
P _ { v } ( t ) = \sum _ { i = 1 } ^ { d } w _ { i } \sigma ( t - t _ { i } ) ,
$$

which is continuous and increasing in t because all $w _ { i } > 0$ . Increasing the firing threshold θ increases the first hitting time $t _ { v } ( \vec { t } ; \theta ) =$ min $\{ t \in \mathbb { R } : P _ { v } ( t ) = \theta \}$ . If $t _ { v } ( \theta _ { 1 } ) < t _ { v } ( \theta _ { 2 } )$ , then any index i with $t _ { i } < t _ { v } ( \theta _ { 1 } )$ also satisfies $t _ { i } < t _ { v } ( \theta _ { 2 } )$ , hence $S ( \theta _ { 1 } ) \subseteq S ( \theta _ { 2 } )$ □

Lemma 6.1 is the basic structural fact behind the next result. It shows that, under shared weights, the causal-set tuple of a shallow SNN must form a chain of nested subsets.

We illustrate this for $m = 2$ and $d = 3$ in the following example.

Example 6.2 $( m = 2 , d = 3 )$ . Let $d = 3$ and consider two hidden neurons with the same positive weight vector ⃗w $\in \mathbb { R } _ { > 0 } ^ { 3 }$ and thresholds $0 < \theta _ { 1 } < \theta _ { 2 }$ . Let $S _ { r }$ be the causal set of neuron r at input <sup>⃗</sup>t. Then, for every <sup>⃗</sup>t, Lemma 6.1 gives $S _ { 1 } \subseteq S _ { 2 }$ . In particular, the number of distinct nonempty causal-set tuples $( S _ { 1 } , S _ { 2 } )$ that can occur is at most the number of nonempty pairs $( S _ { 1 } , S _ { 2 } ) \in 2 ^ { [ d ] } \times 2 ^ { [ d ] }$ with $S _ { 1 } \subseteq S _ { 2 }$ , which equals 19, namely:

$$
\mathrm { ( i ) } ~ | S _ { 2 } | = 1 ~ ( 3 ~ \mathrm { p a i r s } ) . ~ ( \{ 1 \} , \{ 1 \} ) , ( \{ 2 \} , \{ 2 \} ) , ( \{ 3 \} , \{ 3 \} ) .
$$

(ii) $| S _ { 2 } | { = } 2 \ ( { \bf 9 } \ { \bf p a i r s } )$

$$
( \{ 1 \} , \{ 1 , 2 \} ) , ( \{ 2 \} , \{ 1 , 2 \} ) , ( \{ 1 , 2 \} , \{ 1 , 2 \} ) ,
$$

$$
( \{ 1 \} , \{ 1 , 3 \} ) , ( \{ 3 \} , \{ 1 , 3 \} ) , ( \{ 1 , 3 \} , \{ 1 , 3 \} ) ,
$$

$$
( \{ 2 \} , \{ 2 , 3 \} ) , ( \{ 3 \} , \{ 2 , 3 \} ) , ( \{ 2 , 3 \} , \{ 2 , 3 \} ) .
$$

(iii) $| S _ { 2 } | = 3$ (7 pairs).

$$
( \{ 1 \} , \{ 1 , 2 , 3 \} ) , ( \{ 2 \} , \{ 1 , 2 , 3 \} ) , ( \{ 3 \} , \{ 1 , 2 , 3 \} ) ,
$$

$$
( \{ 1 , 2 \} , \{ 1 , 2 , 3 \} ) , ( \{ 1 , 3 \} , \{ 1 , 2 , 3 \} ) , ( \{ 2 , 3 \} , \{ 1 , 2 , 3 \} )
$$

$$
( \{ 1 , 2 , 3 \} , \{ 1 , 2 , 3 \} ) .
$$

The number of nested sequences can be given explicitly as follows.

Lemma 6.2. The number of nondecreasing sequences of nonempty subsets of [d] is

$$
\sum _ { k = 1 } ^ { d } { \binom { d } { k } } \left( m ^ { k } - ( m - 1 ) ^ { k } \right) = ( m + 1 ) ^ { d } - m ^ { d } .\tag{43}
$$

Lemmas 6.1 and 6.2 immediately yield an upper bound on the number of realizable causalset tuples in shared-weights shallow SNNs. We show that this upper bound is attainable, which yields the following result, giving the exact number of the maximum of causal-set tuples realizable in shallow SNNs.

Proposition 6.4. Let $d \geq 2$ and $m \geq 1$ . Consider a shallow SNN with input dimension d and m hidden neurons sharing the same positive weight vector $\vec { w } \in \mathbb R _ { > 0 } ^ { d }$ and having pairwise distinct thresholds $0 < \theta _ { 1 } < \theta _ { 2 } < \cdot \cdot \cdot < \theta _ { m }$ . For an input $\vec { t } \in \mathbb R ^ { d }$ , let $S _ { r } : = S ( \vec { t } ; \theta _ { r } ) \subseteq [ d ]$ denote the nonempty causal set of hidden neuron $r .$ Then the maximum number of distinct realizable causal-set tuples $\left( S _ { 1 } , \ldots , S _ { m } \right)$ is

$$
R _ { \mathrm { m a x } } ^ { \mathrm { s h a r e d } } ( d , m ) = ( m + 1 ) ^ { d } - m ^ { d } .\tag{44}
$$

The full proof is presented in Appendix B.3.

Remark 6.1. The nesting of the causal sets has a direct geometric interpretation in the polyhedral picture of Section 4.3. Since all hidden neurons share the same weight vector, the coeficients ${ \vec { a } } _ { S }$ are identical across neurons, while only the ofsets $b _ { S } = \theta _ { r } / W _ { S }$ vary with the threshold $\theta _ { r }$ . Thus, increasing θ translates each causal boundary parallel to itself. In the $d = 3$ example illustrated in Figure $\mathrm { 7 c } ,$ , the central region $R _ { \{ 1 , 2 , 3 \} }$ expands as $\theta$ increases, while the six unbounded boundaries move outward without changing directions. Hence, for $\theta _ { 1 } < \theta _ { 2 }$ , the corresponding causal partitions form nested, scaled copies of one another. This reflects the inclusion $S _ { 1 } \subseteq S _ { 2 }$

## 6.4 Interpretation

For a single-hidden-layer SNN with input dimension d and m hidden neurons, Propositions 6.2 and 6.3 imply that, for fixed $d ,$

$$
R _ { \mathrm { m a x } } ( d , m ) = \Theta ( m ^ { d - 1 } ) \qquad \mathrm { a s ~ } m \to \infty .
$$

On the other hand, Proposition 6.4 gives, for fixed $m$

$$
R _ { \operatorname* { m a x } } ( d , m ) \geq R _ { \operatorname* { m a x } } ^ { \mathrm { s h a r e d } } ( d , m ) = ( m + 1 ) ^ { d } - m ^ { d } = ( 1 - o ( 1 ) ) ( m + 1 ) ^ { d } \qquad \mathrm { a s ~ } d \to \infty .
$$

For comparison, consider a single-hidden-layer ReLU-ANN with d inputs and m hidden neurons. The classical hyperplane-arrangement formula gives $\begin{array} { r } { R _ { \operatorname* { m a x } } ^ { \mathrm { R e L U } } ( d , m ) = \sum _ { j = 0 } ^ { d } { \binom { m } { j } } } \end{array}$ (see, e.g., [25]). Consequently, for fixed $d ,$

$$
R _ { \operatorname* { m a x } } ^ { \mathrm { R e L U } } ( d , m ) = \Theta ( m ^ { d } ) \qquad \mathrm { a s } \ m \to \infty ,
$$

whereas for fixed m and $d \geq m$

$$
R _ { \operatorname* { m a x } } ^ { \mathrm { R e L U } } ( d , m ) = 2 ^ { m } .
$$

Thus, the two models have qualitatively diferent scaling in the two asymptotic regimes. For fixed input dimension $d ,$ the maximum number of regions of a shallow ReLU-ANN grows as $m ^ { d }$ , which is one power of $m$ faster than the $\Theta ( m ^ { d - 1 } )$ growth of realizable causal-set tuples in a shallow SNN. In contrast, for fixed width m, the ReLU region count saturates at $2 ^ { m }$ once $d \geq m$ , whereas the SNN admits at least $( m + 1 ) ^ { d } - m ^ { d } = ( 1 - o ( 1 ) ) ( m + 1 ) ^ { d }$ distinct causal-set tuples. Hence the SNN lower bound grows exponentially with the input dimension d.

It is also worth noting that arbitrary positive and shared weight SNNs have rather different causal geometry, despite having the same asymptotic dependence on width. With arbitrary positive weights, diferent neurons can introduce causal boundaries with diferent orientations, allowing more freedom in how their causal partitions intersect. Under shared weights, by contrast, diferent neurons have same boundary orientations and varying the thresholds produces the nested structure described above. Thus, the additional geometric freedom provided by neuron-specific weights may afect the exact finite-width count and quantifying this gap would be an interesting direction for future work. Figure 1 illustrates this diference in causal region geometry for a simple case of input dimension $d = 3$

## 7 Causal region complexity for deep SNNs

In this section, we give upper and lower bounds on the maximum number of distinct causal patterns realizable by deep SNNs. We first establish a general upper bound for arbitrary positive weights. Then we give a constructive lower bound that grows exponentially with depth. We further consider the shared weight regime, for which we obtain sharper upper bound.

Consider a feedforward SNN of depth L with layer widths $N _ { 0 } = d , N _ { 1 } , . . . , N _ { L }$ . Given an input spike-time vector $\vec { t } ^ { ( 0 ) } \in \mathbb { R } ^ { d }$ , layer ℓ produces the spike-time vector

$$
\bar { t } ^ { ( \ell ) } ( \bar { t } ^ { ( 0 ) } ) = ( t _ { 1 } ^ { ( \ell ) } ( \bar { t } ^ { ( 0 ) } ) , \dots , t _ { N _ { \ell } } ^ { ( \ell ) } ( \bar { t } ^ { ( 0 ) } ) ) \in \mathbb { R } ^ { N _ { \ell } } .
$$

For the rth neuron in layer ℓ, define its causal set relative to layer $\ell - 1$ by

$$
S _ { r } ^ { ( \ell ) } : = \{ j \in [ N _ { \ell - 1 } ] : t _ { j } ^ { ( \ell - 1 ) } < t _ { r } ^ { ( \ell ) } \} .
$$

Thus, $S _ { r } ^ { ( \ell ) }$ records the presynaptic neurons in layer $\ell - 1$ that spike strictly before neuron $( \ell , r )$ . The full causal pattern of the network is the tuple

$$
\left( ( S _ { r } ^ { ( 1 ) } ) _ { r = 1 } ^ { N _ { 1 } } , ( S _ { r } ^ { ( 2 ) } ) _ { r = 1 } ^ { N _ { 2 } } , \ldots , ( S _ { r } ^ { ( L ) } ) _ { r = 1 } ^ { N _ { L } } \right) .
$$

## 7.1 Upper bounds

The idea is to analyze the layers successively. Once the causal pattern through layer $\ell - 1$ is fixed, all firing times produced by the corresponding subnetwork are afine functions of the original input. Therefore, within each such causal region, the boundaries introduced by layer ℓ are obtained by pulling back the layer-ℓ arrangement through the afine map realized by the preceding layers. Applying the shallow-network bound within each region therefore yields a product of the layerwise upper bounds.

Theorem 7.1. For a feedforward SNN of architecture $( d , N _ { 1 } , \ldots , N _ { L } )$ , we have

$$
R _ { \operatorname* { m a x } } ( d , N _ { 1 } , \dots , N _ { L } ) \leq \prod _ { \ell = 1 } ^ { L } \operatorname* { m i n } \Biggl \{ ( 2 ^ { N _ { \ell - 1 } } - 1 ) ^ { N _ { \ell } } , \sum _ { k = 0 } ^ { N _ { \ell - 1 } - 1 } \binom { N _ { \ell } \kappa _ { N _ { \ell - 1 } } } { k } \Biggr \} ,\tag{45}
$$

where $\kappa _ { N _ { \ell } - 1 } = N _ { \ell - 1 } ( 2 ^ { N _ { \ell } - 1 } - 1 )$ is given in (39).

Proof. We apply the shallow bounds successively to the layers. Consider first layer the $\ell ,$ and suppose that the causal patterns of all preceding layers $1 , \ldots , \ell - 1$ have been fixed. On the corresponding region of the input space, the presynaptic spike times $\vec { t } ^ { \ell - 1 }$ entering layer ℓ are fixed functions of the input. Viewed in its presynaptic spike-time space $\mathbb { R } ^ { N _ { \ell - 1 } }$ , layer ℓ is therefore a shallow SNN with $N _ { \ell - 1 }$ inputs and $N _ { \ell }$ neurons. By Proposition 6.2, the maximum number of regions produced by the arrangement are

$$
\sum _ { k = 0 } ^ { N _ { \ell - 1 } - 1 } \binom { N _ { \ell } \kappa _ { N _ { \ell - 1 } } } { k } .
$$

Restricting this arrangement to the afine image of the preceding region cannot produce more regions than the arrangement in the full space $\mathbb { R } ^ { N _ { \ell } - 1 }$ . Hence, each region generated through layer ℓ − 1 can be subdivided by layer ℓ into at most

$$
\operatorname* { m i n } \Biggl \{ ( 2 ^ { N _ { \ell - 1 } } - 1 ) ^ { N _ { \ell } } , \sum _ { k = 0 } ^ { N _ { \ell - 1 } - 1 } \binom { N _ { \ell } \kappa _ { N _ { \ell - 1 } } } { k } \Biggr \} ,
$$

where $( 2 ^ { N _ { \ell - 1 } } - 1 ) ^ { N _ { \ell } }$ is the naive causal set bound given by Proposition 6.1 for the same layer. Multiplying these factors over $\ell = 1 , \ldots , L$ gives (45). □

The shallow polynomial asymptotic bound in (41) does not directly extend to deep networks when multiple layer widths grow simultaneously, since the presynaptic dimension $N _ { \ell - 1 }$ then also varies with the network widths. We therefore retain the finite product bound in (45) as our general upper bound for deep network. This bound can be quite loose, since it treats the causal sets of diferent neurons within each layer as if they could vary independently. In reality, all neurons in a given layer are driven by the same presynaptic spike-time vector, and deeper layers are further constrained by the causal structure inherited from the preceding layers. In Section 7.3, we exploit these dependencies to obtain sharper bounds in the structured shared-weight regime.

## 7.2 Lower bounds

We now derive a constructive lower bound for an SNN with arbitrary positive weights.

The construction is based on the following principle. Suppose that a subnetwork produces M pairwise distinct causal regions $V _ { 1 } , \dots , V _ { M }$ and maps each of them afinely and bijectively onto the same output region W. If W is itself an admissible input region for another copy of the construction, then the same folding mechanism can be applied again to each of the M existing regions. Iterating this construction K times therefore produces $M ^ { K }$ distinct network-level causal patterns. We next summarize the multi-fold construction underlying Lemma 7.1, which will be subsequently used to prove Theorem 7.2.

![](images/1aa70164b8e43a54ba95ca882c44fc132634bc5f5e531bd54481540838ee01d7.jpg)

![](images/f081db08f0c985747205cc5b87a5b474287bf792edbb94e629ea1c29189c33e4.jpg)

![](images/87fe6883a00a465692543bca72f68f7153453bf31bfa009657c2816e532b32c4.jpg)  
Figure 8: Illustration of the multi-fold construction in Lemma 7.1, shown for $m = 5$ . (Left) The relative-time interval $I = ( a , b ) \subset ( 0 , \infty )$ in the coordinate $x = t _ { 2 } - t _ { 1 }$ lifts to the region $Y _ { I } = \{ ( t _ { 1 } , t _ { 2 } ) \in \mathbb { R } ^ { 2 } { : } t _ { 2 } - t _ { 1 } \in I \}$ The first-layer neurons introduce the causal boundaries $\begin{array} { r } { x \ = \ c _ { r } . } \end{array}$ for $r = 1 , \ldots , 5$ , creating four regions $V _ { 1 } , \ldots , V _ { 4 }$ within $Y _ { I }$ corresponding to the intervals $J _ { k } = \left( c _ { k } , c _ { k + 1 } \right)$ . (Middle) The relative output timing $h ( x )$ forms a sawtooth. Each restriction $h | _ { J _ { k } }$ is afine and maps $J _ { k }$ bijectively onto the same output interval $I ^ { \prime } .$ . (Right) Each region $V _ { k }$ is mapped afinely and bijectively onto the common output region $Y _ { I ^ { \prime } } =$ $\{ ( z _ { - } , z _ { + } ) \in \mathbb { R } ^ { 2 } { : } z _ { - } , z _ { + } ) \in I ^ { \prime } \}$ . The next folding block creates regions $W _ { 1 } , \ldots , W _ { 4 }$ within this common image, with boundaries $z _ { + } - z _ { - } = c _ { r } ^ { \prime }$ . This subdivision pulls back to each region $V _ { k }$

Idea of the multi-fold construction The multi-fold construction is implemented by a two-layer SNN of architecture $( 2 , m , 2 )$ . Let $x = t _ { 2 } - t _ { 1 }$ be the diference between the two input spike times. The first layer introduces m ordered breakpoints $c _ { 1 } < \cdots < c _ { m }$ along the x-axis. Crossing $c _ { r }$ changes the causal set of neuron r, so the $m - 1$ intervals $J _ { k } = \left( c _ { k } , c _ { k + 1 } \right)$ carry pairwise distinct causal patterns. The firing times of the first-layer neurons are piecewiseafine functions of $x .$ The second layer combines these firing times through two output neurons, with firing times $z _ { - }$ and $z _ { + }$ . We consider their relative firing time $h : = z _ { + } - z _ { - } .$ In the construction below, this diference depends only on $x = t _ { 2 } - t _ { 1 }$ , and we therefore write it as $h ( x )$ . By choosing the diference between the two output weight vectors so that the slope of h has constant magnitude and alternating sign on successive intervals $J _ { k } .$ , we obtain a sawtooth function $h .$ . Each interval $J _ { k }$ is mapped afinely and bijectively onto the same output interval $I ^ { \prime } .$ . Hence, the two-layer SNN block maps $m - 1$ distinct causal regions onto a common two-dimensional output region. This output region has the same form as the original input region and can therefore serve as the input to another copy of the block. Figure 8 illustrates the construction formalized in Lemma 7.1.

Lemma 7.1. For any nonempty open interval $J \subseteq ( 0 , \infty )$ , let

$$
Y _ { J } : = \Big \{ ( y _ { 1 } , y _ { 2 } ) \in \mathbb { R } ^ { 2 } : y _ { 2 } - y _ { 1 } \in J \Big \} .
$$

Let $m \geq 3$ and let $I = ( a , b ) \subset ( 0 , \infty )$ . Consider a two-layer SNN with architecture $( 2 , m , 2 )$ whose TTFS layer maps are $F ^ { ( 1 ) } { : \mathbb { R } ^ { 2 } }  \mathbb { R } ^ { m }$ and $F ^ { ( 2 ) } { : \mathbb { R } ^ { m } }  \mathbb { R } ^ { 2 }$ , and whose two-layer map is $\Phi : = F ^ { ( 2 ) } \circ F ^ { ( 1 ) } ; \mathbb { R } ^ { 2 }  \mathbb { R } ^ { 2 }$ . It is possible to choose the weights and thresholds such that the

following holds. There exist $m - 1$ pairwise disjoint nonempty open sets $V _ { 1 } , \ldots , V _ { m - 1 } \subset Y _ { I }$ ， and a nonempty open interval ${ \cal I } ^ { \prime } \subset ( 0 , \infty )$ such that

(i) the sets $V _ { 1 } , \ldots , V _ { m - 1 }$ have pairwise distinct causal patterns in the first layer, and hence pairwise distinct causal patterns in the two-layer SNN;

(ii) for every $k = 1 , \ldots , m - 1$ , the restriction $\Phi | _ { V _ { k } } \colon V _ { k } \to Y _ { I ^ { \prime } }$ is an afine bijection.

Proof. Let $c _ { 1 } < c _ { 2 } < \cdots < c _ { m }$ be equally spaced points in the interval $\boldsymbol { I } = ( a , b )$ , with common spacing $c _ { r + 1 } - c _ { r } = \Delta > 0$ . They partition $\left( { { c _ { 1 } } , { c _ { m } } } \right)$ into $m - 1$ consecutive intervals

$$
J _ { k } : = ( c _ { k } , c _ { k + 1 } ) , \quad { \mathrm { f o r ~ } } k = 1 , \ldots , m - 1 .
$$

The first layer will be constructed so that crossing each point $c _ { r }$ changes the causal set of a neuron, making the intervals $J _ { k }$ correspond to distinct causal regions. Let

$$
V _ { k } : = \{ ( t _ { 1 } , t _ { 2 } ) \in \mathbb { R } ^ { 2 } { : } t _ { 2 } - t _ { 1 } \in J _ { k } \} .
$$

Since $J _ { k } \subset I \subset ( 0 , \infty )$ , every $( t _ { 1 } , t _ { 2 } ) \in V _ { k }$ satisfies $t _ { 2 } > t _ { 1 }$ . Therefore, on $V _ { k }$ , only the causal sets {1} and $\{ 1 , 2 \}$ can occur.

For the first layer, let neuron $r \in [ m ]$ have weights $w _ { r 1 } = w _ { r 2 } = 1 / 2$ and threshold $\theta _ { r } = c _ { r } / 2$ . Writing $x = t _ { 2 } - t _ { 1 }$ , its firing time is

$$
y _ { r } = t _ { 1 } + \phi _ { r } ( x ) , \quad \mathrm { w h e r e \ } \phi _ { r } ( x ) = \left\{ \frac { x + c _ { r } } { 2 } , \quad x < c _ { r } , \right.\tag{46}
$$

Now fix $x \in J _ { k } = \left( c _ { k } , c _ { k + 1 } \right)$ . Since the $c _ { k }$ are ordered, we have

$$
x > c _ { r } { \mathrm { ~ f o r ~ } } r \leq k , { \mathrm { ~ a n d ~ } } x < c _ { r } { \mathrm { ~ f o r ~ } } r \geq k + 1 .
$$

Hence, the first k neurons have causal set $\{ 1 \}$ , while the remaining $m - k$ neurons have causal set {1, 2}. The first-layer causal pattern on $V _ { k }$ is

$$
C _ { k } = \Big ( \underbrace { \{ 1 \} , \dots , \{ 1 \} } _ { k } , \underbrace { \{ 1 , 2 \} , \dots , \{ 1 , 2 \} } _ { m - k } \Big ) .\tag{47}
$$

For diferent k these patterns are pairwise distinct. The first layer has therefore created $m - 1$ distinct causal pieces in the region $Y _ { I } \subset \{ t _ { 2 } > t _ { 1 } \}$

We now construct the second layer which has two output neurons with firing times denoted by z<sub>−</sub> and $z _ { + }$ . Our goal is to choose the second-layer parameters so that, on every causal piece $V _ { k }$ , the relative output timing

$$
h ( x ) : = z _ { + } ( x ) - z _ { - } ( x )\tag{48}
$$

ranges bijectively over the same interval $I ^ { \prime } .$ This provides the folding over the relative time coordinate. We verify at the end of the proof that the full two-dimensional map $\Phi | _ { V _ { k } }$ is an afine bijection from $V _ { k }$ onto $Y _ { I ^ { \prime } }$

For this, we choose the weights $\alpha _ { 1 } ^ { \pm } , \ldots , \alpha _ { m } ^ { \pm } > 0$ such that $\textstyle \sum _ { r = 1 } ^ { m } \alpha _ { r } ^ { \pm } = 1$ , and choose the thresholds $\theta ^ { \pm }$ suficiently large such that both second layer neurons fire after all m first layer neurons, that is, their causal sets are both $\{ 1 , \ldots , m \}$ . Then,

$$
z _ { \pm } = \theta ^ { \pm } + \sum _ { r = 1 } ^ { m } \alpha _ { r } ^ { \pm } y _ { r } .\tag{49}
$$

We control the relative output time $h ( x ) = z _ { + } ( x ) - z _ { - } ( x )$ in (48). Using (46), we have

$$
h ( x ) = z _ { + } ( x ) - z _ { - } ( x ) = \theta ^ { + } - \theta ^ { - } + \sum _ { r = 1 } ^ { m } \gamma _ { r } \phi _ { r } ( x ) ,\tag{50}
$$

where $\gamma _ { r } : = \alpha _ { r } ^ { + } - \alpha _ { r } ^ { - }$ . Since $\scriptstyle \sum _ { r = 1 } ^ { m } \alpha _ { r } ^ { \pm } = 1$ , we have $\scriptstyle \sum _ { r = 1 } ^ { m } \gamma _ { r } = 0$

We now determine the coeficients $\gamma _ { r }$ so that the relative output time h folds all intervals $J _ { k }$ onto the same output interval. Fix $k \in \{ 1 , \ldots , m - 1 \}$ . On $J _ { k } = \left( c _ { k } , c _ { k + 1 } \right)$ , we have $\ v { x } > c _ { r }$ for $r \leq k$ and $x < c _ { r }$ for $r \geq k + 1$ . Thus, by (46), the derivative of $\phi _ { r } ( x )$ satisfies

$$
\phi _ { r } ^ { \prime } ( x ) = \left\{ \begin{array} { l l } { { 0 , } } & { { r \leq k , } } \\ { { 1 } } & { { { r \geq k + 1 } . } } \end{array} \right.
$$

and hence by (50) the derivative of $h ( x )$ satisfies

$$
h ^ { \prime } ( x ) = \frac { 1 } { 2 } \sum _ { r = k + 1 } ^ { m } \gamma _ { r } , \quad \mathrm { f o r } x \in J _ { k } .\tag{51}
$$

Equation (51) determines the slope of h on $J _ { k }$ , which is described by the corresponding tail sum of the coeficients $\gamma _ { r }$ . Since all intervals $J _ { k }$ have the same length $\Delta$ , we can make every interval map onto the same output interval by choosing the slopes to have same magnitude but alternating signs. Then, $h$ increases by the same amount on $J _ { 1 }$ , decreases by the same amount on $J _ { 2 }$ , increases again on $J _ { 3 } ,$ and so on.

Fix $0 < \eta < 1 / m$ . We choose the tail sums to have magnitude $\eta ,$ so that, by (51), the slope of h on each interval $J _ { k }$ has magnitude $\eta / 2$ , with alternating sign. We impose

$$
\sum _ { r = k + 1 } ^ { m } \gamma _ { r } = ( - 1 ) ^ { k + 1 } \eta , { \mathrm { ~ f o r ~ } } k = 1 , \dots , m - 1 .\tag{52}
$$

Subtracting two consecutive tail sums gives

$$
\gamma _ { r } = 2 ( - 1 ) ^ { r } \eta , \mathrm { f o r } 2 \leq r \leq m - 1 ,
$$

while the last condition gives $\gamma _ { m } = ( - 1 ) ^ { m } \eta$ . Finally, since $\scriptstyle \sum _ { r = 1 } ^ { m } \gamma _ { r } = 0$ , we obtain

$$
\gamma _ { 1 } = - \eta , \quad \gamma _ { r } = 2 ( - 1 ) ^ { r } \eta , \quad 2 \leq r \leq m - 1 , \quad \gamma _ { m } = ( - 1 ) ^ { m } \eta .\tag{53}
$$

We now realize these coeficients $\gamma _ { r }$ as diferences of two weight vectors by setting

$$
\alpha _ { r } ^ { \pm } = \frac { 1 } { m } \pm \frac { \gamma _ { r } } { 2 } , \mathrm { ~ f o r ~ } r = 1 , \ldots , m .\tag{54}
$$

Then, it can readily be seen that $\gamma _ { r } = \alpha _ { r } ^ { + } - \alpha _ { r } ^ { - }$ , and since $\textstyle \sum _ { r } \gamma _ { r } = 0$ , we get $\scriptstyle \sum _ { r = 1 } ^ { m } \alpha _ { r } ^ { \pm } = 1$ Moreover,

$$
\alpha _ { r } ^ { \pm } \geq \frac { 1 } { m } - \frac { | \gamma _ { r } | } { 2 } \geq \frac { 1 } { m } - \eta > 0 ,
$$

all second layer weights are positive.

Once the coeficients $\gamma _ { r }$ are fixed, the slopes of h on the intervals $J _ { k }$ are determined. The remaining freedom in (50) is the constant term $\theta ^ { + } - \theta ^ { - }$ <sup>−</sup>, which only shifts the sawtooth vertically.

Fix any $q > 0$ , which will determine the lower level of the sawtooth, an define

$$
\beta : = \sum _ { r = 1 } ^ { m } \gamma _ { r } \phi _ { r } ( c _ { 1 } ) = \gamma _ { 1 } c _ { 1 } + \frac { 1 } { 2 } \sum _ { r = 2 } ^ { m } \gamma _ { r } ( c _ { 1 } + c _ { r } ) = \frac { 1 } { 2 } \sum _ { r = 1 } ^ { m } \gamma _ { r } c _ { r } ,
$$

where the last equality follows from $\begin{array} { r } { \sum _ { r } \gamma _ { r } = 0 } \end{array}$ . Since,

$$
h ( c _ { 1 } ) = \theta ^ { + } - \theta ^ { - } + \beta ,
$$

to enforce $h ( c _ { 1 } ) = q$ , it sufices to define $\delta : = q - \beta$ and choose $\theta ^ { + } - \theta ^ { - } = \delta$

Finally, to ensure that both output neurons fire after all first-layer spikes, we add a suficiently large common ofset. Specifically, set

$$
M : = c _ { m } + 1 + | \delta | , \quad \theta ^ { - } : = M , \quad \theta ^ { + } : = M + \delta .\tag{55}
$$

Then, $\theta ^ { + } - \theta ^ { - } = \delta = q - \beta$ , and $\theta ^ { \pm } > c _ { m } > 0$ . Hence,

$$
h ( c _ { 1 } ) = \theta ^ { + } - \theta ^ { - } + \beta = q .
$$

We now describe the fold explicitly. By (51) and (52),

$$
h ^ { \prime } ( x ) = { \frac { \eta } { 2 } } ( - 1 ) ^ { k + 1 } , { \mathrm { ~ f o r ~ } } x \in J _ { k } .\tag{56}
$$

Thus, $h$ is increasing on $J _ { 1 }$ , decreasing on $J _ { 2 } ,$ increasing on $J _ { 3 }$ , and so on. Since every interval $J _ { k }$ has length $\Delta .$ , the change of h across $J _ { k }$ is

$$
h ( c _ { k + 1 } ) - h ( c _ { k } ) = ( - 1 ) ^ { k + 1 } A ,
$$

where $\begin{array} { r } { A : = \frac { \eta \Delta } { 2 } > 0 } \end{array}$ . Starting from $h ( c _ { 1 } ) = q .$ , it follows that

$$
h ( c _ { k } ) = { \left\{ \begin{array} { l l } { q , } & { k { \mathrm { ~ o d d } } , } \\ { q + A , } & { k { \mathrm { ~ e v e n } } . } \end{array} \right. }\tag{57}
$$

Equivalently, on each interval $J _ { k }$ , we can also write h as

$$
h ( x ) = \left\{ \begin{array} { l l l } { { q + \frac { \eta } { 2 } ( x - c _ { k } ) , } } & { { x \in J _ { k } , } } & { { k \mathrm { ~ o d d } , } } \\ { { } } & { { } } & { { } } \\ { { q + A - \frac { \eta } { 2 } ( x - c _ { k } ) , } } & { { x \in J _ { k } , } } & { { k \mathrm { ~ e v e n } . } } \end{array} \right.\tag{58}
$$

Hence, with $I ^ { \prime } : = ( q , q + A )$ , every restriction

$$
h | _ { J _ { k } } \colon J _ { k } \to I ^ { \prime }
$$

is an afine bijection. Thus, every afine piece of the sawtooth covers the same output interval $I ^ { \prime } .$

It remains to verify that the full two-dimensional map is afine and bijective on each $V _ { k }$ On $J _ { k }$ , using (46) in (49) and $\begin{array} { r } { \sum _ { r } \alpha _ { r } ^ { - } = 1 } \end{array}$ gives

$$
z _ { - } = t _ { 1 } + \theta ^ { - } + \sum _ { r = 1 } ^ { m } \alpha _ { r } ^ { - } \phi _ { r } ( x ) = t _ { 1 } + \psi _ { k } ( x )
$$

where

$$
\psi _ { k } ( x ) = \theta ^ { - } + \sum _ { r = 1 } ^ { k } \alpha _ { r } ^ { - } c _ { r } + \frac { 1 } { 2 } \sum _ { r = k + 1 } ^ { m } \alpha _ { r } ^ { - } ( x + c _ { r } )
$$

is afine in x.

Since by definition, $z _ { + } ( x ) = z _ { - } ( x ) + h ( x )$ , we may equivalently use $( z _ { - } , h )$ as output coordinates. On $V _ { k }$ , we therefore consider the map

$$
( t _ { 1 } , x ) \mapsto ( t _ { 1 } + \psi _ { k } ( x ) , h ( x ) ) ,\tag{59}
$$

whose linear part is $\left( \begin{array} { c c } { { 1 } } & { { \psi _ { k } ^ { \prime } } } \\ { { 0 } } & { { h ^ { \prime } } } \end{array} \right)$ , and the determinant of this matrix is $\begin{array} { r } { h ^ { \prime } = \frac { \eta } { 2 } ( - 1 ) ^ { k + 1 } \neq 0 } \end{array}$ Thus, the afine map is injective on $V _ { k }$ . Moreover, $h ( J _ { k } ) = I ^ { \prime }$ , and for each fixed $x \in J _ { k }$ the coordinate $z _ { - } = t _ { 1 } + \psi _ { k } ( x )$ ranges over all of R as $t _ { 1 }$ ranges over R. Hence, its image is exactly $Y _ { I ^ { \prime } }$ . Therefore, every point of $Y _ { I ^ { \prime } }$ has exactly one preimage in $V _ { k } , { \mathrm { s o } } , \Phi | _ { V _ { k } } \colon V _ { k } \to Y _ { I ^ { \prime } }$ is an afine bijection. □

The following example illustrates the construction for $m = 5 ;$ see Figure 8 for the corresponding geometry.

Example 7.1 (The case $m = 5 . )$ . Consider the construction in Lemma 7.1 with $m = 5$ Choose five equally spaced points $0 < a < c _ { 1 } < \cdot \cdot \cdot < c _ { 5 } < b$ inside the interval $\boldsymbol { I } = \left( a , b \right)$ and let $J _ { k } = \left( c _ { k } , c _ { k + 1 } \right)$ for $k = 1 , \ldots , 4$

Recall that $Y _ { I } = \{ ( t _ { 1 } , t _ { 2 } ) \in \mathbb { R } ^ { 2 } { : } t _ { 2 } - t _ { 1 } \in I \}$ is the two dimensional region between the parallel lines $t _ { 2 } - t _ { 1 } = a$ and $t _ { 2 } - t _ { 1 } = b ;$ see Figure 8. The intervals $J _ { 1 } , \ldots , J _ { 4 }$ correspond to the four regions $V _ { k } = \{ ( t _ { 1 } , t _ { 2 } ) \in \mathbb { R } ^ { 2 } { : } t _ { 2 } - t _ { 1 } \in J _ { k } \}$ for $k = 1 , \ldots , 4$

The first layer has five neurons. Neuron r contributes the breakpoint $c _ { r } ,$ that ${ \mathrm { i s } } ,$ crossing the line $x = t _ { 2 } - t _ { 1 } = c _ { r }$ changes the causal set of neuron r. Hence, the four regions $V _ { 1 } , \ldots , V _ { 4 }$ carry four distinct first-layer causal patterns.

The second layer combines the first-layer firing times into two output spike times $z _ { - }$ and $z _ { + }$ . Their relative firing time $h ( x ) = z _ { + } ( x ) - z _ { - } ( x )$ forms a sawtooth with one afine piece on each $J _ { k }$ , and each restriction $h | _ { J _ { k } } \colon J _ { k } \to I ^ { \prime }$ is an afine bijection onto the same interval $I ^ { \prime } .$

Therefore, for every $k = 1 , \ldots , 4 .$ , the full two-layer map restricts to an afine bijection $\Phi | _ { V _ { k } } \colon V _ { k } \to Y _ { I ^ { \prime } }$ , where $Y _ { I ^ { \prime } } = \{ ( z _ { - } , z _ { + } ) \in \mathbb { R } ^ { 2 } { : } z _ { + } - z _ { - } \in I ^ { \prime } \}$ . Thus, the four distinct causal regions $V _ { 1 } , \ldots , V _ { 4 }$ are all folded onto the same output region $Y _ { I ^ { \prime } }$ . Applying the same construction once more partitions $Y _ { I ^ { \prime } }$ into four new regions $W _ { 1 } , \ldots , W _ { 4 }$ . Since each map $\Phi | _ { V _ { k } }$ is bijective, every $W _ { j }$ has one preimage inside each $V _ { k }$ . Consequently, after two blocks one obtains 16 distinct causal regions.

The example illustrates the general recursive mechanism. Lemma 7.1 shows that a twolayer SNN can map $m - 1$ distinct causal regions $V _ { 1 } , \ldots , V _ { m - 1 }$ contained in $Y _ { I }$ afinely and bijectively onto a common output set $Y _ { I ^ { \prime } }$ , and this output set has the same form as the original input set $Y _ { I }$ . Consequently, the construction can be iterated. Suppose that a second block creates $m - 1$ causal regions $W _ { 1 } , \dots , W _ { m - 1 }$ within $Y _ { I ^ { \prime } }$ . Then, for each preceding region $V _ { k }$ , the preimages

$$
( \Phi | _ { V _ { k } } ) ^ { - 1 } ( W _ { 1 } ) , \ldots , ( \Phi | _ { V _ { k } } ) ^ { - 1 } ( W _ { m - 1 } )
$$

are nonempty and pairwise disjoint. Thus, each existing causal region gives rise to $m - 1$ new causal regions. The first layer of each block creates multiplicity and the second layer resets the resulting regions onto a common reusable set. Iterating the two-layer block yields the following width-dependent exponential lower bound.

Theorem 7.2. Let $L \ge 1$ and $m _ { 1 } , \ldots , m _ { L } \ \geq \ 2$ . Consider a depth-2L SNN with input dimension $N _ { 0 } \geq 2$ and widths satisfying

$$
N _ { 2 \ell - 1 } \geq m _ { \ell } , \quad N _ { 2 \ell } \geq 2 , \quad \ell = 1 , \ldots , L .
$$

Then

$$
R _ { \operatorname* { m a x } } ( N _ { 0 } , N _ { 1 } , \ldots , N _ { 2 L } ) \geq \prod _ { \ell = 1 } ^ { L } ( m _ { \ell } - 1 ) .\tag{60}
$$

Proof. We consider the subnetwork with widths $( 2 , m _ { 1 } , 2 , \ldots , m _ { L } , 2 )$ . Any network satisfying the stated width conditions can reproduce this by choosing the additional neurons so that they remain causally inactive on the sets used in the construction, for example by assigning them suficiently large thresholds.

Let $Y _ { I _ { 0 } }$ be an initial input set of the form considered in Lemma 7.1. Apply the lemma to the first two-layer block, whose intermediate width is $m _ { 1 }$ . This produces $m _ { 1 } - 1$ pairwise distinct causal regions, each of which is mapped afinely and bijectively onto the same output set $Y _ { I _ { 1 } }$

Since $Y _ { I _ { 1 } }$ has the same form as $Y _ { I _ { 0 } }$ , we may apply Lemma 7.1 again, now to a block with intermediate width $m _ { 2 }$ . On each of the $m _ { 1 } - 1$ existing regions, the second block creates $m _ { 2 } - 1$ new causal regions, and maps all of them onto a common output set $Y _ { I _ { 2 } }$ . The first two blocks therefore produce $( m _ { 1 } - 1 ) ( m _ { 2 } - 1 )$ distinct network-level causal regions.

Continuing inductively, after the first ℓ blocks the network has $\textstyle \prod _ { k = 1 } ^ { \ell } ( m _ { k } - 1 )$ distinct causal regions. After L blocks, this gives (60). □

Theorem 7.2 provides a basic lower bound that grows with width and depth. The construction exploits only a single relative spike-time coordinate and can be strengthened in several ways. In particular, several copies of the multi-fold construction can be run in parallel on disjoint pairs of spike times. If nonnegative weights were allowed, the cross-pair weights could be set to zero, so that the folds decouple exactly. If, in block $\ell , r _ { \ell , j }$ neurons are assigned to the jth of $p$ pairs, then the resulting Cartesian-product construction yields $\textstyle \prod _ { j = 1 } ^ { p } ( r _ { \ell , j } - 1 )$ regions per block. Under the strictly positive weight setting considered here, however, this exact decoupling may no longer be possible, since every neuron necessarily receives a nonzero contribution from the other input pairs. It may nevertheless be possible to develop a robust version of the construction, in which the zero-cross pair weights are replaced by suficiently small positive weights while preserving the folding behavior on suitable subregions.

Another possible improvement is to exploit both ordering regions $t _ { 1 } < t _ { 2 }$ and $t _ { 2 } < t _ { 1 }$ , in contrast to the present construction which uses only $x = t _ { 2 } - t _ { 1 } > 0$ , and maps each fold back into an interval $I ^ { \prime } \subset ( 0 , \infty )$ . Using $x < 0$ side as well could therefore produce additional regions, but would require a modified reusable output set and a folding construction that can be iterated across both spike time orderings. We leave these possible extensions of the folding construction for future work.

Finally, the last layer need not have width two: it may instead be replaced by a width-one layer or an afine layer, since the output of the final block need not be mapped onto a reusable set for further iteration.

## 7.3 Deep SNNs with shared weights

We now consider the shared-weights regime and derive sharper upper bounds for this regime. More precisely, we assume that within each layer ℓ, all neurons share the same positive weight vector. We assume without loss of generality that within each layer the thresholds are strictly ordered.

By Lemma 6.1, shared weights and ordered thresholds induce a corresponding ordering of the firing times; for instance, $t _ { 1 } ^ { ( \ell ) } < t _ { 2 } ^ { ( \ell ) } < \dots < t _ { N _ { \ell } } ^ { ( \ell ) }$ . For deep SNNs, the key simplification is that, on any region of the input space where the presynaptic spike times entering a layer have a fixed strict ordering, every causal set in the next layer is a prefix of that ordering. More precisely, let $U \subseteq \mathbb { R } ^ { d }$ be a region of the input space on which the spike times in layer $\ell - 1$ satisfy, for some permutation $\pi$ of $[ N _ { \ell - 1 } ]$

$$
t _ { \pi ( 1 ) } ^ { ( \ell - 1 ) } < t _ { \pi ( 2 ) } ^ { ( \ell - 1 ) } < \cdot \cdot \cdot < t _ { \pi ( N _ { \ell - 1 } ) } ^ { ( \ell - 1 ) } , \quad \mathrm { f o r ~ e v e r y ~ } \bar { t } ^ { ( 0 ) } \in U .
$$

Then, for every $\vec { t } ^ { ( 0 ) } \in U$ , the causal set of any neuron r in layer ℓ is necessarily of the form

$$
S _ { r } ^ { ( \ell ) } = \{ \pi ( 1 ) , \ldots , \pi ( q _ { l , r } ) \}
$$

for some $q _ { \ell , r } \in [ N _ { \ell - 1 } ]$ . We call $q _ { \ell , r }$ the prefix length of neuron $( \ell , r )$

For example, suppose $N _ { \ell - 1 } = 3$ and that, on some region U, we have $t _ { 2 } ^ { ( \ell - 1 ) } < t _ { 1 } ^ { ( \ell - 1 ) } <$ $t _ { 3 } ^ { ( \ell - 1 ) }$ . Then the possible causal sets of a neuron in layer ℓ are $\{ 2 \} , \{ 2 , 1 \}$ , and $\{ 2 , 1 , 3 \}$ , with prefix lengths 1, 2 and 3, respectively.

Moreover, if we index the neurons such that the thresholds increase with the neuron index, then the prefix lengths are increasing with the neuron index. Hence, at each fixed input $\vec { t } ^ { ( 0 ) }$ the prefix lengths satisfy

$$
1 \leq q _ { \ell , 1 } \leq q _ { \ell , 2 } \leq \cdots \leq q _ { \ell , N _ { \ell } } \leq N _ { \ell - 1 } .
$$

We collect these observations in the following Lemma, whose proof is given in Appendix C.

Lemma 7.2. Assume the shared-weights and distinct threshold setting. Fix a layer $\ell \geq 2$ and suppose the neurons are indexed such that $0 < \theta _ { 1 } < \theta _ { 2 } < \cdot \cdot \cdot < \theta _ { N _ { \ell } }$ . Let $U \subseteq \mathbb { R } ^ { d }$ be an input region on which the presynaptic spike times from layer $\ell - 1$ have a fixed strict ordering

$$
t _ { \pi ( 1 ) } ^ { ( \ell - 1 ) } < t _ { \pi ( 2 ) } ^ { ( \ell - 1 ) } < \cdots < t _ { \pi ( N _ { \ell - 1 } ) } ^ { ( \ell - 1 ) }
$$

for some permutation $\pi o f \left[ N _ { \ell - 1 } \right]$ . Then, for every $\vec { t } ^ { ( 0 ) } \in U$ and every neuron $( \ell , r )$ , there exists a prefix length $q _ { \ell , r } \in [ N _ { \ell - 1 } ]$ such that

$$
S _ { r } ^ { ( \ell ) } = \{ \pi ( 1 ) , \ldots , \pi ( q _ { \ell , r } ) \} .
$$

Moreover, the prefix lengths are nondecreasing across neurons, so that

$$
1 \leq q _ { \ell , 1 } \leq q _ { \ell , 2 } \leq \cdots \leq q _ { \ell , N _ { \ell } } \leq N _ { \ell - 1 } .
$$

Lemma 7.2 allows us bound the number of causal regions via a combinatorial argument. On any region with a fixed ordering of the presynaptic spike times, the layer-ℓ causal-set tuple at each input is uniquely encoded by a weakly increasing sequence of prefix lengths $( q _ { \ell , 1 } , \dots , q _ { \ell , N _ { \ell } } ) \in [ N _ { \ell - 1 } ] ^ { N _ { \ell } }$ . These weakly increasing sequences form the basic counting objects that we will use in the upper bound below. A detailed example illustrating this correspondence is given in Appendix C.

Theorem 7.3. Let Φ be a feedforward SNN with architecture $( d , N _ { 1 } , \ldots , N _ { L } )$ . Suppose $t h a t ,$ for every layer $\ell = 1 , \ldots , L$ , all neurons in layer ℓ share the same positive presynaptic weight vector in $\mathbb { R } _ { > 0 } ^ { N _ { \ell - 1 } }$ , and their thresholds satisfy $\theta _ { \ell , 1 } < \theta _ { \ell , 2 } < \dots < \theta _ { \ell , N _ { \ell } }$ . Then the number of distinct global causal patterns realized by Φ is bounded by

$$
R _ { \mathrm { m a x } } ^ { \mathrm { s h a r e d } } ( d , N _ { 1 } , \ldots , N _ { L } ) \leq C ( d , N _ { 1 } ) \cdot \prod _ { \ell = 2 } ^ { L } { \binom { N _ { \ell - 1 } + N _ { \ell } - 1 } { N _ { \ell } } } ,\tag{61}
$$

where $C ( d , N _ { 1 } ) = ( N _ { 1 } + 1 ) ^ { d } - N _ { 1 } ^ { d }$ is the region-count upper bound for the first layer.

Proof. By Proposition 6.4, the first-layer causal-set tuple $( S _ { 1 } ^ { ( 1 ) } , \ldots , S _ { N _ { 1 } } ^ { ( 1 ) } )$ has at most $C ( d , N _ { 1 } )$ possible values. Now fix a layer $\ell \geq 2$ . Since the neurons in layer $\ell - 1$ share the same positive weight vector and have strictly ordered thresholds, Lemma 6.1 gives $t _ { 1 } ^ { ( \ell - 1 ) } < t _ { 2 } ^ { ( \ell - 1 ) } < \dots <$ $t _ { N _ { \ell - 1 } } ^ { ( \ell - 1 ) }$ for every input. Hence, by Lemma 7.2, every causal-set tuple in layer ℓ is uniquely determined by a weakly increasing sequence of prefix lengths $1 \leq q _ { \ell , 1 } \leq \dots \leq q _ { \ell , N _ { \ell } } \leq N _ { \ell - 1 }$ The number of such sequences is

$$
\binom { N _ { \ell - 1 } + N _ { \ell } - 1 } { N _ { \ell } } .
$$

Therefore, for each causal pattern realized through layer $\ell - 1$ , there are at most $\binom { N _ { \ell - 1 } + N _ { \ell } - 1 } { N _ { \ell } }$ possible extensions to layer ℓ. Multiplying these bounds recursively over layers $\ell = 2 , \ldots , L ,$ together with the first-layer bound $C ( d , N _ { 1 } )$ , gives (61). □

Remark 7.1. The product bound in (61) counts all admissible prefix-length tuples layer by layer. In deep networks, however, not all such combinations need be realizable on a fixed input region. The reason is that, on a given region of the input space, the output of a layer may lie in a lower-dimensional afine subset of $\mathbb { R } ^ { N _ { \ell } }$

A simple source of this dimension loss is repetition of prefix lengths. Indeed, suppose that on some region, the prefix-length tuple of a layer is

$$
\vec { q } = ( q _ { 1 } , \dotsc , q _ { N } ) , \quad q _ { 1 } \leq \dotsc \leq q _ { N } ,
$$

and that $\vec { q }$ contains only s distinct prefix lengths. Neurons having the same prefix length use the same presynaptic causal set, and hence, their firing times difer only by a constant. Therefore, the image of the corresponding afine layer map is contained in an afine subspace of dimension at most s.

Thus, the set of outputs of the layer forms a lower-dimensional family rather than a full N<sub>ℓ</sub>-dimensional one, and hence the next layer may not be able to realize all weakly increasing prefix-length tuples counted in the combinatorial upper bound.

For shared weights networks, it is not clear whether it is possible to obtain a lower bound that grows exponentially in the depth of the network. Lemma C.1 shows an obstruction to the folding construction used above. In the case of shared-weights, the (2, m, 2) construction has a monotone relative output time, so the alternating sawtooth needed for the multi-fold cannot be obtained by this construction. This does not rule out other mechanisms for obtaining depth-dependent lower bounds in the shared weight setting. We discuss an alternative route in the interpretation below.

## 7.4 Interpretation

The folding construction used above is inspired by the folding constructions for ReLU networks in [12, 14, 52], which exploit the principle that successive layers can repeatedly fold diferent parts of the input space onto a common region. The construction above shows that an analogous mechanism is available in SNNs. The construction is implemented using blocks of two layers. The first layer creates many distinct causal pieces, while the second combines their firing times so that these pieces are mapped onto the same reusable spike-time region. There is a distinction with the ReLU folding constructions. In the constructions of [12, 14, 52], a collection of ReLU units creates multiple pieces, while an afine combination of their activations folds these pieces onto a common region. This afine combination can be absorbed into the preactivations of the following layer. Moreover, several such one-dimensional folds can be applied in parallel along diferent input coordinates. Their Cartesian product then maps multiple full-dimensional input regions onto a common full-dimensional output region. The use of two-layer blocks in our construction for SNNs should not be interpreted as indicating that two layers are necessary for width-dependent folding. A potentially stronger construction could use a single layer to perform both tasks, creating multiple causal regions while simultaneously mapping them onto a common reusable output region. Such a construction is more constrained, since the same neuron parameters would have to determine both the causal subdivision and the coincidence of the corresponding vector-valued afine images.

These observations suggest several directions for strengthening the SNN construction. One possibility is to run multiple independent multi-folds in parallel along diferent relative spike-time coordinates, producing a Cartesian-product folding construction. More generally, one could seek intrinsically higher-dimensional folds. Either mechanism could lead to stronger lower bounds.

For shared weight networks, another possible route for obtaining lower bounds with exponential growth in depth would be to strengthen one layer attainability result in Lemma B.1. Lemma B.1 shows that the relevant causal patterns are attainable somewhere in the presynaptic spike time space, but this is not suficient for iteration; the next layer only sees the particular afine image produced by the preceding layers. Thus, a recursive lower bound argument would require a more uniform attainability statement, for example, that the desired causal patterns can be realized on every arbitrary nonempty open set, or more generally on the afine images produced by the preceding layers. Remark 7.1 complements this observation by showing that such images may be lower dimensional when prefix lengths are repeated.

There is an analogous loss of geometric freedom for ReLU networks with shared weights. In a shallow ReLU network, sharing the weight vector while varying only the biases produces parallel hyperplanes. Thus, m such parallel hyperplanes divide the space into at most m + 1 regions, independently of the input dimension. For deep shared weight ReLU networks, this restriction persists through depth. Indeed, after the first layer, all activations depend on the input only through the scalar coordinate $s = \langle \vec { w } , \vec { x } \rangle$ , where ⃗w is the shared weight vector. Hence, the entire network factors through a one-dimensional input. Although depth can generate exponentially many one-dimensional linear pieces, weight sharing prevents the network from exploiting several independent input directions. Thus, in both SNN and ReLU networks, weight sharing restricts the geometry available for generating regions, although the resulting partitions have diferent structures.

## 8 Experiments

The goal of our experiments is to estimate the number of linear regions the input space is partitioned into by randomly initialized SNNs. We focus on three main questions. First, we study the efect of width and depth. Second, we compare SNN and ReLU networks under matched width and depth configurations. Third, we examine the efect of restricting the SNN weights by comparing networks with arbitrary positive weights and networks where all neurons in a layer use the same incoming weight vector. The code and implementation for the experiments can be found on GitHub at this link.

## 8.1 Experimental settings

Models An SNN consists of an afine linear encoder that maps inputs to spike times, followed by one or more SNN layers, and an afine decoder that produces class logits. The ReLU baseline is chosen to match the corresponding SNN in encoder size, hidden widths, depth, and decoder size. We report the average results over five random seeds. Full implementation and initialization details are given in Appendix D.

Width and depth configurations For the width experiment, we use a single SNN layer and vary the width over {16, 32, 64, 128, 256, 512}. For the depth experiments, we consider networks of depths up to eight. We consider networks with up to eight layers and compare constant-width networks with architectures whose width decreases progressively through depth, denoted Gradual, with widths [256, 256, 256, 128, 128, 128, 64, 64], and More gradual, with widths [512, 256, 256, 128, 128, 64, 64, 64]. These choices allow us to examine the trade-of between maintaining a large width and adding more layers.

Estimating region complexity We estimate region complexity using a trajectory-based method. We randomly sample pairs of inputs $( x _ { 0 } , x _ { 1 } )$ from the dataset, interpolate along the line segment $x ( t ) = ( 1 - t ) x _ { 0 } + t x _ { 1 } { \mathrm { f o r } } t \in [ 0 , 1 ]$ , and track distinct activation patterns along the trajectory, using 20,000 uniformly spaced intervals. For an SNN, a region is identified by the causal pattern, that is, a binary pattern indicating which input neurons belong to the causal prefix of each non-input neuron. Each unique causal pattern corresponds to a distinct linear region. Along a trajectory, we count the number of distinct causal patterns encountered. For ReLU networks, a region is defined by the binary activation pattern of all ReLU units. For each model and configuration, we report the average number of unique regions encountered per trajectory. The region statistics are averaged over many trajectories and random seeds. We complement this with exact enumeration of regions over two-dimensional slices of the input space in selected experiments.

Data sets We estimate region complexity along straight line interpolations between randomly selected pairs of CIFAR-10 [53] and MNIST [54] samples. Unless otherwise stated, all region statistics reported in this section are measured at initialization, without training. Additional experiments examining region complexity during training, comparisons between positive and arbitrary weight SNNs, and initialization ablations are provided in Appendix D.

## 8.2 Experimental results

Efect of width and depth Figure 9 shows the CIFAR-10 and MNIST results, respectively. For both datasets, the number of regions encountered along a trajectory increases rapidly with the width. The efect of depth is more architecture dependent. At smaller widths, increasing the depth has little efect on the observed region count, whereas at larger widths, deeper networks produce more regions. One possible explanation is that suficiently wide layers preserve a higher-dimensional afine image across depth, allowing subsequent layers to introduce additional subdivisions, while narrow layers may create an early dimensional bottleneck that limits further region growth.

Comparison of independent vs shared weights We compare SNNs with arbitrary positive weights to those with shared weights; see Figure 10. For a single hidden layer, the shared and arbitrary positive weight networks produce very similar region counts over a broad range of widths on both datasets. The depth experiments also remain qualitatively similar, although the arbitrary positive weight SNN generates more regions for some architectures and depths. A more controlled analysis during training could help isolate the efect of weight sharing and quantify its contribution more precisely.

Comparison of SNNs vs ReLU networks The estimated number of regions for SNNs is consistently larger than that of the corresponding ReLU networks. The diference is already visible for shallow networks and remains pronounced across depth on both CIFAR-10 and MNIST, shown in Figure 11.

Exact enumeration over two-dimensional afine slices To complement the trajectorybased region estimates at initialization, we additionally perform exact region enumeration on bounded two-dimensional afine slices of the input space during training, following the general type of exact polyhedral analysis considered in [34]. This experiment is intended as a small scale complement to the trajectory-based study. Figure 12 illustrates how the exact region partition and the corresponding decision boundary evolve during the first few training epochs for SNN and ReLU networks of depth 3 and width 10, respectively. The two-dimensional enumeration reveals substantially richer local partition than is observed along individual onedimensional trajectories, and the SNN realizes a considerably finer partition than the matched ReLU network. Full details of the enumeration procedure are provided in Appendix D.

![](images/849de1df7eb82964fbfbd32fd7786726943effbbec3d33c2a458a448372aa33c.jpg)  
Figure 9: Efect of width and depth on the number of causal regions of SNNs at initialization, evaluated for MNIST (top) and CIFAR-10 (bottom).

Additional experiments Appendix D reports several complementary experiments. First, we show that the region complexity can depend substantially on the choice of initialization. Second, we compare positive and arbitrary weight SNNs to examine how allowing negative weights afects the estimated region count. Finally, we track the evolution of region complexity during training together with the corresponding test accuracy.

## 9 Conclusion

We developed a geometric framework for studying the linear regions of the input-output maps computed by TTFS spiking neural networks in the continuous piecewise-linear setting. A central feature of this framework is that the region structure is described in terms of causal sets: subsets of presynaptic spikes that arrive early enough to influence the firing time of a neuron. This provides a geometric and combinatorial description of TTFS computation that is adapted to asynchronous spike-time dynamics.

![](images/1d6a5a7bb365530931c1c6b6b0bbd2b157f74a4b83600fa74ec1ddc5043adf9f.jpg)  
Figure 10: Comparison of SNNs with arbitrary positive weights and SNNs with shared weights, showing the efect of width and depth, evaluated for MNIST (top) and CIFAR-10 (bottom).

At the level of a single neuron, we characterized the causal regions and showed that the firing-time map is the pointwise minimum of an exponential number of highly structured afine functions. In particular, a TTFS neuron with d inputs can exhibit up to $2 ^ { d } - 1$ distinct causal sets, giving it a substantially diferent computational profile from a ReLU neuron with the same number of inputs and parameters. Building on this characterization, we derived upper and lower bounds on the maximal number of linear regions realized by shallow TTFS networks. For fixed input dimension, the maximal region count grows polynomially with width, while the underlying causal structure imposes strong dependencies among the regions generated by diferent neurons.

For deep networks, we showed that these causal partitions can compose to produce exponential growth with depth. Our construction gives an explicit mechanism by which successive TTFS layers fold the input space and multiply the number of linear regions. Thus, despite the strong causal constraints of individual neurons, deep TTFS networks can generate combinatorially complex input-output maps. We also analyzed networks with shared weights, where the causal sets across neurons become nested. In this setting, we obtained substantially sharper region counts, including an exact count for shallow networks. These results illustrate how architectural constraints such as weight sharing translate directly into constraints on causal geometry and hence on expressivity.

Although both TTFS SNNs and ReLU ANNs compute CPWL maps, their region partitions arise through diferent mechanisms. In ReLU networks, the regions are determined by binary activation patterns, with each neuron being either active or inactive for a given input. In TTFS SNNs, by contrast, the regions are determined by causal patterns, with each neuron associated with a set of presynaptic spikes that causally contribute to its firing. This distinction gives rise to fundamentally diferent combinatorial structures.

![](images/aef853314fbbf33f33597bf08eb1972f38fc51ca6546f6fdd50cadeadf88cecd.jpg)  
Figure 11: Comparison of SNNs and matched ReLU networks, showing the efect of width and depth on the estimated region count, evaluated for MNIST (top) and CIFAR-10 (bottom).

Our experiments complement this theoretical picture by examining the region geometry realized by TTFS networks at initialization. They show that TTFS networks can already realize large numbers of linear regions before training and provide an initial comparison with corresponding ReLU architectures. For ReLU networks, variance-preserving schemes such as He initialization [55] are designed to prevent signals from degenerating through depth. In TTFS networks, the scale of the signals is only part of the picture: the relative ordering and separation of spike times determine causal patterns and the locations of transitions between causal regions. Developing initialization and normalization schemes that explicitly account for this temporal geometry is therefore an interesting direction for future work. We view the experiments presented here as an initial investigation of this question rather than an exhaustive study of SNN initialization.

Future work Several theoretical directions remain open. An immediate objective is to tighten the upper and lower bounds obtained here. One possible route is suggested by the polyhedral representation developed in Section 4. For maxout networks, linear-region counts can be studied through vertices of polytopes constructed by iterated Minkowski sums and convex hulls of the polytopes associated with individual neurons [16, 29, 31, 49]. Since TTFS neurons admit a maxout-like representation with constrained parameters, it is natural to ask whether an analogous polytope calculus can be developed for compositions of TTFS neurons. Understanding the combinatorics of the resulting structured polytopes may lead to sharper region-counting results for both shallow and deep SNNs.

A second direction is to understand more systematically the role of synaptic delays. Their efect on causal geometry depends strongly on how they are parameterized. The proof of Theorem 7.3 relies on two structural facts: increasing thresholds produce nested causal sets across neurons within a layer, as in Lemma 6.1, and, within a fixed region, all neurons see the same ordered presynaptic firing times, so their causal sets are prefixes of a common order, as in Lemma 7.2. Both properties hold in the zero-delay setting considered in the theorem. If delays preserve a common efective presynaptic order within each layer, for example, if all neurons in a layer share the same delay vector from the preceding layer, the same nestedprefix argument continues to apply. By contrast, neuron-specific delay vectors can induce diferent arrival-time orderings for diferent postsynaptic neurons. Causal sets then need not be prefixes of a common order, potentially allowing substantially richer causal partitions and requiring diferent counting techniques.

![](images/0d406fe67629e2f83b50aa128c885aa826f70d730018e11ee11db5847b2a2766.jpg)  
Figure 12: Evolution of linear regions (top) and decision boundary (bottom) during training for an SNN (left panels) and a ReLU network (right panels) of depth 3 and width 10 on MNIST, visualized on a fixed two-dimensional slice of the input space given by the afine span of three randomly selected input samples.

Finally, our theoretical analysis concerns maximal region complexity, whereas the complexity typically realized by a network may be substantially smaller. Developing an expectedregion theory for SNNs under random initialization, and ultimately after training, would complement the extremal bounds developed here and provide a more direct connection between causal geometry and practical network behavior. More generally, the causal-region framework provides a way to relate architectural choices, including depth, width, weight sharing, thresholds, and delays, to the geometry and complexity of the functions computed by TTFS networks. We hope that this framework can serve as a basis for a systematic geometric theory of asynchronous computation in spiking neural networks.

## Acknowledgements

M. Singh and G. Kutyniok acknowledge support by the project Next Generation AI Computing (gAIn), funded by the Bavarian Ministry of Science and the Arts and the Saxon Ministry for Science, Culture, and Tourism. M. Singh and G. Kutyniok are also partially supported by DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Education and Research, and additionally, they also acknowledge support from the Munich Center for Machine Learning (MCML).

G. Kutyniok furthermore acknowledges support by the German Research Foundation under Grants DFG-SPP-2298, KU 1446/31-1 and KU 1446/32-1, and by the Bavarian Ministry for Digital Afairs.

G. Mont´ufar was supported in part by NSF grants DMS-2145630, CCF-2212520, and DMS-2522495; DARPA grant HR00112520014 through the Artificial Intelligence Quantified (AIQ) program; DFG project 464109215 within the Priority Programme SPP 2298 “Theoretical Foundations of Deep Learning”; and the BMFTR through DAAD project 57616814 (SECAI).

## Use of AI-assisted tools

The authors used ChatGPT (OpenAI) to assist with language editing, exploration and refinement of some theoretical statements, and the development of computer code. All mathematical arguments, results, and conclusions were developed and independently verified by the authors, who take full responsibility for the content of the manuscript.

## References

[1] Wulfram Gerstner and Werner M. Kistler. Spiking Neuron Models: Single Neurons, Populations, Plasticity. Cambridge University Press, 2002.

[2] Ziqing Wang, Yuetong Fang, Jiahang Cao, and Renjing Xu. Adaptive calibration: A unified conversion framework of spiking neural networks, 2024.

[3] Haoyang Wang, Ruishan Guo, Pengtao Ma, Ciyu Ruan, Xinyu Luo, Wenhua Ding, Tianyang Zhong, Jingao Xu, Yunhao Liu, and Xinlei Chen. Event camera meets mobile embodied perception: Abstraction, algorithm, acceleration, application, 2025.

[4] Paolo Lunghi, Stefano Silvestrini, Dominik Dold, Gabriele Meoni, Alexander Hadjiivanov, and Dario Izzo. Energy eficiency analysis of spiking neural networks for space applications. Astrodynamics, 9(6):909–932, December 2025. ISSN 2522-0098. doi: 10.1007/s42064-024-0256-y.

[5] Jason K. Eshraghian, Max Ward, Emre O. Neftci, Xinxin Wang, Gregor Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu. Training spiking neural networks using lessons from deep learning. Proceedings of the IEEE, 111(9): 1016–1054, 2023.

[6] Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 36(6):51–63, 2019.

[7] Yufei Guo, Xuhui Huang, and Zhe Ma. Direct learning-based deep spiking neural networks:A review. Front. Neurosci., 17, 2023.

[8] Sander Bohte, Joost Kok, and Han Poutr´e. Error-backpropagation in temporally encoded networks of spiking neurons. Neurocomputing, 48:17–37, 2001. doi: 10.1016/ S0925-2312(01)00658-0.

[9] J. G¨oltz et al. Fast and energy-eficient neuromorphic deep learning with first-spike times. Nature Machine Intelligence, 3:823–835, 2021. doi: 10.1038/s42256-021-00388-x.

[10] Iulia M. Comsa, Krzysztof Potempa, Luca Versari, Thomas Fischbacher, Andrea Gesmundo, and Jyrki Alakuijala. Temporal coding in spiking neural networks with alpha synaptic function. In ICASSP 2020, pages 8529–8533, 2020. doi: 10.1109/ICASSP40776. 2020.9053856.

[11] Julian G¨oltz, Jimmy Weber, Laura Kriener, Sebastian Billaudelle, Peter Lake, Johannes Schemmel, Melika Payvand, and Mihai A. Petrovici. Delgrad: exact event-based gradients for training delays and weights on spiking neuromorphic hardware. Nature Communications, 16(1):8245, 2025. ISSN 2041-1723. doi: 10.1038/s41467-025-63120-y.

[12] Guido Mont´ufar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the number of linear regions of deep neural networks. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2, NIPS’14, page 2924–2932, Cambridge, MA, USA, 2014. MIT Press.

[13] Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the expressive power of deep neural networks. In Proceedings of the International Conference on Machine Learning (ICML), volume 70, pages 2847–2854, 2017.

[14] Thiago Serra, Christian Tjandraatmadja, and Srikumar Ramalingam. Bounding and counting linear regions of deep neural networks. In Proceedings of the International Conference on Machine Learning (ICML), volume 80, pages 4558–4566, 2018.

[15] Boris Hanin and David Rolnick. Deep relu networks have surprisingly few activation patterns. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

[16] Guido Mont´ufar, Yue Ren, and Leon Zhang. Sharp bounds for the number of regions of maxout networks and vertices of minkowski sums. SIAM Journal on Applied Algebra and Geometry, 6(4):618–649, 2022. doi: 10.1137/21M1413699.

[17] Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Cambridge University Press, 2014.

[18] Simon Thorpe, Denis Fize, and Catherine Marlot. Speed of processing in the human visual system. Nature, 381(6582):520–522, 1996.

[19] Wolfgang Maass and Christopher M. Bishop, editors. Pulsed Neural Networks. MIT Press, United States, 1999. ISBN 9780262133500.

[20] Wolfgang Maass. On the relevance of time in neural computation and learning. Theor. Comput. Sci., 261(1):157–178, 2001. ISSN 0304-3975.

[21] Wolfgang Maass. Networks of spiking neurons: the third generation of neural network models. Neural networks, 10(9):1659–1671, 1997.

[22] Manjot Singh, Adalbert Fono, and Gitta Kutyniok. Expressivity of spiking neural networks. arXiv preprint arXiv:2308.08218, 2023.

[23] A. Martina Neuman, Dominik Dold, and Philipp Christian Petersen. Stable learning using spiking neural networks equipped with afine encoders and decoders. Journal of Machine Learning Research, 26(246):1–49, 2025.

[24] Dominik Dold and Philipp Christian Petersen. Causal pieces: analysing and improving spiking neural networks piece by piece, 2025.

[25] Razvan Pascanu, Guido Mont´ufar, and Yoshua Bengio. On the number of response regions of deep feed forward networks with piece-wise linear activations. In International Conference on Learning Representations (ICLR 2014), 2014.

[26] Ian Goodfellow, David Warde-Farley, Mehdi Mirza, and Aaron Courville. Maxout networks. 30th International Conference on Machine Learning, ICML 2013, 1302, 02 2013.

[27] Diane Maclagan and Bernd Sturmfels. Introduction to Tropical Geometry, volume 161. American Mathematical Society, Providence, RI, 2015.

[28] Michael Joswig. Essentials of tropical combinatorics, volume 219 of Graduate Studies in Mathematics. American Mathematical Society, Providence, RI, 2021.

[29] Xiao Zhang, Gregory Naitzat, and Lek-Heng Lim. Tropical geometry of deep neural networks. In International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 5824–5832. PMLR, 2018.

[30] Randall Balestriero and Richard Baraniuk. A spline theory of deep learning. In Proceedings of the International Conference on Machine Learning (ICML), volume 80, pages 374–383, 2018.

[31] Vasileios Charisopoulos and Petros Maragos. A tropical approach to neural networks with piecewise linear activations. arXiv preprint arXiv:1805.08749, 2018.

[32] Marie-Charlotte Brandenburg, Georg Loho, and Guido Mont´ufar. The real tropical geometry of neural networks for binary classification. Transactions on Machine Learning Research, September 2024. ISSN 2835-8856.

[33] Boris Hanin and David Rolnick. Complexity of linear regions in deep networks. In Proceedings of the International Conference on Machine Learning (ICML), volume 97, pages 2596–2604, 2019.

[34] Hanna Tseran and Guido Mont´ufar. On the expected complexity of maxout networks. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc. ISBN 9781713845393.

[35] Alexis Goujon, Arian Etemadi, and Michael Unser. On the number of regions of piecewise linear neural networks. Journal of Computational and Applied Mathematics, 441:115667, 2024.

[36] Niket Nikul Patel and Guido Mont´ufar. On the local complexity of linear regions in deep ReLU networks. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=id2CfAgEAk.

[37] Motasem Alfarra, Adel Bibi, Hasan Abed Al Kader Hammoud, Mohamed Gaafar, and Bernard Ghanem. On the decision boundaries of neural networks: A tropical geometry perspective. IEEE Transactions on Pattern Analysis and Machine Intelligence, PP:1–12, 08 2022. doi: 10.1109/TPAMI.2022.3201490.

[38] Francesco Croce, Maksym Andriushchenko, and Matthias Hein. Provable robustness of relu networks via maximization of linear regions. In Kamalika Chaudhuri and Masashi Sugiyama, editors, Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 2057–2066. PMLR, 16–18 Apr 2019.

[39] Joey Huchette, Gonzalo Mu˜noz, Thiago Serra, and Calvin Tsay. When deep learning meets polyhedral theory: A survey, 2025.

[40] Guoqi Li, Lei Deng, Huajin Tang, Gang Pan, Yonghong Tian, Kaushik Roy, and Wolfgang Maass. Brain-inspired computing: A systematic survey and future trends. Proceedings of the IEEE, pages 1–41, 2024. doi: 10.1109/JPROC.2024.3429360.

[41] Adalbert Fono, Manjot Singh, Ernesto Araya, Philipp C. Petersen, Holger Boche, and Gitta Kutyniok. Mathematical foundations of spiking neural networks: Strengths, challenges, and computational paradigm potential [special issue on the mathematics of deep learning]. IEEE Signal Processing Magazine, 43(2):64–76, 2026. doi: 10.1109/MSP.2025.3597033.

[42] Shao-Qun Zhang and Zhi-Hua Zhou. Theoretically provable spiking neural networks. In NeurIPS, 2022.

[43] Duc Anh Nguyen, Ernesto Araya, Adalbert Fono, and Gitta Kutyniok. Time to spike? Understanding the representational power of spiking neural networks in discrete time. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 45954–45987. PMLR, 13–19 Jul 2025.

[44] Shayan Hundrieser, Philipp Tuchel, Insung Kong, and Johannes Schmidt-Hieber. On the universal representation property of spiking neural networks, 2025.

[45] Ana Stanojevi´c, Stanislaw Wo´zniak, Guillaume Bellec, Giovanni Cherubini, Angeliki Pantazi, and Wulfram Gerstner. High-performance deep spiking neural networks with 0.3 spikes per neuron. Nat. Commun., 15, 2024.

[46] Wolfgang Maass and Michael Schmitt. On the complexity of learning for spiking neurons with temporal coding. Information and Computation, 153(1):26–46, 1999. ISSN 0890- 5401. doi: https://doi.org/10.1006/inco.1999.2806.

[47] Wulfram Gerstner. Time structure of the activity in neural network models. Phys. Rev. E, 51:738–758, 1995. doi: 10.1103/PhysRevE.51.738.

[48] Richard P. Stanley. Enumerative Combinatorics: Volume 1. Cambridge University Press, USA, 2nd edition, 2011.

[49] Andrei Balakin, Shelby Cox, Georg Loho, and Bernd Sturmfels. Maxout polytopes, 2025.

[50] Laura Escobar, Patricio Gallardo, Javier Gonz´alez Anaya, Jos´eL. Gonz´alez, Guido Mont´ufar, and Alejandro H. Morales. Enumeration of max-pooling responses with generalized permutohedra. Annals of Combinatorics, 2025. doi: 10.1007/s00026-025-00782-x.

[51] Thomas. Zaslavsky. Facing up to arrangements : face-count formulas for partitions of space by hyperplanes. Memoirs of the American Mathematical Society ; no. 154. American Mathematical Society, Providence, R.I, 1975. ISBN 0821818546.

[52] Guido Mont´ufar. Notes on the number of linear regions of deep neural networks. SampTA 2017 Special Session Mathematics of Deep Learning, 2017. URL https: //escholarship.org/uc/item/3678s0b3.

[53] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[54] Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.

[55] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 1026–1034, 2015. doi: 10.1109/ICCV.2015.123.

[56] Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu, and Luping Shi. Spatio-temporal backpropagation for training high-performance spiking neural networks. Frontiers in Neuroscience, 12:331, 2018. doi: 10.3389/fnins.2018.00331.

[57] Velibor Bojkovic, Srinivas Anumasa, Giulia De Masi, Bin Gu, and Huan Xiong. Data driven threshold and potential initialization for spiking neural networks. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 4771–4779. PMLR, 2024.

[58] Kaiwei Che, Zhengyu Ma, Yifan Huang, Peng Xue, Li Yuan, Timoth´ee Masquelier, Wei Fang, and Yonghong Tian. Eficiently training time-to-first-spike spiking neural networks from scratch, 2024.

[59] Ahmed Imtiaz Humayun, Randall Balestriero, and Richard Baraniuk. Deep networks always grok and here is why. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20722–20745. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/humayun24a.html.

## A Supplementary details on causal sets and causal regions

## A.1 Proof of Lemma 4.1

Proof of Lemma 4.1. Each inequality in (7) is afine in $\vec { t , }$ and its boundary is one of the hyperplanes in $\mathcal { A } _ { \mathrm { T T F S } }$ . Since C is a connected component of the complement of all such hyperplanes, none of these afine expressions can change sign on C. Thus, the causal set S is constant on C. □

## A.2 Proof of Proposition 5.1

Proof of Proposition 5.1. Fix a causal pattern $\mathbf { S } = ( S _ { i } ^ { \ell } ) _ { \ell , i }$ . We show by induction on layers that, on the set of inputs realizing a causal pattern S up to layer $\ell ,$ all firing times in the original input $\vec { t , }$ and the defining constraints are linear in $\vec { t . }$

For layer $\ell = 1$ . For each neuron $( 1 , i )$ , the presynaptic vector is $\vec { t } ^ { ( 0 ) } = \vec { t } \in \mathbb { R } ^ { N _ { 0 } }$ . Fixing $S _ { i } ^ { 1 }$ means <sup>⃗</sup>t lies in the single-neuron region $R _ { S _ { i } ^ { 1 } }$ determined by weights $( w _ { i j } ^ { 1 } ) _ { j = 1 } ^ { N _ { 0 } }$ and threshold $\theta _ { i } ^ { 1 }$ . On this region,

$$
t _ { i } ^ { 1 } ( \vec { t } ) = t _ { v } ^ { S _ { i } ^ { 1 } } ( \vec { t } ) = \frac { \theta _ { i } ^ { 1 } + \sum _ { j \in S _ { i } ^ { 1 } } w _ { i j } ^ { 1 } t _ { j } } { \sum _ { j \in S _ { i } ^ { 1 } } w _ { i j } ^ { 1 } } ,
$$

which is afine in $\vec { t . }$ The constraints defining $S _ { i } ^ { 1 }$ are linear inequalities (see (12) and (10)) in $\vec { t . }$ Intersecting over all $i \in [ N _ { 1 } ]$ shows that the set of inputs realizing the layer-1 part of S is a open convex polyhedron and that all $t _ { i } ^ { 1 } ( \vec { t } )$ are afine there.

Now, we do the inductive step. Assume the claim holds for layer $\ell - 1$ . On the set of inputs realizing $S _ { \Phi }$ up to layer $\ell - 1$ , each presynaptic time $t _ { j } ^ { ( \ell - 1 ) } ( \vec { t } )$ is afine in $\vec { t . }$ Fix a neuron $( \ell , i )$ and its prescribed causal set $S _ { i } ^ { \ell } \subseteq [ N _ { \ell - 1 } ]$ . Then, we have

$$
t _ { i } ^ { \ell } ( \vec { t } ) = \frac { \theta _ { i } ^ { \ell } + \sum _ { j \in S _ { i } ^ { \ell } } w _ { i j } ^ { \ell } t _ { j } ^ { ( \ell - 1 ) } ( \vec { t } ) } { \sum _ { j \in S _ { i } ^ { \ell } } w _ { i j } ^ { \ell } } .
$$

Since each $t _ { i } ^ { ( \ell - 1 ) } ( \vec { t } )$ is afine linear in $\vec { t , }$ it follows that $t _ { i } ^ { \ell } ( \vec { t } )$ is afine in $\vec { t . }$ Moreover, each causal constraint $t _ { j } ^ { ( \ell - 1 ) } ( \vec { t } ) < t _ { i } ^ { \ell } ( \vec { t } ) \mathrm { ~ o r ~ } t _ { j } ^ { ( \ell - 1 ) } ( \vec { t } ) \geq t _ { i } ^ { \ell } ( \vec { t } )$ becomes a strict or weak linear inequality in $\vec { t . }$ Intersecting these linear constraints over all neurons $( \ell , i )$ and all layers $\ell$ yields that $R _ { \mathbf { S } }$ is a (relatively) open convex polyhedron. The afine formulas show that all firing times are afine linear on $R _ { \mathbf { S } }$

The final statement follows because the sets $\{ R \mathbf { s } \}$ partition $\mathbb { R } ^ { N _ { 0 } }$ up to boundaries where equalities occur. □

## B Supplementary details for shallow SNN

In this section, we provide proofs for the results in the case of shallow SNNs from Section 6. We begin with the arrangement-based upper bound and the asymptotic lower bound, and then turn to the shared weight regime, where exact counting and attainability is established.

## B.1 Proof of Proposition 6.2

Proof of Proposition 6.2. By Lemma 4.1, a causal set $S$ can only change when some relevant defining afine inequality becomes tight, that is, on some hyperplane of the form $t _ { v } ^ { S } = t _ { j } , j \notin S$ Thus, the tuple of causal sets is constant on each region of the arrangement. Hence, the number of distinct tuples is at most the maximal region count cut out by N hyperplanes in $\mathbb { R } ^ { d - 1 }$ with $N = m \kappa _ { d }$ . Combining this with the naive causal set bound from Proposition 6.1, we obtain the desired bound.

For part (i). For fixed d and large m, the top term dominates polynomially, using the fact that $\textstyle { \binom { m \kappa _ { d } } { k } } \leq { \frac { ( m \kappa _ { d } ) ^ { k } } { k ! } }$ and $\kappa _ { d }$ only depending on $d ,$ we can bound

$$
\sum _ { k = 0 } ^ { d - 1 } { \binom { m \kappa _ { d } } { k } } \leq \sum _ { k = 0 } ^ { d - 1 } { \frac { ( m \kappa _ { d } ) ^ { k } } { k ! } } = \mathcal { O } ( ( m \kappa _ { d } ) ^ { d - 1 } ) = \mathcal { O } ( m ^ { d - 1 } ) ,
$$

For part (ii). For the case of fixed m and large $d ,$ we use the crude bound $\begin{array} { r } { \sum _ { k = 0 } ^ { d - 1 } \binom { m \kappa _ { d } } { k } \leq } \end{array}$ $d \binom { m \kappa _ { d } } { d - 1 }$ and the inequality ${ \binom { n } { r } } \leq ( e n / r ) ^ { r }$ to obtain

$$
R _ { \operatorname* { m a x } } ( d , m ) \leq d \left( \frac { e m \kappa _ { d } } { d - 1 } \right) ^ { d - 1 } .
$$

Since $\kappa _ { d } = d ( 2 ^ { d - 1 } - 1 ) = \Theta ( d 2 ^ { d } )$ , for fixed $m ,$ , there is a constant $C > 0$ such that $\begin{array} { r } { \frac { e m \kappa _ { d } } { d - 1 } \leq C 2 ^ { d } } \end{array}$ for large d. Hence,

$$
R _ { \operatorname* { m a x } } ( d , m ) \leq d ( C 2 ^ { d } ) ^ { d - 1 } = 2 ^ { \mathcal { O } ( d ^ { 2 } ) } .
$$

On the other hand, the naive bound satisfies

$$
( 2 ^ { d } - 1 ) ^ { m } \leq 2 ^ { m d } = 2 ^ { \mathcal { O } ( d ) }
$$

for fixed m. Taking the smaller of the two estimates, we obtain the desired result. □

## B.2 Proof of Proposition 6.3

We now give proof for the asymptotic lower bound. On a suitable local domain in the essentialized space $T .$ , each hidden neuron acts as a single hyperplane, and the resulting count is given by the bounded region count of a generic hyperplane arrangement.

Proof of Proposition 6.3. We work in the input space $T .$ For $\vec { t } \in T$ , let $x _ { i } = t _ { i } - t _ { d }$ for $i = 1 , \ldots , d - 1$ . Consider an open box

$$
U : = \{ \vec { x } \in \mathbb { R } ^ { d - 1 } \colon | x _ { i } + 1 | < \varepsilon \} , \quad \mathrm { f o r } \ 0 < \varepsilon < \frac { 1 } { 4 } ,
$$

so that on $U$ , we have $x _ { i } < 0$ and hence $t _ { i } < t _ { d }$ for all $i < d .$ with all times $t _ { 1 } , \ldots , t _ { d - }$ <sub>1</sub> within 2ε of each other. For each hidden neuron $r \in [ m ]$ , let $\vec { w } ^ { ( r ) } \in \mathbb { R } _ { > 0 } ^ { d - 1 }$ be the positive weights for the first $d - 1$ inputs, with $\begin{array} { r } { W _ { r } = \sum _ { i = 1 } ^ { d - 1 } w _ { i } ^ { ( r ) } } \end{array}$ , and choose $\theta _ { r } > 0$ so that

$$
1 - \varepsilon < \theta _ { r } / W _ { r } < 1 + \varepsilon .\tag{62}
$$

On $U ,$ condition (62) forces any firing that occurs before $t _ { d }$ to happen only after all of inputs $1 , \ldots , d - 1$ have arrived, hence the causal set is $[ d - 1 ]$ ]. Moreover,

$$
t _ { v , r } ^ { [ d - 1 ] } ( \vec { t } ) = t _ { d } + \frac { \theta _ { r } + \sum _ { i = 1 } ^ { d - 1 } w _ { i } ^ { ( r ) } x _ { i } } { W _ { r } } ,
$$

so $t _ { v , r } ^ { [ d - 1 ] } ( \vec { t } ) \leq t _ { d }$ holds exactly on the halfspace $\begin{array} { r } { \sum _ { i = 1 } ^ { d - 1 } w _ { i } ^ { ( r ) } x _ { i } + \theta _ { r } \leq 0 } \end{array}$ . Thus, on $U$ each neuron divides the space into two causal regions

$$
S _ { r } ( x ) = \left\{ \begin{array} { l l } { [ d - 1 ] , } & { \sum _ { i = 1 } ^ { d - 1 } w _ { i } ^ { ( r ) } x _ { i } + \theta _ { r } \leq 0 , } \\ { [ d ] , } & { \sum _ { i = 1 } ^ { d - 1 } w _ { i } ^ { ( r ) } x _ { i } + \theta _ { r } > 0 . } \end{array} \right.
$$

Choose the m hyperplanes $\begin{array} { r } { H _ { r } \colon \sum _ { i = 1 } ^ { d - 1 } w _ { i } ^ { ( r ) } x _ { i } + \theta _ { r } \le 0 } \end{array}$ in general position and with all bounded regions contained in U (e.g. choose the ofset $\theta _ { r }$ in such a way that all $H _ { r }$ pass suficiently close to $x ^ { 0 } = ( - 1 , \ldots , - 1 ) \in U ) )$ . Then, distinct bounded regions yield distinct causal sets, hence,

$$
R _ { \mathrm { m a x } } ( d , m ) \geq \# \{ \mathrm { b o u n d e d ~ r e g i o n s ~ o f ~ } \{ H _ { r } \} \} = \binom { m - 1 } { d - 1 } ~ = ~ \Omega ( m ^ { d - 1 } ) .
$$

## B.3 Proof of Proposition 6.4

We now move to the shared-weight regime. The attainability result given in Lemma B.1 shows that every nondecreasing prefix-length sequence can be realized on a nonempty open set inside a fixed ordering cone. Figure 13 illustrates this construction in the case of input dimension $d = 3$ and layer width $m = 2$

Lemma B.1. Let $K , N \ge 1$ , and consider an SNN layer with K presynaptic spike times $\vec { t } : = ( t _ { 1 } , t _ { 2 } , \dotsc , t _ { K } ) \in \mathbb { R } ^ { K }$ and N neurons sharing the same positive weight vector ⃗w $: =$ $( w _ { 1 } , \dots , w _ { K } )$ and having strictly ordered thresholds $0 < \theta _ { 1 } < \theta _ { 2 } < \cdot \cdot \cdot < \theta _ { N }$ . Fix a permutation π of [K] and consider the cone $\mathcal { B } _ { \pi } : = \{ \vec { t } \in \mathbb { R } ^ { K } : t _ { \pi ( 1 ) } < t _ { \pi ( 2 ) } < \dots < t _ { \pi ( K ) } \}$ . Then, for every weakly increasing prefix-length tuple

$$
\vec { q } = ( q _ { 1 } , q _ { 2 } , . . . , q _ { N } ) \in [ K ] ^ { N } , \quad w h e r e \ q _ { 1 } \leq \cdot \cdot \cdot \leq q _ { N } ,
$$

there exists a nonempty open set $Z \subset B _ { \pi }$ such that, for every $\vec { t } \in Z$ and every neuron $r \in [ N ]$ the causal set is $S _ { r } ( \vec { t } ) = S _ { r } = \{ \pi ( 1 ) , \ldots , \pi ( q _ { r } ) \}$

Proof of Lemma B.1. For $a = 1 , \ldots , K$ , write the ordered input spike times as $y _ { a } : = t _ { \pi ( a ) }$ For a threshold $\theta > 0$ , the candidate time corresponding to the prefix $\{ \pi ( 1 ) , \ldots , \pi ( a ) \}$ is

$$
t ^ { a } ( \theta ; \vec { t } ) = \frac { \theta + \sum _ { j = 1 } ^ { a } w _ { \pi ( j ) } y _ { j } } { W _ { a } } , \quad \mathrm { w h e r e ~ } W _ { a } : = \sum _ { j = 1 } ^ { a } w _ { \pi ( j ) } .
$$

The prefix $\{ \pi ( 1 ) , \ldots , \pi ( a ) \}$ is feasible exactly when $y _ { a } < t ^ { a } ( \theta ; \vec { t } ) \leq y _ { a + 1 }$ , with the convention $y _ { K + 1 } : = + \infty$ . We now express these feasibility inequalities in a convenient form. Define

$$
A _ { a } : = \sum _ { j = 1 } ^ { a } w _ { \pi ( j ) } ( y _ { a } - y _ { j } )
$$

![](images/3dcce024d78d27d56cba42f3d15dafa6e93e0b9137571a1d8ac48b12d7c3f3dc.jpg)

![](images/c22f236e9a59de8069b3a7fb61fb65256f4de8a299917a37d849250e6c7cada9.jpg)  
Figure 13: Attainability for $d = 3 , m = 2$ (unit weights, thresholds $\theta _ { 1 } , \theta _ { 2 } > 0 )$ . (a) The braid arrangement partitions input space into six braid cones, each ordering the spike times $t _ { \pi ( 1 ) } < t _ { \pi ( 2 ) } < t _ { \pi ( 3 ) }$ for some permutation π. (b) Fixing one cone reduces feasible causal sets to prefixes of π (fix $t _ { \pi ( 1 ) } < t _ { \pi ( 2 ) } < t _ { \pi ( 3 ) }$ and set $s _ { i } = t _ { \pi ( i ) } )$ . Inside that cone, the attainability inequalities subdivide it into six open regions labeled by the realizable size pairs $( | S ( \theta _ { 1 } ) | , | S ( \theta _ { 2 } ) | ) \in \{ ( 1 , 1 ) , ( 1 , 2 ) , ( 1 , 3 ) , ( 2 , 2 ) , ( 2 , 3 ) , ( 3 , 3 ) \}$ . Choosing the cone $\pi _ { \ i }$ , and choosing a point in the corresponding subregion realizes the desired nested causal set $( S _ { 1 } , S _ { 2 } )$ .

to be the total weighted input accumulated from the first a spikes by the time the ath spike arrives. A direct computation shows that

$$
t ^ { a } ( \theta ; \vec { t } ) - y _ { a } = \frac { \theta - A _ { a } } { W _ { a } } .\tag{63}
$$

Hence, $t ^ { a } ( \theta ; \vec { t } ) > y _ { a } \iff \theta > A _ { a }$ . Similarly, since $A _ { a + 1 } = A _ { a } + W _ { a } ( y _ { a + 1 } - y _ { a } )$ , one obtains

$$
y _ { a + 1 } - t ^ { a } ( \theta ; \vec { t } ) = \frac { A _ { a + 1 } - \theta } { W _ { a } } .\tag{64}
$$

Therefore, from (63) and (64), the candidate with prefix length a is feasible precisely when $A _ { a } < \theta \leq A _ { a + 1 }$

Thus, it sufices to choose increasing levels $0 = A _ { 1 } < A _ { 2 } < \cdots < A _ { K } < A _ { K + 1 } : = +$ ∞ such that, for every neuron $r \in [ N ]$ 2

$$
A _ { q _ { r } } < \theta _ { r } < A _ { q _ { r } + 1 } .
$$

Such a choice is possible because both the prefix lengths and thresholds are ordered, $q _ { 1 } \leq$ $\cdots \leq q _ { N }$ and $\theta _ { 1 } < \dots < \theta _ { N }$ . Indeed, for each $a = 2 , \ldots , K$ , all neurons r satisfying $q _ { r } < a$ precede those satisfying $q _ { r } \geq a ,$ , and hence

$$
\operatorname* { m a x } _ { r : q _ { r } < a } \theta _ { r } < \operatorname* { m i n } _ { r : q _ { r } \geq a } \theta _ { r } .
$$

Therefore, we can place $A _ { a }$ between the thresholds assigned to prefix lengths less than a and those assigned to prefix lengths at least a.

Once such levels $A _ { a }$ are chosen, it remains to realize them by actual spike times. Set $y _ { 1 } = 0$ and recursively define

$$
y _ { a + 1 } - y _ { a } = { \frac { A _ { a + 1 } - A _ { a } } { W _ { a } } } , \qquad a = 1 , \ldots , K - 1 .
$$

These gaps are positive, so the resulting point $\vec { t }$ lies in $B _ { \pi }$ . Moreover, the identity $A _ { a + 1 } =$ $A _ { a } + W _ { a } ( y _ { a + 1 } - y _ { a } )$ shows recursively that the resulting spike times realize precisely the chosen levels $A _ { a } . \mathrm { \ B y }$ construction, every neuron r satisfies $A _ { q _ { r } } < \theta _ { r } < A _ { q _ { r } + 1 }$ , and therefore

$$
t _ { \pi ( q _ { r } ) } < t ^ { q _ { r } } ( \theta _ { r } ; \vec { t } ) < t _ { \pi ( q _ { r } + 1 ) } .
$$

Hence, neuron $r$ has causal set $S _ { r } = \{ \pi ( 1 ) , \ldots , \pi ( q _ { r } ) \}$ . Since all inequalities used above are strict, they remain valid on some neighborhood of the constructed point $\vec { t }$ inside $B _ { \pi }$ . Hence, there exists a nonempty open set $Z \subset B _ { \pi }$ on which the same causal sets persist. □

We now prove the result for shallow SNN in the shared weight regime by first counting the number of nondecreasing sequences of nonempty subsets of $[ d ]$ , and then applying the local attainability result above to show that each such sequence is realizable by a shared weight shallow SNN.

Proof of Lemma 6.2. Consider a nondecreasing sequence $\varnothing \neq S _ { 1 } \subseteq S _ { 2 } \subseteq \cdots \subseteq S _ { m } \subseteq [ d ]$ . Fix the last set $B : = S _ { m }$ with $| B | = k$ . Any chain $S _ { 1 } \subseteq S _ { 2 } \subseteq \cdot \cdot \cdot \subseteq S _ { m } = B$ with $S _ { 1 } \neq \emptyset$ is uniquely determined by, for each $j \in B$ , the first index $p _ { j } \in \{ 1 , \ldots , m \}$ at which $j$ enters the chain, $\mathrm { i . e . , } p _ { j } : = \operatorname* { m i n } \{ r \in [ m ] \colon j \in S _ { r } \} \in [ m ]$ . Thus, there are $m ^ { k }$ chains ending at B. Among them, $S _ { 1 } = \emptyset$ holds exactly when $p _ { j } \in \{ 2 , \ldots , m \}$ for all $j \in B ,$ , which gives $( m - 1 ) ^ { k }$ chains. Therefore, the number of chains ending at B with $S _ { 1 } \neq \emptyset$ equals $m ^ { k } - ( m - 1 ) ^ { k }$ . Summing over all choices of $B \subseteq [ d ]$ with $| B | = k$ gives

$$
\sum _ { k = 1 } ^ { d } { \binom { d } { k } } ( m ^ { k } - ( m - 1 ) ^ { k } )\tag{65}
$$

and by the application of binomial theorem to (65), we obtain (43).

Proof of Proposition $6 . 4 \cdot$ By Lemma 6.1, for each fixed $\vec { t }$ the causal sets are nested as thresholds increase, hence, $S _ { 1 } \subseteq \cdots \subseteq S _ { m }$ . Therefore, Lemma 6.2 gives

$$
R _ { \mathrm { m a x } } ^ { \mathrm { s h a r e d } } \leq ( m + 1 ) ^ { d } - m ^ { d } .
$$

It remains to show that every nested sequence counted by Lemma 6.2 is realizable. Let $\emptyset \neq S _ { 1 } \subseteq S _ { 2 } \subseteq \cdots \subseteq S _ { m } \subseteq [ d ]$ be an arbitrary such sequence. Set $B = S _ { m }$ and $k = | \boldsymbol { B } |$ as before. For each $j ,$ , let $p _ { j } =$ min $\{ r \in [ m ] \colon j \in S _ { r } \}$ denote the first index at which $j$ enters the chain. Order the elements $b _ { 1 } , \ldots , b _ { k }$ in $B$ such that $p ( b _ { 1 } ) \leq \cdot \cdot \cdot \leq p ( b _ { k } )$ , and extend to a permutation π of [d] by arranging the elements of $[ d ] \backslash B$ after $b _ { 1 } , \ldots , b _ { k }$ in arbitrary order. For each $r \in [ m ]$ , define $q _ { r } : = | S _ { r } | \in [ k ]$ . Then, because the chain is nested and the ordering $b _ { 1 } , \ldots , b _ { k }$ is compatible with their entry indices, we have

$$
S _ { r } = \{ b _ { 1 } , \ldots , b _ { q _ { r } } \} = \{ \pi ( 1 ) , \ldots , \pi ( q _ { r } ) \} , \qquad r \in \{ 1 , \ldots , m \} ,
$$

and the sequence $( q _ { 1 } , \dots , q _ { m } )$ is nondecreasing. Now, we apply Lemma B.1 with $K = d , N =$ $m _ { \colon }$ , and the braid cone $B _ { \pi }$ . Since $( q _ { 1 } , \ldots , q _ { m } ) \in [ d ] ^ { m }$ is nondecreasing, the lemma yields a nonempty open set $U \subset B _ { \pi }$ such that for every ${ \vec { t } } \in U$ , neuron r has causal set $S _ { r } ~ =$ $\{ \pi ( 1 ) , \ldots , \pi ( q _ { r } ) \}$ . Hence, the prescribed chain is realized on $U .$ . Since the chain was arbitrary, every chain counted in (44) is realizable. Therefore the upper bound in (44) is tight. □

## C Supplementary details for deep SNN

In this subsection, we give the proofs used in the deep network analysis.

Proof of Lemma 7.2. As a result of Lemma 6.1, the threshold ordering and shared weights imply nested causal sets across neurons in the same layer, and strict ordering of firing times. Indeed, if $t _ { r } ^ { ( \ell ) } \leq t _ { r + 1 } ^ { ( \ell ) }$ , then any presynaptic index $j$ satisfying $t _ { j } ^ { ( \ell - 1 ) } < t _ { r } ^ { ( \ell ) }$ also satisfies $t _ { j } ^ { ( \ell - 1 ) } < t _ { r + 1 } ^ { ( \ell ) }$ . This yields $S _ { r } ^ { ( \ell ) } \subseteq S _ { r + \cdot } ^ { ( \ell ) }$ , which for prefixes is exactly $q _ { \ell , r } \leq q _ { \ell , r + 1 } .$ □

The next example illustrates how Lemma 7.2 and Theorem 7.3 combine to show, in the case of a two-layer network, all potential causal sets for the neurons in the second layer.

Example C.1 $\left( d = N _ { 1 } = N _ { 2 } = 3 \right)$ . Consider a two hidden layer network with $N _ { 0 } = d =$ $3 , N _ { 1 } = N _ { 2 } = 3$ in the shared weight regime.

Layer 1 bound. Proposition 6.4 gives

$$
C ( 3 , 3 ) = \sum _ { k = 1 } ^ { 3 } { \binom { 3 } { k } } ( 3 ^ { k } - 2 ^ { k } ) = { \binom { 3 } { 1 } } ( 3 - 2 ) + { \binom { 3 } { 2 } } ( 9 - 4 ) + { \binom { 3 } { 3 } } ( 2 7 - 8 ) = 3 + 1 5 + 1 9 = 3 7 .
$$

Layer 2 bound via prefixes (relative to the layer-1 order). On some region of the input space, the three layer-1 spike times admit a strict order, that is, there exists a permutation π of $\{ 1 , 2 , 3 \}$ such that $\bar { t } _ { \pi ( 1 ) } ^ { ( 1 ) } < t _ { \pi ( 2 ) } ^ { ( 1 ) } < t _ { \pi ( 3 ) } ^ { ( 1 ) }$ . Lemma 7.2 implies that each layer-2 neuron has a prefix causal set with respect to this order, that is, one of {π(1)}, $\{ \pi ( 1 ) , \pi ( 2 ) \} , \quad \{ \pi ( 1 ) , \pi ( 2 ) , \pi ( 3 ) \}$ and, the corresponding prefix lengths $( q _ { 2 , 1 } , q _ { 2 , 2 } , q _ { 2 , 3 } ) \in \{ 1 , 2 , 3 \} ^ { 3 }$ form a weakly increasing tuple. Hence the number of possible layer-2 label tuples is

$$
{ \binom { N _ { 1 } + N _ { 2 } - 1 } { N _ { 2 } } } = { \binom { 3 + 3 - 1 } { 3 } } = { \binom { 5 } { 3 } } = 1 0 .
$$

Therefore, Theorem 7.3 yields $R _ { \operatorname* { m a x } } ( 3 ; 3 ; 3 ) \le C ( 3 , 3 ) \cdot \binom { 5 } { 3 } = 3 7 0$ . The 10 weakly increasing triples in $\{ 1 , 2 , 3 \} ^ { 3 }$

$$
( 1 , 1 , 1 ) , ( 1 , 1 , 2 ) , ( 1 , 1 , 3 ) , ( 1 , 2 , 2 ) , ( 1 , 2 , 3 ) , ( 1 , 3 , 3 ) , ( 2 , 2 , 2 ) , ( 2 , 2 , 3 ) , ( 2 , 3 , 3 ) , ( 3 , 3 , 3 ) .
$$

Lemma C.1 (Obstruction to a two-layer shared-weight multi-fold). Consider the two-layer multi-fold architecture $( 2 , m , 2 )$ without synaptic delays. Suppose that the neurons within each layer share their incoming positive weights and may difer only in their thresholds. For the ordered-threshold first-layer construction described above, the relative firing time of the two second-layer neurons,

$$
h ( x ) = z _ { + } ( x ) - z _ { - } ( x ) ,
$$

is monotone whenever their thresholds are ordered. In particular, the second layer cannot fold two or more consecutive intervals $J _ { k }$ afinely and bijectively onto a common nondegenerate output interval.

Proof. Write $x = t _ { 2 } - t _ { 1 }$ and, by translation equivariance, set $t _ { 1 } = 0$ . On each interval $J _ { k } = \left( c _ { k } , c _ { k + 1 } \right)$ , the first-layer firing times satisfy

$$
y _ { r } ^ { \prime } ( x ) = { \left\{ \begin{array} { l l } { 0 , } & { r \leq k , } \\ { \alpha , } & { r > k , } \end{array} \right. } \quad \quad \alpha : = { \frac { b } { a + b } } > 0 ,
$$

and remain ordered as $y _ { 1 } ( x ) < \dots < y _ { m } ( x )$

Let $q _ { 1 } , \ldots , q _ { m } > 0$ be the weights shared by the two second-layer neurons, and suppose that their thresholds satisfy $\eta _ { - } < \eta _ { + }$ . Their causal sets are therefore ordered prefixes

$$
S _ { - } = \{ 1 , \ldots , s _ { - } \} , \qquad S _ { + } = \{ 1 , \ldots , s _ { + } \} , \qquad s _ { - } \leq s _ { + } .
$$

Writing $\begin{array} { r } { Q _ { s } = \sum _ { r = 1 } ^ { s } q _ { r } } \end{array}$ , the firing time of a neuron with causal set $\{ 1 , \ldots , s \}$ has derivative

$$
z _ { s } ^ { \prime } ( x ) = \alpha \frac { \sum _ { r = k + 1 } ^ { s } q _ { r } } { Q _ { s } } = \alpha \left\{ \begin{array} { l l } { 0 , } & { s \leq k , } \\ { 1 - \displaystyle \frac { Q _ { k } } { Q _ { s } } , } & { s > k . } \end{array} \right.
$$

For fixed $k ,$ this quantity is nondecreasing in s. Consequently, $z _ { + } ^ { \prime } ( x ) \geq z _ { - } ^ { \prime } ( x )$ wherever the derivatives exist, and hence

$$
h ^ { \prime } ( x ) = z _ { + } ^ { \prime } ( x ) - z _ { - } ^ { \prime } ( x ) \geq 0 .
$$

Since $h$ is continuous and piecewise afine, it is nondecreasing. It therefore cannot map two consecutive intervals afinely and bijectively onto the same nondegenerate interval. □

## D Experimental details

Unless otherwise stated, all experiments are performed at random initialization without training. For each configuration, region complexity is estimated along linear trajectories using 20,000 intervals, corresponding to 20,001 sampled points. Therefore, the largest observable number of regions along a trajectory is 20,001. We additionally report an experiment tracking region complexity during training.

SNN forward pass The afine linear encoder maps the real-valued input to input spike times, and these spike times are then processed by the hidden and output SNN layers. Consider a neuron $j$ with $d$ presynaptic spike times $t _ { 1 } , \ldots , t _ { d }$ . We first sort them as $t _ { ( 1 ) } \leq \cdots \leq$ $t _ { ( d ) }$ , together with their corresponding weights. For the prefix $S _ { j , k } = \{ ( 1 ) , \ldots , ( k ) \}$ , the candidate firing time is

$$
t _ { j } ^ { S _ { j , k } } = \frac { \theta _ { j } + \sum _ { r = 1 } ^ { k } w _ { r } ( t _ { ( r ) } } { \sum _ { r = 1 } ^ { k } w _ { ( r ) j } } .\tag{66}
$$

For positive weights, the causal set $S _ { j }$ is the first prefix for which $t _ { j } ^ { S _ { j , k } } < t _ { ( k + 1 ) }$ , with the full prefix always admissible. The output spike time is then $t _ { j } = t _ { j } ^ { S _ { j } }$ . The weighted numerator and denominator are accumulated sequentially as the sorted inputs are processed. Thus, after sorting, finding $S _ { j }$ requires checking at most d prefix candidates rather than testing the $2 ^ { d }$ possible subsets of presynaptic neurons. In this sense, the forward pass is explicitly eventbased and asynchronous, that is, the efective input to each neuron is not a full activation vector in the ANN sense, but an ordered set of arrival times together with weights and delays. This forward pass construction follows the implementation in [23], but is adapted here to our experimental setting.

Initialization For the ReLU ANN, all weights are initialized with Kaiming normal (He) initialization [55], while biases are independently sampled as $b _ { j } \sim \mathcal { N } ( 0 , 0 . 0 1 ^ { 2 } )$

Init 1: For SNN, the weights are sampled according to $\begin{array} { r } { w _ { i j } = \exp ( \sigma z _ { i j } - \frac { \sigma ^ { 2 } } { 2 } ) } \end{array}$ , where $z _ { i j } \sim \mathcal { N } ( 0 , 1 )$ , with $\sigma = 0 . 5$ . The lognormal distribution guarantees positive weights while the above parametrization keeps their mean equal to one. Thresholds are sampled independently as log $\theta _ { j } \sim U ( \log { 0 . 2 5 } , \log { 4 } )$ . We refer to this setting as Init 1. It is the default SNN initialization used for the experiments reported in the main text.

Init 2: For SNNs we additionally consider a second initialization, denoted Init 2 to illustrate the sensitivity of the region complexity to initialization, while emphasizing that a systematic study of SNN initialization is beyond the scope of this work. The weights use the same lognormal parametrization as Init 1 but with the broader scale $\sigma = 1 . 5$ . To initialize the thresholds, let $t _ { ( 1 ) } < \cdots < t _ { ( d ) }$ denote the sorted presynaptic spike times. For a candidate causal prefix of size $k ,$ the threshold at which the neuron changes between prefixes of sizes k and $k + 1$ is

$$
\theta _ { j , k } ^ { \star } ( x ) = \sum _ { r = 1 } ^ { k } w _ { ( r ) j } ( t _ { ( k + 1 ) } - t _ { ( r ) } ) .
$$

We assign diferent neurons diferent values $k _ { j }$ and initialize

$$
\begin{array} { r } { \theta _ { j } = \mathrm { m e d i a n } _ { x \in \mathcal { D } _ { \mathrm { c a l } } } \theta _ { j , k _ { j } } ^ { \star } ( x ) \exp ( \epsilon _ { j } ) , \quad \mathrm { w h e r e ~ } \epsilon _ { j } \sim \mathcal { N } ( 0 , 0 . 0 8 ^ { 2 } ) , } \end{array}
$$

where $\mathcal { D } _ { \mathrm { c a l } }$ is a fixed small subset of the training data used only to calibrate the initialization.

Thus, diferent neurons are initialized near diferent causal-prefix switching conditions. After each SNN layer, we additionally apply a fixed neuron-wise afine transformation $\widetilde { t } _ { j } =$ $\gamma _ { j } t _ { j } + \beta _ { j }$ , calibrated at initialization to help preserve variation in relative spike times across neurons through depth. These quantities are fixed after initialization and are never trained.

Init 3: For training experiments, we also consider another SNN training-oriented initialization Init 3. Init 3 retains the positive lognormal weights of Init 1, but normalizes incoming weights and calibrates thresholds near a range of causal-prefix switching conditions, following the idea used in Init 2 and motivated more broadly by prior work on SNN-specific normalization and data-dependent initialization [56–58].

Init $\underline { { \not \langle 4 \dot { \cdot } } }$ We also consider an initialization with arbitrary weights,

$$
w _ { i j } = \sigma z _ { i j } , \quad \mathrm { w h e r e } \ z _ { i j } \sim \mathcal { N } ( 0 , 1 ) .
$$

Threshold distribution is kept the same as in Init 1. We denote this configuration as Arbitrary in Figure 15. For arbitrary weights, a causal prefix is admissible only when its cumulative weight is positive and the corresponding candidate firing time occurs before the next presynaptic arrival.

Number of regions at initialization Figure 14 shows that Init 2 produces larger estimated region counts than Init 1 shown in Figure 9. The qualitative behavior is consistent in both initializations, growing both with width and depth. The diference between the two initializations is already visible for depth-one networks. In the considered architectures, the empirical growth with depth does not appear exponential; rather, adding depth has a substantially larger efect when the earlier layers are wider. This partially supports the observation that suficiently wide layers preserve a higher-dimensional afine image across depth, allowing subsequent layers to introduce additional subdivisions. Nonetheless, the results suggest that the number of regions can change significantly with the parameter initialization.

![](images/803947ae0db9a59dbe6ed5502c2bc3d56f2145af52a6255a56449cbf3c6919dd.jpg)  
Figure 14: Efect of width and depth on the number of causal regions of SNNs with alternative initialization (Init 2), evaluated for MNIST (top) and CIFAR-10 (bottom).

Number of regions for positive vs arbitrary weights Figure 15 compares the two positive weight initializations Init 1 and Init 2 with the arbitrary weight initialization Init 4. It shows that the arbitrary weight SNN generally realizes more linear regions than Init 1; although Init 2 attains the largest region counts across most of the considered architectures.

One might expect arbitrary weights to reduce the number of admissible causal prefixes, since negative cumulative input cannot trigger a spike. The observation that Init 4 nevertheless realizes more regions that Init 1 may be explained by cancellations between positive and negative weights, which can make cumulative prefix sums more sensitive to changes in arrival time ordering, leading to more frequent changes in the selected causal prefix. We leave a systematic study of this efect, including the interaction between sign, weight magnitude,

![](images/d91324574cc227f5677ae471e380f4244ea037209fde3d5b6e0b4ed126be4b1e.jpg)  
Figure 15: Comparison of two positive weight initializations with arbitrary weights. Shown are the estimated region counts for arbitrary weight SNN, illustrating the efects of width and depth, evaluated for MNIST (top) and CIFAR-10 (bottom).

and firing admissibility, to future work.

Number of regions during training We train depth one SNN and ReLU networks for 100 epochs and evaluate region complexity every 10 epochs. We consider hidden widths 64, 128, and 256. For each width, the SNN and ReLU have matched encoder, hidden-layer, and decoder dimensions. With delays set to zero, the two models also have same number of parameters. For the generic SNN–ReLU comparison, the batch size is 128, the gradient norm is clipped at 5, and the learning rate is $1 0 ^ { - 3 }$ on MNIST and $1 0 ^ { - 4 }$ on CIFAR-10. We restrict the training experiment to depth one, since training deeper TTFS networks require additional inter-layer scaling and normalization mechanisms that we do not address here. SNN weights and thresholds are trained while constrained to remain positive, and optimization uses AdamW.

As shown in Figure 16, SNN region complexity drops sharply during early training and subsequently partially recovers, while remaining substantially larger than that of the matched ReLU network. This behavior is qualitatively similar to observations reported for ANN [15, 33, 34]. The specific mechanism leading to this dynamics is still unknown, although some advances are being pursued in [36, 59]. We further observe that the estimated number of regions is lower for CIFAR-10 data than for MNIST data. The same hidden widths and corresponding SNN-ReLU architectures are used for both MNIST and CIFAR-10; only the input dimension changes with the dataset. A controlled experiment evaluating the number of regions at initialization while varying the input dimension would be an interesting direction for future work.

In Figure 17, we report test accuracy for the SNN (Init 1), ReLU, and the SNN (Init 3). On MNIST, all three models obtain comparable classification accuracy. Despite their diferent region counts, the models achieve comparable accuracy on MNIST. On CIFAR-10, the SNN with Init 1 performs below ReLU, whereas Init 3 substantially improves SNN performance and approaches the ReLU accuracy. These results highlight that the initialization strategy used for training SNNs can have a significant impact on the test performance. We made similar observations for the training error as well, thus the efect appears to be related to how initialization influences the hardness of optimization. Whereas initialization strategies have been explored intensively for ANNs, we still are missing a comparative systematic analysis for SNNs, which we identify as a promising direction for future work. We also note that the observed performance diference cannot be attributed specifically to the positivity constraint. Both SNN initializations used in this experiment employ positive weights, and we do not train an arbitrary weight SNN. A systematic study separating the efects of initialization, positivity constraints, and optimization is left for future work.

![](images/8556063f5112072376b7497fda15943aefe037890d2b334575e874a8ec62d1c3.jpg)

![](images/3a62e4d6acbca2381d53e0a1b1081267eb078a23c2efb964674294deaa652389.jpg)  
Figure 16: Region complexity during training. Estimated numbers of causal regions for depth one SNN and ReLU networks on MNIST (left) and CIFAR-10 (right) for widths 64, 128, and 256. Region complexity drops sharply during the early stages of training relative to initialization and then partially recovers as training proceeds. Across widths and throughout training, the SNN realizes substantially more regions than the corresponding ReLU network. Solid and dotted curves denote SNN and ReLU networks, respectively.

Exact enumeration on two-dimensional afine slices We complement the trajectorybased estimates with exact enumeration on bounded two-dimensional afine slices of the input space. Given three samples x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, we construct an orthonormal basis e<sub>1</sub>, e<sub>2</sub> for the span of $x _ { 2 } - x _ { 1 }$ and $x _ { 3 } - x _ { 1 }$ and parameterize the slice as $x ( u , v ) = x _ { 1 } + u e _ { 1 } + v e _ { 2 }$ . We restrict $( u , v )$ to the bounded rectangular domain containing the selected samples.

Over the afine slice, linear pieces correspond to linear pieces in $z = ( u , v )$ and linear regions are represented as a polygons

$$
R = \{ z \in \mathbb { R } ^ { 2 } \colon A z \leq b \} .\tag{67}
$$

We retain only full-dimensional polygons. For each candidate region, we compute its center by solving the following linear program

$$
\begin{array} { r l } { \underset { z , r } { \operatorname* { m a x } } } & { r , } \\ { \mathrm { s . t . } } & { A _ { i } z + \| A _ { i } \| _ { 2 } r \leq b _ { i } , \quad \mathrm { f o r } i = 1 , \ldots , m , } \\ & { r \geq 0 . } \end{array}\tag{68}
$$

![](images/eddf539020d30b76e194856cba2f012fd0f2f68d8b39943433abdf300e46fbd2.jpg)

![](images/7de239ceab67c03966fb528fab1ccdaa7b2db7b5ff1e14de0c34617d5d54d65d.jpg)  
Figure 17: Classification accuracy during training. Test accuracy for depth one SNN and ReLU networks on MNIST (left) and CIFAR-10 (right) for widths 128 and 256. Both SNN and ReLU achieve comparable performance on MNIST. On CIFAR-10, the SNN with initialization Init 1 attains lower accuracy than ReLU, while the SNN with initialization Init 3 substantially improves performance and approaches the ReLU accuracy. Solid, dotted, and dashed curves denote SNN Init 1, ReLU, and SNN Init 3, respectively.

A region is retained only if $r > 1 0 ^ { - 9 }$ . This excludes intersections that are only line segments or points and this simultaneously provides an interior point for subsequent subdivision.

For a ReLU neuron, its preactivation is afine inside every activation region. We therefore intersect each polygon with the two half-spaces corresponding to positive and negative preactivation and retain each full-dimensional subregion.

For an SNN neuron $j ,$ , let $t _ { i } ( z )$ denote its presynaptic spike times inside the current parent region. For a causal set $S _ { j }$ , accumulation of the spikes in $S _ { j }$ gives the candidate firing time given in (66), which is afine in z. The causal set is valid exactly where

$$
t _ { i } ( z ) \leq t ^ { S _ { j } } ( z ) , \quad i \in S _ { j } \mathrm { ~ a n d ~ } t ^ { S _ { j } } ( z ) \leq t _ { i } ( z ) , \quad i \notin S _ { j } ,\tag{69}
$$

which again gives linear inequalities in $( u , v )$ . Importantly, regions are indexed by causal sets and not by the complete ordering of the presynaptic spike times. We do not test all $2 ^ { d }$ possible causal sets. Starting from the interior point of a parent polygon, we use the sortedprefix forward pass described above to obtain one feasible $S _ { j }$ , and then traverse neighboring feasible causal regions across their facets. After crossing a candidate facet, the forward pass is reevaluated; a new region is retained only if the causal set changes and its complete inequality system has positive two-dimensional interior. The resulting regions carry afine spike-time representations, allowing the same procedure to be applied recursively across neurons and layers.

All linear programs are solved with scipy.optimize.linprog using the HiGHS backend. The ReLU construction follows the general polyhedral enumeration principle of [34], while the SNN case uses the causal set subdivision and adjacency traversal described above.