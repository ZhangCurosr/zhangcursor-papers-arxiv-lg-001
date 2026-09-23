# OMatG-flash: An All-Atom Flow Map with Reinforce Adjoint Matching for Scalable Materials Discovery

Thomas Egg<sup>1</sup>, Harry Winston Sullivan<sup>2</sup>, Ellad B. Tadmor<sup>2</sup>, Stefano Martiniani<sup>1</sup>

<sup>1</sup>New York University, <sup>2</sup>University of Minnesota

Abstract: The discovery of novel inorganic materials drives technological breakthroughs in critical fields such as computing and energy storage. Generative AI has promised to accelerate the materials discovery pipeline, but state-of-the-art flow and difusion models remain bottlenecked by the cost of proposing candidate materials. To address this, we introduce OMatG-flash, an all-atom flow map for inorganic crystal structure prediction (CSP) and de novo generation (DNG). OMatG-flash is a Pareto-optimal inference engine for materials, sampling candidate materials with an order of magnitude fewer inference steps and less wall-clock time than existing flow and difusion models while demonstrating benchmark performance on par with the state-of-the-art. To enable post-training finetuning we apply Reinforce Adjoint Matching to flow maps, further improving match rates and RMSE on the unconditional CSP task. OMatG-flash showcases the potential of flow maps to accelerate generation of high-quality candidate inorganic materials and demonstrates a step forward in sample throughput necessary for data-hungry materials discovery workflows.

Code: https://github.com/FERMat-ML/OMatG

![](images/2576d065b74ccb25e1d71328746ec223eb548c362f85d45bd3979a42d4e53995.jpg)

## 1 Introduction

A central goal in materials discovery is that of proposing candidate inorganic crystalline materials with desired properties. This has implications in critical fields such as computing [1], energy storage [2], and sustainability [3]. Traditional materials discovery pipelines involve repeated synthesis and experimentation loops which are time-consuming and costly to perform [4]. This has inspired the creation of computational algorithms which leverage molecular dynamics and electronic structure to propose materials without the need for lab-based synthesis [5].

Such algorithms aim to solve this problem via repeated first-principles quantum chemical calculations which yield accurate results but are computationally expensive and often impractical at scale [6]. Despite unfavorable scaling, the deployment of such methods has culminated in the aggregation of large materials science datasets [7–9] along with a wealth of computational tools [10, 11] for analyzing them. These tools, coupled with advances in machine learning and artificial intelligence, have spurred the de velopment of highly accurate machine learning models for applications in chemistry and physics which can approximate energies and forces with chemical accuracy [12–14] or propose candidate compositions and crystal structures [15, 16]. In the domain of generative modeling for materials, difusion- and flow based generative models have shown particularly dominant performance [17–21]. Nevertheless, there is an ever-present need to accelerate the rate at which candidate structures can be proposed. As continued efort and innovation flow into scaling autonomous labs [22, 23] and accelerated downstream processing [24], so too must the scalability of generative algorithms grow to saturate these systems.

![](images/6af44d459eb8853e7a73d9496c1b1880cf1159fdf24dd4de7ae940567ea6e053.jpg)  
Figure 1: Pareto-Optimal Inference. Pareto frontier of combined S.U.N. and M.S.U.N. vs. through put is shown for existing models. OMatG-flash is Pareto-optimal, surpassing existing models by an order of magnitude in terms of throughput while maintaining competitive raw performance on the DNG task. Solid points indicate OMatG-flash under several number of function evaluation (NFE) budgets.

Flow and difusion models are fundamentally limited by the need to numerically integrate diferentia equations to propose candidate materials. To address this we introduce OMatG-flash, a transformerbased flow map for CSP and DNG of inorganic materials. OMatG-flash boasts an order of magnitude faster wall-clock inference time and far fewer neural network evaluations than existing flow and difusion model architectures while maintaining equivalent performance. Furthermore, we extend post-training reinforcement learning methods developed for flow matching to flow maps, attaining performance com petitive with respect to the state-of-the-art on the CSP task with few function evaluations. In contrast to conventional flow map distillation methods, OMatG-flash is trained directly from data unlike teacherdistilled flow maps which require access to a pretrained generative model. Furthermore, OMatG-flash can be optimized for a target reward via Reinforce Adjoint Matching while retaining its low inference cost.

## Our Contributions:

• We introduce OMatG-flash, a Pareto-optimal all-atom flow map that rivals existing flow- and difusion based models on materials-generation benchmarks while substantially improving inference-time scalability via few-step sampling (Figure 1).

• We show that post-training OMatG-flash with Reinforce Adjoint Matching (RAM)—to our knowl edge, its first application to flow maps—achieves state-of-the-art performance on polymorph-aware CSP benchmarks compared to existing difusion- and flow-based models.

• We characterize the key limitations of flow map training for scientific applications and outline prospective future directions for how they may be applied to other problems in materials design.

## 2 Related Work

Flows and Difusions Difusion and flow models formulate generative modeling as a transport problem; mapping a tractable easy-to-sample base distribution to a target distribution that is accessible only through data samples [25]. The difusion model approach uses a noising stochastic diferential equation (SDE) to construct a regression target for estimating the drift of a reverse denoising SDE. With this learned drift, generation proceeds by sampling an initial noisy state and numerically integrating in time to produce an approximate data sample from the target [26, 27]. In flow models the transport is framed deterministically in terms of an instantaneous velocity field which satisfies the probability flow ordinary diferential equation (ODE) describing the probability path of the distribution $\rho _ { t }$ [25, 28, 29]. This velocity field is numerically integrated via deterministic dynamics to draw approximate samples from the target.

Flow Maps and Consistency Models Unfortunately, the iterative numerical integration required to draw samples with flow- and difusion-based models requires repeated model evaluations, hindering time-sensitive performance. This has inspired the creation of flow maps which learn a transport that requires few neural network evaluations per generated sample [30–32]. Flow maps have demonstrated performance matching both difusion and flow models in various domains at a greatly reduced cost [33, 34]. Flow maps exhibit success in language [35], biochemistry [34], and physics [36] while retaining the ability to be learned directly from data. Furthermore, flow maps are being applied for drawing Boltzmann-distributed data in thermodynamic ensembles [37, 38].

Generative Models for Materials Discovery Improvement in the performance and scalability of generative AI has inspired the creation of machine learning models which can rapidly propose physically stable and novel crystalline materials. Difusion- and flow-based methods have demonstrated particularly strong performance in this domain [17–20, 39, 40]. Subsequent advances in both the underlying algorithms and network architectures further improved generative accuracy [21, 41, 42], while condi tioning strategies have enabled the generation of materials with targeted properties [43, 44]. Current CSP and DNG generative models have thus shifted focus to post-training algorithms with reinforcement learning [45, 46], enabling the generative process to push beyond the underlying dataset it was trained on. Parallel to this, substantial efort has been devoted to training large language models to generate text-based representations of crystal structures [47–49].

Despite these advances, existing work for inorganic crystals has largely emphasized improvement in sample quality while giving less attention to generation time. For large-scale materials screening, the relevant objective is not necessarily the accuracy of any individual prediction, but the number of viable candidates produced within a fixed computational budget. A model that generates thousands of can didates in the time another requires to generate tens may therefore prove more useful in data-driven materials discovery, even if its per-sample success rate is lower. This trade-of is becoming increasingly important as autonomous agents enable high-throughput downstream validation and screening workflows to operate with far less manual intervention [50]. As these workflows scale, generative throughput may become the limiting factor, motivating methods capable of rapidly supplying candidates and saturating downstream screening capacity.

## 3 Methods

Unit Cell Representation We represent an N-atom periodic unit cell as an element $\mathbf { c } \in \mathcal { M }$ of the product manifold

$$
\begin{array} { r } { \mathbf { c } = ( \mathbf { A } , \mathbf { F } , \mathbf { y } ) \in \mathcal { M } : = \mathbb { R } ^ { N \times d _ { A } } \times \mathbb { T } _ { [ 0 , 1 ) } ^ { N \times 3 } \times \mathbb { R } ^ { 6 } , } \end{array}\tag{1}
$$

where A is the atomic descriptor matrix with embedding dimension $d _ { A }$ , F represents the fractional coordinates, and y is a rotation-invariant unit-cell representation obtained from the Cholesky decomposition of the lattice metric tensor $\mathbf { G } = \mathbf { L L } ^ { \top }$ (Appendix A). We define the Riemannian stochastic

interpolant and its endpoint-conditional velocity as

$$
\begin{array} { r } { \mathbf { c } _ { t } = \exp _ { \mathbf { c } _ { 0 } } \left( \beta _ { t } \log _ { \mathbf { c } _ { 0 } } ( \mathbf { c } _ { 1 } ) \right) , \qquad \dot { \mathbf { c } } _ { t } = \frac { \dot { \beta } _ { t } } { 1 - \beta _ { t } } \log _ { \mathbf { c } _ { t } } ( \mathbf { c } _ { 1 } ) . } \end{array}\tag{2}
$$

The exponential and logarithm maps act componentwise on M [51], with their definitions and further details regarding the crystal representation provided in Appendix A. We take $\beta _ { t } = t$ to obtain a geodesic interpolant and use a standard normal base distribution $\mathbf { c } _ { 0 } \sim \rho _ { 0 }$ , while $\mathbf { c } _ { 1 } \sim \rho _ { 1 }$ is sampled from the dataset.

Flow Matching The corresponding velocity field which bridges $\rho _ { 0 }$ and $\rho _ { 1 }$ is given by a conditional expectation $\mathbf { v } _ { t } ( \mathbf { c } ) = \mathbb { E } [ \dot { \mathbf { c } } _ { t } \mid \mathbf { c } _ { t } = \mathbf { c } ]$ . Under suitable regularity conditions, this velocity field generates the marginal probability path $\rho _ { t }$ induced by the stochastic interpolant [29]. An approximation $\mathbf { v } _ { t } ^ { \theta }$ is learned by minimizing

$$
\mathcal { L } _ { \mathrm { v e l } } ( \theta ) = \int _ { 0 } ^ { 1 } \mathbb { E } \left[ \left\| \mathbf { v } _ { t } ^ { \theta } ( \mathbf { c } _ { t } ) - \dot { \mathbf { c } } _ { t } \right\| ^ { 2 } \right] \mathrm { d } t\tag{3}
$$

where the expectation is taken over $\mathbf { c } _ { 1 } \sim \rho _ { 1 }$ and ${ \bf c } _ { 0 } \sim \rho _ { 0 }$ . Samples are generated by integrating

$$
\dot { \mathbf { c } } _ { t } = \mathbf { v } _ { t } ^ { \theta } ( \mathbf { c } _ { t } ) , \qquad \mathbf { c } _ { 0 } \sim \rho _ { 0 } , \quad t \in [ 0 , 1 ] ,\tag{4}
$$

giving terminal samples $\mathbf { c } _ { 1 } \sim \rho _ { 1 } ^ { \theta } \approx \rho _ { 1 }$ approximating the dataset.

## 3.1 Flow Maps

A two-time flow map $\mathbf { C } _ { s , u }$ directly transports samples from the marginal $\rho _ { s }$ to $\rho _ { u }$ . For $\mathbf { c } _ { s } \sim \rho _ { s }$ , it is defined as the solution to the ODE over the interval $[ s , u ]$

$$
\mathbf { C } _ { s , u } ( \mathbf { c } _ { s } ) : = \mathbf { c } _ { s } + \int _ { s } ^ { u } \mathbf { v } _ { \tau } ( \mathbf { c } _ { \tau } ) \mathrm { d } \tau .\tag{5}
$$

Rather than learning only the velocity through Equation (3), a flow map model directly approximates the solution operator $\mathbf { C } _ { s , u } ^ { \theta } \approx \mathbf { C } _ { s , u }$ in Equation (5) [31, 52]. This enables sampling from $\rho _ { 1 }$ using a small number of composed function evaluations:

$$
\mathbf { c } _ { t _ { K } } = \left( \mathbf { C } _ { t _ { K - 1 } , t _ { K } } ^ { \theta } \circ \cdot \cdot \cdot \circ \mathbf { C } _ { t _ { 0 } , t _ { 1 } } ^ { \theta } \right) \left( \mathbf { c } _ { t _ { 0 } } \right) , \qquad \mathbf { c } _ { t _ { 0 } } \sim \rho _ { 0 } , \qquad 0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { K } = 1 .\tag{6}
$$

The terminal sample still satisfies $\mathbf { c } _ { t _ { K } } \sim \boldsymbol { \rho } _ { 1 } ^ { \theta } \approx \boldsymbol { \rho } _ { 1 }$ using only K function evaluations compared to a typical numerical discretization of Equation (4). In theory $K = 1$ is suficient to draw samples from the target but, in practice, this is often too error-prone.

The flow map must satisfy a tangency condition:

$$
\begin{array} { r } { \dot { \mathbf { C } } _ { t , t } ( \mathbf { c } _ { t } ) : = \left. \frac { \partial } { \partial u } \mathbf { C } _ { t , u } ( \mathbf { c } _ { t } ) \right| _ { u = t } = \mathbf { v } _ { t } ( \mathbf { c } _ { t } ) . } \end{array}\tag{7}
$$

This condition ensures that, in the instantaneous limit, the flow map evolves according to the velocity field that generates the marginal probability path $\rho _ { t }$ . To enable few-step sampling it must also satisfy a consistency condition:

$$
\mathbf { C } _ { s , w } ( \mathbf { c } _ { s } ) = \mathbf { C } _ { v , w } ( \mathbf { C } _ { s , v } ( \mathbf { c } _ { s } ) ) , \qquad s < v < w .\tag{8}
$$

This ensures that a composition of steps from time s to v, then v to w is equivalent to a single step over the interval s to w.

![](images/04f22ed1b6e8af78d8966ba55d89525e19066bc2efd6b9f7446d2fa8cce35550.jpg)  
Figure 2: Flow Map Training. OMatG-flash training minimizes a tangency loss (Left) which, similar to standard flow matching, ensures that, at diagonal time $t ,$ the velocity associated with the instantaneous flow map is aligned with the ground truth velocity field. This is done by drawing an interpolant (dotted gray line) between noise and data and regressing onto the analytical velocity (bold gray arrow) $\dot { \mathbf { c } } _ { t }$ . In tandem, a consistency loss (Right) enforces that flow map steps can be properly composed to cover any given time range in a consistent fashion such that few-step sampling properly transports samples from base to target using an average velocity field.

Riemannian Flow Maps We employ a Riemannian MeanFlow [30] to parameterize the flow map jump $\mathbf { C } _ { s , u } ^ { \theta }$ . Specifically, we use an endpoint prediction network $\hat { \mathbf { c } } _ { 1 } ^ { \theta }$ so that the jump from time s to $u > s$ becomes

$$
\mathbf { C } _ { s , u } ^ { \theta } ( \mathbf { c } _ { s } ) = \exp _ { \mathbf { c } _ { s } } \left( \frac { u - s } { 1 - s } \log _ { \mathbf { c } _ { s } } \left( \hat { \mathbf { c } } _ { 1 } ^ { \theta } ( \mathbf { c } _ { s } , s , u ) \right) \right) .\tag{9}
$$

To compute $\hat { \mathbf { c } } _ { 1 } ^ { \theta }$ , we apply Crystalite’s neural network architecture [42]. We combine each atomic descriptor with its corresponding fractional coordinate to form one token per atom and encode the lattice as a separate token, producing N + 1 tokens in total. These tokens are then processed by a two-time scalable interpolant transformer (SiT) backbone, foregoing equivariance for speed [53, 54].

A Riemannian flow map [30] is learned by substituting its diagonal velocity $\dot { \mathbf { C } } _ { t , : } ^ { \theta }$ for the learned velocity in Equation (3) to enforce tangency. We then add a consistency loss ${ \mathcal { L } } _ { \mathrm { c o n s } }$ to enforce Equation (8). The total training objective is finally given by a weighted linear combination of a tangency loss,

$$
\mathcal { L } _ { \mathrm { t a n g } } ( \theta ) = \int _ { 0 } ^ { 1 } \mathbb { E } \bigg [ \gamma _ { t } \bigg \| \dot { \mathbf { C } } _ { t , t } ^ { \theta } ( \mathbf { c } _ { t } ) - \dot { \mathbf { c } } _ { t } \bigg \| ^ { 2 } \bigg ] ~ \mathrm { d } t ,\tag{10}
$$

and a consistency loss

$$
\mathcal { L } _ { \mathrm { c o n s } } ( \theta ) = \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { w } \int _ { s } ^ { w } \mathbb { E } \bigg [ \frac { \gamma _ { s } } { ( w - s ) ^ { 2 } } \| \log _ { \mathbf { c } _ { s } } ( \mathbf { C } _ { s , w } ^ { \theta } ( \mathbf { c } _ { s } ) ) - \mathbf { s } \mathbf { g } \big ( \log _ { \mathbf { c } _ { s } } ( \mathbf { C } _ { v , w } ^ { \theta } \big ( \mathbf { C } _ { s , v } ^ { \theta } ( \mathbf { c } _ { s } ) \big ) \big ) \big ) \| ^ { 2 } \bigg ] \ \mathrm { d } v \mathrm { ~ d } s \ \mathrm { d } w .\tag{11}
$$

A weighted combination of these yields a flow map objective for crystalline unit cells:

$$
\mathcal { L } _ { \mathrm { t o t a l } } ( \theta ) = \lambda _ { T } \mathcal { L } _ { \mathrm { t a n g } } ( \theta ) + \lambda _ { C } \mathcal { L } _ { \mathrm { c o n s } } ( \theta ) .\tag{12}
$$

Here, $\lambda _ { \mathrm { T } }$ and $\lambda _ { \mathrm { C } }$ are per-field weights controlling the relative contributions of the tangency and consistency losses, respectively, and sg denotes the stop-gradient operator. γ is a time-dependent weighting function chosen for stability [30] (Appendix B.3). The expectations are taken over endpoints $\mathbf { c } _ { 0 } \sim \rho _ { 0 }$ and $\mathbf { c } _ { 1 } \sim \rho _ { 1 }$ sampled independently. We illustrate flow map training conceptually in Figure 2.

The above losses can be viewed as self-distillation objectives. However, flow maps can also be trained using a student–teacher framework with a pretrained flow-matching velocity model. In this setting, the velocity predicted by the pretrained model at time t for a noisy sample $\mathbf { c } _ { t }$ replaces $\dot { \mathbf { c } } _ { t }$ as the regression target in Equation (3). Our method, OMatG-flash, is self-distilled and does not rely on a teacher model, demonstrating that strong flow map performance does not require prior access to a pretrained generative model. For more background on flow maps, see Appendix B.3.

Data Augmentation OMatG-flash is not translation invariant nor is it rotation equivariant. We learn translation invariance by data augmentation while our canonicalized choice of a cell representation y obviates the need for data augmentation via randomly sampled rotations as the frame is fixed. More on the architecture of OMatG-flash and the chosen data augmentation strategies are given in Appendix C.

## 3.2 Reinforce Adjoint Matching

Reinforce Adjoint Matching (RAM) is a method for performing RL to fine-tune flow-based generative models [55]. The objective is to simulate data from a tilted distribution

$$
\rho _ { 1 } ^ { \star } ( { \bf c } _ { 1 } ) \propto \rho _ { 1 } ( { \bf c } _ { 1 } ) e ^ { r ( { \bf c } _ { 1 } ) }\tag{13}
$$

given access only to a scalar-valued reward, $r ,$ and a reference velocity field, ${ \bf v } _ { t } ^ { \mathrm { r e f } }$ , which can be used to draw samples from $\rho _ { 1 }$ . The core learning task is to estimate the optimally controlled velocity field, $\begin{array} { r } { \mathbf { v } _ { t } ^ { \star } ( \mathbf { c } ) = \mathbf { v } _ { t } ^ { \mathrm { r e f } } ( \mathbf { c } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla _ { \mathbf { c } } V _ { t } ( \mathbf { c } ) } \end{array}$ where $\nabla _ { \mathbf { c } } V _ { t } ( \mathbf { c } )$ is the gradient of a value function [56, 57]. RAM proposes to learn an approximate optimal control using an expression for this value function gradient minus a kinetic term,

$$
\nabla _ { \mathbf { c } } V _ { t } ( \mathbf { c } ) \approx \mathbb { E } \big [ r ( \mathbf { c } _ { 1 } ) \nabla _ { \mathbf { c } _ { t } } \log \rho _ { 1 | t } ( \mathbf { c } _ { 1 } \mid \mathbf { c } _ { t } ) \big | \mathbf { c } _ { t } = \mathbf { c } \big ] ,\tag{14}
$$

where $\rho _ { 1 | t }$ is the conditional distribution of a clean crystal structure $\mathbf { c } _ { 1 }$ given a noisy intermediate $\mathbf { c } _ { t }$ This approximation yields a numerically stable and low-variance objective. Under this approximation, we can take a pretrained $\mathbf { v } _ { t } ^ { \theta }$ and post-train it by minimizing the following RAM loss to approximate the controlled $\mathbf { v } _ { t } ^ { \star }$

$$
\mathcal { L } _ { \mathrm { R A M } } ( \theta ) = \int _ { 0 } ^ { 1 } \mathbb { E } \left[ \left\| \mathbf { v } _ { t } ^ { \theta } ( \mathbf { c } _ { t } ) - \mathbf { s } \mathbf { g } \left( \mathbf { v } _ { t } ^ { \mathrm { r e f } } ( \mathbf { c } _ { t } ) + r ( \mathbf { c } _ { 1 } ) \left( \dot { \mathbf { c } } _ { t } - \mathbf { v } _ { t } ^ { \theta } ( \mathbf { c } _ { t } ) \right) \right) \right\| ^ { 2 } \right] \mathrm { d } t\tag{15}
$$

where ${ \bf v } _ { t } ^ { \mathrm { r e f } }$ is a frozen copy of $\mathbf { v } _ { t } ^ { \theta }$ before any optimization. Illustrated in Figure 3, this objective is estimated by generating data, $\mathbf { c } _ { 1 }$ , under the current model $\mathbf { v } _ { t } ^ { \theta }$ and noising it analytically via the interpolant in Equation (2) to a randomly sampled uniform time $t \in [ 0 , 1 ]$ ]. Unlike policy gradient variants applied to difusion- and flow-based generative models, RAM is formulated to work with deterministic inference. More background on RAM is given in Appendix E.

RAM for All-Atom Flow Maps Flow map inference is not naturally compatible with standard policy-optimization methods such as DDPO [58], PPO [59], and GRPO [60]. Flow maps define deterministic generative dynamics and learn finite-time transport rather than a stochastic policy. Policy-gradient methods require a stochastic process to formulate generation as a Markov decision process. Existing methods for applying such methods to flow matching navigate around this hurdle by either convert ing the ODE in Equation (4) to an SDE via the score or by artificially injecting noise into this ODE [46, 61]. While these approaches have shown promising results, they do not transfer naturally to flow maps, which do not admit a straightforward score-based reformulation as an SDE except for in the instantaneous limit. RAM, however, provides a path for fine-tuning a learned flow map.

We adapt RAM to enable flow map reinforcement learning. By exploiting the tangent condition in Equation (7), we replace the instantaneous velocity $\mathbf { v } _ { t } ^ { \theta }$ in Equation (15) with the diagonal derivative of the flow map $\dot { \mathbf { C } } _ { t , t } ^ { \theta }$ . We jointly maintain consistency away from the diagonal via a consistency loss, ${ \mathcal { L } } _ { \mathrm { c o n s } } ( \theta )$ , to enforce the semigroup property in Equation (8). This gives the flow map RAM loss

![](images/751b030df3a79094c37baa040a799666250bfba4710e43413826083d68bfa608.jpg)

![](images/09e674c92f1a474e4b57a578842752f94af5e4f4d31f99c3bb4ee73c2acb7924.jpg)  
Figure 3: OMatG-flash-RAM Post-Training. A conceptual overview (Left) of RAM post-training for OMatG-flash is presented. OMatG-flash is used to generate a sample, $\mathbf { c } _ { 1 }$ , using few NFE. N replicas of this structure are made and independently renoised. Renoised samples $\{ \mathbf { c } _ { t _ { i } } \} _ { i = 1 } ^ { N }$ and the reward signal $r ( \mathbf { c } _ { 1 } )$ are used to update the model via the RAM objective in Equation (16). (Right) The evolution of relative energy with respect to known crystal structures, RMSE, and match rate all trend desirably during fine-tuning. Relative energies are reported after relaxation, indicating that RAM post-training enables the model to produce structures which minimize to lower energies. Match rate and RMSE are computed before any relaxation.

$$
\mathcal { L } _ { \mathrm { R A M } } ^ { \mathrm { F M } } ( \theta ) = \lambda _ { \mathrm { R A M } } ^ { \star } \int _ { 0 } ^ { 1 } \mathbb { E } \left[ \left. \dot { C } _ { t , t } ^ { \theta } ( \mathbf { c } _ { t } ) - \mathrm { s g } \left( \dot { C } _ { t , t } ^ { \mathrm { r e f } } ( \mathbf { c } _ { t } ) + r ( \mathbf { c } _ { 1 } ) \left( \dot { \mathbf { c } } _ { t } - \dot { C } _ { t , t } ^ { \theta } ( \mathbf { c } _ { t } ) \right) \right) \right. ^ { 2 } \right] \mathrm { d } t + \lambda _ { C } ^ { \star } \mathcal { L } _ { \mathrm { c o n s } } ( \theta )\tag{16}
$$

with chosen weights $\lambda _ { \mathrm { R A M } } ^ { \star }$ and $\lambda _ { C } ^ { \star }$ indicated with a star to denote post-training. Minimizing this objective requires no reparameterization of the learned flow map nor does it require any sort of noise injection which would corrupt the generative dynamics.

We implement RAM post-training only for the CSP task using the energy E as a reward model

$$
r ( \mathbf { c } _ { 1 } ) = - \lambda _ { E } E ( \mathbf { c } _ { 1 } ) .\tag{17}
$$

where $\lambda _ { E }$ is a hyperparameter. This reward pushes the generated distribution towards more stable regions of the energy surface, improving performance at the CSP task. We use the MACE-MPA-0 foundational machine learned interatomic potential (MLIP) as the chosen energy model for this task to balance high-fidelity energy evaluation with speed [13]. To stabilize post-training we compute the reward in Equation (17) after taking 50 relaxation steps for each generated structure $\mathbf { c } _ { 1 }$ . More detail on the OMatG-flash-RAM implementation is provided in Appendix E.

## 4 Experiments

We train OMatG-flash on two inorganic materials datasets to benchmark performance on CSP and DNG: MP-20, which comprises 45,231 crystal structures curated from the Materials Project [7], and Alex-MP-20, which is a larger dataset comprising 675,204 structures from the Materials Project and the Alexandria database [9]. Each dataset contains crystals with at most 20 atoms in the unit cell and covers a diverse set of compositions and structures. We evaluate OMatG-flash (pretrained) for both CSP and DNG of inorganic materials and OMatG-flash-RAM (post-trained) for CSP. All reported metrics for competing models are given as reported in existing work [62, 63].

Table 1: CSP Performance on MP-20 and Alex-MP-20. OMatG-flash and OMatG-flash-RAM demonstrate strong performance on one-to-one match rate where the subscript indicates the NFE used for inference. Values unavailable in the literature are indicated with dashes. k indicates that inference was performed k times and the best match from each generated set is taken. Each slash-separated entry reports valid or invalid / valid, where the first value is computed over all generated structures and the second only over structures classified as valid. Bold indicates best overall and underlined indicates best on a k = 1 budget.
<table><tr><td rowspan="2">Method</td><td rowspan="2">k</td><td colspan="4">MP-20</td><td colspan="4">Alex-MP-20</td></tr><tr><td>MR</td><td>(%) ↑</td><td colspan="2">RMSE↓</td><td colspan="2">MR (%) ↑</td><td colspan="2">RMSE↓</td></tr><tr><td>DiffCSP</td><td>1</td><td>57.82</td><td>52.51</td><td>0.0627</td><td>0.0600</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>FlowMM</td><td>1</td><td>66.22</td><td>59.98</td><td>0.0661</td><td>0.0629</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>MCFlow</td><td>1</td><td>70.38</td><td>/64.08</td><td>0.0592</td><td>0.0561</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>OMatG</td><td>1</td><td>69.83</td><td>63.75</td><td>0.0741</td><td>0.0720</td><td>72.50</td><td>64.71</td><td>0.1261</td><td>0.1251</td></tr><tr><td>Crystalite</td><td>1</td><td></td><td>66.09</td><td></td><td>0.0337</td><td></td><td>68.26</td><td></td><td>0.0317</td></tr><tr><td rowspan="3">OMatG-flash16</td><td>1</td><td>67.15</td><td>60.79</td><td>0.1172</td><td>0.1136</td><td>66.19</td><td>59.05</td><td>0.1356</td><td>0.1342</td></tr><tr><td>3</td><td>75.47</td><td>68.62</td><td>0.0954</td><td>0.0930</td><td>80.76</td><td>72.39</td><td>0.1055</td><td>0.1042</td></tr><tr><td>5</td><td>78.27</td><td>71.19</td><td>0.0878</td><td>0.0860</td><td>85.68</td><td>76.91</td><td>0.0961</td><td>0.0948</td></tr><tr><td rowspan="3">OMatG-flash₁₆-RAM</td><td>1</td><td>68.90</td><td>62.55</td><td>0.0881</td><td>0.0858</td><td>70.52</td><td>62.87</td><td>0.0955</td><td>0.0939</td></tr><tr><td>3</td><td>75.76</td><td>68.89</td><td>0.0743</td><td>0.0728</td><td>81.05</td><td>72.55</td><td>0.0807</td><td>0.0792</td></tr><tr><td>5</td><td>78.30</td><td>71.30</td><td>0.0713</td><td>0.0702</td><td>84.84</td><td>76.08</td><td>0.0764</td><td>0.0749</td></tr></table>

OMatG-flash is trained on the canonical data splits for both of these datasets [17, 20] as well as the new polymorph-split MP-20-ps [64]. Results with respect to the original data splits are given solely to provide a fair comparison to existing work and we strongly urge future generative modeling eforts to train, evaluate, and compare to the polymorph split (ps) versions of this data using appropriate polymorph-aware metrics as plain MP-20 has known leakage across splits [64].

## 4.1 Crystal Structure Prediction

A central challenge in inorganic materials design is to predict crystal structure given only the constituent elements of atoms in the unit cell. The performance of OMatG-flash at CSP is primarily assessed via match rates and root-mean-squared-error (RMSE) between generated and reference crystals using Pymatgen’s StructureMatcher algorithm which is used to report the fraction of generated crystal structures which match an index-paired test-set structure up to a chosen tolerance. While informative, match rates depend on arbitrary indexing which is flawed when considering the polymorphism inherent to crystalline materials [64]. We therefore report and highlight our results on METRe (match-everyoneto-reference) and cRMSE, recently proposed polymorph-aware metrics which improve upon the flawed match rate benchmark. METRe addresses the failures of one-to-one match rate by instead searching for a match among all generated structures for each test set structure to remove any dependence on arbitrary indexing. The associated RMSE values measure the similarity between two matched structures in terms of a unitless distance whereas corrected RMSE (cRMSE) adds a tolerance-based penalty for each non-matching case (Appendix F). For completeness we report both one-to-one match rates and RMSE along with METRe and cRMSE, but strongly urge wider adoption of polymorph-aware METRe and cRMSE. We report the performance of OMatG-flash and OMatG-flash-RAM in Tables 1 and 2. Post-training curves of the match rate, RMSE, and relative energy are shown in Figure 3.

## 4.2 De Novo Generation

DNG is the most general materials design problem, requiring proposal of not only the crystal structure but also the composition of elements within a crystal. We benchmark OMatG-flash’s performance at this task by computing stability, uniqueness, and novelty (S.U.N.) and metastability, uniqueness, and novelty (M.S.U.N.) with respect to a known corpus of reference data [63, 65]. We rely on LeMat-GenBench [63] — an open-source leaderboard of generative models for inorganic materials generation — to compute S.U.N. and M.S.U.N. on 2500 sampled structures in Table 3. Stability and metastability are determined by comparing the energy of each generated structure to a convex hull of stable phases. Uniqueness measures the proportion of generated structures which are distinct chemically or structurally from all other generated structures whereas novelty identifies if a structure is distinct with respect to a chosen reference dataset of inorganic crystals. To identify matching crystals for uniqueness and novelty we use pymatgen’s StructureMatcher algorithm. More information on the definition and practical calculation of these metrics is given in Appendix F. We report metrics for models trained on the MP-20 dataset as this dataset has the most reported entries on the LeMat-GenBench leaderboard.

## 5 Discussion

## 5.1 Results

For both CSP and DNG we compare OMatG-flash to unconditional all-atom difusion- and flow-based models operating on crystalline unit cells. This is done to show how such models for materials can be amortized via a flow map. There are a variety of models which condition on space group to generate more symmetric structures [66–68]. Generative models utilizing diferent representations such as asymmetric units [69] or learned latent spaces [70] have also been shown to yield promising performance for materials. Lastly, language models and unmasking transformer architectures have also demonstrated success for CSP and DNG [43, 47, 62]. MaskGXT [62] is of this kind and is the current state-of-the-art overall for CSP on MP-20 and MP-20-ps with benchmark performance that exceeds ours. Their inference algorithm performs a double rollout, conditioning the second on the space group representation produced by the first, resulting in improved performance over other CSP models. For DNG Crystalite [42] is the current state-of-the-art and is reported as it is an unconditional difusion model. A consensus on which architecture, crystal representation, and choice of conditioning signal is most advantageous for CSP and DNG is still an open question and deserves dedicated study which is out of the scope of this work.

CSP Results Among existing unconditional difusion- and flow-based generative models OMatG-flash-RAM is state-of-the-art on METRe for MP-20-ps while the base OMatG-flash model is competitive with these methods. Post-training OMatG-flash proves remarkably efective for the CSP task overall, simultaneously improving ME-TRe and cRMSE (Table 2) as well as RMSE in every case (Table 1), demonstrating the efectiveness of the proposed RAM extension and underscoring the potential for flow maps to accelerate high-quality CSP. While our results for k > 1 are not meant to be an apples-to-apples comparison with

Table 2: CSP Performance on MP-20-ps. OMatGflash-RAM demonstrates state-of-the-art performance on polymorph-aware METRe with respect to existing flow and difusion models while improving cRMSE. Best is bolded and second-best is underlined.
<table><tr><td>Method</td><td>METRe (%) ↑ cRMSE↓</td></tr><tr><td>DiffCSP</td><td>53.14 0.279</td></tr><tr><td>FlowMM</td><td>65.18 0.226</td></tr><tr><td>MCFlow</td><td>70.70 0.195</td></tr><tr><td>OMatG</td><td>70.50 0.187</td></tr><tr><td>Crystalite</td><td>70.87 0.174</td></tr><tr><td>OMatG-flash16</td><td>68.56 0.250</td></tr><tr><td>OMatG-flash₁₆-RAM</td><td>71.10 0.221</td></tr></table>

existing results against $k = 1$ , we argue that our findings emphasize the benefit of rapid, high-quality inference over more expensive inference with slightly improved raw performance. On MP-20, performing k-inference greatly enhances the one-to-one match rates of OMatG-flash. OMatG-flash enjoys an order of magnitude speedup over existing models. Consequently, k > 1 inference can be done faster than k = 1 for conventional flow and difusion models to yield stronger numbers.

Table 3: DNG Performance. OMatG-flash matches or exceeds all but Crystalite [42] on S.U.N. (all NFE budgets) and combined S.U.N. and M.S.U.N. (8 or 16 NFE). Novelty is second only to MatterGen for 4 NFE [20]. Quality of proposed crystals is robust even to very few inference steps. OMatG-flash metrics are computed using LeMat-GenBench code for 2500 structures. Metrics for all other models appear as reported on the leaderboard. Best is bolded and second-best is underlined. An asterisk indicates that samples were not prerelaxed before benchmarking according to LeMat-GenBench.
<table><tr><td>Method</td><td>Valid ↑</td><td>Novel ↑</td><td>Stable ↑</td><td>S.U.N. ↑</td><td>S.U.N. + M.S.U.N. ↑</td></tr><tr><td>DiffCSP*</td><td>95.7</td><td>66.2</td><td>2.3</td><td>0.1</td><td>8.6</td></tr><tr><td>MatterGen</td><td>95.7</td><td>70.5</td><td>2.0</td><td>0.2</td><td>15.2</td></tr><tr><td>OMatG</td><td>96.4</td><td>51.2</td><td>11.6</td><td>1.0</td><td>19.0</td></tr><tr><td>MCFlow</td><td>97.2</td><td>52.2</td><td>11.9</td><td>0.7</td><td>19.6</td></tr><tr><td>Crystalite</td><td>97.2</td><td>53.2</td><td>12.7</td><td>1.5</td><td>24.1</td></tr><tr><td>OMatG-flash₄</td><td>96.4</td><td>66.9</td><td>7.2</td><td>1.4</td><td>18.2</td></tr><tr><td> $\mathrm { O M a t G  – f l a s h } _ { 8 }$ </td><td>96.8</td><td>63.4</td><td>7.8</td><td>1.0</td><td>21.5</td></tr><tr><td> $\mathrm { O M a t G – f l a s h } _ { 1 6 }$ </td><td>96.8</td><td>61.0</td><td>8.8</td><td>1.2</td><td>22.6</td></tr></table>

Despite high METRe and match rates, the cRMSE and RMSE between ground truth polymorphs and samples produced by OMatG-flash is systematically high for both polymorph-aware and one-to-one matching indicating that, while many structures can be successfully matched to a test set structure, their error relative to that reference is higher than that of a difusion- or flow-based generative model. We argue that this can be attributed to compounding errors in the model exacerbated by performing very few inference steps, a trend that has been observed in few-step inference for physical science applications [30]. We also highlight that post-training greatly improves cRMSE and k-inference RMSE which suggests that reinforcement with respect to domain-specific rewards can correct for the heightened sensitivity to model error that standard flow maps incur to perform amortized inference. Furthermore, assuming that production inference for materials discovery is always followed by MLIP relaxation as in DNG with LeMat-GenBench, we argue that generating structures within the appropriate basin of attraction is of more immediate concern than completely resolving structure.

DNG Results OMatG-flash exhibits high-quality performance that remains robust with as few as four model calls per inference. The novelty of structures proposed by OMatG-flash is high on average, trailing only MatterGen [20] at 4 NFE while exceeding MatterGen’s stability rate by a large margin. The S.U.N. and combined S.U.N. and M.S.U.N. rates of structures generated by OMatG-flash rival the state of the art at a fraction of the inference cost. Except for DifCSP, competing results given in Table 3 are among those which are denoted as prerelaxed on LeMat-GenBench to provide a fair comparison between OMatG-flash and existing models.

We highlight that OMatG-flash-RAM is not benchmarked on this task. As identified in previous work, the S.U.N. and M.S.U.N. metrics are capable of being “hacked” by the naive application of RL [45, 46, 71]. A model can quickly learn that certain compositions and classes of structures are broadly stable or metastable, focusing generation on repetitive materials to maximize the reward. For CSP this is less of an issue, as every proposed material is conditioned on a fixed chemical composition which imposes structure on the energy surface that is dificult to exploit. We argue that S.U.N. and M.S.U.N. are more relevant for assessing performance during pretraining but that post-training reinforcement should be targeted to chemical and structural features to target a specific physical property as opposed to stability or novelty in the broadest sense. We do not consider RAM post-training for DNG in this work and leave post-training of OMatG-flash and reward design to this end as future work.

## 5.2 Pareto-Optimal Materials Generation

Flow and difusion-based models for materials design have demonstrated remarkable capability for sampling stable and novel materials, but the primary advantage of OMatG-flash is scalability at inference time. OMatG-flash demonstrates performance competitive with the state-of-the-art but with massively improved inference-time throughput. In Figure 1 we construct a Pareto frontier of performance vs. inference speed for the DNG task to illustrate this. OMatG-flash exhibits Pareto-optimal performance with respect to existing flow and difusion models, surpassing all except Crystalite in terms of accuracy on LeMat-GenBench with at least an order of magnitude speedup over existing models. Advances in generative AI for science continue to push existing benchmarks further. While improvement in these raw performance indicators is crucial, advancement along other axes such as inference and training speed as well as cost is necessary for scaling high-quality sampling either through hardware or new algorithms. OMatG-flash sets the state-of-the-art in balancing rapid inference with high-quality generation.

Conventionally, human judgment and expensive quantum mechanical calculations for property estimation and relaxation have been the bottleneck in materials-discovery as they were often integrated in the proposal engine. As agentic workflows, GPU-accelerated scientific algorithms, and high-accuracy MLIPs are increasingly amortizing and accelerating these downstream filters [50, 62, 72, 73], efort should similarly be focused on amortizing the generation mechanism itself. With this in mind, OMatGflash is the most capable tool for truly saturating such automated systems and is well-suited to the modern materials science workflow.

## 5.3 Limitations and Future Work

A key limitation of OMatG-flash is the high reported RMSE between generated samples from the base model and ground truth data. This, coupled with high METRe rates, indicates that, while OMatG-flash is quite capable of isolating stable crystalline motifs, it struggles to resolve fine-grained structure. While post-training notably improves the quality of generated samples in terms of RMSE for the CSP task, improving the quality of the base model is of immediate importance. On this note, a possible future direction is to better understand and stabilize RAM post-training for the flow map setting (Appendix E).

An application for OMatG-flash is as an engine for inference time steering where the aim is to take a pretrained generative model and guide it towards desired rewards without changing the weights themselves [74, 75]. Recently, there has been increased interest in using flow maps in conjunction with difusion or flow-based generative models to “look ahead” to t = 1, compute a reward, and to use the reward or its gradient to bias inference towards regions of feature space with high rewards [57, 76, 77]. A conditional version of OMatG-flash could be developed which would open the possibility of guidance methods which are grounded in stochastic optimal control [57, 76].

## 6 Conclusion

In this work we introduced OMatG-flash as an all-atom flow map for materials generation and showcased its capability for rapid inference of high-quality inorganic material candidates. Furthermore, we demonstrated post-training capability to facilitate fine-tuning of OMatG-flash based on desired rewards. Benchmarking on Materials Project data, we show that OMatG-flash is a Pareto-optimal materials generator and pushes the boundaries of inference-time speed while maintaining raw performance on par with the state-of-the-art. As materials discovery shifts toward fully automated agentic workflows and accelerated postprocessing, generation time can no longer be treated as a secondary metric. When one model can generate thousands of candidates in the time another generates one, small diferences in per-sample accuracy are overwhelmed by the diference in throughput.

## 7 Code and Data Availability

The code associated with this work will be released as part of the OMatG package. Relevant model checkpoints will be uploaded to Hugging Face at https://huggingface.co/OMatG. The MP-20 and Alex-MP-20 datasets used for training and evaluating OMatG-flash are all open source.

## 8 Acknowledgments

The authors thank the NYU IT High Performance Computing team for their provision of computational resources and general support. The authors acknowledge funding from NSF Grant OAC-2311632. S. M. acknowledges support from the Simons Center for Computational Physical Chemistry (Simons Foundation grant 839534, MT). The authors gratefully acknowledge use of the research computing resources of the Empire AI Consortium, Inc., with support from the State of New York, the Simons Foundation, and the Secunda Family Foundation.

Large language models assisted with manuscript drafting and editing and research code development;   
the authors verified all results and claims and take full responsibility for the work.

## References

[1] Wei Chen et al. “High-throughput computational discovery of In2Mn2O7 as a high Curie temperature ferromagnetic semiconductor for spintronics”. en. In: npj Computational Materials 5.1 (July 2019), p. 72. issn: 2057-3960. doi: 10.1038/s41524-019-0208-x. url: https://www. nature.com/articles/s41524-019-0208-x (visited on 08/28/2026).

[2] Kinga Pielichowska and Krzysztof Pielichowski. “Phase Change Materials for Thermal Energy Storage”. In: Progress in Materials Science 65 (Aug. 2014), pp. 67–123. issn: 0079-6425. doi: 10.1016/j.pmatsci.2014.03.005. (Visited on 08/27/2026).

[3] Peter G. Boyd et al. “Data-driven design of metal–organic frameworks for wet flue gas CO2 capture”. en. In: Nature 576.7786 (Dec. 2019), pp. 253–256. issn: 1476-4687. doi: 10 . 1038 / s41586-019-1798-7. url: https://www.nature.com/articles/s41586-019-1798-7 (visited on 08/28/2026).

[4] Radislav Potyrailo et al. “Combinatorial and High-Throughput Screening of Materials Libraries: Review of State of the Art”. In: ACS Combinatorial Science 13.6 (June 2011), pp. 579–633. issn: 2156-8952. doi: 10.1021/co200007w. (Visited on 08/27/2026).

[5] Pierre-Paul De Breuck et al. “Generative AI for crystal structures: a review”. en. In: npj Computational Materials 11.1 (Dec. 2025), p. 370. issn: 2057-3960. doi: 10.1038/s41524-025-01881-2. url: https://www.nature.com/articles/s41524-025-01881-2 (visited on 08/29/2026).

[6] Chris J. Pickard and R. J. Needs. “Ab Initio Random Structure Searching”. In: Journal of Physics: Condensed Matter 23.5 (Feb. 2011), p. 053201. issn: 0953-8984, 1361-648X. doi: 10.1088/0953- 8984/23/5/053201. arXiv: 1101.3987 [cond-mat.mtrl-sci]. (Visited on 08/13/2026).

[7] Anubhav Jain et al. “The Materials Project: A materials genome approach to accelerating materials innovation”. In: APL Materials 1.1 (2013), p. 011002. issn: 2166532X. doi: 10.1063/1. 4812323. url: http://link.aip.org/link/AMPADS/v1/i1/p011002/s1%5C&Agg=doi.

[8] D. Zagorac et al. “Recent Developments in the Inorganic Crystal Structure Database: Theoretical Crystal Structure Data and Related Features”. In: Journal of Applied Crystallography 52.5 (2019), pp. 918–925. issn: 1600-5767. doi: 10.1107/S160057671900997X. (Visited on 08/16/2026).

[9] Th´eo Cavignac et al. “AI-Driven expansion and application of the Alexandria database”. In: Journal of Physics: Materials 9.2 (2026), p. 025014. doi: 10.1088/2515- 7639/ae6620. url: https://doi.org/10.1088/2515-7639/ae6620.

[10] Shyue Ping Ong et al. “Python Materials Genomics (Pymatgen): A Robust, Open-Source Python Library for Materials Analysis”. In: Computational Materials Science 68 (Feb. 2013), pp. 314–319. issn: 0927-0256. doi: 10.1016/j.commatsci.2012.10.028. (Visited on 08/16/2026).

[11] Ask Hjorth Larsen et al. “The Atomic Simulation Environment—a Python Library for Working with Atoms”. In: Journal of Physics: Condensed Matter 29.27 (June 2017), p. 273002. issn: 0953- 8984. doi: 10.1088/1361-648X/aa680e. (Visited on 08/16/2026).

[12] Simon Batzner et al. “E(3)-Equivariant Graph Neural Networks for Data-Eficient and Accurate Interatomic Potentials”. In: Nature Communications 13.1 (May 2022), p. 2453. issn: 2041-1723. doi: 10.1038/s41467-022-29939-5. (Visited on 03/22/2024).

[13] Ilyes Batatia et al. “A foundation model for atomistic materials chemistry”. In: (2023). arXiv: 2401.00096 [physics.chem-ph].

[14] Brandon M. Wood et al. UMA: A Family of Universal Models for Atoms. June 2025. doi: 10. 48550/arXiv.2506.23971. arXiv: 2506.23971 [cs]. (Visited on 01/20/2026).

[15] Geofroy Hautier et al. “Data Mined Ionic Substitutions for the Discovery of New Compounds”. In: Inorganic Chemistry 50.2 (Dec. 2010), pp. 656–663. issn: 0020-1669. doi: 10.1021/ic102031h. (Visited on 08/16/2026).

[16] Amil Merchant et al. “Scaling Deep Learning for Materials Discovery”. In: Nature 624.7990 (Dec. 2023), pp. 80–85. issn: 1476-4687. doi: 10.1038/s41586-023-06735-9. (Visited on 08/15/2026).

[17] Tian Xie et al. Crystal Difusion Variational Autoencoder for Periodic Material Generation. Mar. 2022. doi: 10.48550/arXiv.2110.06197. arXiv: 2110.06197 [cs]. (Visited on 01/17/2025).

[18] Rui Jiao et al. Crystal Structure Prediction by Joint Equivariant Difusion. Mar. 2024. arXiv: 2309.04475 [cond-mat]. (Visited on 04/17/2024).

[19] Benjamin Kurt Miller et al. FlowMM: Generating Materials with Riemannian Flow Matching. June 2024. arXiv: 2406.04713 [cond-mat, physics:physics, stat]. (Visited on 06/18/2024).

[20] Claudio Zeni et al. MatterGen: A Generative Model for Inorganic Materials Design. Jan. 2024. arXiv: 2312.03687 [cond-mat]. (Visited on 04/17/2024).

[21] Philipp Hoellmer et al. Open Materials Generation with Stochastic Interpolants. July 2025. doi: 10.48550/arXiv.2502.02582. arXiv: 2502.02582 [cs]. (Visited on 01/20/2026).

[22] Connor W. Coley et al. “A Robotic Platform for Flow Synthesis of Organic Compounds Informed by AI Planning”. In: Science 365.6453 (Aug. 2019), eaax1566. doi: 10.1126/science.aax1566. (Visited on 08/27/2026).

[23] B. P. MacLeod et al. “Self-Driving Laboratory for Accelerated Discovery of Thin-Film Materials”. In: Science Advances 6.20 (May 2020), eaaz8867. doi: 10.1126/sciadv.aaz8867. (Visited on 08/27/2026).

[24] Orion Cohen et al. TorchSim: An Eficient Atomistic Simulation Engine in PyTorch. Aug. 2025. doi: 10 . 48550 / arXiv . 2508 . 06628. arXiv: 2508 . 06628 [physics.comp-ph]. (Visited on 08/27/2026).

[25] Michael S. Albergo, Nicholas M. Bofi, and Eric Vanden-Eijnden. Stochastic Interpolants: A Unifying Framework for Flows and Difusions. Nov. 2023. arXiv: 2303.08797 [cond-mat]. (Visited on 05/28/2024).

[26] Jascha Sohl-Dickstein et al. Deep Unsupervised Learning Using Nonequilibrium Thermodynamics. Nov. 2015. arXiv: 1503.03585 [cond-mat, q-bio, stat]. (Visited on 03/20/2024).

[27] Yang Song et al. Score-Based Generative Modeling through Stochastic Diferential Equations. Feb. 2021. arXiv: 2011.13456 [cs, stat]. (Visited on 04/17/2024).

[28] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. Sept. 2022. doi: 10.48550/arXiv.2209.03003. arXiv: 2209.03003 [cs.LG]. (Visited on 08/13/2026).

[29] Yaron Lipman et al. Flow Matching for Generative Modeling. Feb. 2023. arXiv: 2210.02747 [cs, stat]. (Visited on 04/14/2024).

[30] Dongyeop Woo et al. Riemannian MeanFlow. May 2026. doi: 10.48550/arXiv.2602.07744. arXiv: 2602.07744 [cs.LG]. (Visited on 07/03/2026).

[31] Nicholas M. Bofi, Michael S. Albergo, and Eric Vanden-Eijnden. Flow Map Matching. June 2024. arXiv: 2406.07507 [cs, math]. (Visited on 06/15/2024).

[32] Nicholas M. Bofi, Michael S. Albergo, and Eric Vanden-Eijnden. How to Build a Consistency Model: Learning Flow Maps via Self-Distillation. Oct. 2025. doi: 10.48550/arXiv.2505.18825. arXiv: 2505.18825 [cs.LG]. (Visited on 08/13/2026).

[33] Jaehoon Yoo et al. Self-Conditioned Flow Map Language Models via Fixed-point Flows. July 2026. doi: 10.48550/arXiv.2607.00714. arXiv: 2607.00714 [cs.CL]. (Visited on 08/29/2026).

[34] Gianluca Scarpellini et al. Few-Step Cofolding with All-Atom Flow Maps. June 2026. doi: 10. 48550/arXiv.2606.08375. arXiv: 2606.08375 [cs.LG]. (Visited on 08/13/2026).

[35] Chanhyuk Lee et al. Flow Map Language Models: One-step Language Modeling via Continuous Denoising. arXiv:2602.16813 [cs.CL]. May 2026. doi: 10.48550/arXiv.2602.16813. url: http: //arxiv.org/abs/2602.16813 (visited on 08/28/2026).

[36] Winfried Ripken et al. Learning Hamiltonian Flow Maps: Mean Flow Consistency for Large-Timestep Molecular Dynamics. arXiv:2601.22123 [cs.LG]. June 2026. doi: 10 . 48550 / arXiv . 2601.22123. url: http://arxiv.org/abs/2601.22123 (visited on 08/28/2026).

[37] Danyal Rehman et al. FALCON: Few-step Accurate Likelihoods for Continuous Flows. Dec. 2025. doi: 10.48550/arXiv.2512.09914. arXiv: 2512.09914 [cs.LG]. (Visited on 08/26/2026).

[38] RuiKang OuYang et al. Few-Step Boltzmann Generators via Scalable Likelihood Flow Maps. June 2026. doi: 10.48550/arXiv.2606.29110. arXiv: 2606.29110 [cs.LG]. (Visited on 08/26/2026).

[39] Cheng Zeng et al. MolCrystalFlow: Molecular Crystal Structure Prediction via Flow Matching. arXiv:2602.16020 [cs.LG]. Mar. 2026. doi: 10.48550/arXiv.2602.16020. url: http://arxiv. org/abs/2602.16020 (visited on 08/28/2026).

[40] Alston Lo et al. Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching. en. arXiv:2606.03199 [cs.LG]. June 2026. doi: 10.48550/arXiv.2606.03199. url: http://arxiv. org/abs/2606.03199 (visited on 08/28/2026).

[41] Fran¸cois Cornet et al. Kinetic Langevin Difusion for Crystalline Materials Generation. July 2025. doi: 10.48550/arXiv.2507.03602. arXiv: 2507.03602 [cs.LG]. (Visited on 08/13/2026).

[42] Tin Hadˇzi Veljkovi´c et al. Crystalite: A Lightweight Transformer for Eficient Crystal Modeling. July 2026. doi: 10.48550/arXiv.2604.02270. arXiv: 2604.02270 [cs.LG]. (Visited on 08/13/2026).

[43] Nikita Kazeev et al. Wyckof Transformer: Generation of Symmetric Crystals. June 2025. doi: 10. 48550/arXiv.2503.02407. arXiv: 2503.02407 [cond-mat.mtrl-sci]. (Visited on 08/13/2026).

[44] Pawan Prakash et al. “Guided Difusion for the Discovery of New Superconductors”. In: npj Computational Materials (May 2026). issn: 2057-3960. doi: 10.1038/s41524- 026- 02117- 7. (Visited on 08/13/2026).

[45] Hyunsoo Park and Aron Walsh. Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning. Nov. 2025. doi: 10.48550/arXiv.2511.07158. arXiv: 2511.07158 [cs.LG]. (Visited on 08/15/2026).

[46] Philipp Hoellmer and Stefano Martiniani. Open Materials Generation with Inference-Time Reinforcement Learning. June 2026. doi: 10.48550/arXiv.2602.00424. arXiv: 2602.00424 [cs.LG]. (Visited on 07/04/2026).

[47] Luis M. Antunes, Keith T. Butler, and Ricardo Grau-Crespo. “Crystal Structure Generation with Autoregressive Large Language Modeling”. In: Nature Communications 15.1 (Dec. 2024), p. 10570. issn: 2041-1723. doi: 10.1038/s41467-024-54639-7. (Visited on 08/13/2026).

[48] Andy Xu et al. PLaID++: A Preference Aligned Language Model for Targeted Inorganic Materials Design. June 2026. doi: 10.48550/arXiv.2509.07150. arXiv: 2509.07150 [cs.LG]. (Visited on 08/13/2026).

[49] Cyprien Bone et al. Discovery and Recovery of Crystalline Materials with Property-Conditioned Transformers. June 2026. doi: 10.48550/arXiv.2511.21299. arXiv: 2511.21299 [cond-mat.mtrl-sci]. (Visited on 08/13/2026).

[50] Ferdous Nasri et al. Deterministic access to global viral sequence data enables robust agentic scientific discovery. arXiv:2606.06749 [q-bio.QM]. June 2026. doi: 10.48550/arXiv.2606.06749. url: http://arxiv.org/abs/2606.06749 (visited on 08/28/2026).

[51] Louis Grenioux et al. Boltzmann Generators for Amorphous Particle Systems. July 2026. doi: 10.48550/arXiv.2512.16607. arXiv: 2512.16607 [stat.ML]. (Visited on 08/27/2026).

[52] Yang Song et al. Consistency Models. May 2023. doi: 10 . 48550 / arXiv . 2303 . 01469. arXiv: 2303.01469 [cs.LG]. (Visited on 08/13/2026).

[53] Nanye Ma et al. SiT: Exploring Flow and Difusion-based Generative Models with Scalable Interpolant Transformers. arXiv:2401.08740 [cs.CV]. Sept. 2024. doi: 10.48550/arXiv.2401.08740. url: http://arxiv.org/abs/2401.08740 (visited on 08/28/2026).

[54] William Peebles and Saining Xie. Scalable Difusion Models with Transformers. Mar. 2023. arXiv: 2212.09748 [cs]. (Visited on 05/30/2024).

[55] Andreas Bergmeister et al. Reinforce Adjoint Matching: Scaling RL Post-Training of Difusion and Flow-Matching Models. arXiv:2605.10759 [cs.LG]. May 2026. doi: 10.48550/arXiv.2605.10759. url: http://arxiv.org/abs/2605.10759 (visited on 08/29/2026).

[56] Carles Domingo-Enrich et al. Adjoint Matching: Fine-tuning Flow and Difusion Generative Models with Memoryless Stochastic Optimal Control. Jan. 2025. doi: 10.48550/arXiv.2409.08861. arXiv: 2409.08861 [cs.LG]. (Visited on 08/28/2026).

[57] Peter Potaptchik et al. Meta Flow Maps Enable Scalable Reward Alignment. June 2026. doi: 10.48550/arXiv.2601.14430. arXiv: 2601.14430 [stat.ML]. (Visited on 08/26/2026).

[58] Kevin Black et al. Training Difusion Models with Reinforcement Learning. arXiv:2305.13301 [cs.LG]. Jan. 2024. doi: 10.48550/arXiv.2305.13301. url: http://arxiv.org/abs/2305. 13301 (visited on 08/29/2026).

[59] John Schulman et al. Proximal Policy Optimization Algorithms. Aug. 2017. doi: 10 . 48550 / arXiv.1707.06347. arXiv: 1707.06347 [cs.LG]. (Visited on 08/27/2026).

[60] Zhihong Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. en. arXiv:2402.03300 [cs]. Apr. 2024. doi: 10.48550/arXiv.2402.03300. url: http://arxiv.org/abs/2402.03300 (visited on 03/30/2026).

[61] Jie Liu et al. Flow-GRPO: Training Flow Matching Models via Online RL. Oct. 2025. doi: 10. 48550/arXiv.2505.05470. arXiv: 2505.05470 [cs.CV]. (Visited on 07/03/2026).

[62] Kiyoung Seong, Nayoung Kim, and Sungsoo Ahn. Discovering Crystal Structure Prediction Algorithms with an AI Co-Scientist. June 2026. doi: 10.48550/arXiv.2606.22866. arXiv: 2606. 22866 [cs.LG]. (Visited on 08/13/2026).

[63] Siddharth Betala et al. LeMat-GenBench: A Unified Evaluation Framework for Crystal Generative Models. Jan. 2026. doi: 10.48550/arXiv.2512.04562. arXiv: 2512.04562 [cs.LG]. (Visited on 08/13/2026).

[64] Maya M. Martirossyan et al. All That Structure Matches Does Not Glitter. Sept. 2025. doi: 10.48550/arXiv.2509.12178. arXiv: 2509.12178 [cs]. (Visited on 10/05/2025).

[65] Martin Siron et al. LeMat-Bulk: Aggregating, and de-Duplicating Quantum Chemistry Materials Databases. Nov. 2025. doi: 10.48550/arXiv.2511.05178. arXiv: 2511.05178 [cond-mat.mtrl-sci]. (Visited on 08/29/2026).

[66] Daniel Levy et al. SymmCD: Symmetry-Preserving Crystal Generation with Difusion Models. https://arxiv.org/abs/2502.03638v3. Feb. 2025. (Visited on 08/27/2026).

[67] Rees Chang et al. Space Group Equivariant Crystal Difusion. Oct. 2025. doi: 10.48550/arXiv. 2505.10994. arXiv: 2505.10994 [cond-mat.mtrl-sci]. (Visited on 08/27/2026).

[68] Omri Puny, Yaron Lipman, and Benjamin Kurt Miller. Space Group Conditional Flow Matching. Sept. 2025. doi: 10.48550/arXiv.2509.23822. arXiv: 2509.23822 [cs.LG]. (Visited on 08/27/2026).

[69] Rees Chang et al. SLayerGen: A Crystal Generative Model for All Space and Layer Groups. May 2026. doi: 10.48550/arXiv.2605.08262. arXiv: 2605.08262 [cond-mat.mtrl-sci]. (Visited on 08/29/2026).

[70] Chaitanya K. Joshi et al. All-Atom Difusion Transformers: Unified Generative Modelling of Molecules and Materials. May 2025. doi: 10.48550/arXiv.2503.03965. arXiv: 2503.03965 [cs]. (Visited on 08/23/2025).

[71] Junwu Chen et al. Accelerating Inverse Materials Design Using Generative Difusion Models with Reinforcement Learning. Nov. 2025. doi: 10.48550/arXiv.2511.03112. arXiv: 2511.03112 [physics.chem-ph]. (Visited on 08/15/2026).

[72] Ali E. Ghareeb et al. “A multi-agent system for automating scientific discovery”. en. In: Nature 655.8122 (July 2026), pp. 497–505. issn: 1476-4687. doi: 10.1038/s41586-026-10652-y. url: https://www.nature.com/articles/s41586-026-10652-y (visited on 08/29/2026).

[73] Ilyes Batatia et al. MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry. arXiv:2602.19411 [physics.chem-ph]. Feb. 2026. doi: 10.48550/arXiv.2602.19411. url: http://arxiv.org/abs/2602.19411 (visited on 08/29/2026).

[74] Luhuan Wu et al. Practical and Asymptotically Exact Conditional Sampling in Difusion Models. Nov. 2024. doi: 10 . 48550 / arXiv . 2306 . 17775. arXiv: 2306 . 17775 [stat.ML]. (Visited on 08/26/2026).

[75] Marta Skreta et al. Feynman-Kac Correctors in Difusion: Annealing, Guidance, and Product of Experts. June 2025. doi: 10.48550/arXiv.2503.02819. arXiv: 2503.02819 [cs]. (Visited on 10/13/2025).

[76] Peter Holderrieth et al. Diamond Maps: Eficient Reward Alignment via Stochastic Flow Maps. May 2026. doi: 10 . 48550 / arXiv . 2602 . 05993. arXiv: 2602 . 05993 [cs.LG]. (Visited on 08/26/2026).

[77] Sam McCallum et al. Strong Stochastic Flow Maps. May 2026. doi: 10.48550/arXiv.2606.01086. arXiv: 2606.01086 [cs.LG]. (Visited on 08/26/2026).

[78] Brian D. O. Anderson. “Reverse-Time Difusion Equation Models”. In: Stochastic Processes and their Applications 12.3 (May 1982), pp. 313–326. issn: 0304-4149. doi: 10.1016/0304-4149(82) 90051-5. (Visited on 08/26/2026).

[79] Aapo Hyv¨arinen. “Estimation of Non-Normalized Statistical Models by Score Matching”. In: Journal of Machine Learning Research 6.24 (2005), pp. 695–709. issn: 1533-7928. (Visited on 08/26/2026).

[80] G. E. Uhlenbeck and L. S. Ornstein. “On the Theory of the Brownian Motion”. In: Physical Review 36.5 (Sept. 1930), pp. 823–841. doi: 10.1103/PhysRev.36.823. url: https://link. aps.org/doi/10.1103/PhysRev.36.823 (visited on 08/29/2026).

[81] Oscar Davis et al. Generalised Flow Maps for Few-Step Generative Modelling on Riemannian Manifolds. Feb. 2026. doi: 10.48550/arXiv.2510.21608. arXiv: 2510.21608 [cs.LG]. (Visited on 08/28/2026).

[82] Richard Liaw et al. “Tune: A Research Platform for Distributed Model Selection and Training”. In: arXiv preprint arXiv:1807.05118 (2018).

[83] Daniel W. Davies et al. “SMACT: Semiconducting Materials by Analogy and Chemical Theory”. In: Journal of Open Source Software 4.38 (June 2019), p. 1361. issn: 2475-9066. doi: 10.21105/ joss.01361. (Visited on 09/01/2026).

[84] Mark Neumann et al. Orb: A Fast, Scalable Neural Network Potential. Oct. 2024. doi: 10.48550/ arXiv.2410.22570. arXiv: 2410.22570 [cond-mat.mtrl-sci]. (Visited on 08/29/2026).

[85] Christopher J. Bartel et al. “A Critical Examination of Compound Stability Predictions from Machine-Learned Formation Energies”. In: npj Computational Materials 6.1 (July 2020), p. 97. issn: 2057-3960. doi: 10.1038/s41524-020-00362-y. (Visited on 08/29/2026).

## A Unit Cell Representation

As introduced in Section 3, we consider the materials proposal problem as inference on crystalline unit cells ${ \bf c } = ( { \bf A } , { \bf F } , { \bf y } )$ . The atomic descriptor matrix A is obtained by passing the raw atom types $\mathbf { a } \in \mathbb { Z } _ { + } ^ { N }$ through a chosen featurization map (Appendix C). F denotes fractional coordinates derived from 3-D Cartesian atomic coordinates $\mathbf { X } \in \mathbb { R } ^ { \dot { N } \times 3 }$ using the unit cell L

$$
\mathbf { F } = \mathbf { X } \mathbf { L } ^ { - 1 } .\tag{18}
$$

Given any valid cell matrix $\mathbf { L } ^ { \prime } \in \mathrm { G L } ^ { + } ( 3 , \mathbb { R } )$ , we first construct its metric tensor

$$
\mathbf { G } = \mathbf { L } ^ { \prime } \mathbf { L } ^ { \prime \top } .
$$

We then compute the lower-triangular Cholesky factor L such that

$$
\mathbf { G } = \mathbf { L L } ^ { \top } .
$$

The corresponding six-dimensional latent representation y is then defined by

$$
\begin{array} { r } { { \bf L } ( { \bf y } ) = \left[ \begin{array} { c c c } { e ^ { y _ { 1 } } } & { 0 } & { 0 } \\ { y _ { 2 } } & { e ^ { y _ { 3 } } } & { 0 } \\ { y _ { 4 } } & { y _ { 5 } } & { e ^ { y _ { 6 } } } \end{array} \right] , } \end{array}
$$

or equivalently,

$$
\begin{array} { r } { \mathbf { y } ( \mathbf { L } ) = [ \log L _ { 1 1 } , L _ { 2 1 } , \log L _ { 2 2 } , L _ { 3 1 } , L _ { 3 2 } , \log L _ { 3 3 } ] \in \mathbb { R } ^ { 6 } . } \end{array}
$$

Manifold Operations for Unit Cell Transport We treat the generation of all components of the crystal unit cell as taking place in continuous space. For y and $\mathbf { A }$ we operate in Euclidean space with simple Riemannian exp and log maps which, for elements x and y, are given by

$$
\exp _ { x } ( y ) = x + y \quad { \mathrm { a n d } } \quad \log _ { x } ( y ) = y - x .\tag{19}
$$

For F we consider generation on the periodic flat torus, $\mathbb { T } _ { [ 0 , 1 ) } ^ { N \times 3 } : = ( \mathbb { R } / \mathbb { Z } ) ^ { N \times 3 }$ with

$$
\begin{array} { r } { \exp _ { x } ( y ) = \mathbf { w r a p } ( x + y ) \quad \mathrm { a n d } \quad \log _ { x } ( y ) = \mathbf { w r a p } ( y - x + 0 . 5 ) - 0 . 5 } \end{array}\tag{20}
$$

and wrap operation as $\mathtt { w r a p } ( z ) = z$ mod 1.0. These operations allow us to write a Riemannian stochastic interpolant for each component of the crystalline unit cell c.

## B Generative Modeling as Dynamical Transport

## B.1 Difusions

Difusion models consider bridging a Gaussian base distribution $\rho _ { T }$ with a target distribution $\rho _ { 0 }$ by a stochastic process whose time evolution is given by a forward SDE

$$
\mathrm { d } x _ { t } = f _ { t } ( x _ { t } ) \mathrm { d } t + \sigma _ { t } \mathrm { d } \overrightarrow { W } _ { t } .\tag{21}
$$

Here $f _ { t }$ is a linear drif $f _ { t } ( x ) = A _ { t } x + b _ { t } , \sigma _ { t }$ is a chosen difusion coeficient, and $\mathrm { d } \overrightarrow { W } _ { t }$ is an infinitesimal Wiener increment where the arrow indicates that this process moves forward in time. This SDE gradually applies noise to data, approaching an isotropic Gaussian as $T$ grows to infinity. The objective in score-based difusion models is to learn a reverse-time SDE which generates clean data from noise which, subject to mild conditions on $f _ { t }$ and $\sigma _ { t } ,$ takes the form [26, 27, 78].

$$
\mathrm { d } \boldsymbol { x } _ { t } = \left[ f _ { t } ( \boldsymbol { x } _ { t } ) - \sigma _ { t } ^ { 2 } \nabla _ { \boldsymbol { x } _ { t } } \log \rho _ { t } ( \boldsymbol { x } _ { t } ) \right] \mathrm { d } t + \sigma _ { t } \mathrm { d } \overleftarrow { W } _ { t } .\tag{22}
$$

This is generally done by minimizing a score matching objective [79]. If we parameterize a neural network $s _ { t } ^ { \theta }$ approximating the score $s _ { t } ^ { \theta } \approx \nabla _ { x }$ log $\rho _ { t }$ then we can write the objective

$$
\mathcal { L } _ { \mathrm { S M } } ( \theta ) = \int _ { 0 } ^ { T } \mathbb { E } \left[ \left\| s _ { t } ^ { \theta } ( x _ { t } ) - \nabla _ { x _ { t } } \log \rho _ { t } ( x _ { t } | x _ { 0 } ) \right\| ^ { 2 } \right] \mathrm { d } t\tag{23}
$$

Since the forward SDE has an afine-linear drift and a state-independent difusion coeficient, its transi tion kernel $\rho _ { t } ( x _ { t } | x _ { 0 } )$ is Gaussian and given by the usual solution to the SDE in Equation (21) [80]. With access to an estimator for the score, Equation (22) can be integrated backwards in time to generate data from independent realizations of $\rho _ { T }$

## B.2 Flows

Following the success of difusion models, methods have been formulated to learn a deterministic transport between noise and clean data. In this flow-based generative modeling framework a coupling between a base distribution $\rho _ { 0 }$ and a target $\rho _ { 1 }$ is given by a velocity field [25, 28, 29].<sup>1</sup> The typical approach is to bridge noise and data with an interpolant as in Section 3,

$$
\begin{array} { r } { x _ { t } = \alpha _ { t } x _ { 0 } + \beta _ { t } x _ { 1 } , } \end{array}\tag{24}
$$

where $x _ { 1 }$ is sampled from the target and $x _ { 0 }$ is sampled from noise. Both $\alpha _ { t }$ and $\beta _ { t }$ are scheduling functions which determine how data is noised in time.

Interpolation between noise and data allows for the construction of the flow matching algorithm which aims to learn an instantaneous velocity field $v _ { t }$ . The desired velocity is that which solves a transport equation

$$
\frac { \partial \rho _ { t } } { \partial t } + \nabla \cdot \left( v _ { t } \rho _ { t } \right) = 0 .\tag{25}
$$

This transport equation describes the evolution of a time-dependent density from $t = 0$ to $t = 1$ . It can be shown that this velocity can be written in terms of the interpolant in Equation (24)

$$
v _ { t } ( x ) = \mathbb { E } [ \dot { x } _ { t } \vert x _ { t } = x ]\tag{26}
$$

We approximate this velocity with a neural network $v _ { t } ^ { \theta }$ by minimizing the conditional flow matching loss

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \int _ { 0 } ^ { 1 } \mathbb { E } \left[ \left\| v _ { t } ^ { \theta } ( x _ { t } ) - \dot { x } _ { t } \right\| ^ { 2 } \right] \mathrm { d } t\tag{27}
$$

where the expectation is taken over $( x _ { 0 } , x _ { 1 } )$ pairs. Access to an approximate $v _ { t }$ allows for the generation of new data samples through integration of an ODE forwards in time

$$
\frac { \mathrm { d } x } { \mathrm { d } t } = v _ { t } , \quad \mathrm { s . t . } \quad x _ { 0 } \sim \rho _ { 0 } .\tag{28}
$$

Flows and difusions are closely related paradigms for generative modeling [25]. If $\rho _ { 0 }$ is a standard Gaussian, it is possible to simulate data via a difusion SDE similar to Equation (22) given access to a flow velocity field $v _ { t }$ using the following relation

$$
\nabla _ { x } \log \rho _ { t } ( x ) = \alpha _ { t } ^ { - 1 } \frac { \beta _ { t } v _ { t } ( x ) - \dot { \beta } _ { t } x } { \dot { \beta } _ { t } \alpha _ { t } - \beta _ { t } \dot { \alpha } _ { t } } .\tag{29}
$$

This relationship is used for deriving Flow-GRPO [61]. The relationship between the score and the velocity is similarly used in the derivation of RAM (Appendix E).

## B.3 Flow Maps

The success of flows and difusions has inspired exploration into flow maps and consistency models [31, 32, 52]. Given a base and target distribution bridged by Equation (25) a flow map is a two-time map $X _ { s , u }$ which satisfies

$$
X _ { s , u } ( x _ { s } ) = x _ { u }\tag{30}
$$

where $s , u \in [ 0 , 1 ]$

A flow map must satisfy a tangent and consistency condition. The tangent condition enforces that, in the instantaneous limit, the flow map velocity matches the velocity in Equation (28)

$$
\dot { X } _ { s , s } ( x _ { s } ) : = \left. \frac { \partial } { \partial u } X _ { s , u } ( x _ { s } ) \right| _ { u = s } = v _ { s } ( x _ { s } ) .\tag{31}
$$

A flow map must also be consistent through time. There are three conditions that can be used to enforce consistency:

• Lagrangian

$$
{ \frac { \partial X _ { s , u } ( x _ { s } ) } { \partial u } } = v _ { u } ( X _ { s , u } ( x _ { s } ) )\tag{32}
$$

• Eulerian

$$
\frac { \partial X _ { s , u } ( x _ { s } ) } { \partial s } + \nabla X _ { s , u } ( x _ { s } ) v _ { s } ( x _ { s } ) = 0\tag{33}
$$

• Semigroup

$$
X _ { v , w } ( X _ { s , v } ( x _ { s } ) ) = X _ { s , w } ( x _ { s } )\tag{34}
$$

where $s < v < w$ . Flow maps have been successfully generalized for generative modeling on Riemannian manifolds [30, 81]. As outlined in Section 3 we employ Riemannian MeanFlow with a semigroup consistency objective for training OMatG-flash. Following this prescription [30], we parameterize OMatG-flash as a two-time map

$$
X _ { s , u } ( x _ { s } ) = \exp _ { x _ { s } } { ( ( u - s ) v _ { s , u } ( x _ { s } ) ) }\tag{35}
$$

where $\exp _ { x } y$ is the exponential map associated with a manifold $\mathcal { M }$ and $s < u$ . We find strong perfor mance by parameterizing the average velocity field $v _ { s , u }$ with endpoint prediction. Specifically, given $x _ { s }$ and an estimator of the endpoint $x _ { 1 } ( x _ { s } , s , u )$ we write the two-time velocity

$$
v _ { s , u } ( x _ { s } ) = { \frac { \log _ { x _ { s } } x _ { 1 } ( x _ { s } , s , u ) } { 1 - s } }\tag{36}
$$

where $\log _ { x } y$ is the logarithm map on $\mathcal { M } .$ Composing these two equations above we arrive at the parameterization of OMatG-flash in Equation (9). For both tangency and consistency losses we employ weighting functions to stabilize the loss $\begin{array} { r } { \gamma _ { t } = \frac { 1 - t } { \operatorname* { m a x } ( 1 - t , \epsilon ) } } \end{array}$ . We set $\epsilon = 0 . 1$

## C OMatG-flash Architecture

Atom Type Encoding OMatG-flash applies Crystalite’s subatomic tokenization scheme [42]. For atomic number $Z _ { i } .$ , Crystalite constructs the 34-dimensional descriptor

$$
\begin{array} { r } { \mathbf { d } ( Z _ { i } ) = [ \mathbf { e } _ { r ( Z _ { i } ) } ^ { ( 7 ) }  \mathbf { e } _ { g ( Z _ { i } ) } ^ { ( 1 9 ) }  \mathbf { e } _ { b ( Z _ { i } ) } ^ { ( 4 ) }  ( \frac { n _ { s } } { 2 } , \frac { n _ { p } } { 6 } , \frac { n _ { d } } { 1 0 } , \frac { n _ { f } } { 1 4 } ) ] , } \end{array}\tag{37}
$$

where $\mathbf { e } ^ { ( k ) }$ denotes a k-dimensional one-hot encoding, $r , g ,$ and b denote the element’s period, group, and block, and $( n _ { s } , n _ { p } , n _ { d } , n _ { f } )$ are its ground-state valence-shell occupancies. After standardization,

group weighting, PCA projection, and normalization, each descriptor is mapped to $\mathbf { a } ( Z _ { i } ) \in \mathbb { R } ^ { d _ { A } }$ with $d _ { A } = 2 4$ . The atomic descriptor matrix is formed by stacking these vectors:

$$
\begin{array} { r } { { \bf A } = \left[ \begin{array} { c } { { \bf a } ( Z _ { 1 } ) ^ { \top } } \\ { \vdots } \\ { { \bf a } ( Z _ { N } ) ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { N \times d _ { A } } . } \end{array}\tag{38}
$$

This representation efectively describes the space of elements and circumvents the need for a discrete difusion because the state is part of a continuous manifold.

Atom Type Decoding After integration is complete, each generated atomic descriptor is snapped to the closest valid element descriptor in PCA space. Specifically, for a generated descriptor $\hat { \mathbf { A } } _ { i , 1 }$ , we assign the atomic number

$$
\hat { Z } _ { i } = \underset { Z \in \{ 1 , \ldots , 1 0 0 \} } { \arg \operatorname* { m a x } } \frac { \left. \hat { \mathbf { A } } _ { i , 1 } , \mathbf { a } ( Z ) \right. } { \left\| \hat { \mathbf { A } } _ { i , 1 } \right\| _ { 2 } \left\| \mathbf { a } ( Z ) \right\| _ { 2 } } .\tag{39}
$$

Decoding is just a nearest-neighbor projection under cosine similarity to the set of valid PCA-projected element descriptors.

Neural Network Architecture We parameterize OMatG-flash as an all-atom transformer, again building of of the Crystalite architecture to do so [42].

The atoms in a unit cell c are tokenized into N tokens, one per atom, with one additional token for the lattice. The tokenization of an atom indexed i is a linear combination of a coordinate and atom type embedding

$$
\mathbf { h } _ { i } ^ { \mathrm { a t o m } } = E _ { \mathrm { c o m p } } ( \mathbf { A } _ { i } ) + E _ { \mathrm { c o o r d } } ( \mathrm { S i n E m b } ( \mathbf { F } _ { i } ; N _ { \mathrm { f r e q } } ^ { \mathrm { c o o r d } } ) )\tag{40}
$$

where SinEmb featurizes the coordinates via a Fourier expansion,

$$
\mathrm { S i n E m b } ( x ; N _ { \mathrm { f r e q } } ) = [ \sin ( 2 \pi k x ) , \cos ( 2 \pi k x ) ] _ { k = 1 } ^ { N _ { \mathrm { f r e q } } } ,\tag{41}
$$

and $E _ { \mathrm { c o m p } } , E _ { \mathrm { c o o r d } }$ are two-layer MLPs with a SiLU activation. The lattice token is similarly obtained by passing y through a two-layer MLP embedding layer

$$
{ \bf h } _ { N + 1 } ^ { \mathrm { l a t t i c e } } ( { \bf y } ) = E _ { \mathrm { l a t t i c e } } ( { \bf y } )\tag{42}
$$

yielding a sequence of tokens $\mathbf { t } ^ { ( 0 ) } = [ \mathbf { h } _ { 1 } ^ { \mathrm { a t o m } } , \dots , \mathbf { h } _ { N } ^ { \mathrm { a t o m } } , \mathbf { h } _ { N + 1 } ^ { \mathrm { l a t t i c e } } ]$ which the all-atom transformer processes through layers of self-attention. The superscript indicates that this is the initial tokenized embedding before any transformer layers are applied. OMatG-flash is adapted from the difusion transformer architecture [54]. Our conditioning vector is derived using the left and right foot times. For a source time s and a destination time u this conditioning vector is predicted via a two-layer MLP on s and $\Delta = u - s$

$$
\mathbf { h } _ { \mathrm { c o n d i t i o n i n g } } ( s , u ) = E _ { \mathrm { c o n d } } \left( \mathrm { S i n E m b } ( s ; N _ { \mathrm { f r e q } } ^ { \mathrm { t i m e } } ) , \mathrm { S i n E m b } ( \Delta ; N _ { \mathrm { f r e q } } ^ { \mathrm { t i m e } } ) \right)\tag{43}
$$

$\mathbf { h } _ { \mathrm { c o n d i t i o n i n g } }$ along with $\mathbf { t } ^ { ( 0 ) }$ are passed into the model. $\mathbf { h } _ { \mathrm { c o n d i t i o n i n g } }$ is used to predict scale and shift parameters which normalize the tokens via adaptive layer normalization (adaLN) over the course of several transformer layers.

We enable physics-informed self-attention within the model using Crystalite’s geometry enhancement module. This module computes a block-diagonal bias mask, B(c), that modulates attention only between atom tokens i and $j$ by first computing the approximate minimum-image displacement using $\Delta \mathbf { f } _ { i j } ( \mathbf { r } ) = \mathbf { f } _ { j } - \mathbf { f } _ { i } + \mathbf { r } ,$

$$
\mathbf { r } _ { i j } ^ { \star } = \arg \operatorname* { m i n } _ { \mathbf { r } \in \{ - 1 , 0 , 1 \} ^ { 3 } } \Delta \mathbf { f } _ { i j } ( \mathbf { r } ) \mathbf { G } ( \mathbf { y } ) \Delta \mathbf { f } _ { i j } ^ { \top } ( \mathbf { r } ) ,\tag{44}
$$

then

$$
\Delta \mathbf { f } _ { i j } ^ { \mathrm { m i } } = \mathbf { f } _ { j } - \mathbf { f } _ { i } + \mathbf { r } _ { i j } ^ { \star } .\tag{45}
$$

The normalized minimum-image Cartesian distance becomes

$$
d _ { i j } ^ { \mathrm { m i } } = \frac { \| \Delta \mathbf { f } _ { i j } ^ { \mathrm { m i } } \mathbf { L } ( \mathbf { y } ) \| _ { 2 } } { s ( \mathbf { y } ) }\tag{46}
$$

which is normalized by a cell scale divisor $s ( \mathbf { y } ) = ( a + b + c ) / 3$ derived from the lattice vector lengths. The pairwise geometry is featurized as

$$
\mathbf { e } _ { i j } = [ \mathrm { S i n E m b } ( \Delta \mathbf { f } _ { i j } ^ { \mathrm { m i } } ; N _ { \mathrm { f r e q } } ^ { \mathrm { e d g e } } ) \| \{ \exp [ - \gamma ( d _ { i j } ^ { \mathrm { m i } } - \mu _ { q } ) ^ { 2 } ] \} _ { q = 1 } ^ { N _ { \mathrm { r b f } } } \| \mathbf { y } ^ { \mathrm { Y 1 } } ( \mathbf { y } )  .\tag{47}
$$

Here, $N _ { \mathrm { f r e q } } ^ { \mathrm { e d g e } }$ and $N _ { \mathrm { r b f } }$ are the numbers of Fourier and radial basis features, respectively; $\mu _ { q }$ are uniformly spaced radial centers, $\gamma$ is their inverse squared spacing, and $\mathbf { y } ^ { \mathrm { Y 1 } } ( \mathbf { y } )$ denotes the Y1 lattice features computed from $\mathbf { y } .$

The bias for attention head h is

$$
[ \mathbf { B } ( \mathbf { c } ) ] _ { h i j } = g _ { h } ( s ) \left( - \mathrm { s o f t p l u s } ( w _ { h } ) d _ { i j } ^ { \mathrm { m i } } + [ E _ { \mathrm { e d g e } } ( \mathbf { e } _ { i j } ) ] _ { h } \right) ,\tag{48}
$$

where $w _ { h }$ is a learned head-specific distance parameter, $E _ { \mathrm { e d g e } }$ is a two-layer MLP with hidden dimension $d _ { \mathrm { e d g e } }$ , and

$$
g _ { h } ( s ) = \mathrm { s i g m o i d } \left( \mathrm { s o f t p l u s } ( \alpha _ { h } ) [ - \log ( \operatorname* { m a x } ( s , \epsilon ) ) ] + \beta _ { h } \right) .\tag{49}
$$

Here, s is the left-foot time supplied to the DiT/SiT backbone, $\alpha _ { h }$ and $\beta _ { h }$ are learned head-specific parameters, and $\epsilon > 0$ is a numerical-stability constant. The diagonal of the edge bias and all entries involving padded atoms or the lattice token are set to zero.

The geometry bias is added directly to the scaled dot-product attention logits. For attention head h,

$$
{ \mathrm { A t t n } } _ { h } = \mathrm { s o f t m a x } \left( { \frac { \mathbf { Q } _ { h } \mathbf { K } _ { h } ^ { \top } } { \sqrt { d _ { h } } } } + \mathbf { B } _ { h } ( \mathbf { c } ) + \mathbf { M } \right) \mathbf { V } _ { h } ,\tag{50}
$$

where $\mathbf { Q } _ { h } , \mathbf { K } _ { h } ,$ , and ${ \bf V } _ { h }$ are the query, key, and value projections, $d _ { h }$ is the head dimension, and M is the key-padding mask, taking value −∞ for padded key tokens and zero otherwise. The remainder of the architecture follows standard SiT/DiT token-based setups, using adaLN-conditioned self-attention and MLP residual blocks.

For predicting endpoints which are periodic in the domain [0, 1) we parameterize the endpoint prediction of denoised fractional coordinates,

$$
\hat { \bf F } _ { 1 } = \frac { \mathrm { a t a n 2 } ( \hat { \bf u } _ { 1 } , \hat { \bf v } _ { 1 } ) } { 2 \pi } \mathrm { m o d } 1 . 0 ,\tag{51}
$$

Here, $\hat { \bf u } _ { 1 }$ and $\hat { \mathbf { v } } _ { 1 }$ are predicted by the fractional-coordinate head. We use this so-called circular prediction mode for all realizations of OMatG-flash. $\hat { \bf A } _ { 1 }$ and $\hat { \mathbf { y } } _ { 1 }$ are predicted by linear layer output heads.

Data Augmentation OMatG-flash is not rotation-equivariant nor is it translation invariant however, since F and A are naturally rotation invariant, we need only the lattice in handling rotations. Our choice of modeling $\mathbf { y }$ is rotation-invariant and, thus, we do not consider data augmentation for global rotations of the crystal. OMatG-flash is, however, not translation-invariant and F is sensitive to this. We therefore augment the data with random translations in 3D during training.

## D Hyperparameter Selection

Table 4: OMatG-flash hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Architecture</td><td></td></tr><tr><td>Atomic descriptor dimension  $d _ { A }$ </td><td>24</td></tr><tr><td>Supported atomic numbers Z</td><td>1-100</td></tr><tr><td>Transformer width d</td><td>512</td></tr><tr><td>Transformer layers</td><td>18</td></tr><tr><td>Attention heads</td><td>16</td></tr><tr><td>Coordinate Fourier frequencies</td><td>32</td></tr><tr><td>Time-embedding dimension</td><td>256</td></tr><tr><td>Flow map objective</td><td></td></tr><tr><td>Atomic weights  $\left( \lambda _ { \mathrm { T } , A } , \lambda _ { \mathrm { C } , A } \right)$ </td><td>(2.5, 2.5)</td></tr><tr><td>Coordinate weights  $\big ( \lambda _ { \mathrm { T } , F } , \lambda _ { \mathrm { C } , F } \big )$ </td><td>(1.0, 1.0)</td></tr><tr><td>Lattice weights  $\left( \lambda _ { \mathrm { T } , y } , \lambda _ { \mathrm { C } , y } \right)$ </td><td>(0.05, 0.05)</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>AdamW coefficients  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td>(0.9,0.999)</td></tr><tr><td>AdamW  $\epsilon _ { \mathrm { o p t } }$ </td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>EMA decay</td><td>0.999</td></tr><tr><td>Gradient clipping threshold</td><td>1.0</td></tr><tr><td>RAM post-training</td><td></td></tr><tr><td>Atomic weights  $\left( \lambda _ { \mathrm { R A M } , A } , \lambda _ { \mathrm { C } , A } \right)$ </td><td>(-,-)</td></tr><tr><td>Coordinate weights  $\left( \lambda _ { \mathrm { R A M } , F } , \lambda _ { \mathrm { C } , F } \right)$ </td><td>(0.520, 0.002)</td></tr><tr><td>Lattice weights  $\left( \lambda _ { \mathrm { R A M } , y } , \lambda _ { \mathrm { C } , y } \right)$ </td><td>(0.074, 0.002)</td></tr><tr><td>RAM energy weight  $\lambda _ { E }$ </td><td>0.2</td></tr><tr><td>RAM number groups</td><td>16</td></tr><tr><td>RAM group size</td><td>64</td></tr><tr><td>RAM noising replicas per sample</td><td>4</td></tr><tr><td>RAM learning rate</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>RAM training steps</td><td>250</td></tr></table>

OMatG-flash’s hyperparameters are summarized in Table 4. We minimize flow map and RAM losses perfield using the weights specified. The channel-specific tangency and consistency weights are normalized internally, so only their relative values matter. Hyperparameters for RAM were fine-tuned using Ray Tune [82]; initialization for pretraining was derived from intuition. We use a standard normal base distribution which, after wrapping the fractional coordinate onto the flat torus, is efectively uniform. At inference time we use a uniform time grid.

## E Reinforce Adjoint Matching

Reinforce adjoint matching frames fine-tuning an ODE velocity field in terms of stochastic optimal control. The central objective is to draw samples from a tilted distribution

$$
\rho ^ { \star } ( x ) \propto \rho ( x ) e ^ { r ( x ) }\tag{52}
$$

where r is a scalar-valued reward function [57]. In the stochastic dynamical transport setting, samples from $\rho ^ { \star }$ can be drawn by sampling random Gaussian noise $x _ { 0 }$ and integrating an SDE similar to Equation (22) but written forwards in time

$$
d x _ { t } = \left[ f _ { t } ( x _ { t } ) + \sigma _ { t } ^ { 2 } \nabla _ { x _ { t } } \log \rho _ { t } ( x _ { t } ) + \sigma _ { t } ^ { 2 } u _ { t } ^ { \star } ( x _ { t } ) \right] \mathrm { d } t + \sigma _ { t } \mathrm { d } \overrightarrow { W } _ { t } .\tag{53}
$$

where $\boldsymbol { u } _ { t } ^ { \star }$ is a control field which, to sample from Equation (52), should be chosen to maximize a terminal reward regularized by a path cost term

$$
u ^ { \star } = \arg \operatorname* { m a x } _ { u } \mathbb { E } \left[ r ( x _ { 1 } ) - \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \sigma _ { t } ^ { 2 } \| u _ { \tau } ( x _ { \tau } ) \| ^ { 2 } \mathrm { d } \tau \right] .\tag{54}
$$

We write the optimal controller as $\boldsymbol { u } _ { t } ^ { \star } ( \boldsymbol { x } ) = \nabla _ { \boldsymbol { x } } V _ { t } ( \boldsymbol { x } )$ where $V _ { t }$ is the value function [56]

$$
\nabla _ { x } V _ { t } ( x ) = \mathbb { E } \left[ r ( x _ { 1 } ) \nabla _ { x _ { t } } \log \rho _ { 1 | t } ( x _ { 1 } ^ { u ^ { \star } } | x _ { t } ) \big | x _ { t } = x \right] - \frac { 1 } { 2 } \mathbb { E } \left[ \nabla \int _ { t } ^ { 1 } \sigma _ { t } ^ { 2 } \| u _ { \tau } ^ { \star } ( x _ { \tau } ) \| ^ { 2 } \mathrm { d } \tau \middle | x _ { t } = x \right] .\tag{55}
$$

Dropping the path cost term, we arrive at an approximate value function gradient used in RAM:

$$
\begin{array} { r } { \nabla _ { x } V _ { t } ( x ) \approx \mathbb { E } \left[ r ( x _ { 1 } ) \nabla _ { x _ { t } } \log \rho _ { 1 | t } ( x _ { 1 } ^ { u ^ { \star } } | x _ { t } ) \big | x _ { t } = x \right] . } \end{array}\tag{56}
$$

The superscript $u ^ { \star }$ indicates that the sample was generated using the controller.

RAM Fixed Point Objective In order to estimate the control in Equation (56) we recall the uncontrolled SDE drift

$$
b _ { t } ( x ) = v _ { t } ( x ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla _ { x } \log \rho _ { t } ( x ) ,\tag{57}
$$

and controlled SDE drifts

$$
b _ { t } ( \boldsymbol { x } ) + \sigma _ { t } ^ { 2 } u _ { t } ^ { \star } ( \boldsymbol { x } ) = \boldsymbol { v } _ { t } ^ { \star } ( \boldsymbol { x } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla _ { \boldsymbol { x } } \log \rho _ { t } ^ { \star } ( \boldsymbol { x } ) ,\tag{58}
$$

Subtracting gives

$$
\sigma _ { t } ^ { 2 } u _ { t } ^ { \star } ( x ) = \left( v _ { t } ^ { \star } ( x ) - v _ { t } ( x ) \right) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \left( \nabla _ { x } \log \rho _ { t } ^ { \star } ( x ) - \nabla _ { x } \log \rho _ { t } ( x ) \right) .\tag{59}
$$

Provided that $\rho _ { 0 }$ is Gaussian and that we use a memoryless schedule, we can write $\nabla _ { x } \log \rho _ { t } ( x ) =$ $2 \sigma _ { t } ^ { - 2 } ( v _ { t } ( x ) - \ddot { \beta _ { t } } \beta _ { t } ^ { - 1 } x )$ . Using this score identity, one can show that $v _ { t } ^ { \star }$ satisfies a fixed point

$$
\sigma _ { t } ^ { 2 } u _ { t } ^ { \star } ( x ) = 2 ( v _ { t } ^ { \star } ( x ) - v _ { t } ( x ) )\tag{60}
$$

$$
v _ { t } ^ { \star } ( x ) \approx v _ { t } ( x ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \mathbb { E } \left[ r ( x _ { 1 } ) \nabla _ { x _ { t } } \log \rho _ { 1 | t } ( x _ { 1 } ^ { u ^ { \star } } | x _ { t } ) \big | x _ { t } = x \right]\tag{61}
$$

The last step is to invoke the Bayes bridge score identity shown in the RAM derivation for the linear interpolant schedule $( \alpha _ { t } = 1 - t , \beta _ { t } = t ) \ [ 5 5 ]$

$$
\nabla _ { x _ { t } } \log \rho _ { 1 | t } ( x _ { 1 } | x _ { t } ) = \frac { t } { 1 - t } ( ( x _ { 1 } - x _ { 0 } ) - v _ { t } ^ { \star } ( x _ { t } ) ) ,\tag{62}
$$

and to enforce $\sigma _ { t } ^ { 2 } / 2 = ( 1 - t ) / t$ giving the RAM fixed point condition<sup>2</sup>.

$$
v _ { t } ^ { \star } ( x ) - v _ { t } ( x ) = \mathbb { E } \left[ r ( x _ { 1 } ) ( \dot { x } _ { t } - v _ { t } ^ { \star } ( x _ { t } ) ) | x _ { t } = x \right] .\tag{63}
$$

![](images/4d710097059a49c6b68f4006d651ccdf02a7d9778db551dbc0d4b7f48305fb7f.jpg)

![](images/29b7bbc53dbb0740b65e0a33c980605d017c635a32db18f4e1113b24663b4ff7.jpg)  
Figure 4: Instability in RAM Minimization Early during post-training optimization with no relaxation applied the energy of generated samples decreases by over 0.2 eV/atom on average as shown in the inset. The one-to-one match rate trends slightly upward. Later, the RAM and consistency losses become unstable and model performance collapses.

Application to OMatG-flash We highlight that the application of RAM to our OMatG-flash flow map improves performance notably (Section 4). We do note, however, that in the limit of long training times, we encounter instability and reward collapse and we illustrate this efect in Figure 4. Applying relaxation steps to the generated structure before reward computation improved stability of the algorithm. The numbers for OMatG-flash-RAM reported in Tables 1 and 2 are obtained using the checkpoint which exhibited the lowest average relative energy during training with relaxation applied before reward computation.

We do note that on $\mathbb { T } _ { [ 0 , 1 ) } ^ { N \times 3 }$ the application of the velocity-score reparameterization given in Equation (29) is heuristic and we do not claim an extension of RAM to arbitrary manifolds is theoretically rigorous nor eficacious. Rewards are normalized to group-relative advantages following [55].

## F Benchmark Metrics

## F.1 Crystal Structure Prediction

Match Rate and RMSE Match rate and RMSE are two canonical methods for measuring the performance of generative models on the CSP task. Both rely on the pymatgen StructureMatcher algorithm which determines a match within some specified tolerance by performing translations, rotations, and unimodular lattice operations. If a match is found, an RMSE can be computed which is then normalized by $\sqrt [ 3 ] { V / N _ { \mathrm { a t o m s } } }$ with V the volume of the cell and $N _ { \mathrm { a t o m s } }$ the number of atoms. For CSP metrics we use the settings most often used in the generative modeling literature: stol=0.5, ltol=0.3, angle tol=10.0. This metric is generally computed one-to-one where each generated structure is com pared to its index-matched test set reference.

METRe and cRMSE It has been observed that one-to-one match rate is a suboptimal metric for determining CSP performance [64]. This is due to the phenomenon of polymorphism in crystal structures, where a single composition can yield many diverse crystal structures. METRe attempts to combat this by instead comparing each reference structure against every generated structure. cRMSE is then a corrected version of RMSE where non-matched structures appear as an explicit penalty in the cRMSE computation

![](images/358e4eb5b3e3865fcffcbc327ef3271c61e7b2259ad27cfe19753cb7ea328818.jpg)  
Figure 5: Concept of METRe vs. One-To-One Matching The one-to-one method (Left) counts three non-matches despite the presence of two pairs of structures that could indeed be matched. METRe (Right) correctly scans the generated set for each reference structure, determining that two matches are present in the generated set.

$$
\mathrm { c R M S E } \left( \left\{ \mathbf { c } _ { \mathrm { g e n } , j } \right\} _ { j = 1 } ^ { N _ { \mathrm { g e n } } } ; \left\{ \mathbf { c } _ { \mathrm { r e f } , i } \right\} _ { i = 1 } ^ { N _ { \mathrm { r e f } } } \right) = \frac { 1 } { N _ { \mathrm { r e f } } } \sum _ { i \in \mathcal { X } } \operatorname* { m i n } _ { \mathit { j } \in \mathcal { I } _ { i } } \mathrm { R M S E } ( \mathbf { c } _ { \mathrm { g e n } , j } , \mathbf { c } _ { \mathrm { r e f } , i } ) + \mathrm { s t o 1 } . \frac { \left( N _ { \mathrm { r e f } } - N _ { \mathrm { r e f } } ^ { \mathrm { m a t c h } } \right) } { N _ { \mathrm { r e f } } }\tag{64}
$$

where $\mathbf { c } _ { \mathrm { g e n } }$ is a generated crystal, $\mathbf { c } _ { \mathrm { r e f } }$ is a reference crystal, $N _ { \mathrm { r e f } }$ is the number of reference test set structures, and $N _ { \mathrm { r e f } } ^ { \mathrm { m a t c h } }$ is the number of matched structures. We consider X as

$$
\mathcal { X } = \left\{ i \in \left\{ 1 , . . . , N _ { \mathrm { r e f } } \right\} \ : \ \mathcal { I } _ { i } \neq \emptyset \right\}\tag{65}
$$

where we have $\mathcal { I } _ { i }$ as

$$
\mathcal { T } _ { i } = \{ j \in \{ 1 , . . . , N _ { \mathrm { g e n } } \} \ : \ \mathbf { c } _ { \mathrm { g e n } , j } \ \mathrm { m a t c h e s } \ \mathbf { c } _ { \mathrm { r e f } , i } \} .\tag{66}
$$

We illustrate METRe vs. one-to-one matching in Figure $5 .$

Validity For Table 1 we report results for all structures (valid or invalid) along with values restricted to only those passing certain validity checks (valid). Validity is assessed first via structural assessment followed by SMACT chemical validity. Structural checks ensure that no interparticle distances are less than 0.5<sup>˚</sup>A, the cell volume is greater than $0 . 1 \mathring \mathrm { A } ^ { 3 }$ , and that a polar sine cutof

$$
V / ( a \times b \times c ) \geq 1 0 ^ { - 3 }\tag{67}
$$

is not violated where V is the volume of the cell and $a , b , c$ are lattice vector lengths. If these are satisfied, a crystal structure must then pass a SMACT compositional validity check [83] followed by a chemical fingerprint validity check.

Table 5: DNG Raw Numbers. LeMat-GenBench reports public leaderboard results using one decimal place. For transparency we report the raw numbers computed by the LeMat-GenBench evaluator and extractor scripts. Numbers are computed for an initial pool of 2500 structures. We take numbers for competing models from the public LeMat-GenBench leaderboard.
<table><tr><td>Method</td><td>Valid ↑</td><td>Novel ↑</td><td>Stable ↑</td><td>S.U.N. ↑</td><td>S.U.N. + M.S.U.N. ↑</td></tr><tr><td>DiffCSP</td><td>2392</td><td>1654</td><td>58</td><td>3</td><td>215</td></tr><tr><td>MatterGen</td><td>2392</td><td>1762</td><td>49</td><td>6</td><td>380</td></tr><tr><td>OMatG</td><td>2361</td><td>1255</td><td>284</td><td>24</td><td>464</td></tr><tr><td>MCFlow</td><td>2429</td><td>1306</td><td>298</td><td>18</td><td>490</td></tr><tr><td>Crystalite</td><td>2430</td><td>1331</td><td>317</td><td>38</td><td>604</td></tr><tr><td>OMatG-flash4</td><td>2409</td><td>1672</td><td>180</td><td>34</td><td>455</td></tr><tr><td>OMatG-flash8</td><td>2420</td><td>1584</td><td>196</td><td>26</td><td>537</td></tr><tr><td> $\mathrm { O M a t G – f l a s h } _ { 1 6 }$ </td><td>2420</td><td>1524</td><td>221</td><td>29</td><td>564</td></tr></table>

## F.2 De Novo Generation

All DNG benchmark metrics are computed with code provided by LeMat-GenBench in their open source implementation using the scripts/run benchmarks.py script with comprehensive multi mlip hull configuration flag and structure-matcher fingerprinting [63]. The subsequent results are extracted with scripts/extract benchmark metrics.py. We prerelax our 2500 generated structures using MACE for up to 1000 steps with a force tolerance of 0.02 eV/<sup>˚</sup>A before benchmarking with LeMat-GenBench. To expand on our results in Table 3 we include raw numbers as reported by LeMat for all DNG metrics in Table 5

Stability Stability is computed using a series of foundation model interatomic potentials. In order to determine if a material is stable or not it is input to UMA [14], ORB [84], and MACE [13]. The average energy-above-hull returned by these models is compared with the LeMat convex hull of stable phases corresponding to that reduced composition [85]. A material is stable if and only if its energyabove-hull, $E _ { \mathrm { h u l l } } \leq 0 . 0 ~ \mathrm { e V / a t o m }$ . Metastability for the purposes of M.S.U.N. computation is similarly defined, triggering if $0 . 0 < E _ { \mathrm { h u l l } } \le 0 . 1$ eV/atom. Additionally, a material can only be considered if at least two of the three MLIPs in the ensemble return an energy for that entry.

Uniqueness and Novelty Uniqueness and novelty are computed in the same way but using diferent reference data. For both metrics pymatgen’s StructureMatcher algorithm is used. The tolerances used by LeMat-GenBench are stricter than we use for computing CSP and are set stol=0.3, ltol=0.1, angle tol=5.0. Uniqueness is determined by comparison to all structures in the generated set. Novelty, on the other hand, is determined against a held-out LeMat-Bulk reference set [65].

## F.3 Wall-Clock Inference Speed

We measure sampling speed using the conventions established in the literature. To measure results in accordance with the literature, we use one NVIDIA H100 GPU choosing a batch size of 8192 structures for the DNG task with the hyperparameters as reported in Appendix C. We use bfloat16 inference which is consistent with how OMatG-flash was trained. We define generation time as the time required to sample $\mathbf { c } _ { 0 } .$ , perform inference with OMatG-flash, and decode the data into a standard unit cell representation, excluding I/O. For reporting the Pareto frontier in Figure 1 we use baseline timing measurements as computed by Crystalite [42]. For reporting Crystalite itself we use their “standard” inference setting as they indicate that this is the primary timing one should consider. We emphasize that, even with their reported optimized inference time of 5.14 seconds per thousand structures, our model at NFE=16 is an order of magnitude faster, recording 0.535 seconds per thousand structures.