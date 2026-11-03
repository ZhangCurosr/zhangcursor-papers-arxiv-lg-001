# DeepWaveRL: Self-Supervised Full Waveform Inversion via Reinforcement Learning

Peng Jin<sup>1</sup> Yinan Feng<sup>1</sup> Yinpeng Chen<sup>2</sup> Youzuo Lin<sup>1</sup> Sharon Huang<sup>3</sup>

<sup>1</sup>University of North Carolina, Chapel Hill, U.S.

<sup>2</sup>Google DeepMind, U.S.

<sup>3</sup>Pennsylvania State University, U.S.

pjin@unc.edu, ynf@unc.edu, yinpengc@google.com yzlin@unc.edu, suh972@psu.edu

## Abstract

Full Waveform Inversion (FWI) is a fundamental inverse problem in geophysics to estimate subsurface properties, such as velocity, from seismic measurements. Recently, self-supervised learning methods remove the need for ground-truth velocity maps during training by leveraging a forward operator to simulate seismic data from predictions and backpropagate the gradients of reconstruction loss through it. In this work, we consider a more challenging setting without accessing such gradients. To this end, we incorporate reinforcement learning into self-supervised FWI and propose a framework named DeepWaveRL. Specifically, we reformulate FWI as a policy learning problem, where the policy network predicts velocity maps as actions, and a forward operator is used solely for simulation. A global scalar reward is computed based on data misfit for each prediction and used to weight policy gradients. Thus, learning proceeds without accessing the gradients from the forward operator. Given the indirect and coarse supervision signals, we further identify key strategies to stabilize learning and mitigate the spatial credit assignment bottleneck. Experimental results show that, despite relying entirely on a single scalar reward, DeepWaveRL achieves competitive performance compared to supervised and self-supervised baselines. These findings suggest that our framework offers a flexible and complementary learning paradigm for the FWI research community.

## 1 Introduction

Subsurface imaging is essential for characterizing geological structures and geophysical properties (e.g., velocity and impedance), with applications in energy exploration, carbon capture and sequestration, and earthquake early warning systems. A central technique in this domain is Full Waveform Inversion (FWI), which estimates subsurface velocity maps from seismic measurements. Mathemati cally, for an isotropic medium with constant density, the velocity map and seismic measurements are connected by the acoustic wave equation:

$$
\nabla ^ { 2 } p ( x , z , t ) - \frac { 1 } { v ( x , z ) ^ { 2 } } \frac { \partial ^ { 2 } p ( x , z , t ) } { \partial t ^ { 2 } } = s ( x , z , t ) \ ,\tag{1}
$$

where x denotes the horizontal offset, z the depth, $p ( x , z , t )$ the pressure wavefield at spatial location $( x , z )$ and time $t , v ( x , z )$ the wave propagation velocity at $( x , z ) , s ( x , z , t )$ the source term, and ∇<sup>2</sup> the Laplacian operator. In practice, seismic data are often collected at the surface $( \mathrm { i . e . , } p ( x , z = 0 , t ) )$ While FWI has the potential to produce high-resolution velocity maps, the inverse problem itself is inherently non-linear and ill-posed. Conventional physics-driven approaches requires intensive computation due to repeated forward simulations per sample and exhibit strong sensitivity to noise and initial conditions. These challenges have motivated growing interest in data-driven FWI methods.

![](images/9c51391433d9e9fb4e8588467b3d069fb71ecbe3815fe48959fd35a74e02829c.jpg)  
Figure 1: Comparison between data-driven FWI methods. Top: Supervised learning methods compute loss between predicted and ground-truth velocities; Middle: Exact-gradient self-supervised learning method computes the loss between input and reconstructed seismic data and backpropagates gradients through a forward operator f; Bottom: Our DeepWaveRL uses the forward operator solely to compute a global scalar reward based on data misfit, without accessing gradients from f.

A majority of data-driven methods [21, 25, 10] adopt the supervised learning paradigm and formulate FWI as an image-to-image translation task. As shown in Figure 1, deep neural networks directly learns the mapping from seismic data to velocity maps, enabling fast inference and achieving high accuracy. However, such methods depend on large collections of paired seismic data and velocity maps for training. In real-world scenarios, ground-truth velocity maps are rarely available, as constructing them is extremely time-consuming and requires substantial expertise from geophysicists.

Recent work (UPFWI) [9] explicitly leverages the underlying physics equations and eliminates the need for ground-truth velocity maps via self-supervised learning. As illustrated in Figure 1, UPFWI couples a network with a differentiable forward operator f to simulate seismic data from predictions. Training is driven by minimizing a reconstruction loss in seismic domain, with gradients backpropagated through f. While this approach enables end-to-end learning without ground-truth velocity maps, it relies on access to exact, dense gradients from the forward operator.

In this paper, we push the boundaries of self-supervised FWI by investigating a more challenging optimization regime: learning the high-dimensional spatial mapping using only a single global scalar reward without accessing the gradient information from the forward operator. To this end, we present DeepWaveRL, a self-supervised framework for FWI that incorporates reinforcement learning. As depicted in Figure 1, we formulate FWI as a single-step decision problem, where the input seismic data serve as the state, and a policy network outputs the velocity map as the action. A forward operator f is used to simulate seismic data from the prediction and to compute a global scalar reward based on the seismic data misfit. Policy optimization is performed using a group-based policy gradient algorithm, which updates the network parameters and action distributions by weighting log-probability gradients with the group-relative advantages.

Learning under this strict scalar-reward regime poses several challenges. First and foremost is the spatial credit assignment bottleneck. Because the reward evaluates the entire 2D velocity map globally, the agent lacks localized feedback to determine which specific pixels or structures caused the error. Second, the vast spatial action space triggers a combinatorial explosion, leading to extreme variance in the policy gradient estimator. Finally, seismic measurements exhibit large dynamic ranges, where dominant directive energy overwhelms weaker but informative deeper reflections. This leads to a highly non-convex physical reward landscape and makes stable exploration exceptionally difficult.

To stabilize policy gradients and mitigate these challenges, we introduce three targeted algorithmic strategies. First, we employ a map-level advantage estimator combined with a discretized action space, which effectively bounds the exploration manifold and dramatically reduces the variance of the gradient updates. Second, we introduce a sign-preserving logarithmic reward transformation to reshape the physical reward landscape, compressing dominant wave energy to extract learning signals for deep-region structures. Finally, we demonstrate that initializing the policy with learned structural priors by transfer learning acts as a crucial regularizer, safely guiding the agent past local minima.

To evaluate the efficacy of our framework, we benchmark DeepWaveRL on several datasets from OpenFWI [5], a large-scale, multi-structural dataset collection. Experimental results show that, despite relying entirely on a single scalar reward, our DeepWaveRL attains comparable performance to the supervised baseline InversionNet [21, 10] on CurveVel-A, with a Mean Absolute Error (MAE) of 0.0527 (vs. 0.0409), a Root Mean Squared Error (RMSE) of 0.1012 (vs. 0.0944), and a Structured Similarity (SSIM) of 0.8601 (vs. 0.8796). Furthermore, we demonstrate that leveraging learned structural priors via transfer learning allows DeepWaveRL to generalize efficiently to complex new fault geometries in FlatFault-A and CurveFault-A.

Our contribution is summarized as follows:

• DeepWaveRL Framework: We introduce DeepWaveRL, a reinforcement learning framework for self-supervised FWI under the strict scalar-reward regime.

• Stabilization and Spatial Credit Assignment: We stabilize policy gradients across a massive combinatorial action space and mitigate the spatial credit assignment bottleneck in the high-dimensional domain by integrating a map-level advantage estimator, a discretized spatial action space, and a sign-preserving non-linear reward transformation.

• Competitive Empirical Performance: We demonstrate that DeepWaveRL achieves competitive performance using only a scalar data-misfit reward without requiring access to ground-truth velocity maps or gradients from the forward operator.

• Empirical Variance Analysis: We provide an empirical analysis detailing the variancereduction properties of group-based policy optimization in FWI. We explicitly demonstrate how discrete action spaces and prior regularization overcome the optimization issues in high-dimensional continuous environments.

## 2 Method

In this section, we first briefly summarize the state-of-the-art group-based reinforcement learning algorithms and then present our DeepWaveRL and its components. After that, we provide the comparison of DeepWaveRL with other data-driven methods from a gradient perspective.

## 2.1 Preliminary

Shao et al. [16] introduces Group Relative Policy Optimization (GRPO) that enhances Proximal Policy Optimization (PPO) [15] by omitting the value function and estimating the advantage in a group-relative manner. This is followed by several variants such as Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) [22] and Group Sequence Policy Optimization (GSPO) [26], yielding superior training efficiency and performance. The core idea of GRPO is as follows.

For a specific question $q \sim P ( Q )$ , a group of $G$ responses $\{ o _ { i } \} _ { i = 1 } ^ { G }$ are sampled from an old policy network $\pi _ { \theta _ { \mathrm { o l d } } }$ . Each response $o _ { i }$ is then fed into a reward function to obtain the individual reward $\dot { R _ { i } }$ By normalizing the rewards within each group, an advantage ${ \hat { A } } _ { i }$ is assigned to each response. The policy network is optimized by maximizing the following clipped objective, similar to PPO:

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { q \sim P ( Q ) , \{ \sigma _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o d d } } } ( \cdot | q ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \sigma _ { i } | } \sum _ { t = 1 } ^ { | \sigma _ { i } | } \left[ L ( r _ { i , t } ( \theta ) , \hat { A } _ { i , t } , \epsilon ) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } | | \pi _ { \mathrm { r e f } } ) \right] ,\tag{2}
$$

and the L is given by

$$
L = \operatorname* { m i n } \Big [ r _ { i , t } ( \theta ) \hat { A } _ { i , t } , \mathrm { c l i p } \Big ( r _ { i , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \Big ) \hat { A } _ { i , t } \Big ] .\tag{3}
$$

Here, ϵ is the hyperparameter to determine the clipping boundaries, and $\beta$ is to control the importance of the KL divergence $D _ { \mathrm { K I } }$ between the online policy $\pi \theta$ and the frozen reference policy $\pi _ { \mathrm { r e f } } .$ In addition, $\hat { A } _ { i , t }$ and $r _ { i , t } ( \theta )$

![](images/6291faa8563530ef3633b7bed95a8d82115032027093b9a53a25e9b44381b421.jpg)  
Figure 2: Schematic illustration of the policy optimization pipeline in DeepWaveRL. For brevity, we only show the seismic data from one source.

are the group-based advantage estimation and importance ratio of $o _ { i , t } .$ , which is the t−th token in response $o _ { i }$ They are defined as:

$$
\hat { A _ { i , t } } = \hat { A } _ { i } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { i } \} _ { i = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { i } \} _ { i = 1 } ^ { G } ) } , \qquad r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \bigl ( o _ { i , t } \mid q , o _ { i , < t } \bigr ) } { \pi _ { \theta _ { \mathrm { o l d } } } \bigl ( o _ { i , t } \mid q , o _ { i , < t } \bigr ) } .\tag{4}
$$

## 2.2 Self-Supervised FWI via Reinforcement Learning

As illustrated in Figure 2, DeepWaveRL formulates FWI as a single-step policy learning problem. Each input seismic data $p \in \mathbb { R } ^ { \breve { N } _ { s } \times N _ { t } \times N _ { r } }$ is treated as a state and passed into the policy network π<sub>θ</sub>. Here, $N _ { s }$ denotes the number of sources used during data acquisition, $N _ { t }$ the number of recorded timesteps, and $N _ { r }$ the total number of receivers. The network then outputs the probability distribution of velocity $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { p } ) _ { h , w }$ at each spatial location $( h , w )$ . Following GRPO, we sample a group of G velocity maps $\{ \hat { V _ { i } } \} _ { i = } ^ { G }$ from the predicted joint probability distribution as 2D actions, where each $\hat { V } _ { i } = \{ \hat { v } _ { i , h , w } \} _ { h = 1 } ^ { H } , _ { w = 1 } ^ { W } .$ , and H and W are the vertical and horizontal dimensions of the velocity map. To reduce exploration burden, we use a discrete action space instead of the continuous one.

To assign a reward $R _ { i }$ to each sampled action $\hat { V _ { i } }$ , we employ a forward operator $f$ to simulate seismic data $\tilde { p _ { i } } = \bar { f ( \hat { V _ { i } } ) } \in \mathbb { R } ^ { N _ { s } \times N _ { t } \times N _ { r } }$ . The reward is then computed based on the misfit between the input seismic data p and the reconstruction $\tilde { p _ { i } }$ . We further compute the relative advantage $\hat { A } _ { i }$ within each group following Equation 4, but use a map-level importance ratio.

According to a recent work [26], the mismatch between the unit of reward and the unit of optimization objective can introduce high-variance noise and further lead to model collapse. In our settings, a reward is assigned to the whole 2D velocity map. Therefore, instead of computing the pixel-level importance ratios in Equation $^ { 4 , }$ we define the map-level importance ratio as:

$$
m _ { i } = \left[ \frac { \pi _ { \theta } ( \hat { V } _ { i } | p ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( \hat { V } _ { i } | p ) } \right] ^ { \frac { 1 } { H \cdot W } } = \exp \left[ \frac { 1 } { H \cdot W } \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } \log ( \frac { \pi _ { \theta } ( \hat { v } _ { i , h , w } | p ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( \hat { v } _ { i , h , w } | p ) } \right] .\tag{5}
$$

Consequently, the overall optimization objective can now be written as:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { p \sim \mathcal { P } , \{ \hat { V } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o d } } } ( \cdot \vert p ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \Big \{ \operatorname* { m i n } \left[ m _ { i } ( \theta ) \hat { A } _ { i } , \mathrm { c l i p } \Big ( m _ { i } ( \theta ) , 1 - \varepsilon _ { \mathrm { l o w } } , 1 + \varepsilon _ { \mathrm { h i g h } } \Big ) \hat { A } _ { i } \right] \Big \} ,\tag{6}
$$

where we follow Yu et al. [22] and decouple the lower and higher clipping range as $\varepsilon _ { \mathrm { l o w } }$ and $\varepsilon _ { \mathrm { h i g h } }$ , and $\mathcal { P }$ denotes the distribution of seismic data. We also remove the KL penalty term as our initial model is not as good as common pretrained language models, and we allow the model distribution to diverge from the initial model.

The policy network is thus trained to shift its output distribution toward higher-reward actions. Importantly, the entire training process is self-supervised, without the involvement of ground-truth velocity maps. Noteworthy, the optimization is driven entirely by the global scalar reward, completely removing the reliance on localized, pixel-level gradients from the forward operator.

Additionally, because this scalar-reward formulation requires only the input seismic data p and the forward operator to compute rewards, the policy network can undergo Test-Time Optimization (TTO). During inference, TTO fine-tunes the network weights on unseen test samples using the exact same policy optimization objective (Equation 6). This allows the agent to leverage its learned global structural prior as a stable initialization for sample-specific iterative refinement. This test-time optimization further boosts performance, which will be discussed in Section 3.3.

## 2.3 Strategies for Variance Reduction, Credit Assignment and Regularization

Discrete Action Space. During training, we treat the predicted velocity map $\hat { V } = \{ \hat { v } _ { h , w } \} _ { h = 1 } ^ { H } , _ { w = 1 } ^ { W }$ as a 2D action sampled from the joint distribution. We initially experimented with a continuous action space, where the network outputs two maps of size $H \times W$ , representing the mean $\mu _ { h , w }$ and standard deviation $\sigma _ { h , w }$ of a Gaussian distribution at each spatial location. Hence, the velocity at location $( h , w )$ was sampled as $\hat { v } _ { h , w } \sim \mathcal N ( \mu _ { h , w } , \sigma _ { h , w } )$ . However, this formulation led to highly unstable training and noisy predictions due to the extreme variance of the policy gradient estimator in a high-dimensional continuous space. To bound the exploration manifold and reduce this variance, we discretize the velocity values uniformly into B bins, treating the sampled action as a predicted category. Given an action $a _ { h , w } \in [ 0 , 1 , . . . , B - 1 )$ , we compute the corresponding velocity as

$$
v ( a _ { h , w } ) = \frac { v _ { \operatorname* { m a x } } - v _ { \operatorname* { m i n } } } { B } \cdot ( a _ { h , w } + 0 . 5 ) + v _ { \operatorname* { m i n } } ,\tag{7}
$$

where $v _ { \mathrm { m a x } }$ and $v _ { \mathrm { m i n } }$ are the largest and smallest possible velocities in a dataset. This discrete action space substantially improves both training stability and prediction quality by regularizing the combinatorial search space.

Sign-Preserving Logarithm Transformation in Reward. Similar to the loss function of UPFWI [9], we define the reward as the negative pixel-wise $\ell _ { 1 }$ and $\ell _ { 2 }$ distance between the input and reconstructed seismic data. To further enhance learning in the scalar-reward regime, we apply a sign-preserving logarithm transformation during the computation of rewards as:

$$
p ^ { \prime } = \operatorname { s i g n } ( p ) \cdot \log ( k \cdot | p | + c ) ,\tag{8}
$$

where k and c are hyperparameters to control the strength of the transformation. This non-linear transformation can compress dominant directive wave energy and amplify weaker signals $( \mathrm { e . g . }$ ., reflections and deep arrivals). Without the transformation, the advantage estimator assigns effectively zero credit to improvements in deepregion structures. By balancing the spatial sensitivity of the scalar reward, the log transformation ensures that structural updates in deeper regions yield meaningful, localized gradient signals. The reward function can then be described as:

$$
R = - \ell _ { 1 } ( p ^ { \prime } , \tilde { p } ^ { \prime } ) - \ell _ { 2 } ( p ^ { \prime } , \tilde { p } ^ { \prime } ) .\tag{9}
$$

Transfer Learning using Well-Trained Policies. Directly training our DeepWaveRL from scratch sometimes leads to unstable learning and convergence issues. We provide examples of predicted velocity maps generated by these models in Appendix G.8. The predictions exhibit unrealistically low velocities in deep regions after certain training steps and cannot recover. This is a direct consequence of the agent encountering local minima in the highly non-convex reward landscape. To address this issue, we propose to initialize the policy network with weights from a well-trained model on a different dataset. This transfer learning strategy allows prior knowledge of spatial velocity distributions to be reused across datasets, acting as a strong structural prior that regularizes the agent’s exploration to physically plausible manifolds. This eventually leads to more stable training and improved convergence.

## 2.4 Comparison with Previous FWI Methods from a Gradient Perspective

To demonstrate the relationship among supervised, exact-gradient self-supervised, and our scalar-reward RLbased self-supervised approach, we provide analysis from the perspective of gradient construction and propagation. Full derivations and detailed comparisons are provided in Appendix A.

Supervised: Gradients are directly computed from discrepancies with ground-truth velocity maps, $\nabla _ { \theta } \mathcal { L } _ { \mathrm { s u p } } \propto$ $\frac { \partial \parallel V - \hat { V } \parallel } { \partial \hat { V } }$ . While this yields stable optimization and strong supervision, it is impractical in real-world scenarios due to the scarcity of paired data.

Exact-Gradient Self-Supervised: Methods like UPFWI remove the need for ground-truth maps by backpropagating exact, dense gradients through a fully differentiable forward operator $f ,$ yielding $\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { d i f f } } \propto \frac { \partial f ( \hat { V } ) } { \partial \hat { V } } } \end{array}$ While this enables physics-guided learning, it fundamentally relies on access to analytical gradients, coupling the optimization tightly to ${ \dot { f } } .$

Scalar-Reward RL-Based self-supervised: In contrast to exact-gradient methods, DeepWaveRL converts the global reconstruction error into a scalar reward signal that acts merely as a multiplier for the policy gradients: $\nabla _ { \boldsymbol { \theta } } \mathcal { I } ( \boldsymbol { \theta } ) \propto \mathbb { E } \left[ \hat { A } \cdot \nabla _ { \boldsymbol { \theta } } \log m _ { i } ( \boldsymbol { \theta } ) \right]$

## 3 Experiments

In this section, we evaluate the performance of DeepWaveRL on OpenFWI [5]. We also investigate the impact of the discrete action space and other variance reduction components through ablation studies.

Table 1: Quantitative results evaluated on CVA. For models with discrete predictions, we report the mean estimate.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM↑</td></tr><tr><td rowspan=1 colspan=1>InversionNet</td><td rowspan=1 colspan=1>0.0409</td><td rowspan=1 colspan=1>0.0944</td><td rowspan=1 colspan=1>0.8796</td></tr><tr><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0739</td><td rowspan=1 colspan=1>0.1369</td><td rowspan=1 colspan=1>0.7420</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL</td><td rowspan=1 colspan=1>0.0717</td><td rowspan=1 colspan=1>0.1300</td><td rowspan=1 colspan=1>0.8303</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL + TTO</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.1012</td><td rowspan=1 colspan=1>0.8601</td></tr></table>

![](images/579447a010ec6fbf5ad67dd3aeafe09e24363691f86660774c14a8944d919e6f.jpg)  
Figure 3: Illustration of ground truth and inversion results of different methods on CVA.

## 3.1 Datasets

We verify our method on CurveVel-A (CVA), FlatFault-A (FFA) and CurveFault-A (CFA) of OpenFWI [5], an open-source collection of large-scale, multi-structural benchmark datasets for data-driven seismic FWI.

CVA contains velocity maps composed of curved layers with clear interfaces, while FFA and CFA focus more on geological fault identification and have velocity maps with flat and curved layers, respectively. CVA contains 30K pairs of seismic data and velocity maps. Following the official data split, we use 24K for training and 6K for testing. FFA and CFA contain 54K samples each, and we use 48K/6K splitting.

Each velocity map in all three datasets has a size of 70 × 70, with a grid size of 10 meters in horizontal and depth directions. The velocity value ranges from 1,500 meter/second to 4,500 meter/second. For seismic data, five sources are placed evenly with a 170-meter spacing and a central source frequency of 15 Hz. The seismic data are recorded by 70 receivers at 10-meter intervals, each collecting 1,000 timesteps over 1 second. This results in seismic data of shape $5 \times 1 0 0 0 \times 7 0$ . For additional details, we refer readers to the original OpenFWI paper [5].

## 3.2 Experiment Settings

Evaluation Metrics: We evaluate predicted velocity maps using MAE, RMSE, and Structural Similarity (SSIM), consistent with prior work [21, 7, 5]. MAE and RMSE quantify pixel-wise errors, while SSIM captures perceptual similarity, reflecting the structured information of velocity maps where distortions can be easily perceived by a human. Note that all measurements are computed on normalized velocity maps, with MAE and RMSE in the range [−1, 1], and SSIM in [0, 1].

Comparison: We compare our method to InversionNet [21] which achieves the state-of-the-art performance when trained solely on each dataset [10]. Following the configurations in OpenFWI, we train UPFWI [9] as a self-supervised baseline. Additionally, we implement simultaneous perturbation stochastic approximation (SPSA) [18] as a derivative-free optimization baseline. However, its performance is not comparable to other methods, so we report the results and details of SPSA in Appendix D.

Implementation: We follow the UPFWI paper and incorporate a finite-difference forward operator. For DeepWaveRL, as no gradient from the forward operator is required, we develop a custom, pre-compiled CUDA operator that includes in-place operations for acceleration. Training details are provided in Appendix B.

## 3.3 Main Results

Results on CVA: Table 1 shows the quantitative results of different methods on CVA. For the models that predict velocity as discrete values, we report the mean prediction (expected value) where we compute the expectation over the bin values using their predicted probabilities. Among all the models, InversionNet yield the best performance, which is expected as it leverages supervised learning and directly predicts continuous velocity values. In comparison, our DeepWaveRL with test-time optimization (DeepWaveRL+TTO) attains the second-best performance with a slight gap. Notably, even without TTO, DeepWaveRL achieves better performance compared to UPFWI, and TTO further provides a substantial improvement.

Table 2: Quantitative results evaluated on FFA and CFA.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM↑</td></tr><tr><td rowspan=4 colspan=1>FFA</td><td rowspan=1 colspan=1>InversionNet</td><td rowspan=1 colspan=1>0.0098</td><td rowspan=1 colspan=1>0.0276</td><td rowspan=1 colspan=1>0.9880</td></tr><tr><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0214</td><td rowspan=1 colspan=1>0.0484</td><td rowspan=1 colspan=1>0.9674</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL</td><td rowspan=1 colspan=1>0.0301</td><td rowspan=1 colspan=1>0.0557</td><td rowspan=1 colspan=1>0.9062</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL + TTO</td><td rowspan=1 colspan=1>0.0268</td><td rowspan=1 colspan=1>0.0476</td><td rowspan=1 colspan=1>0.9146</td></tr><tr><td rowspan=4 colspan=1>CFA</td><td rowspan=1 colspan=1>InversionNet</td><td rowspan=1 colspan=1>0.0164</td><td rowspan=1 colspan=1>0.0480</td><td rowspan=1 colspan=1>0.9721</td></tr><tr><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0280</td><td rowspan=1 colspan=1>0.0641</td><td rowspan=1 colspan=1>0.9492</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL</td><td rowspan=1 colspan=1>0.0362</td><td rowspan=1 colspan=1>0.0703</td><td rowspan=1 colspan=1>0.9111</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL + TTO</td><td rowspan=1 colspan=1>0.0278</td><td rowspan=1 colspan=1>0.0502</td><td rowspan=1 colspan=1>0.9323</td></tr></table>

![](images/840313c6cbced05f171c437d6806faa8925b6ef04be93a7fbf9f6f259ff1c506.jpg)  
Figure 4: Illustration of ground truth and inversion results of different methods on FFA (top) and CFA (bottom).

Figure 3 further illustrates examples of ground-truth velocity maps and predictions from different methods. Here, the results of DeepWaveRL models are all mean predictions. Consistent with our quantitative analysis, InversionNet produces sharp layer boundaries and smooth and uniform velocities within layers. The predictions of UPFWI contain artifacts in deep regions. In contrast, DeepWaveRL+TTO yields more details. As highlighted in the first row, the predictions of DeepWaveRL+TTO have more accurate velocities in deep regions, whereas InversionNet introduces additional layers and predicts inaccurate velocities. Another observation is that only DeepWaveRL+TTO precisely reconstructs the subsurface structure in the highlighted areas in the second row. Moreover, when comparing the results of DeepWaveRL and DeepWaveRL+TTO, we find that TTO helps eliminate artifacts, recover curved structures, and improve the accuracy of intra-layer velocities. More visualization results are shown in Appendix G.1.

Transfer Learning Results on FFA and CFA: Table 2 lists the quantitative results on FFA and CFA, and mean predictions are reported. For transfer learning, we adopt the last checkpoint of the DeepWaveRL+TTO on CVA. All the baseline models are trained from scratch. For FFA, our DeepWaveRL+TTO outperforms UPFWI in terms of RMSE. For CFA, it outperforms UPFWI in terms of MAE and RMSE, with small gaps in SSIM. This indicates that DeepWaveRL+TTO yields generally accurate predictions, but there may be small shifts or artifact that are visually noticeable.

The visualization results in Figure 4 further support our hypothesis. Despite some stripe artifacts, our DeepWaveRL and DeepWaveRL+TTO generate more precise details in some shallow regions. As highlighted in the first row (FFA), DeepWaveRL models reconstruct the fault on the top-left corner, while it is barely visible in the predictions of InversionNet and UPFWI. Similarly, in the second row (CFA), our DeepWaveRL models precisely capture the triangle-shaped region. More visualization results are shown in Appendix G.2.

We further report wall-clock time of training and inference and DeepWaveRL+TTO with different computation budgets in Appendix C. Transfer learning for baselines and UPFWI with TTO are also provided in Appendix E for reference.

Table 3: Quantitative results evaluated on CVA, with different choices of action space.
<table><tr><td rowspan=1 colspan=1>Setting</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM↑</td></tr><tr><td rowspan=1 colspan=1>Continuous Action Space</td><td rowspan=1 colspan=1>0.1443</td><td rowspan=1 colspan=1>0.2073</td><td rowspan=1 colspan=1>0.4359</td></tr><tr><td rowspan=1 colspan=1>Discrete Action Space</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.1012</td><td rowspan=1 colspan=1>0.8601</td></tr></table>

![](images/11b2c7d7d27ce07add2060bfe5100e4bdcaa581378278d6851802d73c307921a.jpg)  
Figure 5: Illustration of ground truth and inversion results of DeepWaveRL with different choices of action space

## 3.4 Ablations

Continuous vs. Discrete Action Space: We further analyze how the choice of action space affects model performance. The quantitative results are summarized in Table 3. For simplicity, we denote DeepWaveRL with a discrete action space as DeepWaveRL-D, and with a continuous action space as DeepWaveRL-C. In terms of all three metrics, DeepWaveRL-D outperforms DeepWaveRL-C to a large extent. The gap is particularly large in SSIM, suggesting that DeepWaveRL-C fails to produce results consistent with human perceptual quality.

Figure 5 provides qualitative comparisons between DeepWaveRL-C and DeepWaveRL-D, which further support our quantitative analysis. While DeepWaveRL-C can recover some structures in shallow regions, there are plenty of artifacts all over the predictions. In particular, the high-velocity areas are severely corrupted, making boundaries unrecognizable. Furthermore, these artifacts persist even when training is extended, indicating that the continuous action space poses significant optimization challenges. By contrast, discretization reduces the complexity of the action space and greatly stabilizes training, leading to more accurate and reliable results.

Discretization Resolution: To determine the optimal boundary for the exploration manifold, we evaluate DeepWaveRL across different discretization resolutions in terms of the number of bins B. We employ the last checkpoint of DeepWaveRL+TTO on CVA and append a 1 × 1 convolutional layer as an adapter to output predictions of different bins. The quantitative results listed in Table 5 show a trade-off between physical resolution and optimization difficulties. The visualization results are depicted in Appendix G.3.

Logarithm Transformation: To evaluate the effect of the sign-preserving logarithm transformation, we train our DeepWaveRL without transformation on CVA. The resulting MAE, RMSE, and SSIM are 0.0785, 0.1383, and 0.8195, respectively. Compared to the results of DeepWaveRL with transformation in Table 1, the performance degrades in terms of all three metrics. This is consistent with the visualization results in Appendix G.4, where some details are missing in deep regions.

Group Size Scaling: To empirically analyze how group size G affects the gradient signal-to-noise ratio, we train DeepWaveRL with various group sizes on CVA from scratch. The quantitative results are listed in Table 6. As expected theoretically, the large group size (G = 256) is required to suppress the variance inherent to high-dimensional spatial RL, ensuring the advantage estimator reliably differentiates between spatially coherent and incoherent actions. The visualization results are shown in Appendix G.5.

Prior as Regularization: To analyze the importance of the structural prior to stabilize training, we directly train DeepWaveRL on the test set of CVA, starting from random initialization. Without the prior, DeepWaveRL yields significantly degraded performance. The resulting MAE, RMSE and SSIM are 0.2120, 0.3004, and 0.5832, respectively. Additionally, we transfer the checkpoint on CVA to the test set of FFA and CFA without fine-tuning on the training set. The quantitative results in Table 4 indicate that fine-tuning helps refine the prior. The visualization results are shown in Appendix G.6.

Table 4: Comparison of DeepWaveRL+TTO with and without fine-tuning on the training set of FFA and CFA.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Fine-tuning</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM↑</td></tr><tr><td rowspan=2 colspan=1>FFA</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>0.0268</td><td rowspan=1 colspan=1>0.0476</td><td rowspan=1 colspan=1>0.9146</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.0287</td><td rowspan=1 colspan=1>0.0520</td><td rowspan=1 colspan=1>0.9347</td></tr><tr><td rowspan=2 colspan=1>CFA</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>0.0278</td><td rowspan=1 colspan=1>0.0502</td><td rowspan=1 colspan=1>0.9323</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.0358</td><td rowspan=1 colspan=1>0.0647</td><td rowspan=1 colspan=1>0.9153</td></tr></table>

Table 5: Comparison of DeepWaveRL+TTO with different numbers of bins on CVA.
<table><tr><td rowspan=1 colspan=1># Bins</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM↑</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>0.0546</td><td rowspan=1 colspan=1>0.1027</td><td rowspan=1 colspan=1>0.8493</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.1012</td><td rowspan=1 colspan=1>0.8601</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>0.0519</td><td rowspan=1 colspan=1>0.1010</td><td rowspan=1 colspan=1>0.8617</td></tr><tr><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>0.0521</td><td rowspan=1 colspan=1>0.1010</td><td rowspan=1 colspan=1>0.8609</td></tr></table>

Table 6: Comparison of DeepWaveRL with different group sizes on CVA.
<table><tr><td rowspan=1 colspan=1>Group Size</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM↑</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>0.0948</td><td rowspan=1 colspan=1>0.1535</td><td rowspan=1 colspan=1>0.7878</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.0796</td><td rowspan=1 colspan=1>0.1324</td><td rowspan=1 colspan=1>0.8214</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.0885</td><td rowspan=1 colspan=1>0.1680</td><td rowspan=1 colspan=1>0.8151</td></tr><tr><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>0.0717</td><td rowspan=1 colspan=1>0.1300</td><td rowspan=1 colspan=1>0.8303</td></tr></table>

## 4 Discussion

During qualitative analysis, we observe that mode collapse occurs in the predictions of DeepWaveRL (\~1.85% of test cases), as illustrated in Appendix G.7. This ratio decreases to 1.73% after TTO. There is no consistent pattern in velocity maps or seismic data that guarantees the occurrence of mode collapse. We plan to dive deeper to solve this issue in our future work. Another limitation is that DeepWaveRL still cannot perfectly recover the structures in deeper regions due to the inherent attenuation of signals in these regions. Our sign-preserving logarithm transformation is one of the solutions, but more advanced algorithms are still needed. Furthermore, our DeepWaveRL framework enables another potential research direction, which is to incorporate conventional regularization terms such as total variation into the reward design.

## 5 Related Work

Deep Learning for FWI: Deep learning approaches to FWI span data-driven, physics-informed, and hybrid paradigms [12, 1, 23]. Fully supervised methods [2, 21, 25, 11] learn direct mappings from seismic data to velocity models using paired data, which are costly to acquire and often lead to poor generalization under domain shifts. To reduce reliance on labels and improve robustness, self-supervised strategies have emerged. Feng et al. [6, 7] decouple the seismic encoder and velocity decoder by leveraging latent space correlations, enabling separate training. SiameseFWI [14] explores self-supervision with a Siamese network that better aligns simulated and observed data. Semi-supervised learning has also been explored. [19] proposes a CycleGANbased framework to reconstruct missing low-frequency components in field data. Other methods generate pseudo-labels from unlabeled or auxiliary data [13, 3], bridging the gap between labeled and unlabeled domains. Unsupervised methods such as UPFWI [9] and Jia et al. [8] go further by eliminating labels entirely. These approaches minimize waveform mismatches under physical constraints, using differentiable forward modeling to optimize predicted velocity maps. However, their reliance on computationally intensive and differentiable solvers limits scalability to high-resolution, elastic, or 3D FWI, and precludes use with black-box simulators. Diffusion models offer an alternative by learning generative priors that guide the inverse process via plug-and-play (PnP) denoising [17, 4, 24]. [20] successfully applies this strategy to FWI. While these models avoid paired supervision, they still require large velocity datasets for training and incur high inference costs due to repeated forward simulations during sampling.

Reinforcement learning and group-based policy optimization: Policy-gradient RL methods (e.g., PPO [15]) provide a principled way to optimize stochastic policies via likelihood-ratio estimators, and have been widely used in sequential generation and control. Recent advances in group-/sequence-level policy optimization demonstrate that performing importance-weighting and clipping at the unit-of-reward level (group or sequence) can reduce variance and stabilize training for structured outputs. In particular, DeepSeekMath [16] replaces value-function estimation with group-relative normalization, yielding more efficient and stable updates. They show that this framework can be scaled to mathematical reasoning tasks with strong generalization. DAPO [22] extends the paradigm by decoupling clipping ranges and introducing dynamic sampling, further improving stability under diverse reward distributions. GSPO [26] generalizes these ideas to full sequence-level optimization with the importance ratio based on sequence likelihood, aligning long-horizon objectives with token-level policies. Inspired by this progression, DeepWaveRL adapts the same philosophy to the geophysics domain by treating an entire velocity map as a structured action, making map-level optimization a natural extension of group-based RL methods for physics-driven inverse problems.

## 6 Conclusion

In this study, we introduce DeepWaveRL, a reinforcement learning framework for self-supervised full waveform inversion under the strict scalar-reward regime. By integrating a map-level advantage estimator, discretized spatial actions, a sign-preserving logarithmic reward transformation, and prior regularization via transfer learning, DeepWaveRL overcomes the massive combinatorial action space and spatial credit assignment bottlenecks inherent to the high-dimensional velocity domain. Our empirical analysis demonstrates that these strategies effectively bound the exploration manifold and suppress policy gradient variance, enabling DeepWaveRL to achieve competitive structural fidelity compared to both supervised and self-supervised baselines. This approach provides a flexible paradigm for FWI, offering new possibilities in real-world settings

## References

[1] A. Adler, M. Araya-Polo, and T. Poggio. Deep learning for seismic inverse problems: Toward the acceleration of geophysical analysis workflows. IEEE Signal Processing Magazine, 38:89–119, 2021. doi: https://doi.org/10.1109/MSP.2020.3037429.

[2] M. Araya-Polo, J. Jennings, A. Adler, and T. Dahlke. Deep-learning tomography. The Leading Edge, 37 (1):58–66, 2018.

[3] A. Cai, H. Qiu, and F. Niu. Semi-supervised surface wave tomography with wasserstein cycle-consistent GAN: method and application to southern california plate boundary region. Journal of Geophysical Research: Solid Earth, 127(3):e2021JB023598, 2022. doi: https://doi.org/10.1029/2021JB023598.

[4] H. Chung, J. Kim, M. T. McCann, M. L. Klasky, and J. C. Ye. Diffusion posterior sampling for general noisy inverse problems. In 11th International Conference on Learning Representations, ICLR 2023, 2023.

[5] C. Deng, S. Feng, H. Wang, X. Zhang, P. Jin, Y. Feng, Q. Zeng, Y. Chen, and Y. Lin. Openfwi: Largescale multi-structural benchmark datasets for full waveform inversion. Advances in Neural Information Processing Systems, 35:6007–6020, 2022.

[6] Y. Feng, Y. Chen, S. Feng, P. Jin, Z. Liu, and Y. Lin. An intriguing property of geophysics inversion. In International Conference on Machine Learning, pages 6434–6446. PMLR, 2022.

[7] Y. Feng, Y. Chen, P. Jin, S. Feng, and Y. Lin. Auto-linear phenomenon in subsurface imaging. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

[8] A. Jia, J. Sun, B. Du, and Y. Lin. Seismic full waveform inversion with uncertainty analysis using unsupervised variational deep learning. IEEE Transactions on Geoscience and Remote Sensing, 63:1–16, 2025. doi: 10.1109/TGRS.2025.3564647.

[9] P. Jin, X. Zhang, Y. Chen, S. X. Huang, Z. Liu, and Y. Lin. Unsupervised learning of full-waveform inversion: Connecting CNN and partial differential equation in a loop. In Proceedings of the Tenth International Conference on Learning Representations (ICLR), 2022.

[10] P. Jin, Y. Feng, S. Feng, H. Wang, Y. Chen, B. Consolvo, Z. Liu, and Y. Lin. An empirical study of large-scale data-driven full waveform inversion. Scientific Reports, 14(1):20034, 2024.

[11] S. Li, B. Liu, Y. Ren, Y. Chen, S. Yang, Y. Wang, and P. Jiang. Deep-Learning Inversion of Seismic Data. IEEE Transactions on Geoscience and Remote Sensing, 58(3):2135–2149, Mar. 2020. ISSN 1558-0644. doi: 10.1109/TGRS.2019.2953473.

[12] Y. Lin, J. Theiler, and B. Wohlberg. Physics-guided data-driven seismic inversion: Recent progress and future opportunities in full waveform inversion. IEEE Signal Processing Magazine, 40:115–133, 2023. doi: https://doi.org/10.1109/MSP.2022.3217658.

[13] R. Rojas-Gómez, J. Yang, Y. Lin, J. Theiler, and B. Wohlberg. Physics-consistent data-driven waveform inversion with adaptive data augmentation. IEEE Geoscience and Remote Sensing Letters, 19:1–5, 2022. doi: https://doi.org/10.1109/LGRS.2020.3022021.

[14] O. M. Saad, R. Harsuko, and T. Alkhalifah. SiameseFWI: a deep learning network for enhanced full waveform inversion. Journal of Geophysical Research: Machine Learning and Computation, 1(3): e2024JH000227, 2024. doi: https://doi.org/10.1029/2024JH000227.

[15] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[16] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv. org/abs/2402.03300.

[17] Y. Song, L. Shen, L. Xing, and S. Ermon. Solving inverse problems in medical imaging with score-based generative models. In International Conference on Learning Representations, 2022.

[18] J. C. Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE transactions on automatic control, 37(3):332–341, 2002

[19] H. Sun, Y. Sun, R. Nammourand, C. Rivera, P. Williamson, and L. Demanet. Learning with real data without real labels: a strategy for extrapolated full-waveform inversion with field data. Geophysical Journal International, 235:1761–1777, 2023. doi: https://doi.org/10.1093/gji/ggad330.

[20] F. Wang, X. Huang, and T. Alkhalifah. A prior regularized full waveform inversion using generative diffusion models. IEEE Transactions on Geoscience and Remote Sensing, 61:1–11, 2023. doi: https: //doi.org/10.1109/TGRS.2023.3337014.

[21] Y. Wu and Y. Lin. InversionNet: An efficient and accurate data-driven full waveform inversion. IEEE Transactions on Computational Imaging, 6:419–433, 2019.

[22] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

[23] X. Yu and J. Ma. Deep learning for geophysics: Current and future trends. Reviews of Geophysics, 59: e2021RG000742, 2021. doi: https://doi.org/10.1029/2021RG000742.

[24] B. Zhang, W. Chu, J. Berner, C. Meng, A. Anandkumar, and Y. Song. Improving diffusion inverse problem solving with decoupled noise annealing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20895–20905, 2025.

[25] Z. Zhang, Y. Wu, Z. Zhou, and Y. Lin. VelocityGAN: Subsurface velocity image estimation using conditional adversarial networks. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 705–714. IEEE, 2019.

[26] C. Zheng, S. Liu, M. Li, X.-H. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

## A Comparison with Previous FWI Methods from a Gradient Perspective

To elucidate the relationship among supervised, exact-gradient self-supervised, and our scalar-reward reinforcement learning based self-supervised approaches, we provide a unified analysis from the perspective of gradient construction and propagation.

Supervised Learning: In the supervised paradigm, a neural network $g _ { \theta }$ maps seismic measurements p to a predicted velocity map $\hat { V } = g _ { \boldsymbol { \theta } } ( \boldsymbol { p } )$ . Training relies on paired ground-truth velocity maps v, with a loss function of the form

$$
{ \mathcal L } _ { \mathrm { s u p } } ( \theta ) = \mathbb E _ { ( p , V ) \sim \mathcal D } \parallel V - g _ { \theta } ( p ) \parallel ,\tag{10}
$$

where D denotes the joint distribution of seismic data and corresponding velocity maps, and $\| \cdot \|$ denotes a generic norm $( \mathbf { e . g . , } \ell _ { 1 } , \ell _ { 2 }$ , or mixed/perceptual norms). The gradient is obtained via direct backpropagation:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { s u p } } = \mathbb { E } _ { ( \boldsymbol { p } , \boldsymbol { V } ) \sim \mathcal { D } } \underbrace { \partial \mathcal { L } _ { \mathrm { s u p } } } _ { \partial \hat { V } } \cdot \frac { \partial \hat { V } } { \partial \boldsymbol { \theta } } = \mathbb { E } _ { ( \boldsymbol { p } , \boldsymbol { V } ) \sim \mathcal { D } } \underbrace { \frac { \partial \mathcal { L } _ { \mathrm { s u p } } } { \partial g _ { \boldsymbol { \theta } } ( \boldsymbol { p } ) } } _ { \mathrm { G r o u n d t r u t h i n v o l v e d } } \cdot \frac { \partial g _ { \boldsymbol { \theta } } ( \boldsymbol { p } ) } { \partial \boldsymbol { \theta } } ,\tag{11}
$$

where the learning signal is explicitly anchored to the availability of ground-truth velocity maps. While this yields stable optimization and strong supervision, it is impractical in real-world scenarios due to the scarcity of paired data.

Exact-Gradient Self-Supervised Learning: The UPFWI framework [9] removes the dependence on ground truth by incorporating a differentiable forward operator f that simulates seismic data from predicted velocity maps:

$$
\hat { V } = g _ { \theta } ( p ) , \quad \tilde { p } = f ( \hat { V } ) .\tag{12}
$$

The reconstruction objective is defined as

$$
L _ { \mathrm { d i f f } } ( \theta ) = \mathbb { E } _ { p \sim \mathcal { P } } \lVert p - \tilde { p } \rVert = \mathbb { E } _ { p \sim \mathcal { P } } \lVert p - f ( g _ { \theta } ( p ) ) \rVert ,\tag{13}
$$

with gradients computed via the chain rule:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { p \sim \mathcal { P } } \frac { \partial \mathcal { L } _ { \mathrm { d i f f } } } { \partial \tilde { p } } \cdot \frac { \partial \tilde { p } } { \partial \tilde { V } } \cdot \frac { \partial \hat { V } } { \partial \theta } = \mathbb { E } _ { p \sim \mathcal { P } } \frac { \partial \mathcal { L } _ { \mathrm { d i f f } } } { \partial f ( g _ { \theta } ( p ) ) } \cdot \underbrace { \frac { \partial f ( g _ { \theta } ( p ) ) } { \partial g _ { \theta } ( p ) } } _ { \mathrm { D i f f e r e n t i a b l e } } \cdot \frac { \partial g _ { \theta } ( p ) } { \partial \theta } .\tag{14}
$$

This formulation enables end-to-end training using only seismic data, yet it couples the optimization tightly to the availability of a differentiable solver to compute $\partial f / \partial { \hat { V } }$

Scalar-Reward Self-Supervised Learning via Reinforcement Learning: Our proposed DeepWaveRL frames FWI as a policy optimization problem. With the definition in the above sections, we can derive the gradient of our objective as follows (clipping is omitted for brevity):

$$
\nabla _ { \theta } \mathcal { I } ( \theta ) = \nabla _ { \theta } \mathbb { E } _ { p \sim \mathcal { P } , \{ \hat { V } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \vert p ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \Big \{ m _ { i } ( \theta ) \hat { A } _ { i } \Big \}\tag{15}
$$

$$
= \mathbb { E } _ { p \sim \mathcal { P } , \{ \hat { V } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | p ) } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } m _ { i } ( \theta ) \hat { A } _ { i } \cdot \nabla _ { \theta } \log m _ { i } ( \theta ) \Bigg ]\tag{16}
$$

$$
= \mathbb { E } _ { p \sim \mathcal { P } , \{ \hat { V } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | p ) }\tag{17}
$$

$$
\left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \left( \frac { \pi _ { \theta } ( \hat { V } _ { i } \mid p ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( \hat { V } _ { i } \mid p ) } \right) ^ { \frac { 1 } { H \cdot W } } \underbrace { \hat { A } _ { i } } _ { \sim } \cdot \frac { 1 } { H \cdot W } \sum _ { h , w } \nabla _ { \theta } \log \pi _ { \theta } ( \hat { v } _ { i , h , w } \mid p ) \right] .\tag{18}
$$

Unlike supervised and differentiable self-supervised learning, where the loss must be differentiable with respect to the high-dimensional spatial output $\hat { V }$ and thus couples optimization tightly to the properties of $f ,$ reinforcement learning completely severs the dependency and transforms the reconstruction error into a scalar reward signal that directly re-weight policy gradients.

Thus, the three paradigms can be interpreted within a common gradient-based framework: Supervised: Gradients are directly computed from discrepancies with labeled velocity maps, yielding stable optimization but requiring costly ground truth. Exact-gradient self-supervised: Gradients originate from seismic reconstruction error and backpropagate through a differentiable forward operator, enabling physics-guided learning but restricted by differentiability. Scalar-reward RL-based self-supervised: Gradients arise from log-likelihood weighting in policy space, with seismic misfit entering only as a reward. This bypasses the need for exact physics gradients entirely, and enables stochastic exploration in complex inversion landscapes.

## B Training Details

We normalize the input seismic data to the range $[ - 1 , 1 ]$ and apply the logarithm transformation with $k = 3$ and $c = 0$ on seismic data. For optimization, we employ the AdamW optimizer with momentum parameters $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and a weight decay of $1 \times 1 0 ^ { - 4 }$ to update all parameters of the network. The details of hyperparameters and training settings are provided in Table 7. The $\epsilon _ { \mathrm { l o w } }$ and $\epsilon _ { \mathrm { h i g h } }$ are 0.2 and 0.27, respectively. For the network architecture, we adopt a four-layer encoder-decoder Vision Transformer (ViT), and we append four convolutional blocks with upsampling layers (5× and 2×), batch normalization, and leaky ReLU as activation functions to map the output of the decoder to $7 0 \times 7 0$ velocity map with 100 bins. We implement our models in Pytorch and train them on 16 NVIDIA H100 GPUs.

## C Wall-Clock Time and Performance with Various Budgets

The wall-clock training times are summarized in Table 8, and the single-pass wall-clock inference times are listed in Table 9. The TTO process takes around 1 hour and 15 minutes for CVA and 2 hours 40 minutes for FFA/CFA on 16xH100. Considering the number of test samples (6000), each sample in CVA takes approximately 0.75 seconds, and the one in FFA/CFA takes 1.6 seconds. For TTO on CVA, there are 11,520 forward simulations per sample (720 epochs). For FFA/CFA, there are 25,600 (800 epochs).

Table 7: Training details
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>CVA</td><td rowspan=1 colspan=2>FFA &amp; CFA</td></tr><tr><td rowspan=1 colspan=1>Test-time Optimization</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1>Training Steps</td><td rowspan=1 colspan=1>44,880</td><td rowspan=1 colspan=1>1,440</td><td rowspan=1 colspan=1>7,360</td><td rowspan=1 colspan=1>1,600</td></tr><tr><td rowspan=1 colspan=1>Initial Learning Rate</td><td rowspan=1 colspan=1>8e-4</td><td rowspan=1 colspan=1>1.6e-4</td><td rowspan=1 colspan=1>6.4e-4</td><td rowspan=1 colspan=1>6.4e-4</td></tr><tr><td rowspan=1 colspan=1>Learning Rate Decay</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1,360</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Batch Size</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>2048</td></tr><tr><td rowspan=1 colspan=1>Group Size</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>32</td></tr></table>

We further evaluate the model performance at intermediate TTO milestones to demonstrate the performance of DeepWaveRL+TTO with different computation budgets. The quantitative results are shown in Table 10, the visualized results are shown in Figure 6.

Table 8: Wall-clock training time.
<table><tr><td rowspan=1 colspan=1>Method (Hardware)</td><td rowspan=1 colspan=1>CVA</td><td rowspan=1 colspan=1>FFA/CFA</td></tr><tr><td rowspan=1 colspan=1>InversionNet (1xH100)</td><td rowspan=1 colspan=1>51 mins</td><td rowspan=1 colspan=1>1 hr 30 mins</td></tr><tr><td rowspan=1 colspan=1>UPFWI (4xH100)</td><td rowspan=1 colspan=1>14 hrs 45 mins</td><td rowspan=1 colspan=1>27 hrs</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL (16xH100)</td><td rowspan=1 colspan=1>27 hrs</td><td rowspan=1 colspan=1>11 hrs</td></tr></table>

Table 9: Wall-clock inference time (single pass inference).
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Mean</td><td rowspan=1 colspan=1>Std</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL</td><td rowspan=1 colspan=1>0.3773</td><td rowspan=1 colspan=1>0.8622</td></tr><tr><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.3975</td><td rowspan=1 colspan=1>0.9789</td></tr><tr><td rowspan=1 colspan=1>InversionNet</td><td rowspan=1 colspan=1>0.3381</td><td rowspan=1 colspan=1>0.9418</td></tr></table>

Table 10: Comparison of DeepWaveRL+TTO with different computation budgets.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Epoch</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM ↑</td></tr><tr><td rowspan=3 colspan=1>CVA</td><td rowspan=1 colspan=1>720</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.1012</td><td rowspan=1 colspan=1>0.8601</td></tr><tr><td rowspan=1 colspan=1>360</td><td rowspan=1 colspan=1>0.0569</td><td rowspan=1 colspan=1>0.1075</td><td rowspan=1 colspan=1>0.8524</td></tr><tr><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>0.0676</td><td rowspan=1 colspan=1>0.1172</td><td rowspan=1 colspan=1>0.8455</td></tr><tr><td rowspan=3 colspan=1>FFA</td><td rowspan=1 colspan=1>800</td><td rowspan=1 colspan=1>0.0268</td><td rowspan=1 colspan=1>0.0476</td><td rowspan=1 colspan=1>0.9146</td></tr><tr><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>0.0270</td><td rowspan=1 colspan=1>0.0471</td><td rowspan=1 colspan=1>0.9143</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>0.0277</td><td rowspan=1 colspan=1>0.0487</td><td rowspan=1 colspan=1>0.9134</td></tr><tr><td rowspan=3 colspan=1>CFA</td><td rowspan=1 colspan=1>800</td><td rowspan=1 colspan=1>0.0278</td><td rowspan=1 colspan=1>0.0502</td><td rowspan=1 colspan=1>0.9323</td></tr><tr><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>0.0291</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.9295</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>0.0307</td><td rowspan=1 colspan=1>0.0564</td><td rowspan=1 colspan=1>0.9255</td></tr></table>

## D Derivative-Free Optimization Baseline

We implement simultaneous perturbation stochastic approximation (SPSA) as the derivative-free optimization baseline on a subset of 100 samples from CVA test set.

First, we use a constant velocity of 2500 m/s as the initial model and optimize it with a learning rate of 50,000 and a perturbation magnitude of 100. The optimization is performed for 12,000 iterations per sample (24,000 forward simulations), which is approximately twice simulations of DeepWaveRL+TTO. We observe that SPSA struggled to converge to a physically reasonable velocity map. The resulting MAE, RMSE and SSIM are 0.5020, 0.6444, and 0.1457, respectively.

We then use the mean prediction of DeepWaveRL on CVA training set as the warm-started initial model (as shown in Figure 7). The performance of SPSA improves (MAE:0.3590, RMSE:0.4890, SSIM:0.2226) compared to the one with constant initialization. These results strongly highlight the fundamental difficulty of optimizing a high-dimensional structured space using generic zeroth-order methods.

Comparison of DeepWaveRL and Perturbation-Based Optimizers: Under the scalar-reward regime, the classical alternatives to our method are derivative-free optimization algorithms such as Simultaneous Perturbation

![](images/e572948a91e8c373d75f4ed6039695aa969d1740cddd57e586d9d5309645f6d6.jpg)  
Figure 6: Comparison of DeepWaveRL+TTO with different computation budgets on CVA (top row), FFA (middle row) and CFA(bottom row).

![](images/292d6e5a26b514fab2d9ae954b33fb39a7fbb99c81b93da36c31d36311320cd1.jpg)  
Figure 7: Illustration of ground truth and inversion results of SPSA and DeepWaveRL+TTO on CVA

Stochastic Approximation (SPSA) or Evolutionary Strategies (e.g., CMA-ES). While these paradigms are wellestablished, they face severe scalability bottlenecks when applied to the high-dimensional, structured output space of FWI as they typically perturb the space directly.

In contrast, our DeepWaveRL fundamentally changes the optimization target. Rather than directly perturbing independent velocity pixels, DeepWaveRL optimizes the weights of a deep policy network. The Vision Transformer and convolutional layers impose strong spatial inductive biases, inherently restricting the exploration manifold to physically coherent, spatially smooth structures. Therefore, while both DeepWaveRL and traditional methods reply on the scalar data misfit without accessing gradients from the forward operator, RL provides a vastly superior mechanism for navigating high-dimensional structured physics problems by leveraging the representational power and architectural priors of deep neural networks.

## E Baselines with Transfer Learning and TTO

For baselines with transfer learning, we initialize both InversionNet and UPFWI with their CVA pre-trained weights and fine-tune them on FFA and CFA. The results are shown in Table 11. It is observed that transferring the learned priors yields improved performance for both baselines. However, our core conclusions still hold that DeepWaveRL+TTO remains high competitive with UPFWI on FFA and outperforms UPFWI on CFA in terms of both MAE and RMSE.

We further provide the quantitative results of UPFWI with TTO in Table 12. While we provide these results for illustration, we respectfully emphasize that comparing DeepWaveRL+TTO against UPFWI+TTO bridges two fundamentally different problem settings. Optimizing UPFWI per-sample at test time explicitly requires backpropagating exact, dense gradients through a differentiable forward operator (effectively reverting to classical, gradient-based iterative FWI). In contrast, the foundational premise of DeepWaveRL is to operate in the single scalar reward environment without accessing gradients from the forward operator.

Table 11: Comparison of different models transferred from CVA.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM ↑</td></tr><tr><td rowspan=3 colspan=1>FFA</td><td rowspan=1 colspan=1>InversionNet</td><td rowspan=1 colspan=1>0.0087</td><td rowspan=1 colspan=1>0.0252</td><td rowspan=1 colspan=1>0.9893</td></tr><tr><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0200</td><td rowspan=1 colspan=1>0.0429</td><td rowspan=1 colspan=1>0.9738</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL+TTO</td><td rowspan=1 colspan=1>0.0268</td><td rowspan=1 colspan=1>0.0476</td><td rowspan=1 colspan=1>0.9146</td></tr><tr><td rowspan=3 colspan=1>CFA</td><td rowspan=1 colspan=1>InversionNet</td><td rowspan=1 colspan=1>0.0155</td><td rowspan=1 colspan=1>0.0467</td><td rowspan=1 colspan=1>0.9731</td></tr><tr><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0300</td><td rowspan=1 colspan=1>0.0895</td><td rowspan=1 colspan=1>0.9514</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL+TTO</td><td rowspan=1 colspan=1>0.0278</td><td rowspan=1 colspan=1>0.0502</td><td rowspan=1 colspan=1>0.9323</td></tr></table>

Table 12: Comparison of UPFWI with and without TTO (and with transfer learning from CVA) and DeepWaveRL+TTO.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM ↑</td></tr><tr><td rowspan=3 colspan=1>CVA</td><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0739</td><td rowspan=1 colspan=1>0.1369</td><td rowspan=1 colspan=1>0.7420</td></tr><tr><td rowspan=1 colspan=1>UPFWI+TTO</td><td rowspan=1 colspan=1>0.0666</td><td rowspan=1 colspan=1>0.1243</td><td rowspan=1 colspan=1>0.7634</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL+TTO</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.1012</td><td rowspan=1 colspan=1>0.8601</td></tr><tr><td rowspan=4 colspan=1>FFA</td><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0214</td><td rowspan=1 colspan=1>0.0484</td><td rowspan=1 colspan=1>0.9676</td></tr><tr><td rowspan=1 colspan=1>UPFWI+TTO</td><td rowspan=1 colspan=1>0.0176</td><td rowspan=1 colspan=1>0.0364</td><td rowspan=1 colspan=1>0.9778</td></tr><tr><td rowspan=1 colspan=1>UPFWI+TTO (CVA init.)</td><td rowspan=1 colspan=1>0.0167</td><td rowspan=1 colspan=1>0.0331</td><td rowspan=1 colspan=1>0.0.9800</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL+TTO</td><td rowspan=1 colspan=1>0.0268</td><td rowspan=1 colspan=1>0.0476</td><td rowspan=1 colspan=1>0.9146</td></tr><tr><td rowspan=4 colspan=1>CFA</td><td rowspan=1 colspan=1>UPFWI</td><td rowspan=1 colspan=1>0.0280</td><td rowspan=1 colspan=1>0.0641</td><td rowspan=1 colspan=1>0.9492</td></tr><tr><td rowspan=1 colspan=1>UPFWI+TTO</td><td rowspan=1 colspan=1>0.0231</td><td rowspan=1 colspan=1>0.0494</td><td rowspan=1 colspan=1>0.9639</td></tr><tr><td rowspan=1 colspan=1>UPFWI+TTO (CVA init.)</td><td rowspan=1 colspan=1>0.0278</td><td rowspan=1 colspan=1>0.0859</td><td rowspan=1 colspan=1>0.9563</td></tr><tr><td rowspan=1 colspan=1>DeepWaveRL+TTO</td><td rowspan=1 colspan=1>0.0278</td><td rowspan=1 colspan=1>0.0502</td><td rowspan=1 colspan=1>0.9323</td></tr></table>

## F Robustness Test

We additionally validate the robustness of DeepWaveRL on CVA by adding different levels of gaussian noise to seismic data. The quantitative results listed in Table 13 indicate that our DeepWaveRL is robust to a certain level of noise. The visualization results are shown in Figure 8

Table 13: Performance of DeepWaveRL+TTO tested on seismic data with different noise levels.
<table><tr><td rowspan=1 colspan=1>σ2(×10−3)</td><td rowspan=1 colspan=1>PSNR</td><td rowspan=1 colspan=1>MAE↓</td><td rowspan=1 colspan=1>RMSE↓</td><td rowspan=1 colspan=1>SSIM ↑</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.0527</td><td rowspan=1 colspan=1>0.1012</td><td rowspan=1 colspan=1>0.8601</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>30.00</td><td rowspan=1 colspan=1>0.0593</td><td rowspan=1 colspan=1>0.1104</td><td rowspan=1 colspan=1>0.8446</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>23.01</td><td rowspan=1 colspan=1>0.0907</td><td rowspan=1 colspan=1>0.1504</td><td rowspan=1 colspan=1>0.7931</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>20.00</td><td rowspan=1 colspan=1>0.1358</td><td rowspan=1 colspan=1>0.2090</td><td rowspan=1 colspan=1>0.7418</td></tr></table>

![](images/75478dac37198ca992ef3d42df40da0034cf4bfbe80a406a9b1cae20225bef7c.jpg)  
Figure 8: Comparison of DeepWaveRL+TTO tested on seismic data with different noise levels.

## G Visualizations

## G.1 More Visualization Results on CVA

![](images/71f319b14a704ac67734b26e2243ad1e0b13541cd3d7d0caa3153fec55e6fd1e.jpg)  
Figure 9: Illustration of ground truth and inversion results of different methods on CVA.

## G.2 More Visualization Results on FFA and CFA

![](images/0e6e3e5b7052cc735e5ac21a7e176e49b198429861a0cbda475fcb2b14f602fe.jpg)  
Figure 10: Illustration of ground truth and inversion results of different methods on FFA and CFA.

## G.3 Visualization Results of DeepWaveRL with Various Numbers of Bins

![](images/5dcab424cbaee22beb1cd5b184a3f0f397d5a1aa38a0ff34c2da796f8c677a88.jpg)  
Figure 11: Comparison of DeepWaveRL+TTO with different numbers of bins on CVA.

## G.4 Visualization Results of DeepWaveRL without Logarithm Transformation

![](images/6db8489be72e6eccd63c7a7d868fd7585926091402d85aa51c7bbcae061fa7bd.jpg)  
Figure 12: Illustration of ground truth and inversion results of DeepWaveRL with and without the sign-preserving logarithm transformation

## G.5 Visualization Results of DeepWaveRL with Various Group Sizes

![](images/5dabeac6040d8b332280e6f8cc33956bd2b451d660d64f93be2327068954b714.jpg)  
Figure 13: Comparison of DeepWaveRL with different group sizes on CVA.

## G.6 Visualization Results of DeepWaveRL without Refining the Prior

![](images/ce60c87009514fe7dfa14db43e4fd36f8831a65438306290b5e67ab5d52aac75.jpg)  
Figure 14: Comparison of DeepWaveRL+TTO with and without refining the prior on the training set of FFA (top two rows) and CFA (bottom two rows).

## G.7 Failure Cases

![](images/e7c2c0dd70bdf10c83a4b585b79ba06f1cf7e63e529ca2d91131c2b617060193.jpg)

![](images/16f3dcde1c9e2784773ada3b04242e9f644ca1dc57c9ca04dc15cd8e6fb08359.jpg)  
Figure 15: Examples of failure cases where the predictions collapse to similar patterns.

## G.8 Examples of Unstable Training Results

![](images/b83d9d61addaf1bd62f178e6f45181a4cb2f763bb8f02f7b0922817d9ee3c890.jpg)

![](images/41d93e4b06164067dd18302ee9924aa8a04431a6d0826f8ed6a5e3bba7ff64c5.jpg)  
Figure 16: Examples of predicted velocity maps generated by the model that has experienced unstable training.