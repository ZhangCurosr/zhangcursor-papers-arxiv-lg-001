# Investigating Behavior Cloning from Few Demonstrations for Autonomous Driving based on Bird’s-Eye View in Simulated Cities<sup>⋆</sup>

Eric Aislan Antonelo<sup>1</sup>, Gustavo Claudio Karl Couto<sup>1</sup>, Christian Möller<sup>2</sup>, and Pedro Henrique Fernandes<sup>1</sup>

<sup>1</sup> Automation and Systems Engineering Department, Federal University of Santa Catarina, Florianópolis, Brazil

eric.antonelo@ufsc.br, gustavo.karl.couto@posgrad.ufsc.br <sup>2</sup> Faculty of Science and Engineering, Information Technology Åbo Akademi University, Finland christian.moller@abo.fi

Abstract. This paper investigates the use of Behavior Cloning (BC) for autonomous driving from a bird’s-eye view (BEV) perspective in simulated urban environments. BC uses supervised learning to mimic expert driving behaviors. Previous works have applied BC in the CARLA simulator but did not fully address the challenges of trafic light compliance. Our approach enhances BC by integrating a kernel density estimator to adjust training sample weights based on action density, thereby improving the learning of rare but critical actions such as stopping at red lights and accelerating at green lights, specially in scenarios of scarce number of expert demonstrations. Using BEV inputs, which provide an abstract top-down view of the driving environment, our method simplifies the policy learning process. The trained convolutional neural network (CNN) outputs steering and acceleration actions based on these BEV inputs and additional state variables. Experimental results in the CARLA simulator demonstrate that our weighted BC method significantly improves driving performance, achieving higher route completion compared to standard BC. This weighted approach proved to be crucial in learning correct driving behaviors, particularly in test environments not encountered during training, highlighting its potential for enhancing autonomous vehicle navigation.

Keywords: Autonomous vehicles · Expert demonstrations · Imitation learning · kernel density estimation · CARLA Simulator.

## 1 Introduction

Autonomous driving has traditionally been implemented using separate modules for perception, planning, and control, organized in a sequential pipeline [13,21,15,16]. However, learning-based approaches have been gaining popularity as they attempt to address the complexities of autonomous driving in various scenarios, whether in simulation or real-world environments.

Most of these learning-based approaches rely on Behavior Cloning (BC), which employs supervised learning on a collection of expert demonstrations gathered ofline [3,19,2,6,7]. For example, a human driver generates a set of input (observations) and corresponding desired output pairs.

Codevilla [6,7] employed BC for autonomous driving in the CARLA simulator. He collected a large dataset of human driving and augmented it using image processing techniques to train end-to-end policies conditioned on the desired route. Further improvements in BC were introduced in [4], where a large deep ResNet network was used for feature extraction. This approach combined data from cameras, LiDAR, and radar to create comprehensive feature maps. The work also incorporated a probabilistic motion planner to account for uncertainties in the trajectory.

Couto [11] has shown results on behavior cloning on the CARLA simulator, but the agent did not have to respect trafic lights. In this work, we investigate an approach where the agent is able to navigate the city while respecting trafic lights, i.e., breaking when the light is red, and crossing the trafic light when it is green. Only relying on regular BC to train an agent is not enough in a scenario with limited amount of training routes and where the agent needs to break at the red light and accelerate from stillness when light turns to green. To solve that, we employ a kernel density estimator [18] to estimate the actions density from the expert dataset. Then, the training samples are weighted in the BC loss function individually by setting the weight equal to the inverse of the actions’ density. In this way, we can account for the underrepresented actions in the expert set, i.e., those happening at trafic light transitions. This weighting is needed because we are considering only eight routes as training data.

Instead of learning directly from raw camera pixels as in [11,12], the agent in this work has structured image inputs that are more abstract and meaningful to the navigation task. To decouple feature representation from policy learning [14,8] has its advantages as potentially easing up the learning task. In our context, this decoupling is accomplished by using a mid-level representation of the vehicle’s raw camera images. For example, Bird’s-Eye View (BEV) representations of the road ahead have been used as mid-level inputs for trajectory generation and motor control networks in [2] and [20], respectively. In BEV representation, the camera scene is projected onto a top-down view, segmenting areas of interest, like vehicle lanes and sidewalks, into diferent colors [21]. By using BEV as input for the control policy, the learning process can focus on the navigation problem, leveraging a more abstract and simplified view of the road ahead. The policy here is represented by convolutional neural networks, mapping the BEV image to control actuators such as steering and acceleration.

Experimental results in the CARLA simulator demonstrate that our weighted BC method significantly improves driving performance, achieving higher route completion compared to standard BC. This weighted approach proved to be essential in learning correct driving behaviors, particularly in test environments not encountered during training.

In Section 2, we present the bird’s-eye view representation employed as input to the convolutional neural network, the behavior cloning loss for the agent, and the kernel density estimation used to generate the weights per sample for the modified loss function. The agent architecture is presented in Section 3. Finally, the experiments and conclusions are shown in Sections 4 and 5.

## 2 Methods

## 2.1 Bird’s-Eye View - BEV representation

The bird’s-eye view of a vehicle represents its position and movement in a topdown coordinate system [2]. The vehicle’s location, heading, and speed are represented by $p _ { t } , \theta _ { t } ,$ and $s _ { t }$ respectively. The top-down view is defined so that the agent’s starting position is always at a fixed point within an image (the center of it). Furthermore, it is represented by a set of images of size $W \times H$ pixels, at a ground sampling resolution of $\phi$ meters/pixel. The BEV of the environment moves as the vehicle moves, allowing the agent to see a fixed range of meters in front of it. For instance, the BEV representation for the vehicle whose three frontal cameras are shown in Fig. 1(a) is given in Fig. 1(b), where the desired route, drivable area and lane boundaries are shown as three BEV channels. Besides, other three BEV channels for the trafic lights representing the state of the trafic lights (green or red) at diferent timesteps are shown in Fig. 1(c). Fig. 1(d) shows the integrated six-channel BEV image used as input to the agent.

## 2.2 Behavior Cloning (BC)

Behavior cloning is a supervised learning approach used in imitation learning, where the goal is to learn a policy that mimics the behavior of an expert. In this work, we want to learn a policy for an autonomous vehicle that outputs two actions (acceleration, steering) based on a Bird-Eye-View (BEV) representation of the environment and additional scalar state variables as input. For this reason we need to train a CNN to output these two continuous actions based on input from the BEV and the state variables.

The loss function for behavior cloning is typically based on the diference between the actions predicted by the learned policy and the actions taken by the expert in the training data. Let: $\pi _ { \boldsymbol { \theta } } ( s )$ be the policy parameterized by $\theta ,$ which maps a state s to an action $a ;$ and $\mathcal { D } = \{ ( s _ { i } , a _ { i } ) \} _ { i = 1 } ^ { N }$ be the dataset of N state-action pairs collected from the expert, where $s _ { i }$ is a state and $a _ { i }$ is the corresponding action taken by the expert.

Although the behavior cloning loss function can be defined as the mean squared error (MSE) between the predicted actions and the expert actions (or the

![](images/b5f8df6ae8e5b9e69d85e265aa2c3704d4582e35afdb9928471456aaf80bd8e4.jpg)

![](images/52bae49e85c2557f7cad0bcdb342af8047db23024b5a7faccce39e62c86ee26b.jpg)

![](images/34eb949325d4ab862fceff546b308356eb8b466dc1d1501634de47b08049df52.jpg)

![](images/f2c577118dd3fb76354c47869417e7d47d1761b5de65b4fc5cf01455f1ece16b.jpg)  
(a) Frontal Cameras

![](images/653d9d50a2210eac90d56556f80b841a9e5e18ec13e3443096184cec4aba76f6.jpg)

![](images/578db82f1407c5424aa60adbfa6e303af08408c77e1fdf7fcea9a514061058c0.jpg)  
(b) Bird’s-Eye View (BEV) channels: route, drivable area, and lane boundaries

![](images/242c2ae2c80dffc1f8c5c5ce98a7fc6781273252138da11872d62cb729d1a734.jpg)

![](images/f9d82ac8c44dcaeb571f2535ba716f7da268a33ba2421bfae9e42b4b66b94bb5.jpg)

![](images/c6caab43fc6d259d4c9a8e7e354b5c5469f845940dfbec5a7980c180b4e7a6a2.jpg)  
(c) BEV’s trafic lights channels at timesteps (-1, -9, -16)

![](images/980476462ad7a8edeb7b5cf63180bf4e52e234ed17dfa35f107e8c837554a354.jpg)

![](images/9f3450e9896dc37096044c80a9912efc6290c3504477018fc4e3e674edee9ecc.jpg)  
(d) final BEV image

Fig. 1. a) Images from three frontal cameras positioned on the left, center, and right side of the vehicle. These images were taken after the initial interactions of the agent in the CARLA simulation environment. Each camera captures a 256×144 RGB image. The corresponding bird’s-eye view channels that our agent uses are shown in (b), (c) and (d), computed at the same moment as shown in (a). b) Three BEV 192×192 channels that represent, from left to right, the desired route, drivable area, and lane boundaries. c) BEV’s trafic lights channels at timesteps (-1, -9, -16), i.e., representing the past states of the trafic light channel (grey color is red light of; white color means red ligh on). d) The final image displays all three channels combined in diferent colors.

cross-entropy loss if the actions are discrete), in this work, we model a stochastic policy and, for that reason, the policy is represented by a Beta distribution. The Beta distribution is suitable for modeling variables that are bounded within an interval, such as acceleration and steering actions within $[ - 1 , 1 ]$

Given the actions $a _ { i } = ( a _ { i } ^ { \mathrm { a c c } } , a _ { i } ^ { \mathrm { s t e e r } } )$ and the policy $\pi _ { \theta } ( s _ { i } ) = ( \alpha _ { i } ^ { \mathrm { a c c } } , \beta _ { i } ^ { \mathrm { a c c } } , \alpha _ { i } ^ { \mathrm { s t e e r } }$ $\beta _ { i } ^ { \mathrm { s t e e r } } )$ , where α and $\beta$ are the shape parameters of the Beta distribution, the loss function will be based on the negative log-likelihood of the Beta distribution.

The probability density function of the Beta distribution [10] is:

$$
f ( x ; \alpha , \beta ) = { \frac { x ^ { \alpha - 1 } ( 1 - x ) ^ { \beta - 1 } } { B ( \alpha , \beta ) } } ,
$$

where $B ( \alpha , \beta )$ is the Beta function, a normalization constant to ensure that the total probability is 1. To transform actions in $[ - 1 , 1 ]$ to [0, 1], we use:

$$
a _ { i } ^ { \prime } = { \frac { a _ { i } + 1 } { 2 } }
$$

The negative log-likelihood loss for the Beta distribution can then be written as:

$$
\mathcal { L } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \log f \left( \frac { a _ { i } ^ { \mathrm { a c c } } + 1 } { 2 } ; \alpha _ { i } ^ { \mathrm { a c c } } , \beta _ { i } ^ { \mathrm { a c c } } \right) + \log f \left( \frac { a _ { i } ^ { \mathrm { s t e e r } } + 1 } { 2 } ; \alpha _ { i } ^ { \mathrm { s t e e r } } , \beta _ { i } ^ { \mathrm { s t e e r } } \right) \right] .\tag{1}
$$

This loss function accounts for the probabilistic nature of the Beta distribution, ensuring that the model learns to predict parameters $( \alpha , \beta )$ for the distribution of actions rather than the actions directly. This approach allows for modeling uncertainty and variability in the predicted actions.

## 2.3 Kernel Density Estimation

A Kernel Density Estimator (KDE) is a non-parametric way to estimate the probability density function (PDF) of a random variable. For each data point in the sample, the kernel function is centered at that point. The KDE is then computed by summing up all these kernel functions and normalizing by the number of data points. Mathematically, the KDE for a point x is given by:

$$
\hat { g } ( x ) = \frac { 1 } { N h } \sum _ { j = 1 } ^ { N } K \left( \frac { x - x _ { j } } { h } \right) ,\tag{2}
$$

where ${ \hat { g } } ( x )$ is the estimated density at point $x ,$ N is the number of data points, $h$ is the bandwidth, $x _ { j }$ are the data points, and K is the kernel function.

The Gaussian kernel, often used in KDE, is a type of kernel function that assigns weights to data points based on their distance from the point where the density is being estimated. The Gaussian kernel is defined by the standard normal distribution, and it ensures that closer points have a higher influence on the density estimate than points further away. The mathematical form of the Gaussian kernel $K$ is:

$$
K ( u ) = \frac { 1 } { \sqrt { 2 \pi } } e ^ { - \frac { u ^ { 2 } } { 2 } } ,
$$

where $u$ is the standardized distance between the point where the density is being estimated and the data point, scaled by the bandwidth h.

In this work, the KDE is used to weigh the training samples in the BC loss in (1). By using the inverse of the kernel density estimation of the actions as weights, we give more importance to less frequent actions. This approach can help the model learn better on underrepresented parts of the action space.

The resulting weighted BC loss function can be written as:

$$
\mathcal { L } ( \theta ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { i } \left[ \log f \left( \frac { a _ { i } ^ { \mathrm { a c } } + 1 } { 2 } ; \alpha _ { i } ^ { \mathrm { a c } } , \beta _ { i } ^ { \mathrm { a c } } \right) + \log f \left( \frac { a _ { i } ^ { \mathrm { s t e e r } } + 1 } { 2 } ; \alpha _ { i } ^ { \mathrm { s t e e r } } , \beta _ { i } ^ { \mathrm { s t e e r } } \right) \right] .\tag{3}
$$

where: $\begin{array} { r } { w _ { i } = \frac { 1 } { \hat { g } ( a _ { i } ) } ; } \end{array}$ and ${ \hat { g } } ( a _ { i } )$ is the kernel density estimation evaluated at the action $a _ { i }$ . The KDE is estimated for the actions in the training set. Then the weights $w _ { i }$ are calculated as the inverse of the KDE for each action.

This approach ensures that the loss function accounts for the density of actions in the training set, giving higher importance to less frequent actions.

## 3 Agent

## 3.1 Input representation

The input s to the agent’s policy is a six-channel 192x192 image produced by a module in CARLA simulator that has access to the city map. This image represents the mid-level bird’s-eye view of the vehicle’s current position. In this work, the BEV contains three channels, the road, the route and the lane, and also three more channels representing the history of the trafic lights at time steps -16, -9 and -1.

Additionally, the agent receives the current speed of the vehicle and the last values of the policy actuators (previous acceleration and steering). These additional inputs are incorporated further down in the network, specifically in the first fully connected layer.

## 3.2 Output Representation

The vehicle in CARLA has three actuators: steering [−1, 1], throttle $[ 0 , 1 ]$ , and brake [0, 1]. Our agent’s action space is $\mathbf { a } \in [ - 1 , 1 ] ^ { 2 }$ , where the two components correspond to steering and acceleration. Braking is implied when acceleration is negative, ensuring the agent cannot brake and accelerate simultaneously [17].

Instead of using the Gaussian distribution for the policy’s actions, we employ the Beta distribution $B ( \alpha , \beta )$ . The Beta distribution’s bounded support is suitable for real-world applications like autonomous driving, where actions are naturally constrained [17]. This approach avoids the need for clipping or squashing inputs to enforce constraints, allowing explicit computation of the policy loss $\mathcal { L } _ { P }$

The Beta distribution also handles extreme driving scenarios, such as sharp turns and sudden braking, efectively. The parameters α and $\beta ,$ which are outputs of the policy neural network π<sub>θ</sub>, control the shape of the distribution, enabling the policy to adapt to these critical behaviors.

## 3.3 Network architecture

In this work, we train a network architecture, shown in Fig. 2, to learn to control an autonomous vehicle, using steering and acceleration actuators, based on the current BEV and some state variables as inputs. The Convolutionary Neural Network (CNN) maps the BEV into an embedding vector of size 256. Similarly, the scalar state variables are embedded into a vector of size 256 by a standard MLP (multi-layer perceptron layer). Together they form the current representation of the environment the agent faces. This concatenated embedding vector is processed by another MLP to map to the desired action.

![](images/43d9734fb0fa8bd476eab69c528db6f3653756f7353b7d1989286e8ed23f390c.jpg)  
Fig. 2. Network architecture. The BEV (state) input is processed by a CNN (MLP) to create an image (state) encoding. Both image and state encodings are concatenated and fed into a final MLP that outputs the actions.  
Fig. 3 shows the detailed network architecture.

## 4 Experiments

In this section, the proposed BC with weigthed loss function is compared with the standard BC approach for autonomous vehicle navigation tasks in urban scenarios using the CARLA simulator.

![](images/3bf10ee279c921bf5823511547a96bf2da61d411fb175bd572e6fb953f5a076a.jpg)  
Fig. 3. Detailed network architecture. The BEV input consists of 6 channels of 192x192 resolution, while the State input correponds to the last executed actions, and the current speed measurements. The final policy output corresponds to the two parameters α and β of the Beta distribution for both acceleration and steering (thus, the two heads).

## 4.1 Dataset Generation

The environment and trajectories are sourced from the CARLA Leaderboard evaluation platform [1]. Specifically, the town01 environment from this platform, along with ten predefined trajectories, is used to create the expert dataset.

The expert dataset is generated using a deterministic agent that navigates with a dense point trajectory and a classic PID controller [5]. The dense point trajectory ofers numerous points at a high resolution, while a sparse point trajectory has significantly fewer points, giving the agent only a general direction. Therefore, the dense point trajectory is employed to generate the expert’s training data, whereas the sparse point trajectory is used by the agent for broader guidance. Thus, the agent has no access to the dense point trajectory. Besides, the expert breaks at the detection of red trafic lights and accelerates when the light turns back to green.

In Fig. 4, one of the 10 routes executed by the expert to create the labeled training set of demonstrations is illustrated. The line that starts in yellow and ends in red represents the desired trajectory (not visible to the agent as it is). The sparse trajectory appears as yellow dots, generated every 50 meters or when the vehicle is about to change its movement (e.g., from straight to turn and vice versa).

The ten trajectories in the expert set were recorded at a rate of 10 hertz, producing 10 observation-action pairs per second. For the shortest route of 1480 samples (average route of 2129 samples), this equates to 2.5 minutes (3.5 minutes) of simulated driving. Altogether, the ten trajectories resulted in 21,287 labeled samples (30 GB of uncompressed data). The entire set corresponds to approximately 36 minutes or 8 km of driving.

## 4.2 Training and Inference

The network shown in Fig. 3 is trained with 8 routes from the expert dataset of town01 environment, where the agent must also respect trafic lights. The first agent (Weighted BC) weighs the samples according to the density of the actions using (3) as loss function, whereas the second agent (BC) considers all actions with same weight, i.e., it removes the higher importance of less frequent actions, using (1) as the conventional BC loss function.

## 4.3 Evaluation

Although training is ofline in town01 environment, we still evaluate both BC and weighted BC agents as their model’s parameters are updated. Every eight training epochs, both agents are evaluated using all ten routes from the same training town (Fig. 5). As four random runs are executed for each agent, the mean and standard deviation of the number of incomplete routes is plot in Fig. 5. One incomplete route means that the vehicle committed an infraction before completing the route. As the total number of routes is ten for town01, both BC and Weighted BC start at 10, the minimum performance. As training evolves, both agents improve the performance by committing less infractions, such as going out of the correct lane or passing a red light, but the Weighted BC agent attains the minimum error faster, i.e., it can complete all routes without any infractions and respecting trafic lights.

![](images/4e97375e5268b87a867ce54caa167e1c180f578a408b9b7d8dc0384176bba5ce.jpg)  
(a) town01

![](images/289f053fff5719626f07764669dc2755c569c7b089ed3705db7fb4cc94fb2ff1.jpg)  
(b) town02  
Fig. 4. (a) The town01 environment of the agent includes one of the routes used by the expert to collect data. This highlighted path measures 740 meters, featuring 20 points in the sparse trajectory (indicated by yellow dots) and 762 points in the dense point trajectory (not shown). (b) The town02 is used to evaluate the trained agent in a new environment.

To understand what the action’s density given by KDE and, consequently, the weights in the BC loss function, look like for the first route in town01 (shown in Fig. 4), we record both actuators’s values, acceleration and steering, from our expert as it follows the route as well as the corresponding logarithm of the action’s density for each timestep of the route. These values are plot in Fig. 6, where we can see that diferent actions will produce diferent weights in the loss function since some of them are less frequent than others in the expert dataset. The highest action density corresponds to going straight with constant acceleration and zero steering, for instance, between timesteps 470 and 900. The least frequent action happens after a trafic light turns from red to green, which corresponds to a higher, but shorter, acceleration period.

The driving performance for both agents as well as the expert is presented in Table 1. The understanding of the $\uparrow \mathrm { o r } \downarrow$ in the table is as follows. For instance, the lower (↓) the Red Light Infraction, which is #/Km or the number of infractions per kilometer, the better. In the same vein, the higher the Infraction Penalty (or the Driving score), the better is the performance. The expert has maximum driving score, being able to complete all routes with no red light infraction in both town01 and town02. The number of crossed trafic lights is just a non-performance metric, just counting the times that the vehicle cross them. An equivalent result is achieved by Weighted BC in town01, the training environment, but not by regular BC, which presents a mean driving score of 81. In town02, a test environment, Weighted BC performs significantly better than BC. Videos with the agent controlling the vehicle in CARLA are available at https://sites.google.com/view/weighted-bc, comparing both BC and Weighted BC agents.

![](images/33dbe01463bd0500454f37a01458c0ff3c5a1de1b16b2fc259fed3fb46ddf32c.jpg)  
Fig. 5. Evaluation of both BC policies (weighted and regular) throughout its learning epochs. At every eight training epochs, both BC and Weighted BC agents are evaluated in all ten routes of town01. The vertical axis presents the number of incomplete routes. Four runs were executed for each agent, with mean and standard deviation shown by the solid line and the shaded area, respectively. In the beginning, both agents commit infractions in all routes and, thus, the number of incomplete routes is ten since one infraction ends the route uncompleted. The Weighted BC agent is able to achieve zero infractions by the end of training, unlike the BC agent.

## 5 Conclusion

In this work, we proposed a weighted behavior cloning approach that improves agent learning by giving higher weights to underrepresented actions in the expert training set of driving behaviors. The application consisted of route navigation in two simulated cities in the CARLA realistic simulator for autonomous vehicles, while respecting trafic lights. As the action space of the expert is not uniformly visited, and the expert training dataset is somewhat limited (only 10 routes in town01 ), a Kernel Density Estimator was employed to infer the action density which, in turn, was converted to a weight per observation-action pair. In our investigated scenario, the weighted loss function for BC proved to be crucial to learn the correct behavior, showing significantly better driving performance than regular BC, specially for the test city town02 not used during training. Future work include extensions to cities with pedestrians and other vehicles and more advanced behavior cloning approaches such as implicit behavioral cloning [9] to account for multimodal action distributions.

![](images/bb4da9ade7da8f3c2ad4ca3f68d3dc15498209211655ce1322b2e52882397e41.jpg)

![](images/d0c669031e28de31dd1ff674903fb729e4671a29118020bbd5477576364cc619.jpg)  
Fig. 6. Logarithm of the density of actions, given by KDE, during navigation of a route in town1, shown in Fig. 4. The expert encounters three red trafic lights, during which it breaks (acceleration is −1), and subsequently accelerates as soon as it turns green. Two left turns and two right turns can be observed by examining the steer actuator of the expert.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. (Feb 2020), https://leaderboard.carla.org/

2. Bansal, M., Krizhevsky, A., Ogale, A.: Chaufeurnet: Learning to drive by imitating the best and synthesizing the worst (2018). https://doi.org/10.48550/ARXIV. 1812.03079, https://arxiv.org/abs/1812.03079

3. Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., Jackel, L.D., Monfort, M., Muller, U., Zhang, J., Zhang, X., Zhao, J., Zieba, K.: End to end learning for self-driving cars (2016). https://doi.org/10.48550/ARXIV. 1604.07316, https://arxiv.org/abs/1604.07316

4. Cai, P., Wang, S., Sun, Y., Liu, M.: Probabilistic end-to-end vehicle navigation in complex dynamic environments with multimodal sensor fusion. IEEE Robotics

Table 1. Driving performance and infraction analysis for diferent agents on Town01 and Town02.
<table><tr><td rowspan=2 colspan=1>Agent</td><td rowspan=2 colspan=1>Town</td><td rowspan=1 colspan=5>Metrics</td></tr><tr><td rowspan=1 colspan=1>DrivingScore $( \% , \uparrow ) ^ { \mathbf { a } }$ </td><td rowspan=1 colspan=1>RouteCom-pletion $( \% , \uparrow )$ </td><td rowspan=1 colspan=1>InfractionPenalty $( \% , \uparrow )$ </td><td rowspan=1 colspan=1>Red LightInfraction $( \# / \mathbf { K } \mathbf { m } , \downarrow )$ </td><td rowspan=1 colspan=1>Crossedtrafficlights (#)</td></tr><tr><td rowspan=1 colspan=1>Expert</td><td rowspan=1 colspan=1>town01town02</td><td rowspan=1 colspan=1> $\overline { { 1 0 0 \pm 0 } }$  $1 0 0 \pm 0$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 0 \pm 0 } }$  $1 0 0 \pm 0$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 0 \pm 0 } }$  $1 0 0 \pm 0$ </td><td rowspan=1 colspan=1> $\lvert 0 \pm 0 \rvert$  $0 \pm 0$ </td><td rowspan=1 colspan=1> $\boxed { 4 . 7 \pm 1 . 9 }$  $\left| 9 . 0 \pm 2 . 1 6 \right.$ </td></tr><tr><td rowspan=1 colspan=1>BC</td><td rowspan=1 colspan=1>town01town02</td><td rowspan=1 colspan=1> $\sqrt { 8 1 \pm 2 0 }$  $\lvert 3 4 \pm 1 8$ </td><td rowspan=1 colspan=1> $\overline { { 8 3 \pm 1 8 } }$  $5 3 \pm 1 8$ </td><td rowspan=1 colspan=1> $\overline { { 9 7 \pm 5 } }$  $\lvert 6 6 \pm 2 0$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 3 9 \pm 0 . 6 8 } }$  ${ \lvert { 3 . 0 8 \pm 2 . 4 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 . 7 8 \pm 1 . 0 3 } }$  $| 2 . 7 9 \pm 2 . 2 7$ </td></tr><tr><td rowspan=1 colspan=1>WeightedBC</td><td rowspan=1 colspan=1>town01town02</td><td rowspan=1 colspan=1> $\overline { { 1 0 0 \pm 0 } }$  $\lvert 9 8 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 0 \pm 0 } }$  $1 0 0 \pm 0$ </td><td rowspan=1 colspan=1> $\overline { { 1 0 0 \pm 0 } }$  $\lvert 9 8 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $\overline { { 0 \pm 0 } }$  $\lvert 0 \pm 0$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 7 \pm 1 . 9 } }$  $| 9 . 0 \pm 2 . 1 6$ </td></tr></table>

<sup>a</sup> The Driving Score, a metric from CARLA Leaderboard, is calculated by the product of Route Completion (the percentage of the route distance completed) and an Infraction Penalty which accounts for all violations committed. For instance, if an agent ran two red lights during a route and the penalty for running one red light is 0.7, then the final penalty would be $0 . 7 \times 0 . 7 = 0 . 4 9$

and Automation Letters 5(3), 4218–4224 (2020). https://doi.org/10.1109/LRA. 2020.2994027

5. Chen, D., Zhou, B., Koltun, V., Krähenbühl, P.: Learning by cheating (2019)

6. Codevilla, F., Müller, M., Dosovitskiy, A., López, A.M., Koltun, V.: End-toend driving via conditional imitation learning. 2018 IEEE International Conference on Robotics and Automation (ICRA) pp. 1–9 (2017), https://api. semanticscholar.org/CorpusID:3672301

7. Codevilla, F., Santana, E., Lopez, A., Gaidon, A.: Exploring the limitations of behavior cloning for autonomous driving. In: 2019 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9328–9337 (2019). https://doi.org/10. 1109/ICCV.2019.00942

8. Couto, G.C.K., Antonelo, E.A.: Hierarchical generative adversarial imitation learning with mid-level input generation for autonomous driving on urban environments. arXiv preprint arXiv:2302.04823 (2023)

9. Florence, P., Lynch, C., Zeng, A., Ramirez, O.A., Wahid, A., Downs, L., Wong, A., Lee, J., Mordatch, I., Tompson, J.: Implicit behavioral cloning. In: Conference on Robot Learning. pp. 158–168. PMLR (2022)

10. Gelman, A., Carlin, J.B., Stern, H.S., Rubin, D.B.: Bayesian data analysis. Chapman and Hall/CRC (1995)

11. Karl Couto, G.C., Antonelo, E.A.: Generative adversarial imitation learning for end-to-end autonomous driving on urban environments. In: 2021 IEEE Symposium Series on Computational Intelligence (SSCI). pp. 1–7 (2021). https://doi.org/ 10.1109/SSCI50451.2021.9660156

12. Liang, X., Wang, T., Yang, L., Xing, E.: Cirl: Controllable imitative reinforcement learning for vision-based self-driving. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) Computer Vision – ECCV 2018. pp. 604–620. Springer International Publishing, Cham (2018)

13. Montemerlo, M., Becker, J., Bhat, S., Dahlkamp, H., Dolgov, D., Ettinger, S., Haehnel, D., Hilden, T., Hofmann, G., Huhnke, B., Johnston, D., Klumpp, S.,

Langer, D., Levandowski, A., Levinson, J., Marcil, J., Orenstein, D., Paefgen, J., Penny, I., Thrun, S.: Junior: The stanford entry in the urban challenge. Journal of Field Robotics 25, 569 – 597 (09 2008). https://doi.org/10.1002/rob.20258

14. Ota, K., Jha, D.K., Kanezaki, A.: Training larger networks for deep reinforcement learning. arXiv preprint arXiv:2102.07920 (2021)

15. Paden, B., Cáp, M., Yong, S.Z., Yershov, D.S., Frazzoli, E.: A survey of motion planning and control techniques for self-driving urban vehicles. IEEE Transactions on Intelligent Vehicles 1, 33–55 (2016)

16. Parizotto, L.M.S., Antonelo, E.A.: Cone detection with convolutional neural networks for an autonomous formula student race car. In: 26th ABCM International Congress of Mechanical Engineering (COBEM 2021) (2021)

17. Petrazzini, I.G., Antonelo, E.A.: Proximal policy optimization with continuous bounded action space via the beta distribution. In: 2021 IEEE Symposium Series on Computational Intelligence (SSCI). pp. 1–8. IEEE (2021)

18. Silverman, B.W.: Density estimation for statistics and data analysis. Routledge (2018)

19. Xu, H., Gao, Y., Yu, F., Darrell, T.: End-to-end learning of driving models from large-scale video datasets (2016). https://doi.org/10.48550/ARXIV.1612.01079, https://arxiv.org/abs/1612.01079

20. Zhang, Z., Liniger, A., Dai, D., Yu, F., Van Gool, L.: End-to-end urban driving by imitating a reinforcement learning coach. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 15202–15212 (2021). https://doi. org/10.1109/ICCV48922.2021.01494

21. Zhu, H., Yuen, K.V., Mihaylova, L., Leung, H.: Overview of environment perception for intelligent vehicles. IEEE Transactions on Intelligent Transportation Systems 18(10), 2584–2601 (2017). https://doi.org/10.1109/TITS.2017.2658662