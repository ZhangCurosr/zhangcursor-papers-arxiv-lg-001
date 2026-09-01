# Mode Connectivity Beyond Classifiers: Evidence from Generative and Contrastive Models

Chengzheyi Yao1, Yongzhao Zhang1, Yongding Tian2

1University of Electronic Science and Technology of China 2Delft University of Technology

## Abstract

The loss landscape of Deep Neural Networks (DNNs) exhibits highly complex and non-convex properties. Recent studies have revealed the phenomenon of mode connectivity, demonstrating that independently trained network modes can be connected via a continuous low-loss path. However, existing mode connectivity research is predominantly confined to classifier-based models, leaving it an open question whether similar geometric properties exist in modern complex models. In this paper, we extend the boundaries of mode connectivity to generative and contrastive domains (specifically DDPM and NanoCLIP). Addressing the unique architecture of DDPM and CLIP, we propose an architecture-aware connection building algorithm. Extensive empirical results demonstrate for the first time that we successfully discover mode connectivity between independently trained DDPM and NanoCLIP modes. Our work provides a novel perspective for understanding the geometric properties of the loss landscapes in modern generative and contrastive models.

## 1 Introduction

The loss functions of Deep Neural Networks (DNNs) are highly complex and non-convex, and their geometric properties have long been a subject of extensive research. Early visualizations of loss functions (Foret et al. 2021; Li et al. 2018) supported the intuition that well-trained neural networks converge to isolated basins or local minima separated by high-loss barriers in the loss landscape. However, such visualizations are limited by their reliance on low-dimensional projections of the high-dimensional weight space. Further research suggests the basin model is incomplete.

Mode connectivity emerges by asking whether two independently trained neural networks can be connected by a continuous low-loss path in weight space. The endpoints of such a path are commonly referred to as modes, i.e., well-trained models that attain low loss. Establishing mode connectivity is significant because traditional loss landscape visualizations typically capture low-dimensional slices and therefore fail to reveal the high-dimensional low-loss regions. In contrast, mode connectivity provides a tool for uncovering deeper geometric and topological properties of the loss landscape beyond the isolated-basin picture.

The study of mode connectivity has both theoretical and empirical roots. Singular learning theory (Watanabe

2009) provided an algebraic-geometric perspective on this phenomenon: neural networks are singular statistical models, and the degeneracy of their Fisher information matrices (Fukumizu 1996) naturally gives rise to non-isolated parameter configurations and continuous low-loss subspaces. Empirical evidence for mode connectivity was later established. FGE (Garipov et al. 2018) showed that independently trained modes can be connected by simple curves such as polygonal chains or Bezier curves. AutoNEB (Draxler et al. 2018) iteratively bends linear interpolations into lowloss paths by optimizing intermediate pivots. More recently, LLPF (Tian et al. 2026) proposed a layer-wise algorithm, improving the reliability of connecting independently trained modes across a broader set of neural architectures.

Despite these advances, existing empirical evidence for mode connectivity remains largely confined to classifierbased settings, as shown in Table 1. Although these works cover multiple network architectures, the underlying learning paradigm is still relatively simple: unimodal supervised classification on small-scale CIFAR datasets. Whether mode connectivity exists in modern non-classifier architectures, such as generative and contrastive models, remains largely unexplored.

In this paper, we extend the scope of mode connectivity from classifiers to Denoising Diffusion Probabilistic Models (DDPM) (Ho, Jain, and Abbeel 2020) on Flowers102 (Nilsback and Zisserman 2008) and NanoCLIP (Ali-bey 2024) on Flickr30k (Young et al. 2014). These models introduce new challenges absent from classifiers: (1) DDPM uses a U-Net architecture with skip connections, while NanoCLIP relies on cross-modal alignment between image and text encoders. All previous methods (Tian et al. 2026; Adilova et al. 2024) are not compatible with these models; and (2) these models have higher-dimensional parameter spaces than the classifiers, resulting in larger parameter-space distances between two modes. It is difficult to identify an effective optimization direction towards low-loss regions.

Our success in these models is primarily attributed to two key designs in two-stage mode connectivity pipeline (Tian et al. 2026): (1) in Weighted Movement stage, we develop a dataflow-based layer moving strategy that determines the moving order from the computational dependencies of each architecture, building on the layer-wise methods; and (2) in Training Refinement stage, we first identify that the choice of optimizer is important for higher-dimensional models. Thus, we employ the Adam optimizer (Loshchilov and Hutter 2019) instead of Stochastic Gradient Descent (SGD), on which previous methods mainly rely.

Our main contributions are summarized as follows:

• To the best of our knowledge, we are the first to extend the scope of mode connectivity from conventional image classifiers to representative non-classifier models, including contrastive and generative models.

• We provide empirical evidence that the mode connectivity property holds for DDPM on the Flowers102 dataset and for NanoCLIP on the Flickr30K dataset.

## 2 Related Work

In this section, we present prior mode connectivity studies from three distinct paradigms.

## Theoretical Perspectives of Mode Connectivity

Singular learning theory (SLT) (Watanabe 2009) provides an important theoretical perspective for mode connectivity. In classical regular statistical models, different parameter values typically correspond to locally distinguishable predictive distributions, and the Fisher information matrix is nondegenerate around an optimum. Neural networks, however, are singular statistical models: the parameter-to-function map is generally non-injective due to overparameterization, permutation symmetries (Entezari et al. 2022), scaling invariances (Grigsby, Lindsey, and Rolnick 2023), and redundant representations (Fukumizu 1995). Consequently, the Fisher information matrix may become degenerate near well-trained solutions, and the set of parameters realizing the same or similar functions can form continuous singular structures rather than isolated points.

From the viewpoint of SLT, the existence of flat directions and low-loss subspaces is therefore a natural consequence of the algebraic-geometric structure of neural network parameterizations. This theoretical perspective indicates that trained modes may belong to extended low-loss regions in parameter space. However, SLT does not directly provide algorithms for connecting two given independently trained modes. As a result, empirical methods remain necessary for testing whether a low-loss path exists in neural networks.

## Empirical Studies of Mode Connectivity

The investigation of mode connectivity initially emerged from an empirical approach, AutoNEB (Draxler et al. 2018), which iteratively refines a linear path into a low-loss region by inserting and optimizing a sequence of intermediate pivots. It constructed continuous low-loss paths in architectures like ResNets (He et al. 2016) and DenseNets (Huang et al. 2017). Despite these advancements, previous works were unable to find a low-loss path for any arbitrary mode pairs. To address this limitation, subsequent work (Tian et al. 2026) introduced a Low-Loss Path Finding (LLPF) algorithm based on layer-wise connectivity. By moving the network layer-bylayer, their method successfully expanded to a wider range of architectures (Tan and Le 2019; Sandler et al. 2018; Radosavovic et al. 2020; Zhang et al. 2018; Yu et al. 2018; Hassani et al. 2021) as mentioned in Table 1.

<table><tr><td>Method</td><td>Model architectures</td><td>Datasets</td></tr><tr><td>FGE</td><td>VGG, ResNet, WideResNet</td><td>MNIST, CIFAR-10, CIFAR-100</td></tr><tr><td>AutoNEB</td><td>Basic CNN, ResNet, DenseNet</td><td>CIFAR-10, CIFAR-100</td></tr><tr><td>LLPF</td><td>All above and EfficientNet, MobileNet, CIFAR-100,</td><td>MNIST, CIFAR-10,</td></tr><tr><td></td><td>RegNet, ShuffleNet, DLA, CCT</td><td>ImageNet10</td></tr><tr><td>Ours</td><td>DDPM, NanoCLIP</td><td>Flowers102, Flickr30k</td></tr></table>

Table 1: Comparison of evaluated model architectures and datasets in prior mode connectivity studies.

However, all of these empirical studies share a common limitation: their explorations are strictly confined to classifier-based models. The mode connectivity of generative and contrastive models remains entirely unexplored, which is the primary focus of our work.

## Mode Connectivity and Linear Mode Connectivity

Although linear mode connectivity appears more direct compared to the nonlinear path, the midpoint of linear interpolation often encounters a severe loss barrier (Draxler et al. 2018). It is crucial to highlight the difference between mode connectivity and linear mode connectivity.

For mode connectivity, two independently trained modes correspond to distant and unaligned points. Finding a continuous path between them is already difficult in such a highdimensional space. Requiring the entire path to remain inside the low-loss region makes the problem even more restrictive. As a result, existing studies in this category are primarily driven by empirical methods and remain limited in number. Our work only investigates mode connectivity in independently trained modes.

Linear mode connectivity is commonly observed under two settings: models either share an early portion of their optimization trajectory or are aligned with respect to parameterspace symmetries. In the former setting, models branched from a common early-stage checkpoint can be connected by a low-loss linear path (Frankle et al. 2020; Fort et al. 2020). In the latter setting, permutation-based weight matching can transform independently trained modes into aligned parameterizations that admit nearly barrier-free linear interpolation (Entezari et al. 2022; Ainsworth, Hayase, and Srinivasa 2023; Tran et al. 2025). In a nutshell, linear mode connectivity typically needs shared early training trajectories or invasive weight permutations to align models, which can make it easier to find a connected path by linear interpolation.

## 3 Terminology and Definitions

To provide a unified notation for describing mode connectivity, we first define the key terms used throughout this paper.

![](images/8ed75afca357ea4aa3829bb8e8b4629810f38fc9d560bf9734bfbbaceab42877.jpg)  
Figure 1: Geometric illustration of architecture-aware connection building algorithm. The panel shows a slice of the loss function, where the endpoint modes are connected by a low-loss path. The variance sphere below zooms into a single iteration of the algorithm. Point labels follow the algorithm's notation. The origin is used as the center of the variance sphere, assuming the mean is approximately zero (Equation 4 and Equation 5).

Problem Definition. Mode connectivity can be understood as the problem of determining whether two independently trained modes can be connected by a continuous path in low-loss regions.

A mode refers to a well-trained parameter point that attains low loss. In this paper, we focus on independently trained modes, where different modes are obtained from different random initializations without shared early training checkpoints or weight permutation.

Given a neural network architecture $\mathcal { M } ,$ let $\theta \in \mathbb { R } ^ { D }$ denote its trainable parameters, where D is the total number of trainable parameters. For geometric discussions, we use $P$ to denote the point in weight space corresponding to $\theta .$ Each point $P \in \mathring { \mathbb { R } } ^ { D }$ is associated with a loss value $\bar { \mathcal { L } ( P , D ) }$ evaluated on dataset D. Given a loss threshold $L _ { \mathrm { t h r e s } }$ , we define the low-loss region in weight space as:

$$
S _ { L \le L _ { \mathrm { t h r e s } } } : = \{ P \in \mathbb { R } ^ { D } \mid { \mathcal { L } } ( P , { \mathcal { D } } ) \leq L _ { \mathrm { t h r e s } } \}\tag{1}
$$

Variance Sphere. Neural networks are composed of multiple layers, which decompose the full parameter point $P$ as $\dot { P } = \dot { [ } \dot { P } ^ { ( 1 ) } , P ^ { ( 2 ) } , \dots , P ^ { ( \mathrm { \hat { \cal K } } ) } ]$ , where $\mathbf { \bar { \Phi } } _ { P ^ { ( k ) } } ^ { * } \in \mathbb { R } ^ { \bar { d } _ { k } }$ denotes the parameter vector of the k-th layer. Since the parameters of each layer can be viewed as a distribution of real values, we use $\mathrm { M e a n } ( P ^ { ( k ) } )$ and $\mathrm { V a r } ( P ^ { ( k ) } )$ to denote the empirical mean and variance of the k-th layer, respectively.

Based on the layer-wise variance, we define the Variance Sphere as the set of layer parameters with a fixed variance value. For the k-th layer and a given variance v, the variance sphere is defined as:

$$
S _ { \mathrm { v a r } } ^ { ( k ) } ( v ) : = \{ P ^ { ( k ) } \in \mathbb { R } ^ { d _ { k } } \ | \ \mathrm { V a r } ( P ^ { ( k ) } ) = v \}\tag{2}
$$

For independently trained modes obtained under the same model architecture, their layer-wise mean and variance are expected to be close. This is because common initialization schemes for linear, convolutional, and transformer layers usually sample weights from zero-mean distributions. Their variance is determined by the layer size and computing method (Orr and Müller 1998; Glorot and Bengio 2010; He et al. 2015). Therefore, two independent initializations of the same architecture satisfy:

$$
\theta _ { 0 } , \theta _ { 0 } ^ { \prime } = \mathrm { I n i t } ( \mathcal { M } ) \Longrightarrow \mathrm { V a r } ( \theta _ { 0 } ) = \mathrm { V a r } ( \theta _ { 0 } ^ { \prime } )\tag{3}
$$

As shown by the results in Appendix A, Figure 7, independently trained modes trained with identical hyperparameters on the same dataset tend to lie on nearby variance spheres, while the layer-wise means of the trained parameters typically remain near zero. Thus, for the k-th layer, we adopt the approximation:

$$
\mathrm { M e a n } ( P ^ { ( k ) } ) \approx 0 \quad \mathrm { a n d } \quad \mathrm { V a r } ( \theta _ { n } ^ { k } ) \approx \mathrm { V a r } ( \theta _ { n } ^ { \prime k } )\tag{4}
$$

Under this approximation, the variance of a layer is approximately proportional to the squared Euclidean distance from the corresponding parameter point to the origin in $\mathbb { R } ^ { d _ { k } }$

$$
\left\| { \overrightarrow { O P ^ { ( k ) } } } \right\| ^ { 2 } \propto \operatorname { V a r } ( P ^ { ( k ) } )\tag{5}
$$

Therefore, fixing the variance of a layer approximately fixes its distance to the origin. This explains why $S _ { \mathrm { v a r } } ^ { ( k ) } ( v )$ can be interpreted geometrically as a high-dimensional sphere in the layer-wise parameter space.

## 4 Architecture-aware Connection Building Algorithm

This section presents the proposed architecture-aware connection building algorithm. We first introduce two-stage connection building framework of our algorithm and then detail two key designs in Weighted Movement and Training Refinement as shown in Figure 1.

## Mode-to-mode Connection Building Framework

The core objective of connection building is to iteratively transition from $P _ { 0 }$ toward $P _ { E }$ while ensuring all anchor points remain within the low-loss regions, as presented by the low-loss path in Figure 1. The corresponding procedure is formalized in Algorithm 1 and consists of two stages:

Weighted Movement $( P _ { i } \to M _ { 2 } )$ . We first move the active layers of $P _ { i }$ to obtain $M _ { 1 }$ toward the destination $P _ { E } ,$ controlled by $( \alpha , \beta , \gamma )$ and $\mathcal { U } _ { i }$ .Because averaging independently trained weights reduces their layer-wise variance, it can hinder subsequent training process (Tian et al. 2024). Variance correction rescales $M _ { 1 }$ to $M _ { 2 }$ , which lies on the reference variance sphere of $P _ { 0 }$ as shown in Figure 1.

Training Refinement $( M _ { 2 } \to P _ { i + 1 } )$ . Restoring the layerwise variance does not by itself guarantee that $M _ { 2 }$ lies in low-loss regions, so we then train $M _ { 2 }$ for several rounds, yielding $M _ { 3 }$ . The variance correction is applied once more to define the next anchor $P _ { i + 1 }$ . Repeating this procedure for $T _ { t o t a l }$ iterations generates a sequence from $P _ { 0 }$ to $P _ { T } \approx P _ { E }$ with consecutive points forming the desired low-loss path.

![](images/2532f16e2ddb052a6024b29b959ddd0e36205f60dc2e15adb81def234c282ac6.jpg)  
Figure 2: Dataflow-based layer moving strategies for DDPM and NanoCLIP. $L e f t { \mathrm { : } }$ DDPM layers are added to the moving sequence following a symmetrical head-to-tail schedule. Right: NanoCLIP layers are added through an alternating visual-textual schedule. The Layer Moving Direction indicates the order in which layer groups are introduced into layer moving sequence $\mathcal { U } _ { i } .$

Algorithm 1: Architecture-aware Connection Building Al  
gorithm   
Require: Endpoint modes $P _ { 0 }$ and $P _ { E } ,$ dataset $\mathcal { D } ,$ step-size hyper  
parameters $( \alpha , \beta , \gamma )$ , training hyperparameters ξ, layer moving   
sequence U, total iterations $\bar { T } _ { t o t a l } .$   
1: $\overset { \cdot } { P _ { i } } \overset { \cdot } { = } P _ { 0 } , \mathcal { P }$ append(Pi)   
2: for $i = 0$ to $T _ { t o t a l . } - 1$ do   
3: $s t e p = \alpha \left| \overrightarrow { P _ { i } P _ { E } } \right| + \beta \left| \widehat { P _ { 0 } P _ { E } } \right| + \gamma$   
4: $M _ { 1 } = P _ { i } + s t e p \cdot \Pi _ { \mathcal { U } _ { i } } \left( \overrightarrow { P _ { i } P _ { E } } \right)$   
5: M2 = VarianceCorrection $( \check { M } _ { 1 } , S _ { V a r } )$   
6: $M _ { 3 } = \mathrm { T r a i n } ( M _ { 2 } , \xi , \mathcal { D } ) .$   
7: $P _ { i + 1 } = \mathrm { V } _ { i }$ arianceCorrection $( M _ { 3 } , S _ { V a r } )$   
8: P.appena $l ( P _ { i + 1 } )$   
9: end for   
10: return $\mathcal { P }$   
11:   
12: Function VarianceCorrection $( Q , S _ { V a r } )$   
13: for $W = Q ^ { ( k ) }$ in Q do   
14: $\begin{array} { r } { \bar { W } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \dot { W } [ j ] } \end{array}$   
15: $\begin{array} { r } { \sigma _ { W } ^ { 2 }  \frac { 1 } { n } \overset {  } { \sum } _ { j = 1 } ^ { n } ( W [ j ] - \bar { W } ) ^ { 2 } } \end{array}$   
16: for $j = 1$ to len(W) do   
17: $\begin{array} { r } { W ^ { \prime } [ j ] = \bar { W } + \sqrt { \frac { v } { \sigma _ { W } ^ { 2 } } } \big ( W [ j ] - \bar { W } \big ) } \end{array}$   
18: end for   
19: $\overset { \triangledown } { Q ^ { \prime ( k ) } } = { W ^ { \prime } }$   
20: end for   
21: return $Q ^ { \prime }$

## Dataflow-based Layer Moving Strategy

The most critical hyperparameter of Weighted Movement is the layer moving sequence $\lambda _ { i } ,$ which decides the subset of layers needed to move at a stage. Moving all layers at once failed due to complex architectures of DDPM and NanoCLIP. To solve this, we design dataflow-based layer moving strategy that follows two rules: (1) layers should be processed in the direction of data flow according to model architectures; and (2) attention modules should be processed individually before proceeding to subsequent layers. Figure 2 contrasts the ordinary forward propagation directions with our moving strategy for DDPM and NanoCLIP.

Symmetrical Strategy for DDPM. For DDPM, the network typically adopts a U-Net topology characterized by symmetrical contracting and expanding paths linked via skip connections. Since corresponding downsampling and upsampling blocks handle features at identical spatial resolutions, moving them separately may disrupt the denoising trajectory. To address this, we introduce a symmetrical headto-tail strategy where corresponding layers from the encoder and decoder are added to $\bar { \mathcal { U } } _ { i }$ sequentially, progressing from the outermost layers to the innermost layers. By binding these structurally coupled layers into unified transformation units, the algorithm preserves the integrity of the high-to-low feature hierarchy, enabling stable mode connectivity across generative manifolds.

Alternating Strategy for NanoCLIP. The NanoCLIP architecture utilizes a dual-encoder paradigm where visual and textual features are projected into a shared embedding space via contrastive learning. To maintain the semantic alignment between modalities during path construction, we propose an alternating strategy. Specifically, two visual encoder blocks are added to the ${ \bar { \boldsymbol { u } } } _ { i } ,$ followed by a single layer of the text encoder, and this cross-modal coordination repeats sequentially across all available layers. This strategy ensures that the visual representation space and textual representation space deform symmetrically, preventing the collapse of contrastive alignment along the path.

## Adam-based Refinement for Distant Modes

The purpose of Training Refnement is to return the variancecorrected point $M _ { 2 }$ to the low-loss region after each weighted movement. In the classifier settings, the movements are typically short and localized. Under this condition, a SGD optimizer is often sufficient to identify a nearby descent direction and recover low loss. The DDPM and NanoCLIP models operate in substantially higher-dimensional parameter spaces than the classifier-based models. As a result, independently trained checkpoints exhibit larger parameter distances, including larger layer-wise separations. Empirically, we find that the single global learning-rate scaling of SGD becomes unreliable in this regime: SGD may reduce the loss for small endpoint separations, but frequently fails to restore low loss when the checkpoint separation is large.

![](images/56480738a6ee9cfd77cd4fede645d4187c605d25e7e0807a366ebc056ec891ba.jpg)  
Figure 3: Results of the architecture-aware connection building algorithm on DDPM (first column) and NanoCLIP (second column). The first row shows the layer-wise $L _ { 2 }$ distance from the current point $P _ { i }$ to the end mode $P _ { E }$ , and the legend of layers is omitted due to excessive quantity. The second and third rows respectively report the training and test metrics along the path. The curves show the smoothed trends, and the shaded regions indicate local fluctuation envelopes of the raw metrics. Due to the learning objectives of the models, DDPM is evaluated only with training and test losses, whereas NanoCLIP is evaluated without a test loss.

We address this limitation by implementing training refinement with Adam. Its first and second moment estimates provide coordinate-wise adaptive scaling, which is useful when different modules exhibit substantially different gradient magnitudes. The momentum state also accumulates a stable refinement direction across successive updates. During each refinement phase, Adam updates the parameters once the training loss falls below the threshold or the maximum number of refinement rounds is reached. The optimizer ablation in Figure 5 supports this design: replacing Adam with SGD while retaining all other components prevents the refinement step from maintaining a low-loss trajectory.

## 5 Experimental Evaluation

To evaluate the proposed architecture-aware connection building algorithm, we consider two representative models: DDPM on Flowers102 and NanoCLIP on Flickr30k. For each model, we independently train two modes using different random seeds and subsequently apply our method to construct a low-loss path between them. All experiments follow the architecture-aware layer-moving schedules and configurations described in Appendix C, Table 2.

Overall Performance. Figure 3 evaluates the validity of the constructed paths from three complementary perspectives: (1) for the first row, the layer-wise $L _ { 2 }$ distance between the current point $P _ { i }$ and the end mode $P _ { E }$ should decrease progressively and eventually approach the destination. Small deviations are acceptable because each stage only approximates the ideal movement; (2) for the second row, the training loss should remain within the low-loss threshold throughout the connection process; and (3) for the third row, the test evaluation metrics at the end of the path should converge to those of the starting mode $P _ { 0 } .$ indicating that the constructed path not only reaches the target parameter region but also recovers its generalization performance.

The first row of Figure 3 shows the evolution of the layerwise $L _ { 2 }$ distances. For both DDPM and NanoCLIP, the distances decrease stage by stage following the predefined layer moving strategy, demonstrating that the proposed algorithm consistently moves the current point toward the end mode. Although the final distances of several layers do not converge exactly to zero, they are reduced by at least one order of magnitude compared with their initial values. This behavior is expected because different layers contain substantially different numbers of parameters, leading to different distance scales. Consequently, the relative reduction from the initial distance more accurately reflects the convergence behavior.

![](images/005b6dd9a33c0c2ed885b43d1168a32572693eea6d5261f0f9caa894ac8da002.jpg)

![](images/ee2ffa3aede8ad823aa4b42494deaffc99ffdfcebb95939edfe689614a25a226.jpg)

![](images/16b37b2917c2dcf4177efecd6a50beac633f7c11aee595e82518565afba1b15b.jpg)

![](images/b04be3a558694d4a990b73c3615a9761473c4d5ca543a8dcbe15a88349a57537.jpg)  
Figure 4: Comparison of AutoNEB and our method along constructed paths on DDPM and NanoCLIP. From left to right, the four panels report DDPM training loss, DDPM test loss, NanoCLIP training loss, and NanoCLIP training accuracy, respectively. The horizontal axis denotes the normalized path position, where 0 and 1 correspond to the starting and end modes. The dots on the dashed line are the pivots optimized by AutoNEB. Between adjacent pivots, we also evaluate the metrics of interpolation points along each segment. For visual clarity, the densely sampled curves of our method are smoothed.

For DDPM, the denoising training loss remains below 0.04 throughout the entire connection process, while the testing loss eventually converges to approximately 0.04, which is even lower than the starting mode. For NanoCLIP, the contrastive training loss remains below 0.09 during optimization, while both the training accuracy and Recall@1 stay relatively stable. These results demonstrate that the proposed algorithm is capable of constructing low-loss paths while preserving their generation capability.

Because the procedure produces a discrete sequence of checkpoints, we further verify that the line segments between consecutive checkpoints do not introduce hidden loss barriers. As detailed in Appendix B, dense linear interpolation along the NanoCLIP trajectory reveals no observable barrier, providing empirical evidence that the path remains within the low-loss region.

Comparison Study. Among a limited number of mode connectivity methods, FGE, AutoNEB and LLPF are the relevant baselines for comparison. However, the released implementation of FGE contains an error, and its procedure does not necessarily yield a valid low-loss path (Tian et al. 2026). Our method is based on the layer-wise paradigm of LLPF, so the comparative results with LLPF can actually refer to the ablation results combined with forward order and SGD. We therefore select AutoNEB as the primary baseline. For a controlled comparison, we follow the original AutoNEB training configuration and adapt its pipeline to accommodate DDPM and NanoCLIP as shown in Appendix D, Table 3.

Figure 4 compares AutoNEB with our method along the constructed paths. On DDPM, the maximum training and test losses obtained by AutoNEB are 0.17 and 0.18, respectively whereas our method limits them to 0.03 and 0.15. The difference is substantially larger on NanoCLIP: AutoNEB reaches a maximum training loss of 6.69 and a minimum training accuracy of only 0.05, while our method achieves a maximum training loss of 0.09 and maintains a minimum training accuracy of 0.95. These results show that AutoNEB fails to construct a low-loss path for generative and contrastive models. In contrast, our method maintains consistently low training loss throughout the path.

![](images/1fda71910fba4527db7e32dfbcda74f5912343db9583d00dcd879bbf889fc014.jpg)  
Figure 5: Ablation study of the proposed architecture-aware connection building algorithm on DDPM and NanoCLIP. The left and right columns report the results on DDPM and NanoCLIP, respectively. The first row compares the proposed dataflow-based layer moving strategy with simultaneous movement of all layers. The second row compares the proposed strategy with the forward propagation order. The third row replaces Adam with SGD while keeping the remaining components unchanged. All curves are smoothed.

Ablation Study. We examine two key designs of our method: dataflow-based layer moving strategy and Adambased training refinement. Figure 5 compares the complete method with three variants: Ours w/All Layers, which moves all model layers simultaneously; Ours w/ Forward Order which retains staged movement but activates layers according to the forward propagation order; and Ours w/ SGD, which replaces Adam with SGD. All remaining components and hyperparameters are kept unchanged unless otherwise specified. It is worth noting that we did not construct a complete path in all experiments, as sufficient differences could already be observed within limited ticks.

The first row of Figure 5 investigates whether the dataflowbased layer moving strategy can be replaced by moving all layers simultaneously. On DDPM, the all-layers variant yields an average training loss of 0.031, compared with 0.025 of the complete method. On NanoCLIP, the average training loss of the variant is more than twice that of the complete method. These results indicate that moving all layers increases the optimization difficulty, whereas the proposed strategy more effectively preserves a low-loss trajectory.

The second row compares the proposed strategy with the forward propagation order over the complete trajectories. Our strategy outperforms the forward-order variant on both DDPM and NanoCLIP, reducing the average training loss by 0.002 in each case. This indicates that our dataflow-based order provides a more effective movement schedule than computational order for movement in weight space.

The third row evaluates the optimizer used during training refinement. On DDPM, the complete method achieves an average training loss of 0.024, compared with 0.030 for the SGD variant. The difference is more pronounced on NanoCLIP, where its training loss can increase by two orders of magnitude. These results indicate that the coordinate-wise adaptive updates of Adam improve refinement stability when different models exhibit substantially different gradient magnitudes. Overall, the ablation results show that Adam-based refinement and dataflow-based layer moving strategy provide complementary benefits for preserving low training loss along the constructed paths.

## 6 Discussion

Beyond validating the effectiveness of the proposed method, our empirical results also reveal several observations.

Low training loss does not imply high generation quality. Although mode connectivity is defined primarily by low training loss, examining the image generation capability of intermediate DDPM models provides a complementary assessment. We uniformly sample 41 checkpoints along the path, including both endpoints. Each checkpoint generates 1000 images under an identical sampling configuration and random seed, so corresponding samples differ only in model parameters. Figure 6 shows one representative image from each checkpoint together with its Fréchet Inception Distance (FID). Quantitatively, the intermediate checkpoint generally produces higher FID values than the endpoints, while the training loss of the checkpoints remains basically the same as that of the endpoint. This behavior is expected because the intermediate models are optimized to satisfy the low-loss constraint rather than to minimize FID directly. The results may also indicate that a low-loss region is not necessarily a high-quality generative region.

Endpoint parameter proximity does not imply a flat basin. Prior studies describe well-trained minima as lying within locally flat manifolds. Under this interpretation, once a constructed trajectory approaches the end mode, the remaining neighborhood should be sufficiently flat that direct interpolation introduces little additional loss. We test this expectation around the DDPM endpoint. As shown in the left panel of Figure 9 in Appendix E, the checkpoint at tick 16400 is already strongly aligned with the end mode: layer-wise cosine similarities are concentrated near one across all DDPM modules, indicating that the checkpoint is close to the endpoint in weight space. Nevertheless, the right panel shows that linear interpolation between them creates a maximum training loss of 0.038, compared with a maximum of 0.034 along the previous low-loss path. One possible interpretation is that the low-loss region near the end mode is highly anisotropic: even near a trained mode, the local geometry of foundation-model loss landscapes may be more intricate than the conventional picture of a uniformly flat basin.

![](images/50f8b40e3141f9c624742b95bf656d722bfa6467a75bd3e24f5b0c5937b3bfb4.jpg)

![](images/e681180aa505eb94f36d956b5831992a7262fcf70b705d63ed846974cfa1bcb3.jpg)  
Figure 6: Generation quality evaluation of checkpoints along the constructed low-loss path. Top: representative synthesized images sampled from identical initial Gaussian noise. Bottom: the corresponding FID computed across the 41 sampled checkpoints.

## 7 Conclusion

We investigated whether mode connectivity extends beyond classifiers to independently trained models with generative and contrastive objectives. To address the architectural and optimization challenges posed by DDPM and NanoCLIP, we proposed an architecture-aware connection building algorithm that combines dataflow-based layer movement with Adam-based refinement. Experiments on DDPM trained on Flowers102 and NanoCLIP trained on Flickr30k show that the proposed method constructs continuous low-loss paths while progressively reducing layer-wise parameter distances and maintaining low training loss. Overall, our work is the first to extend mode connectivity to generative and contrastive domains and motivate further studies of losslandscape geometry in modern non-classifier models.

## References

Adilova, L.; Andriushchenko, M.; Kamp, M.; Fischer, A.; and Jaggi, M. 2024. Layer-wise linear mode connectivity In The Twelfth International Conference on Learning Representations.

Ainsworth, S.; Hayase, J.; and Srinivasa, S. 2023. Git Re-

Basin: Merging Models modulo Permutation Symmetries. In The Eleventh International Conference on Learning Representations.

Ali-bey, A. 2024. nanoCLIP: A Lightweight Text-to-Image Retrieval Model. https://github.com/amaralibey/nanoCLIP. GitHub repository, accessed July 14, 2026.

Draxler, F.; Veschgini, K.; Salmhofer, M.; and Hamprecht, F. 2018. Essentially No Barriers in Neural Network Energy Landscape. In Dy, J.; and Krause, A., eds., Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, 1309–1318. PMLR.

Entezari, R.; Sedghi, H.; Saukh, O.; and Neyshabur, B. 2022. The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks. In International Conference on Learning Representations.

Foret, P.; Kleiner, A.; Mobahi, H.; and Neyshabur, B. 2021. Sharpness-aware Minimization for Efficiently Improving Generalization. In International Conference on Learning Representations.

Fort, S.; Dziugaite, G. K.; Paul, M.; Kharaghani, S.; Roy, D. M.; and Ganguli, S. 2020. Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781713829546.

Frankle, J.; Dziugaite, G. K.; Roy, D.; and Carbin, M. 2020. Linear Mode Connectivity and the Lottery Ticket Hypothesis. In III, H. D.; and Singh, A., eds., Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, 3259–3269. PMLR.

Fukumizu, K. 1995. Active Learning in Multilayer Perceptrons. In Touretzky, D.; Mozer, M.; and Hasselmo, M., eds., Advances in Neural Information Processing Systems, volume 8. MIT Press.

Fukumizu, K. 1996. A Regularity Condition of the Information Matrix of a Multilayer Perceptron Network. Neural Networks, 9(5): 871–879.

Garipov, T.; Izmailov, P.; Podoprikhin, D.; Vetrov, D.; and Wilson, A. G. 2018. Loss surfaces, mode connectivity, and fast ensembling of DNNs. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, 8803–8812. Red Hook, NY, USA: Curran Associates Inc.

Glorot, X.; and Bengio, Y. 2010. Understanding the difficulty of training deep feedforward neural networks. In Teh, Y. W.; and Titterington, M., eds., Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, 249–256. Chia Laguna Resort, Sardinia, Italy: PMLR.

Grigsby, J. E.; Lindsey, K.; and Rolnick, D. 2023. Hidden symmetries of ReLU networks. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org.

Hassani, A.; Walton, S.; Shah, N.; Abuduweili, A.; Li, J.; and Shi, H. 2021. Escaping the Big Data Paradigm with Compact Transformers. ArXiv, abs/2104.05704.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2015. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. In 2015 IEEE International Conference on Computer Vision (ICCV), 1026–1034.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep Residual Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770– 778.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Diffusion Probabilistic Models. arXiv:2006.11239.

Huang, G.; Liu, Z.; Van Der Maaten, L.; and Weinberger, K. Q. 2017. Densely Connected Convolutional Networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2261–2269.

Li, H.; Xu, Z.; Taylor, G.; Studer, C.; and Goldstein, T. 2018. Visualizing the Loss Landscape of Neural Nets. In Bengio, S.; Wallach, H.; Larochelle, H.; Grauman, K.; Cesa-Bianchi, N.; and Garnett, R., eds., Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

Nilsback, M.-E.; and Zisserman, A. 2008. Automated Flower Classification over a Large Number of Classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing.

Orr, G. B.; and Müller, K.-R., eds. 1998. Neural Networks: Tricks of the Trade, this book is an outgrowth of a 1996 NIPS workshop. Berlin, Heidelberg: Springer-Verlag. ISBN 3540653112.

Radosavovic, I.; Kosaraju, R. P.; Girshick, R.; He, K.; and Dollar, P. 2020. Designing Network Design Spaces . In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10425–10433. Los Alamitos, CA, USA: IEEE Computer Society.

Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; and Chen, L.-C. 2018. MobileNetV2: Inverted Residuals and Linear Bottlenecks . In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520. Los Alamitos, CA, USA: IEEE Computer Society.

Tan, M.; and Le, Q. V. 2019. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ArXiv, abs/1905.11946.

Tian, Y.; Al-Ars, Z.; Kitsak, M.; and Hofstee, H. P. 2026. Connecting Independently Trained Modes via Layer-Wise Connectivity. In Forty-third International Conference on Machine Learning.

Tian, Y.; Al-Ars, Z.; Kitsak, M.; and Hofstee, P. 2024. Vanishing Variance Problem in Fully Decentralized Neural-Network Systems. arXiv:2404.04616.

Tran, V.-H.; Trinh, V.-H.; Bui, K. V.; and Nguyen, T. M. 2025. On Linear Mode Connectivity of Mixture-of-Experts

Architectures. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Watanabe, S. 2009. Algebraic Geometry and Statistical Learning Theory. USA: Cambridge University Press. ISBN 0521864674.

Young, P.; Lai, A.; Hodosh, M.; and Hockenmaier, J. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2: 67–78.

Yu, F.; Wang, D.; Shelhamer, E.; and Darrell, T. 2018. Deep Layer Aggregation . In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2403–2412. Los Alamitos, CA, USA: IEEE Computer Society.

Zhang, X.; Zhou, X.; Lin, M.; and Sun, J. 2018. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices . In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6848–6856. Los Alamitos, CA, USA: IEEE Computer Society.

## A Empirical Validation of Equation 4

![](images/d5babc3a184c7feea3076b882900c3f2769d6bbc49415ed3a02fdca73f00c62d.jpg)  
Figure 7: Layer-wise weights mean and variance across independently trained NanoCLIP on Flickr30k and DDPM on Flowers102 models with different random seeds. The x-axis denotes the model index, and the y-axis represents the mean or variance of layer parameters. Each curve corresponds to one layer following the PyTorch naming convention. Batch normalization layers are excluded, as their statistics are highly sensitive to the training batches. The results show that layer-wise variances remain consistent across independent training runs and layer-wise means remain centered around zero.

## B Mode-Connection Continuity Check via Linear Interpolation

Figure 8 presents the resulting loss trajectory. Despite the substantially denser sampling, no observable loss barrier appears along the constructed path. The maximum training loss increases by less than 5.3%. The result indicates that the interpolated models remain within the same low-loss region as the optimization trajectory.

![](images/e43cf3c8f24b053b682c907b5632f8c6148677c119c0cd51a0f4cacfa15ce5a3.jpg)  
Figure 8: Mode-connection continuity check for NanoCLIP. Twenty uniformly spaced interpolation points are inserted between every pair of consecutive optimization ticks. All interpolation points also remain within the low-loss region.

## C Hyperparameters and Configurations of Architecture-aware Connection Building Algorithm

<table><tr><td rowspan=1 colspan=2>Training Refinement   Optimizer and   Step-size HyperparametersIteration TStageLayer Moving Sequence U       Round r        Hyperparameter            $( \alpha , \beta , \gamma )$ </td></tr><tr><td rowspan=1 colspan=2>Model: DDPM-Flowers102                               Runtime: 59 hours (NVIDIA RTX 5090)</td></tr><tr><td rowspan=1 colspan=1>26             $\mathrm { S t a g e } 2 5 + \mathbf { u p s . } 1 . 2$ </td><td></td></tr><tr><td rowspan=1 colspan=1>27             $\mathrm { S t a g e } 2 6 + \mathbf { u p s . } 1 . 3$ </td><td></td></tr><tr><td rowspan=1 colspan=1>28            $\mathrm { S t a g e } 2 7 + \mathrm { d o w n s } . 3 . 0$ </td><td></td></tr><tr><td rowspan=1 colspan=1>29            $\mathrm { S t a g e } 2 8 + \mathrm { d o w n s } . 3 . 1$ </td><td></td></tr><tr><td rowspan=1 colspan=1>30            $\mathrm { S t a g e } 2 9 + \mathrm { d o w n s } . 3 . 2$ </td><td></td></tr><tr><td rowspan=1 colspan=1>31            $\mathrm { S t a g e } \ 3 0 + \mathrm { d o w n s } . 3 . 3$ </td><td></td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $\mathrm { S t a g e } \ 3 1 + \mathrm { m i d \_ b l o c k } 1$ </td></tr><tr><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>Stage 32 + mid_attn</td></tr><tr><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1> $\mathrm { S t a g e } 3 3 + \mathrm { m i d \_ b l o c k } 2$ </td></tr><tr><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1> $\mathrm { S t a g e } 3 4 + \mathrm { u p s . } 0 . 0$ </td></tr><tr><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1> $\mathrm { S t a \bar { g e } } 3 5 + \mathrm { u \bar { p s } } . 0 . 1$ </td></tr><tr><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1> $\mathrm { S t a g e } 3 6 + \mathbf { u p s . } 0 . 2$ </td></tr><tr><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>Stage 37 + ups.0.3</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Model: NanoCLIP-Flickr30k                               Runtime: 77 hours (NVIDIA RTX 5090)</td></tr><tr><td rowspan=2 colspan=2>txt.{embeddings+pooler+fc}         4001 $\mathrm { i m g . \{ p a t c h \_ e m b e d + c l s + p o s + m a s k + f c \} }$ 2345              $\mathrm { S t a g e \ 1 + b l o c k s . 0 }$               400 $\mathrm { S t a } \bar { \mathrm { g e } } 2 + \mathrm { b l o c k s . } 1$ 400 $\mathrm { S t a g e } 3 + \mathrm { l a y e r . 0 }$ 400 $\mathrm { S t a g e } ~ 4 + \mathrm { b l o c k s } . 2$ 4006              $\mathrm { S t a } \bar { \mathrm { g e } } 5 + \mathrm { b l o c k s } . 3$               4007               $\mathrm { S t a g e } 6 + \mathrm { l a y e r . l }$                4008              $\mathrm { S t a g e } 7 + \mathrm { b l o c k s } . 4$               4009              $\mathrm { S t a g e } \ 8 + \mathrm { b l o c k s } . 5$               400         Train untilAdamW10              $\mathrm { S t a g e 9 + l a y e r . 2 }$                400         loss&lt; 0.04                                (0.005, 0, 0)11                                                                              $( \mathrm { l r } { = } 1 e ^ { - 4 } , \mathrm { w d } { = } 0 )$  $\mathrm { S t a g e ~ 1 0 + b l o c k s . 6 }$ 400or rounds&gt; 20012             $\mathrm { S t a \bar { g e } ~ 1 1 + b l o c k s . 7 }$               40013             $\mathrm { S t a g e } ~ 1 2 + \mathrm { l a y e r } . 3$               40014             $\mathrm { S t a g e } ~ 1 3 + \mathrm { b l o c k s } . 8$               40015             $\mathrm { S t a \bar { g e } ~ 1 4 + b l o c k s . 9 }$              40016             $\mathrm { S t a g e } 1 5 + \mathrm { l a y e r . 4 }$               40017            $\mathrm { S t a g e ~ 1 6 + b l o c k s . 1 0 }$              40018            $\mathrm { S t a g e ~ 1 7 + b l o c k s . 1 1 }$              40019              $\mathrm { S t a g e } ~ 1 8 + \mathrm { l a y e r } . 5$               40020                All layers                  1000</td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

Table 2: Hyperparameters used for the architecture-aware connection building algorithm. For each model, the whole process is divided into multiple stages, where only the parameter subsets specified in the layer moving sequence are optimized in each stage. The layer names are abbreviated in the table, following the PyTorch implementation of the corresponding architecture. We also report the total runtime for constructing the complete low-loss path.

## D Hyperparameters and Configurations of AutoNEB

<table><tr><td>Cycle</td><td>Updates per Pivot</td><td>Trainable Internal Pivots</td><td>Optimizer and Hyperparameter</td></tr><tr><td colspan="2">Model: DDPM-Flowers102 Model: NanoCLIP-Flickr30k</td><td colspan="2">Runtime: 23 hours (NVIDIA A6000)</td></tr><tr><td colspan="2"></td><td colspan="2">Runtime: 41 hours (NVIDIA A6000)</td></tr><tr><td colspan="2">1</td><td colspan="2">1</td></tr><tr><td>2</td><td>1000 1000</td><td>3</td></tr><tr><td>3</td><td>1000</td><td>7</td></tr><tr><td>4</td><td>1000</td><td>11</td></tr><tr><td>5</td><td>2000</td><td>11</td></tr><tr><td>6</td><td>2000</td><td>11</td></tr><tr><td>7</td><td>1000</td><td>11</td></tr><tr><td>8</td><td>1000</td><td>11</td></tr><tr><td>9</td><td>1000</td><td>11</td></tr><tr><td>10</td><td>1000</td><td>11</td></tr><tr><td>11</td><td>1000</td><td>11</td></tr><tr><td>12</td><td>1000</td><td>11</td></tr><tr><td>13</td><td>1000</td><td>11</td></tr><tr><td>14</td><td>1000</td><td>11</td></tr></table>

Table 3: Hyperparameters used for the AutoNEB baseline. Both DDPM and NanoCLIP use the same 14-cycle optimization schedule. Updates per pivot denotes the number of AdamW updates applied to each trainable internal pivot in a cycle. The two endpoint pivots remain fixed and are excluded from the pivot counts and the final path contains 13 pivots in total. AdamW states are reset at the beginning of each cycle. We also report the total runtime.

## E Experimental Results of Endpoint Parameter Proximity and Linear Interpolation

![](images/a2f52e5adec52abd26967323282a583a859aa1f507d293c1b50ccff5f2dcf313.jpg)

![](images/661a18a6ac8d21ad992026393f6880ffec094e8a24977b3a6c7e9e97c859d85b.jpg)  
Figure 9: Parameter-space proximity and training loss near the DDPM end mode. Left: layer-wise cosine similarity between the DDPM checkpoint at tick 16400 and the end mode. Each box summarizes the distribution of cosine similarity values for multiple layers within one DDPM module, indicating they are highly close in parameter space. Right: training loss near the end mode The left part shows the training loss along the original DDPM low-loss path. The right part shows the training loss obtained by linearly interpolating between the checkpoint at tick 16400 and the end mode using uniformly sampled interpolation points.