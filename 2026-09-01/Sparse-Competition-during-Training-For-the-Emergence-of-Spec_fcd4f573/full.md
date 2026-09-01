# Sparse Competition during Training For the Emergence of Specialized Modules

Baptiste Rossigneux<sup>1</sup> baptiste.rossigneux@inria.fr

<sup>6</sup>Karim Haroun<sup>2</sup> <sub>0</sub>karim.haroun@univ-paris8.fr

<sup>1</sup> Univ. Rennes Inria, IRISA Rennes, France

<sup>2</sup> University of Paris 8 LIASD Paris, France

## Abstract

Modularity in deep neural networks has been proposed as a means of improving both interpretability and training by promoting disentangled representations and reducing redundancy. In this work, we study the emergence of modular structure through competi tion dynamics between groups of neurons during training. We introduce a method that (i) maintains near-baseline accuracy, (ii) induces usage-based modularity by sparsely routing inputs to neuron groups, and (iii) encourages specialization of these modules, such that their activations are correlated with input classes. We evaluate the proposed approach on ImageNet-100 and CIFAR-100 and show that with it, specialized modules emerge without module-level supervision. These modules capture a meaningful high-level structure in the data, with individual modules responding to semantic categories (e.g., dogs or vehicles). We also study the emergence of a hierarchical partition of sub-tasks depending on the number of modules. Our results suggest that competitive dynamics can serve as a simple mechanism for inducing functional modularity in standard architectures.

## Introduction

Modular computation has long been viewed as a promising direction towards more inter-<sub>2</sub>pretable and reusable neural representations. Early work on Mixture-of-Experts and sparse <sup>:</sup>routing [11, 18] proposed architectural solutions to enforce module boundaries. However, iarchitectural separation does not guarantee that modules acquire meaningful functional roles. XA neural network can be made of modules while still representing task-relevant information <sup>r</sup>in a redundant or entangled way.

Recent mechanistic interpretability work [1, 5] reveals that networks decompose computation into identifiable sub-circuits. However, these studies reveal the existence of an internal structure without asking whether it is well-organized. Do networks learn reusable functions or redundant and cluttered heuristics that solve the task? The field’s focus on benchmark performance obscures this distinction. A neural network can achieve high accuracy through reusable localized computations or opaque distributed heuristics.

Subsequent work suggests that internal representations depend not only on the architecture and the task, but also on the training process. For example, open-ended evolutionary systems, such as Picbreeder can produce networks with reusable internal regularities and meaningful responses to perturbations [10, 16, 17]. This motivates our question: can standard network training be biased toward more factored internal organization?

A natural approach is to enforce modularity directly. The recent work on clusterability [8] proposes adding a differentiable penalty to the loss that encourages weights to cluster within pre-defined module groups. This achieves structural modularity because the weight matrices become concentrated within the boundaries of the module. However, this comes at a cost. Enforcing clusterability damages the accuracy of the task and fails to induce semantic specialization, caused by the modules not developing meaningful task-related roles. This results in a tradeoff between accepting reduced accuracy for modular weights and abandoning the structural approach altogether. Therefore, can we achieve both modularity and strong task performance?

In this paper, we propose that modularity should emerge from learned routing, not from weight structure. We introduce a routing-based approach that uses module dropout with dual entropy objectives. The first objective concentrates each input on a few modules, while the second encourages different inputs to use different modules. Modules compete for input, which drives specialization without explicit architectural constraints. On ImageNet-100, sparse top-1 routing remains within roughly one percentage point of the vanilla baseline while inducing strong semantic specialization. Classes with the same coarse label are routed to the same modules approximately 3× more often than expected by chance. This specialization also persists hierarchically as increasing module count refines partitions while preserving semantic coherence.

Moreover, we observe that dense training shows specialized modularity under the same competitive incentives. Yet, sparse training forces modules to be individually deployable, whereas dense training creates co-adapted specialists that rely on each other. This distinction separates the emergence of modular structure from the emergence of useful sparse modularity. The representation structure can be induced just by the training dynamics. Furthermore, we extend our approach in several directions. First, we propose learning module widths through a shared capacity budget to create competition over resources that drive specialization. Second, we apply module routing to multiple layers simultaneously and reveal how semantic hierarchy emerges across depths.

Our main contributions break down as follows:

• We introduce Sparse Competitive Modules (SCM), a label-free routing objective that induces usage-based modularity through per-sample sharpness and population-level balance.

• We show that the resulting modules acquire specialization, with module assignments carrying substantially more class information than without modules or with clustered baselines.

• We demonstrate that sparse competitive training is mechanistically necessary for sparse deployment: dense specialization is not enough.

• We analyze the semantic organization of learned modules across module counts and layers, showing that learned partitions refine hierarchically while preserving semantic coherence.

## 2 Related Works

Most of the recent efforts on modularity tackle either interpretability or efficiency. The interpretability goal consists of decomposing a model into weakly interacting components, making its internal functioning easier to interpret and analyze, while efficiency is achieved through conditional computation. Early work on modular representations in deep neural networks [20] attempts to produce a simplified description of networks through the use of similar connection patterns. Subsequent work formalizes this approach through the notion of clusterability [7], where some parts of the network are thought of as clusters because its parts are densely connected, while they are sparsely connected to the rest. Following this, [8] measures clusterability in a differentiable manner, making it possible to train a neural network on this as an objective, through a loss function that encourages non-interacting clusters. They show that this produces modular components in fully connected layers, but find that these clusters do not reliably become more task-specialized than those in non-modular baselines. Our work addresses this gap. Instead of enforcing modularity as a static property of the weight graph, we study whether sparse competitive routing can induce usage-based modules that are preferentially selected by different visual classes. Moreover, recent work [2] in controlled neural systems shows that structural modularity alone is insufficient for functional specialization and that specialization might depend on resource constraints and architecture. Our work argues that usage-based modularity is sufficient to attain specialization in a neural network. Gradient routing [3] instead localizes capabilities by supplying data-dependent gradient masks. SCM differs in that module assignments are not specified externally: they emerge from label-free competitive routing.

As models scale to billions or more recently to trillions of parameters, Mixture-of-Experts (MoEs) [6, 13, 18] have emerged as a reliable approach to scaling in the architectural design space. MoEs consist of a routing function that activates only a subset of parameters for each input, allowing practitioners to increase model capacity without increasing the computation. In computer vision, V-MoE shows that sparse expert routing can be competitive with dense Vision Transformers while reducing inference cost [15]. Similarly, Soft-MoE shows that it is possible to design fully differentiable MoE architectures to avoid hard token routing [14]. MoEs also emerged in efficient vision architectures such as CNNs [21], showing that experts can be coupled with tightly vision-prior based architectures. Our goal differs from most MoE work. We investigate the relevance of adding experts as an architectural prior that constrains the network’s exploration of the space and its expressivity. We show that softer steering through the loss function can lead to their emergence. This unsupervised emergence of modules is also interesting for interpretability reasons, as it allows us to study the learning dynamics and how they evolve compared to models with enforced modularity through the architecture.

Our routing objective is also closely related to information-maximization clustering objectives such as RIM [9] or Invariant Information Clustering [12] which balance confident assignments and diversity maintenance across a population. The distinction is in the use of the information-maximization: SCM uses not to learn a clustering head or explicit router, but to induce a specialization inside a neural network layer.

Finally, our work is related to mechanistic interpretability methods that seek to design or identify circuits from deep neural networks as human-understandable units [1, 4]. Although we do not introduce new circuit-discovery algorithms, we argue that our contribution to training for emergent modular specialization paves the way for constructing circuits in an unsupervised yet human-understandable manner.

![](images/974016c51957a028481f5b715f36077521d8b1e2d4b40c0acdb5954a990576ae.jpg)  
Figure 1: Overview of Sparse Competitive Modules. (a) A routed layer is partitioned into modules. For each input, module activation energies are converted into routing probabilities, and a hard top-1 mask keeps only the selected module active. (b) The routing objective combines a per-sample sharpness term, which encourages confident module selection, with a batch-level balance term, which prevents collapse by encouraging all modules to be used. (c) Without module-level or semantic supervision, the resulting routing assignments become class-informative, where visually related classes tend to share dominant modules.

## 3 Methodology

This section introduces Sparse Competitive Modules, for which an overview is presented in Figure 1, a training procedure that induces usage-based modularity through the loss function of neural networks. Unlike structural modularity objectives, our approach does not directly form weakly interacting clusters. Instead, it steers the single-module selection per input while preventing collapse. Only one module is activated per routed layer, and the objective encourages the network to balance usage across modules. Importantly, the routing decision is unsupervised by class labels, which are used only in the standard classification loss.

## 3.1 Modularizing a Layer

Let $h _ { \ell } ( x ) \in \mathbb { R } ^ { d _ { \ell } }$ be the activation of a routed layer $\ell \in \mathcal R$ . We divide its coordinates into $K _ { \ell }$ modules $\mathcal { G } _ { \ell , 1 } , \ldots , \mathcal { G } _ { \ell , K _ { \ell } }$ . In the simplest case, these are equal-width contiguous blocks of neurons. This partition defines where modules are, but does not assign any semantic role.

For each module k, let $m _ { \ell , k } \in \{ 0 , 1 \} ^ { d _ { \ell } }$ be the corresponding binary mask. The activation supported by the module k is

$$
h _ { \ell , k } ( x ) = m _ { \ell , k } \odot h _ { \ell } ( x ) , \qquad h _ { \ell } ( x ) = \sum _ { k = 1 } ^ { K _ { \ell } } h _ { \ell , k } ( x ) .\tag{1}
$$

We route inputs using the activation energy of each module,

$$
E _ { \ell , k } ( x ) = \big | \big | m _ { \ell , k } \odot h _ { \ell } ( x ) \big | \big | _ { 2 } + \varepsilon ,\tag{2}
$$

where ε is a small numerical constant stability. A module is selected on the basis of the magnitude of its response to the input. Unlike structural modularity objectives, which impose constraints directly on the weight matrices, our method leaves the weights unconstrained and induces modularity through input-dependent usage patterns. We later relax the equal-width block assumption by using a Gaussian-ring, described at the end of the next subsection.

## 3.2 Sparse Competitive Routing Objective

The goal of our method is to induce structured module usage without providing any classrelated information. For each training sample $( x _ { i } , y _ { i } )$ , the label $y _ { i }$ is used only through the standard classification loss. The routing objective does not observe $y _ { i }$ , it operates only on the activation patterns produced by the network itself. Hence if visual classes become associated with different modules, this structure emerges from competition between modules driven by input-dependent usage, rather than from any externally specified partition of the data.

Our objective combines two competing pressures. The first encourages each input to make a confident routing decision, localizing its computation to a small subset of modules. The second encourages balanced usage across inputs, ensuring that the network retains the capacity of all modules. This balance is necessary because sparse routing has a natural collapse mode that causes all inputs to be routed to the same module. Given $k$ modules of equal size, this would effectively reduce the usable capacity of the routed layer to roughly $1 / k$ of its original capacity. Collapse can be self-reinforcing: a slightly more responsive module receives more routed examples, hence more task gradients, which increase its dominance. Similarly, if one module initially learns a broadly useful classification strategy, task loss alone provides a short-term incentive to route many inputs through that module. As a result, the routing objective must simultaneously produce sharp per-sample decisions while maintaining diversity in module usage at the population level.

For a routed layer $\ell ,$ let $E _ { \ell , k } ( x _ { i } )$ denote the activation energy of the module k for input $x _ { i }$ as defined above. We convert the energy into a differentiable routing distribution as follows:

$$
q _ { \ell , k } ( x _ { i } ) = \frac { \exp ( E _ { \ell , k } ( x _ { i } ) / \tau ) } { \sum _ { k ^ { \prime } = 1 } ^ { K _ { \ell } } \exp ( E _ { \ell , k ^ { \prime } } ( x _ { i } ) / \tau ) } ,\tag{3}
$$

where $\tau > 0$ is a temperature parameter. Smaller values of $\tau$ make the distribution closer to a winner-take-all decision, while larger values provide smoother gradients.

We minimize the entropy of the per-sample routing distribution to induce the first pressure, encouraging each input sample to select a distinct module:

$$
\mathcal { L } _ { \mathrm { s a m p l e } } ^ { ( \ell ) } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } H ( q _ { \ell } ( x _ { i } ) ) = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { k = 1 } ^ { K _ { \ell } } q _ { \ell , k } ( x _ { i } ) \log q _ { \ell , k } ( x _ { i } ) ,\tag{4}
$$

where B denotes the batch size.

We then add the second pressure that maximizes the entropy of the average usage distribution. This is a batch-level balance term under the assumption that, for a sufficiently large batch of inputs, module usage should be evenly distributed:

$$
\mathcal { L } _ { \mathrm { b a l } } ^ { ( \ell ) } = - H ( \bar { q } _ { \ell } ) = \sum _ { k = 1 } ^ { K _ { \ell } } \bar { q } _ { \ell , k } \log \bar { q } _ { \ell , k } .\tag{5}
$$

This objective can be interpreted as encouraging high mutual information between inputs and module assignments, without using labels. The sharpness term reduces the conditional entropy of module assignment for each input, while the balance term raises the marginal entropy of module usage across inputs. Together, these two terms encourage confident, yet diverse, routing decisions. The task loss then determines which assignments are useful for prediction, allowing semantic specialization to emerge indirectly during training.

![](images/a8ca8d5367de5d0ac95de797771924b1d49a24b69a28a1e6192489202a9c940d.jpg)  
Figure 2: Illustration of the working of the Gaussian ring, with a run where the layer fc2 with 512 neurons is modularized into 4 modules. Left is the first epoch, and right is the last epoch. We can see the network learned to get a dominant module 1, without total collapse.

The final training objective is

$$
\mathscr { L } = \mathscr { L } _ { \mathrm { t a s k } } + \alpha \mathscr { L } _ { \mathrm { r o u t e } } , \quad \mathscr { L } _ { \mathrm { r o u t e } } = \mathscr { L } _ { \mathrm { s a m p l e } } + \mathscr { L } _ { \mathrm { b a l } }\tag{6}
$$

where $\mathcal { L } _ { \mathrm { t a s k } }$ denotes the standard supervised loss function and α controls the strength of the routing objective ${ \mathcal { L } } _ { \mathrm { r o u t e } }$ during training.

Gaussian ring module masks for capacity competition. The default SCM formulation partitions a layer into K contiguous blocks of equal width. This imposes a fixed-capacity prior in which each module receives exactly the same number of neurons. To relax this assumption, we replace hard block masks with soft Gaussian masks on a circular neuron axis. We decide to make the axis circular to avoid border effects: with 4 modules, the modules 0 and 3 are only in contact with one module, which would disadvantage them, where we want a fair competition between modules to avoid collapse. The ring simply allows each module to be connected to two modules.

Neuron i and module m are assigned positions $p _ { i } = i / d _ { \ell }$ and $c _ { m } = m / K$ on the unit ring, and module memberships are computed as

$$
G _ { i , m } = \mathrm { s o f t m a x } _ { m } \left( - \frac { d _ { \mathrm { r i n g } } ( p _ { i } , c _ { m } ) ^ { 2 } } { 2 \sigma _ { m } ^ { 2 } T _ { \mathrm { g a u s s } } } \right) ,\tag{7}
$$

where $d _ { \mathrm { r i n g } }$ denotes the periodic distance and $\sigma _ { m }$ is a learnable module width. The widths are constrained by a fixed total budget, so increasing the capacity of one module reduces the capacity available to others. The routing energies are then computed as in the previous formulation, replacing the hard module mask $m _ { \ell , k }$ with the soft membership vector $G _ { : , k }$

This modification does not change the SCM objective or the sparse top-1 routing rule. Instead, it replaces fixed modules of equal width with soft capacity-adaptive modules. As shown in Table 2, the Gaussian ring does not substantially affect accuracy, while it consistently increases class-module mutual information. We therefore use SCM with Gaussian ring masks in the remaining experiments.

## 3.3 Training and Inference

The neural network is trained end-to-end using standard gradient-based training, with a hard top-1 mask applied during the forward, forcing the outputs of all the non-selected modules to be zero. Consequently, gradients obtained through the classification loss update only the selected module in that layer. The auxiliary routing loss is computed from the differentiable energy profile prior to the hard routing decision, allowing the model to reshape its activation energies during training to modify the preferred module assignment for a given input.

At inference time, we apply the same sparse top-1 routing rule as during training:

$$
z _ { \ell } ( x ) = \mathop { \arg \operatorname* { m a x } } _ { k } E _ { \ell , k } ( x ) , \qquad \tilde { h } _ { \ell } ( x ) = m _ { \ell , z _ { \ell } ( x ) } \odot h _ { \ell } ( x ) .\tag{8}
$$

This consistency is important, as dense training relying on usage-based competition can yield class-structured energy profiles without producing modules that operate independently from the rest of the layer. We therefore distinguish sparse competitive training from a dense control setting, as shown in Figure 4-A, in which all modules remain active during training and sparse top-1 routing is applied only at inference time. Task gradients flow only through the selected module, while the routing loss is computed from the pre-argmax distribution.

## 3.4 Usage-Based Modularity Metrics

Following prior work on clusterability-based modularity [8], we aim to quantify how routing induces an informative and balanced distribution of module usage. Ideally, routing should reflect specialized computation, such that different modules are selectively activated for different semantic subsets of data. In a classification task, subtasks are classifications of superclasses. Hence, a specialized module should be detected by a module only being activated when inputs corresponding to classes of a given superclass are present.

Then, we quantify class–module specialization for each routed layer ℓ by computing the mutual information between the class variable C and the selected module index $Z _ { \ell }$

$$
M I ( C ; Z _ { \ell } ) = \sum _ { c , k } p ( c , k ) \log \frac { p ( c , k ) } { p ( c ) p ( k ) } .\tag{9}
$$

## 4 Experiments

## 4.1 Setup

Backbone. For comparability with the clusterability baseline, we use the same global architecture as [8]: a small CNN stem followed by a bias-free fully connected head. After validating our method on CIFAR-10, we scaled the network to train from scratch on CIFAR-100 and ImageNet-100. Our scaled\_cnn maps $x \in \mathbb { R } ^ { 3 \times 2 2 4 \times 2 2 4 }$ through three Conv-BN-ReLU-MaxPool blocks with channels 32,64,128, then adaptive-average-pools to $4 \times 4$ yielding a 2048-dimensional feature vector. The head is $2 0 4 8 \to 5 1 2 \to 5 1 2 \to 5 1 2 \to 1 0 0 .$ with ReLU activations after the first three fully connected layers with no bias terms. We denote the three hidden fully connected layers by $f c 1 , f c 2$ , and $f c 3$ , where fc1 maps the convolutional feature vector to the first 512-dimensional hidden representation, while $f c 2$ and $f c 3$ are 512-to-512 hidden transformations. Routing is applied only to the selected layers among $f c 1 , f c 2$ , and $f c 3$ , and the final 100-way classifier remains dense.

Dataset and training. Our main experiments are conducted on ImageNet-100, a subset of 100 classes from the 1000-class ImageNet dataset. Instead of using a random subset, which could significantly affect the results, we use the standard CMC subset [19], which is designed to provide a more balanced selection of classes across diverse categories. All methods are evaluated with the same backbone and training protocol within each comparison. We report sweeps over routed layers $( f c 2 , f c 3 ,$ , and $f c 2 + f c 3 )$ and over the number of modules $K \in$ {4,8,16}. To demonstrate that our results are not specific to ImageNet-100, we additionally report results on CIFAR-100.

![](images/f79b908929560409c08518c94ce5a164f110a7f41f1b1029d2ca4a2fda444329.jpg)

![](images/75ddb5462e6aba9c670b114336fcd5e816d00f9c6e80dbeb00d6feb4ca498e96.jpg)  
Figure 3: Measures of accuracy and MI of the same CNN model after training on ImageNet-100 CMC. A: The accuracies of our method, clustered baseline from [8] and without modularization. Here, the accuracy of our method corresponds to the top-1 accuracy while other methods are evaluated on dense accuracies. B: the MI between the favored module and the input class.

Training images are processed with ImageNet normalization. Validation images are resized to 256, center-cropped to 224, and normalized in the same manner. Models are trained for 150 epochs with Adam, a batch size of 128, a learning rate of $1 0 ^ { - 3 }$ . Unless stated otherwise, we use a constant $\alpha = 0 . 1$

We compare Sparse Competitive Modules against three baselines. The vanilla baseline is trained with the standard classification loss and no modular objective. The clustered baseline follows the clusterability approach of [8], which encourages weights to concentrate within predefined module groups. Unless otherwise stated, SCM refers to sparse competitive modules with Gaussian ring module masks. Finally, to show that with our method we in fact learn in a similar way as Mixture-of-Experts, but without a learned router, we add a learned router baseline. This baseline uses the same backbone, routed layers, module masks, and number of modules as SCM, but replaces the energy-based routing distribution with a learned linear router $q _ { \ell } ( x ) = \mathrm { s o f t m a x } \left( W _ { \ell } h _ { \ell } ( x ) / \tau \right)$ . During training and evaluation, the selected module is the top-1 router prediction, and all non-selected modules are masked out, as in SCM. The router is trained jointly with the classification objective and standard load-balancing auxiliary losses to avoid collapse. Routing probabilities are computed from module activation energies using temperature $\tau = 1 . 0$

Evaluation. For our method, training and inference use the same sparse top-1 routing rule: only the selected module remains active in a routed layer. We also include a dense-control variant, where the routing objective is used during training but all modules remain active in the forward pass. This separates semantic specialization of the routing scores from sparse deployment of the selected modules.

We report classification accuracy and class-module mutual information $M I ( Y ; M _ { \ell } )$ , where

Table 1: Main comparison on ImageNet-100 CMC and CIFAR-100 with $\mathtt { f } _ { \mathbf { C } 2 }$ routed. Accuracy uses each method’s intended inference mode; MI is computed from the selected/highestenergy module.
<table><tr><td>Dataset</td><td>Method</td><td> $\operatorname { A c c . } \ ( \% )$ </td><td> $\operatorname { M I } ( Y ; M _ { \mathbb { f } \subset 2 } )$ </td></tr><tr><td>ImageNet-100 CMC</td><td>Vanilla</td><td> $5 8 . 7 7 \pm 0 . 3 8 $ </td><td> $0 . 0 3 6 \pm 0 . 0 0 6$ </td></tr><tr><td>ImageNet-100 CMC</td><td>Clustered baseline</td><td> $5 7 . 6 7 \pm 0 . 2 6 $ </td><td> $0 . 0 4 9 \pm 0 . 0 1 2$ </td></tr><tr><td>ImageNet-100 CMC</td><td>SCM</td><td> $5 7 . 8 4 \pm 0 . 5 9$ </td><td> $0 . 6 4 9 \pm 0 . 1 2 8$ </td></tr><tr><td>CIFAR-100</td><td>Vanilla</td><td> $4 7 . 7 6 \pm 2 . 1 0$ </td><td> $0 . 1 1 4 \pm 0 . 0 4 9$ </td></tr><tr><td>CIFAR-100</td><td>Clustered baseline</td><td> $4 9 . 0 0 \pm 1 . 1 0$ </td><td> $0 . 1 3 9 \pm 0 . 0 3 7$ </td></tr><tr><td>CIFAR-100</td><td>SCM</td><td> $4 8 . 9 4 \pm 2 . 4 0$ </td><td> $0 . 4 9 3 \pm 0 . 1 9 4$ </td></tr></table>

$M _ { \ell }$ is the selected module in layer ℓ. To detect collapse and measure the evenness of usage, we also report the effective number of used modules:

$$
\begin{array} { r } { N _ { \mathrm { e f f } } = \exp ( H ( M _ { \ell } ) ) . } \end{array}\tag{10}
$$

For semantic analysis, we measure whether classes from the same coarse ImageNet group are assigned to the same module more often than expected by chance. In the modulecount sweep, we additionally measure whether higher-K partitions refine lower-K partitions using pair retention and weighted child purity.

Finally, to test robustness to the choice of classes, we repeat the main ImageNet-100 experiment on two random class subsets. We report mean and standard deviation across matched seeds when available.

## 4.2 Results

Our method maintains near-baseline accuracy with module specialization Table 1 reports the main ImageNet-100 CMC comparison for routing in $f c 2$ with $k = 4$ modules. These results are also shown in Figure 3. Our method obtains $5 7 . 8 4 \pm 0 . 5 9 \%$ , matching the clustered baseline and remaining within roughly one percentage point of the vanilla model. Thus, SCM does not bring accuracy degradation. We can observe the same on CIFAR-100.

The main difference is not accuracy but the information carried by module usage. Vanilla and clustered models have low class-module mutual information at $f c 2$ , with ${ \cal M } I ( Y ; { \cal M } _ { f c 2 } ) =$ $0 . 0 3 6 \pm 0 . 0 0 6$ and $0 . 0 4 9 \pm 0 . 0 1 2$ , respectively. In contrast, our method reaches $0 . 6 4 9 \pm$ 0.128, an increase of approximately $1 8 \times$ over vanilla and $1 3 \times$ over the clustered baseline. This shows that the selected module is no longer an arbitrary subdivision of the layer: it becomes a class-informative routing variable.

To test whether the CMC split contains an unusually favorable semantic structure, we repeat the experiment on two random ImageNet-100 label subsets. The results in the table of section C of the Appendix show the same qualitative pattern. Thus, the specialization effect is not specific to the CMC label subset: across different class selections, SCM preserves baseline-level accuracy while making module assignments more predictive of class identity.

Increasing the number of modules gives finer specialization, Gaussian ring too. Table 2 isolates the effect of the Gaussian ring masks. The Gaussian ring does not substantially change the accuracy: across $K \in \{ 4 , 8 , 1 6 \}$ , the difference relative to SCM without the ring is small. However, it consistently increases $M I ( Y ; M _ { \mathrm { f c } 2 } )$ , from 0.409 to 0.638 at $K = 4$ from 0.730 to 0.891 at $K = 8$ , and from 1.150 to $1 . 2 9 7$ at $K = 1 6 .$ . Hence, we use SCM with Gaussian-ring competition in the remaining experiments, not for the accuracy, but as a mechanism that strengthens modular specialization.

The learned MoE baseline on Table 2 also shows that we can achieve the accuracy and specialization of a learned router without explicitly using one. At $K = 4$ , it reaches a similar accuracy and mutual information to SCM, but as the number of modules increases, the learned MoE baseline loses in accuracy: 2.3 and 5.8 points in accuracy compared to our method while having a lower MI. The module usage also shows a worse balance overall. Our method is hence able to reproduce (or even surpass in more high pressure settings) a learned router, only with new training dynamics.

Sparse training is necessary for sparse deployment Figures 4-A and B separate two notions of modularity. A dense-control model can learn class-structured routing scores while all modules remain active in the forward pass. However, because every training example can rely on all modules, the selected module is not forced to be individually sufficient. Applying top-1 routing only at evaluation therefore produces modules that are semantically recognizable but not reliably deployable. In contrast, SCM trains and evaluates with the same hard top-1 mask. The selected module receives the task gradient for that example, while non-selected modules do not. This forces each routed module to support the computation assigned to it. The ablation therefore shows that semantic specialization of routing scores is not enough: sparse competitive training is required to obtain modules that remain functional under sparse inference.

Modules recover coarse semantic structure: the emergence of a hierarchy of classes For each class, we define its dominant module as the module selected most often on validation examples, and compare the resulting partition with coarse ImageNet groups, using these groups only for evaluation. Classes from the same coarse group are routed to the same module roughly $3 \times$ more often than chance, suggesting that the modules capture semantic structure rather than arbitrary class identities. We further test whether this structure forms a hierarchy by training independent $f c 2$ models with $K \in \{ 4 , 8 , 1 6 \}$ modules. SCM shows clear coarse-to-fine consistency: pair retention is 28.32% for $4  8$ and 27.15% for $8 \to 1 6$ compared with random baselines of $1 3 . 8 5 \pm 0 . 8 6 \%$ and $6 . 4 0 \pm 0 . 8 9 \%$ . Weighted child purity is also high, at 69.0% and 72.0%. Thus, increasing the number of modules refines broad semantic specialists into finer ones, rather than producing unrelated partitions.

## 4.3 More analysis

Causal specialization tests Finally, we test whether class-module associations are causally relevant rather than merely correlational. For each class, we identify its dominant module, ablate that module at test time, and compare the accuracy drop on classes owned by the ablated module with the drop on all other classes. Table 3 shows that SCM has positive causal gaps in both routed layers: 10.99 percentage points in $f c 2$ and 18.15 in $f c 3$ . Importantly, SCM maintains uniform ownership across the four modules.

The clustered baseline also exhibits causal gaps, but its ownership is highly imbalanced, especially in $f c 3$ , where the effective number of class-owning modules drops to 1.71. Thus,

![](images/126f9c65c32884c418114ead0333bbbad55b8485fe8f72e90edde92c6dc8fa18.jpg)

![](images/5629b19eb96ae8f467a0ce4d3deff8e55f6867f38a26782d54e88654d862765e.jpg)

![](images/f67e48dec656d07e777b1ce80ced00dbc1f9dc4b58d1e7c2d0753c2ad37447cf.jpg)  
Figure 4: Evolution of metrics during training on ImageNet-100 CMC. A. The top-1 accuracies of our method (SCM) compared to no clustering (Vanilla) and the method from [7] (Clustered); in dotted lines are the opposite evaluations compared to training: the dense training is on sparse eval, and the sparse training is on dense eval. Both regimes perform better when evaluated like training, but sparse training is better on dense than dense on sparse eval. B. The evolution of MI between our method trained in sparse and dense training. C. The evolution of the effective number of modules at the end of training. The trainings on our method converge on $N _ { e f f } = 4$ meaning all modules are used evenly.

SCM does not merely increase class-module mutual information; it induces a balanced decomposition whose modules are functionally relevant for their assigned classes. We report these results in Appendix B.

Routing deeper layers improves the accuracy–specialization tradeoff. Next, we ask where SCM should be applied. Table 4 reports a layer-placement sweep on ImageNet-100 CMC using the same scaled\_cnn, and K = 4 modules. Routing in fc1 already induces important class-module specialization, but it comes with a 3.04% accuracy drop relative to the matched dense baseline. Hence, early fully connected features are still too general: forcing a sparse module choice at this stage removes useful shared computation.

Routing later layers gives a better tradeoff. In fc2, sparse top-1 routing slightly exceeds the matched baseline accuracy, 58.06% versus 57.92%, while increasing class-module mutual information from 0.0476 to 0.5049. Routing only fc3 gives the best single-layer specialization, with $M I ( Y ; M ) = 0 . 6 6 2 8$ , and also improves accuracy by 0.32 percentage points. Routing both f c2 and f c3 gives the best overall result, with 59.42% top-1 accuracy and high MI in both layers. This shows that modularity works best in later layers, where features are already semantically organized enough for module selection to be class-informative.

Higher-resolution module partitions refine lower-resolution ones We next ask whether increasing k in completely unrelated runs produces unrelated partitions or a hierarchy of refinements that is compatible across seeds and module numbers: if modules meaningfully partition data on consistent granularity. Table 5 reports pair retention and weighted child purity for independently trained K = 4,8,16 runs. We compare two independently trained partitions using two metrics. Pair retention is the fraction of class pairs assigned to the same module at lower k that remain co-assigned at higher k. We compare it to a random baseline obtained by permuting the higher-k assignments while preserving module sizes. Weighted child purity measures whether each higher-k module mostly comes from a single lower-k parent, weighted by the number of classes in the child module. Table 5 shows that

Table 2: Module-count sweep at $\mathtt { f } _ { \mathtt { C } 2 }$ on ImageNet-100 CMC. MI is measured in nats; MI/lnK is normalized MI. Clustered is evaluated densely; SCM and Learned-MoE use sparse top-1 inference.
<table><tr><td>K</td><td>Method</td><td>Acc. (%)</td><td> $M I ( Y ; M _ { \mathrm { f c } 2 } )$ </td><td> $M I / \ln K$ </td><td> $N _ { \mathrm { e f f } }$ </td></tr><tr><td>4</td><td>Clustered baseline</td><td> $5 7 . 9 2 \pm 0 . 2 2$ </td><td> $0 . 0 4 8 \pm 0 . 0 1 5$ </td><td> $0 . 0 3 5 { \pm } 0 . 0 1 1$ </td><td> $3 . 8 3 \pm 0 . 0 2$ </td></tr><tr><td>4</td><td>Learned-MoE</td><td> $5 7 . 6 0 \pm 0 . 7 0$ </td><td> $0 . 5 0 \pm 0 . 1 0$ </td><td> $0 . 3 6 1 \pm 0 . 0 7 2$ </td><td> $3 . 2 5 \pm 0 . 0 1$ </td></tr><tr><td>4</td><td>SCM w/o Gaussian ring</td><td> $5 7 . 5 3 \pm 0 . 8 1$ </td><td> $0 . 4 0 9 \pm 0 . 0 9 1$ </td><td> $0 . 2 9 5 { \pm } 0 . 0 6 6$ </td><td> $3 . 9 6 \pm 0 . 0 2$ </td></tr><tr><td>4</td><td>SCM + Gaussian ring</td><td> $5 7 . 9 2 \pm 0 . 2 3 $ </td><td> $0 . 6 3 8 \pm 0 . 1 1 1$ </td><td> $0 . 4 6 0 \pm 0 . 0 8 0$ </td><td> $3 . 9 7 \pm 0 . 0 1$ </td></tr><tr><td>8</td><td>Clustered baseline</td><td> $5 8 . 7 4 \pm 0 . 4 4$ </td><td> $0 . 0 6 2 \pm 0 . 0 1 2$ </td><td> $0 . 0 3 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $7 . 8 3 \pm 0 . 0 3$ </td></tr><tr><td>8</td><td>Learned-MoE</td><td> $5 5 . 5 0 \pm 1 . 0 0$ </td><td> $0 . 7 8 \pm 0 . 1 2$ </td><td> $0 . 3 7 5 { \pm } 0 . 0 5 8$ </td><td> $6 . 4 5 \pm 0 . 0 3$ </td></tr><tr><td>8</td><td>SCM w/o Gaussian ring</td><td> $5 7 . 6 1 \pm 0 . 4 6$ </td><td> $0 . 7 3 0 { \pm } 0 . 0 3 9$ </td><td> $0 . 3 5 1 \pm 0 . 0 1 9$ </td><td> $7 . 8 5 \pm 0 . 0 5$ </td></tr><tr><td>8</td><td>SCM + Gaussian ring</td><td> $5 7 . 7 9 \pm 0 . 2 2$ </td><td> $0 . 8 9 1 \pm 0 . 0 5 0$ </td><td> $0 . 4 2 8 \pm 0 . 0 2 4$ </td><td> $7 . 8 3 \pm 0 . 0 8$ </td></tr><tr><td>16</td><td>Clustered baseline</td><td> $5 7 . 5 6 \pm 0 . 8 7$ </td><td> $0 . 1 1 3 \pm 0 . 0 1 9$ </td><td> $0 . 0 4 1 \pm 0 . 0 0 7$ </td><td> $1 5 . 0 9 \pm 0 . 0 8$ </td></tr><tr><td>16</td><td>Learned-MoE</td><td> $5 1 . 0 0 \pm 1 . 5 0 $ </td><td> $0 . 9 0 \pm 0 . 1 5$ </td><td> $0 . 3 2 5 \pm 0 . 0 5 4$ </td><td> $1 4 . 3 4 \pm 0 . 0 9$ </td></tr><tr><td>16</td><td>SCM w/o Gaussian ring</td><td> $5 5 . 7 7 \pm 0 . 7 3$ </td><td> $1 . 1 5 0 \pm 0 . 0 3 0$ </td><td> $0 . 4 1 5 { \pm } 0 . 0 1 1$ </td><td> $1 5 . 3 1 \pm 0 . 1 2$ </td></tr><tr><td>16</td><td>SCM + Gaussian ring</td><td> $5 6 . 8 1 \pm 0 . 3 8$ </td><td> $1 . 2 9 7 \pm 0 . 0 2 6$ </td><td> $0 . 4 6 8 \pm 0 . 0 0 9$ </td><td> $1 5 . 5 2 \pm 0 . 1 0$ </td></tr></table>

Table 3: Causal specialization diagnostic on ImageNet-100 with four modules at fc2 and fc3. Classes are assigned to their dominant module; we then ablate each module and measure the accuracy drop on its own classes and on all other classes. Drops are percentage points. Full per-module results are given in Appendix B.
<table><tr><td>Model</td><td>Layer</td><td>Owned drop</td><td>Other drop</td><td>Gap</td><td>Ownership  $N _ { \mathrm { e f f } }$ </td></tr><tr><td>SCM</td><td>fc2</td><td>22.38</td><td>11.39</td><td>10.99</td><td>3.99</td></tr><tr><td>SCM</td><td>fc3</td><td>27.32</td><td>9.17</td><td>18.15</td><td>3.99</td></tr><tr><td>Clustered</td><td>fc2</td><td>37.40</td><td>15.50</td><td>21.90</td><td>2.48</td></tr><tr><td>Clustered</td><td>fc3</td><td>24.48</td><td>10.59</td><td>13.89</td><td>1.71</td></tr><tr><td>Vanilla</td><td>fc2</td><td>32.22</td><td>21.64</td><td>10.58</td><td>3.39</td></tr><tr><td>Vanilla</td><td>fc3</td><td>19.90</td><td>13.13</td><td>6.77</td><td>3.32</td></tr></table>

SCM obtains a clear refinement structure. SCM has pair retention well above the random baseline for both 4→8 and 8→16 transitions. These fare better than the clustered baseline, being closer or even below their random baseline, and lower than our method. SCM also obtains higher weighted child purity than the clustered baseline. The alluvial visualization in Figure 5 illustrates the same pattern: SCM modules at higher resolution behave like semantic subdivisions of lower-resolution modules, whereas clustered partitions are less stable across independently trained resolutions.

Transfer across architectures. We evaluate our method on more limited settings on ResNet-18 and ViT-Tiny, with results on Table 6. On ResNet-18, SCM reaches 76.88% accuracy with $M I ( Y ; M ) = 1 . 0 6 5$ , compared with 76.12% accuracy for the vanilla model. On ViT-Tiny, SCM increases MI from 0.074 to 1.117 at $\mathtt { f } _ { \mathbf { C } 2 }$ with only a 0.40-point accuracy cost, while at $\pounds _ { \mathrm { C } } 3$ it improves accuracy (80.70% vs. 80.30%) and obtains strong MIs. Table 6 also show that the coarse-to-fine organization also transfer to ResNet-18, as pair retention is 28% and 30% for 4 → 8 and $8 \to 1 6$ , respectively, well above random baselines of 13% and 7%, with child purities of 70% and 72%. These results show that SCM transfers across convolutional, residual, and Transformer architectures.

Table 4: Layer-placement sweep on ImageNet-100 CMC. All rows use scaled\_cnn, seed 42, 150 epochs, and K = 4 modules. Accuracy values are percentages.
<table><tr><td>Routed layers</td><td>Baseline acc.</td><td>Ours dense acc.</td><td>Ours top-1 acc.</td><td>Baseline MI</td><td>Ours MI</td></tr><tr><td> $f c 1$ </td><td>58.80</td><td>46.98</td><td>57.16</td><td>0.0179</td><td>0.4059</td></tr><tr><td> $f c 2$ </td><td>57.92</td><td>53.54</td><td>58.06</td><td>0.0476</td><td>0.5049</td></tr><tr><td> $f c 3$ </td><td>58.80</td><td>55.90</td><td>59.12</td><td>0.0763</td><td>0.6628</td></tr><tr><td> $f c 2 + f c 3$ </td><td>58.12</td><td>36.24</td><td>59.42</td><td>0.049 / 0.095</td><td>0.615 / 0.307</td></tr></table>

Table 5: Hierarchy metrics for the f c2 module-count sweep on ImageNet-100 CMC.
<table><tr><td>Method</td><td>Transition</td><td>Pair retention</td><td>Random baseline</td><td>Weighted child purity</td></tr><tr><td>Ours</td><td> $4  8$ </td><td>28.32</td><td> $1 3 . 8 5 \pm 0 . 8 6$ </td><td>69.00</td></tr><tr><td>Ours</td><td> $8 \to 1 6$ </td><td>27.15</td><td> $6 . 4 0 \pm 0 . 8 9$ </td><td>72.00</td></tr><tr><td>Clustered</td><td> $4  8$ </td><td>21.70</td><td> $2 3 . 7 2 \pm 1 . 5 5$ </td><td>66.00</td></tr><tr><td>Clustered</td><td> $8 \to 1 6$ </td><td>11.33</td><td> $9 . 6 2 \pm 0 . 8 3$ </td><td>48.00</td></tr></table>

## 5 Conclusion

In this work we proposed a method that yields modules in neural networks through competition. The competition is mainly carried through usage-based competition with Sparse Competitive Modules allowing for a modularization that produces meaningfully specialized modules. Our second, complementary way to further increase specialization of modules is through capacity-based competition with Gaussian ring ownership of neurons, where the width of modules is learned. These two ways, as we showed, allow for modules that conserve the accuracy of a non-modularized network, while meaningfully gathering the data into groups at varying granularity, adapting to its number of modules. This work demonstrates a new way to modularize networks: through the loss only, via information theoretical metrics. This approach further manifests that the internal organization does not need to be monitored, but given the right incentives, it can appear spontaneously.

## 6 Limitations and future works

Our experiments are intentionally conducted in a controlled setting, which makes module usage, specialization, and ablation easy to measure. A natural next step is to apply the same objective to larger backbones, larger vision datasets, and models with richer internal structure, language models, and vision-language models.

A second limitation is that only a small number of layers are modularized. Most experiments route one layer, and our multi-layer results are limited to 1 to 3 layers. This is enough to show that routing becomes class-informative and remains usable under sparse inference, but not enough to claim full-network modular computation or complete circuit recovery. The main interpretability promise of modular training is to obtain whole circuits: connected chains of modules that support particular computations across depth. Our preliminary multilayer experiment in Appendix A, where $f c 2$ is routed into four modules and $f c 3$ into eight modules, suggests that this may occur naturally. This suggests that sparse competitive routing may already bias the network toward structured module-to-module pathways.

![](images/84485d135f7004daf0bc6c00db3b600d849d3c182abcc6d364269cec01dd3678.jpg)  
Figure 5: Alluvial visualization of class assignments across independently trained $f c 2$ runs with K = 4,8,16 modules. Each block is a learned module and each flow tracks ImageNet-100 classes across module-count resolutions. We compare the cleaner refinement structure obtained by our method against the two alternative flows shown in the figure. To rule out this structure being a seed artifact, we also show two independent seeds of our method, both of which exhibit similar coarse-to-fine class flows.

(a) Architecture transfer  
(b) ResNet-18 hierarchical refinement
<table><tr><td>Backbone</td><td>Layer</td><td>Vanilla acc.</td><td>SCM acc. / MI</td></tr><tr><td>ResNet-18</td><td>fc2</td><td>76.12</td><td>76.88 / 1.065</td></tr><tr><td>ViT-Tiny</td><td>fc2</td><td>80.66</td><td>80.26 / 1.117</td></tr><tr><td>ViT-Tiny</td><td>fc3</td><td>80.30</td><td>80.70 / 1.130</td></tr></table>

<table><tr><td>Transition</td><td>Pair ret.</td><td>Random</td><td>Child purity</td></tr><tr><td> $4 \to 8$ </td><td>28%</td><td>13%</td><td>70%</td></tr><tr><td> $8 \to 1 6$ </td><td>30%</td><td>7%</td><td>72%</td></tr></table>

Table 6: Transfer beyond scaled\_cnn on ImageNet-100 CMC. (a) Accuracy and classmodule MI for ResNet-18 and ViT-Tiny with $K = 4 ;$ SCM is evaluated with sparse top-1 inference. (b) Coarse-to-fine refinement across independently trained ResNet-18 modulecount configurations.

Aknowledgements. This work was funded by the HOLIGRAIL project (ANR-23-PEIA-0010). Computing and storage resources were provided by GENCI at IDRIS under grant 2026-AD011016984 on the H100 partition of the Jean Zay supercomputer.

## References

[1] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformer-circuits.pub/2023/monosemanticfeatures/index.html.

[2] Gabriel Béna and Dan F. M. Goodman. Dynamics of specialization in neural modules under resource constraints. Nature Communications, 16(1):187, 2025. doi: https://doi.org/10.1038/s41467-024-55188-9. URL https://www.nature.com/ articles/s41467-024-55188-9.

[3] Alex Cloud, Jacob Goldman-Wetzler, Evžen Wybitul, Joseph Miller, and Alexander Matt Turner. Gradient routing: Masking gradients to localize computation in neura networks. arXiv preprint arXiv:2410.04332, 2024.

[4] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.

[5] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.

[6] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. ArXiv, abs/2101.03961, 2021. URL https://api.semanticscholar.org/CorpusID:231573431.

[7] Daniel Filan, Stephen Casper, Shlomi Hod, Cody Wild, Andrew Critch, and Stuart Russell. Clusterability in neural networks, 2021. URL https://arxiv.org/ abs/2103.03386.

[8] Satvik Golechha, Maheep Chaudhary, Joan Velja, Alessandro Abate, and Nandi Schoots. Studying cross-cluster modularity in neural networks, 2025. URL https: //arxiv.org/abs/2502.02470.

[9] Ryan Gomes, Andreas Krause, and Pietro Perona. Discriminative clustering by regularized information maximization. In Neural Information Processing Systems, 2010. URL https://api.semanticscholar.org/CorpusID:2906843.

[10] Joost Huizinga, Kenneth O Stanley, and Jeff Clune. The emergence of canalization and evolvability in an open-ended, interactive evolutionary system. Artificial life, 24(3): 157–181, 2018.

[11] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991.

[12] Xu Ji, Joao F Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9865–9874, 2019.

[13] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=qrwe7XHTmYb.

[14] Joan Puigcerver, Carlos Riquelme, Basil Mustafa, and Neil Houlsby. From sparse to soft mixtures of experts, 2023.

[15] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. In Neural Information Processing Systems, 2021. URL https://api.semanticscholar.org/CorpusID:235417196.

[16] Jimmy Secretan, Nicholas Beato, David B D Ambrosio, Adelein Rodriguez, Adam Campbell, and Kenneth O Stanley. Picbreeder: evolving pictures collaboratively online. In Proceedings of the SIGCHI conference on human factors in computing systems, pages 1759–1768, 2008.

[17] Jimmy Secretan, Nicholas Beato, David B D’Ambrosio, Adelein Rodriguez, Adam Campbell, Jeremiah T Folsom-Kovarik, and Kenneth O Stanley. Picbreeder: A case study in collaborative evolutionary exploration of design space. Evolutionary computation, 19(3):373–403, 2011.

[18] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparselygated mixture-of-experts layer. ArXiv, abs/1701.06538, 2017. URL https://api. semanticscholar.org/CorpusID:12462234.

[19] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020, pages 776–794, Cham, 2020. Springer International Publishing. ISBN 978-3-030-58621-8.

[20] Chihiro Watanabe, Kaoru Hiramatsu, and Kunio Kashino. Modular representation of layered neural networks. Neural Networks, 97, 03 2017. doi: 10.1016/j.neunet.2017. 09.017.

[21] Yihua Zhang, Ruisi Cai, Tianlong Chen, Guanhua Zhang, Huan Zhang, Pin-Yu Chen, Shiyu Chang, Zhangyang Wang, and Sijia Liu. Robust mixture-of-expert training for convolutional neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 90–101, October 2023.

# Supplementary Material: Sparse Competition during Training For the Emergence of Specialized Modules

Baptiste Rossigneux<sup>1</sup> baptiste.rossigneux@inria.fr

Karim Haroun<sup>2</sup>

<sup>1</sup> Univ. Rennes

Inria, IRISA

Rennes, France

karim.haroun@univ-paris8.fr

<sup>2</sup> University of Paris 8

LIASD

Paris, France

This supplementary material is used as the appendix of the paper Sparse Competition during Training for the Emergence of Specialized Modules. Unless otherwise stated, all additional analyzes use the same ImageNet-100 scaled\_cnn setup as in the main paper. We report two complementary diagnostics. First, we visualize whether modular assignments across two routed layers form a coherent cross-layer structure. Second, we provide the full per-module breakdown of the causal specialization test summarized in the main paper.

## A. Cross-layer modular structure

When more than one layer is routed, a natural question is whether modules at different layers form coherent multi-layer circuits, or whether their assignments are independent across layers. To investigate this, we compare two variants with a routed fc2 layer using K = 4 modules and a routed fc3 layer using K = 8 modules. In the unconstrained variant, no explicit relation is imposed between the modules of the two layers. In the constrained variant, we impose a hard hierarchical connectivity prior: each fc2 module is allowed to feed only two designated fc3 modules.

Figure 1 shows the resulting alluvial diagrams. Each flow represents classes that share the same dominant module at fc2 and fc3. The constrained and unconstrained variants produce qualitatively similar flows, suggesting that cross-layer modular organization can arise without explicitly imposing a parent–child module hierarchy. This diagnostic is qualitative, but it supports the view that SCM can align modular assignments across layers through the training objective itself, rather than through a hand-designed connectivity prior.

## B. Per-module causal specialization

The main paper reports an aggregate causal specialization diagnostic. Here we provide the full per-module breakdown. For each trained model and each routed layer, we first assign every class to its dominant module. We then ablate one module at test time and measure the accuracy drop on the classes owned by the ablated module and on all remaining classes. A positive gap indicates that the ablated module is more important for the classes assigned to it than for other classes.

![](images/71082da69e9080aa77e2bb40fcc51989daa67b765b6492cfff15c97f3c291750.jpg)

![](images/bcf8f0f6c5ebc543589c5ac95f2adcc9326c971c3fbf65e81b0dc90059203335.jpg)

![](images/92374524cc7787e56e8653ac91693e974ffbdae4dfd46ef6d8a090c85d7b6d2a.jpg)  
Figure 1: Alluvial visualization of shared dominant classes between a routed $\mathtt { f } _ { \mathbf { C } 2 }$ layer with $K = 4$ modules and a routed fc3 layer with $K = 8$ modules. We compare an unconstrained model with a hierarchy-constrained variant in which each $\mathtt { f } _ { \mathbf { C } 2 }$ module can connect only to two designated $\pounds _ { \mathsf { C } } 3$ modules. The two diagrams are qualitatively similar, indicating that cross-layer modular structure can arise without explicitly enforcing a hierarchical routing prior.

Table 1 shows that SCM has a positive causal gap for every module at both $\mathtt { f } _ { \mathbf { C } 2 }$ and $\mathtt { f } _ { \mathrm { C } 3 }$ . Moreover, SCM assigns classes nearly uniformly across modules: each module owns between 23 and 28 classes. The clustered baseline also exhibits positive causal gaps, but with substantial ownership imbalance. In particular, at $\mathtt { f } _ { \mathrm { C } 3 }$ , one clustered module owns 86 out of 100 classes, while the other three modules own only 4, 8, and 2 classes. Vanilla networks also show positive gaps, but the effect is weaker at $\pounds _ { \mathrm { C } } 3$ . These results support the interpretation that SCM induces balanced, causally relevant specialization rather than only class-correlated routing assignments.

## C. Robustness of the ImageNet-100 CMC results

To test whether the CMC ImageNet-100 split contains an unusually favorable semantic structure, we repeat the main routed-fc2 experiment on two random 100-class subsets. Table 2 shows that SCM preserves baseline-level accuracy while substantially increasing class–module mutual information across both subsets.

Table 1: Per-module causal specialization on ImageNet-100 using scaled\_cnn, seed 42, and four-module fc2+fc3 runs. Classes are assigned to their dominant module, defined as the module selected most often for that class. Each row ablates one module at test time. Owned classes counts the classes assigned to the ablated module. Owned drop and Other drop are the corresponding accuracy decreases, in percentage points, on owned classes and on all remaining classes. Gap is the difference between these two drops; positive values indicate class-specific causal specialization.
<table><tr><td>Model</td><td>Layer</td><td>Module Owned classes</td><td>Owned drop</td><td>Other drop</td><td>Gap</td></tr><tr><td>SCM</td><td>fc2</td><td>1</td><td>25 20.96</td><td>8.24</td><td>12.72</td></tr><tr><td>SCM</td><td>fc2</td><td>2</td><td>23</td><td>23.30</td><td>13.22 10.08</td></tr><tr><td>SCM</td><td>fc2</td><td>3</td><td>24</td><td>20.00 14.39</td><td>5.61</td></tr><tr><td>SCM</td><td>fc2</td><td>4</td><td>28</td><td>24.93 9.53</td><td>15.40</td></tr><tr><td>SCM</td><td>fc3</td><td>1</td><td>24</td><td>23.83 11.95</td><td>11.89</td></tr><tr><td>SCM</td><td>fc3</td><td>2</td><td>25</td><td>25.04</td><td>9.65 15.39</td></tr><tr><td>SCM</td><td>fc3</td><td>3</td><td>23</td><td>25.04</td><td>9.38 15.67</td></tr><tr><td>SCM</td><td>fc3</td><td>4</td><td>28</td><td>34.21</td><td>5.53 28.69</td></tr><tr><td>Clustered baseline</td><td>fc2</td><td>1</td><td>12</td><td>51.83 15.89</td><td>35.95</td></tr><tr><td>Clustered baseline</td><td>fc2</td><td>2</td><td>16</td><td>46.75</td><td>18.12 28.63</td></tr><tr><td>Clustered baseline</td><td>fc2</td><td>3</td><td>3</td><td>67.33</td><td>12.31 55.02</td></tr><tr><td>Clustered baseline</td><td>fc2</td><td>4</td><td>69</td><td>31.42</td><td>17.29 14.13</td></tr><tr><td>Clustered baseline</td><td>fc3</td><td>1</td><td>4</td><td>38.00</td><td>12.23 25.77</td></tr><tr><td>Clustered baseline</td><td>fc3</td><td>2</td><td>8</td><td>50.25</td><td>12.43 37.82</td></tr><tr><td>Clustered baseline</td><td>fc3</td><td>3</td><td>2</td><td>44.00</td><td>8.08 35.92</td></tr><tr><td>Clustered baseline</td><td>fc3</td><td>4</td><td>86</td><td>21.00</td><td>4.86 16.14</td></tr><tr><td>Vanilla</td><td>fc2</td><td>1</td><td>39</td><td>25.03</td><td>17.74 7.29</td></tr><tr><td>Vanilla</td><td>fc2</td><td>2</td><td>38</td><td>38.74</td><td>26.87 11.87</td></tr><tr><td>Vanilla</td><td>fc2</td><td>3</td><td>15</td><td>34.80</td><td>24.78 10.02</td></tr><tr><td>Vanilla</td><td>fc2</td><td>4</td><td>8</td><td>31.50</td><td>17.80 13.70</td></tr><tr><td>Vanilla</td><td>fc3</td><td>1</td><td>34</td><td>20.06</td><td>13.97 6.09</td></tr><tr><td>Vanilla</td><td>fc3</td><td>2</td><td>10</td><td>21.20</td><td>15.42 5.78</td></tr><tr><td>Vanilla</td><td>fc3</td><td>3</td><td>11</td><td>16.18</td><td>11.19 4.99</td></tr><tr><td>Vanilla</td><td>fc3</td><td>4</td><td>45</td><td>20.40</td><td>11.49 8.91</td></tr></table>

Table 2: Results across two random ImageNet-100 label subsets. Sparse top-1 SCM preserves baseline-level accuracy while substantially increasing class-module mutual information. Bold denotes the best value within each metric group.
<table><tr><td rowspan="2">Subset</td><td colspan="4">Accuracy</td><td colspan="2">MI(class; module)</td></tr><tr><td>Vanilla</td><td>Clustered</td><td>SCM dense</td><td>SCM top-1</td><td>SCM</td><td>Clustered</td></tr><tr><td>Label seed 42</td><td>0.5528</td><td>0.5528</td><td>0.5136</td><td>0.5628</td><td>0.5062</td><td>0.0614</td></tr><tr><td>Label seed 43</td><td>0.6284</td><td>0.6306</td><td>0.5972</td><td>0.6282</td><td>0.5832</td><td>0.0657</td></tr><tr><td>Mean</td><td>0.5906</td><td>0.5917</td><td>0.5554</td><td>0.5955</td><td>0.5447</td><td>0.0636</td></tr></table>

## D. More details on the algorithms

Algorithm 1 provides the pseudo-code of the Gaussian ring, and Algorithm 2 provides the pseudo-code summary of our proposed approach, namely Sparse Competitive Modules (SCM).

```latex
Algorithm 1 Gaussian-ring soft module memberships
Require: Layer width $d ,$ number of modules $K ,$ , temperatures $\tau _ { g } , \tau _ { \sigma } > 0$ (usually 1.0), bud
get factor $\beta ,$ , width-floor fraction $\eta = 0 . 1$ , learnable logits $\overset { \vartriangle } { \boldsymbol { a } } \in \mathbb { R } ^ { K }$
Ensure: Membership matrix $G \in [ 0 , \dot { 1 } ] ^ { d \times K }$ with $\textstyle \sum _ { k = 1 } ^ { K } G _ { j k } = 1$
1: Place fixed module centers on the unit ring:
$c _ { k }  { \frac { k - 1 } { K } } ,$ $k = 1 , \ldots , K .$
2: Set the total width budget:
$B _ { \sigma }  \frac { \beta } { K } .$
3: Allocate widths under this budget:
$\sigma _ { \mathrm { m i n } }  \eta \frac { B _ { \sigma } } { K } , \qquad \rho  \mathrm { s o f t m a x } ( a / \tau _ { \sigma } ) ,$
$\sigma _ { k } \gets \sigma _ { \operatorname* { m i n } } + ( B _ { \sigma } - K \sigma _ { \operatorname* { m i n } } ) \rho _ { k } , \qquad k = 1 , \ldots , K .$
4: for neuron $j = 1 , \ldots , d$ do
5: Set ring position:
$p _ { j }  \frac { j - 1 } { d } .$
6: for module $k = 1 , \ldots , K$ do
7: Compute smooth periodic squared distance (for the ring effect):
$\Delta _ { j k } ^ { 2 }  \frac { 1 - \cos ( 2 \pi ( p _ { j } - c _ { k } ) ) } { 2 \pi ^ { 2 } } .$
8: Compute Gaussian log-affinity:
$\ell _ { j k } \gets - \frac { \Delta _ { j k } ^ { 2 } } { 2 \tau _ { g } \sigma _ { k } ^ { 2 } } .$
9: end for
10: Normalize affinities across modules:
$G _ { j , : }  \mathrm { s o f t m a x } ( \ell _ { j , : } ) .$
11: end for
12: return G
```

Algorithm 2 Sparse Competitive Modules (SCM) training loop   
Require: Data ${ \mathcal { D } } ,$ network $f _ { \theta } ^ { \mathrm { ~ ~ } }$ , routed layers ${ \mathcal { L } } ,$ , routing temperature $\tau _ { r }$ , routing weight α   
Ensure: Trained parameters $\theta$ and Gaussian ring logits $\{ a ^ { \ell } \} _ { \ell \in \mathcal { L } }$   
1: for each training batch $( x , y )$ do   
2: Run the forward pass of $f _ { \theta }$ , applying the following update at each routed layer $\ell \in { \mathcal { L } } \colon$   
3: for each routed layer $\ell \in { \mathcal { L } }$ do   
4: Compute pre-activation: $u ^ { \ell } \gets W ^ { \ell } h ^ { \ell - 1 }$   
5: Compute module memberships:   
$G ^ { \ell } \gets \mathrm { G A U S S I A N R I N G } ( d _ { \ell } , K _ { \ell } , \tau _ { g } ^ { \ell } , \tau _ { \sigma } ^ { \ell } , \beta ^ { \ell } , a ^ { \ell } ) .$   
6: Compute module energies:   
$E _ { b k } ^ { \ell } \gets \Big \| \mathrm { R e L U } ( u _ { b } ^ { \ell } ) \odot G _ { : , k } ^ { \ell } \Big \| _ { 2 } + \varepsilon .$   
7: Compute routing probabilities:   
$q _ { b } ^ { \ell } \gets \operatorname { s o f t m a x } ( E _ { b } ^ { \ell } / \tau _ { r } )$   
8: Select the winning module:   
$z _ { b } ^ { \ell } \gets \mathrm { s t o p g r a d } \bigg ( \mathop { \mathrm { a r g m a x } } _ { k } E _ { b k } ^ { \ell } \bigg )$   
9: Apply sparse top-1 routing:   
$\begin{array} { r } { h _ { b } ^ { \ell } \gets \mathrm { R e L U } ( u _ { b } ^ { \ell } ) \odot G _ { : , z _ { b } ^ { \ell } } ^ { \ell } . } \end{array}$   
10: Compute average module usage:   
$\bar { q } _ { k } ^ { \ell } \gets \frac { 1 } { | \mathcal { B } | } \sum _ { b = 1 } ^ { | \mathcal { B } | } q _ { b { k } } ^ { \ell } .$   
11: Compute routed-layer loss:   
$\mathcal { L } _ { \mathrm { r o u t e } } ^ { \ell } \gets - \frac { 1 } { | \mathcal { B } | } \sum _ { b = 1 } ^ { | \mathcal { B } | } \sum _ { k = 1 } ^ { K _ { \ell } } q _ { b k } ^ { \ell } \log q _ { b k } ^ { \ell } + \sum _ { k = 1 } ^ { K _ { \ell } } \bar { q } _ { k } ^ { \ell } \log \bar { q } _ { k } ^ { \ell } .$   
12: end for   
13: Compute total loss:   
$\mathcal { L } \gets \mathrm { C E } ( f _ { \theta } ( x ) , y ) + \alpha \sum _ { \ell \in \mathcal { L } } \mathcal { L } _ { \mathrm { r o u t e } } ^ { \ell } .$   
14: Update $\theta$ and $\{ a ^ { \ell } \} _ { \ell \in \mathcal { L } }$ by backpropagation.   
15: end for