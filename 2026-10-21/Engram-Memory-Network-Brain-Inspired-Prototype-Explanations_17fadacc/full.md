# Engram Memory Network: Brain-Inspired Prototype Explanations

Hyunjun Kim

Department of Electrical and Computer Engineering

Seoul National University

Seoul, South Korea

doctor3390@snu.ac.kr

Myoung Hoon Ha

Center for Neuroscience-inspired AI

KAIST

Daejeon, South Korea

mh.ha.soar@gmail.com

Abstract—Artificial intelligence systems are increasingly being deployed in high-stakes decision-making scenarios. In response, prototype-based models in explainable AI have attracted growing attention for their ability to retrieve visually similar examples as faithful explanations. Although these models achieve competitive classification accuracy, they often exhibit degraded explanation quality across datasets with varying characteristics, thereby limiting their practical applicability. To address this limitation, we draw on insights from cognitive neuroscience, particularly the human memory system, which encodes inputrelevant information into representations that support recognition. In this paper, we propose an Engram Memory Network (EMN), a model that emulates the ventral visual stream and inferior temporal cortex in the human brain. Central to this model is a Hopfield-inspired memory module that iteratively retrieves representative prototypes. These memory-refined latents are used for both reconstruction and classification, encouraging the selected prototypes to capture input-relevant information across diverse datasets. To address the lack of standardized evaluation for prototype explanations, we assess three complementary aspects: perceptual similarity (LPIPS and DISTS) for perceived resemblance, subclass alignment, measuring whether prototypes match the input’s fine-grained category beyond supervised labels, and information preservation (NMI), measuring how much input information is captured by selected prototypes. Experiments show that EMN improves visual similarity by 17.06%, subclass alignment by 104.57%, and NMI by 42.42% on average across CIFAR-10/100, MNIST, HAM10000, and CUB-200, while achieving the highest average classification accuracy (87.62%) among all methods. These findings underscore the effectiveness of neuroscience-inspired explanation mechanisms in advancing the interpretability and reliability of prototype-based models.

## I. INTRODUCTION

Prototype-based explanation models [1], [2] have emerged as a central approach for explainable AI (XAI; [3]). They justify model predictions by retrieving examples that resemble the input. As deep learning extends beyond benchmark classification tasks to high-stakes domains such as medical diagnosis, autonomous driving, and financial decision-making [4], the demand for accurate and interpretable models has grown. In these applications, prototype-based methods hold particular promise, as they aim to provide explanations that are not only faithful but also intuitive and understandable to human users.

While early methods suffered from limited predictive performance [5], [6], recent advances in model architecture and training strategies have substantially improved predictive accuracy [7], [8]. However, consistently achieving high-quality, human-understandable explanations across diverse datasets remains challenging [9].

The explanatory effectiveness of these models varies substantially across datasets. For instance, ProtoVAE [10] reconstructs prototypes from latent vectors and performs well on simple datasets such as MNIST [11], but often produces blurry, ambiguous outputs on more complex datasets such as CIFAR-10 [12]. In contrast, patch-based methods such as ProtoPNet [1] and ProtoViT [8] improve explainability by identifying localized regions of similarity between input and prototypes. These approaches are practical on datasets like CUB-200 [13], where object structures are well-defined and redundant across samples. However, when patch-level spatial correspondence between the input and prototypes is weak, as in CIFAR-10 [12], the selected prototypes may not align clearly with the input, ultimately undermining explanatory validity (See Figure 1).

In addition, metrics for evaluating whether prototypes are genuinely relevant to inputs remain underexplored [9]. We propose three complementary metrics: (i) visual similarity, measured by Learned Perceptual Image Patch Similarity (LPIPS [14]) and Deep Image Structure and Texture Similarity (DISTS [15]), for human-perceived resemblance; (ii) subclass alignment for fine-grained relevance, evaluated by training on superclass labels and measuring whether prototypes match withheld subclasses—critical in applications like medical diagnosis where within-class distinctions (e.g., cancer subtypes) determine explanation validity; and (iii) information preservation, measured by Normalized Mutual Information (NMI [16]) between input images and their selected prototypes, measuring how much input information prototypes capture.

To address the dataset-dependency problem, we draw on insights from the human visual system, where memory representations are believed to encode input-relevant information for recognition [17]. Accordingly, memories that contribute most to such representations may serve as valid explanations; this motivates our use of prototypes.

In this paper, we propose the Engram Memory Network (EMN), which integrates classification, reconstruction, and explanation. Specifically, EMN comprises three components: (i) an encoder that extracts visual features; (ii) an Engram Memory Module (EMM) that performs memory-based association via a modified update rule inspired by a continuous modern Hopfield network [18]; and (iii) a decoder that reconstructs the input. Visual and categorical latent vectors are iteratively refined through interaction with prototype memories. After refinement of the latent vectors, the model predicts the label, reconstructs the image, and selects the top-K activated prototypes for explanation. Furthermore, we introduce regularization that promotes prototype diversity and mitigates over-reliance on a narrow subset of prototypes. We empirically evaluate EMN on diverse benchmarks including CIFAR-10/100 [12], MNIST [11], HAM10000 [19], and CUB-200 [13]. On average across datasets, EMN improves visual similarity (LPIPS [14] and DISTS [15]) by 17.06% over ProtoPFormer [7] and information preservation (NMI [16]) by 42.42% over ProtoViT [8]. EMN achieves 104.57% higher subclass alignment over ProtoPFormer [7], demonstrating its ability to capture fine-grained semantic structure. For classification, EMN achieves the highest average accuracy (87.62%) among all compared methods, notably achieving the best performance on HAM10000 [19] (88.51%). Our further studies in Sec. IV-E and IV-F reveal an explicit accuracy–explanation trade-off: relaxing the separation loss and/or increasing the latent dimensionality improves accuracy (e.g., 97.37% on CIFAR-10 [12], surpassing most baselines) while moderately reducing explanation quality, which nonetheless remains superior to baseline; we adopt a default setting that prioritizes explanation quality over classification accuracy in this paper. These results suggest that EMN produces interpretable explanations grounded in memory-based association, achieving consistent performance across diverse datasets <sup>1</sup>.

![](images/a5ae19f31a7062ac081a8e7363a092a439c1aca9b30eb42e11df760903050afa.jpg)  
Fig. 1. Queries and selected prototypes by baselines (ProtoVAE [10], ProtoPNet [1], ProtoViT [8], ProtoPFormer [7]) and EMN (ours). Each row shows a model’s top-3 retrieved prototypes for four representative queries (A–D); each Query column shows the input image and the three columns on the right shows its retrieved prototypes. In these examples, EMN retrieves prototypes that appear visually clearer and more class-consistent, consistent with the improved explanation metrics reported in Sec. IV.

Our contributions are threefold:

• Brain-Inspired Architecture: We present a model that structurally reflects biological memory recall, leveraging continuous modern Hopfield networks [18] to achieve robust explainability and maintained accuracy across diverse datasets with varying characteristics.

• Novel Prototype Evaluation Protocol: We propose a comprehensive evaluation protocol that assesses visual similarity (LPIPS [14] and DISTS [15]), subclass alignment, and NMI [16].

• Cross-dataset Explanatory Robustness: EMN yields consistently high-quality explanations across diverse datasets, at the cost of a modest decrease in classification accuracy.

## II. RELATED WORK

Post-hoc explanation methods [20], [21] are widely used in XAI. Among them, attribution-based approaches [20], [21] generate explanations after prediction by highlighting influential input regions most associated with the output using saliency maps [20], attention analysis [22], and gradient-based attributions [21]. However, these explanations are external to the model’s decision process, often relying on local cues that may not faithfully reflect the model’s internal decision-making process or provide sufficient guidance to users.

In contrast, self-explainable models [23] embed interpretability mechanisms directly into the model, generating explanations jointly with predictions. Prototype-based models generate explanations by associating each input with prototypical examples, providing case-based justifications grounded in visual evidence. Early CNN-based methods such as ProtoP-Net [1], TesNet [24], and Deformable ProtoPNet [25] rely on patch-to-prototype similarity matching on feature maps, yielding localized explanations. Transformer-based variants like ProtoViT [8] and ProtoPFormer [7] improve classification accuracy and explanation quality, while ProtoVAE [10] learns class-specific prototypes that can be decoded back into input space for visual verification, and SENN [23] decomposes predictions into interpretable concept activations and their corresponding relevance scores.

Despite these advances, our experiments indicate that prior prototype-based methods exhibit dataset-specific limitations. Patch-based models struggle when local similarity between inputs and prototypes is weak, as in CIFAR-10 [12], often yielding poorly aligned explanations (See Figure 1). Latentprototype methods like ProtoVAE [10] mitigate this constraint but produce blurry reconstructions, reducing explainability in complex scenarios.

Overall, prior prototype-based methods differ in how they match inputs to prototypes (e.g., patch-level correspondence or latent reconstruction), yet their explanatory effectiveness can vary substantially with dataset characteristics, particularly when spatial correspondence is weak or when reconstructions become ambiguous.

Motivated by these limitations, we propose the EMN. Rather than relying solely on similarity-based scoring, EMN grounds prototype selection in memory-refined latent representations that jointly support reconstruction and classification, and selects prototypes with the highest contribution to these latents as explanations. By separating visual and categorical latents, EMN enables associative prototype retrieval and yields more consistent explanation quality across datasets while maintaining competitive predictive performance.

## III. ENGRAM MEMORY NETWORK

This section presents EMN, which grounds prototype selection in memory-refined latent representations. The key insight is that prototypes that contribute most to the latents—optimized for both reconstruction and classification—are expected to encode input-relevant information, yielding more consistent explanations across diverse dataset characteristics. The EMN is designed to emulate the neurocognitive process by which humans recognize objects and retrieve relevant memories for interpretation. Its architecture follows a three-stage structure comprising an encoder, an engram memory module, and a decoder. We first describe the neurocognitive basis of our approach (Section III-A), then detail the encoder (Section III-B), the engram memory module (Section III-C), the decoder (Section III-D), and the learning objectives (Section III-E).

## A. Neurocognitive Basis

Human object recognition integrates sensory evidence with stored visual knowledge through hierarchical processing in the visual system [26], [27]. The proposed EMN is inspired by this hierarchy and adopts a three-stage architecture, as illustrated in Figure 2(a–c).

Stage 1: Visual Feature Encoding (Figure 2(a)). Visual signals propagate from the retina to the primary visual cortex (V1). Along the ventral visual stream, representations become progressively more abstract through a hierarchy of visual areas (e.g., V1–V4) and culminate in the inferior temporal cortex (ITC), supporting object recognition [27], [28]. EMN models this process using an encoder f(·) that transforms images into visual latent representations (Figure 2(d) and Section III-B).

Stage 2: Associative Memory Activation (Figure 2(b)). In the brain, ITC activates associative memories—or engrams—through stimulus-driven and recurrent excitation [29]. Formed via Hebbian learning [30], engrams enable pattern completion: partial input can reactivate the full memory trace [31]. For example, seeing a cat may evoke related visual or verbal memories. EMN is motivated by this process using our modified update rule (Section III-C), where visual and categorical latent vectors iteratively interact with prototype memories. These interactions integrate three components: continuous input injection, recurrent memory feedback, and state decay, converging to stable representations that encode both perceptual and semantic information (Figure 2(e) and Section III-C).

Stage 3: Recognition and Reconstruction (Figure 2(c)). Converged activity in ITC informs object classification and memory recall, with feedback signals projecting to early visual areas such as V1. EMN captures this process via a decoder that reconstructs the input from the final latent state (Figure 2(f) and Section III-D).

Since memory-refined latents are designed to capture sufficient information for both reconstruction and classification, prototypes with high contribution to these latents are likely input-relevant, facilitating valid explanations regardless of dataset characteristics.

## B. Encoder

As illustrated in Figure 2(d), the encoder in EMN transforms visual input into latent representations that serve as the initial substrate for memory interaction and subsequent inference.

a) Input Encoding: EMN encodes input images through an encoder $f ( \cdot )$ , mimicking early visual processing along the ventral stream (Figure 2(a)). Given an input image x, the encoder produces a latent vector $\mathbf { h } = \sigma ( f ( \mathbf { x } ) ) \in \mathbb { R } ^ { d _ { 1 } }$ , where $f ( \cdot )$ denotes the encoding module. The activation function $\sigma ( \cdot )$ is biologically inspired and defined piecewise as follows: it outputs ϵx for $x \ < \ 0 ;$ returns x when $0 \leq x \leq 1 ;$ and computes $\epsilon x + 1 - \epsilon \mathrm { f o r } x > 1$ . This formulation ensures nonzero gradients across the domain while encouraging activations to concentrate near biologically plausible firing thresholds.

b) Prototype and Engram Initialization: To model associative memory, EMN initializes a set of N prototype-label pairs $\left( \mathbf { p } _ { \mathbf { x } , i } , p _ { y , i } \right)$ by randomly sampling from the training set, where $\mathbf { p } _ { \mathbf { x } , i }$ is the i-th prototype image, $p _ { y , i }$ is its corresponding class label, and subscripts x and y denote image and label modalities, respectively. These are encoded into memory vectors $\mathbf { m } _ { \mathbf { x } , i } = \sigma ( f ( \mathbf { p } _ { \mathbf { x } , i } ) ) \in \mathbb { R } ^ { d _ { 1 } }$ and $\mathbf { m } _ { y , i } = \sigma ( W _ { y } [ p _ { y , i } ] ) \in$ $\mathbb { R } ^ { d _ { 2 } }$ , where $W _ { y }$ is a class embedding matrix. The resulting set $\{ \mathbf { m } _ { \mathbf { x } , i } , \mathbf { m } _ { y , i } \} _ { i = 1 } ^ { N }$ forms the engram memory, which supports associative retrieval and pattern completion during inference.

## C. Engram Memory Module

To model associative memory dynamics observed in the brain (Figure 2(b) and Section III-A), we introduce the EMM as a module for an iterative process that refines latent states by integrating three components: (i) input injection from the encoder, (ii) memory-driven recurrent feedback, and (iii) leaky decay of the current state (Figure 2(e)). Given an encoded input $\textbf { h } \in \mathbb { R } ^ { d _ { 1 } }$ , the iterative process produces three outputs after $T$ steps: a refined visual latent vector $\mathbf { z } _ { \mathbf { x } } ^ { ( T ) } \in \mathbb { R } ^ { d _ { 1 } }$ used for reconstruction, a categorical latent vector $\mathbf { z } _ { y } ^ { ( T ) } \in \mathbb { R } ^ { d _ { 2 } }$ used for classification, and an activation vector $\mathbf { c } ^ { ( T ) } \in \Delta ^ { N }$ used to select explanatory prototypes. At each iteration $t = 1 , \dots , T$ the update rules are given by:

![](images/e8493037c6e20f187ecc0d1834987e7395356ada20903a28339103f6ace9a1da.jpg)  
Fig. 2. (a–c) Visual pathway from ventral stream to inferior temporal cortex (ITC), enabling associative memory and categorization with feedback to early visual areas. (d–f) EMN models this process via an encoder, memory-guided refinement, and a decoder.

$$
{ \bf z } _ { \bf x } ^ { ( 0 ) } = { \bf 0 } , \quad { \bf z } _ { y } ^ { ( 0 ) } = { \bf 0 } , \quad { \bf c } ^ { ( 0 ) } = \frac { { \bf 1 } } { N }\tag{1}
$$

$$
\mathbf { z } _ { \mathbf { x } } ^ { ( t ) } = \sigma \left( \alpha \mathbf { h } + ( 1 - 2 \alpha ) \mathbf { z } _ { \mathbf { x } } ^ { ( t - 1 ) } + \alpha M _ { \mathbf { x } } \mathbf { c } ^ { ( t - 1 ) } \right)\tag{2}
$$

$$
\mathbf { z } _ { y } ^ { ( t ) } = \sigma \left( ( 1 - \alpha ) \mathbf { z } _ { y } ^ { ( t - 1 ) } + \alpha M _ { y } \mathbf { c } ^ { ( t - 1 ) } \right)\tag{3}
$$

$$
\mathbf { c } ^ { ( t ) } = \phi \left( \beta _ { \mathbf { x } } M _ { \mathbf { x } } ^ { \top } \mathbf { z } _ { \mathbf { x } } ^ { ( t ) } + \beta _ { y } M _ { y } ^ { \top } \mathbf { z } _ { y } ^ { ( t ) } \right) ,\tag{4}
$$

where $\sigma ( \cdot )$ is the activation function from Section III-B, ϕ(·) denotes softmax, $\alpha ~ \in ~ ( 0 , 1 )$ controls the update rate, and $\beta _ { \mathbf { x } } , \beta _ { y } ~ > ~ 0$ are learnable coefficients. Here, $M _ { \mathbf { x } } ~ \in ~ \mathbb { R } ^ { d _ { 1 } \times N }$ and $\bar { M } _ { y } \in \mathbb { R } ^ { d _ { 2 } \times N }$ denote the memory matrices containing latent visual and label prototypes, respectively. The vector $\mathbf { c } ^ { ( t ) }$ reflects the activation levels of the $N$ prototypes based on the current latent state.

Each component of the update mechanism has a specific computational role. Equation 1 zero-initializes the latent states to begin refinement from a neutral point. In Equations 2 and 3, the visual latent vector $\mathbf { z _ { x } }$ and categorical latent vector $\mathbf { z } _ { y }$ are updated, respectively, via: (i) αh provides continuous input from the encoder (for $\mathbf { z _ { x } }$ only), coupling encoder input with memory refinement and enabling end-to-end training; (ii) $\alpha M _ { \mathbf { x } } \mathbf { c } ^ { ( t - 1 ) }$ or $\alpha M _ { y } \mathbf { c } ^ { ( t - 1 ) }$ contributes recurrent feedback weighted by memory activation; (iii) $( 1 - 2 \alpha ) \mathbf { z } _ { \mathbf { x } } ^ { ( t - 1 ) }$ or $( 1 -$ $\alpha ) \mathbf { z } _ { y } ^ { \overline { { ( t - 1 ) } } }$ represents state decay—the difference in decay rates reflects the presence or absence of encoder input. Components (ii), (iii), and the softmax in Equation 4 follow continuous modern Hopfield dynamics [18], while (i) is introduced to apply Hopfield dynamics in latent space.

Differences from [18]: While structurally related to continuous modern Hopfield network [18], this update mechanism differs in three key ways: latent states are zero-initialized (Eq. 1); bottom-up input h is injected at each step (Eq. 2); and the contribution weights $\beta _ { \mathbf { x } } , \beta _ { y }$ are learnable rather than fixed.

## D. Decoder

As shown in Figure 2(f), EMN generates three task-specific outputs—image reconstruction, classification, and explanation—based on the refined latent states $\mathbf { z } _ { \mathbf { x } } ^ { ( T ) } , \mathbf { z } _ { y } ^ { ( T ) }$ , and the engram activation vector $\mathbf { c } ^ { ( T ) }$ . Image reconstruction is performed by a decoder $g _ { \mathrm { r e c } } .$ which maps $\mathbf { z } _ { \mathbf { x } } ^ { ( T ) }$ to $\hat { { \bf x } } = g _ { \mathrm { r e c } } ( { \bf z } _ { \bf x } ^ { ( T ) } )$ , reflecting feedback from ITC to early visual areas. The decoder architecture is flexible and task-agnostic. Classification uses a linear head $g _ { \mathrm { p r e d } }$ applied to $\mathbf { z } _ { y } ^ { ( T ) } \colon \hat { \mathbf { y } } = \beta _ { y } \sigma ( W _ { y } ) \mathbf { z } _ { y } ^ { ( T ) }$ . This design reflects the alignment between categorical activation and class prediction. Explanations are derived from $\mathbf { c } ^ { ( T ) }$ by selecting the top-K activated prototypes, emulating memorybased associative retrieval. These decoding pathways instantiate perception, recognition, and memory retrieval, reflecting distinct cognitive functional roles. Because the same memoryrefined latent supports both reconstruction and classification, prototypes that contribute most to this latent are likely to encode input-relevant information, providing a principled basis for explanation validity across diverse domains.

## E. Learning Objectives

EMN is trained using a multi-objective loss that jointly optimizes classification, reconstruction, and explanation quality:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } + \lambda _ { \mathrm { s e p } } \mathcal { L } _ { \mathrm { s e p } } ,
$$

where each λ balances the contribution of the respective term. The classification loss ensures that the label latent vector predicts the correct class: $\mathcal { L } _ { \mathrm { c l s } } = \mathrm { C r o s s E n t r o p y } ( \hat { \mathbf { y } } , y )$ , encouraging class-consistent engram activations. To retain visual fidelity, the reconstruction loss minimizes the discrepancy between the input and reconstructed image: $\mathcal { L } _ { \mathrm { r e c } } ~ = ~ \| \mathbf { x } - \hat { \mathbf { x } } \| _ { 2 } ^ { 2 }$ . To mitigate prototype overuse and collapse (see Figure 1), EMN introduces two entropy-based losses that regulate memory utilization. The diversity loss promotes global usage of the memory space by maximizing entropy over the batch-averaged activation distribution: $\begin{array} { r } { \mathcal { L } _ { \mathrm { d i v } } = - \mathcal { H } ( \bar { \bar { \mathbf { c } } } ) = \sum _ { i = 1 } ^ { N } \bar { c } _ { i } } \end{array}$ log ¯c , where $\begin{array} { r c l } { \bar { \mathbf c } } & { = } & { \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \mathbf c _ { j } ^ { ( T ) } } \end{array}$ The separation loss encourages local selectivity by aligning each sample’s prototype entropy to a target: $\begin{array} { r } { \mathcal { L } _ { \mathrm { s e p } } = \left| \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \mathcal { H } ( \mathbf { c } _ { j } ^ { ( T ) } ) - \log K \right| } \end{array}$ . This guides each explanation to rely on a small number of relevant prototypes—typically around three—to balance richness and focus. Together, ${ \mathcal { L } } _ { \mathrm { d i v } }$ and $\mathcal { L } _ { \mathrm { s e p } }$ shape the softmax-driven memory dynamics to ensure explanations are diverse across inputs and selective within each instance, enabling EMN to mimic human-like associative recall.

## IV. EXPERIMENTS

We validate the effectiveness of the proposed EMN through comprehensive experiments.

Datasets. We evaluate on CIFAR-10 [12], CIFAR-100 [12], MNIST [11], HAM10000 [19] (medical dermatology), and CUB-200 [13] (fine-grained bird classification).

Model Architecture. Although EMN is compatible with various backbone architectures, we adopt Vision Transformer (ViT) [32] for all primary comparisons to maintain consistency with recent prototype-based models. The encoder uses DeiT-Small [33]<sup>2</sup> and the decoder uses a modified ViT architecture.

Training Process. Training proceeds in three stages: (1) the encoder is initialized with DeiT-Small weights pretrained on ImageNet [34]; (2) the encoder and decoder are jointly trained on the target dataset using classification and reconstruction losses without the engram memory module, for 30 epochs on CIFAR and 50 epochs on other datasets; (3) the full EMN is then trained for 50 epochs on all datasets.

Hyperparameters. We use N=2000 prototypes randomly sampled from the training set with K=3 for explanation. Latent dimensions are $d _ { 1 } { = } 4 0 9 6$ (visual) and $d _ { 2 } { = } 5 1 2$ (categorical). Memory refinement runs for T=24 iterations with update rate $\alpha { = } 0 . 0 5$ and $\varepsilon { = } 0 . 0 0 1$ in $\sigma ( \cdot )$ . Loss weights are $\lambda _ { \mathrm { c l s } } = \lambda _ { \mathrm { r e c } } = \lambda _ { \mathrm { d i v } } = \lambda _ { \mathrm { s e p } } = 1$ . Training uses AdamW [35] with $\mathrm { l r } { = } 1 0 ^ { - 5 }$ , weight decay 0.05, batch size 64. Data augmentation includes RandomHorizontalFlip and ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, and hue=0.1).

Baselines. We compare EMN against ProtoVAE [10], ProtoPNet [1], ProtoViT [8], and ProtoPFormer [7]. All baselines are trained and evaluated according to their official protocols, using the same data splits, preprocessing, and augmentation. All models are implemented in PyTorch [36].

Experimental Environment. All experiments were conducted on a single NVIDIA GPU (32GB VRAM) running

TABLE I  
VISUAL SIMILARITY (LPIPS [14] AND DISTS [15]) COMPARISON ACROSS DATASETS. RESULTS ARE REPORTED AS MEAN ± STD. $( \times 1 0 ^ { - 1 } ;$ LOWER IS BETTER; BOLD INDICATES BEST).
<table><tr><td>Model</td><td colspan="3">LPIPS [14]</td><td colspan="3"></td></tr><tr><td></td><td>top1</td><td></td><td>top3</td><td></td><td>top1</td><td>top3</td></tr><tr><td>(CIFAR-10 [12])</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProtoVAE [10] ProtoPNet [1]</td><td> $7 . 5 0 \pm 0 . 0 6$ </td><td></td><td> $7 . 5 0 \pm 0 . 0 7$ </td><td></td><td></td><td> $6 . 7 5 \pm 0 . 1 1 6 . 7 6 \pm 0 . 1 1$   $6 . 9 5 \pm 0 . 1 9 7 . 0 0 \pm 0 . 3 2 4 . 0 7 \pm 0 . 1 3 4 . 0 2 \pm 0 . 1 2$ </td></tr><tr><td>ProtoViT [8]</td><td> $6 . 2 6 \pm 0 . 1 0 \ 6 . 3 1 \pm 0 . 0 7$ </td><td></td><td></td><td></td><td></td><td> $3 . 6 7 \pm 0 . 0 3 3 . 6 9 \pm 0 . 0 3$ </td></tr><tr><td>ProtoPFormer</td><td></td><td></td><td></td><td></td><td></td><td> $[ 7 ] ~ 6 . 2 3 \pm 0 . 0 2 ~ 6 . 2 4 \pm 0 . 0 2 ~ 3 . 6 8 \pm 0 . 0 1 ~ 3 . 6 9 \pm 0 . 0 2$ </td></tr><tr><td>EMN (ours)</td><td></td><td></td><td></td><td></td><td></td><td> ${ 5 . 3 8 \pm 0 . 0 2 5 . 5 0 \pm 0 . 0 2 3 . 2 2 \pm 0 . 0 1 3 . 2 8 \pm 0 . 0 1 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(CIFAR-100 [12])</td><td></td><td></td><td></td><td></td><td></td><td> $7 . 2 7 \pm 0 . 0 6 7 . 2 7 \pm 0 . 0 6 6 . 3 2 \pm 0 . 0 7 6 . 3 2 \pm 0 . 0 7$ </td></tr><tr><td>ProtoVAE [10] ProtoPNet [1]</td><td></td><td></td><td></td><td></td><td></td><td> $6 . 4 9 \pm 0 . 0 8 6 . 5 7 \pm 0 . 0 8 3 . 8 3 \pm 0 . 0 4 3 . 8 8 \pm 0 . 0 3$ </td></tr><tr><td>ProtoViT [8]</td><td></td><td></td><td></td><td></td><td></td><td> $6 . 1 1 \pm 0 . 0 2 6 . 2 2 \pm 0 . 0 1 3 . 5 7 \pm 0 . 0 1 3 . 6 2 \pm 0 . 0 0$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td> $6 . 3 4 \pm 0 . 0 4 6 . 3 5 \pm 0 . 0 2 3 . 7 1 \pm 0 . 0 2 3 . 7 1 \pm 0 . 0 2$ </td></tr><tr><td>ProtoPFormer [7] EMN (ours)</td><td>5.66 ± 0.02</td><td></td><td> ${ \bf 5 . 8 5 \pm 0 . 0 1 }$ </td><td></td><td></td><td>3.38 ± 0.01 3.49 ± 0.01</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(MNIST [11])</td><td></td><td></td><td></td><td></td><td></td><td> $2 . 3 3 \pm 0 . 0 6 2 . 3 4 \pm 0 . 0 6 2 . 3 8 \pm 0 . 0 2 2 . 3 8 \pm 0 . 0 2$ </td></tr><tr><td>ProtoVAE [10] ProtoPNet [1]</td><td></td><td></td><td></td><td></td><td></td><td> $6 . 7 1 \pm 0 . 0 5 6 . 8 2 \pm 0 . 0 8 4 . 2 8 \pm 0 . 0 5 4 . 3 7 \pm 0 . 0 7$ </td></tr><tr><td>ProtoViT [8]</td><td></td><td></td><td></td><td></td><td></td><td>6.59 ± 0.06 6.73 ± 0.07 4.12 ± 0.02 4.20 ± 0.03</td></tr><tr><td>ProtoPFormer [7]</td><td></td><td></td><td></td><td></td><td></td><td> $3 . 0 6 \pm 0 . 0 5 3 . 0 5 \pm 0 . 0 3 2 . 2 2 \pm 0 . 0 1 2 . 2 2 \pm 0 . 0 1$ </td></tr><tr><td>EMN (ours)</td><td></td><td></td><td></td><td></td><td></td><td>1.70 ± 0.01 1.82 ± 0.01 1.73 ± 0.01 1.78 ± 0.01</td></tr><tr><td>(HAM10000 [19])</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProtoPNet [1]</td><td> $5 . 1 9 \pm 0 . 2 3 5 . 3 5 \pm 0 . 2 1$ </td><td></td><td></td><td></td><td></td><td> $3 . 4 3 \pm 0 . 0 8 3 . 4 8 \pm 0 . 0 7$ </td></tr><tr><td>ProtoViT [8]</td><td> $4 . 6 9 \pm 0 . 1 7$ </td><td></td><td></td><td></td><td>4.71 ± 0.16 3.15 ± 0.03</td><td> $3 . 2 0 \pm 0 . 0 3$ </td></tr><tr><td>ProtoPFormer [7]</td><td> $5 . 4 0 \pm 0 . 5 3 5 . 4 0 \pm 0 . 4 7$ </td><td></td><td></td><td></td><td></td><td> $3 . 6 1 \pm 0 . 4 4 3 . 6 4 \pm 0 . 3 8$ </td></tr><tr><td>EMN (ours)</td><td> $\mathbf { 3 . 5 0 \ : \pm ~ 0 . 0 2 ~ 3 . 5 9 \ : \pm ~ 0 . 0 1 }$ </td><td></td><td></td><td></td><td>2.40 ± 0.02</td><td> ${ \bf 2 . 4 6 \pm 0 . 0 2 }$ </td></tr><tr><td>(CUB-200 [13])</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProtoPNet [1]</td><td></td><td></td><td></td><td></td><td></td><td> $6 . 7 9 \pm 0 . 0 5 6 . 8 1 \pm 0 . 0 5 4 . 1 2 \pm 0 . 0 3 4 . 1 3 \pm 0 . 0 2$ </td></tr><tr><td>ProtoViT [8]</td><td></td><td></td><td></td><td></td><td></td><td> $6 . 8 2 \pm 0 . 0 0 ~ 6 . 8 7 \pm 0 . 0 0 ~ 3 . 9 7 \pm 0 . 0 0 ~ 4 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>ProtoPFormer [7]</td><td></td><td></td><td></td><td></td><td></td><td> $6 . 7 7 \pm 0 . 0 3 6 . 8 0 \pm 0 . 0 2 3 . 7 2 \pm 0 . 0 2 3 . 7 5 \pm 0 . 0 2$ </td></tr><tr><td>EMN (ours)</td><td>6.44 ± 0.02</td><td></td><td> ${ \bf 6 . 5 5 \pm 0 . 0 1 }$ </td><td></td><td> ${ \bf 3 . 5 5 \pm 0 . 0 1 }$ </td><td> ${ \bf 3 . 6 2 \pm 0 . 0 0 }$ </td></tr></table>

Ubuntu Linux. Training EMN on CIFAR-10 takes approximately 3 hours across all training stages.

## A. Visual Similarity

This section assesses the visual similarity between input queries and selected prototypes. Since explanations are intended for human users, the ideal evaluation would rely on human judgments; however, due to feasibility constraints, we employ proxy metrics—LPIPS [14] and DISTS [15]—that align with human perceptual judgments, while imperfect but scalable for large-scale evaluation.

As shown in Table I, EMN achieves the best visual similarity (lowest LPIPS [14] and DISTS [15]) across all five datasets, with an average improvement of 17.06% over the best baseline (ProtoPFormer [7]). These results indicate that EMN retrieves prototypes more perceptually aligned with input queries, providing more interpretable explanations as illustrated in Figure 1. ProtoVAE [10] is excluded from CUB-200 [13] and HAM10000 [19] as its decoded prototypes become uninformative and blurry on these datasets.

## B. Subclass Alignment

Representation learning aims to preserve the latent geometry of real-world objects so that semantically similar instances are placed in close proximity. Since latent structure is unobservable, we use hierarchical labels (superclass–subclass) as a proxy: e.g., “Pine” and “Oak” within “Tree” share more features than items from other superclasses. Models are trained using only superclass labels while subclass labels are withheld; at evaluation, we reveal subclass labels as hidden ground truth to measure prototype–query alignment. To evaluate whether models identify prototypes that are class-consistent and visually aligned with the query at a finer subclass level, we conduct a hierarchical matching experiment using CIFAR-100S, which reorganizes CIFAR-100 [12]’s 100 classes into 20 superclasses (5 subclasses each).<sup>3</sup> ProtoVAE [10] is excluded as its generated prototypes lack subclass labels. Results are summarized in Table II. Table II shows that EMN achieves 39.87% (top-1) and 36.07% (top-3) subclass alignment, 104.57% higher than the best baseline ProtoPFormer [7] (19.49%, 19.42%). This substantial improvement demonstrates that EMN effectively captures fine-grained visual features, enabling input-specific prototype selection rather than generic class-level matching.

TABLE II  
SUBCLASS ALIGNMENT (%) ON CIFAR-100S (HIGHER IS BETTER; CHANCE LEVEL IS 1%).
<table><tr><td>Method</td><td>top1 top3</td></tr><tr><td>ProtoVAE [10]</td><td></td></tr><tr><td>ProtoPNet [1]</td><td> $2 . 2 8 \pm 0 . 9 1$   $1 . 9 5 \pm 0 . 4 7$ </td></tr><tr><td>ProtoViT [8]  $1 5 . 9 3 \pm 6 . 5 5$ </td><td> $1 4 . 0 3 \pm 7 . 0 4$ </td></tr><tr><td>ProtoPFormer [7]</td><td> $1 9 . 4 9 \pm 1 . 6 0$   $1 9 . 4 2 \pm 1 . 3 8$ </td></tr><tr><td>EMN (ours)</td><td> ${ \bf 3 9 . 8 7 \pm 0 . 7 2 }$   ${ \bf 3 6 . 0 7 \pm 0 . 3 6 }$ </td></tr></table>

TABLE III

CLASSIFICATION ACCURACY (%) ACROSS ALL DATASETS (HIGHER IS BETTER; BOLD INDICATES BEST).
<table><tr><td>Model</td><td> $\operatorname { A c c } .$ </td><td>Model</td><td> $\operatorname { A c c } .$ </td></tr><tr><td>(CIFAR-10 [12])</td><td></td><td>(MNIST [11])</td><td></td></tr><tr><td>ProtoVAE [10]</td><td> $8 1 . 5 9 \pm 0 . 3 6$ </td><td>ProtoVAE [10]</td><td> $9 9 . 0 5 \pm 0 . 0 2$ </td></tr><tr><td>ProtoPNet [1]</td><td> $9 4 . 3 0 \pm 0 . 1 2$ </td><td>ProtoPNet [1]</td><td> $9 9 . 0 6 \pm 0 . 1 0$ </td></tr><tr><td>ProtoViT [8]</td><td> $9 7 . 6 3 \pm 0 . 0 8$ </td><td>ProtoViT [8]</td><td> $9 9 . 0 9 \pm 0 . 0 9$ </td></tr><tr><td>ProtoPFormer [7]</td><td> ${ \bf 9 7 . 8 5 \pm 0 . 0 6 }$ </td><td>ProtoPFormer [7]</td><td> ${ \bf 9 9 . 4 1 \pm 0 . 0 5 }$ </td></tr><tr><td>EMN (ours)</td><td> $9 6 . 3 9 \pm 0 . 2 7$ </td><td>EMN (ours)</td><td> $9 9 . 1 9 \pm 0 . 0 8$ </td></tr><tr><td>(CIFAR-100 [12])</td><td></td><td>(HAM10000 [19])</td><td></td></tr><tr><td>ProtoVAE [10]</td><td> $5 0 . 3 7 \pm 0 . 8 9$ </td><td>ProtoVAE [10]</td><td></td></tr><tr><td>ProtoPNet [1]</td><td> $6 8 . 6 3 \pm 0 . 3 5$ </td><td>ProtoPNet [1]</td><td> $6 2 . 9 2 \pm 2 . 8 4$ </td></tr><tr><td>ProtoViT [8]</td><td> $8 6 . 4 1 \pm 0 . 1 3$ </td><td>ProtoViT [8]</td><td> $8 0 . 4 9 \pm 0 . 6 8$ </td></tr><tr><td>ProtoPFormer [7]</td><td> ${ \bf 8 7 . 8 3 \pm 0 . 2 5 }$ </td><td>ProtoPFormer [7]</td><td> $6 6 . 9 5 \pm 0 . 0 0$ </td></tr><tr><td>EMN (ours)</td><td> $8 5 . 2 8 \pm 0 . 2 1$ </td><td>EMN (ours)</td><td> ${ \bf 8 8 . 5 1 \pm 1 . 0 2 }$ </td></tr><tr><td>(CUB-200 [13])</td><td></td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td></td><td></td><td></td></tr><tr><td>ProtoPNet [1]</td><td> $2 2 . 4 3 \pm 2 . 2 9$ </td><td></td><td></td></tr><tr><td>ProtoViT [8]</td><td> $\mathbf { 7 4 . 0 9 \ : \pm { \ : 0 . 2 4 } }$ </td><td></td><td></td></tr><tr><td>ProtoPFormer [7]</td><td> $5 0 . 5 1 \pm 1 . 6 1$ </td><td></td><td></td></tr><tr><td>EMN (ours)</td><td> $6 8 . 7 4 \pm 0 . 5 4$ </td><td></td><td></td></tr></table>

## C. Classification

Table III reports classification accuracy across all datasets. EMN achieves the highest average accuracy (87.62%) among all methods, outperforming the best baseline ProtoViT [8] (87.54%). Notably, on HAM10000 [19] (medical imaging),

TABLE IV  
NMI(X; P) [16] SCORES ACROSS ALL DATASETS $( \times 1 0 ^ { - 1 }$ ; HIGHER IS BETTER; BOLD INDICATES BEST).
<table><tr><td>Model</td><td>top1</td><td>top3</td></tr><tr><td>(CIFAR-10 [12])</td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td> $0 . 0 7 \pm 0 . 0 2$ </td><td> ${ \bf 2 . 4 1 \pm 1 . 3 7 }$ </td></tr><tr><td>ProtoPNet [1]</td><td> $0 . 0 7 \pm 0 . 1 0$ </td><td> $0 . 0 8 \pm 0 . 1 2$ </td></tr><tr><td>ProtoViT [8]</td><td> $0 . 6 0 \pm 0 . 0 8$ </td><td> $0 . 5 0 \pm 0 . 0 5$ </td></tr><tr><td>ProtoPFormer [7]</td><td> $0 . 5 5 \pm 0 . 0 5$ </td><td> $0 . 5 5 \pm 0 . 0 4$ </td></tr><tr><td>EMN (ours)</td><td> ${ \bf 1 . 2 6 \pm 0 . 0 2 }$ </td><td> $1 . 2 1 \pm 0 . 0 2$ </td></tr><tr><td>(CIFAR-100 [12])</td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td> $0 . 0 4 \pm 0 . 0 1$ </td><td> $0 . 0 6 \pm 0 . 0 8$ </td></tr><tr><td>ProtoPNet [1]</td><td> $0 . 2 9 \pm 0 . 0 1$ </td><td> $0 . 2 4 \pm 0 . 0 1$ </td></tr><tr><td>ProtoViT [8]</td><td> $0 . 4 9 \pm 0 . 0 3$ </td><td> $0 . 4 4 \pm 0 . 0 1$ </td></tr><tr><td>ProtoPFormer [7]</td><td> $0 . 4 4 \pm 0 . 0 3$ </td><td> $0 . 4 2 \pm 0 . 0 2$ </td></tr><tr><td>EMN (ours)</td><td> $\mathbf { 1 . 0 5 \ : \pm { \ : 0 . 0 4 } }$ </td><td> ${ \bf 0 . 9 7 \pm 0 . 0 3 }$ </td></tr><tr><td>(MNIST [11])</td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td> ${ \bf 3 . 0 8 \pm 0 . 0 6 }$ </td><td> ${ \bf 3 . 0 5 \pm 0 . 0 6 }$ </td></tr><tr><td>ProtoPNet [1]</td><td></td><td></td></tr><tr><td>ProtoViT [8]</td><td> $1 . 8 4 \pm 0 . 1 0$ </td><td> $1 . 3 5 \pm 0 . 1 0$ </td></tr><tr><td>ProtoPFormer [7]</td><td> $1 . 7 8 \pm 0 . 1 0$ </td><td> $1 . 7 3 \pm 0 . 0 8$ </td></tr><tr><td>EMN (ours)</td><td> $2 . 6 2 \pm 0 . 0 5$ </td><td> $2 . 4 9 \pm 0 . 0 3$ </td></tr><tr><td>(HAM10000 [19])</td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td></td><td></td></tr><tr><td>ProtoPNet [1]</td><td> $0 . 6 8 \pm 0 . 3 8$ </td><td> $0 . 6 3 \pm 0 . 3 5$ </td></tr><tr><td>ProtoViT [8]</td><td> $0 . 9 7 \pm 0 . 1 5$ </td><td> $0 . 7 8 \pm 0 . 1 0$ </td></tr><tr><td>ProtoPFormer [7]</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>EMN (ours)</td><td> ${ \bf 1 . 1 8 \pm 0 . 0 4 }$ </td><td> ${ \bf 1 . 0 9 \pm 0 . 0 4 }$ </td></tr><tr><td>(CUB-200 [13])</td><td></td><td></td></tr><tr><td>ProtoVAE [10]</td><td></td><td></td></tr><tr><td>ProtoPNet [1]</td><td> $0 . 7 0 \pm 0 . 0 3$ </td><td> $0 . 6 5 \pm 0 . 0 3$ </td></tr><tr><td>ProtoViT [8]</td><td> $1 . 3 8 \pm 0 . 0 1$ </td><td> $1 . 3 3 \pm 0 . 0 1$ </td></tr><tr><td>ProtoPFormer [7]</td><td> $1 . 3 8 \pm 0 . 0 3$ </td><td> $1 . 3 2 \pm 0 . 0 2$ </td></tr><tr><td>EMN (ours)</td><td> ${ \bf 1 . 4 1 \pm 0 . 0 3 }$ </td><td> ${ \bf 1 . 3 4 } \pm { \bf 0 . 0 2 }$ </td></tr><tr><td></td><td></td><td></td></tr></table>

EMN achieves the highest accuracy (88.51%), outperforming the best baseline by 8 percentage points.

## D. Prototype Information Content

To evaluate whether prototypes capture meaningful information, we compute $\mathrm { N M I } ( X ; P ) [ 1 6 ] ^ { 4 }$ , which measures how well prototypes preserve input-specific visual details. As shown in Table IV, EMN achieves the best NMI(X; P) [16] on four of five datasets, with an improvement of 42.42% over the best baseline (ProtoViT [8]). The exception is MNIST [11], where ProtoVAE [10] achieves higher NMI(X; P) [16] due to its reconstruction-based approach being well-suited to simple, low-variance images. Overall, these results indicate that EMN’s selected prototypes reflect individual query characteristics, aligning with its design goal of retrieving visually similar examples rather than generic class representatives.

## E. Sensitivity Analysis

Table V shows sensitivity analysis on CIFAR-10 [12]. Increasing the number of prototypes N (from 200 to 5000) consistently improves explanation quality: LPIPS [14] (5.75→5.34), DISTS [15] (3.39→3.18), and NMI(X; P) [16] (1.06→1.28), while classification accuracy remains stable (96.63–96.69%). This indicates that larger memory provides richer retrieval candidates. The target entropy K in $\mathcal { L } _ { \mathrm { s e p } }$ shows robust behavior across $K { \in } \{ 1 , 3 , 5 , 7 \}$ : accuracy 96.17– 96.73%, LPIPS [14] 5.40–5.50, DISTS [15] 3.22–3.33, and NMI(X; P) [16] 1.21–1.24. The categorical latent dimension $d _ { 2 }$ reveals a trade-off: as $d _ { 2 }$ increases from 128 to 1024, LPIPS [14] increases $( 5 . 3 8 {  } 5 . 4 1 )$ , DISTS [15] remains stable (3.21→3.22), and NMI(X; P) [16] slightly decreases (1.25→1.24), while accuracy slightly improves (96.57→96.72%). These results confirm that EMN is robust to hyperparameter choices, with $d _ { 2 }$ being the only source of a modest accuracy–explanation trade-off. We adopt N=2000, K=3, and $d _ { 2 } { = } 5 1 2$ as defaults.

![](images/95f45b4df4861b505b290dec588d8d64a627f924028c765c7ea0e268ad72edd6.jpg)  
Fig. 3. Examples of prototype-based explanations across datasets: (from top to bottom) MNIST [11], HAM10000 [19], and CUB-200 [13]. The Query A-D columns show input queries, and the Prototype A-D column groups display the corresponding top-3 retrieved prototypes.

TABLE V  
SENSITIVITY ANALYSIS ON CIFAR-10 [12]. N: NUMBER OF PROTOTYPES; K: TARGET ENTROPY FOR $\mathcal { L } _ { \mathrm { { S E P } } } ; d _ { 2 } \colon$ CATEGORICAL LATENT DIMENSION. ACC (%) HIGHER IS BETTER; LPIPS [14], DISTS [15] $( \times 1 0 ^ { - 1 } )$ LOWER IS BETTER; NMI [16] (×10<sup>−1</sup>) HIGHER IS BETTER.
<table><tr><td>Setting</td><td>Acc↑</td><td>LPIPS [14]↓</td><td>DISTS [15]↓</td><td> $\mathbf { N M I } ( X ; P ) \ [ \mathbf { 1 6 } ] \uparrow$ </td></tr><tr><td> $N { = } 2 0 0$ </td><td> $9 6 . 6 3 \pm 0 . 2 0$ </td><td> $5 . 7 5 \pm 0 . 0 2$ </td><td> $3 . 3 9 \pm 0 . 0 1$ </td><td> $1 . 0 6 \pm 0 . 0 4$ </td></tr><tr><td> $N { = } 5 0 0$ </td><td> $9 6 . 6 3 \pm 0 . 3 0$ </td><td> $5 . 5 9 \pm 0 . 0 2$ </td><td> $3 . 3 0 \pm 0 . 0 1$ </td><td> $1 . 1 7 \pm 0 . 0 2$ </td></tr><tr><td> $N { = } 2 0 0 0$ </td><td> $9 6 . 6 9 \pm 0 . 2 5$ </td><td> $5 . 4 0 \pm 0 . 0 2$ </td><td> $3 . 2 2 \pm 0 . 0 1$ </td><td> $1 . 2 1 \pm 0 . 0 3$ </td></tr><tr><td> $N { = } 5 0 0 0$ </td><td> $9 6 . 6 9 \pm 0 . 1 9$ </td><td> ${ \bf 5 . 3 4 } \pm { \bf 0 . 0 3 }$ </td><td> ${ \bf 3 . 1 8 \pm 0 . 0 2 }$ </td><td> ${ \bf 1 . 2 8 \pm 0 . 0 2 }$ </td></tr><tr><td> $K { = } 1$ </td><td> $9 6 . 1 7 \pm 0 . 2 4$ </td><td> $5 . 5 0 \pm 0 . 0 4$ </td><td> $3 . 3 3 \pm 0 . 0 2$ </td><td> ${ \bf 1 . 2 4 \pm 0 . 0 2 }$ </td></tr><tr><td> $K { = } 3$ </td><td> $9 6 . 6 9 \pm 0 . 2 5$ </td><td> ${ \bf 5 . 4 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 3 . 2 2 \pm 0 . 0 1 }$ </td><td> $1 . 2 1 \pm 0 . 0 3$ </td></tr><tr><td> $K { = } 5$ </td><td> $9 6 . 7 2 \pm 0 . 1 2$ </td><td> $5 . 4 1 \pm 0 . 0 3$ </td><td> ${ \bf 3 . 2 2 \pm 0 . 0 1 }$ </td><td> $1 . 2 1 \pm 0 . 0 4$ </td></tr><tr><td> $K { = } 7$ </td><td> ${ \bf 9 6 . 7 3 \pm 0 . 2 5 }$ </td><td> $5 . 4 1 \pm 0 . 0 1$ </td><td> ${ \bf 3 . 2 2 \pm 0 . 0 1 }$ </td><td> $1 . 2 1 \pm 0 . 0 3$ </td></tr><tr><td> $d _ { 2 } { = } 1 2 8$ </td><td> $9 6 . 5 7 \pm 0 . 2 6$ </td><td> ${ \bf 5 . 3 8 \pm 0 . 0 3 }$ </td><td> ${ \bf 3 . 2 1 \pm 0 . 0 1 }$ </td><td> $\mathbf { 1 . 2 5 \ : \pm { \ : 0 . 0 3 } }$ </td></tr><tr><td> $d _ { 2 } { = } 2 5 6$ </td><td> $9 6 . 6 2 \pm 0 . 2 5$ </td><td> $5 . 4 0 \pm 0 . 0 1$ </td><td> ${ \bf 3 . 2 1 \pm 0 . 0 1 }$ </td><td> $1 . 2 2 \pm 0 . 0 2$ </td></tr><tr><td> $d _ { 2 } { = } 5 1 2$ </td><td> $9 6 . 6 9 \pm 0 . 2 5$ </td><td> $5 . 4 0 \pm 0 . 0 2$ </td><td> $3 . 2 2 \pm 0 . 0 1$ </td><td> $1 . 2 1 \pm 0 . 0 3$ </td></tr><tr><td> $d _ { 2 } { = } 1 0 2 4$ </td><td> ${ \bf 9 6 . 7 2 \pm 0 . 2 1 }$ </td><td> $5 . 4 1 \pm 0 . 0 3$ </td><td> $3 . 2 2 \pm 0 . 0 1$ </td><td> $1 . 2 4 \pm 0 . 0 4$ </td></tr></table>

TABLE VI
<table><tr><td colspan="4">ABLATION STUDY ON LOSS COMPONENTS (CIFAR-10 [12]). ACC (%) HIGHER IS BETTER; LPIPS [14], DISTS  $[ 1 5 ] ( \times 1 0 ^ { - 1 } )$  LOWER IS BETTER; NMI [16]  $( \times 1 0 ^ { - 1 } )$  HIGHER IS BETTER.</td></tr><tr><td>Setting</td><td>Acc↑</td><td>LPIPS [14]↓</td><td> $\mathbf { D I S T S \ [ 1 5 ] \downarrow }$ </td><td> $\mathbf { N M I } ( X ; P ) \ [ \mathbf { 1 6 } ] \uparrow$ </td></tr><tr><td>EMN (ours)</td><td> $9 6 . 6 9 \pm 0 . 2 5$ </td><td> $5 . 4 0 \pm 0 . 0 2$ </td><td> ${ \bf 3 . 2 2 \pm 0 . 0 1 }$ </td><td> $1 . 2 1 \pm 0 . 0 3$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { r e c } }$ </td><td> $9 6 . 6 7 \pm 0 . 2 8$ </td><td> ${ \bf 5 . 3 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 3 . 2 2 \pm 0 . 0 1 }$ </td><td> $1 . 2 4 \pm 0 . 0 2$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { d i v } }$ </td><td> $9 6 . 4 4 \pm 0 . 3 3$ </td><td> $5 . 4 3 \pm 0 . 0 3$ </td><td> $3 . 3 1 \pm 0 . 0 2$ </td><td> $1 . 2 7 \pm 0 . 0 4$ </td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { s e p } }$ </td><td> $9 7 . 3 7 \pm 0 . 1 7$ </td><td> $5 . 4 8 \pm 0 . 0 2$ </td><td> $3 . 2 6 \pm 0 . 0 1$ </td><td> $1 . 1 7 \pm 0 . 0 1$ </td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e c } } , \mathcal { L } _ { \mathrm { d i v } }$ </td><td> $9 6 . 3 6 \pm 0 . 1 9$ </td><td> $5 . 4 1 \pm 0 . 0 4$ </td><td> $3 . 3 0 \pm 0 . 0 2$ </td><td> $1 . 2 7 \pm 0 . 0 5$ </td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e c } } , \mathcal { L } _ { \mathrm { s e p } }$ </td><td> ${ \bf 9 7 . 3 8 \pm 0 . 1 5 }$ </td><td> $5 . 5 7 \pm 0 . 0 2$ </td><td> $3 . 3 2 \pm 0 . 0 0$ </td><td> $1 . 1 6 \pm 0 . 0 2$ </td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { d i v } } , \mathcal { L } _ { \mathrm { s e p } }$ </td><td> $9 6 . 7 4 \pm 0 . 5 5$ </td><td> $5 . 8 2 \pm 0 . 1 2$ </td><td> $3 . 6 0 \pm 0 . 1 3$ </td><td> ${ \bf 1 . 4 9 \pm 0 . 2 0 }$ </td></tr><tr><td>w/o all</td><td> $9 7 . 1 5 \pm 0 . 1 7$ </td><td> $5 . 5 4 \pm 0 . 0 4$ </td><td> $3 . 4 3 \pm 0 . 0 2$ </td><td> $1 . 3 1 \pm 0 . 0 5$ </td></tr></table>

## F. Ablation Study

Table VI examines the contribution of each loss component. Individual removal of ${ \mathcal { L } } _ { \mathrm { d i v } }$ or $\mathcal { L } _ { \mathrm { s e p } }$ produces only modest changes, but removing both substantially degrades visual similarity (LPIPS [14] 5.40→5.82, DISTS [15] 3.22→3.60), demonstrating their synergistic effect. Notably, removing $\mathcal { L } _ { \mathrm { s e p } }$ alone improves accuracy to 97.37%, revealing an accuracy– explanation trade-off. Nevertheless, even under this setting, EMN still outperforms the best baseline in visual similarity (e.g., LPIPS [14] 5.82 vs. 6.23 for ProtoPFormer [7]). These results reveal that $\mathcal { L } _ { \mathrm { s e p } }$ controls the accuracy–explanation trade-off, while both losses together are essential for visual similarity. We adopt the full configuration to prioritize explanation quality.

## G. Examples of Prototype-based Explanations

Figure 3 presents examples of EMN’s prototype-based explanations across diverse datasets, including MNIST [11], HAM10000 [19], and CUB-200 [13]. For each input query (Query A–D columns; gray background), EMN retrieves the top-3 most similar prototypes from memory, displayed in the corresponding Prototype A–D columns (white background).

The selected prototypes share not only the same class label but also fine-grained visual features with the query. For MNIST [11] Query C, the retrieved prototypes match the query digit $\mathbf { \Delta } ^ { 6 6 } 5 ^ { , 9 }$ in stroke style—specifically, the curvature of the upper loop and overall slant. For HAM10000 [19] Query D, the prototypes share diagnostically relevant features such as the central dark pigmentation pattern and irregular border shape. For CUB-200 [13] Query A, all three prototypes depict birds clinging vertically to a tree trunk—matching not just the species but also the pose of the query image.

These observations qualitatively support the subclass alignment results in Section IV-B: EMN selects prototypes that capture instance-specific visual details beyond class-level similarity. This suggests that the memory-refined latent preserves fine-grained information, enabling more semantically meaningful explanations across domains.

## V. CONCLUSION

Existing prototype-based models often exhibit datasetdependent explanation quality, limiting their reliability across domains. We propose the EMN, which mitigates this limitation by grounding prototype selection in memory-refined latent representations that jointly support reconstruction and classification. EMN retrieves specific training instances as explanations; this instance-based retrieval resembles episodic memory retrieval in cognitive neuroscience [37], which involves recollecting specific events along with their contextual details. The experimental results highlight the potential of incorporating brain-inspired memory dynamics into deep learning systems to improve explainability with a modest accuracy trade-off. Limitations include the absence of formal convergence or stability analysis for the iterative update rule, reliance on proxy metrics rather than human evaluation studies, and baseline comparisons that follow each method’s official protocol without unified reimplementation. A promising direction is to extend EMN toward concept-level (more semantic) prototypes that bridge instance retrieval and higher-level abstraction; we leave this as future work.

## ACKNOWLEDGMENT

This paper was prepared with the assistance of Claude (Anthropic). The AI tool was used for writing assistance and code review.

## REFERENCES

[1] C. Chen, O. Li, A. Tao, A. J. Barnett, C. Rudin, and J. Su, “This looks like that: deep learning for interpretable image recognition,” in Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 2019, pp. 8930–8941.

[2] A. Di Marino, V. Bevilacqua, A. Ciaramella, I. De Falco, and G. Sannino, “Ante-hoc methods for interpretable deep models: A survey,” ACM Computing Surveys, vol. 57, no. 6, pp. 119:1–119:39, 2025.

[3] A. Narayanan and K. J. Bergen, “Prototype-based methods in explainable ai and emerging opportunities in the geosciences,” arXiv preprint arXiv:2410.19856, 2024.

[4] C. Rudin, “Stop explaining black-box machine learning models for highstakes decisions and use interpretable models instead,” Nature Machine Intelligence, vol. 1, no. 5, pp. 206–215, 2019.

[5] J. Bien and R. Tibshirani, “Prototype selection for interpretable classification,” Annals of Applied Statistics, vol. 5, no. 4, pp. 2403–2424, 2011.

[6] O. Li, H. Liu, C. Chen, and C. Rudin, “Deep learning for case-based reasoning through prototypes,” in AAAI, 2018.

[7] M. Xue, Q. Huang, H. Zhang, J. Hu, J. Song, M. Song, and C. Jin, “Protopformer: Concentrating on prototypical parts in vision transformers for interpretable image recognition,” in Proceedings of the 33rd Int’l Joint Conference on Artificial Intelligence (IJCAI 2024), 2024, pp. 1516– 1524.

[8] C. Ma, J. Donnelly, W. Liu, S. Vosoughi, C. Rudin, and C. Chen, “Interpretable image classification with adaptive prototype-based vision transformers,” arXiv preprint arXiv:2410.20722, 2024, submitted Oct 2024.

[9] M. Nauta, J. Trienes, S. Pathak, E. Nguyen, M. Peters, Y. Schmitt, J. Schlotterer, M. van Keulen, and C. Seifert, “From anecdotal evidence¨ to quantitative evaluation methods: A systematic review on evaluating explainable AI,” ACM Computing Surveys, vol. 55, no. 13s, pp. 1–42, 2023.

[10] S. Gautam, A. Boubekki, S. Hansen, S. A. Salahuddin, R. Jenssen, M. M.-C. Hohne, and M. Kampffmeyer, “Protovae: A trustworthy¨ self-explainable prototypical variational model,” in Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022, pp. 31 700– 31 713.

[11] Y. LeCun, C. Cortes, and C. Burges, “Mnist handwritten digit database,” ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, vol. 2, 2010.

[12] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, Tech. Rep., 2009.

[13] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” in Technical Report CNS-TR-2011-001, 2011.

[14] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595.

[15] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 43, no. 5, pp. 1224–1236, 2021.

[16] T. M. Cover and J. A. Thomas, Elements ofInformation Theory, 2nd ed. Wiley-Interscience, 2006.

[17] J. Liu, H. Zhang, T. Yu, L. Ren, D. Ni, Q. Yang, B. Lu, L. Zhang, N. Axmacher, and G. Xue, “Transformative neural representations support long-term episodic memory,” Science Advances, vol. 7, no. 41, p. eabg9715, 2021.

[18] H. Ramsauer, B. Schafl, and et al., “Hopfield networks is all you need,”¨ in Proc. of the Int’l Conference on Learning Representations (ICLR), 2021, arXiv:2008.02217.

[19] P. Tschandl, C. Rosendahl, and H. Kittler, “The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions,” Scientific Data, vol. 5, p. 180161, 2018.

[20] K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep inside convolutional networks: Visualising image classification models and saliency maps,” in arXiv preprint arXiv:1312.6034, 2014.

[21] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 618–626.

[22] S. Jain and B. C. Wallace, “Attention is not explanation,” in Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019, pp. 3543–3556.

[23] D. Alvarez-Melis and T. S. Jaakkola, “Towards robust interpretability with self-explaining neural networks,” in Advances in Neural Information Processing Systems (NeurIPS 2018), 2018, pp. 7786–7795.

[24] J. Wang, H. Liu, X. Wang, and L. Jing, “Interpretable image recognition by constructing transparent embedding space,” in ICCV, 2021, pp. 895– 904.

[25] J. Donnelly, A. J. Barnett, and C. Chen, “Deformable protopnet: An interpretable image classifier using deformable prototypes,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 265–10 275.

[26] D. J. Felleman and D. C. Van Essen, “Distributed hierarchical processing in the primate cerebral cortex,” Cerebral Cortex, vol. 1, no. 1, pp. 1–47, 1991.

[27] J. J. DiCarlo, D. Zoccolan, and N. C. Rust, “How does the brain solve object recognition?” Neuron, vol. 73, no. 3, pp. 415–434, 2012.

[28] D. H. Hubel and T. N. Wiesel, “Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex,” Journal of Physiology, vol. 160, no. 1, pp. 106–154, 1962.

[29] K. Kar, J. Kubilius, K. Schmidt, E. B. Issa, and J. J. DiCarlo, “Evidence that recurrent circuits are critical to the ventral stream’s execution of core object recognition behavior,” Nature Neuroscience, vol. 22, no. 6, pp. 974–983, 2019.

[30] D. O. Hebb, The Organization of Behavior: A Neuropsychological Theory. New York, NY: John Wiley & Sons, 1949.

[31] S. A. Josselyn, S. Kohler, and P. W. Frankland, “Finding the engram,”¨ Nature Reviews Neuroscience, vol. 16, no. 9, pp. 521–534, 2015.

[32] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021.

[33] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation through attention,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 10 347–10 357.

[34] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[35] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), 2019.

[36] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 2019, pp. 8024–8035.

[37] E. Tulving, “Episodic and semantic memory,” in Organization of Memory, E. Tulving and W. Donaldson, Eds. Academic Press, 1972, pp. 381–403.