# Type-IV Code Clone Detection via Layer-Wise Non-Contrastive Representation Learning

Luciano Marchezan, Kevin Delcourt, Eugene Syriani, and Houari Sahraoui

Universite de Montr´ eal. Montreal, QC, Canada´

Email: {luciano.augusto.marchezan.de.paula, kevin.delcourt}@umontreal.ca {syriani, sahraouh}@iro.umontreal.ca

Abstract—Software clones are fragments of code that are similar or functionally equivalent to each other. They pose significant challenges for maintenance, refactoring, and bug detection. Detecting Type-IV clones, which are semantically equivalent but may differ syntactically, is particularly difficult for traditional token- or syntax-based methods. Recent machine learning approaches rely on contrastive learning, which requires careful negative sampling and can introduce bias. In this paper, we propose LWVIC4Code, a non-contrastive representation learning approach specifically designed for Type-IV clone detection. Building on the Variance-Invariance-Covariance Regularization (VICReg) framework and prior layer-wise VICReg training, LWVIC4Code introduces cross-layer consistency regularization and depth-dependent layer weighting to progressively refine semantic information across transformer layers, producing robust and discriminative code representations. We conduct an empirical study comparing LWVIC4Code against a contrastive learning baseline and zero-shot large language models on Python (Kamino) and multi-language (GPTCloneBench) datasets. Results show that LWVIC4Code achieves competitive or superior performance without negative samples, benefits from layer-wise supervision, and generalizes effectively from Python to other languages, particularly Java and C#. These results demonstrate that non-contrastive, layer-wise representation learning is a promising direction for robust semantic code clone detection.

Index Terms—semantic code similarity; clone detection; representation Learning; non-contrastive learning; software maintenance

## I. INTRODUCTION

Software clones, or similar fragments of code within or across software systems, are a common phenomenon in software development [1]. Detecting clones is critical for maintenance tasks such as code reuse, refactoring, bug detection, and quality assurance [2]. For instance, over 50% of files in two major open-source autonomous driving projects contain code clones, a substantial fraction of which are associated with previously reported bugs [3].

Code clones are typically categorized into four types [4], [5]: Types I–III represent exact or near-identical fragments with syntactic similarity, while Type-IV clones are semantic clones, i.e., functionally equivalent fragments that may differ entirely in syntax or structure. Detecting Type-IV clones requires reasoning about program semantics rather than surfacelevel syntactic patterns, which limits the effectiveness of traditional text- or token-based methods, especially when implementations differ in control-flow strategies, library usage, or algorithmic approach [6]–[8].

In this context, the growing adoption of large language models (LLMs) for code generation further amplifies the importance of clone detection, particularly for Type-IV clones. Recent studies indicate that AI-assisted programming is rapidly becoming standard practice in modern development workflows [9]. While LLMs improve productivity, they also introduce new challenges as the generated code may replicate patterns, logic, or entire solutions seen during training, sometimes closely resembling existing open-source implementations [10]. Such duplication is often not syntactically identical but semantically equivalent, making it difficult to detect using traditional clone detection techniques. This raises concerns related to code maintainability, redundancy, and licensing compliance when generated code inadvertently mirrors copyrighted sources [11]. Consequently, robust Type-IV clone detection is increasingly essential for identifying such semantic duplication across diverse implementations, programming styles, and programming languages.

Early research on semantic clone detection explored behavioral equivalence via program transformations [7], whereas more recent studies leverage machine learning (ML) to learn semantic representations of code [12], [13]. Pretrained code embedding models such as CodeBERT [14] and CodeT5 [15] generate vector representations that capture structural and semantic properties of code. These models can be finetuned for downstream tasks such as clone detection. These methods allow models to leverage large amounts of labeled code, typically using contrastive objectives to bring semantically similar fragments closer in embedding space while pushing dissimilar fragments apart [16]. However, contrastive approaches require careful selection of negative samples, which can introduce biases and complicate training [17].

Non-contrastive representation learning provides an alternative by enforcing constraints on the embedding space itself, without the need for negative pairs [18]. Variance-Invariance-Covariance Regularization (VICReg) [19] is a prominent example, optimizing three complementary properties: (i) invariance between different views of the same sample, (ii) variance across embedding dimensions to prevent collapse, and (iii) covariance between features to capture diverse aspects of the input. These properties make VICReg particularly well-suited for Type-IV clone detection, where semantic equivalence exists despite syntactic differences.

Building on this idea, we propose LWVIC4Code, a noncontrastive representation learning approach specifically designed for Type-IV clone detection. Inspired by the layerwise VICReg formulation of Datta et al. [20], our approach applies VICReg objectives at multiple transformer layers and optimizes them jointly through end-to-end training. Building on this foundation, we introduce a cross-layer consistency regularizer and depth-dependent layer weighting to encourage the progressive refinement of semantic information throughout the network. This leads to the introduction of three novel components as part of the LWVIC4Code architecture. The resulting representations are robust and discriminative, making them suitable for detecting semantically equivalent yet syntactically diverse code fragments.

We conduct an empirical study to evaluate the effectiveness of LWVIC4Code for Type-IV clone detection, comparing it with a contrastive learning model (CodeBERT<sub>CL</sub> [21]) and zero-shot LLMs. Specifically, we investigate (i) whether LWVIC4Code achieves comparable or superior performance without negative samples, (ii) the benefits of layer-wise supervision over traditional VICReg, (iii) which aspects of the novel components contribute the most to performance gains, and (iv) the generalization of models trained on Python-only data to other programming languages. Our evaluation uses two benchmark datasets: Kamino, a large Python-only dataset of Type-IV clones [22], and GPTCloneBench, a smaller multilanguage dataset covering Python, Java, C#, and C [23]. Performance is assessed using F -score and Matthews Correlation Coefficient (MCC) [24]. Statistical significance is evaluated using p-values corrected for multiple comparisons via the Benjamini-Hochberg procedure [25].

Results demonstrate that LWVIC4Code achieves strong performance for Type-IV clone detection across multiple programming languages and datasets. For example, on GPT-CloneBench, LWVIC4Code reaches $F _ { 1 }$ scores of 0.977 on C#, 0.968 on Java, and 0.920 on Python, with corresponding MCC values of 0.957, 0.938, and 0.840. These results outperform the contrastive baseline CodeBERT<sub>CL</sub> (e.g., $F _ { 1 }$ 0.899 and MCC 0.794 on C#) while avoiding the need for negative sampling. Overall, LWVIC4Code achieves higher performance than the original VICReg objective on GPTCloneBench $( p \ < \ 0 . 0 5 )$ , with the ablation study demonstrating that both layer-wise weighting and cross-layer consistency contribute to improving representation quality. Despite this, the last-layer-only variant achieves better results in specific scenarios.

In addition, LWVIC4Code outperforms zero-shot LLMbased approaches, achieving significantly higher performance than GPT-OSS and Qwen and comparable performance to DeepSeek-r1. These results demonstrate that a dedicated non-contrastive representation learning approach can provide more reliable and efficient semantic code representations than general-purpose LLMs for Type-IV clone detection, while requiring substantially less inference cost. The main contributions of this paper are:

i) We introduce VIC4Code, a non-contrastive representation learning approach that adapts the VICReg objective to code representation learning, enabling the learning of semantic embeddings without requiring negative samples;

ii) Building on prior layer-wise VICReg formulations, we propose LWVIC4Code, a transformer-based approach for Type-IV clone detection that introduces cross-layer consistency regularization and depth-dependent layer weighting, enabling end-to-end learning of hierarchical code representations and improving the robustness and discriminative power of code embeddings;

iii) A systematic empirical evaluation of LWVIC4Code for Type-IV clone detection, demonstrating that it outperforms a state-of-the-art contrastive learning baseline, improves over the standard VICReg objective, and achieves competitive or superior performance compared to zeroshot LLM-based approaches across multiple programming languages and datasets;

iv) An ablation study of the LWVIC4Code components, demonstrating that layer-wise weighting and crosslayer consistency are important contributors to performance improvements, with layer-wise weighting providing the largest impact;

v) An analysis of the impact of training data composition for LWVIC4Code, comparing Python-only vvs. multi-language training, demonstrating that representations learned from Python transfer effectively to Java and C#, though less to syntactically divergent languages such as C;

vi) A set of models for Type-IV clone detection: two variants of the complete LWVIC4Code (Python-only and multi-language), three variants without each novel component point, and one variant of VIC4Code.

The remainder of the paper is structured as follows. Section II presents background on semantic clone detection and representation learning; Section III describes the model architectures and non-contrastive training strategies; Section IV details the experimental protocol and results; Section V provides analysis and limitations; Section VI reviews related work; and Section VII concludes the paper.

## II. BACKGROUND

In this section, we outline the main challenges motivating our work, introducing key related concepts.

## A. Challenges and Motivation

Over the years, different definitions and classifications of code clones have been proposed [4], [5]. The most widely accepted taxonomy distinguishes four types of clones: (1) Types I–III (textual and syntactic clones): near-identical or slightly modified code fragments that share substantial syntactic similarity, ranging from exact copies to variations through insertions or deletions; and (2) Type-IV (semantic clones): functionally equivalent fragments that may differ entirely in syntax or structure, making them the most challenging to detect.

Type-IV clones are difficult to identify because their similarity lies in program behavior rather than structure. Fig. 1 illustrates two Python implementations of a function that computes the row sums of a matrix. Although both return the same result given the same input, their syntactic representations share almost no overlap (e.g., one uses nested loops while the other uses ‘map‘), making them hard to detect using traditional text- or token-based techniques [6]. Although this example is relatively simple to detect, real-world clones can be far more complex and require significant time and effort to identify, resources that companies could otherwise invest in more productive activities [26].

![](images/3b6c628a26810299c2b432b2a9daa988eea1244bd3d6ba9c10536036049c7946.jpg)  
Fig. 1: Two functionally equivalent but syntactically different functions in Python

The increasing use of LLMs for code generation [9] further exacerbates these challenges, as generated code often introduces semantically equivalent implementations that differ substantially in structure [10]. This shift moves clone creation from explicit copy-paste reuse to implicit, model-generated duplication, which can propagate at scale across projects and developers. Consequently, Type-IV clone detection becomes even more critical in modern development practice [11].

Early work explored semantic clone detection through program transformations and behavioral analysis [7]. More recently, ML techniques have been adopted to capture deeper semantic relationships between code fragments [12], [13]. These approaches typically rely on learning vector representations of code, known as code embeddings, which aim to capture structural and semantic properties of programs [27]. Once learned, such embeddings can be used to measure similarity between code fragments and identify potential clones [28], [29].

Despite these advances, accurately detecting Type-IV clones using traditional methods remains challenging [30], as semantic similarity can manifest through diverse implementation strategies, control-flow structures, or library usage patterns [7]. Rule-based or token-level approaches often fail to capture these high-level behavioral patterns. Consequently, representation learning is a promising direction, as it enables models to automatically transform raw code into structured, informative, and low-dimensional feature spaces [31]. This allows models to capture semantic properties of programs beyond superficial syntactic similarity. In this context, representation learning has emerged as an effective paradigm for learning such representations without requiring large amounts of labeled data, as demonstrated in domains such as computer vision [20].

In source code representation learning, however, contrastive learning methods have become popular [16], [21], [32]. These methods bring semantically related code fragments closer in the embedding space while pushing unrelated fragments apart [16]. However, contrastive approaches require large numbers of negative samples and careful sampling strategies, which can introduce biases and increase training complexity [17]. Moreover, selecting negative examples for semantic clone detection is non-trivial. Common strategies include using Types I–III clones as negatives [23], choosing syntactically similar but semantically different fragments [33], or selecting completely unrelated code [34]. Each approach has tradeoffs, as overly dissimilar negatives may lead models to rely on trivial cues, while syntactically similar negatives require careful curation and may lead to overfitting.

An alternative approach is non-contrastive learning [18]. Rather than relying on negatives, these methods enforce constraints on the embedding space itself to learn invariant representations. Variance-Invariance-Covariance Regularization (VICReg) [19], originally proposed for vision tasks, encourages representations to remain invariant across different views while maintaining variance and decorrelation. These properties are particularly appealing for Type-IV clones, where behavioral equivalence may manifest through diverse syntax, making non-contrastive learning a promising candidate for semantic code analysis [35].

Despite their benefits, non-contrastive methods require a large amount of quality data for effective training and validation. Traditional benchmarks such as BigCloneBench [36] are primarily syntactic, lack balanced negatives, and do not fully capture Type-IV behavior [37], [38]. Other large-scale datasets, such as CodeNet [39], CodeSearchNet [40], and BigCodeBench [41], have significantly advanced research in ML for code. These datasets support tasks such as code generation, summarization, and retrieval. However, they were not specifically designed for clone detection, and therefore lack explicit labels or guarantees of semantic equivalence between code fragments.

More recently, datasets targeting Type-IV clones have begun to emerge. For example, GPTCloneBench [23] contains Type-IV clones across multiple programming languages, including Python, Java, C#, and C. The dataset was generated by prompting GPT with code fragments from Semantic-CloneBench [42] and subsequently refined through manual curation, tool-assisted filtering, functionality testing, and automated validation. While valuable, GPTCloneBench remains relatively small for Type-IV clones (≈ 25k pairs across all programming languages) and may not provide sufficient diversity to train robust representation learning models. Another prominent dataset, Kamino [22], was derived using multiple LLMs to generate code from BigCodeBench entries, followed by extensive testing and filtering. Kamino contains a larger number of Type-IV clones (≈ 79k pairs), making it particularly suitable for training non-contrastive representation learning methods.

Finally, clone detection is a practical, development-oriented challenge [43]. Solutions must be suitable for real-time detection during development. Thus, clone detection mechanisms based on LLMs are typically resource-intensive, whereas lightweight embedding-based models, such as representation learning, offer faster, more efficient detection and easier integration into IDEs. Furthermore, while LLMs provide promising capabilities for semantic reasoning over code, their effectiveness for clone detection remains an active area of research [21], [44].

In summary, two main challenges motivate this study:

(1) Learning semantic representations of code: detecting Type--IV clones requires models that capture behavioral equivalence across highly diverse implementations, a challenge amplified by the current growing use of LLM-generated code in practice; and (2) Limitations of contrastive learning: contrastive methods depend on high-quality negative samples, which are difficult to define for semantic clones and can introduce bias and additional training complexity.

These challenges motivate the need for a non-contrastivebased approach that can learn robust semantic representations without relying on negative sampling.

## B. Code Embeddings for Clone Detection

Pretrained models such as CodeBERT [14] and CodeT5 [15] generate code embeddings by training on large code corpora using self-supervised objectives like masked language modeling. These embeddings capture both structural and semantic patterns in code, making them a strong foundation for various code understanding tasks. In the context of Type-IV clone detection, embeddings are typically used as feature representations for code fragments. Given a pair of code snippets $c _ { 1 }$ and $c _ { 2 }$ , the model produces embeddings $z _ { 1 }$ and $z _ { 2 }$ . The similarity between these embeddings is commonly computed using cosine similarity:

$$
\sin ( z _ { 1 } , z _ { 2 } ) = { \frac { z _ { 1 } \cdot z _ { 2 } } { \| z _ { 1 } \| \| z _ { 2 } \| } } .\tag{1}
$$

A threshold θ is then applied to determine whether the pair constitutes a clone:

$$
\left( c _ { 1 } , c _ { 2 } \right) { \mathrm { ~ i s ~ a ~ c l o n e ~ i f ~ } } \sin ( z _ { 1 } , z _ { 2 } ) \geq \theta .\tag{2}
$$

Choosing an appropriate θ is critical, as it directly affects false positives and false negatives. In practice, it is often selected empirically by evaluating performance metrics (e.g., $F _ { 1 }$ , MCC) across a range of thresholds or using percentilebased approaches on similarity distributions [33].

Pretrained embeddings can be finetuned on labeled clone datasets to improve performance, either using single-language Type-IV clones [22] or combining multiple languages [21], [23]. However, such approaches typically rely on contrastive signals or explicitly labeled negative examples to distinguish semantically similar from dissimilar code. For Type-IV clones, constructing high-quality negatives is particularly challenging and may introduce bias or degrade performance. These limitations motivate the exploration of non-contrastive methods, which learn robust semantic representations without requiring negative examples, as discussed in the next section.

## III. NON-CONTRASTIVE REPRESENTATION LEARNINGFOR TYPE-IV CLONE DETECTION

This section describes the non-contrastive representation learning strategies proposed in this work for detecting Type-IV code clones. Our approach builds on VICReg [19] to capture code semantics through embeddings, adapting it from its original focus on image embeddings.

Algorithm 1 VIC4Code for Type-IV clone detection   
1: Initialize encoder E, projector $P ,$ and optimizer   
2: for epoch = 1 to $N _ { \mathrm { e p o c h s } }$ do   
3: for each batch $( c _ { 1 } , c _ { 2 } ) \sim \mathcal { D }$ do   
4: Encode: $z _ { 1 } = E { \bigl ( } c _ { 1 } { \bigr ) } , z _ { 2 } = E { \bigl ( } c _ { 2 } { \bigr ) }$   
5: Project: $x _ { 1 } = P ( z _ { 1 } ) , x _ { 2 } = P ( z _ { 2 } )$   
6: Compute VICReg loss $\mathcal { L }$   
7: Backpropagate $\mathcal { L }$ and update parameters   
8: Save final encoder weights

## A. VIC4Code: Non-Contrastive Representation Learning

VIC4Code is a transformer-based encoder that leverages non-contrastive representation learning to capture semantic similarity in Type-IV clones. The model operates on two views (i.e., two clones) of the same code fragment, enforcing three complementary constraints on the embedding space: (i) Invariance: intuitively, embeddings of semantically equivalent code fragments should be close, even if their syntax differs; (ii) Variance: each embedding dimension should maintain variability to distinguish non-clones, preventing collapse; and (iii) Covariance: embedding dimensions should capture complementary semantic features, reducing redundancy Formally, let $\ Z = \{ x _ { 1 } , x _ { 2 } \}$ denote the projected embeddings of a batch of code fragment pairs. The VICReg loss combines three complementary terms:

$$
\mathcal { L } = \lambda \mathcal { L } _ { \mathrm { i n v } } + \mu \mathcal { L } _ { \mathrm { v a r } } + \nu \mathcal { L } _ { \mathrm { c o v } } ,\tag{3}
$$

where each term corresponds to one of the constraints (invariance, variance, and covariance) and is weighted by hyperparameters $( \lambda , \mu , \nu )$ . The encoder $E$ produces latent embeddings $z ,$ which are projected via a learnable projector $P$ into a space suitable for VICReg optimization. In addition, the final-layer representation is obtained using the CLS token embedding produced by the encoder.

The training procedure of VIC4Code is summarized in Algorithm 1. The encoder $E$ (line 1) is a transformer-based model with a tokenizer capable of converting raw source code into embeddings, such as CodeBERT [14]. Along with the learnable projector $P$ and the optimizer, the model is initialized before training begins. For each epoch (lines 2– 10), batches of code fragment pairs $\left( c _ { 1 } , c _ { 2 } \right)$ are sampled from the dataset $\mathcal { D }$ (line 3). Each fragment is encoded into latent embeddings $z _ { 1 }$ and $z _ { 2 }$ by the encoder (line 4), and subsequently projected into a higher-dimensional space via $P$ to obtain $x _ { 1 }$ and $x _ { 2 }$ (line 5). The VICReg loss $\mathcal { L }$ is then computed over the projected embeddings (line 6), enforcing the invariance, variance, and covariance constraints. This loss is backpropagated through both the encoder and projector to update all model parameters (line 7). After all epochs are completed, the final encoder weights are saved (line 11) and can be used to generate embeddings for downstream Type-IV clone detection tasks.

## B. LWVIC4Code: Layer-Wise VICReg

Applying the VICReg loss only at the final encoder layer, as in VIC4Code, may miss semantic information captured at intermediate layers. Transformer-based encoders encode hierarchical representations [20]. In this context, early layers capture syntactic patterns, while deeper layers represent higher-level semantics. This observation motivates our layer-wise VICReg approach, which propagates the loss across multiple depths to encourage semantically meaningful embeddings at every layer.

![](images/a6ebf560e891cba0729752ce429d7e482e3898b61f90f4691e73e39b7a2a9028.jpg)  
Fig. 2: LWVIC4Code architecture.

Inspired by the layer-wise VICReg formulation proposed by Datta et al. [20], we extend VIC4Code by applying VICReg objectives at multiple transformer depths. However, unlike the original layer-wise formulation, which optimizes layers using local objectives and layer-local gradient updates, LWVIC4Code is trained end-to-end, allowing gradients from all layer-wise objectives to jointly update the encoder parameters, while the stop-gradient operation is restricted to the previous-layer representation within the cross-layer regularizer. Building on this foundation, we introduce two additional mechanisms specifically designed for code representation learning: (i) a cross-layer consistency regularizer that encourages smooth semantic refinement across layers and (ii) depth-dependent weighting that progressively emphasizes higher-level semantic representations captured by deeper layers.

In summary, we extend VIC4Code by: (i) optimizing representations across all transformer layers; (ii) enforcing crosslayer consistency; and (iii) applying layer-wise weighting. Fig. 2 illustrates the overall LWVIC4Code architecture, including the three novel architectural components that lead to the final LWVIC4Code loss calculation. Each component point is described next.

1) Optimizing representations across layers: Let $\boldsymbol { h } ^ { ( l ) }$ ∈ $\mathbb { R } ^ { \bar { T } \times d }$ denote the hidden representation produced by transformer layer l, where T is the number of input tokens and d is the hidden dimension. Unlike VIC4Code, which relies on the final-layer CLS representation, LWVIC4Code applies masked mean pooling at every transformer layer to obtain layer-specific representations:

$$
p ^ { ( l ) } = \frac { \sum _ { t = 1 } ^ { T } m _ { t } h _ { t } ^ { ( l ) } } { \sum _ { t = 1 } ^ { T } m _ { t } } ,\tag{4}
$$

where $m _ { t }$ denotes the attention mask indicating valid tokens. This operation excludes padding tokens while preserving information from all code tokens. A layer-specific projector $g ^ { ( l ) }$ then maps the pooled representation to the embedding space $z ^ { ( l ) }$ :

$$
z ^ { ( l ) } = g ^ { ( l ) } ( p ^ { ( l ) } ) .\tag{5}
$$

At each layer, the VICReg loss encourages three properties in the embedding space: invariance between semantically similar code fragments, sufficient variance across dimensions to prevent collapse, and reduced covariance between dimensions to capture complementary features:

$$
\begin{array} { r } { \mathcal { L } ^ { ( l ) } = \lambda \mathcal { L } _ { \mathrm { i n v } } ^ { ( l ) } + \mu \mathcal { L } _ { \mathrm { v a r } } ^ { ( l ) } + \nu \mathcal { L } _ { \mathrm { c o v } } ^ { ( l ) } . } \end{array}\tag{6}
$$

2) Enforcing cross-layer consistency: To ensure that semantic information accumulated in earlier layers is not discarded during optimization, we introduce a cross-layer consistency term. The objective is not to make consecutive layer representations identical, but rather to encourage semantic continuity across the hierarchy. In other words, deeper layers should refine and enrich the information learned by previous layers while preserving the core semantic content of the code fragment. Without such regularization, layer-specific VICReg objectives may drive adjacent layers toward substantially different embedding spaces, reducing the benefits of hierarchical representation learning. Therefore, if a code fragment’s embedding at layer l − 1 encodes certain semantic features, the embedding at layer l should refine these features rather than producing a completely unrelated vector.

The cross-layer consistency term is computed independently for both views in a clone pair and then averaged:

$$
\mathcal { L } _ { \mathrm { c r o s s } } ^ { ( l ) } = \frac { 1 } { 2 } \left( \left\| z _ { a } ^ { ( l ) } - \mathbf { s g } \big ( z _ { a } ^ { ( l - 1 ) } \big ) \right\| _ { 2 } ^ { 2 } + \left\| z _ { b } ^ { ( l ) } - \mathbf { s g } \big ( z _ { b } ^ { ( l - 1 ) } \big ) \right\| _ { 2 } ^ { 2 } \right) ,\tag{7}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator, and $z _ { a } ^ { ( l ) } , \dot { z } _ { b } ^ { ( l ) }$ are the embeddings of the two code fragments at layer l (see “2. Cross-layer consistency” in Fig. 2)). We employ an L2 consistency objective because it directly penalizes large representation shifts between adjacent layers while remaining computationally inexpensive and stable to optimize. Unlike cosinebased objectives, which preserve only directional similarity, the L2 formulation constrains both direction and magnitude of the embedding trajectory.

The cross-layer consistency loss acts as an auxiliary regularizer at each transformer depth. To jointly optimize semantic alignment within each layer and semantic continuity across layers, we combine the layer-wise VICReg objective and the cross-layer consistency term into a single training objective:

$$
\mathcal { L } _ { \mathrm { L a y e r V I C R e g } } = \sum _ { l = 1 } ^ { L } w _ { l } \left( \mathcal { L } ^ { ( l ) } + \alpha \mathcal { L } _ { \mathrm { c r o s s } } ^ { ( l ) } \right) ,\tag{8}
$$

Algorithm 2 LWVIC4Code for Type-IV clone detection   
1: Initialize encoder $E ,$ layer-wise projectors $g ^ { ( l ) }$ , and opti  
mizer   
2: for epoch $= 1$ to $N _ { \mathrm { e p o c h s } }$ do   
3: for each batch $( c _ { 1 } , c _ { 2 } ) \sim \mathcal { D } _ { . }$ do   
4: Initialize hidden states $h _ { 1 } ^ { ( 0 ) } , h _ { 2 } ^ { ( 0 ) }$   
5: for layer $l = 1$ to $L$ do   
6: Forward pass: ${ h } _ { 1 } ^ { ( l ) } , { h } _ { 2 } ^ { ( l ) }$   
7: Masked mean pooling: $p _ { 1 } ^ { ( l ) } , p _ { 2 } ^ { ( l ) }$ =   
MeanPoo $( h _ { 1 } ^ { ( l ) } , h _ { 2 } ^ { ( l ) } , m _ { 1 , } m _ { 2 } )$   
8: Project: $z _ { 1 } ^ { ( \bar { l } ) } = g ^ { ( l ) } ( p _ { 1 } ^ { ( \bar { l } ) } ) , z _ { 2 } ^ { ( l ) } = g ^ { ( l ) } ( p _ { 2 } ^ { ( l ) } )$   
9: Compute $\bar { \boldsymbol { \mathcal { L } } } ^ { ( l ) }$   
10: if $l > 1$ then   
11: Compute cross-layer term: $\begin{array} { r c l } { { \mathcal L _ { \mathrm { c r o s s } } ^ { ( l ) } } } & { { = } } & { { \| z ^ { ( l ) } - } } \end{array}$   
$\mathsf { s g } \big ( z ^ { ( l - 1 ) } \big ) \rVert _ { 2 } ^ { 2 }$   
12: Accumulate weighted loss $\boldsymbol { w _ { l } } ( \mathcal { L } ^ { ( l ) } + \alpha \mathcal { L } _ { \mathrm { c r o s s } } ^ { ( l ) } )$   
13: Backpropagate total loss and update parameters   
14: Save final encoder weights

where $w _ { l } = ( l / L ) ^ { 2 }$ emphasizes deeper layers and α controls the cross-layer consistency strength.

LWVIC4Code training procedure is summarized in Algorithm 2. Similar to VIC4Code, it requires an encoder capable of transforming raw code fragments into embedding representations, such as a pretrained tokenizer model (e.g., CodeBERT) (line 1). For each training epoch (line 2), batches of code fragment pairs $\left( c _ { 1 } , c _ { 2 } \right)$ are processed (line 3). The hidden states for both code fragments are initialized at the input layer (line 4), and then a forward pass is performed through each layer l of the encoder (line 5). At every layer, token-level representations are aggregated using masked mean pooling, where padding tokens are ignored according to the attention mask, producing fixed-size vectors $p _ { 1 } ^ { ( l ) }$ and $\mathbf { \bar { \rho } } _ { p _ { 2 } } ^ { ( l ) }$ (line 7).

3) Applying layer-wise weighting: The VICReg loss is computed independently at each layer (line 9). For layers beyond the first, an additional cross-layer consistency term $\mathcal { L } _ { \mathrm { c r o s s } } ^ { ( \tilde { l } ) }$ is calculated to encourage smooth transitions between consecutive layers, where the stop-gradient operation prevents the cross-layer term from directly updating the previous-layer representation, avoiding trivial solutions, while the remaining objectives continue to optimize the encoder end-to-end. (lines 10–11). The combined layer-wise and cross-layer loss is scaled by a depth-dependent weight $w _ { l }$ and accumulated into the total loss (line 12). Once all layers are processed, the total loss is backpropagated to update both the encoder and all layer-specific projectors (line 13). After completing all epochs, the final encoder weights are saved (line 14) for downstream Type-IV clone detection tasks (see “3. Layer-wise weighting” in Fig. 2).

Although the cross-layer consistency objective introduces an additional attraction force between consecutive layer representations, the variance and covariance components inherited from VICReg provide opposing regularization pressures that discourage collapse. The variance term encourages sufficient activation variability across embedding dimensions, whereas the covariance term promotes complementary feature representations. Consequently, the cross-layer objective encourages semantic continuity across the transformer hierarchy while the VICReg regularization terms maintain representation diversity. Moreover, the VICReg-based approach is less sensitive to batch composition than contrastive methods (Section II-B), since variance and covariance regularization prevent representation collapse without requiring negatives.

## IV. EMPIRICAL EVALUATION

The goal of our evaluation is to empirically assess the effectiveness and generalizability of non-contrastive representation learning (LWVIC4Code) for detecting Type-IV code clones. Specifically, we aim to determine whether LWVIC4Code can (i) achieve performance comparable to contrastive approaches and zero-shot LLMs, (ii) benefit from a layer-wise training strategy to improve semantic representation learning, and (iii) generalize across programming languages when trained on a single programming language.

## A. Evaluation Protocol

To guide our empirical study, we define the following research questions (RQ):

RQ1: To what extent does non-contrastive representation learning (LWVIC4Code) compare to state-of-the-art contrastive learning for Type-IV clone detection?

Rationale: We aim to assess whether LWVIC4Code can achieve performance comparable to a state-of-the-art contrastive learning approach while avoiding the need for negative samples.

Method: We compare LWVIC4Code against $\mathrm { C o d e B E R T } _ { C L }$ [21], a recent state-of-the-art contrastive learning model for semantic clone detection. $\mathrm { C o d e B E R T } _ { C L }$ replaces the original classifier of CodeBERT with a contrastive learning objective that minimizes the distance between clone pairs while enforcing a margin between non-clone pairs. Similarity is computed using cosine similarity over the learned representations. During training, $\mathrm { C o d e B E R T } _ { C L }$ requires both positive (clone) and negative (non-clone) pairs, where negative pairs are randomly sampled to maintain a balanced 1:1 ratio. In contrast, LWVIC4Code is trained exclusively using positive clone pairs through the VICReg objective, eliminating the need for negative sampling. Both approaches are trained and evaluated on the Kamino (Python-only, 80/20 train-test split) and GPTCloneBench (Python, Java, C#, and C, 40/60 train-test split) datasets.

Metrics: Performance is evaluated using precision $( P ) _ { : }$ recall (R), F1-score $( F _ { 1 } )$ , and MCC, as it is a more reliable metric for binary classification because it considers all parts of the confusion matrix and avoids inflated performance [24]. For each pair of code embeddings, predictions are obtained by applying a similarity threshold $\tau .$ . Thresholds are explored in the range [0.1, 1.0] with increments of 0.05, and for each setting, the threshold maximizing MCC is selected (breaking ties using $F _ { 1 } )$ . This approach ensures that each model is compared at its optimal operating point, providing a fair assessment of the results. To assess statistical significance, we model the relationship between similarity scores and ground truth labels using Generalized Estimating Equations (GEE) logistic regression [45], which accounts for repeated measurements at the case level. Discriminative performance is quantified via Area Under the Curve (AUC) [46], with differences estimated using clustered bootstrap [47] to account for data dependencies. All p-values from bootstrap comparisons are adjusted using the Benjamini-Hochberg procedure [25] to control the false discovery rate. Differences are considered statistically significant if the adjusted p-value is below 0.05. Analyses are reported per dataset, per language, and overall to ensure robust conclusions.

RQ2: To what extent does a layer-wise training strategy (LWVIC4Code) improve the effectiveness of VIC4Code for Type-IV clone detection?

Rationale: VIC4Code optimizes embeddings only at the final layer, which may miss intermediate semantic features. LWVIC4Code enforces VICReg constraints across all layers, capturing richer semantic representations. We aim to measure whether this layer-wise learning improves alignment between semantically similar code fragments for Type-IV clones. Method: VIC4Code and LWVIC4Code are compared under identical training conditions. Both models are trained on the Kamino and GPTCloneBench datasets using the same encoder architecture, optimization settings, and VICReg coefficients $\left( \lambda ~ = ~ 3 4 , ~ \mu ~ = ~ 3 6 , ~ \nu ~ = ~ 1 \right)$ These coefficient values were selected through iterative testing on dataset samples to maximize $F _ { 1 }$ and MCC performance. Metrics: We use the same evaluation metrics (P, R, $F _ { 1 }$ , MCC) and statistical significance tests as in RQ1.

RQ3: To what extent each layer-wise component of LWVIC4Code contributes to Type-IV clone detection?

Rationale: LWVIC4Code extends VIC4Code through three design choices: (i) optimizing representations across all transformer layers. (ii) enforcing cross-layer consistency, and (iii) applying layer-wise weighting. We aim to quantify the contribution of each component and identify which components are primarily responsible for the observed performance improvements.

Method: We perform an ablation study by comparing the complete LWVIC4Code model against three variants, each removing a single component while keeping all other training settings unchanged: (i) Last Layer Only (LWVIC4Code<sub>LL</sub>), which computes the VICReg objective exclusively on the final transformer layer; (ii) No Cross-Layer Consistency (LWVIC4Code $N C ) .$ , which disables the consistency objective between adjacent transformer layers; and (iii) No Layer-wise Weighting $( \mathrm { L W V I C 4 C o d e } _ { N W } )$ , which assigns equal importance to every transformer layer during optimization. All variants are trained using the same datasets, encoder architectures, optimization settings, and coefficients.

Metrics: We evaluate the same metrics used in the previous RQs (precision, recall, $F _ { 1 }$ , MCC, and AUC), together with the same statistical significance analysis for each variant.

RQ4: To what extent does LWVIC4Code compare to zeroshot LLMs for Type-IV clone detection?

TABLE I: LLM Prompt for Type-IV Clone Detection.
<table><tr><td>System Prompt: You are a careful code reviewer. Determine whether two func- tions are Type-IV (semantic) clones. Output only a JSON object:  $\{ { \ " } \mathrm { p r e d i c t i o n " } \colon 0 ~ \mathrm { o r } ~ 1 \}$  No explanations.</td></tr><tr><td>User Prompt:</td></tr><tr><td>Given two code snippets (code_a, code_b) in a target language, determine if they are semantic clones. Prediction = 1 if the functions are semantic clones; otherwise, 0</td></tr></table>

Rationale: We aim to determine whether LWVIC4Code provides competitive or superior performance to LLMs in detecting semantic clones, without the overhead of zeroshot prompting. Method: We evaluate LWVIC4Code against zero-shot LLMs (deepseek-r1:14b, gpt-oss:20b, qwen3.5:9.65b). A fixed prompt (see Table I) is used to extract predicted labels. No finetuning is applied. While larger LLMs could be explored, we focus on models that are publicly accessible and representative of current practice. Metrics: Evaluation is performed using the same metrics as in the previous RQs (precision, recall, $F _ { 1 }$ , MCC). These metrics, however, are calculated based on the predicted labels (binary task) rather than using cosine similarity like RQs 1-2. For this reason, we use McNemar’s test to assess statistically significant differences between paired classification outcomes using the best threshold for LWVIC4Code to determine a prediction and compare it with the LLM’s predicted result. Similar to the other RQs, all p-values are adjusted using the Benjamini-Hochberg procedure [25] to control the false discovery rate, and differences are considered statistically significant if the adjusted p-value is below 0.05.

RQ5: To what extent can representations learned from Python-only generalize to other programming languages?

Rationale: We investigate the transferability of semantic embeddings to assess whether LWVIC4Code trained on a single language can detect Type-IV clones in other languages. Method: LWVIC4Code is trained exclusively on Python Type-IV clones from Kamino. Evaluation is performed on Java, C#, and C clones from GPTCloneBench. We examine performance across all possible cosine similarity thresholds $\tau \in \left[ 0 . 0 , 1 . 0 \right]$ to assess sensitivity to threshold selection. Metrics: We consider MCC and $F _ { 1 }$ scores for all possible cosine similarity thresholds, ranging [0.0, 1.0] to observe how the different thresholds affect results.

## B. Datasets

a) Kamino Dataset.: The Kamino dataset [22] is a largescale collection of semantically equivalent Python code fragments derived from BigCodeBench [41], comprising 78, 771 Type-IV clone pairs. The clones were generated via a hybrid pipeline that produces behaviorally equivalent but syntactically diverse implementations with LLMs, followed by deterministic validation steps. The pipeline ensures semantic correctness through unit test execution, enforces syntactic diversity using CodeBLEU-based filtering, and selects non-redundant examples via clustering. We use an 80/20 train-test split, following standard practice. Training data is used for RQs 1–3, while the test split is used in all evaluations.

b) GPTCloneBench.: GPTCloneBench [23] is a multilanguage dataset containing Types I–IV clones. The dataset was created by leveraging code fragments from Semantic-CloneBench [42] and prompting GPT-3 to generate semantically equivalent implementations, followed by filtering out syntactic clones using NiCad and manual validation. For this study, we use only the Type-IV subset, which includes code fragments across Python (3 948 pairs), Java (9 935 pairs), C# (6 872 pairs), and C (4 823 pairs). Due to its smaller size, we use a 40/60 train-test split to ensure sufficient test coverage while retaining minimal training data for multilanguage evaluation. Training data is used in RQs 1–2, while the test split is used in all RQs.

## C. Results

In this section, we present the empirical results for each RQ. The approach and complete results are available in our replication package [48].

1) RQ1: LWVIC4Code vs Contrastive Learning: LWVIC4Code consistently outperforms the state-of-theart contrastive learning approach, $\mathbf { C o d e B E R T } _ { C L }$ [21], while requiring only positive clone pairs during training. Across all datasets and programming languages (Table II), LWVIC4Code achieves higher precision, recall, $F _ { 1 }$ , and MCC scores. For example, on GPTCloneBench (C#), LWVIC4Code achieves an $F _ { 1 }$ of 0.977 and an MCC of 0.957, substantially outperforming CodeBERT<sub>CL</sub> $( F _ { 1 } ~ = ~ 0 . 9 0 0 ,$ MCC = 0.794). Similar improvements are observed for Java $( F _ { 1 } \colon 0 . 9 6 8 \ \mathrm { v s } .$ 0.804; MCC: 0.938 vs. 0.612) and Python $( F _ { 1 } \colon 0 . 9 2 0 \ \mathrm { \ v s } .$ 0.777; MCC: 0.840 vs. 0.598). The largest performance gap occurs for C, where LWVIC4Code achieves an $F _ { 1 }$ of 0.866 and an MCC of 0.790, compared to only $F _ { 1 } ~ = ~ 0 . 5 7 0$ and $\mathbf { M C C } = 0 . 4 5 7$ for $\mathrm { C o d e B E R T } _ { C L }$ . On the Python-only Kamino dataset, LWVIC4Code also provides a clear improvement, increasing the $F _ { 1 }$ score from 0.873 to 0.926 and the MCC from 0.745 to 0.853.

Looking at overall discriminative ability using AUC, LWVIC4Code consistently outperforms $\mathbf { C o d e B E R T } _ { C L }$ across all datasets and programming languages. On GPTCloneBench, LWVIC4Code achieves an overall AUC of 0.9864, compared with 0.8881 for CodeBERT , corresponding to an improvement of 0.0983 (95% CI [0.0938, 0.1031], $p < 0 . 0 0 1 $ ). The largest gain is observed for $\Gamma ,$ where LWVIC4Code improves the AUC from 0.7812 to 0.9612. Substantial improvements are also obtained for Java (0.9891 vs. 0.8905),Python (0.9704 vs. 0.8864), and C# (0.9953 vs. 0.9597), all with corrected $p < 0 . 0 0 1$ . On the Python-only Kamino dataset, LWVIC4Code similarly achieves a higher AUC (0.9751) than $\mathbf { C o d e B E R T } _ { C L }$ (0.9326), corresponding to an improvement of 0.0425 (95% CI [0.0402, 0.0448], $p < 0 . 0 0 1 )$ . All reported p-values were adjusted using the Benjamini–Hochberg procedure and remain below 0.05.

RQ1: LWVIC4Code consistently outperforms $\mathrm { C o d e B E R T } _ { C L }$ across all evaluated datasets and programming languages, improving the overall AUC from 0.8881 to 0.9864 on GPTCloneBench and from

0.9326 to 0.9751 on Kamino, with corresponding gains of up to 0.297 in $F _ { 1 }$ and 0.333 in MCC $( p < 0 . 0 0 1 )$

2) RQ2: Layer-wise vs Traditional VIC4Code: Across datasets and languages, LWVIC4Code consistently improves over standard VIC4Code (Table II). On GPTCloneBench, LWVIC4Code achieves its highest $F _ { 1 }$ and MCC on C# (0.977 and 0.957), surpassing VIC4Code (0.965 $F _ { 1 }$ , 0.933 MCC). Improvements are also observed for Java (MCC 0.938 vs. 0.921) and Python (MCC 0.840 vs. 0.837). Even for the lowest-performing language (C), LWVIC4Code outperforms VIC4Code (MCC 0.790 vs. 0.741). All p-values remain below 0.05 after being corrected using the Benjamini-Hochberg procedure.

Considering overall discriminative ability, LWVIC4Code achieves slightly higher AUC values on GPTCloneBench (0.9864 vs. 0.9815) with the largest gains for C, while differences are smaller for other languages. On Python-only Kamino, VIC4Code performs marginally better (AUC 0.9775 vs. 0.9751), but both methods remain highly effective.

RQ2: LWVIC4Code consistently improves over VIC4Code for Type-IV clone detection across programming languages, achieving higher $F _ { 1 }$ and MCC scores on GPTCloneBench, with gains of up to 0.049 MCC on C and 0.024 MCC on $\mathsf { C } \# \mathrm { ~ } ( p \mathrm { ~ } < \mathrm { ~ } 0 . 0 5 )$ . LWVIC4Code also achieves a higher overall AUC on GPTCloneBench (0.9864 vs. 0.9815), while both approaches remain highly effective on Kamino.

3) RQ3: Ablation on LWVIC4Code: The ablation results demonstrate the importance of the proposed three novel components in LWVIC4Code (Table II). Surprisingly, the lastlayer-only variant achieves the highest $F _ { 1 }$ and MCC scores among the evaluated variants on GPTCloneBench. This suggests that, for this dataset, optimizing only the final transformer layer is sufficient to achieve strong discriminative performance, although the reasons for this behavior require further investigation. For example, on C#, LWVIC4Code<sub>LL</sub> improves MCC from 0.957 to 0.964, while on Java it increases MCC from 0.938 to 0.945.

The remaining ablation variants show that the additional layer-wise mechanisms contribute positively to the overall representation quality. Removing cross-layer consistency $( \mathrm { L W V I C 4 C o d e } _ { N C } )$ results in small improvements in isolated cases, such as C on GPTCloneBench $( F _ { 1 } ~ 0 . 9 1 8$ vs. 0.866 for the original LWVIC4Code) and ${ \mathsf { C } } \# { \mathsf { \Gamma } } ( F _ { 1 } \ 0 . 9 7 5 $ vs. 0.977), but generally decreases performance compared to the complete model. Similarly, removing layer-wise weighting (LWVIC4Code ) leads to performance degradation in most settings, with notable reductions for Python on GPT-CloneBench (MCC 0.734) and Kamino (MCC 0.799). Although some configurations remain competitive, these results indicate that assigning equal importance to all layers is less effective than the proposed weighting strategy, suggesting that different transformer layers contribute unequally to the final representation.

The results show that removing layer-wise weighting (LWVIC4Code<sub>NW</sub>) consistently degrades performance, with statistically significant AUC reductions on both GPT-CloneBench (0.9864 vs. 0.9701, $ { p } \textsuperscript { < } \ 0 . 0 0 1 )$ and Kamino (0.9751 vs. 0.9481, $p \ < \ 0 . 0 0 1 )$ . Similarly, removing crosslayer consistency produces significant differences on Kamino $( p < 0 . 0 0 1 )$ , although the AUC difference is smaller (0.9751 vs. 0.9684). For the last-layer-only variant, the results vary across datasets, as it achieves higher AUC on GPTCloneBench (0.9916 vs. 0.9864, $\begin{array} { r l r } { p } & { { } = } & { 0 . 0 0 1 ) } \end{array}$ , whereas LWVIC4Code achieves higher AUC on Kamino (0.9840 vs. 0.9751, $p \ =$ 0.001). This difference is also reflected in the thresholdbased evaluation, where $\mathrm { L W V I C } 4 \mathrm { C o d e } _ { L L }$ performs better than LWVIC4Code for higher θ values, as indicated by the best thresholds reported in Table II, while it is less effective for lower thresholds (θ¡0.75).

TABLE II: Clone detection results for best thresholds (θ). Bold indicates the best result within each dataset/language group, while underlined indicates the second-best result.
<table><tr><td>Model</td><td>DS</td><td>Lang.</td><td>θ| P</td><td></td><td>R  $F _ { 1 }$ </td><td>MCC</td><td></td></tr><tr><td colspan="8">Contrastive Baseline</td></tr><tr><td colspan="8">Java</td></tr><tr><td rowspan="4"> $\mathbf { C o d e B E R T } _ { C L }$ </td><td>C# G</td><td></td><td>.75 .55</td><td>.813 .866</td><td>.795 .804 .935 .899</td><td></td><td>.612 .794</td></tr><tr><td>C</td><td></td><td>.90</td><td>.929</td><td>.411</td><td>.570</td><td>.457</td></tr><tr><td>Py</td><td></td><td>.85</td><td>.853</td><td>.714 .777</td><td></td><td>.598</td></tr><tr><td>K Py</td><td></td><td>.70</td><td>.872</td><td>.874</td><td>.873</td><td>.745</td></tr><tr><td colspan="8">Non-contrastive methods</td></tr><tr><td rowspan="4">LWVIC4Code</td><td>Java</td><td></td><td>.75</td><td></td><td></td><td></td><td>.938</td></tr><tr><td></td><td>C#</td><td>.75</td><td>.952 .967</td><td>.985 .988</td><td>.968 .977</td><td>.957</td></tr><tr><td>G</td><td>C</td><td>.90</td><td>.879</td><td>.854</td><td>.866</td><td>.790</td></tr><tr><td>K</td><td>Py</td><td>.80</td><td>.919</td><td>.921</td><td>.920</td><td>.840</td></tr><tr><td rowspan="5">VIC4Code</td><td>Py</td><td></td><td>.70</td><td>.941</td><td>.910</td><td>.926</td><td>.853</td></tr><tr><td></td><td>Java</td><td>.90</td><td>.949</td><td>.970</td><td>.960</td><td>.921</td></tr><tr><td></td><td>C#</td><td>.85</td><td>.954</td><td>.977</td><td>.965</td><td>.933</td></tr><tr><td>G C</td><td></td><td>.90</td><td>.787</td><td>.900</td><td>.840</td><td>.741</td></tr><tr><td></td><td>Py</td><td>.75</td><td>.912</td><td>.926</td><td>.919</td><td>.837</td></tr><tr><td>LWVIC4Code Ablation</td><td>K</td><td>Py</td><td>.60</td><td>.959</td><td>.934</td><td>.946</td><td>.894</td></tr><tr><td colspan="8"></td></tr><tr><td rowspan="4"> $\mathrm { L W V I C 4 C o d e } _ { L L }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Java</td><td>.80</td><td>.967</td><td>.979</td><td>.973</td><td>.945</td></tr><tr><td>G C# C</td><td></td><td>.90</td><td>.989</td><td>.975</td><td>.982</td><td>.964</td></tr><tr><td></td><td>Py</td><td>.90 .80</td><td>.939 .964</td><td>.917 .943</td><td>.928 .953</td><td>.857 .908</td></tr><tr><td rowspan="4"></td><td>K</td><td>Py</td><td>.75</td><td>.979</td><td>.933</td><td>.956</td><td>.915</td></tr><tr><td></td><td>Java</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>.80</td><td>.956</td><td>.976</td><td>.966</td><td>.931</td></tr><tr><td>G C# C</td><td></td><td>.85</td><td>.976</td><td>.974</td><td>.975</td><td>.950</td></tr><tr><td></td><td>Py</td><td></td><td>.80 .85</td><td>.894 .934</td><td>.944 .929</td><td>.918 .932</td><td>.834 .863</td></tr><tr><td rowspan="4"></td><td>K</td><td>Py</td><td>.80</td><td>.955</td><td>.889</td><td>.921</td><td>.849</td></tr><tr><td></td><td>Java</td><td>.80</td><td>.936</td><td>.942</td><td>.939</td><td></td></tr><tr><td>G C#</td><td></td><td></td><td></td><td>.968</td><td></td><td>.877</td></tr><tr><td>C</td><td></td><td>.75 .85</td><td>.968 .868</td><td>.937</td><td>.968 .901</td><td>.936 .796</td></tr><tr><td></td><td></td><td>Py</td><td>.70</td><td>.816</td><td>.937</td><td>.872</td><td>.734</td></tr><tr><td></td><td>K</td><td>Py</td><td>.70</td><td>.941</td><td>.849</td><td>.893</td><td>.799</td></tr><tr><td colspan="8">LLMs zero-shot</td></tr><tr><td rowspan="4">Deepseek-r1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Java C#</td><td></td><td>.992 .998</td><td>.789 .820</td><td>.879 .900</td><td>.805 .837</td></tr><tr><td>G</td><td></td><td>一 一</td><td>.993</td><td>.558</td><td>.715</td><td>.662</td></tr><tr><td></td><td>C Py</td><td>一</td><td>.994</td><td>.639</td><td>.778</td><td>.681</td></tr><tr><td rowspan="4">Gpt-oss</td><td>K</td><td>Py</td><td>一</td><td>1</td><td>.916</td><td>.956</td><td>.919</td></tr><tr><td></td><td>Java</td><td>一</td><td>.991</td><td>.695</td><td>.817</td><td>.727</td></tr><tr><td>G</td><td>C#</td><td></td><td>.999</td><td>.724</td><td>.840</td><td>.759</td></tr><tr><td></td><td>C Py</td><td>一</td><td>.991 .984</td><td>.412 .537</td><td>.582 .694</td><td>.549 .593</td></tr><tr><td></td><td>K</td><td>Py</td><td>1</td><td>.999</td><td>.776</td><td>.873</td><td>.795</td></tr><tr><td rowspan="5">Qwen</td><td></td><td></td><td></td><td></td><td></td><td></td><td>.603</td></tr><tr><td></td><td>Java C#</td><td>1</td><td>.994</td><td>.531 .585</td><td>.692 .738</td><td>.651</td></tr><tr><td>G</td><td></td><td>一</td><td>.999 .996</td><td>.302</td><td>.464</td><td></td></tr><tr><td>C Py</td><td></td><td>一</td><td>1</td><td>.390</td><td>.561</td><td>.461 .493</td></tr><tr><td>K Py</td><td></td><td>一 一</td><td>1</td><td>.306</td><td>.469</td><td>.425</td></tr></table>

G: GPTCloneBench; K: Kamino

![](images/ac33b3d9e4d18451b310d454f7e4a2c2dea9227b014ac323643fae3454b47518.jpg)  
Fig. 3: Cosine similarity distributions of clone and non-clone pairs for LWVIC4Code variants.

Lastly, an analysis of the similarity score distributions (see Fig. 3) further shows that LWVIC4Code $L L$ produces consistently higher median similarity scores for clone pairs than the complete model across both GPTCloneBench (0.993 vs. 0.983) and Kamino (0.983 vs. 0.977), while also exhibiting lower variance, particularly on GPTCloneBench (std. 0.063 vs. 0.190). These findings explain why $\mathrm { L W V I C } 4 \mathrm { C o d e } _ { L L }$ reaches its optimal performance at higher similarity thresholds, although the full LWVIC4Code model remains more robust across datasets and operating thresholds.

RQ3: LWVIC4Code benefits from its novel components, with layer-wise weighting providing the most consistent improvement for Type-IV clone detection. Removing crosslayer consistency also affects performance, although with smaller and dataset-dependent effects. The last-layer-only variant achieves competitive results and even higher AUC on GPTCloneBench (0.9916 vs. 0.9864), but performs worse on Kamino (0.9751 vs. 0.9840).

4) RQ4: LWVIC4Code vs Zero-Shot LLMs: Across programming languages, LWVIC4Code generally outperforms zero-shot LLMs for Type-IV clone detection (Table II). On GPTCloneBench, LWVIC4Code achieves more consistent results across languages, particularly for structurally diverse ones like $\Gamma ,$ where LLM performance drops sharply (e.g., DeepSeek-r1 MCC is 0.662). The only exception where the LLM outperforms LWVIC4Code is DeepSeek-r1 for Kamino (0.956 $F _ { 1 }$ , 0.919 MCC). These results may be good considering that Kamino used Deepseek-r1 as one of the LLMs to generate its clones. Most of the other cases (GPTCloneBench) showed average to poor performance, with the best results being for DeepSeek-r1 on C# (0.900 $F _ { 1 }$ , 0.837 MCC) and on Java (0.879 $F _ { 1 }$ , 0.805 MCC)

Hence, for all LLMs evaluated on GPTCloneBench, McNemar’s tests indicate statistically significant differences compared to LWVIC4Code after Benjamini–Hochberg correction $( p _ { a d j } < 0 . 0 5 )$ . The largest differences are observed for Qwen and GPT-OSS. On the Kamino dataset, the statistical differences are also significant for all evaluated LLMs. However, the performance gap is smaller for DeepSeek-r1, whereas GPT-OSS and Qwen remain below.

RQ4: On GPTCloneBench, LWVIC4Code achieves up to 0.316 MCC improvement over zero-shot LLMs (e.g., compared to Qwen on $\mathrm { ~ \textsf ~ { ~ C ~ } ~ } ( p \mathrm { ~ \ ~ } < \mathrm { ~ \ ~ } 0 . 0 5 )$ . On Kamino, DeepSeek-r1 is the only LLM that surpasses LWVIC4Code $( F _ { 1 } = 0 . 9 5 6 , \mathrm { M C C } = 0 . 9 1 9 )$

5) RQ5: Transferability of Representations in Python: Fig. 4 presents the MCC and $F _ { 1 }$ scores across thresholds when LWVIC4Code is trained exclusively on Python Type-IV clones from Kamino. Overall, the results demonstrate a strong degree of multi-language generalization, particularly for ${ \mathrm { C } } \#$ and Java.

For these two languages, the model achieves its best performance in the threshold range $\theta \in \left[ 0 . 6 , 0 . 7 \right]$ , where $F _ { 1 }$ scores exceed 0.8 and peak at 0.925 for C#. Performance remains relatively stable up to $\theta = 0 . 7 5$ , after which both MCC and $F _ { 1 }$ decline, indicating that higher thresholds impose stricter similarity requirements that the model cannot satisfy.

In contrast, performance on C is noticeably lower, with a maximum $F _ { 1 }$ slightly above 0.7 and MCC reaching approximately 0.6. This gap can be attributed to the greater syntactic and paradigmatic differences between C and Python, compared to the more structurally similar C# and Java.

RQ5: LWVIC4Code, trained solely on Python data, can effectively transfer to other programming languages without any exposure during training. This indicates that the model captures language-agnostic semantic representations of code, although its effectiveness decreases as the target language diverges further from Python (i.e., on C).

## V. DISCUSSION

## A. Benefits and Limitations

a) Advantages of LWVIC4Code: LWVIC4Code exhibits better performance than the contrastive baseline. LWVIC4Code achieved high $F _ { 1 }$ and MCC scores on GPT-CloneBench for Java and $\mathbf { C } \# .$ , as shown in Table II (0.968 and

![](images/a81e9fa1626227910abc5422272fd6a5fb5cf3d456644826df06f62421f91d6e.jpg)  
Fig. 4: Scores across θ using Python-trained LWVIC4Code.

0.938 for Java, and 0.977 and 0.957 for C#) even though the training sets for these languages were relatively small (4 257 and 2 944 pairs, respectively). Its non-contrastive nature further highlights its efficiency, as negative pairs were not needed to achieve competitive performance, and in some cases, it outperforms zero-shot LLMs such as DeepSeek-r1, GPT-OSS, and Qwen, particularly on structurally diverse languages like $\mathrm { ~ C ~ } ( F _ { 1 } 0 . 8 6 6 , \mathrm { M C C } 0 . 7 9 0 )$ and Python $( F _ { 1 } 0 . 9 2 0 , \mathrm { M C C } 0 . 8 4 0 )$ These results demonstrate that LWVIC4Code can learn robust, language-agnostic semantic representations of code efficiently, even from limited data. These findings directly address the challenges outlined in Section II, demonstrating that effective semantic representations can be learned without reliance on negative sampling.

b) Contribution of LWVIC4Code novel components: The ablation study (RQ3) provides further insights into the design choices behind LWVIC4Code. Removing layer-wise weighting consistently reduces performance across datasets, indicating that not all encoder layers contribute equally to semantic clone representations. These findings are consistent with prior work [20] suggesting that deeper transformer layers encode more abstract semantic information. Similarly, removing crosslayer consistency leads to significant performance degradation on Kamino, demonstrating that explicitly encouraging agreement between representations learned at different depths improves the stability of the embedding space.Interestingly, the last-layer-only variant achieves competitive, and in some cases superior, AUC values, particularly on GPTCloneBench. However, this improvement is dependent on the similarity threshold $\mathbf { \eta } ^ { ( \theta ) } \cdot$ , as the variant performs better only for higher thresholds, while LWVIC4Code provides stronger results across a broader range of thresholds. This indicates that relying exclusively on the final encoder layer may produce highly concentrated similarity scores that are effective in stricter decision scenarios. In contrast, the complete LWVIC4Code architecture provides more stable representations, which may generalize to more scenarios where lower thresholds are intended (e.g., crosslanguage clone detection).

c) Limitations of non-contrastive learning: Since the model relies solely on positive pairs, it requires a sufficiently large and diverse set of examples to avoid collapsing representations toward overly similar embeddings. This is reflected in the evaluation results: on languages with fewer or more syntactically divergent examples (e.g., C), LWVIC4Code achieves lower $F _ { 1 }$ and MCC scores (0.866 and 0.790, respectively) and requires higher thresholds (θ = 0.9) to maintain precision, indicating a bias toward high similarity values. In contrast, contrastive models naturally balance positives and negatives, which helps maintain a wider separation between clone and non-clone embeddings. Moreover, while LWVIC4Code transfers well to languages structurally similar to Python (Java, C#), its effectiveness diminishes as the target language diverges, suggesting that non-contrastive representations may be less robust when the training data does not capture sufficient crosslanguage diversity.

d) Practical implications: LWVIC4Code is a lightweight and efficient model, making it well-suited for real-time integration into development environments such as IDEs. It is also efficient, as detecting clones across the full GPTCloneBench test set (≈ 44k pairs) takes less than 10 minutes, whereas zero-shot LLMs require dozens of hours to process the same data. Moreover, LWVIC4Code can be finetuned incrementally on new positive examples in production without generating negative samples. In contrast, contrastive approaches require careful construction of negative pairs, which can be timeconsuming and impractical in real-world settings. These properties make LWVIC4Code a practical solution for continuous, scalable, and language-agnostic Type-IV clone detection in real-world development environments.

e) Use of LLM-generated data: A key aspect of our study is the use of LLM-generated data, as Kamino consists of Python code produced by LLMs. While this may lead to threats, such a choice reflects a growing reality, as much modern code is now produced or assisted by LLMs in practice [49]. Evaluating LWVIC4Code and other models on such data provides insights into their effectiveness in realistic AI-assisted coding scenarios. Importantly, GPTCloneBench (created by leveraging SemanticCloneBench fragments with GPT-3, followed by manual curation, tool-assisted filtering, functionality testing, and automated validation) contains highquality human-labeled and cross-language code. Experiments on this dataset show that findings from LLM-generated data transfer effectively to realistic code, supporting the practical relevance of our results in modern AI-assisted development settings.

f) Research opportunities: Our findings highlight several opportunities for future work. First, exploring hybrid training strategies that combine non-contrastive and contrastive objectives could help mitigate limitations related to a lack of data and representation collapse, particularly for syntactically divergent languages (e.g., Python vs C). Second, extending LWVIC4Code to additional programming languages and more diverse datasets (including code generated by different LLMs or real-world industrial repositories) would further assess its generalizability. Third, integrating dynamic or adaptive threshold selection mechanisms could enhance practical deployment, reducing reliance on manually tuned θ values (e.g., allowing one to use LWVIC4Code $L L$ instead). Finally, investigating lightweight incremental finetuning and continual learning strategies at production time would allow the model to adapt efficiently to evolving codebases and AI-assisted development environments, supporting scalable, real-time clone detection. More broadly, as this work represents, to the best of our knowledge, the first application of VICReg to code-related tasks, it opens a new line of research on non-contrastive representation learning for software engineering. Beyond clone detection, such methods could be explored for related problems, including code search, defect detection, and program classification, where learning robust semantic representations without reliance on negative sampling may offer significant advantages.

## B. Threats to Validity

Internal validity: Differences in model performance for contrastive, non-contrastive, and LLMs may be affected by implementation details, hyperparameter choices, or dataset preprocessing. We mitigated this by using consistent training settings, architectures, and evaluation protocols across all models and approaches. Additionally, we selected a state-ofthe-art baseline [21] that used a well-established pretrained model (CodeBERT) for contrastive learning and carefully tuned hyperparameters for LWVIC4Code based on preliminary experiments to ensure fair and representative comparisons. To ensure comparability with the contrastive model, we used CodeBERT as the encoder for both LWVIC4Code and VIC4Code, maintaining consistency in the embedding space across all approaches.

External validity: Our results may not fully generalize to other codebases or programming languages. To mitigate this threat, we evaluated models on two complementary datasets: Kamino, a large-scale Python-only dataset, and GPT-CloneBench, a multi-language dataset covering four programming languages. Although Kamino is derived from Big-CodeBench and may include multiple implementations per task, cross-split semantic similarity on task descriptions is low (mean = 0.16, std = 0.14), indicating mostly distinct functionalities. Lastly, Kamino is composed of LLM-generated Python code, which could introduce biases or patterns not present in real-world code. To mitigate this, we also evaluate on GPTCloneBench, a high-quality, curated dataset with crosslanguage and human-labeled clones. Consistent results across both datasets suggest that our findings generalize beyond synthetic LLM-generated data.

Construct validity: Metrics such as MCC, $F _ { 1 } ,$ and similarity thresholds may not capture all aspects of semantic equivalence. Furthermore, selecting one (random) θ could bias the results analysis towards an improvement in a selected direction. To mitigate this, we considered the results for all possible θ, reporting the best for each individual model and programming language. Additionally, prior work has shown that clone detection models such as CodeBERT can achieve artificially high performance when test data is similar to training data [21], [50]. By including multiple languages and ensuring nonoverlapping train-test splits, we reduce this risk and provide a more realistic assessment of model generalization. Furthermore, the results for RQ5 were considered by training and testing on completely different datasets, demonstrating the potential of transfer learning.

Conclusion validity: While MCC and $F _ { 1 }$ are well-known metrics, only considering them could bias the analysis. To mitigate this, we also report the statistical significance tests (GEE, AUC differences with bootstrap confidence intervals, McNemar’s test) to reduce the risk of drawing incorrect inferences. We also applied the Benjamini–Hochberg [25] procedure to control the false discovery rate across multiple comparisons. However, small sample sizes in some language subsets (e.g., C) may affect the reliability of these results.

## VI. RELATED WORK

This section presents an overview of the main approaches proposed for semantic/Type-IV code clone detection.

## A. AST and PDG-Based Approaches

Sheneamer and Kalita [13] combined Abstract Syntax Tree (AST) and Program Dependence Graphs (PDG)-based representations with ML classifiers to detect complex Type-III and Type-IV clones. Their results showed that semantic program representations significantly outperform traditional metric-based approaches, highlighting the importance of incorporating program semantics into clone detection. Recent work [51] leveraged PDG representations with graph analysis techniques to detect semantic clones, reporting strong performance on benchmark datasets. Despite their effectiveness, AST- and PDG-based approaches often suffer from scalability limitations, as constructing and analyzing detailed program graphs can be computationally expensive, making them difficult to apply to large codebases [52].

Furthermore, since our approach is inspired by the layerwise VICReg [20], it is important to highlight what is novel in LWVIC4Code. LWVIC4Code differs from the layer-wise VICReg formulation of Datta et al. in both its application domain (code clone detection vs. image classification) and optimization strategy. Datta et al.’s approach uses layer-wise representation learning on local objectives applied independently at each layer. In contrast, LWVIC4Code applies layer-wise VICReg objectives within a transformer encoder for source code and optimizes the entire network jointly through standard end-to-end backpropagation (“1. Representation across layers” in Fig. 2). The proposed cross-layer consistency regularizer (“2. Cross-layer consistency” in Fig. 2) and depth-dependent weighting (“3. Layer-wise weighting”) further encourage semantic alignment across layers and prioritize deeper semantic representations relevant to Type-IV clone detection.

## B. Representation Learning Approaches

To address scalability limitations, recent work has explored representation learning techniques. Wu et al. [52] proposed a method that models semantic code clones by transforming AST structures into Markov chain representations to train a clone detection model, while Hu et al. [53] instead convert AST structures into graph-based representations to learn semantic embeddings of code. These approaches demonstrate how structural program information can be transformed into representations suitable for machine learning models.

Considering contrastive representation learning, Kitsios et al. [21] demonstrate that contrastive objectives help models generalize to functionalities not observed during training. Similarly, Li et al. [32] proposed a cross-language code clone detection approach that learns embeddings from multiple clone datasets. Their model is able to generalize to unseen programming languages and features by learning languageagnostic semantic representations. This is similar to what was explored with RQ5, which showcases how our LWVIC4Code can be trained in Python only to detect clones in Java and C#. Mohammed et al. [54] proposed a transformer-based model for cross-language semantic clone detection that incorporates semantic hints derived from AST and control-flow graph structures, achieving strong results across multiple languages.

In a different direction, Li et al. [55] apply contrastive learning to binary code embeddings rather than source code. Although primarily designed for security applications such as binary similarity detection, their results further highlight the effectiveness of contrastive representation learning for capturing program semantics. A limitation of contrastive approaches, however, lies in their dependence on carefully constructed positive and negative pairs (as discussed in Section II). Dataset imbalance can significantly impact model performance. For example, modifying the BigCloneBench dataset to balance positive and negative pairs may introduce potential biases during training [21].

To the best of our knowledge, no prior work has employed a non-contrastive objective to train models for code clone detection; alternative training strategies beyond purely contrastive approaches have been explored [56]. For instance, Keller et al. [57] proposed a visualization-based method that transforms source code into image-like representations and leverages transfer learning from computer vision models. By exploiting pretrained visual feature extractors, their approach captures semantic patterns in code; however, it still relies on labeled data. Similarly, Guo et al. [58] introduced a semi-supervised clone detection technique that facilitates review sharing. Their method integrates a convolutional neural network with an autoencoder architecture to learn from both labeled and unlabeled data, thereby improving detection performance.

More recently, Dou et al. [8] propose an ensemble learning approach for automated code clone verification, combining multiple machine learning models to improve the accuracy of clone classification. Their work, however, focuses on verifying candidate clone pairs rather than learning semantic code representations, relying on engineered features and ensemble strategies to distinguish true clones from false positives.

## C. Large Language Model-Based Approaches

LLMs have been explored for semantic clone detection due to their strong capability to capture contextual information in code. Zhang et al. [59] evaluated the ability of GPT-3.5 and GPT-4 to detect semantic clones and found that these models struggled with accurately identifying functional equivalence between programs. Recently, however, several off-the-shelf open-source LLMs perform on par with specialized clone detection models when evaluating functionalities not present during training, highlighting the potential of these models for generalization [21]. Thus, justifying the comparison to LWVIC4Code in RQ4. Nevertheless, challenges remain. In the context of cross-language clone detection, Moumoula et al. [44] show that even advanced LLMs struggle to reliably capture functional equivalence across programming languages. Their findings suggest that dedicated embedding models yield more robust semantic representations and currently achieve state-of-the-art performance for cross-lingual clone detection. This observation aligns with our findings for RQ4, where LLMs achieve competitive results only at lower θ thresholds, yet still underperform compared to both contrastive and noncontrastive embedding models reported in Table II.

## VII. CONCLUSION

This paper presents LWVIC4Code, a non-contrastive representation learning approach for Type-IV code clone detection that adapts the VICReg framework to code and enforces layerwise consistency across transformer layers. By capturing semantic information progressively, LWVIC4Code produces robust embeddings capable of detecting semantically equivalent, syntactically diverse code fragments without relying on negative samples. Our evaluation demonstrates LWVIC4Code’s strong performance across multiple programming languages, achieving better $F _ { 1 }$ and MCC scores relative to a contrastive baseline and zero-shot LLMs. The ablation study showed that removing layer-wise weighting has the biggest negative impact on the model. Despite this, using the last layer only showed improvements for higher θ. Our evaluation also shows that when trained on Python-only data, it can generalize effectively to Java and C#, highlighting its potential for cross-language transfer. Overall, this work contributes a new class of models for Type-IV clone detection, provides insights into layerwise non-contrastive learning, and systematically compares different machine learning strategies, laying the foundation for further research on semantic code representation and multilanguage clone detection.

Future work includes exploring hybrid training strategies that combine non-contrastive and contrastive objectives, extending LWVIC4Code to additional languages and diverse datasets (including LLM-generated and industrial code), integrating dynamic threshold selection, and investigating incremental fine-tuning and continual learning for scalable, realtime clone detection.

## ACKNOWLEDGMENTS

This work was partially funded by the Natural Sciences and Engineering Research Council of Canada (NSERC), grant numbers RGPIN-2025-05677 and RGPIN-2026-06532.

## DATA AVAILABILITY STATEMENT

A replication package is available at [48].

## REFERENCES

[1] Q. U. Ain, W. H. Butt, M. W. Anwar, F. Azam, and B. Maqbool, “A systematic review on code clone detection,” IEEE Access, vol. 7, pp. 86 121–86 144, 2019.

[2] R. Tairas and J. Gray, “Increasing clone maintenance support by unifying clone detection and refactoring activities,” Information and Software Technology, vol. 54, no. 12, pp. 1297–1307, 2012, special Section on Software Reliability and Security.

[3] R. Mo, Y. Jiang, W. Zhan, D. Wang, and Z. Li, “A comprehensive study on code clones in automated driving software,” in 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2023, pp. 1073–1085.

[4] S. Carter, R. Frank, and D. Tansley, “Clone detection in telecommunications software systems: A neural net approach,” in Proceedings of the International Workshop on Applications of Neural Networks to Telecommunications. Psychology Press, 1993, pp. 273–280.

[5] C. K. Roy, J. R. Cordy, and R. Koschke, “Comparison and evaluation of code clone detection techniques and tools: A qualitative approach,” Science of Computer Programming, vol. 74, no. 7, pp. 470–495, 2009.

[6] Y. Wang, Y. Ye, Y. Wu, W. Zhang, Y. Xue, and Y. Liu, “Comparison and evaluation of clone detection techniques with different code representations,” in 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), 2023, pp. 332–344.

[7] M. Gabel, L. Jiang, and Z. Su, “Scalable detection of semantic clones,” in 2008 ACM/IEEE 30th International Conference on Software Engineering, 2008, pp. 321–330.

[8] S. Dou, Y. Hu, S. Feng, Y. Wu, and D. Zou, “Eler: Ensemble learningbased automated verification of code clones,” IEEE Transactions on Software Engineering, vol. 52, no. 3, pp. 967–983, 2026.

[9] V. Murali, C. Maddila, I. Ahmad, M. Bolin, D. Cheng, N. Ghorbani, R. Fernandez, N. Nagappan, and P. C. Rigby, “Ai-assisted code authoring at scale: Fine-tuning, deploying, and mixed methods evaluation,” Proc. ACM Softw. Eng., vol. 1, no. FSE, Jul. 2024.

[10] W. Xu, K. Gao, H. He, and M. Zhou, “Licoeval: Evaluating llms on license compliance in code generation,” in 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE), 2025, pp. 1665–1677.

[11] K. Wang, G. Zhang, Z. Zhou, J. Wu, M. Yu, S. Zhao, C. Yin, J. Fu, Y. Yan, H. Luo, L. Lin, Z. Xu, H. Lu, X. Cao, X. Zhou, W. Jin, F. Meng, S. Xu, J. Mao, Y. Wang, H. Wu, M. Wang, F. Zhang, J. Fang, W. Qu, Y. Liu, C. Liu, Y. Zhang, Q. Li, C. Guo, Y. Qin, Z. Fan, K. Wang, Y. Ding, D. Hong, J. Ji, Y. Lai, Z. Yu, X. Li, Y. Jiang, Y. Li, X. Deng, J. Wu, D. Wang, Y. Huang, Y. Guo, J. tse Huang, Q. Wang, X. Jin, W. Wang, D. Liu, Y. Yue, W. Huang, G. Wan, H. Chang, T. Li, Y. Yu, C. Li, J. Li, L. Bai, J. Zhang, Q. Guo, J. Wang, T. Chen, J. T. Zhou, X. Jia, W. Sun, C. Wu, J. Chen, X. Hu, Y. Li, X. Wang, N. Zhang, L. A. Tuan, G. Xu, J. Zhang, T. Zhang, X. Ma, J. Gu, L. Pang, X. Wang, B. An, J. Sun, M. Bansal, S. Pan, L. Lyu, Y. Elovici, B. Kailkhura, Y. Yang, H. Li, W. Xu, Y. Sun, W. Wang, Q. Li, K. Tang, Y.-G. Jiang, F. Juefei-Xu, H. Xiong, X. Wang, D. Tao, P. S. Yu, Q. Wen, and Y. Liu, “A comprehensive survey in llm(-agent) full stack safety: Data, training and deployment,” 2025. [Online]. Available: https://arxiv.org/abs/2504.15585

[12] P. V. Bhaskar and Geetika, “A comprehensive analysis of unified approaches for revealing code clone detection,” in 2024 4th International Conference on Ubiquitous Computing and Intelligent Information Systems (ICUIS), 2024, pp. 946–953.

[13] A. Sheneamer and J. Kalita, “Semantic clone detection using machine learning,” in 2016 15th IEEE International Conference on Machine Learning and Applications (ICMLA), 2016, pp. 1024–1028.

[14] Y. Wang, W. Wang, S. Joty, and S. C. Hoi, “CodeT5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation,” Online and Punta Cana, Dominican Republic, pp. 8696– 8708, Nov. 2021.

[15] Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin, T. Liu, D. Jiang, and M. Zhou, “Codebert: A pre-trained model for programming and natural languages,” 2020. [Online]. Available: https://arxiv.org/abs/2002.08155

[16] N. D. Q. Bui, Y. Yu, and L. Jiang, “Self-supervised contrastive learning for code retrieval and summarization via semantic-preserving transformations,” in Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, ser. SIGIR ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 511–521.

[17] C. Chen, J. Zhang, Y. Xu, L. Chen, J. Duan, Y. Chen, S. Tran, B. Zeng, and T. Chilimbi, “Why do we need large batchsizes in contrastive learning? a gradient-bias perspective,” Advances in Neural Information Processing Systems, vol. 35, pp. 33 860–33 875, 2022.

[18] R. Balestriero and Y. LeCun, “Contrastive and non-contrastive selfsupervised learning recover global and local spectral embedding meth-

ods,” Advances in Neural Information Processing Systems, vol. 35, pp. 26 671–26 685, 2022.

[19] A. Bardes, J. Ponce, and Y. LeCun, “Vicreg: Variance-invariancecovariance regularization for self-supervised learning,” 2022. [Online]. Available: https://arxiv.org/abs/2105.04906

[20] J. Datta, R. Rabbi, P. Saha, A. N. Zereen, M. Abdullah-Al-Wadud, and J. Uddin, “Deep representation learning using layer-wise vicreg losses: J. datta et al.” Scientific Reports, vol. 15, no. 1, p. 27049, 2025.

[21] K. Kitsios, F. Sovrano, E. T. Barr, and A. Bacchelli, “Detecting semantic clones of unseen functionality,” in 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), 2025, pp. 1312– 1324.

[22] L. Marchezan, E. Syriani, K. Delcourt, and H. Sahraoui, “Automated Type-IV Clone Generation via LLMs and Deterministic Validation,” in 35th SIGSOFT International Symposium on Software Testing and Analysis (ISSTA 2026), 2026.

[23] A. I. Alam, P. R. Roy, F. Al-Omari, C. K. Roy, B. Roy, and K. A. Schneider, “Gptclonebench: A comprehensive benchmark of semantic clones and cross-language clones using gpt-3 model and semanticclonebench,” in 2023 IEEE International Conference on Software Maintenance and Evolution (ICSME), 2023, pp. 1–13.

[24] D. Chicco and G. Jurman, “The advantages of the matthews correlation coefficient (mcc) over f1 score and accuracy in binary classification evaluation,” BMC genomics, vol. 21, no. 1, p. 6, 2020.

[25] Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: a practical and powerful approach to multiple testing,” Journal of the Royal statistical society: series B (Methodological), vol. 57, no. 1, pp. 289–300, 1995.

[26] F. Deissenboeck, B. Hummel, E. Juergens, M. Pfaehler, and B. Schaetz, “Model clone detection in practice,” in Proceedings of the 4th International Workshop on Software Clones, ser. IWSC ’10. New York, NY, USA: Association for Computing Machinery, 2010, p. 57–64.

[27] Z. Chen and M. Monperrus, “A literature study of embeddings on source code,” arXiv preprint arXiv:1904.03061, 2019.

[28] L. Buch and A. Andrzejak, “Learning-based recursive aggregation of¨ abstract syntax trees for code clone detection,” in 2019 IEEE 26th International Conference on Software Analysis, Evolution and Reengineering (SANER). IEEE, 2019, pp. 95–104.

[29] D. DeFreez, A. V. Thakur, and C. Rubio-Gonzalez, “Path-based function´ embedding and its application to specification mining,” arXiv preprint arXiv:1802.07779, 2018.

[30] W. Zhang, S. Guo, H. Zhang, Y. Sui, Y. Xue, and Y. Xu, “Challenging machine learning-based clone detectors via semantic-preserving code transformations,” IEEE Transactions on Software Engineering, vol. 49, no. 5, pp. 3052–3070, 2023.

[31] Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 8, pp. 1798–1828, 2013.

[32] J. Li, C. Tao, Z. Jin, F. Liu, and G. Li, “Zc 3: Zero-shot cross-language code clone detection,” in 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2023, pp. 875–887.

[33] Z. Li, W. Chen, J. Yu, and Z. Lu, “Functional consistency of llm code embeddings: A self-evolving data synthesis framework for benchmarking,” Expert Systems with Applications, vol. 298, p. 129523, 2026.

[34] J. Svajlenko and C. K. Roy, “Evaluating clone detection tools with bigclonebench,” in 2015 IEEE International Conference on Software Maintenance and Evolution (ICSME), 2015, pp. 131–140.

[35] M. Zakeri-Nasrabadi, S. Parsa, M. Ramezani, C. Roy, and M. Ekhtiarzadeh, “A systematic literature review on source code similarity measurement and clone detection: Techniques, applications, and challenges,” Journal of Systems and Software, vol. 204, p. 111796, 2023.

[36] J. Svajlenko and C. K. Roy, “Bigclonebench: A retrospective and roadmap,” in 2022 IEEE 16th International Workshop on Software Clones (IWSC), 2022, pp. 8–9.

[37] J. Krinke and C. Ragkhitwetsagul, “Bigclonebench considered harmful for machine learning,” in 2022 IEEE 16th International Workshop on Software Clones (IWSC), 2022, pp. 1–7.

[38] ——, “How the misuse of a dataset harmed semantic clone detection,” 2025. [Online]. Available: https://arxiv.org/abs/2505.04311

[39] R. Puri, D. S. Kung, G. Janssen, W. Zhang, G. Domeniconi, V. Zolotov, J. Dolby, J. Chen, M. Choudhury, L. Decker, V. Thost, L. Buratti, S. Pujar, S. Ramji, U. Finkler, S. Malaika, and F. Reiss, “Codenet: A large-scale ai for code dataset for learning a diversity of coding tasks,” 2021.

[40] H. Husain, H.-H. Wu, T. Gazit, M. Allamanis, and M. Brockschmidt, “Codesearchnet challenge: Evaluating the state of semantic code search,” 2020.

[41] T. Y. Zhuo, M. C. Vu, J. Chim, H. Hu, W. Yu, R. Widyasari, I. N. B. Yusuf, H. Zhan, J. He, I. Paul et al., “Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions,” International Conference on Learning Representations, vol. 2025, pp. 66 602–66 656, 2025.

[42] F. Al-Omari, C. K. Roy, and T. Chen, “Semanticclonebench: A semantic code clone benchmark using crowd-source knowledge,” in 2020 IEEE 14th International Workshop on Software Clones (IWSC), 2020, pp. 57– 63.

[43] S. Kim and H. Lee, “Software systems at risk: An empirical study of cloned vulnerabilities in practice,” Computers & Security, vol. 77, pp. 720–736, 2018.

[44] M. B. Moumoula, A. K. Kabore, J. Klein, and T. F. Bissyand ´ e,´ “The struggles of llms in cross-lingual code clone detection,” Proc. ACM Softw. Eng., vol. 2, no. FSE, Jun. 2025. [Online]. Available: https://doi.org/10.1145/3715764

[45] J. W. Hardin and J. M. Hilbe, Generalized estimating equations. chapman and hall/CRC, 2002.

[46] A. P. Bradley, “The use of the area under the roc curve in the evaluation of machine learning algorithms,” Pattern recognition, vol. 30, no. 7, pp. 1145–1159, 1997.

[47] M. C. Christman and J. S. Pontius, “Bootstrap confidence intervals for adaptive cluster sampling,” Biometrics, vol. 56, no. 2, pp. 503–510, 2000.

[48] L. Marchezan, K. Delcourt, E. Syriani, and H. Sahraoui, “Type-iv code clone detection via layer-wise non- contrastive representation learning (replication package),” Jul. 2026. [Online]. Available: https: //doi.org/10.5281/zenodo.21363019

[49] B. Tabarsi, H. Reichert, S. Gilson, A. Limke, S. Kuttal, and T. Barnes, “Llms’ reshaping of people, processes, products, and society in software development: A comprehensive exploration with early adopters,” arXiv preprint arXiv:2503.05012, 2025.

[50] T. Sonnekalb, B. Gruner, C.-A. Brust, and P. Mader, “Generalizability of ¨ code clone detection on codebert,” in Proceedings ofthe 37th IEEE/ACM international conference on automated software engineering, 2022, pp. 1–3.

[51] Y. Zou, B. Ban, Y. Xue, and Y. Xu, “Ccgraph: a pdg-based code clone detector with approximate graph matching,” in Proceedings of the 35th IEEE/ACM international conference on automated software engineering, 2020, pp. 931–942.

[52] Y. Wu, S. Feng, D. Zou, and H. Jin, “Detecting semantic code clones by building ast-based markov chains model,” in Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, 2022, pp. 1–13.

[53] Y. Hu, D. Zou, J. Peng, Y. Wu, J. Shan, and H. Jin, “Treecen: Building tree graph for scalable semantic code clone detection,” in Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, 2022, pp. 1–12.

[54] A. K. Mohammed, B. Al-Attar, N. B. Pokale, D. Fallah, A. H. Ahmad, S. A. Mahdi, S. A. Azize, N. Divekar, and R. Sekhar, “Cross-language semantic code clone detection in large-scale software repositories using transformer-based contrastive learning,” in 2025 3rd International Conference on Cyber Resilience (ICCR). IEEE, 2025, pp. 1–8.

[55] Z. Li, D. Wang, S. Zhang, R. Hu, and D. Chen, “Bega: A binary code embedding method based on gravity-model augmentation graph contrastive learning,” in 2025 32nd Asia-Pacific Software Engineering Conference (APSEC). IEEE, 2025, pp. 231–242.

[56] Z. Zhang and T. Saber, “Machine learning approaches to code similarity measurement: A systematic review,” IEEE Access, vol. 13, pp. 51 729– 51 764, 2025.

[57] P. Keller, A. K. Kabore, L. Plein, J. Klein, Y. Le Traon, and T. F.´ Bissyande, “What you see is what it means! semantic representation´ learning of code based on visualization and transfer learning,” ACM Transactions on Software Engineering and Methodology (TOSEM), vol. 31, no. 2, pp. 1–34, 2021.

[58] C. Guo, H. Yang, D. Huang, J. Zhang, N. Dong, J. Xu, and J. Zhu, “Review sharing via deep semi-supervised code clone detection,” IEEE Access, vol. 8, pp. 24 948–24 965, 2020.

[59] Z. Zhang and T. Saber, “Assessing the code clone detection capability of large language models,” in 2024 4th International Conference on Code Quality (ICCQ). IEEE, 2024, pp. 75–83.