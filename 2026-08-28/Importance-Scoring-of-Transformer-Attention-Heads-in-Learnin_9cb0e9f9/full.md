# Importance Scoring of Transformer Attention Heads in Learning Tabular Data

Ahmad JadAllah, Kazi F. Akhter Department of Computer Science Tennessee State University Nashville, TN, USA

Md. Kamrozzaman Bhuiyan Enosis Solutions Dhaka, Bangladesh

Manar D. Samad Department of Computer Science North Carolina Agricultural and Technical University Greensboro, NC, USA

August 28, 2026

## ABSTRACT

Computationally demanding and opaque deep learning models can be better understood and optimized by analyzing how they transform data. While deep transformers have been widely studied in computer vision and natural language processing, their application in tabular data remains relatively underexplored. This paper presents one of the first applications of an importance-scoring metric to interpret multi-head transformer models in learning from tabular data. Experiments conducted on 40 diverse tabular datasets demonstrate robustness to head drops based on the proposed head importance score. In 72.5% of the experimental examples, the model remains most resilient to performance drop when heads with the lowest importance scores are gradually decreased. In contrast, removing the most important attention head first results in the greatest reduction in classification performance. A closer look at individual head importance scores across six attention layers reveals that important heads are scattered across layers without consistent layer-specific trends. In contrast to the image and language domains, the importance of individual attention heads varies considerably across tabular datasets that differ in their schemas and feature spaces. The proposed importance score can improve efficiency and redundancy within transformer architectures. The source code for measuring the importance of individual attention heads is made publicly available. <sup>1</sup>.

Keywords tabular data, multi-head attention, head drop, transformer, importance score

## 1 Introduction

Transformer models have been introduced primarily for language translation and decoding [1, 2], with applications that extend to computer vision [3]. Extensive research has been conducted to evaluate the efficacy of transformers in language and vision applications, especially those involving image and text data. In contrast, tabular data, structured in rows and columns, have seen limited research involving transformer models. Tabular data, which are prevalent in enterprise relational databases and spreadsheets across numerous application domains, are inherently distinct from the commonly studied text and image data. While images and text are sequential and homogeneous distributions of pixels and words, respectively, tabular data are a permutation-invariant collection of heterogeneous features [4, 5, 6]. As a result, numerous intuitions about deep learning that come from language and vision work are challenged when applied to tabular data. These structural differences among tabular datasets may cause individual attention heads to contribute differently across datasets, rather than following the hierarchical patterns observed in language and vision models.

Indeed, transformer models have seen impressive success in classifying [7, 8, 9], transfer learning [10, 11], and the imputation of missing values in tabular datasets [12]. In most cases, transformer architectures developed for large language models are directly adapted to learning tabular data without optimizing for a performance-computational cost trade-off. Weight pruning methods for optimizing transformers have been investigated using language data [13], which have not been explored well in the context of tabular data. Although extensive language corpora exist for training deep transformer models, such large-scale architectures are often not suitable for many tabular datasets with only a limited number of samples. This paper is among the first to quantify the contribution of individual attention heads of a deep transformer model in learning tabular data across different domains and scales. Measuring the contribution of each individual attention head supports selecting and pruning heads for model compression, and this process, in turn, yields an explainable deep learning framework for tabular transformer models.

The remainder of the paper is organized as follows. Section 2 provides background information and a brief review of the literature. Section 3 outlines the methodological framework for the head importance scoring metric. The experimental cases and evaluation methods are discussed in Section 4. Section 5 presents the key findings and their discussion. The paper concludes in Section 6.

## 2 Related work

Multiple attention heads enable a transformer to simultaneously capture diverse patterns from the input, each head attending to different aspects of the input data [1]. Quantifying the contribution of individual heads within an attention layer is central to understanding and controlling redundancy in multi-head transformer architectures. Transformers with multi-head attention have primarily been designed and extensively investigated to model language data. By contrast, because tabular data are heterogeneous and permutation-invariant in their feature space, they may engage with attention heads in a fundamentally different way than semantic word sequences in language data.

Transformer architectures have traditionally been configured as a rigid black box with a fixed number of attention layers and heads per layer, regardless of the target data domain or data types [1, 14]. For example, the first proposed transformer architecture [1] uses six attention layers with eight attention heads per layer. The language encoder model, BERT [14], has 12 layers with 12 heads per layer. Similar setups are routinely adopted across numerous application areas, thereafter, without further tuning to the particular task or dataset. There has been no standard approach for selecting or justifying the number of transformer layers and attention heads. Previous work has demonstrated that these standard transformer architectures are largely redundant in domain-specific applications [15]. Therefore, tailoring transformer architectures to specific datasets and schemas can improve performance, increase computational efficiency, and offer more transparent interpretations of otherwise opaque deep learning mechanisms.

This paper draws its inspiration from a related approach previously introduced to study redundancy in transformerbased language models. Michel et al. [13] have introduced a head-importance metric $I _ { h }$ based on the sensitivity of the loss to the activation of individual heads, which is used as a post-hoc criterion to identify and remove redundant heads from an already-trained model. However, the authors do not share the source code for their implementation of the head importance metric. To the best of our knowledge, similar studies have not been performed on tabular datasets with heterogeneous feature spaces. Many application domains provide datasets with limited sample sizes, which may not fully leverage the strength of deep learning methods.

This paper is one of the first to investigate the head-importance scoring strategy for studying redundancy within a multi-head transformer architecture when learning from tabular datasets across varying domains and schemas. The importance scores for the head are used to dynamically tune head activation during training. The final scores are used to drop heads cumulatively to interpret the importance of individual heads across multiple attention layers. We hypothesize that the vanilla transformer architecture is redundant for certain tabular datasets, which may benefit from data-specific model customization and provide useful interpretation of its contributions.

The contributions of this paper are as follows. First, we measure the importance of individual transformer attention heads for learning on tabular datasets with a heterogeneous feature space. Second, because the authors of [13] do not release their implementation, we implement the importance metric ourselves, adapting it to learning tabular data, and have made our code publicly available. Third, as an exploratory analysis, we examine a dynamic soft-tuning mechanism for attention head activation to study its effect across datasets. Fourth, we visualize the importance score of individual heads across different attention layers to support the interpretation of the otherwise opaque transformer model. Fifth, we conduct experiments on 40 tabular datasets that span various domains and sizes to highlight hetero geneity in transformer interpretations.

## 3 Methodology

This section introduces metrics that quantify the importance of attention heads in transformer models, along with an activation-tuning strategy used during transformer training.

## 3.1 Preliminaries on tabular data

While images are spatial distributions of pixels and language is a sequential arrangement of words, tabular data are arranged in a fixed schema with rows and columns. Tabular data consist of n samples organized in rows, where each row is described by d feature columns, forming a data matrix $\ b { X } \in \mathbb { R } ^ { n \times d }$ . Unlike uniform image pixels or words, the feature space in tabular data is heterogeneous in both types and scales. The diversity of feature spaces is further increased by the variety of application domains, in which the feature space may combine sparse, ordinal, numerical, and categorical variables. Although vision and language models often exhibit stable performance and similar behavior across many datasets, these observations generally do not extend to tabular data due to the aforementioned heterogeneity. The distinctive nature of tabular data and the importance of incorporating data-specific inductive biases are not widely acknowledged in the deep learning literature.

## 3.2 Attention head importance score

The head importance score measures the contribution of each attention head to the loss during training. Inspired by the work of Michel et al. [13] in the natural language domain, we obtain the importance score as the expected absolute inner product between head activation and the loss gradient with respect to that activation as follows.

$$
I _ { h } ^ { ( l ) } = \mathbb { E } _ { x \sim X } \left| \mathrm { A t t } _ { h } ^ { ( l ) } ( x ) ^ { \top } \cdot \frac { \partial \mathcal { L } ( x ) } { \partial \mathrm { A t t } _ { h } ^ { ( l ) } ( x ) } \right|\tag{1}
$$

Here, $\mathrm { A t t } _ { h } ( x ) \in \mathbb { R } ^ { d \times d _ { h } }$ is the activation output representation of the attention head h for input $x ,$ where d is the dimension of x and $d _ { h }$ is the dimension of the attention embedding. The expectation $\mathbb { E } _ { x \sim X }$ is taken over the data distribution $X$ . The absolute value skips the cancellation of opposite signs to reflect the magnitude of each head’s influence on the loss regardless of direction [13]. In practice, the exact computation of this expectation $\mathbb { E } _ { x \sim X }$ is infeasible. We therefore approximate $I _ { h }$ at each training epoch e by evaluating the importance over the full training set in mini-batches. Specifically, at epoch e, the training set is divided into mini-batches $B _ { 1 } , \ldots , B _ { K }$ . For each minibatch $B _ { k }$ , a forward and backward pass is performed, after which the per-head output tensor $\mathrm { A t t } _ { h } ^ { ( l ) } ( x ) \in \mathbb { R } ^ { | B _ { k } | \times d \times d _ { h } }$ and its gradient $\frac { \partial \mathcal { L } } { \partial \mathrm { A t t } _ { h } ^ { ( l ) } ( x ) }$ of the same shape are available. The per-batch importance is computed as the absolute value of the element-wise product summed over the batch, sequence, and head-dimension axes. These per-batch scores are then accumulated across all mini-batches and divided by the total number of training samples $N$ , producing the epoch-level estimate. The importance score of a head is obtained as the gradient-weighted response of attention heads. A higher gradient of training loss would increase the importance of the respective head.

## 3.3 Soft dynamic tuning of attention heads

We employ a continuous gating mechanism to dynamically regulate the activation of individual attention heads during training. For each head h in layer l, a gate value $\{ 0 \le \xi _ { h } ^ { ( l , e ) } \le 1 \}$ scales the head activation before it is combined with the outputs of the remaining heads in that layer:

$$
h _ { \mathrm { s c a l e d } } ^ { ( l ) } ( x ) = \xi _ { h } ^ { ( l , e ) } \cdot \mathrm { A t t } _ { h } ^ { ( l ) } ( x )\tag{2}
$$

where $\mathrm { A t t } _ { h } ^ { ( l ) } ( x )$ is the pre-gating head activation defined in Section III-A. The scaled activations of all K parallel heads in an attention layer are concatenated and projected through the output matrix. After epoch e, the importance vector containing individual head importance scores is as follows.

$$
{ \bf I } ^ { ( l , e ) } = \left( I _ { 1 } ^ { ( l , e ) } , \ldots , I _ { H } ^ { ( l , e ) } \right) .\tag{3}
$$

The gate value to tune the activation of each head (h) under layer l is obtained as follows.

$$
\xi _ { h } ^ { ( l , e ) } = \frac { I _ { h } ^ { ( l , e ) } } { \left\| \mathbf { I } ^ { ( l , e ) } \right\| _ { 2 } + \epsilon } ,\tag{4}
$$

where $\| \cdot \| _ { 2 }$ denotes the Euclidean norm computed over the K heads in layer l, and $\epsilon = 1 0 ^ { - 8 }$ prevents division by zero. Training involving soft tuning proceeds in two phases. During an initial warmup period lasting $E _ { w }$ epochs, a constant gate value (unity) is applied while updating only the model parameters. This enables the network to develop stable feature representations before activating the importance score-based tuning mechanism. Following the warm-up phase, the importance scores $I _ { h } ^ { ( l , e ) }$ are calculated over the entire training set using the method outlined in Section III-A, and the gate values are then updated according to Equation (4).

## 3.4 Transformer architecture

We employ the TransTab architecture [10], a transformer model designed for tabular data, using six attention layers, each containing eight attention heads, yielding a total of 48 heads. In TransTab, the feedforward network is configured with a hidden dimension of 256, employs ReLU as its activation function, and incorporates dropout. For representing feature names and categorical values, the transformer relies on embeddings derived from a pre-trained BERT [14] model. The categorical feature name $\mathrm { ( ^ { 6 6 } c o l o r ^ { 7 } ) }$ and its value $( ^ { 6 6 } \mathrm { r e d } ^ { \prime 3 } )$ are each tokenized and passed through the pretrained BERT model to obtain their respective embeddings, which are then concatenated to form the categorical feature embedding $E _ { \mathrm { { c a t } } }$ . For numerical features, the embedding of the feature name $( ^ { 6 6 } \mathrm { a g e } ^ { , 9 } )$ is multiplied element-wise by the scalar feature value (40) to produce the numerical feature embedding $E _ { \mathrm { n u m } }$ . The categorical and numerical embeddings are concatenated across all features of a sample to form a unified token sequence, which is passed as input to the transformer encoder.

## 4 Experiments and evaluations

Table 1 shows eight experimental cases. The first two scenarios compare model training with and without soft tuning, resulting in fixed performance scores without removing any heads. The next two baselines compare setups with and without soft tuning, where heads are randomly dropped only after the model has finished training. The following four scenarios consider cumulative head drop after completion of model training, progressing from the least important head with $I _ { m i n }$ to the most important head with $I _ { m a x } .$ , and vice versa. As a practical test of a head’s importance, one would expect that eliminating the most important head should cause the greatest drop in performance, thereby offering a way to validate the $I _ { h }$ metric. Intuitively, cumulative head removal is expected to show a gradual decline in downstream test accuracy. The cases that stand out as the strongest in downstream performance after head drop will be identified as the most resilient methods to head drops, facilitating the best performance-cost trade-off.

For model evaluation, datasets are divided into 70:20:10 ratios for training, validation, and test sets, respectively. The validation set is used only to select the best model during training, whereas the final evaluation is performed only on the left-out test data samples. We use the area under the receiver operating characteristic curve (AUC) as the evaluation metric. For multiclass datasets, the AUC is computed using the one-vs-rest (OVR) method with macro averaging. For experimental cases involving the removal of attention heads, we obtain test AUC scores as a function of the number of cumulative heads removed for individual datasets. The importance score is used to determine the order in cumulative head removal across six attention layers. We count the number of datasets in which an experimental case ranks bes throughout progressive head removal. The number of datasets achieving the highest AUC scores for an experimental case is determined when 10, 20, 30, and 40 heads are successively removed following the head importance scores. We also visualize the importance scores of 48 attention heads distributed across six attention layers from the bottom to the top of the transformer model. The statistical difference between the experimental cases is obtained using Wilcoxon signed-rank test across the 40 datasets. Cohen’s d was also calculated to report the effect size.

## 5 Results

## 5.1 Tabular datasets

We investigate the impact of data heterogeneity on outcomes by conducting experiments on 40 tabular datasets obtained from the OpenML [16] repository. Due to space limitations, we include the OpenML IDs for individual datasets in the source code repository. Figure 1 summarizes these datasets with varying size and dimensionality, with sample sizes ranging from 500 to 96,320 and feature counts ranging from 4 to 256 columns. Across all datasets, there are 1,644 features in total, consisting of 1,459 numerical features (88.75%) and 185 categorical or text-based features (11.25%), as illustrated in Figure 1(a). Figure 1(b) shows that most datasets contain between 1K and 5K samples, and fewer datasets include larger sample sizes (5K to 100K).

Table 1: Experimental scenarios designed to assess how effectively various attention-head soft-tuning and headdropping strategies support learning on tabular datasets.
<table><tr><td>Case</td><td>Soft tuning</td><td>Head dropping strategy</td><td>Comment</td></tr><tr><td>1</td><td>No</td><td>None</td><td>Fixed reference Fixed reference</td></tr><tr><td>2</td><td>Yes</td><td>None</td><td></td></tr><tr><td>3 4</td><td>No Yes</td><td>Random Random</td><td>Baseline Baseline</td></tr><tr><td>5</td><td>No</td><td></td><td>Drop the least important first</td></tr><tr><td>6</td><td>Yes</td><td> $I _ { m i n }  I _ { m a x }$   $I _ { m i n }  I _ { m a x }$ </td><td>Drop the least important first</td></tr><tr><td>7</td><td>No</td><td> $I _ { m a x }  I _ { m i n }$ </td><td>Dropping the most important heads</td></tr><tr><td>8</td><td>Yes</td><td> $I _ { m a x }  I _ { m i n }$ </td><td>Dropping the most important heads</td></tr></table>

![](images/322981c458c016c28e05c866f04402896c42fbd1daa5e694d58ab265920b0d0d.jpg)  
(a) Feature distribution

![](images/073b4d96b4234a2cfccde2ae6f9a9d111577ba88ad3145b13d24a4425c4dc511.jpg)  
(b) Distribution of tabular datasets  
Figure 1: Summary of the 40 tabular datasets used in this study. (a) Ratio of numerical and categorical features (b) Number of features and sample distributions.

## 5.2 Overall effects of head drop

Table 2 highlights how various tabular datasets respond differently in different experimental scenarios. Forty datasets are evaluated at four cumulative head-drop levels (10, 20, 30, and 40), resulting in a total of 160 examples. Among the 160 examples, the strategy of gradually removing heads starting from the lowest $I _ { h }$ values and progressing to higher $I _ { h }$ values yields the most robust performance, as demonstrated by 116 examples (72.5% of all examples) combined from cases 5 and 6 in Table 2. This outcome indicates that the $I _ { h }$ metric effectively identifies the heads that can be dropped with the least effect on the final classification task. The paired Wilcoxon signed-rank test confirms that dropping the least important heads first $( I _ { m i n }  I _ { m a x } )$ significantly $_ { ( \mathrm { p < 0 . 0 5 } ) }$ outperforms both random dropping and dropping the most important heads first $( I _ { m a x } \to I _ { m i n } )$ across all head-drop levels (Cohen’s $d = 0 . 4 0 \mathrm { t o } \ 0 . 8 0 )$ . The effectiveness of head importance score is substantiated when dropping the most important head yields significantly worse (p<0.05) performance than random dropping. Among the 116 examples, 35 (30%) show improved performance when head activations are subject to the proposed soft tuning during model training. The remaining 70% of the 116 examples do not benefit from soft tuning of head activations. In general, the heterogeneity of the tabular data schema, domain, and data types may play a major role in reaping the benefits of different learning, tuning, and dropping approaches. For example, 29 of the 160 examples (18%) show the best resilience when heads are dropped randomly, without using $I _ { h }$ scoring for head selection.

Figure 2 shows the mean percentage change in the classification AUC scores for six experimental cases in forty tabular datasets. When the 10 least important heads are dropped based on $I _ { h }$ scores, the vanilla transformer model yields strong resilience without requiring soft tuning of head activations. The severe negative effect of dropping the most important heads (dropping $I _ { h }$ (max) to $I _ { h }$ (min)) is evident in a significantly large drop in AUC scores. Random removal of heads produces a level of resilience that falls between the best case (removing heads in increasing order of their $I _ { h }$ scores) and the worst case (removing them in decreasing order of their $I _ { h }$ scores).

Table 2: Number of tabular datasets in which an experimental case yields the best performance at varying numbers of head drops.
<table><tr><td>Case</td><td>Soft tuning</td><td>Head dropping strategy</td><td colspan="3">Num. of heads dropped 10 20 30</td><td>40</td></tr><tr><td>3</td><td>No</td><td>Random</td><td>7</td><td>10</td><td>6</td><td>6</td></tr><tr><td>4</td><td>Yes</td><td>Random</td><td>2</td><td>1</td><td>0</td><td>1</td></tr><tr><td>5</td><td>No</td><td> $I _ { m i n }  I _ { m a x }$ </td><td>23</td><td>17</td><td>22</td><td>19</td></tr><tr><td>6</td><td>Yes</td><td> $I _ { m i n }  I _ { m a x }$ </td><td>5</td><td>10</td><td>9</td><td>11</td></tr><tr><td>7</td><td>No</td><td> $I _ { m a x }  I _ { m i n }$ </td><td>1</td><td>1</td><td>2</td><td>2</td></tr><tr><td>8</td><td>Yes</td><td> $I _ { m a x }  I _ { m i n }$ </td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td></td><td></td><td>Total datasets</td><td>40</td><td>40</td><td>40</td><td>40</td></tr></table>

![](images/7a134a033a8df2969ed0f7b7934d090ff962f31928c6499c8e200a62ed33ee58.jpg)  
Figure 2: Mean percentage change in AUC relative to the fixed reference (Case 1: no tuning, no dropping). The changes are averaged across 40 tabular datasets, at 10, 20, 30, and 40 cumulative heads dropped. Error bars denote one standard deviation.

## 5.3 Dataset-specific trends

Figure 3 shows the AUC trends with varying percentages of cumulative head drops. The figure illustrates how various tabular datasets exhibit differing degrees of robustness to the head-dropping strategy. The dashed blue and orange lines represent fixed references from case 1 and case 2, respectively. The gap between the two dashed lines indicates that soft tuning of head activations during model training can degrade final classification performance on certain tabular datasets.

Figure 3(a) presents an example using the Pendigits data set in which the soft tuning of head activations during training improves the robustness of the model to subsequent head drops. The Pendigits data set encodes handwritten digit strokes as spatially ordered coordinate sequences, which we speculate may benefit from soft tuning of the attention head activations. However, Figure 3(b) shows that soft tuning of head activations has a negative effect while training on the eye-movements dataset. In this example, soft tuning degrades the baseline model performance (case 2) compared to the case without tuning (case 1). The eye-movements dataset consists of sequential, non-image-based reading measurements, which may not be well suited for soft tuning of the attention heads. Figure 3(c) using the eucalyptus dataset shows similar baseline drop in performance. However, the soft-tuned model maintains its reference AUC scores even after dropping 60% of the total attention heads, whereas the vanilla model without soft tuning starts to lose performance after dropping 15 heads. The eucalyptus data set is small (736 samples) and has mixed feature types, so soft tuning of head activations may compensate for overfitting due to limited sample size.

(a) When $I _ { h }$ tuning of activations during training is the most (b) When the vanilla transformer model without activation robust to head dropping. Using the pendigits data set. tuning is the most robust to head dropping. Using the eye movements data set.  
Figure 3: Effects of dropping heads on AUC scores under different experimental cases.  
![](images/89aa47a118ecf42041141c791c703f9f8d4d67e6e814db2694dd8a0ee26c3a87.jpg)

![](images/e438b0e55f4724fb60b5ea68e84c072023821f37008aab7993f9f996c0c7a615.jpg)

![](images/f937b27509619a4288a5fca076989c32cd874e2e7b32d0f9dddf11b42a417c30.jpg)  
(c) $I _ { h }$ tuning of activation during training underperforms up to dropping 50% of heads but stays robust afterward compared to no tuning. Using the eucalyptus data set.

## 5.4 Visualization of head importance distribution

Vision models show that knowledge or data representations are hierarchically distributed across deep layers. Figure 4 illustrates the attention head importance score distributions for different datasets in different experimental cases. Attention heads with high importance scores are generally scattered throughout the six layers, without showing distinct layer-specific trends. Figure 4(a) illustrates that when the attention heads are soft-tuned during training on the pendigit datasets, the importance scores concentrate in the first heads of each attention layer. Several heads with top importance scores appear in the middle heads and attention layers. Compared to Figure 4(a), importance scores for the eye movement data in Figure 4(b) show most heads with low importance scores. Several heads with very high importance scores scatter around the mid region of attention layers and head distributions. Variation in head importance patterns may stem from soft tuning in head activations, which could promote a more uniform distribution of head importance across attention layers. In some datasets, the first attention layer contains more important heads than the top layer, though this ordering does not hold consistently across datasets. Figure 4(c) illustrates the importance score distribution when dropping heads gives better performance than the vanilla model using the PC3 dataset. In this figure, the lower portion of head 4 (attention layers 0–2) shows high importance scores, in sharp contrast to the upper portion of the attention layers. The opposite is observed for Head 1, where the top half of the attention layers contains the most important heads. Layer-wise, layer 3 has the concentration of the least important heads, except one head that is the most important (score: 0.897) of all heads combined. The PC3 data set consists of software complexity metrics that are mathematically derived from source code. Many of these metrics are inherently correlated, which we speculate may contribute to the concentration of importance in isolated heads.

## 5.5 Summary of results

This paper presents one of the first efforts to study the importance of attention heads in the multi-head transformer architecture while learning tabular datasets. The key findings of this paper can be summarized as follows. First, tabular datasets vary greatly across domains and schemas, and their behavior when processed by multi-head transformers is likewise highly diverse. Therefore, a general pattern or hierarchy observed in vision and language models may not apply to tabular data. Second, the $I _ { h }$ metric is effective in identifying the attention heads that are important for classification tasks that involve tabular datasets. Therefore, head importance scores can be used to measure redundancy in the model, thus helping to interpretability and model compression. Third, the locations of important attention heads vary across datasets, with no consistent concentration in any particular attention layer.

![](images/731c662e787b14e53caa69d28b2e7169daa3672e8c7f4d5fe1b5963968fe871e.jpg)  
(a) When $I _ { h }$ tuning of activations during training and then head dropping is the best approach. Using the pendigits data set.

![](images/2921f5fcf7f29ea271f13298e8b62ffd22157b50cfac7dffce9670519e2f45d4.jpg)  
(b) When head drop without $I _ { h }$ tuning of activations is the best approach. Using the eye movements data set.

![](images/0e8d8237cf6b6354ee534c3b03772d30b46a020927cccf26bc6b51a46a27d10c.jpg)  
(c) When head drop after $I _ { h }$ tuning outperforms the case without dropping. Using the pc3 data set.  
Figure 4: Visualization of attention head importance scores $( I _ { h } )$ for three representative learning scenarios.

## 6 Conclusions

This paper develops and evaluates an importance score to interpret how individual attention heads contribute to learning tabular data in multi-head transformer models. The impact of individual heads differs substantially across datasets, indicating that attention heads do not all contribute equally and that attention layers lack any consistent ordering or hierarchy in terms of head-level contributions. One limitation of our work is that all experiments use a single tabular transformer architecture, TransTab. Therefore, gaining insight into how attention heads operate on a particular dataset across different transformer architectures is essential for making effective use of transformer models.

## References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30.

[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al., Language models are unsupervised multitask learners, OpenAI blog 1 (8) (2019) 9.

[3] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, 2020.

[4] S. B. Rabbani, I. V. Medri, M. D. Samad, Attention versus contrastive learning of tabular data: a data-centric benchmarking, International Journal of Data Science and Analytics (2024) 1–23.

[5] L. Grinsztajn, E. Oyallon, G. Varoquaux, Why do tree-based models still outperform deep learning on typical tabular data?, in: Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https://openreview.net/forum?id=Fp7\_\_phQszn

[6] V. Borisov, T. Leemann, K. Seßler, J. Haug, M. Pawelczyk, G. Kasneci, Deep neural networks and tabular data: A survey, IEEE Transactions on Neural Networks and Learning Systems.

[7] N. Hollmann, S. Muller, L. Purucker, et al., Accurate predictions on small data with a tabular foundation model,¨ Nature 637 (8010) (2025) 319–326. doi:10.1038/s41586-024-08328-6.

[8] G. Somepalli, A. Schwarzschild, M. Goldblum, C. B. Bruss, T. Goldstein, Saint: Improved neural networks for tabular data via row attention and contrastive pre-training, in: NeurIPS 2022 First Table Representation Workshop.

[9] J. Kossen, N. Band, C. Lyle, A. N. Gomez, T. Rainforth, Y. Gal, Self-attention between datapoints: Going beyond individual input-output pairs in deep learning, Advances in Neural Information Processing Systems 34 (2021) 28742–28756.

[10] Z. Wang, J. Sun, Transtab: Learning transferable tabular transformers across tables, in: A. H. Oh, A. Agarwal, D. Belgrave, K. Cho (Eds.), Advances in Neural Information Processing Systems, 2022.

[11] I. Kowsar, K. F. Akhter, M. D. Samad, Llm attention transplant for transfer learning of tabular data across disparate domains, arXiv preprint arXiv:2511.06161.

[12] I. Kowsar, S. B. Rabbani, Y. Hou, M. D. Samad, Deepifsac: Deep imputation of missing values using feature and sample attention within contrastive framework, Knowledge-Based Systems 318 (2025) 113506.

[13] P. Michel, O. Levy, G. Neubig, Are sixteen heads really better than one?, in: H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alche-Buc, E. Fox, R. Garnett (Eds.), Advances in Neural Information Processing Sys- ´ tems, Vol. 32, Curran Associates, Inc., 2019.

[14] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171– 4186.

[15] A. Bao, V. H. Vattikuti, J. Lai, W. Gilpin, Universal redundancies in time series foundation models, arXiv preprint arXiv:2602.01605.

[16] J. Vanschoren, J. N. van Rijn, B. Bischl, L. Torgo, Openml: networked science in machine learning, SIGKDD Explorations 15 (2) (2013) 49–60. doi:10.1145/2641190.2641198.