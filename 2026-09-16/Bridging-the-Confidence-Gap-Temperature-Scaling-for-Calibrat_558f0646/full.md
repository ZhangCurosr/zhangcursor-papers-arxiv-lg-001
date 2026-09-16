# Bridging the Confidence Gap: Temperature Scaling for Calibrating Test-Time Prompt Tuning

Yuwei Liang<sup>1,2</sup> Jian Liang<sup>1,2∗</sup> Dapeng Hu<sup>3</sup> Yinuo Xu<sup>1,2</sup> Ran He<sup>1,2</sup>

<sup>1</sup> School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, China <sup>2</sup> NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences, Beijing, China

<sup>3</sup> Independent Researcher

liangyuwei911@gmail.com, liangjian92@gmail.com

## Abstract

Test-time prompt tuning (TPT) enables adaptation on a single test instance, achieving improved accuracy but often sacrificing calibration performance. Most existing calibration methods introduce additional regularization terms to promote dispersion across text embeddings and reduce calibration error, yet these methods often suffer from a drop in accuracy. Motivated by the well-calibrated nature of zero-shot predictions, we propose CoTS, a simple yet effective post-hoc calibration method that preserves accuracy. Specifically, CoTS applies temperature scaling to minimize the confidence gap between adapted and zero-shot predictions. To fully exploit the potential of multiple augmentations during adaptation, we introduce a weak-strong ensemble strategy that further boosts accuracy. We then apply CoTS to this ensemble, termed E-CoTS, to maintain its well-calibrated property. Extensive experiments on diverse datasets and backbones show that our approaches effectively mitigate miscalibration without compromising primary accuracy. For instance, E-CoTS reduces the average expected calibration error of TPT from 11.90% to 5.38% on ImageNet variants, while even increasing accuracy from 60.74% to 62.95%. Moreover, when integrated with existing calibration methods, E-CoTS usually enhances both accuracy and calibration simultaneously. Code is available at https://github.com/yuweiliang911/CoTS.

## 1 Introduction

Vision-language models (VLMs), such as CLIP [53] and ALIGN [26], are pre-trained to align visual and textual features in a shared embedding space, enabling strong zero-shot performance. To efficiently adapt VLMs to diverse downstream tasks, pioneering works [77, 76] replace hand-crafted prompts with learnable continuous vectors optimized on training data. For unsupervised settings, earlier methods typically require access to multiple samples from either the entire test set [24, 62, 38] or streaming data [42, 69]. In contrast, episodic test-time adaptation (TTA) methods [57, 12, 56] perform prompt tuning using a single test instance, which is more challenging and has received increasing attention. A classic approach, test-time prompt tuning (TPT) [57] adapts CLIP during inference by minimizing the marginal entropy over multiple augmented predictions.

Although TPT [57] improves accuracy over zero-shot CLIP, it is prone to generating overconfident predictions, raising reliability concerns for VLMs in safety-critical applications [29, 50, 31, 58]. A pioneering work, C-TPT [72], shows that well-calibrated prompts tend to produce text embeddings with broader class-wise dispersion and thus introduces a distance-dispersion regularization term to push prompt embeddings away from the class centroid. Building on this insight, subsequent works [55, 1, 13] further propose various regularization terms to encourage angular separation

![](images/be1295f0a7deaead6fb02d1d8ff2ce52a37d11a69d858bbe207c65ff95d9a53e.jpg)  
Figure 1: Average accuracy (%) and expected calibration error (ECE, %) of zero-shot CLIP, TPT, and various calibration methods on fine-grained datasets and ImageNet variants using ViT-B/16.  
between textual features. Although these regularization-based methods achieve lower expected calibration error (ECE) than the original TPT [57], we find they suffer from a noticeable accuracy drop, as shown in Fig. 1. We argue that reduced ECE is less meaningful when it comes at the expense of accuracy, especially as the underlying calibration mechanisms remain poorly interpretable.

Alternatively, SaLS [46] presents the only post-hoc calibration method that preserves accuracy by adjusting logits to remain within the zero-shot range after inference, but its calibration improvement remains limited. It is well known that CLIP [53] is generally well-calibrated [44, 15], with a relatively small gap between accuracy and confidence. Taking a closer look at the accuracy of CLIP and TPT, we find that prompt tuning using a single test instance does not substantially alter accuracy, suggesting that the increased miscalibration stems from a shift in confidence. To validate this hypothesis, we conduct a pilot study that assigns zero-shot confidence to TPT predictions. We find that this straightforward replacement significantly reduces calibration error, surpassing [46, 72] while falling short of [55, 13]. However, this naive confidence replacement introduces a critical inconsistency: the predicted class comes from TPT’s softmax distribution, whereas the confidence comes from the zero-shot distribution, potentially leading to unstable decisions.

Based on these insights, we propose Confidence-based Temperature Scaling (CoTS), a simple yet effective calibration method that follows the classic idea of temperature scaling [16]. Specifically, CoTS optimizes a learnable temperature parameter to align TPT-predicted confidence with zero-shot confidence, thereby mitigating calibration error while retaining accuracy. Furthermore, inspired by prior work [10, 6] that ensembles augmented views to enhance prediction robustness, we introduce a new weak-strong ensemble strategy to further boost accuracy. In this strategy, we retain the weakly augmented view and balance its predictions with those from selected strong augmentations via a weighted trade-off. Empirically, we find that this ensemble strategy indeed boosts accuracy but may disrupt the model’s well-calibrated properties. To mitigate this issue, we further integrate the ensemble strategy with CoTS, termed E-CoTS, achieving a better trade-off between accuracy and calibration. We validate the proposed methods on both fine-grained datasets and ImageNet variants.

To summarize, our contributions are as follows:

• We propose CoTS, a novel post-hoc confidence-based calibration method that uses temperature scaling to align TPT-predicted confidence with zero-shot confidence.

• We devise a simple weak-strong ensemble strategy integrated with CoTS, dubbed E-CoTS, that achieves reliable classification against TPT with superior accuracy and lower calibration error.

• Experiments across diverse datasets and backbones show that our methods reduce ECE to near the well-calibrated zero-shot baseline, while maintaining or even improving the adapted model’s classification accuracy. Both CoTS and E-CoTS are plug-and-play and usually complement existing regularization-based calibration approaches, improving either accuracy or ECE.

## 2 Preliminaries and Motivation

CLIP-based zero-shot classification. CLIP [53] consists of a visual encoder $f _ { v } ( \cdot )$ and a text encoder $f _ { t } ( \cdot )$ . For a K-class classification task with label space $\mathcal { Y } = \{ y _ { 1 } , y _ { 2 } , . . . , \hat { y _ { K } } \}$ , let x denote an input image and $y \in \mathcal { V }$ its ground-truth label. The visual encoder maps x to an image embedding $v \doteq f _ { v } ( x ) \in \mathbb R ^ { d }$ , while each class label $y _ { k } \in \mathcal { V }$ is converted into a textual prompt $c _ { k } \ ( e . g .$ ., using the template $" \mathrm { a }$ photo of a [CLASS]") and then encoded into a text embedding $t _ { k } = f _ { t } ( c _ { k } )$ . Both the image and text embeddings are $l _ { 2 } .$ -normalized, $i . e . , \| v \| = \| t _ { k } \| = 1$ . Then, the k-th values of the logit vector z and softmax probability vector $\mathbf { p } ( x )$ corresponding to class $y _ { k }$ are defined as follows:

$$
\mathbf { z } _ { k } = ( { \boldsymbol { v } } ^ { T } \cdot { \boldsymbol { t } } _ { k } ) / { \tau } _ { \mathrm { c l i p } } , \quad \mathbf { p } _ { k } ( x ) = \mathbf { p } ( y = k | x ) = { \frac { \exp ( \mathbf { z } _ { k } ) } { \sum _ { j = 1 } ^ { K } \exp ( \mathbf { z } _ { j } ) } } ,\tag{1}
$$

where the scaling factor $\tau _ { \mathrm { c l i p } } = 0 . 0 1$ is learned during pre-training [53].

Test-time prompt tuning (TPT). TPT [57] adapts CLIP [53] to a single test image by optimizing text prompts at test time, following the test-time adaptation paradigm [61, 75, 37]. Besides the weak augmented view $A _ { 1 } ( x )$ , TPT generates $( N - 1 )$ randomly augmented views of the test image using AugMix [21], and optimizes the input text prompts by minimizing the marginal entropy loss:

$$
\mathcal { L } _ { \mathrm { T P T } } = - \sum _ { k = 1 } ^ { K } \bar { \mathbf { p } } _ { k } ( x ) \log \bar { \mathbf { p } } _ { k } ( x ) , \quad \bar { \mathbf { p } } _ { k } ( x ) = \frac { 1 } { \rho N } \sum _ { i = 1 } ^ { N } \mathbb { 1 } [ \mathbf { H } ( \mathbf { p } ( \mathcal { A } _ { i } ( x ) ) ) \leq \gamma ] ~ \mathbf { p } _ { k } ( \mathcal { A } _ { i } ( x ) ) .\tag{2}
$$

Here $\bar { \mathbf p } _ { k } ( x )$ is the average softmax probability over selected augmentations that correspond to small entropy values, $\gamma$ is the threshold corresponding to a cutoff percentile $\rho$ (defaulting to 0.1 [57]), and $\mathbf { H } ( \mathbf { p } ( \bar { \mathcal { A } } _ { i } ( x ) ) )$ measures the self-entropy of the prediction on the i-th augmented view.

Temperature scaling. Temperature Scaling (TS) [16] is a classic post-hoc calibration technique [52] that adjusts model confidence without altering its predictions. It introduces a single temperature parameter $\tau > 0$ to rescale the logits before applying the softmax:

$$
\hat { \mathbf { p } } _ { k } ( x ; \tau ) = \frac { \exp ( \mathbf { z } _ { k } / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( \mathbf { z } _ { j } / \tau ) } , \quad k \in [ 1 , \ldots , K ] ,\tag{3}
$$

where $\mathbf { z } _ { k }$ is the logit for class k. A larger τ softens the distribution (reducing overconfidence), while a smaller τ sharpens it. The optimal temperature $\tau ^ { * }$ is typically obtained by minimizing the negative log-likelihood (NLL) on a labeled validation set $V = \{ x _ { i } , y _ { i } \} _ { i = 1 } ^ { | V | }$

$$
\boldsymbol { \tau } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { \tau } } - \sum _ { i = 1 } ^ { | V | } \log \hat { \mathbf { p } } _ { y _ { i } } ( x _ { i } ; \boldsymbol { \tau } ) .\tag{4}
$$

However, no labeled validation set is available in the episodic test-time adaptation problem, motivating alternative strategies for determining the temperature.

## 2.1 A Critical Examination of Prior Calibration Methods

Although TPT [57] improves accuracy compared to zero-shot CLIP [53], recent studies have shown that it produces overconfident predictions [46, 72]. To address this, a line of work has focused on calibrating TPT [72, 55, 17, 1, 13]. These methods introduce regularization terms during prompt optimization to encourage greater dispersion among text embeddings, $e . g .$ , distance-based dispersion [72], orthogonal constraints [55], angular diversity [1], and semantic-aware separation [13]. Alternatively, SaLS [46] identifies the increase in logit range as the cause of miscalibration and adjusts logits to remain within the zero-shot range. To investigate whether these methods truly work as expected, we examine their ECE values and accuracies alongside those of CLIP and TPT in Fig. 1.

Observation 1: Calibration methods reduce TPT’s calibration error but do not significantly outperform zero-shot CLIP. As depicted in Fig. 1, zero-shot predictions maintain low ECE values, confirming that CLIP models are generally well-calibrated [44, 15]. While TPT [57] clearly boosts accuracy, it leads to a substantial increase in ECE. Existing calibration methods significantly reduce the ECE of TPT, but most remain comparable to or slightly worse than zero-shot CLIP; only SoC [13] marginally outperforms it. The only post-hoc approach, SaLS [46], achieves notably weaker ECE reduction than its regularization-based counterparts.

Observation 2: Calibration methods improve accuracy over CLIP but often degrade the accuracy of TPT. As depicted in Fig. 1, all regularization-based methods decrease the accuracy of

TPT. Specifically, the accuracy of O-TPT [55] and SoC [13] degrade substantially, even approaching that of zero-shot CLIP. In contrast, the post-hoc approach SaLS [46] fully preserves TPT’s accuracy.

Taken together, zero-shot CLIP serves as a strong calibration baseline, while TPT provides a clear accuracy gain. Existing regularization-based methods trade accuracy for calibration, whereas SaLS [46] preserves accuracy but offers limited calibration improvement. This reveals an unresolved gap: no existing method simultaneously preserves TPT’s accuracy advantage and reduces calibration error to the well-calibrated zero-shot level.

## 2.2 Confidence Replacement: A Pilot Study

Comparing the accuracy of zero-shot CLIP and TPT, we find that adaptation using a single test instance typically yields limited accuracy gains, suggesting that TPT’s miscalibration stems from a shift in confidence rather than a change in predictive performance. To test this hypothesis, we conduct a simple experiment: for each test sample, we retain the predicted class from TPT but replace its confidence (maximum predicted probability) with the zero-shot confidence of the same sample, and then measure ECE. We refer to this procedure as Confidence Replacement (CoR).

As shown in Fig. 2, this simple replacement leads to a notable reduction in ECE, even outperforming

![](images/c24380f8d1394efa8304cb90f128435a10cf3a71b29e56595a25a508ada7afb5.jpg)  
Figure 2: Comparison of average ECE (%) between CoR and existing calibration methods on fine-grained datasets and ImageNet variants using ViT-B/16.

SaLS [46] and C-TPT [72]. However, CoR remains inferior to recent calibration methods such as SoC [13]. We attribute this to a fundamental inconsistency: the predicted class is derived from the TPT distribution, whereas the confidence comes from the zero-shot distribution, which may lead to incoherent decisions. Nevertheless, this experiment confirms two key findings: (1) the miscalibration of TPT is largely caused by confidence inflation, and (2) zero-shot confidence provides a useful reference for calibration. These observations motivate a principled approach that aligns TPT confidence with zero-shot confidence while maintaining a consistent probability distribution.

## 3 Methodology

Section 3.1 describes the proposed confidence-based temperature scaling method, while Section 3.2 then presents its enhanced variant under a new weak-strong ensemble strategy.

## 3.1 Confidence-based Temperature Scaling

Based on the findings in Section 2.2, we propose Confidence-based Temperature Scaling (CoTS), a post-hoc calibration method that follows the classic idea of temperature scaling [16]. Unlike standard temperature scaling, which requires a labeled validation set to optimize the NLL, CoTS leverages the well-calibrated zero-shot predictions as a label-free reference. Specifically, CoTS optimizes a learnable temperature parameter τ to align TPT-predicted confidence with zero-shot confidence, thereby reducing calibration error while retaining accuracy.

Let $\mathbf { z } ^ { \mathrm { t p t } } ( x )$ and $\mathbf { z } ^ { z s } ( x )$ denote the logits of the TPT-adapted model and the zero-shot model for the test image x, respectively. For a single augmented view, the objective is to minimize the squared difference between the temperature-scaled TPT confidence and the zero-shot confidence: $\begin{array} { r } { \left( \operatorname* { m a x } _ { k } \hat { \mathbf { p } } _ { k } ^ { \mathrm { t p t } } ( x ; \tau ) - \operatorname* { m a x } _ { k } \mathbf { p } _ { k } ^ { z \mathrm { s } } ( x ) \right) ^ { 2 } } \end{array}$ , where $\hat { \mathbf { p } } ^ { \mathrm { t p t } }$ and $\mathbf { p } ^ { z \mathbf { s } }$ are the corresponding softmax probabilities. To fully exploit the multiple augmentations available in TPT and reduce the sensitivity of relying on the weak view itself, we optimize over the set of selected strongly augmented views:

$$
\boldsymbol { \tau } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { \tau } } \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } \left( \operatorname* { m a x } _ { k } \hat { \mathbf { p } } _ { k } ^ { \mathrm { t p t } } ( \boldsymbol { A } _ { i } ( \boldsymbol { x } ) ; \boldsymbol { \tau } ) - \operatorname* { m a x } _ { k } \mathbf { p } _ { k } ^ { \mathrm { z s } } ( \boldsymbol { A } _ { i } ( \boldsymbol { x } ) ) \right) ^ { 2 } ,\tag{5}
$$

where $\Omega = \{ i \mid i > 1 , { \bf H } ( { \bf p } ^ { \mathrm { t p t } } ( { \cal A } _ { i } ( x ) ) ) \leq \gamma ^ { \prime } \}$ denotes the set of strongly augmented views selected by having low entropy after adaptation under the threshold $\gamma ^ { \prime } .$ , excluding the weakly augmented view. Once the optimal parameter $\tau ^ { * }$ is obtained, the calibrated probability is computed as $\hat { \mathbf { p } } ^ { \mathrm { t p \pm } } ( \mathcal { A } _ { 1 } ( x ) ; \tau ^ { * } )$

Compared to existing regularization-based calibration methods [72, 55, 1, 13], which modify the prompt optimization objective and may interfere with discriminative performance, CoTS operates entirely after adaptation and thus avoids any accuracy degradation.

## 3.2 Ensemble with Confidence-Based Temperature Scaling

While CoTS reduces calibration error without degrading the accuracy of TPT, it mainly preserves rather than improves the adapted prediction. We wonder whether we can further improve accuracy while maintaining good calibration, a goal that is especially important since existing calibration methods (e.g., O-TPT [55], SoC [13]) often suffer from a notable accuracy degradation. Orthogonal to these calibration methods, ensembling over augmented views provides an effective way to boost accuracy during CLIP adaptation [10, 56, 6], and ZERO [10] theoretically shows that marginalizing predictions across augmented views can reduce the error bound compared with single-view inference.

Inspired by these insights, we propose a weak-strong ensemble strategy that further exploits the predictions from selected strongly augmented views to boost accuracy. Unlike prior work that uses majority voting [10] or uniform averaging [6] over augmented views, we balance the predictions from weak and strong augmentations using an adaptive weight:

$$
{ \bf p } _ { \mathrm { e n s } } = \alpha \cdot { \bf p } ^ { \mathrm { t p t } } ( \mathcal { A } _ { 1 } ( x ) ) + ( 1 - \alpha ) \cdot \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } { \bf p } ^ { \mathrm { t p t } } ( \mathcal { A } _ { i } ( x ) ) ,\tag{6}
$$

where $\begin{array} { r } { \alpha = \frac { 1 } { K ( K - 1 ) } \sum _ { i \neq j } t _ { i } ^ { T } \cdot t _ { j } } \end{array}$ is the average pairwise cosine similarity among the K normalized text embeddings $\{ t _ { i } \} _ { i = 1 } ^ { K }$ . Intuitively, a higher α indicates greater inter-class similarity in the text embedding space, which makes entropy-based view selection less discriminative (i.e., the average of strong views becomes relatively unreliable). In such cases, the weakly augmented view (original im age) is more reliable, and the ensemble assigns it a larger weight. Conversely, when text embeddings are well-separated (α is small), the entropy-filtered strongly augmented views provide more reliable complementary information. Note that α depends only on the text embeddings and thus varies across tasks and prompts, providing a task-adaptive weighting without per-sample tuning.

Through a preliminary study, we find that directly applying this ensemble on top of calibrated models indeed enhances accuracy but may disrupt their well-calibrated properties. To address this, we integrate the proposed confidence-based temperature scaling into the ensemble strategy. Specifically, we substitute each probability vector in Eq. (6) with its temperature-calibrated version:

$$
\hat { \mathbf { p } } _ { \mathrm { e n s } } = \alpha \cdot \hat { \mathbf { p } } ^ { \mathrm { t p t } } ( \mathcal { A } _ { 1 } ( x ) ; \tau ^ { * } ) + ( 1 - \alpha ) \cdot \frac { 1 } { | \Omega | } \sum _ { i \in \Omega } \hat { \mathbf { p } } ^ { \mathrm { t p t } } ( \mathcal { A } _ { i } ( x ) ; \tau ^ { * } ) .\tag{7}
$$

The combined approach, termed E-CoTS, achieves a better trade-off between accuracy and calibration: the ensemble component enhances classification performance, while temperature scaling reduces calibration error. Both methods are post-hoc, and their pseudocode is provided in Appendix C.

## 4 Experiment

## 4.1 Experimental Setup

Datasets. To comprehensively evaluate the performance of CoTS and E-CoTS, we first use a finegrained benchmark consisting of eleven image classification datasets, including ImageNet [7], DTD [5], Flowers102 [48], Food101 [3], SUN397 [68], Aircraft [43], OxfordPets [51], Caltech101 [11], UCF101 [59], EuroSAT [19], and StanfordCars [32]. To further evaluate robustness under natural distribution shifts, we conduct experiments on four ImageNet variants, including

Table 1: Performance comparison of different methods on fine-grained datasets using ViT-B/16. Accuracy (%) and ECE (%) are reported. In the Average column, green indicates performance improvement over TPT, while red indicates performance degradation.
<table><tr><td>(%)</td><td>Method</td><td>ImgNet</td><td>DTD</td><td>Flowers</td><td>Food101</td><td>SUN397</td><td>Aircraft</td><td>Pets</td><td>Caltech</td><td>UCF101</td><td>EuroSAT</td><td>Cars</td><td>Average</td></tr><tr><td rowspan="10">Aacy</td><td>Zero-Shot [53]</td><td>66.72</td><td>44.33</td><td>67.32</td><td>83.66</td><td>62.58</td><td>23.85</td><td>88.20</td><td>93.96</td><td>65.19</td><td>42.05</td><td>65.56</td><td>63.95</td></tr><tr><td>TPT [57]</td><td>68.91</td><td>47.12</td><td>68.64</td><td>84.65</td><td>65.50</td><td>23.27</td><td>87.25</td><td>94.08</td><td>68.20</td><td>42.91</td><td>66.58</td><td>65.19</td></tr><tr><td>C-TPT [72]</td><td>68.45</td><td>45.17</td><td>69.58</td><td>83.14</td><td>64.53</td><td>24.10</td><td>88.20</td><td>93.75</td><td>65.05</td><td>42.45</td><td>65.80</td><td>64.57-0.63</td></tr><tr><td>Penalty [46]</td><td>68.84</td><td>45.80</td><td>68.36</td><td>84.43</td><td>65.44</td><td>22.75</td><td>82.94</td><td>93.27</td><td>67.09</td><td>41.20</td><td>66.19</td><td>64.21-0.98</td></tr><tr><td>SaLS [46]</td><td>68.91</td><td>47.12</td><td>68.67</td><td>84.65</td><td>65.49</td><td>23.29</td><td>87.25</td><td>94.08</td><td>68.21</td><td>42.91</td><td>66.57</td><td>65.20+0.00</td></tr><tr><td>O-TPT [55]</td><td>67.30</td><td>45.49</td><td>69.03</td><td>82.78</td><td>63.12</td><td>23.65</td><td>88.13</td><td>93.54</td><td>63.82</td><td>42.44</td><td>65.24</td><td>64.05-1.14</td></tr><tr><td>A-TPT [1]</td><td>67.92</td><td>45.17</td><td>69.79</td><td>83.06</td><td>63.20</td><td>23.80</td><td>87.89</td><td>93.24</td><td>64.03</td><td>42.05</td><td>65.20</td><td>64.12-1.07</td></tr><tr><td>SoC [13]</td><td>67.57</td><td>42.79</td><td>68.01</td><td>83.60</td><td>62.37</td><td>24.01</td><td>88.30</td><td>94.01</td><td>65.47</td><td>42.04</td><td>65.31</td><td>63.95-1.24</td></tr><tr><td>CoTS</td><td>68.91</td><td>47.12</td><td>68.64</td><td>84.65</td><td>65.50</td><td>23.27</td><td>87.25</td><td>94.08</td><td>68.20</td><td>42.91</td><td>66.58</td><td>65.19+0.00</td></tr><tr><td>E-CoTS</td><td>69.49</td><td>47.32</td><td>68.25</td><td>84.54</td><td>65.67</td><td>23.59</td><td>86.94</td><td>94.13</td><td>68.25</td><td>41.68</td><td>67.57</td><td>65.22+0.03</td></tr><tr><td rowspan="10">ECE</td><td>Zero-Shot [53]</td><td>1.98</td><td>8.24</td><td>2.73</td><td>2.08</td><td>2.24</td><td>5.53</td><td>4.45</td><td>5.91</td><td>2.95</td><td>7.10</td><td>4.40</td><td>4.33</td></tr><tr><td>TPT [57]</td><td>10.56</td><td>20.96</td><td>13.59</td><td>4.31</td><td>11.23</td><td>16.96</td><td>5.50</td><td>4.36</td><td>11.57</td><td>20.15</td><td>5.08</td><td>11.30</td></tr><tr><td>C-TPT [72]</td><td>5.14</td><td>12.94</td><td>5.16</td><td>3.39</td><td>5.10</td><td>4.16</td><td>1.63</td><td>4.41</td><td>2.49</td><td>11.66</td><td>1.47</td><td>5.23-6.07</td></tr><tr><td>Penalty [46]</td><td>10.38</td><td>14.88</td><td>12.08</td><td>2.67</td><td>10.98</td><td>15.65</td><td>1.84</td><td>4.89</td><td>9.14</td><td>4.62</td><td>3.94</td><td>8.28-3.02</td></tr><tr><td>SaLS [46]</td><td>9.72</td><td>18.61</td><td>11.93</td><td>4.35</td><td>11.01</td><td>15.76</td><td>5.12</td><td>4.37</td><td>10.66</td><td>13.83</td><td>3.94</td><td>9.94-1.36</td></tr><tr><td>O-TPT [55]</td><td>2.03</td><td>7.66</td><td>3.68</td><td>4.28</td><td>8.42</td><td>3.76</td><td>2.13</td><td>4.53</td><td>2.70</td><td>11.52</td><td>1.81</td><td>4.77-6.52</td></tr><tr><td>A-TPT [1]</td><td>2.38</td><td>8.49</td><td>4.23</td><td>3.19</td><td>4.17</td><td>6.25</td><td>2.18</td><td>5.30</td><td>2.63</td><td>7.11</td><td>1.68</td><td>4.33-6.97</td></tr><tr><td>SoC [13]</td><td>3.71</td><td>7.63</td><td>2.98</td><td>2.84</td><td>2.84</td><td>5.41</td><td>1.83</td><td>5.80</td><td>2.71</td><td>7.08</td><td>4.63</td><td>4.31-6.98</td></tr><tr><td>CoTS</td><td>3.39</td><td>6.67</td><td>2.48</td><td>1.85</td><td>3.51</td><td>5.21</td><td>2.91</td><td>6.13</td><td>3.11</td><td>7.58</td><td>4.29</td><td>4.28-7.01</td></tr><tr><td>E-CoTS</td><td>3.00</td><td>6.94</td><td>4.10</td><td>1.29</td><td>2.59</td><td>5.73</td><td>3.43</td><td>4.74</td><td>3.50</td><td>8.23</td><td>3.84</td><td>4.31-6.98</td></tr></table>

ImageNet-A (natural adversarial examples) [22], ImageNet-V (re-collected images) [54], ImageNet-R (artistic renditions) [20], and ImageNet-K (sketch-style images with domain shifts) [65]. In the episodic test-time adaptation setting [9], each data instance is adapted independently of the others.

Baselines. We first compare CoTS with representative calibration methods designed for fine-tuned CLIP [53]. These include regularization-based methods that encourage textual dispersion, such as C-TPT [72], O-TPT [55], A-TPT [1], and SoC [13]; logit-adjustment methods including ZS-Norm [46], Penalty [46], and SaLS [46]. In addition to direct comparison, we also evaluate whether our advanced E-CoTS can serve as a plug-and-play module on top of existing calibration methods [72, 55, 1, 13]. To examine generality beyond TPT methods, we further apply CoTS to other episodic test-time adaptation methods, including TTL [25] and TPS [60].

Implementation details. We use ViT-B/16 [53] as the backbone model and follow the standard TPT [57] protocol. The initial text prompt is set to "a photo of a [CLASS]" and optimized using AdamW [41] with a single gradient step and a learning rate of 0.005. For each test sample, we generate 64 augmented views using AugMix [21] and select confident views following TPT [57]. For temperature learning, we select the top 10% (ρ=0.1) most confident samples, excluding the weak augmentation (original image), and train the temperature parameter for 50 steps. We implement all baselines using their official code and report the average results over three random seeds. We provide additional results (e.g., for ResNet-50, standard deviations, and other metrics) in the Appendix.

## 4.2 Results

Performance on fine-grained datasets. We evaluate the performance on fine-grained datasets with ViT-B/16 in Table 1. CoTS preserves average accuracy of TPT while substantially improving calibration, reducing average ECE from 11.30% to 4.28%, which is the best among all methods. On Flowers, CoTS attains the lowest ECE overall, even outperforming zero-shot. Compared to SaLS [46], the only post-hoc calibration baseline, CoTS achieves much lower ECE. Furthermore, E-CoTS improves the average accuracy and maintains a competitive ECE, demonstrating a better balance between classification performance and calibration.

Performance on ImageNet variants. The results on four ImageNet variants under distribution shifts are demonstrated in Table 2. It is clear that regularization-based methods (O-TPT [55], A-TPT [1], SoC [13]) reduce ECE but at the cost of accuracy. In contrast, CoTS preserves TPT’s average accuracy while more than halving the average ECE, demonstrating its effectiveness without altering predictions. With the weak-strong ensemble strategy, E-CoTS achieves the best average accuracy of 62.95% while maintaining a competitive ECE of 5.38%. These results indicate that our approach not only mitigates miscalibration but also enhances the robustness of test-time prompt tuning.

Table 2: Performance comparison of different methods on ImageNet variants using ViT-B/16. Accuracy (%) and ECE (%) are reported.
<table><tr><td rowspan="2">Method</td><td colspan="5">Accuracy (%)</td><td colspan="5">ECE (%)</td></tr><tr><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td></tr><tr><td>Zero-Shot [53]</td><td>47.83</td><td>60.94</td><td>73.99</td><td>46.10</td><td>57.22</td><td>8.34</td><td>3.18</td><td>3.55</td><td>4.87</td><td>4.98</td></tr><tr><td>TPT [57]</td><td>54.63</td><td>63.45</td><td>77.05</td><td>47.82</td><td>60.74</td><td>15.20</td><td>11.88</td><td>4.84</td><td>15.69</td><td>11.90</td></tr><tr><td>C-TPT [72]</td><td>51.19</td><td>62.58</td><td>75.79</td><td>47.43</td><td> $5 9 . 2 5 _ { - 1 . 4 9 }$ </td><td>8.18</td><td>6.55</td><td>1.74</td><td>10.20</td><td> $6 . 6 7 _ { - 5 . 2 4 }$ </td></tr><tr><td>Penalty [46]</td><td>53.07</td><td>63.40</td><td>76.68</td><td>47.79</td><td> $6 0 . 2 4 _ { - 0 . 5 0 }$ </td><td>13.33</td><td>11.64</td><td>3.60</td><td>15.40</td><td> $1 0 . 9 9 _ { - 0 . 9 1 }$ </td></tr><tr><td>SALS [46]</td><td>54.63</td><td>63.44</td><td>77.05</td><td>47.82</td><td> $6 0 . 7 4 _ { + 0 . 0 0 }$ </td><td>13.95</td><td>10.81</td><td>3.87</td><td>15.29</td><td> $1 0 . 9 8 _ { - 0 . 9 2 }$ </td></tr><tr><td>O-TPT [55]</td><td>48.12</td><td>61.35</td><td>73.98</td><td>46.59</td><td> $5 7 . 5 1 _ { - 3 . 2 3 }$ </td><td>6.68</td><td>3.45</td><td>3.98</td><td>5.76</td><td> $4 . 9 7 _ { - 6 . 9 4 }$ </td></tr><tr><td>A-TPT [1]</td><td>49.16</td><td>61.77</td><td>74.99</td><td>46.89</td><td> $5 8 . 2 0 _ { - 2 . 5 4 }$ </td><td>5.55</td><td>2.85</td><td>4.09</td><td>4.87</td><td> $4 . 3 4 _ { - 7 . 5 6 }$ </td></tr><tr><td>SoC [13]</td><td>49.57</td><td>61.36</td><td>75.11</td><td>46.45</td><td> $5 8 . 1 2 _ { - 2 . 6 2 }$ </td><td>8.15</td><td>3.65</td><td>4.50</td><td>3.16</td><td> $4 . 8 7 _ { - 7 . 0 4 }$ </td></tr><tr><td>CoTS</td><td>54.63</td><td>63.45</td><td>77.05</td><td>47.82</td><td> $6 0 . 7 4 _ { + 0 . 0 0 }$ </td><td>7.85</td><td>4.37</td><td>5.09</td><td>3.87</td><td> $5 . 3 0 _ { - 6 . 6 1 }$ </td></tr><tr><td>E-CoTS</td><td>60.72</td><td>64.49</td><td>77.88</td><td>48.72</td><td> $6 2 . 9 5 _ { + 2 . 2 2 }$ </td><td>8.71</td><td>3.77</td><td>2.94</td><td>6.11</td><td> $5 . 3 8 _ { - 6 . 5 2 }$ </td></tr></table>

Table 3: Compatibility of CoTS, and E-CoTS with different base methods using ViT-B/16. Average accuracy (%) and ECE (%) are reported on fine-grained datasets and ImageNet variants, respectively.
<table><tr><td rowspan="2">(%)</td><td rowspan="2">Method</td><td colspan="5">Fine-grained datasets</td><td colspan="5">ImageNet variants</td></tr><tr><td>C-TPT [72]</td><td>Penalty [46]</td><td>O-TPT [55]</td><td>A-TPT [1]</td><td>SoC [13]</td><td>C-TPT [72]</td><td>Penalty [46]</td><td>O-TPT [55]</td><td>A-TPT [1]</td><td>SoC [13]</td></tr><tr><td>AC</td><td>base + E-CoTS</td><td>64.57 64.64</td><td>64.21 65.25</td><td>64.05 64.93</td><td>64.12 65.07</td><td>63.95 65.01</td><td>59.25 62.74</td><td>60.24 62.65</td><td>57.51 62.01</td><td>58.20 62.32</td><td>58.12 61.84</td></tr><tr><td rowspan="4">ECE</td><td>base</td><td>5.23</td><td>8.28</td><td>4.77</td><td>4.33</td><td>4.31</td><td>6.67</td><td>10.99</td><td>4.97</td><td>4.34</td><td>4.87</td></tr><tr><td>+ SaLS [46]</td><td>4.97</td><td>9.04</td><td>4.56</td><td>4.25</td><td>4.26</td><td>6.17</td><td>11.06</td><td>4.38</td><td>4.80</td><td>5.02</td></tr><tr><td>+ CoTS</td><td>4.53</td><td>3.90</td><td>4.55</td><td>4.38</td><td>4.44</td><td>4.75</td><td>5.14</td><td>4.14</td><td>4.43</td><td>4.37</td></tr><tr><td>+E-CoTS</td><td>4.16</td><td>3.86</td><td>4.30</td><td>3.85</td><td>4.20</td><td>5.14</td><td>5.39</td><td>4.55</td><td>4.87</td><td>4.58</td></tr></table>

Plug-and-play compatibility with calibration methods. Table 3 evaluates whether the proposed CoTS and E-CoTS can be integrated with different base methods. Overall, E-CoTS consistently improves accuracy across all base methods, showing that the weak-strong ensemble strategy is broadly compatible with prior approaches. For example, when combined with O-TPT [55], E-CoTS increases the accuracy on ImageNet variants from 57.51% to 62.01%, showing its effectiveness under distribution shifts. For calibration, CoTS and E-CoTS generally reduce ECE compared with the corresponding base methods, especially for Penalty [46] and C-TPT [72]. These results demonstrate that our method is not restricted to TPT, but can serve as a plug-and-play component to improve the accuracy-calibration trade-off of existing calibration methods.

## 4.3 Ablation Studies

We conduct ablation studies on both temperature scaling and the ensemble design. For temperature scaling in CoTS, we analyze the choice of augmented views and the choice of confidence alignment loss. For the ensemble strategy in E-CoTS, we study the effect of different ensemble weights. We also provide a component-level ablation in Appendix E.6, showing that ensemble-only improves accuracy but may harm calibration, whereas E-CoTS achieves a better accuracy-calibration trade-off.

View selection in CoTS. We analyze how different augmented view selections affect temperature learning, as illustrated in Fig. 3a. Using only the weak view yields limited calibration improvement, with ECE even higher than CoR, indicating that a single weak augmentation does not provide enough confidence variation for reliable temperature estimation. In contrast, using strong augmented views substantially reduces ECE, indicating that strong augmentations offer more informative confidence variations for alignment. However, simply using the Top-6 views is not optimal, since the selected set may still contain weak or overly confident views that provide a limited calibration signal. Among all settings, CoTS with the Top-6 strong views achieves the best calibration performance, obtaining the lowest ECE on both dataset groups.

![](images/b051a68d20b8d31f5a0c8349bba54f8cd6ec27b88026203ff2bf61a405a86f34.jpg)  
(a) View selection for CoTS.

![](images/8979633c11a22f8bcd7c6e5a22b999eb18abb4c5684b4409278a10f86a6324ec.jpg)  
(b) Loss function for CoTS.  
Figure 3: Ablation studies on key components $( i . e .$ , view selection and loss function) of CoTS with ViT-B/16. Average ECE (%) is reported on fine-grained datasets and ImageNet variants.

Table 4: Performance comparison of different methods initialized with CoOp [77] using ViT-B/16. We report average accuracy (%) and ECE (%) on fine-grained datasets and ImageNet variants, and evaluate the effect of integrating CoTS and E-CoTS.
<table><tr><td rowspan="2">(%)</td><td rowspan="2">Method</td><td colspan="5">Fine-grained datasets</td><td colspan="5">ImageNet variants</td></tr><tr><td>TPT [57]</td><td>C-TPT [72]</td><td>O-TPT [55]</td><td>A-TPT [1]</td><td>SoC [13]</td><td>TPT [57]</td><td>C-TPT [72]</td><td>O-TPT [55]</td><td>A-TPT [1]</td><td>SoC [13]</td></tr><tr><td>AC</td><td>base + E-CoTS</td><td>65.67 65.47</td><td>65.12 65.70</td><td>64.58 65.60</td><td>64.68 65.54</td><td>64.23 65.19</td><td>63.08 64.45</td><td>61.51 64.15</td><td>59.16 63.36</td><td>60.48 63.88</td><td>61.45 64.24</td></tr><tr><td rowspan="4">ECE</td><td>base</td><td>17.46</td><td>10.56</td><td>8.43</td><td>9.29</td><td>5.61</td><td>19.95</td><td>15.38</td><td>9.92</td><td>12.20</td><td>10.22</td></tr><tr><td>+ SALS [46]</td><td>13.83</td><td>6.76</td><td>5.38</td><td>6.05</td><td>5.44</td><td>16.00</td><td>9.94</td><td>5.46</td><td>7.26</td><td>8.13</td></tr><tr><td>+ CoTS</td><td>5.61</td><td>5.22</td><td>5.18</td><td>5.11</td><td>5.23</td><td>5.06</td><td>4.62</td><td>4.68</td><td>4.65</td><td>4.44</td></tr><tr><td>+E-CoTS</td><td>6.10</td><td>5.37</td><td>5.30</td><td>5.12</td><td>4.98</td><td>6.44</td><td>5.26</td><td>4.26</td><td>4.80</td><td>4.80</td></tr></table>

Loss function in CoTS. We further compare different loss functions for confidence alignment in Fig. 3b. The results show that all three losses achieve comparable calibration performance, indicating that CoTS is not highly sensitive to the specific loss formulation. Among them, the $\mathcal { L } _ { 2 }$ loss obtains the lowest ECE on both fine-grained datasets and ImageNet variants. Therefore, we adopt $\mathcal { L } _ { 2 }$ as the default objective for temperature learning in our experiments.

Ensemble weight α in E-CoTS. We study the influence of the ensemble weight α in E-CoTS, with results shown in Fig. 4. Using fixed weights underperforms the proposed adaptive weighting strategy, particularly on ImageNet variants (see Appendix E.2). The adaptive strategy achieves a favorable balance between accuracy and calibration, avoiding the need for manual weight selection.

## 4.4 Robustness Analysis

Compatibility with CoOp initialization. We evaluate our method initialized with CoOp [77], with the results shown in Table 4. Compared with the base results of existing regularization-based methods, CoTS consistently achieves much lower ECE across both fine-grained datasets and ImageNet variants. Meanwhile, E-CoTS generally improves accuracy over the corresponding base methods, showing that the weak-strong ensemble remains beneficial under CoOp [77] initialization. Overall, these results demonstrate that our method is compatible with stronger prompt initialization and provides stronger calibration performance than existing regularization-based approaches. We also analyze the sensitivity of our method to hand-crafted prompt initialization in Appendix E.1.

Generalization to other episodic TTA methods We further evaluate CoTS and E-CoTS on other episodic TTA methods. The results in Table 5 show that our method generalizes well beyond TPTbased approaches. In particular, CoTS consistently brings large reductions in ECE on both datasets, substantially outperforming the corresponding base models and also achieving stronger calibration than SaLS [46]. Furthermore, E-CoTS maintains comparable, and sometimes improved accuracy over the base methods. These results indicate that our approach is not limited to a specific adaptation framework, but can serve as an effective calibration strategy for other episodic TTA methods as well.

![](images/9b945ee623825b384cd6c3816b51f571a3fb732c6e0663c120baba7edc6d419a.jpg)  
Table 5: Evaluation of SaLS [46], CoTS, and E-CoTS on TTL [25] and TPS [60] using ViT-B/16. Average accuracy (%) and ECE (%) on fine-grained datasets and ImageNet variants are reported.

Figure 4: Effect of the ensemble weight α on E-CoTS using ViT-B/16. Average accuracy (%) and ECE (%) on fine-grained datasets are reported.
<table><tr><td rowspan="2">(%) Method</td><td rowspan="2"></td><td colspan="2">Fine-grained datasets</td><td colspan="2">ImageNet variants</td></tr><tr><td>TTL [25]</td><td>TPS [60]</td><td>TTL [25]</td><td>TPS [60]</td></tr><tr><td>AC</td><td>base</td><td>64.48</td><td>64.99</td><td>62.46</td><td>61.69</td></tr><tr><td rowspan="4">ECE</td><td>+ E-CoTS</td><td>64.23</td><td>65.11</td><td>62.39</td><td>61.71</td></tr><tr><td>base</td><td>28.60</td><td>16.79</td><td>34.30</td><td>27.16</td></tr><tr><td>+ SaLS [46]</td><td>22.15</td><td>13.69</td><td>28.94</td><td>23.98</td></tr><tr><td>+ CoTS +E-CoTS</td><td>9.82 9.42</td><td>4.87 5.02</td><td>9.42 8.66</td><td>6.67 6.65</td></tr></table>

Table 6: Performance comparison of different methods using ViT-B/16 and ResNet-50. Accuracy (%) and ECE (%) are reported on fine-grained datasets and ImageNet variants, respectively.
<table><tr><td rowspan="3">Method</td><td colspan="4">ViT-B/16</td><td colspan="4">ResNet-50</td></tr><tr><td colspan="2">Fine-grained datasets</td><td colspan="2">ImageNet variants</td><td colspan="2">Fine-grained datasets</td><td colspan="2">ImageNet variants</td></tr><tr><td>ACC (%)</td><td>ECE (%)</td><td>ACC (%)</td><td>ECE (%)</td><td>ACC (%)</td><td>ECE (%)</td><td>ACC (%)</td><td>ECE (%)</td></tr><tr><td>Zero-Shot [53]</td><td>63.95</td><td>4.33</td><td>57.22</td><td>4.98</td><td>56.04</td><td>5.38</td><td>40.70</td><td>7.09</td></tr><tr><td>TPT [57]</td><td>65.19</td><td>11.30</td><td>60.74</td><td>11.90</td><td>58.06</td><td>11.31</td><td>43.83</td><td>17.55</td></tr><tr><td>C-TPT [72]</td><td>64.57</td><td>5.23</td><td>59.25</td><td>6.67</td><td>57.69</td><td>6.44</td><td>42.75</td><td>13.14</td></tr><tr><td>Penalty [46]</td><td>64.21</td><td>8.28</td><td>60.24</td><td>10.99</td><td>57.52</td><td>8.41</td><td>43.66</td><td>17.02</td></tr><tr><td>SALS [46]</td><td>65.20</td><td>9.94</td><td>60.74</td><td>10.98</td><td>58.06</td><td>9.31</td><td>43.83</td><td>15.37</td></tr><tr><td>O-TPT [55]</td><td>64.05</td><td>4.77</td><td>57.51</td><td>4.97</td><td>57.36</td><td>5.67</td><td>40.82</td><td>6.46</td></tr><tr><td>A-TPT [1]</td><td>64.12</td><td>4.33</td><td>58.20</td><td>4.34</td><td>57.30</td><td>4.55</td><td>41.94</td><td>7.76</td></tr><tr><td>SoC [13]</td><td>63.95</td><td>4.31</td><td>58.12</td><td>4.87</td><td>56.19</td><td>5.50</td><td>41.61</td><td>7.52</td></tr><tr><td>CoTS</td><td>65.19</td><td>4.28</td><td>60.74</td><td>5.30</td><td>58.06</td><td>5.04</td><td>43.83</td><td>6.87</td></tr><tr><td>E-CoTS</td><td>65.22</td><td>4.31</td><td>62.95</td><td>5.38</td><td>57.72</td><td>5.04</td><td>45.39</td><td>7.24</td></tr></table>

Overall comparison across backbones. We summarize the average performance on fine-grained datasets and ImageNet variants using both ViT-B/16 and ResNet-50 in Table 6. The improvements are consistent across both backbones, demonstrating the robustness of CoTS and E-CoTS. Overall, both CoTS and E-CoTS provide a superior accuracy-calibration trade-off compared to prior methods.

## 5 Conclusion

This paper studies the calibration problem in test-time prompt tuning from the perspective of confidence correction. We observe that existing calibration-oriented methods often reduce ECE at the cost of classification accuracy. To address this trade-off, we propose CoTS, a post-hoc confidence-based temperature scaling method that uses zero-shot CLIP predictions as a calibration anchor. By only rescaling logits, CoTS improves calibration performance without changing the discriminative benefits of TPT. We further introduce a weak-strong ensemble strategy and integrate it into CoTS, resulting in E-CoTS, which improves accuracy while maintaining reliable calibration. Extensive experiments across multiple datasets and backbones validate that our methods effectively mitigate miscalibration without compromising primary accuracy.

## References

[1] S. A. Ahamed, U. S. K. P. M. Thanthrige, R. Rodrigo, and M. H. Khan. A-TPT: Angular diversity calibration properties for test-time prompt tuning of vision-language models. In Proc. ICLR, 2026.

[2] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra. Weight uncertainty in neural network. In Proc. ICML, pages 1613–1622, 2015.

[3] L. Bossard, M. Guillaumin, and L. Van Gool. Food-101–mining discriminative components with random forests. In Proc. ECCV, pages 446–461, 2014.

[4] G. W. Brier. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78:1–3, 1950.

[5] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi. Describing textures in the wild. In Proc. CVPR, pages 3606–3613, 2014.

[6] K. M. Dafnis and D. N. Metaxas. Test-time spectrum-aware latent steering for zero-shot generalization in vision-language models. In Proc. NeurIPS, pages 151169–151194, 2025.

[7] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proc. CVPR, pages 248–255, 2009.

[8] Z. Ding, X. Han, P. Liu, and M. Niethammer. Local temperature scaling for probability calibration. In Proc. ICCV, pages 6889–6899, 2021.

[9] H. Dong, L. Sheng, J. Liang, R. He, E. Chatzi, and O. Fink. Adapting vision-language models without labels: A comprehensive survey. arXiv preprint arXiv:2508.05547, 2025.

[10] M. Farina, G. Franchi, G. Iacca, M. Mancini, and E. Ricci. Frustratingly easy test-time adaptation of vision-language models. In Proc. NeurIPS, pages 129062–129093, 2024.

[11] L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In Proc. CVPR Workshops, pages 178–178, 2004.

[12] C.-M. Feng, K. Yu, Y. Liu, S. Khan, and W. Zuo. Diverse data augmentation with diffusions for effective test-time prompt tuning. In Proc. ICCV, pages 2704–2714, 2023.

[13] L. Fillioux, O. Chakraborty, I. B. Ayed, P.-H. Cournède, S. Christodoulidis, M. Vakalopoulou, and J. Dolz. Soc: Semantic orthogonal calibration for test-time prompt tuning. In Proc. CVPR, 2026.

[14] Y. Gal and Z. Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Proc. ICML, pages 1050–1059, 2016.

[15] I. Galil, M. Dabbah, and R. El-Yaniv. What can we learn from the selective prediction and uncertainty estimation performance of 523 imagenet classifiers? In Proc. ICLR, 2023.

[16] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In Proc. ICML, pages 1321–1330, 2017.

[17] J. Han and W. Hwang. D-tpt: Dimensional entropy maximization for calibrating test-time prompt tuning in vision-language models. arXiv preprint arXiv:2510.09473, 2025.

[18] Z. Han, C. Gao, J. Liu, J. Zhang, and S. Q. Zhang. Parameter-efficient fine-tuning for large models: A comprehensive survey. Transactions on Machine Learning Research, 2024.

[19] P. Helber, B. Bischke, A. Dengel, and D. Borth. Introducing eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. In IGARSS, pages 204–207, 2018.

[20] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proc. ICCV, pages 8340–8349, 2021.

[21] D. Hendrycks, N. Mu, E. D. Cubuk, B. Zoph, J. Gilmer, and B. Lakshminarayanan. Augmix: A simple data processing method to improve robustness and uncertainty. In Proc. ICLR, 2020.

[22] D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song. Natural adversarial examples. In Proc. CVPR, pages 15262–15271, 2021.

[23] D. Hu, J. Liang, X. Wang, and C.-S. Foo. Pseudo-calibration: Improving predictive uncertainty estimation in unsupervised domain adaptation. In Proc. ICML, pages 19304–19326, 2024.

[24] T. Huang, J. Chu, and F. Wei. Unsupervised prompt learning for vision-language models. arXiv preprint arXiv:2204.03649, 2022.

[25] R. Imam, H. Gani, M. Huzaifa, and K. Nandakumar. Test-time low rank adaptation via confidence maximization for zero-shot generalization of vision-language models. In Proc. WACV, pages 5449–5459, 2025.

[26] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Proc. ICML, pages 4904–4916, 2021.

[27] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim. Visual prompt tuning. In Proc. ECCV, pages 709–727, 2022.

[28] A. Karmanov, D. Guan, S. Lu, A. El Saddik, and E. Xing. Efficient test-time adaptation of vision-language models. In Proc. CVPR, pages 14162–14171, 2024.

[29] A. Khandelwal, L. Weihs, R. Mottaghi, and A. Kembhavi. Simple but effective: Clip embeddings for embodied ai. In Proc. CVPR, pages 14829–14838, 2022.

[30] M. U. Khattak, H. Rasheed, M. Maaz, S. Khan, and F. S. Khan. Maple: Multi-modal prompt learning. In Proc. CVPR, pages 19113–19122, 2023.

[31] T. Koleilat, H. Asgariandehkordi, H. Rivaz, and Y. Xiao. Biomedcoop: Learning to prompt for biomedical vision-language models. In Proc. CVPR, pages 14766–14776, 2025.

[32] J. Krause, M. Stark, J. Deng, and L. Fei-Fei. 3d object representations for fine-grained categorization. In Proc. ICCV Workshops, pages 554–561, 2013.

[33] M. Kull, M. Perello-Nieto, M. Kängsepp, T. S. Filho, H. Song, and P. Flach. Beyond temperature scaling: obtaining well-calibrated multiclass probabilities with dirichlet calibration. In Proc. NeurIPS, pages 12316–12326, 2019.

[34] B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Proc. NeurIPS, pages 6405–6416, 2017.

[35] B. Lester, R. Al-Rfou, and N. Constant. The power of scale for parameter-efficient prompt tuning. In Proc. EMNLP, pages 3045–3059, 2021.

[36] V. Lialin, V. Deshpande, X. Yao, and A. Rumshisky. Scaling down to scale up: A guide to parameterefficient fine-tuning. arXiv preprint arXiv:2303.15647, 2023.

[37] J. Liang, R. He, and T. Tan. A comprehensive survey on test-time adaptation under distribution shifts. Int. J. Comput. Vis., 133(1):31–64, 2025.

[38] J. Liang, L. Sheng, Z. Wang, R. He, and T. Tan. Realistic unsupervised clip fine-tuning with universal entropy optimization. In Proc. ICML, pages 29667–29681, 2024.

[39] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár. Focal loss for dense object detection. In Proc. ICCV, pages 2980–2988, 2017.

[40] B. Liu, I. Ben Ayed, A. Galdran, and J. Dolz. The devil is in the margin: Margin-based label smoothing for network calibration. In Proc. CVPR, pages 80–88, 2022.

[41] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In Proc. ICLR, 2019.

[42] X. Ma, J. Zhang, S. Guo, and W. Xu. Swapprompt: test-time prompt adaptation for vision-language models. In Proc. NeurIPS, pages 65252–65264, 2023.

[43] S. Maji, E. Rahtu, J. Kannala, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

[44] M. Minderer, J. Djolonga, R. Romijnders, F. A. Hubis, X. Zhai, N. Houlsby, D. Tran, and M. Lucic. Revisiting the calibration of modern neural networks. In Proc. NeurIPS, pages 15682–15694, 2021.

[45] R. Müller, S. Kornblith, and G. Hinton. When does label smoothing help? In Proc. NeurIPS, pages 4694–4703, 2019.

[46] B. Murugesan, J. Silva-Rodríguez, I. B. Ayed, and J. Dolz. Robust calibration of large vision-language adapters. In Proc. ECCV, pages 147–165, 2024.

[47] M. P. Naeini, G. Cooper, and M. Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proc. AAAI, pages 2901–2907, 2015.

[48] M.-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In Proc. ICVGIP, pages 722–729, 2008.

[49] J. Nixon, M. W. Dusenberry, L. Zhang, G. Jerfel, and D. Tran. Measuring calibration in deep learning. In Proc. CVPR Workshops, 2019.

[50] C. Pan, B. Yaman, T. Nesti, A. Mallik, A. G. Allievi, S. Velipasalar, and L. Ren. Vlp: Vision language planning for autonomous driving. In Proc. CVPR, pages 14760–14769, 2024.

[51] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. Jawahar. Cats and dogs. In Proc. CVPR, pages 3498–3505, 2012.

[52] J. Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, 10(3):61–74, 1999.

[53] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In Proc. ICML, pages 8748–8763, 2021.

[54] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do imagenet classifiers generalize to imagenet? In Proc. ICML, pages 5389–5400, 2019.

[55] A. Sharifdeen, M. A. Munir, S. Baliah, S. Khan, and M. H. Khan. O-tpt: Orthogonality constraints for calibrating test-time prompt tuning in vision-language models. In Proc. CVPR, pages 19942–19951, 2025.

[56] L. Sheng, J. Liang, Z. Wang, and R. He. R-tpt: Improving adversarial robustness of vision-language models through test-time prompt tuning. In Proc. CVPR, pages 29958–29967, 2025.

[57] M. Shu, W. Nie, D.-A. Huang, Z. Yu, T. Goldstein, A. Anandkumar, and C. Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models. In Proc. NeurIPS, pages 14274–14289, 2022.

[58] J. Silva-Rodríguez, F. Shakeri, H. Bahig, J. Dolz, and I. Ben Ayed. Few-shot, now for real: Medical vlms adaptation without balanced sets or validation. In Proc. MICCAI, pages 237–247, 2025.

[59] K. Soomro, A. R. Zamir, and M. Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

[60] E. Sui, X. Wang, and S. Yeung-Levy. Just shift it: Test-time prototype shifting for zero-shot generalization with vision-language models. In Proc. WACV, pages 825–835, 2025.

[61] Y. Sun, X. Wang, Z. Liu, J. Miller, A. Efros, and M. Hardt. Test-time training with self-supervision for generalization under distribution shifts. In Proc. ICML, pages 9229–9248, 2020.

[62] K. Tanwisuth, S. Zhang, H. Zheng, P. He, and M. Zhou. Pouf: Prompt-oriented unsupervised fine-tuning for large pre-trained models. In Proc. ICML, pages 33816–33832, 2023.

[63] S. Thulasidasan, G. Chennupati, J. Bilmes, T. Bhattacharya, and S. Michalak. On mixup training: improved calibration and predictive uncertainty for deep neural networks. In Proc. NeurIPS, pages 13911–13922, 2019.

[64] C. Wang. Calibration in deep learning: A survey of the state-of-the-art. arXiv preprint arXiv:2308.01222, 2023.

[65] H. Wang, S. Ge, E. P. Xing, and Z. C. Lipton. Learning robust global representations by penalizing local predictive power. In Proc. NeurIPS, pages 10506–10518, 2019.

[66] S. Wang, Y. Li, and H. Wei. Understanding and mitigating miscalibration in prompt tuning for visionlanguage models. In Proc. ICML, pages 63467–63489, 2025.

[67] S. Wang, J. Wang, G. Wang, B. Zhang, K. Zhou, and H. Wei. Open-vocabulary calibration for fine-tuned clip. In Proc. ICML, pages 51734–51754, 2024.

[68] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In Proc. CVPR, pages 3485–3492, 2010.

[69] Z. Xiao, S. Yan, J. Hong, J. Cai, X. Jiang, Y. Hu, J. Shen, C. Wang, and C. G. M. Snoek. Dynaprompt: Dynamic test-time prompt tuning. In Proc. ICLR, 2025.

[70] D. Yan, J. Liang, Y. Wang, S. Lu, R. He, and T. Tan. What if consensus lies? selective-complementary reinforcement learning at test time. In Proc. ACL, pages 28957–28970, 2026.

[71] H. Yao, R. Zhang, and C. Xu. Visual-language prompt tuning with knowledge-guided context optimization. In Proc. CVPR, pages 6757–6767, 2023.

[72] H. S. Yoon, E. Yoon, J. T. J. Tee, M. A. Hasegawa-Johnson, Y. Li, and C. D. Yoo. C-TPT: Calibrated test-time prompt tuning for vision-language models via text feature dispersion. In Proc. ICLR, 2024.

[73] M. Zanella and I. Ben Ayed. On the test-time zero-shot generalization of vision-language models: Do we really need prompt learning? In Proc. CVPR, pages 23783–23793, 2024.

[74] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. In Proc. ICCV, pages 11941–11952, 2023.

[75] M. Zhang, S. Levine, and C. Finn. Memo: test time robustness via adaptation and augmentation. In Proc. NeurIPS, pages 38629–38642, 2022.

[76] K. Zhou, J. Yang, C. C. Loy, and Z. Liu. Conditional prompt learning for vision-language models. In Proc. CVPR, pages 16816–16825, 2022.

[77] K. Zhou, J. Yang, C. C. Loy, and Z. Liu. Learning to prompt for vision-language models. Int. J. Comput. Vis., 130(9):2337–2348, 2022.

## A Limitations and Broader Impact

## A.1 Limitations

While our methods effectively improve calibration for test-time prompt tuning, they are currently evaluated on classification tasks and introduce additional computational overhead. Extending confidence oriented temperature scaling to other vision-language tasks, such as semantic segmentation, object detection, or adversarial robustness, remains to be explored.

## A.2 Broader Impact

This work aims to improve the reliability of vision-language models by mitigating the miscalibration issue in test-time prompt tuning. The proposed methods achieve a favorable balance between accuracy and calibration error. Moreover, our approaches can be readily integrated with existing calibration techniques, highlighting their potential as general plug-and-play modules for test-time adaptation. We hope this work provides useful insights for future research on calibration in vision-language models and encourages the development of more reliable foundation models.

## B Related Work

Prompt Tuning in Vision-Language Models. Vision-Language Models (VLMs) such as CLIP [53] and ALIGN [26] learn a shared embedding space that aligns visual features with corresponding textual descriptions, enabling zero-shot transfer to a variety of downstream classification tasks. To efficiently adapt VLMs to diverse downstream tasks, many parameter-efficient fine-tuning (PEFT) methods [36, 18] have proven effective by optimizing only a small subset of parameters while keeping the rest of the model frozen. One widely used PEFT technique is prompt tuning [35, 27], which adds trainable tokens either to the input or to intermediate layers. In particular, CoOp [77] pioneers prompt learning for VLMs by replacing fixed templates with learnable continuous vectors. Subsequently, CoCoOp [76] incorporates visual cues to generate instance-specific prompts, while MaPLE [30] further enhances context tuning by applying learnable prompts to both the image and text encoders. To avoid relying on annotated downstream training data during adaptation, several methods [24, 62, 38] explore unsupervised prompt learning with pre-trained VLMs. TPT [57] focuses on an interesting unsupervised scenario, where the model is adapted using only a single instance at test time. Such a challenging paradigm has gained increasing attention [25, 10, 73, 60, 56], and our work also studies this problem but focuses more on calibration performance.

Calibration of Deep Neural Networks. In real-world safety-critical decision systems, classification networks must not only be accurate but also indicate when they are likely to be incorrect. Confidence calibration [16, 49] formalizes this by measuring how well a model’s predicted confidence aligns with its actual accuracy. A recent survey [64] categorizes existing calibration methods into post-hoc, regularization, and uncertainty estimation approaches. Post-hoc calibration methods [16, 33, 8] adjust predictions post-training without modifying parameters. A popular example is temperature scaling (TS) [16], which scales logits using a temperature parameter optimized on a validation set.Later methods [33, 8] extend TS to improve robustness under limited validation data, multi-label tasks, and distribution shifts. By contrast, regularization-based calibration methods [16, 40] always incorporate additional regularization objectives during model training. These include explicit penalties on overconfident predictions [16, 40] and implicit techniques (e.g., label smoothing [45], mixup [63], focal loss [39]) that improve calibration and generalization via softened targets. Moreover, uncertainty estimation methods alleviate miscalibration by introducing randomness via Bayesian neural networks [2], ensembles [34], and Monte Carlo dropout [14]. Most existing post-hoc calibration methods require a labeled validation set. One recent approach [23] uses mixup synthesis over unlabeled samples to generate pseudo-labeled data for temperature scaling, whereas our method operates using only a single unlabeled instance. We further provide a hybrid calibration strategy that combines temperature scaling with an output-level ensemble.

Calibration of Vision-Language Models. Although prompt tuning methods are primarily developed to improve accuracy, recent works [67, 66, 46, 72] have also examined the calibration performance of VLMs, particularly CLIP [53]. Specifically, existing few-shot prompt tuning methods are found to lead to a calibration trade-off between base and new classes [67], e.g., CoOp [77] causes overconfidence on new classes, whereas KgCoOp [71] leads to underconfidence on base classes. Meanwhile, a classic unsupervised prompt tuning method, TPT [72], improves accuracy but incurs a high calibration error after test-time adaptation. Since our method belongs to the unsupervised category, we primarily review prior work within this branch of the literature. C-TPT [72] finds that well-calibrated prompts yield text embeddings with broader class-wise dispersion and, on this basis, proposes a dispersionbased regularization term to encourage embeddings to move away from the class centroid. To fully exploit the textual feature space, several recent methods (i.e., O-TPT [55], A-TPT [1], and SoC [13]) encourage angular separation between textual features to promote greater dispersion. Alternatively, D-TPT [17] identifies and mitigates the influence of dominant feature dimensions to improve calibration performance. Unlike these regularization-based methods, we follow TS [16] and propose a simple yet efficient post-hoc calibration method. The most closely related work to ours is SaLS [46], which adjusts logits within the zero-shot range to maintain reliable confidence. Both methods are built on the observation that zero-shot predictions are relatively well-calibrated. However, unlike SaLS [46], our method learns a temperature parameter to bridge the confidence gap before and after adaptation.

## C Pseudocode

Algorithm 1: PyTorch-style code for CoTS and E-CoTS   
# x: input image (C, H, W); f<sub>clip</sub>: pretrained CLIP; A: augmentation   
# N: views nums; ρ: selection ratio; T: steps; ensemble: False/ True   
def E-CoTS (x, f<sub>clip</sub>, A, N, ρ, T):   
# step 1: augment and select confident views   
views = A (x, num\_views=N) // (N, C, H, W)   
logits = f<sub>clip</sub>(views) // (N, K)   
idx<sub>zs</sub> = select\_confident\_views(logits , top=ρ)   
# step 2: test-time prompt tuning   
f<sub>tpt</sub> = run\_tpt\_optimization(f<sub>clip</sub>, views[idx<sub>zs</sub>], steps=T)   
logits = f<sub>tpt</sub>(views)   
# step 3: confidence-based temperature scaling (CoTS, Eq.(5))   
idx<sub>tpt</sub> = select\_confident\_views(logits [1:], top=ρ)   
τ = optimize\_temperature(logits [idx<sub>tpt</sub>], logits<sub>zs</sub>[idx<sub>tpt</sub>])   
pˆ<sup>tpt</sup> = (logits / τ ).softmax(dim=1)   
if ensemble: # step 4: weak-strong ensemble (E-CoTS, Eq.(7))   
α = compute\_average\_similarity()   
pˆ<sub>ecots</sub> = α · pˆ<sup>tpt</sup>[0] + (1 − α) · pˆ<sup>tpt</sup>[idx<sub>tpt</sub>].mean(dim=0)   
return pˆ<sub>ecots</sub>.argmax(), pˆ<sub>ecots</sub>.max()   
else:   
pˆ = pˆ<sup>tpt</sup>[0]   
return pˆ<sub>cots</sub>.argmax(), pˆ<sub>cots</sub>.max()

We provided the pseudocode in Algorithm 1.

## D Additional Experimental Results

## D.1 Computational Efficiency

We measure the computation time of different methods using a single 48GB GPU (NVIDIA RTX A6000). As shown in Table 7, integrating CoTS delivers effective calibration improvement without substantially increasing overall inference cost. These results show that CoTS offers an efficient calibration strategy for test-time prompt tuning.

## D.2 Results with ResNet-50

We further evaluate different methods with ResNet-50 to verify the robustness of our approach across architectures. The results in fine-grained datasets and ImageNet variants are shown in Table 8 and Table 9.

## D.3 Results with Online TTA Framework

To assess the applicability of our approach beyond the standard episodic setting, we integrate CoTS and E-CoTS with TDA [28], an online TTA framework. As shown in Table 10 and Table 11, CoTS substantially improves calibration, while E-CoTS achieves a favorable accuracy–calibration trade-off.

## D.4 Results with SigLIP

We also investigate whether the proposed methods remain effective with a different vision-language model by conducting experiments with SigLIP [74]. Table 12 and Table 13 show that CoTS and E-CoTS consistently provide competitive accuracy and improved calibration.

## D.5 Standard Deviations

We report standard deviations of accuracy and ECE across three seeds in Table 14 and Table 15. On fine-grained datasets and ImageNet, CoTS achieves consistently low variance, indicating stable performance. On ImageNet variants, both CoTS and E-CoTS remain robust across seeds, with E-CoTS maintaining strong calibration despite minor variability. Overall, the results confirm that the proposed methods are reliable and stable while improving calibration under different runs.

## D.6 Calibration Metrics

Calibration error. To quantify model calibration [47, 16], we primarily adopt the Expected Calibration Error (ECE) [47], which measures the discrepancy between predicted probabilities and observed frequencies. Specifically, the predictions are partitioned into M confidence bins $\{ B _ { 1 } , B _ { 2 } , \dots , B _ { M } \}$ based on the confidence values, where $B _ { m }$ contains all samples whose predicted confidence falls into the m-th interval. The ECE is then computed as a weighted average of the absolute difference between accuracy and confidence within each bin:

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { S } \left| \operatorname { a c c } ( B _ { m } ) - \operatorname { c o n f } ( B _ { m } ) \right| ,\tag{8}
$$

Table 7: Computation time (s) of different methods on DTD and ImageNet-A using ViT-B/16. We report the average inference time per sample.
<table><tr><td>Dataset</td><td>TPT [57]</td><td>C-TPT [72]</td><td>O-TPT [55]</td><td>A-TPT [1]</td><td>SoC [13]</td><td>SaLS [46]</td><td>CoTS</td></tr><tr><td>DTD</td><td>0.192</td><td>0.193</td><td>0.206</td><td>0.207</td><td>0.206</td><td>+0.000</td><td>+0.053</td></tr><tr><td>ImageNet-A</td><td>0.223</td><td>0.224</td><td>0.258</td><td>0.258</td><td>0.256</td><td>+0.000</td><td>+0.054</td></tr></table>

Table 8: Performance comparison on fine-grained datasets using ResNet-50. Accuracy (%) and ECE (%) are reported.
<table><tr><td>(%)</td><td>Method</td><td>ImgNet</td><td>DTD</td><td>Flowers</td><td>Food101</td><td>SUN397</td><td>Aircraft</td><td>Pets</td><td>Caltech</td><td>UCF101</td><td>EuroSAT</td><td>Cars</td><td> $\operatorname { A v g } .$ </td></tr><tr><td rowspan="10">ACuacy</td><td>Zero-Shot [53] TPT [57]</td><td>58.17 60.68</td><td>40.37 41.51</td><td>61.67 62.57</td><td>73.94 74.99</td><td>58.84 61.36</td><td>15.75 17.69</td><td>83.54 84.43</td><td>85.88 87.87</td><td>58.84 60.67</td><td>23.67 28.42</td><td>55.78 58.47</td><td>56.04</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>58.06</td></tr><tr><td>C-TPT [72]</td><td>60.47</td><td>41.25</td><td>65.03</td><td>74.84</td><td>60.93</td><td>17.21</td><td>83.66</td><td>87.37</td><td>60.25</td><td>27.18</td><td>56.38</td><td>57.69</td></tr><tr><td>Penalty [46]</td><td>60.64</td><td>41.31</td><td>62.66</td><td>74.70</td><td>61.25</td><td>17.36</td><td>83.94</td><td>87.56</td><td>60.08</td><td>25.67</td><td>57.56</td><td>57.52</td></tr><tr><td>SaLS [46]</td><td>60.69</td><td>41.51</td><td>62.57</td><td>74.99</td><td>61.36</td><td>17.68</td><td>84.43</td><td>87.86</td><td>60.67</td><td>28.42</td><td>58.47</td><td>58.06</td></tr><tr><td>O-TPT [55]</td><td>58.99</td><td>41.65</td><td>65.58</td><td>74.64</td><td>59.61</td><td>17.03</td><td>83.19</td><td>87.07</td><td>59.81</td><td>27.73</td><td>55.64</td><td>57.36</td></tr><tr><td>A-TPT [1]</td><td>59.91</td><td>41.86</td><td>64.20</td><td>74.29</td><td>59.31</td><td>16.00</td><td>82.68</td><td>86.88</td><td>59.81</td><td>29.14</td><td>56.17</td><td>57.30</td></tr><tr><td>SoC [13]</td><td>59.93</td><td>40.37</td><td>61.90</td><td>73.82</td><td>59.04</td><td>15.67</td><td>83.28</td><td>85.84</td><td>58.90</td><td>23.64</td><td>55.68</td><td>56.19</td></tr><tr><td>CoTS</td><td>60.68</td><td>41.51</td><td>62.57</td><td>74.99</td><td>61.36</td><td>17.69</td><td>84.43</td><td>87.87</td><td>60.67</td><td>28.42</td><td>58.47</td><td>58.06</td></tr><tr><td>E-CoTS</td><td>60.90</td><td>41.33</td><td>60.71</td><td>74.27</td><td>61.33</td><td>17.73</td><td>84.12</td><td>87.69</td><td>60.47</td><td>27.45</td><td>58.96</td><td>57.72</td></tr><tr><td rowspan="10">EE</td><td>Zero-Shot [53]</td><td>2.01</td><td>9.04</td><td>3.03</td><td>2.71</td><td>3.81</td><td>6.30</td><td>5.70</td><td>4.41</td><td>3.00</td><td>14.76</td><td>4.40</td><td>5.38</td></tr><tr><td>TPT [57]</td><td>11.37</td><td>25.80</td><td>13.63</td><td>5.21</td><td>9.17</td><td>15.66</td><td>3.94</td><td>3.74</td><td>11.03</td><td>21.10</td><td>3.76</td><td>11.31</td></tr><tr><td>C-TPT [72]</td><td>6.75</td><td>21.65</td><td>3.81</td><td>1.76</td><td>3.01</td><td>10.69</td><td>2.79</td><td>2.61</td><td>3.02</td><td>13.37</td><td>1.33</td><td>6.44</td></tr><tr><td>Penalty [46]</td><td>11.29</td><td>16.96</td><td>11.94</td><td>3.34</td><td>8.98</td><td>13.39</td><td>3.25</td><td>3.66</td><td>10.08</td><td>6.92</td><td>2.65</td><td>8.41</td></tr><tr><td>SaLS [46]</td><td>9.91</td><td>21.66</td><td>11.99</td><td>3.93</td><td>8.66</td><td>15.29</td><td>3.07</td><td>4.02</td><td>8.98</td><td>12.32</td><td>2.54</td><td>9.31</td></tr><tr><td>O-TPT [55]</td><td>3.11</td><td>16.63</td><td>2.40</td><td>1.28</td><td>6.57</td><td>8.18</td><td>3.35</td><td>2.85</td><td>2.35</td><td>13.44</td><td>2.26</td><td>5.67</td></tr><tr><td>A-TPT [1]</td><td>2.32</td><td>15.35</td><td>2.71</td><td>1.55</td><td>4.22</td><td>8.26</td><td>2.72</td><td>4.16</td><td>2.63</td><td>4.71</td><td>1.43</td><td>4.55</td></tr><tr><td>SoC [13]</td><td>3.02</td><td>8.95</td><td>3.34</td><td>3.63</td><td>3.89</td><td>6.36</td><td>5.61</td><td>4.53</td><td>2.37</td><td>14.79</td><td>4.02</td><td>5.50</td></tr><tr><td>CoTS</td><td>3.74</td><td>7.97</td><td>2.62</td><td>3.74</td><td>5.24</td><td>4.88</td><td>5.29</td><td>5.84</td><td>2.79</td><td>7.33</td><td>6.05</td><td>5.04</td></tr><tr><td>E-CoTS</td><td>2.36</td><td>9.47</td><td>6.80</td><td>2.18</td><td>3.36</td><td>5.40</td><td>3.89</td><td>4.40</td><td>3.67</td><td>8.54</td><td>5.42</td><td>5.04</td></tr></table>

Table 9: Performance comparison of different methods on ImageNet variants using ResNet-50. Accuracy (%) and ECE (%) are reported.
<table><tr><td rowspan="2">Method</td><td colspan="5">Accuracy (%)</td><td colspan="5">ECE (%)</td></tr><tr><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Avg.</td><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Avg.</td></tr><tr><td>Zero-Shot [53]</td><td>21.87</td><td>51.45</td><td>56.12</td><td>33.36</td><td>40.70</td><td>21.24</td><td>3.03</td><td>0.98</td><td>3.11</td><td>7.09</td></tr><tr><td>TPT [57]</td><td>26.51</td><td>54.63</td><td>59.04</td><td>35.13</td><td>43.83</td><td>30.96</td><td>13.94</td><td>10.47</td><td>14.84</td><td>17.55</td></tr><tr><td>C-TPT [72]</td><td>24.27</td><td>54.21</td><td>57.77</td><td>34.75</td><td>42.75</td><td>26.61</td><td>9.35</td><td>5.51</td><td>11.08</td><td>13.14</td></tr><tr><td>Penalty [46]</td><td>26.12</td><td>54.58</td><td>58.80</td><td>35.12</td><td>43.66</td><td>29.85</td><td>13.80</td><td>9.73</td><td>14.70</td><td>17.02</td></tr><tr><td>SaLS [46]</td><td>26.52</td><td>54.63</td><td>59.04</td><td>35.13</td><td>43.83</td><td>28.10</td><td>12.27</td><td>7.79</td><td>13.32</td><td>15.37</td></tr><tr><td>O-TPT [55]</td><td>21.59</td><td>52.37</td><td>56.11</td><td>33.20</td><td>40.82</td><td>19.23</td><td>2.38</td><td>1.52</td><td>2.70</td><td>6.46</td></tr><tr><td>A-TPT [1]</td><td>22.56</td><td>53.26</td><td>57.61</td><td>34.32</td><td>41.94</td><td>20.28</td><td>4.11</td><td>0.94</td><td>5.70</td><td>7.76</td></tr><tr><td>SoC [13]</td><td>22.60</td><td>53.28</td><td>56.85</td><td>33.72</td><td>41.61</td><td>19.38</td><td>4.24</td><td>2.47</td><td>3.97</td><td>7.52</td></tr><tr><td>CoTS</td><td>26.51</td><td>54.63</td><td>59.04</td><td>35.13</td><td>43.83</td><td>17.52</td><td>4.55</td><td>3.07</td><td>2.33</td><td>6.87</td></tr><tr><td>E-CoTS</td><td>31.80</td><td>55.20</td><td>58.90</td><td>35.65</td><td>45.39</td><td>17.75</td><td>4.60</td><td>1.72</td><td>4.89</td><td>7.24</td></tr></table>

where $S$ is the total number of samples, $| B _ { m } |$ denotes the number of samples in bin $B _ { m }$ , and acc $( B _ { m } )$ and conf( ${ ( B _ { m } ) }$ represent the fraction of correctly predicted samples and the mean predicted confidence for the m-th bin, respectively. ECE provides a scalar measure of miscalibration: smaller values indicate better alignment between predicted confidences and empirical accuracies.

Brier score. We also report the Brier Score (BS) [4] to evaluate the quality of probabilistic predictions. Different from ECE, which measures the confidence-accuracy discrepancy after binning, the BS directly computes the squared error between the predicted probability distribution and the one-hot ground-truth label. For a multi-class classification problem with K classes, it is formulated as:

$$
\mathbf { B S } = \frac { 1 } { S } \sum _ { i = 1 } ^ { S } \sum _ { c = 1 } ^ { K } \left( p _ { i , c } - y _ { i , c } \right) ^ { 2 } ,\tag{9}
$$

where $S$ denotes the total number of samples, p represents the predicted probability of sample i $p _ { i , c }$ for class $c ,$ and $y _ { i , c }$ is the corresponding one-hot ground-truth label. A lower BS indicates that the predicted probability distribution is closer to the true label distribution. The BS values of different methods with ViT-B/16 are shown in Table 16 and Table 17. Our methods achieve competitive Brier scores across datasets.

Table 10: Performance comparison of different methods integrated with an online TTA framework (TDA [28]) on fine-grained datasets using ViT-B/16. Accuracy (%) and ECE (%) are reported.
<table><tr><td>(%)</td><td>Method</td><td>ImgNet</td><td>DTD</td><td>Flowers</td><td>Food101</td><td>SUN397</td><td>Aircraft</td><td>Pets</td><td>Caltech</td><td>UCF101</td><td>EuroSAT</td><td>Cars</td><td>Average</td></tr><tr><td rowspan="10">ACacy</td><td>TDA [28] C-TPT [72]</td><td>68.27</td><td>46.04</td><td>69.75</td><td>83.68</td><td>65.01</td><td>24.06</td><td>88.36</td><td>94.04</td><td>67.99</td><td>51.04</td><td>66.55</td><td>65.65</td></tr><tr><td></td><td>69.15</td><td>47.22</td><td>70.69</td><td>83.42</td><td>65.52</td><td>24.30</td><td>88.69</td><td>94.08</td><td>67.64</td><td>47.67</td><td>67.01</td><td>65.62</td></tr><tr><td>Penalty [46]</td><td>68.26</td><td>46.04</td><td>69.79</td><td>83.67</td><td>65.09</td><td>24.12</td><td>88.36</td><td>94.00</td><td>67.99</td><td>51.04</td><td>66.57</td><td>65.90</td></tr><tr><td>SaLS [46]</td><td>68.27</td><td>46.04</td><td>69.75</td><td>83.68</td><td>65.01</td><td>24.06</td><td>88.36</td><td>94.04</td><td>67.99</td><td>51.04</td><td>66.55</td><td>65.65</td></tr><tr><td>O-TPT [55]</td><td>68.49</td><td>48.11</td><td>70.77</td><td>83.15</td><td>64.93</td><td>24.09</td><td>88.58</td><td>94.20</td><td>66.98</td><td>47.06</td><td>66.56</td><td>65.44</td></tr><tr><td>A-TPT [1]</td><td>68.83</td><td>48.11</td><td>71.09</td><td>83.22</td><td>65.08</td><td>23.97</td><td>88.55</td><td>93.43</td><td>67.01</td><td>51.04</td><td>66.53</td><td>65.80</td></tr><tr><td>SoC [13]</td><td>68.59</td><td>45.21</td><td>69.71</td><td>83.66</td><td>64.82</td><td>24.24</td><td>88.61</td><td>94.20</td><td>68.17</td><td>51.04</td><td>66.55</td><td>65.62</td></tr><tr><td>CoTS</td><td>68.27</td><td>46.04</td><td>69.75</td><td>83.68</td><td>65.01</td><td>24.06</td><td>88.36</td><td>94.04</td><td>67.99</td><td>51.04</td><td>66.55</td><td>65.65</td></tr><tr><td>E-CoTS</td><td>70.25</td><td>47.16</td><td>69.35</td><td>84.58</td><td>66.96</td><td>25.02</td><td>88.80</td><td>94.24</td><td>69.13</td><td>51.52</td><td>68.85</td><td>66.56</td></tr><tr><td>TDA [28] C-TPT [72]</td><td>5.27</td><td>17.02</td><td>7.76</td><td>2.21</td><td>5.78</td><td>13.98</td><td>2.81</td><td>3.32</td><td>5.88</td><td>7.05</td><td>1.95</td><td>6.78</td></tr><tr><td rowspan="9">ECE</td><td></td><td>8.80</td><td>20.09</td><td>11.35</td><td>1.68</td><td>7.11</td><td>12.54</td><td>2.04</td><td>3.14</td><td>5.41</td><td>12.32</td><td>4.17</td><td>7.98</td></tr><tr><td>Penalty [46]</td><td>5.29</td><td>16.99</td><td>7.71</td><td>2.23</td><td>5.75</td><td>13.94</td><td>2.86</td><td>3.28</td><td>5.92</td><td>7.08</td><td>2.04</td><td>6.64</td></tr><tr><td>SaLS [46]</td><td>3.98</td><td>13.36</td><td>5.59</td><td>1.48</td><td>4.33</td><td>12.10</td><td>3.38</td><td>3.72</td><td>4.78</td><td>6.75</td><td>1.90</td><td>5.74</td></tr><tr><td>O-TPT [55]</td><td>5.80</td><td>15.10</td><td>9.71</td><td>1.92</td><td>3.85</td><td>11.70</td><td>1.79</td><td>3.20</td><td>4.60</td><td>13.42</td><td>3.63</td><td>6.89</td></tr><tr><td>A-TPT [1]</td><td>5.43</td><td>14.88</td><td>10.21</td><td>1.47</td><td>7.58</td><td>14.46</td><td>1.26</td><td>3.16</td><td>4.63</td><td>7.05</td><td>3.97</td><td>6.87</td></tr><tr><td>SoC [13]</td><td>5.35</td><td>16.31</td><td>8.95</td><td>1.48</td><td>5.08</td><td>13.85</td><td>1.43</td><td>3.64</td><td>6.08</td><td>7.02</td><td>1.85</td><td>6.57</td></tr><tr><td>CoTS</td><td>3.47</td><td>7.13</td><td>3.09</td><td>2.54</td><td>4.97</td><td>4.72</td><td>5.02</td><td>6.40</td><td>3.37</td><td>6.66</td><td>5.61</td><td>4.95</td></tr><tr><td>E-CoTS</td><td>3.05</td><td>6.26</td><td>2.77</td><td>3.02</td><td>5.20</td><td>3.80</td><td>3.27</td><td>5.28</td><td>3.13</td><td>6.78</td><td>7.02</td><td>4.65</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Performance comparison of different methods integrated with an online TTA framework (TDA [28]) on ImageNet variants using ViT-B/16. Accuracy (%) and ECE (%) are reported.
<table><tr><td rowspan="2">Method</td><td colspan="5">Accuracy (%)</td><td colspan="5">ECE (%)</td></tr><tr><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td><td> $\mathbf { \nabla } _ { - \mathbf { A } }$ </td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td></tr><tr><td>TDA [28]</td><td>50.03</td><td>61.87</td><td>75.17</td><td>48.84</td><td>58.98</td><td>11.96</td><td>10.06</td><td>1.62</td><td>10.10</td><td>8.44</td></tr><tr><td>C-TPT [72]</td><td>52.92</td><td>63.32</td><td>76.60</td><td>49.21</td><td>60.51</td><td>12.03</td><td>13.06</td><td>1.93</td><td>14.59</td><td>10.40</td></tr><tr><td>Penalty [46]</td><td>50.03</td><td>61.89</td><td>75.17</td><td>48.83</td><td>58.98</td><td>11.98</td><td>10.07</td><td>1.62</td><td>10.14</td><td>8.45</td></tr><tr><td>SaLS [46]</td><td>50.03</td><td>61.87</td><td>75.17</td><td>48.84</td><td>58.98</td><td>10.22</td><td>8.11</td><td>2.23</td><td>7.95</td><td>7.13</td></tr><tr><td>O-TPT [55]</td><td>50.65</td><td>62.27</td><td>75.22</td><td>48.98</td><td>59.28</td><td>10.57</td><td>10.48</td><td>1.94</td><td>10.92</td><td>8.48</td></tr><tr><td>A-TPT [1]</td><td>51.44</td><td>62.57</td><td>75.95</td><td>49.14</td><td>59.77</td><td>9.45</td><td>10.01</td><td>1.84</td><td>10.43</td><td>7.93</td></tr><tr><td>SoC [13]</td><td>51.45</td><td>62.29</td><td>75.96</td><td>48.73</td><td>59.61</td><td>12.03</td><td>9.29</td><td>2.58</td><td>9.07</td><td>8.24</td></tr><tr><td>CoTS</td><td>50.03</td><td>61.87</td><td>75.17</td><td>48.84</td><td>58.98</td><td>6.37</td><td>3.40</td><td>5.43</td><td>2.22</td><td>4.36</td></tr><tr><td>E-CoTS</td><td>59.63</td><td>64.28</td><td>78.36</td><td>51.16</td><td>63.36</td><td>8.18</td><td>3.60</td><td>6.24</td><td>1.89</td><td>4.98</td></tr></table>

Class-wise ECE. We additionally report the Class-wise Expected Calibration Error (CECE) [33], which evaluates calibration across all classes rather than only the predicted class. It is computed as:

$$
\mathrm { C E C E } = \sum _ { m = 1 } ^ { M } \sum _ { c = 1 } ^ { K } \frac { | B _ { m , c } | } { S K } \left| \operatorname { a c c } _ { c } ( B _ { m , c } ) - \operatorname { c o n f } _ { c } ( B _ { m , c } ) \right| ,\tag{10}
$$

where $B _ { m , c }$ denotes the m-th confidence bin for class c. Lower CECE indicates better class-wise calibration. The results are reported in Table 18 and Table 19.

Adaptive ECE. We also report Adaptive Expected Calibration Error (AECE) [49], which uses adaptive bins with approximately equal numbers of samples to reduce the sensitivity to fixed-width binning. It is defined as:

$$
\mathrm { A E C E } = \sum _ { r = 1 } ^ { R } \sum _ { c = 1 } ^ { K } \frac { 1 } { R K } \left| \operatorname { a c c } _ { c } ( B _ { r , c } ) - \operatorname { c o n f } _ { c } ( B _ { r , c } ) \right| .\tag{11}
$$

Lower AECE indicates better calibration. The results are reported in Table 20 and Table 21.

Table 12: Performance comparison of different methods with SigLIP [74] on ImageNet and finegrained datasets. Accuracy (%) and ECE (%) are reported.
<table><tr><td>(%)</td><td>Method</td><td>ImgNet</td><td>DTD</td><td>Flowers</td><td>Food101</td><td>SUN397</td><td>Aircraft</td><td>Pets</td><td>Caltech</td><td>UCF101</td><td>EuroSAT</td><td>Cars</td><td>Average</td></tr><tr><td rowspan="10">Aacy</td><td>Zero-Shot TPT</td><td>75.69</td><td>62.94</td><td>84.41 84.53</td><td>87.30</td><td>69.64</td><td>40.65 40.95</td><td>93.21 92.75</td><td>97.93</td><td>70.82 71.19</td><td>41.35</td><td>90.70</td><td>74.06</td></tr><tr><td></td><td>76.46</td><td>64.60</td><td></td><td>87.60</td><td>70.00</td><td></td><td></td><td>98.13</td><td></td><td>40.68</td><td>91.28</td><td>74.38</td></tr><tr><td>C-TPT</td><td>76.11</td><td>63.18</td><td>84.25</td><td>87.35</td><td>69.70</td><td>40.29</td><td>92.89</td><td>97.97</td><td>70.61</td><td>40.69</td><td>90.67</td><td>73.97</td></tr><tr><td>Penalty</td><td>76.45</td><td>64.07</td><td>84.33</td><td>87.45</td><td>69.91</td><td>40.38</td><td>92.72</td><td>96.19</td><td>70.90</td><td>41.83</td><td>91.11</td><td>74.12</td></tr><tr><td>SaLS</td><td>76.46</td><td>64.60</td><td>84.53</td><td>87.60</td><td>70.00</td><td>40.95</td><td>92.75</td><td>98.13</td><td>71.19</td><td>40.68</td><td>91.28</td><td>74.38</td></tr><tr><td>O-TPT</td><td>75.69</td><td>62.41</td><td>84.33</td><td>87.32</td><td>69.56</td><td>39.78</td><td>92.86</td><td>97.93</td><td>70.18</td><td>40.68</td><td>90.69</td><td>73.77</td></tr><tr><td>A-TPT</td><td>75.91</td><td>62.41</td><td>84.29</td><td>87.31</td><td>69.59</td><td>40.32</td><td>92.89</td><td>98.01</td><td>70.55</td><td>40.60</td><td>90.69</td><td>73.87</td></tr><tr><td>SoC</td><td>75.97</td><td>63.00</td><td>84.33</td><td>87.35</td><td>69.68</td><td>40.65</td><td>93.13</td><td>97.93</td><td>70.82</td><td>41.35</td><td>90.75</td><td>74.09</td></tr><tr><td>CoTS</td><td>76.46</td><td>64.60</td><td>84.53</td><td>87.60</td><td>70.00</td><td>40.95</td><td>92.75</td><td>98.13</td><td>71.19</td><td>40.68</td><td>91.28</td><td>74.38</td></tr><tr><td>E-CoTS</td><td>77.86</td><td>65.19</td><td>84.78</td><td>87.73</td><td>70.04</td><td>39.60</td><td>92.72</td><td>98.22</td><td>71.64</td><td>38.74</td><td>91.38</td><td>74.35</td></tr><tr><td rowspan="10">ECE</td><td>Zero-Shot</td><td>4.54</td><td>6.29</td><td>3.83</td><td>2.04</td><td>3.89</td><td>7.80</td><td>2.54</td><td>1.53</td><td>6.94</td><td>15.99</td><td>0.69</td><td>5.10</td></tr><tr><td>TPT</td><td>6.57</td><td>11.70</td><td>4.08</td><td>3.34</td><td>7.04</td><td>12.90</td><td>1.87</td><td>0.81</td><td>9.52</td><td>19.90</td><td>1.87</td><td>7.24</td></tr><tr><td>C-TPT</td><td>5.81</td><td>9.40</td><td>4.15</td><td>2.57</td><td>5.35</td><td>11.48</td><td>2.12</td><td>1.30</td><td>8.35</td><td>18.39</td><td>0.91</td><td>6.35</td></tr><tr><td>Penalty</td><td>6.49</td><td>9.68</td><td>4.21</td><td>2.69</td><td>6.90</td><td>12.90</td><td>2.60</td><td>1.63</td><td>8.30</td><td>14.35</td><td>1.43</td><td>6.47</td></tr><tr><td>SaLS</td><td>6.29</td><td>10.43</td><td>4.12</td><td>3.05</td><td>6.63</td><td>12.61</td><td>1.87</td><td>0.82</td><td>9.07</td><td>18.95</td><td>1.75</td><td>6.87</td></tr><tr><td>O-TPT</td><td>4.96</td><td>8.76</td><td>3.87</td><td>2.42</td><td>4.63</td><td>11.02</td><td>2.18</td><td>1.38</td><td>8.11</td><td>18.51</td><td>0.74</td><td>6.05</td></tr><tr><td>A-TPT</td><td>5.35</td><td>8.36</td><td>3.99</td><td>2.48</td><td>4.87</td><td>11.28</td><td>2.16</td><td>1.29</td><td>8.23</td><td>17.84</td><td>0.83</td><td>6.06</td></tr><tr><td>SoC</td><td>5.30</td><td>6.30</td><td>3.89</td><td>2.30</td><td>4.36</td><td>7.83</td><td>2.33</td><td>1.21</td><td>7.40</td><td>16.04</td><td>1.03</td><td>5.27</td></tr><tr><td>CoTS</td><td>4.18</td><td>4.67</td><td>4.18</td><td>2.20</td><td>3.68</td><td>7.55</td><td>2.27</td><td>1.37</td><td>6.65</td><td>15.12</td><td>1.18</td><td>4.82</td></tr><tr><td>E-CoTS</td><td>4.22</td><td>5.22</td><td>4.67</td><td>1.95</td><td>3.95</td><td>6.64</td><td>2.34</td><td>1.44</td><td>6.26</td><td>16.27</td><td>0.84</td><td>4.89</td></tr></table>

Table 13: Performance comparison of different methods with SigLIP [74] on ImageNet variants. Accuracy (%) and ECE (%) are reported.
<table><tr><td rowspan="2">Method</td><td colspan="5">Accuracy (%)</td><td colspan="5">ECE (%)</td></tr><tr><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td></tr><tr><td>Zero-Shot</td><td>45.07</td><td>68.43</td><td>89.31</td><td>66.59</td><td>67.35</td><td>15.90</td><td>7.26</td><td>1.42</td><td>7.43</td><td>8.00</td></tr><tr><td>TPT</td><td>46.73</td><td>69.23</td><td>89.89</td><td>67.08</td><td>68.23</td><td>17.25</td><td>9.45</td><td>0.68</td><td>10.37</td><td>9.43</td></tr><tr><td>C-TPT</td><td>45.84</td><td>68.94</td><td>89.57</td><td>66.88</td><td>67.81</td><td>17.08</td><td>8.65</td><td>0.88</td><td>9.57</td><td>9.04</td></tr><tr><td>Penalty</td><td>46.01</td><td>69.19</td><td>89.70</td><td>67.08</td><td>68.00</td><td>16.98</td><td>9.40</td><td>0.78</td><td>10.28</td><td>9.36</td></tr><tr><td>SaLS</td><td>46.73</td><td>69.23</td><td>89.89</td><td>67.08</td><td>68.23</td><td>16.69</td><td>9.11</td><td>0.86</td><td>9.98</td><td>9.16</td></tr><tr><td>O-TPT</td><td>45.25</td><td>68.41</td><td>89.34</td><td>66.56</td><td>67.39</td><td>16.54</td><td>7.79</td><td>1.05</td><td>7.99</td><td>8.34</td></tr><tr><td>A-TPT</td><td>45.45</td><td>68.78</td><td>89.46</td><td>66.80</td><td>67.62</td><td>16.71</td><td>8.10</td><td>0.96</td><td>8.69</td><td>8.62</td></tr><tr><td>SoC</td><td>45.72</td><td>68.75</td><td>89.51</td><td>66.85</td><td>67.71</td><td>16.39</td><td>8.05</td><td>1.02</td><td>8.39</td><td>8.46</td></tr><tr><td>CoTS</td><td>46.73</td><td>69.23</td><td>89.89</td><td>67.08</td><td>68.23</td><td>13.91</td><td>6.72</td><td>1.82</td><td>7.14</td><td>7.40</td></tr><tr><td>E-CoTS</td><td>56.35</td><td>70.70</td><td>90.95</td><td>67.44</td><td>71.36</td><td>10.64</td><td>6.42</td><td>2.44</td><td>7.43</td><td>6.73</td></tr></table>

## E Additional Analysis

## E.1 Sensitivity to Hand-crafted Prompt Initialization

We evaluate the performance of different methods under multiple initial prompt templates on ImageNet variants. Table 22 reports ACC (%) and ECE (%) for each method, showing how CoTS and E-CoTS consistently maintain low ECE while achieving high accuracy across prompts.

## E.2 Ensemble Weight α in E-CoTS

As shown in Fig. 5, the adaptive weighting strategy achieves a better balance between accuracy and calibration, avoiding the need to tune α for different settings.

## E.3 Similarity-based Analysis

Figure 6 shows the weak-strong accuracy ratio plotted against dataset-level similarity. This result demonstrates the effectiveness of the proposed weak-strong ensemble strategy.

Table 14: Standard deviations across three seeds on fine-grained datasets and ImageNet using ViT-B/16. Accuracy(%) and ECE(%) are reported in percentage points.
<table><tr><td></td><td>(%) Method</td><td></td><td>ImgNet DTD</td><td></td><td>Flowers Food101</td><td>SUN397</td><td></td><td>Aircraft Pets</td><td></td><td>Caltech UCF101</td><td>EuroSAT</td><td>Cars</td></tr><tr><td rowspan="11">Aacy ECE</td><td>TPT [57]</td><td>0.04</td><td>0.11</td><td>0.08</td><td>0.02</td><td>0.06</td><td>0.16</td><td>0.11</td><td>0.22</td><td>0.27</td><td>0.09</td><td>0.08</td></tr><tr><td>C-TPT [72]</td><td>0.01</td><td>0.15</td><td>0.12</td><td>0.03</td><td>0.11</td><td>0.09</td><td>0.02</td><td>0.10</td><td>0.05</td><td>0.02</td><td>0.11</td></tr><tr><td>Penalty [46]</td><td>0.02</td><td>0.10</td><td>0.23</td><td>0.01</td><td>0.06</td><td>0.35</td><td>0.19</td><td>0.18</td><td>0.15</td><td>0.06</td><td>0.07</td></tr><tr><td>SALS [46]</td><td>0.04</td><td>0.11</td><td>0.08</td><td>0.02</td><td>0.05</td><td>0.19</td><td>0.11</td><td>0.22</td><td>0.27</td><td>0.09</td><td>0.09</td></tr><tr><td>O-TPT [55]</td><td>0.02</td><td>0.11</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.06</td><td>0.06</td><td>0.12</td></tr><tr><td>A-TPT [1]</td><td>0.02</td><td>0.03</td><td>0.07</td><td>0.03</td><td>0.04</td><td>0.35</td><td>0.01</td><td>0.16</td><td>0.13</td><td>0.00</td><td>0.10</td></tr><tr><td>SoC [13]</td><td>0.03</td><td>0.05</td><td>0.12</td><td>0.03</td><td>0.04</td><td>0.01</td><td>0.12</td><td>0.11</td><td>0.07</td><td>0.01</td><td>0.05</td></tr><tr><td>CoTS</td><td>0.04</td><td>0.11</td><td>0.08</td><td>0.02</td><td>0.06</td><td>0.16</td><td>0.11</td><td>0.22</td><td>0.27</td><td>0.09</td><td>0.08</td></tr><tr><td>E-CoTS</td><td>0.04</td><td>0.28</td><td>0.06</td><td>0.03</td><td>0.09</td><td>0.20</td><td>0.26</td><td>0.12</td><td>0.18</td><td>0.14</td><td>0.13</td></tr><tr><td rowspan="13"></td><td>TPT [57]</td><td>0.07</td><td>0.10</td><td>0.07</td><td>0.04</td><td>0.14</td><td>0.12</td><td>0.10</td><td>0.12</td><td>0.38</td><td>0.04</td><td>0.07</td></tr><tr><td>C-TPT [72]</td><td>0.04</td><td>0.17</td><td>0.10</td><td>0.04</td><td>0.15</td><td>0.07</td><td>0.10</td><td>0.12</td><td>0.22</td><td>0.26</td><td>0.35</td></tr><tr><td>Penalty [46]</td><td>0.04</td><td>0.15</td><td>0.25</td><td>0.07</td><td>0.11</td><td>0.40</td><td>0.18</td><td>0.14</td><td>0.17</td><td>0.15</td><td>0.17</td></tr><tr><td>SALS [46]</td><td>0.06</td><td>0.18</td><td>0.11</td><td>0.01</td><td>0.03</td><td>0.11</td><td>0.09</td><td>0.06</td><td>0.36</td><td>0.38</td><td>0.08</td></tr><tr><td>O-TPT [55]</td><td>0.02</td><td>0.14</td><td>0.14</td><td>0.04</td><td>0.04</td><td>0.13</td><td>0.07</td><td>0.10</td><td>0.07</td><td>0.10</td><td>0.15</td></tr><tr><td>A-TPT [1]</td><td>0.02</td><td>0.25</td><td>0.13</td><td>0.03</td><td>0.08</td><td>0.31</td><td>0.10</td><td>0.16</td><td>0.08</td><td>0.00</td><td>0.10</td></tr><tr><td>SoC [13]</td><td>0.06</td><td>0.02</td><td>0.04</td><td>0.02</td><td>0.02</td><td>0.05</td><td>0.16</td><td>0.09</td><td>0.06</td><td>0.01</td><td>0.05</td></tr><tr><td>CoTS</td><td>0.08</td><td>0.61</td><td>0.12</td><td>0.03</td><td>0.19</td><td>0.28</td><td>0.07</td><td>0.22</td><td>0.29</td><td>0.27</td><td></td></tr><tr><td>E-CoTS</td><td>0.03</td><td>0.86</td><td>0.26</td><td></td><td>0.00</td><td>0.26</td><td>0.13</td><td>0.09</td><td>0.16</td><td>0.36</td><td>0.13</td><td>0.07 0.15</td></tr></table>

Table 15: Standard deviations across three seeds on ImageNet variants using ViT-B/16. Accuracy(%) and ECE(%) are reported in percentage points.
<table><tr><td rowspan="2">Method</td><td colspan="4">Accuracy (%)</td><td colspan="4">ECE (%)</td></tr><tr><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td></tr><tr><td>TPT [57]</td><td>0.07</td><td>0.14</td><td>0.04</td><td>0.03</td><td>0.15</td><td>0.24</td><td>0.03</td><td>0.05</td></tr><tr><td>C-TPT [72]</td><td>0.13</td><td>0.16</td><td>0.06</td><td>0.02</td><td>0.18</td><td>0.05</td><td>0.05</td><td>0.02</td></tr><tr><td>Penalty [46]</td><td>0.13</td><td>0.16</td><td>0.03</td><td>0.02</td><td>0.10</td><td>0.22</td><td>0.09</td><td>0.04</td></tr><tr><td>SALS [46]</td><td>0.07</td><td>0.14</td><td>0.04</td><td>0.03</td><td>0.04</td><td>0.24</td><td>0.13</td><td>0.03</td></tr><tr><td>O-TPT [55]</td><td>0.01</td><td>0.02</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.16</td><td>0.03</td><td>0.05</td></tr><tr><td>A-TPT [1]</td><td>0.04</td><td>0.19</td><td>0.06</td><td>0.04</td><td>0.14</td><td>0.18</td><td>0.07</td><td>0.06</td></tr><tr><td>SoC [13]</td><td>0.10</td><td>0.08</td><td>0.03</td><td>0.03</td><td>0.09</td><td>0.11</td><td>0.04</td><td>0.02</td></tr><tr><td>CoTS</td><td>0.07</td><td>0.14</td><td>0.04</td><td>0.03</td><td>0.29</td><td>0.18</td><td>0.05</td><td>0.08</td></tr><tr><td>E-CoTS</td><td>0.24</td><td>0.11</td><td>0.08</td><td>0.07</td><td>0.02</td><td>0.16</td><td>0.08</td><td>0.08</td></tr></table>

## E.4 Step-wise Analysis

Figure 7 presents ECE and ACC of CoTS and E-CoTS over iteration steps on DTD and ImageNet-A.   
As the number of steps increases, ECE tends to converge while ACC remains stable.

## E.5 Sensitivity to K and $T _ { 0 }$

We further investigate the sensitivity to the number of selected strong views K and the temperature initialization $T _ { 0 }$ . Figure 8 show that CoTS is generally robust to variations in both hyperparameters. We set $K = 6$ and $\breve { T } _ { 0 } = 1 . 0$ by default throughout the experiments.

## E.6 Comparison between Ensemble-Only and E-CoTS

We further compare base methods, ensemble-only, and E-CoTS across different TPT-based calibration methods (Table 23). Ensemble-only often improves accuracy but can sometimes increase ECE. By integrating CoTS into the ensemble prediction, E-CoTS reduces ECE while retaining the accuracy gains from ensembling.

Table 16: Brier scores (%) of different methods on fine-grained datasets with ViT-B/16.
<table><tr><td>(%) Method</td><td></td><td>ImgNet DTD</td><td></td><td>Flowers</td><td>Food101</td><td>SUN397</td><td>Aircraft</td><td>Pets</td><td>Caltech</td><td>UCF101</td><td>EuroSAT</td><td>Cars</td><td>Average</td></tr><tr><td rowspan="10">Brir sore</td><td>Zero-Shot [53]</td><td>16.44</td><td>18.45</td><td>14.22</td><td>9.92</td><td>18.33</td><td>14.77</td><td>7.47</td><td>6.11</td><td>14.55</td><td>19.36</td><td>16.44</td><td>14.19</td></tr><tr><td>TPT [57]</td><td>18.20</td><td>24.82</td><td>16.83</td><td>10.14</td><td>20.15</td><td>18.49</td><td>8.78</td><td>5.76</td><td>16.01</td><td>24.81</td><td>16.54</td><td>16.41</td></tr><tr><td>C-TPT [72]</td><td>16.88</td><td>21.12</td><td>13.95</td><td>10.29</td><td>19.36</td><td>14.40</td><td>7.20</td><td>5.86</td><td>15.17</td><td>22.36</td><td>16.61</td><td>14.84</td></tr><tr><td>Penalty [46]</td><td>18.23</td><td>23.46</td><td>16.28</td><td>10.43</td><td>20.03</td><td>17.37</td><td>10.21</td><td>6.57</td><td>16.13</td><td>19.26</td><td>16.47</td><td>15.86</td></tr><tr><td>SaLS [46]</td><td>17.94</td><td>23.36</td><td>16.23</td><td>9.98</td><td>19.90</td><td>17.61</td><td>8.60</td><td>5.83</td><td>15.75</td><td>22.66</td><td>16.34</td><td>15.84</td></tr><tr><td>O-TPT [55]</td><td>16.16</td><td>19.06</td><td>13.38</td><td>10.56</td><td>19.46</td><td>14.11</td><td>7.26</td><td>5.71</td><td>15.37</td><td>22.73</td><td>16.57</td><td>14.58</td></tr><tr><td>A-TPT [1]</td><td>16.47</td><td>19.71</td><td>13.45</td><td>10.27</td><td>19.00</td><td>15.04</td><td>7.44</td><td>6.34</td><td>15.65</td><td>19.36</td><td>16.53</td><td>14.48</td></tr><tr><td>SoC [13]</td><td>16.63</td><td>18.61</td><td>13.84</td><td>10.10</td><td>18.79</td><td>14.79</td><td>7.00</td><td>6.02</td><td>14.45</td><td>19.37</td><td>16.59</td><td>14.20</td></tr><tr><td>CoTS</td><td>16.61</td><td>19.51</td><td>14.24</td><td>9.73</td><td>18.34</td><td>14.14</td><td>7.43</td><td>6.62</td><td>14.36</td><td>21.62</td><td>16.22</td><td>14.44</td></tr><tr><td>E-CoTS</td><td>16.30</td><td>19.72</td><td>14.36</td><td>9.60</td><td>17.99</td><td>14.30</td><td>7.45</td><td>5.58</td><td>14.03</td><td>21.43</td><td>15.96</td><td>14.25</td></tr></table>

Table 17: Brier scores (%) of different methods on ImageNet variants with ViT-B/16.
<table><tr><td>(%)</td><td>Method</td><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td></tr><tr><td rowspan="9">Bfir score</td><td>Zero-Shot [53]</td><td>20.68</td><td>17.58</td><td>12.45</td><td>17.91</td><td>17.16</td></tr><tr><td>TPT [57]</td><td>23.95</td><td>20.10</td><td>12.82</td><td>22.12</td><td>19.75</td></tr><tr><td>C-TPT [72]</td><td>21.64</td><td>18.56</td><td>12.47</td><td>19.83</td><td>18.13</td></tr><tr><td>Penalty [46]</td><td>23.87</td><td>20.00</td><td>12.97</td><td>22.13</td><td>19.74</td></tr><tr><td>SaLS [46]</td><td>23.34</td><td>19.71</td><td>12.50</td><td>21.68</td><td>19.30</td></tr><tr><td>O-TPT [55]</td><td>20.32</td><td>17.41</td><td>12.40</td><td>17.98</td><td>17.03</td></tr><tr><td>A-TPT [1]</td><td>20.51</td><td>17.69</td><td>12.35</td><td>18.25</td><td>17.20</td></tr><tr><td>SoC [13]</td><td>20.63</td><td>17.66</td><td>12.25</td><td>18.09</td><td>17.16</td></tr><tr><td>CoTS</td><td>21.40</td><td>17.97</td><td>12.67</td><td>18.33</td><td>17.59</td></tr><tr><td></td><td>E-CoTS</td><td>21.93</td><td>17.75</td><td>12.10</td><td>18.39</td><td>17.54</td></tr></table>

## E.7 Comparison with ZERO and Our Ensemble Strategy

Table 24 compares ZERO [10] with our ensemble strategy under both zero-shot and TPT settings. Our ensemble achieves a better accuracy–calibration trade-off than ZERO, while E-CoTS further substantially reduces ECE and maintains competitive accuracy.

Table 18: Class-wise ECE (%) [33] of different methods on fine-grained datasets with ViT-B/16.
<table><tr><td>(%) Method</td><td></td><td>ImgNet DTD</td><td></td><td>Flowers Food101</td><td></td><td>SUN397</td><td>Aircraft Pets</td><td></td><td>Caltech UCF101</td><td></td><td>EuroSAT</td><td>Cars</td><td>Average</td></tr><tr><td rowspan="9">CECE</td><td>Zero-Shot [53]</td><td>0.04</td><td>1.42</td><td>0.62</td><td>0.21</td><td>0.13</td><td>0.57</td><td>0.70</td><td>0.24</td><td>0.52</td><td>6.59</td><td>0.23</td><td>1.03</td></tr><tr><td>TPT [57]</td><td>0.04</td><td>1.49</td><td>0.53</td><td>0.19</td><td>0.12</td><td>0.71</td><td>0.62</td><td>0.17</td><td>0.47</td><td>6.95</td><td>0.22</td><td>1.05</td></tr><tr><td>C-TPT [72]</td><td>0.04</td><td>1.39</td><td>0.54</td><td>0.24</td><td>0.12</td><td>0.61</td><td>0.60</td><td>0.22</td><td>0.55</td><td>7.37</td><td>0.23</td><td>1.08</td></tr><tr><td>Penalty [46]</td><td>0.04</td><td>1.43</td><td>0.54</td><td>0.19</td><td>0.12</td><td>0.66</td><td>0.93</td><td>0.23</td><td>0.49</td><td>5.96</td><td>0.22</td><td>0.98</td></tr><tr><td>SaLS [46]</td><td>0.04</td><td>1.46</td><td>0.53</td><td>0.18</td><td>0.12</td><td>0.69</td><td>0.62</td><td>0.18</td><td>0.47</td><td>6.38</td><td>0.22</td><td>0.99</td></tr><tr><td>O-TPT [55]</td><td>0.04</td><td>1.36</td><td>0.55</td><td>0.25</td><td>0.14</td><td>0.60</td><td>0.60</td><td>0.23</td><td>0.58</td><td>7.39</td><td>0.23</td><td>1.09</td></tr><tr><td>A-TPT [1]</td><td>0.04</td><td>1.38</td><td>0.54</td><td>0.24</td><td>0.13</td><td>0.60</td><td>0.63</td><td>0.25</td><td>0.57</td><td>6.59</td><td>0.23</td><td>1.02</td></tr><tr><td>SoC [13]</td><td>0.04</td><td>1.44</td><td>0.58</td><td>0.23</td><td>0.13</td><td>0.57</td><td>0.62</td><td>0.25</td><td>0.51</td><td>6.58</td><td>0.23</td><td>1.02</td></tr><tr><td>CoTS</td><td>0.04</td><td>1.25</td><td>0.58</td><td>0.20</td><td>0.12</td><td>0.55</td><td>0.72</td><td>0.25</td><td>0.50</td><td>6.07</td><td>0.24</td><td>0.96</td></tr><tr><td>E-CoTS</td><td></td><td>0.04</td><td>1.28</td><td>0.58</td><td>0.20</td><td>0.12</td><td>0.56</td><td>0.70</td><td>0.22</td><td>0.50</td><td>6.02</td><td>0.24</td><td>0.95</td></tr></table>

Table 19: Class-wise ECE (%) [33] of different methods on ImageNet variants with ViT-B/16.
<table><tr><td>(%)</td><td>Method</td><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td></tr><tr><td rowspan="9">CEECE</td><td>Zero-Shot [53]</td><td>0.30</td><td>0.06</td><td>0.18</td><td>0.06</td><td>0.15</td></tr><tr><td>TPT [57]</td><td>0.28</td><td>0.06</td><td>0.14</td><td>0.06</td><td>0.14</td></tr><tr><td>C-TPT [72]</td><td>0.29</td><td>0.06</td><td>0.16</td><td>0.06</td><td>0.14</td></tr><tr><td>Penalty [46]</td><td>0.28</td><td>0.06</td><td>0.14</td><td>0.06</td><td>0.14</td></tr><tr><td>SaLS [46]</td><td>0.28</td><td>0.06</td><td>0.14</td><td>0.06</td><td>0.14</td></tr><tr><td>O-TPT [55]</td><td>0.30</td><td>0.06</td><td>0.18</td><td>0.06</td><td>0.15</td></tr><tr><td>A-TPT [1]</td><td>0.30</td><td>0.06</td><td>0.17</td><td>0.06</td><td>0.15</td></tr><tr><td>SoC [13]</td><td>0.29</td><td>0.06</td><td>0.17</td><td>0.06</td><td>0.15</td></tr><tr><td>CoTS E-CoTS</td><td>0.27 0.26</td><td>0.06 0.06</td><td>0.17 0.16</td><td>0.06 0.06</td><td>0.14 0.13</td></tr></table>

Table 20: Adaptive ECE (%) [49] of different methods on fine-grained datasets with ViT-B/16.
<table><tr><td>(%) Method</td><td></td><td>ImgNet DTD</td><td></td><td>Flowers Food101</td><td></td><td>SUN397</td><td>Aircraft Pets</td><td></td><td>Caltech UCF101</td><td></td><td>EuroSAT</td><td>Cars</td><td>Average</td></tr><tr><td rowspan="9">AEECE</td><td>Zero-Shot [53]</td><td>0.03</td><td>1.33</td><td>0.48</td><td>0.18</td><td>0.10</td><td>0.55</td><td>0.54</td><td>0.17</td><td>0.41</td><td>6.57</td><td>0.12</td><td>0.95</td></tr><tr><td>TPT [57]</td><td>0.02</td><td>1.37</td><td>0.45</td><td>0.15</td><td>0.08</td><td>0.66</td><td>0.54</td><td>0.13</td><td>0.38</td><td>6.93</td><td>0.13</td><td>0.98</td></tr><tr><td>C-TPT [72]</td><td>0.02</td><td>1.24</td><td>0.43</td><td>0.19</td><td>0.09</td><td>0.59</td><td>0.48</td><td>0.15</td><td>0.43</td><td>7.33</td><td>0.13</td><td>1.01</td></tr><tr><td>Penalty [46]</td><td>0.02</td><td>1.35</td><td>0.46</td><td>0.16</td><td>0.08</td><td>0.63</td><td>0.80</td><td>0.16</td><td>0.40</td><td>5.97</td><td>0.13</td><td>0.92</td></tr><tr><td>SaLS [46]</td><td>0.02</td><td>1.35</td><td>0.46</td><td>0.15</td><td>0.08</td><td>0.64</td><td>0.55</td><td>0.14</td><td>0.38</td><td>6.40</td><td>0.13</td><td>0.94</td></tr><tr><td>O-TPT [55]</td><td>0.03</td><td>1.28</td><td>0.43</td><td>0.20</td><td>0.09</td><td>0.60</td><td>0.48</td><td>0.15</td><td>0.46</td><td>7.34</td><td>0.14</td><td>1.02</td></tr><tr><td>A-TPT [1]</td><td>0.03</td><td>1.28</td><td>0.43</td><td>0.20</td><td>0.10</td><td>0.57</td><td>0.49</td><td>0.17</td><td>0.46</td><td>6.57</td><td>0.13</td><td>0.95</td></tr><tr><td>SoC [13]</td><td>0.03</td><td>1.36</td><td>0.46</td><td>0.18</td><td>0.09</td><td>0.55</td><td>0.49</td><td>0.18</td><td>0.41</td><td>6.56</td><td>0.12</td><td>0.95</td></tr><tr><td>CoTS</td><td>0.03</td><td>1.29</td><td>0.48</td><td>0.17</td><td>0.09</td><td>0.56</td><td>0.60</td><td>0.18</td><td>0.41</td><td>6.11</td><td>0.12</td><td>0.91</td></tr><tr><td>E-CoTS</td><td></td><td>0.03</td><td>1.30</td><td>0.49</td><td>0.17</td><td>0.09</td><td>0.56</td><td>0.60</td><td>0.17</td><td>0.41</td><td>6.05</td><td>0.13</td><td>0.91</td></tr></table>

Table 21: Adaptive ECE (%) [49] of different methods on ImageNet variants with ViT-B/16.
<table><tr><td>(%)</td><td>Method</td><td>-A</td><td>-V</td><td>-R</td><td>-Sk</td><td>Average</td></tr><tr><td rowspan="8">ACE</td><td>Zero-Shot [53]</td><td>0.24</td><td>0.03</td><td>0.16</td><td>0.05</td><td>0.12</td></tr><tr><td>TPT [57]</td><td>0.21</td><td>0.03</td><td>0.12</td><td>0.05</td><td>0.10</td></tr><tr><td>C-TPT [72]</td><td>0.23</td><td>0.03</td><td>0.14</td><td>0.05</td><td>0.11</td></tr><tr><td>Penalty [46]</td><td>0.21</td><td>0.03</td><td>0.12</td><td>0.05</td><td>0.10</td></tr><tr><td>SaLS [46]</td><td>0.21</td><td>0.03</td><td>0.12</td><td>0.05</td><td>0.10</td></tr><tr><td>O-TPT [55]</td><td>0.25</td><td>0.03</td><td>0.16</td><td>0.05</td><td>0.12</td></tr><tr><td>A-TPT [1]</td><td>0.25</td><td>0.03</td><td>0.15</td><td>0.05</td><td>0.12</td></tr><tr><td>SoC [13]</td><td>0.24</td><td>0.03</td><td>0.14</td><td>0.05</td><td>0.11</td></tr><tr><td>CoTS E-CoTS</td><td></td><td>0.22 0.21</td><td>0.03 0.03</td><td>0.14 0.14</td><td>0.05 0.05</td><td>0.11 0.11</td></tr></table>

Table 22: Performance comparison under different initial prompts on ImageNet variants. Accuracy (%) and ECE (%) are reported.
<table><tr><td>Prompt template</td><td>Metric (%)</td><td>Zero-Shot [53]</td><td>TPT [57]</td><td>SaLS [46]</td><td>A-TPT [1]</td><td>SoC [13]</td><td>CoTS</td><td>E-CoTS</td></tr><tr><td> $\ " a _ { - } \mathrm { b a d } _ { - \mathrm { p h o t o } _ { - } \circ \mathbf { f } _ { - } \mathbf { a } " }$ </td><td>ACC ECE</td><td>58.60 4.47</td><td>61.61</td><td>61.61</td><td>57.58</td><td>59.03</td><td>61.61</td><td>63.66</td></tr><tr><td></td><td>ACC</td><td></td><td>12.07</td><td>10.61</td><td>5.13</td><td>5.73</td><td>5.16</td><td>5.12</td></tr><tr><td> $\ " a \mathrm { - } \mathrm { b } \mathrm { 1 a c k \mathrm { _ - } a n d \mathrm { _ - } w h i t e \mathrm { _ - } p h o t o \mathrm { _ - } o f \mathrm { _ - } a " }$ </td><td>ECE</td><td>55.46 6.50</td><td>59.30 18.07</td><td>59.30 16.67</td><td>56.84 7.88</td><td>57.09 8.06</td><td>59.30 5.95</td><td>60.69 7.23</td></tr><tr><td> $" { \bf a } _ { - } { \bf b } 1 \mathrm { u r r y } _ { - } \mathrm { p h o t o } _ { - } \circ { \bf f } _ { - } { \bf a } "$ </td><td>ACC</td><td>57.72</td><td>61.11</td><td>61.11</td><td>58.35</td><td>59.10</td><td>61.11</td><td>62.82</td></tr><tr><td></td><td>ECE</td><td>4.87</td><td>15.67</td><td>12.72</td><td>10.27</td><td>8.74</td><td>5.32</td><td>5.25</td></tr><tr><td> $\ " a _ { - } \mathrm { g o o d } _ { - } \mathrm { p h o t o } _ { - } \mathrm { o f } _ { - } \mathrm { a } "$ </td><td>ACC</td><td>57.67</td><td>61.34</td><td>61.34</td><td>58.48</td><td>59.47</td><td>61.34</td><td>63.21</td></tr><tr><td></td><td>ECE</td><td>4.67</td><td>14.90</td><td>12.84</td><td>8.14</td><td>7.86</td><td>5.35</td><td>5.57</td></tr><tr><td>&quot;a_high_contrast_photo_of_a&quot;</td><td>ACC</td><td>56.35</td><td>60.20</td><td>60.20</td><td>57.21</td><td>57.79</td><td>60.20</td><td>61.59</td></tr><tr><td></td><td>ECE</td><td>5.49</td><td>19.30</td><td>16.57</td><td>8.84</td><td>9.62</td><td>5.43</td><td>6.20</td></tr><tr><td></td><td>ACC</td><td>57.63</td><td>61.28</td><td>61.28</td><td>56.57</td><td>58.28</td><td>61.28</td><td>62.82</td></tr><tr><td> $^ { " } \mathsf { a \mathrm { _ { - } \mathrm { 1 0 w \mathrm { _ { - } c o n t r a s t \mathrm { _ { - } p h o t o \mathrm { _ { - } \mathrm { 0 f \mathrm { _ { - } a ^ { \ d } } } } } } } } }$ </td><td>ECE</td><td>5.21</td><td>17.34</td><td>8.12</td><td>8.60</td><td>14.98</td><td>5.31</td><td>5.55</td></tr><tr><td></td><td>ACC</td><td>56.25</td><td>60.10</td><td>60.10</td><td>58.88</td><td></td><td></td><td></td></tr><tr><td>&quot;a_photo_of_a_small&quot;</td><td>ECE</td><td>4.19</td><td>14.97</td><td>12.61</td><td>5.10</td><td>58.15 6.93</td><td>60.10 5.07</td><td>61.67 4.80</td></tr></table>

![](images/b23a9b3c454fda0710421b8216368d0f5c9514a17da5e4dc74ed89d02b71b118.jpg)  
Figure 5: Effect of ensemble weight α on ImageNet variants using ViT-B/16. Accuracy (%) and ECE (%) are reported.

![](images/ebfd0c46e49e1f7ab0fd32a9c35babbebb3cd93b7b5a0724b6d288c2babfe813.jpg)  
Figure 6: Weak-strong accuracy ratio versus dataset-level similarity across different datasets.

![](images/12b96f5f1f2f1e8038e545b8330d3740720c40284266bcd21ccbd54060853d4a.jpg)  
(a) DTD

![](images/82eac04ab6668c6b98c217e826517ff799d6c44e0de1bc4c213ffd9d3bf95d89.jpg)  
(b) ImageNet-A  
Figure 7: ACC(%) and ECE(%) curves of CoTS and E-CoTS on DTD and ImageNet-A over iteration steps using ViT-B/16.

Table 23: Performance of base, direct ensemble, and E-CoTS across different TPT-based methods using ViT-B/16. Accuracy (%) and ECE (%) are reported on fine-grained datasets and ImageNet variants.
<table><tr><td rowspan="2">(%)</td><td rowspan="2">Method</td><td colspan="5">Fine-grained datasets</td><td colspan="5">ImageNet variants</td></tr><tr><td>TPT [57]</td><td>C-TPT [72]</td><td>O-TPT [55]</td><td>A-TPT [1]</td><td>SoC [13]</td><td>TPT [57]</td><td>C-TPT [72]</td><td>O-TPT [55]</td><td>A-TPT [1]</td><td>SoC [13]</td></tr><tr><td rowspan="3">ACC</td><td>base</td><td>65.25</td><td>64.57</td><td>64.04</td><td>64.15</td><td>63.96</td><td>60.72</td><td>59.26</td><td>57.51</td><td>58.18</td><td>58.17</td></tr><tr><td>Ensemble</td><td>65.32</td><td>65.27</td><td>64.99</td><td>65.12</td><td>65.11</td><td>62.90</td><td>62.66</td><td>62.03</td><td>62.41</td><td>61.84</td></tr><tr><td>E-CoTS</td><td>65.28</td><td>65.24</td><td>64.95</td><td>65.09</td><td>65.11</td><td>62.99</td><td>62.67</td><td>61.97</td><td>62.28</td><td>61.81</td></tr><tr><td rowspan="3">ECE</td><td>base</td><td>11.25</td><td>5.24</td><td>4.80</td><td>4.27</td><td>4.30</td><td>12.06</td><td>6.74</td><td>4.99</td><td>4.37</td><td>4.83</td></tr><tr><td>Ensemble</td><td>12.28</td><td>5.35</td><td>4.90</td><td>4.00</td><td>4.12</td><td>13.36</td><td>7.86</td><td>5.36</td><td>5.38</td><td>4.08</td></tr><tr><td>E-CoTS</td><td>4.18</td><td>4.19</td><td>4.29</td><td>3.86</td><td>4.18</td><td>5.47</td><td>5.18</td><td>4.51</td><td>4.84</td><td>4.66</td></tr></table>

![](images/3b8a2f6b18bc5bb702c533491fdb101f2d454d6c65b8bd6dc72d586280d0ff55.jpg)  
Figure 8: Sensitivity analysis of CoTS to the number of selected strong views K and the temperature initialization $T _ { 0 } .$ . ECE (%) is reported on fine-grained datasets and ImageNet variants.

Table 24: Comparison of ZERO [10] and our ensemble strategy using ViT-B/16. Accuracy (%) and ECE (%) are reported on fine-grained datasets and ImageNet variants.
<table><tr><td rowspan="2">Method</td><td colspan="2">Fine-grained datasets</td><td colspan="2">ImageNet variants</td></tr><tr><td>ACC (%)</td><td>ECE (%)</td><td>ACC (%)</td><td>ECE (%)</td></tr><tr><td>Zero-Shot [53]</td><td>63.95</td><td>4.33</td><td>57.22</td><td>4.98</td></tr><tr><td>w/ ZERO [10]</td><td>64.41</td><td>18.24</td><td>62.42</td><td>19.54</td></tr><tr><td>w/ Ensemble (Ours)</td><td>65.21</td><td>4.00</td><td>61.86</td><td>5.03</td></tr><tr><td>TPT [57]</td><td>65.19</td><td>11.30</td><td>60.74</td><td>11.90</td></tr><tr><td>w/ ZERO [10]</td><td>64.65</td><td>27.26</td><td>62.97</td><td>28.34</td></tr><tr><td>w/ Ensemble (Ours)</td><td>65.32</td><td>12.28</td><td>62.90</td><td>13.36</td></tr><tr><td>w/E-CoTS</td><td>65.22</td><td>4.31</td><td>62.95</td><td>5.38</td></tr></table>

![](images/cb0695725f01d65ed6407aec06767148b789e15ea38d7f6d16754d73c86ca5d6.jpg)  
(a) SoC [13]: DTD

![](images/d9650e9e377e513c395ad8a06c91b43c1fe6e33d56803fb98e1f3cb4028f3b70.jpg)  
(b) SoC [13]: Flowers

![](images/3465468ab5bb4cc48ad355421451471383f48fb6b3a5ef5c8eadc9060533d4a0.jpg)  
(c) SoC [13]: Pets

![](images/0990199c13582cbae0f62a989a48aa427a5768a9ea5dd27f42bca34d93239fe7.jpg)  
(d) SoC [13]: IN-V

![](images/9225ff793f42a0afb976ab1d6a4273f10226745a8e74faa519304a34d27aa1ad.jpg)  
(e) O-TPT [55]: DTD

![](images/8fa52e1f4385052b716b1686578df00ffa2e50c4776fb6b03456d03b3d5b4592.jpg)  
(f) O-TPT [55]: Flow-

![](images/7afdd1399636423aa3c133d5980e1ae834e1a8c0eb78cc312212f7682f140f43.jpg)  
(g) O-TPT [55]: Pets

![](images/718860dd4c1a447c5ba1b8eb6eaec5b4fffa75d96f357646d92e4c64ce086434.jpg)  
(h) O-TPT [55]: IN-V

![](images/5e00fb1ff36644918e2d02a41ea7f29b5c60604f3511998ae74ed1a3d6728e74.jpg)  
(i) A-TPT [1]: DTD

ers  
![](images/0bd414b07370008f0986d68fbb20e719cb32e230cee2e38546cddd23f4a6e30e.jpg)  
(j) A-TPT [1]: Flowers

![](images/b7d94768bc7295eb0525e85df5a7b55ef64f8dfeab047bcb59b7c24983b8f529.jpg)

![](images/67d46e326b8d67d74a7266de78f2f270a26e58dde6c095f791bd8e122ef3ba93.jpg)

![](images/0937d26848ff438897044bc373058b2eb5b891bf5e18600afe17b66db98c1820.jpg)

(k) A-TPT [1]: Pets  
![](images/a78f9e72eaea61fb69d67421cc19513acdfa1fbf29959787d5301bbf84d54050.jpg)  
(m) CoTS: DTD

(l) A-TPT [1]: IN-V  
![](images/01e8a103421ab9b3440b49928153a89cb307a3af119c2c25dc76e0e0c43e09bf.jpg)

![](images/01a033be94dcf9956d91f2292b288696eb798918b378cafd67bf65a113682c91.jpg)

![](images/bb0c52cf069a85b321236ff31d8aec3aed7cbd83012a9b3181b5f780a86f160a.jpg)  
(q) E-CoTS: DTD

(n) CoTS: Flowers  
![](images/69f31a3b846eae6ca2dbe64a05b7c32d0936054c52bdc44d572697cae2cf53a7.jpg)  
(r) E-CoTS: Flowers

(o) CoTS: Pets  
![](images/ac95277b2a41ad99a236bbdd3e581584449c74ccbde2e8f3afdb51f4e12eb96f.jpg)  
(s) E-CoTS: Pets

(p) CoTS: IN-V  
![](images/28dbf3db14e571f90ca05ef352cd17c984e38bd528f65eb3355cce88a4e547bb.jpg)  
(t) E-CoTS: IN-V

Figure 9: Reliability diagrams of SoC [13], O-TPT [55], A-TPT [1], CoTS , and E-CoTS on DTD, Flowers, Pets, and ImageNet-V using ViT-B/16.

## F Reliability Diagram

As shown in Fig. 9, CoTS and ecots achieve small accuracy-confidence gaps across datasets, indicating reliable calibration.