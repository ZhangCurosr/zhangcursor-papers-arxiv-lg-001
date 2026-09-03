# RETHINKING THE TEACHER-STUDENT FRAMEWORK FOR TEST-TIME ADAPTATION

Damian Sójka1,2\* Marc Masana3 Bartłomiej Twardowski4,5 Sebastian Cygert6,7

1Poznan University of Technology 2Akces NCBR 3Graz University of Technology 4IDEAS Research Institute 5Computer Vision Center, Universitat Autonoma de Barcelona 6NASK - National Research Institute 7Gdańsk University of Technology

## ABSTRACT

Test-Time Adaptation (TTA) has recently emerged as a promising strategy that allows the adaptation of pre-trained models to changing data distributions at deployment time, without access to any labels. To mitigate error accumulation, researchers have widely adopted the teacher-student framework, though its long-term stability is often taken for granted. In this work, we challenge the common strategy of setting the teacher weights to an exponential moving average of the student by showing that error accumulation still occurs, although it is mostly apparent on longer sequences compared to those commonly utilized. We analyze the stability-plasticity trade-off within the teacher-student framework and propose to use an intransigent teacher that does not update its weights. Surprisingly, we show that this simple change allows TTA methods to significantly improve their performance on multiple datasets with longer scenarios and result in increased robustness to changes in hyperparameters. Finally, we show that those changes can be seamlessly and effectively applied to various architectures and experimental setups, including semantic segmentation. The code is available at https://github.com/dmn-sjk/intransigent\_teacher.

## 1 INTRODUCTION

Machine learning models typically assume that training and testing data originate from a similar distribution. However, in real-world applications, distribution shifts between training (source) and testing (target) data domains are common and can lead to performance issues throughout inference (Geirhos et al., 2019; Hendrycks & Dietterich, 2019; Koh et al., 2021). Test-Time Adaptation (TTA) (Wang et al., 2021) allows for an online adaptation of a pre-trained model to the changing data distributions during testing, in an unsupervised way. While many TTA methods have been developed in recent years (Gong et al., 2022; Goyal et al., 2022; Marsden et al., 2024a; Niu et al., 2022; 2023; Rusak et al., 2022; Sun et al., 2020; Wang et al., 2022; Yuan et al., 2023), important challenges remain, such as adaptation over very long scenarios (Press et al., 2023), robustness to noisy data (Gong et al., 2023), and sensitivity to hyperparameter change (Boudiaf et al., 2022; Zhao et al., 2023; Cygert et al., 2026).

The teacher-student paradigm is a popular framework in TTA (Chen et al., 2022; Döbler et al., 2022; Sójka et al., 2023; Wang et al., 2022; Yuan et al., 2023; Zhou et al., 2024), where the teacher weights are updated as the Exponential Moving Average (EMA) of the student weights (i.e., EMA teacher). This strategy follows pioneering works in semisupervised learning (Laine & Aila, 2017; Tarvainen & Valpola, 2017), representation learning (Grill et al., 2020; He et al., 2020; Oquab et al., 2024), and learning under label noise (Liu et al., 2020; Nguyen et al., 2020). The averaged model provides more accurate and consistent predictions, which the student uses for training. While recent TTA survey (Maharana et al., 2026) suggests this framework inherently prevents model collapse and enables state-of-theart performance, our findings challenge this consensus.

In this work, we present experimental evidence indicating that the teacher-student framework does not strictly prevent error accumulation, which can result in model collapse (i.e., falling below the accuracy of the source model). Moreover, we demonstrate that state-of-the-art TTA methods relying on this stabilization (Chen et al., 2022; Wang et al., 2022; Yuan et al., 2023; Brahma & Rai, 2023) are prone to significant accuracy degradation on extended test sequences. The EMA teacher merely postpones the effects of error accumulation. This reveals a critical flaw in common evaluation protocols, which have previously failed to detect this issue. In search for a solution, we observe that employing a simple technique of making the teacher more intransigent (not updating the teacher's weights) prevents model collapse over very long testing sequences (see Fig. 1). Interestingly, such a teacher can also enable students to fulfill the age-old cliche of surpassing their teacher (see Fig. 7). Based on that finding, we take a closer look at the stability-plasticity trade-off (Mermillod et al., 2013) within teacher-student frameworks and how it affects the final model performance. We show that while increased teacher plasticity can lead to better performance on the current data in the short run, using a more stable teacher leads to a better adaptation over longer scenarios. This is due to the error accumulation of the plastic model, which adapts well to the current data but fails to maintain generalization over time, as explored in Sec. 4.2.

Our contributions are as follows:

• We analyze the teacher-student framework commonly used in TTA, demonstrating that an EMA teacher fails to prevent model collapse on longer test sequences.

• We observe that maintaining the teacher's weights fixed during adaptation (Intransigent Teacher – IT), effectively prevents model collapse while allowing the student to surpass the teacher. We find that IT works well because it avoids negative flips and tends to keep the representation strength intact.

• This simple solution (i.e., IT) can be combined with state-of-the-art methods, mitigating error accumulation by easily replacing any EMA teacher and obtaining reliable performance. IT can serve as a simple yet effective comparative baseline for future TTA works.

![](images/62fd27776ef74981d91519a6e1c374205ece0354b9c2295d604cbb047a7a6023.jpg)  
Figure 1: Average classification accuracy [%] for the baseline methods, with and without our Intransigent Teacher technique, evaluated across the original benchmarks and their long variants (20× repetitions). While baseline methods exhibit performance degradation as test sequence length increases, our Intransigent Teacher modification to the teacher-student framework effectively mitigates this decay and maintains robust accuracy.

## 2 RELATED WORK

Test-time adaptation. TTA (Wang et al., 2021) aims to adapt a pre-trained model to shifting data distributions during testing, without any labels. The adaptation is typically performed using unsupervised optimization objectives, often relying on noisy pseudo-labels, which can lead to error accumulation over multiple iterations. Therefore, numerous strategies have been developed to circumvent that issue, including various reset strategies (Wang et al., 2022; Zhang et al., 2022; Press et al., 2023), filtering unreliable samples (Niu et al., 2022; Zhou et al., 2025), and the teacher-student framework (Chen et al., 2022; Wang et al., 2022; Yuan et al., 2023; Sójka et al., 2023; Brahma & Rai, 2023; Shi et al., 2025: Zhou et al., 2025), among others. The teacher-student framework was introduced to TTA by the CoTTA method (Wang et al., 2022), where the use of exponential moving average (EMA) of weights was proposed. However, as EMA of the student's weights forms the teacher, error accumulation is merely delayed rather than eliminated. In this work, we analyze the teacher-student framework to address these limitations. Other works have explored strategies that retain the fixed source model. ROID (Marsden et al., 2024a) continually ensembles source and gradient-updated model weights to stabilize the adaptation. GRoTTA (Li et al., 2023) and TRIBE (Su et al., 2024) expand on the teacher-student framework by incorporating the source model as a third one for additional regularization. While these methods employ complex designs with extra loss terms and models, we explore how a simple teacher-student framework modification can address the performance degradation problem in TTA, especially for longer sequences. Parallel works (Zhou et al., 2024; Hoang et al., 2024) also propose to evaluate existing TTA methods over very long scenarios. Our work is complementary, since they propose adaptation methods (which require parameter tuning and access to source data (Hoang et al., 2024)), while we focus on evaluating existing approaches on a more extensive experimental setup, and observe that a simpler modification improves the reliability in such conditions.

Plasticity-stability trade-off. EMA ensemble of weights is parametrized by the β trade-off parameter, which determines the balance between retaining old averaged weights of the teacher and incorporating newly updated student ones, commonly set in TTA to 0.999 (Wang et al., 2022) (following temporal ensembling works (Laine & Aila, 2017)). where $\beta = 1$ means full stability (frozen model), and $\beta = 0$ means maximal plasticity. The trade-off has been widely analyzed in continual learning (Mermillod et al., 2013; Chaudhry et al., 2018; Masana et al., 2022), where the learner needs to balance learning of new tasks with the risk of forgetting the previously acquired knowledge. While a variety of strategies have been developed in continual learning, the most successful ones are those that promote stability. Many recent works freeze the feature extractor and learn only the classification head (Ma et al., 2023; Panos et al.

2023; Goswami et al., 2024; Tscheschner et al., 2025). Our work is inspired by those studies and aims to analyze the plasticity-stability trade-off within TTA.

Teacher-student framework. In knowledge distillation (Hinton et al., 2015), usually a larger model (teacher) guides the optimization of the target model (student) by providing informative training signals (teachers' outputs). Such a strategy is also commonly used in continual learning to prevent forgetting of previous tasks (Buzzega et al., 2020; Kirkpatrick et al., 2017). Self-distillation is a special case in which the teacher and student have the same architecture. It has been shown that in such a scenario, the student can outperform its teacher (Furlanello et al., 2018). In their work, the teacher is updated at the end of every training epoch by copying students' weights. They show improvement gains until such a procedure is repeated three times. Since TTA lacks access to labeled data, updating the teacher might result in even more significant error accumulation.

## 3 PRELIMINARIES

This study addresses continual TTA. The goal is to adapt a pre-trained (source) model $f _ { \theta _ { 0 } } ( \cdot )$ , initially trained on a labeled dataset $\mathscr { D } _ { s } { = } \{ ( \pmb { x } _ { i } , \pmb { y } _ { i } ) \} _ { i = 1 } ^ { N _ { s } }$ with $\pmb { x } _ { i } \sim \mathcal { P } _ { s } ( \pmb { x } )$ , to a sequence of unlabeled test data batches arriving sequentially during inference:

$$
\mathcal D _ { t } ^ { ( 1 ) } , \ \cdots , \ \mathcal D _ { t } ^ { ( T ) } \mathrm { ~ w h e r e ~ } \mathcal D _ { t } ^ { ( k ) } = \{ \pmb x _ { j } ^ { ( k ) } \} _ { j = 1 } ^ { N _ { k } } , \ x _ { j } ^ { ( k ) } \sim \mathcal Q ^ { ( k ) } ( { \pmb x } ) ,
$$

and test distributions evolve over time: ${ \mathcal { Q } } ^ { ( k ) } ( { \pmb x } ) \neq { \mathcal { Q } } ^ { ( k + 1 ) } ( { \pmb x } )$ for some k, with ${ \mathcal Q } ^ { ( k ) } ( { \pmb x } ) \neq { \mathcal P } _ { s } ( { \pmb x } )$ in general (i.e., test data is out-of-distribution and non-stationary). Adaptation must be performed on-the-fly using only the current and/or past unlabeled test batches, without access to source data or labels.

Both the student and the teacher start the adaptation with the same set of weights $\pmb { \theta } _ { 0 }$ pre-trained on the source data. During TTA, the student's weights $\pmb { \theta } _ { n } ^ { s }$ at step n are updated via backpropagation. The teacher's weights are updated as an EMA of the student's weights:

$$
\pmb { \theta } _ { n } ^ { t } = \beta \cdot \pmb { \theta } _ { n - 1 } ^ { t } + ( 1 - \beta ) \cdot \pmb { \theta } _ { n } ^ { s }\tag{1}
$$

where $\beta$ (e.g., 0.999) is the EMA factor controlling the momentum of teacher's update. The teacher is usually used for the final predictions.

## 4 PRELIMINARY EXPERIMENTS

In this section, we aim to gain a deeper understanding of the adaptation process within the teacher-student framework, with particular emphasis on the plasticity-stability trade-off. To this end, we conduct two experiments. In the first experiment (Sec. 4.1), adaptation relies exclusively on a simple self-supervised loss combined with the teacher-student framework, deliberately excluding any additional regularization or stabilization techniques. This minimal setup allows us to isolate the intrinsic behavior of the teacher-student mechanism. In the second experiment (Sec. 4.2), we systematically vary the level of plasticity across the teacher-student frameworks used in existing TTA methods and analyze the effects of low-to-none plasticity.

## 4.1 EFFECT OF TEACHER-STUDENT FRAMEWORK ON THE SIMPLE SELF-SUPERVISED ADAPTATION

The teacher-student framework is widely studied in TTA, with great results being achieved by popular methods such as AdaContrast (Chen et al., 2022) and CoTTA (Wang et al., 2022). Although those techniques rely on other components (e.g., memory queue or weight restoration), they share a common trend of incorporating a self-supervision loss. We dissect the objectives directly related to the learning (their self-supervised losses), which allows for analyzing the teacher-student framework impacted only by the component the most related to the stability-plasticity trade-off. We probe standard EMA teachers (ET) as proposed in their original works (both using β = 0.999) and compare them with the presented intransigent teacher (IT) technique (β = 1). Intransigent teachers have all trainable parameters frozen. Their batch normalization statistics are calculated on a per-batch basis (Schneider et al., 2020b), as commonly done in TTA (Niu et al., 2022; 2023; Wang et al., 2021; 2022). The final predictions are taken from the student model. As a motivation example, we evaluate on the popular ImageNet-C corruption benchmark (Hendrycks & Dietterich, 2019) on a common level 5 corruption sequence from Wang et al. (2022). Furthermore, we evaluate on the CCC benchmark (Press et al., 2023) and introduce the setting with ImageNet-C repeated 20 times (ImageNet-C (L)), to better assess the teacher-student framework's performance on longer scenarios. The source model is trained on the clean ImageNet dataset (Deng et al., 2009). Figure 2 shows accuracy over time and Table 1 shows numerical results. where the evaluated objectives are described as:

• Consistency: The loss from CoTTA (Wang et al., 2022), which minimizes the cross-entropy consistency between teacher and student predictions. The input of the teacher is transformed via augmentations.

• Contrastive: The adaptation objective from AdaContrast (Chen et al., 2022), which uses a MoCo-inspired (He et al., 2020) contrastive task in which features from different views of the same image (positive pairs) are pulled closer, while features from different images (negative pairs) are pushed away. Input for both teacher and student is augmented by randomly drawing two strong augmentations.

The results from the above experiment lead to the following observations:

Observation 1: Self-supervised objective combined with an intransigent teacher provides great reliability on its own. This is a very positive result, as recent work (Press et al., 2023) observed that on CCC, most of the current adaptation methods result in model collapse. This is especially interesting in the case of Contrastive, where the contrastive loss changes only the backbone of the model without changing the classification layer, mostly relying on feature alignment.

Observation 2: An EMA on its own does not prevent error accumulation. As clearly shown in Figure 2, the problems when using EMA take some time to become apparent and are only visible over long sequences. For both losses, performance degradation starts rather early, after the 2nd and 3rd loops, respectively.

Observation 3: When using EMA, there is a small gap between the teacher and student accuracies. When an intransigent teacher with consistency loss is used, the student performance is reliable and comparable to the teacher, while in the case of the contrastive loss, the student is able to outperform their teacher significantly.

## 4.2 EFFECTS OF INTRANSIGENCE

To better understand the plasticity-stability trade-off, we evaluate performance when varying $\beta \in [ 0 . 9 , 1 . 0 ]$ , where 1.0 means using an intransigent teacher, and 0.999 is the default value for the two original methods we compare: AdaContrast and CoTTA. We evaluate on CIFAR10-C (C10-C) and ImageNet-C (IN-C) (Hendrycks & Dietterich, 2019) benchmarks, with each common adapta-

Table 1: Mean accuracy [%] of student models on test-time adaptation benchmarks. ET stands for exponential moving average teacher, and IT indicates the intransigent teacher. IN-C (L) stands for the ImageNet-C adaptation sequence being repeated 20 times.
<table><tr><td>Loss</td><td>Teacher</td><td>IN-C</td><td>IN-C (L)</td><td>CCC</td></tr><tr><td>Source</td><td>none</td><td>18.0</td><td>18.0</td><td>16.8</td></tr><tr><td rowspan="2">Consistency</td><td>ET</td><td>27.3</td><td>7.9</td><td>1.6</td></tr><tr><td>IT</td><td>28.8</td><td>31.3</td><td>27.2</td></tr><tr><td rowspan="2">Contrastive</td><td>ET</td><td>35.5</td><td>22.1</td><td>5.8</td></tr><tr><td>IT</td><td>35.4</td><td>36.9</td><td>31.8</td></tr></table>

![](images/835eb3f5da97a72020cac8e746780e8c1f4ce6ede93dc30faba9627c46f237ad.jpg)  
Figure 2: Per-batch accuracy on repeated ImageNet-C (20 loops) with EMA teacher (ET) and intransigent teacher (IT), both for teacher (solid) and student (dashed). Only the self-supervised losses, without any additional components (or restart mechanisms), allow for successful adaptation over long sequences when the teacher is intransigent.

tion sequence from Wang et al. (2022) repeated 20 times (L). The source model is either trained on clean CI-FAR10 (Krizhevsky, 2009) or ImageNet (Deng et al., 2009), depending on the benchmark. The average final performance is reported in Table A.11 (Appendix), while the accuracy evolution throughout the sequence is shown in Figure 3, with the accuracy on the common-length benchmarks indicated at the first tick of the x-axis. Results with different method–dataset combinations are presented in the Appendix (Fig. A.7 and Fig. A.8).

Intransigent teachers guide students more consistently than lenient alternatives. While increasing plasticity (decreasing β) improves short-term student accuracy, performance tend to collapse over longer sequences. For example, using an EMA with $\beta = 0 . 9 9 9 9$ initially outperforms IT, but accuracy degrades significantly toward the end of the test sequence. Consequently, fixed teacher weights provide the most reliable long-term performance. We further demonstrate this vulnerability in Figure A.4 in the Appendix using CoTTA on ImageNet-C. Although a slowly-updating teacher (β = 0.9999) achieves superior results over 20 repetitions, its performance falls significantly below the IT baseline when the sequence extends to 100 repetitions. This highlights a critical failure mode in TTA. Methods often depend on adaptation sequence lengths. While carefully tuned EMA parameters may mitigate this, hyperparameter selection for TTA remains an open challenge (Boudiaf et al., 2022; Zhao et al., 2023) and may necessitate specific assumptions about the deployment environment, such as the length of the adaptation sequence.

![](images/9a6832d4221d531f8939e23da8d283bda141f7d89360c4633b6fd9f4f99d2a4a.jpg)

![](images/293d2f160629c59ddf89a27fcd706857c07e939f4e958f51c7615a06940a06b7.jpg)  
Figure 3: Mean accuracy [%] for each loop of common testing sequence on ImageNet-C (L) with AdaContrast (left) and CIFAR10-C (L) with CoTTA (right). The brown dashed line indicates the source model accuracy as a reference.

## 5 METHODOLOGY

As presented in previous sections, using an intransigent teacher provides a great model consistency throughout longer scenarios without encountering catastrophic error accumulation. Here, we describe how we extend existing TTA methods with an intransigent teacher strategy for further experiments (Sec. 5.1) and provide the theoretical grounding for its effectiveness (Sec. 5.2).

## 5.1 APPOINTING AN INTRANSIGENT TEACHER

We extend 4 popular TTA methods that originally utilize the teacherstudent framework, AdaContrast (Chen et al., 2022), CoTTA (Wang et al., 2022), RoTTA (Yuan et al., 2023), and PETAL (Brahma & Rai, 2023), with our proposed intransigent teacher. The main difference between the classic teacher-student framework with EMA teacher and the intransigent teacher is the value of the $\beta$ parameter. In the case of IT, the parameter $\beta$ is equal to 1, so the weights are fixed at their initial pre-trained values and Eq. 1 simplifies to:

$$
\pmb { \theta } _ { n } ^ { t } = \pmb { \theta } _ { n - 1 } ^ { t } \quad \Longrightarrow \quad \pmb { \theta } _ { n } ^ { t } = \pmb { \theta } _ { 0 }
$$

![](images/f9a15214fe15f185b7ea81ab47b3ebb1b1cc1a79f814dad362305ab18ad18e57.jpg)  
Figure 4: The Intransigent Teacher. The student's weights are updated using gradientbased optimization, while the teacher's weights remain fixed throughout adaptation.

(2)

Additionally, the student model is used to generate the final predictions, regardless of which predictions are used in the original method. We maintain the same usage and adaptation of batch normalization statistics as the underlying TTA method for both the teacher and student networks. Figure 4 illustrates our adaptation process. Descriptions of strategies extended with IT are provided in the Appendix (Sec. A.2).

## 5.2 ANALYTICAL FRAMEWORK FOR INTRANSIGENT TEACHERS

To understand why standard EMA teachers fail on extended sequences, we can formalize the feedback loop of noise inherent in TTA. The student's weights $\pmb { \theta } _ { n } ^ { s }$ at step n are updated via backpropagation on an adaptation loss ${ \bar { \boldsymbol { L } } } ,$ relying on the teacher's pseudo-labels:

$$
\pmb { \theta } _ { n } ^ { s } = \pmb { \theta } _ { n - 1 } ^ { s } - \eta \cdot \nabla _ { \pmb { \theta } ^ { s } } L ( \pmb { \theta } _ { n - 1 } ^ { s } , \pmb { \theta } _ { n - 1 } ^ { t } , \pmb { x } _ { n } )\tag{3}
$$

where η is the learning rate, $\pmb { \theta } _ { n - 1 } ^ { t }$ represents the teacher weights at time $n - 1$ , and ${ \bf { x } } _ { n }$ is the test batch at time n. Due to the absence of ground-truth supervision, this update consist of a constructive adaptation direction and a harmful gradient noise. We can explicitly decompose the student's weight change into these two directions:

$$
\pmb { \theta } _ { n } ^ { s } = \pmb { \theta } _ { n - 1 } ^ { s } + \pmb { d } _ { n } + \pmb { \epsilon } _ { n }\tag{4}
$$

Here, $\pmb { d } _ { n } = - \eta \nabla _ { \pmb { \theta } ^ { s } } L _ { G T } ( \pmb { \theta } _ { n - 1 } ^ { s } , \pmb { y } _ { n } ^ { * } , \pmb { x } _ { n } )$ represents the ideal gradient step dictated by an oracle providing ground-truth labels $\boldsymbol { y } _ { n } ^ { * }$ , and $\epsilon _ { n }$ captures the residual gradient noise injected by incorrect pseudo-labels generated by the teacher. By unrolling the recurrence relation over time $n$ in the EMA teacher update (Eq. 1), we can express the teacher's current weights as a geometrically weighted sum of the source model weights $( \pmb \theta _ { 0 } )$ and all past student states:

$$
\pmb { \theta } _ { n } ^ { t } = \beta ^ { n } \pmb { \theta } _ { 0 } + ( 1 - \beta ) \sum _ { i = 1 } ^ { n } \beta ^ { n - i } \pmb { \theta } _ { i } ^ { s }\tag{5}
$$

This unrolled equation reveals the critical flaw of using $\beta < 1$ for continuous adaptation. Over infinite time horizons (as $n \to \infty )$ , the $\beta ^ { n } \pmb { \theta } _ { 0 }$ term decays to zero, meaning the teacher completely loses its anchor to the robust, pre-trained source distribution. Furthermore, the teacher permanently absorbs the accumulated student noise $( \sum \epsilon _ { i } )$ . Because the student's loss function uses the teacher as a reference, a degraded teacher generates lower-quality pseudo-labels for the next batch. This, in turn, increases the magnitude of the noise $\epsilon _ { n + 1 }$ in the student's next update. This creates a destructive positive feedback loop, illustrating why EMA merely delays, rather than prevents, catastrophic error accumulation on long sequences. When we apply the IT strategy by setting $\beta = 1$ , the EMA recurrence simplifies entirely: $\pmb { \theta } _ { n } ^ { t } = \pmb { \theta } _ { 0 }$ . While it might seem that the student network would merely mimic the frozen teacher, it is instead optimized using consistency or contrastive losses across distinct, augmented views of the data. We attribute this effective learning signal to the forced alignment between the two networks, where one processes a heavily augmented view of the input while the other operates on a clean version. By freezing the teacher, we decouple the pseudo-label generator from the student's noisy updates. While the teacher's supervision is inherently noisy due to the domain shift, this noise distribution remains stationary and non-compounding. This prevents the destructive positive feedback loop of error accumulation. We experimentally verify the error accumulation preventing behavior in Sec. 6.2.

## 6 EXPERIMENTS

Following the insights from preliminary experiments in Sec. 4, we evaluate the intransigent teacher over longer scenarios and with various state-of-the-art methods. The efficacy of the proposed baseline is validated across various model architectures and hyperparameter selections. We assess the error accumulation and the feature representation strength in student models for both original baselines and IT-enhanced approaches. Further details on the experimental setup are provided in Appendix Sec. A.1.

Benchmarks. We adopt the popular corruption benchmarks from Wang et al. (2022); Niu et al. (2022); Döbler et al. (2022): CIFAR10-C (C10-C) and ImageNet-C (IN-C) (Hendrycks & Dietterich, 2019), using the standard corruption sequence at the highest severity. The adaptation on natural domain shifts is tested on DomainNet-126 (DN-126) (Peng et al., 2019) and ImageNet-R (IN-R) (Hendrycks et al., 2021). For DN-126, we use the real domain as the source data and conduct experiments on the remaining domains. Our goal is to focus on very long (L) adaptation sequences to evaluate the methods in terms of stability during long-time operation. For that reason, we repeat the standard test sequences 20 times, and additionally test on the CCC (Press et al., 2023) long sequence without repetition.

Baselines. We apply our modification to 4 teacher-student state-of-the-art frameworks: AdaContrast (Chen et al., 2022), CoTTA (Wang et al., 2022), RoTTA (Yuan et al., 2023), PETAL (Brahma & Rai, 2023), and report their performance with IT using (I-) prefix. We compare the results with the original methods and the following strategies that do not utilize classic teacher-student framework: TENT (Wang et al., 2021), EATA (Niu et al., 2022), SAR (Niu et al., 2023), RDUMB (Press et al., 2023), MEMO (Zhang et al., 2022), and PeTTA (Hoang et al., 2024). The nonadapted model is indicated as Source, and TestBN is a fixed model using batch normalization statistics from the current batch (Li et al., 2016; Schneider et al., 2020b) as commonly done in TTA (Wang et al., 2021; 2022; Niu et al., 2022).

Architectures. Consistent with previous works (Wang et al., 2022; Marsden et al., 2024a), we use WideResNet-28 (Zagoruyko & Komodakis, 2016) models with pre-trained weights from ROBUSTBENCH (Croce et al., 2021) for the main experiments on CIFAR10-C. Similarly, evaluation on ImageNet and DomainNet-126 employ the ResNet50 network with weights sourced from the same model zoo or those provided by Marsden et al. (2024a) for DomainNet-126. Additional experiments in Sec. 6.2 are carried out with ResNet-26GN (RN26GN) (Wu & He, 2018), ResNeXt-50 32x4d (RNXt-50) (Xie et al., 2017), ViT-B16 (Dosovitskiy et al., 2021), SwinViT-T (SViT-T) (Liu et al., 2021) and ConvNeXt tiny (CNXt tiny) (Liu et al., 2022) architectures. Weights for RN26GN are taken from Zhang et al. (2022), as in Marsden et al. (2024a). ToRCHVISION is used to initialize the rest of the mentioned models.

## 6.1 ON ADAPTATION OVER LONG SCENARIOS

The vulnerability of EMA teachers is revealed on extended test sequences. In long scenarios (see Tab.2), methods based on teacher-student framework often achieve lower performance compared to other baselines or even cause the model to collapse, which is especially apparent on ImageNet-C and CCC benchmarks.

Those shortcomings are amplified in the lower batch size setting, potentially caused by a more significant error accumulation due to difficulties in estimating batch statistics and a larger number of adaptation steps. While the goal of using EMA is to provide a more stable adjustment and more accurate pseudo-labels for adaptation, we find this to not hold true for the longer settings. This is in contrast to the evaluation on common sequence length (see Appendix Tab. A.4), where the EMA teacher performs well and the original methods don't cause model collapses in most cases, except for the more challenging lower batch size setting. Results show that the performance of state-of-the-art TTA methods is clearly test sequence-length dependent.

BATCH SIZE 64

Intransigent teachers are very effective at collapse prevention. The evaluated original teacher-student adaptation methods exhibit at least 1% decrease in model accuracy in 21 out of 40 cases for long sequences (Tab. 2, results of AdaContrast, CoTTA, RoTTA, and PETAL on five benchmarks and two batch sizes). When using an intransigent teacher, the collapse happens only three times for the small batch size. However, in these cases, performance is at least close to the source model and may still yield great improvements over the baseline - for instance, I-CoTTA achieves a 44.3% increase in accuracy on DomainNet-126 (L).

Intransigent teachers provide great reliability, out-ofthe-box. Keeping the teacher model intransigent on a batch size 64 allows for accuracy improvements of 12.2 (AdaContrast), 2.4 (CoTTA), and 8.0 (RoTTA) percentage points on average. It is even more effective on a smaller batch size, where the respective improvements are 22.8, 25.2, and 7.4. Note that applying this strategy did not require any hyperparameter tuning nor additional parameters. In fact, it removes the need to set β.

Teacher intransigence and the plasticity bottleneck. While the intransigent teacher provides a robust baseline, it prioritizes stability over plasticity. This trade-off limits adaptation to rapid distribution shifts and prevents the student from exploring parameter spaces distant from the source model. This limitation is most evident when using CoTTA on ImageNet-C (L) and ImageNet-R (L) with a batch size of 64 (Tab. 2). In these scenarios, the use of IT reduces accuracy by 17.4 and 11.0 percentage points, respectively. We attribute this difference to the high baseline performance of CoTTA in these settings, as it outperforms all other tested methods. Here, IT over-regularizes the student and hinders its adaptation capabilities. While an EMA teacher may be preferable when hyperparameters can be precisely tuned, IT remains effective as it still outperforms the source model. Crucially, IT prevents long-term model collapse. Negative flip rate analysis (Sec. 6.2) shows that for CoTTA, the number of samples correctly predicted by the source but incorrectly by the adapted model increases monotonically over repetitions. In contrast, the IT negative flip rate remains constant. This suggests that long enough sequences could eventually cause model collapse, similarly to the 100 repetitions experiment in Appendix Figure A.4.

Table 2: Classification accuracy [%] for long (L) scenarios. Superscript indicates improvements over the baseline. Bold indicates best performing method. Gray color indicates model collapse – performance worse than the nonadapting model (Source). Results averaged from 3 random seeds. \* indicates approximated result, details in Appendix.
<table><tr><td>Method</td><td>C10-C (L)</td><td>IN-C (L)</td><td>IN-R (L)</td><td>DN-126 (L)</td><td>CCC</td><td>Avg.</td></tr><tr><td>Source</td><td>56.5</td><td>18.0</td><td>36.2</td><td>54.7</td><td>16.8</td><td>36.4</td></tr><tr><td>MEMO</td><td>65.6</td><td>25.0</td><td>40.9</td><td>53.2</td><td>19.3*</td><td>40.8</td></tr><tr><td colspan="7">BATCH SIZE 10</td></tr><tr><td>TestBN</td><td>75.1</td><td>26.9</td><td>36.2</td><td>49.6</td><td>22.5</td><td>42.1</td></tr><tr><td>TENT</td><td>39.0</td><td>4.7</td><td>17.4</td><td>10.9</td><td>0.7</td><td>14.5</td></tr><tr><td>EATA</td><td>73.6</td><td>36.4</td><td>44.0</td><td>54.0</td><td>29.7</td><td>47.6</td></tr><tr><td>SAR</td><td>75.2</td><td>30.6</td><td>43.6</td><td>50.8</td><td>20.3</td><td>44.1</td></tr><tr><td>RDUMB</td><td>76.8</td><td>34.3</td><td>40.1</td><td>51.5</td><td>28.1</td><td>46.2</td></tr><tr><td>PeTTA</td><td>83.8</td><td>37.8</td><td>40.5</td><td>55.8</td><td>13.7</td><td>46.3</td></tr><tr><td>AdaCont.</td><td>72.1</td><td>2.3</td><td>8.0</td><td>47.0</td><td>0.4</td><td>26.0</td></tr><tr><td>I-AdaCont.</td><td>84.1+12.0</td><td>39.5+37.2</td><td>35.3+27.3</td><td>63.2+16.2</td><td>21.8+21.4</td><td>48.8+22.8</td></tr><tr><td>CoTTA</td><td>23.8</td><td>3.8</td><td>33.7</td><td>6.0</td><td>17.3</td><td>16.9</td></tr><tr><td>I-CoTTA</td><td>69.7+45.9</td><td>27.6+23.8</td><td>37.4+3.7</td><td>50.3+44.3</td><td>25.7+8.4</td><td>42.1+25.2</td></tr><tr><td>RoTTA</td><td>82.5</td><td>24.4</td><td>43.0</td><td>45.6</td><td></td><td></td></tr><tr><td></td><td>79.0-3.5</td><td>33.6+9.2</td><td>39.9-3.1</td><td>57.8+12.2</td><td>1.1 23.4+22.3</td><td>39.3</td></tr><tr><td>I-RoTTA</td><td></td><td></td><td></td><td></td><td></td><td>46.7+7.4</td></tr><tr><td>PETAL</td><td>68.0</td><td>3.9</td><td>37.9</td><td>47.3</td><td>0.5</td><td>31.5</td></tr><tr><td>I-PETAL</td><td>74.2+6.2</td><td>28.7+24.8</td><td>36.5-1.4</td><td>50.7+3.4</td><td>13.9+13.4</td><td>40.8+9.3</td></tr></table>

<table><tr><td colspan="7"></td></tr><tr><td>TestBN</td><td>79.1</td><td>31.4</td><td>39.6</td><td>54.4</td><td>27.3</td><td>46.4</td></tr><tr><td>TENT</td><td>20.1</td><td>11.1</td><td>36.4</td><td>18.4</td><td>1.2</td><td>17.4</td></tr><tr><td>EATA</td><td>61.6</td><td>43.3</td><td>49.2</td><td>61.9</td><td>36.3</td><td>50.5</td></tr><tr><td>SAR</td><td>79.2</td><td>39.9</td><td>47.3</td><td>59.2</td><td>22.3</td><td>49.6</td></tr><tr><td>RDUMB</td><td>81.1</td><td>41.7</td><td>47.5</td><td>59.0</td><td>37.0</td><td>53.3</td></tr><tr><td>PeTTA</td><td>83.7</td><td>36.9</td><td>42.0</td><td>57.2</td><td>13.4</td><td>46.6</td></tr><tr><td>AdaCont.</td><td>81.8</td><td>18.8</td><td>26.5</td><td>61.7</td><td>2.4</td><td>38.2</td></tr><tr><td>I-AdaCont.</td><td>85.4+3.6</td><td>40.4+21.6</td><td>38.1+11.6</td><td>64.4+2.7</td><td>23.6+21.2</td><td>50.4+12.2</td></tr><tr><td>CoTTA</td><td>56.0</td><td>52.8</td><td>50.5</td><td>45.6</td><td>8.3</td><td>42.7</td></tr><tr><td>I-CoTTA</td><td>68.3+12.3</td><td>35.4-17.4</td><td>39.5-11.0</td><td>56.0+10.4</td><td>26.3+18.0</td><td>45.1+2.4</td></tr><tr><td>RoTTA</td><td>82.3</td><td>13.2</td><td>43.4</td><td>50.3</td><td>1.1</td><td>38.1</td></tr><tr><td>I-RoTTA</td><td>79.7-2.6</td><td>32.9+19.7</td><td>39.7-3.7</td><td>56.6+6.3</td><td>21.7+20.6</td><td>46.1+8.0</td></tr><tr><td>PETAL</td><td>56.3</td><td>36.8</td><td>40.7</td><td>58.6</td><td>13.1</td><td>41.1</td></tr><tr><td>I-PETAL</td><td>78.8+22.5</td><td>32.34.5</td><td>39.8-0.9</td><td>55.4-3.2</td><td>16.3+3.2</td><td>44.5+3.4</td></tr></table>

Intransigent teacher provides competitive stability. PeTTA (Hoang et al., 2024), a TTA method focused on longterm adaptation stability, employs a complex suite of techniques, including a class-balanced memory bank, robust batch normalization, weight- and prediction-based regularization from the source model, an adaptive EMA factor to control teacher updates, and the use of source feature statistics. Despite this complexity, our baseline methods, augmented with an intransigent teacher, achieve competitive or superior performance (Tab.2). Specifically, I-AdaContrast outperforms PeTTA by 1.7 percentage points on CIFAR10-C (L) (batch size 64), while I-RoTTA leads by 2 percentage points on ImageNet-C (L) (batch size 10). On average, I-AdaContrast improves accuracy over PeTTA by 3.8 percentage points (50.4% vs. 46.6%). These results demonstrate that preventing error accumulation in the teacher model is a highly effective, yet simple, strategy for ensuring long-term stability.

## 6.2 DETAILED ANALYSIS

Intransigent teachers work across different architectures. We verify the architectural universality of IT on CIFAR10-C (L) and ImageNet-C (L) using various network families (e.g., ViT, ConvNeXt). It should be noted that the RN26GN, ViT-B16, SViT-T, and CNX-tiny architectures do not include batch normalization layers, so IT's performance is not impacted by statistics recalibration. Because default learning rates are architecture-specific and difficult to set without target labels in TTA, we explore an upper-bound scenario where the learning rate is tuned via an Oracle for both the baseline and IT-augmented methods (Tab. 3). This controlled comparison isolates architectural robustness by providing an optimal hyperparameter setting for less stability-oriented TTA methods. Under these conditions, IT consistently prevents catastrophic collapse and either improves upon or maintains competitive baseline accuracy across the tested architectures. The advantages of IT become even more pronounced in a more realistic evaluation setup, where the learning rate is tuned exclusively for the original methods on standard-length benchmarks and directly reused for IT-augmented versions (see Tab. A.10 in the Appendix). This demonstrates IT's robustness to suboptimal, baselinespecific hyperparameters. While IT-augmented CoTTA and PETAL underperform their base counterparts specifically on ResNeXt50, both still exceed the source model accuracy, resulting in successful adaptation. We attribute this occasional underperformance to the plasticity bottleneck, discussed in Sec. 6.1. Further ablations in the Appendix confirm that IT maintains or improves robustness even when the source model's initial quality is artificially degraded with noise (Sec. B.3).

Table 3: Classification accuracy [%] with different neural network architectures. Superscript indicates improvements over the baseline. The learning rate is adjusted using an Oracle on long (L) benchmarks.
<table><tr><td rowspan="2"></td><td rowspan="2">C10-C (L) RN26GN</td><td colspan="4">IN-C (L)</td></tr><tr><td>RNXt-50</td><td>ViT-B16</td><td>SViT-T</td><td>CNXt tiny</td></tr><tr><td>Source</td><td>67.3</td><td>21.1</td><td>39.8</td><td>28.3</td><td>29.1</td></tr><tr><td>AdaCont.</td><td>75.7</td><td>38.8</td><td>41.5</td><td>30.6</td><td>33.4</td></tr><tr><td>I-AdaCont.</td><td>79.6+3.9</td><td>42.7+3.9</td><td>43.5+2.0</td><td>30.9+0.3</td><td>32.5-0.9</td></tr><tr><td>CoTTA</td><td>57.0</td><td>42.1</td><td>41.7</td><td>28.4</td><td>29.1</td></tr><tr><td>I-CoTTA</td><td>67.3+10.3</td><td>38.3-3.8</td><td>40.7-1.0</td><td>28.9+0.5</td><td>30.5+1.4</td></tr><tr><td>RoTTA</td><td>70.2</td><td>35.6</td><td>40.6</td><td>28.8</td><td>29.0</td></tr><tr><td>I-RoTTA</td><td>72.5+2.3</td><td>36.2+0.6</td><td>42.9+2.3</td><td>28.9+0.1</td><td>29.7+0.7</td></tr><tr><td>PETAL</td><td>24.8</td><td>40.6</td><td>42.8</td><td>23.6</td><td>26.5</td></tr><tr><td>I-PETAL</td><td>60.8+36.0</td><td>35.4-5.2</td><td>41.1-1.7</td><td>27.0+3.4</td><td>28.1+1.6</td></tr></table>

Table 4: Classification accuracy [%] for long (L) scenarios on CIFAR10-C and ImageNet-C with 1×, 10×, 50× learning rate (LR) scaling. Superscript indicates improvements over the baseline.
<table><tr><td></td><td colspan="3">C10-C (L)</td><td colspan="3">IN-C (L)</td></tr><tr><td></td><td>1×LR</td><td>10×LR</td><td>50×LR</td><td>1×LR</td><td>10×LR</td><td>50×LR</td></tr><tr><td>Source AdaCont.</td><td></td><td>56.5 83.6</td><td></td><td></td><td>18.0 19.6</td><td>12.4</td></tr><tr><td>I-AdaCont.</td><td>81.8 85.4+3.6</td><td>86.1+2.5</td><td>80.4 85.2+4.8</td><td>18.8 40.4+21.6</td><td>36.3+16.7</td><td>32.7+20.3</td></tr><tr><td>CoTTA</td><td>56.0</td><td>10.3</td><td>10.1</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>52.8</td><td>4.1</td><td>0.1</td></tr><tr><td>I-CoTTA</td><td>68.3+12.3</td><td>51.9+41.6</td><td>41.4+31.3</td><td>35.4-17.4</td><td>25.7+21.6</td><td>17.3+17.2</td></tr><tr><td>RoTTA</td><td>82.3</td><td>80.6</td><td>28.9</td><td>13.2</td><td>0.2</td><td>0.1</td></tr><tr><td>I-RoTTA</td><td>79.7-2.6</td><td>77.9-2.7</td><td>70.6+41.7</td><td>32.9+19.7</td><td>4.1+3.9</td><td>2.1+2.0</td></tr><tr><td>PETAL</td><td>56.3</td><td>15.9</td><td>18.0</td><td>36.8</td><td>0.4</td><td>0.1</td></tr><tr><td>I-PETAL</td><td> $7 8 . 8 ^ { + 2 2 . 5 }$ </td><td> $5 9 . 7 ^ { + 4 3 . 8 }$ </td><td> $3 5 . 8 ^ { + 1 7 . 8 }$ </td><td> $3 2 . 3 ^ { - 4 . 5 }$ </td><td> $3 6 . 2 ^ { + 3 5 . 8 }$ </td><td>8.1+8.0</td></tr></table>

On robustness to learning rate selection. In Table 4, we verify the learning rate sensitivity. Default AdaContrast shows robustness when 10× learning rate is used, but collapses with a 50× learning rate on ImageNet-C. IT allows increased robustness in all settings and achieves a non-collapsed solution, even for the highest learning rate. CoTTA lacks learning rate robustness, which is mitigated by our IT extension. Our strategy improves RoTTA's performance on CIFAR10-C. However, it rendered our approach almost ineffective on ImageNet-C. Overall, IT does not fully prevent the collapse when significantly altering the learning rate, but seems to promote a better-performing parameter space. Further analyses of hyperparameter tuning complexity (using Oracle and Transfer setups) are detailed in Appendix Sec. B.4, and we show that our observations extend to the semantic segmentation task in Appendix Sec. B.2.

Effects varying temporal correlation. Figure 5 illustrates the impact of varying degrees of class temporal correlation (Gong et al., 2022) on methods enhanced with IT. The analysis reveals that as temporal correlation increases, IT-enhanced methods consistently outperform their original versions, demonstrating improved robustness in these scenarios, and even when the original method achieved better results on uniform class distribution (RoTTA). However, it should be noted that while IT-enhanced methods show improved performance, they are also negatively impacted by the effects of correlation. The degree of improvement varies across different experimental settings.

![](images/c0ff0ad23b2af8e14000fc7f4662bbe6f6928fcc9336a2d522e2d770801ff350.jpg)  
Figure 5: Classification accuracy on CIFAR10-C (L). Samples are sorted by class for different levels of correlation, by varying the Dirichlet concentration parameter δ.

![](images/afb91d5c769e89665a90e4f2ec9795d4925e6ec5509903b87c762e5a30f194cf.jpg)

![](images/fa44db5398685acd1ba876e76f131a0d7ca85d8691ac2149d263ba374c06ed2a.jpg)  
Figure 6: Accuracy difference between student and intransigent teacher in percentage points, with default learning rate values (left) and after parameter search (right).

![](images/6185b5e9f76d81674c173031e28f2d107abde179322641a35f4861c5247c4802.jpg)

![](images/c48c6535743490511155abdd5ac4338dd0136f46322ac033ef908ed2718bfd0c.jpg)  
Figure 7: Per-batch accuracy on repeated ImageNet-C (20 loops) using ResNet50. AdaContrast (left) and RoTTA (right) are compared with an EMA teacher (ET, orange) and our intransigent teacher (IT, blue), both for teacher (solid) and student (dashed). The EMA teacher visibly suffers from error accumulation as the sequence gets longer. After the 3rd loop, IT manages to avoid model collapse and, surprisingly, the student performs significantly better.

![](images/2322de151129ca6cc88936a5babdd53b7924dd077c0ab33b0adaf0d99392f7ab.jpg)

![](images/c2855785a242ebf69f66dae1a5070ee4eff13ccd61d1b8a5ede7337f17af5c43.jpg)

![](images/2b7046b3f5d9a798f734fa52b89b4c056f27655edde693e74288911ca0044780.jpg)

![](images/d507707472f704ab8ea45127eb432a83b3e5d205646ab637933f026d10f7a437.jpg)  
Figure 8: Per-batch Negative Flips Rate (NFR) of the student model averaged over AdaContrast, CoTTA, RoTTA and PETAL methods (ET) on each dataset, along with their IT-enhanced versions (IT). The shaded regions represents ± standard deviation illustrating the variability across methods

Are students better than intransigent teachers? When paired with intransigent teachers, student models often outperform their teachers, whereas EMA keeps teacher-student performance tightly coupled (Fig. 7). As shown in Figure 6, IT enables the student to exceed teacher average accuracy in nearly all configurations, with only two exceptions: I-AdaContrast on ImageNet-R (L) and I-CoTTA on CIFAR10-C (L). In those cases, IT-induced stability limits performance decline but is insufficient to prevent degradation entirely. However, when the learning rate is optimized via an Oracle, students outperform teachers in all cases (Fig. 6, right). Per-batch accuracy plots for remaining method–dataset combinations are presented in the Appendix (Figs. A.9 and A.10).

Evaluating the error accumulation. We hypothesize that the IT-enhanced methods' success in longer scenarios stems from their ability to mitigate error accumulation. We attempt to compare the degree of accumulated error between methods and employ the Negative Flip Rate (NFR) as defined in Yan et al. (2021). The NFR quantifies the fraction of samples initially predicted correctly by the source model but incorrectly by the adapted model. This metric captures degradation of the initial model's knowledge, the direct effect of error accumulation. In Figure 8, we show per-batch NFR of the student model averaged over AdaContrast, CoTTA, RoTTA, and PETAL methods for each benchmark. For specific examples, see the Appendix (Fig. A.11). The NFR for IT-enhanced methods remains consistently stable, showing no increasing trend across any benchmark. It suggests that the fixed weights of the teacher prevent the student from accumulating errors, thereby preserving the initial knowledge. In contrast, the original methods exhibit increasing NFR in most experiments, indicating that the adaptation process degrades the model's knowledge, which suggests error accumulation. In some experiments, the original methods exhibit a stable, low NFR, indicating reliable adaptation that correctly improves the model, which also correlates with higher accuracy in Table 2. Complementary to these findings, we observe that IT also prevents the internal degradation of the model's feature space. As detailed in Appendix Sec. B.1, tracking the effective rank of the student's representations reveals that baseline collapses correlate with a severe decrease in feature diversity. By anchoring the adaptation process, IT consistently maintains high effective rank and prevents dimensional collapse. Overall, IT-enhanced methods offer more consistent behavior and robustness across benchmarks.

## 7 CONCLUSIONS

In this work, we explore the well-known strategy of using an exponential moving average teacher for TTA. Although effective for common sequence lengths, this strategy fails to mitigate long-term error accumulation, which can lead to model collapse. To address the resulting plasticity-stability trade-off, we propose an intransigent teacher strategy compatible with current state-of-the-art methods. Our approach often prevents model collapse across diverse experimental setups and improves robustness to hyperparameter choice without requiring manual tuning. Empirical results demonstrate that the intransigent teacher halts the accumulation of errors and helps to maintain high feature diversity These findings suggest that TTA evaluation must encompass varied sequence lengths to ensure reliability. Our method provides a robust, simple baseline for future long-term adaptation research.

Discussion and limitations. While the intransigent teacher strategy proved highly effective across various scenarios, it can result in slower adaptation in certain cases due to the teacher's fixed behavior. To address this issue, we also present results where the teacher model was allowed to adapt for a fixed number of initial steps (Appendix Sec. B.8). Further exploration of this approach is left for future work. Furthermore, we note that by adding the intransigent teacher to existing methods, we inherit their limitations.

## ACKNOWLEDGMENTS

This research was funded in whole or in part by National Science Centre, Poland, grants no 2024/53/N/ST6/03156 and 2023/51/D/ST6/02846. We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grants no. PLG/2025/018634, PLG/2025/018644, and PLG/2026/019363. This research was funded in whole or in part by the Austrian Science Fund (FWF) 10.55776/COE12. Bartłomiej Twardowski acknowledges the grant RYC2021-032765-I.

## REFERENCES

Motasem Alfarra, Hani Itani, Alejandro Pardo, Shyma Alhuwaider, Merey Ramazanova, Juan C Pérez, Zhipeng Cai, Matthias Müller, and Bernard Ġhanem. Evaluation of test-time adaptation under computational time constraints. In ICML, 2024.

Malik Boudiaf, Romain Müller, Ismail Ben Ayed, and Luca Bertinetto. Parameter-free online test-time adaptation. In CVPR, pp. 8334–8343, 2022.

Dhanajit Brahma and Piyush Rai. A probabilistic framework for lifelong test-time adaptation. In CVPR, pp. 3582– 3591, 2023.

Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930, 2020.

Arslan Chaudhry, Puneet K. Dokania, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

Dian Chen, Dequan Wang, Trevor Darrell, and Sayna Ebrahimi. Contrastive test-time adaptation. In CVPR, pp. 295–305, 2022.

Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence, 40(4):834–848, 2017.

Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021. URL https://openreview.net/forum?id=SSKZPJCt7B.

Sebastian Cygert, Damian Sójka, Tomasz Trzciński, and Bartłomiej Twardowski. Realistic evaluation of test-time adaptation: Unsupervised model selection. In Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP 2026) - Volume 2, pp. 75–86. INSTICC, SciTePress, 2026. ISBN 978-989-758- 744-3. doi:10.5220/0014320100004084.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pp. 248–255, 2009.

Mario Döbler, Robert A. Marsden, and Bin Yang. Robust mean teacher for continual and gradual test-time adaptation. CoRR, abs/2211.13081, 2022.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

Tommaso Furlanello, Zachary Chase Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar. Born-again neural networks. In ICML, 2018

Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In ICLR, 2019.

Taesik Gong, Jongheon Jeong, Taewon Kim, Yewon Kim, Jinwoo Shin, and Sung-Ju Lee. NOTE: robust continual test-time adaptation against temporal correlation. In NeurIPS, 2022.

Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, and Sung-Ju Lee. SoTTA: Robust test-time adaptation on noisy data streams. In NeurIPS, 2023.

Dipam Goswami, Yuyang Liu, Bartłomiej Twardowski, and Joost van de Weijer. Fecam: Exploiting the heterogeneity of class distributions in exemplar-free continual learning. NeurIPS, 36, 2024.

Sachin Goyal, Mingjie Sun, Aditi Raghunathan, and J Zico Kolter. Test time adaptation via conjugate pseudo-labels. NeurIPS, 35:6204–6218, 2022.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, et al. Bootstrap your own latent - a new approach to self-supervised learning. In NeurIPS, 2020.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, pp. 9729–9738, 2020.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. ICLR, 2019.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, pp. 8340–8349, 2021.

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. ArXiv, 2015.

Trung Hieu Hoang, MinhDuc Vo, and Minh Do. Persistent test-time adaptation in recurring testing scenarios. Advances in Neural Information Processing Systems, 37:123402–123442, 2024.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences (PNAS), 2017.

Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran S. Haque, Sara M. Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A benchmark of in-the-wild distribution shifts. In ICML, 2021.

Alex Krizhevsky. Learning multiple layers of features from tiny images. pp. 32–33, 2009. URL https : / /www . cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf.

Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. In ICLR, 2017.

Shuang Li, Longhui Yuan, Binhui Xie, and Tao Yang. Generalized robust test-time adaptation in continuous dynamic scenarios. arXiv preprint arXiv:2310.04714, 2023.

Yanghao Li, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. Revisiting batch normalization for practical domain adaptation. ArXiv, abs/1603.04779, 2016.

Sheng Liu, Jonathan Niles-Weed, Narges Razavian, and Carlos Fernandez-Granda. Early-learning regularization prevents memorization of noisy labels. NeurIPS, 33, 2020.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, pp. 10012–10022, 2021.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. CVPR, 2022.

Chunwei Ma, Zhanghexuan Ji, Ziyun Huang, Yan Shen, Mingchen Gao, and Jinhui Xu. Progressive voronoi diagram subdivision enables accurate data-free class-incremental learning. In The Eleventh International Conference on Learning Representations, 2023.

Sarthak Kumar Maharana, Shambhavi Mishra, Yunbei Zhang, Shuaicheng Niu, Taki Hasan Rafi, Jihun Hamm, Marco Pedersoli, Jose Dolz, and Yunhui Guo. Continual test-time adaptation: A comprehensive survey. 2026

TorchVision maintainers and contributors. Torchvision: Pytorch's computer vision library. https : //github. com/pytorch/vision,2016.

Robert A. Marsden, Mario Döbler, and Bin Yang. Universal test-time adaptation through weight ensembling, diversity weighting, and prior correction. In WACV, 2024a.

Robert A Marsden, Mario Döbler, and Bin Yang. Introducing intermediate domains for effective self-training during test-time. In IJCNN, pp. 1–10. IEEE, 2024b.

Marc Masana, Xialei Liu, Bartłomiej Twardowski, Mikel Menta, Andrew D Bagdanov, and Joost van de Weijer. Classincremental learning: survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022

Martial Mermillod, Aurélia Bugaiska, and Patrick Bonin. The stability-plasticity dilemma: Investigating the continuum from catastrophic forgetting to age-limited learning effects. Frontiers in psychology, 4:54654, 2013.

Duc Tam Nguyen, Chaithanya Kumar Mummadi, Thi Phuong Nhung Ngo, Thi Hoai Phuong Nguyen, Laura Beggel, and Thomas Brox. Self: Learning to filter noisy labels with self-ensembling. In ICLR, 2020.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In ICML, volume 162, pp. 16888–16905, 2022.

Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. In ICLR, 2023.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

Aristeidis Panos, Yuriko Kobe, Daniel Olmeda Reino, Rahaf Aljundi, and Richard E Turner. First session adaptation: A strong replay-free baseline for class-incremental learning. In ICCV, pp. 18820–18830, 2023.

Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, et al. Moment matching for multi-source domain adaptation. In ICCV, pp. 1406–1415, 2019.

Ori Press, Steffen Schneider, Matthias Kümmerer, and Matthias Bethge. Rdumb: A simple approach that questions our progress in continual test-time adaptation. In Advances in Neural Information Processing Systems 36:, 2023.

Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pp. 606–610. IEEE, 2007.

Evgenia. Rusak, Steffen Schneider, George Pachitariu, Luisa Eck, Peter Gehler, Oliver Bringmann, Wieland Brendel, and Matthias Bethge. If your data distribution shifts, use self-learning. Transactions of Machine Learning Research 2022. URL https://openreview.net/forum?id=vqRzLv6POg.

Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias Bethge. Improving robustness against common corruptions by covariate shift adaptation. Advances in neural information processing systems, 33:11539–11551, 2020a.

Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias Bethge. Improving robustness against common corruptions by covariate shift adaptation. NeurIPS, 33:11539–11551, 2020b.

Ziqi Shi, Fan Lyu, Ye Liu, Fanhua Shang, Fuyuan Hu, Wei Feng, Zhang Zhang, and Liang Wang. Controllable continual test-time adaptation. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pp. 1–6. IEEE, 2025.

Damian Sójka, Bartłomiej Twardowski, Tomasz Trzciński, and Sebastian Cygert. Ar-tta: A simple method for realworld continual test-time adaptation. In BMVC, 2023.

Yongyi Su, Xun Xu, and Kui Jia. Towards real-world test-time adaptation: Tri-net self-training with balanced normalization. In AAAI, volume 38, pp. 15126–15135, 2024.

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-time training with selfsupervision for generalization under distribution shifts. In ICML, volume 119, pp. 9229–9248, 2020.

Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. NeurIPS, 30, 2017.

Benedikt Tscheschner, Eduardo Veas, and Marc Masana. Incremental learning with repetition via pseudo-feature projection. In Proceedings of the 28th Computer Vision Winter Workshop, CVWW 2025, February 2025. doi: 10.3217/978-3-99161-022-9-004. 28th Computer Vision Winter Workshop, CVWW 2025 ; Conference date: 12- 02-2025 Through 14-02-2025.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno A. Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. In ICLR, 2021.

Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In CVPR, pp. 7191– 7201, 2022.

Yuxin Wu and Kaiming He. Group normalization. In ECCV, volume 11217, pp. 3–19, 2018.

Zehao Xiao and Cees GM Snoek. Beyond model adaptation at test time: A survey. arXiv preprint arXiv:2411.03687, 2024.

Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In CVPR, pp. 1492–1500, 2017.

Sijie Yan, Yuanjun Xiong, Kaustav Kundu, Shuo Yang, Siqi Deng, Meng Wang, Wei Xia, and Stefano Soatto. Positivecongruent training: Towards regression-free model updates. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14299–14308, 2021.

Longhui Yuan, Binhui Xie, and Shuang Li. Robust test-time adaptation in dynamic scenarios. In CVPR, pp. 15922– 15932, 2023.

Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In BMVC, 2016.

Marvin Zhang, Sergey Levine, and Chelsea Finn. MEMO: test time robustness via adaptation and augmentation. In NeurIPS, 2022.

Hao Zhao, Yuejiang Liu, Alexandre Alahi, and Tao Lin. On pitfalls of test-time adaptation. In ICML, 2023.

Xingzhi Zhou, Zhiliang Tian, Ka Chun Cheung, Simon See, and Nevin L. Zhang. Resilient practical test-time adaptation: Soft batch normalization alignment and entropy-driven memory bank. ArXiv, 2024.

Xingzhi Zhou, Boyang Zhang, Zhiliang Tian, Yibo Zhang, Xin Niu, Ka Chun Cheung, Simon See, and Nevin L. Zhang. Resilient test-time adaptation by mitigating batch-normalization overfitting. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5, 2025. doi: 10.1109/ ICASSP49660.2025.10888647.

## APPENDIX

This appendix provides comprehensive evidence and analysis to support the findings presented in the main paper. Sec. A details the experimental framework, including the setup (Sec. A.1), baseline descriptions and implementations (Sec. A.2, A.3), and specific details for the intransigence experiments (Sec. A.4) and the MEMO results on CCC benchmark (Sec. A.5). It also covers the negative flip rate (NFR) calculation (Sec. A.6) and computational requirements (Sec. A.7).

Sec. B presents several additional experiments to better understand the described problem and the limitations of the proposed technique. These include evaluation of feature representation strength (Sec. B.1), semantic segmentation experiment (Sec. B.2), analysis of the dependency on source model performance (Sec. B.3), experiment with tuning the learning rate value for long scenarios (Sec. B.4), wall-clock time comparison (Sec. B.5), discussion on single image adaptation methods (Sec. B.6), results on common-length benchmarks (Sec. B.7), an investigation of the potential of the adaptive β value for teacher momentum (Sec. B.8), an analysis of student-only adaptation performance (Sec. B.9), an extended experiment showing the effects of intransigence for over 100 loops with the CoTTA method (Sec. B.10), discussion on model reset mechanism (Sec. B.11), the novelty clarification with regard to RDumb (Sec. B.12), augmentation count ablation for CoTTA and PETAL methods B.13, decomposition of the IT student's gain into BN statistics recalibration and teacher distillation B.14, and experiments with fixed batch normalization statisticsB.15 We also include the remaining results of the experiments mentioned in the main paper (Sec. C).

## A ADDITIONAL DETAILS

## A.1 EXPERIMENTAL SETUP

Benchmarks. Our experiments include popular corruption benchmarks CIFAR10-C (C10-C) and ImageNet-C (IN-C) (Hendrycks & Dietterich, 2019), training on clean CIFAR10 (Krizhevsky, 2009) or ImageNet (Deng et al., 2009) images and testing on corrupted images with 15 corruption types at 5 severity levels. We adopt the setup from Wang et al. (2022); Niu et al. (2022); Döbler et al. (2022), using the standard corruption sequence at the highest severity. The adaptation on natural domain shifts is tested utilizing DomainNet-126 (DN-126) (Peng et al., 2019) and ImageNet-R (IN-R) (Hendrycks et al., 2021) datasets. For DN-126, we use the real domain as the source data and conduct experiments on the remaining domains. The clean ImageNet dataset serves as the source data for experiments on ImageNet-R. Our goal is to focus on very long adaptation sequences to evaluate the methods in terms of stability during long-time operation. For that reason, we repeat the standard test sequences of benchmarks described above 20 times. We call this setup a Long (L) scenario. Additionally, we take advantage of the CCC (Press et al., 2023) benchmark, which has a very high sequence length without repetitions. We use a CCC-Medium sequence with a 1k transition speed, which consists of 7,500,000 images. For experiments with semantic segmentation, we utilize the CarlaTTA dataset (Marsden et al., 2024b). The source data consists of images from the clear domain, i.e., images captured during the day in clear weather. For test data, we use the day2night sequence, which goes from the clear domain, gradually transitioning to night. Consistent with image classification experiments, we used the same approach to create the Long (L) scenario.

Architectures. Consistent with previous works (Wang et al., 2022; Marsden et al., 2024a), we use WideResNet-28 (Zagoruyko & Komodakis, 2016) models with pre-trained weights from the RobustBench (Croce et al., 2021) model zoo for the main experiments on CIFAR10-C. Similarly, the tests on ImageNet-based benchmarks and DomainNet-126 employ the ResNet50 network with weights sourced from the same model zoo or those provided by Marsden et al. (2024a) for DomainNet-126. Additional experiments in Sec. 6.2 are carried out with ResNet-26GN (RN26GN) (Wu & He, 2018), ResNeXt-50 32x4d (RNXt-50) (Xie et al., 2017), ViT-B16 (Dosovitskiy et al., 2021), SwinViT-T (SViT-T) (Liu et al., 2021) and ConvNeXt tiny (CNXt tiny) (Liu et al., 2022) architectures. Weights for RN26GN are taken from Zhang et al. (2022), as in Marsden et al. (2024a). The Torchvision (maintainers & contributors, 2016) library is used to initialize the rest of the mentioned models. For semantic segmentation, following Marsden et al. (2024b), we utilize the DeepLab-V2 (Chen et al., 2017) architecture with ResNet-101 backbone. We use the pre-trained source checkpoint provided by Marsden et al. (2024b).

Implementation details. As a testbed for experiments, we adopt the framework from Marsden et al. (2024a) and use the parameters reported in the respective original papers. We use parameters from common sequence length settings while testing on the long (L) scenarios, unless otherwise stated. We utilize two batch sizes for image classification: a standard value of 64, as in multiple previous works (Marsden et al., 2024a; Niu et al., 2022; Press et al., 2023; Wang et al., 2021), and a lower one (equal to 10), resulting in a more challenging setup with more model updates. When running experiments on a smaller batch size, the learning rate is decreased accordingly. In the case of semantic segmentation, we follow the usual batch size of 1 (Marsden et al., 2024b). In-depth implementation details regarding the baselines are provided in Sec. A.3).

## A.2 DESCRIPTIONS OF BASELINES EXTENDED WITH THE INTRANSIGENT TEACHER

AdaContrast conducts weight updates using a three-part loss function: cross-entropy loss, diversity regularization, and contrastive loss. Pseudo-labels are refined by keeping a buffer of previous image features and their predictions. Refined predictions are based on the nearest neighbors of the current feature within the buffer. Statistics in batch normalization layers are updated with EMA.

CoTTA updates the student model by minimizing the cross-entropy consistency between the teacher and the student predictions. Depending on the prediction confidence, the pseudo-labels are the result of averaging predictions on multiple, differently augmented images. At each iteration, there is a small probability for each of the student's weights to be reset to the source pre-trained value. It calculates batch normalization statistics on a per-batch basis.

RoTTA keeps the class-balanced memory buffer of images and uses it to perform the optimization in constant intervals. The loss is weighted based on how long the sample has been stored. Cross-entropy-based consistency between the student and teacher models is utilized for a loss function. Batch normalization statistics are updated via EMA.

PETAL is similar to CoTTA, but it enhances the learning objective by the regularizer term based on a posterior distribution over a source model weights and a data-dependent prior. Moreover, it improves CoTTA's stochastic model reset scheme with the Fisher Information Matrix

## A.3 BASELINES IMPLEMENTATION DETAILS

The experiments are conducted by adapting the code repository of previous test-time adaptation works (Marsden et al., 2024a; Döbler et al., 2022), which provide the implementation of all evaluated state-of-the-art methods, except PETAL and PeTTA. We integrated its original implementation into this unified codebase ourselves. In terms of hyperparameters, we followed the typical implementation for test of batch size of 64.

TENT (Wang et al., 2021), EATA (Niu et al., 2022), SAR (Niu et al., 2023), and RDUMB (Press et al., 2023) use Adam optimizer with a learning rate of 0.001 for CIFAR10-C and SGD optimizer with a learning rate of 0.00025 for other benchmarks. AdaContrast (Chen et al., 2022) utilizes an SGD optimizer with a learning rate set to 0.0002 for all of the benchmarks. CoTTA (Wang et al., 2022) uses Adam optimizer with a learning rate of 0.001 for CIFAR10-C and SGD with a learning rate of 0.01 for the rest of the benchmarks. Adam optimizer with a learning rate set to 0.001 is used by RoTTA (Yuan et al., 2023) for all of the tested datasets. MEMO (Zhang et al., 2022) uses an SGD optimizer with a learning rate of 0.005 for CIFAR10-C and 0.00025 for other datasets. PETAL (Brahma & Rai, 2023) in the original paper uses Adam optimizer with a learning rate of 0.001 for CIFAR10-C and SGD with a learning rate of 0.01 for other datasets. However, since we often experienced poor performance using these values on long scenarios, we utilized 10 times lower learning rates. Due to PETAL's regularizer term becoming unstable at higher learning rates, which leads to model corruption, we excluded PETAL from the learning rate scaling experiments shown in Tab. 4. This decision avoided presenting potentially compromised results. PeTTA (Hoang et al., 2024) use Adam optimizer with learning rate of 0.001 for all of the benchmarks. The learning rate used in experiments with batch size set to 10 was adjusted accordingly by scaling it linearly.

CoTTA (Wang et al., 2022) and PETAL (Brahma & Rai, 2023) methods update the student network using a consistency loss between the student and teacher. If the prediction confidence of the source model is below a certain threshold, the teacher's predictions are averaged over 32 different augmentations of the image which adds 31 additional forward operations per batch. It creates a significant computation overhead and causes the methods to be significantly slower, compared to other state-of-the-art methods. It is especially problematic for long adaptation scenarios, which make the main part of our experiments. Our ablations indicate that using a single augmentation does not alter the results notably (Sec. B.13). Therefore, for the ease of experimentation, we reduce the number of augmentations to 1.

## A.4 DETAILS ON INTRANSIGENT TEACHER EXPERIMENT FROM SECTION 4.1

Our preliminary intransigent teacher experiment is conducted on ImageNet-C, a long ImageNet-C scenario (20× loop of standard ImageNet-C testing sequence), and CCC. We utilize the same model as for our main experiments on ImageNet-based benchmarks - ResNet50 with pre-trained weights from the RobustBench (Croce et al., 2021) model zoo. We use a single loss within the teacher-student framework for the model adaptation during test time – either Consistency from CoTTA or Contrastive from AdaContrast. Any other components of the mentioned state-of-theart methods are not included. Batch normalization statistics are recalculated for each batch. For both of the tested approaches, we use the SGD optimizer with a learning rate of 0.00025.

## A.5 DETAILS ON MEMO RESULTS ON CCC BENCHMARK FROM TAB. 2

The result is based on the first 623,000 images of the benchmark, providing an initial estimate of the method's accuracy. However, due to the benchmark's extensive size (7,500,000 images) and the method's requirement for a batch size of 1, we cannot complete the full experiment in a reasonable amount of time. We estimate that processing the entire dataset requires approximately 972 hours on a single NVIDIA GeForce RTX 4080 GPU. This substantial time requirement underscores the method's significant computational inefficiency.

## A.6 DETAILS ON NEGATIVE FLIP RATE

In Sec. 6.2, we attempt to compare the degree of accumulated error between methods and employ the Negative Flip Rate (NFR) as defined in Yan et al. (2021). The NFR quantifies the fraction of samples initially predicted correctly by the source model but incorrectly by the adapted model, calculated as:

$$
\mathrm { N F R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } ( \hat { y } _ { i } ^ { 0 } = y _ { i } , \hat { y } _ { i } \neq y _ { i } )\tag{6}
$$

where $y _ { i }$ is the ground truth label, $\hat { y } _ { i } ^ { 0 }$ is the source model's prediction, $\hat { y } _ { i }$ is the current model's prediction, and 1(·) is the indicator function. This metric captures degradation of the initial model's knowledge, the direct effect of error accumulation.

## A.7 COMPUTE DETAILS

All experiments are conducted on a single GPU. We utilize either NVIDIA A100 with 40GB of memory or NVIDIA GeForce RTX 4080 with 16GB of memory. The execution time of the experiment greatly varied and was dependent on the dataset, scenario (standard or long), tested method, and batch size. The fastest experiments take about 30 minutes, whereas the longest last up to 36 hours.

## B ADDITIONAL EXPERIMENTS

## B.1 EVALUATION OF FEATURE REPRESENTATION STRENGTH.

To gain additional insight into why the IT works effectively, we assess changes in the representation strength of features extracted from the model's encoder during the adaptation process.

We focus on measuring the diversity of the features and compute the effective rank (Roy & Vetterli, 2007) of the feature representations during the adaptation process. We collect a domain-balanced set of n (n = 150) randomly sampled images from the adaptation sequence of a given benchmark. During TTA, we extract features for this entire set of images using the encoder of the adapted model and construct a feature matrix A. We then compute its singular values $\sigma _ { 1 } , \cdots , \sigma _ { n }$ . Given the distribution of singular values, defined as $\begin{array} { r } { p _ { i } = \frac { \sigma _ { i } } { | \sigma | _ { 1 } } } \end{array}$ , we calculate the effective rank of feature representation matrix A as follows:

$$
E \mathrm { - r a n k } ( A ) = \exp ( - \sum _ { i = 1 } ^ { n } p _ { i } \log ( p _ { i } ) ) .\tag{7}
$$

We track the effective rank of the feature representations of the student model every 50 batches throughout the adaptation sequence. The representative results are presented in Fig. A.1.

We observe that a significant drop in effective rank (e.g., a two-fold decrease) correlates with a collapsed solution, where accuracy falls below that of the source model. This pattern is evident in four out of six collapse cases with a batch size of 64, as reported in Tab. 2: CoTTA and PETAL on C10-C (L), RoTTA on IN-C (L), and CoTTA on DN-126 (L). The use of IT helps maintain higher feature diversity, thereby preventing collapse. For the remaining collapse cases, the effective rank also exhibits a slight downward trend: AdaContrast on IN-R (L) and RoTTA on DN-126 (L).

![](images/cf0e4aff2d865f95bf8185420006ddd8d313b6db33d244154b33006757852edf.jpg)

![](images/cfc81ac79219b5abd8e10761c2673bd99a85a8d60e01d5d807978a87fe984ca9.jpg)

![](images/ac01c2b3fe199c7945e2ae53821043d212f169598c5f0ea4c2706b5af0afd4ee.jpg)  
(a) AdaContrast

![](images/989f04e2911e85566437f44f8286192f4d94fee5ffcf1fa95db76de84734c1b1.jpg)

![](images/d6b538e8ad5f29200b777058bba76f160677950ce2c4d0f292a243b49fdcbe1b.jpg)

![](images/0fffe8f41fc3a5a83a1bbca65095455130ee189453674bcea66d6e50245c9849.jpg)

![](images/058ae1bcb9251471e5756cef47f4d0e9ad78feb653c8660809eba8df1745ddc0.jpg)  
(b) CoTTA

![](images/fe0f0cabc71b85d4485d1b09b02bb8bbb21b9fa728976596f147ce3c21e457db.jpg)

![](images/98d9828c768030dd326446f870b8290a6ce95ca62d723e9db820c7f208607cf8.jpg)

![](images/6cb4087ba0d9d1337e687fcba180d1f71edc04c57c1f2af762aa67973e80deeb.jpg)

![](images/825de6f98e86d9e738aafb2cc8129261e3253c26a3f6f2bee283f1d7295631b9.jpg)

![](images/46c67761b273e1dd6a351ca1604f52874eda632506eee57d80eac9c3ef03d060.jpg)

(c) RoTTA  
![](images/0f22739f765784a12ccc6753fdfbae891903ef84cce633b1ef5066c56b11198d.jpg)

![](images/e0dfe58b7b64f32271ed84da6cc128be034cd455edcc600a738db0d15be2ab27.jpg)  
(d) PETAL

![](images/8ebf880e04bee3d318b029c8b7cf2885e85453fcb56877bf41324d3424eed68f.jpg)

![](images/fbc89baf92760c3169e5d16137a2da2334857ffd9c1dc84bd316f97ccbeb3e20.jpg)  
Figure A.1: Effective rank values of the features extracted from the student model during the TTA every 50 batches on each benchmark using AdaContrast (a), CoTTA (b), RoTTA (c) and PETAL (d) methods and IT-enhanced version.

## B.2 SEMANTIC SEGMENTATION

We evaluate a broader range of tasks and report the mean intersection-over-union (mIoU) results for the semantic segmentation experiment with different learning rate values across the entire long (L) scenario in Tab. A.1, with results for each loop presented separately in Fig. A.2.

We focus exclusively on the CoTTA method, as it is the only baseline developed specifically for this task by its authors. The results demonstrate that our initial observations extend to the semantic segmentation task: CoTTA, which utilizes EMA teacher, is highly sensitive to learning rate selection and degrades model performance over time for all chosen

values, including the default for the standard-length sequence (2.5e-4), except for the lowest value (where its plot line is occluded by the I-CoTTA plot). In contrast, I-CoTTA exhibits greater robustness to learning rate selection, and IT effectively mitigates the error accumulation issue.

![](images/f779a1e92015706a8814289f3a68439909acd375fcb088d8db376890395b9882.jpg)  
Figure A.2: Segmentation results (mIoU [%]) for each repeated day2night sequence on CarlaTTA (Marsden et al., 2024b). Orange and blue colors indicate CoTTA and I-CoTTA, respectively. The darker the color, the higher the learning rate value used to achieve the result. The black dashed line indicates the source model performance as a reference.

Table A.1: Average segmentation (mIoU [%]) on the long day2night sequence from CarlaTTA Marsden et al. (2024b) with different learning rate (LR) values. The superscript indicates the improvements over the baseline.
<table><tr><td>LR</td><td> $\mathbf { 2 . 5 e { - 2 } }$ </td><td> $\mathbf { 2 . 5 e { \cdot 3 } }$ </td><td> $\mathbf { 2 . 5 e { \cdot 4 } }$ </td><td> $\mathbf { 2 . 5 e { \cdot 5 } }$ </td><td> $\mathbf { 2 . 5 e { \cdot 6 } }$ </td></tr><tr><td>Source</td><td></td><td></td><td>58.4</td><td></td><td></td></tr><tr><td>CoTTA</td><td>4.9</td><td> $3 3 . 9$ </td><td>50.9</td><td> $5 8 . 0$ </td><td>61.5</td></tr><tr><td>I-CoTTA</td><td> $2 1 . 6 ^ { + 1 6 . 7 }$ </td><td> $5 4 . 3 ^ { + 2 0 . 4 }$ </td><td> $6 0 . 8 ^ { + 9 . 9 }$ </td><td> $6 1 . 3 ^ { + 3 . 3 }$ </td><td> ${ \bf 6 1 . 7 ^ { + 0 . 2 } }$ </td></tr></table>

![](images/2a5701857d2ae9a95f1492c0d96f106fb0e30f8f00859bb72fd54939ac006ddb.jpg)  
Figure A.3: The overall average accuracy on each of the tested benchmarks of the original methods and IT-enhanced versions using source model weights perturbed using Gaussian noise with increasing standard deviation σ. The $\sigma = 0$ indicates the performance of the model not perturbed by noise. The brown dashed line indicated the source model accuracy.

## B.3 DEPENDENCY ON SOURCE MODEL PERFORMANCE

Since the intransigent teacher strategy fixes the teacher model's weights, its performance may heavily depend on the quality of the source model and could fail if the model is poorly suited to the target domain. To test this hypothesis, we evaluate adaptation performance using source models of varying quality. We generate these models by taking our default model weights and perturbing them with Gaussian noise of varying standard deviation, σ. For each layer, the noise's standard deviation is scaled by the mean value of that layer's weights. The average accuracies across the entire benchmark for each σ are presented in Fig. A.3. We compare the accuracy trends of the original methods with those enhanced by the IT strategy. The results indicate that, in most cases, weaker source models lead to poorer performance in the original baselines. IT-enhanced methods either follow the same trend as the original baselines (e.g., I-AdaContrast and I-RoTTA on C10-C (L), I-PETAL and I-RoTTA on IN-R (L), I-AdaContrast on DN-126 (L)) or exhibit greater robustness to weaker initial training (e.g., I-CoTTA and I-PETAL on C10-C (L), I-CoTTA and I-PETAL on DN-126 (L)). In other words, IT either maintains the original method's sensitivity to source model quality or improves its robustness. We hypothesize that this behavior occurs because, even when the source model's predictions are less accurate, the fixed teacher model is prevented from further performance degradation and error accumulation due to the student's noisy adaptation, while still allowing the student model to update (via backpropagation) on the target data.

Our observations align with those from Zhao et al. (2023), where the authors show that, in general, the performance of TTA methods is highly dependent on the quality of the initial source model.

Table A.2: Classification accuracy [%] for long scenarios with the learning rate (LR) parameter tuned. LR value Default means that the default LR value for the method was used. Transfer IN-C indicates that the LR is tuned utilizing the ImageNet-C benchmark with ground truth labels. In the Oracle rows the learning rate is chosen separately for each of the datasets with Oracle. The batch size is equal to 64.
<table><tr><td>Method</td><td>LR</td><td>C10-C (L)</td><td>IN-C (L)</td><td>IN-R (L)</td><td>DN-126 (L)</td><td>Avg.</td></tr><tr><td rowspan="3">AdaCont.</td><td>Default</td><td>81.8</td><td>18.8</td><td>26.5</td><td>61.7</td><td>47.2</td></tr><tr><td>Transfer IN-C</td><td>81.2</td><td>36.1</td><td>40.8</td><td>59.7</td><td>54.5</td></tr><tr><td>Oracle</td><td>82.6</td><td>36.1</td><td>40.8</td><td>62.7</td><td>55.6</td></tr><tr><td rowspan="3">I-AdaCont.</td><td>Default</td><td>85.4</td><td>40.4</td><td>38.2</td><td>64.4</td><td>57.1+9.9</td></tr><tr><td>Transfer IN-C</td><td>85.4</td><td>40.4</td><td>38.2</td><td>64.4</td><td>57.1+2.6</td></tr><tr><td>Oracle</td><td>86.2</td><td>40.4</td><td>42.3</td><td>64.4</td><td>58.3+2.7</td></tr><tr><td rowspan="3">CoTTA</td><td>Default</td><td>56.0</td><td>52.8</td><td>50.5</td><td>45.6</td><td>51.2</td></tr><tr><td>Transfer IN-C</td><td>11.2</td><td>52.8</td><td>50.5</td><td>45.6</td><td>40.0</td></tr><tr><td>Oracle</td><td>75.8</td><td>52.8</td><td>50.5</td><td>58.7</td><td>59.5</td></tr><tr><td rowspan="3">I-CoTTA</td><td>Default</td><td>68.4</td><td>35.4</td><td>39.6</td><td>56.8</td><td>50.1-1.1</td></tr><tr><td>Transfer IN-C</td><td>52.0</td><td>35.4</td><td>39.6</td><td>56.8</td><td>46.0+6.0</td></tr><tr><td>Oracle</td><td>79.2</td><td>35.4</td><td>40.8</td><td>56.8</td><td>53.1-6.4</td></tr><tr><td rowspan="3">RoTTA</td><td>Default</td><td>82.3</td><td>13.2</td><td>43.4</td><td>50.3</td><td>47.3</td></tr><tr><td>Transfer IN-C</td><td>73.2</td><td>30.6</td><td>41.0</td><td>55.3</td><td>50.1</td></tr><tr><td>Oracle</td><td>82.3</td><td>30.6</td><td>43.4</td><td>55.3</td><td>52.9</td></tr><tr><td rowspan="3">I-RoTTA</td><td>Default</td><td>79.6</td><td>32.7</td><td>39.7</td><td>57.2</td><td>52.3+5.0</td></tr><tr><td>Transfer IN-C</td><td>79.3</td><td>33.3</td><td>39.9</td><td>57.2</td><td>52.4+2.3</td></tr><tr><td>Oracle</td><td>79.6</td><td>33.3</td><td>39.9</td><td>57.2</td><td>52.5-0.4</td></tr></table>

## B.4 TUNING THE LEARNING RATE VALUE FOR LONG SCENARIOS

We investigate whether tuning the learning rate, arguably the most crucial hyperparameter, could enhance the performance of baseline methods in long adaptation scenarios. Following a realistic approach, we employ an Oracle technique on ImageNet-C (L) as a reference benchmark (inspired by Rusak et al. (2022), we name it Transfer IN-C) and apply the selected learning rate across all datasets. We also present results when the learning rate is chosen separately for each of the datasets with an Oracle. Tab. A.2 reports the complexity of hyperparameter optimization in test-time adaptation.

Our findings show the challenges of hyperparameter tuning. For instance, CoTTA achieves superior accuracy with its default learning rate compared to the version tuned on ImageNet-C. While both AdaContrast and RoTTA show improvements when using Transfer IN-C, our IT approach consistently outperforms these methods, even when specifically tuned for long-sequence adaptation on ImageNet-C. The results of Oracle, an (unrealistic) upper bound technique for learning rate tuning, suggest that optimal learning rates could potentially enable the original methods to outperform the IT version in some cases (e.g. CoTTA). However, this is not always the case. When tuned using Oracle,

I-AdaContrast achieves superior accuracy compared to its original version, while Oracle-tuned I-RoTTA performs similarly to its original counterpart. These results underscore both the difficulty of hyperparameter selection and the robust performance of our IT method across varying conditions.

## B.5 WALL-TIME RESULTS

We evaluate computational efficiency by measuring the wall-clock time required to process 10,000 CIFAR10C images on a single RTX 4080 GPU, with results shown in Tab. A.3. Our technique maintains a processing speed comparable to baseline methods, adding no computational overhead. MEMO's result highlights a significant issue with single-image adaptation methods, which exhibit substantially longer processing times.

## B.6 ON SINGLE IMAGE ADAPTATION METHODS.

Single image adaptation approaches, like MEMO (Zhang et al. 2022), are not susceptible to the studied problem of error accumulation. However, those methods have their own issues, e.g., they tend to be significantly more inefficient (Alfarra et al., 2024), as shown in Tab. A.3. MEMO is significantly slower than other techniques, with a wall-clock time around 20× higher than AdaContrast. Moreover, it is outperformed by IT-modified methods and the baselines on most datasets, except ImageNet-R (see Tab. 2and Tab. A.4).

Table A.3: Wall-clock time (seconds) for processing 10,000 images of CIFAR10-C on a single RTX 4080 GPU. The value in superscript indicates the performance improvements over the baseline.

<table><tr><td>Method</td><td>Time [s]</td></tr><tr><td>Source</td><td>3.4</td></tr><tr><td>MEMO</td><td>508.4</td></tr><tr><td>AdaCont.</td><td>25.3</td></tr><tr><td>I-AdaCont. CoTTA</td><td>25.0-0.3 40.7</td></tr><tr><td>I-CoTTA</td><td>40.2-0.5</td></tr><tr><td>RoTTA</td><td></td></tr><tr><td>I-RoTTA</td><td>27.7</td></tr><tr><td>PETAL</td><td>27.5-0.2</td></tr><tr><td></td><td>26.5</td></tr><tr><td>I-PETAL</td><td>26.2-0.3</td></tr></table>

## B.7 RESULTS ON COMMON-LENGTH BENCHMARKS

The overall accuracy of the tested TTA methods on common-length benchmarks is presented in Tab. A.4. The results for IT are mixed, sometimes outperforming the baseline methods and sometimes falling behind. The failures can be attributed to its slow adaptability, which stems from the intransigence issue discussed in Sec. 6.1. Moreover, it is worth noting that the original methods were specifically tuned for these benchmarks, whereas no such tuning was performed for IT. Our goal is to highlight the lack of robustness to longer adaptation sequence lengths exhibited by state-of-the-art teacher-student framework-based TTA methods and to provide a simple and robust solution in the form of the intransigent teacher.

## B.8 POTENTIAL OF ADAPTIVE β VALUE

In Tab. A.5, we explore a dynamic approach to adjusting the teacher model's momentum parameter (β). Our experiment begins with the default value of β = 0.999, allowing initial teacher model plasticity, then transitions to complete weight preservation of IT (β = 1.0) after one full cycle through the data. This hybrid approach outperforms our IT technique in several cases, demonstrating the potential of adaptive momentum strategies.

However, the results are not uniformly positive with our standard IT outperforming the hybrid method in some cases (AdaContrast on ImageNet-C (L) and CoTTA on DomainNet-126 (L)). This suggests that the fixed period length is not a universal value and there is a need to adjust it correctly.

## B.9 USING ONLY STUDENT MODEL

Given that the intransigent teacher strategy maintains fixed teacher model weights, one might question whether the teacher model is truly necessary. To address this, we conduct experiments using only the student network (see Tab. A.6). In these experiments, we take the student model's outputs directly as pseudo-labels, without any teacher involvement.

We included various averaged results based on the hyperparameter selection approach, acknowledging the non-trivial nature of the selection in TTA. Following a realistic approach, we employ an Oracle technique on ImageNet-C (L) as a reference benchmark (inspired by Rusak et al. (2022), we name it Transfer IN-C) and apply the selected learning rate across all datasets. Additionally, we choose the learning rate using Oracle on the first loop of the long (L) scenarios (Transfer 1 ×Loop). We also present results when the learning rate is chosen separately for each of the datasets with the Oracle.

Table A.4: Classification accuracy [%] for common-length benchmarks. The value in superscript indicates the improvements over the baseline.
<table><tr><td rowspan=1 colspan=2>Method         C10-C   IN-C    IN-R   DN-126</td></tr><tr><td rowspan=1 colspan=2>Source           56.5     18.0     36.2     54.7MEMO           65.6     25.0     40.9     53.2</td></tr><tr><td rowspan=1 colspan=2>BATCH SIZE 10</td></tr><tr><td rowspan=1 colspan=2>TestBN           75.0     27.0     36.6     46.5</td></tr><tr><td rowspan=1 colspan=2>TENT            75.7     31.2     38.9     52.4</td></tr><tr><td rowspan=1 colspan=2>EATA            77.4     36.0     43.1      54.4</td></tr><tr><td rowspan=1 colspan=2>SAR              75.8     31.3     41.9     52.8</td></tr><tr><td rowspan=1 colspan=2>RDUMB         77.2     34.8     41.3     52.0</td></tr><tr><td rowspan=1 colspan=2>AdaContrast     81.3     33.3     39.5     56.5</td></tr><tr><td rowspan=1 colspan=2>I-AdaContrast   $8 2 . 0 ^ { + 0 . 7 }$     $3 3 . 8 ^ { + 0 . 5 }$    $3 9 . 8 ^ { + 0 . 3 }$     $5 9 . 6 ^ { + 3 . 1 }$ </td></tr><tr><td rowspan=1 colspan=1>CoTTA75.1</td><td rowspan=1 colspan=1> $2 6 . 4$       $4 1 . 1$       52.0I-CoTTA         $6 9 . 8 ^ { - 5 . 3 }$    $2 8 . 3 ^ { + 1 . 9 }$    $3 5 . 6 ^ { - 5 . 5 }$     $4 9 . 5 ^ { - 2 . 5 }$ </td></tr><tr><td rowspan=1 colspan=2>RoTTA           79.0     29.2     38.6     55.9</td></tr><tr><td rowspan=1 colspan=2>I-RoTTA         $7 3 . 2 \AA ^ { - 5 . 8 }$     $2 9 . 4 ^ { + 0 . 2 }$    $3 9 . 3 ^ { + 0 . 7 } $     $5 6 . 6 ^ { + 0 . 7 }$ </td></tr><tr><td rowspan=1 colspan=2>PETAL           68.6     20.6     37.6     51.9</td></tr><tr><td rowspan=1 colspan=2>I-PETAL         $7 4 . 3 ^ { + 5 . 7 }$     $2 9 . 8 ^ { + 9 . 2 }$    $3 7 . 6 ^ { + 0 . 0 }$    $5 4 . 0 ^ { + 2 . 1 }$ </td></tr><tr><td rowspan=1 colspan=2>BATCH SIZE 64</td></tr><tr><td rowspan=1 colspan=2>TestBN           79.2     31.4     39.7     54.5</td></tr><tr><td rowspan=1 colspan=2>TENT            77.8     37.3     42.6     58.0</td></tr><tr><td rowspan=1 colspan=2>EATA            79.8     42.0     45.8     59.7</td></tr><tr><td rowspan=1 colspan=2>SAR              79.3     37.8     42.8     57.2</td></tr><tr><td rowspan=1 colspan=2>RDUMB         81.4     40.0     46.2     58.9</td></tr><tr><td rowspan=1 colspan=2>AdaContrast     82.6     34.8     40.9     62.0</td></tr><tr><td rowspan=1 colspan=2>I-AdaContrast  82.4-0.2   $3 5 . 1 ^ { + 0 . 3 }$    $4 1 . 0 ^ { + 0 . 1 }$   61.7-0.3</td></tr><tr><td rowspan=1 colspan=2>CoTTA           82.2     36.8     42.8     58.9</td></tr><tr><td rowspan=1 colspan=2>I-CoTTA         $6 8 . 6 \AA ^ { - 1 3 . 6 }$    $3 1 . 7 \AA ^ { - 5 . 1 }$     $3 5 . 9 ^ { - 6 . 9 }$     $5 4 . 4 ^ { - 4 . 5 }$ </td></tr><tr><td rowspan=1 colspan=2>RoTTA           80.9       $3 2 . 4$       $3 9 . 2$        $5 6 . 8$ </td></tr><tr><td rowspan=1 colspan=2>I-RoTTA         $7 6 . 7 ^ { - 4 . 2 }$     $3 0 . 6 ^ { - 1 . 8 }$     $3 9 . 3 ^ { + 0 . 1 }$     $5 6 . 3 ^ { - 0 . 5 }$ </td></tr><tr><td rowspan=1 colspan=2>PETAL           76.6     32.9     40.0      $5 6 . 4$ </td></tr><tr><td rowspan=1 colspan=2>I-PETAL         $7 8 . 5 ^ { + 1 . 9 }$     $3 4 . 2 ^ { + 1 . 3 }$    $3 9 . 6 ^ { - 0 . 4 }$     $5 4 . 4 ^ { - 2 . 0 }$ </td></tr></table>

Table A.5: Classification accuracy [%] for long scenarios with the weights of the teacher fixed only after the 1st loop on the test sequence. The value in superscript indicates the improvements over the IT technique's performance. The batch size is equal to 64.
<table><tr><td>Method</td><td>C10-C (L)</td><td>IN-C (L)</td><td>IN-R (L)</td><td>DN-126 (L)</td><td>Avg.</td></tr><tr><td>AdaCont.</td><td> $8 5 . 2 \AA ^ { - 0 . 1 }$ </td><td> $3 8 . 4 ^ { - 2 . 0 }$ </td><td> $3 8 . 2 ^ { + 0 . 1 }$ </td><td> $6 5 . 3 ^ { + 0 . 9 }$ </td><td> $5 6 . 8 ^ { - 0 . 3 }$ </td></tr><tr><td>CoTTA</td><td> $7 2 . 0 ^ { + 3 . 7 }$ </td><td> $4 5 . 0 ^ { + 9 . 6 }$ </td><td> $4 2 . 8 ^ { + 3 . 3 }$ </td><td> $4 9 . 1 ^ { - 6 . 9 }$ </td><td> $5 2 . 2 ^ { + 2 . 4 }$ </td></tr><tr><td>RoTTA</td><td> $8 0 . 4 ^ { + 0 . 7 }$ </td><td> $3 6 . 1 ^ { + 3 . 2 }$ </td><td> $4 1 . 0 ^ { + 1 . 3 }$ </td><td> $5 7 . 9 ^ { + 1 . 3 }$ </td><td> $5 3 . 9 ^ { + 1 . 7 }$ </td></tr></table>

While it appears that using only the student model can achieve reasonable performance with carefully tuned hyperparameters, it significantly underperforms compared to the intransigent teacher approach. The intransigent teacher delivers consistent results, demonstrating robustness across various hyperparameter selection methods, whereas the student-only model exhibits higher performance variance using different selection approaches.

## B.10 EFFECTS OF INTRANSIGENCE AMOUNT EXTENDED EXPERIMENT

To signify the point of Sec. 6.1, Fig. A.4 shows results where the test sequence is extended to 100 loops of common CIFAR10-C. It verifies that CoTTA with ET and $\beta = 0 . 9 9 9 9$ degrades below the performance of IT, given enough samples in the test sequence. This observation highlights a significant issue of TTA methods, as they can face test sequences of arbitrary lengths after deployment.

Table A.6: Classification accuracy [%] for long (L) scenarios while utilizing only the student model for adaptation (S. only) and tuning the learning rate parameter (LR). The batch size is equal to 64. Avg. Oracle is the average accuracy when the LR value is chosen separately for each of the datasets with Oracle. Avg. Transfer IN-C is the average accuracy with a single LR value chosen on the ImageNet-C dataset using the Oracle method. Average accuracy with the learning rate chosen on standard unrepeated benchmarks using the Oracle is presented in Avg. Transfer 1×Loop column. The value in superscript indicates the decrease below the IT technique's performance.
<table><tr><td rowspan="2">Method</td><td rowspan="2">LR</td><td rowspan="2">C10-C (L)</td><td rowspan="2">IN-C (L)</td><td rowspan="2">IN-R (L)</td><td rowspan="2">DN-126 (L)</td><td colspan="3">Avg.</td></tr><tr><td>Oracle</td><td>Transfer IN-C</td><td>Transfer 1×Loop</td></tr><tr><td>AdaCont. S. only</td><td>1e-5 1e-6 1e-7</td><td>82.4 81.4 79.1</td><td>23.0 36.6 31.9</td><td>37.4 40.7 39.6</td><td>59.1 59.2 57.0</td><td>54.7-3.8</td><td>54.5-2.6</td><td>50.5-6.6</td></tr><tr><td>CoTTA S. only</td><td>1e-4 1e-5 1e-6</td><td>35.3 70.8 79.2</td><td>6.3 34.8 31.9</td><td>14.1 41.3 40.0</td><td>1.1 1.6 7.0</td><td>40.6-12.5</td><td>37.1-8.9</td><td>26.7-23.1</td></tr><tr><td>RoTTA S. only</td><td>1e-4 1e-5 1e-6</td><td>21.9 51.6 86.6</td><td>1.4 14.5 26.0</td><td>5.8 35.6 37.3</td><td>6.9 48.9 53.8</td><td>45.9-6.5</td><td>45.9-6.5</td><td>41.4-10.8</td></tr></table>

![](images/5ecaa2c8f4b6d1ea7104faa14c1a26832260a6abb6008c08ccafa384a9584217.jpg)  
Figure A.4: Mean accuracy [%] of CoTTA with varying β for each loop of common CIFAR10-C testing sequence repeated 100 times. The brown dashed line indicates the accuracy of the source model as a reference.

Table A.7: Classification accuracy [%] for long scenarios with restoration probability parameter p of CoTTA method tuned. The batch size is equal to 64. Avg. Def. is the average accuracy with default p value. Avg. Transfer IN-C is the average accuracy with a single p value chosen on the ImageNet-C dataset using the Oracle method. Average accuracy when the p value is chosen separately for each of the datasets with Oracle is presented in Avg. Oracle column.
<table><tr><td rowspan="2">p-value</td><td rowspan="2">C10-C (L)</td><td rowspan="2">IN-C (L)</td><td rowspan="2">IN-R (L)</td><td rowspan="2">DN-126 (L)</td><td colspan="3">Avg.</td></tr><tr><td>Def.</td><td>Transfer IN-C</td><td>Oracle</td></tr><tr><td>0.1</td><td>73.1</td><td>29.0</td><td>41.8</td><td>26.9</td><td></td><td></td><td></td></tr><tr><td>0.01</td><td>53.7</td><td>24.8</td><td>35.6</td><td>13.7</td><td></td><td></td><td></td></tr><tr><td>0.001 (Def.)</td><td>56.0</td><td>52.8</td><td>50.5</td><td>45.6</td><td>51.2</td><td>51.6</td><td>58.1</td></tr><tr><td>0.0001</td><td>54.7</td><td>53.7</td><td>45.0</td><td>52.9</td><td></td><td></td><td></td></tr><tr><td>0.00001</td><td>54.3</td><td>53.6</td><td>49.0</td><td>55.0</td><td></td><td></td><td></td></tr><tr><td>0.0</td><td>52.7</td><td>53.5</td><td>48.9</td><td>54.5</td><td></td><td></td><td></td></tr></table>

## B.11 DISCUSSION ON MODEL RESET MECHANISM

CoTTA's proposed resetting mechanism aims to preserve source knowledge by stochastically restoring portions of the student model's weights to their original source state during each update iteration. In principle, an effective source knowledge preservation technique should eliminate the need for our IT technique.

However, CoTTA's reset mechanism introduces a restoration probability parameter. To ensure our findings were not biased by suboptimal parameter selection, we conducted parameter tuning experiments, documented in Tab. A.7. These results reveal that the optimal restoration probability varies across datasets, with model performance dependent on this parameter. When following a realistic scenario of tuning on a single dataset, the performance improvements were marginal (Avg. Transfer IN-C). Only by using an Oracle approach on all benchmarks, we observe performance gains, highlighting the practical limitations of this approach.

![](images/b91505dcbeb16b230e5e199da7795dcc668049899f195e35f36f121785e29b96.jpg)  
Figure A.5: Batch-wise accuracy plots of RDumb and I-AdaContrast methods on ImageNet-C (L) benchmark. The accuracy values were smoothed to make the plot clearer. RDumb resets the model every 1000 iterations, which causes significant drops in accuracy after the reset.

## B.12 DISCUSSION ON RDUMB

Similar to us, RDumb focuses on long scenarios and provides a straightforward analysis. However, there are several key differences between our work and RDumb.

First, RDumb experiments exclusively with common corruptions, while we demonstrate the phenomenon across various types of distribution shifts, including ImageNet-R and DomainNet. Additionally, we analyze this issue using several newer methods that incorporate the mean-teacher mechanism, which theoretically should address the problem (whereas only CoTTA among RDumb's baselines used this mechanism). Furthermore, we highlight a potential, simple solution to mitigate the issue, which is not explored in RDumb.

RDumb method mechanism of periodically resetting the model to its initial state leads to significant accuracy drops immediately following each reset, as illustrated in Fig. A.5. Such instability is particularly concerning since reliable test-time adaptation should maintain consistent performance throughout the adaptation process. Furthermore, the same constant reset interval is likely not optimal for every case, which adds a hyperparameter to select. In contrast, our IT approach achieves comparable performance without requiring parameter tuning.

## B.13 AUGMENTATION COUNT ABLATION

The CoTTA (Wang et al., 2022) and PETAL (Brahma & Rai, 2023) methods update the student network using a consistency loss between the student and teacher models. When the prediction confidence of the source model falls below a predefined threshold, the teacher generates pseudo-labels by averaging predictions across 32 distinct random augmentations of each image. This ensembling strategy requires 31 additional forward passes per batch, introducing a severe computational bottleneck that significantly reduces throughput compared to alternative state-of-the-art approaches. To evaluate the necessity of this computational overhead, Fig. A.6 compares pseudo-label generation using either 1 or 32 random augmentations.

Across evaluations on ImageNet-C (L) and ImageNet-R (L), accuracy profiles and degradation trajectories remain remarkably consistent regardless of the augmentation count. On ImageNet-R (L), CoTTA with a single augmentation even exhibits a marginally slower degradation rate toward the end of the sequence, while on DomainNet-126 (L), it yields a 2% to $4 \%$ accuracy improvement over the 32-augmentation baseline. The largest divergence occurs on CIFAR10-C (L), where 32 augmentations yield higher absolute accuracy for both methods. However, the underlying degradation trend persists, and both configurations ultimately collapse. Critically, reducing the augmentation count does not accelerate this collapse, demonstrating that the overall evaluation outcomes remain robust. While a definitive theoretical explanation for this behavior is outside the scope of this work, we hypothesize that averaging predictions across random transformations does not always yield a significant improvement in pseudo-label accuracy.

![](images/1e75f88a72c42aa2cdf691b8ba8170af517890bc6ab5a3197bba87364afc6618.jpg)  
Figure A.6: Mean accuracy [%] for each loop of the common testing sequence on CIFAR10-C (L), ImageNet-C (L), ImageNet-R (L), and DomainNet-126 for CoTTA and PETAL. Methods are tested with pseudo-labels generated by averaging across predictions on either 1 or 32 randomly augmented images. The brown dashed line indicates the source model accuracy as a reference.

## B.14 DISTILLATION VS. BATCH NORMALIZATION RECALIBRATION

To isolate whether the IT's performance gains stem from the distillation process or from simply recalibrating batch normalization (BN) statistics, we evaluate variants of our IT-enhanced methods without backpropagation (w/o BP). We compare these variants against the original baselines and the full IT baselines (Tab. A.8). In the w/o BP configuration, the adaptation process degenerates to adjusting only the student BN statistics, removing all influence from the teacher.

The specific technique for adjusting BN statistics varies across methods: CoTTA and PETAL utilize per-batch BN calculation, matching the TestBN baseline. Hence, their w/o BP results are identical to TestBN. AdaContrast continuously updates the EMA of the BN statistics, while RoTTA leverages its proposed robust batch normalization (RBN) module.

The results show that updating BN statistics alone provides a substantial accuracy boost. However, this update is not the sole driver of optimization. Across all 32 combinations of benchmarks, batch sizes, and baseline methods, the full IT framework outperforms or matches the w/o BP variant in 26 cases. This demonstrates that distillation-based adaptation yields consistent, additive improvements across the majority of evaluations. In the few cases where full IT underperforms relative to w/o BP, the original baseline method itself underperforms compared to both "I-"variants (e.g., PETAL on CIFAR10-C (L) with a batch size of 64, and AdaContrast on ImageNet-R (L) with a batch size of 10). This pattern indicates that when the underlying adaptation method fails over long horizons, stripping away all plasticity except for the BN statistics update yields the most effective results. Consequently, the average accuracy of the full IT method exceeds that of the w/o BP variants in all setups except for CoTTA. We attribute this exception to the poor baseline performance of CoTTA on CIFAR10-C (L), which underscores that the final performance of IT remains fundamentally constrained by its underlying baseline.

In summary, updating batch normalization statistics alone (w/o BP) delivers significant performance gains. However, this approach is strictly limited to architectures containing batch normalization layers. By contrast, distilling knowledge from an intransigent teacher provides further, cross-architecture improvements in most scenarios, validating the broader competitiveness of our approach.

## B.15 IT TEACHER WITH FIXED SOURCE BN STATISTICS

Using batch normalization (BN) statistics computed at test time is a common strategy in TTA. In our case, we adopt the same strategy for computing BN statistics as the baseline methods: CoTTA utilizes the TestBN technique, whereas AdaContrast and RoTTA update BN statistics using exponential moving average-based approaches.

To provide additional insight, we also experimented with fixed teacher statistics at test time, which are calculated during training on the source data (see Table A.9). The results suggest that adjusting the BN statistics in the teacher model improves overall performance, aligning with the findings of the TTA community (Yuan et al., 2023; Schneider et al., 2020a; Xiao & Snoek, 2024). This indicates that adapting these statistics can significantly mitigate performance drops on out-of-distribution data.

Table A.8: Classification accuracy [%] for long (L) scenarios. Superscript indicates improvements over the baseline. Bold indicates best performing method. Gray color indicates model collapse – performance worse than the nonadapting model (Source). Results averaged from 3 random seeds.
<table><tr><td>Method</td><td>C10-C (L)</td><td>IN-C (L)</td><td>IN-R (L)</td><td>DN-126 (L)</td><td>Avg.</td></tr><tr><td>Source</td><td>56.5</td><td>18.0</td><td>36.2</td><td>54.7</td><td>41.4</td></tr><tr><td colspan="6">BATCH SIZE 10</td></tr><tr><td>TestBN</td><td>75.1</td><td>26.9</td><td>36.2</td><td>49.6</td><td>47.0</td></tr><tr><td>AdaCont. I-AdaCont. w/o BP.</td><td>72.1  $7 7 . 7 ^ { + 5 . 6 }$   ${ \bf 8 4 . 1 ^ { + 1 2 . 0 } }$ </td><td>2.3  $2 9 . 6 ^ { + 6 . 3 }$ </td><td>8.0  $3 9 . 0 ^ { + 3 1 . 0 }$   $3 5 . 3 ^ { + 2 7 . 3 }$ </td><td>47.0  $5 4 . 4 ^ { + 7 . 4 }$ </td><td>32.4  $5 0 . 2 ^ { + 1 7 . 8 }$ </td></tr><tr><td>I-AdaCont. CoTTA I-CoTTA. w/o BP.</td><td>23.8  $7 5 . 1 ^ { + 5 1 . 3 }$ </td><td> ${ \bf 3 9 . 5 ^ { + 3 7 . 2 } }$  3.8  $2 6 . 9 ^ { + 2 3 . 1 }$ </td><td>33.7  $3 6 . 2 ^ { + 2 . 5 }$ </td><td> ${ \bf 6 3 . 2 ^ { + 1 6 . 2 } }$  6.0  $4 9 . 6 ^ { + 4 3 . 6 }$ </td><td> ${ \pmb 5 } 5 . 5 ^ { + 2 3 . 1 }$  16.8  $4 7 . 0 ^ { + 3 0 . 2 }$ </td></tr><tr><td>I-CoTTA RoTTA</td><td> $6 9 . 7 ^ { + 4 5 . 9 }$  82.5</td><td> $2 7 . 6 ^ { + 2 3 . 8 }$   $2 4 . 4$ </td><td> $3 7 . 4 ^ { + 3 . 7 }$   $4 3 . 0$ </td><td> $5 0 . 3 ^ { + 4 4 . 3 }$   $4 5 . 6$ </td><td> $4 6 . 3 ^ { + 2 9 . 5 }$  48.9</td></tr><tr><td>I-RoTTA. w/o BP. I-RoTTA</td><td> $7 0 . 7 \AA ^ { - 1 1 . 8 }$   $7 9 . 0 ^ { - 3 . 5 }$ </td><td> $2 4 . 2 ^ { - 0 . 2 }$   $3 3 . 6 ^ { + 9 . 2 }$ </td><td> $3 6 . 8 ^ { - 6 . 2 }$   $3 9 . 9 ^ { - 3 . 1 }$ </td><td> $5 3 . 4 ^ { 7 . 8 }$   $5 7 . 8 ^ { + 1 2 . 2 }$ </td><td> $4 6 . 3 ^ { - 2 . 6 }$   $5 2 . 6 ^ { + 3 . 7 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PETAL</td><td>68.0</td><td>3.9</td><td>37.9</td><td>47.3</td><td>39.3</td></tr><tr><td></td><td> $7 5 . 1 ^ { + 7 . 1 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>I-PETAL. w/o BP.</td><td></td><td> $2 6 . 9 ^ { + 2 3 . 0 }$ </td><td> $3 6 . 2 ^ { - 1 . 7 }$ </td><td> $4 9 . 6 ^ { + 2 . 3 }$ </td><td> $4 7 . 0 ^ { + 7 . 7 }$ </td></tr><tr><td>I-PETAL</td><td> $7 4 . 2 ^ { + 6 . 2 }$ </td><td> $2 8 . 7 ^ { + 2 4 . 8 }$ </td><td> $3 6 . 5 ^ { - 1 . 4 }$ </td><td> $5 0 . 7 ^ { + 3 . 4 }$ </td><td> $4 7 . 5 ^ { + 8 . 2 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td colspan="6">BATCH SIZE 64</td></tr><tr><td>TestBN</td><td>79.1</td><td>31.4</td><td>39.6</td><td>54.4</td><td>51.1</td></tr><tr><td>AdaCont.</td><td>81.8</td><td>18.8</td><td>26.5</td><td>61.7</td><td>47.2</td></tr><tr><td rowspan="2">I-AdaCont. w/o BP. I-AdaCont.</td><td> $7 7 . 9 ^ { - 3 . 9 }$ </td><td> $3 0 . 4 ^ { + 1 1 . 6 }$ </td><td> $3 9 . 4 ^ { + 1 2 . 9 }$ </td><td> $5 5 . 0 ^ { - 6 . 7 }$ </td><td> $5 0 . 7 ^ { + 3 . 5 }$ </td></tr><tr><td> $\mathbf { 8 5 . 4 } ^ { + 3 . 6 }$ </td><td> $4 0 . 4 ^ { + 2 1 . 6 }$ </td><td> $3 8 . 1 ^ { + 1 1 . 6 }$ </td><td> ${ \bf 6 4 . 4 ^ { + 2 . 7 } }$ </td><td> $5 7 . 1 ^ { + 9 . 9 }$ </td></tr><tr><td>CoTTA</td><td> $5 6 . 0$ </td><td>52.8</td><td>50.5</td><td> $4 5 . 6$ </td><td>51.2</td></tr><tr><td rowspan="2">I-CoTTA. w/o BP. I-CoTTA</td><td> $7 9 . 1 ^ { + 2 3 . 1 }$ </td><td> $3 1 . 4 ^ { - 2 1 . 4 }$ </td><td> $3 9 . 6 ^ { - 1 0 . 9 }$ </td><td> $5 4 . 4 ^ { + 8 . 8 }$ </td><td> $5 1 . 1 ^ { - 0 . 1 }$ </td></tr><tr><td> $6 8 . 3 ^ { + 1 2 . 3 }$ </td><td> $3 5 . 4 ^ { - 1 7 . 4 }$ </td><td> $3 9 . 6 ^ { - 1 0 . 9 }$ </td><td> $5 6 . 0 ^ { + 1 0 . 4 }$ </td><td> $4 9 . 8 ^ { - 1 . 4 }$ </td></tr><tr><td>RoTTA</td><td>82.3</td><td>13.2</td><td>43.4</td><td>50.3</td><td>47.3</td></tr><tr><td>I-RoTTA. w/o BP.</td><td> $7 0 . 7 \AA ^ { - 1 1 . 6 }$ </td><td> $2 4 . 2 ^ { + 1 1 . 0 }$ </td><td> $3 6 . 8 ^ { - 6 . 6 }$ </td><td> $5 3 . 4 ^ { + 3 . 1 }$ </td><td> $4 6 . 3 \AA ^ { - 1 . 0 }$ </td></tr><tr><td>I-RoTTA</td><td> $7 9 . 7 ^ { - 2 . 6 }$ </td><td> $3 2 . 9 ^ { + 1 9 . 7 }$ </td><td> $3 9 . 7 ^ { - 3 . 7 }$ </td><td> $5 6 . 6 ^ { + 6 . 3 }$ </td><td> $5 2 . 2 ^ { + 4 . 9 }$ </td></tr><tr><td>PETAL</td><td>56.3</td><td>36.8</td><td>40.7</td><td>58.6</td><td>48.1</td></tr><tr><td>I-PETAL. w/o BP.</td><td> $7 9 . 1 ^ { + 2 2 . 8 }$ </td><td> $3 1 . 4 ^ { - 5 . 4 }$ </td><td> $3 9 . 6 \mathrm { ^ { - 1 . 1 } }$ </td><td> $5 4 . 4 ^ { - 4 . 2 }$ </td><td> $5 1 . 1 ^ { + 3 . 0 }$ </td></tr><tr><td>I-PETAL</td><td> $7 8 . 8 ^ { + 2 2 . 5 }$ </td><td> $3 2 . 3 ^ { - 4 . 5 }$ </td><td> $3 9 . 8 ^ { - 0 . 9 }$ </td><td> $5 5 . 4 ^ { - 3 . 2 }$ </td><td> $5 1 . 6 ^ { + 3 . 5 }$ </td></tr></table>

Table A.9: Classification accuracy [%] for long scenarios while utilizing fixed teacher statistics calculated on source data in batch normalization layers. The batch size is equal to 64. Superscript indicates improvements over the full IT approach.
<table><tr><td>Method</td><td>C10-C (L)</td><td>IN-C (L)</td><td>IN-R (L)</td><td>DN-126 (L)</td><td>Avg.</td></tr><tr><td>I-AdaCont. Src. BN</td><td>83.2</td><td>24.8</td><td>35.3</td><td>63.3</td><td> ${ \bf 5 1 . 7 ^ { - 5 . 4 } }$ </td></tr><tr><td>I-CoTTA Src. BN</td><td>49.4</td><td>17.6</td><td>35.5</td><td>52.3</td><td> $3 8 . 7 ^ { - 1 1 . 1 }$ </td></tr><tr><td>I-RoTTA Src. BN</td><td>61.1</td><td>16.2</td><td>37.4</td><td>51.8</td><td> $4 1 . 6 ^ { - 1 0 . 6 }$ </td></tr></table>

## C REMAINING RESULTS

This section provides additional results not included in the main text. These include results on various architectures when the learning rate is tuned exclusively for the original methods on standard-length benchmarks and directly reused for IT-augmented versions (Tab. A.10), baseline evaluations for different β parameters (Tab. A.11, Figs. A.7 and A.8) and accuracy plots incorporating the IT technique (Figs. A.9 and A.10) for various benchmarks and architectures. Additionally, we include per-batch negative flip rate plots (Fig. A.11).

Table A.10: Classification accuracy [%] for long scenarios on CIFAR10-C and ImageNet-C with different neural network architectures. The value in superscript indicates the improvements over the baseline. The learning rate parameter is adjusted using the Oracle method on the original benchmarks exclusively for original methods and reused for IT-augmented ones.
<table><tr><td colspan="2">C10-C (L)</td><td colspan="4">IN-C (L)</td></tr><tr><td></td><td>RN26GN</td><td>RNXt-50</td><td>ViT-B16</td><td>SViT-T</td><td>CNXt tiny</td></tr><tr><td>Source</td><td>67.3</td><td>21.1</td><td>39.8</td><td>28.3</td><td>29.1</td></tr><tr><td>AdaCont.</td><td>75.5</td><td>21.3</td><td>33.9</td><td>17.0</td><td>19.2</td></tr><tr><td>I-AdaCont.</td><td>79.6+4.1</td><td>42.9+21.6</td><td>43.5+9.6</td><td>30.8+13.8</td><td>32.513.3</td></tr><tr><td>CoTTA</td><td>16.5</td><td>57.1</td><td>35.3</td><td>25.6</td><td>22.1</td></tr><tr><td>I-CoTTA RoTTA</td><td>61.6+45.1</td><td>38.4-18.7</td><td>39.9+4.6</td><td>28.9+3.3</td><td>29.5+7.4</td></tr><tr><td></td><td>62.8</td><td>16.2</td><td>37.1</td><td>7.6</td><td>18.7</td></tr><tr><td>I-RoTTA</td><td>70.5+7.7</td><td>35.2+19.0</td><td>40.7+3.6</td><td>26.5+18.9</td><td>29.7+11.0</td></tr><tr><td>PETAL</td><td>15.0</td><td>56.1</td><td>38.4</td><td>22.1</td><td>24.8</td></tr><tr><td>I-PETAL</td><td>52.8+37.8</td><td>36.2-19.9</td><td>39.41.0</td><td>26.64.5</td><td>27.93.1</td></tr></table>

Table A.11: Mean accuracy [%] for different exponential moving average β parameter. AdaContrast and CoTTA default originally to 0.999. (L) stands for the adaptation sequence being repeated 20 times. Gray color indicates model collapse – performance worse than no adaptation (Source).
<table><tr><td>Dataset</td><td>Method</td><td>0.9</td><td>0.99</td><td>0.999</td><td>0.9995</td><td>0.9999</td><td>IT</td></tr><tr><td>C10-C (L)</td><td>AdaCont. CoTTA</td><td>79.0 10.5</td><td>79.3 14.9</td><td>81.9 55.9</td><td>83.2 68.6</td><td>85.8 78.3</td><td>85.4 68.4</td></tr><tr><td>IN-C (L)</td><td>AdaCont. CoTTA</td><td>1.2 0.3</td><td>5.4 23.6</td><td>18.8 52.8</td><td>25.6 55.1</td><td>38.6 50.9</td><td>40.4 35.4</td></tr></table>

![](images/eced9b3fafca7c83ad56e0c00c1672259dec295454936c3a019fcacefa0cf43c.jpg)

![](images/a63e16eed986f565c0c808c0db001fd91fa3833d7210bad9d42f3b9c4b000c88.jpg)  
Figure A.7: Mean accuracy [%] for each loop of common testing sequence on ImageNet-C (L) using CoTTA (left) and on CIFAR10-C (L) using AdaContrast (right). The brown dashed line indicates the source model accuracy.

![](images/1cfb924e87a35150f83aecc7646022ec00de3e4ebaa771da913f7bfe4e384254.jpg)

![](images/ed64eea1f052b1a43a4c446e48053811f7349e5c6e67c3b0bf372708e996f428.jpg)  
Figure A.8: Mean accuracy [%] for each loop of common testing sequence on ImageNet-C (L) (left) and CIFAR10-C (right) using PETAL. The brown dashed line indicates the source model accuracy.

![](images/9f9dc91e08034a3d52b074a73c9ed45edcbe3ea129f2c955e26e3e7c73f9c939.jpg)  
(a) CoTTA, ResNet50

![](images/59ee8cc1aa4fe175c0681139aa0a35d96a284eaa9cdf5d4102a1054841e2432f.jpg)  
(b) CoTTA, ViT-B16

![](images/bbe60882d22731878389c92614d05cd622478491c9e389ea130737b82d74a761.jpg)  
(c) AdaContrast, ViT-B16

![](images/47121b2ae967831004e0f9058df2af1030c57cd15f6493e6218141e4c97f5087.jpg)  
(d) RoTTA, ViT-B16

![](images/7fa303b1dcc9e36eb2d5e48b5dd32094129e111b912be2b65157b1395dcf7cbd.jpg)  
(e) PETAL, ResNet50

![](images/774a4f16c6f037b422479b2f34bd3cdec9b2824c12c62d68374060be4eca1ebb.jpg)  
(f) PETAL, ViT-B16  
Figure A.9: Per-batch accuracy [%] on ImageNet-C (L). Comparison across different methods and architectures with EMA teacher (ET, orange) and intransigent teacher (IT, blue), for both teacher (solid) and student (dashed).

![](images/1c0bf31b0e300addd16091ec7917544ec1c47cb2574a6da335a32f5f4fc3dbbb.jpg)  
(a) AdaContrast

![](images/67d25477228ed1b7220bc7c2bc7c745619510deaa4efb5159c6c68e02ff0dc40.jpg)  
(b) CoTTA

![](images/272fcafec4813458b7d983ada47c7a99afb06dba6edea98ae53ce5b2782f234f.jpg)  
(c) RoTTA

![](images/412de1a49d289780971801c9456d1f3c0ba5a5e700b735a791d9c182161fe906.jpg)  
(d) PETAL  
Figure A.10: Per-batch accuracy [%] on CIFAR10-C (L) using WideResNet-28. Comparison with EMA teacher (ET, orange) and intransigent teacher (IT, blue), for both teacher (solid) and student (dashed).

![](images/1fa57b8d635df8b9185ea4da3178064fce0a5bac375c28417ffbb3e56b4a07ae.jpg)

![](images/cb160ee15a8259a6824e35c245b58d7d3872c478f482a102d29d05a1f3d924e9.jpg)

![](images/b37ea95fda337e26d2b7c66506b5891593aedc580dfbebb57a080fdfb11a2c9d.jpg)

![](images/77b661aad6b4043c8a1da888cdca2e08e2f3518acd15b2a167199fbb0597be82.jpg)  
(a) AdaContrast

![](images/b6fac2ed68fcdb19875a99fbfb8dedd226cb3802303850298142359f45fe030f.jpg)

![](images/3436c2531b51ac2db6ba4637107156702553a51bda1de889f92058b8da295e99.jpg)

![](images/f605688dee610857d3ed7d840997ee1621cfe1201b86f0768e2c9b3a407fc22d.jpg)

![](images/bf8f3432499dc1c85d853d620af7141110f0cbcfc4dc19b790f4c360000b4f82.jpg)  
(b) CoTTA

![](images/f9f8b6894cd7bf4863031121e8142730451610d33dac4ec6d6f45b40c26279fd.jpg)

![](images/3396aad200638c2ca0207c54107dc231169c5256508975e33bd79e154ad1830a.jpg)

![](images/bfab1ba3c8280f6cbcb8449837131a76b1a9258f1d3da75f93197a620bac0778.jpg)

![](images/ff19b659e854372009f09a2c8212f596f6587c5d0efd56be97eb7fb704db943a.jpg)  
(c) RoTTA

![](images/1cb6e9d6d88790c5411579c710448ac181f2c3d89eaeacf75cf3e27e70471de4.jpg)

![](images/a58e9bf95172401bc12fbb9fcb5993fb1196e7a1b2d4f2d97b0e4b6b8590eab0.jpg)

![](images/bd5a3157b05d6b0e9678f4d3b491a6698deab692144ecc81ffb334a558406ec2.jpg)

![](images/cba37d0155aade0972aba8c5150ab4aa6b8ae672dc96af60b440c75ba9c7ffb3.jpg)  
(d) PETAL  
Figure A.11: Per-batch Negative Flips Rate (NFR) of the student model on each benchmark using AdaContrast (a), CoTTA (b), RoTTA (c) and PETAL (d) methods and IT-enhanced version.