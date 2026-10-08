# Eficient Provably Private Classification with a Tabular Foundation Model

Talal Alrawajfeh∗<sup>1</sup>, Cristiana Diaconu∗<sup>2</sup>, Ossi Räisä<sup>3</sup>, Sebastian Rodriguez Beltran<sup>4</sup>, Yuan He<sup>1</sup>, John Bronskill<sup>2</sup>, Richard E. Turner<sup>2</sup>, and Antti Honkela<sup>1</sup>

Department of Computer Science, University of Helsinki, Helsinki, Finland <sup>2</sup>Department of Engineering, University of Cambridge, Cambridge, United Kingdom <sup>3</sup>CISPA Helmholtz Center for Information Security, Saarbrücken, Germany <sup>4</sup>Current address: Vienna, Austria

Tabular data underpin prediction and decision-making in medicine, finance, government and science, but often contain sensitive individual-level information, creating a need for accurate prediction while preserving privacy. Traditional private learning provides formal privacy guarantees, but requires slow dataset-specific optimisation, sufers substantial utility loss under strong privacy, and is often dificult to apply correctly. Tabular foundation models adapt rapidly to new datasets, but existing models lack formal privacy guarantees, and are highly vulnerable to membership-inference attacks, limiting their use on sensitive data. Here we introduce PrivTab, an easy to use tabular foundation model for diferentially private classification that embeds a privacy mechanism within its architecture. Pretrained on simulated datasets, PrivTab uses in-context learning to transform sensitive rows into compact, provably private summaries—efectively learning how to learn under privacy. PrivTab outperforms private linear and neural-network baselines under moderate-to-strong privacy, shows negligible membership leakage, maintains well-calibrated predictions under strong privacy, and reduces dataset fitting time by 10,000 times, requiring only a single forward pass. By combining formal privacy, speed, and easy of use, PrivTab brings recent advances in AI to applications where sensitive individual-level data have limited their adoption.

Records about individuals are predominantly tabular, from medical histories and financial transactions to census returns and administrative files. Machine learning ofers immense potential to transform the analysis of tabular data, improving predictions and the decisions they inform. Yet a model, or even a single prediction produced by it, can inadvertently reveal sensitive information. This tension raises a central scientific question: can a model deliver fast and accurate classification on sensitive tabular data while formally guaranteeing that any single individual’s record remains private?

Diferential privacy (DP) (Dwork et al., 2006; Dong et al., 2022) has emerged as the principal framework for resolving this conflict, formally limiting how much a model’s outputs can depend on any one individual’s record. In standard private machine learning (Ponomareva et al., 2023), DP stochastic optimisation (Song et al., 2013; Bassily et al., 2014; Abadi et al., 2016) achieves this protection by bounding the contribution of any individual’s record to the parameter updates and injecting calibrated noise during training. However, applying this approach to tabular data creates major operational hurdles. Training must be repeated from scratch for every new dataset and target privacy level, frequently at a steep cost to predictive performance, especially under strong privacy. Moreover, hyperparameter tuning and model selection can weaken privacy guarantees, and their privacy cost must be accounted for alongside model fitting (Papernot and Steinke, 2022). Even when the underlying mechanisms are theoretically private, implemented pipelines can fail to provide the intended guarantee because of privacyaccounting mistakes, noise-calibration errors or subtle implementation bugs (Ding et al., 2018; Bichsel et al., 2021; Tramèr et al., 2022; Nasr et al., 2023; Ganev et al., 2025; Cebere et al., 2026) (Fig. 1d). This reveals a critical gap for domain practitioners—who are rarely privacy specialists—and motivates classification methods with privacy built in that are easy to use and straightforward to audit.

Tabular foundation models (TFMs) address the problem of repeated training from scratch for each new dataset. Models such as TabPFN (Hollmann et al., 2023, 2025) and TabICL (Qu et al., 2025) take as input a labelled context dataset together with new query rows and predict their labels in a single forward pass. This eliminates the need for iterative retraining and enables near-instant fitting and prediction. They achieve this through in-context learning, by pretraining across millions of simulated tasks consisting of context–target tabular dataset pairs. The labelled context dataset is used to fit the model, whereas the held-out target dataset contains the query rows and their target labels which are used to evaluate the predictions and update the model. These simulated tasks contain no sensitive information. However, these models lack formal privacy protections, and for both TabPFN and TabICL, sensitive context rows—or non-private representations of them—must remain available at inference time to generate predictions. Indeed, using only black-box access to model predictions, we demonstrate that TabPFN and TabICL are vulnerable to context membership-inference attacks, allowing an attacker to determine whether a specific individual’s record was included in the context dataset with near-perfect accuracy (Fig. 1e).

![](images/5455165ab7fa1cac1ef77c495d805d33a9c285c56509c7f94a065174157386ce.jpg)

(d) Qualitative comparison
<table><tr><td>Property</td><td>Traditional Private Learning</td><td>Non-Private Tabular In-Context Learning (TabPFN, TabICL)</td><td>PrivTab</td></tr><tr><td>Formal privacy guarantee</td><td>L</td><td>X</td><td>L</td></tr><tr><td>Fast fitting for new datasets</td><td>X</td><td></td><td></td></tr><tr><td>Privacy-critical path is easy to inspect and audit</td><td>X</td><td></td><td>1</td></tr></table>

(e) Membership inference — Maternal Health Risk dataset  
![](images/dc0bde99ff8aa6c94d8d6d9de9b9b9449326fbcc9d1a342a3cf0d66996e9faab.jpg)  
Figure 1: Alternative approaches to tabular prediction from sensitive data. a, Private learning trains a new model on each sensitive dataset, requiring relatively slow dataset-specific optimisation. b, Non-private tabular learning fits new datasets in a single forward pass through in-context learning, but retains sensitive data for prediction. c, PrivTab instead releases a reusable private summary in one forward pass; all subsequent predictions are private due to the post-processing property of DP. Dashed lines mark the privacy boundary, and stopwatches indicate relative rather than measured time. d, Qualitative comparison; green ticks, orange crosses and grey dashes denote presence, absence and non-applicability, respectively. The auditability comparison concerns the complete data-dependent path, including hyperparameter tuning and privacy accounting, rather than an isolated private update (Ponomareva et al., 2023; Papernot and Steinke, 2022; Ganev et al., 2025). e, Membership-inference attack accuracy for the most vulnerable record of each method. For each record in the dataset, we compute the lower endpoint of the 95% confidence interval for attack accuracy; the bar shows the maximum of these lower endpoints. An accuracy of 50% corresponds to perfect privacy. TabPFN v3 and TabICL v2 approach 100%, whereas PrivTab remains much closer to perfect privacy (50%) at the displayed strong and weak formal privacy levels.

To address both the challenges of traditional private learning and the vulnerability of existing TFMs, we introduce PrivTab, which embeds a privacy mechanism directly within a TFM. Evaluated across a broad tabular benchmark, PrivTab achieves stronger predictive performance and better uncertainty estimates than conventional private baselines under moderate-to-strong privacy, remains competitive under weaker privacy, and reduces fitting time from minutes to milliseconds. It also retains its performance in a workflow that includes private preprocessing. Together, these results show that provable and auditable privacy can be built natively into foundation-model architectures, avoiding costly and dificult dataset-specific private training and opening a path to practical, privacy-preserving prediction from sensitive tabular data at scale.

## Principled and eficient privacy-aware in-context learning

PrivTab embeds the privacy mechanism directly within the foundation model architecture at the point where sensitive information enters. It compresses a sensitive context dataset into a compact DP summary (Fig. 1). Once constructed, this summary can be reused for all downstream query rows, avoiding repeated processing of the sensitive context and making prediction independent of the original context size. This design enables eficient fitting and prediction using standard computational hardware. Because all downstream predictions rely solely on this summary, the post-processing property of DP (Dwork et al., 2006) ensures they incur no additional privacy loss, allowing the summary to be stored and queried indefinitely.

Existing work based on pretrained models typically privatise either the inputs to or outputs from a model (Gao et al., 2025; Wu et al., 2024; Carey et al., 2024). While prior approaches have mainly targeted textual data and natural-language tasks (Gao et al., 2025; Wu et al., 2024), DP-TabICL (Carey et al., 2024) computes noisy aggregate statistics over groups of tabular records and converts them into textual examples for large language model (LLM) prediction. Such methods rely on general-purpose LLM inference, which can be substantially more expensive because prediction requires repeated inference for each query row. DP-TabICL, for example, reports prediction times of 0.14–1.11 seconds per query row, and evaluates only 10% of the target rows for the larger datasets owing to inference costs (Carey et al., 2024).

Drawing on TabArena’s computational guidelines (Erickson et al., 2025), which constrain computational resources and limit model evaluation to at most one hour, we adopt a controlled computational setting. We therefore consider only dedicated private tabular methods compatible with this computational regime: methods that can be fitted on a central processing unit (CPU) or a graphics processing unit (GPU) and support eficient prediction on CPU. This is particularly important for broad adoption, since downstream users might have limited computational resources and rely on standard CPU-based hardware.

PrivTab pretrains the foundation model with the privacy mechanism on millions of simulated tasks and across a range of privacy levels, achieving privacy-aware in-context learning—efectively “learning how to learn under privacy” (Räisä et al., 2024). This enables PrivTab to account explicitly for the efect of privacy noise in its predictions and uncertainty estimates—a crucial requirement for high-stakes decision-making. By contrast, standard private-learning pipelines do not provide this guarantee by default unless the efect of privacy noise on predictive uncertainty is explicitly modelled.

Designing a private summary mechanism presents several technical challenges: the influence of each individual record must be bounded, the attention architecture must remain expressive despite the added noise, and the implementation must remain simple enough to be auditable. We address these requirements by designing diferentially private multi-head cross-attention (DP-MHCA), a bounded cross-attention layer that produces a private summary with sensitivity that does not grow with context size. This design confines the privacy-critical operations to a self-contained transformer module.

PrivTab’s privacy-critical path is thus more localised and easier to inspect and audit than that of a conventional private-training pipeline. To strengthen the connection between the mathematical privacy guarantee (Supplement S3) and the executable implementation, we used three complementary approaches. First, we isolated the privacy-critical operations in a self-contained reference implementation (Supplement S3.3). Second, we formally verified the sensitivity bound and adaptive Gaussian diferential privacy (GDP) composition in Lean 4 (de Moura and Ullrich, 2021). Third, empirical sensitivity and privacy audits found behaviour consistent with the intended noise calibration across all evaluated privacy regimes (Supplement S6.2). These checks strengthen confidence that the implementation follows the proved mechanism; the Lean formalisation itself does not verify the exact software execution.

## Private classification across datasets

We evaluated PrivTab on 33 classification datasets from TabArena (Erickson et al., 2025), a standard tabular benchmark, across six privacy regimes and ten independent context–target splits per dataset. In each split, 80% of a dataset’s labelled rows form the context used to fit each method, and the remaining 20% are held-out target rows used to evaluate its predictions. To accommodate varying table sizes, we pretrained two model variants specialised for small and large datasets, routing incoming datasets accordingly. We compared PrivTab with two carefully tuned baselines: diferentially private logistic regression (DP-LR) and a diferentially private multilayer perceptron (DP-MLP). Both are standard model classes for private tabular learning and can be trained using DP stochastic optimisation for each dataset and privacy level. In contrast, PrivTab requires no dataset-specific

(a)  
![](images/d0790837c08f6a9e161b2f6fa68ce0d773e2522e7037af903ace87c76ecb2529.jpg)

(b)  
![](images/dad4e377c7695086ac7b7047c4b1ac10ac05ed078b321c219e1103c92cce9e68.jpg)

(c)  
![](images/8a98a9de24df0709ffc792deb4d6d4820b5be57d0db198e797ad09052c9785b8.jpg)

(d)  
![](images/141dd42191789995e620493d08678585f095d9c57d9f2279e326c479b7b12ce5.jpg)  
Figure 2: Predictive performance and fitting time across 33 TabArena datasets. a, Paired AUC gap, $\mathrm { A U C } _ { m } - \mathrm { A U C } _ { \mathrm { n o n - p r i v a t e L R } } ,$ for each method m (PrivTab, DP-MLP, DP-LR). b, Relative log-loss improvement over non-private logistic regression for the same three methods. Each quantity is computed on matched context– target splits before aggregation; zero denotes parity with non-private logistic regression and positive values mean m is better. Error bars are bootstrap 95% confidence intervals over datasets. c, Elo rating at each privacy level; higher ratings indicate better predictive performance, and shaded bands mark strong $( \mu < 0 . 1 5 )$ , moderate $( 0 . 1 5 \leq \mu < 0 . 6 )$ and weak $( \mu \geq 0 . 6 )$ formal privacy. Elo combines binary-task AUC and multiclass-task log loss and is calibrated separately at each privacy level so that diferentially private logistic regression (DP-LR) scores 1,000. All results use ten context–target splits per dataset. d, Aggregate Elo over all privacy levels versus median fitting time per context–target split on a logarithmic scale, so better methods lie towards the upper left. Horizontal and vertical bars are bootstrap 95% confidence intervals for median fitting time and Elo rating, respectively.

optimisation at any privacy level.

PrivTab is more accurate throughout the moderate-to-strong privacy regimes, with DP-MLP closing the gap only under weak privacy constraints (Figs. 2a and 2b). PrivTab’s advantage is most pronounced in log loss under heavy noise (Fig. 2b), consistent with training through an embedded privacy mechanism helping to preserve reliable uncertainty estimates when privacy bounds are tightest. In aggregate Elo comparisons spanning all datasets, privacy levels and splits, PrivTab achieves the highest overall rating (Fig. 2d), driven by its better performance in the moderate-to-strong privacy regimes (Fig. 2c).

The dataset-level breakdown shows contrasting trends with row count (Fig. 3): PrivTab’s AUC advantage over DP-MLP is largest on smaller datasets and diminishes as datasets grow, whereas its advantage over DP-LR generally grows with dataset size. This is consistent with diferences in how the models exploit additional context: PrivTab, like other tabular foundation models, may eventually saturate as context size increases, while DP-LR has limited capacity to benefit from larger datasets. In contrast, the greater capacity of DP-MLP allows it to continue benefiting from additional training data, thereby narrowing the gap to PrivTab on larger datasets. PrivTab also has a positive AUC gap for most of the 33 datasets at each of the three plotted privacy levels, particularly against

![](images/496928dfa358c0a46bf0e92357c7ed64c603cf5a52eb2c8855cc7edee6fe41a9.jpg)  
Figure 3: AUC gaps versus dataset size across 33 TabArena datasets. a, Mean paired AUC gap between PrivTab and the diferentially private multilayer perceptron (DP-MLP). b, The corresponding gap between PrivTab and diferentially private logistic regression (DP-LR). Columns show privacy levels µ = 0.1, 0.2 and 0.4; each point is a dataset’s mean paired gap across ten context–target splits, plotted against its number of rows on a logarithmic scale. Positive gaps favour PrivTab, while the dashed horizontal line marks equal AUC. Circles denote binary datasets and squares denote multiclass datasets; AUC is macro-averaged one-versus-rest for multiclass tasks. Grey lines are least-squares fits against log dataset size, shown as visual guides; annotations report Spearman’s ρ and the corresponding p-value.

DP-MLP (Fig. S7). The corresponding relationship with feature count is less pronounced. Complete per-dataset breakdowns against DP-MLP and DP-LR are provided in Figs. S5 and S6.

These results are even more striking when considering the computational diference between PrivTab and traditional private learning. PrivTab requires a median fitting time of approximately 10 milliseconds per context– target split—more than four orders of magnitude faster than DP-MLP and DP-LR, which require minutes of dataset-specific private training (Fig. 2d). Across the entire benchmark, fitting DP-LR and DP-MLP required approximately 400 and 130 graphics processing unit (GPU) hours<sup>1</sup>, respectively. In contrast, fitting PrivTab required only 30 GPU seconds. This cost reflects the repeated dataset-specific training and private model selection required by the baselines: there were 198 dataset–privacy configurations per baseline (33 datasets  6 privacy levels). Ten splits per configuration required 1,980 final training runs per baseline, each preceded by private learning-rate selection that consumed part of the privacy budget. In contrast, PrivTab requires neither datasetspecific nor privacy-level-specific optimisation or hyperparameter tuning. Each pretrained PrivTab variant can be used across diferent privacy levels without retraining. Comprehensive dataset-level runtimes for each method are provided in Fig. S8.

On a CPU, median prediction times for the full target set were 66.6 milliseconds (ms) for PrivTab, 0.35 ms for DP-MLP and 0.18 ms for DP-LR (Fig. S9 and Table S11). These medians were computed across the 33 datasets after averaging timings within each dataset.

## Private preprocessing in practice

The TabArena benchmark isolates prediction by applying a common idealised preprocessing procedure to all methods. Here, preprocessing refers to shifting and scaling the columns in the sensitive dataset before model fitting, using statistics computed from those columns. The same statistics are then used to transform the queries accordingly and must therefore be privatised before being released alongside the private summary. We show that this is possible in practice for datasets with public column descriptions detailed enough to infer meaningful clipping intervals, and whose modest feature counts allow private preprocessing to retain useful signal. A clipping interval specifies lower and upper bounds for a numeric feature, with values outside the interval truncated to the nearest bound. These bounds limit the sensitivity of the estimated means and variances, allowing the noise required for private preprocessing to be calibrated. We inferred numeric clipping intervals from the public descriptions using an LLM; no sensitive values were supplied to that model.

b  
![](images/3b0f1fa9d77d44667f53a41bf8fcd3c64c79f79e61aa0f2254de2663fbeb33c7.jpg)

![](images/5375f0784d3d897b75c0593c9b88365fd08aead076c6de26c23c63d68360c895.jpg)  
Figure 4: Private preprocessing case study. a, Mean PrivTab AUC across ten context–target splits at total privacy level $\mu = 0 . 4$ . The three preprocessing variants compare non-private standardisation, partially private preprocessing using oracle bounds computed from empirical dataset minima and maxima, and private preprocessing using bounds inferred from public dataset descriptions by an LLM. Only the LLM-derived bounds avoid using sensitive values to select clipping intervals; the oracle-bounds comparison is partially private. b, Mean AUC for PrivTab, DP-MLP and DP-LR using LLM-derived bounds and private standardisation across the same five datasets and ten splits. Higher AUC is better. Error bars show bootstrap 95% confidence intervals. Legends beneath each panel identify its plotted variants.

The total privacy budget used in the case study here, $\mu = 0 . 4$ , was split between private preprocessing and private prediction or training. Bank Marketing, Pima Diabetes and Maternal Health Risk originate from the UCI Machine Learning Repository (Kelly et al., 2023); the other two datasets are COMPAS from OpenML and ACS Income from Folktables (Bischl et al., 2025; Ding et al., 2021). PrivTab achieved the highest Elo rating across the five datasets in the case study (Fig. S12c), with numerical values reported in Table S25; method-level AUC and log-loss results are provided in Fig. 4b and Fig. S12b, respectively. Oracle (empirical min–max) bounds and LLM-derived bounds yielded similar AUC and log loss on four of the five datasets (Fig. 4a and Fig. S11). The oracle comparison is partially private: its standardisation and prediction are privatised, but its data-dependent bounds do not have an end-to-end privacy guarantee. The LLM-derived bounds use only public descriptions. Pipeline details, prompts, bounds and numerical results appear in Supplement S6.6.

## Non-private models reveal membership

TFMs use a context dataset to infer labels for incoming query rows. Because these models fit directly to the sensitive context dataset, they provide no formal guarantee against membership inference of specific context records. To evaluate this vulnerability, we conducted a likelihood-ratio membership-inference attack (Carlini et al., 2022), which distinguishes members of a context set from non-members based on prediction confidence. Because these models fit to context records without privacy protection, they can exhibit higher confidence when queried on the same context records. In contrast, PrivTab and private-learning methods use DP mechanisms that limit how reliably an attacker can infer context membership from prediction confidence.

We performed the analysis on clinical records from the Maternal Health Risk dataset. Rather than reporting aggregate averages, which can hide highly vulnerable records (Knolle et al., 2026), we report record-level vulnerability using fitted attack accuracy Fig. 1e and the AUC of record-level ROC curves Fig. S3a,b. The conservative lower confidence estimates of fitted attack accuracy in Fig. 1e approach 100% for the most vulnerable records under TabPFN v3 and TabICL v2. In contrast, PrivTab remains much closer to perfect privacy across all evaluated privacy levels. While empirical attacks cannot replace formal proofs, they illustrate the practical importance of DP guarantees. Complete attack specifications, formal confidence bounds and full benchmark evaluations are detailed in the Methods and Supplement S6.1.

![](images/1bfd8afb2adbcc85df9f6256a8dc55d06c7d507a7bc45765cae649ac33d8074b.jpg)  
Figure 5: Record-level membership-inference protection. a, Distribution of the record-wise membershipinference AUC on the Maternal Health Risk dataset. For each record, we compute its AUC and 95% confidence interval, then plot the lower endpoint of that interval. Boxes show the interquartile range and median, whiskers span the central 95%, squares mark the maximum across records, and dotted vertical lines mark the provable AUC upper bound for PrivTab at each privacy level. b, ROC curves for the most vulnerable record at each of the six PrivTab privacy levels and for TabPFN v3 and TabICL v2. Grey dashed lines in a,b denote perfect privacy. PrivTab remains much closer to perfect privacy whereas the most vulnerable rows of the non-private models are highly vulnerable.

## Discussion

Our results suggest that introducing privacy into the TFM paradigm can substantially reduce computational cost and improve ease of use while retaining competitive predictive utility under privacy. Traditional private learning methods enforce privacy through the optimisation procedure for every new dataset, which requires repeatedly fitting and privatising a new model. PrivTab instead places the privacy mechanism inside the foundation-model architecture. The model is pretrained once to perform prediction through a private summary, and fitting to a new sensitive dataset only requires constructing this summary through a single forward pass. This amortises computational cost and allows the model to be used across privacy levels without retraining.

This architectural choice also makes the privacy boundary simpler and easier to audit. Once the sensitive dataset has been converted into the private summary, all later predictions are post-processing and do not require further access to the sensitive records. The private summary mechanism can therefore be inspected, its mathematical guarantees formally verified, and its implementation tested through targeted empirical audits. The summary itself can also be stored and shared with other users for later predictions. Moreover, because PrivTab is pretrained using the same privacy mechanism used at prediction, it learns how to make predictions in the presence of privacy noise rather than encountering privacy noise only during dataset-specific private training.

Several limitations remain. We study only classification and not regression. PrivTab’s privacy model assumes one user per row, and extending the guarantee to settings in which one individual contributes multiple rows is less straightforward than for approaches based on DP stochastic optimisation. PrivTab has a smaller relative advantage when privacy is weak or context sizes are large, as models trained separately on each dataset can benefit more from both the relaxed privacy constraint and additional data. Similar diminishing returns on larger context sizes have also been observed in earlier TFMs such as TabPFN v1, and future research could benefit from recent advancements in this area to improve PrivTab. Finally, preprocessing must also be included in the privacy analysis in practice. Our private-preprocessing experiments demonstrate one possible approach, but the appropriate preprocessing pipeline remains application-dependent. The same preprocessing requirement also applies to the private-learning baselines.

More broadly, our results suggest that foundation models can be designed around privacy from the outset, rather than having privacy added retrospectively to every downstream training procedure. PrivTab provides one example of this principle by learning once how to perform inference through a private representation of the sensitive dataset, while keeping the mechanism that accesses the sensitive data explicit and auditable.

## Methods

## Diferential privacy

Diferential privacy (DP) limits how much the output distribution of a randomised algorithm can change when one person’s record changes (Dwork et al., 2006). Let $\mathcal { D } = \{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { n }$ be a dataset of individual records and $\mathcal { A } : \mathbb { D }  \mathcal { O }$ a randomised mechanism from datasets D to an output space ; in supervised learning, $\mathbf { z } _ { i }$ can be an input–label pair $( \mathbf { x } , y )$ . Two datasets  and $\mathcal { D } ^ { \prime }$ are neighbouring or adjacent, denoted $\mathcal { D } \simeq \mathcal { D } ^ { \prime }$ , if they have the same size and difer in at most one record $\mathbf { z } _ { i }$ . This is known as the replace-one or substitute adjacency relation.

We report privacy in the Gaussian DP (GDP) parameterisation (Dong et al., 2022). A mechanism satisfies $\mu { \mathrm { - } } \mathrm { G D P }$ when its outputs on neighbouring datasets are no easier to distinguish than $\mathcal { N } ( 0 , 1 )$ and $\mathcal { N } ( \mu , 1 )$ in a hypothesis test; smaller µ means stronger privacy. For a vector-valued function $f ,$ the Gaussian mechanism $\dot { \mathcal { A } } ( \mathcal { D } ) = f ( \mathcal { D } ) + \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ has $\mu = \Delta ( f ) / \sigma$ , where σ is the noise standard deviation, I is the identity matrix of the output dimension, and the ℓ<sub>2</sub>-sensitivity is $\begin{array} { r } { \Delta ( f ) = \operatorname* { s u p } _ { \mathcal { D } \simeq \mathcal { D ^ { \prime } } } \| f ( \mathcal { D } ) - f ( \mathcal { D ^ { \prime } } ) \| _ { 2 } } \end{array}$ . Thus, bounding a one-row change in f determines the noise needed for a target privacy level. The adaptive composition of mechanisms with GDP parameters $\mu _ { 1 } , \ldots , \mu _ { k }$ has parameter

$$
\mu _ { \mathrm { t o t a l } } ^ { 2 } = \sum _ { i = 1 } ^ { k } \mu _ { i } ^ { 2 } .\tag{1}
$$

Once a DP summary is released, computation using only that summary is post-processing and costs no additional privacy budget. Formal definitions, conversion between GDP and $( \varepsilon , \delta ) – \mathrm { D P }$ , and the precise adaptive-composition statement are in Supplement S1.2.

## Model architecture

Neural processes (Garnelo et al., 2018a,b) use a labelled context set to make predictions at new query inputs without dataset-specific training, requiring only a single forward pass. Transformers (Vaswani et al., 2017) process collections of input representations using self-attention, which allows each representation to be updated according to its relationships with the others. Attentive and transformer neural processes (Kim et al., 2019; Nguyen and Grover, 2022) adapt this mechanism to conditional prediction, allowing query representations to attend to the labelled context and extract information relevant to each prediction. Latent-bottleneck attentive neural processes (LBANP) (Feng et al., 2023) instead first compress the context into a fixed-size set of latent representations, which provides an intermediate summary through which information passes before predictions are made at the queries.

High-level architecture. PrivTab follows the structure of the LBANP but makes the context-to-summary operation diferentially private (Fig. E1). Each row has a feature vector $\mathbf { x } _ { i } = ( x _ { i 1 } , \dots , x _ { i d } )$ , where d is the number of input features and $x _ { i k }$ is the value of feature k in row i. Let $\mathcal { D } _ { c } = \{ ( \mathbf { x } _ { i } ^ { c } , y _ { i } ^ { c } ) \} _ { i = 1 } ^ { n _ { c } } \mathrm { ~ . ~ }$ be the labelled context and $X _ { t } \overset { \vartriangle } { = } ( { \bf x } _ { 1 } ^ { t } , \ldots , { \bf x } _ { n _ { t } } ^ { t } )$ the query inputs. During pretraining and evaluation, $\mathcal { D } _ { t } \overset { \cdot } { = } \overset { \cdot } { \{ } ( \mathbf { x } _ { j } ^ { t } , y _ { j } ^ { t } ) \} _ { j = 1 } ^ { n _ { t } }$ also contains target labels, which are withheld from the predictor and used only for the learning objective or evaluation.

The row embedding model RowEmbed converts each row into a fixed-width token, using the observed label for context rows and an unknown-label marker for queries. It maps each context row $\left( \mathbf { x } _ { i } ^ { c } , y _ { i } ^ { c } \right)$ to

$$
\mathbf { z } _ { i } ^ { c } = \mathrm { R o w E m b e d } \left( \hat { \mathbf { x } } _ { i } ^ { c } , \mathrm { L a b e l E m b e d } ( y _ { i } ^ { c } ) \right)\tag{2}
$$

and each query $\mathbf { x } _ { j } ^ { t }$ to

$$
\begin{array} { r } { \mathbf { z } _ { j } ^ { t , ( 0 ) } = \mathrm { R o w E m b e d } \left( \hat { \mathbf { x } } _ { j } ^ { t } , \mathrm { L a b e l E m b e d } ( y _ { \mathrm { n u l l } } ) \right) . } \end{array}\tag{3}
$$

Here xˆ is the padded, fixed-width feature representation of x, LabelEmbed is a learned label embedding, and RowEmbed is the row-wise embedding network. The query uses the dedicated unknown-label index $y _ { \mathrm { n u l l } }$ see Supplement S3 for the exact construction. Each row token has width $d _ { \mathrm { t o k } } = 2 5 6$ . The context embeddings pass through the private summary module to produce ${ \tilde { S } } ;$ the prediction module then combines each query embedding with this summary to obtain label probabilities:

$$
\begin{array} { c } { \tilde { S } = \mathrm { P r i v a t e S u m m a r y } _ { \phi } ( \mathbf { z } _ { 1 : n _ { c } } ^ { c } ; \mu ) , } \\ { q _ { \psi } ( y \mid \mathbf { x } _ { j } ^ { t } , \tilde { S } ) = \big [ \mathrm { P r e d i c t i o n } _ { \psi } ( \mathbf { z } _ { j } ^ { t , ( 0 ) } ; \tilde { S } ) \big ] _ { y } . } \end{array}\tag{4}
$$

Here $[ \cdot ] _ { y }$ denotes the selection of the probability for label $y ,$ and ϕ and ψ denote the fixed summary and prediction parameters, respectively, and $\mu$ is the requested privacy level. The model has 9.3 million parameters, with 8.9 million in the private summary module and 0.4 million additional parameters in the prediction module.

Private summary module. The private summary module compresses the labelled context into a fixed-size collection of noisy summaries. The module starts from $m = 1 2 8$ pseudo-tokens, $ { \mathbf { u } } _ { 1 } , \dots ,  { \mathbf { u } } _ { m }$ , learned during pretraining and subsequently fixed. Each of the $L _ { \mathrm { m a i n } } = 3$ main layers first applies private multi-head cross attention (DP-MHCA) to the context embeddings and then multi-head self-attention (MHSA) (Vaswani et al., 2017) to the resulting summary tokens. Denote $\mathbf { u } _ { 1 : m }$ as a short-hand for $ { \mathbf { u } } _ { 1 } , \dots ,  { \mathbf { u } } _ { m }$ , and let $\tilde { \mathbf { s } } _ { 1 : m } ^ { ( 0 ) } = \mathbf { u } _ { 1 : m }$ , then for each layer $\ell = 1 , \ldots , L _ { \mathrm { m a i n } }$

$$
\begin{array} { r l } & { { \bf s } _ { 1 : m } ^ { ( \ell ) } = \mathrm { D P - M H C A } _ { \mathrm { b l o c k } } ^ { ( \ell ) } ( \tilde { \bf s } _ { 1 : m } ^ { ( \ell - 1 ) } ; { \bf z } _ { 1 : n _ { c } } ^ { c } ) , } \\ & { \tilde { \bf s } _ { 1 : m } ^ { ( \ell ) } = \mathrm { M H S A } _ { \mathrm { b l o c k } } ^ { ( \ell ) } ( { \bf s } _ { 1 : m } ^ { ( \ell ) } ) . } \end{array}\tag{5}
$$

The state $\mathbf { s } _ { 1 : m } ^ { ( \ell ) }$ is the intermediate output of the private cross-attention block; $\tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) }$ is the saved summary after the self-attention block. Another $L _ { \mathrm { p o s t } } = 2$ layers refine the summary using only self-attention:

$$
\tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) } = \mathrm { M H S A } _ { \mathrm { b l o c k } } ^ { ( \ell ) } ( \tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell - 1 ) } ) , \qquad \ell = L _ { \mathrm { m a i n } } + 1 , \ldots , L .\tag{6}
$$

Here $L = L _ { \mathrm { m a i n } } + L _ { \mathrm { p o s t } } = 5$ , and the module returns the collection of all saved states,

$$
\tilde { S } = \left( \tilde { \mathbf { s } } _ { 1 : m } ^ { ( 1 ) } , \ldots , \tilde { \mathbf { s } } _ { 1 : m } ^ { ( L ) } \right) = \operatorname { P r i v a t e S u m m a r y } _ { \phi } ( \mathbf { z } _ { 1 : n _ { c } } ^ { c } ; \mu ) .\tag{7}
$$

The operators $\mathrm { D P \mathrm { - M H C A _ { b l o c k } } }$ and $\mathrm { M H S A _ { b l o c k } }$ denote complete blocks, including a token-wise feed-forward network; their details are in Supplement S3. Only DP-MHCA reads the context embeddings; self-attention and the remaining summary updates are post-processing of its private outputs.

DP-MHCA layer. DP-MHCA replaces the usual softmax attention (Vaswani et al., 2017) with a sum of bounded contributions and adds Gaussian noise to protect individual context rows. For each layer ℓ and head h, learned matrices project the summary token $\tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) }$ into a query and the context token $\mathbf { z } _ { j } ^ { c }$ into a key and a value:

$$
\begin{array} { r l } & { \mathbf { q } _ { i } ^ { ( h ) } = \mathbf { W } _ { Q } ^ { ( \ell , h ) } \tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) } , } \\ & { \mathbf { k } _ { j } ^ { ( h ) } = \mathbf { W } _ { K } ^ { ( \ell , h ) } \mathbf { z } _ { j } ^ { c } , } \\ & { \bar { \mathbf { v } } _ { j } ^ { ( h ) } = \mathbf { W } _ { V } ^ { ( \ell , h ) } \mathbf { z } _ { j } ^ { c } . } \end{array}\tag{8}
$$

Here each projection matrix lies in $\mathbb { R } ^ { d _ { h } \times d _ { \mathrm { t o k } } }$ . In the first layer, $\tilde { \mathbf { s } } _ { i } ^ { ( 0 ) } = \mathbf { u } _ { i } ;$ later layers use summary tokens obtained by post-processing previous private outputs. For each summary token i, the attention output before normalisation is

$$
\begin{array} { l l l } { { \displaystyle { \bf a } _ { i } ^ { ( h ) } = \sum _ { j = 1 } ^ { n _ { c } } \mathrm { t a n h } \left( \frac { \langle { \bf q } _ { i } ^ { ( h ) } , { \bf k } _ { j } ^ { ( h ) } \rangle } { \sqrt { d _ { h } } } \right) \frac { { \bar { \bf v } } _ { j } ^ { ( h ) } } { \operatorname* { m a x } \{ \| { \bar { \bf v } } _ { j } ^ { ( h ) } \| _ { 2 } , \varepsilon _ { v } \} } } , }  \\ { { \displaystyle { \tilde { \bf s } } _ { i , \mathrm { a t t n } } = \left[ { \bf a } _ { i } ^ { ( 1 ) } , \ldots , { \bf a } _ { i } ^ { ( H ) } \right] + \xi _ { i } . } } \end{array}\tag{9}
$$

Here $\mathbf { a } _ { i } ^ { ( h ) }$ is the pre-noise head output, $d _ { h }$ is the per-head key dimension, and brackets denote concatenation across heads. The noise vectors $\pmb { \xi } _ { i } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { H d _ { h } } )$ are independent across summary tokens and layers; the noise scale σ is calibrated below. The numerical stabiliser is $\varepsilon _ { v } = 1 0 ^ { - 1 2 }$ . Keys and values are computed row-wise, so each summand depends on one context row.

Normalising the noisy attention outputs controls their scale before the residual update. We applied this step in certain configurations, especially during evaluation, by rescaling with the largest token norm:

$$
\tilde { \bf s } _ { i , \mathrm { a t t n } } = \frac { G \tilde { \bf s } _ { i , \mathrm { a t t n } } } { \operatorname* { m a x } _ { k = 1 , \ldots , m } \lVert \tilde { \bf s } _ { k , \mathrm { a t t n } } \rVert _ { 2 } + \varepsilon _ { s } } ,\tag{10}
$$

where $G = 5 1 2$ and $\varepsilon _ { s } = 1 0 ^ { - 6 }$ in the reported experiments. This preserves the relative magnitudes of the attention outputs while controlling their overall scale, which substantially improved performance on longer contexts (Table S37). These normalised attention outputs precede the residual and feed-forward updates in the private block and the subsequent self-attention block; the saved states $\tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) }$ include those operations. Applied after noise addition, this normalisation is post-processing and does not afect the privacy guarantee.

Privacy guarantees. The following theorem states the privacy guarantee for the context dataset and all predictions made from its private summary.

Theorem (Privacy of PrivTab). Fix the pretrained parameters which are independent from $\mathcal { D } _ { c }$ . For $\mu > 0$ , let each of the $L _ { \operatorname* { m a i n } } D P \ – M H C A$ layers add independent noise from ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ to every attention-output coordinate, where

$$
\sigma = \frac { 2 \sqrt { H m L _ { \mathrm { m a i n } } } } { \mu } .\tag{11}
$$

Releasing the private summary $\tilde { S }$ together with predictions for any number ofpublic queries satisfies $\mu { - } G D P$ with respect to $\mathcal { D } _ { c }$ under replace-one adjacency.

Justification. Theorem S3.1 bounds each layer’s sensitivity of $\operatorname { E q } .$ . (9) by $2 \sqrt { H m }$ , independently of context size. With the noise scale in Eq. (11), each layer satisfies $\mu / \sqrt { L _ { \mathrm { m a i n } } } \mathrm { - G D P }$ , and Theorem S3.2 establishes the overall $\mu { \mathrm { - } } \mathrm { G D P }$ guarantee by adaptive composition. Predictions from the released summary preserve this guarantee by post-processing (Dwork et al., 2006).

Prediction module. The prediction module combines each query with the private summaries to obtain label probabilities. At each layer, query tokens use standard multi-head cross-attention (MHCA) (Vaswani et al., 2017) to attend to the corresponding saved summary state:

$$
\begin{array} { r } { \mathbf { z } _ { 1 : n _ { t } } ^ { t , ( \ell ) } = \operatorname { M H C A } _ { \mathrm { b l o c k } } ^ { ( \ell ) } ( \mathbf { z } _ { 1 : n _ { t } } ^ { t , ( \ell - 1 ) } ; \tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) } ) , \qquad \ell = 1 , \ldots , L . } \end{array}\tag{12}
$$

Here $\mathrm { M H C A _ { b l o c k } }$ denotes the complete query-to-summary block, including pre-normalisation, residual connections and a token-wise feed-forward network. Each query is processed independently using only its embedding and the saved summary states. After the L layers, a token-wise multilayer perceptron maps the final query representation $\mathbf { z } _ { j } ^ { t , ( L ) }$ to a vector of raw class scores (logits) $\hat { \mathbf { y } } _ { j } \in \mathbb { R } ^ { C _ { \mathrm { m a x } } }$ , as in the appendix. For a dataset with $C \leq C _ { \operatorname* { m a x } }$ public classes indexed $1 , \ldots , C _ { \mathrm { { i } } }$ , the probability assigned to class $y$ is

$$
q _ { \psi } ( y \mid \mathbf { x } _ { j } ^ { t } , \tilde { S } ) = \frac { \exp ( \hat { y } _ { j , y } ) } { \sum _ { c = 1 } ^ { C } \exp ( \hat { y } _ { j , c } ) } ,\tag{13}
$$

where $\hat { y } _ { j , c }$ is the score for class c in ${ \hat { \mathbf { y } } } _ { j } .$ , and ψ denotes the predictive-network parameters. The context and query paths share their row embedding weights, so these weights occur in both $\phi$ and $\psi .$ The denominator makes these probabilities sum to one; the predicted class is the one with the largest probability. Because the query-side computation depends on the context only through ${ \tilde { S } } ,$ predictions for any number of public queries are post-processing of the private release.

The efect of re-ordering rows and columns. PrivTab is invariant to permutations of context rows and equivariant to permutations of target rows: reordering the context leaves predictions unchanged, while reordering targets reorders their predicted labels. This follows because the model uses no positional encodings (Vaswani et al., 2017), and every operation before the DP-MHCA layers is row-wise. PrivTab, however, is neither permutation invariant nor equivariant to column ordering. This is because each row is mapped to a fixed-size vector by padding it to a certain number of columns and then feeding it into a fully connected neural network (RowEmbedding). It learns during pretraining, similar to TabPFN and TabICL, to become less sensitive to the exact ordering of the columns through the randomness of the data simulator.

## Pretraining

Following prior work, the model was pretrained once on simulated tabular classification datasets (Müller et al., 2022). We drew tasks from the mixed structural-causal-model simulator introduced for TabICL (Qu et al., 2025). Tasks contained between 2 and 120 active features and at most 10 classes. Each simulated task $\pmb { \tau } =$ $( \mathcal { D } _ { c } , \mathcal { D } _ { t } )$ has the context and labelled target sets defined above. For each task and sampled privacy level $\mu ,$ ${ \tilde { S } } =$ PrivateSummar ${ \bf \Gamma } _ { \phi } ( { \bf z } _ { 1 : n _ { c } } ^ { c } ; \mu )$ was produced by the same noisy summary mechanism used at prediction; only $X _ { t }$ , not target labels, was supplied to the query-side network. More details on pretraining and ablations can be found in Supplement S4 and Supplement S6.7.

Noise-aware objective. PrivTab was trained to predict target labels from noisy private summaries, using the same summary mechanism as at prediction. Let $q _ { \psi } ( y \mid \mathbf { x } , \tilde { S } )$ denote the predicted label probabilities given target features x and noisy private summary ${ \tilde { S } } .$ The main training objective is the target cross-entropy

$$
\mathcal { L } _ { \mathrm { p r i v } } = \mathbb { E } _ { \pmb { \tau } } \left[ - \sum _ { ( \mathbf { x } , y ) \in \mathcal { D } _ { t } } \log q _ { \pmb { \psi } } ( y \mid \mathbf { x } , \tilde { S } ) \right] .\tag{14}
$$

The expectation averages over simulated tasks and the Gaussian noise used to form each private summary. Conditional on ${ \tilde { S } } ,$ the query rows are processed independently, so the joint probability is the product

$$
q _ { \psi } ( y _ { 1 : n _ { t } } ^ { t } \mid X _ { t } , \tilde { S } ) = \prod _ { j = 1 } ^ { n _ { t } } q _ { \psi } ( y _ { j } ^ { t } \mid \mathbf { x } _ { j } ^ { t } , \tilde { S } ) ;\tag{15}
$$

its negative log is the sum in Eq. (14). We evaluated the efect of noise-aware pretraining in Table S32. In contrast, privacy noise in DP stochastic optimisation acts through parameter updates; it is not supplied as an explicit conditioning input when the fitted model produces label probabilities. The probabilistic interpretation of Eq. (14), including its relation to the posterior predictive conditioned on a private summary, is derived in Supplement S2; Theorem S2.1 characterises the objective for a fixed private mechanism.

Training curriculum and model combination. We pretrained two variants for small and large contexts, sampling privacy levels logarithmically and gradually introducing stronger noise. Both started with 128,000 steps using context and target sizes of $^ { 1 0 0 - 2 , 0 4 8 }$ rows, without summary normalisation. The small-context variant continued with the same context range and no normalisation for 384,000 steps in total. The large-context variant continued with contexts of 1,024–8,192 rows and summary normalisation for 256,000 steps in total. Both used summary normalisation at inference. We used the small-context variant for contexts below 4,096 rows and the large-context variant for all others. This threshold was selected on held-out context–target pairs from the TabICL simulator used for pretraining. The same two pretrained variants were used across all real datasets and privacy levels; changing µ changes the summary noise while their weights remain fixed.

Optimisation. We first pretrained the full model, then froze the private summary module late in pretraining and updated only the prediction module for the remaining steps. We optimised cross-entropy using AdamW (Loshchilov and Hutter, 2019), batches of 128 simulated datasets and gradient clipping.

## Private baselines

We compared with DP-LR and DP-MLP, the latter with one hidden layer, both trained using DP Adam (Avent et al., 2020; Bu et al., 2020a). Because the same training data are used in repeated, adaptive updates, the privacy loss must be composed over all T steps. Under the same substitute adjacency as PrivTab, weW numerically computed the privacy profile of the Poisson-subsampled Gaussian mechanism, composed its privacy-loss distribution over $T$ updates, and converted the resulting trade-of function to the smallest valid non-asymptotic µ-GDP guarantee (Gomez et al., 2026). The noise scale was chosen by numerically inverting this accounting for the target training budget; we did not use the asymptotic subsampled-Gaussian GDP approximation. The accountant and update rule are specified in Supplement S5.1.

Model selection stages. Baseline model selection combined broad calibration on separate datasets with private learning-rate selection for each evaluation dataset. First, we used Bayesian optimisation on eight classification datasets from prior work (Hegselmann et al., 2023) to select broad architectural and optimiser settings separately for each baseline and privacy level. Second, because a single learning rate was not reliable across all new datasets, we selected it privately for every evaluation dataset, baseline and privacy level. The context dataset was divided into training and validation subsets. Candidate models were trained privately on the training rows, evaluated with a private validation loss and compared under an explicitly composed learning rate selection budget. We set the privacy parameter for learning-rate selection to $\mu _ { \mathrm { s e l } } = 0 . 5 \mu$ and chose the privacy parameter $\mu _ { \mathrm { t r a i n } }$ for the final training run such that $\mu _ { \mathrm { s e l } } ^ { 2 } + \bar { \mu } _ { \mathrm { t r a i n } } ^ { 2 } = \mu ^ { 2 }$ under GDP composition.

This protocol prevented the comparison from treating data-dependent model selection as free: the privacy cost of all candidate training and validation evaluations was accounted for within the fixed learning-rate selection

budget. It also illustrates why DP stochastic optimisation is a pipeline rather than a single private optimiser call.   
Search grids, fixed hyperparameters and selected configurations are listed in Supplement S5.

## TabArena evaluation

We compared predictive performance across a broad benchmark using the same datasets, splits and privacy levels for all methods. We retrieved the 33 TabArena classification datasets from the TabArena-v0.1 benchmark suite on OpenML (Erickson et al., 2025; Bischl et al., 2025), selecting tasks with at most 100,000 rows and 120 features. This is the same number of compatible datasets from TabArena as TabPFNv2 and consistent with TabArena’s model-specific applicability constraints. The row limit kept baseline evaluation computationally tractable, while the feature limit matched the PrivTab pretraining configuration used here. Each dataset was evaluated at $\mu \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 4 , 0 . 8 , 1 . 6 \}$ using ten independent 80/20 splits into $\mathcal { D } _ { c }$ and a held-out $\mathcal { D } _ { t } ;$ only $X _ { t }$ was supplied to the predictor. All methods received the same numeric and categorical feature conversion. To isolate the private prediction mechanism in this broad benchmark, features were standardised using context-set statistics under a common idealised preprocessing convention. We evaluated the cost of private preprocessing separately, as described below.

Metrics and rankings. We evaluated classification with AUC and label-probability quality with log loss, and summarised model comparisons using Elo rankings. AUC was computed directly for binary tasks and macro-averaged one-versus-rest for multiclass tasks. Log loss (cross-entropy loss) penalises low probabilities assigned to true labels, especially confident incorrect predictions. Following TabArena (Erickson et al., 2025), maximum-likelihood Elo rankings used AUC for binary tasks and log loss for multiclass tasks. Ratings were recalibrated so that DP-LR scored 1,000 at each privacy level and in the aggregate comparison. For dataset-level breakdowns, we averaged the AUC diferences between PrivTab and each specified private baseline over the ten splits.

Uncertainty intervals. We quantified performance uncertainty using 95% confidence intervals (CIs); unless stated otherwise, figures and tables report percentile-bootstrap CIs based on 10,000 replicates. The sampling unit was the dataset for benchmark aggregates and the context–target split for within-dataset evaluations. Record-wise membership-AUC intervals were instead computed using the closed-form standard-error approximation described below.

Fitting time. Runtime measures the GPU time needed to fit the context dataset. We timed each method on one Nvidia V100 using identical software environments and hardware settings, including data transfer and computation. PrivTab required one forward pass to form the private summary; DP Adam baseline timings included private learning-rate selection and training. We reported the median per-context–target-split runtime across datasets, with a percentile-bootstrap 95% CI over datasets.

Prediction time. We separately measured post-fit prediction time on a laptop with a 13th-generation Intel Core i5 CPU across 33 datasets, six privacy levels and ten context–target splits. Predictions were evaluated in batches of up to 1,024 target rows, using one warm-up call and three timed runs per configuration. Timings exclude preprocessing, model loading, fitting and private-summary construction. The complete dataset-level results appear in Fig. S9 and Table S11.

## Private preprocessing in practice

We evaluated prediction together with private standardisation for Bank Marketing, COMPAS, ACS Income, Pima Diabetes and Maternal Health Risk datasets. Bank Marketing and Pima Diabetes were accessed through OpenML and originated in the UCI Machine Learning Repository; Maternal Health Risk was fetched directly from UCI (Bischl et al., 2025; Kelly et al., 2023). COMPAS was accessed through OpenML, and ACS Income is a Folktables task derived from U.S. Census American Community Survey data (Bischl et al., 2025; Ding et al., 2021). There are no missing values in these datasets, so that no data imputation was needed.

Private standardisation. We standardised context and query features using private estimates of means and variances, with public clipping bounds limiting each row’s influence. For each of the d input features, indexed by $k = 1 , \ldots , d ,$ we selected a finite clipping interval $[ a _ { k } , b _ { k } ]$ : numeric-feature intervals came from public descriptions, and categorical variables used publicly known ranges. For context and query rows alike $\mathbf { x } _ { i } = ( x _ { i 1 } , \dots , x _ { i d } )$ , indexed here by i, the clipped value was rescaled to

$$
c _ { i k } = \mathrm { c l i p } _ { [ 0 , 1 ] } \bigg ( \frac { x _ { i k } - a _ { k } } { b _ { k } - a _ { k } } \bigg ) .\tag{16}
$$

Here, $\mathbf { c } _ { i } = ( c _ { i 1 } , \hdots , c _ { i d } )$ denotes the rescaled and clipped row. The bounds determine the scale and cap each feature value before any private statistic is computed. From the $n _ { c }$ context rows only, we released noisy vectors of sums and sums of squares,

$$
\tilde { \mathbf { m } } _ { 1 } = \sum _ { i = 1 } ^ { n _ { c } } \mathbf { c } _ { i } + \boldsymbol { \xi } _ { 1 } , \qquad \tilde { \mathbf { m } } _ { 2 } = \sum _ { i = 1 } ^ { n _ { c } } \mathbf { c } _ { i } ^ { \odot 2 } + \boldsymbol { \xi } _ { 2 } ,\tag{17}
$$

Here the superscript $\odot 2$ means squaring each feature value, and the independent Gaussian noise vectors $\xi _ { 1 }$ and $\xi _ { 2 }$ have standard deviations for each feature of $\sqrt { d } / \mu _ { 1 }$ and $\sqrt { d } / \mu _ { 2 }$ , respectively. Replacement of one row changes either sum vector by at most $\sqrt { d }$ in Euclidean norm. We allocated $\mu _ { 1 } = \mu _ { \mathrm { p r e p } } / 2$ and $\mu _ { 2 } = \sqrt { \mu _ { \mathrm { p r e p } } ^ { 2 } - \mu _ { 1 } ^ { 2 } }$ so these two releases compose to $\mu _ { \mathrm { p r e p } } – \mathrm { G D P }$ . Writing $\tilde { m } _ { 1 k }$ and $\tilde { m } _ { 2 k }$ for the entries of these vectors at feature $k ,$ the mean and variance estimates are $\tilde { m } _ { k } = \tilde { m } _ { 1 k } / n _ { c }$ and $\tilde { v } _ { k } = \tilde { m } _ { 2 k } / n _ { c } - \tilde { m } _ { k } ^ { 2 }$ . When $\tilde { v } _ { k } \ge 1 0 ^ { - 6 }$ and the interval is non-constant, the input z-score was

$$
z _ { i k } = \frac { c _ { i k } - \tilde { m } _ { k } } { \sqrt { \tilde { v } _ { k } } } ,\tag{18}
$$

otherwise that feature value was set to zero.

Privacy guarantees with private standardisation. When clipping bounds are public, the full pipeline’s privacy guarantee follows by composing private standardisation with PrivTab’s private summary mechanism or the baseline’s private training procedure. The private standardisation in Eq. (18) was applied for both context and query rows using the released statistics from the context rows in Eq. (17). Query transformation is post-processing; the context then enters PrivTab’s private summary mechanism or the baseline’s private training procedure. Conditional on the released moments, standardisation remains row-wise, preserving one-row adjacency for the next private mechanism. Clipping bounds must also be chosen without unaccounted access to sensitive data. Our LLM-derived bounds used only public feature descriptions. For comparison, oracle bounds used empirical dataset minima and maxima; this workflow is not fully private because bound selection accesses sensitive data without privacy accounting.

Privacy-budget allocation for the case study. The case-study privacy budget covered both preprocessing and prediction or baseline fitting. The total privacy level was $\mu _ { \mathrm { t o t a l } } = 0 . 4$ . We allocated $\mu _ { \mathrm { p r e p } } = 0 . 2 4$ to the two private moment releases above and $\mu _ { \mathrm { p r e d } } = 0 . 3 2$ to PrivTab’s private summary or to the complete baseline training and model-selection pipeline, giving

$$
\mu _ { \mathrm { t o t a l } } = \sqrt { \mu _ { \mathrm { p r e p } } ^ { 2 } + \mu _ { \mathrm { p r e d } } ^ { 2 } } = 0 . 4
$$

under GDP composition. Results were based on ten independent 80/20 context–target splits. The private moment mechanism, prompts, responses, exact clipping intervals and complete numerical results are given in Supplement S6.6.

## Membership-inference audit

Attack setup. We measured membership exposure by testing whether model predictions reveal that a particular row was included in the context. We used a likelihood-ratio framework (Carlini et al., 2022) on all 1,014 Maternal Health Risk rows. We generated 4,000 balanced context draws, each containing exactly 507 rows, so that every row appeared as a member in exactly 2,000 contexts. For every draw, the model’s true-label confidence log-odds was recorded for all rows, using the same membership masks for every method.

Attack metrics. Record AUC holds the row fixed and measures the fitted Gaussian distinguishability across all contexts in which that row was or was not a member. We computed a closed-form approximation to the standard error of each record-AUC estimate (Knolle et al., 2026) and defined its lower 95% confidence endpoint as $L _ { i } = \operatorname* { m a x } \{ 0 . 5 , \widehat { \mathrm { A U C } } _ { i } - 1 . 9 6 \mathrm { S E } _ { i } \}$ . The Gaussian reference trade-of implies the attack-AUC upper bound $\Phi ( \mu / \sqrt { 2 } )$ , where Φ is the standard normal cumulative distribution function (CDF).

We also reported balanced attack accuracy to summarise how reliably membership can be inferred. We calculated balanced accuracy from fitted Gaussian member (IN) and non-member (OUT) score distributions with variances adjusted by a finite-population correction (FPC) (Jälkö et al., 2026), rather than from observed classification counts. We used the delta method to estimate its standard error at the fitted accuracy-maximising threshold, formed a nominal 95% CI and plotted the largest lower endpoint across records for each displayed method. The calculation and its assumptions are detailed in Supplement S6.1. Complementary point-estimate summaries are reported in Table S5. These attacks measure empirical exposure but cannot establish DP; they complement the mechanism-level guarantee and audit.

Finite-population correction. Sampling contexts without replacement from a fixed pool makes the empirical IN/OUT variance smaller than the variance under independent population draws. We used a finite-population factor $f = 1 - N / N _ { + }$ (Jälkö et al., 2026) and replaced each empirical fitted standard deviation by $\hat { \sigma } _ { \mathrm { F P C } } = \hat { \sigma } / \sqrt { f }$ leaving the fitted means unchanged. Here $N = 5 0 7 , N _ { + } = 1 { , } 0 1 4$ and $f = 0 . 5 ,$ , so the standard deviations were inflated by $\sqrt { 2 }$ . The corrected standard deviations were used both for the analytical record AUCs and for the held-out likelihood-ratio attack.

## Verification and auditing

We assessed the private summary mechanism using formal verification, implementation checks and empirical auditing.

Formal verification. The Lean 4 formalisation verifies the sensitivity bound and adaptive GDP composition underlying the noise calibration of the private summary mechanism. It therefore verifies these central mathematical steps independently of the manuscript proof, while the connection between the formal objects and the running system remains subject to implementation review. The lean implementation with the complete description is available in the source code at https://github.com/TrustworthyMLHelsinki/ PrivTab/tree/main/privtab-lean.

Implementation checks. Automated tests checked that the implementation applied every transformation before DP-MHCA independently to each context row, without introducing unaccounted dependencies between rows. The implementation checks are available in the source code at https: $/ / \mathrm { g } \mathrm { i }$ thub.com/ TrustworthyMLHelsinki/PrivTab/blob/main/experiments/audit\_rowwise.py. The test suite also runs this automatically.

Empirical auditing. We empirically tested the sensitivity bound and the distinguishability of private outputs. We used gradient-based optimisation to search for adjacent context datasets that maximised the distance between their pre-noise summaries. We then repeatedly sampled privatised outputs and applied the optimal likelihoodratio test for each fitted Gaussian pair, using its two noise-free means and calibrated variance. Agreement with the expected trade-of curve is evidence against several implementation errors, but it is not a substitute for the proof. Further details are provided in Supplement S6.2. The empirical auditing checks are available in the source code at https://github.com/TrustworthyMLHelsinki/PrivTab/blob/main/ experiments/dp\_audit.py. The test suite also runs this automatically.

## Code availability

The source code used to implement, train, and evaluate PrivTab, together with the pretrained model weights, is available at https: $/ / \mathrm { g } \mathrm { i }$ thub.com/TrustworthyMLHelsinki/PrivTab. The repository contains complete documentation of the code, including instructions for reproducing the experiments.

## References

Natalia Ponomareva, Hussein Hazimeh, Alex Kurakin, Zheng Xu, Carson Denison, H. Brendan McMahan, Sergei Vassilvitskii, Steve Chien, and Abhradeep Guha Thakurta. How to dp-fy ML: A practical guide to machine learning with diferential privacy. J. Artif. Intell. Res., 77:1113–1201, 2023. doi: 10.1613/JAIR.1.14649.

Nicolas Papernot and Thomas Steinke. Hyperparameter tuning with Renyi diferential privacy. In ICLR 2022, 2022.

Georgi Ganev, Meenatchi Sundaram Muthu Selva Annamalai, and Emiliano De Cristofaro. The elusive pursuit of reproducing PATE-GAN: benchmarking, auditing, debugging. Trans. Mach. Learn. Res., 2025, 2025.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam D. Smith. Calibrating noise to sensitivity in private data analysis. In TCC 2006, volume 3876 of Lecture Notes in Computer Science, pages 265–284, 2006. doi: 10.1007/11681878\_14.

Jinshuo Dong, Aaron Roth, and Weijie J. Su. Gaussian diferential privacy. J. R. Stat. Soc. Series B, 84(1):3–37, 2022. doi: 10.1111/rssb.12454.

Shuang Song, Kamalika Chaudhuri, and Anand D. Sarwate. Stochastic gradient descent with diferentially private updates. In GlobalSIP 2013, pages 245–248, 2013. doi: 10.1109/GLOBALSIP.2013.6736861.

Raef Bassily, Adam D. Smith, and Abhradeep Thakurta. Private empirical risk minimization: Eficient algorithms and tight error bounds. In FOCS 2014, pages 464–473, 2014. doi: 10.1109/FOCS.2014.56.

Martín Abadi, Andy Chu, Ian J. Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with diferential privacy. In CCS 2016, pages 308–318, 2016. doi: 10.1145/2976749.2978318.

Zeyu Ding, Yuxin Wang, Guanhong Wang, Danfeng Zhang, and Daniel Kifer. Detecting violations of diferential privacy. In CCS 2018, pages 475–489, 2018. doi: 10.1145/3243734.3243818.

Benjamin Bichsel, Samuel Stefen, Ilija Bogunovic, and Martin T. Vechev. Dp-sniper: Black-box discovery of diferential privacy violations using classifiers. In IEEE S&P 2021, pages 391–409, 2021. doi: 10.1109/SP40001.2021.00081.

Florian Tramèr, Andreas Terzis, Thomas Steinke, Shuang Song, Matthew Jagielski, and Nicholas Carlini. Debugging diferential privacy: A case study for privacy auditing. arXiv preprint arXiv:2202.12219, 2022.

Milad Nasr, Jamie Hayes, Thomas Steinke, Borja Balle, Florian Tramèr, Matthew Jagielski, Nicholas Carlini, and Andreas Terzis. Tight auditing of diferentially private machine learning. In USENIX Security 2023, pages 1631–1648, 2023.

Tudor Cebere, David Erb, Damien Desfontaines, Aurélien Bellet, and Jack Fitzsimons. Privacy in theory, bugs in practice: Grey-box auditing of diferential privacy libraries. Proc. Priv. Enhancing Technol., 2026(3):467–483, 2026. doi: 10.56553/POPETS-2026-0091.

Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. In ICLR 2023, 2023.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8044):319–326, 2025. doi: 10.1038/S41586-024-08328-6.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. In ICML 2025, volume 267 of Proceedings of Machine Learning Research, pages 50817–50847, 2025.

Fengyu Gao, Ruida Zhou, Tianhao Wang, Cong Shen, and Jing Yang. Data-adaptive diferentially private prompt synthesis for in-context learning. In ICLR 2025, 2025.

Tong Wu, Ashwinee Panda, Jiachen T. Wang, and Prateek Mittal. Privacy-preserving in-context learning for large language models. In ICLR 2024, 2024.

Alycia N. Carey, Karuna Bhaila, Kennedy Edemacu, and Xintao Wu. DP-TabICL: In-context learning with diferentially private tabular data. In IEEE BigData 2024, pages 1552–1557, 2024. doi: 10.1109/BIGDATA62323.2024.10826053.

Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. In NeurIPS 2025 Datasets and Benchmarks Track, 2025. doi: 10.52202/085713-0519.

Ossi Räisä, Stratis Markou, Matthew Ashman, Wessel P. Bruinsma, Marlon Tobaben, Antti Honkela, and Richard E. Turner. Noise-aware diferentially private regression via meta-learning. In NeurIPS 2024, 2024.

Leonardo de Moura and Sebastian Ullrich. The Lean 4 theorem prover and programming language. In Automated Deduction – CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings, page 625–635, 2021.

Markelle Kelly, Rachel Longjohn, and Kolby Nottingham. The UCI machine learning repository, 2023. URL https://archive.ics.uci.edu/.

Bernd Bischl, Giuseppe Casalicchio, Taniya Das, Matthias Feurer, Sebastian Fischer, Pieter Gijsbers, Subhaditya Mukherjee, Andreas C Müller, László Németh, Luis Oala, Lennart Purucker, Sahithya Ravi, Jan N van Rijn, Prabhant Singh, Joaquin Vanschoren, Jos van der Velde, and Marcel Wever. OpenML: Insights from 10 years and more than a thousand papers. Patterns, 6(7):101317, 2025. doi: 10.1016/j.patter.2025.101317. URL https://www.cell.com/patterns/fulltext/S2666-3899(25)00165-5.

Frances Ding, Moritz Hardt, John P. Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair machine learning. In Advances in Neural Information Processing Systems, volume 34, pages 6478–6490, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 32e54441e6382a7fbacbbbaf3c450059-Abstract.html.

Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr. Membership inference attacks from first principles. In 43rd IEEE Symposium on Security and Privacy, SP 2022, San Francisco, CA, USA, May 22-26, 2022, pages 1897–1914, 2022.

Moritz A. Knolle, Martin J. Menten, Friederike Jungmann, Felix Meissen, Ben Glocker, Daniel Rueckert, and Georgios Kaissis. Disparate privacy risks from medical AI. Nature, 656:192–198, 2026. doi: 10.1038/s41586-026-10688-0.

Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, S. M. Ali Eslami, and Yee Whye Teh. Neural processes. arXiv preprint arXiv:1807.01622, 2018a.

Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Jimenez Rezende, and S. M. Ali Eslami. Conditional neural processes. In ICML 2018, volume 80 of Proceedings ofMachine Learning Research, pages 1704–1713, 2018b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS 2017, pages 5998–6008, 2017.

Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, S. M. Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. In ICLR 2019, 2019.

Tung Nguyen and Aditya Grover. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. In ICML 2022, volume 162 of Proceedings of Machine Learning Research, pages 16569–16594, 2022.

Leo Feng, Hossein Hajimirsadeghi, Yoshua Bengio, and Mohamed Osama Ahmed. Latent bottlenecked attentive neural processes. In ICLR 2023, 2023.

Samuel Müller, Noah Hollmann, Sebastian Pineda-Arango, Josif Grabocka, and Frank Hutter. Transformers can do bayesian inference. In ICLR 2022, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR 2019, 2019.

Brendan Avent, Javier González, Tom Diethe, Andrei Paleyes, and Borja Balle. Automatic discovery of privacy–utility pareto fronts. Proceedings on Privacy Enhancing Technologies, 2020(4):5–23, 2020. doi: 10.2478/popets-2020-0060.

Zhiqi Bu, Jinshuo Dong, Qi Long, and Weijie J. Su. Deep learning with gaussian diferential privacy. Harvard Data Science Review, 2(3), 2020a. doi: 10.1162/99608f92.cfc5dd25.

Juan Felipe Gomez, Bogdan Kulynych, Georgios Kaissis, Flavio P. Calmon, Jamie Hayes, Borja Balle, and Antti Honkela. Position: Gaussian DP for reporting diferential privacy guarantees in machine learning. In SaTML 2026, 2026.

Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David A. Sontag. Tabllm: Few-shot classification of tabular data with large language models. In AISTATS 2023, volume 206 of Proceedings ofMachine Learning Research, pages 5549–5581, 2023.

Joonas Jälkö, Gauri Pradhan, Ossi Räisä, and Antti Honkela. On reliability of membership inference vulnerability evaluation. arXiv preprint arXiv:2605.25819, 2026.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam D. Smith. Calibrating noise to sensitivity in private data analysis. J. Priv. Confidentiality, 7(3):17–51, 2016. doi: 10.29012/JPC.V7I3.405.

Bernt Øksendal. Some mathematical preliminaries. In Stochastic Diferential Equations: An Introduction with Applications, pages 7–20. Springer, 2010. doi: 10.1007/978-3-642-14394-6.

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and João Carreira. Perceiver: General perception with iterative attention. In ICML 2021, volume 139 of Proceedings of Machine Learning Research, pages 4651–4664, 2021.

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier J. Hénaf, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and João Carreira. Perceiver IO: A general architecture for structured inputs & outputs. In ICLR 2022, 2022.

Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Mihir Manium, Shi Bin Hoo, Magnus Bühler, Anurag Garg, Dominik Safaric, Jake Robertson, Benjamin Jäger, Simone Alessi, Adrian Hayler, Vladyslav Moroshan, Lennart Purucker, Philipp Singer, Alan Arazi, Julien Siems, Jan Hendrik Metzen, Georg Grab, Nick Erickson, Siyuan Guo, Eliott Kalfon, Simon Bing, David Salinas, Clara Cornu, Lilly Charlotte Wehrhahn, Diana Kriuchkova, Kursat Kaya, Lydia Sidhoum, Marie Salmon, Jerry Chen, Madelon Hulsebos, Yann LeCun, Samuel Müller, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter. TabPFN-3: Technical report. arXiv preprint arXiv:2605.13986, 2026.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICLv2: A better, faster, scalable, and open tabular foundation model. In ICML 2026, 2026.

John Bronskill. Data and Computation Eficient Meta-Learning. PhD thesis, University of Cambridge, 2020.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In ICML 2017, volume 70 of Proceedings ofMachine Learning Research, pages 1126–1135, 2017.

Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999, 2018.

Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. In NeurIPS 2016, pages 3630–3638, 2016.

Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In NeurIPS 2017, pages 4077–4087, 2017.

Jefrey Li, Mikhail Khodak, Sebastian Caldas, and Ameet Talwalkar. Diferentially private meta-learning. In ICLR 2020, 2020.

Xinyu Zhou and Raef Bassily. Task-level diferentially private meta learning. In NeurIPS 2022, 2022.

Xinyu Tang, Richard Shin, Huseyin A. Inan, Andre Manoel, Fatemehsadat Mireshghallah, Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, and Robert Sim. Privacy-preserving in-context learning with diferentially private few-shot generation. In ICLR 2024, 2024.

Antti Koskela, Tejas D. Kulkarni, and Laith Y. Zumot. Diferentially private in-context learning with nearest neighbor search. arXiv preprint arXiv:2511.04332, 2025.

Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. Flocks of stochastic parrots: Diferentially private prompt learning for large language models. In NeurIPS 2023, 2023.

Dina El Zein and James Henderson. Diferential privacy for transformer embeddings of text with nonparametric variational information bottleneck. arXiv preprint arXiv:2601.02307, 2026.

Lei Jimmy Ba, Jamie Ryan Kiros, and Geofrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Noam Shazeer. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415, 2016.

Christian Szegedy, Vincent Vanhoucke, Sergey Iofe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR 2016, pages 2818–2826, 2016. doi: 10.1109/CVPR.2016.308.

Zhiqi Bu, Jinshuo Dong, Qi Long, and Weijie J. Su. Deep learning with Gaussian diferential privacy. Harvard Data Science Review, 2(3), 2020b. doi: 10.1162/99608f92.cfc5dd25.

Frank McSherry. Privacy integrated queries: an extensible platform for privacy-preserving data analysis. Commun. ACM, 53(9):89–97, 2010. doi: 10.1145/1810891.1810916.

## Acknowledgements

This work was supported by the Research Council of Finland (Finnish Center for Artificial Intelligence, FCAI, Grant 356499 and Grant 359111), the Strategic Research Council at the Research Council of Finland (Grant 358247) as well as the European Union (Project 101070617). Richard E. Turner is supported by the EPSRC Probabilistic AI Hub (EP/Y028783/1). Cristiana Diaconu is supported by the Cambridge Trust Scholarship. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the granting authority can be held responsible for them. This work has been performed using resources provided by the CSC– IT Center for Science, Finland (Projects 2003275 & 462001244). We also acknowledge CSC for access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium. The authors acknowledge the research environment provided by ELLIS Institute Finland. We thank Aki Rehn for his assistance in debugging the GPU-based pretraining of PrivTab. We also thank Joonas Jälkö, Mikko A. Heikkilä and Marlon Tobaben for their discussions and comments throughout the project.

## Extended figures

![](images/cfc6bb8e0bdfc301e2fc5e972248307f8386dcd4f359a810235a821580f29a5e.jpg)  
Figure E1: Detailed PrivTab architecture. Context and target rows are first mapped to row embeddings by RowEmbed. In PrivateSummary, a small set of data-independent learnable pseudo-tokens cross-attends to the sensitive context rows through DP-MHCA, producing privatised summaries. The public Prediction module reuses these summaries through cross-attention and token-wise multilayer perceptrons. DP-MHCA is the only operation that transfers context information into the released summaries; target-side computation is post-processing.

## Additional information

Correspondence: Antti Honkela (antti.honkela@helsinki.fi).

## Supplement

## S1 Detailed Background

## S1.1 Notation

Let $d _ { x } \in \mathbb { N }$ denote the maximum input dimension, let $\begin{array} { r } { \mathcal { X } : = \bigcup _ { d = 1 } ^ { d _ { x } } \mathbb { R } ^ { d } } \end{array}$ denote the raw input space, and let $\mathcal { V } \subset \mathbb { N }$ denote the label space. An input-label pair is a tuple $( \mathbf { x } , y ) \in \mathcal { X } \times \mathcal { Y }$ . Let D denote the space of all finite multisets of input-label pairs in $\mathcal { X } \times \mathcal { V }$ , that is,

$$
\mathbb { D } = \bigcup _ { n = 0 } ^ { \infty } \left( \left( \mathcal { X } \times \mathcal { Y } \right) ^ { n } / { \sim } \right) ,\tag{S1}
$$

where two elements of $( \mathcal { X } \times \mathcal { Y } ) ^ { n }$ are equivalent under if one is a permutation of the other. A dataset $\mathcal { D } \in \mathbb { D }$ is a finite collection of input-label pairs, i.e. $\mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ for some $N \geq 0 .$ For $n \geq 1$ , we will write a sequence $z _ { 1 } , \ldots , z _ { n }$ briefly as $z _ { 1 : n }$ . Whenever indexed notation is used for a multiset, it denotes an arbitrary ordered representative; all dataset-level constructions considered below are invariant to the choice of representative.

## S1.2 Diferential Privacy

DP formalises the idea of protecting individuals’ sensitive data when releasing statistics or training machine learning models. It is achieved by limiting how much the algorithm’s output distribution can change when one individual’s data is added, removed, or modified.

Definition S1.1 (Approximate DP; Dwork et al. 2006, 2016). A randomised algorithm or mechanism is $( \varepsilon , \delta , \simeq )$ DP under a binary relation over D, if for any two datasets $\mathcal { D } , \mathcal { D } ^ { \prime } \in \mathbb { D }$ such that $\mathcal { D } \simeq \mathcal { D } ^ { \prime }$ , and for any subset of possible outcomes of ${ \mathcal { A } } ,$ denoted ${ \mathcal { S } } \subseteq \operatorname { R a n g e } ( A )$ ,

$$
\operatorname* { P r } \left[ { A } ( { \mathcal { D } } ) \in { \mathcal { S } } \right] \leq \exp \left( { \varepsilon } \right) \times \operatorname* { P r } \left[ { A } ( { \mathcal { D } } ^ { \prime } ) \in { \mathcal { S } } \right] + \delta .\tag{S2}
$$

Any two datasets $\mathcal { D } \simeq \mathcal { D } ^ { \prime }$ are said to be adjacent datasets and  is called the adjacency relation.

Applying any (randomised) function to the output of a DP mechanism is called post-processing, and it does not weaken the privacy guarantees of the mechanism. In this work, we use the substitute adjacency relation, that is, $\mathcal { D } \simeq \mathcal { D } ^ { \prime }$ if for some $i ^ { \ast } , \mathcal { D } ^ { \prime } = ( \mathcal { D } \setminus \{ ( \mathbf { x } _ { i ^ { \ast } } , y _ { i ^ { \ast } } ) \} ) \cup \{ ( \mathbf { x } _ { i ^ { \ast } } ^ { \prime } , y _ { i ^ { \ast } } ^ { \prime } ) \}$

## S1.3 Gaussian Diferential Privacy

GDP provides a more interpretable characterisation of privacy in terms of hypothesis testing. For a random output O of the mechanism ${ \mathcal { A } } ,$ consider the binary hypothesis test: $H _ { 0 } : O \sim { \mathcal { A } } ( { \mathcal { D } } )$ vs. $H _ { 1 } : O \sim \mathcal { A } ( \mathcal { D } ^ { \prime } )$ . A test is a function

$$
\phi : { \mathrm { R a n g e } } ( A )  [ 0 , 1 ] ,\tag{S3}
$$

where for an observation ${ \bf o } , \phi ( { \bf o } ) = 1$ means rejecting the null hypothesis, and $\phi ( \mathbf { o } ) = 0$ means not rejecting it. The type I error is denoted $\alpha _ { \mathcal { A } } ( \phi ; \mathcal { D } ) = \mathbb { E } _ { O \sim \mathcal { A } ( \mathcal { D } ) } \left[ \phi ( O ) \right]$ and the type II error is denoted $\beta _ { A } ( \phi ; { \mathcal { D } } ^ { \prime } ) =$ $1 - \mathbb { E } _ { O \sim A ( D ^ { \prime } ) } \left[ \phi ( O ) \right]$ . The trade-offunction of is defined as:

$$
T _ { \cal A } ( \alpha ; \mathcal { D } , \mathcal { D } ^ { \prime } ) = \operatorname* { i n f } _ { \phi : \alpha _ { \cal A } ( \phi ; \mathcal { D } ) \le \alpha } \beta _ { \cal A } ( \phi ; \mathcal { D } ^ { \prime } ) .\tag{S4}
$$

Definition S1.2 $( \mu { \mathrm { - G D P } }$ ; Dong et al. 2022). A randomised mechanism satisfies $\mu { \mathrm { - } } \mathrm { G D P }$ if for any two adjacent datasets $\mathcal { D } \simeq \mathcal { D } ^ { \prime }$ , and for all $\alpha \in [ 0 , 1 ]$

$$
\begin{array} { r } { T _ { A } ( \alpha ; \mathcal { D } , \mathcal { D } ^ { \prime } ) \ge \Phi \left( \Phi ^ { - 1 } ( 1 - \alpha ) - \mu \right) = : T _ { \mu } ( \alpha ) , } \end{array}\tag{S5}
$$

where Φ denotes the standard normal CDF.

The function $T _ { \mu } ( \alpha )$ is the trade-of function of the hypothesis test $H _ { 0 } : O \sim \mathcal { N } ( 0 , 1 )$ vs. $H _ { 1 } : O \sim { \mathcal { N } } ( \mu , 1 )$ ) In this work, we will use $\mu { \mathrm { - } } \mathrm { G D P }$ to report privacy guarantees. $\mathrm { A \ } \mu { \mathrm { - G D P } }$ mechanism satisfies $( \varepsilon , \delta ( \varepsilon ) ) \ / – \mathrm { D P }$ for all $\varepsilon > 0$ (Dong et al., 2022), where

$$
\delta ( \varepsilon ) = \Phi \left( - \frac { \varepsilon } { \mu } + \frac { \mu } { 2 } \right) - \exp ( \varepsilon ) \Phi \left( - \frac { \varepsilon } { \mu } - \frac { \mu } { 2 } \right) .\tag{S6}
$$

Applying a DP mechanism multiple times, or applying multiple DP mechanisms to the same dataset, is called composition. If later mechanism applications use outputs of earlier ones in addition to the dataset, the composition is said to be adaptive. GDP admits a tight (adaptive) composition rule determined by the privacy guarantees of individual mechanisms.

Theorem S1.3 (Dong et al. 2022). The (adaptive) composition of $\mathsf { \Pi } _ { \mu _ { i } - G D P }$ mechanisms $( i = 1 , \ldots , T )$ is $\mu { - } G D P ,$ where

$$
\mu = \sqrt { \mu _ { 1 } ^ { 2 } + . . . + \mu _ { T } ^ { 2 } } .\tag{S7}
$$

Here, tight means that this composition rule is exact for $\mu { \mathrm { - G D P } } ;$ in general one cannot replace $\sqrt { \mu _ { 1 } ^ { 2 } + \cdots + \mu _ { T } ^ { 2 } }$ by a uniformly smaller value and still retain a valid guarantee for all adaptive compositions of $\mu _ { i } – \mathrm { G D P }$ mechanisms. The Gaussian mechanism is central to this work. Given a function $f : \mathbb { D } \to \mathbb { R } ^ { d }$ , define

$$
\mathcal { A } _ { \mathrm { g a u s s } } ( \mathcal { D } ) : = f ( \mathcal { D } ) + \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbf { I } _ { d } ) ,\tag{S8}
$$

where $\mathbf { I } _ { d }$ is the d  d identity matrix. The privacy guarantee of $\mathcal { A } _ { \mathrm { g a u s s } }$ is determined by $\sigma$ and the $\ell _ { 2 } \cdot$ -sensitivity of $f _ { i }$ ,

$$
\Delta ( f ) = \operatorname* { s u p } _ { { \mathcal { D } } \simeq { \mathcal { D } ^ { \prime } } } \| f ( { \mathcal { D } } ) - f ( { \mathcal { D } } ^ { \prime } ) \| _ { 2 } ,\tag{S9}
$$

according to the following theorem.

Theorem S1.4 (Dong et al. 2022). The Gaussian mechanism $\mathcal { A } _ { \mathrm { g a u s s } }$ is $\mu { - } G D P$ with $\begin{array} { r } { \mu = \frac { \Delta ( f ) } { \sigma } } \end{array}$

## S1.4 Bayesian Inference

Assume that a dataset $\mathcal { D } = \{ ( \mathbf { x } _ { i } ^ { \mathrm { o b s } } , y _ { i } ^ { \mathrm { o b s } } ) \} _ { i = 1 } ^ { N } \in \mathbb { D }$ consists of observed input-output pairs. We treat the inputs $\mathbf { x } _ { 1 : N } ^ { \mathrm { o b s } }$ as given, and assume that the labels are generated from an underlying conditional data-generating distribution $p ( \boldsymbol { y } \mid \mathbf { x } , \pmb { \theta } )$ , depending on some parameter $\pmb \theta \in \Theta$ . The likelihood of the observed labels, conditioned on the observed inputs, is

$$
p ( \boldsymbol { y } _ { 1 : N } ^ { \mathrm { o b s } } \mid \mathbf { x } _ { 1 : N } ^ { \mathrm { o b s } } , \pmb { \theta } ) = \prod _ { i = 1 } ^ { N } p ( \boldsymbol { y } _ { i } ^ { \mathrm { o b s } } \mid \mathbf { x } _ { i } ^ { \mathrm { o b s } } , \pmb { \theta } ) .\tag{S10}
$$

Given a prior $p ( \pmb \theta )$ , one can compute the posterior using Bayes’ theorem:

$$
p ( \pmb \theta | \mathcal { D } ) = \frac { p ( y _ { 1 : N } ^ { \mathrm { o b s } } \mid \mathbf { x } _ { 1 : N } ^ { \mathrm { o b s } } , \pmb \theta ) p ( \pmb \theta ) } { \int _ { \Theta } p ( y _ { 1 : N } ^ { \mathrm { o b s } } \mid \mathbf { x } _ { 1 : N } ^ { \mathrm { o b s } } , \pmb \theta ^ { \prime } ) p ( \pmb \theta ^ { \prime } ) \mathrm { d } \pmb \theta ^ { \prime } } .\tag{S11}
$$

The posterior predictive can then be obtained by marginalising over the posterior:

$$
p ( y \mid \mathbf { x } , \mathcal { D } ) = \int _ { \Theta } p ( y \mid \mathbf { x } , \pmb \theta ) p ( \pmb \theta \mid \mathcal { D } ) \mathrm { d } \pmb \theta .\tag{S12}
$$

The posterior predictive in $\mathtt { E q }$ . (S12) can be extended to multiple query points. Assuming that the labels are conditionally independent given $\mathbf { x } _ { 1 : n }$ and $\pmb \theta ,$ for $n \geq 1$ we have

$$
\begin{array} { r } { p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \mathcal { D } ) = \int _ { \Theta } p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \pmb { \theta } ) p ( \pmb { \theta } \mid \mathcal { D } ) \mathrm { d } \pmb { \theta } } \\ { = \int _ { \Theta } \displaystyle \prod _ { i = 1 } ^ { n } p ( y _ { i } \mid \mathbf { x } _ { i } , \pmb { \theta } ) p ( \pmb { \theta } \mid \mathcal { D } ) \mathrm { d } \pmb { \theta } . } \end{array}\tag{S13}
$$

## S1.5 Stochastic Processes

A stochastic process (SP) is a collection of random variables

$$
F = \{ Y _ { \mathbf { x } } \} _ { \mathbf { x } \in \mathcal { X } } ,\tag{S14}
$$

indexed by an input space $x ,$ where each $Y _ { \mathbf { x } }$ takes values in an output space $\mathcal { V } .$ . We also write $F ( \mathbf { x } )$ for the random variable $Y _ { \mathbf { x } }$ . Thus, for any $\mathbf { x } \in \mathcal { X }$ , the stochastic process defines a distribution over outputs in $\mathcal { V }$ . If this distribution admits a density, we denote by $p ( \boldsymbol { y } \mid \mathbf { x } )$ the density of $F ( \mathbf { x } )$ evaluated at $y \in \mathcal { V } .$

We denote by $\mathcal { S P } ( \mathcal { X } , \mathcal { Y } )$ the set of stochastic processes indexed by and taking values in . A stochastic process $F \in S \mathcal { P } ( \mathcal { X } , \mathcal { Y } )$ can be characterised by its finite-dimensional marginal distributions $p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } )$ , for all $n \geq 1$ , provided that these marginals satisfy the conditions of the Kolmogorov consistency theorem (Øksendal, 2010). That is, the finite-dimensional marginals must satisfy permutation invariance and marginalisation consistency.

## S1.6 Amortised Bayesian Prediction

Let $F = \{ Y _ { \mathbf { x } } \} _ { \mathbf { x } \in \mathcal { X } } \in \mathcal { S P } ( \mathcal { X } , \mathcal { Y } )$ be a stochastic process. Assume that its finite-dimensional marginal distributions are parameterised by some parameter $\pmb \theta \in \Theta$ , so that for any $n \geq 1$

$$
Y _ { \mathbf { x } _ { 1 } } , \hdots , Y _ { \mathbf { x } _ { n } } \sim p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \pmb { \theta } ) .\tag{S15}
$$

A dataset $\mathcal { D } = \{ ( \mathbf { x } _ { i } ^ { \mathrm { o b s } } , y _ { i } ^ { \mathrm { o b s } } ) \} _ { i = 1 } ^ { N }$ can be viewed as observations of the stochastic process at input locations $\mathbf { x } _ { 1 } ^ { \mathrm { o b s } } , \ldots , \mathbf { x } _ { N } ^ { \mathrm { o b s } }$ . Conditioned on these input locations and on $\theta ,$ the corresponding outputs are sampled as

$$
y _ { 1 : N } ^ { \mathrm { o b s } } \sim p ( y _ { 1 : N } \mid \mathbf { x } _ { 1 : N } ^ { \mathrm { o b s } } , \pmb { \theta } ) ,\tag{S16}
$$

where this distribution is the finite-dimensional marginal distribution of $Y _ { \mathbf { x } _ { 1 } ^ { \mathrm { o b s } } } , \ldots , Y _ { \mathbf { x } _ { N } ^ { \mathrm { o b s } } }$

Given a prior $p ( \pmb \theta )$ , Bayes’ theorem gives the posterior $p ( \pmb { \theta } \mid \mathbf { \mathcal { D } } )$ . The posterior predictive distribution for any $n \geq 1$ is then

$$
p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \mathcal { D } ) = \int _ { \Theta } p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \pmb \theta ) p ( \pmb \theta \mid \mathcal { D } ) \mathrm d \pmb \theta .\tag{S17}
$$

The collection of posterior predictive distributions $\{ p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , { \cal D } ) \} _ { n > 1 }$ defines a posterior stochastic process, denoted by $F _ { \mathcal { D } } \in { \cal S } \mathcal { P } ( \mathcal { X } , \mathcal { Y } )$ .

More generally, one can view prediction under uncertainty as learning an amortised approximation to the Bayesian map that sends an observed dataset to its posterior predictive law. Given a dataset , the ideal Bayesian map

$$
\pi _ { F } : \mathbb { D } \to S \mathcal { P } ( \mathcal { X } , \mathcal { Y } )\tag{S18}
$$

returns the posterior stochastic process $F _ { \mathcal { D } }$ , whose finite-dimensional marginals are the posterior predictive distributions. Amortised Bayesian prediction aims to learn a parametric map that directly approximates this posterior predictive object from observed context data, without performing Bayesian inference from scratch for each new task. Concretely, given a context dataset $\mathcal { D } _ { c } .$ , one learns a predictor of the form

$$
q _ { \mathbf { w } } ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \mathcal { D } _ { c } ) , \quad n \geq 1 ,\tag{S19}
$$

which approximates the corresponding finite-dimensional posterior predictive distributions. This viewpoint is broad and includes neural processes and their attention-based variants as special cases, as well as other transformer-based amortised predictors.

We denote by $\pmb { \tau } = ( \mathcal { D } _ { c } , \mathcal { D } _ { t } )$ a task, consisting of a context set $\mathcal { D } _ { c }$ and a target set

$$
\mathcal { D } _ { t } = \left\{ \left( \mathbf { x } _ { j } ^ { t } , y _ { j } ^ { t } \right) \right\} _ { j = 1 } ^ { n _ { t } } .\tag{S20}
$$

For a task $\pmb { \tau } = ( \mathcal { D } _ { c } , \mathcal { D } _ { t } )$ with $\mathcal { D } _ { t } ~ = ~ ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , y _ { 1 : n _ { t } } ^ { t } )$ , we write $\mathbf { x } _ { 1 : n _ { t } } ^ { t } ( \pmb { \tau } ) : = \mathbf { x } _ { 1 : n _ { t } } ^ { t }$ and $y _ { 1 : n _ { t } } ^ { t } ( \pmb { \tau } ) : = y _ { 1 : n _ { t } } ^ { t }$ . Let $\mathbb { T } \subseteq \mathbb { D } \times \mathbb { D }$ denote the space of such tasks. Tasks are sampled from a task distribution $p ( \tau )$ . This task distribution can be understood as the marginal distribution induced by first sampling a latent parameter θ and then sampling context and target observations from the corresponding stochastic process:

$$
p ( \pmb { \tau } ) = \int _ { \Theta } p ( \pmb { \tau } \mid \pmb { \theta } ) p ( \pmb { \theta } ) \mathrm { d } \pmb { \theta } .\tag{S21}
$$

More explicitly, writing $\pmb { \tau } = ( \mathcal { D } _ { c } , \mathcal { D } _ { t } )$ with $\mathcal { D } _ { c } = ( \mathbf { x } _ { 1 : n _ { c } } ^ { c } , y _ { 1 : n _ { c } } ^ { c } )$ and $\mathcal { D } _ { t } = ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , y _ { 1 : n _ { t } } ^ { t } )$ , we have

$$
p ( \pmb { \tau } \mid \pmb { \theta } ) = p ( y _ { 1 : n _ { c } } ^ { c } , y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { c } } ^ { c } , \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \pmb { \theta } ) p ( \mathbf { x } _ { 1 : n _ { c } } ^ { c } , \mathbf { x } _ { 1 : n _ { t } } ^ { t } \mid \pmb { \theta } ) .\tag{S22}
$$

Under the usual conditional independence assumption given $\pmb \theta ,$ this becomes

$$
p ( \pmb { \tau } \mid \pmb { \theta } ) = \left[ \prod _ { i = 1 } ^ { n _ { c } } p ( y _ { i } ^ { c } \mid \mathbf { x } _ { i } ^ { c } , \pmb { \theta } ) \right] \left[ \prod _ { j = 1 } ^ { n _ { t } } p ( y _ { j } ^ { t } \mid \mathbf { x } _ { j } ^ { t } , \pmb { \theta } ) \right] p ( \mathbf { x } _ { 1 : n _ { c } } ^ { c } , \mathbf { x } _ { 1 : n _ { t } } ^ { t } \mid \pmb { \theta } ) .\tag{S23}
$$

The task τ itself contains only the observed context and target sets; the latent parameter $\pmb \theta$ is not observed by the amortised predictor.

Let $q _ { \pmb { \psi } } ( y _ { 1 : n _ { t } } ^ { t } \ | \ \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \mathcal { D } _ { c } )$ denote the amortised predictive family. The amortised predictor is trained by minimising the expected negative log-likelihood of the target outputs conditioned on the target inputs and the context set:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N L L } } ( \psi ) = \mathbb { E } _ { \tau \sim p ( \tau ) } \left[ - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \mathcal { D } _ { c } \right) \right] . } \end{array}\tag{S24}
$$

If the target labels are conditionally independent under the model, this objective becomes

$$
\mathcal { L } _ { \mathrm { N L L } } ( \psi ) = \mathbb { E } _ { \tau \sim p ( \tau ) } \left[ - \sum _ { j = 1 } ^ { n _ { t } } \log q _ { \psi } \left( y _ { j } ^ { t } \mid \mathbf { x } _ { j } ^ { t } , \mathcal { D } _ { c } \right) \right] .\tag{S25}
$$

In practice, the expectation over tasks is approximated using finitely many sampled tasks

$$
\left\{ \left( \mathcal { D } _ { c } ^ { \left( i \right) } , \mathcal { D } _ { t } ^ { \left( i \right) } \right) \right\} _ { i = 1 } ^ { N _ { \mathrm { t a s k s } } } ,
$$

where

$$
\begin{array} { r } { \mathcal { D } _ { t } ^ { ( i ) } = \left\{ \left( \mathbf { x } _ { j } ^ { t , ( i ) } , y _ { j } ^ { t , ( i ) } \right) \right\} _ { j = 1 } ^ { n _ { t } } . } \end{array}\tag{S26}
$$

This gives the empirical objective

$$
\widehat { \mathcal { L } } _ { \mathrm { N L L } } ( \psi ) = - \frac { 1 } { N _ { \mathrm { t a s k s } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t a s k s } } } \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t , ( i ) } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t , ( i ) } , \mathcal { D } _ { c } ^ { ( i ) } \right) .\tag{S27}
$$

Under the conditional independence factorisation, this becomes

$$
\widehat { \mathcal { L } } _ { \mathrm { N L L } } ( \psi ) = - \frac { 1 } { N _ { \mathrm { t a s k s } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t a s k s } } } \sum _ { j = 1 } ^ { n _ { t } } \log q _ { \psi } \left( y _ { j } ^ { t , ( i ) } \mid \mathbf { x } _ { j } ^ { t , ( i ) } , \mathcal { D } _ { c } ^ { ( i ) } \right) .\tag{S28}
$$

## S1.7 Neural Processes and Conditional Neural Processes

Neural processes (Garnelo et al., 2018a) are a particular family of amortised predictors. In this work, we focus on the conditional neural process (CNP) formulation (Garnelo et al., 2018b). A CNP directly parameterises the finite-dimensional predictive distributions conditioned on a context dataset

$$
\mathcal { D } _ { c } = \{ ( \mathbf { x } _ { i } ^ { c } , y _ { i } ^ { c } ) \} _ { i = 1 } ^ { n _ { c } } .\tag{S29}
$$

Given such a dataset, a neural network with learnable weights $\mathbf { w } \in \mathcal { W }$ defines

$$
q _ { \mathbf { w } } ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \mathcal { D } _ { c } ) , \quad n \geq 1 ,\tag{S30}
$$

as an approximation to the finite-dimensional marginals of $F _ { \mathcal { D } _ { c } } = \pi _ { F } ( \mathcal { D } _ { c } )$ . In practice, neural processes are implemented using an encoder-decoder architecture. Each context pair $\left( \mathbf { x } _ { i } ^ { c } , y _ { i } ^ { c } \right)$ is first mapped to a representation $r _ { i }$ by an encoder network. These context representations are then aggregated into a global summary r using a permutation-invariant operation such as averaging or summation. A decoder network finally combines this summary with a target input x to parameterise the predictive distribution $q _ { \mathbf { w } } ( y \mid \mathbf { x } , \mathcal { D } _ { c } )$ . In a CNP, this aggregation step is the main architectural mechanism by which information from the context set is shared across target predictions.

Latent neural processes may instead introduce a task variable to represent uncertainty over the underlying task. PrivTab follows the deterministic conditional formulation; its attention-based latent bottleneck and connection to prior-fitted prediction are described next.

## S1.8 Transformer Neural Processes

While CNPs provide an amortised approximation to posterior prediction, their predictive performance can be limited by the use of simple permutation-invariant aggregation schemes for the context set. Such aggregators compress the context into a fixed-size representation, but may discard information about interactions among context points and about the relevance of diferent context rows to a given target query. Transformer neural processes address this limitation by replacing hand-crafted aggregation with attention-based set processing.

Transformers and Attention. At a high level, a transformer is a neural architecture built from attention layers and token-wise feed-forward layers (Vaswani et al., 2017). Given a collection of token representations, an attention layer updates each token by combining information from other tokens using data-dependent weights. This allows the representation of each token to depend adaptively on the rest of the set.

For a query token representation q $\in \mathbb { R } ^ { d }$ , a collection of key representations $\mathbf { k } _ { 1 } , \ldots , \mathbf { k } _ { n } \in \mathbb { R } ^ { d }$ , and corresponding value representations $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { n } \in \mathbb { R } ^ { d _ { v } }$ , attention produces an output of the form

$$
\operatorname { A t t n } ( \mathbf { q } , \mathbf { k } _ { 1 : n } , \mathbf { v } _ { 1 : n } ) = \sum _ { i = 1 } ^ { n } a _ { i } ( \mathbf { q } , \mathbf { k } _ { 1 : n } ) \mathbf { v } _ { i } .\tag{S31}
$$

In the transformer architecture, the coeficients $a _ { 1 } , \ldots , a _ { n }$ are defined by applying a softmax to scaled dot-product similarity scores between the query and the keys (Vaswani et al., 2017). In practice, the query, key, and value representations are themselves obtained from token embeddings through learned linear transformations. More explicitly, in this case

$$
a _ { i } ( \mathbf { q } , \mathbf { k } _ { 1 : n } ) = \frac { \exp \left( \langle \mathbf { q } , \mathbf { k } _ { i } \rangle / \sqrt { d } \right) } { \sum _ { j = 1 } ^ { n } \exp \left( \langle \mathbf { q } , \mathbf { k } _ { j } \rangle / \sqrt { d } \right) } , \qquad i = 1 , \dots , n ,\tag{S32}
$$

where $\left. \mathbf { q } , \mathbf { k } _ { i } \right.$ denotes the Euclidean inner product between q and $\mathbf { k } _ { i }$

In self-attention, the query token and the key-value pairs all come from the same set of tokens, so each token can incorporate information from the others in that set. In cross-attention, the query comes from one set and the key-value pairs come from another, allowing one set of tokens to attend to another. Multi-head attention applies several such attention operations in parallel and combines their outputs, allowing the model to represent diferent kinds of interactions simultaneously.

For set-structured data, attention-based architectures are especially attractive because they can model rich interactions among elements while remaining permutation-equivariant when positional encodings are not used. This makes transformers a natural architectural choice for neural processes on tabular datasets.

Row-order symmetry. The row order of a tabular dataset is arbitrary, so a tabular predictor should not depend on the order in which context rows are presented. Let $P _ { c }$ and $P _ { t }$ denote permutation matrices acting on the context and target-row axes, respectively. A set-to-set predictor f is context-row invariant and target-row equivariant when

$$
f \left( P _ { c } \mathbf { \mathcal { D } } _ { c } , P _ { t } \mathbf { x } _ { 1 : n _ { t } } ^ { t } \right) = P _ { t } f \left( \mathbf { \mathcal { D } } _ { c } , \mathbf { x } _ { 1 : n _ { t } } ^ { t } \right) .\tag{S33}
$$

Consequently, permuting context rows leaves every prediction unchanged, while permuting target rows only permutes the corresponding predictions. Token-wise maps, including MLPs and layer normalisation, preserve this property; self-attention is permutation equivariant; and cross-attention is invariant to a permutation of its key-value rows and equivariant to a permutation of its query rows. Finally, a loss that aggregates the per-target losses by summation or averaging is invariant to target-row order. Thus, a row-order-equivariant architecture together with an aggregated loss defines an order-invariant learning objective.

Feature order is diferent. Columns have fixed coordinates in an input vector, and a generic row encoder need not be invariant to a permutation of those coordinates. Exact feature-order invariance therefore requires an additional architectural constraint or explicit symmetrisation; it does not follow from row-order symmetry.

Transformer Neural Processes. The attentive neural process augments the CNP by replacing simple pooling with attention from the target inputs to encoded context representations (Kim et al., 2019). In this architecture, the context rows are first encoded independently, and each target input then attends to these encoded context rows to construct a target-dependent summary. This addresses a key limitation of pooled encoders: diferent target inputs can focus on diferent parts of the context set.

Transformer neural processes (TNPs) extend this idea by introducing attention earlier in the encoder itself (Nguyen and Grover, 2022). Rather than encoding context rows independently and only applying attention at prediction time, a TNP uses attention among context representations as part of the encoding process, allowing the context representation itself to capture interactions among rows before target-dependent prediction is performed. Context and target rows are then mapped to token representations, and self-attention and cross-attention are used to construct target-dependent summaries of the context set. In this way, a TNP can model both interactions within the context set and relevance of context rows to each target input.

$$
q _ { \mathbf { w } } ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \mathcal { D } _ { c } )\tag{S34}
$$

through an attention-based architecture operating on the context and target sets. The context set provides observed input-label pairs, while the target set provides query inputs whose labels are to be predicted. Attention allows the model to form representations that depend jointly on the target inputs and the observed context, making TNPs a natural neural-process analogue of target-dependent posterior prediction.

Several recent architectures improve the computational eficiency of attention-based neural processes by introducing a smaller latent bottleneck between the context set and the target predictions. Latent bottleneck attentive neural processes (LBANPs) compress the context information into a fixed set of latent representations before decoding to the target queries (Feng et al., 2023). Closely related ideas appear in perceiver-style architectures, where a small latent array cross-attends to a large input and is then processed in the latent space (Jaegle et al., 2021, 2022). This can reduce the computational cost of full attention over large sets while still allowing expressive target-dependent prediction. PrivTab adopts this LBANP design, using a fixed set of summary tokens to aggregate context information. It also follows the prior-data fitted network perspective (Müller et al., 2022), in which pretraining on a task prior yields an in-context predictor for new datasets.

This perspective is particularly useful for the private setting considered in this work. Since the sensitive information flows from the context set to the predictions through attention, the attention mechanism is the natural place to introduce a privacy-preserving transformation of the context information before any downstream prediction is performed by post-processing.

## S1.9 Related Work

Non-DP Tabular Foundation Models. TabPFN (Hollmann et al., 2023, 2025) and its extensions, including TabPFN v3 (Grinsztajn et al., 2026), use prior-fitted inference to classify from a context dataset in a single forward pass. TabICL and TabICL v2 (Qu et al., 2025, 2026) follow a closely related in-context-learning approach. These models amortise tabular classification across tasks, but do not provide formal protection for sensitive context data. This matters because prediction confidence can support membership inference (Carlini et al., 2022; Knolle et al., 2026); in our audit, TabPFN v3 and TabICL v2 show high aggregate and record-level attack success (Supplement S6.1). PrivTab retains the in-context setting while protecting the transferred context representation under DP.

Meta-Learning. PrivTab also belongs to the broader meta-learning literature. A useful distinction is between gradient-based adaptation and amortised inference (Bronskill, 2020). Methods such as MAML (Finn et al., 2017) and Reptile (Nichol et al., 2018) adapt to a new task through additional test-time optimisation. By contrast, matching networks (Vinyals et al., 2016), prototypical networks (Snell et al., 2017), and neural-process models (Garnelo et al., 2018b) map a context set directly to predictions. PrivTab follows this latter approach, but explicitly constrains the context-to-prediction map with a DP mechanism.

Neural Processes and Transformer Neural Processes. PrivTab is most closely related to conditional, latent, attentive, and transformer neural processes (Garnelo et al., 2018b,a; Kim et al., 2019; Nguyen and Grover, 2022). These models condition predictions on a context set in a single forward pass rather than fitting a new model for each dataset. We build most directly on the transformer-neural-process viewpoint, but privatise the context-tosummary interface. PrivTab therefore transfers a formally analysable DP summary rather than an unconstrained representation of the context set.

Diferentially Private Machine Learning. Private machine learning most often relies on DP stochastic optimisation (Abadi et al., 2016; Ponomareva et al., 2023). These methods enable private training for many model classes, but privacy remains tied to an adaptive pipeline that includes clipping, accounting, model selection, and hyperparameter tuning. PrivTab explores a diferent design: it privatises a compact dataset representation once and reuses it by post-processing, making it a private amortised-inference procedure rather than a private optimiser.

Diferentially Private Meta-Learning. DPConvCNP (Räisä et al., 2024) is most closely aligned with our noise-aware objective. It studies private regression with convolutional conditional neural processes and similarly trains on the privatised representation observed at test time. Its focus, however, is convolutional regression rather than heterogeneous tabular classification. Other work applies DP during meta-training (Li et al., 2020; Zhou and Bassily, 2022), rather than privatising an inference-time context set while retaining amortised prediction for that dataset.

Diferentially Private In-Context Learning with LLMs. Recent work also studies DP ICL for LLMs. It includes private prompt synthesis (Gao et al., 2025), private aggregation of LLM responses (Wu et al., 2024), DP generation of few-shot examples (Tang et al., 2024), privacy-aware nearest-neighbor retrieval (Koskela et $\mathrm { a l . }$ 2025), and private prompt learning (Duan et al., 2023). DP-TabICL(Carey et al., 2024) privatises tabular records or group statistics and converts them to textual examples that are provided to a general-purpose LLM. PrivTab instead is a standalone tabular model whose privacy mechanism is built into context summarisation; fitting to a new dataset requires neither external LLM prompting nor synthetic-query generation. These methods address text prompts and LLM inference rather than tabular amortised classification, but share the goal of controlling information released from a sensitive context dataset.

Local Diferential Privacy for Embeddings. Related work also privatises learned representations in a local privacy setting. A local DP approach for transformer text embeddings using a variational information bottleneck has also been studied, with privacy characterised using Rényi divergence and Bayesian Diferential Privacy (Zein and Henderson, 2026). This is complementary to our setting: users privatise their own representations before release, whereas PrivTab applies central DP to a shared context dataset for tabular classification.

## S2 Noise-aware Amortised Inference

In the private setting, the predictor does not observe the context dataset $\mathcal { D } _ { c }$ directly. Instead, it observes the output of a DP mechanism applied to the context set. When this private summary is produced by a learned encoder, the mechanism depends on parameters $\phi ,$ which we regard as fixed when defining the corresponding noise-aware target. In PrivTab, this mechanism is instantiated by the DP-MHCA-based private encoder that maps $\mathcal { D } _ { c }$ to a privatised summary. We therefore write $\mathcal { A } _ { \phi }$ for the mechanism, let S denote its output space, and let $\tilde { S } \in \mathbb S$ denote the released random output. Conditional on $\mathcal { D } _ { c } ,$ this output is distributed according to

$$
\tilde { S } \mid \mathcal { D } _ { c } \sim p _ { \mathcal { A } , \phi } ( \cdot \mid \mathcal { D } _ { c } ) ,\tag{S35}
$$

where $p _ { \mathcal { A } , \phi } ( \tilde { s } \mid \mathcal { D } _ { c } )$ denotes the density or mass function induced by $\mathcal { A } _ { \phi } ( \mathcal { D } _ { c } )$ . The realised DP output s˜ may be a noisy statistic, a private representation, or a learned DP summary of the context dataset.

The appropriate predictive object is then not the posterior predictive conditioned on $\mathcal { D } _ { c }$ , but the conditional distribution of the target labels given the public target inputs and the observed private summary. Let $X _ { 1 : n }$ and $Y _ { 1 : n }$ denote the target-input and target-label random variables under the joint task law. For fixed mechanism parameters $\phi ,$ define the noise-aware posterior predictive directly as the regular conditional law induced by the joint task distribution and private mechanism:

$$
p _ { \phi } ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \tilde { s } ) : = \operatorname* { P r } _ { \phi } \left( Y _ { 1 : n } = y _ { 1 : n } \Big \vert X _ { 1 : n } = \mathbf { x } _ { 1 : n } , \tilde { S } = \tilde { s } \right) .\tag{S36}
$$

When the joint law admits the latent-variable factorisation described in Supplement S1.6, this conditional law can equivalently be written as

$$
p _ { \phi } ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \tilde { s } ) = \int _ { \Theta } p ( y _ { 1 : n } \mid \mathbf { x } _ { 1 : n } , \theta ) p _ { \phi } ( \theta \mid \mathbf { x } _ { 1 : n } , \tilde { s } ) \mathrm { d } \theta .\tag{S37}
$$

This distribution accounts for uncertainty induced both by the latent parameter $\pmb \theta ,$ including any information about it carried by the target inputs, and by the randomness of the fixed private mechanism $\mathcal { A } _ { \phi }$

Suppose now that we learn an amortised predictive distribution $q _ { \pmb { \psi } } ( y _ { 1 : n } \ \vert \ \mathbf { x } _ { 1 : n } , \tilde { s } )$ to approximate this noise-aware posterior predictive. In the private setting, the prediction objective is the same expected negative log-likelihood objective defined in Supplement S1.6, except that the confidential context dataset is replaced by the released DP output ${ \tilde { S } } .$ For fixed mechanism parameters $\phi ,$ this gives

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi ) = \mathbb { E } _ { p _ { \phi } ( \tau , \tilde { S } ) } \left[ - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } \right) \right] .\tag{S38}
$$

The following theorem identifies the Kullback–Leibler (KL) divergence objective corresponding to the conditional negative log-likelihood at fixed target size $n _ { t }$

Theorem S2.1. Fix mechanism parameters $\phi$ and a target size $n _ { t } \geq 1$ . Let $\mathbb { T } _ { n _ { t } } \subseteq \mathbb { T }$ denote the subset oftasks whose target set has size $n _ { t }$ , let $\tilde { S } \mid \mathcal { D } _ { c } \sim p _ { \mathcal { A } , \phi } ( \cdot \mid \mathcal { D } _ { c } )$ , and let $p _ { \phi } ^ { ( n _ { t } ) } ( \tau , \tilde { S } )$ denote the resulting joint law on $\mathbb { T } _ { n _ { t } } \times \mathbb { S }$ Define the conditional objective

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) : = \mathbb { E } _ { p _ { \phi } ^ { ( n _ { t } ) } ( \tau , \tilde { S } ) } \left[ - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } \right) \right] .\tag{S39}
$$

Let $q _ { \pmb { \psi } } \left( y _ { 1 : n _ { t } } \ \middle | \ \mathbf { x } _ { 1 : n _ { t } } , \tilde { s } \right)$ be an amortised predictive distribution. Assume that the relevant joint and conditional densities or mass functions on $\mathbb { T } _ { n _ { t } } \times \mathbb { S }$ exist and that $\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } )$ is finite, so that the successive integrations below are justified by Fubini’s theorem. Then

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \mathrm { c o n s t } ( \phi , n _ { t } ) + \mathbb { E } _ { p _ { \phi } ^ { ( n _ { t } ) } ( \tau , \tilde { S } ) } \left[ \mathrm { K L } \left( p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) \right) \Big \Vert q _ { \psi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) \Big ) \right] ,\tag{S40}
$$

where cons $\operatorname { t } ( \phi , n _ { t } )$ does not depend on $\psi .$ Therefore, for fixed $\phi ,$ minimising $\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } )$ is equivalent to minimising the expected KL divergence from the noise-aware posterior predictive $p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } )$ to the amortised predictive distribution $q _ { \psi } ( \cdot | \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } )$

Proof. By definition of expectation with respect to the joint law $p _ { \phi } ^ { ( n _ { t } ) } ( \pmb { \tau } , \tilde { S } )$

$$
{ \mathcal { L } } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \int _ { \mathbb { T } _ { n _ { t } } } \int _ { \mathbb { S } } - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } ( \pmb { \tau } ) \ | \ \mathbf { x } _ { 1 : n _ { t } } ^ { t } ( \pmb { \tau } ) , \tilde { s } \right) p _ { \phi } ^ { ( n _ { t } ) } ( \pmb { \tau } , \tilde { s } ) \mathrm { d } \tilde { s } \mathrm { d } \tau .\tag{S41}
$$

Writing $\pmb { \tau } = ( \mathcal { D } _ { c } , \mathcal { D } _ { t } )$ with $\mathcal { D } _ { t } = ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , y _ { 1 : n _ { t } } ^ { t } )$ , and applying Fubini’s theorem to expand the integral over the task coordinates, this is

$$
\mathcal { L } _ { \mathrm { N L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \int _ { \mathbb { D } } \int _ { \mathcal { X } ^ { n _ { t } } } \int _ { \mathcal { Y } ^ { n _ { t } } } \int _ { \mathbb { S } } - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \right) p _ { \phi } ^ { ( n _ { t } ) } ( { \mathcal { D } } _ { c } , \mathbf { x } _ { 1 : n _ { t } } ^ { t } , y _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \mathrm { d } \tilde { s } \mathrm { d } y _ { 1 : n _ { t } } ^ { t } \mathrm { d } \mathbf { x } _ { 1 : n _ { t } } ^ { t } \mathrm { d } { \mathcal { D } } _ { c } .\tag{S42}
$$

Integrating out the context dataset $\mathcal { D } _ { c } ,$ again by Fubini’s theorem and the definition of the marginal density, gives

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \int _ { \mathcal { X } ^ { n _ { t } } } \int _ { \mathcal { Y } ^ { n _ { t } } } \int _ { \mathbb { S } } - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \right) p _ { \phi } ^ { ( n _ { t } ) } ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , y _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \mathrm { d } \tilde { s } \mathrm { d } y _ { 1 : n _ { t } } ^ { t } \mathrm { d } \mathbf { x } _ { 1 : n _ { t } } ^ { t } .\tag{S43}
$$

Rewriting the joint density by the chain rule for densities or mass functions as $p _ { \phi } ^ { ( n _ { t } ) } ( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) p _ { \phi } ^ { ( n _ { t } ) } ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } )$ gives

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \int _ { \mathcal { X } ^ { n _ { t } } } \int _ { \mathbb { S } } \left[ \int _ { \mathcal { Y } ^ { n _ { t } } } - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \right) p _ { \phi } ^ { \left( n _ { t } \right) } ( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \mathrm { d } y _ { 1 : n _ { t } } ^ { t } \right] .\tag{S44}
$$

$$
\begin{array} { r } { p _ { \phi } ^ { ( n _ { t } ) } ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \mathrm { d } \tilde { s } \mathrm { d } \mathbf { x } _ { 1 : n _ { t } } ^ { t } . } \end{array}
$$

By definition of the noise-aware posterior predictive, the conditional law $p _ { \phi } ^ { ( n _ { t } ) } ( y _ { 1 : n _ { t } } ^ { t } \ | \ \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } )$ is precisely $p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } )$ . Therefore the inner integral is the cross-entropy

$$
\mathcal { H } \left( p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) , q _ { \psi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \right) .\tag{S45}
$$

Hence

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \int _ { \mathcal { X } ^ { n _ { t } } } \int _ { \mathbb { S } } \mathcal { H } \left( p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) , q _ { \psi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \right) p _ { \phi } ^ { ( n _ { t } ) } ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \mathrm { d } \tilde { s } \mathrm { d } \mathbf { x } _ { 1 : n _ { t } } ^ { t } .\tag{S46}
$$

Using the identity

$$
\begin{array} { r } { \mathcal { H } ( p , q ) = \mathcal { H } ( p ) + \mathrm { K L } ( p \Vert q ) , } \end{array}\tag{S47}
$$

pointwise in $( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } )$ , we obtain

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \displaystyle \int _ { \mathcal { X } ^ { n _ { t } } } \int _ { \mathbb { S } } \mathcal { H } \left( p _ { \phi } \big ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \big ) \right) p _ { \phi } ^ { ( n _ { t } ) } \big ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \big ) \mathrm { d } \tilde { s } \mathrm { d } \mathbf { x } _ { 1 : n _ { t } } ^ { t } } \\ & { \qquad + \displaystyle \int _ { \mathcal { X } ^ { n _ { t } } } \int _ { \mathbb { S } } \mathrm { K L } \left( p _ { \phi } \big ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \big ) \big \| q _ { \psi } \big ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \big ) \right) p _ { \phi } ^ { ( n _ { t } ) } \big ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } \big ) \mathrm { d } \tilde { s } \mathrm { d } \mathbf { x } _ { 1 : n _ { t } } ^ { t } . } \end{array}\tag{S48}
$$

Since the KL integrand depends on τ only through the target inputs $\mathbf { x } _ { 1 : n _ { t } } ^ { t }$ and on the private release through ${ \tilde { S } } ,$ another application of the law of the unconscious statistician yields

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , n _ { t } ) = \mathrm { c o n s t } ( \phi , n _ { t } ) + \mathbb { E } _ { p _ { \phi } ^ { ( n _ { t } ) } ( \tau , \tilde { S } ) } \left[ \mathrm { K L } \left( p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) \right) \Big \Vert q _ { \psi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) \Big ) \right] .\tag{S49}
$$

The first term depends only on the data-generating process and the fixed mechanism parameters $\phi ,$ not on ψ. Writing it as

$$
\mathrm { c o n s t } ( \phi , n _ { t } ) = \int _ { \mathcal { X } ^ { n _ { t } } } \int _ { \mathbb { S } } \mathcal { H } \left( p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \right) p _ { \phi } ^ { ( n _ { t } ) } ( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { s } ) \mathrm { d } \tilde { s } \mathrm { d } \mathbf { x } _ { 1 : n _ { t } } ^ { t }\tag{S50}
$$

gives the result.

口

## S3 Diferentially Private Transformer Neural Processes

Diferentially private transformer neural processes are transformer-based amortised predictors for which the sensitive information is contained in the context dataset. We assume that each individual’s sensitive data corresponds to a single row of the private context set $\mathcal { D } _ { c } .$ . Accordingly, privacy is defined with respect to the substitute adjacency relation defined in Supplement S1.2, so that neighbouring datasets difer in exactly one row. This choice is natural for the tabular setting considered here and is compatible with the sequential representation of the context set used by the transformer architecture.

We instantiate PrivTab shown in Fig. E1. The model receives a private context set $\mathcal { D } _ { c }$ together with public target queries $\mathbf { x } _ { 1 : n _ { t } } ^ { t }$ and proceeds in three parts:

1. A row embedding model maps context rows $\{ ( \mathbf { x } _ { i } ^ { c } , y _ { i } ^ { c } ) \} _ { i = 1 } ^ { n _ { c } }$ and target queries $\{ \mathbf { x } _ { j } ^ { t } \} _ { j = } ^ { n _ { t } }$ to token representations.

2. A fixed set of learned pseudo-tokens $\{ { \mathbf { u } } _ { i } \} _ { i = 1 } ^ { m }$ repeatedly aggregates the private context tokens through DP-MHCA, followed by latent self-attention, to produce privatised summary tokens.

3. At each layer, target tokens attend to the privatised summaries through multi-head cross-attention; a final MLP predicts $y _ { 1 : n _ { t } } ^ { t }$ . This target-side computation is post-processing and requires no additional privacy protection.

The final privacy guarantee depends on the number of DP-MHCA compositions. The following subsections develop the model in order: row embedding, the private attention mechanism and its guarantees, the reference implementation, and the complete encoder architecture. The corresponding noise-aware pretraining objective and training procedure are developed separately in Supplement S4.

Row-order equivariance of PrivTab. PrivTab uses no positional encodings. Row-wise embeddings, layer normalisation, feed-forward networks, and decoding are applied independently to each row and do not use row order. DP-MHCA is invariant to context-row permutations because it sums bounded per-row contributions; the remaining latent computation is post-processing. Target-side cross-attention is equivariant to target-row permutations. Hence, the predictor is invariant to context-row order and equivariant to target-row order. Once target predictions are combined in the aggregated training loss, the loss is invariant to permutations of both context and target rows.

## S3.1 Row Embedding Model

The row embedding model maps each context row $\left( \mathbf { x } _ { i } ^ { c } , y _ { i } ^ { c } \right)$ and each target row $\mathbf { x } _ { j } ^ { t }$ to a token representation in $\mathbb { R } ^ { d _ { \mathrm { t o k } } }$ . Let $d \leq d _ { x }$ denote the number of observed features in a given dataset, and let $e _ { \mathrm { p a d } } \in$ R denote a learned padding value. For a raw input row $\mathbf { x } _ { i } ^ { \mathrm { { r a w } } } \in \mathbb { R } ^ { d }$ , we first form a padded vector $\bar { \mathbf { x } } _ { i } \in \mathbb { R } ^ { d _ { x } }$ by

$$
\begin{array} { r } { \bar { \mathbf { x } } _ { i , k } = \left\{ \begin{array} { l l } { \mathbf { x } _ { i , k } ^ { \mathrm { r a w } } , } & { k \leq d , } \\ { e _ { \mathrm { p a d } } , } & { k > d , } \end{array} \right. \quad k = 1 , \ldots , d _ { x } . } \end{array}\tag{S51}
$$

We then define a binary validity mask $\mathbf { m } _ { i } \in \{ 0 , 1 \} ^ { d _ { x } }$ by

$$
\begin{array} { r } { { \bf { m } } _ { i , k } = \left\{ \begin{array} { l l } { 1 , } & { k \leq d , } \\ { 0 , } & { k > d , } \end{array} \right. \quad \quad k = 1 , \ldots , d _ { x } , } \end{array}\tag{S52}
$$

and concatenate the padded row with the mask:

$$
\begin{array} { r } { \tilde { \mathbf { x } } _ { i } = [ \bar { \mathbf { x } } _ { i } , \mathbf { m } _ { i } ] \in \mathbb { R } ^ { 2 d _ { x } } . } \end{array}\tag{S53}
$$

To keep rows with diferent numbers of observed features on a comparable scale, we rescale by

$$
\hat { \mathbf { x } } _ { i } = \sqrt { \frac { d _ { x } } { d } } \tilde { \mathbf { x } } _ { i } \in \mathbb { R } ^ { 2 d _ { x } } .\tag{S54}
$$

Thus, the scaled augmented row itself serves as the x-representation. For classification, the label encoder maps each label $y \in \{ 1 , \ldots , C \}$ , where $C \le C _ { \mathrm { m a x } } = 1 0$ , and a dedicated null label $y _ { \mathrm { n u l l } } = C _ { \mathrm { m a x } } + 1$ to a learned embedding in $\mathbb { R } ^ { d _ { y } }$ . Equivalently, it is a learned lookup table

$$
\mathrm { L a b e l E m b e d } : \{ 1 , \ldots , C _ { \mathrm { m a x } } + 1 \}  \mathbb { R } ^ { d _ { y } } ,\tag{S55}
$$

where the final index is reserved for the null label regardless of the number of classes in the current dataset. In the benchmark configuration, $d _ { y } = 1 0$ . For context rows, the observed label embedding is concatenated with the row representation,

$$
\mathbf { z } _ { i } ^ { c } = \mathrm { R o w E m b e d } \left( \hat { \mathbf { x } } _ { i } ^ { c } , \mathrm { L a b e l E m b e d } ( y _ { i } ^ { c } ) \right) ,\tag{S56}
$$

where RowEmbed $\mathbf { \Psi } ( \mathbf { x } , \mathbf { e } )$ concatenates its two arguments as [x, e] and maps the result into the shared token space using an MLP. For target rows, the same construction is used except that the observed label embedding is replaced by the null-label embedding LabelEmbed $( y _ { \mathrm { n u l l } } )$ to indicate an unknown label and distinguish target from context tokens, giving

$$
\begin{array} { r } { \mathbf { z } _ { j } ^ { t } = \mathrm { R o w E m b e d } \left( \hat { \mathbf { x } } _ { j } ^ { t } , \mathrm { L a b e l E m b e d } ( y _ { \mathrm { n u l l } } ) \right) . } \end{array}\tag{S57}
$$

All previous operations are applied row-wise, which is important both for the privacy analysis and for row-order symmetry. In contrast, the row encoder is not exactly feature-order invariant: a permutation of feature columns permutes coordinates of $\tilde { \mathbf { x } } _ { i }$ , and the subsequent MLP RowEmbed has coordinate-specific weights. As with TabPFN and TabICL (Hollmann et al., 2023; Qu et al., 2025), we rely instead on the randomised synthetic task prior during pretraining to encourage robustness to the arbitrary ordering and roles of features. This provides approximate feature-order invariance in practice, but not an exact architectural guarantee.

## S3.2 DP-MHCA

The DP-MHCA is the central component of PrivTab. It is a multi-head cross-attention mechanism in which a query sequence attends to the private context tokens, followed by Gaussian perturbation of the resulting representation. In the first private layer, these queries are the learned pseudo-tokens $\mathbf { u } _ { 1 : m } ;$ in later private layers, they are the current summary tokens. Let $d _ { \mathrm { t o k } }$ denote the token dimension, H the number of heads, and $d _ { h }$ the per-head dimension, so that the internal attention dimension is $H d _ { h }$ . Let $\mathbf { r } _ { 1 : m } \in ( \mathbb { R } ^ { d _ { \mathrm { t o k } } } ) ^ { m }$ denote the query tokens supplied to a given DP-MHCA layer, and let $\mathbf { z } _ { 1 : n _ { c } } ^ { c } \in ( \mathbb { R } ^ { d _ { \mathrm { t o k } } } ) ^ { n _ { c } }$ denote the private context tokens. These are first mapped by learned linear projections

$$
\mathbf { W } _ { Q } \in \mathbb { R } ^ { H d _ { h } \times d _ { \mathrm { t o k } } } , \qquad \mathbf { W } _ { K } \in \mathbb { R } ^ { H d _ { h } \times d _ { \mathrm { t o k } } } , \qquad \mathbf { W } _ { V } \in \mathbb { R } ^ { H d _ { h } \times d _ { \mathrm { t o k } } } ,\tag{S58}
$$

so that for each query index i and context index $j ,$

$$
\begin{array} { r } { \mathbf { q } _ { i } = \mathbf { W } _ { Q } \mathbf { r } _ { i } \in \mathbb { R } ^ { H d _ { h } } , \qquad \mathbf { k } _ { j } = \mathbf { W } _ { K } \mathbf { z } _ { j } ^ { c } \in \mathbb { R } ^ { H d _ { h } } , \qquad \bar { \mathbf { v } } _ { j } = \mathbf { W } _ { V } \mathbf { z } _ { j } ^ { c } \in \mathbb { R } ^ { H d _ { h } } . } \end{array}\tag{S59}
$$

Each projected vector is then reshaped into H heads of dimension $d _ { h }$ . We denote the resulting per-head vectors by

$$
\mathbf { q } _ { i } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } } , \qquad \mathbf { k } _ { j } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } } , \qquad \bar { \mathbf { v } } _ { j } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } } ,\tag{S60}
$$

for heads $h = 1 , \ldots , H$ , query indices $i = 1 , \ldots , m$ , and context indices $j = 1 , \dots , n _ { c }$

In the implementation, the value vectors are normalised after projection, head by head:

$$
\mathbf { v } _ { j } ^ { ( h ) } = \frac { \bar { \mathbf { v } } _ { j } ^ { ( h ) } } { \operatorname* { m a x } \Bigl \{ \| \bar { \mathbf { v } } _ { j } ^ { ( h ) } \| _ { 2 } , \varepsilon _ { v } \Bigr \} } ,\tag{S61}
$$

where $\varepsilon _ { v } = 1 0 ^ { - 1 2 }$ is the numerical stabiliser used by the implementation. Hence $\| \mathbf { v } _ { i } ^ { ( h ) } \| _ { 2 } \leq 1$ for all h and $j .$

For each head, the cross-attention output is computed using a bounded activation applied to the scaled dot products:

$$
\mathrm { A t t } ^ { ( h ) } \left( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { 1 : n _ { c } } ^ { ( h ) } , \mathbf { v } _ { 1 : n _ { c } } ^ { ( h ) } \right) = \sum _ { j = 1 } ^ { n _ { c } } g \left( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j } ^ { ( h ) } \right) \mathbf { v } _ { j } ^ { ( h ) } ,\tag{S62}
$$

where

$$
g ( \mathbf { q } , \mathbf { k } ) = \operatorname { t a n h } \left( \frac { \langle \mathbf { q } , \mathbf { k } \rangle } { \sqrt { d _ { h } } } \right) .\tag{S63}
$$

Unlike standard transformer attention, these coeficients are not normalised across $j ;$ they are simply bounded by the tanh nonlinearity. Consequently, each summand in Eq. (S62) depends on exactly one context token and hence on exactly one individual’s row in the private context set. Moreover,

$$
\left\| g \left( \mathbf { q } _ { i } ^ { \left( h \right) } , \mathbf { k } _ { j } ^ { \left( h \right) } \right) \mathbf { v } _ { j } ^ { \left( h \right) } \right\| _ { 2 } = \left| g \left( \mathbf { q } _ { i } ^ { \left( h \right) } , \mathbf { k } _ { j } ^ { \left( h \right) } \right) \right| \left\| \mathbf { v } _ { j } ^ { \left( h \right) } \right\| _ { 2 } \leq 1 .\tag{S64}
$$

This bounded-contribution property is what makes the representation amenable to DP.

The head outputs are then concatenated across $h = 1 , \ldots , H$ into the pre-noise attention output ${ \bf a } _ { i } ,$ distinct from the complete block output ${ \bf s } _ { i } ^ { ( \ell ) }$ used in the architecture below:

$$
\mathbf { a } _ { i } = \left[ \mathbf { a } _ { i } ^ { ( 1 ) } , \ldots , \mathbf { a } _ { i } ^ { ( H ) } \right] \in \mathbb { R } ^ { H d _ { h } } ,\tag{S65}
$$

where

$$
\mathbf { a } _ { i } ^ { ( h ) } : = \mathrm { A t t } ^ { ( h ) } \left( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { 1 : n _ { c } } ^ { ( h ) } , \mathbf { v } _ { 1 : n _ { c } } ^ { ( h ) } \right)\tag{S66}
$$

denotes the h-th pre-noise head output for query i. Gaussian noise is added to this concatenated representation; the subscript attn distinguishes this intermediate attention output from the saved layer states $\tilde { \mathbf { s } } _ { i } ^ { ( \ell ) }$ defined below:

$$
\begin{array} { r } { \tilde { \mathbf { s } } _ { i , \mathrm { a t t n } } = \mathbf { a } _ { i } + \pmb { \xi } _ { i } , \qquad \pmb { \xi } _ { i } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } I _ { H d _ { h } } ) , } \end{array}\tag{S67}
$$

where, jointly across the m summary tokens, the concatenated noise vector satisfies

$$
\pmb { \xi } _ { 1 : m } \sim \mathcal { N } \big ( \mathbf { 0 } , \sigma ^ { 2 } I _ { m H d _ { h } } \big ) .\tag{S68}
$$

Thus, the noise coordinates, including the blocks associated with distinct summary tokens, are mutually independent. Although each per-row contribution in DP-MHCA is bounded, the aggregated pre-noise summary can still have a norm that varies substantially across inputs. This variability is beneficial for improving the signal-to-noise ratio (SNR) when the context tokens are more informative, but after Gaussian perturbation a normalisation step is useful to stabilise the noisy attention-output norms before the residual and feed-forward computations. For this reason, we first compute the maximum Euclidean norm across the perturbed tokens and then rescale each perturbed token by

$$
\tilde { \mathbf { s } } _ { i , \mathrm { a t t n } } \gets \frac { G \tilde { \mathbf { s } } _ { i , \mathrm { a t t n } } } { \operatorname* { m a x } _ { k = 1 , \ldots , m } \lVert \tilde { \mathbf { s } } _ { k , \mathrm { a t t n } } \rVert _ { 2 } + \varepsilon _ { s } } , \qquad \varepsilon _ { s } = 1 0 ^ { - 6 } .\tag{S69}
$$

Thus, after this optional step, the largest token norm is strictly below and, when the maximum pre-normalisation norm is large relative to $\varepsilon _ { s } ,$ approximately equal to G. We set $G = 5 1 2$ in all reported experiments. Finally, the resulting vector is mapped back to the original token dimension by an output projection

$$
\mathbf { W } _ { O } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } \times H d _ { h } } , \qquad \mathbf { o } _ { i } = \mathbf { W } _ { O } \tilde { \mathbf { s } } _ { i , \mathrm { a t t n } } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } } .\tag{S70}
$$

In the benchmark configuration, $H d _ { h } = d _ { \mathrm { t o k } } = 2 5 6$ and this output map is the identity. Thus, the DP mechanism in DP-MHCA is applied after the bounded cross-attention computation in $\mathbb { R } ^ { H d _ { h } }$ and before any output mapping back to the token space $\mathbb { R } ^ { d _ { \mathrm { t o k } } }$

For the following results, we consider the deterministic DP-MHCA map before Gaussian perturbation, optional normalisation, and output projection, conditional on a fixed supplied query sequence $\mathbf { r } _ { 1 : m }$ . Writing

$$
\begin{array} { r } { \mathbf { a } _ { 1 : m } ( \mathcal { D } _ { c } ; \mathbf { r } _ { 1 : m } ) = ( \mathbf { a } _ { 1 } ( \mathcal { D } _ { c } ; \mathbf { r } _ { 1 : m } ) , \dots , \mathbf { a } _ { m } ( \mathcal { D } _ { c } ; \mathbf { r } _ { 1 : m } ) ) \in \mathbb { R } ^ { m H d _ { h } } , } \end{array}\tag{S71}
$$

we treat $\mathbf { a } _ { 1 : m } ( \mathcal { D } _ { c } ; \mathbf { r } _ { 1 : m } )$ as the concatenation of the m pre-noise attention outputs defined above.

Privacy and utility guarantees.

Theorem S3.1 (Sensitivity of pre-noise DP-MHCA). Fix any query sequence $\mathbf { r } _ { 1 : m } .$ . Under substitute adjacency, the $\ell _ { 2 }$ -sensitivity of the deterministic pre-noise DP-MHCA map with respect to the private context dataset satisfies

$$
\Delta _ { 2 } ( \mathbf { a } _ { 1 : m } ( \cdot ; \mathbf { r } _ { 1 : m } ) ) \leq 2 \sqrt { H m } .\tag{S72}
$$

Proof. Let $\mathcal { D } _ { c }$ and $\mathcal { D } _ { c } ^ { \prime }$ be neighbouring context datasets under substitute adjacency, so they difer in exactly one row. Since all operations before the attention sum are applied row-wise, this changes exactly one context token in the sequence $\mathbf { z } _ { 1 : n _ { c } } ^ { c }$ . Let this difering token occur at index $j ^ { \star }$ . For any query index i and head $h ,$ , the diference between the two head outputs is

$$
\begin{array} { r l } & { \operatorname { A t t } ^ { ( h ) } \left( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { 1 : n _ { c } } ^ { ( h ) } , \mathbf { v } _ { 1 : n _ { c } } ^ { ( h ) } \right) - \operatorname { A t t } ^ { ( h ) } \left( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { 1 : n _ { c } } ^ { \prime ( h ) } , \mathbf { v } _ { 1 : n _ { c } } ^ { \prime ( h ) } \right) } \\ & { = g \Big ( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j \star } ^ { ( h ) } \Big ) \mathbf { v } _ { j \star } ^ { ( h ) } - g \Big ( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j \star } ^ { \prime ( h ) } \Big ) \mathbf { v } _ { j \star } ^ { \prime ( h ) } , } \end{array}\tag{S73}
$$

because all remaining summands are identical and cancel. By the triangle inequality and the bound established above,

$$
\begin{array} { r l } & { \left\| { \mathrm { A t t } } ^ { ( h ) } \Big ( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { 1 : n _ { c } } ^ { ( h ) } , \mathbf { v } _ { 1 : n _ { c } } ^ { ( h ) } \Big ) - \mathrm { A t t } ^ { ( h ) } \Big ( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { 1 : n _ { c } } ^ { \prime ( h ) } , \mathbf { v } _ { 1 : n _ { c } } ^ { \prime ( h ) } \Big ) \right\| _ { 2 } } \\ & { \leq \left\| g \Big ( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j ^ { \star } } ^ { ( h ) } \Big ) \mathbf { v } _ { j ^ { \star } } ^ { ( h ) } \right\| _ { 2 } + \left\| g \Big ( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j ^ { \star } } ^ { \prime ( h ) } \Big ) \mathbf { v } _ { j ^ { \star } } ^ { \prime ( h ) } \right\| _ { 2 } \leq 2 . } \end{array}\tag{S74}
$$

Thus each head of each summary token changes by at most $2$ in ℓ<sub>2</sub>-norm. Since $\mathbf { a } _ { 1 : m } ( \mathcal { D } _ { c } ; \mathbf { r } _ { 1 : m } )$ is obtained by concatenating the mH head outputs, the squared $\ell _ { 2 } \cdot$ -diference of the full pre-noise representation is at most

$$
\sum _ { i = 1 } ^ { m } \sum _ { h = 1 } ^ { H } 2 ^ { 2 } = 4 H m .\tag{S75}
$$

Taking square roots yields

$$
\begin{array} { r } { \| \mathbf { a } _ { 1 : m } ( \mathcal { D } _ { c } ; \mathbf { r } _ { 1 : m } ) - \mathbf { a } _ { 1 : m } ( \mathcal { D } _ { c } ^ { \prime } ; \mathbf { r } _ { 1 : m } ) \| _ { 2 } \leq 2 \sqrt { H m } , } \end{array}\tag{S76}
$$

which proves the claim.

An important consequence of Theorem S3.1 is that the sensitivity bound does not depend on the context size $n _ { c } .$ . Thus, increasing the number of context rows need not increase the DP noise scale, even though additional informative rows can strengthen the signal accumulated by the attention mechanism.

Theorem S3.2 (µ-GDP of the PrivTab Encoder). Let $L _ { \operatorname* { m a i n } { } } \ge 1$ and $\mu > 0 .$ Under substitute adjacency, $i f$ the encoder contains L DP-MHCA layers, and each layer $\ell \in \{ 1 , \dots , L _ { \operatorname* { m a i n } } \}$ adds isotropic Gaussian noise with standard deviation $\sigma _ { \ell } = 2 \sqrt { H m L _ { \mathrm { m a i n } } } / \mu$ to the full concatenated summary representation, with fresh noise independent across layers and ofall preceding mechanism randomness, then the overall PrivTab encoder satisfies $\mu { - } G D P$ with respect to $\mathcal { D } _ { c }$

Proof. We first note that all row embedding and token encoding operations preceding the DP-MHCA layer are applied point-wise (row-wise) to the private context dataset $\mathcal { D } _ { c }$ . Under substitute adjacency, two adjacent context datasets $\mathcal { D } _ { c } \simeq \mathcal { D } _ { c } ^ { \prime }$ difer in exactly one row. Because the preprocessing is point-wise, this single-row diference propagates to a diference in at most one context token in the resulting sequence of context tokens $\mathbf { z } _ { 1 : n _ { c } } ^ { c }$

Fix any private layer ℓ. Conditional on all previous DP releases, the query sequence supplied to layer ℓ is fixed, although it may depend adaptively on earlier private outputs. Therefore, by Theorem S3.1, the conditional ℓ<sub>2</sub>-sensitivity of the pre-noise summary representation at layer ℓ satisfies $\Delta _ { 2 } \leq 2 \sqrt { H m }$ with respect to the private context dataset.

Under the Gaussian mechanism, adding independent Gaussian noise with standard deviation $\sigma _ { \ell }$ to a query with $\ell _ { 2 } \cdot$ -sensitivity $\Delta _ { 2 }$ satisfies $\mu _ { \ell } { - } \mathrm { G D P }$ with $\begin{array} { r } { \mu _ { \ell } = \frac { \Delta _ { 2 } } { \sigma _ { \ell } } } \end{array}$ (Dong et al., 2022). Substituting the noise standard deviation, we have:

$$
\mu _ { \ell } \leq \frac { 2 \sqrt { H m } } { \sigma _ { \ell } } = \frac { 2 \sqrt { H m } } { \frac { 2 \sqrt { H m L _ { \operatorname* { m a i n } } } } { \mu } } = \frac { \mu } { \sqrt { L _ { \operatorname* { m a i n } } } } .\tag{S77}
$$

Apart from the DP-MHCA accesses to the private context, all intermediate computations in the encoder and decoder are post-processing of previous private releases. In particular, standard self-attention, target-side

cross-attention, feed-forward layers, and the MLP decoder do not consume additional privacy budget beyond the DP-MHCA layers themselves. Since the query sequence of a later DP-MHCA layer may depend on earlier private releases, the resulting composition is adaptive.

By the adaptive composition theorem of GDP (Theorem S1.3), the composition of the $L _ { \mathrm { m a i n } }$ private DP-MHCA mechanisms yields an overall privacy parameter of:

$$
\mu _ { \mathrm { t o t a l } } = \sqrt { \sum _ { \ell = 1 } ^ { L _ { \mathrm { m a i n } } } \mu _ { \ell } ^ { 2 } } \le \sqrt { L _ { \mathrm { m a i n } } \left( \frac { \mu } { \sqrt { L _ { \mathrm { m a i n } } } } \right) ^ { 2 } } = \mu .\tag{S78}
$$

Therefore, the PrivTab encoder satisfies $\mu { \mathrm { - } } \mathrm { G D P }$

This result lifts the per-layer sensitivity bound to an end-to-end privacy guarantee: with the stated noise allocation, the entire encoder satisfies the selected $\mu { \mathrm { - } } \mathrm { G D P }$ budget under substitute adjacency. All subsequent target-side computation is post-processing and therefore incurs no further privacy loss.

Theorem S3.3 (Directional SNR of DP-MHCA). Suppose $\sigma > 0 ,$ and suppose there exist a unit vector u $\in \mathbb { R } ^ { d _ { h } }$ constants $c , \kappa > 0 .$ , an index set $\mathcal { T } \subseteq \{ 1 , \ldots , n _ { c } \}$ , and a constant $R \geq 0$ such that $g ( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j } ^ { ( h ) } ) \geq { \mathfrak { c } }$ and $\langle \mathbf { u } , \mathbf { v } _ { j } ^ { ( h ) } \rangle \geq \kappa { f o r } a l l j \in \mathcal { T } ;$ , and

$$
\sum _ { j \notin \mathbb { Z } } g \left( \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j } ^ { ( h ) } \right) \left. \mathbf { u } , \mathbf { v } _ { j } ^ { ( h ) } \right. \geq - R .\tag{S79}
$$

Then, if Gaussian noise with covariance $\sigma ^ { 2 } I _ { d _ { h } }$ is added to this head output, the directional signal-to-noise ratio

$$
\mathrm { S N R } _ { \mathbf { u } } : = \frac { | \langle \mathbf { u } , \mathbf { a } _ { i } ^ { ( h ) } \rangle | } { \sigma } \geq \frac { \operatorname* { m a x } \{ c \kappa | \mathcal { I } | - R , 0 \} } { \sigma } .\tag{S80}
$$

Proof. Taking the inner product of Eq. (S62) with u gives

$$
\begin{array} { l }  { \displaystyle \left. { { \bf { u } } , { \bf { a } } _ { i } ^ { \left( h \right) } } \right. = \sum _ { j = 1 } ^ { { n _ { c } } } { g \left( { { \bf { q } } _ { i } ^ { \left( h \right) } } , { { \bf { k } } _ { j } ^ { \left( h \right) } } \right) \left. { { \bf { u } } , { \bf { v } } _ { j } ^ { \left( h \right) } } \right. } \ ~ } \\  { \displaystyle ~ = \sum _ { j \in \mathbb { Z } } { g \left( { { \bf { q } } _ { i } ^ { \left( h \right) } } , { { \bf { k } } _ { j } ^ { \left( h \right) } } \right) \left. { { \bf { u } } , { \bf { v } } _ { j } ^ { \left( h \right) } } \right. + \sum _ { j \notin \mathbb { Z } } { g \left( { { \bf { q } } _ { i } ^ { \left( h \right) } } , { { \bf { k } } _ { j } ^ { \left( h \right) } } \right) \left. { { \bf { u } } , { \bf { v } } _ { j } ^ { \left( h \right) } } \right. } } . } \end{array}\tag{S81}
$$

For every $j \in \mathcal { Z }$ , the corresponding summand is at least cκ. Hence

$$
\left. \mathbf { u } , \mathbf { a } _ { i } ^ { ( h ) } \right. \geq \sum _ { j \in \mathbb { Z } } c \kappa - R = c \kappa | \mathbb { Z } | - R .\tag{S82}
$$

If isotropic Gaussian noise $\pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } I _ { d _ { h } } )$ is then added, the projected noise satisfies $\langle { \bf u } , \pmb { \xi } \rangle \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ because $\mathbf { \bar { \boldsymbol { \vert } } \boldsymbol { \vert } \mathbf { u } \vert } \mathbf { \vert } _ { 2 } = 1$ . Therefore,

$$
\lvert \langle \mathbf { u } , \mathbf { a } _ { i } ^ { ( h ) } \rangle \rvert \geq \operatorname* { m a x } \{ c \kappa \lvert \mathcal { T } \rvert - R , 0 \} ,\tag{S83}
$$

which yields the stated lower bound on $\mathrm { S N R } _ { \mathbf { u } }$

The theorem formalises when additional context is useful: rows whose values align with the query direction reinforce one another, whereas the remaining rows can contribute only the bounded ofset R. Thus, the directional signal grows with the number of aligned rows while the noise level is fixed by the privacy mechanism.

Together, the sensitivity and SNR results show that, when a fixed fraction of context rows is aligned with a query and R remains bounded independently of $n _ { c } ,$ the directional signal grows linearly with the number of informative rows without increasing the required DP noise scale.

## S3.3 Reference Implementation

The following listing gives the complete private summary module of the current PrivTab implementation, starting from already embedded context rows. The PrivateSummaryModule class initialises 128 learned pseudotokens, applies three private cross-attention and latent self-attention blocks, then applies two further latent self-attention blocks. It returns all five summary states used by the target-side predictor. Its DPMHCA class uses one 256-dimensional head with bounded tanh attention, Gaussian perturbation, and optional post-noise normalisation; the noisy output is returned directly without an output projection. The module calibrates the per-layer noise standard deviation from the supplied positive µ using Theorems S3.1 and S3.2. The privacy guarantee assumes fixed model weights, independently embedded context rows, and fresh independent standard Gaussian noise at every private layer. Dataset-level preprocessing is outside this listing and must use public or separately privatised statistics. Training uses the default PyTorch noise sampler; private releases supply a secure Gaussian sampler.

```python
"""PrivTab private summary module for already embedded context rows.
The module uses three DP-MHCA layers and two further self-attention layers.
Context embeddings must be row-wise and model weights fixed for the stated
privacy guarantee. The noise sampler must return fresh independent standard
Gaussians on each call. For a private release, pass a secure sampler; the
default PyTorch sampler matches training and benchmark evaluation.
""
import math
import torch
from torch import nn
from torch.nn import functional as F
class DPMHCA(nn.Module):
"""Bounded cross-attention followed by Gaussian noise."""
def __init__(self):
super().__init__()
self.to_q = nn.Linear(256, 256, bias=False)
self.to_k = nn.Linear(256, 256, bias=False)
self.to_v = nn.Linear(256, 256, bias=False)
def forward(self, xq, xkv, sigma, normalize=False, noise_sampler=torch.randn_like):
"""Return private summaries; ``sigma`` has one entry per batch item."""
q = self.to_q(xq)
k = self.to_k(xkv)
v = F.normalize(self.to_v(xkv), p=2, dim=-1)
out = torch.tanh((q @ k.transpose(-1, -2)) * 256**-0.5) @ v
noise = noise_sampler(out)
out = out + noise * sigma[:, None, None]
if normalize:
norm = torch.linalg.vector_norm(out, dim=-1, keepdim=True).amax(
dim=1, keepdim=True
)
out = out / (norm + 1e-6) * 512
return out
class SelfAttention(nn.Module):
"""Standard four-head attention on the summary tokens."""
def __init__(self):
super().__init__()
self.to_q = nn.Linear(256, 256, bias=False)
self.to_k = nn.Linear(256, 256, bias=False)
self.to_v = nn.Linear(256, 256, bias=False)
self.to_out = nn.Sequential(nn.Linear(256, 256), nn.Dropout(0))
def forward(self, xq, xkv, sigma=None, normalize=False, noise_sampler=None):
def heads(x):
return x.reshape(x.shape[0], x.shape[1], 4, 64).transpose(1, 2)
q = heads(self.to_q(xq))
k = heads(self.to_k(xkv))
v = heads(self.to_v(xkv))
out = F.scaled_dot_product_attention(q, k, v, scale=64**-0.5)
out = out.transpose(1, 2).reshape(xq.shape[0], xq.shape[1], 256)
return self.to_out(out)
class SwiGLU(nn.Module):
def __init__(self):
super().__init__()
self.proj_a = nn.Linear(256, 512)
self.proj_b = nn.Linear(256, 512)
def forward(self, x):
return F.silu(self.proj_a(x)) * self.proj_b(x)
```

```python
class AttentionLayer(nn.Module):
"""Pre-normalized attention and feed-forward residual block."""
def __init__(self, attention):
super().__init__()
self.attn = attention
self.ff_block = nn.Sequential(SwiGLU(), nn.Dropout(0), nn.Linear(512, 256), nn.Dropout(0))
self.norm1 = nn.LayerNorm(256)
self.norm2 = nn.LayerNorm(256)
def forward(self, xq, xkv=None, sigma=None, normalize=False, noise_sampler=None):
xkv = xq if xkv is None else xkv
xq = xq + self.attn(self.norm1(xq), self.norm1(xkv), sigma, normalize, noise_sampler)
return xq + self.ff_block(self.norm2(xq))
class PrivateSummaryModule(nn.Module):
"""Return the five summary states consumed by the target-side predictor."""
def __init__(self):
super().__init__()
self.latents = nn.Parameter(torch.randn(128, 256))
self.mhca_ctoq_layers = nn.ModuleList([AttentionLayer(DPMHCA()) for _ in range(3)])
self.mhsa_layers = nn.ModuleList([AttentionLayer(SelfAttention()) for _ in range(3)])
self.post_processing_mhsa_layers = nn.ModuleList(
[AttentionLayer(SelfAttention()) for _ in range(2)]
)
def forward(self, context, mu, normalize=False, noise_sampler=torch.randn_like):
"""Summarize embedded context; ``mu`` has one GDP budget per batch item."""
sigma = 2 * math.sqrt(128 * 3) / mu
latents = self.latents.unsqueeze(0).expand(context.shape[0], -1, -1)
states = []
for private, self_attention in zip(self.mhca_ctoq_layers, self.mhsa_layers):
latents = self_attention(
private(latents, context, sigma, normalize, noise_sampler)
)
states.append(latents)
for self_attention in self.post_processing_mhsa_layers:
latents = self_attention(latents)
states.append(latents)
return torch.stack(states, dim=1)
```

## S3.4 PrivTab Architecture

PrivTab compresses the private context dataset $\mathcal { D } _ { c }$ into five fixed-size privatised summary states using privacypreserving attention layers. We denote their collection by $\tilde { S }$ and the state after layer ℓ by $\tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) }$ . The model then classifies public target queries using only non-private post-processing. The detailed architecture is shown in Fig. E1. It has three components: transformer building blocks, a Perceiver-style private encoder, and an output decoder.

## S3.4.1 Transformer Building Blocks

We use a pre-normalisation configuration (Ba et al., 2016), in which Layer Normalisation (LN) is applied before each attention or feed-forward block and the residual connection is added afterwards.

For standard multi-head self-attention (MHSA) and multi-head cross-attention (MHCA), we use the standard transformer formulation (Vaswani et al., 2017). Let $\mathbf { r } _ { 1 : m }$ denote a sequence of query tokens and let ${ \bf z } _ { 1 : n }$ denote a sequence of key-value tokens, all in $\mathbb { R } ^ { d _ { \mathrm { t o k } } }$ . For multi-head cross-attention, the updated representation for query token $i \in \{ 1 , \ldots , m \}$ is computed head-by-head and then concatenated:

$$
\mathrm { M H C A } \left( \mathbf { r } _ { i } ; \mathbf { z } _ { 1 : n } \right) = \mathbf { W } _ { O } \left[ \mathbf { a } _ { i } ^ { ( 1 ) } , \ldots , \mathbf { a } _ { i } ^ { ( H ) } \right] + \mathbf { b } _ { O } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } } ,\tag{S84}
$$

where $\mathbf { W } _ { O } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } \times H d _ { h } }$ and $\mathbf { b } _ { O } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } }$ are the learnable output projection parameters. Each head output $\mathbf { a } _ { i } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } }$ for $h = 1 , \ldots , H$ is given by

$$
\mathbf { a } _ { i } ^ { ( h ) } = \sum _ { j = 1 } ^ { n } \alpha _ { i , j } ^ { ( h ) } \mathbf { v } _ { j } ^ { ( h ) } ,\tag{S85}
$$

with key, query, and value projections:

$$
\mathbf { q } _ { i } ^ { ( h ) } = \mathbf { W } _ { Q } ^ { ( h ) } \mathbf { r } _ { i } , \qquad \mathbf { k } _ { j } ^ { ( h ) } = \mathbf { W } _ { K } ^ { ( h ) } \mathbf { z } _ { j } , \qquad \mathbf { v } _ { j } ^ { ( h ) } = \mathbf { W } _ { V } ^ { ( h ) } \mathbf { z } _ { j } ,\tag{S86}
$$

using learnable projections $\mathbf { W } _ { Q } ^ { ( h ) } , \mathbf { W } _ { K } ^ { ( h ) } , \mathbf { W } _ { V } ^ { ( h ) } \in \mathbb { R } ^ { d _ { h } \times d _ { \mathrm { t o k } } }$ . The attention weights $\alpha _ { i , j } ^ { ( h ) }$ are

$$
\alpha _ { i , j } ^ { ( h ) } = \frac { \exp \left( \frac { \langle \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j } ^ { ( h ) } \rangle } { \sqrt { d _ { h } } } \right) } { \sum _ { j ^ { \prime } = 1 } ^ { n } \exp \left( \frac { \langle \mathbf { q } _ { i } ^ { ( h ) } , \mathbf { k } _ { j ^ { \prime } } ^ { ( h ) } \rangle } { \sqrt { d _ { h } } } \right) } .\tag{S87}
$$

Multi-head self-attention is the special case in which the queries and key-values coincide:

$$
\mathrm { M H S A } \left( \mathbf { r } _ { i } ; \mathbf { r } _ { 1 : m } \right) = \mathrm { M H C A } \left( \mathbf { r } _ { i } ; \mathbf { r } _ { 1 : m } \right) .\tag{S88}
$$

For the token-wise feed-forward network (FFN), we use the SwiGLU variant of the gated linear unit (Shazeer, 2020). For a token representation $\mathbf { x } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } }$ , the SwiGLU activation and resulting FFN output are

$$
\mathrm { S w i G L U } ( \mathbf { x } ) = \mathrm { S i L U } \left( \mathbf { W } _ { a } ^ { \top } \mathbf { x } + \mathbf { b } _ { a } \right) \odot \left( \mathbf { W } _ { b } ^ { \top } \mathbf { x } + \mathbf { b } _ { b } \right) ,\tag{S89}
$$

$$
\mathrm { F F N } ( \mathbf { x } ) = \mathbf { W } _ { o } ^ { \top } \mathrm { S w i G L U } ( \mathbf { x } ) + \mathbf { b } _ { o } ,\tag{S90}
$$

where $\mathrm { S i L U } ( { \bf t } ) = { \bf t } ($ sigmoid(t) is the sigmoid linear unit, with sigmoid $\mathbf { \Psi } ( t ) = ( 1 + e ^ { - t } ) ^ { - 1 }$ applied coordinatewise; $\mathbf { W } _ { a } , \mathbf { W } _ { b } \ \in \ \mathbb { R } ^ { d _ { \mathrm { t o k } } \times d _ { \mathrm { f f n } } }$ and $\mathbf { W } _ { o } ~ \in ~ \mathbb { R } ^ { d _ { \mathrm { f f n } } \times d _ { \mathrm { t o k } } }$ are learnable projection matrices, $\mathbf { b } _ { a } , \mathbf { b } _ { b } \in \mathbb { R } ^ { d _ { \mathrm { f f n } } }$ and $\mathbf { b } _ { o } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } }$ are bias vectors, and denotes the Hadamard product. In all transformer feed-forward networks, we set $d _ { \mathrm { f f n } } = 2 d _ { \mathrm { t o k } }$ , so the hidden dimension uses a $2 \times$ expansion.

## S3.4.2 Private Perceiver Encoder

The private Perceiver encoder maps the context tokens $\mathbf { z } _ { 1 : n _ { c } } ^ { c }$ and initial target tokens $\mathbf { z } _ { 1 : n _ { t } } ^ { t , ( 0 ) } = \mathbf { z } _ { 1 : n _ { t } } ^ { t }$ to updated target representations using $L _ { \mathrm { m a i n } } = 3$ main layers and $L _ { \mathrm { p o s t } } = 2$ post-processing layers. In the benchmark configuration, the latent bottleneck contains $m = 1 2 8$ learned pseudo-tokens. In standard MHSA and MHCA, we use $H _ { \mathrm { s t d } } = 4$ heads with per-head dimension $d _ { h , \mathrm { s t d } } = 6 4$ , so the internal attention width is $H _ { \mathrm { s t d } } d _ { h , \mathrm { s t d } } = 2 5 6$ . In DP-MHCA, we use $H _ { \mathrm { D P } } = 1$ head with per-head dimension $d _ { h , \mathrm { D P } } = 2 5 6 .$ , so $H _ { \mathrm { D P } } d _ { h , \mathrm { D P } } = 2 5 6$ . The generic $H$ and $d _ { h }$ in the DP-MHCA definitions and theorems above refer to $H _ { \mathrm { D P } }$ and $d _ { h , \mathrm { D P } }$ , respectively. Because the row embedding model is applied row-wise to the private context dataset $\mathcal { D } _ { c }$ , a single-row change between adjacent context datasets afects at most one context token in $\mathbf { z } _ { 1 : n _ { c } } ^ { c }$ . The DP-MHCA block is therefore the component responsible for controlling the sensitivity of the privatised summary. We initialise the summary representations using the learned pseudo-tokens $\mathbf { u } _ { 1 : m } \colon$

$$
\tilde { \mathbf { s } } _ { i } ^ { ( 0 ) } = \mathbf { u } _ { i } , \qquad i = 1 , \ldots , m .\tag{S91}
$$

The complete blocks used in Methods are denoted by $\mathrm { D P - M H C A } _ { \mathrm { b l o c k } } ^ { ( \ell ) } , \mathrm { M H S A } _ { \mathrm { b l o c k } } ^ { ( \ell ) }$ , and $\mathrm { M H C A } _ { \mathrm { b l o c k } } ^ { ( \ell ) }$ . The equations below expand these blocks into attention operations, pre-normalisation, residual updates and feedforward networks; the attention operators DP-MHCA, MHSA, and MHCA denote only the attention operations. For each main layer $\ell = 1 , \ldots , L _ { \mathrm { m a i n } }$ , the summary and target tokens are updated in three stages:

1. DP Cross-Attention Update: The summaries attend to the private context tokens:

$$
\begin{array} { r } { \mathbf { o } _ { i } ^ { ( \ell ) } = \mathrm { D P - M H C A } \left( \mathrm { L N } \left( \tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) } \right) ; \mathrm { L N } \left( \mathbf { z } _ { 1 : n _ { c } } ^ { c } \right) \right) , \qquad i = 1 , \ldots , m , } \end{array}\tag{S92}
$$

$$
\hat { \mathbf { s } } _ { i } ^ { ( \ell ) } = \tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) } + \mathbf { o } _ { i } ^ { ( \ell ) } ,\tag{S93}
$$

$$
\mathbf { s } _ { i } ^ { ( \ell ) } = \hat { \mathbf { s } } _ { i } ^ { ( \ell ) } + \mathrm { F F N } _ { \mathrm { d p } } ^ { ( \ell ) } \left( \mathrm { L N } \left( \hat { \mathbf { s } } _ { i } ^ { ( \ell ) } \right) \right) .\tag{S94}
$$

This DP-MHCA block is the only operation that transfers context information into the released summaries.

2. Latent Self-Attention Update: The summary representations interact in the latent space:

$$
\begin{array} { r } { \mathbf { s } _ { i , \mathrm { s e l f } } ^ { ( \ell ) } = \mathrm { M H S A } \left( \mathrm { L N } \left( \mathbf { s } _ { i } ^ { ( \ell ) } \right) ; \mathrm { L N } \left( \mathbf { s } _ { 1 : m } ^ { ( \ell ) } \right) \right) , \qquad i = 1 , \dots , m , } \end{array}\tag{S95}
$$

$$
\hat { \mathbf { s } } _ { i , \mathrm { s e l f } } ^ { ( \ell ) } = \mathbf { s } _ { i } ^ { ( \ell ) } + \mathbf { s } _ { i , \mathrm { s e l f } } ^ { ( \ell ) } ,
$$

$$
\begin{array} { r } { \tilde { \mathbf { s } } _ { i } ^ { ( \ell ) } = \hat { \mathbf { s } } _ { i , \mathrm { s e l f } } ^ { ( \ell ) } + \mathrm { F F N } _ { \mathrm { s e l f } } ^ { ( \ell ) } \left( \mathrm { L N } \left( \hat { \mathbf { s } } _ { i , \mathrm { s e l f } } ^ { ( \ell ) } \right) \right) . } \end{array}\tag{S96}
$$

(S97)

3. Target Cross-Attention Update: The target tokens attend to the updated privatised summaries:

$$
\mathbf { t } _ { j } ^ { ( \ell ) } = \mathrm { M H C A } \left( \mathrm { L N } \left( \mathbf { z } _ { j } ^ { t , ( \ell - 1 ) } \right) ; \mathrm { L N } \left( \tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) } \right) \right) , \qquad j = 1 , \dotsc , n _ { t } ,\tag{S98}
$$

$$
\hat { \mathbf { z } } _ { j } ^ { t , ( \ell ) } = \mathbf { z } _ { j } ^ { t , ( \ell - 1 ) } + \mathbf { t } _ { j } ^ { ( \ell ) } ,\tag{S99}
$$

$$
\begin{array} { r } { \mathbf { z } _ { j } ^ { t , ( \ell ) } = \hat { \mathbf { z } } _ { j } ^ { t , ( \ell ) } + \mathrm { F F N } _ { \mathrm { t g t } } ^ { ( \ell ) } \left( \mathrm { L N } \left( \hat { \mathbf { z } } _ { j } ^ { t , ( \ell ) } \right) \right) . } \end{array}\tag{S100}
$$

After the main layers, $L _ { \mathrm { p o s t } }$ additional post-processing layers refine the representations. These layers do not access the private context set and therefore preserve the privacy guarantee established by the DP-MHCA blocks. For each layer $\ell = L _ { \mathrm { m a i n } } + 1 , \ldots , L _ { \mathrm { m a i n } } + L _ { \mathrm { p o s t } }$ , we perform:

## 1. Latent Self-Attention Update:

$$
\begin{array} { r } { \mathbf { s } _ { i , \mathrm { s e l f } } ^ { ( \ell ) } = \mathrm { M H S A } \left( \mathrm { L N } \left( \tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) } \right) ; \mathrm { L N } \left( \tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell - 1 ) } \right) \right) , \qquad i = 1 , \dots , m , } \end{array}\tag{S101}
$$

$$
\hat { \mathbf { s } } _ { i } ^ { ( \ell ) } = \tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) } + \mathbf { s } _ { i , \mathrm { s e l f } } ^ { ( \ell ) } ,\tag{S102}
$$

$$
\begin{array} { r } { \tilde { \mathbf { s } } _ { i } ^ { ( \ell ) } = \hat { \mathbf { s } } _ { i } ^ { ( \ell ) } + \mathrm { F F N } _ { \mathrm { s e l f } } ^ { ( \ell ) } \left( \mathrm { L N } \left( \hat { \mathbf { s } } _ { i } ^ { ( \ell ) } \right) \right) . } \end{array}\tag{S103}
$$

2. Target Cross-Attention Update:

$$
\begin{array} { r } { \mathbf { t } _ { j } ^ { ( \ell ) } = \mathrm { M H C A } \left( \mathrm { L N } \left( \mathbf { z } _ { j } ^ { t , ( \ell - 1 ) } \right) ; \mathrm { L N } \left( \tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) } \right) \right) , \qquad j = 1 , \dotsc , n _ { t } , } \end{array}\tag{S104}
$$

$$
\hat { \mathbf { z } } _ { j } ^ { t , ( \ell ) } = \mathbf { z } _ { j } ^ { t , ( \ell - 1 ) } + \mathbf { t } _ { j } ^ { ( \ell ) } ,\tag{S105}
$$

$$
\mathbf { z } _ { j } ^ { t , ( \ell ) } = \hat { \mathbf { z } } _ { j } ^ { t , ( \ell ) } + \mathrm { F F N } _ { \mathrm { t g t } } ^ { ( \ell ) } \left( \mathrm { L N } \left( \hat { \mathbf { z } } _ { j } ^ { t , ( \ell ) } \right) \right) .\tag{S106}
$$

Table S1 summarises the token flow, query/key-value roles, and privacy status of the encoder sub-blocks.

Table S1: Summary of operations and token flow in each layer ℓ of the PrivTab encoder. Layer Normalisation (LN) is applied before each operation in a pre-normalisation layout.
<table><tr><td>Sub-block</td><td>Query Token(s)</td><td>Key-Value Token(s)</td><td>Updated Token(s)</td><td>Privacy Status</td></tr><tr><td colspan="5">Main Layers  $( \ell = 1 , \ldots , L _ { \mathrm { m a i n } } )$ </td></tr><tr><td>DP Cross-Attention</td><td> $\mathrm { S u m m a r i e s } \tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) }$ </td><td>Context  $\mathbf { z } _ { j } ^ { c }$ </td><td> $\mathbf { s } _ { i } ^ { ( \ell ) }$ </td><td>Private (DP-MHCA)</td></tr><tr><td>Latent Self-Attention</td><td> $\mathrm { S u m m a r i e s } \mathbf { s } _ { i } ^ { ( \ell ) }$ </td><td>Summaries  $\mathbf { s } _ { 1 : m } ^ { ( \ell ) }$ </td><td> $\tilde { \mathbf { s } } _ { i } ^ { ( \ell ) }$ </td><td>Public (post-processing)</td></tr><tr><td>Target Cross-Attention</td><td> $\mathrm { T a r g e t s } { \bf z } _ { j } ^ { t , ( \ell - 1 ) }$ </td><td>Summaries  $\tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) }$ </td><td> $\mathbf { z } _ { j } ^ { t , ( \ell ) }$ </td><td>Public (post-processing)</td></tr><tr><td colspan="5">Post-Processing Layers  $( \ell = L _ { \mathrm { m a i n } } + 1 , \ldots , L _ { \mathrm { m a i n } } + L _ { \mathrm { p o s t } } )$ </td></tr><tr><td>Latent Self-Attention</td><td> $\mathrm { S u m m a r i e s } \tilde { \mathbf { s } } _ { i } ^ { ( \ell - 1 ) }$ </td><td> $\mathrm { S u m m a r i e s ~ } \tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell - 1 ) }$ </td><td> $\tilde { \mathbf { s } } _ { i } ^ { ( \ell ) }$ </td><td>Public (post-processing)</td></tr><tr><td>Target Cross-Attention</td><td> $\mathrm { T a r g e t s } { \bf z } _ { j } ^ { t , ( \ell - 1 ) }$ </td><td> $\mathrm { S u m m a r i e s } \tilde { \mathbf { s } } _ { 1 : m } ^ { ( \ell ) }$ </td><td> $\mathbf { z } _ { j } ^ { t , ( \ell ) }$ </td><td>Public (post-processing)</td></tr></table>

## S3.4.3 Output Decoder

Let $L = L _ { \mathrm { m a i n } } + L _ { \mathrm { p o s t } }$ denote the total number of encoder layers. The final target representations $\mathbf { z } _ { 1 : n _ { t } } ^ { t , ( L ) }$ are decoded row-wise to obtain predictive logits. The decoder is a multi-layer perceptron with GELU activations

(Hendrycks and Gimpel, 2016):

$$
\mathbf { h } _ { j } ^ { ( 1 ) } = \mathrm { G E L U } \left( \mathbf { W } _ { d , 1 } ^ { \top } \mathbf { z } _ { j } ^ { t , ( L ) } + \mathbf { b } _ { d , 1 } \right) ,\tag{S107}
$$

$$
\mathbf { h } _ { j } ^ { ( 2 ) } = \mathrm { G E L U } \left( \mathbf { W } _ { d , 2 } ^ { \top } \mathbf { h } _ { j } ^ { ( 1 ) } + \mathbf { b } _ { d , 2 } \right) ,\tag{S108}
$$

$$
\hat { \mathbf { y } } _ { j } = \mathbf { W } _ { d , 3 } ^ { \top } \mathbf { h } _ { j } ^ { ( 2 ) } + \mathbf { b } _ { d , 3 } ,\tag{S109}
$$

where $\hat { \mathbf { y } } _ { j } \in \mathbb { R } ^ { C _ { \operatorname* { m a x } } }$ are the raw output logits for target query j, $\mathbf { W } _ { d , 1 } \in \mathbb { R } ^ { d _ { \mathrm { t o k } } \times d _ { \mathrm { d e c } } }$ $\mathbf { W } _ { d , 2 } \in \mathbb { R } ^ { d _ { \mathrm { d e c } } \times d _ { \mathrm { d e c } } }$ , and $\mathbf { W } _ { d , 3 } \in \mathbf { \bar { \mathbb { R } } } ^ { d _ { \mathrm { d e c } } \times C _ { \mathrm { m a x } } }$ are learnable weight matrices, and $\mathbf { b } _ { d , 1 } , \mathbf { b } _ { d , 2 } , \mathbf { b } _ { d , 3 }$ are bias vectors. We set $d _ { \mathrm { d e c } } = 5 1 2$ . If a dataset has $C \leq C _ { \operatorname* { m a x } }$ classes, only the first $C$ coordinates of $\hat { \mathbf { y } } _ { j }$ are used.

## S4 PrivTab Pretraining

## S4.1 Noise-Aware Pretraining Objective

We pretrain PrivTab by meta-learning across synthetic tabular classification tasks. Each task τ comprises a labelled context set $\mathcal { D } _ { c }$ and a target set $\mathcal { D } _ { t } .$ . The DP-MHCA-based private encoder $\mathcal { A } _ { \phi }$ maps $\mathcal { D } _ { c }$ to a released privatised summary ${ \tilde { S } } ,$ and the predictive network $q _ { \psi }$ maps $( \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } )$ to a distribution over the target labels. We use $\phi$ for parameters that determine the private summary and ψ for parameters that determine predictions. The row embedding network is shared between context and query paths, so these parameter collections overlap in those weights and are trained jointly by minimising target cross-entropy, equivalently the population private negative log-likelihood

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi ) : = \mathbb { E } _ { p _ { \phi } ( \tau , \tilde { S } ) } \left[ - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } \right) \right] .\tag{S110}
$$

As shown in Supplement $S 2 ,$ this is a noise-aware objective: for a fixed private encoder, minimising it fits the posterior predictive distribution conditioned on the privatised summary rather than a predictor that ignores the privacy noise.

Let $N _ { t } ( \pmb { \tau } ) : = | \mathscr { D } _ { t } |$ denote the target-set size, and let $p _ { N _ { t } }$ be its induced distribution under the task simulator. For each fixed $n _ { t }$ , let $p _ { \phi } ^ { ( n _ { t } ) } ( \pmb { \tau } , \tilde { S } )$ denote the joint law of $( \pmb { \tau } , \tilde { S } )$ conditional on $N _ { t } = n _ { t }$ . Applying Theorem S2.1 conditionally on $N _ { t } = \dot { n } _ { t }$ and then averaging over $N _ { t }$ gives

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i y } } ( \psi ; \phi ) = \sum _ { n _ { t } \geq 1 } p _ { N _ { t } } ( n _ { t } ) ( \mathrm { c o n s t } ( \phi , n _ { t } ) + \mathbb { E } _ { p _ { \phi } ^ { ( n _ { t } ) } ( \tau , \tilde { S } ) } [ \mathrm { K L } ( p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) ) \Big \| q _ { \psi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) ) ] ) .\tag{S111}
$$

Thus, when $\phi$ is fixed, optimizing over $\psi$ amounts to fitting $q _ { \psi }$ to the noise-aware posterior predictive induced by $\mathcal { A } _ { \phi }$ . Joint training is therefore the optimisation problem

$$
\operatorname* { m i n } _ { \phi , \psi } \mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi ) ,\tag{S112}
$$

in which $\phi$ changes both the distribution of the released summary, through $p _ { \phi } ( \tau , { \tilde { S } } )$ , and the corresponding family of conditional noise-aware posterior predictives $p _ { \phi } ( \cdot \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } )$ . A useful equivalent view is the profile objective

$$
\mathcal { I } ( \phi ) : = \operatorname* { i n f } _ { \psi } \mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi ) .\tag{S113}
$$

This means that $\phi$ is chosen to define a private mechanism whose induced noise-aware posterior predictive is both informative for prediction and well approximated by the predictive family $q _ { \psi }$

In practice, we first update the model jointly and then freeze the entire encoder late in training, including the shared row embedding, private summary blocks and query-side attention. Only the output decoder remains trainable in this final stage, refining predictions for the fixed representations learned earlier.

## S4.2 Simulator, Curriculum, and Optimisation

To instantiate this objective, we draw the synthetic tasks from the mixed structural causal model (SCM) simulator introduced in TabICL (Qu et al., 2025). Each generated task supplies the labelled context and target sets used by the objective above.

Synthetic Tabular Data Simulator. For each simulated task, we generate a synthetic tabular classification dataset:

$$
\mathcal { D } _ { c } = \{ ( \mathbf { x } _ { i } ^ { c } , y _ { i } ^ { c } ) \} _ { i = 1 } ^ { n _ { c } } ,\tag{S114}
$$

$$
\mathcal { D } _ { t } = \left\{ \left( \mathbf { x } _ { j } ^ { t } , y _ { j } ^ { t } \right) \right\} _ { j = 1 } ^ { n _ { t } } ,\tag{S115}
$$

where the number of active features d is sampled uniformly from [2, 120] and the input feature dimension is padded to $d _ { x } = 1 2 0$ . The number of target classes $C$ is sampled uniformly up to $C _ { \mathrm { m a x } } = 1 0$ . The context size $n _ { c }$ and target size $n _ { t }$ are sampled independently and uniformly from $[ N _ { \mathrm { m i n } } , N _ { \mathrm { m a x } } ]$ for each task. To improve optimisation stability and sample eficiency, we employ a two-stage training curriculum that adjusts the range $[ N _ { \mathrm { m i n } } , N _ { \mathrm { m a x } } ] ;$

1. Stage 1: For the first 128,000 steps, we set $N _ { \mathrm { m i n } } ~ = ~ 1 0 0$ and $N _ { \mathrm { m a x } } = 2 0 4 8 .$ . In this stage, the scale normalisation in Eq. (S69) is disabled.

2. Stage 2: For training steps 128,000 to 256,000, we set $N _ { \mathrm { m i n } } = 1 0 2 4$ and $N _ { \mathrm { m a x } } = 8 1 9 2$ , and enable the scale normalisation in Eq. (S69).

This curriculum enables the model to first stabilise its representation learning on smaller datasets before adapting to larger context scales.

To optimise performance across diferent dataset scales, we train two separate model variants: one trained solely on Stage 1 tasks for an additional 256,000 steps (yielding 384,000 total steps on smaller contexts), and another trained sequentially on Stage 1 and Stage 2 tasks. The small-context variant is trained without summary normalisation, which is enabled at inference; the large-context variant enables it during Stage 2 and at inference. At inference time, we route queries dynamically based on the context size: the smaller-context model is used for datasets with $n _ { c } < 4 0 9 6$ , while the large-context model is used for datasets with $n _ { c } \ge 4 0 9 6$

Privacy Parameter Curriculum. To train the model across a wide range of privacy levels, each simulated task is assigned a privacy parameter µ according to a log-uniform curriculum during meta-training. The overall range is $[ \bar { \mu _ { \operatorname* { m i n } } } , \bar { \mu _ { \operatorname* { m a x } } } ] = [ 0 . 1 6 , 6 4 . 0 ]$ . To ease optimisation at the start of training, we initialise the curriculum with minimal privacy noise by setting the lower bound to $\mu _ { \mathrm { m i n } } ^ { \mathrm { ( c u r r ) } } = 6 4 . 0 .$ . From steps 256 to 38,400, the lower bound $\mu _ { \mathrm { m i n } } ^ { \mathrm { ( c u r r ) } }$ decreases log-linearly from 64.0 to 0.16, while the upper bound remains fixed at 64.0. Afterwards, the curriculum reaches its final state and $\mu$ is sampled according to

$$
\log \mu = u \left( \log \mu _ { \operatorname* { m a x } } - \log \mu _ { \operatorname* { m i n } } \right) + \log \mu _ { \operatorname* { m i n } } ,\tag{S116}
$$

where $u \sim$ Uniform(0, 1). Equivalently, in the final regime $\mu$ is sampled from the density

$$
p ( \mu ) = \frac { 1 } { \mu \left( \log \mu _ { \operatorname* { m a x } } - \log \mu _ { \operatorname* { m i n } } \right) } \mathbf { 1 } _ { [ \mu _ { \operatorname* { m i n } } , \mu _ { \operatorname* { m a x } } ] } ( \mu ) .\tag{S117}
$$

The resulting pretraining objective averages the private negative log-likelihood over both target sizes and privacy levels. Applying Theorem S2.1 pointwise in $( \mu , n _ { t } )$ shows that, in the final regime, training minimises the KL divergence to the corresponding noise-aware posterior predictive on average over both tasks and privacy levels. The derivation below makes this averaging explicit. The curriculum therefore exposes the model first to lower-noise tasks and gradually introduces more challenging high-noise settings.

Objective across privacy levels and target sizes. The resulting pretraining objective is

$$
{ \mathcal E } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \pmb { \psi } ; \pmb { \phi } ) = \int _ { \mu _ { \mathrm { m i n } } } ^ { \mu _ { \mathrm { m a x } } } \sum _ { n _ { t } \geq 1 } { \mathcal L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \pmb { \psi } ; \pmb { \phi } , \mu , n _ { t } ) p _ { N _ { t } } ( n _ { t } ) p ( \mu ) \mathrm { d } \mu ,\tag{S118}
$$

where

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi , \mu , n _ { t } ) : = \mathbb { E } _ { p _ { \phi , \mu } ^ { ( n _ { t } ) } ( \tau , \tilde { S } ) } \left[ - \log q _ { \psi } \left( y _ { 1 : n _ { t } } ^ { t } \mid \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } \right) \right] .\tag{S119}
$$

Here $p _ { \phi , \mu } ^ { ( n _ { t } ) } ( \pmb { \tau } , \tilde { S } )$ denotes the joint law conditional on both the privacy level $\mu$ and the target size $N _ { t } = n _ { t }$ Applying Theorem S2.1 for each fixed pair $( \mu , n _ { t } )$ gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi ) = \displaystyle \int _ { \mu _ { \mathrm { m i n } } } ^ { \mu _ { \mathrm { m a x } } } \sum _ { n _ { t } \geq 1 } ( \mathrm { c o n s t } ( \phi , \mu , n _ { t } )  } \\ & { \qquad \quad  + \mathbb { E } _ { p _ { \phi , \mu } ^ { ( n _ { t } ) } ( \tau , \tilde { S } ) } [ \mathrm { K L } ( p _ { \phi , \mu } ( \cdot \vert \ \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } )  ] \bigg \Vert  q _ { \psi } ( \cdot \vert \ \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) ) ] ) p _ { N _ { t } } ( n _ { t } ) p ( \mu ) \mathrm { d } \mu , } \end{array}\tag{S120}
$$

where cons $( \phi , \mu , n _ { t } )$ does not depend on ψ. Equivalently, writing

$$
\overline { { \mathrm { c o n s t } } } ( \phi ) : = \int _ { \mu _ { \mathrm { m i n } } } ^ { \mu _ { \mathrm { m a x } } } \sum _ { n _ { t } \geq 1 } \mathrm { c o n s t } ( \phi , \mu , n _ { t } ) p _ { N _ { t } } ( n _ { t } ) p ( \mu ) \mathrm { d } \mu ,\tag{S121}
$$

we obtain

$$
\begin{array}{c} \mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r i v } } ( \psi ; \phi ) = \overline { { \mathrm { c o n s t } } } ( \phi ) + \int _ { \mu _ { \mathrm { m i n } } } ^ { \mu _ { \mathrm { m a x } } } \sum _ { n _ { t } \geq 1 }  \\ { \mathbb { E } _ { p _ { \phi , \mu } ^ { ( n _ { t } ) } ( \tau , \tilde { S } ) } \left[ \mathrm { K L } \left( p _ { \phi , \mu } ( \cdot \vert \ \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) \Vert \ q _ { \psi } ( \cdot \vert \ \mathbf { x } _ { 1 : n _ { t } } ^ { t } , \tilde { S } ) \right) \right] p _ { N _ { t } } ( n _ { t } ) p ( \mu ) \mathrm { d } \mu . } \end{array}\tag{S122}
$$

Optimisation and Loss Function. The model receives the context set $\mathcal { D } _ { c }$ and target features $\mathbf { x } _ { 1 : n _ { t } } ^ { t }$ as inputs. The training loss is the cross-entropy, equivalently the negative log-likelihood, computed on the target labels $y _ { 1 : n _ { t } } ^ { t }$ for the active classes. We apply label smoothing (Szegedy et al., 2016) with smoothing factor 0.1. The theoretical result in Theorem S2.1 characterises the corresponding unsmoothed NLL objective; label smoothing is used only as a training regulariser.

We optimise with AdamW (Loshchilov and Hutter, 2019) using peak learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 3 }$ $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 8$ , and $\epsilon = 1 0 ^ { - 8 }$ . We apply gradient clipping with maximum norm 1.0. The learning rate follows a warmup-cosine-constant schedule: it increases linearly from 0 to $1 0 ^ { - 4 }$ over the first 6400 steps, decays with a cosine schedule to $\eta _ { \mathrm { m i n } } = 1 0 ^ { - 5 }$ over 51,200 optimiser steps, and remains constant at $1 0 ^ { - 5 }$ thereafter.

The large-context variant is trained for 256,000 total steps with a batch size of 128 tasks (datasets). At step 243,200, we freeze the entire encoder and continue training only the output decoder for the remaining 12,800 steps. The small-context variant is trained for 384,000 total steps; its final encoder-freezing stage starts at step 371,200, also leaving 12,800 steps to refine the decoder. These final stages stabilise the predictive approximation relative to the learned privatised representation.

## S5 Baseline Details

## S5.1 Accounting for DP Stochastic Optimisation under Substitute Adjacency

For DP stochastic optimisation (Song et al., 2013; Bassily et al., 2014; Abadi et al., 2016), privacy is defined at the row level, so we use the same substitute adjacency relation as in the rest of the paper. Let $\mathbf { \mathcal { D } } \overset { \cdot } { = } \{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$ denote the training set, let $q$ be the Poisson subsampling rate, let $\bar { B } = \operatorname* { m a x } \{ 1 , \lfloor q N \rfloor \}$ be the normalisation based on the expected minibatch size, let $T$ be the number of optimisation steps, let $C > 0$ be the clipping norm, and let $\sigma > 0$ be the standard deviation of the noise added to the clipped gradient sum. At each step t, DP Adam samples a minibatch $B _ { t } \subseteq { \mathcal { D } }$ , clips each per-example gradient to have $\ell _ { 2 } { \mathrm { - n o r m } }$ at most $C _ { i }$ and applies Adam to a noisy gradient of the form

$$
\tilde { \mathbf { g } } _ { t } = \frac { 1 } { \bar { B } } \sum _ { \mathbf { z } _ { i } \in B _ { t } } \mathrm { c l i p } _ { C } ( \nabla _ { \mathbf { w } } \ell ( \mathbf { w } _ { t } ; \mathbf { z } _ { i } ) ) + \frac { \sigma } { \bar { B } } \boldsymbol { \xi } _ { t } , \qquad \boldsymbol { \xi } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) .\tag{S123}
$$

To report privacy in the same form as for PrivTab, we do not use the asymptotic $\mu { \mathrm { - } } \mathrm { G D P }$ approximation for subsampled Gaussian-gradient mechanisms (Bu et al., 2020b). Instead, we compute the privacy profile of the Poisson-subsampled Gaussian mechanism under the replace-one neighbouring relation using a numerical privacy-loss-distribution accountant (Gomez et al., 2026), compose it over $\bar { T }$ steps, and then convert the resulting trade-of function $T$ to the smallest valid non-asymptotic $\mu \cdot$ -GDP guarantee,

$$
\mu ^ { \star } = \operatorname* { i n f } \left\{ \mu \geq 0 : T _ { \mu } ( \alpha ) \leq T ( \alpha ) { \mathrm { f o r ~ a l l } } \alpha \in [ 0 , 1 ] \right\} .\tag{S124}
$$

where $T _ { \mu }$ is the GDP trade-of function from Eq. (S5). For each target privacy budget $\mu ,$ the DP Adam noise standard deviation is then chosen by numerically inverting this map so that the full training procedure satisfies $\mu { \mathrm { - } } \mathrm { G D P }$ under substitute adjacency.

## S5.2 Hyperparameter Calibration and Private Selection

Search spaces, private learning-rate grids and fixed baseline hyperparameters are reported in Tables S2 to S4.

Table S2: Baseline hyperparameter search spaces. Bayesian optimisation is run separately for each private baseline.
<table><tr><td>Baseline</td><td>Hyperparameter</td><td>Search space</td></tr><tr><td>DP-LR</td><td>Learning rate Batch size Clipping norm Epochs</td><td> $[ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ]$  log-uniform {128, 256, 512, 1024} {0.1, 0.5, 1, 2, 5} {100, 300, 500, 1000}</td></tr><tr><td>DP-MLP</td><td>Learning rate Batch size Clipping norm Hidden width Layers</td><td> $[ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ]$  log-uniform {128, 256, 512, 1024} {0.1, 0.5, 1, 2, 5} {64, 128, 256, 512} {1, 2, 3} {100, 300, 500, 1000}</td></tr></table>

DP baseline hyperparameter selection. Based on Figs. S1 and S2, we fix the main hyperparameters of both baselines and tune only the learning rate under DP. For each method, the learning rate is selected from a small grid using a private validation loss. Concretely, after the outer 80/20 context–target split, we divide the context portion into subtraining and validation sets, using a validation fraction of 0.16 within the context set. Relative to the full dataset, this corresponds approximately to a $6 7 / 1 3 / 2 0$ split into subtraining, validation and target data. For any single learning-rate candidate, the DP Adam training run acts only on the subtraining subset while the noisy validation-loss evaluation acts only on the validation subset, so these two components compose by parallel composition (McSherry, 2010). Across the learning-rate grid, however, the same subtraining subset is reused for the diferent candidate training runs and the same validation subset is reused for the diferent candidate validation evaluations, so we also account for composition over all learning-rate candidates, that is, over the grid size. After private learning-rate selection, we perform one additional final DP Adam training run on the union of the subtraining and validation sets using the selected learning rate. In the final protocol, we reserve a fraction 0.5 of the total $\mu$ budget for private learning-rate selection and use the remaining budget for the final training run through ℓ -composition in $\mu { \mathrm { - } } \mathrm { G D P } .$ . The private validation criterion is the mean cross-entropy loss after clipping each per-example validation loss at 5.0.

Table S3: Learning-rate grids used for private model selection. Only the learning rate is selected under DP; all other hyperparameters are fixed from the aggregate hyperparameter-optimisation results in Figs. S1 and S2.
<table><tr><td></td><td>Method Learning-rate grid</td></tr><tr><td>DP-LR</td><td> $\{ 0 . 0 0 5 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \}$ </td></tr><tr><td>DP-MLP</td><td> $\{ 3 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 3 { \times } 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ </td></tr></table>

DP-LR: selected hyperparameters  
![](images/6b099b7d96efe1ba208c7d6f0d2da6b900589111c88f80d544c84d6b07f7b03b.jpg)  
Figure S1: Histogram of the hyperparameters selected for DP-LR across all datasets and context–target splits. The selected values are concentrated enough to justify fixing the non-learning-rate hyperparameters in the final experiments.

Table S4: Fixed hyperparameters used for the DP baselines after inspecting the aggregate hyperparameterselection histograms.
<table><tr><td>Method</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>DP-LR</td><td>Clipping norm</td><td>0.1</td></tr><tr><td>DP-LR</td><td>Subsampling rate q</td><td>0.05</td></tr><tr><td>DP-LR</td><td>Epochs</td><td>1000</td></tr><tr><td>DP-MLP</td><td>Clipping norm</td><td>5.0</td></tr><tr><td>DP-MLP</td><td>Subsampling rate q</td><td>0.05</td></tr><tr><td>DP-MLP</td><td>Epochs</td><td>250</td></tr><tr><td>DP-MLP</td><td>Hidden width</td><td>64</td></tr><tr><td>DP-MLP</td><td>Hidden layers</td><td>1</td></tr></table>

## DP-MLP: selected hyperparameters

![](images/cc1957857fee9c053c60908537908fa8ad2808b1a8ea93eaecd529fb6c5c8de8.jpg)

![](images/5c1ae6e84b9e689d211e055ada8087644d5baf94cd97a074e361bba4c7a64cda.jpg)

![](images/0c730a9304abf7185193658a1e57b289f16e03f2c4ccf13af6f201c33490bcd7.jpg)

![](images/fc5bb9ed7b96f4b9b1bf3838261ce68b496febc525449c17ee3f8a05e1d61999.jpg)

![](images/dd4a73f69c6f682e6b268697b084396f4baf9e64825717b3b95dd242d23fc5b2.jpg)

![](images/ba9126270ecc8d0a5ef7c4bbb7c07f0bb44db5fcaddf3c5bf96472dbbdd6d965.jpg)  
Figure S2: Histogram of the hyperparameters selected for DP-MLP across all datasets and context–target splits. As for DP-LR, the selected values are concentrated enough to justify fixing the non-learning-rate hyperparameters in the final experiments.

## S6 Additional Experiment Details

## S6.1 Membership-Inference Audit Results

Table S5 reports exact FPC-corrected means and complementary 95th-percentile point estimates, while Fig. 5a emphasises the maximum lower record-wise 95% confidence endpoint. The record-level curves in Fig. S3 provide a threshold-specific view of the exposure summarised by the record AUC values.

Table S5: FPC-corrected membership-inference audit on Maternal Health Risk. Aggregate AUC measures attack performance over 100 held-out context draws; record AUC measures each row’s fitted exposure across 4,000 draws. Means are reported with bootstrap 95% CIs over held-out draws or rows, respectively, and the record-level 95th percentile summarises the most exposed 5% of rows. Values near 0.5 indicate perfect privacy.
<table><tr><td>Method</td><td>Aggregate AUC (mean [95% CI])</td><td>Record AUC (mean [95% CI])</td><td>Record AUC (95th percentile)</td></tr><tr><td>PrivTab (µ = 0.05)</td><td>0.501 [0.498, 0.505]</td><td>0.504 [0.504, 0.504]</td><td>0.513</td></tr><tr><td>PrivTab (µ = 0.1)</td><td>0.503 [0.500, 0.507]</td><td>0.505 [0.505, 0.506]</td><td>0.515</td></tr><tr><td>PrivTab (µ = 0.2)</td><td>0.508 [0.504, 0.511]</td><td>0.508 [0.508, 0.509]</td><td>0.519</td></tr><tr><td>PrivTab (µ = 0.4)</td><td>0.515 [0.511, 0.518]</td><td>0.514 [0.514, 0.515]</td><td>0.526</td></tr><tr><td>PrivTab (µ = 0.8)</td><td>0.527 [0.524, 0.530]</td><td>0.523 [0.522, 0.523]</td><td>0.537</td></tr><tr><td>PrivTab (µ = 1.6)</td><td>0.543 [0.540, 0.546]</td><td>0.533 [0.532, 0.534]</td><td>0.554</td></tr><tr><td>TabPFN v3 (non-private)</td><td>0.752 [0.749, 0.755]</td><td>0.686 [0.680, 0.692]</td><td>0.864</td></tr><tr><td>TabICL v2 (non-private)</td><td>0.765 [0.762, 0.767]</td><td>0.708 [0.701, 0.715]</td><td>0.977</td></tr></table>

![](images/08065ececdd312818dbdda1e1bdfd35f00dc6381ac027d8da019781422f886f6.jpg)  
Figure S3: FPC-corrected record-level membership-inference curves. ROC curves are shown for the ten most vulnerable Maternal Health Risk records at each setting. Shades distinguish records, and the diagonal denotes perfect privacy. The fitted IN/OUT standard deviations use the same FPC as Fig. 5a. PrivTab remains substantially closer to the diagonal than non-private TabPFN v3 and TabICL v2, whose most vulnerable records are nearly perfectly distinguishable.

Fitted balanced attack accuracy and its uncertainty. For a fixed record i, let $S _ { i , \mathrm { I N } }$ and $S _ { i , \mathrm { O U T } }$ denote its attack score conditional on the record being a member or a non-member, respectively. We fit separate Gaussian distributions,

$$
S _ { i , \mathrm { I N } } \sim \mathcal { N } ( \mu _ { i , \mathrm { I N } } , \sigma _ { i , \mathrm { I N } } ^ { 2 } ) , \qquad S _ { i , \mathrm { O U T } } \sim \mathcal { N } ( \mu _ { i , \mathrm { O U T } } , \sigma _ { i , \mathrm { O U T } } ^ { 2 } ) ,
$$

where $\sigma _ { i , \mathrm { I N } }$ and $\sigma _ { i , \mathrm { O U T } }$ are the finite-population-corrected standard deviations defined in the Methods. A score above threshold t predicts membership, so, writing Φ for the standard-normal CDF,

$$
\mathrm { T P R } _ { i } ( t ) = \mathrm { P r } ( S _ { i , \mathrm { I N } } > t ) = 1 - \Phi \left( \frac { t - \mu _ { i , \mathrm { I N } } } { \sigma _ { i , \mathrm { I N } } } \right) , \qquad \mathrm { T N R } _ { i } ( t ) = \mathrm { P r } ( S _ { i , \mathrm { O U T } } \leq t ) = \Phi \left( \frac { t - \mu _ { i , \mathrm { O U T } } } { \sigma _ { i , \mathrm { O U T } } } \right) .
$$

With $P = N = 2 { , } 0 0 0$ IN and OUT draws for each record, ordinary accuracy and balanced accuracy have the same form:

$$
{ \frac { \mathrm { T P } + \mathrm { T N } } { P + N } } = { \frac { 1 } { 2 } } \left( { \frac { \mathrm { T P } } { P } } + { \frac { \mathrm { T N } } { N } } \right) .
$$

Replacing the two observed fractions by the fitted Gaussian probabilities gives the fitted balanced accuracy

$$
A _ { i } ( t ; \omega _ { i } ) = \frac { 1 } { 2 } \left[ \mathrm { T P R } _ { i } ( t ) + \mathrm { T N R } _ { i } ( t ) \right] = \frac { 1 } { 2 } \left[ \Phi \left( \frac { \mu _ { i , \mathrm { I N } } - t } { \sigma _ { i , \mathrm { I N } } } \right) + \Phi \left( \frac { t - \mu _ { i , \mathrm { O U T } } } { \sigma _ { i , \mathrm { O U T } } } \right) \right] ,
$$

Here $\pmb { \omega } _ { i } = ( \mu _ { i , \mathrm { I N } } , \mu _ { i , \mathrm { O U T } } , \sigma _ { i , \mathrm { I N } } , \sigma _ { i , \mathrm { O U T } } ) ^ { \mathsf { T } }$ contains the score-model parameters; θ denotes the foundation-model parameters. We choose $\hat { t } _ { i } \in \arg \operatorname* { m a x } _ { t } A _ { i } ( t ; \hat { \omega } _ { i } )$ and write $\hat { A } _ { i } ^ { * } = A _ { i } ( \hat { t } _ { i } ; \hat { \omega } _ { i } )$ , including the limiting accuracy of 0.5 from predicting a single class.

To quantify uncertainty in this fitted accuracy, let $n _ { i , \mathrm { I N } }$ and $n _ { i , \mathrm { O U T } }$ be the numbers of fitted scores, and let $s _ { i , \mathrm { I N } }$ and $s _ { i , \mathrm { O U T } }$ be their uncorrected sample standard deviations. The fitted corrected standard deviations are $\hat { \sigma } _ { i , \mathrm { I N } } = s _ { i , \mathrm { I N } } / \sqrt { f }$ and $\hat { \sigma } _ { i , \mathrm { O U T } } = s _ { i , \mathrm { O U T } } / \sqrt { f }$ , with $f = 0 . 5 .$ . Under independent Gaussian IN and OUT score samples, a large-sample diagonal covariance approximation for $\hat { \omega } _ { i }$ is

$$
\widehat { V } _ { i } = \mathrm { d i a g } \left( \frac { s _ { i , \mathrm { I N } } ^ { 2 } } { n _ { i , \mathrm { I N } } } , \frac { s _ { i , \mathrm { O U T } } ^ { 2 } } { n _ { i , \mathrm { O U T } } } , \frac { \hat { \sigma } _ { i , \mathrm { I N } } ^ { 2 } } { 2 n _ { i , \mathrm { I N } } } , \frac { \hat { \sigma } _ { i , \mathrm { O U T } } ^ { 2 } } { 2 n _ { i , \mathrm { O U T } } } \right) .
$$

The first two entries use the variance of the observed score means; the last two propagate the fixed FPC scaling of the fitted standard deviations. For a fixed threshold, the delta method gives

$$
\widehat { \mathrm { V a r } } [ A _ { i } ( t ; \hat { \omega } _ { i } ) ] = \pmb { g } _ { i } ( t ) ^ { \top } \widehat { V } _ { i } \pmb { g } _ { i } ( t ) , \qquad \pmb { g } _ { i } ( t ) = \nabla _ { \omega } A _ { i } ( t ; \hat { \omega } _ { i } ) ,
$$

and its standard error is the square root of this variance; we obtain the gradient by automatic diferentiation. At a unique interior maximum, $\partial A _ { i } / \partial t = 0$ , so the envelope theorem gives the first-order gradient of max $A _ { i } ( t ; \omega _ { i } )$ by evaluating ${ \bf \mathscr { g } } _ { i } ( t )$ at $\hat { t } _ { i } ^ { \phantom { \dagger } }$ , without diferentiating through the selected threshold. Thus the nominal record-wise 95% interval for the maximised fitted accuracy is

$$
\begin{array} { r } { \hat { A } _ { i } ^ { * } \pm 1 . 9 6 \sqrt { { g } _ { i } ( \hat { t } _ { i } ) ^ { \top } \widehat { V } _ { i } { g } _ { i } ( \hat { t } _ { i } ) } . } \end{array}
$$

For Fig. 1e, we clip its lower endpoint at 0.5 and plot the largest lower endpoint across records for each method. These intervals are pointwise for each record; choosing the largest endpoint across records does not provide a simultaneous 95% coverage guarantee.

## S6.2 Verification and Empirical Audits

We verify the private summary mechanism at three complementary levels: implementation, formal verification and empirical auditing. Together, these checks connect the mathematical privacy argument to the code that implements the mechanism. Empirical agreement cannot replace a proof, but it can reveal implementation errors or violated assumptions that the abstract proof alone cannot detect.

Implementation. DP-MHCA localises the sensitivity-critical query, key and value transformations, bounded attention computation and Gaussian perturbation in one module. The noise scale is calibrated from the number of private layers, attention heads and summary tokens according to the sensitivity and composition results in Theorems S3.1 and S3.2. We also check that every transformation before the first DP-MHCA layer is applied independently to each context row, with no positional encoding or cross-row normalisation that could create an unaccounted dependency. A self-contained PyTorch implementation that closely follows the implementation used in our experiments is provided in Supplement S3.3.

Perfect privacy Empirical ROC (mean) Empirical 95% CI eoretical µ-GDP

Formal verification. We provide a Lean 4 formalisation in DPMHCA.Lean (de Moura and Ullrich, 2021). It formally verifies the 2 Hm sensitivity bound for bounded attention and an abstract adaptive µ-GDP composition theorem that permits deterministic post-processing between private layers. This complements the manuscript proofs of the single-layer sensitivity bound and the privacy guarantee of the complete private summary mechanism. The correspondence between the formal objects and the running implementation remains subject to implementation review, motivating the additional empirical checks below.

Empirical audits. We perform two empirical checks. First, a gradient audit tests that computation before the first DP-MHCA layer is row-wise and detects unintended dependencies between context rows. Second, a sensitivity audit uses gradient-based optimisation to search for adjacent token sets whose pre-noise summaries are far apart. For each optimised pair, we repeatedly sample privatised outputs and estimate the hypothesis-testing trade-of curve using a likelihood-ratio test with access to the two unnoised means and the calibrated Gaussian variance. This gives the auditor the strongest test available for that pair. Agreement with the intended trade-of curve provides evidence against errors in sensitivity control or noise calibration, but is not a substitute for the formal guarantee. The full mechanism-audit plots across privacy levels are shown in Fig. S4.

![](images/cb264e6624485986a4235ec95d4d4a46438fe36e90e1f125d757b90478107dc7.jpg)  
(a) µ = 0.05

![](images/cfbb78f9102301b13c94f4ee6b92d1af511004967b48ac980c6989c0d2d8a569.jpg)  
(b) µ = 0.1

![](images/b6fe877cfa2c605f13365e69844b970a7bc0fbe2e624a63171e084a8abc4c832.jpg)  
(c) µ = 0.2

![](images/0b31c490680783eabc1baebc9f5a5ea780237b25e5772c0ddae261e823ef2452.jpg)  
(d) $\mu = 0 . 4$

![](images/484e8370c675d142ac685dcce42f96e00fbc7b226e769035833eae759b8683b3.jpg)  
(e) $\mu = 0 . 8$

![](images/17c0e209bc454d872b015cf648018aeb8f34e742716d36ea87e25ae05a9cde61.jpg)  
(f) $\mu = 1 . 6$

Figure S4: Empirical audit of the DP-MHCA mechanism across privacy levels. For each $\mu ,$ the audit optimises a pair of adjacent token sets to maximise the pre-noise output diference and then estimates the trade-of curve from repeated samples of the privatised outputs, using 10 repeated audit runs; shaded bands show pointwise bootstrap 95% CIs over those runs. The resulting curves are compared against the strongest likelihood-ratio attack with access to the unnoised means and the calibrated noise standard deviation. The observed behaviour is consistent with the intended privacy calibration across the full range of $\mu .$

## S6.3 TabArena Evaluation Details

## S6.3.1 Preprocessing and Elo calibration

The purpose of this benchmark is to compare the pure classification performance of the methods under a common preprocessing convention, so we treat z-score standardisation by population-level means and standard deviations as an idealised public preprocessing step and apply that convention to PrivTab and the baselines alike. This is also a natural benchmark idealisation because the TabICL simulator standardises each generated dataset using that dataset’s own mean and standard deviation before splitting it into context and target subsets. In the actual evaluation on real datasets, the unknown population means and standard deviations are approximated by the empirical mean and standard deviation of the context set, and those context-set statistics are then used to standardise both context and target features.

The DP-LR and DP-MLP baselines are described in the Methods. We rank PrivTab, DP-LR, DP-MLP and non-private logistic regression using the maximum-likelihood Elo procedure. For interpretability, the fitted ratings are linearly recalibrated so that DP-LR scores 1,000 separately at each privacy level and in the aggregate comparison. This transformation changes the origin of the Elo scale but not the ranking of the methods. We use $1 - \mathrm { A U C }$ as the error metric for binary datasets and log loss for multiclass datasets (Erickson et al., 2025). The dataset-level breakdowns evaluate discrimination on a common scale across all tasks. For dataset $d ,$ split r, privacy level $\mu$ and baseline $b ,$ they compute $\mathrm { A U C } _ { \operatorname* { P r i v T a b } , d , r , \mu } - \mathrm { A U C } _ { b , d , r , \mu } ,$ using macro-averaged one-versus-rest AUC when d is multiclass. Marker colours average this diference over the ten splits within a dataset, whereas panel titles average it over all 330 paired splits.

![](images/07058c5619391167d55ed621a07bc1104271adf3e29f642f445272a50cb982da.jpg)

![](images/1ad9983c1e2ce85d797502d30496bdac907a285664ed197d59be974c5a2d683f.jpg)

![](images/b7efbd24b3ec4d3fb9770e3b4c70324f236907fd9b313bc0e3c5c0d272acda0e.jpg)

![](images/0fa491645b11cef52fa1bfea4846202ff7153700497926e70f99d953c633905a.jpg)  
Figure S5: Dataset-level AUC gap between PrivTab and DP-MLP across all privacy levels. Each marker represents a dataset, positioned by its number of features and rows. Marker colour gives the mean paired AUC gap over ten context–target splits; positive values favour PrivTab. Binary AUC is used for binary datasets and macro-averaged one-versus-rest AUC for multiclass datasets.

![](images/2f285a10e7ee9cd38d40e8edda84ca1bd9f0f9476e5bfd5cd4962704bd3bf3d7.jpg)  
Figure S6: Dataset-level AUC gap between PrivTab and DP-LR across all privacy levels. Plotting conventions match Fig. S5; positive values favour PrivTab.

## S6.3.2 Derived comparison metrics

To interpret the comparison plots, let m denote a method, let b denote the reference baseline, let d denote a dataset, let r denote a context–target split, and let $\mu$ denote the privacy level. Writing $\mathrm { A U C } _ { m , d , r , \mu }$ and $\ell _ { m , d , r , \mu }$ for the corresponding target-set AUC and log loss, the split-level AUC improvement over the baseline is

$$
\Delta _ { \mathrm { A U C } } ( m , b ; d , r , \mu ) : = \mathrm { A U C } _ { m , d , r , \mu } - \mathrm { A U C } _ { b , d , r , \mu } .\tag{S125}
$$

For the non-private logistic-regression reference, we report the relative log-loss reduction

$$
\Delta _ { \mathrm { L L } } ^ { \mathrm { n o n D P } } ( m ; d , r , \mu ) : = \frac { \ell _ { \mathrm { n o n - p r i v a t e L R } , d , r } - \ell _ { m , d , r , \mu } } { \ell _ { \mathrm { n o n - p r i v a t e L R } , d , r } } ,\tag{S126}
$$

so positive values indicate lower loss than the non-private reference. Dividing by the reference loss makes the quantity comparable across datasets with diferent numbers of classes and hence diferent natural log-loss scales. For comparisons against DP-LR, we first average over context–target splits within each dataset and then report the relative log-loss reduction

$$
\begin{array} { r } { \Delta _ { \mathrm { L L } } ^ { \mathrm { r e l } } ( m , \mathrm { D P - L R } ; d , \mu ) : = \frac { \bar { \ell } _ { \mathrm { D P - L R } , d , \mu } - \bar { \ell } _ { m , d , \mu } } { \bar { \ell } _ { \mathrm { D P - L R } , d , \mu } } , } \end{array}\tag{S127}
$$

where $\bar { \ell } _ { m , d , \mu }$ denotes the average loss over context–target splits. Finally, the win-rate plots summarise the indicator comparisons

$$
W _ { \mathrm { A U C } } ( m , b ; d , r , \mu ) : = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { A U C } _ { m , d , r , \mu } > \mathrm { A U C } _ { b , d , r , \mu } , } \\ { 0 . 5 , } & { \mathrm { A U C } _ { m , d , r , \mu } = \mathrm { A U C } _ { b , d , r , \mu } , } \\ { 0 , } & { \mathrm { A U C } _ { m , d , r , \mu } < \mathrm { A U C } _ { b , d , r , \mu } , } \end{array} \right.\tag{S128}
$$

and analogously $W _ { \mathrm { L L } } ( m , b ; d , r , \mu )$ , where smaller log loss is better. Dataset-level win rates are obtained by averaging these quantities over context–target splits.

![](images/c04ea6ed6e983c2e08bb47bf4f3e365263e211bafb8b92d1f23b8b040033b5fa.jpg)  
Figure S7: AUC gaps versus feature count across 33 TabArena datasets. a, PrivTab minus DP-MLP AUC; b, PrivTab minus DP-LR AUC. Columns show privacy levels $\mu = 0 . 1 , 0 . 2$ and 0.4. Each point is a dataset’s mean paired AUC gap across ten context–target splits, plotted against its number of features on a logarithmic scale; positive gaps favour PrivTab. Circles denote binary datasets and squares denote multiclass datasets; AUC is macro-averaged one-versus-rest for multiclass tasks. Dashed horizontal lines mark equal AUC, grey lines are least-squares fits against log feature count, and annotations report Spearman’s ρ and its p-value

Table S7: Numerical values underlying the aggregate ELO coordinates in Figure 2(d), with ratings aggregated over all privacy budgets.
<table><tr><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>PrivTab</td><td>1166.5</td><td>1159.3</td><td>1173.7</td></tr><tr><td>DP-MLP</td><td>1097.5</td><td>1086.1</td><td>1109.2</td></tr><tr><td>DP-LR</td><td>1000.0</td><td>988.1</td><td>1011.9</td></tr></table>

## S6.4 Aggregate Benchmark Comparisons

Elo ratings, comparisons with non-private LR and dataset-level runtimes are reported in Tables S6 to S10.

Table S6: Numerical values underlying the ELO curves in Figure 2(c).
<table><tr><td>µ</td><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>0.05</td><td>PrivTab</td><td>1142.8</td><td>1125.0</td><td>1161.3</td></tr><tr><td>0.05</td><td>DP-LR</td><td>1000.0</td><td>970.4</td><td>1027.8</td></tr><tr><td>0.05</td><td>DP-MLP</td><td>910.7</td><td>882.1</td><td>937.9</td></tr><tr><td>0.1</td><td>PrivTab</td><td>1172.4</td><td>1155.4</td><td>1190.4</td></tr><tr><td>0.1</td><td>DP-LR</td><td>1000.0</td><td>969.7</td><td>1028.7</td></tr><tr><td>0.1</td><td>DP-MLP</td><td>966.7</td><td>939.5</td><td>993.0</td></tr><tr><td>0.2</td><td>PrivTab</td><td>1189.5</td><td>1171.7</td><td>1208.3</td></tr><tr><td>0.2</td><td>DP-MLP</td><td>1038.7</td><td>1011.6</td><td>1065.2</td></tr><tr><td>0.2</td><td>DP-LR</td><td>1000.0</td><td>968.9</td><td>1029.0</td></tr><tr><td>0.4</td><td>PrivTab</td><td>1185.7</td><td>1168.3</td><td>1204.3</td></tr><tr><td>0.4</td><td>DP-MLP</td><td>1131.1</td><td>1104.6</td><td>1158.7</td></tr><tr><td>0.4</td><td>DP-LR</td><td>1000.0</td><td>968.7</td><td>1030.3</td></tr><tr><td>0.8</td><td>DP-MLP</td><td>1243.5</td><td>1216.3</td><td>1273.3</td></tr><tr><td>0.8</td><td>PrivTab</td><td>1174.6</td><td>1157.4</td><td>1193.0</td></tr><tr><td>0.8</td><td>DP-LR</td><td>1000.0</td><td>968.2</td><td>1029.8</td></tr><tr><td>1.6</td><td>DP-MLP</td><td>1296.3</td><td>1265.8</td><td>1330.9</td></tr><tr><td>1.6</td><td>PrivTab</td><td>1155.8</td><td>1136.8</td><td>1176.1</td></tr><tr><td>1.6</td><td>DP-LR</td><td>1000.0</td><td>969.9</td><td>1029.4</td></tr></table>

Table S8: Numerical values underlying Figure 2(a).
<table><tr><td>Method</td><td>µ</td><td>Mean</td><td>CI low</td><td>CI high</td><td>n datasets</td><td>n splits</td></tr><tr><td>DP-LR</td><td>0.05</td><td>-0.1249</td><td>-0.1579</td><td>-0.0950</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.1</td><td>-0.0933</td><td>-0.1210</td><td>-0.0669</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.2</td><td>-0.0663</td><td>-0.0880</td><td>-0.0460</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.4</td><td>-0.0471</td><td>-0.0643</td><td>-0.0314</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.8</td><td>-0.0335</td><td>-0.0471</td><td>-0.0211</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>1.6</td><td>-0.0238</td><td>-0.0359</td><td>-0.0131</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.05</td><td>-0.1924</td><td>-0.2359</td><td>-0.1519</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.1</td><td>-0.1482</td><td>-0.1883</td><td>-0.1092</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.2</td><td>-0.0948</td><td>-0.1255</td><td>-0.0656</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.4</td><td>-0.0505</td><td>-0.0709</td><td>-0.0320</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.8</td><td>-0.0169</td><td>-0.0312</td><td>-0.0030</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>1.6</td><td>0.0007</td><td>-0.0120</td><td>0.0137</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.05</td><td>-0.1222</td><td>-0.1586</td><td>-0.0876</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.1</td><td>-0.0773</td><td>-0.1025</td><td>-0.0532</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.2</td><td>-0.0472</td><td>-0.0665</td><td>-0.0297</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.4</td><td>-0.0255</td><td>-0.0386</td><td>-0.0132</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.8</td><td>-0.0150</td><td>-0.0273</td><td>-0.0036</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>1.6</td><td>-0.0088</td><td>-0.0201</td><td>0.0014</td><td>33</td><td>10</td></tr></table>

Table S9: Numerical values underlying the relative log-loss improvements in Figure 2(b).
<table><tr><td>Method</td><td>µ</td><td>Mean</td><td>CI low</td><td>CI high</td><td>n datasets</td><td>n splits</td></tr><tr><td>DP-LR</td><td>0.05</td><td>-12.0599</td><td>-20.0871</td><td>-6.3234</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.1</td><td>-6.4823</td><td>-10.8479</td><td>-3.1667</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.2</td><td>-3.1523</td><td>-4.8574</td><td>-1.8595</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.4</td><td>-1.5109</td><td>-1.8378</td><td>-1.2354</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>0.8</td><td>-1.1520</td><td>-1.2902</td><td>-1.0277</td><td>33</td><td>10</td></tr><tr><td>DP-LR</td><td>1.6</td><td>-1.0440</td><td>-1.1426</td><td>-0.9516</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.05</td><td>-1.5079</td><td>-2.5324</td><td>-0.8257</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.1</td><td>-1.0130</td><td>-1.6921</td><td>-0.5475</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.2</td><td>-0.7080</td><td>-1.3381</td><td>-0.3270</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.4</td><td>-0.3822</td><td>-0.6844</td><td>-0.1853</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>0.8</td><td>-0.1543</td><td>-0.3202</td><td>-0.0466</td><td>33</td><td>10</td></tr><tr><td>DP-MLP</td><td>1.6</td><td>-0.0560</td><td>-0.1257</td><td>-0.0007</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.05</td><td>-0.5217</td><td>-0.8835</td><td>-0.2618</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.1</td><td>-0.4293</td><td>-0.7594</td><td>-0.1951</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.2</td><td>-0.3435</td><td>-0.6366</td><td>-0.1307</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.4</td><td>-0.2787</td><td>-0.5365</td><td>-0.0878</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>0.8</td><td>-0.2218</td><td>-0.4303</td><td>-0.0657</td><td>33</td><td>10</td></tr><tr><td>PrivTab</td><td>1.6</td><td>-0.1926</td><td>-0.3746</td><td>-0.0561</td><td>33</td><td>10</td></tr></table>

Table S10: Dataset sizes and per-context–target-split runtimes in seconds underlying Figure S8.
<table><tr><td>Dataset</td><td>Rows</td><td>Features</td><td>PrivTab</td><td>DP-MLP</td><td>DP-LR</td></tr><tr><td>Amazon employee access</td><td>32769</td><td>9</td><td>0.037</td><td>524.350</td><td>1540.025</td></tr><tr><td>Anneal</td><td>898</td><td>31</td><td>0.012</td><td>92.250</td><td>284.975</td></tr><tr><td>Bank marketing</td><td>45211</td><td>13</td><td>0.021</td><td>724.675</td><td>2105.300</td></tr><tr><td>Bank customer churn</td><td>10000</td><td>10</td><td>0.013</td><td>202.975</td><td>616.075</td></tr><tr><td>Blood transfusion</td><td>748</td><td>4</td><td>0.013</td><td>91.700</td><td>299.800</td></tr><tr><td>Churn</td><td>5000</td><td>19</td><td>0.013</td><td>145.150</td><td>452.825</td></tr><tr><td>COIL2000</td><td>9822</td><td>85</td><td>0.013</td><td>202.075</td><td>624.475</td></tr><tr><td>Credit-g</td><td>1000</td><td>20</td><td>0.013</td><td>90.425</td><td>300.750</td></tr><tr><td>Credit card default</td><td>30000</td><td>23</td><td>0.015</td><td>498.500</td><td>1401.675</td></tr><tr><td>Diabetes</td><td>768</td><td>8</td><td>0.013</td><td>92.350</td><td>296.650</td></tr><tr><td>Diabetes130US</td><td>71518</td><td>44</td><td>0.030</td><td>1067.925</td><td>3078.550</td></tr><tr><td>E-commerce shipping</td><td>10999</td><td>10</td><td>0.013</td><td>208.600</td><td>634.600</td></tr><tr><td>Fitness club</td><td>1500</td><td>6</td><td>0.013</td><td>92.125</td><td>311.000</td></tr><tr><td>Hazelnut contaminant</td><td>2400</td><td>30</td><td>0.013</td><td>93.250</td><td>299.800</td></tr><tr><td>HELOC</td><td>10459</td><td>23</td><td>0.013</td><td>203.550</td><td>629.150</td></tr><tr><td>HR analytics job change</td><td>19158</td><td>12</td><td>0.014</td><td>321.525</td><td>948.675</td></tr><tr><td>In-vehicle coupon</td><td>12684</td><td>24</td><td>0.014</td><td>241.775</td><td>753.800</td></tr><tr><td>Good customer</td><td>1723</td><td>13</td><td>0.013</td><td>93.250</td><td>297.550</td></tr><tr><td>Marketing campaign</td><td>2240</td><td>25</td><td>0.013</td><td>93.250</td><td>301.350</td></tr><tr><td>Maternal health risk</td><td>1014</td><td>6</td><td>0.012</td><td>93.250</td><td>300.450</td></tr><tr><td>NATICUSdroid</td><td>7491</td><td>86</td><td>0.013</td><td>172.000</td><td>533.475</td></tr><tr><td>Online shoppers</td><td>12330</td><td>17</td><td>0.014</td><td>238.725</td><td>713.125</td></tr><tr><td>Polish bankruptcy</td><td>5910</td><td>64</td><td>0.013</td><td>150.850</td><td>454.375</td></tr><tr><td>QSAR biodeg</td><td>1054</td><td>41</td><td>0.013</td><td>92.475</td><td>307.175</td></tr><tr><td>SDSS17</td><td>78053</td><td>11</td><td>0.029</td><td>1151.625</td><td>3282.500</td></tr><tr><td>Seismic bumps</td><td>2584</td><td>15</td><td>0.013</td><td>91.850</td><td>296.300</td></tr><tr><td>Splice</td><td>3190</td><td>60</td><td>0.012</td><td>100.950</td><td>316.675</td></tr><tr><td>Student dropout</td><td>4424</td><td>36</td><td>0.012</td><td>138.450</td><td>447.400</td></tr><tr><td>Taiwanese bankruptcy</td><td>6819</td><td>94</td><td>0.013</td><td>156.200</td><td>485.800</td></tr><tr><td>Website phishing</td><td>1353</td><td>9</td><td>0.012</td><td>93.975</td><td>300.450</td></tr><tr><td>Wine quality</td><td>6497</td><td>12</td><td>0.012</td><td>156.200</td><td>480.025</td></tr><tr><td>MIC</td><td>1699</td><td>111</td><td>0.012</td><td>91.650</td><td>303.275</td></tr><tr><td>JM1</td><td>10885</td><td>21</td><td>0.013</td><td>204.850</td><td>629.275</td></tr></table>

![](images/82cdb872e436c63de13758ac3944f9942099eba5921075bca95c1c95724a9f35.jpg)  
Figure S8: Fitting time as a function of dataset size. Points are dataset-level measurements and curves are fitted shifted power laws. PrivTab is approximately four orders of magnitude faster than the DP Adam baselines over the studied range.

Prediction time on a laptop. We additionally measured post-fit prediction time on a laptop with a 13thgeneration Intel Core i5 CPU across the same 33 datasets, six privacy levels and ten context–target splits. Each configuration used one warm-up call followed by three timed runs, with a target batch size of 1,024; we report the mean of the per-configuration median times. Timing covers prediction for all target rows, and excludes preprocessing, model loading, fitting and private-summary construction. These measurements assess prediction cost rather than predictive performance. The dataset-level results are reported in Table S11 and Fig. S9.

Table S11: Dataset sizes and post-fit prediction times in milliseconds on a laptop with a 13th-generation Intel Core i5 CPU, underlying Figure S9. Values average the per-configuration median of three timed runs over six privacy levels and ten context–target splits.
<table><tr><td>Dataset</td><td>Target rows</td><td>Features</td><td>PrivTab</td><td>DP-MLP</td><td>DP-LR</td></tr><tr><td>Amazon employee access</td><td>6554</td><td>9</td><td>355</td><td>1.73</td><td>0.727</td></tr><tr><td>Anneal</td><td>180</td><td>31</td><td>15.2</td><td>0.159</td><td>0.114</td></tr><tr><td>Bank marketing</td><td>9042</td><td>13</td><td>582</td><td>2.83</td><td>0.928</td></tr><tr><td>Bank customer churn</td><td>2000</td><td>10</td><td>105</td><td>0.585</td><td>0.2</td></tr><tr><td>Blood transfusion</td><td>150</td><td>4</td><td>13</td><td>0.124</td><td>0.0849</td></tr><tr><td>Churn</td><td>1000</td><td>19</td><td>64.4</td><td>0.346</td><td>0.115</td></tr><tr><td>COIL2000</td><td>1964</td><td>85</td><td>106</td><td>0.549</td><td>0.192</td></tr><tr><td>Credit-g</td><td>200</td><td>20</td><td>15.1</td><td>0.0715</td><td>0.0381</td></tr><tr><td>Credit card default</td><td>6000</td><td>23</td><td>394</td><td>1.7</td><td>0.664</td></tr><tr><td>Diabetes</td><td>154</td><td>8</td><td>11.9</td><td>0.107</td><td>0.0687</td></tr><tr><td>Diabetes130US</td><td>14304</td><td>44</td><td>754</td><td>3.62</td><td>1.12</td></tr><tr><td>E-commerce shipping</td><td>2200</td><td>10</td><td>112</td><td>0.588</td><td>0.259</td></tr><tr><td>Fitness club</td><td>300</td><td>6</td><td>19.1</td><td>0.164</td><td>0.116</td></tr><tr><td>Hazelnut contaminant</td><td>480</td><td>30</td><td>29.8</td><td>0.232</td><td>0.182</td></tr><tr><td>HELOC</td><td>2092</td><td>23</td><td>111</td><td>0.605</td><td>0.245</td></tr><tr><td>HR analytics job change</td><td>3832</td><td>12</td><td>197</td><td>1.06</td><td>0.361</td></tr><tr><td>In-vehicle coupon</td><td>2537</td><td>24</td><td>147</td><td>0.772</td><td>0.355</td></tr><tr><td>Good customer</td><td>345</td><td>13</td><td>24.1</td><td>0.174</td><td>0.123</td></tr><tr><td>JM1</td><td>2177</td><td>21</td><td>127</td><td>1.67</td><td>1.06</td></tr><tr><td>Marketing campaign</td><td>448</td><td>25</td><td>34.5</td><td>0.212</td><td>0.152</td></tr><tr><td>Maternal health risk</td><td>203</td><td>6</td><td>18.9</td><td>0.13</td><td>0.062</td></tr><tr><td>MIC</td><td>340</td><td>111</td><td>35.4</td><td>0.243</td><td>0.0644</td></tr><tr><td>NATICUSdroid</td><td>1498</td><td>86</td><td>96.4</td><td>0.653</td><td>0.233</td></tr><tr><td>Online shoppers</td><td>2466</td><td>17</td><td>142</td><td>0.661</td><td>0.313</td></tr><tr><td>Polish bankruptcy</td><td>1182</td><td>64</td><td>66.6</td><td>0.354</td><td>0.17</td></tr><tr><td>QSAR biodeg</td><td>211</td><td>41</td><td>18.8</td><td>0.114</td><td>0.0754</td></tr><tr><td>SDSS17</td><td>15611</td><td>11</td><td>805</td><td>3.46</td><td>1.46</td></tr><tr><td>Seismic bumps</td><td>517</td><td>15</td><td>28.3</td><td>0.173</td><td>0.116</td></tr><tr><td>Splice</td><td>638</td><td>60</td><td>32.7</td><td>0.202</td><td>0.0834</td></tr><tr><td>Student dropout</td><td>885</td><td>36</td><td>46.9</td><td>0.196</td><td>0.207</td></tr><tr><td>Taiwanese bankruptcy</td><td>1364</td><td>94</td><td>81.8</td><td>0.448</td><td>0.528</td></tr><tr><td>Website phishing</td><td>271</td><td>9</td><td>20.9</td><td>0.244</td><td>0.133</td></tr><tr><td>Wine quality</td><td>1299</td><td>12</td><td>76.5</td><td>1.09</td><td>0.18</td></tr></table>

![](images/ad4a6a6bdce26346a44b4ad3c7f3cfe0cc2b7f1807d1f822918fdc8ae20f6af2.jpg)  
Figure S9: Prediction time as a function of target-set size on a laptop CPU. Points show dataset-level mean post-fit prediction times in milliseconds (ms) on a 13th-generation Intel Core i5, averaged over six privacy levels and ten context–target splits. Curves are shifted power laws fitted in log time as visual guides. Both axes are logarithmic. All methods predict in batches of up to 1,024 target rows.

The following figure summarises the direct comparisons with DP-LR; the corresponding numerical values are reported in Tables S12 to S15.

![](images/2ddc4538673d6901b4b37c5e140f43c99ef20dbbeeebf03ec5b3dff9b0665d94.jpg)  
(a) Mean AUC improvement.

![](images/2352a68149537a902458f33c2e16c277ea6756c1e02269d3cdd3ba2a8242fe8c.jpg)  
(b) AUC win rate.

![](images/ba459d96504bf6bd09f3156da2c2571e437e50eb662ed603e519896d34d5bcce.jpg)  
(c) Relative log-loss reduction.

![](images/07dcfb00f085d37cc1657f362eedd717716be3f90fa1647557e395702aabf9bb.jpg)  
(d) Log-loss win rate.  
Figure S10: Performance relative to DP-LR. Results aggregate ten context–target splits per dataset; error bars show bootstrap 95% CIs over datasets. Positive improvements and win rates above 0.5 favour the compared method. Bands are visual guides, not formal privacy thresholds: strong $( \mu < 0 . 1 5 )$ , moderate $( 0 . 1 5 \leq \mu < 0 . 6 )$ and weak $( \mu \geq 0 . 6 )$ formal privacy.

Table S12: Numerical values underlying Figure S10(a) (AUC improvement over DP-LR).
<table><tr><td>Method</td><td>μ</td><td>Mean</td><td>CI low</td><td>CI high</td><td>n datasets</td></tr><tr><td>PrivTab</td><td>0.05</td><td>0.0027</td><td>-0.0198</td><td>0.0257</td><td>33</td></tr><tr><td>PrivTab</td><td>0.1</td><td>0.0160</td><td>-0.0011</td><td>0.0352</td><td>33</td></tr><tr><td>PrivTab</td><td>0.2</td><td>0.0192</td><td>0.0040</td><td>0.0355</td><td>33</td></tr><tr><td>PrivTab</td><td>0.4</td><td>0.0216</td><td>0.0086</td><td>0.0355</td><td>33</td></tr><tr><td>PrivTab</td><td>0.8</td><td>0.0185</td><td>0.0068</td><td>0.0307</td><td>33</td></tr><tr><td>PrivTab</td><td>1.6</td><td>0.0150</td><td>0.0035</td><td>0.0280</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.05</td><td>-0.0675</td><td>-0.0871</td><td>-0.0476</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.1</td><td>-0.0548</td><td>-0.0758</td><td>-0.0342</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.2</td><td>-0.0284</td><td>-0.0486</td><td>-0.0087</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.4</td><td>-0.0034</td><td>-0.0184</td><td>0.0108</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.8</td><td>0.0166</td><td>0.0060</td><td>0.0283</td><td>33</td></tr><tr><td>DP-MLP</td><td>1.6</td><td>0.0245</td><td>0.0133</td><td>0.0376</td><td>33</td></tr></table>

Table S13: Numerical values underlying Figure S10(b) (AUC win rate over DP-LR).
<table><tr><td>Method</td><td>µ</td><td>Mean</td><td>CI low</td><td>CI high</td><td>n datasets</td></tr><tr><td>PrivTab</td><td>0.05</td><td>0.5455</td><td>0.4485</td><td>0.6424</td><td>33</td></tr><tr><td>PrivTab</td><td>0.1</td><td>0.5939</td><td>0.4909</td><td>0.6970</td><td>33</td></tr><tr><td>PrivTab</td><td>0.2</td><td>0.5788</td><td>0.4576</td><td>0.6939</td><td>33</td></tr><tr><td>PrivTab</td><td>0.4</td><td>0.6303</td><td>0.5061</td><td>0.7455</td><td>33</td></tr><tr><td>PrivTab</td><td>0.8</td><td>0.6424</td><td>0.5152</td><td>0.7606</td><td>33</td></tr><tr><td>PrivTab</td><td>1.6</td><td>0.6273</td><td>0.5000</td><td>0.7545</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.05</td><td>0.2697</td><td>0.2000</td><td>0.3424</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.1</td><td>0.3333</td><td>0.2394</td><td>0.4303</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.2</td><td>0.4485</td><td>0.3394</td><td>0.5636</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.4</td><td>0.5697</td><td>0.4455</td><td>0.6879</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.8</td><td>0.6485</td><td>0.5242</td><td>0.7636</td><td>33</td></tr><tr><td>DP-MLP</td><td>1.6</td><td>0.7727</td><td>0.6818</td><td>0.8606</td><td>33</td></tr></table>

Table S14: Numerical values underlying Figure S10(c) (Relative log-loss reduction over DP-LR).
<table><tr><td>Method</td><td>µ</td><td>Mean</td><td>CI low</td><td>CI high</td><td>n datasets</td></tr><tr><td>PrivTab</td><td>0.05</td><td>0.7061</td><td>0.6100</td><td>0.7827</td><td>33</td></tr><tr><td>PrivTab</td><td>0.1</td><td>0.6333</td><td>0.5370</td><td>0.7092</td><td>33</td></tr><tr><td>PrivTab</td><td>0.2</td><td>0.5598</td><td>0.4586</td><td>0.6342</td><td>33</td></tr><tr><td>PrivTab</td><td>0.4</td><td>0.4684</td><td>0.3679</td><td>0.5398</td><td>33</td></tr><tr><td>PrivTab</td><td>0.8</td><td>0.4292</td><td>0.3305</td><td>0.4971</td><td>33</td></tr><tr><td>PrivTab</td><td>1.6</td><td>0.4109</td><td>0.3036</td><td>0.4811</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.05</td><td>0.6392</td><td>0.5725</td><td>0.7031</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.1</td><td>0.5712</td><td>0.5105</td><td>0.6295</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.2</td><td>0.5263</td><td>0.4869</td><td>0.5669</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.4</td><td>0.4502</td><td>0.4050</td><td>0.4943</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.8</td><td>0.4688</td><td>0.4317</td><td>0.5027</td><td>33</td></tr><tr><td>DP-MLP</td><td>1.6</td><td>0.4812</td><td>0.4547</td><td>0.5067</td><td>33</td></tr></table>

Table S15: Numerical values underlying Figure S10(d) (Log-loss win rate over DP-LR).
<table><tr><td>Method</td><td> $\mu$ </td><td>Mean</td><td>CI low</td><td>CI high</td><td>n datasets</td></tr><tr><td>PrivTab</td><td>0.05</td><td>0.9545</td><td>0.8788</td><td>1.0000</td><td>33</td></tr><tr><td>PrivTab</td><td>0.1</td><td>0.9606</td><td>0.8909</td><td>1.0000</td><td>33</td></tr><tr><td>PrivTab</td><td>0.2</td><td>0.9667</td><td>0.9030</td><td>1.0000</td><td>33</td></tr><tr><td>PrivTab</td><td>0.4</td><td>0.9424</td><td>0.8606</td><td>1.0000</td><td>33</td></tr><tr><td>PrivTab</td><td>0.8</td><td>0.9485</td><td>0.8667</td><td>1.0000</td><td>33</td></tr><tr><td>PrivTab</td><td>1.6</td><td>0.9455</td><td>0.8606</td><td>1.0000</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.05</td><td>0.9121</td><td>0.8636</td><td>0.9545</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.1</td><td>0.9364</td><td>0.9000</td><td>0.9667</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.2</td><td>0.9364</td><td>0.8848</td><td>0.9788</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.4</td><td>0.9455</td><td>0.8939</td><td>0.9818</td><td>33</td></tr><tr><td>DP-MLP</td><td>0.8</td><td>0.9818</td><td>0.9515</td><td>1.0000</td><td>33</td></tr><tr><td>DP-MLP</td><td>1.6</td><td>0.9848</td><td>0.9636</td><td>1.0000</td><td>33</td></tr></table>

## S6.5 Per-dataset Benchmark Results

Per-dataset AUC and log loss are reported for PrivTab in Tables S16 and S17, DP-LR in Tables S18 and S19, DP-MLP in Tables S20 and S21 and non-private LR in Tables S22 and S23.

Table S16: Per-dataset AUC for PrivTab; entries report the mean and bootstrap 95% CI over context–target splits.
<table><tr><td>Dataset</td><td> $\mu = 0 . 0 5$ </td><td> $\mu = 0 . 1$ </td><td> $\mu = 0 . 2$ </td><td> $\mu = 0 . 4$ </td><td> $\mu = 0 . 8$ </td><td> $\mu = 1 . 6$ </td></tr><tr><td>Amazon employee access</td><td>0.562 [0.553,0.571]</td><td>0.560 [0.546,0.573]</td><td>0.570 [0.561,0.579]</td><td>0.569 [0.561,0.578]</td><td>0.569 [0.561,0.577]</td><td>0.570 [0.563,0.576]</td></tr><tr><td>Anneal</td><td>0.638 [0.596,0.673]</td><td>0.685 [0.660,0.708]</td><td>0.735 [0.721,0.749]</td><td>0.827 [0.812,0.844]</td><td>0.852 [0.827,0.875]</td><td>0.882 [0.861,0.906]</td></tr><tr><td>Bank customer churn</td><td>0.753 [0.739,0.766]</td><td>0.776 [0.769,0.783]</td><td>0.785 [0.779,0.792]</td><td>0.790 [0.783,0.797]</td><td>0.793 [0.786,0.800]</td><td>0.794 [0.786,0.801]</td></tr><tr><td>Bank marketing</td><td>0.715 [0.710,0.720]</td><td>0.726 [0.723,0.728]</td><td>0.729 [0.725,0.732]</td><td>0.730 [0.726,0.734]</td><td>0.730 [0.726,0.734]</td><td>0.730 [0.726,0.733]</td></tr><tr><td>Blood transfusion</td><td>0.532 [0.479,0.588]</td><td>0.616 [0.577,0.652]</td><td>0.656 [0.625,0.685]</td><td>0.709 [0.686,0.733]</td><td>0.725 [0.704,0.750]</td><td>0.734 [0.712,0.759]</td></tr><tr><td></td><td>0.748</td><td>0.797</td><td>0.820</td><td>0.839</td><td>0.842</td><td>0.839</td></tr><tr><td>Churn</td><td>[0.722,0.770]</td><td>[0.786,0.808]</td><td>[0.815,0.825]</td><td>[0.833,0.845]</td><td>[0.835,0.850]</td><td>[0.831,0.847]</td></tr><tr><td>COIL2000</td><td>0.603 [0.587,0.620]</td><td>0.625 [0.613,0.636]</td><td>0.650 [0.627,0.669]</td><td>0.692 [0.677,0.706]</td><td>0.700 [0.689,0.711]</td><td>0.701 [0.690,0.712]</td></tr><tr><td>Credit card default</td><td>0.743 [0.737,0.748]</td><td>0.742 [0.736,0.748]</td><td>0.750 [0.745,0.755]</td><td>0.750 [0.746,0.754]</td><td>0.752 [0.747,0.756]</td><td>0.751 [0.747,0.756]</td></tr><tr><td>Credit-g</td><td>0.584 [0.537,0.633]</td><td>0.602 [0.545,0.653]</td><td>0.687 [0.655,0.719]</td><td>0.707 [0.682,0.735]</td><td>0.727 [0.705,0.747]</td><td>0.733 [0.707,0.757]</td></tr><tr><td>Diabetes</td><td>0.574 [0.517,0.624]</td><td>0.703 [0.674,0.738]</td><td>0.771 [0.753,0.789]</td><td>0.814 [0.795,0.831]</td><td>0.824 [0.808,0.839]</td><td>0.828 [0.814,0.840]</td></tr><tr><td>Diabetes130US</td><td>0.562 [0.554,0.571]</td><td>0.565 [0.558,0.571]</td><td>0.564 [0.557,0.570]</td><td>0.565 [0.560,0.570]</td><td>0.567 [0.561,0.572]</td><td>0.567 [0.562,0.571]</td></tr><tr><td>E-commerce shipping</td><td>0.716 [0.711,0.721]</td><td>0.722 [0.715,0.728]</td><td>0.723 [0.715,0.730]</td><td>0.725 [0.719,0.731]</td><td>0.725</td><td>0.725</td></tr><tr><td>Fitness club</td><td>0.646 [0.609,0.685]</td><td>0.708 [0.690,0.726]</td><td>0.747 [0.730,0.765]</td><td>0.776 [0.764,0.788]</td><td>[0.719,0.731] 0.786</td><td>[0.719,0.730] 0.798</td></tr><tr><td>Good customer</td><td>0.487 [0.454,0.516]</td><td>0.551 [0.519,0.585]</td><td>0.576 [0.538,0.616]</td><td>0.668</td><td>[0.774,0.798] 0.694</td><td>[0.789,0.807] 0.711</td></tr><tr><td>Hazelnut contaminant</td><td>0.837 [0.817,0.856]</td><td>0.862 [0.852,0.873]</td><td>0.871 [0.860,0.881]</td><td>[0.647,0.690] 0.882</td><td>[0.674,0.715] 0.893</td><td>[0.689,0.734] 0.897</td></tr><tr><td>HELOC</td><td>0.765 [0.759,0.770]</td><td>0.775 [0.771,0.780]</td><td>0.782 [0.778,0.787]</td><td>[0.875,0.891] 0.785 [0.781,0.789]</td><td>[0.885,0.901] 0.785</td><td>[0.890,0.903] 0.786</td></tr><tr><td>HR analytics job change</td><td>0.751 [0.746,0.756]</td><td>0.758 [0.755,0.762]</td><td>0.763 [0.760,0.766]</td><td>0.768 [0.764,0.771]</td><td>[0.782,0.789] 0.768 [0.765,0.771]</td><td>[0.782,0.790] 0.769 [0.765,0.772]</td></tr><tr><td>In-vehicle coupon</td><td>0.642 [0.635,0.648]</td><td>0.659 [0.652,0.666]</td><td>0.668 [0.663,0.673]</td><td>0.671 [0.666,0.676]</td><td>0.673 [0.667,0.679]</td><td>0.673 [0.668,0.679]</td></tr><tr><td>JM1</td><td>0.684 [0.672,0.696]</td><td>0.700 [0.689,0.712]</td><td>0.707 [0.699,0.718]</td><td>0.709 [0.701,0.718]</td><td>0.711 [0.703,0.720]</td><td>0.709 [0.701,0.719]</td></tr><tr><td>Marketing campaign</td><td>0.735 [0.722,0.752]</td><td>0.752 [0.720,0.783]</td><td>0.803 [0.790,0.817]</td><td>0.830 [0.812,0.846]</td><td>0.845 [0.828,0.860]</td><td>0.853 [0.837,0.869]</td></tr><tr><td>Maternal health risk</td><td>0.603 [0.528,0.670]</td><td>0.694 [0.664,0.721]</td><td>0.761 [0.748,0.773]</td><td>0.787 [0.773,0.802]</td><td>0.809 [0.795,0.823]</td><td>0.817 [0.804,0.832]</td></tr><tr><td>MIC</td><td>0.661 [0.634,0.687]</td><td>0.694 [0.676,0.711]</td><td>0.717 [0.690,0.742]</td><td>0.720 [0.701,0.737]</td><td>0.745 [0.726,0.763]</td><td>0.767 [0.741,0.789]</td></tr><tr><td>NATICUSdroid</td><td>0.947 [0.940,0.953]</td><td>0.958 [0.955,0.962]</td><td>0.962 [0.958,0.965]</td><td>0.963 [0.960,0.966]</td><td>0.963 [0.959,0.967]</td><td>0.962 [0.959,0.965]</td></tr><tr><td>Online shoppers</td><td>0.826 [0.817,0.834]</td><td>0.866 [0.860,0.873]</td><td>0.882 [0.878,0.886]</td><td>0.887 [0.883,0.891]</td><td>0.888 [0.883,0.891]</td><td>0.889 [0.885,0.892]</td></tr><tr><td>Polish bankruptcy</td><td>0.673 [0.625,0.711]</td><td>0.725 [0.691,0.758]</td><td>0.771 [0.750,0.792]</td><td>0.793 [0.771,0.814]</td><td>0.798 [0.775,0.821]</td><td>0.798 [0.777,0.819]</td></tr><tr><td>QSAR biodeg</td><td>0.599 [0.535,0.668]</td><td>0.792 [0.768,0.814]</td><td>0.869 [0.857,0.880]</td><td>0.884 [0.867,0.896]</td><td>0.898 [0.881,0.912]</td><td>0.913 [0.900,0.925]</td></tr><tr><td>SDSS17</td><td>0.950 [0.947,0.953]</td><td>0.954 [0.951,0.956]</td><td>0.953 [0.952,0.954]</td><td>0.953 [0.952,0.954]</td><td>0.953 [0.952,0.954]</td><td>0.953 [0.952,0.954]</td></tr><tr><td>Seismic bumps</td><td>0.705 [0.678,0.733]</td><td>0.729 [0.712,0.745]</td><td>0.712 [0.691,0.735]</td><td>0.742 [0.717,0.767]</td><td>0.746 [0.726,0.769]</td><td>0.760 [0.740,0.782]</td></tr><tr><td>Splice</td><td>0.576 [0.545,0.606]</td><td>0.720 [0.696,0.743]</td><td>0.853 [0.841,0.863]</td><td>0.894 [0.888,0.900]</td><td>0.915 [0.910,0.919]</td><td>0.926 [0.923,0.929]</td></tr><tr><td>Student dropout</td><td>0.735 [0.716,0.756]</td><td>0.801 [0.793,0.808]</td><td>0.832 [0.823,0.839]</td><td>0.853 [0.846,0.860]</td><td>0.862 [0.856,0.868]</td><td>0.865 [0.860,0.869]</td></tr><tr><td>Taiwanese bankruptcy</td><td>0.740 [0.690,0.780]</td><td>0.801 [0.760,0.834]</td><td>0.849 [0.820,0.874]</td><td>0.898 [0.882,0.912]</td><td>0.924 [0.914,0.934]</td><td>0.929 [0.920,0.938]</td></tr><tr><td>Website phishing</td><td>0.666 [0.636,0.695]</td><td>0.768 [0.746,0.790]</td><td>0.794 [0.776,0.814]</td><td>0.834 [0.816,0.851]</td><td>0.871 [0.862,0.880]</td><td>0.894 [0.885,0.905]</td></tr><tr><td>Wine quality</td><td>0.578 [0.548,0.607]</td><td>0.633 [0.602,0.661]</td><td>0.698 [0.664,0.731]</td><td>0.714 [0.688,0.739]</td><td>0.743 [0.714,0.772]</td><td>0.755 [0.742,0.769]</td></tr></table>

Table S17: Per-dataset log loss for PrivTab; entries report the mean and bootstrap 95% CI over context–target splits.
<table><tr><td>Dataset</td><td> $\mu = 0 . 0 5$ </td><td>µ = 0.1</td><td> $\mu = 0 . 2$ </td><td> $\mu = 0 . 4$ </td><td> $\mu = 0 . 8$ </td><td> $\mu = 1 . 6$ </td></tr><tr><td>Amazon employee access</td><td>0.230 [0.228,0.232]</td><td>0.232 [0.231,0.234]</td><td>0.232 [0.231,0.232]</td><td>0.232 [0.232,0.232]</td><td>0.232 [0.231,0.232]</td><td>0.232 [0.231,0.232]</td></tr><tr><td>Anneal</td><td>0.968 [0.926,1.015]</td><td>0.890 [0.863,0.919]</td><td>0.791 [0.765,0.819]</td><td>0.702</td><td>0.538 [0.518,0.557]</td><td>0.432</td></tr><tr><td></td><td>0.450</td><td>0.433</td><td>0.425</td><td>[0.689,0.715] 0.420</td><td>0.420</td><td>[0.418,0.446] 0.420</td></tr><tr><td>Bank customer churn Bank marketing</td><td>[0.444,0.457] 0.345 [0.344,0.346]</td><td>[0.428,0.438] 0.341</td><td>[0.422,0.429] 0.339</td><td>[0.416,0.423] 0.338</td><td>[0.416,0.423] 0.337</td><td>[0.417,0.424] 0.337</td></tr><tr><td>Blood transfusion</td><td>0.562</td><td>[0.340,0.342] 0.541</td><td>[0.338,0.340] 0.532</td><td>[0.337,0.339] 0.506</td><td>[0.337,0.338] 0.497</td><td>[0.337,0.338] 0.488</td></tr><tr><td>Churn</td><td>[0.553,0.570] 0.391</td><td>[0.533,0.550] 0.354</td><td>[0.523,0.541] 0.329</td><td>[0.498,0.514] 0.318</td><td>[0.487,0.507] 0.319</td><td>[0.476,0.499] 0.321</td></tr><tr><td>COIL2000</td><td>[0.386,0.397] 0.238</td><td>[0.346,0.361] 0.228</td><td>[0.325,0.333] 0.232</td><td>[0.314,0.323] 0.230</td><td>[0.315,0.322] 0.231</td><td>[0.317,0.324] 0.231</td></tr><tr><td>Credit card default</td><td>[0.232,0.245] 0.457</td><td>[0.225,0.230] 0.453</td><td>[0.229,0.236] 0.451</td><td>[0.228,0.231] 0.450</td><td>[0.230,0.232] 0.450</td><td>[0.230,0.233] 0.450</td></tr><tr><td></td><td>[0.454,0.461] 0.620</td><td>[0.449,0.457] 0.609</td><td>[0.448,0.455] 0.584</td><td>[0.447,0.454] 0.573</td><td>[0.447,0.454] 0.548</td><td>[0.447,0.453] 0.542</td></tr><tr><td>Credit-g</td><td>[0.606,0.635]</td><td>[0.598,0.620]</td><td>[0.576,0.591]</td><td>[0.564,0.581]</td><td>[0.537,0.560]</td><td>[0.528,0.558]</td></tr><tr><td>Dataset</td><td> $\mu = 0 . 0 5$ </td><td> $\mu = 0 . 1$ </td><td> $\mu = 0 . 2$ </td><td> $\mu = 0 . 4$ </td><td> $\mu = 0 . 8$ </td><td> $\mu = 1 . 6$ </td></tr><tr><td>Diabetes</td><td>0.660 [0.644,0.680]</td><td>0.615 [0.598,0.626]</td><td>0.574 [0.558,0.589]</td><td>0.505 [0.488,0.523]</td><td>0.494 [0.479,0.510]</td><td>0.492 [0.481,0.504]</td></tr><tr><td>Diabetes130US</td><td>0.305 [0.304,0.306]</td><td>0.304 [0.303,0.304]</td><td>0.303 [0.303,0.303]</td><td>0.303 [0.303,0.303]</td><td>0.303 [0.303,0.303]</td><td>0.303 [0.303,0.303]</td></tr><tr><td>E-commerce shipping</td><td>0.589 [0.585,0.594]</td><td>0.576 [0.570,0.581]</td><td>0.576 [0.572,0.581]</td><td>0.573 [0.570,0.577]</td><td>0.569 [0.566,0.573]</td><td>0.568 [0.565,0.571]</td></tr><tr><td>Fitness club</td><td>0.596 [0.586,0.606]</td><td>0.580 [0.575,0.586]</td><td>0.531 [0.520,0.542]</td><td>0.502 [0.495,0.509]</td><td>0.498 [0.492,0.504]</td><td>0.491 [0.484,0.497]</td></tr><tr><td>Good customer</td><td>0.363 [0.357,0.371]</td><td>0.357 [0.354,0.360]</td><td>0.357 [0.354,0.359]</td><td>0.351 [0.348,0.354]</td><td>0.345 [0.339,0.349]</td><td>0.337 [0.331,0.342]</td></tr><tr><td>Hazelnut contaminant</td><td>0.553 [0.530,0.581]</td><td>0.481 [0.469,0.492]</td><td>0.500 [0.484,0.517]</td><td>0.455 [0.442,0.466]</td><td>0.417 [0.403,0.432]</td><td>0.418 [0.402,0.436]</td></tr><tr><td>HELOC</td><td>0.584 [0.579,0.591]</td><td>0.576 [0.570,0.580]</td><td>0.563 [0.558,0.567]</td><td>0.563 [0.559,0.568]</td><td>0.562 [0.558,0.565]</td><td>0.561 [0.556,0.564]</td></tr><tr><td>HR analytics job change</td><td>0.488 [0.484,0.491]</td><td>0.486 [0.485,0.488]</td><td>0.484 [0.482,0.485]</td><td>0.481 [0.479,0.483]</td><td>0.481 [0.480,0.483]</td><td>0.481 [0.480,0.483]</td></tr><tr><td>In-vehicle coupon</td><td>0.656 [0.653,0.659]</td><td>0.646 [0.642,0.649]</td><td>0.641 [0.639,0.643]</td><td>0.639 [0.637,0.641]</td><td>0.638 [0.636,0.641]</td><td>0.638 [0.635,0.640]</td></tr><tr><td>JM1</td><td>0.462 [0.458,0.467]</td><td>0.456 [0.453,0.460]</td><td>0.461 [0.457,0.465]</td><td>0.458 [0.455,0.460]</td><td>0.455 [0.452,0.458]</td><td>0.456 [0.453,0.459]</td></tr><tr><td>Marketing campaign</td><td>0.395 [0.387,0.404]</td><td>0.377 [0.362,0.390]</td><td>0.347 [0.340,0.353]</td><td>0.334 [0.326,0.343]</td><td>0.325 [0.318,0.333]</td><td>0.321 [0.312,0.330]</td></tr><tr><td>Maternal health risk</td><td>1.083 [1.059,1.115]</td><td>1.021 [1.000,1.041]</td><td>0.877 [0.858,0.894]</td><td>0.810 [0.782,0.835]</td><td>0.765 [0.736,0.794]</td><td>0.740 [0.717,0.763]</td></tr><tr><td>MIC</td><td>0.874 [0.831,0.921]</td><td>0.779 [0.762,0.798]</td><td>0.720 [0.707,0.734]</td><td>0.691 [0.682,0.700]</td><td>0.659 [0.650,0.669]</td><td>0.641 [0.629,0.653]</td></tr><tr><td>NATICUSdroid</td><td>0.368 [0.353,0.381]</td><td>0.280 [0.270,0.290]</td><td>0.252 [0.244,0.260]</td><td>0.242 [0.232,0.251]</td><td>0.239 [0.229,0.250]</td><td>0.240 [0.231,0.250]</td></tr><tr><td>Online shoppers</td><td>0.344 [0.338,0.351]</td><td>0.313 [0.307,0.318]</td><td>0.298 [0.294,0.301]</td><td>0.294 [0.290,0.297]</td><td>0.292 [0.288,0.295]</td><td>0.292 [0.289,0.295]</td></tr><tr><td>Polish bankruptcy</td><td>0.243 [0.239,0.246]</td><td>0.242 [0.236,0.248]</td><td>0.242 [0.235,0.249]</td><td>0.246 [0.242,0.251]</td><td>0.248 [0.244,0.253]</td><td>0.250 [0.245,0.255]</td></tr><tr><td>QSAR biodeg</td><td>0.645 [0.639,0.653]</td><td>0.581 [0.570,0.595]</td><td>0.483 [0.465,0.503]</td><td>0.418 [0.399,0.440]</td><td>0.384 [0.360,0.411]</td><td>0.356 [0.338,0.378]</td></tr><tr><td>SDSS17</td><td>0.472 [0.460,0.482]</td><td>0.465 [0.454,0.472]</td><td>0.456 [0.444,0.463]</td><td>0.454 [0.442,0.461]</td><td>0.454 [0.442,0.461]</td><td>0.454 [0.443,0.461]</td></tr><tr><td>Seismic bumps</td><td>0.254 [0.242,0.271]</td><td>0.238 [0.235,0.242]</td><td>0.238 [0.233,0.244]</td><td>0.232 [0.229,0.235]</td><td>0.237 [0.234,0.239]</td><td>0.237 [0.235,0.239]</td></tr><tr><td>Splice</td><td>1.024 [1.018,1.030]</td><td>0.993 [0.986,1.001]</td><td>0.832 [0.798,0.873]</td><td>0.675 [0.665,0.685]</td><td>0.575 [0.564,0.587]</td><td>0.517 [0.509,0.527]</td></tr><tr><td>Student dropout</td><td>0.878 [0.847,0.905]</td><td>0.763 [0.748,0.778]</td><td>0.704 [0.691,0.719]</td><td>0.660 [0.648,0.673]</td><td>0.652 [0.642,0.663]</td><td>0.658 [0.653,0.665]</td></tr><tr><td>Taiwanese bankruptcy</td><td>0.143 [0.137,0.149]</td><td>0.140 [0.137,0.143]</td><td>0.137 [0.133,0.140]</td><td>0.133 [0.130,0.136]</td><td>0.128 [0.126,0.130]</td><td>0.132 [0.129,0.135]</td></tr><tr><td>Website phishing</td><td>0.892 [0.863,0.923]</td><td>0.770 [0.741,0.801]</td><td>0.569 [0.549,0.590]</td><td>0.520 [0.498,0.542]</td><td>0.484 [0.469,0.498]</td><td>0.463 [0.450,0.478]</td></tr><tr><td>Wine quality</td><td>1.290 [1.282,1.298]</td><td>1.211 [1.205,1.217]</td><td>1.160 [1.151,1.167]</td><td>1.143 [1.137,1.148]</td><td>1.118 [1.113,1.123]</td><td>1.118 [1.112,1.123]</td></tr></table>

Table S18: Per-dataset AUC for DP-LR; entries report the mean and bootstrap 95% CI over context–target splits.
<table><tr><td>Dataset</td><td> $\mu = 0 . 0 5$ </td><td> $\mu = 0 . 1$ </td><td> $\mu = 0 . 2$ </td><td> $\mu = 0 . 4$ </td><td> $\mu = 0 . 8$ </td><td> $\mu = 1 . 6$ </td></tr><tr><td>Amazon employee access</td><td>0.523 [0.508,0.538]</td><td>0.526 [0.509,0.542]</td><td>0.524 [0.502,0.545]</td><td>0.539 [0.524,0.552]</td><td>0.548 [0.531,0.562]</td><td>0.553 [0.532,0.567]</td></tr><tr><td>Anneal</td><td>0.573 [0.526,0.618]</td><td>0.681 [0.637,0.729]</td><td>0.790 [0.751,0.831]</td><td>0.863 [0.819,0.904]</td><td>0.899 [0.862,0.936]</td><td>0.943 [0.919,0.963]</td></tr><tr><td>Bank customer churn</td><td>0.709 [0.681,0.735]</td><td>0.726 [0.702,0.746]</td><td>0.727 [0.701,0.746]</td><td>0.730 [0.711,0.747]</td><td>0.728 [0.710,0.745]</td><td>0.739 [0.730,0.748]</td></tr><tr><td>Bank marketing</td><td>0.659 [0.625,0.687]</td><td>0.681 [0.664,0.696]</td><td>0.684 [0.663,0.701]</td><td>0.689 [0.674,0.703]</td><td>0.701 [0.693,0.708]</td><td>0.706 [0.697,0.714]</td></tr><tr><td>Blood transfusion</td><td>0.667 [0.631,0.698]</td><td>0.703 [0.673,0.727]</td><td>0.724 [0.700,0.749]</td><td>0.728 [0.708,0.750]</td><td>0.731 [0.709,0.754]</td><td>0.735 [0.712,0.758]</td></tr><tr><td>Churn</td><td>0.698 [0.661,0.734]</td><td>0.749 [0.724,0.772]</td><td>0.760 [0.730,0.787]</td><td>0.767 [0.744,0.788]</td><td>0.776 [0.756,0.796]</td><td>0.787 [0.772,0.803]</td></tr><tr><td>COIL2000</td><td>0.540 [0.516,0.566]</td><td>0.556 [0.533,0.580]</td><td>0.583 [0.559,0.606]</td><td>0.614 [0.592,0.634]</td><td>0.637 [0.618,0.656]</td><td>0.656 [0.641,0.670]</td></tr><tr><td>Credit card default</td><td>0.709 [0.706,0.712]</td><td>0.713 [0.710,0.718]</td><td>0.716 [0.712,0.720]</td><td>0.716 [0.712,0.721]</td><td>0.716 [0.711,0.721]</td><td>0.716 [0.711,0.721]</td></tr><tr><td>Credit-g</td><td>0.589 [0.557,0.623]</td><td>0.641 [0.604,0.681]</td><td>0.675 [0.637,0.712]</td><td>0.688 [0.651,0.723]</td><td>0.705 [0.676,0.731]</td><td>0.710 [0.689,0.733]</td></tr><tr><td>Diabetes</td><td>0.697 [0.612,0.773] 0.512</td><td>0.743 [0.668,0.804]</td><td>0.774 [0.721,0.819]</td><td>0.811 [0.783,0.838]</td><td>0.831 [0.814,0.848]</td><td>0.836 [0.821,0.849]</td></tr><tr><td>Diabetes130US</td><td>[0.497,0.526]</td><td>0.523 [0.513,0.532]</td><td>0.530 [0.514,0.546]</td><td>0.540 [0.525,0.553]</td><td>0.540 [0.524,0.555]</td><td>0.562 [0.552,0.570]</td></tr><tr><td>E-commerce shipping</td><td>0.720 [0.714,0.725]</td><td>0.723 [0.717,0.729]</td><td>0.724 [0.717,0.730]</td><td>0.725 [0.720,0.730]</td><td>0.725 [0.718,0.731]</td><td>0.724 [0.719,0.730]</td></tr></table>

Continued on next page

Table S18: Per-dataset AUC for DP-LR; entries report the mean and bootstrap 95% CI over context–target splits. (continued)
<table><tr><td>Dataset</td><td> $\mu = 0 . 0 5$ </td><td> $\mu = 0 . 1$ </td><td> $\mu = 0 . 2$ </td><td> $\mu = 0 . 4$ </td><td> $\mu = 0 . 8$ </td><td> $\mu = 1 . 6$ </td></tr><tr><td>Fitness club</td><td>0.725 [0.673,0.767]</td><td>0.764 [0.725,0.793]</td><td>0.799 [0.789,0.809]</td><td>0.806 [0.799,0.813]</td><td>0.804 [0.794,0.813]</td><td>0.811 [0.806,0.816]</td></tr><tr><td>Good customer</td><td>0.544 [0.502,0.589]</td><td>0.555 [0.509,0.602]</td><td>0.587 [0.546,0.629]</td><td>0.614 [0.576,0.652]</td><td>0.643 [0.611,0.675]</td><td>0.663 [0.636,0.693]</td></tr><tr><td>Hazelnut contaminant</td><td>0.812 [0.781,0.842]</td><td>0.870 [0.851,0.887]</td><td>0.905 [0.890,0.918]</td><td>0.929 [0.923,0.936]</td><td>0.941 [0.937,0.945]</td><td>0.947 [0.944,0.951]</td></tr><tr><td>HELOC</td><td>0.753 [0.735,0.767]</td><td>0.772 [0.766,0.778]</td><td>0.775 [0.769,0.781]</td><td>0.777 [0.773,0.781]</td><td>0.778 [0.774,0.782]</td><td>0.779 [0.775,0.783]</td></tr><tr><td>HR analytics job change</td><td>0.744 [0.731,0.755]</td><td>0.754 [0.745,0.762]</td><td>0.756 [0.745,0.764]</td><td>0.758 [0.751,0.765]</td><td>0.761 [0.756,0.766]</td><td>0.761 [0.756,0.766]</td></tr><tr><td>In-vehicle coupon</td><td>0.644 [0.639,0.649]</td><td>0.652 [0.647,0.657]</td><td>0.656 [0.651,0.661]</td><td>0.657 [0.652,0.663]</td><td>0.658 [0.653,0.664]</td><td>0.658 [0.653,0.664]</td></tr><tr><td>JM1</td><td>0.656 [0.622,0.688]</td><td>0.663 [0.633,0.691]</td><td>0.670 [0.647,0.693]</td><td>0.672 [0.654,0.691]</td><td>0.677 [0.664,0.691]</td><td>0.687 [0.677,0.698]</td></tr><tr><td>Marketing campaign</td><td>0.695 [0.609,0.761]</td><td>0.737 [0.654,0.801]</td><td>0.782 [0.719,0.834]</td><td>0.830 [0.798,0.858]</td><td>0.855 [0.831,0.874]</td><td>0.872 [0.863,0.882]</td></tr><tr><td>Maternal health risk</td><td>0.673 [0.635,0.704]</td><td>0.722 [0.684,0.756]</td><td>0.767 [0.742,0.791]</td><td>0.777 [0.762,0.791]</td><td>0.786 [0.774,0.800]</td><td>0.789 [0.773,0.805]</td></tr><tr><td>MIC</td><td>0.524 [0.494,0.556]</td><td>0.569 [0.537,0.603]</td><td>0.627 [0.601,0.653]</td><td>0.663 [0.636,0.687]</td><td>0.704 [0.673,0.731]</td><td>0.732 [0.698,0.761]</td></tr><tr><td>NATICUSdroid</td><td>0.914 [0.889,0.936]</td><td>0.959 [0.955,0.962]</td><td>0.970 [0.967,0.973]</td><td>0.976 [0.973,0.978]</td><td>0.978 [0.976,0.980]</td><td>0.980 [0.978,0.981]</td></tr><tr><td>Online shoppers</td><td>0.833 [0.819,0.847]</td><td>0.842 [0.826,0.857]</td><td>0.854 [0.841,0.865]</td><td>0.864 [0.854,0.872]</td><td>0.871 [0.864,0.878]</td><td>0.877 [0.872,0.883]</td></tr><tr><td>Polish bankruptcy</td><td>0.671 [0.596,0.738]</td><td>0.703 [0.639,0.760]</td><td>0.728 [0.672,0.778]</td><td>0.751 [0.711,0.788]</td><td>0.776 [0.749,0.800]</td><td>0.798 [0.782,0.812]</td></tr><tr><td>QSAR biodeg</td><td>0.737 [0.676,0.794]</td><td>0.795 [0.745,0.839]</td><td>0.847 [0.817,0.876]</td><td>0.887 [0.870,0.903]</td><td>0.908 [0.894,0.921]</td><td>0.918 [0.905,0.930]</td></tr><tr><td>SDSS17</td><td>0.975 [0.973,0.976]</td><td>0.980 [0.979,0.981]</td><td>0.983 [0.982,0.983]</td><td>0.984 [0.983,0.985]</td><td>0.985 [0.985,0.986]</td><td>0.986 [0.985,0.986]</td></tr><tr><td>Seismic bumps</td><td>0.586 [0.515,0.646]</td><td>0.557 [0.492,0.613]</td><td>0.582 [0.522,0.641]</td><td>0.607 [0.564,0.647]</td><td>0.634 [0.584,0.687]</td><td>0.619 [0.537,0.698]</td></tr><tr><td>Splice</td><td>0.656 [0.627,0.684]</td><td>0.757 [0.726,0.784]</td><td>0.860 [0.848,0.870]</td><td>0.902 [0.898,0.907]</td><td>0.921 [0.918,0.925]</td><td>0.930 [0.928,0.934]</td></tr><tr><td>Student dropout</td><td>0.752 [0.725,0.776]</td><td>0.818 [0.794,0.835]</td><td>0.850 [0.843,0.856]</td><td>0.864 [0.858,0.868]</td><td>0.873 [0.867,0.877]</td><td>0.878 [0.872,0.882]</td></tr><tr><td>Taiwanese bankruptcy</td><td>0.628 [0.554,0.706]</td><td>0.666 [0.589,0.743]</td><td>0.721 [0.656,0.787]</td><td>0.792 [0.746,0.839]</td><td>0.850 [0.821,0.881]</td><td>0.884 [0.862,0.907]</td></tr><tr><td>Website phishing</td><td>0.727 [0.701,0.756]</td><td>0.776 [0.754,0.799]</td><td>0.815 [0.793,0.837]</td><td>0.838 [0.821,0.854]</td><td>0.848 [0.834,0.863]</td><td>0.848 [0.833,0.862]</td></tr><tr><td>Wine quality</td><td>0.600 [0.561,0.637]</td><td>0.614 [0.576,0.653]</td><td>0.634 [0.600,0.668]</td><td>0.656 [0.623,0.689]</td><td>0.676 [0.642,0.707]</td><td>0.698 [0.665,0.730]</td></tr></table>

Table S19: Per-dataset log loss for DP-LR; entries report the mean and bootstrap 95% CI over context–target splits.
<table><tr><td>Dataset</td><td> $\mu = 0 . 0 5$ </td><td> $\mu = 0 . 1$ </td><td> $\mu = 0 . 2$ </td><td> $\mu = 0 . 4$ </td><td> $\mu = 0 . 8$ </td><td> $\mu = 1 . 6$ </td></tr><tr><td>Amazon employee access</td><td>0.510 [0.407,0.649]</td><td>0.462 [0.404,0.534]</td><td>0.419 [0.388,0.459]</td><td>0.396 [0.382,0.413]</td><td>0.386 [0.377,0.396]</td><td>0.378 [0.374,0.383]</td></tr><tr><td>Anneal</td><td>17.709 [7.089,30.296]</td><td>7.697 [3.262,13.514]</td><td>4.202 [1.611,7.595]</td><td>0.867 [0.564,1.284]</td><td>0.576 [0.434,0.803]</td><td>0.425 [0.325,0.568]</td></tr><tr><td>Bank customer churn</td><td>1.089 [0.900,1.395]</td><td>0.914 [0.873,0.961]</td><td>0.921 [0.871,0.988]</td><td>0.921 [0.878,0.972]</td><td>0.950 [0.892,1.011]</td><td>0.931 [0.901,0.967]</td></tr><tr><td>Bank marketing</td><td>0.721 [0.662,0.796]</td><td>0.686 [0.654,0.723]</td><td>0.673 [0.653,0.695]</td><td>0.673 [0.662,0.684]</td><td>0.658 [0.650,0.667]</td><td>0.645 [0.639,0.651]</td></tr><tr><td>Blood transfusion</td><td>1.859 [0.925,2.898]</td><td>1.438 [0.898,2.109]</td><td>1.188 [0.910,1.530]</td><td>0.989 [0.889,1.110]</td><td>0.894 [0.822,0.980]</td><td>0.884 [0.820,0.959]</td></tr><tr><td>Churn</td><td>1.496 [0.715,2.575]</td><td>1.053 [0.683,1.709]</td><td>0.908 [0.695,1.266]</td><td>0.741 [0.703,0.779]</td><td>0.725 [0.672,0.793]</td><td>0.694 [0.668,0.722]</td></tr><tr><td>COIL2000</td><td>1.565 [0.611,3.008]</td><td>0.700 [0.540,0.972]</td><td>0.589 [0.511,0.713]</td><td>0.502 [0.483,0.523]</td><td>0.475 [0.461,0.489]</td><td>0.464 [0.449,0.483]</td></tr><tr><td>Credit card default</td><td>0.986 [0.958,1.019]</td><td>0.955 [0.938,0.975]</td><td>0.959 [0.929,0.996]</td><td>0.951 [0.925,0.989]</td><td>0.961 [0.936,0.986]</td><td>0.945 [0.933,0.958]</td></tr><tr><td>Credit-g</td><td>6.380 [2.325,11.534]</td><td>5.008 [1.906,8.743]</td><td>3.255 [1.405,5.645]</td><td>3.165 [1.293,5.416]</td><td>1.567 [1.205,2.093]</td><td>1.374 [1.177,1.630]</td></tr><tr><td>Diabetes</td><td>3.674 [1.101,6.602]</td><td>2.639 [1.071,4.516]</td><td>1.996 [1.021,3.141]</td><td>1.111 [0.833,1.452]</td><td>0.893 [0.771,1.029]</td><td>0.817 [0.752,0.886]</td></tr><tr><td>Diabetes130US</td><td>0.666 [0.618,0.730]</td><td>0.640 [0.611,0.673]</td><td>0.661 [0.607,0.750]</td><td>0.623 [0.601,0.649]</td><td>0.634 [0.606,0.670]</td><td>0.606 [0.596,0.616]</td></tr><tr><td>E-commerce shipping</td><td>1.177 [1.097,1.268]</td><td>1.142 [1.095,1.195]</td><td>1.112 [1.061,1.161]</td><td>1.099 [1.049,1.149]</td><td>1.077 [1.039,1.117]</td><td>1.099 [1.063,1.138]</td></tr><tr><td>Fitness club</td><td>2.262 [1.110,3.668]</td><td>1.672 [0.948,2.681]</td><td>1.002 [0.814,1.276]</td><td>0.950 [0.807,1.160]</td><td>0.950 [0.829,1.107]</td><td>0.895 [0.834,0.978]</td></tr><tr><td>Good customer</td><td>4.229 [1.506,7.231]</td><td>2.298 [0.866,3.999]</td><td>1.097 [0.679,1.900]</td><td>1.067 [0.717,1.520]</td><td>0.676 [0.650,0.701]</td><td>0.670 [0.639,0.704]</td></tr><tr><td>Hazelnut contaminant</td><td>3.756 [1.526,6.168]</td><td>2.017 [0.798,3.632]</td><td>1.127 [0.605,2.047]</td><td>0.592 [0.528,0.654]</td><td>0.556 [0.497,0.611]</td><td>0.532 [0.483,0.578]</td></tr><tr><td>Dataset</td><td>µ = 0.05</td><td>µ = 0.1</td><td>µ = 0.2</td><td>µ = 0.4</td><td>µ = 0.8</td><td>µ = 1.6</td></tr><tr><td>HELOC</td><td>2.161 [1.212,3.704] 0.930</td><td>1.193 [1.130,1.274] 0.902</td><td>1.164 [1.122,1.214]</td><td>1.160 [1.130,1.192]</td><td>1.155 [1.127,1.185]</td><td>1.151 [1.127,1.179]</td></tr><tr><td>HR analytics job change</td><td>[0.888,0.982]</td><td>[0.874,0.928]</td><td>0.917 [0.878,0.967]</td><td>0.925 [0.898,0.954]</td><td>0.913 [0.893,0.934]</td><td>0.910 [0.898,0.924]</td></tr><tr><td>In-vehicle coupon</td><td>1.499 [1.379,1.688]</td><td>1.350 [1.310,1.384]</td><td>1.312 [1.278,1.339]</td><td>1.308 [1.270,1.349]</td><td>1.295 [1.262,1.330]</td><td>1.286 [1.258,1.314]</td></tr><tr><td>JM1</td><td>1.110 [0.865,1.453]</td><td>1.068 [0.868,1.332]</td><td>1.008 [0.866,1.198]</td><td>0.932 [0.861,1.020]</td><td>0.880 [0.851,0.918]</td><td>0.854 [0.841,0.869]</td></tr><tr><td>Marketing campaign</td><td>3.935 [1.311,7.631]</td><td>2.810 [0.991,5.394]</td><td>1.627 [0.677,3.146]</td><td>0.808 [0.612,1.107]</td><td>0.671 [0.543,0.878]</td><td>0.594 [0.539,0.647]</td></tr><tr><td>Maternal health risk</td><td>4.833 [2.167,8.081]</td><td>3.477 [1.491,5.910]</td><td>2.310 [1.229,3.784]</td><td>1.464 [1.270,1.702]</td><td>1.405 [1.270,1.539]</td><td>1.415 [1.272,1.552]</td></tr><tr><td>MIC</td><td>36.899 [13.084,68.713] 5.097</td><td>42.861 [11.482,77.234]</td><td>12.734 [4.895,27.683]</td><td>2.704 [2.324,3.090]</td><td>1.675 [1.527,1.851]</td><td>1.417 [1.301,1.546]</td></tr><tr><td>NATICUSdroid</td><td>[0.728,9.945]</td><td>0.550 [0.496,0.610]</td><td>0.428 [0.394,0.465]</td><td>0.378 [0.353,0.412]</td><td>0.368 [0.343,0.402]</td><td>0.367 [0.340,0.402]</td></tr><tr><td>Online shoppers</td><td>0.647 [0.589,0.726] 1.023</td><td>0.753 [0.624,0.944]</td><td>0.694 [0.619,0.804]</td><td>0.648 [0.597,0.728]</td><td>0.614 [0.591,0.638]</td><td>0.596 [0.580,0.611]</td></tr><tr><td>Polish bankruptcy</td><td>[0.491,1.833] 9.240</td><td>0.816 [0.451,1.402] 6.307</td><td>0.688 [0.421,1.100]</td><td>0.626 [0.399,0.999] 1.322</td><td>0.571 [0.379,0.898]</td><td>0.521 [0.374,0.783]</td></tr><tr><td>QSAR biodeg</td><td>[3.080,16.539] 0.367</td><td>[2.224,11.584] 0.313</td><td>3.443 [1.283,6.817] 0.274</td><td>[0.852,1.849] 0.251</td><td>0.865 [0.639,1.136] 0.238</td><td>0.783 [0.611,0.976] 0.229</td></tr><tr><td>SDSS17</td><td>[0.341,0.396] 1.539</td><td>[0.297,0.331] 1.180</td><td>[0.259,0.291] 0.923</td><td>[0.242,0.261] 0.632</td><td>[0.228,0.248] 0.498</td><td>[0.221,0.237] 0.493</td></tr><tr><td>Seismic bumps</td><td>[0.580,2.664] 16.570</td><td>[0.623,1.856] 8.726</td><td>[0.504,1.447] 1.758</td><td>[0.451,0.903] 1.131</td><td>[0.434,0.578] 1.008</td><td>[0.435,0.557] 0.957</td></tr><tr><td>Splice</td><td>[5.343,29.825] 9.494</td><td>[2.621,17.057] 3.119</td><td>[1.289,2.600] 1.221</td><td>[1.051,1.202] 1.110</td><td>[0.937,1.073] 1.032</td><td>[0.897,1.013] 0.996</td></tr><tr><td>Student dropout</td><td>[3.804,15.852] 1.978</td><td>[1.304,6.668] 0.459</td><td>[1.158,1.288] 0.339</td><td>[1.059,1.162] 0.257</td><td>[0.992,1.077] 0.231</td><td>[0.960,1.039] 0.213</td></tr><tr><td>Taiwanese bankruptcy</td><td>[0.487,4.053] 5.298</td><td>[0.319,0.644] 3.222</td><td>[0.271,0.449] 1.925</td><td>[0.237,0.276] 0.964</td><td>[0.217,0.245] 0.960</td><td>[0.202,0.223] 1.005</td></tr><tr><td>Website phishing</td><td>[2.145,9.043] 2.025</td><td>[1.271,5.536] 1.878</td><td>[0.858,3.555] 1.880</td><td>[0.876,1.066] 1.854</td><td>[0.897,1.041] 1.833</td><td>[0.900,1.149] 1.830</td></tr><tr><td>Wine quality</td><td>[1.737,2.396]</td><td>[1.763,2.071]</td><td>[1.808,1.997]</td><td>[1.807,1.918]</td><td>[1.799,1.875]</td><td>[1.793,1.866]</td></tr></table>

Table S20: Per-dataset AUC for DP-MLP; entries report the mean and bootstrap 95% CI over context–target splits.
<table><tr><td>Dataset</td><td>µ = 0.05</td><td>µ = 0.1</td><td>µ = 0.2</td><td>µ = 0.4</td><td>µ = 0.8</td><td>µ = 1.6</td></tr><tr><td>Amazon employee access</td><td>0.528 [0.511,0.543]</td><td>0.545 [0.532,0.557]</td><td>0.567 [0.557,0.577]</td><td>0.579 [0.571,0.587]</td><td>0.587 [0.576,0.597]</td><td>0.599 [0.587,0.610]</td></tr><tr><td>Anneal</td><td>0.489 [0.431,0.542]</td><td>0.528 [0.467,0.580]</td><td>0.610 [0.539,0.675]</td><td>0.764 [0.714,0.805]</td><td>0.870 [0.836,0.895]</td><td>0.928 [0.909,0.944]</td></tr><tr><td>Bank customer churn</td><td>0.633 [0.595,0.671]</td><td>0.707 [0.685,0.728]</td><td>0.765 [0.752,0.779]</td><td>0.792 [0.777,0.809]</td><td>0.826 [0.813,0.838]</td><td>0.847 [0.839,0.854]</td></tr><tr><td>Bank marketing</td><td>0.698 [0.688,0.707]</td><td>0.719 [0.713,0.724]</td><td>0.729 [0.724,0.734]</td><td>0.733 [0.728,0.738]</td><td>0.740 [0.735,0.744]</td><td>0.745 [0.741,0.748]</td></tr><tr><td>Blood transfusion</td><td>0.563 [0.487,0.630]</td><td>0.594 [0.518,0.659]</td><td>0.621 [0.550,0.688]</td><td>0.661 [0.606,0.712]</td><td>0.723 [0.696,0.752]</td><td>0.741 [0.722,0.761]</td></tr><tr><td>Churn</td><td>0.523 [0.467,0.566]</td><td>0.594 [0.537,0.647]</td><td>0.726 [0.680,0.767]</td><td>0.814 [0.793,0.836]</td><td>0.862 [0.846,0.878]</td><td>0.887 [0.877,0.896]</td></tr><tr><td>COIL2000</td><td>0.524 [0.501,0.548]</td><td>0.551 [0.527,0.574]</td><td>0.587 [0.563,0.610]</td><td>0.626 [0.604,0.647]</td><td>0.646 [0.628,0.664]</td><td>0.679 [0.667,0.690]</td></tr><tr><td>Credit card default</td><td>0.689 [0.680,0.698]</td><td>0.712 [0.702,0.722]</td><td>0.734 [0.724,0.743]</td><td>0.748 [0.743,0.753]</td><td>0.758 [0.753,0.762]</td><td>0.766 [0.762,0.770]</td></tr><tr><td>Credit-g</td><td>0.518 [0.474,0.564]</td><td>0.535 [0.492,0.579]</td><td>0.565 [0.524,0.610]</td><td>0.600 [0.555,0.646]</td><td>0.681 [0.659,0.705]</td><td>0.712 [0.693,0.735]</td></tr><tr><td>Diabetes</td><td>0.593 [0.529,0.656]</td><td>0.643 [0.582,0.697]</td><td>0.704 [0.650,0.753]</td><td>0.761 [0.722,0.794]</td><td>0.812 [0.791,0.833]</td><td>0.827 [0.806,0.847]</td></tr><tr><td>Diabetes130US</td><td>0.544 [0.535,0.552]</td><td>0.570 [0.563,0.576]</td><td>0.583 [0.578,0.587]</td><td>0.592 [0.588,0.595]</td><td>0.598 [0.595,0.602]</td><td>0.604 [0.600,0.608]</td></tr><tr><td>E-commerce shipping</td><td>0.682 [0.657,0.702]</td><td>0.710 [0.698,0.721]</td><td>0.725 [0.716,0.732]</td><td>0.727 [0.719,0.735]</td><td>0.729 [0.721,0.736]</td><td>0.730 [0.724,0.735]</td></tr><tr><td>Fitness club</td><td>0.561 [0.499,0.622]</td><td>0.626 [0.562,0.684]</td><td>0.702 [0.660,0.739]</td><td>0.755 [0.730,0.777]</td><td>0.796 [0.788,0.805]</td><td>0.805 [0.798,0.813]</td></tr><tr><td>Good customer</td><td>0.564 [0.529,0.598]</td><td>0.581 [0.547,0.613]</td><td>0.609 [0.582,0.637]</td><td>0.642 [0.619,0.670]</td><td>0.679 [0.657,0.705]</td><td>0.700 [0.679,0.724]</td></tr><tr><td>Hazelnut contaminant</td><td>0.730 [0.680,0.780]</td><td>0.791 [0.753,0.826]</td><td>0.846 [0.819,0.868]</td><td>0.892 [0.876,0.907]</td><td>0.923 [0.911,0.934]</td><td>0.952 [0.949,0.954]</td></tr><tr><td>HELOC</td><td>0.723 [0.705,0.739]</td><td>0.755 [0.745,0.763]</td><td>0.773 [0.766,0.779]</td><td>0.782 [0.776,0.788]</td><td>0.790 [0.786,0.795]</td><td>0.792 [0.788,0.797]</td></tr><tr><td>HR analytics job change</td><td>0.723 [0.706,0.736]</td><td>0.754 [0.745,0.761]</td><td>0.768 [0.762,0.773]</td><td>0.775 [0.770,0.780]</td><td>0.781 [0.776,0.785]</td><td>0.784 [0.779,0.788]</td></tr><tr><td>In-vehicle coupon</td><td>0.574 [0.561,0.588]</td><td>0.611 [0.599,0.624]</td><td>0.647 [0.639,0.654]</td><td>0.665 [0.659,0.672]</td><td>0.680 [0.674,0.686]</td><td>0.700 [0.696,0.705]</td></tr><tr><td>JM1</td><td>0.663 [0.649,0.679]</td><td>0.692 [0.683,0.703]</td><td>0.707 [0.699,0.716]</td><td>0.714 [0.706,0.723]</td><td>0.715 [0.706,0.724]</td><td>0.722 [0.714,0.730]</td></tr><tr><td>Marketing campaign</td><td>0.537 [0.474,0.603]</td><td>0.570 [0.512,0.632]</td><td>0.682 [0.621,0.735]</td><td>0.811 [0.772,0.844]</td><td>0.857 [0.835,0.876]</td><td>0.869 [0.853,0.885]</td></tr><tr><td>Maternal health risk</td><td>0.560 [0.491,0.632]</td><td>0.606 [0.541,0.673]</td><td>0.664 [0.609,0.722]</td><td>0.729 [0.692,0.769]</td><td>0.786 [0.759,0.812]</td><td>0.812 [0.797,0.828]</td></tr><tr><td>MIC</td><td>0.470 [0.414,0.529]</td><td>0.477 [0.425,0.534]</td><td>0.550 [0.494,0.607]</td><td>0.637 [0.573,0.689]</td><td>0.728 [0.677,0.766]</td><td>0.775 [0.728,0.813]</td></tr><tr><td>NATICUSdroid</td><td>0.903 [0.871,0.930]</td><td>0.951 [0.937,0.961]</td><td>0.966 [0.961,0.971]</td><td>0.973 [0.969,0.977]</td><td>0.979 [0.977,0.981]</td><td>0.981 [0.979,0.983]</td></tr><tr><td>Online shoppers</td><td>0.712 [0.636,0.782]</td><td>0.829 [0.800,0.848]</td><td>0.863 [0.858,0.869]</td><td>0.878 [0.872,0.883]</td><td>0.888 [0.883,0.893]</td><td>0.894 [0.890,0.899]</td></tr><tr><td>Polish bankruptcy</td><td>0.553 [0.472,0.627]</td><td>0.646 [0.570,0.711]</td><td>0.731 [0.703,0.763]</td><td>0.758 [0.731,0.784]</td><td>0.786 [0.767,0.803]</td><td>0.799 [0.783,0.814]</td></tr><tr><td>QSAR biodeg</td><td>0.636 [0.562,0.711]</td><td>0.720 [0.659,0.778]</td><td>0.821 [0.785,0.853]</td><td>0.874 [0.853,0.892]</td><td>0.894 [0.877,0.910]</td><td>0.913 [0.900,0.925]</td></tr><tr><td>SDSS17</td><td>0.967 [0.966,0.968]</td><td>0.976 [0.976,0.977]</td><td>0.982 [0.981,0.983]</td><td>0.985 [0.984,0.986]</td><td>0.987 [0.986,0.988]</td><td>0.989 [0.988,0.989]</td></tr><tr><td>Seismic bumps</td><td>0.495 [0.431,0.563]</td><td>0.499 [0.452,0.539]</td><td>0.629 [0.576,0.676]</td><td>0.688 [0.634,0.730]</td><td>0.741 [0.696,0.782]</td><td>0.765 [0.736,0.797]</td></tr><tr><td>Splice</td><td>0.559 [0.528,0.587]</td><td>0.635 [0.598,0.670]</td><td>0.748 [0.704,0.786]</td><td>0.840 [0.809,0.867]</td><td>0.909 [0.903,0.915]</td><td>0.928 [0.924,0.932]</td></tr><tr><td>Student dropout</td><td>0.695 [0.664,0.723]</td><td>0.759 [0.735,0.782]</td><td>0.821 [0.804,0.833]</td><td>0.851 [0.844,0.857]</td><td>0.864 [0.859,0.868]</td><td>0.877 [0.872,0.882]</td></tr><tr><td>Taiwanese bankruptcy</td><td>0.477 [0.416,0.541]</td><td>0.581 [0.527,0.631]</td><td>0.702 [0.668,0.733]</td><td>0.810 [0.789,0.832]</td><td>0.864 [0.845,0.886]</td><td>0.885 [0.865,0.906]</td></tr><tr><td>Website phishing</td><td>0.590 [0.499,0.678] 0.543</td><td>0.641 [0.552,0.724]</td><td>0.708 [0.636,0.773]</td><td>0.810 [0.775,0.838]</td><td>0.866 [0.855,0.877]</td><td>0.897 [0.888,0.907]</td></tr><tr><td>Wine quality</td><td>[0.514,0.571]</td><td>0.571 [0.547,0.595]</td><td>0.607 [0.584,0.628]</td><td>0.638 [0.613,0.663]</td><td>0.666 [0.638,0.695]</td><td>0.692 [0.661,0.723]</td></tr></table>

Table S21: Per-dataset log loss for DP-MLP; entries report the mean and bootstrap 95% CI over context–target splits.
<table><tr><td>Dataset</td><td>µ = 0.05</td><td>µ = 0.1</td><td>µ = 0.2</td><td>µ = 0.4</td><td>µ = 0.8</td><td>µ = 1.6</td></tr><tr><td>Amazon employee access</td><td>0.283 [0.229,0.358]</td><td>0.250 [0.223,0.298]</td><td>0.224 [0.221,0.227]</td><td>0.221 [0.220,0.223]</td><td>0.220 [0.219,0.221]</td><td>0.219 [0.218,0.220]</td></tr><tr><td>Anneal</td><td>2.720 [1.402,4.559]</td><td>1.983 [1.203,3.363]</td><td>1.709 [1.020,2.640]</td><td>0.826 [0.584,1.141]</td><td>0.513 [0.402,0.668]</td><td>0.309 [0.273,0.345]</td></tr><tr><td>Bank customer churn</td><td>0.510 [0.487,0.536]</td><td>0.479 [0.464,0.496]</td><td>0.435 [0.424,0.445]</td><td>0.410 [0.396,0.422]</td><td>0.384 [0.372,0.394]</td><td>0.377 [0.367,0.388]</td></tr><tr><td>Bank marketing</td><td>0.339 [0.336,0.343]</td><td>0.330 [0.328,0.334]</td><td>0.325 [0.322,0.329]</td><td>0.322 [0.319,0.326]</td><td>0.317 [0.315,0.319]</td><td>0.316 [0.315,0.318]</td></tr><tr><td>Blood transfusion</td><td>0.764 [0.653,0.909]</td><td>0.708 [0.607,0.831]</td><td>0.610 [0.565,0.651]</td><td>0.582 [0.540,0.619]</td><td>0.532 [0.504,0.562]</td><td>0.502 [0.483,0.521]</td></tr><tr><td>Churn</td><td>0.653 [0.452,1.015]</td><td>0.547 [0.407,0.790]</td><td>0.371 [0.354,0.392]</td><td>0.324 [0.310,0.337]</td><td>0.288 [0.270,0.305]</td><td>0.271 [0.256,0.288]</td></tr><tr><td>COIL2000</td><td>0.472 [0.279,0.813]</td><td>0.398 [0.246,0.677]</td><td>0.250 [0.239,0.266]</td><td>0.242 [0.234,0.254]</td><td>0.232 [0.229,0.236]</td><td>0.230 [0.227,0.233]</td></tr><tr><td>Credit card default</td><td>0.488 [0.480,0.498]</td><td>0.473 [0.466,0.480]</td><td>0.466 [0.458,0.473]</td><td>0.457 [0.453,0.461]</td><td>0.455 [0.451,0.459]</td><td>0.457 [0.453,0.461]</td></tr><tr><td>Credit-g</td><td>1.132 [0.666,2.043]</td><td>1.028 [0.652,1.763]</td><td>0.909 [0.629,1.449]</td><td>0.618 [0.600,0.633]</td><td>0.582 [0.567,0.598]</td><td>0.563 [0.545,0.581]</td></tr><tr><td>Diabetes</td><td>0.999 [0.686,1.448]</td><td>0.904 [0.658,1.259]</td><td>0.720 [0.610,0.899]</td><td>0.670 [0.577,0.806]</td><td>0.606 [0.522,0.728]</td><td>0.552 [0.477,0.661]</td></tr><tr><td>Diabetes130US</td><td>0.310 [0.303,0.319]</td><td>0.302 [0.301,0.304]</td><td>0.303 [0.302,0.304]</td><td>0.302 [0.301,0.303]</td><td>0.301 [0.300,0.302]</td><td>0.301 [0.300,0.302]</td></tr><tr><td>E-commerce shipping</td><td>0.638 [0.617,0.657]</td><td>0.607 [0.584,0.630]</td><td>0.562 [0.553,0.576]</td><td>0.545 [0.539,0.550]</td><td>0.536 [0.530,0.542]</td><td>0.531 [0.527,0.535]</td></tr><tr><td>Fitness club</td><td>0.724 [0.637,0.865]</td><td>0.679 [0.603,0.789]</td><td>0.630 [0.568,0.712]</td><td>0.587 [0.535,0.659]</td><td>0.498 [0.491,0.506]</td><td>0.486 [0.481,0.492]</td></tr><tr><td>Good customer</td><td>0.724 [0.521,0.973]</td><td>0.663 [0.485,0.871]</td><td>0.558 [0.408,0.744]</td><td>0.438 [0.353,0.580]</td><td>0.344 [0.335,0.352]</td><td>0.334 [0.324,0.345]</td></tr><tr><td>Hazelnut contaminant</td><td>0.845 [0.603,1.290]</td><td>0.709 [0.559,0.963]</td><td>0.596 [0.495,0.752]</td><td>0.427 [0.394,0.466]</td><td>0.354 [0.335,0.374]</td><td>0.290 [0.281,0.299]</td></tr><tr><td>HELOC</td><td>0.646 [0.629,0.661]</td><td>0.617 [0.600,0.636]</td><td>0.583 [0.569,0.598]</td><td>0.570 [0.560,0.580]</td><td>0.558 [0.550,0.567]</td><td>0.555 [0.550,0.560]</td></tr><tr><td>HR analytics job change</td><td>0.523 [0.512,0.539]</td><td>0.496 [0.488,0.508]</td><td>0.481 [0.475,0.489]</td><td>0.473 [0.468,0.479]</td><td>0.466 [0.462,0.469]</td><td>0.462 [0.458,0.466]</td></tr><tr><td>In-vehicle coupon</td><td>0.683 [0.674,0.695]</td><td>0.673 [0.664,0.683]</td><td>0.658 [0.650,0.669]</td><td>0.643 [0.639,0.647]</td><td>0.635 [0.631,0.638]</td><td>0.632 [0.628,0.636]</td></tr><tr><td>JM1</td><td>0.544 [0.481,0.645]</td><td>0.462 [0.451,0.475]</td><td>0.448 [0.444,0.453]</td><td>0.444 [0.440,0.448]</td><td>0.443 [0.438,0.447]</td><td>0.440 [0.435,0.444]</td></tr><tr><td>Marketing campaign</td><td>0.840 [0.555,1.363]</td><td>0.659 [0.468,1.000]</td><td>0.539 [0.412,0.754]</td><td>0.353 [0.335,0.377]</td><td>0.324 [0.303,0.351]</td><td>0.317 [0.290,0.346]</td></tr><tr><td></td><td>1.312</td><td>1.220</td><td>1.111</td><td>0.960</td><td>0.838</td><td>0.749</td></tr><tr><td>Maternal health risk</td><td>[1.040,1.814]</td><td>[1.002,1.591]</td><td>[0.950,1.354]</td><td>[0.875,1.039]</td><td>[0.769,0.917]</td><td>[0.713,0.786]</td></tr><tr><td>Dataset</td><td> $\mu = 0 . 0 5$ </td><td> $\mu = 0 . 1$ </td><td> $\mu = 0 . 2$ </td><td> $\mu = 0 . 4$ </td><td> $\mu = 0 . 8$ </td><td> $\mu = 1 . 6$ </td></tr><tr><td>MIC</td><td>2.862 [1.492,5.299] 0.458</td><td>2.471 [1.248,4.670] 0.336</td><td>2.002 [0.980,3.891] 0.246</td><td>1.601 [0.755,3.183]</td><td>0.689 [0.606,0.794]</td><td>0.648 [0.560,0.746]</td></tr><tr><td>NATICUSdroid</td><td>[0.405,0.514] 0.487</td><td>[0.288,0.399]</td><td>[0.219,0.275]</td><td>0.222 [0.191,0.254]</td><td>0.177 [0.167,0.191]</td><td>0.178 [0.168,0.192]</td></tr><tr><td>Online shoppers</td><td>[0.383,0.648]</td><td>0.346 [0.331,0.369]</td><td>0.319 [0.311,0.328]</td><td>0.302 [0.294,0.310]</td><td>0.285 [0.279,0.290]</td><td>0.274 [0.269,0.278]</td></tr><tr><td>Polish bankruptcy</td><td>0.481 [0.371,0.598]</td><td>0.354 [0.259,0.480]</td><td>0.312 [0.234,0.429]</td><td>0.291 [0.226,0.387]</td><td>0.222 [0.215,0.230]</td><td>0.220 [0.212,0.227]</td></tr><tr><td>QSAR biodeg</td><td>1.007 [0.626,1.731] 0.298</td><td>0.826 [0.572,1.279]</td><td>0.663 [0.517,0.900]</td><td>0.548 [0.442,0.708]</td><td>0.495 [0.398,0.645]</td><td>0.451 [0.362,0.599]</td></tr><tr><td>SDSS17</td><td>[0.283,0.322]</td><td>0.241 [0.231,0.253]</td><td>0.202 [0.191,0.214]</td><td>0.183 [0.169,0.198]</td><td>0.163 [0.154,0.173]</td><td>0.148 [0.144,0.153]</td></tr><tr><td>Seismic bumps</td><td>0.671 [0.512,0.835]</td><td>0.488 [0.347,0.659]</td><td>0.361 [0.246,0.517]</td><td>0.317 [0.232,0.435]</td><td>0.224 [0.215,0.235]</td><td>0.216 [0.207,0.226]</td></tr><tr><td>Splice</td><td>1.935 [1.032,3.646]</td><td>1.014 [0.973,1.068]</td><td>0.918 [0.879,0.958]</td><td>0.794 [0.729,0.863]</td><td>0.598 [0.549,0.656]</td><td>0.546 [0.497,0.602]</td></tr><tr><td>Student dropout</td><td>0.970 [0.925,1.014]</td><td>0.890 [0.833,0.947]</td><td>0.755 [0.721,0.803]</td><td>0.676 [0.661,0.692]</td><td>0.654 [0.634,0.673]</td><td>0.609 [0.590,0.630]</td></tr><tr><td>Taiwanese bankruptcy</td><td>0.494 [0.249,0.820]</td><td>0.389 [0.187,0.658]</td><td>0.219 [0.150,0.347]</td><td>0.184 [0.130,0.281]</td><td>0.155 [0.115,0.219]</td><td>0.119 [0.104,0.135]</td></tr><tr><td>Website phishing</td><td>1.094 [0.960,1.269]</td><td>1.009 [0.857,1.177]</td><td>0.935 [0.788,1.078]</td><td>0.716 [0.592,0.852]</td><td>0.496 [0.471,0.525]</td><td>0.457 [0.429,0.488]</td></tr><tr><td>Wine quality</td><td>1.468 [1.326,1.622]</td><td>1.221 [1.191,1.256]</td><td>1.157 [1.146,1.167]</td><td>1.124 [1.114,1.133]</td><td>1.102 [1.091,1.112]</td><td>1.088 [1.077,1.100]</td></tr></table>

Table S22: Per-dataset AUC for Non-private LR; entries report the mean and bootstrap 95% CI over context–target splits.
<table><tr><td>Dataset</td><td>Non-DP</td></tr><tr><td>Amazon employee access</td><td>0.582 [0.574,0.590]</td></tr><tr><td></td><td>0.985</td></tr><tr><td>Anneal</td><td>[0.979,0.990] 0.767</td></tr><tr><td>Bank customer churn</td><td>[0.758,0.776] 0.732</td></tr><tr><td>Bank marketing</td><td>[0.729,0.736] 0.728</td></tr><tr><td>Blood transfusion</td><td>[0.710,0.745] 0.780</td></tr><tr><td>Churn</td><td>[0.771,0.788] 0.738</td></tr><tr><td>COIL2000</td><td>[0.730,0.748] 0.742</td></tr><tr><td>Credit card default</td><td>[0.738,0.746] 0.720</td></tr><tr><td>Credit-g</td><td>[0.698,0.743] 0.820</td></tr><tr><td>Diabetes</td><td>[0.801,0.839] 0.607</td></tr><tr><td>Diabetes130US</td><td>[0.604,0.610] 0.708</td></tr><tr><td>E-commerce shipping</td><td>[0.702,0.715] 0.816</td></tr><tr><td>Fitness club</td><td>[0.812,0.820] 0.714</td></tr><tr><td>Good customer</td><td>[0.696,0.735] 0.952</td></tr><tr><td>Hazelnut contaminant</td><td>[0.950,0.955] 0.758</td></tr><tr><td>HELOC</td><td>[0.754,0.762] 0.766</td></tr><tr><td>HR analytics job change</td><td>[0.762,0.770] 0.660</td></tr><tr><td>In-vehicle coupon</td><td>[0.655,0.665] 0.736</td></tr><tr><td>JM1</td><td>[0.728,0.745] 0.889</td></tr><tr><td>Marketing campaign</td><td>[0.880,0.898] 0.793</td></tr><tr><td>Maternal health risk</td><td>[0.777,0.808]</td></tr><tr><td>MIC</td><td>0.756 [0.741,0.771]</td></tr><tr><td>NATICUSdroid</td><td>0.982 [0.980,0.983]</td></tr><tr><td>Online shoppers</td><td>0.896 [0.893,0.899]</td></tr><tr><td colspan="2">Continued on next page</td></tr><tr><td>Polish bankruptcy</td><td>0.881 [0.865,0.896]</td></tr><tr><td>QSAR biodeg</td><td>0.919 [0.909,0.929] 0.987</td></tr><tr><td>SDSS17</td><td>[0.987,0.988]</td></tr><tr><td>Seismic bumps</td><td>0.764 [0.739,0.791] 0.942</td></tr><tr><td>Splice</td><td>[0.939,0.945]</td></tr><tr><td>Student dropout</td><td>0.884 [0.878,0.889]</td></tr><tr><td>Taiwanese bankruptcy</td><td>0.923 [0.910,0.935] 0.866</td></tr><tr><td>Website phishing</td><td>[0.856,0.877] 0.775</td></tr><tr><td>Wine quality</td><td>[0.763,0.787]</td></tr></table>

Table S23: Per-dataset log loss for Non-private LR; entries report the mean and bootstrap 95% CI over context– target splits.
<table><tr><td>Dataset</td><td>Non-DP</td></tr><tr><td></td><td>0.219</td></tr><tr><td>Amazon employee access</td><td>[0.218,0.219] 0.170</td></tr><tr><td>Anneal</td><td>[0.150,0.194] 0.426</td></tr><tr><td>Bank customer churn</td><td>[0.421,0.432] 0.316</td></tr><tr><td>Bank marketing</td><td>[0.314,0.317] 0.494</td></tr><tr><td>Blood transfusion</td><td>[0.482,0.508] 0.341</td></tr><tr><td>Churn</td><td>[0.337,0.345] 0.206</td></tr><tr><td>COIL2000</td><td>[0.204,0.208] 0.455</td></tr><tr><td>Credit card default</td><td>[0.452,0.457] 0.551</td></tr><tr><td>Credit-g</td><td>[0.531,0.571] 0.528</td></tr><tr><td>Diabetes</td><td>[0.504,0.553] 0.291</td></tr><tr><td>Diabetes130US</td><td>[0.291,0.292] 0.587</td></tr><tr><td>E-commerce shipping</td><td>[0.583,0.592]</td></tr><tr><td>Fitness club</td><td>0.468 [0.464,0.472]</td></tr><tr><td>Good customer</td><td>0.325 [0.318,0.331]</td></tr><tr><td>Hazelnut contaminant</td><td>0.272 [0.264,0.280]</td></tr><tr><td>HELOC</td><td>0.589 [0.586,0.593]</td></tr><tr><td>HR analytics job change</td><td>0.480 [0.478,0.482]</td></tr><tr><td>In-vehicle coupon</td><td>0.645 [0.642,0.647]</td></tr><tr><td>JM1</td><td>0.432</td></tr><tr><td></td><td>[0.428,0.436] 0.285</td></tr><tr><td>Marketing campaign</td><td>[0.275,0.296] 0.785</td></tr><tr><td>Maternal health risk</td><td>[0.755,0.815] 0.824</td></tr><tr><td>MIC</td><td>[0.771,0.882] 0.163</td></tr><tr><td>NATICUSdroid</td><td>[0.157,0.172] 0.262</td></tr><tr><td>Online shoppers</td><td>[0.259,0.265] 0.171</td></tr><tr><td>Polish bankruptcy</td><td>[0.163,0.178] 0.342</td></tr><tr><td>QSAR biodeg</td><td>[0.319,0.365]</td></tr><tr><td>SDSS17</td><td>0.141 [0.139,0.143]</td></tr><tr><td colspan="2">Continued on next page</td></tr></table>

Table S23: Per-dataset log loss for Non-private LR; entries report the mean and bootstrap 95% CI over context– target splits. (continued)

<table><tr><td>Dataset</td><td>Non-DP</td></tr><tr><td>Seismic bumps</td><td>0.213 [0.206,0.220]</td></tr><tr><td>Splice</td><td>0.442 [0.428,0.454]</td></tr><tr><td>Student dropout</td><td>0.569 [0.557,0.584]</td></tr><tr><td>Taiwanese bankruptcy</td><td>0.094 [0.088,0.101] 0.502</td></tr><tr><td>Website phishing</td><td>[0.479,0.528] 1.070</td></tr><tr><td>Wine quality</td><td>[1.062,1.078]</td></tr></table>

## S6.6 Private Preprocessing Pipeline Details

This subsection supports the private-preprocessing evaluations in Figs. 4, S11 and S12 with the private moment mechanism, details of LLM-bound inference, numerical results, Elo ratings and per-feature clipping bounds.

![](images/2cee6ed7d5dc793845398bde20ee585cde7d4b72dc187f27f4aed0d662dd046a.jpg)  
Private z-score: LLM bounds Partially private z-score: oracle bounds Non-private z-score

Figure S11: Efect of preprocessing on PrivTab. Mean log loss across ten context–target splits on five datasets, comparing non-private standardisation, partially private standardisation with oracle bounds, and private standardisation with LLM-derived bounds. Lower log loss is better. Error bars show bootstrap 95% confidence intervals. The legend beneath the plot identifies the preprocessing variants.

![](images/18da45cc68ef22640301694f75cf83e34f5deeb749fe6931882af3a0432ddb7e.jpg)  
Figure S12: Performance of privately preprocessed models at $\mu = 0 . 4 . \mathbf { a } , \mathbf { b }$ , AUC and log loss for PrivTab, DP-MLP and DP-LR using LLM-derived bounds and private standardisation across five datasets. c, Elo ratings of these models; DP-LR is calibrated to 1,000. Higher AUC and Elo and lower log loss are better. Error bars show bootstrap 95% confidence intervals. The method legend is beneath all three panels.

## S6.6.1 Private moment mechanism

Let $n _ { c }$ be the context size and $[ a _ { k } , b _ { k } ]$ the fixed clipping interval for input coordinate $k = 1 , \ldots , d .$ . Coordinates with $b _ { k } - a _ { k } < 1 0 ^ { - 6 }$ are set to zero. Otherwise, we clip and rescale each context input as

$$
c _ { i k } = \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , ( x _ { i k } - a _ { k } ) / ( b _ { k } - a _ { k } ) \} \} ,\tag{S129}
$$

and apply the same transformation to target inputs. We add independent Gaussian noise to the context sums and sums of squares:

$$
\tilde { m } _ { 1 k } = \sum _ { i = 1 } ^ { n _ { c } } c _ { i k } + \xi _ { 1 , k } , \qquad \tilde { m } _ { 2 k } = \sum _ { i = 1 } ^ { n _ { c } } c _ { i k } ^ { 2 } + \xi _ { 2 , k } , \qquad \xi _ { 1 , k } \sim \mathcal { N } ( 0 , \sigma _ { 1 } ^ { 2 } ) , \quad \xi _ { 2 , k } \sim \mathcal { N } ( 0 , \sigma _ { 2 } ^ { 2 } ) .\tag{S130}
$$

Replacing one context row changes each sum vector by at most $\sqrt { d }$ in $\ell _ { 2 }$ norm. We allocate $\mu _ { 1 } = \mu _ { \mathrm { p r e p } } / 2$ to the sums and $\mu _ { 2 } = \sqrt { \mu _ { \mathrm { p r e p } } ^ { 2 } - \mu _ { 1 } ^ { 2 } }$ to the sums of squares, with $\sigma _ { 1 } = \sqrt { d } / \mu _ { 1 }$ and $\sigma _ { 2 } = \sqrt { d } / \mu _ { 2 }$ . By GDP composition, the two releases satisfy $\mu _ { \mathrm { p r e p } } – \mathrm { G D P }$ because ${ \sqrt { \mu _ { 1 } ^ { 2 } + \mu _ { 2 } ^ { 2 } } } = \mu _ { \mathrm { p r e p } }$

We compute the noisy mean $\tilde { m } _ { k } = \tilde { m } _ { 1 k } / n _ { c }$ and variance $\tilde { v } _ { k } = \tilde { m } _ { 2 k } / n _ { c } - \tilde { m } _ { k } ^ { 2 } . \mathrm { ~ I f ~ } \tilde { v } _ { k } < 1 0 ^ { - 6 }$ , including when it is negative, or the clipping interval is constant, we set that coordinate to zero for both context and target rows. Otherwise, we obtain the standardised input $z _ { i k }$ by clipping $( c _ { i k } - \tilde { m } _ { k } ) / \sqrt { \tilde { v } _ { k } } \mathrm { t o } [ - 1 0 , 1 0 ]$ . The LLM-derived bounds use public descriptions; the oracle-bound workflow is partially private because its empirical dataset minima and maxima are selected without privacy accounting.

## S6.6.2 LLM Bound-Inference Details

For the end-to-end experiments described in the main-text section on private preprocessing, public clipping bounds for numeric features are inferred with Gemini 2.5 Flash. The Gemini call uses temperature 0, top-k = 1, and a maximum output budget of 8192 tokens. The response is constrained by an explicit JSON schema, and the code issues a plain generate\_content call without enabling any search or retrieval tool. Accordingly, the model is prompted to rely only on the public dataset context, the public column descriptions, the provided units, and general background knowledge, rather than on any online lookup of the dataset.

The prompt is structured so that the model explicitly separates documented public bounds from suggested clipping bounds. For each numeric feature, the response must contain documented lower and upper bounds, a documented-bound status, an evidence field, suggested lower and upper clipping bounds, a suggested-clip status, a guess-rationale field, and a single confidence label. The documented bounds are allowed to be null when the column description does not justify an exact or semantically implied numeric limit. For example, the prompt may allow the model to infer that a quantity is non-negative, or that an age variable refers to adults, while still leaving one side of the documented range unspecified. The evidence field records why a documented bound is justified from the prompt alone. The confidence field is not attached only to the documented part; it is a single feature-level label for the overall returned assessment.

The suggested clipping bounds are the values actually used by preprocessing, and they must always be finite and form a valid interval. When exact public bounds are available, the suggested clipping bounds can simply inherit them. Otherwise, the suggested clipping interval is chosen conservatively from public semantics, units, and general knowledge about the feature meaning. This is why the response distinguishes “what is justified directly by the public description” from “what is a conservative suggested clipping choice for DP preprocessing.” The distinction matters because many features admit only partial public evidence, so one side of the documented range may remain null even though a finite suggested clipping interval is still required operationally. In the implementation, the single confidence label is then conservatively post-processed: if the documentedbound status is semantic\_public\_bound or insufficient\_information, a returned high confidence is downgraded to medium; likewise, if the suggested-clip status is chosen\_clipping\_bound, a returned high confidence is also downgraded to medium. In addition, when the documented-bound status is insufficient\_information, the validator forces both documented bounds to null.

Only columns that genuinely require semantic reasoning are sent to Gemini. Categorical features and columns with direct public bounds, such as publicly documented ordinal codes, bypass the LLM entirely and are handled deterministically in code. For those columns, the implementation sets both documented and suggested bounds directly from the public definition, marks both statuses as exact\_public\_bound, and assigns high confidence. The LLM is therefore queried only for non-categorical columns that do not already have direct public bounds. Prompts are cached before inference, and the resulting bounds are also cached and reused unless the cache is explicitly refreshed. The implementation additionally validates that Gemini returns exactly the requested feature names and, if a batch request fails, falls back to retrying one column at a time.

## Example Prompt.

SYSTEM:   
You are extracting PUBLIC clipping bounds for differentially private preprocessing.   
Return strict JSON only.   
Do not include markdown.   
Do not include explanations outside JSON.   
USER:   
Rules:   
1. Analyze each target column independently.   
2. Do NOT claim exact bounds unless they are explicit or follow from definitions.   
3. Do NOT use empirical min/max values unless explicitly provided in the public description.   
4. If true documented bounds are unavailable, you MUST still provide finite suggested clipping   
,→ bounds.   
5. The suggested clipping bounds must be conservative and based on public semantics, units, and   
,→ dataset context.   
6. Clearly distinguish documented bounds from guessed clipping bounds.   
7. If a public semantic lower bound is clear, include it even when the upper bound is not   
,→ documented.   
8. All suggested clipping bounds must be finite numeric values.   
9. Do not omit a target column. If uncertain, use   
,→ documented\_bound\_status="insufficient\_information"   
and suggested\_clip\_status="chosen\_clipping\_bound".   
10. The top-level JSON value must be an array, not an object wrapper.   
11. Use the dataset-specific guidance in the Dataset context when it defines units,   
special values, or semantic lower/upper bounds.   
Return strict JSON only as an array:   
[   
{   
"feature": string,   
"unit": string | null,   
"documented\_lower\_bound": number | null,

"documented\_upper\_bound": number | null,   
"documented\_bound\_status": "exact\_public\_bound" | "documented\_special\_value"   
| "semantic\_public\_bound" | "insufficient\_information",   
"suggested\_lower\_clip": number,   
"suggested\_upper\_clip": number,   
"suggested\_clip\_status": "exact\_public\_bound" | "semantic\_public\_bound"   
| "chosen\_clipping\_bound",   
"evidence": string,   
"guess\_rationale": string,   
"confidence": "low" | "medium" | "high"   
}   
]   
Dataset context:   
Pima Indians Diabetes dataset from OpenML/UCI. The task is binary prediction of diabetes onset   
from diagnostic measurements. Numeric guidance: pregnancies, glucose, blood pressure, skin   
thickness, insulin, BMI, diabetes pedigree function, and age are non-negative; age is adult   
age in years. Use only public medical/column semantics for clipping bounds.

## Example Response.

```json
{
"dataset": "Pima Indians Diabetes",
"provider": "gemini",
"model": "gemini-2.5-flash",
"results": {
"preg": {
"documented_lower_bound": 0,
"documented_upper_bound": null,
"documented_bound_status": "semantic_public_bound",
"suggested_lower_clip": 0,
"suggested_upper_clip": 17,
"suggested_clip_status": "chosen_clipping_bound",
"evidence": "Dataset context states that pregnancies are non-negative.",
"guess_rationale": "No explicit upper bound is given, so 17 is chosen as a conservative
medically plausible clipping value.",
"confidence": "medium"
},
"plas": {
"documented_lower_bound": 0,
"documented_upper_bound": null,
"documented_bound_status": "semantic_public_bound",
"suggested_lower_clip": 40,
"suggested_upper_clip": 400,
"suggested_clip_status": "chosen_clipping_bound",
"evidence": "The prompt states that glucose is non-negative.",
"guess_rationale": "The final interval is chosen from general medical knowledge about
physiologically plausible glucose values.",
"confidence": "medium"
},
"age": {
"documented_lower_bound": 18,
"documented_upper_bound": null,
"documented_bound_status": "semantic_public_bound",
"suggested_lower_clip": 18,
"suggested_upper_clip": 100,
"suggested_clip_status": "chosen_clipping_bound",
"evidence": "The prompt states that age is adult age in years.",
"guess_rationale": "The lower bound is inherited from the semantic evidence; the upper
bound is a conservative clipping choice based on general knowledge
about adult human ages.",
"confidence": "medium"
}
```

Reading the Bound Tables. The supplement tables of public and LLM-derived bounds use the following column meanings. Feature is the preprocessing feature name. Type distinguishes columns that require LLM-based semantic inference from those that do not: numeric means that the feature was sent to Gemini for bound inference, whereas categorical means that the feature was handled deterministically from public category codes or another direct public definition. Exact min and Exact max are the empirical minimum and maximum values observed in the dataset and are reported only for comparison; they are not treated as public bounds by the end-to-end pipeline. Source records where the operational clipping interval comes from: llm means that the final suggested clipping bounds came from the Gemini response, while categorical means that the LLM was bypassed and the bounds were fixed directly from public information in code. Lower and Upper are the final clipping bounds actually used by preprocessing.

The Confidence column is a feature-level label for the final bound assessment. For rows with Source equal to llm, the values low, medium, and high are taken from the validated Gemini response after the conservative post-processing implemented in end\_to\_end\_cases.py. Concretely, a response does not retain high confidence when the documented part is only semantic rather than exact, when the documented status is insufficient\_information, or when the suggested interval is marked as chosen\_clipping\_ bound rather than directly justified by the public description. The label exact appears only for rows with Source equal to categorical; it indicates that the bounds were not inferred by the LLM and instead came directly from an exact public feature definition handled deterministically in code.

Method-level results, Elo ratings and preprocessing comparisons are reported in Tables S24 to S26. Per-feature clipping bounds are listed in Tables S27 to S31.

Table S24: Numerical values underlying Figure $4 ( \mathrm { c } , \mathrm { d } ) .$
<table><tr><td>Dataset</td><td>Method</td><td>AUC mean</td><td>CI low</td><td>CI high</td><td>Loss mean</td><td>CI low</td><td>CI high</td></tr><tr><td>Bank marketing</td><td>PrivTab</td><td>0.878</td><td>0.875</td><td>0.882</td><td>0.273</td><td>0.271</td><td>0.276</td></tr><tr><td>Bank marketing</td><td>DP-LR</td><td>0.868</td><td>0.861</td><td>0.873</td><td>0.507</td><td>0.490</td><td>0.528</td></tr><tr><td>Bank marketing</td><td>DP-MLP</td><td>0.886</td><td>0.883</td><td>0.889</td><td>0.249</td><td>0.245</td><td>0.252</td></tr><tr><td>COMPAS</td><td>PrivTab</td><td>0.715</td><td>0.708</td><td>0.722</td><td>0.619</td><td>0.615</td><td>0.624</td></tr><tr><td>COMPAS</td><td>DP-LR</td><td>0.707</td><td>0.690</td><td>0.720</td><td>1.282</td><td>1.224</td><td>1.340</td></tr><tr><td>COMPAS</td><td>DP-MLP</td><td>0.691</td><td>0.675</td><td>0.705</td><td>0.642</td><td>0.633</td><td>0.652</td></tr><tr><td>ACS income</td><td>PrivTab</td><td>0.847</td><td>0.846</td><td>0.848</td><td>0.469</td><td>0.468</td><td>0.470</td></tr><tr><td>ACS income</td><td>DP-LR</td><td>0.841</td><td>0.840</td><td>0.842</td><td>0.873</td><td>0.864</td><td>0.883</td></tr><tr><td>ACS income</td><td>DP-MLP</td><td>0.852</td><td>0.851</td><td>0.853</td><td>0.454</td><td>0.450</td><td>0.457</td></tr><tr><td>Pima diabetes</td><td>PrivTab</td><td>0.730</td><td>0.701</td><td>0.759</td><td>0.580</td><td>0.561</td><td>0.598</td></tr><tr><td>Pima diabetes</td><td>DP-LR</td><td>0.726</td><td>0.667</td><td>0.775</td><td>1.655</td><td>0.956</td><td>2.688</td></tr><tr><td>Pima diabetes</td><td>DP-MLP</td><td>0.662</td><td>0.631</td><td>0.692</td><td>0.726</td><td>0.641</td><td>0.861</td></tr><tr><td>Maternal health</td><td>PrivTab</td><td>0.731</td><td>0.704</td><td>0.754</td><td>0.905</td><td>0.872</td><td>0.940</td></tr><tr><td>Maternal health</td><td>DP-LR</td><td>0.715</td><td>0.680</td><td>0.743</td><td>1.364</td><td>1.187</td><td>1.641</td></tr><tr><td>Maternal health</td><td>DP-MLP</td><td>0.629</td><td>0.580</td><td>0.672</td><td>1.053</td><td>1.000</td><td>1.098</td></tr></table>

Table S25: ELO ratings for the private-preprocessing evaluation underlying Fig. 4e. Ratings combine AUC for binary datasets and log loss for multiclass datasets across the five datasets and ten context–target splits at total privacy level $\mu = 0 . 4 .$ . The scale is calibrated so that DP-LR has an ELO rating of 1,000. Intervals are 95% confidence intervals.
<table><tr><td>Method</td><td>ELO rating</td><td>95% CI lower</td><td>95% CI upper</td></tr><tr><td>PrivTab</td><td>1,161.5</td><td>1,122.0</td><td>1,206.2</td></tr><tr><td>DP-MLP</td><td>1,117.5</td><td>1,059.4</td><td>1,181.1</td></tr><tr><td>DP-LR</td><td>1,000.0</td><td>930.9</td><td>1,060.0</td></tr></table>

Table S26: Numerical values underlying Figure 4(a,b). Entries report means with bootstrap 95% CIs over context– target splits; log loss is in nats/sample.
<table><tr><td rowspan=1 colspan=6>Partially private z-score    Private z-scoreDataset          Metric   Non-DP z-score           (oracle bounds)            (LLM bounds)</td></tr><tr><td rowspan=1 colspan=2>Bank marketing  AUC     0.880 [0.878, 0.882]</td><td rowspan=1 colspan=2>0.877 [0.874, 0.880]</td><td rowspan=1 colspan=1>0.878 [</td><td rowspan=1 colspan=1>0.875, 0.882]</td></tr><tr><td rowspan=1 colspan=1>Bank marketing  Log loss  0.272 [</td><td rowspan=1 colspan=1>0.270, 0.274]</td><td rowspan=1 colspan=2>0.273 [0.271, 0.275]</td><td rowspan=1 colspan=1>0.273 [</td><td rowspan=1 colspan=1>0.271, 0.276]</td></tr><tr><td rowspan=1 colspan=1>COMPAS        AUC     0.715 [</td><td rowspan=1 colspan=1>0.708, 0.721]</td><td rowspan=1 colspan=2>0.711 [0.704, 0.718]</td><td rowspan=1 colspan=1>0.715 [</td><td rowspan=1 colspan=1>0.708, 0.722]</td></tr><tr><td rowspan=1 colspan=1>COMPAS        Log loss  0.620 [</td><td rowspan=1 colspan=1>0.616, 0.624]</td><td rowspan=1 colspan=2>0.623 [0.618, 0.628]</td><td rowspan=1 colspan=1>0.619 [0</td><td rowspan=1 colspan=1>.615, 0.624]</td></tr><tr><td rowspan=1 colspan=1>ACS income     AUC     0.847 [</td><td rowspan=1 colspan=1>0.846, 0.848]</td><td rowspan=1 colspan=2>0.847 [0.846, 0.848]</td><td rowspan=1 colspan=1>0.847 [0</td><td rowspan=1 colspan=1>.846, 0.848]</td></tr><tr><td rowspan=1 colspan=1>ACS income     Log loss  0.470 [</td><td rowspan=1 colspan=1>0.469, 0.472]</td><td rowspan=1 colspan=2>0.470 [0.469, 0.472]</td><td rowspan=1 colspan=1>0.469 [</td><td rowspan=1 colspan=1>0.468, 0.470]</td></tr><tr><td rowspan=1 colspan=1>Pima diabetes    AUC     0.801 [</td><td rowspan=1 colspan=1>0.782, 0.820]</td><td rowspan=1 colspan=2>0.724 [0.649, 0.773]</td><td rowspan=1 colspan=1>0.730 [</td><td rowspan=1 colspan=1>0.701, 0.759]</td></tr><tr><td rowspan=1 colspan=1>Pima diabetes    Log loss  0.518 [</td><td rowspan=1 colspan=1>0.499, 0.536]</td><td rowspan=1 colspan=1>0.574 [</td><td rowspan=1 colspan=1>0.551, 0.601]</td><td rowspan=1 colspan=1>0.580 [0</td><td rowspan=1 colspan=1>.561, 0.598]</td></tr><tr><td rowspan=1 colspan=1>Maternal health  AUC     0.807 [</td><td rowspan=1 colspan=1>0.790, 0.822]</td><td rowspan=1 colspan=1>0.789 [</td><td rowspan=1 colspan=1>0.765, 0.811]</td><td rowspan=1 colspan=1>0.731 [</td><td rowspan=1 colspan=1>0.704, 0.754]</td></tr><tr><td rowspan=1 colspan=2>Maternal health  Log loss  0.773 [0.749, 0.797]</td><td rowspan=1 colspan=2>0.818 [0.783, 0.853]</td><td rowspan=1 colspan=2>0.905 [0.872, 0.940]</td></tr></table>

Table S27: Public and LLM-derived bounds for ACS income.
<table><tr><td>Feature</td><td>Type</td><td>Exact min</td><td>Exact max</td><td>Source</td><td>Lower</td><td>Upper</td><td>Confidence</td></tr><tr><td>AGEP</td><td>numeric</td><td>17</td><td>95</td><td>llm</td><td>16</td><td>90</td><td>medium</td></tr><tr><td>SEX</td><td>categorical</td><td>1</td><td>2</td><td>categorical</td><td>1</td><td>2</td><td>exact</td></tr><tr><td>RAC1P</td><td>categorical</td><td>1</td><td>9</td><td>categorical</td><td>1</td><td>9</td><td>exact</td></tr><tr><td>SCHL</td><td>categorical</td><td>1</td><td>24</td><td>categorical</td><td>1</td><td>24</td><td>exact</td></tr><tr><td>MAR</td><td>categorical</td><td>1</td><td>5</td><td>categorical</td><td>1</td><td>5</td><td>exact</td></tr><tr><td>RELP</td><td>categorical</td><td>0</td><td>38</td><td>categorical</td><td>0</td><td>17</td><td>exact</td></tr><tr><td>COW</td><td>categorical</td><td>1</td><td>8</td><td>categorical</td><td>1</td><td>9</td><td>exact</td></tr><tr><td>OCCP</td><td>categorical</td><td>0</td><td>626</td><td>categorical</td><td>0</td><td>626</td><td>exact</td></tr><tr><td>WKHP</td><td>numeric</td><td>1</td><td>99</td><td>llm</td><td>0</td><td>100</td><td>medium</td></tr><tr><td>STATE</td><td>categorical</td><td>0</td><td>50</td><td>categorical</td><td>0</td><td>50</td><td>exact</td></tr></table>

Table S28: Public and LLM-derived bounds for Bank marketing.
<table><tr><td>Feature</td><td>Type</td><td>Exact min</td><td>Exact max</td><td>Source</td><td>Lower</td><td>Upper</td><td>Confidence</td></tr><tr><td>age</td><td>numeric</td><td>18</td><td>95</td><td>llm</td><td>18</td><td>100</td><td>medium</td></tr><tr><td>job</td><td>categorical</td><td>0</td><td>11</td><td>categorical</td><td>0</td><td>11</td><td>exact</td></tr><tr><td>marital</td><td>categorical</td><td>0</td><td>2</td><td>categorical</td><td>0</td><td>2</td><td>exact</td></tr><tr><td>education</td><td>categorical</td><td>0</td><td>3</td><td>categorical</td><td>0</td><td>3</td><td>exact</td></tr><tr><td>default</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>balance</td><td>numeric</td><td>-8019</td><td>102127</td><td>llm</td><td>-100000</td><td>1e+06</td><td>low</td></tr><tr><td>housing</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>loan</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>contact</td><td>categorical</td><td>0</td><td>2</td><td>categorical</td><td>0</td><td>2</td><td>exact</td></tr><tr><td>day</td><td>numeric</td><td>1</td><td>31</td><td>llm</td><td>1</td><td>31</td><td>high</td></tr><tr><td>month</td><td>categorical</td><td>0</td><td>11</td><td>categorical</td><td>0</td><td>11</td><td>exact</td></tr><tr><td>duration</td><td>numeric</td><td>0</td><td>4918</td><td>llm</td><td>0</td><td>3600</td><td>medium</td></tr><tr><td>campaign</td><td>numeric</td><td>1</td><td>63</td><td>llm</td><td>1</td><td>50</td><td>medium</td></tr><tr><td>pdays</td><td>numeric</td><td>-1</td><td>871</td><td>llm</td><td>-1</td><td>7300</td><td>medium</td></tr><tr><td>previous</td><td>numeric</td><td>0</td><td>275</td><td>llm</td><td>0</td><td>50</td><td>medium</td></tr><tr><td>poutcome</td><td>categorical</td><td>0</td><td>3</td><td>categorical</td><td>0</td><td>3</td><td>exact</td></tr></table>

Table S29: Public and LLM-derived bounds for COMPAS.
<table><tr><td>Feature</td><td>Type</td><td>Exact min</td><td>Exact max</td><td>Source</td><td>Lower</td><td>Upper</td><td>Confidence</td></tr><tr><td>sex</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>age</td><td>numeric</td><td>18</td><td>80</td><td>llm</td><td>18</td><td>100</td><td>medium</td></tr><tr><td>juv_fel_count</td><td>numeric</td><td>0</td><td>10</td><td>llm</td><td>0</td><td>10</td><td>medium</td></tr><tr><td>juv_misd_count</td><td>numeric</td><td>0</td><td>13</td><td>llm</td><td>0</td><td>10</td><td>medium</td></tr><tr><td>juv_other_count</td><td>numeric</td><td>0</td><td>7</td><td>llm</td><td>0</td><td>10</td><td>medium</td></tr><tr><td>priors_count</td><td>numeric</td><td>0</td><td>38</td><td>llm</td><td>0</td><td>50</td><td>medium</td></tr><tr><td>age_cat_25-45</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>age_cat_Greaterthan45</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>age_cat_Lessthan25</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>race_African-American</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>race_Caucasian</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>c_charge_degree_F</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr><tr><td>c_charge_degree_M</td><td>categorical</td><td>0</td><td>1</td><td>categorical</td><td>0</td><td>1</td><td>exact</td></tr></table>

Table S30: Public and LLM-derived bounds for Maternal health.
<table><tr><td>Feature</td><td>Type</td><td>Exact min</td><td>Exact max</td><td>Source</td><td>Lower</td><td>Upper</td><td>Confidence</td></tr><tr><td>Age</td><td>numeric</td><td>10</td><td>70</td><td>llm</td><td>10</td><td>60</td><td>medium</td></tr><tr><td>SystolicBP</td><td>numeric</td><td>70</td><td>160</td><td>llm</td><td>70</td><td>250</td><td>medium</td></tr><tr><td>DiastolicBP</td><td>numeric</td><td>49</td><td>100</td><td>llm</td><td>40</td><td>150</td><td>medium</td></tr><tr><td>BS</td><td>numeric</td><td>6</td><td>19</td><td>llm</td><td>0</td><td>40</td><td>medium</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Continued on next page</td><td></td><td></td></tr><tr><td>BodyTemp</td><td>numeric</td><td>98</td><td>103</td><td>llm</td><td>80</td><td>110</td><td>medium</td></tr><tr><td>HeartRate</td><td>numeric</td><td>7</td><td>90</td><td>llm</td><td>0</td><td>250</td><td>medium</td></tr></table>

Table S31: Public and LLM-derived bounds for Pima diabetes.
<table><tr><td>Feature</td><td>Type</td><td>Exact min</td><td>Exact max</td><td>Source</td><td>Lower</td><td>Upper</td><td>Confidence</td></tr><tr><td>preg</td><td>numeric</td><td>0</td><td>17</td><td>llm</td><td>0</td><td>17</td><td>medium</td></tr><tr><td>plas</td><td>numeric</td><td>0</td><td>199</td><td>llm</td><td>40</td><td>400</td><td>medium</td></tr><tr><td>pres</td><td>numeric</td><td>0</td><td>122</td><td>llm</td><td>30</td><td>130</td><td>medium</td></tr><tr><td>skin</td><td>numeric</td><td>0</td><td>99</td><td>llm</td><td>5</td><td>80</td><td>medium</td></tr><tr><td>insu</td><td>numeric</td><td>0</td><td>846</td><td>llm</td><td>10</td><td>500</td><td>medium</td></tr><tr><td>mass</td><td>numeric</td><td>0</td><td>67.1</td><td>llm</td><td>10</td><td>70</td><td>medium</td></tr><tr><td>pedi</td><td>numeric</td><td>0.078</td><td>2.42</td><td>llm</td><td>0</td><td>3</td><td>medium</td></tr><tr><td>age</td><td>numeric</td><td>21</td><td>81</td><td>llm</td><td>18</td><td>100</td><td>medium</td></tr></table>

## S6.7 Ablations

Results for noise-aware pretraining, the privacy curriculum, layer and head counts, the attention function, and normalisation and model combination are reported in Tables S32 to S37, respectively.

## S6.7.1 Ablation: Noise-Aware Pretraining

We compare PrivTab trained with the same privacy-noise mechanism used at inference against a variant trained without noise-aware pretraining. The noise-aware model ranks substantially higher at every privacy level, which indicates that the efect of privatized summaries should be incorporated during pretraining rather than treated as a perturbation added only at deployment time.

Table S32: ELO rankings for the noise-aware-pretraining ablation.
<table><tr><td>μ</td><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>overall</td><td>PrivTab with noise-aware pretraining</td><td>1440.2</td><td>1430.9</td><td>1449.5</td></tr><tr><td>overall</td><td>PrivTab without noise-aware pretraining</td><td>1188.1</td><td>1176.2</td><td>1200.0</td></tr><tr><td>0.05</td><td>PrivTab with noise-aware pretraining</td><td>1404.5</td><td>1381.8</td><td>1427.4</td></tr><tr><td>0.05</td><td>PrivTab without noise-aware pretraining</td><td>1108.9</td><td>1079.7</td><td>1135.6</td></tr><tr><td>0.1</td><td>PrivTab with noise-aware pretraining</td><td>1459.1</td><td>1435.9</td><td>1482.6</td></tr><tr><td>0.1</td><td>PrivTab without noise-aware pretraining</td><td>1084.6</td><td>1055.2</td><td>1111.6</td></tr><tr><td>0.2</td><td>PrivTab with noise-aware pretraining</td><td>1508.7</td><td>1487.3</td><td>1530.3</td></tr><tr><td>0.2</td><td>PrivTab without noise-aware pretraining</td><td>1139.5</td><td>1111.3</td><td>1164.8</td></tr><tr><td>0.4</td><td>PrivTab with noise-aware pretraining</td><td>1492.1</td><td>1467.7</td><td>1516.1</td></tr><tr><td>0.4</td><td>PrivTab without noise-aware pretraining</td><td>1236.1</td><td>1205.8</td><td>1264.0</td></tr><tr><td>0.8</td><td>PrivTab with noise-aware pretraining</td><td>1450.8</td><td>1426.0</td><td>1475.0</td></tr><tr><td>0.8</td><td>PrivTab without noise-aware pretraining</td><td>1279.1</td><td>1247.1</td><td>1308.3</td></tr><tr><td>1.6</td><td>PrivTab with noise-aware pretraining</td><td>1432.1</td><td>1405.9</td><td>1457.5</td></tr><tr><td>1.6</td><td>PrivTab without noise-aware pretraining</td><td>1328.4</td><td>1298.9</td><td>1356.4</td></tr></table>

## S6.7.2 Ablation: µ-Range Curriculum

We compare diferent privacy-range curricula used during PrivTab pretraining. The benchmark configuration with $\mu \in [ 0 . 1 6 , 6 4 ]$ is strongest overall; nearby ranges remain competitive, but shifting the curriculum too far toward either weaker or stronger privacy leads to a measurable degradation in the aggregate ranking.

Table S33: ELO rankings for the µ-range pretraining ablation.
<table><tr><td rowspan=1 colspan=6>μ      Method                                                              ELO CI min CI max</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈ [</td><td rowspan=1 colspan=4>0.16, 64])                                              1533.1  1523.6  1542.6</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.2, 80])</td><td rowspan=1 colspan=1>1501.6</td><td rowspan=1 colspan=2>1493.1  1510.2</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.64, 256])</td><td rowspan=1 colspan=1>1489.9</td><td rowspan=1 colspan=2>1480.7   1498.9</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.1, 40])</td><td rowspan=1 colspan=1>1486.0</td><td rowspan=1 colspan=2>1478.0  1494.0</td></tr><tr><td rowspan=1 colspan=2>overall  PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.4, 160])</td><td rowspan=1 colspan=1>1480.9</td><td rowspan=1 colspan=1>1472.9</td><td rowspan=1 colspan=1>1489.0</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.05, 20])</td><td rowspan=1 colspan=1>1470.4</td><td rowspan=1 colspan=1>1462.8</td><td rowspan=1 colspan=1>1478.2</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.04, 16])</td><td rowspan=1 colspan=1>1450.9</td><td rowspan=1 colspan=1>1442.8</td><td rowspan=1 colspan=1>1458.9</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 320])</td><td rowspan=1 colspan=1>1434.4</td><td rowspan=1 colspan=1>1426.5</td><td rowspan=1 colspan=1>1442.5</td></tr><tr><td rowspan=1 colspan=2>overall PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 16])</td><td rowspan=1 colspan=1>1399.6</td><td rowspan=1 colspan=1>1390.7</td><td rowspan=1 colspan=1>1408.3</td></tr><tr><td rowspan=1 colspan=1>overall</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[2.56, 1024])</td><td rowspan=1 colspan=1>1358.2</td><td rowspan=1 colspan=1>1348.5</td><td rowspan=1 colspan=1>1367.9</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.16,64])</td><td rowspan=1 colspan=1>1483.8</td><td rowspan=1 colspan=1>1459.1</td><td rowspan=1 colspan=1>1509.5</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.05, 20])</td><td rowspan=1 colspan=1>1461.8</td><td rowspan=1 colspan=1>1442.0</td><td rowspan=1 colspan=1>1483.3</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.1, 40])</td><td rowspan=1 colspan=1>1455.5</td><td rowspan=1 colspan=1>1433.9</td><td rowspan=1 colspan=1>1478.3</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.04, 16])</td><td rowspan=1 colspan=1>1449.2</td><td rowspan=1 colspan=1>1429.0</td><td rowspan=1 colspan=1>1470.2</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.2,80])</td><td rowspan=1 colspan=1>1438.1</td><td rowspan=1 colspan=1>1418.0</td><td rowspan=1 colspan=1>1459.6</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.4, 160])</td><td rowspan=1 colspan=1>1432.3</td><td rowspan=1 colspan=1>1412.7</td><td rowspan=1 colspan=1>1453.3</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.64, 256])</td><td rowspan=1 colspan=1>1391.9</td><td rowspan=1 colspan=1>1371.1</td><td rowspan=1 colspan=1>1412.5</td></tr><tr><td rowspan=1 colspan=2>0.05    PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 320])</td><td rowspan=1 colspan=1>1347.3</td><td rowspan=1 colspan=1>1327.4</td><td rowspan=1 colspan=1>1367.6</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 16])</td><td rowspan=1 colspan=1>1332.7</td><td rowspan=1 colspan=1>1311.7</td><td rowspan=1 colspan=1>1353.6</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>PrivTab (µ ∈</td><td rowspan=1 colspan=1>[2.56, 1024])</td><td rowspan=1 colspan=1>1225.2</td><td rowspan=1 colspan=1>1201.6</td><td rowspan=1 colspan=1>1248.0</td></tr><tr><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.1, 40])</td><td rowspan=1 colspan=1>1521.6</td><td rowspan=1 colspan=1>1500.6</td><td rowspan=1 colspan=1>1543.7</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.16, 64]</td><td rowspan=1 colspan=1>1519.8</td><td rowspan=1 colspan=1>1497.9</td><td rowspan=1 colspan=1>1543.5</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.2, 80])</td><td rowspan=1 colspan=1>1497.0</td><td rowspan=1 colspan=1>1476.5</td><td rowspan=1 colspan=1>1518.8</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.05, 20])</td><td rowspan=1 colspan=1>1495.8</td><td rowspan=1 colspan=1>1476.4</td><td rowspan=1 colspan=1>1515.2</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.04, 16])</td><td rowspan=1 colspan=1>1493.8</td><td rowspan=1 colspan=1>1473.4</td><td rowspan=1 colspan=1>1515.2</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.4, 160])</td><td rowspan=1 colspan=1>1482.6</td><td rowspan=1 colspan=1>1463.4</td><td rowspan=1 colspan=1>1502.6</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.64, 256])</td><td rowspan=1 colspan=1>1453.2</td><td rowspan=1 colspan=1>1433.1</td><td rowspan=1 colspan=1>1474.5</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.8, 320]]</td><td rowspan=1 colspan=1>1414.2</td><td rowspan=1 colspan=1>1394.8</td><td rowspan=1 colspan=1>1433.8</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 16])</td><td rowspan=1 colspan=1>1392.1</td><td rowspan=1 colspan=1>1370.5</td><td rowspan=1 colspan=1>1414.0</td></tr><tr><td rowspan=1 colspan=2>0.1     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[2.56, 1024])</td><td rowspan=1 colspan=1>1297.5</td><td rowspan=1 colspan=1>1272.2</td><td rowspan=1 colspan=1>1321.2</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.16, 64])</td><td rowspan=1 colspan=1>1559.4</td><td rowspan=1 colspan=1>1537.8</td><td rowspan=1 colspan=1>1581.9</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.2, 80])</td><td rowspan=1 colspan=1>1547.9</td><td rowspan=1 colspan=1>1526.6</td><td rowspan=1 colspan=1>1570.7</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.1, 40]]</td><td rowspan=1 colspan=1>1545.5</td><td rowspan=1 colspan=1>1526.7</td><td rowspan=1 colspan=1>1565.9</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.05, 20])</td><td rowspan=1 colspan=1>1524.5</td><td rowspan=1 colspan=1>1505.1</td><td rowspan=1 colspan=1>1545.6</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.4, 160])</td><td rowspan=1 colspan=1>1509.6</td><td rowspan=1 colspan=1>1489.8</td><td rowspan=1 colspan=1>1529.7</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.04, 16])</td><td rowspan=1 colspan=1>1504.2</td><td rowspan=1 colspan=1>1485.0</td><td rowspan=1 colspan=1>1524.0</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.64, 256])</td><td rowspan=1 colspan=1>1490.2</td><td rowspan=1 colspan=1>1468.2</td><td rowspan=1 colspan=1>1512.0</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.8, 320])</td><td rowspan=1 colspan=1>1452.7</td><td rowspan=1 colspan=1>1432.1</td><td rowspan=1 colspan=1>1474.0</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 16])</td><td rowspan=1 colspan=1>1433.6</td><td rowspan=1 colspan=1>1412.3</td><td rowspan=1 colspan=1>1454.2</td></tr><tr><td rowspan=1 colspan=2>0.2     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[2.56, 1024])</td><td rowspan=1 colspan=1>1336.7</td><td rowspan=1 colspan=1>1311.5</td><td rowspan=1 colspan=1>1360.4</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.16, 64])</td><td rowspan=1 colspan=1>1595.2</td><td rowspan=1 colspan=1>1572.5</td><td rowspan=1 colspan=1>1619.2</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.2,80])</td><td rowspan=1 colspan=1>1564.8</td><td rowspan=1 colspan=1>1544.4</td><td rowspan=1 colspan=1>1586.1</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.64, 256])</td><td rowspan=1 colspan=1>1557.6</td><td rowspan=1 colspan=1>1535.7</td><td rowspan=1 colspan=1>1580.4</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.1, 40])</td><td rowspan=1 colspan=1>1536.4</td><td rowspan=1 colspan=1>1518.5</td><td rowspan=1 colspan=1>1554.8</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.4, 160])</td><td rowspan=1 colspan=1>1533.0</td><td rowspan=1 colspan=1>1512.7</td><td rowspan=1 colspan=1>1554.1</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.05, 20]]</td><td rowspan=1 colspan=1>1516.0</td><td rowspan=1 colspan=1>1497.8</td><td rowspan=1 colspan=1>1534.9</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.04, 16])</td><td rowspan=1 colspan=1>1493.3</td><td rowspan=1 colspan=1>1473.7</td><td rowspan=1 colspan=1>1512.8</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.8, 320])</td><td rowspan=1 colspan=1>1489.5</td><td rowspan=1 colspan=1>1469.7</td><td rowspan=1 colspan=1>1509.8</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 16])</td><td rowspan=1 colspan=1>1459.8</td><td rowspan=1 colspan=1>1438.0</td><td rowspan=1 colspan=1>1481.8</td></tr><tr><td rowspan=1 colspan=2>0.4     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[2.56, 1024])</td><td rowspan=1 colspan=1>1404.0</td><td rowspan=1 colspan=1>1379.8</td><td rowspan=1 colspan=1>1428.2</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.16, 64])</td><td rowspan=1 colspan=1>1578.5</td><td rowspan=1 colspan=1>1554.7</td><td rowspan=1 colspan=1>1602.8</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.64, 256])</td><td rowspan=1 colspan=1>1567.1</td><td rowspan=1 colspan=1>1543.8</td><td rowspan=1 colspan=1>1590.9</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.2, 80])</td><td rowspan=1 colspan=1>1540.7</td><td rowspan=1 colspan=1>1520.1</td><td rowspan=1 colspan=1>1561.3</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.4, 160])</td><td rowspan=1 colspan=1>1518.6</td><td rowspan=1 colspan=1>1498.1</td><td rowspan=1 colspan=1>1539.9</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.8, 320])</td><td rowspan=1 colspan=1>1484.6</td><td rowspan=1 colspan=1>1465.1</td><td rowspan=1 colspan=1>1504.3</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.1, 40]</td><td rowspan=1 colspan=1>1483.7</td><td rowspan=1 colspan=1>1465.5</td><td rowspan=1 colspan=1>1502.6</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈[</td><td rowspan=1 colspan=1>0.05, 20]]</td><td rowspan=1 colspan=1>1466.8</td><td rowspan=1 colspan=1>1449.3</td><td rowspan=1 colspan=1>1484.7</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[2.56, 1024])</td><td rowspan=1 colspan=1>1451.3</td><td rowspan=1 colspan=1>1427.1</td><td rowspan=1 colspan=1>1475.9</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.04, 16])</td><td rowspan=1 colspan=1>1444.0</td><td rowspan=1 colspan=1>1425.6</td><td rowspan=1 colspan=1>1462.1</td></tr><tr><td rowspan=1 colspan=2>0.8     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.8, 16])</td><td rowspan=1 colspan=1>1425.0</td><td rowspan=1 colspan=1>1400.8</td><td rowspan=1 colspan=1>1448.4</td></tr><tr><td rowspan=1 colspan=2>1.6     PrivTab (µ ∈</td><td rowspan=1 colspan=1>[0.64, 256])</td><td rowspan=1 colspan=1>1548.7</td><td rowspan=1 colspan=1>1524.9</td><td rowspan=1 colspan=1>1573.9</td></tr><tr><td rowspan=1 colspan=2>1.6     PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.16, 64])</td><td rowspan=1 colspan=1>1533.9</td><td rowspan=1 colspan=1>1510.5</td><td rowspan=1 colspan=1>1558.5</td></tr><tr><td rowspan=1 colspan=2>1.6     PrivTab (µ ∈ [</td><td rowspan=1 colspan=1>0.2, 80])</td><td rowspan=1 colspan=1>1489.2</td><td rowspan=1 colspan=1>1469.3</td><td rowspan=1 colspan=1>1509.9</td></tr><tr><td rowspan=1 colspan=2>1.6     PrivTab (µ ∈ [</td><td rowspan=1 colspan=4>0.8, 320])                                              1478.9  1459.6  1498.5</td></tr></table>

Continued on next page

Table S33: ELO rankings for the µ-range pretraining ablation. (continued)
<table><tr><td>μ</td><td>Method</td><td></td><td>CI min</td><td>CI max</td></tr><tr><td>1.6</td><td>PrivTab (µ ∈ [2.56, 1024])</td><td></td><td>1478.0 1454.4</td><td>1502.6</td></tr><tr><td>1.6</td><td>PrivTab (µ ∈ [0.4, 160])</td><td></td><td>1475.2 1454.5 1441.0</td><td>1496.7</td></tr><tr><td>1.6</td><td>PrivTab (µ ∈ [0.1, 40])</td><td></td><td>1423.0</td><td>1459.7</td></tr><tr><td>1.6</td><td>PrivTab (µ ∈ [0.05, 20])</td><td></td><td>1404.5</td><td>1441.9</td></tr><tr><td>1.6</td><td>PrivTab (µ ∈ [0.8, 16])]</td><td></td><td>1386.1</td><td>1434.8</td></tr><tr><td>1.6</td><td>PrivTab (µ ∈ [0.04, 16])</td><td></td><td>1363.4</td><td>1400.0</td></tr></table>

## S6.7.3 Ablation: Number of Private Layers

We vary the number of main DP-MHCA layers while keeping the rest of the architecture fixed. Three private layers give the strongest overall ranking; reducing the depth to two layers weakens performance substantially, whereas increasing to four layers does not improve over the benchmark configuration.

Table S34: ELO rankings for the ablation over the number of main DP-MHCA layers.
<table><tr><td>μ</td><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>overall</td><td>PrivTab (Lmain = 3)</td><td>1507.2</td><td>1496.3</td><td>1518.6</td></tr><tr><td>overall</td><td>PrivTab (Lmain = 4)</td><td>1404.0</td><td>1394.2</td><td>1414.0</td></tr><tr><td>overall</td><td>PrivTab (Lmain = 2)</td><td>1283.2</td><td>1273.8</td><td>1292.9</td></tr><tr><td>0.05</td><td>PrivTab (Lmain = 3)</td><td>1493.3</td><td>1465.6</td><td>1524.2</td></tr><tr><td>0.05</td><td>PrivTab (Lmain = 4)</td><td>1453.3</td><td>1430.0</td><td>1478.8</td></tr><tr><td>0.05</td><td>PrivTab (Lmain = 2)</td><td>1294.6</td><td>1271.2</td><td>1317.0</td></tr><tr><td>0.1</td><td>PrivTab (Lmain = 3)</td><td>1575.4</td><td>1546.5</td><td>1608.5</td></tr><tr><td>0.1</td><td>PrivTab (Lmain = 4)</td><td>1472.5</td><td>1448.7</td><td>1498.9</td></tr><tr><td>0.1</td><td>PrivTab (Lmain = 2)</td><td>1320.6</td><td>1296.5</td><td>1345.4</td></tr><tr><td>0.2</td><td>PrivTab (Lmain = 3)</td><td>1595.6</td><td>1568.0</td><td>1627.4</td></tr><tr><td>0.2</td><td>PrivTab (Lmain = 4)</td><td>1464.7</td><td>1440.8</td><td>1491.8</td></tr><tr><td>0.2</td><td>PrivTab (Lmain = 2)</td><td>1323.5</td><td>1298.7</td><td>1349.5</td></tr><tr><td>0.4</td><td>PrivTab (Lmain = 3)</td><td>1514.4</td><td>1490.5</td><td>1542.4</td></tr><tr><td>0.4</td><td>PrivTab (Lmain = 4)</td><td>1409.2</td><td>1384.9</td><td>1435.1</td></tr><tr><td>0.4</td><td>PrivTab (Lmain = 2)</td><td>1302.7</td><td>1279.3</td><td>1327.8</td></tr><tr><td>0.8</td><td>PrivTab (Lmain = 3)</td><td>1472.8</td><td>1449.8</td><td>1498.4</td></tr><tr><td>0.8</td><td>PrivTab (Lmain = 4)</td><td>1357.1</td><td>1334.3</td><td>1381.4</td></tr><tr><td>0.8</td><td>PrivTab (Lmain = 2)</td><td>1267.9</td><td>1245.1</td><td>1291.9</td></tr><tr><td>1.6</td><td>PrivTab (Lmain = 3)</td><td>1432.3</td><td>1408.8</td><td>1458.9</td></tr><tr><td>1.6</td><td>PrivTab (Lmain = 4)</td><td>1305.2</td><td>1283.5</td><td>1327.8</td></tr><tr><td>1.6</td><td>PrivTab (Lmain = 2)</td><td>1218.3</td><td>1196.0</td><td>1241.1</td></tr></table>

## S6.7.4 Ablation: Number of DP-MHCA Heads

We compare diferent head counts inside DP-MHCA. In this ablation, a single DP-MHCA head performs better overall than using four DP-MHCA heads, especially in the more private regime, which supports the benchmark choice of a single privacy-critical head with the full internal width.

Table S35: ELO rankings for the ablation over the number of DP-MHCA heads.
<table><tr><td>μ</td><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>overall</td><td>PrivTab (HDP = 1)</td><td>1508.9</td><td>1500.1</td><td>1517.9</td></tr><tr><td>overall</td><td>PrivTab (HDP = 4)</td><td>1468.5</td><td>1459.3</td><td>1478.0</td></tr><tr><td>0.05</td><td>PrivTab (HDP = 1)</td><td>1465.8</td><td>1441.1</td><td>1491.8</td></tr><tr><td>0.05</td><td>PrivTab (HDP = 4)</td><td>1422.8</td><td>1399.6</td><td>1448.4</td></tr><tr><td>0.1</td><td>PrivTab (HDP = 1)</td><td>1464.8</td><td>1442.4</td><td>1488.9</td></tr><tr><td>0.1</td><td>PrivTab (HDP = 4)</td><td>1462.6</td><td>1440.0</td><td>1487.7</td></tr><tr><td>0.2</td><td>PrivTab (HDP = 1)</td><td>1527.9</td><td>1507.3</td><td>1550.8</td></tr><tr><td>0.2</td><td>PrivTab (HDP = 4)</td><td>1506.5</td><td>1483.1</td><td>1532.0</td></tr><tr><td>0.4</td><td>PrivTab (HDP = 1)</td><td>1566.0</td><td>1544.8</td><td>1588.8</td></tr><tr><td>0.4</td><td>PrivTab (HDP = 4)</td><td>1517.9</td><td>1495.8</td><td>1542.1</td></tr><tr><td>0.8</td><td>PrivTab (HDP = 1)</td><td>1587.0</td><td>1568.0</td><td>1609.2</td></tr><tr><td>µ</td><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>0.8</td><td>PrivTab  $( H _ { \mathrm { D P } } = 4 ) $ </td><td>1523.6</td><td>1499.9</td><td>1549.3</td></tr><tr><td>1.6</td><td>PrivTab  $( H _ { \mathrm { D P } } = 1 )$ </td><td>1570.8</td><td>1549.0</td><td>1595.9</td></tr><tr><td>1.6</td><td>PrivTab (HDP = 4)</td><td>1492.6</td><td>1468.3</td><td>1519.6</td></tr></table>

## S6.7.5 Ablation: DP-MHCA Interaction Function

We compare alternative bounded interaction functions inside DP-MHCA:

$$
g _ { \mathrm { t a n h } } ( \mathbf { q } , \mathbf { k } ) = \operatorname { t a n h } ( \langle \mathbf { q } , \mathbf { k } \rangle / \sqrt { d _ { h } } ) ,
$$

$$
\begin{array} { r } { g _ { \mathrm { l i n e a r c l i p p e d } } ( \mathbf { q } , \mathbf { k } ) = \langle \mathrm { c l i p } _ { 1 } ( \mathbf { q } ) , \mathrm { c l i p } _ { 1 } ( \mathbf { k } ) \rangle , } \end{array}
$$

$$
g _ { \mathrm { l i n e a r n o r m a l i s e d } } ( \mathbf { q } , \mathbf { k } ) = \langle \mathbf { q } / \| \mathbf { q } \| _ { 2 } , \mathbf { k } / \| \mathbf { k } \| _ { 2 } \rangle ,
$$

$$
g _ { \mathrm { s h i f t e d ~ s o f t s i g n } } ( \mathbf { q } , \mathbf { k } ) = { \frac { 1 } { 2 } } ( 1 + \mathrm { s o f t s i g n } ( \langle \mathbf { q } , \mathbf { k } \rangle ) ) .
$$

The benchmark tanh choice is strongest overall and becomes increasingly preferable as privacy weakens, while the alternatives rank lower in aggregate.

Table S36: ELO rankings for the ablation over the DP-MHCA interaction function.
<table><tr><td>μ</td><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>overall</td><td>PrivTab (tanh)</td><td>1503.9</td><td>1492.4</td><td>1515.9</td></tr><tr><td>overall</td><td>PrivTab (shifted softsign)</td><td>1409.4</td><td>1399.9</td><td>1419.5</td></tr><tr><td>overall</td><td>PrivTab (linear normalised)</td><td>1392.4</td><td>1384.1</td><td>1400.8</td></tr><tr><td>overall</td><td>PrivTab (linear clipped)</td><td>1370.2</td><td>1361.3</td><td>1379.3</td></tr><tr><td>0.05</td><td>PrivTab (linear clipped)</td><td>1397.6</td><td>1376.8</td><td>1420.3</td></tr><tr><td>0.05</td><td>PrivTab (linear normalised)</td><td>1394.7</td><td>1374.9</td><td>1416.1</td></tr><tr><td>0.05</td><td>PrivTab (tanh)</td><td>1374.7</td><td>1349.5</td><td>1401.3</td></tr><tr><td>0.05</td><td>PrivTab (shifted softsign)</td><td>1313.8</td><td>1290.7</td><td>1338.2</td></tr><tr><td>0.1</td><td>PrivTab (tanh)</td><td>1475.1</td><td>1451.1</td><td>1501.5</td></tr><tr><td>0.1</td><td>PrivTab (linear normalised)</td><td>1438.1</td><td>1417.1</td><td>1460.8</td></tr><tr><td>0.1</td><td>PrivTab (linear clipped)</td><td>1432.6</td><td>1409.2</td><td>1457.1</td></tr><tr><td>0.1</td><td>PrivTab (shifted softsign)</td><td>1369.8</td><td>1345.6</td><td>1394.3</td></tr><tr><td>0.2</td><td>PrivTab (tanh)</td><td>1535.3</td><td>1508.3</td><td>1565.5</td></tr><tr><td>0.2</td><td>PrivTab (linear normalised)</td><td>1509.5</td><td>1488.0</td><td>1533.6</td></tr><tr><td>0.2</td><td>PrivTab (linear clipped)</td><td>1473.7</td><td>1451.7</td><td>1498.4</td></tr><tr><td>0.2</td><td>PrivTab (shifted softsign)</td><td>1461.9</td><td>1436.5</td><td>1489.0</td></tr><tr><td>0.4</td><td>PrivTab (tanh)</td><td>1561.3</td><td>1532.4</td><td>1593.8</td></tr><tr><td>0.4</td><td>PrivTab (shifted softsign)</td><td>1462.4</td><td>1439.1</td><td>1487.5</td></tr><tr><td>0.4</td><td>PrivTab (linear normalised)</td><td>1412.2</td><td>1391.3</td><td>1434.6</td></tr><tr><td>0.4</td><td>PrivTab (linear clipped)</td><td>1377.3</td><td>1354.7</td><td>1401.0</td></tr><tr><td>0.8</td><td>PrivTab (tanh)</td><td>1581.6</td><td>1549.2</td><td>1619.9</td></tr><tr><td>0.8</td><td>PrivTab (shifted softsign)</td><td>1455.6</td><td>1432.0</td><td>1481.8</td></tr><tr><td>0.8</td><td>PrivTab (linear normalised)</td><td>1323.1</td><td>1302.7</td><td>1343.7</td></tr><tr><td>0.8</td><td>PrivTab (linear clipped)</td><td>1299.3</td><td>1277.1</td><td>1321.5</td></tr><tr><td>1.6</td><td>PrivTab (tanh)</td><td>1612.2</td><td>1575.6</td><td>1656.1</td></tr><tr><td>1.6</td><td>PrivTab (shifted softsign)</td><td>1469.9</td><td>1446.3</td><td>1495.0</td></tr><tr><td>1.6</td><td>PrivTab (linear normalised)</td><td>1326.8</td><td>1306.7</td><td>1347.3</td></tr><tr><td>1.6</td><td>PrivTab (linear clipped)</td><td>1282.3</td><td>1262.2</td><td>1302.4</td></tr></table>

## S6.7.6 Ablation: Context-Scale Normalization and Model Combination

The small-context model trained on context lengths sampled uniformly from [100, 2048] benefits substantially from enabling the normalisation in Eq. (S69) at inference time, and the large-context model trained on context lengths sampled uniformly from [1024, 8192] with normalisation is also strong. The final combination described in Supplement S4, which routes to the small-context normalised model for context lengths below 4096 and to the large-context normalised model otherwise, achieves the best aggregate ranking. This supports the motivation for the final benchmark model as a context-scale-aware combination rather than a single fixed variant.

Table S37: ELO rankings for the normalisation and model-combination ablation.
<table><tr><td>µ</td><td>Method</td><td>ELO</td><td>CI min</td><td>CI max</td></tr><tr><td>overall</td><td>PrivTab combination (final)</td><td>1578.1</td><td>1571.0</td><td>1585.2</td></tr><tr><td>overall</td><td>PrivTab small-context model + inference normalisation</td><td>1563.4</td><td>1556.1</td><td>1570.6</td></tr><tr><td>overall</td><td>PrivTab large-context model + normalisation</td><td>1552.2</td><td>1543.8</td><td>1560.7</td></tr><tr><td>overall</td><td>DP-MLP</td><td>1500.5</td><td>1489.1</td><td>1512.0</td></tr><tr><td>overall</td><td>PrivTab small-context model</td><td>1414.3</td><td>1404.3</td><td>1424.1</td></tr><tr><td>overall</td><td>DP-LR</td><td>1411.6</td><td>1400.3</td><td>1422.8</td></tr><tr><td>0.05</td><td>PrivTab small-context model + inference normalisation</td><td>1502.9</td><td>1481.9</td><td>1526.1</td></tr><tr><td>0.05</td><td>PrivTab combination (final)</td><td>1496.6</td><td>1477.5</td><td>1517.0</td></tr><tr><td>0.05</td><td>PrivTab large-context model + normalisation</td><td>1451.2</td><td>1431.3</td><td>1472.3</td></tr><tr><td>0.05</td><td>PrivTab small-context model</td><td>1420.1</td><td>1396.0</td><td>1445.2</td></tr><tr><td>0.05</td><td>DP-LR</td><td>1352.5</td><td>1325.7</td><td>1378.9</td></tr><tr><td>0.05</td><td>DP-MLP</td><td>1271.8</td><td>1244.7</td><td>1297.7</td></tr><tr><td>0.1</td><td>PrivTab small-context model + inference normalisation</td><td>1602.6</td><td>1583.9</td><td>1623.7</td></tr><tr><td>0.1</td><td>PrivTab combination (final)</td><td>1597.5</td><td>1580.2</td><td>1616.1</td></tr><tr><td>0.1</td><td>PrivTab large-context model + normalisation</td><td>1539.2</td><td>1519.3</td><td>1560.6</td></tr><tr><td>0.1</td><td>PrivTab small-context model</td><td>1456.7</td><td>1432.5</td><td>1481.9</td></tr><tr><td>0.1</td><td>DP-LR</td><td>1408.3</td><td>1379.7</td><td>1435.8</td></tr><tr><td>0.1</td><td>DP-MLP</td><td>1377.8</td><td>1350.2</td><td>1404.7</td></tr><tr><td>0.2</td><td>PrivTab combination (final)</td><td>1647.6</td><td>1629.6</td><td>1667.3</td></tr><tr><td>0.2</td><td>PrivTab small-context model + inference normalisation</td><td>1633.5</td><td>1614.9</td><td>1654.7</td></tr><tr><td>0.2</td><td>PrivTab large-context model + normalisation</td><td>1588.6</td><td>1568.2</td><td>1611.5</td></tr><tr><td>0.2</td><td>DP-MLP</td><td>1482.2</td><td>1456.2</td><td>1509.0</td></tr><tr><td>0.2</td><td>PrivTab small-context model</td><td>1447.3</td><td>1424.7</td><td>1470.7</td></tr><tr><td>0.2</td><td>DP-LR</td><td>1438.7</td><td>1408.3</td><td>1468.2</td></tr><tr><td>0.4</td><td>PrivTab combination (final)</td><td>1646.4</td><td>1631.1</td><td>1663.9</td></tr><tr><td>0.4</td><td>PrivTab small-context model + inference normalisation</td><td>1633.2</td><td>1617.8</td><td>1650.5</td></tr><tr><td>0.4</td><td>PrivTab large-context model + normalisation</td><td>1623.3</td><td>1602.3</td><td>1646.1</td></tr><tr><td>0.4</td><td>DP-MLP</td><td>1581.3</td><td>1553.6</td><td>1609.9</td></tr><tr><td>0.4</td><td>DP-LR</td><td>1460.9</td><td>1431.5</td><td>1490.3</td></tr><tr><td>0.4</td><td>PrivTab small-context model</td><td>1441.4</td><td>1416.1</td><td>1467.2</td></tr><tr><td>0.8</td><td>DP-MLP</td><td>1683.5</td><td>1654.8</td><td>1716.5</td></tr><tr><td>0.8</td><td>PrivTab large-context model + normalisation</td><td>1624.5</td><td>1603.6</td><td>1647.0</td></tr><tr><td>0.8</td><td>PrivTab combination (final)</td><td>1615.0</td><td>1599.3</td><td>1632.5</td></tr><tr><td>0.8</td><td>PrivTab small-context model + inference normalisation</td><td>1576.6</td><td>1562.6</td><td>1592.2</td></tr><tr><td>0.8</td><td>DP-LR</td><td>1449.3</td><td>1419.8</td><td>1478.4</td></tr><tr><td>0.8</td><td>PrivTab small-context model</td><td>1406.9</td><td>1378.3</td><td>1434.9</td></tr><tr><td>1.6</td><td>DP-MLP</td><td>1727.5</td><td>1694.1</td><td>1765.7</td></tr><tr><td>1.6</td><td>PrivTab large-context model + normalisation</td><td>1591.6</td><td>1569.6</td><td>1614.9</td></tr><tr><td>1.6</td><td>PrivTab combination (final)</td><td>1574.8</td><td>1557.7</td><td>1594.4</td></tr><tr><td>1.6</td><td>PrivTab small-context model + inference normalisation</td><td>1538.6</td><td>1523.3</td><td>1555.2</td></tr><tr><td>1.6</td><td>DP-LR</td><td>1435.9</td><td>1408.2</td><td>1464.2</td></tr><tr><td>1.6</td><td>PrivTab small-context model</td><td>1376.7</td><td>1348.2</td><td>1403.8</td></tr></table>