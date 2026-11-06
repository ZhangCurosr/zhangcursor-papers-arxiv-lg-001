# Adapters as Data: The Latent Space of LoRAs

Moritz Grävinghoff<sup>1,\*</sup> Konstantin Wehmeyer<sup>1</sup> Aron Asefaw<sup>1</sup>

Boris Knyazev<sup>2,3</sup>

Damian Borth<sup>1,3</sup>

<sup>1</sup>University of St.Gallen <sup>2</sup>Université de Montréal <sup>3</sup>Weight Space Labs

## Abstract

LoRA adapters are usually treated as finished artifacts rather than as a data modality. Prior work has shown that LoRA adapters populate a low-dimensional manifold that a linear PCA basis already organizes well enough to support different operations such as sampling, editing, and inversion. We ask what a learned, nonlinear latent adds beyond this. We train a transformer autoencoder on the same population of LoRA identity adapters, mapping each into a latent space and compare every operation against the identical operation applied directly to the raw adapter weights. The latent preserves the capabilities already available in weight space, sampling diverse identities and editing individual attributes. Beyond preservation, we show that the latent is far more robust to displacement than weight space, degrading into distorted but still valid faces rather than texture. Fixed-dimensional weightspace representations cannot natively host adapters of more than one rank, so we extend our sequential autoencoder to hold mixed-rank adapters within one shared latent. This gives a path toward treating adapter weights of varying architecture as a single, general modality rather than one tied to a fixed rank, and points to a learned, nonlinear latent as the mechanism that could carry this generalization further, toward heterogeneous populations of adapters.

## 1 Introduction

Public repositories now host hundreds of thousands of LoRA adapters, yet each is still produced by its own training run and treated as a finished artifact. A network’s weights are also points in a high-dimensional space, and once many networks exist these points form a broader distribution rather than existing in isolation. This premise underlies weight space learning, which treats the parameters of neural networks as a data modality in their own right and learns representations that predict model properties directly from parameters [Unterthiner et al., 2020, Schürholt et al., 2021, Putterman et al., 2025] or synthesize new weights without training them from scratch [Schürholt et al., 2022, 2024, Knyazev et al., 2023, Wang et al., 2024, Asefaw et al., 2026, Hanna et al., 2026].

A narrower section of this literature has begun to ask the same question of Low-Rank Adaptation (LoRA), now the default mechanism for adapting large foundation models such as Stable Diffusion [Rombach et al., 2022] to new identities, styles, and tasks. Since each adapter is still trained separately, composition of existing adapters remains a heuristic and unstable blend, and an adapter’s behavior is hard to predict before generation. weights2weights (w2w) [Dravid et al., 2024] was first to show that a large collection of identity-encoding LoRAs occupies a low-dimensional manifold that a simple PCA basis already organizes well enough to support sampling, editing, and inversion. We build directly on this setting, using the same base model, the same identity-encoding rank-1 adapters, and the same released dataset, but explore a different direction. Other methods generate LoRA weights outright rather than characterizing the space they come from, whether by pairing an autoencoder with a diffusion prior, mapping task prompts directly to weights, or predicting an adapter from a single image within a fixed subspace, and each is evaluated mainly by the quality of the adapter it produces [Jin et al., 2024, Liang et al., 2025, Ruiz et al., 2024]. What remains open is whether the space a method samples from or restricts itself to is organized in a stronger sense than downstream performance alone can reveal. That would mean neighborhoods smooth under perturbation, operations that behave consistently rather than only within the convex hull of what was fit, and a representation flexible enough to host more than one fixed adapter architecture at once.

We study this question on a sample of w2w’s population of roughly 65,000 identity-encoding LoRA adapters, each fine-tuned on a shared frozen diffusion backbone to reproduce one synthetic face identity and annotated with binary CelebA attributes. Rank-1 adapters are our primary setting, matching w2w directly, and we later extend to rank-2 adapters as well. Using the SANE methodology [Schürholt et al., 2024], these weights are tokenized and an autoencoder is trained in a self-supervised manner to map them into a latent space Z, with no contrastive objective, and no attribute conditioning. Our central question is whether Z is a structured representation of the adapter population, rather than merely a compression of it. We probe this along three axes: (i) whether operations already known to work directly on raw adapter weights, sampling new identities and steering individual attributes, are preserved once routed through a learned, nonlinear latent; (ii) whether that latent is more robust to displacements than the raw weight space; and (iii) whether a single autoencoder can absorb adapters of more than one rank. Our contributions are as follows.

• We introduce a methodology for tokenizing LoRA adapters into weight sequences and embedding them into a structured latent space with a self-supervised transformer autoencoder.

• We show that the resulting latent preserves the sampling and editing capabilities already present in the raw weight space, is substantially more robust to displacement, and remains stable under repeated encode-decode cycles.

• We extend the representation to multi-rank LoRA adapters via a padding scheme that lets a single autoencoder process adapters of different rank within one shared latent.

## 2 Related Work

Weight Space Learning. Representation learning over neural network weights spans both model analysis and weight generation. Regarding model analysis, work focuses on predicting model properties from weights [Unterthiner et al., 2020, Eilertsen et al., 2020] or studies the structure of trained weights itself, motivating permutation- and equivariance-aware representations [Navon et al., 2023, Zhou et al., 2023, Kofinas et al., 2024, Lim et al., 2024]. Closer to what we build on, a separate line of work treats populations of weights themselves as a data modality independent of any specific downstream task. Schürholt et al. [2021] introduce self-supervised embeddings of model populations and subsequently show these can be used generatively, sampling unseen but performant weights for initialization, ensembling, and transfer learning [Schürholt et al., 2022]. SANE [Schürholt et al., 2024] scales this idea beyond small networks by tokenizing weight matrices into a sequence and processing it with a transformer autoencoder, which keeps the per-token dimensionality independent of overall model size, and is the architecture we adopt in this work. In parallel, weight generation has also been explored directly with diffusion and other generative priors [Wang et al., 2024, 2025, Soro et al., 2025]. A broader overview is provided in Han et al. [2026], Ettling et al. [2026]. Collectively, this body of work establishes that weight tensors can be treated as a learnable distribution with its own latent structure.

Attribute Editing and Adapters. Low-Rank Adaptation (LoRA) [Hu et al., 2022] freezes a pretrained model and learns a low-rank update for selected weight matrices, typically at a single fixed rank shared across an entire collection of adapters, and is the adapter format we study throughout. FVAE-LoRA [Kumar et al., 2025] further separates task-salient from residual information through variational factorization of layer activations during fine-tuning, whereas we learn a latent over adapter weights. Existing adapters can be combined post hoc, e.g. ZipLoRA [Shah et al., 2024] merges an independently trained subject and style LoRA via a learned coefficient mask, but such methods treat each adapter as fixed and optimize a merging procedure rather than a shared latent. Attribute control is more commonly approached through a learned direction in an existing latent: Concept Sliders [Gandikota et al., 2024] train a dedicated LoRA per attribute on a frozen diffusion model, while InterFaceGAN [Shen et al., 2020], GANSpace [Härkönen et al., 2020], and SeFa [Shen and Zhou, 2021] fit, respectively, a linear SVM boundary, an unsupervised PCA direction, and a closedform eigendecomposition to steer a pretrained GAN’s latent. We apply the same two ideas, a fitted direction for editing and PCA for diversity, one level removed to a latent over adapter weights instead of images.

![](images/53f30f954c9ad0368cf35f2dc727dd015788320757e322ae17d5168d8ff64ada.jpg)  
Figure 1: Overview of our approach. (a) The LoRA matrices A and B are tokenized and encoded by a transformer encoder E into a latent z, which a decoder D reconstructs back into an adapter. (b) We study five operations on this latent: sampling new identities, editing individual attributes, perturbing an adapter by a controlled amount, testing stability under repeated encode-decode cycles, and extending the latent to adapters of more than one rank. (c) The reconstructed or manipulated adapter modulates a frozen latent diffusion model to generate an image conditioned on a prompt p.

Modeling LoRA Populations. A separate line of work generates LoRA adapters directly from a population, rather than editing or combining existing ones one at a time. COND P-DIFF [Jin et al., 2024] adds a conditional diffusion prior over an autoencoded LoRA latent, and Drag-and-Drop LLMs [Liang et al., 2025] uses a hypernetwork to map task prompts directly to LoRA weight updates in a single forward pass, without any latent space. We instead embed adapters without any external conditioning signal, using only the weights themselves, so the same encoder can in principle be applied to any adapter, rather than requiring a paired task description or prompt. weights2weights (w2w) [Dravid et al., 2024] shows that a large collection of identity-encoding LoRAs already lies on a manifold that a simple PCA basis organizes well enough to support sampling, editing, and inversion, and is the direct baseline we build on and compare against throughout this paper. All existing methods are built around adapters of a single fixed rank and offer no way to mix ranks without being rebuilt from scratch. We address that limitation directly by extending a learned latent to hold adapters of more than one rank in a single shared representation.

## 3 Approach

We take a single LoRA identity adapter θ = (A, B), trained on top of a frozen base model, and embed it into a structured latent space from which new adapters can be sampled and edited. Figure 1 shows the full path. Following the SANE scheme [Schürholt et al., 2024], the adapter matrices are reshaped and sliced row-wise into a sequence of fixed-size tokens, yielding a [312 × 320] token sequence for the rank-1 adapters that form our primary setting, together with a positional encoding and a mask marking real weights against padding. A GPT-style, non-causal transformer encoder E maps this sequence to a latent $z = \overrightharpoon { E ( \theta ) } ^ { * } \in \mathbb { R } ^ { 3 1 \widetilde { 2 } \times 2 5 6 }$ , preserving the token axis, and a symmetric decoder D reconstructs the tokenized weights from it, trained with a masked reconstruction objective. Once an adapter is represented as a latent z, we study a range of operations on that space. We sample new latents that decode to novel but coherent adapters and steer a latent along learned directions to edit specific attributes (Section 5). We probe the geometry of that space under perturbation and test its stability under repeated encode-decode cycles (Section 6). We extend the representation itself to host adapters of more than one rank within a single shared latent (Section 7). In every case the manipulation happens on z and is realized by decoding and de-tokenizing the latent into an adapter ${ \hat { \theta } } ,$ applying it to the frozen base model, and rendering it to an image under a fixed prompt and seed, so that differences between renders are attributable to the adapter rather than to the rendering process. When applicable, every operation we apply to z is additionally applied directly to θ, bypassing the autoencoder entirely. This weight-space control is what lets us attribute any structure we find to the latent representation rather than to the adapter population itself. Architecture, training, and tokenization details, including the alternative scheme used for the mixed-rank setting (Section 7), are given in Appendix A.

## 4 Experimental Setup

## 4.1 Data

Our adapters are LoRA fine-tunes of a single, shared, frozen base diffusion model, Realistic Vision v5.1<sup>1</sup>, a community fine-tune of Stable Diffusion 1.5 [Rombach et al., 2022] optimized for photorealistic portraiture. Sharing one base model is what makes it meaningful to compare, interpolate, and embed different adapters, since every θ modifies the same underlying weights. Each adapter is trained with a DreamBooth-style objective [Ruiz et al., 2023] to reproduce one synthetic identity under a fixed identifier token. For our primary, rank-1 setting we follow Dravid et al. [2024], whose released dataset and binary CelebA [Liu et al., 2015] attribute labels we use throughout. The dataset comprises 64,974 identity adapters labeled along 40 CelebA attributes with widely varying prevalence, from near-balanced (Male, 48.3%) to rare (Bald, 2.5%). One attribute, Wearing Hat, is constant at 0% across the dataset and is excluded from any attribute-conditioned experiment, leaving 39 variable attributes. From this pool we subsample 8,192 adapters, split into train, validation, and test, giving ∼ 5,734 training adapters. The full set of released adapters corresponds to only 9,256 distinct underlying identities (median 6 adapters each), and our splits are drawn per adapter rather than per identity, so near-duplicate fits of the same person can appear across different splits. For the rank-2 setting used in our mixed-rank experiments, no equivalent released population exists, so we train a smaller pool of rank-2 adapters ourselves, following the same base model, DreamBooth objective, and identifier-token convention as the rank-1 dataset. Details are given in Appendix A.

## 4.2 Evaluation

Most of our experiments ask qualitative questions, e.g. whether a sampled latent decodes to a genuinely new person, or whether an edit moved one attribute while leaving the rest of the identity unchanged, and a quantitative score can only approximate this. We nonetheless report two scores where they are informative. For weight-space fidelity, we use the coefficient of determination $R ^ { 2 } ( \theta , { \hat { \theta } } )$ between an adapter and its reconstruction, and for image-space fidelity, we compute the LPIPS distance [Zhang et al., 2018] between the renders of an original and a reconstructed or manipulated adapter. Neither is a reliable proxy for identity preservation on its own and aggregating either can hide exactly the qualitative distinctions our experiments turn on, such as an adapter degrading into texture versus into a different but valid face. We therefore treat rendered figures as the primary evidence throughout, and use $R ^ { 2 }$ and LPIPS to establish trends across many checkpoints or identities, and to surface differences too small to see reliably. Across 32 checkpoint–identity pairs, the two scores are in fact tightly coupled (Pearson and Spearman $\rho = - 0 . 9 0 , \dot { p ^ { < } } 1 0 ^ { - 1 1 }$ , also holding under SSIM [Wang et al., 2004] and pixel-wise MSE), confirming that weight-space $R ^ { 2 }$ and perceptual quality track each other closely.

## 5 The Latent Space Preserves Properties of the Weight Space

In this section we test whether operations already known to work directly on raw adapter weights [Dravid et al., 2024] also work when routed through a learned, nonlinear latent trained with no label supervision. We examine two such operations here, sampling new identities and editing existing ones, and defer a third, arithmetic and interpolation between two identities, to Appendix B.1.

## 5.1 Sampling

Can a latent code drawn without reference to any training adapter be decoded into a new, coherent identity, and if so, are the draws diverse rather than collapsing onto a single average?

Setup. The simplest scheme, naive sampling, fits an independent Gaussian to each latent coordinate, using the per-dimension mean and standard deviation of the encoded training latents, and draws each coordinate independently from its own marginal. A sample is drawn as a full $[ 3 1 2 \times 2 5 6 ]$ latent sequence. Because independent per-dimension sampling discards the covariance structure of the latent, the correlations between coordinates that encode which combinations of features co-occur among real identities, we also sample in the principal-component basis of the encoded training latents. A PCA is fitted to the flattened training latents, the top k components are retained, and a sample is drawn as

$$
z _ { \mathrm { s a m p l e } } = { \bar { z } } + \eta \sum _ { j = 1 } ^ { k } \xi _ { j } \ { \sqrt { \lambda _ { j } } } u _ { j } , \qquad \xi _ { j } \sim { \mathcal N } ( 0 , 1 ) ,
$$

where z¯ is the mean latent, $u _ { j }$ and $\lambda _ { j }$ are the j-th principal component and its variance, and $\eta$ is a diversity factor that trades realism against variety: $\eta = 1$ reproduces the natural spread of the population, larger η pushes samples further from the mean.

Results. Naive sampling produces valid, well-formed faces, but nearly the same face every time (Figure 2, left). Decoded identities share pose, coloring, and features to the point of looking like minor variations of one person, confirming that samples decode to coherent faces but collapsing toward an average identity rather than generating diversity. Sampling in the principalcomponent basis recovers the missing variety (Figure 2, right). With $\bar { k } = 2 5 6$ components and $\eta ~ = ~ 1$ , decoded samples are

(a) Naive sampling  
![](images/0eaf1760f4be42af6665fa2644bc24f2727f6671666a189d1a6f9fd15cee6be2.jpg)

![](images/33d381ec8241aac1fd58b899075408ab852c5d1c56ec066319f131a8e97b0621.jpg)

(b) PCA sampling  
![](images/7abafb3f0bd6b7441cfb83c677411a42c79a9238c5fec0af806cfff8f52c40df.jpg)

![](images/cada9702782701d6f0ff78b87ebda3dbf18bc3bc9d8adff60b8bcd3c0c413bf9.jpg)  
Figure 2: (a) Naive sampling collapses toward average-looking identities. (b) PCA sampling with $\eta = 1$ and $k = 2 5 6$ preserves substantially greater diversity.

clearly distinct identities, differing in gender, age, hairstyle, and coloring, while remaining coherent and realistic. So many components are needed because the latent’s variance is spread broadly rather than concentrated in a few directions. The leading component alone explains only 3.1% of it, the top 10 only 11.9%, and even the top 256 together capture just 40.9%.

Relation to w2w. weights2weights [Dravid et al., 2024] follows the same recipe directly on raw adapter weights, fitting a PCA basis over the flattened LoRA updates and sampling coefficients along the leading components to obtain new identities. That we need the identical construction here is consistent with our autoencoder never being pushed to disentangle. Since it was trained with a reconstruction objective alone and no prior encouraging independent, unit-scale dimensions, there is no reason to expect the latent’s coordinate axes to align with the population’s true directions of variation, and naive per-dimension sampling requires exactly that alignment to work.

![](images/f6ed0f5c1db25312fbd27ea7c78249689ccc41a6464e21dc90f9167b9f3079c7.jpg)  
Figure 3: Editing along the fitted ridge direction produces the same full-hair-to-bald transition whether applied to the latent (top) or directly to the raw adapter weights (bottom). The two rows use different γ ranges since the direction’s scale differs between the two spaces.

## 5.2 Editing

Given one identity, can a single named attribute be changed continuously while the rest of the identity is left intact?

Setup. We estimate an attribute direction v by ridge regression, where training latents are flattened into $\mathbf { \dot { Z } } \in \mathbb { R } ^ { N \times D } \left( N = 5 , 7 3 4 , D = 7 9 , 8 7 2 \right)$ , CelebA labels are mapped to $\check { y \in \{ 0 , 1 \} } ^ { N }$ , and v is the ridge solution v = arg $\begin{array} { r } { \operatorname* { m i n } _ { w } \| Z _ { c } w - y _ { c } \| _ { 2 } ^ { 2 } + \lambda \| w \| _ { 2 } ^ { 2 } \left( \lambda = 1 \right) } \end{array}$ on centered data, solved in its dual form since $D \gg N$ . This serves a similar purpose to the implicit shrinkage w2w [Dravid et al., 2024] gets from truncating its PCA basis, though ridge shrinks every direction smoothly rather than discarding low-variance ones outright. An identity is edited as $z ^ { \prime } = z + \gamma v \ ( { \mathrm { w i t h } } \ \gamma = 0 $ giving back the reconstruction), where $\gamma$ is expressed as a fraction of ∥z∥ so that edit strength is comparable across identities and directions. Before rendering any sweep, we also score each direction v by the held-out AUC of adapters ranked by their projection onto v, the probability that a randomly chosen positive outranks a randomly chosen negative.

Results. On Bald, one of the rarest attributes in the dataset (2.5%), the flattened edit still produces a clean, continuous sweep (Figure 3). $\mathrm { A t } \gamma = 0$ the reconstruction renders the unedited identity, and as γ increases the hairline recedes and then disappears into a fully bald head, with intermediate strengths producing intermediate hairlines and the rest of the identity, skin tone, expression, and framing, preserved throughout. The identical ridge direction fitted directly on the raw adapter weights and applied without the autoencoder produces the same qualitative result over a comparable sweep (Figure 3). This matches what w2w [Dravid et al., 2024] already showed for a linear direction fitted post hoc on raw LoRA weights, here with ridge in place of a truncated-PCA pseudoinverse. Applying the same construction to four further attributes on held-out identities (Figure 4, more examples in Figure 11 in Appendix B.2) shows edit quality tracking

![](images/91b285c27abc86bd1ffe6b293068ae56507e3b4bbfde07d7cc296e48ddcfe85d.jpg)  
Figure 4: Edit quality declines with the direction’s heldout AUC, from clean edits for Male and Eyeglasses to a barely visible change for Oval Face.

the AUC score fairly well. Male and Eyeglasses score highest and edit the most cleanly, the former moving continuously between a feminine and a masculine rendering with hair length, jawline, and facial hair changing together, the latter adding glasses as a discrete object. Gray Hair follows close behind with a milder but still clearly visible lightening of the hair, while Oval Face, well below the others, produces an edit that barely registers.

(a) Weight-Space Perturbation  
![](images/f2e8fefc6fcbd413dd48cdd68558b487c5b747f7f818fe774f6ba180b461d5b4.jpg)

(b) Latent-Space Perturbation  
![](images/ff734e375c1d663b77d5005218e06985aa52ac55588c14ac296e7e7edb342d29.jpg)  
Figure 5: Perturbations in (a) weight-space and (b) latent-space across increasing noise levels α. Each column depicts one identity under the same perturbation level.

## 6 The Latent Space Improves Robustness

This section tests the latent under two kinds of displacement: a single perturbation of varying size, compared directly against the same perturbation applied to the raw weights, and repeated encode– decode passes on the same adapter, which is what happens in practice whenever a sampled or edited latent is fed back through the autoencoder.

## 6.1 Perturbation

How far can an adapter be perturbed before the identity breaks down, and does it fail gracefully or collapse abruptly?

Setup. We test how far an adapter can be moved, in either space, before the generated identity breaks down, and whether it degrades smoothly or collapses abruptly. Each perturbation is scaled by the standard deviation of the space it acts in, so a noise scale α denotes the same self-relative displacement, a fraction α of that space’s own spread. In the latent we form $z ^ { \prime } = z + \alpha \hat { \sigma } _ { z } \varepsilon$ for a real identity’s latent $z = E ( { \boldsymbol { \theta } } )$ , with $\varepsilon \sim \mathcal { N } ( 0 , I )$ and $\hat { \sigma } _ { z }$ the global standard deviation of the latent. In weight space we form the analogous $\theta ^ { \prime } = \theta + \alpha \hat { \sigma } _ { \theta } \varepsilon$ , with $\scriptstyle { \hat { \sigma } } _ { \theta }$ the global standard deviation of the adapter population. We sweep α over the same range in both spaces.

Results. Weight-space perturbation degrades the identity abruptly (Figure 5, left). At $\alpha = 0 . 5$ identities remain broadly recognizable, though some detail is already lost. $\mathbf { B y } \alpha = 0 . 7 5$ most have collapsed into texture-like, off-distribution images that are no longer faces. $\mathrm { A t } \alpha = 1 . 0$ every identity has dissolved into noise. The transition is sharp, with little intermediate regime in which the adapter is perturbed yet still valid. Perturbing the latent by the same fractional amount degrades far more gracefully (Figure 5, right). $\mathrm { A t } \alpha = 0 .$ 5 identities are essentially preserved. $\mathrm { A t } \alpha = 0 . 7 5$ , where the weight-space adapters have already collapsed, latent-perturbed identities remain clearly recognizable despite larger variation. Only at $\alpha = 1 . 0$ does the identity begin to break down, and even then it tends to degrade into a different or distorted face rather than into the texture-like noise seen in weight space. The raw weight space supports valid adapters only within a narrow band around each trained adapter. The neighborhood of a real point in the latent space, by contrast, is populated by valid, related identities.

![](images/83d719478bab2d604fade456dc252b04ac944d7b3f410420375ec02c7fe27643.jpg)

![](images/4f74b7108fcbbc8330b0267845546a7fe6528ff82f257b618247418c548dc408.jpg)  
Figure 6: Stability under repeated encode–decode cycles. (a) Weight-space similarity to the original adapter decreases gradually. (b) Rendered images increasingly differ from the original, measured by LPIPS. Shaded regions denote ±1σ. (c) Representative renders after $k = 0 , 1 , 3 , \bar { 5 } , 7$ , 10 cycles.

## 6.2 Reconstruction Drift

Does the small error ofa single encode–decode pass compound under repetition, or does it settle to a fixed point?

Setup. We repeatedly apply the encoder and decoder to the same adapter. Starting from a real adapter $\theta ^ { ( 0 ) } = \theta .$ , we form $\hat { \theta } ^ { ( k ) } = D ( E ( \theta ^ { ( k - 1 ) } ) )$ ) for $k = 1 , \ldots , 1 0$ , applying each $\theta ^ { ( k ) }$ to the frozen base model and rendering it under the fixed prompt and seed used throughout. We track $R ^ { 2 }$ between $\theta ^ { ( k ) }$ and the original ${ \theta } ^ { \left( 0 \right) }$ and the LPIPS distance between their renders. A per-step change that decays toward zero would indicate a fixed point, while a change that stays roughly constant would indicate unbounded accumulation.

Results. Identities survive all ten cycles (Figure 6). Facial structure, hair, coloring, expression, and framing remain intact, and almost all visible change happens at the first cycle alone. The two quantitative measures capture different sides of this. LPIPS saturates, rising to 0.23 after one cycle and reaching only 0.36 after ten. $R ^ { 2 }$ declines steadily from 1 to about 0.945, and the per-cycle loss grows rather than shrinks, from 0.004 over the first five cycles to 0.007 over the second five. Ten cycles are safe in practice, since the renders stay recognizable and the perceptual cost of each additional cycle is small, but that safety reflects the renderer’s tolerance rather than the autoencoder converging. The weight-space deviation gives no sign of stopping, so extrapolating to many more cycles is not supported by this result.

## 7 The Latent Space Generalizes Across Adapter Ranks

Can a single autoencoder absorb adapters ofvarying ranks, treating adapter weights as a general modality rather than one tied to a fixed architecture?

Setup. Doubling the LoRA rank doubles adapter dimensionality from $d = 9 9 , 6 4 8 \mathrm { t o } d = 1 9 9 , 2 9 6 .$ so a rank-2 adapter does not fit the token sequence a rank-1 model expects. Our dense tokenizer assigns each token a fixed range of weight indices. At rank 2, the same token index falls in a different part of the network than it does at rank 1. The positional encoding the encoder relies on would therefore become misleading. We instead use a boundary-respecting sparse tokenizer at width 640, sized to a matrix’s rank-2 parameter count. A rank-1 matrix then occupies only the first half of its token budget, with the remainder marked as padding. Token position refers to the same matrix at both ranks. We train one autoencoder on a pool containing both rank-1 and rank-2 adapters and compare it against the rank-1-only model used throughout the rest of our experiments, reporting reconstruction quality as LPIPS on held-out adapters of each rank.

Rank-1 Adapters Original  
![](images/2a1ec88fbc39dfec729c4db01c129884a526204309cc02794e40712ae82ebf29.jpg)  
Figure 7: A single mixed-rank autoencoder reconstructs recognizable identities at both ranks. Rows 1–3 compare rank-1 adapters reconstructed by the rank-specific model against the mixed-rank model, and rows 4–5 show rank-2 adapters reconstructed by the mixed-rank model.

Results. A single autoencoder, with no increase in latent size or model capacity, reconstructs recognizable identities at both ranks (Figure 7). The rank-2 half of every token is therefore genuinely learned rather than ignored, and a shared latent over two adapter ranks is achievable within the same backbone used throughout the rest of our experiments. This capacity is now split between two ranks rather than dedicated to one, and the cost shows up in fidelity. On rank-1 adapters the mixed model reaches an average LPIPS of 0.29 against 0.22 for the rank-1-only model, about 32% worse, and it reaches 0.35 on rank-2 adapters. Section 4.2 shows that weight-space R<sup>2</sup> and perceptual quality track each other closely, so this gap reflects a loss of reconstruction and places the mixed model in the range we observed for rank-1 adapters before convergence.

## 8 Limitations

Splitting. Train, validation, and test splits are drawn per adapter rather than per identity, so nearduplicate fits of the same person can appear across different splits. This does not affect the qualitative renders that carry most of our claims, but it makes held-out numbers such as the AUC optimistic relative to a strict identity-level split, since a direction can be scored against adapters that are nearduplicates of ones it effectively saw in training. Identity-level splitting is a straightforward fix left to future work.

Reconstruction floor. The best rank-1 model reconstructs adapters to an average LPIPS of about 0.22, and the first encode-decode cycle in our drift experiment costs 0.23, almost the same number, and larger than the drift accumulated over the following nine cycles combined. Every latent operation in this paper is therefore observed through a reconstruction already perceptibly different from its source, which is why the $\gamma = 0$ control in each edit sweep is $D ( \dot { E } ( \bar { \theta } ) )$ rather than θ itself. w2w, which samples and edits directly in weight space, has no equivalent floor.

Scope. Several limits on scope apply throughout. Everything rests on rank-1 and rank-2 updates to the query and value projections of a single frozen backbone, trained by the same DreamBooth-style procedure, so no claim about adapter weights as a general modality follows from this paper alone. Attribute editing also does not strictly require the learned latent, since the same ridge direction fitted directly on raw adapters produces a comparable edit, reproducing w2w in a different basis. The attribute fits themselves are underdetermined. Training uses about 5,734 adapters against a flattened latent of $^ { 7 9 , 8 7 2 }$ dimensions, so the ridge fits interpolate the labels even at $\lambda = 1$ , and the fitted direction itself is not uniquely identified by the data.

Out-of-distribution generalization. We test generalization across LoRA rank but not across subject domain. The autoencoder is trained exclusively on adapters encoding human identities, and whether its latent organizes adapters more broadly, for instance the dog-breed adapters also released by w2w, or has specialized to faces specifically, is a question this paper leaves open for future work.

## 9 Discussion

A structured latent over adapters mainly changes cost rather than capability. A new identity becomes a decoder pass instead of a training run, and a new attribute becomes one closed-form solve against an existing Gram matrix instead of a dedicated LoRA. More broadly, our experiments show that the different operations possible in the LoRA weight space are preserved in the learned latent, with two distinct advantages beyond preservation. The first advantage appears under displacement. Perturbations of comparable magnitude in weight space cause considerable degradation of the identity, while the learned representations show greater robustness and still decode into recognizable identities. This behavior is particularly relevant for downstream operations such as optimization and generation, where greater robustness to perturbations is necessary. The second primary advantage is the possibility to learn representations of different rank adapters. Operations in the raw adapter weight space require a shared dimensionality, whereas our tokenized approach represents different rank adapters within a common latent space.

Taken together, these results show that a learned LoRA latent space provides a more robust and extensible representation for performing multiple different downstream operations. Whether this advantage hold across different domains and tasks remains a key question for moving towards adapter weights as a general data modality.

## 10 Broader Impact

The identities studied here are synthetic, but nothing in the method depends on that, and composing this latent with an inversion step like w2w’s, turning a single photograph into a point in weight space, would in principle extend it to real people, letting a specific individual’s apparent age or gender presentation be moved continuously while their identity is preserved. This kind of consistency is exactly what makes fake personas cheap at scale. A second concern sits in the attribute directions themselves. Several CelebA attributes are annotator judgments rather than observable properties, and fitting a direction to one turns a cultural judgment into a coordinate that can be adjusted continuously on a real person’s face. This paper uses only synthetic identities and does not implement inversion, which keeps it clear of the most direct misuse path, though that is a property of the experiments rather than a guarantee of the method. We take the position that adapters encoding real people should require consent at the level of the identity, obtained beforehand.

## References

Aron Asefaw, Konstantinos Tzevelekakis, Damian Falk, Léo Meynent, and Damian Borth. Weight-CLIP: Aligning datasets and models for weight space learning. In Forty-third International Conference on Machine Learning, 2026.

Amil Dravid, Yossi Gandelsman, Kuan-Chieh Wang, Rameen Abdal, Gordon Wetzstein, Alexei A. Efros, and Kfir Aberman. Interpreting the weight space of customized diffusion models. In Advances in Neural Information Processing Systems, volume 37, pages 137334–137371, 2024.

Gabriel Eilertsen, Daniel Jönsson, Timo Ropinski, Jonas Unger, and Anders Ynnerman. Classifying the classifier: dissecting the weight space of neural networks. In European Conference on Artificial Intelligence, 2020.

Tobias Ettling, Damian Falk, Aron Asefaw, Léo Meynent, Konstantinos Tzevelekakis, Zehong Wang, Kai Wang, Boris Knyazev, Haggai Maron, and Damian Borth. Weight space learning needs to unify benchmarking! a taxonomy of evaluation practices. In The Fortieth Annual Conference on Neural Information Processing Systems Position Paper Track, 2026.

Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept´ sliders: LoRA adaptors for precise control in diffusion models. In European Conference on Computer Vision, pages 172–188. Springer, 2024.

Xiaolong Han, Zehong Wang, Bo Zhao, Binchi Zhang, Jundong Li, Damian Borth, Rose Yu, Haggai Maron, Yanfang Ye, Lu Yin, and Ferrante Neri. A survey of weight space learning: Understanding, representation, and generation. arXiv preprint arXiv:2603.10090, 2026.

Joëlle Hanna, Damian Falk, Stella X. Yu, and Damian Borth. Geosane: Learning geospatial representations from models, not data. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27804–27814, June 2026.

Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. GANSpace: Discovering interpretable GAN controls. In Advances in Neural Information Processing Systems, volume 33, 2020.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. arXiv:2106.09685.

Xiaolong Jin, Kai Wang, Dongwen Tang, Wangbo Zhao, Yukun Zhou, Junshu Tang, and Yang You. Conditional LoRA parameter generation. arXiv preprint arXiv:2408.01415, 2024.

Boris Knyazev, Doha Hwang, and Simon Lacoste-Julien. Can we scale transformers to predict parameters of diverse ImageNet models? In Proceedings of the International Conference on Machine Learning, 2023.

Miltiadis Miltos Kofinas, Boris Knyazev, Yan Zhang, Yunlu Chen, Gertjan J Burghouts, Efstratios Gavves, Cees G Snoek, and David Zhang. Graph neural networks for learning equivariant representations of neural networks. In International Conference on Learning Representations, 2024.

Shashi Kumar, Yacouba Kaloga, John Mitros, Petr Motlicek, and Ina Kodrasi. Latent space factorization in lora. Advances in Neural Information Processing Systems, 38:88111–88138, 2025.

Zhiyuan Liang, Dongwen Tang, Yuhao Zhou, Xuanlei Zhao, Mingjia Shi, Wangbo Zhao, Zekai Li, Peihao Wang, Konstantin Schürholt, Damian Borth, Michael M. Bronstein, Yang You, Zhangyang Wang, and Kai Wang. Drag-and-drop LLMs: Zero-shot prompt-to-weights. In Advances in Neural Information Processing Systems, volume 38, pages 153258–153283, 2025.

Derek Lim, Haggai Maron, Marc T. Law, Jonathan Lorraine, and James Lucas. Graph metanetworks for processing diverse neural architectures. In International Conference on Learning Representations, 2024.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 3730–3738, 2015.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, and Haggai Maron. Equivariant architectures for learning in deep weight spaces. In International Conference on Machine Learning, pages 25790–25816. PMLR, 2023.

Theo Putterman, Derek Lim, Yoav Gelberg, Michael M. Bronstein, Stefanie Jegelka, and Haggai Maron. GL-equivariant metanetworks for learning on low-rank weight spaces. In Proceedings of the Fourth Learning on Graphs Conference, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. HyperDreamBooth: HyperNetworks for fast personalization of text-to-image models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Konstantin Schürholt, Dimche Kostadinov, and Damian Borth. Self-supervised representation learning on neural network weights for model characteristic prediction. In Advances in Neural Information Processing Systems, volume 34, pages 16481–16493, 2021.

Konstantin Schürholt, Boris Knyazev, Xavier Giró-i Nieto, and Damian Borth. Hyper-representations as generative models: Sampling unseen neural network weights. In Advances in Neural Information Processing Systems, volume 35, pages 27906–27920, 2022.

Konstantin Schürholt, Michael W Mahoney, and Damian Borth. Towards scalable and versatile weight space learning. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, pages 43947–43966, 2024.

Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. ZipLoRA: Any subject in any style by effectively merging LoRAs. In European Conference on Computer Vision, pages 422–438. Springer, 2024.

Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in GANs. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of GANs for semantic face editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

Bedionita Soro, Bruno Andreis, Hayeon Lee, Wonyong Jeong, Song Chong, Frank Hutter, and Sung Ju Hwang. Diffusion-based neural network weights generation. In International Conference on Learning Representations, 2025.

Thomas Unterthiner, Daniel Keysers, Sylvain Gelly, Olivier Bousquet, and Ilya Tolstikhin. Predicting neural network accuracy from weights. arXiv preprint arXiv:2002.11448, 2020

Kai Wang, Dongwen Tang, Boya Zeng, Yida Yin, Zhaopan Xu, Yukun Zhou, Zelin Zang, Trevor Darrell, Zhuang Liu, and Yang You. Neural network diffusion. arXiv preprint arXiv:2402.13144, 2024.

Kai Wang, Dongwen Tang, Wangbo Zhao, Konstantin Schürholt, Zhangyang Wang, and Yang You. Scaling up parameter generation: A recurrent diffusion approach. In Advances in Neural Information Processing Systems, volume 38, pages 103214–103246, 2025.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 586–595. IEEE, 2018.

Allan Zhou, Kaien Yang, Kaylee Burns, Adriano Cardace, Yiding Jiang, Samuel Sokota, J Zico Kolter, and Chelsea Finn. Permutation equivariant neural functionals. Advances in neural information processing systems, 36:24966–24992, 2023.

## A Implementation Details

Adapter instantiation. LoRA is inserted into the query and value projections of both self- and cross-attention in the U-Net, while the key projections and all other weights stay frozen. Each pair on an image-space projection contributes 2C parameters at channel width C, and three of the four adapted projections are of this kind. Cross-attention’s value takes the 768-dimensional CLIP text embedding as input, adding a further 768 parameters. Summed over one block this gives $7 C + 7 6 8$ parameters, and over the 16 attention blocks spanning $C \in \{ 3 2 0 , 6 4 0 , 1 2 8 0 \}$ (Table 1), a total adapter dimensionality of $d = 9 9 , 6 4 8$ at rank 1. Doubling the rank doubles the per-block count to $1 4 C + 1 { , } 5 3 6$ and the total to $d = 1 9 9 , 2 9 6$ . The rank-2 adapters used in Section 7 are generated with the original training pipeline released by Dravid et al. [2024], changing only the LoRA rank from 1 to 2. The base model, DreamBooth objective, and prompt template are otherwise identical to the released rank-1 recipe. Our released repository operates on the resulting adapter weights.

Table 1: Per-block and total LoRA parameter count at rank $r = 1 .$
<table><tr><td>Channel width C</td><td>Params per block (7C + 768)</td><td>Blocks</td><td>Total</td></tr><tr><td>320</td><td>3,008</td><td>5</td><td>15,040</td></tr><tr><td>640</td><td>5,248</td><td>5</td><td>26,240</td></tr><tr><td>1280</td><td>9,728</td><td>6</td><td>58,368</td></tr><tr><td></td><td>Total (16 blocks)</td><td>16</td><td>99,648</td></tr></table>

Tokenization. Dense tokenization (Figure 8a) chunks the flattened adapter into fixed-size tokens of width 320, the smallest channel width in our U-Net. The other channel widths are integer multiples of it, so every LoRA matrix except the cross-attention value (which operates on the 768-dimensional text embedding) tiles into a whole number of tokens with no leftover. As an alternative, sparse tokenization (Figure 8b) slices each matrix into its own tokens, fully preserving channel boundaries at the cost of introducing a larger amount of padding. We use dense tokenization by default, since it already preserves the adapter’s structure well enough while avoiding significant padding. For the mixed-rank setting, where rank-1 and rank-2 adapters must share one token sequence, we switch to the sparse tokenizer. Here, each matrix’s token budget is fixed by its rank-2 size, so a rank-1 matrix occupies only the first half of that budget, with the remainder marked as padding by the mask.

![](images/706c438702859af6e94af5fd8d1d3446ed122afcd9dc0aaf0e45ca7b158a4132.jpg)

![](images/b2664e27a62b84b88bbd0ecab5488d17b7b7f9206d4584057d83810645fde1a7.jpg)  
(a) Dense tokenization. Only the final token is padded, (b) Sparse tokenization. The last token of each matrix marked in grey. is padded, marked in grey.  
Figure 8: Dense versus sparse tokenization of a single LoRA adapter.

Architecture. Our primary autoencoder is a GPT-style, non-causal Transformer encoder-decoder following SANE [Schürholt et al., 2024]. The encoder projects each of the $N = 3 1 2$ input tokens from dimension 320 to a model width of 1024, applies 8 layers with 16 attention heads, and projects down to a latent dimension of 256 per token. The decoder is symmetric.

Training. The autoencoder is trained with a masked, batch-normalized MSE computed only over real-weight positions, for 1000 epochs at batch size 32 with AdamW [Loshchilov and Hutter, 2019] (learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } )$ under a one-cycle schedule and a fixed seed of 2020. We use dropout 0.15, full precision, and gradient clipping at a max-norm of 10. The adapter pool used for training is capped at 8,192 adapters and split $7 0 / 1 5 / 1 5$ into train, validation, and test, giving ∼ 5,734 training adapters.

## B Additional Experimental Results

## B.1 Arithmetic and Interpolation

If the latent is organized, then simple vector operations on latent codes should correspond to meaningful, predictable changes in the decoded identity, which we refer to as “face algebra.” We test this directly by applying arithmetic and interpolation to the latents of real identities and inspecting the decoded results.

Setup. Given two identities encoded to latents $z _ { A }$ and $z _ { B } ,$ , we form new latents with a fixed set of elementary operations: the directional differences $z _ { A } \mathrm { ~ - ~ } z _ { B }$ and $z _ { B } \mathrm { ~ - ~ } z _ { A }$ , the midpoint $( z _ { A } + z _ { B } ) / 2$ , the raw sum $z _ { A } + z _ { B }$ , the asymmetric combinations $z _ { A } + z _ { B } / 2$ and $z _ { A } + 2 z _ { B }$ that weight one identity more heavily, and a linear interpolation $z _ { \gamma } = ( 1 - \gamma ) z _ { A } + \gamma z _ { B }$ between the two identities. For interpolation, we additionally apply the identical construction directly to the raw adapter weights $\theta _ { A } , \theta _ { B }$

![](images/f32acbdd7d56c109116e1115ee64a4293ae647d131c3b3a7c8b752d4c02e8589.jpg)  
Figure 9: Arithmetic operations on the latents of two identities, $x _ { A }$ and $x _ { B } ,$ decode to visually sensible results, with the unscaled sum $z _ { A } + z _ { B }$ as the one exception that overloads the combination.

Arithmetic works in the latent. Elementary arithmetic on latent codes translates into visually sensible changes in the decoded identity (Figure 9). For two encoded identities A and $B ,$ the two directional differences $z _ { A } - z _ { B }$ and $z _ { B } - z _ { A }$ subtract one identity’s features from the other in the expected direction. The midpoint $( z _ { A } + z _ { B } ) / 2$ decodes to a clean blend of the two people, and the weighted combinations tilt the result toward the more heavily weighted identity: $z _ { A } + z _ { B } / 2$ gives a subtle influence of $B$ on $A ,$ whereas $z _ { A } + 2 z _ { B }$ amplifies $B$ and pushes it into A. The raw sum $z _ { A } + z _ { B }$ overloads the combination, as expected of an operation that leaves the natural range of the latent. The same pattern holds for other identity pairs.

![](images/4e855211f045770032d17a88c5a44b4bc9327b6b667e73ec5fe44adc15817af5.jpg)  
Figure 10: Interpolating between two identities produces a smooth, valid transition whether performed in the latent (top) or directly in weight space (bottom).

Interpolation also works in weight space. Linearly interpolating between two encoded identities produces a smooth, gradual transition through valid intermediate identities (Figure 10, top). Features change continuously from A to $B$ with no abrupt switch and no invalid faces along the way. Applying the same construction directly to the raw weights (Figure 10, bottom) yields a similarly recognizable transition between the two identities, showing that this operation is preserved even without the autoencoder.

## B.2 Editing

![](images/9e4088c895bcea53b6d2470c6a299dff614417d2526340029c74045d3e28692d.jpg)  
Figure 11: Additional identities for the four attributes in Figure 4, showing that the effect is consistent across identities.