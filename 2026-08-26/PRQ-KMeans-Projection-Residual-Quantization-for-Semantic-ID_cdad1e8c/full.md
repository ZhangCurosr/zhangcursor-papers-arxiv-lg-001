# PRQ-KMeans: Projection Residual Quantization for Semantic ID Tokenization

Yunxiao Luo<sup>∗</sup>, Siyuan Wang<sup>†</sup>, Ben Chen<sup>‡</sup>, Chenyi Lei

<sup>1</sup>Kuaishou Technology

## Abstract

Semantic identifiers (SIDs) represent entities as hierarchical token sequences for generative retrieval and recommendation. Residual-quantization tokenizers construct these sequences by selecting a codeword at each level and passing a residual to the next. We view this process as progressive commonality removal: each token captures a component shared within its group, while later tokens should model the remaining diferences. This view reveals three limitations: a corpus-wide shared component can consume first-level capacity, hard assignment ignores graded similarities to nearby codewords, and full-codeword subtraction can leave variation along the selected-codeword direction in the next residual. We therefore develop our solution in the post-hoc setting, where residual construction is not constrained by input reconstruction. Specifically, we propose PRQ-KMeans, which removes the global-mean component, refines centroids with Top-k similarity-weighted updates, and replaces full-codeword subtraction with a projection residual that removes each representation’s selected-centroid component. Experiments on a large-scale industrial search dataset and four public recommendation benchmarks show that PRQ-KMeans achieves the strongest overall performance among the evaluated tokenizers, including gains of up to 7.4% in HitRate and 11.8% in MRR on the industrial dataset.

## 1 Introduction

Semantic identifiers (SIDs) replace atomic entity identifiers with short token sequences for generative retrieval (Tay et al. 2022; Wang et al. 2022; Sun et al. 2023) and recommendation (Rajput et al. 2023; Singh et al. 2024). Hierarchical SIDs organize these tokens from coarse to fine: related entities can share early tokens, while later tokens distinguish entities that remain under the same prefix. Constructing such an identifier involves two coupled decisions at each level: which token represents the entity at the current position, and what representation is passed forward to construct the next token.

Residual quantization provides a common framework for making these decisions. At each level, it selects a codeword for the current token, subtracts that codeword from the current representation, and passes the resulting residual to the next level. Two major tokenizer paradigms use this recursion. VAE-based methods (Lee et al. 2022; Rajput et al. 2023; Wan et al. 2026) learn hierarchical SIDs through a residual-quantization autoencoder. Post-hoc methods (e.g., RQ-KMeans and RQ-GMM) (Luo et al. 2025; Deng et al. 2025; Tong et al. 2026) instead fit hierarchical codebooks directly over existing embeddings. In both paradigms, the residual becomes the representation processed at the next level. We refer to the component that remains along the selectedcodeword direction after subtraction as residual carryover.

![](images/134ed3f1b51f4cb1bf373548f499da40d8cb2663c8594c472c366c834abfe0ab.jpg)  
Figure 1: Full-centroid subtraction and projection residuals. Full-centroid subtraction may leave positive or negative residual carryover along the selected-centroid direction, whereas projection removes this component and yields a residual orthogonal to that centroid.

Since each later token uses the preceding residual, residual construction determines what information passes across levels. We therefore view residual quantization as progressive commonality removal: the selected codeword represents a component shared within the current group, while the residual should retain the diferences that remain for later tokens. This view reveals three challenges. First, commonality can exist before any cluster-level token is formed. A component shared across the embedding collection may dominate the initial representations. Because it is broadly shared, this component contributes little to distinguishing entities, yet the first codebook may spend part of its capacity modeling this global background rather than coarse diferences among entities.

Second, hard assignment limits how accurately a codeword can summarize its group. Each representation updates only its selected codeword, even when it is also similar to nearby candidates, discarding graded similarity around assignment boundaries. Third, full-codeword subtraction does not generally remove the selected-codeword component accurately from every representation. As Figure 1 illustrates, representations assigned to the same codeword may contain diferent amounts of this component, although the same codeword is subtracted from all of them. This mismatch can leave a positive or negative component along the selected direction, causing the next codebook to revisit variation already represented by the current token. Our preliminary study finds this component measurable in industrial RQ-KMeans residuals and shows that retaining more of it monotonically reduces next-level SID-prefix utilization (Section 3).

VAE-based tokenizers can reshape their latent representations and codebooks through reconstruction and auxiliary SID objectives (Wang et al. 2024; Zhu et al. 2024; Wan et al. 2026). Their residuals, however, are constrained by additive reconstruction: everything left after subtracting the current codeword is passed to later codewords, even a component along the same direction. Discarding that component independently would alter the codeword sum used for reconstruction. Post-hoc tokenizers instead operate on existing embeddings without requiring the codewords to reconstruct the input, allowing the residual to preserve only the diferences needed by later SID levels. We develop our solution in this post-hoc setting using RQ-KMeans.

To address these challenges, we propose PRQ-KMeans, a post-hoc hierarchical SID tokenizer built around projection residuals. To keep the corpus-wide component from consuming first-level capacity, PRQ-KMeans removes each input representation’s projection onto the global mean before fitting the first codebook. To better estimate the component shared within each cluster, Top-k soft refinement lets every representation contribute to several nearby centroids in proportion to their cosine similarities, rather than updating only its most similar centroid. To prevent the selectedcentroid component from being passed to the next level, PRQ-KMeans removes each representation’s own projection onto its selected centroid after hard assignment, rather than subtracting the full centroid vector. The resulting residual is orthogonal to that centroid and is passed forward to fit the next codebook. Together, these operations implement progressive commonality removal from global to local.

Our main contributions are threefold. (1) We introduce the view of hierarchical SID construction as progressive commonality removal and identify three challenges in determining what is passed between successive token levels. (2) We propose PRQ-KMeans, which combines global component removal and Top-k soft centroid refinement with a projection residual that removes this component separately for each representation. (3) We demonstrate the efectiveness of PRQ-KMeans through extensive evaluation on industrial search and four public recommendation benchmarks.

## 2 Related Work

## 2.1 Semantic IDs in Generative Retrieval and Recommendation

Semantic IDs represent entities as short sequences of discrete tokens. For generative retrieval, DSI (Tay et al. 2022) establishes the paradigm of directly generating document identifiers, and NCI (Wang et al. 2022) extends this paradigm with hierarchical semantic identifiers. GenRet (Sun et al. 2023), GLEN (Lee, Choi, and Lee 2023), and MERGE (Zhang et al. 2025a) incorporate semantic, lexical, or relevance signals into identifier construction. OneSearch (Chen et al. 2025) applies hierarchical SIDs to large-scale generative e-commerce search. For generative recommendation and ranking, TIGER (Rajput et al. 2023), Semantic IDs for Ranking (Singh et al. 2024), and OneRec (Deng et al. 2025) use hierarchical SIDs to organize the output space. LETTER (Wang et al. 2024), CoST (Zhu et al. 2024), ETEGRec (Liu et al. 2024), DI-GER (Fu et al. 2026), and UniSID (Jiang et al. 2026) further incorporate collaborative, contrastive, diferentiable, or endto-end supervision into SID learning. For embedding-derived SIDs, the tokenizer determines how continuous entity representations are mapped to discrete token sequences.

## 2.2 Quantization Methods for Hierarchical SIDs

Residual quantization constructs a hierarchical SID by successively quantizing the residual passed from one level to the next. RQ-VAE (Lee et al. 2022; Rajput et al. 2023) and R3-VAE (Wan et al. 2026) use autoencoders and recursive residual quantization to learn hierarchical SIDs. Post-hoc methods fit the hierarchy over existing embeddings: QARM (Luo et al. 2025) and OneRec (Deng et al. 2025) use RQ-KMeans, while RQ-GMM (Tong et al. 2026) uses probabilistic mixture components. Beyond residual quantization, parallel methods generate multiple codes from separate vector subspaces or token branches, including PQ (Jegou, Douze, and Schmid 2011), OPQ (Ge et al. 2013), LLaDA-Rec (Shi et al. 2025), and ACERec (Xia et al. 2026a). Moreover, hybrid tokenizers, including OneSearch (Chen et al. 2025), QARM V2 (Xia et al. 2026b), and Quantizing Intent (Choi et al. 2026), combine residual and non-residual quantizers to capture hierarchical commonality and finer distinctions. Recently, methods such as DOS (Yin et al. 2026), ReSID (Liang et al. 2026), and HHQ (Wang et al. 2026) reshape the quantization space through orthogonal or hyperspherical structures. However, these methods leave the cross-level residual unexplored. Under full-centroid subtraction, the residual can still carry a selected-centroid component into the next codebook, making later levels revisit variation represented earlier. PRQ-KMeans addresses this gap by modeling this carryover and removing it through per-representation projection.

## 3 Preliminaries and Motivation

## 3.1 RQ-KMeans for Hierarchical SIDs

In RQ-KMeans, each codeword is a K-Means centroid. Let $\mathcal { X } = \mathbf { \bar { \Phi } } \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { R } ^ { d }$ be the embeddings used to fit the tokenizer. A depth-L RQ-KMeans tokenizer learns one codebook $\mathcal { C } ^ { ( \ell ) } = \{ \mathbf { c } _ { j } ^ { ( \ell ) } \} _ { j = 1 } ^ { K _ { \ell } }$ at each level and maps entity i to the

SID $( z _ { i } ^ { ( 1 ) } , \dots , z _ { i } ^ { ( L ) } )$ . Starting from $\mathbf { r } _ { i } ^ { ( 1 ) } = \mathbf { x } _ { i } ,$ , Euclidean $\mathrm { K } _ { - }$ Means alternates between nearest-centroid assignment and centroid update:

$$
\begin{array} { r l } & { z _ { i } ^ { ( \ell ) } = \arg \operatorname* { m i n } _ { j } \left\| \mathbf { r } _ { i } ^ { ( \ell ) } - \mathbf { c } _ { j } ^ { ( \ell ) } \right\| _ { 2 } ^ { 2 } , } \\ & { \mathbf { c } _ { j } ^ { ( \ell ) } \gets \frac { 1 } { | \mathcal { Z } _ { j } ^ { ( \ell ) } | } \displaystyle \sum _ { i \in \mathcal { T } _ { j } ^ { ( \ell ) } } \mathbf { r } _ { i } ^ { ( \ell ) } . } \end{array}\tag{1}
$$

where $\mathcal { T } _ { i } ^ { ( \ell ) } = \{ i : z _ { i } ^ { ( \ell ) } = j \}$ . After fitting, $z _ { i } ^ { ( \ell ) }$ becomes the token at level ℓ, and RQ-KMeans constructs the next-level representation by subtracting the selected centroid:

$$
\mathbf { r } _ { i } ^ { ( \ell + 1 ) } = \mathbf { r } _ { i } ^ { ( \ell ) } - \mathbf { c } _ { z _ { i } ^ { ( \ell ) } } ^ { ( \ell ) } .\tag{2}
$$

Some implementations additionally apply the in-place update $\mathbf { r } _ { i } ^ { ( \ell + \bar { 1 } ) } \gets \mathrm { n o r m } \Big ( \mathbf { r } _ { i } ^ { ( \ell + 1 ) } \Big )$ before the next level. Following previous work (Chen et al. 2025, 2026), the industrial analysis below uses this normalized variant. Here, norm $\mathbf { \tau } ( \mathbf { v } ) \ \stackrel { \cdot } { = } \ \mathbf { v } / \left\| \mathbf { v } \right\| _ { 2 }$ for nonzero v. In reconstructionoriented quantization, the residual is the approximation error left after selecting a centroid; in hierarchical SID tokenization, it also determines the representation available to the next level. Under progressive commonality removal, the selected centroid summarizes information shared within its cluster, while the residual should preserve the remaining differences. We next examine whether full-centroid subtraction achieves this goal for each representation.

## 3.2 Residual Carryover after Centroid Subtraction

A centroid is shared by all representations assigned to its cluster, although these representations may contain diferent amounts of its directional component. Consider one representation–centroid pair and abbreviate it as r and $\mathbf { c } \neq \mathbf { 0 } .$ Write $\mathbf { c } ~ = ~ \rho \hat { \mathbf { c } } ,$ , where $\hat { \textbf { c } } = \mathbf { c } / \left\| \mathbf { c } \right\| _ { 2 }$ and $\rho ~ = ~ \| \mathbf { c } \| _ { 2 }$ . Let $\alpha = \mathbf { r } ^ { \top } \hat { \mathbf { c } }$ and $\mathbf { u } = \mathbf { r } - \alpha \hat { \mathbf { c } } ,$ so that u ⊥ cˆ. Then

$$
\begin{array} { c } { \mathbf { r } = \alpha \hat { \mathbf { c } } + \mathbf { u } , } \\ { \mathbf { r } - \mathbf { c } = ( \alpha - \rho ) \hat { \mathbf { c } } + \mathbf { u } . } \end{array}\tag{3}
$$

Full-centroid subtraction applies the shared coeficient $\rho ,$ whereas representation r contains the instance-specific coefficient α. When $\alpha \neq \rho ,$ the residual retains $( \alpha - \rho ) \hat { \mathbf { c } }$ along the selected centroid direction; we call this component residual carryover. Figure 1 illustrates the positive- and negativecarryover cases. An arithmetic-mean centroid matches the average coeficient within its assigned cluster, but not necessarily the coeficient of every representation. Appendix A provides the corresponding identity and derivation.

## 3.3 Empirical Motivation

We first measure residual carryover using a three-level RQ-KMeans tokenizer fitted to representations from the industrial dataset, with codebook sizes 1024, 512, and 128. Section 5.1 and Appendix C provide the setup. For representation $i ,$ define the relative carryover magnitude at level ℓ as

![](images/8a7e09a73d89f7bc02ec279083b94c0c961efe80a2d565e2d4970bbdc77de0b8.jpg)

![](images/4882995770c6a9f4621308de547ee386ca0012ecb665c1ebbdf6b9791c65e233.jpg)  
(a) Residual carryover  
(b) Controlled L2 utilization  
Figure 2: Empirical motivation on the industrial dataset. (a) Mean residual carryover ratio at each RQ-KMeans level. The dashed line marks the expected absolute cosine between two independent isotropic directions in 128 dimensions. (b) L2 prefix utilization after retaining a fraction η of the L1 carryover. L1 assignments stay fixed; L2 is refit for every η.

$$
m _ { i } ^ { ( \ell ) } = \frac { \left| \boldsymbol { \alpha } _ { i } ^ { ( \ell ) } - \boldsymbol { \rho } _ { i } ^ { ( \ell ) } \right| } { \left\| \mathbf { r } _ { i } ^ { ( \ell ) } - \mathbf { c } _ { z _ { i } ^ { ( \ell ) } } ^ { ( \ell ) } \right\| _ { 2 } } .\tag{4}
$$

Figure 2(a) reports mean ratios of 9.89%, 10.17%, and 7.44% at L1, L2, and L3, respectively. For scale, two independent isotropic directions in 128 dimensions have an expected absolute cosine of 7.07%; Appendix D.1 provides the derivation. The L1 and L2 means, corresponding to residuals passed to subsequent codebooks, exceed this reference by 2.82 and 3.10 percentage points, respectively, whereas the terminal L3 value is close to it.

We next test how this component afects the immediately following SID level. Keeping the L1 assignments fixed, we retain a fraction $\eta \in [ 0 , 1 ]$ of the L1 carryover:

$$
{ \bf r } _ { i } ^ { ( 2 ) } ( \eta ) = \mathrm { n o r m } \left( \eta \left( \alpha _ { i } ^ { ( 1 ) } - \rho _ { i } ^ { ( 1 ) } \right) \hat { { \bf c } } _ { i } ^ { ( 1 ) } + { \bf u } _ { i } ^ { ( 1 ) } \right) .\tag{5}
$$

Here, $\hat { \mathbf { c } } _ { i } ^ { ( 1 ) }$ is the direction of the selected L1 centroid. The endpoints $\eta = 1$ and $\eta = 0$ retain the full RQ-KMeans carryover and remove it completely, respectively. For each η, we refit L2 with the same Euclidean K-Means configuration and measure the fraction of the $K _ { 1 } K _ { 2 }$ possible two-token item prefixes assigned to at least one item. Appendix D.1 provides the equivalent operational update and implementation details.

Figure 2(b) shows that L2 prefix utilization increases monotonically from 47.88% at $\eta = 1$ to 48.46% at $\eta = 0 .$ Under this controlled protocol, reducing selected-centroid carryover allows the immediately following level to use a broader set of prefixes. This observation motivates constructing the next-level representation by removing the selectedcentroid component through projection.

## 4 Methodology

## 4.1 Overview

Figure 3 gives an overview of PRQ-KMeans. It first removes the global component from the input embeddings. It then learns the codebooks sequentially: at each level, Top-k soft weights refine the current centroids, hard cosine assignment emits a token, and the projection residual becomes the input used to fit the next codebook. Once the hierarchy is trained, SID encoding repeats the hard assignment and residual update using the learned global mean and codebooks during inference. Projection defines the cross-level update, while global removal and soft refinement prepare its input representations and centroids. Appendix B gives the complete algorithm.

![](images/c5f62533ea7fee4f6154718e4783d84233288d057383bac2bc9d9ce94c3da164.jpg)  
Figure 3: Overview of PRQ-KMeans. PRQ-KMeans initializes the first-level residual by removing the global-mean component. At level ℓ, Top-k soft centroid refinement fits the codebook from the current residuals. Hard cosine assignment then selects a centroid and emits token $z _ { i } ^ { ( \ell ) }$ ; projection removes the selected-centroid component before the residual proceeds to level ℓ + 1. SID encoding repeats the same hard-assignment and projection recursion using the learned codebooks.

## 4.2 Projection Residual Construction

PRQ-KMeans removes the selected-centroid component by requiring the next residual v to be orthogonal to the selected centroid c. Orthogonality alone, however, does not determine a useful residual; even the zero vector satisfies this constraint. We therefore choose the orthogonal residual that changes the current representation r as little as possible:

$$
\mathbf { v } ^ { \star } = \underset { \mathbf { v } } { \arg \operatorname* { m i n } } \ \left\| \mathbf { v } - \mathbf { r } \right\| _ { 2 } ^ { 2 } \quad \mathrm { s . t . } \quad \mathbf { v } ^ { \top } \mathbf { c } = 0 .\tag{6}
$$

Solving this problem gives

$$
\mathbf { v } ^ { \star } = \mathbf { r } - { \frac { \mathbf { r } ^ { \top } \mathbf { c } } { \left\| \mathbf { c } \right\| _ { 2 } ^ { 2 } } } \mathbf { c } .\tag{7}
$$

This solution directly removes the carryover identified in Eq. (3). Recall that $\mathbf { c } = \rho \hat { \mathbf { c } } , \alpha = \mathbf { r } ^ { \top } \hat { \mathbf { c } }$ , and $\mathbf { r } = \alpha \hat { \mathbf { c } } + \mathbf { u }$ . The component subtracted in Eq. (7) is

$$
\frac { \mathbf { r } ^ { \top } \mathbf { c } } { \left\| \mathbf { c } \right\| _ { 2 } ^ { 2 } } \mathbf { c } = \alpha \hat { \mathbf { c } } .
$$

Consequently, $\mathbf { v } ^ { \star } = \mathbf { r } - \alpha \hat { \mathbf { c } } = \mathbf { u }$ . Unlike the RQ-KMeans residual $( \alpha - \rho ) \hat { \mathbf { c } } + \mathbf { u }$ , the projection residual contains no remaining component along the selected centroid.

Proposition 1 formalizes these guarantees.

Proposition 1 (Minimum-change orthogonal residual) For any $\textbf { r } \in \ \mathbb { R } ^ { d }$ and nonzero c, the projection residual $\mathbf { v } ^ { \star }$ in $\overset { \cdot } { E q . } \ ( 7 )$ is the unique solution to Eq. (6). It satisfies $\mathbf { c } ^ { \top } \mathbf { v } ^ { \star } = 0$ and has the smallest Euclidean distance to r among all vectors orthogonal to c.

Centroid magnitude does not afect the component removed by Eq. (7). PRQ-KMeans therefore compares representations and centroids using cosine similarity, which is likewise unafected by centroid magnitude. The same cosine scores are used for Top-k soft refinement and hard assignment. The complete proofs are provided in Appendix A.

## 4.3 Global Component Removal

A component shared across the entire embedding collection can dominate the first partition. Standard K-Means uses the mean ofeach cluster as its centroid, summarizing information shared by the representations in that cluster. PRQ-KMeans applies the same idea before the first SID level: it uses the mean of all input representations to summarize information shared across the collection. We write $\bar { \mathbf { x } } _ { i } = \mathrm { n o r m } ( \mathbf { x } _ { i } )$ for the L2-normalized input embedding. Their global mean is

$$
\mu = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bar { \mathbf { x } } _ { i } .\tag{8}
$$

Before emitting any token, PRQ-KMeans removes the component along this mean and obtains the first-level representation:

$$
\mathbf { r } _ { i } ^ { ( 1 ) } = \operatorname { n o r m } \left( \bar { \mathbf { x } } _ { i } - \frac { \bar { \mathbf { x } } _ { i } ^ { \top } \pmb { \mu } } { \Vert \pmb { \mu } \Vert _ { 2 } ^ { 2 } } \pmb { \mu } \right) .\tag{9}
$$

The global mean is learned once during tokenizer fitting and reused during encoding. Equation (9) produces the representations used to fit the first codebook; subsequent levels apply the same residual construction to the selected centroid.

## 4.4 Top-k Soft Centroid Refinement

Because the residual update depends on the selected centroid, reliable centroid estimation matters. Hard K-Means uses each representation to update only its assigned centroid, ignoring its similarity to nearby centroids around a cluster boundary. During each fitting iteration, PRQ-KMeans computes the cosine scores and selects the Top-k candidate centroids:

$$
\begin{array} { c } { { s _ { i j } ^ { ( \ell ) } = \cos \left( \mathbf { r } _ { i } ^ { ( \ell ) } , \mathbf { c } _ { j } ^ { ( \ell ) } \right) , } } \\ { { \mathcal { N } _ { k } ( \mathbf { r } _ { i } ^ { ( \ell ) } ) = \mathrm { T o p K } _ { j \in \{ 1 , \dots , K _ { \ell } \} } s _ { i j } ^ { ( \ell ) } . } } \end{array}\tag{10}
$$

Within this candidate set, the soft weight is

$$
w _ { i j } ^ { ( \ell ) } = \frac { \exp \left( \beta s _ { i j } ^ { ( \ell ) } \right) } { \sum _ { t \in \mathcal { N } _ { k } ( \mathbf { r } _ { i } ^ { ( \ell ) } ) } \exp \left( \beta s _ { i t } ^ { ( \ell ) } \right) }\tag{11}
$$

for $j \in \mathcal { N } _ { k } ( \mathbf { r } _ { i } ^ { ( \ell ) } )$ , and zero otherwise, where $\beta > 0$ is the softmax concentration (inverse-temperature) parameter; larger $\beta$ produces sharper weights.

Each centroid is updated using the representations for which it is a candidate:

$$
{ \bf c } _ { j } ^ { ( \ell ) } \gets \frac { \sum _ { i } w _ { i j } ^ { ( \ell ) } { \bf r } _ { i } ^ { ( \ell ) } } { \sum _ { i } w _ { i j } ^ { ( \ell ) } } .\tag{12}
$$

Restricting the weights to the $k$ nearest centroids keeps each update local while allowing a boundary representation to contribute to nearby candidates. The score, weight, and centroid-update steps are repeated for a fixed number of fitting iterations. Soft weights are used only while refining the current codebook. After refinement, hard assignment emits the token and constructs the residual passed to the next level.

## 4.5 Sequential Codebook Fitting and SID Encoding

In the recursion, Eq. (7) is applied with ${ \bf r } = { \bf r } _ { i } ^ { ( \ell ) }$ and ${ \bf c } =$ $\mathbf { c } _ { z _ { i } ^ { ( \ell ) } } ^ { ( \ell ) }$ . After fitting $\mathcal { C } ^ { ( \ell ) }$ , PRQ-KMeans assigns each fitting representation to one centroid using hard cosine assignment:

$$
z _ { i } ^ { ( \ell ) } = \arg \operatorname* { m a x } _ { j } \cos \left( \mathbf { r } _ { i } ^ { ( \ell ) } , \mathbf { c } _ { j } ^ { ( \ell ) } \right) .\tag{13}
$$

The selected index $z _ { i } ^ { ( \ell ) }$ gives the current token. Writing the selected centroid as $\mathbf { \boldsymbol { c } } _ { i } ^ { ( \ell ) } = \mathbf { \boldsymbol { c } } _ { z _ { i } ^ { ( \ell ) } } ^ { ( \ell ) }$ , PRQ-KMeans removes its component and normalizes the result in one update:

$$
\mathbf { r } _ { i } ^ { ( \ell + 1 ) } = \operatorname { n o r m } \left( \mathbf { r } _ { i } ^ { ( \ell ) } - \frac { \mathbf { r } _ { i } ^ { ( \ell ) \top } \mathbf { c } _ { i } ^ { ( \ell ) } } { \left\| \mathbf { c } _ { i } ^ { ( \ell ) } \right\| _ { 2 } ^ { 2 } } \mathbf { c } _ { i } ^ { ( \ell ) } \right) .\tag{14}
$$

For a nonzero projection residual, the normalization in Eq. (14) is well defined; vector ${ \bf r } _ { i } ^ { ( \ell + 1 ) }$ is used to fit $\mathcal { C } ^ { ( \ell + 1 ) }$ Repeating this process learns the hierarchy level by level.

Once fitting is complete, SID encoding reuses the learned global mean and codebooks, repeating the same hard assignment, token emission, projection, and normalization steps without further updates.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. We primarily evaluate PRQ-KMeans on a publicly released industrial e-commerce search dataset from prior work (Li et al. 2026). It contains approximately 7.8 million items and 9.0 million training-query records for tokenizer fitting. Following OneSearch (Chen et al. 2025), these data form three training stages of approximately 36 million, 54 million, and 46 million examples, with separate Order and Click test sets from logged purchases and clicks. We also evaluate on the Amazon Sports, Toys, and Clothing benchmarks (He and McAuley 2016) and LastFM (Cantador, Brusilovsky, and Kuflik 2011), following prior work (Rajput et al. 2023; Wan et al. 2026). Appendix C provides the dataset details.

Baselines. We compare PRQ-KMeans with learned residual tokenizers RQ-VAE (Lee et al. 2022) and R3-VAE (Wan et al. 2026), post-hoc tokenizers RQ-KMeans (Luo et al. 2025) and RQ-GMM (Tong et al. 2026), and parallel quantizers PQ-KMeans (Jegou, Douze, and Schmid 2011) and OPQ-KMeans (Ge et al. 2013). For the five-level industrial setting, we also evaluate hybrid tokenizers that combine a hierarchical residual prefix with sufix tokens (Chen et al. 2025; Xia et al. 2026b). RQ-OPQ (Chen et al. 2025) uses two RQ-KMeans levels, a balanced K-Means third level, and two OPQ sufix tokens. ResKmeansFSQ (Xia et al. 2026b) uses three RQ-KMeans levels followed by two FSQ sufix tokens. PRQ-OPQ replaces the first two RQ-KMeans levels of RQ-OPQ with PRQ-KMeans, while retaining its balanced third-level K-Means and two OPQ sufix tokens. This construction evaluates the compatibility of PRQ-KMeans with the balanced third level and OPQ sufix used in RQ-OPQ. Appendix C provides the reproduction and alignment details.

Implementation and Evaluation. For the industrial dataset, we follow OneSearch (Chen et al. 2025), using its 128-dimensional distilled-BGE query and item representations and BART-Base retriever (Lewis et al. 2020). All tokenizers share the three-stage training, decoding, and evaluation pipeline. The three-level setting uses a 1024-512-128 codebook; the five-level setting uses the same three-level prefix followed by two 64-way sufix tokens. For the public benchmarks, we use 768-dimensional Sentence-T5-Base item embeddings (Ni et al. 2021), a 256-256-256 codebook, and the TIGER pipeline (Rajput et al. 2023; Wan et al. 2026). PRQ-KMeans uses $k = 2$ on the public benchmarks and $k = 5$ on the industrial dataset due to their diferent codebook sizes, with $\beta = 1 5$ in both settings. Following OneSearch, industrial evaluation reports HitRate@50 and MRR@50 for Order and Click, together with the independent-code ratio (ICR), prefix utilization, and used-prefix Gini for codebook quality. Public evaluation reports item-level Recall@20 and NDCG@20; Appendix C provides evaluation details. The code will be released upon acceptance.

## 5.2 Overall Performance

Industrial benchmark. We compare three-level codebook quality (Table 1). Among all evaluated methods, PRQ-

<table><tr><td rowspan="3">Method</td><td colspan="4">Downstream Evaluation</td><td colspan="7">Codebook Quality</td></tr><tr><td colspan="2">Order</td><td colspan="2">Click</td><td>Full SID</td><td colspan="3">Utilization (%)</td><td colspan="3">Gini</td></tr><tr><td>HitRate MRR</td><td></td><td>HitRate</td><td>MRR</td><td>ICR (%)</td><td>L1</td><td>L2</td><td>L3</td><td>L1</td><td>L2</td><td>L3</td></tr><tr><td colspan="10">Codebook: 1024–512–128</td></tr><tr><td>RQ-VAE</td><td>0.2474</td><td>0.0501</td><td>0.2921</td><td>0.0446</td><td>54.24</td><td>100.00</td><td>47.55</td><td>3.22</td><td>0.407</td><td>0.795</td><td>0.601</td></tr><tr><td>R3-VAE</td><td>0.2289</td><td>0.0449</td><td>0.2788</td><td>0.0402</td><td>54.96</td><td>99.41</td><td>51.04</td><td>3.34</td><td>0.427</td><td>0.783</td><td>0.592</td></tr><tr><td>RQ-KMeans</td><td>0.2913</td><td>0.0644</td><td>0.3279</td><td>0.0540</td><td>55.52</td><td>100.00</td><td>47.88</td><td>3.61</td><td>0.413</td><td>0.774</td><td>0.569</td></tr><tr><td>RQ-GMM</td><td>0.2687</td><td>0.0594</td><td>0.3071</td><td>0.0483</td><td>50.66</td><td>99.90</td><td>38.57</td><td>3.06</td><td>0.412</td><td>0.783</td><td>0.598</td></tr><tr><td>PQ-KMeans</td><td>0.0400</td><td>0.0061</td><td>0.0521</td><td>0.0054</td><td>50.45</td><td>100.00</td><td>22.52</td><td>0.67</td><td>0.351</td><td>0.934</td><td>0.891</td></tr><tr><td>OPQ-KMeans</td><td>0.1242</td><td>0.0225</td><td>0.1393</td><td>0.0164</td><td>47.35</td><td>100.00</td><td>21.85</td><td>1.15</td><td>0.276</td><td>0.916</td><td>0.804</td></tr><tr><td>PRQ-KMeans</td><td>0.3128</td><td>0.0720</td><td>0.3488</td><td>0.0588</td><td>58.45</td><td>100.00</td><td>57.78</td><td>3.99</td><td>0.298</td><td>0.758</td><td>0.546</td></tr><tr><td colspan="10">Codebook: 1024–512-128–64–64</td></tr><tr><td>ResKmeansFSQ</td><td>0.4038</td><td>0.1194</td><td>0.4364</td><td>0.0969</td><td>89.19</td><td>100.00</td><td>47.88</td><td>3.61</td><td>0.413</td><td>0.774</td><td>0.569</td></tr><tr><td>RQ-OPQ</td><td>0.4276</td><td>0.1340</td><td>0.4650</td><td>0.1083</td><td>98.99</td><td>100.00</td><td>47.88</td><td>8.00</td><td>0.413</td><td>0.774</td><td>0.266</td></tr><tr><td>PRQ-OPQ</td><td>0.4359</td><td>0.1364</td><td>0.4713</td><td>0.1089</td><td>98.74</td><td>100.00</td><td>57.78</td><td>8.42</td><td>0.298</td><td>0.758</td><td>0.241</td></tr></table>

Table 1: Codebook quality and downstream generative retrieval performance on the industrial dataset. ICR denotes the independent-code ratio, the percentage of occupied full SIDs assigned to exactly one item. For each codebook block, bes and second-best values are bold and underlined, respectively; lower Gini is better.

<table><tr><td rowspan="2">Method</td><td colspan="2">Sports</td><td colspan="2">Toys</td><td colspan="2">Clothing</td><td colspan="2">LastFM</td></tr><tr><td>Recall</td><td>NDCG</td><td>Recall</td><td>NDCG</td><td>Recall</td><td>NDCG</td><td>Recall</td><td>NDCG</td></tr><tr><td>RQ-VAE</td><td>0.0522</td><td>0.0219</td><td>0.0751</td><td>0.0321</td><td>0.0357</td><td>0.0141</td><td>0.0082</td><td>0.0033</td></tr><tr><td>R3-VAE</td><td>0.0514</td><td>0.0211</td><td>0.0812</td><td>0.0354</td><td>0.0361</td><td>0.0147</td><td>0.0074</td><td>0.0029</td></tr><tr><td>RQ-KMeans</td><td>0.0528</td><td>0.0215</td><td>0.0824</td><td>0.0346</td><td>0.0330</td><td>0.0132</td><td>0.0115</td><td>0.0042</td></tr><tr><td>RQ-GMM</td><td>0.0506</td><td>0.0209</td><td>0.0711</td><td>0.0287</td><td>0.0341</td><td>0.0133</td><td>0.0082</td><td>0.0024</td></tr><tr><td>PQ-KMeans</td><td>0.0087</td><td>0.0033</td><td>0.0177</td><td>0.0080</td><td>0.0082</td><td>0.0034</td><td>0.0016</td><td>0.0007</td></tr><tr><td>OPQ-KMeans</td><td>0.0379</td><td>0.0151</td><td>0.0403</td><td>0.0171</td><td>0.0233</td><td>0.0087</td><td>0.0033</td><td>0.0014</td></tr><tr><td>PRQ-KMeans</td><td>0.0531</td><td>0.0219</td><td>0.0851</td><td>0.0361</td><td>0.0382</td><td>0.0151</td><td>0.0179</td><td>0.0076</td></tr></table>

Table 2: Performance evaluation on the public benchmarks.

KMeans obtains the highest ICR and L2–L3 prefix utilization together with the lowest L2–L3 Gini. Relative to RQ-KMeans, ICR increases from 55.52% to 58.45%, L2–L3 utilization rises from 47.88%/3.61% to 57.78%/3.99%, and L2–L3 Gini decreases from 0.774/0.569 to 0.758/0.546.

PRQ-KMeans also improves downstream generative retrieval. RQ-KMeans is the strongest three-level baseline on all four metrics. Relative to RQ-KMeans, PRQ-KMeans improves Order HitRate and MRR by 7.4% and 11.8%, respectively, and Click HitRate and MRR by 6.4% and 8.9%. In the five-level hybrid setting, PRQ-OPQ achieves the best Order and Click results on all four downstream metrics among the three aligned tokenizers and improves over RQ-OPQ by 0.6%–1.9%. This result supports the compatibility of the PRQ-KMeans prefix with an OPQ sufix under the same industrial tokenizer interface.

Public benchmarks. Across four benchmarks, PRQ-KMeans obtains the best or tied-best value on all eight metrics and exceeds RQ-KMeans in every comparison (Table 2). The gains are modest on Sports and Toys, larger on Clothing, and most pronounced on LastFM, where Recall increases from 0.0115 to 0.0179 and NDCG from 0.0042 to 0.0076 relative to RQ-KMeans. These results extend the evidence beyond the industrial setting to multiple public domains.

<table><tr><td>Variant</td><td colspan="2">Order</td><td colspan="2">Click</td></tr><tr><td></td><td>HitRate</td><td>MRR</td><td>HitRate</td><td>MRR</td></tr><tr><td>PRQ-KMeans</td><td>0.3128</td><td>0.0720</td><td>0.3488</td><td>0.0588</td></tr><tr><td>Global</td><td>0.3017</td><td>0.0697</td><td>0.3379</td><td>0.0560</td></tr><tr><td>Soft</td><td>0.3067</td><td>0.0705</td><td>0.3460</td><td>0.0584</td></tr><tr><td>Projection</td><td>0.2985</td><td>0.0687</td><td>0.3328</td><td>0.0548</td></tr></table>

Table 3: Ablation study on the industrial dataset.

## 5.3 Ablation Study

Tables 3 and A1 (in Appendix) report the industrial and public ablations, respectively. We compare the complete model with three variants: − Global omits global component removal; − Soft replaces Top-k soft refinement with hard centroid updates; and − Projection retains Global and Top-k refinement but replaces cosine fitting and assignment with Euclidean counterparts and projection residualization with full-centroid subtraction. The complete PRQ-KMeans model achieves the best result on every industrial metric and the best or tied-best result on all eight public metrics, tying − Global on Clothing NDCG. On the industrial dataset, removing Projection produces the largest decrease on all four metrics, while removing Global or Soft also yields consistent drops. On the public benchmarks, removing Projection produces the largest decrease on all six Amazon metrics, while LastFM is most sensitive to Soft. Removing Global lowers seven of eight metrics and ties the complete model on Clothing NDCG.

![](images/e07420570e29ecf17d363e22bf05efe563745d6f92a6794f2856384ca88d6ab9.jpg)  
(a) Top-k

![](images/9f5dc463382e26d07a43ce078325a4d17509f240a8abe58470968369852d2a40.jpg)  
(b) Weight concentration

Figure 4: Hyperparameter sensitivity on the industrial dataset.  
![](images/bd3919921ef5d4637bf2b0ec42128f3475d4c7da6f1d3ca44f267ce633bd6d8b.jpg)

![](images/f5850527aff1c8045b280977f5da84087433fae8ab6c12867f35a83eaa82b298.jpg)  
(a) HitRate  
(b) MRR  
Figure 5: Embedding robustness on the industrial dataset. The aligned BGE representations used in the main experiment are replaced with aligned Qwen3 representations, while the codebook sizes and downstream protocol remain unchanged. KM denotes K-Means.

## 5.4 Sensitivity and Robustness

Hyperparameter sensitivity. We study sensitivity to k, the number of nearby centroids receiving weighted updates, and $\beta ,$ which controls weight concentration. Figure 4 shows that $k = 2$ and $k = 5$ perform better than $k = 1$ and $k = 1 0$ This suggests a trade-of: a small neighborhood limits sharing near cluster boundaries, while larger neighborhoods may weaken locality by updating less similar centroids. For $\beta ,$ $\beta = 1 0$ performs worse, while results remain comparable across $\beta = 1 5 – 2 5$ . This suggests that difuse weights may obscure similarity diferences among the centroids, whereas sharper weights better prioritize the closest candidates. Appendix D.3 provides results on the public benchmarks.

Embedding robustness. To evaluate robustness across input embeddings, we replace the aligned BGE representations used in the main experiment with aligned Qwen3- Embedding-0.6B representations (Zhang et al. 2025b). We rerun five tokenizers using these 128-dimensional representations, with the alignment procedure, codebook sizes, and downstream protocol unchanged. Figure 5 shows that PRQ-KMeans achieves the highest HitRate and MRR on both Order and Click under the alternative embedding, supporting its robustness to diferent input embeddings.

![](images/5742da372b6adcbc205c47ade22b15ca4d37013c610f94b9a470d108228aa567.jpg)

![](images/221ff50a01e4c6bc7cebec0c9a22d8b638c74b150bca1e738ab37053ce6f874e.jpg)  
Figure 6: Centroid visualization on the industrial dataset. L1– L3 centroids from RQ-KMeans and PRQ-KMeans arejointly embedded using one t-SNE fit; both panels share coordinates and axis limits, and colors and markers denote SID levels.

## 5.5 Qualitative Analysis

Centroid visualization. To visualize how residual construction shapes the learned codebooks, we use the L1–L3 centroids of RQ-KMeans and PRQ-KMeans and jointly embed them in two dimensions with t-SNE (Van der Maaten and Hinton 2008). In RQ-KMeans, L1 centroids span a broad region, whereas L2 and especially L3 centroids form a dense core. This pattern is consistent with residual carryover: when the selected-centroid component remains, later codebooks receive representations that still vary along earlier centroid directions and may revisit previously represented variation. In contrast, PRQ-KMeans removes this component before fitting the next codebook, and its L2 and L3 centroids cover a broader region. Together with the higher later-level utilization and lower Gini in Table 1, this visualization provides qualitative support for progressive commonality removal.

Case study. We trace 12 Kitty-themed slippers assigned to one RQ-KMeans full SID. Under PRQ-KMeans, these items retain one shared level-1 prefix, separate into two level-2 prefixes, and form three full SIDs containing four items each. In this group, the later tokens distinguish observed indoor soft-soled, outdoor/dormitory, and mixed-use variants while retaining the shared product theme. Appendix D.4 reports the exact collision-group statistics and selection procedure; it also presents a case study of the weak-intent query “hotel.”

## 6 Conclusion

This paper presents PRQ-KMeans, a post-hoc hierarchical SID tokenizer based on progressive commonality removal. It removes the global component before the first level and refines each codebook with Top-k soft weights. After assignment, it removes the selected-centroid component through projection before fitting the next codebook. PRQ-KMeans achieves the best overall downstream retrieval performance on the industrial dataset and four public benchmarks. Component ablations support the complete design. Together, these results support explicitly controlling what is passed between successive SID levels.

## References

Cantador, I.; Brusilovsky, P.; and Kuflik, T. 2011. Second workshop on information heterogeneity and fusion in recommender systems (HetRec2011). In Proceedings of the fifth ACM conference on Recommender systems, 387–388.

Chen, B.; Guo, X.; Wang, S.; Liang, Z.; Lv, Y.; Ma, Y.; Xiao, X.; Xue, B.; Zhang, X.; Yang, Y.; et al. 2025. Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search. arXiv preprint arXiv:2509.03236.

Chen, B.; Wang, S.; Ma, Y.; Liang, Z.; Zhang, X.; Lv, Y.; Yang, Y.; Dai, H.; Mao, L.; Zhao, T.; et al. 2026. OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework. arXiv preprint arXiv:2603.24422.

Choi, J.; Ye, H.; Ding, Z.; Long, B.; Zelditch, B.; and Vats, A. 2026. Quantizing Intent: Cross-Domain Semantic IDs from Organic Activity for Industrial Ranking. arXiv preprint arXiv:2606.01396.

Deng, J.; Wang, S.; Cai, K.; Ren, L.; Hu, Q.; Ding, W.; Luo, Q.; and Zhou, G. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965.

Fu, J.; Ge, X.; Karatzoglou, A.; Arapakis, I.; Verberne, S.; Jose, J. M.; and Ren, Z. 2026. Diferentiable Semantic ID for Generative Recommendation. arXiv preprint arXiv:2601.19711.

Ge, T.; He, K.; Ke, Q.; and Sun, J. 2013. Optimized product quantization for approximate nearest neighbor search. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2946–2953.

He, R.; and McAuley, J. 2016. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web, 507–517.

Jegou, H.; Douze, M.; and Schmid, C. 2011. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis & Machine Intelligence, 33(01): 117–128.

Jiang, J.; Zhang, X.; Zhang, E.; Xiong, Y.; Zhang, J.; Wang, J.; Yu, H.; Wang, Y.; Wang, H.; Yan, X.; et al. 2026. End-to-End Semantic ID Generation for Generative Advertisement Recommendation. arXiv preprint arXiv:2602.10445.

Lee, D.; Kim, C.; Kim, S.; Cho, M.; and Han, W.-S. 2022. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11523–11532.

Lee, S.; Choi, M.; and Lee, J. 2023. GLEN: Generative retrieval via lexical index learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 7693–7704.

Lewis, M.; Liu, Y.; Goyal, N.; Ghazvininejad, M.; Mohamed, A.; Levy, O.; Stoyanov, V.; and Zettlemoyer, L. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th annual meeting of the association for computational linguistics, 7871–7880.

Li, Y.; Chen, B.; Cheng, M.; Liu, Z.; Zhang, X.; Lei, C.; and Ou, W. 2026. KuaiSearch: A Large-Scale E-Commerce Search Dataset for Recall, Ranking, and Relevance. arXiv preprint arXiv:2602.11518.

Liang, Y.; Zhang, Z.; Zhu, Y.; Zhang, K.; Guo, Z.; Zhou, W.; Yang, Z.; Wu, K.; Ni, Y.; Zeng, A.; et al. 2026. Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs. arXiv preprint arXiv:2602.02338.

Liu, E.; Zheng, B.; Ling, C.; Hu, L.; Li, H.; and Zhao, W. X. 2024. Generative recommender with end-to-end learnable item tokenization. arXiv preprint arXiv:2409.05546.

Luo, X.; Cao, J.; Sun, T.; Yu, J.; Huang, R.; Yuan, W.; Lin, H.; Zheng, Y.; Wang, S.; Hu, Q.; et al. 2025. Qarm: Quantitative alignment multi-modal recommendation at kuaishou. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 5915–5922.

Ni, J.; Abrego, G. H.; Constant, N.; Ma, J.; Hall, K. B.; Cer, D.; and Yang, Y. 2021. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. arXiv preprint arXiv:2108.08877.

Rajput, S.; Mehta, N.; Singh, A.; Keshavan, R. H.; Vu, T.; Heldt, L.; Hong, L.; Tay, Y.; Tran, V. Q.; Samost, J.; et al. 2023. Recommender systems with generative retrieval. In Thirty-seventh Conference on Neural Information Processing Systems.

Shi, T.; Shen, C.; Yu, W.; Nie, S.; Li, C.; Zhang, X.; He, M.; Han, Y.; and Xu, J. 2025. LLaDA-Rec: Discrete Difusion for Parallel Semantic ID Generation in Generative Recommendation. arXiv preprint arXiv:2511.06254.

Singh, A.; Vu, T.; Mehta, N.; Keshavan, R.; Sathiamoorthy, M.; Zheng, Y.; Hong, L.; Heldt, L.; Wei, L.; Tandon, D.; et al. 2024. Better generalization with semantic ids: A case study in ranking for recommendations. In Proceedings ofthe 18th ACM Conference on Recommender Systems, 1039–1044.

Sun, W.; Yan, L.; Chen, Z.; Wang, S.; Zhu, H.; Ren, P.; Chen, Z.; Yin, D.; Rijke, M.; and Ren, Z. 2023. Learning to tokenize for generative retrieval. Advances in Neural Information Processing Systems, 36: 46345–46361.

Tay, Y.; Tran, V.; Dehghani, M.; Ni, J.; Bahri, D.; Mehta, H.; Qin, Z.; Hui, K.; Zhao, Z.; Gupta, J.; et al. 2022. Transformer Memory as a Diferentiable Search Index. Advances in Neural Information Processing Systems, 35: 21831–21843.

Tong, Z.; Liu, J.; Zhang, W.; Ruan, H.; Tang, D.; Zeng, Z.; Zeng, Q.; Zhang, P.; Lu, T.; and Gu, N. 2026. RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction. arXiv preprint arXiv:2602.12593.

Van der Maaten, L.; and Hinton, G. 2008. Visualizing data using t-SNE. Journal of machine learning research, 9(11).

Wan, Q.; Yang, Z.; Yang, D.; Fan, Y.; Yan, X.; Liu, S.; Liu, Y.; Zhang, C.; Xu, W.; Qin, J.; et al. 2026. R3-VAE: Reference Vector-Guided Rating Residual Quantization VAE for Generative Recommendation. arXiv preprint arXiv:2604.11440.

Wang, W.; Bao, H.; Lin, X.; Zhang, J.; Li, Y.; Feng, F.; Ng, S.-K.; and Chua, T.-S. 2024. Learnable item tokenization

for generative recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, 2400–2409.

Wang, Y.; Hou, Y.; Wang, H.; Miao, Z.; Wu, S.; Chen, Q.; Xia, Y.; Chi, C.; Zhao, G.; Liu, Z.; et al. 2022. A neural corpus indexer for document retrieval. Advances in Neural Information Processing Systems, 35: 25600–25614.

Wang, Y.; Wu, B.; Su, Y.; Zhang, T.; Du, Y.; Yu, L.; Guo, J.; and Cheng, X. 2026. Distilling Large Embeddings via Hyperspherical Householder Quantization. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 10562–10576.

Xia, M.; Zhou, Z.; Ma, G.; and Huang, D. 2026a. Unleash the Potential of Long Semantic IDs for Generative Recommendation. arXiv preprint arXiv:2602.13573.

Xia, T.; Zhang, J.; Liu, Y.; Dou, H.; Yin, T.; Cao, J.; Liang, X.; Xie, T.; Liu, L.; Chen, X.; et al. 2026b. QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling. arXiv preprint arXiv:2602.08559.

Yin, J.; Kou, S.; Li, C.; Wang, S.; Huang, Y.; Wei, X.; Zhu, Y.; Wang, H.; and Wang, X. 2026. DOS: Dual-Flow Orthogonal Semantic IDs for Recommendation in Meituan. In Proceedings ofthe ACM Web Conference 2026, 8305–8308.

Zhang, F.; Liu, X.; Jia, X.; Zhang, Y.; Zhang, S.; Li, X.; Zhuang, F.; Lin, W.; and Zhang, Z. 2025a. Multi-level Relevance Document Identifier Learning for Generative Retrieval. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 10066–10080.

Zhang, Q.; Szymanski, L.; Zhang, H.; and Deng, J. D. 2026. How Reliable Are Semantic-ID Tokenizer Comparisons in Generative Recommendation? arXiv preprint arXiv:2605.25330.

Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.; Xie, P.; Yang, A.; Liu, D.; Lin, J.; et al. 2025b. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Zhu, J.; Jin, M.; Liu, Q.; Qiu, Z.; Dong, Z.; and Li, X. 2024. Cost: Contrastive quantization based semantic tokenization for generative recommendation. In Proceedings of the 18th ACM Conference on Recommender Systems, 969–974.

## A Theoretical Proofs

We first derive how representation-level carryover behaves within a cluster, including the case in which the centroid is the arithmetic mean of its assigned representations. We then prove the geometric properties of the projection residual used by PRQ-KMeans.

## A.1 Residual-Carryover Identity

Consider a nonzero centroid $\mathbf { c } ,$ let $\rho = \| \mathbf { c } \| _ { 2 }$ and $\hat { \mathbf { c } } = \mathbf { c } / \rho ,$ and fix a nonempty assigned set $\mathcal { T } _ { j }$ . For each $i \in \mathcal { T } _ { j }$ , write $\mathbf { r } _ { i } = \alpha _ { i } \hat { \mathbf { c } } + \mathbf { u } _ { i }$ , where $\alpha _ { i } = \mathbf { r } _ { i } ^ { \top } \hat { \mathbf { c } }$ and $\mathbf { u } _ { i } ~ \perp ~ \hat { \mathbf { c } } .$ , and let $\bar { \alpha } _ { j } = | \mathcal { T } _ { j } | ^ { - 1 } \sum _ { i \in \mathcal { T } _ { j } } \alpha _ { i }$ . Because the carryover coeficient is $\alpha _ { i } - \rho ,$ , its mean squared magnitude satisfies

$$
\frac { 1 } { | \mathcal { T } _ { j } | } \sum _ { i \in \mathcal { I } _ { j } } ( \alpha _ { i } - \rho ) ^ { 2 } = \mathrm { V a r } _ { \mathbb { Z } _ { j } } ( \alpha ) + ( \bar { \alpha } _ { j } - \rho ) ^ { 2 } .\tag{15}
$$

Here, Var $\begin{array} { r } { \tau _ { j } ( \alpha ) ~ = ~ | \mathcal { T } _ { j } | ^ { - 1 } \sum _ { i \in \mathbb { Z } _ { i } } ( \alpha _ { i } - \bar { \alpha } _ { j } ) ^ { 2 } } \end{array}$ denotes the population variance within the assigned cluster. When c is the arithmetic mean of the assigned representations, $\bar { \alpha } _ { j } = \hat { \mathbf { c } } ^ { \top } \mathbf { c } = \rho .$ . The mean squared carryover then equals the within-cluster variance of $\alpha _ { i }$ and becomes zero only when all assigned representations have the same coeficient along the centroid direction. For a non-mean centroid, the second term in Eq. (15) captures the additional ofset.

## A.2 Proof of Proposition 1

The identity above characterizes why full-centroid subtraction can retain a component along the selected centroid. We now prove that the projection residual removes this component with the minimum Euclidean change, as stated in Proposition 1. Let $\mathbf { c } \in \mathbb { R } ^ { d }$ be nonzero and define

$$
\mathbf { v } ^ { \star } = \mathbf { r } - { \frac { \mathbf { r } ^ { \top } \mathbf { c } } { \left\| \mathbf { c } \right\| _ { 2 } ^ { 2 } } } \mathbf { c } .\tag{16}
$$

Exact orthogonality. By direct expansion,

$$
\mathbf { c } ^ { \top } \mathbf { v } ^ { \star } = \mathbf { c } ^ { \top } \mathbf { r } - { \frac { \mathbf { r } ^ { \top } \mathbf { c } } { \left\| \mathbf { c } \right\| _ { 2 } ^ { 2 } } } \left\| \mathbf { c } \right\| _ { 2 } ^ { 2 } = 0 .\tag{17}
$$

If $\mathbf { v } ^ { \star } \neq \mathbf { 0 } ,$ normalization multiplies it by a positive scalar and therefore preserves orthogonality.

Minimum Euclidean change. The feasible set ${ \mathcal { S } } = \{ { \bf { v } } : { \bf { \sigma } } \}$ $\mathbf { v } ^ { \top } \mathbf { c } = 0 \}$ is a closed linear subspace. Because $\mathbf { v } ^ { \star } \in S$ and $\mathbf { v } ^ { \star } - \mathbf { r } = - ( \mathbf { r } ^ { \top } \mathbf { c } / \left\| \mathbf { c } \right\| _ { 2 } ^ { 2 } ) \mathbf { c } \in \mathrm { s p a n } ( \mathbf { c } ) = \mathcal { S } ^ { \perp }$ , any feasible $\mathbf { v } \in S$ satisfies

$$
\left\| \mathbf { v } - \mathbf { r } \right\| _ { 2 } ^ { 2 } = \left\| \mathbf { v } - \mathbf { v } ^ { \star } \right\| _ { 2 } ^ { 2 } + \left\| \mathbf { v } ^ { \star } - \mathbf { r } \right\| _ { 2 } ^ { 2 } .\tag{18}
$$

The second term is fixed and the first is nonnegative, with equality only when $\mathbf { v } = \mathbf { v } ^ { \star }$ . Thus $\mathbf { v } ^ { \star }$ is the unique solution to Eq. (6). This minimum-change property applies to the projection residual before normalization.

## B Algorithm Details

Algorithm 1 presents the complete fitting and encoding procedure. The fitting stage learns the global mean and levelwise codebooks from the input embeddings. The encoding stage freezes these parameters and generates a hierarchical SID for each embedding.

Algorithm 1 PRQ-KMeans fitting and SID encoding   
Input: embeddings $\begin{array} { r } { \{ { \bf x } _ { i } \} _ { i = 1 } ^ { N } , } \end{array}$ , depth $L ,$ codebook sizes $\{ K _ { \ell } \} _ { \ell = 1 } ^ { L } ,$   
neighborhood size $k ,$ concentration $\beta ,$ and the number ofrefinement   
iterations   
Output: global mean $\mu ,$ codebooks $\{ \mathcal { C } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ , and hierarchical   
SIDs   
Tokenizer Fitting   
Global Component Removal   
$\bar { \mathbf { x } } _ { i } \gets \mathrm { n o r m } ( \mathbf { x } _ { i } )$ for all i   
$\pmb { \mu }  N ^ { - 1 } \dot { \sum _ { i } \bar { \mathbf { x } } _ { i } }$   
$\mathbf { r } _ { i } ^ { ( 1 ) } \gets$ norm $\left( \bar { \mathbf { x } } _ { i } - ( \bar { \mathbf { x } } _ { i } ^ { \top } \pmb { \mu } / \| \pmb { \mu } \| _ { 2 } ^ { 2 } ) \pmb { \mu } \right)$ for all i   
for $\ell = 1 , \dots , L$ do   
Sample $K _ { \ell }$ current residuals to initialize the centroids   
$T o p { \bar { - } } k$ Soft Centroid Refinement   
for each prescribed refinement iteration do   
$s _ { i j } ^ { ( \ell ) } \stackrel { \cdot } {  } \cos ( \mathbf { r } _ { i } ^ { ( \ell ) } , \mathbf { c } _ { j } ^ { ( \ell ) } )$   
$\mathcal { N } _ { k , i } ^ { ( \ell ) } \gets \mathrm { T o p K } _ { j } s _ { i j } ^ { ( \ell ) }$   
$w _ { i j } ^ { ( \ell ) } \gets \frac { \exp ( \beta s _ { i j } ^ { ( \ell ) } ) } { \sum _ { t \in \mathcal { N } _ { k , i } ^ { ( \ell ) } } \exp ( \beta s _ { i t } ^ { ( \ell ) } ) }$ for $j \in \mathcal { N } _ { k , i } ^ { ( \ell ) }$ ; set it to zero   
otherwise   
$\mathbf { c } _ { j } ^ { ( \ell ) } \gets \frac { \sum _ { i } w _ { i j } ^ { ( \ell ) } \mathbf { r } _ { i } ^ { ( \ell ) } } { \sum _ { i } w _ { i j } ^ { ( \ell ) } }$   
end for   
Hard Assignment   
$z _ { i } ^ { ( \ell ) } \gets \arg \operatorname* { m a x } _ { j } \cos ( \mathbf { r } _ { i } ^ { ( \ell ) } , \mathbf { c } _ { j } ^ { ( \ell ) } )$   
$\mathbf { c } _ { i } ^ { ( \ell ) } \gets \mathbf { c } _ { z _ { i } ^ { ( \ell ) } } ^ { ( \ell ) }$   
if $\ell < L$ then   
Projection Residual   
$\mathbf { r } _ { i } ^ { ( \ell + 1 ) } \gets \mathrm { n o r m } \left( \mathbf { r } _ { i } ^ { ( \ell ) } - \frac { \mathbf { r } _ { i } ^ { ( \ell ) \top } \mathbf { c } _ { i } ^ { ( \ell ) } } { \left\| \mathbf { c } _ { i } ^ { ( \ell ) } \right\| _ { 2 } ^ { 2 } } \mathbf { c } _ { i } ^ { ( \ell ) } \right)$   
end if   
end for   
SID Encoding   
for each input embedding $\mathbf { x } _ { i }$ do   
$\bar { \mathbf { x } } _ { i }  \mathrm { n o r m } ( \mathbf { x } _ { i } )$   
$\mathbf { r } _ { i } ^ { ( 1 ) } \gets \mathrm { n o r m } \big ( \bar { \mathbf { x } } _ { i } - ( \bar { \mathbf { x } } _ { i } ^ { \top } { \pmb { \mu } } / \| { \pmb { \mu } } \| _ { 2 } ^ { 2 } ) { \pmb { \mu } } \big )$   
$\mathbf { f o r } \ell = 1 , \ldots , \lfloor$ do   
$z _ { i } ^ { ( \ell ) } \gets \arg \operatorname* { m a x } _ { j } \cos ( \mathbf { r } _ { i } ^ { ( \ell ) } , \mathbf { c } _ { j } ^ { ( \ell ) } )$   
$\mathbf { c } _ { i } ^ { ( \ell ) } \gets \mathbf { c } _ { z _ { i } ^ { ( \ell ) } } ^ { ( \ell ) }$   
if $\ell < L$ <sup>i</sup>then   
$\begin{array} { r l } & { \mathbf { r } _ { i } ^ { ( \ell + 1 ) } \gets \mathrm { n o r m } \left( \mathbf { r } _ { i } ^ { ( \ell ) } - \frac { \mathbf { r } _ { i } ^ { ( \ell ) \top } \mathbf { c } _ { i } ^ { ( \ell ) } } { \left\| \mathbf { c } _ { i } ^ { ( \ell ) } \right\| _ { 2 } ^ { 2 } } \mathbf { c } _ { i } ^ { ( \ell ) } \right) } \end{array}$   
end if   
end for   
Store $\stackrel { \bullet \bf { * } } { ( { z } _ { i } ^ { ( 1 ) } , \ldots , { z } _ { i } ^ { ( L ) } ) }$ as the SID of $\mathbf { x } _ { i }$   
end for   
return $\mu , \{ \mathcal { C } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ , and all stored SIDs

## C Experimental Details

This section describes the public and industrial benchmark protocols, the baseline configurations, and the metrics and reproducibility settings used throughout the experiments.

## C.1 Public Benchmark Setup

The public evaluation uses the GenRec data processing and TIGER encoder–decoder pipeline (Rajput et al. 2023; Wan et al. 2026) on Sports, Toys, Clothing, and LastFM. Tokenizers are fitted on 768-dimensional Sentence-T5-Base item representations (Ni et al. 2021), and every tokenizer produces a three-level SID with a 256-256-256 codebook. PRQ-KMeans uses $k = 2$ and $\beta = 1 5$

All SIDs are evaluated with the same TIGER encoder– decoder, which has four encoder layers, four decoder layers, model dimension 128, feed-forward dimension 1024, and six attention heads. Training runs for at most 200 epochs with learning rate $1 0 ^ { - 4 }$ , zero weight decay, training batch size 256, and inference batch size 96. The best validation checkpoint is selected with patience 10, and decoding uses beam size 30. The public results use item-level Recall and NDCG. While some previous work reports SID-level metrics, we use item-level Recall and NDCG to account for SID collisions (Zhang et al. 2026). Specifically, for each predicted SID with multiple mapped items, the evaluator randomly selects one representative item with fixed seed $^ { 4 2 ; }$ duplicate item predictions are removed before metric computation.

## C.2 Industrial Benchmark Setup

The industrial benchmark follows the OneSearch training and evaluation framework (Chen et al. 2025). Tokenizer fitting uses 7,797,542 items and 9,046,403 training-query records, for 16,843,945 representations in total. The 128-dimensional representations encode query text, item titles, prices, keywords, and OCR-derived text with a distilled BGE encoder aligned by query–query, item–item, and query–item objectives.

The three- and five-level settings use 1024-512-128 and 1024-512-128-64-64 SIDs, respectively. PRQ-KMeans uses $k = 5$ and $\beta = 1 5 .$ All tokenizers share BART-Base (Lewis et al. 2020) and the same three training stages: semantic content alignment with 35,689,588 examples, co-occurrence synchronization with 54,459,755 examples, and user personalization modeling with $4 5 , 9 1 6 , 4 2 7$ examples. Training uses batch size 512 and learning rate $5 \times 1 0 ^ { - 5 }$ ; the three stages run for 4, 3, and 6 epochs, respectively.

Following OneSearch, we construct separate Click and Order test sets from search logs, each containing 30,000 query–item pairs with the corresponding observed behavior. Beam search returns 128 SIDs. For each SID, mapped items are ranked by availability and a composite score derived from historical click-through rate, conversion rate, click count, and order count, and the top five items are retained. The resulting item lists are expanded in beam order, stably deduplicated, and truncated to 50 items. We report HitRate and MRR under the cutof defined in Section 5.1.

## C.3 Baseline Configurations

The public and three-level industrial comparisons include RQ-VAE, R3-VAE, RQ-KMeans, RQ-GMM, PQ-KMeans, and OPQ-KMeans. In the five-level industrial comparison, RQ-OPQ uses two RQ-KMeans levels, a balanced K-Means third level, and two OPQ sufix tokens. ResKmeansFSQ uses three RQ-KMeans levels and maps its L3 residual with a joint 12-bit FSQ code split into two 64-way sufix tokens. PRQ-OPQ replaces the first two RQ-KMeans levels of RQ-OPQ with PRQ-KMeans while retaining its balanced third-level K-Means and two OPQ sufix tokens.

For a fair comparison, each tokenizer uses its prescribed fitting or training schedule. RQ-VAE and R3-VAE are trained for 20 epochs with batch size 4,096 and learning rate $5 \times 1 0 ^ { - 4 }$ . RQ-KMeans and PQ-KMeans use 25 clustering iterations, OPQ-KMeans uses 25 alternating rotation and clustering iterations, and RQ-GMM uses at most 30 EM iterations with variance floor $1 \bar { 0 } ^ { - 6 }$ . PRQ-KMeans uses 25 soft centroid-refinement iterations.

## C.4 Metrics and Reproducibility

Let S be the set of distinct full SIDs assigned to items and let $n ( s )$ be the number of items assigned to $s \in \mathcal { S }$ . The independent-code ratio is

$$
\mathrm { I C R } = \frac { | \{ s \in \mathcal { S } : n ( s ) = 1 \} | } { | \mathcal { S } | } .\tag{19}
$$

Its denominator is the number of distinct SIDs rather than the number of items. For level $\ell ,$ let $\mathcal { P } _ { \ell }$ be the set of occupied length-ℓ SID prefixes. Cumulative prefix utilization is

$$
\mathrm { U t i l } _ { \ell } = \frac { \left| \mathcal { P } _ { \ell } \right| } { \prod _ { t = 1 } ^ { \ell } K _ { t } } .\tag{20}
$$

Used-prefix Gini is computed from the item frequencies of prefixes in $\mathcal { P } _ { \ell }$ only. If $M _ { \ell } = | \mathcal { P } _ { \ell } |$ and $f _ { ( 1 ) } \leq \cdots \leq f _ { ( M _ { \ell } ) }$ are their sorted item frequencies, then

$$
\mathrm { G i n i } _ { \ell } = \frac { 2 \sum _ { q = 1 } ^ { M _ { \ell } } q f _ { ( q ) } } { M _ { \ell } \sum _ { q = 1 } ^ { M _ { \ell } } f _ { ( q ) } } - \frac { M _ { \ell } + 1 } { M _ { \ell } } .\tag{21}
$$

Unused prefixes are excluded, and a lower value indicates a more even distribution among the occupied prefixes. Table 1 reports ICR and utilization as percentages. In the three-level block of Table 1, ICR uses the complete three-token SID. In the five-level block, ICR uses the complete five-token SID, while utilization and Gini report each tokenizer’s L1–L3 prefix.

SID tokenizer fitting, SID encoding, and codebook analyses are conducted on a CPU server with two AMD EPYC 9654 processors (96 cores per socket) and 2.2 TB RAM. Downstream generative-retrieval training and evaluation are conducted on a separate server with eight NVIDIA H800 GPUs (80 GB HBM3 each) and two Intel Xeon Platinum 8558 processors (48 cores per socket).

## D Additional Experimental Results

This section follows the order of the main analysis. It first provides the carryover measurement and controlled-update details used in the motivation, then reports the complete public ablations, public-benchmark sensitivity results, and two qualitative case studies.

## D.1 Residual Carryover Analysis

For the industrial RQ-KMeans tokenizer, we average the carryover ratio in Eq. (4) over the full tokenizer-fitting population. After L2 normalization, the carryover ratio equals the absolute cosine between the next-level representation and the selected centroid. For two independent isotropic unit directions, the expected absolute cosine is

$$
\mathbb { E } \big [ | \mathbf { u } ^ { \mathsf { T } } \mathbf { v } | \big ] = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d + 1 ) / 2 ) } , \qquad \mathbf { u } , \mathbf { v } \overset { \mathrm { i . i . d . } } { \sim } \operatorname { U n i f } ( \mathbb { S } ^ { d - 1 } ) .\tag{22}
$$

For $d = 1 2 8 .$ , this expectation is 7.07%, providing a dimensional scale for the values in Figure 2(a). Figure 2(a) reports $\bar { m } ^ { ( \ell ) }$ for the three levels.

The controlled update on the industrial dataset used in the main text can be written directly in terms of centroid subtraction and projection: let $\mathbf { c } _ { i } = \bar { \mathbf { c } } _ { z _ { i } ^ { ( 1 ) } } ^ { ( 1 ) } , \rho _ { i } = \| \mathbf { c } _ { i } \| _ { 2 } , \hat { \mathbf { c } } _ { i } =$ $\mathbf c _ { i } / \rho _ { i } , \alpha _ { i } = \mathbf r _ { i } ^ { ( 1 ) \top } \hat { \mathbf c } _ { i } ,$ , and $\mathbf { u } _ { i } = \mathbf { r } _ { i } ^ { ( 1 ) } - \alpha _ { i } \hat { \mathbf { c } } _ { i } .$

$$
\begin{array} { c } { \displaystyle \mathbf { r } _ { i } ^ { ( 2 ) } ( \eta ) = \mathrm { n o r m } \left( \mathbf { r } _ { i } ^ { ( 1 ) } - \left[ \eta \mathbf { c } _ { i } + ( 1 - \eta ) \frac { \mathbf { r } _ { i } ^ { ( 1 ) \top } \mathbf { c } _ { i } } { \| \mathbf { c } _ { i } \| _ { 2 } ^ { 2 } } \mathbf { c } _ { i } \right] \right) } \\ { = \mathrm { n o r m } \left( \eta ( \alpha _ { i } - \rho _ { i } ) \hat { \mathbf { c } } _ { i } + \mathbf { u } _ { i } \right) . } \end{array}\tag{23}
$$

The second equality follows from Eq. (3) and shows that the operational update scales only the original carryover term before applying normalization. The L1 codebook and assignments remain fixed, and L2 is refitted for every value of η.

## D.2 Public Benchmark Ablations

Table A1 reports the complete public results for the three ablations, consistent with the main-text ablation findings.

<table><tr><td rowspan="2">Variant</td><td colspan="2">Sports</td><td colspan="2">Toys</td><td colspan="2">Clothing</td><td colspan="2">LastFM</td></tr><tr><td>Recall</td><td>NDCG</td><td>Recall</td><td>NDCG</td><td>Recall</td><td>NDCG</td><td>Recall</td><td>NDCG</td></tr><tr><td>PRQ-KMeans</td><td>0.0531</td><td>0.0219</td><td>0.0851</td><td>0.0361</td><td>0.0382</td><td>0.0151</td><td>0.0179</td><td>0.0076</td></tr><tr><td>Global</td><td>0.0505</td><td>0.0202</td><td>0.0830</td><td>0.0359</td><td>0.0370</td><td>0.0151</td><td>0.0107</td><td>0.0044</td></tr><tr><td>- Soft</td><td>0.0515</td><td>0.0210</td><td>0.0815</td><td>0.0350</td><td>0.0371</td><td>0.0146</td><td>0.0043</td><td>0.0017</td></tr><tr><td> Projection</td><td>0.0503</td><td>0.0201</td><td>0.0810</td><td>0.0345</td><td>0.0360</td><td>0.0141</td><td>0.0099</td><td>0.0044</td></tr></table>

Table A1: Ablation study on the public benchmarks.

## D.3 Public-Benchmark Hyperparameter Sensitivity

We also evaluate hyperparameter sensitivity on Sports, Toys, Clothing, and LastFM. Figure A1 varies $\dot { k } \in \{ 1 , 2 , 5 , \dot { 1 0 } \}$ at fixed $\beta = 1 5$ and $\beta \in \mathsf { \{ 1 0 , 1 5 , 2 0 , 2 5 \} }$ at fixed $k = 2$ For the neighborhood size, k = 2 gives the best or tied-best Recall and NDCG on all four benchmarks, while increasing k further provides no consistent gain. For the concentration parameter, $\beta = 1 5$ performs best on Sports, Clothing, and LastFM, whereas $\beta = 1 0$ performs best on Toys. Thus, the preferred concentration varies across datasets.

![](images/da6b0f7faacdd75673c870c6beced6c8c2a068cd42d6ceeafe4a984f17b11514.jpg)  
Figure A1: Hyperparameter sensitivity on the public benchmarks. Each dataset occupies one panel. The left segment varies Top-k with $\beta \ : = \ : 1 5$ , and the right segment varies weight concentration with $k = 2$ . Recall and NDCG follow the definitions in Section 5.1.

## D.4 Qualitative Case Studies

Tables A2 and A3 examine two aspects of hierarchical SID organization. The collision case follows one fixed item group through the two tokenizers, and the weak-intent case compares the prefix neighborhoods associated with the same query.

For the collision case, we select a 12-item Kitty-themed slippers group using a fixed SID-structure criterion. The group shares one RQ-KMeans full SID. PRQ-KMeans retains one common level-1 prefix, separates the items into two level-2 prefixes, and produces three full SIDs with four

items each. Table A2 summarizes the keywords associated with the resulting PRQ-KMeans branches.
<table><tr><td colspan="4">Full-SID</td></tr><tr><td>branch</td><td>L2</td><td></td><td>Items Keywords</td></tr><tr><td>A</td><td>A</td><td>4</td><td>Outdoor; dormitory; shower.</td></tr><tr><td>B</td><td>B</td><td>4</td><td>Indoor; soft-soled.</td></tr><tr><td>C</td><td>A</td><td>4</td><td>Bathroom; travel; home/outdoor.</td></tr></table>

Table A2: Keyword summaries for the selected 12-item collision group under PRQ-KMeans. Branch labels are anonymized.

For the pre-specified query translated as “hotel,” we identify the level-2 prefix assigned by each tokenizer and inspect the catalog items that share this prefix. The RQ-KMeans prefix contains 22 items and 14 distinct level-3 branches, while the PRQ-KMeans prefix contains 100 items and 29 branches. The latter neighborhood includes observed hotelrelated contexts such as entrances, bathrooms, corridors, and commercial installations.

<table><tr><td>Method</td><td>Items under L2 prefix</td><td>Distinct L3 branches</td></tr><tr><td>RQ-KMeans</td><td>22</td><td>14</td></tr><tr><td>PRQ-KMeans</td><td>100</td><td>29</td></tr></table>

Table A3: Prefix-neighborhood statistics for the weak-intent query translated as “hotel.”