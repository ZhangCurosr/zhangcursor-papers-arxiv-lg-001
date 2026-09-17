# QuanText: Protecting Dataset-Level Secrets in Textual Data Sharing

Shuaiqi Wang<sup>∗</sup>, Zinan Lin<sup>†</sup>, and Giulia Fanti<sup>∗</sup>

<sup>∗</sup>Carnegie Mellon University <sup>†</sup>Microsoft Research

{shuaiqiw,gfanti}@andrew.cmu.edu zinanlin@microsoft.com

Abstract—Natural-language datasets support many downstream applications and research studies. However, released text can reveal sensitive global properties of the data source, such as the proportion of records associated with a particular gender, diagnosis, or political stance. Existing work has largely focused on property inference attacks that recover such global properties, while defenses for protecting these dataset-level secrets remain limited. Differential privacy, though effective for protecting individual records, provides only weak protection for aggregate properties. In this work, we propose Randomized Quantization for Text (QuanText), a training-free and large-language-modelagnostic data release mechanism that protects global secrets in textual datasets while preserving data utility. Given a function of the dataset that represents the secret in question (e.g. proportion of diabetics) and attributes over which the data holder wants to retain utility (e.g., topic and sentiment), QuanText distorts the distribution of the secret quantity and the distributions of correlated attributes by (i) constructing a set of candidate release distributions over attributes (both secret and non-secret), (ii) randomly selecting a distribution that is sufficiently close to the (private) empirical distribution, and (iii) rewriting each private text sample to match the chosen attribute distribution using attribute-related snippets from the original text. QuanText is inspired by the Statistic Maximal Leakage (SML) framework, which bounds leakage about a secret function of a data distribution. We show that under ideal conditions, QuanText theoretically satisfies an SML guarantee. Since these ideal conditions may not hold in practice, experiments on real-world datasets demonstrate that QuanText achieves a better empirical privacy–utility tradeoff than competing data generation baselines. The code for QuanText is available at https://github.com/wsqwsq/QuanText.

Index Terms—Dataset-level privacy, data sharing, Statistic Maximal Leakage, Randomized Quantization.

## I. INTRODUCTION

Natural-language datasets are important for both practical applications and academic research [1], [2]. However, their release also raises significant privacy concerns. Beyond the leakage of sensitive information from individual samples [3], [4], a released dataset may reveal sensitive global properties [5]–[7], such as the proportions of samples associated with certain attribute categories. For example, in medical records or patient–doctor dialogues [8], even after individual identifiers have been removed, the released corpus may expose the fraction of female patients or the prevalence of specific diagnostic categories, revealing sensitive population-level demographic or health information. Similarly, in social-media datasets [9], released text may reveal politically sensitive aggregate information, such as the proportion of posts supporting a controversial topic. Protecting such dataset-level properties can be important for textual data sharing, while the released data should still preserve realistic semantics to remain useful (Fig. 1).

![](images/0d832d16389bc5be1d102c7999e193e812c3da34da3a925ef917725c5a11fa77.jpg)  
Fig. 1: Textual data sharing can reveal sensitive global properties, such as the proportion of patients with a certain disease. We aim to design a data release mechanism that protects such global secrets while preserving the utility of released text.

Several studies have proposed property inference attacks that infer global properties of released data or of the generative models used in the release process [5], [6], [10], [11]. However, comparatively few works focus on defenses that protect such sensitive global properties [10]. Prior studies have shown that differential privacy (DP), while effective for protecting individual samples, can be insufficient for protecting aggregate properties [12]–[15]. Intuitively, DP perturbs individual samples independently, but because the added noise is typically zero-mean, its effects largely cancel out in aggregate, leaving the proportions of the sensitive attribute categories close to their original values. [10] proposes a simple defense that distorts the secret values in released data in the vision setting. However, this approach remains insufficient when other attributes in the dataset are correlated with the secret, since attackers can exploit these correlations to recover the sensitive global property. Designing defenses in textual settings is similarly challenging: first, a single attribute may be expressed across multiple parts of a text; second, textual attributes are often highly correlated, allowing attackers to recover the global secret from seemingly non-sensitive words, phrases, or semantic cues.

To this end, we propose Randomized Quantization for Text (QuanText), a training-free and large-language-model-agnostic data release mechanism that protects global secret proportions of a dataset while limiting leakage through correlated attributes and preserving the utility of released data. At a high level, QuanText proceeds in three stages. (i) First, QuanText efficiently constructs a set of candidate release distributions over the specified attribute types the data holder wants to retain utility on. For example, a user with a dataset of tweets may wish to preserve the distributions of topics and sentiments, both of which may be correlated with the target secret: the proportion of tweets favoring a particular political stance. Since directly releasing the private data (or even the distribution of these attributes) may leak the secret through their correlations, QuanText instead builds alternative candidate distributions with different secret and correlated-attribute marginal distributions. In our example, each candidate distribution specifies a joint distribution over topics and sentiments. To construct these candidates, QuanText uses an auxiliary public dataset: it applies the Chow–Liu algorithm [16] to approximate the joint distribution of attributes as a product distribution of simpler marginal and conditional distributions. (ii) Second, given the private dataset, QuanText determines the released joint distribution by randomly selecting from the subset of candidate distributions closest to the private empirical distribution. (iii) Third, QuanText constructs a released dataset that follows the selected attribute distribution. For each private sample, an attribute combination is assigned according to the release distribution, and a large language model rewrites the sample using the assigned attributes together with the corresponding attribute-relevant snippets extracted from the original text. For example, suppose a private tweet has the attributes (topic = climate change, sentiment = positive) and is assigned a synthetic attribute combination (topic = climate change, sentiment = negative). QuanText extracts relevant snippets from the original tweet, such as “global warming” and “clean energy”, and uses them to generate a new tweet that expresses the assigned topic and sentiment.

Under idealized conditions, we prove that QuanText satisfies a privacy guarantee based on the Statistic Maximal Leakage (SML) framework [7]. SML bounds the leakage of a specified global-level secret under the worst-case data prior and against arbitrary attack strategies, including those that exploit correlated attributes. Since SML is robust to post-processing [7], a dataset released under a Π−SML guarantee can be used for downstream tasks like model training without degrading privacy.

Our theoretical analysis accounts for correlation between textual content and style, such as word choice within individual text samples or text length. We quantify content–style dependence by a worst-case conditional probability ratio l, and prove that the SML is bounded by the mechanism-dependent quantization term plus log l, so weak dependence between content and style causes only limited additional privacy loss.

Since the correlation parameter is difficult to estimate in practice, we compare the empirical privacy and utility performance of QuanText to several common data release baselines. Our results show that QuanText achieves a better empirical privacy–utility trade-off than competing data generation baselines on known property inference attacks.

Our contributions are summarized as follows.

• Mechanism Design: We propose QuanText, a modelagnostic, training-free data release mechanism that protects sensitive global secret proportions of a dataset while limiting leakage through correlated attributes and preserving data utility.

• Privacy Analysis: We analyze the idealized privacy guarantee of QuanText via Statistic Maximal Leakage (SML), which is robust to arbitrary attack strategies and further processing of the released data. We provide an SML guarantee whose additive degradation is controlled by the strength of content–style correlation.

• Empirical Evaluation: We evaluate the privacy and utility performance of QuanText on real-world datasets. Compared with existing data generation baselines, Quan-Text achieves a better privacy–utility trade-off.

## II. RELATED WORK

Attribute Inference Attacks. Attribute inference attacks aim to recover a missing or sensitive attribute of an individual sample from its observed features, thus concern sample-level privacy [17]–[21]. Existing studies typically assume access to a trained classifier. Our setting is fundamentally different. We study dataset or distribution-level privacy, where the sensitive information is a global property of the private dataset, such as the proportion of samples with a particular attribute.

Property Inference Attacks. Property inference attacks, also called distribution inference attacks, aim to infer aggregate properties of a private training dataset or distribution, such as demographic or class-label proportions, rather than attributes of individual records. Most prior work [12]–[14], [22]–[26] targets discriminative classifiers, including fully connected neural networks [23], convolutional neural networks [24], and classifiers in federated learning [14]. This line of work also shows that differential privacy (DP), while designed to protect individual records, provides only weak protection for aggregate statistics [12]–[14]. [15] similarly finds DP may leave attribute correlations vulnerable in classifier-based settings.

A smaller body of work studies property inference for generative models or their synthetic outputs [5], [6], [10], [11]. For Generative Adversarial Networks [5] and diffusion models [10], attackers typically observe generated samples and estimate the target property empirically, assuming the generative distribution reflects the private training distribution. [11] studies the narrower task of property existence inference, which tests whether a target property appears in the training data, i.e., whether its proportion is nonzero, rather than esti mating general proportions. More recently, [6] studies property inference attacks against large language models in both black-box settings, where attackers label generated samples to estimate the secret, and gray-box settings, where attackers use model weights and auxiliary data to train shadow models that map model features to the secret. Our setting is closest to property inference from generated data; however, while prior work primarily studies attacks, our work focuses on defense. Defenses Against Property Inference Attacks. Existing defenses against property inference attacks mainly target classifiers, using techniques such as property unlearning and adversarial training [26]–[28]. These heuristic methods are model-dependent and require modifying the training procedure. In contrast, we consider a model-agnostic, training-free defense at the data-release stage. [10] perturbs the sensitive property in generated data, but changing the secret alone is insufficient: an attacker can still recover it from other attributes correlated with it. Our work addresses this limitation by protecting fine-grained dataset-level proportions while limiting leakage through correlated attributes.

## III. PROBLEM FORMULATION

The data holder has a private textual dataset $\begin{array} { r l } { { \mathcal { D } } } & { { } = } \end{array}$ $\{ x _ { 1 } , \ldots , x _ { n } \}$ of size n, where each sample $x _ { i } \in { \mathcal { X } }$ is a text string from universe X. The data holder aims to release a dataset $\mathcal { D } ^ { \prime }$ via a data generation mechanism ${ \mathcal D } ^ { \prime } = \mathcal { M } ( { \mathcal D } )$ while protecting a count query $G ( \mathcal D )$ measuring the proportion of samples associated with a particular attribute category. That is, suppose there exists an oracle ${ \mathfrak { g } } : { \mathcal { X } }  \{ 0 , 1 \}$ , that outputs 1 if and only if the sample x has a certain property $( \mathrm { e . g . }$ , “has diabetes”), and 0 otherwise. Then

$$
G ( { \mathcal { D } } ) \triangleq { \frac { 1 } { n } } \sum _ { x \in { \mathcal { D } } } { \mathfrak { g } } ( x ) .
$$

For brevity, we will use the shorthand G to denote $G ( \mathcal D )$ The data holder specifies a set of m attribute types of interest, denoted by $\left\{ \phi _ { i } \right\} _ { i = 1 } ^ { \bar { m } } ( \mathbf { e . g . } , \phi _ { 1 } =$ sentiment, $\phi _ { 2 } = \mathrm { s t a n c e } )$ , over which utility should be retained; these attribute types may be correlated with the secret G. A core assumption of this work is that the secret G can be recovered by knowing the values of all attributes $\{ \phi _ { i } \} _ { i = 1 } ^ { m } ;$ we model leakage about the secret from non-attribute content in §IV-B. Each attribute type ϕ<sub>i</sub>, such as sentiment, takes values from a predefined set of categories with size $\psi _ { i }$ , denoted by $\left\{ v _ { j } ^ { i } \right\} _ { j = 1 } ^ { \psi _ { i } }$ , such as positive, negative, and neutral. Following [6], the dataset holder may get access to a public auxiliary dataset $\mathcal { D } _ { \mathrm { a u x } }$ from a similar domain.

After observing the released dataset, the attacker aims to infer the original secret G. We assume that the attacker knows the data generation mechanism and is computationally unbounded.

a) Privacy Constraint: Theoretically, we measure the privacy of a data release mechanism M in the Statistic Maximal Leakage (SML) framework [7], which provides privacy guarantees under any data prior and against arbitrary attack strategies. Let $\mathcal { P }$ denote the prior distribution of data, A be the attack method, G and $\hat { G }$ be the random variables representing the original and attacker-guessed secret values, and G be the set of all possible secret values. SML is defined as

$$
\Pi _ { \mathcal { M } , \mathfrak { g } } = \operatorname* { s u p } _ { \mathcal { P } , \mathcal { A } } \log \frac { \mathbb { P } \left( \hat { G } = G \right) } { \operatorname* { s u p } _ { \mathcal { G } \in \mathbf { G } } \mathbb { P } _ { G } \left( g \right) } .\tag{1}
$$

SML takes the worst-case leakage over all possible data priors and attack methods. Intuitively, SML measures the gain in the attacker’s probability of correctly guessing the secret given the released dataset $\mathcal { D } ^ { \prime }$ . A smaller value of Π $\mathcal { M } , { \mathfrak { g } }$ indicates stronger protection; in particular, if Π $\mathcal { M } , _ { \mathfrak { g } } \le \eta _ { \mathfrak { n } }$ , any attack success probability after release is at most $\alpha e ^ { \eta }$ , where α is the success probability using prior knowledge alone. SML also satisfies post-processing and adaptive composition [7], making the privacy guarantee robust to further processing of the released output and sequential applications of data release mechanisms.

## IV. RANDOMIZED QUANTIZATION FOR TEXT

We design Randomized Quantization for Text (QuanText) as a data release mechanism that limits leakage about the secret quantity G while preserving the utility of the attribute types of interest, $\left\{ \phi _ { i } \right\} _ { i = 1 } ^ { m }$

a) Straw Man Solution: A straw-man design that simply removes samples containing the sensitive property $( \mathrm { e . g . }$ , “has diabetes”) is insufficient, because an attacker may still infer the secret from the distributions of correlated attributes $( \mathrm { e . g . }$ age), which may remain largely unchanged. To mitigate this risk, QuanText perturbs both the distribution of the sensitive property and the distributions of attributes correlated with it.

b) Overview: Our core assumption is that the secret G is revealed exactly by knowing the attribute values of all samples in the private dataset. Hence, our approach is to rewrite samples in the dataset so that their attribute values, in aggregate, do not reveal the secret G. QuanText extracts attributes values from each text sample in D, obtaining an empirical joint distribution $\hat { \mathbb { P } }$ over these attribute values. It also constructs a set of alternative joint distributions over attribute values. To do $\mathbf { s o } ,$ it models the underlying joint attribute distribution with a tree-structured probabilistic graphical model using the Chow-Liu algorithm [16]. Given a set of candidate distributions (or a quantization of the attribute distribution space), QuanText selects uniformly at random from the set of candidate distributions that are closest in total variation distance to the empirical attribute distribution; the size of this subset can be varied to tune privacy guarantees. Once we draw an alternative attribute distribution $\tilde { \mathbb { P } } ,$ we use an LLM to rewrite each sample in D, mapping its attribute values to a synthetic combination of attribute values drawn from P<sup>˜</sup>. To minimize the changes to the dataset, we use the Hungarian algorithm to find a low-cost matching between samples’ true attributes to a list of synthetic attributes; here, cost is defined as the number of differing attribute values.

## A. QuanText Algorithm

We illustrate the pipeline of QuanText in Fig. 2 and present the full algorithm in Alg. 1.

1) Approximate the joint distribution over the attribute types of interest, $\mathbb { P } \left( \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { m } \right)$ , as $\mathbb { P } \left( \phi _ { 1 } \right)$ $\begin{array} { r } { \prod _ { i = 2 } ^ { m } \mathbb { P } \left( \phi _ { i } | \phi _ { \pi ( i ) } \right) } \end{array}$ using the Chow–Liu algorithm [16] on an auxiliary public dataset $\mathcal { D } _ { \mathrm { a u x } }$ (Line 1). Here, $\phi _ { \pi ( i ) }$ denotes the parent of $\phi _ { i }$ in the Chow–Liu tree, and ϕ is the root node.

2) Construct $\gamma$ candidate release distributions per distribution component, namely $\mathbb { P } \left( \phi _ { 1 } \right)$ and $\mathbb { P } \left( \phi _ { i } | \phi _ { \pi ( i ) } \right)$ for all $i \in \{ 2 , 3 , . . . , m \}$ (Line 1). We defer the construction algorithm to §A.

![](images/cff61393c8cd9c830ec652653b04d215183512366bb8b19aea97d17e7e556ed2.jpg)  
Step 5: Match target attribute combinations, and rewrite samples  
Fig. 2: Illustration of QuanText on a customer review dataset, where the attribute types of interest are review star, busines category, and key words. Step 1 approximates the joint attribute distribution with a Chow–Liu tree learned from an auxiliary public dataset. Step 2 constructs candidate release distributions for each distribution component. Steps 3 and 4 randomly select released distributions based on the private dataset and then sample target attribute combinations. Step 5 matches these target combinations to private samples, extracts snippets relevant to the assigned target attributes, and rewrites the samples accordingly to form the released dataset.

3) Compute the empirical distributions from the original private dataset: $\hat { \mathbb { P } } \left( \phi _ { 1 } \right)$ and $\hat { \mathbb { P } } \left( \phi _ { i } | \phi _ { \pi ( i ) } \right)$ for all $i \in$ $\{ 2 , 3 , \ldots , m \}$ (Line 1).

4) Select release distributions for the distribution components, combine them into a released joint distribution, and construct target attribute combinations:

• For each distribution component, determine the released distribution $\tilde { \mathbb { P } } \left( \phi _ { 1 } \right)$ or $\tilde { \mathbb { P } } \left( \phi _ { i } | \phi _ { \pi ( i ) } \right)$ $i \in$ $\{ 2 , 3 , \ldots , m \}$ by (a) identifying the top-k candidate distributions with the smallest total variation (TV) distances from the corresponding empirical private distribution (Line 1), and (b) selecting one of them uniformly at random (Line 1).

• Form the released joint distribution (Line 1): $\begin{array} { r } { \tilde { \mathbb { P } } \left( \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { m } \right) = \tilde { \mathbb { P } } \left( \phi _ { 1 } \right) \cdot \prod _ { i = 2 } ^ { m } \tilde { \mathbb { P } } \left( \phi _ { i } | \phi _ { \pi ( i ) } \right) } \end{array}$

• Draw n target attribute combinations from the released joint distribution, where n is the number of private samples (Line 1). Each combination specifies one category value per attribute type of interest.

5) Rewrite samples to match the released distribution using attribute-related snippets from the original text:

• Label the attribute combination of each private sample via an LLM (Line 1), and map the private samples to the target attribute combinations using the Hungarian algorithm (Line 1).

• For each sample, extract snippets relevant to its mapped target attribute combination using an LLM (Line 1).

• Rewrite each sample with an LLM according to the mapped target attribute combination and the extracted snippets (Line 1).

## B. Idealized Privacy Guarantee

To analyze the Statistic Maximal Leakage (SML) of Quan-Text, following prior work [29]–[31], we decompose textual data along two axes: content $Y ~ \in ~ \mathcal { D }$ and style $Z \in { \mathcal { Z } }$ with $\mathcal { V } \cup \mathcal { Z } \ : = \ : \mathcal { X }$ . Content captures the semantic meaning of the text, including attribute-level information, whereas style refers to non-semantic surface properties, such as word choice, phrasing, and text length. Global-level secrets are functions of the private dataset and may be correlated with content; through content–style dependence, style may therefore also reveal information about the secret.

We quantify the dependence of style on content using the worst-case conditional probability ratio, denoted by $L \left( Y ; Z \right)$

$$
L \left( Y ; Z \right) \triangleq \operatorname* { s u p } _ { y _ { 1 } , y _ { 2 } \in \mathcal { V } ; z \in \mathcal { Z } } \frac { \mathbb { P } \left( Z = z | Y = y _ { 1 } \right) } { \mathbb { P } \left( Z = z | Y = y _ { 2 } \right) } .\tag{2}
$$

Intuitively, $L \left( Y ; Z \right)$ measures how much the content Y can change the likelihood of a particular style Z. When $L \left( Y ; Z \right) = 1$ , the style is independent of the content. We show that $L \left( Y ; Z \right)$ upper bounds the mutual information $I \left( Y ; Z \right)$

$$
I f L \left( Y ; Z \right) \leq l ,
$$

$$
I \left( Y ; Z \right) \leq \log l .
$$

We assume that LLM used in QuanText can generate samples that satisfy the requirements specified in the prompts.

Assumption IV.2 (LLM Capability). LLM rewrite faithfully realizes the assigned attribute combination: for every private sample x, LLM $f ,$ extracted snippets s, and target combination $^ { a , }$ the rewritten sample $x ^ { \prime } \ = \ \mathrm { R E W R I T E } _ { f } ( x , s , a )$ satisfies ${ \mathfrak { b } } ( x ^ { \prime } , a ) = 1$ , where the oracle b outputs 1 if and only if a sample has property combination a.

Algorithm 1: Randomized Quantization for Text   
Input: Private dataset ${ \mathcal { D } } = \{ x _ { 1 } , \ldots , x _ { n } \} ,$ auxiliary dataset   
$\mathcal { D } _ { \mathrm { a u x } } ,$ attribute types of interest $\{ \phi _ { 1 } , \ldots , \phi _ { m } \}$   
number of candidate distributions $\gamma ,$ selection size $k ,$   
large language model $f .$   
Output: Released dataset $\mathcal { D } ^ { \prime } .$   
// Step 1: Chow-Liu approximation   
1 $\begin{array} { r } { \mathbb { P } ( \underline { { \phi _ { 1 } } } , \underline { { \therefore } } , \overline { { \phi } } _ { m } ) \approx \mathbb { P } ( \phi _ { 1 } ) \cdot \prod _ { i = 2 } ^ { m } \mathbb { P } ( \phi _ { i } | \phi _ { \pi ( i ) } )  } \end{array}$ CHOWLIU   
$( \mathcal { D } _ { \mathrm { a u x } } ) .$   
// Step 2-4: Decide released distribution   
2 for each distribution component $\mathbb { P } \in \{ \mathbb { P } ( \phi _ { 1 } ) \}$ ∪   
$\left\{ \mathbb { P } ( \phi _ { i } \mid \phi _ { \pi ( i ) } = v _ { j } ^ { \pi ( i ) } ) : i = 2 , . . . , m , j \in \left[ \psi _ { \pi ( i ) } \right] \right\}$ do   
3 Q ← CANDIDATECONSTRUCTION $( \gamma , \mathbb { P } )$ [Alg. 2].   
4 Compute the empirical distribution P<sup>ˆ</sup> from D.   
5 $\begin{array} { r } { \mathcal { Q } _ { k } \gets \arg \operatorname* { m i n } \mathcal { Q } ^ { \prime } { \subseteq } \mathcal { Q } \colon | \mathcal { Q } ^ { \prime } | { = } k \sum _ { q \in \mathcal { Q } ^ { \prime } } d _ { \mathrm { T V } } ( q , \hat { \mathbb { P } } ) . } \end{array}$   
6 P<sup>˜</sup> ← UNIFORM(Q<sub>k</sub>).   
7 end   
8 $\begin{array} { r } { \tilde { \mathbb { P } } ( \phi _ { 1 } , \dots , \phi _ { m } )  \tilde { \mathbb { P } } ( \phi _ { 1 } ) \cdot \prod _ { i = 2 } ^ { m } \tilde { \mathbb { P } } ( \phi _ { i } \mid \phi _ { \pi ( i ) } ) . } \end{array}$   
9 Draw target attribute combinations   
$\begin{array} { r } { \big \{ \left( \tilde { a } _ { j } ^ { 1 } , \dots , \tilde { a } _ { j } ^ { m } \right) \big \} _ { j = 1 } ^ { n } \overset { \mathrm { i . i . d . } } { \sim } \tilde { \mathbb { P } } ( \phi _ { 1 } , \dots , \phi _ { m } ) . } \end{array}$   
// Step 5: Match and rewrite samples   
10 for $i \gets 1$ to n do   
11 $| | \quad ( \hat { a } _ { i } ^ { 1 } , \dots , \hat { a } _ { i } ^ { m } ) \gets \mathrm { L A B E L A T T R I B U T E S } _ { f } ( x _ { i } ) .$   
12 end   
13 Set $\begin{array} { r } { C _ { i j } \gets \sum _ { r = 1 } ^ { m } \mathbb { 1 } \left[ \hat { a } _ { i } ^ { r } \neq \tilde { a } _ { j } ^ { r } \right] } \end{array}$ for all $i , j \in \{ 1 , \ldots , n \}$   
14 σ ← HUNGARIAN(C), where σ(i) is the target combination   
assigned to x<sub>i</sub>.   
15 for i ← 1 to n do   
16 s<sub>i</sub> ← EXTRACTSNIPPETS $\underset { \ r { f } } { \big ( } x _ { i } , \left( \tilde { a } _ { \sigma ( i ) } ^ { 1 } , \ldots , \tilde { a } _ { \sigma ( i ) } ^ { m } \right) \big )$   
17 $x _ { i } ^ { \prime } \gets \mathrm { R E W R I T E } _ { f } \big ( x _ { i } , s _ { i } , \big ( \tilde { a } _ { \sigma ( i ) } ^ { 1 } , \cdot \cdot \cdot , \tilde { a } _ { \sigma ( i ) } ^ { m } \big ) \big )$   
18 end   
19 $\mathcal { D } ^ { \prime }  \{ x _ { 1 } ^ { \prime } , \ldots , x _ { n } ^ { \prime } \}$   
20 return $\mathcal { D } ^ { \prime } .$

The following theorem characterizes the SML of QuanText.

Theorem IV.3 (SML of QuanText). For any secret defined as the proportion of samples associated with particular attribute categories in the private dataset D, $i f L \left( Y ; Z \right) \leq l ,$ then under Thm. IV.2, the SML of QuanText satisfies

$$
\Pi _ { \mathcal { M } , \mathfrak { g } } \le \log \left( \frac { \gamma } { k } \right) ^ { 1 + \sum _ { i = 2 } ^ { m } \psi _ { \pi ( i ) } } + \log l ,\tag{3}
$$

where $\phi _ { \pi ( i ) }$ denotes the parent of $\phi _ { i }$ in the constructed Chow– Liu tree, $\psi _ { \pi ( i ) }$ is the number of values attribute $\phi _ { \pi ( i ) }$ can take, $\gamma$ is the number ofcandidate distributionsfor each distribution component, and k is the selection size.

(Proof in §B.) Thm. IV.3 shows that QuanText provides a stronger privacy guarantee, i.e., a smaller SML value, when the number of candidate distributions $\gamma$ per conditional attribute is smaller, the number of nearest-neighbor distributions k is larger, or the parent attributes in the Chow–Liu tree have fewer categories. This is consistent with the intuition that, when the mechanism selects from a larger fraction of the candidate distributions and each relevant attribute has fewer possible categories, the released distribution reveals less information for the attacker to use when inferring the secret. Furthermore, content–style correlation increases the privacy loss bound additively by log l.

Since SML admits an operational interpretation, Thm. IV.3 directly bounds the attacker’s success probability.

Proposition IV.4 (Best attack success rate). Let α denote the optimal attack success rate using prior knowledge alone. After observing the dataset released by QuanText, the optimal attack success rate satisfies

$$
\mathbb { P } ( \hat { G } = G ) \le \alpha \cdot \Big ( \frac { \gamma } { k } \Big ) ^ { 1 + \sum _ { i = 2 } ^ { m } \psi _ { \pi ( i ) } } .
$$

## V. EXPERIMENTS

We evaluate the privacy and utility of QuanText on realworld datasets.

## A. Datasets

We use the Tweet Stance [9] and ChatDoctor [8] datasets, and treat them as the real private data to be released and protected.

• Tweet Stance Dataset [9] This dataset contains 4,870 tweets annotated with three labels: Target (five political topics: climate change, atheism, legalization of abortion, Hillary Clinton, and Donald Trump), Stance (Favor, Against, Neither), and Sentiment (Positive, Negative, Neither). We subsample 3,000 tweets as the private dataset and use the remaining 1,870 samples as the auxiliary dataset for the Chow–Liu spanning tree. We consider Target, Stance, and Sentiment as the attributes of interest, yielding the Chow–Liu spanning tree {Target → Stance; Target → Sentiment}. We define the secrets as the proportions of tweets in a specific Target, Stance, or Sentiment category, and in a specific Target–Stance combination. All secrets are specified with a precision of 0.01%.

ChatDoctor Dataset [8] This dataset consists of patient– doctor dialogues. Following [6], we define two types of secrets: (1) the proportion of female samples and (2) the proportions of specific medical diagnoses. For (1), we subsample three datasets of 600 dialogues each with target female ratios 0.3, 0.5, and 0.7, taking gender as the only attribute of interest. For (2), we subsample 600 dialogues, define the secrets as the proportions of samples with mental disorder, digestive disorder, and childbirth, each to a precision of 0.01%, and take diagnosis as the only attribute of interest. In both cases, the single attribute of interest reduces the Chow–Liu tree to a single node.

## B. QuanText and Data Generation Baselines

As shown in Thm. IV.3, for a given dataset, the SML of QuanText is determined by $\frac { \gamma } { k }$ under Thm. IV.2. Intuitively, $\frac { \gamma } { k }$ measures the degree of quantization among candidate distributions, coupled with how closely we stay in the neighborhood of the true empirical attribute distribution. We refer to this ratio as candidate dilution. Although Thm. IV.2 may not hold exactly in practice, we vary the candidate dilution, i.e., <sup>γ</sup> , to control the privacy performance of QuanText in our experiments, where smaller candidate dilution corresponds to stronger privacy. Specifically, we consider $\textstyle { \frac { \gamma } { k } } \in \{ 2 , 3 , 4 , 6 , 8 , 1 2 \}$ and use Llama3.1-8B-Instruct as the backend model.

![](images/8c3b713b01009d484f438fb5197bba1a2b3c957e7bf4a2ce833667657876b6fb.jpg)  
(a) Secret as the proportion of Target = Legalization of Abortion.

![](images/d34bb7d6655b134b67e20e88114073a0fcbefb454ef2a2259dc270114bd299c7.jpg)  
(b) Secret as the proportion of Stance = Favor

![](images/fde5c448c9c2c55c0dd5f0e783967c2e10c3a4afc4d9121f16671acdb65db30e.jpg)  
(c) Secret as the proportion of Sentiment = Negative.

![](images/d77707c7818e9176d81edcfe335a4aaf206bd25331fd70b2a875817d6a8575b8.jpg)  
(d) Secret as the proportion of Stance = Favor & Target = Legal ization of Abortion.  
Fig. 3: Attack MAE vs. aggregate utility under different data generation methods with varying privacy guarantees on Tweet.

As we do not know of any other defenses specifically against target property inference attacks in textual data release settings, we select Differentially Private (DP) synthetic data generation methods as our baselines, specifically, Private Evolution [32]–[34] and DP model fine-tuning [35]–[37]. We also include raw data release and subsampling as naive baselines.

• Raw Data Release: We release the private data directly.

• Subsampling: We randomly subsample and release half of the private dataset.

• Private Evolution (PE) [32]–[34], [38]: PE is a trainingfree method for differentially private synthetic data generation using foundation models [32]–[34], [38]–[40]. Starting from samples generated by a Random API, PE iteratively constructs a DP noisy histogram from privateto-synthetic nearest-neighbor votes, samples from this histogram, and perturbs selected samples via a Variation API. We use Augmented Private Evolution (Aug-PE) [33], the text-generation variant of PE, modifying only the Random and Variation prompts.

• DP Fine-Tuning (DP-FT) [35]–[37]: DP-FT fine-tunes the language model for next-token prediction using differentially private stochastic gradient descent (DP-SGD) [41]. Synthetic data are then generated from the finetuned model according to a generation instruction.

We run PE for 10 iterations and DP-FT for 15 epochs. For both PE and DP-FT, we vary the privacy budget ϵ ∈ 1, 2, 3, 4, and use Llama3.1-8B-Instruct as the backend model.

## C. Evaluation Metrics

Privacy: Although our theoretical arguments suggest that QuanText satisfies an SML guarantee in idealized settings, it is unclear how to estimate the correlation parameter l from Thm. IV.3. Hence, we evaluate an empirical privacy measure that can also be evaluated for the other baseline defenses we consider: we calculate the Mean Absolute Error (MAE) of the labeling-based property inference attack used in [5], [6], [10], described as follows. Higher MAE indicates better performance on protecting global secrets.

• Labeling-based Attack For each sample, the attack uses a large language model to decide whether the sample belongs to the sensitive attribute category. The property ratio is then computed as the number of samples in the sensitive category divided by the total number of samples.

![](images/8443faeeb63a7267569986b8ee68c5c14b09db7e0132552dd64f91729bdfcc9c.jpg)  
(a) Secret as Female Ratio = 0.3.

![](images/00723e7ec8cc2974256a9e581ccb92c0a8914989197a657fdb3f8e59df8773ff.jpg)  
(b) Secret as Female Ratio = 0.5.

![](images/00a3dc0efa391df3ca5171ab349945cd378cec5575387622c09f1c41f89b0df0.jpg)  
(c) Secret as the proportion of Diagnosis = Childbirth.

![](images/05c52d56d846374f6130353b5b0333f4e876a73e9f17d940d92d7928f2254bbf.jpg)  
(d) Secret as the proportion of Diagnosis = Digestive Disorder.  
Fig. 4: Attack MAE vs. aggregate utility under different data generation methods with varying privacy guarantees on ChatDoctor.

In our experiments, we use Llama3.1-8B-Instruct, Mistral-7B-Instruct, and Qwen3-4B-Instruct as the backend models for the labeling-based attack.

Although [6] proposes another property inference attack for text, it targets models trained on private or generated data, where the trained model may encode information about the global secrets. This makes it unsuitable for our setting, in which QuanText is training-free and the attacker infers the secrets directly from the generated dataset.

Utility: Inspired by [42], [43] on evaluating synthetic data, we adopt KNN-Precision, KNN-Recall, Fréchet Inception Distance, and Attribute Matching to evaluate the semantic and statistic performance of the generated data. KNN-Precision and KNN-Recall capture the semantic quality and coverage of the generated data. KNN-Precision and KNN-Recall are defined as the proportions of generated and real samples, respectively, whose embedding distance to at least one sample from the opposite dataset is smaller than the distance to the k-th nearest neighbor within their own dataset. Fréchet Inception Distance (FID) measures the embedding closeness of the real and generated data. Attribute Match (AM) quantifies the agreement between real and synthetic datasets with respect to predefined statistical and semantic features by measuring distances between their corresponding feature distributions. Specifically, it uses the Wasserstein-2 distance for numerical features and Total Variation (TV) distance for categorical features. In our evaluation, we use sample token length as the statistical feature, together with dataset-specific semantic features.

We summarize overall utility as the average of the four metrics above, each rescaled to [0, 1] so that larger values indicate better performance. Although averaging these values may not be ideal because the underlying metrics can have different scales, this practice is sometimes adopted in benchmarks to facilitate visualization and interpretation [43], [44].

## D. Results

We present the privacy and utility performance of QuanText and the baselines under varying privacy budgets for selected secrets on the Tweet Stance and ChatDoctor datasets in Figs. 3 and 4, which highlight two main takeaways:

• QuanText achieves a better privacy–utility trade-off than PE and DP-FT. As shown in Figs. 3 and 4, when the candidate dilution satisfies $\begin{array} { r } { \frac { \gamma } { k } \ \leq \ 8 , } \end{array}$ QuanText consistently achieves both higher attack MAE and a higher aggregate utility score than PE and DP-FT across privacy budgets $\epsilon \in \{ 1 , 2 , 3 , 4 \}$ , indicating superior privacy and utility performance. For both PE and DP-FT, the attack MAE remains consistently low and varies only slightly across different DP budgets ϵ, which is consistent with prior observations that data generation methods with differential privacy guarantees are insufficient to protect global-level properties [7], [45], [46]. We also observe that the attack MAE of Raw Data Release and Subsampling remains near zero, indicating that property inference attack succeeds on these naive baselines.

![](images/7a1ca08f0954a0119cfbecf767f155cb455723811d7df6237cc7d547aca8057b.jpg)  
(a) Secret as the proportion of Target = Legalization of Abortion.

![](images/6b991086ffc018c5655ff7e82e2a9c2988207831d355fdea602d651d2931ec47.jpg)

![](images/12f2f9371d352f56f9e854b2c5425e8efe344c0bd49d32cf018e7b0402f25bf0.jpg)  
(c) Secret as the proportion of Sentiment = Negative.

(b) Secret as the proportion of Stance = Favor.  
![](images/770aee599a63804fecdd5d96a4ad9bcc66124970ed119a0e614ac7eddda617c6.jpg)  
(d) Secret as the proportion of Stance = Favor & Target = Legalization of Abortion.  
Fig. 5: Attack MAE & aggregate utility of QuanText with $\gamma / k = 4$ on Tweet.

• Smaller candidate dilution $\textstyle { \frac { \gamma } { k } }$ improves privacy while preserving utility for QuanText. As shown in Fig. 3a and consistently observed across the other subplots in Figs. 3 and 4, decreasing the candidate dilution from 12 to 2 substantially improves privacy: the attack MAE increases from approximately 0.01 to 0.1. At the same time, this change incurs only a minor utility loss, with the aggregate utility score decreasing from 0.85 to 0.77. Intuitively, this is because QuanText primarily adjusts the proportions of selected attributes, while the reused attribute-related snippets help preserve the semantic quality and coverage of the generated samples. A similar phenomenon has been observed in tabular data generation with SML guarantees [7]. These results indicate that smaller candidate dilution provides a more favorable privacy–utility trade-off in practice.

1) Ablation Studies: We further study how QuanText performs under different parameter choices and against different attacker models.

a) Sensitivity analysis ofparameters γ and k under fixed candidate dilution: In Fig. 5, we fix the candidate dilution at $\mathit { \Pi } _ { \overline { { k } } } ^ { \gamma } ~ = ~ 4$ and vary γ and k proportionally to examine whether their individual values affect the privacy and utility performance of QuanText on the Tweet Stance dataset. For each parameter setting, we run QuanText three times and report the mean and standard deviation of both privacy and utility. We observe that, across different types of secrets, both the attack MAE and aggregate utility remain largely stable as γ and k vary. This demonstrates that QuanText is robust to the specific choices of $\gamma$ and k, provided that the candidate dilution is fixed.

b) Robustness of QuanText to attacks from different backend models: To examine whether QuanText is robust to attacks performed by different models, we consider three backend models: Llama3.1-8B-Instruct, Mistral-7B-Instruct, and Qwen3-4B-Instruct. For each backend model, we perform the labeling-based attack for all secrets on Tweet Stance datasets generated by QuanText with $\begin{array} { r } { \frac { \gamma } { k } = 4 , } \end{array}$ , PE and DP-FT with $\epsilon = 1$ , Raw Data Release, and Subsampling. For each secret, we rank these methods according to their attack MAE. We then compute the Spearman correlation between the rankings produced by each pair of backend models for each secret and average the correlations across secrets. The resulting averaged Spearman correlation matrix is shown in Table I. We observe that the Spearman correlation between any pair of models exceeds 0.6, indicating a strong correlation [47]. This suggests that the relative protection provided by each method is largely preserved across attack models, implying that the privacy advantage of QuanText is not tied to any particular attack model.

TABLE I: Averaged Spearman correlation matrix between backend models. For each backend model and secret, we rank the generation methods by their attack MAE, compute the Spearman correlation between the rankings for each pair of models, and average the correlations across secrets.
<table><tr><td></td><td>Llama</td><td>Mistral</td><td>Qwen</td></tr><tr><td>Llama</td><td>1.0000</td><td>0.7766</td><td>0.6878</td></tr><tr><td>Mistral</td><td>0.7766</td><td>1.0000</td><td>0.7724</td></tr><tr><td>Qwen</td><td>0.6878</td><td>0.7724</td><td>1.0000</td></tr></table>

## VI. CONCLUSION

In this work, we studied privacy-preserving textual data generation for protecting sensitive global properties, specifically the proportions of samples associated with specified attribute categories. We proposed QuanText, a model-agnostic, training-free mechanism that protects global secrets while limiting leakage through correlated attributes and preserving data utility. QuanText efficiently constructs candidate release distributions, randomly selects one close to the private empirical distribution, and rewrites samples using attribute-related snippets to match the chosen distribution. We analyzed its privacy guarantee using Statistic Maximal Leakage and characterized the privacy degradation when textual style is correlated with the secret. Experiments on real-world datasets show that QuanText achieves a favorable privacy–utility trade-off and outperforms differentially private data generation baselines in both privacy and utility.

## REFERENCES

[1] E. M. Bender and B. Friedman, “Data statements for natural language processing: Toward mitigating system bias and enabling better science,” Transactions of the Association for Computational Linguistics, vol. 6, pp. 587–604, 2018.

[2] T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. D. Iii, and K. Crawford, “Datasheets for datasets,” Communications of the ACM, vol. 64, no. 12, pp. 86–92, 2021.

[3] N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson et al., “Extracting training data from large language models,” in USENIX Security Symposium, 2021.

[4] N. Lukas, A. Salem, R. Sim, S. Tople, L. Wutschitz, and S. Zanella-Béguelin, “Analyzing leakage of personally identifiable information in language models,” in 2023 IEEE Symposium on Security and Privacy (SP). IEEE Computer Society, 2023, pp. 346–363.

[5] J. Zhou, Y. Chen, C. Shen, and Y. Zhang, “Property inference attacks against gans,” arXiv preprint arXiv:2111.07608, 2021.

[6] P. Huang, C. Yadav, R. Wu, and K. Chaudhuri, “Can we infer confidential properties of training data from llms?” arXiv preprint arXiv:2506.10364, 2025.

[7] S. Wang, Z. Lin, and G. Fanti, “Statistic maximal leakage,” Entropy, 2026.

[8] Y. Li, Z. Li, K. Zhang, R. Dan, S. Jiang, and Y. Zhang, “Chatdoctor: A medical chat model fine-tuned on a large language model meta-ai (llama) using medical domain knowledge,” Cureus, vol. 15, no. 6, 2023.

[9] S. M. Mohammad, S. Kiritchenko, P. Sobhani, X. Zhu, and C. Cherry, “Semeval-2016 task 6: Detecting stance in tweets,” in Proceedings of the International Workshop on Semantic Evaluation, ser. SemEval ’16, San Diego, California, June 2016.

[10] H. Hu and J. Pang, “Prisampler: Mitigating property inference of diffusion models,” arXiv preprint arXiv:2306.05208, 2023.

[11] L. Wang, J. Wang, J. Wan, L. Long, Z. Yang, and Z. Qin, “Property existence inference against generative models,” in 33rd USENIX Security Symposium (USENIX Security 24), 2024, pp. 2423–2440.

[12] G. Ateniese, L. V. Mancini, A. Spognardi, A. Villani, D. Vitali, and G. Felici, “Hacking smart machines with smarter ones: How to extract meaningful data from machine learning classifiers,” International Journal of Security and Networks, vol. 10, no. 3, pp. 137–150, 2015.

[13] A. Suri, Y. Lu, Y. Chen, and D. Evans, “Dissecting distribution inference,” in 2023 IEEE conference on secure and trustworthy machine learning (saTML). IEEE, 2023, pp. 150–164.

[14] Y. Jiang, X. Luo, Y. Wu, X. Zhu, X. Xiao, and B. C. Ooi, “On data distribution leakage in cross-silo federated learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 7, pp. 3312–3328, 2024.

[15] A.-M. Cre¸tu, F. Guépin, and Y.-A. de Montjoye, “Correlation inference attacks against machine learning models,” Science Advances, vol. 10, no. 28, p. eadj9260, 2024.

[16] C. Chow and C. Liu, “Approximating discrete probability distributions with dependence trees,” IEEE transactions on Information Theory, vol. 14, no. 3, pp. 462–467, 1968.

[17] M. Fredrikson, S. Jha, and T. Ristenpart, “Model inversion attacks that exploit confidence information and basic countermeasures,” in Proceedings of the 22nd ACM SIGSAC conference on computer and communications security, 2015, pp. 1322–1333.

[18] B. Z. H. Zhao, A. Agrawal, C. Coburn, H. J. Asghar, R. Bhaskar, M. A. Kaafar, D. Webb, and P. Dickinson, “On the (in) feasibility of attribute inference attacks on machine learning models,” arXiv preprint arXiv:2103.07101, 2021.

[19] S. Mehnaz, S. V. Dibbo, E. Kabir, N. Li, and E. Bertino, “Are your sensitive attributes private? novel model inversion attribute inference attacks on classification models,” in 31st USENIX security symposium (USENIX Security 22), 2022, pp. 4579–4596.

[20] B. Jayaraman and D. Evans, “Are attribute inference attacks just imputation?” in Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security, 2022, pp. 1569–1582.

[21] V. Duddu and A. Boutet, “Inferring sensitive attributes from model explanations,” in Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022, pp. 416–425.

[22] H. Chaudhari, J. Abascal, A. Oprea, M. Jagielski, F. Tramèr, and J. Ullman, “Snap: Efficient extraction of private properties with poisoning,” in 2023 IEEE Symposium on Security and Privacy (SP). IEEE, 2023, pp. 400–417.

[23] K. Ganju, Q. Wang, W. Yang, C. A. Gunter, and N. Borisov, “Property inference attacks on fully connected neural networks using permutation invariant representations,” in Proceedings of the 2018 ACM SIGSAC conference on computer and communications security, 2018, pp. 619– 633.

[24] A. Suri and D. Evans, “Formalizing and estimating distribution inference risks,” arXiv preprint arXiv:2109.06024, 2021.

[25] W. Zhang, S. Tople, and O. Ohrimenko, “Leakage of dataset properties in {Multi-Party} machine learning,” in 30th USENIX security symposium (USENIX Security 21), 2021, pp. 2687–2704.

[26] V. Hartmann, L. Meynent, M. Peyrard, D. Dimitriadis, S. Tople, and R. West, “Distribution inference risks: Identifying and mitigating sources of leakage,” in 2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2023, pp. 136–149.

[27] J. Stock, J. Wettlaufer, D. Demmler, and H. Federrath, “Lessons learned: defending against property inference attacks,” arXiv preprint arXiv:2205.08821, 2022.

[28] J. Stock, L. Lange, E. Rahm, and H. Federrath, “Property inference as a regression problem: Attacks and defense,” in Proceedings of the International Conference on Security and Cryptography, Bengaluru, India, 2024, pp. 18–19.

[29] T. Shen, T. Lei, R. Barzilay, and T. Jaakkola, “Style transfer from non-parallel text by cross-alignment,” Advances in neural information processing systems, vol. 30, 2017.

[30] Z. Fu, X. Tan, N. Peng, D. Zhao, and R. Yan, “Style transfer in text: Exploration and evaluation,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[31] V. John, L. Mou, H. Bahuleyan, and O. Vechtomova, “Disentangled representation learning for non-parallel text style transfer,” in Proceedings of the 57th annual meeting of the association for computational linguistics, 2019, pp. 424–434.

[32] Z. Lin, S. Gopi, J. Kulkarni, H. Nori, and S. Yekhanin, “Differentially private synthetic data via foundation model APIs 1: Images,” in International Conference on Learning Representations (ICLR), 2024.

[33] C. Xie, Z. Lin, A. Backurs, S. Gopi, D. Yu, H. A. Inan, H. Nori, H. Jiang, H. Zhang, Y. T. Lee et al., “Differentially private synthetic data via foundation model apis 2: Text,” arXiv preprint arXiv:2403.01749, 2024.

[34] Z. Lin, T. Baltrusaitis, W. Wang, and S. Yekhanin, “Differentially private synthetic data via apis 3: Using simulators instead of foundation model,” arXiv preprint arXiv:2502.05505, 2025.

[35] D. Yu, S. Naik, A. Backurs, S. Gopi, H. A. Inan, G. Kamath, J. Kulkarni, Y. T. Lee, A. Manoel, L. Wutschitz et al., “Differentially private finetuning of language models,” arXiv preprint arXiv:2110.06500, 2021.

[36] L. Wutschitz, H. A. Inan, and A. Manoel, “dp-transformers: Training transformer models with differential privacy,” 2022.

[37] X. Yue, H. A. Inan, X. Li, G. Kumar, J. McAnallen, H. Sun, D. Levitan, and R. Sim, “Synthetic text generation with differential privacy: A simple and practical recipe,” in ACL, 2023.

[38] T. Tran, A. Backurs, Z. Lin, V. Reis, L. Xiong, and S. Yekhanin, “Differentially private synthetic data via apis 4: Tabular data,” 2026.

[39] C. Hou, A. Shrivastava, H. Zhan, R. Conway, T. Le, A. Sagar, G. Fanti, and D. Lazar, “Pre-text: training language models on private federated data in the age of llms,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 19 043–19 061.

[40] C. Gong, K. Li, Z. Lin, and T. Wang, “Dpimagebench: A unified benchmark for differentially private image synthesis,” arXiv preprint arXiv:2503.14681, 2025.

[41] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, “Deep learning with differential privacy,” in ACM CCS 2016, 2016, pp. 308–318.

[42] S. Wang, V. Raunak, A. Backurs, V. Reis, P. Zhou, S. Chen, L. Yang, Z. Lin, S. Yekhanin, and G. Fanti, “Struct-bench: A benchmark for differentially private structured text generation,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[43] S. Wang, A. Maddi, Z. Lin, and G. Fanti, “Synae: A framework for measuring the quality of synthetic data for tool-calling agent evaluations,” arXiv preprint arXiv:2605.22564, 2026.

[44] B. Wang, W. Chen, H. Pei, C. Xie, M. Kang, C. Zhang, C. Xu, Z. Xiong, R. Dutta, R. Schaeffer et al., “DecodingTrust: A comprehensive assessment of trustworthiness in GPT models,” in Conference on Neural Information Processing Systems (NeurIPS), 2023.

[45] Z. Lin, S. Wang, V. Sekar, and G. Fanti, “Summary statistic privacy in data sharing,” IEEE Journal on Selected Areas in Information Theory, 2024.

[46] S. Wang, R. Wei, M. Ghassemi, E. Kreacic, and V. K. Potluru, “Guarding multiple secrets: Enhanced summary statistic privacy for data sharing,” arXiv preprint arXiv:2405.13804, 2024.

[47] J. D. Evans, Straightforward statistics for the behavioral sciences. Thomson Brooks/Cole Publishing Co, 1996.

## APPENDIX A

## CANDIDATE DISTRIBUTION CONSTRUCTION

We specify the candidate distribution construction algorithm in Alg. 2. Specifically, for each distribution component P, we draw a pool of L distributions from a symmetric Dirichlet distribution with concentration parameter $\alpha ,$ where the default choice $\alpha = 1$ yields the uniform Dirichlet distribution. We then select $\gamma$ distributions from this pool to form the candidate set, with the goal of making the selected candidate distributions as far apart as possible. To this end, we adopt a greedy farthest-point selection rule: at each step, we choose the distribution that maximizes its minimum TV distance to the set of candidates selected so far. The time complexity of Alg. 2 is $O ( \gamma ^ { 2 } L )$ .

## APPENDIX B

## PROOF OF THM. IV.3

Proof. Let $\mathcal { M } ^ { \prime }$ be the data release mechanism consisting only of the first four steps of QuanText. By [7], we have

$$
\Pi _ { \mathcal { M } ^ { \prime } , \mathfrak { g } } = \operatorname* { s u p } _ { \mathbb { P } _ { \Theta | G } \in \{ 0 , 1 \} } \log \sum _ { \theta ^ { \prime } \in \Theta ^ { \prime } } \operatorname* { s u p } _ { g \in \mathbf { G } } \mathbb { P } _ { \Theta ^ { \prime } | \Theta } \left( \theta ^ { \prime } | \theta _ { g } \right) ,
$$

Algorithm 2: Candidate Distribution Construction   
Input: Distribution component $\mathbb { P } ;$ Dirichlet concentration α;   
pool size $L ;$ number of candidate distributions γ.   
Output: Candidate distribution set Q with size γ.   
// Pool sampling   
1 Initialize pool ${ \mathcal { L } }  { \mathcal { D } } .$   
2 for ℓ ← 1 to L do   
3 Draw $\bar { \mathbb { P } } ^ { ( \ell ) } \sim$ Dirichle $\operatorname { \bar { \rho } _ { s y m } } ( \alpha )$ over the support of P.   
4 ${ \mathcal { L } } \gets { \mathcal { L } } \cup { \bar { \mathbb { P } } } ^ { ( \ell ) }$   
5 end   
// Greedy selection $\scriptscriptstyle \mathrm { \textnormal { \textsf { O E } } } \ \gamma$ candidates   
6 Pick an arbitrary $q _ { 1 } \in { \mathcal { L } }$ and set $\mathcal { Q }  q _ { 1 }$   
7 for $t \gets 2$ to γ do   
8 q<sup>⋆</sup> ← arg max<sub>p∈L\Q</sub> min<sub>q∈Q</sub> d<sub>TV</sub>(p, q).   
9 $\bar { \mathcal { Q } }  \mathcal { Q } \bar { \cup } q ^ { \star }$   
10 end   
11 return Q.

where Θ and $\Theta ^ { \prime }$ denote the distribution parameters of the private and released data respectively, and $\theta _ { g }$ satisfies $\mathbb { P } _ { \Theta | G } \left( \theta _ { g } | g \right) = 1$ . Since the number of distribution components is $\begin{array} { r } { 1 + \sum _ { i = 2 } ^ { m } \left| \phi _ { \pi ( i ) } \right| } \end{array}$ , we have $\begin{array} { r } { | \Theta ^ { \prime } | = \gamma ^ { 1 + \sum _ { i = 2 } ^ { m } \left| \phi _ { \pi ( i ) } \right| } } \end{array}$ and

$$
\mathbb { P } _ { \Theta ^ { \prime } | \Theta } \left( \theta ^ { \prime } | \theta \right) \le \left( \frac { 1 } { k } \right) ^ { 1 + \sum _ { i = 2 } ^ { m } \left| \phi _ { \pi \left( i \right) } \right| } , \quad \forall \theta \in \Theta , \theta ^ { \prime } \in \Theta ^ { \prime } .
$$

Hence, when $L \left( Y ; Z \right) = 0$ , we can get that

$$
\Pi _ { \mathcal { M } ^ { \prime } , \mathfrak { g } } \leq \operatorname* { s u p } _ { \mathbb { P } _ { \Theta | G } \in \{ 0 , 1 \} } \log \sum _ { \theta ^ { \prime } \in \Theta ^ { \prime } } \left( \frac { 1 } { k } \right) ^ { 1 + \sum _ { i = 2 } ^ { m } \left| \phi _ { \pi ( i ) } \right| } = \log \left( \frac { \gamma } { k } \right) ^ { 1 + \sum _ { i = 2 } ^ { m } \left| \phi _ { \pi ( i ) } \right| } .
$$

Since SML satisfies post-processing and, by Thm. IV.2, the attribute-related snippets reveal no information about the secret, the SML of QuanText satisfies

$$
\Pi _ { \mathcal { M } , \mathfrak { g } } \leq \log \left( \frac { \gamma } { k } \right) ^ { 1 + \sum _ { i = 2 } ^ { m } \left| \phi _ { \pi ( i ) } \right| } .
$$

When $L \left( Y ; Z \right) \leq l ,$ under Thm. IV.2, we can get that

$$
\begin{array} { r l } & { \Pi _ { M , \mathfrak { t } , \mathfrak { g } } = \displaystyle \operatorname* { s u p } _ { \mathcal { P } , \ A } \displaystyle \operatorname* { l i m } _ { \mathfrak { t } \in \mathbb { Z } } \frac { \mathfrak { P } ( \hat { G } = G ) } { \operatorname* { s u p } _ { \mathcal { G } \in \mathbb { Z } } \mathbb { P } _ { G } ( g ) } \quad = \displaystyle \operatorname* { s u p } _ { \mathfrak { p } _ { z | \Upsilon } \Gamma ^ { \mathfrak { p } } , \mathfrak { t } , d } \log \frac { \mathbb { P } ( \hat { G } = G ) } { \operatorname* { s u p } _ { \mathcal { G } \in \mathbb { Z } } \mathbb { P } _ { G } ( g ) } } \\ & { = \displaystyle \operatorname* { s u p } _ { \mathcal { Z } _ { | \Upsilon } \mathbb { P } _ { \gamma | G } \in \{ 0 , 1 \} } \log \sum _ { \mathfrak { p ^ { \prime } } \in \mathfrak { e ^ { \prime } } } \operatorname* { s u p } _ { \mathcal { G } } \mathbb { P } \varphi ^ { \prime } | \gamma ( \theta ^ { \prime } | y _ { s } ) } \\ & { \le \displaystyle \operatorname* { s u p } _ { \mathcal { Z } _ { | \Upsilon } \mathbb { P } _ { \gamma | \tau ^ { \mathcal { G } } } \in \{ 0 , 1 \} } \log  \sum _ { \mathfrak { p ^ { \prime } } \in \mathcal { Y } ^ { \prime } , \mathfrak { z } \in \mathcal { L } } \operatorname* { s u p } _ { \mathcal { Z } | \Upsilon } ( z | y _ { g } ) \cdot ( \frac { 1 } { k } ) ^ { 1 + \sum _ { \mathfrak { n } = 2 } ^ { m } | \delta _ { \mathfrak { r } ( \mathfrak { t } ) } | } } \\ &  \le \displaystyle \operatorname* { s u p } _ { \mathcal { Z } _ { | \Upsilon } \mathbb { P } _ { \gamma | \tau ^ { \mathcal { G } } } \in \{ 0 , 1 \} } \log \sum _ { \mathfrak { p ^ { \prime } } \in \mathcal { Y } ^ { \prime } , \mathfrak { z } \in \mathcal { Z } } \varphi \in \mathbb { C } \sum _ { \mathfrak { n } = 2 } ^ { m } | \phi _  \end{array}
$$

where $y _ { g }$ satisfies $\mathbb { P } _ { Y \mid G } \left( y _ { g } | g \right) = 1$ , and $\mathcal { V } ^ { \prime }$ denotes the set of released content parameter vectors. □