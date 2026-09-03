# A Common Measure of Communication for Speech Brain–Computer Interfaces

Dulhan Jayalath Benjamin Ballyk Oiwi Parker Jones

Neural Processing Lab (PNPL ), University of Oxford

## Abstract

Speech brain–computer interfaces (speech BCIs) translate neural activity into language, ofering a path towards restoring speech for people with paralysis and, more broadly, enabling new forms of natural human–computer interaction. Despite this promise, the field lacks a common measure of progress because systems use diferent datasets, recording methods, types of speech, and vo cabularies, so their reported scores are rarely comparable. Underlying this measurement problem are two unresolved questions: (i) what distribution of words should a speech BCI enable a user to communicate, and (ii) how much information from this distribution can a system convey. We address both by deriving open-vocabulary mutual information (OVMI), an information-theoretic quantity that measures the information conveyed by a decoder relative to a reference distribution over the words a user may wish to communicate. This allows capabilities measured under diferent conditions, such as distinct vocabularies, to be evaluated on a common communication scale. We show that ordinarily reported accuracy, word error rate (WER), and other metrics computed only over the words a system supports can overstate how much of a user’s intended speech the system can communicate. We then use OVMI to compare existing systems, expose trade-ofs between how much of the user’s language a system supports and how accurately it decodes those words, show that these comparisons depend on what the user is expected to communicate, and demonstrate that selecting a vocabulary to maximise OVMI yields up to 16.3% relative improvement in accuracy across three speech domains. OVMI therefore provides the speech BCI community with a principled way to compare heterogeneous systems, improve vocabulary design, and measure progress in the field.

OVMI explorer  neural-processing-lab.github.io/OVMI/ Python package § github.com/neural-processing-lab/OVMI

## 1 Introduction

In the latter half of the 1980s, DARPA proposed a speech recognition competition in which diferent systems transcribed the same held-out recordings and were ranked by a single score, settling the longstanding dispute between statistical methods and hand-written rules (Pallett, 2003; Donoho, 2017; Koch and Peterson, 2024). Standardised benchmarks built on this principle, with a shared task, dataset, and metric, have since underpinned progress across much of the machine learning field (Deng et al., 2009; Mnih et al., 2015; Wang et al., 2019). In the same vein, the field of speech BCIs has begun to adopt the same approach, with new benchmarks enabling controlled comparisons of methods that decode neural activity into speech (Willett et al., 2024; Card et al., 2025; Landau et al., 2025; Mantegna et al., 2026a). However, across the wider field, studies cannot all be evaluated with a common benchmark because methods are often specific to diferent experimental settings. Moreover, speech BCIs typically support diferent vocabularies of words, so studies report scores that are conditional on the supported vocabulary. As a result, scores from diferent studies are hard to compare.

If the vocabulary supported by a speech BCI is suficiently large, then it can express most plausible sentences and the choice of supported words matters little. In practice, however, smaller constrained vocabularies are more typical of current systems (Moses et al., 2021; Willett et al., 2023; Özdogan et al., 2025; d’Ascoli et al., 2025; Jayalath and Parker Jones, 2026), where studies construct vocabularies in one of two ways. Some select the most frequent V words from the available data (d’Ascoli et al., 2025; Jayalath and Parker Jones, 2026); others curate a selection of words for a particular use case (Moses et al., 2021). Consequently, two systems with the same vocabulary size and accuracy may support very diferent fractions of what a user wishes to communicate. Conversely, imposing the same vocabulary across datasets would create an unnatural and unfair comparison as those words may occur with very diferent frequencies, or not at all. Thus, neither standardising a universal vocabulary nor using existing study-specific vocabularies provide a satisfactory way to compare studies.

![](images/632513efff71fcfdf385700cd2b1c1b48ecfd4f2b3a207f5551f5a522cc51dca.jpg)  
Figure 1: OVMI maps incomparable evaluations onto a common scale. Speech BCI studies difer in experimental setting, supported vocabulary, and test data, so their reported scores are not comparable. Each study provides a decoding score over its supported vocabulary S, from which we estimate the information conveyed within that vocabulary, I. We define a common reference distribution p(x) over the words a user may wish to communicate, and let C(S) denote the probability that a word from p(x) is supported by the system. OVMI weights the decoded information I by C(S), yielding a score that measures performance against p(x). Evaluating all systems against the same p(x) therefore allows comparing heterogeneous speech BCI studies on a common communication scale.

More fundamentally, no benchmark is able to capture the full range of speech BCI studies. Systems difer in recording method, from intracortical implants to non-invasive MEG and EEG; in speech task, e.g. attempted or perceived speech; in participant population, e.g. healthy volunteers or paralysed patients; and experimental protocol. These diferences often make evaluation on a shared neural dataset impossible Thus, benchmarks provide controlled comparisons of methods within an experimental setting but do not by themselves tell us how capabilities measured in diferent settings relate to one another. For example, an intracortical attempted-speech decoder and a non-invasive perceived-speech decoder cannot be compared as if they were evaluated under the same conditions, yet we may still wish to ask how much communicative capability each demonstrates relative to the same communication objective. This question is also useful retrospectively, because many studies fall outside of the scope of a benchmark, and prospectively, since benchmarks may eventually saturate or be superseded. Therefore, a measure of communication should be meaningful across studies and successive benchmarks without depending on any one benchmark staying relevant.

A benchmark also evaluates speech decoding on a particular language distribution. For example, the 2025 PNPL competition used Sherlock Holmes stories (Landau et al., 2025). Repeated optimisation against such a benchmark may favour methods that decode only this language distribution well. We may instead want to know what a method’s decoding capability measured on a benchmark means relative to other communication distributions, such as conversational or caregiving speech. Collecting a new neural dataset for every such communication objective would be impractical. We therefore need to separate the language distribution used to measure decoding capability from the language distribution against which communication is assessed. Resolving these issues requires answering two underlying questions:

Unanswered questions for measuring progress in speech BCIs

1. What distribution (over words) should a speech BCI enable a user to communicate?

2. How much information from that distribution can a speech BCI convey?

(1) Defining the communication distribution. A speech BCI should ideally allow a user to convey what they would otherwise say aloud. To illustrate the problem, a decoder that is perfectly accurate on a fifty-word vocabulary succeeds only insofar as those fifty words capture what the user wants to say, and a vocabulary suitable for caregiving may be poorly suited to conversation. Moreover, the words a user intends are not uniformly distributed. We therefore model the distribution of words a user wants to communicate as a reference distribution p. This makes the target communication domain explicit and provides a common distribution against which any decoder can be assessed. Thus, the amount of information a decoder conveys depends on both the decoder and the distribution of intended messages.

(2) Measuring communication against a reference distribution. Having specified what a user may wish to communicate through p, we ask how much of that communication a decoder can convey. While accuracy and word error rate (WER) measure decoding fidelity, they remain conditional on the vocabulary supported by the decoder. Moreover, an error rate alone does not quantify how much information has been conveyed as, for example, the same classification accuracy resolves very diferent amounts of uncertainty when choosing among a vocabulary of ten words versus one thousand. Mutual information provides a natural quantity by measuring how much observing the decoder output reduces uncertainty about the intended message.

In BCIs, the standard information-theoretic measure is Wolpaw’s information transfer rate and its refinements (Wolpaw et al., 1998, 2002; Speier et al., 2013). These measures quantify information within a predefined set of symbols. This is natural for classical closed-symbol interfaces, such as character spellers (Farwell and Donchin, 1988), but not when a user may intend words outside of a decoder’s vocabulary. We therefore derive open-vocabulary mutual information (OVMI), which evaluates lexical information when the intended word is drawn from $p .$ As Section 3 shows, OVMI emerges naturally from the decomposition of mutual information where OVMI is in-vocabulary mutual information weighted by the lexical coverage of $p ,$ the probability that an intended word is supported. Evaluating systems against the same $p$ therefore allows heterogeneous studies to be compared on a common scale (Figure 1).

Contributions. We (i) show why conventional measures can overstate communication and derive OVMI, an information-theoretic measure of lexical information transfer relative to an explicit reference communication distribution (Sections 3–4); (ii) use OVMI to compare heterogeneous speech BCI systems on a common scale, revealing how much systems depend on lexical coverage, decoding fidelity, and the intended communication domain (Sections 5.1–5.2); and (iii) show that OVMI can also guide vocabulary selection, improving speech BCI accuracy across three speech domains (Section 5.3).

## 2 Information-Theoretic Evaluation in BCIs

The standard information metric in BCIs is the information transfer rate (ITR) popularised by Wolpaw et al. (1998, 2002) and quoted in recent work (Moses et al., 2021; Perkins et al., 2025; Neuralink Corporation, 2026). Although ITR is reported in bits per minute, Wolpaw’s formulation first computes information per trial and then multiplies by the trial rate. We work at the per-trial level throughout since our primary interest is the information conveyed by each decoding attempt independently of communication speed.

With notation summarised in Appendix A, consider a decoder with a finite vocabulary S of size $V .$ On each trial, the user intends a symbol $X \in S$ , and the decoder outputs a symbol $Y \in S$ . The quantity of interest is the mutual information $I ( X ; Y )$ , which measures how much observing the output $Y$ tells us about the intended symbol X. Wolpaw’s derivation assumes all intended symbols are equally likely, $p ( x ) = 1 / V$ , and the decoder has symmetric errors, meaning it is correct with probability $P$ and otherwise distributes errors uniformly over the remaining $V - 1$ symbols. Under these assumptions (derivation in Appendix B) the mutual information is

$$
I _ { \mathrm { W o l p a w } } ( X ; Y ) = \log _ { 2 } V + P \log _ { 2 } P + ( 1 - P ) \log _ { 2 } \left( { \frac { 1 - P } { V - 1 } } \right) .\tag{1}
$$

The uniform prior assumption is restrictive for natural language, where words are non-uniform and Zipfian (Zipf, 1949). Speier et al. (2013) addressed the problem in P300 character spellers (Farwell and Donchin, 1988) by replacing the uniform prior with empirical character frequencies. More recently, Antonello et al. (2024) proposed a method for information-based evaluation of continuous semantic decoding (Tang et al., 2022). However, these formulations still measure information within a predefined set of symbols and do not account for whether that set can represent all the symbols a user may wish to communicate.

## 3 Open-Vocabulary Mutual Information

A speech BCI can convey an intended word only if the word is supported by the decoder and can be decoded reliably. Conventional accuracy, WER, and in-vocabulary mutual information measure only the latter, whether a decoder can reliably decode words, conditional on the intended word belonging to the decoder vocabulary $S .$ This is appropriate for a classification task, but not for measuring communication when a user may wish to say words outside of the supported vocabulary S.

OVMI therefore evaluates the decoder relative to an external reference distribution p over the words a user may wish to communicate. Its central idea is to weight information conveyed among supported words by the probability that an intended word drawn from $p$ is supported. Thus, two decoders with identical in-vocabulary performance can difer in communicative capability according to OVMI if their vocabularies cover diferent amounts of $p .$ The toy example below illustrates a more extreme case.

Toy example: accuracy and OVMI can rank systems diferently.   
Suppose a user may intend any of 1,000 equally likely words.

<table><tr><td></td><td>System A</td><td>System B</td></tr><tr><td>Vocabulary size</td><td>50</td><td>1,000</td></tr><tr><td>In-vocabulary accuracy</td><td>100%</td><td>50%</td></tr><tr><td>Lexical coverage</td><td>5%</td><td>100%</td></tr><tr><td>OVMI</td><td>0.28 bits</td><td>3.98 bits</td></tr></table>

Despite perfect in-vocabulary decoding, System A can represent only 5% of what the user may wish to say. System B is less accurate but supports every intended word. Accuracy therefore ranks A above B, whereas OVMI ranks B above A by accounting for representability and decoding fidelity.

## 3.1 Formal Definition

Let p be a reference distribution over a lexicon $\Omega ,$ and let $S \subset \Omega$ denote the decoder vocabulary, with $| S | = V$ . For $X \sim p ,$ , define the lexical coverage $\begin{array} { r } { C ( S ) = \mathrm { P r } ( X \in S ) = \sum _ { w \in S } p ( w ) } \end{array}$ , the probability that an intended word is supported by the decoder. For explicitness, we assume that if $X \not \in S$ , then the user abstains from decoding, i.e. does not attempt to think or say the word, and the observable output is $Y = \varnothing$ . If $X \in S$ , the decoder produces some $Y \in S$ . To denote whether the intended word is supported, i.e. whether $X \in S ,$ , we use the indicator $Z = \mathbf { 1 } [ X \in S ] = \mathbf { 1 } [ Y \neq \emptyset ]$

Proposition 1 (Decomposition of mutual information). Under this model,

$$
I ( X ; Y ) = H _ { 2 } ( C ( S ) ) + I ( X ; Y \mid Z ) = H _ { 2 } ( C ( S ) ) + C ( S ) I ( X ; Y \mid X \in S ) ,\tag{2}
$$

where $H _ { 2 } ( p ) = - p \log _ { 2 } p - ( 1 - p ) \log _ { 2 } ( 1 - p )$ is the binary entropy. Proof in Appendix C.

Definition 1 (Open-vocabulary mutual information). We define

$$
I _ { \mathrm { O V M I } } ( S ) : = I ( X ; Y \mid Z ) = C ( S ) I ( X ; Y \mid X \in S ) .\tag{3}
$$

Intuition. How much lexical information is transferred per word the user would want to say?

OVMI is a component of mutual information, stating the information conveyed among supported words, weighted by how often a word drawn from the reference distribution is supported. The remaining term $H _ { 2 } ( C ( S ) )$ reveals whether the intended word lies inside the decoder vocabulary. We exclude it because it contains no information about a word’s identity and counting it would assign information to detecting if a word is in the vocabulary even if the decoder is not able to decode the word.

Relationship to existing metrics. When $C ( S ) = 1$ , OVMI reduces to ordinary mutual information within the decoder vocabulary. If the intended words are additionally uniform and decoding errors are symmetric, it reduces to the per-trial information quantity underlying Wolpaw’s ITR (Wolpaw et al., 1998, 2002). For a trial rate $r , r I _ { \mathrm { O V M I } }$ therefore gives the corresponding open-vocabulary information-transfer rate, with conventional Wolpaw ITR as a special case. Accuracy and WER likewise measure decoding performance within the evaluated vocabulary but do not account for the probability that an intended word is supported. We analyse the impact of this in Section 4.

## 3.2 Estimating OVMI

General estimator. When a full confusion matrix of decoding errors is available, OVMI may be computed from the decoder’s in-vocabulary channel, $K _ { S } ( y \mid x ) = \operatorname* { P r } ( Y = y \mid X = x )$ $K _ { S }$ may be estimated by normalising the rows of the decoder’s confusion matrix. OVMI weights this quantity by the reference distribution restricted to S. We give the full expression in Appendix C.3 with Proposition 2.

Scalar estimator. However, most existing speech BCI studies report only a scalar accuracy or WER, and large confusion matrices are often poorly estimated from limited test data. We therefore use a Wolpaw-like approximation in which every supported word is decoded correctly with probability P and errors are distributed uniformly. We define this next in Corollary 1. We also provide a derivation without this assumption in Appendix C.

Corollary 1 (Scalar accuracy OVMI estimator). Suppose that, for every intended word $x \in S ,$ the decoder outputs the correct word with probability P. Conditional on an error, it distributes the remaining probability uniformly across the other $V - 1$ words. Thus

$$
K _ { S } ( y \mid x ) = \left\{ { P , y = x , } \right.
$$

Under this model, the output distribution on $S$ is

$$
q _ { S } ( j ) = P p _ { S } ( j ) + \frac { 1 - P } { V - 1 } \big ( 1 - p _ { S } ( j ) \big ) , \qquad j \in S .\tag{4}
$$

Consequently (with proof in Appendix C),

$$
I _ { \mathrm { O V M I } } ( S ) = C ( S ) \Big [ H ( q _ { S } ) - h _ { V } ( P ) \Big ] .\tag{5}
$$

Defining P. We instantiate the scalar $P$ in Eq. (5) as macro accuracy $P _ { \mathrm { m a c r o } } ,$ the uniform average of per-word correct-decoding probabilities over $S ,$ rather than as a frequency-weighted (micro) average. The non-uniform source distribution is already modelled by $p$ and enters through C(S), $p _ { S }$ , and the output distribution $q _ { S }$ . Micro accuracy weights classes according to the evaluation set’s frequency distribution, whereas source weighting in OVMI should be determined by the external reference $p .$ Macro accuracy avoids this by expressing how reliably the decoder distinguishes an average symbol in $S ,$ , leaving frequency to enter through the source distribution.

Which estimator to use. We adopt the scalar form (with pseudocode in Appendix D) unless otherwise stated because it admits comparison with prior work that reports only accuracy and vocabulary size. When reliable per-word accuracy estimates are available, OVMI can be computed using the word-specific variant in Appendix $\mathrm { { C . 6 } ; }$ when a reliable empirical confusion matrix is available, it should be computed directly from Proposition 2 (statement and proof in Appendix C). In the non-invasive settings considered here, however, the $O ( | S | ^ { 2 } )$ parameters of a confusion matrix are often poorly estimated under the small, long-tailed held-out sets typical of naturalistic neural recordings (Hamilton and Huth, 2018).

## 4 Common Metrics Overestimate Open-Vocabulary Performance

Before comparing empirical speech decoders, we first isolate the error introduced by the metric itself. Consider a noiseless decoder $( P = 1 )$ whose vocabulary S consists of the top-V most frequent words under the reference distribution. With Wolpaw’s formulation, such a decoder conveys log V bits because the V supported words are assumed to be equally likely. Accounting for their actual, non-uniform frequencies reduces this quantity to the in-vocabulary entropy $H ( p _ { S } )$ , but this assumes that the words a user intends to communicate always lie within S. OVMI corrects this misalignment by accounting for the probability that an intended word is supported at all, giving $C ( S ) H ( p _ { S } )$ with an ideal, noiseless decoder.

Figure 2a shows that these three quantities diverge substantially. Of the three, OVMI is the most conservative measure of information transfer, as $C ( S ) H ( p _ { S } ) \leq H ( p _ { S } ) \leq \log _ { 2 } ( V )$ , with equality on the left when $S$ provides perfect coverage, $C ( S ) = 1$ , of the reference distribution, and equality on the right when $S$ is uniformly distributed. Especially for small vocabularies, open-vocabulary metrics reveal that a system may convey only a small fraction of the information suggested by in-vocabulary entropy.

What causes overestimation? Figure 2b separates the discrepancy into its two sources. For a noiseless decoder, the Wolpaw uniform-prior quantity may be decomposed as

$$
\log _ { 2 } V = \underbrace { C ( S ) H ( p _ { S } ) } _ { \mathrm { O V M I } } + \underbrace { ( 1 - C ( S ) ) H ( p _ { S } ) } _ { \mathrm { C o v e r a g e ~ g a p } } + \underbrace { [ \log _ { 2 } V - H ( p _ { S } ) ] } _ { \mathrm { N o n - u n i f o r m i t y } } .\tag{6}
$$

The first term is OVMI, the second is excess due to evaluating in-vocabulary information without weighting by lexical coverage, and the third is error introduced by treating the supported words as uniformly distributed. At small V, in-vocabulary entropy and the uniform prior vastly overestimate information transfer, driven by incomplete lexical coverage, $C ( S ) \ll 1$ , as even a perfectly accurate decoder cannot communicate an intended word that it does not support. As V increases, coverage approaches unity and OVMI approaches in-vocabulary entropy, however, the gap to the uniform-prior, $\log _ { 2 } ( V )$ , continues to grow due to the Zipfian distribution of words in the reference $p .$ Invasive systems (Moses et al., 2021; Willett et al., 2023) and recent non-invasive decoders (d’Ascoli et al., 2025; Jayalath and Parker Jones, 2026) use vocabularies of 50–250 Zipfian-distributed words, where both sources of overestimation are substantial.

![](images/f70897f272434a204ef1ea0db615fb9f6b5972ed67dfabac71d6b1d555a780d4.jpg)

b  
![](images/7a42d2e1b478ad443f1f001c5c4dd88d9b3aaa8c6a28b0cbfe6dc94ed41368fb.jpg)  
Figure 2: In-vocabulary measures overestimate open communication. (a) Information attributed to a noiseless decoder $( P = 1 )$ as the vocabulary size V increases, with vocabularies formed from the top $V$ most frequent words in p. Wolpaw (uniform prior) assigns log V bits, accounting for non-uniform word frequencies gives $H ( p _ { S } )$ , and OVMI further accounts for coverage, yielding $C ( S ) H ( p _ { S } )$ . The distribution $p$ is computed from a spoken-language frequency norm derived from film and television subtitles (van Heuven et al., 2014), representing broad spoken English. (b) Decomposition of the uniform-prior quantity, log $V =$ $C ( S ) H ( p s ) + ( 1 - C ( S ) ) H ( p s ) + [ \log _ { 2 } V - H ( p s ) ]$ , into OVMI, missing coverage, and the uniform-prior assumption.

Accuracy and WER overstate communication. For a given vocabulary, Wolpaw’s quantity is a monotonic transformation of in-vocabulary accuracy. Hence, accuracy inherits the issue of conditioning on this vocabulary. Under our model, the probability of decoding an intended word is instead

$$
\operatorname* { P r } ( { \hat { X } } = X ) = \operatorname* { P r } ( X \in S ) \operatorname* { P r } ( { \hat { X } } = X \mid X \in S ) = C ( S ) P _ { \mathrm { i n - v o c a b } } .
$$

As a result, even perfect in-vocabulary accuracy can correspond to poor open-vocabulary communication when lexical coverage is low. Likewise, WER measures decoding errors on an evaluation transcript but does not by itself account for how much of an external reference distribution that system can represent. Consequently, high accuracy or low WER can occur at the same time as low open-vocabulary communication when the vocabulary does not cover much of what a user may wish to communicate.

## 5 Results

We now use OVMI to evaluate existing speech BCI systems relative to an explicit communication distribution. Unless otherwise stated, we use SUBTLEX-UK (van Heuven et al., 2014) as $p ,$ treating its word frequencies from film and television subtitles as a broad spoken English reference. We first compare heterogeneous speech decoders with OVMI (Section 5.1), then examine how the comparison changes with p (Section 5.2), and finally test OVMI as an objective for vocabulary selection (Section 5.3). We provide full details on estimating OVMI in Appendix F.

## 5.1 A Common Scale for Heterogeneous Speech BCIs

We compare speech decoding systems that difer in vocabulary, dataset, task, and recording modality by evaluating each against the same reference distribution. For invasive work, the dataset and the decoder are typically reported together. We consider three landmark invasive studies (Moses et al., 2021; Willett et al., 2023; Card et al., 2024) which decode attempted speech from partially paralysed patients. For non-invasive work, datasets are often released independently and subsequently evaluated with multiple decoding meth ods, so we use the strongest evaluated decoder for each dataset. For MEG-MASC (Gwilliams et al., 2023), we train MEG-XL (Jayalath and Parker Jones, 2026), while for the other MEG datasets—LibriBrain100 (Mantegna et al., 2026b) and Armeni (Armeni et al., 2022)—we train the decoder of d’Ascoli et al. (2025). Tang et al. (2022) is included as a study in which the dataset and decoding method are reported together. In these datasets, participants listen to or view naturalistic stimuli such as movies or spoken narratives.

![](images/9ba1c7177e0cdc1c349131f31cd85683a429b6b4ca55170bef8298674639ea59.jpg)  
Lexical coverage C(S)

Figure 3: Speech BCIs lose information through lexical coverage or decoding fidelity. The product of the horizontal and vertical axes gives OVMI, with diagonal contours indicating equal OVMI. The dashed curve shows a noiseless decoder whose vocabulary consists of the top-V most frequent words under the reference distribution. For frequency-selected vocabularies, vertical displacement from the noiseless curve reflects decoding error while movement leftwards reflects information foregone due to the limited coverage of smaller vocabularies.  
![](images/998edb3b70df642311a90029d2ee9a75d2f903def027bae449bfd714de2c37eb.jpg)

![](images/bb22234b9908f60d56d21f88ecb3d5bf53af279a91ebd3a57348105f82a077ec.jpg)  
Figure 4: Speech-BCI progress measured on a common open-vocabulary scale. Reported speechdecoding systems are plotted by publication year using OVMI normalised by the entropy of the reference distribution. We include prior systems which state a vocabulary S and an accuracy or WER suficient to estimate OVMI. The solid line connects the highest-scoring evaluated invasive system in each publication year. Non-invasive results are unconnected because they correspond to diferent datasets rather than successive methods evaluated under a common experimental setting. Values provide a retrospective comparison on a common communication scale. An extended tabular version of these results with uncertainties is available in Appendix E.

Where is information lost? Figure 3 separates the two factors determining OVMI. The 50-word invasive systems (Moses et al., 2021; Willett et al., 2023) convey comparatively high information within their supported vocabularies but cover only a small fraction of broad spoken English. Most evaluated non-invasive systems achieve greater coverage through frequency-selected vocabularies, but substantially lower in-vocabulary information. The later 125k-word invasive systems (Willett et al., 2023; Card et al., 2024) occupy the upper-right of the plot, combining near-complete coverage with high decoding fidelity. For frequency-selected vocabularies, displacement below the noiseless frontier reflects information lost through decoding error, while movement leftwards reflects limited lexical coverage. Curated vocabularies do not need to lie below this frontier because they may trade coverage for greater in-vocabulary entropy. The large-vocabulary invasive systems improve primarily by expanding coverage after decoding fidelity had already become strong, whereas the evaluated non-invasive systems remain limited by decoding fidelity, likely as a consequence of the lower signal fidelity aforded by sensors outside the skull.

Understanding the progress of systems over time. Figure 4 shows that in 2021, the 50-word Moses system conveys 2.4% of the lexical information available in its isolated-word setting, increasing to 4.7% with language-model post-processing. The 50-word Willett results in 2023 remain in the same regime, conveying 6.7% and 6.4% respectively. The major change comes from increasing the vocabulary with Willett’s 125k-word system reaching 72.0%, followed by Card at 93.7% in 2024. Thus, the largest historical increase among the invasive systems occurs with the transition to large-vocabulary decoding, after in-vocabulary decoding fidelity had already approached saturation at smaller vocabulary sizes.

![](images/0783e0766b09a3ba36717db6841e65526a5d3b41c7eafe3babac54ca94eb0377.jpg)  
Figure 5: Speech BCI performance depends on the intended communication domain. Each cell reports OVMI normalised by the entropy of the corresponding reference distribution, as a percentage, with uncertainty beneath. Brackets denote propagated intervals for published invasive results, whereas ± denotes one-SEM uncertainty for Tang and LibriBrain100. Armeni and MEG-MASC are excluded for brevity. Systems are evaluated against four communication references spanning broad spoken English (SUBTLEX-UK; van Heuven et al., 2014), conversational speech (Switchboard; Godfrey et al., 1992), augmentative and alternative communication (AAC/UCV; Erickson et al., 2019), and narrative speech (Sherlock Holmes; Doyle, 1892).

Non-invasive word decoding remains in a lower information regime on this scale. The d’Ascoli et al. (2025) decoder yields 2.4% on LibriBrain100 and 1.8% on Armeni, MEG-XL yields 0.3% on MEG-MASC, and Tang’s fMRI system reaches 3.6%. The particularly low MEG-MASC value occurs in a challenging regime with little data per participant across a large and heterogeneous subject population. For perceived speech, OVMI should be interpreted as lexical decoding capability relative to the same speech distribution, rather than communication. When comparing across settings, it is important to note the underlying paradigm; for example, perceived speech decoding does not by itself imply a useful communication interface.

## 5.2 Information Transfer Depends on the Communication Distribution

OVMI is meaningful only relative to a specified communication target. Figure 5 evaluates the same systems against four reference distributions representing broad spoken English (SUBTLEX-UK), conversation (Switchboard), augmentative and alternative communication (AAC; UCV), and narrative speech (Sherlock). Because these distributions have diferent entropies, we normalise OVMI by $H ( p )$

If information transfer were a property of the decoder alone, then there would be no diference between the columns. Instead, we find that large-vocabulary systems are comparatively insensitive to the choice of p as Card conveys 93.7–97.5% across the four references and the 125k-word Willett system 72.0–76.2%. However, systems with smaller vocabularies are much more sensitive. The 50-word Willett system, for example, conveys 6.4% of broad spoken-English information but 40.4% under the AAC reference and Moses similarly increases from 4.7% to 30.7%. These systems use vocabularies designed around clinical communication, and consequently cover a much larger proportion of the more restricted AAC distribution than of unrestricted speech. LibriBrain100 and the isolated-word Moses system are nearly matched under broad speech (2.43% versus 2.4%), but Moses leads substantially under AAC (16.5% versus 7.34%), whereas LibriBrain100 leads under narrative speech (2.62% versus 1.0%). Whether one system represents progress over another can therefore depend on what the interface is intended to communicate. Accordingly, OVMI should always be reported together with p and when the intended application is uncertain, several plausible reference distributions can be reported as we do here.

## 5.3 OVMI for Vocabulary Selection

Up to this point, we have used OVMI retrospectively to evaluate prior studies. We next ask whether it can also help choose the vocabulary of a system. Recent contrastive decoders restrict a larger candidate lexicon to a smaller vocabulary at inference to improve accuracy (d’Ascoli et al., 2025; Jayalath and Parker Jones, 2026), creating a trade-of between supporting frequent or important words and retaining words that are easier to decode. OVMI provides a natural objective because it accounts for both of these factors.

We test OVMI as a selection objective using the d’Ascoli et al. (2025) contrastive decoder. The model is trained once with a candidate lexicon of 250 words. At inference, decoding requires retrieving the most likely word from a set of candidate embeddings. For each vocabulary size $V ,$ we select a subset $S _ { V }$ and restrict retrieval to that subset, where varying V does not require any changes to the trained decoder. Figure 6 shows the results of selecting vocabularies of varying size according to frequency, validation accuracy, or OVMI. We repeat this for three diferent communication distributions: TIMIT, which has sentences designed to cover a diverse range of phonetic contexts, a set of podcast conversations, and a Sherlock Holmes book chapter. We measure the resulting test accuracy and find that OVMI matches or exceeds all alternatives across vocabulary sizes. It improves results over frequency selection, the strategy used in recent non-invasive decoders (d’Ascoli et al., 2025; Jayalath et al., 2025; Jayalath and Parker Jones, 2026), at smaller vocabulary sizes, with peak relative improvements of 15.4%, 16.3%, and 8.4%. The advantage diminishes at larger vocabulary sizes as expected when the choice of words becomes less restricted.

![](images/f21aa2222bd0432eb8237185590ac9bae95626ea85df9c9ecbcb6b772cdd6738.jpg)  
Figure 6: OVMI-guided vocabulary selection improves accuracy. With a decoder trained on LibriBrain100, we select vocabularies for three of its diferent test domains (TIMIT/Podcasts/Sherlock) according to (i) frequency in the domain training data, (ii) highest top-1 accuracy on the pooled validation data, (iii) maximising OVMI on the validation data with domain training data as p, and (iv) randomly. (Top) Held-out word accuracy, counting words outside the vocabulary as incorrect. Shading denotes the standard error across five seeds. At V = 250, all methods converge because the candidate pool contains 250 words so we omit this trivial endpoint. (Bottom) Mean relative improvement of OVMI over frequency. Stars show statistical significance according to one-sided paired permutation tests $( ^ { * } p < . 0 5 , ^ { * * } p < . 0 1 , ^ { * * * } p < . 0 0 1 )$ . More details in Appendix G.

## 6 Discussion

Scope and limitations. Three caveats bound this work. (i) Our retrospective comparisons rely on the scalar estimator because confusion matrices are rarely reported. When richer statistics are available, OVMI can be computed directly from the empirical confusion matrix (Proposition 2) or from per-word accuracies (Appendix C.6). (ii) OVMI penalises out-of-vocabulary words as unsupported, however, the meaning could be expressed through paraphrasing. We choose to keep the measure tied to the lexical information explicitly available through the decoder vocabulary. (iii) We evaluate lexical rather than contextual information; extending OVMI to a conditional language distribution is left to future work.

<table><tr><td>Question</td><td>Evaluation type</td><td>Cost</td></tr><tr><td>Which method performs best in this setting?</td><td>Benchmark</td><td>Medium</td></tr><tr><td>What does that capability mean under  $p ?$ </td><td>OVMI</td><td>Low</td></tr><tr><td>Is the system practically useful?</td><td>User-centred</td><td>High</td></tr></table>

By evaluating capabilities measured in diferent experimental settings against the same reference distribution, OVMI places heterogeneous speech BCI systems on a common communication scale. It complements controlled benchmarks as benchmarks compare methods within a fixed setting, whereas OVMI relates capabilities across settings to the same communication objective. OVMI can also guide vocabulary design when the supported words are controllable, as our vocabulary selection experiments demonstrate. We note that the comparisons in this work do not erase diferences in experimental setting, and a higher OVMI does not necessarily imply a more practical or clinically useful paradigm; for example, perceived speech decoding on its own is unlikely to constitute a usable communication interface. We recommend reporting OVMI alongside standard decoder metrics for future ease of comparison, and encourage user-centred evaluation before any practical deployment. We hope that OVMI’s lasting role is to give the emerging field of speech BCIs a principled way to compare and improve diferent systems.

## Acknowledgments

DJ would like to thank Charles London for his assistance with checking proofs and derivations, as well as Gilad Landau, Tasha Kim, Miran Özdogan, and Mélanie Schneider for reviewing early drafts of this work.

We would like to acknowledge the use of the University of Oxford Advanced Research Computing (ARC) facility in carrying out this work. http://dx.doi.org/10.5281/zenodo.22558. We are especially grateful to the ARC support team for their timely support as conference deadlines approached.

We also thank Modal Labs, Inc. for a generous compute grant which helped support this project.

DJ is supported by an AWS Studentship from the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems (AIMS) (EP/S024050/1). OPJ and the PNPL group are supported by the MRC (MR/X00757X/1), Royal Society (RG\R1\241267), NSF (2314493), NFRF (NFRFT-2022- 00241), SSHRC (895-2023-1022), and ARIA (SCNI-SE01-P004).

## References

Richard Antonello, Nihita Sarma, Jerry Tang, Jiaru Song, and Alexander Huth. How many bytes can you take out of brain-to-text decoding? arXiv preprint arXiv:2405.14055, 2024.

Kristijan Armeni, Umut Güçlü, Marcel van Gerven, and Jan-Mathijs Schofelen. A 10-hour withinparticipant magnetoencephalography narrative dataset to test models of language comprehension. Scientific Data, 9(1):278, 2022. License: CC-BY-4.0.

Nicholas S. Card, Maitreyee Wairagkar, Carrina Iacobacci, Xianda Hou, Tyler Singer-Clark, Francis R. Willett, Erin M. Kunz, Chaofei Fan, Maryam Vahdati Nia, Darrel R. Deo, Aparna Srinivasan, Eun Young Choi, Matthew F Glasser, Leigh R. Hochberg, Jaimie M. Henderson, Kiarash Shahlaie, Sergey D. Stavisky, and David M. Brandman. An accurate and rapidly calibrating speech neuroprosthesis. The New England Journal of Medicine, 391 7:609–618, 2024.

Nicholas S. Card, Maitreyee Wairagkar, Carrina Iacobacci, Xianda Hou, Tyler Singer-Clark, Francis R. Willett, Erin M. Kunz, Chaofei Fan, Maryam Vahdati Nia, Darrel R. Deo, Aparna Srinivasan, Eun Young Choi, Matthew F. Glasser, Leigh R. Hochberg, Jaimie M. Henderson, Kiarash Shahlaie, Sergey D. Stavisky, and David M. Brandman. Brain-to-text ’25. https://kaggle.com/competitions brain-to-text-25, 2025. Kaggle.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA, pages 248–255. IEEE Computer Society, 2009.

David L. Donoho. 50 years of Data Science. Journal of Computational and Graphical Statistics, 26:745 – 766, 2017.

Arthur Conan Doyle. The Adventures of Sherlock Holmes. 1892.

Stéphane d’Ascoli, Corentin Bel, Jérémy Rapin, Hubert J. Banville, Yohann Benchetrit, Christophe Pallier, and Jean-Rémi King. Towards decoding individual words from non-invasive brain recordings. Nature Communications, 16, 2025.

Karen Erickson, Lori Geist, Penny Hatch, and Nancy Quick. The universal core vocabulary. Technical report, Center for Literacy & Disability Studies, University of North Carolina at Chapel Hill, Chapel Hill, NC, 2019.

Lawrence A. Farwell and Emanuel Donchin. Talking of the top of your head: Toward a mental prosthesis utilizing event-related brain potentials. Electroencephalography and Clinical Neurophysiology, 70 6: 510–23, 1988.

John J. Godfrey, Edward Holliman, and J. McDaniel. SWITCHBOARD: Telephone speech corpus for research and development. [Proceedings] ICASSP-92: 1992 IEEE International Conference on Acoustics, Speech, and Signal Processing, 1:517–520 vol.1, 1992.

Laura Gwilliams, Graham Flick, Alec Marantz, Liina Pylkkänen, David Poeppel, and Jean-Rémi King. Introducing MEG-MASC a high-quality magneto-encephalography dataset for evaluating natural speech processing. Scientific Data, 10(1):862, 2023. License: CC0 1.0 Universal.

Liberty S. Hamilton and Alexander G. Huth. The revolution will not be controlled: Natural stimuli in speech neuroscience. Language, Cognition and Neuroscience, 35:573 – 582, 2018.

Dulhan Jayalath and Oiwi Parker Jones. MEG-XL: Data-eficient brain-to-text via long-context pre-training. International Conference on Machine Learning (ICML), 2026. arXiv preprint arXiv:2602.02494.

Dulhan Jayalath, Gilad Landau, and Oiwi Parker Jones. Unlocking non-invasive brain-to-text. International Conference on Machine Learning (ICML), Workshop on Generative AI and Biology, 2025. arXiv preprint arXiv:2505.13446.

Bernard J Koch and David Peterson. From protoscience to epistemic monoculture: How benchmarking set the stage for the deep learning revolution. arXiv preprint arXiv:2404.06647, 2024.

Gilad Landau, Miran Özdogan, Gereon Elvers, Francesco Mantegna, Pratik Somaiya, Dulhan Jayalath, Luisa Kurth, Teyun Kwon, Brendan Shillingford, Greg Farquhar, Minqi Jiang, Karim Jerbi, Hamza Abdelhedi, Yorguin Mantilla Ramos, Caglar Gulcehre, Mark Woolrich, Natalie Voets, and Oiwi Parker Jones. The 2025 PNPL competition: Speech detection and phoneme classification in the LibriBrain dataset. Advances in Neural Information Processing Systems (NeurIPS), Competition Track, 2025. arXiv preprint arXiv:2506.10165.

Francesco Mantegna, Gereon Elvers, Dulhan Jayalath, Gilad Landau, Tasha Kim, Miran Özdogan, Luisa Kurth, Teyun Kwon, SungJun Cho, Benjamin Ballyk, Alex Fung, Anna Greer, Pratik Somaiya, Christian Herf, Yorguin Mantilla Ramos, Hamza Abdelhedi, Karim Jerbi, Greg Farquhar, Brendan Shillingford, Mark Woolrich, and Oiwi Parker Jones. The 2026 PNPL competition: Word classification and eficient cross-subject generalisation in LibriBrain100, 2026a. Manuscript in preparation.

Francesco Mantegna, Dulhan Jayalath, Gereon Elvers, Tasha Kim, Benjamin Ballyk, Alex Fung, SungJun Cho, Teyun Kwon, Luisa Kurth, Miran Özdogan, Gilad Landau, Pratik Somaiya, Natalie Voets, Mark Woolrich, and Oiwi Parker Jones. LibriBrain100: One hundred hours of broad and deep MEG data for neural speech decoding at scale. arXiv preprint arXiv:2608.25204, 2026b.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin A. Riedmiller, Andreas Kirkeby Fidjeland, Georg Ostrovski, Stig Petersen, Charlie Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. Nature, 518: 529–533, 2015.

David Aaron Moses, Sean L. Metzger, Jessie R. Liu, Gopala Krishna Anumanchipalli, Joseph G. Makin, Pengfei F Sun, Josh Chartier, Maximilian E. Dougherty, Patricia M Liu, Gary M. Abrams, Adelyn P. Tu-Chan, Karunesh Ganguly, and Edward F. Chang. Neuroprosthesis for decoding speech in a paralyzed person with anarthria. The New England Journal of Medicine, 385 3:217–227, 2021.

Neuralink Corporation. Two years of telepathy (article). https://neuralink.com/updates/ two-years-of-telepathy/, 2026. Accessed: 2026-04-20.

David S. Pallett. A look at NIST’s benchmark ASR tests: past, present, and future. 2003 IEEE Workshop on Automatic Speech Recognition and Understanding (IEEE Cat. No.03EX721), pages 483–488, 2003.

Sean M Perkins, Michael Trumpis, Michael E Reitman, Beata Jarosiewicz, Aashish N Patel, Adam Weiss, Jacob W Scott, Kurtis Nishimura, Matthew R Angle, Shaoyu Qiao, et al. SONIC: A benchmarking paradigm for brain-computer interfaces. bioRxiv preprint 2025.09.30.679683, 2025.

W. Speier, Corey W. Arnold, and Nader Pouratian. Evaluating true BCI communication rate through mutual information and language models. PLoS ONE, 8, 2013.

Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth. Semantic reconstruction of continuous language from non-invasive brain recordings. Nature Neuroscience, 26:858–866, 2022.

Walter J. B. van Heuven, Paweł Mandera, Emmanuel Keuleers, and Marc Brysbaert. SUBTLEX-UK: A new and improved word frequency database for British English. Quarterly Journal of Experimenta Psychology, 67:1176 – 1190, 2014.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019.

Francis R. Willett, Erin M. Kunz, Chaofei Fan, Donald T. Avansino, Guy H. Wilson, Eun Young Choi, Foram B. Kamdar, Matthew F. Glasser, Leigh R. Hochberg, Shaul Druckmann, Krishna V. Shenoy, and Jaimie M. Henderson. A high-performance speech neuroprosthesis. Nature, 620:1031 – 1036, 2023.

Francis R Willett, Jingyuan Li, Trung Le, Chaofei Fan, Mingfei Chen, Eli Shlizerman, Yue Chen, Xin Zheng, Tatsuo S Okubo, Tyler Benster, et al. Brain-to-Text benchmark ’24: Lessons learned. arXiv preprint arXiv:2412.17227, 2024.

Jonathan R. Wolpaw, Herbert Ramoser, Dennis J. McFarland, and Gert Pfurtscheller. EEG-based communication: improved accuracy by response verification. IEEE Transactions on Rehabilitation Engineering, 6 3:326–33, 1998.

Jonathan R. Wolpaw, Niels Birbaumer, Dennis J. McFarland, Gert Pfurtscheller, and Theresa M. Vaughan. Brain-computer interfaces for communication and control. Clinical Neurophysiology : Oficial Journal of the International Federation of Clinical Neurophysiology, 113 6:767–91, 2002.

George Kingsley Zipf. Human behavior and the principle of least efort. 1949.

Miran Özdogan, Gilad Landau, Gereon Elvers, Dulhan Jayalath, Pratik Somaiya, Francesco Mantegna, Mark Woolrich, and Oiwi Parker Jones. LibriBrain: Over 50 hours of within-subject MEG to improve speech decoding methods at scale. Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. arXiv preprint arXiv:2506.02098.

A Table of Variables 14   
B Derivation of Wolpaw’s Information Transfer Rate 14   
C Proofs and Supplementary Derivations for OVMI 15   
C.1 Proof of Proposition 1 15   
C.2 Equivalent Entropy Derivation of Proposition 1 16   
C.3 Proposition 2 16   
C.4 Proof of Proposition 2 17   
C.5 Proof of Corollary 1 17   
C.6 Retaining Per-Word Accuracies in OVMI 18   
C.7 Further Observations for the Homogeneous Symmetric Channel 18   
D Closed Form OVMI Pseudocode 19   
E Table of Results 19   
F Cross-Study OVMI Estimation Details 19   
F.1 Estimation from a Scalar Accuracy 20   
F.2 Conversion of Reported Performance Measures 20   
F.3 Reference Distributions 21   
F.4 Normalisation and Uncertainty 21   
F.5 Large-Vocabulary Invasive Systems 21   
G Vocabulary Optimisation Experiment Details 22

## A Table of Variables

Table 1 summarises the core variables introduced in Section 2 and 3.

Table 1: Core variables in the open-vocabulary mutual information formulation. Section, proposition, corollary, and equation references point to where each symbol is first introduced.
<table><tr><td>Symbol Description</td><td></td><td>Introduced in</td></tr><tr><td> $\Omega$ </td><td>Reference lexicon.</td><td>Section 3</td></tr><tr><td> $p$ </td><td>Reference distribution over  $\Omega .$ </td><td>Section 3</td></tr><tr><td>S</td><td>Decoder vocabulary,  $S \subset \Omega .$ </td><td>Section 3</td></tr><tr><td>V</td><td>Decoder vocabulary size,  $V = | S | .$ </td><td>Section 3</td></tr><tr><td> $X$ </td><td>User&#x27;s intended word drawn from the reference distribution,  $X \sim p .$ </td><td>Section 3</td></tr><tr><td> $Y$ </td><td>Decoder output,  $Y \in S \cup \{ \varnothing \} ; Y = \varnothing$  on abstention.</td><td>Section 3</td></tr><tr><td> $Z$ </td><td>In-vocabulary indicator,  $Z = \mathbf { 1 } [ X \in S ] = \mathbf { 1 } [ Y \neq \emptyset ] .$ </td><td>Section 3</td></tr><tr><td> $C ( S )$ </td><td>Lexical coverage of S under  $p , ~ C ( S ) ~ = ~ \operatorname* { P r } ( X ~ \in ~ S ) ~ =$   $\textstyle \sum _ { w \in S } p ( w )$ </td><td>Section 3</td></tr><tr><td> $p _ { S }$ </td><td>In-vocabulary target distribution,  $p _ { S } ( x ) = p ( x ) / C ( S )$   $x \in S .$ </td><td>for Proposition 2</td></tr><tr><td> $K _ { S }$ </td><td>In-vocabulary channel,  $K _ { S } ( y \mid x ) = \operatorname* { P r } ( Y = y \mid X = x )$  for  $x , y \in S .$ </td><td>Proposition 2</td></tr><tr><td> $q _ { S }$ </td><td>In-vocabulary outputdistribution,  $q _ { S } ( y )$  二  $\textstyle \sum _ { x \in S } p _ { S } ( x ) K _ { S } ( y \mid x )$ </td><td>Proposition 2</td></tr><tr><td> $P$ </td><td>Per-symbol correct-decoding probability under the homoge-</td><td>Corollary 1</td></tr><tr><td> $h _ { V } ( u )$ </td><td>neous symmetric channel. Entropy of a V-ary symmetric channel row with correct Equation (13)</td><td></td></tr><tr><td> $I _ { \mathrm { O V M I } } ( S )$ </td><td>probability u. Open-vocabulary mutual information, Iovm1(S) = Def. 1, Eq. (3)  $C ( S ) I ( X ; Y \mid X \in S )$ </td><td></td></tr></table>

## B Derivation of Wolpaw’s Information Transfer Rate

Under Wolpaw’s standard derivation, two simplifying assumptions are made. First, all intended symbols are equally likely, so $p ( x ) = 1 / V$ . Second, the decoder is assumed to have symmetric errors: it is correct with probability P, and when it is wrong, each of the other $V - 1$ symbols is equally likely. Thus

$$
\operatorname* { P r } ( Y = y \mid X = x ) = { \left\{ \begin{array} { l l } { P , } & { y = x , } \\ { 1 - P } \\ { { \overline { { V - 1 } } } , } & { y \neq x . } \end{array} \right. }
$$

These assumptions make the calculation straightforward. Fix any output symbol $y .$ There is exactly one intended symbol, namely $x = y$ , that produces y as the correct output, contributing probability mass $P .$ The remaining $V - 1$ intended symbols can also produce the same y, but only as an error, and each contributes $( 1 - P ) / ( V - 1 )$ . Averaging over the uniform prior therefore gives

$$
p ( y ) = \sum _ { x \in S } p ( y \mid x ) p ( x ) = { \frac { 1 } { V } } \left[ P + ( V - 1 ) { \frac { 1 - P } { V - 1 } } \right] = { \frac { 1 } { V } } .
$$

So the decoder output Y is itself uniform over the vocabulary, and hence

$$
H ( Y ) = \log _ { 2 } { V } .
$$

Next consider the conditional entropy. Once we fix an intended symbol $x ,$ the conditional distribution of $Y$ has one outcome with probability $P$ (the correct symbol) and $V - 1$ outcomes with probability $( 1 - P ) / ( V - 1 )$ (the incorrect symbols). Therefore

$$
H ( Y \mid X = x ) = - P \log _ { 2 } P - ( V - 1 ) { \frac { 1 - P } { V - 1 } } \log _ { 2 } \left( { \frac { 1 - P } { V - 1 } } \right) = - P \log _ { 2 } P - ( 1 - P ) \log _ { 2 } \left( { \frac { 1 - P } { V - 1 } } \right) .
$$

Because this expression does not depend on $x ,$ averaging over x leaves it unchanged:

$$
H ( Y \mid X ) = - P \log _ { 2 } P - ( 1 - P ) \log _ { 2 } \left( { \frac { 1 - P } { V - 1 } } \right) .
$$

Substituting these two terms into $I ( X ; Y ) = H ( Y ) - H ( Y \mid X )$ yields the Wolpaw expression for mutual information in an in-vocabulary decoder:

$$
I _ { \mathrm { W o l p a w } } ( X ; Y ) = \log _ { 2 } V + P \log _ { 2 } P + ( 1 - P ) \log _ { 2 } \left( { \frac { 1 - P } { V - 1 } } \right) .\tag{7}
$$

## C Proofs and Supplementary Derivations for OVMI

We use the convention $0 \log _ { 2 } 0 = 0$

## C.1 Proof of Proposition 1

Proof of Proposition 1. Recall that

$$
Z = \mathbf { 1 } [ X \in S ] = \mathbf { 1 } [ Y \neq \emptyset ] .
$$

Since $Z$ is a deterministic function of Y, adjoining Z to the output does not change the mutual information:

$$
I ( X ; Y ) = I ( X ; Y , Z ) .
$$

Applying the chain rule for mutual information gives

$$
I ( X ; Y , Z ) = I ( X ; Z ) + I ( X ; Y \mid Z ) .
$$

Because Z is also a deterministic function of X,

$$
I ( X ; Z ) = H ( Z ) - H ( Z \mid X ) = H ( Z ) .
$$

Now $\operatorname* { P r } ( Z = 1 ) = C ( S )$ and $\operatorname* { P r } ( Z = 0 ) = 1 - C ( S )$ , so

$$
H ( Z ) = H _ { 2 } ( C ( S ) ) .
$$

It remains to expand the conditional mutual information:

$$
I ( X ; Y \mid Z ) = \sum _ { z \in \{ 0 , 1 \} } \operatorname* { P r } ( Z = z ) I ( X ; Y \mid Z = z ) .
$$

Under our model, $Z = 0$ implies $Y = \varnothing$ deterministically, hence

$$
I ( X ; Y \mid Z = 0 ) = 0 .
$$

Also, $Z = 1$ is exactly the event $X \in S .$ , so

$$
I ( X ; Y \mid Z = 1 ) = I ( X ; Y \mid X \in S ) .
$$

Therefore

$$
I ( X ; Y \mid Z ) = C ( S ) I ( X ; Y \mid X \in S ) .
$$

Combining the pieces yields

$$
I ( X ; Y ) = H _ { 2 } ( C ( S ) ) + I ( X ; Y \mid Z ) = H _ { 2 } ( C ( S ) ) + C ( S ) I ( X ; Y \mid X \in S ) ,
$$

as claimed.

## C.2 Equivalent Entropy Derivation of Proposition 1

Proposition 1 can also be obtained by decomposing $H ( Y )$ and $H ( Y \mid X )$

Because $Z = { \bf 1 } [ Y \neq \infty ]$ is a deterministic function of $Y _ { ; }$ the chain rule gives

$$
H ( Y ) = H ( Z ) + H ( Y \mid Z ) .
$$

Expanding by the two values of $Z ,$

$$
{ \begin{array} { r l } & { H ( Y ) = H _ { 2 } ( C ( S ) ) + \operatorname* { P r } ( Z = 1 ) H ( Y \mid Z = 1 ) + \operatorname* { P r } ( Z = 0 ) H ( Y \mid Z = 0 ) } \\ & { \qquad = H _ { 2 } ( C ( S ) ) + C ( S ) H ( Y \mid Z = 1 ) , } \end{array} }\tag{8}
$$

since $Z = 0$ implies $Y = \varnothing$ deterministically and hence $H ( Y \mid Z = 0 ) = 0$ . Because $Z = 1$ is equivalent to $X \in S$

$$
H ( Y \mid Z = 1 ) = H ( Y \mid X \in S ) .
$$

Thus

$$
H ( Y ) = H _ { 2 } ( C ( S ) ) + C ( S ) H ( Y \mid X \in S ) .\tag{9}
$$

Similarly,

$$
\begin{array} { r l } & { H ( Y \mid X ) = \operatorname* { P r } ( X \in S ) H ( Y \mid X , X \in S ) + \operatorname* { P r } ( X \not \in S ) H ( Y \mid X , X \not \in S ) } \\ & { \qquad = C ( S ) H ( Y \mid X , X \in S ) , } \end{array}\tag{10}
$$

because $X \not \in S$ implies $Y = \varnothing$ deterministically.

Subtracting (10) from (9),

$$
\begin{array} { r l } & { I ( X ; Y ) = H ( Y ) - H ( Y \mid X ) } \\ & { \qquad = H _ { 2 } ( C ( S ) ) + C ( S ) \Big [ H ( Y \mid X \in S ) - H ( Y \mid X , X \in S ) \Big ] } \\ & { \qquad = H _ { 2 } ( C ( S ) ) + C ( S ) I ( X ; Y \mid X \in S ) , } \end{array}\tag{11}
$$

recovering (2).

## C.3 Proposition 2

Proposition 2 (OVMI for a general in-vocabulary decoder). Now condition on the event that the intended word is representable, i.e. $X \in S$ . The corresponding distribution over in-vocabulary target words is $\textstyle p _ { S } ( x ) = \operatorname* { P r } ( X = x \mid X \in S ) = { \frac { p ( x ) } { C ( S ) } }$ , where $x \in S$ , that $i s ,$ the unrestricted word distribution restricted to S and renormalised to sum to one.

For $x , y \in S$ , let $K _ { S } ( y \mid x ) = \operatorname* { P r } ( Y = y \mid X = x )$ denote the decoder’s conditional distribution over outputs when the intended word is x. Thus $K _ { S }$ is the efective channel operating within the deployed vocabulary.

Under $p _ { S }$ and $K _ { S }$ , the resulting output distribution on $S$ is

$$
q _ { S } ( y ) = \sum _ { x \in S } p _ { S } ( x ) K _ { S } ( y \mid x ) , \qquad y \in S .
$$

Then

$$
I _ { \mathrm { O V M I } } ( S ) = C ( S ) \left[ H ( q _ { S } ) - \sum _ { x \in S } p _ { S } ( x ) H ( K _ { S } ( \cdot \mid x ) ) \right]\tag{12}
$$

where entropy $\begin{array} { r } { H ( X ) = - \sum _ { i } p ( x _ { i } ) \log _ { 2 } p ( x _ { i } ) } \end{array}$ for a discrete random variable $X$

We will use the following shorthand

$$
h _ { V } ( u ) : = - u \log _ { 2 } u - ( 1 - u ) \log _ { 2 } \left( \frac { 1 - u } { V - 1 } \right) .\tag{13}
$$

This is the same as the term $H ( Y | X )$ in Wolpaw’s formula in $\operatorname { E q . 1 }$ , describing the entropy of the decoder’s output distribution under a symmetric error assumption where the intended word is output with probability $u ,$ and each of the other $V - 1$ words is output with probability $( 1 - u ) / ( V - 1 )$ .

## C.4 Proof of Proposition 2

Proof of Proposition 2. By Definition 1,

$$
I _ { \mathrm { O V M I } } ( S ) = C ( S ) I ( X ; Y \mid X \in S ) .
$$

Conditioned on $X \in S$ , the input distribution is

$$
p _ { S } ( x ) = \operatorname* { P r } ( X = x \mid X \in S ) = \frac { p ( x ) } { C ( S ) } ,
$$

and the in-vocabulary channel is

$$
K _ { S } ( y \mid x ) = \operatorname* { P r } ( Y = y \mid X = x ) , \qquad x , y \in S .
$$

The output distribution is therefore

$$
q _ { S } ( y ) = \operatorname* { P r } ( Y = y \mid X \in S ) = \sum _ { x \in S } p _ { S } ( x ) K _ { S } ( y \mid x ) .
$$

Applying the standard mutual-information identity $I ( X ; Y ) = H ( Y ) - H ( Y \mid X )$ under the conditional distribution $X \mid X \in S$

$$
I ( X ; Y \mid X \in S ) = H ( q _ { S } ) - \sum _ { x \in S } p _ { S } ( x ) H ( K _ { S } ( \cdot \mid x ) ) .
$$

Multiplying by C(S) yields

$$
I _ { \mathrm { O V M I } } ( S ) = C ( S ) \left[ H ( q _ { S } ) - \sum _ { x \in S } p _ { S } ( x ) H ( K _ { S } ( \cdot \mid x ) ) \right] ,
$$

which is (12).

## C.5 Proof of Corollary 1

Proof of Corollary 1. Under the homogeneous symmetric channel,

$$
K _ { S } ( y \mid x ) = \left\{ { P , \qquad y = x , } \right.
$$

For each $x \in S$ , the entropy of the corresponding channel row is

$$
H ( K _ { S } ( \cdot \ | \ x ) ) = - P \log _ { 2 } P - ( 1 - P ) \log _ { 2 } \biggl ( \frac { 1 - P } { V - 1 } \biggr ) = h _ { V } ( P ) ,
$$

where $h _ { V }$ is defined in (13).

The induced output distribution satisfies

$$
\begin{array} { l } { \displaystyle q _ { S } ( j ) = \sum _ { x \in S } p _ { S } ( x ) K _ { S } ( j \mid x ) } \\ { \displaystyle = p _ { S } ( j ) P + \sum _ { x \neq j } p _ { S } ( x ) \frac { 1 - P } { V - 1 } } \\ { \displaystyle = P p _ { S } ( j ) + \frac { 1 - P } { V - 1 } \big ( 1 - p _ { S } ( j ) \big ) , } \end{array}\tag{14}
$$

which is (4).

Substituting into Proposition $2$ gives

$$
\begin{array} { l } { { \displaystyle { \cal I } _ { \mathrm { O V M I } } ( S ) = C ( S ) \left[ H ( q _ { S } ) - \sum _ { x \in S } p _ { S } ( x ) h _ { V } ( P ) \right] } } \\ { ~ } \\ { { \displaystyle ~ = C ( S ) \Big [ H ( q _ { S } ) - h _ { V } ( P ) \Big ] } , } \end{array}\tag{15}
$$

because $\begin{array} { r } { \sum _ { x \in S } p _ { S } ( x ) = 1 } \end{array}$ . This is (5).

## C.6 Retaining Per-Word Accuracies in OVMI

Corollary 2 (Symmetric error model with word-specific accuracies). Suppose each intended word $x \in S$ has its own correct-decoding probability $P _ { c } ( x )$ . Thus some words may be easier to decode than others. Conditional on an error, however, the decoder distributes the remaining probability uniformly across the other V − 1 in-vocabulary words. Hence

$$
K _ { S } ( y \mid x ) = \left\{ { \begin{array} { l l } { P _ { c } ( x ) , } & { y = x , } \\ { \displaystyle { \frac { 1 - P _ { c } ( x ) } { V - 1 } } , } & { y \neq x . } \end{array} } \right.
$$

Under this model, the output distribution on $S$ is

$$
q _ { S } ( j ) = p _ { S } ( j ) P _ { c } ( j ) + \frac { 1 } { V - 1 } \sum _ { x \neq j } p _ { S } ( x ) \big ( 1 - P _ { c } ( x ) \big ) , \qquad j \in S .\tag{16}
$$

The first term is the probability mass from correctly decoding j when $X = j$ . The second term is the total probability mass assigned to $j$ when some other in-vocabulary word was intended but the decoder makes an error and spreads that error uniformly across the incorrect outputs. Consequently,

$$
I _ { \mathrm { O V M I } } ( S ) = C ( S ) \left[ H ( q _ { S } ) - \sum _ { x \in S } p _ { S } ( x ) h _ { V } \big ( P _ { c } ( x ) \big ) \right] .\tag{17}
$$

This model relaxes the shared-accuracy assumption of Corollary 1, while retaining the simplifying assumption that, conditional on an error, all incorrect outputs are equally likely.

Proof of Corollary 2. Under the word-dependent symmetric channel,

$$
K _ { S } ( y \mid x ) = \left\{ { \begin{array} { l l } { P _ { c } ( x ) , } & { y = x , } \\ { \displaystyle { \frac { 1 - P _ { c } ( x ) } { V - 1 } } , } & { y \neq x . } \end{array} } \right.
$$

For each $x \in S$ , the entropy of the x-th channel row is

$$
H ( K _ { S } ( \cdot \ | \ x ) ) = - P _ { c } ( x ) \log _ { 2 } P _ { c } ( x ) - ( 1 - P _ { c } ( x ) ) \log _ { 2 } \left( \frac { 1 - P _ { c } ( x ) } { V - 1 } \right) = h _ { V } \big ( P _ { c } ( x ) \big ) .
$$

The induced output distribution is

$$
\begin{array} { l } { { q _ { S } ( j ) = \displaystyle \sum _ { x \in S } p _ { S } ( x ) K _ { S } ( j \mid x ) } } \\ { { \ \qquad = p _ { S } ( j ) P _ { c } ( j ) + \displaystyle \sum _ { x \neq j } p _ { S } ( x ) \frac { 1 - P _ { c } ( x ) } { V - 1 } } } \\ { { \ \qquad = p _ { S } ( j ) P _ { c } ( j ) + \displaystyle \frac { 1 } { V - 1 } \sum _ { x \neq j } p _ { S } ( x ) \big ( 1 - P _ { c } ( x ) \big ) , } } \end{array}\tag{18}
$$

which is (16).

Substituting into Proposition 2 yields

$$
I _ { \mathrm { O V M I } } ( S ) = C ( S ) \left[ H ( q _ { S } ) - \sum _ { x \in S } p _ { S } ( x ) h _ { V } \big ( P _ { c } ( x ) \big ) \right] ,
$$

which is (17).

## C.7 Further Observations for the Homogeneous Symmetric Channel

A useful rearrangement of (4) is

$$
q _ { S } ( j ) = \frac { 1 - P } { V - 1 } + \frac { P V - 1 } { V - 1 } p _ { S } ( j ) .\tag{19}
$$

Table 2: Cross-study comparison. Each cell reports OVMI in bits, with $\mathrm { O V M I } / H ( p )$ in parentheses and uncertainty beneath. Rows are ordered within each block by SUBTLEX–UK OVMI. Superscripts on P identify balanced top-1 accuracy (b), isolated-word accuracy (a), and 1 − WER (w); † indicates that $P = 1 \mathrm { - } \mathrm { W E R }$ is a lower bound. Bracketed ranges are 95% sampling intervals for invasive systems using Wilson score intervals for isolated-word accuracies and published sentence/trial intervals for WER-derived rows. ± values are OVMI uncertainties obtained by mapping mean $P$ plus or minus one SEM across three training seeds for non-invasive systems.
<table><tr><td>System</td><td>V</td><td>P</td><td>Broad spoken SUBTLEX-UK  $H ( p ) = 9 . 7 7$ </td><td>Conversational Switchboard  $H ( p ) = 8 . 2 7$ </td><td>AAC / clinical UCV  $H ( p ) = 4 . 3 0$ </td><td>Narrative prose Sherlock  $H ( p ) = 8 . 4 4$ </td></tr><tr><td colspan="5">Attempted speech (invasive)</td><td>4.197 (97.5%)</td><td>8.096 (95.9%)</td></tr><tr><td>Card (+LM)</td><td>125k</td><td> $9 7 . 5 \% ^ { \mathrm { w } \dagger }$ </td><td>[9.093, 9.208] 7.038 (72.0%)</td><td>[7.964,8.057] 6.227 (75.3%)</td><td>[4.171, 4.218] 3.279 (76.2%)</td><td>[8.046, 8.138] 6.311 (74.7%)</td></tr><tr><td>Willett (+LM)</td><td>125k</td><td>76.2%w†</td><td>[6.835, 7.232] 0.656 (6.7%)</td><td>[6.052, 6.394] 0.949 (11.5%)</td><td>[3.189, 3.365] 1.822 (42.3%)</td><td>[6.135,6.479] 0.239 (2.8%)</td></tr><tr><td>Willett (isolated)</td><td>50</td><td>94.0%a</td><td>[0.637, 0.672] 0.621 (6.4%)</td><td>[0.922, 0.971] 0.900 (10.9%)</td><td>[1.776, 1.859] 1.737 (40.4%)</td><td>[0.232, 0.244] 0.227 (2.7%)</td></tr><tr><td>Willett (+LM)</td><td>50</td><td> $9 0 . 9 \% ^ { \mathrm { w \dag } }$ </td><td>[0.599,0.642] 0.459 (4.7%)</td><td>[0.868, 0.929] 0.670 (8.1%)</td><td>[1.681, 1.789] 1.321 (30.7%)</td><td>[0.219,0.234] 0.169 (2.0%)</td></tr><tr><td>Moses (+LM)</td><td>50</td><td> $7 4 . 4 \% ^ { \mathrm { w \dag } }$ </td><td>[0.360, 0.539] 0.238 (2.4%)</td><td>[0.527, 0.783]</td><td>[1.053, 1.529]</td><td>[0.133,0.197]</td></tr><tr><td>Moses (isolated)</td><td>50</td><td> $4 7 . 1 \% ^ { \mathrm { a } }$ </td><td>[0.230, 0.245]</td><td>0.350 (4.2%) [0.339, 0.361]</td><td>0.711 (16.5%) [0.690, 0.733]</td><td>0.088 (1.0%) [0.085,0.091]</td></tr><tr><td colspan="5">Perceived speech (non-invasive)</td><td>0.266 (6.2%)</td><td>0.348 (4.1%)</td></tr><tr><td>Tang 2023</td><td>6867</td><td> $6 . 7 \% ^ { \mathrm { ~ w ~ } \dagger }$ </td><td>0.357 (3.6%) ±0.030 0.237 (2.4%)</td><td>0.376 (4.5%) ±0.031 0.275 (3.3%)</td><td>±0.020 0.316 (7.3%)</td><td>±0.029 0.221 (2.6%)</td></tr><tr><td>LibriBrain100 2025</td><td>50</td><td> $2 5 . 8 \% ^ { \mathrm { b } }$ </td><td>±0.003 0.176 (1.8%)</td><td>±0.003 0.192 (2.3%)</td><td>±0.004 0.254 (5.9%)</td><td>±0.003</td></tr><tr><td>Armeni 2022</td><td>50</td><td> $2 0 . 8 \% ^ { \mathrm { b } }$ </td><td>±0.004 0.033 (0.3%)</td><td>±0.004 0.038 (0.5%)</td><td>±0.005 0.059 (1.4%)</td><td>0.189 (2.2%) ±0.004 0.035 (0.4%)</td></tr><tr><td>MEG-MASC 2023</td><td>50</td><td> $8 . 5 \% ^ { \textrm { b } }$ </td><td>±0.003</td><td>±0.004</td><td>±0.006</td><td>±0.003</td></tr></table>

Equation (19) makes three facts immediate.

First, if $p _ { S }$ is uniform, then $q _ { S }$ is uniform.

Second, if $P = 1 / V$ (chance performance), then $q _ { S }$ is uniform for any $p _ { S }$

Third, if $P \neq 1 / V ,$ , then $q _ { S }$ is non-uniform if and only if $p _ { S }$ is non-uniform. In particular, for informative decoders with $P > 1 / V$ , any non-uniformity in the in-vocabulary language distribution propagates to the output distribution. Consequently,

$$
H ( q _ { S } ) < \log _ { 2 } V
$$

whenever $p _ { S }$ is non-uniform and $P \neq 1 / V$ , since the uniform distribution uniquely maximises entropy on a finite alphabet.

## D Closed Form OVMI Pseudocode

In Figure 7, we provide Python pseudocode for computing OVMI under the homogeneous (single accuracy value) and symmetric error assumption. Note that when computing OVMI, we typically exclude words with fewer than five instances to ensure accuracy estimates for $P _ { \mathrm { m a c r o } }$ are reliable.

## E Table of Results

Table 2 provides the full OVMI scores and uncertainties for the systems discussed in the main text across all the reference distributions explored in this work.

## F Cross-Study OVMI Estimation Details

This appendix describes how we estimate OVMI from the summary statistics reported by existing speech-decoding studies and how we construct the reference distributions used in Section 5. The general definition of OVMI in Proposition 2 requires the decoder channel $K _ { S } ( y \mid x )$ . In retrospective comparisons, however, published studies rarely report complete confusion matrices and instead typically provide a vocabulary size together with accuracy or word error rate (WER). We therefore use the homogeneous symmetric-channel estimator from Eq. (5) whenever a more detailed channel estimate is unavailable.

## F.1 Estimation from a Scalar Accuracy

For a decoder vocabulary S of size $V ,$ and given a single correct-decoding probability $P ,$ we approximate the in-vocabulary channel as

$$
K _ { S } ( y \mid x ) = \left\{ { P , y = x , \atop \displaystyle { \frac { 1 - P } { V - 1 } } , y \neq x } \right.
$$

The induced output distribution is therefore

$$
q _ { S } ( y ) = P p _ { S } ( y ) + \frac { 1 - P } { V - 1 } \left[ 1 - p _ { S } ( y ) \right] ,
$$

and the conditional entropy of the symmetric channel is

$$
h _ { V } ( P ) = - P \log _ { 2 } P - ( 1 - P ) \log _ { 2 } { \frac { 1 - P } { V - 1 } } .
$$

The retrospective estimate used throughout the cross-study comparison is then

$$
\widehat { I } _ { \mathrm { O V M I } } = C ( S ) \left[ H ( q _ { S } ) - h _ { V } ( P ) \right] .
$$

This approximation assumes only that all supported words have the same correct-decoding probability and that errors are distributed uniformly among the remaining $V - 1$ outputs. The reference distribution itself remains non-uniform through $C ( S ) , p _ { S }$ , and $q _ { S }$

Where available, we use macro rather than micro accuracy for P. In particular,

$$
P _ { \mathrm { m a c r o } } = \frac { 1 } { V } \sum _ { x \in S } \operatorname* { P r } ( \hat { Y } = x \mid X = x ) ,
$$

which summarises the reliability of an average supported word without additionally weighting common words by their frequency. Frequency is already represented by the external distribution $p .$

## F.2 Conversion of Reported Performance Measures

Balanced top-1 accuracy. For the non-invasive word-classification results, the reported balanced top-1 accuracy is the macro-average of the class-wise recalls.

Isolated-word accuracy. For studies reporting accuracy on an isolated $V \cdot$ -way word-classification task, we use the reported mean correct-classification probability directly as P. This gives $P = 0 . 4 7 1$ for the isolated 50-word experiment of Moses et al. (2021) and $P = 0 . 9 4 0$ for the corresponding 50-word result of Willett et al. (2023).

Word error rate. Continuous speech decoders generally report WER rather than word classification accuracy. For a reference containing N words, with $N _ { \mathrm { s u b } }$ substitutions, $N _ { \mathrm { d e l } }$ deletions, and $N _ { \mathrm { i n s } }$ insertions,

$$
\mathrm { W E R } = \frac { N _ { \mathrm { s u b } } + N _ { \mathrm { d e l } } + N _ { \mathrm { i n s } } } { N } .
$$

The proportion of intended words that are neither substituted nor deleted is

$$
P _ { \mathrm { c o r r e c t } } = 1 - \frac { N _ { \mathrm { s u b } } + N _ { \mathrm { d e l } } } { N } = 1 - \mathrm { W E R } + \frac { N _ { \mathrm { i n s } } } { N } .
$$

Hence

$$
P _ { \mathrm { c o r r e c t } } \geq 1 - \mathrm { W E R } .
$$

When only WER is available, we therefore set

$$
P = 1 - \mathrm { W E R } .
$$

This is a conservative estimate of per-intended-word correctness because insertions increase WER without corresponding to a failure to recover an intended word. For the language-model-assisted results, this gives $P = 0 . 7 4 4$ for Moses, $P = 0 . 9 0 9$ and 0.762 for the 50-word and 125k-word Willett systems, respectively, and $P = 0 . 9 7 5$ for Card.

## F.3 Reference Distributions

All reference distributions are unigram distributions over word types. For a corpus-derived reference, we estimate

$$
p ( w ) = \frac { n ( w ) } { \sum _ { u \in \Omega } n ( u ) } ,
$$

where $n ( w )$ is the number of occurrences of word w in the reference corpus after lower-casing all words. For a decoder vocabulary S, words outside S remain part of the reference distribution and contribute to the uncovered probability mass $1 - C ( S )$ ; p is therefore never renormalised to the decoder vocabulary before computing coverage.

Broad spoken English: SUBTLEX-UK. Our default reference is SUBTLEX-UK (van Heuven et al., 2014), a word-frequency norm derived from British film and television subtitles. We normalise the SUBTLEX-UK frequency counts to obtain p. This distribution has entropy $H ( p ) = 9 . 7 7$ bits after the preprocessing used in our evaluation and is intended as a broad spoken-English reference.

Conversation: Switchboard. For conversational speech, we construct an empirical unigram distribution from the 36-call Switchboard sample distributed with NLTK. Word counts are aggregated across the sample and normalised to sum to one. The resulting reference has entropy $H ( p ) = 8 . 2 7$ bits.

AAC: Universal Core Vocabulary. For an assistive communication reference, we use the 36-word Universal Core Vocabulary (UCV) of Erickson et al. (2019). UCV specifies a set of words rather than a probability distribution. We therefore assign each UCV word its corresponding SUBTLEX-UK frequency and renormalise within the UCV set,

$$
p _ { \mathrm { U C V } } ( w ) = \frac { f _ { \mathrm { S U B T L E X } } ( w ) } { \sum _ { u \in \mathrm { U C V } } f _ { \mathrm { S U B T L E X } } ( u ) } , \qquad w \in \mathrm { U C V } .
$$

This preserves natural diferences in word frequency rather than treating the 36 words as equiprobable.   
The resulting distribution has $H ( p ) = 4 . 3 0$ bits.

Narrative speech: Sherlock. For narrative prose, we use the held-out LibriBrain100 text corresponding to Session 12, Chapter 1 of The Adventures of Sherlock Holmes (Doyle, 1892). This session belongs to the LibriBrain100 test set and is not used to fit the non-invasive decoders. We count the words occurring in the chapter and normalise their empirical frequencies to form the narrative reference distribution. The resulting distribution has entropy $H ( p ) = 8 . 4 4 ~ \mathrm { b i t s }$ . This reference is intentionally narrower than SUBTLEX-UK and represents the narrative-speech domain on which several of the non-invasive systems are evaluated.

## F.4 Normalisation and Uncertainty

OVMI is reported in bits per intended word drawn from p. When comparing performance across diferent reference distributions, we additionally normalise by the entropy of that distribution,

$$
\mathrm { O V M I } _ { \% } = 1 0 0 \frac { I _ { \mathrm { O V M I } } } { H ( p ) } .
$$

This quantity is the percentage of lexical information available under the stated reference distribution that is conveyed by the system.

For invasive isolated-word results, uncertainty in the reported accuracy is propagated through the OVMI estimator using Wilson score intervals. For WER-derived results, we propagate the corresponding published sentence- or trial-level uncertainty through $P = 1 - \mathrm { W E R }$ . For the non-invasive experiments, we compute OVMI at the mean balanced accuracy and at one standard error above and below that mean across training seeds.

## F.5 Large-Vocabulary Invasive Systems

The 125k-word decoder lexicons used by Willett et al. (2023) and Card et al. (2024) come from a lexicon based on the CMU Pronouncing Dictionary, but are not published in full as far as we are aware. For coverage calculations, we use the CMU Pronouncing Dictionary ourselves, after collapsing pronunciation variants to unique orthographic word forms. This yields 126,052 unique spellings. We retain the reported vocabulary size $V = 1 2 5 { , } 0 0 0$ when evaluating the decoder channel.

## G Vocabulary Optimisation Experiment Details

Data and splits. We use subject 0 from LibriBrain100 and report matched-domain results for TIMIT, Podcasts, and Sherlock. Reference word distributions are estimated exclusively from the training portion of the corresponding domain. For Podcasts, sessions 1–28 form the training set, session 29 is validation, and session 30 is test. For Sherlock, the training set comprises the training runs, with Sherlock1 session 11 used for validation and session 12 for test. TIMIT is split at the utterance level using the canonical LibriBrain100 policy: the 50 development speakers are used for validation and the 24 core-test speakers for testing. The repeated SA utterances are excluded from validation and test, as are training speakers sharing an SX sentence with a core-test speaker. MOCHA-TIMIT training sessions A and D contribute only to decoder training and candidate-word availability.

Candidate vocabulary and neural decoder. The candidate pool is constructed from word tokens with usable neural training windows after removal of incomplete examples. Words must occur at least five times in the pooled neural training data. Eligible words are ordered by decreasing training count and the pool is capped at 250 words. The same candidate pool is used for all domains, selection methods, vocabulary sizes, and model seeds.

We train five independently initialised instances of the contrastive brain-to-T5-embedding decoder by d’Ascoli et al. (2025) for 50 epochs with a batch size of 128. For each seed, the checkpoint is selected using pooled validation contrastive top-1 accuracy. The model has no vocabulary-specific classification head. After training, we construct one normalised target embedding for each candidate word and cache the cosine score between every validation or test neural prediction and all 250 candidates. Consequently, evaluating a vocabulary requires only restricting the cached score matrix to the corresponding columns and applying an argmax; the neural model is never retrained for a particular method or vocabulary size.

Reference distribution. For domain $d ,$ let $p _ { d } ( w )$ denote the empirical probability of word w in that domain’s training text. Probabilities are defined over the full training-text distribution rather than being renormalised over the 250-word candidate pool. Thus, for a selected vocabulary V, its lexical coverage is

$$
C _ { d } ( \mathcal { V } ) = \sum _ { w \in \mathcal { V } } p _ { d } ( w ) .
$$

No validation or test text contributes to $p _ { d } .$

To compute OVMI, a full confusion-channel estimate is used when every word in the trial vocabulary has at least five validation examples. Otherwise, selection falls back to the per-word symmetric-error OVMI approximation. At $V = 1 5 0$ , this fallback was used for all five TIMIT seeds and for Sherlock seeds 0, 3, and $5 ;$ all other final domain–size–seed combinations used the full validation channel.

Statistical testing. For every domain and vocabulary size, OVMI and Frequency are compared using a one-sided paired randomisation test. We use $B = 1 0 0 { , } 0 0 0$ Monte Carlo permutations and calculate the one-sided p-value with the finite-simulation correction

$$
p _ { \mathrm { r a w } } = \frac { 1 + \sum _ { b = 1 } ^ { B } \mathbf { 1 } [ T _ { b } \geq T _ { \mathrm { o b s } } ] } { B + 1 } .
$$

Each $V$ is tested separately. Holm’s step-down correction is then applied across the nine vocabulary sizes within each domain, controlling the error rate separately for TIMIT, Podcasts, and Sherlock.

```python
import numpy as np
2 from scipy . special import xlogy
3
4 _LN2 = np . log (2.0)
5
6 def _xlog2y (x, y):
7 """x * log2 (y), with the convention 0 * log2 (0) = 0."""
8 return xlogy (x , y ) / _LN2
9
10 def compute_ovmi (subset , P_c , freqs , p_freq):
11
12 subset : indices of words in the candidate vocabulary
13 P_c : macro - averaged classification accuracy over the subset
14 freqs : reference unigram frequencies ( aligned with subset )
15 p_freq: total number of words in the reference distribution
16 Returns the open - vocabulary mutual information ( OVMI ).
17
18 V = len ( subset )
19 if V <= 1:
20 return 0.0
21
22 # (1) Coverage : natural -language mass spanned by the subset
23 coverage = freqs .sum () / p_freq
24
25 # (2) Input distribution p_s (x) over subset words
26 if freqs .sum () != 0:
27 p_x = freqs / freqs .sum ()
28 else :
29 return 0
30
31 # (3) Output distribution q(y) assuming errors spread uniformly
32 # over the V-1 incorrect classes
33 p_err = (1 - P_c ) / (V - 1)
34 q_y = p_err + p_x * ( P_c - p_err )
35
36 # (4) Marginal entropy of predictions
37 H_Y = -np.sum ( _xlog2y (q_y , q_y ))
38
39 # (5) Conditional entropy
40 H_YX = - _xlog2y ( P_c , P_c ) - _xlog2y (1 - P_c , p_err )
41
42 # (6) Mutual information , floored at 0 to absorb numerical noise
43 MI = max (0.0 , H_Y - H_YX )
44
45 # (7) Weight by natural -language coverage
46 return coverage * MI
```  
Figure 7: OVMI computation for a candidate vocabulary subset with a single accuracy scalar.