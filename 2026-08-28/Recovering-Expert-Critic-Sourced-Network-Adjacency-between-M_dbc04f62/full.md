# Recovering Expert Critic-Sourced Network Adjacency between Musical Artists from Acoustic Distributions: A Construct-Validity Approach

Elena Badillo-Goicoechea The University of Chicago elenabadillog@uchicago.edu

## Abstract

Music recommendation and discovery systems rely primarily on two kinds of signal: user–item interactions, which fail in the coldstart regime, and intrinsic musical content, which remains available for any recording and is therefore the natural basis for contentbased discovery. We argue that a third signal, largely untapped by these systems, is both richer and more principled: critical adjacency, the pairwise relation established when an expert music critic explicitly links two artists in long-form prose. Critical adjacency encodes deliberate, articulated judgments about which artists belong together, and prior work introducing this approach established its internal validity. We have shown in prior work that it recovers coherent, musicologically interpretable communities, and can match collaborative filtering in user-satisfaction simulations, while requiring no user data [2]. What has been missing is external validation: the extent to which this critic-sourced relation is grounded in (1) the music itself, (2) sociological context, (3) randomness. The main goal of this work is to quantify the first by testing it directly against rich acoustic content, further reframing our question as one of construct validity.

Representing artists as an empirical distribution over 80 lowlevel acoustic descriptors (Essentia) and modeling pairwise artists proximity via marginal optimal-transport (Wasserstein) distances between each of those 80 features, we evaluate the degree to which critical adjacency is sonically recoverable under a cold-start, artist disjoint split. Our ensemble recovers critic-sourced edges with an out-of-sample AUC of 0.767 (95% CI 0.761–0.775). We find that recoverability rises monotonically with critical consensus, reaching AUC 0.865 on multi-source attested edges, while single-source mentions account for much of the residual gap.

Furthermore, stratified evaluations align with sociological models of genre: tightly bounded, scene-based genres exhibit higher sonic recoverability than broad, retrospective industry umbrella terms. These findings demonstrate, on the one hand, that critical discourse constitutes a rich source of information that can be exploited to enhance music recommendation; on the other hand that critical discourse signal can be efectively decomposed into a reproducible “sonic core,” whereas single-source edges mark a “sociological remainder,” potentially driven by narrative position ing, subcultural context, and canonical placement, of interest for musicological studies. This work ofers both a scalable discovery mechanism for cold-start recommendation and a sociologically grounded approach to MIR and MRS research by incorporating insight from the sociology of popular music.

Fengfeng He

The University of Chicago

hef@uchicago.edu

## Keywords

content-based recommendation, cold-start, music information retrieval, optimal transport, construct validity, sociology of music

## 1 Introduction

Within standard music recommendation and discovery systems taxonomies [15, 16, 27], techniques split across collaborative filtering (user logs), content-based methods (raw audio features), and context-based (metadata, text, tags). Systems relying on interactionbased signals like streaming counts and user co-occurrences degrade when items lack historical interaction data, a fundamental boundary known as the cold-start problem [29, 34]. They also display other limitations, being based on purely correlational signals, lacking any musicological grounding.

Content-based methods bridge these gaps by deriving similarity directly from intrinsic item properties rather than accumulated consumer behavior. When evaluating content-based audio representations, Music Information Retrieval (MIR) research often measures acoustic features against high-level cultural categories or editorial targets. However, interpreting these context-based signals requires careful theoretical grounding. In this work, we analyze critical adjacency, defined as the pairwise relation established whenever a professional music critic co-mentions two artists within long-form journalism, such as album reviews. Unlike the interaction and content signals that current systems rely on, critical adjacency is a deliberate, expert-articulated judgment about which artists belong together, and it remains largely untapped by recommendation and discovery systems. Prior work introducing this approach established that these critic-sourced relations form a coherent, internally valid artist graph—validated through community-detection analysis ofits network structure and through simulation ofrecommendation quality [2]; what has not been established is whether that graph is grounded in the music itself.

Rather than treating critical co-mentions as a direct proxy for acoustic similarity, we reframe this relationship as a constructvalidity problem [32]. By establishing purely content-based, distributional representations of audio as an independent physical baseline, we pose a fundamental question: whether critical adjacency is a valid measure for musical similarity. To answer that, we analyze to what extent critical adjacency is explained by acoustic characteristics, and how much of it reflects non-acoustic social context.

Cultural sociology provides essential context for understanding this distinction. Music reviews are inherently dual-natured: critics function as institutional gatekeepers [14] and cultural intermediaries engaged in acts of consecration and symbolic positiontaking [5, 6]. When a critic co-mentions two artists, they construct aesthetic meaning. That connection may stem from shared acoustic characteristics (e.g. timbre, tempo, key, rhythms), but it may equally reflect shared subcultural scenes, political stances, visual personas, or canonical lineage [13, 19].

Testing critic graphs against low-level audio features serves two main functions:

(1) Validating Data Quality: Album reviews capture deliberate, expert-articulated relationships that provide richer structural signal than unmoderated crowd tags or raw web co-occurrences [7].

(2) Measuring Construct Validity: Because critical discourse is shaped by complex social and narrative forces, audio features provide an objective baseline to determine how much of critical adjacency is sonically grounded versus how much forms a non-acoustic “sociological remainder.”

A prerequisite for both is a content-based measure of musical similarity that is faithful to how an artist actually sounds. Our second contribution is such a measure: rather than reducing an artist to a mean feature vector, we represent each artist as a distribution over their tracks and define similarity as the vector of per-feature Wasserstein (optimal-transport) distances between two artists’ dis tributions. This distributional formulation captures diferences in the full shape of an artist’s sound (its spread and multimodality), not only its average, and it is the predictor on which every result below rests.

To evaluate this, we model each artist as a continuous distribution over 80 low-level acoustic features, computing pairwise proximity using marginal Wasserstein (optimal transport) distances [24, 36]. We then train a stacked ensemble learner (Super Learner) [35] to predict critical adjacency across a review-derived graph of 19,059 artists under an artist-disjoint, cold-start evaluation.

Our findings demonstrate that acoustic distributions capture substantial signal present in critical judgments. More importantly, error patterns illuminate the underlying construct: as critical consensus grows from single mentions to multi-source agreement, audio recoverability rises monotonically from AUC 0.767 to 0.865. Multi-critic convergence isolates a reproducible sonic core, while single-source edges mark the sociological remainder where criticism operates primarily through narrative framing and canonical placement.

## 2 Related Work

## 2.1 MIR Construct Validity

The clearest theoretical framework for reframing the paper is the validity typology applied to Music Information Retrieval [32]. They distinguish statistical conclusion, internal, construct, and external validity, and argue that MIR studies often conflate these forms. In particular, they question the common assumption that genreclassification accuracy provides evidence of music similarity. This framework provides the terminology needed to state explicitly that the present study evaluates the construct validity of using critic co-mentions as an operationalization of musical similarity rather than treating that assumption as implicit.

Flexer & Grill [12] provide important context for interpreting the observed AUC values. They show that agreement between human listeners on music similarity is only moderate, while agreement by the same listener across repeated evaluations is only slightly higher. These findings suggest that part of the gap between the observed AUC values of0.767 to 0.865 and perfect classification may reflect disagreement within the critic-based similarity construct itself rather than limitations of the audio features or classification models.

Wiggins [37] presents the opposing position that musical meaning is inherently subjective and context dependent, making the notion of a fixed semantic ground problematic. The consensusstratification results reported in Section 5.1 provide evidence against the strongest version of this argument. Critic relationships supported by multiple independent sources are more readily recovered from audio features than relationships supported by only a single source, suggesting that stronger consensus corresponds to greater acoustic coherence.

Ellis et al. [10] provide the closest precedent for treating musical similarity labels as contestable rather than fixed. They compared several candidate sources of artist similarity, including expertgenerated lists, critic co-mentions, and listener behavior, and found limited agreement between them. This supports interpreting the critic network used in the present study as one possible operationalization of similarity rather than an unquestioned ground truth. Berenzweig et al. [3] provides a natural companion reference for comparisons between acoustic similarity and subjective similarity judgments.

One additional explanation that may warrant investigation is hubness [25]. In high-dimensional nearest-neighbor spaces, a small number of items can become frequent neighbors regardless of their true similarity. Related works suggest that hub recommendations are often judged by listeners to be less perceptually meaningful. An empirical test would examine whether false-positive neighbor retrieval in the present study is concentrated among a small number of hub artists.

## 2.2 Genre as a Limited Proxy for Musical Similarity

The limitations of genre as a standalone proxy for musical similarity are well documented within both the MIR and the sociological literature. McKay & Fujinaga [21] respond directly to the view that genre classification should be abandoned in favor of more general music similarity research because genre labels are inherently ambiguous and subjective. The existence of this debate shows that concerns about the validity of genre as a classification target have long been recognized within the field.

Empirical evidence also highlights the limitations of genre as a labeling framework. Cross-collection evaluations by Bogdanov et al. [4] show that genre classification models generalize poorly across datasets, even when the same genre labels are used, because annotators apply labels such as “Rock” inconsistently. Seyerlehner et al. [30] likewise report human agreement rates ranging from 26% to 71% across a 19-genre labeling scheme. These findings provide empirical context for the substantial variation in within-genre AUC observed across genres in Table 2.

The sociological literature ofers a complementary explanation for this variation. Lena & Peterson [18] identify four broad genre types (avant-garde, scene-based, industry-based, and traditionalist) based on an analysis of 60 forms of American music. Lena [17] further argues that genres commonly evolve through these forms over time. This framework suggests that genres organized around tightly bounded and internally regulated scenes, such as metal, hiphop/rap, or electronic music, may exhibit greater sonic coherence than broader industry-based or traditionalist categories such as classical, pop, or alternative rock. Fabbri’s [11] account of genre as a socially negotiated system of rules, together with Negus’ [22] critique that these rules are historically dynamic rather than fixed, further supports the view that genre is a social classification rather than a naturally defined acoustic category.

Silver et al. [31] reach a similar conclusion from a large-scale network analysis of approximately three million musicians. Rather than finding a single, unified genre structure, they identify multiple genre complexes composed of smaller and diferently organized communities. Their findings are consistent with the genre heterogeneity observed in the present study and support treating genre as a collection of distinct social formations rather than a single, uniform unit of analysis.

## 2.3 Network Approaches and Recommendation Taxonomies

Network-based approaches in Music Recommender Systems have increasingly moved beyond traditional collaborative filtering by representing relationships as graphs [2, 20]. Most existing network models fall into two broad categories [16, 27, 33]. The first consists of behavioral networks derived from user listening histories, session transitions, and playlist co-occurrences. The second consists of contextual networks built from coarse metadata (e.g. genre, geography), or unstructured music-related text such as artist biographies (see Oramas et al. [23]).

Each approach has limitations. Behavioral networks depend on user interactions, making them vulnerable to cold-start problems for new or less popular artists while reinforcing existing popularity patterns. Contextual networks often come from biased and relatively small samples, and contain spurious statistical signal such as viral trends, informal discussion, and search-engine optimization.

Our critical adjacency framework represents a diferent type of network. Instead of relying on user behavior or biased, limited web data, it constructs artist relationships directly from all publicly available professional music criticism, and using natural language processing techniques to further filter out noise. Because these links originate from deliberate acts of aesthetic evaluation, the resulting graph provides a structured representation of perceived musical similarity that is less dependent on interaction history or unmoderated online content. This also makes the framework naturally suitable for evaluating cold-start recommendations.

Placing the critic graph within the broader MRS taxonomy [16, 27] reveals a diference in how co-occurrence data are defined. Early approaches such as Cohen & Fan [7] extracted artist relationships from general web searches. Knees & Schedl [16] later grouped cooccurrence sources into four categories: web pages, microblogs, playlists, and peer-to-peer networks. These categories distinguish sources by platform rather than by authorship, and therefore do not separate expert editorial judgments from consumer-generated activity. Recent reviews [9, 28] show that music recommender systems increasingly combine item similarity with user context [1]. Within this setting, evaluating the construct validity of expertderived similarity provides a way to assess whether one commonly used source of relational information captures meaningful musical relationships.

## 2.4 The Sociology of Music Criticism

The sociology of music criticism provides a theoretical explanation for why critic-based similarity should contain information beyond acoustic resemblance. Rather than treating critical comparisons as neutral descriptions of musical sound, this literature understands them as socially situated acts of cultural evaluation.

Bourdieu [5, 6] provides the central theoretical framework. He distinguishes cultural producers from intermediaries such as critics and describes the process of consecration through which critics confer legitimacy on artists. This relationship is reciprocal: critics often remain associated with artists they championed early, while established artists reinforce the authority of critics who helped construct their reputations. Within this framework, linking two artists in a review is not simply a perceptual judgment about musical similarity; it is also an act of canon formation, field positioning, and cultural evaluation. Critical adjacency should therefore be expected to contain information that extends beyond sonic resemblance alone.

Hirsch [14] complements this perspective by examining the institutional relationship between cultural production and criticism. He describes critics as gatekeepers within the cultural industries and argues that their role depends on maintaining a degree of independence from the musicians and labels whose work they evaluate. Hirsch therefore provides a theoretical basis for treating critic relationships as distinct from other musicians or label promotion.

The historical development of rock criticism further supports this interpretation. Lindberg [19] traces the emergence of music criticism from the specialist press of the 1960s into an established cultural institution with its own conventions, professional identities, and comparative practices. Frith [13] likewise argues that music criticism operates through socially shared forms of evaluation rather than objective descriptions of musical sound.

This perspective also helps explain why cross-genre artist pairs may remain predictable from audio features. Regev [26] argues that popular music increasingly operates within a shared framework of aesthetic cosmopolitanism in which critics, musicians, and audiences draw on common evaluative standards across genres. If critics employ a genre-transcending aesthetic framework when comparing artists, cross-genre relationships need not be substantially less coherent than within-genre relationships. This provides a theoretical explanation for the findings reported in Section 5.2. Bourdieu’s framework has also been extended empirically: Corciolani et al. [8] show that critics occupying diferent positions within the cultural field evaluate artists diferently, suggesting that outlet or critic-level metadata could be examined directly as moderators of whether critic similarities are driven more strongly by acoustic characteristics or by broader social processes.

## 3 Data

Critical adjacency graph. Our target is a graph over musical artists whose edges are derived from music-review sources. Reviews are drawn from a corpus of roughly 65,000 long-form album reviews spanning ten music-criticism outlets (for example Pitchfork, NPR, The Guardian, The Quietus, and Bandcamp Daily), following prior work [2]. Here, a directed edge between any two artists indicates that critics mention one artist in relation to another in their writing (for example, Brian Eno is mentioned in a review of a David Bowie album such as Low, 1977), and the edge weight $a _ { i j }$ counts the total number of times artist � is mentioned across the corpus of artist $j ^ { \prime } s$ reviews, measuring intensity of the pairwise relation. There were broadly 25,000 artists (nodes) represented in the graph. As expected of a critic-sourced graph, edges are sparse (density approximately $8 . 6 \times 1 0 ^ { - 4 } )$ and weakly attested: the median edge weight is one, and roughly four in five artist pairs are linked by a single source.

Audio features. We use the low-level acoustic descriptors computed by the Essentia library and distributed through the AcousticBrainz project, a crowd-sourced corpus in which a single pinned extractor was run on each contributed recording, ensuring crosscontributor comparability. Each recording is described by 80 usable scalar descriptors spanning timbre (MFCC means), timbral variability (MFCC variances), harmony (HPCP), spectral shape, rhythm, tonal features, and loudness. Artists in the critic graph are matched to AcousticBrainz by normalized artist name (casefolded, whitespace- and punctuation-normalized) resolved through MusicBrainz artist identifiers, retaining only artists with at least three matched tracks so that per-artist distributions are estimable. After artist-matching to the critical adjacency graph with the audio corpus and filtering to artists with at least three tracks available, the final sample contained 19,059 artists. After matching, the corpus covered 2,154,520 tracks across the 19,059 artists, with a median of 42 tracks per artist. Importantly, because the Essentia descriptors are computable from any recording, this representation remains available even for artists with no interaction history or critical coverage, or no longer tracked by the MusicBrainz project.

## 4 Methodology & Construct-Validity Framework

The methodology pipeline (Figure 1) integrates an audio content layer of predictor variables with a context data layer of targets to model critic-sourced artist connections under a cold-start framework.

Audio content layer. From the 80 Essentia descriptors (Section 3), we treat each artist as an empirical distribution over their tracks in the 80-dimensional feature space rather than reducing them to a mean vector: two artists may share an average sound yet difer in how widely their catalogue ranges. This motivates our proposed similarity measure. For each of the 80 features we compute the one-dimensional Wasserstein (optimal-transport) distance between the two artists’ empirical distributions of that feature, in closed form as the $L ^ { 1 }$ distance between their quantile functions. Stacking these gives an 80-dimensional distributional distance vector that summarizes how two artists difer across the full acoustic spectrum, feature by feature and in distributional shape rather than in mean alone. This vector is the sole predictor used throughout.<sup>1</sup>

Contextual target layer. The prediction target is the critic adjacency graph of Section 3, constructed by extracting pairwise artist co-mentions from the review corpus via Named Entity Recognition. Entity extraction follows the pipeline of prior work [2]: candidate artist mentions are identified with an NLTK/TextBlob named-entity pass and resolved against the roster of reviewed artists, so that recognized entities are constrained to a known artist vocabulary rather than open-domain names. Although the underlying graph is directed—the number of times artist � is mentioned in artist $j ^ { \prime } s$ reviews need not equal the reverse—our predictor, the marginal Wasserstein distance between two artists’ acoustic distributions, is symmetric by construction. For this work we therefore collapse the graph to its undirected form, defining an (undirected) edge between two artists whenever a co-mention is attested in either direction and setting its weight to the total co-mention count across both directions; the asymmetry itself is a promising target for future, direction-aware modeling. We predict the presence of a criticsourced edge between two artists from their acoustic distributional distance vector alone.

Prediction model . A Super Learner ensemble [35] (stacked over several diferently parametrized binary individual classifiers including: regularized logistic regression, random forests, gradientboosted trees, and a nearest-neighbor learner) was fitted to maps the 80-dimensional Wasserstein distance vector to the probability of a critic-sourced edge being present or not. Positives are all attested edges; negatives were sampled from non-edges at a 3:1 ratio (case-control sampling). We evaluate under an artist-disjoint split: artists are partitioned into training and test sets, the model trains only on pairs internal to the training set and is tested only on pairs internal to the test set, so no test artist is ever observed in training. Predictive performance was measured by Area Under the ROC Curve (AUC), with bootstrapped 95% confidence intervals.

![](images/ba178b26a853d2a50c0b6d32ba5c8481cac56a838b908ebd3be9b1ba502b33f0.jpg)  
Figure 1: Solution architecture. The critic stream (top) builds the directed adjacency graph � from album-review co-mentions, where �<sub>�</sub> counts mentions of artist � in artist �’s review corpus. The audio stream (bottom) represents each artist as an empirical distribution over 80 Essentia descriptors and compares artist pairs by marginal Wasserstein distances. The two streams converge at the learner: the audio distances are the predictors and the critic graph supplies the supervision, yielding a probability of critical adjacency for any artist pair.

Table 1: Audio recoverability rises with critical attestation (artist-disjoint test set).
<table><tr><td>Edge attestation</td><td>AUC</td></tr><tr><td>All edges (≥ 1 mention)</td><td>0.767</td></tr><tr><td>≥ 2 mentions</td><td>0.788</td></tr><tr><td>≥ 3 mentions</td><td>0.815</td></tr><tr><td>≥ 5 mentions (higher consensus)</td><td>0.865</td></tr></table>

## 5 Results

## 5.1 Construct Validity via Critical Consensus

Across the artist-disjoint test split, the ensemble model recovers critical adjacency with an overall AUC of 0.767 (95% CI 0.761–0.775; Figure 2). Under the less stringent pair-random cross-validation, where an artist may appear in both training and test folds, AUC is 0.803 (95% CI 0.800–0.805); the model is well calibrated (Brier 0.142 against 0.188 for an intercept-only baseline). To test our constructvalidity hypothesis, we evaluate model performance as a function of critical attestation: the number of co-mentions establishing the edge.

The model’s performance exhibits a strong upward progression as attestation increases (Table 1): AUC 0.767 at weight ≥ 1, 0.788 at $\geq 2 , 0 . 8 1 5 \mathrm { a t } \geq 3 ,$ and $0 . 8 6 5 ~ \mathrm { a t } \geq 5 $

When an edge is established via higher critical consensus, it exhibits high audio recoverability, reaching AUC 0.865. This convergence isolates a reproducible “sonic core” characterized by objective acoustic traits such as shared timbre, tempo, and rhythmic structure, indicating that repeated expert co-mentions strongly reflect physi cal musical similarity. Conversely, single-source or long-tail edges display significantly lower audio recoverability. Rather than merely indicating model failure, these single-mention links capture a “soci ological remainder”: a discursive space in which critics construct artist relationships using non-acoustic drivers such as shared personas, subcultural narratives, political positioning, and canonical placement. Human inter-rater agreement on music similarity has natural upper bounds [12]; the AUC of 0.865 on consensus edges reflects a high degree of acoustic alignment given the inherent variance in human evaluative judgments.

Figure 2: AUC-ROC for recovering critic-sourced edges from acoustic distributions  
![](images/979dc1a5c24e88be1d2bb917b4f7b458ebfb9bc222e8fdb375cf8887f015d08f.jpg)  
Note: Pair-random cross-validation (AUC 0.803, 95% CI 0.800–0.805) and the stricter artist-disjoint, cold-start evaluation (AUC 0.767, 95% CI 0.761–0.775), where neither artist in a test pair appears in training. The annotated operating point is the Youden-optimal threshold (0.25) on the artist-disjoint curve (sensitivity 0.69, specificity 0.71). Bands are bootstrap 95% confidence intervals; the pair-random band is narrower than the plotted line.

Table 2: Within-genre artist-disjoint AUC: discrimination on test pairs whose artists share an exogenously assigned genre. Bootstrap 95% CIs. Buckets below 100 pairs are shown for completeness and not interpreted.
<table><tr><td>Genre</td><td>Pairs</td><td>Prev.</td><td>AUC (95% CI)</td></tr><tr><td>Hip-Hop/Rap</td><td>331</td><td>0.82</td><td>0.80 (0.73–0.86)</td></tr><tr><td>Electronic</td><td>273</td><td>0.59</td><td>0.80 (0.74–0.85)</td></tr><tr><td>Metal</td><td>251</td><td>0.53</td><td>0.79 (0.74–0.85)</td></tr><tr><td>Indie</td><td>132</td><td>0.55</td><td>0.81 (0.73–0.88)</td></tr><tr><td>Classic Rock Alternative Rock</td><td>112 109</td><td>0.66 0.58</td><td>0.71 (0.60-0.80) 0.72 (0.62–0.81)</td></tr><tr><td colspan="4">Underpowered (&lt; 100 pairs), not interpreted:</td></tr><tr><td>Country/Folk</td><td>84</td><td>0.63</td><td>0.72 (0.60–0.83)</td></tr><tr><td>Pop</td><td>79</td><td>0.71</td><td>0.68 (0.54–0.81)</td></tr><tr><td>Jazz</td><td>75</td><td>0.71</td><td>0.80 (0.66-0.91)</td></tr><tr><td>R&amp;B</td><td>66</td><td>0.89</td><td>0.95 (0.88–0.99)</td></tr><tr><td>Punk</td><td>51</td><td>0.67</td><td>0.80 (0.66–0.92)</td></tr><tr><td>Classical</td><td>20</td><td>0.70</td><td>0.57 (0.29–0.83)</td></tr></table>

## 5.2 Signal Across Genre Typologies

To evaluate whether acoustic features capture fine-grained stylistic relationships beyond broad category matching, we examine model behavior within exogenously assigned genres (labels drawn from a controlled vocabulary independent of the audio and of the review sources). If the model were merely a genre detector, discrimination should collapse within a genre, where genre is held constant. It does not: within every adequately powered genre, AUC remains high (Table 2).

In a manner consistent with genre typology [18], acoustic recoverability varies systematically across organizational forms. Tightly bounded, scene-based genres (Electronic 0.80, Hip-Hop/Rap 0.80, Metal 0.79, Indie 0.81) show strong within-category recoverability, as critical communities enforce clear acoustic and other conventions [11]. Broad industry-umbrella categories (Classic Rock 0.71, Alternative Rock 0.72, Pop 0.68) exhibit lower acoustic recoverability, as critical co-mentions in these spaces are more frequently driven by historical era, commercial stature, or cultural framing. We note that this pattern is suggestive rather than definitive: several sociologically informative buckets (Jazz, Classical, R&B) fall below the power threshold and are not interpreted.

We further contrast pairs whose artists share a genre against pairs spanning genres. Same-genre AUC is 0.796 (95% CI 0.775– 0.818), slightly higher than cross-genre AUC of 0.765 (95% CI 0.753– 0.777). Were genre the operative cue, same-genre pairs (cue absent) would be hard and cross-genre pairs (cue present) easy; the opposite holds. That cross-genre pairs remain nearly as recoverable is consistent with the aesthetic cosmopolitanism [26] through which critical connections across traditional boundaries often track underlying acoustic afinities.

Table 3: Artist-disjoint AUC stratified by exposure proxy (minimum over each pair). Bins are equal-frequency; degree and mention count collapse to three populated bins because the critic graph is heavy-tailed (most artists have degree 0–1).
<table><tr><td>Proxy</td><td>Bin range</td><td>Pairs</td><td>AUC</td></tr><tr><td rowspan="3">Degree</td><td>0-1 (lowest)</td><td>15,641</td><td>0.717</td></tr><tr><td>2-3</td><td>3,055</td><td>0.721</td></tr><tr><td>4-364 (highest)</td><td>5,955</td><td>0.730</td></tr><tr><td rowspan="3">Mentions</td><td>0-1 (lowest)</td><td>15,196</td><td>0.718</td></tr><tr><td>2-4</td><td>3,583</td><td>0.720</td></tr><tr><td>5–508 (highest)</td><td>5,872</td><td>0.733</td></tr><tr><td rowspan="4">Tracks</td><td>3-11 (lowest)</td><td>6,356</td><td>0.679</td></tr><tr><td>12-24</td><td>6,178</td><td>0.715</td></tr><tr><td>25-52</td><td>6,047</td><td>0.712</td></tr><tr><td>53-2614 (highest)</td><td>6,070</td><td>0.748</td></tr></table>

## 5.3 Robustness to Popularity and Exposure

A natural concern is that recoverability may be confounded by popularity or exposure: well-documented artists accrue more reviews, more mentions, higher graph degree, and often more available tracks, any of which could make their relationships easier to model. We assess this directly by stratifying the artist-disjoint test set (24,651 pairs drawn only from the 3,811 held-out artists) along three exposure proxies: critic-graph degree (number of co-mention neighbours), total mention count (summed co-mention weight), and track count (recordings entering the Wasserstein vectors). Each pair is assigned the minimum proxy value across its two artists, so a pair counts as low-exposure whenever either artist is sparsely documented; pairs are then partitioned into equal-frequency bins and AUC is recomputed within each bin from the already-trained model, with no refitting (Table 3).

Recoverability is essentially invariant to degree (0.717 in the least-connected bin vs. 0.730 in the most-connected) and to mention count (0.718 vs. 0.733), and the least-exposed bins contain the large majority of test pairs, so the signal holds precisely where the critic graph is sparsest. Track count shows the only appreciable gradient (0.679 for artists with 3–11 tracks, rising to 0.748 above 53), which we read as a measurement-precision efect—an artist’s acoustic distribution is estimated more accurately from more tracks—rather than a popularity confound, and recoverability remains well above chance even in the most sparsely-sampled bin. Together these results indicate that acoustic recovery of critic adjacency is not an artifact of well-documented or popular artists being easier to model.

## 6 Discussion & Future Directions

By framing critical adjacency as a construct-validity problem, this study demonstrates that music reviews integrate two distinct layers of signal: a sonically recoverable core supported by higher critical consensus, and a non-acoustic sociological remainder. Underpinning this analysis is a methodological contribution of independent interest: a distributional similarity measure that represents each artist as a distribution over acoustic features and compares artists by their per-feature Wasserstein distances. This measure is domainagnostic and requires no interaction data, and its efectiveness here suggests it may serve as a general content-based similarity signal wherever items can be represented as distributions over features. Future research can operationalize this non-acoustic residual directly. First, incorporating reviewer metadata would allow researchers to test whether critics with higher field-specific capital [8] generate edges with larger non-acoustic residuals compared to trade reviewers. Second, combining audio models with other existing metadata such as geography, record label, mutual collaborations, and artist interviews could help explain the sociological remainder (i.e. the portion of artist similarity not explained by audio content). A third and more applied direction follows directly from the cold-start setting: having established that acoustic distance recovers critic adjacency for unseen artists, the predicted adjacency scores can be used to rank candidate neighbors for a cold-start artist, and evaluated as a retrieval task against audio-only and behavioral baselines with ranking metrics such as NDCG and recall. This reframes the present construct-validity result as the retrieval-side foundation of a unified search-and-recommendation setting, in which an inter pretable, privacy-preserving critic signal is combined with content and interaction signals rather than used in isolation; validating that the signal is acoustically grounded is a prerequisite for such joint models. Ultimately, using audio features as an independent physical baseline allows recommender systems to leverage critical discourse responsibly, using higher-consensus edges for robust cold-start discovery while recognizing single-source commentary as a rich record of cultural and narrative history. In doing so, this work completes a three-part case for critical adjacency as a first-class signal for recommendation and discovery: it is theoretically motivated by the sociology of criticism, internally valid as a graph, coherent under community detection and efective in recommendation simulation [2], and, as shown here, externally grounded in the acoustic content of the music itself.

## Ethics and Privacy Statement

This work uses publicly available audio descriptors and a graph derived from publicly available music publications; it involves no personal data or human subjects. A content-based discovery signal of this kind could reinforce existing coverage biases if deployed uncritically, since both its audio corpus and its critical target overrepresent Anglophone and canonical artists.

## References

[1] Gediminas Adomavicius and Alexander Tuzhilin. 2010. Context-Aware Recom mender Systems. Recommender Systems Handbook (2010), 217–253.

[2] [Author removed for review]. 2025. Modeling Artist Influence for Music Selection and Recommendation: A Purely Network-Based Approach. Harvard Data Science Review 7, 4 (2025). doi:10.1162/99608f92.fb935f61 Author name withheld for double-blind review.

[3] Adam Berenzweig, Beth Logan, Daniel P. W. Ellis, and Brian Whitman. 2004. A Large-Scale Evaluation of Acoustic and Subjective Music-Similarity Measures. Computer Music Journal 28, 2 (2004), 63–76. doi:10.1162/01489260432311225

[4] Dmitry Bogdanov, Alastair Porter, Perfecto Herrera, and Xavier Serra. 2016. Cross-collection Evaluation for Music Classification Tasks. In Proc. ISMIR.

[5] Pierre Bourdieu. 1984. Distinction: A Social Critique of the Judgement of Taste. Harvard University Press.

[6] Pierre Bourdieu. 1993. The Field ofCultural Production: Essays on Artand Literature. Columbia University Press.

[7] William W. Cohen and Wei Fan. 2000. Web-collaborative Filtering: Recommending Music by Crawling the Web. Computer Networks 33, 1–6 (2000), 685–698.

[8] Matteo Corciolani, Kent Grayson, and Ashlee Humphreys. 2020. Do More Experi enced Critics Review Diferently? How Field-Specific Cultural Capital Influences

the Judgments of Cultural Intermediaries. European Journal of Marketing 54, 3 (2020), 478–510.

[9] Yashar Deldjoo, Markus Schedl, and Peter Knees. 2024. Content-driven Music Recommendation: Evolution, State of the Art, and Challenges. Computer Science Review 51 (2024), 100618. doi:10.1016/j.cosrev.2024.100618

[10] Daniel P. W. Ellis, Brian Whitman, Adam Berenzweig, and Steve Lawrence. 2002. The Quest for Ground Truth in Musical Artist Similarity. In Proc. ISMIR. 170–177.

[11] Franco Fabbri. 1982. A Theory of Musical Genres: Two Applications. In Popular Music Perspectives, David Horn and Philip Tagg (Eds.). IASPM, 52–81.

[12] Arthur Flexer and Thomas Grill. 2016. The Problem of Limited Inter-rater Agreement in Modelling Music Similarity. Journal of New Music Research 45, 3 (2016), 239–251.

[13] Simon Frith. 1996. Performing Rites: On the Value ofPopular Music. Harvard University Press.

[14] Paul M. Hirsch. 1972. Processing Fads and Fashions: An Organization-Set Analysis of Cultural Industry Systems. Amer. J. Sociology 77, 4 (1972), 639–659.

[15] Umair Javed, Kamran Shaukat, Ibrahim A. Hameed, Farhat Iqbal, Talha Mahboob Alam, and Suhuai Luo. 2021. A Review of Content-Based and Context-Based Recommendation Systems. International Journal ofEmerging Technologies in Learning 16, 3 (2021), 274. doi:10.3991/ijet.v16i03.1885

[16] Peter Knees and Markus Schedl. 2013. A Survey of Music Similarity and Recommendation from Music Context Data. ACM Trans. Multimedia Comput. Commun. Appl. 10, 1 (2013), 1–21. doi:10.1145/2542205.2542206

[17] Jennifer C. Lena. 2012. Banding Together: How Communities Create Genres in Popular Music. Princeton University Press.

[18] Jennifer C. Lena and Richard A. Peterson. 2008. Classification as Culture: Types and Trajectories of Music Genres. American Sociological Review 73, 5 (2008), 697–718.

[19] Ulf Lindberg. 2005. Rock Criticism from the Beginning: Amusers, Bruisers, and Cool-Headed Cruisers. Vol. 5. Peter Lang.

[20] Brian McFee and Gert R. G. Lanckriet. 2012. Hypergraph Models of Playlist Dialects. In Proc. ISMIR. 343–348

[21] Cory McKay and Ichiro Fujinaga. 2006. Musical Genre Classification: Is It Worth Pursuing and How Can It Be Improved?. In Proc. ISMIR. 101–106.

[22] Keith Negus. 1999. Music Genres and Corporate Cultures. Routledge

[23] Sergio Oramas, Mohamed Sordo, Luis Espinosa-Anke, and Xavier Serra. 2015. A Semantic-Based Approach for Artist Similarity. In Proc. 16th International Society for Music Information Retrieval Conference (ISMIR). 100–106.

[24] Gabriel Peyré and Marco Cuturi. 2019. Computational Optimal Transport. Foundations and Trends in Machine Learning 11, 5–6 (2019), 355–607. doi:10.1561/ 2200000073

[25] Miloš Radovanović, Alexandros Nanopoulos, and Mirjana Ivanović. 2009. Nearest Neighbors in High-Dimensional Data: The Emergence and Influence of Hubs. In Proc. 26th ICML. 865–872.

[26] Motti Regev. 2013. Pop-Rock Music: Aesthetic Cosmopolitanism in Late Modernity. John Wiley & Sons.

[27] Markus Schedl, Emilia Gómez, and Julián Urbano. 2014. Music Information Retrieval: Recent Developments and Applications. Foundations and Trends in Information Retrieval 8, 2–3 (2014), 127–261.

[28] Markus Schedl, Hamed Zamani, Ching-Wei Chen, Yashar Deldjoo, and Mehdi Elahi. 2018. Current Challenges and Visions in Music Recommender Systems Research. International Journal ofMultimedia Information Retrieval 7, 2 (2018), 95–116.

[29] Andrew I. Schein, Alexandrin Popescul, Lyle H. Ungar, and David M. Pennock. 2002. Methods and Metrics for Cold-Start Recommendations. In Proc. 25th ACM SIGIR. 253–260. doi:10.1145/564376.564421

[30] Klaus Seyerlehner, Gerhard Widmer, and Peter Knees. 2010. A Comparison of Human, Automatic and Collaborative Music Genre Classification. In International Workshop on Adaptive Multimedia Retrieval. 118–131.

[31] Daniel Silver, Monica Lee, and C. Clayton Childress. 2016. Genre Complexes in Popular Music. PLOS ONE 11, 5 (2016), e0155471. doi:10.1371/journal.pone. 0155471

[32] Bob L. Sturm and Arthur Flexer. 2023. A Review of Validity and Its Relationship to Music Information Research. In Proc. ISMIR. 47–55.

[33] Nikita Vahutre, Sujata S. Adagale, and Priya A. Kadam. 2025. A Comprehensive Survey of Content-Based Music Recommendation Techniques. In Business Data Analytics. Vol. 2358. Springer Nature Switzerland, 360–368. doi:10.1007/978-3- 031-80778-7\_26

[34] Aäron van den Oord, Sander Dieleman, and Benjamin Schrauwen. 2013. Deep Content-based Music Recommendation. In Advances in Neural Information Processing Systems, Vol. 26.

[35] Mark J. van der Laan, Eric C. Polley, and Alan E. Hubbard. 2007. Super Learner. Working Paper 222. U.C. Berkeley Division of Biostatistics. https://biostats. bepress.com/ucbbiostat/paper222

[36] Cédric Villani. 2009. Optimal Transport: Old and New. Vol. 338. Springer Berlin Heidelberg. doi:10.1007/978-3-540-71050-9

[37] Geraint A. Wiggins. 2009. Semantic Gap?? Schemantic Schmap!! Methodolog ical Considerations in the Scientific Study of Music. In 11th IEEE International

Symposium on Multimedia. 477–482.