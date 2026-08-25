# From Multimodal Observation to Interpretable Suggestions: Counterfactual Time-Expanded Relational Modeling of Surgical Teams

Vincenzo Marco De Luca   
vincenzomarco.deluca@unitn.it University of Trento Trento, Italy   
Giovanna Varni   
giovanna.varni@unitn.it   
University of Trento   
Trento, Italy

Antonio Longa antonio.longa@uit.no UiT the Arctic University of Norway Tromsø, Norway

Andrea Passerini   
andrea.passerini@unitn.it   
University of Trento   
Trento, Italy

## Abstract

In surgery, patient safety is threatened not only by technical issues but also by poor teamwork. However, existing surgical AI-based solutions focus mainly on visual workflow and technical execution, neglecting the modeling of team interactions and missing opportunities to actively support clinicians in improving their teamwork skills. To address this gap, we propose a tempo-relational framework for modeling surgical team dynamics from multimodal observations. By leveraging Time-Expanded graphs, the approach captures both relational structure and temporal evolution, achieving strong expressivity while remaining robust in the low-data regime typical of surgical settings. Beyond prediction, such modeling enables the generation of eficient, interpretable, and actionable suggestions for clinicians. More specifically, we generate suggestions via a counterfactual procedure that identifies minimal yet structured changes in individual behaviors and interaction patterns associated with improvements in team performance. Experiments with simulated surgical procedures show that our approach improves predictive performance in diverse behavioral and interaction goals while ofering meaningful insights into team dynamics. This work advances surgical AI beyond outcome-driven prediction towards a socially grounded, team-centric, and actionable paradigm to better understand and support the development of team skills in surgical settings.

## CCS Concepts

• Human-centered computing → Collaborative and social computing; • Applied computing → Life and medical sciences; • Computing methodologies → Machine learning.

## Keywords

Surgical Data Science, Graph Learning, Actionable AI, Counterfactual Explainability, Team Modeling

ACM Reference Format: Vincenzo Marco De Luca, Antonio Longa, Giovanna Varni, and Andrea Passerini. 2026. From Multimodal Observation to Interpretable Suggestions: Counterfactual Time-Expanded Relational Modeling of Surgical Teams . In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3767308.3836430

## 1 Introduction

Actionable AI aims to move beyond prediction by enabling systems that provide interpretable and actionable feedback to support humans in their work and everyday tasks. This requirement is critical in high-stakes settings, such as operating rooms (ORs), in which team performance arises from complex, dynamic, and tightly coordinated interactions. Accurate predictions alone are insuficient; decision support must translate the system’s outputs into actionable guidance that enhances team performance. The OR scenario exemplifies such high-stakes settings. While Surgical Data Science (SDS) has made significant progress in modeling procedural workflows [27, 40, 58, 61] and individual technical skills [29, 32], modeling intra-operative team dynamics and providing actionable insights remain largely underexplored. Building computational models that capture temporal-relational patterns in small-team interactions is particularly challenging in surgical settings due to lim ited data availability and its complex technical setup. Additionally, high-dimensional multimodal data require generalizable models that can leverage all modalities without overfitting.

In this work, we propose a graph-based multimodal team modeling approach particularly suitable for high-stakes scenarios. More specifically, in our setting, the model predicts social and behavioral constructs representing the overall team dynamics. To capture temporal evolution directly within the relational structure, we construct Time-Expanded graphs, enabling message passing along both spatial and temporal dimensions within a single architecture. This approach preserves the expressive power of the model while remaining robust in low-data regimes. Building on this team representation, we introduce a counterfactual approach for actionable reasoning. Unlike previous approaches, which focus solely on the removal of edges [39] or combinatorial enumeration of possible alternative behaviors [50], our method identifies minimal and interpretable team behavioral changes to improve teamwork outcomes based on the previous history observed in the training data. These counterfactual insights are validated in surgical training settings, in which clinical teams perform procedures on high-fidelity simulators, ensuring both validity and reproducibility.

The contributions of this paper are as follows:

• We propose TE-ReNN, a Time-Expanded Relational Neural Network that models multimodal team interactions while remaining robust in the low-data regime;

• We introduce a graph-based counterfactual approach that provides interpretable and actionable recommendations to improve teamwork;

• We demonstrate that our approach outperforms existing approaches in surgical training simulation, highlighting robust multimodal modeling, team-level prediction, and validated counterfactual insights.

## 2 Related Work

Automated team modeling has attracted growing interest, but progres remains limited by the dificulty of collecting multimodal data in multiparty settings and the scarcity of large-scale datasets [6, 9, 10, 30, 46, 52]. Early work focus on static team modeling, in which multi modal features are aggregated across individuals and used to predict team constructs such as cohesion, leadership, or performance via classical ML or shallow neural models [7, 25, 31, 45, 47, 48]. How ever, these approaches ignore temporal dynamics and interpersonal interactions. Temporal models address this limitation by capturing behavioral evolution over time using sequence-based methods such as HMMs, LSTMs, and Transformers [1, 3, 35], as well as dynamical systems and imitation learning [21, 54]. While efective at modeling temporal dependencies, they typically treat team members inde pendently and overlook relational structure. Conversely, relational approaches incorporate interactions among team members through graph-based or network-based models, including Graph Neural Networks (GNNs) and attention mechanisms [5, 33, 34, 55], but often ne glect temporal dynamics or focus on limited tasks. Recent temporal GNNs attempt to bridge this gap [36], yet they are rarely applied to multi-construct team modeling and often lack comprehensive evalu ation. Overall, existing approaches either model time or relations in isolation, leaving a gap for unified approaches that jointly capture temporal, relational, and multimodal aspects of team behavior. A notable exception is the recent work by De Luca et al. [38], which introduces tempo-relational neural networks to model teamwork. However, their approach sufers from the low-data regime typical of medical team analysis, where models are trained on a set of recordings. Indeed, the approach has only been evaluated on much simpler cooperative survival games performed in the laboratory. Our approach sidesteps this problem by directly encoding temporal dependencies within the graph structure. This design enables efec tive message-passing while remaining robust in the low-data and small team regime typical of operating room scenarios. Our experi mental evaluation demonstrates that this constitutes a substantial advantage over alternative approaches. Human-AI collaboration re search has demonstrated that hybrid teams outperform humans or AI alone only when interaction is well-calibrated and systems sup port transparent and adaptive cooperation [2, 42]. Trust calibration and shared mental models are key determinants of team performance that can be supported by multiple explanation modalities while mitigating over-reliance on AI recommendations [17, 22]. In particular, counterfactual explanations provide a formal framework to estimate the efects of hypothetical interventions, enabling systems to reason about what could have happened under alternative actions. For this reason, they have been widely adopted in machine learning to interpret model decisions, often as post-hoc diagnostic tools applied to individual predictions [60]. In high-stakes domains such as healthcare, counterfactual methods have been used to guide treatment selection [26, 53], yet these approaches rarely consider interactions among multiple decision makers or provide structured feedback that can inform team-level decision processes. Similarly, counterfactual policy learning methods optimize intervention outcomes but often ignore the temporal and social dynamics inher ent in collaborative environments. Some recent studies integrate temporal and relational modeling to support counterfactual generation [37, 50], yet these methods are often used as predictive or explanatory tools rather than mechanisms for actionable feedback. From a computational perspective, small-team scenarios introduce additional challenges due to limited data availability, sparsity, and the absence of publicly labeled datasets. Despite these advances, existing approaches present notable limitations. Most counterfactual methods neglect the complex interactions among multiple agents over time. Explanations are rarely translated into actionable guidance for workflow decisions, and the dynamic evolution of teams during collaboration is often unmodeled. Our work addresses these gaps with Time-Expanded Relational Neural Networks (TE-ReNN), which generate structured counterfactual alternatives capturing both temporal evolution and social interaction patterns. Unlike prior methods, TE-ReNN provides actionable feedback at the level of individual decisions, team members’ interactions, and collective outcomes, supporting team decision-making in high-stakes operational contexts rather than producing post-hoc explanations detached from the decision process.

## 3 Method

## 3.1 Time-Expanded Relational Neural Networks

We represent each surgical team as a sequence of multimodal interaction graphs built over fixed temporal windows. The input comprises (i) multimodal features extracted from synchronized audio-video recordings and annotations, including acoustic, motion, and human–object interaction cues, and (ii) team conversation transcripts. Since these modalities are acquired at diferent sampling rates, all features were aggregated into aligned non-overlapping windows of 15 seconds.

Let V denote the set of all team members involved in the surgery. For each temporal window $t \in \{ 1 , . . . , T \}$ , we first build a snapshot interaction graph

$$
G _ { t } = ( V _ { t } , E _ { t } ) ,
$$

where $V _ { t } \subseteq \mathcal { V }$ is the set of team members present during window �. Edges in $E _ { t }$ encode verbal communication assuming that speech is broadcast to all team members present: whenever a team member $v _ { i } \in V _ { t }$ speaks during window �, directed edges $( v _ { i } , v _ { j } )$ are added from the speaker $v _ { i }$ to all the other team members present in that window $v _ { j } \in V _ { t }$ with $j \neq i .$ . This assumption models verbal communication as a one-to-many interaction, reflecting the shared acoustic environment of the OR. Each node $\upsilon \in V _ { t }$ is associated with a multimodal feature vector composed of:

![](images/73b0f09e7ae566b123056d36b331a48a2d5b5c7c8e5a4cbcc561298fa22fabc1.jpg)  
Figure 1: Overview of the interaction modeling pipeline. Top row: multimodal time-series are segmented into fixed temporal windows (15 seconds). Middle row: for each window, a snapshot interaction graph $G _ { t }$ is built, in which nodes represent team members and edges encode broadcast verbal communication. Node features integrate interpretable paralinguistic (eGeMAPS), text, position, and human–tool interaction (relation with objects). Bottom row: snapshot graphs are connected through identity preserving temporal edges into a Time-Expanded Graph $\bar { G } ^ { T E }$ , enabling joint modeling of team members’ coordination and longitudinal individual dynamics across time.

• Paralinguistic features: loudness, alpha-ratio, and harmonicsto-noise ratio from the eGeMAPS 2.0 set [18]. The selection of this subset is detailed in Sec. 4.2.

• Text features: language-model embeddings extracted from the speech transcription associated with each team member speaking within a temporal window.

• Motion features: the spatial position $( x , y )$ of each team member, together with the mean and standard deviation of their displacement within a temporal window.

• Human–tool interaction features: the triplet team member— action—object describing atomic actions in the OR (e.g., anesthesiologist—calibrating—instrument). They are encoded via one-hot vectors and associated with the corresponding team member.

• Role features: a categorical label reporting the functional role of each team member (e.g., head surgeon, nurse), encoded using a one-hot representation.

To capture team coordination dynamics beyond the windowbased resolution, we transform the sequence of snapshot graphs into a Time-Expanded graph, which explicitly encodes temporal evolution in the graph topology, as illustrated in Fig. 1. Given a temporal sequence of snapshots

$$
\mathcal { G } _ { 1 : T } = [ G _ { 1 } , \dots , G _ { T } ] ,
$$

we define the corresponding Time-Expanded graph as

$$
G ^ { T E } = ( V ^ { T E } , E ^ { T E } ) ,
$$

with

$$
V ^ { T E } = \{ v _ { i } ^ { t } \mid v _ { i } \in V _ { t } , \ t = 1 , . . . , T \} .
$$

Each node $\boldsymbol { v } _ { i } ^ { t }$ represents team member $v _ { i }$ at the temporal window �. Consequently, multiple temporal instances of the same team member coexist in the graph, allowing their behavior to be tracked throughout the surgical procedure. To capture both instantaneous and longitudinal dependencies, the edge set is defined as

$$
E ^ { T E } = E ^ { s n a p } \cup E ^ { t e m p } ,
$$

where:

• Intra-snapshot edges:

$$
E ^ { s n a p } = \{ ( v _ { i } ^ { t } , v _ { j } ^ { t } ) ~ | ~ ( v _ { i } , v _ { j } ) \in E _ { t } \} .
$$

These edges preserve the communication structure observed within each temporal window and capture broadcast interactions from a speaking team member $\boldsymbol { v } _ { i } ^ { t }$ to the other team members $\boldsymbol { v } _ { j } ^ { t }$ in the same snapshot.

• Inter-snapshot identity edges:

$$
E ^ { t e m p } = \{ ( v _ { i } ^ { t } , v _ { i } ^ { t + 1 } ) \mid v _ { i } \in V _ { t } \cap V _ { t + 1 } , \ t = 1 , \ldots , T - 1 \} .
$$

These edges connect consecutive temporal instances of team members and preserve identity continuity over time.

This construction encodes temporal dynamics directly into the graph topology, allowing information to propagate both across team members within each window and along the temporal trajectory of every individual. In our implementation, each Time-Expanded graph spans up to six minutes of audio-video recordings, that is corresponding to 24 temporal windows, providing suficient temporal context to capture evolving coordination patterns while remaining computationally tractable. By combining relational structure with interpretable behavioral abstraction, the proposed representation enables the learning of predictive signals of procedural eficiency while maintaining a direct link between model representations and human-adjustable behaviors. This representation is naturally compatible with Message Passing Graph Neural Networks (MP GNNs) [20, 28, 43, 62], as it defines a unified graph space in which information can be iteratively aggregated from both relational and temporal neighborhoods. In particular, the intra-snapshot edges enable the exchange of contextual information across team members within the same temporal window, while the inter-snapshot identity edges enable each representation to evolve along its own temporal trajectory. This design extends standard message passing to a tempo-relational setting, where node embeddings are updated by jointly integrating social interactions and their temporal evolution, without requiring explicit recurrent mechanisms [19]. The resulting Time-Expanded Relational Neural Network (TE-ReNN) is better suited to low-data settings than recurrent-based approaches, as it encodes temporal dependencies directly within the graph structure rather than relying on sequential state propagation, thus better leveraging the message-passing capabilities of graph neural network architectures. In the low-data regime typical of surgical procedures, in which recordings are scarce and annotations are costly, this leads to consistently improved predictive performance across tasks, as demonstrated in our experimental evaluation.

## 3.2 Actionable Counterfactuals

We generate actionable feedback by identifying how a team interaction sequence could evolve toward a more desirable outcome. Let

$$
\mathcal { G } _ { 1 : T } = [ G _ { 1 } , . . . , G _ { T } ]
$$

denote an input sequence of snapshot graphs, and let $y ^ { * }$ denote a target outcome $( \mathrm { e . g . } ,$ , high cooperation). The goal is to extract interpretable modifications to team behaviors that steer the prediction of the model towards $y ^ { * }$ . Note that in this setting, in which multiple team members interact over extended time horizons, standard counterfactual approaches that seek minimal changes to flip a model’s prediction [44, 60] are not directly applicable. Modifying the behavior of a single team member $( \mathrm { e . g . }$ , the lead surgeon) at a given time step may yield a more favorable predicted outcome, but such a counterfactual is unrealistic, as it ignores the cascading efects on other team members and their behaviors in subsequent time steps. To sidestep this issue, we adopt an exemplar-based strategy grounded in real observations, following example-based counterfactual literature [12, 16], where explanations are obtained by comparing an instance with a similar but more desirable example. Accordingly, throughout the paper we refer to our method as an example-based counterfactual explanation. Given a sequence exhibiting suboptimal team performance, we retrieve a sequence with a similar temporal prefix but a desirable outcome, and compare the two prefixes to identify the key behavioral diferences underlying the divergence in performance.

Algorithm 1 Embedding Repository Construction   
Require: Training set ${ \mathcal { D } } ,$ encoder $f _ { \theta }$   
Ensure: Repository $z$   
1: $z \gets 0$   
2: for each sequence $\mathcal { G } _ { 1 : T _ { i } } ^ { i } = [ G _ { 1 } ^ { i } , . . . , G _ { T _ { i } } ^ { i } ] \in \mathcal { D }$ do   
3: for $t = 1$ to $T _ { i }$ do   
4: $\mathcal { G } _ { 1 : t } ^ { i } \gets [ G _ { 1 } ^ { i } , . . . , G _ { t } ^ { i } ]$   
5: construct the corresponding Time-Expanded graph   
$G _ { i , \leq t } ^ { T E }$   
6: $\mathbf { z } _ { \leq t } ^ { i }  f _ { \theta } ( G _ { i , \leq t } ^ { T E } )$   
7: $\mathcal { Z }  \mathcal { Z } \cup \{ ( \mathbf { z } _ { \leq t } ^ { i } , y ^ { i } , \mathcal { G } _ { 1 : t } ^ { i } ) \}$   
8: end for   
9: end for

Overview. The method consists of two stages: (i) ofline construction of an embedding repository, and (ii) online counterfactual generation through retrieval, explanation, and comparison.

Embedding repository. We precompute a repository $z$ of prefixlevel embeddings from the training set, as detailed in Algorithm 1. For each training sequence

$$
\boldsymbol { \mathcal { G } } _ { 1 : T _ { i } } ^ { i } = [ G _ { 1 } ^ { i } , \dots , G _ { T _ { i } } ^ { i } ] ,
$$

we consider all prefixes

$$
\begin{array} { r } { \mathcal { G } _ { 1 : t } ^ { i } = [ G _ { 1 } ^ { i } , . . . , G _ { t } ^ { i } ] , \qquad t = 1 , . . . , T _ { i } . } \end{array}
$$

Each prefix is converted into its corresponding Time-Expanded graph and encoded using the trained GNN $f _ { \theta } ,$ producing a graph embedding $ { \mathbf { z } } _ { \le t } ^ { i }$ . Each entry in $z$ stores the tuple

$$
( \mathbf { z } _ { \leq t } ^ { i } , y ^ { i } , \mathcal { G } _ { 1 : t } ^ { i } ) ,
$$

where $y ^ { i }$ is the outcome label associated with sequence �. This step is performed once and reused at inference time, enabling eficient retrieval of temporally aligned interaction patterns.

ii) Counterfactual generation. At inference time, we analyze the input incrementally using Algorithm 2. For each prefix

$$
\begin{array} { r } { G _ { 1 : t } = [ G _ { 1 } , \dots , G _ { t } ] , } \end{array}
$$

we construct the corresponding Time-Expanded graph $G _ { \leq t } ^ { T E }$ and encode it with $f _ { \theta }$ to obtain the embedding $\mathbf { z } _ { \leq t } .$ We then query the embedding repository $z$ via the function RetrieveImproved to retrieve the embedding which is closest to $\mathbf { Z } _ { \leq t }$ according to cosine similarity, has the same temporal length � and exhibits the desired outcome $y ^ { * }$

Algorithm 2 Counterfactual Generation   
Require: Input sequence $\mathcal { G } _ { 1 : T } = [ G _ { 1 } , . . . , G _ { T } ]$ , target outcome $y ^ { * } { \mathrm { . } }$   
repository $z ,$ encoder $f _ { \theta }$   
Ensure: Counterfactual explanation �   
1: for � = 1 to � do   
2: $\mathcal { G } _ { 1 : t } \gets [ G _ { 1 } , . . . , G _ { t } ]$   
3: construct the corresponding Time-Expanded graph $G _ { \leq t } ^ { T E }$   
4: $\mathbf { z } _ { \leq t } \gets f _ { \theta } ( G _ { \leq t } ^ { T E } )$   
5: $( \mathbf { z } _ { \leq t } ^ { * } , y ^ { * } , \mathcal { G } _ { 1 : t } ^ { * } )$ ← RetrieveImproved $( \mathbf { z } _ { \leq t } , \mathcal { Z } , y ^ { \ast } )$   
6: E ← Explain $( \mathcal { G } _ { 1 : t } ^ { * } , f _ { \theta } )$   
7: for each candidate explanation $e \in { \mathcal { E } }$ do   
8: if Match(�, $\mathbf { \ } _ { G _ { 1 : t } } ) =$ false then   
9: return �   
10: end if   
11: end for   
12: end for   
13: return ∅

$$
( \mathbf { z } _ { \leq t } ^ { * } , y ^ { * } , \mathcal { G } _ { 1 : t } ^ { * } )
$$

Next, we apply Explain to the retrieved prefix $\mathcal { G } _ { 1 : t } ^ { * }$ to identify which elements most strongly support its prediction. Since faithfulness assessments for GNN explanations are sensitive to the underlying architecture [4], we use saliency scores to rank candidate behavioral components. Starting from feature-level saliency scores [56], we derive importance estimates at the node and edge levels, and aggregate them into interpretable behavioral components, such as individual actions or communication cues. This yields an ordered explanation of the retrieved example, ranked from the most to the least influential components.

We then compare these components with the current prefix $\mathcal { G } _ { 1 : t }$ using Match, where we exploit the role labels available in the graph to compare components at role-consistent positions, e.g., surgeon-to-surgeon or nurse-to-nurse. The candidate explanations are examined in decreasing order of importance, and the first salient component that is absent from the input is returned as a counterfactual suggestion. If all salient components are already present, the algorithm proceeds by extending the prefix. Further details on RetrieveImproved, Explain, and Match are reported in the Supplementary Material.

The proposed approach combines example-based retrieval with explanation-based refinement. Retrieval ensures candidate behaviors are realistic by grounding them in observed interaction sequences, while explanations isolate the components responsible for the desirable outcome. Comparing these components with the current interaction sequence yields actionable suggestions and identifies the earliest point an intervention could be recommended.

## 4 Experimental setting

## 4.1 Dataset

The MM-OR [49] dataset is a multimodal corpus of 21 simulated knee replacement surgical procedures, including 14 full and 7 partial knee replacement surgeries. In our experiments, we use 15 procedures, nine of which are full knee replacements. Each procedure typically involves four to six team members, whose functional roles are explicitly annotated, and may include either expert surgeons or trainees. The surgeries are audio-video recorded using five cameras and a single environmental microphone. The dataset also includes annotations about the use of specific tools by team members. Behavioral annotations were performed manually using the validated Observational Teamwork Assessment for Surgery (OTAS) [24] behavioral rating schema, which includes five behavioral dimensions (Communication, Coordination, Cooperation, Leadership, and Situational Awareness) structured across 15 items on a 7-point Likert scale. Three annotators independently scored the videos at sixminute intervals; the resulting scores were averaged and rounded to the nearest value on the Likert scale.

<table><tr><td>Activation</td><td>Control</td><td>Dominance</td><td>Behavioral Class</td></tr><tr><td>Low</td><td>Low</td><td>Low</td><td>Withdrawn / Disengaged</td></tr><tr><td>Low</td><td>Low</td><td>High</td><td>Restrained Conflict</td></tr><tr><td>Low</td><td>High</td><td>Low</td><td>Calm-Cooperative</td></tr><tr><td>Low</td><td>High</td><td>High</td><td>Quiet Authority</td></tr><tr><td>High</td><td>Low</td><td>Low</td><td>Agitated / Over-aroused</td></tr><tr><td>High</td><td>Low</td><td>High</td><td>Dominant-Aggressive</td></tr><tr><td>High</td><td>High</td><td>Low</td><td>Engaged-Cooperative</td></tr><tr><td>High</td><td>High</td><td>High</td><td>Calm Leader</td></tr></table>

Table 1: Behavioral classes defined by combining three paralinguistic features extracted from the eGeMAPS set.

## 4.2 Pre-processing

Automated diarization of audio recordings was performed using Pyannote [11], and transcripts were generated with Whisper [51]. Due to the use of a single environmental microphone, background noise, and frequent code-switching between English and German, these automated methods proved unreliable. Therefore, two annotators manually corrected both the diarization and the transcripts to ensure high-quality data. Paralinguistic feature extraction relies on accurate speaker diarization. Features are extracted independently for each participant: non-speaking intervals are masked and assigned neutral values, while eGeMAPS 2.0 [18] features are extracted for active speakers. To enhance interpretability and enable counterfactual analyses, we selected three features: average loudness, which reflects vocal activation and engagement; harmonicsto-noise ratio, which captures vocal control and tension; and alpha ratio, which is associated with vocal dominance. Each combination of these three indices defines a behavioral class (Tab. 1), allowing each temporal audio window to be mapped to higherlevel behavioral abstractions, such as “Engaged–Cooperative” or “Dominant–Aggressive”. Importantly, this discretization is applied only for interpretability; the model retains continuous eGeMAPS values during training to preserve fine-grained acoustic information and predictive fidelity. This separation ensures that interpretability does not come at the expense of modeling capacity. Textual data from manually corrected transcripts, which include multiple languages, were processed using XLM-RoBERTa [14], a multilingual language model pretrained on over 100 languages. Computer vision features were extracted by detecting each team member in three of the five available camera views and encoding their spatial position, movement, and additional nonverbal cues as feature vectors.

<table><tr><td rowspan="2">Paradigm</td><td rowspan="2">Model</td><td colspan="3">Modeling Dimension</td><td rowspan="2">Avg. F1-macro (%)</td><td colspan="5">F1-macro Measures (%)</td></tr><tr><td>Feature</td><td>Time</td><td>Topology</td><td>Teamwork Comm</td><td>Coord</td><td>Coop</td><td>Lead</td><td> $\mathrm { S i t . \ A w }$ </td></tr><tr><td rowspan="2">SNN</td><td>MLP</td><td>√</td><td>x</td><td>X</td><td> $5 9 . 8 \pm 2 . 2$ </td><td> $6 0 . 1 \pm 2 . 0$ </td><td> $5 8 . 7 \pm 2 . 3 $ </td><td> $6 0 . 3 \pm 2 . 5$ </td><td> $5 9 . 2 \pm { 1 . 8 }$ </td><td> $6 0 . 7 \pm 2 . 2$ </td></tr><tr><td>RF</td><td>√</td><td>X</td><td>X</td><td> $6 0 . 2 \pm 0 . 8$ </td><td> $6 1 . 0 \pm 0 . 9$ </td><td> $5 9 . 5 \pm 0 . 5$ </td><td> $6 0 . 0 \pm 0 . 6$ </td><td> $6 0 . 3 \pm 1 . 0$ </td><td> $6 0 . 3 \pm 1 . 2$ </td></tr><tr><td rowspan="3">TNN</td><td>LSTM</td><td>√</td><td>L</td><td>X</td><td> $6 2 . 3 \pm 1 . 6$ </td><td> $6 3 . 0 \pm 2 . 4$ </td><td> $6 1 . 8 \pm 1 . 4$ </td><td> $6 2 . 5 \pm 1 . 8$ </td><td> $6 2 . 0 \pm 1 . 1$ </td><td> $6 2 . 3 \pm 1 . 1$ </td></tr><tr><td>GRU</td><td></td><td></td><td>x</td><td> $6 1 . 8 \pm 1 . 8$ </td><td> $6 2 . 2 \pm 2 . 1$ </td><td> $6 1 . 5 \pm 1 . 7$ </td><td> $6 1 . 7 \pm 2 . 3$ </td><td> $6 1 . 8 \pm 1 . 7$ </td><td> $6 1 . 8 \pm 1 . 2$ </td></tr><tr><td>MHA</td><td>√</td><td>√</td><td>X</td><td> $6 3 . 8 \pm 2 . 0$ </td><td> $6 4 . 0 \pm 2 . 1$ </td><td> $6 3 . 5 \pm 2 . 5$ </td><td> $6 3 . 9 \pm 2 . 8$ </td><td> $6 3 . 6 \pm 1 . 4$ </td><td> $6 3 . 9 \pm 1 . 4$ </td></tr><tr><td rowspan="3">RENN</td><td>GCN</td><td>√</td><td>X</td><td>√</td><td> $6 1 . 4 \pm 1 . 5$ </td><td> $6 1 . 6 \pm 1 . 4$ </td><td> $6 1 . 2 \pm 1 . 2$ </td><td> $6 1 . 5 \pm 1 . 5$ </td><td> $6 1 . 3 \pm 1 . 5$ </td><td> $6 1 . 5 \pm 1 . 8$ </td></tr><tr><td>GAT</td><td></td><td>x</td><td></td><td> $6 2 . 1 \pm 1 . 8$ </td><td> $6 2 . 4 \pm 1 . 9$ </td><td> $6 1 . 8 \pm 2 . 4$ </td><td> $6 2 . 0 \pm 2 . 1$ </td><td> $6 2 . 0 \pm 1 . 2$ </td><td> $6 2 . 2 \pm 1 . 5$ </td></tr><tr><td>GIN</td><td></td><td>X</td><td>V</td><td> $6 5 . 2 \pm 1 . 5$ </td><td> $6 5 . 0 \pm 1 . 6$ </td><td> $6 5 . 1 \pm 1 . 4$ </td><td> $6 5 . 3 \pm 1 . 3$ </td><td> $6 5 . 2 \pm 1 . 8$ </td><td> $6 5 . 4 \pm 1 . 9$ </td></tr><tr><td rowspan="3">Space-then-Time TRENN</td><td>GCN+MHA</td><td></td><td></td><td>√</td><td> $6 4 . 0 \pm 1 . 5$ </td><td> $6 4 . 1 \pm 1 . 7$ </td><td> $6 3 . 8 \pm 1 . 9$ </td><td> $6 4 . 2 \pm 1 . 4$ </td><td> $6 3 . 9 \pm 1 . 2$ </td><td> $6 4 . 0 \pm 1 . 4$ </td></tr><tr><td>GAT+MHA</td><td></td><td></td><td></td><td> $6 7 . 2 \pm 1 . 7$ </td><td> $6 7 . 5 \pm 2 . 2$ </td><td> $6 6 . 8 \pm 1 . 8$ </td><td> $6 7 . 3 \pm 1 . 8$ </td><td> $6 7 . 0 \pm 1 . 0$ </td><td> $6 7 . 2 \pm 1 . 5$ </td></tr><tr><td>GIN+MHA</td><td></td><td></td><td></td><td> $6 3 . 2 \pm 1 . 5$ </td><td> $6 3 . 0 \pm 1 . 8$ </td><td> $6 3 . 1 \pm 1 . 5$ </td><td> $6 3 . 3 \pm 1 . 5$ </td><td> $6 3 . 2 \pm 1 . 3$ </td><td> $6 3 . 2 \pm 1 . 4$ </td></tr><tr><td rowspan="3">Time-then-Space TRENN</td><td>MHA+GCN</td><td></td><td></td><td>√</td><td> $6 5 . 2 \pm 1 . 3$ </td><td> $6 4 . 6 \pm 0 . 9$ </td><td> $6 6 . 0 \pm 1 . 8$ </td><td> $6 6 . 0 \pm 1 . 1$ </td><td> $6 4 . 4 \pm 1 . 8$ </td><td> $6 5 . 0 \pm 0 . 9$ </td></tr><tr><td> $\mathrm { M H A ^ { + } G A T }$ </td><td></td><td></td><td></td><td> $6 5 . 8 \pm 1 . 5$ </td><td> $6 6 . 4 \pm 1 . 7$ </td><td> $6 5 . 2 \pm 1 . 3$ </td><td> $6 5 . 8 \pm 1 . 1$ </td><td> $6 5 . 0 \pm 2 . 0$ </td><td> $6 6 . 6 \pm 1 . 4$ </td></tr><tr><td>MHA+GIN</td><td></td><td></td><td></td><td> $6 4 . 2 \pm 1 . 2$ </td><td> $6 4 . 6 \pm 1 . 2$ </td><td> $6 3 . 8 \pm 0 . 7$ </td><td> $6 4 . 4 \pm 1 . 6$ </td><td> $6 4 . 3 \pm 1 . 6$ </td><td> $6 3 . 9 \pm 0 . 9$ </td></tr><tr><td rowspan="3">Standard TRENN</td><td>TGN</td><td></td><td></td><td>√</td><td> $6 5 . 0 \pm 1 . 4$ </td><td> $6 4 . 8 \pm 1 . 5$ </td><td> $6 5 . 2 \pm 1 . 6$ </td><td> $6 5 . 3 \pm 1 . 3$ </td><td> $6 4 . 9 \pm 1 . 4$ </td><td> $6 5 . 0 \pm 1 . 5$ </td></tr><tr><td>DySAT</td><td></td><td></td><td>√</td><td> $6 5 . 6 \pm 1 . 3$ </td><td> $6 5 . 8 \pm 1 . 4$ </td><td> $6 5 . 4 \pm 1 . 5$ </td><td> $6 5 . 7 \pm 1 . 3$ </td><td> $6 5 . 5 \pm 1 . 1$ </td><td> $6 5 . 6 \pm 1 . 2$ </td></tr><tr><td>EvolveGCN</td><td></td><td></td><td></td><td> $6 6 . 0 \pm 1 . 3$ </td><td> $6 5 . 8 \pm 1 . 4$ </td><td> $6 6 . 1 \pm 1 . 6$ </td><td> $6 6 . 2 \pm 1 . 3$ </td><td> $6 5 . 9 \pm 1 . 4$ </td><td> $6 6 . 0 \pm 1 . 2$ </td></tr><tr><td rowspan="3">TE-RENN</td><td>TE-GCN</td><td></td><td></td><td>√</td><td> $7 2 . 4 \pm 1 . 7$ </td><td> $7 2 . 0 \pm 1 . 9$ </td><td> $7 2 . 2 \pm 1 . 9$ </td><td> $7 2 . 5 \pm 1 . 4$ </td><td> $7 2 . 3 \pm 1 . 6$ </td><td> $7 2 . 5 \pm 1 . 6$ </td></tr><tr><td>TE-GAT</td><td></td><td></td><td></td><td> $7 3 . 1 \pm 1 . 7$ </td><td> $7 3 . 0 \pm 1 . 7$ </td><td> $7 3 . 2 \pm 1 . 9$ </td><td> $7 3 . 3 \pm 1 . 8$ </td><td> $7 3 . 1 \pm 1 . 1$ </td><td> $7 3 . 0 \pm 1 . 9$ </td></tr><tr><td>TE-GIN</td><td></td><td></td><td></td><td> $7 4 . 2 \pm 1 . 7$ </td><td> $7 4 . 0 \pm 1 . 8$ </td><td> $7 4 . 1 \pm 1 . 6$ </td><td> $7 4 . 3 \pm 1 . 9$ </td><td> $7 4 . 2 \pm 1 . 2$ </td><td> $7 4 . 4 \pm 2 . 1$ </td></tr></table>

Table 2: Average F1-macro (%) and standard deviation (%) over ten seeds. Teamwork average F1-macro denotes the mean across five OTAS dimensions: Communication (Comm), Coordination (Coord), Cooperation (Coop), Leadership (Lead), and Situational Awareness (Sit. Aw). Bold indicates the best overall performance.

![](images/0ee40ec13bf62e5f6419b5a0dc58e5572a03ba82301eb8f8b599397f7f336f5d.jpg)  
Figure 2: Average number of iterations required to find an appropriate counterfactual using Alg. 2.

## 4.3 Baselines

Multiple computational approaches have been proposed to model team behaviors, difering in their ability to capture temporal and relational dependencies in general domains [38] and high-stakes domains [15]. We organized competitors in terms of the amount of information they are capable of handling (see Modeling Dimension in Tab. 2). Static Neural Networks (SNN) model individual features only, and we include them as baselines using minimal information. Temporal Neural Networks (TNN) capture temporal evolution [3, 8, 21, 35, 64]. We considered both recurrent (LSTM [23] and

GRU [13]) and attention-based (MHA [57]) strategies. Relational Neural Networks (ReNN) focus on interactions, modelled via graph neural network architectures [33, 41, 55, 64]. We considered standard GNN backbones, namely GCN [28], GAT [59], and GIN [63]. Finally, Tempo-Relational Neural Networks (TReNN) [38] combine both temporal and relational aspects, and represent the most expressive class of models. We distinguish three alternative strategies to integrate temporal and relational dependencies: (i) space-thentime, where relational representations are first computed at each timestep and then processed temporally; (ii) time-then-space, where temporal encoding is first applied to node features and subsequently propagated through the graph; and (iii) joint temporal-relational models, which jointly capture graph structure and temporal evolution through dynamic graph neural network architectures. In particular, for space-then-time and time-then-space TReNN we varied the spatial modeling paradigm (i.e., GCN, GAT, GIN) and used MHA for temporal modeling, as it outperformed the other strategies in this domain. While in joint tempo-relational models we include TGN, DySAT, and EvolveGCN, which difer in how temporal dynamics are incorporated, ranging from event-based memory mechanisms to temporal attention and evolving graph convolution operators. Including these variants enables a systematic comparison across diferent integration strategies, as summarized in Table 2.

![](images/dc359698af6e35ed0ed73017e8fa1f8c92dea38e38b3e6fdb0659c2387926cfe.jpg)  
Figure 3: Average distance between the original and counterfactual paralinguistic features of the targeted team member, using Alg. 2.

## 5 Results

In this section, we first introduce the research questions, followed by the evaluation protocol, implementation details, and experimental results. Our work aims to address the following research questions:

Q1: Does TE-ReNN outperform existing paradigms in modeling small teams’ behavior in a low-data regime?

Q2: Does TE-ReNN support counterfactual algorithms capable of generalizing across topology and features while being interpretable and actionable to optimize teamwork?

## 5.1 Predictive performance

We evaluated the predictive performance of TE-ReNN by training it to predict the average annotators’ scores for each of the five dimensions of the OTAS behavioral rating schema, and comparing it to the baselines presented in Section 4.3. We formulated the problem as a multi-class classification task. Models optimize a combination of a cross-entropy and an ordinal loss function, which explicitly accounts for the ordering of the Likert ratings by penalizing predictions proportionally to their distance from the ground-truth class. To mitigate class imbalance, the loss was weighted according to class frequencies, assigning larger penalties to errors on underrepresented classes. The same training objective was adopted for all baselines, and a separate model was trained for each OTAS dimension. We employed a leave-one-team-out evaluation protocol, training on � − 1 procedures and testing on the single left-out one, and iterating over the left-out procedure. We additionally reserved one training procedure for validation. Table 2 reports the mean and standard deviation of average F1-score for each OTAS dimension, as well as their average, which serves as an overall measure of teamwork. Results show a substantial advantage of TE-ReNN over all baselines across all OTAS dimensions, regardless of the chosen GNN backbone. As expected, the best performance is obtained when TE-ReNN is paired with the most expressive backbone (GIN), although diferences across backbones remain limited. In conclusion, TE-ReNN achieves an improvement of approximately 7% over the strongest baseline (MHA+GAT). The magnitude of this gain, together with its weak dependence on the backbone architecture, highlights the robustness and efectiveness of TE-ReNN in the low-data regime. This is particularly evident given that diferences among baselines are comparatively modest, even when they leverage varying amounts of information (e.g., TReNN vs. TNN or ReNN).

## 5.2 Counterfactual assessment

Understanding how team dynamics can be improved is crucial in high-stakes collaborative settings. One approach is to generate counterfactuals as feedback for the team. The flexibility of TE-ReNN makes it particularly suitable for this task by capturing both temporal and relational information: inter-snapshot edges encode temporal dependencies, while intra-snapshot edges capture interactions among team members. To improve interpretability, we select a subset of high-level features that directly map to observable behaviors (Sec. 4.2). These include paralinguistic features, whose mapping to semantically meaningful behavioral classes (Table 1) enables actionable behavioral suggestions (e.g., remaining calmer or more dominant), and topological features, whose counterfactuals correspond to changes in team interactions. The counterfactual algorithm proposed in Sec. 3.2 was evaluated by measuring the iterations required to identify a desired behavioral pattern absent from the original graph. The algorithm searches for the closest exemplar in the embedding space and derives its factual explanation while accounting for team roles and temporal context. We grouped counterfactuals by desired score improvement, $\Delta = f ( G ^ { \prime } ) - f ( G )$ , where � is the predictive model, � the original graph, and $G ^ { \prime }$ the counterfactual exemplar By construction, $\Delta > 0 ,$ and valid counterfactuals are required to achieve a predicted score of 6 or 7 (Alg. 2).

Counterfactuals were analyzed separately according to the features they modify. For topological changes, we compared communication links between members with the same role in the original and counterfactual graphs. Fig. 2 reports the average iterations required to achieve the desired improvement as a function of Δ. Paralinguistic counterfactuals are particularly efective for Leadership and Cooperation, likely because changes in the leader’s behavior strongly influence the team, whereas topological counterfactuals perform better for Situational Awareness and Communication, and to a lesser extent Coordination. Unlike topological counterfactuals, which consist of adding or removing a communication edge, paralinguistic counterfactuals require transitions between behavioral classes (Table 1). Fig. 3 reports the behavioral class distance between original and counterfactual behaviors. Communication, Coordination, and Cooperation require only limited behavioral changes, even for large score improvements, whereas Leadership and Situa tional Awareness generally require more substantial modifications, particularly from poor initial performance.

![](images/a302449b9c7e4afa1628c7d1b7a74cd76582a6c61109bb73ded92dd4df225ead.jpg)  
Figure 4: Examples of counterfactuals generated from the dataset. Panels 1) and 2) show topological counterfactuals: in 1), the leader’s distraction leads to poor communication, suggesting a more prompt response; in 2), irrelevant conversation reduces situational awareness, suggesting greater attention and fewer of-topic discussions. Panels 3) and 4) show paralinguistic counterfactuals based on the behavioral classes in Tab. 1: in 3), the leader is disengaged from the nurse’s request, reducing team communication, and the counterfactual suggests a calm-cooperative behavior; in 4), the leader exhibits dominant-aggressive behavior toward the nurse, reducing cooperation, and the counterfactual recommends a calm behavioral style.

In addition, we conducted a human evaluation with two independent annotators to compare our counterfactual algorithm with the state-of-the-art method CoDy [50], the strongest competing approach for TReNN. The generated counterfactuals were evaluated in terms of Realism (Plausibility), Punctuality (Timing), Helpfulness (expected performance impact), and Minimality (change size). All dimensions were rated on a 5-point scale (0 = lowest, 4 = highest). Our example-based approach achieved higher scores than CoDy in realism (2.3 vs. 1.1), punctuality (2.1 vs. 1.2), and helpfulness (2.2 vs. 0.9), while obtaining a slightly lower score in minimality (2.0 vs. 2.3), as CoDy always modifies a single edge. Overall, these results indicate that TE-ReNN produces counterfactual suggestions that are perceived as more realistic, timely, and helpful than the current state-of-the-art, with only a minor trade-of in minimality.

Qualitative examples. The examples in Fig. 4 illustrate the actionable nature of the proposed counterfactual framework across both relational and paralinguistic dimensions. In all four cases, the observed interaction sequence is associated with a low predicted teamwork score, and the counterfactual identifies a minimal, interpretable modification leading to a higher predicted score. In the first case, characterized by poor Communication, the method suggests adding missing communicative links, indicating that the leader should respond more promptly to a nurse’s request. In the second case, associated with low Situational Awareness, the counterfactual recommends removing unnecessary interactions to reduce of-topic communication. The third and fourth examples demonstrate the framework’s ability to operate on paralinguistic behavior. For poor Communication, it recommends shifting toward a more responsive communication style; for poor Cooperation, it suggests reducing an aggressive vocal attitude, moving from a dominant-aggressive to a calmer leadership profile. Overall, these counterfactuals provide concrete, human-interpretable guidance on how team behavior should change, either by modifying interaction patterns or communication style, highlighting the practical value of the proposed approach in high-stakes collaborative settings.

## 6 Conclusions

In this work, we introduced TE-ReNN, a Time-Expanded Relational Neural Network for multimodal modeling of small surgical teams in low-data settings. By embedding temporal evolution directly into the graph structure, the proposed approach jointly captures interpersonal interactions and their dynamics over time, leading to consistent improvements over strong static, temporal, relational, and tempo-relational baselines across all OTAS teamwork dimensions. Building on this representation, we also proposed an exemplar-based counterfactual approach for actionable reasoning. Unlike post-hoc explanations that only justify model predictions, our method identifies minimal and interpretable behavioral changes grounded in real observations, providing actionable suggestions at both relational and paralinguistic levels. The qualitative examples further show that these counterfactuals can be translated into concrete recommendations to improve teamwork dimensions in highstakes team settings. Overall, our results demonstrate that Time-Expanded multimodal graph representations constitute a promising foundation for actionable AI in collaborative environments.

## Acknowledgments

Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them. Grant Agreement no. 101120763 - TANGO. GV and VMDL acknowledge the support of the MUR PNRR project FAIR - Future AI Research (PE00000013) funded by the NextGenerationEU. AL was supported by the Research Council ofNorway through its Centre ofExcellence Integreat - The Norwegian Centre for knowledge-driven machine learning, project number 332645. We acknowledge Rafaella Sabrina Fellone and Vincenza Moncelli for supporting the annotation process.

## References

[1] Ahmed Amer, Chirag Bhuvaneshwara, Gowtham K. Addluri, Mohammed M. Shaik, Vedant Bonde, and Philipp Müller. 2023. Backchannel Detection and Agreement Estimation from Video with Transformer Networks. In 2023 International Joint Conference on Neural Networks (IJCNN). 1–8.

[2] Robert W Andrews, J Mason Lilly, Divya Srivastava, and Karen M Feigh. 2023. The role of shared mental models in human-AI teams: a theoretical review. Theoretical Issues in Ergonomics Science 24, 2 (2023), 129–175.

[3] Umut Avci and Oya Aran. 2016. Predicting the performance in decision-making tasks: From individual cues to group interaction. IEEE Transactions on Multimedia 18, 4 (2016), 643–658.

[4] Steve Azzolin, Antonio Longa, Stefano Teso, and Andrea Passerini. 2025. Recon sidering faithfulness in regular, self-explainable and domain invariant GNNs. In International Conference on Learning Representations, Vol. 2025. 48644–48677.

[5] Chongyang Bai, Maksim Bolonkin, Viney Regunath, and VS Subrahmanian. 2023. DIPS: a dyadic impression prediction system for group interaction videos. ACM Transactions on Multimedia Computing, Communications and Applications 19, 1s (2023), 1–24.

[6] Cigdem Beyan, Nicolò Carissimi, Francesca Capozzi, Sebastiano Vascon, Matteo Bustreo, Antonio Pierro, Cristina Becchio, and Vittorio Murino. 2016. Detecting emergent leader in a meeting environment using nonverbal visual features only. In Proceedings of the 18th ACM international conference on multimodal interaction. 317–324.

[7] Cigdem Beyan, Vasiliki-Maria Katsageorgiou, and Vittorio Murino. 2017. Moving as a leader: Detecting emergent leadership in small groups using body pose. In Proceedings ofthe 25th ACM international conference on Multimedia. 1425–1433.

[8] Cigdem Beyan, Vasiliki-Maria Katsageorgiou, and Vittorio Murino. 2019. A Sequential Data Analysis Approach to Detect Emergent Leaders in Small Groups. IEEE Transactions on Multimedia 21, 8 (2019), 2107–2116.

[9] Indrani Bhattacharya, Michael Foley, Christine Ku, Ni Zhang, Tongtao Zhang, Cameron Mine, Manling Li, Heng Ji, Christoph Riedl, Brooke Foucault Welles, et al. 2019. The unobtrusive group interaction (UGI) corpus. In Proceedings of the 10th ACM Multimedia Systems Conference. 249–254.

[10] McKenzie Braley and Gabriel Murray. 2018. The group afect and performance (GAP) corpus. (2018), 1–9.

[11] Hervé Bredin, Ruiqing Yin, Juan Manuel Coria, Gregory Gelly, Pavel Korshunov, Marvin Lavechin, Diego Fustes, Hadrien Titeux, Wassim Bouaziz, and Marie-Philippe Gill. 2020. Pyannote. audio: neural building blocks for speaker diarization. In ICASSP 2020-2020 IEEE International conference on acoustics, speech and signal processing (ICASSP). IEEE, 7124–7128.

[12] Dieter Brughmans, Pieter Leyman, and David Martens. 2024. NICE: an algorithm for nearest instance counterfactual explanations. Data Mining and Knowledge Discovery 38, 5 (2024), 2665–2703

[13] Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Alessandro Moschitti, Bo Pang, and Walter Daelemans (Eds.). Association for Computational Linguistics, Doha, Qatar, 1724–1734. doi:10.3115 v1/D14-1179

[14] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guil laume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised Cross-lingual Representation Learn ing at Scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 8440.

[15] Vincenzo Marco De Luca, Antonio Longa, Giovanna Varni, and Andrea Passerini. 2026. Actionable Real-Time Modeling of Surgical Team Dynamics via Time Expanded Interaction Graphs. In Proceedings ofthe 5th International Conference on Hybrid Human-Artificial Intelligence (HHAI 2026) (Frontiers in Artificial Intelligence and Applications, Vol. 423), Maryam Alimardani, Tom Lenaerts, André Meyer-Vitali, Ann Nowé, Joost Vennekens, and Shenghui Wang (Eds.). IOS Press, Amsterdam, The Netherlands, 431–440. doi:10.3233/FAIA260530

[16] Eoin Delaney, Derek Greene, and Mark T. Keane. 2021. Instance-Based Counterfactual Explanations for Time Series Classification. In Case-Based Reasoning Research and Development, Antonio A. Sánchez-Ruiz and Michael W. Floyd (Eds.). Springer International Publishing, Cham, 32–47.

[17] Malin Eiband, Hanna Schneider, Mark Bilandzic, Julian Fazekas-Con, Mareike Haug, and Heinrich Hussmann. 2018. Bringing transparency design into practice. In Proceedings ofthe 23rd international conference on intelligent user interfaces. 211–223.

[18] Florian Eyben, Klaus R Scherer, Björn W Schuller, Johan Sundberg, Elisabeth André, Carlos Busso, Laurence Y Devillers, Julien Epps, Petri Laukka, Shrikanth S Narayanan, et al. 2015. The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and afective computing. IEEE transactions on afective computing 7, 2 (2015), 190–202.

[19] Jianfei Gao and Bruno Ribeiro. 2022. On the equivalence between temporal and static equivariant graph representations. In International Conference on Machine Learning. PMLR, 7052–7076.

[20] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. 2017. Neural Message Passing for Quantum Chemistry. In Proceedings of the 34th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 70), Doina Precup and Yee Whye Teh (Eds.). PMLR, 1263– 1272. https://proceedings.mlr.press/v70/gilmer17a.html

[21] Jamie C Gorman, Terri A Dunbar, David Grimm, and Christina L Gipson. 2017. Understanding and modeling teams as dynamical systems. Frontiers in psychology 8 (2017), 1053.

[22] Jonathan L Herlocker, Joseph A Konstan, and John Riedl. 2000. Explaining collaborative filtering recommendations. In Proceedings ofthe 2000 ACM conference on Computer supported cooperative work. 241–250.

[23] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory. Neural Computation 9, 8 (1997), 1735–1780. doi:10.1162/neco.1997.9.8.1735

[24] Louise Hull, Sonal Arora, Eva Kassab, Roger Kneebone, and Nick Sevdalis. 2011. Observational teamwork assessment for surgery: content validation and tool refinement. Journal ofthe American College ofSurgeons 212, 2 (2011), 234–243.

[25] Hayley Hung and Daniel Gatica-Perez. 2010. Estimating Cohesion in Small Groups Using Audio-Visual Nonverbal Behavior. IEEE Transactions on Multimedia 12, 6 (2010), 563–575.

[26] Fredrik Johansson, Uri Shalit, and David Sontag. 2016. Learning representations for counterfactual inference. In International conference on machine learning. PMLR, 3020–3029.

[27] Michal Kawka, Tamara MH Gall, Chihua Fang, Rong Liu, and Long R Jiao. 2022. Intraoperative video analysis and machine learning models will change the future of surgical training. Intelligent Surgery 1 (2022), 13–15.

[28] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In International Conference on Learning Representations (ICLR).

[29] Spiros Kostopoulos, Dionisis Cavouras, Dimitris Glotsos, and Constantinos Loukas. 2025. Prediction of remaining surgery duration based on machine learning methods and laparoscopic annotation data. Biomedical Engineering/Biomedizinische Technik 70, 3 (2025), 229–239.

[30] Wessel Kraaij, Thomas Hain, Mike Lincoln, and Wilfried Post. 2005. The AMI meeting corpus. In Proc. International Conference on Methods and Techniques in Behavioral Research. 1–4.

[31] Uliyana Kubasova, Gabriel Murray, and McKenzie Braley. 2019. Analyzing verbal and nonverbal features for predicting group performance. arXiv preprint arXiv:1907.01369 (2019).

[32] Kyle Lam, Junhong Chen, Zeyu Wang, Fahad M Iqbal, Ara Darzi, Benny Lo, Sanjay Purkayastha, and James M Kinross. 2022. Machine learning for technical skill assessment in surgery: a systematic review. NPJ digital medicine 5, 1 (2022), 24.

[33] Jia Li, Yangchen Yu, Yin Chen, Yu Zhang, Peng Jia, Yunbo Xu, Ziqiang Li, Meng Wang, and Richang Hong. 2024. DAT: Dialogue-Aware Transformer with Modality-Group Fusion for Human Engagement Estimation. In Proceedings of the 32nd ACM International Conference on Multimedia. 11397–11403.

[34] Yun-Shao Lin and Chi-Chun Lee. 2020. Predicting performance outcome with a conversational graph convolutional network for small group interactions. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 8044–8048.

[35] Yun-Shao Lin, Yi-Ching Liu, and Chi-Chun Lee. 2023. An Interaction-processguided Framework for Small-group Performance Prediction. ACM Transactions on Multimedia Computing, Communications and Applications 19, 2 (2023), 1–25.

[36] Antonio Longa, Veronica Lachi, Gabriele Santin, Monica Bianchini, Bruno Lepri, Pietro Lio, franco scarselli, and Andrea Passerini. 2023. Graph Neural Networks for Temporal Graphs: State of the Art, Open Challenges, and Opportunities. Transactions on Machine Learning Research (2023). https://openreview.net/forum? id=pHCdMat0gI

[37] Mingjian Lu, Haolai Che, Yangxin Fan, Qu Liu, Fei Shao, Tingjian Ge, Xusheng Xiao, and Yinghui Wu. [n. d.]. Training-free Counterfactual Explanation for Temporal Graph Model Inference. In The Fourteenth International Conference on Learning Representations.

[38] Vincenzo Marco De Luca, Giovanna Varni, and Andrea Passerini. 2026. Boosting Team Modeling through Tempo–Relational Representation Learning. Cognitive Computation 18, 1 (25 May 2026), 61. doi:10.1007/s12559-026-10581-y

[39] Ana Lucic, Maartje A Ter Hoeve, Gabriele Tolomei, Maarten De Rijke, and Fabrizio Silvestri. 2022. Cf-gnnexplainer: Counterfactual explanations for graph neural networks. In International conference on artificial intelligence and statistics. PMLR, 4499–4511.

[40] Lena Maier-Hein, Swaroop S Vedula, Stefanie Speidel, Nassir Navab, Ron Kikinis, Adrian Park, Matthias Eisenmann, Hubertus Feussner, Germain Forestier, Stamatia Giannarou, et al. 2017. Surgical data science for next-generation interventions. Nature Biomedical Engineering 1, 9 (2017), 691–696.

[41] Manasi Malik and Leyla Isik. 2023. Relational visual representations underlie human social interaction recognition. Nature Communications 14, 1 (2023), 7317.

[42] Israel Mateos-Aparicio-Ruiz, Pedro Montealegre-Macias, Oscar Deniz, and Gloria Bueno. 2025. Spatio-temporal graph neural networks for human-AI collaborative decision-making. Machine Learning with Applications (2025), 100771.

[43] Alessio Micheli. 2009. Neural network for graphs: A contextual constructive approach. IEEE Transactions on Neural Networks 20, 3 (2009), 498–511.

[44] Ramaravind K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers through diverse counterfactual explanations. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (Barcelona, Spain) (FAT\* ’20). Association for Computing Machinery, New York, NY, USA, 607–617. doi:10.1145/3351095.3372850

[45] Philipp Müller, Michal Balazia, Tobias Baur, Michael Dietz, Alexander Heimerl, Anna Penzkofer, Dominik Schiller, François Brémond, Jan Alexandersson, Elisa beth André, et al. 2024. MultiMediate’24: Multi-Domain Engagement Estimation. In Proceedings ofthe 32nd ACM International Conference on Multimedia. 11377– 11382.

[46] Philipp Müller, Michael Dietz, Dominik Schiller, Dominike Thomas, Guanhua Zhang, Patrick Gebhard, Elisabeth André, and Andreas Bulling. 2021. Multimediate: Multi-modal group behaviour analysis for artificial mediation. In Proceedings ofthe 29th ACM International Conference on Multimedia. 4878–4882.

[47] Gabriel Murray and Catharine Oertel. 2018. Predicting group performance in task-based interaction. In Proceedings of the 20th ACM International Conference on Multimodal Interaction. 14–20.

[48] Marjolein C Nanninga, Yanxia Zhang, Nale Lehmann-Willenbrock, Zoltán Szlávik, and Hayley Hung. 2017. Estimating Verbal Expressions of Task and Social Cohesion in Meetings by Quantifying Paralinguistic Mimicry. In Proceedings of the 19th ACM International Conference on Multimodal Interaction. Association for Computing Machinery, 206–215.

[49] Ege Özsoy, Chantal Pellegrini, Tobias Czempiel, Felix Tristram, Kun Yuan, David Bani-Harouni, Ulrich Eck, Benjamin Busam, Matthias Keicher, and Nassir Navab. 2025. Mm-or: A large multimodal operating room dataset for semantic understanding of high-intensity surgical environments. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 19378–19389.

[50] Zhan Qu, Daniel Gomm, and Michael Färber. 2025. CoDy: Counterfactual Explainers for Dynamic Graphs. In International Conference on Machine Learning. PMLR, 50762–50785.

[51] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning. PMLR, 28492–28518.

[52] Dairazalia Sanchez-Cortes, Oya Aran, and Daniel Gatica-Perez. 2011. An audio visual corpus for emergent leader analysis. In Workshop on multimodal corpora for machine learning: taking stock and road mapping the future, ICMI-MLMI. Citeseer.

[53] Patrick Schwab, Lorenz Linhardt, Stefan Bauer, Joachim M Buhmann, and Walter Karlen. 2020. Learning counterfactual representations for estimating individ ual dose-response curves. In Proceedings of the AAAI conference on artificial

intelligence, Vol. 34. 5612–5619.

[54] Sangwon Seo, Bing Han, and Vaibhav Unhelkar. 2023. Automated task-time interventions to improve teamwork using imitation learning. arXiv preprint arXiv:2303.00413 (2023).

[55] Garima Sharma, Shreya Ghosh, Abhinav Dhall, Munawar Hayat, Jianfei Cai, and Tom Gedeon. 2023. GraphITTI: Attributed Graph-based Dominance Ranking in Social Interaction Videos. In Companion Publication ofthe 25th International Conference on Multimodal Interaction. 323–329.

[56] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. 2013. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034 (2013)

[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems, I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.), Vol. 30. Curran Associates, Inc. https://proceedings.neurips.cc/paper\_files/paper/ 2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

[58] S Swaroop Vedula and Gregory D Hager. 2017. Surgical data science: the new knowledge domain. Innovative surgical sciences 2, 3 (2017), 109–121.

[59] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2017. Graph Attention Networks. In ICLR 2018. http: //arxiv.org/abs/1710.10903

[60] Sandra Wachter, Brent Mittelstadt, and Chris Russell. 2017. Counterfactual explanations without opening the black box: Automated decisions and the GDPR. Harv. JL & Tech. 31 (2017), 841.

[61] Thomas M Ward, Pietro Mascagni, Amin Madani, Nicolas Padoy, Silvana Perretta, and Daniel A Hashimoto. 2021. Surgical data science and artificial intelligence for surgical education. Journal ofSurgical Oncology 124, 2 (2021), 221–230

[62] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu. 2021. A Comprehensive Survey on Graph Neural Networks. IEEE Transactions on Neural Networks and Learning Systems 32, 1 (Jan. 2021), 4–24. doi:10.1109/tnnls.2020.2978386

[63] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2019. How Powerful are Graph Neural Networks?. In International Conference on Learning Representations. https://openreview.net/forum?id=ryGs6iA5Km

[64] Fangkai Yang, Wenjie Yin, Tetsunari Inamura, Mårten Björkman, and Christopher Peters. 2020. Group behavior recognition using attention-and graph-based neural networks. In ECAI 2020. IOS Press, 1626–1633.