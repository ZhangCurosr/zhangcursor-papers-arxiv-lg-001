# TimeCues Studio: A Workspace for Music Annotation and Algorithm Prototyping

Sapir Caduri   
sapir.caduri@gmail.com   
Bar-Ilan University   
Ramat Gan, Israel   
Yoav Goldberg   
yoav.goldberg@cs.biu.ac.il   
Bar-Ilan University   
Ramat Gan, Israel

## Ab<sub>s</sub>t<sub>rac</sub>t

Multimedia applications require precise music annotation—labeled positions, segments, or loops—placed by hand or algorithmically. Machine-learning algorithms are scalable and efective but need annotated training data, scarce for many tasks. TimeCues Studio is an open-source workspace where algorithm-development teams annotate a music corpus, compare detection algorithms against those annotations, and prototype new ones. Unlike existing tools built for a single track at a time, TimeCues targets teams annotating whole collections, tightly integrated with algorithm development. Annota tors place several marker types—each supporting ambiguity-aware labeling—on a grid-locked timeline that visualizes many music features, including separated audio stems. The same timeline drives an algorithm-comparison engine with bundled baselines, a Python sandbox for prototyping new models, and an ambiguity-aware evaluator that honors the structured fields. The same visualization suits solo annotators on music-sync projects. TimeCues is MIT-licensed and deploys via one Docker Compose command.

## CCS Conce<sub>p</sub>ts

• Human-centered computing → User interface toolkits; • Applied computing → Sound and music computing; • Infor mation systems → Multimedia information systems.

## Ke<sub>y</sub>words

annotation tool; beat-locked annotation; human-in-the-loop; changepoint detection; ensemble consensus; open-source

## ACM Reference Format:

Sapir Caduri and Yoav Goldberg. 2026. TimeCues Studio: A Workspace for Music Annotation and Algorithm Prototyping. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10– 14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 8 pages. https: //doi.org/10.1145/3767308.3834750

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Automated music analysis is foundational to many multimedia applications and has been studied over the decades through a wide range of machine-learning models, supporting use cases that span automatic DJ mixing and track navigation to live multi-modal synchronization between music and visualization, such as stage light ing and visual efects [36, 41]. These downstream uses of music synchronization demand very precise transitions: a lighting cue or video transition that misses the boundary is immediately visible and audible to the audience. Despite years of work on audio-driven show control—from real-time beat tracking and feature-extraction pipelines for stage lighting [11, 12] to recent deep-learning attempts at end-to-end sound-to-DMX mapping [17]—automatically synchronizing visuals to music remains an open problem. Solving it requires annotations at multiple granularities: coarse segment boundaries that mark where tension builds, releases, or shifts mood, and finegrained markers that pin small sonic events—a snare hit, a transient, a vocal entry—to the minor visual accents synchronized with them.

![](images/25a72e7e7e53c0c268a9c439ce6242e788d4d79754ff77aa4889f2a3897e2331.jpg)

Progress requires labeled corpora tuned to these timing demands and to a wide range of labeling tasks, plus an evaluation environment in which predictions can be inspected against those references. Existing annotation software is built for editing individual tracks [4, 5]; even when repurposed for machine learning, it lacks corpus management, team coordination, and built-in algorithm comparison, so each algorithmic evaluation requires a pipeline of separate tools. These tools also provide no built-in support for speeding up annotation with algorithm predictions that need only light manual correction, or for injecting one’s own detection scripts. We present TimeCues Studio, an open-source web application that unifies corpus annotation, team coordination, and algorithm evaluation, analysis and comparison in a single workspace. We now describe its key principles and features.

Team collaboration. For corpus creation and management, an admin assembles the track list (left panel of Figure 1), refines the suggested BPM, and re-aligns the beat grid [32], then hands the prepared corpus to a team that annotates every track against that shared grid. A ‘Team dashboard’ shows per-annotator progress, pairwise inter-annotator agreement, and access controls.

Flexible annotation markers. To fit diferent music analysis tasks, TimeCues supports several marker types, letting users place exact segment boundaries, single time cues, overlapping time spans, and DJ-oriented loops and patterns directly on the metric grid. For exact details about the various annotation markers, see Section 3.

Ambiguity-aware annotations. Annotation can be ambiguous: a section boundary can have several valid starting points [50], while points in between are not acceptable at all. In playlist generation, a transition might start at multiple points, all valid. In interactive audio looping, a producer might cue a section from either the start of a vocal pickup or the first heavy downbeat. An algorithm should be credited for hitting any candidate and penalized only when it misses all of them. Existing tools record a single timestamp per boundary and score it against a fixed tolerance window [41, 45], or rely on frame-matching metrics [32], which fail to handle discrete multi-candidate boundaries gracefully.

![](images/fdc7ae3ff1b3acd5c4772f00f3fa7d658ba4a72f67e92fe07ee150bacd11d696.jpg)  
Fi ure 1: TimeCues Annotator Tool. Top: la er ickers, zoom, beat- rid and sna controls (76 BPM here). Left: son s sidebar. Center: audio visualization with annotation layers, a floating edit card, and an active-layer card with thumbnails. Right: A<sub>nnotate s</sub>id<sub>e</sub>b<sub>ar w</sub>ith <sub>mar</sub>k<sub>er</sub> t<sub>a</sub>b<sub>s an</sub>d <sub>mar</sub>k<sub>er</sub> li<sub>s</sub>t<sub>.</sub>

In TimeCues Studio, we introduce an extended scheme that lets annotators mark all valid candidates for a single boundary, and our evaluator credits a match against any of them. The same multicandidate logic extends to spans and cues through layers: markers grouped in one layer are treated as semantically equivalent—say, three candidate vocal-entry cues (breath, consonant attack, vowel onset), or three competing phrasings ofan instrumental fill—and the evaluator credits any prediction that hits a member. Like boundaries, each item carries a critical-versus-optional flag, so the dashboard can score the moments that matter most separately.

Algorithm development and evaluation. TimeCues provides a <sub>un</sub>ifi<sub>e</sub>d <sub>env</sub>i<sub>ronmen</sub>t f<sub>or manua</sub>l <sub>anno</sub>t<sub>a</sub>ti<sub>on, sem</sub>i<sub>au</sub>t<sub>oma</sub>t<sub>e</sub>d annotation, and algorithmic evaluation. In this single workspace, users can quickly prototype their own models via a built-in Python editor and compare them against an engine running diverse algorithmic baselines side by side on a shared timeline (Figure 2), enabling rapid diagnostic feedback during hypothesis iteration. The engine exposes an extensibility layer for user-authored Python scripts, termed Custom Detectors, which inject hypotheses onto the timeline. Beyond algorithmic evaluation, this mechanism serves as a semi-automated annotation assistant: scripts suggest candidate markers that annotators accept or reject in the annotation tool, accelerating curation.

Testing and scoring models. The evaluation dashboard (Figure 2) ships a suite of bundled detection algorithms from several families [22, 39, 53] and scores their predictions against any chosen reference. Alongside the metrics from mir\_eval [45]—the de facto Python library for music-analysis evaluation—we add an ambiguity scheme evaluator that credits matches against any candidate in a multi-candidate boundary or layer, weights optional alternatives by a tunable factor, and reports critical-only metrics.

The same engine also computes AutoGuess (Section 4), a consensus that clusters several algorithms’ predictions into highagreement candidates—semiautomated annotation suggestions that double as a tunable baseline.

Rich visualization. TimeCues is built on the premise that visualization is central to both music annotation and algorithm analysis, and so supports a wide, flexible set of visual layers. The annotation and inspection workspaces share a single visual component (the center canvas of Figure 1): stacked, synchronized feature layers over one zoom-and-scroll timeline, a beat grid that markers can snap to so boundaries line up cleanly with the beat, and per-stem rendering so entries and exits can be seen on screen instead of being found by listening over and over.

In EDM, mashup, and DJ-style annotation tasks—a domain whose tempo and spectral structure (the distribution of energy across frequency bands) have been studied extensively for genre and subgenre characterization [6]—structural boundaries are almost always band-level events—structural changes that happen within a specific frequency band (bass, mids, or treble) rather than across the spectrum as a whole. A drop is bass suddenly appearing; a buildup is treble swelling; a breakdown is bass thinning. Locating these events on a standard monochrome amplitude waveform requires repeated listening, since the waveform collapses all frequencies into a single silhouette, while a spectrogram swings the other way, packing every frequency into dense per-pixel detail that is too noisy to scan at a glance. We therefore set the frequency-colored 3-band waveform as the default visualization (top of the center canvas in Figure 1, labeled 3-Band): bass, mids, and treble are drawn in separate colors and stacked into one silhouette, so band-level events can be located by sight rather than by repeated listening. The 3-band view is a longstanding staple ofcommercial DJ software, adapted here from production tools [35, 38, 42], but has not been adopted by open-source annotation tools, which still inherit the monochrome waveform or the dense spectrogram from generic audio editors [4, 5].

Open and extensible. TimeCues Studio is MIT-licensed and built for extension. A settings dashboard exposes configuration options such as default-value control and custom taxonomy definition; a single Docker Compose command brings up the full system. The container ships a small demo corpus of three CC0-licensed tracks<sup>1</sup> so the tool can be tried end-to-end without supplying audio first.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Annotation software. The standard tools work one track at a time: Sonic Visualiser [5] and Praat [4] have no notion of a corpus, a team, or an algorithm in the loop. Web tools widen the surface but still miss musical structure—Songle [13] and Tony [28] target active listening and pitch, while audino [14], GECKO [26], and BAT [33] target tags and transcripts. The closest, LabelBuddy [43], frames annotation as verification over containerized back-ends; TimeCues takes that further with browser-authored custom detectors and a structure-aware scheme.

Human-in-the-loop, evaluation & consensus. TimeCues’s accept/reject workflow follows a long line of human-in-the-loop audio annotation systems [21, 48, 57]. mir\_eval [45] records a single timestamp per reference, yet SALAMI [50] and McFee et al. [31] both show that choice is arbitrary; and when annotators disagree, the field aggregates labels [55] by methods from consensus clustering [46] (the basis of our AutoGuess) to IRT weighting [37]. TimeCues joins these threads, coupling scheme-aware scoring with live inter-annotator agreement (IAA).

EDM & show control. EDM has drawn focused study—drops [56], peak structure [51], subgenre and tempo [6], timbre [47], and downbeats [18]—while a growing set of downstream consumers, from beattracked stage lighting [11, 12] and soundtoDMX [17] to Songle Sync [20] and ConcertCue [9], need the authoring environment TimeCues provides.

## 3 A<sub>nno</sub>t<sub>a</sub>ti<sub>on</sub> Lif<sub>ecyc</sub>l<sub>e</sub>

TimeCues lets a team annotate a whole corpus in one place. An admin signs in, prepares the data, reviewers tag the songs on their

![](images/0ae96f7395531da56b3036da890f3062e0da3421063a556744454f2ded53b595.jpg)  
Fi<sub>g</sub>ure 2: Al<sub>g</sub>orithm Ins<sub>p</sub>ect: each detector <sub>g</sub>ets its own row b<sub>enea</sub>th th<sub>e s</sub>h<sub>are</sub>d 3<sub>-</sub>b<sub>an</sub>d <sub>wave</sub>f<sub>orm</sub>

own with per-track progress visible in the corpus sidebar (left panel of Figure 1), and a Team Dashboard surfaces team-level progress and how well annotators agree.

## 3.1 Dataset Pre<sub>p</sub>aration

An admin uploads a folder of audio files and sets a beat grid for each song. The importer accepts a mixed folder of audio and any existing metadata or annotations, previewing each decision before writing to disk. The grid runs in one of three modes: Static BPM for a fixed tempo, with five estimators [3, 32] suggesting values as chips; Dynamic for tempos that drift, with an editable tempo curve; or Manual for live or stitched takes, where each beat line can be placed by hand. A metronome plays so the grid can be checked by ear. Reviewers are then invited into the Annotator Tool (Figure 1).

The same page is where the dataset is managed: storage can be inspected and freed, and the full set of annotations, audio, stems, and algorithm outputs can be exported as JSON, Audacity [1], Sonic Visualiser [5], JAMS [19], MIDI [34], or REAPER [7], and imported from JSON, Audacity, JAMS, and CSV.

## 3<sub>.</sub>2 A<sub>nno</sub>t<sub>a</sub>t<sub>or</sub> T<sub>oo</sub>l

The annotator opens a song and listens while the audio visualization scrolls across a timeline locked to the grid (center canvas ofFigure 1). Annotation is organized into layers: a layer is a group of markers that share one type and meaning. Layers can be added as the task requires, with one tab per marker type at the top of the right panel in Figure 1: single points in time (cues), regions that do not overlap (boundaries), regions that can overlap (spans), or repeated regions (DJ-style loops/patterns).

The tool is built to cut work, save time, and reduce guesswork. Markers are placed with the mouse or keyboard (pressing ? shows the full shortcut list). The annotator can zoom all stacked visualizations at once down to the sample, then drag-select a region and play it on repeat to isolate an event by ear. Labels and descriptions are editable; regions can be dragged, trimmed, or moved; markers snap to the grid to line up with the producer’s beats in DAW-made music. The floating edit card overlaid on the center canvas in Figure 1 holds the structured fields described in Section 3.2. Every change auto-saves, with stepwise undo and redo.

Diferent analysis and annotation tasks need diferent signals. The default view is the 3-band waveform, which colors bass, mids, and treble separately. To our knowledge no other open-source annotator does this, and it makes events like an EDM kick pattern or a section change easy to spot. Beside it, the annotator can stack additional signals from the standard music-analysis feature set [32, 36] (such as spectrogram, MFCC, chroma, and others), toggled from the Signals picker. Any of these signals can also be rendered for a single Demucs stem—Demucs [8] is a neural source-separation model that splits a mix into vocals, drums, bass, and other, so the annotator can, for example, view the chroma of the vocals alone or the spectrogram of just the drum stem. Demucs runs built-in and the stem is chosen from a per-stem source selector at the top of the center canvas (Source: Full mix /Vocals /Drums /Bass /Other).

Annotators do not need to start from a blank canvas. A Custom Detector (Section 4) pre-fills a layer with algorithm suggestions, turning the task into accept/reject. AutoGuess (Section 4) takes the consensus across a chosen set of algorithms and surfaces the clusters as review cards on the timeline. Markers from either source can be copied into the manual layer, where they become part of the gold annotation. This speeds up labeling while keeping the manual layer a fully human-verified gold reference: because every algorithm carries some error or tolerance window, each copied marker is still vetted and adjusted by the annotator rather than trusted as-is. Each layer has a status (Not started/In progress/Reviewed) and a per-marker stopwatch, so time spent is recorded alongside the label.

Extended-Annotation. Each marker carries two fields beyond its label, both used by the evaluator (Section 4). Multi-candidate captures events with several valid timestamps—a sung phrase may legitimately start at the breath, the consonant, or the vowel: cues and boundaries record every valid moment, and a prediction counts as a hit if it matches any of them. Region markers (spans, loops, and patterns) cover a whole region, not just a moment—an entire alternative loop or pattern can be equally valid—so they express this equivalence by grouping candidates in a layer rather than as timestamps on one marker. Criticality (critical/optional) flags events whose timing tolerance difers—a drop must land exactly, a side fill can drift. Every marker is critical by default; the annotator clears the star on a thumbnail in the active layer card (bottom of the center canvas in Figure 1) to downgrade it to optional, and the metrics in Section 4 score the critical ones separately.

## 4 Model Develo<sub>p</sub>ment Lifec<sub>y</sub>cle

Once a corpus has been annotated, the same workspace becomes a place to develop and compare detection models against those annotations—same audio, grid, feature cache, and canvas; only the dashboard changes. A researcher integrates a new model as a Custom Detector and applies it to some or all tracks.

Algorithm Inspect (Figure 2) runs a set of bundled baselines that span the major families of music structure analysis (MSA): early novelty-curve segmentation [10], feature-based MSAF pipelines [29, 30, 39, 40, 54] and tensor decompositions [27], change-point detection [52, 53], and deep models on demixed audio [15, 22]. It puts a representative of each on one canvas. The Custom Detector hook (Section 4) welcomes new entrants.

Detection is not limited to boundaries. Beat and onset cues come from librosa and madmom [3, 32], and an optional experimentalmodels profile adds opt-in detectors for the remaining marker types, each a separate sidecar gated behind a per-family setting: downbeat and meter tracking (BeatNet [16]), Krumhansl–Schmuckler key estimation [24], chord recognition, and polyphonic note transcription (Basic Pitch [2]) for cues; voice-activity (Silero-VAD [49])

and singing-voice detection (JDC-Net [25]), AudioSet event tagging (PANNs [23]), and percussive activity for spans; loop candidates from chroma autocorrelation; and lyric transcription (Whisper [44]).

The comparison runs across two dashboards. The single-song view overlays each enabled detector as its own row beneath a shared timeline alongside the chosen reference (manual or AutoGuess), scored side by side by mir\_eval and the ambiguity scheme evaluator (Section 4). The multi-song Dataset Evaluation view (bottom tab) runs any subset across the corpus and aggregates per-song scores into a ranked table. Boundary detection is the default scope; cues, spans, loops, and patterns are available under an optional flag, and any marker type also remains reachable through Custom Detectors.

Scheme-Aware Scoring. Beyond mir\_eval, a scheme-aware evaluator adds matching rules tuned to multi-candidate and criticality (Section 3.2), parameterized by tolerance � and optional weight �<sub>o</sub> ∈ [0, 1] (default 0.5). Multi-candidate: a prediction within � of any reference candidate counts as a hit. Criticality: each reference � has $w ( g ) = 1$ if critical, $w _ { 0 }$ if optional; recall and Mean Nearest-Boundary Distance are weighted by $w ( g )$ , and Critical Section Recall reports the fraction of critical references hit within �.

Custom Detectors & AutoGuess. The Playground lets researchers prototype new detection ideas in a browser-based editor. A Custom Detector is a short Python script that can emit any marker type and plays two roles at once: an algorithm candidate ranked alongside the bundled baselines, and an annotation assistant whose predictions surface in the Annotator Tool for accept/reject.

AutoGuess folds a chosen subset of algorithms into one by clustering predictions within a window, keeping clusters above an agreement threshold, and surfacing each as a review card on the timeline badged with how many algorithms agreed; as a tunable baseline, sweeping window, threshold, �, and centroid method ranks consensus configurations by F1 Score.

System. TimeCues ships as a multi-arch Docker image (amd64/ arm64) and brings up the full system with a single docker compose up; the design choices behind the multi-service split, multi-arch build, detector sandbox, and shared feature cache are documented in the project repository.

Intended Audience & Applications. TimeCues serves three audiences. Music-analysis researchers use Algorithm Inspect, the evaluator, and the Custom Detector hook to compare structure detectors against multi-annotator references. Annotation teams use the Team Dashboard, the multi-candidate scheme, and live agreement tracking to coordinate work that would otherwise sit in ofline scripts. Show engineers (lighting, video sync, DJ tools, mashups) use the critical/optional and multi-candidate fields to record the timing tolerance each event needs—tight on a drop, loose on a side fill.

Reusability & Configuration. We have described only the core of the tool here, for space. More features are available and can be configured through the settings page. A user manual and guided walkthrough are bundled in-app. Project page: sapirca.github.io/ timecues-studio; source: github.com/sapirca/timecues-studio; setup and usage docs are in ./INSTALL.md and ./docs/USER\_GUIDE.md.

## R<sub>e</sub>f<sub>erences</sub>

[1] Audacity Team. [n. d.]. Audacity: Free, Open Source, Cross-Platform Audio Software. https://www.audacityteam.org. Label track import/export format used for time-stamped annotations..

[2] Rachel M. Bittner, Juan José Bosch, David Rubinstein, Gabriel Meseguer-Brocal, and Sebastian Ewert. 2022. A Lightweight Instrument-Agnostic Model for Poly phonic Note Transcription and Multipitch Estimation. In Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Singapore, 781–785. doi:10.1109/ICASSP43922.2022.9746549

[3] Sebastian Böck, Filip Korzeniowski, Jan Schlüter, Florian Krebs, and Gerhard Widmer. 2016. madmom: A New Python Audio and Music Signal Processing Library. In Proceedings ofthe 24th ACM International Conference on Multimedia. Association for Computing Machinery, Amsterdam, The Netherlands, 1174–1178. doi:10.1145/2964284.2973795

[4] Paul Boersma and Vincent Van Heuven. 2001. Speak and unSpeak with PRAAT. Glot International 5, 9/10 (2001), 341–347.

[5] Chris Cannam, Christian Landone, and Mark Sandler. 2010. Sonic Visualiser: an open source application for viewing, analysing, and annotating music audio files. In Proceedings ofthe 18th ACM International Conference on Multimedia (MM ’10). Association for Computing Machinery, New York, NY, USA, 1467–1468. doi:10.1145/1873951.1874248

[6] Antonio Caparrini, Javier Arroyo, Laura Pérez-Molina, and Jaime Sánchez Hernández. 2020. Automatic subgenre classification in an electronic dance music taxonomy. Journal ofNew Music Research 49, 3 (2020), 269–284. doi:10.1080/ 09298215.2020.1761399

[7] Cockos Inc. [n. d.]. REAPER: Digital Audio Workstation. https://www.reaper.fm. Project files used to round-trip markers and regions..

[8] Alexandre Défossez, Nicolas Usunier, Léon Bottou, and Francis Bach. 2019. Music Source Separation in the Waveform Domain. arXiv:1911.13254 [cs.SD] https: //arxiv.org/abs/1911.13254

[9] Eran Egozy and Ian Clester. 2022. Computer-Assisted Measure Detection in a Music Score-Following Application. In Proceedings of the 4th International Workshop on Reading Music Systems. Online, 33–36.

[10] Jonathan Foote. 2000. Automatic Audio Segmentation Using a Measure of Audio Novelty. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), Vol. 1. IEEE, New York, NY, USA, 452–455. doi:10.1109/ICME.2000. 869637

[11] Ren Gang, Gregory Bocko, Justin Lundberg, Stephen Roessner, Dave Headlam, and Mark F. Bocko. 2011. A Real-Time Signal Processing Framework of Musical Expressive Feature Extraction Using Matlab. In Proceedings ofthe 12th International Society for Music Information Retrieval Conference (ISMIR). Internationa Society for Music Information Retrieval, Miami, FL, USA, 115–120.

[12] Masataka Goto. 2001. An audio-based real-time beat tracking system for music with or without drum-sounds. Journal ofNew Music Research 30, 2 (2001), 159– 171.

[13] Masataka Goto, Kazuyoshi Yoshii, Hiromasa Fujihara, Matthias Mauch, and Tomoyasu Nakano. 2011. Songle: A Web Service for Active Music Listening Improved by User Contributions. In Proceedings ofthe 12th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Miami, FL, USA, 311–316.

[14] Manraj Singh Grover, Pakhi Bamdev, Ratin Kumar Brala, Yaman Kumar, Mika Hama, and Rajiv Ratn Shah. 2020. audino: A Modern Annotation Tool for Audio and Speech. arXiv:2006.05236 [cs.SD] https://arxiv.org/abs/2006.05236

[15] Chunbo Hao, Ruibin Yuan, Jixun Yao, Qixin Deng, Xinyi Bai, Yanbo Wang, Wei Xue, and Lei Xie. 2026. SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision. arXiv:2510.02797 [eess.AS] https://arxiv.org/abs/ 2510.02797

[16] Mojtaba Heydari, Frank Cwitkowitz, and Zhiyao Duan. 2021. BeatNet: CRNN and Particle Filtering for Online Joint Beat, Downbeat and Meter Tracking. In Proceedings of the 22nd International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Online, 270–277. https://archives.ismir.net/ismir2021/paper/000033.pdf

[17] Manuel Hils. 2023. Deep Learning for Sound-to-Light Automation in Stage Lighting Applications. Master’s thesis. Technical University of Munich (TUM), Munich, Germany. https://collab.dvb.bayern/spaces/TUMldv/pages/191204145/Deep+ Learning+for+Sound-to-Light+Automation+in+Stage+Lighting+Applications

[18] Jason A. Hockman, Matthew E. P. Davies, and Ichiro Fujinaga. 2012. One in the Jungle: Downbeat Detection in Hardcore, Jungle, and Drum and Bass. In Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR). FEUP Edições, Porto, Portugal, 169–174.

[19] Eric J. Humphrey, Justin Salamon, Oriol Nieto, Jon Forsyth, Rachel M. Bittner, and Juan Pablo Bello. 2014. JAMS: A JSON Annotated Music Specification for Reproducible MIR Research. In Proceedings ofthe 15th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Taipei, Taiwan, 591–596. https://archives.ismir.net ismir2014/paper/000355.pdf

[20] Jun Kato, Masa Ogata, Takahiro Inoue, and Masataka Goto. 2018. Songle Sync: A Large-Scale Web-based Platform for Controlling Various Devices in Synchronization with Music. In Proceedings ofthe 26th ACM International Conference on Multimedia. Association for Computing Machinery, Seoul, Republic of Korea, 1697–1705. doi:10.1145/3240508.3240619

[21] Bongjun Kim and Bryan Pardo. 2018. A Human-in-the-Loop System for Sound Event Detection and Annotation. ACM Transactions on Interactive Intelligent Systems 8, 2 (2018), 13:1–13:23. doi:10.1145/3214366

[22] Taejun Kim and Juhan Nam. 2023. All-in-One Metrical and Functional Structure Analysis with Neighborhood Attentions on Demixed Audio. In Proceedings of the IEEE Workshop on Applications ofSignal Processing to Audio and Acoustics (WASPAA). IEEE, New Paltz, NY, USA, 1–5. doi:10.1109/WASPAA58266.2023. 10248148

[23] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. 2020. PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020), 2880–2894. doi:10.1109/TASLP.2020.3030497

[24] Carol L. Krumhansl. 1990. Cognitive Foundations of Musical Pitch. Oxford University Press, New York, NY, USA. doi:10.1093/acprof:oso/9780195148367.001.0001

[25] Sangeun Kum and Juhan Nam. 2019. Joint Detection and Classification of Singing Voice Melody Using Convolutional Recurrent Neural Networks. Applied Sciences 9, 7 (2019), 1324. doi:10.3390/app9071324

[26] Golan Levy, Raquel Sitman, Ido Amir, Eduard Golshtein, Ran Mochary, Eilon Reshef, Roi Reichart, and Omri Allouche. 2019. GECKO — A Tool for Efective Annotation ofHuman Conversations. In Proceedings ofthe 20th Annual Conference of the International Speech Communication Association (Interspeech). ISCA, Graz, Austria, 3677–3678.

[27] Axel Marmoret, Jérémy E. Cohen, Frédéric Bimbot, and Nancy Bertin. 2020. Uncovering audio patterns in music with Nonnegative Tucker Decomposition for structural segmentation. In Proceedings ofthe 21st International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Montréal, Canada, 788–794. https://archives.ismir.net/ ismir2020/paper/000078.pd

[28] Matthias Mauch, Chris Cannam, Rachel M. Bittner, George Fazekas, Justin Salamon, Jiajie Dai, Juan Pablo Bello, and Simon Dixon. 2015. Computer-Aided Melody Note Transcription Using the Tony Software: Accuracy and Eficiency. In Proceedings ofthe First International Conference on Technologies for Music Notation and Representation (TENOR). Institut de Recherche en Musicologie, Paris, France, 23–30.

[29] Brian McFee and Daniel P. W. Ellis. 2014. Analyzing Song Structure with Spectral Clustering. In Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Taipei, Taiwan, 405–410.

[30] Brian McFee and Daniel P. W. Ellis. 2014. Learning to Segment Songs with Ordinal Linear Discriminant Analysis. In Proceedings ofthe IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Florence, Italy, 5197–5201. doi:10.1109/ICASSP.2014.6854594

[31] Brian McFee, Oriol Nieto, Morwaread M. Farbood, and Juan Pablo Bello. 2017. Evaluating Hierarchical Structure in Music Annotations. Frontiers in Psychology 8 (2017), 1337. doi:10.3389/fpsyg.2017.01337

[32] Brian McFee, Colin Rafel, Dawen Liang, Daniel P. W. Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. 2015. librosa: Audio and Music Signal Analysis in Python. In Proceedings ofthe 14th Python in Science Conference (SciPy). scipy.org, Austin, TX, USA, 18–24. doi:10.25080/Majora-7b98e3ed-003

[33] Blai Meléndez-Catalán, Emilio Molina, and Emilia Gómez Gutiérrez. 2017. BAT: An open-source, web-based audio events annotation tool. In Proceedings ofthe 3rd International Web Audio Conference (WAC). London, United Kingdom.

[34] MIDI Manufacturers Association. 1996. The Complete MIDI 1.0 Detailed Specifi cation. https://www.midi.org.

[35] Mixxx Development Team. [n. d.]. Mixxx: Multi-Band Frequency-Coloured Waveform Renderers. https://mixxx.org.

[36] Meinard Müller. 2015. Fundamentals ofMusic Processing: Audio, Analysis, Algorithms, Applications (1 ed.). Springer Cham, Cham, Switzerland. doi:10.1007/978- 3-319-21945-5

[37] Tomoyasu Nakano and Masataka Goto. 2024. Using Item Response Theory to Aggregate Music Annotation Results of Multiple Annotators. In Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, San Francisco, USA, 1076– 1084.

[38] Native Instruments. [n. d.]. Traktor Pro: Frequency-Coloured Waveform Display. https://www.native-instruments.com/traktor.

[39] Oriol Nieto and Juan Pablo Bello. 2016. Systematic Exploration of Computational Music Structure Research. In Proceedings of the 17th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, New York, NY, USA, 547–553. https://archives.ismir.net/ ismir2016/paper/000043.pdf

[40] Oriol Nieto and Tristan Jehan. 2013. Convex Non-Negative Matrix Factorization for Automatic Music Structure Identification. In Proceedings ofthe 38th IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Vancouver, Canada, 236–240. doi:10.1109/ICASSP.2013.6637644

[41] Oriol Nieto, Gautham J. Mysore, Cheng-i Wang, Jordan B. L. Smith, Jan Schlüter, Thomas Grill, and Brian McFee. 2020. Audio-Based Music Structure Analysis: Current Trends, Open Challenges, and Applications. Transactions of the International Society for Music Information Retrieval 3, 1 (2020), 246–263. doi:10.5334/tismir.54

[42] Pioneer DJ Corporation. [n. d.]. Rekordbox: 3-Band Frequency-Coloured Waveform Display. https://rekordbox.com.

[43] Ioannis Prokopiou, Ioannis Sina, Agisilaos Kounelis, Pantelis Vikatos, and Themos Stafylakis. 2026. LabelBuddy: An Open Source Music and Audio Language Annotation Tagging Tool Using AI Assistance. In Proceedings ofthe 4th Workshop on NLPfor Music and Audio (NLP4MusA 2026). Association for Computational Linguistics, Rabat, Morocco, 7–12. doi:10.18653/v1/2026.nlp4musa-1.2

[44] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust Speech Recognition via Large-Scale Weak Supervision. In Proceedings ofthe 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202). PMLR, Honolulu, HI, USA, 28492–28518. https://proceedings.mlr.press/v202/radford23a.htm

[45] Colin Rafel, Brian McFee, Eric J. Humphrey, Justin Salamon, Oriol Nieto, Dawen Liang, and Daniel P. W. Ellis. 2014. mir\_eval: A Transparent Implementation of Common MIR Metrics. In Proceedings ofthe 15th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Taipei, Taiwan, 367–372. https://archives.ismir.net/ismir2014/ paper/000320.pdf

[46] Iris Yuping Ren, Hendrik Vincent Koops, Anja Volk, and Wouter Swierstra. 2017. In Search of the Consensus Among Musical Pattern Discovery Algorithms. In Proceedings ofthe 18th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Suzhou, China, 671–678. https://archives.ismir.net/ismir2017/paper/000120.pdf

[47] Bruno Rocha, Niels Bogaards, and Aline Honingh. 2013. Segmentation and Timbre Similarity in Electronic Dance Music. In Proceedings ofthe 10th Sound and Music Computing Conference (SMC). Stockholm, Sweden, 754–761. doi:10. 5281/zenodo.850377

[48] António Sá Pinto. 2025. Towards Human-in-the-Loop Onset Detection: A Transfer Learning Approach for Maracatu. In Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Daejeon, South Korea, 320–327. arXiv:2507.04858 [cs.SD] https://arxiv.org/abs/2507.04858

[49] Silero Team. 2024. Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier. https://github.com/

snakers4/silero-vad.

[50] Jordan B. L. Smith, John Ashley Burgoyne, Ichiro Fujinaga, David De Roure, and J. Stephen Downie. 2011. Design and Creation of a Large-Scale Database of Structural Annotations. In Proceedings ofthe 12th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Miami, FL, USA, 555–560. https://ismir2011.ismir.net/ papers/PS4-14.pdf

[51] Ragnhild Torvanger Solberg and Nicola Dibben. 2019. Peak Experiences with Electronic Dance Music: Subjective Experiences, Physiological Responses, and Musical Characteristics of the Break Routine. Music Perception 36, 4 (2019), 371–389. doi:10.1525/mp.2019.36.4.371

[52] Charles Truong, Laurent Oudre, and Nicolas Vayatis. 2018. ruptures: change point detection in Python. arXiv:1801.00826 [stat.CO] https://arxiv.org/abs/1801.00826

[53] Charles Truong, Laurent Oudre, and Nicolas Vayatis. 2020. Selective Review of Ofline Change Point Detection Methods. Signal Processing 167 (2020), 107299. doi:10.1016/j.sigpro.2019.107299

[54] Colin Vaz, Asterios Toutios, and Shrikanth Narayanan. 2016. Convex Hull Convolutive Non-negative Matrix Factorization for Uncovering Temporal Patterns in Multivariate Time-Series Data. In Proceedings ofthe Annual Conference ofthe International Speech Communication Association (Interspeech). ISCA, San Francisco, CA, USA, 963–967. doi:10.21437/Interspeech.2016-571

[55] Cheng-i Wang, Gautham J. Mysore, and Shlomo Dubnov. 2017. Re-Visiting the Music Segmentation Problem with Crowdsourcing. In Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Suzhou, China, 738–744. https://archives.ismir.net/ismir2017/paper/000102.pdf

[56] Karthik Yadati, Martha Larson, Cynthia C. S. Liem, and Alan Hanjalic. 2014. Detecting Drops in Electronic Dance Music: Content based approaches to a socially significant music event. In Proceedings ofthe 15th International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Taipei, Taiwan, 143–148. https://archives.ismir.net/ ismir2014/paper/000297.pdf

[57] Kazuhiko Yamamoto. 2021. Human-in-the-Loop Adaptation for Interactive Musical Beat Tracking. In Proceedings of the 22nd International Society for Music Information Retrieval Conference (ISMIR). International Society for Music Information Retrieval, Online, 794–801. https://archives.ismir.net/ismir2021/paper/ 000099.pdf

## A<sub>pp</sub>endix

## A Case Stud<sub>y</sub>: Annotatin<sub>g</sub> an EDM Cor<sub>p</sub>us

The lead author and four collaborators used TimeCues to build a 109-track EDM corpus, spanning subgenres from Afro and Organic House to Psybass, Glitch Hop, and Melodic Techno. The annotations show a consistent structural vocabulary—a median of about ten boundaries per track, dominated by drops (39%) and buildups (23%), then breakdowns (11%), intros and outros (10% each), bridges (5%), and silences (3%). The criticality field was actively used, with about 20% of boundaries flagged optional and the rest critical—the hard, beat-locked transitions that must land exactly.

EDM structure is mostly band-level, so on the frequency-colored 3-band waveform (Figure 1) these events read as shapes, and annotators placed most boundaries by sight, often faster than by ear. Boundaries were often ambiguous, heard at the last bar ofthe outgoing section or the first of the incoming one; rather than force a single timestamp, annotators recorded every defensible position with the multi-candidate field (Section 3.2). Because each annotator’s layer is stored separately yet shares the beat grid, the Team Dashboard ranks tracks by inter-annotator disagreement, so the admin can target the least-agreed tracks and consolidate each into one reviewed annotation. Agreement is the pairwise tolerance-windowed bound ary F1—greedy one-to-one matching within �, with label agreement over the matched boundaries—not a categorical Kappa or Alpha, which miss continuous multi-candidate alignment.

## B T<sub>ec</sub>hni<sub>ca</sub>l Ar<sub>c</sub>hit<sub>ec</sub>t<sub>u</sub>r<sub>e a</sub>nd D<sub>es</sub>i<sub>g</sub>n Ch<sub>o</sub>i<sub>ces</sub>

## B<sub>.</sub>1 M<sub>o</sub>d<sub>e</sub>l<sub>-ass</sub>i<sub>s</sub>t<sub>e</sub>d <sub>anno</sub>t<sub>a</sub>ti<sub>on</sub>

Integrating detection models directly lets TimeCues request predictions on demand and drop them onto the canvas as editable pre-annotations, shifting the task from creation to verification—the accept/reject loop of Section 4 (Figure 3). The verified labels accumulate into a corpus for fine-tuning, closing a human-in-the-loop cycle. This matters most when pretrained models underperform on under-represented material—a beat tracker trained on steady 4/4 pop drifts on frequent tempo changes—where a few corrected examples adapt a detector with little efort [48].

![](images/bdc20c0aa7038dff43ad3c455f1bc34f48439e7c89714c9dfd0684da5e66b2d3.jpg)  
Figure 3: AutoGuess review band: accept/reject cards.

## <sup>B</sup>.2 <sup>O</sup>ne <sub>p</sub>rocess <sub>p</sub>er concern

Backend capabilities are decoupled into separate services, each pinning its own dependencies. This matters because the underlying audio and deep-learning libraries pin conflicting versions ofNumPy,

PyTorch, and Cython, so one combined image is fragile; isolated services track their upstreams independently and restart on failure without taking the workspace down.

## B.3 Multi-arch ima<sub>g</sub>e<sub>,</sub> sin<sub>g</sub>le source

The Docker image builds for amd64 and arm64 from one Dockerfile via buildx. Researchers develop on Apple Silicon, lab servers run Intel, the cloud runs either, and one tag covers all three. Compose profiles (base, gpu, cpu, experimental-models) add opt-in services on top of the base install, so a first-time user does not download CUDA or experimental model weights they will not use.

## B<sub>.</sub>4 S<sub>an</sub>db<sub>oxe</sub>d P<sub>y</sub>th<sub>on</sub> d<sub>e</sub>t<sub>ec</sub>t<sub>ors</sub>

A Custom Detector is a short Python script the researcher edits in the Playground editor (Figure 4); hitting Run executes it and the result lands on the canvas. Scripts run server-side in an isolated child process with memory, CPU, and wall-clock limits, so an honest mistake—a runaway loop, an accidental allocation—is contained. The same script serves both interactive single-song testing and batch evaluation across the corpus.

![](images/8dd21d47f056e8c649a0761a9dce09794db40975e20a830eccccb2d188e7ae25.jpg)  
Fi<sub>g</sub>ure 4: Pla<sub>yg</sub>round: in-browser editor for Custom Detect<sub>ors, w</sub>hi<sub>c</sub>h <sub>can</sub> f<sub>ee</sub>d th<sub>e</sub> i<sub>nspec</sub>t<sub>or,</sub> th<sub>e anno</sub>t<sub>a</sub>t<sub>or, or</sub> b<sub>o</sub>th<sub>.</sub>

## B<sub>.</sub>5 P<sub>re-compu</sub>t<sub>e</sub>d f<sub>ea</sub>t<sub>ure cac</sub>h<sub>e</sub>

The feature server caches every expensive computation (mel-spectrogram, MFCC, chroma, tempogram, SSM, novelty, Demucs stems) under (song-slug,feature-name) keys with file-hash invalidation. Custom detectors and built-in baselines share the same cache, so a set of ruptures configurations and MSAF segmenters all read the same chroma frames. Adding a new configuration costs only its detection-and-scoring pass, not another round of feature extraction. Every signal ofered by the Signals picker (Figure 5) and every Demucs stem in the source selector (Figure 6) is served from this cache, so toggling a signal or switching stems is instant after the first computation.

![](images/20be364747d067a547f25d29944fedc1f46a3dbb2ec6e65245b71805ee839bab.jpg)  
Fi<sub>g</sub>ure 8: On-disk foot<sub>p</sub>rint for a 109-track cor<sub>p</sub>us: stems<sub>,</sub> anal<sub>y</sub>sis<sub>,</sub> raw MSAF out<sub>p</sub>ut<sub>,</sub> BPM<sub>,</sub> and al<sub>g</sub>orithm clusters are <sub>accoun</sub>t<sub>e</sub>d <sub>separa</sub>t<sub>e</sub>l<sub>y,</sub> <sub>an</sub>d <sub>regenera</sub>bl<sub>e</sub> <sub>cac</sub>h<sub>es</sub> <sub>are</sub> <sub>c</sub>l<sub>eare</sub>d i<sub>n</sub> <sub>one</sub> <sub>c</sub>li<sub>c</sub>k<sub>.</sub>

![](images/72c2ad580fbf4bd3c2ea529e7c9fd8c70e938b89f3ce53c625a26e43f7291d0a.jpg)

Fi<sub>g</sub>ure 5: The Signals <sub>p</sub>icker to<sub>gg</sub>les anal<sub>y</sub>sis la<sub>y</sub>ers— waveform, EQ, spectrogram, MFCC, chroma, tempogram, SSM<sub>,</sub> ener<sub>gy,</sub> bri<sub>g</sub>htness<sub>,</sub> novelt<sub>y,</sub> onsets<sub>,</sub> s<sub>p</sub>ectral flux—as <sub>a sync</sub>h<sub>ron</sub>i<sub>ze</sub>d <sub>s</sub>t<sub>ac</sub>k <sub>over</sub> th<sub>e s</sub>h<sub>are</sub>d ti<sub>me</sub>li<sub>ne.</sub>  
![](images/3222ccddfdcce0e671c562dd84b33bed56f5ba7688a48ee2f8238f410ab3b535.jpg)  
Fi<sub>g</sub>ure 6: Per-stem source selector: an<sub>y</sub> si<sub>g</sub>nal can be rendered for the full mix or a single Demucs stem (Vocals/Drums/Bass/ Other). Vocals is selected here.

![](images/825a933b5f12d1905df6a9e1249971549c6e183da4bb1f3098fa09858b826d17.jpg)  
Fi<sub>g</sub>ure 7: Li<sub>g</sub>htwei<sub>g</sub>ht si<sub>g</sub>n-in: an annotator identifies via Goo<sub>g</sub>le or a <sub>p</sub>lain username/email<sub>,</sub> and that identit<sub>y</sub> is att<sub>ac</sub>h<sub>e</sub>d t<sub>o every save</sub>d <sub>anno</sub>t<sub>a</sub>ti<sub>on so mu</sub>lti<sub>p</sub>l<sub>e anno</sub>t<sub>a</sub>t<sub>ors</sub>’ l<sub>ayers can</sub> b<sub>e compare</sub>d l<sub>a</sub>t<sub>er.</sub>

## B<sub>.</sub>6 Fil<sub>es over</sub> d<sub>a</sub>t<sub>a</sub>b<sub>ase</sub>

Annotations live as JSON files on disk, one per song per layer per annotator, at

data/annotations/<layer>/<annotator>/<slug>.json

Per-annotator subdirectories stop annotators from overwriting each other; the annotator chosen at sign-in (Figure 7)—via Google or a plain username/email—supplies the <annotator> path component. Because each category lives in its own files, a storage panel accounts for each separately and clears regenerable caches in one click (Figure 8).

## B<sub>.</sub>7 Sh<sub>are</sub>d b<sub>ea</sub>t<sub>-</sub>l<sub>oc</sub>k<sub>e</sub>d <sub>canvas</sub>

A per-song beat grid drives every view, and markers snap to it (toggleable) so a manual boundary and an algorithm prediction all line up on the same lines. The same grid serves Dataset Prep, the Annotator Tool, and Algorithm Inspect, so an alignment set once holds everywhere; grid mode and the snap toggle live on a shared canvas toolbar (Figure 9).

![](images/8c0e3dda1dc7a0f9bf8d8f13c20f343ab2655e840e0c57010e5c68ffcb480e05.jpg)  
Fi<sub>gure</sub> 9<sub>:</sub> C<sub>anvas</sub> t<sub>oo</sub>lb<sub>ar</sub> <sub>s</sub>h<sub>are</sub>d <sub>across</sub> <sub>a</sub>ll th<sub>ree</sub> <sub>wor</sub>k<sub>spaces:</sub> the Annotations and Signals la<sub>y</sub>er <sub>p</sub>ickers<sub>,</sub> zoom<sub>,</sub> the Grid <sub>mo</sub>d<sub>e</sub> <sub>se</sub>l<sub>ec</sub>t<sub>or,</sub> th<sub>e</sub> <sub>snap</sub> t<sub>ogg</sub>l<sub>e,</sub> <sub>an</sub>d M<sub>i</sub>sc <sub>op</sub>ti<sub>ons.</sub>

## B<sub>.</sub>8 St<sub>an</sub>d<sub>ar</sub>d f<sub>orma</sub>t<sub>s, no</sub> l<sub>oc</sub>k<sub>-</sub>i<sub>n</sub>

Export covers JSON, JAMS, Audacity, Sonic Visualiser, MIDI, and REAPER; import covers JSON, JAMS, Audacity, and CSV. The schemeaware evaluator and multicandidate format are specified separately from the GUI, so a team can stop using the workspace and still keep their annotations.

## B<sub>.</sub>9 D<sub>e</sub>m<sub>o w</sub>ith<sub>ou</sub>t <sub>se</sub>r<sub>ve</sub>r <sub>s</sub>t<sub>a</sub>t<sub>e</sub>

A bundled demo of three CC0 tracks with pre-baked Demucs stems ships inside the image and runs client-side via localStorage, so the tool can be tried end-to-end without uploading audio, creating an account, or touching server-side data.

## B.10 One-command de<sub>p</sub>lo<sub>y</sub>

A provided deployment pipeline builds for both architectures and ships the image.