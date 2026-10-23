# NNINTERACTIVE AGENT: GENERAL-PURPOSE AI AGENTS AS 3D IMAGE ANNOTATORS

TECHNICAL REPORT

Fabian Isensee<sup>1,2</sup> Constantin Ulrich Harsy<sup>1,3</sup> Maximilian Rokuss<sup>1,6</sup> Benjamin Hamm<sup>1,7</sup> Philipp Vollmuth<sup>1,4</sup> Klaus H. Maier-Hein<sup>1,2,3,5,6,7</sup>

<sup>1</sup>Division of Medical Image Computing (MIC), German Cancer Research Center (DKFZ), Heidelberg, Germany <sup>2</sup>National Center for Tumor Diseases (NCT) Heidelberg, a partnership between the German Cancer Research Center (DKFZ) and Heidelberg University Hospital (UKHD), Heidelberg, Germany

<sup>3</sup>Helmholtz Imaging, German Cancer Research Center (DKFZ), Heidelberg, Germany   
<sup>4</sup>Division for Computational Radiology & Clinical AI, Clinic for Neuroradiology, University Hospital Bonn, Bonn, Germany   
<sup>5</sup>Pattern Analysis and Learning Group, Department of Radiation Oncology, Heidelberg University Hospital, Heidelberg, Germany <sup>6</sup>Faculty of Mathematics and Computer Science, Heidelberg University, Germany <sup>7</sup>Medical Faculty, Heidelberg University, Germany

October 23, 2026

## ABSTRACT

Interactive 3D segmentation models such as nnInteractive turn a few spatial prompts (points, scribbles, boxes) into accurate volumetric masks, but a human still does the annotation work: finding each structure, placing prompts, checking the result and correcting it. General-purpose AI agents can now interpret images, call tools and write code. We show that such agents can take over this work without additional training: the user makes a request in plain language, and the agent works out what has to be segmented, locates the targets, prompts nnInteractive, inspects the 3D result, corrects it and responds to feedback. Asked for “the muscles used for chewing”, an agent identifies and segments the masseters and pterygoids; asked to count liver lesions, it writes its own code to find candidates and splits a merged lesion on request. Our open-source nninteractive-agent provides nnInteractive as tools over the Model Context Protocol (MCP), together with a skill that encodes how experienced users prompt it. On a benchmark spanning CT, MRI, PET and micro-CT of humans, a mouse and an ant (14 images; 51 targets from organs and tumours to muscles, foot bones, vessel trees and an insect brain), Claude Code and Codex reach a mean Dice of 0.81 and 0.84 from the request alone; when told where each target is, both reach 0.85 and prompt nearly as effectively as an oracle that places prompts using the reference segmentation. A text-prompted 3D model (VoxTell) is more accurate on most standard radiological targets but less robust: it returns an empty mask for one of them and fails on the leg muscles, the foot bones, the mouse and the ant, where the agents remain accurate. Code: https://github.com/MIC-DKFZ/nninteractive-agent

## 1 Introduction

Segmentation turns images into quantitative measurements such as organ volumes, tumour burden or cell counts. Dedicated models segment the structures they were trained on; everything else is delineated by people, increasingly with the help of promptable models. nnInteractive [4], for example, predicts a full 3D mask from points, scribbles, boxes or lassos placed in single slices and was trained on more than 120 volumetric datasets spanning CT, MRI, PET and 3D microscopy. Such a model makes every prompt count, but the annotation process itself remains with a human: deciding what the target is and where it lies, placing prompts, judging the 3D result, correcting it, and revising it when someone asks for changes.

![](images/402ba6f801c12943ab14857c793b42539dd8b6c61a4dc689a9449a0be3612ea4.jpg)  
Masseters and medial and lateral pterygoids on both sides. It left out the temporalis, which “lies almost entirely above the scanned region”.

“Now segment the muscles used for turning the head.”

![](images/f6116480278c2df737f80efb08bbc52ee5b5660d55424f24df0c35c663e5d0b0.jpg)  
Both sternocleidomastoids. It named the deeper rotators it left out, because their borders are too faint on this scan, and ofered to try.

“Segment every rib as a separate object and name them by side and number.”

![](images/94d7e8bbbfb7a78e24453c15fe6a66ae35744dd4363eac51b8dcc65de82c97ab.jpg)  
24 named ribs. It found and fixed its own mistakes: collarbones taken for rib 1, a missed rib, ribs stopping short of the spine.

![](images/c0806ac941dec93d05cc9e7a431025b7c06646d9d650fd4c3e3d1dc97bc2de42.jpg)  
It first segmented the battery as the flashlight and said it was unsure. Told “the flashlight is somewhere else, missing its battery”, it found the empty flashlight.

Figure 1: From a request to 3D masks. Four requests to Claude Code (Opus 5.5 medium effort) operating nnInteractive in the web demo: the request, the result in 3D and in one slice (zoomed in for the suitcase), and what the agent did. The second request followed the first in the same conversation. Images: TotalSegmentator MRI [1], RibFrac [2, 3], a CT scan of a packed suitcase (image provided by Smiths Detection).

Text-prompted medical segmentation models take the human out of this loop [5, 6, 7, 8]. They map the name or description of a structure directly to a mask and can be very accurate, but their accuracy depends on how close a request is to their training data.

General-purpose AI agents such as Claude Code and Codex combine a multimodal large language model with tool use and code execution. They interpret images, have broad anatomical knowledge, follow multi-step instructions, write code and can check their own work. They cannot, however, produce a voxel-accurate 3D mask by themselves. A promptable model like nnInteractive can, but it cannot decide what to segment. The two are complementary.

Combining them, we let an agent take over the annotation process. The user makes a request in plain language; the agent works out what has to be segmented, finds the targets, prompts nnInteractive, inspects the 3D result, corrects it and responds to feedback, while nnInteractive turns each prompt into a 3D mask (Fig. 1). Asked for the muscles used for chewing, the agent works out which muscles these are, segments them and explains why it left one out; asked to count liver lesions, it writes code to find candidates, flags a lesion that may be two, and splits it when the user asks (Fig. 5). No component is trained for this purpose. nninteractive-agent provides nnInteractive as a set of tools over the Model Context Protocol (MCP) [9], together with a skill, a document that describes how experienced users prompt nnInteractive.

We evaluate nninteractive-agent with Claude Code and Codex on a benchmark of 14 images and compare the agents with simulated prompts and with a text-prompted model. Agents that call segmentation models exist, including training-free agents for 3D images (Section 2). What we add is an extension that lets an unmodified general-purpose agent carry out the interactive 3D annotation process itself (finding, prompting, checking, correcting) across modalities, species and structures, and an evaluation of its segmentation accuracy. The tool, the skill, a web demo and the benchmark are open source.

## 2 Related work

Promptable segmentation. SAM [10] established segmentation from points and boxes in 2D. There have also been some notable evaluations and adaptations in the medical domain [11, 12, 13]. In 3D, SAM-Med3D [14] and VISTA3D [15] accept point prompts, SegVol [5] points and boxes, and MedSAM2 [16] propagates a box drawn on one slice through the volume. nnInteractive [4] accepts points, scribbles, boxes and lassos placed in single slices and predicts a full 3D mask. In their interactive modes these models are class-agnostic: what is segmented is decided by whoever places the prompts, usually a human. This makes them a natural tool for an agent. We use nnInteractive for three reasons: it supports the widest range of prompt types among these models, including scribbles, which describe elongated or irregular structures that a single point or box cannot; it clearly outperformed existing promptable 3D methods in extensive benchmarking [4]; and it is open-set, segmenting whatever the prompts describe, and was trained across CT, MRI, PET and 3D microscopy.

Text-prompted segmentation. SegVol [5], SAT [6] and VoxTell [8] map the name or description of a structure directly to a 3D mask, and SAM 3 [17] does so for concepts in natural images and videos. These models need no operator, and some are designed to generalise beyond their training data: VoxTell, our text-prompted baseline, reports strong zero-shot results on unseen datasets and on classes related to those it was trained on. Their accuracy on medical images still depends on how close a request is to the training distribution, whereas that of an agent operating a class-agnostic model depends on whether it recognises the target in the image and prompts the model well.

Agents that prompt segmentation models. Closest to our work are agents that operate a promptable model. SegAgent [18], MedSAM-Agent [19] and IBISAgent [20] train multimodal models on annotation trajectories, partly with reinforcement learning, to issue clicks and boxes to SAM-type models on 2D images. Our agent, in contrast, is not trained at all: it learns how to prompt from the skill and keeps its general abilities. The SAM 3 Agent [17] and the MedSAM-3 Agent [21] use off-the-shelf multimodal models, but they prompt a concept segmenter with text and inspect the returned masks on 2D images, leaving the spatial decisions to the segmenter. Seg-Agent [22], also training-free, does make the spatial decisions: it places visual prompts for SAM and refines them after inspecting candidate masks drawn on 2D natural images. In MedOpenClaw [23], frontier agents operate 3D Slicer and can call VISTA3D, but segmentation is an auxiliary step for diagnostic questions, its accuracy is not evaluated, and the authors found the agents’ multi-step segmentation workflows unreliable. In our work, interactive 3D segmentation is the task itself: the agent places spatial prompts, including scribbles, checks the result in rendered slices and 3D views, and corrects it. We evaluate this process for segmentation accuracy across imaging modalities and species.

Training-free agents and imaging software. Agents built on frontier models without additional training are increasingly used in medical imaging, also for 3D images. Erdur et al. [24] run brain MRI pipelines (registration, automatic tumour and anatomy segmentation, volumetry) through text-only tool calls, and CT-Flow [25] provides CT viewing, measurement and automatic segmentation tools through MCP servers for question answering and reporting. HealthAgentBench [26] evaluates unmodified Claude Code and Codex on healthcare tasks and finds imaging tasks, including 3D CT, the hardest, which suggests that agents need suitable tools for images. Agents can also operate imaging software directly, by writing code in napari [27] or through generic MCP servers for 3D Slicer and napari [28, 29]. Most of these systems rely on automatic segmentation models, which limits them to the structures those models know, and many are dedicated agent frameworks around their own tools or viewers. nninteractive-agent instead extends an existing general-purpose agent: an MCP server and a skill that any Claude Code or Codex session can load, without a viewer.

## 3 nninteractive-agent

## 3.1 Division of labour

The agent decides what to segment and where to prompt, and it judges the result; nnInteractive turns each prompt into a 3D mask (Fig. 2). The agent never draws a mask itself and never needs to be precise to the voxel: a scribble along the middle of a structure is enough for nnInteractive to find its boundary in 3D. Conversely, nnInteractive needs no knowledge of what a “left parotid gland” is: the agent supplies it through the location of its prompts. This split puts each capability where it is strongest, and nothing in the system is specific to an anatomy, a modality or a species.

![](images/c25a2d706cf6d091663cf4fa14637520798546bfea6730302f3b3b85c53e7af7.jpg)  
Figure 2: Overview. The agent reads the skill and works through the tools of the MCP server. The server holds the image, the objects segmented so far and an nnInteractive session. Each prompt goes through the server to nnInteractive; the agent checks the resulting 3D mask in the returned text and in rendered views, and repeats this until it accepts the object. It reports to the user once per request. Example: “Segment and count all liver lesions” on MSD Liver, case 51 [30, 31]. nnInteractive runs locally or on a shared server.

## 3.2 Tools and workspace

The MCP server keeps one workspace per agent session: the image, one nnInteractive session, the object being segmented with its prompts, and an instance map of the objects committed so far. The agent works with 19 tools in four groups.

Objects. nnInteractive segments one object at a time. The agent starts an object, adds prompts, and commits it to the instance map, where every voxel belongs to one object. A structure inside another one, such as a lesion in an organ, takes those voxels from the outer object when the agent asks for it; otherwise an overlap with a committed neighbour is treated as a leak and reported. Committed objects can be reopened for corrections.

Prompts. A point marks a tiny object or a tiny error; a box encloses the object in one slice and is allowed only as an object’s first prompt; a scribble is a line drawn in one slice. nnInteractive accepts any binary mask as a scribble; the agent specifies a polyline, which the tool rasterises 2 pixels wide. Points and scribbles can be positive or negative. Boxes and an object’s first scribble are previewed: the tool draws them on a zoomed view without sending them, so the agent can check the coordinates it read off a rendered image. After every prompt the tool reports how the mask changed and flags common problems, such as a prediction that extends past the box it came from.

Views. The agent sees the image only through rendered views (Fig. 3; more examples in Figs. S6–S9 of the supplement): orthogonal slices with absolute voxel coordinates on the axes, patient directions and object outlines; montages of many slices; projections; and 3D surface renderings of an object from several directions. Statistics (size, connected components, overlaps) complement them.

Python. A persistent Python namespace gives the agent the image array and the workspace. The agent uses it to find candidates (thresholds, connected components, maximum intensity projections), to loop over many instances, and to draw its own figures (Fig. 5a). The tool can be restricted to computations or switched off.

![](images/2b940805f34182010e291d9e25fc5e7e92c63e2b44e43b85259759aa8c2105ac.jpg)

Figure 3: What the agent sees, here Codex segmenting the brainstem in a head-and-neck MRI (headneck\_mr, HaN-Seg [32]; unguided track). (a) The preview of its first scribble, in a sagittal slice. (b) The 3D rendering after this prompt: the object (169 ml) extends far to both sides (Dice 0.28), and (c) the montage shows that it covers the cerebellum. (d) The second prompt, a negative scribble through the cerebellum in an axial slice, removed most of the leak (Dice 0.80). (e) The third prompt, a short negative scribble at the lower end, shown in a sagittal slice with the mask after the second prompt. (f) The 3D rendering before the commit (Dice 0.82). The agent committed the brainstem with the note “Brainstem separated from cerebellum and cervical cord; superior midbrain boundary and cervicomedullary transition are approximate on 3 mm slices.” Panels (a)–(c) and (f) are renders the agent looked at. It sent the two corrections without a preview, so (d) and (e) were drawn afterwards with the same renderer from the logged prompts and masks.  
![](images/6aa019f4fcab244846534fe740581672a382180dee474ff91fc3c2ed29dea394.jpg)  
Figure 4: Excerpts from the skill (verbatim; [...] marks omitted text).

Objects larger than about 320 voxels along an axis are segmented on a downsampled copy of the image and scaled back up, since nnInteractive predicts 192<sup>3</sup> patches and is most accurate on objects that fit into one patch or require only small AutoZoom factors. The agent only provides a size estimate and continues to work in full-resolution coordinates.

## 3.3 The skill

Tools alone do not make a good annotator. The skill, a Markdown document that agents load when asked to segment, encodes how experienced users work with nnInteractive. It describes how nnInteractive behaves, how to read coordinates off rendered views, a step-by-step workflow, a playbook for corrections and rules for working efficiently (Fig. 4). Skills of this kind have also been shown to help agents operate scientific visualisation software [33].

(a) a figure the agent made for itself: liver mask (yellow), dark candidates inside it (red)  
![](images/3b7ab7ea45713b6197639ffbd406253a6e120a4ffaa3023b712ded0aadfcf4ea.jpg)  
Figure 5: Liver lesions from a one-line request. (a) A figure the agent drew for itself with its Python tool to review lesion candidates. (b) The committed lesions in one slice. (c) Lesion 6 as first segmented and (d) after the user asked to split it.

## 3.4 Deployment

nninteractive-agent setup claude (or codex) registers the MCP server and installs the skill, so that every new session can segment images; any other MCP client can use the server, and clients without skill support read the skill as an MCP prompt. nnInteractive runs in the server’s process on a local GPU or on a shared nninteractive-server. Open models behind an Anthropic-compatible API can drive Claude Code as well. As a proof of concept, Qwen3.8-27B segmented a spleen (MSD Spleen, case 10 [30, 31]) with Dice 0.93 in 34 minutes, and the benchmark’s bladder tumour, with the location given, with Dice 0.60 (Claude Code and Codex: 0.72 to 0.81). It ran on a shared server under high load, so we did not pursue this further. A web demo with 2D and 3D views and a chat lets people use the agent with immediate visual feedback (Fig. S5 in the supplement).

## 4 Agentic behaviour in practice

Before we measure how accurately agents segment named targets (Section 5), we show what an agent does with requests as people make them: often less specific, sometimes defined by function rather than by name, and with feedback along the way. The following examples come from the web demo with Claude Code (Opus 5.5, medium effort).

## 4.1 Finding and counting liver lesions

Asked only to “segment and count all liver lesions” in a contrast-enhanced CT (MSD Liver, case 155 [30, 31]), the agent made its own plan (Fig. 5, box below). It first segmented the liver as a helper object. In Python, it smoothed the image, thresholded dark voxels inside the eroded liver mask, labelled the connected components and drew a figure of candidate regions for itself (Fig. 5a). It then segmented each lesion as its own nnInteractive object, mostly starting from a box, separated two touching nodules with negative scribbles, reviewed the remaining candidates (dismissing vessels, fissures and noise), and deleted the helper. After three minutes and ten prompts it reported six definite lesions and one possible one, and flagged that lesion 6 might be two touching nodules. When told “lesion 6 seems to be a merger. Split into 2”, it checked the split with an intensity-based watershed of its own, reopened the object, removed the lateral nodule with a negative scribble, segmented that nodule as a new object, and confirmed that both boundaries agreed (Fig. 5c,d); this took 46 seconds and two prompts.

## Conversation excerpt (shortened)

User: segment and count all liver lesions

Agent: I’ve got a liver mask; now finding hypodense lesion candidates inside it. [. . . ] I count 6 definite liver lesions and 1 possible one. [. . . ] To find them, I outlined the liver as a temporary helper (deleted afterwards), searched it for dark regions, and looked at each candidate in zoomed slices. [. . . ] Whether #6 is one lesion or two: at axis0 158–162 it shows as two touching nodules that merge higher up. I kept it as one mass.

User: lesion 6 seems to be a merger. Split into 2

Agent: Splitting #6 into a lateral and a medial nodule. These are clearly separate at axis0 158–162. [. . . ] The split agrees with a separate check based on image brightness, which put the boundary in the same place. Count: 7 definite lesions (#1–6, #8), plus the uncertain small spot #7.

## 4.2 Requests that need knowledge

Four further requests (Fig. 1) need knowledge rather than search. The first three need anatomical knowledge; the fourth is not a medical image at all:

• “Segment the muscles used for chewing.” (TotalSegmentator MRI, s0506 [1]): the agent translated function into anatomy and segmented both masseters and both medial and lateral pterygoids (2 minutes 18 seconds, 7 prompts). It left out the temporalis and explained why: it “lies almost entirely above the scanned region”.

• “Now segment the muscles used for turning the head.”: both sternocleidomastoids (1 minute 30 seconds, 2 prompts). The agent named the deeper rotators it did not segment, because their borders were too faint on this scan, and offered to try.

• “Segment every rib as a separate object and name them by side and number.” (RibFrac, case 100 [2, 3]): the agent found 12 rib candidates per side with its own bone thresholding and prompted each with points along it. It noticed that its first pass had taken both collarbones for rib 1 and missed the right fifth rib under the shoulder blade, and its batch-wise reviews found that most ribs stopped short of the spine. It fixed all of this, rebuilt three ribs whose extensions had spilled into the spine, and reported that the numbering was counted from the first rib and not checked against the vertebrae. The result is 24 named ribs after 10.6 minutes and 223 prompts (\$3.86 at API prices).

• “snorkel and flashlight” (a CT scan of a packed suitcase, image provided by Smiths Detection): an image far from the medical and microscopy images nnInteractive was trained on. The agent recognised the snorkel as the J-shaped tube along one side of the case and segmented it, including its mouthpiece, with four scribbles. For the flashlight it chose a dense metal cylinder after ruling out several other items, and said it was unsure: it saw no lens, reflector or switch, and the cylinder could also be a large battery (3 minutes 12 seconds, 5 prompts). Told that this was likely the battery and that the flashlight, missing its battery, was somewhere else, the agent found an empty tube with a flared head next to it, which it had taken for a cup-like container before. It segmented the tube with one scribble and re-created the first object as “battery” (46 seconds, 2 prompts).

## 5 Benchmark

Images and targets. The benchmark consists of 14 public images: six CT, five MRI, one PET and two micro-CT, of humans, a mouse and an ant (Fig. 6, Table S1). Its 51 targets range from organs and tumours to muscles, bones, vessel trees and an insect brain. Every target is identifiable from its description alone: it is named, unique within the image, and its side is specified where relevant. Besides all 14 images, we report two groups: the nine images of human CT, MRI and PET of the head and neck, chest, abdomen and pelvis (radiology; 25 targets), the typical domain of medical segmentation models, and five images whose targets lie outside it (legs, mouse, ant; 26 targets): single muscles of the thigh and calf in fat-weighted research MRI, the 17 bones of the left foot in a whole-body CT of a cadaver, and a tumour and a brain in micro-CTs of a mouse and an ant. nnInteractive was not trained on any of these images, so the agents’ results are not inflated by training-data overlap; several sources are part of VoxTell’s training data (Section 6.3). The references of the foot were made by the source’s annotators with nnInteractive in 3D Slicer and checked by them, which may favour the tracks that use nnInteractive.

## Tracks.

• Unguided: the agent gets the image and the list of targets by name (Fig. S1 in the supplement). It has to find and segment them.

• Guided: the same, plus one voxel inside each target (the voxel deepest inside the reference). This separates segmenting from finding.

![](images/0248cc6d8468c127c918b6b186f55d9cd5ea3cde12142d016d90ad532741beab.jpg)  
Figure 6: The benchmark images, one slice each. Upper rows: the reference; lower rows: what Claude Code segmented from the request alone (unguided track, the middle of three repeats), with that run’s Dice averaged over the image’s targets. Fat-weighted MRI (calf, thigh) is shown on a log scale; for the whole-body CT (foot\_ct), a sagittal slice through the left foot in a bone window. All tracks: Figs. S3 and S4 in the supplement.

• One point: no agent; one positive point at the guided voxel per target. This is nnInteractive with the simplest possible prompt.

• Oracles: no agent; up to five simulated prompts per target with knowledge of the reference, each aimed at the largest error of the current mask (the largest connected region it misses or adds; before the first prompt, the whole target). The point oracle places a point at the voxel deepest inside that region, the scribble oracle a line along the middle of the region (the longest path through its skeleton) in the slice where the region is largest. Both use the same tools as the agents. We record the Dice after every prompt. As the agents prompt almost only with scribbles (Table 2), we compare each agent with the scribble oracle at the number of prompts the agent sent on each target (after five prompts for the 4–11% of targets where an agent sent more).

• VoxTell: no agent; VoxTell v1.1 [8] with exactly the target texts the agents get, all targets of an image in one call (for the 17 foot bones three per call, since VoxTell’s outputs for all 17 do not fit into the memory of our GPU; on another image, splitting the prompts changed its masks by less than 0.002 Dice).

Scoring. Each target is scored by Dice, using its object as it was when the agent last committed it; a later object that takes voxels from it (a tumour inside the liver) does not count against it. A missing target scores 0. Means are taken per image first: the targets of an image are averaged, and a track’s mean is the mean over the images (all 14, or those of one group), so that the image with 17 targets counts as much as an image with one.

Setup. nnInteractive ran locally with its public v1.0 weights on one NVIDIA RTX 4090. The agents were Claude Code 2.1 with Claude Opus 5.5 and Codex 0.162 with gpt-6.1-sol, both at medium reasoning effort. They worked through the nnInteractive tools, including a Python tool for their own code. Each agent track and the one-point track were run three times; the oracles and VoxTell once.

Table 1: Benchmark results (mean Dice over images, each image’s targets averaged first; agents: mean ± standard deviation of the track mean over three repeats, i.e. run-to-run variation; for the uncertainty over images see Section 6.1). Numbers of images in brackets. Radiology: human CT, MRI and PET of the head and neck, chest, abdomen and pelvis (25 targets); legs, mouse, ant: the thigh and calf MRIs, the foot bones in a whole-body CT, and the micro-CTs of a mouse and an ant (26 targets). Rad. without adrenal\_ct: the radiology images except the one on which VoxTell returns an empty mask (Section 6.3). “Scribble oracle, same prompts”: its Dice after as many prompts on each target as the agent sent on it (after five where the agent sent more). Best Dice per column in bold.
<table><tr><td rowspan="2">Track</td><td colspan="4">Dice</td><td rowspan="2">Prompts per target</td><td rowspan="2">Scribble oracle, same prompts</td><td rowspan="2">Minutes per image</td></tr><tr><td>all (14)</td><td>radiology (9)</td><td>rad. without adrenal_ct (8)</td><td>legs, mouse, ant (5)</td></tr><tr><td>Codex, guided</td><td></td><td></td><td>0.812</td><td>0.912</td><td>2.9</td><td>0.869</td><td>3.8</td></tr><tr><td>Claude Code, guided</td><td> $\mathbf { 0 . 8 5 5 } \pm 0 . 0 0 2$   $0 . 8 4 7 \pm 0 . 0 0 6$ </td><td>0.823 0.828</td><td>0.818</td><td>0.880</td><td>2.2</td><td>0.844</td><td>3.6</td></tr><tr><td>Codex, unguided</td><td> $0 . 8 4 2 \pm 0 . 0 0 4$ </td><td>0.815</td><td>0.804</td><td>0.892</td><td>2.8</td><td>0.886</td><td>4.2</td></tr><tr><td>Claude Code, unguided</td><td> $0 . 8 1 4 \pm 0 . 0 0 6$ </td><td>0.814</td><td>0.803</td><td>0.813</td><td>2.3</td><td>0.854</td><td>4.3</td></tr><tr><td>One point, no agent</td><td>0.709</td><td>0.669</td><td>0.651</td><td>0.779</td><td>1</td><td>0.769</td><td>0.2</td></tr><tr><td>VoxTell, text only</td><td>0.539</td><td>0.777</td><td>0.875</td><td>0.110</td><td>一</td><td></td><td>1.5</td></tr></table>

## 6 Results

## 6.1 The agents’ prompts are far better than a single click

With the request alone, the agents reach a mean Dice of 0.81 (Claude Code) and 0.84 (Codex); with one voxel per target as a hint, both reach 0.85 (Table 1). nnInteractive with one click per target at that same voxel reaches 0.71. The gain is largest where a single point cannot describe the target: on the liver of liver\_vessels\_ct, one point gives Dice 0.00 and the agents 0.95 to 0.96; on the liver tumour of hcc\_ct, 0.13 against 0.96. Locating the targets costs the agents little: the unguided tracks are only 0.01 (Codex) and 0.03 (Claude Code) below the guided ones. No agent run left a target without an object.

With 14 images, these means carry considerable uncertainty: 95% intervals from a bootstrap over images are 0.11 to 0.14 wide for the agents, for example [0.79, 0.91] for Codex, guided. Comparisons on the same images are more precise. Every agent track is ahead of one click, by 0.11 [0.01, 0.22] (Claude Code, unguided) to 0.15 [0.05, 0.26] (Codex, guided). The location hint helps Claude Code (+0.03 [0.01, 0.07]) but not Codex significantly (+0.01 [−0.001, 0.027]). The two agents do not differ clearly: without the hint, Codex is ahead by 0.03, but the interval includes zero ([−0.003, 0.070]); with the hint, the difference is 0.01 [−0.01, 0.03]. The Dice of every target in every track is listed in Table S2 of the supplement.

## 6.2 The agents’ prompts are close to the oracles’; they stop earlier

To judge the quality of the agents’ prompts, we compare them with the scribble oracle at the same number of prompts: for every target, its Dice after as many prompts as the agent sent on it (Table 1, Fig. 7b). With the location given, the scribble oracle is level with Claude Code (0.844 against 0.847; 95% interval of the difference [−0.03, 0.02]) and 0.015 ahead of Codex ([−0.01, 0.04]). Without the location, the scribble oracle is further ahead (0.04 for both agents): it knows where each target lies; the agents have to find it. After their first prompt, the guided agents reach 0.79, about as much as the scribble oracle after one prompt (0.77; the agents see their first scribble or box in a preview before sending it, whereas the oracle’s first scribble follows a fixed rule), and without the hint 0.74 (Codex) and 0.76 (Claude Code). The agents send 2.2 to 2.9 prompts per target, while the oracles keep improving with every further prompt (Fig. 7a): the agents judge their results from rendered views and stop when a result looks right, whereas the oracles compare each result with the reference, whose exact extent a request in plain language cannot fully convey.

## 6.3 A text-prompted model is more accurate within its domain, but brittle

VoxTell segments all targets of an image in about 20 seconds from the target texts alone (17 minutes for the 17 foot bones in the whole-body CT) and reaches a mean Dice of 0.54. This average conceals two distinct regimes. On seven of the nine radiology images (and on 19 of their 25 targets) it is more accurate than every agent, by 0.02 to 0.09 per image, and it does well on thin structures that the agents find hard (left optic nerve 0.59 against 0.02 to 0.34 for the agents, spinal cord 0.61 against 0.46 to 0.55, hepatic veins 0.64 against 0.27 to 0.53); on hcc\_ct it is level with them. On the adrenal tumour, however, it returns an empty mask, so its mean over the nine images (0.78) is below the agents’ (0.81 to

![](images/53c2aac33a9505cdef7e698596253faf5f69cd2ee392253c329ddf84f5a28feb.jpg)  
Figure 7: Agents compared with oracles (mean Dice over images). (a) Dice over the number of prompts per target: oracle trajectories, the agents at their mean number of prompts (filled: guided, hollow: unguided) and after their first prompt on each target (small markers at one prompt), nnInteractive with one point, and VoxTell. (b) Each agent next to the scribble oracle given the agent’s own number of prompts on each target.

Table 2: How the agents work (all runs: 14 images, three repeats each). Prompt types as a share of all prompts sent; “from code”: prompts the agent sent from its own Python code. Prompts and views per target: means over images.
<table><tr><td></td><td>Prompts per target</td><td>Scribbles</td><td>Boxes</td><td>Points</td><td>From code</td><td>Views per target</td></tr><tr><td>Claude Code, guided</td><td>2.2</td><td>94%</td><td>4%</td><td>2%</td><td>14%</td><td>9.1</td></tr><tr><td>Claude Code, unguided</td><td>2.3</td><td>94%</td><td>3%</td><td>2%</td><td>17%</td><td>10.4</td></tr><tr><td>Codex, guided</td><td>2.9</td><td>100%</td><td>0%</td><td>0%</td><td>0%</td><td>13.9</td></tr><tr><td>Codex, unguided</td><td>2.8</td><td>99%</td><td>0%</td><td>1%</td><td>0%</td><td>14.3</td></tr></table>

0.83; Table 1); without this image, it is 0.88 against 0.80 to 0.82. The empty mask is a matter of wording: “the tumor of the right adrenal gland”, the benchmark’s request, gives Dice 0, while “adrenal tumor” gives 0.79 (a check outside the benchmark). Several of the radiology source datasets are part of VoxTell’s training data (HaN-Seg, HNTS-MRG, RIDER Lung CT, HCC-TACE-Seg), so these numbers may be optimistic. On the second group VoxTell fails: it reaches 0.10 on the leg muscles (agents 0.74 to 0.87), 0.03 on the foot bones (agents 0.98), where five of its 17 masks are empty and most of the others mainly cover a different bone, and 0.17 on the mouse and the ant (agents 0.80 to 0.92). Its vocabulary of precomputed prompts names all of these muscles and each tarsal bone (the metatarsals and toe bones only as groups); the images, however, are far from standard radiology: fat-weighted research MRI, a whole-body CT of a cadaver and animals. This is the expected behaviour of a model that maps text to masks directly. The agent, in contrast, does not map text to masks: it recognises the target in the image and prompts nnInteractive.

## 6.4 Where the agents struggle

Thin and tubular structures are the hardest targets for the agents: the portal and hepatic veins (0.27 to 0.53), the left optic nerve (0.02 to 0.34) and the spinal cord (0.46 to 0.55). They are hard to prompt, since a scribble must follow a narrow structure across slices, and hard to judge from renderings. Per-image results for all tracks are in Fig. S2 and Figs. S3–S4 of the supplement, per-target results in Table S2. Without the location hint, Codex is ahead of Claude Code (0.84 against 0.81), mostly on the calf muscles and the ant brain; on this small benchmark, the difference is not clearly significant (Section 6.1).

## 6.5 How the agents work

Both agents follow the skill: almost all prompts are scribbles, boxes and points are rare (Table 2). Claude Code sends about a sixth of its prompts from its own Python code; Codex prompts only through the tools but looks at more views, including four to six times as many 3D renderings. An image takes about four minutes (3.6 targets on average). At API prices, Claude Code’s runs cost \$1.29 (guided) and \$1.46 (unguided) per image (\$115 for both tracks together), mostly for 2.8 to 3.1 million cached input tokens; Codex used about 1.7 to 1.9 million input tokens per image, 96% of them cached. The agents’ notes are informative: objects whose commit note expresses uncertainty (“approximate”, “uncertain”, . . . ) reach a mean Dice of 0.72 (Claude Code) and 0.68 (Codex), against 0.92 and 0.93 for the others.

## 7 Discussion

An extension, not a new agent. nninteractive-agent adds a capability to existing agents rather than building a new one. This has three consequences. First, it is not tied to one model: the same tools already serve two commercial agents and an open model, and we expect a better model to become a better annotator without any change on our side. Second, the agent keeps its general abilities (anatomical knowledge, coding, web search, conversation), on which the examples in Section 4 rely and which a specialized model fine-tuned for promptable segmentation tasks does not provide to the same extent. Third, segmentation becomes one tool among others in the agent’s usual environment: it needs no dedicated viewer, its results can feed into whatever else the agent does with the data, and further tools, such as a text-prompted model, a detector or a registration tool, could be added in the same way. The other side of this is a dependence on the agents’ providers, both for the agents’ quality and for which data may be sent to them (see Limitations).

What nninteractive-agent enables. With nninteractive-agent, an existing agent takes over the part of interactive segmentation that a promptable model leaves to its user. Told where a target is, it prompts nnInteractive almost as effectively as a simulated user who knows the answer, at the same number of prompts, and without that hint it loses little. More importantly, it handles requests that no fixed interface anticipates: counting lesions it first has to find, muscles described by their function, two dozen ribs that have to be numbered. It brings anatomical knowledge, plans the work, writes code where code is the better tool, checks its results, and says what it is unsure about. These abilities are the agent’s own; nninteractive-agent, a set of tools and a skill, connects them to nnInteractive without training a model.

Agentic versus text-prompted segmentation. The comparison with VoxTell shows the trade-off. A text-prompted model is fast and, on most standard radiology targets, more accurate than the agents. It is also brittle: a request worded differently from what it expects can return an empty mask, and on images far from standard radiology (single muscles in fat-weighted MRI, the bones of the foot, animals) it fails. The agent needs to recognise the target in the image and to prompt nnInteractive, which is not tied to a fixed set of structures; it therefore keeps working where the text-prompted model stops, at the price of minutes per image. The two are complementary. A text-prompted model could be one more tool for the agent, giving a first draft for standard targets that the agent checks and corrects with nnInteractive; we leave this for future work.

Limitations. The agents stop earlier than an oracle would; better self-assessment is the most direct way towards oracle performance. Thin and tubular structures remain difficult to prompt and to judge from renderings. A run takes minutes and, at API prices, costs \$1.3 to \$1.5 per image, which suits annotation and research use better than the processing of large cohorts. We did not compare the agents with human annotators. An experienced nnInteractive user would likely be more accurate, particularly on thin structures, and faster per target; the agent’s advantage is that nobody needs to sit at the screen. With a commercial agent, rendered views of the image are sent to the LLM provider, which has to be compatible with the terms of use of the data; open models served in-house avoid this; we tested one only as a proof of concept (Section 3.4). The benchmark is small (14 images, 51 targets), the oracles follow simple rules and show what well-placed prompts achieve rather than an upper bound. The nnInteractive and VoxTell weights are licensed for non-commercial use only.

## 8 Conclusion

General-purpose AI agents can act as 3D image annotators: with a set of tools and a skill, and without additional training, they operate nnInteractive much as human annotators do. They place prompts in slices, check the 3D result and correct it. From plain-language requests they reach a mean Dice of 0.81 to 0.84 on a diverse benchmark, and 0.85 when told where each target is, prompting nearly as effectively as an oracle given the same number of prompts; they keep working on data where a text-prompted model fails. Beyond prompting, they plan, write their own code and take corrections in plain language. As agents improve, we expect this way of annotating images to improve with them.

Code availability. nninteractive-agent, comprising the MCP server, the skill, the web demo, the benchmark scripts and the aggregated benchmark results (per track and per image): https://github.com/MIC-DKFZ/ nninteractive-agent (Apache-2.0). nnInteractive: https://github.com/MIC-DKFZ/nnInteractive.

Data availability. All images used in this work, except the suitcase CT scan in Fig. 1 (provided by Smiths Detection), are available from their original sources, the thigh and calf MRI on request; we do not redistribute any image or reference mask. The benchmark sources, their licences and references are listed in Table S1. The build script (benchmark/prepare.py) downloads and converts them, derives the reference masks as defined in benchmark/cases.json and verifies the result against published checksums. Some sources require a free registration (VEELA) or a manual download in a web browser (ant brains). The thigh and calf MRI is no longer in its figshare record and is available from its authors on request.<sup>1</sup> The Visible Human images are courtesy of the U.S. National Library of Medicine. The examples in Section 4 use MSD Liver (cases 51 and 155; CC BY-SA 4.0) [30, 31], TotalSegmentator MRI v2.0.0 (case s0506; CC BY-NC-SA 2.0) [1] and RibFrac (case 100; CC BY-NC 4.0) [2, 3]; the open-model test in Section 3.4 uses MSD Spleen (case 10; CC BY-SA 4.0).

Acknowledgements. This work was supported by the NCT Heidelberg Image Data Analysis Infrastructure (IDAI) and by the Helmholtz Association under the Helmholtz Foundation Model Initiative (HFMI), project “The Human Radiome Project” (THRP). MR was supported by the Google PhD Fellowship Program. BH was funded by “NUM 3.0” (FKZ: 01KX2524) and by the Vector Stiftung. PV is partially funded by an Else Kröner Clinician Scientist Endowed Professorship from the Else Kröner Fresenius Foundation (reference number: 2022\_EKCS.17).

## References

[1] Tugba Akinci D’Antonoli, Lucas K. Berger, Ashraya K. Indrakanti, et al. TotalSegmentator MRI: Robust sequence-independent segmentation of multiple anatomic structures in MRI. Radiology, 314(2):e241613, 2025.

[2] Liang Jin, Jiancheng Yang, Kaiming Kuang, et al. Deep-learning-assisted detection and segmentation of rib fractures from CT scans: Development and validation of FracNet. eBioMedicine, 62:103106, 2020.

[3] Jiancheng Yang, Rui Shi, Liang Jin, et al. Deep rib fracture instance segmentation and classification from CT on the RibFrac challenge. IEEE Transactions on Medical Imaging, 44(8):3410–3427, 2025.

[4] Fabian Isensee, Maximilian Rokuss, Lars Krämer, Stefan Dinkelacker, Ashis Ravindran, Florian Stritzke, Benjamin Hamm, Tassilo Wald, Moritz Langenberg, Constantin Ulrich, Jonathan Deissler, Ralf Floca, and Klaus Maier-Hein. nnInteractive: Redefining 3D promptable segmentation. arXiv preprint arXiv:2503.08373, 2025.

[5] Yuxin Du, Fan Bai, Tiejun Huang, and Bo Zhao. SegVol: Universal and interactive volumetric medical image segmentation. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 110746– 110783, 2024.

[6] Ziheng Zhao, Yao Zhang, Chaoyi Wu, et al. Large-vocabulary segmentation for medical images with text prompts. npj Digital Medicine, 8:566, 2025.

[7] Theodore Zhao, Yu Gu, Jianwei Yang, et al. A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities. Nature Methods, 22(1):166–176, 2025.

[8] Maximilian Rokuss, Moritz Langenberg, Yannick Kirchhoff, et al. VoxTell: Free-text promptable universal 3D medical image segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 37538–37557, 2026. arXiv:2511.11450.

[9] Anthropic. Introducing the Model Context Protocol. https://www.anthropic.com/news/ model-context-protocol, November 2024. Accessed 2026-10-19.

[10] Alexander Kirillov, Eric Mintun, Nikhila Ravi, et al. Segment anything. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4026, 2023.

[11] Tassilo Wald, Saikat Roy, Gregor Koehler, Nico Disch, Maximilian Rouven Rokuss, Julius Holzschuh, David Zimmerer, and Klaus Maier-Hein. SAM.MD: Zero-shot medical image segmentation capabilities of the Segment Anything Model. In Medical Imaging with Deep Learning (MIDL), Short Paper Track, 2023.

[12] Hallee E. Wong, Marianne Rakic, John Guttag, and Adrian V. Dalca. ScribblePrompt: Fast and flexible interactive segmentation for any biomedical image. In Computer Vision – ECCV 2024, Lecture Notes in Computer Science, pages 207–229. Springer, 2024.

[13] Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature Communications, 15:654, 2024.

[14] Haoyu Wang, Sizheng Guo, Jin Ye, et al. SAM-Med3D: A vision foundation model for general-purpose segmentation on volumetric medical images. IEEE Transactions on Neural Networks and Learning Systems, 36(10):17599–17612, 2025.

[15] Yufan He, Pengfei Guo, Yucheng Tang, et al. VISTA3D: A unified segmentation foundation model for 3D medical imaging. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20863–20873, 2025.

[16] Jun Ma, Zongxin Yang, Sumin Kim, et al. MedSAM2: Segment anything in 3D medical images and videos. arXiv preprint arXiv:2504.03600, 2025.

[17] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, et al. SAM 3: Segment anything with concepts. In International Conference on Learning Representations (ICLR), 2026.

[18] Muzhi Zhu, Yuzhuo Tian, Hao Chen, et al. SegAgent: Exploring pixel understanding capabilities in MLLMs by imitating human annotator trajectories. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3686–3696, 2025.

[19] Shengyuan Liu, Liuxin Bao, Qi Yang, et al. MedSAM-Agent: Empowering interactive medical image segmentation with multi-turn agentic reinforcement learning. arXiv preprint arXiv:2602.03320, 2026.

[20] Yankai Jiang, Qiaoru Li, Binlu Xu, et al. IBISAgent: Reinforcing pixel-level visual reasoning in MLLMs for universal biomedical object referring and segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20996–21005, 2026.

[21] Anglin Liu, Xu R. Cao, Yifan Shen, Yi Lu, Xiang Li, Qianqian Chen, and Jintai Chen. MedSAM3: Delving into segment anything with medical concepts. arXiv preprint arXiv:2511.19046, 2025.

[22] Chao Hao, Jun Xu, Ji Du, et al. Seg-Agent: Test-time multimodal reasoning for training-free language-guided segmentation. arXiv preprint arXiv:2605.12953, 2026.

[23] Weixiang Shen, Chengzhi Shen, Yanzhu Hu, et al. MedOpenClaw and MedFlowBench: Auditing medical agents in full-study workflows. arXiv preprint arXiv:2603.24649, 2026.

[24] Ayhan Can Erdur, Daniel Scholz, Jiazhen Pan, Benedikt Wiestler, Daniel Rueckert, and Jan C. Peeken. Agentic large language models for training-free neuro-radiological image analysis. arXiv preprint arXiv:2604.16729, 2026.

[25] Yannian Gu, Xizhuo Zhang, Linjie Mu, et al. CT-Flow: Orchestrating CT interpretation workflow with model context protocol servers. arXiv preprint arXiv:2603.00123, 2026.

[26] Qianchu Liu, Sheng Zhang, Guanghui Qin, et al. HealthAgentBench: A unified benchmark suite of realistic agentic healthcare environments for challenging frontier AI agents. arXiv preprint arXiv:2606.31179, 2026.

[27] Loïc A. Royer. Omega — harnessing the power of large language models for bioimage analysis. Nature Methods, 21(8):1371–1373, 2024.

[28] zhaoyouj. MCP-Slicer: 3D Slicer Model Context Protocol integration. https://github.com/zhaoyouj/ mcp-slicer, 2025. Accessed 2026-10-20.

[29] Royer Lab. napari-mcp: MCP server for AI-powered napari control. https://github.com/royerlab/ napari-mcp, 2025. Accessed 2026-10-20.

[30] Amber L. Simpson, Michela Antonelli, Spyridon Bakas, et al. A large annotated medical image dataset for the development and evaluation of segmentation algorithms. arXiv preprint arXiv:1902.09063, 2019.

[31] Michela Antonelli, Annika Reinke, Spyridon Bakas, et al. The medical segmentation decathlon. Nature Communications, 13:4128, 2022.

[32] Gašper Podobnik, Primož Strojan, Primož Peterlin, Bulat Ibragimov, and Tomaž Vrtovec. HaN-Seg: The head and neck organ-at-risk CT and MR segmentation dataset. Medical Physics, 50(3):1917–1927, 2023.

[33] Kuangshi Ai, Haichao Miao, Kaiyuan Tang, Shusen Liu, and Chaoli Wang. SciVisAgentSkills: Design and evaluation of agent skills for scientific data analysis and visualization. arXiv preprint arXiv:2606.05525, 2026.

[34] Petr Jordan, Philip M. Adamson, Vrunda Bhattbhatt, et al. Pediatric chest/abdomen/pelvic CT exams with expert organ contours (Pediatric-CT-SEG). The Cancer Imaging Archive, 2021. Version 2. https://doi.org/10. 7937/TCIA.X0H0-1706.

[35] Ahmed W. Moawad, David Fuentes, Ali Morshid, et al. Multimodality annotated HCC cases with and without advanced imaging segmentation (HCC-TACE-Seg). The Cancer Imaging Archive, 2021. Version 1. https: //doi.org/10.7937/TCIA.5FNA-0924.

[36] Ahmed W. Moawad, Ayahallah A. Ahmed, Mohab ElMohr, et al. Voxel-level segmentation of pathologicallyproven adrenocortical carcinoma with Ki-67 expression (Adrenal-ACC-Ki67-Seg). The Cancer Imaging Archive, 2023. Version 1. https://doi.org/10.7937/1FPG-VM46.

[37] Binsheng Zhao, Lawrence H. Schwartz, Mark G. Kris, and Gregory J. Riely. Coffee-break lung CT collection with scan images reconstructed at multiple imaging parameters. The Cancer Imaging Archive, 2015. RIDER Lung CT, version 3. https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR.

[38] Leonard Wee, Hugo Aerts, Petros Kalendralis, and Andre Dekker. RIDER Lung CT segmentation labels from: Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. The Cancer Imaging Archive, 2020. Version 2. https://doi.org/10.7937/tcia.2020.jit9grk8.

[39] Paul Kinahan, Mark Muzi, Brian Bialecki, Ben Herman, and Laura Coombs. Data from the ACRIN 6668 trial NSCLC-FDG-PET. The Cancer Imaging Archive, 2019. Version 2. https://doi.org/10.7937/tcia.2019. 30ilqfcl.

[40] Jeff Van Oss, Gowtham Krishnan Murugesan, Diana McCrumb, and Rahul Soni. Image segmentations produced by BAMF under the AIMI annotations initiative. Zenodo, 2024. Version 2.0.2. https://doi.org/10.5281/ zenodo.13244892.

[41] Gowtham Krishnan Murugesan, Diana McCrumb, Mariam Aboian, et al. AI-generated annotations dataset for diverse cancer radiology collections in NCI image data commons. Scientific Data, 11:1165, 2024.

[42] Kareem Wahid, Cem Dede, Mohamed Naser, and Clifton Fuller. Training dataset for HNTSMRG 2024 challenge. Zenodo, 2024. https://doi.org/10.5281/zenodo.11199559.

[43] Kangyang Cao, Yujian Zou, Chang Zhang, et al. A multicenter bladder cancer MRI dataset and baseline evaluation of federated learning in clinical application. Scientific Data, 11:1147, 2024.

[44] Vincent Wohlfarth, Yeshe Manuel Kway, Ananya Sood, et al. Deep learning-based framework for comprehensive quantification of thigh and calf muscles and adipose tissues from MRI. Magnetic Resonance Materials in Physics, Biology and Medicine, 39(2):305–318, 2026.

[45] Seyed Hamidreza Alavi and Malte Asseln. Visible-Human-3d-Models. Hugging Face, 2026. https://doi. org/10.57967/hf/10464.

[46] Thor E. Andreassen, Donald R. Hume, Landon D. Hamilton, Karen E. Walker, Sean E. Higinbotham, and Kevin B. Shelburne. Three dimensional lower extremity musculoskeletal geometry of the visible human female and male. Scientific Data, 10:34, 2023.

[47] Tugçe Toprak, Ziya Ata Yazıcı,˘ ˙Ilkay Öksüz, et al. VEELA challenge – vessel extraction and extrication for liver analysis. In IEEE 35th International Workshop on Machine Learningfor Signal Processing (MLSP), pages 1–6, 2025.

[48] Ziya Ata Yazıcı, N. Sinem Gezer, ˙Ilkay Öksüz, et al. VEELA: A clinically-constrained benchmark for liver vessel segmentation in computed tomography angiography. arXiv preprint arXiv:2605.22357, 2026.

[49] Ali Emre Kavur, M. Alper Selver, Oguz Dicle, Mustafa Barı¸s, and N. Sinem Gezer. CHAOS – combined (CT-MR)˘ healthy abdominal organ segmentation challenge data. Zenodo, 2019. Version 1.03. https://doi.org/10. 5281/zenodo.3362844.

[50] Malte Jensen, Andreas Clemmensen, Jacob Gorm Hansen, et al. 3D whole body preclinical micro-CT database of subcutaneous tumors in mice with annotations from 3 annotators. Scientific Data, 11:1021, 2024.

[51] Evropi Toulkeridou, Carlos Gutierrez, Daniel Baum, Kenji Doya, and Evan Economo. Automated segmentation of insect anatomy from micro-CT images using deep learning. Natural Sciences, 3(4):e20230010, 2023.

[52] Evropi Toulkeridou, Carlos Gutierrez, Daniel Baum, Kenji Doya, and Evan Economo. Automated segmentation of insect anatomy from micro-CT images using deep learning (data). Dryad, 2023. Version 5. https://doi. org/10.5061/dryad.qz612jmgv.

## Supplementary material

Table S1: Benchmark sources. The build script downloads these sources and derives the reference masks as defined in benchmark/cases.json.
<table><tr><td>Image</td><td>Targets</td><td>Source, licence</td></tr><tr><td>pediatric_ct</td><td>liver, spleen, right kidney, bladder, heart, pancreas, gallbladder, left adrenal</td><td>Pediatric-CT-SEG (TCIA) [34], CC BY 4.0</td></tr><tr><td>hcc_ct</td><td>liver, liver tumour</td><td>HCC-TACE-Seg (TCIA) [35], CC BY 4.0</td></tr><tr><td>adrenal_ct</td><td>adrenal tumour</td><td>Adrenal-ACC-Ki67-Seg (TCIA) [36], CC BY 4.0</td></tr><tr><td>lung_ct</td><td>lung tumour</td><td>RIDER Lung CT, version 3 (TCIA) [37], CC BY 4.0; tumour mask from RIDER-LungCT-Seg</td></tr><tr><td>lung-pet</td><td>FDG-avid tumour</td><td>(TCIA) [38], CC BY 3.0 ACRIN-NSCLC-FDG-PET (TCIA) [39], CC BY 3.0; lesion mask from the BAMF/AIMI annotations [40, 41], CC BY 4.0</td></tr><tr><td>headneck_mr</td><td>mandible, brainstem, left parotid, right submandibular gland, thyroid, spinal cord,</td><td>HaN-Seg [32], CC BY-NC-ND 4.0</td></tr><tr><td>headneck_tumor_mr</td><td>left optic nerve primary tumour</td><td>HNTS-MRG 2024 [42], CC BY-NC 4.0</td></tr><tr><td>bladder_mr</td><td>bladder tumour</td><td>FedBCa [43], CC BY 4.0</td></tr><tr><td>thigh_mr</td><td>rectus femoris, sartorius, gracilis, vastus lateralis</td><td>Thigh and calf muscle MRI [44], on request from the authors (former figshare release: CC BY 4.0, non-commercial use only)</td></tr><tr><td>calf_mr</td><td>soleus, tibialis anterior, medial gastrocnemius</td><td>same</td></tr><tr><td>foot_ct</td><td>the 17 bones of the left foot: talus, calcaneus, navicular, cuboid, three cuneiforms, five metatarsals, the phalanges of each toe</td><td>Visible Human Full-Skeleton 3D Bone Models (Hugging Face) [45], CC BY 4.0, on the CT of the NLM Visible Human Female as aligned by [46], CC BY 4.0; courtesy of the U.S. National</td></tr><tr><td>liver_vessels_ct</td><td>liver, portal vein, hepatic veins</td><td>Library of Medicine VEELA 2025 [47, 48]; images from CHAOS [49], CC BY-NC-SA 4.0; no licence stated for the VEELA labels</td></tr><tr><td>mouse_microct</td><td>subcutaneous tumour</td><td>Subcutaneous tumour micro-CT database (ERDA) [50], CC BY (version not stated)</td></tr><tr><td>ant_microct</td><td>brain</td><td>Ant brains in micro-CT (Dryad) [51, 52], CC0 1.0</td></tr></table>

Table S2: Dice per target and track (agents and one point: mean over three repeats; oracles: after five prompts).
<table><tr><td>Image</td><td>Target</td><td>Codex, guided</td><td>Claude Code, guided</td><td>Codex, unguided</td><td>Claude Code, unguided</td><td>One point</td><td>VoxTell</td><td>Scribble oracle, 5 prompts</td><td>Point oracle, 5 prompts</td></tr><tr><td>adrenal_ct</td><td>adrenal tumour</td><td>0.91</td><td>0.91</td><td>0.90</td><td>0.90</td><td>0.82</td><td>0.00*</td><td>0.93</td><td>0.91</td></tr><tr><td>ant_microct</td><td>brain</td><td>0.93</td><td>0.79</td><td>0.93</td><td>0.79</td><td>0.44</td><td>0.33</td><td>0.94</td><td>0.74</td></tr><tr><td>bladder_mr</td><td>bladder tumour</td><td>0.76</td><td>0.77</td><td>0.71</td><td>0.76</td><td>0.81</td><td>0.86</td><td>0.86</td><td>0.85</td></tr><tr><td>calf_mr</td><td>soleus</td><td>0.75</td><td>0.79</td><td>0.74</td><td>0.58</td><td>0.57</td><td>0.00</td><td>0.88</td><td>0.84</td></tr><tr><td></td><td>tibialis anterior</td><td>0.83</td><td>0.81</td><td>0.82</td><td>0.63</td><td>0.72</td><td>0.06</td><td>0.91</td><td>0.86</td></tr><tr><td></td><td>medial gastrocnemius</td><td>0.86</td><td>0.87</td><td>0.80</td><td>0.46</td><td>0.90</td><td>0.00</td><td>0.95</td><td>0.94</td></tr><tr><td>foot_ct</td><td>left talus</td><td>0.99</td><td>1.00</td><td>0.99</td><td>0.99</td><td>1.00</td><td>0.38</td><td>0.99</td><td>0.99</td></tr><tr><td></td><td>left calcaneus</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.06</td><td>0.99</td><td>0.99</td></tr><tr><td></td><td>left navicular</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>1.00</td><td>0.00</td><td>0.99</td><td>0.99</td></tr><tr><td></td><td>left cuboid</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.00</td><td>0.99</td><td>0.99</td></tr><tr><td></td><td>left medial cuneiform</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.00</td><td>0.99</td><td>0.99</td></tr><tr><td></td><td>left intermediate cuneiform</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.99</td><td>0.00</td><td>0.98</td><td>0.98</td></tr><tr><td></td><td>left lateral cuneiform</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.00</td><td>0.98</td><td>0.99</td></tr><tr><td></td><td>left metatarsal 1</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.92</td><td>0.00</td><td>0.99</td><td>0.99</td></tr><tr><td></td><td>left metatarsal 2</td><td>0.97</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.99</td><td>0.00</td><td>0.98</td><td>0.99</td></tr><tr><td></td><td>left metatarsal 3</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.00</td><td>0.98</td><td>0.98</td></tr><tr><td></td><td>left metatarsal 4</td><td>0.98</td><td>0.98</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.00</td><td>0.97</td><td>0.98</td></tr><tr><td></td><td>left metatarsal 5</td><td>0.99</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.91</td><td>0.00</td><td>0.99</td><td>0.99</td></tr><tr><td></td><td>left big toe phalanges</td><td>0.97</td><td>0.97</td><td>0.97</td><td>0.98</td><td>0.78</td><td>0.00</td><td>0.98</td><td>0.99</td></tr><tr><td></td><td>left toe 2 phalanges</td><td>0.96</td><td>0.97</td><td>0.97</td><td>0.96</td><td>0.88</td><td>0.03</td><td>0.98</td><td>0.98</td></tr><tr><td></td><td>left toe 3 phalanges</td><td>0.92</td><td>0.92</td><td>0.98</td><td>0.94</td><td>0.87</td><td>0.00</td><td>0.96</td><td>0.98</td></tr><tr><td></td><td>left toe 4 phalanges</td><td>0.97</td><td>0.96</td><td>0.97</td><td>0.96</td><td>0.86</td><td>0.00</td><td>0.98</td><td>0.97</td></tr><tr><td></td><td>left toe 5 phalanges</td><td>0.97</td><td>0.95</td><td>0.97</td><td>0.96</td><td>0.94</td><td>0.00</td><td>0.98</td><td>0.98</td></tr><tr><td>hcc_ct</td><td>liver</td><td>0.97</td><td>0.96</td><td>0.97</td><td>0.97</td><td>0.61</td><td>0.97</td><td>0.88</td><td>0.97</td></tr><tr><td></td><td>liver tumour</td><td>0.96</td><td>0.96</td><td>0.96</td><td>0.96</td><td>0.13</td><td>0.94</td><td>0.96</td><td>0.97</td></tr><tr><td>headneck_mr</td><td>mandible</td><td>0.74</td><td>0.70</td><td>0.77</td><td>0.73</td><td>0.20</td><td>0.81</td><td>0.80</td><td>0.82</td></tr><tr><td></td><td>brainstem</td><td>0.77</td><td>0.81</td><td>0.80</td><td>0.75</td><td>0.27</td><td>0.86</td><td>0.82</td><td>0.81</td></tr><tr><td></td><td>left parotid gland</td><td>0.82</td><td>0.83 0.79</td><td>0.81 0.80</td><td>0.83</td><td>0.83 0.80</td><td>0.87 0.85</td><td>0.90</td><td>0.89</td></tr><tr><td></td><td>right submandibular gland</td><td>0.80 0.68</td><td>0.68</td><td>0.66</td><td>0.80 0.66</td><td>0.48</td><td>0.76</td><td>0.89 0.77</td><td>0.85</td></tr><tr><td></td><td>thyroid gland spinal cord</td><td>0.49</td><td>0.46</td><td>0.52</td><td>0.55</td><td>0.41</td><td>0.61</td><td>0.79</td><td>0.72 0.57</td></tr><tr><td></td><td>left optic nerve</td><td>0.27</td><td>0.22</td><td>0.34</td><td>0.02</td><td>0.02</td><td>0.59</td><td>0.64</td><td>0.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>headneck_tumor_mr</td><td>primary tumour</td><td>0.79</td><td>0.79</td><td>0.80</td><td>0.78</td><td>0.74</td><td>0.88</td><td>0.88</td><td>0.88</td></tr><tr><td>liver_vessels_ct</td><td>liver</td><td>0.95</td><td>0.96</td><td>0.95</td><td>0.96</td><td>0.00</td><td>0.96</td><td>0.96</td><td>0.95</td></tr><tr><td></td><td>portal vein hepatic veins</td><td>0.41 0.42</td><td>0.47 0.53</td><td>0.42 0.27</td><td>0.50 0.30</td><td>0.35 0.20</td><td>0.53 0.64</td><td>0.49 0.44</td><td>0.47 0.53</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>lung_ct</td><td>lung tumour</td><td>0.91</td><td>0.91</td><td>0.92</td><td>0.90</td><td>0.92</td><td>0.95</td><td>0.93</td><td>0.94</td></tr><tr><td>lung_pet</td><td>tumour</td><td>0.93</td><td>0.94</td><td>0.93</td><td>0.94</td><td>0.92</td><td>0.96</td><td>0.94</td><td>0.94</td></tr><tr><td>mouse_microct</td><td>tumour</td><td>0.91</td><td>0.89</td><td>0.83</td><td>0.82</td><td>0.91</td><td>0.00</td><td>0.96</td><td>0.96</td></tr><tr><td>pediatric_ct</td><td>liver</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98 0.98</td><td>0.98 0.98</td><td>0.98 0.98</td><td>0.98 0.98</td><td>0.98 0.98</td></tr><tr><td></td><td>spleen right kidney</td><td>0.98 0.98</td><td>0.98 0.97</td><td>0.98 0.98</td></table>

With the prompt “adrenal tumor”, VoxTell reaches Dice 0.79 (a check outside the benchmark).

![](images/36f46ccdcd69e4118c4d48333bcc76aefd3af7f0051a4448d55eb734035d486a.jpg)  
Figure S1: What the agents and VoxTell receive (verbatim; file paths shortened). Every request follows one template: a one-line description of the image with its path, then the targets in a fixed order, each with an id and a text, and in the guided track one voxel inside each target. Codex receives the instructions (a) at the start of the message, Claude Code as an appended system prompt (with “load the nninteractive:” in place of “read”). VoxTell receives only the target texts. The description and target texts of every image are listed in benchmark/cases.json.

![](images/b43f290b3a60ff26e004a0f219f7027ea86cfb43dd648b4ad2f7cf06ea3c7c97.jpg)  
Figure S2: Mean Dice per image and track (over the image’s targets and, for the agents, the three repeats).

VoxTell (text)

![](images/a6716ca6fb83f84b34b1313fd940d2c6d36376a8ebcd4431d3411060337bde7f.jpg)  
Figure S3: All tracks, images 1–7: reference, VoxTell, and the middle of three repeats of each agent track. Numbers: Dice of that run, averaged over the image’s targets.

![](images/4c55301133f1a4253b50ca1b82591dd94f95187905db52b3616944b3abb243a6.jpg)  
Figure S4: All tracks, images 8–14 (as Fig. S3).

![](images/94c3b255df96ace479d45b6f222178ab9eae49f6c887901a06deb8540a3a2b8a.jpg)  
Figure S5: The web demo after “Segment and count all liver lesions.” (MSD Liver, case 51, Claude Opus 5.5): axial, coronal, sagittal and 3D views, the objects, and the agent’s report.

(a) first scribble (preview)  
(b) after prompt 1: Dice 0.62  
(c) correction (preview)  
![](images/c75a042bbc6016c9fcd7c19da5b91b56a6e144c456cd1f8552743aa80c103356.jpg)  
Figure S6: Previews and slices: Claude Code segmenting the heart in a paediatric CT (pediatric\_ct, Pediatric-CT-SEG [34]; guided track). All panels are renders the agent requested. (a) The preview of its first scribble. (b) A montage after this prompt: only the left chambers are segmented (Dice 0.62). (c) The preview of a correction: the line lies in the lung, and the tool reports air intensities along it. The agent dropped this line without sending it. (d) A zoomed view of the slice: the mask covers the left chambers but not the right heart. The agent wrote “Heart prediction only covers the left chambers; adding the right heart with a positive scribble” and (e) drew the line again, inside the right heart, and sent it. (f) The montage after this second prompt. The agent committed the heart (Dice 0.92) with the note “Includes proximal pulmonary trunk root; great vessels otherwise excluded.”

(a) view  
![](images/b7d90b2b2392e7f446e1ca3f8b3e393249774ae95015c577f3e522bbadfe58f0.jpg)

(b) scribble preview  
![](images/c9eebf3bbe247c1a5f7111d0be3085132607d471a0f38867f140a26c7e81b3b3.jpg)

(c) shortened, then sent  
![](images/51906c5ebe87dceace4d8da790056e5c78dd3ef3c940618e18af6492284e6f31.jpg)

(d) result, axial and coronal  
![](images/e58a4c198f9a06e58bbaa76c117db54e6cd1e462a33947ddb81387c2f9a2e9dc.jpg)

(e) render3d, before the commit  
![](images/1a5d978442e2087b08d823e73e5bc73cfed649c3952e5c5f24a761da1c329255.jpg)  
Figure S7: A single prompt: Claude Code segmenting the pancreas, a low-contrast organ, in a paediatric CT (pediatric\_ct; guided track). (a) The view in which the agent looked for the pancreas, with voxel coordinates, patient directions and the organs committed so far. (b) The preview of its first scribble. (c) The agent shortened the scribble and sent it. (d) The result of this single prompt, checked in an axial and a coronal slice. (e) The 3D rendering of the pancreas (yellow) with its committed neighbours. The agent then committed it (Dice 0.89) with the note “Low contrast against adjacent bowel; tail and head boundaries uncertain.”

(a) view  
![](images/188b9c50d23af03799e1790e7f26dbf2224380292ea052453a30fba4ca0f76e2.jpg)

(b) scribble preview  
![](images/ca0a03e6cc3a1c33157c362448fa9e8d0e7f2aa3b93447ba6cf010492c1d07fa.jpg)

(c) result, coronal  
![](images/2a87dd9e65f2124a0e2961e1804bfb9bc9a621b6c1763433d7db7b86efd4e8a6.jpg)

(d) second scribble  
![](images/be1857b097aaddd6bdf1194267786a272c576945fd7d6ddf7a0959c783110602.jpg)

(e) render3d, before the commit  
![](images/1241d4ceb1d3e872929f9272f0318b19d3dbd013a658a1a6cce3beca83e62a82.jpg)  
Figure S8: A check in another plane: Claude Code segmenting the liver in a contrast-enhanced CT from the request alone (hcc\_ct, HCC-TACE-Seg [35]; unguided track). (a) An axial view with voxel coordinates and patient directions. (b) The preview of its first scribble, along the liver. (c) The result after that prompt (Dice 0.66), checked in a coronal slice: the large tumour is missing (the agent segments the liver including its tumour and separates the tumour later). (d) The preview of a second scribble in that slice, which adds the tumour. (e) The 3D rendering of the liver before the agent commits it (Dice 0.97).

(a) view  
![](images/e6c6174928f28f0580315abaed2dc5f832bedaea285a4162291f412474c5d3e0.jpg)  
(b) scribble preview (c) result, another slice  
(d) render3d  
Figure S9: An insect brain: Claude Code segmenting the brain in a micro-CT of an ant’s head (ant\_microct [51, 52]; guided track). (a) A view with voxel coordinates and anatomical directions. (b) The preview of its first scribble, along the brain. (c) The result after that prompt, checked in another slice. (d) The 3D rendering of the object from three directions. The final object, after four prompts (one of them undone), reaches Dice 0.82.