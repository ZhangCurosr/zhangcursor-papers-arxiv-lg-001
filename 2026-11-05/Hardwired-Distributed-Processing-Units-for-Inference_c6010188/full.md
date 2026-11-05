# Hardwired Distributed Processing Units for Inference:

# Neural Decisions at the Sensor that Free the Host

Raj Mali

Independent Researcher

raj.mali.research@gmail.com

ORCID: 0009-0001-6351-0062

Abstract—A camera normally sends every pixel to a host processor, which has to wait for a whole frame before it can decide anything. Hardwired distributed processing units (HDPUs) move routine work to the sensor, but so far we have used them only for pre-processing: edges, corners and a steering signal. Here we show that an HDPU can take a neural decision as well, and so lift the inference off the host. We build one for zebra-crossing detection on a low-cost Artix-7 FPGA. An OV7670 camera feeds a hardware pipeline that reduces each frame to 80 × 60 pixels, restores its contrast and evaluates a four-layer binarized CNN, with no processor, no operating system and no full-resolution frame store. The unit decides on every camera frame, in exactly 2,269,477 clock cycles (34.04 ms) whatever the image, and on the board it returns the same scores and decisions as the bit-exact software model on all 639 test images. The whole detection path takes 1,814 LUTs. On street photographs, 95.7 % of its answers are correct when it answers instead of reporting “not sure”, and a hardware contrast stage lifts accuracy on camera-like dashcam frames from 47.7 % to 78.2 %. The host runs none of this work. It is spared 8.13 ms of a core in every 37.49-ms frame period, and a full-precision ResNet-18 would take longer than a frame period even on 16 threads. It also receives 16 bytes per frame instead of 614,400, 38,400 times less.

Index Terms—hardwired distributed processing units, nearsensor inference, binarized neural networks, FPGA, offload, determinism

## I. INTRODUCTION

Most vision systems still follow one routine. The camera sends every pixel to a host processor, the host stores the frame, and only then does a program run over it. Processors do this well, but the routine has two costs that grow with the work we ask of it. Nothing can be decided before a whole frame has arrived, and every frame takes a share of the host’s time, whether or not anything interesting is in it.

In earlier work we proposed taking the routine, always-on part of that job away from the host. A hardwired distributed processing unit (HDPU) is a small fixed datapath attached to the sensor and dedicated to one task. It works on the pixel stream as the pixels arrive, shares no bus or scheduler with its neighbours, and passes only a compact result to the host [1]. Our first units, built from one library of operators that share a timing contract [2], did pre-processing: Canny edges, Harris corners and a line tracker that emits a steering code. The decision itself was still left to the host.

In modern systems that decision is made by a neural network, and a network is normally run as a program. A processor, or an accelerator that the processor controls [3], fetches instructions and weights from memory for every input. This is flexible, but it is also where most of the energy goes [4], [5], and the host stays busy for as long as the network runs. Industry has started to move the other way. Taalas builds chips in which the weights and dataflow of one model are part of the silicon, and reports up to 17,000 generated tokens per second for Llama 3.1 8B [6]; in August 2026 AMD announced that it would acquire the company [7].

This paper asks whether the sensor can take the same step. Can an HDPU not only prepare the data but also make the decision? We answer with a working unit. It detects zebra crossings, a task that matters for navigation aids for visually impaired pedestrians and for driver assistance [8]–[10], on a low-cost Artix-7 FPGA, with a binarized CNN [11], [12] that replaces multiply-accumulate with XNOR and population count. Our contributions are:

1) An inference HDPU. A unit that turns camera frames into decisions at the sensor, with no processor in the loop. It decides on every frame in a fixed 2,269,477 clock cycles, and on the board its scores and decisions equal those of its bit-exact model on all 639 test images (Sections III and V-B).

2) The sensor side of the decision. Pre-processing, a hardware contrast stage that recovers the accuracy a real sensor costs (47.7 % to 78.2 % on camera-like frames), and a decision with a not-sure band that makes 95.7 % of its street-photo answers correct, all in 1,814 LUTs (Sections III and V-D).

3) A complete verification chain, from a Python integer model through RTL simulation to the physical board, with every level compared against the same model (Section IV).

4) A measured account of what the host is spared: the inference time, the share of every frame period, and the data per frame, which falls from 614,400 to 16 bytes (Section V-G).

## II. BACKGROUND AND RELATED WORK

HDPUs. An HDPU is a fixed datapath for one task, with no instruction stream, operating system or frame buffer, placed at the sensor in the spirit of near-sensor computing [13]. Several HDPUs can watch the same pixel stream without a common bus, arbiter or scheduler, so adding a unit costs logic but no time. A Canny unit, a Harris corner unit and a line tracker already run side by side on one camera stream and return part of a host core to the host in every frame [1], all built from 3×3 operators that answer exactly two clock cycles after their window [2].

Binarized networks. A binarized network constrains weights and activations to ±1, so a multiply-accumulate over n inputs becomes an XNOR of two n-bit words followed by a population count [11], [12]. Batch normalisation followed by the sign function folds into one integer threshold per neuron [14].

Inference on FPGAs. FINN builds streaming dataflow accelerators for binarized networks whose folding factors trade throughput for area [14], [15]; its CIFAR-10 prototype classifies 21,906 images per second at 80.1 % accuracy on a Zynq board [14]. hls4ml unrolls small networks with their weights as constants to reach sub-microsecond decisions in particle-physics triggers [16], and LogicNets and LUTNet map neurons directly onto LUT truth tables [17], [18]. We use the same techniques. What we add is the placement: the unit sits at the camera, decides on every frame, and needs nothing around it.

Model-specific silicon. The TPU executes inference in a systolic array that receives its instructions and data from a host over PCIe [3]. Groq’s tensor streaming processor schedules all data movement at compile time, so its execution is deterministic [19]. Taalas goes furthest and makes the weights part of the chip [6]. An inference HDPU shares the last two properties, fixed timing and weights beside the datapath, and sits at the sensor.

Offload and baselines. Offload pays when invoking the accelerator and moving data to it cost little relative to the work [20], and reported speed-ups often shrink against optimised processor code [21]. An HDPU is never invoked and never receives data from the host, so the cost it removes is the host’s own: time and traffic.

Zebra-crossing detection. Published detectors run on phones [8], [9] or on embedded GPU modules; CDNet reports an F1 score above 94 % at 33.1 frames per second on a Jetson Nano [10]. The engine of the unit described here was first designed and verified in simulation in [22].

## III. AN INFERENCE HDPU AT THE SENSOR

## A. Organisation

Fig. 1 contrasts the two organisations. In the host-centred one, the frame is stored, moved to the host and processed by a program, so the decision cannot exist before the frame has arrived and the host has spent its time. In the inference HDPU the unit reads the camera pins directly, in parallel with the path that displays the picture, so it never asks anyone for data. It has no instruction stream: the network’s weights and thresholds sit in on-chip memory beside a fixed datapath. Its schedule is fixed, because a decision takes the same number of clock cycles for every frame, a number set by the network’s layer table and by nothing else. And its output is a decision of a few bytes, not a frame.

## B. From pixels to the network’s input

Fig. 2 shows the detection path. The OV7670 delivers 640 × 480 RGB565 frames [23] from a 21.33 MHz clock; with its default VGA timing that is 26.68 frames per second, or 37.49 ms per frame (model). A pre-processing block converts every pixel to gray as $( 7 7 R + 1 5 0 G + 2 9 B ) \gg 8 .$ , exactly as in training, sums 8 × 8 blocks, shifts by six, and writes the resulting 80 × 60 image into one of two alternating frame banks. The sums accumulate as the pixels arrive, so no fullresolution frame is ever stored for detection, and a frame is used only if all of its 480 lines are complete.

The sensor’s pictures are flat (Fig. 3). The network was trained on photographs that use the full range of gray levels, so an auto-contrast stage stretches every frame before the network sees it. A 256-bin histogram gives the 25th darkest and the 25th brightest pixel, a 0.5 % trim so that a lamp or a dark corner does not defeat the stretch, and the gain is capped at 2.7 so that flat surfaces are not turned into stripe-like texture.

## C. The inference engine

One engine evaluates all five layers in turn (Table I). The first layer applies binary 3×3 weights to the 8-bit pixels, offset by 128. The next three apply binary 3 × 3 weights to binary feature maps by XNOR and population count over all input channels at once, and each is followed by a threshold and a 2×2 max-pool, which on binary values is the OR of four bits. The output neuron sums 960 binary products into a score that is compared with a threshold of 477. All thresholds come from folding batch normalisation in Python, with negative gains folded into the weights so that every comparison is ≥. In all, a decision takes 37.1 million binary multiply-accumulates, from 93,408 one-bit weights (11.4 KB).

The engine walks the loop nest $( p _ { y } , p _ { x } , s , c , k _ { y } , k _ { x } )$ over pooled output positions, the four positions of each pooling window, output channels and kernel taps. It processes one kernel tap per clock in a three-stage pipeline: addresses; a partial sum, which is either a signed pixel or the population count of a 64-bit XNOR; and accumulation, threshold, pooling and write-back. Feature maps are stored as one 64-bit word per pixel, one bit per channel; the weights occupy 2,048 words of 64 bits and the thresholds 256 words of 16 bits. A decision therefore takes

$$
C = 2 + 4 L + \sum _ { l = 1 } ^ { L } p _ { w , l } p _ { h , l } s _ { l } c _ { l } k _ { w , l } k _ { h , l }\tag{1}
$$

clock cycles, for L layers with $p _ { w , l } \times p _ { h , l }$ output positions, $s _ { l } = 4$ for a pooled layer and 1 otherwise, c<sub>l</sub> output channels and a $k _ { w , l } \times k _ { h , i }$ <sub>l</sub> kernel. The 2 + 4L cycles start the engine, drain the pipeline after each layer and finish. For the deployed network $C = 2 { , } 2 6 9 { , } 4 7 7 ,$ , or 34.04 ms at 66.67 MHz, and the count depends on the layer table alone, not on the image. Simulation confirms it.

(a) Host-centred: every frame reaches the host  
![](images/06539a6e8e9cb62cd1e7d2f50f792791a8d53ff38f74cc3dacc1af9d13ee4096.jpg)  
waits for the whole frame; busy for every decision

(b) HDPU at the sensor: only the decision leaves  
![](images/9ff2e7941a3a88ec9d5534ea101537df3e5cc3628ac4044b1cc41e0c5bca9c07.jpg)  
reduces the frame as it arrives; takes the decision; no host time

Fig. 1. Two ways to get a decision from a camera. (a) The host-centred routine moves every frame to the host, which fetches instructions and weights and runs the network. (b) The inference HDPU reduces the frame as it arrives and takes the decision at the sensor; the host receives 16 bytes and does no work.  
![](images/12b8a59b7922f57556b1b98f5dc295dd528b55c41332fb930378b60c3a1d9431.jpg)  
Fig. 2. The detection path of the inference HDPU (blue) listens to the camera pins in parallel with the picture path (grey) and needs no processor. The network’s weights and thresholds (green) sit in on-chip memory beside the engine. The host, if present, receives a decision of 16 bytes per frame.

(a) OV7670 frame  
(b) 80x60, gray 58-148  
(c) after auto-contrast  
![](images/8ae7cfed13b7b78c3636264231c384b820c97453eaae53ff3107e54cde22aa09.jpg)  
Fig. 3. A frame captured from the board’s OV7670 (a), its $8 0 \times 6 0$ network input, which spans only a narrow range of gray levels (b), and the same input after the hardware auto-contrast stage (c).

## D. The decision

The margin m is the score minus the threshold of 477. It is averaged over the last 4 frames, and the averaged margin gives the unit’s state: crossing if $m \geq 3 3 .$ , no crossing if $m < - 3 1$ -9 and not sure in between. If no result arrives for one second the state becomes waiting. The band was chosen on validation photographs only. It is what lets the unit say “not sure” instead of guessing, and Section V-D shows how much that is worth.

TABLE I  
LAYERS OF THE DEPLOYED NETWORK AND THE CLOCK CYCLES EACH TAKES IN THE ENGINE (EQ. (1))
<table><tr><td>Layer</td><td>Operation</td><td>Output</td><td>Clocks</td><td>Share</td></tr><tr><td>L0</td><td>conv  $3 \times 3 ,$  8-bit input</td><td> $4 0 \times 3 0 \times 3 2$ </td><td>1,382,400</td><td>60.9 %</td></tr><tr><td>L1</td><td>conv  $3 \times 3 ,$  XNOR</td><td> $2 0 \times 1 5 \times 6 4$ </td><td>691,200</td><td>30.5 %</td></tr><tr><td>L2</td><td>conv  $3 \times 3 ,$  XNOR</td><td> $1 0 \times 7 \times 6 4$ </td><td>161,280</td><td>7.1 %</td></tr><tr><td>L3</td><td>conv  $3 \times 3 ,$  XNOR</td><td> $5 \times 3 \times 6 4$ </td><td>34,560</td><td>1.5 %</td></tr><tr><td>L4</td><td>fully connected</td><td>score</td><td>15</td><td>0.0 %</td></tr><tr><td colspan="2">Total, with 22 control clocks</td><td colspan="2">2,269,477</td><td></td></tr></table>

## IV. VERIFICATION AND METHODOLOGY

Bit-exact chain. A Python integer model computes exactly what the hardware computes, including the pixel offset, the padding values, the thresholds and the pooling. Every level of the design is compared against it (Table II). In simulation the engine matches the model on all 639 test images, in the score, in the decision and in every word of every intermediate feature map. On the board, each test image is sent over the serial link into a dedicated frame bank and classified by the FPGA, and its score and decision are compared with the model’s. The self-test also checks 20 golden vectors.

Data. The data set contains 11,281 images, 7,552 of them with a crossing. From 14 sources, 7,097 images are used for training, 1,905 for validation and 2,164 for testing. T1 is the set of 524 dashcam frames from CDSet [24] in the test split, and T2 contains 115 street photographs by photographers outside the training sources. The network was trained with knowledge distillation from a full-precision ResNet-18 [25], [26].

TABLE II  
VERIFICATION CHAIN: EVERY LEVEL COMPARED WITH THE PYTHON INTEGER MODEL
<table><tr><td>Level</td><td>Compared with the model</td><td>Result</td></tr><tr><td>Simulation</td><td>camera pins to the 80 × 60 image</td><td>3/3 frames</td></tr><tr><td>Simulation</td><td>engine: score, decision, every feature-map word</td><td>639/639 images</td></tr><tr><td>Simulation</td><td>serial link, contrast, engine, decision (real OV7670 frame)</td><td>20/20 checks</td></tr><tr><td>Board</td><td>self-test, golden vectors over the link</td><td>8/8 checks</td></tr><tr><td>Board</td><td>T1, score and decision</td><td>524/524 images</td></tr><tr><td>Board</td><td>T2, score and decision</td><td>115/115 images</td></tr></table>

Host. The host is the Intel Core i5-12500H laptop processor of [1], [27]. It ran the network as the PyTorch model used in training, with batch size one under native Windows, after 20 warm-up runs, taking the median of 200 runs, with one thread and with 16 threads. A full-precision ResNet-18 ran under the same conditions. These runs were not controlled as in [1] (power plan, pinning).

Power. The only power figures are Vivado’s vectorless estimates of the routed design, which the tool itself rates as low confidence. No power was measured at the supply, so we compare no energies across platforms.

Labels. Numbers are measured on the board, from simulation, from post-route reports (implementation), from the model, derived from these, or estimated. The text says which whenever it is not obvious.

## V. RESULTS

## A. Headline results

Table III collects the main results of the unit. The sections that follow give the evidence.

TABLE III  
HEADLINE RESULTS OF THE INFERENCE HDPU
<table><tr><td>Result</td><td>Value</td><td>Source</td></tr><tr><td>Decision time, every frame, any image</td><td>34.04 ms (2,269,477 clocks) 0</td><td>simulation</td></tr><tr><td>Frames skipped on the live camera</td><td></td><td>hardware</td></tr><tr><td>Score and decision equal to the model</td><td>639 of 639 images</td><td>hardware</td></tr><tr><td>Accuracy, T1 / T2</td><td>79.39 / 79.13 %</td><td>hardware</td></tr><tr><td>Correct when answering, T1 / T2</td><td>90.2 / 95.7 %</td><td>hardware</td></tr><tr><td>Camera-like frames, T1: without → with contrast stage</td><td>47.7 % → 78.2 %</td><td>model</td></tr><tr><td>Logic, detection path / whole system</td><td>1,814 / 3,370 LUTs</td><td>implementation</td></tr><tr><td>Host time spared per frame (one core)</td><td>8.13 ms, 21.7 % of a frame</td><td>host</td></tr><tr><td>Data to the host per frame</td><td>16 B, 38,400× less</td><td>derived</td></tr></table>

## B. On the board

Table IV summarises the board results. The unit evaluated every frame of the live camera: its on-chip counter registered 27 decisions in a one-second window, with no frame skipped. It returned exactly the scores and decisions of its software model on all 639 test images, so the hardware reproduces the accuracy measured in software.

TABLE IV  
MEASURED ON THE EDGE ARTIX-7 BOARD
<table><tr><td>Check</td><td>Result</td></tr><tr><td>Self-test</td><td>8 of 8 checks pass</td></tr><tr><td>Camera frames committed</td><td>27.5 frames/s, none rejected</td></tr><tr><td>Decisions on the live camera</td><td>27/s, no frame skipped</td></tr><tr><td>T1, dashcam frames, through the FPGA</td><td>524/524 equal to the model, 79.39 %</td></tr><tr><td>T2, street photos, through the FPGA</td><td>115/115 equal to the model, 79.13 %</td></tr></table>

## C. A fixed decision time inside the frame budget

Every decision takes 2,269,477 clock cycles, 34.04 ms, whatever the image (Eq. (1); simulation). Nothing in the unit depends on the image content, on other tasks or on history, so the decision time is a property of the design. It is also shorter than the camera’s frame period of 37.49 ms, which leaves 3.44 ms to spare in every frame (Table V). The engine is busy 90.8 % of the time, never falls behind the camera, and starts its next decision as soon as the next frame is committed (Fig. 4). Counting from the end of a frame, including the contrast stage, the decision follows after about 34.2 ms (derived).

TABLE V  
FRAME BUDGET OF THE INFERENCE HDPU
<table><tr><td>Quantity</td><td>Value</td><td>Source</td></tr><tr><td>Camera frame period</td><td>37.49 ms</td><td>model</td></tr><tr><td>Decision time</td><td>34.04 ms</td><td>simulation</td></tr><tr><td>Margin per frame</td><td>3.44 ms</td><td>derived</td></tr><tr><td>Frame end to decision</td><td>about 34.2 ms</td><td>derived</td></tr><tr><td>Engine busy</td><td>90.8 % of the frame</td><td>derived</td></tr><tr><td>Highest decision rate</td><td>29.4/s (camera: 26.68/s)</td><td>derived</td></tr><tr><td>Frames skipped on the board</td><td>0</td><td>hardware</td></tr></table>

## D. Accuracy and answers

Table VI gives the decisions of the unit on both test sets, identical to the model’s. On the dashcam frames it is conservative (precision 93.8 %, recall 67.5 %); on the street photographs it is balanced (precision 81.2 %, recall 83.6 %).

TABLE VI  
DECISIONS OF THE UNIT ON THE HELD-OUT TEST SETS
<table><tr><td></td><td>T1 (dashcam)</td><td>T2 (street)</td></tr><tr><td>Images (with a crossing)</td><td>524 (292)</td><td>115 (67)</td></tr><tr><td>True / false detections</td><td>197  /  13</td><td>56 /  13</td></tr><tr><td>Missed / true rejections</td><td>95 / 219</td><td>11 / 35</td></tr><tr><td>Precision / recall (%)</td><td>93.8 / 67.5</td><td>81.2 / 83.6</td></tr><tr><td>F1  / accuracy (%)</td><td>78.5 / 79.4</td><td>82.4 / 79.1</td></tr><tr><td>Answered outside the band (%)</td><td>50.8</td><td>40.0</td></tr><tr><td>Correct among answered (%)</td><td>90.2</td><td>95.7</td></tr></table>

The not-sure band is what turns an average classifier into a reliable one. Fig. 5 shows why: most of the overlap between the two classes falls inside the band, so withholding an answer there removes most of the errors. When the unit does answer, which it does for 50.8 % of the images of T1 and 40.0 % of T2, 90.2 % and 95.7 % of its answers are correct. Ranking every image by confidence extends this to any coverage (Fig. 6,

![](images/fb5b9f0adcc257bff142cf26d032101542d3e061c5dbcc77fcab03f6c6358897.jpg)  
time (ms); host rows show the same task run on one laptop core (PyTorch, median)  
Fig. 4. Three consecutive frame periods to scale. The HDPU engine (blue) finishes each decision inside the next frame period, never missing a frame. The two host rows show how long the same task takes on one laptop core (median, PyTorch, Section IV), not a measured sequence: the binarized network takes 8.13 ms, and the full-precision ResNet-18 takes longer than a frame period.

Table VII): on street photographs the most confident 30 % of the images are 97.1 % correct, and on dashcam frames the most confident 40 % are 93.8 % correct. A full-precision ResNet-18 on the same $8 0 \times 6 0$ input reaches 95.7 % on T2, so binarization, and not the input, limits accuracy.

![](images/54290d234476b84e6212d435f6e4487a44a66319abb4d29115fd5eb4abccea0d.jpg)

![](images/b5a9c00247502108565425aedfb09175fcd1a34503275cb2ac6ad85aa0d09f3a.jpg)  
Fig. 5. Margin (score minus the output threshold of 477) of every test image, from the bit-exact model; the board returned the same scores for all 639 images. Crossings above and non-crossings below the axis; grey: the not-sure band.

TABLE VII  
ACCURACY OF THE MOST CONFIDENT ANSWERS THROUGH THE CAMERA PATH (MODEL)
<table><tr><td>Share answered (%)</td><td>30</td><td>40</td><td>50</td><td>60</td><td>80</td><td>100</td></tr><tr><td>T1 (dashcam), correct (%)</td><td>92.4</td><td>93.8</td><td>91.6</td><td>90.4</td><td>83.5</td><td>77.5</td></tr><tr><td>T2 (street), correct (%)</td><td>97.1</td><td>95.7</td><td>93.1</td><td>88.4</td><td>81.5</td><td>73.0</td></tr></table>

## E. The real camera and the contrast stage

The sensor matters as much as the network. A frame captured from the board’s camera spans only a narrow range of gray levels (Fig. 3), a range the network never saw in training. On dashcam frames degraded to the camera’s contrast, accuracy falls to 47.7 % (T1) and 67.0 % (T2), and the hardware contrast stage restores 78.2 % and 78.3 %. It does so across the board (Fig. 7): averaged over nine camera effects (contrast, brightness, gamma, blur, noise and a lamp in view), accuracy rises from 70.7 % to 76.9 %, and the trimmed histogram keeps the gain when a lamp is in view, where a plain min–max stretch fails. The stage costs a small loss on images that already use the full range, which is why photographs sent from a PC bypass it, and it takes an estimated 0.15 ms.

![](images/17d69187c640e1dc88c9f1bdb164790e8bb3eee1ff1570de64501ae0324051ac.jpg)  
Fig. 6. Share of correct answers against the share of images answered, most confident first, through the camera path (model). The dashed line is the fullprecision ResNet-18 on all of T2.

## F. Cost

Table VIII and Fig. 8 give the cost. The whole detection path, from pre-processing to the serial link, uses 1,814 LUTs, and the inference engine itself 744 LUTs and 10 block RAMs, with no DSP slice. The complete system, including the camera interface, the picture path and the displays, uses 3,370 LUTs (16.2 % of the device), 19 of its 50 block RAMs and 1 DSP slice. All clocks meet timing after routing, the engine’s clock with 2.616 ns of setup slack at 66.67 MHz.

Vivado estimates the complete design at 0.294 W, 0.221 W dynamic and 0.073 W static. The engine accounts for only 22 mW of it, or 0.75 mJ per decision (derived from the estimate); clock generation (109 mW) is the largest single part. Since no power was measured at the supply, we compare no energies across platforms.

![](images/096dca67d65e3826d8d61da41e36d2ae7b7f56fd25fad00d5fd3c9c70a28e59f.jpg)  
Fig. 7. Accuracy under nine camera effects, mean of T1 and T2 (model), with the network input as the camera gives it and with the hardware contrast stage.

TABLE VIII  
RESOURCES ON THE XC7A35T-1 (VIVADO 2024.2, ROUTED). <sup>∗</sup>FROM THE HIERARCHICAL REPORT; LUT SHARING MAKES THE SUM OF THE ROWS SLIGHTLY LARGER THAN THE TOTAL.
<table><tr><td>Block</td><td>LUT</td><td>FF</td><td>BRAM36</td><td>DSP</td></tr><tr><td>Inference engine (BCNN core)</td><td>744</td><td>321</td><td>10</td><td>0</td></tr><tr><td>Pre-processing (gray, 8 × 8 means)</td><td>211</td><td>257</td><td>0</td><td>0</td></tr><tr><td>Auto-contrast, loader, decision</td><td>615</td><td>593</td><td>0</td><td>1</td></tr><tr><td>Frame banks, serial link</td><td>244</td><td>132</td><td>7</td><td>0</td></tr><tr><td>Camera, SDRAM, VGA picture path</td><td>1,069</td><td>769</td><td>1.5</td><td>0</td></tr><tr><td>Overlay, LCD, buzzer, clocks</td><td>490</td><td>658</td><td>0.5</td><td>0</td></tr><tr><td>Complete system*</td><td>3,370</td><td>2,730</td><td>19</td><td>1</td></tr></table>

## G. What the host is spared

The decision of the unit is work that the host never does. Running the same binarized network as the PyTorch model used in training takes 8.13 ms of one host core in every frame period, 21.7 % of it, and 27.9 % with 16 threads (Table IX). A full-precision ResNet-18, which reaches 95.7 % on T2, needs more than a frame period: 56.39 ms on one thread and 42.49 ms on all 16, against 37.49 ms (derived from medians). An inference HDPU removes all of this from the host, and with it the transfer of the frame.

The saving in data is just as large. To decide from a raw frame, the host must first receive 614,400 bytes. The unit’s decision is 16 bytes, 38,400 times less, and even the image the network saw together with its decision (4,826 bytes) is 127 times smaller than the frame (Table X, Fig. 9).

TABLE IX  
HOST TIME FOR THE SAME INFERENCE (PYTORCH, BATCH 1, MEDIAN OF 200 RUNS; UNCONTROLLED), AGAINST THE INFERENCE HDPU
<table><tr><td>Host task per frame</td><td>1 thread</td><td>16 threads</td><td>Share of frame, 1 thread</td></tr><tr><td>Binarized network</td><td>8.13 ms</td><td>10.47 ms</td><td>21.7 %</td></tr><tr><td>ResNet-18, full precision</td><td>56.39 ms</td><td>42.49 ms</td><td>150%</td></tr><tr><td>Inference HDPU</td><td></td><td>no host time</td><td>0%</td></tr></table>

TABLE X

MESSAGES FROM THE UNIT TO THE HOST, PER FRAME (DERIVED FROM THE SERIAL PROTOCOL)
<table><tr><td>Message to the host</td><td>Bytes Smaller than a frame by</td></tr><tr><td>Raw RGB565 frame</td><td></td></tr><tr><td>614,400 Snapshot: the 80 × 60 image the network saw, with its decision 4,826</td><td>1× 127×</td></tr><tr><td></td><td>14,629×</td></tr><tr><td>Decision (state, score, margin, average, counters)</td><td>38,400×</td></tr></table>

## H. Positioning

Table XI sets the unit beside published zebra-crossing detectors and a binarized-network accelerator. Each reports on its own data and task, so the accuracies cannot be ranked. What differs is what each needs around it. The published detectors run on a phone or a GPU module with an operating system, and FINN’s prototype on a Zynq SoC board, whereas this unit needs only its board and the camera. It has no processor and no operating system, it decides on every frame, and it leaves the host free.

## VI. DISCUSSION

From pre-processing to decisions. The units of [1], [2] turned the pixel stream into edges, corners and a steering code. This unit turns it into a decision. They all share one organisation: a fixed datapath at the sensor, a time fixed by design, and a compact result for the host. HDPUs do not compete for a bus or a scheduler, so an inference unit can in principle sit beside the Canny, Harris and line units on one camera stream, and the host would then receive edges, corners, a steering code and a decision, instead of frames.

What the result shows. The unit gives a neural decision the same properties as the pre-processing units before it: it decides at a time that does not depend on the image, it needs no processor, and it takes its work off the host. The contrast stage shows that these properties extend to the sensor side of the decision too. A learned decision is only as good as the signal it receives, and the cheap hardware stage in front of the network recovers most of what the real sensor costs.

Limitations. The power figures are Vivado estimates; no power was measured at the supply. The frame period is the model value of the camera’s register settings, and the board’s counters agree with it to within their one-frame resolution. The decision time is exact by construction and in simulation, but was not timed on the board with counters. The host runs were not controlled, and used PyTorch. T2 has 115 images, so its 95 % confidence interval is about ±7 points, and the livecamera tests are qualitative. The paper makes no claim about inference speed on images already stored in memory. Board power, on-chip latency counters, a controlled host baseline and an inference unit running beside the pre-processing units of [1] on one camera stream are the next steps.

(a) logic by block, routed  
![](images/98bdaf31372659b0d0a28ba3e549ae25e38a3fbb64622ee6b8ab320b2d2ff4ac.jpg)  
LUTs (system 3,370; detection path 1,814)

(b) estimated power by block  
![](images/317b3a3aa33682341c4b29960511d3260f4ee7f47548d2fa4147ecb6a61e485d.jpg)  
Fig. 8. (a) LUTs by block of the routed design. (b) Vivado’s vectorless power estimate by block; the inference engine (blue) is a small part of the total.

TABLE XI  
POSITIONING AGAINST PUBLISHED SYSTEMS, AS REPORTED BY THEIR AUTHORS. ACCURACIES REFER TO EACH SYSTEM’S OWN DATA AND TASK. <sup>†</sup>REPORTED FOR THE CLASSIFICATION OF PEDESTRIAN TRAFFIC LIGHTS.
<table><tr><td>System</td><td>Platform</td><td>What it needs around it</td><td>Rate</td><td>Reported result</td></tr><tr><td>ZebraRecognizer [8]</td><td>iPhone 5S</td><td>phone, operating system</td><td>about 25 fps</td><td>precision 1.0, recall 0.93</td></tr><tr><td>LYTNet [9]</td><td>iPhone 7</td><td>phone, operating system</td><td>20 fps</td><td>94%†</td></tr><tr><td>CDNet [10]</td><td>Jetson Nano</td><td>GPU module, operating system</td><td>33.1 fps</td><td>F1 above 94 %</td></tr><tr><td>FINN CNV [14]</td><td>ZC706 (Zynq-7045)</td><td>Zynq SoC board</td><td>21,906 images/s</td><td>80.1 % (CIFAR-10)</td></tr><tr><td>This work</td><td>XC7A35T and a camera</td><td>nothing else: no processor, no OS</td><td>26.68 fps (every frame)</td><td>79.1 % (T2); 95.7 % when answering</td></tr></table>

![](images/284f32ceaacb9cf3569366d44c789e3aa3437effe12b42199d107b767300ecfc.jpg)  
Fig. 9. Bytes per frame that the host would receive, on a log scale.

## VII. CONCLUSION

An inference HDPU takes a machine-learning decision at the sensor with no processor in the loop. On a low-cost Artix-7 FPGA it decides on every camera frame in a fixed 2,269,477 clock cycles, matches its software model on all 639 test images on the board, and uses 1,814 LUTs from pixels to decision. Its hardware contrast stage lifts accuracy on camera-like frames from 47.7 % to 78.2 %, and its not-sure band makes 95.7 % of its street-photo answers correct. The host runs none of the inference and receives 16 bytes per frame instead of a frame.

Together with the pre-processing units before it [1], [2], this shows that hardwired distributed processing units at the sensor can take on both the preparation and the decisions of a vision system, and leave the host free for the work that needs it.

## REFERENCES

[1] R. Mali, “Hardwired distributed processing units at the sensor: Truly parallel vision pre-processing that frees the host CPU,” arXiv:2610.38752 [cs.DC], 2026. [Online]. Available: https://arxiv.org/ abs/2610.38752

[2] ——, “Fast path, exact path: One library of hardwired operators for streaming and exact Canny edge detection on a low-cost FPGA,” arXiv:2610.33107 [cs.AR], 2026. [Online]. Available: https: //arxiv.org/abs/2610.33107

[3] N. P. Jouppi, C. Young, N. Patil, D. Patterson et al., “In-datacenter performance analysis of a tensor processing unit,” in Proc. International Symposium on Computer Architecture (ISCA), 2017, pp. 1–12.

[4] M. Horowitz, “Computing’s energy problem (and what we can do about it),” in Proc. IEEE International Solid-State Circuits Conference (ISSCC), 2014, pp. 10–14.

[5] R. Hameed, W. Qadeer, M. Wachs, O. Azizi, A. Solomatnikov, B. C. Lee, S. Richardson, C. Kozyrakis, and M. Horowitz, “Understanding sources of inefficiency in general-purpose chips,” in Proc. International Symposium on Computer Architecture (ISCA), 2010.

[6] J.-L. Aufranc, “Taalas HC1 hardwired Llama-3.1 8B AI accelerator delivers up to 17,000 tokens/s,” CNX Software, https://www.cnx-software.com/2026/02/22/ taalas-hc1-hardwired-llama-3-1-8b-ai-accelerator-delivers-up-to-17000- Feb. 2026, vendor-reported throughput; company announcement at https://taalas.com/the-path-to-ubiquitous-ai/.

[7] D. Riley, “AMD acquires Taalas to hardwire AI models into silicon,” SiliconANGLE, https://siliconangle.com/2026/08/06/ amd-acquires-taalas-hardwire-ai-models-silicon/, Aug. 2026.

[8] S. Mascetti, D. Ahmetovic, A. Gerino, and C. Bernareggi, “ZebraRecognizer: Pedestrian crossing recognition for people with visual impairment or blindness,” Pattern Recognition, vol. 60, pp. 405–419, 2016.

[9] S. Yu, H. Lee, and J. Kim, “LYTNet: A convolutional neural network for real-time pedestrian traffic lights and zebra crossing recognition for the visually impaired,” arXiv:1907.09706, 2019.

[10] Z.-D. Zhang, M.-L. Tan, Z.-C. Lan, H.-C. Liu, L. Pei, and W.-X. Yu, “CDNet: A real-time and robust crosswalk detection network on Jetson Nano based on YOLOv5,” Neural Computing and Applications, vol. 34, no. 13, pp. 10 719–10 730, 2022.

[11] M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio, “Binarized neural networks: Training deep neural networks with weights and activations constrained to +1 or −1,” arXiv:1602.02830, 2016.

[12] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, “XNOR-Net: ImageNet classification using binary convolutional neural networks,” in Proc. European Conference on Computer Vision (ECCV), ser. Lecture Notes in Computer Science, 2016, pp. 525–542.

[13] F. Zhou and Y. Chai, “Near-sensor and in-sensor computing,” Nature Electronics, vol. 3, no. 11, pp. 664–671, 2020.

[14] Y. Umuroglu, N. J. Fraser, G. Gambardella, M. Blott, P. Leong, M. Jahre, and K. Vissers, “FINN: A framework for fast, scalable binarized neural network inference,” in Proc. ACM/SIGDA International Symposium on Field-Programmable Gate Arrays (FPGA), 2017, pp. 65–74.

[15] M. Blott, T. B. Preußer, N. J. Fraser, G. Gambardella, K. O’Brien, Y. Umuroglu, M. Leeser, and K. Vissers, “FINN-R: An end-to-end deeplearning framework for fast exploration of quantized neural networks,” ACM Transactions on Reconfigurable Technology and Systems, vol. 11, no. 3, pp. 1–23, 2018.

[16] J. Duarte, S. Han, P. Harris, S. Jindariani, E. Kreinar, B. Kreis, J. Ngadiuba, M. Pierini, R. Rivera, N. Tran, and Z. Wu, “Fast inference of deep neural networks in FPGAs for particle physics,” Journal of Instrumentation, vol. 13, no. 07, p. P07027, 2018.

[17] Y. Umuroglu, Y. Akhauri, N. J. Fraser, and M. Blott, “LogicNets: Codesigned neural networks and circuits for extreme-throughput applications,” in Proc. International Conference on Field-Programmable Logic and Applications (FPL), 2020, pp. 291–297.

[18] E. Wang, J. J. Davis, P. Y. K. Cheung, and G. A. Constantinides, “LUTNet: Rethinking inference in FPGA soft logic,” in Proc. IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2019, pp. 26–34.

[19] D. Abts, J. Ross, J. Sparling, M. Wong-VanHaren, M. Baker et al., “Think fast: A tensor streaming processor (TSP) for accelerating deep learning workloads,” in Proc. International Symposium on Computer Architecture (ISCA), 2020, pp. 145–158.

[20] M. S. B. Altaf and D. A. Wood, “LogCA: A high-level performance model for hardware accelerators,” in Proc. International Symposium on Computer Architecture (ISCA), 2017.

[21] V. W. Lee, C. Kim, J. Chhugani, M. Deisher, D. Kim, A. D. Nguyen, N. Satish, M. Smelyanskiy, S. Chennupaty, P. Hammarlund, R. Singhal, and P. Dubey, “Debunking the 100X GPU vs. CPU myth: An evaluation of throughput computing on CPU and GPU,” in Proc. International Symposium on Computer Architecture (ISCA), 2010, pp. 451–460.

[22] R. Mali, “Zebra crossing detection with a binarized CNN on an Artix-7 FPGA: Project report,” Final-year project report, Vidyalankar Institute of Technology, Mumbai, Oct. 2026.

[23] OmniVision, “OV7670/OV7171 CMOS VGA (640x480) CameraChip sensor,” Preliminary datasheet, version 1.4, 2006.

[24] Z.-D. Zhang, M.-L. Tan, Z.-C. Lan, H.-C. Liu, L. Pei, and W.-X. Yu, “CDSet: Crosswalk DataSet for zebra crossing detection,” Zenodo, 2022.

[25] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015.

[26] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[27] Intel, “Intel Core i5-12500H processor product specifications,” https://www.intel.com/content/www/us/en/products/sku/96141/ intel-core-i512500h-processor-18m-cache-up-to-4-50-ghz/ specifications.html, accessed Oct. 2026.