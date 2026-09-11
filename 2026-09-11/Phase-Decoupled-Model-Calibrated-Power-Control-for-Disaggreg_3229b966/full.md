# Phase-Decoupled, Model-Calibrated Power Control for Disaggregated LLM Serving

Jae Gon Kim, Donghoon Yoo, Hanyul Ryu, Sungho Ha, Juyeon Lee, Soojung Ryu

Xenoscube, Inc.

https://xenoscube.ai/

September 10, 2026

## Abstract

Datacenter GPU power is the binding constraint on LLM serving capacity, and production serving has shifted to prefill/decode (PD) disaggregation. Deploying NVIDIA’s Max-Q inference profile in-flight on a disaggregated B200 system, we found its realized gain modest (+8.6% tokens/J), model-dependent, and carrying a mean end-to-end latency cost (+5.2%) that throughput-only evaluation does not surface; the profile also applies one setting to prefill and decode GPUs that operate in opposite hardware regimes. We hypothesize that the optimal power setting is a property of the deployed (model, quantization, engine, hardware) combination rather than of the GPU class, that each lane warrants its own profile, and that converting SLO headroom into energy safely requires latency-gated calibration under a runtime SLO guard rather than a fixed recipe. We present a phase-decoupled, model-calibrated controller: the prefill lane runs under an SM-clock window whose floor is a latency guarantee by construction, and the decode lane under a power cap placed by automatic calibration just above a measured throughput/latency clif. Because a disaggregated decode lane draws flat, memory-bound power, the cap binds continuously, the reactive-overshoot weakness that led POLCA to reject capping is absent, and the GPU’s own power manager retains throughput under the cap. On an 8× B200 node serving Qwen3-Coder-480B (FP8) under agentic load, our balanced mode delivers +20.4% tokens/J at +3.5% mean e2e versus +8.6% at +5.2% for Max-Q, a Pareto improvement on both axes. On Qwen3-235B-A22B (NVFP4) every operating mode meets the ITL-p99 SLO in every repetition, where both vendor profiles miss it in some. A decode-actuator A/B shows the calibrated cap beats both throughput-optimal and aggressive static clock locks (+24.1% vs +17.2% vs +15.5% tokens/J), and a three-day sustained run saves 32.3% of a lane pair’s electricity. Both models are Mixture-of-Experts (MoE); an exploratory dense-model comparison recovers roughly 5× less, so we scope our claims to MoE serving.

## 1 Introduction

Power, not silicon, is the marginal resource of LLM serving: GPU clusters saturate datacenter power envelopes before they exhaust rack space, so every watt shaved from serving becomes deployable capacity [2, 3]. Three recipe families dominate practice.

1. Vendor static profiles. NVIDIA’s workload power profiles (“Max-Q Inference”) apply a fixed, per-GPU-class multi-knob recipe (TGP limit, Fmax cap, DVFS curve, memory clock); for LLM inference the reported gains are 9-12% datacenter power savings at 2-3% performance loss [1]. The recipe is phase-blind: the same setting governs a GPU doing compute-bound prefill and one doing memory-bound decode, and it is validated per GPU class, not per model or serving stack.

2. Cluster power-management frameworks. POLCA [2] characterized LLM power on A100s and concluded that frequency locking is the reliable reclaim mechanism, explicitly rejecting power capping as reactive and unpredictable. TAPAS [9] manages thermal and power at cluster scale through placement and routing.

3. Phase-aware DVFS controllers (2025-26). VoltanaLLM [6], DualScale [7], GreenLLM [13], throttLL’eM [14] and DynamoLLM [8] recognize the prefill/decode asymmetry and control energy per phase. Without exception their actuator is frequency selection, and a 2026 characterization study argues that decode power capping is an illusion that clock locking dominates [15].

In parallel, production stacks (Splitwise [3], DistServe [4], Mooncake [5], NVIDIA Dynamo) disaggregate prefill and decode onto separate GPU pools because the two phases have opposite bottlenecks: prefill is compute-bound and power-hungry; decode is memory-bandwidth-bound and draws flat, stable power well below TDP. Disaggregation turns the transient phase distinction that prior power work had to chase in time into a static spatial property of a GPU lane. The phase-aware DVFS line exploits this for when to actuate, but inherited what to actuate (clocks) from the POLCA-era consensus without re-examining it under disaggregation.

Motivating observation. Deploying recipe (1) on our disaggregated B200 system yielded +8.6% tokens/J at +5.2% mean end-to-end latency on a 480B agentic workload (§5.1): a genuine but modest improvement, with a latency cost that the profile’s throughput-oriented validation does not surface. At the same time, even at full target concurrency, the deployment served every request with visible headroom against its latency SLO: the GPUs ran as fast as the silicon allows to finish work the service contract never asked to be finished that fast. Can that headroom be converted into energy rather than burned as unrequested speed (§5.2), and can a controller that derives its settings from the deployed stack convert more of it than one fixed, conservative profile (§5.1)? None of the three recipe families derives its settings from the specific combination being served, so none can locate where the convertible headroom lies. We hypothesize:

• H1 (per-combination profile). The optimal power setting is a property of the deployed (model, quantization, engine stack, hardware) combination, not of the GPU class. It must therefore be measured per combination, which is practical only if calibration is automatic.

• H2 (per-phase profile). Under PD-disaggregation, a distinct profile for the prefill lane and the decode lane recovers energy that any single phase-blind setting necessarily forgoes, since no single operating point suits two opposing hardware regimes.

• H3 (SLO-gated control). Converting the headroom safely requires setpoints accepted against the tail SLO and a runtime guard that enforces it when conditions shift. A static class profile can do neither, so it must either leave margin it cannot spend or violate the tail when conditions it cannot react to arrive.

Our experiments support H1 and H2 directly, and H3 in part (§5): our calibrated operating modes achieve 1.5-3.4× the vendor profile’s eficiency gain, and every latency-gated mode holds the tail SLO in every repetition where both static vendor profiles miss it in a fraction of runs (§5.1). We are explicit about what the data does not isolate: the runtime guard never had cause to fire in any recorded run, so the compliance contrast is attributable to latency-gated calibration rather than to runtime intervention, and a direct test of the guard remains outstanding (§6). The design principle throughout is that each element of a power-control policy (the mechanism, its setpoint, and the metric used to validate it) should be derived from measurements of the deployed stack; at several points conclusions drawn from intuition or from throughput-only evaluation were contradicted by measurement, including our own initial actuator assignment (§3.1).

A finding beyond the hypotheses. Testing H2 showed that the two lanes benefit from diferent mechanisms, not merely diferent settings. On a disaggregated decode lane, a calibrated power cap outperforms static clock locking, the actuator POLCA recommended: calibrated below the lane’s natural draw, the cap binds continuously (contrary to the characterization in [15], such a cap is not inert), and the GPU’s built-in power manager apportions frequency under it to preserve throughput, whereas a low lock collapses throughput (-31% to -33% in our A/B) and a high lock saves little (§5.3). Prefill, being compute-bound, is the lane where clocks belong: power tracks SM clock tightly, and a calibrated clock window whose floor is chosen from measurement gives a latency guarantee by construction. Both setpoints must be calibrated per (model, quantization, engine stack): the decode clif and the natural draw move with engine configuration (CUDAgraph settings, KV-transfer backend, engine version) as well as with model and quantization, and a roughl one-hour automated pass tracks this while recording the engine fingerprint as a first-class artifact. Our A/B compares the cap against static locks; a comparison against adaptive per-iteration DVFS controllers [6, 7] is outside this paper’s evidence (§6).

Contributions.

1. System (§3). A phase-decoupled actuation design with heterogeneous mechanisms per lane (prefill: DVFS window ladder; decode: calibrated power cap), fully automatic per-(model, quantization, engine) calibration with a latency-gated acceptance criterion, and an operating-mode ladder (PERF/BAL/EFF) that exposes the eficiency/latency trade as an operator-selectable, SLO-guarded knob. Phase-aware energy control for disaggregated serving is not new [6, 7]; choosing the mechanism per lane from measurement, calibrating it per serving stack, and gating it on tail latency are.

2. Measurement (§5). The first head-to-head, in-flight comparison against the shipped NVIDIA Max-Q profiles on B200 disaggregated serving, across two MoE models (480B FP8, 235B NVFP4) and two workload mixes, extended along the load axis (nine concurrency points) and the time axis (three 24-hour runs). Our modes Pareto-dominate Max-Q on the agentic workload and contain its operating point inside our ladder on the standard workload.

3. Methodology (§4, §5.4). Evidence that throughput-only metrics hide end-to-end latency damage; a mean-e2e (Little’s law) + ITL-p99 evaluation lens; and a repeated-calibration study showing that clif estimation on decode’s flat throughput-power curve is noise-fragile and that throughput-only acceptance admits SLO-violating setpoints.

Scope. We control steady-state serving eficiency on a single node’s lanes, on MoE models. The mechanism we exploit, decode-lane power slack, is largest where routing sparsity leaves the lane under-saturated; Appendix A shows the same cap-based mechanism recovers roughly 5× less eficiency on a dense model of comparable scale, so we make no dense-model claims. POLCA’s cluster-level oversubscription goal is complementary (§6). Training is out of scope: disaggregation’s spatial phase separation does not exist there, and Zeus [17], Perseus [18] and EnvPipe [19] already cover training energy.

## 2 Background and Motivation

## 2.1 PD-disaggregated serving

An LLM request has two phases with opposite hardware signatures. Prefill processes the whole prompt in parallel: high arithmetic intensity, tensor cores saturated, power spiking toward TDP. Decode emits tokens one at a time: dominated by weight and KV-cache reads, memory-bandwidth-bound, with GPU power flat and well under TDP. Serving stacks disaggregate the phases onto separate GPU pools connected by a KV-cache transfer path (Figure 1, top) to protect decode’s inter-token latency (ITL) from prefill interference and to scale the pools independently [3, 4, 5].

Disaggregation has an under-exploited corollary for power management: each GPU’s bottleneck regime is now known statically. A decode-pool GPU remains memory-bound for the entirety of its service assignment, so the phase-identification problem that dominated prior phase-aware power work reduces to lane identity.

## 2.2 GPU power knobs, and what prior work concluded

GPUs expose two in-band knobs: frequency locking (pin the SM clock) and power capping (the on-device power manager throttles clocks whenever draw would exceed the cap). POLCA [2], characterizing A100-era colocated serving, observed that capping is reactive, since prompt-phase spikes can overshoot the cap before throttling engages, while frequency locking reclaims power predictably. It built its policy on clock caps while conceding their cost: “frequency locking incurs performance impact throughout the execution, and not just when the power utilization is high.” Under disaggregation both premises flip. The overshoot scenario requires prefill spikes, and a decode lane has none: capping a decode lane is steady-state control of a flat signal. And the lock’s constant tax is no longer acceptable, because on a memory-bound lane the throughput-vs-power curve is flat near the top and clif-like below; the eficient operating point sits just above a clif that a static clock cannot find, and pinning clocks discards the power manager’s ability to reallocate budget between SM and memory dynamically.

An analytical lens on the mechanism split. The standard CMOS model puts GPU power at

$$
P \approx P _ { \mathrm { s t a t i c } } + C V ^ { 2 } f \sim A + B f ^ { 1 + \alpha } , \alpha > 0\tag{1}
$$

since voltage co-scales with frequency. Writing execution time as $T \sim f ^ { - \beta }$ , with $\beta \in [ 0 , 1 ]$ the workload’s arithmetic-intensity coeficient (β=1 fully compute-bound, $\beta { = } 0$ fully memory-bound), energy per unit of work becomes

$$
E \ = \ P T \ \sim \ A f ^ { - \beta } + B f ^ { 1 + \alpha - \beta }\tag{2}
$$

VoltanaLLM derives this same model to motivate per-phase frequency selection [6]; we push it one step further, to the choice of mechanism. For the prefill lane $\beta \approx 1$ : time responds nearly proportionally to clock, so a calibrated clock floor is a latency guarantee by construction, and power tracks $f ^ { 1 + \alpha }$ , which makes the clock window the natural eficiency knob. For the decode lane $\beta \ll 1 \colon$ lowering frequency buys power at a sub-linear time cost, but only down to the bandwidth-saturation point below which the pipeline collapses (the clif of §3.3) [6, 7]. Moreover $\beta$ rises with batch occupancy, so the energy-optimal frequency moves with load, and a static lock must sit conservatively high or risk crossing the moving optimum. The phase-aware DVFS line answers by predicting the optimum in software per iteration; a calibrated cap answers by fixing the power budget and delegating the frequency decision to the GPU’s own power manager, which re-solves the allocation continuously at microsecond scale with no model to mistrain. Our $\mathrm { A } / \mathrm { B }$ (§5.3) tests the cap against the static-lock answer; the software-predicted answer is not in our evidence.

NVIDIA’s workload power profiles [1] take a simpler position: one validated multi-knob recipe per GPU class, applied uniformly. This is deployable in minutes and is the natural production baseline, which is why we benchmark against the real in-flight profiles (§4). NVIDIA’s own evaluation reports that multi-knob profiles including a TGP power limit outperform pure frequency scaling by 7-9% in performance at equal power savings on B200 [1], vendor-reported evidence of the mechanism asymmetry we isolate per phase.

## 2.3 Why calibration must be automatic and engine-aware

The decode clif (the cap below which throughput and latency collapse) and the natural power (the uncapped draw that anchors the prefill window) are functions of model, quantization, and the serving engine’s configuration. Across engine configurations of the same model on the same GPUs we measured natural decode power move by roughly 3%, and observed ITL shifts across engine versions of the same stack. A 3% shift is comparable to the run-to-run dispersion of our short-run measurements (§5.1), so we take it as motivation for fingerprinting rather than as a result in itself; the operational point is that a profile validated on one stack has no mechanism to notice that the next stack difers. Our calibration pass (§3.3) therefore records the engine fingerprint (CUDA-graph on/of, KV-transfer backend, engine version) alongside the setpoints, and any fingerprint change triggers recalibration.

## 3 Design

Figure 1 shows the system. The serving plane is untouched, stock NVIDIA Dynamo + sglang-runtime PD-disaggregated serving. Our control plane attaches per-lane actuators driven by calibration artifacts, an operating-mode selector, and an SLO guard.

Problem statement. A PD-disaggregated deployment exposes two lanes $\ell \in \{ P , D \}$ with independent actuators: a prefill SM-clock window $W = [ f _ { \mathrm { l o } } , f _ { \mathrm { h i } } ]$ and a decode power cap c. Given the service’s tail SLO Λ (an ITL-p99 bound, with prompt latency tracked as a secondary guard), the controller solves

$$
\operatorname* { m a x } _ { ( c , W ) } ~ { \mathrm { ~ t o k e n s / J ~ \pi ~ s . t . ~ } } ~ \Pi \mathrm { T L \mathrm { - } p 9 9 } \leq \Lambda ~ { \mathrm { a t ~ t h e ~ d e p l o y m e n t ' s ~ o p e r a t i n g ~ l o a d } }\tag{3}
$$

with $( c , W )$ restricted to the hardware’s actuation grid and drawn from a calibration artifact A indexed by (model, quantization, engine, hardware). H1 says A must be measured per combination; H2 says it assigns distinct settings, indeed distinct mechanisms, to the two lanes; H3 says the constraint binds at the tail, so feasibility must be established against tail latency and enforced at runtime. The components are per-lane actuation (§3.1), an operating-mode ladder exposing solutions at three risk postures (§3.2), and the calibration pass that populates A (§3.3).

![](images/02ecd8e56aec4f731a110006d8689489ac745bc1e6c08f52db80f6b4bec2393d.jpg)  
Figure 1: Phase-decoupled control architecture. The serving plane is an unmodified PD-disaggregated stack; the control plane assigns each lane its own actuator (SM-clock window for prefill, calibrated power cap for decode), parameterized by a per-(model, quant, engine) calibration store, an operating-mode ladder, and an ITL-p99 SLO guard.

## 3.1 Phase-decoupled actuation

Decode lane: power cap. The actuator is a single per-GPU power cap chosen from the calibrated clif (§3.3). The cap bounds peak draw while delegating the SM/memory frequency split to the GPU’s own power manager, which performs microsecond-scale DVFS under the cap. This adaptivity is the root cause of capping’s win over static clock locks in our A/B (§5.3): the cap achieves power reduction and throughput retention simultaneously, which no single locked frequency can.

Prefill lane: SM-clock window ladder. Prefill power tracks clock nearly proportionally, so we control clocks directly, but as a window rather than a point lock. Each operating mode confines the clock to a calibrated band spanning roughly 20-30% of the achievable clock range (boundaries are per-combination calibration outputs; the most eficient mode’s floor sits near half of the maximum clock). The window floor is the calibrated clock below which prompt-latency SLOs would be at risk, so selecting a mode selects a latency guarantee by construction. Within the window, a lightweight runtime controller adjusts the operating clock with lane-level load, its actuation rate bounded so as not to stress the driver stack. The in-window policy belongs to the commercial implementation behind this section’s contracts (see Reproducibility); the guarantees evaluated in §5 derive from the window bounds, and all results are gathered with the controller active. Because the controller was active in every arm, our measurements do not separate the floor’s contribution from the in-window policy’s (§5.4, §6).

On the assignment of mechanisms to lanes. The symmetric intuition (“cap the compute-bound phase, DVFS the memory-bound phase”) was our own initial design, and measurement reversed it (§5.3). Compute-bound prefill responds to clocks predictably enough that a window gives a guarantee; memory-bound decode responds to clocks so nonlinearly (flat-then-clif) that only the cap’s self-DVFS adaptivity extracts the eficiency without collapsing throughput. Both prior recipes (POLCA’s all-frequency, Max-Q’s uniform profile) encode the un-measured intuition. Appendix A documents the selection campaign behind this choice, spanning four base models, three quantizations, dense/MoE/hybrid architectures, three workload shapes, and both colocated and disaggregated topologies.

## 3.2 Operating-mode ladder

The controller’s objective is fixed: maximize energy eficiency subject to the latency SLO. Latency enters as a constraint every setpoint must satisfy, never as a quantity traded in the open. Operators pick one of three modes per lane pair, each mapping to (decode cap, prefill window) pairs from the calibration artifact; the modes difer only in how much of the calibrated headroom they spend:

• PERF (Performance) places the decode cap at the lane’s natural draw and keeps the prefill window in its upper range. Steady-state behavior is essentially unthrottled; PERF buys a deterministic peak bound (the cap clips excursions above natural draw) plus a small eficiency gain, at near-baseline latency.

• BAL (Balanced) sets both lanes to intermediate calibrated setpoints. It is the default recommendation: a majority of the EFF-level gain at a substantially smaller latency cost (§5.2).

• EFF (Eficiency) applies the deepest SLO-admissible setpoints the calibration found: the decode cap just above the calibrated clif and the prefill window floor at the latency guarantee. It suits workloads with latency headroom (the agentic mix of §5.1).

All three modes come from one calibration artifact, so the ladder is monotone by construction and every rung is subject to the same runtime SLO guard, which monitors tail latency per lane and, on a violation, releases the ofending lane’s setpoint toward the hardware default until compliance is restored (Algorithm 1, lines 7-9). This is the axis Max-Q lacks: its single profile is one operating point, which our measurements place inside the ladder (between PERF and BAL on the standard workload). Algorithm 1 states the runtime loop at contract level; the loop’s structure is what a reimplementation would need to reproduce our SLO-compliance results. Every result in §5 was gathered with the guard armed, except the Figure 7 capture, which states so; in no recorded run did the guard trip (§5.1, §5.7), so §5 exercises lines 1-6 and 10 of the loop and establishes lines 7-9 by contract rather than by observation.

Algorithm 1 Runtime mode execution with SLO guard (contract level)   
Require: calibration artifact A (§3.3); operating mode m ∈ {PERF, BAL, EFF}; tail SLO Λ (ITL-p99);   
per-lane telemetry stream   
Ensure: continuously enforced lane setpoints   
1: $( c _ { m } , W _ { m } ) \gets \mathcal { A } [ m ]$ ▷ decode cap, prefill clock window   
2: apply power cap $c _ { m }$ to every decode-lane GPU   
3: confine prefill-lane clocks to $W _ { m }$ ▷ floor of $W _ { m }$ = latency guarantee   
4: loop at each control interval   
5: for all lanes $\ell \in \{ P , D \}$ do   
6: $\lambda _ { \ell } \gets$ observed tail latency ▷ ITL-p99 (decode), TTFT (prefill)   
7: if $\lambda _ { \ell }$ violates Λ then   
8: release lane ${ \boldsymbol { \ell } } { \boldsymbol { \mathrm { s } } }$ setpoint toward the hardware default   
9: hold until compliance is restored, then re-apply $A [ m ]$   
10: end if   
11: end for   
12: within $W _ { m } { : }$ the runtime controller moves the prefill clock with lane load   
(rate-bounded; policy replaceable — the $\ S 5$ guarantees derive from the window bounds, lines 1–9)   
13: end loop

Algorithm 2 Calibration acceptance gate (contract level)   
Require: candidate setpoint u (a decode cap or a prefill window floor); tail SLO Λ; tolerance ε; serving-parity   
load L (closed loop, same $N$ as service, mixed prefill+decode trafic)   
Ensure: accept/reject; on accept, an entry in artifact A   
1: $T _ { \mathrm { r e f } } $ reference throughput at hardware defaults under L ▷ multi-repetition, noise-robust statistic; a   
single-run $T _ { \mathrm { r e f } }$ is unsafe on decode’s flat curve (§5.4)   
2: run the live stack at u under L   
3: accept u if throughput $( u ) \geq ( 1 - \varepsilon ) T _ { \mathrm { r e f } }$ and $\mathrm { I T L - p 9 9 } ( u ) \leq \Lambda$ ▷ the latency-aware criterion; omitting it   
admits SLO-violating setpoints (§5.4)   
4: on accept: record u in A together with the engine-stack fingerprint (CUDA-graph configuration, KV  
transfer backend, engine version)   
5: any fingerprint change invalidates A and triggers recalibration

## 3.3 Automatic calibration

For each (model, quantization, engine) combination, an automated pass runs ofline against the live serving stack and produces three outputs: (i) the natural (uncapped) power of each lane, which anchors all operating windows; (ii) decode power-cap setpoints per mode, each accepted only if it preserves both throughput and the ITL-p99 SLO under serving-parity load; and (iii) prefill clock-window boundaries per mode. The pass completes in about an hour without operator involvement and records the engine-stack fingerprint alongside the setpoints. The setpoint search is a replaceable component: any search whose outputs pass the acceptance gate of Algorithm 2 yields valid setpoints; §5.4 characterizes our procedure’s output stability, and every setpoint used in the evaluation is reported in $\ S 5 .$

The relative setpoints reported in $\ S 5$ (decode caps at 80-90% of natural draw) are outputs of this calibration for the specific model, quantization, engine stack and PD configuration measured, not universal constants. They difer between our own two models (≈85% of natural on 480B TP4 PD versus 80-90% on 235B TP1 PD), and per H1 no fixed fraction should be expected to transfer.

Latency-aware acceptance is not optional. An earlier calibrator revision accepted setpoints on throughput alone; on 235B it selected an EFF cap near 65% of natural draw that preserved throughput within tolerance while pushing $\mathrm { I T L - p 9 9 }$ beyond the SLO in the large majority of repeated runs. Adding a tail-latency criterion moved the accepted setpoint to 80% of natural draw and restored 100% SLO compliance at +22.4% tokens $\cdot / \mathrm { J }$ (§5.2). Algorithm 2 fixes this acceptance gate, the part of calibration our correctness claims depend on and that a reader can port to their own calibrator.

![](images/d0809a997de8336605bf663812fa0d0478755d78d170d22a8a658e1ebf92a129.jpg)  
(b) What the artifact encodes: the mode ladder on each lane's actuation axis

![](images/b757ddf63a059e2bb591aee0539c9c03b7f49a2711ef066c23760fe530e0675a.jpg)  
positions are schematic; each marker is a measured, per-combination calibration output (§3.3), not a universal constan

Figure 2: Calibration contract and artifact semantics. (a) The setpoint search is replaceable; its contract is fixed: it runs against the live stack, accepts a setpoint only if throughput is preserved and ITL-p99 stays within the SLO, and emits an artifact that a fingerprint change invalidates. (b) What the artifact encodes on each lane’s actuation axis: on the decode power axis, the region below the calibrated clif is rejected by the acceptance gate and the three mode caps span the calibrated headroom between the clif (EFF) and the natural draw (PERF); on the prefill clock axis, each mode selects a window whose floor is the latency guarantee. Positions are schematic; every marker is a per-combination measurement, not a universal constant.

![](images/11b7991bdf577431c53f71aeef1f0817e90a01412dee3490f36975f60da62965.jpg)  
Figure 3: Measured decode cap-sweep from the calibration pass (235B NVFP4, TP1 PD, calibration load). Throughput (left axis) is flat from the natural draw down to the calibrated clif (≈430 W, the minimum-J/tok point) and collapses below it (-36% at 350 W); ITL p95 (right axis) stays well inside the SLO everywhere above the clif. The calibrated mode caps span the headroom: EFF 460 W just above the clif, BAL 547 W, PERF uncapped at the natural draw. This is the measured counterpart of Figure 2b’s schematic; the sweep reports ITL p95, and the acceptance gate additionally checks ITL-p99 under serving-parity load (Algorithm 2).

## 4 Experimental Setup

<table><tr><td>Item</td><td>Value</td></tr><tr><td>Node</td><td>8× NVIDIA B200 (single node)</td></tr><tr><td>Serving stack</td><td>NVIDIA Dynamo sglang-runtime:1.2.0, PD-disaggregated, cache-aware router</td></tr><tr><td>Models</td><td>Qwen3-Coder-480B FP8 (TP4 prefill GPUs 0-3 / TP4 decode GPUs 4-7); Qwen3-235B-A22B NVFP4 (TP1 PD, GPUs 0-1)</td></tr><tr><td>Load</td><td>Closed-loop at the router; concurrency 48 (480B) / 24 (235B)</td></tr><tr><td>Workloads</td><td>agentic (long-context, tool-call-shaped mix) and standard (chat-shaped mix)</td></tr><tr><td>Arms</td><td>BL (uncontrolled), NVIDIA Max-P (profile 0,6), Max-Q (profile 1,6), both real in-flight profiles rather than reconstructions, and our PERF/BAL/EFF operating modes</td></tr><tr><td>Campaigns</td><td>The head-to-head and ladder results (§5.1-§5.5) are single-load-point studies at the concurrency above. Two further campaigns extend the evaluation along one axis each and are reported with their own scope: the load axis (§5.6, 235B, nine concurrency points) and the time axis (§5.7, 235B, three 24-hour runs). Their absolute values are not</td></tr><tr><td>Energy metric</td><td>energy-accounting scope, each stated in place. tokens/J over all 8 GPUs (including idle/other lanes; the denominator is never restricted to actively controlled lanes)</td></tr><tr><td>Latency</td><td>mean end-to-end (Little's-law lens: at fixed closed-loop concurrency, mean-e2e α</td></tr><tr><td>metrics</td><td>1/completion rate), TTFT p95/p99, ITL p95/p99; every arm is judged against a fixed ITL-p99 service SLO</td></tr><tr><td>Repetitions</td><td>N = 3 per arm and workload cell in the head-to-head campaigns of §5.1 (both models); dispersion is reported as ±1σ across repetitions where N &gt; 2. The load-axis campaign of §5.6 uses N = 3 per cell and the time-axis campaign of §5.7 three full days</td></tr><tr><td>Data drop rule</td><td>Intermittent TTFT blowups (&gt;15 s; stack-level issue, arm-independent, a small fraction of repetitions) dropped pairwise so all arms see identical repetition sets</td></tr></table>

Workload composition and concurrency. Both workloads are closed-loop request streams issued against the router at fixed concurrency: N logical clients each hold exactly one request in flight and issue the next the moment the previous completes, so in-flight requests are pinned at N for the entire run. Both lanes therefore stay at a steady operating point, which makes the lane power profiles stationary enough to control and to measure; load is identical across arms by construction, since an arm that slows down receives requests more slowly and no arm is penalized by queue buildup; and with N fixed, Little’s law ties mean end-to-end latency to the request completion rate exactly (below). N is chosen per deployment to saturate it (48 on the 8-GPU TP4/TP4 configuration, 24 on the 2-GPU TP1 configuration), and calibration runs under the same construction, so setpoints are measured at the load they will serve. The agentic workload models tool-using agent trafic: long, context-accumulating prompts with comparatively long generations, stressing both large prompt ingests and sustained decode. The standard workload models interactive chat trafic and is derived from the public ShareGPT conversation corpus [16]: request content and input/output length distributions are sampled from ShareGPT conversations, giving shorter prompts and generations at higher request turnover and a prefill-heavier duty cycle per token served. The two bracket the phase balance a production mix moves between; §5.2 shows the appropriate operating mode difers between them, which is itself part of the argument for a mode ladder. The generator is seeded per run (--seed 1234 throughout), so a repetition replays the same request stream. The standard workload samples input length from {64, 128, 256, 512, 1024, 2048} tokens with weights {0.18, 0.24, 0.26, 0.18, 0.10, 0.04} and output length from {32, 64, 128, 256, 512, 1024} with weights {0.16, 0.22, 0.26, 0.20, 0.12, 0.04}. The agentic workload issues 4 tool-calling turns per client, base input 512 tokens and 128 generated tokens per turn, a tool result of 256 tokens appended to the context after each turn, and a think-gap drawn uniformly from 0.5-3 s between turns.

Benchmark provenance and arrival control. The standard workload’s requests come from ShareGPT [16], the source used by several serving-systems evaluations, so its content and length distributions are externally anchored. What we deliberately replace is the arrival process: both workloads are issued through the closed-loop harness rather than by replaying trace timestamps, because the paired methodology of §5 requires a stream that is exactly reproducible and load-controlled, and because tokens/J is only a meaningful property of an operating point if load is stationary over the integration window. Native-timestamp replay is open-loop and non-stationary, which breaks the fixed-N construction and injects arrival variance large enough to mask single-digit-percent efects in energy. The agentic workload is constructed: context accumulating across tool-calling turns dominates our target deployments but is not represented in any public trace Evaluating under native trace arrival processes is future work (§6).

Why mean-e2e rather than throughput alone. Throughput-only reporting conceals the latency cost of power control: an arm can maintain aggregate tokens/s while individual requests lengthen. At fixed closed-loop concurrency N, Little’s law ties mean end-to-end latency to the request completion rate µ exactly:

$$
\bar { t } _ { \mathrm { e 2 e } } ~ = ~ \frac { N } { \mu }\tag{4}
$$

Because every arm serves the identical request stream, per-request token counts match across arms and $\mu$ difers from token goodput only by a constant, so ∆mean-e2e is a workload-normalized measure of latency degradation with no free parameters; tail ITL (p99) additionally guards the streaming experience. We report both for every arm. Under this lens, settings that appear cost-free by throughput (including Max-Q’s reported results) exhibit a measurable latency cost (§5.1).

On the strength of the vendor baseline. Comparing against a reconstruction of the vendor profile rather than the profile itself understates the baseline. Running both in the same block on the same hardware, a uniform SM-clock lock, the natural reconstruction, saves 4.0-4.1% of baseline power, while the in-flight profile saves 12.5-12.9%: a factor of three. Every Max-Q comparison in this paper uses the shipped profile.

Quantization on the actuation grid. Caps and clocks are quantized (cap grid steps, discrete supported clocks); a policy computed of-grid silently snaps to the grid at actuation time. Our harness measures at

(a) Agentic workload: tool-using agent traffic

Schematic request anatomy (segment lengths not to scale). Both workloads are issued closed-loop at fixed concurrency against the router (§4).  
![](images/d6d611ea81887810bacec92970a655f86d86c2bc1e1ce0fc53a3ed0e96f16d41.jpg)  
long, growing prompt ingests · long sustained generations · low request turnover · per-token duty cycle leans to decode  
(b) Standard workload: interactive chat traffic (ShareGPT-derived)

![](images/85d4886612759ff21cb3d093d88f7e1061ebb3a56e0020ab0eb1783e68fadf2d.jpg)  
Figure 4: Request anatomy of the two workloads (schematic; segment lengths not to scale). The agentic workload accumulates context across tool-calling turns, so prefill ingests grow while generations remain long; the standard workload issues short independent requests at high turnover, with content and lengths drawn from ShareGPT [16]. The two bracket the prefill/decode balance a production mix moves between.

actuated values only.

Energy measurement. Per-GPU power is sampled in-band (NVML, with DCGM activity counters alongside) at a 100 ms cadence; energy is the time integral of measured power over the full run window, summed across all 8 GPUs of the node. Token counts are taken client-side from the streamed responses, so tokens/J divides externally observed work by internally measured energy and cannot be inflated by either side alone.

## 5 Evaluation

## 5.1 Head-to-head vs NVIDIA Max-Q

480B agentic (primary result). In-flight Max-Q yields +8.6% tokens/J at +5.2% mean-e2e. Our BAL mode yields +20.4% at +3.5% and EFF +29.3% at +4.9%: 2.4× and 3.4× the eficiency gain, each at lower latency than the vendor profile, so both are Pareto improvements on both axes. Repeated reconfirmation on the same stack under the standard workload gives EFF +23.6±3.7% and BAL +24.3±0.7% tokens/J, with every arm passing the ITL-p99 SLO in every repetition. The 480B agentic figures above are two-repetition means from a single session (N = 2 per arm), so no ±1σ is quoted for them; the reconfirmation values that follow carry their own dispersion.

235B (repeated paired study). With calibration frozen and paired repetitions per arm (anomalous repetitions dropped pairwise; Figure 5b):
<table><tr><td>Workload</td><td>Arm</td><td>∆tok/J</td><td>Mean power saved</td><td>∆mean-e2e</td><td>ITL-p99 SLO</td></tr><tr><td>agentic</td><td>Max-P</td><td>+0.7±6.5%</td><td>+0.3%</td><td>-0.1%</td><td>96%</td></tr><tr><td></td><td>Max-Q</td><td>+11.4±6.1%</td><td>+12.4%</td><td>+2.8%</td><td>96%</td></tr><tr><td rowspan="7">standard</td><td>PERF (ours)</td><td>+0.2±5.8%</td><td>+0.4%</td><td>+0.7%</td><td>100%</td></tr><tr><td>BAL (ours)</td><td>+15.8±5.5%</td><td>+15.8%</td><td>+3.0%</td><td>100%</td></tr><tr><td>EFF (ours)</td><td>+38.8±5.1%</td><td>+32.2%</td><td>+6.6%</td><td>100%</td></tr><tr><td>Max-P</td><td>-0.1±1.7%</td><td>-0.2%</td><td>-0.2%</td><td>100%</td></tr><tr><td>Max-Q</td><td>+11.8±2.0%</td><td>+12.7%</td><td>+2.8%</td><td>100%</td></tr><tr><td>PERF (ours)</td><td>+7.6±3.2%</td><td>+10.8%</td><td>+1.7%</td><td>100%</td></tr><tr><td>BAL (ours)</td><td> $\mathbf { + 1 7 . 6 { \pm 2 . 6 } \% }$ </td><td>+22.6%</td><td>+3.8%</td><td>100%</td></tr><tr><td></td><td>EFF (ours)</td><td> $+ 2 2 . 4 { \pm } 2 . 4 \%$ </td><td>+29.5%</td><td>+8.9%</td><td>100%</td></tr></table>

The three rows per workload marked (ours) are the operating-mode ladder of §3.2 in ladder order. Mean power saved is the reduction in mean draw over the same lanes the eficiency figure is computed on; baseline draw is 799 W on the agentic workload and 896 W on the standard workload. Max-P is statistically indistinguishable from the uncontrolled baseline on both workloads, and on the agentic workload so is our PERF mode.

Two readings of the table carry the argument. First, on the agentic workload EFF exceeds Max-Q by a margin far outside the ±1σ bands (+38.8±5.1% against +11.4±6.1%) at a +6.6% mean-e2e cost and a 5.8% TTFT p95 improvement; BAL’s advantage over Max-Q (+15.8±5.5% against +11.4±6.1%, at a matched latency cost of +3.0% versus +2.8%) lies within the overlap of the bands and we do not claim it as a separate result. On the standard workload, where dispersion is smaller, BAL exceeds Max-Q outside the bands (+17.6±2.6% against +11.8±2.0%) at a mean-e2e cost one point above the vendor profile’s, and EFF pushes to +22.4±2.4% at +8.9%. Second, the SLO column: both vendor profiles miss the tail SLO in a fraction of agentic repetitions (96% compliance), while every one of our modes holds it in every repetition on both workloads. We attribute the contrast to the acceptance gate of Algorithm 2, which admitted every setpoint only after it held ITL-p99 under serving-parity load, whereas a class-level profile is never tested against this deployment’s tail. The runtime guard (Algorithm 1, lines 7-9) did not trip in any of these runs, so the study shows what latency-gated calibration alone achieves, and H3’s guard clause is established by contract rather than by observation (§6).

## 5.2 The operating-mode ladder is monotone and SLO-guarded

The three (ours) rows of the §5.1 table are the operating-mode ladder. Figure 6 plots the standard-workload ladder (235B, production calibrator revision; baseline 1.20±0.03 tokens/J at 896 W).

On the standard workload the ladder is monotone on both axes, eficiency $+ 7 . 6 \%  + 1 7 . 6 \%  + 2 2 . 4 \%$ and mean power saved $+ 1 0 . 8 \%  + 2 2 . 6 \%  + 2 9 . 5 \%$ from PERF to EFF, and Max-Q’s operating point lies strictly inside it, between PERF and BAL on eficiency. On the agentic workload PERF is indistinguishable from baseline, so monotonicity there is established for the BAL → EFF step $( + 1 5 . 8 \% \to + 3 8 . 8 \% )$ . The mean-e2e increases down the ladder are the price of the additional eficiency, and the correct lens for them is the latency budget: every rung keeps $\mathrm { I T L - p 9 9 }$ within the SLO, so the ladder spends headroom the SLO already grants, and the operator chooses how much of it to convert. The agentic ladder rises far steeper because the agentic mix leaves more latency headroom to spend; the appropriate mode is workload-dependent (agentic → EFF, standard → BAL), and the existence of this dependence argues for a ladder over a single profile. Figure 7 shows the controller’s runtime behavior at the setpoint level under a load transient.

## 5.3 Decode actuator A/B: power cap beats static clock locking

<table><tr><td>Workload</td><td>Decode arm</td><td>∆tok/J</td><td>∆tok/s</td><td>∆decode power</td><td>ITL-p99 SLO</td></tr><tr><td rowspan="3">agentic</td><td>calibrated power cap (ours)</td><td>+24.1%</td><td>-8%</td><td>-16%</td><td>pass</td></tr><tr><td>clock lock, throughput-optimal</td><td>+17.2%</td><td>-9%</td><td>-9%</td><td>pass</td></tr><tr><td>clock lock, aggressive</td><td>+15.5%</td><td>-31%</td><td>-37%</td><td>pass</td></tr><tr><td rowspan="3">standard</td><td>calibrated power cap (ours)</td><td>+18.8%</td><td>-12%</td><td>-20%</td><td>pass</td></tr><tr><td>clock lock, throughput-optimal</td><td>+15.4%</td><td>-3%</td><td>-5%</td><td>pass</td></tr><tr><td>clock lock, aggressive</td><td>+8.8%</td><td>-33%</td><td>-37%</td><td>pass</td></tr></table>

The cap arm is the calibrated EFF cap of §3.3, placed below the lane’s natural draw at the latencygated clif. The two lock arms bracket the static-clock design space: the throughput-optimal lock is the highest-throughput static clock, the strongest case for POLCA-style locking, and the aggressive lock trades throughput for the deepest power reduction; the exact lock frequencies are sweep outputs for this stack and carry no meaning beyond these two roles. All arms share the calibrated prefill floor, so the ∆tok/J column includes the prefill contribution and the comparison between rows isolates the decode lever. The ranking cap > lock-optimal > lock-aggressive holds in both workloads. The aggressive lock saves power but collapses throughput (-31% to -33%); the throughput-optimal lock preserves throughput but barely cuts decode power (-5% to -9%), so its eficiency gain is mostly inherited from the shared prefill floor. The cap achieves both because the GPU self-DVFSes under it: peak power is bounded while clock allocation remains adaptive. A static frequency cannot occupy both ends of a flat-then-clif curve; a cap can, by construction.

What this does and does not refute. POLCA’s Insight 7 rejected capping as reactive and unpredictable, based on colocated A100 serving with prompt spikes, measured through a throughput lens. On a disaggregated decode lane there are no spikes to overshoot, the calibrated clif makes the cap’s performance efect deterministic, and POLCA’s chosen mechanism, the static clock lock, is the arm that loses the A/B; POLCA’s own caveat, that locks tax execution constantly, is the cost we avoid by confining clock control to the prefill lane. The A/B therefore refutes the frequency-first position for static locking on disaggregated decode lanes. It does not test adaptive per-iteration DVFS of the kind VoltanaLLM and DualScale implement [6, 7]; that comparison would need their controllers running on our stack under our SLO gate (§6). The characterization in [15], which reports clock locking dominating capping in decode, is reconciled by cap placement: a cap at or near TDP on a memory-bound lane never binds and yields no savings. Our caps bind by construction (≈85% of natural draw on 480B and 80% on 235B, at a clif jointly gated on throughput and ITL-p99), and the studies difer in kind, theirs an open-loop characterization across attention architectures, ours a closed-loop serving A/B with total-node energy accounting and SLO gates. The two are consistent measurements of diferent operating points, uncalibrated caps and calibrated ones, which is itself evidence that calibration, rather than mechanism choice alone, is the load-bearing component.

Binding fraction. The load-axis campaign (§5.6) measured how often the cap actually binds. Across the concurrency range the decode cap is engaged 94-99% of the time under the standard workload, and only 13-24% of the time under the agentic workload above concurrency 32, with the same controller and setpoint; the eficiency advantage tracks that diference. We report the association and stop there: the binding fraction is an outcome of the treatment and the realized load, not an independent variable. A probe underlines the caution: removing the think gaps from the agentic workload was expected to raise binding by removing idle time, but binding fell from 55% to 23.5% and the gain from +18.1% to +6.1%, because gap-free workers synchronize onto turn boundaries and the lane alternates between prefill and decode waves that keep decode draw below the cap. What sets binding is the phase-overlap structure, not the idle fraction.

## 5.4 Calibration robustness: the repeated-calibration study

Repeated calibrate-and-measure cycles on 480B exposed two failure classes that any deployable calibrator must fix, and that to our knowledge no prior power-management evaluation reports.

1. Setpoint instability on flat curves. Decode’s throughput-vs-cap curve is flat above the clif, so small noise in the reference against which candidates are judged translates into large shifts in the estimated clif. An early calibrator revision produced decode setpoints from ≈79% of natural draw up to efectively uncapped across runs (bimodal, most near ≈82% of natural with occasional collapses to no-cap; Figure 9). The production calibrator resolves this with the multi-repetition, noise-robust reference statistic that Algorithm 2 (line 1) requires, and its output is stable across the final ladder the ladder is monotone on both axes; the vendor profile's operating point fal s strictly inside it (between PERF and BAL)

![](images/ff8ae0b31c400a6ea4cf28c1e873abe5ca9eddd106d6d820c4a4f752f6fb60e3.jpg)  
Figure 5: Eficiency/latency plane, agentic workload on both models. (a) 480B: both BAL and EFF lie above and to the left of Max-Q — 2.4× and 3.4× its eficiency gain at lower mean-e2e — so each is a Pareto improvement on both axes. (b) 235B (paired repeated runs; error bars ±1σ): the result reproduces on a second model; BAL matches Max-Q’s latency cost while exceeding its gain, and EFF delivers 3.4× the gain. Every plotted arm meets the ITL-p99 SLO. The standard-workload ladder, where eficiency is bought with latency inside the SLO envelope, is reported in §5.2.

(a) Δ tokens/J vs baseline (replicated, ±1σ)  
(b) mean node power saved vs baseline  
![](images/3276f77b8f0694d5d26147927dfe77112cb33bc829d46734cbfda522284202c8.jpg)  
standard workload, 235B · every mode passes the ITL-p99 SLO in every repetition · Max-P and Max-Q shown as dashed references

Figure 6: The operating-mode ladder on the standard workload (235B). Eficiency gain (a) and mean node power saved (b) both increase monotonically down the ladder, every mode passes the ITL-p99 SLO, and the vendor profile’s operating point (dashed) falls strictly inside the ladder.

Runtime behavior under EFF, with a 60 s concurrency spike (480B FP8, TP4 PD, agentic)  
![](images/0cde3a37c69f437de43dc895f85c0caef48a3f1a20e0ed5a23c12ed2e21baf67.jpg)

![](images/a96fa593b2efa2baa568dc088149b37c53c6c7e94e055422177a30df5d4962d1.jpg)  
Figure 7: Runtime behavior under EFF (480B, agentic; 100 ms telemetry), with a 60 s load spike of +64 concurrency injected mid-run (shaded). Top: the prefill lane holds its in-window operating clock (1155 MHz) through the spike. Middle: decode-lane draw rides the commanded 500 W cap — the cap binds continuously, spike included. Bottom: rolling ITL p99 (5 s window) sits at 24–28 ms in steady state, is pushed across the 30 ms SLO line during the spike (peak 96 ms), and returns to baseline within seconds of the spike ending; the run’s aggregate ITL-p99 is 29.5 ms, inside the SLO. This capture isolates the steady-state contract — window held, cap binding — through a load transient, so the harness left the runtime guard of §3.2 out of the loop and the trace shows the crossing-and-recovery the guard is designed to bound. Guard behavior under load is the subject of §5.6, where the SLO sets the usable ceiling. Complements the run-level lane timeline of Figure A4.

![](images/a66cd2cca1f5dfff6a1e005f3b7a269cddffc73bf475d7d2e9558bd15a9f0968.jpg)  
Figure 8: Decode-lane actuator A/B on 480B (all arms share the same calibrated prefill eficiency floor; only the decode lever difers). Power cap wins on eficiency in both workloads while retaining throughput.

(§5.2). Estimating a crossing point on a flat curve is inherently noise-fragile, and reporting single-run clif estimates without a reproducibility study is unsafe.

2. Throughput-blind acceptance violates SLOs. The 235B EFF cap accepted on throughput alone (§3.3) passed throughput while violating the ITL-p99 SLO in the large majority of runs; the latencyaware criterion restored full compliance at +22.4% tokens/J. Any setpoint chosen by throughput alone, a calibrated cap, a locked clock, or a vendor profile, can hide a tail-latency violation.

Robustness of the primary results is otherwise good: paired-∆ statistics across the repeated runs are tight (standard Max-Q +11.8±2.0%, our BAL +17.6±2.6%), and the intermittent TTFT blowups are arm-independent stack behavior, dropped pairwise, with root cause tracked separately.

Where the power savings come from: per-lane attribution. The repeated study’s lane-level telemetry decomposes the node power saving by lane. At EFF on 480B, the prefill lane contributes the bulk of the reduction, lane power -43% (standard) and -47% (agentic), while the calibrated decode cap contributes -16% and -14% respectively, for node totals of -26% and -28%. The asymmetry is consistent with Equation (1): prefill power tracks f<sup>1+α</sup>, so a window floor purchases a large power reduction, whereas decode already draws flat and well below TDP, leaving the cap a thinner but latency-cheap margin. The two mechanisms also difer in output stability: the prefill window floor was identical in every one of the repeated calibrations, while the decode cap carried all of the run-to-run variance (failure class 1). Two attributions this telemetry cannot make must be stated. It attributes power, not eficiency: a factorial ablation (prefill-window-only / decode-cap-only / both, each against BL on both workloads, reporting the §4 metric triple) was not carried and remains outstanding. And because the in-window runtime controller was active in every arm (§3.1), the prefill-lane figure bundles the window floor with the in-window policy; a floor-only arm (window pinned at its floor, controller inactive) would separate the two.

## 5.5 Composability: substitutes, not complements

A 4-arm study (BL / Max-Q / ours / ours+Max-Q) on 480B indicated that once our controller governs the lanes, additionally enabling Max-Q yields no further eficiency: the vendor profile and our controller act on the same power headroom, and ours captures it at a more favorable eficiency-latency exchange. The

![](images/cab0a12c5beac73f4a50c3e809c0aa08d10603c8a5b17981250682f7da3ceade.jpg)  
Figure 9: Accepted EFF decode cap across 30 back-to-back independent calibrations of the same (model, quant, engine) combination, early calibrator revision (480B). The estimates cluster at 460–500 W (21/30) but scatter to 540–808 W and collapse to no-cap twice — the signature of estimating a crossing point on a flat curve from single-run reference measurements. The production revision’s noise-robust acceptance settles at 520 W on the same stack (dashed). Single-run clif estimates reported without a reproducibility study are unsafe (failure class 1).

mechanisms are therefore alternatives rather than complements, and the Pareto comparison of §5.1 is the decision criterion.

## 5.6 Load dependence: the advantage across the concurrency range

The results above are measured at one concurrency per workload. To characterize the load axis we swept nine concurrency points from 4 to 96 on 235B, with the engine’s maximum running requests raised from 16 to 64 so that client concurrency, not the engine queue, sets the ofered load. Arms are BL, the vendor Max-Q profile and EFF, paired block-by-block within a session, three repetitions per cell.

On standard trafic the advantage holds across the whole range and grows with load, from $+ 5 . 8 { \pm } 0 . 4 \%$ at concurrency 4 to +22.9±1.1% at 56; against the uncontrolled baseline the same sweep rises from +15.8% at concurrency 4 into roughly the +30% band from 32 upward. We do not name a peak: two clean sessions at concurrency 64 disagreed by 6.0 percentage points, and two further sessions established +30.6±1.8% as the representative value, which leaves neither “the gain peaks at 64” nor “it declines by 96” supportable. The session-to-session disagreement is also a caution about the ±1σ figures elsewhere in this paper, which are within-session dispersions.

On agentic trafic the advantage is real but thin, between +2.3% and +6.9% at every point, with no trend. The reason is in the load rather than the controller: completed requests on the agentic baseline are flat from concurrency 24 to 96 (595 → 612 → 596 per 180 s) while mean power falls (735 → 609 W), because the think gaps cap the arrival rate. Past roughly concurrency 24-32 the x axis measures waiting workers rather than demand, and the decode lane spends most of its time below the cap (§5.3).

Mode ordering is stable across load. In all twelve workload-concurrency cells the point estimates order as $\mathrm { E F F > B A L > P E R F \geq M a x { - } Q } .$ . The cells share sessions and baselines, so this is one consistent pattern rather than twelve confirmations, and we do not read diferences within ±1.3 percentage points, the largest ordering efect between arms, as a distinguished advantage.

The SLO sets the usable ceiling, and it is below the eficiency ceiling. EFF holds ITL-p99 at concurrency 48 (two sessions, 6/6 runs, 27.4-28.7 ms) and 56 (one session, 3/3, 28.1-28.6 ms), and breaks it at 64 (two sessions, 6/6, 30.2-32.8 ms) and 96, so the compliant ceiling falls in [56, 64) on this grid. Eficiency keeps climbing above it (concurrency 64 measured +36.7% against baseline in one session), but an operator bound by the tail SLO cannot spend that: the eficiency number alone would have recommended a setting

Efficiency advantage over the vendor Max-Q profile across the load axis (235B NVFP4, TP1 PD, EFF mode)

![](images/64564bab6db2bb25955d0005a6fe5fa72b84d728b6bd8d48d3717b80a9df7aa4.jpg)  
Figure 10: Eficiency advantage of EFF over the shipped Max-Q profile across nine concurrency points (235B NVFP4, TP1 PD; block-paired ∆ tokens/J, bars ±1σ). On the standard workload the advantage holds at every point and widens with load; on the agentic workload it is real but flat and thin. Above the shaded boundary EFF no longer holds the ITL-p99 SLO, so the widening advantage there is not adoptable under an SLO-bound frame.

the latency axis forbids. The bracket is conditional on this grid and on a healthy stack; from concurrency 48 upward we observed intermittent router-circuit episodes afecting all arms, including the baseline.

## 5.7 Sustained operation: what a day of running actually saves

Every number above is a ratio over a run of a few minutes. The operational question is how much electricity a day of serving costs, which requires establishing that short-run ratios survive a day of thermal and workload reality. We ran the three arms in parallel on one node, each on its own GPU pair (Max-Q, EFF, and the uncontrolled baseline), under the standard workload at concurrency 24, in 96 segments of 900 s per day, and repeated the whole day three times. Running the arms simultaneously removes time-of-day drift from the comparison by construction; the cost is that the three arms share the host’s CPU, PCIe fabric and cooling, and we did not isolate cross-arm coupling through those shared resources. Energy is integrated from 1 Hz telemetry over each arm’s two GPUs, so these absolute values are not comparable with the eight-GPU accounting used elsewhere.

<table><tr><td>Arm</td><td></td><td>Energy (kWh/day) Saved vs baseline</td><td>Saved vs baseline (%) ∆ tokens/kJ</td><td></td></tr><tr><td>Uncontrolled baseline</td><td> $2 2 . 8 7 \pm 0 . 0 4$ </td><td></td><td></td><td></td></tr><tr><td>Max-Q</td><td> $1 9 . 2 6 \pm 0 . 0 3$ </td><td> $3 . 6 2 \pm 0 . 0 0 \mathrm { k W h }$ </td><td> $1 5 . 8 \pm 0 . 2 \%$ </td><td> $+ 1 2 . 6 \pm 0 . 2 \%$ </td></tr><tr><td>EFF (ours)</td><td> ${ \bf 1 5 . 4 9 \pm 0 . 0 5 }$ </td><td> $\mathbf { 7 . 3 9 \pm 0 . 0 2 \ k W h }$ </td><td> $\mathbf { 3 2 . 3 \ : \pm { 0 . 2 \% } }$ </td><td> $\mathbf { + 1 9 . 5 \ : \pm { \ : 0 . 4 \% } }$ </td></tr></table>

Over three independent days the controller saved ${ \bf 7 . 3 9 \ \pm \ 0 . 0 2 k W h / d a y }$ , 32.3% of what the uncontrolled lane pair consumed, and $3 . 7 7 \pm 0 . 0 3 \mathrm { k W h / d a y }$ more than the vendor profile. The run-to-run standard deviation of $0 . 0 2 \ \mathrm { k W h / d a y }$ is the tightest reproducibility figure in this paper. Max-Q’s +12.6% tokens/kJ here is measured under this campaign’s two-GPU accounting and 900 s windows and is not the same quantity as its +11.8% in the §5.1 table.

Nothing drifted. Decode natural draw held at 688-689 W, the cap stayed engaged 95-97% of the time, the prefill lane held its in-window operating point at 1230 MHz, and decode temperature settled at 45-52 °C, all flat across 24 hours, three times over. Thermal equilibrium is reached within the first hour and the calibrated setpoints remain valid beyond it. The tail SLO held in all 864 segments, and the runtime guard never had cause to fire.

Short-run ratios extrapolate; short-run absolute energy does not. Projecting one hour to a full day over-predicted absolute energy in every arm of every run, by 1.8-10.1%, and the sign never flipped: the first hour is a cold machine drawing more for the same work. The diference between arms extrapolated within -6.6 to +3.0% without a systematic direction. An operator can carry a short-run $\Delta \%$ into a capacity estimate, but must measure to state kilowatt-hours.

Two operational notes. The shipped Max-Q profile is not a static cap: the enforced limit sits at 1000 W and moves intermittently within 567-994 W, which is part of why a static clock lock is not a substitute for it (§4). And at the end of the first day the command clearing the vendor profile silently did nothing (it requires the profile list as an argument) and the profile stayed resident; the between-run check caught it, the clear command was pinned and followed by a read-back, and the two subsequent days verified clean. Leaked power state is exactly the failure that silently invalidates the run that follows it.

## 6 Discussion

Scope vs POLCA. POLCA solves cluster-level power oversubscription (fast reclaim when a breaker threshold is breached); we solve steady-state lane eficiency, and the rebuttal in §5.3 is scoped accordingly. For the reclaim use case, a calibrated cap is a deterministic, pre-validated reclaim point, and B200-generation in-band capping latency is far below the 40 s out-of-band budget POLCA designed around; quantifying reclaim latency under our controller is future work.

Hardware and model generality. All results are single-node B200 on two Qwen3-family MoE models (480B FP8, 235B NVFP4) under one engine stack. The mechanism argument (flat memory-bound decode power; self-DVFS under caps) is not Blackwell-specific, but clif positions move across GPU generations, which is an argument for automatic calibration rather than against the design. The mechanism is architecture specific by construction: MoE’s per-token expert sparsity leaves the decode lane below its power ceiling, which is exactly the slack a calibrated cap reclaims. Appendix A found this margin roughly 5× smalle on a comparably sized dense model (+5.7±8.0% vs +26.0±6.4% cap-only tokens/J), because dense models saturate the tensor pipes and draw near the device ceiling. We make no dense-model claims; a controlled dense-model evaluation under the full controller is left to future work.

Datacenter implications. In a power-constrained facility, the tokens a fixed provisioned budget can deliver scale with energy eficiency, so the gains of §5 translate into capacity: +17.6-22.4% more SLO-compliant tokens per provisioned megawatt on the standard workload and +15.8-38.8% on the agentic workload, against +11.4-11.8% for the vendor profile on the same model. Mean node power reductions compound this (a -22.6% mean draw admits roughly 29% more serving instances under the same budget), and §5.7 puts an absolute figure on one lane pair: $7 . 3 9 \pm 0 . 0 2$ kWh saved per day against no control, 3.77 kWh against the vendor profile. The cap-based decode lane adds a property scheduling-based savings lack: the per-GPU peak is bounded deterministically by the commanded cap, the quantity provisioning and oversubscription planning actually consume.

Limitations. (i) A residual eficiency-latency trade remains on the standard workload (up to +8.9% mean-e2e at EFF); the ladder makes it explicit and selectable rather than hidden. (ii) The intermittent TTFT anomalies in our baseline (\~15% of repetitions) are a serving-stack artifact whose root cause remains open. (iii) The scope is a single node: cross-node power shifting and cluster-level scheduling (as in DynamoLLM [8]) are orthogonal layers that compose above per-lane control. (iv) Calibration time is the principal operational cost: a vendor profile applies in minutes, whereas our pass takes about an hour per (model, quantization, engine, PD configuration) and re-runs whenever the fingerprint changes; at serving scale it amortizes within hours, but class-level profiles do not carry it. (v) The load and time axes are characterized only on 235B; the 480B results are single-point, and the SLO-compliant ceiling is a bracket on one grid under a healthy stack. (vi) The decode A/B compares the cap against static clock locks only; adaptive per-iteration DVFS controllers [6, 7] were not run on our stack, so the claim that capping beats frequency control is established against static locking, not against those systems. (vii) The prefill-lane saving bundles the window floor with a proprietary in-window controller, and the factorial per-lane ablation that would attribute the eficiency gain by mechanism remains outstanding (§5.4). (viii) Both workloads use closed-loop arrivals; external validity under native trace arrival processes is untested.

![](images/d364df108aeff18fe33440d3f6a7438deb37408a3ecb989cab55d785b19f6ece.jpg)  
indexed capacity: at a fixed facility power budget, deliverable SLO-compliant tokens scale with tokens/J (§5.1, 235B, paired repeated runs)  
Figure 11: Facility-level translation of the eficiency results: SLO-compliant tokens deliverable per provisioned megawatt, indexed to baseline = 100 (235B, paired repeated runs). The vendor profile buys roughly 11 points of capacity; per-stack calibrated modes buy 22 to 39 depending on workload.

## 7 Related Work

Disaggregated serving substrate. Splitwise [3] established phase splitting (1.4× throughput at 20% lower cost) and characterized the phases’ distinct power profiles, observing that decode does not need the latest GPUs’ full capability: power diferentiation at provisioning time, which our runtime-calibrated, SLO-gated actuation generalizes. DistServe [4] and Mooncake [5] optimize goodput without power actuation. These systems are the substrate on which our work operates rather than points of comparison.

Vendor static profiles. NVIDIA’s workload power profiles [1] are the deployable baseline, and their own data shows multi-knob recipes including a power limit beating frequency-only scaling by 7-9%. We compare against the real in-flight profiles and show a calibrated, phase-decoupled ladder Pareto-dominates them on the agentic workload (§5.1).

Cloud LLM power management. POLCA [2] supplied the phase power characterization we build on and concluded that frequency capping is the reliable control; §5.3 shows this conclusion inverts for static locking on disaggregated decode lanes. TAPAS [9] manages thermal and power via placement, routing and reconfiguration, and RAPID [10] does power-aware prefill/decode role assignment; both are cluster-level scheduling levers orthogonal to and composable with per-lane actuation.

Phase-aware DVFS controllers. VoltanaLLM [6] is the closest system work: feedback-driven perphase frequency control plus state-space routing in P/D-disaggregated SGLang, up to 36.3% energy savings. DualScale [7] co-designs phase-aware placement with per-iteration DVFS (ILP re-provisioning every five minutes, then MPC for prefill and slack-aware stepping for decode; up to -39% prefill and -48% decode energy vs DistServe on 16× H100). Its own decomposition is instructive: the coarse tier, which chooses instance counts, parallelism, routing weights and a static baseline frequency per instance, captures most of the saving, while per-iteration DVFS adds between -4% and +20% on top. A well-chosen static setpoint is where the energy is, which is the premise of our calibration and mode ladder, and that static frequency is exactly the quantity we replace with a calibrated cap on the decode lane. DualScale also trains its models per (model, GPU) pair without an engine-stack fingerprint (§2.3). GreenLLM [13] fits latency-power models over SM frequency per phase for colocated serving; throttLL’eM [14] predictively throttles frequency phase-uniformly; DynamoLLM [8] reconfigures instance count, parallelism and frequency per load epoch from ofline profiles. All five actuate through frequency; none compares frequency against calibrated power caps on the decode lane, calibrates per engine stack, or evaluates against the in-flight vendor profile. Our work composes with them at the scheduling layer and difers on the actuation mechanism; a direct comparison against their adaptive DVFS is the open experiment of §6.

Power capping for inference. PALS [11] treats NVML power caps plus batch size as joint knobs for MoE serving in vLLM (up to 26.3% eficiency, 4-7× fewer QoS violations); ICPP’25 work [12] argues caps should be a first-class inference-runtime primitive. Both operate on monolithic serving without phase decoupling. Conversely, a 2026 characterization [15] argues decode power capping is an illusion that clock locking dominates; §5.3 reconciles this with our opposite finding through cap placement.

Training energy. Zeus [17] tunes power limits and batch size for training; Perseus [18] removes energy bloat along pipeline critical paths, the training-side analogue of phase decoupling; EnvPipe [19] exploits pipeline bubbles. These motivate our exclusion of training (time-synchronized phases, no spatial lanes) rather than competing with it.

Positioning. No prior system (i) selects the actuation mechanism per disaggregated lane from measurement, (ii) derives cap and clock setpoints from an automatic per-(model, quantization, engine-stack) calibration with a latency-gated clif and an engine fingerprint, or (iii) evaluates power control directly against the shipped vendor profile on the mean-e2e + tail-ITL plane. Phase-awareness alone ceased to be novel in 2025; per-lane mechanism selection, stack-tracking calibration, and latency-inclusive evaluation are the contributions of this paper.

## 8 Conclusion

This work began from the gap between the reported and the realized eficiency of a shipped vendor power profile on disaggregated LLM serving. Evaluated in-flight against NVIDIA Max-Q on B200 hardware, a phase-decoupled, per-stack-calibrated operating-mode ladder achieves 1.5-3.4× the vendor profile’s eficiency gain, Pareto-dominating it on agentic workloads, and every latency-gated operating point holds the tail SLO in every repetition where the static vendor profiles miss it in a fraction of runs. The measurements also extended our hypotheses in one respect we had not anticipated: the two lanes require diferent mechanisms, not merely diferent settings. On decode lanes a calibrated power cap outperforms the static frequency locking recommended by prior work, because disaggregation converts the phase-management problem from a temporal one into a spatial one and thereby restores the viability of an actuator the literature had set aside. The broader lesson is methodological: power-control conclusions derived from intuition or validated solely through throughput (the vendor’s, prior frameworks’, and initially our own) were not sustained once an end-to-end latency axis was applied.

## Future Work

Four experiments would close the gaps §6 identifies: running an adaptive per-phase DVFS controller [6] on our stack under our SLO gate as a fourth arm of the decode $\mathrm { A } / \mathrm { B } ;$ a fault-injection study that forces the runtime guard to trip and measures its recovery time and cost; the factorial per-lane ablation with a floor-only prefill arm; and evaluation under native trace arrival processes. Beyond these, a training-side comparison is deferred (training lacks the spatial phase separation our lane actuators exploit, phase transitions outrun VRM slew of \~12.5 ms, and Zeus/Perseus/EnvPipe already serve that regime), cross-node reclaim with calibrated caps (POLCA’s oversubscription scenario) is planned, and we are working on the approach’s principal cost, the calibration pass: shortening its wall time and developing online calibration that refines setpoints from live serving telemetry under the same SLO guard.

## Reproducibility

Everything needed to reproduce every measured configuration is disclosed: the setpoints of every arm (§5, relative to the lane’s natural draw and maximum clock, the two anchors any deployment can measure locally), the calibration acceptance contract (Algorithm 2), the runtime-guard contract (Algorithm 1), the source data behind the measured figures, and the measurement protocol (pairwise drops, actuation-grid quantization, serving-parity load). Raw per-repetition JSON and the calibration artifacts used in each experiment, including their engine-stack fingerprints, are retained and available on request.

Two components are described only at the level of their behavioral contract: the setpoint-search heuristic and the in-window clock policy, both part of our production system’s control logic. Any search whose outputs pass the acceptance gate of Algorithm 2 produces valid setpoints, and §5.4 characterizes the output stability of ours. The in-window policy is a diferent case: the guarantees of §5 derive from the window bounds, but because the policy was active in every measured arm, its contribution to the prefill-lane saving is not separated from the floor’s (§5.4, §6), and a floor-only reproduction is the experiment that would separate them.

## A The actuator-selection campaign

The per-lane design of §3.1 (decode: power cap; prefill: clock window) and the choice of PD-disaggregated serving as the target emerged from an exploratory campaign, conducted before the main evaluation, across four base models (a 70B dense, a 405B dense, a 480B MoE with 35B active parameters, and a 120B hybrid Mamba-Transformer MoE with 12B active), three quantizations (BF16, FP8, NVFP4), three workload shapes (prefill-heavy, decode-heavy, mixed), and both colocated and PD-disaggregated topologies. This appendix summarizes the evidence for three questions: why target disaggregated serving (A.2), why the power cap and why MoE models benefit most (A.3), and what phase-decoupled actuation does at run granularity (A.4). Consistent with §3.3, all setpoints are reported relative to the lane’s natural draw $( P _ { \mathrm { n a t } } )$ or the maximum SM clock.

## A.1 Phase signatures are model-specific but topology-invariant

Using per-GPU activity counters (tensor-pipe utilization vs DRAM utilization, sampled at 100 ms), prefill and decode form disjoint clusters for every model tested, and a threshold classifier derived from these counters labels held-out samples with 100% accuracy. The separation gap, however, varies by two orders of magnitude (Figure A1), and the same 70B model measured colocated and PD-disaggregated produced statistically indistinguishable signatures (prefill counter-ratio 4.51 vs 4.48; decode 0.032 in both).

Two consequences carried into the main design: phase behavior is a property of the (model, quantization) pair rather than of the deployment topology, so a per-model characterization transfers across topologies; and under disaggregation a lane’s phase is static, which eliminates runtime phase detection entirely (§2.1). The same counters also quantify how far each architecture sits from compute saturation, which A.3 connects to the size of the reclaimable gain.

## A.2 Why disaggregated serving is the target

In colocated serving one GPU carries both execution regimes, so it holds exactly one power setting, and that single setting faces a zero-sum trade between the two panels of Figure A2. A setting sized for decode-phase eficiency wins panel (b) and loses panel (a): it is near-lossless within the decode phase, but the moment a prefill burst arrives on the same GPU, tail latency collapses (TTFT p99 +1005% on a colocated MoE deployment). A setting relaxed enough to protect prefill wins panel (a) and loses panel (b): latency stays bounded, but roughly a third of the achievable decode-phase gain is forgone (+18-20% versus +28-31% on the same deployment). Disaggregation removes the trade itself: each lane holds exactly one regime, statically (A.1), so each lane holds its own mechanism and setpoint, and the rightmost column of Figure A2 is the only one that is best in both panels simultaneously. The main-body results confirm this; on the agentic workload, +38.8% tokens/J arrives together with a 5.8% TTFT p95 improvement and full ITL-p99 SLO compliance (§5.1).

![](images/e340163a977bad91a07531f60a8af6b29d4163b27e1051de5e56a35bb4fc3609.jpg)  
Figure A1: Prefill/decode activity signatures per model (tensor-pipe : DRAM counter ratio, log scale). Every model separates cleanly, but the separation gap spans 2.7× to 141×, so classification thresholds cannot be hard-coded; the signature is invariant to serving topology.

A second reason to target disaggregation directly: mechanism verdicts measured on colocated serving do not transfer. The combined cap-plus-clock-lock arm that gained +44% tokens/J on the colocated 70B model retained only +4% when the same model was redeployed PD-disaggregated. Any conclusion about actuation therefore had to be re-established on the disaggregated target itself, which is what §5.3 does.

## A.3 Why the power cap, and why MoE benefits most

The mechanistic asymmetry between the two in-band actuators is the foundation of the §3.1 design. A locked clock stretches every instruction on the critical path, whether that path is expert-dispatch communication (MoE) or long prefill compute (large dense models); a power cap is only an upper bound, under which the GPU schedules its own clocks, so most of the energy saving arrives at near-zero latency cost. In the per-configuration sweep on the colocated 480B MoE model, the cap-only arm was Pareto-dominant outright: +30.8% tokens/J with no inter-token-latency cost and full SLO compliance.

The size of the cap-reclaimable gain is itself architecture-dependent (Figure A3). At comparable total scale, the cap-only arm recovered +26.0±6.4% tokens/J on the 480B MoE but only +5.7±8.0% on the 405B dense model. The activity counters of A.1 explain the diference: dense models at frontier scale saturate the tensor pipes (prefill tensor activity 0.54-0.86) and draw at or near the device power ceiling, leaving little for a cap to reclaim without cutting into demand; MoE models activate a fraction of their parameters per token (tensor activity 0.07-0.24), run well below the ceiling, and pair that headroom with a flat, memory-bound decode profile that tolerates a binding cap. This is why the two MoE models of the main body yield +17.6% to +38.8% (§5.1), and it sharpens H1: the reclaimable slack, not just the setpoint, is a property of the deployed model.

A methodological note from the same campaign: single-trial results on tail percentiles were unreliable enough to produce two conclusions that replication overturned, which set the replication discipline used throughout §5.

![](images/289779518681798b0fcf2cd2d198b59f8fe3b547c2987ac369f99be22c48cadb.jpg)  
Figure A3: Why MoE benefits most. (a) At comparable total scale, cap-only control recovers roughly five times more eficiency on the MoE model than on the dense model. (b) The mechanism: dense models saturate compute and draw at the device ceiling; MoE models run far below saturation, leaving slack that a calibrated cap converts into savings.

![](images/bf5ea0ba5850fb72889e1ac896fbfe5f40a366ca4c7182aa243cb7f8c36bb000.jpg)

(b) energy-efficiency gain (Δ tokens/J vs baseline)  
![](images/8d6b018b70024178bc9c50a5ee8b7bc045ac25fb169cedb3a61f2446a46946e1.jpg)  
Figure A2: The colocated dilemma, and its structural resolution by disaggregation. Panels share the x axis: (a) prefill tail-latency cost and (b) energy-eficiency gain for the same three configurations. A single colocated setting either collapses prefill latency (left, axis broken at +1005%) or forgoes a large fraction of the decode-phase gain (middle); per-lane mechanisms on disaggregated serving (right) obtain the largest eficiency gain while improving tail latency.  
(a) cap-only efficiency gain by architecture (replicated, ±1σ)  
(b) prefill tensor-core activity (p50)

![](images/6d08a7611cf44a10c231e6f2b20ec57d645a65a0144bdc1b20f088f8e4e1999c.jpg)  
Each lane is normalized to its own natural draw (P\_nat). The prefill lane keeps its bursty compute-bound profile; the clock window lowers burst peaks at a small stretch in burst duration. The decode lane is flat and memory-bound; the cap binds below P\_nat for the entire run Illustrative reconstruction from the PD serving A/B of §5.3 (480B, TP4/TP4): levels are measured per-lane means; fine variation is illustrative.

Figure A4: Lane-level power timelines under phase-decoupled control, both workloads (illustrative reconstruction from the PD serving A/B of §5.3; each lane normalized to its own natural draw, levels are measured per-lane means). The clock window shaves prefill burst peaks; the cap binds continuously through decode’s flat draw.

## A.4 Phase-decoupled actuation at run level

Figure A4 shows what the §3.1 assignment does during steady disaggregated serving, per lane and per workload. The prefill lane keeps its bursty, compute-bound profile; the clock window lowers the power peaks of each burst at the cost of a small stretch in burst duration, with the window floor bounding the latency impact by construction. The decode lane draws flat, memory-bound power; the cap binds below the lane’s natural draw for the entire run, cutting decode draw by 16% (agentic) and 20% (standard) while the GPU’s self-DVFS preserves token cadence within the ITL-p99 SLO on both workloads. The two mechanisms act simultaneously on diferent silicon, which is precisely what colocated serving cannot express.

The same protocol included a dry-run arm (control plane active, no hardware writes), which was indistinguishable from baseline on every metric; the control plane itself adds no measurable overhead.

## A.5 What the campaign established

(1) Mechanism and setpoint must be measured per deployment: every intermediate heuristic we formed was overturned either by replication or by a topology change (A.2). (2) The power cap was the only actuator whose latency cost remained bounded across all models, workloads, and topologies tested, which made it the decode-lane default that §5.3 then confirmed head-to-head against static locks under SLO constraints. (3) Median metrics conceal actuation damage: a configuration can look lossless at the median while violating the tail SLO outright, which is why the acceptance criterion of §3.3 and the evaluation lens of §4 operate on tail percentiles. (4) Single-trial results on tail percentiles are unreliable; every campaign conclusion cited here is replicated.

## References

[1] S. Narayanaswamy, P. Patel, I. Karlin, U. Gupta, V. Saripalli, and Y. Guo (NVIDIA). Datacenter energy optimized power profiles. arXiv:2510.03872, 2025.

[2] P. Patel, E. Choukse, C. Zhang, Í. Goiri, B. Warrier, N. Mahalingam, and R. Bianchini. Characterizing power management opportunities for LLMs in the cloud. In ASPLOS, 2024.

[3] P. Patel, E. Choukse, C. Zhang, A. Shah, Í. Goiri, S. Maleki, and R. Bianchini. Splitwise: Eficient generative LLM inference using phase splitting. In ISCA, 2024.

[4] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang. DistServe: Disaggregating prefill and decoding for goodput-optimized LLM serving. In OSDI, 2024.

[5] R. Qin et al. Mooncake: A KVCache-centric disaggregated architecture for LLM serving. In USENIX FAST, 2025.

[6] J. Yu, A. Taneja, J. Lin, and M. Zhang. VoltanaLLM: Energy-eficient and SLO-aware disaggregated LLM serving via adaptive frequency control and state-space routing. In ISC High Performance, 2026. arXiv:2509.04827.

[7] O. Basit, Y. Liu, Z. J. Kong, and Y. C. Hu. DualScale: Energy-eficient disaggregated LLM serving via phase-aware placement and DVFS. arXiv:2602.18755v3, 2026.

[8] J. Stojkovic, C. Zhang, Í. Goiri, J. Torrellas, and E. Choukse. DynamoLLM: Designing LLM inference clusters for performance and energy eficiency. In HPCA, 2025.

[9] J. Stojkovic, C. Zhang, Í. Goiri, E. Choukse, H. Qiu, R. Fonseca, and R. Bianchini. TAPAS: Thermal- and power-aware scheduling for LLM inference in cloud platforms. In ASPLOS, 2025.

[10] Y. Jiang, S. Chowdhary, N. Morris, R. Jain, S. Manne, and S. Bayliss (AMD). RAPID: Power-aware dynamic reallocation for inference. arXiv:2601.12241, 2026.

[11] C. Hankendi, R. Shahout, M. Yu, and A. Coskun. PALS: Power-aware LLM serving for mixture-of-experts models. arXiv:2605.21427, 2026.

[12] Y. Ma, S. Subramaniyan, and X. Wang. Power capping of GPU servers for machine learning inference optimization. In ICPP, 2025.

[13] Q. Liu, D. Huang, M. Zapater, and D. Atienza. GreenLLM: SLO-aware dynamic frequency scaling for energyeficient LLM serving. arXiv:2508.16449, 2025.

[14] A. K. Kakolyris, D. Masouros, P. Vavaroutsos, S. Xydis, and D. Soudris. throttLL’eM: Predictive GPU throttling for energy eficient LLM inference serving. In HPCA, 2025. arXiv:2408.05235.

[15] B. Ma, A. Afzal, J. Eitzinger, and G. Wellein. The illusion of power capping in LLM decode: A phase-aware energy characterisation across attention architectures. arXiv:2605.11999, 2026.

[16] ShareGPT conversation corpus. https://sharegpt.com.

[17] J. You, J.-W. Chung, and M. Chowdhury. Zeus: Understanding and optimizing GPU energy consumption of DNN training. In NSDI, 2023.

[18] J.-W. Chung, Y. Gu, I. Jang, L. Meng, N. Bansal, and M. Chowdhury. Perseus: Reducing energy bloat in large model training. In SOSP, 2024.

[19] S. Choi, I. Koo, J. Ahn, M. Jeon, and Y. Kwon. EnvPipe: Performance-preserving DNN training framework for saving energy. In USENIX ATC, 2023.