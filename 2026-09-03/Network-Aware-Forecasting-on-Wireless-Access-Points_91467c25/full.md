# Network-Aware Forecasting on Wireless Access Points

Niloo Bahadori Cisco Systems nbahador@cisco.com

Swadhin Pradhan Cisco Systems swapradh@cisco.com

Peiman Amini Cisco Systems pamini@cisco.com

## Abstract

Enterprise wireless access points (APs) are promising platforms for predictive machine learning (ML), but their primary responsibility remains providing wireless connectivity and network services. Predictive inference must therefore share an AP’s CPU and memory with packet processing, Wi-Fi and IoT radio operations, and client management. This resource contention creates two risks: a model that performs well on proxy hardware may be too slow on the target AP, while a model that fits in isolation may still degrade network services under load. We define network-aware deployability using two gates: qualification of the model and its execution path on the target AP, followed by validation of its execution profile under packet-service and forecasting constraints. Our benchmarks show that edge testbeds do not reliably capture target behavior. Across matched artifacts and serving settings, five model implementations run 6.1–19.1× slower on an AP than on a Raspberry Pi 5, while peak memory usage differs by up to 22%. Moreover, two forecasting foundation models of similar size differ in AP latency by 19×. When serving a smaller model across 13 parallel streams at a 30 s cadence under network saturation, default execution increases p99 round-trip time (RTT) by 76% and reduces throughput by 7.06%. Understanding these trade-offs is essential for live deployment if we aim to use APs for both networking and ML workloads.

## 1 Introduction

Enterprise access points (APs) are continuously operating network appliances at the enterprise edge. They provide client access; process and forward or tunnel traffic; execute Wi-Fi, IoT-radio, and control-plane functions; and manage connected clients. They also have direct visibility into clientonboarding events, local system health, and wired-uplink telemetry, making them useful location for forecasting wireless, networking, and system behavior. Prior work showed that network-native time-series foundation models (TSFMs) can forecast wireless telemetry and that a compact model can achieve sub-second inference on AP-class proxy hardware [1]. We ask the follow-up deployment question: when can recurring inference run on an AP without interfering with the network it serves?

An AP differs from a general-purpose edge computer, even with similar CPU and memory capacity, because uninterrupted network service takes priority over secondary workloads. AP’s resources must first support processing and forwarding traffic, managing the radio and control plane, and serving connected clients. The compute headroom available for predictive inference is therefore workload-dependent and can shrink as traffic, client count, and control-plane activity increase. On-AP inference is useful only when it meets the required prediction cadence without increasing client latency or reducing throughput.

Our experiments show two independent failure modes. First, parameter count and proxy hardware latency do not reliably identify the production operating point. Second, even a resident model that completes an individual inference in under one second can perturb throughput and tail latency under contention. These observations motivate two deployment checks: qualify the complete model and execution path on the target AP before deployment, and then verify that the recurring multi-stream forecasting workload remains within packet-service and prediction-timing budgets as network load changes (as shown in Fig. 1). We therefore formulate on-AP forecasting as an asymmetric, network aware co-location problem, rather than as a one-time model-placement decision or fair resource sharing among peer inference workloads. We make three contributions:

![](images/b9b4191a12b94e55b3c4dd5019ff887cd93e349ef8cbb94afbe66c28061019bd.jpg)  
Figure 1: System model.

1. A two-gate deploy methodology that qualifies the complete model and execution path on the target AP and then certifies load-specific execution profiles against a deployment contract containing packet-service and forecast-deadline requirements.

2. We perform a matched AP/Raspberry Pi 5 characterization showing that inference latency transfers differently across models and cannot be estimated using a fixed proxy-to-device scaling factor.

3. We evaluate a target-feasible workload serving 13 forecast streams under network contention and show that the runtime configuration determines whether it meets the network-performance budget.

## 2 Two-Gate Deployability

We express on-AP deployability as two feasibility gates over the complete model (shown in Fig. 1) and execution path, rather than as a single multi-objective optimizer. Let c denote the deployment context for a particular target environment. The context captures the target AP, application and networkservice class, and specifies the required forecast quality, resource budgets, acceptable forecast age, and tolerable packet-service degradation. For example, a latency-sensitive robotic-control WLAN may use a tighter packet-latency budget than a conventional office WLAN.

Gate 1: target-AP qualification. A candidate $x = ( f , e )$ combines a forecasting model f with an execution configuration e, including the artifact, precision, runtime, threading, CPU placement and priority, and parallelism. For deployment context c, the candidate passes the target-AP gate when

$$
\begin{array} { r } { x \in \mathcal { X } _ { \mathrm { A P } } ( c ) \longleftrightarrow \left\{ \begin{array} { l l } { Q _ { i } ( x ) \geq q _ { i } ^ { \star } ( c ) , \qquad \forall i \in \mathcal { D } , } \\ { M _ { \mathrm { p k } } ( x ) \leq B _ { M } ( c ) , } \\ { T _ { \mathrm { c y c } , 0 } ^ { 9 9 } ( x , S ) \leq D _ { \mathrm { c y c } } ( c ) . } \end{array} \right. } \end{array}\tag{1}
$$

Here, $q _ { i } ^ { \star } ( c )$ is the required forecast quality for domain i, $B _ { M } ( { \boldsymbol { c } } )$ is the memory budget available on the target AP, and $D _ { \mathrm { c y c } } ( c )$ is the forecast-cycle deadline specified by the deployment context.

Gate 2: load-dependent operating envelope. For a networking workload w and a fixed execution profile π, let $Y _ { 0 } ( w )$ and $R _ { 0 } ( w )$ be throughput and p99 RTT under a matched no-ML baseline, and let $Y _ { \pi } ( w )$ and ${ \dot { R } } _ { \pi } ( w )$ be the corresponding measurements with recurring inference. We report nonnegative degradation:

$$
\begin{array} { l } { { d _ { \mathrm { t p } } ( w , \pi ) = \displaystyle \left[ \frac { Y _ { 0 } ( w ) - Y _ { \pi } ( w ) } { Y _ { 0 } ( w ) } \right] _ { + } , } } \\ { { d _ { \mathrm { r t t } } ( w , \pi ) = \displaystyle \left[ \frac { R _ { \pi } ( w ) - R _ { 0 } ( w ) } { R _ { 0 } ( w ) } \right] _ { + } , } } \end{array} \quad \begin{array} { r l } { { [ z ] _ { + } \triangleq \operatorname* { m a x } ( z , 0 ) . } } \end{array}\tag{2}
$$

![](images/c94acfd05e6759db53aa7f4f869666f837537af7404b01fc5b6b3bd57c549468.jpg)

![](images/55429106a697696a992f0440b4f110c99dbd7d6666eb20904aff5314d2880d61.jpg)  
Figure 2: Target device characterization. Left: Peak RSS, Raspberry Pi 5 vs. access point. Right: Per forecasting task inference latency.

Thus, an observed throughput improvement or RTT reduction is reported as zero degradation rather than as an ML-attributed service benefit.

A profile is admissible for workload w and deployment context c when

$$
\pi \in \Pi _ { \mathrm { a d m } } ( w , c ) \longleftrightarrow \left\{ \begin{array} { l l } { d _ { \mathrm { t p } } ( w , \pi ) \leq \epsilon _ { \mathrm { t p } } ( c ) , } \\ { d _ { \mathrm { r t t } } ( w , \pi ) \leq \epsilon _ { \mathrm { r t t } } ( c ) , } \\ { T _ { \mathrm { c y c } } ^ { 9 9 } ( x , w , \pi , \mathcal { S } ) \leq D _ { \mathrm { c y c } } ( c ) . } \end{array} \right.\tag{3}
$$

Forecast-accuracy thresholds and deadlines follow the required prediction cadence and validity interval of each use-case. Memory and compute budgets reflect the $\mathrm { A P } ^ { * } \mathrm { s }$ capacity after reserving resources for networking, while throughput and tail-latency limits follow network-service requirements.

## 3 Evaluation

Setup. We evaluate a production-class, accelerator-free AP with a four-core ARM processor and approximately 2 GB of RAM. The resident candidate, APEX 2 (8.8M parameters), uses a 288 seconds long context sampled at 1 Hz and produces a point forecast plus nine quantiles. It runs every 30 s across 13 streams: client onboarding (2), AP self-health (5), and wired uplink (6). These domains collectively span three essential layers of the service path: access, local platform operation, and upstream connectivity. Each exhibits distinct dynamics, predictability, and error costs. Yet all compete for the same limited compute and memory resources. For qualification, each model uses the same artifact and serving configuration on the AP and Raspberry Pi 5. Peak RSS and latency include the unoptimized proof-of-concept runtime and temporary tensors; therefore, the absolute values do not represent final product footprints.

We evaluate the two deployment gates separately. Gate 1 measures model latency and memory on the target AP. The forecast quality of these models on wireless telemetry is established in [1], here we only study the systems gates. For Gate 2, one associated wireless client generates sustained traffic while recurring inference serves all 13 streams. Under default execution, we sweep approximately 100 Mbps, 250 Mbps, and saturation, conducting three 60 s trials per condition. A separate saturation run compares default and service-prioritized execution against matched no-ML baselines. The impact of the network is quantified relative to these baselines using throughput and p99 RTT. Forecast serviceability is assessed using the full 13-stream cycle-completion time under both execution profiles and the number of deadline misses. A cycle violates the freshness requirement if it exceeds the deployed cadence, $D _ { \mathrm { c y c } } = 3 0 \mathrm { s }$ , chosen as a practical timescale for the use cases considered. We set a 5% network degradation as engineering tolerance. Together, the two gates distinguish target-device feasibility in isolation from safe coexistence under network contention.

Target hardware changes the feasible set. Figure 2 shows strong model-dependent AP/Pi latency slowdowns (6.1–19.1×), ruling out a fixed proxy-to-target scaling factor and showing that parameter count alone is insufficient. In contrast, peak RSS transfers closely (within 22%), indicating that latency is dominated by model-platform interaction rather than coarse resource specifications.

Table 1: Network Impact and Forecast Cycle of recurring APEX 2 inference on the production AP.  
(a) Network Impact under Increasing Load
<table><tr><td>Network load</td><td>Throughput loss</td><td>p99 RTT inflation</td></tr><tr><td>~100 Mbps</td><td>0%</td><td>0%</td></tr><tr><td>~250 Mbps</td><td>0%</td><td>53%</td></tr><tr><td>Saturation</td><td>7.06%</td><td>76%</td></tr></table>

(b) Saturation: Service Protection versus Forecast Freshness
<table><tr><td>Profile</td><td>Throughput loss</td><td>p99 RTT inflation</td><td>Cycle p99</td><td>Deadline misses</td></tr><tr><td>Default</td><td>7.06%</td><td>76%</td><td>8.31 s</td><td>0/100</td></tr><tr><td>Service-prioritized</td><td>0%</td><td>0%</td><td>17.5 s</td><td>0/100</td></tr></table>

Network protection and inference speed. Under default ML execution, increasing network load first affects tail latency: at approximately 250 Mbps, p99 RTT increases by 53% (no throughput loss). At saturation, the throughput loss reaches 7.06% and the inflation of the p99 RTT reaches 76%. Service-prioritized execution protects networking by restricting and deprioritizing inference, which leads to an admissible saturation operating point with no measured nonnegative network degradation. This protection slows the 13-stream forecast cycle from 8.31 s to 17.5 s p99, but both profiles remain below the deployed 30 s deadline with zero observed deadline misses. Our observations point to inference-runtime memory allocation and memory-system contention as key contributors to throughput loss and tail-latency inflation. However, because we do not isolate these mechanisms individually, we present this interpretation as a systems-level hypothesis rather than a component-level attribution.

## 4 Related Work

Platform-aware mobile systems establish that FLOPs and parameter counts poorly predict latency, motivating target-device measurement [2; 3]. HAQ uses hardware feedback for mixed-precision quantization [4], while [5] specializes subnetworks across devices and latency constraints. MCUNet co-designs models and inference software under tight edge constraints [6]; DeepMon accelerates continuous vision on mobile GPUs [7]; and NestDNN adapts competing DNNs to changing mobile resources [8]. Clockwork extends this systems perspective to predictable inference and request-level isolation in cloud model serving [9]. These works enhance efficiency, match workloads to resources, or isolate inference. Enterprise APs instead prioritize forwarding, radio/control-plane functions, and client service; inference is secondary and useful only while fresh. Deployment must therefore balance forecast freshness with network impact under contention.

For temporal forecasting, TimesFM and Toto 2.0 provide general baselines [10; 11], TimeRAN develops lightweight foundation models for RAN analytics [12], and APEX introduced networknative pre-training and compact variants evaluated on AP-class proxy hardware [1]. Our work bridges these lines by fixing the forecasting task and defining network-aware deployability as target-AP qualification of the model and execution path, followed by validation under packet-service and forecasting constraints. This approach separates model qualification from service admission and complements model-centric forecasting evaluations with constraint-aware systems validation. Overall, network-aware deployability extends deployment evaluation beyond proxy performance to include target-device feasibility and safe coexistence with production networking.

## 5 Conclusion

Predictive inference on an enterprise AP is an asymmetric co-location problem, not merely an edgemodel optimization problem. Our measurements show that the target and service feasibility are independent: a model and execution path qualified on the target device can still become serviceinfeasible under contention. Such workloads must therefore operate within a network-load-dependent envelope that protects networking and forecast freshness. Hence, the deployment objective is not just to deliver minimum inference latency but an optimal execution profile that preserves packet service while meeting forecast deadlines.

## References

[1] S. Pradhan, N. Bahadori, and P. Amini. APEX: A Network-Native Time-Series Foundation Model for Forecasting and Anomaly Detection for Wireless Edge Operations. FMSD Workshop at ICML, 2026.

[2] M. Tan et al. MnasNet: Platform-Aware Neural Architecture Search for Mobile. CVPR, 2019.

[3] T.-J. Yang et al. NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications. ECCV, 2018.

[4] K. Wang et al. HAQ: Hardware-Aware Automated Quantization with Mixed Precision. CVPR, 2019.

[5] H. Cai et al. Once for All: Train One Network and Specialize It for Efficient Deployment. ICLR, 2020.

[6] J. Lin et al. MCUNet: Tiny Deep Learning on IoT Devices. NeurIPS, 2020.

[7] L. N. Huynh, Y. Lee, and R. K. Balan. DeepMon: Mobile GPU-Based Deep Learning Framework for Continuous Vision Applications. MobiSys, 2017.

[8] B. Fang, X. Zeng, and M. Zhang. NestDNN: Resource-Aware Multi-Tenant On-Device Deep Learning for Continuous Mobile Vision. MobiCom, 2018.

[9] A. Gujarati et al. Serving DNNs like Clockwork: Performance Predictability from the Bottom Up. OSDI, 2020.

[10] A. Das, W. Kong, R. Sen, and Y. Zhou. A Decoder-Only Foundation Model for Time-Series Forecasting. ICML, 2024.

[11] E. Khwaja et al. Toto 2.0: Time Series Forecasting Enters the Scaling Era. arXiv:2605.20119, 2026.

[12] I. Panitsas and L. Tassiulas. A Family of Open Time-Series Foundation Models for the Radio Access Network. arXiv:2604.04271, 2026.