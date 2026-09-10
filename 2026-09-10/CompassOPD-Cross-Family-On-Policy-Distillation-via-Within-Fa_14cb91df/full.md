# CompassOPD: Cross-Family On-Policy Distillation via Within-Family Likelihood Shifts

Naibin Gu<sup>1,2</sup>, Qingyi Si<sup>3</sup>, Chenxu Yang<sup>1,2</sup>, Chuanyu Qin<sup>1,2</sup>, Junhao Zhou<sup>1,2</sup>, Peng Fu<sup>1,2</sup>\*, Zheng Lin<sup>1,2</sup>, Weiping Wang<sup>1</sup>

<sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China   
<sup>2</sup>School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China <sup>3</sup>JD.COM {gunaibin,fupeng}@iie.ac.cn

## Abstract

On-policy distillation (OPD) provides dense token-level supervision on student-generated trajectories. Although OPD performs strongly when teacher and student belong to the same model family, we find that its effectiveness degrades in cross-family settings even after tokenizer alignment, with substantially stronger external teachers offering little additional improvement. To understand this disconnect, we decompose the cross-family OPD signal into two components: an offset between a lowcapability teacher-family reference and the student, and the within-family log-likelihood shift from that reference to the strong teacher. Standard OPD transfers both components together, allowing the offset to dominate the update di rection and obscure the changes associated with teacher capability improvements. We propose CompassOPD, which removes this off set and transfers the within-family shift, while a frozen student reference anchors updates to the student’s initial policy. Thus, both teacherside and student-side changes are measured within their respective model families. Experiments across three student families and mul tiple teacher families show that CompassOPD consistently outperforms standard cross-family OPD, improving average reasoning accuracy by up to 5.50 points. For an MoE teacher, we further construct the reference directly from the teacher checkpoint by reducing expert activation, eliminating the need for a separate reference checkpoint while retaining a 3.43-point gain over OPD.

## 1 Introduction

On-policy distillation (OPD) has emerged as an effective paradigm for post-training large language models (Agarwal et al., 2024; Gu et al., 2026b; Lu and Lab, 2025). It samples trajectories from the student policy and uses the teacher’s token-level likelihoods to provide dense supervision along the states that the student actually visits. This on-policy interaction supplies token-level learning signals while avoiding the distribution gap introduced by teacher-generated training data. By combining onpolicy exploration with dense token-level feedback, OPD has shown strong empirical gains on reasoning tasks (Yang et al., 2026a,b; Gu et al., 2026a).

However, existing OPD methods have mainly been studied in scenarios where the teacher and student models belong to the same model family.<sup>1</sup> Such models largely share training recipes and policy structures, with capability differences typically arising from model scale or post-training strategies (Li et al., 2026; Ma et al., 2026). In practice, the student’s model family may not provide a sufficiently strong teacher model, making it necessary to use stronger models from other families. Prior work has addressed the representational differences caused by different tokenizers through vocabulary or text-space alignment, enabling crossmodel likelihood comparison. Such alignment makes likelihood scores comparable over shared textual units (Sun et al., 2026; Niu et al., 2026; Wang et al., 2026), but leaves open whether the resulting supervision effectively transfers teacher capabilities across model families.

We find that, despite vocabulary alignment, replacing a same-family teacher model with a crossfamily teacher model of similar capability does not preserve the benefits of OPD. Figure 1(a) shows that same-family OPD produces sustained improvements, whereas cross-family OPD plateaus quickly. Moreover, as shown in Figure 1(b), external teacher models with substantially different capabilities yield distilled student models with nearly identical performance. Thus, under cross-family OPD, improvements in teacher capability do not reliably translate into greater gains in student performance.

To understand this phenomenon, we further compare the OPD update directions produced by models of different scales within the same teacher family on identical student trajectories. Surprisingly, as shown in Figure 1(c), despite the substantial capability gap, the strongest teacher and the smallest model in the teacher family still assign the same update direction on 70% of aligned textual units. This prompts us to ask: which part of the strong teacher signal truly reflects its capability improvement? Using a low-capability model from the same family as a reference, we find that the cross-family OPD signal can be decomposed into two components: one is the pre-existing cross-family offset between the reference model and the student, and the other is the within-family likelihood change from the lowcapability reference model to the strong teacher model. Standard OPD transfers both components together; when the cross-family offset determines the sign of the combined signal, the update follows the direction already induced by the low-capability reference, while the strong teacher’s additional contribution changes only its magnitude.

Based on this observation, we propose CompassOPD. CompassOPD removes the cross-family offset and transfers only the within-family likelihood change. A frozen copy of the student’s initial policy provides an anchor that modulates the update according to the student’s displacement from that initialization. In this way, both teacher-side and student-side variations are measured within their respective model families. Through this design, CompassOPD transfers the policy change directions associated with teacher capability improvement rather than directly matching the absolute likelihoods of a cross-family teacher model.

Experiments across three student families and multiple teacher families show that CompassOPD consistently outperforms standard cross-family OPD under matched student initializations, improving average reasoning accuracy by up to 5.50 points. Further analysis shows that restoring the removed cross-family offset degrades performance. With a fixed teacher-family reference, CompassOPD also yields progressively better student performance as teacher scale increases. For MoE teacher models, we further construct a low-capability reference model from the teacher checkpoint itself by reducing expert activation, thereby eliminating the need for a separately released reference model while still retaining substantial improvements over OPD.

## 2 Preliminaries

On-Policy Distillation. Given a prompt $x ,$ the student policy $\pi _ { \theta }$ generates a response $y \sim \pi _ { \theta } ( \cdot |$ x). At generation step $t ,$ let $c _ { t } = ( x , y _ { < t } )$ denote the student-visited context and let $y _ { t }$ be the sampled action. OPD (Lu and Lab, 2025) asks a teacher $\tau$ to score the same action under the same textual context and constructs the token-level signal

$$
A _ { t } ^ { \mathrm { O P D } } = \log { \mathcal { T } } ( y _ { t } \mid c _ { t } ) - \log { \pi _ { \theta } ( y _ { t } \mid c _ { t } ) } .\tag{1}
$$

The detached signal is used as an advantage in the policy update, , with positive values encouraging the sampled action and negative values discouraging it:

$$
\mathcal { L } _ { \mathrm { O P D } } = - \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot | x ) } \left[ A _ { t } ^ { \mathrm { O P D } } \log \pi _ { \theta } ( y _ { t } \mid c _ { t } ) \right] ,\tag{2}
$$

Because the teacher evaluates trajectories generated by the current student, OPD provides dense supervision on states that the student actually visits.

Cross-Tokenizer Likelihood Alignment. Equation 1 assumes that the teacher and student represent the sampled action with compatible tokens. Prior work on cross-tokenizer distillation addresses this mismatch by aligning model likelihoods in text space (Sun et al., 2026; Niu et al., 2026). Following this line of work, we decode the student response into text, re-encode it with the teacher tokenizer, and align the two token sequences according to their shared textual boundaries. This procedure partitions the response into aligned textual units

$$
{ \mathcal { U } } ( y ) = \{ z _ { 1 } , z _ { 2 } , . . . , z _ { M } \} ,\tag{3}
$$

where each unit is either a one-to-one token match or a span represented by different numbers of teacher and student tokens. We use $\ell _ { \mathcal { M } } ( z _ { u } \mid c _ { u } )$ to denote the aligned log-likelihood score assigned by model M to unit $z _ { u }$ under its preceding textual context $c _ { u }$ . When considering an individual aligned unit, we omit the index u and write $( z , c )$ We apply the same text-space alignment procedure to standard cross-family OPD and CompassOPD, ensuring that differences in the alignment procedure do not confound their comparison.

## 3 Understanding Cross-Family OPD

When the teacher model and the student model come from the same model family, for example both being Qwen3 series models, OPD has already demonstrated significant performance gains (Yang

![](images/216fc49b95148cc5bb2fa182a9064553ffa6e9e89a3a327722722644f900b854.jpg)

![](images/ae0b5d7d580c251bbd9f3deaa7ba86bb4b4558e7b805f5aa29076a5ca6900634.jpg)

![](images/92120aa23a11a5ae881bc8ec95ead7cdcfcf4458aa609787a075274a89ce81ab.jpg)  
0.8B OPD signal

![](images/86a303778a3bb0e82301c30eb534d258499bdd5f6267ba8dc6c119bdaa25c722.jpg)  
(a) Same-family vs. cross-family(b) Transfer of the teacher capability gap. (c) OPD signal similarity within the teacher family. OPD.

Figure 1: Motivating observations for cross-family OPD. (a) Performance comparison of same-family and crossfamily OPD. The same-family teacher model uses Qwen3-30B-A3B-Instruct, while the cross-family teacher model uses Qwen3.5-35B-A3B; the student model is Qwen3-4B in both cases. (b) The performance gap between Qwen3.5- 35B-A3B and Qwen3.5-4B, and the gap between the Qwen3-4B student models distilled from these two teacher models. Hollow and solid markers denote the teacher model gap and the distilled student model gap, respectively. (c) When scoring the same aligned student-generated actions, Qwen3.5-0.8B agrees in update direction with Qwen3.5-35B-A3B and Qwen3.5-4B at approximately 70% of positions, revealing a substantial OPD component shared across teacher scales.

et al., 2025). A natural question is whether these gains still hold when the teacher model is switched to a model from a different family.

## 3.1 Same-Family and Cross-Family OPD

We first consider a relatively close model family for comparison. In the same-family setting, we distill from a Qwen3-30B-A3B-Instruct teacher model to a Qwen3-4B student model. In the cross-family setting, we use Qwen3.5-35B-A3B (Qwen Team, 2026) as the teacher model. To reduce differences in output style, the cross-family student model is first initialized by performing SFT on trajectories generated by the teacher (see Appendix A for details).

The experimental results are shown in Figure 1(a). Same-family OPD produces sustained improvements in reasoning accuracy, whereas crossfamily OPD plateaus after limited early gains despite the preceding teacher-trajectory SFT.

Is the performance limitation of cross-family OPD related to teacher capability? After observing the limited gains achieved with Qwen3.5- 35B-A3B, we examine whether student gains track teacher capability. If a weaker cross-family teacher were used, the gains from distillation would be even worse, while a stronger teacher should correspondingly improve performance. We compared two teachers from the Qwen3.5 family but with significantly different performance levels, Qwen3.5-4B and Qwen3.5-35B-A3B, for performing OPD on the same Qwen3-4B student model.

Figure 1(b) reveals a clear misalignment between teacher capability and student model improvement. Across benchmarks, Qwen3.5-35B-A3B outperforms Qwen3.5-4B by 16.7 points on average. However, the Qwen3-4B student models distilled from these two teachers differ by only 0.1 points. On several benchmarks, the student distilled from the stronger teacher performs no better, and sometimes slightly worse, than the student distilled from the weaker teacher. Thus, the large capability gap within the teacher family translates into only marginal differences in the gains achieved by cross-family OPD.

Taken together, these observations lead to a more specific question: Why does the large capability gap between the two teachers translate into only a marginal difference in student model gains under cross-family OPD?

## 3.2 Dissecting cross-family OPD signals

To understand why the large teacher capability gap in Figure 1(b) leads to only marginal differences in student performance, we examine how tokenlevel OPD supervision signals vary with teacher capability. We use Qwen3.5-0.8B, Qwen3.5-4B, and Qwen3.5-35B-A3B to score the same crossfamily student trajectories, with these three models covering a wide range of reasoning performance within the same model family.

For an aligned textual unit z under context c, the OPD signal produced by a scoring model M is

$$
A _ { \mathcal { M } } ( z , c ) = \ell _ { \mathcal { M } } ( z \mid c ) - \ell _ { \pi _ { \theta } } ( z \mid c ) .\tag{4}
$$

Its sign determines whether OPD increases or decreases the likelihood of that observed action.

Figure 1(c) compares the signals produced by Qwen3.5-0.8B with those produced by the two larger models. Surprisingly, Qwen3.5-0.8B assigns the same update direction as Qwen3.5-35B-A3B on 70.0% of the aligned units, and the same direction as Qwen3.5-4B on 71.8% of the aligned units. Despite the large capability differences, the two larger models frequently induce the same update direction as the low-capability model, even though the latter has weaker reasoning abilities than the cross-family student. This motivates using the low-capability model as a reference to separate the OPD signal it already induces from the likelihood changes introduced by a stronger teacher.

![](images/668302e2bd2319caab7dc447672e3a001431ba8eb502a0d4e91deabd2064868c.jpg)  
Figure 2: Overview of CompassOPD. Given student-generated trajectories, the strong teacher T and its within-family reference $R _ { T }$ score the same aligned textual actions. Their likelihood difference isolates the capability-sensitive within-family shift by removing the reference-anchored cross-family offset. The resulting signal is combined with the student’s displacement from its initial policy to guide the on-policy update.

What masks the updates introduced by crossfamily teachers? For the same student action z under context c, let T denote the strong teacher and R a low-capability reference from the same teacher family. Using R as the OPD teacher yields:

$$
A _ { \mathcal { R } } ( z , c ) = \ell _ { \mathcal { R } } ( z \mid c ) - \ell _ { \pi _ { \theta } } ( z \mid c ) .\tag{5}
$$

For the strong teacher $\tau$ , adding and subtracting the reference score gives:

$$
\begin{array} { r l } & { \quad A _ { \mathcal { T } } ( z , c ) = \ell _ { \mathcal { T } } ( z \mid c ) - \ell _ { \pi _ { \theta } } ( z \mid c ) } \\ & { = \underbrace { \ell _ { \mathcal { R } } ( z \mid c ) - \ell _ { \pi _ { \theta } } ( z \mid c ) } _ { O ( z , c ) } + \underbrace { \ell _ { \mathcal { T } } ( z \mid c ) - \ell _ { \mathcal { R } } ( z \mid c ) } _ { \Delta _ { T } ( z , c ) } . } \end{array}\tag{6}
$$

The first component O is the cross-family OPD signal already induced by the low-capability reference R. It captures the log-likelihood discrepancy between R and the current student, forming the baseline OPD signal to which the strong teacher T adds its within-family log-likelihood shift. The second component $\Delta _ { T }$ measures the log-likelihood change from R to T on the same student action. Since this comparison is entirely within the teacher family, it provides a more direct signal of which student actions gain or lose support as teacher capability increases.

Standard cross-family OPD transfers the combined signal $O + \Delta _ { T }$ to the student. When the offset and the within-family shift favor opposite directions, an offset with greater magnitude causes the update to follow the reference-induced direction. Figure 1(c) shows that the strong teacher T and the low-capability reference R produce the same OPD update direction at a substantial fraction of aligned units. At these positions, the teacherfamily shift affects the update magnitude without reversing the direction already induced by the reference.

This observation motivates CompassOPD. We remove the shared cross-family offset O and use $\Delta _ { T }$ as the capability-sensitive distillation signal. The goal remains to transfer the strong teacher model’s capability, and the within-family likelihood change provides a clearer direction for achieving this goal.

## 4 CompassOPD

In this section, we introduce CompassOPD. Building on the preceding signal decomposition, CompassOPD removes the shared cross-family offset and uses the within-family log-likelihood shift to supervise student-generated trajectories. A frozen student reference policy further anchors the update according to the student’s displacement from its initial policy. Together, these teacher-side and student-side comparisons guide capability transfer across model families. The overall workflow is illustrated in Figure 2.

## 4.1 A Capability Signal from the Teacher Family

Let $\tau$ denote the strong teacher and R a lowcapability reference model from the same teacher family. Given a prompt x, the current student policy $\pi _ { \theta }$ samples a response, and both frozen models score the same aligned textual units in that response. For an aligned unit z under context $^ { c , }$ CompassOPD removes the shared cross-family offset O from the strong-teacher OPD signal $A \tau$

$$
\begin{array} { r l } { \Delta _ { T } ( z , c ) = A _ { T } ( z , c ) - O ( z , c ) } & { { } } \\ { = ( \ell _ { T } - \ell _ { \pi _ { \theta } } ) - ( \ell _ { \mathcal { R } } - \ell _ { \pi _ { \theta } } ) } & { { } } \\ { = \ell _ { T } ( z \mid c ) - \ell _ { \mathcal { R } } ( z \mid c ) . } & { { } } \end{array}\tag{7}
$$

Here, the student likelihood cancels in the subtraction, leaving a log-likelihood change measured entirely within the teacher family. A positive $\Delta _ { T }$ means that $\tau$ assigns higher likelihood to the same aligned action than R, while a negative value means that $\tau$ assigns lower likelihood than $\mathcal { R }$ Comparing $\Delta _ { T }$ across candidate continuations under the same context reveals how their relative preference changes from R to T.

## 4.2 Anchoring the Student Update

The teacher-family shift indicates which sampled actions should receive more or less support, but does not account for changes the student has already made. We therefore retain a frozen student reference policy $\pi _ { \mathrm { r e f } }$ , initialized from the student before OPD training, to anchor updates relative to that starting point. For each sampled student token $y _ { t }$ under context $c _ { t } .$ , we measure the change in its log-likelihood relative to the reference:

$$
\Delta _ { S } ( y _ { t } , c _ { t } ) = \log \pi _ { \theta } ( y _ { t } \mid c _ { t } ) - \log \pi _ { \mathrm { r e f } } ( y _ { t } \mid c _ { t } ) .\tag{8}
$$

Let $\Delta _ { T , t }$ denote the teacher-family shift assigned to student token position t through text-space alignment. CompassOPD combines this signal with the student’s displacement to construct the sampled advantage:

$$
A _ { t } ^ { \mathrm { C o m p a s s } } = \mathrm { s g } \left[ \Delta _ { T , t } - \alpha \Delta _ { S } ( y _ { t } , c _ { t } ) \right] ,\tag{9}
$$

where sg denotes stop-gradient and α controls the strength of the student reference anchor.

At the beginning of training, $\pi _ { \theta } ~ = ~ \pi _ { \mathrm { r e f } }$ and $\Delta _ { S } = 0$ , so the update is determined entirely by the teacher-family shift. As the student moves an action’s log-likelihood in the direction favored by the teacher signal, the anchor term $- \alpha \Delta s$ opposes further movement in that direction. To interpret this feedback, consider a policy π over aligned textual actions z at a fixed context $c .$ The following reference-regularized objective captures the balance between following the teacher-family shift and remaining close to the student reference:

$$
\begin{array} { r l } & { \mathcal { I } _ { \mathrm { C o m p a s s } } ( \pi ; c ) = \mathbb { E } _ { z \sim \pi ( \cdot | c ) } \left[ \Delta _ { T } ( z , c ) \right] } \\ & { \phantom { \mathcal { I } _ { \mathrm { C o m p a s s } } ( \pi ; c ) = } - \alpha D _ { \mathrm { K L } } \left( \pi ( \cdot \mid c ) \| \pi _ { \mathrm { r e f } } ( \cdot \mid c ) \right) . } \end{array}\tag{10}
$$

To see how the teacher-family shift and the student reference jointly shape the target distribution, we examine the maximizer of this objective. For $\alpha >$ 0, it takes the form (see Appendix B for a proof):

$$
\pi ^ { * } ( z \mid c ) = \frac { \pi _ { \mathrm { r e f } } ( z \mid c ) \exp { ( \Delta _ { T } ( z , c ) / \alpha ) } } { Z ( c ) } ,\tag{11}
$$

where $Z ( c )$ is the normalization factor ensuring that $\textstyle \sum _ { z } \pi ^ { * } ( z \mid c ) = 1$ . The exponential factor reweights the student’s reference policy according to the teacher-family shift, expressing the transferred changes relative to the student’s own initialization. In training, we substitute $A _ { t } ^ { \mathrm { C o m p a s s } }$ for the standard OPD advantage and otherwise retain the same on-policy optimization procedure.

MoE self-reference. To obtain a teacher-family reference without a separate checkpoint, we also consider using an MoE teacher under two expert activation configurations (Figure 2(c)). The standard configuration serves as $\tau _ { \ast }$ , while a configuration with fewer activated experts serves as $\mathcal { R } _ { : }$ , sharing the same frozen weights. Their log-likelihood difference on the same aligned student-generated actions supplies $\Delta _ { T }$ in Equation 7.

<table><tr><td>Method</td><td colspan="8">AIME24 AIME25 AIME26 HMMT25Feb HMMT25Nov HMMT26 MATH-500 Avg.</td></tr><tr><td colspan="16" rowspan="19">Student: Granite4.1-3B</td></tr><tr><td>Base 4.79</td><td>7.92</td><td>5.83</td><td>1.25</td><td>1.46</td><td>3.03</td></tr><tr><td>SFT</td><td>21.25</td><td>19.58 19.79</td><td>8.54</td><td>6.67</td><td>14.39</td><td>64.31 12.66 80.52 24.39</td></tr><tr><td>OPD</td><td>25.00</td><td>18.96 17.29</td><td>12.29</td><td>11.04</td><td>13.83</td><td>79.12 25.36</td></tr><tr><td>CompassOPD</td><td>33.75</td><td>24.58 25.62</td><td>12.29</td><td>14.37</td><td>21.02</td><td>30.86</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>84.36</td></tr><tr><td colspan="7">Student: Qwen3-4B</td></tr><tr><td>Base</td><td>24.79</td><td>18.33</td><td>18.12</td><td>12.50</td><td>8.33</td><td>17.80</td><td>83.25 26.16</td></tr><tr><td>SFT</td><td>28.33</td><td>24.17</td><td>22.92</td><td>13.54</td><td>12.50 15.53</td><td>84.51</td><td>28.79</td></tr><tr><td>OPD</td><td>36.25</td><td>32.29</td><td>30.83</td><td>17.50</td><td>20.00</td><td>21.02</td><td>86.70 34.94</td></tr><tr><td>CompassOPD</td><td>40.00</td><td>32.50</td><td>35.42</td><td>18.12</td><td>20.21</td><td>24.43</td><td>88.85 37.08</td></tr><tr><td colspan="8">Student: OLMo-3-7B</td></tr><tr><td>Base</td><td>5.63</td><td>5.21</td><td>6.88</td><td>1.67</td><td>3.12</td><td>2.65</td><td>64.51</td><td>12.81</td></tr><tr><td>SFT</td><td>30.00</td><td>25.83</td><td>28.54</td><td>13.33</td><td>14.17</td><td>16.67</td><td>83.95</td><td>30.36</td></tr><tr><td>OPD</td><td>31.46</td><td>29.17</td><td>28.54</td><td>16.88</td><td>15.00</td><td>23.30</td><td>86.14</td><td>32.93</td></tr><tr><td>CompassOPD</td><td>36.67</td><td>30.00</td><td>29.38</td><td>17.50</td><td>19.79</td><td>23.48</td><td>86.38</td><td>34.74</td></tr></table>

Table 1: Main results with Qwen3.5-35B-A3B as the teacher and Qwen3.5-0.8B as the reference for CompassOPD. OPD and CompassOPD share the same SFT initialization for each student. Best results within each student group are shown in bold.

## 5 Experiments

## 5.1 Experimental Setup

Training Data and Benchmarks. We evaluate CompassOPD on reasoning tasks. We use DAPO-Math (Yu et al., 2025) and DeepScaleR-Preview (Luo et al., 2025) for on-policy training. Evaluation is conducted on seven benchmarks: AIME 2024 (Zhang and Math-AI, 2024), AIME 2025 (Zhang and Math-AI, 2025), AIME 2026 (Zhang and Math-AI, 2026), HMMT 2025 February, HMMT 2025 November, HMMT 2026 (Dekoninck et al., 2026), and MATH-500 (Lightman et al., 2023). For each benchmark, we report mean accuracy over 16 independent generations per problem.

Models and baselines. We evaluate CompassOPD across three teacher families: Qwen3.5, Qwen3, and Mistral. The teachers are Qwen3.5- 35B-A3B in non-thinking mode, Qwen3-30B-A3B-Instruct-2507, and Ministral-3-14B (Liu et al., 2026), respectively. Each teacher is paired with a lower-capability reference from its own family. Our student models span three model families, represented by Granite4.1-3B (IBM Granite Team, 2026), Qwen3-4B, and OLMo-3-7B (Olmo et al., 2026). We compare the original model (Base), the teacher-trajectory SFT checkpoint, standard sampled OPD, and CompassOPD. For each comparison, both distillation methods start from the same SFT checkpoint trained on reasoning trajectories generated by the strong teacher. Further analyses examine teacher scale within Qwen3.5 with a fixed 0.8B reference and MoE self-reference constructed by reducing expert activation (see Appendix A for details).

Implementation details. We implement all methods using VERL (Sheng et al., 2024) and use vLLM (Kwon et al., 2023) for sampling generation and teacher scoring. The learning rate is set to 1 × 10<sup>−6</sup>, with a batch size of 128. The maximum prompt length and response length are 2,048 and 24,576 tokens, respectively. For CompassOPD, we set α = 0.5. OPD and CompassOPD use the same student initialization, token/span alignment.

## 5.2 Main Results

Tables 1 and 2 report cross-family distillation results spanning three teacher families and three student families. CompassOPD consistently outperforms standard OPD across all five configurations.

Performance across student families. Table 1 compares the three student families with Qwen3.5- 35B-A3B as the teacher. CompassOPD improves upon OPD in 20 of the 21 student–benchmark combinations, with one tie. The gains are most pronounced on Granite4.1-3B, where standard OPD provides limited improvement beyond SFT. CompassOPD also improves Qwen3-4B and OLMo-3- 7B, where standard OPD already yields clear gains, demonstrating benefits across students with different responses to cross-family distillation.

<table><tr><td colspan="9">Method AIME24 AIME25 AIME26 HMMT25Feb  $\mathrm { H M M T } 2 5 _ { \mathrm { N o v } }$  HMMT26 MATH-500 Avg.</td></tr><tr><td>Base</td><td>4.79</td><td>7.92</td><td>5.83</td><td>1.25</td><td>1.46</td><td>3.03</td><td>64.31</td><td>12.66</td></tr><tr><td colspan="9">Teacher: Qwen3-30B-A3B (Ref. Qwen3-0.6B)</td></tr><tr><td>SFT</td><td>21.04</td><td>20.62</td><td>17.71</td><td>11.87</td><td>8.33</td><td>13.83</td><td>79.60</td><td>24.71</td></tr><tr><td>OPD</td><td>22.50</td><td>25.21</td><td>23.13</td><td>13.33</td><td>14.37</td><td>19.13</td><td>83.20</td><td>28.70</td></tr><tr><td>CompassOPD</td><td>28.54</td><td>30.21</td><td>26.46</td><td>16.88</td><td>15.00</td><td>21.97</td><td>84.45</td><td>31.93</td></tr><tr><td colspan="9">Teacher: Ministral-3-14B (Ref. Ministral-3-3B)</td></tr><tr><td>SFT</td><td>18.54</td><td>18.12</td><td>14.79</td><td>9.79</td><td>6.25</td><td>13.26</td><td>78.44</td><td>22.74</td></tr><tr><td>OPD</td><td>21.46</td><td>16.67</td><td>18.12</td><td>11.46</td><td>8.75</td><td>13.26</td><td>78.93</td><td>24.09</td></tr><tr><td>CompassOPD</td><td>22.29</td><td>24.17</td><td>21.46</td><td>11.46</td><td>10.42</td><td>16.67</td><td>80.01</td><td>26.64</td></tr></table>

Table 2: Results with Qwen3 and Mistral teachers on Granite4.1-3B. Within each teacher group, OPD and CompassOPD share the same teacher-specific SFT initialization. Best results are shown in bold.

![](images/c464f1f18704e2e5e3bf631b012f753ab6b67a5193afb8f1e25b04a8a30a2401.jpg)  
(a) Anchor sensitivity.

![](images/34ca20e4a1d73bad2b6fd8bdaa870bd9f2086641830ffb915f4d111e24d5772f.jpg)  
(b) Offset restoration.  
Figure 3: Component analysis of CompassOPD on Granite4.1-3B, with Qwen3.5-35B-A3B as the teacher and Qwen3.5-0.8B as the reference. (a) Student anchor sensitivity. (b) Offset restoration with $\alpha = 0 . 5 ;$ the dashed line denotes standard OPD.

Performance across teacher families. Table 2 compares OPD and CompassOPD using Qwen3 and Mistral teachers with Granite4.1-3B as the student. CompassOPD improves average performance under both teacher families, outperforming OPD in 13 of the 14 teacher–benchmark combinations, with one tie. Together with the Qwen3.5 results, these findings demonstrate consistent improvements across all three evaluated teacher families.

## 5.3 Component Analysis

Sensitivity to the student reference anchor. Figure 3(a) shows that performance improves as the anchor strength increases from $\alpha = 0$ to $\alpha = 0 . 5$ where it reaches its highest value. When $\alpha = 0 ,$ , the update relies entirely on the teacher-family shift;

a moderate anchor additionally accounts for the student’s displacement from its initial policy. Performance remains competitive at $\alpha = 1 . 0$ but declines under stronger anchoring, suggesting that excessive regularization restricts student updates. We set $\alpha = 0 . 5$ in our main experiments.

Impact of the offset on cross-family OPD. To test whether the cross-family offset identified in Section 3 limits capability transfer, we gradually restore it while keeping the student anchor fixed at $\alpha = 0 . 5 \mathrm { : }$

$$
A _ { t } ( \beta ) = \mathrm { s g } \left[ \Delta _ { T , t } + \beta O _ { t } - \alpha \Delta _ { S } ( y _ { t } , c _ { t } ) \right] .\tag{12}
$$

Here, $O _ { t }$ is the offset assigned to student token position t, and $\beta$ controls the fraction restored, with $\beta = 0$ recovering CompassOPD. Figure 3(b) shows that the average accuracy of Granite4.1-3B decreases monotonically as more of the offset is restored, removing most of the gain over OPD. This decline under a fixed student anchor demonstrates the benefit of removing the cross-family offset for capability transfer.

## 5.4 Teacher and Reference Configurations

Effect of teacher–reference scale separation. We examine how the scale separation between T and R affects relative supervision. With Qwen3.5- 0.8B fixed as R, we progressively reduce T from Qwen3.5-35B-A3B to 9B, 4B, and 2B. Table 3 shows that the average accuracy of Granite4.1-3B decreases as the teacher approaches the reference in scale, consistent with the within-family shift becoming less informative as their separation narrows. Nevertheless, the 2B teacher paired with the 0.8B reference still outperforms standard OPD with the 35B-A3B teacher. Thus, even modest within-family variation can provide more effective supervision than the absolute likelihoods of a much larger cross-family teacher model. This reliance on a teacher-family reference raises a practical question: what if the teacher family lacks a suitable low-capability checkpoint to serve as R?

<table><tr><td>Configuration</td><td colspan="7">AIME24 AIME25 AIME26 HMMT25Feb  $\mathrm { H M M T } 2 5 _ { \mathrm { N o v } }$  HMMT26 MATH-500 Avg.</td></tr><tr><td>OPD</td><td>25.00</td><td>18.96</td><td>17.29</td><td>12.29</td><td>11.04</td><td>13.83</td><td>79.12</td><td>25.36</td></tr><tr><td colspan="9">CompassOPD with  $\mathcal { R } = 0 . 8 \mathrm { B }$ </td></tr><tr><td> $\mathcal { T } = 3 5 \mathrm { B } { \cdot } \mathrm { A } 3 \mathrm { B }$ </td><td>33.75</td><td>24.58</td><td>25.62</td><td>12.29</td><td>14.37</td><td>21.02</td><td>84.36</td><td>30.86</td></tr><tr><td> $\mathcal { T } = 9 \mathrm { B }$ </td><td>31.25</td><td>22.92</td><td>23.75</td><td>12.92</td><td>13.33</td><td>18.56</td><td>83.76</td><td>29.50</td></tr><tr><td> $\tau = 4 \mathrm { B }$ </td><td>30.21</td><td>23.13</td><td>24.58</td><td>12.08</td><td>9.17</td><td>17.80</td><td>83.61</td><td>28.65</td></tr><tr><td> $\tau = 2 \mathrm { B }$ </td><td>24.37</td><td>19.79</td><td>21.67</td><td>12.92</td><td>9.58</td><td>15.53</td><td>81.66</td><td>26.50</td></tr></table>

Table 3: Results on Granite4.1-3B with Qwen3.5 teachers of different scales. CompassOPD uses a fixed Qwen3.5- 0.8B reference, while OPD uses the 35B-A3B teacher.

<table><tr><td>Method</td><td>Reference R</td><td>Avg.</td></tr><tr><td>SFT OPD</td><td>一</td><td>24.39 25.36</td></tr><tr><td>CompassOPD</td><td>Separate checkpoint (0.8B)</td><td>30.86</td></tr><tr><td>CompassOPD</td><td>Teacher self-reference</td><td>28.79</td></tr></table>

Table 4: Reference constructions for Granite4.1-3B with Qwen3.5-35B-A3B as the teacher. Self-reference uses the same teacher checkpoint with expert activation reduced to approximately 2B active parameters.

Can a teacher model provide its own reference? CompassOPD relies on a suitable low-capability reference within the teacher family to remove the offset. A practical concern is that some top-tier very large models may not offer publicly accessible smaller versions within the same family, such as Kimi K3 (Team et al., 2026). However, these models commonly use MoE architectures, which provide a natural alternative: the same checkpoint can be run under different activation capacities. We therefore use Qwen3.5-35B-A3B under its standard routing configuration as T, and construct a reference R with roughly 2B activated parameters by enabling only one shared expert and one finegrained expert. This design varies the activation capacity while preserving the checkpoint and model family, allowing the teacher model to provide its own within-family reference.

Table 4 shows that this self-reference configuration improves average accuracy over standard OPD by 3.43 points. Although it trails the dedicated

0.8B reference by 2.07 points, it retains a substantial improvement without requiring a separately released reference checkpoint. The result shows that useful within-family variation can come from different activation capacities of a single teacher, providing a practical alternative when a suitable reference model is unavailable.

## 6 Related Work

Cross-family distillation. Different training recipes and tokenizers complicate distribution matching in cross-family distillation. Sequencelevel distillation transfers knowledge through teacher-generated text, which students can process with their own tokenizers (Kim and Rush, 2016). For finer-grained supervision, DSKD (Zhang et al., 2024) combines dual-space distillation with crossmodel attention to align representations and distributions. ULD (Boizard et al., 2025) uses optimal transport to compare distributions across vocabularies, while ALM (Minixhofer et al., 2025) matches likelihoods on corresponding text segments. Recent work extends cross-tokenizer distillation to student-generated trajectories through shared text units, token mapping, or byte-prefix marginalization (Sun et al., 2026; Niu et al., 2026; Wang et al., 2026). Together, these methods provide comparable supervision interfaces across model families.

On-policy distillation. On-policy distillation obtains teacher feedback on student-generated trajectories, covering the states visited by the student. GKD (Agarwal et al., 2024) studies distillation on student-generated sequences with different divergence objectives, while MiniLLM (Gu et al., 2026b) adopts reverse KL optimization. We follow the sampled OPD formulation of Lu and Lab (2025), using teacher–student log-probability differences as token-level advantages. Recent analysis shows that transfer also depends on reasoningpattern compatibility and the teacher’s additional capabilities, beyond its standalone performance (Li et al., 2026). Concurrently with our work, other studies have explored contrastive supervision in OPD. RLCSD (Pan et al., 2026) contrasts correct and incorrect solution hints to mitigate style drift, while W2S-OPD (Yu et al., 2026) aims to improve a stronger student by leveraging differences between smaller models within the same family, thereby enabling weak-to-strong OPD. These methods pursue objectives distinct from ours. CompassOPD is the first work to identify the cross-family offset as a central limitation in cross-family OPD and to propose removing it while using within-family likelihood shifts for cross-family capability transfer.

## 7 Conclusion

This paper reveals a key issue in cross-family OPD: standard OPD mixes the cross-family offset with within-family likelihood changes, allowing the former to mask capability-sensitive supervision. CompassOPD removes this offset, performs distillation using relative likelihood changes within the teacher family, and anchors student updates to the student’s initial policy. Experiments covering three student families and multiple teacher families show that CompassOPD consistently outperforms standard cross-family OPD. Further analysis validates the roles of offset removal and student-side anchoring. The self-reference construction based on MoE expert activation also demonstrates that the method can be applied without an independent small-scale reference checkpoint. Overall, relative changes within the teacher family provide a more effective supervision interface for cross-family capability transfer.

## Limitations

Due to computational constraints, we have not evaluated CompassOPD on very large models. Nevertheless, our experiments across multiple teacher and student sizes demonstrate its effectiveness at different model scales. Extending this validation to substantially larger models remains a direction for future work.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of lan-

guage models: Learning from self-generated mistakes. Preprint, arXiv:2306.13649.

Nicolas Boizard, Kevin El Haddad, Céline Hudelot, and Pierre Colombo. 2025. Towards cross-tokenizer distillation: the universal logit distillation loss for llms. Preprint, arXiv:2402.12030.

Jasper Dekoninck, Nikola Jovanovic, Tim Gehrunger,´ Kári Rögnvaldsson, Ivo Petrov, Chenhao Sun, and Martin Vechev. 2026. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms.

Naibin Gu, Chenxu Yang, Qingyi Si, Chuanyu Qin, Dingyu Yao, Peng Fu, Zheng Lin, Weiping Wang, Nan Duan, and Jiaqi Wang. 2026a. Co-evolving policy distillation. Preprint, arXiv:2604.27083.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2026b. Minillm: On-policy distillation of large language models. Preprint, arXiv:2306.08543.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, and 31 others. 2025. Openthoughts: Data recipes for reasoning models. Preprint, arXiv:2506.04178.

IBM Granite Team. 2026. Granite-4.1-3B. Hugging Face. Model card.

Yoon Kim and Alexander M. Rush. 2016. Sequencelevel knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas. Association for Computational Linguistics.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. Preprint, arXiv:2309.06180.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. 2026. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. Preprint, arXiv:2604.13016.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. Preprint, arXiv:2305.20050.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, Alexandre Sablayrolles, Amélie Héliou, Amos You, Andy Ehrenberg, Andy

Lo, Anton Eliseev, Antonia Calvi, Avinash Sooriyarachchi, Baptiste Bout, and 101 others. 2026. Ministral 3. Preprint, arXiv:2601.08584.

Kevin Lu and Thinking Machines Lab. 2025. Onpolicy distillation. Thinking Machines Lab: Connectionism. Https://thinkingmachines.ai/blog/onpolicy-distillation.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. 2025. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. Notion Blog.

Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, Jinhao Dong, Zhifang Sui, and Fuli Luo. 2026. Mopd: Multi-teacher on-policy distillation for capability integration in llm posttraining. Preprint, arXiv:2606.30406.

Benjamin Minixhofer, Ivan Vulic, and Edoardo Maria´ Ponti. 2025. Universal cross-tokenizer distillation via approximate likelihood matching. Preprint, arXiv:2503.20083.

Yifan Niu, Han Xiao, Dongyi Liu, Zelong Wang, Dihong Gong, Yasheng Wang, and Jia Li. 2026. Breaking the tokenizer barrier: On-policy distillation across model families. Preprint, arXiv:2606.09456.

Team Olmo, :, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, and 50 others. 2026. Olmo 3. Preprint, arXiv:2512.13961.

Leyi Pan, Shuchang Tao, Yunpeng Zhai, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei Liu, and Lijie Wen. 2026. Rlcsd: Reinforcement learning with contrastive on-policy self-distillation. Preprint, arXiv:2606.11709.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Jie Sun, Mao Zheng, Mingyang Song, Qiyong Zhong, Yilin Cheng, Bichuan Feng, Pengfei Liu, Junfeng Fang, and Xiang Wang. 2026. Simct: Recovering lost supervision for cross-tokenizer on-policy distillation. Preprint, arXiv:2605.07711.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, M. C., Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y. Charles, H. S. Che, Guanduo Chen, Guangyu Chen, Guanzheng Chen, Huarong Chen, Jia Chen, Jianlong Chen, Jun Chen, and 383

others. 2026. Kimi k3: Open frontier intelligence. Preprint, arXiv:2607.24653.

ModelScope Team. 2024. EvalScope: Evaluation framework for large models.

Hao Wang, Kun Yuan, Wenlin Zhong, Minglei Zhang, Han Xiao, Ming Sun, and Honggang Qi. 2026. Cross-tokenizer on-policy distillation via byte-prefix marginalization. Preprint, arXiv:2607.22334.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. 2026a. Self-distilled rlvr. Preprint, arXiv:2604.03128.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. 2026b. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. Preprint, arXiv:2602.12125.

Fangxu Yu, Weijia Xu, Michael Xu, Tianyi Zhou, and Zinan Lin. 2026. Weak-to-strong on-policy distillation. Preprint, arXiv:2607.26246.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Songming Zhang, Xue Zhang, Zengkui Sun, Yufeng Chen, and Jinan Xu. 2024. Dual-space knowledge distillation for large language models. Preprint, arXiv:2406.17328.

Yifan Zhang and Team Math-AI. 2024. American invitational mathematics examination (aime) 2024.

Yifan Zhang and Team Math-AI. 2025. American invitational mathematics examination (aime) 2025.

Yifan Zhang and Team Math-AI. 2026. American invitational mathematics examination (aime) 2026.

## Appendix

## A Implementation Details

## A.1 Model Configuration

The main experiments use Qwen3.5-35B-A3B as the strong teacher $\tau$ and Qwen3.5-0.8B as the teacher-family reference R, with both models running in non-thinking mode. The student models are Granite4.1-3B, Qwen3-4B, and OLMo-3-7B-Instruct-SFT.

For teacher-family generalization, we use Qwen3-30B-A3B-Instruct-2507 with Qwen3-0.6B, and Ministral-3-14B-Instruct-2512 with Ministral-3-3B-Instruct-2512, as teacher–reference pairs for Granite4.1-3B. For the teacher––reference scale separation experiments, we fix Qwen3.5-0.8B as R and vary $\tau$ across Qwen3.5-35B-A3B, 9B, 4B, and 2B. All four configurations use the same Granite4.1-3B SFT checkpoint trained on trajectories generated by Qwen3.5-35B-A3B.

During distillation, the strong teacher $\tau$ , the teacher-family reference R, and the frozen copy of the student’s initial policy $\pi _ { \mathrm { r e f } }$ remain fixed. Only the current student policy $\pi _ { \theta }$ is updated. Standard OPD and CompassOPD start from the same SFT checkpoint in every comparison.

## A.2 Cold Start Settings

Cross-family models often differ in response style, format, and reasoning length. Following prior work on vocabulary alignment (Sun et al., 2026), we first perform supervised fine-tuning on the student using reasoning trajectories generated by the strong teacher to reduce these differences before on-policy distillation. The resulting checkpoint initializes both standard OPD and CompassOPD.

We use the OpenThoughts-114k (Guha et al., 2025) dataset, which includes 89,120 math problems, 19,904 code problems, and 4,933 problems from biology, physics, chemistry, and logic puzzles. For each teacher family, we replace the original dataset answers with responses generated by its strong teacher in non-thinking mode. We retain only samples that terminate normally and have non-empty responses.

The cold-start stage uses full-parameter supervised fine-tuning with a maximum sequence length of 32,768 tokens. We use a learning rate of $2 \times 1 0 ^ { - 5 }$ , weight decay of 10<sup>−6</sup>, a cosine learning rate schedule with 10% warmup, and a batch size of 32, and train for one epoch. This stage provides a common initialization adapted to the teacher’s responses. For each teacher–student combination, standard OPD and CompassOPD use the same cold-start data and SFT checkpoint.

## A.3 Distillation Settings

Based on model capability, we use the DAPO-Math dataset for Qwen3-4B, and the DeepScaleR-Preview dataset for Granite4.1-3B and OLMo-3- 7B. Rollouts use temperature 1.0 and top-p = 1.0, with maximum prompt and response lengths of 2,048 and 24,576 tokens, respectively.

The optimizer is AdamW with a constant learning rate of $1 \times 1 0 ^ { - 6 }$ and no warmup. The PPO minibatch size is 64, and the micro-batch size per GPU is 1. The training objective uses the distillation signal without task correctness rewards. Within each comparison, standard OPD and CompassOPD use the same number of training steps and sampled student responses. All experiments are conducted on eight NVIDIA H200 GPUs.

## A.4 Cross-Tokenizer Alignment

For each student-generated response, we score the same text with the strong teacher and the teacherfamily reference. Token sequences across models are monotonically aligned according to shared text boundaries. For one-to-one token matches, we use the corresponding token log-likelihood directly. For text segments represented by different numbers of tokens, we construct a common span and average the token log-likelihoods within that span to reduce scale differences associated with tokenizer granularity.

For CompassOPD, the teacher-family shift is computed from the aligned teacher and reference scores and assigned to every student token within the corresponding span. The student-reference displacement is computed directly at each student token position. Positions where consistent text boundaries cannot be formed are excluded by an alignment mask and do not contribute to the loss. Standard cross-family OPD and CompassOPD use the same alignment procedure.

## A.5 MoE Self-Reference

In the self-reference experiments, T and R share the same Qwen3.5-35B-A3B checkpoint. The strong teacher uses the default MoE routing configuration, activating eight fine-grained experts per token in addition to the shared expert. The reference configuration activates one fine-grained expert while retaining the shared expert, corresponding to approximately 2B activated parameters. The two configurations share all model weights, the tokenizer, and the input context, and differ only in expert activation. We perform two separate forward passes with frozen weights and use the difference between their aligned log-likelihoods as $\Delta _ { T }$

## A.6 Evaluation Protocol

All models use their native chat templates. During evaluation, the maximum generation length is 24,576 tokens, the temperature is $0 . 7$ , top-p = 1.0, $ { \mathrm { t o p } }  { - } k = 4 0$ , and the presence penalty is 2.0.

Each model independently samples 16 responses per problem on benchmarks. For each benchmark, we report accuracy averaged over all sampled responses. The Avg. column reports the unweighted mean of the seven benchmark accuracies. Evaluation uses the automatic scoring pipeline of EvalScope (Team, 2024).

## B Analysis of the CompassOPD Target Policy

We analyze Equation 10 at a fixed context $^ { c , }$ holding $\Delta _ { T }$ and $\pi _ { \mathrm { r e f } }$ fixed. All policies are defined over the same aligned textual action space. Assume $\alpha > 0$ , that $\pi _ { \mathrm { r e f } }$ is positive on this action space, and that the quantities below are finite. Define the normalization factor

$$
Z ( c ) = \sum _ { z } \pi _ { \mathrm { r e f } } ( z \mid c ) \exp \left( \Delta _ { T } ( z , c ) / \alpha \right) .\tag{13}
$$

The policy in Equation 11 satisfies

$$
\log { \frac { \pi ^ { * } ( z \mid c ) } { \pi _ { \mathrm { r e f } } ( z \mid c ) } } = { \frac { \Delta _ { T } ( z , c ) } { \alpha } } - \log Z ( c ) .
$$

Substituting this identity into Equation 10 yields

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { C o m p a s s } } ( \pi ; c ) = \alpha \log Z ( c ) \qquad } \\ { - \alpha D _ { \mathrm { K L } } \left( \pi ( \cdot \mid c ) \| \pi ^ { * } ( \cdot \mid c ) \right) . } \end{array}\tag{14}
$$

Since $Z ( c )$ is independent of $\pi$ and the KL divergence is nonnegative, the objective is bounded above by α log $Z ( c )$ , with equality if and only if $\pi ( \cdot \mid c ) = \pi ^ { * } ( \cdot \mid c )$ . This establishes the optimality of Equation 11.