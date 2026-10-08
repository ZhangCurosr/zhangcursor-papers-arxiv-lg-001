# Force without transmission: a depth-induced rank collapse that no loss on the representation reopens

Martin Hofmann<sup>1</sup> Patrick M¨ader<sup>1,2,3</sup>

<sup>1</sup> Data-intensive Systems and Visualization Group (dAI.SY), Technische Universit¨at Ilmenau, Max-Planck-Ring 14, 98693 Ilmenau, Thuringia, Germany 2 German Centre for Integrative Biodiversity Research (iDiv) Halle–Jena–Leipzig, Deutscher Platz 5e, 04103 Leipzig, Saxony, Germany <sup>3</sup> Faculty of Biological Sciences, Friedrich Schiller University, F¨urstengraben 1, 07745 Jena, Thuringia, Germany

martin.hofmann@tu-ilmenau.de patrick.maeder@tu-ilmenau.de ORCID: 0000-0002-4440-3317 0000-0001-6871-2707

October 2026

## Abstract

Training can drive a transformer into a rank collapse: all token representations point in one direction, and learning stops. In a related collapse of attention, a loss term with a bounded corrective force repairs the network during the run. We ask whether such a term repairs rank collapse. We collapse small transformers by weakening their skip connection and treat copies of the collapsed network. No added loss term repaired the collapse, although the stronger kind pushed with about a tenth of the task gradient. The reason was the path, not the strength. The task gradient no longer reached the query and key weights, which decide where attention looks, and the added term’s gradient faded before the blocks where the collapse forms. Restoring the skip connection, which changes no weight, reopened this path at once. The rank then recovered, but only far above the scale of collapse. After a burst of high learning rate the path stayed open and the rank recovered untreated. Registered predictions from the path ranked recovery times but did not transfer to this cause. In every case the loss stayed above that of a healthy network after the rank recovered. Whether a collapsed network can be repaired depends on whether the gradient still reaches the weights that must change, not on how strongly a loss term pushes.

## 1 Introduction

A transformer keeps tokens apart by passing each token’s own representation forward through a skip connection and adding what attention and the feed-forward layer compute. If the skip connection is weakened enough, attention averages the tokens again and again across the blocks, and their representations converge onto a single direction. This is depth-induced rank collapse (Dong et al., 2021). The network can no longer tell tokens apart, and training does not lead it out: in that state the gradient that would sharpen attention vanishes (Noci et al., 2022). The known defences act before the fact, by changing the architecture or the optimizer (Zhai et al., 2023; Henry et al., 2020; Noci et al., 2022; Ren et al., 2026).

Once a run has collapsed, the architecture is fixed and the cheapest lever left is an extra term in the loss. Whether such a term can repair the collapse is not obvious. For the related collapse of attention entropy the answer depends on what the term measures. A term built on the quantity that has collapsed loses its push exactly there, because its gradient fades to zero; a term built on a spectral quantity keeps a push of bounded size (Zhai et al., 2023; Bardes et al., 2022; Mei et al., 2020; Rusch et al., 2022). We call the size of the gradient a term produces at the weights its force, and the two kinds a vanishing and a bounded force. In attention-entropy collapse a term of the bounded kind does repair an established collapse from saved checkpoints (Hofmann and M¨ader, 2026). This suggests a simple rule: choose a term whose force does not vanish. But a force exerted on the loss acts on the weights only through the backward pass, the same chain of blocks that has collapsed. A large force at the loss is therefore not the same as a force that arrives. We call the part of a gradient that reaches the weights of a given block its transmission, and the route it takes through the stack its path.

We test which of the two decides. We collapse small transformers by weakening their skip connection, save the collapsed network, and continue copies of it under diferent treatments: two loss terms that difer only in whether their force vanishes, an untreated copy, and copies in which the skip connection is restored. We measure how large each force is and, block by block, how much of it reaches the attention weights. We then test, with predictions registered before the runs, whether this measurement of the path tells in advance which networks will recover, both on new networks and under a second cause of collapse, a short burst of high learning rate. A second question concerns time: the theory of rank collapse is stated at initialisation (Dong et al., 2021; Noci et al., 2022; Giorlandino and Goldt, 2026), and we ask whether it matters how long a network has been collapsed before it is treated (compare Ersoy and Wiesner, 2026).

The question matters beyond this failure. Auxiliary losses, regularisers and penalties are routinely added to steer a network, and they are usually judged by their size at the loss. Every such term reaches the weights through the backward pass, so whether it can act depends on the state of that pass. A collapse is a case in which the state of the pass can be changed by one runtime setting and measured directly. The paper makes four contributions:

1. two loss terms on the rank of the token representations that difer only in whether their force vanishes at the collapse, tested as repairs of a settled rank collapse;

2. a block-by-block measurement of where the task gradient and the added gradient arrive inside the collapsed network, before and after the skip connection is restored;

3. a test of how the time spent collapsed changes what a restoration of the skip connection achieves;

4. two registered tests of the path measurement as a predictor of recovery.

## 2 Two kinds of corrective force

To compare a vanishing with a bounded force, we need two loss terms that measure the same thing and difer only in how their gradient behaves at the collapse. The thing measured is how many directions the token representations of a block span. For a block’s token matrix $\bar { X ^ { \mathbf { \alpha } } } \in \mathbb { R } ^ { T \times d }$ , centred over tokens to $\tilde { X }$ with singular values $\sigma _ { 1 } \geq \cdot \cdot \cdot \geq \sigma _ { d }$ , the stable rank srank $( \tilde { X } ) = \| \tilde { X } \| _ { F } ^ { 2 } / \sigma _ { 1 } ^ { 2 }$ , in short the rank, is 1 when all tokens point in one direction and grows with the number of directions they span. With $E = \tilde { X } - \sigma _ { 1 } u _ { 1 } v _ { 1 } ^ { \top }$ the part outside the top direction and $Q = \sqrt { \mathrm { s r a n k - 1 } } = \| E \| _ { F } / \sigma _ { 1 }$ its relative size, the two terms are

$$
R _ { \mathrm { p e n } } ( \tilde { X } ) = 1 - \frac { Q ^ { 2 } } { d - 1 } , ~ R _ { \mathrm { h i n g e } } ( \tilde { X } ) = \operatorname * { m a x } ( 0 , s _ { 0 } - Q ) ,\tag{1}
$$

where $s _ { 0 }$ is the healthy value of $Q$ . The penalty is the rank itself, rescaled to 1 at the collapse. Its gradient is proportional to $Q$ and vanishes as the rank approaches one. The hinge acts only below the healthy value, and its gradient keeps a constant size $1 / \sigma _ { 1 }$ in the direction that would spread the tokens, because $\nabla _ { \tilde { X } } Q = E / ( \| \hat { E } \| _ { F } \sigma _ { 1 } ) - ( \| \hat { E } \| _ { F } / \sigma _ { 1 } ^ { 2 } ) u _ { 1 } v _ { 1 } ^ { \top }$ . Both terms act on every block, averaged over blocks, and each enters the loss with a weight chosen so that it adds the same amount to the loss when it is switched on. A third term, the mean squared cosine between token pairs, is a common choice in the literature and serves as a comparison.

## 3 How the collapse is produced and treated

We need a collapse that is reproducible, settled and harmful, and a way to apply every treatment to the same collapse. We train small language models: pre-LN decoder transformers of width 128 with four heads, context 256 and the GPT-2 vocabulary, at depths 12 and 24 (about 9 and 11 M parameters), with AdamW at a learning rate of $1 0 ^ { - 3 }$ and gradient clipping at norm 1, on a fixed stream of FineWeb-Edu batches (Penedo et al., 2024). Query–key normalisation (Henry et al., 2020) prevents the other kind of collapse, attention entropy collapse; we checked that it did not occur.

Table 1: Treatments started from each saved collapsed network. $s _ { 0 } = \sqrt { \mathrm { s r a n k } _ { \mathrm { h e a l t h y } } - 1 } .$
<table><tr><td>Treatment</td><td>Added loss</td><td>Force class</td><td>Added loss when switched on</td></tr><tr><td>penalty  $1 \times / \ 4 \times$ </td><td> $\beta \left( 1 - Q ^ { 2 } / ( d - 1 ) \right)$ </td><td>vanishing</td><td> $\Lambda ~ / ~ 4 \Lambda$ </td></tr><tr><td>hinge  $1 \times / \ 4 \times$ </td><td> $\beta \operatorname* { m a x } ( 0 , s _ { 0 } - Q )$ </td><td>bounded</td><td> $\Lambda ~ / ~ 4 \Lambda$ </td></tr><tr><td>cosine 1×</td><td> $\beta U$ </td><td>neither</td><td>Λ</td></tr><tr><td>untreated</td><td></td><td>control</td><td></td></tr><tr><td>reversed ramp</td><td></td><td>driver reversed</td><td></td></tr></table>

Each block computes $x  \alpha x + \mathrm { A t t n } ( \mathrm { L N } ( x ) )$ and $x  \alpha x + \mathrm { M L P } ( \mathrm { L N } ( x ) )$ , so α scales the skip connection. It is 1 for the first 1,500 steps and then ramps down to $\alpha _ { \mathrm { m i n } } = 0 . 0 2$ by step 3,000. We call this ramp the driver of the collapse. The rank of the last block is measured every ten steps on a fixed probe batch. A network counts as collapsed below a rank of 1.5 and is saved once the collapse has settled. It counts as recovered when the 50-step median of its rank reaches half the healthy value; recovery in this paper always means recovery of the rank, and the loss is reported separately. Each treatment continues the saved network for 4,000 steps. The details of calibration and detection are in appendix A.

The treatments (table 1) are the penalty and the hinge at one and four times a matched added loss of $\Lambda = 0 . 1$ nats, the cosine term, an untreated copy, and the reversed ramp, in which α returns to 1 at the rate it came down. Setting α back is a runtime action: it changes no weight, only how much of each block’s input passes through. Two depths and five seeds give ten collapse events; a replication with five new seeds per depth followed under the same rules. The treatments and their evaluation were fixed before the main runs. After them we added branches that set α back to a fixed value at once or after a delay, the block-by-block gradient measurements, and two tests whose predictions were committed to the repository before their runs; all of these are marked where they appear.

## 4 The collapse is harmful and does not heal

Before asking what repairs the collapse, we need to know that it is damage and that it stays. When it was saved, the collapsed network’s loss lay 1.73–1.94 nats above that of a healthy network trained on the same data, at 7.55–7.63 nats, close to the 7.64 nats of a model that only knows how often each token occurs. Left untreated for 4,000 steps, the rank stayed below the recovery threshold in $3 / 3$ collapse events at depth 12 and $5 / 5$ at depth 24, and in all 10 events of the replication. Attention in the collapsed state was close to uniform: every token attended to all tokens almost equally (normalised entropy 0.94–0.99, against 0.65–0.69 in healthy networks). The collapse was a stable, harmful state that training alone did not leave.

## 5 No added loss repairs it, although its force is large

The first question is whether either kind of loss term repairs the collapse. All loss-based branches ran with the skip scale held at $\alpha _ { \mathrm { { m i n } } } ,$ so each term worked against the cause of the collapse. None recovered the rank: penalty $0 / 8$ , hinge $0 / 8 ,$ penalty $4 \times 0 / 8 ,$ hinge $4 \times 0 / 8$ and cosine $0 / 8$ (figure 1), and none of the loss-based branches of the replication. Exempting the added gradient from clipping, at up to ten times the strength, changed nothing (0 of 32 branches recovered). The loss terms also did not merely hasten a recovery that would have come anyway, because the untreated copies did not recover either.

A failed repair could mean that the force was too weak. We therefore measured it at the parameters, as the ratio ρ of the added gradient to the task gradient. The penalty’s force faded as designed, and the hinge’s stayed: the median per collapse event was 0.0032 for the penalty and 0.0887 for the hinge, about 28 times more, and 0.3071 for the hinge at four times the strength (appendix $\operatorname { B } ;$ replication 0.0037 and 0.0782). The distinction between the two kinds of force thus held for rank collapse, and the bounded force was large. It still did not repair the collapse.

![](images/7e5fe9aac65e71a976995e6a3d67de1e6297caf791376fe9e3bcded0da9e7ce8.jpg)  
Figure 1: Stable rank of the last block after the treatment is switched on, all treatments and collapse events of the main study (50-step median, as in the recovery rule; dotted line: recovery threshold). Only the reversed ramp recovers.

## 6 The force does not reach the weights that must change

If the force is large but has no efect, it may not arrive where it is needed. For tokens to become distinct again, attention must stop averaging them, so the weights that decide where attention looks, the query and key projections, have to change. We measured, block by block, how much of the task gradient and of the hinge’s gradient reached the attention weights of each block, on all 18 saved collapsed networks of the main study and the replication, without any training (figure 2, top). The healthy reference is the network of the same seed trained without the driver.

The task gradient arrived weakly at the blocks nearest the output and hardly at all at those nearest the input. On a block’s attention weights (query, key and value) it was $8 \cdot 1 0 ^ { - 3 } $ in the last block and $1 0 ^ { - 8 }$ in the first at depth 12, against about $7 \cdot 1 0 ^ { - 2 }$ per block in a healthy network; at depth 24 it ran from $5 \cdot 1 0 ^ { - 3 } ~ $ in the last block to $2 \cdot 1 0 ^ { - 1 1 }$ in the first, and it fell by more than three orders of magnitude from the last block to the first in 17 of 18 networks. Almost all of it sat on the value projection. The query and key weights received less than $1 0 ^ { - 6 }$ in every block of 17 of 18 networks, against $3 \cdot 1 0 ^ { - 2 } \mathrm { ~ t o ~ } 5 \cdot 1 0 ^ { - 2 }$ per block in healthy ones. This is the result that Noci et al. (2022) derive at initialisation, here in trained networks: when all tokens look alike, changing where a token attends changes nothing, so no gradient asks for it.

The hinge’s gradient had the opposite profile. It was 0.34 in the first block at depth 12, more than the healthy task gradient, and $6 \cdot 1 0 ^ { - 4 } $ in the last, because each block’s term can only act through the diversity its own input still has. The blocks that complete the collapse, near the output, therefore received almost nothing from either side. The hinge branches show the consequence: at the end of the budget the rank was 8–9 in the first block but 1.6–2.7 in the middle one, so the diversity the hinge restored near the input was lost again on the way to the output.

depth 12: collapsed state (α = 0.02), 8 networks  
![](images/52f90416de529593d574d8f348bf7fb5ed5c4b056f392d2ae8d6254a29b26ff7.jpg)

depth 24: collapsed state (α = 0.02), 10 networks  
![](images/3c754d4c4a4fa14942f0c098be0a7c9dc49c23492443b6fcb4be4a9b98375434.jpg)  
task loss, healthy network (α = 1)

depth 12: task loss, query and key, after the switch  
![](images/df2de8d307e4d82c74a15992a7d1bd77abe5f96a4c07595b446cdcaf48ae8655.jpg)

depth 24: task loss, query and key, after the switch  
![](images/9836a623fcd0939c24a424912c5b934ed865aa087c5e9591c6f1d5ba75c3d62d.jpg)  
α = 0.3, before any update α = 1, before any update healthy network (α = 1)

Figure 2: Gradient path in the saved collapsed networks of the main study and the replication (lines: median over networks; bands: range). Top: per-block gradient norm at the collapsed state for the task loss on the attention input weights (solid) and on the query and key part alone (dotted), for the per-block hinge at matched added loss (dashed), and for the task loss in the healthy network of the same seed (black). Bottom: task gradient on the query and key weights after the skip scale is set to 0.3 or 1, at the first step before any update (solid) and after 50 updates of the branch (dashed). Values below $1 0 ^ { - 1 3 }$ are numerically zero in single precision and are drawn at the lower edge.

## 7 Restoring the path recovers the rank, with hysteresis

If the path is what fails, reopening it should help where added force did not. Restoring the skip connection does exactly that without touching a weight. On the reversed ramp every network recovered its rank, $8 / 8$ in the main study and 10/10 in the replication. Setting α back to a fixed value at once, a switch, reopened the path immediately (figure 2, bottom): at the first step after α was set to 1, before any update, the task gradient on the query and key weights rose by a factor of about $1 0 ^ { 7 }$ even in the block where it rose least, and in the median block it reached 1.2 times the healthy value. The rank itself had not yet recovered at that moment in most networks; the recovery was learning made possible by the reopened path.

Recovery did not happen where the collapse had begun. The collapse appeared when α reached 0.02–0.03, but on the way back the rank recovered only at α between 0.25 and 0.97 at depth 12 and at α = 1.00 at depth 24 (figure 3; replication 0.95–1.00 and 1.00), at least 12 times above the point of collapse in all 18 events. This is a hysteresis. Because the ramp also lets time pass, we checked it by setting α to a fixed value at once: at 0.3 and 0.6, 7 of 8 and 8 of 8 networks recovered within 34–121 steps, but at 0.1, five times the scale of collapse, only 2 of 8 did, and 5 of 10 in the replication. On the way down every network had still been healthy at that scale. A restoration therefore has to go well beyond the point where the collapse began.

![](images/35569b34e795c4d7667cdfbbfe0d17bb367d07f21e72b62227aea828cd16f4a3.jpg)  
Figure 3: Stable rank of the last block against the skip scale α while it is ramped down (drivers) and, from the saved collapsed state, while it is ramped back up at the same rate (main study). Dotted line: recovery threshold.

Time mattered as well. The same switch to 0.3 applied after 500, 1 500 or 3 000 further steps in the collapsed state recovered later or not at all: at depth 12 the time to recovery rose with the delay in every seed, and at depth 24 no delayed branch recovered in the main study or in the replication, where this was a registered prediction (appendix C). What changes with time was not identified; the attention pattern did not.

Restoring the rank did not restore the loss. All 42 networks that recovered by rank after the skip was restored ended the budget 0.21–0.56 nats above the healthy network at the same step (median 0.44). Part of this gap may be the training time lost while collapsed.

## 8 Does the path predict recovery?

If the path decides, measuring it should tell in advance which networks will recover. We tested this twice, each time committing the predictions and the evaluation to the repository before the runs; the details are in appendix E. The predictor was the median over blocks of the task gradient on the query and key weights, read after the skip is restored and before any update.

On the ten replication networks the predictor ranked the recovery times after a switch to 0.3 as registered (Spearman −0.67 over 9 recovered networks). Its threshold, fitted on the main study, classified 8 of 10 outcomes at the weaker switch to 0.1, one short of the registered target. A simple state measure did as well: the loss gap, the excess of the collapsed network’s loss over that of a healthy one.

The second test changed the cause of the collapse. On 8 new healthy depth-24 networks the learning rate was multiplied by 16, or by 32, for 200 updates with the skip intact, and each network then continued untreated. The rank fell below the collapse threshold within ten updates, but the path stayed open: the query and key gradient at the end of the burst was $1 . 9 \cdot 1 0 ^ { - 4 }  – 7 . 1 \cdot 1 0 ^ { - 4 }$ at 16, far above the networks collapsed by a weakened skip (figure 4, left). At 16 all 8 networks recovered their rank without treatment; at 32 none did within the budget, although their rank rose throughout (figure 4, right). The networks collapsed by a weakened skip did not move in the same time. The threshold from the first test did not carry over (3 of 8 correct at 32), and within the networks at 16 the predictor did not order the recovery times (Spearman −0.10; the registered main test was missed), whereas the loss gap did (+0.79). Training stayed numerically stable, and the networks that recovered their rank ended 0.86–1.04 nats above a copy of the same network trained without the burst.

![](images/aea8efb20e49551c6333e692aafdc6180e4e6740e0a25f0f9c26b12be93f1dbb.jpg)

![](images/7f7c174cd239ff05ff0bb44651843a841a887a72990409d1a065aaff8a79875b.jpg)  
Figure 4: Registered test under a learning-rate excursion with the skip intact. Left: query–key gradient at the saved state (median over blocks) for the depth-24 networks collapsed by a weakened skip (section 6), the held-out excursion networks at multipliers 16 and 32, and healthy depth-24 networks; filled: recovered by rank, open: not, triangles: development networks (appendix E); dashed: threshold frozen in the first registered test; dotted: development threshold. Right: stable rank of the last block after the saved state, excursion networks against the untreated depth-24 networks of the main study (50-step median; dotted: recovery threshold).

## 9 Discussion

What decided repair. The added loss terms failed although their force was large, and they failed because almost none of it reached the weights that had to change. Restoring the skip connection changed no weight but reopened the path, and the network then learned its way out. When the collapse came from a burst of high learning rate instead, the path stayed open and the rank recovered without help. In attention-entropy collapse a loss term of the bounded kind does repair (Hofmann and M¨ader, 2026); there the path was not measured. Together the cases separate the size of a corrective force from whether it arrives. In rank collapse arrival was the binding constraint.

What carries beyond this collapse. Any term added to a loss acts on the weights only through the backward pass. Its size at the loss says little about its efect if that pass no longer reaches the weights the term is meant to move. Measuring the gradient where it has to act, block by block, is cheap and needs no training; here it distinguished the collapse driven by a weakened skip, in which the path was closed, from the rank loss after a learning-rate burst, in which it stayed open. It did not, however, predict recovery better than a simple measure of the loss in our registered tests, so it is a diagnostic of the mechanism rather than a better forecast.

A recovered diagnostic is not a recovered function. Every network in this paper that recovered by rank still had a loss above its control at the end of the budget, after the skip was restored by 0.21–0.56 nats and after the learning-rate burst by 0.86–1.04. The rank is the measure on which collapse and recovery are defined here, and it reported recovery while the function had not recovered. Whether a run has recovered should be judged by continuing it and measuring its loss. For the learning-rate burst, a rollback to the checkpoint before it would have cost 200 steps.

## 10 Scope and limitations

This paper does not claim that loss-based repair fails for rank collapse in general. The collapse of the main study is driven by a weakened skip connection; with the skip intact, a weight-decay ramp and a larger initialisation gain produced no collapse, and the learning-rate burst lowered the rank without closing the path. The networks have 9 to 11 million parameters and are trained on one data stream. The paper does not claim that the path measurement predicts recovery better than simple state measures, that a network that recovered by rank has recovered its function, or what changes with time in the collapsed state. The gradient path was measured on one probe batch per network, the registered tests rest on ten and eight held-out networks of one system, and the design was fixed before the main runs but not independently timestamped. Alternative explanations we considered are discussed in appendix F.

## 11 Conclusion

A loss term can only repair what its gradient reaches. In a depth-induced rank collapse the corrective force of a well-designed term was large, but almost none of it reached the attention weights of the blocks that complete the collapse. Restoring the skip connection reopened that path and the network recovered its rank, only well above the scale of collapse, without its loss returning to that of a healthy network. Whether a collapsed network came back depended on how much of the gradient’s path was restored, not on what was added to the loss.

## Author contributions

M.H. conceived the study, wrote the protocol and the code, ran and analysed the experiments and wrote the manuscript. P.M. supervised the work and revised the manuscript.

## Reproducibility statement

Code, preregistration, protocol, frozen constants, analysis scripts and the scripts that generate this manuscript are kept in one version-controlled repository and are available from the authors on request. Freeze commit e 2be93a7052c2998f52543db78f0457815d82569; preregistration SHA-256 919b08148f80196daa97a51fc8fb b5cccdea729a204596cdf29da03ff8744d61; calibration summary SHA-256 0c735b79cff13134a45c8ead5b 130a4431aa8ee605b023dca1e7a49f4d14175b. Every measured quantity in this paper is a macro rendered from the analysis outputs; tables and figures are produced by the same scripts, and an audit checks hashes, placeholders and unbound decimals before every build. Training ran on NVIDIA A100-SXM4-40GB GPUs (54 GPU-hours of wall time for calibration and main study at the cluster share available); the gradient-path measurements ran on CPU from the saved states. The data stream is the FineWeb-Edu sample prepared with the GPT-2 tokenizer and a SHA-256 manifest by a sibling project. The registrations of section 8 and appendix E were committed before the runs they predict (commits 6358641, b4b0963 and bf90737).

## Use of AI-assisted tools

Claude Code (Anthropic; Claude Fable 5.1 and Claude Opus 5.5, September and October 2026) assisted with research code, run orchestration, analysis scripts, literature lookup and drafting. The authors designed and verified the study and are responsible for its interpretation, citations and final text.

## References

Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations (ICLR), 2022. arXiv:2105.04906.

Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pages 2793–2803. PMLR, 2021.

Ibrahim Talha Ersoy and Karoline Wiesner. Noise-driven escape from metastable phases explains grokking in deep neural networks. In HiLD 2026: 4th Workshop on High-dimensional Learning Dynamics, 2026. arXiv:2606.17120.

Alessio Giorlandino and Sebastian Goldt. Two failure modes of deep transformers and how to avoid them: a unified theory of signal propagation at initialisation. In International Conference on Learning Representations (ICLR), 2026. arXiv:2505.24333.

Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4246–4253, 2020. doi:10.18653/v1/2020.findings-emnlp.379.

Martin Hofmann and Patrick M¨ader. Repairing attention collapse through the loss with a spectrally gated penalty, 2026. Submitted.

Jincheng Mei, Chenjun Xiao, Bo Dai, Lihong Li, Csaba Szepesv´ari, and Dale Schuurmans. Escaping the gravitational pull of softmax. In Advances in Neural Information Processing Systems 33 (NeurIPS), pages 21130–21140, 2020.

Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signa propagation in transformers: Theoretical perspectives and the role of rank collapse. In Advances in Neural Information Processing Systems 35 (NeurIPS), pages 27198–27211, 2022. doi:10.52202/068431-1972.

Guilherme Penedo, Hynek Kydl´ıˇcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Rafel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems 37 (NeurIPS), Datasets and Benchmarks Track, pages 30811–30849, 2024. doi:10.52202/079017-0970.

Lianhai Ren, Yucheng Ding, Xiao Liu, Peng Cheng, and Yeyun Gong. MSign: An optimizer preventing training instability in large language models via stable rank restoration. arXiv preprint arXiv:2602.01734, 2026.

T. Konstantin Rusch, Benjamin P. Chamberlain, James Rowbottom, Siddhartha Mishra, and Michael M. Bronstein. Graph-coupled oscillator networks. In Proceedings of the 39th International Conference on Machine Learning (ICML), volume 162 of Proceedings of Machine Learning Research, pages 18888–18909. PMLR, 2022.

Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and Joshua M. Susskind. Stabilizing transformer training by preventing attention entropy collapse. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine Learning Research, pages 40770–40803. PMLR, 2023.

## A Design, calibration and detection

Driver. The floor $\alpha _ { \mathrm { m i n } } = 0 . 0 2$ is the largest value on a calibration grid at which every seed collapsed and no untreated network recovered. At $\alpha _ { \mathrm { m i n } } = 0 . 0 3$ every depth-12 network kept its rank, while at depth 24 some seeds lost it, one of them only briefly; above 0.03 none did. The driver is a knife-edge (figure 5).

Detection, settling and recovery. The collapse threshold is a stable rank below 1.5 on the last block. Because the collapse oscillates for a few hundred steps after it first appears, a network is declared settled, and saved, at the first measurement at least 100 steps later whose ten-measurement median is still below 1.5. Recovery is declared when the 50-step median of the stable rank reaches half its healthy value (3.95 at depth 12, 3.86 at depth 24). Of the ten main-study collapse events, 8 settled and 2 (both at depth 12) dipped below the threshold for one to three measurements and healed by themselves; those two were never treated. Healthy networks dip below the collapse threshold early in training (24 of 26 control runs, all within the first 160 steps) and reach the recovery threshold by step 550; the ramp starts late enough that this transient is never mistaken for the collapse.

The collapsed state. The collapsed network is not simply the healthy network with a small skip. The healthy weights evaluated at $\alpha = 0 . 0 2$ give a loss of 24–59 and a gradient norm near 265, whereas the collapsed weights sit at the frequency-table loss with a gradient norm of order one.

Loss terms. The hinge’s gradient uses only the top singular pair, which is well defined as long as $\sigma _ { 1 }$ is separated from $\sigma _ { 2 } ;$ since $\sigma _ { 2 } \le \| E \| _ { F } = \sigma _ { 1 } Q$ , it is separated at the collapse. The loss actually added on the first training step was within a tenth of its target in $3 1 / 4 0$ loss-treated branches. The cosine term’s gradient is neither fading nor constant near the collapse. During calibration, none of 23 checks with the hinge on the last block only, on every block, or at ten times Λ recovered.

Calibration used seeds 101–103. At $\alpha _ { \mathrm { m i n } } = 0 . 0 3 \ : 2 / 6$ runs crossed the collapse threshold, one of which settled, and none above did; 0.02 and 0 collapsed $6 / 6$ and $6 / 6$ , and $1 2 / 1 2$ untreated branches stayed collapsed. Changes made during calibration and before the main runs: the ramp start moved from step 200 to 1,500 because of the healthy network’s early dip; settling and recovery use medians because the collapse oscillates; the loss terms act on every block rather than the last one only, because a last-block term failed identically for both classes; the $\alpha _ { \mathrm { m i n } }$ grid was widened to locate the knife-edge; the cosine term was labelled class-less; the interpretation row for “both treatments fail” was added. Additions after the main runs are reporting only: the settled/self-healing classification, the $\rho$ table, the per-block gradient measurements on all saved collapsed networks and after the switch, and the post hoc branches. The branches on the replication networks in appendix E were registered before they ran.

![](images/b69c141fa5c035bba84380887c0a94cf62709cbf7641a88d5260927289209773.jpg)

![](images/8f92e663c850c73c5a8b1f17a65e1519e10c8d1c493bf035a5bf0b820f0038d0.jpg)  
Figure 5: Left: depth-12 drivers over the $\alpha _ { \mathrm { { m i n } } }$ grid; at this depth only $\alpha _ { \mathrm { m i n } } \leq 0 . 0 2$ collapses, and it does so when α reaches the floor. Right: the healthy network $( \alpha = 1 )$ dips to the detection threshold early in training and heals by itself.

## B Force measurement

On a synthetic matrix $X = a v ^ { \top } + \delta E$ , in which every token lies on one direction v and a random matrix E scaled by δ sets the distance from rank one, the penalty’s gradient falls from $4 . 9 \cdot 1 0 ^ { - 4 }$ at δ = 1 to 3.5 · 10<sup>−8</sup> at $\delta = 0 . 0 0 0 1$ , while the hinge’s stays at 0.022 (figure 6, left). In the audit that preceded the main runs the ratio ρ at a collapsed calibration checkpoint was 0.0013 for the penalty and 0.074 for the hinge. The cosine term sat between the two (0.0618). Table 2 gives the ratio per treatment.

![](images/0665b6c1b4626a1e1c9c44c6e5467cacff1341b131bf20f8ffcfaab1932565bc.jpg)

![](images/93ae48e2a8b3a282e4f905265410889fc716c475884b8a5367b9b8a6f7f1b4c7.jpg)  
parameter distance from the collapsed checkpoint

![](images/748e8959f392eab3dc9302bac5e6c3e1b9753ada9aabc42ed782a7f989b3eba9.jpg)  
Figure 6: Force of each loss term. Left: gradient norm with respect to the synthetic token matrix as its distance δ from rank one shrinks; the penalty fades linearly, the hinge stays constant, the cosine term is non-monotone because its row normalisation diverges for tokens near the mean. Middle: ratio ρ of added-loss to task gradient at the parameters of a collapsed calibration checkpoint and at points along the straight line towards the healthy network of the same seed and, as a control, the same distance along a random direction. Right: $\rho$ per treatment over the eight treated collapse events.

Table 2: Ratio $\rho$ of added-loss to task gradient norm per treatment (median over measurements per collapse event; median and range over the eight events) and the fraction of steps on which gradient clipping was active.
<table><tr><td>Treatment</td><td>events</td><td> $\rho$  median</td><td>range over events</td><td>clip fraction</td></tr><tr><td>penalty 1×</td><td>8</td><td>0.0032</td><td>0.0027-0.0043</td><td>0.22</td></tr><tr><td>hinge 1×</td><td>8</td><td>0.0887</td><td>0.0550-0.1630</td><td>0.44</td></tr><tr><td>penalty 4×</td><td>8</td><td>0.0130</td><td>0.0111-0.0170</td><td>0.21</td></tr><tr><td>hinge 4×</td><td>8</td><td>0.3071</td><td>0.1942-0.6194</td><td>0.60</td></tr><tr><td>cosine 1×</td><td>8</td><td>0.0618</td><td>0.0380-0.0881</td><td>0.40</td></tr></table>

## C Post hoc checks

The following checks were run after the main analysis and are labelled post hoc. They use the recovery rule fixed before the main runs but were not part of the design, and they are not pooled with the main result. Branches run: immediate switch 24 of 24, delayed switch 24 of 24, unclipped added loss 32 of 32; intact-skip drivers 24 of 24.

Immediate switch. Branches of the eight saved collapsed networks with the skip scale set to a fixed value for the full budget:
<table><tr><td>α held</td><td>recovered / n median</td><td> $T _ { \mathrm { r e c } }$  (recovered)</td><td>final srank range</td></tr><tr><td>0.1</td><td>2/8</td><td>94</td><td>1.00-7.09</td></tr><tr><td>0.3</td><td>7/8</td><td>71</td><td>1.48-7.18</td></tr><tr><td>0.6</td><td>8/8</td><td>50</td><td>6.40-7.42</td></tr></table>

Delayed switch. Branches kept at $\alpha _ { \mathrm { m i n } }$ for a further 500, 1 500 or 3 000 steps before the switch to $\alpha = 0 . 3$ (the same switch as in the undelayed branches above):
<table><tr><td>steps at  $\alpha _ { \mathrm { m i n } }$  before the switch to 0.3 recovered / n median</td><td></td><td> $T _ { \mathrm { r e c } }$  after the switch</td><td>final srank range</td></tr><tr><td>500</td><td>2/8</td><td>150</td><td>1.04-6.86</td></tr><tr><td>1500</td><td>3/8</td><td>290</td><td>1.00-6.95</td></tr><tr><td>3000</td><td>1/8</td><td>3770</td><td>1.02-3.60</td></tr></table>

![](images/19751f730552402a1dee992451f9b9dbf75743a586293f2cb60c7c1ae9003ea4.jpg)  
Figure 7: Filled: main study, post hoc. Open: the same switches on the replication networks, registered before they ran. Steps to recovery after the switch to $\alpha = 0 . 3$ against the steps spent at $\alpha _ { \mathrm { { m i n } } }$ before the switch (0: undelayed). Crosses at the top: no recovery within the budget.

Added-loss gradient exempt from clipping. With the task gradient clipped alone and the added-loss gradient added unclipped: penalty $1 \times : 0 / 8$ recovered (final srank 1.02–1.99); hinge $1 \times : 0 / 8$ recovered (final srank 1.00–2.60); hinge $4 \times : \ 0 / 8$ recovered (final srank 1.01–3.63); hinge $1 0 \times : \ 0 / 8$ recovered (final srank 1.41–3.40).

Intact-skip drivers (exploratory). Attempts to produce a collapse with the skip connection intact $( \alpha \equiv 1 )$ , with an uncentred stable rank as the detector, over 9000 steps each: weight-decay ramp to 10: detected $0 / 6 ,$ , settled $0 / 6 ;$ initialisation gain 4: detected $0 / 6 ,$ settled $0 / 6 ;$ initialisation gain 8: detected $0 / 6 ,$ settled $0 / 6 ;$ initialisation gain 16: detected $0 / 6$ , settled $0 / 6$ . Neither driver produced a settled collapse, so no branches were made from them.

Recovery after delay. This section reports a post hoc observation of the main study. The same switch to $\alpha = 0 . 3 $ , applied after the branch has spent a further 500, 1 500 or 3 000 steps at $\alpha _ { \mathrm { { m i n } } }$ , took longer or failed (figure $7 ;$ pooled medians and final stable ranks in the table above). At depth $1 2 , 3 / 3$ recovered without delay, $2 / 3$ after 500 steps (in 130–170 steps), $3 / 3$ after 1 500 (in 240–440) and $1 / 3$ after 3 000 (at 3770). Within each seed the recoveries took longer the longer the delay. At depth 24 no delayed branch recovered $( 0 / 5 , 0 / 5$ $0 / 5 )$ , although $4 / 5$ recovered without delay. With so few events per delay and depth the counts alone are weak evidence. The observation rests on the depth-12 recovery times, which rose with the delay without overlap among the events that recover, and on the depth-24 contrast between the undelayed switch and every delayed one. A delayed full restoration was not run. The reversed ramp is partly a delayed case: by the time it has raised α back to 0.3 the network has spent about 430 further steps collapsed.

What changes with time is not the attention pattern: its normalised entropy was unchanged after 3 500 untreated steps (0.94–0.99).

## D Replication with new seeds

The same design, rules and data order were run once more with five new seeds per depth. The results are reported next to the main study, not pooled with it. Of 10 collapse events, 10 settled and none healed by itself. Again no loss-based treatment recovered the rank within the budget, and neither did the untreated network (0 of 10 in each of the 6 treatments). At the parameters the penalty delivered a force of 0.0037 of the task gradient and the hinge 0.0782 (medians over collapse events). On the reversed ramp every network recovered, but only after α had risen far past the value at which it collapsed: at α between 0.95 and 1.00 at depth 12 (5 of 5) and at $\alpha = 1 . 0 0$ at depth 24 (5 of 5). The outcome fixed in advance for the two loss terms, both failing while the untreated network stays collapsed, was reproduced.

## E Registered tests in detail

Path after the switch. The rule fixed for this check before it was run, a query–key gradient of at least $1 0 ^ { - 2 }$ in every block, was met at $\alpha = 1$ in 15 of 18 networks and at $\alpha = 0 . 3$ in 1. The healthy networks themselves met it in 12 of 18, so the rule was stricter than the healthy state. On the attention input weights, the quantity of the profile in section $6 ,$ it was met in 18 of 18 networks at $\alpha = 1$ and in 9 at $\alpha = 0 . 3$ . The switch by itself restored the rank only in part: before any update the stable rank of the last block lay above the recovery threshold in 0 of 18 networks at $\alpha = 0 . 3$ and in 4 at $\alpha = 1$ . The rest of the recovery was learning made possible by the reopened path. After 50 updates the rank lay above the recovery threshold in 9 of 18 networks at $\alpha = 0 . 3$ and in 8 at $\alpha = 1$ . These updates were replayed from the saved weights, optimizer state and data position. Where the logged main-study branches were measured at the same steps, the replayed stable rank agreed with the logged one to 2 % in the median.

First test: the replication networks. The gradient path of section 7 is read before any update, so it can predict what a restoration will do. We tested this on the ten saved networks of the replication, on which no switch had been run. Before any branch ran, the predictor, four predictions and the evaluation script were committed to the repository. The predictor is the median over blocks of the task gradient on the query and key weights at the first step after the switch, before any update. For the switch to $\alpha = 0 . 1$ its threshold was fitted on the eight main-study networks, which it separates without error $( 8 / 8 )$ , and frozen at $3 . 1 \cdot 1 0 ^ { - 5 }$

The predictions were these. P1: at $\alpha = 0 . 1$ exactly the networks above the threshold recover; met if at least 9 of 10 are classified correctly. P2: at $\alpha = 0 . 3$ the predictor ranks the recovery times, with a Spearman correlation of −0.4 or lower $( - 0 . 7 9$ on the main study). P3: the force of the hinge, its ratio $\rho ,$ does not predict these outcomes. P4: rival measures of the state predict worse: the loss gap at detection, the task gradient norm at the saved state, and depth alone. Depth matters because the predictor is larger at depth 12; the predictor and depth disagree on two of the ten networks.

P1 was missed by one network: 8 of 10 were classified correctly (figure 8, left). At $\alpha = 0 . 1 , 3$ of 5 networks recovered at depth 12 and 2 of 5 at depth 24. $\mathrm { S o } ,$ as in the main study (section 7), half or more of the networks stayed collapsed at five times the scale of collapse without any delay. One error is the depth-12 network whose query–key gradient was not near zero in the collapsed state (section 6); it lay above the threshold and did not recover. The other is a depth-24 network below the threshold that recovered. On the two networks where the predictor and depth disagree, the predictor was right both times; depth alone classified 6 of 10. P2 was met: 9 networks recovered at $\alpha = 0 . 3$ , and the Spearman correlation between predictor and recovery time was $- 0 . 6 7$ (figure 8, right). P3 was missed. The hinge force classified 5 of 10 at $\alpha = 0 . 1$ , but it correlated with the recovery time at $+ 0 . 4 5$ , above the registered bound of 0.4 in magnitude: a larger force went with slower recovery. P4 was missed. The gradient norm did worse (4 of 10, Spearman $- 0 . 0 8 )$ , but the loss gap at detection did as well as the path predictor (8 of 10, Spearman −0.68).

The delayed switch of section 7 was registered for the same networks. The prediction was no recovery at depth 24 at any delay, and at depth 12 recovery times that do not fall and recoveries that do not return as the delay grows. Both held (figure 7). At depth 24 none of the 15 delayed branches recovered, although 4 of 5 recovered without delay. At depth 12, 5 of 5 recovered without delay, 3 after 500 steps (in 110–180 steps), 2 after 1 500 (in 260–300) and 0 after 3 000, and within each seed the time to recovery rose with the delay.

Second test: a learning-rate burst. The predictor of the first test and its threshold were fixed on networks whose collapse was driven by a weakened skip. We tested whether both transfer unchanged to a diferent driver. On 8 healthy depth-24 networks that had not been used before, the learning rate was multiplied by 16, and in a second condition by 32, for 200 updates with the skip intact. Each network was saved at the end of the excursion and continued untreated for 4,000 steps; a paired branch at the base rate was the control. The query–key gradient was measured at the saved state, and the prediction for each network was committed before its continuation ran. Two further networks, seeds 2 and 4, served for development. They set the second condition, development thresholds for the path and for two state measures, and the targets. The two state measures, taken at the saved state, are the loss gap, here the probe loss above that of the paired control, and the norm of the task gradient. The registration fixed seven tests. T0 asks whether the path is open, T1 whether the frozen threshold classifies recovery, T1b and T2 whether the development threshold does and whether it beats the state measures and the multiplier alone. T3, the main test, asks whether the path orders the recovery times within one multiplier, and T3r whether the state measures order them worse. T4 asks whether networks that recover by rank are still harmed, that is, end the budget with a loss on unseen data more than 0.1 nats above the paired control.

![](images/6ed89db868738cf776411afefb888ce9db62bc4a82ae64f9000e9d5b4d98dcad.jpg)

![](images/98dc4ff67bb4773bf3040dfb340295b40ea6f43af4057e9e7c22e98cd7d3e8a0.jpg)  
Figure 8: Registered prediction on the ten replication networks. Left: outcome of the switch to $\alpha = 0 . 1$ against the predictor; dashed line: threshold fitted on the main study (grey squares) and frozen before the branches ran; crosses: wrong predictions. Right: steps to recovery after the switch to α = 0.3 against the predictor at that scale.

The path stayed open. At the saved state the query–key gradient was $1 . 9 \cdot 1 0 ^ { - 4 } – 7 . 1 \cdot 1 0 ^ { - 4 }$ at multiplier 16 and $1 . 7 \cdot 1 0 ^ { - 5 } – 1 . 5 \cdot 1 0 ^ { - 4 }$ at 32, far above the networks collapsed by a weakened skip and below healthy ones (figure 4, left). All 8 networks at 16 lay above the frozen threshold (T0, target 7 of 8, met). At 16 all 8 recovered by rank, after 1110–3540 steps. At 32 none did within the budget, although the rank rose throughout and ended at 2.7–3.4 (figure 4, right). Training stayed numerically stable at both multipliers; the harm was functional, not a divergence. The networks collapsed by a weakened skip did not move in the same budget.

The threshold did not transfer. It classified 8 of 8 networks correctly at 16 and 3 of 8 at 32 (T1, target 7 per condition, met and missed as registered). The development threshold classified 16 of 16 (T1b, target 13, met). But so did the multiplier alone, and the development thresholds of the loss gap and the gradient norm classified 14 and 12. A strict win over every rival was therefore missed (T2). Within the condition at 16, where the time to recovery varied, the query–key gradient did not order it: the Spearman correlation was −0.10 over 8 recovered networks (T3, the main test, target −0.4 or lower, missed). The loss gap ordered it better (+0.79; a larger gap went with slower recovery), and so did the gradient norm (+0.55; T3r, missed). Over both conditions together every measure ranked the outcome (query–key gradient −0.82, loss gap +0.89, multiplier +0.93), because each separates the two conditions.

The rank recovered, the loss did not. All 8 networks that recovered by rank ended the budget 0.86–1.04 nats above the paired control (T4, target three quarters, met). Under this driver the path stayed open and the rank recovered untreated, as the path account predicts. The threshold fitted on the skip system did not carry over, and the path measurement did not order recovery better than the loss gap. The excursion left a loss that the rank does not show.

## F Alternative explanations

• That the loss-based treatments fail for a reason of protocol rather than mechanism. The unclipped branches of section 5 answer this.

• That the hysteresis is an efect of time spent collapsed rather than a shifted skip scale of recovery. The immediate switches answer this: at 0.3 and 0.6 most networks recover within about a hundred steps, and at 0.1 fewer than half recover although no time has passed.

• That the force measurement is true by construction. In representation space it is. The ratio at the parameters is not, and a single collapse event with a large penalty ratio would have overturned the classification.

• That the two self-healing dips reflect a detector tuned to the data. The replication under the same rules had none.

• That a collapse driven by removing the skip is trivial. This is true of its existence (Dong et al., 2021), not of its persistence or of the failure of loss-based repair.

• That the result holds only because the skip was weakened. Under a learning-rate excursion with the skip intact the path stayed open and the rank recovered untreated, as the path account predicts (section 8). A settled collapse with the skip intact was not produced, so whether such a collapse would close the path is untested.

## G Interpretation fixed in advance

Before the main runs, the outcome “both loss-based treatments fail while the untreated network stays collapsed and the loss stays flat” was mapped to the reading “the force classification carries over, the repair does not”, and it is the outcome of the frozen comparison. A secondary prediction concerned three markers of the collapse along the ramp, the spectral gap $\sigma _ { 2 } / \sigma _ { 1 }$ , the mean squared cosine U and the loss gap to the healthy network, each timed at the first measurement where it is half-way from its value before the ramp to its value at detection. The predicted order, gap before U before loss gap, occurred in 0/10 events, and no order is resolvable at this ramp rate. The crossings fall at steps 2,990–3,030, 2,980–3,040 and 2,960–3,000, all within a few measurements of the moment α reaches its floor, with the loss gap no later than either spectral marker in 10/10 events.