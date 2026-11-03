# A Living Leaderboard for Graph Machine Learning

Martin Ritzert

Leipzig University, Germany

martin.ritzert@uni-leipzig.de

## Abstract

Graph machine learning benchmarking results are scattered across papers that use quite different sets of datasets and baseline methods, making it difficult to tell whether a reported gain reflects architectural progress or merely an outdated comparison or weakly tuned baseline. We introduce a living, graph-learningspecific leaderboard that organizes reported results by dataset, putting new results into context. As of 15 October 2026 our LLM-based extraction pipeline extracted 27,723 eligible results across 111 datasets from 885 peer-reviewed papers. Our analysis of these results quantifies concerns about graph-learning benchmarking: legacy datasets are still actively used and many result tables contain baseline results that are drastically lower than those already available in the literature, e.g. 45% Amazon-ratings tables and 37% for Peptides-struct. The leaderboard allows to quickly compare new results against compatible prior results, aiming to improve the overall quality of graph learning benchmarking. The leaderboard is available at https://graphlearningbenchmarks.github.io, the code at https://github.com/martinalex000/benchmarking\_website.

## 1 Why a Living Graph-Learning Leaderboard?

Benchmarking quality in graph machine learning has a long tradition of being questioned, both through large-scale benchmarking efforts [1–4] as well as a recent position paper suggesting that weak benchmarks will hinder further progress in graph machine learning [5]. Common problems identified in the literature include inadequately tuned baselines [4], selective choice of baselines to compare against [1], as well as the use of outdated datasets [3]. In particular, there is no universally accepted and adopted set of modern graph-learning benchmarks and no tuning process defined, even though the community made progress on this with the OGB [3], LRGB [6], and GraphBench [7] benchmark suites where the latter also includes suggestions on how baselines should be tuned.

We created an LLM-powered pipeline to extract benchmarking results from papers and analyzed 885 peer-reviewed papers, collected by recursively following incoming and outgoing citation links from a set of seed papers, e.g.[3, 6]. Based on the extracted results, we were able to quantify benchmarking problems and trends. In particular, we aim to answer the following questions:

• MLP baselines: In how many cases did even the best method in a paper not beat a plain MLP?

• Legacy benchmarks: How did the use of legacy datasets such as ENZYMES or Texas and Squirrel develop over time? Are they still actively used?

• Stale baselines: How often did papers report outdated performance metrics even though better ones were available at publication?

• Architectural direction: Did graph transformers take over all the leaderboards?

The main contributions of the paper are (i) the living leaderboard itself and (ii) a quantitative analysis of benchmarking problems at scale. Overall, outdated datasets and stale baselines are a problem at literature scale. Thus, extended baseline tuning is necessary (see also [5, 7]) and the community should focus on modern datasets. Through our automated extraction pipeline and planned weekly integration of new papers, our coverage is much higher than manually maintained baselines and allows to track both suggested models as well as baselines and their development over time.

![](images/7193e118e1bc0cad2410ab7a058879a1c464c2c7567d16119e655b7d45005996.jpg)  
Figure 1: LLM-powered extraction pipeline. All three round operations are based on LLM calls. The first two are single calls, while the final extraction is one call per detected dataset.

## 1.1 Related Work

Several leaderboards exist throughout machine learning, with the most well-known one being paperswithcode.co and an LLM-specific example being artificialanalysis.ai/ leaderboards/models. Within the graph-learning community, OGB maintains a leaderboard for their datasets [3], which is manually maintained and misses a few recent results found by our pipeline. As for benchmarking problems in graph ML, the position paper [5] summarizes major issues. Other relevant work includes benchmarking efforts [1], the observation that MLPs are surprisingly strong [4], and proposals for higher benchmarking standards [2, 3, 7].

## 2 Leaderboard Features and Implementation

Our leaderboard contains two main types of pages: dataset pages that list all benchmarking results for that dataset and papers pages that display the results of the paper in context with GCN and MLP baselines (if missing) and the top 3 leaderboard entries. The dataset pages contain a plot showing the SOTA over time together with baselines (see Figure 2) and a filterable table with results and their sources. Additionally, the often competitive performance of simple MPGNNs and MLPs are highlighted.

![](images/6207490cff98e9d96b55edb63325f68e98e06f7788c3c87fb09c48f988bfafde.jpg)  
Figure 2: SOTA-over-time plot for ogbn-arxiv highlighting the SOTA, as well as GCN and MLP baselines.

LLM-powered extraction. Our extraction pipeline pro-

ceeds through (i) candidate screening, (ii) dataset discovery, (iii) table selection, and (iv) result extraction as shown in Figure 1. Candidate screening is based on the title and abstract only and asks whether the paper is about graph learning and thus may plausibly contain graph machine learning benchmark results. After the initial relevance screening, we download the arXiv source (a pdf pipeline is under development) and identify table environments. We then send the introduction and experiment section, table captions, first row and column of each table into the LLM (we used Gemma4-26B-A4B, 4-bit) to identify which table contains results for which dataset. Based on this information and the chosen tables, we then extract result entries (including stds while skipping ablations) per dataset and store them in a database. Rows only enter a leaderboard when the extracted task matches. Custom, OOD, few-shot, continual, and attack settings are excluded. We additionally extract metadata such as codebase URLs or the type of each method (MPGNN, GT, MLP) which can be used for filtering.

## 3 Analysis

All analyses use a snapshot from 15 October 2026. On that date, the curated peer-reviewed corpus contained 885 papers, 111 datasets, and 27,723 eligible extracted results. When both results report a standard deviation, we use a two-sided Welch’s t-test at the 5% significance level. When only one result reports a standard deviation, we compare the other value, treated as fixed, with the 95% confidence interval for the mean. When neither result reports a standard deviation, unequal values are treated as significantly different. We do not correct for multiple comparisons. When repetition counts are unavailable, we assume four runs, which we assume to be a minimum bar for stds to be reported in the first place. This assumption is conservative when the actual (unavailable) count is larger, as it generally makes differences harder to become significant.

![](images/19fbb7e4896b0a46bb43e116db0aaf36dd8c20ff39b5f89648868fe9ff63e32d.jpg)  
Figure 3: Best non-MLP method per paper against the overall best MLP (as of today). Especially on text-heavy legacy datasets often no reported result in a paper outperforms a plain MLP.

## 3.1 The Importance of Node Features (MLP Performance)

MLPs provide a surprisingly hard-to-beat baseline on a large number of node classification datasets [4, 8]. We thus analyzed for each dataset the fraction of papers where non-MLP methods are better, tied, or worse than the best eligible MLP result in the literature (irrespective of publication date, as MLPs are an established baseline). In Figure 3 we see that especially on the legacy heterophilic datasets Texas, Cornell, Actor, and Wisconsin, graph learning techniques are not providing any benefits over plain MLPs. This also holds for other datasets with textual node features such as PubMed or Coauthor-Physics/CS (even without LLM embeddings). One should thus first take MLPs seriously as baselines in graph machine learning and second, as pointed out by Katsman et al. [4] and avoid using datasets where the graph structure is not important to track progress in graph ML.

## 3.2 Usage of Legacy Datasets

Over the last years, several papers have introduced new and modern datasets of high quality, such as OGB [3] or recently GraphBench [7]. We thus ask to what extent dataset usage has been shifting from TU Dortmund datasets [9] to those newer ones. The (legacy) heterophilic datasets Texas, Squirrel, etc. [10] contain problems (size, test data leakage, MLP performance), and Roman-Empire, Tolokers, etc. should be used instead [11]. We are thus interested in how far usage has shifted from the two legacy datasets to modern alternatives and plot the number of papers reporting results on at least one of those in Figure 4 (papers may contribute to multiple datasets). On our collected data we observe a diverging picture: while TU datasets are being used less, legacy heterophilic graphs are now used by 42.4% of 2025 conference papers using at least one of the five plotted benchmark groups, making them the second-most-used group. As those datasets are ill-suited to measure graph-specific progress [4, 11] this is an alarming trend.

![](images/60047a2fe534a4e8e8100943448315aa4f633aeb24979ca7ba14c6353ecd1da4.jpg)  
Figure 4: Dataset usage over time. Legacy datasets are still actively used.

## 3.3 Baseline Gaps

Gaps between reported results and corresponding literature values can inflate apparent architectural progress. (e.g. due to suboptimal hyperparameter configurations) An extreme example was shown by Bechler-Speicher et al. [5] where the SOTA of ENZYMES dropped according to several papers. Quantifying this, in Figure 5 we count in how many papers a baseline is reported weaker than its corpus value at time of publication (red) and overall (orange). In our collected corpus, 67.1% of analyzed papers include at least one result with a significantly better value already recorded for the same method in another paper at publication. The observed rates are lower for OGB, which has a manually maintained leaderboard and thus an easy way of looking up the SOTA, and for the relatively recent LRGB benchmark.

![](images/6228f8fda082811c41c76d384475fc751987a787d001a7f77c7a28515c02b9bd.jpg)  
Figure 5: Fraction of papers where at least one baseline had better results elsewhere.

Generally, the high values observed in our corpus indicate a general problem in benchmarking. Appendix Figure 9 shows the same for individual results and reaching up to 29.6% results lower than their corpus value (on Actor), underscoring the need for both high rigor while tuning baselines as well as a centralized place to look up benchmarking results. To measure severity, we compute for each method x the relative reduction in its distance to the current SOTA when replacing the paper-reported value with its corpus value: <sup>|Paper(x)−Corpus(x)|</sup> , where SOTA is the best recorded perfor-|Paper(x)−SOTA| mance on the dataset across all methods. The histogram in Figure 6 shows that in 38.5% of cases the best literature value was at least halfway to the SOTA, i.e. the (baseline) method is a lot better than the paper’s table suggested.

![](images/ddb0f8ce22eeb64f68b8ddd84fbf33cb7f8dc89c95b071a30ff6b681fe74f30d.jpg)  
Figure 6: Relative reduction in a method’s distance to the SOTA using its corpus value instead of its paper value

## 3.4 Message Passing vs. Graph Transformers

There has been a long-running debate over whether graph transformers would take over graph-learning leaderboards, as transformers did in other areas. We thus count how many datasets (with a minimum of five papers reporting on them) are led by each architecture category, e.g. message-passing GNNs, graph transformers, LLMs, or traditional methods such as SVMs and boosting. Each dataset has unit weight, distributed evenly among all methods treated as tied for being the best. With 39.1% GTs vs 47.2% MPGNNs, neither category dominates and the picture remained stable for the last two years, see Figure 7. MPGNNs lead on node-level tasks, while GTs have a slightly larger share on graph-level tasks (see Appendix). We can also observe that traditional methods are still strong contenders on several datasets, especially around molecules.

![](images/2e3367fe72036046980ab32b431d72db515f9e53d2fc55a8d74503c682b04eef.jpg)  
Figure 7: Architecture of best models over datasets. MPGNNs are leading but not dominating.

## 4 Limitations, Future Directions, and Conclusion

Around 80% of NeurIPS, ICLR, and ICML papers from 2022–2024 are available on arXiv (Table 1 in the appendix), limiting source-based coverage. We thus plan to develop a PDF extraction path for papers without available LAT<sub>E</sub>X sources. The second limitation is that incompatible results may be merged into the same leaderboard, e.g. from different evaluation protocols or different metrics. Figuring out whether a result is really compatible is non-trivial even for humans, and exactly where we plan to improve the current pipeline. Until then, we have guards in place (same metric, outlier detection, manual exclusion, rules against typical cases) avoiding most incompatible results.

In this paper, we presented a leaderboard specifically for graph machine learning that allows users to quickly identify the best baselines for any given dataset. Within our collected corpus, we found that legacy datasets are even gaining popularity and that there are often better results for reported baselines, especially for MLPs: for example, on 57.3% of papers reporting Cornell results, the best known MLP outperforms all reported non-MLP methods in that paper on that dataset. Further, neither GTs nor MPGNNs dominate the observed frontier. Overall, we observed rising use of legacy datasets with known problems, stronger-than-expected MLP baselines, and regular reporting of results for which the corpus contains a better one under the same name.

## References

[1] Federico Errica, Marco Podda, Davide Bacciu, and Alessio Micheli. A fair comparison of graph neural networks for graph classification. In International Conference on Learning Representations, 2020. 1, 2

[2] Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. Journal of Machine Learning Research, 24(43):1–48, 2023. 2

[3] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. volume 33, pages 22118–22133, 2020. 1, 2, 3

[4] Isay Katsman, Ethan Lou, and Anna Gilbert. Revisiting graph learning benchmarks: When is the graph actually necessary? Neural Processing Letters, 58(2):24, 2026. 1, 2, 3

[5] Maya Bechler-Speicher, Ben Finkelshtein, Fabrizio Frasca, Luis Müller, Jan Tönshoff, Antoine Siraudin, Viktor Zaverkin, Michael M Bronstein, Mathias Niepert, Bryan Perozzi, et al. Position: Graph learning will lose relevance due to poor benchmarks. arXiv preprint arXiv:2502.14546, 2025. 1, 2, 3

[6] Vijay Prakash Dwivedi, Ladislav Rampášek, Michael Galkin, Ali Parviz, Guy Wolf, Anh Tuan Luu, and Dominique Beaini. Long range graph benchmark. Advances in Neural Information Processing Systems, 35:22326–22340, 2022. 1, 6, 9

[7] Timo Stoll, Chendi Qian, Ben Finkelshtein, Ali Parviz, Darius Weber, Fabrizio Frasca, Hadar Shavit, Antoine Siraudin, Arman Mielke, Marie Anastacio, et al. GraphBench: Next-generation graph learning benchmarking. arXiv preprint arXiv:2512.04475, 2025. 1, 2, 3, 6

[8] Shichang Zhang, Yozen Liu, Yizhou Sun, and Neil Shah. Graph-less neural networks: Teaching old mlps new tricks via distillation. arXiv preprint arXiv:2110.08727, 2021. 3

[9] Christopher Morris, Nils M Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. Tudataset: A collection of benchmark datasets for learning with graphs. arXiv preprint arXiv:2007.08663, 2020. 3

[10] Hongbin Pei, Bingzhe Wei, Kevin Chen Chuan Chang, Yu Lei, and Bo Yang. GEOM-GCN: Geometric graph convolutional networks. In 8th International Conference on Learning Representations, ICLR 2020, 2020. 3

[11] Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila Prokhorenkova. A critical look at the evaluation of gnns under heterophily: Are we really making progress? arXiv preprint arXiv:2302.11640, 2023. 3

[12] Derek Lim, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, and Ser-Nam Lim. Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods. arXiv preprint arXiv:2110.14446, 2021. 7

[13] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. Advances in Neural Information Processing Systems, 35:14501–14515, 2022. 8

[14] Jan Tönshoff, Martin Ritzert, Eran Rosenbluth, and Martin Grohe. Where did the gap go? reassessing the long-range graph benchmark. arXiv preprint arXiv:2309.00367, 2023. 9

## A Technical Details

The extraction is based on LLMs and in our case a 4-bit-quantized GEMMA-4 26B-A4B-IT instance was selected as a fast and cost-effective model for structured extraction. We also tested against smaller and larger families (gpt-oss, Qwen, GLM, MiniMax, see Table 2) and found that Gemma and GLM both generally worked well, but prompts have to be adapted to the model at hand to reliably ignore ablation tables and properly normalize method names. For the initial abstract screening, we relied on a GPT-OSS-120B model, which is considerably faster than our Gemma instance. In addition to the extraction pipeline, we tried a simple one-shot approach, essentially telling the model to extract all benchmark results, which turned out to be less reliable. For extracting the number of repetition for each dataset, a more advanced model was necessary where we used GLM-5.3-FLASH.

In order to render the website, we export the results to Markdown and then use Hugo to generate a static website. The SOTA-over-time plot as well as the filtering and sorting of tables are done client-side using JavaScript.

We ran our experiments on a server with EPYC 7352 CPUs and 1 TB RAM. In general, the hardware requirements (except for running the LLM) are quite low and it would have been possible to run the experiment on a laptop with API access to a language model. Our language models were run using vLLM on two NVIDIA A100 40 GB cards. The current extraction pipeline is relatively inefficient with respect to token usage; a full re-evaluation takes approximately 3 days at ∼20 papers/h (∼70 papers evaluated in parallel to boost throughput).

## B Selection of Papers to Be Parsed

Our corpus consists of conference papers that are on arXiv since the available LAT Xsource makes parsing tables easier. We plan to extend the pipeline to include pdfs directly through e.g. OpenReview to get complete coverage of major ML conferences. About 80% of Semantic Scholarindexed NeurIPS, ICLR, and ICML papers from 2022–2024 have an arXiv identifier, see Table 1. To build the corpus, we relied on citation links and Semantic Scholar, initialized with graph-learning papers including LRGB [6] and GraphBench [7]. The corpus was then expanded through the papers’ citation neighborhoods, following both references and citations whenever an arXiv version was available (externalIds.ArXiv). A candidate was excluded only if its title and abstract did not pass the graph-learning relevance pre-screen, or if full-text processing yielded no numeric benchmark result compatible with a canonical dataset–task pairing tracked by the leaderboard.

Table 1: Share of Semantic Scholarindexed conference papers with an arXiv identifier.
<table><tr><td>Conference</td><td>2022</td><td>2023</td><td>2024</td></tr><tr><td>NeurIPS</td><td>79.5%</td><td>79.4%</td><td>80.3%</td></tr><tr><td>ICLR</td><td>80.4%</td><td>83.1%</td><td>84.2%</td></tr><tr><td>ICML</td><td>83.0%</td><td>84.2%</td><td>80.2%</td></tr></table>

## C Validation of the LLM pipeline

Since the pipeline is fully automatic, we tested its performance on multiple ways. This included a simple check that modern LLMs are indeed capable of extracting numbers from tables (Section C.1), a check of the pre-screening method (Section C.2), as well as an evaluation of the extraction pipeline covering datasets and methods in Section C.3. Further, there is a module that finds the number of repetitions for each experiment on a table level. Together with the information on whether a result has been trained by the paper themselves, we use this as the basis for our statistical evaluation.

## C.1 Extraction Agreement Case Study

We found that transcription stability is not an issue with 2026 models (in contrast to early LLMs), and all tested models are capable of successfully copying numbers from a table into JSON. We illustrate this in Table 2, which shows that different models generally agree on the numerical values and differ mainly in which rows are extracted or skipped and in how method names are normalized. Through manual verification, we were able to check that indeed all of the values are in the paper and that several models (GLM, Gamma4 MoE, gpt-oss) did extract all existing rows. In the concrete example, the Minibatching variants ignored by the other models seem to be part of the method and should thus be extracted, but they can easily be mistaken for an ablation and skipped, as those models did.

Table 2: Example extracted results on Penn94 from [12]. While LLMs disagree on which methods to extract (skipping variants, but synchronized), they show perfect reconstruction of numbers.
<table><tr><td>Method</td><td>Human</td><td>GLM-5.2</td><td>Gemma-4-26B-A4B</td><td>Gemma-4-31B</td><td>GPT-OSS-120B</td><td>MiniMax-M3</td><td>Llama-3.3-70B</td><td>Values match</td></tr><tr><td>APPNP</td><td>0.7433</td><td>0.7433</td><td>0.7433</td><td>0.7433</td><td>0.7433</td><td>0.7433</td><td>0.7433</td><td></td></tr><tr><td>C&amp;S 1-hop</td><td>0.7428</td><td>0.7428</td><td>0.7428</td><td>0.7428</td><td>0.7428</td><td>0.7428</td><td>0.7428</td><td></td></tr><tr><td>C&amp;S 2-hop</td><td>0.7840</td><td>0.7840</td><td>0.7840</td><td>0.7840</td><td>0.7840</td><td>0.7840</td><td>0.7840</td><td></td></tr><tr><td>GAT</td><td>0.8153</td><td>0.8153</td><td>0.8153</td><td>0.8153</td><td>0.8153</td><td>0.8153</td><td>0.8153</td><td></td></tr><tr><td>GATJK</td><td>0.8069</td><td>0.8069</td><td>0.8069</td><td>0.8069</td><td>0.8069</td><td>0.8069</td><td>0.8069</td><td></td></tr><tr><td>GCN</td><td>0.8247</td><td>0.8247</td><td>0.8247</td><td>0.8247</td><td>0.8247</td><td>0.8247</td><td>0.8247</td><td></td></tr><tr><td>GCNII</td><td>0.8292</td><td>0.8292</td><td>0.8292</td><td>0.8292</td><td>0.8292</td><td>0.8292</td><td>0.8292</td><td></td></tr><tr><td>GCNJK</td><td>0.8163</td><td>0.8163</td><td>0.8163</td><td>0.8163</td><td>0.8163</td><td>0.8163</td><td>0.8163</td><td></td></tr><tr><td>GCNJK-Cluster</td><td>0.6999</td><td>0.6999</td><td>0.6999</td><td></td><td>0.6999</td><td></td><td></td><td></td></tr><tr><td>GCNJK-SAINT-Node</td><td>0.7280</td><td>0.7280</td><td>0.7280</td><td></td><td>0.7280</td><td></td><td></td><td></td></tr><tr><td>GCNJK-SAINT-RW</td><td>0.7229</td><td>0.7229</td><td>0.7229</td><td></td><td>0.7229</td><td></td><td></td><td></td></tr><tr><td>GPR-GNN</td><td>0.8138</td><td>0.8138</td><td>0.8138</td><td>0.8138</td><td>0.8138</td><td>0.8138</td><td>0.8138</td><td></td></tr><tr><td>L Prop 1-hop</td><td>0.6321</td><td>0.6321</td><td>0.6321</td><td>0.6321</td><td>0.6321</td><td>0.6321</td><td>0.6321</td><td></td></tr><tr><td>L Prop 2-hop</td><td>0.7413</td><td>0.7413</td><td>0.7413</td><td>0.7413</td><td>0.7413</td><td>0.7413</td><td>0.7413</td><td></td></tr><tr><td>LINK</td><td>0.8079</td><td>0.8079</td><td>0.8079</td><td>0.8079</td><td>0.8079</td><td>0.8079</td><td>0.8079</td><td></td></tr><tr><td>LINK Minibatch</td><td>0.8161</td><td>0.8161</td><td>0.8161</td><td></td><td>0.8161</td><td></td><td></td><td></td></tr><tr><td>LINKX</td><td>0.8471</td><td>0.8471</td><td>0.8471</td><td>0.8471</td><td>0.8471</td><td>0.8471</td><td>0.8471</td><td></td></tr><tr><td>LINKX Minibatch</td><td>0.8450</td><td>0.8450</td><td>0.8450</td><td></td><td>0.8450</td><td></td><td></td><td></td></tr><tr><td>MixHop</td><td>0.8347</td><td>0.8347</td><td>0.8347</td><td>0.8347</td><td>0.8347</td><td>0.8347</td><td>0.8347</td><td></td></tr><tr><td>MixHop-Cluster</td><td>0.7579</td><td>0.7579</td><td>0.7579</td><td></td><td>0.7579</td><td></td><td></td><td></td></tr><tr><td>MixHop-SAINT-Node</td><td>0.7561</td><td>0.7561</td><td>0.7561</td><td></td><td>0.7561</td><td></td><td></td><td></td></tr><tr><td>MixHop-SAINT-RW</td><td>0.7638</td><td>0.7638</td><td>0.7638</td><td></td><td>0.7638</td><td></td><td></td><td></td></tr><tr><td>MLP</td><td>0.7361</td><td>0.7361</td><td>0.7361</td><td>0.7361</td><td>0.7361</td><td>0.7361</td><td>0.7361</td><td></td></tr><tr><td>MLP Minibatch</td><td>0.7424</td><td>0.7424</td><td>0.7424</td><td></td><td>0.7424</td><td></td><td></td><td></td></tr><tr><td>SGC 1-hop</td><td>0.6679</td><td>0.6679</td><td>0.6679</td><td>0.6679</td><td>0.6679</td><td>0.6679</td><td>0.6679</td><td></td></tr><tr><td>SGC 2-hop</td><td>0.7609</td><td>0.7609</td><td>0.7609</td><td>0.7609</td><td>0.7609</td><td>0.7609</td><td>0.7609</td><td></td></tr></table>

For more general numbers, we ran the six LLMs GEMMA-4-26B-A4B 4-bit (our main driver for the extraction pipeline) as well as GLM-5.2, GEMMA-4-31B, GPT-OSS-120B, MINIMAX-M3, and LLAMA-3.3-70B on 10 papers selected to contain relatively many and well-distributed results (2110.14446, 2205.12454, 2305.08018, 2305.13315, 2305.18407, 2306.16976, 2405.02795, 2405.16435, 2502.09263, 2506.04765). Table 2 is representative of the strong agreement on numerical values observed across the ten-paper evaluation. Outside QM9, only 30 of 16,298 pairwise comparisons differed beyond rounding, involving a small number of differences in experimental settings, method variants, and row or column assignments. QM9 initially also exposed unit-normalization issues. The main challenge is consistently deciding which rows represent models to include and which represent variants to exclude, together with normalizing model names, rather than numerical transcription.

## C.2 Pre-Screening

As detailed in the main paper, the pre-screening is tasked to distinguish graph learning papers from other papers. For that we first filter based on keywords such as “graph neural network” and then send title and abstract of the paper to a fast LLM, in our case GPT-OSS-120B. We manually evaluated the success of this process by looking (post-hoc) at three different cases: 10 papers accepted by pre-screening that contribute no results, 10 rejected papers that contain specific keywords such as MPGNN or graph embedding, and 10 rejected papers that contain the word graph but none of the specific keywords in title or abstract. We found that in all 30 cases, manual verification agreed with the LLM screening result: the accepted papers were all about graph learning, but considered tasks that are not yet tracked such as graph generation and adversarial attacks. On the other hand, the rejected papers were indeed mostly math papers. It turned out that the term “graph embedding" which we used as one of the filters is regularly used outside of graph ML and was a major driver of papers that made it to the LLM and were rejected afterwards.

While the pre-screening works as expected, only about half of the papers accepted by pre-screening contribute results. This is because the pre-screening check only very generally checks whether the paper has anything to do with graph learning including areas that are not tracked yet, e.g. graph generation. Tightening the pre-screening objective could thus make extraction more efficient.

## C.3 Result extraction

This is the main experiment to validate the pipeline. It is split into two parts: dataset detection and model detection. For both experiments we evaluated a shared set of 10 papers with results such that we can also estimate the end-to-end precision and recall.

Dataset detection: In our experiment on 10 papers with results we observed 100% precision and only 62.1% recall which is mainly due to an overly-pessimistic evaluation of result compatibility since raw YAML extraction achieved a recall of 96.1% but lower precision.

Method detection: On the same 10 papers as before, our pipeline was able to detect almost all methods (98.7% precision, 86.2% recall). The main problems were connected to multiple variants of a method being reported where the extraction pipeline normalized some but not others from the same table. For each candidate result on a given dataset, the extraction prompt supplied the canonical task, metric, and split information for that dataset and asked the model to extract this information for each result. Parser-side checks then rejected explicit mismatches.

While the pipeline is tuned to err on the safe side and in doubt reject results, we aim to improve the recall while keeping the precision. In our current implementation it is possible (and actually happens) that individual results from a table are marked as incompatible while the remainder of the table makes it into the database. We plan to remedy that by including an additional verifier step and shifting the compatibility decisions from the result to the dataset level.

End-to-End: On the same manually annotated 10-paper evaluation, after removing the known incompatible rows, the extraction achieved an end-to-end precision of 99.2% and 63.9% recall. Again, our main focus lies on precision, but it is clear that the recall leaves a lot of room for improvement which we plan to address in the near future.

## Repetition Counts:

For the statistical evaluation, we needed the repetition counts behind each reported standard deviation. For that we ran another extractor, this time based on GLM-5.3-FLASH, extracting the information on a table-level. Again, we used 10 papers for manual verification (a different set from above) and found essentially perfect performance. We then matched the information to the results using another LLM call which we verified on yet another 10-paper set, this time two cross-validation values were not properly propagated with the model erring on the safe side. By integrating the count extraction directly in the extraction process, the whole matching step is no longer necessary, simplifying our setup. We note that this will involve switching to a more capable model as the success rate of smaller models on the counts, which are often hidden somewhere in the appendix with highly variable formulations, was not satisfactory.

## D Additional Results

## D.1 Message-Passing vs. Graph Transformers

In the main paper, we showed that in general GTs and MPGNNs together dominate graph learning leaderboards with MPGNNs having the larger aggregate share. With graph learning being split in a multitude of tasks and GTs limited to moderately-sized graphs due to the quadratic scaling of attention, we were interested in how that distributes between node-level and graph-level tasks, shown in Figure 8. We observe that MPGNNs lead on node-level tasks (52.8% versus 30.0% for GTs), while GTs have a larger share on graph-level tasks (49.6% versus 43.0% for MPGNNs). We thus conclude that neither method is clearly superior to the other and it really depends on the dataset at hand, not only the task.

Note that methods are “unknown” if our extraction process was unable to detect the method’s type and “traditional” includes SVMs, random forests, and all kinds of boosting methods. For this experiment we considered both pure transformers as well as hybrid architectures such as GPS [13] to be part of the “transformer” class while the leaderboard website distinguishes between them.

## D.2 Reported Results vs. Literature Values

In general, we assumed for this analysis that methods that are reported with the same name are compatible. We note that this is not perfectly well-defined, e.g. is a GCN allowed to use skip connections? Or positional encodings? Or virtual nodes? For our analysis we generally trust the authors to name methods appropriately and use a different name for bigger modifications of the base method. We additionally matched methods by their base-name, mapping GCN, GCN<sup>⋆</sup>, and GCN (tuned) to the same “GCN” method. In general, the method-name normalization is performed by the LLM. There, reimplementations by the paper authors and configuration variants representing the same underlying method are consolidated, while qualifiers identifying genuinely distinct methods are preserved.

![](images/e806a35dafeffbaad9c2b7371775f2d26e6758138978a8c0d7af1ff50ecb0858.jpg)  
2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026  
Figure 8: Architecture of best models over graph learning datasets. MPGNNs lead on node-level tasks, while GTs are leading but not dominating on graph-level tasks.

![](images/5501268603a41f7a8381d5d428025e22c7d08937d35ff2a5f6be5b70f2023917.jpg)  
Figure 9: Result comparisons per dataset. Red shows results where a better result was known at publication; orange lists results for which a better result (typically better hyperparameters) has been reported under the same name later.

Here we extend Figure 5 by showing individual results instead of aggregating to the paper level: for each dataset, a result is marked in red when a significantly better value for the same method was available at publication, and in orange when a significantly better value was found only since publication. In Figure 9 we see that it is extremely common to have at least one result which does not represent the literature value. Across all comparable results, 16.8% have a significantly better prior literature value, rising to 24.8% when later literature values are included, clearly showing that the problem is widespread.

We additionally present a detailed version of Figure 6 split by dataset group as given in Figures 5 and 9. The most obvious feature is the peak at 100% for LRGB, which is due to a simple GCN being SOTA on Peptides-struct [14] while being underreported by the original paper [6]. The plots in Figure 10 show that the share of selected stale results closing at least half of the gap ranges from 26.2% on citation datasets to 75.0% on LRGB. Compared to the figure in the main paper, the interpretation is more nuanced: while on LRGB those massive improvements are common, on other datasets the difference is more moderate but still substantial. Tönshoff et al. [14] showed that through underreported numbers on LRGB, the gains of transformers looked larger than they actually were. Here, in Figure 10 and Figure 9, we showed that such underreported results are common throughout the corpus and that the underreporting is often substantial.

## D.3 Standard-Deviation Reporting

Based on the extracted data, we analyzed how many papers report standard deviations, counting papers with more than 30% of results containing one. Standard-deviation reporting improved from

![](images/cfc15551def831aa761ccaef028bc5a9395c8f026253da99510c72804d16d957.jpg)  
Share of gap to current SOTA closed at publication

Figure 10: Relative reduction in a method’s distance to the SOTA using its corpus value instead of its paper-reported value, by dataset group.  
![](images/365a54da794b14451e06c4609a6aa838438bf1c98146d8e69de0b83bd7bbf330.jpg)  
Figure 11: Fraction of papers reporting standard deviations. Starting with less than 40% in 2017 it rose to ∼80% in 2022 and stagnated there since.

36.4% of papers in 2017 to roughly 80% since 2022 (Figure 11). Although standard deviations are now common in graph machine learning, around one in five papers still does not report them for its experiments.

## E LLM Statement

Language models were used during the development of the extraction pipeline and website, as well as in generating the plots. LLMs used include the GPT family as well as Composer 2.5 and Kimi-2.6. GPT 5.6-sol was used during the writing process (grammar, proofreading, drafting an early version) and GPT 6 for polishing the camera ready version.