# A Later Test Set Is Not a New Domain: Pretraining Familiarity Survives a Contamination-Free Hold-Out

Mahdi Naser Moghadasi BrightMind AI mahdi@brightmind-ai.com

Faezeh Ghaderi University of Texas at Arlington faezeh.ghaderi@mavs.uta.edu

## Abstract

Time-series foundation models are evaluated almost exclusively on public archives that predate them, so a strong score cannot be separated from having seen the test set during pretraining. The obvious remedy is a hold-out that postdates the models. We build one: thirteen forecasters — four classical, three trained per dataset, six pretrained — on seven groups drawn from five domains, every observation published after the last model was released, and every dataset rebuildable without an API key. Under this protocol pretrained models win 5 of 7 groups, lose one to a Theta baseline, and on daily exchange rates are indistinguishable from a seasonal naive forecast, along with every other method tested.

We then ask what separates the wins from the losses, and report a negative result: the two intrinsic properties one would reach for — seasonal strength and spectral entropy, measured on the input window — do not account for the pattern, and seasonal strength is if anything negatively associated with the advantage. What does track it is corpus familiarity. Our largest gain (28% lower MASE than the best classical method, on weekly Wikipedia pageviews) falls on Wikipedia pageviews, the domain TimesFM’s authors describe as the bulk of its pretraining corpus, at the same granularities and differing only in time window. Within the pretrained family, where every model forecasts identical series so that series difficulty cancels, the TimesFM family outranks the Chronos family by -0.53 on Wikipedia against -0.09 everywhere else (1,500 vs. 754 series, Mann–Whitney $p < 1 0 ^ { - 5 } )$ . We conclude that a temporal hold-out removes memorisation of a window but not familiarity with a domain, that benchmarks therefore need domain hold-outs stated relative to disclosed corpora, and that the practitioner’s question is less which model is better than whether their domain is one the model was raised on.

## 1 Introduction

The claim attached to time-series foundation models is that a single pretrained network forecasts unseen series without being fitted to them, and that it does so better than methods estimated on the target series itself. That claim is usually supported by evaluation on a standard collection of public archives. Those archives are the difficulty. Every widely used benchmark set — the M-competition data, ETT, the Monash repository — predates the models being evaluated and is distributed in the same public form that pretraining corpora are assembled from. A high score is therefore consistent with two very different explanations, generalisation and memorisation, and the published evaluations cannot distinguish them.

The remedy is not a better statistic but a better hold-out. This paper evaluates on observations that did not exist when the models were trained. We assemble seven forecasting groups from five domains, every one of them published continuously by an institutional source, and place the entire test window after the release date of the most recently released model in the comparison. Nothing in the hold-out can have been memorised because none of it had happened yet.

Under that protocol the headline finding of the foundation-model literature survives only in part. Pretrained models are clearly better on some domains and clearly not better on others. The interesting question is what separates them, and our answer is not the one we set out to give: we expected a property of the series and found a property of the corpus. The domain on which pretrained models gain most is the domain one of them was largely pretrained on — the same source and the same granularities, differing only in time window — and the advantage is largest for exactly the model whose corpus it is.

This is not memorisation of the test set; the hold-out makes that impossible. It is the weaker but more durable form of the same problem. A model that has read a decade of Wikipedia pageview series has learned what Wikipedia pageview series do, and January 2026 pageviews still do it. Moving the test window forward controls for the first kind of contamination and leaves the second untouched, which means the field’s evaluations cannot be repaired by recency alone.

## Contributions.

1. A contamination-free protocol, with a hold-out that postdates every evaluated model, over five domains built only from keyless, continuously published sources so that any reader can rebuild the exact panel from the fetchers we release.

2. Two negative results. Pretrained models do not beat classical baselines everywhere: we identify one domain where a Theta baseline ranks first and one where nothing beats seasonal naive. And the intrinsic explanation fails — neither seasonal strength nor spectral entropy accounts for where the advantage appears.

3. Evidence that pretraining familiarity persists past a temporal hold-out, from a withinfamily comparison on identical series that no property of those series can explain.

4. Significance rather than decimal places. We report multiple-comparison rank intervals and corrected paired tests, and show that most published-style gaps between leading pretrained models fall inside them.

5. Cost alongside accuracy, measured on identical hardware in the same runs.

## 2 Related work

Pretrained forecasters. The premise shared across this family is that one network, pretrained on a large corpus of series, forecasts unseen series without being fitted to them. Chronos [Ansari et al., 2024] tokenises scaled series and trains a language-model architecture over the token sequence, with Chronos-Bolt and Chronos-2 [Chronos Team, Amazon Web Services, 2025] as faster successors. TimesFM [Das et al., 2024] is a decoder-only patched transformer. Moirai [Woo et al., 2024] handles arbitrary frequency and multivariate input through masked patch prediction. Lag-Llama [Rasul et al., 2023] conditions on lagged values for probabilistic forecasting, TimeGPT [Garza et al., 2023] is a commercial entrant, MOMENT [Goswami et al., 2024] targets several tasks beyond forecasting, Tiny Time Mixers [Ekambaram et al., 2024] pursues the same zero-shot goal at a fraction of the parameter count, and Timer [Liu et al., 2024b] and Sundial [Liu et al., 2025] extend the generative-pretraining approach. What matters for this paper is a property they share rather than one that separates them: their corpora are disclosed in summary at best, so no evaluation can verify by inspection that a given test set was excluded from training.

The supervised lineage they replace. The models a pretrained forecaster is meant to displace are the ones trained on the target dataset: DeepAR [Salinas et al., 2020] and Temporal Fusion Transformers [Lim et al., 2021] for probabilistic and multi-horizon forecasting, N-BEATS [Oreshkin et al., 2020] — which we run as a baseline — and the long-horizon transformer line from Informer [Zhou et al., 2021] and Autoformer [Wu et al., 2021] through PatchTST [Nie et al., 2023] and iTransformer [Liu et al., 2024a]. That line has its own history of contested gains: Zeng et al. [2023] showed a linear model matching or beating a generation of these architectures, and Makridakis et al. [2018] had earlier found machine-learning methods losing to statistical ones across the M3 series once evaluation was done carefully. Our study inherits that scepticism rather than inventing it.

Benchmarks, and what they are made of. The M4 [Makridakis et al., 2020] and M5 [Makridakis et al., 2022] competitions set the conventions we follow, and the Monash archive [Godahewa et al., 2021] assembled the collection most zero-shot claims are measured against. More recent efforts target foundation models directly: GIFT-Eval [Aksu et al., 2024] spans many domains and frequencies, and fev-bench [Shchur et al., 2025] emphasises realistic tasks with covariates. All of them are built from data that was public before the models were trained. That is not a criticism of their construction — it is what makes results comparable across papers — but it is why a benchmark assembled from published archives cannot separate generalisation from recall, however many domains it contains. Independent evaluations have already found the zero-shot claims fragile: Toner et al. [2025] report that a seasonal naive forecast beat every foundation model they tested on cloud-infrastructure data, at every horizon; Xu et al. [2024b] find specialised foundation models struggling against supervised baselines across modalities; and Tan et al. [2024] question whether language models contribute anything to forecasting at all. Our exchange-rate result is of the same kind, and our contribution is to say precisely where that regime begins and ends.

Contamination. Benchmark contamination is well documented for language models [Sainz et al., 2023, Xu et al., 2024a], where the usual approach is to detect overlap between a test set and a training corpus after the fact. Two recent papers raise the problem specifically for time-series foundation models, and this work sits directly between them.

Meyer et al. [2025] audit dataset lineage across published models and find that only a small minority of the datasets in circulation have never appeared in some model’s pretraining or fine-tuning corpus. Their empirical work demonstrates that leakage can inflate results rather than measuring how much it does for any released model, and they close by calling for evaluation on continuously sourced, post-release test data. This paper is an attempt to build what they ask for.

Li et al. [2026] approach it from the opposite side, with a detector: they probe a model’s adaptation dynamics — contamination shows up as faster loss reduction with less movement in the backbone — to decide whether a given dataset was in a given corpus. That answers a membership question, per dataset, without quantifying how much a reported zero-shot number is inflated, and without needing a clean test set.

Our contribution requires no access to any corpus and asks neither question. We evaluate on observations that did not exist when the models were trained, and then compare performance on domains that are in a model’s corpus against domains that are not. The three are complementary: an audit finds membership a temporal hold-out cannot see, and — as §4.3 argues — a temporal hold-out exposes a residual effect that a membership audit would not flag at all, because under this protocol no test observation is in any corpus and every audit would correctly return “clean”.

Classical baselines and how to compare them. Theta [Assimakopoulos and Nikolopoulos, 2000] won M3 and remains hard to beat on seasonal data; automatic ARIMA and ETS [Hyndman et al., 2008] are the standard automated alternatives, and we use the statsforecast implementations [Garza et al., 2022] so the baselines are the ones a practitioner would actually run. Our reporting follows the forecasting literature rather than the deep-learning one: MASE as defined by Hyndman and Koehler [2006], and comparison by average rank with the intervals of Demsar [2006], in theˇ spirit of Koning et al. [2005], who showed for M3 that many of the gaps separating competition entries were not statistically distinguishable at all. That is the lesson we apply to Table 2.

## 3 Protocol

## 3.1 Contamination-free hold-out

The forecast origin is fixed at 1 January 2026 for every group; each model receives the observations preceding it as context and is scored on the horizon that follows. Every evaluated checkpoint — amazon/chronos-bolt-small and -base, amazon/chronos-2, google/timesfm-2.5-200m-pytorch, google/timesfm-3.0-pytorch and Salesforce/moirai-2.0-R-small — was published before that date, so no test observation existed when any of them was trained.

We deliberately do not tabulate pretraining cutoffs. Several of these models disclose their corpora only in summary, and a table of dates assembled from release notes would give the protocol a precision it does not have. The argument does not need it: an observation recorded in 2026 cannot appear in a checkpoint published in 2025, whatever that checkpoint’s internal cutoff was. What we can pin exactly is the software, and requirements-lock.txt records every library version used to produce these numbers.

## 3.2 Domains

All five domains are published continuously by an institution, are retrievable without registration or an API key, and are not redistributions of an existing benchmark. Table 1 lists the resulting groups.

Wikipedia pageviews (daily, weekly, monthly). Human attention: strongly periodic, heavy-tailed, punctuated by exogenous spikes.

Weather (hourly, ERA5 reanalysis via Open-Meteo). Smooth, dominated by a clean daily cycle.

Air quality (hourly, CAMS via Open-Meteo). Same sampling rate and daily cycle as weather, but spiky and heavy-tailed. Included specifically to separate “good at hourly data” from “good at smooth data”.

Electricity (hourly, Danish grid settlement). The domain most often claimed in the foundation-model literature, but evaluated here without the contamination that the ETT datasets carry.

Exchange rates (daily, ECB reference rates). The adversarial case: the standing result in forecasting is that nothing reliably beats a naive forecast, so this domain tests whether a benchmark can detect the absence of an effect.

Series selection is specified before any result is seen — the full city list, the full currency list, the publisher’s own column set — and nothing is added or removed afterwards. One mechanical exclusion applies: a series taking fewer than three distinct values across its entire input window is dropped, because MASE divides by an in-sample seasonal-naive error that is then zero and the metric is undefined. Exactly one series in the study meets this condition (a hydro-generation column for a Danish price area with no hydro, reported as zero throughout), and the rule is stated in these terms rather than as a threshold because a relative one cannot catch it: its scale is negligible in absolute terms and large relative to itself.

## 3.3 Horizons, seasonality and context

Horizons and seasonal periods follow the M4 conventions for the matching frequency, so that numbers here sit beside prior work rather than beside a convention we invented: 14 steps at daily with m = 7, 13 at weekly, 48 at hourly with m = 24. Three choices depart from that and each is a judgement we state rather than bury.

Weekly seasonality is m = 1, not 52. This is M4’s own setting for its weekly group, adopted there because a seasonal ARIMA search at m = 52 is intractable; it exhausted memory here too. Monthly Wikipedia uses a horizon of 8 rather than M4’s 18, because with the origin at 1 January 2026 only eight complete months of hold-out exist — a direct consequence of insisting the test window be recent, and the clearest cost of that insistence.

The hourly groups are capped at 2,016 context points, twelve weeks, applied identically to every model. The ERA5 series run to 17,000 points, longer than anything in M4; no pretrained model can attend to that (their windows stop between 512 and 2,048) and a seasonal ARIMA search over it does not terminate. Truncating in the loader rather than per model keeps the comparison honest: every method sees exactly the same input. Groups with more than 500 series are subsampled to 500 with a fixed seed, which is recorded in the harness.

## 3.4 Metrics and significance

We report MASE and sMAPE as defined by the M4 organisers, and weighted quantile loss and 80% interval coverage for the probabilistic models. Because a table of mean losses invites ranking by the third decimal place, our primary comparison is by rank: series are ranked across models, and each model receives an average rank with a Nemenyi interval derived from the Friedman statistic, following the multiple-comparisons-with-the-best presentation used to report M5. We additionally test each model against the per-group winner with a Wilcoxon signed-rank test, Holm-corrected across comparisons. We use signed-rank rather than a paired t-test because MASE across a panel is strongly right-skewed, and we avoid the label “Diebold–Mariano” [Diebold and Mariano, 1995] because that test compares forecast errors over time within a single series, which is not the comparison a panel benchmark makes.

Our reading of MASE follows Hyndman and Koehler [2006], and the decision to report four measures rather than nominate one follows Kolassa [2020]: the ranking of forecasts is not invariant to the error measure, so a paper that reports a single metric has made a choice it usually does not defend. Weighted quantile loss is a proper scoring rule in the sense of Gneiting and Raftery [2007], which matters for $\ S 4 . 6 - \mathbf { a }$ model cannot improve it by misreporting its own uncertainty, so the calibration failures reported there are not an artefact of the score.

The gap between a mean and a rank is not academic here, and we met it by accident. Before the exclusion above was in place, the near-constant Danish hydro series gave the two neural models mean MASE values of order $1 0 ^ { 6 }$ on the electricity group — one series in forty-seven moving the group mean by six orders of magnitude, while every model’s typical behaviour on that group was unremarkable. The rank-based conclusion for that group was identical with and without the series. A benchmark reported as a mean is one degenerate series away from a headline; reported as ranks it is not.

## 4 Results

## 4.1 Accuracy

![](images/2a392d2cdfe3b64e982746cc2bbd468711ffc65693556ae9e3b81475d73651a8.jpg)  
Figure 1: Average rank with Nemenyi intervals, per group. Any model whose interval crosses the dashed line is statistically indistinguishable from the best model in that panel. Exchange rates, on the bottom row, is the panel where every interval overlaps every other and the Friedman test does not reject.

Table 2 is the paper’s central result, and it does not read like the tables in the papers introducing these models. A pretrained model holds the top rank in 6 of the 7 groups, but that count overstates the case and we do not use it: in 1 of those the Friedman test does not reject, so the column has a winner in the arithmetic sense only. Counting a rank-first under a null result as a win is precisely the overclaim this paper exists to argue against. The defensible statement is that pretrained models win 5 groups, lose one, and are indistinguishable from everything else — seasonal naive included — on the last.

The two exceptions are the informative ones. On electricity Theta ranks first and the leading pretrained models are merely indistinguishable from it. On exchange rates the tied set contains every method tested; the benchmark detects no difference between any of the thirteen, which is the right answer for that domain rather than a failure of the design.

<table><tr><td>Model</td><td>Wikipedia (daily)</td><td>Wikipedia (weekly)</td><td>Wikipedia (monthly)</td><td>Weather (hourly)</td><td>Air quality (hourly)</td><td>Electricity (hourly)</td><td>Exchange rates (daily)</td></tr><tr><td>Classical</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SeasonalNaive</td><td>1.080</td><td>1.642</td><td>0.822</td><td>1.359</td><td>1.142</td><td>1.443</td><td>2.405</td></tr><tr><td>Theta</td><td>0.961</td><td>1.499</td><td>0.746</td><td>1.280</td><td>1.120</td><td>1.045</td><td>2.448</td></tr><tr><td>AutoETS</td><td>1.149</td><td>1.817</td><td>0.941</td><td>1.558</td><td>1.514</td><td>1.255</td><td>2.468</td></tr><tr><td>AutoARIMA</td><td>0.928</td><td>1.505</td><td>0.853</td><td>1.191</td><td>1.024</td><td>1.183</td><td>2.502</td></tr><tr><td>Trained per dataset</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LightGBM</td><td>1.162</td><td>3.772</td><td>1.545</td><td>1.118</td><td>0.983</td><td>1.260</td><td>3.227</td></tr><tr><td>LSTM</td><td>0.915</td><td>1.202</td><td>0.786</td><td>1.269</td><td>0.988</td><td>1.376</td><td>2.859</td></tr><tr><td>NBEATS</td><td>0.870</td><td>1.326</td><td>0.819</td><td>1.093</td><td>0.903</td><td>1.227</td><td>2.539</td></tr><tr><td>Pretrained</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChronosBoltSmall</td><td>0.790</td><td>1.132</td><td>0.641</td><td>1.067</td><td>0.923</td><td>1.200</td><td>2.614</td></tr><tr><td>ChronosBoltBase</td><td>0.774</td><td>1.143</td><td>0.635</td><td>1.055</td><td>0.915</td><td>1.279</td><td>2.501</td></tr><tr><td>Chronos2</td><td>0.734</td><td>1.106</td><td>0.620</td><td>1.043</td><td>0.895</td><td>1.155</td><td>2.390</td></tr><tr><td>TimesFM TimesFM3</td><td>0.695 0.689</td><td>1.089 1.073</td><td>0.621 0.623</td><td>1.073 1.069</td><td>0.903 0.901</td><td>1.201</td><td>2.400</td></tr><tr><td>Moirai2</td><td>0.748</td><td>1.135</td><td>0.629</td><td>1.067</td><td>0.906</td><td>1.157 1.185</td><td>2.418 2.517</td></tr></table>

Table 1: Mean MASE by group; lower is better, best per column in bold.

<table><tr><td>Model</td><td>Wikipedia (daily)</td><td>Wikipedia (weekly)</td><td>Wikipedia (monthly)</td><td>Weather (hourly)</td><td>Air quality (hourly)</td><td>Electricity (hourly)</td><td>Exchange rates (daily)</td></tr><tr><td>Chronos2</td><td>5.29</td><td>5.31</td><td>5.20</td><td>5.48</td><td>5.76</td><td>5.78</td><td>6.28</td></tr><tr><td>TimesFM3</td><td>4.58</td><td>4.89</td><td>5.44</td><td>5.88</td><td>5.68</td><td>5.70</td><td>7.14</td></tr><tr><td>TimesFM</td><td>4.64</td><td>4.98</td><td>5.22</td><td>6.17</td><td>5.89</td><td>6.89</td><td>5.93</td></tr><tr><td>Moirai2</td><td>5.15</td><td>5.83</td><td>5.15</td><td>5.96</td><td>5.76</td><td>6.26</td><td>6.76</td></tr><tr><td>ChronosBoltBase</td><td>6.10</td><td>6.00</td><td>5.53</td><td>5.89</td><td>6.31</td><td>7.09</td><td>6.79</td></tr><tr><td>ChronosBoltSmall</td><td>6.31</td><td>6.21</td><td>6.16</td><td>6.10</td><td>6.47</td><td>6.65</td><td>8.17</td></tr><tr><td>NBEATS</td><td>7.24</td><td>7.15</td><td>7.59</td><td>6.31</td><td>6.16</td><td>7.33</td><td>6.72</td></tr><tr><td>Theta</td><td>7.90</td><td>7.51</td><td>7.09</td><td>8.18</td><td>8.15</td><td>5.04</td><td>6.48</td></tr><tr><td>AutoARIMA</td><td>8.03</td><td>8.21</td><td>8.50</td><td>7.45</td><td>7.15</td><td>6.35</td><td>6.97</td></tr><tr><td>LSTM</td><td>8.31</td><td>6.69</td><td>7.73</td><td>8.24</td><td>7.44</td><td>9.15</td><td>8.79</td></tr><tr><td>AutoETS</td><td>9.31</td><td>8.79</td><td>8.01</td><td>9.46</td><td>10.13</td><td>7.24</td><td>6.31</td></tr><tr><td>SeasonalNaive</td><td>9.13 9.01</td><td>8.24 11.19</td><td>8.30</td><td>9.18</td><td>8.90</td><td>10.39</td><td>6.41</td></tr><tr><td>LightGBM</td><td></td><td></td><td>11.08</td><td>6.71</td><td>7.21</td><td>7.13</td><td>8.24</td></tr></table>

Table 2: Average rank across series (lower is better). Underlined entries are statistically indistinguish able from the best model in that column under the Nemenyi interval.

## 4.2 The seven groups, one at a time

Wikipedia pageviews (500 series each, daily / weekly / monthly). The strongest result for pretrained models anywhere in the study, and the least surprising once §4.3 is read. TimesFM-3 ranks first on daily and weekly with only two models tied with it; Moirai-2 takes monthly. Every classical method is clearly separated below. These series are bursty — a death, an election or a film release moves a page by an order of magnitude overnight — and a method fitted to the preceding few hundred observations has no way to anticipate the shape of such a burst, whereas a model trained on a decade of them does.

Weather (300 series, hourly). Chronos-2 first, with five models tied. The daily cycle here is close to deterministic, and the interesting observation is that this makes the pretrained advantage smaller, not larger: when the structure is clean, a method estimated on the series itself recovers most of it, and there is less left for a prior to supply.

Air quality (379 series, hourly). Same sampling rate and same daily cycle as weather, but spiky, and the pretrained models pull further ahead than they do on weather. This is the comparison the domain was added for, and it rules out the simplest explanation of the weather result — that pretrained models are simply good at hourly data.

Electricity (46 series, hourly). The one group a classical method wins. Theta ranks first, six models are tied with it, and no pretrained model is separated from it in either direction. Load and generation on a national grid are the most regular series in the study: strong daily and weekly cycles, physically bounded, with little of the burst behaviour that Wikipedia has. It is also the domain the foundation-model literature most often claims, and the claim is normally supported on ETT — data from 2016–2018 that sits in every pretraining corpus.

Exchange rates (29 series, daily). The Friedman test does not reject. Every method, seasonal naive included, sits in one undifferentiated tied set. We include this group precisely so the benchmark can produce that answer: a design that cannot return “no difference” on a domain where no difference exists is not measuring what it claims to measure. The small panel widens the intervals, and we would not read the null as proof of no effect — but the point estimate for every model is also essentially the naive one.

<table><tr><td>Group</td><td>Seasonal strength</td><td>Spectral entropy</td><td>Pretrained gain</td></tr><tr><td>Wikipedia (daily)</td><td>0.10</td><td>0.93</td><td>+25.8%</td></tr><tr><td>Wikipedia (weekly)</td><td>0.00</td><td>0.92</td><td>+28.4%</td></tr><tr><td>Wikipedia (monthly)</td><td>0.06</td><td>0.87</td><td>+16.9%</td></tr><tr><td>Weather (hourly)</td><td>0.77</td><td>0.69</td><td>+12.5%</td></tr><tr><td>Air quality (hourly)</td><td>0.34</td><td>0.84</td><td>+12.6%</td></tr><tr><td>Electricity (hourly)</td><td>0.47</td><td>0.83</td><td>-10.6%</td></tr><tr><td>Exchange rates (daily)</td><td>0.00</td><td>0.93</td><td>n.s.</td></tr></table>

Table 3: Series properties measured on the input window only, against the observed advantage. Neither column orders the last one.

## 4.3 The intrinsic explanation, and why we abandoned it

Ordering the groups by the gap between the best pretrained model and the best classical one gives a gradient: largest on Wikipedia traffic, smaller on weather and air quality, negative on electricity, null on exchange rates. The natural hypothesis is that some property of the series explains it, and the natural candidates are periodicity and noise — pretraining should help most where a repeating structure exists but is hard to estimate from a few hundred observations alone.

We tested that directly. For each group we measured, on the input window only, the median seasonal strength (from an STL decomposition at the group’s seasonal period) and the median spectral entropy of the differenced series, both as defined in the feature literature used to characterise the M4 panel (Table 3). Neither explains the pattern. Seasonal strength is if anything negatively associated with the advantage: Wikipedia series have almost no measurable seasonality at the tested period and show the largest gains, while weather is the most strongly seasonal group in the study and shows a middling one. Spectral entropy fails in the other direction: exchange rates are as high-entropy as Wikipedia traffic and show no advantage at all. With seven groups no correlation here is statistically meaningful in any case, and we report the measurement rather than a story fitted to it.

## 4.4 Corpus familiarity survives the hold-out

What does line up with the gradient is not a property of the series but of the models’ training data. TimesFM’s authors describe its pretraining corpus as dominated by Wikipedia pageviews — on the order of $1 0 ^ { 1 1 }$ time points at hourly, daily, weekly and monthly granularity, running to late 2023. Our three Wikipedia groups are pageviews at daily, weekly and monthly granularity, beginning in 2026. The hold-out is clean in time and identical in kind.

A domain being both the largest corpus component and the largest measured gain is suggestive, but it is one coincidence and admits a benign reading: perhaps Wikipedia traffic is simply a domain where all pretrained models do well. We can separate the two, because the pretrained models forecast the same series. Any property of those series — difficulty, seasonality, noise — applies equally to all of them and cancels in a within-family comparison. What does not cancel is what each model read during pretraining.

Ranking the six pretrained models against each other on every series, the TimesFM family sits -0.53 ranks below the Chronos family on Wikipedia and only -0.09 below it on the other four domains (1,500 vs. 754 series; Mann–Whitney $p < 1 0 ^ { - 5 }$ , rank-biserial -0.13). The model with Wikipedia in its corpus is specifically better at Wikipedia, and only there. The effect is modest in size, and we present it as such — but its direction is not something the series can account for.

We therefore read the gradient as a corpus-overlap effect rather than a capability one, and note that this is a hypothesis our design supports rather than a demonstrated mechanism. The decisive experiment is a benchmark whose domains are stratified by membership in each model’s disclosed corpus, which requires disclosure the field does not currently provide. That, rather than a further increase in the size of the pretraining set, seems to us the useful next contribution.

## 4.5 What this means for evaluation

A temporal hold-out is necessary and not sufficient. It rules out the model having seen these observations; it does not rule out the model having spent its training on this kind of series, and the second effect is large enough to reorder a leaderboard. Reporting a single average across domains hides it entirely, because the average is dominated by whichever domains happen to be in the benchmark, and those are the well-published ones — which are also the ones most likely to be in a pretraining corpus.

## 4.6 Probabilistic forecasts, where the ordering changes

Point accuracy is only half of what these models produce, and the half that is usually reported. Every probabilistic method here also emits nine quantiles, from which we compute weighted quantile loss and the empirical coverage of the nominal 80% interval (Table 4). Two things appear that the point metrics hide.

<table><tr><td>Model</td><td>Wikipedia (daily)</td><td>Wikipedia (weekly)</td><td>Wikipedia (monthly)</td><td>Weather (hourly)</td><td>Air quality (hourly)</td><td>Electricity (hourly)</td><td>Exchange rates (daily)</td></tr><tr><td colspan="6">Weighted quantile loss (lower is better)</td><td></td><td></td><td></td></tr><tr><td>AutoARIMA</td><td>0.547</td><td>0.514</td><td>0.518</td><td>0.140 0.190</td><td>0.306</td><td>0.313</td><td></td><td>0.003</td></tr><tr><td>AutoETS</td><td>†</td><td>†</td><td>4.274</td><td>0.122</td><td></td><td>0.479</td><td>0.377</td><td>0.004</td></tr><tr><td>Chronos2</td><td>0.217</td><td>0.292</td><td>0.334</td><td></td><td></td><td>0.260</td><td>0.300</td><td>0.003</td></tr><tr><td>ChronosBoltBase</td><td>0.228</td><td>0.300</td><td>0.335</td><td>0.125</td><td></td><td>0.268</td><td>0.358</td><td>0.004</td></tr><tr><td>ChronosBoltSmall</td><td>0.233</td><td>0.300</td><td>0.340</td><td>0.125</td><td></td><td>0.264</td><td>0.365</td><td>0.004</td></tr><tr><td>LSTM</td><td>0.281</td><td>0.313</td><td>0.412</td><td>0.151</td><td></td><td>0.298</td><td>0.407</td><td>0.005</td></tr><tr><td>Moirai2 NBEATS</td><td>0.222</td><td>0.298</td><td>0.328</td><td>0.124</td><td></td><td>0.266</td><td>0.310</td><td>0.004</td></tr><tr><td>Theta</td><td>0.268</td><td>0.334</td><td>0.399</td><td>0.131</td><td></td><td>0.268</td><td>0.303</td><td>0.004</td></tr><tr><td>TimesFM</td><td>0.701 0.209</td><td>0.627 0.294</td><td>0.530 0.327</td><td>0.160 0.127</td><td></td><td>0.354 0.263</td><td>0.298 0.318</td><td>0.004</td></tr><tr><td>TimesFM3</td><td>0.207</td><td>0.283</td><td>0.329</td><td>0.127</td><td></td><td>0.262</td><td>0.294</td><td>0.003</td></tr><tr><td>80% interval coverage (nominal 0.80)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.003</td></tr><tr><td colspan="9"></td></tr><tr><td>AutoARIMA</td><td>0.96</td><td>0.94</td><td>0.91</td><td></td><td>0.72</td><td>0.82</td><td>0.74</td><td>0.88</td></tr><tr><td>AutoETS</td><td>0.94</td><td>0.95</td><td></td><td>0.88</td><td>0.80</td><td>0.89</td><td>0.83</td><td>0.88</td></tr><tr><td>Chronos2</td><td>0.76</td><td>0.72</td><td></td><td>0.70</td><td>0.73</td><td>0.74</td><td>0.65</td><td>0.75</td></tr><tr><td>ChronosBoltBase</td><td>0.79</td><td>0.75</td><td></td><td>0.73</td><td>0.73</td><td>0.72</td><td>0.62</td><td>0.76</td></tr><tr><td>ChronosBoltSmall</td><td>0.78</td><td>0.77</td><td></td><td>0.75</td><td>0.74</td><td>0.74</td><td>0.64</td><td>0.72</td></tr><tr><td>LSTM</td><td>0.84</td><td>0.75</td><td></td><td>0.86</td><td>0.76</td><td>0.80</td><td>0.61</td><td>0.63</td></tr><tr><td>Moirai2</td><td>0.74</td><td>0.71 0.69</td><td></td><td>0.70</td><td>0.74</td><td>0.73</td><td>0.68</td><td>0.66</td></tr><tr><td>NBEATS</td><td>0.78 0.97</td><td>0.94</td><td></td><td>0.78 0.93</td><td>0.78 0.85</td><td>0.82</td><td>0.65</td><td>0.55</td></tr><tr><td>Theta</td><td>0.82</td><td>0.77</td><td></td><td>0.70</td><td>0.76</td><td>0.83 0.78</td><td>0.86 0.65</td><td>0.87</td></tr><tr><td>TimesFM TimesFM3</td><td>0.79</td><td>0.74</td><td></td><td>0.72</td><td>0.74</td><td>0.75</td><td></td><td>0.75</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.69</td><td>0.74</td></tr></table>

Table 4: Probabilistic accuracy and calibration. † marks a divergence (see §5), not a large loss. Coverage should be read against a nominal 0.80: below is overconfident, above is conservative.

First, the ranking by quantile loss largely agrees with the ranking by MASE, which is reassuring and not very interesting. Second, and less comfortably, the calibration splits cleanly along family lines. Every pretrained model’s 80% interval covers less than 80% of outcomes — the worst, Moirai2, by 9 percentage points — while all three automatic classical methods cover more than nominal, Theta by 9 points. The pretrained models are systematically overconfident and the classical ones systematically cautious, consistently, across every domain in the study.

This matters more than the accuracy gap for anyone using a forecast to size a buffer, a reserve or an inventory. A model that is 10% better on MASE and whose 80% interval is really a 71% interval is not obviously the better tool for that job, and nothing in the point-accuracy tables that dominate this literature would tell you so.

The direction is familiar from elsewhere in deep learning: Guo et al. [2017] documented the same overconfidence in modern classifiers, and post-hoc recalibration [Kuleshov et al., 2018] and conformal methods for time series [Stankeviciˇ ut¯ e et al., 2021] are the established remedies. We propose none˙ of them here. The contribution is the measurement — that this well-known pathology is present, consistently and by family, in the pretrained forecasters currently being recommended for zero-shot use, and that no published leaderboard reports it.

## 4.7 Cost

The accuracy differences among the leading pretrained models are small enough to sit inside their rank intervals in most groups. Their costs are not comparable in the same way. An automatic ARIMA search costs a median 54× more per series than a pretrained forward pass, and that factor is not a constant: it runs from 5× on weekly series to 2,689× on weather (hourly), since the order search scales with series length and seasonal period while a forward pass does not. On the 7 groups where it completed, it was beaten on accuracy by the best pretrained model in 7. For a practitioner the comparison that matters is therefore not among the pretrained models, whose differences are mostly not resolvable, but between that family and the automated classical search it would replace — and there the case is strongest exactly where the search is most expensive.

![](images/530d4256472e0bd310d75217d5fda1e91d3f57b359bd7bb32570df84e45b0b48.jpg)

Figure 2: Cost against accuracy, aggregated over the seven groups. The pretrained models occupy a tight cluster at the top left — cheapest and most accurate — while the automatic ARIMA search sits two orders of magnitude to the right of them and ranks worse than a Theta baseline that costs a hundredth as much.
<table><tr><td>Model</td><td>Median s/series</td><td>Relative to ChronosBoltSmall</td></tr><tr><td>SeasonalNaive</td><td>0.0000</td><td>free</td></tr><tr><td>ChronosBoltSmall</td><td>0.0125</td><td>1.0×</td></tr><tr><td>Moirai2</td><td>0.0141</td><td>1.1×</td></tr><tr><td>ChronosBoltBase</td><td>0.0174</td><td>1.4×</td></tr><tr><td>Chronos2</td><td>0.0181</td><td>1.4×</td></tr><tr><td>TimesFM3</td><td>0.0269</td><td>2.2×</td></tr><tr><td>Theta</td><td>0.0429</td><td>3.4×</td></tr><tr><td>LightGBM</td><td>0.0550</td><td>4.4×</td></tr><tr><td>NBEATS</td><td>0.0627</td><td>5.0×</td></tr><tr><td>TimesFM</td><td>0.0726</td><td>5.8×</td></tr><tr><td>LSTM</td><td>0.0887</td><td>7.1×</td></tr><tr><td>AutoETS</td><td>0.1362</td><td>11×</td></tr><tr><td>AutoARIMA</td><td>3.8484</td><td>308×</td></tr></table>

Table 5: Median wall-clock seconds per series on identical hardware, including model load.

## 5 Failure modes we hit

Benchmark papers report the runs that worked. The ones that did not are more useful to the next person, and three of ours changed a number that would otherwise have been published.

A metric with no denominator. The Danish grid publishes a hydro-generation column for a price area with essentially no hydro. It is zero throughout, apart from a single $\mathrm { \overline { { 1 0 } } } ^ { - 6 }$ reading. MASE divides by the in-sample seasonal-naive error, which for that series is $1 0 ^ { - 9 }$ , so the two neural models — which predicted a small positive quantity rather than exactly zero — scored $4 \times 1 0 ^ { 8 }$ and $6 \times 1 0 ^ { 7 }$ . One series in forty-seven moved the group’s mean MASE by six orders of magnitude. The rank-based conclusion for that group was unchanged, which is the argument for ranks made concrete rather than in the abstract. The series is now excluded by the rule in §3.2, and we flag that a relative threshold does not catch it: its scale is negligible in absolute terms and enormous relative to itself.

A classical baseline that diverges rather than degrades. AutoETS produces weighted quantile losses of order 10<sup>14</sup> on two of the three Wikipedia groups — marked † in Table 4. This is not an artefact of our harness: the fitted model genuinely explodes on Wikipedia’s spiky, heavy-tailed traffic, whose training ranges span three orders of magnitude within a single series, and it emits negative quantiles for a strictly non-negative quantity. We report it rather than silently substituting a damped variant, because an automatic method that fails this way on real data is a finding about the method. Its point forecasts on the same groups are poor but finite, which is why the failure is invisible in a MASE-only table.

Numerical precision on accelerator hardware. TimesFM raises a type error under float64 on Apple’s Metal backend and must fall back to CPU for the affected series; Moirai-2 required the same treatment. Neither changes the forecasts, but both change the timing, and a cost table that did not disclose it would be comparing a GPU path against a CPU one without saying so. The timings in Table 5 are as measured, with the fallbacks in place, and results/holdout 2026/timing.jsonl records the device actually used for every cell.

What we checked rather than assumed. Only three models are re-run under multiple seeds, on the claim that the other nine are deterministic. That claim is load bearing — if a pretrained model quietly sampled, its single reported number would be one draw and every significance test here would be understated — so we tested it rather than asserting it. Each of the nine was run twice under different seeds and the point forecasts compared exactly: all nine agree to the last bit (max |∆| = 0), and the check is released as scripts/check determinism.py with its output in results/holdout 2026/determinism <group>.json. Across the three stochastic models, the spread in mean MASE over three seeds is below 8% on every group.

## 6 Limitations

The corpus-overlap finding is an association, not a demonstrated mechanism. It rests on one corpus whose composition is publicly described and one domain that matches it; a model could be better at Wikipedia traffic for reasons unrelated to having read Wikipedia traffic, and our within-family comparison narrows that possibility without closing it. The effect is also modest — a rank-biserial correlation of -0.13 — and we resist reading a large practical consequence into it. Settling the question needs a benchmark stratified by corpus membership across several models, which in turn needs disclosure that most published corpora do not provide.

Beyond that: the hold-out is one forecast origin per group, where a rolling-origin evaluation would estimate variance across time as well as across series. Two domains contribute fewer than fifty series, so their rank intervals are wide, and the electricity result should be read as “no evidence of an advantage” rather than a demonstrated deficit. Pretraining cutoffs are taken from model documentation and cannot be independently verified; where a cutoff is undisclosed we substitute the release date. And the study covers five domains, more than the standard two but far from the diversity of real forecasting practice — with the caveat, central to our argument, that adding whichever domains are easiest to obtain is likely to add exactly the well-published ones that pretraining corpora already contain.

## 7 Conclusion

Evaluated on data that postdates them, time-series foundation models are strong on some domains, unnecessary on others, and useless where nothing works. We set out to explain that spread with a property of the series and could not: neither periodicity nor entropy orders it. What orders it is which domains the models were raised on. The largest advantage in our study falls on the domain that constitutes the bulk of one model’s pretraining corpus, at the same granularities, and it is that model’s family which holds the advantage there and nowhere else.

The practical consequence for benchmark design is that moving the test window forward is not enough. It is necessary — without it, nothing can be concluded at all — but it controls only for having seen these observations, not for having spent a hundred billion time points learning what this kind of observation looks like. A benchmark that wants to measure generalisation has to hold out domains, not just dates, and it can only do so if pretraining corpora are disclosed well enough to say which domains those are. The practitioner’s version of the same point is shorter: before asking which model is best, ask whether your data looks like the data it was raised on.

We release the fetchers, the harness and the per-series losses so that both the negative result and the corpus effect can be tested on domains we did not consider.

## 8 Dataset documentation

What we release, and what we do not. We release the fetchers, not the data. Every group is rebuilt by a script in data/sources/ that retrieves the observations directly from the publisher, with no credentials, and writes a gzipped CSV alongside a manifest recording the request parameters, the retrieval timestamp, the series count and a SHA-256 of the output. Redistributing the observations ourselves would add a copy that can drift from the source and would put us in the position of relicensing data we do not own; a fetcher plus a checksum gives a reader the same panel without either problem. The consequence is that a rebuild months from now returns a longer series than ours, which is why the forecast origin is fixed in the harness rather than implied by the file.

Sources and terms. Wikimedia pageviews come from the Wikimedia REST API; weather and air quality from Open-Meteo’s ERA5 and CAMS archives; electricity from Energinet’s Energi Data Service; exchange rates from the European Central Bank’s SDMX API. All four are institutional open-data services that require no registration. Each publisher’s own terms govern reuse of its data, and users of this benchmark are subject to them; the code we release is ours and carries the repository’s licence.

Maintenance and drift. The panel is defined by a rule — the full city list, the full currency list, the publisher’s own column set — and not by a frozen file, so it is stable under re-fetching except where a publisher adds or retires a series. The manifests make such a change visible rather than silent: a differing series count or checksum against the values we report is the signal that the panel has moved. We regard this as the correct trade for a benchmark whose entire premise is that the test window must keep moving forward. A frozen file would become contaminated the moment the next generation of models is trained on it, which is precisely the failure the paper documents.

Machine-readable metadata. The repository carries a Croissant record (croissant.json) describing all five sources, with the series count, date range and SHA-256 of each, generated from the fetchers’ own manifests by scripts/make croissant.py rather than written by hand — a hand-copied checksum is one that will be wrong after the next re-fetch.

Known limitations of the data. Two groups are small (46 and 29 series), which widens their rank intervals materially. The weather and air-quality panels are reanalysis products rather than station readings, so they are smoother than instrument data would be. The electricity panel covers one country’s grid. And every domain here is one that somebody chose to publish continuously and openly, which is a biased sample of forecasting problems — biased, specifically, towards the kind of data that ends up in a pretraining corpus, which is a caveat that cuts against our own headline effect as much as for it.

## Reproducibility

Code, data fetchers and per-series results: https://github.com/mahdinaser/tsfm-bench

Every dataset is rebuilt by a script in data/sources/ requiring no credentials. scripts/run holdout.sh reproduces the runs, scripts/significance.py the tests, and scripts/make tables.py every table and every number quoted above. Per-series losses are released alongside the aggregates so that any alternative test can be applied without re-running the models.

## References

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. GIFT-Eval: A benchmark for general time series forecasting model evaluation. arXiv preprint arXiv:2410.10393, 2024.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. arXiv:2403.07815.

Vassilios Assimakopoulos and Konstantinos Nikolopoulos. The theta model: A decomposition approach to forecasting. International Journal of Forecasting, 16(4):521–530, 2000.

Chronos Team, Amazon Web Services. Chronos-2. arXiv preprint arXiv:2510.15821, 2025. Identifier taken from the authors’ repository; not independently confirmed. Chronos-Bolt has no separate publication and is documented by its model card.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In International Conference on Machine Learning (ICML), volume 235 of PMLR, pages 10148–10167, 2024. arXiv:2310.10688.

Janez Demsar. Statistical comparisons of classifiers over multiple data sets. ˇ Journal of Machine Learning Research, 7:1–30, 2006.

Francis X. Diebold and Roberto S. Mariano. Comparing predictive accuracy. Journal ofBusiness & Economic Statistics, 13(3):253–263, 1995. doi: 10.1080/07350015.1995.10524599.

Vijay Ekambaram, Arindam Jati, Pankaj Dayama, Sumanta Mukherjee, Nam H. Nguyen, Wesley M. Gifford, Chandra Reddy, and Jayant Kalagnanam. Tiny time mixers (TTMs): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series. In Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2401.03955.

Azul Garza, Max Mergenthaler Canseco, Cristian Challu, and Kin G. Olivares. StatsForecast:´ Lightning fast forecasting with statistical and econometric models. PyCon Salt Lake City, Utah, US, 2022. Software; no peer-reviewed publication exists. https://github.com/Nixtla/ statsforecast.

Azul Garza, Cristian Challu, and Max Mergenthaler-Canseco. Timegpt-1. arXiv preprint arXiv:2310.03589, 2023.

Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359–378, 2007. doi: 10.1198/016214506000001437.

Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob J. Hyndman, and Pablo Montero-Manso. Monash time series forecasting archive. In NeurIPS Datasets and Benchmarks Track, 2021.

Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. MOMENT: A family of open time-series foundation models. In International Conference on Machine Learning (ICML), volume 235 of PMLR, 2024. arXiv:2402.03885.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning (ICML), volume 70 of PMLR, 2017. arXiv:1706.04599.

Rob J. Hyndman and Anne B. Koehler. Another look at measures of forecast accuracy. International Journal ofForecasting, 22(4):679–688, 2006. doi: 10.1016/j.ijforecast.2006.03.001.

Rob J. Hyndman, Anne B. Koehler, J. Keith Ord, and Ralph D. Snyder. Forecasting with Exponential Smoothing: The State Space Approach. Springer, 2008.

Stephan Kolassa. Why the “best” point forecast depends on the error or accuracy measure. International Journal ofForecasting, 36(1):208–211, 2020. doi: 10.1016/j.ijforecast.2019.02.017.

Alex J. Koning, Philip Hans Franses, Michele Hibon, and H. O. Stekler. The M3 competition:\` Statistical tests of the results. International Journal ofForecasting, 21(3):397–409, 2005.

Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon. Accurate uncertainties for deep learning using calibrated regression. In International Conference on Machine Learning (ICML), volume 80 of PMLR, 2018. arXiv:1807.00263.

Hongkai Li, Shifeng Xie, Lefei Shen, Zhuo Li, Mouxiang Chen, Xiaobin Zhang, Han Fu, Jianling Sun, Xiaoxue Ren, and Chenghao Liu. TSFMAudit: Data contamination auditing in forecasting time series foundation models. arXiv preprint arXiv:2605.26161, 2026.

Bryan Lim, Sercan O. Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for<sup>¨</sup> interpretable multi-horizon time series forecasting. International Journal ofForecasting, 37(4): 1748–1764, 2021. doi: 10.1016/j.ijforecast.2021.03.012.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. iTransformer: Inverted transformers are effective for time series forecasting. In International Conference on Learning Representations (ICLR), 2024a. Spotlight. arXiv:2310.06625.

Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Timer: Generative pre-trained transformers are large time series models. In International Conference on Machine Learning (ICML), 2024b. arXiv:2402.02368.

Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Sundial: A family of highly capable time series foundation models. arXiv preprint arXiv:2502.00816, 2025. Reported as ICML 2025 Oral by the authors’ repository; not stated on arXiv.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. Statistical and machine learning forecasting methods: Concerns and ways forward. PLOS ONE, 13(3):e0194889, 2018. doi: 10.1371/journal.pone.0194889.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The M4 competition: 100,000 time series and 61 forecasting methods. International Journal ofForecasting, 36(1):54–74, 2020. doi: 10.1016/j.ijforecast.2019.04.014.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. M5 accuracy competition: Results, findings, and conclusions. International Journal ofForecasting, 38(4):1346–1364, 2022.

Marcel Meyer, Sascha Kaltenpoth, Kevin Zalipski, and Oliver Muller. Rethinking evaluation in the ¨ era of time series foundation models: (un)known information leakage challenges. arXiv preprint arXiv:2510.13654, 2025.

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations (ICLR), 2023. arXiv:2211.14730.

Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations (ICLR), 2020. arXiv:1905.10437.

Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Bilos, Hena Ghonia, Nadhir Hassen, et al. Lag-llama: Towards ˇ foundation models for probabilistic time series forecasting. arXiv preprint arXiv:2310.08278, 2023.

Oscar Sainz, Jon Campos, Iker Garc´ıa-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10776–10787, 2023. arXiv:2310.18018.

David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3): 1181–1191, 2020. doi: 10.1016/j.ijforecast.2019.07.001.

Oleksandr Shchur, Abdul Fatir Ansari, Caner Turkmen, Lorenzo Stella, Nick Erickson, Pablo Guerron, Michael Bohlke-Schneider, and Yuyang Wang. fev-bench: A realistic benchmark for time series forecasting. arXiv preprint arXiv:2509.26468, 2025.

Kamile Stankevi˙ ciˇ ut¯ e, Ahmed M. Alaa, and Mihaela van der Schaar. Conformal time-series forecast-˙ ing. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, 2021.

Mingtian Tan, Mike A. Merrill, Vinayak Gupta, Tim Althoff, and Thomas Hartvigsen. Are language models actually useful for time series forecasting? In Advances in Neural Information Processing Systems (NeurIPS), 2024. Spotlight. arXiv:2406.16964.

William Toner, Thomas L. Lee, Artjom Joosen, Rajkarn Singh, and Martin Asenov. Performance of zero-shot time series foundation models on cloud data. In ICLR Workshop: I Can’t Believe It’s Not Better, 2025. arXiv:2502.12944.

Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. In International Conference on Machine Learning (ICML), volume 235 of PMLR, pages 53140–53164, 2024. arXiv:2402.02592 (identifier not re-checked against the abstract page).

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 22419–22430, 2021. arXiv:2106.13008.

Cheng Xu, Shuhao Guan, Derek Greene, and M-Tahar Kechadi. Benchmark data contamination of large language models: A survey. arXiv preprint arXiv:2406.04244, 2024a.

Zongzhe Xu, Ritvik Gupta, Wenduo Cheng, Alexander Shen, Junhong Shen, Ameet Talwalkar, and Mikhail Khodak. Specialized foundation models struggle to beat supervised baselines. arXiv preprint arXiv:2411.02796, 2024b.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 11121–11128, 2023. doi: 10.1609/aaai.v37i9.26317.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 35, pages 11106–11115, 2021. arXiv:2012.07436.