# Random Exploration is Random

Yingkang Luo<sup>1,</sup> <sup>2\*</sup>, Hua-Dong Xiong<sup>2</sup>, Hanbo Xie<sup>2</sup>, Caroline Phelps<sup>2</sup>, Madeline Harms<sup>3</sup>, Ryan Smith<sup>4</sup>, James Waltz<sup>5</sup>, Robert Wilson<sup>2\*</sup>

<sup>1</sup>College of Computing, Georgia Institute of Technology . <sup>2</sup>School of Psychology, Georgia Institute of Technology . <sup>3</sup>Department of Psychology, University of Minnesota Duluth . <sup>4</sup>Laureate Institute for Brain Research . <sup>5</sup>School of Medicine, University of Maryland .

\*Corresponding author(s). E-mail(s): yluo344@gatech.edu; rwilson337@gatech.edu;

Contributing authors: hdx@gatech.edu; hanboxie1997@gatech.edu; cphelps35@gatech.edu; madeline.harms@gmail.com; rsmith@laureateinstitute.org; JWaltz@som.umaryland.edu;

## Abstract

Decision making often requires balancing exploitation of familiar options with exploration of uncertain alternatives. One strategy for exploration is random exploration, in which choices become more variable and less tied to the option that currently appears best. However, it remains unclear whether this variability reflects genuine stochasticity or instead a deterministic structure that traditional cognitive models fail to capture. We examined this question using the Horizon Task in an aggregated dataset of 1,392 participants completing over 1,270,695 trials. In this task, exploration becomes more valuable when participants have more future choices, and choice variability increases under longer horizons. We tested whether more expressive neural network models could recover deterministic structure from this additional variability. Although neural networks provided better overall fits to behavior than the traditional model, long-horizon choices remained consistently less predictable than short-horizon choices. These findings show that limited model expressiveness cannot fully account for the increased variability associated with random exploration, suggesting random exploration truly is substantially random.

Keywords: Random Exploration, Neural Networks, Explore-exploit, Decision-making

## 1 Introduction

When dining out, do you order a familiar favorite or try something diferent? You can’t go wrong with a steak, but the sea urchin stir-fry might taste even better. There are also spontaneous moments when we act without quite knowing what is driving us: the server arrives, and suddenly we hear ourselves ordering the fish-head soup! Every time we make a decision, we are facing the explore-exploit dilemma: choosing between exploiting what we already know and exploring uncertain alternatives that may improve future decisions [1–4]. It can be as simple as deciding what to eat or as complex as choosing a profession. Exploiting a familiar option is safe and predictable, but it may mean missing out on potentially better alternatives. Trying something new may reveal a better option, but it also comes with the risk of disappointment.

Additionally, the value of exploration often depends on how many future opportunities remain. Imagine that you are staying in a new city for several months. Trying an unfamiliar dish may help you discover a favorite that you can return to many times. If you are leaving tomorrow, however, there may be little opportunity to benefit from what you learn. This number of remaining opportunities is known as the horizon. When horizon is long, new information becomes more valuable, encouraging exploration. Conversely, when future opportunities are limited, exploitation becomes more favorable [5–8].

Previous work has identified two strategies for navigating the explore-exploit dilemma: directed and random exploration. Directed exploration involves deliberately favoring more informative options. Random exploration involves noisier choices that are less tied to expected value or the degree of uncertainty [5, 6, 9–19]. This noise can draw an individual away from the option with the highest expected reward. Although introducing randomness, or stochasticity into decision-making might seem counterproductive, it provides an efective and computationally inexpensive way to explore. Stochastic methods are therefore widely used in machine-learning algorithms for search and optimization [20–25]. Consistently choosing the option with the highest expected value risks becoming trapped in a familiar but suboptimal pattern. Stochasticity disrupts this pattern and creates opportunities to discover better options.

However, is behavioral variability commonly attributed to random noise truly random, or does it reflect structured processes that commonly used models fail to identify? Evidence from neuroscience suggests that genuine neural noise may contribute to random exploration. Neural activity is inherently variable, and this variability can produce stochastic behavior [26]. During random exploration, monkeys show increased neural variability and weaker choice-related signals in prefrontal decision-making circuits [27]. At the same time, random exploration appears to be strategically regulated. Like directed exploration, it increases when exploration is more valuable, and studies in rodents and songbirds suggest that this may involve increased neural variability [28, 29]. Together, these findings suggest that apparently random behavior reflects a regulated process.

Furthermore, not all behavioral variability reflects irreducible randomness. Apparent randomness can also arise from noise in how task or action values are learned rather than from noise at the moment of choice [30]. Sensory misestimation, imprecise learning, and simplified beliefs can produce variable behavior even when the individual is attempting to follow a value-maximizing strategy [31, 32]. Wang and colleagues examined this issue by modifying the Horizon Task to present participants with repeated identical games. The task compares choices under short (H1) and long (H6) horizons, where exploration becomes more valuable in H6. Some choice variability was consistent across repetitions, while a substantial portion remained unexplained, supporting the possibility of a genuinely stochastic component [33]. However, repeating the same game only controls the choices and rewards within that game. It does not account for influences from earlier games, gradual changes in strategy, or other unobserved factors [34]. Thus, variability that appears random may still reflect deterministic processes that the model has not captured.

In practice, randomness is identified as variance in behavior that remains unexplained after a model has accounted for the variables of interest. Traditional cognitive models often represent the unexplained choice variability through a decision-noise parameter [1, 35]. However, these simplified assumptions restrict the behavioral relationships they can capture. Some apparently random variability may therefore be systematic but unmodeled, making the model’s expressive power a key limitation.

The current study addresses this problem using deep neural networks in previously collected Horizon Task data with over one million trials [1, 36–42]. Neural networks are powerful function approximators with scalable numbers of trainable parameters and flexible architectures [43–45]. This allows them to capture complex behavioral patterns that traditional cognitive models may have misclassified as randomness. In the Horizon Task, we observed greater behavioral variability at longer horizons, which was attributed as decision noise by the traditional model. If this variability contains additional predictable structure, neural networks should improve prediction more at long- than short-horizon games, reducing the predictability gap between conditions. If long-horizon choices remain less predictable, limited model expressiveness cannot fully explain the additional variability, supporting a genuinely stochastic component of random exploration.

## 2 Results

Before presenting the results, we briefly describe the Horizon Task and provide an overview of our modeling approach.

## 2.1 The Horizon Task

In the Horizon Task [1], participants play a series of games in which they choose between two slot machines to maximize their total reward. Each machine produces rewards from a Gaussian distribution with a diferent mean unknown to the participants. They must therefore balance exploiting the option with the highest expected reward against exploring to gain additional information.

![](images/2060df23164dc1b4baa07d087112cbf725c8f90332f8a1596430c2539ef44818.jpg)  
D

![](images/c8d6fa24f5a35056012b818c517cefdf7770ea4ebba10f645b1597602ddecdd9.jpg)

![](images/d66586a00ffdd2f4d52de42b5afc10313a3f34d556027c95a0d6753285fac5df.jpg)  
Fig. 1 The Horizon Task. (A) The first four instructed trials, during which participants are required to select the highlighted green option between the two slot machines. The structure of the forced trials is identical across horizon conditions, difering only in the total number of trials. (B, C) Following the instructed trials, participants complete free-choice trials until the end of the game. During this phase, they choose freely between the two slot machines. In the short-horizon condition (B), participants make one additional free choice, whereas in the long-horizon condition (C), they make six additional free choices. (D) Illustration of the uncertainty conditions. On the left, the instructed trials reveal an unequal amount of information per machine [1 3]. On the right, the instructed trials reveal an equal amount of information per machine [2 2]. Analyses focus on the first free-choice trial, highlighted in green.

Each game begins with four instructed trials, during which participants are required to choose a specified machine (Fig. 1). These trials control how much information is available about each option before participants make a free choice. In the unequal uncertainty condition, one machine is played once and the other three times ([1 3]). In the equal uncertainty condition, both machines are played twice ([2 2]). After the instructed trials, participants make either one free choice in Horizon 1 (H1) or six free choices in Horizon 6 (H6). In H1, information gained through exploration cannot be used in a later choice. In H6, that information may improve future decisions, making exploration more valuable. We focus primarily on the first free-choice trial, where the available choice and reward information is controlled in the same way across horizon conditions. Comparing behavior between H1 and H6 on this trial allows us to measure horizon-related changes in behavioral variability.

## 2.2 Modeling Overview

We analyzed an aggregated Horizon Task dataset of 1,392 participants. We used the original Horizon Task model as a baseline [1] and confirmed that the dataset reproduced the established efects of horizon on directed and random exploration. We then used neural networks (NNs) to capture additional predictable structure available in the task. To reduce model-specific limitations, we evaluated several architectures with diferent representational capacities. The multilayer perceptron (MLP) received all trial information simultaneously, whereas recurrent neural networks (RNNs) processed choices and rewards sequentially, similar to how information was presented to participants. We also evaluated Transformer-based models to capture flexible dependencies across trials.

We first tested whether the neural networks captured more predictable behavioral structure than the traditional cognitive model. Then, we examined whether this additional predictive capacity reduced the H1–H6 predictability gap across several complementary analyses. In the simplest case, we modeled behavior at the pooled-population level using only the instructed-trial choice–reward sequence, horizon condition, and uncertainty condition. We then incorporated subject identity, allowing the models to learn individual behavioral patterns. To rule out learning efects from prior games, we repeated the analysis using only the first game completed by each participant. We next tested for cross-game history efects using RNNs with a learned persistence parameter that carried information forward from previous games within each participant. Finally, we tracked predictive performance across all six free-choice trials in H6 to examine how predictability changed as the remaining horizon shortened.

## 2.3 Unexplained variance of the traditional cognitive model increases with horizon

We first fitted the standard Horizon Task logistic model to establish a baseline for behavioral variability explained by the model. Consistent with previous findings, participants showed greater information seeking in H6 than in H1 (Fig. 2A). This was reflected in a larger information bonus in the long-horizon condition (t(1389) = −15.74, p < .001) (Fig. 2C).

![](images/e6879b9d1b89b0c888fdece5af3a4a02c6d788288b4cc7e282dcd750c6597e38.jpg)

![](images/1c1197c1e9b7ce282439969f2294060c055e33c1faf5e7711cc7d6f13fc7a455.jpg)

C  
![](images/fb9413fcaa05988f3b9b8e00b3c0bde2e2dfab2500ea14f7115a63dd9899714a.jpg)

![](images/a8a108f679bac28e232be939d0424f559f38df56911676d25042d5631262719b.jpg)

![](images/99d97e039857016305bf0c9581f679e78728c65c82bfd5e4a4a5d5079d856cc9.jpg)  
Fig. 2 Choice curves of the Horizon Task. (A, B) Choice curves showing the probability of selecting an option as a function of the diference in observed mean rewards. (A) Probability of choosing the more informative option in the unequal uncertainty condition (i.e., the machine played once during the forced trials). The observed mean reward diference is the reward of the more informative option minus that of the less informative option. (B) Probability of choosing the left option in equal condition as a function of the observed mean reward diference between the left and right options. (C, D, E) Parameter estimates from logistic regression fits across horizon conditions. (C) Information bonus in the unequal uncertainty condition. (D, E) Decision noise parameters in the unequal and equal uncertainty conditions, respectively. \*\*\* denotes a significant diference between horizon conditions at $p < 0 . 0 0 1$

Additionally, the model attributed more behavioral variability to decision noise as horizon increased (Fig. 2B). Choices were more variable in H6 than in H1, and the fitted decision-noise parameter was higher in H6 in both uncertainty conditions (Fig. 2D,E). Decision noise increased significantly in the equal uncertainty condition $( [ 2 ~ 2 ] \colon { t } ( 1 3 8 9 ) = - 1 4 . 9 7 , \ : p < . 0 0 1 )$ and in the unequal uncertainty condition ([1 3]: $t ( 1 2 6 7 ) = - 1 2 . 2 5 , p < . 0 0 1 )$ ).

To mirror the neural network analyses, we also evaluated the logistic model as an out-of-sample predictive baseline under diferent fitting procedures (Table 1). The global fit estimated a single set of parameters across the pooled population, similar to the population-level fitting commonly used in neural network models [46]. The individual fit instead estimated a separate set of logistic-model parameters for each participant, which were then averaged for group-level summaries. Under both procedures, H6 choices were harder to predict than H1 choices. The H1–H6 accuracy diference was 7.5 percentage points for the global fit and 6.1 percentage points for the individual fits. We also evaluated the logistic model using only the first game completed by each participant. The predictability gap persisted, with an accuracy of 0.881 in H1 compared with 0.767 in H6. Thus, the lower predictability of H6 behavior was present before participants accumulated experience across repeated games. Overall, the aggregated dataset reproduced the established Horizon Task efects and showed that long-horizon behavior was less predictable across fitting procedures and from the first game.

Table 1 Horizon-specific test accuracy of the logistic regression baselines.
<table><tr><td>Baseline fitting procedure</td><td>H1 accuracy</td><td>H6 accuracy</td><td>H1-H6 difference</td></tr><tr><td>Global fit</td><td>0.807</td><td>0.732</td><td>0.075</td></tr><tr><td>Individual fits</td><td>0.799</td><td>0.737</td><td>0.061</td></tr><tr><td>First-game fit</td><td>0.881</td><td>0.767</td><td>0.113</td></tr></table>

## 2.4 Neural network models fit better than the baseline cognitive model

The traditional Horizon Task model has only a small number of free parameters, which limits the range of behavioral structure it can capture. As a result, deterministic patterns that fall outside the model may be absorbed into the decision-noise term and appear stochastic. We therefore tested whether more expressive neural network models provided a better fit to behavior. The neural network models captured more behavioral structure than the baseline logistic model. We compared model fit using McFadden’s pseudo- $R ^ { 2 }$ relative to the null model (Table 2). The logistic regression baseline achieved a pseudo- $R ^ { 2 }$ of 0.217. In contrast, all neural network models from the subject-level analysis achieved higher values, ranging from 0.312 to 0.385. Overall, the neural network models increased McFadden’s pseudo- $R ^ { 2 }$ by 0.095–0.168 relative to the logistic regression baseline.

Table 2 Model comparison using NLL and McFadden’s pseudo- $R ^ { 2 }$
<table><tr><td>Model</td><td>NLL</td><td>Pseudo-R²</td><td>∆R² vs. Baseline</td></tr><tr><td>MLP</td><td>0.4440</td><td>0.3593</td><td>0.1426</td></tr><tr><td>RNN</td><td>0.4604</td><td>0.3356</td><td>0.1189</td></tr><tr><td>LSTM</td><td>0.4282</td><td>0.3821</td><td>0.1655</td></tr><tr><td>GRU</td><td>0.4350</td><td>0.3722</td><td>0.1556</td></tr><tr><td>TinyGRU</td><td>0.4769</td><td>0.3118</td><td>0.0952</td></tr><tr><td>Transformer</td><td>0.4323</td><td>0.3761</td><td>0.1595</td></tr><tr><td>Self-Attention</td><td>0.4262</td><td>0.3850</td><td>0.1684</td></tr><tr><td>Baseline</td><td>0.5428</td><td>0.2167</td><td>0.0000</td></tr></table>

## 2.5 Unexplained variance of all neural network models increases with horizon

We first tested the simplest case by training neural network models on the pooled population without subject identity. These models predicted how a typical participant in the pooled population would choose. Across all architectures, test accuracy was higher in H1 than in H6 (Fig. 3 A). Mean H1 accuracy was 0.811 (95% CI: [0.808, 0.815]), compared with 0.731 for H6 (95% CI: [0.727, 0.736]). H1 accuracy was therefore 8.0 percentage points higher than H6 accuracy. Thus, the global neural network models did not close the predictability gap between horizon conditions. H6 choices remained less predictable than H1 choices, consistent with greater exploratory variability in the long-horizon condition.

Adding subject identity allowed us to examine whether individual behavioral patterns accounted for the H1–H6 predictability gap. Test accuracy remained higher in H1 than in H6 across all neural network models with encoded subject identity information (Fig. 3 B). Across models, mean H1 accuracy was 0.819 (95% CI: [0.807, 0.830]), compared with 0.759 for H6 (95% CI: [0.749, 0.770]). H1 accuracy was therefore 6.0 percentage points higher than H6 accuracy. Thus, encoding subject identity improved mean predictive accuracy relative to the logistic baseline but did not eliminate the H1–H6 predictability gap.

Global Comparison Test Accuracy (H1 vs H6)  
![](images/12ef55d8d31cb0dfb716b06df73d3720653bd0cc39ebc9dcbadfd3cc2b5846b6.jpg)

Subject-Level Test Accuracy (H1 vs H6)  
![](images/7ee9dbf642d9ec9d7b12af38a181c3107530786afd294ce92ed3f46712a3b792.jpg)  
Fig. 3 Test accuracy by horizon across neural network architectures. (A) Global analysis, where models predicted choices without participant identity. (B) Subject-level analysis, where participant identity was included as an additional input. Each point shows the test accuracy of one architecture, computed separately for H1 and H6 after restoring the checkpoint with the lowest validation loss. Horizontal error bars indicate 95% Wilson confidence intervals. H1 accuracy is shown in blue and H6 accuracy in red. Dashed vertical lines indicate the corresponding horizon-specific test accuracy of the logistic regression baseline. Abbreviations: MLP, multilayer perceptron; RNN, recurrent neural network; LSTM, long short-term memory; GRU, gated recurrent unit; H1, Horizon 1; H6, Horizon 6.

## 2.6 Unexplained variance of the neural network models increases from the very first game

Behavior may change as participants gain experience with the task, raising the possibility that the H1–H6 predictability gap develops through learning or strategy adaptation across repeated games [34]. To test this possibility, we repeated the analysis using only the first game played by each participant. Test accuracy was higher in H1 than in H6 across all neural network architectures (Fig. 4). Mean H1 accuracy was 0.828 (95% CI: [0.800, 0.856]), compared with 0.753 for H6 (95% CI: [0.731, 0.776]), corresponding to an H1–H6 diference of 7.5 percentage points. The predictability gap was therefore already present during participants’ first exposure to the task, suggesting that it did not arise solely from learning or adaptation across repeated games.

![](images/70217172df683ec610d493f9bfd58556a89feff2cb081fb68dfba43f4831d516.jpg)  
Fig. 4 Predictive performance in the first game. Test accuracy was compared between H1 and H6 using only the first game played by each participant. Each point shows the test accuracy of one neural network architecture after restoring the checkpoint with the lowest validation loss, computed separately for the H1 and H6 conditions. Horizontal error bars indicate 95% Wilson confidence intervals for test accuracy. H1 accuracy is shown in blue and H6 accuracy is shown in red. Dashed vertical lines indicate the corresponding horizon-specific test accuracy of the first-game logistic regression baseline. Abbreviations: MLP, multilayer perceptron; RNN, recurrent neural network; LSTM, long short-term memory; GRU, gated recurrent unit; H1, Horizon 1; H6, Horizon 6.

## 2.7 Unexplained variance is not accounted for by history efects from game to game

Previous analyses captured dependencies between trials within the current game, but behavior may also depend on experiences from earlier games. These cross-game history efects provide another possible source of deterministic structure. Repeating the same game controls the choices and rewards within that game, but does not control for influences carried over from previous games [33]. To test for these efects, we used RNNs that propagated their hidden state from one game to the next within each participant. A learned parameter $\beta$ controlled how strongly information from the previous game influenced the current game.

Including cross-game history did not eliminate the predictability gap between H1 and H6 (Fig. 5). Across models, mean H1 accuracy was 0.784 (95% CI: [0.776, 0.792]), compared with 0.730 for H6 (95% CI: [0.724, 0.736]), corresponding to an H1–H6 diference of 5.4 percentage points. The models also learned relatively low $\beta$ values (LSTM $= 0 . 1 3 9 .$ , RNN = 0.183, GRU = 0.245, TinyGRU = 0.019), indicating that prediction relied more heavily on information from the current game than on information carried over from previous games. Thus, history from prior games’ instructed phases provided little additional explanation for the lower predictability of H6 choices.

![](images/dd9a39a769670fb0f2e188f77197a50c6be2df0e901f3b53bef9db4bb4408ea9.jpg)  
Fig. 5 Cross-game history analysis. Predictive accuracy was compared between H1 and H6 for recurrent models that incorporated cross-game history through the persistence parameter $\beta .$ Each point shows the test accuracy of one architecture after restoring the checkpoint with the lowest validation loss, computed separately for the H1 and H6 conditions. Horizontal error bars indicate 95% Wilson confidence intervals for test accuracy. H1 accuracy is shown in blue and H6 accuracy is shown in red. Dashed vertical lines indicate the corresponding horizon-specific test accuracy of the individual-fit logistic regression baseline. Abbreviations: RNN, recurrent neural network; LSTM, long short-term memory; GRU, gated recurrent unit; H1, Horizon 1; H6, Horizon 6.

## 2.8 Unexplained variance decreases over the course of horizon 6 games

If the increased unpredictability in H6 reflects the value of future exploration, it should decrease as the remaining horizon shortens within a game. We therefore examined how predictive accuracy changed across the six free-choice trials in H6. As the game progressed, choices became increasingly predictable (Fig. 6). Accuracy was lowest in the early H6 trials and generally increased across later trials. However, even by the final free-choice trial, H6 accuracy remained below the H1 benchmark. Overall, predictability increased as the remaining horizon shortened, consistent with greater behavioral variability when future exploration was more valuable.

![](images/8787a0e11be8946cf6d124b8b2345dd06899eee1f958d803980274c27c791215.jpg)  
Fig. 6 Predictive accuracy across H6 free-choice trials. The red line shows Seq2Seq model accuracy across the six free-choice trials in the H6 condition, from $C _ { 4 }$ to $C _ { 9 }$ . The dashed blue line indicates the H1 benchmark accuracy. Predictive accuracy increased across later H6 trials as the remaining horizon shortened, but final-trial H6 performance remained below the H1 benchmark.

## 3 Discussion

We investigated whether behavior previously described as random exploration could instead be explained by a deterministic structure that simpler cognitive models fail to capture. Neural network models captured more behavioral structure than the traditional logistic model, suggesting that some predictable information was missed by the simpler model. However, this additional model capacity did not completely eliminate the lower predictability of H6 choices. The reported H1–H6 accuracy gap narrowed in the subject-level and first-game comparisons, suggesting that the networks recovered some predictable structure in long-horizon choices. Nevertheless, H6 choices remained less predictable across the neural-network analyses, a residual diference that does not by itself establish irreducible stochasticity.

We tested several possible sources of deterministic structure. At first, the models used information from the current game, which allowed them to account for nonlinear relationships among the instructed choices, rewards, and task conditions, while recur rent and attention-based models could additionally capture sequential dependencies within the game. Nevertheless, H6 choices remained less predictable than H1 choices.

We then included subject identity, which allowed the models to capture individual behavioral patterns and tested whether the H1–H6 predictability gap was an artifact of averaging across individuals. The predictive accuracy improved, but the H1–H6 gap remained.

Moreover, we also tested whether the predictability diference depended on participants’ experience with the task. The H1–H6 predictability gap was already present when the analysis was restricted to each participant’s first game. We also tested whether experiences from previous games contributed to the gap by incorporating cross-game history into the models. However, this additional information still did not eliminate the predictability gap. Together, these findings suggest that the lower predictability of H6 choices did not arise solely from learning or strategy adaptation across repeated games. Within H6 games, choices became more predictable as the remaining horizon shortened, consistent with the increase in variability being linked to opportunities for future exploration. Together, these analyses show that the lower predictability of long-horizon choices persists after accounting for several sources of deterministic structure available in the Horizon Task.

A key limitation is that even highly expressive models can only capture deterministic structure that is present in the measured data. Our neural networks had access to choices, rewards, task conditions, subject identity, and, in some analyses, previous game history. However, other factors that may systematically influence choice were not measured. These could include where participants were looking, subtle movements or fidgeting, and internal states such as attention, arousal, or boredom. For example, spontaneous movements that are typically unrelated to the task can account for substantial trial-to-trial variation in neural activity [47]. If similar unmeasured factors influence exploratory choices, their efects would appear as unexplained variability in our models even if they were deterministic.

Future work could address this limitation by measuring a richer set of behavioral and physiological signals alongside choice. Eye tracking could test whether gaze patterns change with horizon, while video or motion tracking could measure fidgeting and other spontaneous movements. Physiological measures, such as pupil dilation and heart rate, could similarly capture changes in internal state. Incorporating these features would provide a stronger test of how much of the remaining variability in long-horizon choices is recoverable.

A second limitation is that we prioritized predictive performance over cognitive constraints and interpretability. To obtain the best possible fit to behavior, we did not examine how the networks represented the behavioral structure they learned. Several approaches could make these models more interpretable. Machine-learning models can be constrained to produce explicit psychological theories [46], while small recurrent networks can reveal compact latent strategies through analysis of their internal dynamics [48]. Related work has also used structured cognitive models to separate interpretable components of complex decision behavior [49]. Applying similar approaches here could reveal whether the neural networks recover known components of exploration, such as reward sensitivity, uncertainty, and information seeking, or identify additional behavioral states that are not represented in the traditional model.

Taken together, our findings suggest that the additional variability observed during long-horizon exploration cannot be explained by limited model expressiveness alone. More flexible models recovered substantially more predictable structure in behavior, yet long-horizon choices remained less predictable across the sources of information we examined. Within the behavioral measurements available in the Horizon Task, this residual variability is therefore consistent with a genuinely stochastic component of random exploration. Future work that incorporates richer behavioral and physiological measurements may help determine how much of this remaining variability can ultimately be explained.

## 4 Methods

## 4.1 The Horizon Task

In the Horizon Task, participants choose between two slot machines, or one-armed bandits, with the goal of maximizing their total reward [1]. When selected, each bandit produces a reward sampled from a Gaussian distribution. Rewards are truncated to fall between 1 and 100 points and rounded to the nearest integer. The two bandits have diferent mean rewards, such that one option is better on average. The mean of one bandit is set to either 40 or 60 points and randomly assigned to the left or right side, while the mean of the other bandit difers by 4, 8, 12, 20, or 30 points. Both reward distributions have a standard deviation of 8 points.

Each game contains either 5 or 10 trials (Fig 1). Five-trial games correspond to the short-horizon condition (H1), whereas 10-trial games correspond to the long-horizon condition (H6). Each game begins with four instructed trials. During these trials, participants are required to select a specified bandit and cannot choose the other option. The instructed trials control the amount of information available about each bandit before participants make their first free choice. In the unequal-information condition ([1 3]), one bandit is sampled once and the other three times. In the equalinformation condition ([2 2]), both bandits are sampled twice.

Following the instructed trials, participants make either one or six free choices, depending on the horizon condition. In H1, participants make a single free choice before the game ends. In H6, participants make six free choices. Because information acquired through exploration can influence subsequent choices only in H6, exploration has greater potential value in the long-horizon condition. Our primary analyses focus on the first free-choice trial, where the information available before choice is matched across horizon conditions.

## 4.2 Data set

We aggregated data from multiple studies using the Horizon Task, with a total of 1,392 unique participants (714 females, 650 males; age range = 10–74) completing 169,376 games (84,613 H1 and 84,763 H6). Table 3 provides a detailed breakdown of demographic information, where each age group is defined as: teens (≤17 years), young adults (18–34 years), midlife adults (35–59 years), and older adults (≥60 years). All participants were recruited in person. No participants were excluded.

Table 3 Participant demographics across datasets stratified by age group
<table><tr><td rowspan="2">Paper</td><td rowspan="2">Total  $N$ </td><td rowspan="2">Group</td><td colspan="3">Gender</td><td colspan="3"> $\mathrm { A g e }$ </td><td>Location</td></tr><tr><td></td><td>M</td><td>0</td><td>n</td><td>Mean</td><td>SD</td><td>City, State</td></tr><tr><td rowspan="4">Harms et al. [36]</td><td rowspan="4">123</td><td>Teens</td><td>25</td><td>29</td><td>0</td><td>54</td><td>11.31</td><td>1.10</td><td rowspan="4">Madison, WI</td></tr><tr><td>Young</td><td>14</td><td>43</td><td>0</td><td>57</td><td>19.33</td><td>2.63</td></tr><tr><td>Midlife</td><td>1</td><td>1</td><td>0</td><td>2</td><td>37.50</td><td>0.71</td></tr><tr><td>Unknown</td><td>0</td><td>0</td><td>10</td><td>10</td><td></td><td>一</td></tr><tr><td>Phelps et al. [37]</td><td rowspan="2">349</td><td>Young</td><td>84</td><td>65</td><td>2</td><td>151</td><td>21.43</td><td>2.87</td><td rowspan="2">Tucson, AZ</td></tr><tr><td>Mizell et al. [38]</td><td>OA</td><td>117</td><td>80</td><td>1</td><td>198</td><td>69.17</td><td>2.71</td></tr><tr><td rowspan="4">Sadeghiyeh et al. [39]</td><td rowspan="4">369</td><td>Teens</td><td>3</td><td>3</td><td>0</td><td>6</td><td>17.00</td><td>0.00</td><td rowspan="4">Tucson, AZ</td></tr><tr><td>Young</td><td>105</td><td>234</td><td>1</td><td>340</td><td>19.46</td><td>1.57</td></tr><tr><td>Midlife</td><td>1</td><td>0</td><td>0</td><td>1</td><td>50.00</td><td>一</td></tr><tr><td>Unknown</td><td>2</td><td>8</td><td>12</td><td>22</td><td></td><td>一</td></tr><tr><td rowspan="2">Smith et al. [40]</td><td rowspan="2">421</td><td>Young</td><td>286</td><td>106</td><td>0</td><td>392</td><td>22.70</td><td>4.29</td><td rowspan="2">Tucson, AZ</td></tr><tr><td>Midlife</td><td>18</td><td>11</td><td>0</td><td>29</td><td>37.34</td><td>1.93</td></tr><tr><td rowspan="2">Somerville et al. [41] Wilson et al. [1]</td><td rowspan="2">92</td><td>Teens</td><td>47</td><td>44</td><td>0</td><td>91</td><td>14.82</td><td>1.76</td><td rowspan="2">Cambridge, MA Princeton, NJ</td></tr><tr><td>Unknown</td><td>0</td><td>1</td><td>0</td><td>1</td><td></td><td>一</td></tr><tr><td rowspan="3">Waltz et al. [42]</td><td rowspan="3">38</td><td>Young</td><td>6</td><td>12</td><td>0</td><td>18</td><td>26.72</td><td>4.43</td><td rowspan="3">Baltimore, MD</td></tr><tr><td>Midlife</td><td>5</td><td>13</td><td>0</td><td>18</td><td>45.44</td><td>4.36</td></tr><tr><td>Unknown</td><td>0</td><td>0</td><td>2</td><td>2</td><td></td><td>一</td></tr><tr><td rowspan="6">Total</td><td rowspan="6">1392</td><td>Teens</td><td>75</td><td>76</td><td>0</td><td>151</td><td>13.66</td><td>2.35</td><td rowspan="6"></td></tr><tr><td>Young</td><td>495</td><td>460</td><td>3</td><td>958</td><td>21.22</td><td>3.64</td></tr><tr><td>Midlife</td><td>25</td><td>25</td><td>0</td><td>50</td><td>40.52</td><td>5.08</td></tr><tr><td>OA</td><td>117</td><td>80</td><td>1</td><td>198</td><td>69.17</td><td>2.71</td></tr><tr><td>Unknown</td><td>2</td><td>9</td><td>24</td><td>35</td><td></td><td>一</td></tr><tr><td>All</td><td>714</td><td>650</td><td>28</td><td>1392</td><td>一</td><td>一</td></tr></table>

## 4.3 Baseline Model

We used the logistic model from Wilson et al. [1] as our baseline model. Let $Q ( a )$ denote the value of action $^ { a , }$ where the optimal policy selects the action with the highest value. The value of choosing option a on the first free-choice trial is

$$
Q ( a ) = R _ { a } + \alpha I _ { a } + B _ { a } ,
$$

where $R _ { a }$ is the observed mean reward of option a from the instructed trials, $I _ { a }$ encodes the uncertainty condition, and $B _ { a }$ represents a spatial bias.

To compute the probability of choosing option a over option $b ,$ we calculated the diference in choice values and passed it through a logistic function:

$$
p ( { \mathrm { c h o o s e ~ a } } ) = \sigma \bigg ( \frac { Q ( a ) - Q ( b ) } { \eta _ { d } } \bigg ) = \frac { 1 } { 1 + \exp \Big ( - \frac { \Delta R + \alpha \Delta I + B } { \eta _ { d } } \Big ) }
$$

where $\sigma ( \cdot )$ denotes the logistic sigmoid function. $\eta _ { d }$ is the decision noise parameter associated with random exploration. $\Delta R = R _ { a } - R _ { b }$ represents the diference in observed mean rewards. $\Delta I = + 1$ when option a is more informative, $\Delta I = - 1$ when option b is more informative, and $\Delta I = 0$ in the equal uncertainty condition.

The logistic model contains three free parameters: the information bonus $\alpha ,$ spatial bias $B ,$ and decision noise $\eta _ { d }$ . We fitted the model diferently depending on the level of analysis. First, following the original Horizon Task analysis [1], the parameters were estimated separately for each participant and horizon condition. Subject-level choice parameter estimates were then averaged to obtain group-level choice curves and parameter summaries for H1 and H6. This analysis allowed us to confirm the established Horizon Task efects in the aggregated dataset. We compared parameter estimates between horizon conditions using paired t-tests. Each comparison included only participants with complete data for the relevant conditions, resulting in diferent sample sizes and degrees of freedom across tests. We additionally fitted one set of parameters to the pooled population and one set of parameters using only the first game played by each participant. In both cases, the model was fitted separately for the H1 and H6 conditions.

## 4.4 Neural Networks

We used neural networks to test whether they explained more behavioral variability than the baseline model. More importantly, we asked whether this additional model capacity reduced the predictability gap between H1 and H6. First, we trained models on the pooled population without subject identity. We then included subject identity to predict held-out games from participants represented during training. To test whether the gap was present before participants accumulated experience across repeated games, we repeated the analysis using only each participant’s first game. Next, we incorporated cross-game history to test whether previous games provided useful behavioral context. Finally, we examined how predictability changed across the free-choice trials in H6. The data splitting procedures for these analyses are described below.

## 4.4.1 Data Processing

Model inputs were organized into static and sequential features. Static features encoded the horizon and uncertainty conditions. Sequential features consisted of choice–reward pairs $( C _ { t } , R _ { t } )$ from the four instructed trials, where $t \in { 0 , 1 , 2 , 3 }$ indexes the trial within a game. Sequential inputs were normalized using statistics computed from the training set. The same transformation was then applied to the validation and test sets to prevent information leakage. The architectures difered in how they combined the static and sequential features, as described in the following section.

All neural network analyses used an $8 0 / 1 0 / 1 0$ train–validation–test split with a fixed random seed of 42. The split difered between the global and subject-level analyses. For the global models, data were split across the pooled dataset, and the training set was randomly shufled. These models treated participants as samples from a pooled population and evaluated prediction across the general dataset. For subjectlevel modeling, the split was applied within each participant. Specifically, 80% of each participant’s games were used for training, 10% for validation, and the remaining 10% for test evaluation. We encoded subject identity as an additional static feature using one-hot encoding. For the cross-game analysis, games were ordered chronologically within each participant.

## 4.4.2 Implementation

We trained the neural networks using the Adam optimizer [50] with a learning rate of 0.001 and a batch size of 32. Training batches were shufled at each epoch, and the models minimized cross-entropy loss for a maximum of 100 epochs. For the crossgame models, chronological game order was preserved within each participant; only independent participant sequences were shufled during training. At each epoch, the model updated its parameters on the training set and was then evaluated on the validation set.

We applied early stopping with a patience of five epochs. Whenever validation loss improved, we saved a copy of the model parameters from that epoch as the current best checkpoint. After early stopping, the checkpoint associated with the lowest validation loss was restored. This ensured that any decline during the patience window did not afect the final evaluation. The restored model was then evaluated once on the held-out test set. No additional regularization, such as dropout, was used because preliminary experiments showed reduced validation performance.

## 4.4.3 Architectures

We primarily evaluated recurrent neural networks (RNNs) [51, 52] and Transformerbased architectures [53]. We also implemented a three-layer multilayer perceptron (MLP) as a feedforward baseline to compare predictive performance across model classes. The architectures followed a hierarchy based on the behavioral information available to prediction. The MLP, base recurrent models, and attention-based models used information from the current game to capture nonlinear relationships and sequential dependencies among choices, rewards, and task conditions. We then extended the recurrent models to include history from previous games. By propagating the hidden state from one game to the next, these models could test whether influences from earlier games contributed to the unexplained variability. This addressed a limitation of the repeated-game approach, which controlled the immediate choice problem but did not account for cross-game history [33]. We tested several conventional hidden-unit configurations and selected architectures that produced stable performance. Across these configurations, predictive accuracy difered by less than 1%.

## MLP

The MLP architecture concatenated sequential and static inputs into a single vector and predicted the binary choice on the first free-choice trial (Fig. 7). The network consisted of two fully connected (FC) hidden layers. A final linear layer generated logits over the two choice options, followed by a softmax function to estimate choice probabilities.

## RNN based Models

To capture the sequential structure of choices and rewards within each game, we evaluated several recurrent architectures, including vanilla RNNs, Gated Recurrent Units (GRUs) [51], and Long Short-Term Memory (LSTM) networks [52]. In the primary configuration (Fig. 8a), the recurrent layer processed only the sequential inputs. The final hidden representation was concatenated with the static inputs and passed through an FC layer before the output layer. Inspired by Li et al. [48], we also included a Tiny-GRU variant with only two hidden units to establish a lower bound for representational capacity.

To test whether a participant’s previous games provided useful behavioral context, we developed a cross-game variant in which the recurrent hidden state was carried over across games within each subject (Fig. 8b). In this variant, static inputs were integrated directly into the recurrent layer at each timestep, and the hidden state from game $g - 1$ was used to initialize the state of game g. The importance of game history was modulated through a learnable parameter $\beta \in [ 0 , 1 ]$ initialized at 0.5:

$$
h _ { 0 } ^ { ( g ) } = \beta h _ { n } ^ { \left( g - 1 \right) } + \left( 1 - \beta \right) h _ { 0 } ^ { \mathrm { i n i t } } .
$$

This allowed the model to capture cross-game behavioral dependencies within each subject. Each game gave the four instructed choice–reward pairs, horizon, and uncertainty condition. The state carried between games therefore summarized prior games’ instructed phases.

## Transformer-Based Models

We also explored a Transformer encoder and a simplified self-attention variant (Fig. 9). Sequential inputs were projected into a 64-dimensional embedding space, after which sinusoidal positional encodings were added [53]. In both architectures, the representation from the final timestep was concatenated with static inputs and passed through an FC layer before classification.

## Seq2Seq GRU Model

To examine the evolution of predictive accuracy across the six free-choice trials in the H6 condition, we implemented a sequence-to-sequence (Seq2Seq) GRU framework [54]. The encoder processed the four instructed trials to produce a context vector. This vector was concatenated with static features and projected through a linear layer to initialize the decoder. The decoder then autoregressively generated predicted choices for the subsequent trials. We used teacher forcing during training and student forcing during evaluation to prevent ground-truth leakage (Fig. 10). This architecture allowed us to track predictive accuracy across the H6 free-choice trials. It also allowed us to test whether performance on the final H6 trial converged toward the H1 condition.

## 4.5 Evaluation

We used multiple metrics to evaluate model performance. First, we quantified model fit using negative log-likelihood (NLL) and McFadden’s pseudo- $R ^ { 2 }$ . The null model predicted a constant probability of choosing right based on the empirical choice frequency.

The logistic regression model provided the baseline NLL. For neural networks, NLL was the held-out test loss from the checkpoint selected by lowest validation loss. For each comparison, the logistic model was fitted by minimizing negative log-likelihood on the neural networks’ 80% training fold and evaluated on the identical held-out 10% test games; the intermediate 10% validation fold was unused by the logistic fit. To compare model fit between the neural networks and the baseline logistic model, we used NN models with encoded subject identities. Both analyses predict decisions at the subject level, rather than generalizing across the pooled population. McFadden’s pseudo-R<sup>2</sup> was then computed relative to the null model.

Second, we evaluated predictive accuracy for each model and examined model convergence across training epochs. For fair comparison with the NNs, the logistic regression baseline used out-of-sample accuracy. We then computed predictive accuracy separately for the H1 and H6 conditions.

Acknowledgments. R01AG061888 to RCW R56 AG061888 Arizona Alzheimer’s Disease Core Center pilot grant (P30 AG019610)

## References

[1] Wilson, R.C., Geana, A., White, J.M., Ludvig, E.A., Cohen, J.D.: Humans use directed and random exploration to solve the explore–exploit dilemma. Journal of Experimental Psychology: General 143, 2074–2081 (2014) https://doi.org/10. 1037/a0038199

[2] Feng, S.F., Wang, S., Zarnescu, S., Wilson, R.C.: The dynamics of explore–exploit decisions reveal a signal-to-noise mechanism for random exploration. Scientific Reports 11 (2021) https://doi.org/10.1038/s41598-021-82530-8

[3] Sutton, R.S., Barto, A.G.: Reinforcement learning: An introduction. IEEE Transactions on Neural Networks 9(5), 1054–1054 (1998) https://doi.org/10.1109/ TNN.1998.712192

[4] Kaelbling, L.P., Littman, M.L., Moore, A.W.: Reinforcement learning: A survey (1996) arXiv:cs/9605103 [cs.AI]

[5] Bellman, R.: A problem in the sequential design of experiments. Sankhy¯a: The Indian Journal of Statistics (1933-1960) 16(3/4), 221–229 (1956)

[6] Gittins, J.C.: Bandit processes and dynamic allocation indices. Journal of the Royal Statistical Society Series B: Statistical Methodology 41(2), 148–164 (1979)

[7] Zhang, S., Yu, A.J.: Forgetful bayes and myopic planning: Human learning and decision-making in a bandit setting. Advances in neural information processing systems 26 (2013)

[8] Averbeck, B.B.: Theory of choice in bandit, information sampling and foraging tasks. PLoS computational biology 11(3), 1004164 (2015)

[9] Wilson, R.C., Bonawitz, E., Costa, V.D., Ebitz, R.B.: Balancing exploration and exploitation with information and randomization. Current Opinion in Behavioral Sciences 38, 49–56 (2021) https://doi.org/10.1016/J.COBEHA.2020.10.001

[10] Krebs, J.R., Kacelnik, A., Taylor, P.: Test of optimal sampling by foraging great tits. Nature 275(5675), 27–31 (1978)

[11] Watkins, C., et al.: Learning from delayed rewards. (1989)

[12] Gershman, S.J., Tzovaras, B.G.: Dopaminergic genes are associated with both directed and random exploration. Neuropsychologia 120, 97–104 (2018)

[13] Schulz, E., Gershman, S.J.: The algorithmic architecture of exploration in the human brain. Current opinion in neurobiology 55, 7–14 (2019)

[14] Gershman, S.J.: Deconstructing the human algorithms for exploration. Cognition 173, 34–42 (2018)

[15] Frank, M.J., Doll, B.B., Oas-Terpstra, J., Moreno, F.: Prefrontal and striatal dopaminergic genes predict individual diferences in exploration and exploitation. Nature neuroscience 12(8), 1062–1068 (2009)

[16] Badre, D., Doll, B.B., Long, N.M., Frank, M.J.: Rostrolateral prefrontal cortex and individual diferences in uncertainty-driven exploration. Neuron 73(3), 595– 607 (2012)

[17] Cavanagh, J.F., Figueroa, C.M., Cohen, M.X., Frank, M.J.: Frontal theta reflects uncertainty and unexpectedness during exploration and exploitation. Cerebral cortex 22(11), 2575–2586 (2012)

[18] Speekenbrink, M., Konstantinidis, E.: Uncertainty and exploration in a restless bandit problem. Topics in cognitive science 7(2), 351–367 (2015)

[19] Daw, N.D., O’doherty, J.P., Dayan, P., Seymour, B., Dolan, R.J.: Cortical substrates for exploratory decisions in humans. Nature 441(7095), 876–879 (2006)

[20] Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P.: Optimization by simulated annealing. Science 220(4598), 671–680 (1983) https://doi.org/10.1126/science.220.4598.671

[21] Cern´y, V.: Thermodynamical approach to the traveling salesman problem: An <sup>ˇ</sup> eficient simulation algorithm. Journal of Optimization Theory and Applications 45(1), 41–51 (1985) https://doi.org/10.1007/BF00940812

[22] Geman, S., Geman, D.: Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-6(6), 721–741 (1984) https://doi.org/10.1109/ TPAMI.1984.4767596

[23] Metropolis, N., Rosenbluth, A.W., Rosenbluth, M.N., Teller, A.H., Teller, E.: Equation of state calculations by fast computing machines. The Journal of Chemical Physics 21(6), 1087–1092 (1953) https://doi.org/10.1063/1.1699114

[24] Hastings, W.K.: Monte carlo sampling methods using markov chains and their applications. Biometrika 57(1), 97–109 (1970) https://doi.org/10.1093/biomet/ 57.1.97

[25] Gelfand, A.E., Smith, A.F.M.: Sampling-based approaches to calculating marginal densities. Journal of the American Statistical Association 85(410), 398–409 (1990) https://doi.org/10.1080/01621459.1990.10476213

[26] Faisal, A.A., Selen, L.P., Wolpert, D.M.: Noise in the nervous system. Nature reviews neuroscience 9(4), 292–303 (2008)

[27] Ebitz, R.B., Albarran, E., Moore, T.: Exploration disrupts choice-predictive signals and alters dynamics in prefrontal cortex. Neuron 97(2), 450–4619 (2018) https://doi.org/10.1016/j.neuron.2017.12.007

[28] Tervo, D.G.R., Proskurin, M., Manakov, M., Kabra, M., Vollmer, A., Branson, K., Karpova, A.Y.: Behavioral variability through stochastic choice and its gating by anterior cingulate cortex. Cell 159, 21–32 (2014) https://doi.org/10.1016/j. cell.2014.08.037

[29] Budzillo, A., Dufy, A., Miller, K.E., Fairhall, A.L., Perkel, D.J.: Dopaminergic modulation of basal ganglia output through coupled excitation–inhibition. Proceedings of the National Academy of Sciences 114(22), 5713–5718 (2017) https://doi.org/10.1073/pnas.1611146114

[30] Findling, C., Skvortsova, V., Dromnelle, R., Palminteri, S., Wyart, V.: Computational noise in reward-guided learning drives behavioral variability in volatile environments. Nature Neuroscience 22(12), 2066–2077 (2019) https://doi.org/10. 1038/s41593-019-0518-9

[31] Osborne, L.C., Lisberger, S.G., Bialek, W.: A sensory source for motor variation. Nature 437(7057), 412–416 (2005)

[32] Beck, J.M., Ma, W.J., Pitkow, X., Latham, P.E., Pouget, A.: Not noisy, just wrong: The role of suboptimal inference in behavioral variability. Neuron 74, 30–39 (2012) https://doi.org/10.1016/j.neuron.2012.03.016

[33] Wang, S., Wilson, R.C.: Separating random and deterministic sources of computational noise in explore-exploit decisions. PLOS Computational Biology 22, 1014026 (2026) https://doi.org/10.1371/JOURNAL.PCBI.1014026

[34] Xiong, H.-D., Ji-An, L., Wilson, R.C., Mattar, M.G.: Human strategy adaptation in reinforcement learning resembles policy gradient ascent. bioRxiv (2025) https:

[35] Gershman, S.J.: Uncertainty and exploration. Decision 6(3), 277–286 (2019) https://doi.org/10.1037/dec0000101

[36] Harms, M.B., Xu, Y., Green, C.S., Woodard, K., Wilson, R., Pollak, S.D.: The structure and development of explore-exploit decision making. Cognitive Psychology 150 (2024) https://doi.org/10.1016/j.cogpsych.2024.101650

[37] Phelps, C.E., Frisvold, A.E., Sreeram, M., Antoniou, V., Oliveira, L.E., Tooke, S.R., Lu, Y.Z., Lemon, A.C., Fraire, J.U., Delval, A.G., Smith, B.K., Spangenberg, D.H., Mithelavage, M.N., Ngo, M.N., Keller, E.M., Issac, L.J., Harader, S.A., Mizell, J.-M., Wang, S., Keung, W., Sundman, M.H., Franchetti, M.-K., Chou, Y.-h., Alexander, G.E., Wilson, R.C.: The dynamics of explore-exploit decisions suggest a threshold mechanism for reduced random exploration in older adults. bioRxiv, 2025–0213638023 (2025) https://doi.org/10.1101/2025.02.13.638023

[38] Mizell, J.M., Wang, S., Frisvold, A., Alvarado, L., Farrell-Skupny, A., Keung, W., Phelps, C.E., Sundman, M.H., Franchetti, M.K., Chou, Y., Alexander, G.E., Wilson, R.C.: Diferential impacts of healthy cognitive aging on directed and random exploration. Psychology and aging 39, 88–101 (2024) https://doi.org/10. 1037/PAG0000791

[39] Sadeghiyeh, H., Wang, S., Alberhasky, M.R., Kyllo, H.M., Shenhav, A., Wilson, R.C.: Temporal discounting correlates with directed exploration but not with random exploration. Scientific Reports 10 (2020) https://doi.org/10.1038/ s41598-020-60576-4

[40] Smith, R., Taylor, S., Wilson, R.C., Chuning, A.E., Persich, M.R., Wang, S., Killgore, W.D.S.: Lower levels of directed exploration and reflective thinking are associated with greater anxiety and depression. Frontiers in Psychiatry 12 (2022) https://doi.org/10.3389/fpsyt.2021.782136

[41] Somerville, L.H., Sasse, S.F., Garrad, M.C., Drysdale, A.T., Akar, N.A., Insel, C., Wilson, R.C.: Charting the expansion of strategic exploratory behavior during adolescence. Journal of experimental psychology. General 146, 155–164 (2017) https://doi.org/10.1037/XGE0000250

[42] Waltz, J.A., Wilson, R.C., Albrecht, M.A., Frank, M.J., Gold, J.M.: Diferential efects of psychotic illness on directed and random exploration. Computational psychiatry (Cambridge, Mass.) 4, 18 (2020) https://doi.org/10.1162/CPSY A 00027

[43] LeCun, Y., Bengio, Y., Hinton, G.: Deep learning. nature 521(7553), 436–444 (2015)

[44] Hornik, K.: Approximation capabilities of multilayer feedforward networks.

[45] Cybenko, G.: Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems 2, 303–314 (1989) https://doi.org/10. 1007/BF02551274

[46] Peterson, J.C., Bourgin, D.D., Agrawal, M., Reichman, D., Grifiths, T.L.: Using large-scale experiments and machine learning to discover theories of human decision-making. Science 372(6547), 1209–1214 (2021) https://doi.org/10.1126/ science.abe2629

[47] Musall, S., Kaufman, M.T., Juavinett, A.L., Gluf, S., Churchland, A.K.: Singletrial neural dynamics are dominated by richly varied movements. Nature Neuroscience 22(10), 1677–1686 (2019) https://doi.org/10.1038/s41593-019-0502-4

[48] Li, J.-A., Benna, M.K., Mattar, M.G.: Discovering cognitive strategies with tiny recurrent neural networks. Nature (2025) https://doi.org/10.1038/ s41586-025-09142-4

[49] Wang, S., Wilson, R.C., Ebner, N.C., Lighthall, N.R.: Modeling value integration during decision making under uncertainty with the florida and georgia gambling task. Scientific Reports 15, 36826 (2025) https://doi.org/10.1038/ s41598-025-08333-3

[50] Kingma, D.P., Ba, J.L.: Adam: A method for stochastic optimization. 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings (2014)

[51] Cho, K., Merri¨enboer, B.V., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y.: Learning phrase representations using rnn encoder-decoder for statistical machine translation. EMNLP 2014 - 2014 Conference on Empirical Methods in Natural Language Processing, Proceedings of the Conference, 1724– 1734 (2014) https://doi.org/10.3115/v1/d14-1179

[52] Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural computation 9(8), 1735–1780 (1997)

[53] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

[54] Sutskever, I., Vinyals, O., Le, Q.V.: Sequence to sequence learning with neural networks. Advances in Neural Information Processing Systems 4, 3104–3112 (2014)

![](images/c273bfcffc62783da837f7c15e7c3912fabe651cee7efa8ac664dce8f958c9c3.jpg)  
Fig. 7 MLP Architecture. The model received a single concatenated vector of sequential and static inputs. Sequential inputs consisted of choice–reward pairs $( C _ { t } , R _ { t } )$ from the instructed trials (t ∈ {0, 1, 2, 3}), while static inputs encoded horizon length (H) and uncertainty condition (UC). The architecture consisted of two FC layers (32 and 16 units, respectively) with ReLU activations. The output layer applied a softmax function to generate choice probabilities for the first free-choice trial. $C _ { 4 }$ denotes the first free-choice response, as trial indexing began at 0.

![](images/00ca4a1dd7c20128cf23f6270991e26244ab6cf454b929306f36fd2fb75e7c7c.jpg)  
a. RNNs with separated inputs

![](images/b33f1f9391ed941989d82e7ce4172e004569f8ba963625d6183f35b1925c4896.jpg)  
b. RNNs with concatenated inputs

Fig. 8 RNN-based Architectures. The vanilla RNN and GRU used 64 hidden units, whereas the LSTM used 128 hidden units. The TinyGRU variant had two hidden units. (a) The recurrent layer received only sequential inputs. The final hidden state was concatenated with static inputs. (b) The recurrent layer received both sequential inputs and static inputs at each time step. (a, b) In both architectures, the resulting representation was passed through two FC layers, identical to those used in the MLP architecture, to generate a softmax prediction of the first free-choice response.

![](images/9ae8b4bcb7350778b79215bf213a9154e9625a149b7fdc811c7043d0ae14495b.jpg)  
a. Transformer encoder

![](images/ddeadbcc4a456120ac02447e9fae2653dd1b9c1998d0e0e839f50b533a9a6be8.jpg)  
b. Attention  
Fig. 9 Transformer-based Architectures. Sequential inputs were first projected into a 64-dimensional representation with sinusoidal positional encoding. (a) The Transformer encoder had four attention heads and a feedforward network with 128 hidden units using ReLU activation. (b) The attentiononly variant consisted of eight attention heads.

![](images/91d887f47e32648ffefefe98ba34480d93e512ba7646c12ab986f91ed03a056f.jpg)  
a. Teacher forcing

![](images/4b4a32c80fee87aaf6df4e103c7e0776f6a935b93b2d11530197792b7c770e58.jpg)  
b. Student forcing

Fig. 10 Seq2Seq Architecture. The model implemented an encoder–decoder recurrent framework. (a) In the teacher-forcing variant, the decoder received the ground-truth choice at each time step during training to predict the subsequent choice $( t \in \{ 4 , \ldots , 9 \} )$ . (b) In the student-forcing variant, the decoder used its own previous prediction as input to generate the next choice.