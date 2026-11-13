# OPTIMAL FEATURE SELECTION FOR FIREWALL LOG ANALYSISUSING MACHINE LEARNING AND HYBRID METAHEURISTICALGORITHMS

## A PREPRINT

Seungwoo Han   
Tokyo University of   
Agriculture and Technology   
Tokyo, Japan   
seungwoohan0108@gmail.com   
Jeuk Yu   
Sahmyook University   
Seoul, Korea   
jipsina71@gmail.com   
Gil Hong   
Sahmyook University   
Seoul, Korea   
wideroad95@nate.com   
Sangjun Lee   
Sahmyook University   
Seoul, Korea   
tjdtlf1234@naver.com   
Jewan Kim   
Sahmyook University   
Seoul, Korea   
wpdhks6@gmail.com   
Byeongok Cho   
Sahmyook University   
Seoul, Korea   
winjbo1021@gmail.com   
Jusung Jeon   
Sahmyook University   
Seoul, Korea   
gggg1371@naver.com

November 13, 2026

## ABSTRACT

Firewall log classification is important to monitor network traffic. Machine learning classifiers using network-related features have shown good results for this task, but high-dimensional features slow down classification. In this paper, we apply feature selection using bee swarm optimization with Q-learning-based local search (QBSO-FS) to the UCI Internet Firewall Data Set, together with random forest (RF), k-nearest neighbor (KNN), and Bernoulli naïve Bayes (Bernoulli NB) classifiers. The first version of this preprint selected and evaluated feature subsets on the same 5-fold cross-validation of the full data set, which contains 12.8% exact duplicates; the reported scores are therefore optimistically biased. In this extended version, we re-evaluate the method with a leakagefree protocol: the data are split into training, validation, and test sets, features are selected using the validation set only, and accuracy and macro-averaged scores are measured once on the untouched test set, for both the raw data and a deduplicated, preprocessed version. On a single stratified split, QBSO-FS selects 5–7 features. The selected subsets clearly improve Macro-F1 over all features for KNN on both data sets and for RF on the deduplicated data (by 8–19 percentage points). For RF on the raw data and for Bernoulli NB, the differences (0.2–1.5 percentage points) may not hold for a different split. RF with 7 selected features reaches 99.90% accuracy and 94.88% Macro-F1 on the deduplicated data. On the same split, this exceeds our re-implementations of the deep neural network configurations of previous studies and the feature subset selected by particle swarm optimization in one of them, while the run time saved on this low-dimensional data set is small. These results suggest that the method can support automatic firewall log analysis, while evaluation over multiple splits remains future work.

Keywords Bee swarm optimization · Reinforcement learning · Feature selection · Firewall log · Data leakage

## 1 Introduction

Firewalls ensure that a computer network operates properly and safely. As the protection of personal information [1] becomes increasingly important, so does the protection of the networks that carry it. Firewall logs are key evidence for identifying intrusions, including insider and outsider threats [2]. Beyond traditional rule-based analysis, machine learning and deep learning have been applied to log classification and intrusion prevention [3–8]. For a quick response, log analysis and intrusion detection should classify traffic efficiently with as few inputs as possible, yet the features used for classification are often chosen based on the researcher’s experience. Feature and parameter selection has therefore been studied using information gain, genetic algorithms, and reinforcement learning [9–12]. In this paper<sup>1</sup>, we search for an optimal feature subset using bee swarm optimization combined with reinforcement learning, namely the Q-learning-based bee swarm optimization for feature selection (QBSO-FS) [12].

In this extended version, we additionally revisit the evaluation protocol of our original experiment. In the original setting, both the fitness used to search for features and the reported performance were computed by cross-validation on the full data set, and the data set itself contains a substantial number of exactly duplicated records. Both factors can leak information from the evaluation data into model and feature selection and make the reported scores optimistic. The contributions of this extension are as follows:

• We identify 8,362 exactly duplicated records (12.8%) in the Internet Firewall Data Set and build a deduplicated, preprocessed version of the data.

• We repeat the QBSO-FS feature selection with the hyper-parameters of the original study, using a leakage-free train/validation/test protocol in which features are selected on a separate validation set and performance is reported only on an untouched test set.

• We compare random forest (RF), k-nearest neighbor (KNN), and Bernoulli naïve Bayes (Bernoulli NB) classifiers on both the raw and the deduplicated data, with and without feature selection.

• We re-evaluate the configurations of previous studies on the same data set under the same protocol, and compare run times for feature selection, training, and inference.

## 2 Materials and methods

## 2.1 Data set

We used the Internet Firewall Data Set [4, 15, 16] from the University of California, Irvine (UCI) Machine Learning Repository. It contains 65,532 records with 11 features and one of four labels: allow, deny, drop, or reset-both. Table 1 shows the class distribution, and Table 2 lists the features. NAT stands for network address translation; the NAT source and NAT destination ports are the port numbers after translation by the firewall.

Table 1: Class distribution of the raw data set
<table><tr><td>Label</td><td>Records</td></tr><tr><td>Allow</td><td>37,640</td></tr><tr><td>Deny</td><td>14,987</td></tr><tr><td>Drop</td><td>12,851</td></tr><tr><td>Reset-both</td><td>54</td></tr><tr><td>Total</td><td>65,532</td></tr></table>

## 2.2 Duplicate records and preprocessing

The raw data set contains 8,362 rows (12.8%) that are exact duplicates of another row, including the label. When such a data set is split into training and evaluation partitions at random, identical records fall on both sides of the split, and a classifier that memorizes the training set is rewarded for recall rather than generalization. Redundant records of this kind are a known source of biased evaluation in network intrusion data sets [17], and more generally a form of data leakage [18]. To quantify this effect we additionally built a processed version of the data set as follows:

Table 2: Features of the raw data set and of the processed data set (Section 2.2)
<table><tr><td>Feature</td><td>Description</td><td>In the processed data</td></tr><tr><td>Source port</td><td>Source port of the session</td><td>unchanged</td></tr><tr><td>Destination port</td><td>Destination port of the session</td><td>unchanged</td></tr><tr><td>NAT source port</td><td>Source port after NAT</td><td>unchanged</td></tr><tr><td>NAT destination port</td><td>Destination port after NAT</td><td>unchanged</td></tr><tr><td>Bytes</td><td>Total number of bytes</td><td>log(1 + x)</td></tr><tr><td>Bytes sent</td><td>Number of bytes sent</td><td>log(1 + x)</td></tr><tr><td>Bytes received</td><td>Number of bytes received</td><td>log(1 + x)</td></tr><tr><td>Packets</td><td>Total number of packets</td><td>log(1 + x)</td></tr><tr><td>Packets sent</td><td>Number of packets sent (pkts_sent)</td><td>log(1 + x)</td></tr><tr><td>Packets received</td><td>Number of packets received (pkts_received)</td><td>log(1 + x)</td></tr><tr><td>Elapsed time</td><td>Duration of the session in seconds</td><td>log(1 + x)</td></tr><tr><td colspan="3">Derived features (processed data only)</td></tr><tr><td>NAT-source-is-zero</td><td>1 if the NAT source port is 0, otherwise 0</td><td>binary</td></tr><tr><td>NAT-destination-is-zero</td><td>1 if the NAT destination port is 0, otherwise 0</td><td>binary</td></tr><tr><td>Source port range</td><td>Range of the source port (well-known, registered, dynamic)</td><td>categorical (0, 1, 2)</td></tr><tr><td>Destination port range</td><td>Range of the destination port (as above)</td><td>categorical (0, 1, 2)</td></tr><tr><td>Bytes per packet</td><td>log(1 + bytes/packets)</td><td>continuous</td></tr></table>

1. Exact duplicate rows were removed, leaving 57,170 records (Table 3).

2. The heavy-tailed volume and time features (bytes, bytes sent, bytes received, packets, packets sent, packets received, and elapsed time) were transformed with $\log ( 1 + x )$

3. Five derived features were added (Table 2): the NAT-source-is-zero and NAT-destination-is-zero flags, the source and destination port ranges defined by the Internet Assigned Numbers Authority (IANA) [19] (wellknown 0–1023, registered 1024–49151, dynamic 49152–65535), and the bytes per packet.

All transformations are applied row by row and do not use statistics estimated from the data, so they do not introduce leakage across partitions. The derived features were defined a priori from domain knowledge of ports and NAT, before any of the experiments reported here, and were not adjusted based on the results. The processed data set therefore has 16 features. Finally, 9 distinct feature vectors occur with more than one label (46 rows in the raw data and 18 rows after deduplication, mostly deny versus drop). These rows are not duplicates and were kept, since removing them would require deciding which label is correct.

Table 3: Class distribution before and after deduplication
<table><tr><td>Label</td><td>Raw</td><td>Deduplicated</td><td>Removed</td></tr><tr><td>Allow</td><td>37,640</td><td>37,439</td><td>201</td></tr><tr><td>Deny</td><td>14,987</td><td>8,042</td><td>6,945</td></tr><tr><td>Drop</td><td>12,851</td><td>11,635</td><td>1,216</td></tr><tr><td>Reset-both</td><td>54</td><td>54</td><td>0</td></tr><tr><td>Total</td><td>65,532</td><td>57,170</td><td>8,362</td></tr></table>

## 2.3 Feature selection with QBSO-FS

## 2.3.1 Bee swarm optimization algorithm

The bee swarm optimization algorithm (BSO) [20] is inspired by the foraging behavior of bees. Each bee explores the search space from its own starting point, and the bees cooperate by sharing the solutions they find over a number of iterations. BSO was later adapted to feature selection as BSO-FS [21], and extended with Q-learning-based local search as QBSO-FS [12], which we use in this work. A solution is a binary vector $\mathbf { s } \in \{ 0 , 1 \} ^ { n }$ that marks the selected features, and its fitness $f ( \mathbf { s } ) \in [ 0 , 1 ]$ is the accuracy of the classifier trained on those features (Section 2.5). The search proceeds as follows; the hyper-parameters named in the steps are listed in Table 4.

1. Initialization. A reference solution $\mathbf { s } _ { \mathrm { r e f } }$ is drawn at random, with each feature selected with probability 0.2.

2. Search area. In each iteration, $\mathbf { S } _ { \mathrm { r e f } }$ is added to a tabu list $\tau _ { \ast }$ , and B starting solutions, one per bee, are generated from it, where B is the Number of bees. The Flip parameter $F$ controls how far these solutions are from $\mathbf { s } _ { \mathrm { r e f } }$ Bee $b \left( b = 0 , 1 , \ldots , F - 1 \right)$ flips every F-th feature starting at position $b ,$

$$
\mathbf { s } ^ { ( b ) } = \mathbf { s } _ { \mathrm { r e f } } \oplus \sum _ { \substack { m : m F + b < n } } \mathbf { e } _ { m F + b } ,\tag{1}
$$

where ⊕ is the element-wise exclusive OR (XOR) and $\mathbf { e } _ { j }$ is the j-th unit vector, so that different bees start in different, non-overlapping directions. If $B > F$ , the next $F$ bees each flip a contiguous block of about $n / F$ features, and any remaining bees flip one randomly chosen feature. With the settings of Table 4 $( B = 3$ $\dot { F } = 5 )$ , all three bees are generated by Eq. (1).

3. Local search. Each bee improves its starting solution by a local search. In QBSO-FS this is the Q-learning procedure of Section 2.3.2, whose number of steps is governed by Local iteration.

4. Communication. The bees report their solutions and fitness values to a shared “dance table”, from which the next reference solution is chosen. If the best bee improves on the best solution found so far, or on the last solution in $\tau _ { \ast }$ it becomes the new $\mathbf { s } _ { \mathrm { r e f } }$ and the number of remaining chances is reset to Max chance. Otherwise, one chance is consumed. While chances remain, the best bee is still chosen (intensification). When they are exhausted, the bee farthest in Hamming distance from the solutions in $\tau$ is chosen instead (diversification), which lets the swarm escape a local optimum. Ties in fitness are broken in favor of the solution farther from T and with fewer features.

5. Termination. Steps 2–4 are repeated for Max iteration iterations, and the best solution found is returned.

The total number of local searches is therefore $B \times$ Max iteration, and the search budget is controlled mainly by these two parameters together with Local iteration.

## 2.3.2 Q-learning local search

Reinforcement learning (RL) trains an agent to choose actions that maximize its cumulative reward. QBSO-FS uses Q-learning [22] as the local search of each bee. The state is the current feature subset, an action flips one feature, and the reward favors higher accuracy and, at equal accuracy, fewer features. To reduce the number of candidate actions, only features on which the current state and the best solution differ are considered. We now state the procedure formally.

Formulation. Let n be the number of features. A state is a binary vector

$$
\mathbf { s } = ( s _ { 1 } , \ldots , s _ { n } ) \in \{ 0 , 1 \} ^ { n } , \qquad | \mathbf { s } | = \sum _ { j = 1 } ^ { n } s _ { j } ,\tag{2}
$$

where $s _ { j } = 1$ means that feature $j$ is selected. An action $a \in \{ 1 , \ldots , n \}$ flips one feature:

$$
\delta ( \mathbf { s } , a ) = { \left\{ \begin{array} { l l } { \mathbf { s } \oplus \mathbf { e } _ { a } } & { { \mathrm { i f ~ } } | \mathbf { s } \oplus \mathbf { e } _ { a } | > 0 , } \\ { \mathbf { s } } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{3}
$$

so the empty subset is never visited.

Action space reduction. Let $\mathbf { s } ^ { * }$ be the best solution found so far by the swarm. Only features on which the current state and the best solution disagree, as determined by their XOR, are considered:

$$
{ \cal D } ( { \bf s } ) = \{ j \mid s _ { j } \oplus s _ { j } ^ { * } = 1 \} .\tag{4}
$$

The candidate action set $\mathcal { A } ( \mathbf { s } )$ is a uniformly random subset of $D ( \mathbf { s } )$ of size $F \mathrm { i f } \ | D ( \mathbf { s } ) | > F$ , and a single random element of $D ( \mathbf { s } )$ otherwise. If $D ( \mathbf { s } )$ is empty, the previous action set is kept.

Policy. Actions are chosen ϵ-greedily with respect to the Q-value plus the fitness of the resulting state:

$$
a _ { t } = \left\{ \begin{array} { l l } { \arg \displaystyle \operatorname* { m a x } _ { a \in \mathcal { A } ( \mathbf { s } _ { t } ) } \left[ Q ( \mathbf { s } _ { t } , a ) + f \big ( \delta ( \mathbf { s } _ { t } , a ) \big ) \right] } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - \epsilon , } \\ { \mathrm { u n i f o r m ~ r a n d o m ~ } a \in \mathcal { A } ( \mathbf { s } _ { t } ) } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \epsilon , } \end{array} \right.\tag{5}
$$

with ties broken at random. After each step, the exploration rate decays linearly, $\epsilon \gets \operatorname* { m a x } ( 0 , \epsilon - 1 0 ^ { - 4 } )$ .

Reward. With $\mathbf { s } _ { t + 1 } = \delta ( \mathbf { s } _ { t } , a _ { t } )$ , the reward is

$$
r _ { t } = \left\{ \begin{array} { l l } { f ( \mathbf { s } _ { t + 1 } ) } & { \mathrm { i f ~ } f ( \mathbf { s } _ { t + 1 } ) > f ( \mathbf { s } _ { t } ) , } \\ { f ( \mathbf { s } _ { t + 1 } ) - f ( \mathbf { s } _ { t } ) } & { \mathrm { i f ~ } f ( \mathbf { s } _ { t + 1 } ) < f ( \mathbf { s } _ { t } ) , } \\ { \frac { 1 } { 4 } f ( \mathbf { s } _ { t + 1 } ) } & { \mathrm { i f ~ } f ( \mathbf { s } _ { t + 1 } ) = f ( \mathbf { s } _ { t } ) \mathrm { ~ a n d ~ } | \mathbf { s } _ { t + 1 } | < | \mathbf { s } _ { t } | , } \\ { - \frac { 1 } { 4 } f ( \mathbf { s } _ { t + 1 } ) } & { \mathrm { i f ~ } f ( \mathbf { s } _ { t + 1 } ) = f ( \mathbf { s } _ { t } ) \mathrm { ~ a n d ~ } | \mathbf { s } _ { t + 1 } | \geq | \mathbf { s } _ { t } | . } \end{array} \right.\tag{6}
$$

The feature-count terms therefore act as a tie-breaker that favors smaller subsets when accuracy does not change.

Q-value update. The Q-table is updated with the standard Q-learning rule [22], using the learning rate α and the discount factor γ of Table 4:

$$
Q ( \mathbf { s } _ { t } , a _ { t } ) \gets ( 1 - \alpha ) Q ( \mathbf { s } _ { t } , a _ { t } ) + \alpha \Big [ r _ { t } + \gamma Q ( \mathbf { s } _ { t + 1 } , a ^ { \prime } ) \Big ] , \qquad a ^ { \prime } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \big [ Q ( \mathbf { s } _ { t + 1 } , a ) + f \big ( \delta ( \mathbf { s } _ { t + 1 } , a ) \big ) \big ] ,\tag{7}
$$

where $a ^ { \prime }$ is the greedy action in the next state under the same criterion as the policy. All Q-values are initialized to zero, and the Q-table is shared by all bees, so the experience gathered by one bee guides the local search of the others.

Local search schedule. In BSO iteration i, each bee performs

$$
K _ { i } = \operatorname* { m i n } \bigl ( \lfloor i / L \rfloor + 1 , L \bigr )\tag{8}
$$

Q-learning steps, where L is the Local iteration parameter. With $L = 2$ and three iterations, this gives $K = 1 , 2 , 2$ , so the search explores in the first iteration and exploits more in later ones. The fitness of a bee is f(s) of its final state.

## 2.3.3 Hyper-parameters

Table 4 lists the hyper-parameters of QBSO-FS, which are the same in the original and in the leakage-free experiment.

Table 4: Hyper-parameters of QBSO-FS
<table><tr><td>Name</td><td>Value</td><td>Description</td></tr><tr><td>Flip (F)</td><td>5</td><td>Spacing of the flipped features in the search area (Eq. (1)) and size of the candidate action set</td></tr><tr><td>Max chance</td><td>3</td><td>Iterations without improvement allowed before diversification</td></tr><tr><td>Number of bees (B)</td><td>3</td><td>Number of bees, i.e. starting solutions per iteration</td></tr><tr><td>Max iteration</td><td>3</td><td>Number of BSO iterations</td></tr><tr><td>Local iteration (L)</td><td>2</td><td>Upper bound on the Q-learning steps per bee and iteration</td></tr><tr><td>α</td><td>0.8</td><td>Learning rate of Q-learning</td></tr><tr><td>γ</td><td>0.3</td><td>Discount factor of Q-learning</td></tr><tr><td>€</td><td>0.05</td><td>Initial probability of a random action</td></tr></table>

## 2.3.4 Why BSO for wrapper feature selection

Wrapper methods evaluate each candidate subset by training the classifier itself [23], so the search strategy determines both the quality of the result and the number of costly evaluations. BSO-FS has also been applied in network security, for example to select features for detecting external sinkhole attacks in large-scale wireless sensor networks [24]. Compared with common alternatives, BSO, and QBSO-FS in particular, has the following properties.

• Escaping local optima. Greedy sequential methods such as forward or backward selection never revisit a decision and can be trapped in a local optimum (the nesting effect) [25]. BSO keeps a tabu list [26] of past reference solutions and switches from intensification to diversification once Max chance iterations pass without improvement. It therefore explicitly moves away from regions that have already been explored.

• Native binary search. A feature subset is a bit vector, and BSO operates on it directly through bit flips (Eq. (1)). Particle swarm optimization (PSO) [27] was designed for continuous spaces and needs a transfer function that maps velocities to bit probabilities in its binary version [28]. Genetic algorithms (GA) [29] also work on bit strings, but their crossover and mutation operators are random.

• Systematic, diverse neighborhoods. The Flip-based search area assigns each bee a different, non-overlapping set of features to change. A few bees thus cover distinct directions around the reference solution, and the population can be very small (three bees in this work). With a population-based method such as GA or PSO, evaluating a whole population in every generation is the dominant cost of a wrapper.

• Learningfrom experience. The local search is guided by a Q-table shared by all bees $( \operatorname { E q . } ( 7 ) ) .$ , so flips that improved the fitness before are preferred later. The XOR-based action space restricts the moves to features on which the current state and the best solution so far disagree.

• Preferencefor compact subsets. At equal fitness, the reward $\left( \mathrm { E q . } \left( 6 \right) \right)$ and the tie-breaking rule of the dance table both favor solutions with fewer features. Wrappers that maximize accuracy alone do not include this preference.

These are structural advantages and do not guarantee better results on every problem [30]. In this study we did not run GA or PSO under the same evaluation budget. The empirical evidence here is limited to the comparison with all features and with the subset reported for tuned particle swarm optimization (TPSO) [31] (Section 3.3). In our experiments, QBSO-FS needed 270–551 subset evaluations (Table 13), a small fraction of the $2 ^ { 1 1 } - 1 = 2 { , } 0 4 7$ and $2 ^ { 1 6 } - \dot { 1 } = 6 5 { , } 5 3 5$ non-empty subsets of the raw and processed data.

## 2.4 Classifiers

We compare three classifiers: RF [32], KNN [33], and Bernoulli NB [34]. Bernoulli NB binarizes every input at a threshold of 0, so each feature is reduced to a zero/non-zero indicator. Default hyper-parameters are used for all classifiers.

## 2.5 Evaluation protocol

Evaluation metrics. Let h be a trained classifier and S a set of N labeled records, with true labels $y _ { i }$ and predicted labels ${ \hat { y } } _ { i } .$ , and let $\mathcal { C }$ be the set of C = 4 classes. For each class $c \in \mathcal { C } , \mathrm { T P } _ { c } , \mathrm { F P } _ { c } ,$ and $\mathrm { F N } _ { c }$ denote the numbers of true positives, false positives, and false negatives when c is treated as the positive class. Accuracy (Acc.) is the fraction of correctly classified records,

$$
\operatorname { A c c } ( h , \mathcal { S } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \left[ \hat { y } _ { i } = y _ { i } \right] = \frac { 1 } { N } \sum _ { c \in \mathcal { C } } \operatorname { T P } _ { c } ,\tag{9}
$$

and the per-class precision (P), recall (R), and F1 score are

$$
P _ { c } = { \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } } } , \qquad R _ { c } = { \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F N } _ { c } } } , \qquad \mathrm { F 1 } _ { c } = { \frac { 2 \ P _ { c } R _ { c } } { P _ { c } + R _ { c } } } .\tag{10}
$$

The macro-averaged precision (Macro-P), recall (Macro-R), and F1 score (Macro-F1) give each class the same weight regardless of its size:

$$
\mathbf { M a c r o - P } = \frac { 1 } { C } \sum _ { c \in \mathcal { C } } P _ { c } , \qquad \mathbf { M a c r o - R } = \frac { 1 } { C } \sum _ { c \in \mathcal { C } } R _ { c } , \qquad \mathbf { M a c r o - F } 1 = \frac { 1 } { C } \sum _ { c \in \mathcal { C } } \mathbf { F } 1 _ { c } .\tag{11}
$$

Macro-F1 is thus the mean of the per-class F1 scores, not the F1 score of Macro-P and Macro-R. If a class is never predicted, its precision (and hence its F1 score) is set to zero. For previous studies whose averaging is not stated (Table 11), P, R, and F1 refer to the values reported by the authors. Because the classes are highly imbalanced, Acc. is dominated by the allow class, whereas the macro-averaged scores are sensitive to the 54-record reset-both class. Differences between scores are given in percentage points (pp).

Original protocol. In our original experiment, the fitness of a candidate feature subset was the mean accuracy of stratified 5-fold cross-validation (CV) over the entire data set, and the performance reported in Section 3.1 was obtained from the same folds. Because the subset is chosen to maximize exactly the score that is then reported, this estimate is optimistically biased (selection bias [35]), and the duplicated records described in Section 2.2 further inflate it. Formally, with D the full data set split into folds $\mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { 5 }$

$$
f _ { \mathrm { o r i g } } ( \mathbf { s } ) = \frac { 1 } { 5 } \sum _ { k = 1 } ^ { 5 } \operatorname { A c c } \big ( h _ { \mathbf { s } } ^ { \mathcal { ( D \setminus D _ { k } ) } } , \mathcal { D } _ { k } \big ) ,\tag{12}
$$

where $h _ { \mathbf { s } } ^ { ( { \cal S } ) }$ denotes the classifier trained on set $s$ using only the features selected by s.

Leakage-free protocol. For the additional experiment we use a held-out design. Each data set is split, stratified by label, into a training set (60%), a validation set (20%), and a test set (20%). During feature selection, the fitness of a subset is the accuracy on the validation set of a classifier fitted on the training set,

$$
f _ { \mathrm { v a l } } ( \mathbf { s } ) = \mathrm { A c c } \big ( h _ { \mathbf { s } } ^ { ( \mathcal { D } _ { \mathrm { t r a i n } } ) } , \mathcal { D } _ { \mathrm { v a l } } \big ) ,\tag{13}
$$

and the test set is not touched during the search. After the search, the classifier is refitted on the union of the training and validation sets using the selected features and evaluated once on the test set, i.e. we report $\mathrm { A c c } \big ( h _ { \hat { \mathbf { s } } } ^ { ( { D _ { \mathrm { t r a i n } } } \cup { D _ { \mathrm { v a l } } } ) } , { D _ { \mathrm { t e s t } } } \big )$ and the macro-averaged scores for the selected subset ˆs. The same refit-and-test procedure with all features serves as the baseline. The protocol is applied to both the raw data (duplicates kept) and the processed, deduplicated data.

## 2.6 Implementation

All classifiers in the leakage-free experiment, including the multilayer perceptrons (MLPs) in Section 3.3, were implemented with scikit-learn [36] (version 1.9.1); the original experiment (Section 3.1) used version 0.24.2. For the leakage-free experiment, we used the open-source QBSO-FS implementation [37] of Sadeg et al. [12] with Q-learning local search, the hyper-parameters of Table 4, and the reward of Eq. (6).

## 3 Results

## 3.1 Original experiment

This subsection reproduces the results reported in the first version of this preprint [13]. They were obtained with the original protocol (Section 2.5): feature subsets were selected from the 11 features of the raw data set and scored with the same 5-fold CV on the full, non-deduplicated data, and Table 6 reports the best score among the candidate subsets. These values are therefore optimistically biased. We keep them for reference only; the leakage-free estimates in Section 3.2 supersede them.

The feature combinations selected from the 11 raw features are shown in Table 5.

Table 5: Feature combinations selected in the original experiment (raw data set, 11 features)
<table><tr><td>Classifier</td><td>Selected feature combination</td></tr><tr><td>RF</td><td>Destination port, NAT source port, NAT destination port, bytes, bytes received, packets, elapsed time, packets sent, packets received</td></tr><tr><td>KNN</td><td>Destination port, NAT destination port, bytes, bytes sent, packets, elapsed time</td></tr><tr><td>Bernoulli NB</td><td>Source port, bytes, bytes sent, elapsed time, packets received</td></tr></table>

Table 6 shows, for each classifier, the best accuracy among the candidate subsets.

Table 6: Best accuracy (%) among candidate subsets (original protocol; optimistically biased)
<table><tr><td>Classifier</td><td>Best Acc.</td></tr><tr><td>RF</td><td>99.87</td></tr><tr><td>KNN</td><td>99.78</td></tr><tr><td>Bernoulli NB</td><td>80.22</td></tr></table>

We also compared RF with the selected features and with all features in terms of Acc., Macro-P, Macro-R, and Macro-F1, each averaged over the five folds. The results are shown in Table 7.

Table 7: Comparing all and selected features using RF (%) (original protocol, 5-fold CV on the full raw data set with 11 features; optimistically biased)
<table><tr><td>Feature selected</td><td>Acc.</td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td></tr><tr><td>All</td><td>99.80</td><td>93.06</td><td>81.71</td><td>84.22</td></tr><tr><td>Selected</td><td>99.87</td><td>96.62</td><td>88.73</td><td>91.27</td></tr></table>

## 3.2 Leakage-free experiment

Table 8 summarizes the partitions used in the leakage-free protocol (Section 2.5). For the raw data, 2,365 of the 13,107 test records (18.0%) have an identical copy (features and label) in the training or validation set. After deduplication, no test record appears in the training or validation set.

The feature subsets selected on the validation set are listed in Table 9. On the raw data, the subsets chosen for KNN and Bernoulli NB are very close to those of the original experiment (Table 5): the Bernoulli NB subset is identical, and the KNN subset differs only by replacing packets with bytes received. The RF subset is smaller (5 instead of 9 features).

Table 8: Train / validation / test partitions (stratified 60/20/20)
<table><tr><td>Data set</td><td>Features</td><td>Train</td><td>Validation</td><td>Test</td><td>Test rows duplicated in train+val</td></tr><tr><td>Raw (duplicates kept)</td><td>11</td><td>39,318</td><td>13,107</td><td>13,107</td><td>2,365 (18.0%)</td></tr><tr><td>Processed (deduplicated)</td><td>16</td><td>34,302</td><td>11,434</td><td>11,434</td><td>0 (0.0%)</td></tr></table>

Table 9: Feature subsets selected on the validation set (leakage-free experiment)
<table><tr><td>Data set</td><td>Classifier</td><td>#</td><td>Selected features</td></tr><tr><td rowspan="3">Raw</td><td>RF</td><td>5</td><td>Destination port, NAT source port, bytes, bytes received, elapsed time</td></tr><tr><td>KNN</td><td>6</td><td>Destination port, NAT destination port, bytes, bytes sent, bytes received, elapsed time</td></tr><tr><td>Bernoulli NB</td><td>5</td><td>Source port, bytes, bytes sent, elapsed time, packets received</td></tr><tr><td rowspan="3">Processed</td><td>RF</td><td>7</td><td>Destination port, bytes received, packets, elapsed time, packets sent, NAT-destination-is-zero, destination port range</td></tr><tr><td>KNN</td><td>7</td><td>Bytes sent, packets, elapsed time, NAT-source-is-zero, NAT- destination-is-zero, destination port range, bytes per packet</td></tr><tr><td>Bernoulli NB</td><td>7</td><td>Source port, bytes, elapsed time, packets received, source port range, destination port range, bytes per packet</td></tr></table>

On the processed data, every classifier selects 7 of the 16 features, and each subset contains at least one of the derived features.

Table 10 reports the test-set performance of each classifier with all features and with the selected subset. “Val. acc.” is the validation accuracy of the classifier trained on the training set; for the selected subset, it is the fitness maximized by QBSO-FS.

Table 10: Test-set performance (%) of the leakage-free experiment. Classifiers are refitted on train+validation and evaluated once on the held-out test set.
<table><tr><td>Data set</td><td>Classifier</td><td>Features</td><td>#</td><td>Val. acc.</td><td>Accuracy</td><td>Macro-P</td><td>Macro-R</td><td>Macro-F1</td></tr><tr><td rowspan="5">Raw</td><td>RF</td><td>All</td><td>11</td><td>99.82</td><td>99.78</td><td>99.75</td><td>83.91</td><td>88.12</td></tr><tr><td rowspan="2"></td><td>Selected</td><td>5</td><td>99.86</td><td>99.84</td><td>95.65</td><td>86.26</td><td>89.57</td></tr><tr><td>All</td><td>11</td><td>99.34</td><td>99.37</td><td>74.43</td><td>74.60</td><td>74.51</td></tr><tr><td rowspan="2">KNN Bernoulli NB</td><td>Selected</td><td>6</td><td>99.73</td><td>99.76</td><td>96.66</td><td>90.72</td><td>93.22</td></tr><tr><td>All</td><td>11</td><td>79.90</td><td>79.80</td><td>38.28</td><td>49.77</td><td>42.25</td></tr><tr><td rowspan="6">Processed</td><td></td><td>Selected</td><td>5</td><td>80.27</td><td>80.19</td><td>38.39</td><td>49.94</td><td>42.42</td></tr><tr><td rowspan="2">RF</td><td>All</td><td>16</td><td>99.75</td><td>99.79</td><td>91.35</td><td>79.35</td><td>81.89</td></tr><tr><td>Selected</td><td>7</td><td>99.87</td><td>99.90</td><td>97.11</td><td>93.06</td><td>94.88</td></tr><tr><td rowspan="2">KNN</td><td>All</td><td>16</td><td>99.19</td><td>99.20</td><td>73.97</td><td>74.33</td><td>74.15</td></tr><tr><td>Selected</td><td>7</td><td>99.85</td><td>99.83</td><td>99.74</td><td>79.40</td><td>82.49</td></tr><tr><td rowspan="2">Bernoulli NB</td><td>All</td><td>16</td><td>98.27</td><td>98.15</td><td>73.23</td><td>72.94</td><td>73.04</td></tr><tr><td>Selected</td><td>7</td><td>99.03</td><td>98.91</td><td>73.71</td><td>73.23</td><td>73.43</td></tr></table>

In all six combinations of data set and classifier, the selected subset uses fewer than half of the features, and on this split its test accuracy and Macro-F1 are higher than those of the full feature set. The size of the difference, however, varies widely. It is large for KNN (Macro-F1 +18.71 pp on the raw data and +8.34 pp on the processed data) and for RF on the processed data (+12.99 pp). It is small for RF on the raw data (+1.45 pp, with lower Macro-P than all features) and for Bernoulli NB (+0.17 pp on the raw data and +0.39 pp on the processed data). Differences of this size could be reversed by a different split, so for these cases we can only conclude that the selected subset performs about as well as all features with fewer than half of them. The test accuracy of every selected subset is within 0.12 pp of its validation fitness, so there is no sign that the search overfitted the validation set. The best result is obtained by RF on the deduplicated data with 7 features (99.90% accuracy, 94.88% Macro-F1). Because all numbers come from a single split, differences of a few tenths of a pp in accuracy, and of a few pp in the macro-averaged scores (Section 4), should not be over-interpreted.

## 3.3 Comparison with previous studies

Table 11 summarizes studies that classify the same UCI Internet Firewall Data Set, as reported by their authors. Ertam and Kaya [4], who collected the data set, classified it with multiclass support vector machines (SVMs) using linear, polynomial, radial basis function (RBF), and sigmoid kernels. Lillmond and Suddul [38] trained a deep neural network (DNN), implemented as an MLP, on all 11 features. Verma and Sahu [31] selected 5 features with a TPSO wrapper that uses a Naïve Bayes classifier on the raw feature values as the objective, and then trained a DNN on them. Talaeepoor et al. [39] compared a decision tree, KNN, and neural networks under several k-fold settings and found the decision tree to perform best. The reported figures are not directly comparable. The studies differ in the number of classes (Lillmond and Suddul merge reset-both into deny and undersample the majority classes), in how features are selected and evaluated (Lillmond and Suddul [38], Verma and Sahu [31], and Talaeepoor et al. [39], like our original experiment, use CV on the full, non-deduplicated data, and Ertam and Kaya [4] do not report their split), and in how precision, recall, and F1 are averaged.

Table 11: Previous studies on the UCI Internet Firewall Data Set, as reported by the authors (%).
<table><tr><td>Study</td><td>Data</td><td>Feature selection</td><td>Evaluation</td><td>Reported result</td><td>Reported time</td></tr><tr><td>Ertam and Kaya [4]</td><td>4 classes, duplicates kept</td><td>None (11 features)</td><td>Multiclass SVM (linear, polynomial, RBF,</td><td>Best F1 76.4 (RBF; P 63.0, R 97.1); accuracy not reported</td><td></td></tr><tr><td>Lillmond and Suddul [38]</td><td>3 classes (reset-both merged into deny), undersampled</td><td>None (11 features)</td><td>reported 5-fold CV, MLP (10,5)</td><td>Acc. 94.49, F1 94.97</td><td></td></tr><tr><td>Verma and Sahu [31]</td><td>4 classes, duplicates kept</td><td>TPSO wrapper (Naïve Bayes, 10-fold CV on fuli data), 5 features</td><td>10-fold CV, DNN</td><td>Acc. 98.29, P 100, R 99.99, F1</td><td>Model build 845.13 s (all) / 668.66 s (TPSO)</td></tr><tr><td>Talaeepoor et al. [39]</td><td>4 classes, duplicates kept</td><td>None (11 features); parameters tuned by grid search</td><td>2-, 5-, 10-, 20-fold CV, decision tree</td><td>Acc. 99.85, P 94.08, R 87.69, F1 90.77 (2-fold)</td><td></td></tr><tr><td>Ours, original [13] (Section 3.1)</td><td>4 classes, duplicates kept</td><td>QBSO-FS (5-fold CV on full data), 9 features</td><td>5-fold CV, RF</td><td>Acc. 99.87, Macro-F1 91.27</td><td></td></tr><tr><td>Ours, leakage-free (Section 3.2)</td><td>4 classes, deduplicated</td><td>QBSO-FS (validation set), 7 features</td><td>Held-out test, RF</td><td>Acc. 99.90, Macro-F1 94.88</td><td>Search 96.3 s, training 0.17 s</td></tr></table>

To compare the approaches on equal terms, we re-ran their configurations under our leakage-free protocol, with the same split and the same four classes (Table 12). We evaluated three feature sets with each model: all features, the 5-feature TPSO subset reported by Verma and Sahu, and the QBSO-FS subset. In addition to RF, KNN, and Bernoulli NB, we included the two DNN configurations of the prior studies, implemented as MLPs with standard scaling. DNN-L follows Lillmond and Suddul [38] (two hidden layers of 10 and 5 neurons with an L2 penalty), and DNN-V follows Verma and Sahu [31] (hidden layers of 11 neurons trained with stochastic gradient descent (SGD); because the number of hidden layers is not stated, we used two). We did not run QBSO-FS with a DNN in the loop because of its cost, and the RF subset was reused for the DNNs instead. Table 13 reports the corresponding run times.

For RF, KNN, and Bernoulli NB, the QBSO-FS subset gives the highest Macro-F1 on both data sets on this split. For Bernoulli NB, however, its margin over all features and over the TPSO subset on the raw data is below 0.2 pp. The TPSO subset is close to all features for RF and KNN on the raw data, and it is clearly worse for Bernoulli NB on the processed data (85.09% accuracy versus 98.91% for the QBSO subset). For the two DNNs, the opposite holds. The TPSO subset performs about as well as all features, whereas the RF-selected QBSO subset lowers accuracy by about 1 pp. The best overall result, RF with the 7-feature QBSO subset on the deduplicated data (99.90% accuracy, 94.88% Macro-F1), exceeds the best of our DNN re-implementations (99.81% accuracy, 82.47% Macro-F1) and needs 0.17 s of training instead of 3.72 s. The Macro-R of most DNN runs stays below 75%, which is consistent with the reset-both class being rarely or never predicted.

The QBSO-FS search takes between 3 s (Bernoulli NB) and 96 s (RF) and is performed once. It uses 270–551 subset evaluations, each of which is a single fit on the training set. With the reported population size and number of iterations (20 and 20), TPSO evaluates about 400 subsets, each with a 10-fold CV of a Naïve Bayes classifier; its search time was not reported. Once a subset is chosen, the savings in training and inference time are small in absolute terms on this data set. For example, RF training drops from 0.29 s to 0.24 s on the raw data. The savings are also not consistent: KNN inference becomes slower with its QBSO subset (10.10 to 15.53 µs per record), and MLP training time is governed by the number of iterations until convergence rather than by the input dimension.

Table 12: Re-evaluation of prior configurations under the leakage-free protocol (test set, %). “TPSO” denotes the 5-feature subset selected in the TPSO study; “QBSO” denotes the 5–7-feature subset selected by QBSO-FS on the validation set for the same classifier (Table 9; for the two DNNs, the RF subset is reused). Best Macro-F1 per row in bold.
<table><tr><td></td><td></td><td colspan="2">All</td><td colspan="2">TPSO [31]</td><td colspan="2">QBSO [12]</td></tr><tr><td>Data set</td><td>Model</td><td>Acc.</td><td>Macro-F1</td><td>Acc.</td><td>Macro-F1</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td rowspan="5">Raw</td><td>RF</td><td>99.78</td><td>88.12</td><td>99.69</td><td>88.63</td><td>99.84</td><td>89.57</td></tr><tr><td>KNN</td><td>99.37</td><td>74.51</td><td>99.12</td><td>74.32</td><td>99.76</td><td>93.22</td></tr><tr><td>Bernoulli NB</td><td>79.80</td><td>42.25</td><td>80.13</td><td>42.39</td><td>80.19</td><td>42.42</td></tr><tr><td>DNN-L (MLP 10–5)</td><td>99.64</td><td>74.72</td><td>99.65</td><td>74.72</td><td>98.12</td><td>72.93</td></tr><tr><td>DNN-V (MLP 11–11, SGD)</td><td>99.02</td><td>74.00</td><td>99.05</td><td>74.05</td><td>98.14</td><td>72.99</td></tr><tr><td rowspan="5">Processed</td><td>RF</td><td>99.79</td><td>81.89</td><td>99.82</td><td>87.28</td><td>99.90</td><td>94.88</td></tr><tr><td>KNN</td><td>99.20</td><td>74.15</td><td>99.19</td><td>74.13</td><td>99.83</td><td>82.49</td></tr><tr><td>Bernoulli NB</td><td>98.15</td><td>73.04</td><td>85.09</td><td>43.44</td><td>98.91</td><td>73.43</td></tr><tr><td>DNN-L (MLP 10–5)</td><td>99.81</td><td>82.47</td><td>99.73</td><td>74.68</td><td>98.90</td><td>73.41</td></tr><tr><td>DNN-V (MLP 11–11, SGD)</td><td>99.50</td><td>74.36</td><td>99.02</td><td>73.66</td><td>98.85</td><td>73.37</td></tr></table>

Table 13: Run time of feature selection, training (fit on train+validation), and inference (per test record). Medians of three runs for RF, KNN, and Bernoulli NB, single run for DNNs; Apple M3 Max (16 cores).
<table><tr><td rowspan="2">Data set</td><td rowspan="2">Model</td><td colspan="2">QBSO-FS search</td><td colspan="3">Training time (s)</td><td colspan="2">Inference (μs / record)</td></tr><tr><td>Time (s)</td><td>Evals</td><td>All</td><td>TPSO</td><td>QBSO</td><td>All</td><td>QBSO</td></tr><tr><td rowspan="5">Raw</td><td>RF</td><td>71.2</td><td>303</td><td>0.29</td><td>0.29</td><td>0.24</td><td>1.13</td><td>1.13</td></tr><tr><td>KNN</td><td>51.8</td><td>270</td><td>0.06</td><td>0.06</td><td>0.06</td><td>10.10</td><td>15.53</td></tr><tr><td>Bernoulli NB</td><td>17.3</td><td>292</td><td>0.08</td><td>0.07</td><td>0.07</td><td>0.10</td><td>0.07</td></tr><tr><td>DNN-L (MLP 10–5)</td><td>一</td><td>一</td><td>6.71</td><td>7.32</td><td>4.09</td><td>0.19</td><td>0.17</td></tr><tr><td>DNN-V (MLP 11–11, SGD)</td><td>一</td><td>1</td><td>9.18</td><td>8.66</td><td>10.03</td><td>0.20</td><td>0.18</td></tr><tr><td rowspan="5">Processed</td><td>RF</td><td>96.3</td><td>551</td><td>0.21</td><td>0.22</td><td>0.17</td><td>1.29</td><td>1.31</td></tr><tr><td>KNN</td><td>81.5</td><td>547</td><td>0.00</td><td>0.01</td><td>0.01</td><td>11.80</td><td>16.68</td></tr><tr><td>Bernoulli NB</td><td>3.1</td><td>536</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.12</td><td>0.08</td></tr><tr><td>DNN-L (MLP 10–5)</td><td>一</td><td>一</td><td>3.72</td><td>8.57</td><td>1.45</td><td>0.18</td><td>0.17</td></tr><tr><td>DNN-V (MLP 11–11, SGD)</td><td>一</td><td>一</td><td>2.21</td><td>3.34</td><td>2.25</td><td>0.18</td><td>0.17</td></tr></table>

## 4 Discussion

Effect of leakage on the reported accuracy. Although 18% of the raw test records are exact copies of training records, the test accuracy of RF and KNN is almost the same on the raw and on the deduplicated data (Table 10). The leakage-free accuracies of the selected subsets (99.84% and 99.90% for RF, 99.76% and 99.83% for KNN) are also close to the biased values of the original experiment (99.87% and 99.78%, Table 6). This suggests that, for this data set, the optimism in the original accuracy figures is small. A plausible reason is that most duplicates belong to the deny class (6,945 of 8,362), which is already well separated by port and volume features. However, this comparison does not isolate the two sources of bias. The raw and processed data differ in both duplicates and features, and the original figures come from the first version of this study [13], whose implementation differs from the one used here. Quantifying selection bias and duplicate leakage separately would require re-running the original protocol with the same implementation, and evaluating a deduplicated data set that keeps the original 11 features. Regardless of the size of the effect, the deduplicated held-out protocol is the estimate that should be reported, because it removes both sources of optimism by construction.

Macro-averaged metrics and the reset-both class. Accuracy is dominated by the three large classes, whereas the macro-averaged scores are strongly influenced by reset-both, which has only 54 records in total and about 11 in each test set. A single misclassified reset-both record changes Macro-R by roughly 2 pp. This explains the larger and less regular differences in Macro-R and Macro-F1 across settings, for example KNN on the processed data, where precision is near-perfect but Macro-R is 79.40%. Conclusions about minority-class performance should therefore be drawn with care, and repeated splits or a class-weighted fitness would be needed to optimize them reliably.

Effect of preprocessing on Bernoulli NB. Bernoulli NB binarizes every input at zero, so on the raw data most features carry little information (ports, bytes, and packets are almost always non-zero) and accuracy stays at about 80%. On the processed data, the NAT-source-is-zero and NAT-destination-is-zero flags and the source and destination port ranges are binary or low-cardinality indicators, and accuracy rises to 98.15% with all features and 98.91% with the selected subset. Preprocessing is thus far more important for Bernoulli NB than feature selection.

Comparison with previous studies. The numbers reported in previous studies (Table 11) should not be compared directly with ours or with each other, because the class definitions, balancing, feature selection data, and averaging differ. The common-protocol re-evaluation (Table 12) therefore gives the more reliable comparison. It also shows that wrapper-selected subsets [23] are classifier-specific: the RF subset does not transfer to an MLP, and the TPSO subset, chosen with a Naïve Bayes classifier on raw values, does not suit Bernoulli NB on the processed features. A subset selected for one classifier should therefore not be reused for another without re-evaluation.

Run time. A motivation of this study was that high-dimensional features take a long time to classify. With only 11 to 16 features, this motivation does not hold for the present data set: the one-off search cost of QBSO-FS (at most about 1.5 minutes) is larger than the training time it saves (Section 3.3). The main benefit of feature selection here is therefore better accuracy and Macro-F1 for RF and KNN, rather than speed. Run time comparisons with previous studies are not possible, because they used different hardware, did not report search times, and in one case reported no times at all.

Limitations. Several features are recorded only after the firewall has applied its action. Bytes received and packets received are zero in 99.9% of deny and 100% of drop sessions, but in only 9.8% of allow sessions; elapsed time and the NAT source port are zero in 99.9–100% of deny and drop sessions, compared with 1.0% and 1.5% of allow sessions. The classifiers therefore largely recognize the consequences of the action rather than predict it from information available when the decision is made. This makes the task easier than real-time policy decisions, and limits the direct use of the models to retrospective log analysis and auditing. KNN was run without feature scaling, so part of its gain from feature selection may come from removing features with large numeric ranges, such as ports. The leakage-free experiment uses a single stratified split and a single run of the search, and the QBSO-FS budget of Table 4 is small (3 bees, 3 iterations). Small differences, in particular in the macro-averaged scores, are within the variability that can be expected from a different split or run. In addition, apart from the TPSO subset of Verma and Sahu [31], which we evaluated but did not re-run, QBSO-FS is compared only with using all features. Comparisons with simpler feature selection baselines, such as ranking by random forest importance or mutual information, or random subsets evaluated with the same budget, are not included. Repeating the protocol over several splits and runs, testing whether the improvement is significant, and adding such baselines are left for future work.

## 5 Conclusion

In this paper, we applied QBSO-FS to select features for firewall log classification. The first version of this study [13] reported that the selected features outperformed all features. However, those results were obtained by selecting and evaluating feature subsets on the same CV folds of a data set that contains 12.8% duplicated records, and they are therefore optimistically biased.

In the extended experiment, we removed exactly duplicated records and selected features on a separate validation set, keeping the test set untouched. Under this leakage-free protocol, QBSO-FS with the original hyper-parameters still selected compact subsets (5–7 features). On a single split, these subsets clearly improved Macro-F1 over all features for KNN on both data sets and for RF on the deduplicated data. For RF on the raw data and for Bernoulli NB, the differences were small and may not be robust, so in these cases the subsets should be regarded as matching, rather than exceeding, the full feature set. RF with 7 selected features on the deduplicated data reached 99.90% accuracy and 94.88% Macro-F1. These results suggest that, at least for KNN and RF, the benefit of QBSO-FS feature selection for firewall log classification is not merely an artifact of test-set leakage. Confirming the size of the benefit over multiple splits and against other feature selection baselines is left for future work.

## Code and data availability

The Internet Firewall Data Set used in this study is publicly available from the UCI Machine Learning Repository at https://archive.ics.uci.edu/dataset/542/internet+firewall+data [15]. For the implementation of QBSO-FS, please refer to https://github.com/amineremache/qbso-fs [37].

## References

[1] J. A. Castañeda and F. J. Montoro. The effect of internet general privacy concern on customer behavior. Electronic Commerce Research, 7(2):117–141, 2007.

[2] N. K. Singh, D. S. Tomar, and B. N. Roy. An approach to understand the end user behavior through log analysis. International Journal ofComputer Applications, 5(11):27–34, 2010.

[3] S. Allagi and R. Rachh. Analysis of network log data using machine learning. In 2019 IEEE 5th International Conference for Convergence in Technology (I2CT), pages 1–3, 2019.

[4] F. Ertam and M. Kaya. Classification of firewall log files with multiclass support vector machine. In 2018 6th International Symposium on Digital Forensic and Security (ISDFS), pages 1–4, 2018.

[5] A. L. Buczak and E. Guven. A survey of data mining and machine learning methods for cyber security intrusion detection. IEEE Communications Surveys & Tutorials, 18(2):1153–1176, 2016.

[6] D. C. Le, N. Zincir-Heywood, and M. I. Heywood. Analyzing data granularity levels for insider threat detection using machine learning. IEEE Transactions on Network and Service Management, 17(1):30–44, 2020.

[7] A. Javaid, Q. Niyaz, W. Sun, and M. Alam. A deep learning approach for network intrusion detection system. In Proceedings of the 9th EAI International Conference on Bio-inspired Information and Communications Technologies (formerly BIONETICS), pages 21–26, 2016.

[8] D. Sharma, V. Wason, and P. Johri. Optimized classification of firewall log data using heterogeneous ensemble techniques. In 2021 International Conference on Advance Computing and Innovative Technologies in Engineering (ICACITE), pages 368–372, 2021.

[9] S. Lei. A feature selection method based on information gain and genetic algorithm. In 2012 International Conference on Computer Science and Electronics Engineering, pages 355–358, 2012.

[10] J. Janisch, T. Pevný, and V. Lisý. Classification with costly features using deep reinforcement learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 33, pages 3959–3966, 2019.

[11] W. Fan, K. Liu, H. Liu, P. Wang, Y. Ge, and Y. Fu. AutoFS: Automated feature selection via diversity-aware interactive reinforcement learning. In 2020 IEEE International Conference on Data Mining (ICDM), pages 1008–1013, 2020.

[12] S. Sadeg, L. Hamdad, A. R. Remache, M. N. Karech, K. Benatchba, and Z. Habbas. QBSO-FS: A reinforcement learning based bee swarm optimization metaheuristic for feature selection. In International Work-Conference on Artificial Neural Networks, pages 785–796, 2019.

[13] S. Han, G. Hong, J. Kim, J. Yu, S. Lee, B. Cho, and J. Jeon. Optimal feature selection for firewall log analysis using machine learning and hybrid metaheuristic algorithms. engrXiv preprint, 2021.

[14] G. Hong et al. A study on the feature selection for firewall log analysis using bee swarm optimization with reinforcement learning. In The Korean Institute of Communications and Information Sciences Winter Conference, pages 921–922, 2021.

[15] F. Ertam. Internet firewall data. UCI Machine Learning Repository, 2019.

[16] A. Frank and A. Asuncion. UCI machine learning repository. http://archive.ics.uci.edu/ml, 2010. Irvine, CA: University of California, School of Information and Computer Science.

[17] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani. A detailed analysis of the KDD CUP 99 data set. In 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), pages 1–6, 2009.

[18] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman. Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4):15:1–15:21, 2012.

[19] M. Cotton, L. Eggert, J. Touch, M. Westerlund, and S. Cheshire. Internet assigned numbers authority (IANA) procedures for the management of the service name and transport protocol port number registry. RFC 6335, Internet Engineering Task Force, 2011.

[20] H. Drias, S. Sadeg, and S. Yahi. Cooperative bees swarm for solving the maximum weighted satisfiability problem. In Computational Intelligence and Bioinspired Systems (IWANN 2005), volume 3512 of Lecture Notes in Computer Science, pages 318–325. Springer, 2005.

[21] S. Sadeg, L. Hamdad, K. Benatchba, and Z. Habbas. BSO-FS: Bee swarm optimization for feature selection in classification. In Advances in Computational Intelligence (IWANN 2015), volume 9094 of Lecture Notes in Computer Science, pages 387–399. Springer, 2015.

[22] C. J. Watkins and P. Dayan. Q-learning. Machine Learning, 8(3–4):279–292, 1992.

[23] R. Kohavi and G. H. John. Wrappers for feature subset selection. Artificial Intelligence, 97(1–2):273–324, 1997.

[24] S. Han, S. Kitagata, I. Chanpornpakdi, T. Tanaka, and S. M. Nam. External Sinkhole Attack Detection in Large-Scale WSNs Using Metaheuristic Feature Selection. In Proc. IEEE 15th Global Conference on Consumer Electronics (GCCE), pages 1542–1544, 2026.

[25] P. Pudil, J. Novovicová, and J. Kittler. Floating search methods in feature selection. ˇ Pattern Recognition Letters, 15(11):1119–1125, 1994.

[26] F. Glover. Tabu search—part I. ORSA Journal on Computing, 1(3):190–206, 1989.

[27] J. Kennedy and R. Eberhart. Particle swarm optimization. In Proceedings of ICNN’95 – International Conference on Neural Networks, volume 4, pages 1942–1948, 1995.

[28] J. Kennedy and R. C. Eberhart. A discrete binary version of the particle swarm algorithm. In 1997 IEEE International Conference on Systems, Man, and Cybernetics, volume 5, pages 4104–4108, 1997.

[29] W. Siedlecki and J. Sklansky. A note on genetic algorithms for large-scale feature selection. Pattern Recognition Letters, 10(5):335–347, 1989.

[30] D. H. Wolpert and W. G. Macready. No free lunch theorems for optimization. IEEE Transactions on Evolutionary Computation, 1(1):67–82, 1997.

[31] P. Verma and S. K. Sahu. Unlocking the potential of deep neural networks with tuned particle swarm optimization for firewall log data analysis: Empowering network security insights. In 2024 OPJU International Technology Conference (OTCON) on Smart Computingfor Innovation and Advancement in Industry 4.0. IEEE, 2024.

[32] L. Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[33] T. Cover and P. Hart. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1): 21–27, 1967.

[34] A. McCallum and K. Nigam. A comparison of event models for naive Bayes text classification. In AAAI-98 Workshop on Learning for Text Categorization, pages 41–48, 1998.

[35] C. Ambroise and G. J. McLachlan. Selection bias in gene extraction on the basis of microarray gene-expression data. Proceedings ofthe National Academy ofSciences, 99(10):6562–6566, 2002.

[36] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and É. Duchesnay. Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12:2825–2830, 2011.

[37] A. R. Remache. QBSO-FS: Official Python implementation. https://github.com/amineremache/qbso-fs, 2019. GitHub repository.

[38] C. Lillmond and G. Suddul. A deep neural network approach for analysis of firewall log data. In International Conference on Advances in Technology and Computing (ICATC-2021), pages 42–46, Sri Lanka, 2021. Faculty of Computing and Technology, University of Kelaniya.

[39] M. Talaeepoor, B. Soleimani Neysiani, and M. Davarpanah Jezi. Improving the performance of firewalls through network traffics logs’ classification of Firat dataset using decision tree algorithm. In 2024 10th International Conference on Web Research (ICWR). IEEE, 2024.