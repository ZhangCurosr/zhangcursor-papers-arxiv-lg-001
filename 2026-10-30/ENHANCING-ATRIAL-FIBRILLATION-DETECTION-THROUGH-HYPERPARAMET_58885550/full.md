# ENHANCING ATRIAL FIBRILLATION DETECTION THROUGH HYPERPARAMETER OPTIMIZATION

Sara Gjorgjieva<sup>1</sup> <sup>2</sup> <sup>[0009-0009-2014-4515]</sup>, Ilinka Ivanoska<sup>2</sup> <sup>[0000-0002-9363-1527]</sup>, Marjan Gusev<sup>1</sup> <sup>2</sup> <sup>[0000-0003-0351-9783]</sup>

<sup>1</sup>Innovation Dooel, Skopje, North Macedonia <sup>2</sup>Ss. Cyril and Methodius University in Skopje, Faculty of Computer Science and Engineering, North Macedonia

sara.gjorgjieva@innovation.com.mk {ilinka.ivanoska, marjan.gushev}@finki.ukim.mk

In ICT Innovations 2025. AI and the Digital Frontier, Communications in Computer and Information Science series, Springer (In Press).

## ABSTRACT

This study explores the role of hyperparameter optimization (HPO) in enhancing the detection of Atrial Fibrillation (AF) using both machine learning and deep learning models. We evaluated a range of standard classifiers - Decision Tree, Random Forest, and XGBoost classifiers, alongside deep learning architectures including Recurrent Neural Networks, Deep Neural Networks, and Siamese Neural Networks. To fine-tune these models, we applied four HPO methods: Genetic Algorithm, Hyperband, Optuna, and SMAC3. The experiments were conducted using inter-patient and intrapatient data splits to assess model generalization and robustness in the presence of class imbalance, which is reflective of real-world AF occurrence. Our results, based on PhysioNet electrocardiogram datasets, showed that the Hyperband-optimized XGBoost model outperformed all others, achieving a peak F1 score of 96.95%. These findings highlight the critical role of suitable HPO strategies in enhancing the accuracy and robustness of AF classification outcomes in clinical settings.

Keywords Atrial Fibrillation · AF · ECG · Genetic Algorithm · Hyperband · Optuna · SMAC3 · deep learning · machine learning

## 1 Introduction

Machine learning (ML) has revolutionized various domains by offering innovative solutions to real-world challenges, improving the quality of life, and extending life expectancy. Its versatility stems from the fact that ML models are highly adaptable and can be applied across diverse fields, regardless of the subject matter. However, selecting the most suitable model for a problem can be challenging, requiring substantial time, resources, and expertise.

Once a model is selected, the next critical step is optimizing its performance. This involves fine-tuning the model parameters to ensure it delivers the best results for the given task. Model effectiveness is typically evaluated based on its ability to generalize and perform with high accuracy to unseen data, striking a balance between underfitting and overfitting. Achieving this balance depends on two key factors: the model features and the hyperparameters that dictate the model’s learning behavior [1, 2]. Although related, hyperparameters and parameters differ significantly in their roles. Parameters, such as weights and coefficients, are determined by the model during training and represent how predictor variables influence the target variable. On the other hand, hyperparameters are external configuration variables set before training and control the model’s structure, learning rate, and other operational aspects. While default hyperparameter values exist, they often need to be tailored to the specific dataset and model through experimentation, known as hyperparameter tuning or hyperparameter optimization (HPO). HPO involves systematically testing different hyperparameter combinations to identify the one that achieves the best performance. Unlike parameters adjusted during training, hyperparameters are predefined and remain fixed throughout training. Choosing the correct hyperparameter values directly impacts model performance, making hyperparameter tuning a critical step in the ML pipeline [3, 4]. Data scientists usually enhance a model’s predictive capabilities by fine-tuning hyperparameters and evaluating the performance of new data. This study focuses on the importance of HPO in improving the detection of atrial fibrillation (AF), and careful experimentation is planned to achieve the best results. We evaluate the effect of various HPO techniques: Hyperband [5, 6] , Optuna [7, 8], Genetic Algorithm (GA) [9, 10], and Sequential Model-Based Algorithm Configuration (SMAC3) [11–13], highlighting their unique advantages.

The main research question is to evaluate selected HPO methods for AF detection models, focusing on robustness and the generalizability of optimized hyperparameters across different patient distributions and benchmark datasets. Unlike previous studies, our work explicitly incorporates inter- and intra-patient evaluations to assess robustness more thoroughly, where intra-patient refers to training and testing within the same dataset, and inter-patient refers to training on one dataset and testing on another.

Section 1 outlines the significance of AF detection and the role of HPO in enhancing model performance. Section 2 reviews prior work on AF detection using ML. Section 3 describes the datasets, models, and HPO techniques applied in this study. Section 4 presents the experimental setup, implementation tools, execution time, and model performance results. Section 5 compares our findings with prior research and evaluates the strengths and limitations of the HPO methods used. Finally, Section 6 summarizes the key insights and suggests directions for future work

## 2 Related Work

Several studies have applied HPO techniques to improve AF detection with various ML models. A common limitation is the lack of explicit handling of class imbalance—most optimize for accuracy, which biases models toward the majority class. Without F1-score or AUC, tuning often yields poor recall for AF cases, reducing robustness. Our study addresses this by emphasizing metrics that better reflect real-world detection challenges.

Optuna has been widely adopted for AF detection due to its flexibility. Tudjarski et al. [14] used Optuna with XGBoost on 41-beat windows, achieving 99% F1-score. Rawshani et al. [15] applied it in a multimodal framework combining CNNs (AlexNet, VGG-16) with XGBoost, reporting 97.23% AUC-ROC. However, both relied on simple splits (80/20, 70/30) with mixed patient samples, limiting generalizability. By contrast, our work uses multiple datasets and patient-wise splits for broader robustness.

GA has also shown promise in complex search spaces. Al Qaraghuli et al. [16] fine-tuned a CNN using GA, achieving 100% sensitivity, 98.90% specificity, and 99.95% accuracy on MIT-BIH AF. Yet, unclear data-splitting raises concerns about overfitting.

Hyperband has been applied to DL optimization by balancing exploration and exploitation. Han et al. [17] used it on CNN and LSTM, achieving 72.80% accuracy and 59.31% macro-F1, slightly outperforming manual tuning. Chen et al. [18] optimized a Dual-Input Mixed Neural Network (DMNN) with ECG and demographic data, reaching 83.4% AUC. While Han et al. used 5-fold CV on CinC 2017, Chen et al. used two datasets but only one experiment from a single institution, risking demographic bias.

Our review found no prior use of SMAC3 for AF detection.

While all HPO methods improve performance, their strategies and efficiency differ. Hyperband is efficient for DL with limited resources; Optuna explores structured spaces via probabilistic modeling; GA handles high-dimensional searches but is computationally expensive. The choice depends on model complexity, dataset properties, and resources. Importantly, these methods do not inherently address imbalance—bias often arises from accuracy-based tuning. To counter this, we use F1-score and AUC as optimization metrics for more reliable performance on AF datasets.

## 3 Methodology

## 3.1 Datasets

We used two publicly available ECG datasets from PhysioNet [19]: the MIT-BIH Arrhythmia Database (MITDB) [20] and the Long-Term AF Database (LTAFDB) [21]. MITDB contains 30-minute ECG recordings from 48 patients, with over 100,000 annotated heartbeats, approximately 10% correspond to AF. LTAFDB offers 24 hours recordings from 83 patients, with a higher AF prevalence of about 50%. Due to its large size, we used a subset of around 940,000 annotated heartbeats, obtained by segmenting the recordings into fixed-length windows of 42 beats across all patients.

![](images/d3b151340b9e1f32451ebb1bb814982284e0d6df136fc7e247e0516f09d36974.jpg)  
Figure 1: Overview of the HPO-supported ML model development

## 3.2 Models

To effectively detect AFIB, we explored a range of models spanning from traditional machine learning to more advanced neural architectures. Decision Trees (DT) [22] serve as a straightforward baseline, offering quick and interpretable classifications by recursively splitting features, though their simplicity often limits their ability to model complex ECG dynamics.

Building on this, Random Forests (RF) [23] introduce ensemble learning by combining multiple DTs trained on random subsets of the data, which improves robustness and reduces the risk of overfitting—particularly useful for handling noise in ECG recordings. XGBoost (XGB) [24] further enhances tree-based modeling by applying gradient boosting with built-in regularization, allowing it to capture subtle patterns in structured heartbeat features while maintaining generalization.

In contrast to these tree-based methods, Recurrent Neural Networks (RNNs) [25] are well-suited for sequential data like ECG signals, as they can model temporal dependencies between heartbeats. Our implementation of RNNs includes techniques like dropout, L2 regularization, and Leaky ReLU activations to improve learning stability and prevent overfitting. Deep Neural Networks (DNNs) [26], while not inherently sequential, extract hierarchical features through stacked dense layers and are particularly effective when working with engineered inputs; regularization and dropout here also support better generalization.

Finally, Siamese Neural Networks (SNNs) [27] take a unique approach by learning to compare pairs of ECG segments using twin networks with shared weights. This architecture excels in tasks that require identifying subtle rhythm similarities or differences—making it especially powerful for distinguishing AFIB from normal sinus rhythm in a patient-agnostic way.

## 3.3 Hyperparameter Optimization Techniques

In our previous work [28], we manually set the hyperparameters for these models. However, manual tuning is both time-consuming and inefficient, often failing to identify optimal parameter combinations for best performance.

Even though there is no single HPO method or universal set of hyperparameters that guarantees optimal performance across all models and datasets. Therefore, a systematic and tailored approach is essential. The key challenge lies in selecting the next combination of hyperparameters to test the exploration of new configurations, in order to consistently achieve better performance.

To ensure diversity in optimization strategies, we selected four HPO techniques that employ fundamentally different search mechanisms, allowing us to compare their effectiveness in improving AF detection: Hyperband [18] , Optuna [14] , [15], GA [16], and SMAC3. Each of these approaches offers different optimization strategies and is suited to specific scenarios, providing a robust framework for improving model performance in detecting AF.

Fig. 1 provides an overview of the architecture used to optimize ML model development through HPO techniques. In the following sections, we briefly describe how each technique functions and why it is relevant for optimizing ML models in the context of AF classification.

Hyperband is a powerful HPO technique that uses bandit-based early stopping to efficiently allocate computational resources. It is especially beneficial when working with complex time-series models, such as those used in AF detection, as it quickly eliminates poorly performing hyperparameter configurations and prevents excessive resource consumption. In this domain, where models process large volumes of ECG signal data, Hyperband ensures that only the most promising configurations are further explored. This dynamic resource allocation accelerates the tuning process for computationally intensive models like RNNs and DNNs, helping improve both training efficiency and performance.

Optuna is a modern HPO method that uses Bayesian Optimization and Tree-structured Parzen Estimators (TPE) to efficiently search for optimal hyperparameters. It intelligently balances exploration and exploitation, making it more effective than grid or random search in high-dimensional spaces. In AF detection, where fine-tuning impacts critical aspects like feature extraction and model sensitivity, Optuna’s built-in pruning mechanism is especially useful, it can stop unpromising trials early, reducing training time without compromising performance. This is particularly advantageous when optimizing complex models like neural networks or XGBoost.

Genetic Algorithm (GA) mimics the process of natural selection to evolve better hyperparameter combinations across generations. GA’s population-based search is beneficial for AF detection models due to its ability to explore a wide and diverse range of configurations. Techniques like mutation and crossover help GA escape local optima, making it well-suited for tuning highly nonlinear models like DNNs and SNNs. This is particularly important when working with datasets such as MITDB and LTAFDB, where data distributions vary and require adaptive tuning for optimal classification.

SMAC3(Sequential Model-Based Algorithm Configuration) is a Bayesian optimization approach that builds a probabilistic model of the search space and iteratively refines it to guide the search. For AF detection tasks, where models must capture subtle ECG signal variations, SMAC3 offers a systematic and data-driven method to identify high-performing hyperparameter configurations. It is especially useful for tree-based models like RF, DT, and XGB, as it can efficiently tune hyperparameters such as depth, learning rate, and feature criteria. By learning from past evaluations, SMAC3 reduces redundant trials and focuses computational effort on the most promising regions of the search space.

## 4 Experimental Setup and Results

## 4.1 Experiments

To evaluate the effectiveness of HPO, we designed four experimental setups that vary in training and testing data splits. These experiments are categorized into intra-dataset (same dataset for training and testing) and inter-dataset (different datasets for training and testing) evaluations. Only clean AF segments were used for training, while testing included all signal types.

Intra-dataset evaluations use an 80/20 split:

E11: MITDB is used for both training and testing.

E22: LTAFDB is used for both training and testing.

Inter-dataset evaluations involve cross-dataset testing:

E12: Train on MITDB, test on LTAFDB.

E21: Train on LTAFDB, test on MITDB.

These setups allow us to assess model robustness across intra-patient and inter-patient scenarios. In E11 and E22, samples from the same patients are kept separate between training and testing (intra-patient split). In E12 and E21, datasets differ completely (inter-patient setting), helping evaluate generalization across patient populations.

Table 1 summarizes the experimental configurations applied to each HPO method.

## 4.2 Implementation

All experiments were implemented in Python using Scikit-learn, XGBoost, TensorFlow, and PyTorch. For hyperparameter optimization, we applied Optuna, GA (DEAP), Hyperband (HpBandSter), and SMAC3.

Training was conducted on an AMD Ryzen 7 4800H CPU, leveraging all cores for parallelization. No GPU was used, and computations relied on the default system configuration. Each method ran 50 trials, with the F1-score as the optimization objective to balance precision and recall.

Table 1: Specification of experiments
<table><tr><td>Exp.</td><td>Train</td><td>Test</td><td>Evaluation</td><td>Split</td></tr><tr><td>E11</td><td>MITDB</td><td>MITDB</td><td>intra-patient</td><td>80/20</td></tr><tr><td>E12</td><td>MITDB</td><td>LTAFDB</td><td>inter-patient</td><td>dataset</td></tr><tr><td>E21</td><td>LTAFDB</td><td>MITDB</td><td>inter-patient</td><td>dataset</td></tr><tr><td>E22</td><td>LTAFDB</td><td>LTAFDB</td><td>intra-patient</td><td>80/20</td></tr></table>

Table 2 summarizes the configurations of the HPO techniques. Parameter settings were chosen based on prior work, recommendations from the respective libraries, and empirical testing on smaller pilot runs. For example, Hyperband parameters were tuned to allow early stopping without sacrificing exploration; Optuna was capped at 50 trials to ensure convergence within the available budget; GA settings balanced population size and mutation rate to explore both shallow and deeper evolutionary searches; and SMAC3 emphasized quality-driven optimization to avoid overfitting to rapid gains. These choices reflect trade-offs between computational efficiency, robustness, and thoroughness in exploring the hyperparameter space.

Table 2: HPO Models Configurations
<table><tr><td>Optimizer</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="2">Hyperband</td><td>factor max resources</td><td>3 100</td></tr><tr><td>min resources max epoch</td><td>20 20</td></tr><tr><td>Optuna</td><td>trials</td><td>50</td></tr><tr><td>GA</td><td>population ngen cxpb mutpb</td><td>[10,30,70] [5,10,30] 0.5 0.2</td></tr><tr><td>SMAC3</td><td>trials objectives</td><td>[30,50,150] quality</td></tr></table>

## 4.3 Execution Time Results

Fig. 2 presents the execution times (in seconds on a logarithmic scale) for all experiments applying different HPO techniques. Hyperband exhibits the lowest execution time for the neural networks, while SMAC3 performs the fastest for the RNN and SNN. Overall, Hyperband and SMAC3 are the most computationally efficient HPO methods.

In contrast, GA has the highest execution times, particularly for complex models like RF, DNN, RNN, and SNN, indicating its high computational cost. Optuna is also relatively expensive for deep learning architectures. These result reflect differences in optimization strategies, where Hyperband and SMAC3 achieve efficiency through early stopping and surrogate modeling, while Optuna and GA rely on more exhaustive search procedures.

## 4.4 Performance Results

Given the class imbalance typical in AF detection, relying solely on accuracy can be misleading. Therefore, we report multiple metrics—Precision, Recall (Sensitivity), F1-score, Accuracy, and AUC—to provide a more reliable and comprehensive evaluation. These metrics help balance false positives and false negatives, which is critical in clinical applications where both misdiagnosis and missed detection carry risks.

The results, summarized in Tab. 3–4, highlight the effectiveness of HPO techniques in improving classification performance. For clarity, we bolded the best F1-score for each experiment–model pair, since F1 best reflects balanced performance under class imbalance.

![](images/c65b4de79773fb9ca5e618ab23685d9ceadfafb5cde2ca9ea17ea7c5a08b7167.jpg)  
Figure 2: Visual representation for the execution time(s) for different HPO

Table 3: Performance Metrics (in %) for Experiments E11 (left) and E12 (right)
<table><tr><td></td><td colspan="6">Experiment E11</td></tr><tr><td>Opt.</td><td>Model</td><td>Acc</td><td>PPV</td><td>Sen</td><td>F1</td><td>AUC</td></tr><tr><td>Hyperband</td><td>DT</td><td>96.87</td><td>79.02</td><td>96.72</td><td>86.98</td><td>96.80</td></tr><tr><td></td><td>RF</td><td>97.24</td><td>83.11</td><td>93.50</td><td>88.00</td><td>95.60</td></tr><tr><td></td><td>XGB</td><td>97.55</td><td>86.15</td><td>92.19</td><td>89.07</td><td>95.20</td></tr><tr><td></td><td>DNN</td><td>96.92</td><td>79.90</td><td>95.53</td><td>87.02</td><td>96.31</td></tr><tr><td></td><td>RNN</td><td>97.18</td><td>81.74</td><td>95.19</td><td>96.31</td><td>87.95</td></tr><tr><td></td><td>SNN</td><td>96.92</td><td>79.90</td><td>95.53</td><td>87.02</td><td>96.31</td></tr><tr><td>Optuna</td><td>DT</td><td>97.55</td><td>85.03</td><td>93.83</td><td>89.22</td><td>95.92</td></tr><tr><td></td><td>RF</td><td>97.51</td><td>85.28</td><td>93.04</td><td>88.99</td><td>95.55</td></tr><tr><td></td><td>XGB</td><td>97.53</td><td>85.78</td><td>92.48</td><td>89.00</td><td>95.31</td></tr><tr><td></td><td>DNN</td><td>97.16</td><td>84.35</td><td>90.55</td><td>87.34</td><td>94.26</td></tr><tr><td></td><td>RNN</td><td>97.36</td><td>86.03</td><td>90.21</td><td>88.07</td><td>94.22</td></tr><tr><td></td><td>SNN</td><td>96.84</td><td>79.24</td><td>95.87</td><td>86.77</td><td>96.41</td></tr><tr><td>GA</td><td>DT</td><td>96.96</td><td>81.05</td><td>93.85</td><td>86.98</td><td>95.59</td></tr><tr><td></td><td>RF</td><td>97.52</td><td>88.09</td><td>89.08</td><td>88.58</td><td>93.81</td></tr><tr><td></td><td>XGB</td><td>97.68</td><td>86.88</td><td>92.48</td><td>89.59</td><td>95.39</td></tr><tr><td></td><td>DNN</td><td>97.14</td><td>81.43</td><td>95.25</td><td>87.80</td><td>96.31</td></tr><tr><td></td><td>RNN</td><td>96.91</td><td>79.47</td><td>96.32</td><td>87.09</td><td>96.65</td></tr><tr><td></td><td>SNN</td><td>96.99</td><td>80.47</td><td>95.31</td><td>87.26</td><td>96.25</td></tr><tr><td>SMAC3</td><td>DT</td><td>97.57</td><td>86.06</td><td>92.32</td><td>89.08</td><td>95.26</td></tr><tr><td></td><td>RF</td><td>97.43</td><td>84.56</td><td>93.12</td><td>88.63</td><td>95.53</td></tr><tr><td></td><td>XGB</td><td>97.67</td><td>86.45</td><td>92.89</td><td>89.55</td><td>95.57</td></tr><tr><td></td><td>DNN</td><td>97.05</td><td>82.68</td><td>91.81</td><td>87.01</td><td>94.75</td></tr><tr><td></td><td>RNN</td><td>96.63</td><td>83.52</td><td>85.51</td><td>84.50</td><td>91.74</td></tr><tr><td></td><td>SNN</td><td>77.40</td><td>32.74</td><td>99.83</td><td>49.31</td><td>87.23</td></tr><tr><td>Without HPO</td><td>RF</td><td>一</td><td>89.36</td><td>91.66</td><td>90.49</td><td>一</td></tr></table>

<table><tr><td>Model</td><td></td><td>Acc</td><td>PPV</td><td>Sen</td><td>F1</td><td>AUC</td></tr><tr><td></td><td>DT</td><td>96.85</td><td>97.02</td><td>90.06</td><td>93.41</td><td>94.57</td></tr><tr><td></td><td>RF</td><td>96.05</td><td>97.48</td><td>86.30</td><td>91.55</td><td>92.78</td></tr><tr><td></td><td>XGB</td><td>96.98</td><td>97.32</td><td>90.30</td><td>93.68</td><td>94.74</td></tr><tr><td></td><td>DNN</td><td>96.29</td><td>97.84</td><td>86.95</td><td>92.07</td><td>93.16</td></tr><tr><td></td><td>RNN</td><td>96.98</td><td>97.71</td><td>89.89</td><td>94.60</td><td>93.64</td></tr><tr><td></td><td>SNN</td><td>97.16</td><td>95.00</td><td>93.47</td><td>94.23</td><td>95.92</td></tr><tr><td></td><td>DT</td><td>97.34</td><td>96.46</td><td>92.65</td><td>94.52</td><td>95.77</td></tr><tr><td></td><td>RF</td><td>97.16</td><td>97.34</td><td>91.03</td><td>94.08</td><td>95.10</td></tr><tr><td></td><td>XGB</td><td>96.98</td><td>97.32</td><td>90.29</td><td>93.68</td><td>94.74</td></tr><tr><td></td><td>DNN</td><td>97.35</td><td>93.29</td><td>96.20</td><td>94.73</td><td>96.96</td></tr><tr><td></td><td>RNN</td><td>97.15</td><td>97.38</td><td>90.95</td><td>94.05</td><td>95.07</td></tr><tr><td></td><td>SNN</td><td>96.86</td><td>91.70</td><td>96.00</td><td>93.80</td><td>96.57</td></tr><tr><td></td><td>DT</td><td>97.34</td><td>96.46</td><td>92.65</td><td>94.52</td><td>95.77</td></tr><tr><td></td><td>RF</td><td>95.61</td><td>97.34</td><td>84.59</td><td>90.51</td><td>91.91</td></tr><tr><td></td><td>XGB</td><td>97.40</td><td>96.93</td><td>92.45</td><td>94.64</td><td>95.74</td></tr><tr><td></td><td>DNN</td><td>97.43</td><td>96.53</td><td>92.96</td><td>94.71</td><td>95.93</td></tr><tr><td></td><td>RNN</td><td>97.54</td><td>96.57</td><td>93.39</td><td>94.95</td><td>96.15</td></tr><tr><td></td><td>SNN</td><td>97.34</td><td>95.78</td><td>93.35</td><td>94.55</td><td>96.00</td></tr><tr><td></td><td>DT</td><td>96.92</td><td>97.22</td><td>90.15</td><td>93.55</td><td>94.65</td></tr><tr><td></td><td>RF</td><td>97.18</td><td>97.29</td><td>91.14</td><td>94.12</td><td>95.15</td></tr><tr><td></td><td>XGB</td><td>96.95</td><td>97.29</td><td>90.20</td><td>93.61</td><td>94.69</td></tr><tr><td></td><td>DNN</td><td>97.40</td><td>95.58</td><td>93.84</td><td>94.70</td><td>96.21</td></tr><tr><td></td><td>RNN</td><td>97.49</td><td>95.49</td><td>94.32</td><td>94.90</td><td>96.42</td></tr><tr><td></td><td>SNN</td><td>96.68</td><td>97.45</td><td>88.92</td><td>92.99</td><td>94.08</td></tr><tr><td></td><td>RF</td><td>一</td><td>89.36</td><td>91.66</td><td>90.49</td><td></td></tr></table>

Table 4: Performance Metrics (in %) for Experiments E21 (left) and E22 (right)
<table><tr><td></td><td colspan="6">Experinent E21</td></tr><tr><td>Opt.</td><td>Model</td><td>Acc</td><td>PPV</td><td>Sen</td><td>F1</td><td>AUC</td></tr><tr><td>Hyperband</td><td>DT</td><td>89.22</td><td>50.99</td><td>98.82</td><td>67.27</td><td>93.42</td></tr><tr><td></td><td>RF</td><td>91.07</td><td>55.83</td><td>97.46</td><td>70.99</td><td>93.86</td></tr><tr><td></td><td>XGB</td><td>91.04</td><td>55.77</td><td>97.06</td><td>70.84</td><td>93.67</td></tr><tr><td></td><td>DNN</td><td>91.31</td><td>56.50</td><td>97.45</td><td>71.53</td><td>93.99</td></tr><tr><td></td><td>RNN</td><td>88.92</td><td>50.30</td><td>98.27</td><td>93.01</td><td>66.54</td></tr><tr><td></td><td>SNN</td><td>92.84</td><td>88.71</td><td>41.41</td><td>56.46</td><td>70.37</td></tr><tr><td>Optuna</td><td>DT</td><td>89.22</td><td>50.99</td><td>98.82</td><td>67.27</td><td>93.42</td></tr><tr><td></td><td>RF</td><td>90.82</td><td>55.09</td><td>97.72</td><td>70.46</td><td>93.83</td></tr><tr><td></td><td>XGB</td><td>91.28</td><td>56.47</td><td>96.97</td><td>71.38</td><td>93.77</td></tr><tr><td></td><td>DNN</td><td>94.52</td><td>70.34</td><td>88.35</td><td>78.32</td><td>91.83</td></tr><tr><td></td><td>RNN</td><td>90.77</td><td>54.99</td><td>97.30</td><td>70.27</td><td>93.63</td></tr><tr><td></td><td>SNN</td><td>91.29</td><td>56.47</td><td>97.04</td><td>71.40</td><td>93.80</td></tr><tr><td>GA</td><td>DT</td><td>90.34</td><td>53.79</td><td>98.10</td><td>69.48</td><td>93.73</td></tr><tr><td></td><td>RF</td><td>90.73</td><td>55.06</td><td>93.92</td><td>69.42</td><td>92.12</td></tr><tr><td></td><td>XGB</td><td>94.10</td><td>68.52</td><td>87.61</td><td>76.90</td><td>91.27</td></tr><tr><td></td><td>DNN</td><td>92.50</td><td>60.41</td><td>95.95</td><td>74.14</td><td>94.01</td></tr><tr><td></td><td>RNN</td><td>91.40</td><td>56.87</td><td>96.25</td><td>71.50</td><td>93.52</td></tr><tr><td></td><td>SNN</td><td>92.24</td><td>59.46</td><td>96.55</td><td>73.59</td><td>94.12</td></tr><tr><td>SMAC3</td><td>DT</td><td>91.28</td><td>56.47</td><td>96.80</td><td>71.33</td><td>93.69</td></tr><tr><td></td><td>RF</td><td>90.79</td><td>55.01</td><td>97.84</td><td>70.42</td><td>93.87</td></tr><tr><td></td><td>XGB</td><td>91.20</td><td>56.24</td><td>96.98</td><td>71.19</td><td>93.73</td></tr><tr><td></td><td>DNN</td><td>92.38</td><td>59.96</td><td>96.20</td><td>73.88</td><td>94.05</td></tr><tr><td></td><td>RNN</td><td>89.88</td><td>52.58</td><td>98.40</td><td>68.54</td><td>93.60</td></tr><tr><td></td><td>SNN</td><td>71.79</td><td>28.43</td><td>100.00</td><td>44.27</td><td>84.11</td></tr><tr><td>Without HPO</td><td>SNN</td><td></td><td>64.36</td><td>93.78</td><td>76.34</td><td></td></tr></table>

<table><tr><td colspan="7">Experiment E22</td></tr><tr><td></td><td>Model</td><td>Acc</td><td>PPV</td><td>Sen</td><td>F1</td><td>AUC</td></tr><tr><td></td><td>DT</td><td>98.18</td><td>96.33</td><td>96.59</td><td>96.46</td><td>97.66</td></tr><tr><td></td><td>RF</td><td>98.18</td><td>96.67</td><td>96.22</td><td>96.44</td><td>97.54</td></tr><tr><td></td><td>XGB</td><td>98.43</td><td>97.11</td><td>96.80</td><td>96.95</td><td>97.90</td></tr><tr><td></td><td>DNN</td><td>98.24</td><td>97.25</td><td>95.83</td><td>96.54</td><td>97.45</td></tr><tr><td></td><td>RNN</td><td>98.32</td><td>96.80</td><td>96.64</td><td>96.71</td><td>96.72</td></tr><tr><td></td><td>SNN</td><td>80.49</td><td>99.84</td><td>23.97</td><td>38.66</td><td>61.98</td></tr><tr><td></td><td>DT</td><td>98.33</td><td>96.94</td><td>96.54</td><td>96.74</td><td>97.74</td></tr><tr><td></td><td>RF</td><td>98.26</td><td>96.46</td><td>96.75</td><td>96.61</td><td>97.77</td></tr><tr><td></td><td>XGB</td><td>98.34</td><td>96.96</td><td>96.57</td><td>96.77</td><td>97.76</td></tr><tr><td></td><td>DNN</td><td>98.30</td><td>96.93</td><td>96.44</td><td>96.68</td><td>97.69</td></tr><tr><td></td><td>RNN</td><td>98.30</td><td>96.36</td><td>97.03</td><td>96.69</td><td>97.88</td></tr><tr><td></td><td>SNN</td><td>98.31</td><td>96.97</td><td>96.44</td><td>96.70</td><td>97.70</td></tr><tr><td></td><td>DT</td><td>98.18</td><td>96.33</td><td>96.59</td><td>96.46</td><td>97.66</td></tr><tr><td></td><td>RF</td><td>98.05</td><td>96.89</td><td>95.48</td><td>96.18</td><td>97.21</td></tr><tr><td></td><td>XGB</td><td>98.35</td><td>97.02</td><td>96.53</td><td>96.77</td><td>97.75</td></tr><tr><td></td><td>DNN</td><td>98.29</td><td>96.80</td><td>96.54</td><td>96.67</td><td>97.72</td></tr><tr><td></td><td>RNN</td><td>98.31</td><td>96.96</td><td>96.43</td><td>96.69</td><td>97.69</td></tr><tr><td></td><td>SNN</td><td>98.10</td><td>97.75</td><td>94.78</td><td>96.24</td><td>97.01</td></tr><tr><td></td><td>DT</td><td>98.34</td><td>96.91</td><td>96.60</td><td>96.76</td><td>97.77</td></tr><tr><td></td><td>RF</td><td>98.24</td><td>96.48</td><td>96.69</td><td>96.59</td><td>97.74</td></tr><tr><td></td><td>XGB</td><td>98.37</td><td>96.99</td><td>96.66</td><td>96.83</td><td>97.81</td></tr><tr><td></td><td>DNN</td><td>98.27</td><td>96.69</td><td>96.55</td><td>96.62</td><td>97.71</td></tr><tr><td></td><td>RNN</td><td>98.20</td><td>97.01</td><td>96.03</td><td>96.52</td><td>97.50</td></tr><tr><td></td><td>SNN</td><td>96.31</td><td>96.49</td><td>88.89</td><td>92.53</td><td>93.88</td></tr><tr><td></td><td>CB</td><td>一</td><td>97.06</td><td>96.58</td><td>96.82</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

In Experiment E11, GA performed well on accuracy and PPV with RF, while SMAC3 reached the highest sensitivity (99.83%) with SNN but with very low PPV, showing the recall–precision trade-off. Hyperband achieved the best F1 and AUC, offering a more balanced outcome.

In E12, GA again excelled with the highest accuracy (97.54%) and F1 (94.95%) using RNN. Optuna maximized sensitivity (96.20%) with DNN but slightly reduced PPV. SMAC3 and Hyperband maintained strong PPV, especially with DT and RF.

In E21, Optuna reached the highest accuracy (94.52%) with DNN. Hyperband excelled in PPV and F1 with SNN, while SMAC3 achieved perfect sensitivity (100%) but with low PPV, leading to many false positives.

In E22, Hyperband was the overall best, with the highest accuracy (98.43%), F1 (96.95%), and AUC (97.90%) using XGB, though sometimes at the cost of sensitivity.

Across experiments, SMAC3 was the most stable, consistently balancing sensitivity, F1, and AUC, particularly with DL models. Hyperband and SMAC3 delivered the most competitive results overall—Hyperband often producing the top F1-scores, while SMAC3 provided robust, balanced performance across models.

## 5 Discussion

## 5.1 Performance vs. Efficiency Trade-offs

Among all experiments, E22 consistently demonstrated the best and most stable performance across all models. This makes it the most suitable setup for benchmarking. Within E22, the XGB classifier tuned with Hyperband achieved the highest performance overall, reaching an F1-score of 96.95%. This highlights not only the strength of tree-based methods for AF detection but also the importance of effective hyperparameter optimization.

Hyperband proved to be the most efficient HPO method, combining high predictive performance with computational efficiency. By aggressively eliminating underperforming configurations early, Hyperband explores a wide hyperparame ter space with reduced training cost. Compared to more resource-intensive methods like GA and Optuna, which may require more iterations or model evaluations, Hyperband achieves a strong balance between speed and model quality.

Table 5 presents a comparative overview of the best results obtained with each optimizer. Hyperband slightly outperformed GA, Optuna, and SMAC3 in all performance metrics, establishing it as the most effective HPO strategy for this task.

SMAC3 also demonstrated robust and stable performance across experiments, especially in terms of sensitivity and F1-score. While slightly slower than Hyperband, its model-based search strategy provides a reliable alternative when consistent performance is prioritized over speed.

## 5.2 Effect of Hyperparameter Optimization Across Experiments

Hyperparameter optimization led to substantial improvements across many models and experiments, highlighting its critical role in boosting performance. In particular as we can note from Fig. 3, Experiment E21 demonstrated the most significant gains. The RNN model tuned with Hyperband saw its F1-score jump from 72.61% to 93.01%, suggesting that recurrent models are highly sensitive to parameter tuning and benefit greatly from resource-aware optimization. Similarly, DNN with Optuna improved from 71.18% to 78.32%, and XGB with GA improved from 73.14% to 76.90%.

Table 5: The best results for E22
<table><tr><td>Optimizer</td><td>Model</td><td>Acc</td><td>PPV</td><td>Sen</td><td>F1</td></tr><tr><td>GA</td><td>XGB</td><td>98.35</td><td>97.02</td><td>96.53</td><td>96.77</td></tr><tr><td>Hyperband</td><td>XGB</td><td>98.43</td><td>97.11</td><td>96.80</td><td>96.95</td></tr><tr><td>Optuna</td><td>XGB</td><td>98.34</td><td>96.96</td><td>96.57</td><td>96.77</td></tr><tr><td>SMAC3</td><td>XGB</td><td>98.37</td><td>96.99</td><td>96.66</td><td>96.83</td></tr></table>

![](images/d57a7e4ffbd3a2de9e8c7fc532043d557dfee6c3480f56a5159d3d87550026b3.jpg)  
(a) Results for E11

![](images/31c2bf296f09252feae6396ed8376339438bf4e14e0a1250dbb763662e292ab4.jpg)  
(b) Results for E12

![](images/b5a9665233d00a89eecaa36e8fb65de3d823066738d1af2e79c8ba43386ed294.jpg)  
(c) Results for E21

![](images/8d5a8d0a603902aa8fd008cfa5f815e9f9ac08893b05811d5cb34c847a27e07d.jpg)  
(d) Results for E22  
Figure 3: Performance comparison for all experiments (E11, E12, E21, E22).

In Experiment E12, traditional models like DT and RF showed modest but consistent gains. DT improved from 93.85% to 94.52%, RF from 93.42% to 94.08% and XGB with GA reached 94.64%. These results reinforce the effectiveness of GA for fine-tuning tree-based methods.

Even in high-performing settings like E22, minor improvements were observed. XGB with Hyperband improved slightly from 96.77% to 96.95% and XGB with SMAC3 reached 96.83%. This demonstrates that even when models are near their performance ceiling, fine-tuning can achieve incremental gains.

However, not all optimization efforts resulted in improvement. For example, DT and RF in E21 underperformed after Hyperband tuning, and several models in E22 experienced slight drops. These setbacks may stem from overfitting to the validation set, poorly defined search spaces, or convergence to local optima. In scenarios where baseline models already perform well, aggressive tuning can introduce instability or noise, leading to marginal or negative changes.

Overall, the results suggest that while HPO is not universally beneficial, it can dramatically improve outcomes when model complexity, data distribution, and optimizer strategy are well-aligned.

## 5.3 Comparison with State-of-the-Art Approaches

Table 6 compares our optimized models against several state-of-the-art methods for AF detection. Some studies applied Grid Search (GS), which systematically evaluates predefined parameter combinations. Other work used BOHB (Bayesian Optimization with Hyperband), which balances exploration and exploitation more efficiently.

As opposed to these prior works, our study systematically applied and compared multiple HPO techniques to optimize both ML and DL models. As a result, our best-performing model—XGB tuned with Hyperband—achieved an F1- score of 96.95% and an AUC of 97.90%, outperforming most prior work in both predictive strength and robustness. Additionally, our inter-patient split setup ensures a more realistic generalization evaluation compared to the commonly used mixed-patient splits in other studies.

Table 6: Comparison with state-of-the-art approaches
<table><tr><td>Ref.</td><td>Optimizer</td><td>Model</td><td>Method</td><td>Split</td><td>Acc</td><td>Sen</td><td>PPV</td><td>F1</td><td>AUC</td></tr><tr><td>[14]</td><td>Optuna</td><td>XGB</td><td>mixed</td><td>70/30</td><td>-</td><td>=</td><td>-</td><td>99.00</td><td>–</td></tr><tr><td>[15]</td><td>Optuna</td><td>AlexNet</td><td>mixed</td><td>80/20</td><td>-</td><td>-</td><td>一</td><td></td><td>97.23</td></tr><tr><td>[16]</td><td>GA</td><td>CNN</td><td>mixed</td><td>-</td><td>99.95</td><td>100</td><td>-</td><td></td><td>-</td></tr><tr><td>[29]</td><td>GS</td><td>SVM</td><td>mixed</td><td>60/40</td><td>96.50</td><td>-</td><td>-</td><td>一</td><td>-</td></tr><tr><td>[17]</td><td>BOHB</td><td>CNN/LSTM</td><td>mixed</td><td>80/20</td><td>72.80</td><td>-</td><td>-</td><td>59.31</td><td>-</td></tr><tr><td>[18]</td><td>Hyperband</td><td>DMNN</td><td>intra</td><td>dataset</td><td>-</td><td>-</td><td>1</td><td>-</td><td>83.40</td></tr><tr><td>our</td><td>Hyperband</td><td>RNN</td><td>intra</td><td>80/20</td><td>97.18</td><td>81.74</td><td>95.19</td><td>96.31</td><td>87.95</td></tr><tr><td>our</td><td>Hyperband</td><td>XGB</td><td>intra</td><td>80/20</td><td>98.43</td><td>97.11</td><td>96.80</td><td>96.95</td><td>97.90</td></tr><tr><td>our</td><td>GA</td><td>RNN</td><td>inter</td><td>dataset</td><td>97.54</td><td>96.57</td><td>93.39</td><td>94.95</td><td>96.15</td></tr><tr><td>our</td><td>Hyperband</td><td>RNN</td><td>inter</td><td>dataset</td><td>88.92</td><td>50.30</td><td>98.27</td><td>93.01</td><td>66.54</td></tr></table>

## 5.4 Impact of Dataset Splitting on Generalization

Dataset splitting strategy plays a critical role in evaluating model generalizability. Many prior studies use mixed or intra-patient splits, where data from the same individuals appear in both training and testing sets. While this often yields higher performance, it risks overfitting to patient-specific patterns.

In contrast, our study adopts a strict inter-patient split, ensuring no overlap between patients across training and testing. Although this typically leads to slightly lower metrics, it provides a more realistic and robust assessment of model performance.

The strong results achieved under this setup demonstrate the models’ ability to generalize across individuals—an essential requirement for clinical deployment and real world applicability.

## 6 Conclusion

This study evaluated the impact of HPO techniques on AF detection using a range of ML and DL models. The results show that effective HPO significantly improves classification performance across both traditional and neural network-based models.

Among the tested combinations, XGBoost optimized with Hyperband under an inter-patient 80/20 split achieved the highest overall performance, with an AUC of 97.90%, accuracy of 98.43%, and sensitivity of 96.80%. These results highlight strong predictive capability and generalization, especially in inter-patient settings that better reflect real-world clinical variability and reduce the risk of overfitting to patient-specific patterns.

The findings also emphasize the limitations of relying on single dataset training or mixed-patient splits, which may produce overly optimistic results. In contrast, our use of inter-dataset and inter-patient evaluations provides a more robust assessment of model stability and generalizability.

Notably, we introduced SMAC3—a rarely used HPO method in AF detection—which demonstrated consistent performance and shows promise for broader application in medical time-series optimization tasks.

For future work, we plan to expand the evaluation to more diverse ECG datasets and explore domain adaptation and transfer learning techniques. Additionally, developing ensemble approaches and resource-efficient models will be critical to supporting real-time, clinically deployable AF detection systems.

## References

[1] Hilde JP Weerts, Andreas C Mueller, and Joaquin Vanschoren. Importance of tuning hyperparameters of machine learning algorithms. arXiv preprint arXiv:2007.07588, 2020.

[2] Constantin Aliferis and Gyorgy Simon. Overfitting, underfitting and general model overconfidence and underperformance pitfalls and best practices in machine learning and ai. Artificial intelligence and machine learning in health care and medical sciences: Best practices and pitfalls, pages 477–524, 2024.

[3] Matthias Feurer and Frank Hutter. Hyperparameter optimization. Springer International Publishing, 2019.

[4] Kizito Nyuytiymbiy. Parameters and hyperparameters in machine learning and deep learning. Towards Data Science, 30:12, 2020.

[5] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18(185):1–52, 2018.

[6] Bernd Bischl, Martin Binder, Michel Lang, Tobias Pielok, Jakob Richter, Stefan Coors, Janek Thomas, Theresa Ullmann, Marc Becker, Anne-Laure Boulesteix, et al. Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 13(2):e1484, 2023.

[7] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2623–2631, 2019.

[8] Tanay Agrawal. Optuna and automl. In Hyperparameter Optimization in Machine Learning: Make Your Machine Learning and Deep Learning Models More Efficient, pages 109–129. Springer, 2020.

[9] Annu Lambora, Kunal Gupta, and Kriti Chopra. Genetic algorithm-a literature review. In 2019 international conference on machine learning, big data, cloud and parallel computing (COMITCon), pages 380–384. IEEE, 2019.

[10] Sourabh Katoch, Sumit Singh Chauhan, and Vijay Kumar. A review on genetic algorithm: past, present, and future. Multimedia tools and applications, 80:8091–8126, 2021.

[11] Marius Lindauer, Katharina Eggensperger, Matthias Feurer, André Biedenkapp, Difan Deng, Carolin Benjamins, Tim Ruhkopf, René Sass, and Frank Hutter. Smac3: A versatile bayesian optimization package for hyperparameter optimization. Journal ofMachine Learning Research, 23(54):1–9, 2022.

[12] Jeroen G Rook, Carolin Benjamins, Jakob Bossek, Heike Trautmann, Holger H Hoos, and Marius Lindauer. Mo-smac: multi-objective sequential model-based algorithm configuration. Evolutionary Computation, pages 1–25, 2025.

[13] Anh Truong, Austin Walters, Jeremy Goodsitt, Keegan Hines, C Bayan Bruss, and Reza Farivar. Towards automated machine learning: Evaluation and comparison of automl approaches and tools. In 2019 IEEE 31st international conference on tools with artificial intelligence (ICTAI), pages 1471–1479. IEEE, 2019.

[14] Stojancho Tudjarski, Tomislav Ignjatov, and Marjan Gusev. A new ml-based afib detector. In 2021 29th Telecommunications Forum (TELFOR), pages 1–4. IEEE, 2021.

[15] Araz Rawshani, Aidin Rawshani, J Gustav Smith, Jan Borén, Deepak L Bhatt, Mats Börjesson, Johan Engdahl, Peter Lundgren, Antros Louca, Truls Råmunddal, et al. Enhancing atrial fibrillation prediction using multimodal neural networks and xgboost integration. 2024.

[16] Hawraa Al Qaraghuli, Reza Sheibani, and Hamid Tabatabaee. Detection of atrial fibrillation using variable length genetic algorithm and convolutional neural network. Concurrency and Computation: Practice and Experience, 34(10):e6789, 2022.

[17] Seungwoo Han, Heesang Eom, Juhyeong Kim, and Cheolsoo Park. Optimal dnn architecture search using bayesian optimization hyperband for arrhythmia detection. In 2020 IEEE Wireless Power Transfer Conference (WPTC), pages 357–360. IEEE, 2020.

[18] Wei-Wen Chen, Chih-Min Liu, Chien-Chao Tseng, Ching-Chun Huang, I-Chien Wu, Pei-Fen Chen, Shih-Lin Chang, Yenn-Jiang Lin, Li-Wei Lo, Fa-Po Chung, et al. Identifying the presence of atrial fibrillation during sinus rhythm using a dual-input mixed neural network with ecg coloring technology. BMC Medical Research Methodology, 24(1):318, 2024.

[19] George B Moody. Physionet. In Encyclopedia of Computational Neuroscience, pages 2806–2808. Springer, 2022.

[20] George B Moody and Roger G Mark. Mit-bih arrhythmia database. (No Title), 1992.

[21] Bartosz Biczuk, Szymon Bus, Sebastian´ Zurek, Jarosław Piskorski, and Przemysław Guzik. prr30, prr3. 25% and<sup>˙</sup> asymmetrical entropy descriptors in atrial fibrillation detection. Entropy, 26(4):296, 2024.

[22] Guangyu Bin, Minggang Shao, Guanghong Bin, Jiao Huang, Dingchang Zheng, and Shuicai Wu. Detection of atrial fibrillation using decision tree ensemble. In 2017 Computing in Cardiology (CinC), pages 1–4. IEEE, 2017.

[23] Morteza Zabihi, Ali Bahrami Rad, Aggelos K Katsaggelos, Serkan Kiranyaz, Susanna Narkilahti, and Moncef Gabbouj. Detection of atrial fibrillation in ecg hand-held devices using a random forest classifier. In 2017 computing in cardiology (cinc), pages 1–4. IEEE, 2017.

[24] Yao Chen, Xiao Wang, Yonghan Jung, Vida Abedi, Ramin Zand, Marvi Bikak, and Mohammad Adibuzzaman. Classification of short single-lead electrocardiograms (ecgs) for atrial fibrillation detection using piecewise linear spline and xgboost. Physiological measurement, 39(10):104006, 2018.

[25] Supreeth P Shashikumar, Amit J Shah, Gari D Clifford, and Shamim Nemati. Detection of paroxysmal atrial fibrillation using attention-based bidirectional recurrent neural networks. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 715–723, 2018.

[26] Sushravya Raghunath, John M Pfeifer, Alvaro E Ulloa-Cerna, Arun Nemani, Tanner Carbonati, Linyuan Jing, David P vanMaanen, Dustin N Hartzel, Jeffery A Ruhl, Braxton F Lagerman, et al. Deep neural networks can predict new-onset atrial fibrillation from the 12-lead ecg and help identify those at risk of atrial fibrillation–related stroke. Circulation, 143(13):1287–1298, 2021.

[27] Yiuwai Ng, Min-Tsun Liao, Ting-Li Chen, Chih-Kuo Lee, Cheng-Ying Chou, and Weichung Wang. Few-shot transfer learning for personalized atrial fibrillation detection using patient-based siamese network with single-lead ecg records. Artificial Intelligence in Medicine, 144:102644, 2023.

[28] Sara Gjorgjieva, Ana Angjelevska, Dimitar Trajanov, and Marjan Gusev. A new afib detection method with deep neural networks. page 6. in press, 2024.

[29] Sreenivasulu Ummadisetty and Madhavi Tatineni. Atrial fibrillation detection using ecg recordings based on genetic optimization. Engineering Proceedings, 66(1):21, 2024.