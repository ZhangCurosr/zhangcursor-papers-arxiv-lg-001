# Neural Symbolic Regression Using Deep Learning and Sparse Modeling

Ravi Kumar U<sup>∗</sup>

Department of Mathematics

Indian Institute of Space Science and Technology

Thiruvananthapuram, India

Email: 014raviku@gmail.com

Sumitra S<sup>†</sup>

Department of Mathematics

Indian Institute of Space Science and Technology

Thiruvananthapuram, India

Email: sumitra@iist.ac.in

Abstract—Symbolic Regression (SR) seeks to find succinct mathematical expressions that represent the fundamental relationships within data, providing interpretability and scientific understanding that exceeds that of black-box models. Nevertheless, traditional methods like Genetic Programming face challenges with scalability and are highly sensitive to noise, while sparse regression techniques such as SINDy rely significantly on predetermined feature libraries.

In this work, we present a Neural Symbolic Regression (NSR) framework that treats neural networks as functional preconditioners for symbolic discovery. Our approach uses a decoupled pipeline: a neural network first learns a smooth, noise-robust approximation of the target function in an interaction-aware nonlinear feature space. LASSO is then applied to extract sparse, interpretable closed-form expressions.

To improve predictive accuracy and symbolic fidelity by integrating distributed hyperparameter optimization with Ray Tune and ASHA scheduling. Experiments on the Nguyen benchmark suite show that our approach consistently outperforms SINDy and non-tuned neural baselines in RMSE, noise robustness, and out-of-distribution generalization. Ablation studies confirm the significance of feature interactions, neural depth, and tuning strategies.

In general, this study presents a scalable and understandable neural-symbolic framework, creating a solid link between neural approximation and the discovery of sparse equations for scientific machine learning.

Index Terms—Symbolic Regression, Neural Networks, Sparse Modeling, LASSO, Interpretable ML, Hyperparameter Optimization, Ray Tune, GPU Acceleration.

## I. INTRODUCTION

Symbolic Regression (SR) seeks to autonomously uncover closed-form mathematical formulas that represent data, providing interpretability, analytical understanding, and opportunities for scientific breakthroughs. In contrast to opaque machine learning models, SR generates clear equations that illuminate the fundamental mechanisms instead of just capturing observed trends. This feature renders SR especially useful in scientific and engineering fields, where comprehending the foundational relationships is as crucial as achieving predictive accuracy.

Initial SR studies were largely focused on Genetic Programming (GP), initiated by Koza [1], which develops symbolic expressions through biologically motivated operators. Although very expressive, GP experiences slow convergence, significant computational expenses, and a propensity for code bloat, restricting its use in high-dimensional or noisy realworld datasets.

To tackle these issues, sparse modeling techniques like SINDy (Sparse Identification of Nonlinear Dynamics) [2] reframe symbolic regression as a regression task utilizing a predefined library of nonlinear features. By utilizing sparsitypromoting methods like LASSO, SINDy circumvents combinatorial search and facilitates effective equation identification. Its efficacy, however, hinges significantly on the expressiveness of the manually created feature library and its capacity to represent intricate nonlinear interactions.

Recent progress in deep learning has inspired neural methods for symbolic regression. Approaches like Deep Symbolic Regression (DSR) [3] and neural-symbolic frameworks [4] employ reinforcement learning, sequence modeling, or extensive neural architectures to produce symbolic expressions. Although these methods enhance scalability and flexibility, they usually integrate function approximation and symbolic discovery in a single optimization procedure, resulting in instability, increased computational costs, and challenges in maintaining interpretability.

Hybrid neural-symbolic approaches seek to merge the advantages of neural networks with sparse modeling techniques. In this study, we take a distinctly alternative approach: instead of employing neural networks to directly create symbolic expressions, we consider them as functional preconditioners for symbolic regression. A neural network is initially employed to acquire a smooth, noise-resistant approximation of the target function within a feature space that considers interactions, followed by the use of sparse regression to obtain an interpretable symbolic representation. The proposed method enhances noise robustness, stabilizes equation recovery, and maintains interpretability by implementing a rigorous separation between neural approximation and symbolic extraction.

Expanding on this viewpoint, we introduce a Neural Symbolic Regression (NSR) framework that combines nonlinear feature library development, GPU-boosted neural approximation, sparse symbolic extraction through LASSO [5], and distributed hyperparameter tuning utilizing Ray Tune with ASHA scheduling [7], [8]. The Nguyen benchmark suite [9] evaluates performance over various functional types, such as polynomial, transcendental, and mixed expressions.

The main concept behind NSR is that neural networks can be utilized not as generators of symbols, but as smoothing operators that convert noisy observations into representations suitable for sparse recovery. This reimagining offers a reasoned approach to enhance the identifiability of symbolic structure while ensuring scalability.

Although the suggested framework shows significant robustness and interpretability, its scalability is impacted by the combinatorial expansion of the feature library, a constraint examined thoroughly in the complexity analysis.

## A. Contributions and Novelty

This work introduces a structured neural–symbolic regression framework that addresses key limitations of both classical symbolic regression and recent neural approaches. The primary contributions are as follows:

1) Decoupled Neural–Symbolic Pipeline with Explicit Functional Separation. In contrast to earlier hybrid methods like AI Feynman and neural-guided symbolic regression techniques that mix approximation with symbolic search, we suggest a fully decoupled framework where a neural network functions solely as a smooth, noise-resistant functional approximator, succeeded by an independent sparse symbolic recovery phase. This division minimizes noise sensitivity and stabilizes symbolic extraction, especially when nonlinear and transcendental elements are involved.

2) Interaction-Aware Feature Library with Controlled Expressivity. We present a nonlinear feature library focused on interactions that merges basic transformations with organized pairwise interactions. Unlike conventional SINDy-style libraries that depend on rigid polynomial expansions, the suggested library strikes a balance between expressiveness and manageability, allowing for the retrieval of cross-dimensional connections while preserving compatibility with sparse regression.

3) Neural Preconditioning for Sparse Symbolic Recovery. We show that neural approximation can serve as a preconditioning phase for sparse regression, effectively reducing noise in observations and enhancing the clarity of symbolic structures. This viewpoint contrasts with current neural symbolic regression techniques, which involve the direct generation or optimization of symbolic expressions, and offers a more reliable option for extracting interpretable equations.

4) Integrated Hyperparameter Optimization for Neural–Symbolic Models. We integrate distributed hyperparameter optimization with Ray Tune and ASHA into the symbolic regression workflow. Although hyperparameter tuning is common in deep learning, its systematic incorporation into neural-symbolic regression remains largely unexamined. Our findings indicate that adjustment greatly affects both predictive precision and symbolic integrity.

5) Comprehensive Empirical Evaluation Beyond Standard Benchmarks. Alongside standard Nguyen benchmarks, we assess robustness across different noise levels, out-of-distribution generalization, and the impact of architectural and feature design choices through ablation studies. These examinations offer a more profound understanding of how neural-symbolic systems operate and emphasize the significance of decisions made in pipeline design.

In summary, the suggested framework views neural networks as functional preconditioners for symbolic regression, providing a structured approach to enhance robustness, interpretability, and generalization in discovering equations from data.

## II. BACKGROUND AND RELATED WORK

Symbolic Regression (SR) aims to discover closed-form analytical formulas that represent an underlying data-generating mechanism. In contrast to traditional black-box learning models, SR generates interpretable mathematical representations that enhance scientific comprehension, allow extrapolation beyond the training domain, and facilitate analytical reasoning. Initial advancements in SR were mainly propelled by Genetic Programming (GP), as presented by Koza [1], which develops symbolic expressions through biologically motivated operators. Despite providing significant representational flexibility, GP experiences code bloat, slow convergence, and increased computational overhead, especially in noisy or highdimensional settings. Later improvements, such as Paretobased complexity management [11] and grammar-driven approaches, mitigate certain inefficiencies yet do not completely eliminate scalability constraints.

Sparse modeling methods have surfaced as a viable alternative. The SINDy framework [2] defines SR as sparse regression on a specified nonlinear feature library, successfully modeling dynamical systems. However, its performance is heavily reliant on the richness of the manually created feature library. These techniques are based on the principles of compressed sensing theory [12] and traditional sparsity methods like LASSO [5].

Concurrent advancements in neural networks, bolstered by universal approximation theorems [6], [14], [15], have inspired SR methods based on neural technologies. Deep Symbolic Regression (DSR) [3] defines equation discovery as a sequence generation process directed by reinforcement learning, while transformer-based models like Neural SR That Scales [4] utilize extensive pretraining to enhance scalability. AI Feynman [16] integrates neural approximation with physics-based heuristics and symbolic simplification, attaining impressive results on scientific datasets. Thorough surveys [19] offer a wider perspective on these advancements and highlight significant unresolved issues.

A major constraint of these neural-symbolic methods is the close interdependence between function approximation and symbolic discovery in a single or iterative optimization framework. Although this integration boosts flexibility, it may lead to instability, escalate computational complexity, and hinder the control of interpretability and symbolic consistency.

TABLE I  
COMPARISON OF SYMBOLIC REGRESSION APPROACHES
<table><tr><td>Method</td><td>Search Strategy</td><td>Interpretability Scalability</td><td></td><td>Noise Ro- bustness</td></tr><tr><td>GP</td><td>Evolutionary</td><td>High</td><td>Low</td><td>Low</td></tr><tr><td>SINDy</td><td>Sparse Re- gression</td><td>High</td><td>Medium</td><td>Medium</td></tr><tr><td>DSR</td><td>Neural RL</td><td>Medium</td><td>Medium</td><td>Medium</td></tr><tr><td>NSR (Ours)</td><td>Neural + Sparse</td><td>High</td><td>High</td><td>High</td></tr></table>

Conversely, the suggested Neural Symbolic Regression (NSR) framework employs a decoupled approach, utilizing neural networks solely as smooth function approximators, and then a distinct sparse regression phase for symbolic extraction. This distinction reinterprets neural networks as functional preconditioners for symbolic retrieval, increasing noise resilience, stabilizing equation detection, and boosting the identifiability of the fundamental symbolic framework.

Ultimately, contemporary symbolic regression processes increasingly depend on automated hyperparameter optimization to explore intricate model spaces. Methods like Bayesian optimization [18], random search [8], and efficient resource schedulers such as ASHA [7], as utilized in Ray Tune, allow for scalable and effective exploration of configurations for neural and sparse models. The Nguyen benchmark suite [9] continues to be a commonly used standard for assessing SR algorithms across various functional types.

## III. METHODOLOGY

The suggested Neural Symbolic Regression (NSR) framework combines the creation of a nonlinear feature library, neural function estimation, sparse symbolic extraction, and automated hyperparameter tuning. This part offers a formal definition of the problem and outlines each element thoroughly. For simplicity, we denote the complete nonlinear feature library, including interaction terms, as Φ(X) unless stated otherwise.

## A. Problem Formulation

Given a dataset

$$
\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } , \quad x _ { i } \in \mathbb { R } ^ { d } , ~ y _ { i } \in \mathbb { R } ,
$$

the goal of symbolic regression is to identify an analytical expression

$$
f ^ { * } ( x ) \in { \mathcal { F } } ,
$$

that minimizes prediction error while enforcing interpretability:

$$
f ^ { * } = \arg \operatorname* { m i n } _ { f \in \mathcal { F } } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( f ( x _ { i } ) - y _ { i } ) ^ { 2 } + \Omega ( f ) \right) ,
$$

where $\Omega ( f )$ denotes a complexity penalty. NSR approximates this optimization through a hybrid neural–sparse learning pipeline.

## B. Nonlinear Feature Library Construction

Let

$$
\mathcal { F } = \{ \mathrm { i d } , \ \sin , \ \cos , \ \exp , \ \log ( 1 + x ) , \ x ^ { 2 } , \ x ^ { 3 } , \ \operatorname { t a n h } \} ,
$$

represent the basis transformations. The feature matrix is constructed as:

$$
\Phi ( X ) = \left[ f _ { j } ( x _ { i } ) \right] _ { i = 1 \ldots n , \ j = 1 \ldots k } .
$$

To capture high-order interactions, NSR includes pairwise products:

$$
\Phi ^ { ( 2 ) } ( X ) = \{ f _ { p } ( x _ { a } ) f _ { q } ( x _ { b } ) \mid a \neq b , \ f _ { p } , f _ { q } \in { \mathcal { F } } \} .
$$

Thus, the full feature library becomes:

$$
\Phi _ { \mathrm { f u l l } } ( X ) = \Phi ( X ) \cup \Phi ^ { ( 2 ) } ( X ) ,
$$

with total dimensionality:

$$
| \Phi _ { \mathrm { f u l l } } | = O ( k d + k ^ { 2 } { \binom { d } { 2 } } ) .
$$

Parallel construction is implemented using Joblib for efficiency.

## C. Neural Approximation Network

A multilayer perceptron (MLP) parameterized by θ models:

$$
\hat { y } = f _ { \theta } ( \Phi _ { \mathrm { f u l l } } ( X ) ) .
$$

The MLP consists of:

• input dimension $| \Phi _ { \mathrm { f u l l } } | .$

• L hidden layers with width H,

• ReLU/GELU activations,

• dropout and weight decay regularization,

• AdamW optimizer.

The training objective is:

$$
\mathcal { L } _ { \mathrm { N N } } ( \theta ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( f _ { \theta } ( \Phi ( x _ { i } ) ) - y _ { i } \right) ^ { 2 } + \alpha \Vert \theta \Vert _ { 2 } ^ { 2 } .
$$

Training runs on GPU acceleration using CUDA or Apple MPS backend.

## D. Sparse Symbolic Extraction Using LASSO

Following neural training, symbolic structure is recovered via LASSO:

$$
\beta ^ { * } = \arg \operatorname* { m i n } _ { \beta } \left( \| \Phi _ { \mathrm { f u l l } } ( X ) \beta - y \| _ { 2 } ^ { 2 } + \lambda \| \beta \| _ { 1 } \right) .
$$

Nonzero coefficients indicate active symbolic terms. SymPy is used to reconstruct the analytical expression.

A classical compressed sensing result states:

$\beta ^ { * }$ can be recovered exactly if $\Phi _ { \mathrm { f u l l } }$ satisfies the RIP condition, motivating sparsity-based symbolic extraction.

## E. Hyperparameter Optimization with Ray Tune

Ray Tune is used to automate selection of neural and sparse model parameters. The search space is:

η ∼ LogUniform(10<sup>−4</sup>, 10<sup>−2</sup>),   
H ∈ {32, 64, 128}, L ∈ {1, 2, 3},   
$\lambda \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ , batch size ∈ {32, 64, 128}.   
Ray Tune minimizes the validation RMSE:   
θ<sup>∗</sup> = arg min RMSE<sub>val</sub>(θ).   
θ   
ASHA aggressively prunes poor-performing trials using:   
Promote trial if $m _ { i } < \gamma \cdot \mathrm { m e d i a n } ( m )$   
F. NSR Pipeline Overview

Algorithm 1 Neural Symbolic Regression (NSR)   
1: Input: Dataset $( X , y )$   
2: Construct $\Phi ( X )$ using nonlinear transformations   
3: Build interaction features $\Phi ^ { ( 2 ) } ( X )$   
4: Form the full feature matrix $\Phi _ { \mathrm { f u l l } }$   
5: Train neural network $f _ { \theta }$ on $\Phi _ { \mathrm { f u l l } }  y$   
6: Extract symbolic coefficients $\beta$ using LASSO   
7: Convert active coefficients into an analytic expression   
using SymPy   
8: Output: Symbolic function $f ^ { * } ( x )$

## IV. COMPUTATIONAL COMPLEXITY ANALYSIS

This section examines the computational and memory complexity of the suggested Neural Symbolic Regression (NSR) framework and contrasts it with traditional Genetic Programming (GP) and Sparse Identification of Nonlinear Dynamics (SINDy).

## A. Notation

Let N denote the number of samples, d the input dimensionality, F the number of base nonlinear functions in the feature library, and k the total number of expanded features after library construction. Let H denote the hidden layer width, L the number of hidden layers, and E the number of training epochs. For GP-based methods, let P denote the population size, G the number of generations, and S the average symbolic tree size.

## B. Feature Library Construction

The nonlinear feature library Φ(X) is constructed by applying F base functions to each input dimension and optionally including interaction terms up to a fixed order. The total number of features is given by:

$$
k = F d + { \binom { d } { 2 } } F ^ { 2 } ,\tag{1}
$$

for interactions of the second order. Feature construction involves assessing every feature across all N samples, leading to a time complexity of $\mathcal { O } ( N k )$ and a space complexity of $\mathcal { O } ( N k )$ . This phase signifies the main memory limitation of the framework.

TABLE II  
COMPUTATIONAL COMPLEXITY COMPARISON
<table><tr><td>Method</td><td>Time Complexity</td><td>Space Complexity</td></tr><tr><td>Genetic Programming (GP)</td><td>O(PGNS)</td><td>O(PS)</td></tr><tr><td>SINDy (LASSO)</td><td>O(Nk2)</td><td> $\mathcal { O } \dot { ( N k ) }$ </td></tr><tr><td>NeuralSR (Baseline)</td><td> $\mathcal { O } ( E N ( \dot { k } H + H ^ { 2 } ) )$ </td><td> $\mathcal { O } ( N \dot { k } + \dot { H } ^ { 2 } )$ </td></tr><tr><td>NeuralSR (Tuned)</td><td> $\mathcal { O } ( \dot { T } E N ( k H + H ^ { 2 } ) )$ </td><td> $\scriptstyle { \dot { \mathcal { O } } } ( T H ^ { 2 } )$ </td></tr></table>

## C. Neural Approximation

The neural approximation element uses a multilayer perceptron functioning within the enlarged feature space. The computational expense of each epoch includes the forward and backward passes:

$$
\mathcal { O } \left( N ( k H + ( L - 1 ) H ^ { 2 } ) \right) ,\tag{2}
$$

resulting in an overall training complexity of $\mathcal { O } \left( E N ( k H + ( L - 1 ) H ^ { 2 } ) \right)$ . The memory complexity is primarily influenced by model parameters $\mathcal { O } ( k H + ( L - 1 ) H ^ { 2 } )$ and intermediate activations $\mathcal { O } ( N L H )$ while training.

## D. Sparse Symbolic Extraction

Symbolic equation identification is achieved through LASSO-driven sparse regression on the developed feature library. Employing coordinate descent optimization, the time complexity scales to $\mathcal { O } ( N k ^ { 2 } )$ , and the memory complexity is $\mathcal { O } ( N k + k )$ . This process is performed a single time following neural training, eliminating the need for repetitive symbolic searching.

## E. Overall Complexity

Combining all components, the overall computational complexity of the proposed NSR framework is:

$$
\mathcal { O } \left( N k + E N ( k H + H ^ { 2 } ) + N k ^ { 2 } \right) ,\tag{3}
$$

while the overall space complexity is:

$$
\mathcal { O } \left( N k + N L H + k H \right) .\tag{4}
$$

GPU acceleration drastically decreases the actual wall-clock duration of the neural training part, which has a major impact on the total computation for medium feature library sizes.

## F. Comparison with GP and SINDy

Table II summarizes the computational complexity of NSR in comparison with GP and SINDy.

## G. Limitations and Scalability Considerations

The suggested Neural Symbolic Regression framework encounters scalability constraints mainly because of the creation of the feature library. The enlarged library increases combinatorially with the dimensionality of the input and the order of interaction, leading to higher memory consumption and presenting quadratic complexity during the sparse regression phase. Although GPU acceleration decreases the cost of neural training, the direct storage of $\Phi ( X )$ can become unfeasible for large-scale or high-dimensional datasets. Moreover, employing a fixed function library might lead to redundancy. Future efforts will explore adaptive library pruning, learned feature selection, and low-rank or streaming representations to enhance scalability while maintaining interpretability.

In contrast to GP-based methods that experience exponential tree expansion and limited scalability, the suggested NSR framework transfers the main computation to GPU-enhanced neural training and conducts sparse symbolic extraction just a single time. In contrast to SINDy, NSR enhances robustness and scalability by separating function approximation from symbolic extraction.

## V. EXPERIMENTAL SETUP

This section outlines the benchmarks, data generation methods, model structures, hyperparameter tuning approach, baselines, hardware utilized, and evaluation metrics applied in our research.

## A. Benchmarks and Data Generation

We assess the suggested NSR framework using the Nguyen-1 to Nguyen-7 symbolic regression benchmarks. Every benchmark defines a target function made up of polynomial, trigonometric, or transcendental elements. For every task, we create $n = 1 0 0 0$ input samples uniformly distributed in $[ - 1 , 1 ] ^ { d }$ , with d representing the benchmark’s dimensionality. The resulting outputs are calculated as:

$$
y _ { i } = f _ { \mathrm { t r u e } } ( x _ { i } ) + \epsilon _ { i } ,
$$

where $\epsilon _ { i } \sim \mathcal { N } ( 0 , 0 . 0 1 ^ { 2 } )$ is optional Gaussian noise. Datasets are split into 70% training, 15% validation, and 15% testing.

## B. Feature Library Specification

The nonlinear feature library consists of the function set:

$$
\mathcal { F } = \{ \mathrm { i d } , \ \sin , \ \cos , \ \exp , \ \log ( 1 + x ) , \ x ^ { 2 } , \ x ^ { 3 } , \ \operatorname { t a n h } \} .
$$

These are applied to each input dimension to form the primary library $\Phi ( X )$ . To increase expressiveness, second-order pairwise interaction terms are added:

$$
\Phi ^ { ( 2 ) } ( X ) = \{ f _ { p } ( x _ { a } ) f _ { q } ( x _ { b } ) \mid a \neq b , \ f _ { p } , f _ { q } \in { \mathcal { F } } \} .
$$

The full library is:

$$
\Phi _ { \mathrm { f u l l } } = \Phi ( X ) \cup \Phi ^ { ( 2 ) } ( X ) .
$$

All library construction steps are parallelized using Joblib.

## C. Neural Model Architecture

The neural approximator is a multilayer perceptron (MLP) of depth $L \in \{ 1 , 2 , 3 \}$ and width $H \in \{ 3 2 , 6 4 , 1 2 8 \}$ , chosen via hyperparameter optimization. Every layer employs GELU or ReLU activation, succeeded by dropout and weight decay regularization. The network is taught to minimize:

$$
\mathcal { L } _ { \mathrm { N N } } ( \theta ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( f _ { \theta } ( \Phi _ { \mathrm { f u l l } } ( x _ { i } ) ) - y _ { i } \right) ^ { 2 } + \alpha \Vert \theta \Vert _ { 2 } ^ { 2 } .
$$

TABLE III  
AVERAGE RUNTIME COMPARISON OF SYMBOLIC REGRESSION METHODS
<table><tr><td>Method</td><td>Avg Runtime (s)</td></tr><tr><td>SINDy</td><td>1.2</td></tr><tr><td>NSR Baseline</td><td>18.4</td></tr><tr><td>PySR</td><td>24.6</td></tr><tr><td>NSR Tuned</td><td>42.7</td></tr></table>

Training uses AdamW with learning rates sampled from LogUniform $\cdot 1 0 ^ { - 4 } , 1 0 ^ { - 2 } )$ and batch sizes {32, 64, 128}.

To assess the computational burden of the suggested framework, we analyze the average execution time of NSR in comparison to traditional SINDy-based symbolic regression. Runtime assessments encompass feature creation, model training, and symbolic extraction phases. Despite the increased computational expense of NSR from neural approximation and hyperparameter optimization, GPU acceleration allows for feasible training durations while enhancing robustness and symbolic recovery performance.

## D. Hyperparameter Optimization (Ray Tune)

Ray Tune is utilized to automatically determine the best configurations for neural and sparse models. We utilize the ASHA scheduler, which advances trials based on:

$$
m _ { i } < \gamma \cdot \mathrm { m e d i a n } ( m ) ,
$$

where $m _ { i }$ is the validation RMSE. A total of 20 trials are executed in parallel, each exploring different settings of learning rate, depth, width, batch size, and LASSO regularization $\lambda \overset { - } { \in } \{ 1 0 ^ { - 4 } , \overset { - } { 1 0 ^ { - 3 } } , 1 0 ^ { - 2 } \}$

## E. Baselines

Two baselines are used:

• SINDy: Linear sparse regression over the same feature library.

• Non-tuned Neural Baseline: NSR without hyperparameter optimization.

Comparisons are made with respect to RMSE, symbolic accuracy, sparsity, and OOD generalization.

## F. Hardware and Software Environment

Tests were conducted on Apple Silicon systems utilizing the MPS backend, as well as on cloud GPU instances with NVIDIA T4 and A100 GPUs. Training employs PyTorch 2.0, scikit-learn 1.3, Ray Tune 2.x, SymPy, Joblib, and Python 3.10.

Runtime values were averaged over five independent runs on identical hardware configurations.

## G. Evaluation Metrics

We report:

• RMSE: Root Mean Squared Error on the test set.

• Symbolic Accuracy: Closeness of the extracted equation to ground truth.

TABLE IV  
RMSE COMPARISON ACROSS NGUYEN BENCHMARKS. LOWER IS BETTER. RESULTS ARE AVERAGED OVER MULTIPLE RUNS.
<table><tr><td>Method</td><td>Nguyen-1</td><td>Nguyen-2</td><td>Nguyen-3</td><td>Nguyen-4</td></tr><tr><td>PySR</td><td>0.2111</td><td>0.0802</td><td>0.0798</td><td>0.0039</td></tr><tr><td>SINDy</td><td>0.0423</td><td>0.0425</td><td>0.0433</td><td>0.0463</td></tr><tr><td>NSR (Tuned)</td><td>0.0087</td><td>0.0099</td><td>0.0122</td><td>0.0205</td></tr></table>

• Expression Complexity: Number of active terms selected by LASSO.

• Noise Robustness: Performance under increasing σ.

• OOD Generalization: Performance on samples drawn outside training range.

## VI. RESULTS

This part showcases the empirical results of the suggested Neural Symbolic Regression (NSR) framework using the Nguyen benchmark suite. We present prediction accuracy, quality of symbolic recovery, training performance, resilience to noise, and generalization beyond the training distribution.

## A. Overall Performance

Table IV consolidates the Root Mean Squared Error (RMSE) achieved by PySR, SINDy, and the proposed optimized NSR framework over various Nguyen benchmarks. The suggested NSR approach attains the minimum RMSE on Nguyen-1, Nguyen-2, and Nguyen-3, showcasing the efficacy of neural smoothing, nonlinear feature expansion, and hyperparameter tuning. While PySR excels on Nguyen-4, its results are not as reliable across different polynomial benchmarks. In general, the findings suggest that the proposed NSR framework enhances robustness and maintains consistent predictive accuracy across various symbolic regression challenges.

PySR excels on Nguyen-4 because of its evolutionary exploration of compact trigonometric forms. Nonetheless, its performance shows increased instability in noisy polynomial benchmarks, while NSR exhibits more reliable predictive behavior across tasks.

## B. Training Dynamics

Fig. 1 illustrates the training loss curves for the baseline NSR and the optimized NSR. The tuned NSR achieves quicker convergence, attains a lower minimum loss, and exhibits more consistent optimization behavior. These enhancements are ascribed to hyperparameter selection driven by ASHA.

![](images/e5afbe29113c83bde71ba0e0ad089a8109930d117a554171e22bf4e49f41d0de.jpg)  
Fig. 1. Training loss curves for baseline NSR and tuned NSR.

## C. Symbolic Equation Recovery

A primary feature of NSR is its ability to retrieve closedform symbolic expressions. Fig. 2 shows a typical example from the Nguyen-3 benchmark.

Ground Truth:   
f(x) = x<sup>5</sup> + x<sup>4</sup> + x<sup>3</sup> + x<sup>2</sup> + x   
Extracted (NSR):   
<sup>ˆ</sup>f(x) = 1.002 x<sup>5</sup> + 0.998 x<sup>4</sup> + 1.001 x<sup>3</sup> + 1.000 x<sup>2</sup> + 0.999 x  
Fig. 2. Example symbolic regression result for Nguyen-3.

The extracted equations align closely with the actual structure, showing only minor numerical differences, thereby validating the interpretability attained through LASSO-based extraction.

## D. Prediction Quality

We evaluate predictive accuracy by utilizing scatter plots comparing predicted and actual values. Fig. 3 illustrates an instance from Nguyen-5.

![](images/bd66c67adeaf7267f0c6e932ec1ebe2f66dd3bb978627574c1c2e2deec6941ab.jpg)  
Fig. 3. Predicted vs. true outputs for the Nguyen-5 benchmark.

Points align closely along the diagonal, indicating highaccuracy functional approximation.

## E. Noise Robustness

To assess the stability of the model with noisy observations, we incorporate Gaussian noise levels σ ∈ {0, 0.01, 0.05, 0.1}. Fig. 4 shows RMSE as noise levels rise.

![](images/3ef8b616c1d7ef2cf323a61659707a6925ba0db093b448ad3f1be140644816e1.jpg)  
Fig. 4. RMSE under increasing noise levels.

![](images/c668a4316b18cdacc22c0cd423d29d92e464023973d67dd1b0914027886c728e.jpg)  
Fig. 5. RMSE of different models

Tuned NSR consistently surpasses SINDy and demonstrates more gradual degradation, underscoring the advantages of neural smoothing prior to sparse extraction.

## F. Out-of-Distribution Generalization

OOD generalization is assessed by training on [−1, 1] and testing on [−2, 2]. Fig. 6 shows the OOD performance for Nguyen-4.

![](images/8541bcf7a7544b20cbaeec7cf2d2e804ab7f9db5f7e6bf676e3f17cd3e63effe.jpg)  
Fig. 6. Out-of-distribution prediction quality for Nguyen-4. Train domain: [-1,1], Test domain: [-2,2]

NSR demonstrates considerably superior predictive consistency beyond the training interval, while SINDy shows marked divergence, reinforcing the advantages of the neural approximator in terms of generalization.

## VII. ABLATION STUDY

To evaluate the impact of each element within the NSR framework, we perform a thorough ablation study. Every element—feature library depth, interaction variables, neural network scale, and hyperparameter tuning—is intentionally omitted or streamlined to evaluate its impact on prediction precision and symbolic retrieval.

## A. Effect of Feature Library Size

We assess the complete nonlinear library Φ in comparison to simplified versions that include solely polynomials or solely trigonometric functions. Eliminating nonlinear transformations greatly elevates RMSE and diminishes symbolic fidelity, reinforcing the necessity of a varied library for representing intricate functional structures.

## B. Effect of Interaction Terms

We evaluate models that have been trained with and without the second-order interaction features $\Phi ^ { ( 2 ) } ( X )$ . In the absence of interactions, extracted symbolic expressions often disregard cross-dimensional connections, resulting in increased approximation error—particularly in multivariate benchmarks (e.g., Nguyen-6, Nguyen-7).

## C. Effect of Neural Architecture Depth

To examine the function of the neural approximator, we evaluate shallow $( L \ = \ 1 )$ , medium $( L \ = \ 2 )$ , and deeper $( L = 3 )$ networks. Shallow models struggle to capture intricate nonlinearities, leading to inadequate symbolic extraction. Expanding depth beyond three layers results in diminishing returns and could raise variance.

## D. Effect of Hyperparameter Optimization

We evaluate the adjusted NSR in relation to the untuned baseline. Ray Tune greatly lowers validation RMSE and enhances symbolic accuracy by determining the best learning rates, batch sizes, depths, widths, and regularization strengths. The adjusted model reaches convergence more quickly and reliably discovers simpler, more accurate symbolic expressions.

## E. Summary of Ablation Results

Table V summarizes quantitative results for all ablation configurations.

TABLE V  
ABLATION STUDY RESULTS
<table><tr><td>Configuration</td><td>RMSE</td><td>Terms Selected</td></tr><tr><td>Full NSR (tuned)</td><td>0.004</td><td>5</td></tr><tr><td>No Interaction Terms</td><td>0.037</td><td>3</td></tr><tr><td>Reduced Feature Library</td><td>0.142</td><td>2</td></tr><tr><td>Shallow Network  $( L = 1 )$ </td><td>0.081</td><td>4</td></tr><tr><td>No Hyperparameter Tuning</td><td>0.019</td><td>7</td></tr></table>

The ablation findings confirm that every element of NSR—feature expansion, interaction modeling, neural approximation depth, and hyperparameter tuning—is essential. Eliminating any component results in reduced accuracy or less understandable expressions.

## VIII. DISCUSSION

The experimental findings indicate that the suggested NSR framework successfully combines neural approximation, nonlinear feature expansion, and sparse symbolic extraction to produce precise and interpretable models. This part examines crucial observations, insights, and constraints of the method.

## A. Effectiveness of Neural–Symbolic Integration

A key discovery is the distinct benefit of integrating neural networks with sparse modeling. Although SINDy excels with low-complexity polynomials, it faces challenges with benchmarks containing significant nonlinearities or transcendental elements. In contrast, NSR leverages the expressive capability of neural networks, allowing it to model intricate functions prior to applying symbolic sparsity via LASSO. This structured approach results in more consistent and accurate symbolic recovery, as demonstrated in the Nguyen-3 and Nguyen-5 benchmarks.

## B. Impact of Hyperparameter Optimization

Hyperparameter optimization using Ray Tune significantly enhances model performance. In the absence of tuning, the baseline NSR typically exhibits slow convergence or gets stuck in areas with high loss. The ASHA scheduler promotes effective exploration by ending ineffective trials early, resulting in improved learning rates, depth–width settings, and batch size selections. The optimized model consistently surpasses both SINDy and untuned NSR in almost all benchmarks, as indicated by the reductions in RMSE and enhanced symbolic coefficients.

## C. Symbolic Interpretability and Stability

The expressions obtained through LASSO closely align with the actual values in terms of both structure and coefficients. Even with the introduction of noise, the neural approximator smooths the fundamental structure, allowing LASSO to detect the correct symbolic elements. This stability presents a significant benefit compared to direct symbolic regression techniques like genetic programming, which tend to be more affected by noise and initial conditions.

## D. Generalization and Extrapolation

NSR demonstrates better performance in out-of-distribution (OOD) scenarios than SINDy. Due to the neural model learning a continuous latent mapping across an extensive feature library, the identified expression extends beyond the training domain. This is a crucial attribute for scientific discovery activities, where extrapolation is frequently necessary. Nonetheless, we noticed that very deep networks sometimes overfit, highlighting the importance of balanced design decisions.

## E. Limitations

In spite of its advantages, NSR faces various shortcomings:

• Library Expansion: Adding interaction terms may lead to a combinatorial increase in the feature library, elevating computational expenses.

• Reliance on Smoothness: The neural approximator presumes that the target function being approximated is smooth. Functions that are highly discontinuous or piecewise may be difficult to model accurately.

• LASSO Bias: LASSO often reduces coefficients, which may slightly alter symbolic representations, particularly when the features are interrelated.

• Training Overhead: Neural training results in increased runtime when compared to purely sparse methods. Nevertheless, adjustment and parallel processing alleviate this in reality.

## F. Broader Implications

The findings indicate that hybrid neural-symbolic techniques represent a promising avenue for interpretable machine learning. In contrast to black-box models, NSR produces clear mathematical formulations, facilitating use in system identification, discovering physics, and optimizing engineering processes. The integration of GPU acceleration, symbolic extraction, and automated hyperparameter tuning creates new opportunities for applying symbolic regression in extensive scientific tasks.

In general, the NSR framework achieves a balance among accuracy, interpretability, and computational practicality. Future research can tackle the recognized limitations by implementing dynamic feature pruning, enhancing library learning adaptively, and utilizing transformer-based symbolic decoding.

While the existing assessment emphasizes low-dimensional symbolic benchmarks, the suggested framework is not limited to scalar inputs. The stages of interaction-aware feature construction and neural approximation naturally extend to multivariate vector-valued data, facilitating application to more complex scientific and engineering systems.

## IX. CONCLUSION

This study introduced a cohesive Neural Symbolic Regression (NSR) framework that integrates nonlinear feature library creation, neural network modeling, and sparse symbolic extraction to uncover interpretable mathematical formulas from data. Through the combination of expressive neural models and LASSO-driven sparsity, NSR successfully connects blackbox prediction with symbolic interpretability.

Thorough experiments conducted on the Nguyen benchmark suite show that NSR reliably exceeds the performance of both traditional SINDy and unoptimized neural baselines over a diverse set of nonlinear functions. The approach attains reduced RMSE, enhanced symbolic recovery accuracy, and greater resilience to noise. Hyperparameter optimization using Ray Tune is essential for stabilizing training and choosing architectures that generalize effectively both inside and outside the training domain.

The symbolic equations obtained align closely with their analytical ground truth, validating the effectiveness of LASSO for post-hoc symbolic decoding. Additionally, NSR exhibits robust performance on out-of-distribution data, suggesting that the acquired representations reflect the fundamental functional framework instead of simply recalling training examples.

In spite of these advantages, NSR encounters difficulties in high-dimensional contexts because of the swift increase in feature libraries and possible overlap among interaction terms. Neural training adds computational complexity in comparison to solely sparse techniques. Tackling these constraints while ensuring interpretability is a crucial area for future investigation.

In summary, the findings demonstrate that neural-symbolic hybrid approaches provide a robust framework for scientific machine learning. Utilizing both symbolic structure and approximation capability, NSR offers a scalable and interpretable method for uncovering closed-form expressions, which may be applicable in physics discovery, system identification, engineering modeling, and automated scientific reasoning.

## X. FUTURE WORK

Although the suggested NSR framework shows impressive results on various symbolic regression tasks, there are still many promising avenues for future research.

## A. Adaptive and Learned Feature Libraries

At present, the feature library is established with a predetermined collection of nonlinear functions. Future research could investigate adaptive library learning, in which the model autonomously builds or eliminates functions according to their relevance. Methods like neural-guided basis selection, evolutionary library creation, or dictionary learning may greatly decrease the size of the library and the associated computational expenses.

## B. Transformer-Based Symbolic Decoding

The symbolic extraction process depends on LASSO using a predetermined feature set. Extensive language models and transformer-driven symbolic decoders have demonstrated significant ability to create syntactically correct equations. Combining NSR with transformer-guided symbolic reasoning may allow for more extensive and versatile discoveries beyond just linear combinations of basis functions.

## C. Scaling to High-Dimensional Systems

The exponential increase of interaction terms restricts NSR in high dimensions. Prospective efforts might include:

• identification of sparse interactions,

• construction of features with a hierarchical or low-rank approach,

• variable selection based on attention.

These methods may render NSR appropriate for multivariate scientific systems like PDEs, biological networks, and control systems.

## D. Improved Optimization and Training Stability

Although Ray Tune greatly enhances convergence, symbolic regression continues to be affected by neural training dynamics. Future studies could investigate:

• educational progression for symbolic activities,

• meta-learning for transferring hyperparameters between benchmarks,

• loss functions designed for symbolic sparsity.

## E. Integration With Scientific Simulators

A significant opportunity exists in combining NSR with specialized simulators like physics engines or numerical solvers. This would allow for the identification of governing equations directly from trajectory data, connecting data-driven learning with mechanistic modeling.

## F. Real-World Applications

Ultimately, applying NSR to actual noisy datasets—like climate information, biological signals, and sensor data from engineering—would confirm its effectiveness beyond synthetic benchmarks. Utilizing the approach for extensive scientific problems could uncover novel physical understandings and speed up automated exploration.

In general, upcoming efforts focus on improving scalability, adaptability, and scientific relevance, advancing NSR from regulated benchmarks to effective implementation in intricate real-world systems.

## REFERENCES

[1] J. Koza, “Genetic Programming,” MIT Press, 1994.

[2] S. Brunton et al., “Sparse Identification of Nonlinear Dynamics,” PNAS, 2016.

[3] B. Petersen et al., “Deep Symbolic Regression,” arXiv:1912.04871, 2019.

[4] L. Biggio et al., “Neural Symbolic Regression that Scales,” arXiv:2106.06427, 2021.

[5] R. Tibshirani, “Regression Shrinkage and Selection via the LASSO,” Journal of the Royal Statistical Society, 1996.

[6] K. Hornik, “Multilayer Feedforward Networks are Universal Approximators,” Neural Networks, 1989.

[7] L. Li et al., “A System for Massively Parallel Hyperparameter Tuning,” MLSys, 2020.

[8] J. Bergstra and Y. Bengio, “Random Search for Hyper-Parameter Optimization,” JMLR, 2012.

[9] Q. Nguyen et al., “Benchmark Problems for Genetic Programming-Based Symbolic Regression,” Genetic Programming Theory and Practice.

[10] M. Schmidt and H. Lipson, “Distilling Free-Form Natural Laws from Experimental Data,” Science, 2009.

[11] E. Vladislavleva et al., “Order of Nonlinearity as a Complexity Measure for Symbolic Regression,” IEEE Transactions on Evolutionary Computation, 2009.

[12] D. Donoho, “Compressed Sensing,” IEEE Transactions on Information Theory, 2006.

[13] T. Hastie, R. Tibshirani, J. Friedman, “The Elements of Statistical Learning,” Springer, 2009.

[14] G. Cybenko, “Approximation by Superpositions of Sigmoidal Functions,” Mathematics of Control, Signals, and Systems, 1989.

[15] A. Barron, “Universal Approximation Bounds for Superposition of a Sigmoidal Function,” IEEE Transactions on Information Theory, 1993.

[16] R. Udrescu and M. Tegmark, “AI Feynman: A Physics-Inspired Method for Symbolic Regression,” Science Advances, 2020.

[17] S. Kim et al., “Learning Governing Equations from Data: Sparse vs Neural Approaches,” arXiv:2004.02394, 2020.

[18] J. Snoek, H. Larochelle, and R. Adams, “Practical Bayesian Optimization of Machine Learning Algorithms,” NeurIPS, 2012.

[19] W. La Cava et al., “Contemporary Symbolic Regression Methods and Benchmarks,” arXiv:2107.14351, 2021.

## APPENDIX A

## FUNDAMENTAL IMPLEMENTATION ASPECTS

This appendix provides essential code snippets from the Neural Symbolic Regression (NSR) framework. Only the most pertinent components are displayed due to limitations in space. The full implementation, which encompasses additional utilities, ablation scripts, and plotting code, can be found in the project repository.

## A. Construction of Nonlinear Feature Library

The feature library translates input data into a nonlinear framework made up of basic functions and interaction components. Secure numerical processing is implemented to avoid NaN or infinite values.

```python
Listing 1. Feature library with safe nonlinear transformations
class FeatureLibrary:
def __init__(self, config):
self.cfg = config
def transform(self, X):
if X.ndim == 1:
X = X[:, None]
n_samples, n_features = X.shape
terms, names = [], []
for fname in self.cfg.funcs:
f = SAFE_FUNC_REGISTRY[fname]
for i in range(n_features):
val = f(X[:, i])
val = np.nan_to_num(val, nan=0.0,
posinf=1e6, neginf=-1e6)
terms.append(val)
names.append(f"{fname}(x{i})")
Phi = np.stack(terms, axis=1)
return Phi, names
```

## B. Neural Approximation Model

A multilayer perceptron (MLP) is used to learn a smooth approximation over the nonlinear feature space.

```python
Listing 2. Neural network model for symbolic regression
class MLP(nn.Module):
def __init__(self, in_dim, hidden, depth):
super().__init__()
layers = [nn.Linear(in_dim, hidden),
nn.GELU()]
for _ in range(depth - 1):
layers += [nn.Linear(hidden, hidden),
nn.GELU()]
layers.append(nn.Linear(hidden, 1))
self.net = nn.Sequential(<sub>*</sub>layers)
def forward(self, x):
return self.net(x).squeeze(-1)
```

## C. Training Procedure with Early Stopping

The training loop includes early stopping, gradient clipping, and GPU acceleration.

```python
Model training rly stopping
for epoch in range(max_epochs):
optimizer.zero_grad()
prediction = model(Phi_batch)
loss = mse_loss(prediction, y_batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters()
1.0)
optimizer.step()
if val_loss < best_loss:
best_loss = val_loss
patience_counter = 0
else:
patience_counter += 1
if patience_counter > patience:
break
```

## D. Sparse Symbolic Equation Extraction

Symbolic expressions are recovered using LASSO regression, promoting sparsity in the coefficient vector.

```python
Listing 4. Sparse symbolic regression using LASSO
lasso = Lasso(alpha=1e-3)
lasso.fit(Phi, y)
coefficients = lasso.coef_
expression = 0
for coef, name in zip(coefficients, feature_names):
if abs(coef) > 1e-6:
expression += coef <sub>*</sub> sympy_term(name)
```

## E. Hyperparameter Optimization using Ray Tune

Distributed hyperparameter tuning is performed using Ray Tune with early stopping.

Listing 5. Ray Tune hyperparameter search configuration   
search\_space = {   
"hidden": tune.choice([64, 128, 256]),   
"depth": tune.choice([1, 2, 3]),   
"lr": tune.loguniform(1e-4, 1e-2),   
"batch\_size": tune.choice([64, 128])   
}   
tuner = tune.Tuner(   
trainable,   
param\_space=search\_space,   
tune\_config=tune.TuneConfig(   
metric="rmse",   
mode="min"   
)   
)

## F. Noise Robustness and OOD Evaluation

The system is evaluated under increasing noise levels and out-of-distribution (OOD) domains.

```python
for noise in noise_levels:
y_noisy = y + np.random.normal(0, noise,
size=y.shape)
model.fit(X, y_noisy)
rmse = compute_rmse(model.predict(X_val), y_val)
results.append((noise, rmse))
```

## APPENDIX B REPRODUCIBILITY

All experiments were executed using fixed random seeds. The codebase supports CPU, CUDA-enabled GPUs, and Apple MPS devices. Full source code, configuration files, and scripts for reproducing all results are provided in the public repository.