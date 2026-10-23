# PYTHIA, a transferable machine learning framework for simultaneous defect and adsorption site optimization

Ioannis Kouroudis<sup>1,2∗</sup> Zhaosu Gu<sup>1,2</sup> Poonam<sup>2</sup> Leon M¨uller<sup>3</sup>

Prof. Dr. rer. nat. Alessio Gagliardi<sup>2†</sup>

<sup>1</sup> These authors contributed equally

Chair of Simulation of Nanosystems for Energy Conversion, Department of Electrical Engineering, TUM School of Computation, Information and Technology, Atomistic Modeling Center (AMC), Munich Data Science Institute (MDSI), Technical University of Munich, Hans-Piloty-Straße 1, 85748 Garching, Germany

Fritz-Haber-Institut der Max-Planck-Gesellschaft, Faradayweg 4-6, Berlin, Germany

## Abstract

Machine learning has permeated material optimization, providing high value. Nevertheless, its reliance on a high number of samples limits its applicability to relatively explored material families. In this paper we extend the applicability of machine learning to novel and understudied materials through the PYTHIA (Probabilistic Yet Transferable Hierarchical Interaction Architecture) framework. Pythia utilizes two of the most powerful machine learning concepts, foundational models and transfer learning. We employ a complex foundational model trained on a large dataset of chemically similar molecules, on a target that is physically close to our prediction target. Then, we freeze the weights and perform subspace alignment with the use of a transformer layer and add uncertainty quantification with gaussian processes. These two layers can be trained on a limited dataset yet allow us to leverage the gained knowledge and the studied materials and complex architectures. The resulting model is used as a surrogate for a Bayesian Optimization scheme. To showcase the robustness of the framework, we applied it to the investigation of adsorption and defect optimization of two distinct systems. Not only were we able to correctly and eficiently identify the optimal adsorption sites of even complex adsorbates, the algorithm showed remarkable accuracy even for systems not present in its training set.

## 1 Introduction

Adsorption plays an important role in the interaction of nanostructured materials with molecules and ions. These interactions depend on the types of atoms present and their local arrangement. Identifying stable adsorption structures also helps explain electrochemical reactions at metal surfaces [1]. For materials with diferent compositions or defects, this search becomes more dificult. Changes in the local atomic environment can afect both the preferred adsorption position and the strength of binding. To investigate this, Density functional theory (DFT) is widely used. Mapping the potential energy surface (PES) requires calculations for many molecular positions and orientations [2], a search that can be automated by various tools which help generate, screen, and relax these structures [3, 4]. Other methods examine diferent parts of the energy landscape. Minima hopping and evolutionary algorithms search for structures with low energies [5, 6]. The nudged elastic band (NEB) method finds minimum energy paths between two energy states [7], while metadynamics explores free energy landscapes [8, 9]. These methods provide useful information, but their computational cost can remain high when each energy evaluation requires a DFT calculation. The cost increases further when several substrate compositions and defect structures are studied.

Machine learning can reduce the number of expensive calculations. It has been applied to many problems in materials research [10, 11, 12]. In atomistic simulations, machine learning models can describe surface reaction PESs [13] and predict adsorption energies on nanoclusters [14]. Graph neural networks have also been used to predict molecular adsorption energies on metals [15], and learned interatomic potentials can speed up adsorption searches [16]. However, these models need suitable training data. For a new adsorption system, generating enough DFT data can account for much of the total computational cost, often times providing a full system investigation before the model is even trained.

Active learning (AL) and Bayesian optimization (BO) help address this problem by using the available data to choose new calculations. Bayesian models can learn atomic interactions during simulations [17, 18], and uncertainty estimation is also an active area of research for molecular models [19]. Related approaches have been used for crystal structure prediction [20], the optimization of nanoporous materials [21], materials discovery with graph models [22], and catalyst screening through adsorption calculations [23]. Machine learning can also accelerate searches for saddle points and reaction paths [24, 25], as well as stable adsorbate geometries [26].

The choice of structural representation afects how well these methods work. BO has been used to search for molecular conformers by varying torsion angles [27]. The Bayesian Optimization Structure Search (BOSS) framework uses structural coordinates to explore energy landscapes [28]. Its applications include molecular adsorption on surfaces [29] and the search for molecular conformations on metal clusters [30]. These coordinates are useful when the main degrees of freedom can be identified in advance. However, a representation designed for one system may need to be changed when the molecular composition, substrate, or defects change. Learning a representation directly from atomic structures ofers a way to handle these diferences.

In our previous work, we developed AUGUR to identify adsorption sites by combining a graph neural network, Gaussian process (GP) regression, and BO [31]. AUGUR learns a structural representation through a graph neural network (GNN) and uses it to predict energies and estimate uncertainty via a GP, without requiring manually designed molecular descriptors. A very relevant question however is, how this approach can benefit from knowledge learned on other atomic systems. Such knowledge could reduce the amount of data needed for a new adsorption problem. This is particularly useful when the search includes severa compositions or defect types, each with only a limited number of reference calculations.

Pretrained atomic models provide a starting point for this transfer. The MACE architecture [32] has been used to develop foundation models trained on many diferent structures [33]. The atomic features learned by these models can then be used in new applications. However, pretraining does not guarantee accurate adsorption energies for every system, especially when its chemical environments or reference calculations difer from those used in training. Previous work has shown that adapting a pretrained potential while keeping some parameters fixed can reduce the amount of training data required [34] and even extrapolate to new properties [35]. For adsorption searches, we also need a model that adapts to the available data and estimates uncertainty to guide the choice of new calculations.

To address the above concerns we introduce PYTHIA, which combines a pretrained MACE model, a Transformer, GP regression, and BO. The pretrained MACE provides atomic features that contain relevant descriptors for an adjacent system and property. A Transformer [36] processes these features and produces a representation that has translated the features to the current system. This vector is then used as input for the GP that predicts the adsorption energy and its corresponding uncertainty. During training, the MACE parameters remain fixed, while the Transformer and GP are trained on data from the systems being studied. This allows the model to reuse features learned during pretraining and adapt them to the target energy prediction problem. BO then uses these predictions to select candidate defects and adsorption positions for further electronic structure calculations. This pipeline provides all the benefits of large and complex models, without the usual necessity of large datasets.

We verify PYTHIA on two highly relevant cases, optimizing both the defects of the cluster and the position of the adsorbate simultaneously. The first is the Zn adsorption on a defective Chini cluster. This system is of high interest due to its role in the chemistry of metal carbonyl clusters [37] and their encapsulation properties. The second examines adsorption of formaldehyde on a buckled silicon layer [38, 39]. Formaldehyde is a harmful chemical and silicene is a periodic and complex structure. Therefore the optimization of this system provides both a high environmental impact as well as a heightened complexity. These systems provide diferent local chemical environments for deep testing of the model. We compare PYTHIA with two baseline models, AUGUR and a fully trained MACE with a GP as a final layer and was found to outperform both by a significant margin showing that it has higher complexity than AUGUR and lower data demand then the full MACE. We also examine the predictions for defect groups not present in the training data and found our model highly accurate. To summarise, we show that PYTHIA combines the high accuracy of complex models at a fraction of the data cost. Additional to the predictions, PYTHIA also provides highly tailored optimization, identifying sites with many times better adsorption energy than present in the training set. With novel materials emerging fast, this data-eficient transferable framework can significantly expedite chemical investigation alleviating the need for time consuming data acquisition.

## 2 Methods

## 2.1 PYTHIA pipeline

![](images/12f8ef8d9a88a854f96aada67d05a5f2373317355a920cebf5c20feeda4bc293.jpg)  
Figure 1: Pythia Pipeline. The top row corresponds to the predictive part of the framework while the bottom is the permutation of defects and placements, along with the optimization. Both together consitute the tota optimization process.

In Fig. 1 the entire PYTHIA pipeline is shown for adsorption site optimization. It must be noted that the pipeline provides a prediction for the overall interaction energy surface and its corresponding uncertainty. The combination of those will allow the user to decide on an individual basis whether the added computational cost of another DFT round ofsets the potential uncertainty decrease of the predictions.

## 2.2 Bayesian Optimization

Bayesian Optimization (BO) is an optimization algorithm that has proven itself to be both data eficient and accurate even in non-convex/concave optimization problems. It is especially useful in optimization problems where there is a costly underlying evaluation function, such as time consuming experiments or simulations. Data-driven models substitute these procedures with a comparatively inexpensive surrogate, that, however, requires data to be trained. Oftentimes, this data demand is high enough as to invalidate the eventual benefits. Therefore an eficient and data-sparse dataset is necessary to leverage the full advantage of machine learning. Bayesian optimization gives answers to this bottleneck, simultaneously optimizing both the underlying problem as well as the sampling one. This is done by combining two components, a stochastic predictor, typically a GP regressor and a function that weighs each hopeful optimization point based on its optimality and amount of new information it brings into the model. This function is known as the acquisition function and can take many forms depending on the individual case needs. For the remainder of the chapter, we will describe a minimization process, but everything can be applied without loss of generality to any optimization problem. In the present work, we seek to find the position of adsorption and set defects that will result in the lowest possible energy. To this end, we can generate a number of random positions and evaluate their respective energies using the uncertainty-aware predictor. The prediction’s uncertainty is reflected in its standard deviation. Initially, we choose to evaluate points predicted to have high standard deviation, regardless of their predicted energy. In this way, we are giving the model information about compositions and sites far removed from the ones already in the training set. Once this process has suficiently trained the model, the focus can shift on determining the systems that exhibit good adsorption energies, thus fulfilling our original optimization goal. These stages are called exploration and exploitation respectively. Note that there are many ways to optimize the acquisition functions. The high number of physical constraints however made most of them not trivially applicable. To this end, we have chosen to generate 10.000 individual systems and predict them. Though this is slower than traditional gradient based approaches, it allows for the strict control of the evaluated system’s physical feasibility. Further, PYTHIA is eficient enough for the evaluation to still be performed in negligible time frames. More information about how the points were generated can be found in Section 2.3.

## 2.3 Point generation

The convergence of Bayesian Optimization can be significantly sped up if we introduce a layer of physical understanding of the problem. The simplest way is to generate a number of points (in our case, 10.000) that adhere to a set of constraints. A random number of defects and a corresponding number of random atoms of the cluster is chosen. These atoms are permuted, each by a unique, randomly selected element from the valid pre-defined defect elements. This has created our defective adsorbant. Subsequently, one random atom of the adsorbant and one random atom of the adsorbate is chosen. The chosen adsorbate atom is placed on the surface of a sphere with a radius determined by the physical limits of the interaction distance between the two atom types and centered on the chosen adsorbant atom. A random rotation matrix is defined and the adsorbate is rotated thus. The resulting adsorbate is tested to ensure that none of its atoms pass through or lie in the adsorbant. This is done by performing a Delaunay tessellation on the cluster coordinates and determining the location of the adsorbent with respect to the resultant convex hull coordinate comparison. If this constraint is violated, the system is discarded and the process restarts. If the system passes this check, the distance between the adsorbate and remaining adsorbant atoms is calculated. If through the rotations any atom is placed closer than physically possibly the system is discarded. Otherwise, the new system is added to the sample set and evaluated through the pipeline. The sample generation process can be summarized as follows:

1. Choose a random number of cluster atoms and exchange them for a set of random elements that are capable of producing valid, defective structures.

2. Choose a random atom of the cluster, provided adsorption is physically possible on it.

3. Place a random adsorbent on the surface of a sphere with a radius determined by physical knowledge of the maximum and minimum interaction distance of these two atoms. The functionality to determine a specific adsorbent atom for placing on the sphere is also provided.

4. Rotate the adsorbent around its center of mass by a random set of angles.

5. Test if the adsorbent either lies in or passes through the cluster.

6. If the above constraint is not violated, test if the adsorbent is closer than a distance of tolerance to any cluster atoms.

7. If the above constraint is not violated, the sample is admitted into the list of viable samples.

## 2.4 Gaussian Processes

In the current project, the uncertainty quantifying model chosen was Gaussian processes as it combines robustness to overfitting and the well-documented accuracy of global kernel methods. It is a data-driven stochastic algorithm that models predictions as the posterior of the Bayes formula. The prior and the likelihood are modeled as Gaussian distributions whose parameters are optimized based on the already measured samples. In this way, the algorithm elegantly provides an inherent uncertainty quantification, which in data sets of small size can be crucial. The fundamental principle of the method is that the true underlying physical process y can be modeled by a model f, tuned on a set of points X using a multivariate Gaussian distribution. This is the prior distribution, and without loss of generality, we can assume it is centered around zero. This assumption can be made more robust by standardizing the data so that their targets also fall within a zero mean Gaussian distribution. The covariance matrix can be defined by the use of a kernel function. A kernel is a distance-like user-chosen function that encodes the correlation of two outputs given the distance of their respective inputs. The prior would then take the mathematical form shown in

$$
\mathbf { f } | \mathbf { X } \sim { \mathcal { N } } ( 0 , \mathbf { K } _ { \mathbf { x } , \mathbf { x } } )\tag{1}
$$

where ${ \bf K } _ { { \bf x } , { \bf x } }$ is the kernel function applied pairwise to all input pairs to construct the covariance matrix.

The likelihood of the observations given the model can be represented as a noisy normal distribution around the model predictions:

$$
\mathbf { y } | \mathbf { f } \sim \mathcal { N } ( \mathbf { f } , \sigma ^ { 2 } \mathbf { I } )\tag{2}
$$

Equation (1) and Equation (2) can be combined through the Bayes formula (Equation (3)) to compute the posterior distribution which will serve as our prediction.

$$
P ( \boldsymbol { \theta } | \mathbf { D } ) = \frac { P ( \mathbf { D } | \boldsymbol { \theta } ) P ( \boldsymbol { \theta } ) } { P ( \mathbf { D } ) } .\tag{3}
$$

Solving the above process analytically parametrizes the posterior prediction and generates the prediction distribution.

$$
\mu _ { \mathbf { x } ^ { \star } } = \mathbf { K _ { f ^ { \star } , f } } [ \mathbf { K _ { f , f } } + \sigma ^ { \mathbf { 2 } } \mathbf { I } ] ^ { - 1 } \mathbf { y }\tag{4a}
$$

$$
\sigma _ { \mathbf { x } ^ { \star } } ^ { 2 } = \mathbf { K } _ { \mathbf { x } ^ { \star } , \mathbf { x } ^ { \star } } - \mathbf { K } _ { \mathbf { x } ^ { \star } , \mathbf { x } } [ \mathbf { K } _ { \mathbf { f } , \mathbf { f } } + \sigma ^ { 2 } \mathbf { I } ] ^ { - 1 } \mathbf { K } _ { \mathbf { x } , \mathbf { x } ^ { \star } }\tag{4b}
$$

where the points for prediction and training are denoted with and without the superscript ⋆, respectively. Gaussian processes only require the definition of the prior mean and covariance matrix. The first can be routinely assumed to be zero, though other variations exist, such as a constant mean or a mean that linearly depends on the input values. The latter can be defined with a kernel function whose parameters should, in principle, be uncorrelated to the observations. However, it is common practice to optimize these parameters to maximize the marginal log likelihood to achieve better prediction results. Note that a complete mathematical analysis of Gaussian processes and their derivation is beyond the scope of this work. A comprehensive description can be found in Ref. [40].

It must be noted that PYTHIA aims at providing predictions and optimization with a limited dataset. If the complexity of the system rises significantly, one must resort to the so-called batch GP [41]. These algorithms should be able to handle large datasets, though at the cost of somewhat decreased performance. However, with state of the art algorithms implemented in gpytorch [42, 43] Gps can easily handle datasets of a million samples with very limited approximations.

## 2.5 Foundational Model

Gaussian processes are a versatile family of algorithms. Nevertheless, they sufer significantly with feature spaces of large dimensionality. To this end, we have elected to process the structure using a Graph Neural Network (GNN). The complexity and sheer size of the chemical space needed to be investigated however makes the use of a simple GNN impractical. Therefore, we used an established and powerful foundationa model, MACE [32]. This is done without loss of generality and showcases the power of such architectures. Other architectures, such as NequIP [44] or potentially newer models can be seamlessly integrated.

## 2.6 Translation Layer

MACE is a powerful model, but its accuracy comes at a cost of high complexity and large parameter space. The typical size of datasets for adsorption application, however, does not sufice for the training of such a large model. Therefore, we are forced to utilize pre-trained models on adjacent properties and chemica species. Novel systems, however, difer significantly from the existing databases and models derived from these databases are not of-the-box applicable. For the model to be transferable, there is need for a translation layer that adapts the information extracted to the current case.

The final output of the model however is too specific to use as all the general chemical information have been condensed to a target specific representation. To this end, we extract a deeper MACE layer, containing a rich molecular representation, to be used as input to the translation layer.

The translation layer we have chosen is in the form of transformers. Transformers are a class of deep learning architectures designed to model complex dependencies in sequential or structured data using a mechanism known as self-attention. Introduced by Vaswani et al. (2017) [36], the transformer architecture replaced traditional recurrent and convolutional models in many applications by enabling parallel processing of input data and capturing long-range relationships more efectively.

At the core of the transformer is the multi-head self-attention mechanism, which computes contextualized representations of each input token by attending to all other tokens in the sequence. Each attention head learns to focus on diferent types of relationships by projecting the input into query, key, and value vectors. The attention weights are calculated as scaled dot-products between queries and keys, and these weights are used to combine the value vectors, allowing the model to integrate information from relevant positions in the sequence.

Therefore, their ease of training on sparse data and their natural translation interpretation makes them suitable for transferring the knowledge of the foundational model on our very specific task and dataset.

We do not need to retrain the MACE, a feat both computationally expensive and data ineficient. Instead we leverage its knowledge and only train the translation layer. The pipeline so far however lacks the element of uncertainty quantification. To add this, we use the translated vector as input to a Gaussian Processes model which provides the final prediction and the uncertainty quantification of this prediction. This allows us to leverage said predictions for Bayesian Optimization.

## 3 Results

To evaluate PYTHIA we deployed it on two distinct cases, that of the Zn adsorption on the chini cluster family and the adsorption of formaldehyde $\mathrm { ( C H _ { 2 } O ) }$ on the silicene. In both cases we optimize simultaneously for optimal adsorption position on adsorbent as well as for number and type of defects in the chemical systems. We compare this approach with both the AUGUR framework as well as deploying the foundational model to be fully trained solely on the acquired dataset.

## 3.1 Zn adsorption on Chini Clusters

Chini clusters are platinum (Pt) carbonyl clusters with a distinct chemical formula $[ \mathrm { P t _ { 3 } } ( \mathrm { C O } ) _ { 6 } ] _ { n } ^ { 2 - }$ , built from n stacked layers of triangular Pt units. These are widely used as precursors for synthesising atomprecise Pt nanoclusters (Pt-APNCs) which are obtained by encapsulating them inside a Zeolitic-imidazolate framework (ZIF-8) through a bottle-around ship approach [37]. ZIF-8 assembles from $\mathrm { Z n ^ { 2 + } }$ nodes and 2- methylimidazolate linkers, so the chini cluster interacts with the Zn ions while the framework forms around it. We therefore used $\mathrm { Z n }$ adsorption energy as an optimization target, where more negative values denote stronger binding. The detailed methodology for computing energies is given in SI Section 2.1.

To generate APNCs with higher density of catalytically active sites, experimentalists have tried to substitute Pt with various metal atoms in the chini cluster geometry and have been successful [45]. Therefore, substitution of the metal atoms in the chini cluster architecture is a relevant design variable. We used Ni and Pd as substituents as both belong to group 10 and are isovalent with Pt. Substitution therefore preserves the valence electron count of the cluster and is expected to retain the chini framework, while changing the electronic structure that controls $\mathrm { Z n }$ binding. We considered clusters with $n = 1$ and 2 and allowed any number of Pt atoms, up to all of them, to be substituted. Each candidate structure was thus defined by its dopant configuration together with the Zn position, and PYTHIA optimises both jointly.

We assessed and compared three models for adsorption energy prediction: the AUGUR baseline, MACE model trained fully with a GP regressor as a final layer, and PYTHIA, which combines a frozen pretrained MACE model with Transformers and Gaussian process.

![](images/4d94d3a0eb063f7efad94321f946cac2638d90a2a3a5940018d35b10580891cd.jpg)

![](images/7cc0f06c52a2dc443a4104c13d11831488b2abc296d6a189aa46b90be88f0aab.jpg)

![](images/d5f7cafb6e0cae8d7496b5f1540217b76d3967941e249b31be6d7067eca9b045.jpg)  
Figure 2: Comparison of $\mathrm { Z n ^ { 2 + } }$ adsorption energy predictions for doped Chini clusters a) a GNN with GP as the final layer, b) a MACE model trained fully with GP as a final layer c) PYTHIA. The points predicted correspond to systems with defect configuration not present in the training set.

Figure 2 summarizes the relationship between predicted and reference adsorption energies. PYTHIA aligns most closely with the diagonal, returning an $R ^ { 2 }$ value of 0.83, a mean absolute error (MAE) of 0.33 eV, and a root mean square error (RMSE) of $0 . 3 7 \mathrm { e V }$ . This represents a significant improvement over the next best model, ranging between $8 0 \% - 1 0 0 \%$ , depending on the chosen metric.

PYTHIA correctly captures the trends of the adsorption landscape of $\mathrm { Z n }$ adsorption on the chini clusters with respect to their defect engineering. Specifically, it identified highly promising dopants, and adsorption positions in the chini family. The predicted global minimum for this cluster lies at the top or symmetric bottom sites with adsorption energy close to −6.1 eV. It must be noted, that this optimal adsorption site was identified after just 1 optimization run, with the model being trained on a dataset that exhibited an optimum adsorption energy of −4.74 eV and did not contain $\mathrm { P d } _ { 3 }$ . PYTHIA provided an improvement of approximately 30% on the adsorption energy. Additionally, PYTHIA provides the full energy surface and an uncertainty quantification of any defective surface and can be seen in Fig. 3. This allows for a deeper investigation of systems that range too far from the acquired model understanding and substitutes significan simulation times with minutes of predictions.

![](images/b846343771234c5ec17db447b292a5fe095fcb15a2356bcd27fdb28388959a4f.jpg)  
Figure 3: Energy landscape (left) and corresponding uncertainty (right) for the chini cluster $\mathrm { ( P d _ { 3 } } ,$ , center) that exhibits the best Zn adsorption energy. As can be seen, the model uncertainty is uniformly low, which implies the model has acquired an understanding of the entire energy surface.

## 3.2 Formaldehyde adsorption on Silicene

The chini cluster provided a good baseline but the limited range of defects does not fully showcase the strength of PYTHIA. To this end, we apply our pipeline to a second system, that of formaldehyde adsorbed by defective silicene. This system allows for more diverse and higher number of defects on the cluster, with an added layer of complexity due to its periodic nature.. Additionally, the adsorbent has multiple atoms and needs to be optimized rotationally as well as positionally.

Specifically, silicene was selected as the model substrate because its low-buckled structure and mixed $\mathrm { s p } ^ { 2 } / \bar { \mathrm { s p } } ^ { 3 }$ bonding make it more chemically reactive than graphene [46]. This renders its adsorption properties highly sensitive to local structural and electronic modifications, driving considerable interest in silicene for chemical sensing and surface functionalization [47]. Formaldehyde $\mathrm { ( C H _ { 2 } O ) }$ was chosen as the target adsor bate due to its importance as a hazardous volatile organic compound, for which sensitive room-temperature detection remains an active area of research [48]. To generate a chemically diverse optimization problem, we considered substitutional dopants spanning multiple classes with distinct electronic efects: isovalent Group IV elements (C, Ge), donor-type Group V elements (N, P, As), acceptor-type Group III elements (B, Al, Ga, In), and transition metals with partially filled d states. Main-group dopants have been extensively investigated as a means of tailoring silicene’s electronic structure, band gap, charge distribution, and adsorption behavior [49, 50, 51], while transition-metal dopants introduce localized d-derived states that can give rise to magnetism, topological phases, and enhanced adsorbate interactions [52, 53, 54]. Together, these dopant classes span a broad range of chemically distinct electronic environments, providing a diverse optimization landscape for the active-learning framework. Finally, multiple simultaneous substitutions were included to further increase the complexity of the search space, allowing the framework to explore cooperative efects between diferent dopant species rather than being restricted to isolated substitutional defects.

The comparison between the baseline AUGUR model, fully trained MACE and PYTHIA can be seen in Fig. 4. Unlike the chini case, silicene has a much richer defect space and the adsorbate $\mathrm { C H _ { 2 } O }$ more complex placement options. It also follows that eficiency of PYTHIA will be much more prominently present. AUGUR fails entirely as the high complexity of the space cannot be easily captured with a simple graph neural network. In addition, the limited dataset does not allow for a full MACE training. Again, PYTHIA outperforms both approaches by a significant margin.

(a)  
![](images/16ba7a453f1d78724f5082ad586fdacb7050d411d872c4cc06b90ac74c230631.jpg)

(b)  
![](images/40da287eb9a45d13d97db0257093293c500fd34edb52b3a0a42cdbf7cc4f0c05.jpg)

(c)  
![](images/2629b26934d6a47f6fc67656b003fcad998e69a90558914690110699b6dbdb0b.jpg)  
Figure 4: Comparison of adsorption energy prediction accuracy for a defective $\mathrm { S i l i c e n e { - } C H _ { 2 } O }$ system for a) a GNN with GP as the final layer, b) a MACE model trained fully with GP as a final layer c) PYTHIA

We use the PYTHIA model to optimize both defects of adsorbent and position of adsorbate. For reasons of numerical stability we restricted our defect optimization to maximum 3 defects in one adsorbent. This is not a limitation of PYTHIA and can be easily adjusted to fit a new case with more defects.

The full energy surface of the best adsorbent can be found in Fig. 5. The reason for the relative discontinuity of the energy landscape is the fact that the adsorbate can be rotated, leading to many overlapping and wildly difering energies in comparatively close proximity. This, however, is only for visualization purposes and does not afect the optimization process. It must be noted that the position and defects determined have an energy of -1.12 eV, with only 2 iterations. The best interaction energy of the original dataset was -0.332 $\mathrm { e V } ,$ with PYTHIA providing and improvement of more than 230%. Further, the sheer efort of generating enough DFT calculations to fully explore the energy space now readily available would have taken many years of intense computational efort. PYTHIA however, provides this in minutes, allowing for a deeper understanding of the structure, far exceeding the mere optimum adsorption point determination.

![](images/5ee2c9205cc1a1cb9def20f7f2cac32e386753039683db682cc2fd051b066087.jpg)  
Figure 5: Energy landscape (left) and corresponding uncertainty (right) for the silicene that exhibits the best adsorption properties (Ga defect). The position of each energy point is that of the first Carbon of formaldehyde. Silicene is periodic so the energy surface is non closed. Further, formaldehyde can freely rotate and therefore each position can have multiple energies associated with it. As can be seen, the model uncertainty is uniformly low, which implies the model has acquired an understanding of the entire energy surface.

## 4 Discussion

In this work, we introduced PYTHIA (Probabilistic Yet Transferable Hierarchical Interaction Architecture), a data-eficient transferable machine learning framework for identifying optimal molecular interaction sites and defect choices for a variety of systems. By combining a pre-trained, frozen foundational model (MACE) with a lightweight transformer-based translation layer and a Gaussian Process head, PYTHIA is able to achieve high accuracy with limited data, while simultaneously providing a robust uncertainty estimate. This makes PYTHIA suitable for a Bayesian Optimization framework, where eficient information acquisition is equally important to final accuracy. We have tested our pipeline in two highly relevant cases, namely Zn adsorption on the doped chini cluster family and $\mathrm { C H _ { 2 } O }$ adsorption on doped silicene. Across both cases, PYTHIA outperformed all benchmarks leading to high improvement in the defect-site dual optimization problem, as well as resulting in much more accurate final models. It must be noted that the prediction accuracy of PYTHIA was high, even in defects not present in the dataset, thus paving the way to a full chemical space investigation, a luxury rarely aforded with conventional approaches.

This modularity, combined with the demonstrated transferability to systems and defect configurations not explicitly represented in the training set, suggests that PYTHIA is broadly applicable to other classes of interaction-site optimization problems in computational materials discovery — including catalyst design, gas sensing, and surface functionalization, where DFT data is scarce but adjacent, richly-labeled datasets exist to transfer from. Lastly, PYTHIA is not limited by the case study or optimization objective. It can just as seamlessly be applied to determining ideal solvents as well as defects and adsorption sites. Additionally, as state of the art advances new models can seamlessly substitute MACE thus keeping this framework consistently relevant. The full pipeline will provide even more value in highly novel and complex materials, where large datasets do not exist and data accuracy is as vital as good optimization results.

## Data Availability

Data will be made available via github upon acceptance for publication.

## Code Availability

Code will be made available via github upon acceptance for publication.

## ACKNOWLEDGEMENTS

A.G. acknowledges financial support from TUM Innovation Network for Artificial Intelligence powered Multifunctional Material Design (ARTEMIS) and funding in the framework of Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC 2089/1 – 390776260 (econversion). Lastly, we wish to express our gratitude to Dr. Inigo Iribarren for creating the flowchart for this work.

The Authors declare no Competing Financial or Non-Financial Interests.

## References

[1] Egill Sk´ulason et al. “Density functional theory calculations for the hydrogen evolution reaction in an electrochemical double layer on the Pt(111) electrode”. In: Phys. Chem. Chem. Phys. 9.25 (2007), 3241–3250. issn: 1463-9084. doi: 10.1039/b700099e. url: http://dx.doi.org/10.1039/b700099e.

[2] C. M. Wei, A. Groß, and M. Schefler. “Ab initiocalculation of the potential energy surface for the dissociation of H2 on the sulfur-covered Pd(100) surface”. In: Physical Review B 57.24 (June 1998), pp. 15572–15584. issn: 1095-3795. doi: 10.1103/physrevb.57.15572.

[3] Carles Mart´ı et al. “DockOnSurf: A Python Code for the High-Throughput Screening of Flexible Molecules Adsorbed on Surfaces”. In: Journal of Chemical Information and Modeling 61.7 (June 2021), pp. 3386–3396. issn: 1549-960X. doi: 10.1021/acs.jcim.1c00256.

[4] Enrico Pedretti, Paolo Restuccia, and M. Clelia Righi. “Xsorb: A software for identifying the most stable adsorption configuration and energy of a molecule on a crystal surface”. In: Computer Physics Communications 291 (Oct. 2023), p. 108827. issn: 0010-4655. doi: 10.1016/j.cpc.2023.108827. url: http://dx.doi.org/10.1016/j.cpc.2023.108827.

[5] Stefan Goedecker. “Minima hopping: An eficient search method for the global minimum of the potential energy surface of complex molecular systems”. In: The Journal of Chemical Physics 120.21 (June 2004), 9911–9917. issn: 1089-7690. doi: 10.1063/1.1724816. url: http://dx.doi.org/10.1063/1. 1724816.

[6] Sandro E. Schonborn et al. “The performance of minima hopping and evolutionary algorithms for cluster structure prediction”. In: The Journal of Chemical Physics 130.14 (Apr. 2009). issn: 1089- 7690. doi: 10.1063/1.3097197. url: http://dx.doi.org/10.1063/1.3097197.

[7] Hannes Jonsson, Greg Mills, and Karsten W Jacobsen. “Nudged elastic band method for finding minimum energy paths of transitions”. In: Classical and quantum dynamics in condensed phase simulations. World Scientific, 1998, pp. 385–404.

[8] Alessandro Barducci, Giovanni Bussi, and Michele Parrinello. “Well-tempered metadynamics: a smoothly converging and tunable free-energy method”. In: Physical review letters 100.2 (2008), p. 020603.

[9] Azade YazdanYar, Ulrich Aschauer, and Paul Bowen. “Adsorption Free Energy of Single Amino Acids at the Rutile (110)/Water Interface Studied by Well-Tempered Metadynamics”. In: The Journal of Physical Chemistry C 122.21 (May 2018), pp. 11355–11363. issn: 1932-7455. doi: 10.1021/acs.jpcc. 7b12614. url: http://dx.doi.org/10.1021/acs.jpcc.7b12614.

[10] Felix Mayr et al. “Machine learning and optoelectronic materials discovery: a growing synergy”. In: The Journal of Physical Chemistry Letters 13.8 (2022), pp. 1940–1951.

[11] Carola Lampe et al. “Rapid Data-Eficient Optimization of Perovskite Nanocrystal Syntheses through Machine Learning Algorithm Fusion”. In: Advanced Materials 35.16 (2023), p. 2208772.

[12] Ioannis Kouroudis, Manuel Gosswein, and Alessio Gagliardi. “Utilizing data-driven optimization to automate the parametrization of kinetic Monte Carlo models”. In: The Journal of Physical Chemistry A 127.28 (2023), pp. 5967–5978.

[13] S¨onke Lorenz, Matthias Schefler, and Axel Gross. “Descriptions of surface chemical reactions using a neural network representation of the potential-energy surface”. In: Physical Review B 73.11 (Mar. 2006), p. 115431. issn: 1550-235X. doi: 10.1103/physrevb.73.115431.

[14] Marc OJ Jager et al. “Machine learning hydrogen adsorption on nanoclusters through structural descriptors”. In: npj Computational Materials 4.1 (2018), p. 37.

[15] Sergio Pablo-Garc´ıa et al. “Fast evaluation of the adsorption energy of organic molecules on metals via graph neural networks”. In: Nature Computational Science 3.5 (May 2023), 433–442. issn: 2662-8457. doi: 10.1038/s43588-023-00437-y. url: http://dx.doi.org/10.1038/s43588-023-00437-y.

[16] Janice Lan et al. “AdsorbML: a leap in eficiency for adsorption energy calculations using generalizable machine learning potentials”. In: npj Computational Materials 9.1 (Sept. 2023). issn: 2057-3960. doi: 10.1038/s41524-023-01121-5. url: http://dx.doi.org/10.1038/s41524-023-01121-5.

[17] Ryosuke Jinnouchi et al. “Phase Transitions of Hybrid Perovskites Simulated by Machine-Learning Force Fields Trained on the Fly with Bayesian Inference”. In: Physical Review Letters 122.22 (June 2019). doi: 10.1103/physrevlett.122.225701.

[18] Yu Xie et al. “Uncertainty-aware molecular dynamics from Bayesian active learning for phase transformations and thermal transport in SiC”. In: npj Computational Materials 9.1 (Mar. 2023). doi: 10.1038/s41524-023-00988-8.

[19] Tom Wollschlager et al. “Uncertainty estimation for molecules: Desiderata and methods”. In: International Conference on Machine Learning. PMLR. 2023, pp. 37133–37156.

[20] Tomoki Yamashita et al. “Crystal structure prediction accelerated by Bayesian optimization”. In: Physical Review Materials 2.1 (Jan. 2018), p. 013803. doi: 10.1103/physrevmaterials.2.013803.

[21] Aryan Deshwal, Cory M. Simon, and Janardhan Rao Doppa. “Bayesian optimization of nanoporous materials”. In: Molecular Systems Design & Engineering 6.12 (2021), pp. 1066–1086. doi: 10.1039/ d1me00093d.

[22] Yunxing Zuo et al. “Accelerating materials discovery with Bayesian optimization and graph deep learning”. In: Materials Today 51 (Dec. 2021), 126–135. issn: 1369-7021. doi: 10.1016/j.mattod. 2021.08.012. url: http://dx.doi.org/10.1016/j.mattod.2021.08.012.

[23] Kevin Tran and Zachary W. Ulissi. “Active learning across intermetallics to guide discovery of electrocatalysts for CO2 reduction and H2 evolution”. In: Nature Catalysis 1.9 (Sept. 2018), pp. 696–703. issn: 2520-1158. doi: 10.1038/s41929-018-0142-1.

[24] Andrew A. Peterson. “Acceleration of saddle-point searches with machine learning”. In: The Journal of Chemical Physics 145.7 (Aug. 2016). issn: 1089-7690. doi: 10.1063/1.4960708.

[25] Olli-Pekka Koistinen et al. “Nudged Elastic Band Calculations Accelerated with Gaussian Process Regression Based on Inverse Interatomic Distances”. In: Journal of Chemical Theory and Computation 15.12 (Oct. 2019), 6738–6751. issn: 1549-9626. doi: 10.1021/acs.jctc.9b00692. url: http://dx. doi.org/10.1021/acs.jctc.9b00692.

[26] Hyunwook Jung et al. “Machine-Learning Driven Global Optimization of Surface Adsorbate Geometries”. In: (June 2023). doi: 10.26434/chemrxiv-2022-q3j0s-v3.

[27] Lucian Chan, Geofrey R. Hutchison, and Garrett M. Morris. “Bayesian optimization for conformer generation”. In: Journal of Cheminformatics 11.1 (May 2019). issn: 1758-2946. doi: 10.1186/s13321- 019-0354-7. url: http://dx.doi.org/10.1186/s13321-019-0354-7.

[28] Milica Todorovic et al. “Bayesian inference of atomistic structure in functional materials”. In: Npj computational materials 5.1 (2019), p. 35.

[29] Jari J¨arvi, Patrick Rinke, and Milica Todorovi´c. “Detecting stable adsorbates of (1S)-camphor on Cu(111) with Bayesian optimization”. In: Beilstein Journal ofNanotechnology 11 (Oct. 2020), pp. 1577– 1589. issn: 2190-4286. doi: 10.3762/bjnano.11.140.

[30] Lincan Fang et al. “Exploring the Conformers of an Organic Molecule on a Metal Cluster with Bayesian Optimization”. In: Journal of Chemical Information and Modeling 63.3 (Jan. 2023), pp. 745–752. issn: 1549-960X. doi: 10.1021/acs.jcim.2c01120.

[31] Ioannis Kouroudis et al. “AUGUR, a flexible and eficient optimization algorithm for identification of optimal adsorption sites”. In: npj Computational Materials 11.1 (2025), p. 136.

[32] Ilyes Batatia et al. “MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields”. In: Advances in Neural Information Processing Systems. Ed. by Alice H. Oh et al. 2022. url: https://openreview.net/forum?id=YPpSngE-ZU.

[33] Ilyes Batatia, Philipp Benner, Yuan Chiang, et al. “A foundation model for atomistic materials chemistry”. In: The Journal of Chemical Physics 163.18 (2025), p. 184110. doi: 10.1063/5.0297006.

[34] Mariia Radova et al. “Fine-tuning foundation models of materials interatomic potentials with frozen transfer learning”. In: npj Computational Materials 11 (2025), p. 237. doi: 10.1038/s41524-025- 01727-x.

[35] Ioannis Kouroudis et al. “A green solvent screening tool for emerging materials via uncertainty aware, transformer enhanced transfer learning”. In: Scientific Reports 16.1 (2026), p. 30092. issn: 2045-2322. doi: 10.1038/s41598-026-72415-z. url: https://doi.org/10.1038/s41598-026-72415-z.

[36] Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural information processing systems 30 (2017).

[37] Kathrin L. Kollmannsberger et al. “Mechanistic Insights into ZIF-8 Encapsulation of Atom-Precise Pt(M) Carbonyl Clusters”. In: Chemistry of Materials 35.14 (2023), pp. 5475–5486. doi: 10.1021/ acs.chemmater.3c00807.

[38] S. Cahangirov et al. “Two- and One-Dimensional Honeycomb Structures of Silicon and Germanium”. In: Phys. Rev. Lett. 102 (23 June 2009), p. 236804. doi: 10.1103/PhysRevLett.102.236804. url: https://link.aps.org/doi/10.1103/PhysRevLett.102.236804.

[39] Kyozaburo Takeda and Kenji Shiraishi. “Theoretical possibility of stage corrugation in Si and Ge analogs of graphite”. In: Phys. Rev. B 50 (20 Nov. 1994), pp. 14916–14922. doi: 10.1103/PhysRevB. 50.14916. url: https://link.aps.org/doi/10.1103/PhysRevB.50.14916.

[40] Christopher K. I. Williams and Carl Edward Rasmussen. Gaussian Processes for Machine Learning. Vol. 2. 3. MIT press Cambridge, MA, 2006.

[41] Erik A Daxberger and Bryan Kian Hsiang Low. “Distributed batch Gaussian process optimization”. In: International conference on machine learning. PMLR. 2017, pp. 951–960.

[42] Jacob Gardner et al. “Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration”. In: Advances in neural information processing systems 31 (2018).

[43] Geof Pleiss et al. “Constant-time predictive distributions for Gaussian processes”. In: International Conference on Machine Learning. PMLR. 2018, pp. 4114–4123.

[44] Simon Batzner et al. “E (3)-equivariant graph neural networks for data-eficient and accurate interatomic potentials”. In: Nature communications 13.1 (2022), p. 2453.

[45] Cristiana Cesari et al. “Heterometallic Ni–Pt Chini-Type Carbonyl Clusters: An Example of Molecular Random Alloy Clusters”. In: Inorganic Chemistry 60.12 (June 2021), pp. 8811–8825. issn: 0020-1669. doi: 10.1021/acs.inorgchem.1c00752. eprint: https://pubs.acs.org/inocaj/article-pdf/60/ 12/8811/5715196/ic1c00752.pdf. url: https://doi.org/10.1021/acs.inorgchem.1c00752.

[46] Laurence Masson and Geofroy Pr´evot. “Epitaxial Growth and Structural Properties of Silicene and Other 2D Allotropes of Si”. In: Nanoscale Advances 5.6 (2023), pp. 1574–1599. issn: 2516-0230. doi: 10.1039/D2NA00808D. url: https://pubs.rsc.org/na/article/5/6/1574-1599/773295 (visited on 07/19/2026).

[47] Alessandro Molle et al. “Silicene, Silicene Derivatives, and Their Device Applications”. In: Chemical Society Reviews 47.16 (2018), pp. 6370–6387. issn: 0306-0012, 1460-4744. doi: 10.1039/C8CS00338F. url: https://pubs.rsc.org/cs/article/47/16/6370-6387/600673 (visited on 07/19/2026).

[48] Yong Kun Jo et al. “Exclusive and Ultrasensitive Detection of Formaldehyde at Room Temperature Using a Flexible and Monolithic Chemiresistive Sensor”. In: Nature Communications 12.1 (Aug. 16, 2021), p. 4955. issn: 2041-1723. doi: 10.1038/s41467-021-25290-3. url: https://www.nature. com/articles/s41467-021-25290-3 (visited on 07/19/2026).

[49] Deepak Arumugam et al. “Exploring the Electrocatalytic Performance of Silicene and Single Atom Doped Silicene for HER, OER and ORR Activity Using Density Functional Theory”. In: Chemical Physics Impact 9 (Dec. 2024), p. 100664. issn: 26670224. doi: 10.1016/j.chphi.2024.100664. url: https://linkinghub.elsevier.com/retrieve/pii/S2667022424002081 (visited on 07/19/2026).

[50] Aditya Tiwari et al. “Comparative Analysis of Boron, Nitrogen, and Phosphorous Doping in Monolayer of Semi-Metallic Xenes (Graphene, Silicene, and Germanene) - A First Principle Calculation Based Approach”. In: Materials Science in Semiconductor Processing 153 (Jan. 2023), p. 107121. issn: 13698001. doi: 10.1016/j.mssp.2022.107121. url: https://linkinghub.elsevier.com/retrieve/pii/ S1369800122006473 (visited on 07/19/2026).

[51] Tamiru Teshome and Ayan Datta. “Efect of Doping in Controlling the Structure, Reactivity, and Electronic Properties of Pristine and Ca(II)-Intercalated Layered Silicene”. In: The Journal of Physical Chemistry C 121.28 (July 20, 2017), pp. 15169–15180. issn: 1932-7447, 1932-7455. doi: 10.1021/ acs.jpcc.7b03002. url: https://pubs.acs.org/doi/10.1021/acs.jpcc.7b03002 (visited on 07/19/2026).

[52] Xiao-Long Zhang, Lan-Feng Liu, and Wu-Ming Liu. “Quantum Anomalous Hall Efect and Tunable Topological States in 3d Transition Metals Doped Silicene”. In: Scientific Reports 3.1 (Oct. 9, 2013), p. 2908. issn: 2045-2322. doi: 10.1038/srep02908. url: https://www.nature.com/articles/ srep02908 (visited on 07/19/2026).

[53] P. Kalpana Devi and K.K. Singh. “Improving the Electronic and Optical Properties of Silicene by Doping with Transition Metals – A DFT Investigation”. In: Computational and Theoretical Chemistry 1245 (Mar. 2025), p. 115117. issn: 2210271X. doi: 10.1016/j.comptc.2025.115117. url: https: //linkinghub.elsevier.com/retrieve/pii/S2210271X25000532 (visited on 07/19/2026).

[54] Nzar Rauf Abdullah et al. “Spin-Polarised DFT Modeling of Electronic, Magnetic, Thermal and Optical Properties of Silicene Doped with Transition Metals”. In: Physica E: Low-dimensional Systems and Nanostructures 129 (May 2021), p. 114644. issn: 13869477. doi: 10.1016/j.physe.2021.114644. url: https://linkinghub.elsevier.com/retrieve/pii/S1386947721000266 (visited on 07/19/2026).