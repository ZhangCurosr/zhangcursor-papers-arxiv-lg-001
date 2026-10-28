# LEARNING ATOMIC INTERACTIONS FOR THE THERMODYNAMICS OF LIQUID MIXTURES

Marco Hoffmann   
RPTU Kaiserslautern-Landau Kaiserslautern   
marco.hoffmann@rptu.de

Sebastian Schmitt RPTU Kaiserslautern-Landau Kaiserslautern sebastian.schmitt@rptu.de

Fabian Jirasek   
RPTU Kaiserslautern-Landau Kaiserslautern   
fabian.jirasek@rptu.de

Hans Hasse RPTU Kaiserslautern-Landau Kaiserslautern hans.hasse@rptu.de

## ABSTRACT

Predictive models for activity coefficients, which describe the non-ideality of liquid mixtures, are foundational for process design and optimization. Group-contribution methods, such as UNIFAC, are computationally cheap and well established in process simulators. Modern machine learning models, such as HANNA, offer higher accuracy and broader applicability, but their integration into process simulation is expensive or even infeasible. In this work, we introduce COGNAC, a graph neural network that learns atom-level geometry and interaction parameters that depend on the molecular neighborhood and evaluates them in the UNIFAC framework. Trained on about 820,000 experimental data points, COGNAC achieves significantly higher accuracy than UNIFAC and HANNA for binary and multi-component mixtures, reduces error relative to HANNA by 13–63%, reliably predicts multi-phase equilibria and identifies azeotropes, yields interpretable parameters, and is strictly thermodynamically consistent. COGNAC evaluations run at a cost comparable to UNIFAC, thereby unlocking the potential of machine-learning models for process simulations.

## 1 Main

The feasibility and cost of distillation, extraction, and crystallization processes are governed by the non-ideality of the liquid phase. This non-ideality is quantified by the activity coefficients of the components, which are therefore central to the design and optimization of chemical processes. Because experimental data are unavailable for most mixtures of interest, activity coefficients are predicted with models for the excess Gibbs energy g<sup>E</sup>, which express the non-ideality as a function of composition and temperature.

The most established predictive excess Gibbs energy models are group contribution (GC) methods, in particular UNIFAC and its variants [1, 2, 3, 4, 5, 6]. They decompose molecules into predefined structural groups and describe non-ideal mixture behavior through interactions between these groups. These models have been widely adopted in chemical engineering because they are computationally inexpensive and are implemented in most process simulators. At the same time, the predefined group structure limits the molecular resolution of UNIFAC models because they resolve the molecular neighborhood only coarsely. For example, modified UNIFAC (Dortmund), called mod. UNIFAC in the following, accounts for the molecular neighborhood of a hydroxy group using subgroups that distinguish only the immediate bonding situation (OH bound to a primary, secondary, or tertiary carbon), but share their interaction parameters such that effects that extend beyond the nearest neighbors are lost. This limits the description of multifunctional molecules and structural isomers that decompose into the same groups. Furthermore, GC method apply only to decomposable components for which both suitable structural groups and fitted parameters are available.

Quantum-chemistry (QC) based methods, such as COSMO-RS [7] and COSMO-SAC [8], compute activity coefficients from molecular surface-charge distributions and need no component-specific parameterizations. However, they require an expensive QC calculation per component and have not shown a systematic accuracy advantage over GC methods where both apply [9, 10, 11].

machine learning (ML) has recently opened alternatives to predictive thermodynamics, including hybrid strategies [12, 13, 14] that embed ML in an established physics-based model and train it to predict the model’s parameters [15, 16, 17, 18]. Hayer et al. [19, 20] applied matrix completion methods (MCM) to complete the interaction parameter tables of UNIFAC [1] and mod. UNIFAC [3], yielding models with broader applicability and higher accuracy than their predecessors. However, these new methods still rely on the predefined functional groups of the original UNIFAC formulation, and certain types of molecules (e.g., most ionic liquids) remain out of scope. Following a similar idea, SPT-NRTL [21] predicts the parameters of the Non-Random-Two-Liquid (NRTL) model [22] from SMILES with a transformer [23].

Another hybrid strategy uses flexible neural networks to directly predict excess Gibbs energy or activity coefficients while incorporating physical constraints into the ML architecture or training loss [24, 25, 26, 27, 28]. The most accurate of these models, HANNA [27, 28], relies on pre-trained molecular embeddings (ChemBERTa-2 [29]) and enforces Gibbs–Duhem consistency, equivariance, and the pure-component limits by construction. The hybrid models are accurate and broadly applicable because they need only molecular structure as input. However, because the neural network is part of the thermodynamic model, it must be evaluated at every simulation step. This makes neural-network-based models prohibitively expensive for simulation purposes.

Here, we introduce COGNAC (local-COmposition Graph Neural network for the prediction of Activity Coefficients), a g<sup>E</sup> model that combines a graph neural network (GNN) with the local-composition framework of mod. UNIFAC. In this framework, mixture non-ideality arises from an ensemble of pairwise interacting sites, a picture that QC-based methods also share (Supplementary Note 1). COGNAC retains the local-composition framework but fundamentally changes what an interaction site is: instead of predefined structural groups, it uses individual atoms, whose geometry and interaction parameters depend on their molecular neighborhood. In COGNAC, each atom’s molecular neighborhood extends over several bonds and influences all its parameters, including pair interactions. COGNAC therefore captures how neighboring functional groups influence each other, something structural groups cannot resolve. For the user, a single network evaluation turns the SMILES of the components in a mixture into a mod. UNIFAC parameter table that any process simulator can use.

Fig. 1 gives an overview of the COGNAC architecture. A message-passing neural network [32] encodes each atom in each component together with its molecular neighborhood within four bonds; symmetry-equivalent atoms receive identical embeddings, and hydrogens bound to non-carbon atoms are explicit nodes, making hydrogen-bond donors and acceptors distinct sites. From these embeddings, one network predicts each atom’s volume and surface-area parameters, and another predicts the interaction parameters for every atom pair. The parameters enter the unchanged mod. UNIFAC equations: molecular geometry in the combinatorial contribution, surface areas and pair interactions in the residual contribution.

COGNAC is trained end-to-end on 822,780 experimental data points for 48,432 binary mixtures from the Dortmund Data Bank (DDB) [33], including vapor–liquid equilibria (VLE), liquid–liquid equilibria (LLE), solid–liquid equilibria (SLE), activity coefficients at infinite dilution (ACT), and excess enthalpies (HE). The learned atomic surface areas and molecular volumes are regularized against COSMO [34] cavity data from the CHAOS database [35]. Training on LLE data is enabled by a differentiable phase-split solver combining the convex envelope method [36, 37, 38] with a Newton step (Methods).

Separating parameter prediction from thermodynamic evaluation has several advantages. The excess Gibbs energy and all derived properties are smooth and satisfy the Gibbs–Duhem equation, permutation symmetry, and the pure-component limits by construction; the same equations describe binary and multi-component mixtures without additional parameters or a separate multi-component approximation; the learned geometry and interaction parameters are interpretable; and because they are generated once for a specified component set and reused across compositions and temperatures, phase-equilibrium calculations and process simulations require no further neural network evaluations.

![](images/d2ab88c78851e13baef5f014aab5c398aea4fa620e046551be58f175ea866de9.jpg)

## d Application

![](images/9a0aef8c148fcdaf8aefd06279d82b450a6061a4fca090b314b36b193f56f6d2.jpg)  
Figure 1: Overview of the COGNAC model architecture and its application. a, Molecular graphs constructed from SMILES [30] using RDKit [31] are encoded by a message-passing neural network (MPNN). Heavy atoms and hydrogens bound to non-carbon atoms form explicit nodes, allowing hydrogen-bond donors and acceptors to be resolved. Atoms with identical molecular neighborhoods within the message-passing radius obtain identical embeddings. b, Two prediction heads map the context-aware atom embeddings $\mathbf { h } _ { a }$ to volume and surface-area parameters $R _ { a }$ and $Q _ { a } ,$ and pairs of embeddings to the symmetric interaction coefficients $a _ { a b } , b _ { a b } ,$ and $c _ { a b }$ of the exported parameter table (Methods). Summing the respective atom-level geometry parameters gives the molecular quantities $r _ { i }$ and $q _ { i }$ . $\mathbf { c } ,$ The learned parameters and the state point (composition x and temperature T) are processed in a local-composition framework. The combinatorial contribution accounts for size and shape effects, while atomic surface areas and pair interactions determine the residual contribution. Together, the two contributions yield the activity coefficient of component i. d, For a specified component set, parameters need to be generated only once and can be reused across compositions and temperatures without further neural network evaluations. The obtained atom-level parameter vectors R and Q and interaction matrices ${ \mathbf { a } } ,$ b, and c enable, e.g., the prediction of vapor–liquid (VLE), liquid–liquid (LLE), and solid–liquid (SLE) equilibria, azeotropes, and residue curve maps for binary and multi-component mixtures, and can easily be implemented in process simulations.

b  
![](images/96bf5c17b1d12680b388e2f27eaf6946cdd63ddab9368b9b7af6a97b522b8b9c.jpg)

## 2 Results

## 2.1 Prediction of phase equilibria and excess properties

We benchmark COGNAC against mod. UNIFAC [6], the most established predictive activity coefficient model, and HANNA [28], the most accurate published ML model, on binary and ternary mixtures (Fig. 2).

![](images/ecf0d6f1476e508642c4f4fc458907b9d565635fd860933434172c4e8f491374.jpg)

![](images/30ef5d40a10baf5ea966240d91610216032295e999d6355d02773ce3f4eb71a2.jpg)

![](images/8d6300bde408cc6ebb31d041d05fa8431e28a106e746317135be3be8c0e00ac6.jpg)

![](images/4cad76b5a11b19610a3cbab56789da28f7947644f094552ed47a8bbe9f5eec16.jpg)

![](images/bab4d3f45ffd7737259262cab763c5ac0186dedf8a9b40b3bd9391ba0728ed27.jpg)

![](images/e6e85601cd0dc7a5d8f87e62093c717b5028526215899d0d2b73662b7742d47c.jpg)

![](images/cba111fe0ae7a236210dd0bf518c4eadb21e5a4ec4127369fa8b9ab2a37a1c50.jpg)

![](images/460b904b2aebf42b645681e6c14b1ec69b534ccd6394030ad98a09a1d89f7270.jpg)

![](images/1723ecd554a4fd9074c2b20ac0a4a96fa768e71033ddfa2b3b372867bda731f7.jpg)

![](images/c66709a3453542f17a219e51f80475314a10a6fd2e466d7355002ff958ae07c0.jpg)  
e

c  
Diphenylmethane (1) + guaiacol (2)  
![](images/d6bd7dbe8e4af0253eec10e3139bb9233885e05a4014d87c8ae2706a37067df9.jpg)

![](images/43c9f619c5f8274b3950451863f5242e09e379d1f7aecdcbb5fee122e32af558.jpg)

1-Butanol (1) + diethyl malonate (2)  
![](images/46f9798078d82923046c89d2aceb480af8721d290fce6b346339410a97a6e80e.jpg)  
Figure 2: Results of COGNAC and comparison to mod. UNIFAC and HANNA. a, System-wise mean absolute errors $\mathrm { M A E _ { s y s } }$ for the binary data, resolved by data type. Results of COGNAC and HANNA are obtained by system-wise ten-fold cross-validation, ensuring unseen test systems. Data types: activity coefficients from vapor–liquid equilibria with (TPXY) and total pressures without (TPX) reported vapor composition, activity coefficients at infinite dilution (ACT), liquid–liquid equilibria (LLE), excess enthalpies (HE), and activity coefficients from solid–liquid equilibria (SLE). The scores are calculated from the absolute deviations in the logarithmic activity coefficient (TPXY, ACT, SLE), logarithmic pressure (TPX), liquid mole fractions (LLE), and excess enthalpies in kJ $\mathrm { m o l ^ { - 1 } }$ (HE). Circles denote the median, diamonds the mean, and the shaded bars the interquartile range of the distribution of $\mathrm { M A E _ { s y s } }$ over all systems; n is the number of systems in the respective panel, restricted to those systems that all three models can describe. The percentage above each panel shows the relative change in the median $\mathrm { M A E _ { s y s } }$ of COGNAC relative to HANNA; negative values indicate that COGNAC is more accurate. Mod. UNIFAC describes 50% of the binary systems of the full data set (Extended Data Fig. 1). COGNAC and HANNA were trained and evaluated on the same data, mod. UNIFAC was used as published. b, Same comparison for the ternary data set, predicted with the full-data models of COGNAC and HANNA. Mod. UNIFAC describes 52% of the ternary systems of the full data set. c–e, Exemplary COGNAC predictions for different binary data types against experimental data from unseen systems: solid–liquid equilibrium with eutectic point (c; the dashed line is the ideal-solution liquidus), heteroazeotrope with three-phase equilibrium at atmospheric pressure (d; the horizontal line marks the temperature at which vapor and two liquid phases coexist; below it, the green lines are the compositions of the two coexisting liquid phases), and excess enthalpies at different temperatures (e).

We compare six data types: activity coefficients from VLE with specified vapor composition (TPXY), total pressures from VLE without reported vapor composition (TPX), phase compositions in LLE, activity coefficients derived from SLE, data for activity coefficients at infinite dilution (ACT), and excess enthalpies (HE). For each data type, we calculate the system-wise mean absolute errors $\mathrm { M A E _ { s y s } }$ , which we use as the primary error metric (Methods). Binary results are out-of-fold predictions of a ten-fold cross-validation over systems with HANNA retrained on identical folds; ternary results use COGNAC and HANNA trained once on all binary data; no multi-component data was used for training. Mod. UNIFAC, used as published, describes only 43–83% of the binary and 21–85% of the ternary systems depending on the data type. Fig. 2 therefore compares the systems all three models describe; the full data set comparison between COGNAC and HANNA is shown in Extended Data Fig. 1.

For the binary mixtures, COGNAC yields the lowest median and mean $\mathrm { M A E _ { s y s } }$ across all data types. Relative to HANNA, the median error is 13–32% lower across data types, and relative to mod. UNIFAC, 8–53% lower. The improvement is consistent: HANNA is ahead only in one fold on one data type (SLE), so no data type improves at the expense of another. This advantage of COGNAC holds also on the full data set, including the systems mod. UNIFAC cannot describe (Extended Data Fig. 1).

HANNA extrapolates from binary to ternary mixtures via geometric projection, whereas COGNAC evaluates the same equations over atoms from any number of components. This mechanistic extrapolation pays off (Fig. 2b): COGNAC reduces the median $\mathrm { M A E _ { s y s } }$ relative to HANNA by 17–49%, more than for the binary mixtures for every data type and most clearly for ACT (49 vs. 25%) and HE (45 vs. 32%). It is also at least as accurate as mod. UNIFAC for every ternary data type (equal medians for TPX, 18–44% lower otherwise), although mod. UNIFAC was fitted to ternary data and COGNAC was not. The same holds for quaternary mixtures (Extended Data Fig. 2): 27–63% lower median errors than HANNA and 9–18% lower than mod. UNIFAC on the systems all three models describe, and 22–61% lower than HANNA on the full data set.

This advantage holds across architectural choices: in an ablation study of the temperature dependence of the interaction parameters and the residual-term closure, all six variants beat HANNA on every binary data type (Extended Data Table 1, Supplementary Note 2). COGNAC uses the closed-form residual term of UNIFAC, which sets the surrounding sites to unit activity, an approximation to COSMOSPACE’s self-consistent closure [39] that must be solved iteratively. We also tested a self-consistent variant, COGNAC-IMP, that improves the mean $\mathrm { M A E _ { s y s } }$ of binary activity coefficients at infinite dilution by 4.5% and of excess enthalpies by 4.8%, with mixed effects on the other data types and ternary and quaternary mixtures (Supplementary Note 3, Supplementary Table 1). We therefore use the closed form for COGNAC, which keeps the mod. UNIFAC equations unchanged and needs no iterative solver.

Fig. 2c–e shows three out-of-fold predictions: a binary SLE, a heteroazeotrope, and excess enthalpies at three temperatures. The heteroazeotrope in Fig. 2d addresses a long-standing problem: classical $g ^ { \mathrm { E } }$ models [40, 41] struggle to describe VLE and LLE simultaneously with a single parameterization, and the original UNIFAC required a separate LLE parameter table [2]. Accordingly, mod. UNIFAC is about as accurate as HANNA for VLE but considerably less accurate for LLE (Fig. 2a), whereas HANNA, due to its greater flexibility, describes both well [28]. COGNAC is the most accurate for both using the unchanged mod. UNIFAC equations. This suggests that UNIFAC’s limitation stems not from its equations but from its rigid group definitions.

Predicting the behavior of aqueous mixtures remains challenging for thermodynamic models: water is a small, strongly self associating molecule, and hydrogen bonding governs mixture behavior. Aqueous mixtures are the hardest case for all models (Extended Data Fig. 3), yet COGNAC is more accurate than HANNA for every data type here as well, with the largest gains for ternary VLE and ACT. We attribute this to the explicit polar hydrogen nodes: water is represented by three interaction sites, and a hydrogen bond is a donor–acceptor pair interaction rather than a collapsed property of the water molecule. In an ablation that treats all hydrogen atoms implicitly, with architecture, data, hyperparameters, and seed unchanged (which also removes the donor–acceptor sign regularizer, Methods), the median error of aqueous systems rises by 31% but that of non-aqueous systems b only 1.7%, and aqueous mixtures get worse in all eleven combinations of mixture order and data type (Extended Data Fig. 4).

A known weakness of mod. UNIFAC is largely removed by COGNAC: with its published parameters, mod. UNIFAC underestimates the activity coefficients of alkanes at infinite dilution in water, a compromise that was deliberately accepted in favor of an accurate description of alcohol–water systems [42]. COGNAC, evaluating the same equations, describes alkanes and alcohols in water well (Extended Data Fig. 5).

COGNAC also resolves structural isomers with identical group counts, for which mod. UNIFAC predicts the same activity coefficients by construction. Among such isomer pairs whose measured ln $\gamma ^ { \infty }$ differ by at least 0.15, COGNAC reproduces the experimental order in 84.5% of 1,368 cases (Extended Data Fig. 6, Supplementary Note 4).

## 2.2 Prediction of azeotropes and residue curve maps

Azeotropes, where vapor and liquid have the same composition, limit the feasibility of simple distillation and, in multi-component mixtures, divide the composition space into distillation regions that residue curves cannot cross. We assess COGNAC’s performance in predicting azeotropes on the DDB azeotropic data bank [33], which contains both positive samples (azeotropic mixtures and

their azeotrope type) and negative samples (zeotropic mixtures). We locate all homogeneous azeotropes from

$$
y _ { i } ( \mathbf { x } , T ) = x _ { i } , \qquad i = 1 , \ldots , N - 1 ,\tag{1}
$$

at the reported temperature (Supplementary Note 5). A prediction is counted as correct if it reproduces the existence and type of the measured azeotrope. Binary azeotropes are out-of-fold predictions; ternary azeotropes are predicted with the full-data models, trained on all binary data. Fig. 3 summarizes the results.

Figs. 3a and 3b compare the predicted azeotrope type with the measured one. Binary azeotropes are minimum-boiling $( p _ { \mathrm { m a x } } )$ or maximum-boiling $( p _ { \mathrm { m i n } } )$ azeotropes, and a few binary mixtures show both at the same temperature (double). Ternary azeotropes can additionally form saddle-point azeotropes. COGNAC correctly classifies 92.5% of the 25,894 data points and HANNA 90.9%;

![](images/2fa22b3663c991cdd2756334e57378baf0547e170d3a1e637ab06cf8300d0d1c.jpg)

b  
![](images/22d461784277af8d72a78071f9ddead5739a0037194e564cdf42d5d3bc166f00.jpg)  
c

![](images/3249954035b3d26ac2076437a38373851802e2186de1c1d1aaa134372fb5e50a.jpg)

![](images/fa28a1a78a16f7d9dd36381a59a2f2ea6ecf6ef19d4d5a958c2d4b88c4c37b45.jpg)

Figure 3: COGNAC predictions of azeotropes and residue curve maps. $\mathbf { a } , \mathbf { b } ,$ Classification matrices for azeotropes in binary (a) and ternary (b) mixtures: predicted class against experimental class. The diagonal marks correct prediction of azeotrope existence and class with COGNAC; the corresponding matrices for HANNA and mod. UNIFAC are shown in Extended Data Fig. 7. Azeotrope types are minimum-boiling $( p _ { \mathrm { m a x } } )$ , maximum-boiling $( p _ { \mathrm { m i n } } )$ , double azeotrope (double), saddle-point (saddle), and zeotropic (none). n denotes the number of data points. $\mathbf { c } ,$ Predicted compositions at the azeotropic point, compared with experimental values, for all components in the mixture. Only azeotropes for which existence and type were predicted correctly are shown; n is the number of azeotropes, each contributing one point per component. d,e, Isothermal residue curve maps predicted by COGNAC for acetone + chloroform + hexane at 318.15 K (d) and for 2-propanol + nitromethane + 1,1,2,2-tetrachloro-1,2-difluoroethane (R112) at 347.65 K (e), a mixture outside the scope of mod. UNIFAC (no group decomposition for R112). Orange lines are residue curves, black lines are the distillation boundaries. The pressures at the vertices are the pure-component vapor pressures at the given temperature; the pressure next to a ternary azeotrope is its predicted value. Predicted singular points are marked by symbols encoding their type (unstable node, stable node, saddle). Experimental azeotropes are shown in red: the ternary azeotrope with it reported type, binary azeotropes measured within 5 K with the symbol of the nearest predicted singular point on that edge.

mod. UNIFAC applies to only 76% of them, where it correctly classifies 92.1% against 92.8% for COGNAC on the same subset (Extended Data Fig. 7). For the 686 ternary data points, COGNAC reaches 83.2% and HANNA 64.7% accuracy; mod. UNIFAC applies to 83% and classifies 78.5% of those correctly, against 85.5% for COGNAC. The three models are thus similar for binary azeotropes, and COGNAC is clearly ahead for ternary ones. The one class COGNAC misses is the rare double azeotrope: it recovers 2 of the 58 measured points that all three models evaluate, HANNA 36 and mod. UNIFAC 28 (Supplementary Table 2).

Fig. 3c shows that COGNAC also predicts the azeotropic compositions well. For the azeotropes that all three models classify correctly, the median absolute deviation over all the mole fractions is 0.020 (binary) and 0.022 mol mol<sup>−1</sup> (ternary) for COGNAC, 0.024 and 0.056 mol mol<sup>−1</sup> for HANNA, and 0.019 and 0.025 mol $\mathrm { m o l } ^ { - 1 }$ for mod. UNIFAC (Supplementary Note 6), whose parameters were fitted to the DDB azeotropic data [43].

Fig. 3d and Fig. 3e show isothermal residue curve maps (RCMs) for two ternary mixtures. They trace the liquid composition during open evaporation and visualize the distillation regions defined by the pure components and azeotropes. COGNAC predicts all three binary azeotropes and the ternary azeotrope for acetone + chloroform + hexane (Fig. 3d) with their measured types. 2-propanol + nitromethane + R112 (Fig. 3e), outside the scope of mod. UNIFAC, is split into three distillation regions by a ternary minimum-boiling azeotrope. The predicted ternary azeotropes deviate from the measured ones by at most 0.028 and 0.018 mol mol<sup>−1</sup> in any mole fraction, and the predicted pressure deviates by 2.0% and 1.8%, respectively.

## 2.3 Model interpretability

COGNAC learns interpretable atom-level volume and surface area parameters from the data, regularized towards COSMO cavity geometries (Methods). As an example, in Fig. 4a, the molecular graph of 4-methylhexanoic acid shows that an atom’s predicted exposed surface area decreases with its number of covalent bonds, from the terminal CH groups to the branching CH group and the carboxyl carbon. For 45,488 molecules outside our training and evaluation data sets, the predicted molecular volumes and surface areas deviate from the COSMO references by $1 1 . 0 \mathring \mathrm { A } ^ { 3 }$ and $1 4 . 1 \mathring \mathrm { A } ^ { 2 }$ on average $( 1 . 7 \mathring { \mathrm { A } } ^ { 2 }$ per atom) (Extended Data Fig. 8).

The pair interactions in COGNAC are modeled through the symmetric exchange energy $\Delta u _ { a b } = u _ { a b } - 0 . 5 ( u _ { a a } + u _ { b b } )$ , the difference between the unlike interaction a–b and the average of the two like interactions a–a and b–b (Methods); negative values mean that unlike contacts are favorable. For pairs of a hydrogen-bond donor and an acceptor atom, the sign of $\Delta u _ { a b }$ is not entirely free: during training, a regularizer penalizes positive, i.e., repulsive, values for these pairs (Methods). We learn their magnitudes, as well as the sign and magnitude for all other atom pairs, solely from the thermodynamic data. Consistent with chemical intuition, the interaction between the water oxygen and the alkyl group (including the implicitly treated hydrogen atoms) has the highest value; the lowest values occur for the amine nitrogen with hydrogen atoms, reflecting the hydrogen bonds formed (Fig. 4b). Figs. 4c and 4d quantify this dependence on the molecular neighborhood for a single atom type. For the carbon atoms of 1-hexanol and the oxygen of water (Fig. 4c), the exchange energy is $4 . 7 \mathrm { k J m o l ^ { - 1 } }$ for the carbon bonded to the hydroxyl group and $1 5 . 1 \mathrm { k J } \mathrm { m o l } ^ { - 1 }$ for the next one; from the third carbon on, it is close to the value of an alkane carbon $( 1 7 . 4 \mathrm { k J } \mathrm { m o l } ^ { - 1 }$ for the central $\mathrm { C H _ { 2 } }$ groups of hexane). Thus, the hydroxyl group’s influence extends over about two bonds. The CH<sub>2</sub> group next to other substituents also interacts more favorably with water oxygen than an alkane CH group (Fig. 4d). In mod. UNIFAC, by contrast, all carbon atoms of 1-hexanol are CH groups with identical interaction parameters, and the influence of a substituent on its neighboring $\mathrm { C H _ { 2 } }$ group is only captured where a larger predefined group, such as CH<sub>2</sub>Cl or $\mathrm { C H _ { 2 } N H _ { 2 } }$ , includes it.

a  
![](images/5cc13d6a2d1cef16cee4641311f92209da55c30d67f4cbe1eca91dddea8caff9.jpg)

![](images/3865104345497fb5fc25247b97ebe5e6ca23983b3fd59587a6b5bdd70e31a78e.jpg)

c  
![](images/2ef52a27514af1f3bdfa173f07ea1691e213373ba2335b8827258ea06df4fb40.jpg)

![](images/6eefc3b48e9cdfd9f5b4be552c598bbcdabf284a41174bb6106b639591c3c0d0.jpg)  
Figure 4: Interpretability of COGNAC. a, Molecular graph of 4-methylhexanoic acid. The surface area of the orange circles surrounding each atom is scaled with the predicted atom surface area from COGNAC; carbon-bound hydrogens are folded into their carbon atom, and the hollow gray circle has an area of $2 0 \mathring \mathrm { A } ^ { 2 }$ . b, Matrix of averaged exchange energies $\overline { { \Delta u _ { a b } } }$ at 298.15 K between selected atom types, obtained from all atom pairs of all binary mixtures in the data set, including pairs of atoms of the same component. Pairs of the same atom type are not shown; their $\overline { { \Delta u _ { a b } } }$ would not vanish either, since each type pools atoms with different molecular neighborhoods and $\Delta u _ { a b } = 0$ holds only for two atoms with identical embeddings. The color scale is centered at zero and scaled separately to the most negative and the most positive value. c, Exchange energy $\Delta u _ { a b }$ at 298.15 K between each carbon atom of 1-hexanol and the oxygen atom of water, over the number of bonds between the carbon atom and the hydroxyl oxygen. d, Exchange energy between the CH carbon atom bonded to the substituent X of 1-substituted hexanes $( \mathrm { C _ { 5 } H _ { 1 1 } - C H _ { 2 } - X } )$ and the oxygen atom of water. The dashed lines in c and d mark the central $\mathrm { C H _ { 2 } }$ groups of hexane (C3 and C4, equivalent by symmetry); the colors follow the color scale of b.

## 2.4 Application in process simulation

The ultimate purpose of a predictive $g ^ { \mathrm { E } }$ model is to support process design and optimization, for which it must be integrated into process simulators. Here, COGNAC has a significant advantage over other ML-based models: Starting from only the SMILES of the components, COGNAC predicts the geometry parameters $R _ { a }$ and $Q _ { o }$ <sub>a</sub> for every atom and the interaction parameters $a _ { a b } , b _ { a b } , c _ { a b }$ for every pair. These parameters can then be used in the standard mod. UNIFAC equation framework, which is implemented in all common process simulators and is cheap to evaluate.

To demonstrate COGNAC’s computational efficiency, we model batch distillations of a ternary and a five-component mixture in our laboratory-scale column [44] (Supplementary Note 7). The predictions agree well with the experimental data as shown in Figs. 5a and 5b. In Fig. 5c, we compare the time per integration step of the simulation using COGNAC with four other models for the same task. NRTL is the cheapest to evaluate, but it is not a predictive model and must be parameterized for every binary subsystem before the simulation. The computational costs of mod. UNIFAC and COGNAC differ because they have different numbers of geometric and interaction parameters, a direct consequence of the different numbers of structural groups/atoms considered in the models. The chosen five-component mixture is an extreme example where mod. UNIFAC heavily reuses the same groups in several components $\mathrm { ( e . g . , C H _ { 3 } }$ and $\mathrm { C H _ { 2 } }$ recur across the alcohols and the ketone), while COGNAC differentiates between almost all atoms. Nevertheless, the computational costs of using COGNAC and mod. UNIFAC are of the same order of magnitude: per integration step, COGNAC needs 1.9 (ternary mixture) and 2.4 times (five-component mixture) the time of mod. UNIFAC. This demonstrates COGNAC’s suitability for process simulations. We can further reduce computational cost by exploiting the symmetry of the COGNAC interaction parameters $( { \bf e . g . } , a _ { a b } = a _ { b a } )$ . However, we deliberately used the full symmetric matrix here to simulate its use in the standard mod. UNIFAC framework without customizations.

a  
![](images/0121ec2ab1bf585f61cd0eaae5d1ef490d5faf8613ed9c720493b8a8b266498b.jpg)  
b

![](images/91b550bb632ef836b081d86f08506c2a99a73f4503e272d4233d7335a4166664.jpg)

c  
![](images/202bf968eabcf3b91396bc1998d063741d9a05f3aed64a8dc4afba6f0c99c531.jpg)  
Figure 5: Dynamic simulation of a batch distillation using a Julia-based process simulator and the process model from Werner et al. [45]. a,b, Liquid mole fractions $x _ { i } ^ { \mathrm { r e s } }$ in the reboiler, i.e., in the liquid residue, over the course of the batch distillation of a ternary mixture (1-butanol + 2-propanol + water, head pressure 70 kPa; a) and a five-component mixture (methanol + ethanol + 2-propanol + 1-butanol + acetone, head pressure 50 kPa; b), simulated with COGNAC (lines). Symbols denote experimental data from a laboratory-scale plant [44]. c, Wall-clock time that the integrator spends per time step for five thermodynamic models and the two mixtures. Median of up to 10 repeated solves; the compilation time of the generated residual functions is measured separately and excluded.

## 3 Discussion

COGNAC shows that the accuracy gap between group-contribution methods like mod. UNIFAC and modern machine learning models like HANNA is not a property of the local-composition equations but of the rigid, predefined groups. Replacing the groups with atoms whose parameters depend on their molecular neighborhood, learned end-to-end from 822,780 binary data points, makes the unchanged mod. UNIFAC equations more accurate than HANNA for every data type of binary, ternary, and quaternary mixtures, including aqueous mixtures, and for azeotrope predictions, while keeping the simple deployment path of a group-contribution method. One network evaluation per mixture yields a parameter table valid for all temperatures and compositions, and a process simulation runs within a factor of about two of mod. UNIFAC and two orders of magnitude faster than with the HANNA ensemble. Because the activity coefficients are derived from the excess Gibbs energy, they strictly satisfy the Gibbs–Duhem equation, and the mod. UNIFAC equations ensure permutation symmetry and the correct pure-component limits. COGNAC robustly extends to any number of components without additional parameters or approximations.

The architecture also offers insight into the mixture physics with geometry parameters that were regularized against COSMO cavities and interaction parameters that align well with chemical intuition. This increases the interpretability over other “black-box” machine learning models, but also over the standard mod. UNIFAC, where effects like hydrogen bonding are only implicitly contained in the group parameters.

COGNAC covers all molecules and ionic liquids built from the twelve elements in our data set (Methods). Mod. UNIFAC describes only about half of the systems in our data; the other half, which contains many ionic liquids and multifunctional molecules, cannot be described with the published groups and parameters. Electrolytes and polymers are currently outside COGNAC’s domain; however, extensions analogous to those of UNIFAC for long-range electrostatic interactions and free-volume effects are possible. The idea behind COGNAC generalizes beyond mod. UNIFAC and applies to any group-contribution method.

We make the trained models and corresponding code available as open source (Code availability) and on our website MLPROP, which provides an interactive user interface to calculate activity coefficients, excess enthalpies, and phase equilibria with COGNAC. Users can also download the parameter tables for any specified system for direct implementation in process simulators. The model is also available as a Model Context Protocol (MCP) connector for thermodynamic calculations in large language models.

## 4 Methods

## 4.1 Equation framework

COGNAC is a hybrid model for excess Gibbs energy $g ^ { \mathrm { E } }$ that predicts the activity coefficient $\gamma _ { i }$ in liquid mixtures as a function of temperature $\dot { T }$ and composition $\mathbf { x } .$ The activity coefficient $\gamma _ { i }$ of component i in the mixture is formally separated into a combinatorial part, which accounts for differences in molecular size and shape, and a residual part, which accounts for the interactions between molecular sites and contains the entire excess enthalpy:

$$
\ln \gamma _ { i } = \ln \gamma _ { i } ^ { \mathrm { C } } + \ln \gamma _ { i } ^ { \mathrm { R } }\tag{2}
$$

COGNAC computes both contributions from learned molecular representations shared across the two terms. For the combinatorial part, we adopt the expression of mod. UNIFAC [3] with coordination number $z = 1 0$ :

$$
\ln \gamma _ { i } ^ { \mathrm { C } } = 1 - V _ { i } ^ { \prime } + \ln V _ { i } ^ { \prime } - { \frac { z } { 2 } } q _ { i } \left( 1 - { \frac { V _ { i } } { F _ { i } } } + \ln { \frac { V _ { i } } { F _ { i } } } \right) ,\tag{3}
$$

with the volume and surface area fractions:

$$
V _ { i } = { \frac { r _ { i } } { \sum _ { k } x _ { k } r _ { k } } } , \qquad V _ { i } ^ { \prime } = { \frac { r _ { i } ^ { 3 / 4 } } { \sum _ { k } x _ { k } r _ { k } ^ { 3 / 4 } } } , \qquad F _ { i } = { \frac { q _ { i } } { \sum _ { k } x _ { k } q _ { k } } } ,\tag{4}
$$

where $x _ { i }$ is the mole fraction of component i in the mixture. Here, $r _ { i }$ and $q _ { i }$ are the molecular volume and surface area parameters of component i, respectively. The exponent $3 / 4$ in $V _ { i } ^ { \prime }$ is the empirical correction to the Staverman–Guggenheim entropy of mixing that was introduced with mod. UNIFAC. Setting $V _ { i } ^ { \prime } \stackrel { } { = } V _ { i }$ recovers the Staverman–Guggenheim expression of the original UNIFAC exactly:

$$
\ln \gamma _ { i } ^ { \mathrm { { C , S G } } } = \ln \left( { \frac { \Phi _ { i } } { x _ { i } } } \right) + { \frac { z } { 2 } } q _ { i } \ln \left( { \frac { \theta _ { i } } { \Phi _ { i } } } \right) + \ell _ { i } - { \frac { \Phi _ { i } } { x _ { i } } } \sum _ { k } x _ { k } \ell _ { k } , \qquad \ell _ { i } = { \frac { z } { 2 } } ( r _ { i } - q _ { i } ) - ( r _ { i } - 1 ) ,\tag{5}
$$

with the volume fraction $\Phi _ { i } = x _ { i } V _ { i }$ and the surface area fraction $\theta _ { i } = x _ { i } F _ { i }$ , so that the $3 / 4$ exponent is the only difference between the two combinatorial terms.

The molecular parameters $r _ { i }$ and $q _ { i }$ are obtained by summing over the contributions $R _ { a }$ and $Q _ { a }$ of all nodes of the molecular graph of component i (all heavy atoms and explicit polar hydrogen atoms, see Section 4.2):

$$
r _ { i } = \sum _ { a \in \mathcal { V } _ { i } } R _ { a } , \qquad q _ { i } = \sum _ { a \in \mathcal { V } _ { i } } Q _ { a } .\tag{6}
$$

Here, $\nu _ { i }$ is the set of all nodes in the molecular graph of component i. Unlike UNIFAC, the $R _ { a }$ and $Q _ { a }$ values are not tabulated from predefined groups; instead, the model learns them end-to-end during training for each node, depending not only on atom type but also on its molecular neighborhood. As in, e.g., the UNIFAC groups $\mathrm { C H _ { 3 } , C H _ { 2 } , }$ , and CH, whose parameters include bonded hydrogen atoms, the volume and surface area contributions of hydrogen atoms bonded to carbon are implicitly absorbed into the predicted $R _ { a }$ and $Q _ { a }$ of the parent carbon atom. By contrast, hydrogen atoms bonded to non-carbon atoms are represented as explicit nodes and receive their own volume and surface area parameters (see Section 4.2).

The residual contribution of component i is computed as a sum of differences between the residual activity coefficients of its constituent atoms in the mixture and in the pure-component reference state:

$$
\ln \gamma _ { i } ^ { \mathrm { R } } = \sum _ { a \in \mathcal { V } _ { i } } Q _ { a } \left( \ln \Gamma _ { a } ^ { \mathrm { m i x } } - \ln \Gamma _ { a } ^ { ( i ) } \right) ,\tag{7}
$$

Here, $\Gamma _ { a }$ is the residual activity coefficient of a single surface segment of atom a, evaluated in the mixture $( \Gamma _ { a } ^ { \mathrm { m i x } } )$ and in the pure-component reference state of component i $( \Gamma _ { a } ^ { ( i ) } )$ . The weight $Q _ { a }$ is the surface area of atom a measured in units of the standard segment area. It thus counts the number of segments the atom contributes to the ensemble, so that every atom enters Eq. (7) in proportion to the surface it exposes. $\nu _ { i }$ denotes the set of atomic graph nodes of component i, i.e., its heavy atoms and explicit polar hydrogen atoms (see Section 4.2). The segment residual activity coefficients ln $\Gamma _ { a }$ follow from the closed-form UNIFAC expression, applied to atoms instead of structural groups. It requires two quantities, both of which are outputs of the model parts described in the following sections: the surface area fraction $\Theta _ { a }$ of every atom, which follows from the predicted surface area parameters $Q _ { a }$ (Eq. (6), Section 4.4), and the Boltzmann factor $\Psi _ { a b }$ of every atom pair, which follows from the predicted exchange energy $\Delta u _ { a b }$ (Eq. (11) below, Section 4.5). The surface area fractions in the mixture and in the pure-component reference state of component i are calculated via

$$
\Theta _ { a } ^ { \mathrm { m i x } } = \frac { Q _ { a } x _ { i ( a ) } } { \sum _ { j } x _ { j } q _ { j } } , \quad \Theta _ { a } ^ { ( i ) } = \frac { Q _ { a } } { q _ { i } } ,\tag{8}
$$

respectively, where i(a) denotes the component containing atom a. With the auxiliary sum

$$
s _ { b } = \sum _ { c \in \mathcal { V } } \Theta _ { c } \Psi _ { c b } ,\tag{9}
$$

the residual activity coefficient of a segment of atom a is

$$
\ln \Gamma _ { a } = 1 - \ln s _ { a } - \sum _ { b \in \mathcal { V } } \frac { \Theta _ { b } \Psi _ { a b } } { s _ { b } } .\tag{10}
$$

Weighted by $Q _ { a }$ in Eq. (7), this is the residual term of mod. UNIFAC, in which the surface area parameter appears as a prefactor of the bracket. COGNAC therefore uses the equations of mod. UNIFAC unchanged, with atoms taking the place of the structural groups. For $\Gamma _ { a } ^ { \mathrm { m i x } }$ , the sums run over $\textstyle \mathcal { V } = \bigcup _ { i } \dot { \mathcal { V } } _ { i }$ , i.e., over all atoms b of all components present in the mixture, with $\Theta ^ { \mathrm { m i x } }$ , so that the same equations apply to a mixture with any number of components; for the pure-component reference $\Gamma _ { a } ^ { \left( i \right) }$ , they run over $\nu _ { i }$ only, with $\bar { \Theta } ^ { ( i ) }$ . All quantities on the right-hand side follow directly from Θ and Ψ without iteration, so the residual term is cheap to evaluate and differentiable in a single forward pass. We also tested a self-consistent alternative to Eq. (10), the implicit closure of COSMOSPACE [39]; we describe and evaluate this model variant, COGNAC-IMP (IMP: implicit closure), in Supplementary Note 3, including the settings of its iterative solver (Supplementary Fig. 1) and its accuracy relative to COGNAC (Supplementary Table 1).

The Boltzmann factor $\Psi _ { a b }$ is formed from the symmetric exchange energy $\Delta u _ { a b }$ of the atom pair, i.e., the excess of the unlike a–b interaction over the mean of the two like (a–a and b–b) interactions, which is predicted by the model. Following mod. UNIFAC, its temperature dependence is a quadratic polynomial:

$$
\Psi _ { a b } = \exp \left( - \frac { \Delta u _ { a b } ( T ) } { R T } \right) , \qquad \frac { \Delta u _ { a b } ( T ) } { R } = A _ { a b } + B _ { a b } \tau + C _ { a b } \tau ^ { 2 } , \qquad \tau = \frac { T } { T _ { \mathrm { r e f } } } - 1 ,\tag{11}
$$

with $T _ { \mathrm { r e f } } = 2 9 8 . 1 5 \ : \mathrm { K }$ . The three coefficients $A _ { a b } , B _ { a b } ,$ , and $C _ { a b }$ are temperature-independent, carry units of kelvin, and are predicted per atom pair by a permutation-invariant network from the two atom embeddings alone. They are the atom-pair analog of the tabulated group interaction parameters of mod. UNIFAC. However, unlike the published mod. UNIFAC parameters, they are derived using a symmetric exchange formulation, which ensures $\Delta u _ { a b } = \Delta u _ { b a }$ and $\Delta u _ { a a } = 0$ and, hence, $\Psi _ { a a } = 1$ exactly and at every temperature. Section 4.5 gives a detailed derivation of the transformation between the two sets of coefficients and, with it, the route to a parameter table that a process simulator can use directly.

To summarize, the COGNAC equation framework requires the following quantities as input: (i) atom-level volume and surface area parameters $R _ { a }$ and $Q _ { a }$ for the combinatorial and residual contributions; (ii) atom-level pair exchange energies $\Delta u _ { a b } ( T )$ for the residual contribution. These quantities are predicted from the GNN-learned atom-level representations, as described in the following sections.

## 4.2 Molecular graphs

Each component i is represented as a molecular graph $\mathcal { G } _ { i } = ( \mathcal { V } _ { i } , \mathcal { E } _ { i } )$ constructed from a SMILES string using RDKit [31]. The node set $\nu _ { i }$ comprises all heavy atoms of the molecule and, additionally, every hydrogen atom that is not bonded to a carbon atom, which is included as an explicit node; we refer to these as polar hydrogen atoms in the following. We treat only hydrogen atoms bonded to carbon implicitly; they are encoded only as a feature of the parent carbon atom. This hybrid treatment keeps the graphs compact for apolar molecules while explicitly resolving these polar hydrogens, which can, e.g., act as hydrogen-bond donors: this way, water, for example, is represented by three nodes instead of a single one, and alcohols, carboxylic acids, and amines expose their donor hydrogens as separate interaction sites. This lets the model describe hydrogen bonding through attractive pair interactions between distinct donor and acceptor sites rather than absorbing these effects into the parameters of the parent heavy atoms.

The edge set $\mathcal { E } _ { i }$ contains all covalent bonds in component $i ,$ including bonds between explicit hydrogen nodes and their parent atoms, but excluding bonds between implicitly treated hydrogen atoms and their carbon atoms. For each node $a \in \mathcal { V } _ { i }$ , we compute six integer-valued categorical features: atomic number, total degree, formal charge, hybridization state, aromaticity, and the number of implicitly treated hydrogen atoms attached to the node. The total degree counts all bonded neighbors, including implicitly treated hydrogens. It is therefore not the degree of the node in the graph: a methyl carbon has total degree four but onl one graph neighbor. Note that for heavy atoms whose hydrogens are explicit nodes $( \mathrm { e . g . }$ , a hydroxyl oxygen), the last feature i always zero, as the hydrogens are instead resolved by the graph topology. For each covalent bond $a { - } b ,$ we encode three categorical edge features: bond type (single, double, triple, or aromatic), a binary conjugation flag, and a binary ring-membership flag. The hybridization state distinguishes sp, $\mathrm { s p ^ { 2 } , s p ^ { 3 } , s p ^ { 3 } d }$ , and $\mathrm { s p } ^ { 3 } \mathrm { \dot { d } ^ { 2 } }$ and assigns all other states, in particular the s state of the explicit hydrogen nodes, to an additional category. The bond type includes an additional category for other bond types, which does not occur in the data set used here.

## 4.3 Local message passing encoder

Each categorical atom and bond feature is mapped to a D-dimensional vector $( D = 2 5 6 )$ via a dedicated learned embedding table. The per-feature embeddings are summed and layer-normalized to give the initial atom states ${ \bf h } _ { a } ^ { ( 0 ) }$ and bond states ${ \bf e } _ { a b } ^ { ( 0 ) }$ :

$$
\mathbf { h } _ { a } ^ { ( 0 ) } = \mathrm { L N } \left( \sum _ { f = 1 } ^ { 6 } \mathbf { E } _ { f } \left[ \chi _ { a } ^ { ( f ) } \right] \right) , \quad \mathbf { e } _ { a b } ^ { ( 0 ) } = \mathrm { L N } \left( \sum _ { g = 1 } ^ { 3 } \mathbf { B } _ { g } \left[ \chi _ { a b } ^ { ( g ) } \right] \right) .\tag{12}
$$

Here, $\chi _ { a } ^ { ( f ) }$ is the integer category of the $f \mathrm { - t h }$ atom feature of atom a and $\chi _ { a b } ^ { ( g ) }$ that of the g-th bond feature of the bond $a { - } b$ (see Section 4.2). $\mathbf { E } _ { f }$ and $\mathbf { B } _ { g }$ are the learned embedding tables of these features, i.e., matrices with one D-dimensional row per category, and $\mathbf { E } _ { f } ^ { } [ \chi ]$ denotes the row of category $\chi .$ LN denotes layer normalization [46], which shifts and scales a vector to zero mean and unit variance over its $D$ entries and then applies a learned elementwise affine transformation.

We then refine the atom states with $L = 4$ message-passing layers. At layer $\ell ,$ each atom a aggregates messages from its covalently bonded neighbors ${ \mathcal { N } } ( a )$ and updates its state via a residual connection:

$$
{ \bf m } _ { a } ^ { ( \ell ) } = \sum _ { b \in \mathcal { N } ( a ) } \phi _ { \mathrm { m s g } } ^ { ( \ell ) } \left( { \bf h } _ { a } ^ { ( \ell ) } , { \bf h } _ { b } ^ { ( \ell ) } , { \bf e } _ { a b } ^ { ( 0 ) } \right) ,\tag{13}
$$

$$
\begin{array} { r } { \mathbf { h } _ { a } ^ { ( \ell + 1 ) } = \mathrm { L N } \Big ( \mathbf { h } _ { a } ^ { ( \ell ) } + \phi _ { \mathrm { u p d } } ^ { ( \ell ) } \Big ( \mathbf { h } _ { a } ^ { ( \ell ) } , \mathbf { m } _ { a } ^ { ( \ell ) } \Big ) \Big ) . } \end{array}\tag{14}
$$

Both $\phi _ { \mathrm { m s g } } ^ { ( \ell ) }$ and $\phi _ { \mathrm { u p d } } ^ { ( \ell ) }$ are two-layer multilayer perceptrons (MLPs) with sigmoid linear unit (SiLU) activations. The message (msg) function takes the concatenated central-atom state, neighbor state, and bond embedding as input $( 3 D  D  D ) ;$ the update (upd) function takes the concatenated atom state and aggregated message as input $( 2 D \to D \to D )$ . Bond embeddings ${ \bf e } _ { a b } ^ { ( 0 ) }$ are not updated across layers and serve as static edge context. After L layers, $\mathbf { h } _ { a } ^ { ( L ) } \in \mathbb { R } ^ { D }$ encodes the chemical identity of atom a and its molecular neighborhood up to a radius of $L = 4$ bonds. For brevity we use $\mathbf { h } _ { a } \equiv \mathbf { h } _ { a } ^ { ( L ) }$ below.

## 4.4 Atom-level volume and surface area

A shared two-output MLP head predicts two positive scalar quantities for each atom a from its final embedding $\mathbf { h } _ { a } \colon$ an effective cavity volume $V _ { a }$ and an effective cavity surface area $A _ { a } .$ . Softplus activations ensure positivity:

$$
V _ { a } = \mathrm { s o f t p l u s } ( f _ { V } ( { \bf h } _ { a } ) ) + \epsilon ,\tag{15}
$$

$$
A _ { a } = \mathrm { s o f t p l u s } ( f _ { A } ( \mathbf { h } _ { a } ) ) + \epsilon .\tag{16}
$$

Here, $f _ { V }$ and $f _ { A }$ denote the two outputs of this head, an MLP with two hidden layers of widths D and $D / 2$ and SiLU activations, and $\epsilon = 1 0 ^ { - 8 } ( \mathrm { i n } \mathring { \mathrm { A } } ^ { 3 }$ and $\mathring \mathrm { A } ^ { 2 }$ , respectively) is a small constant that keeps both quantities strictly positive.

These atom-level quantities are normalized with global reference scales $V _ { \mathrm { r e f } } > 0$ and $A _ { \mathrm { r e f } } > 0 ;$

$$
R _ { a } = \frac { V _ { a } } { V _ { \mathrm { r e f } } } , \quad Q _ { a } = \frac { A _ { a } } { A _ { \mathrm { r e f } } } \quad \mathrm { w i t h } \quad V _ { \mathrm { r e f } } = 2 0 . 1 8 \textup { \AA } ^ { 3 } , \quad A _ { \mathrm { r e f } } = 2 1 . 1 7 \textup { \AA } ^ { 2 } .\tag{17}
$$

We thereby follow the definition of the “standard segment” proposed by Abrams and Prausnitz for UNIQUAC [47] and define a single $\mathrm { C H _ { 2 } }$ group in an infinitely long polyethylene molecule as the reference. However, unlike the original UNIQUAC formulation, our reference values are not the van der Waals volume and surface area of the standard segment, but the effective cavity volume and surface area from COSMO results in the CHAOS database [35]. During training, we also regularize the predicted geometries against the corresponding COSMO cavity values from the CHAOS database, which grounds the learned geometries in physical reality. We constrain the two quantities at different levels of resolution: the surface area is regularized per atom, whereas the volume is regularized per molecule; i.e., we compare the predicted total cavity volume of a component against its COSMO cavity volume. The per-atom COSMO area targets are constructed consistently with the graph representation: the cavity areas of implicitly treated (carbon-bonded) hydrogens are folded onto their parent carbon atom. In contrast, each explicit polar hydrogen retains its own cavity area. Section 4.7 provides details on the regularization.

The molecular-level structural parameters $r _ { i }$ and $q _ { i }$ then follow by summing the atom-level contributions over the molecular graph, see Eq. (6).

## 4.5 Pairwise atom exchange energies

The three coefficients of the interaction polynomial in Eq. (11) are predicted by the interaction head, a permutation-invariant neural network with three outputs (an $\mathrm { M L P 3 } \dot { D } \dot {  } D  D / 2 \dot {  } 3$ with SiLU activations) that takes as input the pairwise feature vector

$$
\mathbf { f } _ { a b } = [ \mathbf { h } _ { a } + \mathbf { h } _ { b } , \quad | \mathbf { h } _ { a } - \mathbf { h } _ { b } | , \quad \mathbf { h } _ { a } \odot \mathbf { h } _ { b } ] \in \mathbb { R } ^ { 3 D } ,\tag{18}
$$

where ⊙ denotes the elementwise product. All three components of $\mathbf { f } _ { a b }$ are symmetric under permutation of $a$ and $b ,$ so the predicted coefficients, and hence $\Delta u _ { a b }$ , are symmetric. Note that temperature is not among the inputs: it enters only through the polynomial in Eq. (11), so the head predicts a temperature-independent parameter triple per atom pair.

In the first step, the interaction head outputs the three raw parameters $\tilde { A } _ { a b } , \tilde { B } _ { a b }$ , and ${ \tilde { C } } _ { a b }$ . Using the symmetric exchange formulation, these are transformed as follows:

$$
\begin{array} { r l r } { X _ { a b } = \tilde { X } _ { a b } - \frac { 1 } { 2 } \left( \tilde { X } _ { a a } + \tilde { X } _ { b b } \right) , } & { { } } & { X \in \{ A , B , C \} , } \end{array}\tag{19}
$$

where the like-pair terms $\tilde { X } _ { a a }$ and unlike-pair terms $\tilde { X } _ { a b }$ are obtained from the same head. We apply the subtraction separately to each of the three channels. Because the mod. UNIFAC polynomial is linear in the coefficients, this is equivalent to applying the exchange formulation on the interaction energy itself, with $u _ { a b } ( T ) / R = \tilde { A } _ { a b } + \tilde { B } _ { a b } \tau + \tilde { C } _ { a b } \tau ^ { 2 } ;$

$$
\begin{array} { r l } {  { \frac { \Delta u _ { a b } ( T ) } { R } = \frac { 1 } { R } [ u _ { a b } ( T ) - \frac { 1 } { 2 } ( u _ { a a } ( T ) + u _ { b b } ( T ) ) ] } } \\ & { = [ \tilde { A } _ { a b } + \tilde { B } _ { a b } \tau + \tilde { C } _ { a b } \tau ^ { 2 } ] - \frac { 1 } { 2 } [ \tilde { A } _ { a a } + \tilde { B } _ { a a } \tau + \tilde { C } _ { a a } \tau ^ { 2 } + \tilde { A } _ { b b } + \tilde { B } _ { b b } \tau + \tilde { C } _ { b b } \tau ^ { 2 } ] } \\ & { = \tilde { A } _ { a b } - \frac { 1 } { 2 } ( \tilde { A } _ { a a } + \tilde { A } _ { b b } ) + [ \tilde { B } _ { a b } - \frac { 1 } { 2 } ( \tilde { B } _ { a a } + \tilde { B } _ { b b } ) ] \tau + [ \tilde { C } _ { a b } - \frac { 1 } { 2 } ( \tilde { C } _ { a a } + \tilde { C } _ { b b } ) ] \tau ^ { 2 } } \\ & { = A _ { a b } + B _ { a b } \tau + C _ { a b } \tau ^ { 2 } } \end{array}\tag{20}
$$

$\Delta u _ { a b }$ is thus an exchange energy as defined in COSMOSPACE [39], with its temperature dependence constrained to the quadratic form of mod. UNIFAC. The coefficients of Eq. (19) are converted from the reduced basis $( 1 , { \bar { \tau } } , \tau ^ { 2 } )$ used internally to the published basis $( 1 , T , T ^ { 2 } )$ by the exact change of basis

$$
a _ { a b } = A _ { a b } - B _ { a b } + C _ { a b } , \quad \quad b _ { a b } = \frac { B _ { a b } - 2 C _ { a b } } { T _ { \mathrm { r e f } } } , \quad \quad c _ { a b } = \frac { C _ { a b } } { T _ { \mathrm { r e f } } ^ { 2 } } ,\tag{21}
$$

where $\left( a _ { a b } , b _ { a b } , c _ { a b } \right)$ are the coefficients for a mod. UNIFAC group interaction table in which every heavy atom and every explicit hydrogen atom of the mixture is a group. Because the exchange subtraction is performed before conversion, the exported coefficients always satisfy $a _ { a a } = b _ { a a } = c _ { a a } = 0$ . Note that, e.g., the coefficients between two CH2 groups are not necessarily zero if their molecular neighborhoods are different. The obtained interaction coefficient matrix is always symmetric, i.e., $a _ { a b } = a _ { b a }$ $b _ { a b } = b _ { b a } .$ , and $c _ { a b } = c _ { b a }$

The reduced basis in Eq. (11) is necessary for stable training, since over the data’s temperature range τ stays of order one whereas T and $T ^ { 2 }$ are of order $\mathrm { \bar { 1 0 ^ { 2 } } }$ and $1 0 ^ { 5 }$ , respectively. As a single network emits all three coefficients, the original basis makes the three output channels differ in gradient magnitude by up to five orders of magnitude, and the resulting $\bar { \Delta u _ { a b } } ( T )$ is already far outside the physical range at initialization; in the reduced basis, the three channels share one scale. The output weights of the B and C channels are initialized to zero, so the model starts with a temperature-independent interaction parameter and develops temperature dependence only where the data require it. This mirrors the published table of mod. UNIFAC, where most group pairs have $c = 0 ,$ , and many have $b = 0$

While the message-passing encoder sees each component’s molecular graph separately, the interaction head operates on the full mixture, which includes all atomic nodes from all components. Thus, every pair of atoms in the mixture is evaluated, irrespective of whether the two atoms belong to the same component or to different ones, and including the self-pairs $a = b ,$ , which supply the like-pair terms of Eq. (19). This is required because the interaction between all atoms in the mixture must be calculated to obtain the residual activity coefficients, following the equation framework described in Section 4.1. We evaluate the interaction head a second time on each isolated component, which supplies the $\Psi _ { a b }$ for $\Gamma _ { a } ^ { \left( i \right) }$ in the pure-component reference state.

Because donor hydrogens are explicit nodes, the model resolves a hydrogen bond between a donor hydrogen and an acceptor atom as an explicit pair interaction. In our experiments, we noticed that the model did not always assign this interaction the expected strongly attractive (i.e., negative) value for $\Delta u _ { a b } .$ , but, due to its flexibility, it occasionally spread this interaction onto other attached atoms. For this reason, we introduced a regularization term that encourages the correct sign of hydrogen-bond interactions (see Section 4.7). Aside from this, we determine all learned interactions from experimental data alone.

## 4.6 Data processing and splitting

We obtained all experimental data used to train and evaluate the model from the Dortmund Data Bank (DDB) [33]. Vapor–liquid equilibrium data were adopted from the $\mathbf { \hat { \mu } } ^ { 6 6 } \mathbf { V } \mathbf { L E } ^ { 3 }$ and $\mathrm { \ddot { H P V } ^ { 5 } }$ (high-pressure equilibria) databases. We also used the following databases: $\ " { \bf A } { \bf C } { \bf T } ^ { \bf 5 }$ (activity coefficients at infinite dilution), “LLE” (liquid–liquid equilibria), “HE” (excess enthalpies), and “SLE” (solid–liquid equilibria). We used Antoine parameters to calculate pure-component vapor pressures (supplemented by measured pure-component points where the VLE data sets report them). We also obtained melting temperatures (“PCP\_MPT”) and melting enthalpies (“PCP\_HFS”) from the DDB.

In the first step of data preprocessing, we unified all extracted data by column name, converted to SI units as needed, and reordered components so that all data for the same system used the same component order. One exception is the ACT data, where component order matters and the second component was always defined as the one at infinite dilution. For every molecule in the data set, we generated a canonical SMILES using RDKit [31]. In this process, we excluded all rows containing a component without a valid SMILES. The data set covers molecular components and ionic liquids; all molecules are built from the elements H, B, C, N, O, F, Si, P, S, Cl, Br, and I. In the following, the type-specific processing of the raw data is described for binary mixtures, which form the entire training data. We used the data for ternary and quaternary mixtures only for evaluation and converted them in the same way, extending them to the additional components. We carried out the manual inspection described below only for binary data.

VLE and HPV. In the VLE and HPV data, only points at finite dilution were considered, and the pure-component edges $( x _ { 1 } \in \{ 0 , 1 \}$ or $y _ { 1 } \in \{ 0 , 1 \} )$ ) were removed. Where an isothermal data set itself reports a measured pure-component point $( x _ { 1 } \in \{ \bar { 0 } , 1 \bar { \} } )$ ), we used its total pressure directly as the pure-component vapor pressure $p _ { i } ^ { \mathrm { s } }$ . Otherwise, we computed $p _ { i } ^ { \mathrm { s } }$ from the DDB Antoine equation

$$
\log _ { 1 0 } \frac { p _ { i } ^ { \mathrm { s } } } { \mathrm { m m H g } } = A _ { i } - \frac { B _ { i } } { C _ { i } + \vartheta / ^ { \circ } \mathrm { C } } ,\tag{22}
$$

using only parameter sets whose temperature range contains $\vartheta .$ For isobaric data sets that report a pure-component point, we recalibrated the Antoine equation to it: we divided the measured pressure at this point by the Antoine vapor pressure at its measured temperature (i.e., the pure-component boiling temperature) and multiplied $p _ { i } ^ { \mathrm { s } }$ of all data points of the data set by this factor. This is equivalent to shifting $A _ { i }$ such that the Antoine equation passes through the measured point while retaining its temperature dependence. Using the edge points and recalibration substantially improved the quality of the derived activity coefficients, especially near the pure-component limit, where they must converge to unity. We dropped data points for which $p _ { i } ^ { \mathrm { s } }$ could be obtained in neither way.

The activity coefficients were derived using extended Raoult’s law:

$$
\gamma _ { i } = { \frac { p y _ { i } } { p _ { i } ^ { \mathrm { s } } x _ { i } } }\tag{23}
$$

Because this assumes an ideal vapor phase, we excluded all data points with total pressure above 10 bar.

Rows are then classified into two data types for the training: First, a row with $T , p , x _ { 1 } , y _ { 1 }$ , and both vapor pressures $p _ { 1 } ^ { s } ( T ) , p _ { 2 } ^ { s } ( T )$ at the reported temperature yields both activity coefficients and is labeled as TPXY. Second, we label a row missing vapor-phase composition information as TPX and retain it without activity coefficients, since it still contains the total pressure as a valid training target. All other rows (e.g., without pressure information) were dropped.

LLE. For the LLE data, the DDB reports the temperature $T ,$ the pressure p (if no value was reported, we assumed atmospheric pressure), and the compositions of the liquid phases in equilibrium, $\bar { \mathbf { \Phi } } _ { x _ { 1 } ^ { \prime } }$ and $\dot { x _ { 1 } ^ { \prime \prime } }$ . If the composition of both phases is reported in the DDB, we processed the data such that $\dot { x } _ { 1 } ^ { \prime } < x _ { 1 } ^ { \prime \prime }$ always holds. If only one phase composition was reported, we kept the DDB phase assignment, as it was correct in almost all cases. Because the COGNAC framework does not include pressure dependence, we excluded data points with pressures above 50 bar.

SLE. We used the following equation to calculate activity coefficients from the SLE data:

$$
\gamma _ { i } = \frac { 1 } { x _ { i } } \exp \left[ \frac { \Delta h _ { \mathrm { f u s } , i } } { R T _ { \mathrm { f u s } , i } } \frac { T - T _ { \mathrm { f u s } , i } } { T } \right] .\tag{24}
$$

We thereby used the average of all reported melting temperatures $T _ { \mathrm { f u s } , i }$ and enthalpies of fusion $\Delta h _ { \mathrm { f u s } , i }$ for pure component i in the DDB (below 50 bar or with no reported pressure). This formula holds only on a liquidus branch where the pure component i is in equilibrium with the liquid phase. To identify the branches, we manually checked all reported mixtures with at least 5 data points. We saved the composition ranges of each eligible liquidus branch (i.e., the range for which the liquid is actually in equilibrium with a pure component solid phase). We used only data from these branches to calculate SLE activity coefficients; all other data were dropped.

Manual inspection. Before training, we visually inspected the binary data mixture by mixture, with simple fits to the data of each mixture as a guide. We rejected individual outliers as well as entire data sets that were internally inconsistent or contradicted the other data of the mixture, and accepted the remaining data points. For VLE, HE, and LLE, mixtures whose data are near-ideal or closely described by such a fit were accepted automatically. Only mixtures with enough data points for such a comparison were inspected (at least 5 for SLE, HE, and LLE, and at least 10 for VLE and ACT); the data points of all other mixtures are labeled “unreviewed” and retained. For SLE, the inspection also identified the composition range of each liquidus branch, and only these branches are used. No model predictions were used. Details are given in Supplementary Note 8.

Data split. The pool of data for binary mixtures used for training and evaluating COGNAC contains all accepted and unreviewed data points (i.e., all data points except those rejected in the review); in total, 822,780 data points for 48,432 binary systems were used. To assess COGNAC’s prediction performance on the binary data, we split this data set system-wise into ten folds for ten-fold cross-validation. Folds were assigned per pair of canonical SMILES such that the same binary systems never fall into different folds. The assignment was stratified such that every data type is distributed evenly across the folds (greedy assignment, random seed 0). In run k of the cross-validation, fold k is used as the test set, fold $k + 1$ (modulo ten) as the validation set, and the remaining eight folds for training. Every system is thus predicted exactly once by a model that has not seen ${ \mathrm { i t } } ,$ and all binary results in this work are these out-of-fold predictions. Supplementary Tables 8 and 9 summarize the number of data points and systems per fold and data type. All binary results are evaluated on the accepted data points only, except for the ACT data, for which most systems contain too few data points for a review, and accepted and unreviewed data points are both used; results including the unreviewed data points of the other data types are given in Supplementary Note 8 (Supplementary Fig. 2 and Supplementary Table 10).

The ternary and quaternary data sets are evaluated on all data points from the DDB, except for 18 ternary LLE data points rejected in an inspection of the ternary LLE data. The quaternary data set comprises 21,221 TPXY, TPX, and LLE data points of 1,032 systems and is evaluated in Extended Data Fig. 2. The ACT data of the ternary data set (solute at infinite dilution in a mixed solvent) also include 1,674 data points in which the solvent is a pure component; since these are binary activity coefficients at infinite dilution, they were excluded from all ternary results. For the evaluation of ternary and quaternary mixtures, the prediction of ternary azeotropes, the interpretability analysis, and the process simulations, COGNAC was retrained on the full binary data (full-data model), using nine folds (739,675 data points) for training and the tenth fold (83,105 data points) as the validation set, without a test set, since none of these evaluations use binary test data.

Both ablation studies follow the protocol above: the architectural ablation (Supplementary Note 2) uses ten-fold cross-validation on the binary mixtures, and the hydrogen ablation (Extended Data Fig. 4) additionally uses the full-data model for the ternary mixtures.

## 4.7 Model training

COGNAC is trained jointly on six types of thermodynamic data for binary mixtures, which differ in what observable is directly measured and how the model prediction enters the loss. In all cases, the Huber (smooth L1) loss function is used:

$$
\ell _ { \delta } ( r ) = \left\{ \begin{array} { l l } { r ^ { 2 } } & { | r | < \delta , } \\ { | r | - \displaystyle \frac { \delta } { 2 } } & { | r | \geq \delta , } \end{array} \right.\tag{25}
$$

with r being the loss quantity and the threshold $\delta \ : = \ : 1$ unless stated otherwise. The Huber loss matches the curvature of mean squared error near the origin while degrading to mean absolute error for large residuals, reducing sensitivity to outliers in experimental data. Throughout this section, a hat denotes a thermodynamic property predicted by the model (e.g., ln $\widehat { \gamma } )$ to distinguish it from the corresponding experimental value. Internal model parameters, such as the cavity geometries $A _ { a }$ and $r _ { i }$ or the interaction energies $\Delta u _ { a b } .$ , have no experimental counterpart and are therefore written without a hat.

Let $\boldsymbol { B } _ { D }$ denote the set of data points of type D in a mini-batch, and let $b \in B _ { D }$ be one such data point. The contribution of data type D is the mean of a data-point-wise loss $\ell _ { b } ^ { D }$ over $\boldsymbol { B } _ { D }$

$$
\mathcal { L } _ { D } = \frac { 1 } { | B _ { D } | } \sum _ { b \in B _ { D } } \ell _ { b } ^ { D } ,\tag{26}
$$

which defines the terms $\mathcal { L } _ { \mathrm { T P X Y } } , \mathcal { L } _ { \mathrm { S L E } } , \mathcal { L } _ { \mathrm { A C T } } , \mathcal { L } _ { \mathrm { T P X } } , \mathcal { L } _ { \mathrm { H E } }$ , and $\mathcal { L } _ { \mathrm { L L E } }$ of the total training objective, Eq. (42). Targets are missing in parts of the data: an SLE data point yields only the activity coefficient of the component that is in equilibrium with its pure solid, and an LLE data point often reports only one branch of the tie line. Wherever a data point carries several targets, $\ell _ { b } ^ { D }$ is therefore the mean of the expression given below over the finite targets of that data point. Since Eq. (26) performs the same reduction for every data type, we state only the per-target expressions below.

TPXY and SLE. Both data types provide activity coefficients at a given composition and temperature, which the model predicts directly, so both enter through the same residual

$$
\ell _ { 1 } \left( \ln \widehat { \gamma } _ { b , i } - \ln \gamma _ { b , i } \right) , \qquad i \in \{ 1 , 2 \} .\tag{27}
$$

For the TPXY data, both ln $\gamma _ { 1 }$ and ln $\gamma _ { 2 }$ are obtained from the VLE measurements through the extended Raoult’s law, Eq. (23) (Section 4.6). The SLE data give, via Eq. (24), the activity coefficient of the component in equilibrium with its pure solid, i.e., in general only for one of the two components. Both terms enter the total objective unweighted and together anchor the scale of the predicted activity coefficients. For the TPXY term, the mean in Eq. (26) is taken directly over all activity coefficients in the mini-batch, so that each activity coefficient is weighted equally.

ACT. Each ACT data point directly provides the activity coefficient at infinite dilution, ln $\gamma _ { 2 } ^ { \infty }$ , of a solute (component 2) in a solvent (component 1). For training on these data, the model is evaluated at the composition $x _ { 1 } = 1$ , and only the predicted activity coefficient of the solute enters the loss,

$$
\ell _ { 1 } \left( \ln \widehat { \gamma } _ { b , 2 } ^ { \infty } - \ln \gamma _ { b , 2 } ^ { \infty } \right) .\tag{28}
$$

TPX. TPX data are bubble-line measurements that report the total pressure p at a given liquid composition $x _ { 1 }$ and temperature T, but not the vapor composition, so they cannot be used to derive activity coefficients. Assuming an ideal vapor phase, the total pressure predicted from the predicted activity coefficients is

$$
\hat { p } = p _ { 1 } ^ { \mathrm { s } } ( T ) x _ { 1 , b } \hat { \gamma } _ { 1 , b } + p _ { 2 } ^ { \mathrm { s } } ( T ) x _ { 2 , b } \hat { \gamma } _ { 2 , b } ,\tag{29}
$$

and the residual is taken on its logarithm, so that data points spanning several orders of magnitude in pressure contribute comparably,

$$
\ell _ { 1 } ( \ln ( \hat { p } / \mathrm { b a r } ) - \ln ( p / \mathrm { b a r } ) ) .\tag{30}
$$

HE. Excess enthalpy data, $h ^ { \mathrm { E } }$ in kJ $\mathrm { m o l ^ { - 1 } }$ , relate to $g ^ { \mathrm { E } }$ through the Gibbs–Helmholtz equation

$$
h ^ { \mathrm { E } } = - R T ^ { 2 } \Bigg ( \frac { \partial \left( g ^ { \mathrm { E } } / R T \right) } { \partial T } \Bigg ) _ { p , x } .\tag{31}
$$

Since $g ^ { \mathrm { E } } / R T = \textstyle \sum _ { i } x _ { i }$ ln $\gamma _ { i }$ and the model predicts ln $\gamma _ { i }$ as a differentiable function of $T ,$ , the derivative in Eq. (31) is evaluated by automatic differentiation with respect to the input temperature,

$$
\hat { h } _ { b } ^ { E } = - R T _ { b } ^ { 2 } \frac { \partial } { \partial T } \left( \sum _ { i } x _ { i , b } \ln \widehat { \gamma } _ { i , b } \right) ,\tag{32}
$$

and the residual is

$$
\ell _ { 1 } \left( \hat { h } _ { b } ^ { E } - h _ { b } ^ { \mathrm { E } } \right) .\tag{33}
$$

This term back-propagates gradients through the temperature derivative of the model output, so it trains temperature sensitivity directly, rather than only through the temperature dependence of the activity coefficients.

LLE. Liquid–liquid equilibrium data provide the compositions of the two coexisting liquid phases, $x _ { 1 } ^ { \prime }$ and $x _ { 1 } ^ { \prime \prime }$ . The model does not predict these compositions directly; they follow from the phase equilibrium conditions applied to the predicted liquid-phase Gibbs energy of mixing,

$$
{ \frac { \Delta g _ { \mathrm { m i x } } } { R T } } = \sum _ { i } x _ { i } \left( \ln x _ { i } + \ln \gamma _ { i } ( \mathbf { x } , T ) \right) ,\tag{34}
$$

and must therefore be obtained from it by a solver that is itself differentiable. We use the convex envelope method (CEM) [36] in a first step to obtain an approximate but non-differentiable solution. We evaluate the $\Delta g _ { \mathrm { m i x } }$ graph on a uniform grid of $G = 4 0 1$ compositions in one batched model call. The phase split is read off the lower convex hull of the resulting curve: an interval over which the hull lies strictly below $\Delta g _ { \mathrm { m i x } }$ is a miscibility gap. Its two endpoints are the binodal compositions. Two separate miscibility gaps at the same temperature appear as two separate intervals, since the hull touches $\Delta g _ { \mathrm { m i x } }$ between them; in this rare case, we compare the widest gap with the data point. Subsequently, we use a single Newton step to obtain a differentiable solution. For a data point $b ,$ we write the two unknowns of the binary phase split as $\bar { \mathbf { w } } _ { b } = ( x _ { 1 , b } ^ { \prime } , x _ { 1 , b } ^ { \prime \prime } )$ , with the corresponding phase compositions $\mathbf { x } _ { b } ^ { \prime } = ( x _ { 1 , b } ^ { \prime } , 1 - x _ { 1 , b } ^ { \prime } )$ and $\mathbf { x } _ { b } ^ { \prime \prime } = ( x _ { 1 , b } ^ { \prime \prime } , 1 - x _ { 1 , b } ^ { \prime \prime } )$ . The phase split predicted by the model with parameters $\theta$ is a root of the isoactivity residual

$$
F _ { i } ( \mathbf { w } _ { b } ; \theta ) = x _ { i , b } ^ { \prime } \gamma _ { i } ( \mathbf { x } _ { b } ^ { \prime } , T _ { b } ; \theta ) - x _ { i , b } ^ { \prime \prime } \gamma _ { i } ( \mathbf { x } _ { b } ^ { \prime \prime } , T _ { b } ; \theta ) , \qquad i = 1 , 2 ,\tag{35}
$$

and the grid result $\mathbf { w } _ { \mathrm { C E M } , b }$ is refined by one Newton step on this residual,

$$
\hat { \mathbf { w } } _ { b } = \mathbf { w } _ { \mathrm { C E M } , b } - \mathbf { J } _ { b } ^ { - 1 } F \big ( \mathbf { w } _ { \mathrm { C E M } , b } ; \theta \big ) , \qquad ( \mathbf { J } _ { b } ) _ { i j } = \left. \frac { \partial F _ { i } } { \partial w _ { j } } \right| _ { \mathbf { w } _ { \mathrm { C E M } , b } } ,\tag{36}
$$

in which ${ \bf w } _ { \mathrm { C E M } , b }$ and $\mathbf { J } _ { b }$ are treated as constants and only $F$ retains its dependence on θ. The result $\hat { \mathbf { w } } _ { b } = ( \hat { x } _ { 1 , b } ^ { \prime } , \hat { x } _ { 1 , b } ^ { \prime \prime } )$ is the predicted phase split.

The Newton step serves two purposes. First, the grid result is only accurate to the grid spacing and, being a grid point, does not change with $\theta ,$ so that it provides no gradient; the Newton step moves the prediction towards the root of $F ,$ so that the loss is evaluated at the composition to which its gradient refers. Second, differentiating Eq. (36) by ordinary backpropagation yields $\mathrm { d } \hat { \mathbf { w } } _ { b } / \mathrm { d } \theta = - \mathbf { J } _ { b } ^ { - 1 } \partial F / \partial \theta$ , which is the implicit function theorem gradient $- \left( \partial F / \partial \mathbf { w } \right) ^ { - 1 } \partial F / \partial \theta$ of the phase split, with the Jacobian evaluated at the grid result. It equals the exact implicit derivative only when the grid result coincides with the root; its accuracy therefore depends on the grid error and on the conditioning of J. A small Tikhonov term $( 1 0 ^ { - 4 } \mathbf { I } )$ is added to J before it is inverted, the step is restricted to a trust region of 0.05 in mole fraction, and data points whose predicted gap is narrower than 0.01 are left at their grid value, since J becomes singular as the critical point is approached. Because the condition number of J reaches $\mathcal { O } ( 1 0 ^ { 4 } )$ for strongly associating systems, we evaluate and differentiate this part of the computational graph in full single precision, even when reduced-precision matrix multiplication is enabled elsewhere.

The residual is the difference between the two predicted phase compositions and the measured ones. Since a deviation in mole fraction can never exceed one, the default threshold $\delta = 1$ would keep the Huber loss quadratic over the entire possible range; we therefore use $\delta = 0 . 3 5$ , so that large deviations enter linearly,

$$
\ell _ { 0 . 3 5 } \bigl ( \hat { x } _ { 1 , b } ^ { \prime } - x _ { 1 , b } ^ { \prime } \bigr ) \quad \mathrm { a n d } \quad \ell _ { 0 . 3 5 } \bigl ( \hat { x } _ { 1 , b } ^ { \prime \prime } - x _ { 1 , b } ^ { \prime \prime } \bigr ) .\tag{37}
$$

If the hull finds no miscibility gap, the model predicts a homogeneous mixture, no phase compositions exist, and the data point contributes zero. This has an important consequence: $\mathcal { L } _ { \mathrm { L L E } }$ can refine a phase split that the model already predicts, but it can never create one: the detection of the gap from the convex hull is a discrete operation that is not differentiable, so the grid in the first step is evaluated without tracking gradients and no gradient reaches $\theta$ through the detection of the gap itself. At the beginning of training, the predicted $\Delta g _ { \mathrm { m i x } }$ is close to ideal and hence convex everywhere; no data point splits, and $\mathcal { L } _ { \mathrm { L L E } }$ contributes no gradient at all, so that liquid–liquid demixing is initially learned only indirectly, through the other data types.

In our previous work [28], we addressed this problem by introducing a “Gibbs loss” that encourages the model to produce a correctly shaped $\Delta g _ { \mathrm { m i x } }$ by enforcing a negative second derivative on LLE data points during training. This idea was further developed by Ben Hicham et al. [48], who constrain the local curvature of $\Delta g _ { \mathrm { m i x } }$ at the compositions at which a phase split was measured: both measured binodal compositions should lie in a locally stable region of the surface and the midpoint of the gap, $\begin{array} { r } { z _ { b } = \frac { 1 } { 2 } ( x _ { 1 , b } ^ { \prime } + x _ { 1 , b } ^ { \prime \prime } ) } \end{array}$ , in an unstable one. While this condition is neither necessary nor sufficient for a phase split [48], the penalty based on it provides a gradient towards demixing where the model does not yet predict a phase split. For a binary mixture the composition Hessian of $\Delta g _ { \mathrm { m i x } }$ has a single independent element,

$$
H _ { b } ( x _ { 1 } ) = \frac { \partial ^ { 2 } } { \partial x _ { 1 } ^ { 2 } } \bigg ( \frac { \Delta g _ { \mathrm { m i x } } } { R T } \bigg ) _ { b } ,\tag{38}
$$

and the data-point-wise loss is the sum of three hinge penalties,

$$
\ell _ { b } ^ { \mathrm { H } } = \mathrm { R e L U } \big ( \varepsilon _ { \mathrm { H } } - H _ { b } ( x _ { 1 , b } ^ { \prime } ) \big ) + \mathrm { R e L U } \big ( \varepsilon _ { \mathrm { H } } - H _ { b } ( x _ { 1 , b } ^ { \prime \prime } ) \big ) + \mathrm { R e L U } \big ( H _ { b } ( z _ { b } ) + \varepsilon _ { \mathrm { H } } \big ) ,\tag{39}
$$

where $\varepsilon _ { \mathrm { H } } = 0 . 1$ is the margin each of the three inequalities must exceed before the corresponding penalty vanishes. $H _ { b }$ is computed exactly by twofold automatic differentiation of the predicted $\Delta g _ { \mathrm { m i x } }$ with respect to composition, so that Eq. (39) remains differentiable with respect to $\theta .$

This term is the counterpart of $\mathcal { L } _ { \mathrm { L L E } }$ , not a duplicate. Because it is evaluated at measured compositions, it has a gradient on precisely those data points for which the solver found no split and $\mathcal { L } _ { \mathrm { L L E } }$ is identically zero, including at initialization. It is therefore applied to every LLE data point during training, not only to those the solver failed to split. The interplay of the two terms is directly visible in the training history: the logged $\mathcal { L } _ { \mathrm { L L E } }$ rises over the first few epochs before it falls, because data points that previously predicted no split begin to split and enter the loss for the first time.

Much of the LLE data is single-branch mutual-solubility data for which only one endpoint is reported, so $z _ { b }$ is undefined. Such a data point contributes only the hinge penalty at the endpoint that was measured. Since an ideal mixture already satisfies this penalty, it does not encourage demixing by itself; single-branch data points shape the predicted gap only through $\dot { \mathcal { L } } _ { \mathrm { L L E } }$ once the model predicts a split.

Geometry regularization. We regularize the predicted cavity geometries against the COSMO reference values in the CHAOS database [35] (see Section 4.4) during training. The penalty is taken in the logarithm, so that it constrains the relative rather than the absolute deviation and hence does not concentrate on the largest atoms and molecules,

$$
\mathcal { L } _ { \mathrm { g e o m } } = \left. \left( \ln A _ { a } - \ln A _ { a } ^ { \mathrm { C O S M O } } \right) ^ { 2 } \right. _ { a } + \left. \left( \ln ( r _ { i } V _ { \mathrm { r e f } } ) - \ln V _ { i } ^ { \mathrm { C O S M O } } \right) ^ { 2 } \right. _ { i } ,\tag{40}
$$

where $\langle \cdot \rangle _ { a }$ averages over the atoms and $\langle \cdot \rangle _ { i }$ over the components of the mini-batch for which a reference value is available. As discussed in Section 4.4, the surface area is constrained per atom whereas the cavity volume is constrained per molecule, i.e., on the summed molecular volume $r _ { i } V _ { \mathrm { r e f } }$ . COSMO results are available only for a subset of molecules in the training set; atoms and molecules without a reference value are masked out and contribute nothing to Eq. (40), so the loss aligns predicted geometries with COSMO cavities on the remaining molecules as a generalization rather than a fit.

Hydrogen-bond sign regularizer. Under the explicit polar-hydrogen graph representation (Section 4.2), a hydrogen bond between a donor and an acceptor atom is resolved as an individual atom-pair interaction, which should therefore be attractive, i.e., $\Delta u _ { a b } < 0 ( \mathrm { E q . } ( 1 1 ) )$ ). Without an explicit physical prior, we found that the model instead distributed the effect of hydrogen bonding across several neighboring atoms, yielding inconsistent, sometimes repulsive, individual pair energies, so the sign of $\Delta u _ { a b }$ for an individual donor/acceptor pair carried no reliable physical meaning. We therefore add a hinge penalty that discourages repulsive interactions between hydrogen-bond donor and acceptor atoms. We consider an atom as a donor if it is an explicit hydrogen node bonded to N, O, or S; explicit hydrogen nodes bonded to any other atom are not counted as donors. Furthermore, we consider an atom as an acceptor if it is oxygen, or a nitrogen without positive formal charge that is neither a pyrrole-type aromatic N–H (whose lone pair is delocalized into the aromatic ring) nor part of a formally charged aromatic ring (e.g., imidazolium, pyridinium), in which the charge is delocalized onto the ring nitrogen. For every donor/acceptor atom pair (a, b) found in a mixture batch, the penalty is

$$
\mathcal { L } _ { \mathrm { h b o n d } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( a , b ) \in \mathcal { P } } \left[ \mathrm { R e L U } \left( \frac { \Delta u _ { a b } } { \Delta u _ { \mathrm { s c a l e } } } \right) \right] ^ { 2 } , \qquad \Delta u _ { \mathrm { s c a l e } } = R \cdot 1 0 0 ~ \mathrm { K } \approx 0 . 8 3 ~ \mathrm { k J } \mathrm { m o l } ^ { - 1 } ,\tag{41}
$$

where $\mathcal { P }$ is the set of donor/acceptor atom pairs in the batch (each unordered pair counted once), so the penalty is only nonzero when a donor/acceptor pair is repulsive $( \Delta u _ { a b } > 0 ) ;$ ; the constant $\Delta u _ { \mathrm { s c a l e } }$ only sets the scale of the term relative to the others, as $\Delta u _ { a b }$ reaches the order of $1 0 \mathrm { k J } \mathrm { m o l } ^ { - 1 }$ . This term is included with weight $\lambda _ { \mathrm { h b o n d } }$ and is only active when polar hydrogens are represented explicitly, since donor atoms only exist as such under that representation.

Total training objective. The full loss is the sum of all active terms,

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { T P X Y } } + \mathcal { L } _ { \mathrm { S L E } } + \lambda _ { \mathrm { a c t } } \mathcal { L } _ { \mathrm { A C T } } + \lambda _ { \mathrm { t p x } } \mathcal { L } _ { \mathrm { T P X } } + \lambda _ { \mathrm { h e } } \mathcal { L } _ { \mathrm { H E } } } \\ { + \lambda _ { \mathrm { l l e } } \mathcal { L } _ { \mathrm { L L E } } + \lambda _ { \mathrm { H } } \mathcal { L } _ { \mathrm { H } } + \lambda _ { \mathrm { g e o m } } \mathcal { L } _ { \mathrm { g e o m } } + \lambda _ { \mathrm { h b o n d } } \mathcal { L } _ { \mathrm { h b o n d } } , } \end{array}\tag{42}
$$

with $\lambda _ { \mathrm { a c t } } = 0 . 5 , \lambda _ { \mathrm { t p x } } = 1 . 0 , \lambda _ { \mathrm { h e } } = 0 . 5 , \lambda _ { \mathrm { l l e } } = 2 . 0 , \lambda _ { \mathrm { H } } = 0 . 0 5 , \lambda _ { \mathrm { g e o m } } = 0 . 1 , \mathrm { a n d } \lambda _ { \mathrm { h b o n d } } = 1 . 0 , \lambda _ { \mathrm { e v e } } = 0 . 1 , \lambda _ { \mathrm { e v e } } = 0 . 1 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { p h e } } = 0 . 1 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { e v e } } = 1 0 , \lambda _ { \mathrm { e v e } } = 1 0$ . Only terms for which the mini-batch contains data points contribute; a term without active data points contributes zero and produces no gradient. We evaluate the two regularizers separately on the data points of each data type except LLE present in the mini-batch and add their means, so they enter Eq. (42) once per data type present.

Optimization. The model is trained for 100 epochs with AdamW at a peak learning rate of $5 \times 1 0 ^ { - 4 }$ and weight decay of $1 0 ^ { - 4 } ,$ following a one-cycle schedule that raises the learning rate over the first 30% of the training steps and anneals it over the remainder (PyTorch OneCycleLR with default settings otherwise, stepped per mini-batch). The batch size is 512, gradients are clipped to a norm of $5 ,$ and the random seed is 42. The model is implemented in $\mathrm { P y ' }$ Torch [49] and $\mathrm { P y }$ Torch Geometric [50]. The ten models of the cross-validation and the full-data model for the multi-component results share this configuration. Throughout this work, we report the final checkpoint rather than the one with the lowest validation loss, so that every model is trained for the same fixed schedule and the validation data are used only for monitoring.

## 4.8 Evaluation

For every data type, the absolute errors of all data points of a system are averaged to its system-wise mean absolute error MAE<sub>sys</sub>, and we report the median, the mean, and the interquartile range of $\mathrm { M A E _ { s y s } }$ over the systems; relative changes between models refer to the median. The errors are taken in ln $\gamma _ { i }$ for TPXY, ACT, and SLE, in ln p for TPX, in the mole fractions of all coexisting phases for LLE, and in $h ^ { \mathrm { E } } ( \mathrm { i n } \mathrm { k J } \mathrm { m o l } ^ { - 1 } )$ for HE. Because mod. UNIFAC does not apply to all systems (missing group decompositions or interaction parameters), the comparisons in the main text are restricted to the systems that all three models can describe; for LLE, additionally restricted to the systems for which all three models predict a phase split for at least one data point. Since every LLE data point corresponds to a measured phase split, this criterion is itself a test of the models, which they pass for a similar share of systems (Supplementary Note 9, Supplementary Table 11). Results on all systems that COGNAC and HANNA describe are given in Extended Data Figs. 1 and 2.

HANNA was retrained with its published architecture and hyperparameters on the same data as COGNAC, including the SLE data: one model per fold for the binary cross-validation, and one model on the nine training folds for the multi-component mixtures, with the tenth fold as the validation set that the published HANNA procedure requires and that COGNAC uses only for monitoring, since we report the final checkpoint. The multi-component data were never used for training, but parts of their binary subsystems are contained in the binary training data. Mod. UNIFAC was used as published; its parameters were fitted to DDB data that include many of the benchmark systems, among them ternary mixtures.

## Data availability

The experimental data were licensed from the Dortmund Data Bank [33] and cannot be redistributed. The COSMO cavity data are from the CHAOS database [35] and available on Zenodo.

## Code availability

The COGNAC code and the COGNAC and COGNAC-IMP models trained on the full binary are available at https://github. com/marco-hoffmann/COGNAC under the MIT license. An interactive interface that predicts activity coefficients and phase equilibria and returns parameter tables, an application programming interface, and a Model Context Protocol (MCP) [51] connector for applications based on large language models are provided at MLPROP. Additionally, a Julia implementation of both COGNAC models is available in the Julia package MLThermoProperties.jl. The code for the batch distillation simulations is provided in https://github.com/se-schmitt/BatchDistillation.jl.

## Acknowledgements

The authors gratefully acknowledge financial support by the German Research Foundation (DFG) through the Emmy-Noether program (grant number 528649696), Priority Program SPP 2331 “Machine Learning in Chemical Engineering” (grant number 553345933), Priority Program SPP 2363 “Molecular Machine Learning” (grant number 497201843), and Research Training Group GRK 2908 “WERA” (grant number 503479768), and Germany’s Federal Ministry of Economic Affairs and Energy within the project DiKey. Model training and evaluation were carried out on the high-performance computer Elwetritsch at RPTU (grant RPTU-MLVT).

## Author contributions

M.H.: Conceptualization; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing – original draft; Visualization. S.S.: Methodology; Software; Formal analysis; Investigation; Writing – original draft. H.H.: Conceptualiza tion; Writing – review & editing; Supervision; Project administration; Funding acquisition. F.J.: Conceptualization; Methodology; Writing – review & editing; Supervision; Project administration; Funding acquisition.

## Competing interests

The authors declare no competing interests.

## References

[1] Aage Fredenslund, Russell L. Jones, and John M. Prausnitz. Group-contribution estimation of activity coefficients in nonideal liquid mixtures. AIChE Journal, 21(6):1086–1099, November 1975.

[2] Thomas Magnussen, Peter Rasmussen, and Aage Fredenslund. UNIFAC parameter table for prediction of liquid-liquid equilibriums. Industrial & Engineering Chemistry Process Design and Development, 20(2):331–339, April 1981.

[3] Ulrich Weidlich and Jürgen Gmehling. A modified UNIFAC model. 1. Prediction of VLE, $h ^ { \mathrm { E } }$ , and $\gamma ^ { \infty }$ . Industrial & Engineering Chemistry Research, 26(7):1372–1381, July 1987.

[4] Roland Wittig, Jürgen Lohmann, and Jürgen Gmehling. Vapor–liquid equilibria by UNIFAC group contribution. 6. Revision and Extension. Industrial & Engineering Chemistry Research, 42(1):183–188, January 2003.

[5] Zhigang Lei, Jiguo Zhang, Qunsheng Li, and Biaohua Chen. UNIFAC model for ionic liquids. Industrial & Engineering Chemistry Research, 48(5):2697–2704, January 2009.

[6] Dana Constantinescu and Jürgen Gmehling. Further development of modified UNIFAC (Dortmund): Revision and extension 6. Journal ofChemical & Engineering Data, 61(8):2738–2748, May 2016.

[7] Andreas Klamt. Conductor-like screening model for real solvents: A new approach to the quantitative calculation of solvation phenomena. The Journal ofPhysical Chemistry, 99(7):2224–2235, February 1995.

[8] Shiang-Tai Lin and Stanley I. Sandler. A priori phase equilibrium prediction from a segment contribution solvation model. Industrial & Engineering Chemistry Research, 41(5):899–913, March 2002.

[9] Hans Grensemann and Jürgen Gmehling. Performance of a conductor-like screening model for real solvents model in comparison to classical group contribution methods. Industrial & Engineering Chemistry Research, 44(5):1610–1624, February 2005.

[10] Robin Fingerhut, Wei-Lin Chen, Andre Schedemann, Wilfried Cordes, Jürgen Rarey, Chieh-Ming Hsieh, Jadran Vrabec, and Shiang-Tai Lin. Comprehensive assessment of COSMO-SAC models for predictions of fluid-phase equilibria. Industrial & Engineering Chemistry Research, 56(35):9868–9884, August 2017.

[11] Zhimin Xue, Tiancheng Mu, and Jürgen Gmehling. Comparison of the a priori COSMO-RS models and group contribution methods: Original UNIFAC, modified UNIFAC(do), and modified UNIFAC(do) consortium. Industrial & Engineering Chemistry Research, 51(36):11809–11817, August 2012.

[12] Fabian Jirasek and Hans Hasse. Combining Machine Learning with Physical Knowledge in Thermodynamic Modeling of Fluid Mixtures. Annual Review ofChemical and Biomolecular Engineering, 14(1):31–51, June 2023.

[13] Fabian Jirasek, Robert Bamler, and Stephan Mandt. Hybridizing physical and data-driven prediction methods for physico chemical properties. Chemical Communications, 56(82):12407–12410, 2020.

[14] Hans Hasse, Sebastian Schmitt, and Fabian Jirasek. Artificial intelligence in thermodynamics: hybrid modeling of thermophysical properties of fluids. Current Opinion in Chemical Engineering, 51:101236, March 2026.

[15] Marco Hoffmann, Nicolas Hayer, Maximilian Kohns, Fabian Jirasek, and Hans Hasse. Prediction of pair interactions in mixtures by matrix completion. Physical Chemistry Chemical Physics, 26(28):19390–19397, 2024.

[16] Marco Hoffmann, Hans Hasse, and Fabian Jirasek. GRAPPA—A hybrid graph neural network for predicting pure component vapor pressures. Chemical Engineering Journal Advances, 22:100750, May 2025.

[17] Fabian Jirasek, Robert Bamler, Sophie Fellenz, Michael Bortz, Marius Kloft, Stephan Mandt, and Hans Hasse. Making thermodynamic models of mixtures predictive by machine learning: matrix completion of pair interactions. Chemical Science, 13(17):4854–4862, April 2022.

[18] Fabian Jirasek, Nicolas Hayer, Rima Abbas, Bastian Schmid, and Hans Hasse. Prediction of parameters of group contribution models of mixtures by matrix completion. Physical Chemistry Chemical Physics, 25(2):1054–1062, 2023.

[19] Nicolas Hayer, Thorsten Wendel, Stephan Mandt, Hans Hasse, and Fabian Jirasek. Advancing thermodynamic groupcontribution methods by machine learning: UNIFAC 2.0. Chemical Engineering Journal, 504:158667, January 2025.

[20] Nicolas Hayer, Hans Hasse, and Fabian Jirasek. Modified UNIFAC 2.0—a group-contribution method completed with machine learning. Industrial & Engineering Chemistry Research, 64(20):10304–10313, May 2025.

[21] Benedikt Winter, Clemens Winter, Timm Esper, Johannes Schilling, and André Bardow. SPT-NRTL: A physics-guided machine learning model to predict thermodynamically consistent activity coefficients. Fluid Phase Equilibria, 568:113731, 2023.

[22] Henri Renon and J. M. Prausnitz. Local compositions in thermodynamic excess functions for liquid mixtures. AIChE Journal, 14(1):135–144, January 1968.

[23] Benedikt Winter, Clemens Winter, Johannes Schilling, and André Bardow. A smile is all you need: predicting limiting activity coefficients from SMILES with natural language processing. Digital Discovery, 1(6):859–869, 2022.

[24] Edgar Ivan Sanchez Medina, Steffen Linke, Martin Stoll, and Kai Sundmacher. Gibbs–Helmholtz graph neural network: capturing the temperature dependency of activity coefficients at infinite dilution. Digital Discovery, 2(3):781–798, 2023.

[25] Jan G. Rittig, Kobi C. Felton, Alexei A. Lapkin, and Alexander Mitsos. Gibbs–Duhem-informed neural networks for binary activity coefficient prediction. Digital Discovery, 2(6):1752–1767, December 2023.

[26] Jan G. Rittig and Alexander Mitsos. Thermodynamics-consistent graph neural networks. Chemical Science, 15(44):18504– 18512, 2024.

[27] Thomas Specht, Mayank Nagda, Sophie Fellenz, Stephan Mandt, Hans Hasse, and Fabian Jirasek. HANNA: hard-constraint neural network for consistent activity coefficient prediction. Chemical Science, 15(47):19777–19786, 2024.

[28] Marco Hoffmann, Thomas Specht, Quirin Göttl, Jakob Burger, Stephan Mandt, Hans Hasse, and Fabian Jirasek. Thermodynamically consistent machine learning model for excess Gibbs energy. Nature Communications, 17(1):3485, April 2026.

[29] Walid Ahmad, Elana Simon, Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar. ChemBERTa-2: Towards Chemical Foundation Models, 2022. arXiv:2209.01712. ELLIS Machine Learning for Molecule Discovery Workshop, NeurIPS 2022.

[30] David Weininger. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal ofChemical Information and Computer Sciences, 28(1):31–36, February 1988.

[31] RDKit: Open-source cheminformatics. http://www.rdkit.org. Version 2025.09.5.

[32] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1263–1272, 2017.

[33] Dortmund data bank. https://www.ddbst.com, 2026. Version 2026.

[34] A. Klamt and G. Schüürmann. COSMO: a new approach to dielectric screening in solvents with explicit expressions for the screening energy and its gradient. J. Chem. Soc., Perkin Trans. 2, (5):799–805, 1993.

[35] Dominik Gond, Justus Arweiler, Thomas Specht, Hans Hasse, and Fabian Jirasek. CHAOS - a large-scale database for σ-profiles and other molecular descriptors. Journal ofChemical Information and Modeling, 66(7):3969–3975, April 2026.

[36] O. Ryll, S. Blagov, and H. Hasse. Convex envelope method for the determination of fluid phase diagrams. Fluid Phase Equilibria, 324:108–116, June 2012.

[37] Quirin Göttl, Jonathan Pirnay, Dominik G. Grimm, and Jakob Burger. Convex Envelope Method for determining liquid multi-phase equilibria in systems with arbitrary number of components. Computers & Chemical Engineering, 177:108321, September 2023.

[38] Quirin Göttl, Natalie Rosen, and Jakob Burger. Convex envelope method for $T , p$ flash calculations for mixtures with an arbitrary number of components and arbitrary aggregate states. Computers & Chemical Engineering, 203:109326, December 2025.

[39] Andreas Klamt, Gerard J. P. Krooshof, and Ross Taylor. COSMOSPACE: Alternative to conventional activity-coefficient models. AIChE Journal, 48(10):2332–2349, October 2002.

[40] Jürgen Rarey. Extended flexibility for G<sup>E</sup> models and simultaneous description of vapor–liquid equilibrium and liquid–liquid equilibrium using a nonlinear transformation of the concentration dependence. Industrial & Engineering Chemistry Research, 44:7600–7608, 2005.

[41] A. Marcilla, M. M. Olaya, and J. A. Reyes-Labarta. Simultaneous VLLE data correlation for ternary systems: Modification of the NRTL equation for improved calculations. Fluid Phase Equilibria, 426:47–55, 2016.

[42] Antje Jakob, Hans Grensemann, Jürgen Lohmann, and Jürgen Gmehling. Further development of modified UNIFAC (Dortmund): Revision and extension 5. Industrial & Engineering Chemistry Research, 45(23):7924–7933, October 2006.

[43] Jürgen Gmehling, Jürgen Lohmann, Antje Jakob, Jiding Li, and Ralph Joh. A modified UNIFAC (Dortmund) model. 3. Revision and Extension. Industrial & Engineering Chemistry Research, 37(12):4876–4882, 1998.

[44] Justus Arweiler, Indra Jungjohann, Aparna Muraleedharan, Heike Leitte, Jakob Burger, Kerstin Münnemann, Fabian Jirasek, and Hans Hasse. Batch distillation data for developing machine learning anomaly detection methods. Scientific Data, 13(1):513, 2026.

[45] Jennifer Werner, Justus Arweiler, Indra Jungjohann, Jochen Schmid, Fabian Jirasek, Hans Hasse, and Michael Bortz. Automated batch distillation process simulation for a large hybrid dataset for deep anomaly detection. Computers & Chemical Engineering, 215:109827, 2026.

[46] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv:1607.06450, 2016.

[47] Denis S. Abrams and John M. Prausnitz. Statistical thermodynamics of liquid mixtures: A new expression for the excess Gibbs energy of partly or completely miscible systems. AIChE Journal, 21(1):116–128, January 1975.

[48] Karim K. Ben Hicham, Moreno Ascani, Jan G. Rittig, and Alexander Mitsos. Differentiable Thermodynamic Phase-Equilibria for Machine Learning. Journal ofChemical Information and Modeling, 66(17):10487–10502, August 2026.

[49] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pages 8024–8035, 2019.

[50] Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with PyTorch Geometric. arXiv:1903.02428, 2019.

[51] Model Context Protocol. https://modelcontextprotocol.io, 2024.

![](images/d7f7c6ea3c7be751e8610cc7cd28e281bcaa797bfa64c9ec9ce5391e4f5623e4.jpg)

![](images/97a042ab806e27474212e27f54bea2e7db2b575acbe09a9620ac666260a617f4.jpg)

![](images/5d942e81bc8a721798350a982bae15836f863045609f252d5897bf73b968fbc2.jpg)

![](images/5e27a75f21bca5c2962a78c852db1909d26a352c18a9821f989a30d03c4bd303.jpg)

![](images/4ce966dd17bd25a5c2597e2db97fb8d0b67a376a31aab7a61304b673239a9875.jpg)

![](images/25ab64b7c10b8386c9b1eb9ac4691f9bff1592ed7fb8f9242bccdd23287283b0.jpg)

![](images/7414471d376a7c9f80f3b415a39fd60ffd242ec3731d4a7b192d1f7a6977e9f5.jpg)

![](images/7bda238a2c8097a701133be394cd2d5dd1ad466d65cf56044c9e80b1356e8793.jpg)

![](images/e53098778c1dc06745f0ab002b63924682b6cb60ff102839536fc1273e0e2d47.jpg)

![](images/c000448d24c7dc3e1af0986f8196bd43df2013cc388a33b40423600d6f930e1a.jpg)

![](images/c4a1b6aeb519a9d82a0e810b7c641c67ecb689ff24bdebacee5fbcb3b6a65ecf.jpg)  
Extended Data Fig. 1: Benchmark of COGNAC against HANNA. a, System-wise mean absolute errors $\mathrm { M A E _ { s y s } }$ for the binary data, resolved by data type; each system is predicted by the COGNAC and HANNA models on the ten-fold cross-validation set it did not see during training. Circles denote the median, diamonds the mean, and shaded bars the interquartile range of the distribution over systems. n is the number of systems in each panel. For LLE, a system is included if both models predict a phase split for at least one of its data points. The percentage above each panel shows the relative change in the median $\mathrm { \bar { M A E } _ { s y s } }$ of COGNAC relative to HANNA; negative values indicate that COGNAC is more accurate. Both models were trained on the same binary data set. The systems outside the coverage of mod. UNIFAC, included here but not in Fig. 2, contain components that mod. UNIFAC cannot decompose or for which interaction parameters are missing; many of them are ionic liquids. b Same comparison for the ternary data set, predicted with the full-data models of COGNAC and HANNA.

![](images/1b5d23cdcea1ca8f1ba62e705baa26dc755aa96865823abbda32b24a9c2bcd41.jpg)

Extended Data Fig. 2: Benchmark of COGNAC against mod. UNIFAC and HANNA on the quaternary data set, analogous to Fig. 2b. The data set comprises 21,221 data points for 1,032 systems: 3,750 TPXY data points (134 systems), 1,113 TPX data points (38 systems), and 16,358 LLE data points (888 systems, including 234 three-liquid data points). COGNAC and HANNA were trained only on binary data, so every quaternary prediction extrapolates the number of components; mod. UNIFAC was used as published and is the only model whose parameters were also fitted to multi-component data. LLE predictions use the same phase-equilibrium procedure as for the ternary mixtures, with the four-component composition space discretized into 6,545 points. System-wise mean absolute errors $\mathrm { M A E _ { s y s } }$ of the logarithmic activity coefficients (TPXY), the logarithmic pressure (TPX), and the liquid mole fractions (LLE, all phases and components). a, Shared subset: systems that all three models can describe; for LLE, only systems for which all three models predict a phase split are included. Relative to mod. UNIFAC, COGNAC reduces the median $\mathrm { M A E _ { s y s } }$ by 9% (TPXY), 11% (TPX), and 18% (LLE). b, Full data set: all systems that COGNAC and HANNA both describe, without restricting the comparison to the coverage of mod. UNIFAC. Circles denote the median, diamonds the mean, and shaded bars the interquartile range of the distribution over systems; n is the number of systems. The percentage above each panel shows the relative change in the median $\mathrm { M A E _ { s y s } }$ of COGNAC relative to HANNA; negative values indicate that COGNAC is more accurate. The number of TPX systems is small, so these values should be read with caution.

a Binary, aqueous  
d Ternary, non-aqueous  
![](images/6fcbf6a1a995e1d39160aaa3c141ff6f1ce63a7c7d20411edd81a6956addd7b9.jpg)

![](images/438c1dc6f61ba4b6f7d6bf9d3cda5d9be06577d733de20b92b44ea7d894a4406.jpg)

![](images/f01c22c34c23025b55e028d23eef320a8a60341e9309c9477ecfa6046a162c54.jpg)

![](images/5bf20fd3ac0bb6409605339db8ecf2c35e9e9761810f71ee023146b74511cca2.jpg)

![](images/b5966fbc6cff1106a7ab597427953d47012400f0278b94432de260ebff7e9c9d.jpg)

![](images/2b8c6516e06058923f5d7e19540af01bf7c6b401986c0d5105967284c28b52c1.jpg)

![](images/2e895086026caa13f1e9a325dc479596d35d3f039f2ec2031ecd3d1b5a0a1306.jpg)

![](images/e52ace227eb58bd769783cfd575fb55fdf10b0c2e79dbba53524781d6b43bf27.jpg)

![](images/39d8af34d4492fbd485d1a4664627e98275e4a6d9f789fb4ef5340c1d0add3a2.jpg)

![](images/ccfcc69c4377bf45c01eb07739cfda537f9f65f403031d8dc1f64a632d35435d.jpg)

![](images/24a050edd30d89bc6dec9b75fe4b3f4822e650317516c94c0885edf72f002e0a.jpg)

![](images/c0a9155670d941b0bf8db85538096734331e4341f4cc67eb912c2953e8cd367d.jpg)

![](images/fafed2f4d27f353224776299c096e99f15ce3225f072ebc2e2f2a7a675f3f594.jpg)

![](images/f5fdc9ecce70a5c66c699ae3c9a500b38344f4f0df0e376dc42cc6340b3b7ed7.jpg)

![](images/1bb64149c5eaf96c1d539884aee1f30fa57bdca06aa2f28929bb4193fcc85591.jpg)

![](images/29b6bb4c160825eb45e6e7dd0b9c6cf28323c5140f02bd61e02a2b4198cfc7a0.jpg)

![](images/c5cbadb7cc49c83ce5db5e0170032f8905605bde8764bef40c552e5f9071f85b.jpg)

![](images/711b7d3948511608f9ad3cb33e27bec4bed0e4e5d5bf18ef063772eab3949732.jpg)

![](images/710e734acc49ef7003a08381b70b6edc4e26fe932769bb2627170f10283f5e74.jpg)

![](images/868f1b778e206029ea379f4bf18755edcd80ba012de301c73c39bf4f768e5dc6.jpg)

![](images/916d72e5195330261fd3c244c5e9dfc5b4cf8c6ac40a73bdcfac4725aebb3458.jpg)

![](images/88aca1be8236d7a14b0230f87515e143dcf216e99ce888f780f7d2af8508ae07.jpg)  
Extended Data Fig. 3: Benchmark of HANNA and COGNAC for aqueous and non-aqueous mixtures. System-wise mean absolute errors $\mathrm { M A E _ { s y s } }$ for binary (a, b) and ternary (c, d) mixtures that contain water (a, c) or not (b, d), on all systems that both models describe. Binary: out-of-fold predictions of the ten-fold cross-validation; ternary: full-data models. The data points are those of Extended Data Fig. 1, split by whether a component is water. Circles denote the median, diamonds the mean, and shaded bars the interquartile range across systems; n is the number of systems. The percentage above each panel shows the relative change in the median $\mathrm { M A E _ { s y s } }$ of COGNAC relative to HANNA; negative values indicate that COGNAC is more accurate. For every data type and both mixture classes, COGNAC gives the lower median ${ \mathrm { M A E } } _ { \mathrm { s y s } }$ . For the binary mixtures, the relative reduction is similar with and without water except for TPX; for the ternary mixtures, it is considerably larger for aqueous mixtures in TPXY and ACT, larger in TPX, and smaller in LLE and HE.

a <sub>Binary</sub>  
![](images/9032a647d39d7b9399e3d62922efeb59b89928fbfa77f0abd2c5f85599ad097a.jpg)  
b <sub>Ternary</sub>

![](images/02f4aab516e275577172d05893f3c6eb07de57a44b2456dfa273bf3e1a2ceee9.jpg)

![](images/0bd2b6a1e49f7a75d507c1be66ba572c471034ac06a637258ce89b00c7683340.jpg)

![](images/1468baaa9ff46710ca814f12eb482189cedd7a05138d994d299b35b7d667cc26.jpg)

![](images/67a61adc6df278b96a877248fc9325bddd74d005ba5f2d229a368db1f81be1be.jpg)

![](images/9945af0e53416e146149b1a15d1a6327cf243e1e878381e2fb541c3406ffbcfd.jpg)

![](images/9f905739c855607fb5b945830aeaff7a68459bdfc93b4eba16848d0f626405b2.jpg)

![](images/c50688db356a2645aac3dd138ce1aea105d2da95ed1b158fd8e137316cb20632.jpg)

![](images/6f13ed308f114a7bcb9498bd2ed14cf0cd3e3ed078919fe58ab1c3b81558445c.jpg)

![](images/752a586683e8e7499700b5c0290cad51de025c6f88e725d7633e07017793d2d8.jpg)

![](images/220cb460b55d6841210434f7250ddd510015fc5bb7511f9cca3c3c6fd7fe6c82.jpg)

Extended Data $\mathrm { F i g . }$ . 4: Effect of the hydrogen representation on COGNAC accuracy, resolved by mixture and data type. Bars show the median $\mathrm { M A E _ { s y s } }$ of the production model, in which polar hydrogen atoms are explicit graph nodes, and of an ablated model in which all hydrogen atoms are treated implicitly, so that water is a single node. Architecture, training data, loss weights, hyperparameters, and random seed are identical; the only further difference is that the sign regularization of the donor–acceptor interactions, which is defined on the explicit polar hydrogen atoms, has no counterpart in the ablated model, so the comparison measures the combined effect of the explicit hydrogen nodes and this regularizer. We trained the ablated model in every fold of the ten-fold cross-validation and evaluated it like the production model: binary mixtures on out-of-fold predictions, and ternary mixtures with the full-data model, on all systems in these data sets that both models describe; n is the number of systems. For aqueous mixtures, the implicit treatment increases the error for every data type, most strongly for binary LLE (+101.8%) and TPX $( + 6 2 . 2 \% ) \div$ ; without water, the two representations differ by less than 5% except for ternary ACT and HE. For reference, the bars also show the median $\mathrm { M A E _ { s y s } }$ of HANNA, trained on the same folds. HANNA is scored on all systems for which it yields a prediction; for LLE, it predicts no phase split for some systems, which are excluded (binary: 1,840 of 1,998 non-aqueous and 906 of 917 aqueous systems; ternary: 2,414 of 2,482 and 2,897 of 2,933). Even the ablated model is more accurate than HANNA in 18 of the 22 cells; HANNA is ahead only for binary TPXY, LLE, HE, and SLE of aqueous mixtures.

![](images/a2bf37c92fc7f1c50a1637a849767461ab2e013e96ba9d34e702d9860c376925.jpg)  
Extended Data Fig. 5: Activity coefficients at infinite dilution of alkanes (top) and alcohols (bottom) in water, predicted with mod. UNIFAC, mod. UNIFAC 2.0, HANNA, COGNAC, and COGNAC-IMP, against the experimental values: binary ACT data with water as the solvent, with the solutes classified by their DDB component class. COGNAC, COGNAC-IMP, and HANNA: out-of-fold predictions of the ten-fold cross-validation. Mod. UNIFAC 2.0 [20] is a version of mod. UNIFAC in which the entire interaction-parameter matrix was fitted simultaneously via matrix completion rather than sequentially; both mod. UNIFAC versions are used as published and were fitted to DDB data that likely include these data points, which favors them. Only data points for which all models yield a prediction are shown (367 data points of 41 alkane–water and 784 data points of 67 alcohol–water systems). MAE is the mean absolute deviation in ln $\gamma _ { i } ^ { \infty }$ . Mod. UNIFAC underestimates ln $\gamma _ { i } ^ { \infty }$ of every alkane in water; mod. UNIFAC 2.0 reduces the error but still underestimates 97% of the alkane data points, so the deficiency is not an artifact of the sequential fitting but of the static group decomposition itself. COGNAC, which evaluates the same equations with atom-level parameters, and HANNA, which does not build on a local-composition model, describe both families well.

![](images/aa6ed1ae397abdd48d5e25249266b85af98f2fa7ee3f67c5022095e30e3bf6ee.jpg)

![](images/0df7559827e068b84e7e16898ec32b86776715a43eb9605fda76713e819386e4.jpg)  
Extended Data Fig. 6: Activity coefficients of the three benzenediol isomers at infinite dilution in diisopropyl ether (a) and in 4-methyl-2-pentanone (b) at 298.15 K, predicted with mod. UNIFAC and COGNAC. In mod. UNIFAC, the three isomers have identical group counts, so it predicts one value for all benzenediols. Experimental values are shown as filled circles, and COGNAC predictions are shown as open diamonds connected by a thin line to the corresponding measurement. The substitution pattern of each component is given in parentheses. All six COGNAC activity coefficients are out-of-fold predictions; i.e., each was predicted by a cross-validation model that did not see the respective mixture during training.

![](images/3c2029b3748a02e545ba86eb647545ba8d5cbab9530d28a0cd9b8a67452923de.jpg)

![](images/b64a568fc814319986209e1526ff7ef22dd73967e6938b614cb80866bc1ec3b6.jpg)

![](images/754d404a618b1748477196663dcd395a8e84ec0719dd8514df8a5402ab9763e8.jpg)

![](images/873fb1934a2b8462d0ed4cc6d499e73ba8fc71f152da03b396204b5e458bd1ae.jpg)

![](images/10e4cb5a3dc0fa053c027cc19cdf140d3c44006c0da477a17fc228c7669d3cee.jpg)

![](images/2df0e6dae61640b524535503869c9509e6fd2330c4667df3dbb003d08136d2fc.jpg)  
Extended Data Fig. 7: Azeotrope predictions of HANNA (a–c) and mod. UNIFAC (d–f), analogous to Fig. $3 \mathrm { a - c . } \ \mathbf { a } , \mathbf { b } , \mathbf { d } , \mathbf { e } .$ Classification matrices for binary (a,d) and ternary (b,e) mixtures, predicted class against experimental class; the diagonal marks correct prediction of azeotrope existence and class. c,f, Predicted against experimental compositions at the azeotropic point, over all components, for the azeotropes whose existence and type were predicted correctly, excluding double azeotropes. Each panel shows the model on all data points for which it yields a prediction, so n is smaller for mod. UNIFAC; the classification shares compared in Supplementary Note 6 are evaluated on the data points that all three models describe.

![](images/e8b9410055cbdc9d953ffd3eee22d754fdb02179835ff3f44bd4583b8dce9820.jpg)

![](images/9bea79679831ec5f8cf976e058e408d8fa9fe4d8143b1430f04a61d93c13e371.jpg)

![](images/7d32af679cd4f11f61f23472a1814b2b607e35a3059163564df0f7c372e09a56.jpg)  
Extended Data Fig. 8: COGNAC cavity geometries compared with COSMO references from the CHAOS database [35], predicted by the full-data COGNAC model. a, molecular volume $V _ { i } ; \mathbf { b } .$ , molecular surface area $A _ { i } ;$ and c, atomic surface area $A _ { a }$ . Molecular values are the sums of the atom-level geometry-head outputs, $\begin{array} { r } { V _ { i } ^ { \mathrm { p r e d } } = \sum _ { a \in i } V _ { a } } \end{array}$ and $\begin{array} { r } { A _ { i } ^ { \mathrm { p r e d } } = \sum _ { a \in i } A _ { a } . } \end{array}$ . The comparison is restricted to the 45,488 CHAOS molecules that occur in no mixture of the thermodynamic data and consist only of elements in the data set; since the COSMO regularizer used in training covers only molecules in the thermodynamic data, none of the compared values entered training. Atom-level areas are available for 45,427 of these molecules (675,822 atoms); carbon-bonded hydrogen areas are folded onto their parent atoms, and polar hydrogens are explicit, consistent with the model’s molecular graphs Superscripts pred and COSMO distinguish predictions and reference values; dashed lines indicate parity. The axis limits run from the first percentile of the smaller coordinate in each pair to the 99th percentile of the larger one, retaining about 98% of the points; the reported counts and statistics include points outside the displayed limits, excluding 701 zero-area references from the logarithmic comparison. $R _ { \mathrm { l o g } } ^ { 2 }$ denotes the coefficient of determination against parity in logarithmic space; MAE is calculated in the physical units shown and amounts to $1 1 . 0 2 \mathring { \mathrm { A } } ^ { 3 }$ (molecular volume), $1 4 . 1 3 \mathring { \mathrm { A } } ^ { 2 }$ (molecular surface area), and $1 . 6 9 \mathring { \mathrm { A } } ^ { 2 }$ (atomic surface area).

Extended Data Table 1: Ablation study comparing six COGNAC model variants. The models differ in the implemented temperature dependence (UNIFAC, free form, mod. UNIFAC) and the closure of the residual activity coefficients (explicit, implicit). Each variant was trained on the ten folds of the cross-validation (Methods) with identical hyperparameters and the same single random seed per fold, so that the variants differ only in the stated switches (the UNIFAC variants also in the combinatorial term, see Supplementary Note 2); differences of a few percent are within the variation between independently trained models. Scores are the mean $\mathrm { M A E _ { s y s } }$ over the out-of-fold predictions of all binary systems; as for the comparison of the closures in Supplementary Note 3, the mean over the systems is used rather than the median of the benchmarks, so that the tail of the error distribution is included. The lowest value per data type among the variants is printed in bold. HANNA denotes the HANNA results of the same ten-fold cross-validation; for LLE, they yield a prediction for 2,653 of the 2,784 systems and are scored on these. COGNAC is the mod. UNIFAC variant with explicit closure; COGNAC-IMP is the one with implicit closure (Supplementary Note 3).
<table><tr><td>Temperature dependence</td><td>Closure</td><td>TPXY</td><td>TPX</td><td>ACT</td><td>LLE</td><td>SLE</td><td>HE</td></tr><tr><td>UNIFAC</td><td>explicit</td><td>0.083</td><td>0.090</td><td>0.213</td><td>0.066</td><td>0.134</td><td>0.147</td></tr><tr><td>UNIFAC</td><td>implicit</td><td>0.085</td><td>0.092</td><td>0.201</td><td>0.067</td><td>0.129</td><td>0.159</td></tr><tr><td>free form</td><td>explicit</td><td>0.078</td><td>0.088</td><td>0.198</td><td>0.062</td><td>0.128</td><td>0.147</td></tr><tr><td>free form</td><td>implicit</td><td>0.077</td><td>0.085</td><td>0.188</td><td>0.060</td><td>0.124</td><td>0.139</td></tr><tr><td>mod. UNIFAC</td><td>explicit</td><td>0.080</td><td>0.086</td><td>0.194</td><td>0.059</td><td>0.124</td><td>0.145</td></tr><tr><td>mod. UNIFAC</td><td>implicit</td><td>0.078</td><td>0.085</td><td>0.185</td><td>0.060</td><td>0.121</td><td>0.138</td></tr><tr><td>HANNA</td><td></td><td>0.088</td><td>0.094</td><td>0.242</td><td>0.077</td><td>0.141</td><td>0.184</td></tr></table>