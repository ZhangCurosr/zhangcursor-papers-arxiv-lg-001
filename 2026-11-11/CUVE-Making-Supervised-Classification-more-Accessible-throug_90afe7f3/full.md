# CUVE: Making Supervised Classification more Accessible through Visualization

Sanchitha Kuppusami<sup>1</sup> <sup>a</sup>, Xiyao Wang<sup>1</sup> <sup>b</sup> and Miguel A. Nacenta<sup>1</sup> <sup>c</sup>

<sup>1</sup>Department of Computer Science, University of Victoria, Victoria, Canada sanchu.kuppusami@gmail.com, xiyao.wang23@gmail.com, nacenta@uvic.ca

Keywords: Machine Learning, Supervised Classification, VLAIR, Visualization, Visual Interpretability, XAI

## Abstract:

We present CUVE, an interface that enables people to build classifiers by generating visualizations that are interpretable by both humans and learning algorithms (the Visualization as Intermediate Representation ap proach). This UI enables users to create supervised classifiers and to visually investigate how to improve them and how they carry out their function. A preliminary study investigates the usability of our prototype and compares it with a traditional text-based programming interface.

## 1 INTRODUCTION

An open problem in machine learning is that it is difficult for humans to understand and explain why a classifier has classified an instance in a particular way. This problem is often referred to as the “black box” problem of ML. It is important because the ability to inspect the mechanisms by which a decision is made is important for accountability, debugging and trust (Doshi-Velez and Kim, 2017).

Several approaches have been proposed to ameliorate this issue within the subarea of explainable, interpretable or intelligible AI. One proposed technique is using VisuaLization As Intermediate Representation (VLAIR (Jiang et al., 2020; Jiang et al., 2022)). The core idea is that, instead of training a deep learning network directly on data, it is trained on visualizations of the data. The transformation of the data into visualizations can be determined or guided by a human in analogously to how visualizations are designed for regular data: the specification of a mapping from data to visual marks and features. For example, a card transaction will generate a small visualization that could have a large red circle if the transaction comes from a location far from the credit card holder’s home, and a small green circle if from a nearby location. Other features of the transaction will be mapped to other marks in the images, such as other shapes, points or gradients of different colors, sizes and lengths. The system is trained using these images as the labeled data. This technique partially addresses the black box problem because sharing a representation (the visualization) gives humans a visual understanding of the data and, importantly, gives the human an opportunity to infer visually how the ML classification works. For example, by inspecting the visualizations of instances classified as one category, a person can try to infer which visual features they have in common. In other words, the visual representation becomes a machine-human shared representation of the data that might allow the human to infer how the machine might be categorizing instances.

Current applications of VLAIR-like approaches allow its implementation by manually modifying the custom visualization and training software. E.g., Jiang et al. (Jiang et al., 2022) use pyplot in a jupyter notebook to generate visualizations used to train a model with pytorch. This is powerful but only accessible to people with fluency in advanced tools and programming languages. Moreover, even for experts, it can be tedious and inefficient.

CUVE (Classification Understood through Visual Externalizations) is a tool that integrates visualization design and comparison and model training in a single UI to facilitate applying the VLAIR approach. This paper describes CUVE and a study’s results comparing CUVE and a traditional notebook-based alternative. Non-experts had a more positive appreciation of the CUVE tool and found that it could improve several aspects of the classification implementation process for both experts and non-experts.

## 2 Background and Related Work

Our work concerns the task of supervised classification, where a learning algorithm uses data labeled in advance, to train a model that can then assign a class to any item, based only on the value of its features (Singh et al., 2016). Classification can take place on images or regular data. Computer Vision (CV) has long leveraged DL algorithms to classify images (Byerly et al., 2022).

Jiang et al. proposed to leverage CV approaches to classify non-visual/non-spatial data, which they call VisuaLizations As Intermediate Representations (VLAIR (Jiang et al., 2020; Jiang et al., 2022)). The main idea is that any data can be transformed into a visualization in the same way that data analysis and visualization professionals visually chart their data (Card, 1999; Munzner, 2014)) and then, instead of training an algorithm directly on the raw, typically non-spatial data, the algorithm is trained with the generated visualizations. Jiang et al. showed accuracy advantages, which has also be used in other contexts for this reason (Zeng et al., 2014). However, more interestingly than its performance, VLAIR enables both machine and humans to share a visual representation. This might be useful to ameliorate the “black box” problem (Confalonieri et al., 2021), that is, the difficulty for humans to interpret or make sense of the process of classification.

The black box problem or, more generally, the issues of interpretability of complex machine learning models, are a longstanding and growing concern in the ML community and in society at large. There is a general acceptance that most ML models act as black box models because they are too complex for humans to understand how they make predictions (Loyola-Gonzalez, 2019), at least without help. There are two slightly different aims in the literature that pursues this kind of facilitation. Interpretability is mostly “associated with the intuition behind the outputs of a model. The idea is that the more interpretable a ML system is, the easier it is to identify cause-and-effect relationships within the system’s inputs and outputs (Linardatos et al., 2020)” and Explainability is “associated with the internal logic and mechanics that are inside a machine learning system. The more explainable a model, the deeper the understanding that humans achieve in terms of the internal procedures that take place while the model is training or making decisions” (Linardatos et al., 2020).

The overlapping research areas of AI interpretability and explainable AI (XAI) are too vast to survey here (examples of good surveys are (Adadi and Berrada, 2018; Gohel et al., 2021; Kalasampath et al., 2025)). A popular classification of XAI methods distinguishes between model-specific methods such as Class Activation Mapping and a family of related methods (CAM (Zhou et al., 2016; Selvaraju et al., 2016; Selvaraju et al., 2017) (mainly used for CNN algorithms), backpropagation-based methods such as DeepLIFT (Shrikumar et al., 2017), Layer-wise Relevance Propagation (Bach et al., 2015)) and perturbation-based methods such as Occlusion (Ivanovs et al., 2021) and RISE (Petsiuk et al., 2018); model-agnostic methods, of which LIME (Ribeiro et al., 2016) and SHAP (Lundberg and Lee, 2017) are the best well known. Finally, a range of different methods have been used to represent the relationship between the model outcome and the features in the data, such as partial dependence plots (Wright, 1950), ice (Gerelt-Od et al., ) and ale (Apley and Zhu, 2020). Key differences between these types of visualizations and VLAIR is that in VLAIR the human designs the visualization, and the visualization is an output to be consumed both by the human and the machine.

## 3 Goals and Design Principles

We have two main goals; we seek to make ML classification more interpretable (G1) and we seek to make ML classification more accessible (G2). We address G1 by implementing a pure VLAIR approach, and G2 by providing an interface that does not require textual programming.

To address the goals, we adopted a set of design principles: DP1 Support operation without coding (avoid writing textual programming whenever possible by leveraging the constructive visualization approach (Huron et al., 2014)); DP2 Leverage visuals as much as possible (the design should use visuals and visualizations as much as possible); DP3 Enable iterative refinement (support progressive iterative refinement of the model and of the users understanding of the dataset and how the model operates); DP4 Avoid unnecessary complexity (the design should be clean, simple, include only necessary elements and use familiar UI language); DP5 Avoid menus and hidden operations (expose as much functionality as possible without requiring complex navigation); and DP6 Avoid reliance on visual memory (Display relevant information on one single screen when possible).

In this section we describe CUVE’s design, linking its features to the goals and design principles.

CUVE is implemented as a web application designed around five steps (web tabs), each corresponding to a step of the supervised classification workflow. It does not require coding because functions are available through widgets and canvases (DP1, DP2). Each tab/step leads to the next, creating a new tab in the browser. Opening a new tab does not close the previous one so that the user can switch back, make changes, and compare without memorizing or reconstructing everything (DP3).

Tab 1 displays a big data table representation (Figure 1) which shows the result of uploading data in .xlsx and .csv formats, and that Tab 2 enables selection of the data types for each column, addressing missing data situations, and creating new columns with simple formulas based on other columns.

![](images/ac7531581812d3187602eb5b91f9f260edee3b066831f19719bd4ed3b86e1fa8.jpg)  
Figure 1: File Uploading and Inspection screen.

Tab 3 (Fig. 2) corresponds to input selection and is also an extension of feature engineering in the traditional ML workflow. It lets users select and combine desired features to construct a visualization, which will later serve as input for classification. It is based on iVolver (Mendez et al., 2016), a visual construc-´ tive visualization approach which supports graphical manipulation, accommodating design goals DP4 and DP5 (this approach has also been used in other contexts (Halaharvi et al., 2024; Mendez et al., 2017;´ Mendez et al., 2018)). It supports multiple functions.´

In Tab 3 users create marks (e.g., rectangles, and circles) by dragging and dropping from the top panel (Figure 2.B) into the construction canvas (Figure 2.C). which also makes a dedicated panel appears to let users define the mark’s visual parameters, such as size and color. These parameters can be linked to data features displayed as buttons in the middle panel (Figure 2.D).

The system has a default 2D coordinate system on the canvas, allowing users to change each mark’s position. Each mark’s orientation, and scale can also be changed or linked to data. Additionally, users can create other coordinate systems with relative transformations to the default one. The position of coordinate systems can also be associated to data variables.

Tab3 offers a transformation Canvas (Figure 2.D) that can combine and transform data columns into outputs that, in turn, can feed the positions or attributes of marks.

Finally, the user can preview the visualizations created for each row of the data. The user first selects which section of the visualization will fit within the output images. Then, selecting data rows in the left panel (Figure 2.F) displays the corresponding visualizations in the right panel (Figure 2.G). It is also possible to visualize a superposition of all the outputs to see the ranges and density of outputs in different regions of the image.

Tab 4 (Fig. 3) supports training and testing. Users visually inspect produced images, define and adjust the training and test set ratio, train a model, test it, overview the results, and validate the model.

Upon entering tab 4, all images created from the previous tab are first grouped by labels, i.e., by the assigned classification output (Fig. 3.A). For each label, images are displayed (Fig. 3.E) and randomly separated into training and test datasets with a userdefinable ratio (Fig. 3.A). Their role (training or test) is visually distinguished by using solid (training) and dotted (test) outlines (DP2). Users can manually change whether an image is selected for training or test by clicking on it.

When pressing on the “train” button (Fig. 3.F), the system starts to train the model with the provided dataset in the background. The system can display training logs and will show a message when it is finished to inform the user (DP5). After training, users can press the test button and test (Fig. 3.F). The system then shows a confusion matrix of true and false positive and negatives (Fig. 3.J), as well as four selected evaluation metrics (accuracy, precision, recall, and F1-score, Fig. 3.I). In-depth result analysis happens in the next tab. When users are satisfied with the test results and done experimenting with different visualization mappings in other tabs, they can perform the validation with a portion of data (this is configurable in Tab 1). The four metrics and the confusion matrix also update with the validation results. Performing validation means the end of all operations, thus users should no longer modify any tabs.

Tab 5 (Fig. 4) allows users to group the images according to their classification results and visually inspect them to identify patterns. The goal is to makesense from the results and generate insights for improvements. This tab displays two confusion matrices for training and testing (marked G and H in Fig. 4 respectively) and four panels (C, D, E, and F) for image visualization. Each value on the confusion matrix is associated with a checkbox; enabling a checkbox will display all images of this group on the selected panel. All panels can thus be configured separately and flexibly adjusted to help users understand the results through comparison (DP4).

![](images/004bf9e13b2fb0282ba71e64385b515a2e12c1c1a2f045b4f96c2fb82be3a9e5.jpg)  
Figure 2: Tab 3: Creating visualizations.

![](images/8c3e451c5bcc0319aa922a17e2629d07cf1429105eca0211d92c1cfdbd2d4007.jpg)  
Figure 3: Algorithm training and tests.

![](images/3fa84cd6fa3ada6af1a31de971337f80a516e1461649f0b1bfc0a93b14734731.jpg)  
Figure 4: Algorithm training and tests.

This design is intended to facilitate the comparison of different groups of results visually. Users visually assess the differences and similarities between cases which were correctly and incorrectly classified, perhaps allowing the user to improve the visual mapping for better classification or to avoid errors. Visual comparisons between the categories can reveal the patterns that make them different.

## 5 CUVE: Evaluation

We conducted a user study (IRB-approved) where participants developed an ML model with both CUVE and Python during a single 60-75 min session, for \$15. We recruited 16 participants (6 female and 10 male, aged between 22 and 39) in two groups: nonexperts only knew what ML is but did not have practical experience, while experts had created at least one ML model by themselves. All at least basic knowledge of Python programming and ML.

## 5.1 Study Design, Tasks and Procedure

All participants completed two phases for each of two conditions (C1:VLAIR and a C2:Jupyter Notebook).

The order of conditions was counterbalanced; participants used a different dataset in each condition.

In Phase 1 the experimenter guided the participant through a multi-step supervised classification process. For the CUVE condition, we pre-loaded the dataset. Then, tab by tab, we asked them what they expected was the functionality of each tab, and then taught them to use it. For constructing the visualization, we guided them step by step to create an example visualization mapping (Fig. 5). For the Python condition we provided the Python code in four cells: data loading, feature engineering, model training and classification, and results display. A set of functions allowed participants to quickly modify features, including deleting, creating new features based on existing ones, and performing arithmetic operations and to query the results. In Phase 2 participants played freely with each system and modified it. After each condition they filled a questionnaire. At the end, asked follow-up questions about their preferences and perceived differences between the two tools.

![](images/b346ac73576e63fb52abb5fd319e7a9a672e537e369489f1bfc941c30f531951.jpg)

Figure 5: An example of the Visualization that the instructor guided the participate to create during the study, using water probability datasets.  
![](images/b8424cf0ea0381780e4fc62eb79d666481c17bacae4933273b54630d560076ba.jpg)  
Figure 6: Average differences in aggregated SUS scores. Error bars represent 95% bootstrapped CI.

## 6 Results

We gathered and analyzed both quantitative data (SUS and preference ratings)and qualitative data (comments, feedback, and answers to the interview questions) from each of the 16 participants.

## 6.1 Quantitative Analysis

Non-experts rated CUVE with a conclusively higher SUS score than Python (on average, a difference of 16.3—see Figure 6). When considering all participants together, the difference is only 6.4 on average. Experts also generally rated Python higher than nonexperts (17.1 avg. difference).

4 out of 8 experts preferred CUVE. For nonexperts 6 preferred the Python condition.

## 6.2 Qualitative Analysis

We combined inductive and deductive coding with thematic analysis to analyze participants’ comments. The authors iterated on a codebook that, in its final version, contained 45 codes and 5 themes. Here we only summarize the most salient results.

All participants mentioned that CUVE was helpful to understand the data and/or to find patterns when looking at the small multiples (e. g., “because like even when I was seeingfor 20 seconds, I started seeking like a pattern ...” [P1], “it’s easier to see when I have visuals to compare more quickly than just numbers [P12]).

Three experts (P2, P5, and P8) and three nonexperts (P1, P6, P13) said that it was easy to understand why a particular prediction is made and to debug the model (e. g., “I think that one could be used when you are stuck and you don’t know what is wrong with your model” [P8].

When using Python, twelve participants complained about the difficulty of understanding or finding patterns in data. For example, P1 said that “I would need time to look at them because it’s hard to figure out if there is a similarity because I need to go through the numbers at least one by one”. P5 specifically mentioned that he could not understand why the model made that prediction (“I can’t understand why thefalse positive values have occurred here”). In contrast, four other participants (P4, P7, P8, and P10) who were all ML experts said that they were able to find at least some patterns in data (“Not much. But yes, It does” [P10]).

7 out of 16 participants (P2, P5, P6, P7, P11, P12, and P15) confirmed that CUVE helped them learn/understand ML without coding “I think it would benefit new users or new people that are learning machine learning” [P12],“help you visualize the data like easily and you do not really have to know how to code. ... It has a lot of built-in features that you can use easily without programming...” [P11]). Nevertheless, several participants (including P1, P3, P4, P5, P6, P7, and P8), insisted that some prior knowledge is still required.

Reusability came up as an issue of CUVE compared to the Python approach because they have to create a new visualization for every problem (P15) whereas they don’t need to do it with Python (P12 and P15).

## 7 Discussion, Limitations and Conclusion

Overall we have demonstrated, through the implementation of CUVE, the feasibility of a VLAIR approach as an end-user interactive system. CUVE also incorporates several small innovations: it allows users to construct visualizations visually and it offers comprehensive features to compare the small multiples visualizations generated through the visual mapping.

An initial evaluation showed that the approach could be viable for both experts and non-experts as a UI that does not require programming. Participants appreciate the improved ability to understand the data that the visual approach offers. However, participants remain skeptical of replacing programming completely and suspect that the non-programming approach might also have additional demands such as visualization expertise.

## REFERENCES

Adadi, A. and Berrada, M. (2018). Peeking inside the black-box: a survey on explainable artificial intelligence (xai). IEEE access, 6:52138–52160.

Apley, D. W. and Zhu, J. (2020). Visualizing the effects of predictor variables in black box supervised learning models. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 82(4):1059–1086.

Bach, S., Binder, A., Montavon, G., Klauschen, F., Muller,¨ K.-R., and Samek, W. (2015). On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 10(7):e0130140.

Byerly, A., Kalganova, T., and Ott, R. (2022). The Current State of the Art in Deep Learning for Image Classification: A Review. In Arai, K., editor, Intelligent Computing, pages 88–105, Cham. Springer.

Card, M. (1999). Readings in information visualization: using vision to think. Morgan Kaufmann.

Confalonieri, R., Coba, L., Wagner, B., and Besold, T. R. (2021). A historical perspective of explainable Artificial Intelligence. WIREs Data Mining and Knowledge Discovery, 11(1):e1391.

Doshi-Velez, F. and Kim, B. (2017). Towards A Rigorous Science of Interpretable Machine Learning. arXiv:1702.08608 [stat].

Gerelt-Od, A.-U., Kho, L., Ngo, A., and Yeh, A. Generating feature impact from individual conditional expectation plots.

Gohel, P., Singh, P., and Mohanty, M. (2021). Explainable ai: current status and future directions. arXiv preprint arXiv:2107.07045.

Halaharvi, S., Mendez, G. G., Mansoor, H., Yong, Q., Mi-´ lani, A. M. P., Storey, M.-A., and Nacenta, M. A. (2024). P-Inti: Interactive Visual Representation of Programming Concepts for Learning and Instruction. In Proc. IEEE (VL/HCC), pages 199–210.

Huron, S., Carpendale, S., Thudt, A., Tang, A., and Mauerer, M. (2014). Constructive visualization. In Proc. DIS, pages 433–442.

Ivanovs, M., Kadikis, R., and Ozols, K. (2021). Perturbation-based methods for explaining deep neural networks: A survey. Pattern Recognition Letters, 150:228–234.

Jiang, A., Nacenta, M. A., Terzic, K., and Ye, J. (2020). Visualization as intermediate representations (vlair) for human activity recognition. In Proceedings ofthe 14th EAI International Conference on Pervasive Computing Technologies for Healthcare, pages 201–210.

Jiang, A., Nacenta, M. A., and Ye, J. (2022). Visualizations as intermediate representations (vlair): An approach for applying deep learning-based computer vision to non-image-based data. Visual Informatics, 6(3).

Kalasampath, K., Spoorthi, K. N., Sajeev, S., Kuppa, S. S., Ajay, K., and Maruthamuthu, A. (2025). A Literature Review on Applications of Explainable Artificial Intelligence (XAI). IEEE Access, 13:41111–41140.

Linardatos, P., Papastefanopoulos, V., and Kotsiantis, S. (2020). Explainable ai: A review of machine learning interpretability methods. Entropy, 23(1):18.

Loyola-Gonzalez, O. (2019). Black-box vs. white-box: Understanding their advantages and weaknesses from a practical point of view. IEEE access, 7:154096– 154113.

Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

Mendez, G. G., Nacenta, M. A., and Vandenheste, S.´ (2016). ivolver: Interactive visual language for visualization extraction and reconstruction. In Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems, pages 4073–4085.

Munzner, T. (2014). Visualization analysis and design. CRC press.

Mendez, G. G., Hinrichs, U., and Nacenta, M. A. (2017).´ Bottom-up vs. Top-down: Trade-offs in Efficiency, Understanding, Freedom and Creativity with InfoVis Tools. In Proc. CHI, pages 841–852. ACM.

Mendez, G. G., Nacenta, M. A., and Hinrichs, U. (2018).´ Considering Agency and Data Granularity in the Design of Visualization Tools. In Proc. CHI, CHI ’18, pages 1–14, New York, NY, USA. ACM.

Petsiuk, V., Das, A., and Saenko, K. (2018). Rise: Randomized input sampling for explanation of black-box models. arXiv preprint arXiv:1806.07421.

Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). ” why should i trust you?” explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1135–1144.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings ofthe IEEE international conference on computer vision, pages 618–626.

Selvaraju, R. R., Das, A., Vedantam, R., Cogswell, M., Parikh, D., and Batra, D. (2016). Grad-cam: Why did you say that? arXiv preprint arXiv:1611.07450.

Shrikumar, A., Greenside, P., and Kundaje, A. (2017). Learning important features through propagating activation differences. In International conference on machine learning, pages 3145–3153. PMLR.

Singh, A., Thakur, N., and Sharma, A. (2016). A review of supervised machine learning algorithms. In International Conference on Computing for Sustainable Global Development (INDIACom), pages 1310–1315.

Wright, R. (1950). Interpreting black-box machine learning models using partial dependence and individual conditional expectation plots. Exploring SAS® Enterprise Miner Special Collection, 2018.

Zeng, M., Nguyen, L. T., Yu, B., Mengshoel, O. J., Zhu, J., Wu, P., and Zhang, J. (2014). Convolutional Neural Networks for human activity recognition using mobile sensors. In International Conference on Mobile Computing, Applications and Services, pages 197–205.

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A. (2016). Learning deep features for discriminative localization. In Proc. CVPR, pages 2921–2929.