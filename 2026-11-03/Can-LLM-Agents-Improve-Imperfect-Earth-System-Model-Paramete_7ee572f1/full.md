# Can LLM Agents Improve Imperfect Earth System Model Parameterizations?

Kara D. Lamb Department of Earth and Environmental Engineering Learning the Earth with Artificial Intelligence and Physics Center Columbia University New York, NY 10027 kl3231@columbia.edu

## Abstract

Earth system models (ESMs) rely on parameterizations to represent unresolved processes such as convection, cloud microphysics, and turbulence. However, translating advances in physical understanding into stable, validated parameterizations and integrating them into ESMs often takes years. Agentic AI could accelerate this process and enable more systematic exploration of design choices. Using warm-rain collision–coalescence as a test case, we evaluate agents’ ability to synthesize scientific literature, reimplement published bulk parameterizations, and propose new functional forms as differentiable JAX code. We compare conventional parameterizations, data-driven emulators, and agent-generated closures against high-fidelity superdroplet simulations, repeating unattended experiments across ten language models. With expert guidance or iterative feedback, agents produced executable, differentiable, mass-conserving code, although unattended performance depended strongly on the scheme and language model. Iterative feedback lowered median rollout error for eight of ten models. Retuned physical schemes matched or outperformed many learned parameterizations at substantially lower computational cost, but two of 100 unattended agent-generated closures achieved slightly lower rain-mass error than the most accurate retuned scheme on held-out simulations.

## 1 Introduction

Earth System Models (ESMs) rely on parameterizations to represent unresolved processes such as clouds, turbulence, and ocean eddies. Recent advances in machine learning (ML) have enabled emulation of model components (Gentine et al., 2018; Rasp et al., 2018; Bolton and Zanna, 2019) and even full model surrogates (Allen et al., 2025; Bi et al., 2023; Lam et al., 2023; Price et al., 2025). Despite these promising advances, physically grounded parameterizations remain essential for interpretability, scientific insight, and robustness for out-of-distribution climate prediction(Lamb et al., 2026). However, it often takes years to translate new physical understanding into stable, validated parameterizations that can be integrated into ESMs.

Although ML has been widely applied to individual modeling tasks, improving complex scientific models requires coordinated efforts involving literature review, data analysis, code development, calibration, and evaluation. Recent progress in AI agents and multi-agent AI systems offers a promising method to speed up this process (Tom et al., 2024; Gottweis et al., 2025). Specialized agents can syn thesize scientific knowledge, analyze data, generate code, and evaluate candidate models in response to quantitative feedback (OpenAI, 2026). Although agentic approaches are beginning to transform laboratory sciences, their ability to improve complex, high-dimensional computational models such as ESMs remains largely unexplored. Initial applications for ESM development have included the agentic translation of legacy Fortran model components into differentiable JAX code(Lahlou et al., 2026) demonstrating these systems can assist with specific bottlenecks in model development.

![](images/78a92b69578cf1e918b6d44288903fac31f09e332d3f9e3980b61b3d8ff6edbe.jpg)  
Figure 1: Overview of the analysis and methodology.

Here we explore whether agentic AI can be used to develop and systematically improve the subgrid-scale parameterizations used in ESMs, going beyond translating existing code to explore where agentic AI can accelerate the model development process. As a proof of concept, we focus on warm-rain collision–coalescence, a process for which high-fidelity reference simulations and a range of established physical and machine-learning parameterizations provide a controlled test bed. We evaluate whether agentic workflows can synthesize information from the literature, translate existing schemes into differentiable JAX code, propose and refine new closures, calibrate model parameters, and compare competing approaches within a common framework. By comparing expert-guided, single-shot, and iterative workflows, we differentiate the ability to generate functional code from the more demanding task of producing physically consistent, numerically stable, and more accurate (with respect to the reference simulations) parameterizations. This analysis highlights where agentic AI can already accelerate parameterization development, and where expert intervention remains essential.

We ask three questions: (1) Can LLM agents reliably translate published parameterizations into accurate, differentiable code? (2) Does automated evaluation and feedback improve agent-generated closures? and (3) Do agent-generated or published learned closures outperform retuned physical parameterizations?

## 2 An Atmospheric Cloud Parameterization Test Case

As a test case, we focus on warm-rain collision–coalescence, an important cloud microphysical process controlling precipitation formation, which is typically represented in ESMs using simplified bulk parameterizations (Figure 1a). In a two-moment scheme, the cloud (non-precipitating) and rain (precipitating) droplet populations are each represented by their grid-averaged number and mass mixing ratios, giving four prognostic variables $( N _ { c } , L _ { c } , \dot { N _ { r } } , L _ { r } )$ . Their evolution is described by four parameterized process rates (autoconversion, accretion, cloud self-collection, and rain selfcollection), expressed as four coupled ODEs that conserve total mass. Classical schemes use analytic or empirically derived expressions for these rates (e.g. Kessler, 1969; Berry and Reinhardt, 1974a; Khairoutdinov and Kogan, 2000; Seifert and Beheng, 2001), while recent approaches have used neural network regression (Seifert and Rasp, 2020; Sharma and Greenberg, 2025).

Because time-resolved observations of collision–coalescence inside clouds are not currently technologically possible (Morrison et al., 2020), we evaluate the parameterizations against high-fidelity McSnow simulations(Brdar and Seifert, 2018). McSnow explicitly represents the stochastic collisioncoalescence process using Lagrangian superdroplets, providing a high-fidelity reference that is too computationally expensive for use in ESMs. We use the dataset developed by Sharma and Greenberg (2025), which contains 819 initial conditions: 719 for training and validation and 100 for testing. The McSnow configuration, initial-conditions, and additional details are provided in the Appendix.

## 3 Evaluation of Agentic AI for Parameterization Development

To explore agentic translation of historical schemes into JAX, reproduction of existing ML closures, and the proposal of new closures (Figure 1b), we evaluate three different paradigms for LLM code development: human-in-the-loop agentic AI, single-shot API calls to an LLM, and iterative API calls with feedback. In the first, an expert remains actively involved in guiding the agent and directing iterative refinements. In the latter two, we use controlled single-shot and iterative API calls to evaluate what an LLM can accomplish without human intervention. Expert-guided experiments used Claude Opus 4.8, and unattended experiments were run with ten different language models (Appendix I), so differences between the two settings reflect both expert supervision and variations across LLM models. The resulting implementations are evaluated for agreement with reference schemes and compared in a common evaluation framework (Figure 1c).

Expert-in-loop literature scheme translation We first evaluate whether agentic AI can translate established collision–coalescence parameterizations into a common JAX framework. We selected 4 traditional parameterizations that span a range of formulations and levels of complexity: (Kessler, 1969; Berry and Reinhardt, 1974a; Khairoutdinov and Kogan, 2000; Seifert and Beheng, 2001). Because these schemes use different prognostic variables and include both single- and two-moment representations of the droplet size distributions, their implementation requires more than a direct translation of individual equations. The agent must reconcile variable definitions, units, and process-rate conventions while mapping each scheme onto a standardized model interface and state representation. Additional details of these parameterizations and their functional forms are given in the Appendix.

Using expert-in-the-loop development, we constructed a JAX-native bulk microphysics library containing differentiable implementations of all four schemes. Very little expert guidance was required, with only 5 total prompts and < 1 hour of active time required to translate all 4 schemes (Table 6). All 4 resulting parameterizations were differentiable, mass conserving, and stable during prognostic integration. These expert-supervised implementations are used as reference baselines for evaluating the accuracy and reliability of the less closely supervised LLM translations.

Expert-in-the-loop reconstruction of machine-learning baselines We next used expert-in-theloop agentic development to reconstruct two machine learned collision–coalescence parameterizations in JAX: the process-rate neural networks of Seifert and Rasp (2020) and SuperdropNet (Sharma and Greenberg, 2025). Given the original publications and McSnow training data, the agent closely reproduced the single-step and prognostic behavior of the Seifert and Rasp model. For SuperdropNet, it reconstructed the architecture and loss function but did not reproduce the reported autoregressive performance. The primary difficulty was reproducing the optimization procedure and training curriculum rather than implementing the network itself. The two reconstructions required 4.0 and 7.1 hours of active expert involvement, respectively (Table 6). Additional methodological details and reconstructed evaluation figures are provided in Appendix H.

Expert-in-the-loop development of a novel physics-constrained Neural ODE closure We also used an expert-in-the-loop workflow to develop a physics-constrained Neural ODE for the four bulk moments (Chen et al., 2019). The agent proposed an architecture that represented the learned tendencies through physically interpretable process rates and enforced mass conservation by construction. Development required substantial expert involvement: 131 prompts and approximately 22 active hours (Table 6). Although the resulting model captured the qualitative evolution of the McSnow trajectories, it remained less accurate than several classic and machine learned baselines (Section 4). The agent contributed useful architectural ideas by recombining principles from existing physical schemes, but did not produce a superior parameterization.

Single-shot API calls for historic parameterization reconstruction We tested whether the four literature parameterizations could be implemented through unattended, single-shot LLM calls. For each scheme, each of ten language models (Appendix I) received a standardized prompt, the original publication as a PDF, and the target JAX interface. We performed ten independent calls per scheme and model with no shared context, allowing us to evaluate variability in translating scientific descriptions into executable code.

Each implementation was tested for successful execution, automatic differentiation, mass conservation, and stability over a six-hour box-model rollout (Figure 1c). Successful implementations were then compared with the expert-supervised JAX library and McSnow reference simulations. Performance depended strongly on the scheme and model (Figure 5), with the family of Claude models performing best at the translation task. Across the ten models, Kessler and Khairoutdinov & Kogan translations passed all structural tests in 99/100 and 94/100 cases, compared with 85/100 for Seifert & Beheng and 74/100 for Berry & Reinhardt. For Berry & Reinhardt, the models received Parts I and II of the series (Berry and Reinhardt, 1974a), which describe the numerical method and the evolution of the drop spectrum; the rate equations appear in Part IV (Berry and Reinhardt, 1974b), which was not provided. Its lower pass rate therefore partly reflects missing information in the inputs, and its accuracy is measured against our Long-kernel implementation (Appendix) rather than the Part IV equations.

Automated LLM closure generation with iterative feedback. We next tested whether LLMs could develop parameterizations autonomously using limited examples and automated feedback. Each model received a small sample of McSnow results, a common set of input variables, and physical and computational requirements. We compared single-shot generation with iterative refinement, in which models revised their candidates using feedback from training simulations. Final performance was assessed on 100 held-out simulations that were never used for feedback or candidate selection.

Across ten language models, iterative refinement over four rounds reduced median rain-mass error for eight models, although the improvement was statistically significant for only one (Figure 6). For half of the models, the best of four independent attempts performed at least as well as the median iteratively refined candidate. Differences among language models were larger than differences between singleshot and iterative performance. Nevertheless, some models consistently produced accurate closures: overall, 19 of 100 iteratively refined candidates outperformed the most accurate published physical scheme, Seifert & Beheng, including seven of ten candidates from Claude Opus 5.5, and two, both from Claude Opus 5.5, outperformed Seifert & Beheng retuned against McSnow (Appendix I). The best performing Claude scheme retained the Seifert & Beheng autoconversion moment dependence but modified the correction functions and number budgets (Table 8).

Extending refinement from four to twelve rounds improved performance for both Claude Opus 5.5 and Gemini 4 Argon (Figure 7). The larger gain occurred for Gemini, whose shorter searches frequently failed to produce a working closure, and the best candidate from each model outperformed published Seifert & Beheng, with Claude’s best candidate out-performing the retuned scheme. These results suggest that automated feedback can help LLMs discover accurate parameterizations, but its effectiveness depends strongly on the language model and the duration of the search.

## 4 Systematic Comparison of Implemented Parameterizations

Implementing all parameterizations within a common JAX framework allows them to be compared systematically using the same state representation, reference simulations, numerical integration, and metrics. We evaluate both rain-formation timing and the complete rollout. The metric $t _ { 1 0 }$ measures when 10% of the initial cloud mass has been converted to the rain category (in minutes), while rollout accuracy is measured using mean absolute percentage error (MAPE) across $\left( N _ { c } , L _ { c } , N _ { r } , L _ { r } \right)$ on the held-out McSnow test set; MAPE for $L _ { r }$ is shown in Figure 2. Further details are in Appendix E.

Because classical schemes were calibrated for particular reference models and initial conditions, their published coefficients may not provide the best achievable performance on the McSnow data set. We therefore retuned each scheme on the training set and evaluated both published and retuned versions on 100 test simulations, to separate the limitations of the functional form from those of its original calibration. Retuning substantially improved most classical schemes. We compared them with the expert-guided JAX implementations of Seifert and Rasp, SuperdropNet, and the physics-constrained NODE, as well as the two best-performing invented schemes (from Claude Opus 5.5). Advantages of the machine learned models over the published schemes often narrowed or disappeared after retuning, demonstrating the importance of comparing learned closures with recalibrated physical baselines.

The suitability of a parameterization for integration in an ESM depends not just on accuracy, but also on numerical stability, physical consistency, computational cost, and robustness outside the training distribution. We quantify complexity using floating-point operations per model step, a hardwareindependent measure that does not capture memory use or realized runtime. Several retuned physical schemes achieved errors comparable to the learned closures at substantially lower cost, although SuperdropNet most accurately captured the timing of the cloud-to-rain transition (Figure 2). No single scheme is uniformly superior across all evaluation criteria. This comparison illustrates the value of treating parameterization development as a systematic model-selection problem.

![](images/7f8a86e3afe02b48c6605ebdc71bebb58d8c232288bc75600162abf5bf7ec9b0.jpg)

![](images/9d34e8f4ea72b92153683db1ee9c025f5648a3d49f1ea17e5dcb9540e3f47ab4.jpg)  
Figure 2: Parameterization performance. (a) Mean error in rain-conversion timescales. Published and retuned coefficients are shown for the four literature schemes. (b) Rain-mass rollout error versus computational complexity, measured in floating-point operations per model step. Here, Superdropnet and Process-rate MLP refer to our reimplemented versions.

## 5 Discussion

Physically grounded parameterizations remain essential for providing the interpretability, physical insight, and robustness required for climate prediction. The results presented here demonstrate that agentic AI can reconstruct historical parameterizations as differentiable JAX code, reconstruct machine learned baselines, and propose new physics-constrained closures. However, the agents were generally more successful at satisfying structural requirements such as executability, differentiability, mass conservation, and numerical stability than at improving predictive accuracy (Table A). These capabilities could accelerate the currently slow cycle of parameterization development, calibration, and evaluation, while making comparisons among alternative formulations more systematic and reproducible. The next step is to extend this evaluation beyond an isolated collision–coalescence box model. Recently developed JAX-based models such as the JAX Circulation Model or LegoESM provide a pathway for testing compatible implementations across a hierarchy ranging from process-level benchmarks and single-column models to intermediate complexity climate models and, eventually, coupled Earth system simulations (Davenport et al., 2026; Gentine et al., 2026). Performance at the process scale does not guarantee good performance after coupling, where interactions with dynamics, thermodynamics, and other processes may amplify small errors or change the relative ranking of competing schemes. Although we focus here on one process, the workflow is not specific to it: any parameterization that can be written as a differentiable mapping from the resolved state to tendencies and evaluated against reference simulations can be developed in the same way.

Closing the loop between parameterization generation, integration, evaluation, and revision could provide a foundation for increasingly self-improving models. Such models could systematically incorporate new observations, identify conditions for which available reference data are inadequate, design targeted high-fidelity simulations, compare alternative physical and data-driven formulations, and use the resulting evidence to revise their parameterizations. The present study addresses only a subset of this cycle, but demonstrates several of the capabilities needed to support it. However, important challenges still remain. It is not yet clear whether agentic AI can generate genuinely improved physical representations rather than primarily recombining ideas from existing approaches. Agentic development can also produce increasingly large and complex code bases containing subtle or hallucinated errors that may be difficult to identify. Agent-generated parameterizations must therefore meet the same requirements for physical consistency, numerical stability, reproducibility, and scientific validation as human-generated code. For parameterizations incorporated into ESMs used for weather forecasting or climate adaptation and mitigation decisions, additional safeguards should include independent review, automated conservation and stability tests, and out-of-distribution evaluation.

## Acknowledgments and Disclosure of Funding

We acknowledge funding and Google Cloud Platform support from a Google Award for Machine Learning Research & Education with TPUs. We also acknowledge funding and computational resources from the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research, as part of the Genesis Mission National Science and Technology Challenges under Award # DE-SC0026566. We acknowledge support from NSF through the Learning the Earth with Artificial Intelligence and Physics (LEAP) Science and Technology Center (STC) (Award #2019625).

## References

Pierre Gentine, Mike Pritchard, Stephan Rasp, Gael Reinaudi, and Galen Yacalis. Could machine learning break the convection parameterization deadlock? Geophysical Research Letters, 45(11): 5742–5751, 2018.

Stephan Rasp, Michael S Pritchard, and Pierre Gentine. Deep learning to represent subgrid processes in climate models. Proceedings ofthe national academy ofsciences, 115(39):9684–9689, 2018.

Thomas Bolton and Laure Zanna. Applications of deep learning to ocean data inference and subgrid parameterization. Journal ofAdvances in Modeling Earth Systems, 11(1):376–399, 2019.

Anna Allen, Stratis Markou, Will Tebbutt, James Requeima, Wessel P Bruinsma, Tom R Andersson, Michael Herzog, Nicholas D Lane, Matthew Chantry, J Scott Hosking, et al. End-to-end data-driven weather prediction. Nature, pages 1–8, 2025.

Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate mediumrange global weather forecasting with 3d neural networks. Nature, 619(7970):533–538, 2023.

Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful medium-range global weather forecasting. Science, 382(6677):1416–1421, 2023.

Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, et al. Probabilistic weather forecasting with machine learning. Nature, 637(8044):84–90, 2025.

Kara D. Lamb, Clare E. Singer, Kaitlyn Loftus, Hugh Morrison, Margaret Powell, Joseph Ko, Jatan Buch, Arthur Z. Hu, Marcus van Lier Walqui, and Pierre Gentine. Perspectives on Systematic Cloud Microphysics Scheme Development With Machine Learning. Journal ofAdvances in Modeling Earth Systems, 18(1):e2025MS005341, 2026. ISSN 1942-2466. doi: 10.1029/2025MS005341. URL https://onlinelibrary.wiley.com/doi/abs/10.1029/2025MS005341. \_eprint: https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2025MS005341.

Gary Tom, Stefan P. Schmid, Sterling G. Baird, Yang Cao, Kourosh Darvish, Han Hao, Stanley Lo, Sergio Pablo-García, Ella M. Rajaonson, Marta Skreta, Naruki Yoshikawa, Samantha Corapi, Gun Deniz Akkoc, Felix Strieth-Kalthoff, Martin Seifrid, and Alán Aspuru-Guzik. Self-Driving Laboratories for Chemistry and Materials Science. Chemical Reviews, 124(16):9633–9732, August 2024. ISSN 0009-2665. doi: 10.1021/acs.chemrev.4c00055. URL https://doi.org/10.1021/ acs.chemrev.4c00055.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, Khaled Saab, Dan Popovici, Jacob Blum, Fan Zhang, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Pushmeet Kohli, Yossi Matias, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R. D. Costa, José R. Penadés, Gary Peltz, Yunhan Xu, Annalisa Pawlosky, Alan Karthikesalingam, and Vivek Natarajan. Towards an AI co-scientist, February 2025. URL http://arxiv.org/abs/2502.18864. arXiv:2502.18864 [cs].

OpenAI. AI as a Scientific Collaborator. January 2026. URL https://cdn.openai.com/pdf/ f4b4a5da-b2de-418d-9fcd-6b293e9dc157/oai\_ai-as-a-scientific-collaborator\_ jan-2026.pdf.

Aya Lahlou, Linnia Hawkins, and Pierre Gentine. Systematic LLM Translation of Legacy Scientific Code to Differentiable Frameworks: Application to a Land Surface Model, June 2026. URL http://arxiv.org/abs/2606.07681. arXiv:2606.07681 [cs.SE].

Edwin Kessler. On the distribution and continuity of water substance in atmospheric circulations. In On the distribution and continuity of water substance in atmospheric circulations, pages 1–84. Springer, 1969.

Edwin X Berry and Richard L Reinhardt. An analysis of cloud drop growth by collection: Part i. double distributions. Journal of Atmospheric Sciences, 31(7):1814–1824, 1974a.

Marat Khairoutdinov and Yefim Kogan. A New Cloud Physics Parameterization in a Large-Eddy Simulation Model of Marine Stratocumulus. Monthly Weather Review, 128(1):229–243, 2000. ISSN 1520-0493, 0027-0644. doi: 10.1175/1520-0493(2000)128<0229:ANCPPI>2.0.CO;2. Publisher: American Meteorological Society Section: Monthly Weather Review.

Axel Seifert and Klaus D Beheng. A double-moment parameterization for simulating autoconversion, accretion and selfcollection. Atmospheric research, 59:265–281, 2001.

Axel Seifert and Stephan Rasp. Potential and limitations of machine learning for modeling warmrain cloud microphysical processes. Journal of Advances in Modeling Earth Systems, 12(12): e2020MS002301, 2020.

Shivani Sharma and David S. Greenberg. Superdropnet: A stable and accurate machine learning proxy for droplet-based cloud microphysics. Journal of Advances in Modeling Earth Systems, 17(6):e2024MS004279, 2025. doi: https://doi.org/10.1029/2024MS004279. e2024MS004279 2024MS004279.

Hugh Morrison, Marcus van Lier?Walqui, Ann M. Fridlind, Wojciech W. Grabowski, Jerry Y. Harrington, Corinna Hoose, Alexei Korolev, Matthew R. Kumjian, Jason A. Milbrandt, Hanna Pawlowska, Derek J. Posselt, Olivier P. Prat, Karly J. Reimel, Shin?Ichiro Shima, Bastiaan Van Diedenhoven, and Lulin Xue. Confronting the Challenge of Modeling Cloud and Precipitation Microphysics. Journal ofAdvances in Modeling Earth Systems, 12(8), 2020. ISSN 1942-2466, 1942-2466. doi: 10.1029/2019MS001689.

Slavko Brdar and A Seifert. Mcsnow: A monte-carlo particle model for riming and aggregation of ice particles in a multidimensional microphysical phase space. Journal ofadvances in modeling earth systems, 10(1):187–206, 2018.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural Ordinary Differential Equations, 2019. arXiv:1806.07366 [cs].

Edwin X Berry and Richard L Reinhardt. An analysis of cloud drop growth by collection: Part IV. a new parameterization. Journal of Atmospheric Sciences, 31(8):2127–2135, 1974b.

Ellen H. Davenport, J. Varan Madan, Rebecca Gjini, Jared Brzenski, Nick Ho, Tien-Yiao Hsu, Yueshan Liang, Zhixing Liu, Veeramakali Manivannan, Eric Pham, Rohith Vutukuru, Andrew I. L. Williams, Zhiqi Yang, Rose Yu, Nicholas J. Lutsko, Stephan Hoyer, and Duncan Watson-Parris. JCM v1.0: A Differentiable, Intermediate-Complexity Atmospheric Model. EGUsphere, pages 1– 20, January 2026. doi: 10.5194/egusphere-2025-6266. URL https://egusphere.copernicus. org/preprints/2026/egusphere-2025-6266/.

Pierre Gentine, Dhruv Balwada, Aytaç Paçal, Linnia Hawkins, Alistair Adcroft, Hang Fan, Jianing Fang, Aya Lahlou, Kara D Lamb, David M Lawrence, et al. legoesm: a modular, differentiable, multiscale, ai-ready earth system model built with ai agents. arXiv preprint arXiv:2610.11883, 2026.

William D Hall. A detailed microphysical model within a two-dimensional dynamic framework: Model description and preliminary results. Journal ofAtmospheric Sciences, 37(11):2486–2507, 1980.

KD Beheng. A parameterization of warm cloud microphysical conversion processes. Atmospheric Research, 33(1-4):193–206, 1994.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The main claims are supported by our analysis.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss both the performance and failures of our current approaches.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: This paper does not include theoretical results.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We have provided details of the LLMs used, the experiments performed, and the relative performance of different approaches.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [No]

Justification: Since this is a workshop submission, we plan to release code upon acceptance of an expanded version of this paper in a peer-reviewed subject matter journal.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Details of the training and test data is given in the main paper and appendix. Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Information about error bars are provided in the relevant figure captions.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Time and computational resources are explicitly evaluated in the paper.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in this paper conforms to the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Caveats around the need to rigorously evaluate code for ESMs is discussed in the Discussion.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper poses no such risks.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, data and models are properly credited.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: Model code will be released upon publication of these results in a peer-reviewed journal. Data is accessible via the original SuperdropNet paper.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: Neither crowdsourcing nor human subjects were involved.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: Neither crowdsourcing nor human subjects were involved.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: Claude Code Opus 4.8 was used in the generation of code (as described in the manuscript) with human-in-the-loop oversight. Single-shot and iterative API calls to ten models were used to evaluate typical LLMs failure modes. Claude Code and ChatGPT were given access to the draft manuscript to provide targeted criticism and feedback based on the implemented code base.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.

## A Results of experiments

<table><tr><td>Research question</td><td>Answer</td></tr><tr><td>(1) Can LLM agents translate published schemes into differentiable code?</td><td>V Yes with light expert guidance; unattended success depends on the scheme and the model.</td></tr><tr><td>(2) Does automated feedback improve agent-generated closures?</td><td>A Structural validity improves reliably; accuracy gains are mod- est and depend on the model. Twelve-round searches im- proved performance for the two models examined.</td></tr><tr><td>(3) Do agent-generated or learned closures outperform retuned physical schemes?</td><td>A Two agent-generated closures do in rain-mass error, both from Claude Opus 5.5; learned closures only for the timing of rain onset.</td></tr></table>

<sub>Ta</sub>bl<sub>e 1: Answers to t</sub>h<sub>e t</sub>h<sub>ree researc</sub>h <sub>quest</sub>i<sub>ons.</sub> ✓ <sub>yes;</sub> . <sub>yes, w</sub>i<sub>t</sub>h <sub>caveats.</sub>

## B High fidelity reference simulations

As described in the main text, box model simulations using the McSnow Superdroplet Microphysics scheme (Brdar and Seifert, 2018) are used as high fidelity reference schemes for the collisioncoalescence process. These data sets are stored as arrays with dimensions

$$
{ \mathrm { ( t i m e , f e a t u r e , i n i t i a l c o n d i t i o n , e n s e m b l e m e m b e r ) } } = ( 3 5 9 9 , 1 8 , N , 1 0 0 ) ,\tag{1}
$$

where $N = 7 1 9$ for the training and validation data and $N = 1 0 0$ for the held-out test data. Model output is recorded every $\Delta t = 2 0 \mathrm { ~ s ~ }$ , and each initial condition contains 100 stochastic ensemble members; here we average over the 100 ensemble members for each set of initial conditions before fitting or optimizing parameters against the McSnow simulations. The 18 features include six cloud and rain moments, $\bar { ( L _ { c } , N _ { c } , Z _ { c } , L _ { r } , N _ { r } , Z _ { r } ) }$ ; diagnosed number- and mass-transfer rates associated with autoconversion and accretion; self-collection diagnostics; and the initial liquid water content, mean cloud-droplet radius, and droplet-distribution shape parameter, $( \mathrm { l w c } _ { 0 } , r _ { c , 0 } , \nu _ { 0 } )$ . Because the time required to convert cloud droplets to rain varies among initial conditions, shorter trajectories are padded to the maximum array length. A corresponding mask is applied so that padded values are excluded from training and evaluation.

The reference simulations use a hydrodynamic gravitational collection kernel,

$$
K ( x , y ) = \pi \left[ r ( x ) + r ( y ) \right] ^ { 2 } \left| v ( x ) - v ( y ) \right| E _ { \mathrm { c o l l } } ( x , y ) ,\tag{2}
$$

where x and y are droplet masses, $r ( x )$ is the corresponding droplet radius, v(x) is the terminal fall velocity, and $E _ { \mathrm { c o l l } } ( x , y )$ is the tabulated collision efficiency of Hall (1980) with the modifications of Beheng (1994). The geometric cross section and terminal-velocity difference describe the volume swept out by droplets of different sizes, while $E _ { \mathrm { c o l l } }$ accounts for the fraction of hydrodynamic encounters that result in a collision.

Each simulation is initialized by sampling droplets of total mass $L _ { 0 }$ from a generalized gamma distribution with shape parameter ν and mean droplet radius $\bar { r } _ { 0 } ; \nu = 0$ recovers an exponential distribution. Three parameters are varied, on a full factorial grid (Table 2). The box volume $( \mathrm { { 2 5 0 0 m ^ { 3 } } ) }$ and the superdroplet multiplicity (26500) are held fixed, as is the gamma DSD family.

Table 2: Initial conditions varied in the McSnow box-model ensemble. The grid is $1 3 \times 7 \times 9 = 8 1 9$ distributions, split into 719 training and 100 test cases, each run as a 100-member stochastic ensemble. Values follow Sharma & Greenberg (2025, Table 1), whose initial-condition set is inherited unchanged from Seifert & Rasp (2020).
<table><tr><td>Quantity</td><td>Symbol</td><td>Values</td></tr><tr><td>Initial liquid water content</td><td> $L _ { 0 }$ </td><td>{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 2.0} g m</td></tr><tr><td>Initial mean droplet radius</td><td> $\bar { r } _ { 0 }$ </td><td>{9, 10, 11, 12, 13, 14, 15} µm</td></tr><tr><td>Gamma shape parameter</td><td>ν</td><td>{0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4} (unitless)</td></tr></table>

## C Traditional Bulk parameterizations

Collision–coalescence leads to the generation of rain in clouds and is typically represented using a simplified bulk parameterization (Figure 1a). In a typical two-moment bulk representation for warmrain cloud microphysics, the droplet size distribution in each ESM model grid box is divided into cloud (non-precipitating) and rain (precipitating) droplet populations. Each population is represented by its number and mass mixing ratios, giving four prognostic variables $( \bar { N _ { c } } , \bar { L } _ { c } , N _ { r } , L _ { r } )$ that describes the microphysical state of the cloud; here N refers to number and L refers to mass, with c denoting cloud and r denoting rain populations. In the two category, two moment representation, the collision– coalescence process is four separate process rates: autoconversion (AU), which transfers mass and number from the cloud to the rain category; accretion $\mathrm { ( A C ) }$ , in which raindrops collect cloud droplets; and self-collection within the cloud $( \bar { \mathrm { S C } _ { c } } )$ and rain $( \mathsf { S C } _ { r } )$ categories. Bulk parameterizations express these four process rates as a set of coupled ODEs that are functions of the four prognostic moments and any prescribed properties of the droplet size distribution, while conserving total mass. Single moment schemes (such as Kessler, 1969) use only the two mass moments as representations of the cloud and rain populations.

In Table 3, we provide a summary of the 4 traditional bulk parameterizations for collision-coalescence that we re-implemented as differentiable code in JAX, and that are used as a comparison for datadriven and agentic AI parameterization development approaches. This list of bulk parameterizations for collision-coalescence is by no means comprehensive, but was selected to provide a range of parameterizations of varying sophistication and complexity. Kessler, 1969 is the simplest and predicts only the mass mixing ratios of the cloud and rain populations; Seifert and Beheng, 2001 is the most complex. All 4 parameterizations are still regularly used in atmospheric models.

Table 3: Implemented bulk collision–coalescence schemes. AU denotes autoconversion, AC accretion, and $S { \bf C } _ { r }$ and $\mathbf { S } \mathbf { C } _ { c }$ rain and cloud self-collection, respectively. Moment order denotes the number of predicted moments per hydrometeor category. The common box-model interface uses $( N _ { c } , L _ { c } , \bar { N _ { r } } , L _ { r } )$ ; schemes that do not predict number moments use diagnostic or prescribed treatments.
<table><tr><td>Reference</td><td>Moment order</td><td>Processes</td></tr><tr><td>(Kessler, 1969)</td><td>1</td><td>AU, AC</td></tr><tr><td>(Berry and Reinhardt, 1974a)</td><td>2</td><td>AU, AC</td></tr><tr><td>(Khairoutdinov and Kogan, 2000)</td><td>2</td><td>AU, AC</td></tr><tr><td>(Seifert and Beheng, 2001)</td><td>2</td><td> $\mathrm { A U } , \mathrm { A C } , \mathrm { S C } _ { r } , \mathrm { S C } _ { c }$ </td></tr></table>

Below, we provide the original equations for the 4 reference parameterizations, and highlight in red the parameters which are retuned against the McSnow simulations (training data set).

Kessler (1969): Threshold autoconversion, with $\updownarrow L _ { r } / \mathrm { d } t \ : = \ : k _ { 1 } ( L _ { c } \ : - \ : a )$ for $L _ { c } \ > \ a ,$ , where $k _ { 1 } = 1 0 ^ { - 3 } \mathrm { s ^ { - 1 } }$ and $a = 0 . 5 \mathrm { g } \mathrm { m } ^ { - 3 }$ , and accretion $k _ { 2 } E N _ { 0 } ^ { 1 / 8 } L _ { c } L _ { r } ^ { 7 / 8 }$ with $k _ { 2 } = 6 . 9 6 \times 1 0 ^ { - 4 } , E = 1$ and $N _ { 0 } = 1 0 ^ { 7 } \mathrm { m ^ { - 4 } }$ (Kessler, 1969, Table 4). Kessler works in water content rather than mixing ratio, so in SI units $k _ { \mathrm { a c } } = 2 . 2 0$ and no air density appears. $N _ { c }$ is treated diagnostically using a fixed mean mass ${ \bar { x } } _ { c } ,$ while $N _ { r }$ is not predicted. This provides the simplest reference scheme. Four parameters are retuned against the McSnow training set.

$$
\mathrm { a u t o c o n v e r s i o n : \quad } P _ { \mathrm { a u } } = k _ { \mathrm { a u } } \operatorname* { m a x } \bigl ( 0 , \ L _ { c } - L _ { c 0 } \bigr )\tag{3}
$$

$$
\begin{array} { r l } { \mathrm { a c c r e t i o n : ~ } } & { { } P _ { \mathrm { a c } } = k _ { \mathrm { a c } } L _ { c } L _ { r } ^ { \beta } } \end{array}\tag{4}
$$

$$
\frac { \mathrm { d } L _ { r } } { \mathrm { d } t } = P _ { \mathrm { a u } } + P _ { \mathrm { a c } } , \qquad \frac { \mathrm { d } L _ { c } } { \mathrm { d } t } = - \frac { \mathrm { d } L _ { r } } { \mathrm { d } t } , \qquad \frac { \mathrm { d } N _ { c } } { \mathrm { d } t } = \frac { \mathrm { d } L _ { c } } { \mathrm { d } t } \frac { N _ { c } } { L _ { c } } , \qquad \frac { \mathrm { d } N _ { r } } { \mathrm { d } t } = 0 .\tag{5}
$$

Kessler predicts no number concentrations; we add a proportional cloud-number sink, which holds the mean drop mass $\hat { x } _ { c }$ constant. The rain number concentration $N _ { r }$ is not predicted, so no number parameters appear.

Berry & Reinhardt (1974), two-moment form: Berry and Reinhardt (1974a) derived autoconversion from numerical solutions of the collection equation; their rate equations are given in Part IV of the series (Berry and Reinhardt, 1974b). The scheme implemented here is not those rate equations but the corresponding two-moment form for the Long (1974) kernel, $\mathrm { d } L _ { r } / \mathrm { d } t | _ { \mathrm { A U } } = C _ { \mathrm { A U } } L _ { c } ^ { 2 } \bar { x } _ { c } ^ { 2 }$ with accretion $k _ { \mathrm { c r } } L _ { c } L _ { r }$ (Seifert and Beheng, 2001, Eqs. 16 and 21): the limit of Seifert & Beheng without the universal functions that describe the feedback from existing rain on autoconversion and accretion. Three parameters are optimized:

$$
P _ { \mathrm { a u } } = C _ { \mathrm { a u } } \ L _ { c } ^ { 2 } { \bar { x } } _ { c } ^ { 2 }\tag{6}
$$

$$
P _ { \mathrm { a c } } = k _ { \mathrm { c r } } ~ L _ { c } L _ { r }\tag{7}
$$

$$
\frac { \mathrm { d } L _ { r } } { \mathrm { d } t } = P _ { \mathrm { a u } } + P _ { \mathrm { a c } } , \qquad \frac { \mathrm { d } N _ { r } } { \mathrm { d } t } = \frac { P _ { \mathrm { a u } } } { x _ { r } ^ { * } } , \qquad \frac { \mathrm { d } N _ { c } } { \mathrm { d } t } = - \frac { \nu _ { 2 } } { \nu _ { 1 } } \frac { P _ { \mathrm { a u } } } { \bar { x } _ { c } } - \frac { P _ { \mathrm { a c } } } { \bar { x } _ { c } } ,\tag{8}
$$

with the shape-parameter ratio $\nu _ { 2 } / \nu _ { 1 } = 3 / 2 ( \nu _ { c } = 1 )$ held fixed. $\boldsymbol { x } _ { r } ^ { * }$ is the mass of a newly-formed raindrop.

Khairoutdinov & Kogan (2000) Empirical power-law fits to large-eddy simulations of marine stratocumulus, including $\mathrm { d } { { q } _ { r } } / \mathrm { d } t = 1 3 5 \mathrm { { 0 } } q _ { c } ^ { 2 . 4 7 } ( \dot { N } _ { c } / 1 0 ^ { 6 } ) ^ { - 1 . 7 9 }$ for autoconversion and $6 7 \left( q _ { c } q _ { r } \right) ^ { 1 . 1 5 }$ for accretion. New drizzle drops have radius $r _ { 0 } = 2 5 \mu \mathrm { m }$ , and the drops lost from the cloud have the mean cloud-drop mass (Khairoutdinov and Kogan, 2000, Eqs. 32 and 35); the scheme has no rain self-collection. The scheme is based on regression within a specific cloud regime rather than an explicit process-rate theory. Six parameters are retuned: both prefactors, all three power-law exponents and the new-drop mass $x _ { r 0 }$

$$
P _ { \mathrm { { a u } } } = \rho ~ A _ { \mathrm { { a u } } } ~ q _ { c } ^ { a } \left( 1 0 ^ { - 6 } N _ { c } \right) ^ { b }\tag{9}
$$

$$
P _ { \mathrm { a c } } = \rho A _ { \mathrm { a c } } \left( q _ { c } q _ { r } \right) ^ { c }\tag{10}
$$

$$
\frac { \mathrm { d } L _ { r } } { \mathrm { d } t } = P _ { \mathrm { a u } } + P _ { \mathrm { a c } } , \qquad \frac { \mathrm { d } N _ { c } } { \mathrm { d } t } = - \frac { P _ { \mathrm { a u } } + P _ { \mathrm { a c } } } { \bar { x } _ { c } } , \qquad \frac { \mathrm { d } N _ { r } } { \mathrm { d } t } = \frac { P _ { \mathrm { a u } } } { x _ { r 0 } } .\tag{11}
$$

Seifert & Beheng (2001) The most complete scheme considered here. It combines autoconversion for the Long (1974) kernel with the universal function $\Phi _ { \mathrm { a u } } ( \tau )$ describing the cloud–rain transition, accretion modified by $\Phi _ { \mathrm { a c } } ( \tau )$ , and cloud and rain self-collection. It predicts all four moments directly and can become stiff at high $L _ { c } .$ . Eight fitted parameters are retuned, including the two constants of the autoconversion universal function $\Phi _ { \mathrm { a u } }$ and the one in the accretion function $\Phi _ { \mathrm { a c } }$

$$
\Phi _ { \mathrm { a u } } ( \tau ) = \alpha \tau ^ { \beta } \bigl ( 1 - \tau ^ { \beta } \bigr ) ^ { 3 }\tag{12}
$$

$$
P _ { \mathrm { a u } } = C _ { \mathrm { a u } } L _ { c } ^ { 2 } { \bar { x } } _ { c } ^ { 2 } \left[ 1 + { \frac { \Phi _ { \mathrm { a u } } ( \tau ) } { ( 1 - \tau ) ^ { 2 } } } \right]\tag{13}
$$

$$
\Phi _ { \mathrm { a c } } ( \tau ) = \left( \frac { \tau } { \tau + \tau _ { 0 } } \right) ^ { 4 }\tag{14}
$$

$$
P _ { \mathrm { a c } } = k _ { \mathrm { c r } } L _ { c } L _ { r } \Phi _ { \mathrm { a c } } ( \tau )
$$

$$
{ \frac { \mathrm { d } L _ { r } } { \mathrm { d } t } } = P _ { \mathrm { a u } } + P _ { \mathrm { a c } } , \qquad { \frac { \mathrm { d } L _ { c } } { \mathrm { d } t } } = - { \frac { \mathrm { d } L _ { r } } { \mathrm { d } t } }\tag{15}
$$

(16)

$$
\begin{array} { r l r } { \frac { \mathrm { d } N _ { c } } { \mathrm { d } t } = } & { { } } & { \underbrace { - C _ { \mathrm { s c } } L _ { c } N _ { c } \bar { x } _ { c } } _ { \mathrm { d } t } \mathrm { ~  ~ \rho ~ } - \frac { P _ { \mathrm { a c } } } { \bar { x } _ { c } } } \end{array}\tag{17}
$$

$$
\frac { \mathrm { d } N _ { r } } { \mathrm { d } t } = \frac { P _ { \mathrm { a u } } } { x ^ { * } } - \underbrace { k _ { \mathrm { r r } } N _ { r } L _ { r } } _ { \mathrm { r a i n ~ s e l f - c o l l e c t i o n } }\tag{18}
$$

Following Seifert and Beheng (2001, Appendix A), autoconversion removes $2 P _ { \mathrm { a u } } / x ^ { * }$ cloud droplets and cloud self-collection is defined so that the two processes together remove $k _ { c } \frac { \nu + 2 } { \nu + 1 } L _ { c } ^ { 2 }$ , with $\begin{array} { r } { C _ { \mathrm { s c } } ~ = ~ k _ { c } \frac { \nu + 2 } { \nu + 1 } } \end{array}$ Accretion and rain self-collection use the same Long-kernel constant, $\begin{array} { r l } { k _ { r } } & { { } = } \end{array}$ 5.78 $\mathrm { m ^ { 3 } k g ^ { - 1 } s ^ { - 1 } }$ ; here $k _ { \mathrm { c r } }$ and $k _ { \mathrm { r r } }$ are retuned separately.

## D Retuned parameters in traditional bulk schemes against the McSnow training data set

Table 4 provides the retuned parameters in the 4 bulk parameterizations after the optimization against the McSnow training data set. Table 5 gives the training loss and held-out test loss for the 4 bulk parameterizations using their original published parameters and their retuned parameters, scored against the McSnow training and test data sets, respectively.

Table 4: Parameters for the 4 bulk reference schemes, retuned after optimizing against the McSnow training data set.
<table><tr><td>Scheme</td><td>Parameter</td><td>Published</td><td>Fitted</td><td>Ratio</td></tr><tr><td rowspan="4">Kessler</td><td> $k _ { \mathrm { a u } }$ </td><td> $1 . 0 0 \times 1 0 ^ { - 3 }$ </td><td> $6 . 4 8 \times 1 0 ^ { - 5 }$ </td><td>0.065</td></tr><tr><td> $L _ { c 0 }$ </td><td> $5 . 0 0 \times 1 0 ^ { - 4 }$ </td><td> $9 . 0 9 \times 1 0 ^ { - 4 }$ </td><td>1.82</td></tr><tr><td> $k _ { \mathrm { a c } }$ </td><td>2.201</td><td>2.063</td><td>0.94</td></tr><tr><td> $\beta$ </td><td>0.875</td><td>0.979</td><td>1.12</td></tr><tr><td rowspan="3">Berry &amp; Reinhardt</td><td> $C _ { \mathrm { a u } }$ </td><td> $6 . 8 1 \times 1 0 ^ { 1 8 }$ </td><td> $6 . 8 0 \times 1 0 ^ { 1 8 }$ </td><td>1.00</td></tr><tr><td> $k _ { \mathrm { c r } }$ </td><td> $5 . 7 8$ </td><td> $2 . 8 4 8$ </td><td>0.49</td></tr><tr><td> $\boldsymbol { x } _ { r } ^ { * }$ </td><td> $2 . 6 0 \times 1 0 ^ { - 1 0 }$ </td><td> $3 . 5 3 \times 1 0 ^ { - 9 }$ </td><td>13.6</td></tr><tr><td rowspan="6">Khairoutdinov &amp; Kogan</td><td> $A _ { \mathrm { a u } }$ </td><td>1350</td><td>2919</td><td>2.16</td></tr><tr><td> $a$ </td><td>2.47</td><td>2.694</td><td>一</td></tr><tr><td> $b$ </td><td>-1.79</td><td>-3.219</td><td></td></tr><tr><td> $A _ { \mathrm { a c } }$ </td><td>67.0</td><td>801</td><td>12.0</td></tr><tr><td> $c$ </td><td>1.15</td><td>1.206</td><td></td></tr><tr><td> $x _ { r 0 }$ </td><td> $6 . 5 4 \times 1 0 ^ { - 1 1 }$ </td><td> $2 . 5 2 \times 1 0 ^ { - 1 0 }$ </td><td>3.85</td></tr><tr><td rowspan="8">Seifert &amp; Beheng</td><td> $C _ { \mathrm { a u } }$ </td><td> $6 . 8 1 \times 1 0 ^ { 1 8 }$ </td><td> $1 . 2 1 \times 1 0 ^ { 1 8 }$ </td><td>0.18</td></tr><tr><td> $\alpha$ </td><td>600</td><td>25390</td><td>42.3</td></tr><tr><td> $\beta$ </td><td>0.68</td><td>0.933</td><td></td></tr><tr><td> $k _ { \mathrm { c r } }$ </td><td>5.78</td><td>3.814</td><td>0.66</td></tr><tr><td> $\tau _ { 0 }$ </td><td> $5 . 0 0 \times 1 0 ^ { - 4 }$ </td><td> $4 . 8 8 \times 1 0 ^ { - 5 }$ </td><td>0.098</td></tr><tr><td> $k _ { \mathrm { { r } } }$ </td><td> $5 . 7 8$ </td><td>3.857</td><td>0.67</td></tr><tr><td> $C _ { \mathrm { s c } }$ </td><td> $1 . 4 2 \times 1 0 ^ { 1 0 }$ </td><td> $7 . 0 0 \times 1 0 ^ { 9 }$ </td><td>0.50</td></tr><tr><td> $x ^ { * }$ </td><td> $2 . 6 0 \times 1 0 ^ { - 1 0 }$ </td><td> $3 . 2 1 \times 1 0 ^ { - 9 }$ </td><td>12.4</td></tr></table>

Table 5: Training loss and test loss scored against the McSnow reference data sets for the published and retuned parameters for the 4 reference schemes.
<table><tr><td></td><td colspan="2">Train loss</td><td colspan="2">Held-out test loss</td></tr><tr><td>Scheme</td><td>published</td><td>fitted</td><td>published</td><td>fitted</td></tr><tr><td>Kessler</td><td>3.127</td><td>0.800</td><td>3.988</td><td>0.776</td></tr><tr><td>Berry &amp; Reinhardt</td><td>1.737</td><td>0.733</td><td>2.301</td><td>0.739</td></tr><tr><td>Khairoutdinov &amp; Kogan</td><td>4.556</td><td>0.645</td><td>6.021</td><td>0.632</td></tr><tr><td>Seifert &amp; Beheng</td><td>1.096</td><td>0.283</td><td>1.464</td><td>0.272</td></tr></table>

## E Metrics used for evaluation of bulk parameterization performance

The paper uses three metrics to evaluate the different collision-coalescence parameterizations: relative $L _ { 2 }$ error, MAPE, and $t _ { 1 0 }$

• Relative $L _ { 2 }$ error: for each test trajectory and moment, $\| \hat { x } - x \| _ { 2 } / \| x \| _ { 2 }$ over the time series. The reported value is the median over test trajectories. Rain-mass error in Section 3 is the relative $L _ { 2 }$ error of $L _ { r }$

• MAPE (Figure 2b): for each trajectory, 100 × mean<sub>t</sub> $| \hat { x } _ { t } - x _ { t } | / | x _ { t } |$ , over time steps with $| x _ { t } | >$ $1 0 ^ { - 2 0 }$ in SI units; reported as the median over the 100 test trajectories.

• t : time at which 10% of the initial cloud mass has been converted to rain. We report the median absolute error over test trajectories, in minutes.

## F Expert-in-the-loop effort

Table 6 provides information about the number of prompts required for each of the 4 expert-in-theloop modeling tasks described in the main text, as well as the approximate amount of time that each model component required for active development. These were determined from Claude’s history (history.jsonl).

Table 6: Expert-in-the-loop effort for each agent-developed component, counted from the prompt history and corroborated by the git repo commit history. Substantive refer to prompts that direct work (excluding bookkeeping such as git commits), while corrective refers to prompts pointing out defects or mismatches. Active refers to total estimated time spent actively working on that component.
<table><tr><td>Component</td><td>Sessions</td><td>Prompts</td><td>Substantive</td><td>Corrective</td><td>Active (h)</td><td>Commits</td></tr><tr><td>Translate 4 literature schemes</td><td>1</td><td>5</td><td>4</td><td>2</td><td>0.5</td><td>3</td></tr><tr><td>Seifert &amp; Rasp (2020) MLPs</td><td>3</td><td>25</td><td>19</td><td>4</td><td>4.0</td><td>2</td></tr><tr><td>SuperdropNet (2025)</td><td>3</td><td>36</td><td>24</td><td>8</td><td>7.1</td><td>3</td></tr><tr><td>Physics-constrained Neural ODE</td><td>5</td><td>131</td><td>101</td><td>14</td><td>22.0</td><td>19</td></tr></table>

## G Physics-constrained Neural ODE closure

To explore whether agentic AI could come up with genuinely new data-driven parameterization approaches for collision-coalescence, we explored expert-in-the-loop development of a Neural ODEbased (Chen et al., 2019) closure for the evolution of the four two-moment state variables. Rather than proposing a generic black-box network, Claude drew on structural ideas from the historical bulk schemes to factor the learned tendencies into physically meaningful process rates. The architecture enforced mass conservation by construction and was designed so that its learned components could be associated with recognizable transfers between cloud and rain categories. These choices produced a more interpretable and physically constrained model than an unconstrained state-to-state neural network.

Developing the closure nevertheless required substantial expert intervention (Table 6), with 131 prompts over 22 active hours. Multiple iterations were needed to refine the approach, with design iterations focused on architecture refinement and training objective exploration dominating the exploration phase. Although the resulting NODE captured the qualitative evolution of the McSnow trajectories, it remained less accurate than several of the reconstructed classical and machine-learning parameterizations (Section 4). Given the extent of the expert intervention, we interpret this experiment as human–agent co-development rather than autonomous parameterization discovery. It shows that an agent can help explore architectures that combine learned components with known physical structure, but not that agentic development will necessarily produce a more accurate closure.

## H Reconstruction of machine-learning baselines

For both reconstructions, the coding agent was given the original publication and the McSnow training dataset and was guided through implementation, training, and comparison with the published results. Table 6 summarizes the expert involvement required for each reconstruction.

Seifert and Rasp (2020). The parameterization developed by Seifert and Rasp (2020) uses four separate multilayer perceptrons to predict rates of autoconversion, accretion, cloud self-collection, and rain self-collection. These rates define the right-hand side of the ODEs governing the four bulk moments, which are integrated using a forward Euler scheme. The agent implemented the networks and training procedure in JAX and reproduced the published evaluation framework. The reconstructed model closely reproduced both the single-step rate predictions and prognostic behavior of the original model. When initialized from the same conditions, its rollouts also reproduced the reported behavior relative to the McSnow simulations and the Seifert and Beheng parameterization (Figure 3). The reconstruction required three development sessions, 19 substantive prompts, four corrective prompts, and approximately 4.0 hours of active expert involvement (Table 6).

![](images/270ecc86fa9beab34d40eb9197a3ebd85a29246dfad9e113e1d1acb071feb6d1.jpg)  
Figure 3: Reconstruction of Figure 7 from Seifert and Rasp, 2020.

SuperdropNet. SuperdropNet (Sharma and Greenberg, 2025) predicts tendencies for the four bulk moments rather than separate conventional process rates and applies physical constraints to the resulting state updates. Its training procedure progressively increases the autoregressive rollout horizon from one to 25 time steps, with the model from each stage used to initialize the next.

The agent reconstructed the network architecture and overall loss function, but reproducing the published training performance proved significantly more difficult. The resulting model captured the qualitative behavior of SuperdropNet but did not achieve the same accuracy during autoregressive rollout (Figure 4). The main difficulty was reproducing the complete optimization procedure, including the learning-rate schedule, curriculum transitions, and stopping criteria. The reconstruction required three development sessions, 24 substantive prompts, 8 corrective prompts, and approximately 7.1 hours of active expert involvement (Table 6). These examples show that agentic AI can reliably reconstruct relatively simple ML parameterizations when the architecture and training objective are clearly specified. Reproducing more complex models with autoregressive or curriculum-based training remains considerably more dependent on expert intervention.

![](images/bea8a17649449d253328ee1a1d0fe5bd13f51aba4074b09b9822c2ea4075c60c.jpg)  
Figure 4: Reconstruction of Figure 10 from Sharma and Greenberg, 2025.

## I Single shot LLM calls and iterative refinement experiments

## I.1 Models used in experiments

Claude Opus 4.8, used for the expert-guided experiments, was not available through the American Science Cloud model gateway that was used for the majority of the single shot and iterative refinement experiments. Claude Opus 5.5, a later model in the same family, is used as the closest same-family comparison for the unattended experiments.

Cost of the unattended experiments. Here we estimate token costs for the single-shot and iterative experiments, priced at the provider’s public standard-tier rate.
<table><tr><td>Model</td><td>Translation (40 calls)</td><td>Single-shot (10 calls)</td><td>Iterative (10×4 rounds)</td><td>Iterative (5×12 rounds)</td><td>Total</td></tr><tr><td>Gemini 2.5 Pro</td><td>5.62†</td><td>1.77†</td><td>4.15†</td><td>一</td><td>11.54</td></tr><tr><td>Gemini 3.1 Pro (preview)</td><td>5.48</td><td>3.65</td><td>13.23</td><td>一</td><td>22.37</td></tr><tr><td>Gemini 4 Argon</td><td>1.95§</td><td>0.288</td><td>2.22§</td><td>1.758</td><td>6.21</td></tr><tr><td>Claude Haiku 5.5</td><td>0.66</td><td>0.08</td><td>0.15</td><td>一</td><td>0.89</td></tr><tr><td>Claude Sonnet 5.5</td><td>3.72</td><td>3.48</td><td>9.15</td><td>一</td><td>16.34</td></tr><tr><td>Claude Opus 5.5</td><td>6.14</td><td>5.55</td><td>10.16</td><td>6.50</td><td>28.34</td></tr><tr><td>GPT-5.6 Terra</td><td>2.60</td><td>0.83</td><td>1.90</td><td></td><td>5.33</td></tr><tr><td>GPT-6 Luna</td><td>0.14</td><td>0.04</td><td>0.10</td><td></td><td>0.29</td></tr><tr><td>GPT-6 Sol</td><td>1.84</td><td>1.03</td><td>2.01</td><td>一</td><td>4.88</td></tr><tr><td>GPT-6 Astra</td><td>8.11</td><td>4.86</td><td>7.96</td><td>一</td><td>20.92</td></tr><tr><td>Total</td><td>36.27</td><td>21.57</td><td>51.03</td><td>8.24</td><td>117.11</td></tr></table>

Table 7: Cost (USD) of the unattended experiments per model, at each provider’s public standard-tier API prices (October 2026). Thinking tokens are billed as output and cached input at the cache-read rate. Each cell is the mean cost per call times the number of calls in the column header (one set of experiments, without retries). Runs were made through the AmSC Model Access Gateway, so these are list-price equivalents. <sup>†</sup>Usage was not recorded for the original Gemini 2.5 Pro runs (called through the Gemini API); estimated from a re-measured sample of the same calls (4 translations, 2 single-shot, 1 iterative candidate). <sup>‡</sup>The 10 Berry & Reinhardt translations were made through the Gemini API and are estimated from the mean Gemini 3.1 Pro translation call. <sup>§</sup>Gemini 4 Argon was run at its default setting, without reasoning effort set, and used almost no thinking tokens; with thinking enabled its cost would be substantially higher.

## I.2 Summary of experiments with different models

Here we provide results for the single shot API calls and automated LLM closure generation with iterative feedback. We compare performance across 10 different large language models. Repeating

the unattended experiments across models separates the effect of model choice from that of the workflow, and reveals failure modes common to all models.

![](images/83130769c83156d555e15c835bf2ec311464bf01db34a55dff00c955a602cee5.jpg)  
Figure 5: Failure modes for single-shot translation of historical schemes by model.

![](images/e5687c8e30a66ab18fb83e4f59f7b88f8c4273f960f322a2541e10994353bf72.jpg)  
Figure 6: Accuracy of single-shot and iteratively refined agent-generated closures relative to the McSnow test data and historical parameterizations (shown as dashed lines scored against the McSnow test data). Here, the performance for the historical parameterizations assumes their published (rather than retuned) parameters.

![](images/5a4d60b48c8781f6316176a6e13148d178a33f53b361e3148407940b6fdad6c5.jpg)  
Figure 7: Accuracy of iteratively refined agent-generated closures relative to the McSnow test data and historical parameterizations for the 12 round iterative refinement rounds.

Best agent-generated closure vs Seifert & Beheng. The best closure (Claude Opus 5.5, iterative sequence 01) is a modified Seifert & Beheng scheme rather than a new one. A comparison between the schemes is shown in Table 8.
<table><tr><td>Process</td><td>Seifert &amp; Beheng (2001)</td><td>Claude Opus 5.5 closure</td><td>Relation</td></tr><tr><td>Autoconversion, moment dependence</td><td> $C _ { a u } L _ { c } ^ { 2 } x _ { c } ^ { 2 }$ </td><td> $K _ { a u } L _ { c } ^ { 2 } x _ { c } ^ { 2 }$ </td><td>Same form</td></tr><tr><td>Autoconversion, rain-fraction</td><td> $1 + \Phi _ { a u } ( \tau ) / ( 1 - \tau ) ^ { 2 } ,$   $\Phi _ { a u } = 6 0 0 \tau ^ { 0 . 6 8 } ( 1 - \tau ^ { 0 . 6 8 } ) ^ { 3 }$ </td><td> $\epsilon + \tau / [ 1 + ( \tau / \tau _ { s } ) ^ { 0 . 6 } ] ,$   $\tau _ { s } = 7 . 3 \times 1 0 ^ { - 5 }$ </td><td>Different</td></tr><tr><td>correction Accretion, kernel Accretion, correction</td><td> $K _ { c r } L _ { c } L _ { r } , K _ { c r } = 5 . 7 8$   $[ \tau / ( \tau + 5 \times 1 0 ^ { - 4 } ) ] ^ { 4 }$  (rain fraction)</td><td> $K _ { a c } L _ { c } L _ { r } , K _ { a c } = 5 . 5$   $( x _ { c } / 1 . 3 \times 1 0 ^ { - 1 1 } \mathrm { k g } ) ^ { 0 . 2 5 }$  (cloud-drop</td><td>Same form Different</td></tr><tr><td>Cloud number, autoconversion + self-collection</td><td> $\begin{array} { r } { \frac { \nu + 2 } { \nu + 1 } K _ { c c } L _ { c } ^ { 2 } , K _ { c c } = 9 . 4 4 \times 1 0 ^ { 9 } , \nu = } \\ { 1 } \end{array}$ </td><td>mass)  $K _ { c c } L _ { c } ^ { 2 } + P _ { a u } / x _ { c }$ </td><td>Different split</td></tr><tr><td>Rain self-collection</td><td> $- K _ { r r } N _ { r } L _ { r } , K _ { r r } = 5 . 7 8$ </td><td> $- 5 . 7 8 N _ { r } L _ { r } { \big / } \quad [ 1 \quad + \quad ( x _ { r } / 2 . 6 \times$   $1 0 ^ { - 6 } \mathrm { k g } ) ^ { 0 . 6 8 } ]$ </td><td>Same form, large drops</td></tr><tr><td>Mass of a new rain drop</td><td> $x ^ { * } = 2 . 6 \times 1 0 ^ { - 1 0 } { \mathrm { k g } }$ </td><td> $5 \times 1 0 ^ { - 1 0 } \mathrm { k g }$ </td><td>limited Same role</td></tr><tr><td>Cloud number lost by accretion</td><td> $C / x _ { c }$ </td><td> $0 . 8 C / x _ { c }$ </td><td>Rescaled</td></tr><tr><td> $L _ { r }$  relative L2 error</td><td></td><td></td><td></td></tr><tr><td> $t _ { 1 0 }$  error [min]</td><td>0.26 published, 0.18 retuned 7.7 published, 6.0 retuned</td><td>0.16</td><td></td></tr><tr><td> $L _ { r }$  rollout MAPE [%]</td><td></td><td>5.5</td><td></td></tr><tr><td></td><td>191 published, 47 retuned</td><td>48</td><td></td></tr><tr><td>FLOPs per step</td><td>66</td><td>67</td><td></td></tr></table>

Table 8: Process formulations of Seifert & Beheng (2001) and of the most accurate agent-generated closure (Claude Opus 5.5, iterative), and their errors on the 100 held-out McSnow test simulations. $\tau = L _ { r } / ( L _ { c } + L _ { r } )$ is the rain fraction and $x _ { c } , x _ { r }$ the mean cloud- and rain-drop masses.