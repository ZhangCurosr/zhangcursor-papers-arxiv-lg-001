# A Unifying Perspective on Probabilities as Model Predictions

Benedikt H¨oltgen

Hasso Plattner Institute, University of Potsdam

## Abstract

Although probabilistic statements are ubiquitous, foundational disagreements persist about their understanding, as exemplified by debates between Bayesians and frequentists; moreover, it is unclear when and why acting on them actually leads to desirable outcomes. Here, we argue that every probability is the output of a prediction method, that is, it depends on both a particular way of constructing abstractions and a way of transforming them into predictions. Through this, we provide a unifying perspective on supposedly diferent kinds of probabilities and show that even supposedly objective ones are model-dependent. We demonstrate that when a finite calibration criterion is met, one can anticipate the distribution of utilities for a given policy and inform successful decision-making on finite sets of events. Based on the notion of prediction methods, inductive arguments, and the probability calculus, we explain the feasibility of the calibration criterion in many settings. Overall, we develop a coherent perspective on probabilities and their use, connecting key intuitions behind other interpretations along the way.

## 1 Introduction

Probabilities are as various as the faces to be seen at will in fretwork or paperhangings

— George Eliot (1871): Middlemarch

We make probabilistic statements and use probabilistic reasoning all the time: If the predicted ‘probability of rain’ is suficiently high, you bring your umbrella or even stay at home. You decide to undergo surgery if this is thought to significantly ‘raise your chances’ of recovery. Given that such probabilistic statements permeate both science and our everyday lives, it is quite remarkable that it is still an open question what exactly we mean by them and how they are useful: Do they refer to degrees of belief, to relative frequencies of repeated trials, or to physical properties? The meaning of probability is considered to ‘bear at least indirectly, and sometimes directly, upon central scientific, social scientific, and philosophical concerns’ (H´ajek, 2023). In machine learning, the meaning of probability is increasingly recognised as a ‘pressing question’ (Burhanpurkar et al., 2021); as put by Cynthia Dwork, ‘without an answer to this definitional question, we don’t even know what it is that the ideal algorithm should satisfy’ (Dwork, 2022).

A common view in both statistics and philosophy is that there are two kinds of probabilities, which one may refer to as aleatory and epistemic, respectively (Hacking, 1975, p. 13–15). Aleatory probabilities are grounded in the world, and potentially objective, for example in gambling or in other observable frequencies. Epistemic probabilities are more speculative, linked to uncertain predictions and credences.<sup>1</sup> In philosophy, these concepts are often linked through the ‘Principal Principle’, stating roughly that once a chance is learned, it should be adopted as credence. Similarly, it is often assumed ‘that the job of statistics is to identify the data-generating process’ (Vovk and Shafer, 2025, p. 160), or that in machine learning, ‘we would like to match the true data-generating distribution (Goodfellow, Bengio, and Courville, 2016, p. 130).

In contrast to these views, we develop a perspective that unifies these supposedly diferent kinds of probabilities, arguing that all probabilities are constructed and model-dependent. In this work, we lay out a descriptive account of probability as predictions that are useful when they are finitely calibrated on certain sets. We do not argue for a specific normative position about which predictions should be allowed to be called probabilities; however, our pragmatic perspective can shed new light on common notions of rational belief (abiding by the probability calculus) and decision-making (maximising expected utility).

The paper is structured as follows. In Section 2, we introduce the notion of prediction methods and show that they cover not only obvious examples like rain forecasts but also supposedly objective probabilities such as relative frequencies and gambling odds. In Section 3, we demonstrate how predictions that are finitely calibrated on relevant sets help us to make actually good decisions. In Section 4, we elucidate why calibration is often feasible by drawing connections to the problem of induction and the probability calculus. Lastly, we turn to to the literature on interpretations of probability and argue that our account satisfies general desiderata (Section 5) and captures key intuitions behind other interpretations (Section 6).

## 2 Behind Every Probability is a Prediction Method

At the heart of our perspective on probability is the insight that each probability comes from a prediction method. We first introduce the notion of prediction methods with an intuitive case and then walk through two perhaps less intuitive, seemingly aleatory examples.

## 2.1 Predictors and prediction methods

All probability assignments involve predictions based on abstractions; the overall process of arriving at such a prediction we call a prediction method. We distinguish the predictor— the model that can be seen as a (mathematical) function—from the prediction method that is applied to an actual situation. The latter involves the construction of an abstraction (potentially including measurements) before applying the former (Figure 1). In the case of rain forecasts, the prediction method consists in first taking measurements (of temperatures, air pressure, etc.) and then feeding them to a computer model (the predictor) that outputs a prediction for the occurrence of rain.

Definition 1 (Predictor).   
A predictor is a function $\mathsf { p } : \mathcal { X } \times \mathcal { A }  \mathbb { R }$ on some set X and algebra<sup>2</sup> A.

$$
{ \begin{array} { r l } & { { \mathrm { s e l e c t ~ p r e d i c t o r ~ } } \mathsf { p } : { \mathcal { X } } \times { \mathcal { A } } \to \mathbb { R } } \\ & { { \mathrm { c o n s t r u c t ~ a b s t r a c t i o n ~ } } x \in { \mathcal { X } } } \end{array} } \quad \left\} { \begin{array} { r l } & { { \mathrm { c o m p u t e ~ } } \mathsf { p } ( x , A ) } \\ & { { \mathrm { c o m p u t e ~ } } \mathsf { p } ( x , A ) } \end{array} } \right.
$$

Figure 1: A prediction method gives a prediction for an event A in some situation by selecting a predictor $\mathsf { p } : \mathcal { X } \times \mathcal { A }  \mathbb { R }$ with $A \in { \mathcal { A } }$ , constructing an abstraction $x \in \mathcal { X }$ of the situation, and computing ${ \mathsf p } ( x , A )$ . For example, p can be a computer model that predicts the event ‘rain’ based on measurements x.

## Definition 2 (Prediction method).

A prediction method for an event $A \in { \mathcal { A } }$ is an implicit or explicit scheme for selecting a predictor p : $\mathcal { X } \times \mathcal { A }  \mathbb { R }$ and constructing an abstraction $x \in \mathcal { X }$ of the given situation.

The predictor takes two arguments: the abstraction on which to base the prediction, and the event to predict. In many settings, one of the two arguments is efectively ignored. For rain forecasts, the algebra of events $\{ \emptyset , \{ r a i n \} , \{ \neg r a i n \} , \{ r a i n , \neg r a i n \} \}$ is only implicit. By specifying a prediction $p _ { i }$ for $\{ r a i n \}$ , we intuitively assign predictions $0 , ( 1 - p _ { i } )$ , and 1 to the other events, respectively; we will turn to this in Section 4.2. Note that the specification of events $A \in { \mathcal { A } }$ also involves choices of definition or measurement. For example, how much rain counts as ‘no rain’ or in which area it is recorded is more or less implicit—and depends on value judgements (Douglas, 2000). These choices can, however, be seen as external to the choice of prediction method (although the availability of methods can influence the choice of target), so we do not discuss them further. Importantly, our notion of prediction does not require that the predicted events lie in the future, it sufices that the observations/labels are not available to the predictor.

In the rain forecasting example, the abstraction made by the prediction method consists in taking specific measurements of temperature, air pressure, et cetera. More generally, any sort of prediction requires focusing on a subset of all the information that could be taken into account—an abstraction of the situation. Any given situation has an enormous amount of potentially relevant information; typically, we decide what to look at based on experience as well as common sense or expert knowledge. For rain forecasts, temperature, and air pressure are more interesting quantities than the current GDP. Diferent models for rain prediction (the predictors) can also be based on diferent ways of measuring temperature—for example, diferent granularity, location, and timing of the measurements. Diferent models can also work very diferently—they may rely on simple look-up tables or sophisticated simulations. They may even rely on human forecasters who also only use limited information for their forecasts. While it is dificult to speak of human predictors as stable mathematical objects, they can arguably be approximated as such.

## 2.2 Example: Symmetry-based predictions

In many settings, probabilities are intuitively not thought to depend on modelling choices or a particular prediction method; this includes probabilities for gambling devices. Assume you go to a casino where they ofer a novel game based on a symmetrical 8-sided and a symmetrical 20-sided ‘die’ (an octahedron and an icosahedron). Given that it is an oficial casino, you assume that the dice are indeed symmetrical. How do you make predictions? You can represent any possible outcome that you wish to predict as the set of admissible combinations of faces, which can e.g. be represented as the algebra $\mathcal { A } = 2 ^ { \{ 1 , . . . , 8 \} \times \{ 1 , . . . , 2 0 \} }$

For the prediction, you presumably ignore the name of the croupier, the surface of the table, and so on, and only focus on the symmetry of the dice. Each face of the octahedron corresponds to a prediction of $\frac { 1 } { 8 }$ and each face of the icosahedron corresponds to a prediction of ${ \frac { 1 } { 2 0 } } .$ . How exactly this reasoning is captured by a predictor ${ \mathsf { p } } : { \mathcal { X } } \times { \mathcal { A } } \to { \mathbb { R } }$ is underdetermined; in particular, what counts as an argument versus as a part or parameter of the predictor: You may consider a predictor that only takes symmetrical dice, such that the input space ${ \mathcal { X } } = \{ x \}$ can be ignored. Alternatively, you may take $\mathcal { X } = \Delta _ { 8 } \times \Delta _ { 2 0 }$ to be the set of all possible combinations of potentially biased 8-sided and 20-sided dice, of which you consider the element $x : = ( \textstyle { \frac { 1 } { 8 } } , . . . , \textstyle { \frac { 1 } { 8 } } , \frac { 1 } { 2 0 } , . . . , \frac { 1 } { 2 0 } ) \in \mathcal { X }$ , reflecting your assumption of fairness. You may also take a still larger X that can also capture dice with other numbers of faces. In any case, you proceed by transforming your abstraction of fair dice into a number in [0, 1] through a combinatorial model p. You construct an abstraction and you calculate.

Such symmetry-based prediction methods in gambling situations typically assume that the device is ‘fair’. Indeed, gambling devices are produced in such a way that each outcome should occur equally often, which allows us to make roughly accurate predictions about how often events occur on large samples.<sup>3</sup> We know from experience that the process of rolling a fair die is so opaque and chaotic that it is practically impossible for us to predict better than uniformly. If we were more proficient in discerning minuscule variations in die throws and background conditions (as Laplace’s demon would be), we might be able to make finer predictions. Indeed, Edward Thorp and Claude Shannon developed a device to better predict roulette outcomes which they successfully deployed in casinos in the 1960s (Thorp, 1998).After all, the assumption of a fair die or roulette wheel is a particular abstraction of (your beliefs about) the situation.

## 2.3 Example: Frequency-based predictions

Predictions that are explicitly based on looking up relative frequencies of similar events use a very simple type of prediction method. Such a method could be used for gambling instead of (or combined with) symmetry-based considerations. There are also more interesting examples, such as medical risks. Doctors typically base risk predictions on past experience, sometimes by explicitly looking up data about similar people. In an example recently discussed in (P. Dawid, 2017), which we shall return to later, Angelina Jolie got told that she had an 87% risk of breast cancer—with the number presumably coming from statistical data about women with a particular genetic mutation. Hence, the doctors used the following prediction method: They chose a particular abstraction of Angelina, as a woman with this gene mutation, and then used a predictor that is basically a look-up table. Presumably, women without that gene mutation get assigned into diferent categories (or ‘reference classes’) for which there is enough data for doctors to believe that the relative frequency in this category is stable over time. Diferent doctors may use diferent categories, that is, diferent abstractions. As in the rain example, we can consider this as a case where the implicit 4-element algebra is ignored. Alternatively, we could see it as an application of a more general predictor that can output predictions for diferent diseases, based on multiple look-up tables. This would make A more complex by adding more diseases as fundamental events and means that X needs to be fine enough that all relevant categories for all diseases

## 3 Finite Calibration Makes Probabilities Useful

The previous section argued that probabilities are outputs of prediction methods and thus constructed; this raises the question why we construct them and how they work. Although it is often assumed that probabilistic predictions are useful for decision-making, it has not been demonstrated in general terms how or under which conditions this is the case. In this section, we show how finite sets of predictions are useful to us if they satisfy a form of calibration. Although the technical details of the general perspective are almost trivial, it seems to not have been discussed before, let alone its relevance appreciated. From this general understanding, the idea of expected utility maximisation and the more common understanding of calibration emerge as special cases. For the purpose of this section, it is enough to think of predictions as arbitrary real numbers $p _ { i } \in \mathbb { R } . ^ { 5 } \mathrm { A s }$ before, a prediction $p _ { i }$ relates to an event $A _ { i }$ and its label $y _ { i } \in \{ 0 , 1 \}$ where $y _ { i } = 1$ or $y _ { i } = 0$ denotes that the event does or does not occur, respectively.

## 3.1 Predicting numbers of events

What is the diference between a prediction of 0.6 and a prediction of 0.9, given that the predicted event either does or does not occur? An important diference surfaces when considering multiple predictions: In general, of 100 events with prediction 0.6, we intuitively expect roughly 60 to occur, whereas of 100 events with prediction 0.9, we would expect roughly 90 to occur. We can formalise this as a quality criterion for predictions called calibration: For a given set of events, the sum of our predictions should coincide with the number of occurring events.

## Definition 3 (Calibration).

Predictions $p _ { 1 } , . . . , p _ { d } \in \mathbb { R }$ are said to be calibrated for observations $y _ { 1 } , . . . , y _ { d } \in \{ 0 , 1 \}$ if they satisfy

$$
\sum _ { i = 1 } ^ { d } p _ { i } = \sum _ { i = 1 } ^ { d } y _ { i } .\tag{1}
$$

Often, calibration is understood more narrowly as what P. Dawid (2017) calls ‘probability calibration’, namely calibration on sets of equal prediction. While this unnecessarily narrow understanding of calibration may be partly due to historical reasons, as discussed in (H¨oltgen and Robert C Williamson, 2023), it is also particularly relevant in many settings (Section 3.3). Our definition is similar to the definition of calibration of A. P. Dawid (1985), except that there, it is restricted to infinite sub-sequences of infinite sequences of outcomes whose averages are assumed to converge (which essentially presupposes the existence of objective, frequentist probabilities). We argue that predicting how many events of certain sets will occur, i.e. calibration in the general sense, is the purpose for having probabilities in the first place.

The intuition for this is simple: If I knew that my predictions are calibrated on a set of events, then the sum of the predictions tells me how many of the set of events will occur.

This is very helpful, for example, in gambling settings: if I can reliably predict how often a repeatable event with given payof will occur, then I know how much I should bet in each instance to come out positively in the end. An important aspect here is that I do not care which of the events will occur, because the payout is always the same. This is what calibration delivers: It tells you how many events will occur, without telling you which ones.

An implication of this is that calibration is only useful if I care equally about each event. This notion of ‘caring equally’ is captured in formal models by the condition that all events give me the same utility. Intuitively, a person’s utility is a numerical representation of how much the person values the (non-)occurrence of an event (which is clearly an idealisation). In our setting of binary events, we will use utility functions $u _ { i } : \{ 0 , 1 \} \mapsto \mathbb { R }$ where $u _ { i } ( 0 )$ and $u _ { i } ( 1 )$ capture how much I value $y _ { i } = 0$ and $y _ { i } = 1$ , respectively. Now, calibration can help us to foresee the (non-normalised) distribution of utilities that I will receive: If my predictions are calibrated on sets of equal utility, then I can predict the number of occurring events for each utility by summing up the relevant predictions.

## 3.2 Predicting cumulative utility

While we will come back to general utility distributions in Section 3.4, we will for now focus on a particularly intuitive property of that distribution, which we call cumulative utility.

Definition 4 (Cumulative utility).

My cumulative utility over a set of outcomes $\{ y _ { 1 } , . . . , y _ { d } \}$ given utility functions $u _ { 1 } , . . . , u _ { d }$

$$
\sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] .\tag{2}
$$

As $y _ { i }$ denotes whether event 1 or 0 occurs, the cumulative utility is the sum over all utilities that I actually receive. This captures how well I will be of overall. We now formally show that with suitably calibrated predictions, it is possible to estimate cumulative utility through a very familiar quantity. The simple idea is that if for each utility value, I correctly predict how many events with this utility will occur, then I will correctly predict my cumulative utility. Note that for our purposes, it would be mathematically equivalent to consider the average utility I get at each time step, i.e. the cumulative utility divided by d.

Proposition 5 (Predicting cumulative utility).

Let there be d predictions $p _ { i } \in \mathbb { R }$ for binary outcomes $y _ { i } \in \{ 0 , 1 \} , i \in \{ 1 , . . . , d \}$ with utility functions $u _ { i } : \{ 0 , 1 \} \to \mathbb { R }$ and assume that the predictions are calibrated on sets of equal utility $u _ { i } ( 0 )$ and on sets of equal utility $u _ { i } ( 1 )$ , formalised by the assumption that $\forall u ^ { \prime } \in \mathcal { U }$ :

$$
\sum _ { i : u _ { i } ( 1 ) = u ^ { \prime } } y _ { i } = \sum _ { i : u _ { i } ( 1 ) = u ^ { \prime } } p _ { i } \qquad a n d \qquad \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } y _ { i } = \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } p _ { i } ,\tag{3}
$$

where U denotes the set of all values that the $u _ { i }$ can take, i.e. $\begin{array} { r } { \mathcal { U } : = \bigcup _ { 1 \leq i \leq d } \{ u _ { i } ( 0 ) , u _ { i } ( 1 ) \} \subset \dag } \end{array}$   
R.

Then I can correctly predict my cumulative utility (LHS) via

$$
\sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] = \sum _ { i = 1 } ^ { d } \left[ p _ { i } u _ { i } ( 1 ) + ( 1 - p _ { i } ) u _ { i } ( 0 ) \right] .\tag{4}
$$

Proof.

$$
\sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] = \sum _ { u ^ { \prime } \in \mathcal { U } } \left( \sum _ { i : u _ { i } ( 1 ) = u ^ { \prime } } y _ { i } + \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } ( 1 - y _ { i } ) \right) \cdot u ^ { \prime }\tag{5}
$$

$$
= \sum _ { u ^ { \prime } \in \mathcal { U } } \left( \sum _ { i : u _ { i } \left( 1 \right) = u ^ { \prime } } p _ { i } + \sum _ { i : u _ { i } \left( 0 \right) = u ^ { \prime } } \left( 1 - p _ { i } \right) \right) \cdot u ^ { \prime }\tag{6}
$$

$$
= \sum _ { i = 1 } ^ { d } \left[ p _ { i } u _ { i } ( 1 ) + ( 1 - p _ { i } ) u _ { i } ( 0 ) \right]\tag{7}
$$

where for $( 5 ) = ( 6 )$ , we use the calibration criterion (3).

Given that the assumption of exact calibration on all sets of equal utility is very strong, we also show that approximate calibration (Appendix $\mathrm { A . 1 } )$ and calibration on sets of ap proximately equal utility (Appendix A.2) sufice for approximately correct predictions of the cumulative utility. Furthermore, a weaker calibration criterion for imprecise predictions makes it possible to incorporate risk aversion (Appendix $\mathrm { A . 3 } )$ . Note that the RHS of (4) is the sum over my expected utilities (my expected cumulative utility), in the conventional probabilistic framework where $\hat { \mathsf { Y } } _ { i }$ is the random variable with distribution $P _ { i }$ that takes value 1 rather than 0 with probability $p _ { i } \colon$

$$
\sum _ { i = 1 } ^ { d } \left[ p _ { i } u _ { i } ( 1 ) + ( 1 - p _ { i } ) u _ { i } ( 0 ) \right] = \sum _ { i = 1 } ^ { d } \mathbb { E } _ { P _ { i } } [ u _ { i } ( \hat { \mathsf { Y } } _ { i } ) ] .\tag{8}
$$

This means that the policy of expected utility maximisation (EUM) is the policy that actually maximises my cumulative utility if my predictions are calibrated! To capture the idea of maximising utility, we need a notion of decisions between acts, which is not yet part of the setup. For simplicity, we consider d binary decisions, at step i consisting in a choice between $( p _ { i } ^ { a } , u _ { i } ^ { a } )$ and $( p _ { i } ^ { b } , u _ { i } ^ { b } )$ .

Corollary 6 (Comparing policies by expected utility).

For $i \in \{ 1 , . . . , d \}$ , let there be predictions $p _ { i } ^ { a } , p _ { i } ^ { b } \in \mathbb { R }$ for binary outcome $y _ { i } \in \{ 0 , 1 \}$ and utility functions $u _ { i } ^ { b } , u _ { i } ^ { b } : \{ 0 , 1 \} \to \mathbb { R }$ . For $\pi \in \{ a , b \}$ , policy π is then given by predictions $p _ { 1 } ^ { \pi } , . . . , p _ { d } ^ { \pi }$ and utility functions $u _ { 1 } ^ { \pi } , . . . , u _ { d } ^ { \pi }$ . Assume that the predictions of both policies are calibrated on sets of equal utility in the sense of (3) for their respective utility functions.

Then, for $\hat { \mathsf { Y } } _ { i } ^ { \pi }$ and $P _ { i } ^ { \pi }$ as in $( \delta )$ , the policy with the higher expected cumulative utility $\begin{array} { r l } {  { \sum _ { i = 1 } ^ { d } \mathbb { E } _ { P _ { i } ^ { \pi } } [ u _ { i } ^ { \pi } ( \hat { \mathsf { Y } } _ { i } ^ { \pi } ) ] } \quad } & { { } } \end{array}$ will actually provide the higher cumulative utility.

While EUM can be seen as a descriptive theory of human decision-making, it is often also assumed as a normative principle (e.g. by Hedden (2013)). We can now give conditions under which this is a good policy in the sense that it leads to desired outcomes: when we are calibrated on sets of equal utility and when we care about maximising cumulative utility. We now discuss a specific case of utility settings that has received much attention in statistics and machine learning, before we widen the scope again and consider cases where we are not interested in cumulative utility.

## 3.3 Probability calibration

We now illustrate the above with an example. Let there be d days where for $i \in \{ 1 , . . . , d \}$ $p _ { i } \in \{ 0 , 0 . 1 , 0 . 2 , . . . , 1 \}$ is the daily rain forecast and $y _ { i } \in \{ 0 , 1 \}$ denotes whether it actually rains $( y _ { i } = 1$ denoting rain). Now assume that across days, my attitude towards rain does not change but that it depends on whether I brought an umbrella: Let my utilities be given by $u ^ { a } ( 1 ) = 0$ and $u ^ { a } ( 0 ) = - 1$ if I brought an umbrella and $u ^ { b } ( 1 ) = - 3$ and $u ^ { b } ( 0 ) = 0$ if I did not bring one. Then my predicted utility when bringing an umbrella on day i is

$$
p _ { i } \cdot u ^ { a } ( 1 ) + ( 1 - p _ { i } ) \cdot u ^ { a } ( 0 ) = ( 1 - p _ { i } ) \cdot ( - 1 ) = p _ { i } - 1\tag{9}
$$

whereas my predicted utility when not bringing an umbrella is

$$
p _ { i } \cdot u ^ { b } ( 1 ) + ( 1 - p _ { i } ) \cdot u ^ { b } ( 0 ) = - 3 p _ { i } .\tag{10}
$$

As $p _ { i } - 1 > - 3 p _ { i } \Leftrightarrow p _ { i } > 0 . 2 5$ , I maximise predicted utility (per day) if I bring an umbrella on days where $p _ { i } > 0 . 2 5$ . This policy induces the utility functions

$$
u _ { i } = \left\{ \begin{array} { l l } { u ^ { a } } & { \mathrm { ~ i f ~ } p _ { i } > 0 . 2 5 } \\ { u ^ { b } } & { \mathrm { ~ i f ~ } p _ { i } < 0 . 2 5 . } \end{array} \right.\tag{11}
$$

The calibration criterion (3) in Proposition 5 for this case amounts to

$$
\sum _ { i : p _ { i } < 0 . 2 5 } y _ { i } = \sum _ { i : p _ { i } < 0 . 2 5 } p _ { i } \qquad \mathrm { a n d } \qquad \sum _ { i : p _ { i } > 0 . 2 5 } y _ { i } = \sum _ { i : p _ { i } > 0 . 2 5 } p _ { i } .\tag{12}
$$

If this condition is satisfied, my cumulative utility will coincide with the sum of my daily predicted utilities.

To assess the relative merits of this particular policy, we need to compare it with other policies. (12) is one instance of a prediction-dependent threshold policy, where I bring an umbrella whenever the predicted probability of rain is higher than a certain threshold (in this case, 0.25). For Proposition 5 to apply to such a policy with any threshold $t \in [ 0 , 1 ]$ the calibration criterion is

$$
\sum _ { i : p _ { i } < t } y _ { i } = \sum _ { i : p _ { i } < t } p _ { i } \qquad { \mathrm { a n d } } \qquad \sum _ { i : p _ { i } > t } y _ { i } = \sum _ { i : p _ { i } > t } p _ { i } .\tag{13}
$$

Now note that since the possible utility functions are assumed to be the same each day, for this condition to be satisfied for all $t \in [ 0 , 1 ]$ , it is enough to satisfy

$$
\forall v \in \{ 0 , 0 . 1 , 0 . 2 , . . . , 1 \} : \sum _ { i : p _ { i } = v } y _ { i } = \sum _ { i : p _ { i } = v } p _ { i } .\tag{14}
$$

Now this is just the common condition of calibration on sets of equal prediction, which is commonly expected of rain forecasters (Gigerenzer et al., 2005) and which ‘even inexperienced forecasters are capable of displaying’, except for extreme predictions (Sanders, 1963, p. 191); see also Murphy and Winkler (1977). Under this fairly benign assumption, choosing 0.25 as my threshold maximises not only my predicted utility but also my actual cumulative utility among all threshold-based policies due to Proposition 5! Hence, people can tailor their policies to their personal utilities and, thus, their decisions to the forecasts. This also demonstrates why probabilistic forecasts are useful even in a deterministic world without ‘real’ probabilities (cf. Section 5.1). We would like to highlight that calibration on sets of equal prediction thus derives its importance (and prevalence) from their concurrence with the sets of equal utility when considering threshold-based policies.

This observation is closely linked to the connection between probability calibration and swap-regret that was shown for the first time by Foster and Vohra (1998). In this work, a randomised forecasting algorithm which minimises the maximum regret under a permutation of predictions (the maximum swap regret) was used to achieve low calibration error; this was used to show that one can always achieve probability-calibrated forecasts, at least in probability. While the importance of the reverse direction has since been appreciated (Noarov and Roth, 2024), the benefit of calibration is commonly only framed in terms of regret; in our view, the connection to swap regret should be seen as a corollary of the more fundamental function of probability per se, the accurate prediction of numbers of occurring events. Closest to ours is the perspective of Zhao et al. (2021) which focusses on the predictability of average loss, but is restricted to loss functions which depend only on prediction and outcome, not on the more general utility of the event.

Note that the calibration criterion was fairly benign in our rain example because the utilities are the same every day and we restricted our comparisons to the 10 thresholdbased policies (arguably the only sensible policies here). The story would be more complex if we took the utility to also depend e.g. on wind speed (because it afects the eficacy of umbrellas) or on the day of the week. In general, for d binary decisions, there are $2 ^ { d }$ possible combinations of decisions, i.e. policies! Accordingly, if we wanted to compare the cumulative utility of all possible choices via Proposition 5, this would lead to a very strong calibration criterion—in fact, it would require perfect binary predictions $p _ { i } = y _ { i }$ . This should not be surprising, as the best combination of decisions would be to always bring an umbrella if and only if it rains, which we can only ensure if we can discriminate perfectly between rainy and dry days.

It is, therefore, important to emphasise that calibration on sets of equal prediction should not be the only quality criterion for predictions. Consider the rain example above: While the constant base rate predictor also satisfies the calibration criterion (14), more refined predictions would allow people to better tailor their decisions to their utilities. Another property of interest is, thus, what is sometimes called sharpness or refinement, relating to the information content of a predictor (DeGroot and Fienberg, 1983). The Brier score, to take a criterion important in rain forecasting, can be decomposed into two terms measuring probability calibration and sharpness, respectively (Sanders, 1963). These two properties are in tension in the sense that it is more dificult to be calibrated on more informative predictions. While the focus of this work is how predictions can be useful in general, rather than their evaluation, we presently also mention some connections to the latter.

## 3.4 Calibration revisited

We now briefly consider alternatives to the cumulative utility as the quantity of interest. Recall from Section 3.1 that predictions calibrated on sets of equal utility not only allow us to predict the cumulative (or average) utility but the distribution of utilities more generally. Let $\mathcal { D } : = \{ \mu : \mathbb { R } \to \mathbb { N } _ { \geq 0 } \}$ denote the space of utility distributions, i.e. functions indicating how often diferent utility values occur.<sup>6</sup> Let $\mathcal { U } _ { \mu } : = \{ u \in \mathbb { R } : \mu ( u ) > 0 \}$ denote the support of a utility distribution $\mu \in { \mathcal { D } } ,$ i.e. the set of utility values that do occur in the distribution µ. With this notation, we can describe the cumulative utility of a distribution $\mu \in \mathcal { D }$ as $\textstyle \sum _ { u \in { \mathcal { U } } _ { u } } u \cdot \mu ( u )$ . Another potentially relevant property of utility distributions is the smallest received utility, min $\mathcal { U } _ { \mu }$ . For the umbrella policies, optimising for this property would mean that we should always bring an umbrella when $p _ { i } > 0$ , as we will otherwise incur a utility of −3 at some point—assuming that the calibration condition (14) holds. The minimum is quite an extreme property of a distribution as it ignores most information about the distribution (in our case, all events except those with the lowest utility). Optimising other properties of utility distributions will lead to other decision criteria but these questions are not the focus of this paper, interesting as they are.

In concurrent work, Perdomo and Recht (2025) claim that ‘[t]he utility of calibration comes in terms of communication’ (p. 15) and ‘emphasize that beyond [the] property of shared interpretation, calibration doesn’t mean much’ (p. 18). Against this view, we highlight three interrelated perspectives on the importance of calibration. First, the equivalence of swap regret and calibration highlights that the ‘best-response action’ which maximises EU will fare better than the reverse policy. In the rain case, even just matching the base rate would already allow to always take the action compatible with the more probable event— which is not impressive but better than the opposite. Arguably, this is only one part of a second, broader perspective on the choice of predictor given some policy and utility/loss: Calibrated predictors guarantee better decision outcomes than miscalibrated predictors with the same discriminative power. This perspective is explored further in many works connecting calibration to loss minimisation (Kleinberg et al., 2023; Gopalan, Kim, and Reingold, 2023; Feng and Tang, 2025; Derr, Finocchiaro, and Robert C Williamson, 2025). The third perspective is closer to our basic idea of predicting utility distributions: If we assume calibration on relevant sets, we can choose among policies based on the utility distributions they will, respectively, lead to. H¨oltgen and Robert C. Williamson (2025) demonstrate how this perspective can be fruitfully applied to causal inference settings. These perspectives are diferent sides of the same gambling device; combining the latter two, one may suggest choosing policy and predictor together, based on the promised utility distribution and on how realistic it is that the required calibration conditions are met. This highlights a fundamental question: How can we say anything about whether a set of predictions will be calibrated?

## 4 How Is Calibration Possible?

Even if we showed calibration can make predictions useful, this only helps with understanding probability in the case that is also realistic to achieve calibration. For arbitrary sets of predictions, there need not be a reason to assume that such a criterion would be satisfied. This points to the importance of considering the methods that generated the predictions, which we discussed in Section 2. Based on the notion of calibration for predictions $p _ { 1 } , . . . , p _ { d }$ as defined above, we can define calibration for predictors and prediction methods. For this, we consider the predictions of d events $A _ { 1 } , . . . , A _ { d } \in { \mathcal { A } }$ , with a prediction method that uses abstractions $x _ { 1 } , . . . , x _ { d } \in \mathcal { X }$

Definition 7 (Calibration of predictors and prediction methods).

A predictor (or prediction method) is said to be calibrated on events $A _ { 1 } , . . . , A _ { d } \in { \mathcal { A } }$ for observations $y _ { 1 } , . . . , y _ { d } \ i f$ its predictions $p _ { 1 } , . . . , p _ { d }$ are calibrated.

Prediction methods provide a first way to draw connections between individual predictions, which is necessary to even start talking about satisfying criteria on sets of events—but how can we hope for calibration on unseen sets of events? A first idea of providing calibrated predictors may be to directly optimise for it, in what has been called defensive forecasting (Vovk, Takemura, and Shafer, 2005) or forecast hedging (Foster and Hart, 2021). This literature emerged in response to Schervish (1985) showing that calibration in the classical sense cannot be guaranteed by any algorithm. Foster and Vohra (1998) showed that, under very mild assumptions, probability calibration in the sense of a low ECE can be guaranteed (in probability) by a stochastic algorithm that minimises swap regret (for the Brier score). However, such backward-looking algorithms often simply converge to (more or less stable) base rates, without refined predictions that allow well-informed policies. While they can be adapted to smaller patches of the input space (see also the literature on multi-calibration (H´ebert-Johnson et al., 2018)), this is not much better than simply predicting the average per patch, thereby deciding in advance which patches get individual decisions. In this, such approaches are more constrained than, e.g., human weather forecasters who were trained to first sort similar weather situations into ‘categories of likelihood of occurrence’ and then predict a calibrated forecast per category (Sanders, 1963, p. 200).

This example and those of Section 2 may suggest that induction about calibration always reduces to stable relative frequencies of repeated trials; this is also not the case. Consider simulation models in the rain prediction example: If the measurements are fine enough, it may be the case that no input $x _ { i } ~ \in ~ { \mathcal { X } }$ occurs more than once and no prediction is issued more than once, implying that there are no repeated trials. Further examples are given by logistic regression or more complex Machine Learning models used for probabilistic predictions. In general, calibrated predictors may rely on some structure in the relationship between inputs and labels that does not reduce to stable relative frequencies. Otherwise, simulation-based and ML-based predictions could simply be replaced by ‘reference class forecasting’ (Flyvbjerg, Glenting, and Rønnest, 2004).

Empirically, we do see that prediction methods can often be designed such that they are (approximately) calibrated on sets of interest, which simply means that they neither systematically over- nor systematically under-predict. Besides rain forecasts and the examples in Sections 2.2 and 2.3, we can point to machine learning (ML) models which often aim for calibration on sets of equal prediction. It has been observed that especially modern, over-parameterised ML models need explicit post-processing, whereas others are automatically calibrated on sets of equal prediction (Guo et al., 2017). One could argue that on a high level, humans also do something like this post-processing: if we are repeatedly overor under-predicting (i.e. are not calibrated) on sets of interest, we will ideally notice that; since we do not know on which of the individual events our predictions were too low/high, we systematically increase/decrease our predictions in similar situations<sup>7</sup> in the future. But why should future predictions then still be calibrated?

## 4.1 Induction and the feasibility of calibration

Any prediction method, indeed any prediction about the future, relies on an inductive assumption: that the future will resemble the past in some relevant way. This relevance can be made more precise for our purpose: that a prediction method which has repeatedly proven to be (approximately) calibrated on some sets in the past will be (approximately) calibrated on similar sets in the future. Below, we prove a formal result in support of this particular inductive assumption, similar to the argument for induction made by Williams (1947). In contrast to the cited work, we are dealing not only with integers but with real numbers, which is why we draw on an established concentration inequality.<sup>8</sup> In particular, we make use of the combinatorial bound provided by Hoefding’s inequality for drawing without replacement.

Proposition 8 (Calibration on samples from a population).

Take a predictor ${ \mathsf { p } } : { \mathcal { X } } \times { \mathcal { A } }  [ 0 , 1 ]$ and N prediction instances represented by $( x _ { i } , A _ { i } ) \in$ $\mathcal { X } \times \mathcal { A }$ . Now consider drawing a ‘sample’ of d instances from the ‘population’ of N instances. Then the samples $\{ j _ { 1 } , . . . , j _ { d } \} \subset \{ 1 , . . . , N \}$ whose average calibration error difers by more than ϵ from the average calibration error of the population, i.e. where

$$
\frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left( p _ { j _ { i } } - y _ { j _ { i } } \right) - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( p _ { i } - y _ { i } \right) \geq \epsilon ,\tag{15}
$$

(y<sub>i</sub> denoting whether $A _ { i }$ occurs and $p _ { i } : = { \mathsf { p } } ( x _ { i } , A _ { i } ) )$ make up for less than exp $\left( - \frac { 1 } { 2 } d \epsilon ^ { 2 } \right)$ of all possible samples of that size.

Proof.

Our result follows directly from Hoefding’s inequality for drawing without replacement. We simply insert $z _ { i } : = p _ { i } - y _ { i } , a = - 1$ and $b = + 1$ in the below statement taken from Proposition 1.2 of Bardenet and Maillard (2015):

Let $Z = ( z _ { 1 } , . . . , z _ { N } )$ be a finite population of $N$ points and $Z _ { 1 } , . . . , Z _ { d }$ be a random sample drawn without replacement from Z. Let

$$
a : = \operatorname* { m i n } _ { 1 \leq i \leq N } z _ { i } \quad { \mathrm { a n d } } \quad b : = \operatorname* { m a x } _ { 1 \leq i \leq N } z _ { i } .\tag{16}
$$

Then for all $\epsilon > 0$

$$
\mu \left[ \frac { 1 } { d } \sum _ { i = 1 } ^ { d } Z _ { i } - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z _ { i } \geq \epsilon \right] \leq \exp \left( - \frac { 2 d \epsilon ^ { 2 } } { ( b - a ) ^ { 2 } } \right) ,\tag{17}
$$

where $\mu$ measures the proportion of admissible combinations in drawing d of the N points. □

Hence, the average calibration error on large enough samples will mostly be close to the average calibration error of the whole population. In a move analogous to that of Williams (1947), we can also infer that if I am approximately calibrated on a large enough sample from a population or set of prediction instances, I will in most cases also be calibrated on the whole set and, thus, on similar sets in the future. Let us illustrate the bound with concrete numbers. If I have an average calibration error of 0.2 on the whole population, then I will get a calibration error of less than 0.05 in less than 10% of possible samples of size $d = 2 0 0 \mathrm { : }$ ; for $d = 5 0 0$ , this ratio goes down to 0.4%. Hence, the vast majority of possible samples will not mislead me into thinking that I will be well-calibrated in the future in such a setting. Note that the proposition only provides an upper bound, so that the actual number of non-representative samples will be lower still. While this result does not prove the possibility of induction, it shows that calibration in the past is an indicator for calibration in the future—on sets that can be thought to be drawn from the same population. Whether this is a sensible model in a given situation depends on whether there is reason to believe that the sample is unbiased.<sup>9</sup>

Another interesting implication of the result concerns the mixing of predictions from multiple, diferent calibrated prediction methods. If n prediction methods are calibrated on d events each, then the resulting n·d predictions are clearly also calibrated on the $n { \cdot } d$ events; Proposition 8 can also be applied to this larger set of predictions, now inferring from the population to subsets: It shows that most large enough subsets of these $n \cdot d$ predictions will also be approximately calibrated, even though they come from a mix of prediction methods. This is important because it shows that the Proposition is not only relevant for predictions from the same prediction method. In sum, we can give arguments why, in certain cases, we expect to be calibrated on future events—but we can never be sure: ‘Nature will always maintain her rights, and prevail in the end over any abstract reasoning whatsoever’ (Hume, $1 7 7 7 , \mathrm { p } . 5 . 1 . 2 )$

## 4.2 Extrapolating calibration

Another way of generating sets of calibrated predictions is through certain other sets of calibrated prediction—by using probabilistic reasoning. Even attentive readers probably missed the interesting fact in Proposition 5 that $1 - p _ { i }$ automatically emerged as the prediction for $1 - y _ { i }$ , without imposing Kolmogorov’s axioms. We now show more generally that predictors need to satisfy these axioms (in their second argument) in order to be calibrated on certain $\mathrm { s e t s . } ^ { 1 0 }$ Take a predictor $\mathsf { p } : \mathcal { X } \times \mathcal { A }  \mathbb { R }$ and d prediction instances represented by $( x _ { i } , A _ { i } ) \in \mathcal { X } \times \mathcal { A }$ with $p _ { i } : = { \mathsf p } ( x _ { i } , A _ { i } )$ and let $y _ { i } \in \{ 0 , 1 \}$ denote whether $A _ { i }$ occurs. Let $y _ { A } , y _ { B } , y _ { A \cup B } , y _ { \Omega } \in \{ 0 , 1 \}$ denote whether events $A , B , A \cup B , \Omega \in { \mathcal { A } }$ occur at the last instance $d .$ Let A be an algebra over some set Ω where Ω is a sure event: It exhausts all possibilities; that is, for its label, it is known that $y _ { \Omega } = 1$

1. Non-negativity: If there is a nontrivial subset $I : = \{ i : p _ { i } < 0 \} \subset \{ 1 , . . . , d \}$ where p predicts negative values, then p cannot be calibrated on this subset, regardless of whether the predicted events occur:

$$
\sum _ { i \in I } p _ { i } < 0 \leq \sum _ { i \in I } y _ { i } .\tag{18}
$$

2. Normalisation: Let $A _ { d } = \Omega$ and p be calibrated on the set $\{ 1 , . . . , d - 1 \}$ . Then p is calibrated on $\{ 1 , . . . , d \}$ if and only if $\mathsf { p } ( x _ { d } , \Omega ) = 1$ , regardless of $x _ { d }$

3. Additivity: Let $A , B \in { \mathcal { A } }$ be disjoint events in the sense that $y _ { A } + y _ { B } \leq 1$ (i.e. it cannot be that both labels are equal to 1 at the same instance); this implies $y _ { A } + y _ { B } = y _ { A \cup B }$ Now assume p is calibrated on the set $S : = \{ ( x _ { 1 } , A _ { 1 } ) , . . . , ( x _ { d - 1 } , A _ { d - 1 } ) , ( x _ { d } , A ) , ( x _ { d } , B ) \}$ ,

where p is used for two predictions at instance d. Then

$$
{ \begin{array} { r l } { \displaystyle \mathsf { p } ( x _ { d } , A \cup B ) + \sum _ { i = 1 } ^ { d - 1 } p _ { i } - y _ { A \cup B } - \sum _ { C \in J } y _ { i } = \mathsf { p } ( x _ { d } , A \cup B ) + \sum _ { i = 1 } ^ { d - 1 } p _ { i } - y _ { A } - y _ { B } - \sum _ { i = 1 } ^ { d - 1 } y _ { i } } & \\ { \displaystyle } & { \left( 1 9 \right) } \\ { \displaystyle } & { = \mathsf { p } ( x _ { d } , A \cup B ) - \mathsf { p } ( x _ { d } , A ) - \mathsf { p } ( x _ { d } , B ) \qquad ( 2 0 ) } \\ { \displaystyle } & { \qquad + \mathsf { p } ( x _ { d } , A ) + \mathsf { p } ( x _ { d } , B ) + \sum _ { i = 1 } ^ { d - 1 } p _ { i } - y _ { A } - \sum _ { i = 1 } ^ { d - 1 } y _ { i } } \\ { \displaystyle } & { = \mathsf { p } ( x _ { d } , A \cup B ) - \mathsf { p } ( x _ { d } , A ) - \mathsf { p } ( x _ { d } , B ) \qquad ( 2 1 ) } \end{array} }
$$

where the last step uses the assumption of calibration on S. So under that assumption, p is calibrated on $\{ 1 , . . . , d \}$ with $A _ { d } = A \cup B$ if and only if $\mathsf { p } ( x _ { d } , A \cup B ) = \mathsf { p } ( x _ { d } , A ) +$ $\mathsf { p } ( x _ { d } , B )$ , regardless of $x _ { d }$

The sets that allow if-and-only-if statements are quite specific here; in this sense, it resembles Dutch book arguments, where any single inconsistency can in theory be exploited indefinitely. Here, however, the implications are more practical: If someone is perfectly calibrated on forecasting ‘rain’ but does not obey the probability axioms on one ‘no rain forecast, then for some utility functions (in the setting of Section 3.3), the best-response policy is guaranteed to lead to sub-optimal decisions due to miscalibration.

We can also motivate the definition of conditional probabilities by the demand for calibration (somewhat analogous to definitions via relative frequencies). Consider the task of predicting events $A , B \in { \mathcal { A } }$ at d instances. For ease of presentation, assume that all d inputs coincide, i.e. $x _ { 1 } = . . . , x _ { d } = x \in \mathcal { X }$ . This allows us to drop p’s dependence on $x \in \mathcal { X }$ and consider a predictor ${ \mathsf { p } } : { \mathcal { A } }  [ 0 , 1 ]$ in the following derivation; a more general version is presented in Appendix B. Now assume p to be calibrated on $A \cap B$ and on B across the d instances, where $y _ { i } ^ { A \cap B }$ and $y _ { i } ^ { B }$ denote whether $A \cap B$ and B occur at instance $i \in \{ 1 , . . . , d \}$ respectively. That is, assume $\begin{array} { r } { \sum _ { i = 1 } ^ { d } \mathfrak { p } ( A \cap B ) = \sum _ { i = 1 } ^ { d } y _ { i } ^ { A \cap B } } \end{array}$ and $\begin{array} { r } { \mathsf { p } ( B ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } y _ { i } ^ { B } > 0 } \end{array}$ Then p is calibrated on $A | B$ for {i : y<sup>B</sup><sub>i</sub> = 1} (i.e. for the set of steps where $B$ occurs, see first line below) if and only if it satisfies $\begin{array} { r } { \mathsf { p } ( A | B ) = \frac { \mathsf { p } ( A \cap B ) } { \mathsf { p } ( B ) } } \end{array}$

$$
\begin{array} { r l r l } & { \displaystyle \sum _ { i = 1 } ^ { n } p ( A | B ) = \sum _ { i \neq j = 1 } ^ { n } \frac { \partial ^ { i } } { \partial x ^ { i } } } \\ & { \displaystyle \sum _ { i = 1 } ^ { n } p ( A | B ) = \sum _ { i \neq j = 1 } ^ { n } \frac { \partial ^ { i } } { \partial x ^ { i } } } \\ & { \displaystyle \sum _ { i = 1 } ^ { n } p ( A | B ) = \sum _ { i \neq j = 1 } ^ { n } \frac { \partial ^ { i } } { \partial x ^ { i } } } \\ & { \displaystyle \sum _ { i = 1 } ^ { n } p ( A | B ) = \sum _ { i = 1 } ^ { n } y ^ { i \neq j \neq j } } \\ & { \displaystyle \sum _ { i = 2 } ^ { n } p ( A | B ) = \sum _ { i = 1 } ^ { n } y ^ { i \neq j \neq j } } \\ & { \displaystyle \sum _ { i \neq j = 1 } ^ { n } p ( A | B ) = \sum _ { i } ^ { n } p ( A \cap B ) } \\ & { \displaystyle \sum _ { i \neq j = 1 } ^ { n } p ( A | B ) = \sum _ { i } ^ { n } p ( A \cap B ) } \\ & { \displaystyle \sum _ { i = 1 } ^ { n } p ( A | B ) = \sum _ { i } ^ { n } p ( A \cap B ) } \\ & { \displaystyle ( A | B ) = \sum _ { i = 1 } ^ { n } p ( A \cap B ) } \\ & { \displaystyle \sum _ { i = 1 } ^ { n } p ( A | B ) = \sum _ { i } ^ { n } p ( A \cap B ) . } \end{array} \quad \quad \mathrm { ( b y ~ c a l i n c a t i o n ~ o ^ i ~ p o ~ A ~ ) }
$$

Summing up, the probability calculus can be seen as a sound and complete system for generating calibrated predictions on certain sets from calibrated predictions on related sets. While this does not settle the question whether all predictors need to follow the probability calculus (i.e. that they are probability measures in their second argument), it does provide a pro tanto reason.

## 5 A Unified Interpretation of Probability

Norms of belief are as remote from empirical claims about nature as is Hume’s simpler subjectivism. Propensity theories of probability propose a physical property that cannot be recorded and does not necessitate or preclude any occurrence. [. . . ] any limiting-frequency claim is consistent with any claim about any finite collection of events.

— Clark Glymour (2001)

Russell’s famous dictum that ‘probability is the most important concept in modern science, especially as nobody has the slightest notion what it means’ (cited by Bell (1945, p. 582)) is almost a century old; but while there have certainly been many new developments, a satisfying interpretation is still lacking. The purpose of this section is to argue that his lacuna can be filled by the perspective on probabilities put forward in the present paper. To make this argument, we now specifically relate our account to the literature on interpretations of probability. In the most authoritative up-to-date treatment of the subject, H´ajek (2023) asks ‘what do we want from our interpretations of probability, specifically?’ (original emphasis) and then answers by suggesting a list of desiderata (drawing on Salmon (1966)). Some of these we have already covered above: Our account satisfies ‘non-triviality’ (not just zero and one) and ‘admissibility with respect to this or that axiomatization’ (motivating the axioms of the probability calculus); it also illuminates ‘ampliative inferences’ in the sense that it allows to reason about the justification of probabilistic statements based on other probabilistic statements (Section 4). We now, in turn, discuss H´ajek’s remaining desiderata: the applicability to science, rational belief, frequencies, and rational decision making.

## 5.1 Relation to science and the question of (in-)determinism

During the nineteenth century it became possible to see that the world might be regular and yet not subject to universal laws of nature. A space was cleared for chance.

— Ian Hacking (1990)

Assessing the applicability to science means checking the compatibility with scientific practice and current scientific theories. In particular, the question of whether the universe is deterministic is sometimes taken to bear directly on how we should think about probability. For example, David Lewis (1980, p. 120) thought that objective or physical probabilities rely on indeterminism. Karl Popper (1959) also proposed the propensity account of probability (according to which probabilities are physical properties) in the context of Quantum Mechanics (QM). We should briefly note that QM does not require the world to be indeterministic, given that diferent, empirically indistinguishable interpretations of QM disagree on this question (not even getting into the question of scientific realism). So there is certainly no need to presuppose this. But even if QM came with some notion of true probabilities, it is unclear that they would be of any relevance to the probabilities we deal with day-to-day, for two reasons: First, QM probabilities need to be described by a more general theory of probability than that axiomatised by Kolmogorov (Streater, 2000). Second and more importantly, even for a coin flip, we would never have access to the true QM-based probabilities: We are neither able to determine the initial conditions, that is, the complete wave function, nor to take into account the extremely high number of occurring quantum interactions.<sup>11</sup> The resulting values may, thus, vastly difer from any predictions we are able to make—which, as we have shown, are still useful. In sum, we do not see compelling reasons to suppose either determinism or indeterminism, nor to think that indeterminism at the level of QM would contribute much to probabilistic reasoning. It is therefore a strength of our account that, showing how probabilities can be constructed and used, it remains agnostic regarding the question of determinism.

Indeed, the intuition that probabilities are objective may depend less on QM and more on the often strong interpersonal agreements about ‘correct’ prediction methods e.g. for gambling. In the words of Michael Strevens (2006, p. 31), probabilities in such settings ‘have attained a certain kind of stability under the impact of additional information. This stability gives them the appearance of objectivity, hence of reality, hence of physicality’. We argued in Section 2.2 that this appearance is misleading, as illustrated by the roulette story of Thorp and Shannon. In line with this, the ‘erosion of determinism’ indeed did not follow the advent of QM but of higher-level statistical regularities discovered during the previous century, as captured in Hacking’s epigraph above. It is also important to note that the higher-level sciences depend heavily on abstractions such as the ones that feature in our notion of prediction methods. Abstractions have been observed to be both part of scientists tacit knowledge (Polanyi, 1958), especially the ‘ability to recognize a given situation as like some and unlike others’ (Kuhn, 2012, p. 195), and a substantial part of conscious scientific work (Danks, 2015; Potochnik, 2017)—yet they, arguably, still remain an under-explored topic.

## 5.2 Relation to (rational) degrees of belief

Chances are degrees of belief [. . . ]; not those of any actual person, but in a simplified system to which those of actual people, especially the speaker, in part approximate.

Frank P. Ramsey (1928/1931)

While probabilities are often said to have a direct connection to degrees of belief, we argue that the notion of probability does not depend on degrees of belief: prediction methods can be used in a purely mechanical way to get desirable outcomes on aggregate, without an entity involved that is commonly held to have beliefs. A simple machine that takes measurements and uses a predictor could make decisions based on probabilities without it being plausible to ascribe to it beliefs that come in degrees. However, probabilities can also be used to describe degrees of belief under uncertainty. In most cases, it is dificult to pin down a particular prediction method, especially as human predictions tend to be qualitative. But humans also take only specific information into account and exploit regularities such as symmetries or stable relative frequencies. In some cases, the gap between human reasoning and quantitative prediction methods can become fairly small—it appears that some people, modestly described as ‘super-forecasters’, are particularly good at making calibrated quantitative predictions (Mellers et al., 2015). The notion of prediction methods can also shed light on imprecise notions of (subjective) uncertainty. Particularly vague degrees of belief or disagreements between diferent methods can be represented by imprecise predictions (Appendix A.3)<sup>12</sup> while Knightian uncertainty (Knight, 1921) corresponds to the absence of a trusted prediction method. In this sense, Section 2 also tells us an idealised story of human reasoning: Consciously or not, humans often implement something close to prediction methods—in that sense, probabilities can model human degrees of belief, a view also expressed in Ramsey’s epigraph.

While probabilities should not be taken as actual degrees of belief, we can also model consistent decision-making by humans or machines as if they had certain degrees of belief: Savage (1972) famously showed that actions which follow certain consistency criteria can be viewed as maximising expected utility for implicit utility functions and probabilities. Expected utility maximisation (EUM) is, thus, often taken to be an approximation of human decision-making, that is, as a descriptive theory: We tend to make decisions such that good outcomes seem more likely to us. There are, of course, considerable caveats. The most crucial ones are arguably diminishing marginal utility and risk aversion, already highlighted by Ramsey (1926/1931, p. 172) and analysed e.g. in (Wakker, 1994). In Ramsey’s words, EUM as a modelling tool is an ‘artificial system of psychology, which like Newtonian mechanics can, I think, still be profitably used even though it is known to be false (Ramsey, 1926/1931, p. 173). So, we can connect degrees of belief to probabilities, by mod elling reasoning and decision-making through prediction methods and Savage-style decision theory–but stop short of equating them.

There is, of course, some flexibility in deciding what to use the word ‘probability’ for. One may want to use it for the assignments in Savage-style models, that is, for implicit degrees of belief that can be assigned whenever some agent acts consistently. It seems, however, more consistent with everyday use to reserve it for the predictions themselves, for weather predictions and coin flips, and to say that we can model consistent decision as if they follow EUM under certain probabilities. As shown in Section 4.2, we can get calibrated predictions from calibrated predictions of related events using the probability calculus. This provides a pro tanto reason for considering the probability axioms to constitute constraints on rationality. We emphasise again that we do not put forward a normative theory of rational belief or action here, but a descriptive perspective on probability that can illuminate its perceived connections to rationality.

## 5.3 Relation to frequencies and the reference class problem

If we are asked to find the probability holding for an individual future event, we must first incorporate the case in a suitable reference class. An individual thing or event may be incorporated in many reference classes, from which diferent probabilities will result. This ambiguity has been called the problem of the reference class.

Hans Reichenbach (1949)

Although our perspective implies that probabilities are constructed, it also explains their strong connection to observed relative frequencies. Indeed, if we restricted calibration to sets of equal probability (as calibration is sometimes understood), the relationship would be even closer: Then, the calibration condition would be equivalent to the definition of probability in finitary frequentism:

$$
\sum _ { i = 1 } ^ { n } p _ { i } = \sum _ { i = 1 } ^ { n } y _ { i } \qquad \Leftrightarrow \qquad p _ { i } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } y _ { i } .\tag{22}
$$

Instead of taking this as a definition, we think it more adequate to see it as a special case of our main quality criterion. Not just because such finitary definitions are problematic (H´ajek, 1996),<sup>13</sup> but also because it would imply too narrow an evaluation criterion.

The ties between frequentism and our account become particularly clear in reference to the so-called reference class problem. Its metaphysical version is a problem for objectivist theories like frequentism that claim a unique true probability for each event (H´ajek, 2007). The epistemic version concerns the question of how a reference class should be chosen for a given event, considering that diferent choices would lead to diferent probabilities. This has led H´ajek, for example, to argue that conditional probabilities are actually primitive, as probabilities are always conditional on a certain conceptualisation of events. While our predictors do resemble them, they are not strictly speaking conditional probabilities, as X is just an arbitrary set without well-defined probabilities; as we show in Section 4.2, it makes more sense to consider conditional probabilities to further depend on a particular (perhaps implicit) abstraction $x \in \mathcal { X }$ (see the implicit joint ‘conditioning’ in Appendix B). H´ajek (2007) also supposed that, rather than a marginal probability, ‘[v]arious frequentists could tell us the conditional probability that John Smith will live to age 61, given that he is a consumptive Englishman aged $5 0 ^ { \circ }$ (ibid., p. 582, original emphasis). However, there can be diferent mortality tables resulting in diferent ratios—more generally, the choice of abstraction does not yet fix the prediction. This aspect is clear for the prediction method perspective, as diferent methods may use the same scheme of abstraction but diferent predictors, relying e.g. on diferent mortality tables.<sup>14</sup> For these reasons, we agree with Freedman (1997, p. 23) that ‘probability is a subtler idea than relative frequency’. We would also argue that the reference class problem is not actually a problem. Diferent prediction methods may be calibrated on diferent sets, so one can choose a prediction method that promises calibration on sets of interest. This relates to to the more general idea of the ‘goal-dependence in scientific ontology’ (Danks, 2015).

## 5.4 Relation to rational decision-making and individual predictions

Legends of prediction are common throughout the whole Household of Man. God speaks, spirits speak, computers speak. Oracular ambiguity or statistical probability provides loopholes, and discrepancies are expunged by Faith.

— Ursula K. Le Guin (1969): The Left Hand of Darkness

Our discussions have focused on sets of predictions rather than individual ones. This is not a coincidence, as we take probabilities to not be free-floating numbers but to rely on prediction methods which are useful when sets of predictions are calibrated.<sup>15</sup> But what exactly is the relation between a prediction and the corresponding event? And can we evaluate the quality of a single non-trivial prediction? That is, is a prediction of 0.6 better than a prediction of 0.4 if the predicted event occurs? What should we do if we only get a single prediction (for some level of utility)?

The first three questions all relate to the dependence of a prediction on the method that generated it. It is important to emphasise that the probability is not a property of the event, as it is constructed and depends on the choices of both the abstraction and the predictor. As discussed in Sections 2.2 and 5.1, gambling setups only appear to have objective probabilities because of their relative stability under additional information We also mentioned the example of Angelina Jolie, who stated ‘My doctors estimated that I had an 87 per cent risk of breast cancer’, with the number presumably coming from statistical data about women with a particular genetic mutation. P. Dawid (2017) asks, ‘Was Angelina (or her doctors) right to interpret it as her own individual risk?’ (p. 3456). On our account, they were—with the qualification that this risk is model-dependent and constructed rather than objective and discovered—as is any other probability. Which prediction method is most useful depends on which sets we want to be calibrated on (although the perfect binary predictor is always optimal). In hindsight, we can usually say which prediction would have been good or correct. The validation of prediction methods cannot, however, be thus reduced to comparisons between individual predictions, not even with proper scoring rules (Gneiting and Raftery, 2007). They do allow us to put a number on our intuition that 0.6 is somehow a better prediction than 0.4 if the predicted event occurs; but so does any notion of calibration error (as the ℓ loss deployed in Appendix A.1). After all, proper scoring rules are meant to be ‘appropriate for evaluating and comparing forecasters who repeatedly present their predictions’ (DeGroot and Fienberg, 1983, p. 12, emphasis added).

The fourth and last question about acting on single predictions is related but more complex. In general, we suggest that policies rather than single actions should be the subject of justification and evaluation. An ex-post evaluation of a decision would ignore the prediction and just consider whether an alternative decision would have been better in hindsight—this is not particularly helpful. Instead, what is familiar also from legal and ethical reasoning (especially deontological, but even rule-consequentialist), is to judge decisions by the reasons or maxims that they were based on.<sup>16</sup> For example, we showed that maximising expected utility is a good policy if we can assume calibration on sets of equal utility and wish to maximise cumulative utility (Section 3.1). As noted before, being calibrated for all possible combinations of decisions would require perfect discrimination. In the umbrella example of Section 3.3, we showed that the calibration criterion can be more benign when comparing a more restricted set of sensible policies. But the problem is more dificult e.g. when we only have a few predictions for particularly grave events: If we only make a few high-stakes decisions, such as a choice of treatment for breast cancer (where one may even argue that the concept of numerical utility breaks down), it seems too big of an assumption to hope for calibration on such a small set. That being said, the combinatorial reasoning from Section 4.1 also extends to single predictions: Good calibration in the past gives some reason to believe in low calibration error on the single prediction, i.e. the more reason to believe in the event, the higher the prediction: The strength of this mathematically-grounded pro-tanto reason is monotonous in the number of events, so some (even if small) reason to believe will remain.<sup>17</sup> In general, for such situations, it may be more sensible to be risk-averse than in low-stakes settings where there are multiple events with comparable utility (cf. Buchak, 2013; Thoma, 2019).<sup>18</sup> A way to model this would be via imprecise calibration as explored in Appendix A.3.

## 6 Comparison with Conventional Interpretations

H´ajek (2023) notes that ‘[e]ach interpretation that we have canvassed seems to capture some crucial insight into a concept of [probability], yet falls short of doing complete justice to this concept.’ Any new satisfactory account of probability should, thus, be expected to make proponents of other accounts feel vindicated on some aspects that are particularly close to their hearts. We think that this is the case for our notion of probabilities as outputs of prediction methods aiming to predict numbers of occurring events. It is interesting to note, for example, that our predictors $\mathsf { p } : \mathcal { X } \times \mathcal { A }  \mathbb { R }$ resemble the confirmation function central to logical accounts of probability, such as that of Carnap (1950) or Keynes (1921) (with precursors as early as Leibniz, cf. (Hacking, 1975)). Our predictors, however, are neither objective relations nor relations between propositions—they are functions of abstractions in a set X and events in an algebra A. In this section, we briefly survey a number of other prominent interpretations and highlight what we take to be the most interesting similarities and diferences w.r.t. our account.

Bayesianism roughly posits that probability and its theory are concerned with degrees of belief and rationality constraints thereon. What we agree with is that probabilities are constructed and that it is misguided to search for true probabilities. However, we ground them in prediction methods rather than degrees of belief (Section 5.2) and highlight that these methods aim to track structure in sets of observations. This makes it possible to replace notions of internal cohesion or rationality with that of empirical calibration, and thereby a guide to decision-making that guarantees good outcomes. A Bayesian account that is particularly close to ours is that of Philip P. Dawid (2017).<sup>19</sup> On the one hand, his suggestion to arrive at ‘probability forecast[s] by assessing the odds at which I would be willing to bet’ (p. 3471) is clearly Bayesian in the tradition of (De Finetti, 1937). On the other hand, he also suggests to evaluate individual predictions on aggregate data via calibration—although its precise scope and relevance do not become entirely clear. In particular, it remains unclear why calibration on future data is important and on which (finite/infinite) sets it matters.<sup>20</sup> In comparison, our notion of prediction methods focuses on (potentially) inter-subjective models and the role of abstraction, which is decoupled from the events $A \in { \mathcal { A } }$ that we wish to be calibrated on. In a way, then, we posit a variant of Bayesianism without degrees of belief or betting and with a more concrete connection to the world, enabling not only the avoidance of sure loss in Dutch books but successful action in everyday life.

Hypothetical frequentism can be defined as the suggestion that ‘the probability of an attribute A in a reference class B is the value the limiting relative frequency of occurrences of A within B would be if B were infinite’ (H´ajek, 2023). This captures the intuition of identifying probabilities with ratios in repeated trials. While this sounds very diferent to our account at first glance, we already discussed two similarities in Section 5.3: One is the dependence of individual probabilities on other events and on a choice of abstraction (via prediction methods, in our case), leading us to a generalisation of the reference class problem. Furthermore, equating probabilities with relative frequencies is a special case of our notion of calibration, which we consider for finite sets. We do reject the jump to declaring that probabilities themselves are ‘out there’ in any interesting sense. The finite frequentist account of Glymour (2001), mentioned in footnote 13, provides, in a sense, an intermediate account.

Karl Popper abandoned frequentism in favour of his propensity account because the former could not make sense of sequences with few trials. He thus proposed that frequentists should alter their theory by letting it ‘say that admissible sequences must be either virtual or actual sequences which are characterised by a set of generating conditions—by a set of conditions whose repeated realisation produces the elements of the sequence’ (Popper, 1959, p. 34, original emphasis). This is still an objectivist theory but dispenses with the reliance on infinite trials, instead invoking a new sort of mysterious property (especially in the case of a deterministic universe, which Popper did not seem to assume). We argued that relevant probabilities are independent of ‘true’ probabilities that may or may not be implied by Quantum Mechanics (Section 5.1). Propensity accounts often have a frequentist flavour, indirectly highlighting the importance of sets of events. It is interesting to note that, as the equivalence classes of generative conditions are idealisations (ignoring background conditions, cf. Section 2.2), they can be seen as abstractions made by prediction methods. However, propensities are typically thought to be physical rather than model-dependent, which is in stark contrast to our account—although the relevant literature sometimes also invites a reading of model-dependent propensities.

Another interesting interpretation of probability is the best-systems account of David Lewis (1994), which also posits objective chances: On this view, ‘the chances are what the probabilistic laws of the best system say they are’ (p. 480). ‘The best system is the one that strikes as good a balance as truth will allow between simplicity and strength. [. . . ] If nature is kind, the best system will be robustly best [. . . ] It’s a reasonable hope’ (ibid., p. 478f). Now this account presupposes what may seem a tremendous kindness of nature as well as a perhaps weak notion of truth and objectivity—the latter fits well into Lewis’ Humean view on laws of nature. What is interesting here about Lewis’ account is that it resonates with the hope for a best level of predictive depth expressed by P. Dawid (2017, p. 3465)—in turn similar to the ‘primary resolution’ of Li and Meng (2021). Indeed, Dawid could be seen as linking the best-systems view on probability with our more pragmatic notion of model-based predictions. The clearest diferences on the side of Lewis are the integration within a more global systematisation of the universe and the belief in objectivity, hinging on the existence of a privileged description. If there were an objectively best predictor and we assigned to it some notion of truth, these diferences would blur.<sup>21</sup> However, this hope for or pretension of objectivity is also what Clark Glymour criticises in typical frequentist takes.

In line with our analysis, Glymour thinks that central problems with Bayesianism and frequentism lie, respectively, in the neglect of empirical claims and the unnecessary stipulation of objectively true probabilistic statements:

The sometimes bitter debates between those who describe themselves as frequentists and those who describe themselves as subjective Bayesians has often turned on charges by the former that the latter abandon the “objectivity” of science and by the latter that the former dissemble about the “subjectivity” of their probability judgements. My belief is that, among statisticians anyway, the dispute often confuses content with justification. The “objectivity” of the frequentists is in the content of their probability judgements, which, while usually stated as about an unempirical probability, are often really vague empirical claims about finite frequencies. That sort of objectivity is genuinely lost in subjective Bayesian interpretations. The “subjectivity” kept hidden by frequentists is that there is often no explicit justification beyond their own opinion for aspects of their empirical claims. That subjectivity can be made entirely explicit without sacrificing the objective–that is empirical–content of frequency claims, and its recognition does not require, or even invite, recourse to subjective probability. Bayesian criticisms do address a confused and uncertain frequentist statistical practice, in which the point of making empirical claims is often forgotten or fudged. (Glymour, 2001, p. 299f)

We have argued that our account avoids these problems by stating that probabilities are constructed rather than discovered while still taking their justification directly from empirical observations. Even more, we connect successful decision-making with empirical evaluation and assumptions about induction through a general notion of calibration, which has not been considered a central concept by any of the conventional accounts.

## 7 Conclusion

Relying on the notions of finite calibration and prediction methods, we have provided a more or less pragmatic account of probabilities and how they are useful for decision-making. We showed that if predictions satisfy an (often feasible) calibration criterion, then it is possible to predict the distribution of utilities that a given policy will yield. In particular, the sum of one’s predicted utilities will match the actual cumulative utility, which can provide a rationale for expected utility maximisation. A central element of our account is the semi-formal notion of prediction methods that construct abstractions of given situations and feed them to a model. Arguably, the novelty here consists less in the consideration of predictions than in the connections drawn to abstractions and to successful decision-making via calibration in very general terms. Indeed, we argue that this perspective elucidates the relationship between gambling odds and rain predictions, uniting the alleged two faces of probability by tying together abstraction, forecasting, probability theory, and empirically successful decision-making.

The understanding of probability also has important implications for machine learning. For example, it underscores the relevance of evaluating calibration beyond sets of equal predictions, as already explored by P. Dawid (2017) and H¨oltgen and Robert C Williamson (2023). Furthermore, it underscores that one should be wary of the often-invoked concept of a ‘true distribution’ from which one can ‘sample’ once one steps outside of the casino or other highly controlled settings. Given the centrality of probability for causality, many considerations also spill over to the latter; indeed, we take a deeper dive into causal inference from this perspective in concurrent work, also highlighting the role of calibration (H¨oltgen and Robert C. Williamson, 2025). It has been observed that algorithmic predictions based on machine learning tend to convey an air of authority and objectivity, as the many choices involved in data collection (abstraction scheme) and model tuning (choice of predictor) often remain beneath the surface (Moss, 2022). This is particularly relevant for the justification of predictions that inform decisions about people (H¨oltgen and Robert C. Williamson, 2026). Our work highlights that probabilities, e.g. of finding a job, are not properties of people; instead, they depend on the selected abstraction and model, which, in turn, depends on data about other people. Hence also our answer to Cynthia Dwork’s question from the introduction: There is no ideal algorithm, as there are no true probabilities to uncover, and diferent algorithms can be better suited for diferent goals. While we hope that this work helps to sharpen the view on probability, a sea of open questions still calls for further exploration.

## Acknowledgments

For helpful feedback on previous versions, I would like to thank Ben Jantzen, Bob Williamson, Elisa Nguyen, Jannik Th¨ummel, Kate Vredenburgh, Konstantin Genin, Rabanus Derr, and Timo Freiesleben. This work was funded by the German Federal Ministry of Education and Research (BMBF): T¨ubingen AI Center, FKZ: 01IS18039A. I also thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for their support.

## A Generalising Proposition 5

## A.1 Approximate calibration

Here, we generalise Proposition 5 to only require approximate calibration—we give a bound on how large the calibration error on each set of equal utility can be in order to keep the diference between predicted and cumulative utility below some $\epsilon > 0$

Proposition 9 (Predicting cumulative utility: Approximate calibration).

We assume the same setting as in Proposition 5 except that we now require all utilities to be positive—one may otherwise simply shift the values to a positive domain. If we then replace condition (3) with the assumption that $\forall u ^ { \prime } \in \mathcal { U }$

$$
\left| \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } y _ { i } - \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } p _ { i } \right| \leq \frac { \epsilon } { 2 \cdot u ^ { \prime } \cdot | \mathcal { U } | }\tag{23}
$$

and the same $f o r ~ u _ { i } ( 1 )$ , then

$$
\left| \sum _ { i = 1 } ^ { d } [ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) ] - \sum _ { i = 1 } ^ { d } [ p _ { i } u _ { i } ( 1 ) + ( 1 - p _ { i } ) u _ { i } ( 0 ) ] \right| \leq \epsilon .\tag{24}
$$

Proof.

$$
\left| \sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] - \sum _ { i = 1 } ^ { d } \left[ p _ { i } u _ { i } ( 1 ) + ( 1 - p _ { i } ) u _ { i } ( 0 ) \right] \right|
$$

$$
= \left| \sum _ { u ^ { \prime } \in \mathcal { U } } \left( \left( \sum _ { \substack { i : u _ { i } \left( 1 \right) = u ^ { \prime } } } y _ { i } - \sum _ { \substack { i : u _ { i } \left( 1 \right) = u ^ { \prime } } } p _ { i } \right) + \left( \sum _ { \substack { i : u _ { i } \left( 0 \right) = u ^ { \prime } } } p _ { i } - \sum _ { \substack { i : u _ { i } \left( 0 \right) = u ^ { \prime } } } y _ { i } \right) \right) \cdot u ^ { \prime } \right|\tag{25}
$$

$$
\leq \sum _ { u ^ { \prime } \in \mathcal { U } } \left( \left| \sum _ { i : u _ { i } \left( 1 \right) = u ^ { \prime } } y _ { i } - \sum _ { i : u _ { i } \left( 1 \right) = u ^ { \prime } } p _ { i } \right| + \left| \sum _ { i : u _ { i } \left( 0 \right) = u ^ { \prime } } p _ { i } - \sum _ { i : u _ { i } \left( 0 \right) = u ^ { \prime } } y _ { i } \right| \right) \cdot u ^ { \prime }\tag{26}
$$

$$
\leq \sum _ { u ^ { \prime } \in \mathcal { U } } \left( \frac { \epsilon } { 2 \cdot u ^ { \prime } \cdot | \mathcal { U } | } + \frac { \epsilon } { 2 \cdot u ^ { \prime } \cdot | \mathcal { U } | } \right) \cdot u ^ { \prime }\tag{27}
$$

(28)

While we use a symmetric $\ell ^ { 1 }$ loss here, it may be interesting to also look into other measures of error. For example, for settings where under-prediction and over-prediction are valued diferently, it may be instructive to look into asymmetric error functions.

## A.2 Approximate utility level sets

Here, we generalise Proposition 5 to only require calibration on sets of approximately equal utility: For this, we divide the utility spectrum into bins of some size $\delta > 0$ and bound the resulting diference between predicted and cumulative utility by a term dependent on δ and the number of predictions d.

Proposition 10 (Predicting cumulative utility: Approximate utility).

We assume the same setting as in Proposition 5 except that we now require all utilities to be positive—otherwise, one may simply shift the values to a positive domain. We partition the interval of relevant utilities from the lowest $u _ { i } ( a )$ to the highest $u _ { i } ( a )$ with $i \in \{ 1 , . . . , d \} , a \in$ $\{ 0 , 1 \}$ into bins $B _ { 1 } , . . . , B _ { m }$ of size $\leq \delta$ . If we then replace condition (3) with the assumption that $\forall k \in \{ 1 , . . . , m \}$ ,

$$
\sum _ { i : u _ { i } ( 0 ) \in B _ { k } } y _ { i } = \sum _ { i : u _ { i } ( 0 ) \in B _ { k } } p _ { i } \qquad a n d \qquad \sum _ { i : u _ { i } ( 1 ) \in B _ { k } } y _ { i } = \sum _ { i : u _ { i } ( 1 ) \in B _ { k } } p _ { i }\tag{29}
$$

then

$$
\left| \sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] - \sum _ { i = 1 } ^ { d } \left[ p _ { i } u _ { i } ( 1 ) + ( 1 - p _ { i } ) u _ { i } ( 0 ) \right] \right| \leq \delta \cdot d .\tag{30}
$$

Proof.

The maximal mismatch occurs when for each bin $B _ { k }$ and each $a \in \{ 0 , 1 \}$ , one half of the $\{ i : u _ { i } ( a ) \in B _ { k } \}$ , we have $( y _ { i } - p _ { i } ) = 1$ and $u _ { i } ( a ) = m _ { k } + \delta / 2$ whereas for the other half, $\left( y _ { i } - p _ { i } \right) = - 1$ and $u _ { i } ( a ) = m _ { k } - \delta / 2$ , with $m _ { k }$ denoting the midpoint of $B _ { k }$ . This gives

$$
\forall k \in \{ 1 , . . . , m \} , a \in \{ 0 , 1 \} : \quad \left| \sum _ { i : u _ { i } ( a ) \in B _ { k } } ( y _ { i } - p _ { i } ) \cdot u _ { i } ( a ) \right| \leq \left| \sum _ { i : u _ { i } ( a ) \in B _ { k } } \delta / 2 \right| = b _ { k } ^ { a } \cdot \delta / 2\tag{31}
$$

where $b _ { k } ^ { a } : = | \{ 1 \leq i \leq d | u _ { i } ( a ) \in B _ { k } \}$ |. Therefore,

$$
\left| \sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] - \sum _ { i = 1 } ^ { d } \left[ p _ { i } u _ { i } ( 1 ) + ( 1 - p _ { i } ) u _ { i } ( 0 ) \right] \right|\tag{32}
$$

$$
\leq \left| \sum _ { i = 1 } ^ { d } [ y _ { i } \cdot u _ { i } ( 1 ) - p _ { i } \cdot u _ { i } ( 1 ) ] \right| + \left| \sum _ { i = 1 } ^ { d } [ ( 1 - y _ { i } ) \cdot u _ { i } ( 0 ) - ( 1 - p _ { i } ) \cdot u _ { i } ( 0 ) ] \right|\tag{33}
$$

$$
= \left| \sum _ { i = 1 } ^ { d } [ ( y _ { i } - p _ { i } ) \cdot u _ { i } ( 1 ) ] \right| + \left| \sum _ { i = 1 } ^ { d } [ ( y _ { i } - p _ { i } ) \cdot u _ { i } ( 0 ) ] \right|\tag{34}
$$

$$
\leq \sum _ { k = 1 } ^ { m } \left( \left| \sum _ { \substack { i : u _ { i } ( 1 ) \in B _ { k } } } \left( y _ { i } - p _ { i } \right) \cdot u _ { i } ( 1 ) \right| + \left| \sum _ { \substack { i : u _ { i } ( 0 ) \in B _ { k } } } \left( y _ { i } - p _ { i } \right) \cdot u _ { i } ( 0 ) \right| \right)\tag{35}
$$

$$
\leq \sum _ { k = 1 } ^ { m } \left( b _ { k } ^ { 1 } \cdot \delta / 2 + b _ { k } ^ { 0 } \cdot \delta / 2 \right)
$$

$$
= \delta \cdot d\tag{36}
$$

(37)

## A.3 Imprecise calibration

We now consider imprecise forecasts which give interval predictions $[ a , b ] \subset \mathbb { R }$ and represent them as tuples $p ^ { * } = ( \underline { { p } } , \bar { p } ) \in \mathbb { R } ^ { 2 }$ of the lower and upper probability. This allows for a <sup>¯</sup>weaker calibration criterion where the number of occurring events need not exactly match the sum of predictions, but should lie between the sum of the lower and the sum of the higher predictions.

Definition 11 (Imprecise calibration).

Imprecise predictions $p _ { 1 } ^ { * } , . . . , p _ { d } ^ { * }$ are said to be imprecisely calibrated for observations $y _ { 1 } , . . . , y _ { d } \in$ {0, 1} if they satisfy

$$
\sum _ { i = 1 } ^ { d } { \underline { { p } } } _ { i } \leq \sum _ { i = 1 } ^ { d } y _ { i } \quad a n d \quad \sum _ { i = 1 } ^ { d } { \bar { p } } _ { i } \geq \sum _ { i = 1 } ^ { d } y _ { i } .\tag{38}
$$

Note that for our definition, the vacuous forecast that always predicts $( 0 , 1 )$ is always imprecisely calibrated.<sup>22</sup> One could also apply the criterion of imprecise calibration to a set of precise predictions, by simply converting every precise prediction $p _ { i }$ into an imprecise forecast $[ p _ { i } - \epsilon , p _ { i } + \epsilon ]$ for some ϵ—this epsilon may also monotonically decrease in $d$ to account for lower variance on larger sets.

Proposition 12 (Predicting cumulative utility, imprecise version).

Let there be d imprecise predictions $p _ { i } ^ { * } \in \mathbb { R } ^ { 2 }$ for binary outcomes $y _ { i } \in \{ 0 , 1 \} , i \in \{ 1 , . . . , d \}$ with utility functions $u _ { i } : \{ 0 , 1 \} $ R and assume that the predictions are imprecisely calibrated on sets of equal $u t i l i t y \ u _ { i } ( 0 )$ and on sets of equal utility $u _ { i } ( 1 )$ (formalised in $( 4 1 )$

below).

Then I can correctly predict a range for my cumulative utility (LHS) via

$$
\sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] > \sum _ { i = 1 } ^ { d } \left[ \underline { { { p } } } _ { i } u _ { i } ( 1 ) + ( 1 - \underline { { { p } } } _ { i } ) u _ { i } ( 0 ) \right]\tag{39}
$$

and

$$
\sum _ { i = 1 } ^ { d } \left[ y _ { i } u _ { i } ( 1 ) + ( 1 - y _ { i } ) u _ { i } ( 0 ) \right] < \sum _ { i = 1 } ^ { d } \left[ \bar { p } _ { i } u _ { i } ( 1 ) + ( 1 - \bar { p } _ { i } ) u _ { i } ( 0 ) \right]\tag{40}
$$

The proof is analogous to that of Proposition 5, now with the calibration assumptions

$$
\begin{array} { r l r l } & { \displaystyle \sum _ { i : u _ { i } ( 1 ) = u ^ { \prime } } y _ { i } > \sum _ { i : u _ { i } ( 1 ) = u ^ { \prime } } \underline { { p } } _ { i } \quad } & { \mathrm { a n d } } & { \displaystyle \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } y _ { i } > \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } \underline { { p } } _ { i } , } \\ & { \displaystyle \sum _ { i : u _ { i } ( 1 ) = u ^ { \prime } } y _ { i } < \sum _ { i : u _ { i } ( 1 ) = u ^ { \prime } } \bar { p } _ { i } \quad } & { \mathrm { a n d } } & { \displaystyle \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } y _ { i } < \sum _ { i : u _ { i } ( 0 ) = u ^ { \prime } } \bar { p } _ { i } . } \end{array}\tag{41}
$$

This allows people to not only optimise their utility but to also take risk-averse or riskseeking inclinations into account—selecting policies not based on the expected exact cumulative utility but on e.g. the lowest or highest estimation of it. Here, we can see an analogy between the move from deterministic to probabilistic and the move from precise to imprecise predictions: The former allows people to take their (cardinal) preferences into account (Section 3.3), whereas the latter allows them to take their risk aversion into account. If we know that we will be calibrated, risk aversion does not make much sense. Cases where we are less sure of it can be represented by an assumption of imprecise calibration. Note that this notion of risk-aversion also captures unwillingness to bet, for decisions between the utility function of a bet and the constant zero utility function with $u ( 0 ) = u ( 1 ) = 0$

## B Conditional probabilities, generalised

We here generalise the analysis of conditional probabilities in Section 4.2. Consider a predictor ${ \mathsf { p } } : { \mathcal { X } } \times { \mathcal { A } }  [ 0 , 1 ]$ , events $A _ { 1 } , . . . , A _ { d } , B \in { \mathcal { A } }$ , and inputs $x _ { 1 } , . . . , x _ { d } \in \mathcal { X }$ . We assume $\mathsf { p } ( x _ { 1 } , B ) = \ldots = \mathsf { p } ( x _ { d } , B )$ and that p is calibrated on $\{ ( x _ { i } , A _ { i } \cap B ) : 1 \leq i \leq d \}$ and $\{ ( x _ { i } , B ) : 1 \leq i \leq d \}$ , that is,

$$
\sum _ { i = 1 } ^ { d } { \mathsf { p } } ( x _ { i } , A _ { i } \cap B ) = \sum _ { i = 1 } ^ { d } y _ { i } ^ { A _ { i } \cap B }\tag{42}
$$

and

$$
\mathsf { p } ( x _ { 1 } , B ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } y _ { i } ^ { B } > 0 .\tag{43}
$$

Then p is calibrated on $\{ ( x _ { i } , A _ { i } | B ) : y _ { i } ^ { B } = 1 \}$ (i.e. for the set of steps where B occurs) if and only if it satisfies

$$
\sum _ { i = 1 } ^ { d } \mathsf { p } ( x _ { i } , A _ { i } | B ) = \sum _ { i = 1 } ^ { d } \frac { \mathsf { p } ( x _ { i } , A _ { i } \cap B ) } { \mathsf { p } ( x _ { i } , B ) } ,\tag{44}
$$

as we derive below. In particular, a suficient condition is

$$
\mathsf { p } ( x _ { i } , A _ { i } | B ) = \frac { \mathsf { p } ( x _ { i } , A _ { i } \cap B ) } { \mathsf { p } ( x _ { i } , B ) } .\tag{45}
$$

Now consider the special case where $A _ { i } = \ldots = A _ { d } = : A$ and $x _ { i } = \ldots = x _ { d } = : x$ . Here, the familiar definition of conditional probabilities

$$
\mathsf { p } ( x , A | B ) = \frac { \mathsf { p } ( x , A \cap B ) } { \mathsf { p } ( x , B ) }\tag{46}
$$

is necessary and suficient for $p$ to be calibrated for predictions of $A | B$ based on inputs x on the set of steps where B occurs. That is, making predictions for $A | B$ rather than A allows us to be calibrated on the set where B occurs (which predictions for A would usually not be).

Now the promised derivation of the characterisation (44):

$$
\begin{array} { r l r l } & { \displaystyle \sum _ { s \geq i = 1 } ^ { n } p ( x , s , i | B ) = \sum _ { v \leq i = 1 } ^ { n } y _ { i } ^ { A _ { 1 } ( s ) } } \\ & { \displaystyle \sum _ { s \geq i = 1 } ^ { n } p ( x , s , i | B ) = \sum _ { v \leq i = 1 } ^ { n } y _ { i } ^ { A _ { 1 } ( s ) } } & & { \displaystyle ( \operatorname* { s i n } \omega _ { s } \omega _ { s } ^ { A _ { 1 } ( s ) } - \omega _ { s } ^ { A _ { 1 } ( s ) } - \gamma _ { s } ^ { B _ { 1 } ( s ) } ) \operatorname* { s i n } \ y _ { i } ^ { B } = 1 ; } \\ & { \displaystyle \sum _ { s \geq i = 1 } ^ { n } p ( x , s , i | B ) = \sum _ { s \geq i = 1 } ^ { n } y _ { i } ^ { A _ { 1 } ( s ) } } \\ & { \displaystyle \sum _ { s \geq i = 1 } ^ { n } p ( x , s , i | B ) = \sum _ { t = 1 } ^ { n } y _ { i } ^ { A _ { 1 } ( s ) } } & & { \displaystyle ( \operatorname* { s i n } \omega _ { s } \omega _ { s } ^ { A _ { 1 } ( s ) } - \gamma _ { s } ^ { B _ { 1 } ( s ) } - \gamma _ { s } ^ { B _ { 1 } ( s ) } ) } \\ & { \displaystyle \sum _ { s \geq i = 1 } ^ { n } p ( x , s , i | B ) = \sum _ { s \geq 1 } ^ { n } p ( x _ { s } , A _ { 1 } ( s ) B ) } & & { \displaystyle ( \mathrm { b y ~ ( \mathrm { d } 2 ) } ) } \\ & { \displaystyle \operatorname* { s i n } \omega _ { s } ^ { B _ { 1 } ( s ) } - \sum _ { s \geq i = 1 } ^ { n } p ( x , s , i | B ) = \sum _ { s \geq i = 1 } ^ { n } \operatorname* { s i n } \{ \mathrm { d } B \} } & & { \displaystyle ( \mathrm { b y ~ ( \mathrm { d } 3 ) } ) } \\ &  \displaystyle \sum _ { s \geq i = 1 } ^ { n } p ( x , s , i | B ) = \sum _ { s \geq i = 1 } ^ { n } p ( x _ { s } , A _  \end{array}
$$

## References

Bardenet, R´emi and Odalric-Ambrym Maillard (2015). “Concentration inequalities for sampling without replacement”. In: Bernoulli 21.3, pp. 1361–1385.

Bell, Eric Temple (1945). The development of mathematics. 2nd ed. McGraw-Hill Book Company.

Black, Emily, Manish Raghavan, and Solon Barocas (2022). “Model multiplicity: Opportunities, concerns, and solutions”. In: Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, pp. 850–863.

Breiman, Leo (2001). “Statistical modeling: The two cultures (with comments and a rejoinder by the author)”. In: Statistical science 16.3, pp. 199–231.

Buchak, Lara (2013). Risk and rationality. Oxford University Press.

Burhanpurkar, Maya et al. (2021). “Scafolding sets”. In: arXiv preprint arXiv:2111.03135.

Carnap, Rudolf (1950). Logical foundations of probability. Unicersity of Chicago Press.

Cournot, Antoine Augustin (1843). Exposition de la th´eorie des chances et des probabilit´es. L. Hachette.

Danks, David (2015). “Goal-dependence in (scientific) ontology”. In: Synthese 192, pp. 3601– 3616.

Dawid, A Philip (1985). “Calibration-based empirical probability”. In: The Annals of Statistics 13.4, pp. 1251–1274.

Dawid, Philip (2017). “On individual risk”. In: Synthese 194.9, pp. 3445–3474.

De Cooman, Gert and Jasper De Bock (2022). “Randomness is inherently imprecise”. In: International Journal of Approximate Reasoning 141, pp. 28–68.

De Finetti, Bruno (1937). “La pr´evision: ses lois logiques, ses sources subjectives”. In: Annales de l’institut Henri Poincar´e. Vol. 7. 1, pp. 1–68.

De Finetti, Bruno and Leonard J Savage (1962). “Sul modo di scegliere le probabilit\`a iniziali”. In: Biblioteca del Metron, Serie C 1, pp. 81–154.

DeGroot, Morris H and Stephen E Fienberg (1983). “The comparison and evaluation of forecasters”. In: Journal of the Royal Statistical Society: Series D (The Statistician) 32.1-2, pp. 12–22.

Derr, Rabanus, Jessie Finocchiaro, and Robert C Williamson (2025). “Three Types of Calibration with Properties and their Semantic and Formal Relationships”. In: arXiv preprint arXiv:2504.18395.

Derr, Rabanus and Robert C Williamson (2023). “Systems of precision: coherent probabilities on pre-Dynkin systems and coherent previsions on linear subspaces”. In: Entropy 25.9, p. 1283.

Douglas, Heather (2000). “Inductive risk and values in science”. In: Philosophy of Science 67.4, pp. 559–579.

Dwork, Cynthia (2022). “Fairness, randomness, and the crystal ball”. In: Munich AI Lectures. url: https://www.youtube.com/watch?v=n4XftI9G0fA (visited on 04/08/2023).

Eliot, George (1871). Middlemarch. William Blackwood and Sons.

Feduzi, Alberto, Jochen Runde, and Carlo Zappia (2012). “De Finetti on the insurance of risks and uncertainties”. In: The British journal for the philosophy of science.

Feng, Yiding and Wei Tang (2025). “Measuring Informativeness Gap of (Mis) Calibrated Predictors”. In: arXiv preprint arXiv:2507.12094.

Flyvbjerg, Bent, Carsten Glenting, and Arne Rønnest (2004). “Procedures for dealing with optimism bias in transport planning”. In: London: The British Department for Transport, Guidance Document.

Foster, Dean P and Sergiu Hart (2021). “Forecast hedging and calibration”. In: Journal of Political Economy 129.12, pp. 3447–3490.

Foster, Dean P and Rakesh V Vohra (1998). “Asymptotic calibration”. In: Biometrika 85.2, pp. 379–390.

Freedman, David (1997). “Some issues in the foundation of statistics”. In: Topics in the Foundation of Statistics, pp. 19–39.

Fr¨ohlich, Christian and Robert C Williamson (2024). “Insights from insurance for fair machine learning: Responsibility, performativity and aggregates”. In: Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency.

Garber, Daniel and Sandy Zabell (1979). “On the emergence of probability”. In: Archive for History of Exact Sciences, pp. 33–53.

Gigerenzer, Gerd et al. (2005). ““A 30% chance of rain tomorrow”: How does the public understand probabilistic weather forecasts?” In: Risk Analysis: An International Journal 25.3, pp. 623–629.

Glymour, Clark (2001). “Instrumental probability”. In: The Monist 84.2, pp. 284–300.

Gneiting, Tilmann and Adrian E Raftery (2007). “Strictly proper scoring rules, prediction, and estimation”. In: Journal of the American statistical Association 102.477, pp. 359– 378.

Goodfellow, Ian, Yoshua Bengio, and Aaron Courville (2016). Deep Learning. http://www. deeplearningbook.org. MIT Press.

Goodman, Nelson (1972). “Seven Strictures on Similarity”. In: Problems and Projects. Indianapolis: Bobbs-Merrill, pp. 437–446.

Gopalan, Parikshit, Michael P Kim, and Omer Reingold (2023). “Characterizing notions of omniprediction via multicalibration”. In: arXiv preprint arXiv:2302.06726.

Guo, Chuan et al. (2017). “On calibration of modern neural networks”. In: International Conference on Machine Learning. PMLR, pp. 1321–1330.

Hacking, Ian (1975). The emergence of probability. Cambridge University Press.

— (1990). The taming of chance. 17. Cambridge University Press.

H´ajek, Alan (1996). ““Mises redux”—redux: Fifteen arguments against finite frequentism”. In: Erkenntnis 45, pp. 209–227.

— (2007). “The reference class problem is your problem too”. In: Synthese 156, pp. 563– 585.

(2023). “Interpretations of Probability”. In: The Stanford Encyclopedia of Philosophy. Ed. by Edward N. Zalta and Uri Nodelman. Winter 2023. Metaphysics Research Lab, Stanford University.

H´ebert-Johnson, Ursula et al. (2018). “Multicalibration: Calibration for the (computationallyidentifiable) masses”. In: International Conference on Machine Learning. PMLR, pp. 1939– 1948.

Hedden, Brian (2013). “Incoherence without exploitability”. In: Noˆus 47.3, pp. 482–495.

H¨oltgen, Benedikt and Robert C Williamson (2023). “On the richness of calibration”. In: Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, pp. 1124–1138.

— (2025). “Formalising causal inference as prediction on a target population”. In: arXiv preprint arXiv:2407.17385.

— (2026). “The costs of pretending that there are data-generating probabilitiy distributions in the social world”. In: Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency.

Hume, David (1777). An enquiry concerning human understanding.

Keynes, John Maynard (1921). A treatise on probability. Macmillan & Co.

Kleinberg, Bobby et al. (2023). “U-calibration: Forecasting for an unknown agent”. In: The Thirty Sixth Annual Conference on Learning Theory. PMLR, pp. 5143–5145.

Knight, Frank Hyneman (1921). Risk, uncertainty and profit. Vol. 31. Houghton Miflin.

Kuhn, Thomas S. (2012). “Postscript”. In: The Structure of Scientific Revolutions. 4th edition. Chicago: University of Chicago Press.

Le Guin, Ursula K (1969). The left hand of darkness. Ace Books.

Lewis, David (1980). “A subjectivist’s guide to objective chance”. In: Philosophical Papers (1986) 2, pp. 83–132.

— (1994). “Humean supervenience debugged”. In: Mind 103.412, pp. 473–490.

Li, Xinran and Xiao-Li Meng (2021). “A multi-resolution theory for approximating infinitep-zero-n: Transitional inference, individualized predictions, and a world without biasvariance tradeof”. In: Journal of the American Statistical Association.

Mellers, Barbara et al. (2015). “Identifying and cultivating superforecasters as a method of improving probabilistic predictions”. In: Perspectives on Psychological Science 10.3, pp. 267–281.

Meng, Xiao-Li (2018). “Statistical paradises and paradoxes in big data (i) law of large populations, big data paradox, and the 2016 us presidential election”. In: The Annals of Applied Statistics 12.2, pp. 685–726.

Merleau-Ponty, Maurice (1955). Les aventures de la dialectique. Gallimard.

Moss, Emanuel (2022). “The objective function: Science and society in the age of machine intelligence”. In: arXiv preprint arXiv:2209.10418.

Murphy, Allan H and Robert L Winkler (1977). “Reliability of subjective probability forecasts of precipitation and temperature”. In: Journal of the Royal Statistical Society Series C: Applied Statistics 26.1, pp. 41–47.

Noarov, Georgy and Aaron Roth (2024). Calibration for decision making: A principled approach to trustworthy ml, 2024. url: https://www.let-all.com/blog/2024/03/13/ calibration- for- decision- making- a- principled- approach- to- trustworthyml/ (visited on 09/29/2025).

Peirce, Charles Sanders (1878). “The doctrine of chances”. In: 12, pp. 604–615.

Perdomo, Juan Carlos and Benjamin Recht (2025). “In Defense of Defensive Forecasting”. In: arXiv preprint arXiv:2506.11848.

Poisson, Sim´eon-Denis (1837). Recherches sur la probabilit´e des jugements en mati\`ere criminelle et en mati\`ere civile: pr´ec´ed´ees des r\`egles g´en´erales du calcul des probabilit´es. Bachelier.

Polanyi, Michael (1958). Personal knowledge. University of Chicago Press.

Popper, Karl R (1959). “The propensity interpretation of probability”. In: The British journal for the philosophy of science 10.37, pp. 25–42.

Potochnik, Angela (2017). “Idealization and the Aims of Science”. In: Idealization and the Aims of Science. University of Chicago Press.

Ramsey, Frank P (1928/1931). “Further considerations”. In: The Foundations of Mathematics and other Logical Essays. Ed. by R.B. Braithwaite. Routledge. Chap. VIII, pp. 199– 211.

— (1926/1931). “Truth and probability”. In: The Foundations of Mathematics and other Logical Essays. Ed. by R.B. Braithwaite. Routledge. Chap. VII, pp. 156–198.

Reichenbach, Hans (1949). The theory of probability. University of California Press.

Salmon, Wesley C (1966). The foundations of scientific inference. University of Pittsburgh Press.

Sanders, Frederick (1963). “On subjective probability forecasting”. In: Journal of Applied Meteorology and Climatology 2.2, pp. 191–201.

Savage, Leonard J (1972). The foundations of statistics. 2nd edition. John Wiley and Sons.

Schervish, Mark J (1985). “Discussion: Calibration-based empirical probability”. In: The Annals of Statistics 13.4, pp. 1274–1282.

Seidenfeld, Teddy, Mark J Schervish, and Joseph B Kadane (2012). “Forecasting with imprecise probabilities”. In: International Journal of Approximate Reasoning 53.8, pp. 1248– 1261.

Shafer, Glenn and Vladimir Vovk (2019). Game-theoretic foundations for probability and finance. John Wiley & Sons.

Streater, RF (2000). “Classical and quantum probability”. In: Journal of Mathematical Physics 41.6, pp. 3556–3603.

Strevens, Michael (2006). “Probability and chance”. In: Encyclopedia of Philosophy, second edition. Macmillan Reference USA, Detroit.

Thoma, Johanna (2019). “Risk aversion and the long run”. In: Ethics 129.2, pp. 230–253.

Thorp, Edward O (1998). “The invention of the first wearable computer”. In: Digest of Papers. Second International Symposium on Wearable Computers. IEEE, pp. 4–8.

van Fraassen, Bas C (1983). “Calibration: A frequency justification for personal probability”. In: Physics, Philosophy and Psychoanalysis: Essays in Honour of Adolf Gr¨unbaum, pp. 295–319.

Vovk, Vladimir and Glenn Shafer (2025). “A conversation with A. Philip Dawid”. In: Statistical Science 40.1, pp. 148–166.

Vovk, Vladimir, Akimichi Takemura, and Glenn Shafer (2005). “Defensive forecasting”. In: AISTATS. Vol. 2005, pp. 365–372.

Wakker, Peter (1994). “Separating marginal utility and probabilistic risk aversion”. In: Theory and Decision 36.1, pp. 1–44.

Walley, Peter (1991). Statistical reasoning with imprecise probabilities. Chapman-Hall.

Zhao, Shengjia et al. (2021). “Calibrating predictions to decisions: A novel approach to multi-class calibration”. In: Advances in Neural Information Processing Systems 34, pp. 22313–22324.

Williams, Donald (1947). The ground of induction. Harvard University Press.